// The ABI, implemented against ExecuTorch.
//
// Nothing here decides anything: it translates. Which backend an artifact runs
// on was decided when it was lowered, whether a tensor fits was decided by the
// manifest, and what the numbers should be was decided by the export. What this
// file does is carry those across a C boundary without losing the reason a call
// failed, because a status code that says "execution failed" and nothing else
// sends someone to read the wrong file.

#include "fluttorch_executorch.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::aten::ScalarType;
using executorch::extension::BufferDataLoader;
using executorch::extension::Module;
using executorch::extension::make_tensor_ptr;
using executorch::runtime::EValue;
using executorch::runtime::Error;

namespace {

// What this build can actually run, which is not what ExecuTorch supports: a
// backend appears here only once it is both linked and registered, so the list a
// caller reads is the list that will load rather than a menu of intentions.
const char* const kBackends[] = {
    "xnnpack",
#if defined(FLUTTORCH_WITH_COREML)
    "coreml",
#endif
};
constexpr int32_t kBackendCount =
    static_cast<int32_t>(sizeof(kBackends) / sizeof(kBackends[0]));

thread_local std::string g_error;

ft_status_t fail(ft_status_t status, const std::string& message) {
  g_error = message;
  return status;
}

const char* canonical_backend(const char* name) {
  for (int32_t i = 0; i < kBackendCount; i++) {
    if (std::strcmp(kBackends[i], name) == 0) return kBackends[i];
  }
  return nullptr;
}

// The manifest's dtype ordering, which the Dart side reads by position. Adding
// one in the middle would silently reinterpret every mask ever written, so the
// order is the contract and this table is the place it is written down twice on
// purpose.
ScalarType scalar_type_of(int32_t dtype) {
  switch (dtype) {
    case 0: return ScalarType::Float;   // float32
    case 1: return ScalarType::Double;  // float64
    case 2: return ScalarType::Half;    // float16
    case 3: return ScalarType::BFloat16;
    case 4: return ScalarType::Char;    // int8
    case 5: return ScalarType::Short;   // int16
    case 6: return ScalarType::Int;     // int32
    case 7: return ScalarType::Long;    // int64
    case 8: return ScalarType::Byte;    // uint8
    case 9: return ScalarType::Bool;
    default: return ScalarType::Undefined;
  }
}

constexpr int64_t kSupportedDtypes =
    (1 << 0) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) |
    (1 << 9);

std::string describe(Error error) {
  return "ExecuTorch error " + std::to_string(static_cast<int>(error));
}

} // namespace

struct ft_model {
  // The artifact is copied because the Module borrows it for its whole life and
  // the caller's buffer is borrowed only for the load call.
  std::vector<uint8_t> artifact;
  std::unique_ptr<BufferDataLoader> loader;
  std::unique_ptr<Module> module;
  const char* backend;
  bool deterministic;
};

ft_status_t ft_backends(const char** out_names, int32_t capacity,
                        int32_t* out_count) {
  if (out_names == nullptr || out_count == nullptr) {
    return fail(FT_ERROR_INVALID_ARGUMENT, "out_names and out_count are required");
  }
  *out_count = kBackendCount;
  for (int32_t i = 0; i < kBackendCount && i < capacity; i++) {
    out_names[i] = kBackends[i];
  }
  return FT_OK;
}

ft_status_t ft_capabilities(const char* backend, ft_capabilities_t* out) {
  if (out == nullptr) return fail(FT_ERROR_INVALID_ARGUMENT, "out is required");
  const char* name = canonical_backend(backend == nullptr ? kBackends[0] : backend);
  if (name == nullptr) {
    return fail(FT_ERROR_BACKEND_UNAVAILABLE,
                "not compiled into this build of the binding");
  }
  out->backend = name;
  // Core ML compiles the model for the Neural Engine or the GPU when it decides
  // to, and neither promises a fixed reduction order. XNNPACK on a single-thread
  // pool does, which is the difference between a tolerance measuring the model
  // and one absorbing the hardware's mood.
  const bool is_xnnpack = std::strcmp(name, "xnnpack") == 0;
  // Reported as absent rather than assumed present. Reading intermediates needs
  // an event tracer and an artifact carrying debug handles, and this build links
  // neither, so claiming the capability would make the gate attribute a drift it
  // cannot see.
  out->supports_taps = 0;
  // A single-threaded pool fixes the order of every parallel reduction, which is
  // what makes two runs of one input agree bit for bit.
  out->supports_determinism = is_xnnpack ? 1 : 0;
  out->max_tensor_bytes = 0; // unknown, which is not the same as unlimited
  out->dtypes = kSupportedDtypes;
  return FT_OK;
}

ft_status_t ft_load(const uint8_t* artifact, int64_t length, const char* backend,
                    int32_t deterministic, ft_model_t** out_model) {
  if (artifact == nullptr || out_model == nullptr) {
    return fail(FT_ERROR_INVALID_ARGUMENT, "artifact and out_model are required");
  }
  if (length <= 0) {
    return fail(FT_ERROR_ARTIFACT_UNREADABLE, "the artifact is empty");
  }
  const char* name = canonical_backend(backend == nullptr ? kBackends[0] : backend);
  if (name == nullptr) {
    return fail(FT_ERROR_BACKEND_UNAVAILABLE,
                "not compiled into this build of the binding");
  }

  executorch::runtime::runtime_init();

  auto model = std::make_unique<ft_model>();
  model->artifact.assign(artifact, artifact + length);
  model->loader = std::make_unique<BufferDataLoader>(model->artifact.data(),
                                                     model->artifact.size());
  model->module = std::make_unique<Module>(
      std::unique_ptr<executorch::runtime::DataLoader>(model->loader.get()));
  model->backend = name;
  model->deterministic = deterministic != 0;

  // Loading here rather than lazily on the first run, so a bad artifact fails
  // where the caller can still choose another one.
  if (deterministic != 0 && std::strcmp(name, "xnnpack") != 0) {
    return fail(FT_ERROR_CAPABILITY_UNAVAILABLE,
                std::string("backend ") + name +
                    " cannot promise a repeatable reduction order");
  }

  const Error error = model->module->load_method("forward");
  if (error != Error::Ok) {
    return fail(FT_ERROR_ARTIFACT_UNREADABLE,
                "the artifact has no loadable forward method: " + describe(error));
  }

  // The loader is owned by the Module now; releasing our handle avoids a double
  // free without giving up the ability to keep the bytes alive alongside it.
  model->loader.release();
  *out_model = model.release();
  return FT_OK;
}

const char* ft_model_backend(ft_model_t* model) {
  return model == nullptr ? nullptr : model->backend;
}

namespace {

ft_status_t to_evalues(const ft_tensor_t* tensors, int32_t count,
                       std::vector<executorch::extension::TensorPtr>& owned,
                       std::vector<EValue>& out) {
  for (int32_t i = 0; i < count; i++) {
    const ft_tensor_t& t = tensors[i];
    if (t.data == nullptr) {
      return fail(FT_ERROR_INVALID_ARGUMENT, "a tensor carries no data");
    }
    const ScalarType type = scalar_type_of(t.dtype);
    if (type == ScalarType::Undefined) {
      return fail(FT_ERROR_INVALID_ARGUMENT,
                  "dtype index " + std::to_string(t.dtype) + " is not one this "
                  "build can carry");
    }
    std::vector<executorch::aten::SizesType> sizes;
    sizes.reserve(static_cast<size_t>(t.rank));
    for (int32_t d = 0; d < t.rank; d++) {
      sizes.push_back(static_cast<executorch::aten::SizesType>(t.shape[d]));
    }
    owned.push_back(make_tensor_ptr(std::move(sizes), t.data, type));
    out.emplace_back(*owned.back());
  }
  return FT_OK;
}

} // namespace

ft_status_t ft_run(ft_model_t* model, const ft_tensor_t* inputs,
                   int32_t input_count, ft_tensor_t* outputs,
                   int32_t output_count) {
  if (model == nullptr) return fail(FT_ERROR_INVALID_ARGUMENT, "model is required");

  std::vector<executorch::extension::TensorPtr> owned;
  std::vector<EValue> values;
  owned.reserve(static_cast<size_t>(input_count));
  values.reserve(static_cast<size_t>(input_count));
  const ft_status_t marshalled = to_evalues(inputs, input_count, owned, values);
  if (marshalled != FT_OK) return marshalled;

  const auto result = model->module->execute("forward", values);
  if (!result.ok()) {
    return fail(FT_ERROR_EXECUTION_FAILED,
                "forward failed: " + describe(result.error()));
  }

  const auto& produced = result.get();
  if (static_cast<int32_t>(produced.size()) != output_count) {
    return fail(FT_ERROR_SHAPE_MISMATCH,
                "the model returned " + std::to_string(produced.size()) +
                    " outputs, the caller supplied " + std::to_string(output_count));
  }

  for (int32_t i = 0; i < output_count; i++) {
    if (!produced[i].isTensor()) {
      return fail(FT_ERROR_SHAPE_MISMATCH,
                  "output " + std::to_string(i) + " is not a tensor");
    }
    const auto tensor = produced[i].toTensor();
    const int64_t bytes = static_cast<int64_t>(tensor.nbytes());
    if (bytes != outputs[i].byte_length) {
      return fail(FT_ERROR_SHAPE_MISMATCH,
                  "output " + std::to_string(i) + " is " + std::to_string(bytes) +
                      " bytes and the buffer supplied is " +
                      std::to_string(outputs[i].byte_length));
    }
    std::memcpy(outputs[i].data, tensor.const_data_ptr(),
                static_cast<size_t>(bytes));
  }
  return FT_OK;
}

ft_status_t ft_run_with_taps(ft_model_t* model, const ft_tensor_t* inputs,
                             int32_t input_count, ft_tensor_t* outputs,
                             int32_t output_count, const char** layer_names,
                             int32_t layer_count, ft_tensor_t* out_activations,
                             int32_t* out_captured) {
  (void)inputs;
  (void)input_count;
  (void)outputs;
  (void)output_count;
  (void)layer_names;
  (void)layer_count;
  (void)out_activations;
  if (out_captured != nullptr) *out_captured = 0;
  // Refused rather than answered with nothing captured, which the gate would
  // read as a run where every layer agreed. Intermediates need an event tracer
  // and an artifact carrying debug handles; ft_capabilities reports the absence
  // so a caller never reaches this by surprise.
  return fail(FT_ERROR_CAPABILITY_UNAVAILABLE,
              "this build links no event tracer, so no intermediate is readable");
}

void ft_dispose(ft_model_t* model) { delete model; }

const char* ft_last_error(void) { return g_error.c_str(); }
