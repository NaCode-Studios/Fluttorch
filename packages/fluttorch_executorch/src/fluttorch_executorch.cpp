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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/event_tracer.h>
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
// "portable" is unconditional and first among equals: it names the absence of a
// delegate rather than one of them, and the runtime's own kernels are linked
// into every build of this library. An artifact lowered without a partitioner
// runs there, which is what makes layer attribution possible at all.
const char* const kBackends[] = {
    "portable",
    "xnnpack",
#if defined(FLUTTORCH_WITH_COREML)
    "coreml",
#endif
#if defined(FLUTTORCH_WITH_MPS)
    "mps",
#endif
#if defined(FLUTTORCH_WITH_METAL)
    "metal",
#endif
#if defined(FLUTTORCH_WITH_VULKAN)
    "vulkan",
#endif
#if defined(FLUTTORCH_WITH_MLX)
    "mlx",
#endif
#if defined(FLUTTORCH_WITH_QNN)
    "qnn",
#endif
};
constexpr int32_t kBackendCount =
    static_cast<int32_t>(sizeof(kBackends) / sizeof(kBackends[0]));

// Named rather than positional. The default used to be whatever stood first in
// the table, which was fine while the table had two entries and became a way to
// change what an unpinned load runs by editing a list.
constexpr const char* kDefaultBackend = "xnnpack";

// Backends whose reduction order is fixed, so two runs of one input agree bit
// for bit. Portable qualifies for the plainest reason: it is the runtime's own
// kernels with nothing parallel underneath. XNNPACK qualifies on a
// single-threaded pool. Every accelerator below schedules work it does not
// promise to schedule the same way twice.
bool is_deterministic(const char* name) {
  return std::strcmp(name, "portable") == 0 || std::strcmp(name, "xnnpack") == 0;
}

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

#if defined(FLUTTORCH_WITH_TAPS)
// Keeps the intermediates a caller asked for and drops the rest.
//
// ExecuTorch hands a tracer one value at a time with no name attached: which op
// produced it is whatever debug handle the executor set before the instruction
// ran. That is the same handle the export recorded per tap, which is what makes
// the two ends of a tap meet without either side inventing a mapping.
//
// Only the wanted handles are copied. A tracer that kept everything would
// allocate the whole graph to answer for three layers, and on a real model the
// intermediates dwarf the model.
class TapCollector : public executorch::runtime::EventTracer {
 public:
  std::set<int64_t> wanted;
  std::map<int64_t, std::vector<uint8_t>> captured;

  void reset(const int64_t* handles, int32_t count) {
    wanted.clear();
    captured.clear();
    for (int32_t i = 0; i < count; i++) wanted.insert(handles[i]);
    set_event_tracer_debug_level(
        executorch::runtime::EventTracerDebugLogLevel::kIntermediateOutputs);
  }

  // Off between tapped runs, because the level is what the executor checks
  // before it offers a value: an ordinary ft_run should not pay for taps.
  void stop() {
    set_event_tracer_debug_level(
        executorch::runtime::EventTracerDebugLogLevel::kNoLogging);
    wanted.clear();
  }

  executorch::runtime::Result<bool> log_evalue(
      const EValue& evalue,
      executorch::runtime::LoggedEValueType type) override {
    if (type != executorch::runtime::LoggedEValueType::kIntermediateOutput) {
      return true;
    }
    const int64_t handle = static_cast<int64_t>(current_debug_handle());
    if (wanted.find(handle) == wanted.end()) return true;
    if (!evalue.isTensor()) return true;

    const auto tensor = evalue.toTensor();
    const uint8_t* bytes = static_cast<const uint8_t*>(tensor.const_data_ptr());
    // Assigned rather than appended: an out-variant op writes its output last,
    // so the value standing when the instruction ends is the one the layer
    // produced.
    captured[handle].assign(bytes, bytes + tensor.nbytes());
    return true;
  }

  // Everything below is the rest of the interface, which this tracer does not
  // use. Profiling is not what taps are for, and a delegate reports nothing
  // from inside itself in any case.
  void create_event_block(const char*) override {}
  executorch::runtime::EventTracerEntry start_profiling(
      const char*, executorch::runtime::ChainID,
      executorch::runtime::DebugHandle) override {
    return {};
  }
  void end_profiling(executorch::runtime::EventTracerEntry) override {}
  executorch::runtime::EventTracerEntry start_profiling_delegate(
      const char*, executorch::runtime::DelegateDebugIntId) override {
    return {};
  }
  void end_profiling_delegate(executorch::runtime::EventTracerEntry, const void*,
                              size_t) override {}
  void log_profiling_delegate(const char*,
                              executorch::runtime::DelegateDebugIntId,
                              et_timestamp_t, et_timestamp_t, const void*,
                              size_t) override {}
  void track_allocation(executorch::runtime::AllocatorID, size_t) override {}
  executorch::runtime::AllocatorID track_allocator(const char*) override {
    return 0;
  }
  void set_delegation_intermediate_output_filter(
      executorch::runtime::EventTracerFilterBase*) override {}
  executorch::runtime::Result<bool> log_intermediate_output_delegate(
      const char*, executorch::runtime::DelegateDebugIntId,
      const executorch::aten::Tensor&) override {
    return true;
  }
  executorch::runtime::Result<bool> log_intermediate_output_delegate(
      const char*, executorch::runtime::DelegateDebugIntId,
      const executorch::runtime::ArrayRef<executorch::aten::Tensor>) override {
    return true;
  }
  executorch::runtime::Result<bool> log_intermediate_output_delegate(
      const char*, executorch::runtime::DelegateDebugIntId, const int&) override {
    return true;
  }
  executorch::runtime::Result<bool> log_intermediate_output_delegate(
      const char*, executorch::runtime::DelegateDebugIntId, const bool&) override {
    return true;
  }
  executorch::runtime::Result<bool> log_intermediate_output_delegate(
      const char*, executorch::runtime::DelegateDebugIntId,
      const double&) override {
    return true;
  }
};
#endif

} // namespace

struct ft_model {
  // The artifact is copied because the Module borrows it for its whole life and
  // the caller's buffer is borrowed only for the load call.
  std::vector<uint8_t> artifact;
  std::unique_ptr<BufferDataLoader> loader;
  std::unique_ptr<Module> module;
  const char* backend;
  bool deterministic;
#if defined(FLUTTORCH_WITH_TAPS)
  // Owned by the Module, borrowed here. A tracer can only be given at
  // construction, so it outlives every run rather than being attached to one.
  TapCollector* taps = nullptr;
#endif
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
  const char* name =
      canonical_backend(backend == nullptr ? kDefaultBackend : backend);
  if (name == nullptr) {
    return fail(FT_ERROR_BACKEND_UNAVAILABLE,
                "not compiled into this build of the binding");
  }
  out->backend = name;
  // Core ML compiles the model for the Neural Engine or the GPU when it decides
  // to, and neither promises a fixed reduction order. XNNPACK on a single-thread
  // pool does, which is the difference between a tolerance measuring the model
  // and one absorbing the hardware's mood.

  // A property of the build, not of the backend: the tracer is compiled in or it
  // is not. Whether a given artifact answers is a separate question the run
  // reports per tap, because a delegated partition is opaque even to a build
  // that can read every op it executes itself.
#if defined(FLUTTORCH_WITH_TAPS)
  out->supports_taps = 1;
#else
  out->supports_taps = 0;
#endif
  // A single-threaded pool fixes the order of every parallel reduction, which is
  // what makes two runs of one input agree bit for bit.
  out->supports_determinism = is_deterministic(name) ? 1 : 0;
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
  const char* name =
      canonical_backend(backend == nullptr ? kDefaultBackend : backend);
  if (name == nullptr) {
    return fail(FT_ERROR_BACKEND_UNAVAILABLE,
                "not compiled into this build of the binding");
  }

  executorch::runtime::runtime_init();

  auto model = std::make_unique<ft_model>();
  model->artifact.assign(artifact, artifact + length);
  model->loader = std::make_unique<BufferDataLoader>(model->artifact.data(),
                                                     model->artifact.size());
#if defined(FLUTTORCH_WITH_TAPS)
  auto collector = std::make_unique<TapCollector>();
  model->taps = collector.get();
  model->module = std::make_unique<Module>(
      std::unique_ptr<executorch::runtime::DataLoader>(model->loader.get()),
      /*memory_allocator=*/nullptr, /*temp_allocator=*/nullptr,
      std::move(collector));
#else
  model->module = std::make_unique<Module>(
      std::unique_ptr<executorch::runtime::DataLoader>(model->loader.get()));
#endif
  model->backend = name;
  model->deterministic = deterministic != 0;

  // Loading here rather than lazily on the first run, so a bad artifact fails
  // where the caller can still choose another one.
  if (deterministic != 0 && !is_deterministic(name)) {
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
                             int32_t output_count, const int64_t* layer_handles,
                             int32_t layer_count, ft_tensor_t* out_activations,
                             int32_t* out_captured) {
#if !defined(FLUTTORCH_WITH_TAPS)
  (void)model;
  (void)inputs;
  (void)input_count;
  (void)outputs;
  (void)output_count;
  (void)layer_handles;
  (void)layer_count;
  (void)out_activations;
  if (out_captured != nullptr) *out_captured = 0;
  // Refused rather than answered with nothing captured, which the gate would
  // read as a run where every layer agreed. ft_capabilities reports the absence
  // so a caller never reaches this by surprise.
  return fail(FT_ERROR_CAPABILITY_UNAVAILABLE,
              "this build compiles no event tracer, so no intermediate is readable");
#else
  if (model == nullptr) return fail(FT_ERROR_INVALID_ARGUMENT, "model is required");
  if (layer_count > 0 && (layer_handles == nullptr || out_activations == nullptr)) {
    return fail(FT_ERROR_INVALID_ARGUMENT,
                "asking for taps needs both the handles and the buffers to fill");
  }
  if (out_captured == nullptr) {
    return fail(FT_ERROR_INVALID_ARGUMENT,
                "out_captured is required: a caller that cannot tell a captured "
                "tap from an absent one would read a gap as agreement");
  }

  // Capture is switched on only around this run, so an ordinary ft_run through
  // the same model does not pay for a tracer nobody asked to read.
  model->taps->reset(layer_handles, layer_count);
  const ft_status_t ran = ft_run(model, inputs, input_count, outputs, output_count);
  model->taps->stop();
  if (ran != FT_OK) return ran;

  int32_t captured = 0;
  for (int32_t i = 0; i < layer_count; i++) {
    const auto found = model->taps->captured.find(layer_handles[i]);
    if (found == model->taps->captured.end()) {
      // Left as the caller supplied it. Absent is a reading, not a failure: a
      // delegated partition answers for nothing inside itself.
      continue;
    }
    const std::vector<uint8_t>& bytes = found->second;
    if (static_cast<int64_t>(bytes.size()) != out_activations[i].byte_length) {
      return fail(FT_ERROR_SHAPE_MISMATCH,
                  "tap " + std::to_string(i) + " is " + std::to_string(bytes.size()) +
                      " bytes and the buffer supplied is " +
                      std::to_string(out_activations[i].byte_length));
    }
    std::memcpy(out_activations[i].data, bytes.data(), bytes.size());
    captured |= 1 << i;
  }
  *out_captured = captured;
  return FT_OK;
#endif
}

void ft_dispose(ft_model_t* model) { delete model; }

const char* ft_last_error(void) { return g_error.c_str(); }
