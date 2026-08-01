// The ABI, implemented against ONNX Runtime.
//
// The same header ExecuTorch's implementation answers to, and deliberately so.
// A second runtime is a second implementation of one seam, not a second seam:
// duplicating the contract is how two implementations stop agreeing about what
// a status code means, and the Dart side above it does not change at all.
//
// What differs is what the words mean underneath. A "backend" here is an ONNX
// Runtime execution provider rather than an ExecuTorch delegate, which is why
// the manifest records the runtime and the backend separately.

#include "../../fluttorch_executorch/src/fluttorch_executorch.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_c_api.h>

namespace {

const OrtApi* ort() {
  static const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  return api;
}

// Providers this build can select, in the order a caller meets them. CPU is
// unconditional because ONNX Runtime always has it; Core ML is compiled in on
// Apple platforms and reports itself absent elsewhere, the same way the
// ExecuTorch side reports a delegate it was not linked with.
const char* const kBackends[] = {
    "cpu",
#if defined(FLUTTORCH_ONNX_WITH_COREML)
    "coreml",
#endif
};
constexpr int32_t kBackendCount =
    static_cast<int32_t>(sizeof(kBackends) / sizeof(kBackends[0]));

constexpr const char* kDefaultBackend = "cpu";

thread_local std::string g_error;

ft_status_t fail(ft_status_t status, const std::string& message) {
  g_error = message;
  return status;
}

// Releases a status whose failure the caller has already decided not to act on.
//
// Not the same as ignoring it. These calls configure or interrogate a session
// that was already created, and a failure means the default stands, which is
// what the code below is written to survive. What is not survivable is leaking
// the status object, and that is what this is for.
void discard(OrtStatus* status) {
  if (status != nullptr) ort()->ReleaseStatus(status);
}

ft_status_t from_ort(OrtStatus* status, ft_status_t code,
                     const std::string& doing) {
  if (status == nullptr) return FT_OK;
  const std::string message = doing + ": " + ort()->GetErrorMessage(status);
  ort()->ReleaseStatus(status);
  return fail(code, message);
}

const char* canonical_backend(const char* name) {
  for (int32_t i = 0; i < kBackendCount; i++) {
    if (std::strcmp(kBackends[i], name) == 0) return kBackends[i];
  }
  return nullptr;
}

// The manifest's dtype ordering, which the Dart side reads by position. The same
// table as the ExecuTorch side and for the same reason: the order is the
// contract, and writing it down twice on purpose is cheaper than a mask that
// silently reinterprets every model ever exported.
ONNXTensorElementDataType element_type_of(int32_t dtype) {
  switch (dtype) {
    case 0: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case 1: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case 2: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case 3: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case 4: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case 5: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case 6: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case 7: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case 8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case 9: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

constexpr int64_t kSupportedDtypes =
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) |
    (1 << 8) | (1 << 9);

OrtEnv* environment() {
  // One per process. ONNX Runtime's env owns the thread pools and the logger,
  // and creating one per model is how an app ends up with a pool per model.
  static OrtEnv* env = [] {
    OrtEnv* e = nullptr;
    discard(ort()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "fluttorch", &e));
    return e;
  }();
  return env;
}

} // namespace

struct ft_model {
  // Copied because the session reads the bytes for its whole life and the
  // caller's buffer is borrowed only for the load call.
  std::vector<uint8_t> artifact;
  OrtSession* session = nullptr;
  OrtMemoryInfo* memory = nullptr;
  const char* backend = nullptr;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  ~ft_model() {
    if (session != nullptr) ort()->ReleaseSession(session);
    if (memory != nullptr) ort()->ReleaseMemoryInfo(memory);
  }
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
  const char* name = canonical_backend(backend == nullptr ? kDefaultBackend : backend);
  if (name == nullptr) {
    return fail(FT_ERROR_BACKEND_UNAVAILABLE,
                "not compiled into this build of the binding");
  }
  out->backend = name;
  // No taps. ONNX Runtime can be asked for intermediate values by naming them
  // as outputs, which is a different thing from reading them back mid-run and
  // needs the graph edited before it is loaded. Reported absent rather than
  // approximated, because a gate that received the wrong tensor would attribute
  // a drift to the wrong layer.
  out->supports_taps = 0;
  // CPU fixes its reduction order; Core ML schedules across the Neural Engine
  // and the GPU and undertakes nothing about how.
  out->supports_determinism = std::strcmp(name, "cpu") == 0 ? 1 : 0;
  out->max_tensor_bytes = 0;
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
  const char* name = canonical_backend(backend == nullptr ? kDefaultBackend : backend);
  if (name == nullptr) {
    return fail(FT_ERROR_BACKEND_UNAVAILABLE,
                "not compiled into this build of the binding");
  }
  if (deterministic != 0 && std::strcmp(name, "cpu") != 0) {
    return fail(FT_ERROR_CAPABILITY_UNAVAILABLE,
                std::string("provider ") + name +
                    " cannot promise a repeatable reduction order");
  }

  auto model = std::make_unique<ft_model>();
  model->artifact.assign(artifact, artifact + length);
  model->backend = name;

  OrtSessionOptions* options = nullptr;
  ft_status_t status = from_ort(ort()->CreateSessionOptions(&options),
                                FT_ERROR_OUT_OF_MEMORY, "creating session options");
  if (status != FT_OK) return status;
  std::unique_ptr<OrtSessionOptions, void (*)(OrtSessionOptions*)> owned(
      options, [](OrtSessionOptions* o) { ort()->ReleaseSessionOptions(o); });

  if (deterministic != 0) {
    // One thread each way. A parallel reduction is where two runs of one input
    // stop agreeing bit for bit, which is the whole of what this flag buys.
    discard(ort()->SetIntraOpNumThreads(options, 1));
    discard(ort()->SetInterOpNumThreads(options, 1));
  }

  status = from_ort(
      ort()->CreateSessionFromArray(environment(), model->artifact.data(),
                                    model->artifact.size(), options,
                                    &model->session),
      FT_ERROR_ARTIFACT_UNREADABLE, "the artifact is not a loadable ONNX graph");
  if (status != FT_OK) return status;

  status = from_ort(
      ort()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &model->memory),
      FT_ERROR_OUT_OF_MEMORY, "describing host memory");
  if (status != FT_OK) return status;

  // The names the graph declares, read once. Run takes them by string on every
  // call, and asking the session each time would allocate per inference.
  OrtAllocator* allocator = nullptr;
  status = from_ort(ort()->GetAllocatorWithDefaultOptions(&allocator),
                    FT_ERROR_OUT_OF_MEMORY, "taking the default allocator");
  if (status != FT_OK) return status;

  size_t count = 0;
  discard(ort()->SessionGetInputCount(model->session, &count));
  for (size_t i = 0; i < count; i++) {
    char* n = nullptr;
    if (ort()->SessionGetInputName(model->session, i, allocator, &n) == nullptr) {
      model->input_names.emplace_back(n);
      allocator->Free(allocator, n);
    }
  }
  discard(ort()->SessionGetOutputCount(model->session, &count));
  for (size_t i = 0; i < count; i++) {
    char* n = nullptr;
    if (ort()->SessionGetOutputName(model->session, i, allocator, &n) == nullptr) {
      model->output_names.emplace_back(n);
      allocator->Free(allocator, n);
    }
  }

  *out_model = model.release();
  return FT_OK;
}

const char* ft_model_backend(ft_model_t* model) {
  return model == nullptr ? nullptr : model->backend;
}

ft_status_t ft_run(ft_model_t* model, const ft_tensor_t* inputs,
                   int32_t input_count, ft_tensor_t* outputs,
                   int32_t output_count) {
  if (model == nullptr) return fail(FT_ERROR_INVALID_ARGUMENT, "model is required");
  if (static_cast<size_t>(input_count) != model->input_names.size()) {
    return fail(FT_ERROR_SHAPE_MISMATCH,
                "the graph takes " + std::to_string(model->input_names.size()) +
                    " inputs and " + std::to_string(input_count) + " were supplied");
  }
  if (static_cast<size_t>(output_count) != model->output_names.size()) {
    return fail(FT_ERROR_SHAPE_MISMATCH,
                "the graph returns " + std::to_string(model->output_names.size()) +
                    " outputs and " + std::to_string(output_count) +
                    " buffers were supplied");
  }

  std::vector<OrtValue*> values(static_cast<size_t>(input_count), nullptr);
  // Released whatever happens: an early return on a shape mismatch would
  // otherwise leak every tensor built before it.
  struct Release {
    std::vector<OrtValue*>& v;
    ~Release() {
      for (OrtValue* value : v) {
        if (value != nullptr) ort()->ReleaseValue(value);
      }
    }
  } release{values};

  for (int32_t i = 0; i < input_count; i++) {
    const ft_tensor_t& t = inputs[i];
    if (t.data == nullptr) {
      return fail(FT_ERROR_INVALID_ARGUMENT, "a tensor carries no data");
    }
    const ONNXTensorElementDataType type = element_type_of(t.dtype);
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return fail(FT_ERROR_INVALID_ARGUMENT,
                  "dtype index " + std::to_string(t.dtype) +
                      " is not one this build can carry");
    }
    const ft_status_t status = from_ort(
        ort()->CreateTensorWithDataAsOrtValue(
            model->memory, t.data,
            static_cast<size_t>(t.byte_length), t.shape,
            static_cast<size_t>(t.rank), type, &values[static_cast<size_t>(i)]),
        FT_ERROR_INVALID_ARGUMENT, "wrapping input " + std::to_string(i));
    if (status != FT_OK) return status;
  }

  std::vector<const char*> in_names;
  std::vector<const char*> out_names;
  for (const std::string& n : model->input_names) in_names.push_back(n.c_str());
  for (const std::string& n : model->output_names) out_names.push_back(n.c_str());

  std::vector<OrtValue*> produced(static_cast<size_t>(output_count), nullptr);
  Release release_out{produced};

  const ft_status_t ran = from_ort(
      ort()->Run(model->session, nullptr, in_names.data(), values.data(),
                 values.size(), out_names.data(), out_names.size(),
                 produced.data()),
      FT_ERROR_EXECUTION_FAILED, "the graph failed to run");
  if (ran != FT_OK) return ran;

  for (int32_t i = 0; i < output_count; i++) {
    OrtValue* value = produced[static_cast<size_t>(i)];
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (ort()->GetTensorTypeAndShape(value, &info) != nullptr) {
      return fail(FT_ERROR_EXECUTION_FAILED,
                  "output " + std::to_string(i) + " is not a tensor");
    }
    size_t elements = 0;
    discard(ort()->GetTensorShapeElementCount(info, &elements));
    ort()->ReleaseTensorTypeAndShapeInfo(info);

    void* data = nullptr;
    discard(ort()->GetTensorMutableData(value, &data));
    // Bytes rather than elements, because the caller sized its buffer from the
    // manifest and the manifest is what decides the width.
    const int64_t bytes = outputs[i].byte_length;
    if (data == nullptr || bytes <= 0) {
      return fail(FT_ERROR_SHAPE_MISMATCH,
                  "output " + std::to_string(i) + " has no buffer to write into");
    }
    std::memcpy(outputs[i].data, data, static_cast<size_t>(bytes));
  }
  return FT_OK;
}

ft_status_t ft_run_with_taps(ft_model_t* model, const ft_tensor_t* inputs,
                             int32_t input_count, ft_tensor_t* outputs,
                             int32_t output_count, const int64_t* layer_handles,
                             int32_t layer_count, ft_tensor_t* out_activations,
                             int32_t* out_captured) {
  (void)model;
  (void)inputs;
  (void)input_count;
  (void)outputs;
  (void)output_count;
  (void)layer_handles;
  (void)layer_count;
  (void)out_activations;
  if (out_captured != nullptr) *out_captured = 0;
  // Refused rather than answered with nothing. ONNX Runtime reads an
  // intermediate by having it named as a graph output, which means editing the
  // graph before it is loaded rather than observing a run, and an export that
  // did that would be a different artifact from the one the goldens describe.
  return fail(FT_ERROR_CAPABILITY_UNAVAILABLE,
              "ONNX Runtime reads intermediates by naming them as outputs, which "
              "is a property of the graph rather than of the run");
}

void ft_dispose(ft_model_t* model) { delete model; }

const char* ft_last_error(void) { return g_error.c_str(); }
