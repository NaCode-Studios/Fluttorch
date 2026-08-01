// The ABI, implemented against LiteRT.
//
// The third implementation of one seam, and the same header the other two
// answer to. What a second and a third runtime buy is the claim that the seam
// is a seam: the manifest, the goldens and the gate above it do not change, and
// what changes is the engine.
//
// A "backend" here is a LiteRT accelerator rather than an ExecuTorch delegate or
// an ONNX Runtime provider, which is why the manifest keeps runtime and backend
// as separate fields.

#include "../../fluttorch_executorch/src/fluttorch_executorch.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <litert/c/litert_common.h>
#include <litert/c/litert_compiled_model.h>
#include <litert/c/litert_environment.h>
#include <litert/c/litert_model.h>
#include <litert/c/litert_options.h>
#include <litert/c/litert_tensor_buffer.h>
#include <litert/c/litert_tensor_buffer_types.h>

namespace {

// CPU only. LiteRT reaches an NPU through a vendor runtime that is not in the
// wheel and is not on this machine, and a backend this build cannot select is
// one it does not name, the same rule the other two implementations follow.
const char* const kBackends[] = {"cpu"};
constexpr int32_t kBackendCount = 1;
constexpr const char* kDefaultBackend = "cpu";

thread_local std::string g_error;

ft_status_t fail(ft_status_t status, const std::string& message) {
  g_error = message;
  return status;
}

// LiteRT returns a code and nothing else, so the message has to be built from
// what the caller was doing. A status of "execution failed" with no context is
// exactly what this project refuses to hand back.
ft_status_t from_litert(LiteRtStatus status, ft_status_t code,
                        const std::string& doing) {
  if (status == kLiteRtStatusOk) return FT_OK;
  return fail(code, doing + ": LiteRT status " + std::to_string(status));
}

const char* canonical_backend(const char* name) {
  for (int32_t i = 0; i < kBackendCount; i++) {
    if (std::strcmp(kBackends[i], name) == 0) return kBackends[i];
  }
  return nullptr;
}

// The manifest's dtype ordering, mapped to LiteRT's element types. The third
// copy of this table, and deliberately explicit in all three: the order is the
// contract, and a mask that drifts silently reinterprets every model.
LiteRtElementType element_type_of(int32_t dtype) {
  switch (dtype) {
    case 0: return kLiteRtElementTypeFloat32;
    case 1: return kLiteRtElementTypeFloat64;
    case 2: return kLiteRtElementTypeFloat16;
    case 3: return kLiteRtElementTypeBFloat16;
    case 4: return kLiteRtElementTypeInt8;
    case 5: return kLiteRtElementTypeInt16;
    case 6: return kLiteRtElementTypeInt32;
    case 7: return kLiteRtElementTypeInt64;
    case 8: return kLiteRtElementTypeUInt8;
    case 9: return kLiteRtElementTypeBool;
    default: return kLiteRtElementTypeNone;
  }
}

constexpr int64_t kSupportedDtypes =
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) |
    (1 << 7) | (1 << 8) | (1 << 9);

LiteRtEnvironment environment() {
  // One per process. The environment owns the accelerator registry, and one per
  // model is how an app pays for that registry once per model.
  static LiteRtEnvironment env = [] {
    LiteRtEnvironment e = nullptr;
    LiteRtCreateEnvironment(0, nullptr, &e);
    return e;
  }();
  return env;
}

} // namespace

struct ft_model {
  // Copied because LiteRT reads the flatbuffer for the model's whole life and
  // the caller's buffer is borrowed only for the load call.
  std::vector<uint8_t> artifact;
  LiteRtModel model = nullptr;
  LiteRtCompiledModel compiled = nullptr;
  const char* backend = nullptr;

  ~ft_model() {
    if (compiled != nullptr) LiteRtDestroyCompiledModel(compiled);
    if (model != nullptr) LiteRtDestroyModel(model);
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
  // No taps. A LiteRT graph carries no handle for an intermediate, and reading
  // one means marking it an output before conversion, which produces a
  // different artifact from the one the goldens describe.
  out->supports_taps = 0;
  // CPU fixes its reduction order, and it is the only accelerator this build
  // can select.
  out->supports_determinism = 1;
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
  (void)deterministic; // CPU is the only accelerator here, and it is repeatable.

  auto model = std::make_unique<ft_model>();
  model->artifact.assign(artifact, artifact + length);
  model->backend = name;

  ft_status_t status = from_litert(
      LiteRtCreateModelFromBuffer(environment(), model->artifact.data(),
                                  model->artifact.size(), &model->model),
      FT_ERROR_ARTIFACT_UNREADABLE, "the artifact is not a loadable LiteRT model");
  if (status != FT_OK) return status;

  LiteRtOptions options = nullptr;
  status = from_litert(LiteRtCreateOptions(&options), FT_ERROR_OUT_OF_MEMORY,
                       "creating compilation options");
  if (status != FT_OK) return status;
  std::unique_ptr<std::remove_pointer_t<LiteRtOptions>,
                  void (*)(LiteRtOptions)>
      owned(options, LiteRtDestroyOptions);

  // Named rather than left at the default. Compilation is refused outright
  // without a hardware set, with kLiteRtStatusErrorInvalidArgument and no
  // indication that the options are what it is complaining about.
  status = from_litert(
      LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu),
      FT_ERROR_BACKEND_UNAVAILABLE, "selecting the CPU accelerator");
  if (status != FT_OK) return status;

  status = from_litert(
      LiteRtCreateCompiledModel(environment(), model->model, options,
                                &model->compiled),
      FT_ERROR_ARTIFACT_UNREADABLE, "compiling the model");
  if (status != FT_OK) return status;

  *out_model = model.release();
  return FT_OK;
}

const char* ft_model_backend(ft_model_t* model) {
  return model == nullptr ? nullptr : model->backend;
}

namespace {

// Buffers handed to LiteRT, and the aligned storage behind them.
//
// This is where this implementation cannot keep a promise the ABI makes
// elsewhere. LiteRT refuses host memory that is not aligned to
// LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, which is 64, and the buffers arriving
// from Dart carry no such guarantee: what comes back is
// kLiteRtStatusErrorRuntimeFailure with nothing saying alignment is the
// subject. So every tensor is copied into aligned storage on the way in and
// copied back out on the way out.
//
// Two copies per run rather than none, and worth stating plainly rather than
// hiding: the caller-owned buffers the other two runtimes write into directly
// buy nothing here. Fixing it means aligning the allocation on the Dart side,
// which is a change to the seam rather than to this file.
struct Buffers {
  std::vector<LiteRtTensorBuffer> handles;
  std::vector<void*> storage;

  ~Buffers() {
    for (LiteRtTensorBuffer b : handles) {
      if (b != nullptr) LiteRtDestroyTensorBuffer(b);
    }
    for (void* p : storage) {
      if (p != nullptr) std::free(p);
    }
  }

  // Aligned storage of `bytes`, owned until this goes out of scope.
  void* aligned(size_t bytes) {
    void* p = nullptr;
    const size_t rounded =
        ((bytes + LITERT_HOST_MEMORY_BUFFER_ALIGNMENT - 1) /
         LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) *
        LITERT_HOST_MEMORY_BUFFER_ALIGNMENT;
    if (posix_memalign(&p, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, rounded) != 0) {
      return nullptr;
    }
    storage.push_back(p);
    return p;
  }
};

ft_status_t wrap(Buffers& owner, const ft_tensor_t& t, const char* role,
                 int32_t index, LiteRtTensorBuffer* out) {
  if (t.data == nullptr) {
    return fail(FT_ERROR_INVALID_ARGUMENT,
                std::string(role) + " " + std::to_string(index) +
                    " carries no data");
  }
  const LiteRtElementType type = element_type_of(t.dtype);
  if (type == kLiteRtElementTypeNone) {
    return fail(FT_ERROR_INVALID_ARGUMENT,
                "dtype index " + std::to_string(t.dtype) +
                    " is not one this build can carry");
  }

  LiteRtRankedTensorType described = {};
  described.element_type = type;
  described.layout.rank = static_cast<uint32_t>(t.rank);
  for (int32_t d = 0; d < t.rank && d < LITERT_TENSOR_MAX_RANK; d++) {
    described.layout.dimensions[d] = static_cast<int32_t>(t.shape[d]);
  }

  void* aligned = owner.aligned(static_cast<size_t>(t.byte_length));
  if (aligned == nullptr) {
    return fail(FT_ERROR_OUT_OF_MEMORY,
                std::string("no aligned storage for ") + role + " " +
                    std::to_string(index));
  }
  std::memcpy(aligned, t.data, static_cast<size_t>(t.byte_length));

  // No deallocator: the storage is owned by Buffers and released with it.
  return from_litert(
      LiteRtCreateTensorBufferFromHostMemory(
          &described, aligned, static_cast<size_t>(t.byte_length), nullptr, out),
      FT_ERROR_INVALID_ARGUMENT,
      std::string("wrapping ") + role + " " + std::to_string(index));
}

} // namespace

ft_status_t ft_run(ft_model_t* model, const ft_tensor_t* inputs,
                   int32_t input_count, ft_tensor_t* outputs,
                   int32_t output_count) {
  if (model == nullptr) return fail(FT_ERROR_INVALID_ARGUMENT, "model is required");

  Buffers in;
  in.handles.assign(static_cast<size_t>(input_count), nullptr);
  for (int32_t i = 0; i < input_count; i++) {
    const ft_status_t status =
        wrap(in, inputs[i], "input", i, &in.handles[static_cast<size_t>(i)]);
    if (status != FT_OK) return status;
  }

  Buffers out;
  out.handles.assign(static_cast<size_t>(output_count), nullptr);
  for (int32_t i = 0; i < output_count; i++) {
    const ft_status_t status =
        wrap(out, outputs[i], "output", i, &out.handles[static_cast<size_t>(i)]);
    if (status != FT_OK) return status;
  }

  // Signature zero: an export from this toolchain carries one, and a model with
  // several is a thing the manifest would have to describe before a caller
  // could choose between them.
  const ft_status_t ran = from_litert(
      LiteRtRunCompiledModel(model->compiled, 0, in.handles.size(),
                             in.handles.data(), out.handles.size(),
                             out.handles.data()),
      FT_ERROR_EXECUTION_FAILED, "the model failed to run");
  if (ran != FT_OK) return ran;

  // Back out of the aligned storage and into what the caller owns.
  for (int32_t i = 0; i < output_count; i++) {
    std::memcpy(outputs[i].data, out.storage[static_cast<size_t>(i)],
                static_cast<size_t>(outputs[i].byte_length));
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
  return fail(FT_ERROR_CAPABILITY_UNAVAILABLE,
              "a LiteRT graph carries no handle for an intermediate; reading one "
              "means marking it an output before conversion, which produces a "
              "different artifact from the one the goldens describe");
}

void ft_dispose(ft_model_t* model) { delete model; }

const char* ft_last_error(void) { return g_error.c_str(); }
