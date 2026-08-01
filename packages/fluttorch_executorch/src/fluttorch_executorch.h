// The C ABI between Fluttorch and ExecuTorch.
//
// ExecuTorch's public API is C++, and dart:ffi binds a C ABI, so something has
// to sit between them. Making that seam an explicit header rather than letting
// it emerge from the implementation is what allows the Dart half of the binding
// to be written and tested before the native half exists, and it is the same
// reason FluttorchRuntime exists one layer up.
//
// Four things are declared here that no published Dart binding to ExecuTorch
// exposes, and they are the whole reason this binding is being written rather
// than depended on:
//
//   1. Backend selection at load time, so a parity matrix can pin one.
//   2. Deterministic execution, so a tolerance does not have to absorb
//      run-to-run noise.
//   3. Activation taps, so a drift can be attributed to the layer that caused it.
//   4. Caller-supplied output buffers, so repeated inference stops allocating.
//
// Every call returns an ft_status_t and never throws: an exception crossing the
// FFI boundary terminates the process, and a model that fails to load is an
// ordinary Tuesday.

#ifndef FLUTTORCH_EXECUTORCH_H_
#define FLUTTORCH_EXECUTORCH_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  FT_OK = 0,
  FT_ERROR_INVALID_ARGUMENT = 1,
  FT_ERROR_ARTIFACT_UNREADABLE = 2,
  // The named backend is not compiled into this build or not available on this
  // device. Distinct from a load failure: the caller can choose another one.
  FT_ERROR_BACKEND_UNAVAILABLE = 3,
  // The build can run the model but not the way it was asked to, which is what
  // requesting determinism or taps on a backend that has neither produces.
  FT_ERROR_CAPABILITY_UNAVAILABLE = 4,
  FT_ERROR_SHAPE_MISMATCH = 5,
  FT_ERROR_EXECUTION_FAILED = 6,
  FT_ERROR_OUT_OF_MEMORY = 7,
} ft_status_t;

typedef struct ft_model ft_model_t;

// What a build can do, filled in by ft_capabilities. Reported rather than
// assumed, because a taps-capable build on a device whose delegate does not
// support them can still only compare final outputs.
typedef struct {
  const char* backend;          // e.g. "xnnpack", "coreml"
  int32_t supports_taps;        // 0 or 1
  int32_t supports_determinism; // 0 or 1
  int64_t max_tensor_bytes;     // 0 when the limit is unknown, never "unlimited"
  // Element types this backend accepts, as a bitmask over the manifest's dtype
  // ordering. Reported rather than assumed for the same reason as the rest: a
  // backend that cannot carry a model's types should fail at load with a list,
  // not at inference with a coercion.
  int64_t dtypes;
} ft_capabilities_t;

// One tensor crossing the boundary. The bytes are borrowed for the duration of
// the call and never freed by the callee, which is what lets Dart pass a buffer
// it already owns and get its outputs written into it.
typedef struct {
  void* data;
  int64_t byte_length;
  const int64_t* shape;
  int32_t rank;
  int32_t dtype; // index into the manifest's dtype ordering
} ft_tensor_t;

// Backends this build was compiled with, whether or not the device can run
// them. Writes at most `capacity` entries and always reports how many exist.
ft_status_t ft_backends(const char** out_names, int32_t capacity,
                        int32_t* out_count);

// What the named backend can do on this device. A null backend asks about the
// one the build prefers.
ft_status_t ft_capabilities(const char* backend, ft_capabilities_t* out);

// Loads a serialised model.
//
// `backend` pins the delegate rather than accepting whatever the artifact was
// lowered for, and null means the preferred available one. `deterministic`
// asks for repeatable execution and fails with FT_ERROR_CAPABILITY_UNAVAILABLE
// rather than quietly running non-deterministically, because a tolerance chosen
// against a promise that was silently dropped is a tolerance measuring noise.
ft_status_t ft_load(const uint8_t* artifact, int64_t length, const char* backend,
                    int32_t deterministic, ft_model_t** out_model);

// The backend actually in use, which is not necessarily the one requested: a
// runtime that fell back has to be able to say so, or the report names the
// wrong hardware.
const char* ft_model_backend(ft_model_t* model);

// Runs inference into buffers the caller owns. Outputs are written in place and
// nothing is allocated across the boundary.
ft_status_t ft_run(ft_model_t* model, const ft_tensor_t* inputs,
                   int32_t input_count, ft_tensor_t* outputs,
                   int32_t output_count);

// Runs inference and captures the named intermediates.
//
// `layer_names` selects the taps; a name the graph does not carry is left
// absent rather than zero-filled, and `out_captured` says which were filled, so
// the caller can tell a layer that agreed from a layer nobody looked at.
ft_status_t ft_run_with_taps(ft_model_t* model, const ft_tensor_t* inputs,
                             int32_t input_count, ft_tensor_t* outputs,
                             int32_t output_count, const char** layer_names,
                             int32_t layer_count, ft_tensor_t* out_activations,
                             int32_t* out_captured);

// Releases the model. Calling it twice is not an error; using the model
// afterwards is.
void ft_dispose(ft_model_t* model);

// The last failure in human-readable form, valid until the next call on this
// thread. A status code says what went wrong, this says which tensor.
const char* ft_last_error(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FLUTTORCH_EXECUTORCH_H_
