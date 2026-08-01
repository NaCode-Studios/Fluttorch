// A complete implementation of the ABI, with no ExecuTorch behind it.
//
// It is not a mock of the binding: it is a real shared library, compiled by the
// test, that the real dart:ffi layer really calls. What it proves is everything
// the boundary itself can get wrong and no Dart-side fake can catch: struct
// layout and field offsets, string arrays, pointer arithmetic over tensor
// arrays, and whether the bytes the callee writes are the bytes Dart reads back.
//
// What it cannot prove is that ExecuTorch runs a model, which is the other half
// and needs the toolchain and a device.

#include "../../src/fluttorch_executorch.h"

#include <stdlib.h>
#include <string.h>

static const char* kBackends[] = {"xnnpack", "coreml"};
static const int32_t kBackendCount = 2;

// float32, int8 and bool, as bits over the manifest's dtype ordering.
static const int64_t kDtypes = (1 << 0) | (1 << 4) | (1 << 9);

static char g_error[256] = {0};

struct ft_model {
  const char* backend;
  int32_t deterministic;
  // Counts runs, so a test can prove one call crossed the boundary once.
  int32_t runs;
};

static ft_status_t fail(ft_status_t status, const char* message) {
  strncpy(g_error, message, sizeof(g_error) - 1);
  return status;
}

// Returns this library's own copy of the name, or NULL. Callers hand over
// strings that live only for the duration of the call, so anything kept past it
// has to point at storage the library owns. Keeping the caller's pointer reads
// freed memory the moment the arena behind it goes, which is a use-after-free
// that returns plausible text rather than crashing.
static const char* canonical_backend(const char* name) {
  for (int32_t i = 0; i < kBackendCount; i++) {
    if (strcmp(kBackends[i], name) == 0) return kBackends[i];
  }
  return NULL;
}

ft_status_t ft_backends(const char** out_names, int32_t capacity,
                        int32_t* out_count) {
  if (out_names == NULL || out_count == NULL) {
    return fail(FT_ERROR_INVALID_ARGUMENT, "out_names and out_count are required");
  }
  *out_count = kBackendCount;
  for (int32_t i = 0; i < kBackendCount && i < capacity; i++) {
    out_names[i] = kBackends[i];
  }
  return FT_OK;
}

ft_status_t ft_capabilities(const char* backend, ft_capabilities_t* out) {
  if (out == NULL) return fail(FT_ERROR_INVALID_ARGUMENT, "out is required");
  const char* name = canonical_backend(backend == NULL ? kBackends[0] : backend);
  if (name == NULL) {
    return fail(FT_ERROR_BACKEND_UNAVAILABLE, "not compiled into this build");
  }
  out->backend = name;
  // Only the first backend taps, so a test can tell the two apart by capability
  // rather than by name.
  out->supports_taps = strcmp(name, kBackends[0]) == 0 ? 1 : 0;
  out->supports_determinism = 1;
  out->max_tensor_bytes = 1048576;
  out->dtypes = kDtypes;
  return FT_OK;
}

ft_status_t ft_load(const uint8_t* artifact, int64_t length, const char* backend,
                    int32_t deterministic, ft_model_t** out_model) {
  if (artifact == NULL || out_model == NULL) {
    return fail(FT_ERROR_INVALID_ARGUMENT, "artifact and out_model are required");
  }
  if (length < 4) {
    return fail(FT_ERROR_ARTIFACT_UNREADABLE, "shorter than any header");
  }
  const char* name = canonical_backend(backend == NULL ? kBackends[0] : backend);
  if (name == NULL) {
    return fail(FT_ERROR_BACKEND_UNAVAILABLE, "not compiled into this build");
  }
  ft_model_t* model = (ft_model_t*)calloc(1, sizeof(ft_model_t));
  if (model == NULL) return fail(FT_ERROR_OUT_OF_MEMORY, "calloc");
  model->backend = name;
  model->deterministic = deterministic;
  *out_model = model;
  return FT_OK;
}

const char* ft_model_backend(ft_model_t* model) {
  return model == NULL ? NULL : model->backend;
}

// Writes each output as the byte-wise sum of every input byte, which is a
// function of what actually crossed the boundary rather than a constant: a
// binding that marshalled the wrong bytes produces the wrong answer instead of
// the right one by luck.
static ft_status_t run_into(ft_model_t* model, const ft_tensor_t* inputs,
                            int32_t input_count, ft_tensor_t* outputs,
                            int32_t output_count) {
  if (model == NULL) return fail(FT_ERROR_INVALID_ARGUMENT, "model is required");
  uint8_t sum = 0;
  for (int32_t i = 0; i < input_count; i++) {
    if (inputs[i].data == NULL) {
      return fail(FT_ERROR_INVALID_ARGUMENT, "an input carries no data");
    }
    if (inputs[i].rank < 1) {
      return fail(FT_ERROR_SHAPE_MISMATCH, "an input has rank zero");
    }
    for (int64_t b = 0; b < inputs[i].byte_length; b++) {
      sum = (uint8_t)(sum + ((const uint8_t*)inputs[i].data)[b]);
    }
  }
  for (int32_t i = 0; i < output_count; i++) {
    if (outputs[i].data == NULL) {
      return fail(FT_ERROR_INVALID_ARGUMENT, "an output carries no buffer");
    }
    memset(outputs[i].data, sum, (size_t)outputs[i].byte_length);
  }
  model->runs++;
  return FT_OK;
}

ft_status_t ft_run(ft_model_t* model, const ft_tensor_t* inputs,
                   int32_t input_count, ft_tensor_t* outputs,
                   int32_t output_count) {
  return run_into(model, inputs, input_count, outputs, output_count);
}

ft_status_t ft_run_with_taps(ft_model_t* model, const ft_tensor_t* inputs,
                             int32_t input_count, ft_tensor_t* outputs,
                             int32_t output_count, const char** layer_names,
                             int32_t layer_count, ft_tensor_t* out_activations,
                             int32_t* out_captured) {
  if (out_captured == NULL) {
    return fail(FT_ERROR_INVALID_ARGUMENT, "out_captured is required");
  }
  if (model == NULL || strcmp(model->backend, kBackends[0]) != 0) {
    return fail(FT_ERROR_CAPABILITY_UNAVAILABLE, "this backend has no taps");
  }
  ft_status_t status = run_into(model, inputs, input_count, outputs, output_count);
  if (status != FT_OK) return status;

  // Every layer but the last is captured. A graph that does not carry a
  // requested tap leaves it out, and the caller is told how many were filled
  // rather than being handed a zeroed tensor that reads as agreement.
  int32_t captured = layer_count > 0 ? layer_count - 1 : 0;
  static int64_t shape[1] = {2};
  static uint8_t bytes[8] = {0};
  for (int32_t i = 0; i < captured; i++) {
    bytes[0] = (uint8_t)(i + 1);
    out_activations[i].data = bytes;
    out_activations[i].byte_length = 8;
    out_activations[i].shape = shape;
    out_activations[i].rank = 1;
    out_activations[i].dtype = 0; // float32
  }
  *out_captured = captured;
  (void)layer_names;
  return FT_OK;
}

void ft_dispose(ft_model_t* model) { free(model); }

const char* ft_last_error(void) { return g_error; }
