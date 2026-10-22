#pragma once
#include <stdint.h>
#include <stdbool.h>

#만약 defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#아니면
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     정수, 정수, 정수);

#ifdef __cplusplus
}
#endif
