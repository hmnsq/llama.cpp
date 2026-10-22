#pragma once
#include <stdint.h>
#include <stdbool.h>

#如果 defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#否则
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     整数, 整数, 整数);

#ifdef __cplusplus
}
#endif
