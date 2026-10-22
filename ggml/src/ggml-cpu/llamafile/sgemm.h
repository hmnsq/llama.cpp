#pragma once
#include <stdint.h>
#include <stdbool.h>

#གལ་ཏེ defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#དེ་མིན
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(const sa_nracanaa ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     ཧྲིལ, ཧྲིལ, ཧྲིལ);

#ifdef __cplusplus
}
#endif
