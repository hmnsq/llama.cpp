#pragma once
#include <stdint.h>
#include <stdbool.h>

#ถ้า defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#มิฉะนั้น
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(const sa_nracanaa ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     จำนวนเต็ม, จำนวนเต็ม, จำนวนเต็ม);

#ifdef __cplusplus
}
#endif
