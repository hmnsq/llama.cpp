#pragma once
#include <stdint.h>
#include <stdbool.h>

#ᬬᬾᬦ᭄ defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#ᬮᬶᬬ
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(const sa_nracanaa ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     puur_naa_nka, puur_naa_nka, puur_naa_nka);

#ifdef __cplusplus
}
#endif
