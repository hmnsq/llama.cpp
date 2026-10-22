#pragma once
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>

#ಯದಿ defined(__VXE__) || defined(__VXE2__)
#ಅನ್ತರ್ಭಾವ <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#ಅನ್ಯಥಾ
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

bool llamafile_sgemm(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ಧ್ರುವ ಶೂನ್ಯ *, int64_t, ಧ್ರುವ ಶೂನ್ಯ *, int64_t, ಶೂನ್ಯ *, int64_t,
                     ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ);

#ifdef __cplusplus
}
#endif
