#pragma once
#شامل <stdint.h>
#شامل <stdbool.h>

#اگر defined(__VXE__) || defined(__VXE2__)
#شامل <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#وگرنه
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
بیرونی "C" {
#endif

bool llamafile_sgemm(ثابت ساختار ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ثابت تهی *, int64_t, ثابت تهی *, int64_t, تهی *, int64_t,
                     صحیح, صحیح, صحیح);

#ifdef __cplusplus
}
#endif
