#pragma once
#ضمن <stdint.h>
#ضمن <stdbool.h>

#אם defined(__VXE__) || defined(__VXE2__)
#ضمن <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#אחרת
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
خارجي "C" {
#endif

bool llamafile_sgemm(ثابت بنية ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ثابت فراغ *, int64_t, ثابت فراغ *, int64_t, فراغ *, int64_t,
                     صحيح, صحيح, صحيح);

#ifdef __cplusplus
}
#endif
