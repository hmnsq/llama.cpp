#رهنمود once
#ضمن <stdint.h>
#ضمن <stdbool.h>

#إذا defined(__VXE__) || defined(__VXE2__)
#ضمن <vecintrin.h>
#پایان_اگر

#اگر_تعریف_شده _MSC_VER
#تعریف_کن NOINLINE __declspec(noinline)
#وإلا
#تعریف_کن NOINLINE __attribute__((__noinline__))
#پایان_اگر

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

bool llamafile_sgemm(ثابت بنية ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ثابت فراغ *, int64_t, ثابت فراغ *, int64_t, فراغ *, int64_t,
                     صحيح, صحيح, صحيح);

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر
