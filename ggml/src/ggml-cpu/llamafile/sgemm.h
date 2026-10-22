#ہدایت once
#ضمن <stdint.h>
#ضمن <stdbool.h>

#إذا defined(__VXE__) || defined(__VXE2__)
#ضمن <vecintrin.h>
#جے_انت

#جے_تعریف_شدہ _MSC_VER
#تعریف_کرو NOINLINE __declspec(noinline)
#وإلا
#تعریف_کرو NOINLINE __attribute__((__noinline__))
#جے_انت

#جے_تعریف_شدہ __cplusplus
خارجي "C" {
#جے_انت

bool llamafile_sgemm(ثابت بنية ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ثابت فراغ *, int64_t, ثابت فراغ *, int64_t, فراغ *, int64_t,
                     صحيح, صحيح, صحيح);

#جے_تعریف_شدہ __cplusplus
}
#جے_انت
