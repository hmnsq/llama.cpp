#هدايت once
#ضمن <stdint.h>
#ضمن <stdbool.h>

#إذا defined(__VXE__) || defined(__VXE2__)
#ضمن <vecintrin.h>
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل _MSC_VER
#وصف_ڪريو NOINLINE __declspec(noinline)
#وإلا
#وصف_ڪريو NOINLINE __attribute__((__noinline__))
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل __cplusplus
خارجي "C" {
#جيڪڏهن_پورو

bool llamafile_sgemm(ثابت بنية ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ثابت فراغ *, int64_t, ثابت فراغ *, int64_t, فراغ *, int64_t,
                     صحيح, صحيح, صحيح);

#جيڪڏهن_وصف_ٿيل __cplusplus
}
#جيڪڏهن_پورو
