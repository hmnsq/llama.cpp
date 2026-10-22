#لارښوونه once
#ضمن <stdint.h>
#ضمن <stdbool.h>

#إذا defined(__VXE__) || defined(__VXE2__)
#ضمن <vecintrin.h>
#د_که_پای

#که_تعریف_شوی _MSC_VER
#تعریف_کړه NOINLINE __declspec(noinline)
#وإلا
#تعریف_کړه NOINLINE __attribute__((__noinline__))
#د_که_پای

#که_تعریف_شوی __cplusplus
خارجي "C" {
#د_که_پای

bool llamafile_sgemm(ثابت بنية ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ثابت فراغ *, int64_t, ثابت فراغ *, int64_t, فراغ *, int64_t,
                     صحيح, صحيح, صحيح);

#که_تعریف_شوی __cplusplus
}
#د_که_پای
