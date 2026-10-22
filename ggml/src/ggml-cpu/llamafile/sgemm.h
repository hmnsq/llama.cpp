#ہدایت once
#شامل <stdint.h>
#شامل <stdbool.h>

#اگر defined(__VXE__) || defined(__VXE2__)
#شامل <vecintrin.h>
#اگر_ختم

#اگر_تعریف_شدہ _MSC_VER
#تعریف_کرو NOINLINE __declspec(noinline)
#ورنہ
#تعریف_کرو NOINLINE __attribute__((__noinline__))
#اگر_ختم

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

bool llamafile_sgemm(ثابت ساخت ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ثابت خالی *, int64_t, ثابت خالی *, int64_t, خالی *, int64_t,
                     صحیح, صحیح, صحیح);

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم
