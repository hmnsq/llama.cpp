#pragma once
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stdbool.h>

#যদি defined(__VXE__) || defined(__VXE2__)
#অন্তর্ভাব <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#অন্যথা
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

bool llamafile_sgemm(ধ্রুবক কাঠামো ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ধ্রুবক শূন্য *, int64_t, ধ্রুবক শূন্য *, int64_t, শূন্য *, int64_t,
                     পূর্ণসংখ্যা, পূর্ণসংখ্যা, পূর্ণসংখ্যা);

#ifdef __cplusplus
}
#endif
