#pragma once
#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <stdbool.h>

#যদি defined(__VXE__) || defined(__VXE2__)
#অন্তর্ভাৱ <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#অন্যথা
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
বাহ্য "C" {
#endif

bool llamafile_sgemm(ধ্রুৱ সংরচনা ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ধ্রুৱ শূন্য *, int64_t, ধ্রুৱ শূন্য *, int64_t, শূন্য *, int64_t,
                     পূর্ণাঙ্ক, পূর্ণাঙ্ক, পূর্ণাঙ্ক);

#ifdef __cplusplus
}
#endif
