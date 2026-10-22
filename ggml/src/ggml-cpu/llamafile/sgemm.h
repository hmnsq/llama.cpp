#pragma once
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stdbool.h>

#ଯଦି defined(__VXE__) || defined(__VXE2__)
#ଅନ୍ତର୍ଭାଵ <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#ଅନ୍ୟଥା
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

bool llamafile_sgemm(ଅଚଳ ସଂରଚନା ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ଅଚଳ ଶୂନ୍ୟ *, int64_t, ଅଚଳ ଶୂନ୍ୟ *, int64_t, ଶୂନ୍ୟ *, int64_t,
                     ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା);

#ifdef __cplusplus
}
#endif
