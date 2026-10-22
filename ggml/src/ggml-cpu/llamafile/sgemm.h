#pragma once
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdbool.h>

#जइ defined(__VXE__) || defined(__VXE2__)
#अन्तर्भाव <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#अण्णहा
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
बाह्य "C" {
#endif

bool llamafile_sgemm(ध्रुव संरचना ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ध्रुव शून्य *, int64_t, ध्रुव शून्य *, int64_t, शून्य *, int64_t,
                     पूर्णाङ्क, पूर्णाङ्क, पूर्णाङ्क);

#ifdef __cplusplus
}
#endif
