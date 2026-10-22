#pragma once
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdbool.h>

#जर defined(__VXE__) || defined(__VXE2__)
#अन्तर्भाव <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#अन्यथा
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
बाह्य "C" {
#endif

bool llamafile_sgemm(स्थिरांक संरचना ggml_compute_params * params, int64_t, int64_t, int64_t,
                     स्थिरांक शून्य *, int64_t, स्थिरांक शून्य *, int64_t, शून्य *, int64_t,
                     पूर्णांक, पूर्णांक, पूर्णांक);

#ifdef __cplusplus
}
#endif
