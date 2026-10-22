#निर्देशः once
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdbool.h>

#यदि defined(__VXE__) || defined(__VXE2__)
#अन्तर्भाव <vecintrin.h>
#यदि_अन्तः

#यदि_परिभाषितम् _MSC_VER
#परिभाषय NOINLINE __declspec(noinline)
#अन्यथा
#परिभाषय NOINLINE __attribute__((__noinline__))
#यदि_अन्तः

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

bool llamafile_sgemm(ध्रुव संरचना ggml_compute_params * params, int64_t, int64_t, int64_t,
                     ध्रुव शून्य *, int64_t, ध्रुव शून्य *, int64_t, शून्य *, int64_t,
                     पूर्णाङ्क, पूर्णाङ्क, पूर्णाङ्क);

#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः
