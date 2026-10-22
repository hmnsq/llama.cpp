#आशय once
#समावेश <stdint.h>
#समावेश <stdbool.h>

#अगर defined(__VXE__) || defined(__VXE2__)
#समावेश <vecintrin.h>
#पूर्ण_अगर

#अगर_घोषित _MSC_VER
#घोषणा NOINLINE __declspec(noinline)
#अन्यथा
#घोषणा NOINLINE __attribute__((__noinline__))
#पूर्ण_अगर

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

bool llamafile_sgemm(स्थिर काष्ठा ggml_compute_params * params, int64_t, int64_t, int64_t,
                     स्थिर व्योम *, int64_t, स्थिर व्योम *, int64_t, व्योम *, int64_t,
                     पूर्णांक, पूर्णांक, पूर्णांक);

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर
