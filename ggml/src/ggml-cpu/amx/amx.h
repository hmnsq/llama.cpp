#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-cpu-impl.h"

// GGML internal header

#सचे defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(शून्य);
#endif
