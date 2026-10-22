#ضمن "ggml-backend.h"
#ضمن "ggml-cpu-impl.h"

// GGML internal header

#אם defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(فراغ);
#endif
