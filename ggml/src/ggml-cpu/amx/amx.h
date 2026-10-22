#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-cpu-impl.h"

// GGML internal header

#যদি defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(শূন্য);
#endif
