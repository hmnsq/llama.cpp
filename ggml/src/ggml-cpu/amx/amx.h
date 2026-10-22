#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu-impl.h"

// GGML internal header

#ଯଦି defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(ଶୂନ୍ୟ);
#endif
