#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-cpu-impl.h"

// GGML internal header

#જો defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(શૂન્ય);
#endif
