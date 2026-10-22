#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-cpu-impl.h"

// GGML internal header

#എങ്കിൽ defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(ശൂന്യം);
#endif
