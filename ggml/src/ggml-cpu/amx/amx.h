#ضمن "ggml-backend.h"
#ضمن "ggml-cpu-impl.h"

// GGML internal header

#إذا defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_میانگیر_type_t ggml_backend_amx_میانگیر_type(فراغ);
#پایان_اگر
