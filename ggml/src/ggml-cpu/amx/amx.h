#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-cpu-impl.h"

// GGML internal header

#ఒకవేళ defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_బఫరు_type_t ggml_backend_amx_బఫరు_type(శూన్య);
#అయితే_ముగింపు
