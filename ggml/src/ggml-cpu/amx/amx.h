#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-cpu-impl.h"

// GGML internal header

#यदि defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_बफर_type_t ggml_backend_amx_बफर_type(शून्य);
#यदि_अन्तः
