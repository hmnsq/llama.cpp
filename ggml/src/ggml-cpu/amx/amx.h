#समावेश "ggml-backend.h"
#समावेश "ggml-cpu-impl.h"

// GGML internal header

#अगर defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_बफर_type_t ggml_backend_amx_बफर_type(व्योम);
#पूर्ण_अगर
