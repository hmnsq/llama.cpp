#अगर_घोषित GGML_USE_CPU_HBM

#समावेश "ggml-backend.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-cpu.h"
#समावेश "ggml-impl.h"

#समावेश "hbm.h"

// buffer type HBM

#समावेश <hbwmalloc.h>

अटल स्थिर अक्षर * ggml_backend_cpu_hbm_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    वापस "CPU_HBM";

    GGML_UNUSED(buft);
}

अटल व्योम ggml_backend_cpu_hbm_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    hbw_free(बफर->context);
}

अटल ggml_backend_बफर_t ggml_backend_cpu_hbm_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft,
                                                                           माप_प्रकार                     size) {
    व्योम * ptr;
    पूर्णांक    परिणाम = hbw_posix_memalign(&ptr, ggml_backend_cpu_बफर_type_get_alignment(buft), size);
    अगर (परिणाम != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        वापस शून्य;
    }

    ggml_backend_बफर_t बफर = ggml_backend_cpu_बफर_from_ptr(ptr, size);
    बफर->buft                 = buft;
    बफर->iface.free_बफर    = ggml_backend_cpu_hbm_बफर_free_बफर;

    वापस बफर;
}

ggml_backend_बफर_type_t ggml_backend_cpu_hbm_बफर_type(व्योम) {
    अटल काष्ठा ggml_backend_बफर_type ggml_backend_cpu_बफर_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_बफर_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_बफर_type_alloc_बफर,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_बफर_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ शून्य,
                           /* .is_host              = */ ggml_backend_cpu_बफर_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    वापस &ggml_backend_cpu_बफर_type_hbm;
}
#पूर्ण_अगर
