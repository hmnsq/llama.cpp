#यदि_परिभाषितम् GGML_USE_CPU_HBM

#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव "hbm.h"

// buffer type HBM

#अन्तर्भाव <hbwmalloc.h>

स्थिर ध्रुव वर्ण * ggml_backend_cpu_hbm_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    प्रत्यागम "CPU_HBM";

    GGML_अप्रयुक्तम्(buft);
}

स्थिर शून्य ggml_backend_cpu_hbm_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    hbw_free(बफर->context);
}

स्थिर ggml_backend_बफर_t ggml_backend_cpu_hbm_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft,
                                                                           size_t                     size) {
    शून्य * ptr;
    पूर्णाङ्क    परिणाम = hbw_posix_memalign(&ptr, ggml_backend_cpu_बफर_type_get_alignment(buft), size);
    यदि (परिणाम != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        प्रत्यागम रिक्त;
    }

    ggml_backend_बफर_t बफर = ggml_backend_cpu_बफर_from_ptr(ptr, size);
    बफर->buft                 = buft;
    बफर->iface.free_बफर    = ggml_backend_cpu_hbm_बफर_free_बफर;

    प्रत्यागम बफर;
}

ggml_backend_बफर_type_t ggml_backend_cpu_hbm_बफर_type(शून्य) {
    स्थिर संरचना ggml_backend_बफर_type ggml_backend_cpu_बफर_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_बफर_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_बफर_type_alloc_बफर,
                           /* .alloc_buffer_n       = */ शून्यसूचकः,
                           /* .get_alignment        = */ ggml_backend_cpu_बफर_type_get_alignment,
                           /* .get_max_size         = */ शून्यसूचकः,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ शून्यसूचकः,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ रिक्त,
                           /* .is_host              = */ ggml_backend_cpu_बफर_type_is_host,
                           },
        /* .context  = */ शून्यसूचकः,
    };

    प्रत्यागम &ggml_backend_cpu_बफर_type_hbm;
}
#यदि_अन्तः
