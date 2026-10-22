#ifdef GGML_USE_CPU_HBM

#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव "hbm.h"

// buffer type HBM

#अन्तर्भाव <hbwmalloc.h>

स्थिर ध्रुव वर्ण * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    प्रत्यागम "CPU_HBM";

    GGML_UNUSED(buft);
}

स्थिर शून्य ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

स्थिर ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    शून्य * ptr;
    पूर्णाङ्क    result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    जइ (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        प्रत्यागम रिक्त;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_hbm_buffer_free_buffer;

    प्रत्यागम buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(शून्य) {
    स्थिर संरचना ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ रिक्त,
                           /* .is_host              = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    प्रत्यागम &ggml_backend_cpu_buffer_type_hbm;
}
#endif
