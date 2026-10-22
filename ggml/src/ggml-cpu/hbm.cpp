#ifdef GGML_USE_CPU_HBM

#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-cpu.h"
#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ "hbm.h"

// buffer type HBM

#അന്തര്ഭാവ <hbwmalloc.h>

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    തിരികെ "CPU_HBM";

    GGML_UNUSED(buft);
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    ശൂന്യം * ptr;
    പൂർണ്ണസംഖ്യ    result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    എങ്കിൽ (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        തിരികെ ശൂന്യ;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_hbm_buffer_free_buffer;

    തിരികെ buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(ശൂന്യം) {
    സ്ഥിതം ഘടന ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ശൂന്യ,
                           /* .is_host              = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    തിരികെ &ggml_backend_cpu_buffer_type_hbm;
}
#endif
