#ifdef GGML_USE_CPU_HBM

#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ "hbm.h"

// buffer type HBM

#ಅನ್ತರ್ಭಾವ <hbwmalloc.h>

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ "CPU_HBM";

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    ಶೂನ್ಯ * ptr;
    ಪೂರ್ಣಾಙ್ಕ    result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    ಯದಿ (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_hbm_buffer_free_buffer;

    ಪ್ರತ್ಯಾಗಮ buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ರಿಕ್ತ,
                           /* .is_host              = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_cpu_buffer_type_hbm;
}
#endif
