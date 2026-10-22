#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_HBM

#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ "hbm.h"

// buffer type HBM

#ಅನ್ತರ್ಭಾವ <hbwmalloc.h>

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cpu_hbm_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ "CPU_HBM";

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_hbm_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    hbw_free(ಬಫರ್->context);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_cpu_hbm_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft,
                                                                           size_t                     size) {
    ಶೂನ್ಯ * ptr;
    ಪೂರ್ಣಾಂಕ    ಫಲಿತಾಂಶ = hbw_posix_memalign(&ptr, ggml_backend_cpu_ಬಫರ್_type_get_alignment(buft), size);
    ಆದರೆ (ಫಲಿತಾಂಶ != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_cpu_ಬಫರ್_from_ptr(ptr, size);
    ಬಫರ್->buft                 = buft;
    ಬಫರ್->iface.free_ಬಫರ್    = ggml_backend_cpu_hbm_ಬಫರ್_free_ಬಫರ್;

    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

ggml_backend_ಬಫರ್_type_t ggml_backend_cpu_hbm_ಬಫರ್_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_type ggml_backend_cpu_ಬಫರ್_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_ಬಫರ್_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_ಬಫರ್_type_alloc_ಬಫರ್,
                           /* .alloc_buffer_n       = */ ಶೂನ್ಯಸೂಚಕ,
                           /* .get_alignment        = */ ggml_backend_cpu_ಬಫರ್_type_get_alignment,
                           /* .get_max_size         = */ ಶೂನ್ಯಸೂಚಕ,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ಶೂನ್ಯಸೂಚಕ,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ಖಾಲಿ,
                           /* .is_host              = */ ggml_backend_cpu_ಬಫರ್_type_is_host,
                           },
        /* .context  = */ ಶೂನ್ಯಸೂಚಕ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_cpu_ಬಫರ್_type_hbm;
}
#ಆದರೆ_ಅಂತ್ಯ
