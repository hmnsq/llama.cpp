#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_HBM

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "hbm.h"

// buffer type HBM

#ਅਨ੍ਤਰ੍ਭਾਵ <hbwmalloc.h>

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cpu_hbm_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ "CPU_HBM";

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_hbm_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    hbw_free(ਬਫਰ->context);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_cpu_hbm_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft,
                                                                           size_t                     size) {
    ਖਾਲੀ * ptr;
    ਪੂਰਨ    ਨਤੀਜਾ = hbw_posix_memalign(&ptr, ggml_backend_cpu_ਬਫਰ_type_get_alignment(buft), size);
    ਜੇ (ਨਤੀਜਾ != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_cpu_ਬਫਰ_from_ptr(ptr, size);
    ਬਫਰ->buft                 = buft;
    ਬਫਰ->iface.free_ਬਫਰ    = ggml_backend_cpu_hbm_ਬਫਰ_free_ਬਫਰ;

    ਵਾਪਸ ਬਫਰ;
}

ggml_backend_ਬਫਰ_type_t ggml_backend_cpu_hbm_ਬਫਰ_type(ਖਾਲੀ) {
    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_cpu_ਬਫਰ_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_ਬਫਰ_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_ਬਫਰ_type_alloc_ਬਫਰ,
                           /* .alloc_buffer_n       = */ ਖਾਲੀ_ਸੂਚਕ,
                           /* .get_alignment        = */ ggml_backend_cpu_ਬਫਰ_type_get_alignment,
                           /* .get_max_size         = */ ਖਾਲੀ_ਸੂਚਕ,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ਖਾਲੀ_ਸੂਚਕ,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ਨਿਰਾ,
                           /* .is_host              = */ ggml_backend_cpu_ਬਫਰ_type_is_host,
                           },
        /* .context  = */ ਖਾਲੀ_ਸੂਚਕ,
    };

    ਵਾਪਸ &ggml_backend_cpu_ਬਫਰ_type_hbm;
}
#ਜੇ_ਅੰਤ
