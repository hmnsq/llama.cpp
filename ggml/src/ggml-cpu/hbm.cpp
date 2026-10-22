#اگر_تعریف_شدہ GGML_USE_CPU_HBM

#شامل "ggml-backend.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-cpu.h"
#شامل "ggml-impl.h"

#شامل "hbm.h"

// buffer type HBM

#شامل <hbwmalloc.h>

ساکن ثابت حرف * ggml_backend_cpu_hbm_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    واپس "CPU_HBM";

    GGML_غیر_مستعمل(buft);
}

ساکن خالی ggml_backend_cpu_hbm_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    hbw_free(بفر->context);
}

ساکن ggml_backend_بفر_t ggml_backend_cpu_hbm_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                           size_t                     size) {
    خالی * ptr;
    صحیح    نتیجہ = hbw_posix_memalign(&ptr, ggml_backend_cpu_بفر_type_get_alignment(buft), size);
    اگر (نتیجہ != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        واپس تہی;
    }

    ggml_backend_بفر_t بفر = ggml_backend_cpu_بفر_from_ptr(ptr, size);
    بفر->buft                 = buft;
    بفر->iface.free_بفر    = ggml_backend_cpu_hbm_بفر_free_بفر;

    واپس بفر;
}

ggml_backend_بفر_type_t ggml_backend_cpu_hbm_بفر_type(خالی) {
    ساکن ساخت ggml_backend_بفر_type ggml_backend_cpu_بفر_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_بفر_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_بفر_type_alloc_بفر,
                           /* .alloc_buffer_n       = */ خالی_اشاریہ,
                           /* .get_alignment        = */ ggml_backend_cpu_بفر_type_get_alignment,
                           /* .get_max_size         = */ خالی_اشاریہ,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ خالی_اشاریہ,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ تہی,
                           /* .is_host              = */ ggml_backend_cpu_بفر_type_is_host,
                           },
        /* .context  = */ خالی_اشاریہ,
    };

    واپس &ggml_backend_cpu_بفر_type_hbm;
}
#اگر_ختم
