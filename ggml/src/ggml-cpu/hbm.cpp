#که_تعریف_شوی GGML_USE_CPU_HBM

#ضمن "ggml-backend.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cpu.h"
#ضمن "ggml-impl.h"

#ضمن "hbm.h"

// buffer type HBM

#ضمن <hbwmalloc.h>

ساكن ثابت حرف * ggml_backend_cpu_hbm_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع "CPU_HBM";

    GGML_ناکارول_شوی(buft);
}

ساكن فراغ ggml_backend_cpu_hbm_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    hbw_free(بفر->context);
}

ساكن ggml_backend_بفر_t ggml_backend_cpu_hbm_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                           size_t                     size) {
    فراغ * ptr;
    صحيح    پایله = hbw_posix_memalign(&ptr, ggml_backend_cpu_بفر_type_get_alignment(buft), size);
    إذا (پایله != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        أرجع عدم;
    }

    ggml_backend_بفر_t بفر = ggml_backend_cpu_بفر_from_ptr(ptr, size);
    بفر->buft                 = buft;
    بفر->iface.free_بفر    = ggml_backend_cpu_hbm_بفر_free_بفر;

    أرجع بفر;
}

ggml_backend_بفر_type_t ggml_backend_cpu_hbm_بفر_type(فراغ) {
    ساكن بنية ggml_backend_بفر_type ggml_backend_cpu_بفر_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_بفر_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_بفر_type_alloc_بفر,
                           /* .alloc_buffer_n       = */ تش_اشاره,
                           /* .get_alignment        = */ ggml_backend_cpu_بفر_type_get_alignment,
                           /* .get_max_size         = */ تش_اشاره,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ تش_اشاره,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ عدم,
                           /* .is_host              = */ ggml_backend_cpu_بفر_type_is_host,
                           },
        /* .context  = */ تش_اشاره,
    };

    أرجع &ggml_backend_cpu_بفر_type_hbm;
}
#د_که_پای
