#اگر_تعریف_شده GGML_USE_CPU_HBM

#ضمن "ggml-backend.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cpu.h"
#ضمن "ggml-impl.h"

#ضمن "hbm.h"

// buffer type HBM

#ضمن <hbwmalloc.h>

ساكن ثابت حرف * ggml_backend_cpu_hbm_میانگیر_type_get_name(ggml_backend_میانگیر_type_t buft) {
    أرجع "CPU_HBM";

    GGML_استفادهنشده(buft);
}

ساكن فراغ ggml_backend_cpu_hbm_میانگیر_free_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    hbw_free(میانگیر->context);
}

ساكن ggml_backend_میانگیر_t ggml_backend_cpu_hbm_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft,
                                                                           size_t                     size) {
    فراغ * ptr;
    صحيح    نتیجه = hbw_posix_memalign(&ptr, ggml_backend_cpu_میانگیر_type_get_alignment(buft), size);
    إذا (نتیجه != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        أرجع عدم;
    }

    ggml_backend_میانگیر_t میانگیر = ggml_backend_cpu_میانگیر_from_ptr(ptr, size);
    میانگیر->buft                 = buft;
    میانگیر->iface.free_میانگیر    = ggml_backend_cpu_hbm_میانگیر_free_میانگیر;

    أرجع میانگیر;
}

ggml_backend_میانگیر_type_t ggml_backend_cpu_hbm_میانگیر_type(فراغ) {
    ساكن بنية ggml_backend_میانگیر_type ggml_backend_cpu_میانگیر_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_میانگیر_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_میانگیر_type_alloc_میانگیر,
                           /* .alloc_buffer_n       = */ اشاره_تهی,
                           /* .get_alignment        = */ ggml_backend_cpu_میانگیر_type_get_alignment,
                           /* .get_max_size         = */ اشاره_تهی,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ اشاره_تهی,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ عدم,
                           /* .is_host              = */ ggml_backend_cpu_میانگیر_type_is_host,
                           },
        /* .context  = */ اشاره_تهی,
    };

    أرجع &ggml_backend_cpu_میانگیر_type_hbm;
}
#پایان_اگر
