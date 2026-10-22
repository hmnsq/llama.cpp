#ifdef GGML_USE_CPU_HBM

#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-cpu.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব "hbm.h"

// buffer type HBM

#অন্তর্ভাব <hbwmalloc.h>

স্থির ধ্রুবক অক্ষর * ggml_backend_cpu_hbm_বাফার_type_get_name(ggml_backend_বাফার_type_t buft) {
    ফেরত "CPU_HBM";

    GGML_UNUSED(buft);
}

স্থির শূন্য ggml_backend_cpu_hbm_বাফার_free_বাফার(ggml_backend_বাফার_t বাফার) {
    hbw_free(বাফার->context);
}

স্থির ggml_backend_বাফার_t ggml_backend_cpu_hbm_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft,
                                                                           size_t                     size) {
    শূন্য * ptr;
    পূর্ণসংখ্যা    ফলাফল = hbw_posix_memalign(&ptr, ggml_backend_cpu_বাফার_type_get_alignment(buft), size);
    যদি (ফলাফল != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        ফেরত ফাঁকা;
    }

    ggml_backend_বাফার_t বাফার = ggml_backend_cpu_বাফার_from_ptr(ptr, size);
    বাফার->buft                 = buft;
    বাফার->iface.free_বাফার    = ggml_backend_cpu_hbm_বাফার_free_বাফার;

    ফেরত বাফার;
}

ggml_backend_বাফার_type_t ggml_backend_cpu_hbm_বাফার_type(শূন্য) {
    স্থির কাঠামো ggml_backend_বাফার_type ggml_backend_cpu_বাফার_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_বাফার_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_বাফার_type_alloc_বাফার,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_বাফার_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ফাঁকা,
                           /* .is_host              = */ ggml_backend_cpu_বাফার_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    ফেরত &ggml_backend_cpu_বাফার_type_hbm;
}
#endif
