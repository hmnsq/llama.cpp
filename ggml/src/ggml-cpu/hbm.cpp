#ifdef GGML_USE_CPU_HBM

#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#ଅନ୍ତର୍ଭାଵ "hbm.h"

// buffer type HBM

#ଅନ୍ତର୍ଭାଵ <hbwmalloc.h>

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ "CPU_HBM";

    GGML_UNUSED(buft);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    ଶୂନ୍ୟ * ptr;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    ଯଦି (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_hbm_buffer_free_buffer;

    ଫେରନ୍ତୁ buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(ଶୂନ୍ୟ) {
    ସ୍ଥିର ସଂରଚନା ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ଖାଲି,
                           /* .is_host              = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    ଫେରନ୍ତୁ &ggml_backend_cpu_buffer_type_hbm;
}
#endif
