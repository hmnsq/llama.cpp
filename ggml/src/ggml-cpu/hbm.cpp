#ifdef GGML_USE_CPU_HBM

#অন্তর্ভাৱ "ggml-backend.h"
#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-cpu.h"
#অন্তর্ভাৱ "ggml-impl.h"

#অন্তর্ভাৱ "hbm.h"

// buffer type HBM

#অন্তর্ভাৱ <hbwmalloc.h>

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ঘূৰাই "CPU_HBM";

    GGML_UNUSED(buft);
}

স্থির শূন্য ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

স্থির ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    শূন্য * ptr;
    পূর্ণাঙ্ক    result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    যদি (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        ঘূৰাই রিক্ত;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_hbm_buffer_free_buffer;

    ঘূৰাই buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(শূন্য) {
    স্থির সংরচনা ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ রিক্ত,
                           /* .is_host              = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    ঘূৰাই &ggml_backend_cpu_buffer_type_hbm;
}
#endif
