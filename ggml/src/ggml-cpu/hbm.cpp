#ifdef GGML_USE_CPU_HBM

#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-cpu.h"
#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ "hbm.h"

// buffer type HBM

#அந்தர்பாவ <hbwmalloc.h>

நிலையான மாறிலி எழுத்து * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    திரும்பு "CPU_HBM";

    GGML_UNUSED(buft);
}

நிலையான வெறுமை ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

நிலையான ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    வெறுமை * ptr;
    முழு    result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    எனில் (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        திரும்பு வெற்று;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_hbm_buffer_free_buffer;

    திரும்பு buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(வெறுமை) {
    நிலையான கட்டமைப்பு ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ வெற்று,
                           /* .is_host              = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    திரும்பு &ggml_backend_cpu_buffer_type_hbm;
}
#endif
