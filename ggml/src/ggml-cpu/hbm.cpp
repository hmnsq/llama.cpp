#ifdef GGML_USE_CPU_HBM

#شامل "ggml-backend.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-cpu.h"
#شامل "ggml-impl.h"

#شامل "hbm.h"

// buffer type HBM

#شامل <hbwmalloc.h>

ایستا ثابت نویسه * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    بازگشت "CPU_HBM";

    GGML_UNUSED(buft);
}

ایستا تهی ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

ایستا ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    تهی * ptr;
    صحیح    result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    اگر (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        بازگشت پوچ;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_hbm_buffer_free_buffer;

    بازگشت buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(تهی) {
    ایستا ساختار ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ پوچ,
                           /* .is_host              = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    بازگشت &ggml_backend_cpu_buffer_type_hbm;
}
#endif
