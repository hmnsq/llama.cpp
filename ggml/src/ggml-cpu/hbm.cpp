#ifdef GGML_USE_CPU_HBM

#ضمن "ggml-backend.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cpu.h"
#ضمن "ggml-impl.h"

#ضمن "hbm.h"

// buffer type HBM

#ضمن <hbwmalloc.h>

ساكن ثابت حرف * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    إرجاع "CPU_HBM";

    GGML_UNUSED(buft);
}

ساكن فراغ ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

ساكن ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    فراغ * ptr;
    صحيح    result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    إذا (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        إرجاع عدم;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_hbm_buffer_free_buffer;

    إرجاع buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(فراغ) {
    ساكن بنية ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ عدم,
                           /* .is_host              = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    إرجاع &ggml_backend_cpu_buffer_type_hbm;
}
#endif
