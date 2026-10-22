#ifdef GGML_USE_CPU_HBM

#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-cpu.h"
#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ "hbm.h"

// buffer type HBM

#અન્તર્ભાવ <hbwmalloc.h>

સ્થિર અચલ અક્ષર * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    પાછા "CPU_HBM";

    GGML_UNUSED(buft);
}

સ્થિર શૂન્ય ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

સ્થિર ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    શૂન્ય * ptr;
    પૂર્ણાંક    result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    જો (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        પાછા ખાલી;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_hbm_buffer_free_buffer;

    પાછા buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(શૂન્ય) {
    સ્થિર સંરચના ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ nullptr,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ખાલી,
                           /* .is_host              = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    પાછા &ggml_backend_cpu_buffer_type_hbm;
}
#endif
