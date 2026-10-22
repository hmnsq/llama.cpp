#నిర్వచితమైతే GGML_USE_CPU_HBM

#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ "hbm.h"

// buffer type HBM

#అన్తర్భావ <hbwmalloc.h>

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cpu_hbm_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి "CPU_HBM";

    GGML_వాడని(buft);
}

స్థిర శూన్య ggml_backend_cpu_hbm_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    hbw_free(బఫరు->context);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_cpu_hbm_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft,
                                                                           size_t                     size) {
    శూన్య * ptr;
    పూర్ణాంకం    ఫలితం = hbw_posix_memalign(&ptr, ggml_backend_cpu_బఫరు_type_get_alignment(buft), size);
    ఒకవేళ (ఫలితం != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        తిరిగి ఖాళీ;
    }

    ggml_backend_బఫరు_t బఫరు = ggml_backend_cpu_బఫరు_from_ptr(ptr, size);
    బఫరు->buft                 = buft;
    బఫరు->iface.free_బఫరు    = ggml_backend_cpu_hbm_బఫరు_free_బఫరు;

    తిరిగి బఫరు;
}

ggml_backend_బఫరు_type_t ggml_backend_cpu_hbm_బఫరు_type(శూన్య) {
    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_cpu_బఫరు_type_hbm = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_hbm_బఫరు_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_hbm_బఫరు_type_alloc_బఫరు,
                           /* .alloc_buffer_n       = */ శూన్యసూచిక,
                           /* .get_alignment        = */ ggml_backend_cpu_బఫరు_type_get_alignment,
                           /* .get_max_size         = */ శూన్యసూచిక,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ శూన్యసూచిక,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ఖాళీ,
                           /* .is_host              = */ ggml_backend_cpu_బఫరు_type_is_host,
                           },
        /* .context  = */ శూన్యసూచిక,
    };

    తిరిగి &ggml_backend_cpu_బఫరు_type_hbm;
}
#అయితే_ముగింపు
