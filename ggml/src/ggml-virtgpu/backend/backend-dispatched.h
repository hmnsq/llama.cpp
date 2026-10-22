#సూచన once

// clang-format off
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstddef>

#అన్తర్భావ <ggml-backend.h>

#అన్తర్భావ "backend-convert.h"
#అన్తర్భావ "backend-virgl-apir.h"
#అన్తర్భావ "shared/apir_backend.h"
#అన్తర్భావ "shared/apir_cs.h"
#అన్తర్భావ "shared/apir_cs_ggml.h"
// clang-format on

#నిర్వచించు GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

నిర్మాణం virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

రకనిర్వచనం uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#అన్తర్భావ "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(శూన్య * ggml_backend_reg_fct_p);
