#ہدایت once

// clang-format off
#شامل <cstdint>
#شامل <cstddef>

#شامل <ggml-backend.h>

#شامل "backend-convert.h"
#شامل "backend-virgl-apir.h"
#شامل "shared/apir_backend.h"
#شامل "shared/apir_cs.h"
#شامل "shared/apir_cs_ggml.h"
// clang-format on

#تعریف_کرو GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

ساخت virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

قسمتعریف uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#شامل "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(خالی * ggml_backend_reg_fct_p);
