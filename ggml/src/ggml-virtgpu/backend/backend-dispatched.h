#pragma once

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

#define GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

ساختار virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

تعریفنوع uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#شامل "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(تهی * ggml_backend_reg_fct_p);
