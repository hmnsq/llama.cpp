#pragma once

// clang-format off
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstddef>

#અન્તર્ભાવ <ggml-backend.h>

#અન્તર્ભાવ "backend-convert.h"
#અન્તર્ભાવ "backend-virgl-apir.h"
#અન્તર્ભાવ "shared/apir_backend.h"
#અન્તર્ભાવ "shared/apir_cs.h"
#અન્તર્ભાવ "shared/apir_cs_ggml.h"
// clang-format on

#define GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

સંરચના virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

પ્રકારવ્યાખ્યા uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#અન્તર્ભાવ "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(શૂન્ય * ggml_backend_reg_fct_p);
