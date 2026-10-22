#pragma once

// clang-format off
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstddef>

#அந்தர்பாவ <ggml-backend.h>

#அந்தர்பாவ "backend-convert.h"
#அந்தர்பாவ "backend-virgl-apir.h"
#அந்தர்பாவ "shared/apir_backend.h"
#அந்தர்பாவ "shared/apir_cs.h"
#அந்தர்பாவ "shared/apir_cs_ggml.h"
// clang-format on

#define GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

கட்டமைப்பு virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

வகைவரை uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#அந்தர்பாவ "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(வெறுமை * ggml_backend_reg_fct_p);
