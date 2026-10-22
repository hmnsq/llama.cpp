#pragma once

// clang-format off
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstddef>

#അന്തര്ഭാവ <ggml-backend.h>

#അന്തര്ഭാവ "backend-convert.h"
#അന്തര്ഭാവ "backend-virgl-apir.h"
#അന്തര്ഭാവ "shared/apir_backend.h"
#അന്തര്ഭാവ "shared/apir_cs.h"
#അന്തര്ഭാവ "shared/apir_cs_ggml.h"
// clang-format on

#define GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

ഘടന virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

തരനിർവചനം uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#അന്തര്ഭാവ "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(ശൂന്യം * ggml_backend_reg_fct_p);
