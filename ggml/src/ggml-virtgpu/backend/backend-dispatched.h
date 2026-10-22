#pragma once

// clang-format off
#ضمن <cstdint>
#ضمن <cstddef>

#ضمن <ggml-backend.h>

#ضمن "backend-convert.h"
#ضمن "backend-virgl-apir.h"
#ضمن "shared/apir_backend.h"
#ضمن "shared/apir_cs.h"
#ضمن "shared/apir_cs_ggml.h"
// clang-format on

#define GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

بنية virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

تعريفنوع uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#ضمن "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(فراغ * ggml_backend_reg_fct_p);
