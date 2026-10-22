#pragma once

// clang-format off
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cstddef>

#ଅନ୍ତର୍ଭାଵ <ggml-backend.h>

#ଅନ୍ତର୍ଭାଵ "backend-convert.h"
#ଅନ୍ତର୍ଭାଵ "backend-virgl-apir.h"
#ଅନ୍ତର୍ଭାଵ "shared/apir_backend.h"
#ଅନ୍ତର୍ଭାଵ "shared/apir_cs.h"
#ଅନ୍ତର୍ଭାଵ "shared/apir_cs_ggml.h"
// clang-format on

#define GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

ସଂରଚନା virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

ପ୍ରକାରବ୍ୟାଖ୍ୟା uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#ଅନ୍ତର୍ଭାଵ "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(ଶୂନ୍ୟ * ggml_backend_reg_fct_p);
