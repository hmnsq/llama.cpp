#pragma once

// clang-format off
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstddef>

#ಅನ್ತರ್ಭಾವ <ggml-backend.h>

#ಅನ್ತರ್ಭಾವ "backend-convert.h"
#ಅನ್ತರ್ಭಾವ "backend-virgl-apir.h"
#ಅನ್ತರ್ಭಾವ "shared/apir_backend.h"
#ಅನ್ತರ್ಭಾವ "shared/apir_cs.h"
#ಅನ್ತರ್ಭಾವ "shared/apir_cs_ggml.h"
// clang-format on

#define GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

ಸಂರಚನಾ virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

ಪ್ರಕಾರನಿರ್ದೇಶ uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#ಅನ್ತರ್ಭಾವ "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(ಶೂನ್ಯ * ggml_backend_reg_fct_p);
