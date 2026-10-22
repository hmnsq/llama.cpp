#pragma once

// clang-format off
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstddef>

#অন্তর্ভাব <ggml-backend.h>

#অন্তর্ভাব "backend-convert.h"
#অন্তর্ভাব "backend-virgl-apir.h"
#অন্তর্ভাব "shared/apir_backend.h"
#অন্তর্ভাব "shared/apir_cs.h"
#অন্তর্ভাব "shared/apir_cs_ggml.h"
// clang-format on

#define GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

সংরচনা virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

প্রকারনির্দেশ uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#অন্তর্ভাব "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(শূন্য * ggml_backend_reg_fct_p);
