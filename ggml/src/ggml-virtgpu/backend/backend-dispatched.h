#pragma once

// clang-format off
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstddef>

#अन्तर्भाव <ggml-backend.h>

#अन्तर्भाव "backend-convert.h"
#अन्तर्भाव "backend-virgl-apir.h"
#अन्तर्भाव "shared/apir_backend.h"
#अन्तर्भाव "shared/apir_cs.h"
#अन्तर्भाव "shared/apir_cs_ggml.h"
// clang-format on

#define GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

संरचना virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

प्रकारनिर्देश uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#अन्तर्भाव "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(शून्य * ggml_backend_reg_fct_p);
