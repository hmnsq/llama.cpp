#आशय once

// clang-format off
#समावेश <cstdint>
#समावेश <cstddef>

#समावेश <ggml-backend.h>

#समावेश "backend-convert.h"
#समावेश "backend-virgl-apir.h"
#समावेश "shared/apir_backend.h"
#समावेश "shared/apir_cs.h"
#समावेश "shared/apir_cs_ggml.h"
// clang-format on

#घोषणा GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

काष्ठा virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

प्रकार uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#समावेश "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(व्योम * ggml_backend_reg_fct_p);
