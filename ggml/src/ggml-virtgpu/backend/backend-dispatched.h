#ਹਦਾਇਤ once

// clang-format off
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>

#ਅਨ੍ਤਰ੍ਭਾਵ <ggml-backend.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "backend-convert.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "backend-virgl-apir.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "shared/apir_backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "shared/apir_cs.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "shared/apir_cs_ggml.h"
// clang-format on

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "

ਬਣਤਰ virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#ਅਨ੍ਤਰ੍ਭਾਵ "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(ਖਾਲੀ * ggml_backend_reg_fct_p);
