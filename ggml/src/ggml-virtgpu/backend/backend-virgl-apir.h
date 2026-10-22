#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "shared/api_remoting.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdarg>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>

ਬਾਹਰੀ ggml_backend_reg_t reg;
ਬਾਹਰੀ ggml_backend_dev_t dev;
ਬਾਹਰੀ ggml_backend_t     bck;

ਬਣਤਰ virgl_apir_callbacks {
    ਅਚਲ ਅੱਖਰ * (*get_config)(uint32_t virgl_ctx_id, ਅਚਲ ਅੱਖਰ * ਕੁੰਜੀ);
    ਖਾਲੀ * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

ਬਾਹਰੀ "C" {
Apir_ਲੱਦੋ_Library_Return_ਕੋਡ apir_backend_initialize(uint32_t virgl_ctx_id, ਬਣਤਰ virgl_apir_callbacks * virgl_cbs);
ਖਾਲੀ                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  ਅੱਖਰ *                 dec_cur,
                                                  ਅਚਲ ਅੱਖਰ *           dec_ਸਿਰਾ,
                                                  ਅੱਖਰ *                 enc_cur,
                                                  ਅਚਲ ਅੱਖਰ *           enc_ਸਿਰਾ,
                                                  ਅੱਖਰ **                enc_cur_after);
}
