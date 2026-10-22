#هدايت once

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "shared/api_remoting.h"

#ضمن <cstdarg>
#ضمن <cstdio>
#ضمن <cstdlib>

خارجي ggml_backend_reg_t reg;
خارجي ggml_backend_dev_t dev;
خارجي ggml_backend_t     bck;

بنية virgl_apir_callbacks {
    ثابت حرف * (*get_config)(uint32_t virgl_ctx_id, ثابت حرف * ڪنجي);
    فراغ * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

خارجي "C" {
Apir_ڀريو_Library_Return_ڪوڊ apir_backend_initialize(uint32_t virgl_ctx_id, بنية virgl_apir_callbacks * virgl_cbs);
فراغ                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  حرف *                 dec_cur,
                                                  ثابت حرف *           dec_پڇاڙي,
                                                  حرف *                 enc_cur,
                                                  ثابت حرف *           enc_پڇاڙي,
                                                  حرف **                enc_cur_after);
}
