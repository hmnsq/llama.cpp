#ہدایت once

#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"
#شامل "shared/api_remoting.h"

#شامل <cstdarg>
#شامل <cstdio>
#شامل <cstdlib>

خارجی ggml_backend_reg_t reg;
خارجی ggml_backend_dev_t dev;
خارجی ggml_backend_t     bck;

ساخت virgl_apir_callbacks {
    ثابت حرف * (*get_config)(uint32_t virgl_ctx_id, ثابت حرف * کلید);
    خالی * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

خارجی "C" {
Apir_بھرو_Library_Return_رمز apir_backend_initialize(uint32_t virgl_ctx_id, ساخت virgl_apir_callbacks * virgl_cbs);
خالی                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  حرف *                 dec_cur,
                                                  ثابت حرف *           dec_انتہا,
                                                  حرف *                 enc_cur,
                                                  ثابت حرف *           enc_انتہا,
                                                  حرف **                enc_cur_after);
}
