#pragma once

#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"
#شامل "shared/api_remoting.h"

#شامل <cstdarg>
#شامل <cstdio>
#شامل <cstdlib>

بیرونی ggml_backend_reg_t reg;
بیرونی ggml_backend_dev_t dev;
بیرونی ggml_backend_t     bck;

ساختار virgl_apir_callbacks {
    ثابت نویسه * (*get_config)(uint32_t virgl_ctx_id, ثابت نویسه * کلید);
    تهی * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

بیرونی "C" {
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, ساختار virgl_apir_callbacks * virgl_cbs);
تهی                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  نویسه *                 dec_cur,
                                                  ثابت نویسه *           dec_end,
                                                  نویسه *                 enc_cur,
                                                  ثابت نویسه *           enc_end,
                                                  نویسه **                enc_cur_after);
}
