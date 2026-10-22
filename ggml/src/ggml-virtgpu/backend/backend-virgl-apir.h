#pragma once

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "shared/api_remoting.h"

#অন্তর্ভাব <cstdarg>
#অন্তর্ভাব <cstdio>
#অন্তর্ভাব <cstdlib>

বাহ্য ggml_backend_reg_t reg;
বাহ্য ggml_backend_dev_t dev;
বাহ্য ggml_backend_t     bck;

সংরচনা virgl_apir_callbacks {
    ধ্রুব বর্ণ * (*get_config)(uint32_t virgl_ctx_id, ধ্রুব বর্ণ * চাবি);
    শূন্য * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

বাহ্য "C" {
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, সংরচনা virgl_apir_callbacks * virgl_cbs);
শূন্য                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  বর্ণ *                 dec_cur,
                                                  ধ্রুব বর্ণ *           dec_end,
                                                  বর্ণ *                 enc_cur,
                                                  ধ্রুব বর্ণ *           enc_end,
                                                  বর্ণ **                enc_cur_after);
}
