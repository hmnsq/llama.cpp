#pragma once

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "shared/api_remoting.h"

#ಅನ್ತರ್ಭಾವ <cstdarg>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstdlib>

ಬಾಹ್ಯ ggml_backend_reg_t reg;
ಬಾಹ್ಯ ggml_backend_dev_t dev;
ಬಾಹ್ಯ ggml_backend_t     bck;

ಸಂರಚನಾ virgl_apir_callbacks {
    ಧ್ರುವ ವರ್ಣ * (*get_config)(uint32_t virgl_ctx_id, ಧ್ರುವ ವರ್ಣ * ಕೀಲಿ);
    ಶೂನ್ಯ * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

ಬಾಹ್ಯ "C" {
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, ಸಂರಚನಾ virgl_apir_callbacks * virgl_cbs);
ಶೂನ್ಯ                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  ವರ್ಣ *                 dec_cur,
                                                  ಧ್ರುವ ವರ್ಣ *           dec_end,
                                                  ವರ್ಣ *                 enc_cur,
                                                  ಧ್ರುವ ವರ್ಣ *           enc_end,
                                                  ವರ್ಣ **                enc_cur_after);
}
