#ಸೂಚನೆ once

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

ರಚನೆ virgl_apir_callbacks {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * (*get_config)(uint32_t virgl_ctx_id, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೀಲಿ);
    ಶೂನ್ಯ * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

ಬಾಹ್ಯ "C" {
Apir_ತುಂಬು_Library_Return_ಸಂಕೇತ apir_backend_initialize(uint32_t virgl_ctx_id, ರಚನೆ virgl_apir_callbacks * virgl_cbs);
ಶೂನ್ಯ                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  ಅಕ್ಷರ *                 dec_cur,
                                                  ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *           dec_ಕೊನೆ,
                                                  ಅಕ್ಷರ *                 enc_cur,
                                                  ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *           enc_ಕೊನೆ,
                                                  ಅಕ್ಷರ **                enc_cur_after);
}
