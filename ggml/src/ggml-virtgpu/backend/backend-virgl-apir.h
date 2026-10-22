#pragma once

#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "shared/api_remoting.h"

#અન્તર્ભાવ <cstdarg>
#અન્તર્ભાવ <cstdio>
#અન્તર્ભાવ <cstdlib>

બાહ્ય ggml_backend_reg_t reg;
બાહ્ય ggml_backend_dev_t dev;
બાહ્ય ggml_backend_t     bck;

સંરચના virgl_apir_callbacks {
    અચલ અક્ષર * (*get_config)(uint32_t virgl_ctx_id, અચલ અક્ષર * ચાવી);
    શૂન્ય * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

બાહ્ય "C" {
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, સંરચના virgl_apir_callbacks * virgl_cbs);
શૂન્ય                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  અક્ષર *                 dec_cur,
                                                  અચલ અક્ષર *           dec_end,
                                                  અક્ષર *                 enc_cur,
                                                  અચલ અક્ષર *           enc_end,
                                                  અક્ષર **                enc_cur_after);
}
