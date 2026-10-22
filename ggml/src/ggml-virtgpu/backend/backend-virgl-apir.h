#pragma once

#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "shared/api_remoting.h"

#அந்தர்பாவ <cstdarg>
#அந்தர்பாவ <cstdio>
#அந்தர்பாவ <cstdlib>

வெளி ggml_backend_reg_t reg;
வெளி ggml_backend_dev_t dev;
வெளி ggml_backend_t     bck;

கட்டமைப்பு virgl_apir_callbacks {
    மாறிலி எழுத்து * (*get_config)(uint32_t virgl_ctx_id, மாறிலி எழுத்து * சாவி);
    வெறுமை * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

வெளி "C" {
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, கட்டமைப்பு virgl_apir_callbacks * virgl_cbs);
வெறுமை                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  எழுத்து *                 dec_cur,
                                                  மாறிலி எழுத்து *           dec_end,
                                                  எழுத்து *                 enc_cur,
                                                  மாறிலி எழுத்து *           enc_end,
                                                  எழுத்து **                enc_cur_after);
}
