#pragma once

#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "shared/api_remoting.h"

#അന്തര്ഭാവ <cstdarg>
#അന്തര്ഭാവ <cstdio>
#അന്തര്ഭാവ <cstdlib>

ബാഹ്യ ggml_backend_reg_t reg;
ബാഹ്യ ggml_backend_dev_t dev;
ബാഹ്യ ggml_backend_t     bck;

ഘടന virgl_apir_callbacks {
    സ്ഥിരം അക്ഷരം * (*get_config)(uint32_t virgl_ctx_id, സ്ഥിരം അക്ഷരം * താക്കോൽ);
    ശൂന്യം * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

ബാഹ്യ "C" {
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, ഘടന virgl_apir_callbacks * virgl_cbs);
ശൂന്യം                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  അക്ഷരം *                 dec_cur,
                                                  സ്ഥിരം അക്ഷരം *           dec_end,
                                                  അക്ഷരം *                 enc_cur,
                                                  സ്ഥിരം അക്ഷരം *           enc_end,
                                                  അക്ഷരം **                enc_cur_after);
}
