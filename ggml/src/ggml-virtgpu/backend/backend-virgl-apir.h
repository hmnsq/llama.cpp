#pragma once

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "shared/api_remoting.h"

#अन्तर्भाव <cstdarg>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstdlib>

बाह्य ggml_backend_reg_t reg;
बाह्य ggml_backend_dev_t dev;
बाह्य ggml_backend_t     bck;

संरचना virgl_apir_callbacks {
    स्थिरांक अक्षर * (*get_config)(uint32_t virgl_ctx_id, स्थिरांक अक्षर * किल्ली);
    शून्य * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

बाह्य "C" {
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, संरचना virgl_apir_callbacks * virgl_cbs);
शून्य                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  अक्षर *                 dec_cur,
                                                  स्थिरांक अक्षर *           dec_end,
                                                  अक्षर *                 enc_cur,
                                                  स्थिरांक अक्षर *           enc_end,
                                                  अक्षर **                enc_cur_after);
}
