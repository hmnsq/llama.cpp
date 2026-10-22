#आशय once

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"
#समावेश "shared/api_remoting.h"

#समावेश <cstdarg>
#समावेश <cstdio>
#समावेश <cstdlib>

बाह्य ggml_backend_reg_t reg;
बाह्य ggml_backend_dev_t dev;
बाह्य ggml_backend_t     bck;

काष्ठा virgl_apir_callbacks {
    स्थिर अक्षर * (*get_config)(uint32_t virgl_ctx_id, स्थिर अक्षर * कुंजी);
    व्योम * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

बाह्य "C" {
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, काष्ठा virgl_apir_callbacks * virgl_cbs);
व्योम                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  अक्षर *                 dec_cur,
                                                  स्थिर अक्षर *           dec_end,
                                                  अक्षर *                 enc_cur,
                                                  स्थिर अक्षर *           enc_end,
                                                  अक्षर **                enc_cur_after);
}
