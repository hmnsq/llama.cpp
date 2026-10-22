#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "shared/api_remoting.h"

#ଅନ୍ତର୍ଭାଵ <cstdarg>
#ଅନ୍ତର୍ଭାଵ <cstdio>
#ଅନ୍ତର୍ଭାଵ <cstdlib>

ବାହ୍ୟ ggml_backend_reg_t reg;
ବାହ୍ୟ ggml_backend_dev_t dev;
ବାହ୍ୟ ggml_backend_t     bck;

ସଂରଚନା virgl_apir_callbacks {
    ଅଚଳ ଅକ୍ଷର * (*get_config)(uint32_t virgl_ctx_id, ଅଚଳ ଅକ୍ଷର * ଚାବି);
    ଶୂନ୍ୟ * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

ବାହ୍ୟ "C" {
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, ସଂରଚନା virgl_apir_callbacks * virgl_cbs);
ଶୂନ୍ୟ                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  ଅକ୍ଷର *                 dec_cur,
                                                  ଅଚଳ ଅକ୍ଷର *           dec_end,
                                                  ଅକ୍ଷର *                 enc_cur,
                                                  ଅଚଳ ଅକ୍ଷର *           enc_end,
                                                  ଅକ୍ଷର **                enc_cur_after);
}
