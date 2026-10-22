#సూచన once

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "shared/api_remoting.h"

#అన్తర్భావ <cstdarg>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstdlib>

బాహ్య ggml_backend_reg_t reg;
బాహ్య ggml_backend_dev_t dev;
బాహ్య ggml_backend_t     bck;

నిర్మాణం virgl_apir_callbacks {
    స్థిరరాశి అక్షరం * (*get_config)(uint32_t virgl_ctx_id, స్థిరరాశి అక్షరం * కీలకం);
    శూన్య * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

బాహ్య "C" {
Apir_ఎక్కించు_Library_Return_సంకేతం apir_backend_initialize(uint32_t virgl_ctx_id, నిర్మాణం virgl_apir_callbacks * virgl_cbs);
శూన్య                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  అక్షరం *                 dec_cur,
                                                  స్థిరరాశి అక్షరం *           dec_చివర,
                                                  అక్షరం *                 enc_cur,
                                                  స్థిరరాశి అక్షరం *           enc_చివర,
                                                  అక్షరం **                enc_cur_after);
}
