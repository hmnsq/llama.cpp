#సూచన once

// clang-format off
#అన్తర్భావ "virtgpu.h"
#అన్తర్భావ "ggml-remoting.h"
#అన్తర్భావ "backend/shared/apir_backend.h"
#అన్తర్భావ "backend/shared/apir_cs_ggml.h"
#అన్తర్భావ "ggml-backend-impl.h"
// clang-format on

#నిర్వచించు REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                                           \
    int32_t      REMOTE_CALL_PREPARE_forward_flag = (int32_t) apir_command_type__;                                     \
    స్థిరరాశి అక్షరం * REMOTE_CALL_PREPARE_command_name = apir_dispatch_command_name(apir_command_type__);                   \
    చేయి {                                                                                                               \
        encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, REMOTE_CALL_PREPARE_forward_flag); \
        ఒకవేళ (!encoder_name) {                                                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                        \
        }                                                                                                              \
    } అయితే (0)

#నిర్వచించు REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                     \
    చేయి {                                                                                                    \
        ret_name = (Apir_Forward_Return_సంకేతం) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, ఖాళీ); \
        ఒకవేళ (!decoder_name) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                        \
        }                                                                                                   \
        ఒకవేళ (ret_name < APIR_FORWARD_పునాది_సూచీ) {                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,            \
                       apir_forward_error(ret_name), ret_name);                                             \
        }                                                                                                   \
        ret_name = (Apir_Forward_Return_సంకేతం) (ret_name - APIR_FORWARD_పునాది_సూచీ);                            \
        ఒకవేళ (ret_name != 0) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "backend function '%s' failed (return code: %d)",                       \
                       REMOTE_CALL_PREPARE_command_name, ret_name);                                         \
        }                                                                                                   \
    } అయితే (0)
