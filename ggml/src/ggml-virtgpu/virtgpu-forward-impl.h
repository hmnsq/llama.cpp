#ਹਦਾਇਤ once

// clang-format off
#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-remoting.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "backend/shared/apir_backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "backend/shared/apir_cs_ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
// clang-format on

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                                           \
    int32_t      REMOTE_CALL_PREPARE_forward_flag = (int32_t) apir_command_type__;                                     \
    ਅਚਲ ਅੱਖਰ * REMOTE_CALL_PREPARE_command_name = apir_dispatch_command_name(apir_command_type__);                   \
    ਕਰੋ {                                                                                                               \
        encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, REMOTE_CALL_PREPARE_forward_flag); \
        ਜੇ (!encoder_name) {                                                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                        \
        }                                                                                                              \
    } ਜਦੋਂ (0)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                     \
    ਕਰੋ {                                                                                                    \
        ret_name = (Apir_Forward_Return_ਕੋਡ) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, ਨਿਰਾ); \
        ਜੇ (!decoder_name) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                        \
        }                                                                                                   \
        ਜੇ (ret_name < APIR_FORWARD_ਅਧਾਰ_ਸੂਚਕਾਂਕ) {                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,            \
                       apir_forward_error(ret_name), ret_name);                                             \
        }                                                                                                   \
        ret_name = (Apir_Forward_Return_ਕੋਡ) (ret_name - APIR_FORWARD_ਅਧਾਰ_ਸੂਚਕਾਂਕ);                            \
        ਜੇ (ret_name != 0) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "backend function '%s' failed (return code: %d)",                       \
                       REMOTE_CALL_PREPARE_command_name, ret_name);                                         \
        }                                                                                                   \
    } ਜਦੋਂ (0)
