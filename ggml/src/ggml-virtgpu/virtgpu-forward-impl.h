#ಸೂಚನೆ once

// clang-format off
#ಅನ್ತರ್ಭಾವ "virtgpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"
#ಅನ್ತರ್ಭಾವ "backend/shared/apir_backend.h"
#ಅನ್ತರ್ಭಾವ "backend/shared/apir_cs_ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
// clang-format on

#ವ್ಯಾಖ್ಯಾನಿಸು REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                                           \
    int32_t      REMOTE_CALL_PREPARE_forward_flag = (int32_t) apir_command_type__;                                     \
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * REMOTE_CALL_PREPARE_command_name = apir_dispatch_command_name(apir_command_type__);                   \
    ಮಾಡು {                                                                                                               \
        encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, REMOTE_CALL_PREPARE_forward_flag); \
        ಆದರೆ (!encoder_name) {                                                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                        \
        }                                                                                                              \
    } ಆಗ (0)

#ವ್ಯಾಖ್ಯಾನಿಸು REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                     \
    ಮಾಡು {                                                                                                    \
        ret_name = (Apir_Forward_Return_ಸಂಕೇತ) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, ಖಾಲಿ); \
        ಆದರೆ (!decoder_name) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                        \
        }                                                                                                   \
        ಆದರೆ (ret_name < APIR_FORWARD_ತಳ_ಸೂಚ್ಯಂಕ) {                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,            \
                       apir_forward_error(ret_name), ret_name);                                             \
        }                                                                                                   \
        ret_name = (Apir_Forward_Return_ಸಂಕೇತ) (ret_name - APIR_FORWARD_ತಳ_ಸೂಚ್ಯಂಕ);                            \
        ಆದರೆ (ret_name != 0) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "backend function '%s' failed (return code: %d)",                       \
                       REMOTE_CALL_PREPARE_command_name, ret_name);                                         \
        }                                                                                                   \
    } ಆಗ (0)
