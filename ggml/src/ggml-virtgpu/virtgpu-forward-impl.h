#ہدایت once

// clang-format off
#شامل "virtgpu.h"
#شامل "ggml-remoting.h"
#شامل "backend/shared/apir_backend.h"
#شامل "backend/shared/apir_cs_ggml.h"
#شامل "ggml-backend-impl.h"
// clang-format on

#تعریف_کرو REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                                           \
    int32_t      REMOTE_CALL_PREPARE_forward_flag = (int32_t) apir_command_type__;                                     \
    ثابت حرف * REMOTE_CALL_PREPARE_command_name = apir_dispatch_command_name(apir_command_type__);                   \
    کرو {                                                                                                               \
        encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, REMOTE_CALL_PREPARE_forward_flag); \
        اگر (!encoder_name) {                                                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                        \
        }                                                                                                              \
    } جب (0)

#تعریف_کرو REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                     \
    کرو {                                                                                                    \
        ret_name = (Apir_Forward_Return_رمز) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, تہی); \
        اگر (!decoder_name) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                        \
        }                                                                                                   \
        اگر (ret_name < APIR_FORWARD_بنیاد_اشاریہ) {                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,            \
                       apir_forward_error(ret_name), ret_name);                                             \
        }                                                                                                   \
        ret_name = (Apir_Forward_Return_رمز) (ret_name - APIR_FORWARD_بنیاد_اشاریہ);                            \
        اگر (ret_name != 0) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "backend function '%s' failed (return code: %d)",                       \
                       REMOTE_CALL_PREPARE_command_name, ret_name);                                         \
        }                                                                                                   \
    } جب (0)
