#هدايت once

// clang-format off
#ضمن "virtgpu.h"
#ضمن "ggml-remoting.h"
#ضمن "backend/shared/apir_backend.h"
#ضمن "backend/shared/apir_cs_ggml.h"
#ضمن "ggml-backend-impl.h"
// clang-format on

#وصف_ڪريو REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                                           \
    int32_t      REMOTE_CALL_PREPARE_forward_flag = (int32_t) apir_command_type__;                                     \
    ثابت حرف * REMOTE_CALL_PREPARE_command_name = apir_dispatch_command_name(apir_command_type__);                   \
    افعل {                                                                                                               \
        encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, REMOTE_CALL_PREPARE_forward_flag); \
        إذا (!encoder_name) {                                                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                        \
        }                                                                                                              \
    } طالما (0)

#وصف_ڪريو REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                     \
    افعل {                                                                                                    \
        ret_name = (Apir_Forward_Return_ڪوڊ) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, عدم); \
        إذا (!decoder_name) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                        \
        }                                                                                                   \
        إذا (ret_name < APIR_FORWARD_بنياد_اشاريو) {                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,            \
                       apir_forward_error(ret_name), ret_name);                                             \
        }                                                                                                   \
        ret_name = (Apir_Forward_Return_ڪوڊ) (ret_name - APIR_FORWARD_بنياد_اشاريو);                            \
        إذا (ret_name != 0) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "backend function '%s' failed (return code: %d)",                       \
                       REMOTE_CALL_PREPARE_command_name, ret_name);                                         \
        }                                                                                                   \
    } طالما (0)
