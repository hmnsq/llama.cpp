#pragma once

// clang-format off
#अन्तर्भाव "virtgpu.h"
#अन्तर्भाव "ggml-remoting.h"
#अन्तर्भाव "backend/shared/apir_backend.h"
#अन्तर्भाव "backend/shared/apir_cs_ggml.h"
#अन्तर्भाव "ggml-backend-impl.h"
// clang-format on

#define REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                                           \
    int32_t      REMOTE_CALL_PREPARE_forward_flag = (int32_t) apir_command_type__;                                     \
    ध्रुव वर्ण * REMOTE_CALL_PREPARE_command_name = apir_dispatch_command_name(apir_command_type__);                   \
    करेहि {                                                                                                               \
        encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, REMOTE_CALL_PREPARE_forward_flag); \
        जइ (!encoder_name) {                                                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                        \
        }                                                                                                              \
    } जाव (0)

#define REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                     \
    करेहि {                                                                                                    \
        ret_name = (ApirForwardReturnCode) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, रिक्त); \
        जइ (!decoder_name) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                        \
        }                                                                                                   \
        जइ (ret_name < APIR_FORWARD_BASE_INDEX) {                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,            \
                       apir_forward_error(ret_name), ret_name);                                             \
        }                                                                                                   \
        ret_name = (ApirForwardReturnCode) (ret_name - APIR_FORWARD_BASE_INDEX);                            \
        जइ (ret_name != 0) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "backend function '%s' failed (return code: %d)",                       \
                       REMOTE_CALL_PREPARE_command_name, ret_name);                                         \
        }                                                                                                   \
    } जाव (0)
