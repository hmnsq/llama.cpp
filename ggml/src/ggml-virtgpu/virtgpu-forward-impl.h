#pragma once

// clang-format off
#شامل "virtgpu.h"
#شامل "ggml-remoting.h"
#شامل "backend/shared/apir_backend.h"
#شامل "backend/shared/apir_cs_ggml.h"
#شامل "ggml-backend-impl.h"
// clang-format on

#define REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                                           \
    int32_t      REMOTE_CALL_PREPARE_forward_flag = (int32_t) apir_command_type__;                                     \
    ثابت نویسه * REMOTE_CALL_PREPARE_command_name = apir_dispatch_command_name(apir_command_type__);                   \
    انجامبده {                                                                                                               \
        encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, REMOTE_CALL_PREPARE_forward_flag); \
        اگر (!encoder_name) {                                                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                        \
        }                                                                                                              \
    } وقتی (0)

#define REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                     \
    انجامبده {                                                                                                    \
        ret_name = (ApirForwardReturnCode) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, پوچ); \
        اگر (!decoder_name) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                        \
        }                                                                                                   \
        اگر (ret_name < APIR_FORWARD_BASE_INDEX) {                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,            \
                       apir_forward_error(ret_name), ret_name);                                             \
        }                                                                                                   \
        ret_name = (ApirForwardReturnCode) (ret_name - APIR_FORWARD_BASE_INDEX);                            \
        اگر (ret_name != 0) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "backend function '%s' failed (return code: %d)",                       \
                       REMOTE_CALL_PREPARE_command_name, ret_name);                                         \
        }                                                                                                   \
    } وقتی (0)
