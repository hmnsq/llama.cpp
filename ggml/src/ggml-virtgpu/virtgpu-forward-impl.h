#pragma once

// clang-format off
#અન્તર્ભાવ "virtgpu.h"
#અન્તર્ભાવ "ggml-remoting.h"
#અન્તર્ભાવ "backend/shared/apir_backend.h"
#અન્તર્ભાવ "backend/shared/apir_cs_ggml.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
// clang-format on

#define REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                                           \
    int32_t      REMOTE_CALL_PREPARE_forward_flag = (int32_t) apir_command_type__;                                     \
    અચલ અક્ષર * REMOTE_CALL_PREPARE_command_name = apir_dispatch_command_name(apir_command_type__);                   \
    કરો {                                                                                                               \
        encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, REMOTE_CALL_PREPARE_forward_flag); \
        જો (!encoder_name) {                                                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                        \
        }                                                                                                              \
    } જ્યારે (0)

#define REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                     \
    કરો {                                                                                                    \
        ret_name = (ApirForwardReturnCode) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, ખાલી); \
        જો (!decoder_name) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                        \
        }                                                                                                   \
        જો (ret_name < APIR_FORWARD_BASE_INDEX) {                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,            \
                       apir_forward_error(ret_name), ret_name);                                             \
        }                                                                                                   \
        ret_name = (ApirForwardReturnCode) (ret_name - APIR_FORWARD_BASE_INDEX);                            \
        જો (ret_name != 0) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "backend function '%s' failed (return code: %d)",                       \
                       REMOTE_CALL_PREPARE_command_name, ret_name);                                         \
        }                                                                                                   \
    } જ્યારે (0)
