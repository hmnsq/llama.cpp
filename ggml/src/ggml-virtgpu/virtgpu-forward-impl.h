#निर्देशः once

// clang-format off
#अन्तर्भाव "virtgpu.h"
#अन्तर्भाव "ggml-remoting.h"
#अन्तर्भाव "backend/shared/apir_backend.h"
#अन्तर्भाव "backend/shared/apir_cs_ggml.h"
#अन्तर्भाव "ggml-backend-impl.h"
// clang-format on

#परिभाषय REMOTE_CALL_PREPARE(gpu_dev_name, encoder_name, apir_command_type__)                                           \
    int32_t      REMOTE_CALL_PREPARE_forward_flag = (int32_t) apir_command_type__;                                     \
    ध्रुव वर्ण * REMOTE_CALL_PREPARE_command_name = apir_dispatch_command_name(apir_command_type__);                   \
    कुरु {                                                                                                               \
        encoder_name = remote_call_prepare(gpu_dev_name, APIR_COMMAND_TYPE_FORWARD, REMOTE_CALL_PREPARE_forward_flag); \
        यदि (!encoder_name) {                                                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);                        \
        }                                                                                                              \
    } यावत् (0)

#परिभाषय REMOTE_CALL(gpu_dev_name, encoder_name, decoder_name, ret_name)                                     \
    कुरु {                                                                                                    \
        ret_name = (Apir_Forward_Return_संहिता) remote_call(gpu_dev_name, encoder_name, &decoder_name, 0, रिक्त); \
        यदि (!decoder_name) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to kick the remote call", __func__);                        \
        }                                                                                                   \
        यदि (ret_name < APIR_FORWARD_आधारः_सूचकांक) {                                                           \
            GGML_ABORT(GGML_VIRTGPU "%s: failed to forward the API call: %s: code %d", __func__,            \
                       apir_forward_error(ret_name), ret_name);                                             \
        }                                                                                                   \
        ret_name = (Apir_Forward_Return_संहिता) (ret_name - APIR_FORWARD_आधारः_सूचकांक);                            \
        यदि (ret_name != 0) {                                                                                \
            GGML_ABORT(GGML_VIRTGPU "backend function '%s' failed (return code: %d)",                       \
                       REMOTE_CALL_PREPARE_command_name, ret_name);                                         \
        }                                                                                                   \
    } यावत् (0)
