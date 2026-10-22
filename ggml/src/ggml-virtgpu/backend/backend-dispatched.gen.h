#ಸೂಚನೆ once

/* device */
uint32_t backend_device_get_device_ಎಣಿಕೆ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_ಎಣಿಕೆ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_ಸ್ಮರಣೆ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_ಬಫರ್_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_ಬಫರ್_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* buffer-type */
uint32_t backend_ಬಫರ್_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ಬಫರ್_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ಬಫರ್_type_get_max_ಗಾತ್ರ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_ಬಫರ್_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ಬಫರ್_type_alloc_ಬಫರ್(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/*ಬಫರ್*/
uint32_t backend_ಬಫರ್_get_ತಳ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ಬಫರ್_ಗಣ_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ಬಫರ್_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ಬಫರ್_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ಬಫರ್_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ಬಫರ್_free_ಬಫರ್(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* backend */
uint32_t backend_backend_ನಕ್ಷೆ_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

ಬಾಹ್ಯ "C" {
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ backend_dispatch_t apir_backend_dispatch_ಕೋಷ್ಟಕ[APIR_BACKEND_DISPATCH_ಕೋಷ್ಟಕ_ಎಣಿಕೆ] = {

    /* device */

    /* APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT  = */ backend_device_get_device_ಎಣಿಕೆ,
    /* APIR_COMMAND_TYPE_DEVICE_GET_COUNT  = */ backend_device_get_ಎಣಿಕೆ,
    /* APIR_COMMAND_TYPE_DEVICE_GET_NAME  = */ backend_device_get_name,
    /* APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = */ backend_device_get_description,
    /* APIR_COMMAND_TYPE_DEVICE_GET_TYPE  = */ backend_device_get_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_MEMORY  = */ backend_device_get_ಸ್ಮರಣೆ,
    /* APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP  = */ backend_device_supports_op,
    /* APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE  = */ backend_device_get_ಬಫರ್_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_PROPS  = */ backend_device_get_props,
    /* APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR  = */ backend_device_ಬಫರ್_from_ptr,

    /* buffer-type */

    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME  = */ backend_ಬಫರ್_type_get_name,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT  = */ backend_ಬಫರ್_type_get_alignment,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE  = */ backend_ಬಫರ್_type_get_max_ಗಾತ್ರ,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST  = */ backend_ಬಫರ್_type_is_host /* DEPRECATED */,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER  = */ backend_ಬಫರ್_type_alloc_ಬಫರ್,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE  = */ backend_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,

    /*ಬಫರ್*/

    /* APIR_COMMAND_TYPE_BUFFER_GET_BASE  = */ backend_ಬಫರ್_get_ತಳ,
    /* APIR_COMMAND_TYPE_BUFFER_SET_TENSOR  = */ backend_ಬಫರ್_ಗಣ_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_GET_TENSOR  = */ backend_ಬಫರ್_get_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR  = */ backend_ಬಫರ್_cpy_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CLEAR  = */ backend_ಬಫರ್_clear,
    /* APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER  = */ backend_ಬಫರ್_free_ಬಫರ್,

    /* backend */

    /* APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE  = */ backend_backend_ನಕ್ಷೆ_compute,
};
}
