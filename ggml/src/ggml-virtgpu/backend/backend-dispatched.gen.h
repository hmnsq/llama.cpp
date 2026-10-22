#निर्देशः once

/* device */
uint32_t backend_device_get_device_गणनम्(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_गणनम्(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_स्मृतिः(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_बफर_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_बफर_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* buffer-type */
uint32_t backend_बफर_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_बफर_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_बफर_type_get_max_आकार(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_बफर_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_बफर_type_alloc_बफर(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_बफर_type_get_alloc_आकार(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/*बफर*/
uint32_t backend_बफर_get_आधारः(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_बफर_समुच्चयः_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_बफर_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_बफर_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_बफर_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_बफर_free_बफर(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* backend */
uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

बाह्य "C" {
स्थिर ध्रुव backend_dispatch_t apir_backend_dispatch_सारणी[APIR_BACKEND_DISPATCH_सारणी_गणनम्] = {

    /* device */

    /* APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT  = */ backend_device_get_device_गणनम्,
    /* APIR_COMMAND_TYPE_DEVICE_GET_COUNT  = */ backend_device_get_गणनम्,
    /* APIR_COMMAND_TYPE_DEVICE_GET_NAME  = */ backend_device_get_name,
    /* APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = */ backend_device_get_description,
    /* APIR_COMMAND_TYPE_DEVICE_GET_TYPE  = */ backend_device_get_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_MEMORY  = */ backend_device_get_स्मृतिः,
    /* APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP  = */ backend_device_supports_op,
    /* APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE  = */ backend_device_get_बफर_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_PROPS  = */ backend_device_get_props,
    /* APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR  = */ backend_device_बफर_from_ptr,

    /* buffer-type */

    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME  = */ backend_बफर_type_get_name,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT  = */ backend_बफर_type_get_alignment,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE  = */ backend_बफर_type_get_max_आकार,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST  = */ backend_बफर_type_is_host /* DEPRECATED */,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER  = */ backend_बफर_type_alloc_बफर,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE  = */ backend_बफर_type_get_alloc_आकार,

    /*बफर*/

    /* APIR_COMMAND_TYPE_BUFFER_GET_BASE  = */ backend_बफर_get_आधारः,
    /* APIR_COMMAND_TYPE_BUFFER_SET_TENSOR  = */ backend_बफर_समुच्चयः_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_GET_TENSOR  = */ backend_बफर_get_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR  = */ backend_बफर_cpy_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CLEAR  = */ backend_बफर_clear,
    /* APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER  = */ backend_बफर_free_बफर,

    /* backend */

    /* APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE  = */ backend_backend_graph_compute,
};
}
