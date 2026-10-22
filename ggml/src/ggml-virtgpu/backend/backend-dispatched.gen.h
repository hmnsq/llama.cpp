#ہدایت once

/* device */
uint32_t backend_device_get_device_گنتی(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_گنتی(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_حافظہ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_بفر_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_بفر_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* buffer-type */
uint32_t backend_بفر_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_بفر_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_بفر_type_get_max_سائز(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_بفر_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_بفر_type_alloc_بفر(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_بفر_type_get_alloc_سائز(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/*بفر*/
uint32_t backend_بفر_get_بنیاد(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_بفر_سیٹ_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_بفر_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_بفر_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_بفر_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_بفر_free_بفر(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* backend */
uint32_t backend_backend_گراف_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

خارجی "C" {
ساکن ثابت backend_dispatch_t apir_backend_dispatch_جدول[APIR_BACKEND_DISPATCH_جدول_گنتی] = {

    /* device */

    /* APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT  = */ backend_device_get_device_گنتی,
    /* APIR_COMMAND_TYPE_DEVICE_GET_COUNT  = */ backend_device_get_گنتی,
    /* APIR_COMMAND_TYPE_DEVICE_GET_NAME  = */ backend_device_get_name,
    /* APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = */ backend_device_get_description,
    /* APIR_COMMAND_TYPE_DEVICE_GET_TYPE  = */ backend_device_get_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_MEMORY  = */ backend_device_get_حافظہ,
    /* APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP  = */ backend_device_supports_op,
    /* APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE  = */ backend_device_get_بفر_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_PROPS  = */ backend_device_get_props,
    /* APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR  = */ backend_device_بفر_from_ptr,

    /* buffer-type */

    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME  = */ backend_بفر_type_get_name,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT  = */ backend_بفر_type_get_alignment,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE  = */ backend_بفر_type_get_max_سائز,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST  = */ backend_بفر_type_is_host /* DEPRECATED */,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER  = */ backend_بفر_type_alloc_بفر,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE  = */ backend_بفر_type_get_alloc_سائز,

    /*بفر*/

    /* APIR_COMMAND_TYPE_BUFFER_GET_BASE  = */ backend_بفر_get_بنیاد,
    /* APIR_COMMAND_TYPE_BUFFER_SET_TENSOR  = */ backend_بفر_سیٹ_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_GET_TENSOR  = */ backend_بفر_get_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR  = */ backend_بفر_cpy_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CLEAR  = */ backend_بفر_clear,
    /* APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER  = */ backend_بفر_free_بفر,

    /* backend */

    /* APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE  = */ backend_backend_گراف_compute,
};
}
