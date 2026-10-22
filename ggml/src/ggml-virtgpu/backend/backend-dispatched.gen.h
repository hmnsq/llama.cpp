#ਹਦਾਇਤ once

/* device */
uint32_t backend_device_get_device_ਗਿਣਤੀ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_ਗਿਣਤੀ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_ਸਮ੍ਰਿਤੀ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_ਬਫਰ_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_ਬਫਰ_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* buffer-type */
uint32_t backend_ਬਫਰ_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ਬਫਰ_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ਬਫਰ_type_get_max_ਮਾਪ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_ਬਫਰ_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ਬਫਰ_type_alloc_ਬਫਰ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ਬਫਰ_type_get_alloc_ਮਾਪ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/*ਬਫਰ*/
uint32_t backend_ਬਫਰ_get_ਅਧਾਰ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ਬਫਰ_ਸੈੱਟ_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ਬਫਰ_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ਬਫਰ_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ਬਫਰ_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_ਬਫਰ_free_ਬਫਰ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* backend */
uint32_t backend_backend_ਗ੍ਰਾਫ_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

ਬਾਹਰੀ "C" {
ਸਥਿਰ ਅਚਲ backend_dispatch_t apir_backend_dispatch_ਸਾਰਣੀ[APIR_BACKEND_DISPATCH_ਸਾਰਣੀ_ਗਿਣਤੀ] = {

    /* device */

    /* APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT  = */ backend_device_get_device_ਗਿਣਤੀ,
    /* APIR_COMMAND_TYPE_DEVICE_GET_COUNT  = */ backend_device_get_ਗਿਣਤੀ,
    /* APIR_COMMAND_TYPE_DEVICE_GET_NAME  = */ backend_device_get_name,
    /* APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = */ backend_device_get_description,
    /* APIR_COMMAND_TYPE_DEVICE_GET_TYPE  = */ backend_device_get_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_MEMORY  = */ backend_device_get_ਸਮ੍ਰਿਤੀ,
    /* APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP  = */ backend_device_supports_op,
    /* APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE  = */ backend_device_get_ਬਫਰ_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_PROPS  = */ backend_device_get_props,
    /* APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR  = */ backend_device_ਬਫਰ_from_ptr,

    /* buffer-type */

    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME  = */ backend_ਬਫਰ_type_get_name,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT  = */ backend_ਬਫਰ_type_get_alignment,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE  = */ backend_ਬਫਰ_type_get_max_ਮਾਪ,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST  = */ backend_ਬਫਰ_type_is_host /* DEPRECATED */,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER  = */ backend_ਬਫਰ_type_alloc_ਬਫਰ,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE  = */ backend_ਬਫਰ_type_get_alloc_ਮਾਪ,

    /*ਬਫਰ*/

    /* APIR_COMMAND_TYPE_BUFFER_GET_BASE  = */ backend_ਬਫਰ_get_ਅਧਾਰ,
    /* APIR_COMMAND_TYPE_BUFFER_SET_TENSOR  = */ backend_ਬਫਰ_ਸੈੱਟ_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_GET_TENSOR  = */ backend_ਬਫਰ_get_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR  = */ backend_ਬਫਰ_cpy_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CLEAR  = */ backend_ਬਫਰ_clear,
    /* APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER  = */ backend_ਬਫਰ_free_ਬਫਰ,

    /* backend */

    /* APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE  = */ backend_backend_ਗ੍ਰਾਫ_compute,
};
}
