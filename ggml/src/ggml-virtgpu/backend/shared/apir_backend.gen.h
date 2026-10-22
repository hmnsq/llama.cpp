ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਗਿਣਤੀ ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_ਗਿਣਤੀ = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_ਗਿਣਤੀ        = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME         = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE         = 4,
    APIR_COMMAND_TYPE_DEVICE_GET_ਸਮ੍ਰਿਤੀ       = 5,
    APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP      = 6,
    APIR_COMMAND_TYPE_DEVICE_GET_ਬਫਰ_TYPE  = 7,
    APIR_COMMAND_TYPE_DEVICE_GET_PROPS        = 8,
    APIR_COMMAND_TYPE_DEVICE_ਬਫਰ_FROM_PTR  = 9,

    /* buffer-type */
    APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_NAME       = 10,
    APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_ALIGNMENT  = 11,
    APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_MAX_ਮਾਪ   = 12,
    APIR_COMMAND_TYPE_ਬਫਰ_TYPE_IS_HOST        = 13,
    APIR_COMMAND_TYPE_ਬਫਰ_TYPE_ALLOC_ਬਫਰ   = 14,
    APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_ALLOC_ਮਾਪ = 15,

    /*ਬਫਰ*/
    APIR_COMMAND_TYPE_ਬਫਰ_GET_ਅਧਾਰ    = 16,
    APIR_COMMAND_TYPE_ਬਫਰ_ਸੈੱਟ_TENSOR  = 17,
    APIR_COMMAND_TYPE_ਬਫਰ_GET_TENSOR  = 18,
    APIR_COMMAND_TYPE_ਬਫਰ_CPY_TENSOR  = 19,
    APIR_COMMAND_TYPE_ਬਫਰ_CLEAR       = 20,
    APIR_COMMAND_TYPE_ਬਫਰ_FREE_ਬਫਰ = 21,

    /* backend */
    APIR_COMMAND_TYPE_BACKEND_ਗ੍ਰਾਫ_COMPUTE = 22,

    // last command_type index + 1
    APIR_BACKEND_DISPATCH_ਸਾਰਣੀ_ਗਿਣਤੀ = 23,
} ApirBackendCommandType;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ਅੱਖਰ * apir_dispatch_command_name(ApirBackendCommandType type) {
    ਚੋਣ (type) {
        /* device */
        ਹਾਲਤ APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_ਗਿਣਤੀ:
            ਵਾਪਸ "device_get_device_count";
        ਹਾਲਤ APIR_COMMAND_TYPE_DEVICE_GET_ਗਿਣਤੀ:
            ਵਾਪਸ "device_get_count";
        ਹਾਲਤ APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            ਵਾਪਸ "device_get_name";
        ਹਾਲਤ APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            ਵਾਪਸ "device_get_description";
        ਹਾਲਤ APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            ਵਾਪਸ "device_get_type";
        ਹਾਲਤ APIR_COMMAND_TYPE_DEVICE_GET_ਸਮ੍ਰਿਤੀ:
            ਵਾਪਸ "device_get_memory";
        ਹਾਲਤ APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            ਵਾਪਸ "device_supports_op";
        ਹਾਲਤ APIR_COMMAND_TYPE_DEVICE_GET_ਬਫਰ_TYPE:
            ਵਾਪਸ "device_get_buffer_type";
        ਹਾਲਤ APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            ਵਾਪਸ "device_get_props";
        ਹਾਲਤ APIR_COMMAND_TYPE_DEVICE_ਬਫਰ_FROM_PTR:
            ਵਾਪਸ "device_buffer_from_ptr";
        /* buffer-type */
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_NAME:
            ਵਾਪਸ "buffer_type_get_name";
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_ALIGNMENT:
            ਵਾਪਸ "buffer_type_get_alignment";
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_MAX_ਮਾਪ:
            ਵਾਪਸ "buffer_type_get_max_size";
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_TYPE_IS_HOST:
            ਵਾਪਸ "buffer_type_is_host";
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_TYPE_ALLOC_ਬਫਰ:
            ਵਾਪਸ "buffer_type_alloc_buffer";
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_ALLOC_ਮਾਪ:
            ਵਾਪਸ "buffer_type_get_alloc_size";
        /*ਬਫਰ*/
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_GET_ਅਧਾਰ:
            ਵਾਪਸ "buffer_get_base";
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_ਸੈੱਟ_TENSOR:
            ਵਾਪਸ "buffer_set_tensor";
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_GET_TENSOR:
            ਵਾਪਸ "buffer_get_tensor";
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_CPY_TENSOR:
            ਵਾਪਸ "buffer_cpy_tensor";
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_CLEAR:
            ਵਾਪਸ "buffer_clear";
        ਹਾਲਤ APIR_COMMAND_TYPE_ਬਫਰ_FREE_ਬਫਰ:
            ਵਾਪਸ "buffer_free_buffer";
        /* backend */
        ਹਾਲਤ APIR_COMMAND_TYPE_BACKEND_ਗ੍ਰਾਫ_COMPUTE:
            ਵਾਪਸ "backend_graph_compute";

        ਮੂਲ:
            ਵਾਪਸ "unknown";
    }
}
