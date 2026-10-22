प्रकारनिर्देश गणना ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_गणनम् = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_गणनम्        = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME         = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE         = 4,
    APIR_COMMAND_TYPE_DEVICE_GET_MEMORY       = 5,
    APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP      = 6,
    APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE  = 7,
    APIR_COMMAND_TYPE_DEVICE_GET_PROPS        = 8,
    APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR  = 9,

    /* buffer-type */
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME       = 10,
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT  = 11,
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE   = 12,
    APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST        = 13,
    APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER   = 14,
    APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE = 15,

    /* buffer */
    APIR_COMMAND_TYPE_BUFFER_GET_BASE    = 16,
    APIR_COMMAND_TYPE_BUFFER_SET_TENSOR  = 17,
    APIR_COMMAND_TYPE_BUFFER_GET_TENSOR  = 18,
    APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR  = 19,
    APIR_COMMAND_TYPE_BUFFER_CLEAR       = 20,
    APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER = 21,

    /* backend */
    APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE = 22,

    // last command_type index + 1
    APIR_BACKEND_DISPATCH_TABLE_गणनम् = 23,
} ApirBackendCommandType;

स्थिर अन्तःस्थ ध्रुव वर्ण * apir_dispatch_command_name(ApirBackendCommandType type) {
    विकल्प (type) {
        /* device */
        पक्ष APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_गणनम्:
            पटिनिवत्त "device_get_device_count";
        पक्ष APIR_COMMAND_TYPE_DEVICE_GET_गणनम्:
            पटिनिवत्त "device_get_count";
        पक्ष APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            पटिनिवत्त "device_get_name";
        पक्ष APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            पटिनिवत्त "device_get_description";
        पक्ष APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            पटिनिवत्त "device_get_type";
        पक्ष APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            पटिनिवत्त "device_get_memory";
        पक्ष APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            पटिनिवत्त "device_supports_op";
        पक्ष APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            पटिनिवत्त "device_get_buffer_type";
        पक्ष APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            पटिनिवत्त "device_get_props";
        पक्ष APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            पटिनिवत्त "device_buffer_from_ptr";
        /* buffer-type */
        पक्ष APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            पटिनिवत्त "buffer_type_get_name";
        पक्ष APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            पटिनिवत्त "buffer_type_get_alignment";
        पक्ष APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            पटिनिवत्त "buffer_type_get_max_size";
        पक्ष APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            पटिनिवत्त "buffer_type_is_host";
        पक्ष APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            पटिनिवत्त "buffer_type_alloc_buffer";
        पक्ष APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            पटिनिवत्त "buffer_type_get_alloc_size";
        /* buffer */
        पक्ष APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            पटिनिवत्त "buffer_get_base";
        पक्ष APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            पटिनिवत्त "buffer_set_tensor";
        पक्ष APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            पटिनिवत्त "buffer_get_tensor";
        पक्ष APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            पटिनिवत्त "buffer_cpy_tensor";
        पक्ष APIR_COMMAND_TYPE_BUFFER_CLEAR:
            पटिनिवत्त "buffer_clear";
        पक्ष APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            पटिनिवत्त "buffer_free_buffer";
        /* backend */
        पक्ष APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            पटिनिवत्त "backend_graph_compute";

        स्वभाव:
            पटिनिवत्त "unknown";
    }
}
