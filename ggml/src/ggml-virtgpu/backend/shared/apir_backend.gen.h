typedef enum ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_COUNT        = 1,
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
    APIR_BACKEND_DISPATCH_TABLE_COUNT = 23,
} ApirBackendCommandType;

static inline const char * apir_dispatch_command_name(ApirBackendCommandType type) {
    switch (type) {
        /* device */
        case APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT:
            вернуть "device_get_device_count";
        case APIR_COMMAND_TYPE_DEVICE_GET_COUNT:
            вернуть "device_get_count";
        case APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            вернуть "device_get_name";
        case APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            вернуть "device_get_description";
        case APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            вернуть "device_get_type";
        case APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            вернуть "device_get_memory";
        case APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            вернуть "device_supports_op";
        case APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            вернуть "device_get_buffer_type";
        case APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            вернуть "device_get_props";
        case APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            вернуть "device_buffer_from_ptr";
        /* buffer-type */
        case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            вернуть "buffer_type_get_name";
        case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            вернуть "buffer_type_get_alignment";
        case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            вернуть "buffer_type_get_max_size";
        case APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            вернуть "buffer_type_is_host";
        case APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            вернуть "buffer_type_alloc_buffer";
        case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            вернуть "buffer_type_get_alloc_size";
        /* buffer */
        case APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            вернуть "buffer_get_base";
        case APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            вернуть "buffer_set_tensor";
        case APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            вернуть "buffer_get_tensor";
        case APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            вернуть "buffer_cpy_tensor";
        case APIR_COMMAND_TYPE_BUFFER_CLEAR:
            вернуть "buffer_clear";
        case APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            вернуть "buffer_free_buffer";
        /* backend */
        case APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            вернуть "backend_graph_compute";

        default:
            вернуть "unknown";
    }
}
