வகைவரை எண்ணிக்கை ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_மொத்தம் = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_மொத்தம்        = 1,
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
    APIR_BACKEND_DISPATCH_TABLE_மொத்தம் = 23,
} ApirBackendCommandType;

நிலையான உள்வரி மாறிலி எழுத்து * apir_dispatch_command_name(ApirBackendCommandType type) {
    தேர்வு (type) {
        /* device */
        நிலை APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_மொத்தம்:
            திரும்பு "device_get_device_count";
        நிலை APIR_COMMAND_TYPE_DEVICE_GET_மொத்தம்:
            திரும்பு "device_get_count";
        நிலை APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            திரும்பு "device_get_name";
        நிலை APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            திரும்பு "device_get_description";
        நிலை APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            திரும்பு "device_get_type";
        நிலை APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            திரும்பு "device_get_memory";
        நிலை APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            திரும்பு "device_supports_op";
        நிலை APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            திரும்பு "device_get_buffer_type";
        நிலை APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            திரும்பு "device_get_props";
        நிலை APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            திரும்பு "device_buffer_from_ptr";
        /* buffer-type */
        நிலை APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            திரும்பு "buffer_type_get_name";
        நிலை APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            திரும்பு "buffer_type_get_alignment";
        நிலை APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            திரும்பு "buffer_type_get_max_size";
        நிலை APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            திரும்பு "buffer_type_is_host";
        நிலை APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            திரும்பு "buffer_type_alloc_buffer";
        நிலை APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            திரும்பு "buffer_type_get_alloc_size";
        /* buffer */
        நிலை APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            திரும்பு "buffer_get_base";
        நிலை APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            திரும்பு "buffer_set_tensor";
        நிலை APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            திரும்பு "buffer_get_tensor";
        நிலை APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            திரும்பு "buffer_cpy_tensor";
        நிலை APIR_COMMAND_TYPE_BUFFER_CLEAR:
            திரும்பு "buffer_clear";
        நிலை APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            திரும்பு "buffer_free_buffer";
        /* backend */
        நிலை APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            திரும்பு "backend_graph_compute";

        இயல்பு:
            திரும்பு "unknown";
    }
}
