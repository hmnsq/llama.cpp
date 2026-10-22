प्रकारव्याख्या गणना ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_मोजणी = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_मोजणी        = 1,
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
    APIR_BACKEND_DISPATCH_TABLE_मोजणी = 23,
} ApirBackendCommandType;

स्थिर ओळीत स्थिरांक अक्षर * apir_dispatch_command_name(ApirBackendCommandType type) {
    निवड (type) {
        /* device */
        स्थिती APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_मोजणी:
            परत "device_get_device_count";
        स्थिती APIR_COMMAND_TYPE_DEVICE_GET_मोजणी:
            परत "device_get_count";
        स्थिती APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            परत "device_get_name";
        स्थिती APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            परत "device_get_description";
        स्थिती APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            परत "device_get_type";
        स्थिती APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            परत "device_get_memory";
        स्थिती APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            परत "device_supports_op";
        स्थिती APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            परत "device_get_buffer_type";
        स्थिती APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            परत "device_get_props";
        स्थिती APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            परत "device_buffer_from_ptr";
        /* buffer-type */
        स्थिती APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            परत "buffer_type_get_name";
        स्थिती APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            परत "buffer_type_get_alignment";
        स्थिती APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            परत "buffer_type_get_max_size";
        स्थिती APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            परत "buffer_type_is_host";
        स्थिती APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            परत "buffer_type_alloc_buffer";
        स्थिती APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            परत "buffer_type_get_alloc_size";
        /* buffer */
        स्थिती APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            परत "buffer_get_base";
        स्थिती APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            परत "buffer_set_tensor";
        स्थिती APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            परत "buffer_get_tensor";
        स्थिती APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            परत "buffer_cpy_tensor";
        स्थिती APIR_COMMAND_TYPE_BUFFER_CLEAR:
            परत "buffer_clear";
        स्थिती APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            परत "buffer_free_buffer";
        /* backend */
        स्थिती APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            परत "backend_graph_compute";

        पूर्वनिर्धारित:
            परत "unknown";
    }
}
