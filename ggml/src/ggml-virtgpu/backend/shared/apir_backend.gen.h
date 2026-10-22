തരനിർവചനം എണ്ണൽ ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_എണ്ണം = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_എണ്ണം        = 1,
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
    APIR_BACKEND_DISPATCH_TABLE_എണ്ണം = 23,
} ApirBackendCommandType;

സ്ഥിതം വരിയിൽ സ്ഥിരം അക്ഷരം * apir_dispatch_command_name(ApirBackendCommandType type) {
    തിരഞ്ഞെടുപ്പ് (type) {
        /* device */
        അവസ്ഥ APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_എണ്ണം:
            തിരികെ "device_get_device_count";
        അവസ്ഥ APIR_COMMAND_TYPE_DEVICE_GET_എണ്ണം:
            തിരികെ "device_get_count";
        അവസ്ഥ APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            തിരികെ "device_get_name";
        അവസ്ഥ APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            തിരികെ "device_get_description";
        അവസ്ഥ APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            തിരികെ "device_get_type";
        അവസ്ഥ APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            തിരികെ "device_get_memory";
        അവസ്ഥ APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            തിരികെ "device_supports_op";
        അവസ്ഥ APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            തിരികെ "device_get_buffer_type";
        അവസ്ഥ APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            തിരികെ "device_get_props";
        അവസ്ഥ APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            തിരികെ "device_buffer_from_ptr";
        /* buffer-type */
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            തിരികെ "buffer_type_get_name";
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            തിരികെ "buffer_type_get_alignment";
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            തിരികെ "buffer_type_get_max_size";
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            തിരികെ "buffer_type_is_host";
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            തിരികെ "buffer_type_alloc_buffer";
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            തിരികെ "buffer_type_get_alloc_size";
        /* buffer */
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            തിരികെ "buffer_get_base";
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            തിരികെ "buffer_set_tensor";
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            തിരികെ "buffer_get_tensor";
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            തിരികെ "buffer_cpy_tensor";
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_CLEAR:
            തിരികെ "buffer_clear";
        അവസ്ഥ APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            തിരികെ "buffer_free_buffer";
        /* backend */
        അവസ്ഥ APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            തിരികെ "backend_graph_compute";

        സ്ഥിരസ്ഥിതി:
            തിരികെ "unknown";
    }
}
