প্রকারনির্দেশ গণনা ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_গুনতি = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_গুনতি        = 1,
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
    APIR_BACKEND_DISPATCH_TABLE_গুনতি = 23,
} ApirBackendCommandType;

স্থির অন্তঃস্থ ধ্রুব বর্ণ * apir_dispatch_command_name(ApirBackendCommandType type) {
    বিকল্প (type) {
        /* device */
        পক্ষ APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_গুনতি:
            প্রত্যাগম "device_get_device_count";
        পক্ষ APIR_COMMAND_TYPE_DEVICE_GET_গুনতি:
            প্রত্যাগম "device_get_count";
        পক্ষ APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            প্রত্যাগম "device_get_name";
        পক্ষ APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            প্রত্যাগম "device_get_description";
        পক্ষ APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            প্রত্যাগম "device_get_type";
        পক্ষ APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            প্রত্যাগম "device_get_memory";
        পক্ষ APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            প্রত্যাগম "device_supports_op";
        পক্ষ APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            প্রত্যাগম "device_get_buffer_type";
        পক্ষ APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            প্রত্যাগম "device_get_props";
        পক্ষ APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            প্রত্যাগম "device_buffer_from_ptr";
        /* buffer-type */
        পক্ষ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            প্রত্যাগম "buffer_type_get_name";
        পক্ষ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            প্রত্যাগম "buffer_type_get_alignment";
        পক্ষ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            প্রত্যাগম "buffer_type_get_max_size";
        পক্ষ APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            প্রত্যাগম "buffer_type_is_host";
        পক্ষ APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            প্রত্যাগম "buffer_type_alloc_buffer";
        পক্ষ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            প্রত্যাগম "buffer_type_get_alloc_size";
        /* buffer */
        পক্ষ APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            প্রত্যাগম "buffer_get_base";
        পক্ষ APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            প্রত্যাগম "buffer_set_tensor";
        পক্ষ APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            প্রত্যাগম "buffer_get_tensor";
        পক্ষ APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            প্রত্যাগম "buffer_cpy_tensor";
        পক্ষ APIR_COMMAND_TYPE_BUFFER_CLEAR:
            প্রত্যাগম "buffer_clear";
        পক্ষ APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            প্রত্যাগম "buffer_free_buffer";
        /* backend */
        পক্ষ APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            প্রত্যাগম "backend_graph_compute";

        স্বভাব:
            প্রত্যাগম "unknown";
    }
}
