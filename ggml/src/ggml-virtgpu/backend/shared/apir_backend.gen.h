ପ୍ରକାରବ୍ୟାଖ୍ୟା ଗଣନା ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_ଗଣନା = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_ଗଣନା        = 1,
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
    APIR_BACKEND_DISPATCH_TABLE_ଗଣନା = 23,
} ApirBackendCommandType;

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ ଅକ୍ଷର * apir_dispatch_command_name(ApirBackendCommandType type) {
    ବାଛ (type) {
        /* device */
        ସ୍ଥିତି APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_ଗଣନା:
            ଫେରନ୍ତୁ "device_get_device_count";
        ସ୍ଥିତି APIR_COMMAND_TYPE_DEVICE_GET_ଗଣନା:
            ଫେରନ୍ତୁ "device_get_count";
        ସ୍ଥିତି APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            ଫେରନ୍ତୁ "device_get_name";
        ସ୍ଥିତି APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            ଫେରନ୍ତୁ "device_get_description";
        ସ୍ଥିତି APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            ଫେରନ୍ତୁ "device_get_type";
        ସ୍ଥିତି APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            ଫେରନ୍ତୁ "device_get_memory";
        ସ୍ଥିତି APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            ଫେରନ୍ତୁ "device_supports_op";
        ସ୍ଥିତି APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            ଫେରନ୍ତୁ "device_get_buffer_type";
        ସ୍ଥିତି APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            ଫେରନ୍ତୁ "device_get_props";
        ସ୍ଥିତି APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            ଫେରନ୍ତୁ "device_buffer_from_ptr";
        /* buffer-type */
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            ଫେରନ୍ତୁ "buffer_type_get_name";
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            ଫେରନ୍ତୁ "buffer_type_get_alignment";
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            ଫେରନ୍ତୁ "buffer_type_get_max_size";
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            ଫେରନ୍ତୁ "buffer_type_is_host";
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            ଫେରନ୍ତୁ "buffer_type_alloc_buffer";
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            ଫେରନ୍ତୁ "buffer_type_get_alloc_size";
        /* buffer */
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            ଫେରନ୍ତୁ "buffer_get_base";
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            ଫେରନ୍ତୁ "buffer_set_tensor";
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            ଫେରନ୍ତୁ "buffer_get_tensor";
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            ଫେରନ୍ତୁ "buffer_cpy_tensor";
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_CLEAR:
            ଫେରନ୍ତୁ "buffer_clear";
        ସ୍ଥିତି APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            ଫେରନ୍ତୁ "buffer_free_buffer";
        /* backend */
        ସ୍ଥିତି APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            ଫେରନ୍ତୁ "backend_graph_compute";

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ "unknown";
    }
}
