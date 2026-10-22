تعریفنوع شمارش ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_شمار = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_شمار        = 1,
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
    APIR_BACKEND_DISPATCH_TABLE_شمار = 23,
} ApirBackendCommandType;

ایستا درونخطی ثابت نویسه * apir_dispatch_command_name(ApirBackendCommandType type) {
    انتخاب (type) {
        /* device */
        حالت APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_شمار:
            بازگشت "device_get_device_count";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_شمار:
            بازگشت "device_get_count";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            بازگشت "device_get_name";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            بازگشت "device_get_description";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            بازگشت "device_get_type";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            بازگشت "device_get_memory";
        حالت APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            بازگشت "device_supports_op";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            بازگشت "device_get_buffer_type";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            بازگشت "device_get_props";
        حالت APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            بازگشت "device_buffer_from_ptr";
        /* buffer-type */
        حالت APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            بازگشت "buffer_type_get_name";
        حالت APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            بازگشت "buffer_type_get_alignment";
        حالت APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            بازگشت "buffer_type_get_max_size";
        حالت APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            بازگشت "buffer_type_is_host";
        حالت APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            بازگشت "buffer_type_alloc_buffer";
        حالت APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            بازگشت "buffer_type_get_alloc_size";
        /* buffer */
        حالت APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            بازگشت "buffer_get_base";
        حالت APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            بازگشت "buffer_set_tensor";
        حالت APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            بازگشت "buffer_get_tensor";
        حالت APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            بازگشت "buffer_cpy_tensor";
        حالت APIR_COMMAND_TYPE_BUFFER_CLEAR:
            بازگشت "buffer_clear";
        حالت APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            بازگشت "buffer_free_buffer";
        /* backend */
        حالت APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            بازگشت "backend_graph_compute";

        پیشفرض:
            بازگشت "unknown";
    }
}
