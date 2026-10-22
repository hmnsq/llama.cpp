પ્રકારવ્યાખ્યા ગણના ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_ગણતરી = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_ગણતરી        = 1,
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
    APIR_BACKEND_DISPATCH_TABLE_ગણતરી = 23,
} ApirBackendCommandType;

સ્થિર અંતઃરેખ અચલ અક્ષર * apir_dispatch_command_name(ApirBackendCommandType type) {
    પસંદગી (type) {
        /* device */
        સ્થિતિ APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_ગણતરી:
            પાછા "device_get_device_count";
        સ્થિતિ APIR_COMMAND_TYPE_DEVICE_GET_ગણતરી:
            પાછા "device_get_count";
        સ્થિતિ APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            પાછા "device_get_name";
        સ્થિતિ APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            પાછા "device_get_description";
        સ્થિતિ APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            પાછા "device_get_type";
        સ્થિતિ APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            પાછા "device_get_memory";
        સ્થિતિ APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            પાછા "device_supports_op";
        સ્થિતિ APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            પાછા "device_get_buffer_type";
        સ્થિતિ APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            પાછા "device_get_props";
        સ્થિતિ APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            પાછા "device_buffer_from_ptr";
        /* buffer-type */
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            પાછા "buffer_type_get_name";
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            પાછા "buffer_type_get_alignment";
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            પાછા "buffer_type_get_max_size";
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            પાછા "buffer_type_is_host";
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            પાછા "buffer_type_alloc_buffer";
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            પાછા "buffer_type_get_alloc_size";
        /* buffer */
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            પાછા "buffer_get_base";
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            પાછા "buffer_set_tensor";
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            પાછા "buffer_get_tensor";
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            પાછા "buffer_cpy_tensor";
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_CLEAR:
            પાછા "buffer_clear";
        સ્થિતિ APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            પાછા "buffer_free_buffer";
        /* backend */
        સ્થિતિ APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            પાછા "backend_graph_compute";

        મૂળભૂત:
            પાછા "unknown";
    }
}
