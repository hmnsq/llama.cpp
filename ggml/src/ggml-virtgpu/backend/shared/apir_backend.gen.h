প্রকারলেখ গণনা ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_গুনতি = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_গুনতি        = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME         = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE         = 4,
    APIR_COMMAND_TYPE_DEVICE_GET_MEMORY       = 5,
    APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP      = 6,
    APIR_COMMAND_TYPE_DEVICE_GET_বাফার_TYPE  = 7,
    APIR_COMMAND_TYPE_DEVICE_GET_PROPS        = 8,
    APIR_COMMAND_TYPE_DEVICE_বাফার_FROM_PTR  = 9,

    /* buffer-type */
    APIR_COMMAND_TYPE_বাফার_TYPE_GET_NAME       = 10,
    APIR_COMMAND_TYPE_বাফার_TYPE_GET_ALIGNMENT  = 11,
    APIR_COMMAND_TYPE_বাফার_TYPE_GET_MAX_আকার   = 12,
    APIR_COMMAND_TYPE_বাফার_TYPE_IS_HOST        = 13,
    APIR_COMMAND_TYPE_বাফার_TYPE_ALLOC_বাফার   = 14,
    APIR_COMMAND_TYPE_বাফার_TYPE_GET_ALLOC_আকার = 15,

    /*বাফার*/
    APIR_COMMAND_TYPE_বাফার_GET_BASE    = 16,
    APIR_COMMAND_TYPE_বাফার_SET_TENSOR  = 17,
    APIR_COMMAND_TYPE_বাফার_GET_TENSOR  = 18,
    APIR_COMMAND_TYPE_বাফার_CPY_TENSOR  = 19,
    APIR_COMMAND_TYPE_বাফার_CLEAR       = 20,
    APIR_COMMAND_TYPE_বাফার_FREE_বাফার = 21,

    /* backend */
    APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE = 22,

    // last command_type index + 1
    APIR_BACKEND_DISPATCH_TABLE_গুনতি = 23,
} ApirBackendCommandType;

স্থির অন্তঃরেখ ধ্রুবক অক্ষর * apir_dispatch_command_name(ApirBackendCommandType type) {
    নির্বাচন (type) {
        /* device */
        ক্ষেত্র APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_গুনতি:
            ফেরত "device_get_device_count";
        ক্ষেত্র APIR_COMMAND_TYPE_DEVICE_GET_গুনতি:
            ফেরত "device_get_count";
        ক্ষেত্র APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            ফেরত "device_get_name";
        ক্ষেত্র APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            ফেরত "device_get_description";
        ক্ষেত্র APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            ফেরত "device_get_type";
        ক্ষেত্র APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            ফেরত "device_get_memory";
        ক্ষেত্র APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            ফেরত "device_supports_op";
        ক্ষেত্র APIR_COMMAND_TYPE_DEVICE_GET_বাফার_TYPE:
            ফেরত "device_get_buffer_type";
        ক্ষেত্র APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            ফেরত "device_get_props";
        ক্ষেত্র APIR_COMMAND_TYPE_DEVICE_বাফার_FROM_PTR:
            ফেরত "device_buffer_from_ptr";
        /* buffer-type */
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_TYPE_GET_NAME:
            ফেরত "buffer_type_get_name";
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_TYPE_GET_ALIGNMENT:
            ফেরত "buffer_type_get_alignment";
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_TYPE_GET_MAX_আকার:
            ফেরত "buffer_type_get_max_size";
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_TYPE_IS_HOST:
            ফেরত "buffer_type_is_host";
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_TYPE_ALLOC_বাফার:
            ফেরত "buffer_type_alloc_buffer";
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_TYPE_GET_ALLOC_আকার:
            ফেরত "buffer_type_get_alloc_size";
        /*বাফার*/
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_GET_BASE:
            ফেরত "buffer_get_base";
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_SET_TENSOR:
            ফেরত "buffer_set_tensor";
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_GET_TENSOR:
            ফেরত "buffer_get_tensor";
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_CPY_TENSOR:
            ফেরত "buffer_cpy_tensor";
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_CLEAR:
            ফেরত "buffer_clear";
        ক্ষেত্র APIR_COMMAND_TYPE_বাফার_FREE_বাফার:
            ফেরত "buffer_free_buffer";
        /* backend */
        ক্ষেত্র APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            ফেরত "backend_graph_compute";

        পূর্বনির্ধারিত:
            ফেরত "unknown";
    }
}
