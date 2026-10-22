تعريفنوع تعداد ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_شمېر = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_شمېر        = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME         = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE         = 4,
    APIR_COMMAND_TYPE_DEVICE_GET_حافظه       = 5,
    APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP      = 6,
    APIR_COMMAND_TYPE_DEVICE_GET_بفر_TYPE  = 7,
    APIR_COMMAND_TYPE_DEVICE_GET_PROPS        = 8,
    APIR_COMMAND_TYPE_DEVICE_بفر_FROM_PTR  = 9,

    /* buffer-type */
    APIR_COMMAND_TYPE_بفر_TYPE_GET_NAME       = 10,
    APIR_COMMAND_TYPE_بفر_TYPE_GET_ALIGNMENT  = 11,
    APIR_COMMAND_TYPE_بفر_TYPE_GET_MAX_اندازه   = 12,
    APIR_COMMAND_TYPE_بفر_TYPE_IS_HOST        = 13,
    APIR_COMMAND_TYPE_بفر_TYPE_ALLOC_بفر   = 14,
    APIR_COMMAND_TYPE_بفر_TYPE_GET_ALLOC_اندازه = 15,

    /*بفر*/
    APIR_COMMAND_TYPE_بفر_GET_بنسټ    = 16,
    APIR_COMMAND_TYPE_بفر_سېټ_TENSOR  = 17,
    APIR_COMMAND_TYPE_بفر_GET_TENSOR  = 18,
    APIR_COMMAND_TYPE_بفر_CPY_TENSOR  = 19,
    APIR_COMMAND_TYPE_بفر_CLEAR       = 20,
    APIR_COMMAND_TYPE_بفر_FREE_بفر = 21,

    /* backend */
    APIR_COMMAND_TYPE_BACKEND_ګراف_COMPUTE = 22,

    // last command_type index + 1
    APIR_BACKEND_DISPATCH_جدول_شمېر = 23,
} ApirBackendCommandType;

ساكن مضمن ثابت حرف * apir_dispatch_command_name(ApirBackendCommandType type) {
    بدل (type) {
        /* device */
        حالة APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_شمېر:
            أرجع "device_get_device_count";
        حالة APIR_COMMAND_TYPE_DEVICE_GET_شمېر:
            أرجع "device_get_count";
        حالة APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            أرجع "device_get_name";
        حالة APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            أرجع "device_get_description";
        حالة APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            أرجع "device_get_type";
        حالة APIR_COMMAND_TYPE_DEVICE_GET_حافظه:
            أرجع "device_get_memory";
        حالة APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            أرجع "device_supports_op";
        حالة APIR_COMMAND_TYPE_DEVICE_GET_بفر_TYPE:
            أرجع "device_get_buffer_type";
        حالة APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            أرجع "device_get_props";
        حالة APIR_COMMAND_TYPE_DEVICE_بفر_FROM_PTR:
            أرجع "device_buffer_from_ptr";
        /* buffer-type */
        حالة APIR_COMMAND_TYPE_بفر_TYPE_GET_NAME:
            أرجع "buffer_type_get_name";
        حالة APIR_COMMAND_TYPE_بفر_TYPE_GET_ALIGNMENT:
            أرجع "buffer_type_get_alignment";
        حالة APIR_COMMAND_TYPE_بفر_TYPE_GET_MAX_اندازه:
            أرجع "buffer_type_get_max_size";
        حالة APIR_COMMAND_TYPE_بفر_TYPE_IS_HOST:
            أرجع "buffer_type_is_host";
        حالة APIR_COMMAND_TYPE_بفر_TYPE_ALLOC_بفر:
            أرجع "buffer_type_alloc_buffer";
        حالة APIR_COMMAND_TYPE_بفر_TYPE_GET_ALLOC_اندازه:
            أرجع "buffer_type_get_alloc_size";
        /*بفر*/
        حالة APIR_COMMAND_TYPE_بفر_GET_بنسټ:
            أرجع "buffer_get_base";
        حالة APIR_COMMAND_TYPE_بفر_سېټ_TENSOR:
            أرجع "buffer_set_tensor";
        حالة APIR_COMMAND_TYPE_بفر_GET_TENSOR:
            أرجع "buffer_get_tensor";
        حالة APIR_COMMAND_TYPE_بفر_CPY_TENSOR:
            أرجع "buffer_cpy_tensor";
        حالة APIR_COMMAND_TYPE_بفر_CLEAR:
            أرجع "buffer_clear";
        حالة APIR_COMMAND_TYPE_بفر_FREE_بفر:
            أرجع "buffer_free_buffer";
        /* backend */
        حالة APIR_COMMAND_TYPE_BACKEND_ګراف_COMPUTE:
            أرجع "backend_graph_compute";

        افتراضي:
            أرجع "unknown";
    }
}
