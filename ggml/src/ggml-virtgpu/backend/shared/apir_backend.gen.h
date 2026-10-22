قسمتعریف شمار ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_گنتی = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_گنتی        = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME         = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE         = 4,
    APIR_COMMAND_TYPE_DEVICE_GET_حافظہ       = 5,
    APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP      = 6,
    APIR_COMMAND_TYPE_DEVICE_GET_بفر_TYPE  = 7,
    APIR_COMMAND_TYPE_DEVICE_GET_PROPS        = 8,
    APIR_COMMAND_TYPE_DEVICE_بفر_FROM_PTR  = 9,

    /* buffer-type */
    APIR_COMMAND_TYPE_بفر_TYPE_GET_NAME       = 10,
    APIR_COMMAND_TYPE_بفر_TYPE_GET_ALIGNMENT  = 11,
    APIR_COMMAND_TYPE_بفر_TYPE_GET_MAX_سائز   = 12,
    APIR_COMMAND_TYPE_بفر_TYPE_IS_HOST        = 13,
    APIR_COMMAND_TYPE_بفر_TYPE_ALLOC_بفر   = 14,
    APIR_COMMAND_TYPE_بفر_TYPE_GET_ALLOC_سائز = 15,

    /*بفر*/
    APIR_COMMAND_TYPE_بفر_GET_بنیاد    = 16,
    APIR_COMMAND_TYPE_بفر_سیٹ_TENSOR  = 17,
    APIR_COMMAND_TYPE_بفر_GET_TENSOR  = 18,
    APIR_COMMAND_TYPE_بفر_CPY_TENSOR  = 19,
    APIR_COMMAND_TYPE_بفر_CLEAR       = 20,
    APIR_COMMAND_TYPE_بفر_FREE_بفر = 21,

    /* backend */
    APIR_COMMAND_TYPE_BACKEND_گراف_COMPUTE = 22,

    // last command_type index + 1
    APIR_BACKEND_DISPATCH_جدول_گنتی = 23,
} ApirBackendCommandType;

ساکن اندرونی ثابت حرف * apir_dispatch_command_name(ApirBackendCommandType type) {
    انتخاب (type) {
        /* device */
        حالت APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_گنتی:
            واپس "device_get_device_count";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_گنتی:
            واپس "device_get_count";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            واپس "device_get_name";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            واپس "device_get_description";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            واپس "device_get_type";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_حافظہ:
            واپس "device_get_memory";
        حالت APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            واپس "device_supports_op";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_بفر_TYPE:
            واپس "device_get_buffer_type";
        حالت APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            واپس "device_get_props";
        حالت APIR_COMMAND_TYPE_DEVICE_بفر_FROM_PTR:
            واپس "device_buffer_from_ptr";
        /* buffer-type */
        حالت APIR_COMMAND_TYPE_بفر_TYPE_GET_NAME:
            واپس "buffer_type_get_name";
        حالت APIR_COMMAND_TYPE_بفر_TYPE_GET_ALIGNMENT:
            واپس "buffer_type_get_alignment";
        حالت APIR_COMMAND_TYPE_بفر_TYPE_GET_MAX_سائز:
            واپس "buffer_type_get_max_size";
        حالت APIR_COMMAND_TYPE_بفر_TYPE_IS_HOST:
            واپس "buffer_type_is_host";
        حالت APIR_COMMAND_TYPE_بفر_TYPE_ALLOC_بفر:
            واپس "buffer_type_alloc_buffer";
        حالت APIR_COMMAND_TYPE_بفر_TYPE_GET_ALLOC_سائز:
            واپس "buffer_type_get_alloc_size";
        /*بفر*/
        حالت APIR_COMMAND_TYPE_بفر_GET_بنیاد:
            واپس "buffer_get_base";
        حالت APIR_COMMAND_TYPE_بفر_سیٹ_TENSOR:
            واپس "buffer_set_tensor";
        حالت APIR_COMMAND_TYPE_بفر_GET_TENSOR:
            واپس "buffer_get_tensor";
        حالت APIR_COMMAND_TYPE_بفر_CPY_TENSOR:
            واپس "buffer_cpy_tensor";
        حالت APIR_COMMAND_TYPE_بفر_CLEAR:
            واپس "buffer_clear";
        حالت APIR_COMMAND_TYPE_بفر_FREE_بفر:
            واپس "buffer_free_buffer";
        /* backend */
        حالت APIR_COMMAND_TYPE_BACKEND_گراف_COMPUTE:
            واپس "backend_graph_compute";

        طےشدہ:
            واپس "unknown";
    }
}
