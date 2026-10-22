प्रकार क्रमागत ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_गिनती = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_गिनती        = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME         = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE         = 4,
    APIR_COMMAND_TYPE_DEVICE_GET_MEMORY       = 5,
    APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP      = 6,
    APIR_COMMAND_TYPE_DEVICE_GET_बफर_TYPE  = 7,
    APIR_COMMAND_TYPE_DEVICE_GET_PROPS        = 8,
    APIR_COMMAND_TYPE_DEVICE_बफर_FROM_PTR  = 9,

    /* buffer-type */
    APIR_COMMAND_TYPE_बफर_TYPE_GET_NAME       = 10,
    APIR_COMMAND_TYPE_बफर_TYPE_GET_ALIGNMENT  = 11,
    APIR_COMMAND_TYPE_बफर_TYPE_GET_MAX_आकार   = 12,
    APIR_COMMAND_TYPE_बफर_TYPE_IS_HOST        = 13,
    APIR_COMMAND_TYPE_बफर_TYPE_ALLOC_बफर   = 14,
    APIR_COMMAND_TYPE_बफर_TYPE_GET_ALLOC_आकार = 15,

    /*बफर*/
    APIR_COMMAND_TYPE_बफर_GET_BASE    = 16,
    APIR_COMMAND_TYPE_बफर_SET_TENSOR  = 17,
    APIR_COMMAND_TYPE_बफर_GET_TENSOR  = 18,
    APIR_COMMAND_TYPE_बफर_CPY_TENSOR  = 19,
    APIR_COMMAND_TYPE_बफर_CLEAR       = 20,
    APIR_COMMAND_TYPE_बफर_FREE_बफर = 21,

    /* backend */
    APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE = 22,

    // last command_type index + 1
    APIR_BACKEND_DISPATCH_TABLE_गिनती = 23,
} ApirBackendCommandType;

अटल अंतरभूत स्थिर अक्षर * apir_dispatch_command_name(ApirBackendCommandType type) {
    चयन (type) {
        /* device */
        हाल APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_गिनती:
            वापस "device_get_device_count";
        हाल APIR_COMMAND_TYPE_DEVICE_GET_गिनती:
            वापस "device_get_count";
        हाल APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            वापस "device_get_name";
        हाल APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            वापस "device_get_description";
        हाल APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            वापस "device_get_type";
        हाल APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            वापस "device_get_memory";
        हाल APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            वापस "device_supports_op";
        हाल APIR_COMMAND_TYPE_DEVICE_GET_बफर_TYPE:
            वापस "device_get_buffer_type";
        हाल APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            वापस "device_get_props";
        हाल APIR_COMMAND_TYPE_DEVICE_बफर_FROM_PTR:
            वापस "device_buffer_from_ptr";
        /* buffer-type */
        हाल APIR_COMMAND_TYPE_बफर_TYPE_GET_NAME:
            वापस "buffer_type_get_name";
        हाल APIR_COMMAND_TYPE_बफर_TYPE_GET_ALIGNMENT:
            वापस "buffer_type_get_alignment";
        हाल APIR_COMMAND_TYPE_बफर_TYPE_GET_MAX_आकार:
            वापस "buffer_type_get_max_size";
        हाल APIR_COMMAND_TYPE_बफर_TYPE_IS_HOST:
            वापस "buffer_type_is_host";
        हाल APIR_COMMAND_TYPE_बफर_TYPE_ALLOC_बफर:
            वापस "buffer_type_alloc_buffer";
        हाल APIR_COMMAND_TYPE_बफर_TYPE_GET_ALLOC_आकार:
            वापस "buffer_type_get_alloc_size";
        /*बफर*/
        हाल APIR_COMMAND_TYPE_बफर_GET_BASE:
            वापस "buffer_get_base";
        हाल APIR_COMMAND_TYPE_बफर_SET_TENSOR:
            वापस "buffer_set_tensor";
        हाल APIR_COMMAND_TYPE_बफर_GET_TENSOR:
            वापस "buffer_get_tensor";
        हाल APIR_COMMAND_TYPE_बफर_CPY_TENSOR:
            वापस "buffer_cpy_tensor";
        हाल APIR_COMMAND_TYPE_बफर_CLEAR:
            वापस "buffer_clear";
        हाल APIR_COMMAND_TYPE_बफर_FREE_बफर:
            वापस "buffer_free_buffer";
        /* backend */
        हाल APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            वापस "backend_graph_compute";

        शेष:
            वापस "unknown";
    }
}
