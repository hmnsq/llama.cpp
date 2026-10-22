typedef ga_nanaa ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_COUNT        = 1,
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
    APIR_BACKEND_DISPATCH_TABLE_COUNT = 23,
} ApirBackendCommandType;

static anta_hstha const var_na * apir_dispatch_command_name(ApirBackendCommandType type) {
    switch (type) {
        /* device */
        pak_sa APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT:
            ᨀᨘᨒ "device_get_device_count";
        pak_sa APIR_COMMAND_TYPE_DEVICE_GET_COUNT:
            ᨀᨘᨒ "device_get_count";
        pak_sa APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            ᨀᨘᨒ "device_get_name";
        pak_sa APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            ᨀᨘᨒ "device_get_description";
        pak_sa APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            ᨀᨘᨒ "device_get_type";
        pak_sa APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            ᨀᨘᨒ "device_get_memory";
        pak_sa APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            ᨀᨘᨒ "device_supports_op";
        pak_sa APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            ᨀᨘᨒ "device_get_buffer_type";
        pak_sa APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            ᨀᨘᨒ "device_get_props";
        pak_sa APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            ᨀᨘᨒ "device_buffer_from_ptr";
        /* buffer-type */
        pak_sa APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            ᨀᨘᨒ "buffer_type_get_name";
        pak_sa APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            ᨀᨘᨒ "buffer_type_get_alignment";
        pak_sa APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            ᨀᨘᨒ "buffer_type_get_max_size";
        pak_sa APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            ᨀᨘᨒ "buffer_type_is_host";
        pak_sa APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            ᨀᨘᨒ "buffer_type_alloc_buffer";
        pak_sa APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            ᨀᨘᨒ "buffer_type_get_alloc_size";
        /* buffer */
        pak_sa APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            ᨀᨘᨒ "buffer_get_base";
        pak_sa APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            ᨀᨘᨒ "buffer_set_tensor";
        pak_sa APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            ᨀᨘᨒ "buffer_get_tensor";
        pak_sa APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            ᨀᨘᨒ "buffer_cpy_tensor";
        pak_sa APIR_COMMAND_TYPE_BUFFER_CLEAR:
            ᨀᨘᨒ "buffer_clear";
        pak_sa APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            ᨀᨘᨒ "buffer_free_buffer";
        /* backend */
        pak_sa APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            ᨀᨘᨒ "backend_graph_compute";

        default:
            ᨀᨘᨒ "unknown";
    }
}
