ಪ್ರಕಾರನಿರ್ದೇಶ ಗಣನಾ ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_ಎಣಿಕೆ = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_ಎಣಿಕೆ        = 1,
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
    APIR_BACKEND_DISPATCH_TABLE_ಎಣಿಕೆ = 23,
} ApirBackendCommandType;

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ ವರ್ಣ * apir_dispatch_command_name(ApirBackendCommandType type) {
    ವಿಕಲ್ಪ (type) {
        /* device */
        ಪಕ್ಷ APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_ಎಣಿಕೆ:
            ಪ್ರತ್ಯಾಗಮ "device_get_device_count";
        ಪಕ್ಷ APIR_COMMAND_TYPE_DEVICE_GET_ಎಣಿಕೆ:
            ಪ್ರತ್ಯಾಗಮ "device_get_count";
        ಪಕ್ಷ APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            ಪ್ರತ್ಯಾಗಮ "device_get_name";
        ಪಕ್ಷ APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            ಪ್ರತ್ಯಾಗಮ "device_get_description";
        ಪಕ್ಷ APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            ಪ್ರತ್ಯಾಗಮ "device_get_type";
        ಪಕ್ಷ APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            ಪ್ರತ್ಯಾಗಮ "device_get_memory";
        ಪಕ್ಷ APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            ಪ್ರತ್ಯಾಗಮ "device_supports_op";
        ಪಕ್ಷ APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            ಪ್ರತ್ಯಾಗಮ "device_get_buffer_type";
        ಪಕ್ಷ APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            ಪ್ರತ್ಯಾಗಮ "device_get_props";
        ಪಕ್ಷ APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            ಪ್ರತ್ಯಾಗಮ "device_buffer_from_ptr";
        /* buffer-type */
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            ಪ್ರತ್ಯಾಗಮ "buffer_type_get_name";
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            ಪ್ರತ್ಯಾಗಮ "buffer_type_get_alignment";
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            ಪ್ರತ್ಯಾಗಮ "buffer_type_get_max_size";
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            ಪ್ರತ್ಯಾಗಮ "buffer_type_is_host";
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            ಪ್ರತ್ಯಾಗಮ "buffer_type_alloc_buffer";
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            ಪ್ರತ್ಯಾಗಮ "buffer_type_get_alloc_size";
        /* buffer */
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            ಪ್ರತ್ಯಾಗಮ "buffer_get_base";
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            ಪ್ರತ್ಯಾಗಮ "buffer_set_tensor";
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            ಪ್ರತ್ಯಾಗಮ "buffer_get_tensor";
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            ಪ್ರತ್ಯಾಗಮ "buffer_cpy_tensor";
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_CLEAR:
            ಪ್ರತ್ಯಾಗಮ "buffer_clear";
        ಪಕ್ಷ APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            ಪ್ರತ್ಯಾಗಮ "buffer_free_buffer";
        /* backend */
        ಪಕ್ಷ APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            ಪ್ರತ್ಯಾಗಮ "backend_graph_compute";

        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ "unknown";
    }
}
