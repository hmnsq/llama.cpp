ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಎಣಿಕೆ ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_ಎಣಿಕೆ = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_ಎಣಿಕೆ        = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME         = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE         = 4,
    APIR_COMMAND_TYPE_DEVICE_GET_ಸ್ಮರಣೆ       = 5,
    APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP      = 6,
    APIR_COMMAND_TYPE_DEVICE_GET_ಬಫರ್_TYPE  = 7,
    APIR_COMMAND_TYPE_DEVICE_GET_PROPS        = 8,
    APIR_COMMAND_TYPE_DEVICE_ಬಫರ್_FROM_PTR  = 9,

    /* buffer-type */
    APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_NAME       = 10,
    APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_ALIGNMENT  = 11,
    APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_MAX_ಗಾತ್ರ   = 12,
    APIR_COMMAND_TYPE_ಬಫರ್_TYPE_IS_HOST        = 13,
    APIR_COMMAND_TYPE_ಬಫರ್_TYPE_ALLOC_ಬಫರ್   = 14,
    APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_ALLOC_ಗಾತ್ರ = 15,

    /*ಬಫರ್*/
    APIR_COMMAND_TYPE_ಬಫರ್_GET_ತಳ    = 16,
    APIR_COMMAND_TYPE_ಬಫರ್_ಗಣ_TENSOR  = 17,
    APIR_COMMAND_TYPE_ಬಫರ್_GET_TENSOR  = 18,
    APIR_COMMAND_TYPE_ಬಫರ್_CPY_TENSOR  = 19,
    APIR_COMMAND_TYPE_ಬಫರ್_CLEAR       = 20,
    APIR_COMMAND_TYPE_ಬಫರ್_FREE_ಬಫರ್ = 21,

    /* backend */
    APIR_COMMAND_TYPE_BACKEND_ನಕ್ಷೆ_COMPUTE = 22,

    // last command_type index + 1
    APIR_BACKEND_DISPATCH_ಕೋಷ್ಟಕ_ಎಣಿಕೆ = 23,
} ApirBackendCommandType;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * apir_dispatch_command_name(ApirBackendCommandType type) {
    ಆಯ್ಕೆ (type) {
        /* device */
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_ಎಣಿಕೆ:
            ಹಿಂತಿರುಗಿ "device_get_device_count";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_DEVICE_GET_ಎಣಿಕೆ:
            ಹಿಂತಿರುಗಿ "device_get_count";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            ಹಿಂತಿರುಗಿ "device_get_name";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            ಹಿಂತಿರುಗಿ "device_get_description";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            ಹಿಂತಿರುಗಿ "device_get_type";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_DEVICE_GET_ಸ್ಮರಣೆ:
            ಹಿಂತಿರುಗಿ "device_get_memory";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            ಹಿಂತಿರುಗಿ "device_supports_op";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_DEVICE_GET_ಬಫರ್_TYPE:
            ಹಿಂತಿರುಗಿ "device_get_buffer_type";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            ಹಿಂತಿರುಗಿ "device_get_props";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_DEVICE_ಬಫರ್_FROM_PTR:
            ಹಿಂತಿರುಗಿ "device_buffer_from_ptr";
        /* buffer-type */
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_NAME:
            ಹಿಂತಿರುಗಿ "buffer_type_get_name";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_ALIGNMENT:
            ಹಿಂತಿರುಗಿ "buffer_type_get_alignment";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_MAX_ಗಾತ್ರ:
            ಹಿಂತಿರುಗಿ "buffer_type_get_max_size";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_TYPE_IS_HOST:
            ಹಿಂತಿರುಗಿ "buffer_type_is_host";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_TYPE_ALLOC_ಬಫರ್:
            ಹಿಂತಿರುಗಿ "buffer_type_alloc_buffer";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_ALLOC_ಗಾತ್ರ:
            ಹಿಂತಿರುಗಿ "buffer_type_get_alloc_size";
        /*ಬಫರ್*/
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_GET_ತಳ:
            ಹಿಂತಿರುಗಿ "buffer_get_base";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_ಗಣ_TENSOR:
            ಹಿಂತಿರುಗಿ "buffer_set_tensor";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_GET_TENSOR:
            ಹಿಂತಿರುಗಿ "buffer_get_tensor";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_CPY_TENSOR:
            ಹಿಂತಿರುಗಿ "buffer_cpy_tensor";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_CLEAR:
            ಹಿಂತಿರುಗಿ "buffer_clear";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_ಬಫರ್_FREE_ಬಫರ್:
            ಹಿಂತಿರುಗಿ "buffer_free_buffer";
        /* backend */
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_BACKEND_ನಕ್ಷೆ_COMPUTE:
            ಹಿಂತಿರುಗಿ "backend_graph_compute";

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ "unknown";
    }
}
