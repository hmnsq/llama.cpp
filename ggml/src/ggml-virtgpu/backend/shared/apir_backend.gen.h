రకనిర్వచనం గణన ApirBackendCommandType {

    /* device */
    APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_లెక్క = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_లెక్క        = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME         = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE         = 4,
    APIR_COMMAND_TYPE_DEVICE_GET_జ్ఞాపకం       = 5,
    APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP      = 6,
    APIR_COMMAND_TYPE_DEVICE_GET_బఫరు_TYPE  = 7,
    APIR_COMMAND_TYPE_DEVICE_GET_PROPS        = 8,
    APIR_COMMAND_TYPE_DEVICE_బఫరు_FROM_PTR  = 9,

    /* buffer-type */
    APIR_COMMAND_TYPE_బఫరు_TYPE_GET_NAME       = 10,
    APIR_COMMAND_TYPE_బఫరు_TYPE_GET_ALIGNMENT  = 11,
    APIR_COMMAND_TYPE_బఫరు_TYPE_GET_MAX_కొలత   = 12,
    APIR_COMMAND_TYPE_బఫరు_TYPE_IS_HOST        = 13,
    APIR_COMMAND_TYPE_బఫరు_TYPE_ALLOC_బఫరు   = 14,
    APIR_COMMAND_TYPE_బఫరు_TYPE_GET_ALLOC_కొలత = 15,

    /*బఫరు*/
    APIR_COMMAND_TYPE_బఫరు_GET_పునాది    = 16,
    APIR_COMMAND_TYPE_బఫరు_సమితి_TENSOR  = 17,
    APIR_COMMAND_TYPE_బఫరు_GET_TENSOR  = 18,
    APIR_COMMAND_TYPE_బఫరు_CPY_TENSOR  = 19,
    APIR_COMMAND_TYPE_బఫరు_CLEAR       = 20,
    APIR_COMMAND_TYPE_బఫరు_FREE_బఫరు = 21,

    /* backend */
    APIR_COMMAND_TYPE_BACKEND_లేఖాచిత్రం_COMPUTE = 22,

    // last command_type index + 1
    APIR_BACKEND_DISPATCH_పట్టిక_లెక్క = 23,
} ApirBackendCommandType;

స్థిర అంతఃవరుస స్థిరరాశి అక్షరం * apir_dispatch_command_name(ApirBackendCommandType type) {
    ఎంపిక (type) {
        /* device */
        స్థితి APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_లెక్క:
            తిరిగి "device_get_device_count";
        స్థితి APIR_COMMAND_TYPE_DEVICE_GET_లెక్క:
            తిరిగి "device_get_count";
        స్థితి APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            తిరిగి "device_get_name";
        స్థితి APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            తిరిగి "device_get_description";
        స్థితి APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            తిరిగి "device_get_type";
        స్థితి APIR_COMMAND_TYPE_DEVICE_GET_జ్ఞాపకం:
            తిరిగి "device_get_memory";
        స్థితి APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            తిరిగి "device_supports_op";
        స్థితి APIR_COMMAND_TYPE_DEVICE_GET_బఫరు_TYPE:
            తిరిగి "device_get_buffer_type";
        స్థితి APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            తిరిగి "device_get_props";
        స్థితి APIR_COMMAND_TYPE_DEVICE_బఫరు_FROM_PTR:
            తిరిగి "device_buffer_from_ptr";
        /* buffer-type */
        స్థితి APIR_COMMAND_TYPE_బఫరు_TYPE_GET_NAME:
            తిరిగి "buffer_type_get_name";
        స్థితి APIR_COMMAND_TYPE_బఫరు_TYPE_GET_ALIGNMENT:
            తిరిగి "buffer_type_get_alignment";
        స్థితి APIR_COMMAND_TYPE_బఫరు_TYPE_GET_MAX_కొలత:
            తిరిగి "buffer_type_get_max_size";
        స్థితి APIR_COMMAND_TYPE_బఫరు_TYPE_IS_HOST:
            తిరిగి "buffer_type_is_host";
        స్థితి APIR_COMMAND_TYPE_బఫరు_TYPE_ALLOC_బఫరు:
            తిరిగి "buffer_type_alloc_buffer";
        స్థితి APIR_COMMAND_TYPE_బఫరు_TYPE_GET_ALLOC_కొలత:
            తిరిగి "buffer_type_get_alloc_size";
        /*బఫరు*/
        స్థితి APIR_COMMAND_TYPE_బఫరు_GET_పునాది:
            తిరిగి "buffer_get_base";
        స్థితి APIR_COMMAND_TYPE_బఫరు_సమితి_TENSOR:
            తిరిగి "buffer_set_tensor";
        స్థితి APIR_COMMAND_TYPE_బఫరు_GET_TENSOR:
            తిరిగి "buffer_get_tensor";
        స్థితి APIR_COMMAND_TYPE_బఫరు_CPY_TENSOR:
            తిరిగి "buffer_cpy_tensor";
        స్థితి APIR_COMMAND_TYPE_బఫరు_CLEAR:
            తిరిగి "buffer_clear";
        స్థితి APIR_COMMAND_TYPE_బఫరు_FREE_బఫరు:
            తిరిగి "buffer_free_buffer";
        /* backend */
        స్థితి APIR_COMMAND_TYPE_BACKEND_లేఖాచిత్రం_COMPUTE:
            తిరిగి "backend_graph_compute";

        అప్రమేయ:
            తిరిగి "unknown";
    }
}
