#సూచన once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#అన్తర్భావ <unistd.h>

#అన్తర్భావ <cstdint>

#నిర్వచించు APIR_PROTOCOL_MAJOR 0
#నిర్వచించు APIR_PROTOCOL_MINOR 2

#నిర్వచించు APIR_HANDSHAKE_MAGIC 0xab1e

గణన ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_నిడివి = 3,
};

రకనిర్వచనం uint64_t ApirCommandFlags;

గణన Apir_ఎక్కించు_Library_Return_సంకేతం {
    APIR_ఎక్కించు_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_ఎక్కించు_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_ఎక్కించు_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_ఎక్కించు_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_ఎక్కించు_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_ఎక్కించు_LIBRARY_చిహ్నం_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_ఎక్కించు_LIBRARY_INIT_పునాది_సూచీ                = 6,
};

గణన Apir_Forward_Return_సంకేతం {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_గమ్యం_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_పునాది_సూచీ             = 4,
};

__attribute__((వాడని_2)) స్థిర అంతఃవరుస స్థిరరాశి అక్షరం * apir_command_name(ApirCommandType type) {
    ఎంపిక (type) {
        స్థితి APIR_COMMAND_TYPE_HANDSHAKE:
            తిరిగి "HandShake";
        స్థితి APIR_COMMAND_TYPE_LOADLIBRARY:
            తిరిగి "LoadLibrary";
        స్థితి APIR_COMMAND_TYPE_FORWARD:
            తిరిగి "Forward";
        అప్రమేయ:
            తిరిగి "unknown";
    }
}

__attribute__((వాడని_2)) స్థిర స్థిరరాశి అక్షరం * apir_ఎక్కించు_library_error(Apir_ఎక్కించు_Library_Return_సంకేతం సంకేతం_2) {
#నిర్వచించు APIR_ఎక్కించు_LIBRARY_ERROR(సంకేతం_name) \
    చేయి {                                   \
        ఒకవేళ (సంకేతం_2 == సంకేతం_name)             \
            తిరిగి #code_name;             \
    } అయితే (0)

    APIR_ఎక్కించు_LIBRARY_ERROR(APIR_ఎక్కించు_LIBRARY_SUCCESS);
    APIR_ఎక్కించు_LIBRARY_ERROR(APIR_ఎక్కించు_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_ఎక్కించు_LIBRARY_ERROR(APIR_ఎక్కించు_LIBRARY_ALREADY_LOADED);
    APIR_ఎక్కించు_LIBRARY_ERROR(APIR_ఎక్కించు_LIBRARY_ENV_VAR_MISSING);
    APIR_ఎక్కించు_LIBRARY_ERROR(APIR_ఎక్కించు_LIBRARY_CANNOT_OPEN);
    APIR_ఎక్కించు_LIBRARY_ERROR(APIR_ఎక్కించు_LIBRARY_చిహ్నం_MISSING);
    APIR_ఎక్కించు_LIBRARY_ERROR(APIR_ఎక్కించు_LIBRARY_INIT_పునాది_సూచీ);

    తిరిగి "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#నిర్వచనం_రద్దు APIR_ఎక్కించు_LIBRARY_ERROR
}

__attribute__((వాడని_2)) స్థిర స్థిరరాశి అక్షరం * apir_forward_error(Apir_Forward_Return_సంకేతం సంకేతం_2) {
#నిర్వచించు APIR_FORWARD_ERROR(సంకేతం_name) \
    చేయి {                              \
        ఒకవేళ (సంకేతం_2 == సంకేతం_name)        \
            తిరిగి #code_name;        \
    } అయితే (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_గమ్యం_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_పునాది_సూచీ);

    తిరిగి "Unknown APIR_COMMAND_TYPE_FORWARD error";

#నిర్వచనం_రద్దు APIR_FORWARD_ERROR
}
