#आशय once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#समावेश <unistd.h>

#समावेश <cstdint>

#घोषणा APIR_PROTOCOL_MAJOR 0
#घोषणा APIR_PROTOCOL_MINOR 2

#घोषणा APIR_HANDSHAKE_MAGIC 0xab1e

क्रमागत ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_लंबाई = 3,
};

प्रकार uint64_t ApirCommandFlags;

क्रमागत ApirLoadLibraryReturnCode {
    APIR_LOAD_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_LOAD_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_LOAD_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_LOAD_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_LOAD_LIBRARY_प्रतीक_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_LOAD_LIBRARY_INIT_BASE_सूचकांक                = 6,
};

क्रमागत ApirForwardReturnCode {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_TO_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_BASE_सूचकांक             = 4,
};

__attribute__((unused)) अटल अंतरभूत स्थिर अक्षर * apir_command_name(ApirCommandType type) {
    चयन (type) {
        हाल APIR_COMMAND_TYPE_HANDSHAKE:
            वापस "HandShake";
        हाल APIR_COMMAND_TYPE_LOADLIBRARY:
            वापस "LoadLibrary";
        हाल APIR_COMMAND_TYPE_FORWARD:
            वापस "Forward";
        शेष:
            वापस "unknown";
    }
}

__attribute__((unused)) अटल स्थिर अक्षर * apir_load_library_error(ApirLoadLibraryReturnCode code) {
#घोषणा APIR_LOAD_LIBRARY_ERROR(code_name) \
    करो {                                   \
        अगर (code == code_name)             \
            वापस #code_name;             \
    } जबतक (0)

    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_SUCCESS);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_ALREADY_LOADED);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_ENV_VAR_MISSING);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_CANNOT_OPEN);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_प्रतीक_MISSING);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_INIT_BASE_सूचकांक);

    वापस "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#अघोषित APIR_LOAD_LIBRARY_ERROR
}

__attribute__((unused)) अटल स्थिर अक्षर * apir_forward_error(ApirForwardReturnCode code) {
#घोषणा APIR_FORWARD_ERROR(code_name) \
    करो {                              \
        अगर (code == code_name)        \
            वापस #code_name;        \
    } जबतक (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_TO_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_BASE_सूचकांक);

    वापस "Unknown APIR_COMMAND_TYPE_FORWARD error";

#अघोषित APIR_FORWARD_ERROR
}
