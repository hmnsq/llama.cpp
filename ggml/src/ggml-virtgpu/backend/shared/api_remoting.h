#निर्देशः once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#अन्तर्भाव <unistd.h>

#अन्तर्भाव <cstdint>

#परिभाषय APIR_PROTOCOL_MAJOR 0
#परिभाषय APIR_PROTOCOL_MINOR 2

#परिभाषय APIR_HANDSHAKE_MAGIC 0xab1e

गणना ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_दीर्घता = 3,
};

प्रकारनिर्देश uint64_t ApirCommandFlags;

गणना Apir_आरोपय_Library_Return_संहिता {
    APIR_आरोपय_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_आरोपय_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_आरोपय_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_आरोपय_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_आरोपय_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_आरोपय_LIBRARY_चिह्नम्_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_आरोपय_LIBRARY_INIT_आधारः_सूचकांक                = 6,
};

गणना Apir_Forward_Return_संहिता {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_गन्तव्यम्_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_आधारः_सूचकांक             = 4,
};

__attribute__((अप्रयुक्तम्_2)) स्थिर अन्तःस्थ ध्रुव वर्ण * apir_command_name(ApirCommandType type) {
    विकल्प (type) {
        पक्ष APIR_COMMAND_TYPE_HANDSHAKE:
            प्रत्यागम "HandShake";
        पक्ष APIR_COMMAND_TYPE_LOADLIBRARY:
            प्रत्यागम "LoadLibrary";
        पक्ष APIR_COMMAND_TYPE_FORWARD:
            प्रत्यागम "Forward";
        स्वभाव:
            प्रत्यागम "unknown";
    }
}

__attribute__((अप्रयुक्तम्_2)) स्थिर ध्रुव वर्ण * apir_आरोपय_library_error(Apir_आरोपय_Library_Return_संहिता संहिता) {
#परिभाषय APIR_आरोपय_LIBRARY_ERROR(संहिता_name) \
    कुरु {                                   \
        यदि (संहिता == संहिता_name)             \
            प्रत्यागम #code_name;             \
    } यावत् (0)

    APIR_आरोपय_LIBRARY_ERROR(APIR_आरोपय_LIBRARY_SUCCESS);
    APIR_आरोपय_LIBRARY_ERROR(APIR_आरोपय_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_आरोपय_LIBRARY_ERROR(APIR_आरोपय_LIBRARY_ALREADY_LOADED);
    APIR_आरोपय_LIBRARY_ERROR(APIR_आरोपय_LIBRARY_ENV_VAR_MISSING);
    APIR_आरोपय_LIBRARY_ERROR(APIR_आरोपय_LIBRARY_CANNOT_OPEN);
    APIR_आरोपय_LIBRARY_ERROR(APIR_आरोपय_LIBRARY_चिह्नम्_MISSING);
    APIR_आरोपय_LIBRARY_ERROR(APIR_आरोपय_LIBRARY_INIT_आधारः_सूचकांक);

    प्रत्यागम "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#अपरिभाषय APIR_आरोपय_LIBRARY_ERROR
}

__attribute__((अप्रयुक्तम्_2)) स्थिर ध्रुव वर्ण * apir_forward_error(Apir_Forward_Return_संहिता संहिता) {
#परिभाषय APIR_FORWARD_ERROR(संहिता_name) \
    कुरु {                              \
        यदि (संहिता == संहिता_name)        \
            प्रत्यागम #code_name;        \
    } यावत् (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_गन्तव्यम्_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_आधारः_सूचकांक);

    प्रत्यागम "Unknown APIR_COMMAND_TYPE_FORWARD error";

#अपरिभाषय APIR_FORWARD_ERROR
}
