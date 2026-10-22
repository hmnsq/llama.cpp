#ਹਦਾਇਤ once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_PROTOCOL_MAJOR 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_PROTOCOL_MINOR 2

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_HANDSHAKE_MAGIC 0xab1e

ਗਿਣਤੀ ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_ਲੰਮਾਈ = 3,
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint64_t ApirCommandFlags;

ਗਿਣਤੀ Apir_ਲੱਦੋ_Library_Return_ਕੋਡ {
    APIR_ਲੱਦੋ_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_ਲੱਦੋ_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_ਲੱਦੋ_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_ਲੱਦੋ_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_ਲੱਦੋ_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_ਲੱਦੋ_LIBRARY_ਚਿੰਨ੍ਹ_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_ਲੱਦੋ_LIBRARY_INIT_ਅਧਾਰ_ਸੂਚਕਾਂਕ                = 6,
};

ਗਿਣਤੀ Apir_Forward_Return_ਕੋਡ {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_ਮੰਜ਼ਿਲ_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_ਅਧਾਰ_ਸੂਚਕਾਂਕ             = 4,
};

__attribute__((ਅਣਵਰਤਿਆ_2)) ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ਅੱਖਰ * apir_command_name(ApirCommandType type) {
    ਚੋਣ (type) {
        ਹਾਲਤ APIR_COMMAND_TYPE_HANDSHAKE:
            ਵਾਪਸ "HandShake";
        ਹਾਲਤ APIR_COMMAND_TYPE_LOADLIBRARY:
            ਵਾਪਸ "LoadLibrary";
        ਹਾਲਤ APIR_COMMAND_TYPE_FORWARD:
            ਵਾਪਸ "Forward";
        ਮੂਲ:
            ਵਾਪਸ "unknown";
    }
}

__attribute__((ਅਣਵਰਤਿਆ_2)) ਸਥਿਰ ਅਚਲ ਅੱਖਰ * apir_ਲੱਦੋ_library_error(Apir_ਲੱਦੋ_Library_Return_ਕੋਡ ਕੋਡ) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_ਲੱਦੋ_LIBRARY_ERROR(ਕੋਡ_name) \
    ਕਰੋ {                                   \
        ਜੇ (ਕੋਡ == ਕੋਡ_name)             \
            ਵਾਪਸ #code_name;             \
    } ਜਦੋਂ (0)

    APIR_ਲੱਦੋ_LIBRARY_ERROR(APIR_ਲੱਦੋ_LIBRARY_SUCCESS);
    APIR_ਲੱਦੋ_LIBRARY_ERROR(APIR_ਲੱਦੋ_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_ਲੱਦੋ_LIBRARY_ERROR(APIR_ਲੱਦੋ_LIBRARY_ALREADY_LOADED);
    APIR_ਲੱਦੋ_LIBRARY_ERROR(APIR_ਲੱਦੋ_LIBRARY_ENV_VAR_MISSING);
    APIR_ਲੱਦੋ_LIBRARY_ERROR(APIR_ਲੱਦੋ_LIBRARY_CANNOT_OPEN);
    APIR_ਲੱਦੋ_LIBRARY_ERROR(APIR_ਲੱਦੋ_LIBRARY_ਚਿੰਨ੍ਹ_MISSING);
    APIR_ਲੱਦੋ_LIBRARY_ERROR(APIR_ਲੱਦੋ_LIBRARY_INIT_ਅਧਾਰ_ਸੂਚਕਾਂਕ);

    ਵਾਪਸ "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ APIR_ਲੱਦੋ_LIBRARY_ERROR
}

__attribute__((ਅਣਵਰਤਿਆ_2)) ਸਥਿਰ ਅਚਲ ਅੱਖਰ * apir_forward_error(Apir_Forward_Return_ਕੋਡ ਕੋਡ) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_FORWARD_ERROR(ਕੋਡ_name) \
    ਕਰੋ {                              \
        ਜੇ (ਕੋਡ == ਕੋਡ_name)        \
            ਵਾਪਸ #code_name;        \
    } ਜਦੋਂ (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_ਮੰਜ਼ਿਲ_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_ਅਧਾਰ_ਸੂਚਕਾਂਕ);

    ਵਾਪਸ "Unknown APIR_COMMAND_TYPE_FORWARD error";

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ APIR_FORWARD_ERROR
}
