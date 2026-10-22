#لارښوونه once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#ضمن <unistd.h>

#ضمن <cstdint>

#تعریف_کړه APIR_PROTOCOL_MAJOR 0
#تعریف_کړه APIR_PROTOCOL_MINOR 2

#تعریف_کړه APIR_HANDSHAKE_MAGIC 0xab1e

تعداد ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_اوږدتیا = 3,
};

تعريفنوع uint64_t ApirCommandFlags;

تعداد Apir_بارول_Library_Return_کوډ {
    APIR_بارول_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_بارول_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_بارول_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_بارول_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_بارول_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_بارول_LIBRARY_نښه_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_بارول_LIBRARY_INIT_بنسټ_شاخص                = 6,
};

تعداد Apir_Forward_Return_کوډ {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_منزل_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_بنسټ_شاخص             = 4,
};

__attribute__((ناکارول_شوی_2)) ساكن مضمن ثابت حرف * apir_command_name(ApirCommandType type) {
    بدل (type) {
        حالة APIR_COMMAND_TYPE_HANDSHAKE:
            أرجع "HandShake";
        حالة APIR_COMMAND_TYPE_LOADLIBRARY:
            أرجع "LoadLibrary";
        حالة APIR_COMMAND_TYPE_FORWARD:
            أرجع "Forward";
        افتراضي:
            أرجع "unknown";
    }
}

__attribute__((ناکارول_شوی_2)) ساكن ثابت حرف * apir_بارول_library_error(Apir_بارول_Library_Return_کوډ کوډ) {
#تعریف_کړه APIR_بارول_LIBRARY_ERROR(کوډ_name) \
    افعل {                                   \
        إذا (کوډ == کوډ_name)             \
            أرجع #code_name;             \
    } طالما (0)

    APIR_بارول_LIBRARY_ERROR(APIR_بارول_LIBRARY_SUCCESS);
    APIR_بارول_LIBRARY_ERROR(APIR_بارول_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_بارول_LIBRARY_ERROR(APIR_بارول_LIBRARY_ALREADY_LOADED);
    APIR_بارول_LIBRARY_ERROR(APIR_بارول_LIBRARY_ENV_VAR_MISSING);
    APIR_بارول_LIBRARY_ERROR(APIR_بارول_LIBRARY_CANNOT_OPEN);
    APIR_بارول_LIBRARY_ERROR(APIR_بارول_LIBRARY_نښه_MISSING);
    APIR_بارول_LIBRARY_ERROR(APIR_بارول_LIBRARY_INIT_بنسټ_شاخص);

    أرجع "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#تعریف_لغوه APIR_بارول_LIBRARY_ERROR
}

__attribute__((ناکارول_شوی_2)) ساكن ثابت حرف * apir_forward_error(Apir_Forward_Return_کوډ کوډ) {
#تعریف_کړه APIR_FORWARD_ERROR(کوډ_name) \
    افعل {                              \
        إذا (کوډ == کوډ_name)        \
            أرجع #code_name;        \
    } طالما (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_منزل_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_بنسټ_شاخص);

    أرجع "Unknown APIR_COMMAND_TYPE_FORWARD error";

#تعریف_لغوه APIR_FORWARD_ERROR
}
