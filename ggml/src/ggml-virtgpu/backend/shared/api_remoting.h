#هدايت once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#ضمن <unistd.h>

#ضمن <cstdint>

#وصف_ڪريو APIR_PROTOCOL_MAJOR 0
#وصف_ڪريو APIR_PROTOCOL_MINOR 2

#وصف_ڪريو APIR_HANDSHAKE_MAGIC 0xab1e

تعداد ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_ڊيگهائي = 3,
};

تعريفنوع uint64_t ApirCommandFlags;

تعداد Apir_ڀريو_Library_Return_ڪوڊ {
    APIR_ڀريو_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_ڀريو_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_ڀريو_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_ڀريو_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_ڀريو_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_ڀريو_LIBRARY_نشان_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_ڀريو_LIBRARY_INIT_بنياد_اشاريو                = 6,
};

تعداد Apir_Forward_Return_ڪوڊ {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_منزل_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_بنياد_اشاريو             = 4,
};

__attribute__((اڻ_استعمال_ٿيل_2)) ساكن مضمن ثابت حرف * apir_command_name(ApirCommandType type) {
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

__attribute__((اڻ_استعمال_ٿيل_2)) ساكن ثابت حرف * apir_ڀريو_library_error(Apir_ڀريو_Library_Return_ڪوڊ ڪوڊ) {
#وصف_ڪريو APIR_ڀريو_LIBRARY_ERROR(ڪوڊ_name) \
    افعل {                                   \
        إذا (ڪوڊ == ڪوڊ_name)             \
            أرجع #code_name;             \
    } طالما (0)

    APIR_ڀريو_LIBRARY_ERROR(APIR_ڀريو_LIBRARY_SUCCESS);
    APIR_ڀريو_LIBRARY_ERROR(APIR_ڀريو_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_ڀريو_LIBRARY_ERROR(APIR_ڀريو_LIBRARY_ALREADY_LOADED);
    APIR_ڀريو_LIBRARY_ERROR(APIR_ڀريو_LIBRARY_ENV_VAR_MISSING);
    APIR_ڀريو_LIBRARY_ERROR(APIR_ڀريو_LIBRARY_CANNOT_OPEN);
    APIR_ڀريو_LIBRARY_ERROR(APIR_ڀريو_LIBRARY_نشان_MISSING);
    APIR_ڀريو_LIBRARY_ERROR(APIR_ڀريو_LIBRARY_INIT_بنياد_اشاريو);

    أرجع "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#وصف_رد APIR_ڀريو_LIBRARY_ERROR
}

__attribute__((اڻ_استعمال_ٿيل_2)) ساكن ثابت حرف * apir_forward_error(Apir_Forward_Return_ڪوڊ ڪوڊ) {
#وصف_ڪريو APIR_FORWARD_ERROR(ڪوڊ_name) \
    افعل {                              \
        إذا (ڪوڊ == ڪوڊ_name)        \
            أرجع #code_name;        \
    } طالما (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_منزل_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_بنياد_اشاريو);

    أرجع "Unknown APIR_COMMAND_TYPE_FORWARD error";

#وصف_رد APIR_FORWARD_ERROR
}
