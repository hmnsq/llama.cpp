#رهنمود once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#ضمن <unistd.h>

#ضمن <cstdint>

#تعریف_کن APIR_PROTOCOL_MAJOR 0
#تعریف_کن APIR_PROTOCOL_MINOR 2

#تعریف_کن APIR_HANDSHAKE_MAGIC 0xab1e

تعداد ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_درازی = 3,
};

تعريفنوع uint64_t ApirCommandFlags;

تعداد Apir_بارگیری_Library_Return_کد {
    APIR_بارگیری_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_بارگیری_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_بارگیری_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_بارگیری_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_بارگیری_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_بارگیری_LIBRARY_نماد_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_بارگیری_LIBRARY_INIT_پایه_شاخص                = 6,
};

تعداد Apir_Forward_Return_کد {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_مقصد_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_پایه_شاخص             = 4,
};

__attribute__((استفادهنشده_2)) ساكن مضمن ثابت حرف * apir_command_name(ApirCommandType type) {
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

__attribute__((استفادهنشده_2)) ساكن ثابت حرف * apir_بارگیری_library_error(Apir_بارگیری_Library_Return_کد کد) {
#تعریف_کن APIR_بارگیری_LIBRARY_ERROR(کد_name) \
    افعل {                                   \
        إذا (کد == کد_name)             \
            أرجع #code_name;             \
    } طالما (0)

    APIR_بارگیری_LIBRARY_ERROR(APIR_بارگیری_LIBRARY_SUCCESS);
    APIR_بارگیری_LIBRARY_ERROR(APIR_بارگیری_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_بارگیری_LIBRARY_ERROR(APIR_بارگیری_LIBRARY_ALREADY_LOADED);
    APIR_بارگیری_LIBRARY_ERROR(APIR_بارگیری_LIBRARY_ENV_VAR_MISSING);
    APIR_بارگیری_LIBRARY_ERROR(APIR_بارگیری_LIBRARY_CANNOT_OPEN);
    APIR_بارگیری_LIBRARY_ERROR(APIR_بارگیری_LIBRARY_نماد_MISSING);
    APIR_بارگیری_LIBRARY_ERROR(APIR_بارگیری_LIBRARY_INIT_پایه_شاخص);

    أرجع "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#لغو_تعریف APIR_بارگیری_LIBRARY_ERROR
}

__attribute__((استفادهنشده_2)) ساكن ثابت حرف * apir_forward_error(Apir_Forward_Return_کد کد) {
#تعریف_کن APIR_FORWARD_ERROR(کد_name) \
    افعل {                              \
        إذا (کد == کد_name)        \
            أرجع #code_name;        \
    } طالما (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_مقصد_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_پایه_شاخص);

    أرجع "Unknown APIR_COMMAND_TYPE_FORWARD error";

#لغو_تعریف APIR_FORWARD_ERROR
}
