#ہدایت once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#ضمن <unistd.h>

#ضمن <cstdint>

#تعریف_کرو APIR_PROTOCOL_MAJOR 0
#تعریف_کرو APIR_PROTOCOL_MINOR 2

#تعریف_کرو APIR_HANDSHAKE_MAGIC 0xab1e

تعداد ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_لمائی = 3,
};

تعريفنوع uint64_t ApirCommandFlags;

تعداد Apir_لدو_Library_Return_کوڈ {
    APIR_لدو_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_لدو_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_لدو_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_لدو_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_لدو_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_لدو_LIBRARY_علامت_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_لدو_LIBRARY_INIT_ادھار_اشاریہ                = 6,
};

تعداد Apir_Forward_Return_کوڈ {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_منزل_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_ادھار_اشاریہ             = 4,
};

__attribute__((بن_ورتیا_2)) ساكن مضمن ثابت حرف * apir_command_name(ApirCommandType type) {
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

__attribute__((بن_ورتیا_2)) ساكن ثابت حرف * apir_لدو_library_error(Apir_لدو_Library_Return_کوڈ کوڈ) {
#تعریف_کرو APIR_لدو_LIBRARY_ERROR(کوڈ_name) \
    افعل {                                   \
        إذا (کوڈ == کوڈ_name)             \
            أرجع #code_name;             \
    } طالما (0)

    APIR_لدو_LIBRARY_ERROR(APIR_لدو_LIBRARY_SUCCESS);
    APIR_لدو_LIBRARY_ERROR(APIR_لدو_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_لدو_LIBRARY_ERROR(APIR_لدو_LIBRARY_ALREADY_LOADED);
    APIR_لدو_LIBRARY_ERROR(APIR_لدو_LIBRARY_ENV_VAR_MISSING);
    APIR_لدو_LIBRARY_ERROR(APIR_لدو_LIBRARY_CANNOT_OPEN);
    APIR_لدو_LIBRARY_ERROR(APIR_لدو_LIBRARY_علامت_MISSING);
    APIR_لدو_LIBRARY_ERROR(APIR_لدو_LIBRARY_INIT_ادھار_اشاریہ);

    أرجع "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#تعریف_رد APIR_لدو_LIBRARY_ERROR
}

__attribute__((بن_ورتیا_2)) ساكن ثابت حرف * apir_forward_error(Apir_Forward_Return_کوڈ کوڈ) {
#تعریف_کرو APIR_FORWARD_ERROR(کوڈ_name) \
    افعل {                              \
        إذا (کوڈ == کوڈ_name)        \
            أرجع #code_name;        \
    } طالما (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_منزل_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_ادھار_اشاریہ);

    أرجع "Unknown APIR_COMMAND_TYPE_FORWARD error";

#تعریف_رد APIR_FORWARD_ERROR
}
