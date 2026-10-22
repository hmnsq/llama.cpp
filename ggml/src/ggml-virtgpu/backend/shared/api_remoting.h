#ہدایت once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#شامل <unistd.h>

#شامل <cstdint>

#تعریف_کرو APIR_PROTOCOL_MAJOR 0
#تعریف_کرو APIR_PROTOCOL_MINOR 2

#تعریف_کرو APIR_HANDSHAKE_MAGIC 0xab1e

شمار ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_لمبائی = 3,
};

قسمتعریف uint64_t ApirCommandFlags;

شمار Apir_بھرو_Library_Return_رمز {
    APIR_بھرو_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_بھرو_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_بھرو_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_بھرو_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_بھرو_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_بھرو_LIBRARY_علامت_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_بھرو_LIBRARY_INIT_بنیاد_اشاریہ                = 6,
};

شمار Apir_Forward_Return_رمز {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_مقصد_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_بنیاد_اشاریہ             = 4,
};

__attribute__((غیر_مستعمل_2)) ساکن اندرونی ثابت حرف * apir_command_name(ApirCommandType type) {
    انتخاب (type) {
        حالت APIR_COMMAND_TYPE_HANDSHAKE:
            واپس "HandShake";
        حالت APIR_COMMAND_TYPE_LOADLIBRARY:
            واپس "LoadLibrary";
        حالت APIR_COMMAND_TYPE_FORWARD:
            واپس "Forward";
        طےشدہ:
            واپس "unknown";
    }
}

__attribute__((غیر_مستعمل_2)) ساکن ثابت حرف * apir_بھرو_library_error(Apir_بھرو_Library_Return_رمز رمز) {
#تعریف_کرو APIR_بھرو_LIBRARY_ERROR(رمز_name) \
    کرو {                                   \
        اگر (رمز == رمز_name)             \
            واپس #code_name;             \
    } جب (0)

    APIR_بھرو_LIBRARY_ERROR(APIR_بھرو_LIBRARY_SUCCESS);
    APIR_بھرو_LIBRARY_ERROR(APIR_بھرو_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_بھرو_LIBRARY_ERROR(APIR_بھرو_LIBRARY_ALREADY_LOADED);
    APIR_بھرو_LIBRARY_ERROR(APIR_بھرو_LIBRARY_ENV_VAR_MISSING);
    APIR_بھرو_LIBRARY_ERROR(APIR_بھرو_LIBRARY_CANNOT_OPEN);
    APIR_بھرو_LIBRARY_ERROR(APIR_بھرو_LIBRARY_علامت_MISSING);
    APIR_بھرو_LIBRARY_ERROR(APIR_بھرو_LIBRARY_INIT_بنیاد_اشاریہ);

    واپس "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#تعریف_منسوخ APIR_بھرو_LIBRARY_ERROR
}

__attribute__((غیر_مستعمل_2)) ساکن ثابت حرف * apir_forward_error(Apir_Forward_Return_رمز رمز) {
#تعریف_کرو APIR_FORWARD_ERROR(رمز_name) \
    کرو {                              \
        اگر (رمز == رمز_name)        \
            واپس #code_name;        \
    } جب (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_مقصد_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_بنیاد_اشاریہ);

    واپس "Unknown APIR_COMMAND_TYPE_FORWARD error";

#تعریف_منسوخ APIR_FORWARD_ERROR
}
