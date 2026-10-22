#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "apir_backend.gen.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>  // for uintptr_t
#ਅਨ੍ਤਰ੍ਭਾਵ <time.h>    // for timespec, clock_gettime

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_SUCCESS                     0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY    2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS     3
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS        4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_BACKEND_FAILED              5
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED          6
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_ALREADY_INITED              7
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_NO_DEVICE                   8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED         9

// new entries here need to be added to the apir_backend_initialize_error function below

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_FORWARD_ਸੂਚਕਾਂਕ_INVALID 6

// 0 is fast, 1 avoids the backend to crash if an unsupported tensor is received
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_CHECK_SUPPORTS_OP 0

ਕਿਸਮਪਰਿਭਾਸ਼ਾ uintptr_t apir_ਬਫਰ_type_host_handle_t;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ uintptr_t apir_ਬਫਰ_host_handle_t;

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * apir_backend_initialize_error(ਪੂਰਨ ਕੋਡ) {
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_BACKEND_INITIALIZE_ERROR(ਕੋਡ_name) \
    ਕਰੋ {                                         \
        ਜੇ (ਕੋਡ == ਕੋਡ_name)                   \
            ਵਾਪਸ #code_name;                   \
    } ਜਦੋਂ (0)

    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_SUCCESS);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_BACKEND_FAILED);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_ALREADY_INITED);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_NO_DEVICE);
    APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED);

    ਵਾਪਸ "Unknown APIR_BACKEND_INITIALIZE error:/";

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ APIR_BACKEND_INITIALIZE_ERROR
}
