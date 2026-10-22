#ہدایت once

#شامل "apir_backend.gen.h"

#شامل <stdint.h>  // for uintptr_t
#شامل <time.h>    // for timespec, clock_gettime

#تعریف_کرو APIR_BACKEND_INITIALIZE_SUCCESS                     0
#تعریف_کرو APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY 1
#تعریف_کرو APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY    2
#تعریف_کرو APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS     3
#تعریف_کرو APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS        4
#تعریف_کرو APIR_BACKEND_INITIALIZE_BACKEND_FAILED              5
#تعریف_کرو APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED          6
#تعریف_کرو APIR_BACKEND_INITIALIZE_ALREADY_INITED              7
#تعریف_کرو APIR_BACKEND_INITIALIZE_NO_DEVICE                   8
#تعریف_کرو APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED         9

// new entries here need to be added to the apir_backend_initialize_error function below

#تعریف_کرو APIR_BACKEND_FORWARD_اشاریہ_INVALID 6

// 0 is fast, 1 avoids the backend to crash if an unsupported tensor is received
#تعریف_کرو APIR_BACKEND_CHECK_SUPPORTS_OP 0

قسمتعریف uintptr_t apir_بفر_type_host_handle_t;
قسمتعریف uintptr_t apir_بفر_host_handle_t;

ساکن ثابت حرف * apir_backend_initialize_error(صحیح رمز) {
#تعریف_کرو APIR_BACKEND_INITIALIZE_ERROR(رمز_name) \
    کرو {                                         \
        اگر (رمز == رمز_name)                   \
            واپس #code_name;                   \
    } جب (0)

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

    واپس "Unknown APIR_BACKEND_INITIALIZE error:/";

#تعریف_منسوخ APIR_BACKEND_INITIALIZE_ERROR
}
