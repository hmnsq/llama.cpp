#निर्देशः once

#अन्तर्भाव "apir_backend.gen.h"

#अन्तर्भाव <stdint.h>  // for uintptr_t
#अन्तर्भाव <time.h>    // for timespec, clock_gettime

#परिभाषय APIR_BACKEND_INITIALIZE_SUCCESS                     0
#परिभाषय APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY 1
#परिभाषय APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY    2
#परिभाषय APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS     3
#परिभाषय APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS        4
#परिभाषय APIR_BACKEND_INITIALIZE_BACKEND_FAILED              5
#परिभाषय APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED          6
#परिभाषय APIR_BACKEND_INITIALIZE_ALREADY_INITED              7
#परिभाषय APIR_BACKEND_INITIALIZE_NO_DEVICE                   8
#परिभाषय APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED         9

// new entries here need to be added to the apir_backend_initialize_error function below

#परिभाषय APIR_BACKEND_FORWARD_सूचकांक_INVALID 6

// 0 is fast, 1 avoids the backend to crash if an unsupported tensor is received
#परिभाषय APIR_BACKEND_CHECK_SUPPORTS_OP 0

प्रकारनिर्देश uintptr_t apir_बफर_type_host_handle_t;
प्रकारनिर्देश uintptr_t apir_बफर_host_handle_t;

स्थिर ध्रुव वर्ण * apir_backend_initialize_error(पूर्णाङ्क संहिता) {
#परिभाषय APIR_BACKEND_INITIALIZE_ERROR(संहिता_name) \
    कुरु {                                         \
        यदि (संहिता == संहिता_name)                   \
            प्रत्यागम #code_name;                   \
    } यावत् (0)

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

    प्रत्यागम "Unknown APIR_BACKEND_INITIALIZE error:/";

#अपरिभाषय APIR_BACKEND_INITIALIZE_ERROR
}
