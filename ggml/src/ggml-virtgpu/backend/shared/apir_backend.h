#pragma once

#ಅನ್ತರ್ಭಾವ "apir_backend.gen.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>  // for uintptr_t
#ಅನ್ತರ್ಭಾವ <time.h>    // for timespec, clock_gettime

#define APIR_BACKEND_INITIALIZE_SUCCESS                     0
#define APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY 1
#define APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY    2
#define APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS     3
#define APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS        4
#define APIR_BACKEND_INITIALIZE_BACKEND_FAILED              5
#define APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED          6
#define APIR_BACKEND_INITIALIZE_ALREADY_INITED              7
#define APIR_BACKEND_INITIALIZE_NO_DEVICE                   8
#define APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED         9

// new entries here need to be added to the apir_backend_initialize_error function below

#define APIR_BACKEND_FORWARD_INDEX_INVALID 6

// 0 is fast, 1 avoids the backend to crash if an unsupported tensor is received
#define APIR_BACKEND_CHECK_SUPPORTS_OP 0

ಪ್ರಕಾರನಿರ್ದೇಶ uintptr_t apir_buffer_type_host_handle_t;
ಪ್ರಕಾರನಿರ್ದೇಶ uintptr_t apir_buffer_host_handle_t;

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * apir_backend_initialize_error(ಪೂರ್ಣಾಙ್ಕ code) {
#define APIR_BACKEND_INITIALIZE_ERROR(code_name) \
    ಕುರು {                                         \
        ಯದಿ (code == code_name)                   \
            ಪ್ರತ್ಯಾಗಮ #code_name;                   \
    } ಯಾವತ್ (0)

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

    ಪ್ರತ್ಯಾಗಮ "Unknown APIR_BACKEND_INITIALIZE error:/";

#undef APIR_BACKEND_INITIALIZE_ERROR
}
