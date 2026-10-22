#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "apir_backend.gen.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>  // for uintptr_t
#ಅನ್ತರ್ಭಾವ <time.h>    // for timespec, clock_gettime

#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_SUCCESS                     0
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY 1
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY    2
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS     3
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS        4
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_BACKEND_FAILED              5
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED          6
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_ALREADY_INITED              7
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_NO_DEVICE                   8
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED         9

// new entries here need to be added to the apir_backend_initialize_error function below

#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_FORWARD_ಸೂಚ್ಯಂಕ_INVALID 6

// 0 is fast, 1 avoids the backend to crash if an unsupported tensor is received
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_CHECK_SUPPORTS_OP 0

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uintptr_t apir_ಬಫರ್_type_host_handle_t;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uintptr_t apir_ಬಫರ್_host_handle_t;

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * apir_backend_initialize_error(ಪೂರ್ಣಾಂಕ ಸಂಕೇತ_2) {
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_BACKEND_INITIALIZE_ERROR(ಸಂಕೇತ_name) \
    ಮಾಡು {                                         \
        ಆದರೆ (ಸಂಕೇತ_2 == ಸಂಕೇತ_name)                   \
            ಹಿಂತಿರುಗಿ #code_name;                   \
    } ಆಗ (0)

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

    ಹಿಂತಿರುಗಿ "Unknown APIR_BACKEND_INITIALIZE error:/";

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು APIR_BACKEND_INITIALIZE_ERROR
}
