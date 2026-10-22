#సూచన once

#అన్తర్భావ "apir_backend.gen.h"

#అన్తర్భావ <stdint.h>  // for uintptr_t
#అన్తర్భావ <time.h>    // for timespec, clock_gettime

#నిర్వచించు APIR_BACKEND_INITIALIZE_SUCCESS                     0
#నిర్వచించు APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY 1
#నిర్వచించు APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY    2
#నిర్వచించు APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS     3
#నిర్వచించు APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS        4
#నిర్వచించు APIR_BACKEND_INITIALIZE_BACKEND_FAILED              5
#నిర్వచించు APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED          6
#నిర్వచించు APIR_BACKEND_INITIALIZE_ALREADY_INITED              7
#నిర్వచించు APIR_BACKEND_INITIALIZE_NO_DEVICE                   8
#నిర్వచించు APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED         9

// new entries here need to be added to the apir_backend_initialize_error function below

#నిర్వచించు APIR_BACKEND_FORWARD_సూచీ_INVALID 6

// 0 is fast, 1 avoids the backend to crash if an unsupported tensor is received
#నిర్వచించు APIR_BACKEND_CHECK_SUPPORTS_OP 0

రకనిర్వచనం uintptr_t apir_బఫరు_type_host_handle_t;
రకనిర్వచనం uintptr_t apir_బఫరు_host_handle_t;

స్థిర స్థిరరాశి అక్షరం * apir_backend_initialize_error(పూర్ణాంకం సంకేతం_2) {
#నిర్వచించు APIR_BACKEND_INITIALIZE_ERROR(సంకేతం_name) \
    చేయి {                                         \
        ఒకవేళ (సంకేతం_2 == సంకేతం_name)                   \
            తిరిగి #code_name;                   \
    } అయితే (0)

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

    తిరిగి "Unknown APIR_BACKEND_INITIALIZE error:/";

#నిర్వచనం_రద్దు APIR_BACKEND_INITIALIZE_ERROR
}
