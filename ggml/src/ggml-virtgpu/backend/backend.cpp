#ಅನ್ತರ್ಭಾವ "backend-dispatched.h"
#ಅನ್ತರ್ಭಾವ "backend-virgl-apir.h"
#ಅನ್ತರ್ಭಾವ "shared/api_remoting.h"
#ಅನ್ತರ್ಭಾವ "shared/apir_backend.h"
#ಅನ್ತರ್ಭಾವ "shared/apir_cs.h"

#ಅನ್ತರ್ಭಾವ <dlfcn.h>
#ಅನ್ತರ್ಭಾವ <ggml-backend.h>

#ಅನ್ತರ್ಭಾವ <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

ಸ್ಥಿರ ಶೂನ್ಯ * backend_library_handle = ರಿಕ್ತ;
ಸ್ಥಿರ FILE * apir_logfile           = ರಿಕ್ತ;

ಸ್ಥಿರ ಶೂನ್ಯ log_to_file_callback(ಗಣನಾ ggml_log_level level, ಧ್ರುವ ವರ್ಣ * text, ಶೂನ್ಯ * user_data) {
    FILE * logfile = (FILE *) user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

ಬಾಹ್ಯ "C" {
ಶೂನ್ಯ apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    ಪ್ರತಿ (ಧ್ರುವ auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    ಯದಿ (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = ರಿಕ್ತ;
    }

    ಯದಿ (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = ರಿಕ್ತ;
    }
}

#define APIR_GGML_LIBRARY_PATH_ಕೀಲಿ "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_ಕೀಲಿ  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, ಸಂರಚನಾ virgl_apir_callbacks * virgl_cbs) {
    ಧ್ರುವ ವರ್ಣ * dlsym_error;

    ಧ್ರುವ ವರ್ಣ * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    ಯದಿ (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        ಯದಿ (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } ಅನ್ಯಥಾ {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    ಧ್ರುವ ವರ್ಣ * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_ಕೀಲಿ);
    ಧ್ರುವ ವರ್ಣ * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_ಕೀಲಿ);
    ಧ್ರುವ ವರ್ಣ * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    ಯದಿ (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        ಪ್ರತ್ಯಾಗಮ APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    ಯದಿ (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        ಪ್ರತ್ಯಾಗಮ APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    ಯದಿ (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        ಪ್ರತ್ಯಾಗಮ APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    ಶೂನ್ಯ * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    ಯದಿ (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        ಪ್ರತ್ಯಾಗಮ APIR_LOAD_LIBRARY_ಚಿಹ್ನೆ_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    ಪ್ರತ್ಯಾಗಮ (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 ವರ್ಣ *                 dec_cur,
                                 ಧ್ರುವ ವರ್ಣ *           dec_end,
                                 ವರ್ಣ *                 enc_cur,
                                 ಧ್ರುವ ವರ್ಣ *           enc_end,
                                 ವರ್ಣ **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = ಅಸತ್ಯ,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = ಅಸತ್ಯ,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    ಯದಿ (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_ಎಣಿಕೆ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_ಎಣಿಕೆ);
        ಪ್ರತ್ಯಾಗಮ APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    ಪ್ರತ್ಯಾಗಮ ret;
}
}
