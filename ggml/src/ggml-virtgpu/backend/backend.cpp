#ಅನ್ತರ್ಭಾವ "backend-dispatched.h"
#ಅನ್ತರ್ಭಾವ "backend-virgl-apir.h"
#ಅನ್ತರ್ಭಾವ "shared/api_remoting.h"
#ಅನ್ತರ್ಭಾವ "shared/apir_backend.h"
#ಅನ್ತರ್ಭಾವ "shared/apir_cs.h"

#ಅನ್ತರ್ಭಾವ <dlfcn.h>
#ಅನ್ತರ್ಭಾವ <ggml-backend.h>

#ಅನ್ತರ್ಭಾವ <iostream>

#ವ್ಯಾಖ್ಯಾನಿಸು APIR_LLAMA_CPP_GGML_LIBRARY_ಮಾರ್ಗ_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_LLAMA_CPP_LOG_ಗಮ್ಯ_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

ಸ್ಥಿರ ಶೂನ್ಯ * backend_library_handle = ಖಾಲಿ;
ಸ್ಥಿರ FILE * apir_logfile           = ಖಾಲಿ;

ಸ್ಥಿರ ಶೂನ್ಯ log_ಗಮ್ಯ_file_callback(ಎಣಿಕೆ ggml_log_level level, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಪಠ್ಯ_2, ಶೂನ್ಯ * user_ದತ್ತಾಂಶ) {
    FILE * logfile = (FILE *) user_ದತ್ತಾಂಶ;
    fprintf(logfile, "[%d] %s", level, ಪಠ್ಯ_2);
    fflush(logfile);
}

ಬಾಹ್ಯ "C" {
ಶೂನ್ಯ apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_ಬಳಸದ(virgl_ctx_id);

    ಸ್ವಯಂಚಾಲಿತ buffers = apir_get_track_backend_buffers();
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಬಫರ್ : buffers) {
        apir_untrack_backend_ಬಫರ್(ಬಫರ್);
        ಬಫರ್->iface.free_ಬಫರ್(ಬಫರ್);
    }

    ಆದರೆ (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = ಖಾಲಿ;
    }

    ಆದರೆ (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = ಖಾಲಿ;
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು APIR_GGML_LIBRARY_ಮಾರ್ಗ_ಕೀಲಿ "ggml.library.path"
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_GGML_LIBRARY_REG_ಕೀಲಿ  "ggml.library.reg"

Apir_ತುಂಬು_Library_Return_ಸಂಕೇತ apir_backend_initialize(uint32_t virgl_ctx_id, ರಚನೆ virgl_apir_callbacks * virgl_cbs) {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * dlsym_error;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * apir_log_ಗಮ್ಯ_file = getenv(APIR_LLAMA_CPP_LOG_ಗಮ್ಯ_FILE_ENV);
    ಆದರೆ (apir_log_ಗಮ್ಯ_file) {
        apir_logfile = fopen(apir_log_ಗಮ್ಯ_file, "w");
        ಆದರೆ (apir_logfile) {
            ggml_log_ಗಣ(log_ಗಮ್ಯ_file_callback, apir_logfile);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_ಗಮ್ಯ_file);
        }
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_ಮಾರ್ಗ_ಕೀಲಿ);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_ಕೀಲಿ);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    ಆದರೆ (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_ಮಾರ್ಗ_ENV);

        ಹಿಂತಿರುಗಿ APIR_ತುಂಬು_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    ಆದರೆ (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        ಹಿಂತಿರುಗಿ APIR_ತುಂಬು_LIBRARY_CANNOT_OPEN;
    }

    ಆದರೆ (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        ಹಿಂತಿರುಗಿ APIR_ತುಂಬು_LIBRARY_ENV_VAR_MISSING;
    }

    ಶೂನ್ಯ * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    ಆದರೆ (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        ಹಿಂತಿರುಗಿ APIR_ತುಂಬು_LIBRARY_ಚಿಹ್ನೆ_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    ಹಿಂತಿರುಗಿ (Apir_ತುಂಬು_Library_Return_ಸಂಕೇತ) (APIR_ತುಂಬು_LIBRARY_INIT_ತಳ_ಸೂಚ್ಯಂಕ + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 ಅಕ್ಷರ *                 dec_cur,
                                 ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *           dec_ಕೊನೆ,
                                 ಅಕ್ಷರ *                 enc_cur,
                                 ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *           enc_ಕೊನೆ,
                                 ಅಕ್ಷರ **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_ಕೊನೆ,
        .fatal = ಸುಳ್ಳು,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_ಕೊನೆ,
        .fatal = ಸುಳ್ಳು,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    ಆದರೆ (cmd_type >= APIR_BACKEND_DISPATCH_ಕೋಷ್ಟಕ_ಎಣಿಕೆ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_ಕೋಷ್ಟಕ_ಎಣಿಕೆ);
        ಹಿಂತಿರುಗಿ APIR_BACKEND_FORWARD_ಸೂಚ್ಯಂಕ_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_ಕೋಷ್ಟಕ[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    ಹಿಂತಿರುಗಿ ret;
}
}
