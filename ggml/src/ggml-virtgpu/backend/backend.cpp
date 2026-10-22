#ਅਨ੍ਤਰ੍ਭਾਵ "backend-dispatched.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "backend-virgl-apir.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "shared/api_remoting.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "shared/apir_backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "shared/apir_cs.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <dlfcn.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <ggml-backend.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_LLAMA_CPP_GGML_LIBRARY_ਰਸਤਾ_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_LLAMA_CPP_LOG_ਮੰਜ਼ਿਲ_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

ਸਥਿਰ ਖਾਲੀ * backend_library_handle = ਨਿਰਾ;
ਸਥਿਰ FILE * apir_logfile           = ਨਿਰਾ;

ਸਥਿਰ ਖਾਲੀ log_ਮੰਜ਼ਿਲ_file_callback(ਗਿਣਤੀ ggml_log_level level, ਅਚਲ ਅੱਖਰ * ਪਾਠ_2, ਖਾਲੀ * user_ਅੰਕੜੇ) {
    FILE * logfile = (FILE *) user_ਅੰਕੜੇ;
    fprintf(logfile, "[%d] %s", level, ਪਾਠ_2);
    fflush(logfile);
}

ਬਾਹਰੀ "C" {
ਖਾਲੀ apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_ਅਣਵਰਤਿਆ(virgl_ctx_id);

    ਸਵੈਚਾਲਿਤ buffers = apir_get_track_backend_buffers();
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਬਫਰ : buffers) {
        apir_untrack_backend_ਬਫਰ(ਬਫਰ);
        ਬਫਰ->iface.free_ਬਫਰ(ਬਫਰ);
    }

    ਜੇ (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = ਨਿਰਾ;
    }

    ਜੇ (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = ਨਿਰਾ;
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_GGML_LIBRARY_ਰਸਤਾ_ਕੁੰਜੀ "ggml.library.path"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ APIR_GGML_LIBRARY_REG_ਕੁੰਜੀ  "ggml.library.reg"

Apir_ਲੱਦੋ_Library_Return_ਕੋਡ apir_backend_initialize(uint32_t virgl_ctx_id, ਬਣਤਰ virgl_apir_callbacks * virgl_cbs) {
    ਅਚਲ ਅੱਖਰ * dlsym_error;

    ਅਚਲ ਅੱਖਰ * apir_log_ਮੰਜ਼ਿਲ_file = getenv(APIR_LLAMA_CPP_LOG_ਮੰਜ਼ਿਲ_FILE_ENV);
    ਜੇ (apir_log_ਮੰਜ਼ਿਲ_file) {
        apir_logfile = fopen(apir_log_ਮੰਜ਼ਿਲ_file, "w");
        ਜੇ (apir_logfile) {
            ggml_log_ਸੈੱਟ(log_ਮੰਜ਼ਿਲ_file_callback, apir_logfile);
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_ਮੰਜ਼ਿਲ_file);
        }
    }

    ਅਚਲ ਅੱਖਰ * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_ਰਸਤਾ_ਕੁੰਜੀ);
    ਅਚਲ ਅੱਖਰ * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_ਕੁੰਜੀ);
    ਅਚਲ ਅੱਖਰ * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    ਜੇ (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_ਰਸਤਾ_ENV);

        ਵਾਪਸ APIR_ਲੱਦੋ_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    ਜੇ (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        ਵਾਪਸ APIR_ਲੱਦੋ_LIBRARY_CANNOT_OPEN;
    }

    ਜੇ (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        ਵਾਪਸ APIR_ਲੱਦੋ_LIBRARY_ENV_VAR_MISSING;
    }

    ਖਾਲੀ * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    ਜੇ (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        ਵਾਪਸ APIR_ਲੱਦੋ_LIBRARY_ਚਿੰਨ੍ਹ_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    ਵਾਪਸ (Apir_ਲੱਦੋ_Library_Return_ਕੋਡ) (APIR_ਲੱਦੋ_LIBRARY_INIT_ਅਧਾਰ_ਸੂਚਕਾਂਕ + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 ਅੱਖਰ *                 dec_cur,
                                 ਅਚਲ ਅੱਖਰ *           dec_ਸਿਰਾ,
                                 ਅੱਖਰ *                 enc_cur,
                                 ਅਚਲ ਅੱਖਰ *           enc_ਸਿਰਾ,
                                 ਅੱਖਰ **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_ਸਿਰਾ,
        .fatal = ਝੂਠ,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_ਸਿਰਾ,
        .fatal = ਝੂਠ,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    ਜੇ (cmd_type >= APIR_BACKEND_DISPATCH_ਸਾਰਣੀ_ਗਿਣਤੀ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_ਸਾਰਣੀ_ਗਿਣਤੀ);
        ਵਾਪਸ APIR_BACKEND_FORWARD_ਸੂਚਕਾਂਕ_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_ਸਾਰਣੀ[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    ਵਾਪਸ ret;
}
}
