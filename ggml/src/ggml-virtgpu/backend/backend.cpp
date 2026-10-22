#अन्तर्भाव "backend-dispatched.h"
#अन्तर्भाव "backend-virgl-apir.h"
#अन्तर्भाव "shared/api_remoting.h"
#अन्तर्भाव "shared/apir_backend.h"
#अन्तर्भाव "shared/apir_cs.h"

#अन्तर्भाव <dlfcn.h>
#अन्तर्भाव <ggml-backend.h>

#अन्तर्भाव <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

स्थिर शून्य * backend_library_handle = रिक्त;
स्थिर FILE * apir_logfile           = रिक्त;

स्थिर शून्य log_to_file_callback(गणना ggml_log_level level, ध्रुव वर्ण * text, शून्य * user_data) {
    FILE * logfile = (FILE *) user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

बाह्य "C" {
शून्य apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    प्रति (ध्रुव auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    जइ (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = रिक्त;
    }

    जइ (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = रिक्त;
    }
}

#define APIR_GGML_LIBRARY_PATH_कुञ्चिका "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_कुञ्चिका  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, संरचना virgl_apir_callbacks * virgl_cbs) {
    ध्रुव वर्ण * dlsym_error;

    ध्रुव वर्ण * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    जइ (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        जइ (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } अण्णहा {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    ध्रुव वर्ण * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_कुञ्चिका);
    ध्रुव वर्ण * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_कुञ्चिका);
    ध्रुव वर्ण * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    जइ (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        प्रत्यागम APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    जइ (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        प्रत्यागम APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    जइ (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        प्रत्यागम APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    शून्य * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    जइ (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        प्रत्यागम APIR_LOAD_LIBRARY_चिह्नम्_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    प्रत्यागम (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 वर्ण *                 dec_cur,
                                 ध्रुव वर्ण *           dec_end,
                                 वर्ण *                 enc_cur,
                                 ध्रुव वर्ण *           enc_end,
                                 वर्ण **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = मिच्छ,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = मिच्छ,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    जइ (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_गणनम्) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_गणनम्);
        प्रत्यागम APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    प्रत्यागम ret;
}
}
