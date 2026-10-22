#അന്തര്ഭാവ "backend-dispatched.h"
#അന്തര്ഭാവ "backend-virgl-apir.h"
#അന്തര്ഭാവ "shared/api_remoting.h"
#അന്തര്ഭാവ "shared/apir_backend.h"
#അന്തര്ഭാവ "shared/apir_cs.h"

#അന്തര്ഭാവ <dlfcn.h>
#അന്തര്ഭാവ <ggml-backend.h>

#അന്തര്ഭാവ <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

സ്ഥിതം ശൂന്യം * backend_library_handle = ശൂന്യ;
സ്ഥിതം FILE * apir_logfile           = ശൂന്യ;

സ്ഥിതം ശൂന്യം log_to_file_callback(എണ്ണൽ ggml_log_level level, സ്ഥിരം അക്ഷരം * text, ശൂന്യം * user_data) {
    FILE * logfile = (FILE *) user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

ബാഹ്യ "C" {
ശൂന്യം apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    വേണ്ടി (സ്ഥിരം auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    എങ്കിൽ (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = ശൂന്യ;
    }

    എങ്കിൽ (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = ശൂന്യ;
    }
}

#define APIR_GGML_LIBRARY_PATH_താക്കോൽ "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_താക്കോൽ  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, ഘടന virgl_apir_callbacks * virgl_cbs) {
    സ്ഥിരം അക്ഷരം * dlsym_error;

    സ്ഥിരം അക്ഷരം * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    എങ്കിൽ (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        എങ്കിൽ (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } അല്ലെങ്കിൽ {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    സ്ഥിരം അക്ഷരം * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_താക്കോൽ);
    സ്ഥിരം അക്ഷരം * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_താക്കോൽ);
    സ്ഥിരം അക്ഷരം * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    എങ്കിൽ (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        തിരികെ APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    എങ്കിൽ (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        തിരികെ APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    എങ്കിൽ (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        തിരികെ APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    ശൂന്യം * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    എങ്കിൽ (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        തിരികെ APIR_LOAD_LIBRARY_ചിഹ്നം_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    തിരികെ (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 അക്ഷരം *                 dec_cur,
                                 സ്ഥിരം അക്ഷരം *           dec_end,
                                 അക്ഷരം *                 enc_cur,
                                 സ്ഥിരം അക്ഷരം *           enc_end,
                                 അക്ഷരം **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = അസത്യം,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = അസത്യം,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    എങ്കിൽ (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_എണ്ണം) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_എണ്ണം);
        തിരികെ APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    തിരികെ ret;
}
}
