#समावेश "backend-dispatched.h"
#समावेश "backend-virgl-apir.h"
#समावेश "shared/api_remoting.h"
#समावेश "shared/apir_backend.h"
#समावेश "shared/apir_cs.h"

#समावेश <dlfcn.h>
#समावेश <ggml-backend.h>

#समावेश <iostream>

#घोषणा APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#घोषणा APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#घोषणा APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#घोषणा GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

अटल व्योम * backend_library_handle = शून्य;
अटल खाता * apir_logfile           = शून्य;

अटल व्योम log_to_file_callback(क्रमागत ggml_log_level level, स्थिर अक्षर * text, व्योम * user_तथ्य) {
    खाता * logfile = (खाता *) user_तथ्य;
    ख_लिखो(logfile, "[%d] %s", level, text);
    ख_साफ(logfile);
}

बाह्य "C" {
व्योम apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    स्वतः buffers = apir_get_track_backend_buffers();
    क्रम (स्थिर स्वतः & बफर : buffers) {
        apir_untrack_backend_बफर(बफर);
        बफर->iface.free_बफर(बफर);
    }

    अगर (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = शून्य;
    }

    अगर (apir_logfile) {
        ख_बंद(apir_logfile);
        apir_logfile = शून्य;
    }
}

#घोषणा APIR_GGML_LIBRARY_PATH_कुंजी "ggml.library.path"
#घोषणा APIR_GGML_LIBRARY_REG_कुंजी  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, काष्ठा virgl_apir_callbacks * virgl_cbs) {
    स्थिर अक्षर * dlsym_error;

    स्थिर अक्षर * apir_log_to_file = दो_पर्या(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    अगर (apir_log_to_file) {
        apir_logfile = ख_खोलो(apir_log_to_file, "w");
        अगर (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } अन्यथा {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    स्थिर अक्षर * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_कुंजी);
    स्थिर अक्षर * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_कुंजी);
    स्थिर अक्षर * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    अगर (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        वापस APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    अगर (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        वापस APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    अगर (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        वापस APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    व्योम * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    अगर (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        वापस APIR_LOAD_LIBRARY_प्रतीक_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    वापस (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_सूचकांक + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 अक्षर *                 dec_cur,
                                 स्थिर अक्षर *           dec_end,
                                 अक्षर *                 enc_cur,
                                 स्थिर अक्षर *           enc_end,
                                 अक्षर **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = false,
    };

    apir_decoder डेसी = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = false,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    अगर (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_गिनती) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_गिनती);
        वापस APIR_BACKEND_FORWARD_सूचकांक_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &डेसी, &ctx);

    *enc_cur_after = enc.cur;

    वापस ret;
}
}
