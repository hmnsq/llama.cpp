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
स्थिर खाता * apir_logfile           = रिक्त;

स्थिर शून्य log_to_file_callback(गणना ggml_log_level level, अचर वर्ण * text, शून्य * user_data) {
    खाता * logfile = (खाता *) user_data;
    ख_लिखो(logfile, "[%d] %s", level, text);
    ख_साफ(logfile);
}

बाह्य "C" {
शून्य apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    स्वतः buffers = apir_get_track_backend_buffers();
    लागि (अचर स्वतः & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    यदि (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = रिक्त;
    }

    यदि (apir_logfile) {
        ख_बंद(apir_logfile);
        apir_logfile = रिक्त;
    }
}

#define APIR_GGML_LIBRARY_PATH_साँचो "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_साँचो  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, संरचना virgl_apir_callbacks * virgl_cbs) {
    अचर वर्ण * dlsym_error;

    अचर वर्ण * apir_log_to_file = दो_पर्या(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    यदि (apir_log_to_file) {
        apir_logfile = ख_खोलो(apir_log_to_file, "w");
        यदि (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } अन्यथा {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    अचर वर्ण * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_साँचो);
    अचर वर्ण * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_साँचो);
    अचर वर्ण * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    यदि (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        फर्काऊ APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    यदि (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        फर्काऊ APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    यदि (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        फर्काऊ APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    शून्य * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    यदि (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        फर्काऊ APIR_LOAD_LIBRARY_चिह्न_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    फर्काऊ (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 वर्ण *                 dec_cur,
                                 अचर वर्ण *           dec_end,
                                 वर्ण *                 enc_cur,
                                 अचर वर्ण *           enc_end,
                                 वर्ण **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = असत्य,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = असत्य,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    यदि (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_गन्ती) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_गन्ती);
        फर्काऊ APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    फर्काऊ ret;
}
}
