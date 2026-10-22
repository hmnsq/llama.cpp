#அந்தர்பாவ "backend-dispatched.h"
#அந்தர்பாவ "backend-virgl-apir.h"
#அந்தர்பாவ "shared/api_remoting.h"
#அந்தர்பாவ "shared/apir_backend.h"
#அந்தர்பாவ "shared/apir_cs.h"

#அந்தர்பாவ <dlfcn.h>
#அந்தர்பாவ <ggml-backend.h>

#அந்தர்பாவ <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

நிலையான வெறுமை * backend_library_handle = வெற்று;
நிலையான FILE * apir_logfile           = வெற்று;

நிலையான வெறுமை log_to_file_callback(எண்ணிக்கை ggml_log_level level, மாறிலி எழுத்து * text, வெறுமை * user_data) {
    FILE * logfile = (FILE *) user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

வெளி "C" {
வெறுமை apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    க்கு (மாறிலி auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    எனில் (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = வெற்று;
    }

    எனில் (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = வெற்று;
    }
}

#define APIR_GGML_LIBRARY_PATH_சாவி "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_சாவி  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, கட்டமைப்பு virgl_apir_callbacks * virgl_cbs) {
    மாறிலி எழுத்து * dlsym_error;

    மாறிலி எழுத்து * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    எனில் (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        எனில் (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } இல்லையெனில் {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    மாறிலி எழுத்து * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_சாவி);
    மாறிலி எழுத்து * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_சாவி);
    மாறிலி எழுத்து * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    எனில் (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        திரும்பு APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    எனில் (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        திரும்பு APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    எனில் (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        திரும்பு APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    வெறுமை * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    எனில் (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        திரும்பு APIR_LOAD_LIBRARY_குறியீடு_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    திரும்பு (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 எழுத்து *                 dec_cur,
                                 மாறிலி எழுத்து *           dec_end,
                                 எழுத்து *                 enc_cur,
                                 மாறிலி எழுத்து *           enc_end,
                                 எழுத்து **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = பொய்,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = பொய்,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    எனில் (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_மொத்தம்) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_மொத்தம்);
        திரும்பு APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    திரும்பு ret;
}
}
