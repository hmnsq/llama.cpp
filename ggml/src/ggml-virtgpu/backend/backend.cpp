#شامل "backend-dispatched.h"
#شامل "backend-virgl-apir.h"
#شامل "shared/api_remoting.h"
#شامل "shared/apir_backend.h"
#شامل "shared/apir_cs.h"

#شامل <dlfcn.h>
#شامل <ggml-backend.h>

#شامل <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

ایستا تهی * backend_library_handle = پوچ;
ایستا FILE * apir_logfile           = پوچ;

ایستا تهی log_to_file_callback(شمارش ggml_log_level level, ثابت نویسه * text, تهی * user_data) {
    FILE * logfile = (FILE *) user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

بیرونی "C" {
تهی apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    برای (ثابت auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    اگر (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = پوچ;
    }

    اگر (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = پوچ;
    }
}

#define APIR_GGML_LIBRARY_PATH_کلید "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_کلید  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, ساختار virgl_apir_callbacks * virgl_cbs) {
    ثابت نویسه * dlsym_error;

    ثابت نویسه * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    اگر (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        اگر (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } وگرنه {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    ثابت نویسه * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_کلید);
    ثابت نویسه * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_کلید);
    ثابت نویسه * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    اگر (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        بازگشت APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    اگر (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        بازگشت APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    اگر (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        بازگشت APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    تهی * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    اگر (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        بازگشت APIR_LOAD_LIBRARY_نماد_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    بازگشت (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 نویسه *                 dec_cur,
                                 ثابت نویسه *           dec_end,
                                 نویسه *                 enc_cur,
                                 ثابت نویسه *           enc_end,
                                 نویسه **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = نادرست,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = نادرست,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    اگر (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_شمار) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_شمار);
        بازگشت APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    بازگشت ret;
}
}
