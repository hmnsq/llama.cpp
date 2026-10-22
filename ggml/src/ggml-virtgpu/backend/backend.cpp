#شامل "backend-dispatched.h"
#شامل "backend-virgl-apir.h"
#شامل "shared/api_remoting.h"
#شامل "shared/apir_backend.h"
#شامل "shared/apir_cs.h"

#شامل <dlfcn.h>
#شامل <ggml-backend.h>

#شامل <iostream>

#تعریف_کرو APIR_LLAMA_CPP_GGML_LIBRARY_راستہ_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#تعریف_کرو APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#تعریف_کرو APIR_LLAMA_CPP_LOG_مقصد_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#تعریف_کرو GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

ساکن خالی * backend_library_handle = تہی;
ساکن FILE * apir_logfile           = تہی;

ساکن خالی log_مقصد_file_callback(شمار ggml_log_level level, ثابت حرف * متن_2, خالی * user_ڈیٹا) {
    FILE * logfile = (FILE *) user_ڈیٹا;
    fprintf(logfile, "[%d] %s", level, متن_2);
    fflush(logfile);
}

خارجی "C" {
خالی apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_غیر_مستعمل(virgl_ctx_id);

    خودکار buffers = apir_get_track_backend_buffers();
    برائے (ثابت خودکار & بفر : buffers) {
        apir_untrack_backend_بفر(بفر);
        بفر->iface.free_بفر(بفر);
    }

    اگر (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = تہی;
    }

    اگر (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = تہی;
    }
}

#تعریف_کرو APIR_GGML_LIBRARY_راستہ_کلید "ggml.library.path"
#تعریف_کرو APIR_GGML_LIBRARY_REG_کلید  "ggml.library.reg"

Apir_بھرو_Library_Return_رمز apir_backend_initialize(uint32_t virgl_ctx_id, ساخت virgl_apir_callbacks * virgl_cbs) {
    ثابت حرف * dlsym_error;

    ثابت حرف * apir_log_مقصد_file = getenv(APIR_LLAMA_CPP_LOG_مقصد_FILE_ENV);
    اگر (apir_log_مقصد_file) {
        apir_logfile = fopen(apir_log_مقصد_file, "w");
        اگر (apir_logfile) {
            ggml_log_سیٹ(log_مقصد_file_callback, apir_logfile);
        } ورنہ {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_مقصد_file);
        }
    }

    ثابت حرف * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_راستہ_کلید);
    ثابت حرف * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_کلید);
    ثابت حرف * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    اگر (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_راستہ_ENV);

        واپس APIR_بھرو_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    اگر (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        واپس APIR_بھرو_LIBRARY_CANNOT_OPEN;
    }

    اگر (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        واپس APIR_بھرو_LIBRARY_ENV_VAR_MISSING;
    }

    خالی * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    اگر (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        واپس APIR_بھرو_LIBRARY_علامت_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    واپس (Apir_بھرو_Library_Return_رمز) (APIR_بھرو_LIBRARY_INIT_بنیاد_اشاریہ + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 حرف *                 dec_cur,
                                 ثابت حرف *           dec_انتہا,
                                 حرف *                 enc_cur,
                                 ثابت حرف *           enc_انتہا,
                                 حرف **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_انتہا,
        .fatal = جھوٹ,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_انتہا,
        .fatal = جھوٹ,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    اگر (cmd_type >= APIR_BACKEND_DISPATCH_جدول_گنتی) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_جدول_گنتی);
        واپس APIR_BACKEND_FORWARD_اشاریہ_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_جدول[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    واپس ret;
}
}
