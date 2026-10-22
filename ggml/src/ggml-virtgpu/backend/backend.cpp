#ضمن "backend-dispatched.h"
#ضمن "backend-virgl-apir.h"
#ضمن "shared/api_remoting.h"
#ضمن "shared/apir_backend.h"
#ضمن "shared/apir_cs.h"

#ضمن <dlfcn.h>
#ضمن <ggml-backend.h>

#ضمن <iostream>

#تعریف_کن APIR_LLAMA_CPP_GGML_LIBRARY_مسیر_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#تعریف_کن APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#تعریف_کن APIR_LLAMA_CPP_LOG_مقصد_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#تعریف_کن GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

ساكن فراغ * backend_library_handle = عدم;
ساكن FILE * apir_logfile           = عدم;

ساكن فراغ log_مقصد_file_callback(تعداد ggml_log_level level, ثابت حرف * متن_2, فراغ * user_داده) {
    FILE * logfile = (FILE *) user_داده;
    fprintf(logfile, "[%d] %s", level, متن_2);
    fflush(logfile);
}

خارجي "C" {
فراغ apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_استفادهنشده(virgl_ctx_id);

    خودکار buffers = apir_get_track_backend_buffers();
    لكل (ثابت خودکار & میانگیر : buffers) {
        apir_untrack_backend_میانگیر(میانگیر);
        میانگیر->iface.free_میانگیر(میانگیر);
    }

    إذا (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = عدم;
    }

    إذا (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = عدم;
    }
}

#تعریف_کن APIR_GGML_LIBRARY_مسیر_کلید "ggml.library.path"
#تعریف_کن APIR_GGML_LIBRARY_REG_کلید  "ggml.library.reg"

Apir_بارگیری_Library_Return_کد apir_backend_initialize(uint32_t virgl_ctx_id, بنية virgl_apir_callbacks * virgl_cbs) {
    ثابت حرف * dlsym_error;

    ثابت حرف * apir_log_مقصد_file = getenv(APIR_LLAMA_CPP_LOG_مقصد_FILE_ENV);
    إذا (apir_log_مقصد_file) {
        apir_logfile = fopen(apir_log_مقصد_file, "w");
        إذا (apir_logfile) {
            ggml_log_مجموعه(log_مقصد_file_callback, apir_logfile);
        } وإلا {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_مقصد_file);
        }
    }

    ثابت حرف * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_مسیر_کلید);
    ثابت حرف * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_کلید);
    ثابت حرف * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    إذا (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_مسیر_ENV);

        أرجع APIR_بارگیری_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    إذا (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        أرجع APIR_بارگیری_LIBRARY_CANNOT_OPEN;
    }

    إذا (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        أرجع APIR_بارگیری_LIBRARY_ENV_VAR_MISSING;
    }

    فراغ * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    إذا (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        أرجع APIR_بارگیری_LIBRARY_نماد_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    أرجع (Apir_بارگیری_Library_Return_کد) (APIR_بارگیری_LIBRARY_INIT_پایه_شاخص + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 حرف *                 dec_cur,
                                 ثابت حرف *           dec_انتها,
                                 حرف *                 enc_cur,
                                 ثابت حرف *           enc_انتها,
                                 حرف **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_انتها,
        .fatal = خطأ,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_انتها,
        .fatal = خطأ,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    إذا (cmd_type >= APIR_BACKEND_DISPATCH_جدول_شمار) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_جدول_شمار);
        أرجع APIR_BACKEND_FORWARD_شاخص_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_جدول[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    أرجع ret;
}
}
