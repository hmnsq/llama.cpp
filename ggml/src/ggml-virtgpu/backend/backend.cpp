#অন্তর্ভাব "backend-dispatched.h"
#অন্তর্ভাব "backend-virgl-apir.h"
#অন্তর্ভাব "shared/api_remoting.h"
#অন্তর্ভাব "shared/apir_backend.h"
#অন্তর্ভাব "shared/apir_cs.h"

#অন্তর্ভাব <dlfcn.h>
#অন্তর্ভাব <ggml-backend.h>

#অন্তর্ভাব <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

স্থির শূন্য * backend_library_handle = ফাঁকা;
স্থির FILE * apir_logfile           = ফাঁকা;

স্থির শূন্য log_to_file_callback(গণনা ggml_log_level level, ধ্রুবক অক্ষর * text, শূন্য * user_তথ্য) {
    FILE * logfile = (FILE *) user_তথ্য;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

বাহ্যিক "C" {
শূন্য apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    জন্য (ধ্রুবক auto & বাফার : buffers) {
        apir_untrack_backend_বাফার(বাফার);
        বাফার->iface.free_বাফার(বাফার);
    }

    যদি (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = ফাঁকা;
    }

    যদি (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = ফাঁকা;
    }
}

#define APIR_GGML_LIBRARY_PATH_চাবি "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_চাবি  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, কাঠামো virgl_apir_callbacks * virgl_cbs) {
    ধ্রুবক অক্ষর * dlsym_error;

    ধ্রুবক অক্ষর * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    যদি (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        যদি (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } অন্যথা {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    ধ্রুবক অক্ষর * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_চাবি);
    ধ্রুবক অক্ষর * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_চাবি);
    ধ্রুবক অক্ষর * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    যদি (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        ফেরত APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    যদি (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        ফেরত APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    যদি (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        ফেরত APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    শূন্য * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    যদি (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        ফেরত APIR_LOAD_LIBRARY_প্রতীক_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    ফেরত (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_সূচক + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 অক্ষর *                 dec_cur,
                                 ধ্রুবক অক্ষর *           dec_end,
                                 অক্ষর *                 enc_cur,
                                 ধ্রুবক অক্ষর *           enc_end,
                                 অক্ষর **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = মিথ্যা,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = মিথ্যা,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    যদি (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_গুনতি) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_গুনতি);
        ফেরত APIR_BACKEND_FORWARD_সূচক_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    ফেরত ret;
}
}
