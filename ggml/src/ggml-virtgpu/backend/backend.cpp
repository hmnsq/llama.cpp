#অন্তর্ভাৱ "backend-dispatched.h"
#অন্তর্ভাৱ "backend-virgl-apir.h"
#অন্তর্ভাৱ "shared/api_remoting.h"
#অন্তর্ভাৱ "shared/apir_backend.h"
#অন্তর্ভাৱ "shared/apir_cs.h"

#অন্তর্ভাৱ <dlfcn.h>
#অন্তর্ভাৱ <ggml-backend.h>

#অন্তর্ভাৱ <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

স্থির শূন্য * backend_library_handle = রিক্ত;
স্থির FILE * apir_logfile           = রিক্ত;

স্থির শূন্য log_to_file_callback(গণনা ggml_log_level level, ধ্রুৱ ৱর্ণ * text, শূন্য * user_data) {
    FILE * logfile = (FILE *) user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

বাহ্য "C" {
শূন্য apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    বাবে (ধ্রুৱ auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    যদি (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = রিক্ত;
    }

    যদি (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = রিক্ত;
    }
}

#define APIR_GGML_LIBRARY_PATH_চাবি "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_চাবি  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, সংরচনা virgl_apir_callbacks * virgl_cbs) {
    ধ্রুৱ ৱর্ণ * dlsym_error;

    ধ্রুৱ ৱর্ণ * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    যদি (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        যদি (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } অন্যথা {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    ধ্রুৱ ৱর্ণ * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_চাবি);
    ধ্রুৱ ৱর্ণ * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_চাবি);
    ধ্রুৱ ৱর্ণ * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    যদি (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        ঘূৰাই APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    যদি (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        ঘূৰাই APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    যদি (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        ঘূৰাই APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    শূন্য * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    যদি (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        ঘূৰাই APIR_LOAD_LIBRARY_প্ৰতীক_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    ঘূৰাই (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 ৱর্ণ *                 dec_cur,
                                 ধ্রুৱ ৱর্ণ *           dec_end,
                                 ৱর্ণ *                 enc_cur,
                                 ধ্রুৱ ৱর্ণ *           enc_end,
                                 ৱর্ণ **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = মিছা,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = মিছা,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    যদি (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_গুনতি) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_গুনতি);
        ঘূৰাই APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    ঘূৰাই ret;
}
}
