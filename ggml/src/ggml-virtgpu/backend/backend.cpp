#ଅନ୍ତର୍ଭାଵ "backend-dispatched.h"
#ଅନ୍ତର୍ଭାଵ "backend-virgl-apir.h"
#ଅନ୍ତର୍ଭାଵ "shared/api_remoting.h"
#ଅନ୍ତର୍ଭାଵ "shared/apir_backend.h"
#ଅନ୍ତର୍ଭାଵ "shared/apir_cs.h"

#ଅନ୍ତର୍ଭାଵ <dlfcn.h>
#ଅନ୍ତର୍ଭାଵ <ggml-backend.h>

#ଅନ୍ତର୍ଭାଵ <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

ସ୍ଥିର ଶୂନ୍ୟ * backend_library_handle = ଖାଲି;
ସ୍ଥିର FILE * apir_logfile           = ଖାଲି;

ସ୍ଥିର ଶୂନ୍ୟ log_to_file_callback(ଗଣନା ggml_log_level level, ଅଚଳ ଅକ୍ଷର * text, ଶୂନ୍ୟ * user_data) {
    FILE * logfile = (FILE *) user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

ବାହ୍ୟ "C" {
ଶୂନ୍ୟ apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    ପାଇଁ (ଅଚଳ auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    ଯଦି (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = ଖାଲି;
    }

    ଯଦି (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = ଖାଲି;
    }
}

#define APIR_GGML_LIBRARY_PATH_ଚାବି "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_ଚାବି  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, ସଂରଚନା virgl_apir_callbacks * virgl_cbs) {
    ଅଚଳ ଅକ୍ଷର * dlsym_error;

    ଅଚଳ ଅକ୍ଷର * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    ଯଦି (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        ଯଦି (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } ଅନ୍ୟଥା {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    ଅଚଳ ଅକ୍ଷର * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_ଚାବି);
    ଅଚଳ ଅକ୍ଷର * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_ଚାବି);
    ଅଚଳ ଅକ୍ଷର * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    ଯଦି (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        ଫେରନ୍ତୁ APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    ଯଦି (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        ଫେରନ୍ତୁ APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    ଯଦି (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        ଫେରନ୍ତୁ APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    ଶୂନ୍ୟ * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    ଯଦି (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        ଫେରନ୍ତୁ APIR_LOAD_LIBRARY_ପ୍ରତୀକ_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    ଫେରନ୍ତୁ (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 ଅକ୍ଷର *                 dec_cur,
                                 ଅଚଳ ଅକ୍ଷର *           dec_end,
                                 ଅକ୍ଷର *                 enc_cur,
                                 ଅଚଳ ଅକ୍ଷର *           enc_end,
                                 ଅକ୍ଷର **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = ମିଥ୍ୟା,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = ମିଥ୍ୟା,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    ଯଦି (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_ଗଣନା) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_ଗଣନା);
        ଫେରନ୍ତୁ APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    ଫେରନ୍ତୁ ret;
}
}
