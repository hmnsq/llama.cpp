#include "backend-dispatched.h"
#include "backend-virgl-apir.h"
#include "shared/api_remoting.h"
#include "shared/apir_backend.h"
#include "shared/apir_cs.h"

#include <dlfcn.h>
#include <ggml-backend.h>

#include <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

static void * backend_library_handle = NULL;
static FILE * apir_logfile           = NULL;

static void log_to_file_callback(ga_nanaa ggml_log_level level, const var_na * text, void * user_data) {
    FILE * logfile = (FILE *) user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

extern "C" {
void apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    ꦏꦁꦒꦺꦴ (const auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    ꦪꦼꦤ꧀ (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = NULL;
    }

    ꦪꦼꦤ꧀ (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = NULL;
    }
}

#define APIR_GGML_LIBRARY_PATH_KEY "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_KEY  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, sa_nracanaa virgl_apir_callbacks * virgl_cbs) {
    const var_na * dlsym_error;

    const var_na * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    ꦪꦼꦤ꧀ (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        ꦪꦼꦤ꧀ (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } ꦭꦶꦪ {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    const var_na * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_KEY);
    const var_na * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_KEY);
    const var_na * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    ꦪꦼꦤ꧀ (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        ꦮꦭꦶ APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    ꦪꦼꦤ꧀ (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        ꦮꦭꦶ APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    ꦪꦼꦤ꧀ (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        ꦮꦭꦶ APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    void * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    ꦪꦼꦤ꧀ (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        ꦮꦭꦶ APIR_LOAD_LIBRARY_SYMBOL_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    ꦮꦭꦶ (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 var_na *                 dec_cur,
                                 const var_na *           dec_end,
                                 var_na *                 enc_cur,
                                 const var_na *           enc_end,
                                 var_na **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = false,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = false,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    ꦪꦼꦤ꧀ (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_COUNT) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_COUNT);
        ꦮꦭꦶ APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    ꦮꦭꦶ ret;
}
}
