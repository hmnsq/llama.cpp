#અન્તર્ભાવ "backend-dispatched.h"
#અન્તર્ભાવ "backend-virgl-apir.h"
#અન્તર્ભાવ "shared/api_remoting.h"
#અન્તર્ભાવ "shared/apir_backend.h"
#અન્તર્ભાવ "shared/apir_cs.h"

#અન્તર્ભાવ <dlfcn.h>
#અન્તર્ભાવ <ggml-backend.h>

#અન્તર્ભાવ <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

સ્થિર શૂન્ય * backend_library_handle = ખાલી;
સ્થિર FILE * apir_logfile           = ખાલી;

સ્થિર શૂન્ય log_to_file_callback(ગણના ggml_log_level level, અચલ અક્ષર * text, શૂન્ય * user_data) {
    FILE * logfile = (FILE *) user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

બાહ્ય "C" {
શૂન્ય apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    માટે (અચલ auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    જો (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = ખાલી;
    }

    જો (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = ખાલી;
    }
}

#define APIR_GGML_LIBRARY_PATH_ચાવી "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_ચાવી  "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, સંરચના virgl_apir_callbacks * virgl_cbs) {
    અચલ અક્ષર * dlsym_error;

    અચલ અક્ષર * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    જો (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        જો (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } નહીંતર {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    અચલ અક્ષર * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_ચાવી);
    અચલ અક્ષર * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_ચાવી);
    અચલ અક્ષર * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    જો (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        પાછા APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    જો (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        પાછા APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    જો (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        પાછા APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    શૂન્ય * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    જો (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        પાછા APIR_LOAD_LIBRARY_પ્રતીક_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    પાછા (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 અક્ષર *                 dec_cur,
                                 અચલ અક્ષર *           dec_end,
                                 અક્ષર *                 enc_cur,
                                 અચલ અક્ષર *           enc_end,
                                 અક્ષર **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = ખોટું,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = ખોટું,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    જો (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_ગણતરી) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_ગણતરી);
        પાછા APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    પાછા ret;
}
}
