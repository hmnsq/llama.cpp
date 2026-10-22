#అన్తర్భావ "backend-dispatched.h"
#అన్తర్భావ "backend-virgl-apir.h"
#అన్తర్భావ "shared/api_remoting.h"
#అన్తర్భావ "shared/apir_backend.h"
#అన్తర్భావ "shared/apir_cs.h"

#అన్తర్భావ <dlfcn.h>
#అన్తర్భావ <ggml-backend.h>

#అన్తర్భావ <iostream>

#నిర్వచించు APIR_LLAMA_CPP_GGML_LIBRARY_మార్గం_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#నిర్వచించు APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#నిర్వచించు APIR_LLAMA_CPP_LOG_గమ్యం_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#నిర్వచించు GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

స్థిర శూన్య * backend_library_handle = ఖాళీ;
స్థిర FILE * apir_logfile           = ఖాళీ;

స్థిర శూన్య log_గమ్యం_file_callback(గణన ggml_log_level level, స్థిరరాశి అక్షరం * పాఠ్యం_2, శూన్య * user_దత్తాంశం) {
    FILE * logfile = (FILE *) user_దత్తాంశం;
    fprintf(logfile, "[%d] %s", level, పాఠ్యం_2);
    fflush(logfile);
}

బాహ్య "C" {
శూన్య apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_వాడని(virgl_ctx_id);

    స్వయంచాలక buffers = apir_get_track_backend_buffers();
    కొరకు (స్థిరరాశి స్వయంచాలక & బఫరు : buffers) {
        apir_untrack_backend_బఫరు(బఫరు);
        బఫరు->iface.free_బఫరు(బఫరు);
    }

    ఒకవేళ (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = ఖాళీ;
    }

    ఒకవేళ (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = ఖాళీ;
    }
}

#నిర్వచించు APIR_GGML_LIBRARY_మార్గం_కీలకం "ggml.library.path"
#నిర్వచించు APIR_GGML_LIBRARY_REG_కీలకం  "ggml.library.reg"

Apir_ఎక్కించు_Library_Return_సంకేతం apir_backend_initialize(uint32_t virgl_ctx_id, నిర్మాణం virgl_apir_callbacks * virgl_cbs) {
    స్థిరరాశి అక్షరం * dlsym_error;

    స్థిరరాశి అక్షరం * apir_log_గమ్యం_file = getenv(APIR_LLAMA_CPP_LOG_గమ్యం_FILE_ENV);
    ఒకవేళ (apir_log_గమ్యం_file) {
        apir_logfile = fopen(apir_log_గమ్యం_file, "w");
        ఒకవేళ (apir_logfile) {
            ggml_log_సమితి(log_గమ్యం_file_callback, apir_logfile);
        } లేకపోతే {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_గమ్యం_file);
        }
    }

    స్థిరరాశి అక్షరం * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_మార్గం_కీలకం);
    స్థిరరాశి అక్షరం * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_కీలకం);
    స్థిరరాశి అక్షరం * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    ఒకవేళ (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_మార్గం_ENV);

        తిరిగి APIR_ఎక్కించు_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    ఒకవేళ (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        తిరిగి APIR_ఎక్కించు_LIBRARY_CANNOT_OPEN;
    }

    ఒకవేళ (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        తిరిగి APIR_ఎక్కించు_LIBRARY_ENV_VAR_MISSING;
    }

    శూన్య * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    ఒకవేళ (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        తిరిగి APIR_ఎక్కించు_LIBRARY_చిహ్నం_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    తిరిగి (Apir_ఎక్కించు_Library_Return_సంకేతం) (APIR_ఎక్కించు_LIBRARY_INIT_పునాది_సూచీ + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 అక్షరం *                 dec_cur,
                                 స్థిరరాశి అక్షరం *           dec_చివర,
                                 అక్షరం *                 enc_cur,
                                 స్థిరరాశి అక్షరం *           enc_చివర,
                                 అక్షరం **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_చివర,
        .fatal = అబద్ధం,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_చివర,
        .fatal = అబద్ధం,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    ఒకవేళ (cmd_type >= APIR_BACKEND_DISPATCH_పట్టిక_లెక్క) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_పట్టిక_లెక్క);
        తిరిగి APIR_BACKEND_FORWARD_సూచీ_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_పట్టిక[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    తిరిగి ret;
}
}
