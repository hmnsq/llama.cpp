#అన్తర్భావ "ggml-remoting.h"
#అన్తర్భావ "ggml-virtgpu.h"

#అన్తర్భావ <iostream>
#అన్తర్భావ <mutex>

శూన్య ggml_virtgpu_cleanup(virtgpu * gpu);

స్థిర virtgpu * apir_initialize() {
    స్థిర virtgpu *         gpu         = ఖాళీ;
    స్థిర std::atomic<తార్కిక> initialized = అబద్ధం;

    ఒకవేళ (initialized) {
        // fast track
        తిరిగి gpu;
    }

    {
        స్థిర std::mutex           mutex;
        std::lock_guard<std::mutex> తాళం(mutex);

        ఒకవేళ (initialized) {
            // thread safe
            తిరిగి gpu;
        }

        gpu = create_virtgpu();
        ఒకవేళ (!gpu) {
            initialized = నిజం;
            తిరిగి ఖాళీ;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        ఒకవేళ (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_లెక్క = apir_device_get_లెక్క(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            అక్షరం * rmt_device_name = apir_device_get_name(gpu);
            ఒకవేళ (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = అక్షరశ్రేణి_పొడవు(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (అక్షరం *) జ్ఞాపకం_ఇవ్వు(device_name_len);
            ఒకవేళ (!gpu->cached_device_info.name) {
                విడుదల(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            విడుదల(rmt_device_name);
        }

        apir_device_get_జ్ఞాపకం(gpu, &gpu->cached_device_info.జ్ఞాపకం_free, &gpu->cached_device_info.జ్ఞాపకం_మొత్తం);

        apir_బఫరు_type_host_handle_t buft_host_handle = apir_device_get_బఫరు_type(gpu);
        gpu->cached_బఫరు_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            అక్షరం * rmt_name = apir_బఫరు_type_get_name(gpu, buft_host_handle);
            ఒకవేళ (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = అక్షరశ్రేణి_పొడవు(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_బఫరు_type.name = (అక్షరం *) జ్ఞాపకం_ఇవ్వు(prefixed_len);
            ఒకవేళ (!gpu->cached_బఫరు_type.name) {
                విడుదల(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_బఫరు_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            విడుదల(rmt_name);
        }

        gpu->cached_బఫరు_type.alignment = apir_బఫరు_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_బఫరు_type.max_size  = apir_బఫరు_type_get_max_కొలత(gpu, buft_host_handle);

        initialized = నిజం;
    }

    తిరిగి gpu;
}

స్థిర పూర్ణాంకం ggml_backend_remoting_get_device_లెక్క() {
    virtgpu * gpu = apir_initialize();
    ఒకవేళ (!gpu) {
        తిరిగి 0;
    }

    తిరిగి gpu->cached_device_info.device_లెక్క;
}

స్థిర size_t ggml_backend_remoting_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    వాడని(reg);

    తిరిగి ggml_backend_remoting_get_device_లెక్క();
}

స్థిర std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    తిరిగి devices[device];
}

స్థిర శూన్య ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    ఒకవేళ (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        తిరిగి;
    }

    virtgpu * gpu = apir_initialize();
    ఒకవేళ (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        తిరిగి;
    }

    స్థిర std::atomic<తార్కిక> initialized = అబద్ధం;

    ఒకవేళ (initialized) {
        తిరిగి;  // fast track
    }

    {
        స్థిర std::mutex           mutex;
        std::lock_guard<std::mutex> తాళం(mutex);
        ఒకవేళ (!initialized) {
            కొరకు (పూర్ణాంకం i = 0; i < ggml_backend_remoting_get_device_లెక్క(); i++) {
                ggml_backend_remoting_device_context * ctx       = కొత్త ggml_backend_remoting_device_context;
                అక్షరం                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = కొత్త ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = నిజం;
        }
    }
}

స్థిర ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    వాడని(reg);

    తిరిగి ggml_backend_remoting_get_device(device);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    వాడని(reg);

    తిరిగి GGML_VIRTGPU_NAME;
}

స్థిర స్థిరరాశి ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_లెక్క,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ ఖాళీ,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    ఒకవేళ (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    స్థిర ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    స్థిర తార్కిక initialized = అబద్ధం;
    ఒకవేళ (initialized) {
        తిరిగి &reg;
    }
    initialized = నిజం;

    ggml_backend_remoting_reg_init_devices(&reg);

    తిరిగి &reg;
}

// public function, not exposed in the GGML interface at the moment
శూన్య ggml_virtgpu_cleanup(virtgpu * gpu) {
    ఒకవేళ (gpu->cached_device_info.name) {
        విడుదల(gpu->cached_device_info.name);
        gpu->cached_device_info.name = ఖాళీ;
    }
    ఒకవేళ (gpu->cached_device_info.description) {
        విడుదల(gpu->cached_device_info.description);
        gpu->cached_device_info.description = ఖాళీ;
    }
    ఒకవేళ (gpu->cached_బఫరు_type.name) {
        విడుదల(gpu->cached_బఫరు_type.name);
        gpu->cached_బఫరు_type.name = ఖాళీ;
    }

    mtx_destroy(&gpu->దత్తాంశం_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
