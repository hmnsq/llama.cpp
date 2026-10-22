#अन्तर्भाव "ggml-remoting.h"
#अन्तर्भाव "ggml-virtgpu.h"

#अन्तर्भाव <iostream>
#अन्तर्भाव <mutex>

शून्य ggml_virtgpu_cleanup(virtgpu * gpu);

स्थिर virtgpu * apir_initialize() {
    स्थिर virtgpu *         gpu         = रिक्त;
    स्थिर std::atomic<तार्किक> initialized = मिच्छा;

    सचे (initialized) {
        // fast track
        पटिनिवत्त gpu;
    }

    {
        स्थिर std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        सचे (initialized) {
            // thread safe
            पटिनिवत्त gpu;
        }

        gpu = create_virtgpu();
        सचे (!gpu) {
            initialized = सच्च;
            पटिनिवत्त रिक्त;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        सचे (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_गणनम् = apir_device_get_गणनम्(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            वर्ण * rmt_device_name = apir_device_get_name(gpu);
            सचे (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (वर्ण *) malloc(device_name_len);
            सचे (!gpu->cached_device_info.name) {
                मुक्त(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            मुक्त(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_total);

        apir_buffer_type_host_handle_t buft_host_handle = apir_device_get_buffer_type(gpu);
        gpu->cached_buffer_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            वर्ण * rmt_name = apir_buffer_type_get_name(gpu, buft_host_handle);
            सचे (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (वर्ण *) malloc(prefixed_len);
            सचे (!gpu->cached_buffer_type.name) {
                मुक्त(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            मुक्त(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = सच्च;
    }

    पटिनिवत्त gpu;
}

स्थिर पूर्णाङ्क ggml_backend_remoting_get_device_गणनम्() {
    virtgpu * gpu = apir_initialize();
    सचे (!gpu) {
        पटिनिवत्त 0;
    }

    पटिनिवत्त gpu->cached_device_info.device_गणनम्;
}

स्थिर size_t ggml_backend_remoting_reg_get_device_गणनम्(ggml_backend_reg_t reg) {
    UNUSED(reg);

    पटिनिवत्त ggml_backend_remoting_get_device_गणनम्();
}

स्थिर std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    पटिनिवत्त devices[device];
}

स्थिर शून्य ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    सचे (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        पटिनिवत्त;
    }

    virtgpu * gpu = apir_initialize();
    सचे (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        पटिनिवत्त;
    }

    स्थिर std::atomic<तार्किक> initialized = मिच्छा;

    सचे (initialized) {
        पटिनिवत्त;  // fast track
    }

    {
        स्थिर std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        सचे (!initialized) {
            अत्थाय (पूर्णाङ्क i = 0; i < ggml_backend_remoting_get_device_गणनम्(); i++) {
                ggml_backend_remoting_device_context * ctx       = नूतन ggml_backend_remoting_device_context;
                वर्ण                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = नूतन ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = सच्च;
        }
    }
}

स्थिर ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    पटिनिवत्त ggml_backend_remoting_get_device(device);
}

स्थिर ध्रुव वर्ण * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    पटिनिवत्त GGML_VIRTGPU_NAME;
}

स्थिर ध्रुव ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_गणनम्,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ रिक्त,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    सचे (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    स्थिर ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    स्थिर तार्किक initialized = मिच्छा;
    सचे (initialized) {
        पटिनिवत्त &reg;
    }
    initialized = सच्च;

    ggml_backend_remoting_reg_init_devices(&reg);

    पटिनिवत्त &reg;
}

// public function, not exposed in the GGML interface at the moment
शून्य ggml_virtgpu_cleanup(virtgpu * gpu) {
    सचे (gpu->cached_device_info.name) {
        मुक्त(gpu->cached_device_info.name);
        gpu->cached_device_info.name = रिक्त;
    }
    सचे (gpu->cached_device_info.description) {
        मुक्त(gpu->cached_device_info.description);
        gpu->cached_device_info.description = रिक्त;
    }
    सचे (gpu->cached_buffer_type.name) {
        मुक्त(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = रिक्त;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
