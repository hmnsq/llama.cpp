#समावेश "ggml-remoting.h"
#समावेश "ggml-virtgpu.h"

#समावेश <iostream>
#समावेश <mutex>

व्योम ggml_virtgpu_cleanup(virtgpu * gpu);

अटल virtgpu * apir_initialize() {
    अटल virtgpu *         gpu         = शून्य;
    अटल std::atomic<bool> initialized = false;

    अगर (initialized) {
        // fast track
        वापस gpu;
    }

    {
        अटल std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        अगर (initialized) {
            // thread safe
            वापस gpu;
        }

        gpu = create_virtgpu();
        अगर (!gpu) {
            initialized = true;
            वापस शून्य;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        अगर (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_गिनती = apir_device_get_गिनती(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            अक्षर * rmt_device_name = apir_device_get_name(gpu);
            अगर (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            माप_प्रकार device_name_len       = म_माप(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (अक्षर *) दो_स्मृति(device_name_len);
            अगर (!gpu->cached_device_info.name) {
                मुक्त(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            मुक्त(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_योग);

        apir_बफर_type_host_handle_t buft_host_handle = apir_device_get_बफर_type(gpu);
        gpu->cached_बफर_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            अक्षर * rmt_name = apir_बफर_type_get_name(gpu, buft_host_handle);
            अगर (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            माप_प्रकार prefixed_len          = म_माप(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_बफर_type.name = (अक्षर *) दो_स्मृति(prefixed_len);
            अगर (!gpu->cached_बफर_type.name) {
                मुक्त(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_बफर_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            मुक्त(rmt_name);
        }

        gpu->cached_बफर_type.alignment = apir_बफर_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_बफर_type.max_size  = apir_बफर_type_get_max_आकार(gpu, buft_host_handle);

        initialized = true;
    }

    वापस gpu;
}

अटल पूर्णांक ggml_backend_remoting_get_device_गिनती() {
    virtgpu * gpu = apir_initialize();
    अगर (!gpu) {
        वापस 0;
    }

    वापस gpu->cached_device_info.device_गिनती;
}

अटल माप_प्रकार ggml_backend_remoting_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    UNUSED(reg);

    वापस ggml_backend_remoting_get_device_गिनती();
}

अटल std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(माप_प्रकार device) {
    GGML_ASSERT(device < devices.size());
    वापस devices[device];
}

अटल व्योम ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    अगर (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        वापस;
    }

    virtgpu * gpu = apir_initialize();
    अगर (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        वापस;
    }

    अटल std::atomic<bool> initialized = false;

    अगर (initialized) {
        वापस;  // fast track
    }

    {
        अटल std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        अगर (!initialized) {
            क्रम (पूर्णांक i = 0; i < ggml_backend_remoting_get_device_गिनती(); i++) {
                ggml_backend_remoting_device_context * ctx       = नया ggml_backend_remoting_device_context;
                अक्षर                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = नया ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = true;
        }
    }
}

अटल ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार device) {
    UNUSED(reg);

    वापस ggml_backend_remoting_get_device(device);
}

अटल स्थिर अक्षर * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    वापस GGML_VIRTGPU_NAME;
}

अटल स्थिर ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_गिनती,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ शून्य,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    अगर (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    अटल ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    अटल bool initialized = false;
    अगर (initialized) {
        वापस &reg;
    }
    initialized = true;

    ggml_backend_remoting_reg_init_devices(&reg);

    वापस &reg;
}

// public function, not exposed in the GGML interface at the moment
व्योम ggml_virtgpu_cleanup(virtgpu * gpu) {
    अगर (gpu->cached_device_info.name) {
        मुक्त(gpu->cached_device_info.name);
        gpu->cached_device_info.name = शून्य;
    }
    अगर (gpu->cached_device_info.description) {
        मुक्त(gpu->cached_device_info.description);
        gpu->cached_device_info.description = शून्य;
    }
    अगर (gpu->cached_बफर_type.name) {
        मुक्त(gpu->cached_बफर_type.name);
        gpu->cached_बफर_type.name = शून्य;
    }

    mtx_destroy(&gpu->तथ्य_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
