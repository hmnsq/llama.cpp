#अन्तर्भाव "ggml-remoting.h"
#अन्तर्भाव "ggml-virtgpu.h"

#अन्तर्भाव <iostream>
#अन्तर्भाव <mutex>

शून्य ggml_virtgpu_cleanup(virtgpu * gpu);

स्थिर virtgpu * apir_initialize() {
    स्थिर virtgpu *         gpu         = रिक्त;
    स्थिर std::atomic<तार्किक> initialized = असत्य;

    यदि (initialized) {
        // fast track
        फर्काऊ gpu;
    }

    {
        स्थिर std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        यदि (initialized) {
            // thread safe
            फर्काऊ gpu;
        }

        gpu = create_virtgpu();
        यदि (!gpu) {
            initialized = सत्य;
            फर्काऊ रिक्त;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        यदि (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_गन्ती = apir_device_get_गन्ती(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            वर्ण * rmt_device_name = apir_device_get_name(gpu);
            यदि (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            माप_प्रकार device_name_len       = म_माप(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (वर्ण *) दो_स्मृति(device_name_len);
            यदि (!gpu->cached_device_info.name) {
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
            यदि (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            माप_प्रकार prefixed_len          = म_माप(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (वर्ण *) दो_स्मृति(prefixed_len);
            यदि (!gpu->cached_buffer_type.name) {
                मुक्त(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            मुक्त(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = सत्य;
    }

    फर्काऊ gpu;
}

स्थिर पूर्णाङ्क ggml_backend_remoting_get_device_गन्ती() {
    virtgpu * gpu = apir_initialize();
    यदि (!gpu) {
        फर्काऊ 0;
    }

    फर्काऊ gpu->cached_device_info.device_गन्ती;
}

स्थिर माप_प्रकार ggml_backend_remoting_reg_get_device_गन्ती(ggml_backend_reg_t reg) {
    UNUSED(reg);

    फर्काऊ ggml_backend_remoting_get_device_गन्ती();
}

स्थिर std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(माप_प्रकार device) {
    GGML_ASSERT(device < devices.size());
    फर्काऊ devices[device];
}

स्थिर शून्य ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    यदि (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        फर्काऊ;
    }

    virtgpu * gpu = apir_initialize();
    यदि (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        फर्काऊ;
    }

    स्थिर std::atomic<तार्किक> initialized = असत्य;

    यदि (initialized) {
        फर्काऊ;  // fast track
    }

    {
        स्थिर std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        यदि (!initialized) {
            लागि (पूर्णाङ्क i = 0; i < ggml_backend_remoting_get_device_गन्ती(); i++) {
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
            initialized = सत्य;
        }
    }
}

स्थिर ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार device) {
    UNUSED(reg);

    फर्काऊ ggml_backend_remoting_get_device(device);
}

स्थिर अचर वर्ण * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    फर्काऊ GGML_VIRTGPU_NAME;
}

स्थिर अचर ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_गन्ती,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ रिक्त,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    यदि (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    स्थिर ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    स्थिर तार्किक initialized = असत्य;
    यदि (initialized) {
        फर्काऊ &reg;
    }
    initialized = सत्य;

    ggml_backend_remoting_reg_init_devices(&reg);

    फर्काऊ &reg;
}

// public function, not exposed in the GGML interface at the moment
शून्य ggml_virtgpu_cleanup(virtgpu * gpu) {
    यदि (gpu->cached_device_info.name) {
        मुक्त(gpu->cached_device_info.name);
        gpu->cached_device_info.name = रिक्त;
    }
    यदि (gpu->cached_device_info.description) {
        मुक्त(gpu->cached_device_info.description);
        gpu->cached_device_info.description = रिक्त;
    }
    यदि (gpu->cached_buffer_type.name) {
        मुक्त(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = रिक्त;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
