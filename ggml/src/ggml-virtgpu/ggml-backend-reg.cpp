#અન્તર્ભાવ "ggml-remoting.h"
#અન્તર્ભાવ "ggml-virtgpu.h"

#અન્તર્ભાવ <iostream>
#અન્તર્ભાવ <mutex>

શૂન્ય ggml_virtgpu_cleanup(virtgpu * gpu);

સ્થિર virtgpu * apir_initialize() {
    સ્થિર virtgpu *         gpu         = ખાલી;
    સ્થિર std::atomic<તાર્કિક> initialized = ખોટું;

    જો (initialized) {
        // fast track
        પાછા gpu;
    }

    {
        સ્થિર std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        જો (initialized) {
            // thread safe
            પાછા gpu;
        }

        gpu = create_virtgpu();
        જો (!gpu) {
            initialized = સાચું;
            પાછા ખાલી;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        જો (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_ગણતરી = apir_device_get_ગણતરી(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            અક્ષર * rmt_device_name = apir_device_get_name(gpu);
            જો (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (અક્ષર *) malloc(device_name_len);
            જો (!gpu->cached_device_info.name) {
                મુક્ત(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            મુક્ત(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_total);

        apir_buffer_type_host_handle_t buft_host_handle = apir_device_get_buffer_type(gpu);
        gpu->cached_buffer_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            અક્ષર * rmt_name = apir_buffer_type_get_name(gpu, buft_host_handle);
            જો (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (અક્ષર *) malloc(prefixed_len);
            જો (!gpu->cached_buffer_type.name) {
                મુક્ત(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            મુક્ત(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = સાચું;
    }

    પાછા gpu;
}

સ્થિર પૂર્ણાંક ggml_backend_remoting_get_device_ગણતરી() {
    virtgpu * gpu = apir_initialize();
    જો (!gpu) {
        પાછા 0;
    }

    પાછા gpu->cached_device_info.device_ગણતરી;
}

સ્થિર size_t ggml_backend_remoting_reg_get_device_ગણતરી(ggml_backend_reg_t reg) {
    UNUSED(reg);

    પાછા ggml_backend_remoting_get_device_ગણતરી();
}

સ્થિર std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    પાછા devices[device];
}

સ્થિર શૂન્ય ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    જો (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        પાછા;
    }

    virtgpu * gpu = apir_initialize();
    જો (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        પાછા;
    }

    સ્થિર std::atomic<તાર્કિક> initialized = ખોટું;

    જો (initialized) {
        પાછા;  // fast track
    }

    {
        સ્થિર std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        જો (!initialized) {
            માટે (પૂર્ણાંક i = 0; i < ggml_backend_remoting_get_device_ગણતરી(); i++) {
                ggml_backend_remoting_device_context * ctx       = નવું ggml_backend_remoting_device_context;
                અક્ષર                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = નવું ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = સાચું;
        }
    }
}

સ્થિર ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    પાછા ggml_backend_remoting_get_device(device);
}

સ્થિર અચલ અક્ષર * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    પાછા GGML_VIRTGPU_NAME;
}

સ્થિર અચલ ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_ગણતરી,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ ખાલી,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    જો (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    સ્થિર ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    સ્થિર તાર્કિક initialized = ખોટું;
    જો (initialized) {
        પાછા &reg;
    }
    initialized = સાચું;

    ggml_backend_remoting_reg_init_devices(&reg);

    પાછા &reg;
}

// public function, not exposed in the GGML interface at the moment
શૂન્ય ggml_virtgpu_cleanup(virtgpu * gpu) {
    જો (gpu->cached_device_info.name) {
        મુક્ત(gpu->cached_device_info.name);
        gpu->cached_device_info.name = ખાલી;
    }
    જો (gpu->cached_device_info.description) {
        મુક્ત(gpu->cached_device_info.description);
        gpu->cached_device_info.description = ખાલી;
    }
    જો (gpu->cached_buffer_type.name) {
        મુક્ત(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = ખાલી;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
