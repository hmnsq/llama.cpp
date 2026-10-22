#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"
#ಅನ್ತರ್ಭಾವ "ggml-virtgpu.h"

#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <mutex>

ಶೂನ್ಯ ggml_virtgpu_cleanup(virtgpu * gpu);

ಸ್ಥಿರ virtgpu * apir_initialize() {
    ಸ್ಥಿರ virtgpu *         gpu         = ರಿಕ್ತ;
    ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ> initialized = ಅಸತ್ಯ;

    ಯದಿ (initialized) {
        // fast track
        ಪ್ರತ್ಯಾಗಮ gpu;
    }

    {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ಯದಿ (initialized) {
            // thread safe
            ಪ್ರತ್ಯಾಗಮ gpu;
        }

        gpu = create_virtgpu();
        ಯದಿ (!gpu) {
            initialized = ಸತ್ಯ;
            ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        ಯದಿ (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_ಎಣಿಕೆ = apir_device_get_ಎಣಿಕೆ(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            ವರ್ಣ * rmt_device_name = apir_device_get_name(gpu);
            ಯದಿ (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (ವರ್ಣ *) malloc(device_name_len);
            ಯದಿ (!gpu->cached_device_info.name) {
                ಮುಕ್ತ(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            ಮುಕ್ತ(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_total);

        apir_buffer_type_host_handle_t buft_host_handle = apir_device_get_buffer_type(gpu);
        gpu->cached_buffer_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            ವರ್ಣ * rmt_name = apir_buffer_type_get_name(gpu, buft_host_handle);
            ಯದಿ (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (ವರ್ಣ *) malloc(prefixed_len);
            ಯದಿ (!gpu->cached_buffer_type.name) {
                ಮುಕ್ತ(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            ಮುಕ್ತ(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ gpu;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_remoting_get_device_ಎಣಿಕೆ() {
    virtgpu * gpu = apir_initialize();
    ಯದಿ (!gpu) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಪ್ರತ್ಯಾಗಮ gpu->cached_device_info.device_ಎಣಿಕೆ;
}

ಸ್ಥಿರ size_t ggml_backend_remoting_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    UNUSED(reg);

    ಪ್ರತ್ಯಾಗಮ ggml_backend_remoting_get_device_ಎಣಿಕೆ();
}

ಸ್ಥಿರ std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    ಪ್ರತ್ಯಾಗಮ devices[device];
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    ಯದಿ (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        ಪ್ರತ್ಯಾಗಮ;
    }

    virtgpu * gpu = apir_initialize();
    ಯದಿ (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ> initialized = ಅಸತ್ಯ;

    ಯದಿ (initialized) {
        ಪ್ರತ್ಯಾಗಮ;  // fast track
    }

    {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        ಯದಿ (!initialized) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ggml_backend_remoting_get_device_ಎಣಿಕೆ(); i++) {
                ggml_backend_remoting_device_context * ctx       = ನೂತನ ggml_backend_remoting_device_context;
                ವರ್ಣ                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = ನೂತನ ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = ಸತ್ಯ;
        }
    }
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    ಪ್ರತ್ಯಾಗಮ ggml_backend_remoting_get_device(device);
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    ಪ್ರತ್ಯಾಗಮ GGML_VIRTGPU_NAME;
}

ಸ್ಥಿರ ಧ್ರುವ ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ ರಿಕ್ತ,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    ಯದಿ (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    ಸ್ಥಿರ ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಅಸತ್ಯ;
    ಯದಿ (initialized) {
        ಪ್ರತ್ಯಾಗಮ &reg;
    }
    initialized = ಸತ್ಯ;

    ggml_backend_remoting_reg_init_devices(&reg);

    ಪ್ರತ್ಯಾಗಮ &reg;
}

// public function, not exposed in the GGML interface at the moment
ಶೂನ್ಯ ggml_virtgpu_cleanup(virtgpu * gpu) {
    ಯದಿ (gpu->cached_device_info.name) {
        ಮುಕ್ತ(gpu->cached_device_info.name);
        gpu->cached_device_info.name = ರಿಕ್ತ;
    }
    ಯದಿ (gpu->cached_device_info.description) {
        ಮುಕ್ತ(gpu->cached_device_info.description);
        gpu->cached_device_info.description = ರಿಕ್ತ;
    }
    ಯದಿ (gpu->cached_buffer_type.name) {
        ಮುಕ್ತ(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = ರಿಕ್ತ;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
