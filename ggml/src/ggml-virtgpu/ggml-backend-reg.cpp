#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-remoting.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-virtgpu.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>

ਖਾਲੀ ggml_virtgpu_cleanup(virtgpu * gpu);

ਸਥਿਰ virtgpu * apir_initialize() {
    ਸਥਿਰ virtgpu *         gpu         = ਨਿਰਾ;
    ਸਥਿਰ std::atomic<ਤਾਰਕਿਕ> initialized = ਝੂਠ;

    ਜੇ (initialized) {
        // fast track
        ਵਾਪਸ gpu;
    }

    {
        ਸਥਿਰ std::mutex           mutex;
        std::lock_guard<std::mutex> ਤਾਲਾ(mutex);

        ਜੇ (initialized) {
            // thread safe
            ਵਾਪਸ gpu;
        }

        gpu = create_virtgpu();
        ਜੇ (!gpu) {
            initialized = ਸੱਚ;
            ਵਾਪਸ ਨਿਰਾ;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        ਜੇ (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_ਗਿਣਤੀ = apir_device_get_ਗਿਣਤੀ(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            ਅੱਖਰ * rmt_device_name = apir_device_get_name(gpu);
            ਜੇ (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = ਲੜੀ_ਲੰਬਾਈ(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (ਅੱਖਰ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(device_name_len);
            ਜੇ (!gpu->cached_device_info.name) {
                ਮੁਕਤ(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            ਮੁਕਤ(rmt_device_name);
        }

        apir_device_get_ਸਮ੍ਰਿਤੀ(gpu, &gpu->cached_device_info.ਸਮ੍ਰਿਤੀ_free, &gpu->cached_device_info.ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);

        apir_ਬਫਰ_type_host_handle_t buft_host_handle = apir_device_get_ਬਫਰ_type(gpu);
        gpu->cached_ਬਫਰ_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            ਅੱਖਰ * rmt_name = apir_ਬਫਰ_type_get_name(gpu, buft_host_handle);
            ਜੇ (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = ਲੜੀ_ਲੰਬਾਈ(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_ਬਫਰ_type.name = (ਅੱਖਰ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(prefixed_len);
            ਜੇ (!gpu->cached_ਬਫਰ_type.name) {
                ਮੁਕਤ(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_ਬਫਰ_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            ਮੁਕਤ(rmt_name);
        }

        gpu->cached_ਬਫਰ_type.alignment = apir_ਬਫਰ_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_ਬਫਰ_type.max_size  = apir_ਬਫਰ_type_get_max_ਮਾਪ(gpu, buft_host_handle);

        initialized = ਸੱਚ;
    }

    ਵਾਪਸ gpu;
}

ਸਥਿਰ ਪੂਰਨ ggml_backend_remoting_get_device_ਗਿਣਤੀ() {
    virtgpu * gpu = apir_initialize();
    ਜੇ (!gpu) {
        ਵਾਪਸ 0;
    }

    ਵਾਪਸ gpu->cached_device_info.device_ਗਿਣਤੀ;
}

ਸਥਿਰ size_t ggml_backend_remoting_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ਅਣਵਰਤਿਆ(reg);

    ਵਾਪਸ ggml_backend_remoting_get_device_ਗਿਣਤੀ();
}

ਸਥਿਰ std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    ਵਾਪਸ devices[device];
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    ਜੇ (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        ਵਾਪਸ;
    }

    virtgpu * gpu = apir_initialize();
    ਜੇ (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        ਵਾਪਸ;
    }

    ਸਥਿਰ std::atomic<ਤਾਰਕਿਕ> initialized = ਝੂਠ;

    ਜੇ (initialized) {
        ਵਾਪਸ;  // fast track
    }

    {
        ਸਥਿਰ std::mutex           mutex;
        std::lock_guard<std::mutex> ਤਾਲਾ(mutex);
        ਜੇ (!initialized) {
            ਲਈ (ਪੂਰਨ i = 0; i < ggml_backend_remoting_get_device_ਗਿਣਤੀ(); i++) {
                ggml_backend_remoting_device_context * ctx       = ਨਵਾਂ ggml_backend_remoting_device_context;
                ਅੱਖਰ                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = ਨਵਾਂ ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = ਸੱਚ;
        }
    }
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    ਅਣਵਰਤਿਆ(reg);

    ਵਾਪਸ ggml_backend_remoting_get_device(device);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    ਅਣਵਰਤਿਆ(reg);

    ਵਾਪਸ GGML_VIRTGPU_NAME;
}

ਸਥਿਰ ਅਚਲ ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ ਨਿਰਾ,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    ਜੇ (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    ਸਥਿਰ ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;
    ਜੇ (initialized) {
        ਵਾਪਸ &reg;
    }
    initialized = ਸੱਚ;

    ggml_backend_remoting_reg_init_devices(&reg);

    ਵਾਪਸ &reg;
}

// public function, not exposed in the GGML interface at the moment
ਖਾਲੀ ggml_virtgpu_cleanup(virtgpu * gpu) {
    ਜੇ (gpu->cached_device_info.name) {
        ਮੁਕਤ(gpu->cached_device_info.name);
        gpu->cached_device_info.name = ਨਿਰਾ;
    }
    ਜੇ (gpu->cached_device_info.description) {
        ਮੁਕਤ(gpu->cached_device_info.description);
        gpu->cached_device_info.description = ਨਿਰਾ;
    }
    ਜੇ (gpu->cached_ਬਫਰ_type.name) {
        ਮੁਕਤ(gpu->cached_ਬਫਰ_type.name);
        gpu->cached_ਬਫਰ_type.name = ਨਿਰਾ;
    }

    mtx_destroy(&gpu->ਅੰਕੜੇ_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
