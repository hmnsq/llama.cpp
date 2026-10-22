#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"
#ಅನ್ತರ್ಭಾವ "ggml-virtgpu.h"

#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <mutex>

ಶೂನ್ಯ ggml_virtgpu_cleanup(virtgpu * gpu);

ಸ್ಥಿರ virtgpu * apir_initialize() {
    ಸ್ಥಿರ virtgpu *         gpu         = ಖಾಲಿ;
    ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ> initialized = ಸುಳ್ಳು;

    ಆದರೆ (initialized) {
        // fast track
        ಹಿಂತಿರುಗಿ gpu;
    }

    {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> ಬೀಗ(mutex);

        ಆದರೆ (initialized) {
            // thread safe
            ಹಿಂತಿರುಗಿ gpu;
        }

        gpu = create_virtgpu();
        ಆದರೆ (!gpu) {
            initialized = ನಿಜ;
            ಹಿಂತಿರುಗಿ ಖಾಲಿ;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        ಆದರೆ (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_ಎಣಿಕೆ = apir_device_get_ಎಣಿಕೆ(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            ಅಕ್ಷರ * rmt_device_name = apir_device_get_name(gpu);
            ಆದರೆ (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (ಅಕ್ಷರ *) ಸ್ಮರಣೆ_ಕೊಡು(device_name_len);
            ಆದರೆ (!gpu->cached_device_info.name) {
                ಬಿಡುಗಡೆ(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            ಬಿಡುಗಡೆ(rmt_device_name);
        }

        apir_device_get_ಸ್ಮರಣೆ(gpu, &gpu->cached_device_info.ಸ್ಮರಣೆ_free, &gpu->cached_device_info.ಸ್ಮರಣೆ_ಒಟ್ಟು);

        apir_ಬಫರ್_type_host_handle_t buft_host_handle = apir_device_get_ಬಫರ್_type(gpu);
        gpu->cached_ಬಫರ್_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            ಅಕ್ಷರ * rmt_name = apir_ಬಫರ್_type_get_name(gpu, buft_host_handle);
            ಆದರೆ (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_ಬಫರ್_type.name = (ಅಕ್ಷರ *) ಸ್ಮರಣೆ_ಕೊಡು(prefixed_len);
            ಆದರೆ (!gpu->cached_ಬಫರ್_type.name) {
                ಬಿಡುಗಡೆ(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_ಬಫರ್_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            ಬಿಡುಗಡೆ(rmt_name);
        }

        gpu->cached_ಬಫರ್_type.alignment = apir_ಬಫರ್_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_ಬಫರ್_type.max_size  = apir_ಬಫರ್_type_get_max_ಗಾತ್ರ(gpu, buft_host_handle);

        initialized = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ gpu;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_remoting_get_device_ಎಣಿಕೆ() {
    virtgpu * gpu = apir_initialize();
    ಆದರೆ (!gpu) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಹಿಂತಿರುಗಿ gpu->cached_device_info.device_ಎಣಿಕೆ;
}

ಸ್ಥಿರ size_t ggml_backend_remoting_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಬಳಸದ(reg);

    ಹಿಂತಿರುಗಿ ggml_backend_remoting_get_device_ಎಣಿಕೆ();
}

ಸ್ಥಿರ std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    ಹಿಂತಿರುಗಿ devices[device];
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    ಆದರೆ (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        ಹಿಂತಿರುಗಿ;
    }

    virtgpu * gpu = apir_initialize();
    ಆದರೆ (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ> initialized = ಸುಳ್ಳು;

    ಆದರೆ (initialized) {
        ಹಿಂತಿರುಗಿ;  // fast track
    }

    {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> ಬೀಗ(mutex);
        ಆದರೆ (!initialized) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_backend_remoting_get_device_ಎಣಿಕೆ(); i++) {
                ggml_backend_remoting_device_context * ctx       = ಹೊಸ ggml_backend_remoting_device_context;
                ಅಕ್ಷರ                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = ಹೊಸ ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = ನಿಜ;
        }
    }
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    ಬಳಸದ(reg);

    ಹಿಂತಿರುಗಿ ggml_backend_remoting_get_device(device);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    ಬಳಸದ(reg);

    ಹಿಂತಿರುಗಿ GGML_VIRTGPU_NAME;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ ಖಾಲಿ,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    ಆದರೆ (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    ಸ್ಥಿರ ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;
    ಆದರೆ (initialized) {
        ಹಿಂತಿರುಗಿ &reg;
    }
    initialized = ನಿಜ;

    ggml_backend_remoting_reg_init_devices(&reg);

    ಹಿಂತಿರುಗಿ &reg;
}

// public function, not exposed in the GGML interface at the moment
ಶೂನ್ಯ ggml_virtgpu_cleanup(virtgpu * gpu) {
    ಆದರೆ (gpu->cached_device_info.name) {
        ಬಿಡುಗಡೆ(gpu->cached_device_info.name);
        gpu->cached_device_info.name = ಖಾಲಿ;
    }
    ಆದರೆ (gpu->cached_device_info.description) {
        ಬಿಡುಗಡೆ(gpu->cached_device_info.description);
        gpu->cached_device_info.description = ಖಾಲಿ;
    }
    ಆದರೆ (gpu->cached_ಬಫರ್_type.name) {
        ಬಿಡುಗಡೆ(gpu->cached_ಬಫರ್_type.name);
        gpu->cached_ಬಫರ್_type.name = ಖಾಲಿ;
    }

    mtx_destroy(&gpu->ದತ್ತಾಂಶ_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
