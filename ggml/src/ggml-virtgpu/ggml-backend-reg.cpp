#അന്തര്ഭാവ "ggml-remoting.h"
#അന്തര്ഭാവ "ggml-virtgpu.h"

#അന്തര്ഭാവ <iostream>
#അന്തര്ഭാവ <mutex>

ശൂന്യം ggml_virtgpu_cleanup(virtgpu * gpu);

സ്ഥിതം virtgpu * apir_initialize() {
    സ്ഥിതം virtgpu *         gpu         = ശൂന്യ;
    സ്ഥിതം std::atomic<തർക്കം> initialized = അസത്യം;

    എങ്കിൽ (initialized) {
        // fast track
        തിരികെ gpu;
    }

    {
        സ്ഥിതം std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        എങ്കിൽ (initialized) {
            // thread safe
            തിരികെ gpu;
        }

        gpu = create_virtgpu();
        എങ്കിൽ (!gpu) {
            initialized = സത്യം;
            തിരികെ ശൂന്യ;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        എങ്കിൽ (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_എണ്ണം = apir_device_get_എണ്ണം(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            അക്ഷരം * rmt_device_name = apir_device_get_name(gpu);
            എങ്കിൽ (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (അക്ഷരം *) malloc(device_name_len);
            എങ്കിൽ (!gpu->cached_device_info.name) {
                സ്വതന്ത്രം(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            സ്വതന്ത്രം(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_total);

        apir_buffer_type_host_handle_t buft_host_handle = apir_device_get_buffer_type(gpu);
        gpu->cached_buffer_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            അക്ഷരം * rmt_name = apir_buffer_type_get_name(gpu, buft_host_handle);
            എങ്കിൽ (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (അക്ഷരം *) malloc(prefixed_len);
            എങ്കിൽ (!gpu->cached_buffer_type.name) {
                സ്വതന്ത്രം(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            സ്വതന്ത്രം(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = സത്യം;
    }

    തിരികെ gpu;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_remoting_get_device_എണ്ണം() {
    virtgpu * gpu = apir_initialize();
    എങ്കിൽ (!gpu) {
        തിരികെ 0;
    }

    തിരികെ gpu->cached_device_info.device_എണ്ണം;
}

സ്ഥിതം size_t ggml_backend_remoting_reg_get_device_എണ്ണം(ggml_backend_reg_t reg) {
    UNUSED(reg);

    തിരികെ ggml_backend_remoting_get_device_എണ്ണം();
}

സ്ഥിതം std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    തിരികെ devices[device];
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    എങ്കിൽ (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        തിരികെ;
    }

    virtgpu * gpu = apir_initialize();
    എങ്കിൽ (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        തിരികെ;
    }

    സ്ഥിതം std::atomic<തർക്കം> initialized = അസത്യം;

    എങ്കിൽ (initialized) {
        തിരികെ;  // fast track
    }

    {
        സ്ഥിതം std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        എങ്കിൽ (!initialized) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_backend_remoting_get_device_എണ്ണം(); i++) {
                ggml_backend_remoting_device_context * ctx       = പുതിയ ggml_backend_remoting_device_context;
                അക്ഷരം                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = പുതിയ ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = സത്യം;
        }
    }
}

സ്ഥിതം ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    തിരികെ ggml_backend_remoting_get_device(device);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    തിരികെ GGML_VIRTGPU_NAME;
}

സ്ഥിതം സ്ഥിരം ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_എണ്ണം,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ ശൂന്യ,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    എങ്കിൽ (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    സ്ഥിതം ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    സ്ഥിതം തർക്കം initialized = അസത്യം;
    എങ്കിൽ (initialized) {
        തിരികെ &reg;
    }
    initialized = സത്യം;

    ggml_backend_remoting_reg_init_devices(&reg);

    തിരികെ &reg;
}

// public function, not exposed in the GGML interface at the moment
ശൂന്യം ggml_virtgpu_cleanup(virtgpu * gpu) {
    എങ്കിൽ (gpu->cached_device_info.name) {
        സ്വതന്ത്രം(gpu->cached_device_info.name);
        gpu->cached_device_info.name = ശൂന്യ;
    }
    എങ്കിൽ (gpu->cached_device_info.description) {
        സ്വതന്ത്രം(gpu->cached_device_info.description);
        gpu->cached_device_info.description = ശൂന്യ;
    }
    എങ്കിൽ (gpu->cached_buffer_type.name) {
        സ്വതന്ത്രം(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = ശൂന്യ;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
