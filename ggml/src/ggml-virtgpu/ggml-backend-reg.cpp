#அந்தர்பாவ "ggml-remoting.h"
#அந்தர்பாவ "ggml-virtgpu.h"

#அந்தர்பாவ <iostream>
#அந்தர்பாவ <mutex>

வெறுமை ggml_virtgpu_cleanup(virtgpu * gpu);

நிலையான virtgpu * apir_initialize() {
    நிலையான virtgpu *         gpu         = வெற்று;
    நிலையான std::atomic<தருக்க> initialized = பொய்;

    எனில் (initialized) {
        // fast track
        திரும்பு gpu;
    }

    {
        நிலையான std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        எனில் (initialized) {
            // thread safe
            திரும்பு gpu;
        }

        gpu = create_virtgpu();
        எனில் (!gpu) {
            initialized = உண்மை;
            திரும்பு வெற்று;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        எனில் (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_மொத்தம் = apir_device_get_மொத்தம்(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            எழுத்து * rmt_device_name = apir_device_get_name(gpu);
            எனில் (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (எழுத்து *) malloc(device_name_len);
            எனில் (!gpu->cached_device_info.name) {
                விடுவி(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            விடுவி(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_total);

        apir_buffer_type_host_handle_t buft_host_handle = apir_device_get_buffer_type(gpu);
        gpu->cached_buffer_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            எழுத்து * rmt_name = apir_buffer_type_get_name(gpu, buft_host_handle);
            எனில் (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (எழுத்து *) malloc(prefixed_len);
            எனில் (!gpu->cached_buffer_type.name) {
                விடுவி(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            விடுவி(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = உண்மை;
    }

    திரும்பு gpu;
}

நிலையான முழு ggml_backend_remoting_get_device_மொத்தம்() {
    virtgpu * gpu = apir_initialize();
    எனில் (!gpu) {
        திரும்பு 0;
    }

    திரும்பு gpu->cached_device_info.device_மொத்தம்;
}

நிலையான size_t ggml_backend_remoting_reg_get_device_மொத்தம்(ggml_backend_reg_t reg) {
    UNUSED(reg);

    திரும்பு ggml_backend_remoting_get_device_மொத்தம்();
}

நிலையான std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    திரும்பு devices[device];
}

நிலையான வெறுமை ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    எனில் (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        திரும்பு;
    }

    virtgpu * gpu = apir_initialize();
    எனில் (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        திரும்பு;
    }

    நிலையான std::atomic<தருக்க> initialized = பொய்;

    எனில் (initialized) {
        திரும்பு;  // fast track
    }

    {
        நிலையான std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        எனில் (!initialized) {
            க்கு (முழு i = 0; i < ggml_backend_remoting_get_device_மொத்தம்(); i++) {
                ggml_backend_remoting_device_context * ctx       = புதிய ggml_backend_remoting_device_context;
                எழுத்து                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = புதிய ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = உண்மை;
        }
    }
}

நிலையான ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    திரும்பு ggml_backend_remoting_get_device(device);
}

நிலையான மாறிலி எழுத்து * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    திரும்பு GGML_VIRTGPU_NAME;
}

நிலையான மாறிலி ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_மொத்தம்,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ வெற்று,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    எனில் (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    நிலையான ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    நிலையான தருக்க initialized = பொய்;
    எனில் (initialized) {
        திரும்பு &reg;
    }
    initialized = உண்மை;

    ggml_backend_remoting_reg_init_devices(&reg);

    திரும்பு &reg;
}

// public function, not exposed in the GGML interface at the moment
வெறுமை ggml_virtgpu_cleanup(virtgpu * gpu) {
    எனில் (gpu->cached_device_info.name) {
        விடுவி(gpu->cached_device_info.name);
        gpu->cached_device_info.name = வெற்று;
    }
    எனில் (gpu->cached_device_info.description) {
        விடுவி(gpu->cached_device_info.description);
        gpu->cached_device_info.description = வெற்று;
    }
    எனில் (gpu->cached_buffer_type.name) {
        விடுவி(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = வெற்று;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
