#شامل "ggml-remoting.h"
#شامل "ggml-virtgpu.h"

#شامل <iostream>
#شامل <mutex>

تهی ggml_virtgpu_cleanup(virtgpu * gpu);

ایستا virtgpu * apir_initialize() {
    ایستا virtgpu *         gpu         = پوچ;
    ایستا std::atomic<منطقی> initialized = نادرست;

    اگر (initialized) {
        // fast track
        بازگشت gpu;
    }

    {
        ایستا std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        اگر (initialized) {
            // thread safe
            بازگشت gpu;
        }

        gpu = create_virtgpu();
        اگر (!gpu) {
            initialized = درست;
            بازگشت پوچ;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        اگر (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_شمار = apir_device_get_شمار(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            نویسه * rmt_device_name = apir_device_get_name(gpu);
            اگر (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (نویسه *) malloc(device_name_len);
            اگر (!gpu->cached_device_info.name) {
                آزاد(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            آزاد(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_total);

        apir_buffer_type_host_handle_t buft_host_handle = apir_device_get_buffer_type(gpu);
        gpu->cached_buffer_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            نویسه * rmt_name = apir_buffer_type_get_name(gpu, buft_host_handle);
            اگر (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (نویسه *) malloc(prefixed_len);
            اگر (!gpu->cached_buffer_type.name) {
                آزاد(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            آزاد(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = درست;
    }

    بازگشت gpu;
}

ایستا صحیح ggml_backend_remoting_get_device_شمار() {
    virtgpu * gpu = apir_initialize();
    اگر (!gpu) {
        بازگشت 0;
    }

    بازگشت gpu->cached_device_info.device_شمار;
}

ایستا size_t ggml_backend_remoting_reg_get_device_شمار(ggml_backend_reg_t reg) {
    UNUSED(reg);

    بازگشت ggml_backend_remoting_get_device_شمار();
}

ایستا std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    بازگشت devices[device];
}

ایستا تهی ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    اگر (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        بازگشت;
    }

    virtgpu * gpu = apir_initialize();
    اگر (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        بازگشت;
    }

    ایستا std::atomic<منطقی> initialized = نادرست;

    اگر (initialized) {
        بازگشت;  // fast track
    }

    {
        ایستا std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        اگر (!initialized) {
            برای (صحیح i = 0; i < ggml_backend_remoting_get_device_شمار(); i++) {
                ggml_backend_remoting_device_context * ctx       = جدید ggml_backend_remoting_device_context;
                نویسه                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = جدید ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = درست;
        }
    }
}

ایستا ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    بازگشت ggml_backend_remoting_get_device(device);
}

ایستا ثابت نویسه * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    بازگشت GGML_VIRTGPU_NAME;
}

ایستا ثابت ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_شمار,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ پوچ,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    اگر (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    ایستا ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    ایستا منطقی initialized = نادرست;
    اگر (initialized) {
        بازگشت &reg;
    }
    initialized = درست;

    ggml_backend_remoting_reg_init_devices(&reg);

    بازگشت &reg;
}

// public function, not exposed in the GGML interface at the moment
تهی ggml_virtgpu_cleanup(virtgpu * gpu) {
    اگر (gpu->cached_device_info.name) {
        آزاد(gpu->cached_device_info.name);
        gpu->cached_device_info.name = پوچ;
    }
    اگر (gpu->cached_device_info.description) {
        آزاد(gpu->cached_device_info.description);
        gpu->cached_device_info.description = پوچ;
    }
    اگر (gpu->cached_buffer_type.name) {
        آزاد(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = پوچ;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
