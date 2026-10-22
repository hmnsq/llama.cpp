#شامل "ggml-remoting.h"
#شامل "ggml-virtgpu.h"

#شامل <iostream>
#شامل <mutex>

خالی ggml_virtgpu_cleanup(virtgpu * gpu);

ساکن virtgpu * apir_initialize() {
    ساکن virtgpu *         gpu         = تہی;
    ساکن std::atomic<منطقی> initialized = جھوٹ;

    اگر (initialized) {
        // fast track
        واپس gpu;
    }

    {
        ساکن std::mutex           mutex;
        std::lock_guard<std::mutex> تالا(mutex);

        اگر (initialized) {
            // thread safe
            واپس gpu;
        }

        gpu = create_virtgpu();
        اگر (!gpu) {
            initialized = سچ;
            واپس تہی;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        اگر (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_گنتی = apir_device_get_گنتی(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            حرف * rmt_device_name = apir_device_get_name(gpu);
            اگر (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = لڑی_لمبائی(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (حرف *) حافظہ_دو(device_name_len);
            اگر (!gpu->cached_device_info.name) {
                آزاد(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            آزاد(rmt_device_name);
        }

        apir_device_get_حافظہ(gpu, &gpu->cached_device_info.حافظہ_free, &gpu->cached_device_info.حافظہ_کل);

        apir_بفر_type_host_handle_t buft_host_handle = apir_device_get_بفر_type(gpu);
        gpu->cached_بفر_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            حرف * rmt_name = apir_بفر_type_get_name(gpu, buft_host_handle);
            اگر (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = لڑی_لمبائی(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_بفر_type.name = (حرف *) حافظہ_دو(prefixed_len);
            اگر (!gpu->cached_بفر_type.name) {
                آزاد(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_بفر_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            آزاد(rmt_name);
        }

        gpu->cached_بفر_type.alignment = apir_بفر_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_بفر_type.max_size  = apir_بفر_type_get_max_سائز(gpu, buft_host_handle);

        initialized = سچ;
    }

    واپس gpu;
}

ساکن صحیح ggml_backend_remoting_get_device_گنتی() {
    virtgpu * gpu = apir_initialize();
    اگر (!gpu) {
        واپس 0;
    }

    واپس gpu->cached_device_info.device_گنتی;
}

ساکن size_t ggml_backend_remoting_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    غیر_مستعمل(reg);

    واپس ggml_backend_remoting_get_device_گنتی();
}

ساکن std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    واپس devices[device];
}

ساکن خالی ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    اگر (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        واپس;
    }

    virtgpu * gpu = apir_initialize();
    اگر (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        واپس;
    }

    ساکن std::atomic<منطقی> initialized = جھوٹ;

    اگر (initialized) {
        واپس;  // fast track
    }

    {
        ساکن std::mutex           mutex;
        std::lock_guard<std::mutex> تالا(mutex);
        اگر (!initialized) {
            برائے (صحیح i = 0; i < ggml_backend_remoting_get_device_گنتی(); i++) {
                ggml_backend_remoting_device_context * ctx       = نیا ggml_backend_remoting_device_context;
                حرف                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = نیا ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = سچ;
        }
    }
}

ساکن ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    غیر_مستعمل(reg);

    واپس ggml_backend_remoting_get_device(device);
}

ساکن ثابت حرف * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    غیر_مستعمل(reg);

    واپس GGML_VIRTGPU_NAME;
}

ساکن ثابت ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_گنتی,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ تہی,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    اگر (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    ساکن ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخہ,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    ساکن منطقی initialized = جھوٹ;
    اگر (initialized) {
        واپس &reg;
    }
    initialized = سچ;

    ggml_backend_remoting_reg_init_devices(&reg);

    واپس &reg;
}

// public function, not exposed in the GGML interface at the moment
خالی ggml_virtgpu_cleanup(virtgpu * gpu) {
    اگر (gpu->cached_device_info.name) {
        آزاد(gpu->cached_device_info.name);
        gpu->cached_device_info.name = تہی;
    }
    اگر (gpu->cached_device_info.description) {
        آزاد(gpu->cached_device_info.description);
        gpu->cached_device_info.description = تہی;
    }
    اگر (gpu->cached_بفر_type.name) {
        آزاد(gpu->cached_بفر_type.name);
        gpu->cached_بفر_type.name = تہی;
    }

    mtx_destroy(&gpu->ڈیٹا_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
