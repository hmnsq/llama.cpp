#ضمن "ggml-remoting.h"
#ضمن "ggml-virtgpu.h"

#ضمن <iostream>
#ضمن <mutex>

فراغ ggml_virtgpu_cleanup(virtgpu * gpu);

ساكن virtgpu * apir_initialize() {
    ساكن virtgpu *         gpu         = عدم;
    ساكن std::atomic<منطقي> initialized = خطأ;

    𐡄𐡍 (initialized) {
        // fast track
        أرجع gpu;
    }

    {
        ساكن std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        𐡄𐡍 (initialized) {
            // thread safe
            أرجع gpu;
        }

        gpu = create_virtgpu();
        𐡄𐡍 (!gpu) {
            initialized = 𐡒𐡔𐡈;
            أرجع عدم;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        𐡄𐡍 (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_count = apir_device_get_count(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            حرف * rmt_device_name = apir_device_get_name(gpu);
            𐡄𐡍 (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (حرف *) malloc(device_name_len);
            𐡄𐡍 (!gpu->cached_device_info.name) {
                حرر(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            حرر(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_total);

        apir_buffer_type_host_handle_t buft_host_handle = apir_device_get_buffer_type(gpu);
        gpu->cached_buffer_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            حرف * rmt_name = apir_buffer_type_get_name(gpu, buft_host_handle);
            𐡄𐡍 (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (حرف *) malloc(prefixed_len);
            𐡄𐡍 (!gpu->cached_buffer_type.name) {
                حرر(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            حرر(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = 𐡒𐡔𐡈;
    }

    أرجع gpu;
}

ساكن صحيح ggml_backend_remoting_get_device_count() {
    virtgpu * gpu = apir_initialize();
    𐡄𐡍 (!gpu) {
        أرجع 0;
    }

    أرجع gpu->cached_device_info.device_count;
}

ساكن size_t ggml_backend_remoting_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);

    أرجع ggml_backend_remoting_get_device_count();
}

ساكن std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    أرجع devices[device];
}

ساكن فراغ ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    𐡄𐡍 (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        أرجع;
    }

    virtgpu * gpu = apir_initialize();
    𐡄𐡍 (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        أرجع;
    }

    ساكن std::atomic<منطقي> initialized = خطأ;

    𐡄𐡍 (initialized) {
        أرجع;  // fast track
    }

    {
        ساكن std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        𐡄𐡍 (!initialized) {
            لكل (صحيح i = 0; i < ggml_backend_remoting_get_device_count(); i++) {
                ggml_backend_remoting_device_context * ctx       = جديد ggml_backend_remoting_device_context;
                حرف                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = جديد ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = 𐡒𐡔𐡈;
        }
    }
}

ساكن ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    أرجع ggml_backend_remoting_get_device(device);
}

ساكن ثابت حرف * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    أرجع GGML_VIRTGPU_NAME;
}

ساكن ثابت ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_count,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ عدم,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    𐡄𐡍 (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    ساكن ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    ساكن منطقي initialized = خطأ;
    𐡄𐡍 (initialized) {
        أرجع &reg;
    }
    initialized = 𐡒𐡔𐡈;

    ggml_backend_remoting_reg_init_devices(&reg);

    أرجع &reg;
}

// public function, not exposed in the GGML interface at the moment
فراغ ggml_virtgpu_cleanup(virtgpu * gpu) {
    𐡄𐡍 (gpu->cached_device_info.name) {
        حرر(gpu->cached_device_info.name);
        gpu->cached_device_info.name = عدم;
    }
    𐡄𐡍 (gpu->cached_device_info.description) {
        حرر(gpu->cached_device_info.description);
        gpu->cached_device_info.description = عدم;
    }
    𐡄𐡍 (gpu->cached_buffer_type.name) {
        حرر(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = عدم;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
