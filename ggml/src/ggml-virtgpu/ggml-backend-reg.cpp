#অন্তর্ভাৱ "ggml-remoting.h"
#অন্তর্ভাৱ "ggml-virtgpu.h"

#অন্তর্ভাৱ <iostream>
#অন্তর্ভাৱ <mutex>

শূন্য ggml_virtgpu_cleanup(virtgpu * gpu);

স্থির virtgpu * apir_initialize() {
    স্থির virtgpu *         gpu         = রিক্ত;
    স্থির std::atomic<তার্কিক> initialized = মিছা;

    যদি (initialized) {
        // fast track
        ঘূৰাই gpu;
    }

    {
        স্থির std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        যদি (initialized) {
            // thread safe
            ঘূৰাই gpu;
        }

        gpu = create_virtgpu();
        যদি (!gpu) {
            initialized = সঁচা;
            ঘূৰাই রিক্ত;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        যদি (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_গুনতি = apir_device_get_গুনতি(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            ৱর্ণ * rmt_device_name = apir_device_get_name(gpu);
            যদি (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (ৱর্ণ *) malloc(device_name_len);
            যদি (!gpu->cached_device_info.name) {
                মুক্ত(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            মুক্ত(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_total);

        apir_buffer_type_host_handle_t buft_host_handle = apir_device_get_buffer_type(gpu);
        gpu->cached_buffer_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            ৱর্ণ * rmt_name = apir_buffer_type_get_name(gpu, buft_host_handle);
            যদি (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (ৱর্ণ *) malloc(prefixed_len);
            যদি (!gpu->cached_buffer_type.name) {
                মুক্ত(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            মুক্ত(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = সঁচা;
    }

    ঘূৰাই gpu;
}

স্থির পূর্ণাঙ্ক ggml_backend_remoting_get_device_গুনতি() {
    virtgpu * gpu = apir_initialize();
    যদি (!gpu) {
        ঘূৰাই 0;
    }

    ঘূৰাই gpu->cached_device_info.device_গুনতি;
}

স্থির size_t ggml_backend_remoting_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    UNUSED(reg);

    ঘূৰাই ggml_backend_remoting_get_device_গুনতি();
}

স্থির std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    ঘূৰাই devices[device];
}

স্থির শূন্য ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    যদি (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        ঘূৰাই;
    }

    virtgpu * gpu = apir_initialize();
    যদি (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        ঘূৰাই;
    }

    স্থির std::atomic<তার্কিক> initialized = মিছা;

    যদি (initialized) {
        ঘূৰাই;  // fast track
    }

    {
        স্থির std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        যদি (!initialized) {
            বাবে (পূর্ণাঙ্ক i = 0; i < ggml_backend_remoting_get_device_গুনতি(); i++) {
                ggml_backend_remoting_device_context * ctx       = নূতন ggml_backend_remoting_device_context;
                ৱর্ণ                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = নূতন ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = সঁচা;
        }
    }
}

স্থির ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    ঘূৰাই ggml_backend_remoting_get_device(device);
}

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    ঘূৰাই GGML_VIRTGPU_NAME;
}

স্থির ধ্রুৱ ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ রিক্ত,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    যদি (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    স্থির ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    স্থির তার্কিক initialized = মিছা;
    যদি (initialized) {
        ঘূৰাই &reg;
    }
    initialized = সঁচা;

    ggml_backend_remoting_reg_init_devices(&reg);

    ঘূৰাই &reg;
}

// public function, not exposed in the GGML interface at the moment
শূন্য ggml_virtgpu_cleanup(virtgpu * gpu) {
    যদি (gpu->cached_device_info.name) {
        মুক্ত(gpu->cached_device_info.name);
        gpu->cached_device_info.name = রিক্ত;
    }
    যদি (gpu->cached_device_info.description) {
        মুক্ত(gpu->cached_device_info.description);
        gpu->cached_device_info.description = রিক্ত;
    }
    যদি (gpu->cached_buffer_type.name) {
        মুক্ত(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = রিক্ত;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
