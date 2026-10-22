#অন্তর্ভাব "ggml-remoting.h"
#অন্তর্ভাব "ggml-virtgpu.h"

#অন্তর্ভাব <iostream>
#অন্তর্ভাব <mutex>

শূন্য ggml_virtgpu_cleanup(virtgpu * gpu);

স্থির virtgpu * apir_initialize() {
    স্থির virtgpu *         gpu         = ফাঁকা;
    স্থির std::atomic<যৌক্তিক> initialized = মিথ্যা;

    যদি (initialized) {
        // fast track
        ফেরত gpu;
    }

    {
        স্থির std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        যদি (initialized) {
            // thread safe
            ফেরত gpu;
        }

        gpu = create_virtgpu();
        যদি (!gpu) {
            initialized = সত্য;
            ফেরত ফাঁকা;
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
            অক্ষর * rmt_device_name = apir_device_get_name(gpu);
            যদি (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (অক্ষর *) malloc(device_name_len);
            যদি (!gpu->cached_device_info.name) {
                মুক্ত(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            মুক্ত(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_যোগ);

        apir_বাফার_type_host_handle_t buft_host_handle = apir_device_get_বাফার_type(gpu);
        gpu->cached_বাফার_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            অক্ষর * rmt_name = apir_বাফার_type_get_name(gpu, buft_host_handle);
            যদি (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_বাফার_type.name = (অক্ষর *) malloc(prefixed_len);
            যদি (!gpu->cached_বাফার_type.name) {
                মুক্ত(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_বাফার_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            মুক্ত(rmt_name);
        }

        gpu->cached_বাফার_type.alignment = apir_বাফার_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_বাফার_type.max_size  = apir_বাফার_type_get_max_আকার(gpu, buft_host_handle);

        initialized = সত্য;
    }

    ফেরত gpu;
}

স্থির পূর্ণসংখ্যা ggml_backend_remoting_get_device_গুনতি() {
    virtgpu * gpu = apir_initialize();
    যদি (!gpu) {
        ফেরত 0;
    }

    ফেরত gpu->cached_device_info.device_গুনতি;
}

স্থির size_t ggml_backend_remoting_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    UNUSED(reg);

    ফেরত ggml_backend_remoting_get_device_গুনতি();
}

স্থির std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    ফেরত devices[device];
}

স্থির শূন্য ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    যদি (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        ফেরত;
    }

    virtgpu * gpu = apir_initialize();
    যদি (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        ফেরত;
    }

    স্থির std::atomic<যৌক্তিক> initialized = মিথ্যা;

    যদি (initialized) {
        ফেরত;  // fast track
    }

    {
        স্থির std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        যদি (!initialized) {
            জন্য (পূর্ণসংখ্যা i = 0; i < ggml_backend_remoting_get_device_গুনতি(); i++) {
                ggml_backend_remoting_device_context * ctx       = নতুন ggml_backend_remoting_device_context;
                অক্ষর                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = নতুন ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = সত্য;
        }
    }
}

স্থির ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    ফেরত ggml_backend_remoting_get_device(device);
}

স্থির ধ্রুবক অক্ষর * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    ফেরত GGML_VIRTGPU_NAME;
}

স্থির ধ্রুবক ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ ফাঁকা,
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

    স্থির যৌক্তিক initialized = মিথ্যা;
    যদি (initialized) {
        ফেরত &reg;
    }
    initialized = সত্য;

    ggml_backend_remoting_reg_init_devices(&reg);

    ফেরত &reg;
}

// public function, not exposed in the GGML interface at the moment
শূন্য ggml_virtgpu_cleanup(virtgpu * gpu) {
    যদি (gpu->cached_device_info.name) {
        মুক্ত(gpu->cached_device_info.name);
        gpu->cached_device_info.name = ফাঁকা;
    }
    যদি (gpu->cached_device_info.description) {
        মুক্ত(gpu->cached_device_info.description);
        gpu->cached_device_info.description = ফাঁকা;
    }
    যদি (gpu->cached_বাফার_type.name) {
        মুক্ত(gpu->cached_বাফার_type.name);
        gpu->cached_বাফার_type.name = ফাঁকা;
    }

    mtx_destroy(&gpu->তথ্য_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
