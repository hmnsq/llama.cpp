#ଅନ୍ତର୍ଭାଵ "ggml-remoting.h"
#ଅନ୍ତର୍ଭାଵ "ggml-virtgpu.h"

#ଅନ୍ତର୍ଭାଵ <iostream>
#ଅନ୍ତର୍ଭାଵ <mutex>

ଶୂନ୍ୟ ggml_virtgpu_cleanup(virtgpu * gpu);

ସ୍ଥିର virtgpu * apir_initialize() {
    ସ୍ଥିର virtgpu *         gpu         = ଖାଲି;
    ସ୍ଥିର std::atomic<ତାର୍କିକ> initialized = ମିଥ୍ୟା;

    ଯଦି (initialized) {
        // fast track
        ଫେରନ୍ତୁ gpu;
    }

    {
        ସ୍ଥିର std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ଯଦି (initialized) {
            // thread safe
            ଫେରନ୍ତୁ gpu;
        }

        gpu = create_virtgpu();
        ଯଦି (!gpu) {
            initialized = ସତ୍ୟ;
            ଫେରନ୍ତୁ ଖାଲି;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        ଯଦି (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_ଗଣନା = apir_device_get_ଗଣନା(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            ଅକ୍ଷର * rmt_device_name = apir_device_get_name(gpu);
            ଯଦି (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (ଅକ୍ଷର *) malloc(device_name_len);
            ଯଦି (!gpu->cached_device_info.name) {
                ମୁକ୍ତ(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            ମୁକ୍ତ(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_total);

        apir_buffer_type_host_handle_t buft_host_handle = apir_device_get_buffer_type(gpu);
        gpu->cached_buffer_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            ଅକ୍ଷର * rmt_name = apir_buffer_type_get_name(gpu, buft_host_handle);
            ଯଦି (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (ଅକ୍ଷର *) malloc(prefixed_len);
            ଯଦି (!gpu->cached_buffer_type.name) {
                ମୁକ୍ତ(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            ମୁକ୍ତ(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ gpu;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_remoting_get_device_ଗଣନା() {
    virtgpu * gpu = apir_initialize();
    ଯଦି (!gpu) {
        ଫେରନ୍ତୁ 0;
    }

    ଫେରନ୍ତୁ gpu->cached_device_info.device_ଗଣନା;
}

ସ୍ଥିର size_t ggml_backend_remoting_reg_get_device_ଗଣନା(ggml_backend_reg_t reg) {
    UNUSED(reg);

    ଫେରନ୍ତୁ ggml_backend_remoting_get_device_ଗଣନା();
}

ସ୍ଥିର std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    ଫେରନ୍ତୁ devices[device];
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    ଯଦି (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        ଫେରନ୍ତୁ;
    }

    virtgpu * gpu = apir_initialize();
    ଯଦି (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        ଫେରନ୍ତୁ;
    }

    ସ୍ଥିର std::atomic<ତାର୍କିକ> initialized = ମିଥ୍ୟା;

    ଯଦି (initialized) {
        ଫେରନ୍ତୁ;  // fast track
    }

    {
        ସ୍ଥିର std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        ଯଦି (!initialized) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_backend_remoting_get_device_ଗଣନା(); i++) {
                ggml_backend_remoting_device_context * ctx       = ନୂତନ ggml_backend_remoting_device_context;
                ଅକ୍ଷର                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = ନୂତନ ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = ସତ୍ୟ;
        }
    }
}

ସ୍ଥିର ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    ଫେରନ୍ତୁ ggml_backend_remoting_get_device(device);
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    ଫେରନ୍ତୁ GGML_VIRTGPU_NAME;
}

ସ୍ଥିର ଅଚଳ ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_ଗଣନା,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ ଖାଲି,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    ଯଦି (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    ସ୍ଥିର ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    ସ୍ଥିର ତାର୍କିକ initialized = ମିଥ୍ୟା;
    ଯଦି (initialized) {
        ଫେରନ୍ତୁ &reg;
    }
    initialized = ସତ୍ୟ;

    ggml_backend_remoting_reg_init_devices(&reg);

    ଫେରନ୍ତୁ &reg;
}

// public function, not exposed in the GGML interface at the moment
ଶୂନ୍ୟ ggml_virtgpu_cleanup(virtgpu * gpu) {
    ଯଦି (gpu->cached_device_info.name) {
        ମୁକ୍ତ(gpu->cached_device_info.name);
        gpu->cached_device_info.name = ଖାଲି;
    }
    ଯଦି (gpu->cached_device_info.description) {
        ମୁକ୍ତ(gpu->cached_device_info.description);
        gpu->cached_device_info.description = ଖାଲି;
    }
    ଯଦି (gpu->cached_buffer_type.name) {
        ମୁକ୍ତ(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = ଖାଲି;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
