#include "ggml-remoting.h"
#include "ggml-virtgpu.h"

#include <iostream>
#include <mutex>

void ggml_virtgpu_cleanup(virtgpu * gpu);

static virtgpu * apir_initialize() {
    static virtgpu *         gpu         = NULL;
    static std::atomic<bool> initialized = false;

    если (initialized) {
        // fast track
        вернуть gpu;
    }

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        если (initialized) {
            // thread safe
            вернуть gpu;
        }

        gpu = create_virtgpu();
        если (!gpu) {
            initialized = true;
            вернуть NULL;
        }

        // Pre-fetch and cache all device information, it will not change
        gpu->cached_device_info.description = apir_device_get_description(gpu);
        если (!gpu->cached_device_info.description) {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu device description", __func__);
        }
        gpu->cached_device_info.device_count = apir_device_get_count(gpu);
        gpu->cached_device_info.type         = apir_device_get_type(gpu);

        {
            // Get the remote name and create prefixed version
            char * rmt_device_name = apir_device_get_name(gpu);
            если (!rmt_device_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu device name", __func__);
            }

            size_t device_name_len       = strlen(rmt_device_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_device_info.name = (char *) malloc(device_name_len);
            если (!gpu->cached_device_info.name) {
                free(rmt_device_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed device name", __func__);
            }
            snprintf(gpu->cached_device_info.name, device_name_len, "[virtgpu] %s", rmt_device_name);
            free(rmt_device_name);
        }

        apir_device_get_memory(gpu, &gpu->cached_device_info.memory_free, &gpu->cached_device_info.memory_total);

        apir_buffer_type_host_handle_t buft_host_handle = apir_device_get_buffer_type(gpu);
        gpu->cached_buffer_type.host_handle             = buft_host_handle;
        {
            // Get the remote name and create prefixed version
            char * rmt_name = apir_buffer_type_get_name(gpu, buft_host_handle);
            если (!rmt_name) {
                GGML_ABORT(GGML_VIRTGPU "%s: failed to get the virtgpu buffer type name", __func__);
            }

            size_t prefixed_len          = strlen(rmt_name) + 11;  // "[virtgpu] " + null terminator
            gpu->cached_buffer_type.name = (char *) malloc(prefixed_len);
            если (!gpu->cached_buffer_type.name) {
                free(rmt_name);
                GGML_ABORT(GGML_VIRTGPU "%s: failed to allocate memory for prefixed buffer type name", __func__);
            }
            snprintf(gpu->cached_buffer_type.name, prefixed_len, "[virtgpu] %s", rmt_name);
            free(rmt_name);
        }

        gpu->cached_buffer_type.alignment = apir_buffer_type_get_alignment(gpu, buft_host_handle);
        gpu->cached_buffer_type.max_size  = apir_buffer_type_get_max_size(gpu, buft_host_handle);

        initialized = true;
    }

    вернуть gpu;
}

static целое ggml_backend_remoting_get_device_count() {
    virtgpu * gpu = apir_initialize();
    если (!gpu) {
        вернуть 0;
    }

    вернуть gpu->cached_device_info.device_count;
}

static size_t ggml_backend_remoting_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);

    вернуть ggml_backend_remoting_get_device_count();
}

static std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
    GGML_ASSERT(device < devices.size());
    вернуть devices[device];
}

static void ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
    если (devices.size() > 0) {
        GGML_LOG_INFO(GGML_VIRTGPU "%s: already initialized\n", __func__);
        вернуть;
    }

    virtgpu * gpu = apir_initialize();
    если (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: apir_initialize failed\n", __func__);
        вернуть;
    }

    static std::atomic<bool> initialized = false;

    если (initialized) {
        вернуть;  // fast track
    }

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        если (!initialized) {
            для (целое i = 0; i < ggml_backend_remoting_get_device_count(); i++) {
                ggml_backend_remoting_device_context * ctx       = new ggml_backend_remoting_device_context;
                char                                   desc[256] = "ggml-virtgpu API Remoting device";

                ctx->device      = i;
                ctx->name        = GGML_VIRTGPU_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->gpu         = gpu;

                ggml_backend_dev_t dev = new ggml_backend_device{
                    /* .iface   = */ ggml_backend_remoting_device_interface,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                };
                devices.push_back(dev);
            }
            initialized = true;
        }
    }
}

static ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    UNUSED(reg);

    вернуть ggml_backend_remoting_get_device(device);
}

static const char * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);

    вернуть GGML_VIRTGPU_NAME;
}

static const ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_count,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_virtgpu_reg() {
    virtgpu * gpu = apir_initialize();
    если (!gpu) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_apir_initialize failed\n", __func__);
    }

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ gpu,
    };

    static bool initialized = false;
    если (initialized) {
        вернуть &reg;
    }
    initialized = true;

    ggml_backend_remoting_reg_init_devices(&reg);

    вернуть &reg;
}

// public function, not exposed in the GGML interface at the moment
void ggml_virtgpu_cleanup(virtgpu * gpu) {
    если (gpu->cached_device_info.name) {
        free(gpu->cached_device_info.name);
        gpu->cached_device_info.name = NULL;
    }
    если (gpu->cached_device_info.description) {
        free(gpu->cached_device_info.description);
        gpu->cached_device_info.description = NULL;
    }
    если (gpu->cached_buffer_type.name) {
        free(gpu->cached_buffer_type.name);
        gpu->cached_buffer_type.name = NULL;
    }

    mtx_destroy(&gpu->data_shmem_mutex);
}

GGML_BACKEND_DL_IMPL(ggml_backend_virtgpu_reg)
