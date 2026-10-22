#ಅನ್ತರ್ಭಾವ "virtgpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <cstdlib>

ಸ್ಥಿರ virt_gpu_result_t virtgpu_open_device(virtgpu * gpu, ಧ್ರುವ drmDevicePtr dev);
ಸ್ಥಿರ virt_gpu_result_t virtgpu_open(virtgpu * gpu);

ಸ್ಥಿರ virt_gpu_result_t virtgpu_init_capset(virtgpu * gpu);
ಸ್ಥಿರ virt_gpu_result_t virtgpu_init_context(virtgpu * gpu);

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ      virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id);
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ      virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                       virgl_renderer_capset id,
                                       uint32_t              version,
                                       ಶೂನ್ಯ *                capset,
                                       size_t                capset_size);
ಸ್ಥಿರ uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param);
ಸ್ಥಿರ ಶೂನ್ಯ     virtgpu_init_renderer_info(virtgpu * gpu);

ಸ್ಥಿರ ಶೂನ್ಯ log_call_duration(long long call_duration_ns, ಧ್ರುವ ವರ್ಣ * name);

ಧ್ರುವ uint64_t APIR_HANDSHAKE_MAX_WAIT_MS   = 2 * 1000;   // 2s
ಧ್ರುವ uint64_t APIR_LOADLIBRARY_MAX_WAIT_MS = 60 * 1000;  // 60s

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ virtgpu_handshake(virtgpu * gpu) {
    apir_encoder * encoder;
    apir_decoder * decoder;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_HANDSHAKE, 0);
    ಯದಿ (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    /* write handshake props */

    uint32_t guest_major = APIR_PROTOCOL_MAJOR;
    uint32_t guest_minor = APIR_PROTOCOL_MINOR;
    apir_encode_uint32_t(encoder, &guest_major);
    apir_encode_uint32_t(encoder, &guest_minor);

    /* *** */

    uint32_t  ret_magic;
    long long call_duration_ns;
    ret_magic = remote_call(gpu, encoder, &decoder, APIR_HANDSHAKE_MAX_WAIT_MS, &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting handshake");

    ಯದಿ (!decoder) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: failed to initiate the communication with the virglrenderer library. "
                   "Most likely, the wrong virglrenderer library was loaded in the hypervisor.",
                   __func__);
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    /* read handshake return values */

    uint32_t host_major;
    uint32_t host_minor;

    ಯದಿ (ret_magic != APIR_HANDSHAKE_MAGIC) {
        GGML_ABORT(GGML_VIRTGPU "%s: handshake with the virglrenderer failed (code=%d | %s)", __func__, ret_magic,
                   apir_backend_initialize_error(ret_magic));
    } ಅನ್ಯಥಾ {
        apir_decode_uint32_t(decoder, &host_major);
        apir_decode_uint32_t(decoder, &host_minor);
    }

    remote_call_finish(gpu, encoder, decoder);

    ಯದಿ (ret_magic != APIR_HANDSHAKE_MAGIC) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: Guest is running with %u.%u\n", __func__, guest_major, guest_minor);
    GGML_LOG_INFO(GGML_VIRTGPU "%s: Host is running with %u.%u\n", __func__, host_major, host_minor);

    ಯದಿ (guest_major != host_major) {
        GGML_LOG_ERROR(GGML_VIRTGPU "Host major (%d) and guest major (%d) version differ\n", host_major, guest_major);
    } ಅನ್ಯಥಾ ಯದಿ (guest_minor != host_minor) {
        GGML_LOG_WARN(GGML_VIRTGPU "Host minor (%d) and guest minor (%d) version differ\n", host_minor, guest_minor);
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}

ಸ್ಥಿರ ApirLoadLibraryReturnCode virtgpu_load_library(virtgpu * gpu) {
    apir_encoder *            encoder;
    apir_decoder *            decoder;
    ApirLoadLibraryReturnCode ret;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_LOADLIBRARY, 0);
    ಯದಿ (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to prepare the API Remoting command encoder", __func__);
        ಪ್ರತ್ಯಾಗಮ APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    long long call_duration_ns;

    ret = (ApirLoadLibraryReturnCode) remote_call(gpu, encoder, &decoder, APIR_LOADLIBRARY_MAX_WAIT_MS,
                                                  &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting LoadLibrary");

    ಯದಿ (!decoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to trigger the API Remoting hypercall.\n", __func__);
        ಪ್ರತ್ಯಾಗಮ APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    remote_call_finish(gpu, encoder, decoder);

    ಯದಿ (ret == APIR_LOAD_LIBRARY_SUCCESS) {
        GGML_LOG_INFO(GGML_VIRTGPU "The API Remoting backend was successfully loaded and initialized\n");

        ಪ್ರತ್ಯಾಗಮ ret;
    }

    // something wrong happened, find out what.
    ಯದಿ (ret < APIR_LOAD_LIBRARY_INIT_BASE_INDEX) {
        ಯದಿ (ret == APIR_LOAD_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library, "
                       "some environment variables are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_load_library_error(ret));
        } ಅನ್ಯಥಾ ಯದಿ (ret == APIR_LOAD_LIBRARY_CANNOT_OPEN) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_load_library_error(ret));
        } ಅನ್ಯಥಾ ಯದಿ (ret == APIR_LOAD_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: could not load the backend library, some symbols are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s) ",
                       __func__, apir_load_library_error(ret));
        } ಅನ್ಯಥಾ {
            GGML_ABORT(GGML_VIRTGPU "%s: virglrenderer could not load the API Remoting backend library. (%s - code %d)",
                       __func__, apir_load_library_error(ret), ret);
        }
        ಪ್ರತ್ಯಾಗಮ ret;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: virglrenderer successfully loaded the API Remoting backend library.\n", __func__);

    ApirLoadLibraryReturnCode apir_ret = (ApirLoadLibraryReturnCode) (ret - APIR_LOAD_LIBRARY_INIT_BASE_INDEX);

    ಯದಿ (apir_ret == APIR_LOAD_LIBRARY_CANNOT_OPEN) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library. "
                   "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                   __func__, apir_load_library_error(apir_ret));
    } ಅನ್ಯಥಾ ಯದಿ (apir_ret == APIR_LOAD_LIBRARY_ಚಿಹ್ನೆ_MISSING) {
        GGML_ABORT(
            GGML_VIRTGPU
            "%s: the API Remoting backend library couldn't load the GGML backend library, some symbols are missing. "
            "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
            __func__, apir_load_library_error(apir_ret));
    } ಅನ್ಯಥಾ ಯದಿ (apir_ret < APIR_LOAD_LIBRARY_INIT_BASE_INDEX) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library: apir code=%d | %s)",
                   __func__, apir_ret, apir_load_library_error(apir_ret));
    } ಅನ್ಯಥಾ {
        uint32_t lib_ret = apir_ret - APIR_LOAD_LIBRARY_INIT_BASE_INDEX;
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library failed to initialize its backend library: apir code=%d)",
                   __func__, lib_ret);
    }
    ಪ್ರತ್ಯಾಗಮ ret;
}

virtgpu * create_virtgpu() {
    virtgpu * gpu = ನೂತನ virtgpu();

    gpu->use_apir_capset = getenv("GGML_REMOTING_USE_APIR_CAPSET") != nullptr;
    util_sparse_array_init(&gpu->shmem_array, ಪರಿಮಾಣ(virtgpu_shmem), 1024);

    // Initialize mutex to protect shared data_shmem buffer
    ಯದಿ (mtx_init(&gpu->data_shmem_mutex, mtx_plain) != thrd_success) {
        delete gpu;
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize data_shmem mutex", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (virtgpu_open(gpu) != APIR_SUCCESS) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to open the virtgpu device\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (virtgpu_init_capset(gpu) != APIR_SUCCESS) {
        ಯದಿ (gpu->use_apir_capset) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: failed to initialize the virtgpu APIR capset. Make sure that the virglrenderer library "
                       "supports it.",
                       __func__);
        } ಅನ್ಯಥಾ {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu Venus capset", __func__);
        }
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (virtgpu_init_context(gpu) != APIR_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the GPU context", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (virtgpu_shmem_create(gpu, SHMEM_REPLY_SIZE, &gpu->reply_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared reply memory pages", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (virtgpu_shmem_create(gpu, SHMEM_DATA_SIZE, &gpu->data_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared data memory pages", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (virtgpu_handshake(gpu)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to handshake with the virglrenderer library", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (virtgpu_load_library(gpu) != APIR_LOAD_LIBRARY_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to load the backend library", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತ್ಯಾಗಮ gpu;
}

ಸ್ಥಿರ virt_gpu_result_t virtgpu_open(virtgpu * gpu) {
    drmDevicePtr devs[8];
    ಪೂರ್ಣಾಙ್ಕ          count = drmGetDevices2(0, devs, ARRAY_SIZE(devs));
    ಯದಿ (count < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to enumerate DRM devices\n", __func__);
        ಪ್ರತ್ಯಾಗಮ APIR_ERROR_INITIALIZATION_FAILED;
    }

    virt_gpu_result_t result = APIR_ERROR_INITIALIZATION_FAILED;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < count; i++) {
        result = virtgpu_open_device(gpu, devs[i]);
        ಯದಿ (result == APIR_SUCCESS) {
            ಭಙ್ಗ;
        }
    }

    drmFreeDevices(devs, count);

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸ್ಥಿರ virt_gpu_result_t virtgpu_open_device(virtgpu * gpu, ಧ್ರುವ drmDevicePtr dev) {
    ಧ್ರುವ ವರ್ಣ * ಗಂಟು_path = dev->nodes[DRM_ಗಂಟು_RENDER];

    ಪೂರ್ಣಾಙ್ಕ fd = open(ಗಂಟು_path, O_RDWR | O_CLOEXEC);
    ಯದಿ (fd < 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to open %s", __func__, ಗಂಟು_path);
        ಪ್ರತ್ಯಾಗಮ APIR_ERROR_INITIALIZATION_FAILED;
    }

    drmVersionPtr version = drmGetVersion(fd);
    ಯದಿ (!version || strcmp(version->name, "virtio_gpu") || version->version_major != 0) {
        ಯದಿ (version) {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: unknown DRM driver %s version %d\n", __func__, version->name,
                           version->version_major);
        } ಅನ್ಯಥಾ {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get DRM driver version\n", __func__);
        }

        ಯದಿ (version) {
            drmFreeVersion(version);
        }
        close(fd);
        ಪ್ರತ್ಯಾಗಮ APIR_ERROR_INITIALIZATION_FAILED;
    }

    gpu->fd = fd;

    drmFreeVersion(version);

    GGML_LOG_INFO(GGML_VIRTGPU "using DRM device %s\n", ಗಂಟು_path);

    ಪ್ರತ್ಯಾಗಮ APIR_SUCCESS;
}

ಸ್ಥಿರ virt_gpu_result_t virtgpu_init_context(virtgpu * gpu) {
    ಪ್ರತಿಜ್ಞಾ(!gpu->capset.version);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ret = virtgpu_ioctl_context_init(gpu, gpu->capset.id);
    ಯದಿ (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to initialize context: %s\n", __func__, strerror(errno));
        ಪ್ರತ್ಯಾಗಮ APIR_ERROR_INITIALIZATION_FAILED;
    }

    ಪ್ರತ್ಯಾಗಮ APIR_SUCCESS;
}

ಸ್ಥಿರ virt_gpu_result_t virtgpu_init_capset(virtgpu * gpu) {
    ಯದಿ (gpu->use_apir_capset) {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the APIR capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_APIR;
    } ಅನ್ಯಥಾ {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the Venus capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_VENUS;
    }
    gpu->capset.version = 0;

    ಪೂರ್ಣಾಙ್ಕ ret =
        virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.version, &gpu->capset.data, ಪರಿಮಾಣ(gpu->capset.data));

    ಯದಿ (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get APIR v%d capset: %s\n", __func__, gpu->capset.version,
                       strerror(errno));
        ಪ್ರತ್ಯಾಗಮ APIR_ERROR_INITIALIZATION_FAILED;
    }

    ಪ್ರತಿಜ್ಞಾ(gpu->capset.data.supports_blob_resources);

    ಪ್ರತ್ಯಾಗಮ APIR_SUCCESS;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id) {
    drm_virtgpu_context_set_param ctx_set_params[3] = {
        {
         .param = VIRTGPU_CONTEXT_PARAM_CAPSET_ID,
         .value = capset_id,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_ಅಂಕೆ_RINGS,
         .value = 1,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_POLL_RINGS_MASK,
         .value = 0, /* don't generate drm_events on fence signaling */
        },
    };

    drm_virtgpu_context_init args = {
        .ಅಂಕೆ_params     = ARRAY_SIZE(ctx_set_params),
        .pad            = 0,
        .ctx_set_params = (uintptr_t) &ctx_set_params,
    };

    ಪ್ರತ್ಯಾಗಮ virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &args);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                  virgl_renderer_capset id,
                                  uint32_t              version,
                                  ಶೂನ್ಯ *                capset,
                                  size_t                capset_size) {
    drm_virtgpu_get_caps args = {
        .cap_set_id  = id,
        .cap_set_ver = version,
        .addr        = (uintptr_t) capset,
        .size        = (__u32) capset_size,
        .pad         = 0,
    };

    ಪ್ರತ್ಯಾಗಮ virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GET_CAPS, &args);
}

ಸ್ಥಿರ uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param) {
    /* val must be zeroed because kernel only writes the lower 32 bits */
    uint64_t             val  = 0;
    drm_virtgpu_getparam args = {
        .param = param,
        .value = (uintptr_t) &val,
    };

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ret = virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GETPARAM, &args);
    ಪ್ರತ್ಯಾಗಮ ret ? 0 : val;
}

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags) {
    /*
     * Prepare the command encoder and its buffer
     */

    thread_local ವರ್ಣ encoder_buffer[4096];

    thread_local apir_encoder enc;
    enc = {
        .cur   = encoder_buffer,
        .start = encoder_buffer,
        .end   = encoder_buffer + ಪರಿಮಾಣ(encoder_buffer),
        .fatal = ಅಸತ್ಯ,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    ಯದಿ (!gpu->use_apir_capset) {
        cmd_type += VENUS_COMMAND_TYPE_LENGTH;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t reply_res_id = gpu->reply_shmem.res_id;
    apir_encode_uint32_t(&enc, &reply_res_id);

    ಪ್ರತ್ಯಾಗಮ &enc;
}

ಶೂನ್ಯ remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec) {
    UNUSED(gpu);

    ಯದಿ (!enc) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) encoder", __func__);
    }

    ಯದಿ (!dec) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) decoder", __func__);
    }

    ಯದಿ (apir_encoder_get_fatal(enc)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to encode the output parameters.", __func__);
    }

    ಯದಿ (apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to decode the input parameters.", __func__);
    }
}

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  encoder,
                     apir_decoder ** decoder,
                     ಭಿನ್ನ           max_wait_ms,
                     long long *     call_duration_ns) {
    /*
     * Prepare the reply notification pointer
     */

    ಚಞ್ಚಲ std::atomic_uint * atomic_reply_notif = (ಚಞ್ಚಲ std::atomic_uint *) gpu->reply_shmem.mmap_ptr;
    *atomic_reply_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    drm_virtgpu_execbuffer args = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .ಅಂಕೆ_bo_handles = 0,

        .fence_fd         = 0,
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .ಅಂಕೆ_in_syncobjs  = 0,
        .ಅಂಕೆ_out_syncobjs = 0,
        .in_syncobjs      = 0,
        .out_syncobjs     = 0,
    };

    *decoder = ರಿಕ್ತ;

    ಪೂರ್ಣಾಙ್ಕ ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &args);

    ಯದಿ (ret != 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: the virtgpu EXECBUFFER ioctl failed (%d)", __func__, ret);
    }

    /*
     * Wait for the response notification
     */
    timer_data wait_host_reply_timer = { 0, 0, 0 };

    start_timer(&wait_host_reply_timer);

    timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    long long start_time = (long long) ts_start.tv_sec * 1000000000LL + ts_start.tv_nsec;

    ತಾರ್ಕಿಕ     timedout    = ಅಸತ್ಯ;
    uint32_t notif_ಮೌಲ್ಯ = 0;
    ಯಾವತ್ (ಸತ್ಯ) {
        notif_ಮೌಲ್ಯ = std::atomic_load_explicit(atomic_reply_notif, std::memory_order_acquire);

        ಯದಿ (notif_ಮೌಲ್ಯ != 0) {
            ಭಙ್ಗ;
        }

        int64_t base_sleep_us = 15;

        os_time_sleep(base_sleep_us);

        ಯದಿ (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            long long end_time    = (long long) ts_end.tv_sec * 1000000000LL + ts_end.tv_nsec;
            ಭಿನ್ನ     duration_ms = (end_time - start_time) / 1000000;

            ಯದಿ (duration_ms > max_wait_ms) {
                timedout = ಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
    }

    ಯದಿ (call_duration_ns) {
        *call_duration_ns = stop_timer(&wait_host_reply_timer);
    }

    ಯದಿ (max_wait_ms && timedout) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: timed out waiting for the host answer...\n", __func__);
        ಪ್ರತ್ಯಾಗಮ APIR_FORWARD_TIMEOUT;
    }

    /*
     * Prepare the decoder
     */
    ಸ್ಥಿರ apir_decoder response_dec;
    response_dec.cur = (ವರ್ಣ *) gpu->reply_shmem.mmap_ptr + ಪರಿಮಾಣ(*atomic_reply_notif);
    response_dec.end = (ವರ್ಣ *) gpu->reply_shmem.mmap_ptr + gpu->reply_shmem.mmap_size;
    *decoder         = &response_dec;

    // extract the actual return value from the notif flag
    uint32_t returned_ಮೌಲ್ಯ = notif_ಮೌಲ್ಯ - 1;
    ಪ್ರತ್ಯಾಗಮ returned_ಮೌಲ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ log_call_duration(long long call_duration_ns, ಧ್ರುವ ವರ್ಣ * name) {
    ದ್ವಿಗುಣ call_duration_ms = (ದ್ವಿಗುಣ) call_duration_ns / 1e6;  // 1 millisecond = 1e6 nanoseconds
    ದ್ವಿಗುಣ call_duration_s  = (ದ್ವಿಗುಣ) call_duration_ns / 1e9;  // 1 second = 1e9 nanoseconds

    ಯದಿ (call_duration_s > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fs for the %s host reply...\n", call_duration_s, name);
    } ಅನ್ಯಥಾ ಯದಿ (call_duration_ms > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fms for the %s host reply...\n", call_duration_ms, name);
    } ಅನ್ಯಥಾ {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %lldns for the %s host reply...\n", call_duration_ns, name);
    }
}
