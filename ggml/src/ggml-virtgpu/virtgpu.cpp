#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-remoting.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cerrno>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>

ਸਥਿਰ virt_gpu_ਨਤੀਜਾ_t virtgpu_open_device(virtgpu * gpu, ਅਚਲ drmDevicePtr dev);
ਸਥਿਰ virt_gpu_ਨਤੀਜਾ_t virtgpu_open(virtgpu * gpu);

ਸਥਿਰ virt_gpu_ਨਤੀਜਾ_t virtgpu_init_capset(virtgpu * gpu);
ਸਥਿਰ virt_gpu_ਨਤੀਜਾ_t virtgpu_init_context(virtgpu * gpu);

ਸਥਿਰ ਪੂਰਨ      virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id);
ਸਥਿਰ ਪੂਰਨ      virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                       virgl_renderer_capset id,
                                       uint32_t              ਸੰਸਕਰਣ_2,
                                       ਖਾਲੀ *                capset,
                                       size_t                capset_ਮਾਪ);
ਸਥਿਰ uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param);
ਸਥਿਰ ਖਾਲੀ     virtgpu_init_renderer_info(virtgpu * gpu);

ਸਥਿਰ ਖਾਲੀ log_call_duration(ਲੰਮਾ ਲੰਮਾ call_duration_ns, ਅਚਲ ਅੱਖਰ * name);

ਅਚਲ uint64_t APIR_HANDSHAKE_MAX_WAIT_MS   = 2 * 1000;   // 2s
ਅਚਲ uint64_t APIR_LOADLIBRARY_MAX_WAIT_MS = 60 * 1000;  // 60s

ਸਥਿਰ ਪੂਰਨ virtgpu_handshake(virtgpu * gpu) {
    apir_encoder * encoder;
    apir_decoder * decoder;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_HANDSHAKE, 0);
    ਜੇ (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);
        ਵਾਪਸ 1;
    }

    /* write handshake props */

    uint32_t guest_major = APIR_PROTOCOL_MAJOR;
    uint32_t guest_minor = APIR_PROTOCOL_MINOR;
    apir_encode_uint32_t(encoder, &guest_major);
    apir_encode_uint32_t(encoder, &guest_minor);

    /* *** */

    uint32_t  ret_magic;
    ਲੰਮਾ ਲੰਮਾ call_duration_ns;
    ret_magic = remote_call(gpu, encoder, &decoder, APIR_HANDSHAKE_MAX_WAIT_MS, &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting handshake");

    ਜੇ (!decoder) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: failed to initiate the communication with the virglrenderer library. "
                   "Most likely, the wrong virglrenderer library was loaded in the hypervisor.",
                   __func__);
        ਵਾਪਸ 1;
    }

    /* read handshake return values */

    uint32_t host_major;
    uint32_t host_minor;

    ਜੇ (ret_magic != APIR_HANDSHAKE_MAGIC) {
        GGML_ABORT(GGML_VIRTGPU "%s: handshake with the virglrenderer failed (code=%d | %s)", __func__, ret_magic,
                   apir_backend_initialize_error(ret_magic));
    } ਨਹੀਂਤਾਂ {
        apir_decode_uint32_t(decoder, &host_major);
        apir_decode_uint32_t(decoder, &host_minor);
    }

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਜੇ (ret_magic != APIR_HANDSHAKE_MAGIC) {
        ਵਾਪਸ 1;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: Guest is running with %u.%u\n", __func__, guest_major, guest_minor);
    GGML_LOG_INFO(GGML_VIRTGPU "%s: Host is running with %u.%u\n", __func__, host_major, host_minor);

    ਜੇ (guest_major != host_major) {
        GGML_LOG_ERROR(GGML_VIRTGPU "Host major (%d) and guest major (%d) version differ\n", host_major, guest_major);
    } ਨਹੀਂਤਾਂ ਜੇ (guest_minor != host_minor) {
        GGML_LOG_WARN(GGML_VIRTGPU "Host minor (%d) and guest minor (%d) version differ\n", host_minor, guest_minor);
    }

    ਵਾਪਸ 0;
}

ਸਥਿਰ Apir_ਲੱਦੋ_Library_Return_ਕੋਡ virtgpu_ਲੱਦੋ_library(virtgpu * gpu) {
    apir_encoder *            encoder;
    apir_decoder *            decoder;
    Apir_ਲੱਦੋ_Library_Return_ਕੋਡ ret;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_LOADLIBRARY, 0);
    ਜੇ (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to prepare the API Remoting command encoder", __func__);
        ਵਾਪਸ APIR_ਲੱਦੋ_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    ਲੰਮਾ ਲੰਮਾ call_duration_ns;

    ret = (Apir_ਲੱਦੋ_Library_Return_ਕੋਡ) remote_call(gpu, encoder, &decoder, APIR_LOADLIBRARY_MAX_WAIT_MS,
                                                  &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting LoadLibrary");

    ਜੇ (!decoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to trigger the API Remoting hypercall.\n", __func__);
        ਵਾਪਸ APIR_ਲੱਦੋ_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਜੇ (ret == APIR_ਲੱਦੋ_LIBRARY_SUCCESS) {
        GGML_LOG_INFO(GGML_VIRTGPU "The API Remoting backend was successfully loaded and initialized\n");

        ਵਾਪਸ ret;
    }

    // something wrong happened, find out what.
    ਜੇ (ret < APIR_ਲੱਦੋ_LIBRARY_INIT_ਅਧਾਰ_ਸੂਚਕਾਂਕ) {
        ਜੇ (ret == APIR_ਲੱਦੋ_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library, "
                       "some environment variables are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_ਲੱਦੋ_library_error(ret));
        } ਨਹੀਂਤਾਂ ਜੇ (ret == APIR_ਲੱਦੋ_LIBRARY_CANNOT_OPEN) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_ਲੱਦੋ_library_error(ret));
        } ਨਹੀਂਤਾਂ ਜੇ (ret == APIR_ਲੱਦੋ_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: could not load the backend library, some symbols are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s) ",
                       __func__, apir_ਲੱਦੋ_library_error(ret));
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT(GGML_VIRTGPU "%s: virglrenderer could not load the API Remoting backend library. (%s - code %d)",
                       __func__, apir_ਲੱਦੋ_library_error(ret), ret);
        }
        ਵਾਪਸ ret;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: virglrenderer successfully loaded the API Remoting backend library.\n", __func__);

    Apir_ਲੱਦੋ_Library_Return_ਕੋਡ apir_ret = (Apir_ਲੱਦੋ_Library_Return_ਕੋਡ) (ret - APIR_ਲੱਦੋ_LIBRARY_INIT_ਅਧਾਰ_ਸੂਚਕਾਂਕ);

    ਜੇ (apir_ret == APIR_ਲੱਦੋ_LIBRARY_CANNOT_OPEN) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library. "
                   "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                   __func__, apir_ਲੱਦੋ_library_error(apir_ret));
    } ਨਹੀਂਤਾਂ ਜੇ (apir_ret == APIR_ਲੱਦੋ_LIBRARY_ਚਿੰਨ੍ਹ_MISSING) {
        GGML_ABORT(
            GGML_VIRTGPU
            "%s: the API Remoting backend library couldn't load the GGML backend library, some symbols are missing. "
            "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
            __func__, apir_ਲੱਦੋ_library_error(apir_ret));
    } ਨਹੀਂਤਾਂ ਜੇ (apir_ret < APIR_ਲੱਦੋ_LIBRARY_INIT_ਅਧਾਰ_ਸੂਚਕਾਂਕ) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library: apir code=%d | %s)",
                   __func__, apir_ret, apir_ਲੱਦੋ_library_error(apir_ret));
    } ਨਹੀਂਤਾਂ {
        uint32_t lib_ret = apir_ret - APIR_ਲੱਦੋ_LIBRARY_INIT_ਅਧਾਰ_ਸੂਚਕਾਂਕ;
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library failed to initialize its backend library: apir code=%d)",
                   __func__, lib_ret);
    }
    ਵਾਪਸ ret;
}

virtgpu * create_virtgpu() {
    virtgpu * gpu = ਨਵਾਂ virtgpu();

    gpu->use_apir_capset = getenv("GGML_REMOTING_USE_APIR_CAPSET") != ਖਾਲੀ_ਸੂਚਕ;
    util_sparse_array_init(&gpu->shmem_array, ਆਕਾਰ(virtgpu_shmem), 1024);

    // Initialize mutex to protect shared data_shmem buffer
    ਜੇ (mtx_init(&gpu->ਅੰਕੜੇ_shmem_mutex, mtx_plain) != thrd_success) {
        ਮਿਟਾਓ gpu;
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize data_shmem mutex", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (virtgpu_open(gpu) != APIR_SUCCESS) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to open the virtgpu device\n", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (virtgpu_init_capset(gpu) != APIR_SUCCESS) {
        ਜੇ (gpu->use_apir_capset) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: failed to initialize the virtgpu APIR capset. Make sure that the virglrenderer library "
                       "supports it.",
                       __func__);
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu Venus capset", __func__);
        }
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (virtgpu_init_context(gpu) != APIR_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the GPU context", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (virtgpu_shmem_create(gpu, SHMEM_ਜਵਾਬ_ਮਾਪ, &gpu->ਜਵਾਬ_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared reply memory pages", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (virtgpu_shmem_create(gpu, SHMEM_ਅੰਕੜੇ_ਮਾਪ, &gpu->ਅੰਕੜੇ_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared data memory pages", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (virtgpu_handshake(gpu)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to handshake with the virglrenderer library", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (virtgpu_ਲੱਦੋ_library(gpu) != APIR_ਲੱਦੋ_LIBRARY_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to load the backend library", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ gpu;
}

ਸਥਿਰ virt_gpu_ਨਤੀਜਾ_t virtgpu_open(virtgpu * gpu) {
    drmDevicePtr devs[8];
    ਪੂਰਨ          count = drmGetDevices2(0, devs, ARRAY_ਮਾਪ(devs));
    ਜੇ (count < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to enumerate DRM devices\n", __func__);
        ਵਾਪਸ APIR_ERROR_INITIALIZATION_FAILED;
    }

    virt_gpu_ਨਤੀਜਾ_t ਨਤੀਜਾ = APIR_ERROR_INITIALIZATION_FAILED;
    ਲਈ (ਪੂਰਨ i = 0; i < count; i++) {
        ਨਤੀਜਾ = virtgpu_open_device(gpu, devs[i]);
        ਜੇ (ਨਤੀਜਾ == APIR_SUCCESS) {
            ਰੋਕੋ;
        }
    }

    drmFreeDevices(devs, count);

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ virt_gpu_ਨਤੀਜਾ_t virtgpu_open_device(virtgpu * gpu, ਅਚਲ drmDevicePtr dev) {
    ਅਚਲ ਅੱਖਰ * ਗੰਢ_ਰਸਤਾ = dev->nodes[DRM_ਗੰਢ_RENDER];

    ਪੂਰਨ fd = open(ਗੰਢ_ਰਸਤਾ, O_RDWR | O_CLOEXEC);
    ਜੇ (fd < 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to open %s", __func__, ਗੰਢ_ਰਸਤਾ);
        ਵਾਪਸ APIR_ERROR_INITIALIZATION_FAILED;
    }

    drm_ਸੰਸਕਰਣ_Ptr ਸੰਸਕਰਣ_2 = drm_Get_ਸੰਸਕਰਣ(fd);
    ਜੇ (!ਸੰਸਕਰਣ_2 || strcmp(ਸੰਸਕਰਣ_2->name, "virtio_gpu") || ਸੰਸਕਰਣ_2->ਸੰਸਕਰਣ_major != 0) {
        ਜੇ (ਸੰਸਕਰਣ_2) {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: unknown DRM driver %s version %d\n", __func__, ਸੰਸਕਰਣ_2->name,
                           ਸੰਸਕਰਣ_2->ਸੰਸਕਰਣ_major);
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get DRM driver version\n", __func__);
        }

        ਜੇ (ਸੰਸਕਰਣ_2) {
            drm_Free_ਸੰਸਕਰਣ(ਸੰਸਕਰਣ_2);
        }
        close(fd);
        ਵਾਪਸ APIR_ERROR_INITIALIZATION_FAILED;
    }

    gpu->fd = fd;

    drm_Free_ਸੰਸਕਰਣ(ਸੰਸਕਰਣ_2);

    GGML_LOG_INFO(GGML_VIRTGPU "using DRM device %s\n", ਗੰਢ_ਰਸਤਾ);

    ਵਾਪਸ APIR_SUCCESS;
}

ਸਥਿਰ virt_gpu_ਨਤੀਜਾ_t virtgpu_init_context(virtgpu * gpu) {
    ਪ੍ਰਤਿਜ੍ਞਾ(!gpu->capset.ਸੰਸਕਰਣ_2);
    ਅਚਲ ਪੂਰਨ ret = virtgpu_ioctl_context_init(gpu, gpu->capset.id);
    ਜੇ (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to initialize context: %s\n", __func__, strerror(errno));
        ਵਾਪਸ APIR_ERROR_INITIALIZATION_FAILED;
    }

    ਵਾਪਸ APIR_SUCCESS;
}

ਸਥਿਰ virt_gpu_ਨਤੀਜਾ_t virtgpu_init_capset(virtgpu * gpu) {
    ਜੇ (gpu->use_apir_capset) {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the APIR capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_APIR;
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the Venus capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_VENUS;
    }
    gpu->capset.ਸੰਸਕਰਣ_2 = 0;

    ਪੂਰਨ ret =
        virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.ਸੰਸਕਰਣ_2, &gpu->capset.ਅੰਕੜੇ, ਆਕਾਰ(gpu->capset.ਅੰਕੜੇ));

    ਜੇ (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get APIR v%d capset: %s\n", __func__, gpu->capset.ਸੰਸਕਰਣ_2,
                       strerror(errno));
        ਵਾਪਸ APIR_ERROR_INITIALIZATION_FAILED;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(gpu->capset.ਅੰਕੜੇ.supports_blob_resources);

    ਵਾਪਸ APIR_SUCCESS;
}

ਸਥਿਰ ਪੂਰਨ virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id) {
    drm_virtgpu_context_ਸੈੱਟ_param ctx_ਸੈੱਟ_params[3] = {
        {
         .param = VIRTGPU_CONTEXT_PARAM_CAPSET_ID,
         .value = capset_id,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_ਅੰਕ_RINGS,
         .value = 1,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_POLL_RINGS_ਨਕਾਬ,
         .value = 0, /* don't generate drm_events on fence signaling */
        },
    };

    drm_virtgpu_context_init ਦਲੀਲਾਂ_2 = {
        .ਅੰਕ_params     = ARRAY_ਮਾਪ(ctx_ਸੈੱਟ_params),
        .pad            = 0,
        .ctx_ਸੈੱਟ_params = (uintptr_t) &ctx_ਸੈੱਟ_params,
    };

    ਵਾਪਸ virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &ਦਲੀਲਾਂ_2);
}

ਸਥਿਰ ਪੂਰਨ virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                  virgl_renderer_capset id,
                                  uint32_t              ਸੰਸਕਰਣ_2,
                                  ਖਾਲੀ *                capset,
                                  size_t                capset_ਮਾਪ) {
    drm_virtgpu_get_caps ਦਲੀਲਾਂ_2 = {
        .cap_ਸੈੱਟ_id  = id,
        .cap_ਸੈੱਟ_ver = ਸੰਸਕਰਣ_2,
        .addr        = (uintptr_t) capset,
        .size        = (__u32) capset_ਮਾਪ,
        .pad         = 0,
    };

    ਵਾਪਸ virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GET_CAPS, &ਦਲੀਲਾਂ_2);
}

ਸਥਿਰ uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param) {
    /* val must be zeroed because kernel only writes the lower 32 bits */
    uint64_t             val  = 0;
    drm_virtgpu_getparam ਦਲੀਲਾਂ_2 = {
        .param = param,
        .value = (uintptr_t) &val,
    };

    ਅਚਲ ਪੂਰਨ ret = virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GETPARAM, &ਦਲੀਲਾਂ_2);
    ਵਾਪਸ ret ? 0 : val;
}

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags) {
    /*
     * Prepare the command encoder and its buffer
     */

    thread_local ਅੱਖਰ encoder_ਬਫਰ[4096];

    thread_local apir_encoder enc;
    enc = {
        .cur   = encoder_ਬਫਰ,
        .start = encoder_ਬਫਰ,
        .end   = encoder_ਬਫਰ + ਆਕਾਰ(encoder_ਬਫਰ),
        .fatal = ਝੂਠ,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    ਜੇ (!gpu->use_apir_capset) {
        cmd_type += VENUS_COMMAND_TYPE_ਲੰਮਾਈ;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t ਜਵਾਬ_res_id = gpu->ਜਵਾਬ_shmem.res_id;
    apir_encode_uint32_t(&enc, &ਜਵਾਬ_res_id);

    ਵਾਪਸ &enc;
}

ਖਾਲੀ remote_call_ਅੰਤ(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec) {
    ਅਣਵਰਤਿਆ(gpu);

    ਜੇ (!enc) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) encoder", __func__);
    }

    ਜੇ (!dec) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) decoder", __func__);
    }

    ਜੇ (apir_encoder_get_fatal(enc)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to encode the output parameters.", __func__);
    }

    ਜੇ (apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to decode the input parameters.", __func__);
    }
}

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  encoder,
                     apir_decoder ** decoder,
                     ਦਸ਼ਮਲਵ           max_wait_ms,
                     ਲੰਮਾ ਲੰਮਾ *     call_duration_ns) {
    /*
     * Prepare the reply notification pointer
     */

    ਅਸਥਿਰ std::atomic_uint * atomic_ਜਵਾਬ_notif = (ਅਸਥਿਰ std::atomic_uint *) gpu->ਜਵਾਬ_shmem.mmap_ptr;
    *atomic_ਜਵਾਬ_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    drm_virtgpu_execbuffer ਦਲੀਲਾਂ_2 = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .ਅੰਕ_bo_handles = 0,

        .fence_fd         = 0,
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .ਅੰਕ_ਵਿੱਚ_syncobjs  = 0,
        .ਅੰਕ_ਬਾਹਰਲਾ_syncobjs = 0,
        .ਵਿੱਚ_syncobjs      = 0,
        .ਬਾਹਰਲਾ_syncobjs     = 0,
    };

    *decoder = ਨਿਰਾ;

    ਪੂਰਨ ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &ਦਲੀਲਾਂ_2);

    ਜੇ (ret != 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: the virtgpu EXECBUFFER ioctl failed (%d)", __func__, ret);
    }

    /*
     * Wait for the response notification
     */
    timer_ਅੰਕੜੇ wait_host_ਜਵਾਬ_timer = { 0, 0, 0 };

    ਸ਼ੁਰੂ_timer(&wait_host_ਜਵਾਬ_timer);

    timespec ts_ਸ਼ੁਰੂ, ts_ਸਿਰਾ;
    clock_gettime(CLOCK_MONOTONIC, &ts_ਸ਼ੁਰੂ);
    ਲੰਮਾ ਲੰਮਾ ਸ਼ੁਰੂ_ਸਮਾਂ = (ਲੰਮਾ ਲੰਮਾ) ts_ਸ਼ੁਰੂ.tv_sec * 1000000000LL + ts_ਸ਼ੁਰੂ.tv_nsec;

    ਤਾਰਕਿਕ     timedout    = ਝੂਠ;
    uint32_t notif_ਮੁੱਲ = 0;
    ਜਦੋਂ (ਸੱਚ) {
        notif_ਮੁੱਲ = std::atomic_load_explicit(atomic_ਜਵਾਬ_notif, std::memory_order_acquire);

        ਜੇ (notif_ਮੁੱਲ != 0) {
            ਰੋਕੋ;
        }

        int64_t ਅਧਾਰ_sleep_us = 15;

        os_ਸਮਾਂ_sleep(ਅਧਾਰ_sleep_us);

        ਜੇ (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_ਸਿਰਾ);
            ਲੰਮਾ ਲੰਮਾ ਸਿਰਾ_ਸਮਾਂ    = (ਲੰਮਾ ਲੰਮਾ) ts_ਸਿਰਾ.tv_sec * 1000000000LL + ts_ਸਿਰਾ.tv_nsec;
            ਦਸ਼ਮਲਵ     duration_ms = (ਸਿਰਾ_ਸਮਾਂ - ਸ਼ੁਰੂ_ਸਮਾਂ) / 1000000;

            ਜੇ (duration_ms > max_wait_ms) {
                timedout = ਸੱਚ;
                ਰੋਕੋ;
            }
        }
    }

    ਜੇ (call_duration_ns) {
        *call_duration_ns = stop_timer(&wait_host_ਜਵਾਬ_timer);
    }

    ਜੇ (max_wait_ms && timedout) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: timed out waiting for the host answer...\n", __func__);
        ਵਾਪਸ APIR_FORWARD_TIMEOUT;
    }

    /*
     * Prepare the decoder
     */
    ਸਥਿਰ apir_decoder response_dec;
    response_dec.cur = (ਅੱਖਰ *) gpu->ਜਵਾਬ_shmem.mmap_ptr + ਆਕਾਰ(*atomic_ਜਵਾਬ_notif);
    response_dec.end = (ਅੱਖਰ *) gpu->ਜਵਾਬ_shmem.mmap_ptr + gpu->ਜਵਾਬ_shmem.mmap_ਮਾਪ;
    *decoder         = &response_dec;

    // extract the actual return value from the notif flag
    uint32_t returned_ਮੁੱਲ = notif_ਮੁੱਲ - 1;
    ਵਾਪਸ returned_ਮੁੱਲ;
}

ਸਥਿਰ ਖਾਲੀ log_call_duration(ਲੰਮਾ ਲੰਮਾ call_duration_ns, ਅਚਲ ਅੱਖਰ * name) {
    ਦੁੱਗਣਾ call_duration_ms = (ਦੁੱਗਣਾ) call_duration_ns / 1e6;  // 1 millisecond = 1e6 nanoseconds
    ਦੁੱਗਣਾ call_duration_s  = (ਦੁੱਗਣਾ) call_duration_ns / 1e9;  // 1 second = 1e9 nanoseconds

    ਜੇ (call_duration_s > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fs for the %s host reply...\n", call_duration_s, name);
    } ਨਹੀਂਤਾਂ ਜੇ (call_duration_ms > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fms for the %s host reply...\n", call_duration_ms, name);
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %lldns for the %s host reply...\n", call_duration_ns, name);
    }
}
