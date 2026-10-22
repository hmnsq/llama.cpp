#అన్తర్భావ "virtgpu.h"
#అన్తర్భావ "ggml-remoting.h"

#అన్తర్భావ <stdio.h>
#అన్తర్భావ <unistd.h>

#అన్తర్భావ <cassert>
#అన్తర్భావ <cerrno>
#అన్తర్భావ <cstdlib>

స్థిర virt_gpu_ఫలితం_t virtgpu_open_device(virtgpu * gpu, స్థిరరాశి drmDevicePtr dev);
స్థిర virt_gpu_ఫలితం_t virtgpu_open(virtgpu * gpu);

స్థిర virt_gpu_ఫలితం_t virtgpu_init_capset(virtgpu * gpu);
స్థిర virt_gpu_ఫలితం_t virtgpu_init_context(virtgpu * gpu);

స్థిర పూర్ణాంకం      virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id);
స్థిర పూర్ణాంకం      virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                       virgl_renderer_capset id,
                                       uint32_t              సంస్కరణ_2,
                                       శూన్య *                capset,
                                       size_t                capset_కొలత);
స్థిర uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param);
స్థిర శూన్య     virtgpu_init_renderer_info(virtgpu * gpu);

స్థిర శూన్య log_call_duration(దీర్ఘ దీర్ఘ call_duration_ns, స్థిరరాశి అక్షరం * name);

స్థిరరాశి uint64_t APIR_HANDSHAKE_MAX_WAIT_MS   = 2 * 1000;   // 2s
స్థిరరాశి uint64_t APIR_LOADLIBRARY_MAX_WAIT_MS = 60 * 1000;  // 60s

స్థిర పూర్ణాంకం virtgpu_handshake(virtgpu * gpu) {
    apir_encoder * encoder;
    apir_decoder * decoder;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_HANDSHAKE, 0);
    ఒకవేళ (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);
        తిరిగి 1;
    }

    /* write handshake props */

    uint32_t guest_major = APIR_PROTOCOL_MAJOR;
    uint32_t guest_minor = APIR_PROTOCOL_MINOR;
    apir_encode_uint32_t(encoder, &guest_major);
    apir_encode_uint32_t(encoder, &guest_minor);

    /* *** */

    uint32_t  ret_magic;
    దీర్ఘ దీర్ఘ call_duration_ns;
    ret_magic = remote_call(gpu, encoder, &decoder, APIR_HANDSHAKE_MAX_WAIT_MS, &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting handshake");

    ఒకవేళ (!decoder) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: failed to initiate the communication with the virglrenderer library. "
                   "Most likely, the wrong virglrenderer library was loaded in the hypervisor.",
                   __func__);
        తిరిగి 1;
    }

    /* read handshake return values */

    uint32_t host_major;
    uint32_t host_minor;

    ఒకవేళ (ret_magic != APIR_HANDSHAKE_MAGIC) {
        GGML_ABORT(GGML_VIRTGPU "%s: handshake with the virglrenderer failed (code=%d | %s)", __func__, ret_magic,
                   apir_backend_initialize_error(ret_magic));
    } లేకపోతే {
        apir_decode_uint32_t(decoder, &host_major);
        apir_decode_uint32_t(decoder, &host_minor);
    }

    remote_call_ముగింపు(gpu, encoder, decoder);

    ఒకవేళ (ret_magic != APIR_HANDSHAKE_MAGIC) {
        తిరిగి 1;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: Guest is running with %u.%u\n", __func__, guest_major, guest_minor);
    GGML_LOG_INFO(GGML_VIRTGPU "%s: Host is running with %u.%u\n", __func__, host_major, host_minor);

    ఒకవేళ (guest_major != host_major) {
        GGML_LOG_ERROR(GGML_VIRTGPU "Host major (%d) and guest major (%d) version differ\n", host_major, guest_major);
    } లేకపోతే ఒకవేళ (guest_minor != host_minor) {
        GGML_LOG_WARN(GGML_VIRTGPU "Host minor (%d) and guest minor (%d) version differ\n", host_minor, guest_minor);
    }

    తిరిగి 0;
}

స్థిర Apir_ఎక్కించు_Library_Return_సంకేతం virtgpu_ఎక్కించు_library(virtgpu * gpu) {
    apir_encoder *            encoder;
    apir_decoder *            decoder;
    Apir_ఎక్కించు_Library_Return_సంకేతం ret;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_LOADLIBRARY, 0);
    ఒకవేళ (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to prepare the API Remoting command encoder", __func__);
        తిరిగి APIR_ఎక్కించు_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    దీర్ఘ దీర్ఘ call_duration_ns;

    ret = (Apir_ఎక్కించు_Library_Return_సంకేతం) remote_call(gpu, encoder, &decoder, APIR_LOADLIBRARY_MAX_WAIT_MS,
                                                  &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting LoadLibrary");

    ఒకవేళ (!decoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to trigger the API Remoting hypercall.\n", __func__);
        తిరిగి APIR_ఎక్కించు_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    remote_call_ముగింపు(gpu, encoder, decoder);

    ఒకవేళ (ret == APIR_ఎక్కించు_LIBRARY_SUCCESS) {
        GGML_LOG_INFO(GGML_VIRTGPU "The API Remoting backend was successfully loaded and initialized\n");

        తిరిగి ret;
    }

    // something wrong happened, find out what.
    ఒకవేళ (ret < APIR_ఎక్కించు_LIBRARY_INIT_పునాది_సూచీ) {
        ఒకవేళ (ret == APIR_ఎక్కించు_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library, "
                       "some environment variables are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_ఎక్కించు_library_error(ret));
        } లేకపోతే ఒకవేళ (ret == APIR_ఎక్కించు_LIBRARY_CANNOT_OPEN) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_ఎక్కించు_library_error(ret));
        } లేకపోతే ఒకవేళ (ret == APIR_ఎక్కించు_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: could not load the backend library, some symbols are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s) ",
                       __func__, apir_ఎక్కించు_library_error(ret));
        } లేకపోతే {
            GGML_ABORT(GGML_VIRTGPU "%s: virglrenderer could not load the API Remoting backend library. (%s - code %d)",
                       __func__, apir_ఎక్కించు_library_error(ret), ret);
        }
        తిరిగి ret;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: virglrenderer successfully loaded the API Remoting backend library.\n", __func__);

    Apir_ఎక్కించు_Library_Return_సంకేతం apir_ret = (Apir_ఎక్కించు_Library_Return_సంకేతం) (ret - APIR_ఎక్కించు_LIBRARY_INIT_పునాది_సూచీ);

    ఒకవేళ (apir_ret == APIR_ఎక్కించు_LIBRARY_CANNOT_OPEN) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library. "
                   "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                   __func__, apir_ఎక్కించు_library_error(apir_ret));
    } లేకపోతే ఒకవేళ (apir_ret == APIR_ఎక్కించు_LIBRARY_చిహ్నం_MISSING) {
        GGML_ABORT(
            GGML_VIRTGPU
            "%s: the API Remoting backend library couldn't load the GGML backend library, some symbols are missing. "
            "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
            __func__, apir_ఎక్కించు_library_error(apir_ret));
    } లేకపోతే ఒకవేళ (apir_ret < APIR_ఎక్కించు_LIBRARY_INIT_పునాది_సూచీ) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library: apir code=%d | %s)",
                   __func__, apir_ret, apir_ఎక్కించు_library_error(apir_ret));
    } లేకపోతే {
        uint32_t lib_ret = apir_ret - APIR_ఎక్కించు_LIBRARY_INIT_పునాది_సూచీ;
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library failed to initialize its backend library: apir code=%d)",
                   __func__, lib_ret);
    }
    తిరిగి ret;
}

virtgpu * create_virtgpu() {
    virtgpu * gpu = కొత్త virtgpu();

    gpu->use_apir_capset = getenv("GGML_REMOTING_USE_APIR_CAPSET") != శూన్యసూచిక;
    util_sparse_array_init(&gpu->shmem_array, పరిమాణం(virtgpu_shmem), 1024);

    // Initialize mutex to protect shared data_shmem buffer
    ఒకవేళ (mtx_init(&gpu->దత్తాంశం_shmem_mutex, mtx_plain) != thrd_success) {
        తొలగించు gpu;
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize data_shmem mutex", __func__);
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (virtgpu_open(gpu) != APIR_SUCCESS) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to open the virtgpu device\n", __func__);
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (virtgpu_init_capset(gpu) != APIR_SUCCESS) {
        ఒకవేళ (gpu->use_apir_capset) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: failed to initialize the virtgpu APIR capset. Make sure that the virglrenderer library "
                       "supports it.",
                       __func__);
        } లేకపోతే {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu Venus capset", __func__);
        }
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (virtgpu_init_context(gpu) != APIR_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the GPU context", __func__);
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (virtgpu_shmem_create(gpu, SHMEM_జవాబు_కొలత, &gpu->జవాబు_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared reply memory pages", __func__);
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (virtgpu_shmem_create(gpu, SHMEM_దత్తాంశం_కొలత, &gpu->దత్తాంశం_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared data memory pages", __func__);
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (virtgpu_handshake(gpu)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to handshake with the virglrenderer library", __func__);
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (virtgpu_ఎక్కించు_library(gpu) != APIR_ఎక్కించు_LIBRARY_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to load the backend library", __func__);
        తిరిగి ఖాళీ;
    }

    తిరిగి gpu;
}

స్థిర virt_gpu_ఫలితం_t virtgpu_open(virtgpu * gpu) {
    drmDevicePtr devs[8];
    పూర్ణాంకం          count = drmGetDevices2(0, devs, ARRAY_కొలత(devs));
    ఒకవేళ (count < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to enumerate DRM devices\n", __func__);
        తిరిగి APIR_ERROR_INITIALIZATION_FAILED;
    }

    virt_gpu_ఫలితం_t ఫలితం = APIR_ERROR_INITIALIZATION_FAILED;
    కొరకు (పూర్ణాంకం i = 0; i < count; i++) {
        ఫలితం = virtgpu_open_device(gpu, devs[i]);
        ఒకవేళ (ఫలితం == APIR_SUCCESS) {
            ఆపు;
        }
    }

    drmFreeDevices(devs, count);

    తిరిగి ఫలితం;
}

స్థిర virt_gpu_ఫలితం_t virtgpu_open_device(virtgpu * gpu, స్థిరరాశి drmDevicePtr dev) {
    స్థిరరాశి అక్షరం * కణుపు_మార్గం = dev->nodes[DRM_కణుపు_RENDER];

    పూర్ణాంకం fd = open(కణుపు_మార్గం, O_RDWR | O_CLOEXEC);
    ఒకవేళ (fd < 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to open %s", __func__, కణుపు_మార్గం);
        తిరిగి APIR_ERROR_INITIALIZATION_FAILED;
    }

    drm_సంస్కరణ_Ptr సంస్కరణ_2 = drm_Get_సంస్కరణ(fd);
    ఒకవేళ (!సంస్కరణ_2 || strcmp(సంస్కరణ_2->name, "virtio_gpu") || సంస్కరణ_2->సంస్కరణ_major != 0) {
        ఒకవేళ (సంస్కరణ_2) {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: unknown DRM driver %s version %d\n", __func__, సంస్కరణ_2->name,
                           సంస్కరణ_2->సంస్కరణ_major);
        } లేకపోతే {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get DRM driver version\n", __func__);
        }

        ఒకవేళ (సంస్కరణ_2) {
            drm_Free_సంస్కరణ(సంస్కరణ_2);
        }
        close(fd);
        తిరిగి APIR_ERROR_INITIALIZATION_FAILED;
    }

    gpu->fd = fd;

    drm_Free_సంస్కరణ(సంస్కరణ_2);

    GGML_LOG_INFO(GGML_VIRTGPU "using DRM device %s\n", కణుపు_మార్గం);

    తిరిగి APIR_SUCCESS;
}

స్థిర virt_gpu_ఫలితం_t virtgpu_init_context(virtgpu * gpu) {
    ప్రతిజ్ఞా(!gpu->capset.సంస్కరణ_2);
    స్థిరరాశి పూర్ణాంకం ret = virtgpu_ioctl_context_init(gpu, gpu->capset.id);
    ఒకవేళ (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to initialize context: %s\n", __func__, strerror(errno));
        తిరిగి APIR_ERROR_INITIALIZATION_FAILED;
    }

    తిరిగి APIR_SUCCESS;
}

స్థిర virt_gpu_ఫలితం_t virtgpu_init_capset(virtgpu * gpu) {
    ఒకవేళ (gpu->use_apir_capset) {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the APIR capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_APIR;
    } లేకపోతే {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the Venus capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_VENUS;
    }
    gpu->capset.సంస్కరణ_2 = 0;

    పూర్ణాంకం ret =
        virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.సంస్కరణ_2, &gpu->capset.దత్తాంశం, పరిమాణం(gpu->capset.దత్తాంశం));

    ఒకవేళ (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get APIR v%d capset: %s\n", __func__, gpu->capset.సంస్కరణ_2,
                       strerror(errno));
        తిరిగి APIR_ERROR_INITIALIZATION_FAILED;
    }

    ప్రతిజ్ఞా(gpu->capset.దత్తాంశం.supports_blob_resources);

    తిరిగి APIR_SUCCESS;
}

స్థిర పూర్ణాంకం virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id) {
    drm_virtgpu_context_సమితి_param ctx_సమితి_params[3] = {
        {
         .param = VIRTGPU_CONTEXT_PARAM_CAPSET_ID,
         .value = capset_id,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_అంకె_RINGS,
         .value = 1,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_POLL_RINGS_ముసుగు,
         .value = 0, /* don't generate drm_events on fence signaling */
        },
    };

    drm_virtgpu_context_init వాదనలు_2 = {
        .అంకె_params     = ARRAY_కొలత(ctx_సమితి_params),
        .pad            = 0,
        .ctx_సమితి_params = (uintptr_t) &ctx_సమితి_params,
    };

    తిరిగి virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &వాదనలు_2);
}

స్థిర పూర్ణాంకం virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                  virgl_renderer_capset id,
                                  uint32_t              సంస్కరణ_2,
                                  శూన్య *                capset,
                                  size_t                capset_కొలత) {
    drm_virtgpu_get_caps వాదనలు_2 = {
        .cap_సమితి_id  = id,
        .cap_సమితి_ver = సంస్కరణ_2,
        .addr        = (uintptr_t) capset,
        .size        = (__u32) capset_కొలత,
        .pad         = 0,
    };

    తిరిగి virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GET_CAPS, &వాదనలు_2);
}

స్థిర uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param) {
    /* val must be zeroed because kernel only writes the lower 32 bits */
    uint64_t             val  = 0;
    drm_virtgpu_getparam వాదనలు_2 = {
        .param = param,
        .value = (uintptr_t) &val,
    };

    స్థిరరాశి పూర్ణాంకం ret = virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GETPARAM, &వాదనలు_2);
    తిరిగి ret ? 0 : val;
}

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags) {
    /*
     * Prepare the command encoder and its buffer
     */

    thread_local అక్షరం encoder_బఫరు[4096];

    thread_local apir_encoder enc;
    enc = {
        .cur   = encoder_బఫరు,
        .start = encoder_బఫరు,
        .end   = encoder_బఫరు + పరిమాణం(encoder_బఫరు),
        .fatal = అబద్ధం,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    ఒకవేళ (!gpu->use_apir_capset) {
        cmd_type += VENUS_COMMAND_TYPE_నిడివి;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t జవాబు_res_id = gpu->జవాబు_shmem.res_id;
    apir_encode_uint32_t(&enc, &జవాబు_res_id);

    తిరిగి &enc;
}

శూన్య remote_call_ముగింపు(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec) {
    వాడని(gpu);

    ఒకవేళ (!enc) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) encoder", __func__);
    }

    ఒకవేళ (!dec) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) decoder", __func__);
    }

    ఒకవేళ (apir_encoder_get_fatal(enc)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to encode the output parameters.", __func__);
    }

    ఒకవేళ (apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to decode the input parameters.", __func__);
    }
}

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  encoder,
                     apir_decoder ** decoder,
                     దశాంశం           max_wait_ms,
                     దీర్ఘ దీర్ఘ *     call_duration_ns) {
    /*
     * Prepare the reply notification pointer
     */

    అస్థిర std::atomic_uint * atomic_జవాబు_notif = (అస్థిర std::atomic_uint *) gpu->జవాబు_shmem.mmap_ptr;
    *atomic_జవాబు_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    drm_virtgpu_execbuffer వాదనలు_2 = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .అంకె_bo_handles = 0,

        .fence_fd         = 0,
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .అంకె_లోపల_syncobjs  = 0,
        .అంకె_బహిర్గతం_syncobjs = 0,
        .లోపల_syncobjs      = 0,
        .బహిర్గతం_syncobjs     = 0,
    };

    *decoder = ఖాళీ;

    పూర్ణాంకం ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &వాదనలు_2);

    ఒకవేళ (ret != 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: the virtgpu EXECBUFFER ioctl failed (%d)", __func__, ret);
    }

    /*
     * Wait for the response notification
     */
    timer_దత్తాంశం wait_host_జవాబు_timer = { 0, 0, 0 };

    ప్రారంభం_timer(&wait_host_జవాబు_timer);

    timespec ts_ప్రారంభం, ts_చివర;
    clock_gettime(CLOCK_MONOTONIC, &ts_ప్రారంభం);
    దీర్ఘ దీర్ఘ ప్రారంభం_సమయం = (దీర్ఘ దీర్ఘ) ts_ప్రారంభం.tv_sec * 1000000000LL + ts_ప్రారంభం.tv_nsec;

    తార్కిక     timedout    = అబద్ధం;
    uint32_t notif_విలువ = 0;
    అయితే (నిజం) {
        notif_విలువ = std::atomic_load_explicit(atomic_జవాబు_notif, std::memory_order_acquire);

        ఒకవేళ (notif_విలువ != 0) {
            ఆపు;
        }

        int64_t పునాది_sleep_us = 15;

        os_సమయం_sleep(పునాది_sleep_us);

        ఒకవేళ (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_చివర);
            దీర్ఘ దీర్ఘ చివర_సమయం    = (దీర్ఘ దీర్ఘ) ts_చివర.tv_sec * 1000000000LL + ts_చివర.tv_nsec;
            దశాంశం     duration_ms = (చివర_సమయం - ప్రారంభం_సమయం) / 1000000;

            ఒకవేళ (duration_ms > max_wait_ms) {
                timedout = నిజం;
                ఆపు;
            }
        }
    }

    ఒకవేళ (call_duration_ns) {
        *call_duration_ns = stop_timer(&wait_host_జవాబు_timer);
    }

    ఒకవేళ (max_wait_ms && timedout) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: timed out waiting for the host answer...\n", __func__);
        తిరిగి APIR_FORWARD_TIMEOUT;
    }

    /*
     * Prepare the decoder
     */
    స్థిర apir_decoder response_dec;
    response_dec.cur = (అక్షరం *) gpu->జవాబు_shmem.mmap_ptr + పరిమాణం(*atomic_జవాబు_notif);
    response_dec.end = (అక్షరం *) gpu->జవాబు_shmem.mmap_ptr + gpu->జవాబు_shmem.mmap_కొలత;
    *decoder         = &response_dec;

    // extract the actual return value from the notif flag
    uint32_t returned_విలువ = notif_విలువ - 1;
    తిరిగి returned_విలువ;
}

స్థిర శూన్య log_call_duration(దీర్ఘ దీర్ఘ call_duration_ns, స్థిరరాశి అక్షరం * name) {
    ద్విగుణ call_duration_ms = (ద్విగుణ) call_duration_ns / 1e6;  // 1 millisecond = 1e6 nanoseconds
    ద్విగుణ call_duration_s  = (ద్విగుణ) call_duration_ns / 1e9;  // 1 second = 1e9 nanoseconds

    ఒకవేళ (call_duration_s > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fs for the %s host reply...\n", call_duration_s, name);
    } లేకపోతే ఒకవేళ (call_duration_ms > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fms for the %s host reply...\n", call_duration_ms, name);
    } లేకపోతే {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %lldns for the %s host reply...\n", call_duration_ns, name);
    }
}
