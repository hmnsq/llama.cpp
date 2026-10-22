#समावेश "virtgpu.h"
#समावेश "ggml-remoting.h"

#समावेश <stdio.h>
#समावेश <unistd.h>

#समावेश <cassert>
#समावेश <cerrno>
#समावेश <cstdlib>

अटल virt_gpu_परिणाम_t virtgpu_open_device(virtgpu * gpu, स्थिर drmDevicePtr dev);
अटल virt_gpu_परिणाम_t virtgpu_open(virtgpu * gpu);

अटल virt_gpu_परिणाम_t virtgpu_init_capset(virtgpu * gpu);
अटल virt_gpu_परिणाम_t virtgpu_init_context(virtgpu * gpu);

अटल पूर्णांक      virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id);
अटल पूर्णांक      virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                       virgl_renderer_capset id,
                                       uint32_t              version,
                                       व्योम *                capset,
                                       माप_प्रकार                capset_आकार);
अटल uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param);
अटल व्योम     virtgpu_init_renderer_info(virtgpu * gpu);

अटल व्योम log_call_duration(दीर्घ दीर्घ call_duration_ns, स्थिर अक्षर * name);

स्थिर uint64_t APIR_HANDSHAKE_MAX_WAIT_MS   = 2 * 1000;   // 2s
स्थिर uint64_t APIR_LOADLIBRARY_MAX_WAIT_MS = 60 * 1000;  // 60s

अटल पूर्णांक virtgpu_handshake(virtgpu * gpu) {
    apir_encoder * encoder;
    apir_decoder * decoder;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_HANDSHAKE, 0);
    अगर (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);
        वापस 1;
    }

    /* write handshake props */

    uint32_t guest_major = APIR_PROTOCOL_MAJOR;
    uint32_t guest_minor = APIR_PROTOCOL_MINOR;
    apir_encode_uint32_t(encoder, &guest_major);
    apir_encode_uint32_t(encoder, &guest_minor);

    /* *** */

    uint32_t  ret_magic;
    दीर्घ दीर्घ call_duration_ns;
    ret_magic = remote_call(gpu, encoder, &decoder, APIR_HANDSHAKE_MAX_WAIT_MS, &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting handshake");

    अगर (!decoder) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: failed to initiate the communication with the virglrenderer library. "
                   "Most likely, the wrong virglrenderer library was loaded in the hypervisor.",
                   __func__);
        वापस 1;
    }

    /* read handshake return values */

    uint32_t host_major;
    uint32_t host_minor;

    अगर (ret_magic != APIR_HANDSHAKE_MAGIC) {
        GGML_ABORT(GGML_VIRTGPU "%s: handshake with the virglrenderer failed (code=%d | %s)", __func__, ret_magic,
                   apir_backend_initialize_error(ret_magic));
    } अन्यथा {
        apir_decode_uint32_t(decoder, &host_major);
        apir_decode_uint32_t(decoder, &host_minor);
    }

    remote_call_finish(gpu, encoder, decoder);

    अगर (ret_magic != APIR_HANDSHAKE_MAGIC) {
        वापस 1;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: Guest is running with %u.%u\n", __func__, guest_major, guest_minor);
    GGML_LOG_INFO(GGML_VIRTGPU "%s: Host is running with %u.%u\n", __func__, host_major, host_minor);

    अगर (guest_major != host_major) {
        GGML_LOG_ERROR(GGML_VIRTGPU "Host major (%d) and guest major (%d) version differ\n", host_major, guest_major);
    } अन्यथा अगर (guest_minor != host_minor) {
        GGML_LOG_WARN(GGML_VIRTGPU "Host minor (%d) and guest minor (%d) version differ\n", host_minor, guest_minor);
    }

    वापस 0;
}

अटल ApirLoadLibraryReturnCode virtgpu_load_library(virtgpu * gpu) {
    apir_encoder *            encoder;
    apir_decoder *            decoder;
    ApirLoadLibraryReturnCode ret;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_LOADLIBRARY, 0);
    अगर (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to prepare the API Remoting command encoder", __func__);
        वापस APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    दीर्घ दीर्घ call_duration_ns;

    ret = (ApirLoadLibraryReturnCode) remote_call(gpu, encoder, &decoder, APIR_LOADLIBRARY_MAX_WAIT_MS,
                                                  &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting LoadLibrary");

    अगर (!decoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to trigger the API Remoting hypercall.\n", __func__);
        वापस APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    remote_call_finish(gpu, encoder, decoder);

    अगर (ret == APIR_LOAD_LIBRARY_SUCCESS) {
        GGML_LOG_INFO(GGML_VIRTGPU "The API Remoting backend was successfully loaded and initialized\n");

        वापस ret;
    }

    // something wrong happened, find out what.
    अगर (ret < APIR_LOAD_LIBRARY_INIT_BASE_सूचकांक) {
        अगर (ret == APIR_LOAD_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library, "
                       "some environment variables are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_load_library_error(ret));
        } अन्यथा अगर (ret == APIR_LOAD_LIBRARY_CANNOT_OPEN) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_load_library_error(ret));
        } अन्यथा अगर (ret == APIR_LOAD_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: could not load the backend library, some symbols are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s) ",
                       __func__, apir_load_library_error(ret));
        } अन्यथा {
            GGML_ABORT(GGML_VIRTGPU "%s: virglrenderer could not load the API Remoting backend library. (%s - code %d)",
                       __func__, apir_load_library_error(ret), ret);
        }
        वापस ret;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: virglrenderer successfully loaded the API Remoting backend library.\n", __func__);

    ApirLoadLibraryReturnCode apir_ret = (ApirLoadLibraryReturnCode) (ret - APIR_LOAD_LIBRARY_INIT_BASE_सूचकांक);

    अगर (apir_ret == APIR_LOAD_LIBRARY_CANNOT_OPEN) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library. "
                   "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                   __func__, apir_load_library_error(apir_ret));
    } अन्यथा अगर (apir_ret == APIR_LOAD_LIBRARY_प्रतीक_MISSING) {
        GGML_ABORT(
            GGML_VIRTGPU
            "%s: the API Remoting backend library couldn't load the GGML backend library, some symbols are missing. "
            "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
            __func__, apir_load_library_error(apir_ret));
    } अन्यथा अगर (apir_ret < APIR_LOAD_LIBRARY_INIT_BASE_सूचकांक) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library: apir code=%d | %s)",
                   __func__, apir_ret, apir_load_library_error(apir_ret));
    } अन्यथा {
        uint32_t lib_ret = apir_ret - APIR_LOAD_LIBRARY_INIT_BASE_सूचकांक;
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library failed to initialize its backend library: apir code=%d)",
                   __func__, lib_ret);
    }
    वापस ret;
}

virtgpu * create_virtgpu() {
    virtgpu * gpu = नया virtgpu();

    gpu->use_apir_capset = दो_पर्या("GGML_REMOTING_USE_APIR_CAPSET") != nullptr;
    util_sparse_array_init(&gpu->shmem_array, माप(virtgpu_shmem), 1024);

    // Initialize mutex to protect shared data_shmem buffer
    अगर (mtx_init(&gpu->तथ्य_shmem_mutex, mtx_plain) != thrd_success) {
        मिटाओ gpu;
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize data_shmem mutex", __func__);
        वापस शून्य;
    }

    अगर (virtgpu_open(gpu) != APIR_SUCCESS) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to open the virtgpu device\n", __func__);
        वापस शून्य;
    }

    अगर (virtgpu_init_capset(gpu) != APIR_SUCCESS) {
        अगर (gpu->use_apir_capset) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: failed to initialize the virtgpu APIR capset. Make sure that the virglrenderer library "
                       "supports it.",
                       __func__);
        } अन्यथा {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu Venus capset", __func__);
        }
        वापस शून्य;
    }

    अगर (virtgpu_init_context(gpu) != APIR_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the GPU context", __func__);
        वापस शून्य;
    }

    अगर (virtgpu_shmem_create(gpu, SHMEM_REPLY_आकार, &gpu->reply_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared reply memory pages", __func__);
        वापस शून्य;
    }

    अगर (virtgpu_shmem_create(gpu, SHMEM_तथ्य_आकार, &gpu->तथ्य_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared data memory pages", __func__);
        वापस शून्य;
    }

    अगर (virtgpu_handshake(gpu)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to handshake with the virglrenderer library", __func__);
        वापस शून्य;
    }

    अगर (virtgpu_load_library(gpu) != APIR_LOAD_LIBRARY_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to load the backend library", __func__);
        वापस शून्य;
    }

    वापस gpu;
}

अटल virt_gpu_परिणाम_t virtgpu_open(virtgpu * gpu) {
    drmDevicePtr devs[8];
    पूर्णांक          count = drmGetDevices2(0, devs, ARRAY_आकार(devs));
    अगर (count < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to enumerate DRM devices\n", __func__);
        वापस APIR_ERROR_INITIALIZATION_FAILED;
    }

    virt_gpu_परिणाम_t परिणाम = APIR_ERROR_INITIALIZATION_FAILED;
    क्रम (पूर्णांक i = 0; i < count; i++) {
        परिणाम = virtgpu_open_device(gpu, devs[i]);
        अगर (परिणाम == APIR_SUCCESS) {
            अवरोध;
        }
    }

    drmFreeDevices(devs, count);

    वापस परिणाम;
}

अटल virt_gpu_परिणाम_t virtgpu_open_device(virtgpu * gpu, स्थिर drmDevicePtr dev) {
    स्थिर अक्षर * गाँठ_path = dev->nodes[DRM_गाँठ_RENDER];

    पूर्णांक fd = खोलो(गाँठ_path, O_RDWR | O_CLOEXEC);
    अगर (fd < 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to open %s", __func__, गाँठ_path);
        वापस APIR_ERROR_INITIALIZATION_FAILED;
    }

    drmVersionPtr version = drmGetVersion(fd);
    अगर (!version || म_भेद(version->name, "virtio_gpu") || version->version_major != 0) {
        अगर (version) {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: unknown DRM driver %s version %d\n", __func__, version->name,
                           version->version_major);
        } अन्यथा {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get DRM driver version\n", __func__);
        }

        अगर (version) {
            drmFreeVersion(version);
        }
        बंद(fd);
        वापस APIR_ERROR_INITIALIZATION_FAILED;
    }

    gpu->fd = fd;

    drmFreeVersion(version);

    GGML_LOG_INFO(GGML_VIRTGPU "using DRM device %s\n", गाँठ_path);

    वापस APIR_SUCCESS;
}

अटल virt_gpu_परिणाम_t virtgpu_init_context(virtgpu * gpu) {
    निश्चित(!gpu->capset.version);
    स्थिर पूर्णांक ret = virtgpu_ioctl_context_init(gpu, gpu->capset.id);
    अगर (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to initialize context: %s\n", __func__, म_त्रुटि(त्रुटि_सं));
        वापस APIR_ERROR_INITIALIZATION_FAILED;
    }

    वापस APIR_SUCCESS;
}

अटल virt_gpu_परिणाम_t virtgpu_init_capset(virtgpu * gpu) {
    अगर (gpu->use_apir_capset) {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the APIR capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_APIR;
    } अन्यथा {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the Venus capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_VENUS;
    }
    gpu->capset.version = 0;

    पूर्णांक ret =
        virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.version, &gpu->capset.तथ्य, माप(gpu->capset.तथ्य));

    अगर (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get APIR v%d capset: %s\n", __func__, gpu->capset.version,
                       म_त्रुटि(त्रुटि_सं));
        वापस APIR_ERROR_INITIALIZATION_FAILED;
    }

    निश्चित(gpu->capset.तथ्य.supports_blob_resources);

    वापस APIR_SUCCESS;
}

अटल पूर्णांक virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id) {
    drm_virtgpu_context_set_param ctx_set_params[3] = {
        {
         .param = VIRTGPU_CONTEXT_PARAM_CAPSET_ID,
         .value = capset_id,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_अंक_RINGS,
         .value = 1,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_POLL_RINGS_MASK,
         .value = 0, /* don't generate drm_events on fence signaling */
        },
    };

    drm_virtgpu_context_init args = {
        .अंक_params     = ARRAY_आकार(ctx_set_params),
        .pad            = 0,
        .ctx_set_params = (uintptr_t) &ctx_set_params,
    };

    वापस virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &args);
}

अटल पूर्णांक virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                  virgl_renderer_capset id,
                                  uint32_t              version,
                                  व्योम *                capset,
                                  माप_प्रकार                capset_आकार) {
    drm_virtgpu_get_caps args = {
        .cap_set_id  = id,
        .cap_set_ver = version,
        .addr        = (uintptr_t) capset,
        .size        = (__u32) capset_आकार,
        .pad         = 0,
    };

    वापस virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GET_CAPS, &args);
}

अटल uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param) {
    /* val must be zeroed because kernel only writes the lower 32 bits */
    uint64_t             val  = 0;
    drm_virtgpu_getparam args = {
        .param = param,
        .value = (uintptr_t) &val,
    };

    स्थिर पूर्णांक ret = virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GETPARAM, &args);
    वापस ret ? 0 : val;
}

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags) {
    /*
     * Prepare the command encoder and its buffer
     */

    thread_local अक्षर encoder_बफर[4096];

    thread_local apir_encoder enc;
    enc = {
        .cur   = encoder_बफर,
        .start = encoder_बफर,
        .end   = encoder_बफर + माप(encoder_बफर),
        .fatal = false,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    अगर (!gpu->use_apir_capset) {
        cmd_type += VENUS_COMMAND_TYPE_लंबाई;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t reply_res_id = gpu->reply_shmem.res_id;
    apir_encode_uint32_t(&enc, &reply_res_id);

    वापस &enc;
}

व्योम remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * डेसी) {
    UNUSED(gpu);

    अगर (!enc) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) encoder", __func__);
    }

    अगर (!डेसी) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) decoder", __func__);
    }

    अगर (apir_encoder_get_fatal(enc)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to encode the output parameters.", __func__);
    }

    अगर (apir_decoder_get_fatal(डेसी)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to decode the input parameters.", __func__);
    }
}

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  encoder,
                     apir_decoder ** decoder,
                     भग्न           max_wait_ms,
                     दीर्घ दीर्घ *     call_duration_ns) {
    /*
     * Prepare the reply notification pointer
     */

    अस्थिर std::atomic_uint * atomic_reply_notif = (अस्थिर std::atomic_uint *) gpu->reply_shmem.mmap_ptr;
    *atomic_reply_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    drm_virtgpu_execbuffer args = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .अंक_bo_handles = 0,

        .fence_fd         = 0,
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .अंक_in_syncobjs  = 0,
        .अंक_out_syncobjs = 0,
        .in_syncobjs      = 0,
        .out_syncobjs     = 0,
    };

    *decoder = शून्य;

    पूर्णांक ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &args);

    अगर (ret != 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: the virtgpu EXECBUFFER ioctl failed (%d)", __func__, ret);
    }

    /*
     * Wait for the response notification
     */
    timer_तथ्य wait_host_reply_timer = { 0, 0, 0 };

    start_timer(&wait_host_reply_timer);

    timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    दीर्घ दीर्घ start_time = (दीर्घ दीर्घ) ts_start.tv_sec * 1000000000LL + ts_start.tv_nsec;

    bool     timedout    = false;
    uint32_t notif_मूल्य = 0;
    जबतक (true) {
        notif_मूल्य = std::atomic_load_explicit(atomic_reply_notif, std::memory_order_acquire);

        अगर (notif_मूल्य != 0) {
            अवरोध;
        }

        int64_t base_sleep_us = 15;

        os_time_sleep(base_sleep_us);

        अगर (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            दीर्घ दीर्घ end_time    = (दीर्घ दीर्घ) ts_end.tv_sec * 1000000000LL + ts_end.tv_nsec;
            भग्न     duration_ms = (end_time - start_time) / 1000000;

            अगर (duration_ms > max_wait_ms) {
                timedout = true;
                अवरोध;
            }
        }
    }

    अगर (call_duration_ns) {
        *call_duration_ns = stop_timer(&wait_host_reply_timer);
    }

    अगर (max_wait_ms && timedout) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: timed out waiting for the host answer...\n", __func__);
        वापस APIR_FORWARD_TIMEOUT;
    }

    /*
     * Prepare the decoder
     */
    अटल apir_decoder response_dec;
    response_dec.cur = (अक्षर *) gpu->reply_shmem.mmap_ptr + माप(*atomic_reply_notif);
    response_dec.end = (अक्षर *) gpu->reply_shmem.mmap_ptr + gpu->reply_shmem.mmap_आकार;
    *decoder         = &response_dec;

    // extract the actual return value from the notif flag
    uint32_t returned_मूल्य = notif_मूल्य - 1;
    वापस returned_मूल्य;
}

अटल व्योम log_call_duration(दीर्घ दीर्घ call_duration_ns, स्थिर अक्षर * name) {
    द्विगुन call_duration_ms = (द्विगुन) call_duration_ns / 1e6;  // 1 millisecond = 1e6 nanoseconds
    द्विगुन call_duration_s  = (द्विगुन) call_duration_ns / 1e9;  // 1 second = 1e9 nanoseconds

    अगर (call_duration_s > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fs for the %s host reply...\n", call_duration_s, name);
    } अन्यथा अगर (call_duration_ms > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fms for the %s host reply...\n", call_duration_ms, name);
    } अन्यथा {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %lldns for the %s host reply...\n", call_duration_ns, name);
    }
}
