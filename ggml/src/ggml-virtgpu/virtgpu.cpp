#अन्तर्भाव "virtgpu.h"
#अन्तर्भाव "ggml-remoting.h"

#अन्तर्भाव <stdio.h>
#अन्तर्भाव <unistd.h>

#अन्तर्भाव <cassert>
#अन्तर्भाव <cerrno>
#अन्तर्भाव <cstdlib>

स्थिर virt_gpu_परिणाम_t virtgpu_open_device(virtgpu * gpu, ध्रुव drmDevicePtr dev);
स्थिर virt_gpu_परिणाम_t virtgpu_open(virtgpu * gpu);

स्थिर virt_gpu_परिणाम_t virtgpu_init_capset(virtgpu * gpu);
स्थिर virt_gpu_परिणाम_t virtgpu_init_context(virtgpu * gpu);

स्थिर पूर्णाङ्क      virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id);
स्थिर पूर्णाङ्क      virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                       virgl_renderer_capset id,
                                       uint32_t              संस्करणम्_2,
                                       शून्य *                capset,
                                       size_t                capset_आकार);
स्थिर uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param);
स्थिर शून्य     virtgpu_init_renderer_info(virtgpu * gpu);

स्थिर शून्य log_call_duration(दीर्घ दीर्घ call_duration_ns, ध्रुव वर्ण * name);

ध्रुव uint64_t APIR_HANDSHAKE_MAX_WAIT_MS   = 2 * 1000;   // 2s
ध्रुव uint64_t APIR_LOADLIBRARY_MAX_WAIT_MS = 60 * 1000;  // 60s

स्थिर पूर्णाङ्क virtgpu_handshake(virtgpu * gpu) {
    apir_encoder * encoder;
    apir_decoder * decoder;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_HANDSHAKE, 0);
    यदि (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);
        प्रत्यागम 1;
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

    यदि (!decoder) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: failed to initiate the communication with the virglrenderer library. "
                   "Most likely, the wrong virglrenderer library was loaded in the hypervisor.",
                   __func__);
        प्रत्यागम 1;
    }

    /* read handshake return values */

    uint32_t host_major;
    uint32_t host_minor;

    यदि (ret_magic != APIR_HANDSHAKE_MAGIC) {
        GGML_ABORT(GGML_VIRTGPU "%s: handshake with the virglrenderer failed (code=%d | %s)", __func__, ret_magic,
                   apir_backend_initialize_error(ret_magic));
    } अन्यथा {
        apir_decode_uint32_t(decoder, &host_major);
        apir_decode_uint32_t(decoder, &host_minor);
    }

    remote_call_finish(gpu, encoder, decoder);

    यदि (ret_magic != APIR_HANDSHAKE_MAGIC) {
        प्रत्यागम 1;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: Guest is running with %u.%u\n", __func__, guest_major, guest_minor);
    GGML_LOG_INFO(GGML_VIRTGPU "%s: Host is running with %u.%u\n", __func__, host_major, host_minor);

    यदि (guest_major != host_major) {
        GGML_LOG_ERROR(GGML_VIRTGPU "Host major (%d) and guest major (%d) version differ\n", host_major, guest_major);
    } अन्यथा यदि (guest_minor != host_minor) {
        GGML_LOG_WARN(GGML_VIRTGPU "Host minor (%d) and guest minor (%d) version differ\n", host_minor, guest_minor);
    }

    प्रत्यागम 0;
}

स्थिर Apir_आरोपय_Library_Return_संहिता virtgpu_आरोपय_library(virtgpu * gpu) {
    apir_encoder *            encoder;
    apir_decoder *            decoder;
    Apir_आरोपय_Library_Return_संहिता ret;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_LOADLIBRARY, 0);
    यदि (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to prepare the API Remoting command encoder", __func__);
        प्रत्यागम APIR_आरोपय_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    दीर्घ दीर्घ call_duration_ns;

    ret = (Apir_आरोपय_Library_Return_संहिता) remote_call(gpu, encoder, &decoder, APIR_LOADLIBRARY_MAX_WAIT_MS,
                                                  &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting LoadLibrary");

    यदि (!decoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to trigger the API Remoting hypercall.\n", __func__);
        प्रत्यागम APIR_आरोपय_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    remote_call_finish(gpu, encoder, decoder);

    यदि (ret == APIR_आरोपय_LIBRARY_SUCCESS) {
        GGML_LOG_INFO(GGML_VIRTGPU "The API Remoting backend was successfully loaded and initialized\n");

        प्रत्यागम ret;
    }

    // something wrong happened, find out what.
    यदि (ret < APIR_आरोपय_LIBRARY_INIT_आधारः_सूचकांक) {
        यदि (ret == APIR_आरोपय_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library, "
                       "some environment variables are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_आरोपय_library_error(ret));
        } अन्यथा यदि (ret == APIR_आरोपय_LIBRARY_CANNOT_OPEN) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_आरोपय_library_error(ret));
        } अन्यथा यदि (ret == APIR_आरोपय_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: could not load the backend library, some symbols are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s) ",
                       __func__, apir_आरोपय_library_error(ret));
        } अन्यथा {
            GGML_ABORT(GGML_VIRTGPU "%s: virglrenderer could not load the API Remoting backend library. (%s - code %d)",
                       __func__, apir_आरोपय_library_error(ret), ret);
        }
        प्रत्यागम ret;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: virglrenderer successfully loaded the API Remoting backend library.\n", __func__);

    Apir_आरोपय_Library_Return_संहिता apir_ret = (Apir_आरोपय_Library_Return_संहिता) (ret - APIR_आरोपय_LIBRARY_INIT_आधारः_सूचकांक);

    यदि (apir_ret == APIR_आरोपय_LIBRARY_CANNOT_OPEN) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library. "
                   "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                   __func__, apir_आरोपय_library_error(apir_ret));
    } अन्यथा यदि (apir_ret == APIR_आरोपय_LIBRARY_चिह्नम्_MISSING) {
        GGML_ABORT(
            GGML_VIRTGPU
            "%s: the API Remoting backend library couldn't load the GGML backend library, some symbols are missing. "
            "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
            __func__, apir_आरोपय_library_error(apir_ret));
    } अन्यथा यदि (apir_ret < APIR_आरोपय_LIBRARY_INIT_आधारः_सूचकांक) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library: apir code=%d | %s)",
                   __func__, apir_ret, apir_आरोपय_library_error(apir_ret));
    } अन्यथा {
        uint32_t lib_ret = apir_ret - APIR_आरोपय_LIBRARY_INIT_आधारः_सूचकांक;
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library failed to initialize its backend library: apir code=%d)",
                   __func__, lib_ret);
    }
    प्रत्यागम ret;
}

virtgpu * create_virtgpu() {
    virtgpu * gpu = नूतन virtgpu();

    gpu->use_apir_capset = getenv("GGML_REMOTING_USE_APIR_CAPSET") != शून्यसूचकः;
    util_sparse_array_init(&gpu->shmem_array, परिमाण(virtgpu_shmem), 1024);

    // Initialize mutex to protect shared data_shmem buffer
    यदि (mtx_init(&gpu->दत्त_shmem_mutex, mtx_plain) != thrd_success) {
        विलोपय gpu;
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize data_shmem mutex", __func__);
        प्रत्यागम रिक्त;
    }

    यदि (virtgpu_open(gpu) != APIR_SUCCESS) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to open the virtgpu device\n", __func__);
        प्रत्यागम रिक्त;
    }

    यदि (virtgpu_init_capset(gpu) != APIR_SUCCESS) {
        यदि (gpu->use_apir_capset) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: failed to initialize the virtgpu APIR capset. Make sure that the virglrenderer library "
                       "supports it.",
                       __func__);
        } अन्यथा {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu Venus capset", __func__);
        }
        प्रत्यागम रिक्त;
    }

    यदि (virtgpu_init_context(gpu) != APIR_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the GPU context", __func__);
        प्रत्यागम रिक्त;
    }

    यदि (virtgpu_shmem_create(gpu, SHMEM_प्रत्युत्तरम्_आकार, &gpu->प्रत्युत्तरम्_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared reply memory pages", __func__);
        प्रत्यागम रिक्त;
    }

    यदि (virtgpu_shmem_create(gpu, SHMEM_दत्त_आकार, &gpu->दत्त_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared data memory pages", __func__);
        प्रत्यागम रिक्त;
    }

    यदि (virtgpu_handshake(gpu)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to handshake with the virglrenderer library", __func__);
        प्रत्यागम रिक्त;
    }

    यदि (virtgpu_आरोपय_library(gpu) != APIR_आरोपय_LIBRARY_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to load the backend library", __func__);
        प्रत्यागम रिक्त;
    }

    प्रत्यागम gpu;
}

स्थिर virt_gpu_परिणाम_t virtgpu_open(virtgpu * gpu) {
    drmDevicePtr devs[8];
    पूर्णाङ्क          count = drmGetDevices2(0, devs, ARRAY_आकार(devs));
    यदि (count < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to enumerate DRM devices\n", __func__);
        प्रत्यागम APIR_ERROR_INITIALIZATION_FAILED;
    }

    virt_gpu_परिणाम_t परिणाम = APIR_ERROR_INITIALIZATION_FAILED;
    प्रति (पूर्णाङ्क i = 0; i < count; i++) {
        परिणाम = virtgpu_open_device(gpu, devs[i]);
        यदि (परिणाम == APIR_SUCCESS) {
            भङ्ग;
        }
    }

    drmFreeDevices(devs, count);

    प्रत्यागम परिणाम;
}

स्थिर virt_gpu_परिणाम_t virtgpu_open_device(virtgpu * gpu, ध्रुव drmDevicePtr dev) {
    ध्रुव वर्ण * ग्रन्थिः_मार्गः = dev->nodes[DRM_ग्रन्थिः_RENDER];

    पूर्णाङ्क fd = उद्घाटय(ग्रन्थिः_मार्गः, O_RDWR | O_CLOEXEC);
    यदि (fd < 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to open %s", __func__, ग्रन्थिः_मार्गः);
        प्रत्यागम APIR_ERROR_INITIALIZATION_FAILED;
    }

    drm_संस्करणम्_Ptr संस्करणम्_2 = drm_Get_संस्करणम्(fd);
    यदि (!संस्करणम्_2 || वाक्यं_तुलय(संस्करणम्_2->name, "virtio_gpu") || संस्करणम्_2->संस्करणम्_major != 0) {
        यदि (संस्करणम्_2) {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: unknown DRM driver %s version %d\n", __func__, संस्करणम्_2->name,
                           संस्करणम्_2->संस्करणम्_major);
        } अन्यथा {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get DRM driver version\n", __func__);
        }

        यदि (संस्करणम्_2) {
            drm_Free_संस्करणम्(संस्करणम्_2);
        }
        close(fd);
        प्रत्यागम APIR_ERROR_INITIALIZATION_FAILED;
    }

    gpu->fd = fd;

    drm_Free_संस्करणम्(संस्करणम्_2);

    GGML_LOG_INFO(GGML_VIRTGPU "using DRM device %s\n", ग्रन्थिः_मार्गः);

    प्रत्यागम APIR_SUCCESS;
}

स्थिर virt_gpu_परिणाम_t virtgpu_init_context(virtgpu * gpu) {
    प्रतिज्ञा(!gpu->capset.संस्करणम्_2);
    ध्रुव पूर्णाङ्क ret = virtgpu_ioctl_context_init(gpu, gpu->capset.id);
    यदि (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to initialize context: %s\n", __func__, strerror(errno));
        प्रत्यागम APIR_ERROR_INITIALIZATION_FAILED;
    }

    प्रत्यागम APIR_SUCCESS;
}

स्थिर virt_gpu_परिणाम_t virtgpu_init_capset(virtgpu * gpu) {
    यदि (gpu->use_apir_capset) {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the APIR capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_APIR;
    } अन्यथा {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the Venus capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_VENUS;
    }
    gpu->capset.संस्करणम्_2 = 0;

    पूर्णाङ्क ret =
        virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.संस्करणम्_2, &gpu->capset.दत्त, परिमाण(gpu->capset.दत्त));

    यदि (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get APIR v%d capset: %s\n", __func__, gpu->capset.संस्करणम्_2,
                       strerror(errno));
        प्रत्यागम APIR_ERROR_INITIALIZATION_FAILED;
    }

    प्रतिज्ञा(gpu->capset.दत्त.supports_blob_resources);

    प्रत्यागम APIR_SUCCESS;
}

स्थिर पूर्णाङ्क virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id) {
    drm_virtgpu_context_समुच्चयः_param ctx_समुच्चयः_params[3] = {
        {
         .param = VIRTGPU_CONTEXT_PARAM_CAPSET_ID,
         .value = capset_id,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_अङ्कः_RINGS,
         .value = 1,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_POLL_RINGS_आवरणम्,
         .value = 0, /* don't generate drm_events on fence signaling */
        },
    };

    drm_virtgpu_context_init तर्काः_2 = {
        .अङ्कः_params     = ARRAY_आकार(ctx_समुच्चयः_params),
        .pad            = 0,
        .ctx_समुच्चयः_params = (uintptr_t) &ctx_समुच्चयः_params,
    };

    प्रत्यागम virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &तर्काः_2);
}

स्थिर पूर्णाङ्क virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                  virgl_renderer_capset id,
                                  uint32_t              संस्करणम्_2,
                                  शून्य *                capset,
                                  size_t                capset_आकार) {
    drm_virtgpu_get_caps तर्काः_2 = {
        .cap_समुच्चयः_id  = id,
        .cap_समुच्चयः_ver = संस्करणम्_2,
        .addr        = (uintptr_t) capset,
        .size        = (__u32) capset_आकार,
        .pad         = 0,
    };

    प्रत्यागम virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GET_CAPS, &तर्काः_2);
}

स्थिर uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param) {
    /* val must be zeroed because kernel only writes the lower 32 bits */
    uint64_t             val  = 0;
    drm_virtgpu_getparam तर्काः_2 = {
        .param = param,
        .value = (uintptr_t) &val,
    };

    ध्रुव पूर्णाङ्क ret = virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GETPARAM, &तर्काः_2);
    प्रत्यागम ret ? 0 : val;
}

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags) {
    /*
     * Prepare the command encoder and its buffer
     */

    thread_local वर्ण encoder_बफर[4096];

    thread_local apir_encoder enc;
    enc = {
        .cur   = encoder_बफर,
        .start = encoder_बफर,
        .end   = encoder_बफर + परिमाण(encoder_बफर),
        .fatal = असत्य,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    यदि (!gpu->use_apir_capset) {
        cmd_type += VENUS_COMMAND_TYPE_दीर्घता;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t प्रत्युत्तरम्_res_id = gpu->प्रत्युत्तरम्_shmem.res_id;
    apir_encode_uint32_t(&enc, &प्रत्युत्तरम्_res_id);

    प्रत्यागम &enc;
}

शून्य remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec) {
    अप्रयुक्तम्(gpu);

    यदि (!enc) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) encoder", __func__);
    }

    यदि (!dec) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) decoder", __func__);
    }

    यदि (apir_encoder_get_fatal(enc)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to encode the output parameters.", __func__);
    }

    यदि (apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to decode the input parameters.", __func__);
    }
}

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  encoder,
                     apir_decoder ** decoder,
                     भिन्न           max_wait_ms,
                     दीर्घ दीर्घ *     call_duration_ns) {
    /*
     * Prepare the reply notification pointer
     */

    चञ्चल std::atomic_uint * atomic_प्रत्युत्तरम्_notif = (चञ्चल std::atomic_uint *) gpu->प्रत्युत्तरम्_shmem.mmap_ptr;
    *atomic_प्रत्युत्तरम्_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    drm_virtgpu_execbuffer तर्काः_2 = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .अङ्कः_bo_handles = 0,

        .fence_fd         = 0,
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .अङ्कः_अन्तर्_syncobjs  = 0,
        .अङ्कः_बहिर्गमनम्_syncobjs = 0,
        .अन्तर्_syncobjs      = 0,
        .बहिर्गमनम्_syncobjs     = 0,
    };

    *decoder = रिक्त;

    पूर्णाङ्क ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &तर्काः_2);

    यदि (ret != 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: the virtgpu EXECBUFFER ioctl failed (%d)", __func__, ret);
    }

    /*
     * Wait for the response notification
     */
    timer_दत्त wait_host_प्रत्युत्तरम्_timer = { 0, 0, 0 };

    start_timer(&wait_host_प्रत्युत्तरम्_timer);

    timespec ts_start, ts_अन्तः;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    दीर्घ दीर्घ start_time = (दीर्घ दीर्घ) ts_start.tv_sec * 1000000000LL + ts_start.tv_nsec;

    तार्किक     timedout    = असत्य;
    uint32_t notif_मानम् = 0;
    यावत् (सत्य) {
        notif_मानम् = std::atomic_load_explicit(atomic_प्रत्युत्तरम्_notif, std::memory_order_acquire);

        यदि (notif_मानम् != 0) {
            भङ्ग;
        }

        int64_t आधारः_sleep_us = 15;

        os_time_sleep(आधारः_sleep_us);

        यदि (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_अन्तः);
            दीर्घ दीर्घ अन्तः_time    = (दीर्घ दीर्घ) ts_अन्तः.tv_sec * 1000000000LL + ts_अन्तः.tv_nsec;
            भिन्न     duration_ms = (अन्तः_time - start_time) / 1000000;

            यदि (duration_ms > max_wait_ms) {
                timedout = सत्य;
                भङ्ग;
            }
        }
    }

    यदि (call_duration_ns) {
        *call_duration_ns = stop_timer(&wait_host_प्रत्युत्तरम्_timer);
    }

    यदि (max_wait_ms && timedout) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: timed out waiting for the host answer...\n", __func__);
        प्रत्यागम APIR_FORWARD_TIMEOUT;
    }

    /*
     * Prepare the decoder
     */
    स्थिर apir_decoder response_dec;
    response_dec.cur = (वर्ण *) gpu->प्रत्युत्तरम्_shmem.mmap_ptr + परिमाण(*atomic_प्रत्युत्तरम्_notif);
    response_dec.end = (वर्ण *) gpu->प्रत्युत्तरम्_shmem.mmap_ptr + gpu->प्रत्युत्तरम्_shmem.mmap_आकार;
    *decoder         = &response_dec;

    // extract the actual return value from the notif flag
    uint32_t returned_मानम् = notif_मानम् - 1;
    प्रत्यागम returned_मानम्;
}

स्थिर शून्य log_call_duration(दीर्घ दीर्घ call_duration_ns, ध्रुव वर्ण * name) {
    द्विगुण call_duration_ms = (द्विगुण) call_duration_ns / 1e6;  // 1 millisecond = 1e6 nanoseconds
    द्विगुण call_duration_s  = (द्विगुण) call_duration_ns / 1e9;  // 1 second = 1e9 nanoseconds

    यदि (call_duration_s > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fs for the %s host reply...\n", call_duration_s, name);
    } अन्यथा यदि (call_duration_ms > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fms for the %s host reply...\n", call_duration_ms, name);
    } अन्यथा {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %lldns for the %s host reply...\n", call_duration_ns, name);
    }
}
