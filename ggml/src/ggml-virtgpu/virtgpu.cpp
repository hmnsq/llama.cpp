#شامل "virtgpu.h"
#شامل "ggml-remoting.h"

#شامل <stdio.h>
#شامل <unistd.h>

#شامل <cassert>
#شامل <cerrno>
#شامل <cstdlib>

ساکن virt_gpu_نتیجہ_t virtgpu_open_device(virtgpu * gpu, ثابت drmDevicePtr dev);
ساکن virt_gpu_نتیجہ_t virtgpu_open(virtgpu * gpu);

ساکن virt_gpu_نتیجہ_t virtgpu_init_capset(virtgpu * gpu);
ساکن virt_gpu_نتیجہ_t virtgpu_init_context(virtgpu * gpu);

ساکن صحیح      virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id);
ساکن صحیح      virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                       virgl_renderer_capset id,
                                       uint32_t              نسخہ_2,
                                       خالی *                capset,
                                       size_t                capset_سائز);
ساکن uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param);
ساکن خالی     virtgpu_init_renderer_info(virtgpu * gpu);

ساکن خالی log_call_duration(طویل طویل call_duration_ns, ثابت حرف * name);

ثابت uint64_t APIR_HANDSHAKE_MAX_WAIT_MS   = 2 * 1000;   // 2s
ثابت uint64_t APIR_LOADLIBRARY_MAX_WAIT_MS = 60 * 1000;  // 60s

ساکن صحیح virtgpu_handshake(virtgpu * gpu) {
    apir_encoder * encoder;
    apir_decoder * decoder;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_HANDSHAKE, 0);
    اگر (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);
        واپس 1;
    }

    /* write handshake props */

    uint32_t guest_major = APIR_PROTOCOL_MAJOR;
    uint32_t guest_minor = APIR_PROTOCOL_MINOR;
    apir_encode_uint32_t(encoder, &guest_major);
    apir_encode_uint32_t(encoder, &guest_minor);

    /* *** */

    uint32_t  ret_magic;
    طویل طویل call_duration_ns;
    ret_magic = remote_call(gpu, encoder, &decoder, APIR_HANDSHAKE_MAX_WAIT_MS, &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting handshake");

    اگر (!decoder) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: failed to initiate the communication with the virglrenderer library. "
                   "Most likely, the wrong virglrenderer library was loaded in the hypervisor.",
                   __func__);
        واپس 1;
    }

    /* read handshake return values */

    uint32_t host_major;
    uint32_t host_minor;

    اگر (ret_magic != APIR_HANDSHAKE_MAGIC) {
        GGML_ABORT(GGML_VIRTGPU "%s: handshake with the virglrenderer failed (code=%d | %s)", __func__, ret_magic,
                   apir_backend_initialize_error(ret_magic));
    } ورنہ {
        apir_decode_uint32_t(decoder, &host_major);
        apir_decode_uint32_t(decoder, &host_minor);
    }

    remote_call_اختتام(gpu, encoder, decoder);

    اگر (ret_magic != APIR_HANDSHAKE_MAGIC) {
        واپس 1;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: Guest is running with %u.%u\n", __func__, guest_major, guest_minor);
    GGML_LOG_INFO(GGML_VIRTGPU "%s: Host is running with %u.%u\n", __func__, host_major, host_minor);

    اگر (guest_major != host_major) {
        GGML_LOG_ERROR(GGML_VIRTGPU "Host major (%d) and guest major (%d) version differ\n", host_major, guest_major);
    } ورنہ اگر (guest_minor != host_minor) {
        GGML_LOG_WARN(GGML_VIRTGPU "Host minor (%d) and guest minor (%d) version differ\n", host_minor, guest_minor);
    }

    واپس 0;
}

ساکن Apir_بھرو_Library_Return_رمز virtgpu_بھرو_library(virtgpu * gpu) {
    apir_encoder *            encoder;
    apir_decoder *            decoder;
    Apir_بھرو_Library_Return_رمز ret;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_LOADLIBRARY, 0);
    اگر (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to prepare the API Remoting command encoder", __func__);
        واپس APIR_بھرو_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    طویل طویل call_duration_ns;

    ret = (Apir_بھرو_Library_Return_رمز) remote_call(gpu, encoder, &decoder, APIR_LOADLIBRARY_MAX_WAIT_MS,
                                                  &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting LoadLibrary");

    اگر (!decoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to trigger the API Remoting hypercall.\n", __func__);
        واپس APIR_بھرو_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    remote_call_اختتام(gpu, encoder, decoder);

    اگر (ret == APIR_بھرو_LIBRARY_SUCCESS) {
        GGML_LOG_INFO(GGML_VIRTGPU "The API Remoting backend was successfully loaded and initialized\n");

        واپس ret;
    }

    // something wrong happened, find out what.
    اگر (ret < APIR_بھرو_LIBRARY_INIT_بنیاد_اشاریہ) {
        اگر (ret == APIR_بھرو_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library, "
                       "some environment variables are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_بھرو_library_error(ret));
        } ورنہ اگر (ret == APIR_بھرو_LIBRARY_CANNOT_OPEN) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_بھرو_library_error(ret));
        } ورنہ اگر (ret == APIR_بھرو_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: could not load the backend library, some symbols are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s) ",
                       __func__, apir_بھرو_library_error(ret));
        } ورنہ {
            GGML_ABORT(GGML_VIRTGPU "%s: virglrenderer could not load the API Remoting backend library. (%s - code %d)",
                       __func__, apir_بھرو_library_error(ret), ret);
        }
        واپس ret;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: virglrenderer successfully loaded the API Remoting backend library.\n", __func__);

    Apir_بھرو_Library_Return_رمز apir_ret = (Apir_بھرو_Library_Return_رمز) (ret - APIR_بھرو_LIBRARY_INIT_بنیاد_اشاریہ);

    اگر (apir_ret == APIR_بھرو_LIBRARY_CANNOT_OPEN) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library. "
                   "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                   __func__, apir_بھرو_library_error(apir_ret));
    } ورنہ اگر (apir_ret == APIR_بھرو_LIBRARY_علامت_MISSING) {
        GGML_ABORT(
            GGML_VIRTGPU
            "%s: the API Remoting backend library couldn't load the GGML backend library, some symbols are missing. "
            "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
            __func__, apir_بھرو_library_error(apir_ret));
    } ورنہ اگر (apir_ret < APIR_بھرو_LIBRARY_INIT_بنیاد_اشاریہ) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library: apir code=%d | %s)",
                   __func__, apir_ret, apir_بھرو_library_error(apir_ret));
    } ورنہ {
        uint32_t lib_ret = apir_ret - APIR_بھرو_LIBRARY_INIT_بنیاد_اشاریہ;
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library failed to initialize its backend library: apir code=%d)",
                   __func__, lib_ret);
    }
    واپس ret;
}

virtgpu * create_virtgpu() {
    virtgpu * gpu = نیا virtgpu();

    gpu->use_apir_capset = getenv("GGML_REMOTING_USE_APIR_CAPSET") != خالی_اشاریہ;
    util_sparse_array_init(&gpu->shmem_array, حجم(virtgpu_shmem), 1024);

    // Initialize mutex to protect shared data_shmem buffer
    اگر (mtx_init(&gpu->ڈیٹا_shmem_mutex, mtx_plain) != thrd_success) {
        حذف gpu;
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize data_shmem mutex", __func__);
        واپس تہی;
    }

    اگر (virtgpu_open(gpu) != APIR_SUCCESS) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to open the virtgpu device\n", __func__);
        واپس تہی;
    }

    اگر (virtgpu_init_capset(gpu) != APIR_SUCCESS) {
        اگر (gpu->use_apir_capset) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: failed to initialize the virtgpu APIR capset. Make sure that the virglrenderer library "
                       "supports it.",
                       __func__);
        } ورنہ {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu Venus capset", __func__);
        }
        واپس تہی;
    }

    اگر (virtgpu_init_context(gpu) != APIR_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the GPU context", __func__);
        واپس تہی;
    }

    اگر (virtgpu_shmem_create(gpu, SHMEM_جواب_سائز, &gpu->جواب_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared reply memory pages", __func__);
        واپس تہی;
    }

    اگر (virtgpu_shmem_create(gpu, SHMEM_ڈیٹا_سائز, &gpu->ڈیٹا_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared data memory pages", __func__);
        واپس تہی;
    }

    اگر (virtgpu_handshake(gpu)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to handshake with the virglrenderer library", __func__);
        واپس تہی;
    }

    اگر (virtgpu_بھرو_library(gpu) != APIR_بھرو_LIBRARY_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to load the backend library", __func__);
        واپس تہی;
    }

    واپس gpu;
}

ساکن virt_gpu_نتیجہ_t virtgpu_open(virtgpu * gpu) {
    drmDevicePtr devs[8];
    صحیح          count = drmGetDevices2(0, devs, ARRAY_سائز(devs));
    اگر (count < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to enumerate DRM devices\n", __func__);
        واپس APIR_ERROR_INITIALIZATION_FAILED;
    }

    virt_gpu_نتیجہ_t نتیجہ = APIR_ERROR_INITIALIZATION_FAILED;
    برائے (صحیح i = 0; i < count; i++) {
        نتیجہ = virtgpu_open_device(gpu, devs[i]);
        اگر (نتیجہ == APIR_SUCCESS) {
            توڑو;
        }
    }

    drmFreeDevices(devs, count);

    واپس نتیجہ;
}

ساکن virt_gpu_نتیجہ_t virtgpu_open_device(virtgpu * gpu, ثابت drmDevicePtr dev) {
    ثابت حرف * گرہ_راستہ = dev->nodes[DRM_گرہ_RENDER];

    صحیح fd = open(گرہ_راستہ, O_RDWR | O_CLOEXEC);
    اگر (fd < 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to open %s", __func__, گرہ_راستہ);
        واپس APIR_ERROR_INITIALIZATION_FAILED;
    }

    drm_نسخہ_Ptr نسخہ_2 = drm_Get_نسخہ(fd);
    اگر (!نسخہ_2 || strcmp(نسخہ_2->name, "virtio_gpu") || نسخہ_2->نسخہ_major != 0) {
        اگر (نسخہ_2) {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: unknown DRM driver %s version %d\n", __func__, نسخہ_2->name,
                           نسخہ_2->نسخہ_major);
        } ورنہ {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get DRM driver version\n", __func__);
        }

        اگر (نسخہ_2) {
            drm_Free_نسخہ(نسخہ_2);
        }
        close(fd);
        واپس APIR_ERROR_INITIALIZATION_FAILED;
    }

    gpu->fd = fd;

    drm_Free_نسخہ(نسخہ_2);

    GGML_LOG_INFO(GGML_VIRTGPU "using DRM device %s\n", گرہ_راستہ);

    واپس APIR_SUCCESS;
}

ساکن virt_gpu_نتیجہ_t virtgpu_init_context(virtgpu * gpu) {
    أكد(!gpu->capset.نسخہ_2);
    ثابت صحیح ret = virtgpu_ioctl_context_init(gpu, gpu->capset.id);
    اگر (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to initialize context: %s\n", __func__, strerror(errno));
        واپس APIR_ERROR_INITIALIZATION_FAILED;
    }

    واپس APIR_SUCCESS;
}

ساکن virt_gpu_نتیجہ_t virtgpu_init_capset(virtgpu * gpu) {
    اگر (gpu->use_apir_capset) {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the APIR capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_APIR;
    } ورنہ {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the Venus capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_VENUS;
    }
    gpu->capset.نسخہ_2 = 0;

    صحیح ret =
        virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.نسخہ_2, &gpu->capset.ڈیٹا, حجم(gpu->capset.ڈیٹا));

    اگر (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get APIR v%d capset: %s\n", __func__, gpu->capset.نسخہ_2,
                       strerror(errno));
        واپس APIR_ERROR_INITIALIZATION_FAILED;
    }

    أكد(gpu->capset.ڈیٹا.supports_blob_resources);

    واپس APIR_SUCCESS;
}

ساکن صحیح virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id) {
    drm_virtgpu_context_سیٹ_param ctx_سیٹ_params[3] = {
        {
         .param = VIRTGPU_CONTEXT_PARAM_CAPSET_ID,
         .value = capset_id,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_عدد_RINGS,
         .value = 1,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_POLL_RINGS_نقاب,
         .value = 0, /* don't generate drm_events on fence signaling */
        },
    };

    drm_virtgpu_context_init دلائل_2 = {
        .عدد_params     = ARRAY_سائز(ctx_سیٹ_params),
        .pad            = 0,
        .ctx_سیٹ_params = (uintptr_t) &ctx_سیٹ_params,
    };

    واپس virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &دلائل_2);
}

ساکن صحیح virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                  virgl_renderer_capset id,
                                  uint32_t              نسخہ_2,
                                  خالی *                capset,
                                  size_t                capset_سائز) {
    drm_virtgpu_get_caps دلائل_2 = {
        .cap_سیٹ_id  = id,
        .cap_سیٹ_ver = نسخہ_2,
        .addr        = (uintptr_t) capset,
        .size        = (__u32) capset_سائز,
        .pad         = 0,
    };

    واپس virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GET_CAPS, &دلائل_2);
}

ساکن uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param) {
    /* val must be zeroed because kernel only writes the lower 32 bits */
    uint64_t             val  = 0;
    drm_virtgpu_getparam دلائل_2 = {
        .param = param,
        .value = (uintptr_t) &val,
    };

    ثابت صحیح ret = virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GETPARAM, &دلائل_2);
    واپس ret ? 0 : val;
}

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags) {
    /*
     * Prepare the command encoder and its buffer
     */

    thread_local حرف encoder_بفر[4096];

    thread_local apir_encoder enc;
    enc = {
        .cur   = encoder_بفر,
        .start = encoder_بفر,
        .end   = encoder_بفر + حجم(encoder_بفر),
        .fatal = جھوٹ,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    اگر (!gpu->use_apir_capset) {
        cmd_type += VENUS_COMMAND_TYPE_لمبائی;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t جواب_res_id = gpu->جواب_shmem.res_id;
    apir_encode_uint32_t(&enc, &جواب_res_id);

    واپس &enc;
}

خالی remote_call_اختتام(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec) {
    غیر_مستعمل(gpu);

    اگر (!enc) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) encoder", __func__);
    }

    اگر (!dec) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) decoder", __func__);
    }

    اگر (apir_encoder_get_fatal(enc)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to encode the output parameters.", __func__);
    }

    اگر (apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to decode the input parameters.", __func__);
    }
}

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  encoder,
                     apir_decoder ** decoder,
                     اعشاری           max_wait_ms,
                     طویل طویل *     call_duration_ns) {
    /*
     * Prepare the reply notification pointer
     */

    متلون std::atomic_uint * atomic_جواب_notif = (متلون std::atomic_uint *) gpu->جواب_shmem.mmap_ptr;
    *atomic_جواب_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    drm_virtgpu_execbuffer دلائل_2 = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .عدد_bo_handles = 0,

        .fence_fd         = 0,
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .عدد_اندر_syncobjs  = 0,
        .عدد_برآمد_syncobjs = 0,
        .اندر_syncobjs      = 0,
        .برآمد_syncobjs     = 0,
    };

    *decoder = تہی;

    صحیح ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &دلائل_2);

    اگر (ret != 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: the virtgpu EXECBUFFER ioctl failed (%d)", __func__, ret);
    }

    /*
     * Wait for the response notification
     */
    timer_ڈیٹا wait_host_جواب_timer = { 0, 0, 0 };

    آغاز_timer(&wait_host_جواب_timer);

    timespec ts_آغاز, ts_انتہا;
    clock_gettime(CLOCK_MONOTONIC, &ts_آغاز);
    طویل طویل آغاز_وقت = (طویل طویل) ts_آغاز.tv_sec * 1000000000LL + ts_آغاز.tv_nsec;

    منطقی     timedout    = جھوٹ;
    uint32_t notif_قدر = 0;
    جب (سچ) {
        notif_قدر = std::atomic_load_explicit(atomic_جواب_notif, std::memory_order_acquire);

        اگر (notif_قدر != 0) {
            توڑو;
        }

        int64_t بنیاد_sleep_us = 15;

        os_وقت_sleep(بنیاد_sleep_us);

        اگر (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_انتہا);
            طویل طویل انتہا_وقت    = (طویل طویل) ts_انتہا.tv_sec * 1000000000LL + ts_انتہا.tv_nsec;
            اعشاری     duration_ms = (انتہا_وقت - آغاز_وقت) / 1000000;

            اگر (duration_ms > max_wait_ms) {
                timedout = سچ;
                توڑو;
            }
        }
    }

    اگر (call_duration_ns) {
        *call_duration_ns = stop_timer(&wait_host_جواب_timer);
    }

    اگر (max_wait_ms && timedout) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: timed out waiting for the host answer...\n", __func__);
        واپس APIR_FORWARD_TIMEOUT;
    }

    /*
     * Prepare the decoder
     */
    ساکن apir_decoder response_dec;
    response_dec.cur = (حرف *) gpu->جواب_shmem.mmap_ptr + حجم(*atomic_جواب_notif);
    response_dec.end = (حرف *) gpu->جواب_shmem.mmap_ptr + gpu->جواب_shmem.mmap_سائز;
    *decoder         = &response_dec;

    // extract the actual return value from the notif flag
    uint32_t returned_قدر = notif_قدر - 1;
    واپس returned_قدر;
}

ساکن خالی log_call_duration(طویل طویل call_duration_ns, ثابت حرف * name) {
    دوہرا call_duration_ms = (دوہرا) call_duration_ns / 1e6;  // 1 millisecond = 1e6 nanoseconds
    دوہرا call_duration_s  = (دوہرا) call_duration_ns / 1e9;  // 1 second = 1e9 nanoseconds

    اگر (call_duration_s > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fs for the %s host reply...\n", call_duration_s, name);
    } ورنہ اگر (call_duration_ms > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fms for the %s host reply...\n", call_duration_ms, name);
    } ورنہ {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %lldns for the %s host reply...\n", call_duration_ns, name);
    }
}
