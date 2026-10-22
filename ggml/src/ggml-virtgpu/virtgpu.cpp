#ضمن "virtgpu.h"
#ضمن "ggml-remoting.h"

#ضمن <stdio.h>
#ضمن <unistd.h>

#ضمن <cassert>
#ضمن <cerrno>
#ضمن <cstdlib>

ساكن virt_gpu_نتیجہ_t virtgpu_open_device(virtgpu * gpu, ثابت drmDevicePtr dev);
ساكن virt_gpu_نتیجہ_t virtgpu_open(virtgpu * gpu);

ساكن virt_gpu_نتیجہ_t virtgpu_init_capset(virtgpu * gpu);
ساكن virt_gpu_نتیجہ_t virtgpu_init_context(virtgpu * gpu);

ساكن صحيح      virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id);
ساكن صحيح      virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                       virgl_renderer_capset id,
                                       uint32_t              نسخہ_2,
                                       فراغ *                capset,
                                       size_t                capset_ناپ);
ساكن uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param);
ساكن فراغ     virtgpu_init_renderer_info(virtgpu * gpu);

ساكن فراغ log_call_duration(لما لما call_duration_ns, ثابت حرف * name);

ثابت uint64_t APIR_HANDSHAKE_MAX_WAIT_MS   = 2 * 1000;   // 2s
ثابت uint64_t APIR_LOADLIBRARY_MAX_WAIT_MS = 60 * 1000;  // 60s

ساكن صحيح virtgpu_handshake(virtgpu * gpu) {
    apir_encoder * encoder;
    apir_decoder * decoder;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_HANDSHAKE, 0);
    إذا (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);
        أرجع 1;
    }

    /* write handshake props */

    uint32_t guest_major = APIR_PROTOCOL_MAJOR;
    uint32_t guest_minor = APIR_PROTOCOL_MINOR;
    apir_encode_uint32_t(encoder, &guest_major);
    apir_encode_uint32_t(encoder, &guest_minor);

    /* *** */

    uint32_t  ret_magic;
    لما لما call_duration_ns;
    ret_magic = remote_call(gpu, encoder, &decoder, APIR_HANDSHAKE_MAX_WAIT_MS, &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting handshake");

    إذا (!decoder) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: failed to initiate the communication with the virglrenderer library. "
                   "Most likely, the wrong virglrenderer library was loaded in the hypervisor.",
                   __func__);
        أرجع 1;
    }

    /* read handshake return values */

    uint32_t host_major;
    uint32_t host_minor;

    إذا (ret_magic != APIR_HANDSHAKE_MAGIC) {
        GGML_ABORT(GGML_VIRTGPU "%s: handshake with the virglrenderer failed (code=%d | %s)", __func__, ret_magic,
                   apir_backend_initialize_error(ret_magic));
    } وإلا {
        apir_decode_uint32_t(decoder, &host_major);
        apir_decode_uint32_t(decoder, &host_minor);
    }

    remote_call_انت(gpu, encoder, decoder);

    إذا (ret_magic != APIR_HANDSHAKE_MAGIC) {
        أرجع 1;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: Guest is running with %u.%u\n", __func__, guest_major, guest_minor);
    GGML_LOG_INFO(GGML_VIRTGPU "%s: Host is running with %u.%u\n", __func__, host_major, host_minor);

    إذا (guest_major != host_major) {
        GGML_LOG_ERROR(GGML_VIRTGPU "Host major (%d) and guest major (%d) version differ\n", host_major, guest_major);
    } وإلا إذا (guest_minor != host_minor) {
        GGML_LOG_WARN(GGML_VIRTGPU "Host minor (%d) and guest minor (%d) version differ\n", host_minor, guest_minor);
    }

    أرجع 0;
}

ساكن Apir_لدو_Library_Return_کوڈ virtgpu_لدو_library(virtgpu * gpu) {
    apir_encoder *            encoder;
    apir_decoder *            decoder;
    Apir_لدو_Library_Return_کوڈ ret;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_LOADLIBRARY, 0);
    إذا (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to prepare the API Remoting command encoder", __func__);
        أرجع APIR_لدو_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    لما لما call_duration_ns;

    ret = (Apir_لدو_Library_Return_کوڈ) remote_call(gpu, encoder, &decoder, APIR_LOADLIBRARY_MAX_WAIT_MS,
                                                  &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting LoadLibrary");

    إذا (!decoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to trigger the API Remoting hypercall.\n", __func__);
        أرجع APIR_لدو_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    remote_call_انت(gpu, encoder, decoder);

    إذا (ret == APIR_لدو_LIBRARY_SUCCESS) {
        GGML_LOG_INFO(GGML_VIRTGPU "The API Remoting backend was successfully loaded and initialized\n");

        أرجع ret;
    }

    // something wrong happened, find out what.
    إذا (ret < APIR_لدو_LIBRARY_INIT_ادھار_اشاریہ) {
        إذا (ret == APIR_لدو_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library, "
                       "some environment variables are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_لدو_library_error(ret));
        } وإلا إذا (ret == APIR_لدو_LIBRARY_CANNOT_OPEN) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_لدو_library_error(ret));
        } وإلا إذا (ret == APIR_لدو_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: could not load the backend library, some symbols are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s) ",
                       __func__, apir_لدو_library_error(ret));
        } وإلا {
            GGML_ABORT(GGML_VIRTGPU "%s: virglrenderer could not load the API Remoting backend library. (%s - code %d)",
                       __func__, apir_لدو_library_error(ret), ret);
        }
        أرجع ret;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: virglrenderer successfully loaded the API Remoting backend library.\n", __func__);

    Apir_لدو_Library_Return_کوڈ apir_ret = (Apir_لدو_Library_Return_کوڈ) (ret - APIR_لدو_LIBRARY_INIT_ادھار_اشاریہ);

    إذا (apir_ret == APIR_لدو_LIBRARY_CANNOT_OPEN) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library. "
                   "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                   __func__, apir_لدو_library_error(apir_ret));
    } وإلا إذا (apir_ret == APIR_لدو_LIBRARY_علامت_MISSING) {
        GGML_ABORT(
            GGML_VIRTGPU
            "%s: the API Remoting backend library couldn't load the GGML backend library, some symbols are missing. "
            "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
            __func__, apir_لدو_library_error(apir_ret));
    } وإلا إذا (apir_ret < APIR_لدو_LIBRARY_INIT_ادھار_اشاریہ) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library: apir code=%d | %s)",
                   __func__, apir_ret, apir_لدو_library_error(apir_ret));
    } وإلا {
        uint32_t lib_ret = apir_ret - APIR_لدو_LIBRARY_INIT_ادھار_اشاریہ;
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library failed to initialize its backend library: apir code=%d)",
                   __func__, lib_ret);
    }
    أرجع ret;
}

virtgpu * create_virtgpu() {
    virtgpu * gpu = جديد virtgpu();

    gpu->use_apir_capset = getenv("GGML_REMOTING_USE_APIR_CAPSET") != خالی_سوچک;
    util_sparse_array_init(&gpu->shmem_array, حجم(virtgpu_shmem), 1024);

    // Initialize mutex to protect shared data_shmem buffer
    إذا (mtx_init(&gpu->انکڑے_shmem_mutex, mtx_plain) != thrd_success) {
        مٹاؤ gpu;
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize data_shmem mutex", __func__);
        أرجع عدم;
    }

    إذا (virtgpu_open(gpu) != APIR_SUCCESS) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to open the virtgpu device\n", __func__);
        أرجع عدم;
    }

    إذا (virtgpu_init_capset(gpu) != APIR_SUCCESS) {
        إذا (gpu->use_apir_capset) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: failed to initialize the virtgpu APIR capset. Make sure that the virglrenderer library "
                       "supports it.",
                       __func__);
        } وإلا {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu Venus capset", __func__);
        }
        أرجع عدم;
    }

    إذا (virtgpu_init_context(gpu) != APIR_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the GPU context", __func__);
        أرجع عدم;
    }

    إذا (virtgpu_shmem_create(gpu, SHMEM_جواب_ناپ, &gpu->جواب_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared reply memory pages", __func__);
        أرجع عدم;
    }

    إذا (virtgpu_shmem_create(gpu, SHMEM_انکڑے_ناپ, &gpu->انکڑے_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared data memory pages", __func__);
        أرجع عدم;
    }

    إذا (virtgpu_handshake(gpu)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to handshake with the virglrenderer library", __func__);
        أرجع عدم;
    }

    إذا (virtgpu_لدو_library(gpu) != APIR_لدو_LIBRARY_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to load the backend library", __func__);
        أرجع عدم;
    }

    أرجع gpu;
}

ساكن virt_gpu_نتیجہ_t virtgpu_open(virtgpu * gpu) {
    drmDevicePtr devs[8];
    صحيح          count = drmGetDevices2(0, devs, ARRAY_ناپ(devs));
    إذا (count < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to enumerate DRM devices\n", __func__);
        أرجع APIR_ERROR_INITIALIZATION_FAILED;
    }

    virt_gpu_نتیجہ_t نتیجہ = APIR_ERROR_INITIALIZATION_FAILED;
    لكل (صحيح i = 0; i < count; i++) {
        نتیجہ = virtgpu_open_device(gpu, devs[i]);
        إذا (نتیجہ == APIR_SUCCESS) {
            اكسر;
        }
    }

    drmFreeDevices(devs, count);

    أرجع نتیجہ;
}

ساكن virt_gpu_نتیجہ_t virtgpu_open_device(virtgpu * gpu, ثابت drmDevicePtr dev) {
    ثابت حرف * گرہ_رستہ = dev->nodes[DRM_گرہ_RENDER];

    صحيح fd = open(گرہ_رستہ, O_RDWR | O_CLOEXEC);
    إذا (fd < 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to open %s", __func__, گرہ_رستہ);
        أرجع APIR_ERROR_INITIALIZATION_FAILED;
    }

    drm_نسخہ_Ptr نسخہ_2 = drm_Get_نسخہ(fd);
    إذا (!نسخہ_2 || strcmp(نسخہ_2->name, "virtio_gpu") || نسخہ_2->نسخہ_major != 0) {
        إذا (نسخہ_2) {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: unknown DRM driver %s version %d\n", __func__, نسخہ_2->name,
                           نسخہ_2->نسخہ_major);
        } وإلا {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get DRM driver version\n", __func__);
        }

        إذا (نسخہ_2) {
            drm_Free_نسخہ(نسخہ_2);
        }
        close(fd);
        أرجع APIR_ERROR_INITIALIZATION_FAILED;
    }

    gpu->fd = fd;

    drm_Free_نسخہ(نسخہ_2);

    GGML_LOG_INFO(GGML_VIRTGPU "using DRM device %s\n", گرہ_رستہ);

    أرجع APIR_SUCCESS;
}

ساكن virt_gpu_نتیجہ_t virtgpu_init_context(virtgpu * gpu) {
    أكد(!gpu->capset.نسخہ_2);
    ثابت صحيح ret = virtgpu_ioctl_context_init(gpu, gpu->capset.id);
    إذا (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to initialize context: %s\n", __func__, strerror(errno));
        أرجع APIR_ERROR_INITIALIZATION_FAILED;
    }

    أرجع APIR_SUCCESS;
}

ساكن virt_gpu_نتیجہ_t virtgpu_init_capset(virtgpu * gpu) {
    إذا (gpu->use_apir_capset) {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the APIR capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_APIR;
    } وإلا {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the Venus capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_VENUS;
    }
    gpu->capset.نسخہ_2 = 0;

    صحيح ret =
        virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.نسخہ_2, &gpu->capset.انکڑے, حجم(gpu->capset.انکڑے));

    إذا (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get APIR v%d capset: %s\n", __func__, gpu->capset.نسخہ_2,
                       strerror(errno));
        أرجع APIR_ERROR_INITIALIZATION_FAILED;
    }

    أكد(gpu->capset.انکڑے.supports_blob_resources);

    أرجع APIR_SUCCESS;
}

ساكن صحيح virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id) {
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

    drm_virtgpu_context_init دلیلاں_2 = {
        .عدد_params     = ARRAY_ناپ(ctx_سیٹ_params),
        .pad            = 0,
        .ctx_سیٹ_params = (uintptr_t) &ctx_سیٹ_params,
    };

    أرجع virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &دلیلاں_2);
}

ساكن صحيح virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                  virgl_renderer_capset id,
                                  uint32_t              نسخہ_2,
                                  فراغ *                capset,
                                  size_t                capset_ناپ) {
    drm_virtgpu_get_caps دلیلاں_2 = {
        .cap_سیٹ_id  = id,
        .cap_سیٹ_ver = نسخہ_2,
        .addr        = (uintptr_t) capset,
        .size        = (__u32) capset_ناپ,
        .pad         = 0,
    };

    أرجع virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GET_CAPS, &دلیلاں_2);
}

ساكن uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param) {
    /* val must be zeroed because kernel only writes the lower 32 bits */
    uint64_t             val  = 0;
    drm_virtgpu_getparam دلیلاں_2 = {
        .param = param,
        .value = (uintptr_t) &val,
    };

    ثابت صحيح ret = virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GETPARAM, &دلیلاں_2);
    أرجع ret ? 0 : val;
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
        .fatal = خطأ,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    إذا (!gpu->use_apir_capset) {
        cmd_type += VENUS_COMMAND_TYPE_لمائی;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t جواب_res_id = gpu->جواب_shmem.res_id;
    apir_encode_uint32_t(&enc, &جواب_res_id);

    أرجع &enc;
}

فراغ remote_call_انت(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec) {
    بن_ورتیا(gpu);

    إذا (!enc) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) encoder", __func__);
    }

    إذا (!dec) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) decoder", __func__);
    }

    إذا (apir_encoder_get_fatal(enc)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to encode the output parameters.", __func__);
    }

    إذا (apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to decode the input parameters.", __func__);
    }
}

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  encoder,
                     apir_decoder ** decoder,
                     عائم           max_wait_ms,
                     لما لما *     call_duration_ns) {
    /*
     * Prepare the reply notification pointer
     */

    متطاير std::atomic_uint * atomic_جواب_notif = (متطاير std::atomic_uint *) gpu->جواب_shmem.mmap_ptr;
    *atomic_جواب_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    drm_virtgpu_execbuffer دلیلاں_2 = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .عدد_bo_handles = 0,

        .fence_fd         = 0,
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .عدد_وچ_syncobjs  = 0,
        .عدد_باہرلا_syncobjs = 0,
        .وچ_syncobjs      = 0,
        .باہرلا_syncobjs     = 0,
    };

    *decoder = عدم;

    صحيح ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &دلیلاں_2);

    إذا (ret != 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: the virtgpu EXECBUFFER ioctl failed (%d)", __func__, ret);
    }

    /*
     * Wait for the response notification
     */
    timer_انکڑے wait_host_جواب_timer = { 0, 0, 0 };

    شروع_timer(&wait_host_جواب_timer);

    timespec ts_شروع, ts_سرا;
    clock_gettime(CLOCK_MONOTONIC, &ts_شروع);
    لما لما شروع_ویلا = (لما لما) ts_شروع.tv_sec * 1000000000LL + ts_شروع.tv_nsec;

    منطقي     timedout    = خطأ;
    uint32_t notif_قدر = 0;
    طالما (سچ) {
        notif_قدر = std::atomic_load_explicit(atomic_جواب_notif, std::memory_order_acquire);

        إذا (notif_قدر != 0) {
            اكسر;
        }

        int64_t ادھار_sleep_us = 15;

        os_ویلا_sleep(ادھار_sleep_us);

        إذا (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_سرا);
            لما لما سرا_ویلا    = (لما لما) ts_سرا.tv_sec * 1000000000LL + ts_سرا.tv_nsec;
            عائم     duration_ms = (سرا_ویلا - شروع_ویلا) / 1000000;

            إذا (duration_ms > max_wait_ms) {
                timedout = سچ;
                اكسر;
            }
        }
    }

    إذا (call_duration_ns) {
        *call_duration_ns = stop_timer(&wait_host_جواب_timer);
    }

    إذا (max_wait_ms && timedout) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: timed out waiting for the host answer...\n", __func__);
        أرجع APIR_FORWARD_TIMEOUT;
    }

    /*
     * Prepare the decoder
     */
    ساكن apir_decoder response_dec;
    response_dec.cur = (حرف *) gpu->جواب_shmem.mmap_ptr + حجم(*atomic_جواب_notif);
    response_dec.end = (حرف *) gpu->جواب_shmem.mmap_ptr + gpu->جواب_shmem.mmap_ناپ;
    *decoder         = &response_dec;

    // extract the actual return value from the notif flag
    uint32_t returned_قدر = notif_قدر - 1;
    أرجع returned_قدر;
}

ساكن فراغ log_call_duration(لما لما call_duration_ns, ثابت حرف * name) {
    مزدوج call_duration_ms = (مزدوج) call_duration_ns / 1e6;  // 1 millisecond = 1e6 nanoseconds
    مزدوج call_duration_s  = (مزدوج) call_duration_ns / 1e9;  // 1 second = 1e9 nanoseconds

    إذا (call_duration_s > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fs for the %s host reply...\n", call_duration_s, name);
    } وإلا إذا (call_duration_ms > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fms for the %s host reply...\n", call_duration_ms, name);
    } وإلا {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %lldns for the %s host reply...\n", call_duration_ns, name);
    }
}
