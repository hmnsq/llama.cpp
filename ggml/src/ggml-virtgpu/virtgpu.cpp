#ಅನ್ತರ್ಭಾವ "virtgpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <cstdlib>

ಸ್ಥಿರ virt_gpu_ಫಲಿತಾಂಶ_t virtgpu_open_device(virtgpu * gpu, ಸ್ಥಿರಾಂಕ drmDevicePtr dev);
ಸ್ಥಿರ virt_gpu_ಫಲಿತಾಂಶ_t virtgpu_open(virtgpu * gpu);

ಸ್ಥಿರ virt_gpu_ಫಲಿತಾಂಶ_t virtgpu_init_capset(virtgpu * gpu);
ಸ್ಥಿರ virt_gpu_ಫಲಿತಾಂಶ_t virtgpu_init_context(virtgpu * gpu);

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ      virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id);
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ      virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                       virgl_renderer_capset id,
                                       uint32_t              ಆವೃತ್ತಿ_2,
                                       ಶೂನ್ಯ *                capset,
                                       size_t                capset_ಗಾತ್ರ);
ಸ್ಥಿರ uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param);
ಸ್ಥಿರ ಶೂನ್ಯ     virtgpu_init_renderer_info(virtgpu * gpu);

ಸ್ಥಿರ ಶೂನ್ಯ log_call_duration(ದೀರ್ಘ ದೀರ್ಘ call_duration_ns, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name);

ಸ್ಥಿರಾಂಕ uint64_t APIR_HANDSHAKE_MAX_WAIT_MS   = 2 * 1000;   // 2s
ಸ್ಥಿರಾಂಕ uint64_t APIR_LOADLIBRARY_MAX_WAIT_MS = 60 * 1000;  // 60s

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ virtgpu_handshake(virtgpu * gpu) {
    apir_encoder * encoder;
    apir_decoder * decoder;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_HANDSHAKE, 0);
    ಆದರೆ (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to prepare the remote call encoder", __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    /* write handshake props */

    uint32_t guest_major = APIR_PROTOCOL_MAJOR;
    uint32_t guest_minor = APIR_PROTOCOL_MINOR;
    apir_encode_uint32_t(encoder, &guest_major);
    apir_encode_uint32_t(encoder, &guest_minor);

    /* *** */

    uint32_t  ret_magic;
    ದೀರ್ಘ ದೀರ್ಘ call_duration_ns;
    ret_magic = remote_call(gpu, encoder, &decoder, APIR_HANDSHAKE_MAX_WAIT_MS, &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting handshake");

    ಆದರೆ (!decoder) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: failed to initiate the communication with the virglrenderer library. "
                   "Most likely, the wrong virglrenderer library was loaded in the hypervisor.",
                   __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    /* read handshake return values */

    uint32_t host_major;
    uint32_t host_minor;

    ಆದರೆ (ret_magic != APIR_HANDSHAKE_MAGIC) {
        GGML_ABORT(GGML_VIRTGPU "%s: handshake with the virglrenderer failed (code=%d | %s)", __func__, ret_magic,
                   apir_backend_initialize_error(ret_magic));
    } ಇಲ್ಲದಿದ್ದರೆ {
        apir_decode_uint32_t(decoder, &host_major);
        apir_decode_uint32_t(decoder, &host_minor);
    }

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಆದರೆ (ret_magic != APIR_HANDSHAKE_MAGIC) {
        ಹಿಂತಿರುಗಿ 1;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: Guest is running with %u.%u\n", __func__, guest_major, guest_minor);
    GGML_LOG_INFO(GGML_VIRTGPU "%s: Host is running with %u.%u\n", __func__, host_major, host_minor);

    ಆದರೆ (guest_major != host_major) {
        GGML_LOG_ERROR(GGML_VIRTGPU "Host major (%d) and guest major (%d) version differ\n", host_major, guest_major);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (guest_minor != host_minor) {
        GGML_LOG_WARN(GGML_VIRTGPU "Host minor (%d) and guest minor (%d) version differ\n", host_minor, guest_minor);
    }

    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ Apir_ತುಂಬು_Library_Return_ಸಂಕೇತ virtgpu_ತುಂಬು_library(virtgpu * gpu) {
    apir_encoder *            encoder;
    apir_decoder *            decoder;
    Apir_ತುಂಬು_Library_Return_ಸಂಕೇತ ret;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_LOADLIBRARY, 0);
    ಆದರೆ (!encoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to prepare the API Remoting command encoder", __func__);
        ಹಿಂತಿರುಗಿ APIR_ತುಂಬು_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    ದೀರ್ಘ ದೀರ್ಘ call_duration_ns;

    ret = (Apir_ತುಂಬು_Library_Return_ಸಂಕೇತ) remote_call(gpu, encoder, &decoder, APIR_LOADLIBRARY_MAX_WAIT_MS,
                                                  &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting LoadLibrary");

    ಆದರೆ (!decoder) {
        GGML_ABORT(GGML_VIRTGPU "%s: hypercall error: failed to trigger the API Remoting hypercall.\n", __func__);
        ಹಿಂತಿರುಗಿ APIR_ತುಂಬು_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಆದರೆ (ret == APIR_ತುಂಬು_LIBRARY_SUCCESS) {
        GGML_LOG_INFO(GGML_VIRTGPU "The API Remoting backend was successfully loaded and initialized\n");

        ಹಿಂತಿರುಗಿ ret;
    }

    // something wrong happened, find out what.
    ಆದರೆ (ret < APIR_ತುಂಬು_LIBRARY_INIT_ತಳ_ಸೂಚ್ಯಂಕ) {
        ಆದರೆ (ret == APIR_ತುಂಬು_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library, "
                       "some environment variables are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_ತುಂಬು_library_error(ret));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ret == APIR_ತುಂಬು_LIBRARY_CANNOT_OPEN) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: virglrenderer could not open the API Remoting backend library. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                       __func__, apir_ತುಂಬು_library_error(ret));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ret == APIR_ತುಂಬು_LIBRARY_ENV_VAR_MISSING) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: could not load the backend library, some symbols are missing. "
                       "Make sure virglrenderer is correctly configured by the hypervisor. (%s) ",
                       __func__, apir_ತುಂಬು_library_error(ret));
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT(GGML_VIRTGPU "%s: virglrenderer could not load the API Remoting backend library. (%s - code %d)",
                       __func__, apir_ತುಂಬು_library_error(ret), ret);
        }
        ಹಿಂತಿರುಗಿ ret;
    }

    GGML_LOG_INFO(GGML_VIRTGPU "%s: virglrenderer successfully loaded the API Remoting backend library.\n", __func__);

    Apir_ತುಂಬು_Library_Return_ಸಂಕೇತ apir_ret = (Apir_ತುಂಬು_Library_Return_ಸಂಕೇತ) (ret - APIR_ತುಂಬು_LIBRARY_INIT_ತಳ_ಸೂಚ್ಯಂಕ);

    ಆದರೆ (apir_ret == APIR_ತುಂಬು_LIBRARY_CANNOT_OPEN) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library. "
                   "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
                   __func__, apir_ತುಂಬು_library_error(apir_ret));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (apir_ret == APIR_ತುಂಬು_LIBRARY_ಚಿಹ್ನೆ_MISSING) {
        GGML_ABORT(
            GGML_VIRTGPU
            "%s: the API Remoting backend library couldn't load the GGML backend library, some symbols are missing. "
            "Make sure virglrenderer is correctly configured by the hypervisor. (%s)",
            __func__, apir_ತುಂಬು_library_error(apir_ret));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (apir_ret < APIR_ತುಂಬು_LIBRARY_INIT_ತಳ_ಸೂಚ್ಯಂಕ) {
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library couldn't load the GGML backend library: apir code=%d | %s)",
                   __func__, apir_ret, apir_ತುಂಬು_library_error(apir_ret));
    } ಇಲ್ಲದಿದ್ದರೆ {
        uint32_t lib_ret = apir_ret - APIR_ತುಂಬು_LIBRARY_INIT_ತಳ_ಸೂಚ್ಯಂಕ;
        GGML_ABORT(GGML_VIRTGPU
                   "%s: the API Remoting backend library failed to initialize its backend library: apir code=%d)",
                   __func__, lib_ret);
    }
    ಹಿಂತಿರುಗಿ ret;
}

virtgpu * create_virtgpu() {
    virtgpu * gpu = ಹೊಸ virtgpu();

    gpu->use_apir_capset = getenv("GGML_REMOTING_USE_APIR_CAPSET") != ಶೂನ್ಯಸೂಚಕ;
    util_sparse_array_init(&gpu->shmem_array, ಗಾತ್ರ(virtgpu_shmem), 1024);

    // Initialize mutex to protect shared data_shmem buffer
    ಆದರೆ (mtx_init(&gpu->ದತ್ತಾಂಶ_shmem_mutex, mtx_plain) != thrd_success) {
        ಅಳಿಸು gpu;
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize data_shmem mutex", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (virtgpu_open(gpu) != APIR_SUCCESS) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to open the virtgpu device\n", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (virtgpu_init_capset(gpu) != APIR_SUCCESS) {
        ಆದರೆ (gpu->use_apir_capset) {
            GGML_ABORT(GGML_VIRTGPU
                       "%s: failed to initialize the virtgpu APIR capset. Make sure that the virglrenderer library "
                       "supports it.",
                       __func__);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the virtgpu Venus capset", __func__);
        }
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (virtgpu_init_context(gpu) != APIR_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to initialize the GPU context", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (virtgpu_shmem_create(gpu, SHMEM_ಉತ್ತರ_ಗಾತ್ರ, &gpu->ಉತ್ತರ_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared reply memory pages", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (virtgpu_shmem_create(gpu, SHMEM_ದತ್ತಾಂಶ_ಗಾತ್ರ, &gpu->ದತ್ತಾಂಶ_shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to create the shared data memory pages", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (virtgpu_handshake(gpu)) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to handshake with the virglrenderer library", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (virtgpu_ತುಂಬು_library(gpu) != APIR_ತುಂಬು_LIBRARY_SUCCESS) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to load the backend library", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ gpu;
}

ಸ್ಥಿರ virt_gpu_ಫಲಿತಾಂಶ_t virtgpu_open(virtgpu * gpu) {
    drmDevicePtr devs[8];
    ಪೂರ್ಣಾಂಕ          count = drmGetDevices2(0, devs, ARRAY_ಗಾತ್ರ(devs));
    ಆದರೆ (count < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to enumerate DRM devices\n", __func__);
        ಹಿಂತಿರುಗಿ APIR_ERROR_INITIALIZATION_FAILED;
    }

    virt_gpu_ಫಲಿತಾಂಶ_t ಫಲಿತಾಂಶ = APIR_ERROR_INITIALIZATION_FAILED;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < count; i++) {
        ಫಲಿತಾಂಶ = virtgpu_open_device(gpu, devs[i]);
        ಆದರೆ (ಫಲಿತಾಂಶ == APIR_SUCCESS) {
            ನಿಲ್ಲಿಸು;
        }
    }

    drmFreeDevices(devs, count);

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ virt_gpu_ಫಲಿತಾಂಶ_t virtgpu_open_device(virtgpu * gpu, ಸ್ಥಿರಾಂಕ drmDevicePtr dev) {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಗಂಟು_ಮಾರ್ಗ = dev->nodes[DRM_ಗಂಟು_RENDER];

    ಪೂರ್ಣಾಂಕ fd = open(ಗಂಟು_ಮಾರ್ಗ, O_RDWR | O_CLOEXEC);
    ಆದರೆ (fd < 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: failed to open %s", __func__, ಗಂಟು_ಮಾರ್ಗ);
        ಹಿಂತಿರುಗಿ APIR_ERROR_INITIALIZATION_FAILED;
    }

    drm_ಆವೃತ್ತಿ_Ptr ಆವೃತ್ತಿ_2 = drm_Get_ಆವೃತ್ತಿ(fd);
    ಆದರೆ (!ಆವೃತ್ತಿ_2 || strcmp(ಆವೃತ್ತಿ_2->name, "virtio_gpu") || ಆವೃತ್ತಿ_2->ಆವೃತ್ತಿ_major != 0) {
        ಆದರೆ (ಆವೃತ್ತಿ_2) {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: unknown DRM driver %s version %d\n", __func__, ಆವೃತ್ತಿ_2->name,
                           ಆವೃತ್ತಿ_2->ಆವೃತ್ತಿ_major);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get DRM driver version\n", __func__);
        }

        ಆದರೆ (ಆವೃತ್ತಿ_2) {
            drm_Free_ಆವೃತ್ತಿ(ಆವೃತ್ತಿ_2);
        }
        close(fd);
        ಹಿಂತಿರುಗಿ APIR_ERROR_INITIALIZATION_FAILED;
    }

    gpu->fd = fd;

    drm_Free_ಆವೃತ್ತಿ(ಆವೃತ್ತಿ_2);

    GGML_LOG_INFO(GGML_VIRTGPU "using DRM device %s\n", ಗಂಟು_ಮಾರ್ಗ);

    ಹಿಂತಿರುಗಿ APIR_SUCCESS;
}

ಸ್ಥಿರ virt_gpu_ಫಲಿತಾಂಶ_t virtgpu_init_context(virtgpu * gpu) {
    ಪ್ರತಿಜ್ಞಾ(!gpu->capset.ಆವೃತ್ತಿ_2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ret = virtgpu_ioctl_context_init(gpu, gpu->capset.id);
    ಆದರೆ (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to initialize context: %s\n", __func__, strerror(errno));
        ಹಿಂತಿರುಗಿ APIR_ERROR_INITIALIZATION_FAILED;
    }

    ಹಿಂತಿರುಗಿ APIR_SUCCESS;
}

ಸ್ಥಿರ virt_gpu_ಫಲಿತಾಂಶ_t virtgpu_init_capset(virtgpu * gpu) {
    ಆದರೆ (gpu->use_apir_capset) {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the APIR capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_APIR;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_INFO(GGML_VIRTGPU "Using the Venus capset\n");
        gpu->capset.id = VIRTGPU_DRM_CAPSET_VENUS;
    }
    gpu->capset.ಆವೃತ್ತಿ_2 = 0;

    ಪೂರ್ಣಾಂಕ ret =
        virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.ಆವೃತ್ತಿ_2, &gpu->capset.ದತ್ತಾಂಶ, ಗಾತ್ರ(gpu->capset.ದತ್ತಾಂಶ));

    ಆದರೆ (ret) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: failed to get APIR v%d capset: %s\n", __func__, gpu->capset.ಆವೃತ್ತಿ_2,
                       strerror(errno));
        ಹಿಂತಿರುಗಿ APIR_ERROR_INITIALIZATION_FAILED;
    }

    ಪ್ರತಿಜ್ಞಾ(gpu->capset.ದತ್ತಾಂಶ.supports_blob_resources);

    ಹಿಂತಿರುಗಿ APIR_SUCCESS;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ virtgpu_ioctl_context_init(virtgpu * gpu, virgl_renderer_capset capset_id) {
    drm_virtgpu_context_ಗಣ_param ctx_ಗಣ_params[3] = {
        {
         .param = VIRTGPU_CONTEXT_PARAM_CAPSET_ID,
         .value = capset_id,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_ಅಂಕೆ_RINGS,
         .value = 1,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_POLL_RINGS_ಮುಖವಾಡ,
         .value = 0, /* don't generate drm_events on fence signaling */
        },
    };

    drm_virtgpu_context_init ವಾದಗಳು_2 = {
        .ಅಂಕೆ_params     = ARRAY_ಗಾತ್ರ(ctx_ಗಣ_params),
        .pad            = 0,
        .ctx_ಗಣ_params = (uintptr_t) &ctx_ಗಣ_params,
    };

    ಹಿಂತಿರುಗಿ virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &ವಾದಗಳು_2);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ virtgpu_ioctl_get_caps(virtgpu *             gpu,
                                  virgl_renderer_capset id,
                                  uint32_t              ಆವೃತ್ತಿ_2,
                                  ಶೂನ್ಯ *                capset,
                                  size_t                capset_ಗಾತ್ರ) {
    drm_virtgpu_get_caps ವಾದಗಳು_2 = {
        .cap_ಗಣ_id  = id,
        .cap_ಗಣ_ver = ಆವೃತ್ತಿ_2,
        .addr        = (uintptr_t) capset,
        .size        = (__u32) capset_ಗಾತ್ರ,
        .pad         = 0,
    };

    ಹಿಂತಿರುಗಿ virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GET_CAPS, &ವಾದಗಳು_2);
}

ಸ್ಥಿರ uint64_t virtgpu_ioctl_getparam(virtgpu * gpu, uint64_t param) {
    /* val must be zeroed because kernel only writes the lower 32 bits */
    uint64_t             val  = 0;
    drm_virtgpu_getparam ವಾದಗಳು_2 = {
        .param = param,
        .value = (uintptr_t) &val,
    };

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ret = virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GETPARAM, &ವಾದಗಳು_2);
    ಹಿಂತಿರುಗಿ ret ? 0 : val;
}

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags) {
    /*
     * Prepare the command encoder and its buffer
     */

    thread_local ಅಕ್ಷರ encoder_ಬಫರ್[4096];

    thread_local apir_encoder enc;
    enc = {
        .cur   = encoder_ಬಫರ್,
        .start = encoder_ಬಫರ್,
        .end   = encoder_ಬಫರ್ + ಗಾತ್ರ(encoder_ಬಫರ್),
        .fatal = ಸುಳ್ಳು,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    ಆದರೆ (!gpu->use_apir_capset) {
        cmd_type += VENUS_COMMAND_TYPE_ಉದ್ದಳತೆ;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t ಉತ್ತರ_res_id = gpu->ಉತ್ತರ_shmem.res_id;
    apir_encode_uint32_t(&enc, &ಉತ್ತರ_res_id);

    ಹಿಂತಿರುಗಿ &enc;
}

ಶೂನ್ಯ remote_call_ಮುಕ್ತಾಯ(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec) {
    ಬಳಸದ(gpu);

    ಆದರೆ (!enc) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) encoder", __func__);
    }

    ಆದರೆ (!dec) {
        GGML_ABORT(GGML_VIRTGPU "%s: Invalid (null) decoder", __func__);
    }

    ಆದರೆ (apir_encoder_get_fatal(enc)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to encode the output parameters.", __func__);
    }

    ಆದರೆ (apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Failed to decode the input parameters.", __func__);
    }
}

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  encoder,
                     apir_decoder ** decoder,
                     ದಶಮಾಂಶ           max_wait_ms,
                     ದೀರ್ಘ ದೀರ್ಘ *     call_duration_ns) {
    /*
     * Prepare the reply notification pointer
     */

    ಅಸ್ಥಿರ std::atomic_uint * atomic_ಉತ್ತರ_notif = (ಅಸ್ಥಿರ std::atomic_uint *) gpu->ಉತ್ತರ_shmem.mmap_ptr;
    *atomic_ಉತ್ತರ_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    drm_virtgpu_execbuffer ವಾದಗಳು_2 = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .ಅಂಕೆ_bo_handles = 0,

        .fence_fd         = 0,
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .ಅಂಕೆ_ಒಳ_syncobjs  = 0,
        .ಅಂಕೆ_ಹೊರಹೋಗು_syncobjs = 0,
        .ಒಳ_syncobjs      = 0,
        .ಹೊರಹೋಗು_syncobjs     = 0,
    };

    *decoder = ಖಾಲಿ;

    ಪೂರ್ಣಾಂಕ ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &ವಾದಗಳು_2);

    ಆದರೆ (ret != 0) {
        GGML_ABORT(GGML_VIRTGPU "%s: the virtgpu EXECBUFFER ioctl failed (%d)", __func__, ret);
    }

    /*
     * Wait for the response notification
     */
    timer_ದತ್ತಾಂಶ wait_host_ಉತ್ತರ_timer = { 0, 0, 0 };

    ಆರಂಭ_timer(&wait_host_ಉತ್ತರ_timer);

    timespec ts_ಆರಂಭ, ts_ಕೊನೆ;
    clock_gettime(CLOCK_MONOTONIC, &ts_ಆರಂಭ);
    ದೀರ್ಘ ದೀರ್ಘ ಆರಂಭ_ಸಮಯ = (ದೀರ್ಘ ದೀರ್ಘ) ts_ಆರಂಭ.tv_sec * 1000000000LL + ts_ಆರಂಭ.tv_nsec;

    ತಾರ್ಕಿಕ     timedout    = ಸುಳ್ಳು;
    uint32_t notif_ಮೌಲ್ಯ = 0;
    ಆಗ (ನಿಜ) {
        notif_ಮೌಲ್ಯ = std::atomic_load_explicit(atomic_ಉತ್ತರ_notif, std::memory_order_acquire);

        ಆದರೆ (notif_ಮೌಲ್ಯ != 0) {
            ನಿಲ್ಲಿಸು;
        }

        int64_t ತಳ_sleep_us = 15;

        os_ಸಮಯ_sleep(ತಳ_sleep_us);

        ಆದರೆ (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_ಕೊನೆ);
            ದೀರ್ಘ ದೀರ್ಘ ಕೊನೆ_ಸಮಯ    = (ದೀರ್ಘ ದೀರ್ಘ) ts_ಕೊನೆ.tv_sec * 1000000000LL + ts_ಕೊನೆ.tv_nsec;
            ದಶಮಾಂಶ     duration_ms = (ಕೊನೆ_ಸಮಯ - ಆರಂಭ_ಸಮಯ) / 1000000;

            ಆದರೆ (duration_ms > max_wait_ms) {
                timedout = ನಿಜ;
                ನಿಲ್ಲಿಸು;
            }
        }
    }

    ಆದರೆ (call_duration_ns) {
        *call_duration_ns = stop_timer(&wait_host_ಉತ್ತರ_timer);
    }

    ಆದರೆ (max_wait_ms && timedout) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: timed out waiting for the host answer...\n", __func__);
        ಹಿಂತಿರುಗಿ APIR_FORWARD_TIMEOUT;
    }

    /*
     * Prepare the decoder
     */
    ಸ್ಥಿರ apir_decoder response_dec;
    response_dec.cur = (ಅಕ್ಷರ *) gpu->ಉತ್ತರ_shmem.mmap_ptr + ಗಾತ್ರ(*atomic_ಉತ್ತರ_notif);
    response_dec.end = (ಅಕ್ಷರ *) gpu->ಉತ್ತರ_shmem.mmap_ptr + gpu->ಉತ್ತರ_shmem.mmap_ಗಾತ್ರ;
    *decoder         = &response_dec;

    // extract the actual return value from the notif flag
    uint32_t returned_ಮೌಲ್ಯ = notif_ಮೌಲ್ಯ - 1;
    ಹಿಂತಿರುಗಿ returned_ಮೌಲ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ log_call_duration(ದೀರ್ಘ ದೀರ್ಘ call_duration_ns, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ದ್ವಿಗುಣ call_duration_ms = (ದ್ವಿಗುಣ) call_duration_ns / 1e6;  // 1 millisecond = 1e6 nanoseconds
    ದ್ವಿಗುಣ call_duration_s  = (ದ್ವಿಗುಣ) call_duration_ns / 1e9;  // 1 second = 1e9 nanoseconds

    ಆದರೆ (call_duration_s > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fs for the %s host reply...\n", call_duration_s, name);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (call_duration_ms > 1) {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %.2fms for the %s host reply...\n", call_duration_ms, name);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_INFO(GGML_VIRTGPU "waited %lldns for the %s host reply...\n", call_duration_ns, name);
    }
}
