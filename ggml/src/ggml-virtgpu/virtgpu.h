#pragma once

// clang-format off
#अन्तर्भाव "virtgpu-utils.h"
#अन्तर्भाव "virtgpu-shm.h"
#अन्तर्भाव "virtgpu-apir.h"

#अन्तर्भाव "backend/shared/api_remoting.h"
#अन्तर्भाव "backend/shared/apir_cs.h"

#अन्तर्भाव <fcntl.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <sys/stat.h>
#अन्तर्भाव <sys/sysmacros.h>
#अन्तर्भाव <threads.h>
#अन्तर्भाव <xf86drm.h>

#अन्तर्भाव <cstring>

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#अन्तर्भाव "apir_hw.h"
#अन्तर्भाव <drm/virtgpu_drm.h>
#अन्तर्भाव "venus_hw.h"
// clang-format on

#ifndef VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#endif

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#define VENUS_COMMAND_TYPE_LENGTH 331

#ifndef VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#endif

प्रकारव्याख्या uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID       0x1af4
#define VIRTGPU_PCI_DEVICE_ID       0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM    9

#define SHMEM_DATA_SIZE  0x1830000  // 24MiB
#define SHMEM_REPLY_SIZE 0x4000

#define ARRAY_SIZE(x) (आकार(x) / आकार((x)[0]))

गणना virt_gpu_result_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#define PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

संरचना virtgpu {
    bool use_apir_capset;

    पूर्णांक fd;

    संरचना {
        virgl_renderer_capset      id;
        uint32_t                   version;
        virgl_renderer_capset_apir data;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem reply_shmem;
    virtgpu_shmem data_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t data_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    संरचना {
        अक्षर *   description;
        अक्षर *   name;
        int32_t  device_मोजणी;
        uint32_t type;
        माप_प्रकार   memory_free;
        माप_प्रकार   memory_total;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    संरचना {
        apir_buffer_type_host_handle_t host_handle;
        अक्षर *                         name;
        माप_प्रकार                         alignment;
        माप_प्रकार                         max_size;
    } cached_buffer_type;
};

स्थिर ओळीत पूर्णांक virtgpu_ioctl(virtgpu * gpu, अचिन्हित दीर्घ request, शून्य * args) {
    परत drmIoctl(gpu->fd, request, args);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     दशांश           max_wait_ms,
                     दीर्घ दीर्घ *     call_duration_ns);

शून्य remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
