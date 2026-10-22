#pragma once

// clang-format off
#അന്തര്ഭാവ "virtgpu-utils.h"
#അന്തര്ഭാവ "virtgpu-shm.h"
#അന്തര്ഭാവ "virtgpu-apir.h"

#അന്തര്ഭാവ "backend/shared/api_remoting.h"
#അന്തര്ഭാവ "backend/shared/apir_cs.h"

#അന്തര്ഭാവ <fcntl.h>
#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <sys/stat.h>
#അന്തര്ഭാവ <sys/sysmacros.h>
#അന്തര്ഭാവ <threads.h>
#അന്തര്ഭാവ <xf86drm.h>

#അന്തര്ഭാവ <cstring>

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#അന്തര്ഭാവ "apir_hw.h"
#അന്തര്ഭാവ <drm/virtgpu_drm.h>
#അന്തര്ഭാവ "venus_hw.h"
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

തരനിർവചനം uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID       0x1af4
#define VIRTGPU_PCI_DEVICE_ID       0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM    9

#define SHMEM_DATA_SIZE  0x1830000  // 24MiB
#define SHMEM_REPLY_SIZE 0x4000

#define ARRAY_SIZE(x) (വലുപ്പം(x) / വലുപ്പം((x)[0]))

എണ്ണൽ virt_gpu_result_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#define PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

ഘടന virtgpu {
    bool use_apir_capset;

    പൂർണ്ണസംഖ്യ fd;

    ഘടന {
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
    ഘടന {
        അക്ഷരം *   description;
        അക്ഷരം *   name;
        int32_t  device_എണ്ണം;
        uint32_t type;
        size_t   memory_free;
        size_t   memory_total;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    ഘടന {
        apir_buffer_type_host_handle_t host_handle;
        അക്ഷരം *                         name;
        size_t                         alignment;
        size_t                         max_size;
    } cached_buffer_type;
};

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ virtgpu_ioctl(virtgpu * gpu, unsigned long request, ശൂന്യം * args) {
    തിരികെ drmIoctl(gpu->fd, request, args);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     ദശാംശം           max_wait_ms,
                     long long *     call_duration_ns);

ശൂന്യം remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
