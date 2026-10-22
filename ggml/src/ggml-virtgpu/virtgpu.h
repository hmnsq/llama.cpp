#pragma once

// clang-format off
#அந்தர்பாவ "virtgpu-utils.h"
#அந்தர்பாவ "virtgpu-shm.h"
#அந்தர்பாவ "virtgpu-apir.h"

#அந்தர்பாவ "backend/shared/api_remoting.h"
#அந்தர்பாவ "backend/shared/apir_cs.h"

#அந்தர்பாவ <fcntl.h>
#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <sys/stat.h>
#அந்தர்பாவ <sys/sysmacros.h>
#அந்தர்பாவ <threads.h>
#அந்தர்பாவ <xf86drm.h>

#அந்தர்பாவ <cstring>

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#அந்தர்பாவ "apir_hw.h"
#அந்தர்பாவ <drm/virtgpu_drm.h>
#அந்தர்பாவ "venus_hw.h"
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

வகைவரை uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID       0x1af4
#define VIRTGPU_PCI_DEVICE_ID       0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM    9

#define SHMEM_DATA_SIZE  0x1830000  // 24MiB
#define SHMEM_REPLY_SIZE 0x4000

#define ARRAY_SIZE(x) (அளவு(x) / அளவு((x)[0]))

எண்ணிக்கை virt_gpu_result_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#define PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

கட்டமைப்பு virtgpu {
    bool use_apir_capset;

    முழு fd;

    கட்டமைப்பு {
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
    கட்டமைப்பு {
        எழுத்து *   description;
        எழுத்து *   name;
        int32_t  device_மொத்தம்;
        uint32_t type;
        size_t   memory_free;
        size_t   memory_total;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    கட்டமைப்பு {
        apir_buffer_type_host_handle_t host_handle;
        எழுத்து *                         name;
        size_t                         alignment;
        size_t                         max_size;
    } cached_buffer_type;
};

நிலையான உள்வரி முழு virtgpu_ioctl(virtgpu * gpu, unsigned long request, வெறுமை * args) {
    திரும்பு drmIoctl(gpu->fd, request, args);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     தசம           max_wait_ms,
                     long long *     call_duration_ns);

வெறுமை remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
