#pragma once

// clang-format off
#અન્તર્ભાવ "virtgpu-utils.h"
#અન્તર્ભાવ "virtgpu-shm.h"
#અન્તર્ભાવ "virtgpu-apir.h"

#અન્તર્ભાવ "backend/shared/api_remoting.h"
#અન્તર્ભાવ "backend/shared/apir_cs.h"

#અન્તર્ભાવ <fcntl.h>
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <sys/stat.h>
#અન્તર્ભાવ <sys/sysmacros.h>
#અન્તર્ભાવ <threads.h>
#અન્તર્ભાવ <xf86drm.h>

#અન્તર્ભાવ <cstring>

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#અન્તર્ભાવ "apir_hw.h"
#અન્તર્ભાવ <drm/virtgpu_drm.h>
#અન્તર્ભાવ "venus_hw.h"
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

પ્રકારવ્યાખ્યા uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID       0x1af4
#define VIRTGPU_PCI_DEVICE_ID       0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM    9

#define SHMEM_DATA_SIZE  0x1830000  // 24MiB
#define SHMEM_REPLY_SIZE 0x4000

#define ARRAY_SIZE(x) (કદ(x) / કદ((x)[0]))

ગણના virt_gpu_result_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#define PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

સંરચના virtgpu {
    bool use_apir_capset;

    પૂર્ણાંક fd;

    સંરચના {
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
    સંરચના {
        અક્ષર *   description;
        અક્ષર *   name;
        int32_t  device_ગણતરી;
        uint32_t type;
        size_t   memory_free;
        size_t   memory_total;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    સંરચના {
        apir_buffer_type_host_handle_t host_handle;
        અક્ષર *                         name;
        size_t                         alignment;
        size_t                         max_size;
    } cached_buffer_type;
};

સ્થિર અંતઃરેખ પૂર્ણાંક virtgpu_ioctl(virtgpu * gpu, unsigned long request, શૂન્ય * args) {
    પાછા drmIoctl(gpu->fd, request, args);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     દશાંશ           max_wait_ms,
                     long long *     call_duration_ns);

શૂન્ય remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
