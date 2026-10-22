#pragma once

// clang-format off
#অন্তর্ভাৱ "virtgpu-utils.h"
#অন্তর্ভাৱ "virtgpu-shm.h"
#অন্তর্ভাৱ "virtgpu-apir.h"

#অন্তর্ভাৱ "backend/shared/api_remoting.h"
#অন্তর্ভাৱ "backend/shared/apir_cs.h"

#অন্তর্ভাৱ <fcntl.h>
#অন্তর্ভাৱ <stdbool.h>
#অন্তর্ভাৱ <stdio.h>
#অন্তর্ভাৱ <sys/stat.h>
#অন্তর্ভাৱ <sys/sysmacros.h>
#অন্তর্ভাৱ <threads.h>
#অন্তর্ভাৱ <xf86drm.h>

#অন্তর্ভাৱ <cstring>

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#অন্তর্ভাৱ "apir_hw.h"
#অন্তর্ভাৱ <drm/virtgpu_drm.h>
#অন্তর্ভাৱ "venus_hw.h"
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

প্রকারনির্দেশ uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID       0x1af4
#define VIRTGPU_PCI_DEVICE_ID       0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM    9

#define SHMEM_DATA_SIZE  0x1830000  // 24MiB
#define SHMEM_REPLY_SIZE 0x4000

#define ARRAY_SIZE(x) (পরিমাণ(x) / পরিমাণ((x)[0]))

গণনা virt_gpu_result_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#define PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

সংরচনা virtgpu {
    bool use_apir_capset;

    পূর্ণাঙ্ক fd;

    সংরচনা {
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
    সংরচনা {
        ৱর্ণ *   description;
        ৱর্ণ *   name;
        int32_t  device_গুনতি;
        uint32_t type;
        size_t   memory_free;
        size_t   memory_total;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    সংরচনা {
        apir_buffer_type_host_handle_t host_handle;
        ৱর্ণ *                         name;
        size_t                         alignment;
        size_t                         max_size;
    } cached_buffer_type;
};

স্থির অন্তঃস্থ পূর্ণাঙ্ক virtgpu_ioctl(virtgpu * gpu, unsigned long request, শূন্য * args) {
    ঘূৰাই drmIoctl(gpu->fd, request, args);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     ভিন্ন           max_wait_ms,
                     long long *     call_duration_ns);

শূন্য remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
