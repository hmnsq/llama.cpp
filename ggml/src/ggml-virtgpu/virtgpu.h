#pragma once

// clang-format off
#অন্তর্ভাব "virtgpu-utils.h"
#অন্তর্ভাব "virtgpu-shm.h"
#অন্তর্ভাব "virtgpu-apir.h"

#অন্তর্ভাব "backend/shared/api_remoting.h"
#অন্তর্ভাব "backend/shared/apir_cs.h"

#অন্তর্ভাব <fcntl.h>
#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <sys/stat.h>
#অন্তর্ভাব <sys/sysmacros.h>
#অন্তর্ভাব <threads.h>
#অন্তর্ভাব <xf86drm.h>

#অন্তর্ভাব <cstring>

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#অন্তর্ভাব "apir_hw.h"
#অন্তর্ভাব <drm/virtgpu_drm.h>
#অন্তর্ভাব "venus_hw.h"
// clang-format on

#ifndef VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#endif

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#define VENUS_COMMAND_TYPE_দৈর্ঘ্য 331

#ifndef VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#endif

প্রকারলেখ uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID       0x1af4
#define VIRTGPU_PCI_DEVICE_ID       0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM    9

#define SHMEM_তথ্য_আকার  0x1830000  // 24MiB
#define SHMEM_REPLY_আকার 0x4000

#define ARRAY_আকার(x) (আকার(x) / আকার((x)[0]))

গণনা virt_gpu_ফলাফল_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#define PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

কাঠামো virtgpu {
    bool use_apir_capset;

    পূর্ণসংখ্যা fd;

    কাঠামো {
        virgl_renderer_capset      id;
        uint32_t                   version;
        virgl_renderer_capset_apir তথ্য;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem reply_shmem;
    virtgpu_shmem তথ্য_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t তথ্য_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    কাঠামো {
        অক্ষর *   description;
        অক্ষর *   name;
        int32_t  device_গুনতি;
        uint32_t type;
        size_t   memory_free;
        size_t   memory_যোগ;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    কাঠামো {
        apir_বাফার_type_host_handle_t host_handle;
        অক্ষর *                         name;
        size_t                         alignment;
        size_t                         max_আকার;
    } cached_বাফার_type;
};

স্থির অন্তঃরেখ পূর্ণসংখ্যা virtgpu_ioctl(virtgpu * gpu, unsigned long request, শূন্য * args) {
    ফেরত drmIoctl(gpu->fd, request, args);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     দশমিক           max_wait_ms,
                     long long *     call_duration_ns);

শূন্য remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
