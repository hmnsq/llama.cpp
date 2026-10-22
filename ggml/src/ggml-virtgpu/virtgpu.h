#pragma once

// clang-format off
#ଅନ୍ତର୍ଭାଵ "virtgpu-utils.h"
#ଅନ୍ତର୍ଭାଵ "virtgpu-shm.h"
#ଅନ୍ତର୍ଭାଵ "virtgpu-apir.h"

#ଅନ୍ତର୍ଭାଵ "backend/shared/api_remoting.h"
#ଅନ୍ତର୍ଭାଵ "backend/shared/apir_cs.h"

#ଅନ୍ତର୍ଭାଵ <fcntl.h>
#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <sys/stat.h>
#ଅନ୍ତର୍ଭାଵ <sys/sysmacros.h>
#ଅନ୍ତର୍ଭାଵ <threads.h>
#ଅନ୍ତର୍ଭାଵ <xf86drm.h>

#ଅନ୍ତର୍ଭାଵ <cstring>

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#ଅନ୍ତର୍ଭାଵ "apir_hw.h"
#ଅନ୍ତର୍ଭାଵ <drm/virtgpu_drm.h>
#ଅନ୍ତର୍ଭାଵ "venus_hw.h"
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

ପ୍ରକାରବ୍ୟାଖ୍ୟା uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID       0x1af4
#define VIRTGPU_PCI_DEVICE_ID       0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM    9

#define SHMEM_DATA_SIZE  0x1830000  // 24MiB
#define SHMEM_REPLY_SIZE 0x4000

#define ARRAY_SIZE(x) (ଆକାର(x) / ଆକାର((x)[0]))

ଗଣନା virt_gpu_result_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#define PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

ସଂରଚନା virtgpu {
    bool use_apir_capset;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd;

    ସଂରଚନା {
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
    ସଂରଚନା {
        ଅକ୍ଷର *   description;
        ଅକ୍ଷର *   name;
        int32_t  device_ଗଣନା;
        uint32_t type;
        size_t   memory_free;
        size_t   memory_total;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    ସଂରଚନା {
        apir_buffer_type_host_handle_t host_handle;
        ଅକ୍ଷର *                         name;
        size_t                         alignment;
        size_t                         max_size;
    } cached_buffer_type;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା virtgpu_ioctl(virtgpu * gpu, unsigned long request, ଶୂନ୍ୟ * args) {
    ଫେରନ୍ତୁ drmIoctl(gpu->fd, request, args);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     ଦଶମିକ           max_wait_ms,
                     long long *     call_duration_ns);

ଶୂନ୍ୟ remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
