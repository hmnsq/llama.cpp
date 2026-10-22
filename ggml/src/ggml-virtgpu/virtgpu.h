#هدايت once

// clang-format off
#ضمن "virtgpu-utils.h"
#ضمن "virtgpu-shm.h"
#ضمن "virtgpu-apir.h"

#ضمن "backend/shared/api_remoting.h"
#ضمن "backend/shared/apir_cs.h"

#ضمن <fcntl.h>
#ضمن <stdbool.h>
#ضمن <stdio.h>
#ضمن <sys/stat.h>
#ضمن <sys/sysmacros.h>
#ضمن <threads.h>
#ضمن <xf86drm.h>

#ضمن <cstring>

#وصف_ڪريو VIRGL_RENDERER_UNSTABLE_APIS 1
#ضمن "apir_hw.h"
#ضمن <drm/virtgpu_drm.h>
#ضمن "venus_hw.h"
// clang-format on

#جيڪڏهن_وصف_نه_ٿيل VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#جيڪڏهن_پورو

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#وصف_ڪريو VENUS_COMMAND_TYPE_ڊيگهائي 331

#جيڪڏهن_وصف_نه_ٿيل VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#جيڪڏهن_پورو

تعريفنوع uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#وصف_ڪريو VIRTGPU_PCI_VENDOR_ID       0x1af4
#وصف_ڪريو VIRTGPU_PCI_DEVICE_ID       0x1050
#وصف_ڪريو VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#وصف_ڪريو VIRTGPU_PARAM_GUEST_VRAM    9

#وصف_ڪريو SHMEM_معلومات_سائيز  0x1830000  // 24MiB
#وصف_ڪريو SHMEM_جواب_سائيز 0x4000

#وصف_ڪريو ARRAY_سائيز(x) (حجم(x) / حجم((x)[0]))

تعداد virt_gpu_نتيجو_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#وصف_ڪريو PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

بنية virtgpu {
    bool use_apir_capset;

    صحيح fd;

    بنية {
        virgl_renderer_capset      id;
        uint32_t                   نسخو_2;
        virgl_renderer_capset_apir معلومات;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem جواب_shmem;
    virtgpu_shmem معلومات_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t معلومات_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    بنية {
        حرف *   description;
        حرف *   name;
        int32_t  device_ڳڻپ;
        uint32_t type;
        size_t   يادگيري_free;
        size_t   يادگيري_ڪل;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    بنية {
        apir_بفر_type_host_handle_t host_handle;
        حرف *                         name;
        size_t                         alignment;
        size_t                         max_سائيز;
    } cached_بفر_type;
};

ساكن مضمن صحيح virtgpu_ioctl(virtgpu * gpu, بنانشان ڊگهو request, فراغ * دليلون_2) {
    أرجع drmIoctl(gpu->fd, request, دليلون_2);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     عائم           max_wait_ms,
                     ڊگهو ڊگهو *     call_duration_ns);

فراغ remote_call_پڄاڻي(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
