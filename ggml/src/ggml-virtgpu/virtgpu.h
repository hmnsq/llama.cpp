#لارښوونه once

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

#تعریف_کړه VIRGL_RENDERER_UNSTABLE_APIS 1
#ضمن "apir_hw.h"
#ضمن <drm/virtgpu_drm.h>
#ضمن "venus_hw.h"
// clang-format on

#که_نه_تعریف_شوی VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#د_که_پای

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#تعریف_کړه VENUS_COMMAND_TYPE_اوږدتیا 331

#که_نه_تعریف_شوی VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#د_که_پای

تعريفنوع uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#تعریف_کړه VIRTGPU_PCI_VENDOR_ID       0x1af4
#تعریف_کړه VIRTGPU_PCI_DEVICE_ID       0x1050
#تعریف_کړه VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#تعریف_کړه VIRTGPU_PARAM_GUEST_VRAM    9

#تعریف_کړه SHMEM_معلومات_اندازه  0x1830000  // 24MiB
#تعریف_کړه SHMEM_ځواب_اندازه 0x4000

#تعریف_کړه ARRAY_اندازه(x) (حجم(x) / حجم((x)[0]))

تعداد virt_gpu_پایله_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#تعریف_کړه PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

بنية virtgpu {
    bool use_apir_capset;

    صحيح fd;

    بنية {
        virgl_renderer_capset      id;
        uint32_t                   نسخه_2;
        virgl_renderer_capset_apir معلومات;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem ځواب_shmem;
    virtgpu_shmem معلومات_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t معلومات_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    بنية {
        حرف *   description;
        حرف *   name;
        int32_t  device_شمېر;
        uint32_t type;
        size_t   حافظه_free;
        size_t   حافظه_ټول;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    بنية {
        apir_بفر_type_host_handle_t host_handle;
        حرف *                         name;
        size_t                         alignment;
        size_t                         max_اندازه;
    } cached_بفر_type;
};

ساكن مضمن صحيح virtgpu_ioctl(virtgpu * gpu, بینښه اوږد request, فراغ * دلیلونه_2) {
    أرجع drmIoctl(gpu->fd, request, دلیلونه_2);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     عائم           max_wait_ms,
                     اوږد اوږد *     call_duration_ns);

فراغ remote_call_پای(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
