#رهنمود once

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

#تعریف_کن VIRGL_RENDERER_UNSTABLE_APIS 1
#ضمن "apir_hw.h"
#ضمن <drm/virtgpu_drm.h>
#ضمن "venus_hw.h"
// clang-format on

#اگر_تعریف_نشده VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#پایان_اگر

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#تعریف_کن VENUS_COMMAND_TYPE_درازی 331

#اگر_تعریف_نشده VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#پایان_اگر

تعريفنوع uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#تعریف_کن VIRTGPU_PCI_VENDOR_ID       0x1af4
#تعریف_کن VIRTGPU_PCI_DEVICE_ID       0x1050
#تعریف_کن VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#تعریف_کن VIRTGPU_PARAM_GUEST_VRAM    9

#تعریف_کن SHMEM_داده_اندازه  0x1830000  // 24MiB
#تعریف_کن SHMEM_پاسخ_اندازه 0x4000

#تعریف_کن ARRAY_اندازه(x) (حجم(x) / حجم((x)[0]))

تعداد virt_gpu_نتیجه_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#تعریف_کن PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

بنية virtgpu {
    bool use_apir_capset;

    صحيح fd;

    بنية {
        virgl_renderer_capset      id;
        uint32_t                   نسخه_2;
        virgl_renderer_capset_apir داده;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem پاسخ_shmem;
    virtgpu_shmem داده_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t داده_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    بنية {
        حرف *   description;
        حرف *   name;
        int32_t  device_شمار;
        uint32_t type;
        size_t   حافظه_free;
        size_t   حافظه_مجموع;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    بنية {
        apir_میانگیر_type_host_handle_t host_handle;
        حرف *                         name;
        size_t                         alignment;
        size_t                         max_اندازه;
    } cached_میانگیر_type;
};

ساكن مضمن صحيح virtgpu_ioctl(virtgpu * gpu, بیعلامت دراز request, فراغ * آرگومانها_2) {
    أرجع drmIoctl(gpu->fd, request, آرگومانها_2);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     عائم           max_wait_ms,
                     دراز دراز *     call_duration_ns);

فراغ remote_call_پایان(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
