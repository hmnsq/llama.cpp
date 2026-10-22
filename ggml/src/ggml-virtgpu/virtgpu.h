#ہدایت once

// clang-format off
#شامل "virtgpu-utils.h"
#شامل "virtgpu-shm.h"
#شامل "virtgpu-apir.h"

#شامل "backend/shared/api_remoting.h"
#شامل "backend/shared/apir_cs.h"

#شامل <fcntl.h>
#شامل <stdbool.h>
#شامل <stdio.h>
#شامل <sys/stat.h>
#شامل <sys/sysmacros.h>
#شامل <threads.h>
#شامل <xf86drm.h>

#شامل <cstring>

#تعریف_کرو VIRGL_RENDERER_UNSTABLE_APIS 1
#شامل "apir_hw.h"
#شامل <drm/virtgpu_drm.h>
#شامل "venus_hw.h"
// clang-format on

#اگر_غیر_تعریف_شدہ VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#اگر_ختم

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#تعریف_کرو VENUS_COMMAND_TYPE_لمبائی 331

#اگر_غیر_تعریف_شدہ VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#اگر_ختم

قسمتعریف uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#تعریف_کرو VIRTGPU_PCI_VENDOR_ID       0x1af4
#تعریف_کرو VIRTGPU_PCI_DEVICE_ID       0x1050
#تعریف_کرو VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#تعریف_کرو VIRTGPU_PARAM_GUEST_VRAM    9

#تعریف_کرو SHMEM_ڈیٹا_سائز  0x1830000  // 24MiB
#تعریف_کرو SHMEM_جواب_سائز 0x4000

#تعریف_کرو ARRAY_سائز(x) (حجم(x) / حجم((x)[0]))

شمار virt_gpu_نتیجہ_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#تعریف_کرو PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

ساخت virtgpu {
    bool use_apir_capset;

    صحیح fd;

    ساخت {
        virgl_renderer_capset      id;
        uint32_t                   نسخہ_2;
        virgl_renderer_capset_apir ڈیٹا;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem جواب_shmem;
    virtgpu_shmem ڈیٹا_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t ڈیٹا_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    ساخت {
        حرف *   description;
        حرف *   name;
        int32_t  device_گنتی;
        uint32_t type;
        size_t   حافظہ_free;
        size_t   حافظہ_کل;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    ساخت {
        apir_بفر_type_host_handle_t host_handle;
        حرف *                         name;
        size_t                         alignment;
        size_t                         max_سائز;
    } cached_بفر_type;
};

ساکن اندرونی صحیح virtgpu_ioctl(virtgpu * gpu, غیرعلامتی طویل request, خالی * دلائل_2) {
    واپس drmIoctl(gpu->fd, request, دلائل_2);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     اعشاری           max_wait_ms,
                     طویل طویل *     call_duration_ns);

خالی remote_call_اختتام(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
