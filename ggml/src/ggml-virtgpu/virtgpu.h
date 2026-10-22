#ਹਦਾਇਤ once

// clang-format off
#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-shm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-apir.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "backend/shared/api_remoting.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "backend/shared/apir_cs.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <fcntl.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/stat.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/sysmacros.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <threads.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <xf86drm.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VIRGL_RENDERER_UNSTABLE_APIS 1
#ਅਨ੍ਤਰ੍ਭਾਵ "apir_hw.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <drm/virtgpu_drm.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "venus_hw.h"
// clang-format on

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#ਜੇ_ਅੰਤ

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VENUS_COMMAND_TYPE_ਲੰਮਾਈ 331

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VIRTGPU_PCI_VENDOR_ID       0x1af4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VIRTGPU_PCI_DEVICE_ID       0x1050
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VIRTGPU_PARAM_GUEST_VRAM    9

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SHMEM_ਅੰਕੜੇ_ਮਾਪ  0x1830000  // 24MiB
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SHMEM_ਜਵਾਬ_ਮਾਪ 0x4000

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ARRAY_ਮਾਪ(x) (ਆਕਾਰ(x) / ਆਕਾਰ((x)[0]))

ਗਿਣਤੀ virt_gpu_ਨਤੀਜਾ_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

ਬਣਤਰ virtgpu {
    bool use_apir_capset;

    ਪੂਰਨ fd;

    ਬਣਤਰ {
        virgl_renderer_capset      id;
        uint32_t                   ਸੰਸਕਰਣ_2;
        virgl_renderer_capset_apir ਅੰਕੜੇ;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem ਜਵਾਬ_shmem;
    virtgpu_shmem ਅੰਕੜੇ_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t ਅੰਕੜੇ_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    ਬਣਤਰ {
        ਅੱਖਰ *   description;
        ਅੱਖਰ *   name;
        int32_t  device_ਗਿਣਤੀ;
        uint32_t type;
        size_t   ਸਮ੍ਰਿਤੀ_free;
        size_t   ਸਮ੍ਰਿਤੀ_ਕੁੱਲ;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    ਬਣਤਰ {
        apir_ਬਫਰ_type_host_handle_t host_handle;
        ਅੱਖਰ *                         name;
        size_t                         alignment;
        size_t                         max_ਮਾਪ;
    } cached_ਬਫਰ_type;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ virtgpu_ioctl(virtgpu * gpu, ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ request, ਖਾਲੀ * ਦਲੀਲਾਂ_2) {
    ਵਾਪਸ drmIoctl(gpu->fd, request, ਦਲੀਲਾਂ_2);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     ਦਸ਼ਮਲਵ           max_wait_ms,
                     ਲੰਮਾ ਲੰਮਾ *     call_duration_ns);

ਖਾਲੀ remote_call_ਅੰਤ(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
