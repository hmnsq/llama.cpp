#ಸೂಚನೆ once

// clang-format off
#ಅನ್ತರ್ಭಾವ "virtgpu-utils.h"
#ಅನ್ತರ್ಭಾವ "virtgpu-shm.h"
#ಅನ್ತರ್ಭಾವ "virtgpu-apir.h"

#ಅನ್ತರ್ಭಾವ "backend/shared/api_remoting.h"
#ಅನ್ತರ್ಭಾವ "backend/shared/apir_cs.h"

#ಅನ್ತರ್ಭಾವ <fcntl.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
#ಅನ್ತರ್ಭಾವ <sys/sysmacros.h>
#ಅನ್ತರ್ಭಾವ <threads.h>
#ಅನ್ತರ್ಭಾವ <xf86drm.h>

#ಅನ್ತರ್ಭಾವ <cstring>

#ವ್ಯಾಖ್ಯಾನಿಸು VIRGL_RENDERER_UNSTABLE_APIS 1
#ಅನ್ತರ್ಭಾವ "apir_hw.h"
#ಅನ್ತರ್ಭಾವ <drm/virtgpu_drm.h>
#ಅನ್ತರ್ಭಾವ "venus_hw.h"
// clang-format on

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#ಆದರೆ_ಅಂತ್ಯ

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#ವ್ಯಾಖ್ಯಾನಿಸು VENUS_COMMAND_TYPE_ಉದ್ದಳತೆ 331

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#ವ್ಯಾಖ್ಯಾನಿಸು VIRTGPU_PCI_VENDOR_ID       0x1af4
#ವ್ಯಾಖ್ಯಾನಿಸು VIRTGPU_PCI_DEVICE_ID       0x1050
#ವ್ಯಾಖ್ಯಾನಿಸು VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#ವ್ಯಾಖ್ಯಾನಿಸು VIRTGPU_PARAM_GUEST_VRAM    9

#ವ್ಯಾಖ್ಯಾನಿಸು SHMEM_ದತ್ತಾಂಶ_ಗಾತ್ರ  0x1830000  // 24MiB
#ವ್ಯಾಖ್ಯಾನಿಸು SHMEM_ಉತ್ತರ_ಗಾತ್ರ 0x4000

#ವ್ಯಾಖ್ಯಾನಿಸು ARRAY_ಗಾತ್ರ(x) (ಗಾತ್ರ(x) / ಗಾತ್ರ((x)[0]))

ಎಣಿಕೆ virt_gpu_ಫಲಿತಾಂಶ_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#ವ್ಯಾಖ್ಯಾನಿಸು PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

ರಚನೆ virtgpu {
    bool use_apir_capset;

    ಪೂರ್ಣಾಂಕ fd;

    ರಚನೆ {
        virgl_renderer_capset      id;
        uint32_t                   ಆವೃತ್ತಿ_2;
        virgl_renderer_capset_apir ದತ್ತಾಂಶ;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem ಉತ್ತರ_shmem;
    virtgpu_shmem ದತ್ತಾಂಶ_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t ದತ್ತಾಂಶ_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    ರಚನೆ {
        ಅಕ್ಷರ *   description;
        ಅಕ್ಷರ *   name;
        int32_t  device_ಎಣಿಕೆ;
        uint32_t type;
        size_t   ಸ್ಮರಣೆ_free;
        size_t   ಸ್ಮರಣೆ_ಒಟ್ಟು;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    ರಚನೆ {
        apir_ಬಫರ್_type_host_handle_t host_handle;
        ಅಕ್ಷರ *                         name;
        size_t                         alignment;
        size_t                         max_ಗಾತ್ರ;
    } cached_ಬಫರ್_type;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ virtgpu_ioctl(virtgpu * gpu, ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ request, ಶೂನ್ಯ * ವಾದಗಳು_2) {
    ಹಿಂತಿರುಗಿ drmIoctl(gpu->fd, request, ವಾದಗಳು_2);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     ದಶಮಾಂಶ           max_wait_ms,
                     ದೀರ್ಘ ದೀರ್ಘ *     call_duration_ns);

ಶೂನ್ಯ remote_call_ಮುಕ್ತಾಯ(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
