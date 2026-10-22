#निर्देशः once

// clang-format off
#अन्तर्भाव "virtgpu-utils.h"
#अन्तर्भाव "virtgpu-shm.h"
#अन्तर्भाव "virtgpu-apir.h"

#अन्तर्भाव "backend/shared/api_remoting.h"
#अन्तर्भाव "backend/shared/apir_cs.h"

#अन्तर्भाव <fcntl.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <sys/stat.h>
#अन्तर्भाव <sys/sysmacros.h>
#अन्तर्भाव <threads.h>
#अन्तर्भाव <xf86drm.h>

#अन्तर्भाव <cstring>

#परिभाषय VIRGL_RENDERER_UNSTABLE_APIS 1
#अन्तर्भाव "apir_hw.h"
#अन्तर्भाव <drm/virtgpu_drm.h>
#अन्तर्भाव "venus_hw.h"
// clang-format on

#यदि_अपरिभाषितम् VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#यदि_अन्तः

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#परिभाषय VENUS_COMMAND_TYPE_दीर्घता 331

#यदि_अपरिभाषितम् VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#यदि_अन्तः

प्रकारनिर्देश uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#परिभाषय VIRTGPU_PCI_VENDOR_ID       0x1af4
#परिभाषय VIRTGPU_PCI_DEVICE_ID       0x1050
#परिभाषय VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#परिभाषय VIRTGPU_PARAM_GUEST_VRAM    9

#परिभाषय SHMEM_दत्त_आकार  0x1830000  // 24MiB
#परिभाषय SHMEM_प्रत्युत्तरम्_आकार 0x4000

#परिभाषय ARRAY_आकार(x) (परिमाण(x) / परिमाण((x)[0]))

गणना virt_gpu_परिणाम_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#परिभाषय PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

संरचना virtgpu {
    bool use_apir_capset;

    पूर्णाङ्क fd;

    संरचना {
        virgl_renderer_capset      id;
        uint32_t                   संस्करणम्_2;
        virgl_renderer_capset_apir दत्त;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem प्रत्युत्तरम्_shmem;
    virtgpu_shmem दत्त_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t दत्त_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    संरचना {
        वर्ण *   description;
        वर्ण *   name;
        int32_t  device_गणनम्;
        uint32_t type;
        size_t   स्मृतिः_free;
        size_t   स्मृतिः_योग;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    संरचना {
        apir_बफर_type_host_handle_t host_handle;
        वर्ण *                         name;
        size_t                         alignment;
        size_t                         max_आकार;
    } cached_बफर_type;
};

स्थिर अन्तःस्थ पूर्णाङ्क virtgpu_ioctl(virtgpu * gpu, अचिह्नित दीर्घ request, शून्य * तर्काः_2) {
    प्रत्यागम drmIoctl(gpu->fd, request, तर्काः_2);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     भिन्न           max_wait_ms,
                     दीर्घ दीर्घ *     call_duration_ns);

शून्य remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
