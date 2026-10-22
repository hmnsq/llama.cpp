#సూచన once

// clang-format off
#అన్తర్భావ "virtgpu-utils.h"
#అన్తర్భావ "virtgpu-shm.h"
#అన్తర్భావ "virtgpu-apir.h"

#అన్తర్భావ "backend/shared/api_remoting.h"
#అన్తర్భావ "backend/shared/apir_cs.h"

#అన్తర్భావ <fcntl.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <sys/stat.h>
#అన్తర్భావ <sys/sysmacros.h>
#అన్తర్భావ <threads.h>
#అన్తర్భావ <xf86drm.h>

#అన్తర్భావ <cstring>

#నిర్వచించు VIRGL_RENDERER_UNSTABLE_APIS 1
#అన్తర్భావ "apir_hw.h"
#అన్తర్భావ <drm/virtgpu_drm.h>
#అన్తర్భావ "venus_hw.h"
// clang-format on

#నిర్వచితం_కాకపోతే VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#అయితే_ముగింపు

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#నిర్వచించు VENUS_COMMAND_TYPE_నిడివి 331

#నిర్వచితం_కాకపోతే VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#అయితే_ముగింపు

రకనిర్వచనం uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#నిర్వచించు VIRTGPU_PCI_VENDOR_ID       0x1af4
#నిర్వచించు VIRTGPU_PCI_DEVICE_ID       0x1050
#నిర్వచించు VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#నిర్వచించు VIRTGPU_PARAM_GUEST_VRAM    9

#నిర్వచించు SHMEM_దత్తాంశం_కొలత  0x1830000  // 24MiB
#నిర్వచించు SHMEM_జవాబు_కొలత 0x4000

#నిర్వచించు ARRAY_కొలత(x) (పరిమాణం(x) / పరిమాణం((x)[0]))

గణన virt_gpu_ఫలితం_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#నిర్వచించు PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

నిర్మాణం virtgpu {
    bool use_apir_capset;

    పూర్ణాంకం fd;

    నిర్మాణం {
        virgl_renderer_capset      id;
        uint32_t                   సంస్కరణ_2;
        virgl_renderer_capset_apir దత్తాంశం;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem జవాబు_shmem;
    virtgpu_shmem దత్తాంశం_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t దత్తాంశం_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    నిర్మాణం {
        అక్షరం *   description;
        అక్షరం *   name;
        int32_t  device_లెక్క;
        uint32_t type;
        size_t   జ్ఞాపకం_free;
        size_t   జ్ఞాపకం_మొత్తం;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    నిర్మాణం {
        apir_బఫరు_type_host_handle_t host_handle;
        అక్షరం *                         name;
        size_t                         alignment;
        size_t                         max_కొలత;
    } cached_బఫరు_type;
};

స్థిర అంతఃవరుస పూర్ణాంకం virtgpu_ioctl(virtgpu * gpu, గుర్తులేని దీర్ఘ request, శూన్య * వాదనలు_2) {
    తిరిగి drmIoctl(gpu->fd, request, వాదనలు_2);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     దశాంశం           max_wait_ms,
                     దీర్ఘ దీర్ఘ *     call_duration_ns);

శూన్య remote_call_ముగింపు(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
