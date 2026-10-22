#आशय once

// clang-format off
#समावेश "virtgpu-utils.h"
#समावेश "virtgpu-shm.h"
#समावेश "virtgpu-apir.h"

#समावेश "backend/shared/api_remoting.h"
#समावेश "backend/shared/apir_cs.h"

#समावेश <fcntl.h>
#समावेश <stdbool.h>
#समावेश <stdio.h>
#समावेश <sys/stat.h>
#समावेश <sys/sysmacros.h>
#समावेश <threads.h>
#समावेश <xf86drm.h>

#समावेश <cstring>

#घोषणा VIRGL_RENDERER_UNSTABLE_APIS 1
#समावेश "apir_hw.h"
#समावेश <drm/virtgpu_drm.h>
#समावेश "venus_hw.h"
// clang-format on

#अगर_अघोषित VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10
#पूर्ण_अगर

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#घोषणा VENUS_COMMAND_TYPE_लंबाई 331

#अगर_अघोषित VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16
#    define VIRTGPU_DRM_CAPSET_VENUS 4
#पूर्ण_अगर

प्रकार uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#घोषणा VIRTGPU_PCI_VENDOR_ID       0x1af4
#घोषणा VIRTGPU_PCI_DEVICE_ID       0x1050
#घोषणा VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#घोषणा VIRTGPU_PARAM_GUEST_VRAM    9

#घोषणा SHMEM_तथ्य_आकार  0x1830000  // 24MiB
#घोषणा SHMEM_REPLY_आकार 0x4000

#घोषणा ARRAY_आकार(x) (माप(x) / माप((x)[0]))

क्रमागत virt_gpu_परिणाम_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#घोषणा PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

काष्ठा virtgpu {
    bool use_apir_capset;

    पूर्णांक fd;

    काष्ठा {
        virgl_renderer_capset      id;
        uint32_t                   version;
        virgl_renderer_capset_apir तथ्य;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem reply_shmem;
    virtgpu_shmem तथ्य_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t तथ्य_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    काष्ठा {
        अक्षर *   description;
        अक्षर *   name;
        int32_t  device_गिनती;
        uint32_t type;
        माप_प्रकार   memory_free;
        माप_प्रकार   memory_योग;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    काष्ठा {
        apir_बफर_type_host_handle_t host_handle;
        अक्षर *                         name;
        माप_प्रकार                         alignment;
        माप_प्रकार                         max_आकार;
    } cached_बफर_type;
};

अटल अंतरभूत पूर्णांक virtgpu_ioctl(virtgpu * gpu, अचिन्हित दीर्घ request, व्योम * args) {
    वापस drmIoctl(gpu->fd, request, args);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     भग्न           max_wait_ms,
                     दीर्घ दीर्घ *     call_duration_ns);

व्योम remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
