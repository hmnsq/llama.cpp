#شامل "virtgpu-shm.h"

#شامل "virtgpu.h"
#شامل "ggml-remoting.h"

#شامل <assert.h>

ساکن uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
                                                   uint32_t   blob_mem,
                                                   uint32_t   blob_flags,
                                                   size_t     blob_سائز,
                                                   uint64_t   blob_id,
                                                   uint32_t * res_id) {
#اگر_تعریف_شدہ SIMULATE_BO_سائز_FIX
    blob_سائز = align64(blob_سائز, 4096);
#اگر_ختم

    drm_virtgpu_resource_create_blob دلائل_2 = {
        .blob_mem   = blob_mem,
        .blob_flags = blob_flags,
        .bo_handle  = 0,
        .res_handle = 0,
        .size       = blob_سائز,
        .pad        = 0,
        .cmd_سائز   = 0,
        .cmd        = 0,
        .blob_id    = blob_id,
    };

    اگر (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &دلائل_2)) {
        واپس 0;
    }

    *res_id = دلائل_2.res_handle;
    واپس دلائل_2.bo_handle;
}

ساکن خالی virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close دلائل_2 = {
        .handle = gem_handle,
        .pad    = 0,
    };

    ثابت صحیح ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &دلائل_2);
    أكد(!ret);
#اگر_تعریف_شدہ NDEBUG
    غیر_مستعمل(ret);
#اگر_ختم
}

ساکن خالی * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, size_t size) {
    drm_virtgpu_map دلائل_2 = {
        .انحراف = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    اگر (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &دلائل_2)) {
        واپس تہی;
    }

    خالی * ptr = mmap(تہی, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, دلائل_2.انحراف);
    اگر (ptr == MAP_FAILED) {
        واپس تہی;
    }

    واپس ptr;
}

خالی virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_سائز);
    virtgpu_ioctl_gem_close(gpu, shmem->gem_handle);
}

صحیح virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    اگر (!gem_handle) {
        واپس 1;
    }

    خالی * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    اگر (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        واپس 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_سائز  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    واپس 0;
}
