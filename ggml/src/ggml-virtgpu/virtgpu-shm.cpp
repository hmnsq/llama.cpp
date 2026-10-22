#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-shm.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-remoting.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>

ਸਥਿਰ uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
                                                   uint32_t   blob_mem,
                                                   uint32_t   blob_flags,
                                                   size_t     blob_ਮਾਪ,
                                                   uint64_t   blob_id,
                                                   uint32_t * res_id) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SIMULATE_BO_ਮਾਪ_FIX
    blob_ਮਾਪ = align64(blob_ਮਾਪ, 4096);
#ਜੇ_ਅੰਤ

    drm_virtgpu_resource_create_blob ਦਲੀਲਾਂ_2 = {
        .blob_mem   = blob_mem,
        .blob_flags = blob_flags,
        .bo_handle  = 0,
        .res_handle = 0,
        .size       = blob_ਮਾਪ,
        .pad        = 0,
        .cmd_ਮਾਪ   = 0,
        .cmd        = 0,
        .blob_id    = blob_id,
    };

    ਜੇ (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &ਦਲੀਲਾਂ_2)) {
        ਵਾਪਸ 0;
    }

    *res_id = ਦਲੀਲਾਂ_2.res_handle;
    ਵਾਪਸ ਦਲੀਲਾਂ_2.bo_handle;
}

ਸਥਿਰ ਖਾਲੀ virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close ਦਲੀਲਾਂ_2 = {
        .handle = gem_handle,
        .pad    = 0,
    };

    ਅਚਲ ਪੂਰਨ ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &ਦਲੀਲਾਂ_2);
    ਪ੍ਰਤਿਜ੍ਞਾ(!ret);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ NDEBUG
    ਅਣਵਰਤਿਆ(ret);
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ਖਾਲੀ * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, size_t size) {
    drm_virtgpu_map ਦਲੀਲਾਂ_2 = {
        .ਵਿੱਥ = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    ਜੇ (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &ਦਲੀਲਾਂ_2)) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਖਾਲੀ * ptr = mmap(ਨਿਰਾ, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, ਦਲੀਲਾਂ_2.ਵਿੱਥ);
    ਜੇ (ptr == MAP_FAILED) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ ptr;
}

ਖਾਲੀ virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_ਮਾਪ);
    virtgpu_ioctl_gem_close(gpu, shmem->gem_handle);
}

ਪੂਰਨ virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    ਜੇ (!gem_handle) {
        ਵਾਪਸ 1;
    }

    ਖਾਲੀ * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    ਜੇ (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        ਵਾਪਸ 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_ਮਾਪ  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    ਵਾਪਸ 0;
}
