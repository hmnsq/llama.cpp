#ضمن "virtgpu-shm.h"

#ضمن "virtgpu.h"
#ضمن "ggml-remoting.h"

#ضمن <assert.h>

ساكن uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
                                                   uint32_t   blob_mem,
                                                   uint32_t   blob_flags,
                                                   size_t     blob_size,
                                                   uint64_t   blob_id,
                                                   uint32_t * res_id) {
#ifdef SIMULATE_BO_SIZE_FIX
    blob_size = align64(blob_size, 4096);
#endif

    drm_virtgpu_resource_create_blob وسائط_2 = {
        .blob_mem   = blob_mem,
        .blob_flags = blob_flags,
        .bo_handle  = 0,
        .res_handle = 0,
        .size       = blob_size,
        .pad        = 0,
        .cmd_size   = 0,
        .cmd        = 0,
        .blob_id    = blob_id,
    };

    إذا (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &وسائط_2)) {
        أرجع 0;
    }

    *res_id = وسائط_2.res_handle;
    أرجع وسائط_2.bo_handle;
}

ساكن فراغ virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close وسائط_2 = {
        .handle = gem_handle,
        .pad    = 0,
    };

    ثابت صحيح ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &وسائط_2);
    أكد(!ret);
#ifdef NDEBUG
    UNUSED(ret);
#endif
}

ساكن فراغ * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, size_t size) {
    drm_virtgpu_map وسائط_2 = {
        .offset = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    إذا (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &وسائط_2)) {
        أرجع عدم;
    }

    فراغ * ptr = mmap(عدم, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, وسائط_2.offset);
    إذا (ptr == MAP_FAILED) {
        أرجع عدم;
    }

    أرجع ptr;
}

فراغ virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_size);
    virtgpu_ioctl_gem_close(gpu, shmem->gem_handle);
}

صحيح virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    إذا (!gem_handle) {
        أرجع 1;
    }

    فراغ * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    إذا (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        أرجع 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_size  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    أرجع 0;
}
