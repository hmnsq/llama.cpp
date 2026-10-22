#ضمن "virtgpu-shm.h"

#ضمن "virtgpu.h"
#ضمن "ggml-remoting.h"

#ضمن <assert.h>

ساكن uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
                                                   uint32_t   blob_mem,
                                                   uint32_t   blob_flags,
                                                   size_t     blob_سائيز,
                                                   uint64_t   blob_id,
                                                   uint32_t * res_id) {
#جيڪڏهن_وصف_ٿيل SIMULATE_BO_سائيز_FIX
    blob_سائيز = align64(blob_سائيز, 4096);
#جيڪڏهن_پورو

    drm_virtgpu_resource_create_blob دليلون_2 = {
        .blob_mem   = blob_mem,
        .blob_flags = blob_flags,
        .bo_handle  = 0,
        .res_handle = 0,
        .size       = blob_سائيز,
        .pad        = 0,
        .cmd_سائيز   = 0,
        .cmd        = 0,
        .blob_id    = blob_id,
    };

    إذا (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &دليلون_2)) {
        أرجع 0;
    }

    *res_id = دليلون_2.res_handle;
    أرجع دليلون_2.bo_handle;
}

ساكن فراغ virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close دليلون_2 = {
        .handle = gem_handle,
        .pad    = 0,
    };

    ثابت صحيح ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &دليلون_2);
    أكد(!ret);
#جيڪڏهن_وصف_ٿيل NDEBUG
    اڻ_استعمال_ٿيل(ret);
#جيڪڏهن_پورو
}

ساكن فراغ * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, size_t size) {
    drm_virtgpu_map دليلون_2 = {
        .وٿي = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    إذا (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &دليلون_2)) {
        أرجع عدم;
    }

    فراغ * ptr = mmap(عدم, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, دليلون_2.وٿي);
    إذا (ptr == MAP_FAILED) {
        أرجع عدم;
    }

    أرجع ptr;
}

فراغ virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_سائيز);
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
    shmem->mmap_سائيز  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    أرجع 0;
}
