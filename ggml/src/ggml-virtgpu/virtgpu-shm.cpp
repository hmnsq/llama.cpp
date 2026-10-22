#అన్తర్భావ "virtgpu-shm.h"

#అన్తర్భావ "virtgpu.h"
#అన్తర్భావ "ggml-remoting.h"

#అన్తర్భావ <assert.h>

స్థిర uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
                                                   uint32_t   blob_mem,
                                                   uint32_t   blob_flags,
                                                   size_t     blob_కొలత,
                                                   uint64_t   blob_id,
                                                   uint32_t * res_id) {
#నిర్వచితమైతే SIMULATE_BO_కొలత_FIX
    blob_కొలత = align64(blob_కొలత, 4096);
#అయితే_ముగింపు

    drm_virtgpu_resource_create_blob వాదనలు_2 = {
        .blob_mem   = blob_mem,
        .blob_flags = blob_flags,
        .bo_handle  = 0,
        .res_handle = 0,
        .size       = blob_కొలత,
        .pad        = 0,
        .cmd_కొలత   = 0,
        .cmd        = 0,
        .blob_id    = blob_id,
    };

    ఒకవేళ (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &వాదనలు_2)) {
        తిరిగి 0;
    }

    *res_id = వాదనలు_2.res_handle;
    తిరిగి వాదనలు_2.bo_handle;
}

స్థిర శూన్య virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close వాదనలు_2 = {
        .handle = gem_handle,
        .pad    = 0,
    };

    స్థిరరాశి పూర్ణాంకం ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &వాదనలు_2);
    ప్రతిజ్ఞా(!ret);
#నిర్వచితమైతే NDEBUG
    వాడని(ret);
#అయితే_ముగింపు
}

స్థిర శూన్య * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, size_t size) {
    drm_virtgpu_map వాదనలు_2 = {
        .స్థానభ్రంశం = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    ఒకవేళ (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &వాదనలు_2)) {
        తిరిగి ఖాళీ;
    }

    శూన్య * ptr = mmap(ఖాళీ, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, వాదనలు_2.స్థానభ్రంశం);
    ఒకవేళ (ptr == MAP_FAILED) {
        తిరిగి ఖాళీ;
    }

    తిరిగి ptr;
}

శూన్య virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_కొలత);
    virtgpu_ioctl_gem_close(gpu, shmem->gem_handle);
}

పూర్ణాంకం virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    ఒకవేళ (!gem_handle) {
        తిరిగి 1;
    }

    శూన్య * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    ఒకవేళ (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        తిరిగి 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_కొలత  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    తిరిగి 0;
}
