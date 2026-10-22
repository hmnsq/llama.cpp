#அந்தர்பாவ "virtgpu-shm.h"

#அந்தர்பாவ "virtgpu.h"
#அந்தர்பாவ "ggml-remoting.h"

#அந்தர்பாவ <assert.h>

நிலையான uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
                                                   uint32_t   blob_mem,
                                                   uint32_t   blob_flags,
                                                   size_t     blob_size,
                                                   uint64_t   blob_id,
                                                   uint32_t * res_id) {
#ifdef SIMULATE_BO_SIZE_FIX
    blob_size = align64(blob_size, 4096);
#endif

    drm_virtgpu_resource_create_blob args = {
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

    எனில் (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &args)) {
        திரும்பு 0;
    }

    *res_id = args.res_handle;
    திரும்பு args.bo_handle;
}

நிலையான வெறுமை virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close args = {
        .handle = gem_handle,
        .pad    = 0,
    };

    மாறிலி முழு ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &args);
    ப்ரதிஜ்ஞா(!ret);
#ifdef NDEBUG
    UNUSED(ret);
#endif
}

நிலையான வெறுமை * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, size_t size) {
    drm_virtgpu_map args = {
        .offset = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    எனில் (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &args)) {
        திரும்பு வெற்று;
    }

    வெறுமை * ptr = mmap(வெற்று, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, args.offset);
    எனில் (ptr == MAP_FAILED) {
        திரும்பு வெற்று;
    }

    திரும்பு ptr;
}

வெறுமை virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_size);
    virtgpu_ioctl_gem_close(gpu, shmem->gem_handle);
}

முழு virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    எனில் (!gem_handle) {
        திரும்பு 1;
    }

    வெறுமை * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    எனில் (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        திரும்பு 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_size  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    திரும்பு 0;
}
