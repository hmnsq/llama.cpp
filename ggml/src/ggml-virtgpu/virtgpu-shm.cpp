#ଅନ୍ତର୍ଭାଵ "virtgpu-shm.h"

#ଅନ୍ତର୍ଭାଵ "virtgpu.h"
#ଅନ୍ତର୍ଭାଵ "ggml-remoting.h"

#ଅନ୍ତର୍ଭାଵ <assert.h>

ସ୍ଥିର uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
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

    ଯଦି (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &args)) {
        ଫେରନ୍ତୁ 0;
    }

    *res_id = args.res_handle;
    ଫେରନ୍ତୁ args.bo_handle;
}

ସ୍ଥିର ଶୂନ୍ୟ virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close args = {
        .handle = gem_handle,
        .pad    = 0,
    };

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &args);
    ପ୍ରତିଜ୍ଞା(!ret);
#ifdef NDEBUG
    UNUSED(ret);
#endif
}

ସ୍ଥିର ଶୂନ୍ୟ * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, size_t size) {
    drm_virtgpu_map args = {
        .offset = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    ଯଦି (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &args)) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଶୂନ୍ୟ * ptr = mmap(ଖାଲି, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, args.offset);
    ଯଦି (ptr == MAP_FAILED) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଫେରନ୍ତୁ ptr;
}

ଶୂନ୍ୟ virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_size);
    virtgpu_ioctl_gem_close(gpu, shmem->gem_handle);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    ଯଦି (!gem_handle) {
        ଫେରନ୍ତୁ 1;
    }

    ଶୂନ୍ୟ * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    ଯଦି (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        ଫେରନ୍ତୁ 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_size  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    ଫେରନ୍ତୁ 0;
}
