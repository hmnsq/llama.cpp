#অন্তর্ভাব "virtgpu-shm.h"

#অন্তর্ভাব "virtgpu.h"
#অন্তর্ভাব "ggml-remoting.h"

#অন্তর্ভাব <assert.h>

স্থির uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
                                                   uint32_t   blob_mem,
                                                   uint32_t   blob_flags,
                                                   size_t     blob_আকার,
                                                   uint64_t   blob_id,
                                                   uint32_t * res_id) {
#ifdef SIMULATE_BO_আকার_FIX
    blob_আকার = align64(blob_আকার, 4096);
#endif

    drm_virtgpu_resource_create_blob args = {
        .blob_mem   = blob_mem,
        .blob_flags = blob_flags,
        .bo_handle  = 0,
        .res_handle = 0,
        .size       = blob_আকার,
        .pad        = 0,
        .cmd_আকার   = 0,
        .cmd        = 0,
        .blob_id    = blob_id,
    };

    যদি (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &args)) {
        ফেরত 0;
    }

    *res_id = args.res_handle;
    ফেরত args.bo_handle;
}

স্থির শূন্য virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close args = {
        .handle = gem_handle,
        .pad    = 0,
    };

    ধ্রুবক পূর্ণসংখ্যা ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &args);
    প্রতিজ্ঞা(!ret);
#ifdef NDEBUG
    UNUSED(ret);
#endif
}

স্থির শূন্য * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, size_t size) {
    drm_virtgpu_map args = {
        .offset = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    যদি (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &args)) {
        ফেরত ফাঁকা;
    }

    শূন্য * ptr = mmap(ফাঁকা, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, args.offset);
    যদি (ptr == MAP_FAILED) {
        ফেরত ফাঁকা;
    }

    ফেরত ptr;
}

শূন্য virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_আকার);
    virtgpu_ioctl_gem_close(gpu, shmem->gem_handle);
}

পূর্ণসংখ্যা virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    যদি (!gem_handle) {
        ফেরত 1;
    }

    শূন্য * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    যদি (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        ফেরত 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_আকার  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    ফেরত 0;
}
