#अन्तर्भाव "virtgpu-shm.h"

#अन्तर्भाव "virtgpu.h"
#अन्तर्भाव "ggml-remoting.h"

#अन्तर्भाव <assert.h>

स्थिर uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
                                                   uint32_t   blob_mem,
                                                   uint32_t   blob_flags,
                                                   माप_प्रकार     blob_size,
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

    यदि (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &args)) {
        फर्काऊ 0;
    }

    *res_id = args.res_handle;
    फर्काऊ args.bo_handle;
}

स्थिर शून्य virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close args = {
        .handle = gem_handle,
        .pad    = 0,
    };

    अचर पूर्णाङ्क ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &args);
    प्रतिज्ञा(!ret);
#ifdef न_संशोधन
    UNUSED(ret);
#endif
}

स्थिर शून्य * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, माप_प्रकार size) {
    drm_virtgpu_map args = {
        .offset = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    यदि (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &args)) {
        फर्काऊ रिक्त;
    }

    शून्य * ptr = mmap(रिक्त, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, args.offset);
    यदि (ptr == MAP_FAILED) {
        फर्काऊ रिक्त;
    }

    फर्काऊ ptr;
}

शून्य virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_size);
    virtgpu_ioctl_gem_close(gpu, shmem->gem_handle);
}

पूर्णाङ्क virtgpu_shmem_create(virtgpu * gpu, माप_प्रकार size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    यदि (!gem_handle) {
        फर्काऊ 1;
    }

    शून्य * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    यदि (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        फर्काऊ 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_size  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    फर्काऊ 0;
}
