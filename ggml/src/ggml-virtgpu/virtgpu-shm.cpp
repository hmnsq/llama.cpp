#अन्तर्भाव "virtgpu-shm.h"

#अन्तर्भाव "virtgpu.h"
#अन्तर्भाव "ggml-remoting.h"

#अन्तर्भाव <assert.h>

स्थिर uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
                                                   uint32_t   blob_mem,
                                                   uint32_t   blob_flags,
                                                   size_t     blob_आकार,
                                                   uint64_t   blob_id,
                                                   uint32_t * res_id) {
#यदि_परिभाषितम् SIMULATE_BO_आकार_FIX
    blob_आकार = align64(blob_आकार, 4096);
#यदि_अन्तः

    drm_virtgpu_resource_create_blob तर्काः_2 = {
        .blob_mem   = blob_mem,
        .blob_flags = blob_flags,
        .bo_handle  = 0,
        .res_handle = 0,
        .size       = blob_आकार,
        .pad        = 0,
        .cmd_आकार   = 0,
        .cmd        = 0,
        .blob_id    = blob_id,
    };

    यदि (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &तर्काः_2)) {
        प्रत्यागम 0;
    }

    *res_id = तर्काः_2.res_handle;
    प्रत्यागम तर्काः_2.bo_handle;
}

स्थिर शून्य virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close तर्काः_2 = {
        .handle = gem_handle,
        .pad    = 0,
    };

    ध्रुव पूर्णाङ्क ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &तर्काः_2);
    प्रतिज्ञा(!ret);
#यदि_परिभाषितम् NDEBUG
    अप्रयुक्तम्(ret);
#यदि_अन्तः
}

स्थिर शून्य * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, size_t size) {
    drm_virtgpu_map तर्काः_2 = {
        .विस्थापनम् = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    यदि (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &तर्काः_2)) {
        प्रत्यागम रिक्त;
    }

    शून्य * ptr = mmap(रिक्त, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, तर्काः_2.विस्थापनम्);
    यदि (ptr == MAP_FAILED) {
        प्रत्यागम रिक्त;
    }

    प्रत्यागम ptr;
}

शून्य virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_आकार);
    virtgpu_ioctl_gem_close(gpu, shmem->gem_handle);
}

पूर्णाङ्क virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    यदि (!gem_handle) {
        प्रत्यागम 1;
    }

    शून्य * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    यदि (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        प्रत्यागम 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_आकार  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    प्रत्यागम 0;
}
