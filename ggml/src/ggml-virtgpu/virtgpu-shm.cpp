#समावेश "virtgpu-shm.h"

#समावेश "virtgpu.h"
#समावेश "ggml-remoting.h"

#समावेश <assert.h>

अटल uint32_t virtgpu_ioctl_resource_create_blob(virtgpu *  gpu,
                                                   uint32_t   blob_mem,
                                                   uint32_t   blob_flags,
                                                   माप_प्रकार     blob_आकार,
                                                   uint64_t   blob_id,
                                                   uint32_t * res_id) {
#अगर_घोषित SIMULATE_BO_आकार_FIX
    blob_आकार = align64(blob_आकार, 4096);
#पूर्ण_अगर

    drm_virtgpu_resource_create_blob args = {
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

    अगर (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_RESOURCE_CREATE_BLOB, &args)) {
        वापस 0;
    }

    *res_id = args.res_handle;
    वापस args.bo_handle;
}

अटल व्योम virtgpu_ioctl_gem_close(virtgpu * gpu, uint32_t gem_handle) {
    drm_gem_close args = {
        .handle = gem_handle,
        .pad    = 0,
    };

    स्थिर पूर्णांक ret = virtgpu_ioctl(gpu, DRM_IOCTL_GEM_CLOSE, &args);
    निश्चित(!ret);
#अगर_घोषित न_संशोधन
    UNUSED(ret);
#पूर्ण_अगर
}

अटल व्योम * virtgpu_ioctl_map(virtgpu * gpu, uint32_t gem_handle, माप_प्रकार size) {
    drm_virtgpu_map args = {
        .offset = 0,
        .handle = gem_handle,
        .pad    = 0,
    };

    अगर (virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_MAP, &args)) {
        वापस शून्य;
    }

    व्योम * ptr = mmap(शून्य, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu->fd, args.offset);
    अगर (ptr == MAP_FAILED) {
        वापस शून्य;
    }

    वापस ptr;
}

व्योम virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem) {
    munmap(shmem->mmap_ptr, shmem->mmap_आकार);
    virtgpu_ioctl_gem_close(gpu, shmem->gem_handle);
}

पूर्णांक virtgpu_shmem_create(virtgpu * gpu, माप_प्रकार size, virtgpu_shmem * shmem) {
    size = align64(size, 16384);

    uint32_t res_id;
    uint32_t gem_handle = virtgpu_ioctl_resource_create_blob(gpu, VIRTGPU_BLOB_MEM_HOST3D,
                                                             VIRTGPU_BLOB_FLAG_USE_MAPPABLE, size, 0, &res_id);

    अगर (!gem_handle) {
        वापस 1;
    }

    व्योम * ptr = virtgpu_ioctl_map(gpu, gem_handle, size);
    अगर (!ptr) {
        virtgpu_ioctl_gem_close(gpu, gem_handle);
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: virtgpu_ioctl_map failed\n", __func__);
        वापस 1;
    }

    shmem->res_id     = res_id;
    shmem->mmap_आकार  = size;
    shmem->mmap_ptr   = ptr;
    shmem->gem_handle = gem_handle;

    वापस 0;
}
