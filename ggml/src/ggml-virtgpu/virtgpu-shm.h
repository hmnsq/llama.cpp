#pragma once

#شامل "virtgpu-utils.h"

#شامل <sys/mman.h>

#شامل <atomic>
#شامل <cassert>
#شامل <cstddef>
#شامل <cstdint>

ساختار virtgpu;

ساختار virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_size;
    تهی *   mmap_ptr;

    uint32_t gem_handle;
};

صحیح  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);
تهی virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
