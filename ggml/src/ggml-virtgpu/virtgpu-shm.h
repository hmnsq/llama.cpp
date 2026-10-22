#ہدایت once

#شامل "virtgpu-utils.h"

#شامل <sys/mman.h>

#شامل <atomic>
#شامل <cassert>
#شامل <cstddef>
#شامل <cstdint>

ساخت virtgpu;

ساخت virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_سائز;
    خالی *   mmap_ptr;

    uint32_t gem_handle;
};

صحیح  virtgpu_shmem_create(virtgpu * gpu, size_t سائز, virtgpu_shmem * shmem);
خالی virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
