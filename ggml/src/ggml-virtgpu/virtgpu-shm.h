#pragma once

#অন্তর্ভাব "virtgpu-utils.h"

#অন্তর্ভাব <sys/mman.h>

#অন্তর্ভাব <atomic>
#অন্তর্ভাব <cassert>
#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>

সংরচনা virtgpu;

সংরচনা virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_size;
    শূন্য *   mmap_ptr;

    uint32_t gem_handle;
};

পূর্ণাঙ্ক  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);
শূন্য virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
