#pragma once

#অন্তর্ভাৱ "virtgpu-utils.h"

#অন্তর্ভাৱ <sys/mman.h>

#অন্তর্ভাৱ <atomic>
#অন্তর্ভাৱ <cassert>
#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <cstdint>

সংরচনা virtgpu;

সংরচনা virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_size;
    শূন্য *   mmap_ptr;

    uint32_t gem_handle;
};

পূর্ণাঙ্ক  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);
শূন্য virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
