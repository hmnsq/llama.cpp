#pragma once

#അന്തര്ഭാവ "virtgpu-utils.h"

#അന്തര്ഭാവ <sys/mman.h>

#അന്തര്ഭാവ <atomic>
#അന്തര്ഭാവ <cassert>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>

ഘടന virtgpu;

ഘടന virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_size;
    ശൂന്യം *   mmap_ptr;

    uint32_t gem_handle;
};

പൂർണ്ണസംഖ്യ  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);
ശൂന്യം virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
