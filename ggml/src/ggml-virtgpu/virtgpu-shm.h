#pragma once

#ಅನ್ತರ್ಭಾವ "virtgpu-utils.h"

#ಅನ್ತರ್ಭಾವ <sys/mman.h>

#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>

ಸಂರಚನಾ virtgpu;

ಸಂರಚನಾ virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_size;
    ಶೂನ್ಯ *   mmap_ptr;

    uint32_t gem_handle;
};

ಪೂರ್ಣಾಙ್ಕ  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);
ಶೂನ್ಯ virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
