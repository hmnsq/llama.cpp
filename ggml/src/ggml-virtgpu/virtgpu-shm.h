#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "virtgpu-utils.h"

#ಅನ್ತರ್ಭಾವ <sys/mman.h>

#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>

ರಚನೆ virtgpu;

ರಚನೆ virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_ಗಾತ್ರ;
    ಶೂನ್ಯ *   mmap_ptr;

    uint32_t gem_handle;
};

ಪೂರ್ಣಾಂಕ  virtgpu_shmem_create(virtgpu * gpu, size_t ಗಾತ್ರ_2, virtgpu_shmem * shmem);
ಶೂನ್ಯ virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
