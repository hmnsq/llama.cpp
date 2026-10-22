#pragma once

#अन्तर्भाव "virtgpu-utils.h"

#अन्तर्भाव <sys/mman.h>

#अन्तर्भाव <atomic>
#अन्तर्भाव <cassert>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>

संरचना virtgpu;

संरचना virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_size;
    शून्य *   mmap_ptr;

    uint32_t gem_handle;
};

पूर्णाङ्क  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);
शून्य virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
