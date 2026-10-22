#pragma once

#অন্তর্ভাব "virtgpu-utils.h"

#অন্তর্ভাব <sys/mman.h>

#অন্তর্ভাব <atomic>
#অন্তর্ভাব <cassert>
#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>

কাঠামো virtgpu;

কাঠামো virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_আকার;
    শূন্য *   mmap_ptr;

    uint32_t gem_handle;
};

পূর্ণসংখ্যা  virtgpu_shmem_create(virtgpu * gpu, size_t আকার_2, virtgpu_shmem * shmem);
শূন্য virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
