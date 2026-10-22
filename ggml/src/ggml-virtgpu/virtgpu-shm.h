#رهنمود once

#ضمن "virtgpu-utils.h"

#ضمن <sys/mman.h>

#ضمن <atomic>
#ضمن <cassert>
#ضمن <cstddef>
#ضمن <cstdint>

بنية virtgpu;

بنية virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_اندازه;
    فراغ *   mmap_ptr;

    uint32_t gem_handle;
};

صحيح  virtgpu_shmem_create(virtgpu * gpu, size_t اندازه, virtgpu_shmem * shmem);
فراغ virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
