#आशय once

#समावेश "virtgpu-utils.h"

#समावेश <sys/mman.h>

#समावेश <atomic>
#समावेश <cassert>
#समावेश <cstddef>
#समावेश <cstdint>

काष्ठा virtgpu;

काष्ठा virtgpu_shmem {
    uint32_t res_id;
    माप_प्रकार   mmap_आकार;
    व्योम *   mmap_ptr;

    uint32_t gem_handle;
};

पूर्णांक  virtgpu_shmem_create(virtgpu * gpu, माप_प्रकार आकार, virtgpu_shmem * shmem);
व्योम virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
