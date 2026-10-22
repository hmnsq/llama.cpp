#సూచన once

#అన్తర్భావ "virtgpu-utils.h"

#అన్తర్భావ <sys/mman.h>

#అన్తర్భావ <atomic>
#అన్తర్భావ <cassert>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>

నిర్మాణం virtgpu;

నిర్మాణం virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_కొలత;
    శూన్య *   mmap_ptr;

    uint32_t gem_handle;
};

పూర్ణాంకం  virtgpu_shmem_create(virtgpu * gpu, size_t కొలత, virtgpu_shmem * shmem);
శూన్య virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
