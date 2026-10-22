#pragma once

#અન્તર્ભાવ "virtgpu-utils.h"

#અન્તર્ભાવ <sys/mman.h>

#અન્તર્ભાવ <atomic>
#અન્તર્ભાવ <cassert>
#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>

સંરચના virtgpu;

સંરચના virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_size;
    શૂન્ય *   mmap_ptr;

    uint32_t gem_handle;
};

પૂર્ણાંક  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);
શૂન્ય virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
