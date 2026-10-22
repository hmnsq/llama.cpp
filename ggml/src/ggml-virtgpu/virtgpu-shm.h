#pragma once

#அந்தர்பாவ "virtgpu-utils.h"

#அந்தர்பாவ <sys/mman.h>

#அந்தர்பாவ <atomic>
#அந்தர்பாவ <cassert>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>

கட்டமைப்பு virtgpu;

கட்டமைப்பு virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_size;
    வெறுமை *   mmap_ptr;

    uint32_t gem_handle;
};

முழு  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);
வெறுமை virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
