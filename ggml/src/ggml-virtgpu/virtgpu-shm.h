#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <sys/mman.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

ਬਣਤਰ virtgpu;

ਬਣਤਰ virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_ਮਾਪ;
    ਖਾਲੀ *   mmap_ptr;

    uint32_t gem_handle;
};

ਪੂਰਨ  virtgpu_shmem_create(virtgpu * gpu, size_t ਮਾਪ, virtgpu_shmem * shmem);
ਖਾਲੀ virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
