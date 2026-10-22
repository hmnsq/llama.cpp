#pragma once

#ଅନ୍ତର୍ଭାଵ "virtgpu-utils.h"

#ଅନ୍ତର୍ଭାଵ <sys/mman.h>

#ଅନ୍ତର୍ଭାଵ <atomic>
#ଅନ୍ତର୍ଭାଵ <cassert>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>

ସଂରଚନା virtgpu;

ସଂରଚନା virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_size;
    ଶୂନ୍ୟ *   mmap_ptr;

    uint32_t gem_handle;
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);
ଶୂନ୍ୟ virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
