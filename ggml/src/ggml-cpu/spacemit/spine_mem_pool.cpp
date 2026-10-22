#அந்தர்பாவ "spine_mem_pool.h"

#அந்தர்பாவ "common.h"
#அந்தர்பாவ "ime_env.h"
#அந்தர்பாவ "spine_tcm.h"

#அந்தர்பாவ <fcntl.h>
#அந்தர்பாவ <sys/ioctl.h>
#அந்தர்பாவ <sys/mman.h>
#அந்தர்பாவ <unistd.h>

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cerrno>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <limits>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <mutex>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <vector>

நாமாகாஶ ggml::cpu::riscv64_spacemit {
நாமாகாஶ {

constexpr size_t   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr எழுத்து     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr எழுத்து     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

கட்டமைப்பு hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, கட்டமைப்பு hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

கட்டமைப்பு free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

கட்டமைப்பு pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    முழு                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

கட்டமைப்பு pool_allocation {
    வெறுமை * chunk_base{ nullptr };
    size_t chunk_size{ 0 };
    வெறுமை * base{ nullptr };
    size_t size{ 0 };
};

தருக்க is_power_of_two(size_t value) {
    திரும்பு value != 0 && (value & (value - 1)) == 0;
}

தருக்க align_up(size_t value, size_t alignment, size_t * aligned_மதிப்பு) {
    எனில் (aligned_மதிப்பு == nullptr || alignment == 0) {
        திரும்பு பொய்;
    }

    மாறிலி size_t remainder = value % alignment;
    எனில் (remainder == 0) {
        *aligned_மதிப்பு = value;
        திரும்பு உண்மை;
    }

    மாறிலி size_t padding = alignment - remainder;
    எனில் (value > std::numeric_limits<size_t>::max() - padding) {
        திரும்பு பொய்;
    }

    *aligned_மதிப்பு = value + padding;
    திரும்பு உண்மை;
}

தருக்க align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_மதிப்பு) {
    எனில் (aligned_மதிப்பு == nullptr || alignment == 0) {
        திரும்பு பொய்;
    }

    மாறிலி uintptr_t remainder = value % alignment;
    எனில் (remainder == 0) {
        *aligned_மதிப்பு = value;
        திரும்பு உண்மை;
    }

    மாறிலி uintptr_t padding = alignment - remainder;
    எனில் (value > std::numeric_limits<uintptr_t>::max() - padding) {
        திரும்பு பொய்;
    }

    *aligned_மதிப்பு = value + padding;
    திரும்பு உண்மை;
}

வகுப்பு spine_mem_pool_manager {
  பொது:
    explicit spine_mem_pool_manager(size_t default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    மெய்நிகர் ~spine_mem_pool_manager() = இயல்பு;

    வெறுமை * alloc(size_t size, size_t alignment) {
        எனில் (size == 0 || !is_power_of_two(alignment)) {
            திரும்பு nullptr;
        }

        size_t aligned_size = 0;
        எனில் (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            திரும்பு nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        எனில் (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            எனில் (!add_chunk_locked(aligned_size, alignment)) {
                திரும்பு nullptr;
            }

            எனில் (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                திரும்பு nullptr;
            }
        }

        முயல் {
            மாறிலி auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            எனில் (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                திரும்பு nullptr;
            }
        } பிடி (மாறிலி std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            எறி;
        }

        திரும்பு allocation.base;
    }

    வெறுமை விடுவி(வெறுமை * base) {
        எனில் (base == nullptr) {
            திரும்பு;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        எனில் (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            திரும்பு;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        எனில் (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            திரும்பு;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        எனில் (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            திரும்பு;
        }

        மாறிலி size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        எனில் (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            திரும்பு;
        }

        செருகு_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  பாதுகாக்கப்பட்ட:
    வெறுமை release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        க்கு (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_size() மாறிலி { திரும்பு default_chunk_size_; }

    நிலையான வெறுமை clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    மெய்நிகர் தருக்க alloc_chunk(size_t min_size, size_t alignment, வெறுமை * hint_addr, pool_chunk * chunk) = 0;
    மெய்நிகர் வெறுமை dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  தனிப்பட்ட:
    கட்டமைப்பு alloc_candidate {
        size_t    chunk_index{ 0 };
        size_t    block_index{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        தருக்க      valid{ பொய் };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(மாறிலி pool_allocation & allocation) {
        திரும்பு std::find_if(chunks_.begin(), chunks_.end(), [&](மாறிலி pool_chunk & chunk) {
            திரும்பு chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    தருக்க add_chunk_locked(size_t min_size, size_t alignment) {
        pool_chunk   chunk;
        மாறிலி size_t chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        வெறுமை *       hint_addr     = nullptr;

        க்கு (மாறிலி auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            எனில் (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        எனில் (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            திரும்பு பொய்;
        }

        எனில் (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            திரும்பு பொய்;
        }

        முயல் {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } பிடி (மாறிலி std::bad_alloc &) {
            dealloc_chunk(&chunk);
            எறி;
        }

        திரும்பு உண்மை;
    }

    வெறுமை rollback_allocation_locked(மாறிலி pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        எனில் (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            திரும்பு;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        எனில் (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            திரும்பு;
        }

        மாறிலி size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        எனில் (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            திரும்பு;
        }

        செருகு_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    தருக்க try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        க்கு (size_t chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            மாறிலி auto & chunk = chunks_[chunk_index];
            க்கு (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                மாறிலி auto & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                மாறிலி auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                எனில் (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    தொடர்;
                }

                எனில் (aligned_addr < block_addr) {
                    தொடர்;
                }

                மாறிலி size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                மாறிலி size_t padding        = aligned_offset - block.offset;
                எனில் (padding > block.size || size > block.size - padding) {
                    தொடர்;
                }

                எனில் (!best.valid || aligned_addr < best.address) {
                    best.chunk_index    = chunk_index;
                    best.block_index    = block_index;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = உண்மை;
                }
            }
        }

        எனில் (!best.valid) {
            திரும்பு பொய்;
        }

        auto &           chunk     = chunks_[best.chunk_index];
        மாறிலி free_block block     = chunk.free_blocks[best.block_index];
        மாறிலி size_t     padding   = best.aligned_offset - block.offset;
        மாறிலி size_t     alloc_end = best.aligned_offset + size;
        மாறிலி size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        auto செருகு_it = chunk.free_blocks.begin() + best.block_index;
        எனில் (padding != 0) {
            செருகு_it = chunk.free_blocks.insert(செருகு_it, { block.offset, padding });
            ++செருகு_it;
        }
        எனில் (alloc_end < block_end) {
            chunk.free_blocks.insert(செருகு_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        திரும்பு உண்மை;
    }

    வெறுமை maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        எனில் (chunk_it->free_blocks.size() != 1) {
            திரும்பு;
        }

        மாறிலி auto & block = chunk_it->free_blocks.front();
        எனில் (block.offset != 0 || block.size != chunk_it->size) {
            திரும்பு;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    வெறுமை செருகு_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        போது (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        எனில் (it != chunk.free_blocks.begin()) {
            மாறிலி auto & prev = *(it - 1);
            எனில் (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                திரும்பு;
            }
        }

        எனில் (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            திரும்பு;
        }

        it = chunk.free_blocks.insert(it, block);

        எனில் (it != chunk.free_blocks.begin()) {
            auto prev = it - 1;
            எனில் (prev->offset + prev->size == it->offset) {
                it->offset = prev->offset;
                it->size += prev->size;
                it = chunk.free_blocks.erase(prev);
            }
        }

        எனில் (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<வெறுமை *, pool_allocation> allocations_;
    size_t                                      default_chunk_size_{ 0 };
};

வகுப்பு spine_mem_pool_posix final : பொது spine_mem_pool_manager {
  பொது:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  தனிப்பட்ட:
    தருக்க alloc_chunk(size_t min_size, size_t alignment, வெறுமை * hint_addr, pool_chunk * chunk) override {
        (வெறுமை) hint_addr;

        மாறிலி size_t alloc_alignment = std::max(alignment, அளவு(வெறுமை *));
        வெறுமை *       base            = nullptr;
        மாறிலி முழு    rc              = posix_memalign(&base, alloc_alignment, min_size);
        எனில் (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            திரும்பு பொய்;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        திரும்பு உண்மை;
    }

    வெறுமை dealloc_chunk(pool_chunk * chunk) override {
        std::விடுவி(chunk->base);
        clear_chunk(chunk);
    }
};

வகுப்பு spine_mem_pool_transparent_hugepage final : பொது spine_mem_pool_manager {
  பொது:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  தனிப்பட்ட:
    தருக்க alloc_chunk(size_t min_size, size_t alignment, வெறுமை * hint_addr, pool_chunk * chunk) override {
        (வெறுமை) alignment;

        size_t chunk_size = 0;
        எனில் (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            திரும்பு பொய்;
        }

        வெறுமை * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        எனில் (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           errno);
            திரும்பு பொய்;
        }

        எனில் (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, errno);
            munmap(map_addr, chunk_size);
            திரும்பு பொய்;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        திரும்பு உண்மை;
    }

    வெறுமை dealloc_chunk(pool_chunk * chunk) override {
        எனில் (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

வகுப்பு spine_mem_pool_hugetlb_1g final : பொது spine_mem_pool_manager {
  பொது:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  தனிப்பட்ட:
    தருக்க alloc_chunk(size_t min_size, size_t alignment, வெறுமை * hint_addr, pool_chunk * chunk) override {
        (வெறுமை) alignment;
        (வெறுமை) hint_addr;

        size_t region_size = 0;
        எனில் (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            திரும்பு பொய்;
        }

        மாறிலி முழு fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        எனில் (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            திரும்பு பொய்;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        எனில் (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, errno);
            close(fd);
            திரும்பு பொய்;
        }

        வெறுமை * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        எனில் (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            திரும்பு பொய்;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        திரும்பு உண்மை;
    }

    வெறுமை dealloc_chunk(pool_chunk * chunk) override {
        எனில் (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        எனில் (chunk->fd >= 0) {
            எனில் (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

வகுப்பு spine_mem_pool_shared_mem final : பொது spine_mem_pool_manager {
  பொது:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  தனிப்பட்ட:
    தருக்க alloc_chunk(size_t min_size, size_t alignment, வெறுமை * hint_addr, pool_chunk * chunk) override {
        (வெறுமை) alignment;

        எனில் (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            திரும்பு பொய்;
        }

        எனில் (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            திரும்பு பொய்;
        }

        மாறிலி முழு fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        எனில் (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            திரும்பு பொய்;
        }

        வெறுமை * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        எனில் (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), errno);
            close(fd);
            திரும்பு பொய்;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        திரும்பு உண்மை;
    }

    வெறுமை dealloc_chunk(pool_chunk * chunk) override {
        எனில் (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        எனில் (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    நிலையான std::once_flag                          pool_once;
    நிலையான std::unique_ptr<spine_mem_pool_manager> selected_pool;
    நிலையான spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    எனில் (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        தேர்வு (selected_backend) {
            நிலை spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                நிறுத்து;
            நிலை spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                நிறுத்து;
            நிலை spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                நிறுத்து;
            நிலை spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                நிறுத்து;
        }
    });

    எனில் (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<முழு>(backend), static_cast<முழு>(selected_backend));
    }

    எனில் (selected_pool) {
        திரும்பு *selected_pool;
    }

    எறி std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    நிலையான std::once_flag                             shared_mem_pool_once;
    நிலையான std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    எனில் (shared_mem_pool) {
        திரும்பு *shared_mem_pool;
    }

    எறி std::bad_alloc();
}

}  // namespace

தருக்க spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    எனில் (info == nullptr) {
        திரும்பு பொய்;
    }

    *info = {};

    எனில் (spine_tcm_open_handle(வெற்று) != 0 || !spine_tcm_is_available()) {
        திரும்பு பொய்;
    }

    spine_tcm_mem_info_t mem_info;
    எனில் (spine_tcm_mem_info(&mem_info) != 0) {
        திரும்பு பொய்;
    }

    info->available   = உண்மை;
    info->blk_size    = mem_info.blk_size;
    info->blk_இலக்கம்     = mem_info.blk_இலக்கம்;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    திரும்பு உண்மை;
}

வெறுமை * spine_mem_pool_tcm_mem_get(முழு cpu_id) noexcept {
    திரும்பு spine_tcm_mem_get(cpu_id);
}

வெறுமை * spine_mem_pool_tcm_mem_wait(முழு cpu_id) noexcept {
    திரும்பு spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

முழு spine_mem_pool_tcm_mem_release(முழு cpu_id) noexcept {
    திரும்பு spine_tcm_mem_release(cpu_id);
}

வெறுமை * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    முயல் {
        திரும்பு get_spine_mem_pool_manager().alloc(size, alignment);
    } பிடி (மாறிலி std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        திரும்பு nullptr;
    }
}

வெறுமை * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    முயல் {
        திரும்பு get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } பிடி (மாறிலி std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        திரும்பு nullptr;
    }
}

வெறுமை spine_mem_pool_free(வெறுமை * base) noexcept {
    முயல் {
        get_spine_mem_pool_manager().விடுவி(base);
    } பிடி (மாறிலி std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

வெறுமை spine_mem_pool_shared_mem_free(வெறுமை * base) noexcept {
    முயல் {
        get_spine_mem_pool_shared_mem_manager().விடுவி(base);
    } பிடி (மாறிலி std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

வெளி "C" {
வெறுமை * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    வெறுமை * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    எனில் (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    திரும்பு result;
}

வெறுமை ggml_backend_cpu_riscv64_spacemit_free_shared(வெறுமை * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
