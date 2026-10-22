#ಅನ್ತರ್ಭಾವ "spine_mem_pool.h"

#ಅನ್ತರ್ಭಾವ "common.h"
#ಅನ್ತರ್ಭಾವ "ime_env.h"
#ಅನ್ತರ್ಭಾವ "spine_tcm.h"

#ಅನ್ತರ್ಭಾವ <fcntl.h>
#ಅನ್ತರ್ಭಾವ <sys/ioctl.h>
#ಅನ್ತರ್ಭಾವ <sys/mman.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <limits>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ggml::cpu::riscv64_spacemit {
ನಾಮಾಕಾಶ {

constexpr size_t   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr ವರ್ಣ     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr ವರ್ಣ     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

ಸಂರಚನಾ hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, ಸಂರಚನಾ hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

ಸಂರಚನಾ free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

ಸಂರಚನಾ pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    ಪೂರ್ಣಾಙ್ಕ                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

ಸಂರಚನಾ pool_allocation {
    ಶೂನ್ಯ * chunk_base{ nullptr };
    size_t chunk_size{ 0 };
    ಶೂನ್ಯ * base{ nullptr };
    size_t size{ 0 };
};

ತಾರ್ಕಿಕ is_power_of_two(size_t value) {
    ಪ್ರತ್ಯಾಗಮ value != 0 && (value & (value - 1)) == 0;
}

ತಾರ್ಕಿಕ align_up(size_t value, size_t alignment, size_t * aligned_ಮೌಲ್ಯ) {
    ಯದಿ (aligned_ಮೌಲ್ಯ == nullptr || alignment == 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ size_t remainder = value % alignment;
    ಯದಿ (remainder == 0) {
        *aligned_ಮೌಲ್ಯ = value;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಧ್ರುವ size_t padding = alignment - remainder;
    ಯದಿ (value > std::numeric_limits<size_t>::max() - padding) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    *aligned_ಮೌಲ್ಯ = value + padding;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_ಮೌಲ್ಯ) {
    ಯದಿ (aligned_ಮೌಲ್ಯ == nullptr || alignment == 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uintptr_t remainder = value % alignment;
    ಯದಿ (remainder == 0) {
        *aligned_ಮೌಲ್ಯ = value;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಧ್ರುವ uintptr_t padding = alignment - remainder;
    ಯದಿ (value > std::numeric_limits<uintptr_t>::max() - padding) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    *aligned_ಮೌಲ್ಯ = value + padding;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ವರ್ಗ spine_mem_pool_manager {
  ಸಾರ್ವಜನಿಕ:
    explicit spine_mem_pool_manager(size_t default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    ಆಭಾಸ ~spine_mem_pool_manager() = ಸ್ವಭಾವ;

    ಶೂನ್ಯ * alloc(size_t size, size_t alignment) {
        ಯದಿ (size == 0 || !is_power_of_two(alignment)) {
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        size_t aligned_size = 0;
        ಯದಿ (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        ಯದಿ (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            ಯದಿ (!add_chunk_locked(aligned_size, alignment)) {
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }

            ಯದಿ (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
        }

        ಯತ್ನ {
            ಧ್ರುವ auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            ಯದಿ (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
        } ಗೃಹಾಣ (ಧ್ರುವ std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            ಕ್ಷಿಪ;
        }

        ಪ್ರತ್ಯಾಗಮ allocation.base;
    }

    ಶೂನ್ಯ ಮುಕ್ತ(ಶೂನ್ಯ * base) {
        ಯದಿ (base == nullptr) {
            ಪ್ರತ್ಯಾಗಮ;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        ಯದಿ (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            ಪ್ರತ್ಯಾಗಮ;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        ಯದಿ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ಪ್ರತ್ಯಾಗಮ;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        ಯದಿ (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಧ್ರುವ size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        ಯದಿ (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಸೇರಿಸು_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  ರಕ್ಷಿತ:
    ಶೂನ್ಯ release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        ಪ್ರತಿ (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ default_chunk_size_; }

    ಸ್ಥಿರ ಶೂನ್ಯ clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    ಆಭಾಸ ತಾರ್ಕಿಕ alloc_chunk(size_t min_size, size_t alignment, ಶೂನ್ಯ * hint_addr, pool_chunk * chunk) = 0;
    ಆಭಾಸ ಶೂನ್ಯ dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  ಗುಪ್ತ:
    ಸಂರಚನಾ alloc_candidate {
        size_t    chunk_index{ 0 };
        size_t    block_index{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        ತಾರ್ಕಿಕ      valid{ ಅಸತ್ಯ };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(ಧ್ರುವ pool_allocation & allocation) {
        ಪ್ರತ್ಯಾಗಮ std::find_if(chunks_.begin(), chunks_.end(), [&](ಧ್ರುವ pool_chunk & chunk) {
            ಪ್ರತ್ಯಾಗಮ chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    ತಾರ್ಕಿಕ add_chunk_locked(size_t min_size, size_t alignment) {
        pool_chunk   chunk;
        ಧ್ರುವ size_t chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        ಶೂನ್ಯ *       hint_addr     = nullptr;

        ಪ್ರತಿ (ಧ್ರುವ auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            ಯದಿ (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        ಯದಿ (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯತ್ನ {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } ಗೃಹಾಣ (ಧ್ರುವ std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ಕ್ಷಿಪ;
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಶೂನ್ಯ rollback_allocation_locked(ಧ್ರುವ pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        ಯದಿ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            ಪ್ರತ್ಯಾಗಮ;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        ಯದಿ (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಧ್ರುವ size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        ಯದಿ (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಸೇರಿಸು_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    ತಾರ್ಕಿಕ try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        ಪ್ರತಿ (size_t chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            ಧ್ರುವ auto & chunk = chunks_[chunk_index];
            ಪ್ರತಿ (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                ಧ್ರುವ auto & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                ಧ್ರುವ auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                ಯದಿ (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    ಅನುವರ್ತ;
                }

                ಯದಿ (aligned_addr < block_addr) {
                    ಅನುವರ್ತ;
                }

                ಧ್ರುವ size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                ಧ್ರುವ size_t padding        = aligned_offset - block.offset;
                ಯದಿ (padding > block.size || size > block.size - padding) {
                    ಅನುವರ್ತ;
                }

                ಯದಿ (!best.valid || aligned_addr < best.address) {
                    best.chunk_index    = chunk_index;
                    best.block_index    = block_index;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = ಸತ್ಯ;
                }
            }
        }

        ಯದಿ (!best.valid) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        auto &           chunk     = chunks_[best.chunk_index];
        ಧ್ರುವ free_block block     = chunk.free_blocks[best.block_index];
        ಧ್ರುವ size_t     padding   = best.aligned_offset - block.offset;
        ಧ್ರುವ size_t     alloc_end = best.aligned_offset + size;
        ಧ್ರುವ size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        auto ಸೇರಿಸು_it = chunk.free_blocks.begin() + best.block_index;
        ಯದಿ (padding != 0) {
            ಸೇರಿಸು_it = chunk.free_blocks.insert(ಸೇರಿಸು_it, { block.offset, padding });
            ++ಸೇರಿಸು_it;
        }
        ಯದಿ (alloc_end < block_end) {
            chunk.free_blocks.insert(ಸೇರಿಸು_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಶೂನ್ಯ maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        ಯದಿ (chunk_it->free_blocks.size() != 1) {
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಧ್ರುವ auto & block = chunk_it->free_blocks.front();
        ಯದಿ (block.offset != 0 || block.size != chunk_it->size) {
            ಪ್ರತ್ಯಾಗಮ;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    ಶೂನ್ಯ ಸೇರಿಸು_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        ಯಾವತ್ (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        ಯದಿ (it != chunk.free_blocks.begin()) {
            ಧ್ರುವ auto & prev = *(it - 1);
            ಯದಿ (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                ಪ್ರತ್ಯಾಗಮ;
            }
        }

        ಯದಿ (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            ಪ್ರತ್ಯಾಗಮ;
        }

        it = chunk.free_blocks.insert(it, block);

        ಯದಿ (it != chunk.free_blocks.begin()) {
            auto prev = it - 1;
            ಯದಿ (prev->offset + prev->size == it->offset) {
                it->offset = prev->offset;
                it->size += prev->size;
                it = chunk.free_blocks.erase(prev);
            }
        }

        ಯದಿ (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<ಶೂನ್ಯ *, pool_allocation> allocations_;
    size_t                                      default_chunk_size_{ 0 };
};

ವರ್ಗ spine_mem_pool_posix final : ಸಾರ್ವಜನಿಕ spine_mem_pool_manager {
  ಸಾರ್ವಜನಿಕ:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  ಗುಪ್ತ:
    ತಾರ್ಕಿಕ alloc_chunk(size_t min_size, size_t alignment, ಶೂನ್ಯ * hint_addr, pool_chunk * chunk) override {
        (ಶೂನ್ಯ) hint_addr;

        ಧ್ರುವ size_t alloc_alignment = std::max(alignment, ಪರಿಮಾಣ(ಶೂನ್ಯ *));
        ಶೂನ್ಯ *       base            = nullptr;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ    rc              = posix_memalign(&base, alloc_alignment, min_size);
        ಯದಿ (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಶೂನ್ಯ dealloc_chunk(pool_chunk * chunk) override {
        std::ಮುಕ್ತ(chunk->base);
        clear_chunk(chunk);
    }
};

ವರ್ಗ spine_mem_pool_transparent_hugepage final : ಸಾರ್ವಜನಿಕ spine_mem_pool_manager {
  ಸಾರ್ವಜನಿಕ:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  ಗುಪ್ತ:
    ತಾರ್ಕಿಕ alloc_chunk(size_t min_size, size_t alignment, ಶೂನ್ಯ * hint_addr, pool_chunk * chunk) override {
        (ಶೂನ್ಯ) alignment;

        size_t chunk_size = 0;
        ಯದಿ (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಶೂನ್ಯ * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ಯದಿ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           errno);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, errno);
            munmap(map_addr, chunk_size);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಶೂನ್ಯ dealloc_chunk(pool_chunk * chunk) override {
        ಯದಿ (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

ವರ್ಗ spine_mem_pool_hugetlb_1g final : ಸಾರ್ವಜನಿಕ spine_mem_pool_manager {
  ಸಾರ್ವಜನಿಕ:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  ಗುಪ್ತ:
    ತಾರ್ಕಿಕ alloc_chunk(size_t min_size, size_t alignment, ಶೂನ್ಯ * hint_addr, pool_chunk * chunk) override {
        (ಶೂನ್ಯ) alignment;
        (ಶೂನ್ಯ) hint_addr;

        size_t region_size = 0;
        ಯದಿ (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        ಯದಿ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        ಯದಿ (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, errno);
            close(fd);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಶೂನ್ಯ * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ಯದಿ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಶೂನ್ಯ dealloc_chunk(pool_chunk * chunk) override {
        ಯದಿ (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        ಯದಿ (chunk->fd >= 0) {
            ಯದಿ (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

ವರ್ಗ spine_mem_pool_shared_mem final : ಸಾರ್ವಜನಿಕ spine_mem_pool_manager {
  ಸಾರ್ವಜನಿಕ:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  ಗುಪ್ತ:
    ತಾರ್ಕಿಕ alloc_chunk(size_t min_size, size_t alignment, ಶೂನ್ಯ * hint_addr, pool_chunk * chunk) override {
        (ಶೂನ್ಯ) alignment;

        ಯದಿ (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        ಯದಿ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಶೂನ್ಯ * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ಯದಿ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), errno);
            close(fd);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಶೂನ್ಯ dealloc_chunk(pool_chunk * chunk) override {
        ಯದಿ (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        ಯದಿ (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    ಸ್ಥಿರ std::once_flag                          pool_once;
    ಸ್ಥಿರ std::unique_ptr<spine_mem_pool_manager> selected_pool;
    ಸ್ಥಿರ spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    ಯದಿ (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        ವಿಕಲ್ಪ (selected_backend) {
            ಪಕ್ಷ spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                ಭಙ್ಗ;
            ಪಕ್ಷ spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ಭಙ್ಗ;
            ಪಕ್ಷ spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                ಭಙ್ಗ;
            ಪಕ್ಷ spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ಭಙ್ಗ;
        }
    });

    ಯದಿ (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<ಪೂರ್ಣಾಙ್ಕ>(backend), static_cast<ಪೂರ್ಣಾಙ್ಕ>(selected_backend));
    }

    ಯದಿ (selected_pool) {
        ಪ್ರತ್ಯಾಗಮ *selected_pool;
    }

    ಕ್ಷಿಪ std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    ಸ್ಥಿರ std::once_flag                             shared_mem_pool_once;
    ಸ್ಥಿರ std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    ಯದಿ (shared_mem_pool) {
        ಪ್ರತ್ಯಾಗಮ *shared_mem_pool;
    }

    ಕ್ಷಿಪ std::bad_alloc();
}

}  // namespace

ತಾರ್ಕಿಕ spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    ಯದಿ (info == nullptr) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    *info = {};

    ಯದಿ (spine_tcm_open_handle(ರಿಕ್ತ) != 0 || !spine_tcm_is_available()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    spine_tcm_mem_info_t mem_info;
    ಯದಿ (spine_tcm_mem_info(&mem_info) != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    info->available   = ಸತ್ಯ;
    info->blk_size    = mem_info.blk_size;
    info->blk_ಅಂಕೆ     = mem_info.blk_ಅಂಕೆ;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಶೂನ್ಯ * spine_mem_pool_tcm_mem_get(ಪೂರ್ಣಾಙ್ಕ cpu_id) noexcept {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_mem_get(cpu_id);
}

ಶೂನ್ಯ * spine_mem_pool_tcm_mem_wait(ಪೂರ್ಣಾಙ್ಕ cpu_id) noexcept {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

ಪೂರ್ಣಾಙ್ಕ spine_mem_pool_tcm_mem_release(ಪೂರ್ಣಾಙ್ಕ cpu_id) noexcept {
    ಪ್ರತ್ಯಾಗಮ spine_tcm_mem_release(cpu_id);
}

ಶೂನ್ಯ * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    ಯತ್ನ {
        ಪ್ರತ್ಯಾಗಮ get_spine_mem_pool_manager().alloc(size, alignment);
    } ಗೃಹಾಣ (ಧ್ರುವ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
}

ಶೂನ್ಯ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    ಯತ್ನ {
        ಪ್ರತ್ಯಾಗಮ get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } ಗೃಹಾಣ (ಧ್ರುವ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
}

ಶೂನ್ಯ spine_mem_pool_free(ಶೂನ್ಯ * base) noexcept {
    ಯತ್ನ {
        get_spine_mem_pool_manager().ಮುಕ್ತ(base);
    } ಗೃಹಾಣ (ಧ್ರುವ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

ಶೂನ್ಯ spine_mem_pool_shared_mem_free(ಶೂನ್ಯ * base) noexcept {
    ಯತ್ನ {
        get_spine_mem_pool_shared_mem_manager().ಮುಕ್ತ(base);
    } ಗೃಹಾಣ (ಧ್ರುವ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

ಬಾಹ್ಯ "C" {
ಶೂನ್ಯ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    ಶೂನ್ಯ * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    ಯದಿ (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    ಪ್ರತ್ಯಾಗಮ result;
}

ಶೂನ್ಯ ggml_backend_cpu_riscv64_spacemit_free_shared(ಶೂನ್ಯ * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
