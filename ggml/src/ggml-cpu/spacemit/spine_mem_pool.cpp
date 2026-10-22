#অন্তর্ভাৱ "spine_mem_pool.h"

#অন্তর্ভাৱ "common.h"
#অন্তর্ভাৱ "ime_env.h"
#অন্তর্ভাৱ "spine_tcm.h"

#অন্তর্ভাৱ <fcntl.h>
#অন্তর্ভাৱ <sys/ioctl.h>
#অন্তর্ভাৱ <sys/mman.h>
#অন্তর্ভাৱ <unistd.h>

#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <cerrno>
#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <cstdlib>
#অন্তর্ভাৱ <limits>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <mutex>
#অন্তর্ভাৱ <unordered_map>
#অন্তর্ভাৱ <vector>

নামাকাশ ggml::cpu::riscv64_spacemit {
নামাকাশ {

constexpr size_t   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr ৱর্ণ     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr ৱর্ণ     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

সংরচনা hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, সংরচনা hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

সংরচনা free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

সংরচনা pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    পূর্ণাঙ্ক                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

সংরচনা pool_allocation {
    শূন্য * chunk_base{ nullptr };
    size_t chunk_size{ 0 };
    শূন্য * base{ nullptr };
    size_t size{ 0 };
};

তার্কিক is_power_of_two(size_t value) {
    ঘূৰাই value != 0 && (value & (value - 1)) == 0;
}

তার্কিক align_up(size_t value, size_t alignment, size_t * aligned_মান) {
    যদি (aligned_মান == nullptr || alignment == 0) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ size_t remainder = value % alignment;
    যদি (remainder == 0) {
        *aligned_মান = value;
        ঘূৰাই সঁচা;
    }

    ধ্রুৱ size_t padding = alignment - remainder;
    যদি (value > std::numeric_limits<size_t>::max() - padding) {
        ঘূৰাই মিছা;
    }

    *aligned_মান = value + padding;
    ঘূৰাই সঁচা;
}

তার্কিক align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_মান) {
    যদি (aligned_মান == nullptr || alignment == 0) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ uintptr_t remainder = value % alignment;
    যদি (remainder == 0) {
        *aligned_মান = value;
        ঘূৰাই সঁচা;
    }

    ধ্রুৱ uintptr_t padding = alignment - remainder;
    যদি (value > std::numeric_limits<uintptr_t>::max() - padding) {
        ঘূৰাই মিছা;
    }

    *aligned_মান = value + padding;
    ঘূৰাই সঁচা;
}

শ্ৰেণী spine_mem_pool_manager {
  সার্ৱজনিক:
    explicit spine_mem_pool_manager(size_t default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    আভাস ~spine_mem_pool_manager() = স্ৱভাৱ;

    শূন্য * alloc(size_t size, size_t alignment) {
        যদি (size == 0 || !is_power_of_two(alignment)) {
            ঘূৰাই nullptr;
        }

        size_t aligned_size = 0;
        যদি (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            ঘূৰাই nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        যদি (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            যদি (!add_chunk_locked(aligned_size, alignment)) {
                ঘূৰাই nullptr;
            }

            যদি (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                ঘূৰাই nullptr;
            }
        }

        যত্ন {
            ধ্রুৱ auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            যদি (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                ঘূৰাই nullptr;
            }
        } গৃহাণ (ধ্রুৱ std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            ক্ষিপ;
        }

        ঘূৰাই allocation.base;
    }

    শূন্য মুক্ত(শূন্য * base) {
        যদি (base == nullptr) {
            ঘূৰাই;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        যদি (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            ঘূৰাই;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        যদি (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ঘূৰাই;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        যদি (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            ঘূৰাই;
        }

        ধ্রুৱ size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        যদি (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            ঘূৰাই;
        }

        ঢোকাও_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  রক্ষিত:
    শূন্য release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        বাবে (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_size() ধ্রুৱ { ঘূৰাই default_chunk_size_; }

    স্থির শূন্য clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    আভাস তার্কিক alloc_chunk(size_t min_size, size_t alignment, শূন্য * hint_addr, pool_chunk * chunk) = 0;
    আভাস শূন্য dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  গুপ্ত:
    সংরচনা alloc_candidate {
        size_t    chunk_index{ 0 };
        size_t    block_index{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        তার্কিক      valid{ মিছা };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(ধ্রুৱ pool_allocation & allocation) {
        ঘূৰাই std::find_if(chunks_.begin(), chunks_.end(), [&](ধ্রুৱ pool_chunk & chunk) {
            ঘূৰাই chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    তার্কিক add_chunk_locked(size_t min_size, size_t alignment) {
        pool_chunk   chunk;
        ধ্রুৱ size_t chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        শূন্য *       hint_addr     = nullptr;

        বাবে (ধ্রুৱ auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            যদি (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        যদি (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            ঘূৰাই মিছা;
        }

        যদি (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            ঘূৰাই মিছা;
        }

        যত্ন {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } গৃহাণ (ধ্রুৱ std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ক্ষিপ;
        }

        ঘূৰাই সঁচা;
    }

    শূন্য rollback_allocation_locked(ধ্রুৱ pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        যদি (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            ঘূৰাই;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        যদি (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            ঘূৰাই;
        }

        ধ্রুৱ size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        যদি (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ঘূৰাই;
        }

        ঢোকাও_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    তার্কিক try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        বাবে (size_t chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            ধ্রুৱ auto & chunk = chunks_[chunk_index];
            বাবে (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                ধ্রুৱ auto & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                ধ্রুৱ auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                যদি (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    অনুৱর্ত;
                }

                যদি (aligned_addr < block_addr) {
                    অনুৱর্ত;
                }

                ধ্রুৱ size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                ধ্রুৱ size_t padding        = aligned_offset - block.offset;
                যদি (padding > block.size || size > block.size - padding) {
                    অনুৱর্ত;
                }

                যদি (!best.valid || aligned_addr < best.address) {
                    best.chunk_index    = chunk_index;
                    best.block_index    = block_index;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = সঁচা;
                }
            }
        }

        যদি (!best.valid) {
            ঘূৰাই মিছা;
        }

        auto &           chunk     = chunks_[best.chunk_index];
        ধ্রুৱ free_block block     = chunk.free_blocks[best.block_index];
        ধ্রুৱ size_t     padding   = best.aligned_offset - block.offset;
        ধ্রুৱ size_t     alloc_end = best.aligned_offset + size;
        ধ্রুৱ size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        auto ঢোকাও_it = chunk.free_blocks.begin() + best.block_index;
        যদি (padding != 0) {
            ঢোকাও_it = chunk.free_blocks.insert(ঢোকাও_it, { block.offset, padding });
            ++ঢোকাও_it;
        }
        যদি (alloc_end < block_end) {
            chunk.free_blocks.insert(ঢোকাও_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        ঘূৰাই সঁচা;
    }

    শূন্য maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        যদি (chunk_it->free_blocks.size() != 1) {
            ঘূৰাই;
        }

        ধ্রুৱ auto & block = chunk_it->free_blocks.front();
        যদি (block.offset != 0 || block.size != chunk_it->size) {
            ঘূৰাই;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    শূন্য ঢোকাও_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        যেতিয়ালৈ (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        যদি (it != chunk.free_blocks.begin()) {
            ধ্রুৱ auto & prev = *(it - 1);
            যদি (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                ঘূৰাই;
            }
        }

        যদি (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            ঘূৰাই;
        }

        it = chunk.free_blocks.insert(it, block);

        যদি (it != chunk.free_blocks.begin()) {
            auto prev = it - 1;
            যদি (prev->offset + prev->size == it->offset) {
                it->offset = prev->offset;
                it->size += prev->size;
                it = chunk.free_blocks.erase(prev);
            }
        }

        যদি (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<শূন্য *, pool_allocation> allocations_;
    size_t                                      default_chunk_size_{ 0 };
};

শ্ৰেণী spine_mem_pool_posix final : সার্ৱজনিক spine_mem_pool_manager {
  সার্ৱজনিক:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  গুপ্ত:
    তার্কিক alloc_chunk(size_t min_size, size_t alignment, শূন্য * hint_addr, pool_chunk * chunk) override {
        (শূন্য) hint_addr;

        ধ্রুৱ size_t alloc_alignment = std::max(alignment, পরিমাণ(শূন্য *));
        শূন্য *       base            = nullptr;
        ধ্রুৱ পূর্ণাঙ্ক    rc              = posix_memalign(&base, alloc_alignment, min_size);
        যদি (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            ঘূৰাই মিছা;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        ঘূৰাই সঁচা;
    }

    শূন্য dealloc_chunk(pool_chunk * chunk) override {
        std::মুক্ত(chunk->base);
        clear_chunk(chunk);
    }
};

শ্ৰেণী spine_mem_pool_transparent_hugepage final : সার্ৱজনিক spine_mem_pool_manager {
  সার্ৱজনিক:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  গুপ্ত:
    তার্কিক alloc_chunk(size_t min_size, size_t alignment, শূন্য * hint_addr, pool_chunk * chunk) override {
        (শূন্য) alignment;

        size_t chunk_size = 0;
        যদি (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            ঘূৰাই মিছা;
        }

        শূন্য * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        যদি (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           errno);
            ঘূৰাই মিছা;
        }

        যদি (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, errno);
            munmap(map_addr, chunk_size);
            ঘূৰাই মিছা;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        ঘূৰাই সঁচা;
    }

    শূন্য dealloc_chunk(pool_chunk * chunk) override {
        যদি (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

শ্ৰেণী spine_mem_pool_hugetlb_1g final : সার্ৱজনিক spine_mem_pool_manager {
  সার্ৱজনিক:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  গুপ্ত:
    তার্কিক alloc_chunk(size_t min_size, size_t alignment, শূন্য * hint_addr, pool_chunk * chunk) override {
        (শূন্য) alignment;
        (শূন্য) hint_addr;

        size_t region_size = 0;
        যদি (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            ঘূৰাই মিছা;
        }

        ধ্রুৱ পূর্ণাঙ্ক fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        যদি (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            ঘূৰাই মিছা;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        যদি (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, errno);
            close(fd);
            ঘূৰাই মিছা;
        }

        শূন্য * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        যদি (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            ঘূৰাই মিছা;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        ঘূৰাই সঁচা;
    }

    শূন্য dealloc_chunk(pool_chunk * chunk) override {
        যদি (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        যদি (chunk->fd >= 0) {
            যদি (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

শ্ৰেণী spine_mem_pool_shared_mem final : সার্ৱজনিক spine_mem_pool_manager {
  সার্ৱজনিক:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  গুপ্ত:
    তার্কিক alloc_chunk(size_t min_size, size_t alignment, শূন্য * hint_addr, pool_chunk * chunk) override {
        (শূন্য) alignment;

        যদি (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            ঘূৰাই মিছা;
        }

        যদি (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            ঘূৰাই মিছা;
        }

        ধ্রুৱ পূর্ণাঙ্ক fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        যদি (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            ঘূৰাই মিছা;
        }

        শূন্য * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        যদি (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), errno);
            close(fd);
            ঘূৰাই মিছা;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        ঘূৰাই সঁচা;
    }

    শূন্য dealloc_chunk(pool_chunk * chunk) override {
        যদি (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        যদি (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    স্থির std::once_flag                          pool_once;
    স্থির std::unique_ptr<spine_mem_pool_manager> selected_pool;
    স্থির spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    যদি (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        ৱিকল্প (selected_backend) {
            পক্ষ spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                ভঙ্গ;
            পক্ষ spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ভঙ্গ;
            পক্ষ spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                ভঙ্গ;
            পক্ষ spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ভঙ্গ;
        }
    });

    যদি (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<পূর্ণাঙ্ক>(backend), static_cast<পূর্ণাঙ্ক>(selected_backend));
    }

    যদি (selected_pool) {
        ঘূৰাই *selected_pool;
    }

    ক্ষিপ std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    স্থির std::once_flag                             shared_mem_pool_once;
    স্থির std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    যদি (shared_mem_pool) {
        ঘূৰাই *shared_mem_pool;
    }

    ক্ষিপ std::bad_alloc();
}

}  // namespace

তার্কিক spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    যদি (info == nullptr) {
        ঘূৰাই মিছা;
    }

    *info = {};

    যদি (spine_tcm_open_handle(রিক্ত) != 0 || !spine_tcm_is_available()) {
        ঘূৰাই মিছা;
    }

    spine_tcm_mem_info_t mem_info;
    যদি (spine_tcm_mem_info(&mem_info) != 0) {
        ঘূৰাই মিছা;
    }

    info->available   = সঁচা;
    info->blk_size    = mem_info.blk_size;
    info->blk_অঙ্ক     = mem_info.blk_অঙ্ক;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    ঘূৰাই সঁচা;
}

শূন্য * spine_mem_pool_tcm_mem_get(পূর্ণাঙ্ক cpu_id) noexcept {
    ঘূৰাই spine_tcm_mem_get(cpu_id);
}

শূন্য * spine_mem_pool_tcm_mem_wait(পূর্ণাঙ্ক cpu_id) noexcept {
    ঘূৰাই spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

পূর্ণাঙ্ক spine_mem_pool_tcm_mem_release(পূর্ণাঙ্ক cpu_id) noexcept {
    ঘূৰাই spine_tcm_mem_release(cpu_id);
}

শূন্য * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    যত্ন {
        ঘূৰাই get_spine_mem_pool_manager().alloc(size, alignment);
    } গৃহাণ (ধ্রুৱ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        ঘূৰাই nullptr;
    }
}

শূন্য * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    যত্ন {
        ঘূৰাই get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } গৃহাণ (ধ্রুৱ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        ঘূৰাই nullptr;
    }
}

শূন্য spine_mem_pool_free(শূন্য * base) noexcept {
    যত্ন {
        get_spine_mem_pool_manager().মুক্ত(base);
    } গৃহাণ (ধ্রুৱ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

শূন্য spine_mem_pool_shared_mem_free(শূন্য * base) noexcept {
    যত্ন {
        get_spine_mem_pool_shared_mem_manager().মুক্ত(base);
    } গৃহাণ (ধ্রুৱ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

বাহ্য "C" {
শূন্য * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    শূন্য * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    যদি (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    ঘূৰাই result;
}

শূন্য ggml_backend_cpu_riscv64_spacemit_free_shared(শূন্য * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
