#ଅନ୍ତର୍ଭାଵ "spine_mem_pool.h"

#ଅନ୍ତର୍ଭାଵ "common.h"
#ଅନ୍ତର୍ଭାଵ "ime_env.h"
#ଅନ୍ତର୍ଭାଵ "spine_tcm.h"

#ଅନ୍ତର୍ଭାଵ <fcntl.h>
#ଅନ୍ତର୍ଭାଵ <sys/ioctl.h>
#ଅନ୍ତର୍ଭାଵ <sys/mman.h>
#ଅନ୍ତର୍ଭାଵ <unistd.h>

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cerrno>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <limits>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <mutex>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ggml::cpu::riscv64_spacemit {
ନାମାକାଶ {

constexpr size_t   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr ଅକ୍ଷର     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr ଅକ୍ଷର     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

ସଂରଚନା hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, ସଂରଚନା hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

ସଂରଚନା free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

ସଂରଚନା pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

ସଂରଚନା pool_allocation {
    ଶୂନ୍ୟ * chunk_base{ nullptr };
    size_t chunk_size{ 0 };
    ଶୂନ୍ୟ * base{ nullptr };
    size_t size{ 0 };
};

ତାର୍କିକ is_power_of_two(size_t value) {
    ଫେରନ୍ତୁ value != 0 && (value & (value - 1)) == 0;
}

ତାର୍କିକ align_up(size_t value, size_t alignment, size_t * aligned_ମୂଲ୍ୟ) {
    ଯଦି (aligned_ମୂଲ୍ୟ == nullptr || alignment == 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ size_t remainder = value % alignment;
    ଯଦି (remainder == 0) {
        *aligned_ମୂଲ୍ୟ = value;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଅଚଳ size_t padding = alignment - remainder;
    ଯଦି (value > std::numeric_limits<size_t>::max() - padding) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    *aligned_ମୂଲ୍ୟ = value + padding;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_ମୂଲ୍ୟ) {
    ଯଦି (aligned_ମୂଲ୍ୟ == nullptr || alignment == 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uintptr_t remainder = value % alignment;
    ଯଦି (remainder == 0) {
        *aligned_ମୂଲ୍ୟ = value;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଅଚଳ uintptr_t padding = alignment - remainder;
    ଯଦି (value > std::numeric_limits<uintptr_t>::max() - padding) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    *aligned_ମୂଲ୍ୟ = value + padding;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ଶ୍ରେଣୀ spine_mem_pool_manager {
  ସର୍ବସାଧାରଣ:
    explicit spine_mem_pool_manager(size_t default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    ଆଭାସୀ ~spine_mem_pool_manager() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;

    ଶୂନ୍ୟ * alloc(size_t size, size_t alignment) {
        ଯଦି (size == 0 || !is_power_of_two(alignment)) {
            ଫେରନ୍ତୁ nullptr;
        }

        size_t aligned_size = 0;
        ଯଦି (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            ଫେରନ୍ତୁ nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        ଯଦି (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            ଯଦି (!add_chunk_locked(aligned_size, alignment)) {
                ଫେରନ୍ତୁ nullptr;
            }

            ଯଦି (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                ଫେରନ୍ତୁ nullptr;
            }
        }

        ଚେଷ୍ଟା {
            ଅଚଳ auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            ଯଦି (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                ଫେରନ୍ତୁ nullptr;
            }
        } ଧର (ଅଚଳ std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            ଫିଙ୍ଗ;
        }

        ଫେରନ୍ତୁ allocation.base;
    }

    ଶୂନ୍ୟ ମୁକ୍ତ(ଶୂନ୍ୟ * base) {
        ଯଦି (base == nullptr) {
            ଫେରନ୍ତୁ;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        ଯଦି (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            ଫେରନ୍ତୁ;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        ଯଦି (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ଫେରନ୍ତୁ;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        ଯଦି (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            ଫେରନ୍ତୁ;
        }

        ଅଚଳ size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        ଯଦି (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            ଫେରନ୍ତୁ;
        }

        ଭର୍ତ୍ତି_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  ସୁରକ୍ଷିତ:
    ଶୂନ୍ୟ release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        ପାଇଁ (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_size() ଅଚଳ { ଫେରନ୍ତୁ default_chunk_size_; }

    ସ୍ଥିର ଶୂନ୍ୟ clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    ଆଭାସୀ ତାର୍କିକ alloc_chunk(size_t min_size, size_t alignment, ଶୂନ୍ୟ * hint_addr, pool_chunk * chunk) = 0;
    ଆଭାସୀ ଶୂନ୍ୟ dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  ବ୍ୟକ୍ତିଗତ:
    ସଂରଚନା alloc_candidate {
        size_t    chunk_index{ 0 };
        size_t    block_index{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        ତାର୍କିକ      valid{ ମିଥ୍ୟା };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(ଅଚଳ pool_allocation & allocation) {
        ଫେରନ୍ତୁ std::find_if(chunks_.begin(), chunks_.end(), [&](ଅଚଳ pool_chunk & chunk) {
            ଫେରନ୍ତୁ chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    ତାର୍କିକ add_chunk_locked(size_t min_size, size_t alignment) {
        pool_chunk   chunk;
        ଅଚଳ size_t chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        ଶୂନ୍ୟ *       hint_addr     = nullptr;

        ପାଇଁ (ଅଚଳ auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            ଯଦି (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        ଯଦି (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଚେଷ୍ଟା {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } ଧର (ଅଚଳ std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ଫିଙ୍ଗ;
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଶୂନ୍ୟ rollback_allocation_locked(ଅଚଳ pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        ଯଦି (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            ଫେରନ୍ତୁ;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        ଯଦି (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            ଫେରନ୍ତୁ;
        }

        ଅଚଳ size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        ଯଦି (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ଫେରନ୍ତୁ;
        }

        ଭର୍ତ୍ତି_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    ତାର୍କିକ try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        ପାଇଁ (size_t chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            ଅଚଳ auto & chunk = chunks_[chunk_index];
            ପାଇଁ (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                ଅଚଳ auto & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                ଅଚଳ auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                ଯଦି (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    ଜାରି;
                }

                ଯଦି (aligned_addr < block_addr) {
                    ଜାରି;
                }

                ଅଚଳ size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                ଅଚଳ size_t padding        = aligned_offset - block.offset;
                ଯଦି (padding > block.size || size > block.size - padding) {
                    ଜାରି;
                }

                ଯଦି (!best.valid || aligned_addr < best.address) {
                    best.chunk_index    = chunk_index;
                    best.block_index    = block_index;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = ସତ୍ୟ;
                }
            }
        }

        ଯଦି (!best.valid) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        auto &           chunk     = chunks_[best.chunk_index];
        ଅଚଳ free_block block     = chunk.free_blocks[best.block_index];
        ଅଚଳ size_t     padding   = best.aligned_offset - block.offset;
        ଅଚଳ size_t     alloc_end = best.aligned_offset + size;
        ଅଚଳ size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        auto ଭର୍ତ୍ତି_it = chunk.free_blocks.begin() + best.block_index;
        ଯଦି (padding != 0) {
            ଭର୍ତ୍ତି_it = chunk.free_blocks.insert(ଭର୍ତ୍ତି_it, { block.offset, padding });
            ++ଭର୍ତ୍ତି_it;
        }
        ଯଦି (alloc_end < block_end) {
            chunk.free_blocks.insert(ଭର୍ତ୍ତି_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଶୂନ୍ୟ maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        ଯଦି (chunk_it->free_blocks.size() != 1) {
            ଫେରନ୍ତୁ;
        }

        ଅଚଳ auto & block = chunk_it->free_blocks.front();
        ଯଦି (block.offset != 0 || block.size != chunk_it->size) {
            ଫେରନ୍ତୁ;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    ଶୂନ୍ୟ ଭର୍ତ୍ତି_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        ଯେତେବେଳେ (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        ଯଦି (it != chunk.free_blocks.begin()) {
            ଅଚଳ auto & prev = *(it - 1);
            ଯଦି (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                ଫେରନ୍ତୁ;
            }
        }

        ଯଦି (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            ଫେରନ୍ତୁ;
        }

        it = chunk.free_blocks.insert(it, block);

        ଯଦି (it != chunk.free_blocks.begin()) {
            auto prev = it - 1;
            ଯଦି (prev->offset + prev->size == it->offset) {
                it->offset = prev->offset;
                it->size += prev->size;
                it = chunk.free_blocks.erase(prev);
            }
        }

        ଯଦି (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<ଶୂନ୍ୟ *, pool_allocation> allocations_;
    size_t                                      default_chunk_size_{ 0 };
};

ଶ୍ରେଣୀ spine_mem_pool_posix final : ସର୍ବସାଧାରଣ spine_mem_pool_manager {
  ସର୍ବସାଧାରଣ:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  ବ୍ୟକ୍ତିଗତ:
    ତାର୍କିକ alloc_chunk(size_t min_size, size_t alignment, ଶୂନ୍ୟ * hint_addr, pool_chunk * chunk) override {
        (ଶୂନ୍ୟ) hint_addr;

        ଅଚଳ size_t alloc_alignment = std::max(alignment, ଆକାର(ଶୂନ୍ୟ *));
        ଶୂନ୍ୟ *       base            = nullptr;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    rc              = posix_memalign(&base, alloc_alignment, min_size);
        ଯଦି (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଶୂନ୍ୟ dealloc_chunk(pool_chunk * chunk) override {
        std::ମୁକ୍ତ(chunk->base);
        clear_chunk(chunk);
    }
};

ଶ୍ରେଣୀ spine_mem_pool_transparent_hugepage final : ସର୍ବସାଧାରଣ spine_mem_pool_manager {
  ସର୍ବସାଧାରଣ:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  ବ୍ୟକ୍ତିଗତ:
    ତାର୍କିକ alloc_chunk(size_t min_size, size_t alignment, ଶୂନ୍ୟ * hint_addr, pool_chunk * chunk) override {
        (ଶୂନ୍ୟ) alignment;

        size_t chunk_size = 0;
        ଯଦି (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଶୂନ୍ୟ * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ଯଦି (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           errno);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, errno);
            munmap(map_addr, chunk_size);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଶୂନ୍ୟ dealloc_chunk(pool_chunk * chunk) override {
        ଯଦି (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

ଶ୍ରେଣୀ spine_mem_pool_hugetlb_1g final : ସର୍ବସାଧାରଣ spine_mem_pool_manager {
  ସର୍ବସାଧାରଣ:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  ବ୍ୟକ୍ତିଗତ:
    ତାର୍କିକ alloc_chunk(size_t min_size, size_t alignment, ଶୂନ୍ୟ * hint_addr, pool_chunk * chunk) override {
        (ଶୂନ୍ୟ) alignment;
        (ଶୂନ୍ୟ) hint_addr;

        size_t region_size = 0;
        ଯଦି (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        ଯଦି (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        ଯଦି (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, errno);
            close(fd);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଶୂନ୍ୟ * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ଯଦି (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଶୂନ୍ୟ dealloc_chunk(pool_chunk * chunk) override {
        ଯଦି (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        ଯଦି (chunk->fd >= 0) {
            ଯଦି (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

ଶ୍ରେଣୀ spine_mem_pool_shared_mem final : ସର୍ବସାଧାରଣ spine_mem_pool_manager {
  ସର୍ବସାଧାରଣ:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  ବ୍ୟକ୍ତିଗତ:
    ତାର୍କିକ alloc_chunk(size_t min_size, size_t alignment, ଶୂନ୍ୟ * hint_addr, pool_chunk * chunk) override {
        (ଶୂନ୍ୟ) alignment;

        ଯଦି (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        ଯଦି (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଶୂନ୍ୟ * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ଯଦି (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), errno);
            close(fd);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଶୂନ୍ୟ dealloc_chunk(pool_chunk * chunk) override {
        ଯଦି (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        ଯଦି (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    ସ୍ଥିର std::once_flag                          pool_once;
    ସ୍ଥିର std::unique_ptr<spine_mem_pool_manager> selected_pool;
    ସ୍ଥିର spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    ଯଦି (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        ବାଛ (selected_backend) {
            ସ୍ଥିତି spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                ଭାଙ୍ଗ;
            ସ୍ଥିତି spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ଭାଙ୍ଗ;
            ସ୍ଥିତି spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                ଭାଙ୍ଗ;
            ସ୍ଥିତି spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ଭାଙ୍ଗ;
        }
    });

    ଯଦି (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(backend), static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(selected_backend));
    }

    ଯଦି (selected_pool) {
        ଫେରନ୍ତୁ *selected_pool;
    }

    ଫିଙ୍ଗ std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    ସ୍ଥିର std::once_flag                             shared_mem_pool_once;
    ସ୍ଥିର std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    ଯଦି (shared_mem_pool) {
        ଫେରନ୍ତୁ *shared_mem_pool;
    }

    ଫିଙ୍ଗ std::bad_alloc();
}

}  // namespace

ତାର୍କିକ spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    ଯଦି (info == nullptr) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    *info = {};

    ଯଦି (spine_tcm_open_handle(ଖାଲି) != 0 || !spine_tcm_is_available()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    spine_tcm_mem_info_t mem_info;
    ଯଦି (spine_tcm_mem_info(&mem_info) != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    info->available   = ସତ୍ୟ;
    info->blk_size    = mem_info.blk_size;
    info->blk_ଅଙ୍କ     = mem_info.blk_ଅଙ୍କ;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ଶୂନ୍ୟ * spine_mem_pool_tcm_mem_get(ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_id) noexcept {
    ଫେରନ୍ତୁ spine_tcm_mem_get(cpu_id);
}

ଶୂନ୍ୟ * spine_mem_pool_tcm_mem_wait(ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_id) noexcept {
    ଫେରନ୍ତୁ spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା spine_mem_pool_tcm_mem_release(ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_id) noexcept {
    ଫେରନ୍ତୁ spine_tcm_mem_release(cpu_id);
}

ଶୂନ୍ୟ * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    ଚେଷ୍ଟା {
        ଫେରନ୍ତୁ get_spine_mem_pool_manager().alloc(size, alignment);
    } ଧର (ଅଚଳ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        ଫେରନ୍ତୁ nullptr;
    }
}

ଶୂନ୍ୟ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    ଚେଷ୍ଟା {
        ଫେରନ୍ତୁ get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } ଧର (ଅଚଳ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        ଫେରନ୍ତୁ nullptr;
    }
}

ଶୂନ୍ୟ spine_mem_pool_free(ଶୂନ୍ୟ * base) noexcept {
    ଚେଷ୍ଟା {
        get_spine_mem_pool_manager().ମୁକ୍ତ(base);
    } ଧର (ଅଚଳ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

ଶୂନ୍ୟ spine_mem_pool_shared_mem_free(ଶୂନ୍ୟ * base) noexcept {
    ଚେଷ୍ଟା {
        get_spine_mem_pool_shared_mem_manager().ମୁକ୍ତ(base);
    } ଧର (ଅଚଳ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

ବାହ୍ୟ "C" {
ଶୂନ୍ୟ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    ଶୂନ୍ୟ * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    ଯଦି (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    ଫେରନ୍ତୁ result;
}

ଶୂନ୍ୟ ggml_backend_cpu_riscv64_spacemit_free_shared(ଶୂନ୍ୟ * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
