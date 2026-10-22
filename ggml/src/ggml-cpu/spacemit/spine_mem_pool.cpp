#അന്തര്ഭാവ "spine_mem_pool.h"

#അന്തര്ഭാവ "common.h"
#അന്തര്ഭാവ "ime_env.h"
#അന്തര്ഭാവ "spine_tcm.h"

#അന്തര്ഭാവ <fcntl.h>
#അന്തര്ഭാവ <sys/ioctl.h>
#അന്തര്ഭാവ <sys/mman.h>
#അന്തര്ഭാവ <unistd.h>

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cerrno>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <limits>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <mutex>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <vector>

നാമാകാശ ggml::cpu::riscv64_spacemit {
നാമാകാശ {

constexpr size_t   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr അക്ഷരം     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr അക്ഷരം     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

ഘടന hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, ഘടന hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

ഘടന free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

ഘടന pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    പൂർണ്ണസംഖ്യ                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

ഘടന pool_allocation {
    ശൂന്യം * chunk_base{ nullptr };
    size_t chunk_size{ 0 };
    ശൂന്യം * base{ nullptr };
    size_t size{ 0 };
};

തർക്കം is_power_of_two(size_t value) {
    തിരികെ value != 0 && (value & (value - 1)) == 0;
}

തർക്കം align_up(size_t value, size_t alignment, size_t * aligned_മൂല്യം) {
    എങ്കിൽ (aligned_മൂല്യം == nullptr || alignment == 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം size_t remainder = value % alignment;
    എങ്കിൽ (remainder == 0) {
        *aligned_മൂല്യം = value;
        തിരികെ സത്യം;
    }

    സ്ഥിരം size_t padding = alignment - remainder;
    എങ്കിൽ (value > std::numeric_limits<size_t>::max() - padding) {
        തിരികെ അസത്യം;
    }

    *aligned_മൂല്യം = value + padding;
    തിരികെ സത്യം;
}

തർക്കം align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_മൂല്യം) {
    എങ്കിൽ (aligned_മൂല്യം == nullptr || alignment == 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uintptr_t remainder = value % alignment;
    എങ്കിൽ (remainder == 0) {
        *aligned_മൂല്യം = value;
        തിരികെ സത്യം;
    }

    സ്ഥിരം uintptr_t padding = alignment - remainder;
    എങ്കിൽ (value > std::numeric_limits<uintptr_t>::max() - padding) {
        തിരികെ അസത്യം;
    }

    *aligned_മൂല്യം = value + padding;
    തിരികെ സത്യം;
}

വർഗ്ഗം spine_mem_pool_manager {
  പൊതു:
    explicit spine_mem_pool_manager(size_t default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    മായാ ~spine_mem_pool_manager() = സ്ഥിരസ്ഥിതി;

    ശൂന്യം * alloc(size_t size, size_t alignment) {
        എങ്കിൽ (size == 0 || !is_power_of_two(alignment)) {
            തിരികെ nullptr;
        }

        size_t aligned_size = 0;
        എങ്കിൽ (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            തിരികെ nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        എങ്കിൽ (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            എങ്കിൽ (!add_chunk_locked(aligned_size, alignment)) {
                തിരികെ nullptr;
            }

            എങ്കിൽ (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                തിരികെ nullptr;
            }
        }

        ശ്രമിക്കുക {
            സ്ഥിരം auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            എങ്കിൽ (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                തിരികെ nullptr;
            }
        } പിടിക്കുക (സ്ഥിരം std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            എറിയുക;
        }

        തിരികെ allocation.base;
    }

    ശൂന്യം സ്വതന്ത്രം(ശൂന്യം * base) {
        എങ്കിൽ (base == nullptr) {
            തിരികെ;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        എങ്കിൽ (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            തിരികെ;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        എങ്കിൽ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            തിരികെ;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        എങ്കിൽ (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            തിരികെ;
        }

        സ്ഥിരം size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        എങ്കിൽ (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            തിരികെ;
        }

        തിരുകുക_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  സംരക്ഷിത:
    ശൂന്യം release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        വേണ്ടി (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_size() സ്ഥിരം { തിരികെ default_chunk_size_; }

    സ്ഥിതം ശൂന്യം clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    മായാ തർക്കം alloc_chunk(size_t min_size, size_t alignment, ശൂന്യം * hint_addr, pool_chunk * chunk) = 0;
    മായാ ശൂന്യം dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  സ്വകാര്യ:
    ഘടന alloc_candidate {
        size_t    chunk_index{ 0 };
        size_t    block_index{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        തർക്കം      valid{ അസത്യം };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(സ്ഥിരം pool_allocation & allocation) {
        തിരികെ std::find_if(chunks_.begin(), chunks_.end(), [&](സ്ഥിരം pool_chunk & chunk) {
            തിരികെ chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    തർക്കം add_chunk_locked(size_t min_size, size_t alignment) {
        pool_chunk   chunk;
        സ്ഥിരം size_t chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        ശൂന്യം *       hint_addr     = nullptr;

        വേണ്ടി (സ്ഥിരം auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            എങ്കിൽ (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        എങ്കിൽ (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            തിരികെ അസത്യം;
        }

        ശ്രമിക്കുക {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } പിടിക്കുക (സ്ഥിരം std::bad_alloc &) {
            dealloc_chunk(&chunk);
            എറിയുക;
        }

        തിരികെ സത്യം;
    }

    ശൂന്യം rollback_allocation_locked(സ്ഥിരം pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        എങ്കിൽ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            തിരികെ;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        എങ്കിൽ (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            തിരികെ;
        }

        സ്ഥിരം size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        എങ്കിൽ (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            തിരികെ;
        }

        തിരുകുക_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    തർക്കം try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        വേണ്ടി (size_t chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            സ്ഥിരം auto & chunk = chunks_[chunk_index];
            വേണ്ടി (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                സ്ഥിരം auto & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                സ്ഥിരം auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                എങ്കിൽ (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    തുടരുക;
                }

                എങ്കിൽ (aligned_addr < block_addr) {
                    തുടരുക;
                }

                സ്ഥിരം size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                സ്ഥിരം size_t padding        = aligned_offset - block.offset;
                എങ്കിൽ (padding > block.size || size > block.size - padding) {
                    തുടരുക;
                }

                എങ്കിൽ (!best.valid || aligned_addr < best.address) {
                    best.chunk_index    = chunk_index;
                    best.block_index    = block_index;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = സത്യം;
                }
            }
        }

        എങ്കിൽ (!best.valid) {
            തിരികെ അസത്യം;
        }

        auto &           chunk     = chunks_[best.chunk_index];
        സ്ഥിരം free_block block     = chunk.free_blocks[best.block_index];
        സ്ഥിരം size_t     padding   = best.aligned_offset - block.offset;
        സ്ഥിരം size_t     alloc_end = best.aligned_offset + size;
        സ്ഥിരം size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        auto തിരുകുക_it = chunk.free_blocks.begin() + best.block_index;
        എങ്കിൽ (padding != 0) {
            തിരുകുക_it = chunk.free_blocks.insert(തിരുകുക_it, { block.offset, padding });
            ++തിരുകുക_it;
        }
        എങ്കിൽ (alloc_end < block_end) {
            chunk.free_blocks.insert(തിരുകുക_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        തിരികെ സത്യം;
    }

    ശൂന്യം maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        എങ്കിൽ (chunk_it->free_blocks.size() != 1) {
            തിരികെ;
        }

        സ്ഥിരം auto & block = chunk_it->free_blocks.front();
        എങ്കിൽ (block.offset != 0 || block.size != chunk_it->size) {
            തിരികെ;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    ശൂന്യം തിരുകുക_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        ആകുമ്പോൾ (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        എങ്കിൽ (it != chunk.free_blocks.begin()) {
            സ്ഥിരം auto & prev = *(it - 1);
            എങ്കിൽ (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                തിരികെ;
            }
        }

        എങ്കിൽ (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            തിരികെ;
        }

        it = chunk.free_blocks.insert(it, block);

        എങ്കിൽ (it != chunk.free_blocks.begin()) {
            auto prev = it - 1;
            എങ്കിൽ (prev->offset + prev->size == it->offset) {
                it->offset = prev->offset;
                it->size += prev->size;
                it = chunk.free_blocks.erase(prev);
            }
        }

        എങ്കിൽ (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<ശൂന്യം *, pool_allocation> allocations_;
    size_t                                      default_chunk_size_{ 0 };
};

വർഗ്ഗം spine_mem_pool_posix final : പൊതു spine_mem_pool_manager {
  പൊതു:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  സ്വകാര്യ:
    തർക്കം alloc_chunk(size_t min_size, size_t alignment, ശൂന്യം * hint_addr, pool_chunk * chunk) override {
        (ശൂന്യം) hint_addr;

        സ്ഥിരം size_t alloc_alignment = std::max(alignment, വലുപ്പം(ശൂന്യം *));
        ശൂന്യം *       base            = nullptr;
        സ്ഥിരം പൂർണ്ണസംഖ്യ    rc              = posix_memalign(&base, alloc_alignment, min_size);
        എങ്കിൽ (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            തിരികെ അസത്യം;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        തിരികെ സത്യം;
    }

    ശൂന്യം dealloc_chunk(pool_chunk * chunk) override {
        std::സ്വതന്ത്രം(chunk->base);
        clear_chunk(chunk);
    }
};

വർഗ്ഗം spine_mem_pool_transparent_hugepage final : പൊതു spine_mem_pool_manager {
  പൊതു:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  സ്വകാര്യ:
    തർക്കം alloc_chunk(size_t min_size, size_t alignment, ശൂന്യം * hint_addr, pool_chunk * chunk) override {
        (ശൂന്യം) alignment;

        size_t chunk_size = 0;
        എങ്കിൽ (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            തിരികെ അസത്യം;
        }

        ശൂന്യം * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        എങ്കിൽ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           errno);
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, errno);
            munmap(map_addr, chunk_size);
            തിരികെ അസത്യം;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        തിരികെ സത്യം;
    }

    ശൂന്യം dealloc_chunk(pool_chunk * chunk) override {
        എങ്കിൽ (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

വർഗ്ഗം spine_mem_pool_hugetlb_1g final : പൊതു spine_mem_pool_manager {
  പൊതു:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  സ്വകാര്യ:
    തർക്കം alloc_chunk(size_t min_size, size_t alignment, ശൂന്യം * hint_addr, pool_chunk * chunk) override {
        (ശൂന്യം) alignment;
        (ശൂന്യം) hint_addr;

        size_t region_size = 0;
        എങ്കിൽ (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            തിരികെ അസത്യം;
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        എങ്കിൽ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            തിരികെ അസത്യം;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        എങ്കിൽ (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, errno);
            close(fd);
            തിരികെ അസത്യം;
        }

        ശൂന്യം * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        എങ്കിൽ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            തിരികെ അസത്യം;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        തിരികെ സത്യം;
    }

    ശൂന്യം dealloc_chunk(pool_chunk * chunk) override {
        എങ്കിൽ (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        എങ്കിൽ (chunk->fd >= 0) {
            എങ്കിൽ (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

വർഗ്ഗം spine_mem_pool_shared_mem final : പൊതു spine_mem_pool_manager {
  പൊതു:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  സ്വകാര്യ:
    തർക്കം alloc_chunk(size_t min_size, size_t alignment, ശൂന്യം * hint_addr, pool_chunk * chunk) override {
        (ശൂന്യം) alignment;

        എങ്കിൽ (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            തിരികെ അസത്യം;
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        എങ്കിൽ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            തിരികെ അസത്യം;
        }

        ശൂന്യം * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        എങ്കിൽ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), errno);
            close(fd);
            തിരികെ അസത്യം;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        തിരികെ സത്യം;
    }

    ശൂന്യം dealloc_chunk(pool_chunk * chunk) override {
        എങ്കിൽ (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        എങ്കിൽ (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    സ്ഥിതം std::once_flag                          pool_once;
    സ്ഥിതം std::unique_ptr<spine_mem_pool_manager> selected_pool;
    സ്ഥിതം spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    എങ്കിൽ (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        തിരഞ്ഞെടുപ്പ് (selected_backend) {
            അവസ്ഥ spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                നിർത്തുക;
            അവസ്ഥ spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                നിർത്തുക;
            അവസ്ഥ spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                നിർത്തുക;
            അവസ്ഥ spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                നിർത്തുക;
        }
    });

    എങ്കിൽ (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<പൂർണ്ണസംഖ്യ>(backend), static_cast<പൂർണ്ണസംഖ്യ>(selected_backend));
    }

    എങ്കിൽ (selected_pool) {
        തിരികെ *selected_pool;
    }

    എറിയുക std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    സ്ഥിതം std::once_flag                             shared_mem_pool_once;
    സ്ഥിതം std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    എങ്കിൽ (shared_mem_pool) {
        തിരികെ *shared_mem_pool;
    }

    എറിയുക std::bad_alloc();
}

}  // namespace

തർക്കം spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    എങ്കിൽ (info == nullptr) {
        തിരികെ അസത്യം;
    }

    *info = {};

    എങ്കിൽ (spine_tcm_open_handle(ശൂന്യ) != 0 || !spine_tcm_is_available()) {
        തിരികെ അസത്യം;
    }

    spine_tcm_mem_info_t mem_info;
    എങ്കിൽ (spine_tcm_mem_info(&mem_info) != 0) {
        തിരികെ അസത്യം;
    }

    info->available   = സത്യം;
    info->blk_size    = mem_info.blk_size;
    info->blk_അക്കം     = mem_info.blk_അക്കം;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    തിരികെ സത്യം;
}

ശൂന്യം * spine_mem_pool_tcm_mem_get(പൂർണ്ണസംഖ്യ cpu_id) noexcept {
    തിരികെ spine_tcm_mem_get(cpu_id);
}

ശൂന്യം * spine_mem_pool_tcm_mem_wait(പൂർണ്ണസംഖ്യ cpu_id) noexcept {
    തിരികെ spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

പൂർണ്ണസംഖ്യ spine_mem_pool_tcm_mem_release(പൂർണ്ണസംഖ്യ cpu_id) noexcept {
    തിരികെ spine_tcm_mem_release(cpu_id);
}

ശൂന്യം * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    ശ്രമിക്കുക {
        തിരികെ get_spine_mem_pool_manager().alloc(size, alignment);
    } പിടിക്കുക (സ്ഥിരം std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        തിരികെ nullptr;
    }
}

ശൂന്യം * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    ശ്രമിക്കുക {
        തിരികെ get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } പിടിക്കുക (സ്ഥിരം std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        തിരികെ nullptr;
    }
}

ശൂന്യം spine_mem_pool_free(ശൂന്യം * base) noexcept {
    ശ്രമിക്കുക {
        get_spine_mem_pool_manager().സ്വതന്ത്രം(base);
    } പിടിക്കുക (സ്ഥിരം std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

ശൂന്യം spine_mem_pool_shared_mem_free(ശൂന്യം * base) noexcept {
    ശ്രമിക്കുക {
        get_spine_mem_pool_shared_mem_manager().സ്വതന്ത്രം(base);
    } പിടിക്കുക (സ്ഥിരം std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

ബാഹ്യ "C" {
ശൂന്യം * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    ശൂന്യം * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    എങ്കിൽ (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    തിരികെ result;
}

ശൂന്യം ggml_backend_cpu_riscv64_spacemit_free_shared(ശൂന്യം * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
