#ضمن "spine_mem_pool.h"

#ضمن "common.h"
#ضمن "ime_env.h"
#ضمن "spine_tcm.h"

#ضمن <fcntl.h>
#ضمن <sys/ioctl.h>
#ضمن <sys/mman.h>
#ضمن <unistd.h>

#ضمن <algorithm>
#ضمن <cerrno>
#ضمن <cstdint>
#ضمن <cstdlib>
#ضمن <limits>
#ضمن <memory>
#ضمن <mutex>
#ضمن <unordered_map>
#ضمن <vector>

نطاق ggml::cpu::riscv64_spacemit {
نطاق {

constexpr size_t   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr حرف     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr حرف     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

بنية hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, بنية hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

بنية free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

بنية pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    صحيح                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

بنية pool_allocation {
    فراغ * chunk_base{ nullptr };
    size_t chunk_size{ 0 };
    فراغ * base{ nullptr };
    size_t size{ 0 };
};

منطقي is_power_of_two(size_t value) {
    ܦܢܐ value != 0 && (value & (value - 1)) == 0;
}

منطقي align_up(size_t value, size_t alignment, size_t * aligned_value) {
    ܐܢ (aligned_value == nullptr || alignment == 0) {
        ܦܢܐ ܕܓܠܐ;
    }

    ثابت size_t remainder = value % alignment;
    ܐܢ (remainder == 0) {
        *aligned_value = value;
        ܦܢܐ ܫܪܪܐ;
    }

    ثابت size_t padding = alignment - remainder;
    ܐܢ (value > std::numeric_limits<size_t>::max() - padding) {
        ܦܢܐ ܕܓܠܐ;
    }

    *aligned_value = value + padding;
    ܦܢܐ ܫܪܪܐ;
}

منطقي align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_value) {
    ܐܢ (aligned_value == nullptr || alignment == 0) {
        ܦܢܐ ܕܓܠܐ;
    }

    ثابت uintptr_t remainder = value % alignment;
    ܐܢ (remainder == 0) {
        *aligned_value = value;
        ܦܢܐ ܫܪܪܐ;
    }

    ثابت uintptr_t padding = alignment - remainder;
    ܐܢ (value > std::numeric_limits<uintptr_t>::max() - padding) {
        ܦܢܐ ܕܓܠܐ;
    }

    *aligned_value = value + padding;
    ܦܢܐ ܫܪܪܐ;
}

صنف spine_mem_pool_manager {
  عام:
    explicit spine_mem_pool_manager(size_t default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    virtual ~spine_mem_pool_manager() = افتراضي;

    فراغ * alloc(size_t size, size_t alignment) {
        ܐܢ (size == 0 || !is_power_of_two(alignment)) {
            ܦܢܐ nullptr;
        }

        size_t aligned_size = 0;
        ܐܢ (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            ܦܢܐ nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        ܐܢ (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            ܐܢ (!add_chunk_locked(aligned_size, alignment)) {
                ܦܢܐ nullptr;
            }

            ܐܢ (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                ܦܢܐ nullptr;
            }
        }

        حاول {
            ثابت auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            ܐܢ (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                ܦܢܐ nullptr;
            }
        } امسك (ثابت std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            ارم;
        }

        ܦܢܐ allocation.base;
    }

    فراغ حرر(فراغ * base) {
        ܐܢ (base == nullptr) {
            ܦܢܐ;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        ܐܢ (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            ܦܢܐ;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        ܐܢ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ܦܢܐ;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        ܐܢ (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            ܦܢܐ;
        }

        ثابت size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        ܐܢ (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            ܦܢܐ;
        }

        insert_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  محمي:
    فراغ release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        لكل (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_size() ثابت { ܦܢܐ default_chunk_size_; }

    ساكن فراغ clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    virtual منطقي alloc_chunk(size_t min_size, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) = 0;
    virtual فراغ dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  خاص:
    بنية alloc_candidate {
        size_t    chunk_index{ 0 };
        size_t    block_index{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        منطقي      valid{ ܕܓܠܐ };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(ثابت pool_allocation & allocation) {
        ܦܢܐ std::find_if(chunks_.begin(), chunks_.end(), [&](ثابت pool_chunk & chunk) {
            ܦܢܐ chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    منطقي add_chunk_locked(size_t min_size, size_t alignment) {
        pool_chunk   chunk;
        ثابت size_t chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        فراغ *       hint_addr     = nullptr;

        لكل (ثابت auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            ܐܢ (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        ܐܢ (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            ܦܢܐ ܕܓܠܐ;
        }

        ܐܢ (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            ܦܢܐ ܕܓܠܐ;
        }

        حاول {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } امسك (ثابت std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ارم;
        }

        ܦܢܐ ܫܪܪܐ;
    }

    فراغ rollback_allocation_locked(ثابت pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        ܐܢ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            ܦܢܐ;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        ܐܢ (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            ܦܢܐ;
        }

        ثابت size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        ܐܢ (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ܦܢܐ;
        }

        insert_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    منطقي try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        لكل (size_t chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            ثابت auto & chunk = chunks_[chunk_index];
            لكل (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                ثابت auto & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                ثابت auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                ܐܢ (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    استمر;
                }

                ܐܢ (aligned_addr < block_addr) {
                    استمر;
                }

                ثابت size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                ثابت size_t padding        = aligned_offset - block.offset;
                ܐܢ (padding > block.size || size > block.size - padding) {
                    استمر;
                }

                ܐܢ (!best.valid || aligned_addr < best.address) {
                    best.chunk_index    = chunk_index;
                    best.block_index    = block_index;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = ܫܪܪܐ;
                }
            }
        }

        ܐܢ (!best.valid) {
            ܦܢܐ ܕܓܠܐ;
        }

        auto &           chunk     = chunks_[best.chunk_index];
        ثابت free_block block     = chunk.free_blocks[best.block_index];
        ثابت size_t     padding   = best.aligned_offset - block.offset;
        ثابت size_t     alloc_end = best.aligned_offset + size;
        ثابت size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        auto insert_it = chunk.free_blocks.begin() + best.block_index;
        ܐܢ (padding != 0) {
            insert_it = chunk.free_blocks.insert(insert_it, { block.offset, padding });
            ++insert_it;
        }
        ܐܢ (alloc_end < block_end) {
            chunk.free_blocks.insert(insert_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        ܦܢܐ ܫܪܪܐ;
    }

    فراغ maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        ܐܢ (chunk_it->free_blocks.size() != 1) {
            ܦܢܐ;
        }

        ثابت auto & block = chunk_it->free_blocks.front();
        ܐܢ (block.offset != 0 || block.size != chunk_it->size) {
            ܦܢܐ;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    فراغ insert_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        ܟܡܐ (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        ܐܢ (it != chunk.free_blocks.begin()) {
            ثابت auto & prev = *(it - 1);
            ܐܢ (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                ܦܢܐ;
            }
        }

        ܐܢ (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            ܦܢܐ;
        }

        it = chunk.free_blocks.insert(it, block);

        ܐܢ (it != chunk.free_blocks.begin()) {
            auto prev = it - 1;
            ܐܢ (prev->offset + prev->size == it->offset) {
                it->offset = prev->offset;
                it->size += prev->size;
                it = chunk.free_blocks.erase(prev);
            }
        }

        ܐܢ (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<فراغ *, pool_allocation> allocations_;
    size_t                                      default_chunk_size_{ 0 };
};

صنف spine_mem_pool_posix final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_size, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) hint_addr;

        ثابت size_t alloc_alignment = std::max(alignment, حجم(فراغ *));
        فراغ *       base            = nullptr;
        ثابت صحيح    rc              = posix_memalign(&base, alloc_alignment, min_size);
        ܐܢ (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            ܦܢܐ ܕܓܠܐ;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        ܦܢܐ ܫܪܪܐ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        std::حرر(chunk->base);
        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_transparent_hugepage final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_size, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;

        size_t chunk_size = 0;
        ܐܢ (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            ܦܢܐ ܕܓܠܐ;
        }

        فراغ * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ܐܢ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           errno);
            ܦܢܐ ܕܓܠܐ;
        }

        ܐܢ (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, errno);
            munmap(map_addr, chunk_size);
            ܦܢܐ ܕܓܠܐ;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        ܦܢܐ ܫܪܪܐ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        ܐܢ (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_hugetlb_1g final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_size, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;
        (فراغ) hint_addr;

        size_t region_size = 0;
        ܐܢ (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            ܦܢܐ ܕܓܠܐ;
        }

        ثابت صحيح fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        ܐܢ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            ܦܢܐ ܕܓܠܐ;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        ܐܢ (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, errno);
            close(fd);
            ܦܢܐ ܕܓܠܐ;
        }

        فراغ * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ܐܢ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            ܦܢܐ ܕܓܠܐ;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        ܦܢܐ ܫܪܪܐ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        ܐܢ (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        ܐܢ (chunk->fd >= 0) {
            ܐܢ (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_shared_mem final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_size, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;

        ܐܢ (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            ܦܢܐ ܕܓܠܐ;
        }

        ܐܢ (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            ܦܢܐ ܕܓܠܐ;
        }

        ثابت صحيح fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        ܐܢ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            ܦܢܐ ܕܓܠܐ;
        }

        فراغ * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ܐܢ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), errno);
            close(fd);
            ܦܢܐ ܕܓܠܐ;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        ܦܢܐ ܫܪܪܐ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        ܐܢ (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        ܐܢ (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    ساكن std::once_flag                          pool_once;
    ساكن std::unique_ptr<spine_mem_pool_manager> selected_pool;
    ساكن spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    ܐܢ (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        بدل (selected_backend) {
            حالة spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                اكسر;
            حالة spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                اكسر;
            حالة spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                اكسر;
            حالة spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                اكسر;
        }
    });

    ܐܢ (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<صحيح>(backend), static_cast<صحيح>(selected_backend));
    }

    ܐܢ (selected_pool) {
        ܦܢܐ *selected_pool;
    }

    ارم std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    ساكن std::once_flag                             shared_mem_pool_once;
    ساكن std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    ܐܢ (shared_mem_pool) {
        ܦܢܐ *shared_mem_pool;
    }

    ارم std::bad_alloc();
}

}  // namespace

منطقي spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    ܐܢ (info == nullptr) {
        ܦܢܐ ܕܓܠܐ;
    }

    *info = {};

    ܐܢ (spine_tcm_open_handle(عدم) != 0 || !spine_tcm_is_available()) {
        ܦܢܐ ܕܓܠܐ;
    }

    spine_tcm_mem_info_t mem_info;
    ܐܢ (spine_tcm_mem_info(&mem_info) != 0) {
        ܦܢܐ ܕܓܠܐ;
    }

    info->available   = ܫܪܪܐ;
    info->blk_size    = mem_info.blk_size;
    info->blk_num     = mem_info.blk_num;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    ܦܢܐ ܫܪܪܐ;
}

فراغ * spine_mem_pool_tcm_mem_get(صحيح cpu_id) noexcept {
    ܦܢܐ spine_tcm_mem_get(cpu_id);
}

فراغ * spine_mem_pool_tcm_mem_wait(صحيح cpu_id) noexcept {
    ܦܢܐ spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

صحيح spine_mem_pool_tcm_mem_release(صحيح cpu_id) noexcept {
    ܦܢܐ spine_tcm_mem_release(cpu_id);
}

فراغ * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    حاول {
        ܦܢܐ get_spine_mem_pool_manager().alloc(size, alignment);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        ܦܢܐ nullptr;
    }
}

فراغ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    حاول {
        ܦܢܐ get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        ܦܢܐ nullptr;
    }
}

فراغ spine_mem_pool_free(فراغ * base) noexcept {
    حاول {
        get_spine_mem_pool_manager().حرر(base);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

فراغ spine_mem_pool_shared_mem_free(فراغ * base) noexcept {
    حاول {
        get_spine_mem_pool_shared_mem_manager().حرر(base);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

خارجي "C" {
فراغ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    فراغ * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    ܐܢ (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    ܦܢܐ result;
}

فراغ ggml_backend_cpu_riscv64_spacemit_free_shared(فراغ * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
