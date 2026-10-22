#অন্তর্ভাব "spine_mem_pool.h"

#অন্তর্ভাব "common.h"
#অন্তর্ভাব "ime_env.h"
#অন্তর্ভাব "spine_tcm.h"

#অন্তর্ভাব <fcntl.h>
#অন্তর্ভাব <sys/ioctl.h>
#অন্তর্ভাব <sys/mman.h>
#অন্তর্ভাব <unistd.h>

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cerrno>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <limits>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <mutex>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <vector>

নামাকাশ ggml::cpu::riscv64_spacemit {
নামাকাশ {

constexpr size_t   SPINE_MEM_POOL_CHUNK_আকার         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_আকার   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1_G_REGION_আকার     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr অক্ষর     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr অক্ষর     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

কাঠামো hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, কাঠামো hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

কাঠামো free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

কাঠামো pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    পূর্ণসংখ্যা                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

কাঠামো pool_allocation {
    শূন্য * chunk_base{ nullptr };
    size_t chunk_আকার{ 0 };
    শূন্য * base{ nullptr };
    size_t size{ 0 };
};

যৌক্তিক is_power_of_two(size_t value) {
    ফেরত value != 0 && (value & (value - 1)) == 0;
}

যৌক্তিক align_up(size_t value, size_t alignment, size_t * aligned_মান) {
    যদি (aligned_মান == nullptr || alignment == 0) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক size_t remainder = value % alignment;
    যদি (remainder == 0) {
        *aligned_মান = value;
        ফেরত সত্য;
    }

    ধ্রুবক size_t padding = alignment - remainder;
    যদি (value > std::numeric_limits<size_t>::max() - padding) {
        ফেরত মিথ্যা;
    }

    *aligned_মান = value + padding;
    ফেরত সত্য;
}

যৌক্তিক align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_মান) {
    যদি (aligned_মান == nullptr || alignment == 0) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক uintptr_t remainder = value % alignment;
    যদি (remainder == 0) {
        *aligned_মান = value;
        ফেরত সত্য;
    }

    ধ্রুবক uintptr_t padding = alignment - remainder;
    যদি (value > std::numeric_limits<uintptr_t>::max() - padding) {
        ফেরত মিথ্যা;
    }

    *aligned_মান = value + padding;
    ফেরত সত্য;
}

শ্রেণী spine_mem_pool_manager {
  সর্বজনীন:
    explicit spine_mem_pool_manager(size_t default_chunk_আকার) : default_chunk_আকার_2(default_chunk_আকার) {}

    আভাসী ~spine_mem_pool_manager() = পূর্বনির্ধারিত;

    শূন্য * alloc(size_t size, size_t alignment) {
        যদি (size == 0 || !is_power_of_two(alignment)) {
            ফেরত nullptr;
        }

        size_t aligned_আকার = 0;
        যদি (!align_up(size, alignment, &aligned_আকার)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            ফেরত nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        যদি (!try_alloc_locked(aligned_আকার, alignment, &allocation)) {
            যদি (!add_chunk_locked(aligned_আকার, alignment)) {
                ফেরত nullptr;
            }

            যদি (!try_alloc_locked(aligned_আকার, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_আকার, alignment);
                ফেরত nullptr;
            }
        }

        চেষ্টা {
            ধ্রুবক auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            যদি (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                ফেরত nullptr;
            }
        } ধরো (ধ্রুবক std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            ছোড়ো;
        }

        ফেরত allocation.base;
    }

    শূন্য মুক্ত(শূন্য * base) {
        যদি (base == nullptr) {
            ফেরত;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        যদি (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            ফেরত;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        যদি (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ফেরত;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        যদি (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            ফেরত;
        }

        ধ্রুবক size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        যদি (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            ফেরত;
        }

        ঢোকাও_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  সুরক্ষিত:
    শূন্য release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        জন্য (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_আকার() ধ্রুবক { ফেরত default_chunk_আকার_2; }

    স্থির শূন্য clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    আভাসী যৌক্তিক alloc_chunk(size_t min_আকার, size_t alignment, শূন্য * hint_addr, pool_chunk * chunk) = 0;
    আভাসী শূন্য dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  ব্যক্তিগত:
    কাঠামো alloc_candidate {
        size_t    chunk_সূচক{ 0 };
        size_t    block_সূচক{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        যৌক্তিক      valid{ মিথ্যা };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(ধ্রুবক pool_allocation & allocation) {
        ফেরত std::find_if(chunks_.begin(), chunks_.end(), [&](ধ্রুবক pool_chunk & chunk) {
            ফেরত chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_আকার;
        });
    }

    যৌক্তিক add_chunk_locked(size_t min_আকার, size_t alignment) {
        pool_chunk   chunk;
        ধ্রুবক size_t chunk_request = default_chunk_আকার_2 == 0 ? min_আকার : std::max(min_আকার, default_chunk_আকার_2);
        শূন্য *       hint_addr     = nullptr;

        জন্য (ধ্রুবক auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            যদি (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        যদি (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            ফেরত মিথ্যা;
        }

        যদি (chunk.base == nullptr || chunk.size < min_আকার) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_আকার, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            ফেরত মিথ্যা;
        }

        চেষ্টা {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } ধরো (ধ্রুবক std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ছোড়ো;
        }

        ফেরত সত্য;
    }

    শূন্য rollback_allocation_locked(ধ্রুবক pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        যদি (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            ফেরত;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        যদি (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            ফেরত;
        }

        ধ্রুবক size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        যদি (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ফেরত;
        }

        ঢোকাও_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    যৌক্তিক try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        জন্য (size_t chunk_সূচক = 0; chunk_সূচক < chunks_.size(); ++chunk_সূচক) {
            ধ্রুবক auto & chunk = chunks_[chunk_সূচক];
            জন্য (size_t block_সূচক = 0; block_সূচক < chunk.free_blocks.size(); ++block_সূচক) {
                ধ্রুবক auto & block = chunk.free_blocks[block_সূচক];

                uintptr_t  aligned_addr = 0;
                ধ্রুবক auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                যদি (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    চালু;
                }

                যদি (aligned_addr < block_addr) {
                    চালু;
                }

                ধ্রুবক size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                ধ্রুবক size_t padding        = aligned_offset - block.offset;
                যদি (padding > block.size || size > block.size - padding) {
                    চালু;
                }

                যদি (!best.valid || aligned_addr < best.address) {
                    best.chunk_সূচক    = chunk_সূচক;
                    best.block_সূচক    = block_সূচক;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = সত্য;
                }
            }
        }

        যদি (!best.valid) {
            ফেরত মিথ্যা;
        }

        auto &           chunk     = chunks_[best.chunk_সূচক];
        ধ্রুবক free_block block     = chunk.free_blocks[best.block_সূচক];
        ধ্রুবক size_t     padding   = best.aligned_offset - block.offset;
        ধ্রুবক size_t     alloc_end = best.aligned_offset + size;
        ধ্রুবক size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_সূচক);
        auto ঢোকাও_it = chunk.free_blocks.begin() + best.block_সূচক;
        যদি (padding != 0) {
            ঢোকাও_it = chunk.free_blocks.insert(ঢোকাও_it, { block.offset, padding });
            ++ঢোকাও_it;
        }
        যদি (alloc_end < block_end) {
            chunk.free_blocks.insert(ঢোকাও_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_আকার = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        ফেরত সত্য;
    }

    শূন্য maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        যদি (chunk_it->free_blocks.size() != 1) {
            ফেরত;
        }

        ধ্রুবক auto & block = chunk_it->free_blocks.front();
        যদি (block.offset != 0 || block.size != chunk_it->size) {
            ফেরত;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    শূন্য ঢোকাও_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        যখন (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        যদি (it != chunk.free_blocks.begin()) {
            ধ্রুবক auto & পূর্ব = *(it - 1);
            যদি (পূর্ব.offset + পূর্ব.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                ফেরত;
            }
        }

        যদি (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            ফেরত;
        }

        it = chunk.free_blocks.insert(it, block);

        যদি (it != chunk.free_blocks.begin()) {
            auto পূর্ব = it - 1;
            যদি (পূর্ব->offset + পূর্ব->size == it->offset) {
                it->offset = পূর্ব->offset;
                it->size += পূর্ব->size;
                it = chunk.free_blocks.erase(পূর্ব);
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
    size_t                                      default_chunk_আকার_2{ 0 };
};

শ্রেণী spine_mem_pool_posix final : সর্বজনীন spine_mem_pool_manager {
  সর্বজনীন:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  ব্যক্তিগত:
    যৌক্তিক alloc_chunk(size_t min_আকার, size_t alignment, শূন্য * hint_addr, pool_chunk * chunk) override {
        (শূন্য) hint_addr;

        ধ্রুবক size_t alloc_alignment = std::max(alignment, আকার(শূন্য *));
        শূন্য *       base            = nullptr;
        ধ্রুবক পূর্ণসংখ্যা    rc              = posix_memalign(&base, alloc_alignment, min_আকার);
        যদি (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_আকার, alloc_alignment, rc);
            ফেরত মিথ্যা;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_আকার;
        chunk->fd   = -1;
        ফেরত সত্য;
    }

    শূন্য dealloc_chunk(pool_chunk * chunk) override {
        std::মুক্ত(chunk->base);
        clear_chunk(chunk);
    }
};

শ্রেণী spine_mem_pool_transparent_hugepage final : সর্বজনীন spine_mem_pool_manager {
  সর্বজনীন:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_আকার) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  ব্যক্তিগত:
    যৌক্তিক alloc_chunk(size_t min_আকার, size_t alignment, শূন্য * hint_addr, pool_chunk * chunk) override {
        (শূন্য) alignment;

        size_t chunk_আকার = 0;
        যদি (!align_up(min_আকার, default_chunk_আকার(), &chunk_আকার)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_আকার);
            ফেরত মিথ্যা;
        }

        শূন্য * map_addr = mmap(hint_addr, chunk_আকার, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        যদি (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_আকার,
                           errno);
            ফেরত মিথ্যা;
        }

        যদি (madvise(map_addr, chunk_আকার, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_আকার, errno);
            munmap(map_addr, chunk_আকার);
            ফেরত মিথ্যা;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_আকার;
        chunk->fd   = -1;
        ফেরত সত্য;
    }

    শূন্য dealloc_chunk(pool_chunk * chunk) override {
        যদি (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

শ্রেণী spine_mem_pool_hugetlb_1g final : সর্বজনীন spine_mem_pool_manager {
  সর্বজনীন:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1_G_REGION_আকার) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  ব্যক্তিগত:
    যৌক্তিক alloc_chunk(size_t min_আকার, size_t alignment, শূন্য * hint_addr, pool_chunk * chunk) override {
        (শূন্য) alignment;
        (শূন্য) hint_addr;

        size_t region_আকার = 0;
        যদি (!align_up(min_আকার, SPINE_MEM_POOL_1_G_REGION_আকার, &region_আকার)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_আকার);
            ফেরত মিথ্যা;
        }

        ধ্রুবক পূর্ণসংখ্যা fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        যদি (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            ফেরত মিথ্যা;
        }

        hugetlb_1g_region region;
        region.size  = region_আকার;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        যদি (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_আকার, errno);
            close(fd);
            ফেরত মিথ্যা;
        }

        শূন্য * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        যদি (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            ফেরত মিথ্যা;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        ফেরত সত্য;
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

শ্রেণী spine_mem_pool_shared_mem final : সর্বজনীন spine_mem_pool_manager {
  সর্বজনীন:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_আকার) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  ব্যক্তিগত:
    যৌক্তিক alloc_chunk(size_t min_আকার, size_t alignment, শূন্য * hint_addr, pool_chunk * chunk) override {
        (শূন্য) alignment;

        যদি (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            ফেরত মিথ্যা;
        }

        যদি (min_আকার > default_chunk_আকার()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_আকার, default_chunk_আকার());
            ফেরত মিথ্যা;
        }

        ধ্রুবক পূর্ণসংখ্যা fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        যদি (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            ফেরত মিথ্যা;
        }

        শূন্য * map_addr = mmap(nullptr, default_chunk_আকার(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        যদি (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_আকার(), errno);
            close(fd);
            ফেরত মিথ্যা;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_আকার();
        chunk->fd   = fd;
        ফেরত সত্য;
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

        নির্বাচন (selected_backend) {
            ক্ষেত্র spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                বিরতি;
            ক্ষেত্র spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                বিরতি;
            ক্ষেত্র spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                বিরতি;
            ক্ষেত্র spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                বিরতি;
        }
    });

    যদি (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<পূর্ণসংখ্যা>(backend), static_cast<পূর্ণসংখ্যা>(selected_backend));
    }

    যদি (selected_pool) {
        ফেরত *selected_pool;
    }

    ছোড়ো std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    স্থির std::once_flag                             shared_mem_pool_once;
    স্থির std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    যদি (shared_mem_pool) {
        ফেরত *shared_mem_pool;
    }

    ছোড়ো std::bad_alloc();
}

}  // namespace

যৌক্তিক spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    যদি (info == nullptr) {
        ফেরত মিথ্যা;
    }

    *info = {};

    যদি (spine_tcm_open_handle(ফাঁকা) != 0 || !spine_tcm_is_available()) {
        ফেরত মিথ্যা;
    }

    spine_tcm_mem_info_t mem_info;
    যদি (spine_tcm_mem_info(&mem_info) != 0) {
        ফেরত মিথ্যা;
    }

    info->available   = সত্য;
    info->blk_আকার    = mem_info.blk_আকার;
    info->blk_অঙ্ক     = mem_info.blk_অঙ্ক;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    ফেরত সত্য;
}

শূন্য * spine_mem_pool_tcm_mem_get(পূর্ণসংখ্যা cpu_id) noexcept {
    ফেরত spine_tcm_mem_get(cpu_id);
}

শূন্য * spine_mem_pool_tcm_mem_wait(পূর্ণসংখ্যা cpu_id) noexcept {
    ফেরত spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

পূর্ণসংখ্যা spine_mem_pool_tcm_mem_release(পূর্ণসংখ্যা cpu_id) noexcept {
    ফেরত spine_tcm_mem_release(cpu_id);
}

শূন্য * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    চেষ্টা {
        ফেরত get_spine_mem_pool_manager().alloc(size, alignment);
    } ধরো (ধ্রুবক std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        ফেরত nullptr;
    }
}

শূন্য * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    চেষ্টা {
        ফেরত get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } ধরো (ধ্রুবক std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        ফেরত nullptr;
    }
}

শূন্য spine_mem_pool_free(শূন্য * base) noexcept {
    চেষ্টা {
        get_spine_mem_pool_manager().মুক্ত(base);
    } ধরো (ধ্রুবক std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

শূন্য spine_mem_pool_shared_mem_free(শূন্য * base) noexcept {
    চেষ্টা {
        get_spine_mem_pool_shared_mem_manager().মুক্ত(base);
    } ধরো (ধ্রুবক std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

বাহ্যিক "C" {
শূন্য * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    শূন্য * ফলাফল = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    যদি (ফলাফল == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    ফেরত ফলাফল;
}

শূন্য ggml_backend_cpu_riscv64_spacemit_free_shared(শূন্য * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
