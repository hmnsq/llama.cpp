#شامل "spine_mem_pool.h"

#شامل "common.h"
#شامل "ime_env.h"
#شامل "spine_tcm.h"

#شامل <fcntl.h>
#شامل <sys/ioctl.h>
#شامل <sys/mman.h>
#شامل <unistd.h>

#شامل <algorithm>
#شامل <cerrno>
#شامل <cstdint>
#شامل <cstdlib>
#شامل <limits>
#شامل <memory>
#شامل <mutex>
#شامل <unordered_map>
#شامل <vector>

فضاینام ggml::cpu::riscv64_spacemit {
فضاینام {

constexpr size_t   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr نویسه     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr نویسه     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

ساختار hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, ساختار hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

ساختار free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

ساختار pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    صحیح                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

ساختار pool_allocation {
    تهی * chunk_base{ nullptr };
    size_t chunk_size{ 0 };
    تهی * base{ nullptr };
    size_t size{ 0 };
};

منطقی is_power_of_two(size_t value) {
    بازگشت value != 0 && (value & (value - 1)) == 0;
}

منطقی align_up(size_t value, size_t alignment, size_t * aligned_مقدار) {
    اگر (aligned_مقدار == nullptr || alignment == 0) {
        بازگشت نادرست;
    }

    ثابت size_t remainder = value % alignment;
    اگر (remainder == 0) {
        *aligned_مقدار = value;
        بازگشت درست;
    }

    ثابت size_t padding = alignment - remainder;
    اگر (value > std::numeric_limits<size_t>::max() - padding) {
        بازگشت نادرست;
    }

    *aligned_مقدار = value + padding;
    بازگشت درست;
}

منطقی align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_مقدار) {
    اگر (aligned_مقدار == nullptr || alignment == 0) {
        بازگشت نادرست;
    }

    ثابت uintptr_t remainder = value % alignment;
    اگر (remainder == 0) {
        *aligned_مقدار = value;
        بازگشت درست;
    }

    ثابت uintptr_t padding = alignment - remainder;
    اگر (value > std::numeric_limits<uintptr_t>::max() - padding) {
        بازگشت نادرست;
    }

    *aligned_مقدار = value + padding;
    بازگشت درست;
}

رده spine_mem_pool_manager {
  عمومی:
    explicit spine_mem_pool_manager(size_t default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    مجازی ~spine_mem_pool_manager() = پیشفرض;

    تهی * alloc(size_t size, size_t alignment) {
        اگر (size == 0 || !is_power_of_two(alignment)) {
            بازگشت nullptr;
        }

        size_t aligned_size = 0;
        اگر (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            بازگشت nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        اگر (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            اگر (!add_chunk_locked(aligned_size, alignment)) {
                بازگشت nullptr;
            }

            اگر (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                بازگشت nullptr;
            }
        }

        تلاش {
            ثابت auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            اگر (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                بازگشت nullptr;
            }
        } بگیر (ثابت std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            پرتابکن;
        }

        بازگشت allocation.base;
    }

    تهی آزاد(تهی * base) {
        اگر (base == nullptr) {
            بازگشت;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        اگر (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            بازگشت;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        اگر (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            بازگشت;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        اگر (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            بازگشت;
        }

        ثابت size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        اگر (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            بازگشت;
        }

        درج_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  محافظتشده:
    تهی release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        برای (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_size() ثابت { بازگشت default_chunk_size_; }

    ایستا تهی clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    مجازی منطقی alloc_chunk(size_t min_size, size_t alignment, تهی * hint_addr, pool_chunk * chunk) = 0;
    مجازی تهی dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  خصوصی:
    ساختار alloc_candidate {
        size_t    chunk_index{ 0 };
        size_t    block_index{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        منطقی      valid{ نادرست };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(ثابت pool_allocation & allocation) {
        بازگشت std::find_if(chunks_.begin(), chunks_.end(), [&](ثابت pool_chunk & chunk) {
            بازگشت chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    منطقی add_chunk_locked(size_t min_size, size_t alignment) {
        pool_chunk   chunk;
        ثابت size_t chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        تهی *       hint_addr     = nullptr;

        برای (ثابت auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            اگر (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        اگر (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            بازگشت نادرست;
        }

        اگر (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            بازگشت نادرست;
        }

        تلاش {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } بگیر (ثابت std::bad_alloc &) {
            dealloc_chunk(&chunk);
            پرتابکن;
        }

        بازگشت درست;
    }

    تهی rollback_allocation_locked(ثابت pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        اگر (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            بازگشت;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        اگر (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            بازگشت;
        }

        ثابت size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        اگر (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            بازگشت;
        }

        درج_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    منطقی try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        برای (size_t chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            ثابت auto & chunk = chunks_[chunk_index];
            برای (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                ثابت auto & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                ثابت auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                اگر (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    ادامه;
                }

                اگر (aligned_addr < block_addr) {
                    ادامه;
                }

                ثابت size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                ثابت size_t padding        = aligned_offset - block.offset;
                اگر (padding > block.size || size > block.size - padding) {
                    ادامه;
                }

                اگر (!best.valid || aligned_addr < best.address) {
                    best.chunk_index    = chunk_index;
                    best.block_index    = block_index;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = درست;
                }
            }
        }

        اگر (!best.valid) {
            بازگشت نادرست;
        }

        auto &           chunk     = chunks_[best.chunk_index];
        ثابت free_block block     = chunk.free_blocks[best.block_index];
        ثابت size_t     padding   = best.aligned_offset - block.offset;
        ثابت size_t     alloc_end = best.aligned_offset + size;
        ثابت size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        auto درج_it = chunk.free_blocks.begin() + best.block_index;
        اگر (padding != 0) {
            درج_it = chunk.free_blocks.insert(درج_it, { block.offset, padding });
            ++درج_it;
        }
        اگر (alloc_end < block_end) {
            chunk.free_blocks.insert(درج_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        بازگشت درست;
    }

    تهی maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        اگر (chunk_it->free_blocks.size() != 1) {
            بازگشت;
        }

        ثابت auto & block = chunk_it->free_blocks.front();
        اگر (block.offset != 0 || block.size != chunk_it->size) {
            بازگشت;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    تهی درج_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        وقتی (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        اگر (it != chunk.free_blocks.begin()) {
            ثابت auto & prev = *(it - 1);
            اگر (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                بازگشت;
            }
        }

        اگر (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            بازگشت;
        }

        it = chunk.free_blocks.insert(it, block);

        اگر (it != chunk.free_blocks.begin()) {
            auto prev = it - 1;
            اگر (prev->offset + prev->size == it->offset) {
                it->offset = prev->offset;
                it->size += prev->size;
                it = chunk.free_blocks.erase(prev);
            }
        }

        اگر (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<تهی *, pool_allocation> allocations_;
    size_t                                      default_chunk_size_{ 0 };
};

رده spine_mem_pool_posix final : عمومی spine_mem_pool_manager {
  عمومی:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  خصوصی:
    منطقی alloc_chunk(size_t min_size, size_t alignment, تهی * hint_addr, pool_chunk * chunk) override {
        (تهی) hint_addr;

        ثابت size_t alloc_alignment = std::max(alignment, اندازه(تهی *));
        تهی *       base            = nullptr;
        ثابت صحیح    rc              = posix_memalign(&base, alloc_alignment, min_size);
        اگر (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            بازگشت نادرست;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        بازگشت درست;
    }

    تهی dealloc_chunk(pool_chunk * chunk) override {
        std::آزاد(chunk->base);
        clear_chunk(chunk);
    }
};

رده spine_mem_pool_transparent_hugepage final : عمومی spine_mem_pool_manager {
  عمومی:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  خصوصی:
    منطقی alloc_chunk(size_t min_size, size_t alignment, تهی * hint_addr, pool_chunk * chunk) override {
        (تهی) alignment;

        size_t chunk_size = 0;
        اگر (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            بازگشت نادرست;
        }

        تهی * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        اگر (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           errno);
            بازگشت نادرست;
        }

        اگر (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, errno);
            munmap(map_addr, chunk_size);
            بازگشت نادرست;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        بازگشت درست;
    }

    تهی dealloc_chunk(pool_chunk * chunk) override {
        اگر (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

رده spine_mem_pool_hugetlb_1g final : عمومی spine_mem_pool_manager {
  عمومی:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  خصوصی:
    منطقی alloc_chunk(size_t min_size, size_t alignment, تهی * hint_addr, pool_chunk * chunk) override {
        (تهی) alignment;
        (تهی) hint_addr;

        size_t region_size = 0;
        اگر (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            بازگشت نادرست;
        }

        ثابت صحیح fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        اگر (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            بازگشت نادرست;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        اگر (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, errno);
            close(fd);
            بازگشت نادرست;
        }

        تهی * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        اگر (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            بازگشت نادرست;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        بازگشت درست;
    }

    تهی dealloc_chunk(pool_chunk * chunk) override {
        اگر (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        اگر (chunk->fd >= 0) {
            اگر (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

رده spine_mem_pool_shared_mem final : عمومی spine_mem_pool_manager {
  عمومی:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  خصوصی:
    منطقی alloc_chunk(size_t min_size, size_t alignment, تهی * hint_addr, pool_chunk * chunk) override {
        (تهی) alignment;

        اگر (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            بازگشت نادرست;
        }

        اگر (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            بازگشت نادرست;
        }

        ثابت صحیح fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        اگر (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            بازگشت نادرست;
        }

        تهی * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        اگر (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), errno);
            close(fd);
            بازگشت نادرست;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        بازگشت درست;
    }

    تهی dealloc_chunk(pool_chunk * chunk) override {
        اگر (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        اگر (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    ایستا std::once_flag                          pool_once;
    ایستا std::unique_ptr<spine_mem_pool_manager> selected_pool;
    ایستا spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    اگر (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        انتخاب (selected_backend) {
            حالت spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                بشکن;
            حالت spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                بشکن;
            حالت spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                بشکن;
            حالت spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                بشکن;
        }
    });

    اگر (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<صحیح>(backend), static_cast<صحیح>(selected_backend));
    }

    اگر (selected_pool) {
        بازگشت *selected_pool;
    }

    پرتابکن std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    ایستا std::once_flag                             shared_mem_pool_once;
    ایستا std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    اگر (shared_mem_pool) {
        بازگشت *shared_mem_pool;
    }

    پرتابکن std::bad_alloc();
}

}  // namespace

منطقی spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    اگر (info == nullptr) {
        بازگشت نادرست;
    }

    *info = {};

    اگر (spine_tcm_open_handle(پوچ) != 0 || !spine_tcm_is_available()) {
        بازگشت نادرست;
    }

    spine_tcm_mem_info_t mem_info;
    اگر (spine_tcm_mem_info(&mem_info) != 0) {
        بازگشت نادرست;
    }

    info->available   = درست;
    info->blk_size    = mem_info.blk_size;
    info->blk_رقم     = mem_info.blk_رقم;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    بازگشت درست;
}

تهی * spine_mem_pool_tcm_mem_get(صحیح cpu_id) noexcept {
    بازگشت spine_tcm_mem_get(cpu_id);
}

تهی * spine_mem_pool_tcm_mem_wait(صحیح cpu_id) noexcept {
    بازگشت spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

صحیح spine_mem_pool_tcm_mem_release(صحیح cpu_id) noexcept {
    بازگشت spine_tcm_mem_release(cpu_id);
}

تهی * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    تلاش {
        بازگشت get_spine_mem_pool_manager().alloc(size, alignment);
    } بگیر (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        بازگشت nullptr;
    }
}

تهی * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    تلاش {
        بازگشت get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } بگیر (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        بازگشت nullptr;
    }
}

تهی spine_mem_pool_free(تهی * base) noexcept {
    تلاش {
        get_spine_mem_pool_manager().آزاد(base);
    } بگیر (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

تهی spine_mem_pool_shared_mem_free(تهی * base) noexcept {
    تلاش {
        get_spine_mem_pool_shared_mem_manager().آزاد(base);
    } بگیر (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

بیرونی "C" {
تهی * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    تهی * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    اگر (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    بازگشت result;
}

تهی ggml_backend_cpu_riscv64_spacemit_free_shared(تهی * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
