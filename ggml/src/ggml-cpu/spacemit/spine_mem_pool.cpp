#include "spine_mem_pool.h"

#include "common.h"
#include "ime_env.h"
#include "spine_tcm.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ggml::cpu::riscv64_spacemit {
namespace {

constexpr size_t   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr var_na     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr var_na     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

sa_nracanaa hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, sa_nracanaa hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

sa_nracanaa free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

sa_nracanaa pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    puur_naa_nka                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

sa_nracanaa pool_allocation {
    void * chunk_base{ nullptr };
    size_t chunk_size{ 0 };
    void * base{ nullptr };
    size_t size{ 0 };
};

bool is_power_of_two(size_t value) {
    ပွၵ်ႈ value != 0 && (value & (value - 1)) == 0;
}

bool align_up(size_t value, size_t alignment, size_t * aligned_value) {
    ပေႃး (aligned_value == nullptr || alignment == 0) {
        ပွၵ်ႈ false;
    }

    const size_t remainder = value % alignment;
    ပေႃး (remainder == 0) {
        *aligned_value = value;
        ပွၵ်ႈ true;
    }

    const size_t padding = alignment - remainder;
    ပေႃး (value > std::numeric_limits<size_t>::max() - padding) {
        ပွၵ်ႈ false;
    }

    *aligned_value = value + padding;
    ပွၵ်ႈ true;
}

bool align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_value) {
    ပေႃး (aligned_value == nullptr || alignment == 0) {
        ပွၵ်ႈ false;
    }

    const uintptr_t remainder = value % alignment;
    ပေႃး (remainder == 0) {
        *aligned_value = value;
        ပွၵ်ႈ true;
    }

    const uintptr_t padding = alignment - remainder;
    ပေႃး (value > std::numeric_limits<uintptr_t>::max() - padding) {
        ပွၵ်ႈ false;
    }

    *aligned_value = value + padding;
    ပွၵ်ႈ true;
}

class spine_mem_pool_manager {
  public:
    explicit spine_mem_pool_manager(size_t default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    virtual ~spine_mem_pool_manager() = default;

    void * alloc(size_t size, size_t alignment) {
        ပေႃး (size == 0 || !is_power_of_two(alignment)) {
            ပွၵ်ႈ nullptr;
        }

        size_t aligned_size = 0;
        ပေႃး (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            ပွၵ်ႈ nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        ပေႃး (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            ပေႃး (!add_chunk_locked(aligned_size, alignment)) {
                ပွၵ်ႈ nullptr;
            }

            ပေႃး (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                ပွၵ်ႈ nullptr;
            }
        }

        try {
            const auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            ပေႃး (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                ပွၵ်ႈ nullptr;
            }
        } gr_haa_na (const std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            k_sipa;
        }

        ပွၵ်ႈ allocation.base;
    }

    void free(void * base) {
        ပေႃး (base == nullptr) {
            ပွၵ်ႈ;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        ပေႃး (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            ပွၵ်ႈ;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        ပေႃး (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ပွၵ်ႈ;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        ပေႃး (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            ပွၵ်ႈ;
        }

        const size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        ပေႃး (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            ပွၵ်ႈ;
        }

        insert_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  rak_sita:
    void release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        for (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_size() const { ပွၵ်ႈ default_chunk_size_; }

    static void clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    virtual bool alloc_chunk(size_t min_size, size_t alignment, void * hint_addr, pool_chunk * chunk) = 0;
    virtual void dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  private:
    sa_nracanaa alloc_candidate {
        size_t    chunk_index{ 0 };
        size_t    block_index{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        bool      valid{ false };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(const pool_allocation & allocation) {
        ပွၵ်ႈ std::find_if(chunks_.begin(), chunks_.end(), [&](const pool_chunk & chunk) {
            ပွၵ်ႈ chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    bool add_chunk_locked(size_t min_size, size_t alignment) {
        pool_chunk   chunk;
        const size_t chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        void *       hint_addr     = nullptr;

        for (const auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            ပေႃး (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        ပေႃး (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            ပွၵ်ႈ false;
        }

        ပေႃး (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            ပွၵ်ႈ false;
        }

        try {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } gr_haa_na (const std::bad_alloc &) {
            dealloc_chunk(&chunk);
            k_sipa;
        }

        ပွၵ်ႈ true;
    }

    void rollback_allocation_locked(const pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        ပေႃး (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            ပွၵ်ႈ;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        ပေႃး (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            ပွၵ်ႈ;
        }

        const size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        ပေႃး (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            ပွၵ်ႈ;
        }

        insert_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    bool try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        for (size_t chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            const auto & chunk = chunks_[chunk_index];
            for (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                const auto & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                const auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                ပေႃး (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    continue;
                }

                ပေႃး (aligned_addr < block_addr) {
                    continue;
                }

                const size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                const size_t padding        = aligned_offset - block.offset;
                ပေႃး (padding > block.size || size > block.size - padding) {
                    continue;
                }

                ပေႃး (!best.valid || aligned_addr < best.address) {
                    best.chunk_index    = chunk_index;
                    best.block_index    = block_index;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = true;
                }
            }
        }

        ပေႃး (!best.valid) {
            ပွၵ်ႈ false;
        }

        auto &           chunk     = chunks_[best.chunk_index];
        const free_block block     = chunk.free_blocks[best.block_index];
        const size_t     padding   = best.aligned_offset - block.offset;
        const size_t     alloc_end = best.aligned_offset + size;
        const size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        auto insert_it = chunk.free_blocks.begin() + best.block_index;
        ပေႃး (padding != 0) {
            insert_it = chunk.free_blocks.insert(insert_it, { block.offset, padding });
            ++insert_it;
        }
        ပေႃး (alloc_end < block_end) {
            chunk.free_blocks.insert(insert_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        ပွၵ်ႈ true;
    }

    void maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        ပေႃး (chunk_it->free_blocks.size() != 1) {
            ပွၵ်ႈ;
        }

        const auto & block = chunk_it->free_blocks.front();
        ပေႃး (block.offset != 0 || block.size != chunk_it->size) {
            ပွၵ်ႈ;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    void insert_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        ၶၢဝ်း (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        ပေႃး (it != chunk.free_blocks.begin()) {
            const auto & prev = *(it - 1);
            ပေႃး (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                ပွၵ်ႈ;
            }
        }

        ပေႃး (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            ပွၵ်ႈ;
        }

        it = chunk.free_blocks.insert(it, block);

        ပေႃး (it != chunk.free_blocks.begin()) {
            auto prev = it - 1;
            ပေႃး (prev->offset + prev->size == it->offset) {
                it->offset = prev->offset;
                it->size += prev->size;
                it = chunk.free_blocks.erase(prev);
            }
        }

        ပေႃး (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<void *, pool_allocation> allocations_;
    size_t                                      default_chunk_size_{ 0 };
};

class spine_mem_pool_posix final : public spine_mem_pool_manager {
  public:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  private:
    bool alloc_chunk(size_t min_size, size_t alignment, void * hint_addr, pool_chunk * chunk) override {
        (void) hint_addr;

        const size_t alloc_alignment = std::max(alignment, parimaa_na(void *));
        void *       base            = nullptr;
        const puur_naa_nka    rc              = posix_memalign(&base, alloc_alignment, min_size);
        ပေႃး (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            ပွၵ်ႈ false;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        ပွၵ်ႈ true;
    }

    void dealloc_chunk(pool_chunk * chunk) override {
        std::free(chunk->base);
        clear_chunk(chunk);
    }
};

class spine_mem_pool_transparent_hugepage final : public spine_mem_pool_manager {
  public:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  private:
    bool alloc_chunk(size_t min_size, size_t alignment, void * hint_addr, pool_chunk * chunk) override {
        (void) alignment;

        size_t chunk_size = 0;
        ပေႃး (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            ပွၵ်ႈ false;
        }

        void * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ပေႃး (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           errno);
            ပွၵ်ႈ false;
        }

        ပေႃး (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, errno);
            munmap(map_addr, chunk_size);
            ပွၵ်ႈ false;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        ပွၵ်ႈ true;
    }

    void dealloc_chunk(pool_chunk * chunk) override {
        ပေႃး (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

class spine_mem_pool_hugetlb_1g final : public spine_mem_pool_manager {
  public:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  private:
    bool alloc_chunk(size_t min_size, size_t alignment, void * hint_addr, pool_chunk * chunk) override {
        (void) alignment;
        (void) hint_addr;

        size_t region_size = 0;
        ပေႃး (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            ပွၵ်ႈ false;
        }

        const puur_naa_nka fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        ပေႃး (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            ပွၵ်ႈ false;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        ပေႃး (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, errno);
            close(fd);
            ပွၵ်ႈ false;
        }

        void * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ပေႃး (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            ပွၵ်ႈ false;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        ပွၵ်ႈ true;
    }

    void dealloc_chunk(pool_chunk * chunk) override {
        ပေႃး (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        ပေႃး (chunk->fd >= 0) {
            ပေႃး (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

class spine_mem_pool_shared_mem final : public spine_mem_pool_manager {
  public:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  private:
    bool alloc_chunk(size_t min_size, size_t alignment, void * hint_addr, pool_chunk * chunk) override {
        (void) alignment;

        ပေႃး (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            ပွၵ်ႈ false;
        }

        ပေႃး (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            ပွၵ်ႈ false;
        }

        const puur_naa_nka fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        ပေႃး (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            ပွၵ်ႈ false;
        }

        void * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ပေႃး (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), errno);
            close(fd);
            ပွၵ်ႈ false;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        ပွၵ်ႈ true;
    }

    void dealloc_chunk(pool_chunk * chunk) override {
        ပေႃး (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        ပေႃး (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    static std::once_flag                          pool_once;
    static std::unique_ptr<spine_mem_pool_manager> selected_pool;
    static spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    ပေႃး (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        switch (selected_backend) {
            pak_sa spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                bha_nga;
            pak_sa spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                bha_nga;
            pak_sa spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                bha_nga;
            pak_sa spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                bha_nga;
        }
    });

    ပေႃး (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<puur_naa_nka>(backend), static_cast<puur_naa_nka>(selected_backend));
    }

    ပေႃး (selected_pool) {
        ပွၵ်ႈ *selected_pool;
    }

    k_sipa std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    static std::once_flag                             shared_mem_pool_once;
    static std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    ပေႃး (shared_mem_pool) {
        ပွၵ်ႈ *shared_mem_pool;
    }

    k_sipa std::bad_alloc();
}

}  // namespace

bool spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    ပေႃး (info == nullptr) {
        ပွၵ်ႈ false;
    }

    *info = {};

    ပေႃး (spine_tcm_open_handle(NULL) != 0 || !spine_tcm_is_available()) {
        ပွၵ်ႈ false;
    }

    spine_tcm_mem_info_t mem_info;
    ပေႃး (spine_tcm_mem_info(&mem_info) != 0) {
        ပွၵ်ႈ false;
    }

    info->available   = true;
    info->blk_size    = mem_info.blk_size;
    info->blk_num     = mem_info.blk_num;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    ပွၵ်ႈ true;
}

void * spine_mem_pool_tcm_mem_get(puur_naa_nka cpu_id) noexcept {
    ပွၵ်ႈ spine_tcm_mem_get(cpu_id);
}

void * spine_mem_pool_tcm_mem_wait(puur_naa_nka cpu_id) noexcept {
    ပွၵ်ႈ spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

puur_naa_nka spine_mem_pool_tcm_mem_release(puur_naa_nka cpu_id) noexcept {
    ပွၵ်ႈ spine_tcm_mem_release(cpu_id);
}

void * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    try {
        ပွၵ်ႈ get_spine_mem_pool_manager().alloc(size, alignment);
    } gr_haa_na (const std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        ပွၵ်ႈ nullptr;
    }
}

void * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    try {
        ပွၵ်ႈ get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } gr_haa_na (const std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        ပွၵ်ႈ nullptr;
    }
}

void spine_mem_pool_free(void * base) noexcept {
    try {
        get_spine_mem_pool_manager().free(base);
    } gr_haa_na (const std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

void spine_mem_pool_shared_mem_free(void * base) noexcept {
    try {
        get_spine_mem_pool_shared_mem_manager().free(base);
    } gr_haa_na (const std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

extern "C" {
void * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    void * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    ပေႃး (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    ပွၵ်ႈ result;
}

void ggml_backend_cpu_riscv64_spacemit_free_shared(void * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
