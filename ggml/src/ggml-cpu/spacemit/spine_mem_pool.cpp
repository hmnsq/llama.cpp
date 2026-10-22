#અન્તર્ભાવ "spine_mem_pool.h"

#અન્તર્ભાવ "common.h"
#અન્તર્ભાવ "ime_env.h"
#અન્તર્ભાવ "spine_tcm.h"

#અન્તર્ભાવ <fcntl.h>
#અન્તર્ભાવ <sys/ioctl.h>
#અન્તર્ભાવ <sys/mman.h>
#અન્તર્ભાવ <unistd.h>

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cerrno>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <limits>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <unordered_map>
#અન્તર્ભાવ <vector>

નામાકાશ ggml::cpu::riscv64_spacemit {
નામાકાશ {

constexpr size_t   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr અક્ષર     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr અક્ષર     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

સંરચના hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, સંરચના hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

સંરચના free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

સંરચના pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    પૂર્ણાંક                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

સંરચના pool_allocation {
    શૂન્ય * chunk_base{ nullptr };
    size_t chunk_size{ 0 };
    શૂન્ય * base{ nullptr };
    size_t size{ 0 };
};

તાર્કિક is_power_of_two(size_t value) {
    પાછા value != 0 && (value & (value - 1)) == 0;
}

તાર્કિક align_up(size_t value, size_t alignment, size_t * aligned_કિંમત) {
    જો (aligned_કિંમત == nullptr || alignment == 0) {
        પાછા ખોટું;
    }

    અચલ size_t remainder = value % alignment;
    જો (remainder == 0) {
        *aligned_કિંમત = value;
        પાછા સાચું;
    }

    અચલ size_t padding = alignment - remainder;
    જો (value > std::numeric_limits<size_t>::max() - padding) {
        પાછા ખોટું;
    }

    *aligned_કિંમત = value + padding;
    પાછા સાચું;
}

તાર્કિક align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_કિંમત) {
    જો (aligned_કિંમત == nullptr || alignment == 0) {
        પાછા ખોટું;
    }

    અચલ uintptr_t remainder = value % alignment;
    જો (remainder == 0) {
        *aligned_કિંમત = value;
        પાછા સાચું;
    }

    અચલ uintptr_t padding = alignment - remainder;
    જો (value > std::numeric_limits<uintptr_t>::max() - padding) {
        પાછા ખોટું;
    }

    *aligned_કિંમત = value + padding;
    પાછા સાચું;
}

વર્ગ spine_mem_pool_manager {
  જાહેર:
    explicit spine_mem_pool_manager(size_t default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    આભાસી ~spine_mem_pool_manager() = મૂળભૂત;

    શૂન્ય * alloc(size_t size, size_t alignment) {
        જો (size == 0 || !is_power_of_two(alignment)) {
            પાછા nullptr;
        }

        size_t aligned_size = 0;
        જો (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            પાછા nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        જો (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            જો (!add_chunk_locked(aligned_size, alignment)) {
                પાછા nullptr;
            }

            જો (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                પાછા nullptr;
            }
        }

        પ્રયાસ {
            અચલ auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            જો (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                પાછા nullptr;
            }
        } પકડો (અચલ std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            ફેંકો;
        }

        પાછા allocation.base;
    }

    શૂન્ય મુક્ત(શૂન્ય * base) {
        જો (base == nullptr) {
            પાછા;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        જો (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            પાછા;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        જો (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            પાછા;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        જો (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            પાછા;
        }

        અચલ size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        જો (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            પાછા;
        }

        ઉમેરો_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  સુરક્ષિત:
    શૂન્ય release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        માટે (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_size() અચલ { પાછા default_chunk_size_; }

    સ્થિર શૂન્ય clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    આભાસી તાર્કિક alloc_chunk(size_t min_size, size_t alignment, શૂન્ય * hint_addr, pool_chunk * chunk) = 0;
    આભાસી શૂન્ય dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  ખાનગી:
    સંરચના alloc_candidate {
        size_t    chunk_index{ 0 };
        size_t    block_index{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        તાર્કિક      valid{ ખોટું };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(અચલ pool_allocation & allocation) {
        પાછા std::find_if(chunks_.begin(), chunks_.end(), [&](અચલ pool_chunk & chunk) {
            પાછા chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    તાર્કિક add_chunk_locked(size_t min_size, size_t alignment) {
        pool_chunk   chunk;
        અચલ size_t chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        શૂન્ય *       hint_addr     = nullptr;

        માટે (અચલ auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            જો (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        જો (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            પાછા ખોટું;
        }

        જો (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            પાછા ખોટું;
        }

        પ્રયાસ {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } પકડો (અચલ std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ફેંકો;
        }

        પાછા સાચું;
    }

    શૂન્ય rollback_allocation_locked(અચલ pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        જો (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            પાછા;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        જો (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            પાછા;
        }

        અચલ size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        જો (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            પાછા;
        }

        ઉમેરો_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    તાર્કિક try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        માટે (size_t chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            અચલ auto & chunk = chunks_[chunk_index];
            માટે (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                અચલ auto & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                અચલ auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                જો (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    ચાલુ;
                }

                જો (aligned_addr < block_addr) {
                    ચાલુ;
                }

                અચલ size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                અચલ size_t padding        = aligned_offset - block.offset;
                જો (padding > block.size || size > block.size - padding) {
                    ચાલુ;
                }

                જો (!best.valid || aligned_addr < best.address) {
                    best.chunk_index    = chunk_index;
                    best.block_index    = block_index;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = સાચું;
                }
            }
        }

        જો (!best.valid) {
            પાછા ખોટું;
        }

        auto &           chunk     = chunks_[best.chunk_index];
        અચલ free_block block     = chunk.free_blocks[best.block_index];
        અચલ size_t     padding   = best.aligned_offset - block.offset;
        અચલ size_t     alloc_end = best.aligned_offset + size;
        અચલ size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        auto ઉમેરો_it = chunk.free_blocks.begin() + best.block_index;
        જો (padding != 0) {
            ઉમેરો_it = chunk.free_blocks.insert(ઉમેરો_it, { block.offset, padding });
            ++ઉમેરો_it;
        }
        જો (alloc_end < block_end) {
            chunk.free_blocks.insert(ઉમેરો_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        પાછા સાચું;
    }

    શૂન્ય maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        જો (chunk_it->free_blocks.size() != 1) {
            પાછા;
        }

        અચલ auto & block = chunk_it->free_blocks.front();
        જો (block.offset != 0 || block.size != chunk_it->size) {
            પાછા;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    શૂન્ય ઉમેરો_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        જ્યારે (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        જો (it != chunk.free_blocks.begin()) {
            અચલ auto & prev = *(it - 1);
            જો (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                પાછા;
            }
        }

        જો (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            પાછા;
        }

        it = chunk.free_blocks.insert(it, block);

        જો (it != chunk.free_blocks.begin()) {
            auto prev = it - 1;
            જો (prev->offset + prev->size == it->offset) {
                it->offset = prev->offset;
                it->size += prev->size;
                it = chunk.free_blocks.erase(prev);
            }
        }

        જો (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<શૂન્ય *, pool_allocation> allocations_;
    size_t                                      default_chunk_size_{ 0 };
};

વર્ગ spine_mem_pool_posix final : જાહેર spine_mem_pool_manager {
  જાહેર:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  ખાનગી:
    તાર્કિક alloc_chunk(size_t min_size, size_t alignment, શૂન્ય * hint_addr, pool_chunk * chunk) override {
        (શૂન્ય) hint_addr;

        અચલ size_t alloc_alignment = std::max(alignment, કદ(શૂન્ય *));
        શૂન્ય *       base            = nullptr;
        અચલ પૂર્ણાંક    rc              = posix_memalign(&base, alloc_alignment, min_size);
        જો (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            પાછા ખોટું;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        પાછા સાચું;
    }

    શૂન્ય dealloc_chunk(pool_chunk * chunk) override {
        std::મુક્ત(chunk->base);
        clear_chunk(chunk);
    }
};

વર્ગ spine_mem_pool_transparent_hugepage final : જાહેર spine_mem_pool_manager {
  જાહેર:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  ખાનગી:
    તાર્કિક alloc_chunk(size_t min_size, size_t alignment, શૂન્ય * hint_addr, pool_chunk * chunk) override {
        (શૂન્ય) alignment;

        size_t chunk_size = 0;
        જો (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            પાછા ખોટું;
        }

        શૂન્ય * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        જો (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           errno);
            પાછા ખોટું;
        }

        જો (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, errno);
            munmap(map_addr, chunk_size);
            પાછા ખોટું;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        પાછા સાચું;
    }

    શૂન્ય dealloc_chunk(pool_chunk * chunk) override {
        જો (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

વર્ગ spine_mem_pool_hugetlb_1g final : જાહેર spine_mem_pool_manager {
  જાહેર:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  ખાનગી:
    તાર્કિક alloc_chunk(size_t min_size, size_t alignment, શૂન્ય * hint_addr, pool_chunk * chunk) override {
        (શૂન્ય) alignment;
        (શૂન્ય) hint_addr;

        size_t region_size = 0;
        જો (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            પાછા ખોટું;
        }

        અચલ પૂર્ણાંક fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        જો (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            પાછા ખોટું;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        જો (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, errno);
            close(fd);
            પાછા ખોટું;
        }

        શૂન્ય * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        જો (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            પાછા ખોટું;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        પાછા સાચું;
    }

    શૂન્ય dealloc_chunk(pool_chunk * chunk) override {
        જો (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        જો (chunk->fd >= 0) {
            જો (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

વર્ગ spine_mem_pool_shared_mem final : જાહેર spine_mem_pool_manager {
  જાહેર:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  ખાનગી:
    તાર્કિક alloc_chunk(size_t min_size, size_t alignment, શૂન્ય * hint_addr, pool_chunk * chunk) override {
        (શૂન્ય) alignment;

        જો (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            પાછા ખોટું;
        }

        જો (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            પાછા ખોટું;
        }

        અચલ પૂર્ણાંક fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        જો (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            પાછા ખોટું;
        }

        શૂન્ય * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        જો (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), errno);
            close(fd);
            પાછા ખોટું;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        પાછા સાચું;
    }

    શૂન્ય dealloc_chunk(pool_chunk * chunk) override {
        જો (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        જો (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    સ્થિર std::once_flag                          pool_once;
    સ્થિર std::unique_ptr<spine_mem_pool_manager> selected_pool;
    સ્થિર spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    જો (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        પસંદગી (selected_backend) {
            સ્થિતિ spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                તોડો;
            સ્થિતિ spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                તોડો;
            સ્થિતિ spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                તોડો;
            સ્થિતિ spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                તોડો;
        }
    });

    જો (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<પૂર્ણાંક>(backend), static_cast<પૂર્ણાંક>(selected_backend));
    }

    જો (selected_pool) {
        પાછા *selected_pool;
    }

    ફેંકો std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    સ્થિર std::once_flag                             shared_mem_pool_once;
    સ્થિર std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    જો (shared_mem_pool) {
        પાછા *shared_mem_pool;
    }

    ફેંકો std::bad_alloc();
}

}  // namespace

તાર્કિક spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    જો (info == nullptr) {
        પાછા ખોટું;
    }

    *info = {};

    જો (spine_tcm_open_handle(ખાલી) != 0 || !spine_tcm_is_available()) {
        પાછા ખોટું;
    }

    spine_tcm_mem_info_t mem_info;
    જો (spine_tcm_mem_info(&mem_info) != 0) {
        પાછા ખોટું;
    }

    info->available   = સાચું;
    info->blk_size    = mem_info.blk_size;
    info->blk_અંક     = mem_info.blk_અંક;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    પાછા સાચું;
}

શૂન્ય * spine_mem_pool_tcm_mem_get(પૂર્ણાંક cpu_id) noexcept {
    પાછા spine_tcm_mem_get(cpu_id);
}

શૂન્ય * spine_mem_pool_tcm_mem_wait(પૂર્ણાંક cpu_id) noexcept {
    પાછા spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

પૂર્ણાંક spine_mem_pool_tcm_mem_release(પૂર્ણાંક cpu_id) noexcept {
    પાછા spine_tcm_mem_release(cpu_id);
}

શૂન્ય * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    પ્રયાસ {
        પાછા get_spine_mem_pool_manager().alloc(size, alignment);
    } પકડો (અચલ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        પાછા nullptr;
    }
}

શૂન્ય * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    પ્રયાસ {
        પાછા get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } પકડો (અચલ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        પાછા nullptr;
    }
}

શૂન્ય spine_mem_pool_free(શૂન્ય * base) noexcept {
    પ્રયાસ {
        get_spine_mem_pool_manager().મુક્ત(base);
    } પકડો (અચલ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

શૂન્ય spine_mem_pool_shared_mem_free(શૂન્ય * base) noexcept {
    પ્રયાસ {
        get_spine_mem_pool_shared_mem_manager().મુક્ત(base);
    } પકડો (અચલ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

બાહ્ય "C" {
શૂન્ય * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    શૂન્ય * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    જો (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    પાછા result;
}

શૂન્ય ggml_backend_cpu_riscv64_spacemit_free_shared(શૂન્ય * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
