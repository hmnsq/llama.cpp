#अन्तर्भाव "spine_mem_pool.h"

#अन्तर्भाव "common.h"
#अन्तर्भाव "ime_env.h"
#अन्तर्भाव "spine_tcm.h"

#अन्तर्भाव <fcntl.h>
#अन्तर्भाव <sys/ioctl.h>
#अन्तर्भाव <sys/mman.h>
#अन्तर्भाव <unistd.h>

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cerrno>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <limits>
#अन्तर्भाव <memory>
#अन्तर्भाव <mutex>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <vector>

नामाकाश ggml::cpu::riscv64_spacemit {
नामाकाश {

constexpr size_t   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr वर्ण     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr वर्ण     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

संरचना hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, संरचना hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

संरचना free_block {
    size_t offset{ 0 };
    size_t size{ 0 };
};

संरचना pool_chunk {
    uint8_t *               base{ nullptr };
    size_t                  size{ 0 };
    पूर्णाङ्क                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

संरचना pool_allocation {
    शून्य * chunk_base{ nullptr };
    size_t chunk_size{ 0 };
    शून्य * base{ nullptr };
    size_t size{ 0 };
};

तार्किक is_power_of_two(size_t value) {
    प्रत्यागम value != 0 && (value & (value - 1)) == 0;
}

तार्किक align_up(size_t value, size_t alignment, size_t * aligned_मूल्य) {
    यदि (aligned_मूल्य == nullptr || alignment == 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव size_t remainder = value % alignment;
    यदि (remainder == 0) {
        *aligned_मूल्य = value;
        प्रत्यागम सत्य;
    }

    ध्रुव size_t padding = alignment - remainder;
    यदि (value > std::numeric_limits<size_t>::max() - padding) {
        प्रत्यागम असत्य;
    }

    *aligned_मूल्य = value + padding;
    प्रत्यागम सत्य;
}

तार्किक align_up_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_मूल्य) {
    यदि (aligned_मूल्य == nullptr || alignment == 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव uintptr_t remainder = value % alignment;
    यदि (remainder == 0) {
        *aligned_मूल्य = value;
        प्रत्यागम सत्य;
    }

    ध्रुव uintptr_t padding = alignment - remainder;
    यदि (value > std::numeric_limits<uintptr_t>::max() - padding) {
        प्रत्यागम असत्य;
    }

    *aligned_मूल्य = value + padding;
    प्रत्यागम सत्य;
}

वर्ग spine_mem_pool_manager {
  सार्वजनिक:
    explicit spine_mem_pool_manager(size_t default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    आभास ~spine_mem_pool_manager() = स्वभाव;

    शून्य * alloc(size_t size, size_t alignment) {
        यदि (size == 0 || !is_power_of_two(alignment)) {
            प्रत्यागम nullptr;
        }

        size_t aligned_size = 0;
        यदि (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            प्रत्यागम nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        यदि (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            यदि (!add_chunk_locked(aligned_size, alignment)) {
                प्रत्यागम nullptr;
            }

            यदि (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                प्रत्यागम nullptr;
            }
        }

        यत्न {
            ध्रुव auto [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            यदि (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                प्रत्यागम nullptr;
            }
        } गृहाण (ध्रुव std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            क्षिप;
        }

        प्रत्यागम allocation.base;
    }

    शून्य मुक्त(शून्य * base) {
        यदि (base == nullptr) {
            प्रत्यागम;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto allocation_it = allocations_.find(base);
        यदि (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            प्रत्यागम;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        auto chunk_it = find_chunk_locked(allocation);
        यदि (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            प्रत्यागम;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        यदि (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            प्रत्यागम;
        }

        ध्रुव size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        यदि (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            प्रत्यागम;
        }

        जोड़ो_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  रक्षित:
    शून्य release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        प्रति (auto & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_size() ध्रुव { प्रत्यागम default_chunk_size_; }

    स्थिर शून्य clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    आभास तार्किक alloc_chunk(size_t min_size, size_t alignment, शून्य * hint_addr, pool_chunk * chunk) = 0;
    आभास शून्य dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  गुप्त:
    संरचना alloc_candidate {
        size_t    chunk_index{ 0 };
        size_t    block_index{ 0 };
        size_t    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        तार्किक      valid{ असत्य };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(ध्रुव pool_allocation & allocation) {
        प्रत्यागम std::find_if(chunks_.begin(), chunks_.end(), [&](ध्रुव pool_chunk & chunk) {
            प्रत्यागम chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    तार्किक add_chunk_locked(size_t min_size, size_t alignment) {
        pool_chunk   chunk;
        ध्रुव size_t chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        शून्य *       hint_addr     = nullptr;

        प्रति (ध्रुव auto & existing_chunk : chunks_) {
            auto * chunk_end = existing_chunk.base + existing_chunk.size;
            यदि (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        यदि (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            प्रत्यागम असत्य;
        }

        यदि (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            प्रत्यागम असत्य;
        }

        यत्न {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } गृहाण (ध्रुव std::bad_alloc &) {
            dealloc_chunk(&chunk);
            क्षिप;
        }

        प्रत्यागम सत्य;
    }

    शून्य rollback_allocation_locked(ध्रुव pool_allocation & allocation) {
        auto chunk_it = find_chunk_locked(allocation);
        यदि (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            प्रत्यागम;
        }

        auto * chunk_base = chunk_it->base;
        auto * alloc_base = static_cast<uint8_t *>(allocation.base);
        यदि (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            प्रत्यागम;
        }

        ध्रुव size_t offset = static_cast<size_t>(alloc_base - chunk_base);
        यदि (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            प्रत्यागम;
        }

        जोड़ो_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    तार्किक try_alloc_locked(size_t size, size_t alignment, pool_allocation * allocation) {
        alloc_candidate best;

        प्रति (size_t chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            ध्रुव auto & chunk = chunks_[chunk_index];
            प्रति (size_t block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                ध्रुव auto & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                ध्रुव auto block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                यदि (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    अनुवर्त;
                }

                यदि (aligned_addr < block_addr) {
                    अनुवर्त;
                }

                ध्रुव size_t aligned_offset = block.offset + static_cast<size_t>(aligned_addr - block_addr);
                ध्रुव size_t padding        = aligned_offset - block.offset;
                यदि (padding > block.size || size > block.size - padding) {
                    अनुवर्त;
                }

                यदि (!best.valid || aligned_addr < best.address) {
                    best.chunk_index    = chunk_index;
                    best.block_index    = block_index;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = सत्य;
                }
            }
        }

        यदि (!best.valid) {
            प्रत्यागम असत्य;
        }

        auto &           chunk     = chunks_[best.chunk_index];
        ध्रुव free_block block     = chunk.free_blocks[best.block_index];
        ध्रुव size_t     padding   = best.aligned_offset - block.offset;
        ध्रुव size_t     alloc_end = best.aligned_offset + size;
        ध्रुव size_t     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        auto जोड़ो_it = chunk.free_blocks.begin() + best.block_index;
        यदि (padding != 0) {
            जोड़ो_it = chunk.free_blocks.insert(जोड़ो_it, { block.offset, padding });
            ++जोड़ो_it;
        }
        यदि (alloc_end < block_end) {
            chunk.free_blocks.insert(जोड़ो_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        प्रत्यागम सत्य;
    }

    शून्य maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        यदि (chunk_it->free_blocks.size() != 1) {
            प्रत्यागम;
        }

        ध्रुव auto & block = chunk_it->free_blocks.front();
        यदि (block.offset != 0 || block.size != chunk_it->size) {
            प्रत्यागम;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    शून्य जोड़ो_free_block_locked(pool_chunk & chunk, free_block block) {
        auto it = chunk.free_blocks.begin();
        यावत् (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        यदि (it != chunk.free_blocks.begin()) {
            ध्रुव auto & prev = *(it - 1);
            यदि (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                प्रत्यागम;
            }
        }

        यदि (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            प्रत्यागम;
        }

        it = chunk.free_blocks.insert(it, block);

        यदि (it != chunk.free_blocks.begin()) {
            auto prev = it - 1;
            यदि (prev->offset + prev->size == it->offset) {
                it->offset = prev->offset;
                it->size += prev->size;
                it = chunk.free_blocks.erase(prev);
            }
        }

        यदि (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<शून्य *, pool_allocation> allocations_;
    size_t                                      default_chunk_size_{ 0 };
};

वर्ग spine_mem_pool_posix final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(size_t min_size, size_t alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) hint_addr;

        ध्रुव size_t alloc_alignment = std::max(alignment, परिमाण(शून्य *));
        शून्य *       base            = nullptr;
        ध्रुव पूर्णाङ्क    rc              = posix_memalign(&base, alloc_alignment, min_size);
        यदि (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            प्रत्यागम असत्य;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        प्रत्यागम सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        std::मुक्त(chunk->base);
        clear_chunk(chunk);
    }
};

वर्ग spine_mem_pool_transparent_hugepage final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(size_t min_size, size_t alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) alignment;

        size_t chunk_size = 0;
        यदि (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            प्रत्यागम असत्य;
        }

        शून्य * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        यदि (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           errno);
            प्रत्यागम असत्य;
        }

        यदि (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, errno);
            munmap(map_addr, chunk_size);
            प्रत्यागम असत्य;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        प्रत्यागम सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        यदि (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

वर्ग spine_mem_pool_hugetlb_1g final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(size_t min_size, size_t alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) alignment;
        (शून्य) hint_addr;

        size_t region_size = 0;
        यदि (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        यदि (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            प्रत्यागम असत्य;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        यदि (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, errno);
            close(fd);
            प्रत्यागम असत्य;
        }

        शून्य * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        यदि (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<unsigned long long>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            प्रत्यागम असत्य;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        प्रत्यागम सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        यदि (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        यदि (chunk->fd >= 0) {
            यदि (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

वर्ग spine_mem_pool_shared_mem final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(size_t min_size, size_t alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) alignment;

        यदि (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            प्रत्यागम असत्य;
        }

        यदि (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        यदि (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            प्रत्यागम असत्य;
        }

        शून्य * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        यदि (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), errno);
            close(fd);
            प्रत्यागम असत्य;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        प्रत्यागम सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        यदि (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, errno);
        }

        यदि (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    स्थिर std::once_flag                          pool_once;
    स्थिर std::unique_ptr<spine_mem_pool_manager> selected_pool;
    स्थिर spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    यदि (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        विकल्प (selected_backend) {
            पक्ष spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                भङ्ग;
            पक्ष spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                भङ्ग;
            पक्ष spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                भङ्ग;
            पक्ष spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                भङ्ग;
        }
    });

    यदि (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<पूर्णाङ्क>(backend), static_cast<पूर्णाङ्क>(selected_backend));
    }

    यदि (selected_pool) {
        प्रत्यागम *selected_pool;
    }

    क्षिप std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    स्थिर std::once_flag                             shared_mem_pool_once;
    स्थिर std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    यदि (shared_mem_pool) {
        प्रत्यागम *shared_mem_pool;
    }

    क्षिप std::bad_alloc();
}

}  // namespace

तार्किक spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    यदि (info == nullptr) {
        प्रत्यागम असत्य;
    }

    *info = {};

    यदि (spine_tcm_open_handle(रिक्त) != 0 || !spine_tcm_is_available()) {
        प्रत्यागम असत्य;
    }

    spine_tcm_mem_info_t mem_info;
    यदि (spine_tcm_mem_info(&mem_info) != 0) {
        प्रत्यागम असत्य;
    }

    info->available   = सत्य;
    info->blk_size    = mem_info.blk_size;
    info->blk_अंक     = mem_info.blk_अंक;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    प्रत्यागम सत्य;
}

शून्य * spine_mem_pool_tcm_mem_get(पूर्णाङ्क cpu_id) noexcept {
    प्रत्यागम spine_tcm_mem_get(cpu_id);
}

शून्य * spine_mem_pool_tcm_mem_wait(पूर्णाङ्क cpu_id) noexcept {
    प्रत्यागम spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

पूर्णाङ्क spine_mem_pool_tcm_mem_release(पूर्णाङ्क cpu_id) noexcept {
    प्रत्यागम spine_tcm_mem_release(cpu_id);
}

शून्य * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    यत्न {
        प्रत्यागम get_spine_mem_pool_manager().alloc(size, alignment);
    } गृहाण (ध्रुव std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        प्रत्यागम nullptr;
    }
}

शून्य * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    यत्न {
        प्रत्यागम get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } गृहाण (ध्रुव std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        प्रत्यागम nullptr;
    }
}

शून्य spine_mem_pool_free(शून्य * base) noexcept {
    यत्न {
        get_spine_mem_pool_manager().मुक्त(base);
    } गृहाण (ध्रुव std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

शून्य spine_mem_pool_shared_mem_free(शून्य * base) noexcept {
    यत्न {
        get_spine_mem_pool_shared_mem_manager().मुक्त(base);
    } गृहाण (ध्रुव std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

बाह्य "C" {
शून्य * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    शून्य * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    यदि (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    प्रत्यागम result;
}

शून्य ggml_backend_cpu_riscv64_spacemit_free_shared(शून्य * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
