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

constexpr माप_प्रकार   SPINE_MEM_POOL_CHUNK_SIZE         = 512ull * 1024ull * 1024ull;
constexpr माप_प्रकार   SPINE_SHARE_MEM_POOL_CHUNK_SIZE   = 512ull * 1024ull;
constexpr माप_प्रकार   SPINE_MEM_POOL_1G_REGION_SIZE     = 1ull << 30;
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
    माप_प्रकार offset{ 0 };
    माप_प्रकार size{ 0 };
};

संरचना pool_chunk {
    uint8_t *               base{ nullptr };
    माप_प्रकार                  size{ 0 };
    पूर्णाङ्क                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

संरचना pool_allocation {
    शून्य * chunk_base{ nullptr };
    माप_प्रकार chunk_size{ 0 };
    शून्य * base{ nullptr };
    माप_प्रकार size{ 0 };
};

तार्किक is_power_of_two(माप_प्रकार value) {
    फर्काऊ value != 0 && (value & (value - 1)) == 0;
}

तार्किक align_up(माप_प्रकार value, माप_प्रकार alignment, माप_प्रकार * aligned_मान) {
    यदि (aligned_मान == nullptr || alignment == 0) {
        फर्काऊ असत्य;
    }

    अचर माप_प्रकार remainder = value % alignment;
    यदि (remainder == 0) {
        *aligned_मान = value;
        फर्काऊ सत्य;
    }

    अचर माप_प्रकार padding = alignment - remainder;
    यदि (value > std::numeric_limits<माप_प्रकार>::max() - padding) {
        फर्काऊ असत्य;
    }

    *aligned_मान = value + padding;
    फर्काऊ सत्य;
}

तार्किक align_up_uintptr(uintptr_t value, माप_प्रकार alignment, uintptr_t * aligned_मान) {
    यदि (aligned_मान == nullptr || alignment == 0) {
        फर्काऊ असत्य;
    }

    अचर uintptr_t remainder = value % alignment;
    यदि (remainder == 0) {
        *aligned_मान = value;
        फर्काऊ सत्य;
    }

    अचर uintptr_t padding = alignment - remainder;
    यदि (value > std::numeric_limits<uintptr_t>::max() - padding) {
        फर्काऊ असत्य;
    }

    *aligned_मान = value + padding;
    फर्काऊ सत्य;
}

वर्ग spine_mem_pool_manager {
  सार्वजनिक:
    explicit spine_mem_pool_manager(माप_प्रकार default_chunk_size) : default_chunk_size_(default_chunk_size) {}

    आभास ~spine_mem_pool_manager() = स्वभाव;

    शून्य * alloc(माप_प्रकार size, माप_प्रकार alignment) {
        यदि (size == 0 || !is_power_of_two(alignment)) {
            फर्काऊ nullptr;
        }

        माप_प्रकार aligned_size = 0;
        यदि (!align_up(size, alignment, &aligned_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            फर्काऊ nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        यदि (!try_alloc_locked(aligned_size, alignment, &allocation)) {
            यदि (!add_chunk_locked(aligned_size, alignment)) {
                फर्काऊ nullptr;
            }

            यदि (!try_alloc_locked(aligned_size, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_size, alignment);
                फर्काऊ nullptr;
            }
        }

        यत्न {
            अचर स्वतः [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            यदि (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                फर्काऊ nullptr;
            }
        } गृहाण (अचर std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            क्षिप;
        }

        फर्काऊ allocation.base;
    }

    शून्य मुक्त(शून्य * base) {
        यदि (base == nullptr) {
            फर्काऊ;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        स्वतः allocation_it = allocations_.find(base);
        यदि (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            फर्काऊ;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        स्वतः chunk_it = find_chunk_locked(allocation);
        यदि (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            फर्काऊ;
        }

        स्वतः * chunk_base = chunk_it->base;
        स्वतः * alloc_base = static_cast<uint8_t *>(allocation.base);
        यदि (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            फर्काऊ;
        }

        अचर माप_प्रकार offset = static_cast<माप_प्रकार>(alloc_base - chunk_base);
        यदि (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            फर्काऊ;
        }

        घुसाउ_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  रक्षित:
    शून्य release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.clear();
        लागि (स्वतः & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    माप_प्रकार default_chunk_size() अचर { फर्काऊ default_chunk_size_; }

    स्थिर शून्य clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.clear();
    }

    आभास तार्किक alloc_chunk(माप_प्रकार min_size, माप_प्रकार alignment, शून्य * hint_addr, pool_chunk * chunk) = 0;
    आभास शून्य dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  गुप्त:
    संरचना alloc_candidate {
        माप_प्रकार    chunk_index{ 0 };
        माप_प्रकार    block_index{ 0 };
        माप_प्रकार    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        तार्किक      valid{ असत्य };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(अचर pool_allocation & allocation) {
        फर्काऊ std::find_if(chunks_.begin(), chunks_.end(), [&](अचर pool_chunk & chunk) {
            फर्काऊ chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_size;
        });
    }

    तार्किक add_chunk_locked(माप_प्रकार min_size, माप_प्रकार alignment) {
        pool_chunk   chunk;
        अचर माप_प्रकार chunk_request = default_chunk_size_ == 0 ? min_size : std::max(min_size, default_chunk_size_);
        शून्य *       hint_addr     = nullptr;

        लागि (अचर स्वतः & existing_chunk : chunks_) {
            स्वतः * chunk_end = existing_chunk.base + existing_chunk.size;
            यदि (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        यदि (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            फर्काऊ असत्य;
        }

        यदि (chunk.base == nullptr || chunk.size < min_size) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_size, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            फर्काऊ असत्य;
        }

        यत्न {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } गृहाण (अचर std::bad_alloc &) {
            dealloc_chunk(&chunk);
            क्षिप;
        }

        फर्काऊ सत्य;
    }

    शून्य rollback_allocation_locked(अचर pool_allocation & allocation) {
        स्वतः chunk_it = find_chunk_locked(allocation);
        यदि (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            फर्काऊ;
        }

        स्वतः * chunk_base = chunk_it->base;
        स्वतः * alloc_base = static_cast<uint8_t *>(allocation.base);
        यदि (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            फर्काऊ;
        }

        अचर माप_प्रकार offset = static_cast<माप_प्रकार>(alloc_base - chunk_base);
        यदि (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            फर्काऊ;
        }

        घुसाउ_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    तार्किक try_alloc_locked(माप_प्रकार size, माप_प्रकार alignment, pool_allocation * allocation) {
        alloc_candidate best;

        लागि (माप_प्रकार chunk_index = 0; chunk_index < chunks_.size(); ++chunk_index) {
            अचर स्वतः & chunk = chunks_[chunk_index];
            लागि (माप_प्रकार block_index = 0; block_index < chunk.free_blocks.size(); ++block_index) {
                अचर स्वतः & block = chunk.free_blocks[block_index];

                uintptr_t  aligned_addr = 0;
                अचर स्वतः block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                यदि (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    जारी;
                }

                यदि (aligned_addr < block_addr) {
                    जारी;
                }

                अचर माप_प्रकार aligned_offset = block.offset + static_cast<माप_प्रकार>(aligned_addr - block_addr);
                अचर माप_प्रकार padding        = aligned_offset - block.offset;
                यदि (padding > block.size || size > block.size - padding) {
                    जारी;
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
            फर्काऊ असत्य;
        }

        स्वतः &           chunk     = chunks_[best.chunk_index];
        अचर free_block block     = chunk.free_blocks[best.block_index];
        अचर माप_प्रकार     padding   = best.aligned_offset - block.offset;
        अचर माप_प्रकार     alloc_end = best.aligned_offset + size;
        अचर माप_प्रकार     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_index);
        स्वतः घुसाउ_it = chunk.free_blocks.begin() + best.block_index;
        यदि (padding != 0) {
            घुसाउ_it = chunk.free_blocks.insert(घुसाउ_it, { block.offset, padding });
            ++घुसाउ_it;
        }
        यदि (alloc_end < block_end) {
            chunk.free_blocks.insert(घुसाउ_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_size = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        फर्काऊ सत्य;
    }

    शून्य maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        यदि (chunk_it->free_blocks.size() != 1) {
            फर्काऊ;
        }

        अचर स्वतः & block = chunk_it->free_blocks.front();
        यदि (block.offset != 0 || block.size != chunk_it->size) {
            फर्काऊ;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    शून्य घुसाउ_free_block_locked(pool_chunk & chunk, free_block block) {
        स्वतः it = chunk.free_blocks.begin();
        जबसम्म (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        यदि (it != chunk.free_blocks.begin()) {
            अचर स्वतः & prev = *(it - 1);
            यदि (prev.offset + prev.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                फर्काऊ;
            }
        }

        यदि (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            फर्काऊ;
        }

        it = chunk.free_blocks.insert(it, block);

        यदि (it != chunk.free_blocks.begin()) {
            स्वतः prev = it - 1;
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
    माप_प्रकार                                      default_chunk_size_{ 0 };
};

वर्ग spine_mem_pool_posix final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(माप_प्रकार min_size, माप_प्रकार alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) hint_addr;

        अचर माप_प्रकार alloc_alignment = std::max(alignment, परिमाण(शून्य *));
        शून्य *       base            = nullptr;
        अचर पूर्णाङ्क    rc              = posix_memalign(&base, alloc_alignment, min_size);
        यदि (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_size, alloc_alignment, rc);
            फर्काऊ असत्य;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_size;
        chunk->fd   = -1;
        फर्काऊ सत्य;
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
    तार्किक alloc_chunk(माप_प्रकार min_size, माप_प्रकार alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) alignment;

        माप_प्रकार chunk_size = 0;
        यदि (!align_up(min_size, default_chunk_size(), &chunk_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_size);
            फर्काऊ असत्य;
        }

        शून्य * map_addr = mmap(hint_addr, chunk_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        यदि (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_size,
                           त्रुटि_सं);
            फर्काऊ असत्य;
        }

        यदि (madvise(map_addr, chunk_size, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_size, त्रुटि_सं);
            munmap(map_addr, chunk_size);
            फर्काऊ असत्य;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_size;
        chunk->fd   = -1;
        फर्काऊ सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        यदि (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, त्रुटि_सं);
        }

        clear_chunk(chunk);
    }
};

वर्ग spine_mem_pool_hugetlb_1g final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(माप_प्रकार min_size, माप_प्रकार alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) alignment;
        (शून्य) hint_addr;

        माप_प्रकार region_size = 0;
        यदि (!align_up(min_size, SPINE_MEM_POOL_1G_REGION_SIZE, &region_size)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_size);
            फर्काऊ असत्य;
        }

        अचर पूर्णाङ्क fd = खोलो(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        यदि (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, त्रुटि_सं);
            फर्काऊ असत्य;
        }

        hugetlb_1g_region region;
        region.size  = region_size;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        यदि (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_size, त्रुटि_सं);
            बंद(fd);
            फर्काऊ असत्य;
        }

        शून्य * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        यदि (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<अचिन्हित दीर्घ दीर्घ>(region.size), त्रुटि_सं);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            बंद(fd);
            फर्काऊ असत्य;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        फर्काऊ सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        यदि (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, त्रुटि_सं);
        }

        यदि (chunk->fd >= 0) {
            यदि (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, त्रुटि_सं);
            }

            बंद(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

वर्ग spine_mem_pool_shared_mem final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(माप_प्रकार min_size, माप_प्रकार alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) alignment;

        यदि (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            फर्काऊ असत्य;
        }

        यदि (min_size > default_chunk_size()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_size, default_chunk_size());
            फर्काऊ असत्य;
        }

        अचर पूर्णाङ्क fd = खोलो(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        यदि (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, त्रुटि_सं);
            फर्काऊ असत्य;
        }

        शून्य * map_addr = mmap(nullptr, default_chunk_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        यदि (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_size(), त्रुटि_सं);
            बंद(fd);
            फर्काऊ असत्य;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_size();
        chunk->fd   = fd;
        फर्काऊ सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        यदि (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, त्रुटि_सं);
        }

        यदि (chunk->fd >= 0) {
            बंद(chunk->fd);
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
                रोक;
            पक्ष spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                रोक;
            पक्ष spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                रोक;
            पक्ष spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                रोक;
        }
    });

    यदि (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<पूर्णाङ्क>(backend), static_cast<पूर्णाङ्क>(selected_backend));
    }

    यदि (selected_pool) {
        फर्काऊ *selected_pool;
    }

    क्षिप std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    स्थिर std::once_flag                             shared_mem_pool_once;
    स्थिर std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    यदि (shared_mem_pool) {
        फर्काऊ *shared_mem_pool;
    }

    क्षिप std::bad_alloc();
}

}  // namespace

तार्किक spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    यदि (info == nullptr) {
        फर्काऊ असत्य;
    }

    *info = {};

    यदि (spine_tcm_open_handle(रिक्त) != 0 || !spine_tcm_is_available()) {
        फर्काऊ असत्य;
    }

    spine_tcm_mem_info_t mem_info;
    यदि (spine_tcm_mem_info(&mem_info) != 0) {
        फर्काऊ असत्य;
    }

    info->available   = सत्य;
    info->blk_size    = mem_info.blk_size;
    info->blk_अङ्क     = mem_info.blk_अङ्क;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    फर्काऊ सत्य;
}

शून्य * spine_mem_pool_tcm_mem_get(पूर्णाङ्क cpu_id) noexcept {
    फर्काऊ spine_tcm_mem_get(cpu_id);
}

शून्य * spine_mem_pool_tcm_mem_wait(पूर्णाङ्क cpu_id) noexcept {
    फर्काऊ spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

पूर्णाङ्क spine_mem_pool_tcm_mem_release(पूर्णाङ्क cpu_id) noexcept {
    फर्काऊ spine_tcm_mem_release(cpu_id);
}

शून्य * spine_mem_pool_alloc(माप_प्रकार size, माप_प्रकार alignment) noexcept {
    यत्न {
        फर्काऊ get_spine_mem_pool_manager().alloc(size, alignment);
    } गृहाण (अचर std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        फर्काऊ nullptr;
    }
}

शून्य * spine_mem_pool_shared_mem_alloc(माप_प्रकार size, माप_प्रकार alignment) noexcept {
    यत्न {
        फर्काऊ get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } गृहाण (अचर std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        फर्काऊ nullptr;
    }
}

शून्य spine_mem_pool_free(शून्य * base) noexcept {
    यत्न {
        get_spine_mem_pool_manager().मुक्त(base);
    } गृहाण (अचर std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

शून्य spine_mem_pool_shared_mem_free(शून्य * base) noexcept {
    यत्न {
        get_spine_mem_pool_shared_mem_manager().मुक्त(base);
    } गृहाण (अचर std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

बाह्य "C" {
शून्य * ggml_backend_cpu_riscv64_spacemit_alloc_shared(माप_प्रकार size, माप_प्रकार alignment) {
    शून्य * result = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    यदि (result == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    फर्काऊ result;
}

शून्य ggml_backend_cpu_riscv64_spacemit_free_shared(शून्य * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
