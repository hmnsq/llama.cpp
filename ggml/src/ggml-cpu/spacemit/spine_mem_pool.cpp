#समावेश "spine_mem_pool.h"

#समावेश "common.h"
#समावेश "ime_env.h"
#समावेश "spine_tcm.h"

#समावेश <fcntl.h>
#समावेश <sys/ioctl.h>
#समावेश <sys/mman.h>
#समावेश <unistd.h>

#समावेश <algorithm>
#समावेश <cerrno>
#समावेश <cstdint>
#समावेश <cstdlib>
#समावेश <limits>
#समावेश <memory>
#समावेश <mutex>
#समावेश <unordered_map>
#समावेश <vector>

namespace ggml::cpu::riscv64_spacemit {
namespace {

constexpr माप_प्रकार   SPINE_MEM_POOL_CHUNK_आकार         = 512ull * 1024ull * 1024ull;
constexpr माप_प्रकार   SPINE_SHARE_MEM_POOL_CHUNK_आकार   = 512ull * 1024ull;
constexpr माप_प्रकार   SPINE_MEM_POOL_1_G_REGION_आकार     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr अक्षर     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr अक्षर     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

काष्ठा hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#घोषणा HUGETLB_1G_IOC_MAGIC 'M'
#घोषणा HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, काष्ठा hugetlb_1g_region)
#घोषणा HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

काष्ठा free_block {
    माप_प्रकार offset{ 0 };
    माप_प्रकार size{ 0 };
};

काष्ठा pool_chunk {
    uint8_t *               base{ nullptr };
    माप_प्रकार                  size{ 0 };
    पूर्णांक                     fd{ -1 };
    std::vector<free_block> free_blocks;
};

काष्ठा pool_allocation {
    व्योम * chunk_base{ nullptr };
    माप_प्रकार chunk_आकार{ 0 };
    व्योम * base{ nullptr };
    माप_प्रकार size{ 0 };
};

bool is_power_of_two(माप_प्रकार value) {
    वापस value != 0 && (value & (value - 1)) == 0;
}

bool align_up(माप_प्रकार value, माप_प्रकार alignment, माप_प्रकार * aligned_मूल्य) {
    अगर (aligned_मूल्य == nullptr || alignment == 0) {
        वापस false;
    }

    स्थिर माप_प्रकार remainder = value % alignment;
    अगर (remainder == 0) {
        *aligned_मूल्य = value;
        वापस true;
    }

    स्थिर माप_प्रकार padding = alignment - remainder;
    अगर (value > std::numeric_limits<माप_प्रकार>::max() - padding) {
        वापस false;
    }

    *aligned_मूल्य = value + padding;
    वापस true;
}

bool align_up_uintptr(uintptr_t value, माप_प्रकार alignment, uintptr_t * aligned_मूल्य) {
    अगर (aligned_मूल्य == nullptr || alignment == 0) {
        वापस false;
    }

    स्थिर uintptr_t remainder = value % alignment;
    अगर (remainder == 0) {
        *aligned_मूल्य = value;
        वापस true;
    }

    स्थिर uintptr_t padding = alignment - remainder;
    अगर (value > std::numeric_limits<uintptr_t>::max() - padding) {
        वापस false;
    }

    *aligned_मूल्य = value + padding;
    वापस true;
}

श्रेणी spine_mem_pool_manager {
  खुला:
    explicit spine_mem_pool_manager(माप_प्रकार default_chunk_आकार) : default_chunk_आकार_2(default_chunk_आकार) {}

    भव ~spine_mem_pool_manager() = शेष;

    व्योम * alloc(माप_प्रकार size, माप_प्रकार alignment) {
        अगर (size == 0 || !is_power_of_two(alignment)) {
            वापस nullptr;
        }

        माप_प्रकार aligned_आकार = 0;
        अगर (!align_up(size, alignment, &aligned_आकार)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            वापस nullptr;
        }

        pool_allocation allocation;

        std::lock_guard<std::mutex> lock(mutex_);

        अगर (!try_alloc_locked(aligned_आकार, alignment, &allocation)) {
            अगर (!add_chunk_locked(aligned_आकार, alignment)) {
                वापस nullptr;
            }

            अगर (!try_alloc_locked(aligned_आकार, alignment, &allocation)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_आकार, alignment);
                वापस nullptr;
            }
        }

        प्रयास {
            स्थिर स्वतः [allocation_it, inserted] = allocations_.emplace(allocation.base, allocation);
            अगर (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, allocation.base);
                rollback_allocation_locked(allocation);
                वापस nullptr;
            }
        } पकड़ो (स्थिर std::bad_alloc &) {
            rollback_allocation_locked(allocation);
            फेंको;
        }

        वापस allocation.base;
    }

    व्योम मुक्त(व्योम * base) {
        अगर (base == nullptr) {
            वापस;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        स्वतः allocation_it = allocations_.find(base);
        अगर (allocation_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, base);
            वापस;
        }

        pool_allocation allocation = allocation_it->second;
        allocations_.erase(allocation_it);

        स्वतः chunk_it = find_chunk_locked(allocation);
        अगर (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            वापस;
        }

        स्वतः * chunk_base = chunk_it->base;
        स्वतः * alloc_base = static_cast<uint8_t *>(allocation.base);
        अगर (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           allocation.base, chunk_base, chunk_base + chunk_it->size);
            वापस;
        }

        स्थिर माप_प्रकार offset = static_cast<माप_प्रकार>(alloc_base - chunk_base);
        अगर (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           allocation.base, allocation.size, chunk_it->size);
            वापस;
        }

        जोड़ो_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

  रक्षित:
    व्योम release_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);

        allocations_.सफा();
        क्रम (स्वतः & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.सफा();
    }

    माप_प्रकार default_chunk_आकार() स्थिर { वापस default_chunk_आकार_2; }

    अटल व्योम clear_chunk(pool_chunk * chunk) {
        chunk->base = nullptr;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_blocks.सफा();
    }

    भव bool alloc_chunk(माप_प्रकार min_आकार, माप_प्रकार alignment, व्योम * hint_addr, pool_chunk * chunk) = 0;
    भव व्योम dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  निजी:
    काष्ठा alloc_candidate {
        माप_प्रकार    chunk_सूचकांक{ 0 };
        माप_प्रकार    block_सूचकांक{ 0 };
        माप_प्रकार    aligned_offset{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        bool      valid{ false };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(स्थिर pool_allocation & allocation) {
        वापस std::find_if(chunks_.begin(), chunks_.end(), [&](स्थिर pool_chunk & chunk) {
            वापस chunk.base == allocation.chunk_base && chunk.size == allocation.chunk_आकार;
        });
    }

    bool add_chunk_locked(माप_प्रकार min_आकार, माप_प्रकार alignment) {
        pool_chunk   chunk;
        स्थिर माप_प्रकार chunk_request = default_chunk_आकार_2 == 0 ? min_आकार : std::max(min_आकार, default_chunk_आकार_2);
        व्योम *       hint_addr     = nullptr;

        क्रम (स्थिर स्वतः & existing_chunk : chunks_) {
            स्वतः * chunk_end = existing_chunk.base + existing_chunk.size;
            अगर (hint_addr == nullptr || chunk_end > hint_addr) {
                hint_addr = chunk_end;
            }
        }

        अगर (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            वापस false;
        }

        अगर (chunk.base == nullptr || chunk.size < min_आकार) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_आकार, chunk.base, chunk.size);
            dealloc_chunk(&chunk);
            वापस false;
        }

        प्रयास {
            chunk.free_blocks.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } पकड़ो (स्थिर std::bad_alloc &) {
            dealloc_chunk(&chunk);
            फेंको;
        }

        वापस true;
    }

    व्योम rollback_allocation_locked(स्थिर pool_allocation & allocation) {
        स्वतः chunk_it = find_chunk_locked(allocation);
        अगर (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, allocation.base);
            वापस;
        }

        स्वतः * chunk_base = chunk_it->base;
        स्वतः * alloc_base = static_cast<uint8_t *>(allocation.base);
        अगर (alloc_base < chunk_base || alloc_base >= chunk_base + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, allocation.base);
            वापस;
        }

        स्थिर माप_प्रकार offset = static_cast<माप_प्रकार>(alloc_base - chunk_base);
        अगर (offset > chunk_it->size || allocation.size > chunk_it->size - offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           allocation.base, allocation.size);
            वापस;
        }

        जोड़ो_free_block_locked(*chunk_it, { offset, allocation.size });
        maybe_release_empty_chunk_locked(chunk_it);
    }

    bool try_alloc_locked(माप_प्रकार size, माप_प्रकार alignment, pool_allocation * allocation) {
        alloc_candidate best;

        क्रम (माप_प्रकार chunk_सूचकांक = 0; chunk_सूचकांक < chunks_.size(); ++chunk_सूचकांक) {
            स्थिर स्वतः & chunk = chunks_[chunk_सूचकांक];
            क्रम (माप_प्रकार block_सूचकांक = 0; block_सूचकांक < chunk.free_blocks.size(); ++block_सूचकांक) {
                स्थिर स्वतः & block = chunk.free_blocks[block_सूचकांक];

                uintptr_t  aligned_addr = 0;
                स्थिर स्वतः block_addr   = reinterpret_cast<uintptr_t>(chunk.base + block.offset);
                अगर (!align_up_uintptr(block_addr, alignment, &aligned_addr)) {
                    जारी;
                }

                अगर (aligned_addr < block_addr) {
                    जारी;
                }

                स्थिर माप_प्रकार aligned_offset = block.offset + static_cast<माप_प्रकार>(aligned_addr - block_addr);
                स्थिर माप_प्रकार padding        = aligned_offset - block.offset;
                अगर (padding > block.size || size > block.size - padding) {
                    जारी;
                }

                अगर (!best.valid || aligned_addr < best.address) {
                    best.chunk_सूचकांक    = chunk_सूचकांक;
                    best.block_सूचकांक    = block_सूचकांक;
                    best.aligned_offset = aligned_offset;
                    best.address        = aligned_addr;
                    best.valid          = true;
                }
            }
        }

        अगर (!best.valid) {
            वापस false;
        }

        स्वतः &           chunk     = chunks_[best.chunk_सूचकांक];
        स्थिर free_block block     = chunk.free_blocks[best.block_सूचकांक];
        स्थिर माप_प्रकार     padding   = best.aligned_offset - block.offset;
        स्थिर माप_प्रकार     alloc_end = best.aligned_offset + size;
        स्थिर माप_प्रकार     block_end = block.offset + block.size;

        chunk.free_blocks.erase(chunk.free_blocks.begin() + best.block_सूचकांक);
        स्वतः जोड़ो_it = chunk.free_blocks.begin() + best.block_सूचकांक;
        अगर (padding != 0) {
            जोड़ो_it = chunk.free_blocks.insert(जोड़ो_it, { block.offset, padding });
            ++जोड़ो_it;
        }
        अगर (alloc_end < block_end) {
            chunk.free_blocks.insert(जोड़ो_it, { alloc_end, block_end - alloc_end });
        }

        allocation->chunk_base = chunk.base;
        allocation->chunk_आकार = chunk.size;
        allocation->base       = chunk.base + best.aligned_offset;
        allocation->size       = size;
        वापस true;
    }

    व्योम maybe_release_empty_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        अगर (chunk_it->free_blocks.size() != 1) {
            वापस;
        }

        स्थिर स्वतः & block = chunk_it->free_blocks.front();
        अगर (block.offset != 0 || block.size != chunk_it->size) {
            वापस;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    व्योम जोड़ो_free_block_locked(pool_chunk & chunk, free_block block) {
        स्वतः it = chunk.free_blocks.begin();
        जबतक (it != chunk.free_blocks.end() && it->offset < block.offset) {
            ++it;
        }

        अगर (it != chunk.free_blocks.begin()) {
            स्थिर स्वतः & पूर्व = *(it - 1);
            अगर (पूर्व.offset + पूर्व.size > block.offset) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               block.offset, block.size);
                वापस;
            }
        }

        अगर (it != chunk.free_blocks.end() && block.offset + block.size > it->offset) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           block.offset, block.size);
            वापस;
        }

        it = chunk.free_blocks.insert(it, block);

        अगर (it != chunk.free_blocks.begin()) {
            स्वतः पूर्व = it - 1;
            अगर (पूर्व->offset + पूर्व->size == it->offset) {
                it->offset = पूर्व->offset;
                it->size += पूर्व->size;
                it = chunk.free_blocks.erase(पूर्व);
            }
        }

        अगर (it + 1 != chunk.free_blocks.end() && it->offset + it->size == (it + 1)->offset) {
            it->size += (it + 1)->size;
            chunk.free_blocks.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<व्योम *, pool_allocation> allocations_;
    माप_प्रकार                                      default_chunk_आकार_2{ 0 };
};

श्रेणी spine_mem_pool_posix final : खुला spine_mem_pool_manager {
  खुला:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  निजी:
    bool alloc_chunk(माप_प्रकार min_आकार, माप_प्रकार alignment, व्योम * hint_addr, pool_chunk * chunk) override {
        (व्योम) hint_addr;

        स्थिर माप_प्रकार alloc_alignment = std::max(alignment, माप(व्योम *));
        व्योम *       base            = nullptr;
        स्थिर पूर्णांक    rc              = posix_memalign(&base, alloc_alignment, min_आकार);
        अगर (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_आकार, alloc_alignment, rc);
            वापस false;
        }

        chunk->base = static_cast<uint8_t *>(base);
        chunk->size = min_आकार;
        chunk->fd   = -1;
        वापस true;
    }

    व्योम dealloc_chunk(pool_chunk * chunk) override {
        std::मुक्त(chunk->base);
        clear_chunk(chunk);
    }
};

श्रेणी spine_mem_pool_transparent_hugepage final : खुला spine_mem_pool_manager {
  खुला:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_आकार) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  निजी:
    bool alloc_chunk(माप_प्रकार min_आकार, माप_प्रकार alignment, व्योम * hint_addr, pool_chunk * chunk) override {
        (व्योम) alignment;

        माप_प्रकार chunk_आकार = 0;
        अगर (!align_up(min_आकार, default_chunk_आकार(), &chunk_आकार)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_आकार);
            वापस false;
        }

        व्योम * map_addr = mmap(hint_addr, chunk_आकार, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        अगर (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_आकार,
                           त्रुटि_सं);
            वापस false;
        }

        अगर (madvise(map_addr, chunk_आकार, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_आकार, त्रुटि_सं);
            munmap(map_addr, chunk_आकार);
            वापस false;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_आकार;
        chunk->fd   = -1;
        वापस true;
    }

    व्योम dealloc_chunk(pool_chunk * chunk) override {
        अगर (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->base, chunk->size, त्रुटि_सं);
        }

        clear_chunk(chunk);
    }
};

श्रेणी spine_mem_pool_hugetlb_1g final : खुला spine_mem_pool_manager {
  खुला:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1_G_REGION_आकार) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  निजी:
    bool alloc_chunk(माप_प्रकार min_आकार, माप_प्रकार alignment, व्योम * hint_addr, pool_chunk * chunk) override {
        (व्योम) alignment;
        (व्योम) hint_addr;

        माप_प्रकार region_आकार = 0;
        अगर (!align_up(min_आकार, SPINE_MEM_POOL_1_G_REGION_आकार, &region_आकार)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_आकार);
            वापस false;
        }

        स्थिर पूर्णांक fd = खोलो(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        अगर (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, त्रुटि_सं);
            वापस false;
        }

        hugetlb_1g_region region;
        region.size  = region_आकार;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        अगर (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_आकार, त्रुटि_सं);
            बंद(fd);
            वापस false;
        }

        व्योम * map_addr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        अगर (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<अचिन्हित दीर्घ दीर्घ>(region.size), त्रुटि_सं);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            बंद(fd);
            वापस false;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        वापस true;
    }

    व्योम dealloc_chunk(pool_chunk * chunk) override {
        अगर (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, त्रुटि_सं);
        }

        अगर (chunk->fd >= 0) {
            अगर (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->base, त्रुटि_सं);
            }

            बंद(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

श्रेणी spine_mem_pool_shared_mem final : खुला spine_mem_pool_manager {
  खुला:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_आकार) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  निजी:
    bool alloc_chunk(माप_प्रकार min_आकार, माप_प्रकार alignment, व्योम * hint_addr, pool_chunk * chunk) override {
        (व्योम) alignment;

        अगर (hint_addr != nullptr) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            वापस false;
        }

        अगर (min_आकार > default_chunk_आकार()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_आकार, default_chunk_आकार());
            वापस false;
        }

        स्थिर पूर्णांक fd = खोलो(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        अगर (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, त्रुटि_सं);
            वापस false;
        }

        व्योम * map_addr = mmap(nullptr, default_chunk_आकार(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        अगर (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_आकार(), त्रुटि_सं);
            बंद(fd);
            वापस false;
        }

        chunk->base = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_आकार();
        chunk->fd   = fd;
        वापस true;
    }

    व्योम dealloc_chunk(pool_chunk * chunk) override {
        अगर (chunk->base != nullptr && chunk->size != 0 && munmap(chunk->base, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->base, chunk->size, त्रुटि_सं);
        }

        अगर (chunk->fd >= 0) {
            बंद(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    अटल std::once_flag                          pool_once;
    अटल std::unique_ptr<spine_mem_pool_manager> selected_pool;
    अटल spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    अगर (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        चयन (selected_backend) {
            हाल spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                अवरोध;
            हाल spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                अवरोध;
            हाल spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                अवरोध;
            हाल spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                अवरोध;
        }
    });

    अगर (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<पूर्णांक>(backend), static_cast<पूर्णांक>(selected_backend));
    }

    अगर (selected_pool) {
        वापस *selected_pool;
    }

    फेंको std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    अटल std::once_flag                             shared_mem_pool_once;
    अटल std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    अगर (shared_mem_pool) {
        वापस *shared_mem_pool;
    }

    फेंको std::bad_alloc();
}

}  // namespace

bool spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    अगर (info == nullptr) {
        वापस false;
    }

    *info = {};

    अगर (spine_tcm_open_handle(शून्य) != 0 || !spine_tcm_is_available()) {
        वापस false;
    }

    spine_tcm_mem_info_t mem_info;
    अगर (spine_tcm_mem_info(&mem_info) != 0) {
        वापस false;
    }

    info->available   = true;
    info->blk_आकार    = mem_info.blk_आकार;
    info->blk_अंक     = mem_info.blk_अंक;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    वापस true;
}

व्योम * spine_mem_pool_tcm_mem_get(पूर्णांक cpu_id) noexcept {
    वापस spine_tcm_mem_get(cpu_id);
}

व्योम * spine_mem_pool_tcm_mem_wait(पूर्णांक cpu_id) noexcept {
    वापस spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

पूर्णांक spine_mem_pool_tcm_mem_release(पूर्णांक cpu_id) noexcept {
    वापस spine_tcm_mem_release(cpu_id);
}

व्योम * spine_mem_pool_alloc(माप_प्रकार size, माप_प्रकार alignment) noexcept {
    प्रयास {
        वापस get_spine_mem_pool_manager().alloc(size, alignment);
    } पकड़ो (स्थिर std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        वापस nullptr;
    }
}

व्योम * spine_mem_pool_shared_mem_alloc(माप_प्रकार size, माप_प्रकार alignment) noexcept {
    प्रयास {
        वापस get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } पकड़ो (स्थिर std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        वापस nullptr;
    }
}

व्योम spine_mem_pool_free(व्योम * base) noexcept {
    प्रयास {
        get_spine_mem_pool_manager().मुक्त(base);
    } पकड़ो (स्थिर std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, base);
    }
}

व्योम spine_mem_pool_shared_mem_free(व्योम * base) noexcept {
    प्रयास {
        get_spine_mem_pool_shared_mem_manager().मुक्त(base);
    } पकड़ो (स्थिर std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, base);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

बाह्य "C" {
व्योम * ggml_backend_cpu_riscv64_spacemit_alloc_shared(माप_प्रकार size, माप_प्रकार alignment) {
    व्योम * परिणाम = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    अगर (परिणाम == nullptr) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    वापस परिणाम;
}

व्योम ggml_backend_cpu_riscv64_spacemit_free_shared(व्योम * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
