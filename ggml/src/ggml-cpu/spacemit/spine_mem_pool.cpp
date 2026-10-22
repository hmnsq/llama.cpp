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

constexpr size_t   SPINE_MEM_POOL_CHUNK_आकार         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_आकार   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1_G_REGION_आकार     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr वर्ण     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr वर्ण     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

संरचना hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#परिभाषय HUGETLB_1G_IOC_MAGIC 'M'
#परिभाषय HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, संरचना hugetlb_1g_region)
#परिभाषय HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

संरचना free_खण्डम् {
    size_t विस्थापनम्{ 0 };
    size_t size{ 0 };
};

संरचना pool_chunk {
    uint8_t *               आधारः{ शून्यसूचकः };
    size_t                  size{ 0 };
    पूर्णाङ्क                     fd{ -1 };
    std::vector<free_खण्डम्> free_खण्डानि;
};

संरचना pool_आवण्टनम् {
    शून्य * chunk_आधारः{ शून्यसूचकः };
    size_t chunk_आकार{ 0 };
    शून्य * आधारः{ शून्यसूचकः };
    size_t size{ 0 };
};

तार्किक is_power_सम्बन्धि_two(size_t value) {
    प्रत्यागम value != 0 && (value & (value - 1)) == 0;
}

तार्किक align_ऊर्ध्वम्(size_t value, size_t alignment, size_t * aligned_मानम्) {
    यदि (aligned_मानम् == शून्यसूचकः || alignment == 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव size_t remainder = value % alignment;
    यदि (remainder == 0) {
        *aligned_मानम् = value;
        प्रत्यागम सत्य;
    }

    ध्रुव size_t padding = alignment - remainder;
    यदि (value > std::numeric_limits<size_t>::max() - padding) {
        प्रत्यागम असत्य;
    }

    *aligned_मानम् = value + padding;
    प्रत्यागम सत्य;
}

तार्किक align_ऊर्ध्वम्_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_मानम्) {
    यदि (aligned_मानम् == शून्यसूचकः || alignment == 0) {
        प्रत्यागम असत्य;
    }

    ध्रुव uintptr_t remainder = value % alignment;
    यदि (remainder == 0) {
        *aligned_मानम् = value;
        प्रत्यागम सत्य;
    }

    ध्रुव uintptr_t padding = alignment - remainder;
    यदि (value > std::numeric_limits<uintptr_t>::max() - padding) {
        प्रत्यागम असत्य;
    }

    *aligned_मानम् = value + padding;
    प्रत्यागम सत्य;
}

वर्ग spine_mem_pool_manager {
  सार्वजनिक:
    स्पष्ट spine_mem_pool_manager(size_t default_chunk_आकार) : default_chunk_आकार_2(default_chunk_आकार) {}

    आभास ~spine_mem_pool_manager() = स्वभाव;

    शून्य * alloc(size_t size, size_t alignment) {
        यदि (size == 0 || !is_power_सम्बन्धि_two(alignment)) {
            प्रत्यागम शून्यसूचकः;
        }

        size_t aligned_आकार = 0;
        यदि (!align_ऊर्ध्वम्(size, alignment, &aligned_आकार)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            प्रत्यागम शून्यसूचकः;
        }

        pool_आवण्टनम् आवण्टनम्;

        std::lock_guard<std::mutex> तालकम्(mutex_);

        यदि (!try_alloc_locked(aligned_आकार, alignment, &आवण्टनम्)) {
            यदि (!सङ्कलय_chunk_locked(aligned_आकार, alignment)) {
                प्रत्यागम शून्यसूचकः;
            }

            यदि (!try_alloc_locked(aligned_आकार, alignment, &आवण्टनम्)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_आकार, alignment);
                प्रत्यागम शून्यसूचकः;
            }
        }

        यत्न {
            ध्रुव स्वचालित [आवण्टनम्_it, inserted] = allocations_.emplace(आवण्टनम्.आधारः, आवण्टनम्);
            यदि (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, आवण्टनम्.आधारः);
                rollback_आवण्टनम्_locked(आवण्टनम्);
                प्रत्यागम शून्यसूचकः;
            }
        } गृहाण (ध्रुव std::bad_alloc &) {
            rollback_आवण्टनम्_locked(आवण्टनम्);
            क्षिप;
        }

        प्रत्यागम आवण्टनम्.आधारः;
    }

    शून्य मुक्त(शून्य * आधारः) {
        यदि (आधारः == शून्यसूचकः) {
            प्रत्यागम;
        }

        std::lock_guard<std::mutex> तालकम्(mutex_);

        स्वचालित आवण्टनम्_it = allocations_.find(आधारः);
        यदि (आवण्टनम्_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, आधारः);
            प्रत्यागम;
        }

        pool_आवण्टनम् आवण्टनम् = आवण्टनम्_it->second;
        allocations_.erase(आवण्टनम्_it);

        स्वचालित chunk_it = find_chunk_locked(आवण्टनम्);
        यदि (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           आवण्टनम्.आधारः, आवण्टनम्.size);
            प्रत्यागम;
        }

        स्वचालित * chunk_आधारः = chunk_it->आधारः;
        स्वचालित * alloc_आधारः = static_cast<uint8_t *>(आवण्टनम्.आधारः);
        यदि (alloc_आधारः < chunk_आधारः || alloc_आधारः >= chunk_आधारः + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           आवण्टनम्.आधारः, chunk_आधारः, chunk_आधारः + chunk_it->size);
            प्रत्यागम;
        }

        ध्रुव size_t विस्थापनम् = static_cast<size_t>(alloc_आधारः - chunk_आधारः);
        यदि (विस्थापनम् > chunk_it->size || आवण्टनम्.size > chunk_it->size - विस्थापनम्) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           आवण्टनम्.आधारः, आवण्टनम्.size, chunk_it->size);
            प्रत्यागम;
        }

        निवेशय_free_खण्डम्_locked(*chunk_it, { विस्थापनम्, आवण्टनम्.size });
        maybe_release_रिक्तानि_chunk_locked(chunk_it);
    }

  रक्षित:
    शून्य release_chunks() {
        std::lock_guard<std::mutex> तालकम्(mutex_);

        allocations_.clear();
        प्रति (स्वचालित & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_आकार() ध्रुव { प्रत्यागम default_chunk_आकार_2; }

    स्थिर शून्य clear_chunk(pool_chunk * chunk) {
        chunk->आधारः = शून्यसूचकः;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_खण्डानि.clear();
    }

    आभास तार्किक alloc_chunk(size_t min_आकार, size_t alignment, शून्य * hint_addr, pool_chunk * chunk) = 0;
    आभास शून्य dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  गुप्त:
    संरचना alloc_candidate {
        size_t    chunk_सूचकांक{ 0 };
        size_t    खण्डम्_सूचकांक{ 0 };
        size_t    aligned_विस्थापनम्{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        तार्किक      valid{ असत्य };
    };

    std::vector<pool_chunk>::iterator find_chunk_locked(ध्रुव pool_आवण्टनम् & आवण्टनम्) {
        प्रत्यागम std::find_if(chunks_.begin(), chunks_.end(), [&](ध्रुव pool_chunk & chunk) {
            प्रत्यागम chunk.आधारः == आवण्टनम्.chunk_आधारः && chunk.size == आवण्टनम्.chunk_आकार;
        });
    }

    तार्किक सङ्कलय_chunk_locked(size_t min_आकार, size_t alignment) {
        pool_chunk   chunk;
        ध्रुव size_t chunk_request = default_chunk_आकार_2 == 0 ? min_आकार : std::max(min_आकार, default_chunk_आकार_2);
        शून्य *       hint_addr     = शून्यसूचकः;

        प्रति (ध्रुव स्वचालित & existing_chunk : chunks_) {
            स्वचालित * chunk_अन्तः = existing_chunk.आधारः + existing_chunk.size;
            यदि (hint_addr == शून्यसूचकः || chunk_अन्तः > hint_addr) {
                hint_addr = chunk_अन्तः;
            }
        }

        यदि (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            प्रत्यागम असत्य;
        }

        यदि (chunk.आधारः == शून्यसूचकः || chunk.size < min_आकार) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_आकार, chunk.आधारः, chunk.size);
            dealloc_chunk(&chunk);
            प्रत्यागम असत्य;
        }

        यत्न {
            chunk.free_खण्डानि.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } गृहाण (ध्रुव std::bad_alloc &) {
            dealloc_chunk(&chunk);
            क्षिप;
        }

        प्रत्यागम सत्य;
    }

    शून्य rollback_आवण्टनम्_locked(ध्रुव pool_आवण्टनम् & आवण्टनम्) {
        स्वचालित chunk_it = find_chunk_locked(आवण्टनम्);
        यदि (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, आवण्टनम्.आधारः);
            प्रत्यागम;
        }

        स्वचालित * chunk_आधारः = chunk_it->आधारः;
        स्वचालित * alloc_आधारः = static_cast<uint8_t *>(आवण्टनम्.आधारः);
        यदि (alloc_आधारः < chunk_आधारः || alloc_आधारः >= chunk_आधारः + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, आवण्टनम्.आधारः);
            प्रत्यागम;
        }

        ध्रुव size_t विस्थापनम् = static_cast<size_t>(alloc_आधारः - chunk_आधारः);
        यदि (विस्थापनम् > chunk_it->size || आवण्टनम्.size > chunk_it->size - विस्थापनम्) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           आवण्टनम्.आधारः, आवण्टनम्.size);
            प्रत्यागम;
        }

        निवेशय_free_खण्डम्_locked(*chunk_it, { विस्थापनम्, आवण्टनम्.size });
        maybe_release_रिक्तानि_chunk_locked(chunk_it);
    }

    तार्किक try_alloc_locked(size_t size, size_t alignment, pool_आवण्टनम् * आवण्टनम्) {
        alloc_candidate best;

        प्रति (size_t chunk_सूचकांक = 0; chunk_सूचकांक < chunks_.size(); ++chunk_सूचकांक) {
            ध्रुव स्वचालित & chunk = chunks_[chunk_सूचकांक];
            प्रति (size_t खण्डम्_सूचकांक = 0; खण्डम्_सूचकांक < chunk.free_खण्डानि.size(); ++खण्डम्_सूचकांक) {
                ध्रुव स्वचालित & खण्डम् = chunk.free_खण्डानि[खण्डम्_सूचकांक];

                uintptr_t  aligned_addr = 0;
                ध्रुव स्वचालित खण्डम्_addr   = reinterpret_cast<uintptr_t>(chunk.आधारः + खण्डम्.विस्थापनम्);
                यदि (!align_ऊर्ध्वम्_uintptr(खण्डम्_addr, alignment, &aligned_addr)) {
                    अनुवर्त;
                }

                यदि (aligned_addr < खण्डम्_addr) {
                    अनुवर्त;
                }

                ध्रुव size_t aligned_विस्थापनम् = खण्डम्.विस्थापनम् + static_cast<size_t>(aligned_addr - खण्डम्_addr);
                ध्रुव size_t padding        = aligned_विस्थापनम् - खण्डम्.विस्थापनम्;
                यदि (padding > खण्डम्.size || size > खण्डम्.size - padding) {
                    अनुवर्त;
                }

                यदि (!best.valid || aligned_addr < best.address) {
                    best.chunk_सूचकांक    = chunk_सूचकांक;
                    best.खण्डम्_सूचकांक    = खण्डम्_सूचकांक;
                    best.aligned_विस्थापनम् = aligned_विस्थापनम्;
                    best.address        = aligned_addr;
                    best.valid          = सत्य;
                }
            }
        }

        यदि (!best.valid) {
            प्रत्यागम असत्य;
        }

        स्वचालित &           chunk     = chunks_[best.chunk_सूचकांक];
        ध्रुव free_खण्डम् खण्डम्     = chunk.free_खण्डानि[best.खण्डम्_सूचकांक];
        ध्रुव size_t     padding   = best.aligned_विस्थापनम् - खण्डम्.विस्थापनम्;
        ध्रुव size_t     alloc_अन्तः = best.aligned_विस्थापनम् + size;
        ध्रुव size_t     खण्डम्_अन्तः = खण्डम्.विस्थापनम् + खण्डम्.size;

        chunk.free_खण्डानि.erase(chunk.free_खण्डानि.begin() + best.खण्डम्_सूचकांक);
        स्वचालित निवेशय_it = chunk.free_खण्डानि.begin() + best.खण्डम्_सूचकांक;
        यदि (padding != 0) {
            निवेशय_it = chunk.free_खण्डानि.insert(निवेशय_it, { खण्डम्.विस्थापनम्, padding });
            ++निवेशय_it;
        }
        यदि (alloc_अन्तः < खण्डम्_अन्तः) {
            chunk.free_खण्डानि.insert(निवेशय_it, { alloc_अन्तः, खण्डम्_अन्तः - alloc_अन्तः });
        }

        आवण्टनम्->chunk_आधारः = chunk.आधारः;
        आवण्टनम्->chunk_आकार = chunk.size;
        आवण्टनम्->आधारः       = chunk.आधारः + best.aligned_विस्थापनम्;
        आवण्टनम्->size       = size;
        प्रत्यागम सत्य;
    }

    शून्य maybe_release_रिक्तानि_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        यदि (chunk_it->free_खण्डानि.size() != 1) {
            प्रत्यागम;
        }

        ध्रुव स्वचालित & खण्डम् = chunk_it->free_खण्डानि.front();
        यदि (खण्डम्.विस्थापनम् != 0 || खण्डम्.size != chunk_it->size) {
            प्रत्यागम;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    शून्य निवेशय_free_खण्डम्_locked(pool_chunk & chunk, free_खण्डम् खण्डम्) {
        स्वचालित it = chunk.free_खण्डानि.begin();
        यावत् (it != chunk.free_खण्डानि.end() && it->विस्थापनम् < खण्डम्.विस्थापनम्) {
            ++it;
        }

        यदि (it != chunk.free_खण्डानि.begin()) {
            ध्रुव स्वचालित & पूर्व = *(it - 1);
            यदि (पूर्व.विस्थापनम् + पूर्व.size > खण्डम्.विस्थापनम्) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               खण्डम्.विस्थापनम्, खण्डम्.size);
                प्रत्यागम;
            }
        }

        यदि (it != chunk.free_खण्डानि.end() && खण्डम्.विस्थापनम् + खण्डम्.size > it->विस्थापनम्) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           खण्डम्.विस्थापनम्, खण्डम्.size);
            प्रत्यागम;
        }

        it = chunk.free_खण्डानि.insert(it, खण्डम्);

        यदि (it != chunk.free_खण्डानि.begin()) {
            स्वचालित पूर्व = it - 1;
            यदि (पूर्व->विस्थापनम् + पूर्व->size == it->विस्थापनम्) {
                it->विस्थापनम् = पूर्व->विस्थापनम्;
                it->size += पूर्व->size;
                it = chunk.free_खण्डानि.erase(पूर्व);
            }
        }

        यदि (it + 1 != chunk.free_खण्डानि.end() && it->विस्थापनम् + it->size == (it + 1)->विस्थापनम्) {
            it->size += (it + 1)->size;
            chunk.free_खण्डानि.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<शून्य *, pool_आवण्टनम्> allocations_;
    size_t                                      default_chunk_आकार_2{ 0 };
};

वर्ग spine_mem_pool_posix final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(size_t min_आकार, size_t alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) hint_addr;

        ध्रुव size_t alloc_alignment = std::max(alignment, परिमाण(शून्य *));
        शून्य *       आधारः            = शून्यसूचकः;
        ध्रुव पूर्णाङ्क    rc              = posix_memalign(&आधारः, alloc_alignment, min_आकार);
        यदि (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_आकार, alloc_alignment, rc);
            प्रत्यागम असत्य;
        }

        chunk->आधारः = static_cast<uint8_t *>(आधारः);
        chunk->size = min_आकार;
        chunk->fd   = -1;
        प्रत्यागम सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        std::मुक्त(chunk->आधारः);
        clear_chunk(chunk);
    }
};

वर्ग spine_mem_pool_transparent_hugepage final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_आकार) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(size_t min_आकार, size_t alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) alignment;

        size_t chunk_आकार = 0;
        यदि (!align_ऊर्ध्वम्(min_आकार, default_chunk_आकार(), &chunk_आकार)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_आकार);
            प्रत्यागम असत्य;
        }

        शून्य * map_addr = mmap(hint_addr, chunk_आकार, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        यदि (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_आकार,
                           errno);
            प्रत्यागम असत्य;
        }

        यदि (madvise(map_addr, chunk_आकार, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_आकार, errno);
            munmap(map_addr, chunk_आकार);
            प्रत्यागम असत्य;
        }

        chunk->आधारः = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_आकार;
        chunk->fd   = -1;
        प्रत्यागम सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        यदि (chunk->आधारः != शून्यसूचकः && chunk->size != 0 && munmap(chunk->आधारः, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->आधारः, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

वर्ग spine_mem_pool_hugetlb_1g final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1_G_REGION_आकार) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(size_t min_आकार, size_t alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) alignment;
        (शून्य) hint_addr;

        size_t region_आकार = 0;
        यदि (!align_ऊर्ध्वम्(min_आकार, SPINE_MEM_POOL_1_G_REGION_आकार, &region_आकार)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_आकार);
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क fd = उद्घाटय(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        यदि (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            प्रत्यागम असत्य;
        }

        hugetlb_1g_region region;
        region.size  = region_आकार;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        यदि (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_आकार, errno);
            close(fd);
            प्रत्यागम असत्य;
        }

        शून्य * map_addr = mmap(शून्यसूचकः, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        यदि (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<अचिह्नित दीर्घ दीर्घ>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            प्रत्यागम असत्य;
        }

        chunk->आधारः = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        प्रत्यागम सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        यदि (chunk->आधारः != शून्यसूचकः && chunk->size != 0 && munmap(chunk->आधारः, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->आधारः, chunk->size, errno);
        }

        यदि (chunk->fd >= 0) {
            यदि (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->आधारः, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

वर्ग spine_mem_pool_shared_mem final : सार्वजनिक spine_mem_pool_manager {
  सार्वजनिक:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_आकार) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  गुप्त:
    तार्किक alloc_chunk(size_t min_आकार, size_t alignment, शून्य * hint_addr, pool_chunk * chunk) override {
        (शून्य) alignment;

        यदि (hint_addr != शून्यसूचकः) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            प्रत्यागम असत्य;
        }

        यदि (min_आकार > default_chunk_आकार()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_आकार, default_chunk_आकार());
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क fd = उद्घाटय(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        यदि (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            प्रत्यागम असत्य;
        }

        शून्य * map_addr = mmap(शून्यसूचकः, default_chunk_आकार(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        यदि (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_आकार(), errno);
            close(fd);
            प्रत्यागम असत्य;
        }

        chunk->आधारः = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_आकार();
        chunk->fd   = fd;
        प्रत्यागम सत्य;
    }

    शून्य dealloc_chunk(pool_chunk * chunk) override {
        यदि (chunk->आधारः != शून्यसूचकः && chunk->size != 0 && munmap(chunk->आधारः, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->आधारः, chunk->size, errno);
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
    यदि (info == शून्यसूचकः) {
        प्रत्यागम असत्य;
    }

    *info = {};

    यदि (spine_tcm_open_handle(रिक्त) != 0 || !spine_tcm_is_उपलब्धम्()) {
        प्रत्यागम असत्य;
    }

    spine_tcm_mem_info_t mem_info;
    यदि (spine_tcm_mem_info(&mem_info) != 0) {
        प्रत्यागम असत्य;
    }

    info->उपलब्धम्   = सत्य;
    info->blk_आकार    = mem_info.blk_आकार;
    info->blk_अङ्कः     = mem_info.blk_अङ्कः;
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
        प्रत्यागम शून्यसूचकः;
    }
}

शून्य * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    यत्न {
        प्रत्यागम get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } गृहाण (ध्रुव std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        प्रत्यागम शून्यसूचकः;
    }
}

शून्य spine_mem_pool_free(शून्य * आधारः) noexcept {
    यत्न {
        get_spine_mem_pool_manager().मुक्त(आधारः);
    } गृहाण (ध्रुव std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, आधारः);
    }
}

शून्य spine_mem_pool_shared_mem_free(शून्य * आधारः) noexcept {
    यत्न {
        get_spine_mem_pool_shared_mem_manager().मुक्त(आधारः);
    } गृहाण (ध्रुव std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, आधारः);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

बाह्य "C" {
शून्य * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    शून्य * परिणाम = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    यदि (परिणाम == शून्यसूचकः) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    प्रत्यागम परिणाम;
}

शून्य ggml_backend_cpu_riscv64_spacemit_free_shared(शून्य * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
