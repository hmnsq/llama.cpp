#అన్తర్భావ "spine_mem_pool.h"

#అన్తర్భావ "common.h"
#అన్తర్భావ "ime_env.h"
#అన్తర్భావ "spine_tcm.h"

#అన్తర్భావ <fcntl.h>
#అన్తర్భావ <sys/ioctl.h>
#అన్తర్భావ <sys/mman.h>
#అన్తర్భావ <unistd.h>

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cerrno>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <limits>
#అన్తర్భావ <memory>
#అన్తర్భావ <mutex>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <vector>

నామాకాశ ggml::cpu::riscv64_spacemit {
నామాకాశ {

constexpr size_t   SPINE_MEM_POOL_CHUNK_కొలత         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_కొలత   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1_G_REGION_కొలత     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr అక్షరం     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr అక్షరం     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

నిర్మాణం hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#నిర్వచించు HUGETLB_1G_IOC_MAGIC 'M'
#నిర్వచించు HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, నిర్మాణం hugetlb_1g_region)
#నిర్వచించు HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

నిర్మాణం free_ఖండం {
    size_t స్థానభ్రంశం{ 0 };
    size_t size{ 0 };
};

నిర్మాణం pool_chunk {
    uint8_t *               పునాది{ శూన్యసూచిక };
    size_t                  size{ 0 };
    పూర్ణాంకం                     fd{ -1 };
    std::vector<free_ఖండం> free_ఖండాలు;
};

నిర్మాణం pool_కేటాయింపు {
    శూన్య * chunk_పునాది{ శూన్యసూచిక };
    size_t chunk_కొలత{ 0 };
    శూన్య * పునాది{ శూన్యసూచిక };
    size_t size{ 0 };
};

తార్కిక is_power_యొక్క_two(size_t value) {
    తిరిగి value != 0 && (value & (value - 1)) == 0;
}

తార్కిక align_పైకి(size_t value, size_t alignment, size_t * aligned_విలువ) {
    ఒకవేళ (aligned_విలువ == శూన్యసూచిక || alignment == 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి size_t remainder = value % alignment;
    ఒకవేళ (remainder == 0) {
        *aligned_విలువ = value;
        తిరిగి నిజం;
    }

    స్థిరరాశి size_t padding = alignment - remainder;
    ఒకవేళ (value > std::numeric_limits<size_t>::max() - padding) {
        తిరిగి అబద్ధం;
    }

    *aligned_విలువ = value + padding;
    తిరిగి నిజం;
}

తార్కిక align_పైకి_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_విలువ) {
    ఒకవేళ (aligned_విలువ == శూన్యసూచిక || alignment == 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uintptr_t remainder = value % alignment;
    ఒకవేళ (remainder == 0) {
        *aligned_విలువ = value;
        తిరిగి నిజం;
    }

    స్థిరరాశి uintptr_t padding = alignment - remainder;
    ఒకవేళ (value > std::numeric_limits<uintptr_t>::max() - padding) {
        తిరిగి అబద్ధం;
    }

    *aligned_విలువ = value + padding;
    తిరిగి నిజం;
}

తరగతి spine_mem_pool_manager {
  బహిరంగ:
    స్పష్టమైన spine_mem_pool_manager(size_t default_chunk_కొలత) : default_chunk_కొలత_2(default_chunk_కొలత) {}

    మాయా ~spine_mem_pool_manager() = అప్రమేయ;

    శూన్య * alloc(size_t size, size_t alignment) {
        ఒకవేళ (size == 0 || !is_power_యొక్క_two(alignment)) {
            తిరిగి శూన్యసూచిక;
        }

        size_t aligned_కొలత = 0;
        ఒకవేళ (!align_పైకి(size, alignment, &aligned_కొలత)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            తిరిగి శూన్యసూచిక;
        }

        pool_కేటాయింపు కేటాయింపు;

        std::lock_guard<std::mutex> తాళం(mutex_);

        ఒకవేళ (!try_alloc_locked(aligned_కొలత, alignment, &కేటాయింపు)) {
            ఒకవేళ (!కూడు_chunk_locked(aligned_కొలత, alignment)) {
                తిరిగి శూన్యసూచిక;
            }

            ఒకవేళ (!try_alloc_locked(aligned_కొలత, alignment, &కేటాయింపు)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_కొలత, alignment);
                తిరిగి శూన్యసూచిక;
            }
        }

        ప్రయత్నించు {
            స్థిరరాశి స్వయంచాలక [కేటాయింపు_it, inserted] = allocations_.emplace(కేటాయింపు.పునాది, కేటాయింపు);
            ఒకవేళ (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, కేటాయింపు.పునాది);
                rollback_కేటాయింపు_locked(కేటాయింపు);
                తిరిగి శూన్యసూచిక;
            }
        } పట్టుకో (స్థిరరాశి std::bad_alloc &) {
            rollback_కేటాయింపు_locked(కేటాయింపు);
            విసురు;
        }

        తిరిగి కేటాయింపు.పునాది;
    }

    శూన్య విడుదల(శూన్య * పునాది) {
        ఒకవేళ (పునాది == శూన్యసూచిక) {
            తిరిగి;
        }

        std::lock_guard<std::mutex> తాళం(mutex_);

        స్వయంచాలక కేటాయింపు_it = allocations_.find(పునాది);
        ఒకవేళ (కేటాయింపు_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, పునాది);
            తిరిగి;
        }

        pool_కేటాయింపు కేటాయింపు = కేటాయింపు_it->second;
        allocations_.erase(కేటాయింపు_it);

        స్వయంచాలక chunk_it = కనుగొను_chunk_locked(కేటాయింపు);
        ఒకవేళ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           కేటాయింపు.పునాది, కేటాయింపు.size);
            తిరిగి;
        }

        స్వయంచాలక * chunk_పునాది = chunk_it->పునాది;
        స్వయంచాలక * alloc_పునాది = static_cast<uint8_t *>(కేటాయింపు.పునాది);
        ఒకవేళ (alloc_పునాది < chunk_పునాది || alloc_పునాది >= chunk_పునాది + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           కేటాయింపు.పునాది, chunk_పునాది, chunk_పునాది + chunk_it->size);
            తిరిగి;
        }

        స్థిరరాశి size_t స్థానభ్రంశం = static_cast<size_t>(alloc_పునాది - chunk_పునాది);
        ఒకవేళ (స్థానభ్రంశం > chunk_it->size || కేటాయింపు.size > chunk_it->size - స్థానభ్రంశం) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           కేటాయింపు.పునాది, కేటాయింపు.size, chunk_it->size);
            తిరిగి;
        }

        చొప్పించు_free_ఖండం_locked(*chunk_it, { స్థానభ్రంశం, కేటాయింపు.size });
        maybe_release_ఖాళీలు_chunk_locked(chunk_it);
    }

  రక్షిత:
    శూన్య release_chunks() {
        std::lock_guard<std::mutex> తాళం(mutex_);

        allocations_.clear();
        కొరకు (స్వయంచాలక & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_కొలత() స్థిరరాశి { తిరిగి default_chunk_కొలత_2; }

    స్థిర శూన్య clear_chunk(pool_chunk * chunk) {
        chunk->పునాది = శూన్యసూచిక;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_ఖండాలు.clear();
    }

    మాయా తార్కిక alloc_chunk(size_t min_కొలత, size_t alignment, శూన్య * hint_addr, pool_chunk * chunk) = 0;
    మాయా శూన్య dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  అంతరంగ:
    నిర్మాణం alloc_candidate {
        size_t    chunk_సూచీ{ 0 };
        size_t    ఖండం_సూచీ{ 0 };
        size_t    aligned_స్థానభ్రంశం{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        తార్కిక      valid{ అబద్ధం };
    };

    std::vector<pool_chunk>::iterator కనుగొను_chunk_locked(స్థిరరాశి pool_కేటాయింపు & కేటాయింపు) {
        తిరిగి std::find_if(chunks_.begin(), chunks_.end(), [&](స్థిరరాశి pool_chunk & chunk) {
            తిరిగి chunk.పునాది == కేటాయింపు.chunk_పునాది && chunk.size == కేటాయింపు.chunk_కొలత;
        });
    }

    తార్కిక కూడు_chunk_locked(size_t min_కొలత, size_t alignment) {
        pool_chunk   chunk;
        స్థిరరాశి size_t chunk_request = default_chunk_కొలత_2 == 0 ? min_కొలత : std::max(min_కొలత, default_chunk_కొలత_2);
        శూన్య *       hint_addr     = శూన్యసూచిక;

        కొరకు (స్థిరరాశి స్వయంచాలక & existing_chunk : chunks_) {
            స్వయంచాలక * chunk_చివర = existing_chunk.పునాది + existing_chunk.size;
            ఒకవేళ (hint_addr == శూన్యసూచిక || chunk_చివర > hint_addr) {
                hint_addr = chunk_చివర;
            }
        }

        ఒకవేళ (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (chunk.పునాది == శూన్యసూచిక || chunk.size < min_కొలత) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_కొలత, chunk.పునాది, chunk.size);
            dealloc_chunk(&chunk);
            తిరిగి అబద్ధం;
        }

        ప్రయత్నించు {
            chunk.free_ఖండాలు.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } పట్టుకో (స్థిరరాశి std::bad_alloc &) {
            dealloc_chunk(&chunk);
            విసురు;
        }

        తిరిగి నిజం;
    }

    శూన్య rollback_కేటాయింపు_locked(స్థిరరాశి pool_కేటాయింపు & కేటాయింపు) {
        స్వయంచాలక chunk_it = కనుగొను_chunk_locked(కేటాయింపు);
        ఒకవేళ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, కేటాయింపు.పునాది);
            తిరిగి;
        }

        స్వయంచాలక * chunk_పునాది = chunk_it->పునాది;
        స్వయంచాలక * alloc_పునాది = static_cast<uint8_t *>(కేటాయింపు.పునాది);
        ఒకవేళ (alloc_పునాది < chunk_పునాది || alloc_పునాది >= chunk_పునాది + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, కేటాయింపు.పునాది);
            తిరిగి;
        }

        స్థిరరాశి size_t స్థానభ్రంశం = static_cast<size_t>(alloc_పునాది - chunk_పునాది);
        ఒకవేళ (స్థానభ్రంశం > chunk_it->size || కేటాయింపు.size > chunk_it->size - స్థానభ్రంశం) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           కేటాయింపు.పునాది, కేటాయింపు.size);
            తిరిగి;
        }

        చొప్పించు_free_ఖండం_locked(*chunk_it, { స్థానభ్రంశం, కేటాయింపు.size });
        maybe_release_ఖాళీలు_chunk_locked(chunk_it);
    }

    తార్కిక try_alloc_locked(size_t size, size_t alignment, pool_కేటాయింపు * కేటాయింపు) {
        alloc_candidate ఉత్తమ;

        కొరకు (size_t chunk_సూచీ = 0; chunk_సూచీ < chunks_.size(); ++chunk_సూచీ) {
            స్థిరరాశి స్వయంచాలక & chunk = chunks_[chunk_సూచీ];
            కొరకు (size_t ఖండం_సూచీ = 0; ఖండం_సూచీ < chunk.free_ఖండాలు.size(); ++ఖండం_సూచీ) {
                స్థిరరాశి స్వయంచాలక & ఖండం = chunk.free_ఖండాలు[ఖండం_సూచీ];

                uintptr_t  aligned_addr = 0;
                స్థిరరాశి స్వయంచాలక ఖండం_addr   = reinterpret_cast<uintptr_t>(chunk.పునాది + ఖండం.స్థానభ్రంశం);
                ఒకవేళ (!align_పైకి_uintptr(ఖండం_addr, alignment, &aligned_addr)) {
                    కొనసాగు;
                }

                ఒకవేళ (aligned_addr < ఖండం_addr) {
                    కొనసాగు;
                }

                స్థిరరాశి size_t aligned_స్థానభ్రంశం = ఖండం.స్థానభ్రంశం + static_cast<size_t>(aligned_addr - ఖండం_addr);
                స్థిరరాశి size_t padding        = aligned_స్థానభ్రంశం - ఖండం.స్థానభ్రంశం;
                ఒకవేళ (padding > ఖండం.size || size > ఖండం.size - padding) {
                    కొనసాగు;
                }

                ఒకవేళ (!ఉత్తమ.valid || aligned_addr < ఉత్తమ.address) {
                    ఉత్తమ.chunk_సూచీ    = chunk_సూచీ;
                    ఉత్తమ.ఖండం_సూచీ    = ఖండం_సూచీ;
                    ఉత్తమ.aligned_స్థానభ్రంశం = aligned_స్థానభ్రంశం;
                    ఉత్తమ.address        = aligned_addr;
                    ఉత్తమ.valid          = నిజం;
                }
            }
        }

        ఒకవేళ (!ఉత్తమ.valid) {
            తిరిగి అబద్ధం;
        }

        స్వయంచాలక &           chunk     = chunks_[ఉత్తమ.chunk_సూచీ];
        స్థిరరాశి free_ఖండం ఖండం     = chunk.free_ఖండాలు[ఉత్తమ.ఖండం_సూచీ];
        స్థిరరాశి size_t     padding   = ఉత్తమ.aligned_స్థానభ్రంశం - ఖండం.స్థానభ్రంశం;
        స్థిరరాశి size_t     alloc_చివర = ఉత్తమ.aligned_స్థానభ్రంశం + size;
        స్థిరరాశి size_t     ఖండం_చివర = ఖండం.స్థానభ్రంశం + ఖండం.size;

        chunk.free_ఖండాలు.erase(chunk.free_ఖండాలు.begin() + ఉత్తమ.ఖండం_సూచీ);
        స్వయంచాలక చొప్పించు_it = chunk.free_ఖండాలు.begin() + ఉత్తమ.ఖండం_సూచీ;
        ఒకవేళ (padding != 0) {
            చొప్పించు_it = chunk.free_ఖండాలు.insert(చొప్పించు_it, { ఖండం.స్థానభ్రంశం, padding });
            ++చొప్పించు_it;
        }
        ఒకవేళ (alloc_చివర < ఖండం_చివర) {
            chunk.free_ఖండాలు.insert(చొప్పించు_it, { alloc_చివర, ఖండం_చివర - alloc_చివర });
        }

        కేటాయింపు->chunk_పునాది = chunk.పునాది;
        కేటాయింపు->chunk_కొలత = chunk.size;
        కేటాయింపు->పునాది       = chunk.పునాది + ఉత్తమ.aligned_స్థానభ్రంశం;
        కేటాయింపు->size       = size;
        తిరిగి నిజం;
    }

    శూన్య maybe_release_ఖాళీలు_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        ఒకవేళ (chunk_it->free_ఖండాలు.size() != 1) {
            తిరిగి;
        }

        స్థిరరాశి స్వయంచాలక & ఖండం = chunk_it->free_ఖండాలు.front();
        ఒకవేళ (ఖండం.స్థానభ్రంశం != 0 || ఖండం.size != chunk_it->size) {
            తిరిగి;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    శూన్య చొప్పించు_free_ఖండం_locked(pool_chunk & chunk, free_ఖండం ఖండం) {
        స్వయంచాలక it = chunk.free_ఖండాలు.begin();
        అయితే (it != chunk.free_ఖండాలు.end() && it->స్థానభ్రంశం < ఖండం.స్థానభ్రంశం) {
            ++it;
        }

        ఒకవేళ (it != chunk.free_ఖండాలు.begin()) {
            స్థిరరాశి స్వయంచాలక & prev = *(it - 1);
            ఒకవేళ (prev.స్థానభ్రంశం + prev.size > ఖండం.స్థానభ్రంశం) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               ఖండం.స్థానభ్రంశం, ఖండం.size);
                తిరిగి;
            }
        }

        ఒకవేళ (it != chunk.free_ఖండాలు.end() && ఖండం.స్థానభ్రంశం + ఖండం.size > it->స్థానభ్రంశం) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           ఖండం.స్థానభ్రంశం, ఖండం.size);
            తిరిగి;
        }

        it = chunk.free_ఖండాలు.insert(it, ఖండం);

        ఒకవేళ (it != chunk.free_ఖండాలు.begin()) {
            స్వయంచాలక prev = it - 1;
            ఒకవేళ (prev->స్థానభ్రంశం + prev->size == it->స్థానభ్రంశం) {
                it->స్థానభ్రంశం = prev->స్థానభ్రంశం;
                it->size += prev->size;
                it = chunk.free_ఖండాలు.erase(prev);
            }
        }

        ఒకవేళ (it + 1 != chunk.free_ఖండాలు.end() && it->స్థానభ్రంశం + it->size == (it + 1)->స్థానభ్రంశం) {
            it->size += (it + 1)->size;
            chunk.free_ఖండాలు.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<శూన్య *, pool_కేటాయింపు> allocations_;
    size_t                                      default_chunk_కొలత_2{ 0 };
};

తరగతి spine_mem_pool_posix final : బహిరంగ spine_mem_pool_manager {
  బహిరంగ:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  అంతరంగ:
    తార్కిక alloc_chunk(size_t min_కొలత, size_t alignment, శూన్య * hint_addr, pool_chunk * chunk) override {
        (శూన్య) hint_addr;

        స్థిరరాశి size_t alloc_alignment = std::max(alignment, పరిమాణం(శూన్య *));
        శూన్య *       పునాది            = శూన్యసూచిక;
        స్థిరరాశి పూర్ణాంకం    rc              = posix_memalign(&పునాది, alloc_alignment, min_కొలత);
        ఒకవేళ (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_కొలత, alloc_alignment, rc);
            తిరిగి అబద్ధం;
        }

        chunk->పునాది = static_cast<uint8_t *>(పునాది);
        chunk->size = min_కొలత;
        chunk->fd   = -1;
        తిరిగి నిజం;
    }

    శూన్య dealloc_chunk(pool_chunk * chunk) override {
        std::విడుదల(chunk->పునాది);
        clear_chunk(chunk);
    }
};

తరగతి spine_mem_pool_transparent_hugepage final : బహిరంగ spine_mem_pool_manager {
  బహిరంగ:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_కొలత) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  అంతరంగ:
    తార్కిక alloc_chunk(size_t min_కొలత, size_t alignment, శూన్య * hint_addr, pool_chunk * chunk) override {
        (శూన్య) alignment;

        size_t chunk_కొలత = 0;
        ఒకవేళ (!align_పైకి(min_కొలత, default_chunk_కొలత(), &chunk_కొలత)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_కొలత);
            తిరిగి అబద్ధం;
        }

        శూన్య * map_addr = mmap(hint_addr, chunk_కొలత, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ఒకవేళ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_కొలత,
                           errno);
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (madvise(map_addr, chunk_కొలత, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_కొలత, errno);
            munmap(map_addr, chunk_కొలత);
            తిరిగి అబద్ధం;
        }

        chunk->పునాది = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_కొలత;
        chunk->fd   = -1;
        తిరిగి నిజం;
    }

    శూన్య dealloc_chunk(pool_chunk * chunk) override {
        ఒకవేళ (chunk->పునాది != శూన్యసూచిక && chunk->size != 0 && munmap(chunk->పునాది, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->పునాది, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

తరగతి spine_mem_pool_hugetlb_1g final : బహిరంగ spine_mem_pool_manager {
  బహిరంగ:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1_G_REGION_కొలత) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  అంతరంగ:
    తార్కిక alloc_chunk(size_t min_కొలత, size_t alignment, శూన్య * hint_addr, pool_chunk * chunk) override {
        (శూన్య) alignment;
        (శూన్య) hint_addr;

        size_t region_కొలత = 0;
        ఒకవేళ (!align_పైకి(min_కొలత, SPINE_MEM_POOL_1_G_REGION_కొలత, &region_కొలత)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_కొలత);
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి పూర్ణాంకం fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        ఒకవేళ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            తిరిగి అబద్ధం;
        }

        hugetlb_1g_region region;
        region.size  = region_కొలత;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        ఒకవేళ (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_కొలత, errno);
            close(fd);
            తిరిగి అబద్ధం;
        }

        శూన్య * map_addr = mmap(శూన్యసూచిక, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ఒకవేళ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<గుర్తులేని దీర్ఘ దీర్ఘ>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            తిరిగి అబద్ధం;
        }

        chunk->పునాది = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        తిరిగి నిజం;
    }

    శూన్య dealloc_chunk(pool_chunk * chunk) override {
        ఒకవేళ (chunk->పునాది != శూన్యసూచిక && chunk->size != 0 && munmap(chunk->పునాది, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->పునాది, chunk->size, errno);
        }

        ఒకవేళ (chunk->fd >= 0) {
            ఒకవేళ (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->పునాది, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

తరగతి spine_mem_pool_shared_mem final : బహిరంగ spine_mem_pool_manager {
  బహిరంగ:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_కొలత) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  అంతరంగ:
    తార్కిక alloc_chunk(size_t min_కొలత, size_t alignment, శూన్య * hint_addr, pool_chunk * chunk) override {
        (శూన్య) alignment;

        ఒకవేళ (hint_addr != శూన్యసూచిక) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (min_కొలత > default_chunk_కొలత()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_కొలత, default_chunk_కొలత());
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి పూర్ణాంకం fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        ఒకవేళ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            తిరిగి అబద్ధం;
        }

        శూన్య * map_addr = mmap(శూన్యసూచిక, default_chunk_కొలత(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ఒకవేళ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_కొలత(), errno);
            close(fd);
            తిరిగి అబద్ధం;
        }

        chunk->పునాది = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_కొలత();
        chunk->fd   = fd;
        తిరిగి నిజం;
    }

    శూన్య dealloc_chunk(pool_chunk * chunk) override {
        ఒకవేళ (chunk->పునాది != శూన్యసూచిక && chunk->size != 0 && munmap(chunk->పునాది, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->పునాది, chunk->size, errno);
        }

        ఒకవేళ (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    స్థిర std::once_flag                          pool_once;
    స్థిర std::unique_ptr<spine_mem_pool_manager> selected_pool;
    స్థిర spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    ఒకవేళ (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        ఎంపిక (selected_backend) {
            స్థితి spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                ఆపు;
            స్థితి spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ఆపు;
            స్థితి spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                ఆపు;
            స్థితి spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ఆపు;
        }
    });

    ఒకవేళ (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<పూర్ణాంకం>(backend), static_cast<పూర్ణాంకం>(selected_backend));
    }

    ఒకవేళ (selected_pool) {
        తిరిగి *selected_pool;
    }

    విసురు std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    స్థిర std::once_flag                             shared_mem_pool_once;
    స్థిర std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    ఒకవేళ (shared_mem_pool) {
        తిరిగి *shared_mem_pool;
    }

    విసురు std::bad_alloc();
}

}  // namespace

తార్కిక spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    ఒకవేళ (info == శూన్యసూచిక) {
        తిరిగి అబద్ధం;
    }

    *info = {};

    ఒకవేళ (spine_tcm_open_handle(ఖాళీ) != 0 || !spine_tcm_is_అందుబాటు()) {
        తిరిగి అబద్ధం;
    }

    spine_tcm_mem_info_t mem_info;
    ఒకవేళ (spine_tcm_mem_info(&mem_info) != 0) {
        తిరిగి అబద్ధం;
    }

    info->అందుబాటు   = నిజం;
    info->blk_కొలత    = mem_info.blk_కొలత;
    info->blk_అంకె     = mem_info.blk_అంకె;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    తిరిగి నిజం;
}

శూన్య * spine_mem_pool_tcm_mem_get(పూర్ణాంకం cpu_id) noexcept {
    తిరిగి spine_tcm_mem_get(cpu_id);
}

శూన్య * spine_mem_pool_tcm_mem_wait(పూర్ణాంకం cpu_id) noexcept {
    తిరిగి spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

పూర్ణాంకం spine_mem_pool_tcm_mem_release(పూర్ణాంకం cpu_id) noexcept {
    తిరిగి spine_tcm_mem_release(cpu_id);
}

శూన్య * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    ప్రయత్నించు {
        తిరిగి get_spine_mem_pool_manager().alloc(size, alignment);
    } పట్టుకో (స్థిరరాశి std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        తిరిగి శూన్యసూచిక;
    }
}

శూన్య * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    ప్రయత్నించు {
        తిరిగి get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } పట్టుకో (స్థిరరాశి std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        తిరిగి శూన్యసూచిక;
    }
}

శూన్య spine_mem_pool_free(శూన్య * పునాది) noexcept {
    ప్రయత్నించు {
        get_spine_mem_pool_manager().విడుదల(పునాది);
    } పట్టుకో (స్థిరరాశి std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, పునాది);
    }
}

శూన్య spine_mem_pool_shared_mem_free(శూన్య * పునాది) noexcept {
    ప్రయత్నించు {
        get_spine_mem_pool_shared_mem_manager().విడుదల(పునాది);
    } పట్టుకో (స్థిరరాశి std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, పునాది);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

బాహ్య "C" {
శూన్య * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    శూన్య * ఫలితం = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    ఒకవేళ (ఫలితం == శూన్యసూచిక) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    తిరిగి ఫలితం;
}

శూన్య ggml_backend_cpu_riscv64_spacemit_free_shared(శూన్య * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
