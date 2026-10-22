#ਅਨ੍ਤਰ੍ਭਾਵ "spine_mem_pool.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ime_env.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "spine_tcm.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <fcntl.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/ioctl.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/mman.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cerrno>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ggml::cpu::riscv64_spacemit {
ਨਾਮਾਕਾਸ਼ {

constexpr size_t   SPINE_MEM_POOL_CHUNK_ਮਾਪ         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_ਮਾਪ   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1_G_REGION_ਮਾਪ     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr ਅੱਖਰ     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr ਅੱਖਰ     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

ਬਣਤਰ hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HUGETLB_1G_IOC_MAGIC 'M'
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, ਬਣਤਰ hugetlb_1g_region)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

ਬਣਤਰ free_ਖੰਡ {
    size_t ਵਿੱਥ{ 0 };
    size_t size{ 0 };
};

ਬਣਤਰ pool_chunk {
    uint8_t *               ਅਧਾਰ{ ਖਾਲੀ_ਸੂਚਕ };
    size_t                  size{ 0 };
    ਪੂਰਨ                     fd{ -1 };
    std::vector<free_ਖੰਡ> free_ਖੰਡਾਂ;
};

ਬਣਤਰ pool_ਵੰਡ {
    ਖਾਲੀ * chunk_ਅਧਾਰ{ ਖਾਲੀ_ਸੂਚਕ };
    size_t chunk_ਮਾਪ{ 0 };
    ਖਾਲੀ * ਅਧਾਰ{ ਖਾਲੀ_ਸੂਚਕ };
    size_t size{ 0 };
};

ਤਾਰਕਿਕ is_power_ਦਾ_two(size_t value) {
    ਵਾਪਸ value != 0 && (value & (value - 1)) == 0;
}

ਤਾਰਕਿਕ align_ਉੱਪਰ(size_t value, size_t alignment, size_t * aligned_ਮੁੱਲ) {
    ਜੇ (aligned_ਮੁੱਲ == ਖਾਲੀ_ਸੂਚਕ || alignment == 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ size_t remainder = value % alignment;
    ਜੇ (remainder == 0) {
        *aligned_ਮੁੱਲ = value;
        ਵਾਪਸ ਸੱਚ;
    }

    ਅਚਲ size_t padding = alignment - remainder;
    ਜੇ (value > std::numeric_limits<size_t>::max() - padding) {
        ਵਾਪਸ ਝੂਠ;
    }

    *aligned_ਮੁੱਲ = value + padding;
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ align_ਉੱਪਰ_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_ਮੁੱਲ) {
    ਜੇ (aligned_ਮੁੱਲ == ਖਾਲੀ_ਸੂਚਕ || alignment == 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uintptr_t remainder = value % alignment;
    ਜੇ (remainder == 0) {
        *aligned_ਮੁੱਲ = value;
        ਵਾਪਸ ਸੱਚ;
    }

    ਅਚਲ uintptr_t padding = alignment - remainder;
    ਜੇ (value > std::numeric_limits<uintptr_t>::max() - padding) {
        ਵਾਪਸ ਝੂਠ;
    }

    *aligned_ਮੁੱਲ = value + padding;
    ਵਾਪਸ ਸੱਚ;
}

ਵਰਗ spine_mem_pool_manager {
  ਜਨਤਕ:
    ਸਪਸ਼ਟ spine_mem_pool_manager(size_t default_chunk_ਮਾਪ) : default_chunk_ਮਾਪ_2(default_chunk_ਮਾਪ) {}

    ਆਭਾਸੀ ~spine_mem_pool_manager() = ਮੂਲ;

    ਖਾਲੀ * alloc(size_t size, size_t alignment) {
        ਜੇ (size == 0 || !is_power_ਦਾ_two(alignment)) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        size_t aligned_ਮਾਪ = 0;
        ਜੇ (!align_ਉੱਪਰ(size, alignment, &aligned_ਮਾਪ)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        pool_ਵੰਡ ਵੰਡ;

        std::lock_guard<std::mutex> ਤਾਲਾ(mutex_);

        ਜੇ (!try_alloc_locked(aligned_ਮਾਪ, alignment, &ਵੰਡ)) {
            ਜੇ (!ਜੋੜੋ_chunk_locked(aligned_ਮਾਪ, alignment)) {
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }

            ਜੇ (!try_alloc_locked(aligned_ਮਾਪ, alignment, &ਵੰਡ)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_ਮਾਪ, alignment);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
        }

        ਕੋਸ਼ਿਸ਼ {
            ਅਚਲ ਸਵੈਚਾਲਿਤ [ਵੰਡ_it, inserted] = allocations_.emplace(ਵੰਡ.ਅਧਾਰ, ਵੰਡ);
            ਜੇ (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, ਵੰਡ.ਅਧਾਰ);
                rollback_ਵੰਡ_locked(ਵੰਡ);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
        } ਫੜੋ (ਅਚਲ std::bad_alloc &) {
            rollback_ਵੰਡ_locked(ਵੰਡ);
            ਸੁੱਟੋ;
        }

        ਵਾਪਸ ਵੰਡ.ਅਧਾਰ;
    }

    ਖਾਲੀ ਮੁਕਤ(ਖਾਲੀ * ਅਧਾਰ) {
        ਜੇ (ਅਧਾਰ == ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ;
        }

        std::lock_guard<std::mutex> ਤਾਲਾ(mutex_);

        ਸਵੈਚਾਲਿਤ ਵੰਡ_it = allocations_.find(ਅਧਾਰ);
        ਜੇ (ਵੰਡ_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, ਅਧਾਰ);
            ਵਾਪਸ;
        }

        pool_ਵੰਡ ਵੰਡ = ਵੰਡ_it->second;
        allocations_.erase(ਵੰਡ_it);

        ਸਵੈਚਾਲਿਤ chunk_it = ਲੱਭੋ_chunk_locked(ਵੰਡ);
        ਜੇ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           ਵੰਡ.ਅਧਾਰ, ਵੰਡ.size);
            ਵਾਪਸ;
        }

        ਸਵੈਚਾਲਿਤ * chunk_ਅਧਾਰ = chunk_it->ਅਧਾਰ;
        ਸਵੈਚਾਲਿਤ * alloc_ਅਧਾਰ = static_cast<uint8_t *>(ਵੰਡ.ਅਧਾਰ);
        ਜੇ (alloc_ਅਧਾਰ < chunk_ਅਧਾਰ || alloc_ਅਧਾਰ >= chunk_ਅਧਾਰ + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           ਵੰਡ.ਅਧਾਰ, chunk_ਅਧਾਰ, chunk_ਅਧਾਰ + chunk_it->size);
            ਵਾਪਸ;
        }

        ਅਚਲ size_t ਵਿੱਥ = static_cast<size_t>(alloc_ਅਧਾਰ - chunk_ਅਧਾਰ);
        ਜੇ (ਵਿੱਥ > chunk_it->size || ਵੰਡ.size > chunk_it->size - ਵਿੱਥ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           ਵੰਡ.ਅਧਾਰ, ਵੰਡ.size, chunk_it->size);
            ਵਾਪਸ;
        }

        ਪਾਓ_free_ਖੰਡ_locked(*chunk_it, { ਵਿੱਥ, ਵੰਡ.size });
        maybe_release_ਖਾਲੀ_ਥਾਵਾਂ_chunk_locked(chunk_it);
    }

  ਸੁਰੱਖਿਅਤ:
    ਖਾਲੀ release_chunks() {
        std::lock_guard<std::mutex> ਤਾਲਾ(mutex_);

        allocations_.clear();
        ਲਈ (ਸਵੈਚਾਲਿਤ & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_ਮਾਪ() ਅਚਲ { ਵਾਪਸ default_chunk_ਮਾਪ_2; }

    ਸਥਿਰ ਖਾਲੀ clear_chunk(pool_chunk * chunk) {
        chunk->ਅਧਾਰ = ਖਾਲੀ_ਸੂਚਕ;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_ਖੰਡਾਂ.clear();
    }

    ਆਭਾਸੀ ਤਾਰਕਿਕ alloc_chunk(size_t min_ਮਾਪ, size_t alignment, ਖਾਲੀ * hint_addr, pool_chunk * chunk) = 0;
    ਆਭਾਸੀ ਖਾਲੀ dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  ਨਿਜੀ:
    ਬਣਤਰ alloc_candidate {
        size_t    chunk_ਸੂਚਕਾਂਕ{ 0 };
        size_t    ਖੰਡ_ਸੂਚਕਾਂਕ{ 0 };
        size_t    aligned_ਵਿੱਥ{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        ਤਾਰਕਿਕ      valid{ ਝੂਠ };
    };

    std::vector<pool_chunk>::iterator ਲੱਭੋ_chunk_locked(ਅਚਲ pool_ਵੰਡ & ਵੰਡ) {
        ਵਾਪਸ std::find_if(chunks_.begin(), chunks_.end(), [&](ਅਚਲ pool_chunk & chunk) {
            ਵਾਪਸ chunk.ਅਧਾਰ == ਵੰਡ.chunk_ਅਧਾਰ && chunk.size == ਵੰਡ.chunk_ਮਾਪ;
        });
    }

    ਤਾਰਕਿਕ ਜੋੜੋ_chunk_locked(size_t min_ਮਾਪ, size_t alignment) {
        pool_chunk   chunk;
        ਅਚਲ size_t chunk_request = default_chunk_ਮਾਪ_2 == 0 ? min_ਮਾਪ : std::max(min_ਮਾਪ, default_chunk_ਮਾਪ_2);
        ਖਾਲੀ *       hint_addr     = ਖਾਲੀ_ਸੂਚਕ;

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & existing_chunk : chunks_) {
            ਸਵੈਚਾਲਿਤ * chunk_ਸਿਰਾ = existing_chunk.ਅਧਾਰ + existing_chunk.size;
            ਜੇ (hint_addr == ਖਾਲੀ_ਸੂਚਕ || chunk_ਸਿਰਾ > hint_addr) {
                hint_addr = chunk_ਸਿਰਾ;
            }
        }

        ਜੇ (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (chunk.ਅਧਾਰ == ਖਾਲੀ_ਸੂਚਕ || chunk.size < min_ਮਾਪ) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_ਮਾਪ, chunk.ਅਧਾਰ, chunk.size);
            dealloc_chunk(&chunk);
            ਵਾਪਸ ਝੂਠ;
        }

        ਕੋਸ਼ਿਸ਼ {
            chunk.free_ਖੰਡਾਂ.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } ਫੜੋ (ਅਚਲ std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ਸੁੱਟੋ;
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਖਾਲੀ rollback_ਵੰਡ_locked(ਅਚਲ pool_ਵੰਡ & ਵੰਡ) {
        ਸਵੈਚਾਲਿਤ chunk_it = ਲੱਭੋ_chunk_locked(ਵੰਡ);
        ਜੇ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, ਵੰਡ.ਅਧਾਰ);
            ਵਾਪਸ;
        }

        ਸਵੈਚਾਲਿਤ * chunk_ਅਧਾਰ = chunk_it->ਅਧਾਰ;
        ਸਵੈਚਾਲਿਤ * alloc_ਅਧਾਰ = static_cast<uint8_t *>(ਵੰਡ.ਅਧਾਰ);
        ਜੇ (alloc_ਅਧਾਰ < chunk_ਅਧਾਰ || alloc_ਅਧਾਰ >= chunk_ਅਧਾਰ + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, ਵੰਡ.ਅਧਾਰ);
            ਵਾਪਸ;
        }

        ਅਚਲ size_t ਵਿੱਥ = static_cast<size_t>(alloc_ਅਧਾਰ - chunk_ਅਧਾਰ);
        ਜੇ (ਵਿੱਥ > chunk_it->size || ਵੰਡ.size > chunk_it->size - ਵਿੱਥ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           ਵੰਡ.ਅਧਾਰ, ਵੰਡ.size);
            ਵਾਪਸ;
        }

        ਪਾਓ_free_ਖੰਡ_locked(*chunk_it, { ਵਿੱਥ, ਵੰਡ.size });
        maybe_release_ਖਾਲੀ_ਥਾਵਾਂ_chunk_locked(chunk_it);
    }

    ਤਾਰਕਿਕ try_alloc_locked(size_t size, size_t alignment, pool_ਵੰਡ * ਵੰਡ) {
        alloc_candidate ਉੱਤਮ;

        ਲਈ (size_t chunk_ਸੂਚਕਾਂਕ = 0; chunk_ਸੂਚਕਾਂਕ < chunks_.size(); ++chunk_ਸੂਚਕਾਂਕ) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ & chunk = chunks_[chunk_ਸੂਚਕਾਂਕ];
            ਲਈ (size_t ਖੰਡ_ਸੂਚਕਾਂਕ = 0; ਖੰਡ_ਸੂਚਕਾਂਕ < chunk.free_ਖੰਡਾਂ.size(); ++ਖੰਡ_ਸੂਚਕਾਂਕ) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ & ਖੰਡ = chunk.free_ਖੰਡਾਂ[ਖੰਡ_ਸੂਚਕਾਂਕ];

                uintptr_t  aligned_addr = 0;
                ਅਚਲ ਸਵੈਚਾਲਿਤ ਖੰਡ_addr   = reinterpret_cast<uintptr_t>(chunk.ਅਧਾਰ + ਖੰਡ.ਵਿੱਥ);
                ਜੇ (!align_ਉੱਪਰ_uintptr(ਖੰਡ_addr, alignment, &aligned_addr)) {
                    ਜਾਰੀ;
                }

                ਜੇ (aligned_addr < ਖੰਡ_addr) {
                    ਜਾਰੀ;
                }

                ਅਚਲ size_t aligned_ਵਿੱਥ = ਖੰਡ.ਵਿੱਥ + static_cast<size_t>(aligned_addr - ਖੰਡ_addr);
                ਅਚਲ size_t padding        = aligned_ਵਿੱਥ - ਖੰਡ.ਵਿੱਥ;
                ਜੇ (padding > ਖੰਡ.size || size > ਖੰਡ.size - padding) {
                    ਜਾਰੀ;
                }

                ਜੇ (!ਉੱਤਮ.valid || aligned_addr < ਉੱਤਮ.address) {
                    ਉੱਤਮ.chunk_ਸੂਚਕਾਂਕ    = chunk_ਸੂਚਕਾਂਕ;
                    ਉੱਤਮ.ਖੰਡ_ਸੂਚਕਾਂਕ    = ਖੰਡ_ਸੂਚਕਾਂਕ;
                    ਉੱਤਮ.aligned_ਵਿੱਥ = aligned_ਵਿੱਥ;
                    ਉੱਤਮ.address        = aligned_addr;
                    ਉੱਤਮ.valid          = ਸੱਚ;
                }
            }
        }

        ਜੇ (!ਉੱਤਮ.valid) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਸਵੈਚਾਲਿਤ &           chunk     = chunks_[ਉੱਤਮ.chunk_ਸੂਚਕਾਂਕ];
        ਅਚਲ free_ਖੰਡ ਖੰਡ     = chunk.free_ਖੰਡਾਂ[ਉੱਤਮ.ਖੰਡ_ਸੂਚਕਾਂਕ];
        ਅਚਲ size_t     padding   = ਉੱਤਮ.aligned_ਵਿੱਥ - ਖੰਡ.ਵਿੱਥ;
        ਅਚਲ size_t     alloc_ਸਿਰਾ = ਉੱਤਮ.aligned_ਵਿੱਥ + size;
        ਅਚਲ size_t     ਖੰਡ_ਸਿਰਾ = ਖੰਡ.ਵਿੱਥ + ਖੰਡ.size;

        chunk.free_ਖੰਡਾਂ.erase(chunk.free_ਖੰਡਾਂ.begin() + ਉੱਤਮ.ਖੰਡ_ਸੂਚਕਾਂਕ);
        ਸਵੈਚਾਲਿਤ ਪਾਓ_it = chunk.free_ਖੰਡਾਂ.begin() + ਉੱਤਮ.ਖੰਡ_ਸੂਚਕਾਂਕ;
        ਜੇ (padding != 0) {
            ਪਾਓ_it = chunk.free_ਖੰਡਾਂ.insert(ਪਾਓ_it, { ਖੰਡ.ਵਿੱਥ, padding });
            ++ਪਾਓ_it;
        }
        ਜੇ (alloc_ਸਿਰਾ < ਖੰਡ_ਸਿਰਾ) {
            chunk.free_ਖੰਡਾਂ.insert(ਪਾਓ_it, { alloc_ਸਿਰਾ, ਖੰਡ_ਸਿਰਾ - alloc_ਸਿਰਾ });
        }

        ਵੰਡ->chunk_ਅਧਾਰ = chunk.ਅਧਾਰ;
        ਵੰਡ->chunk_ਮਾਪ = chunk.size;
        ਵੰਡ->ਅਧਾਰ       = chunk.ਅਧਾਰ + ਉੱਤਮ.aligned_ਵਿੱਥ;
        ਵੰਡ->size       = size;
        ਵਾਪਸ ਸੱਚ;
    }

    ਖਾਲੀ maybe_release_ਖਾਲੀ_ਥਾਵਾਂ_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        ਜੇ (chunk_it->free_ਖੰਡਾਂ.size() != 1) {
            ਵਾਪਸ;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ & ਖੰਡ = chunk_it->free_ਖੰਡਾਂ.front();
        ਜੇ (ਖੰਡ.ਵਿੱਥ != 0 || ਖੰਡ.size != chunk_it->size) {
            ਵਾਪਸ;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    ਖਾਲੀ ਪਾਓ_free_ਖੰਡ_locked(pool_chunk & chunk, free_ਖੰਡ ਖੰਡ) {
        ਸਵੈਚਾਲਿਤ it = chunk.free_ਖੰਡਾਂ.begin();
        ਜਦੋਂ (it != chunk.free_ਖੰਡਾਂ.end() && it->ਵਿੱਥ < ਖੰਡ.ਵਿੱਥ) {
            ++it;
        }

        ਜੇ (it != chunk.free_ਖੰਡਾਂ.begin()) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ & prev = *(it - 1);
            ਜੇ (prev.ਵਿੱਥ + prev.size > ਖੰਡ.ਵਿੱਥ) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               ਖੰਡ.ਵਿੱਥ, ਖੰਡ.size);
                ਵਾਪਸ;
            }
        }

        ਜੇ (it != chunk.free_ਖੰਡਾਂ.end() && ਖੰਡ.ਵਿੱਥ + ਖੰਡ.size > it->ਵਿੱਥ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           ਖੰਡ.ਵਿੱਥ, ਖੰਡ.size);
            ਵਾਪਸ;
        }

        it = chunk.free_ਖੰਡਾਂ.insert(it, ਖੰਡ);

        ਜੇ (it != chunk.free_ਖੰਡਾਂ.begin()) {
            ਸਵੈਚਾਲਿਤ prev = it - 1;
            ਜੇ (prev->ਵਿੱਥ + prev->size == it->ਵਿੱਥ) {
                it->ਵਿੱਥ = prev->ਵਿੱਥ;
                it->size += prev->size;
                it = chunk.free_ਖੰਡਾਂ.erase(prev);
            }
        }

        ਜੇ (it + 1 != chunk.free_ਖੰਡਾਂ.end() && it->ਵਿੱਥ + it->size == (it + 1)->ਵਿੱਥ) {
            it->size += (it + 1)->size;
            chunk.free_ਖੰਡਾਂ.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<ਖਾਲੀ *, pool_ਵੰਡ> allocations_;
    size_t                                      default_chunk_ਮਾਪ_2{ 0 };
};

ਵਰਗ spine_mem_pool_posix final : ਜਨਤਕ spine_mem_pool_manager {
  ਜਨਤਕ:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  ਨਿਜੀ:
    ਤਾਰਕਿਕ alloc_chunk(size_t min_ਮਾਪ, size_t alignment, ਖਾਲੀ * hint_addr, pool_chunk * chunk) override {
        (ਖਾਲੀ) hint_addr;

        ਅਚਲ size_t alloc_alignment = std::max(alignment, ਆਕਾਰ(ਖਾਲੀ *));
        ਖਾਲੀ *       ਅਧਾਰ            = ਖਾਲੀ_ਸੂਚਕ;
        ਅਚਲ ਪੂਰਨ    rc              = posix_memalign(&ਅਧਾਰ, alloc_alignment, min_ਮਾਪ);
        ਜੇ (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_ਮਾਪ, alloc_alignment, rc);
            ਵਾਪਸ ਝੂਠ;
        }

        chunk->ਅਧਾਰ = static_cast<uint8_t *>(ਅਧਾਰ);
        chunk->size = min_ਮਾਪ;
        chunk->fd   = -1;
        ਵਾਪਸ ਸੱਚ;
    }

    ਖਾਲੀ dealloc_chunk(pool_chunk * chunk) override {
        std::ਮੁਕਤ(chunk->ਅਧਾਰ);
        clear_chunk(chunk);
    }
};

ਵਰਗ spine_mem_pool_transparent_hugepage final : ਜਨਤਕ spine_mem_pool_manager {
  ਜਨਤਕ:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_ਮਾਪ) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  ਨਿਜੀ:
    ਤਾਰਕਿਕ alloc_chunk(size_t min_ਮਾਪ, size_t alignment, ਖਾਲੀ * hint_addr, pool_chunk * chunk) override {
        (ਖਾਲੀ) alignment;

        size_t chunk_ਮਾਪ = 0;
        ਜੇ (!align_ਉੱਪਰ(min_ਮਾਪ, default_chunk_ਮਾਪ(), &chunk_ਮਾਪ)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_ਮਾਪ);
            ਵਾਪਸ ਝੂਠ;
        }

        ਖਾਲੀ * map_addr = mmap(hint_addr, chunk_ਮਾਪ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ਜੇ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_ਮਾਪ,
                           errno);
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (madvise(map_addr, chunk_ਮਾਪ, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_ਮਾਪ, errno);
            munmap(map_addr, chunk_ਮਾਪ);
            ਵਾਪਸ ਝੂਠ;
        }

        chunk->ਅਧਾਰ = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_ਮਾਪ;
        chunk->fd   = -1;
        ਵਾਪਸ ਸੱਚ;
    }

    ਖਾਲੀ dealloc_chunk(pool_chunk * chunk) override {
        ਜੇ (chunk->ਅਧਾਰ != ਖਾਲੀ_ਸੂਚਕ && chunk->size != 0 && munmap(chunk->ਅਧਾਰ, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->ਅਧਾਰ, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

ਵਰਗ spine_mem_pool_hugetlb_1g final : ਜਨਤਕ spine_mem_pool_manager {
  ਜਨਤਕ:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1_G_REGION_ਮਾਪ) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  ਨਿਜੀ:
    ਤਾਰਕਿਕ alloc_chunk(size_t min_ਮਾਪ, size_t alignment, ਖਾਲੀ * hint_addr, pool_chunk * chunk) override {
        (ਖਾਲੀ) alignment;
        (ਖਾਲੀ) hint_addr;

        size_t region_ਮਾਪ = 0;
        ਜੇ (!align_ਉੱਪਰ(min_ਮਾਪ, SPINE_MEM_POOL_1_G_REGION_ਮਾਪ, &region_ਮਾਪ)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_ਮਾਪ);
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਪੂਰਨ fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        ਜੇ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            ਵਾਪਸ ਝੂਠ;
        }

        hugetlb_1g_region region;
        region.size  = region_ਮਾਪ;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        ਜੇ (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_ਮਾਪ, errno);
            close(fd);
            ਵਾਪਸ ਝੂਠ;
        }

        ਖਾਲੀ * map_addr = mmap(ਖਾਲੀ_ਸੂਚਕ, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ਜੇ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            ਵਾਪਸ ਝੂਠ;
        }

        chunk->ਅਧਾਰ = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        ਵਾਪਸ ਸੱਚ;
    }

    ਖਾਲੀ dealloc_chunk(pool_chunk * chunk) override {
        ਜੇ (chunk->ਅਧਾਰ != ਖਾਲੀ_ਸੂਚਕ && chunk->size != 0 && munmap(chunk->ਅਧਾਰ, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->ਅਧਾਰ, chunk->size, errno);
        }

        ਜੇ (chunk->fd >= 0) {
            ਜੇ (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->ਅਧਾਰ, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

ਵਰਗ spine_mem_pool_shared_mem final : ਜਨਤਕ spine_mem_pool_manager {
  ਜਨਤਕ:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_ਮਾਪ) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  ਨਿਜੀ:
    ਤਾਰਕਿਕ alloc_chunk(size_t min_ਮਾਪ, size_t alignment, ਖਾਲੀ * hint_addr, pool_chunk * chunk) override {
        (ਖਾਲੀ) alignment;

        ਜੇ (hint_addr != ਖਾਲੀ_ਸੂਚਕ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (min_ਮਾਪ > default_chunk_ਮਾਪ()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_ਮਾਪ, default_chunk_ਮਾਪ());
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਪੂਰਨ fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        ਜੇ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            ਵਾਪਸ ਝੂਠ;
        }

        ਖਾਲੀ * map_addr = mmap(ਖਾਲੀ_ਸੂਚਕ, default_chunk_ਮਾਪ(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ਜੇ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_ਮਾਪ(), errno);
            close(fd);
            ਵਾਪਸ ਝੂਠ;
        }

        chunk->ਅਧਾਰ = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_ਮਾਪ();
        chunk->fd   = fd;
        ਵਾਪਸ ਸੱਚ;
    }

    ਖਾਲੀ dealloc_chunk(pool_chunk * chunk) override {
        ਜੇ (chunk->ਅਧਾਰ != ਖਾਲੀ_ਸੂਚਕ && chunk->size != 0 && munmap(chunk->ਅਧਾਰ, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->ਅਧਾਰ, chunk->size, errno);
        }

        ਜੇ (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    ਸਥਿਰ std::once_flag                          pool_once;
    ਸਥਿਰ std::unique_ptr<spine_mem_pool_manager> selected_pool;
    ਸਥਿਰ spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    ਜੇ (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        ਚੋਣ (selected_backend) {
            ਹਾਲਤ spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                ਰੋਕੋ;
            ਹਾਲਤ spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ਰੋਕੋ;
            ਹਾਲਤ spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                ਰੋਕੋ;
            ਹਾਲਤ spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ਰੋਕੋ;
        }
    });

    ਜੇ (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<ਪੂਰਨ>(backend), static_cast<ਪੂਰਨ>(selected_backend));
    }

    ਜੇ (selected_pool) {
        ਵਾਪਸ *selected_pool;
    }

    ਸੁੱਟੋ std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    ਸਥਿਰ std::once_flag                             shared_mem_pool_once;
    ਸਥਿਰ std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    ਜੇ (shared_mem_pool) {
        ਵਾਪਸ *shared_mem_pool;
    }

    ਸੁੱਟੋ std::bad_alloc();
}

}  // namespace

ਤਾਰਕਿਕ spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    ਜੇ (info == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਝੂਠ;
    }

    *info = {};

    ਜੇ (spine_tcm_open_handle(ਨਿਰਾ) != 0 || !spine_tcm_is_ਉਪਲਬਧ()) {
        ਵਾਪਸ ਝੂਠ;
    }

    spine_tcm_mem_info_t mem_info;
    ਜੇ (spine_tcm_mem_info(&mem_info) != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    info->ਉਪਲਬਧ   = ਸੱਚ;
    info->blk_ਮਾਪ    = mem_info.blk_ਮਾਪ;
    info->blk_ਅੰਕ     = mem_info.blk_ਅੰਕ;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    ਵਾਪਸ ਸੱਚ;
}

ਖਾਲੀ * spine_mem_pool_tcm_mem_get(ਪੂਰਨ cpu_id) noexcept {
    ਵਾਪਸ spine_tcm_mem_get(cpu_id);
}

ਖਾਲੀ * spine_mem_pool_tcm_mem_wait(ਪੂਰਨ cpu_id) noexcept {
    ਵਾਪਸ spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

ਪੂਰਨ spine_mem_pool_tcm_mem_release(ਪੂਰਨ cpu_id) noexcept {
    ਵਾਪਸ spine_tcm_mem_release(cpu_id);
}

ਖਾਲੀ * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    ਕੋਸ਼ਿਸ਼ {
        ਵਾਪਸ get_spine_mem_pool_manager().alloc(size, alignment);
    } ਫੜੋ (ਅਚਲ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
}

ਖਾਲੀ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    ਕੋਸ਼ਿਸ਼ {
        ਵਾਪਸ get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } ਫੜੋ (ਅਚਲ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
}

ਖਾਲੀ spine_mem_pool_free(ਖਾਲੀ * ਅਧਾਰ) noexcept {
    ਕੋਸ਼ਿਸ਼ {
        get_spine_mem_pool_manager().ਮੁਕਤ(ਅਧਾਰ);
    } ਫੜੋ (ਅਚਲ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, ਅਧਾਰ);
    }
}

ਖਾਲੀ spine_mem_pool_shared_mem_free(ਖਾਲੀ * ਅਧਾਰ) noexcept {
    ਕੋਸ਼ਿਸ਼ {
        get_spine_mem_pool_shared_mem_manager().ਮੁਕਤ(ਅਧਾਰ);
    } ਫੜੋ (ਅਚਲ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, ਅਧਾਰ);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

ਬਾਹਰੀ "C" {
ਖਾਲੀ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    ਖਾਲੀ * ਨਤੀਜਾ = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    ਜੇ (ਨਤੀਜਾ == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਖਾਲੀ ggml_backend_cpu_riscv64_spacemit_free_shared(ਖਾਲੀ * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
