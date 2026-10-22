#ಅನ್ತರ್ಭಾವ "spine_mem_pool.h"

#ಅನ್ತರ್ಭಾವ "common.h"
#ಅನ್ತರ್ಭಾವ "ime_env.h"
#ಅನ್ತರ್ಭಾವ "spine_tcm.h"

#ಅನ್ತರ್ಭಾವ <fcntl.h>
#ಅನ್ತರ್ಭಾವ <sys/ioctl.h>
#ಅನ್ತರ್ಭಾವ <sys/mman.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <limits>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ggml::cpu::riscv64_spacemit {
ನಾಮಾಕಾಶ {

constexpr size_t   SPINE_MEM_POOL_CHUNK_ಗಾತ್ರ         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_ಗಾತ್ರ   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1_G_REGION_ಗಾತ್ರ     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr ಅಕ್ಷರ     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr ಅಕ್ಷರ     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

ರಚನೆ hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#ವ್ಯಾಖ್ಯಾನಿಸು HUGETLB_1G_IOC_MAGIC 'M'
#ವ್ಯಾಖ್ಯಾನಿಸು HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, ರಚನೆ hugetlb_1g_region)
#ವ್ಯಾಖ್ಯಾನಿಸು HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

ರಚನೆ free_ಖಂಡ {
    size_t ಸ್ಥಾನಾಂತರ{ 0 };
    size_t size{ 0 };
};

ರಚನೆ pool_chunk {
    uint8_t *               ತಳ{ ಶೂನ್ಯಸೂಚಕ };
    size_t                  size{ 0 };
    ಪೂರ್ಣಾಂಕ                     fd{ -1 };
    std::vector<free_ಖಂಡ> free_ಖಂಡಗಳು;
};

ರಚನೆ pool_ಹಂಚಿಕೆ {
    ಶೂನ್ಯ * chunk_ತಳ{ ಶೂನ್ಯಸೂಚಕ };
    size_t chunk_ಗಾತ್ರ{ 0 };
    ಶೂನ್ಯ * ತಳ{ ಶೂನ್ಯಸೂಚಕ };
    size_t size{ 0 };
};

ತಾರ್ಕಿಕ is_power_ದ_two(size_t value) {
    ಹಿಂತಿರುಗಿ value != 0 && (value & (value - 1)) == 0;
}

ತಾರ್ಕಿಕ align_ಮೇಲೆ(size_t value, size_t alignment, size_t * aligned_ಮೌಲ್ಯ) {
    ಆದರೆ (aligned_ಮೌಲ್ಯ == ಶೂನ್ಯಸೂಚಕ || alignment == 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ size_t remainder = value % alignment;
    ಆದರೆ (remainder == 0) {
        *aligned_ಮೌಲ್ಯ = value;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಸ್ಥಿರಾಂಕ size_t padding = alignment - remainder;
    ಆದರೆ (value > std::numeric_limits<size_t>::max() - padding) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    *aligned_ಮೌಲ್ಯ = value + padding;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ align_ಮೇಲೆ_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_ಮೌಲ್ಯ) {
    ಆದರೆ (aligned_ಮೌಲ್ಯ == ಶೂನ್ಯಸೂಚಕ || alignment == 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uintptr_t remainder = value % alignment;
    ಆದರೆ (remainder == 0) {
        *aligned_ಮೌಲ್ಯ = value;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಸ್ಥಿರಾಂಕ uintptr_t padding = alignment - remainder;
    ಆದರೆ (value > std::numeric_limits<uintptr_t>::max() - padding) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    *aligned_ಮೌಲ್ಯ = value + padding;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ವರ್ಗ spine_mem_pool_manager {
  ಸಾರ್ವಜನಿಕ:
    ಸ್ಪಷ್ಟ spine_mem_pool_manager(size_t default_chunk_ಗಾತ್ರ) : default_chunk_ಗಾತ್ರ_2(default_chunk_ಗಾತ್ರ) {}

    ವಾಸ್ತವಿಕ ~spine_mem_pool_manager() = ಪೂರ್ವನಿಯೋಜಿತ;

    ಶೂನ್ಯ * alloc(size_t size, size_t alignment) {
        ಆದರೆ (size == 0 || !is_power_ದ_two(alignment)) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        size_t aligned_ಗಾತ್ರ = 0;
        ಆದರೆ (!align_ಮೇಲೆ(size, alignment, &aligned_ಗಾತ್ರ)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        pool_ಹಂಚಿಕೆ ಹಂಚಿಕೆ;

        std::lock_guard<std::mutex> ಬೀಗ(mutex_);

        ಆದರೆ (!try_alloc_locked(aligned_ಗಾತ್ರ, alignment, &ಹಂಚಿಕೆ)) {
            ಆದರೆ (!ಕೂಡಿಸು_chunk_locked(aligned_ಗಾತ್ರ, alignment)) {
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }

            ಆದರೆ (!try_alloc_locked(aligned_ಗಾತ್ರ, alignment, &ಹಂಚಿಕೆ)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_ಗಾತ್ರ, alignment);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
        }

        ಪ್ರಯತ್ನಿಸು {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ [ಹಂಚಿಕೆ_it, inserted] = allocations_.emplace(ಹಂಚಿಕೆ.ತಳ, ಹಂಚಿಕೆ);
            ಆದರೆ (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, ಹಂಚಿಕೆ.ತಳ);
                rollback_ಹಂಚಿಕೆ_locked(ಹಂಚಿಕೆ);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
        } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::bad_alloc &) {
            rollback_ಹಂಚಿಕೆ_locked(ಹಂಚಿಕೆ);
            ಎಸೆ;
        }

        ಹಿಂತಿರುಗಿ ಹಂಚಿಕೆ.ತಳ;
    }

    ಶೂನ್ಯ ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ತಳ) {
        ಆದರೆ (ತಳ == ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ;
        }

        std::lock_guard<std::mutex> ಬೀಗ(mutex_);

        ಸ್ವಯಂಚಾಲಿತ ಹಂಚಿಕೆ_it = allocations_.find(ತಳ);
        ಆದರೆ (ಹಂಚಿಕೆ_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, ತಳ);
            ಹಿಂತಿರುಗಿ;
        }

        pool_ಹಂಚಿಕೆ ಹಂಚಿಕೆ = ಹಂಚಿಕೆ_it->second;
        allocations_.erase(ಹಂಚಿಕೆ_it);

        ಸ್ವಯಂಚಾಲಿತ chunk_it = ಕಂಡುಹಿಡಿ_chunk_locked(ಹಂಚಿಕೆ);
        ಆದರೆ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           ಹಂಚಿಕೆ.ತಳ, ಹಂಚಿಕೆ.size);
            ಹಿಂತಿರುಗಿ;
        }

        ಸ್ವಯಂಚಾಲಿತ * chunk_ತಳ = chunk_it->ತಳ;
        ಸ್ವಯಂಚಾಲಿತ * alloc_ತಳ = static_cast<uint8_t *>(ಹಂಚಿಕೆ.ತಳ);
        ಆದರೆ (alloc_ತಳ < chunk_ತಳ || alloc_ತಳ >= chunk_ತಳ + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           ಹಂಚಿಕೆ.ತಳ, chunk_ತಳ, chunk_ತಳ + chunk_it->size);
            ಹಿಂತಿರುಗಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ = static_cast<size_t>(alloc_ತಳ - chunk_ತಳ);
        ಆದರೆ (ಸ್ಥಾನಾಂತರ > chunk_it->size || ಹಂಚಿಕೆ.size > chunk_it->size - ಸ್ಥಾನಾಂತರ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           ಹಂಚಿಕೆ.ತಳ, ಹಂಚಿಕೆ.size, chunk_it->size);
            ಹಿಂತಿರುಗಿ;
        }

        ಸೇರಿಸು_free_ಖಂಡ_locked(*chunk_it, { ಸ್ಥಾನಾಂತರ, ಹಂಚಿಕೆ.size });
        maybe_release_ಖಾಲಿಗಳು_chunk_locked(chunk_it);
    }

  ಸಂರಕ್ಷಿತ:
    ಶೂನ್ಯ release_chunks() {
        std::lock_guard<std::mutex> ಬೀಗ(mutex_);

        allocations_.clear();
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ default_chunk_ಗಾತ್ರ_2; }

    ಸ್ಥಿರ ಶೂನ್ಯ clear_chunk(pool_chunk * chunk) {
        chunk->ತಳ = ಶೂನ್ಯಸೂಚಕ;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_ಖಂಡಗಳು.clear();
    }

    ವಾಸ್ತವಿಕ ತಾರ್ಕಿಕ alloc_chunk(size_t min_ಗಾತ್ರ, size_t alignment, ಶೂನ್ಯ * hint_addr, pool_chunk * chunk) = 0;
    ವಾಸ್ತವಿಕ ಶೂನ್ಯ dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  ಖಾಸಗಿ:
    ರಚನೆ alloc_candidate {
        size_t    chunk_ಸೂಚ್ಯಂಕ{ 0 };
        size_t    ಖಂಡ_ಸೂಚ್ಯಂಕ{ 0 };
        size_t    aligned_ಸ್ಥಾನಾಂತರ{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        ತಾರ್ಕಿಕ      valid{ ಸುಳ್ಳು };
    };

    std::vector<pool_chunk>::iterator ಕಂಡುಹಿಡಿ_chunk_locked(ಸ್ಥಿರಾಂಕ pool_ಹಂಚಿಕೆ & ಹಂಚಿಕೆ) {
        ಹಿಂತಿರುಗಿ std::find_if(chunks_.begin(), chunks_.end(), [&](ಸ್ಥಿರಾಂಕ pool_chunk & chunk) {
            ಹಿಂತಿರುಗಿ chunk.ತಳ == ಹಂಚಿಕೆ.chunk_ತಳ && chunk.size == ಹಂಚಿಕೆ.chunk_ಗಾತ್ರ;
        });
    }

    ತಾರ್ಕಿಕ ಕೂಡಿಸು_chunk_locked(size_t min_ಗಾತ್ರ, size_t alignment) {
        pool_chunk   chunk;
        ಸ್ಥಿರಾಂಕ size_t chunk_request = default_chunk_ಗಾತ್ರ_2 == 0 ? min_ಗಾತ್ರ : std::max(min_ಗಾತ್ರ, default_chunk_ಗಾತ್ರ_2);
        ಶೂನ್ಯ *       hint_addr     = ಶೂನ್ಯಸೂಚಕ;

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & existing_chunk : chunks_) {
            ಸ್ವಯಂಚಾಲಿತ * chunk_ಕೊನೆ = existing_chunk.ತಳ + existing_chunk.size;
            ಆದರೆ (hint_addr == ಶೂನ್ಯಸೂಚಕ || chunk_ಕೊನೆ > hint_addr) {
                hint_addr = chunk_ಕೊನೆ;
            }
        }

        ಆದರೆ (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (chunk.ತಳ == ಶೂನ್ಯಸೂಚಕ || chunk.size < min_ಗಾತ್ರ) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_ಗಾತ್ರ, chunk.ತಳ, chunk.size);
            dealloc_chunk(&chunk);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಪ್ರಯತ್ನಿಸು {
            chunk.free_ಖಂಡಗಳು.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ಎಸೆ;
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಶೂನ್ಯ rollback_ಹಂಚಿಕೆ_locked(ಸ್ಥಿರಾಂಕ pool_ಹಂಚಿಕೆ & ಹಂಚಿಕೆ) {
        ಸ್ವಯಂಚಾಲಿತ chunk_it = ಕಂಡುಹಿಡಿ_chunk_locked(ಹಂಚಿಕೆ);
        ಆದರೆ (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, ಹಂಚಿಕೆ.ತಳ);
            ಹಿಂತಿರುಗಿ;
        }

        ಸ್ವಯಂಚಾಲಿತ * chunk_ತಳ = chunk_it->ತಳ;
        ಸ್ವಯಂಚಾಲಿತ * alloc_ತಳ = static_cast<uint8_t *>(ಹಂಚಿಕೆ.ತಳ);
        ಆದರೆ (alloc_ತಳ < chunk_ತಳ || alloc_ತಳ >= chunk_ತಳ + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, ಹಂಚಿಕೆ.ತಳ);
            ಹಿಂತಿರುಗಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ = static_cast<size_t>(alloc_ತಳ - chunk_ತಳ);
        ಆದರೆ (ಸ್ಥಾನಾಂತರ > chunk_it->size || ಹಂಚಿಕೆ.size > chunk_it->size - ಸ್ಥಾನಾಂತರ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           ಹಂಚಿಕೆ.ತಳ, ಹಂಚಿಕೆ.size);
            ಹಿಂತಿರುಗಿ;
        }

        ಸೇರಿಸು_free_ಖಂಡ_locked(*chunk_it, { ಸ್ಥಾನಾಂತರ, ಹಂಚಿಕೆ.size });
        maybe_release_ಖಾಲಿಗಳು_chunk_locked(chunk_it);
    }

    ತಾರ್ಕಿಕ try_alloc_locked(size_t size, size_t alignment, pool_ಹಂಚಿಕೆ * ಹಂಚಿಕೆ) {
        alloc_candidate ಉತ್ತಮ;

        ಗಾಗಿ (size_t chunk_ಸೂಚ್ಯಂಕ = 0; chunk_ಸೂಚ್ಯಂಕ < chunks_.size(); ++chunk_ಸೂಚ್ಯಂಕ) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & chunk = chunks_[chunk_ಸೂಚ್ಯಂಕ];
            ಗಾಗಿ (size_t ಖಂಡ_ಸೂಚ್ಯಂಕ = 0; ಖಂಡ_ಸೂಚ್ಯಂಕ < chunk.free_ಖಂಡಗಳು.size(); ++ಖಂಡ_ಸೂಚ್ಯಂಕ) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಖಂಡ = chunk.free_ಖಂಡಗಳು[ಖಂಡ_ಸೂಚ್ಯಂಕ];

                uintptr_t  aligned_addr = 0;
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಖಂಡ_addr   = reinterpret_cast<uintptr_t>(chunk.ತಳ + ಖಂಡ.ಸ್ಥಾನಾಂತರ);
                ಆದರೆ (!align_ಮೇಲೆ_uintptr(ಖಂಡ_addr, alignment, &aligned_addr)) {
                    ಮುಂದುವರಿ;
                }

                ಆದರೆ (aligned_addr < ಖಂಡ_addr) {
                    ಮುಂದುವರಿ;
                }

                ಸ್ಥಿರಾಂಕ size_t aligned_ಸ್ಥಾನಾಂತರ = ಖಂಡ.ಸ್ಥಾನಾಂತರ + static_cast<size_t>(aligned_addr - ಖಂಡ_addr);
                ಸ್ಥಿರಾಂಕ size_t padding        = aligned_ಸ್ಥಾನಾಂತರ - ಖಂಡ.ಸ್ಥಾನಾಂತರ;
                ಆದರೆ (padding > ಖಂಡ.size || size > ಖಂಡ.size - padding) {
                    ಮುಂದುವರಿ;
                }

                ಆದರೆ (!ಉತ್ತಮ.valid || aligned_addr < ಉತ್ತಮ.address) {
                    ಉತ್ತಮ.chunk_ಸೂಚ್ಯಂಕ    = chunk_ಸೂಚ್ಯಂಕ;
                    ಉತ್ತಮ.ಖಂಡ_ಸೂಚ್ಯಂಕ    = ಖಂಡ_ಸೂಚ್ಯಂಕ;
                    ಉತ್ತಮ.aligned_ಸ್ಥಾನಾಂತರ = aligned_ಸ್ಥಾನಾಂತರ;
                    ಉತ್ತಮ.address        = aligned_addr;
                    ಉತ್ತಮ.valid          = ನಿಜ;
                }
            }
        }

        ಆದರೆ (!ಉತ್ತಮ.valid) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ವಯಂಚಾಲಿತ &           chunk     = chunks_[ಉತ್ತಮ.chunk_ಸೂಚ್ಯಂಕ];
        ಸ್ಥಿರಾಂಕ free_ಖಂಡ ಖಂಡ     = chunk.free_ಖಂಡಗಳು[ಉತ್ತಮ.ಖಂಡ_ಸೂಚ್ಯಂಕ];
        ಸ್ಥಿರಾಂಕ size_t     padding   = ಉತ್ತಮ.aligned_ಸ್ಥಾನಾಂತರ - ಖಂಡ.ಸ್ಥಾನಾಂತರ;
        ಸ್ಥಿರಾಂಕ size_t     alloc_ಕೊನೆ = ಉತ್ತಮ.aligned_ಸ್ಥಾನಾಂತರ + size;
        ಸ್ಥಿರಾಂಕ size_t     ಖಂಡ_ಕೊನೆ = ಖಂಡ.ಸ್ಥಾನಾಂತರ + ಖಂಡ.size;

        chunk.free_ಖಂಡಗಳು.erase(chunk.free_ಖಂಡಗಳು.begin() + ಉತ್ತಮ.ಖಂಡ_ಸೂಚ್ಯಂಕ);
        ಸ್ವಯಂಚಾಲಿತ ಸೇರಿಸು_it = chunk.free_ಖಂಡಗಳು.begin() + ಉತ್ತಮ.ಖಂಡ_ಸೂಚ್ಯಂಕ;
        ಆದರೆ (padding != 0) {
            ಸೇರಿಸು_it = chunk.free_ಖಂಡಗಳು.insert(ಸೇರಿಸು_it, { ಖಂಡ.ಸ್ಥಾನಾಂತರ, padding });
            ++ಸೇರಿಸು_it;
        }
        ಆದರೆ (alloc_ಕೊನೆ < ಖಂಡ_ಕೊನೆ) {
            chunk.free_ಖಂಡಗಳು.insert(ಸೇರಿಸು_it, { alloc_ಕೊನೆ, ಖಂಡ_ಕೊನೆ - alloc_ಕೊನೆ });
        }

        ಹಂಚಿಕೆ->chunk_ತಳ = chunk.ತಳ;
        ಹಂಚಿಕೆ->chunk_ಗಾತ್ರ = chunk.size;
        ಹಂಚಿಕೆ->ತಳ       = chunk.ತಳ + ಉತ್ತಮ.aligned_ಸ್ಥಾನಾಂತರ;
        ಹಂಚಿಕೆ->size       = size;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಶೂನ್ಯ maybe_release_ಖಾಲಿಗಳು_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        ಆದರೆ (chunk_it->free_ಖಂಡಗಳು.size() != 1) {
            ಹಿಂತಿರುಗಿ;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಖಂಡ = chunk_it->free_ಖಂಡಗಳು.front();
        ಆದರೆ (ಖಂಡ.ಸ್ಥಾನಾಂತರ != 0 || ಖಂಡ.size != chunk_it->size) {
            ಹಿಂತಿರುಗಿ;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    ಶೂನ್ಯ ಸೇರಿಸು_free_ಖಂಡ_locked(pool_chunk & chunk, free_ಖಂಡ ಖಂಡ) {
        ಸ್ವಯಂಚಾಲಿತ it = chunk.free_ಖಂಡಗಳು.begin();
        ಆಗ (it != chunk.free_ಖಂಡಗಳು.end() && it->ಸ್ಥಾನಾಂತರ < ಖಂಡ.ಸ್ಥಾನಾಂತರ) {
            ++it;
        }

        ಆದರೆ (it != chunk.free_ಖಂಡಗಳು.begin()) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & prev = *(it - 1);
            ಆದರೆ (prev.ಸ್ಥಾನಾಂತರ + prev.size > ಖಂಡ.ಸ್ಥಾನಾಂತರ) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               ಖಂಡ.ಸ್ಥಾನಾಂತರ, ಖಂಡ.size);
                ಹಿಂತಿರುಗಿ;
            }
        }

        ಆದರೆ (it != chunk.free_ಖಂಡಗಳು.end() && ಖಂಡ.ಸ್ಥಾನಾಂತರ + ಖಂಡ.size > it->ಸ್ಥಾನಾಂತರ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           ಖಂಡ.ಸ್ಥಾನಾಂತರ, ಖಂಡ.size);
            ಹಿಂತಿರುಗಿ;
        }

        it = chunk.free_ಖಂಡಗಳು.insert(it, ಖಂಡ);

        ಆದರೆ (it != chunk.free_ಖಂಡಗಳು.begin()) {
            ಸ್ವಯಂಚಾಲಿತ prev = it - 1;
            ಆದರೆ (prev->ಸ್ಥಾನಾಂತರ + prev->size == it->ಸ್ಥಾನಾಂತರ) {
                it->ಸ್ಥಾನಾಂತರ = prev->ಸ್ಥಾನಾಂತರ;
                it->size += prev->size;
                it = chunk.free_ಖಂಡಗಳು.erase(prev);
            }
        }

        ಆದರೆ (it + 1 != chunk.free_ಖಂಡಗಳು.end() && it->ಸ್ಥಾನಾಂತರ + it->size == (it + 1)->ಸ್ಥಾನಾಂತರ) {
            it->size += (it + 1)->size;
            chunk.free_ಖಂಡಗಳು.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<ಶೂನ್ಯ *, pool_ಹಂಚಿಕೆ> allocations_;
    size_t                                      default_chunk_ಗಾತ್ರ_2{ 0 };
};

ವರ್ಗ spine_mem_pool_posix final : ಸಾರ್ವಜನಿಕ spine_mem_pool_manager {
  ಸಾರ್ವಜನಿಕ:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  ಖಾಸಗಿ:
    ತಾರ್ಕಿಕ alloc_chunk(size_t min_ಗಾತ್ರ, size_t alignment, ಶೂನ್ಯ * hint_addr, pool_chunk * chunk) override {
        (ಶೂನ್ಯ) hint_addr;

        ಸ್ಥಿರಾಂಕ size_t alloc_alignment = std::max(alignment, ಗಾತ್ರ(ಶೂನ್ಯ *));
        ಶೂನ್ಯ *       ತಳ            = ಶೂನ್ಯಸೂಚಕ;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    rc              = posix_memalign(&ತಳ, alloc_alignment, min_ಗಾತ್ರ);
        ಆದರೆ (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_ಗಾತ್ರ, alloc_alignment, rc);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        chunk->ತಳ = static_cast<uint8_t *>(ತಳ);
        chunk->size = min_ಗಾತ್ರ;
        chunk->fd   = -1;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಶೂನ್ಯ dealloc_chunk(pool_chunk * chunk) override {
        std::ಬಿಡುಗಡೆ(chunk->ತಳ);
        clear_chunk(chunk);
    }
};

ವರ್ಗ spine_mem_pool_transparent_hugepage final : ಸಾರ್ವಜನಿಕ spine_mem_pool_manager {
  ಸಾರ್ವಜನಿಕ:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_ಗಾತ್ರ) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  ಖಾಸಗಿ:
    ತಾರ್ಕಿಕ alloc_chunk(size_t min_ಗಾತ್ರ, size_t alignment, ಶೂನ್ಯ * hint_addr, pool_chunk * chunk) override {
        (ಶೂನ್ಯ) alignment;

        size_t chunk_ಗಾತ್ರ = 0;
        ಆದರೆ (!align_ಮೇಲೆ(min_ಗಾತ್ರ, default_chunk_ಗಾತ್ರ(), &chunk_ಗಾತ್ರ)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_ಗಾತ್ರ);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಶೂನ್ಯ * map_addr = mmap(hint_addr, chunk_ಗಾತ್ರ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ಆದರೆ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_ಗಾತ್ರ,
                           errno);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (madvise(map_addr, chunk_ಗಾತ್ರ, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_ಗಾತ್ರ, errno);
            munmap(map_addr, chunk_ಗಾತ್ರ);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        chunk->ತಳ = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_ಗಾತ್ರ;
        chunk->fd   = -1;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಶೂನ್ಯ dealloc_chunk(pool_chunk * chunk) override {
        ಆದರೆ (chunk->ತಳ != ಶೂನ್ಯಸೂಚಕ && chunk->size != 0 && munmap(chunk->ತಳ, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->ತಳ, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

ವರ್ಗ spine_mem_pool_hugetlb_1g final : ಸಾರ್ವಜನಿಕ spine_mem_pool_manager {
  ಸಾರ್ವಜನಿಕ:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1_G_REGION_ಗಾತ್ರ) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  ಖಾಸಗಿ:
    ತಾರ್ಕಿಕ alloc_chunk(size_t min_ಗಾತ್ರ, size_t alignment, ಶೂನ್ಯ * hint_addr, pool_chunk * chunk) override {
        (ಶೂನ್ಯ) alignment;
        (ಶೂನ್ಯ) hint_addr;

        size_t region_ಗಾತ್ರ = 0;
        ಆದರೆ (!align_ಮೇಲೆ(min_ಗಾತ್ರ, SPINE_MEM_POOL_1_G_REGION_ಗಾತ್ರ, &region_ಗಾತ್ರ)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_ಗಾತ್ರ);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        ಆದರೆ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        hugetlb_1g_region region;
        region.size  = region_ಗಾತ್ರ;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        ಆದರೆ (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_ಗಾತ್ರ, errno);
            close(fd);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಶೂನ್ಯ * map_addr = mmap(ಶೂನ್ಯಸೂಚಕ, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ಆದರೆ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        chunk->ತಳ = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಶೂನ್ಯ dealloc_chunk(pool_chunk * chunk) override {
        ಆದರೆ (chunk->ತಳ != ಶೂನ್ಯಸೂಚಕ && chunk->size != 0 && munmap(chunk->ತಳ, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->ತಳ, chunk->size, errno);
        }

        ಆದರೆ (chunk->fd >= 0) {
            ಆದರೆ (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->ತಳ, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

ವರ್ಗ spine_mem_pool_shared_mem final : ಸಾರ್ವಜನಿಕ spine_mem_pool_manager {
  ಸಾರ್ವಜನಿಕ:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_ಗಾತ್ರ) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  ಖಾಸಗಿ:
    ತಾರ್ಕಿಕ alloc_chunk(size_t min_ಗಾತ್ರ, size_t alignment, ಶೂನ್ಯ * hint_addr, pool_chunk * chunk) override {
        (ಶೂನ್ಯ) alignment;

        ಆದರೆ (hint_addr != ಶೂನ್ಯಸೂಚಕ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (min_ಗಾತ್ರ > default_chunk_ಗಾತ್ರ()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_ಗಾತ್ರ, default_chunk_ಗಾತ್ರ());
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        ಆದರೆ (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಶೂನ್ಯ * map_addr = mmap(ಶೂನ್ಯಸೂಚಕ, default_chunk_ಗಾತ್ರ(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ಆದರೆ (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_ಗಾತ್ರ(), errno);
            close(fd);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        chunk->ತಳ = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_ಗಾತ್ರ();
        chunk->fd   = fd;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಶೂನ್ಯ dealloc_chunk(pool_chunk * chunk) override {
        ಆದರೆ (chunk->ತಳ != ಶೂನ್ಯಸೂಚಕ && chunk->size != 0 && munmap(chunk->ತಳ, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->ತಳ, chunk->size, errno);
        }

        ಆದರೆ (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    ಸ್ಥಿರ std::once_flag                          pool_once;
    ಸ್ಥಿರ std::unique_ptr<spine_mem_pool_manager> selected_pool;
    ಸ್ಥಿರ spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    ಆದರೆ (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        ಆಯ್ಕೆ (selected_backend) {
            ಸ್ಥಿತಿ spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                ನಿಲ್ಲಿಸು;
        }
    });

    ಆದರೆ (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<ಪೂರ್ಣಾಂಕ>(backend), static_cast<ಪೂರ್ಣಾಂಕ>(selected_backend));
    }

    ಆದರೆ (selected_pool) {
        ಹಿಂತಿರುಗಿ *selected_pool;
    }

    ಎಸೆ std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    ಸ್ಥಿರ std::once_flag                             shared_mem_pool_once;
    ಸ್ಥಿರ std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    ಆದರೆ (shared_mem_pool) {
        ಹಿಂತಿರುಗಿ *shared_mem_pool;
    }

    ಎಸೆ std::bad_alloc();
}

}  // namespace

ತಾರ್ಕಿಕ spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    ಆದರೆ (info == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    *info = {};

    ಆದರೆ (spine_tcm_open_handle(ಖಾಲಿ) != 0 || !spine_tcm_is_ಲಭ್ಯ()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    spine_tcm_mem_info_t mem_info;
    ಆದರೆ (spine_tcm_mem_info(&mem_info) != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    info->ಲಭ್ಯ   = ನಿಜ;
    info->blk_ಗಾತ್ರ    = mem_info.blk_ಗಾತ್ರ;
    info->blk_ಅಂಕೆ     = mem_info.blk_ಅಂಕೆ;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಶೂನ್ಯ * spine_mem_pool_tcm_mem_get(ಪೂರ್ಣಾಂಕ cpu_id) noexcept {
    ಹಿಂತಿರುಗಿ spine_tcm_mem_get(cpu_id);
}

ಶೂನ್ಯ * spine_mem_pool_tcm_mem_wait(ಪೂರ್ಣಾಂಕ cpu_id) noexcept {
    ಹಿಂತಿರುಗಿ spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

ಪೂರ್ಣಾಂಕ spine_mem_pool_tcm_mem_release(ಪೂರ್ಣಾಂಕ cpu_id) noexcept {
    ಹಿಂತಿರುಗಿ spine_tcm_mem_release(cpu_id);
}

ಶೂನ್ಯ * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    ಪ್ರಯತ್ನಿಸು {
        ಹಿಂತಿರುಗಿ get_spine_mem_pool_manager().alloc(size, alignment);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
}

ಶೂನ್ಯ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    ಪ್ರಯತ್ನಿಸು {
        ಹಿಂತಿರುಗಿ get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
}

ಶೂನ್ಯ spine_mem_pool_free(ಶೂನ್ಯ * ತಳ) noexcept {
    ಪ್ರಯತ್ನಿಸು {
        get_spine_mem_pool_manager().ಬಿಡುಗಡೆ(ತಳ);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, ತಳ);
    }
}

ಶೂನ್ಯ spine_mem_pool_shared_mem_free(ಶೂನ್ಯ * ತಳ) noexcept {
    ಪ್ರಯತ್ನಿಸು {
        get_spine_mem_pool_shared_mem_manager().ಬಿಡುಗಡೆ(ತಳ);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, ತಳ);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

ಬಾಹ್ಯ "C" {
ಶೂನ್ಯ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    ಶೂನ್ಯ * ಫಲಿತಾಂಶ = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    ಆದರೆ (ಫಲಿತಾಂಶ == ಶೂನ್ಯಸೂಚಕ) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಶೂನ್ಯ ggml_backend_cpu_riscv64_spacemit_free_shared(ಶೂನ್ಯ * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
