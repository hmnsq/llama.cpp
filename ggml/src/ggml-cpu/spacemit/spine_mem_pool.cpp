#ضمن "spine_mem_pool.h"

#ضمن "common.h"
#ضمن "ime_env.h"
#ضمن "spine_tcm.h"

#ضمن <fcntl.h>
#ضمن <sys/ioctl.h>
#ضمن <sys/mman.h>
#ضمن <unistd.h>

#ضمن <algorithm>
#ضمن <cerrno>
#ضمن <cstdint>
#ضمن <cstdlib>
#ضمن <limits>
#ضمن <memory>
#ضمن <mutex>
#ضمن <unordered_map>
#ضمن <vector>

نطاق ggml::cpu::riscv64_spacemit {
نطاق {

constexpr size_t   SPINE_MEM_POOL_CHUNK_ناپ         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_ناپ   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1_G_REGION_ناپ     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr حرف     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr حرف     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

بنية hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#تعریف_کرو HUGETLB_1G_IOC_MAGIC 'M'
#تعریف_کرو HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, بنية hugetlb_1g_region)
#تعریف_کرو HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

بنية free_بلاک {
    size_t وتھ{ 0 };
    size_t size{ 0 };
};

بنية pool_chunk {
    uint8_t *               ادھار{ خالی_سوچک };
    size_t                  size{ 0 };
    صحيح                     fd{ -1 };
    std::vector<free_بلاک> free_بلاکاں;
};

بنية pool_ونڈ {
    فراغ * chunk_ادھار{ خالی_سوچک };
    size_t chunk_ناپ{ 0 };
    فراغ * ادھار{ خالی_سوچک };
    size_t size{ 0 };
};

منطقي is_power_دا_two(size_t value) {
    أرجع value != 0 && (value & (value - 1)) == 0;
}

منطقي align_اتے(size_t value, size_t alignment, size_t * aligned_قدر) {
    إذا (aligned_قدر == خالی_سوچک || alignment == 0) {
        أرجع خطأ;
    }

    ثابت size_t remainder = value % alignment;
    إذا (remainder == 0) {
        *aligned_قدر = value;
        أرجع سچ;
    }

    ثابت size_t padding = alignment - remainder;
    إذا (value > std::numeric_limits<size_t>::max() - padding) {
        أرجع خطأ;
    }

    *aligned_قدر = value + padding;
    أرجع سچ;
}

منطقي align_اتے_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_قدر) {
    إذا (aligned_قدر == خالی_سوچک || alignment == 0) {
        أرجع خطأ;
    }

    ثابت uintptr_t remainder = value % alignment;
    إذا (remainder == 0) {
        *aligned_قدر = value;
        أرجع سچ;
    }

    ثابت uintptr_t padding = alignment - remainder;
    إذا (value > std::numeric_limits<uintptr_t>::max() - padding) {
        أرجع خطأ;
    }

    *aligned_قدر = value + padding;
    أرجع سچ;
}

صنف spine_mem_pool_manager {
  عام:
    واضح spine_mem_pool_manager(size_t default_chunk_ناپ) : default_chunk_ناپ_2(default_chunk_ناپ) {}

    مجازی ~spine_mem_pool_manager() = افتراضي;

    فراغ * alloc(size_t size, size_t alignment) {
        إذا (size == 0 || !is_power_دا_two(alignment)) {
            أرجع خالی_سوچک;
        }

        size_t aligned_ناپ = 0;
        إذا (!align_اتے(size, alignment, &aligned_ناپ)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            أرجع خالی_سوچک;
        }

        pool_ونڈ ونڈ;

        std::lock_guard<std::mutex> جندرا(mutex_);

        إذا (!try_alloc_locked(aligned_ناپ, alignment, &ونڈ)) {
            إذا (!جوڑو_chunk_locked(aligned_ناپ, alignment)) {
                أرجع خالی_سوچک;
            }

            إذا (!try_alloc_locked(aligned_ناپ, alignment, &ونڈ)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_ناپ, alignment);
                أرجع خالی_سوچک;
            }
        }

        حاول {
            ثابت خودکار [ونڈ_it, inserted] = allocations_.emplace(ونڈ.ادھار, ونڈ);
            إذا (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, ونڈ.ادھار);
                rollback_ونڈ_locked(ونڈ);
                أرجع خالی_سوچک;
            }
        } امسك (ثابت std::bad_alloc &) {
            rollback_ونڈ_locked(ونڈ);
            ارم;
        }

        أرجع ونڈ.ادھار;
    }

    فراغ حرر(فراغ * ادھار) {
        إذا (ادھار == خالی_سوچک) {
            أرجع;
        }

        std::lock_guard<std::mutex> جندرا(mutex_);

        خودکار ونڈ_it = allocations_.find(ادھار);
        إذا (ونڈ_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, ادھار);
            أرجع;
        }

        pool_ونڈ ونڈ = ونڈ_it->second;
        allocations_.erase(ونڈ_it);

        خودکار chunk_it = لبھو_chunk_locked(ونڈ);
        إذا (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           ونڈ.ادھار, ونڈ.size);
            أرجع;
        }

        خودکار * chunk_ادھار = chunk_it->ادھار;
        خودکار * alloc_ادھار = static_cast<uint8_t *>(ونڈ.ادھار);
        إذا (alloc_ادھار < chunk_ادھار || alloc_ادھار >= chunk_ادھار + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           ونڈ.ادھار, chunk_ادھار, chunk_ادھار + chunk_it->size);
            أرجع;
        }

        ثابت size_t وتھ = static_cast<size_t>(alloc_ادھار - chunk_ادھار);
        إذا (وتھ > chunk_it->size || ونڈ.size > chunk_it->size - وتھ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           ونڈ.ادھار, ونڈ.size, chunk_it->size);
            أرجع;
        }

        داخل_کرو_free_بلاک_locked(*chunk_it, { وتھ, ونڈ.size });
        maybe_release_خالی_تھاواں_chunk_locked(chunk_it);
    }

  محمي:
    فراغ release_chunks() {
        std::lock_guard<std::mutex> جندرا(mutex_);

        allocations_.clear();
        لكل (خودکار & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_ناپ() ثابت { أرجع default_chunk_ناپ_2; }

    ساكن فراغ clear_chunk(pool_chunk * chunk) {
        chunk->ادھار = خالی_سوچک;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_بلاکاں.clear();
    }

    مجازی منطقي alloc_chunk(size_t min_ناپ, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) = 0;
    مجازی فراغ dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  خاص:
    بنية alloc_candidate {
        size_t    chunk_اشاریہ{ 0 };
        size_t    بلاک_اشاریہ{ 0 };
        size_t    aligned_وتھ{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        منطقي      valid{ خطأ };
    };

    std::vector<pool_chunk>::iterator لبھو_chunk_locked(ثابت pool_ونڈ & ونڈ) {
        أرجع std::find_if(chunks_.begin(), chunks_.end(), [&](ثابت pool_chunk & chunk) {
            أرجع chunk.ادھار == ونڈ.chunk_ادھار && chunk.size == ونڈ.chunk_ناپ;
        });
    }

    منطقي جوڑو_chunk_locked(size_t min_ناپ, size_t alignment) {
        pool_chunk   chunk;
        ثابت size_t chunk_request = default_chunk_ناپ_2 == 0 ? min_ناپ : std::max(min_ناپ, default_chunk_ناپ_2);
        فراغ *       hint_addr     = خالی_سوچک;

        لكل (ثابت خودکار & existing_chunk : chunks_) {
            خودکار * chunk_سرا = existing_chunk.ادھار + existing_chunk.size;
            إذا (hint_addr == خالی_سوچک || chunk_سرا > hint_addr) {
                hint_addr = chunk_سرا;
            }
        }

        إذا (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            أرجع خطأ;
        }

        إذا (chunk.ادھار == خالی_سوچک || chunk.size < min_ناپ) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_ناپ, chunk.ادھار, chunk.size);
            dealloc_chunk(&chunk);
            أرجع خطأ;
        }

        حاول {
            chunk.free_بلاکاں.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } امسك (ثابت std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ارم;
        }

        أرجع سچ;
    }

    فراغ rollback_ونڈ_locked(ثابت pool_ونڈ & ونڈ) {
        خودکار chunk_it = لبھو_chunk_locked(ونڈ);
        إذا (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, ونڈ.ادھار);
            أرجع;
        }

        خودکار * chunk_ادھار = chunk_it->ادھار;
        خودکار * alloc_ادھار = static_cast<uint8_t *>(ونڈ.ادھار);
        إذا (alloc_ادھار < chunk_ادھار || alloc_ادھار >= chunk_ادھار + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, ونڈ.ادھار);
            أرجع;
        }

        ثابت size_t وتھ = static_cast<size_t>(alloc_ادھار - chunk_ادھار);
        إذا (وتھ > chunk_it->size || ونڈ.size > chunk_it->size - وتھ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           ونڈ.ادھار, ونڈ.size);
            أرجع;
        }

        داخل_کرو_free_بلاک_locked(*chunk_it, { وتھ, ونڈ.size });
        maybe_release_خالی_تھاواں_chunk_locked(chunk_it);
    }

    منطقي try_alloc_locked(size_t size, size_t alignment, pool_ونڈ * ونڈ) {
        alloc_candidate بہترین;

        لكل (size_t chunk_اشاریہ = 0; chunk_اشاریہ < chunks_.size(); ++chunk_اشاریہ) {
            ثابت خودکار & chunk = chunks_[chunk_اشاریہ];
            لكل (size_t بلاک_اشاریہ = 0; بلاک_اشاریہ < chunk.free_بلاکاں.size(); ++بلاک_اشاریہ) {
                ثابت خودکار & بلاک = chunk.free_بلاکاں[بلاک_اشاریہ];

                uintptr_t  aligned_addr = 0;
                ثابت خودکار بلاک_addr   = reinterpret_cast<uintptr_t>(chunk.ادھار + بلاک.وتھ);
                إذا (!align_اتے_uintptr(بلاک_addr, alignment, &aligned_addr)) {
                    استمر;
                }

                إذا (aligned_addr < بلاک_addr) {
                    استمر;
                }

                ثابت size_t aligned_وتھ = بلاک.وتھ + static_cast<size_t>(aligned_addr - بلاک_addr);
                ثابت size_t padding        = aligned_وتھ - بلاک.وتھ;
                إذا (padding > بلاک.size || size > بلاک.size - padding) {
                    استمر;
                }

                إذا (!بہترین.valid || aligned_addr < بہترین.address) {
                    بہترین.chunk_اشاریہ    = chunk_اشاریہ;
                    بہترین.بلاک_اشاریہ    = بلاک_اشاریہ;
                    بہترین.aligned_وتھ = aligned_وتھ;
                    بہترین.address        = aligned_addr;
                    بہترین.valid          = سچ;
                }
            }
        }

        إذا (!بہترین.valid) {
            أرجع خطأ;
        }

        خودکار &           chunk     = chunks_[بہترین.chunk_اشاریہ];
        ثابت free_بلاک بلاک     = chunk.free_بلاکاں[بہترین.بلاک_اشاریہ];
        ثابت size_t     padding   = بہترین.aligned_وتھ - بلاک.وتھ;
        ثابت size_t     alloc_سرا = بہترین.aligned_وتھ + size;
        ثابت size_t     بلاک_سرا = بلاک.وتھ + بلاک.size;

        chunk.free_بلاکاں.erase(chunk.free_بلاکاں.begin() + بہترین.بلاک_اشاریہ);
        خودکار داخل_کرو_it = chunk.free_بلاکاں.begin() + بہترین.بلاک_اشاریہ;
        إذا (padding != 0) {
            داخل_کرو_it = chunk.free_بلاکاں.insert(داخل_کرو_it, { بلاک.وتھ, padding });
            ++داخل_کرو_it;
        }
        إذا (alloc_سرا < بلاک_سرا) {
            chunk.free_بلاکاں.insert(داخل_کرو_it, { alloc_سرا, بلاک_سرا - alloc_سرا });
        }

        ونڈ->chunk_ادھار = chunk.ادھار;
        ونڈ->chunk_ناپ = chunk.size;
        ونڈ->ادھار       = chunk.ادھار + بہترین.aligned_وتھ;
        ونڈ->size       = size;
        أرجع سچ;
    }

    فراغ maybe_release_خالی_تھاواں_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        إذا (chunk_it->free_بلاکاں.size() != 1) {
            أرجع;
        }

        ثابت خودکار & بلاک = chunk_it->free_بلاکاں.front();
        إذا (بلاک.وتھ != 0 || بلاک.size != chunk_it->size) {
            أرجع;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    فراغ داخل_کرو_free_بلاک_locked(pool_chunk & chunk, free_بلاک بلاک) {
        خودکار it = chunk.free_بلاکاں.begin();
        طالما (it != chunk.free_بلاکاں.end() && it->وتھ < بلاک.وتھ) {
            ++it;
        }

        إذا (it != chunk.free_بلاکاں.begin()) {
            ثابت خودکار & prev = *(it - 1);
            إذا (prev.وتھ + prev.size > بلاک.وتھ) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               بلاک.وتھ, بلاک.size);
                أرجع;
            }
        }

        إذا (it != chunk.free_بلاکاں.end() && بلاک.وتھ + بلاک.size > it->وتھ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           بلاک.وتھ, بلاک.size);
            أرجع;
        }

        it = chunk.free_بلاکاں.insert(it, بلاک);

        إذا (it != chunk.free_بلاکاں.begin()) {
            خودکار prev = it - 1;
            إذا (prev->وتھ + prev->size == it->وتھ) {
                it->وتھ = prev->وتھ;
                it->size += prev->size;
                it = chunk.free_بلاکاں.erase(prev);
            }
        }

        إذا (it + 1 != chunk.free_بلاکاں.end() && it->وتھ + it->size == (it + 1)->وتھ) {
            it->size += (it + 1)->size;
            chunk.free_بلاکاں.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<فراغ *, pool_ونڈ> allocations_;
    size_t                                      default_chunk_ناپ_2{ 0 };
};

صنف spine_mem_pool_posix final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_ناپ, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) hint_addr;

        ثابت size_t alloc_alignment = std::max(alignment, حجم(فراغ *));
        فراغ *       ادھار            = خالی_سوچک;
        ثابت صحيح    rc              = posix_memalign(&ادھار, alloc_alignment, min_ناپ);
        إذا (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_ناپ, alloc_alignment, rc);
            أرجع خطأ;
        }

        chunk->ادھار = static_cast<uint8_t *>(ادھار);
        chunk->size = min_ناپ;
        chunk->fd   = -1;
        أرجع سچ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        std::حرر(chunk->ادھار);
        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_transparent_hugepage final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_ناپ) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_ناپ, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;

        size_t chunk_ناپ = 0;
        إذا (!align_اتے(min_ناپ, default_chunk_ناپ(), &chunk_ناپ)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_ناپ);
            أرجع خطأ;
        }

        فراغ * map_addr = mmap(hint_addr, chunk_ناپ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_ناپ,
                           errno);
            أرجع خطأ;
        }

        إذا (madvise(map_addr, chunk_ناپ, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_ناپ, errno);
            munmap(map_addr, chunk_ناپ);
            أرجع خطأ;
        }

        chunk->ادھار = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_ناپ;
        chunk->fd   = -1;
        أرجع سچ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->ادھار != خالی_سوچک && chunk->size != 0 && munmap(chunk->ادھار, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->ادھار, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_hugetlb_1g final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1_G_REGION_ناپ) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_ناپ, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;
        (فراغ) hint_addr;

        size_t region_ناپ = 0;
        إذا (!align_اتے(min_ناپ, SPINE_MEM_POOL_1_G_REGION_ناپ, &region_ناپ)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_ناپ);
            أرجع خطأ;
        }

        ثابت صحيح fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        إذا (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            أرجع خطأ;
        }

        hugetlb_1g_region region;
        region.size  = region_ناپ;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        إذا (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_ناپ, errno);
            close(fd);
            أرجع خطأ;
        }

        فراغ * map_addr = mmap(خالی_سوچک, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<بنانشانی لما لما>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            أرجع خطأ;
        }

        chunk->ادھار = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        أرجع سچ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->ادھار != خالی_سوچک && chunk->size != 0 && munmap(chunk->ادھار, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->ادھار, chunk->size, errno);
        }

        إذا (chunk->fd >= 0) {
            إذا (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->ادھار, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_shared_mem final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_ناپ) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_ناپ, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;

        إذا (hint_addr != خالی_سوچک) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            أرجع خطأ;
        }

        إذا (min_ناپ > default_chunk_ناپ()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_ناپ, default_chunk_ناپ());
            أرجع خطأ;
        }

        ثابت صحيح fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        إذا (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            أرجع خطأ;
        }

        فراغ * map_addr = mmap(خالی_سوچک, default_chunk_ناپ(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_ناپ(), errno);
            close(fd);
            أرجع خطأ;
        }

        chunk->ادھار = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_ناپ();
        chunk->fd   = fd;
        أرجع سچ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->ادھار != خالی_سوچک && chunk->size != 0 && munmap(chunk->ادھار, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->ادھار, chunk->size, errno);
        }

        إذا (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    ساكن std::once_flag                          pool_once;
    ساكن std::unique_ptr<spine_mem_pool_manager> selected_pool;
    ساكن spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    إذا (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        بدل (selected_backend) {
            حالة spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                اكسر;
            حالة spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                اكسر;
            حالة spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                اكسر;
            حالة spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                اكسر;
        }
    });

    إذا (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<صحيح>(backend), static_cast<صحيح>(selected_backend));
    }

    إذا (selected_pool) {
        أرجع *selected_pool;
    }

    ارم std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    ساكن std::once_flag                             shared_mem_pool_once;
    ساكن std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    إذا (shared_mem_pool) {
        أرجع *shared_mem_pool;
    }

    ارم std::bad_alloc();
}

}  // namespace

منطقي spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    إذا (info == خالی_سوچک) {
        أرجع خطأ;
    }

    *info = {};

    إذا (spine_tcm_open_handle(عدم) != 0 || !spine_tcm_is_دستیاب()) {
        أرجع خطأ;
    }

    spine_tcm_mem_info_t mem_info;
    إذا (spine_tcm_mem_info(&mem_info) != 0) {
        أرجع خطأ;
    }

    info->دستیاب   = سچ;
    info->blk_ناپ    = mem_info.blk_ناپ;
    info->blk_عدد     = mem_info.blk_عدد;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    أرجع سچ;
}

فراغ * spine_mem_pool_tcm_mem_get(صحيح cpu_id) noexcept {
    أرجع spine_tcm_mem_get(cpu_id);
}

فراغ * spine_mem_pool_tcm_mem_wait(صحيح cpu_id) noexcept {
    أرجع spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

صحيح spine_mem_pool_tcm_mem_release(صحيح cpu_id) noexcept {
    أرجع spine_tcm_mem_release(cpu_id);
}

فراغ * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    حاول {
        أرجع get_spine_mem_pool_manager().alloc(size, alignment);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        أرجع خالی_سوچک;
    }
}

فراغ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    حاول {
        أرجع get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        أرجع خالی_سوچک;
    }
}

فراغ spine_mem_pool_free(فراغ * ادھار) noexcept {
    حاول {
        get_spine_mem_pool_manager().حرر(ادھار);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, ادھار);
    }
}

فراغ spine_mem_pool_shared_mem_free(فراغ * ادھار) noexcept {
    حاول {
        get_spine_mem_pool_shared_mem_manager().حرر(ادھار);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, ادھار);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

خارجي "C" {
فراغ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    فراغ * نتیجہ = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    إذا (نتیجہ == خالی_سوچک) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    أرجع نتیجہ;
}

فراغ ggml_backend_cpu_riscv64_spacemit_free_shared(فراغ * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
