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

constexpr size_t   SPINE_MEM_POOL_CHUNK_سائيز         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_سائيز   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1_G_REGION_سائيز     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr حرف     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr حرف     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

بنية hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#وصف_ڪريو HUGETLB_1G_IOC_MAGIC 'M'
#وصف_ڪريو HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, بنية hugetlb_1g_region)
#وصف_ڪريو HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

بنية free_بلاڪ {
    size_t وٿي{ 0 };
    size_t size{ 0 };
};

بنية pool_chunk {
    uint8_t *               بنياد{ خالي_اشارو };
    size_t                  size{ 0 };
    صحيح                     fd{ -1 };
    std::vector<free_بلاڪ> free_بلاڪون;
};

بنية pool_ورهاست {
    فراغ * chunk_بنياد{ خالي_اشارو };
    size_t chunk_سائيز{ 0 };
    فراغ * بنياد{ خالي_اشارو };
    size_t size{ 0 };
};

منطقي is_power_جو_two(size_t value) {
    أرجع value != 0 && (value & (value - 1)) == 0;
}

منطقي align_مٿي(size_t value, size_t alignment, size_t * aligned_قدر) {
    إذا (aligned_قدر == خالي_اشارو || alignment == 0) {
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

منطقي align_مٿي_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_قدر) {
    إذا (aligned_قدر == خالي_اشارو || alignment == 0) {
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
    واضح spine_mem_pool_manager(size_t default_chunk_سائيز) : default_chunk_سائيز_2(default_chunk_سائيز) {}

    مجازي ~spine_mem_pool_manager() = افتراضي;

    فراغ * alloc(size_t size, size_t alignment) {
        إذا (size == 0 || !is_power_جو_two(alignment)) {
            أرجع خالي_اشارو;
        }

        size_t aligned_سائيز = 0;
        إذا (!align_مٿي(size, alignment, &aligned_سائيز)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            أرجع خالي_اشارو;
        }

        pool_ورهاست ورهاست;

        std::lock_guard<std::mutex> تالو(mutex_);

        إذا (!try_alloc_locked(aligned_سائيز, alignment, &ورهاست)) {
            إذا (!جوڙيو_chunk_locked(aligned_سائيز, alignment)) {
                أرجع خالي_اشارو;
            }

            إذا (!try_alloc_locked(aligned_سائيز, alignment, &ورهاست)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_سائيز, alignment);
                أرجع خالي_اشارو;
            }
        }

        حاول {
            ثابت خودڪار [ورهاست_it, inserted] = allocations_.emplace(ورهاست.بنياد, ورهاست);
            إذا (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, ورهاست.بنياد);
                rollback_ورهاست_locked(ورهاست);
                أرجع خالي_اشارو;
            }
        } امسك (ثابت std::bad_alloc &) {
            rollback_ورهاست_locked(ورهاست);
            ارم;
        }

        أرجع ورهاست.بنياد;
    }

    فراغ حرر(فراغ * بنياد) {
        إذا (بنياد == خالي_اشارو) {
            أرجع;
        }

        std::lock_guard<std::mutex> تالو(mutex_);

        خودڪار ورهاست_it = allocations_.find(بنياد);
        إذا (ورهاست_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, بنياد);
            أرجع;
        }

        pool_ورهاست ورهاست = ورهاست_it->second;
        allocations_.erase(ورهاست_it);

        خودڪار chunk_it = ڳوليو_chunk_locked(ورهاست);
        إذا (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           ورهاست.بنياد, ورهاست.size);
            أرجع;
        }

        خودڪار * chunk_بنياد = chunk_it->بنياد;
        خودڪار * alloc_بنياد = static_cast<uint8_t *>(ورهاست.بنياد);
        إذا (alloc_بنياد < chunk_بنياد || alloc_بنياد >= chunk_بنياد + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           ورهاست.بنياد, chunk_بنياد, chunk_بنياد + chunk_it->size);
            أرجع;
        }

        ثابت size_t وٿي = static_cast<size_t>(alloc_بنياد - chunk_بنياد);
        إذا (وٿي > chunk_it->size || ورهاست.size > chunk_it->size - وٿي) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           ورهاست.بنياد, ورهاست.size, chunk_it->size);
            أرجع;
        }

        وجهو_free_بلاڪ_locked(*chunk_it, { وٿي, ورهاست.size });
        maybe_release_خالي_جايون_chunk_locked(chunk_it);
    }

  محمي:
    فراغ release_chunks() {
        std::lock_guard<std::mutex> تالو(mutex_);

        allocations_.clear();
        لكل (خودڪار & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_سائيز() ثابت { أرجع default_chunk_سائيز_2; }

    ساكن فراغ clear_chunk(pool_chunk * chunk) {
        chunk->بنياد = خالي_اشارو;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_بلاڪون.clear();
    }

    مجازي منطقي alloc_chunk(size_t min_سائيز, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) = 0;
    مجازي فراغ dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  خاص:
    بنية alloc_candidate {
        size_t    chunk_اشاريو{ 0 };
        size_t    بلاڪ_اشاريو{ 0 };
        size_t    aligned_وٿي{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        منطقي      valid{ خطأ };
    };

    std::vector<pool_chunk>::iterator ڳوليو_chunk_locked(ثابت pool_ورهاست & ورهاست) {
        أرجع std::find_if(chunks_.begin(), chunks_.end(), [&](ثابت pool_chunk & chunk) {
            أرجع chunk.بنياد == ورهاست.chunk_بنياد && chunk.size == ورهاست.chunk_سائيز;
        });
    }

    منطقي جوڙيو_chunk_locked(size_t min_سائيز, size_t alignment) {
        pool_chunk   chunk;
        ثابت size_t chunk_request = default_chunk_سائيز_2 == 0 ? min_سائيز : std::max(min_سائيز, default_chunk_سائيز_2);
        فراغ *       hint_addr     = خالي_اشارو;

        لكل (ثابت خودڪار & existing_chunk : chunks_) {
            خودڪار * chunk_پڇاڙي = existing_chunk.بنياد + existing_chunk.size;
            إذا (hint_addr == خالي_اشارو || chunk_پڇاڙي > hint_addr) {
                hint_addr = chunk_پڇاڙي;
            }
        }

        إذا (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            أرجع خطأ;
        }

        إذا (chunk.بنياد == خالي_اشارو || chunk.size < min_سائيز) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_سائيز, chunk.بنياد, chunk.size);
            dealloc_chunk(&chunk);
            أرجع خطأ;
        }

        حاول {
            chunk.free_بلاڪون.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } امسك (ثابت std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ارم;
        }

        أرجع سچ;
    }

    فراغ rollback_ورهاست_locked(ثابت pool_ورهاست & ورهاست) {
        خودڪار chunk_it = ڳوليو_chunk_locked(ورهاست);
        إذا (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, ورهاست.بنياد);
            أرجع;
        }

        خودڪار * chunk_بنياد = chunk_it->بنياد;
        خودڪار * alloc_بنياد = static_cast<uint8_t *>(ورهاست.بنياد);
        إذا (alloc_بنياد < chunk_بنياد || alloc_بنياد >= chunk_بنياد + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, ورهاست.بنياد);
            أرجع;
        }

        ثابت size_t وٿي = static_cast<size_t>(alloc_بنياد - chunk_بنياد);
        إذا (وٿي > chunk_it->size || ورهاست.size > chunk_it->size - وٿي) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           ورهاست.بنياد, ورهاست.size);
            أرجع;
        }

        وجهو_free_بلاڪ_locked(*chunk_it, { وٿي, ورهاست.size });
        maybe_release_خالي_جايون_chunk_locked(chunk_it);
    }

    منطقي try_alloc_locked(size_t size, size_t alignment, pool_ورهاست * ورهاست) {
        alloc_candidate بهترين;

        لكل (size_t chunk_اشاريو = 0; chunk_اشاريو < chunks_.size(); ++chunk_اشاريو) {
            ثابت خودڪار & chunk = chunks_[chunk_اشاريو];
            لكل (size_t بلاڪ_اشاريو = 0; بلاڪ_اشاريو < chunk.free_بلاڪون.size(); ++بلاڪ_اشاريو) {
                ثابت خودڪار & بلاڪ = chunk.free_بلاڪون[بلاڪ_اشاريو];

                uintptr_t  aligned_addr = 0;
                ثابت خودڪار بلاڪ_addr   = reinterpret_cast<uintptr_t>(chunk.بنياد + بلاڪ.وٿي);
                إذا (!align_مٿي_uintptr(بلاڪ_addr, alignment, &aligned_addr)) {
                    استمر;
                }

                إذا (aligned_addr < بلاڪ_addr) {
                    استمر;
                }

                ثابت size_t aligned_وٿي = بلاڪ.وٿي + static_cast<size_t>(aligned_addr - بلاڪ_addr);
                ثابت size_t padding        = aligned_وٿي - بلاڪ.وٿي;
                إذا (padding > بلاڪ.size || size > بلاڪ.size - padding) {
                    استمر;
                }

                إذا (!بهترين.valid || aligned_addr < بهترين.address) {
                    بهترين.chunk_اشاريو    = chunk_اشاريو;
                    بهترين.بلاڪ_اشاريو    = بلاڪ_اشاريو;
                    بهترين.aligned_وٿي = aligned_وٿي;
                    بهترين.address        = aligned_addr;
                    بهترين.valid          = سچ;
                }
            }
        }

        إذا (!بهترين.valid) {
            أرجع خطأ;
        }

        خودڪار &           chunk     = chunks_[بهترين.chunk_اشاريو];
        ثابت free_بلاڪ بلاڪ     = chunk.free_بلاڪون[بهترين.بلاڪ_اشاريو];
        ثابت size_t     padding   = بهترين.aligned_وٿي - بلاڪ.وٿي;
        ثابت size_t     alloc_پڇاڙي = بهترين.aligned_وٿي + size;
        ثابت size_t     بلاڪ_پڇاڙي = بلاڪ.وٿي + بلاڪ.size;

        chunk.free_بلاڪون.erase(chunk.free_بلاڪون.begin() + بهترين.بلاڪ_اشاريو);
        خودڪار وجهو_it = chunk.free_بلاڪون.begin() + بهترين.بلاڪ_اشاريو;
        إذا (padding != 0) {
            وجهو_it = chunk.free_بلاڪون.insert(وجهو_it, { بلاڪ.وٿي, padding });
            ++وجهو_it;
        }
        إذا (alloc_پڇاڙي < بلاڪ_پڇاڙي) {
            chunk.free_بلاڪون.insert(وجهو_it, { alloc_پڇاڙي, بلاڪ_پڇاڙي - alloc_پڇاڙي });
        }

        ورهاست->chunk_بنياد = chunk.بنياد;
        ورهاست->chunk_سائيز = chunk.size;
        ورهاست->بنياد       = chunk.بنياد + بهترين.aligned_وٿي;
        ورهاست->size       = size;
        أرجع سچ;
    }

    فراغ maybe_release_خالي_جايون_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        إذا (chunk_it->free_بلاڪون.size() != 1) {
            أرجع;
        }

        ثابت خودڪار & بلاڪ = chunk_it->free_بلاڪون.front();
        إذا (بلاڪ.وٿي != 0 || بلاڪ.size != chunk_it->size) {
            أرجع;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    فراغ وجهو_free_بلاڪ_locked(pool_chunk & chunk, free_بلاڪ بلاڪ) {
        خودڪار it = chunk.free_بلاڪون.begin();
        طالما (it != chunk.free_بلاڪون.end() && it->وٿي < بلاڪ.وٿي) {
            ++it;
        }

        إذا (it != chunk.free_بلاڪون.begin()) {
            ثابت خودڪار & prev = *(it - 1);
            إذا (prev.وٿي + prev.size > بلاڪ.وٿي) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               بلاڪ.وٿي, بلاڪ.size);
                أرجع;
            }
        }

        إذا (it != chunk.free_بلاڪون.end() && بلاڪ.وٿي + بلاڪ.size > it->وٿي) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           بلاڪ.وٿي, بلاڪ.size);
            أرجع;
        }

        it = chunk.free_بلاڪون.insert(it, بلاڪ);

        إذا (it != chunk.free_بلاڪون.begin()) {
            خودڪار prev = it - 1;
            إذا (prev->وٿي + prev->size == it->وٿي) {
                it->وٿي = prev->وٿي;
                it->size += prev->size;
                it = chunk.free_بلاڪون.erase(prev);
            }
        }

        إذا (it + 1 != chunk.free_بلاڪون.end() && it->وٿي + it->size == (it + 1)->وٿي) {
            it->size += (it + 1)->size;
            chunk.free_بلاڪون.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<فراغ *, pool_ورهاست> allocations_;
    size_t                                      default_chunk_سائيز_2{ 0 };
};

صنف spine_mem_pool_posix final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_سائيز, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) hint_addr;

        ثابت size_t alloc_alignment = std::max(alignment, حجم(فراغ *));
        فراغ *       بنياد            = خالي_اشارو;
        ثابت صحيح    rc              = posix_memalign(&بنياد, alloc_alignment, min_سائيز);
        إذا (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_سائيز, alloc_alignment, rc);
            أرجع خطأ;
        }

        chunk->بنياد = static_cast<uint8_t *>(بنياد);
        chunk->size = min_سائيز;
        chunk->fd   = -1;
        أرجع سچ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        std::حرر(chunk->بنياد);
        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_transparent_hugepage final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_سائيز) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_سائيز, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;

        size_t chunk_سائيز = 0;
        إذا (!align_مٿي(min_سائيز, default_chunk_سائيز(), &chunk_سائيز)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_سائيز);
            أرجع خطأ;
        }

        فراغ * map_addr = mmap(hint_addr, chunk_سائيز, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_سائيز,
                           errno);
            أرجع خطأ;
        }

        إذا (madvise(map_addr, chunk_سائيز, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_سائيز, errno);
            munmap(map_addr, chunk_سائيز);
            أرجع خطأ;
        }

        chunk->بنياد = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_سائيز;
        chunk->fd   = -1;
        أرجع سچ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->بنياد != خالي_اشارو && chunk->size != 0 && munmap(chunk->بنياد, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->بنياد, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_hugetlb_1g final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1_G_REGION_سائيز) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_سائيز, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;
        (فراغ) hint_addr;

        size_t region_سائيز = 0;
        إذا (!align_مٿي(min_سائيز, SPINE_MEM_POOL_1_G_REGION_سائيز, &region_سائيز)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_سائيز);
            أرجع خطأ;
        }

        ثابت صحيح fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        إذا (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            أرجع خطأ;
        }

        hugetlb_1g_region region;
        region.size  = region_سائيز;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        إذا (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_سائيز, errno);
            close(fd);
            أرجع خطأ;
        }

        فراغ * map_addr = mmap(خالي_اشارو, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<بنانشان ڊگهو ڊگهو>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            أرجع خطأ;
        }

        chunk->بنياد = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        أرجع سچ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->بنياد != خالي_اشارو && chunk->size != 0 && munmap(chunk->بنياد, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->بنياد, chunk->size, errno);
        }

        إذا (chunk->fd >= 0) {
            إذا (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->بنياد, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_shared_mem final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_سائيز) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_سائيز, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;

        إذا (hint_addr != خالي_اشارو) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            أرجع خطأ;
        }

        إذا (min_سائيز > default_chunk_سائيز()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_سائيز, default_chunk_سائيز());
            أرجع خطأ;
        }

        ثابت صحيح fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        إذا (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            أرجع خطأ;
        }

        فراغ * map_addr = mmap(خالي_اشارو, default_chunk_سائيز(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_سائيز(), errno);
            close(fd);
            أرجع خطأ;
        }

        chunk->بنياد = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_سائيز();
        chunk->fd   = fd;
        أرجع سچ;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->بنياد != خالي_اشارو && chunk->size != 0 && munmap(chunk->بنياد, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->بنياد, chunk->size, errno);
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
    إذا (info == خالي_اشارو) {
        أرجع خطأ;
    }

    *info = {};

    إذا (spine_tcm_open_handle(عدم) != 0 || !spine_tcm_is_موجود()) {
        أرجع خطأ;
    }

    spine_tcm_mem_info_t mem_info;
    إذا (spine_tcm_mem_info(&mem_info) != 0) {
        أرجع خطأ;
    }

    info->موجود   = سچ;
    info->blk_سائيز    = mem_info.blk_سائيز;
    info->blk_انگ     = mem_info.blk_انگ;
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
        أرجع خالي_اشارو;
    }
}

فراغ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    حاول {
        أرجع get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        أرجع خالي_اشارو;
    }
}

فراغ spine_mem_pool_free(فراغ * بنياد) noexcept {
    حاول {
        get_spine_mem_pool_manager().حرر(بنياد);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, بنياد);
    }
}

فراغ spine_mem_pool_shared_mem_free(فراغ * بنياد) noexcept {
    حاول {
        get_spine_mem_pool_shared_mem_manager().حرر(بنياد);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, بنياد);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

خارجي "C" {
فراغ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    فراغ * نتيجو = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    إذا (نتيجو == خالي_اشارو) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    أرجع نتيجو;
}

فراغ ggml_backend_cpu_riscv64_spacemit_free_shared(فراغ * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
