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

constexpr size_t   SPINE_MEM_POOL_CHUNK_اندازه         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_اندازه   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1_G_REGION_اندازه     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr حرف     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr حرف     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

بنية hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#تعریف_کړه HUGETLB_1G_IOC_MAGIC 'M'
#تعریف_کړه HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, بنية hugetlb_1g_region)
#تعریف_کړه HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

بنية free_بلاک {
    size_t دننه_واټن{ 0 };
    size_t size{ 0 };
};

بنية pool_chunk {
    uint8_t *               بنسټ{ تش_اشاره };
    size_t                  size{ 0 };
    صحيح                     fd{ -1 };
    std::vector<free_بلاک> free_بلاکونه;
};

بنية pool_ویش {
    فراغ * chunk_بنسټ{ تش_اشاره };
    size_t chunk_اندازه{ 0 };
    فراغ * بنسټ{ تش_اشاره };
    size_t size{ 0 };
};

منطقي is_power_د_two(size_t value) {
    أرجع value != 0 && (value & (value - 1)) == 0;
}

منطقي align_پورته(size_t value, size_t alignment, size_t * aligned_ارزښت) {
    إذا (aligned_ارزښت == تش_اشاره || alignment == 0) {
        أرجع خطأ;
    }

    ثابت size_t remainder = value % alignment;
    إذا (remainder == 0) {
        *aligned_ارزښت = value;
        أرجع ریښتیا;
    }

    ثابت size_t padding = alignment - remainder;
    إذا (value > std::numeric_limits<size_t>::max() - padding) {
        أرجع خطأ;
    }

    *aligned_ارزښت = value + padding;
    أرجع ریښتیا;
}

منطقي align_پورته_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_ارزښت) {
    إذا (aligned_ارزښت == تش_اشاره || alignment == 0) {
        أرجع خطأ;
    }

    ثابت uintptr_t remainder = value % alignment;
    إذا (remainder == 0) {
        *aligned_ارزښت = value;
        أرجع ریښتیا;
    }

    ثابت uintptr_t padding = alignment - remainder;
    إذا (value > std::numeric_limits<uintptr_t>::max() - padding) {
        أرجع خطأ;
    }

    *aligned_ارزښت = value + padding;
    أرجع ریښتیا;
}

صنف spine_mem_pool_manager {
  عام:
    څرګند spine_mem_pool_manager(size_t default_chunk_اندازه) : default_chunk_اندازه_2(default_chunk_اندازه) {}

    مجازي ~spine_mem_pool_manager() = افتراضي;

    فراغ * alloc(size_t size, size_t alignment) {
        إذا (size == 0 || !is_power_د_two(alignment)) {
            أرجع تش_اشاره;
        }

        size_t aligned_اندازه = 0;
        إذا (!align_پورته(size, alignment, &aligned_اندازه)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            أرجع تش_اشاره;
        }

        pool_ویش ویش;

        std::lock_guard<std::mutex> قلف(mutex_);

        إذا (!try_alloc_locked(aligned_اندازه, alignment, &ویش)) {
            إذا (!جمع_کړه_chunk_locked(aligned_اندازه, alignment)) {
                أرجع تش_اشاره;
            }

            إذا (!try_alloc_locked(aligned_اندازه, alignment, &ویش)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_اندازه, alignment);
                أرجع تش_اشاره;
            }
        }

        حاول {
            ثابت خپلکاری [ویش_it, inserted] = allocations_.emplace(ویش.بنسټ, ویش);
            إذا (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, ویش.بنسټ);
                rollback_ویش_locked(ویش);
                أرجع تش_اشاره;
            }
        } امسك (ثابت std::bad_alloc &) {
            rollback_ویش_locked(ویش);
            ارم;
        }

        أرجع ویش.بنسټ;
    }

    فراغ حرر(فراغ * بنسټ) {
        إذا (بنسټ == تش_اشاره) {
            أرجع;
        }

        std::lock_guard<std::mutex> قلف(mutex_);

        خپلکاری ویش_it = allocations_.find(بنسټ);
        إذا (ویش_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, بنسټ);
            أرجع;
        }

        pool_ویش ویش = ویش_it->second;
        allocations_.erase(ویش_it);

        خپلکاری chunk_it = پیدا_کړه_chunk_locked(ویش);
        إذا (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           ویش.بنسټ, ویش.size);
            أرجع;
        }

        خپلکاری * chunk_بنسټ = chunk_it->بنسټ;
        خپلکاری * alloc_بنسټ = static_cast<uint8_t *>(ویش.بنسټ);
        إذا (alloc_بنسټ < chunk_بنسټ || alloc_بنسټ >= chunk_بنسټ + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           ویش.بنسټ, chunk_بنسټ, chunk_بنسټ + chunk_it->size);
            أرجع;
        }

        ثابت size_t دننه_واټن = static_cast<size_t>(alloc_بنسټ - chunk_بنسټ);
        إذا (دننه_واټن > chunk_it->size || ویش.size > chunk_it->size - دننه_واټن) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           ویش.بنسټ, ویش.size, chunk_it->size);
            أرجع;
        }

        ورننباسه_free_بلاک_locked(*chunk_it, { دننه_واټن, ویش.size });
        maybe_release_تشه_chunk_locked(chunk_it);
    }

  محمي:
    فراغ release_chunks() {
        std::lock_guard<std::mutex> قلف(mutex_);

        allocations_.clear();
        لكل (خپلکاری & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_اندازه() ثابت { أرجع default_chunk_اندازه_2; }

    ساكن فراغ clear_chunk(pool_chunk * chunk) {
        chunk->بنسټ = تش_اشاره;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_بلاکونه.clear();
    }

    مجازي منطقي alloc_chunk(size_t min_اندازه, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) = 0;
    مجازي فراغ dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  خاص:
    بنية alloc_candidate {
        size_t    chunk_شاخص{ 0 };
        size_t    بلاک_شاخص{ 0 };
        size_t    aligned_دننه_واټن{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        منطقي      valid{ خطأ };
    };

    std::vector<pool_chunk>::iterator پیدا_کړه_chunk_locked(ثابت pool_ویش & ویش) {
        أرجع std::find_if(chunks_.begin(), chunks_.end(), [&](ثابت pool_chunk & chunk) {
            أرجع chunk.بنسټ == ویش.chunk_بنسټ && chunk.size == ویش.chunk_اندازه;
        });
    }

    منطقي جمع_کړه_chunk_locked(size_t min_اندازه, size_t alignment) {
        pool_chunk   chunk;
        ثابت size_t chunk_request = default_chunk_اندازه_2 == 0 ? min_اندازه : std::max(min_اندازه, default_chunk_اندازه_2);
        فراغ *       hint_addr     = تش_اشاره;

        لكل (ثابت خپلکاری & existing_chunk : chunks_) {
            خپلکاری * chunk_پای_ټکی = existing_chunk.بنسټ + existing_chunk.size;
            إذا (hint_addr == تش_اشاره || chunk_پای_ټکی > hint_addr) {
                hint_addr = chunk_پای_ټکی;
            }
        }

        إذا (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            أرجع خطأ;
        }

        إذا (chunk.بنسټ == تش_اشاره || chunk.size < min_اندازه) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_اندازه, chunk.بنسټ, chunk.size);
            dealloc_chunk(&chunk);
            أرجع خطأ;
        }

        حاول {
            chunk.free_بلاکونه.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } امسك (ثابت std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ارم;
        }

        أرجع ریښتیا;
    }

    فراغ rollback_ویش_locked(ثابت pool_ویش & ویش) {
        خپلکاری chunk_it = پیدا_کړه_chunk_locked(ویش);
        إذا (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, ویش.بنسټ);
            أرجع;
        }

        خپلکاری * chunk_بنسټ = chunk_it->بنسټ;
        خپلکاری * alloc_بنسټ = static_cast<uint8_t *>(ویش.بنسټ);
        إذا (alloc_بنسټ < chunk_بنسټ || alloc_بنسټ >= chunk_بنسټ + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, ویش.بنسټ);
            أرجع;
        }

        ثابت size_t دننه_واټن = static_cast<size_t>(alloc_بنسټ - chunk_بنسټ);
        إذا (دننه_واټن > chunk_it->size || ویش.size > chunk_it->size - دننه_واټن) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           ویش.بنسټ, ویش.size);
            أرجع;
        }

        ورننباسه_free_بلاک_locked(*chunk_it, { دننه_واټن, ویش.size });
        maybe_release_تشه_chunk_locked(chunk_it);
    }

    منطقي try_alloc_locked(size_t size, size_t alignment, pool_ویش * ویش) {
        alloc_candidate غوره;

        لكل (size_t chunk_شاخص = 0; chunk_شاخص < chunks_.size(); ++chunk_شاخص) {
            ثابت خپلکاری & chunk = chunks_[chunk_شاخص];
            لكل (size_t بلاک_شاخص = 0; بلاک_شاخص < chunk.free_بلاکونه.size(); ++بلاک_شاخص) {
                ثابت خپلکاری & بلاک = chunk.free_بلاکونه[بلاک_شاخص];

                uintptr_t  aligned_addr = 0;
                ثابت خپلکاری بلاک_addr   = reinterpret_cast<uintptr_t>(chunk.بنسټ + بلاک.دننه_واټن);
                إذا (!align_پورته_uintptr(بلاک_addr, alignment, &aligned_addr)) {
                    استمر;
                }

                إذا (aligned_addr < بلاک_addr) {
                    استمر;
                }

                ثابت size_t aligned_دننه_واټن = بلاک.دننه_واټن + static_cast<size_t>(aligned_addr - بلاک_addr);
                ثابت size_t padding        = aligned_دننه_واټن - بلاک.دننه_واټن;
                إذا (padding > بلاک.size || size > بلاک.size - padding) {
                    استمر;
                }

                إذا (!غوره.valid || aligned_addr < غوره.address) {
                    غوره.chunk_شاخص    = chunk_شاخص;
                    غوره.بلاک_شاخص    = بلاک_شاخص;
                    غوره.aligned_دننه_واټن = aligned_دننه_واټن;
                    غوره.address        = aligned_addr;
                    غوره.valid          = ریښتیا;
                }
            }
        }

        إذا (!غوره.valid) {
            أرجع خطأ;
        }

        خپلکاری &           chunk     = chunks_[غوره.chunk_شاخص];
        ثابت free_بلاک بلاک     = chunk.free_بلاکونه[غوره.بلاک_شاخص];
        ثابت size_t     padding   = غوره.aligned_دننه_واټن - بلاک.دننه_واټن;
        ثابت size_t     alloc_پای_ټکی = غوره.aligned_دننه_واټن + size;
        ثابت size_t     بلاک_پای_ټکی = بلاک.دننه_واټن + بلاک.size;

        chunk.free_بلاکونه.erase(chunk.free_بلاکونه.begin() + غوره.بلاک_شاخص);
        خپلکاری ورننباسه_it = chunk.free_بلاکونه.begin() + غوره.بلاک_شاخص;
        إذا (padding != 0) {
            ورننباسه_it = chunk.free_بلاکونه.insert(ورننباسه_it, { بلاک.دننه_واټن, padding });
            ++ورننباسه_it;
        }
        إذا (alloc_پای_ټکی < بلاک_پای_ټکی) {
            chunk.free_بلاکونه.insert(ورننباسه_it, { alloc_پای_ټکی, بلاک_پای_ټکی - alloc_پای_ټکی });
        }

        ویش->chunk_بنسټ = chunk.بنسټ;
        ویش->chunk_اندازه = chunk.size;
        ویش->بنسټ       = chunk.بنسټ + غوره.aligned_دننه_واټن;
        ویش->size       = size;
        أرجع ریښتیا;
    }

    فراغ maybe_release_تشه_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        إذا (chunk_it->free_بلاکونه.size() != 1) {
            أرجع;
        }

        ثابت خپلکاری & بلاک = chunk_it->free_بلاکونه.front();
        إذا (بلاک.دننه_واټن != 0 || بلاک.size != chunk_it->size) {
            أرجع;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    فراغ ورننباسه_free_بلاک_locked(pool_chunk & chunk, free_بلاک بلاک) {
        خپلکاری it = chunk.free_بلاکونه.begin();
        طالما (it != chunk.free_بلاکونه.end() && it->دننه_واټن < بلاک.دننه_واټن) {
            ++it;
        }

        إذا (it != chunk.free_بلاکونه.begin()) {
            ثابت خپلکاری & prev = *(it - 1);
            إذا (prev.دننه_واټن + prev.size > بلاک.دننه_واټن) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               بلاک.دننه_واټن, بلاک.size);
                أرجع;
            }
        }

        إذا (it != chunk.free_بلاکونه.end() && بلاک.دننه_واټن + بلاک.size > it->دننه_واټن) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           بلاک.دننه_واټن, بلاک.size);
            أرجع;
        }

        it = chunk.free_بلاکونه.insert(it, بلاک);

        إذا (it != chunk.free_بلاکونه.begin()) {
            خپلکاری prev = it - 1;
            إذا (prev->دننه_واټن + prev->size == it->دننه_واټن) {
                it->دننه_واټن = prev->دننه_واټن;
                it->size += prev->size;
                it = chunk.free_بلاکونه.erase(prev);
            }
        }

        إذا (it + 1 != chunk.free_بلاکونه.end() && it->دننه_واټن + it->size == (it + 1)->دننه_واټن) {
            it->size += (it + 1)->size;
            chunk.free_بلاکونه.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<فراغ *, pool_ویش> allocations_;
    size_t                                      default_chunk_اندازه_2{ 0 };
};

صنف spine_mem_pool_posix final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_اندازه, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) hint_addr;

        ثابت size_t alloc_alignment = std::max(alignment, حجم(فراغ *));
        فراغ *       بنسټ            = تش_اشاره;
        ثابت صحيح    rc              = posix_memalign(&بنسټ, alloc_alignment, min_اندازه);
        إذا (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_اندازه, alloc_alignment, rc);
            أرجع خطأ;
        }

        chunk->بنسټ = static_cast<uint8_t *>(بنسټ);
        chunk->size = min_اندازه;
        chunk->fd   = -1;
        أرجع ریښتیا;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        std::حرر(chunk->بنسټ);
        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_transparent_hugepage final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_اندازه) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_اندازه, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;

        size_t chunk_اندازه = 0;
        إذا (!align_پورته(min_اندازه, default_chunk_اندازه(), &chunk_اندازه)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_اندازه);
            أرجع خطأ;
        }

        فراغ * map_addr = mmap(hint_addr, chunk_اندازه, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_اندازه,
                           errno);
            أرجع خطأ;
        }

        إذا (madvise(map_addr, chunk_اندازه, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_اندازه, errno);
            munmap(map_addr, chunk_اندازه);
            أرجع خطأ;
        }

        chunk->بنسټ = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_اندازه;
        chunk->fd   = -1;
        أرجع ریښتیا;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->بنسټ != تش_اشاره && chunk->size != 0 && munmap(chunk->بنسټ, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->بنسټ, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_hugetlb_1g final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1_G_REGION_اندازه) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_اندازه, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;
        (فراغ) hint_addr;

        size_t region_اندازه = 0;
        إذا (!align_پورته(min_اندازه, SPINE_MEM_POOL_1_G_REGION_اندازه, &region_اندازه)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_اندازه);
            أرجع خطأ;
        }

        ثابت صحيح fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        إذا (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            أرجع خطأ;
        }

        hugetlb_1g_region region;
        region.size  = region_اندازه;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        إذا (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_اندازه, errno);
            close(fd);
            أرجع خطأ;
        }

        فراغ * map_addr = mmap(تش_اشاره, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<بینښه اوږد اوږد>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            أرجع خطأ;
        }

        chunk->بنسټ = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        أرجع ریښتیا;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->بنسټ != تش_اشاره && chunk->size != 0 && munmap(chunk->بنسټ, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->بنسټ, chunk->size, errno);
        }

        إذا (chunk->fd >= 0) {
            إذا (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->بنسټ, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

صنف spine_mem_pool_shared_mem final : عام spine_mem_pool_manager {
  عام:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_اندازه) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  خاص:
    منطقي alloc_chunk(size_t min_اندازه, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) override {
        (فراغ) alignment;

        إذا (hint_addr != تش_اشاره) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            أرجع خطأ;
        }

        إذا (min_اندازه > default_chunk_اندازه()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_اندازه, default_chunk_اندازه());
            أرجع خطأ;
        }

        ثابت صحيح fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        إذا (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            أرجع خطأ;
        }

        فراغ * map_addr = mmap(تش_اشاره, default_chunk_اندازه(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_اندازه(), errno);
            close(fd);
            أرجع خطأ;
        }

        chunk->بنسټ = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_اندازه();
        chunk->fd   = fd;
        أرجع ریښتیا;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->بنسټ != تش_اشاره && chunk->size != 0 && munmap(chunk->بنسټ, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->بنسټ, chunk->size, errno);
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
    إذا (info == تش_اشاره) {
        أرجع خطأ;
    }

    *info = {};

    إذا (spine_tcm_open_handle(عدم) != 0 || !spine_tcm_is_شته()) {
        أرجع خطأ;
    }

    spine_tcm_mem_info_t mem_info;
    إذا (spine_tcm_mem_info(&mem_info) != 0) {
        أرجع خطأ;
    }

    info->شته   = ریښتیا;
    info->blk_اندازه    = mem_info.blk_اندازه;
    info->blk_عدد     = mem_info.blk_عدد;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    أرجع ریښتیا;
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
        أرجع تش_اشاره;
    }
}

فراغ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    حاول {
        أرجع get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        أرجع تش_اشاره;
    }
}

فراغ spine_mem_pool_free(فراغ * بنسټ) noexcept {
    حاول {
        get_spine_mem_pool_manager().حرر(بنسټ);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, بنسټ);
    }
}

فراغ spine_mem_pool_shared_mem_free(فراغ * بنسټ) noexcept {
    حاول {
        get_spine_mem_pool_shared_mem_manager().حرر(بنسټ);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, بنسټ);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

خارجي "C" {
فراغ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    فراغ * پایله = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    إذا (پایله == تش_اشاره) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    أرجع پایله;
}

فراغ ggml_backend_cpu_riscv64_spacemit_free_shared(فراغ * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
