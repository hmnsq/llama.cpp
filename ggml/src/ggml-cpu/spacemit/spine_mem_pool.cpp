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

#تعریف_کن HUGETLB_1G_IOC_MAGIC 'M'
#تعریف_کن HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, بنية hugetlb_1g_region)
#تعریف_کن HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

بنية free_بلوک {
    size_t جابه_جایی{ 0 };
    size_t size{ 0 };
};

بنية pool_chunk {
    uint8_t *               پایه{ اشاره_تهی };
    size_t                  size{ 0 };
    صحيح                     fd{ -1 };
    std::vector<free_بلوک> free_بلوکها;
};

بنية pool_تخصیص {
    فراغ * chunk_پایه{ اشاره_تهی };
    size_t chunk_اندازه{ 0 };
    فراغ * پایه{ اشاره_تهی };
    size_t size{ 0 };
};

منطقي is_power_از_two(size_t value) {
    أرجع value != 0 && (value & (value - 1)) == 0;
}

منطقي align_به_بالا(size_t value, size_t alignment, size_t * aligned_مقدار) {
    إذا (aligned_مقدار == اشاره_تهی || alignment == 0) {
        أرجع خطأ;
    }

    ثابت size_t remainder = value % alignment;
    إذا (remainder == 0) {
        *aligned_مقدار = value;
        أرجع درست;
    }

    ثابت size_t padding = alignment - remainder;
    إذا (value > std::numeric_limits<size_t>::max() - padding) {
        أرجع خطأ;
    }

    *aligned_مقدار = value + padding;
    أرجع درست;
}

منطقي align_به_بالا_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_مقدار) {
    إذا (aligned_مقدار == اشاره_تهی || alignment == 0) {
        أرجع خطأ;
    }

    ثابت uintptr_t remainder = value % alignment;
    إذا (remainder == 0) {
        *aligned_مقدار = value;
        أرجع درست;
    }

    ثابت uintptr_t padding = alignment - remainder;
    إذا (value > std::numeric_limits<uintptr_t>::max() - padding) {
        أرجع خطأ;
    }

    *aligned_مقدار = value + padding;
    أرجع درست;
}

صنف spine_mem_pool_manager {
  عام:
    صریح spine_mem_pool_manager(size_t default_chunk_اندازه) : default_chunk_اندازه_2(default_chunk_اندازه) {}

    مجازی ~spine_mem_pool_manager() = افتراضي;

    فراغ * alloc(size_t size, size_t alignment) {
        إذا (size == 0 || !is_power_از_two(alignment)) {
            أرجع اشاره_تهی;
        }

        size_t aligned_اندازه = 0;
        إذا (!align_به_بالا(size, alignment, &aligned_اندازه)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            أرجع اشاره_تهی;
        }

        pool_تخصیص تخصیص;

        std::lock_guard<std::mutex> قفل(mutex_);

        إذا (!try_alloc_locked(aligned_اندازه, alignment, &تخصیص)) {
            إذا (!جمع_کن_chunk_locked(aligned_اندازه, alignment)) {
                أرجع اشاره_تهی;
            }

            إذا (!try_alloc_locked(aligned_اندازه, alignment, &تخصیص)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_اندازه, alignment);
                أرجع اشاره_تهی;
            }
        }

        حاول {
            ثابت خودکار [تخصیص_it, inserted] = allocations_.emplace(تخصیص.پایه, تخصیص);
            إذا (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, تخصیص.پایه);
                rollback_تخصیص_locked(تخصیص);
                أرجع اشاره_تهی;
            }
        } امسك (ثابت std::bad_alloc &) {
            rollback_تخصیص_locked(تخصیص);
            ارم;
        }

        أرجع تخصیص.پایه;
    }

    فراغ حرر(فراغ * پایه) {
        إذا (پایه == اشاره_تهی) {
            أرجع;
        }

        std::lock_guard<std::mutex> قفل(mutex_);

        خودکار تخصیص_it = allocations_.find(پایه);
        إذا (تخصیص_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, پایه);
            أرجع;
        }

        pool_تخصیص تخصیص = تخصیص_it->second;
        allocations_.erase(تخصیص_it);

        خودکار chunk_it = پیدا_کن_chunk_locked(تخصیص);
        إذا (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           تخصیص.پایه, تخصیص.size);
            أرجع;
        }

        خودکار * chunk_پایه = chunk_it->پایه;
        خودکار * alloc_پایه = static_cast<uint8_t *>(تخصیص.پایه);
        إذا (alloc_پایه < chunk_پایه || alloc_پایه >= chunk_پایه + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           تخصیص.پایه, chunk_پایه, chunk_پایه + chunk_it->size);
            أرجع;
        }

        ثابت size_t جابه_جایی = static_cast<size_t>(alloc_پایه - chunk_پایه);
        إذا (جابه_جایی > chunk_it->size || تخصیص.size > chunk_it->size - جابه_جایی) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           تخصیص.پایه, تخصیص.size, chunk_it->size);
            أرجع;
        }

        درج_free_بلوک_locked(*chunk_it, { جابه_جایی, تخصیص.size });
        maybe_release_خالی_chunk_locked(chunk_it);
    }

  محمي:
    فراغ release_chunks() {
        std::lock_guard<std::mutex> قفل(mutex_);

        allocations_.clear();
        لكل (خودکار & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_اندازه() ثابت { أرجع default_chunk_اندازه_2; }

    ساكن فراغ clear_chunk(pool_chunk * chunk) {
        chunk->پایه = اشاره_تهی;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_بلوکها.clear();
    }

    مجازی منطقي alloc_chunk(size_t min_اندازه, size_t alignment, فراغ * hint_addr, pool_chunk * chunk) = 0;
    مجازی فراغ dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  خاص:
    بنية alloc_candidate {
        size_t    chunk_شاخص{ 0 };
        size_t    بلوک_شاخص{ 0 };
        size_t    aligned_جابه_جایی{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        منطقي      valid{ خطأ };
    };

    std::vector<pool_chunk>::iterator پیدا_کن_chunk_locked(ثابت pool_تخصیص & تخصیص) {
        أرجع std::find_if(chunks_.begin(), chunks_.end(), [&](ثابت pool_chunk & chunk) {
            أرجع chunk.پایه == تخصیص.chunk_پایه && chunk.size == تخصیص.chunk_اندازه;
        });
    }

    منطقي جمع_کن_chunk_locked(size_t min_اندازه, size_t alignment) {
        pool_chunk   chunk;
        ثابت size_t chunk_request = default_chunk_اندازه_2 == 0 ? min_اندازه : std::max(min_اندازه, default_chunk_اندازه_2);
        فراغ *       hint_addr     = اشاره_تهی;

        لكل (ثابت خودکار & existing_chunk : chunks_) {
            خودکار * chunk_انتها = existing_chunk.پایه + existing_chunk.size;
            إذا (hint_addr == اشاره_تهی || chunk_انتها > hint_addr) {
                hint_addr = chunk_انتها;
            }
        }

        إذا (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            أرجع خطأ;
        }

        إذا (chunk.پایه == اشاره_تهی || chunk.size < min_اندازه) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_اندازه, chunk.پایه, chunk.size);
            dealloc_chunk(&chunk);
            أرجع خطأ;
        }

        حاول {
            chunk.free_بلوکها.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } امسك (ثابت std::bad_alloc &) {
            dealloc_chunk(&chunk);
            ارم;
        }

        أرجع درست;
    }

    فراغ rollback_تخصیص_locked(ثابت pool_تخصیص & تخصیص) {
        خودکار chunk_it = پیدا_کن_chunk_locked(تخصیص);
        إذا (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, تخصیص.پایه);
            أرجع;
        }

        خودکار * chunk_پایه = chunk_it->پایه;
        خودکار * alloc_پایه = static_cast<uint8_t *>(تخصیص.پایه);
        إذا (alloc_پایه < chunk_پایه || alloc_پایه >= chunk_پایه + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, تخصیص.پایه);
            أرجع;
        }

        ثابت size_t جابه_جایی = static_cast<size_t>(alloc_پایه - chunk_پایه);
        إذا (جابه_جایی > chunk_it->size || تخصیص.size > chunk_it->size - جابه_جایی) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           تخصیص.پایه, تخصیص.size);
            أرجع;
        }

        درج_free_بلوک_locked(*chunk_it, { جابه_جایی, تخصیص.size });
        maybe_release_خالی_chunk_locked(chunk_it);
    }

    منطقي try_alloc_locked(size_t size, size_t alignment, pool_تخصیص * تخصیص) {
        alloc_candidate بهترین;

        لكل (size_t chunk_شاخص = 0; chunk_شاخص < chunks_.size(); ++chunk_شاخص) {
            ثابت خودکار & chunk = chunks_[chunk_شاخص];
            لكل (size_t بلوک_شاخص = 0; بلوک_شاخص < chunk.free_بلوکها.size(); ++بلوک_شاخص) {
                ثابت خودکار & بلوک = chunk.free_بلوکها[بلوک_شاخص];

                uintptr_t  aligned_addr = 0;
                ثابت خودکار بلوک_addr   = reinterpret_cast<uintptr_t>(chunk.پایه + بلوک.جابه_جایی);
                إذا (!align_به_بالا_uintptr(بلوک_addr, alignment, &aligned_addr)) {
                    استمر;
                }

                إذا (aligned_addr < بلوک_addr) {
                    استمر;
                }

                ثابت size_t aligned_جابه_جایی = بلوک.جابه_جایی + static_cast<size_t>(aligned_addr - بلوک_addr);
                ثابت size_t padding        = aligned_جابه_جایی - بلوک.جابه_جایی;
                إذا (padding > بلوک.size || size > بلوک.size - padding) {
                    استمر;
                }

                إذا (!بهترین.valid || aligned_addr < بهترین.address) {
                    بهترین.chunk_شاخص    = chunk_شاخص;
                    بهترین.بلوک_شاخص    = بلوک_شاخص;
                    بهترین.aligned_جابه_جایی = aligned_جابه_جایی;
                    بهترین.address        = aligned_addr;
                    بهترین.valid          = درست;
                }
            }
        }

        إذا (!بهترین.valid) {
            أرجع خطأ;
        }

        خودکار &           chunk     = chunks_[بهترین.chunk_شاخص];
        ثابت free_بلوک بلوک     = chunk.free_بلوکها[بهترین.بلوک_شاخص];
        ثابت size_t     padding   = بهترین.aligned_جابه_جایی - بلوک.جابه_جایی;
        ثابت size_t     alloc_انتها = بهترین.aligned_جابه_جایی + size;
        ثابت size_t     بلوک_انتها = بلوک.جابه_جایی + بلوک.size;

        chunk.free_بلوکها.erase(chunk.free_بلوکها.begin() + بهترین.بلوک_شاخص);
        خودکار درج_it = chunk.free_بلوکها.begin() + بهترین.بلوک_شاخص;
        إذا (padding != 0) {
            درج_it = chunk.free_بلوکها.insert(درج_it, { بلوک.جابه_جایی, padding });
            ++درج_it;
        }
        إذا (alloc_انتها < بلوک_انتها) {
            chunk.free_بلوکها.insert(درج_it, { alloc_انتها, بلوک_انتها - alloc_انتها });
        }

        تخصیص->chunk_پایه = chunk.پایه;
        تخصیص->chunk_اندازه = chunk.size;
        تخصیص->پایه       = chunk.پایه + بهترین.aligned_جابه_جایی;
        تخصیص->size       = size;
        أرجع درست;
    }

    فراغ maybe_release_خالی_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        إذا (chunk_it->free_بلوکها.size() != 1) {
            أرجع;
        }

        ثابت خودکار & بلوک = chunk_it->free_بلوکها.front();
        إذا (بلوک.جابه_جایی != 0 || بلوک.size != chunk_it->size) {
            أرجع;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    فراغ درج_free_بلوک_locked(pool_chunk & chunk, free_بلوک بلوک) {
        خودکار it = chunk.free_بلوکها.begin();
        طالما (it != chunk.free_بلوکها.end() && it->جابه_جایی < بلوک.جابه_جایی) {
            ++it;
        }

        إذا (it != chunk.free_بلوکها.begin()) {
            ثابت خودکار & prev = *(it - 1);
            إذا (prev.جابه_جایی + prev.size > بلوک.جابه_جایی) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               بلوک.جابه_جایی, بلوک.size);
                أرجع;
            }
        }

        إذا (it != chunk.free_بلوکها.end() && بلوک.جابه_جایی + بلوک.size > it->جابه_جایی) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           بلوک.جابه_جایی, بلوک.size);
            أرجع;
        }

        it = chunk.free_بلوکها.insert(it, بلوک);

        إذا (it != chunk.free_بلوکها.begin()) {
            خودکار prev = it - 1;
            إذا (prev->جابه_جایی + prev->size == it->جابه_جایی) {
                it->جابه_جایی = prev->جابه_جایی;
                it->size += prev->size;
                it = chunk.free_بلوکها.erase(prev);
            }
        }

        إذا (it + 1 != chunk.free_بلوکها.end() && it->جابه_جایی + it->size == (it + 1)->جابه_جایی) {
            it->size += (it + 1)->size;
            chunk.free_بلوکها.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<فراغ *, pool_تخصیص> allocations_;
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
        فراغ *       پایه            = اشاره_تهی;
        ثابت صحيح    rc              = posix_memalign(&پایه, alloc_alignment, min_اندازه);
        إذا (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_اندازه, alloc_alignment, rc);
            أرجع خطأ;
        }

        chunk->پایه = static_cast<uint8_t *>(پایه);
        chunk->size = min_اندازه;
        chunk->fd   = -1;
        أرجع درست;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        std::حرر(chunk->پایه);
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
        إذا (!align_به_بالا(min_اندازه, default_chunk_اندازه(), &chunk_اندازه)) {
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

        chunk->پایه = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_اندازه;
        chunk->fd   = -1;
        أرجع درست;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->پایه != اشاره_تهی && chunk->size != 0 && munmap(chunk->پایه, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->پایه, chunk->size, errno);
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
        إذا (!align_به_بالا(min_اندازه, SPINE_MEM_POOL_1_G_REGION_اندازه, &region_اندازه)) {
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

        فراغ * map_addr = mmap(اشاره_تهی, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<بیعلامت دراز دراز>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            أرجع خطأ;
        }

        chunk->پایه = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        أرجع درست;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->پایه != اشاره_تهی && chunk->size != 0 && munmap(chunk->پایه, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->پایه, chunk->size, errno);
        }

        إذا (chunk->fd >= 0) {
            إذا (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->پایه, errno);
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

        إذا (hint_addr != اشاره_تهی) {
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

        فراغ * map_addr = mmap(اشاره_تهی, default_chunk_اندازه(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        إذا (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_اندازه(), errno);
            close(fd);
            أرجع خطأ;
        }

        chunk->پایه = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_اندازه();
        chunk->fd   = fd;
        أرجع درست;
    }

    فراغ dealloc_chunk(pool_chunk * chunk) override {
        إذا (chunk->پایه != اشاره_تهی && chunk->size != 0 && munmap(chunk->پایه, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->پایه, chunk->size, errno);
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
    إذا (info == اشاره_تهی) {
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

    info->موجود   = درست;
    info->blk_اندازه    = mem_info.blk_اندازه;
    info->blk_رقم     = mem_info.blk_رقم;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    أرجع درست;
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
        أرجع اشاره_تهی;
    }
}

فراغ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    حاول {
        أرجع get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        أرجع اشاره_تهی;
    }
}

فراغ spine_mem_pool_free(فراغ * پایه) noexcept {
    حاول {
        get_spine_mem_pool_manager().حرر(پایه);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, پایه);
    }
}

فراغ spine_mem_pool_shared_mem_free(فراغ * پایه) noexcept {
    حاول {
        get_spine_mem_pool_shared_mem_manager().حرر(پایه);
    } امسك (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, پایه);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

خارجي "C" {
فراغ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    فراغ * نتیجه = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    إذا (نتیجه == اشاره_تهی) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    أرجع نتیجه;
}

فراغ ggml_backend_cpu_riscv64_spacemit_free_shared(فراغ * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
