#شامل "spine_mem_pool.h"

#شامل "common.h"
#شامل "ime_env.h"
#شامل "spine_tcm.h"

#شامل <fcntl.h>
#شامل <sys/ioctl.h>
#شامل <sys/mman.h>
#شامل <unistd.h>

#شامل <algorithm>
#شامل <cerrno>
#شامل <cstdint>
#شامل <cstdlib>
#شامل <limits>
#شامل <memory>
#شامل <mutex>
#شامل <unordered_map>
#شامل <vector>

نامخلا ggml::cpu::riscv64_spacemit {
نامخلا {

constexpr size_t   SPINE_MEM_POOL_CHUNK_سائز         = 512ull * 1024ull * 1024ull;
constexpr size_t   SPINE_SHARE_MEM_POOL_CHUNK_سائز   = 512ull * 1024ull;
constexpr size_t   SPINE_MEM_POOL_1_G_REGION_سائز     = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD       = 1ull << 0;
constexpr حرف     SPINE_MEM_POOL_HUGETLB_1G_DEV[]   = "/dev/hugetlb_1g";
constexpr حرف     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

ساخت hugetlb_1g_region {
    uint64_t size{ 0 };
    uint64_t dma_addr{ 0 };
    uint64_t flags{ 0 };
    uint64_t reserved{ 0 };
};

#تعریف_کرو HUGETLB_1G_IOC_MAGIC 'M'
#تعریف_کرو HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, ساخت hugetlb_1g_region)
#تعریف_کرو HUGETLB_1G_IOC_FREE  _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

ساخت free_بلاک {
    size_t انحراف{ 0 };
    size_t size{ 0 };
};

ساخت pool_chunk {
    uint8_t *               بنیاد{ خالی_اشاریہ };
    size_t                  size{ 0 };
    صحیح                     fd{ -1 };
    std::vector<free_بلاک> free_بلاکیں;
};

ساخت pool_تخصیص {
    خالی * chunk_بنیاد{ خالی_اشاریہ };
    size_t chunk_سائز{ 0 };
    خالی * بنیاد{ خالی_اشاریہ };
    size_t size{ 0 };
};

منطقی is_power_کا_two(size_t value) {
    واپس value != 0 && (value & (value - 1)) == 0;
}

منطقی align_اوپر(size_t value, size_t alignment, size_t * aligned_قدر) {
    اگر (aligned_قدر == خالی_اشاریہ || alignment == 0) {
        واپس جھوٹ;
    }

    ثابت size_t remainder = value % alignment;
    اگر (remainder == 0) {
        *aligned_قدر = value;
        واپس سچ;
    }

    ثابت size_t padding = alignment - remainder;
    اگر (value > std::numeric_limits<size_t>::max() - padding) {
        واپس جھوٹ;
    }

    *aligned_قدر = value + padding;
    واپس سچ;
}

منطقی align_اوپر_uintptr(uintptr_t value, size_t alignment, uintptr_t * aligned_قدر) {
    اگر (aligned_قدر == خالی_اشاریہ || alignment == 0) {
        واپس جھوٹ;
    }

    ثابت uintptr_t remainder = value % alignment;
    اگر (remainder == 0) {
        *aligned_قدر = value;
        واپس سچ;
    }

    ثابت uintptr_t padding = alignment - remainder;
    اگر (value > std::numeric_limits<uintptr_t>::max() - padding) {
        واپس جھوٹ;
    }

    *aligned_قدر = value + padding;
    واپس سچ;
}

جماعت spine_mem_pool_manager {
  عوامی:
    صریح spine_mem_pool_manager(size_t default_chunk_سائز) : default_chunk_سائز_2(default_chunk_سائز) {}

    مجازی ~spine_mem_pool_manager() = طےشدہ;

    خالی * alloc(size_t size, size_t alignment) {
        اگر (size == 0 || !is_power_کا_two(alignment)) {
            واپس خالی_اشاریہ;
        }

        size_t aligned_سائز = 0;
        اگر (!align_اوپر(size, alignment, &aligned_سائز)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, size,
                           alignment);
            واپس خالی_اشاریہ;
        }

        pool_تخصیص تخصیص;

        std::lock_guard<std::mutex> تالا(mutex_);

        اگر (!try_alloc_locked(aligned_سائز, alignment, &تخصیص)) {
            اگر (!جوڑو_chunk_locked(aligned_سائز, alignment)) {
                واپس خالی_اشاریہ;
            }

            اگر (!try_alloc_locked(aligned_سائز, alignment, &تخصیص)) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n",
                               __func__, aligned_سائز, alignment);
                واپس خالی_اشاریہ;
            }
        }

        کوشش {
            ثابت خودکار [تخصیص_it, inserted] = allocations_.emplace(تخصیص.بنیاد, تخصیص);
            اگر (!inserted) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, تخصیص.بنیاد);
                rollback_تخصیص_locked(تخصیص);
                واپس خالی_اشاریہ;
            }
        } پکڑو (ثابت std::bad_alloc &) {
            rollback_تخصیص_locked(تخصیص);
            پھینکو;
        }

        واپس تخصیص.بنیاد;
    }

    خالی آزاد(خالی * بنیاد) {
        اگر (بنیاد == خالی_اشاریہ) {
            واپس;
        }

        std::lock_guard<std::mutex> تالا(mutex_);

        خودکار تخصیص_it = allocations_.find(بنیاد);
        اگر (تخصیص_it == allocations_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, بنیاد);
            واپس;
        }

        pool_تخصیص تخصیص = تخصیص_it->second;
        allocations_.erase(تخصیص_it);

        خودکار chunk_it = ڈھونڈو_chunk_locked(تخصیص);
        اگر (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                           تخصیص.بنیاد, تخصیص.size);
            واپس;
        }

        خودکار * chunk_بنیاد = chunk_it->بنیاد;
        خودکار * alloc_بنیاد = static_cast<uint8_t *>(تخصیص.بنیاد);
        اگر (alloc_بنیاد < chunk_بنیاد || alloc_بنیاد >= chunk_بنیاد + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__,
                           تخصیص.بنیاد, chunk_بنیاد, chunk_بنیاد + chunk_it->size);
            واپس;
        }

        ثابت size_t انحراف = static_cast<size_t>(alloc_بنیاد - chunk_بنیاد);
        اگر (انحراف > chunk_it->size || تخصیص.size > chunk_it->size - انحراف) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                           تخصیص.بنیاد, تخصیص.size, chunk_it->size);
            واپس;
        }

        داخل_کرو_free_بلاک_locked(*chunk_it, { انحراف, تخصیص.size });
        maybe_release_خالی_خانے_chunk_locked(chunk_it);
    }

  محفوظ:
    خالی release_chunks() {
        std::lock_guard<std::mutex> تالا(mutex_);

        allocations_.clear();
        برائے (خودکار & chunk : chunks_) {
            dealloc_chunk(&chunk);
        }
        chunks_.clear();
    }

    size_t default_chunk_سائز() ثابت { واپس default_chunk_سائز_2; }

    ساکن خالی clear_chunk(pool_chunk * chunk) {
        chunk->بنیاد = خالی_اشاریہ;
        chunk->size = 0;
        chunk->fd   = -1;
        chunk->free_بلاکیں.clear();
    }

    مجازی منطقی alloc_chunk(size_t min_سائز, size_t alignment, خالی * hint_addr, pool_chunk * chunk) = 0;
    مجازی خالی dealloc_chunk(pool_chunk * chunk)                                                    = 0;

  نجی:
    ساخت alloc_candidate {
        size_t    chunk_اشاریہ{ 0 };
        size_t    بلاک_اشاریہ{ 0 };
        size_t    aligned_انحراف{ 0 };
        uintptr_t address{ std::numeric_limits<uintptr_t>::max() };
        منطقی      valid{ جھوٹ };
    };

    std::vector<pool_chunk>::iterator ڈھونڈو_chunk_locked(ثابت pool_تخصیص & تخصیص) {
        واپس std::find_if(chunks_.begin(), chunks_.end(), [&](ثابت pool_chunk & chunk) {
            واپس chunk.بنیاد == تخصیص.chunk_بنیاد && chunk.size == تخصیص.chunk_سائز;
        });
    }

    منطقی جوڑو_chunk_locked(size_t min_سائز, size_t alignment) {
        pool_chunk   chunk;
        ثابت size_t chunk_request = default_chunk_سائز_2 == 0 ? min_سائز : std::max(min_سائز, default_chunk_سائز_2);
        خالی *       hint_addr     = خالی_اشاریہ;

        برائے (ثابت خودکار & existing_chunk : chunks_) {
            خودکار * chunk_انتہا = existing_chunk.بنیاد + existing_chunk.size;
            اگر (hint_addr == خالی_اشاریہ || chunk_انتہا > hint_addr) {
                hint_addr = chunk_انتہا;
            }
        }

        اگر (!alloc_chunk(chunk_request, alignment, hint_addr, &chunk)) {
            واپس جھوٹ;
        }

        اگر (chunk.بنیاد == خالی_اشاریہ || chunk.size < min_سائز) {
            GGML_LOG_ERROR(
                "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
                __func__, min_سائز, chunk.بنیاد, chunk.size);
            dealloc_chunk(&chunk);
            واپس جھوٹ;
        }

        کوشش {
            chunk.free_بلاکیں.push_back({ 0, chunk.size });
            chunks_.push_back(std::move(chunk));
        } پکڑو (ثابت std::bad_alloc &) {
            dealloc_chunk(&chunk);
            پھینکو;
        }

        واپس سچ;
    }

    خالی rollback_تخصیص_locked(ثابت pool_تخصیص & تخصیص) {
        خودکار chunk_it = ڈھونڈو_chunk_locked(تخصیص);
        اگر (chunk_it == chunks_.end()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n",
                           __func__, تخصیص.بنیاد);
            واپس;
        }

        خودکار * chunk_بنیاد = chunk_it->بنیاد;
        خودکار * alloc_بنیاد = static_cast<uint8_t *>(تخصیص.بنیاد);
        اگر (alloc_بنیاد < chunk_بنیاد || alloc_بنیاد >= chunk_بنیاد + chunk_it->size) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n",
                           __func__, تخصیص.بنیاد);
            واپس;
        }

        ثابت size_t انحراف = static_cast<size_t>(alloc_بنیاد - chunk_بنیاد);
        اگر (انحراف > chunk_it->size || تخصیص.size > chunk_it->size - انحراف) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                           تخصیص.بنیاد, تخصیص.size);
            واپس;
        }

        داخل_کرو_free_بلاک_locked(*chunk_it, { انحراف, تخصیص.size });
        maybe_release_خالی_خانے_chunk_locked(chunk_it);
    }

    منطقی try_alloc_locked(size_t size, size_t alignment, pool_تخصیص * تخصیص) {
        alloc_candidate بہترین;

        برائے (size_t chunk_اشاریہ = 0; chunk_اشاریہ < chunks_.size(); ++chunk_اشاریہ) {
            ثابت خودکار & chunk = chunks_[chunk_اشاریہ];
            برائے (size_t بلاک_اشاریہ = 0; بلاک_اشاریہ < chunk.free_بلاکیں.size(); ++بلاک_اشاریہ) {
                ثابت خودکار & بلاک = chunk.free_بلاکیں[بلاک_اشاریہ];

                uintptr_t  aligned_addr = 0;
                ثابت خودکار بلاک_addr   = reinterpret_cast<uintptr_t>(chunk.بنیاد + بلاک.انحراف);
                اگر (!align_اوپر_uintptr(بلاک_addr, alignment, &aligned_addr)) {
                    جاری;
                }

                اگر (aligned_addr < بلاک_addr) {
                    جاری;
                }

                ثابت size_t aligned_انحراف = بلاک.انحراف + static_cast<size_t>(aligned_addr - بلاک_addr);
                ثابت size_t padding        = aligned_انحراف - بلاک.انحراف;
                اگر (padding > بلاک.size || size > بلاک.size - padding) {
                    جاری;
                }

                اگر (!بہترین.valid || aligned_addr < بہترین.address) {
                    بہترین.chunk_اشاریہ    = chunk_اشاریہ;
                    بہترین.بلاک_اشاریہ    = بلاک_اشاریہ;
                    بہترین.aligned_انحراف = aligned_انحراف;
                    بہترین.address        = aligned_addr;
                    بہترین.valid          = سچ;
                }
            }
        }

        اگر (!بہترین.valid) {
            واپس جھوٹ;
        }

        خودکار &           chunk     = chunks_[بہترین.chunk_اشاریہ];
        ثابت free_بلاک بلاک     = chunk.free_بلاکیں[بہترین.بلاک_اشاریہ];
        ثابت size_t     padding   = بہترین.aligned_انحراف - بلاک.انحراف;
        ثابت size_t     alloc_انتہا = بہترین.aligned_انحراف + size;
        ثابت size_t     بلاک_انتہا = بلاک.انحراف + بلاک.size;

        chunk.free_بلاکیں.erase(chunk.free_بلاکیں.begin() + بہترین.بلاک_اشاریہ);
        خودکار داخل_کرو_it = chunk.free_بلاکیں.begin() + بہترین.بلاک_اشاریہ;
        اگر (padding != 0) {
            داخل_کرو_it = chunk.free_بلاکیں.insert(داخل_کرو_it, { بلاک.انحراف, padding });
            ++داخل_کرو_it;
        }
        اگر (alloc_انتہا < بلاک_انتہا) {
            chunk.free_بلاکیں.insert(داخل_کرو_it, { alloc_انتہا, بلاک_انتہا - alloc_انتہا });
        }

        تخصیص->chunk_بنیاد = chunk.بنیاد;
        تخصیص->chunk_سائز = chunk.size;
        تخصیص->بنیاد       = chunk.بنیاد + بہترین.aligned_انحراف;
        تخصیص->size       = size;
        واپس سچ;
    }

    خالی maybe_release_خالی_خانے_chunk_locked(std::vector<pool_chunk>::iterator chunk_it) {
        اگر (chunk_it->free_بلاکیں.size() != 1) {
            واپس;
        }

        ثابت خودکار & بلاک = chunk_it->free_بلاکیں.front();
        اگر (بلاک.انحراف != 0 || بلاک.size != chunk_it->size) {
            واپس;
        }

        dealloc_chunk(&*chunk_it);
        chunks_.erase(chunk_it);
    }

    خالی داخل_کرو_free_بلاک_locked(pool_chunk & chunk, free_بلاک بلاک) {
        خودکار it = chunk.free_بلاکیں.begin();
        جب (it != chunk.free_بلاکیں.end() && it->انحراف < بلاک.انحراف) {
            ++it;
        }

        اگر (it != chunk.free_بلاکیں.begin()) {
            ثابت خودکار & پچھلا = *(it - 1);
            اگر (پچھلا.انحراف + پچھلا.size > بلاک.انحراف) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                               بلاک.انحراف, بلاک.size);
                واپس;
            }
        }

        اگر (it != chunk.free_بلاکیں.end() && بلاک.انحراف + بلاک.size > it->انحراف) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                           بلاک.انحراف, بلاک.size);
            واپس;
        }

        it = chunk.free_بلاکیں.insert(it, بلاک);

        اگر (it != chunk.free_بلاکیں.begin()) {
            خودکار پچھلا = it - 1;
            اگر (پچھلا->انحراف + پچھلا->size == it->انحراف) {
                it->انحراف = پچھلا->انحراف;
                it->size += پچھلا->size;
                it = chunk.free_بلاکیں.erase(پچھلا);
            }
        }

        اگر (it + 1 != chunk.free_بلاکیں.end() && it->انحراف + it->size == (it + 1)->انحراف) {
            it->size += (it + 1)->size;
            chunk.free_بلاکیں.erase(it + 1);
        }
    }

    std::mutex                                  mutex_;
    std::vector<pool_chunk>                     chunks_;
    std::unordered_map<خالی *, pool_تخصیص> allocations_;
    size_t                                      default_chunk_سائز_2{ 0 };
};

جماعت spine_mem_pool_posix final : عوامی spine_mem_pool_manager {
  عوامی:
    spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

    ~spine_mem_pool_posix() override { release_chunks(); }

  نجی:
    منطقی alloc_chunk(size_t min_سائز, size_t alignment, خالی * hint_addr, pool_chunk * chunk) override {
        (خالی) hint_addr;

        ثابت size_t alloc_alignment = std::max(alignment, حجم(خالی *));
        خالی *       بنیاد            = خالی_اشاریہ;
        ثابت صحیح    rc              = posix_memalign(&بنیاد, alloc_alignment, min_سائز);
        اگر (rc != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n",
                           __func__, min_سائز, alloc_alignment, rc);
            واپس جھوٹ;
        }

        chunk->بنیاد = static_cast<uint8_t *>(بنیاد);
        chunk->size = min_سائز;
        chunk->fd   = -1;
        واپس سچ;
    }

    خالی dealloc_chunk(pool_chunk * chunk) override {
        std::آزاد(chunk->بنیاد);
        clear_chunk(chunk);
    }
};

جماعت spine_mem_pool_transparent_hugepage final : عوامی spine_mem_pool_manager {
  عوامی:
    spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_سائز) {}

    ~spine_mem_pool_transparent_hugepage() override { release_chunks(); }

  نجی:
    منطقی alloc_chunk(size_t min_سائز, size_t alignment, خالی * hint_addr, pool_chunk * chunk) override {
        (خالی) alignment;

        size_t chunk_سائز = 0;
        اگر (!align_اوپر(min_سائز, default_chunk_سائز(), &chunk_سائز)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, min_سائز);
            واپس جھوٹ;
        }

        خالی * map_addr = mmap(hint_addr, chunk_سائز, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        اگر (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, chunk_سائز,
                           errno);
            واپس جھوٹ;
        }

        اگر (madvise(map_addr, chunk_سائز, MADV_HUGEPAGE) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n",
                           __func__, chunk_سائز, errno);
            munmap(map_addr, chunk_سائز);
            واپس جھوٹ;
        }

        chunk->بنیاد = static_cast<uint8_t *>(map_addr);
        chunk->size = chunk_سائز;
        chunk->fd   = -1;
        واپس سچ;
    }

    خالی dealloc_chunk(pool_chunk * chunk) override {
        اگر (chunk->بنیاد != خالی_اشاریہ && chunk->size != 0 && munmap(chunk->بنیاد, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                           chunk->بنیاد, chunk->size, errno);
        }

        clear_chunk(chunk);
    }
};

جماعت spine_mem_pool_hugetlb_1g final : عوامی spine_mem_pool_manager {
  عوامی:
    spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1_G_REGION_سائز) {}

    ~spine_mem_pool_hugetlb_1g() override { release_chunks(); }

  نجی:
    منطقی alloc_chunk(size_t min_سائز, size_t alignment, خالی * hint_addr, pool_chunk * chunk) override {
        (خالی) alignment;
        (خالی) hint_addr;

        size_t region_سائز = 0;
        اگر (!align_اوپر(min_سائز, SPINE_MEM_POOL_1_G_REGION_سائز, &region_سائز)) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, min_سائز);
            واپس جھوٹ;
        }

        ثابت صحیح fd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
        اگر (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_HUGETLB_1G_DEV, errno);
            واپس جھوٹ;
        }

        hugetlb_1g_region region;
        region.size  = region_سائز;
        region.flags = HUGETLB_1G_FLAG_REQUIRE_PUD;
        اگر (ioctl(fd, HUGETLB_1G_IOC_ALLOC, &region) < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                           region_سائز, errno);
            close(fd);
            واپس جھوٹ;
        }

        خالی * map_addr = mmap(خالی_اشاریہ, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        اگر (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                           static_cast<غیرعلامتی طویل طویل>(region.size), errno);
            ioctl(fd, HUGETLB_1G_IOC_FREE);
            close(fd);
            واپس جھوٹ;
        }

        chunk->بنیاد = static_cast<uint8_t *>(map_addr);
        chunk->size = region.size;
        chunk->fd   = fd;
        واپس سچ;
    }

    خالی dealloc_chunk(pool_chunk * chunk) override {
        اگر (chunk->بنیاد != خالی_اشاریہ && chunk->size != 0 && munmap(chunk->بنیاد, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n",
                           __func__, chunk->بنیاد, chunk->size, errno);
        }

        اگر (chunk->fd >= 0) {
            اگر (ioctl(chunk->fd, HUGETLB_1G_IOC_FREE) < 0) {
                GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n",
                               __func__, chunk->بنیاد, errno);
            }

            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

جماعت spine_mem_pool_shared_mem final : عوامی spine_mem_pool_manager {
  عوامی:
    spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_سائز) {}

    ~spine_mem_pool_shared_mem() override { release_chunks(); }

  نجی:
    منطقی alloc_chunk(size_t min_سائز, size_t alignment, خالی * hint_addr, pool_chunk * chunk) override {
        (خالی) alignment;

        اگر (hint_addr != خالی_اشاریہ) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
            واپس جھوٹ;
        }

        اگر (min_سائز > default_chunk_سائز()) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__,
                           min_سائز, default_chunk_سائز());
            واپس جھوٹ;
        }

        ثابت صحیح fd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
        اگر (fd < 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, errno);
            واپس جھوٹ;
        }

        خالی * map_addr = mmap(خالی_اشاریہ, default_chunk_سائز(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        اگر (map_addr == MAP_FAILED) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                           SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, default_chunk_سائز(), errno);
            close(fd);
            واپس جھوٹ;
        }

        chunk->بنیاد = static_cast<uint8_t *>(map_addr);
        chunk->size = default_chunk_سائز();
        chunk->fd   = fd;
        واپس سچ;
    }

    خالی dealloc_chunk(pool_chunk * chunk) override {
        اگر (chunk->بنیاد != خالی_اشاریہ && chunk->size != 0 && munmap(chunk->بنیاد, chunk->size) != 0) {
            GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n",
                           __func__, chunk->بنیاد, chunk->size, errno);
        }

        اگر (chunk->fd >= 0) {
            close(chunk->fd);
        }

        clear_chunk(chunk);
    }
};

spine_mem_pool_manager & get_spine_mem_pool_manager() {
    ساکن std::once_flag                          pool_once;
    ساکن std::unique_ptr<spine_mem_pool_manager> selected_pool;
    ساکن spine_mem_pool_backend                  selected_backend = spine_mem_pool_backend::none;

    spine_mem_pool_backend backend = global_spine_env_info.mem_backend;
    اگر (backend == spine_mem_pool_backend::none) {
        backend = spine_mem_pool_backend::transparent_hugepage;
    }

    std::call_once(pool_once, [&]() {
        selected_backend = backend;

        انتخاب (selected_backend) {
            حالت spine_mem_pool_backend::posix_memalign:
                selected_pool = std::make_unique<spine_mem_pool_posix>();
                توڑو;
            حالت spine_mem_pool_backend::transparent_hugepage:
                selected_pool = std::make_unique<spine_mem_pool_transparent_hugepage>();
                توڑو;
            حالت spine_mem_pool_backend::hugetlb_1g:
                selected_pool = std::make_unique<spine_mem_pool_hugetlb_1g>();
                توڑو;
            حالت spine_mem_pool_backend::none:
                selected_backend = spine_mem_pool_backend::transparent_hugepage;
                selected_pool    = std::make_unique<spine_mem_pool_transparent_hugepage>();
                توڑو;
        }
    });

    اگر (backend != selected_backend) {
        GGML_LOG_ERROR(
            "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
            "selected=%d\n",
            __func__, static_cast<صحیح>(backend), static_cast<صحیح>(selected_backend));
    }

    اگر (selected_pool) {
        واپس *selected_pool;
    }

    پھینکو std::bad_alloc();
}

spine_mem_pool_manager & get_spine_mem_pool_shared_mem_manager() {
    ساکن std::once_flag                             shared_mem_pool_once;
    ساکن std::unique_ptr<spine_mem_pool_shared_mem> shared_mem_pool;

    std::call_once(shared_mem_pool_once, [&]() { shared_mem_pool = std::make_unique<spine_mem_pool_shared_mem>(); });

    اگر (shared_mem_pool) {
        واپس *shared_mem_pool;
    }

    پھینکو std::bad_alloc();
}

}  // namespace

منطقی spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept {
    اگر (info == خالی_اشاریہ) {
        واپس جھوٹ;
    }

    *info = {};

    اگر (spine_tcm_open_handle(تہی) != 0 || !spine_tcm_is_دستیاب()) {
        واپس جھوٹ;
    }

    spine_tcm_mem_info_t mem_info;
    اگر (spine_tcm_mem_info(&mem_info) != 0) {
        واپس جھوٹ;
    }

    info->دستیاب   = سچ;
    info->blk_سائز    = mem_info.blk_سائز;
    info->blk_عدد     = mem_info.blk_عدد;
    info->is_fake_tcm = mem_info.is_fake_tcm != 0;
    واپس سچ;
}

خالی * spine_mem_pool_tcm_mem_get(صحیح cpu_id) noexcept {
    واپس spine_tcm_mem_get(cpu_id);
}

خالی * spine_mem_pool_tcm_mem_wait(صحیح cpu_id) noexcept {
    واپس spine_tcm_mem_try_wait(cpu_id, 1000 * 1000);
}

صحیح spine_mem_pool_tcm_mem_release(صحیح cpu_id) noexcept {
    واپس spine_tcm_mem_release(cpu_id);
}

خالی * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept {
    کوشش {
        واپس get_spine_mem_pool_manager().alloc(size, alignment);
    } پکڑو (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, size);
        واپس خالی_اشاریہ;
    }
}

خالی * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept {
    کوشش {
        واپس get_spine_mem_pool_shared_mem_manager().alloc(size, alignment);
    } پکڑو (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, size);
        واپس خالی_اشاریہ;
    }
}

خالی spine_mem_pool_free(خالی * بنیاد) noexcept {
    کوشش {
        get_spine_mem_pool_manager().آزاد(بنیاد);
    } پکڑو (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, بنیاد);
    }
}

خالی spine_mem_pool_shared_mem_free(خالی * بنیاد) noexcept {
    کوشش {
        get_spine_mem_pool_shared_mem_manager().آزاد(بنیاد);
    } پکڑو (ثابت std::bad_alloc &) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, بنیاد);
    }
}

}  // namespace ggml::cpu::riscv64_spacemit

خارجی "C" {
خالی * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment) {
    خالی * نتیجہ = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(size, alignment);
    اگر (نتیجہ == خالی_اشاریہ) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                       size, alignment);
    }
    واپس نتیجہ;
}

خالی ggml_backend_cpu_riscv64_spacemit_free_shared(خالی * ptr) {
    ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(ptr);
}
}
