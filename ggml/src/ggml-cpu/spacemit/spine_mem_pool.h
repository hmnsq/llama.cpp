#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

ਗਿਣਤੀ class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

ਬਣਤਰ spine_mem_pool_tcm_info {
    bool   ਉਪਲਬਧ{ false };
    size_t blk_ਮਾਪ{ 0 };
    size_t blk_ਅੰਕ{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
ਖਾਲੀ * spine_mem_pool_tcm_mem_get(ਪੂਰਨ cpu_id) noexcept;
ਖਾਲੀ * spine_mem_pool_tcm_mem_wait(ਪੂਰਨ cpu_id) noexcept;
ਪੂਰਨ    spine_mem_pool_tcm_mem_release(ਪੂਰਨ cpu_id) noexcept;

ਖਾਲੀ * spine_mem_pool_alloc(size_t ਮਾਪ, size_t alignment) noexcept;
ਖਾਲੀ * spine_mem_pool_shared_mem_alloc(size_t ਮਾਪ, size_t alignment) noexcept;
ਖਾਲੀ   spine_mem_pool_free(ਖਾਲੀ * ਅਧਾਰ) noexcept;
ਖਾਲੀ   spine_mem_pool_shared_mem_free(ਖਾਲੀ * ਅਧਾਰ) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
