#సూచన once

#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

గణన class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

నిర్మాణం spine_mem_pool_tcm_info {
    bool   అందుబాటు{ false };
    size_t blk_కొలత{ 0 };
    size_t blk_అంకె{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
శూన్య * spine_mem_pool_tcm_mem_get(పూర్ణాంకం cpu_id) noexcept;
శూన్య * spine_mem_pool_tcm_mem_wait(పూర్ణాంకం cpu_id) noexcept;
పూర్ణాంకం    spine_mem_pool_tcm_mem_release(పూర్ణాంకం cpu_id) noexcept;

శూన్య * spine_mem_pool_alloc(size_t కొలత, size_t alignment) noexcept;
శూన్య * spine_mem_pool_shared_mem_alloc(size_t కొలత, size_t alignment) noexcept;
శూన్య   spine_mem_pool_free(శూన్య * పునాది) noexcept;
శూన్య   spine_mem_pool_shared_mem_free(శూన్య * పునాది) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
