#pragma once

#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

எண்ணிக்கை class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

கட்டமைப்பு spine_mem_pool_tcm_info {
    bool   available{ false };
    size_t blk_size{ 0 };
    size_t blk_இலக்கம்{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
வெறுமை * spine_mem_pool_tcm_mem_get(முழு cpu_id) noexcept;
வெறுமை * spine_mem_pool_tcm_mem_wait(முழு cpu_id) noexcept;
முழு    spine_mem_pool_tcm_mem_release(முழு cpu_id) noexcept;

வெறுமை * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept;
வெறுமை * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept;
வெறுமை   spine_mem_pool_free(வெறுமை * base) noexcept;
வெறுமை   spine_mem_pool_shared_mem_free(வெறுமை * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
