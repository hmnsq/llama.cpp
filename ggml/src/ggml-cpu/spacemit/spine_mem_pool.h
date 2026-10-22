#pragma once

#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>

namespace ggml::cpu::riscv64_spacemit {

गणना class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

संरचना spine_mem_pool_tcm_info {
    bool   available{ false };
    size_t blk_size{ 0 };
    size_t blk_अंक{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
शून्य * spine_mem_pool_tcm_mem_get(पूर्णाङ्क cpu_id) noexcept;
शून्य * spine_mem_pool_tcm_mem_wait(पूर्णाङ्क cpu_id) noexcept;
पूर्णाङ्क    spine_mem_pool_tcm_mem_release(पूर्णाङ्क cpu_id) noexcept;

शून्य * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept;
शून्य * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept;
शून्य   spine_mem_pool_free(शून्य * base) noexcept;
शून्य   spine_mem_pool_shared_mem_free(शून्य * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
