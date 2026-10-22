#pragma once

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

ಗಣನಾ class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

ಸಂರಚನಾ spine_mem_pool_tcm_info {
    bool   available{ false };
    size_t blk_size{ 0 };
    size_t blk_ಅಂಕೆ{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
ಶೂನ್ಯ * spine_mem_pool_tcm_mem_get(ಪೂರ್ಣಾಙ್ಕ cpu_id) noexcept;
ಶೂನ್ಯ * spine_mem_pool_tcm_mem_wait(ಪೂರ್ಣಾಙ್ಕ cpu_id) noexcept;
ಪೂರ್ಣಾಙ್ಕ    spine_mem_pool_tcm_mem_release(ಪೂರ್ಣಾಙ್ಕ cpu_id) noexcept;

ಶೂನ್ಯ * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept;
ಶೂನ್ಯ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept;
ಶೂನ್ಯ   spine_mem_pool_free(ಶೂನ್ಯ * base) noexcept;
ಶೂನ್ಯ   spine_mem_pool_shared_mem_free(ಶೂನ್ಯ * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
