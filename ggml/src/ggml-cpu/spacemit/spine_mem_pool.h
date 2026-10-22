#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

ಎಣಿಕೆ class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

ರಚನೆ spine_mem_pool_tcm_info {
    bool   ಲಭ್ಯ{ false };
    size_t blk_ಗಾತ್ರ{ 0 };
    size_t blk_ಅಂಕೆ{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
ಶೂನ್ಯ * spine_mem_pool_tcm_mem_get(ಪೂರ್ಣಾಂಕ cpu_id) noexcept;
ಶೂನ್ಯ * spine_mem_pool_tcm_mem_wait(ಪೂರ್ಣಾಂಕ cpu_id) noexcept;
ಪೂರ್ಣಾಂಕ    spine_mem_pool_tcm_mem_release(ಪೂರ್ಣಾಂಕ cpu_id) noexcept;

ಶೂನ್ಯ * spine_mem_pool_alloc(size_t ಗಾತ್ರ_2, size_t alignment) noexcept;
ಶೂನ್ಯ * spine_mem_pool_shared_mem_alloc(size_t ಗಾತ್ರ_2, size_t alignment) noexcept;
ಶೂನ್ಯ   spine_mem_pool_free(ಶೂನ್ಯ * ತಳ) noexcept;
ಶೂನ್ಯ   spine_mem_pool_shared_mem_free(ಶೂನ್ಯ * ತಳ) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
