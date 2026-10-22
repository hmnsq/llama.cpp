#pragma once

#شامل <cstddef>
#شامل <cstdint>

namespace ggml::cpu::riscv64_spacemit {

شمارش class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

ساختار spine_mem_pool_tcm_info {
    bool   available{ false };
    size_t blk_size{ 0 };
    size_t blk_رقم{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
تهی * spine_mem_pool_tcm_mem_get(صحیح cpu_id) noexcept;
تهی * spine_mem_pool_tcm_mem_wait(صحیح cpu_id) noexcept;
صحیح    spine_mem_pool_tcm_mem_release(صحیح cpu_id) noexcept;

تهی * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept;
تهی * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept;
تهی   spine_mem_pool_free(تهی * base) noexcept;
تهی   spine_mem_pool_shared_mem_free(تهی * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
