#pragma once

#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

എണ്ണൽ class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

ഘടന spine_mem_pool_tcm_info {
    bool   available{ false };
    size_t blk_size{ 0 };
    size_t blk_അക്കം{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
ശൂന്യം * spine_mem_pool_tcm_mem_get(പൂർണ്ണസംഖ്യ cpu_id) noexcept;
ശൂന്യം * spine_mem_pool_tcm_mem_wait(പൂർണ്ണസംഖ്യ cpu_id) noexcept;
പൂർണ്ണസംഖ്യ    spine_mem_pool_tcm_mem_release(പൂർണ്ണസംഖ്യ cpu_id) noexcept;

ശൂന്യം * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept;
ശൂന്യം * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept;
ശൂന്യം   spine_mem_pool_free(ശൂന്യം * base) noexcept;
ശൂന്യം   spine_mem_pool_shared_mem_free(ശൂന്യം * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
