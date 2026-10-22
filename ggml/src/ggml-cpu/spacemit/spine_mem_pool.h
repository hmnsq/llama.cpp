#pragma once

#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>

namespace ggml::cpu::riscv64_spacemit {

গণনা class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

সংরচনা spine_mem_pool_tcm_info {
    bool   available{ false };
    size_t blk_size{ 0 };
    size_t blk_অঙ্ক{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
শূন্য * spine_mem_pool_tcm_mem_get(পূর্ণাঙ্ক cpu_id) noexcept;
শূন্য * spine_mem_pool_tcm_mem_wait(পূর্ণাঙ্ক cpu_id) noexcept;
পূর্ণাঙ্ক    spine_mem_pool_tcm_mem_release(পূর্ণাঙ্ক cpu_id) noexcept;

শূন্য * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept;
শূন্য * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept;
শূন্য   spine_mem_pool_free(শূন্য * base) noexcept;
শূন্য   spine_mem_pool_shared_mem_free(শূন্য * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
