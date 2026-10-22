#ہدایت once

#شامل <cstddef>
#شامل <cstdint>

namespace ggml::cpu::riscv64_spacemit {

شمار class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

ساخت spine_mem_pool_tcm_info {
    bool   دستیاب{ false };
    size_t blk_سائز{ 0 };
    size_t blk_عدد{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
خالی * spine_mem_pool_tcm_mem_get(صحیح cpu_id) noexcept;
خالی * spine_mem_pool_tcm_mem_wait(صحیح cpu_id) noexcept;
صحیح    spine_mem_pool_tcm_mem_release(صحیح cpu_id) noexcept;

خالی * spine_mem_pool_alloc(size_t سائز, size_t alignment) noexcept;
خالی * spine_mem_pool_shared_mem_alloc(size_t سائز, size_t alignment) noexcept;
خالی   spine_mem_pool_free(خالی * بنیاد) noexcept;
خالی   spine_mem_pool_shared_mem_free(خالی * بنیاد) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
