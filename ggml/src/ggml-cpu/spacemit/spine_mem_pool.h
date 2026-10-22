#رهنمود once

#ضمن <cstddef>
#ضمن <cstdint>

namespace ggml::cpu::riscv64_spacemit {

تعداد class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

بنية spine_mem_pool_tcm_info {
    bool   موجود{ false };
    size_t blk_اندازه{ 0 };
    size_t blk_رقم{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
فراغ * spine_mem_pool_tcm_mem_get(صحيح cpu_id) noexcept;
فراغ * spine_mem_pool_tcm_mem_wait(صحيح cpu_id) noexcept;
صحيح    spine_mem_pool_tcm_mem_release(صحيح cpu_id) noexcept;

فراغ * spine_mem_pool_alloc(size_t اندازه, size_t alignment) noexcept;
فراغ * spine_mem_pool_shared_mem_alloc(size_t اندازه, size_t alignment) noexcept;
فراغ   spine_mem_pool_free(فراغ * پایه) noexcept;
فراغ   spine_mem_pool_shared_mem_free(فراغ * پایه) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
