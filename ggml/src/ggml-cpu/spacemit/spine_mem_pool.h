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
    माप_प्रकार blk_size{ 0 };
    माप_प्रकार blk_अंक{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
शून्य * spine_mem_pool_tcm_mem_get(पूर्णांक cpu_id) noexcept;
शून्य * spine_mem_pool_tcm_mem_wait(पूर्णांक cpu_id) noexcept;
पूर्णांक    spine_mem_pool_tcm_mem_release(पूर्णांक cpu_id) noexcept;

शून्य * spine_mem_pool_alloc(माप_प्रकार size, माप_प्रकार alignment) noexcept;
शून्य * spine_mem_pool_shared_mem_alloc(माप_प्रकार size, माप_प्रकार alignment) noexcept;
शून्य   spine_mem_pool_free(शून्य * base) noexcept;
शून्य   spine_mem_pool_shared_mem_free(शून्य * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
