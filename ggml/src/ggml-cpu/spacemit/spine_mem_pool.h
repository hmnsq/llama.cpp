#आशय once

#समावेश <cstddef>
#समावेश <cstdint>

namespace ggml::cpu::riscv64_spacemit {

क्रमागत class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

काष्ठा spine_mem_pool_tcm_info {
    bool   available{ false };
    माप_प्रकार blk_आकार{ 0 };
    माप_प्रकार blk_अंक{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
व्योम * spine_mem_pool_tcm_mem_get(पूर्णांक cpu_id) noexcept;
व्योम * spine_mem_pool_tcm_mem_wait(पूर्णांक cpu_id) noexcept;
पूर्णांक    spine_mem_pool_tcm_mem_release(पूर्णांक cpu_id) noexcept;

व्योम * spine_mem_pool_alloc(माप_प्रकार आकार, माप_प्रकार alignment) noexcept;
व्योम * spine_mem_pool_shared_mem_alloc(माप_प्रकार आकार, माप_प्रकार alignment) noexcept;
व्योम   spine_mem_pool_free(व्योम * base) noexcept;
व्योम   spine_mem_pool_shared_mem_free(व्योम * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
