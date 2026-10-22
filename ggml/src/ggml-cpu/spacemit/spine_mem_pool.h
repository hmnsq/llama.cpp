#pragma once

#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

ଗଣନା class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

ସଂରଚନା spine_mem_pool_tcm_info {
    bool   available{ false };
    size_t blk_size{ 0 };
    size_t blk_ଅଙ୍କ{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
ଶୂନ୍ୟ * spine_mem_pool_tcm_mem_get(ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_id) noexcept;
ଶୂନ୍ୟ * spine_mem_pool_tcm_mem_wait(ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_id) noexcept;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା    spine_mem_pool_tcm_mem_release(ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpu_id) noexcept;

ଶୂନ୍ୟ * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept;
ଶୂନ୍ୟ * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept;
ଶୂନ୍ୟ   spine_mem_pool_free(ଶୂନ୍ୟ * base) noexcept;
ଶୂନ୍ୟ   spine_mem_pool_shared_mem_free(ଶୂନ୍ୟ * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
