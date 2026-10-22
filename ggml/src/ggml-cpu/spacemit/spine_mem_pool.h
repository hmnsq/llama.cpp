#pragma once

#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>

namespace ggml::cpu::riscv64_spacemit {

ગણના class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

સંરચના spine_mem_pool_tcm_info {
    bool   available{ false };
    size_t blk_size{ 0 };
    size_t blk_અંક{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
શૂન્ય * spine_mem_pool_tcm_mem_get(પૂર્ણાંક cpu_id) noexcept;
શૂન્ય * spine_mem_pool_tcm_mem_wait(પૂર્ણાંક cpu_id) noexcept;
પૂર્ણાંક    spine_mem_pool_tcm_mem_release(પૂર્ણાંક cpu_id) noexcept;

શૂન્ય * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept;
શૂન્ય * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept;
શૂન્ય   spine_mem_pool_free(શૂન્ય * base) noexcept;
શૂન્ય   spine_mem_pool_shared_mem_free(શૂન્ય * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
