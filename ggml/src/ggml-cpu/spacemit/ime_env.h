#pragma once

#அந்தர்பாவ "spine_barrier.h"
#அந்தர்பாவ "spine_mem_pool.h"

#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <vector>

namespace ggml::cpu::riscv64_spacemit {

constexpr uint64_t spine_invalid_core_id    = 0xFFFFFFFF;
constexpr size_t   spine_init_barrier_மொத்தம் = 16;

எண்ணிக்கை class spine_core_arch_id : uint16_t {
    core_arch_none = 0,
    core_arch_x60  = 0x503C,
    core_arch_x100 = 0x5064,
    core_arch_x200 = 0x50C8,
    core_arch_a60  = 0xA03C,
    core_arch_a100 = 0xA064,
    core_arch_a200 = 0xA0C8,
};

கட்டமைப்பு spine_core_info {
    uint64_t           core_id{ spine_invalid_core_id };
    spine_core_arch_id arch_id{ spine_core_arch_id::core_arch_none };

    நிலையான bool get_spine_core_info(std::vector<spine_core_info> & result);
};

கட்டமைப்பு spine_env_info {
    std::vector<spine_core_info> core_info_list;
    std::vector<முழு>             perfer_core_ids;
    முழு                          aicpu_id_offset{ 0 };
    முழு                          இலக்கம்_cores{ 0 };
    முழு                          இலக்கம்_perfer_cores{ 0 };
    spine_core_arch_id           perfer_core_arch_id{ spine_core_arch_id::core_arch_none };
    bool                         exclude_main_thread{ false };
    bool                         use_ime2{ false };
    bool                         use_ime1{ false };
    bool                         use_tcm{ false };
    spine_mem_pool_backend       mem_backend{ spine_mem_pool_backend::transparent_hugepage };
    uint64_t                     tcm_blk_size{ 0 };
    uint64_t                     cpu_mask{ 0 };
    spine_barrier_t *            init_barrier{ nullptr };
    bool                         init_barrier_is_shared_mem{ false };

    spine_env_info();
    ~spine_env_info();
};

வெளி spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
