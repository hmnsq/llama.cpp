#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "spine_barrier.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "spine_mem_pool.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

namespace ggml::cpu::riscv64_spacemit {

constexpr uint64_t spine_invalid_core_id    = 0xFFFFFFFF;
constexpr size_t   spine_init_barrier_ਗਿਣਤੀ = 16;

ਗਿਣਤੀ class spine_core_arch_id : uint16_t {
    core_arch_none = 0,
    core_arch_x60  = 0x503C,
    core_arch_x100 = 0x5064,
    core_arch_x200 = 0x50C8,
    core_arch_a60  = 0xA03C,
    core_arch_a100 = 0xA064,
    core_arch_a200 = 0xA0C8,
};

ਬਣਤਰ spine_core_info {
    uint64_t           core_id{ spine_invalid_core_id };
    spine_core_arch_id arch_id{ spine_core_arch_id::core_arch_none };

    ਸਥਿਰ bool get_spine_core_info(std::ਸਦਿਸ਼<spine_core_info> & ਨਤੀਜਾ);
};

ਬਣਤਰ spine_env_info {
    std::ਸਦਿਸ਼<spine_core_info> core_info_list;
    std::ਸਦਿਸ਼<ਪੂਰਨ>             perfer_core_ids;
    ਪੂਰਨ                          aicpu_id_ਵਿੱਥ{ 0 };
    ਪੂਰਨ                          ਅੰਕ_cores{ 0 };
    ਪੂਰਨ                          ਅੰਕ_perfer_cores{ 0 };
    spine_core_arch_id           perfer_core_arch_id{ spine_core_arch_id::core_arch_none };
    bool                         exclude_main_thread{ false };
    bool                         use_ime2{ false };
    bool                         use_ime1{ false };
    bool                         use_tcm{ false };
    spine_mem_pool_backend       mem_backend{ spine_mem_pool_backend::transparent_hugepage };
    uint64_t                     tcm_blk_ਮਾਪ{ 0 };
    uint64_t                     cpu_ਨਕਾਬ{ 0 };
    spine_barrier_t *            init_barrier{ nullptr };
    bool                         init_barrier_is_shared_mem{ false };

    spine_env_info();
    ~spine_env_info();
};

ਬਾਹਰੀ spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
