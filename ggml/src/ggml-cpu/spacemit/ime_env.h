#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "spine_barrier.h"
#ಅನ್ತರ್ಭಾವ "spine_mem_pool.h"

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <vector>

namespace ggml::cpu::riscv64_spacemit {

constexpr uint64_t spine_invalid_core_id    = 0xFFFFFFFF;
constexpr size_t   spine_init_barrier_ಎಣಿಕೆ = 16;

ಎಣಿಕೆ class spine_core_arch_id : uint16_t {
    core_arch_none = 0,
    core_arch_x60  = 0x503C,
    core_arch_x100 = 0x5064,
    core_arch_x200 = 0x50C8,
    core_arch_a60  = 0xA03C,
    core_arch_a100 = 0xA064,
    core_arch_a200 = 0xA0C8,
};

ರಚನೆ spine_core_info {
    uint64_t           core_id{ spine_invalid_core_id };
    spine_core_arch_id arch_id{ spine_core_arch_id::core_arch_none };

    ಸ್ಥಿರ bool get_spine_core_info(std::ಸದಿಶ<spine_core_info> & ಫಲಿತಾಂಶ);
};

ರಚನೆ spine_env_info {
    std::ಸದಿಶ<spine_core_info> core_info_list;
    std::ಸದಿಶ<ಪೂರ್ಣಾಂಕ>             perfer_core_ids;
    ಪೂರ್ಣಾಂಕ                          aicpu_id_ಸ್ಥಾನಾಂತರ{ 0 };
    ಪೂರ್ಣಾಂಕ                          ಅಂಕೆ_cores{ 0 };
    ಪೂರ್ಣಾಂಕ                          ಅಂಕೆ_perfer_cores{ 0 };
    spine_core_arch_id           perfer_core_arch_id{ spine_core_arch_id::core_arch_none };
    bool                         exclude_main_thread{ false };
    bool                         use_ime2{ false };
    bool                         use_ime1{ false };
    bool                         use_tcm{ false };
    spine_mem_pool_backend       mem_backend{ spine_mem_pool_backend::transparent_hugepage };
    uint64_t                     tcm_blk_ಗಾತ್ರ{ 0 };
    uint64_t                     cpu_ಮುಖವಾಡ{ 0 };
    spine_barrier_t *            init_barrier{ nullptr };
    bool                         init_barrier_is_shared_mem{ false };

    spine_env_info();
    ~spine_env_info();
};

ಬಾಹ್ಯ spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
