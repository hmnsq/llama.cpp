#ಅನ್ತರ್ಭಾವ "ime_env.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "spine_mem_pool.h"

#ಅನ್ತರ್ಭಾವ <fcntl.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <cctype>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <thread>
#ಅನ್ತರ್ಭಾವ <unordered_map>

ನಾಮಾಕಾಶ ggml::cpu::riscv64_spacemit {
ತಾರ್ಕಿಕ spine_core_info::get_spine_core_info(std::vector<spine_core_info> & ಫಲಿತಾಂಶ) {
    ಸ್ಥಿರ std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    ಫಲಿತಾಂಶ.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   ಸಾಲು;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    ತಾರ್ಕಿಕ     has_processor     = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ     has_marchid       = ಸುಳ್ಳು;

    ಆದರೆ (!file.is_open()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆಗ (std::getline(file, ಸಾಲು)) {
        ಆದರೆ (ಸಾಲು.substr(0, 9) == "processor") {
            ಆದರೆ (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = ಸಾಲು.find(':');
            ಆದರೆ (colon_pos != std::string::npos) {
                current_processor = std::stoi(ಸಾಲು.substr(colon_pos + 1));
                has_processor     = ನಿಜ;
            }

            has_marchid = ಸುಳ್ಳು;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಸಾಲು.substr(0, 7) == "marchid") {
            size_t colon_pos = ಸಾಲು.find(':');
            ಆದರೆ (colon_pos != std::string::npos) {
                std::string marchid_str = ಸಾಲು.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), ಖಾಲಿಯೇ), marchid_str.end());
                current_marchid = std::stoull(marchid_str, ಶೂನ್ಯಸೂಚಕ, 16);
                has_marchid     = ನಿಜ;
            }
        }
    }

    ಆದರೆ (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    ಆದರೆ (has_processor && has_marchid) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & cpu_info : cpu_info_list) {
            ಆದರೆ (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                ಸ್ವಯಂಚಾಲಿತ core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                ಫಲಿತಾಂಶ.push_back(core_info);
            }
        }
    }

    ಹಿಂತಿರುಗಿ has_processor && has_marchid;
}

ನಾಮಾಕಾಶ {
uint16_t hex_string_ಗಮ್ಯ_u16(ಸ್ಥಿರಾಂಕ std::string & hex_str) {
    ಪ್ರಯತ್ನಿಸು {
        size_t pos = 0;
        ಆದರೆ (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ಫಲಿತಾಂಶ = std::stoul(hex_str.substr(pos), ಶೂನ್ಯಸೂಚಕ, 16);
        ಆದರೆ (ಫಲಿತಾಂಶ > std::numeric_limits<uint16_t>::max()) {
            ಎಸೆ std::out_of_range("Converted value is out of range for uint16_t");
        }
        ಹಿಂತಿರುಗಿ static_cast<uint16_t>(ಫಲಿತಾಂಶ);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::invalid_argument & e) {
        ಎಸೆ std::invalid_argument("Invalid hexadecimal string");
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::out_of_range & e) {
        ಎಸೆ;
    }
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * spine_mem_pool_backend_ಗಮ್ಯ_string(spine_mem_pool_backend backend) {
    ಆಯ್ಕೆ (backend) {
        ಸ್ಥಿತಿ spine_mem_pool_backend::none:
            ಹಿಂತಿರುಗಿ "NONE";
        ಸ್ಥಿತಿ spine_mem_pool_backend::posix_memalign:
            ಹಿಂತಿರುಗಿ "POSIX";
        ಸ್ಥಿತಿ spine_mem_pool_backend::transparent_hugepage:
            ಹಿಂತಿರುಗಿ "HPAGE";
        ಸ್ಥಿತಿ spine_mem_pool_backend::hugetlb_1g:
            ಹಿಂತಿರುಗಿ "HPAGE1GB";
    }

    ಹಿಂತಿರುಗಿ "unknown";
}

spine_mem_pool_backend parse_mem_backend(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * mem_backend_str) {
    ಆದರೆ (mem_backend_str == ಶೂನ್ಯಸೂಚಕ || mem_backend_str[0] == '\0') {
        ಹಿಂತಿರುಗಿ spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ ch) { ಹಿಂತಿರುಗಿ static_cast<ಅಕ್ಷರ>(std::tolower(ch)); });

    ಆದರೆ (value == "none") {
        ಹಿಂತಿರುಗಿ spine_mem_pool_backend::none;
    }

    ಆದರೆ (value == "posix") {
        ಹಿಂತಿರುಗಿ spine_mem_pool_backend::posix_memalign;
    }

    ಆದರೆ (value == "hpage") {
        ಹಿಂತಿರುಗಿ spine_mem_pool_backend::transparent_hugepage;
    }

    ಆದರೆ (value == "hpage1gb") {
        ಹಿಂತಿರುಗಿ spine_mem_pool_backend::hugetlb_1g;
    }

    ಎಸೆ std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    ಅಂಕೆ_cores = static_cast<ಪೂರ್ಣಾಂಕ>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    ಆದರೆ (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    ಆದರೆ (core_info_list.size() == 0) {
        ಅಕ್ಷರ * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        ಆದರೆ (spine_core_arch_str != ಶೂನ್ಯಸೂಚಕ) {
            ಸ್ವಯಂಚಾಲಿತ arch_id = hex_string_ಗಮ್ಯ_u16(spine_core_arch_str);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_cores; i++) {
                ಸ್ವಯಂಚಾಲಿತ core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    ಆದರೆ (core_info_list.size() == 0) {
        ಎಸೆ std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    ಅಕ್ಷರ * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    ಆದರೆ (spine_perfer_core_arch_str != ಶೂನ್ಯಸೂಚಕ && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_ಗಮ್ಯ_u16(spine_perfer_core_arch_str) };
    }

    ಅಕ್ಷರ *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<ಪೂರ್ಣಾಂಕ> perfer_core_id_vec;
    ಆದರೆ (spine_perfer_core_id_str != ಶೂನ್ಯಸೂಚಕ && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        ಆಗ ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(ಅಂಕೆ_cores);
    ಆದರೆ (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & core_info : core_info_list) {
            ಸ್ವಯಂಚಾಲಿತ core_arch_id   = core_info.arch_id;
            ಸ್ವಯಂಚಾಲಿತ core_arch_ತಲೆ = (uint16_t) (core_arch_id) >> 12;
            ಆದರೆ (core_arch_ತಲೆ == 0xA) {
                ಅಂಕೆ_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_ಮುಖವಾಡ |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & core_info : core_info_list) {
            ಸ್ವಯಂಚಾಲಿತ core_arch_id = core_info.arch_id;
            ಆದರೆ (core_arch_id == perfer_core_arch_id) {
                ಅಂಕೆ_perfer_cores++;
                cpu_ಮುಖವಾಡ |= (1ULL << core_info.core_id);

                ಸ್ವಯಂಚಾಲಿತ core_arch_ತಲೆ = (uint16_t) (core_arch_id) >> 12;
                ಆದರೆ (core_arch_ತಲೆ == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        ಆದರೆ (ಅಂಕೆ_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    ಆದರೆ (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_ಮುಖವಾಡ         = 0;
        ಅಂಕೆ_perfer_cores = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ core_id : perfer_core_id_vec) {
            ಆದರೆ (core_id < 0 || core_id >= ಅಂಕೆ_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           ಅಂಕೆ_cores - 1);
            }
            ಸ್ವಯಂಚಾಲಿತ core_info    = core_info_list[core_id];
            ಸ್ವಯಂಚಾಲಿತ core_arch_id = core_info.arch_id;
            ಆದರೆ (core_arch_id == perfer_core_arch_id) {
                cpu_ಮುಖವಾಡ |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        ಅಂಕೆ_perfer_cores = static_cast<ಪೂರ್ಣಾಂಕ>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    ಅಕ್ಷರ * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    ಸ್ವಯಂಚಾಲಿತ   user_disable_tcm      = spine_disable_tcm_str != ಶೂನ್ಯಸೂಚಕ && strcmp(spine_disable_tcm_str, "0") != 0;

    ಆದರೆ (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        ಆದರೆ (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.ಲಭ್ಯ;
            tcm_blk_ಗಾತ್ರ = tcm_info.blk_ಗಾತ್ರ;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_ಗಾತ್ರ, tcm_info.blk_ಅಂಕೆ, tcm_info.is_fake_tcm);

            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & core_info : core_info_list) {
                ಸ್ವಯಂಚಾಲಿತ core_arch_ತಲೆ = (uint16_t) (core_info.arch_id) >> 12;
                ಆದರೆ (core_arch_ತಲೆ != 0xA) {
                    aicpu_id_ಸ್ಥಾನಾಂತರ++;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ನಿಲ್ಲಿಸು;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        ಅಂಕೆ_cores, ಅಂಕೆ_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_ಗಮ್ಯ_string(mem_backend), cpu_ಮುಖವಾಡ, aicpu_id_ಸ್ಥಾನಾಂತರ);

    ಸ್ಥಿರಾಂಕ size_t init_barrier_ಗಾತ್ರ = ಗಾತ್ರ(spine_barrier_t) * spine_init_barrier_ಎಣಿಕೆ;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_ಗಾತ್ರ, alignof(spine_barrier_t)));
    ಆದರೆ (init_barrier != ಶೂನ್ಯಸೂಚಕ) {
        init_barrier_is_shared_mem = ನಿಜ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = ಹೊಸ spine_barrier_t[spine_init_barrier_ಎಣಿಕೆ];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_ಎಣಿಕೆ, 2);
}

spine_env_info::~spine_env_info() {
    ಆದರೆ (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಅಳಿಸು[] init_barrier;
    }

    init_barrier               = ಶೂನ್ಯಸೂಚಕ;
    init_barrier_is_shared_mem = ಸುಳ್ಳು;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
