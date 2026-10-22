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
ತಾರ್ಕಿಕ spine_core_info::get_spine_core_info(std::vector<spine_core_info> & result) {
    ಸ್ಥಿರ std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    result.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   ಸಾಲು;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    ತಾರ್ಕಿಕ     has_processor     = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ     has_marchid       = ಅಸತ್ಯ;

    ಯದಿ (!file.is_open()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯಾವತ್ (std::getline(file, ಸಾಲು)) {
        ಯದಿ (ಸಾಲು.substr(0, 9) == "processor") {
            ಯದಿ (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = ಸಾಲು.find(':');
            ಯದಿ (colon_pos != std::string::npos) {
                current_processor = std::stoi(ಸಾಲು.substr(colon_pos + 1));
                has_processor     = ಸತ್ಯ;
            }

            has_marchid = ಅಸತ್ಯ;
        } ಅನ್ಯಥಾ ಯದಿ (ಸಾಲು.substr(0, 7) == "marchid") {
            size_t colon_pos = ಸಾಲು.find(':');
            ಯದಿ (colon_pos != std::string::npos) {
                std::string marchid_str = ಸಾಲು.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), isspace), marchid_str.end());
                current_marchid = std::stoull(marchid_str, nullptr, 16);
                has_marchid     = ಸತ್ಯ;
            }
        }
    }

    ಯದಿ (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    ಯದಿ (has_processor && has_marchid) {
        ಪ್ರತಿ (auto & cpu_info : cpu_info_list) {
            ಯದಿ (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                auto core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                result.push_back(core_info);
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ has_processor && has_marchid;
}

ನಾಮಾಕಾಶ {
uint16_t hex_string_to_u16(ಧ್ರುವ std::string & hex_str) {
    ಯತ್ನ {
        size_t pos = 0;
        ಯದಿ (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        unsigned long result = std::stoul(hex_str.substr(pos), nullptr, 16);
        ಯದಿ (result > std::numeric_limits<uint16_t>::max()) {
            ಕ್ಷಿಪ std::out_of_range("Converted value is out of range for uint16_t");
        }
        ಪ್ರತ್ಯಾಗಮ static_cast<uint16_t>(result);
    } ಗೃಹಾಣ (ಧ್ರುವ std::invalid_argument & e) {
        ಕ್ಷಿಪ std::invalid_argument("Invalid hexadecimal string");
    } ಗೃಹಾಣ (ಧ್ರುವ std::out_of_range & e) {
        ಕ್ಷಿಪ;
    }
}

ಧ್ರುವ ವರ್ಣ * spine_mem_pool_backend_to_string(spine_mem_pool_backend backend) {
    ವಿಕಲ್ಪ (backend) {
        ಪಕ್ಷ spine_mem_pool_backend::none:
            ಪ್ರತ್ಯಾಗಮ "NONE";
        ಪಕ್ಷ spine_mem_pool_backend::posix_memalign:
            ಪ್ರತ್ಯಾಗಮ "POSIX";
        ಪಕ್ಷ spine_mem_pool_backend::transparent_hugepage:
            ಪ್ರತ್ಯಾಗಮ "HPAGE";
        ಪಕ್ಷ spine_mem_pool_backend::hugetlb_1g:
            ಪ್ರತ್ಯಾಗಮ "HPAGE1GB";
    }

    ಪ್ರತ್ಯಾಗಮ "unknown";
}

spine_mem_pool_backend parse_mem_backend(ಧ್ರುವ ವರ್ಣ * mem_backend_str) {
    ಯದಿ (mem_backend_str == nullptr || mem_backend_str[0] == '\0') {
        ಪ್ರತ್ಯಾಗಮ spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned ವರ್ಣ ch) { ಪ್ರತ್ಯಾಗಮ static_cast<ವರ್ಣ>(std::tolower(ch)); });

    ಯದಿ (value == "none") {
        ಪ್ರತ್ಯಾಗಮ spine_mem_pool_backend::none;
    }

    ಯದಿ (value == "posix") {
        ಪ್ರತ್ಯಾಗಮ spine_mem_pool_backend::posix_memalign;
    }

    ಯದಿ (value == "hpage") {
        ಪ್ರತ್ಯಾಗಮ spine_mem_pool_backend::transparent_hugepage;
    }

    ಯದಿ (value == "hpage1gb") {
        ಪ್ರತ್ಯಾಗಮ spine_mem_pool_backend::hugetlb_1g;
    }

    ಕ್ಷಿಪ std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    ಅಂಕೆ_cores = static_cast<ಪೂರ್ಣಾಙ್ಕ>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    ಯದಿ (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    ಯದಿ (core_info_list.size() == 0) {
        ವರ್ಣ * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        ಯದಿ (spine_core_arch_str != nullptr) {
            auto arch_id = hex_string_to_u16(spine_core_arch_str);
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_cores; i++) {
                auto core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    ಯದಿ (core_info_list.size() == 0) {
        ಕ್ಷಿಪ std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    ವರ್ಣ * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    ಯದಿ (spine_perfer_core_arch_str != nullptr && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_to_u16(spine_perfer_core_arch_str) };
    }

    ವರ್ಣ *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<ಪೂರ್ಣಾಙ್ಕ> perfer_core_id_vec;
    ಯದಿ (spine_perfer_core_id_str != nullptr && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        ಯಾವತ್ ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(ಅಂಕೆ_cores);
    ಯದಿ (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        ಪ್ರತಿ (auto & core_info : core_info_list) {
            auto core_arch_id   = core_info.arch_id;
            auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
            ಯದಿ (core_arch_head == 0xA) {
                ಅಂಕೆ_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_mask |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿ (auto & core_info : core_info_list) {
            auto core_arch_id = core_info.arch_id;
            ಯದಿ (core_arch_id == perfer_core_arch_id) {
                ಅಂಕೆ_perfer_cores++;
                cpu_mask |= (1ULL << core_info.core_id);

                auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
                ಯದಿ (core_arch_head == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        ಯದಿ (ಅಂಕೆ_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    ಯದಿ (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_mask         = 0;
        ಅಂಕೆ_perfer_cores = 0;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ core_id : perfer_core_id_vec) {
            ಯದಿ (core_id < 0 || core_id >= ಅಂಕೆ_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           ಅಂಕೆ_cores - 1);
            }
            auto core_info    = core_info_list[core_id];
            auto core_arch_id = core_info.arch_id;
            ಯದಿ (core_arch_id == perfer_core_arch_id) {
                cpu_mask |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } ಅನ್ಯಥಾ {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        ಅಂಕೆ_perfer_cores = static_cast<ಪೂರ್ಣಾಙ್ಕ>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    ವರ್ಣ * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    auto   user_disable_tcm      = spine_disable_tcm_str != nullptr && strcmp(spine_disable_tcm_str, "0") != 0;

    ಯದಿ (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        ಯದಿ (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.available;
            tcm_blk_size = tcm_info.blk_size;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_size, tcm_info.blk_ಅಂಕೆ, tcm_info.is_fake_tcm);

            ಪ್ರತಿ (auto & core_info : core_info_list) {
                auto core_arch_head = (uint16_t) (core_info.arch_id) >> 12;
                ಯದಿ (core_arch_head != 0xA) {
                    aicpu_id_offset++;
                } ಅನ್ಯಥಾ {
                    ಭಙ್ಗ;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        ಅಂಕೆ_cores, ಅಂಕೆ_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_to_string(mem_backend), cpu_mask, aicpu_id_offset);

    ಧ್ರುವ size_t init_barrier_size = ಪರಿಮಾಣ(spine_barrier_t) * spine_init_barrier_ಎಣಿಕೆ;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_size, alignof(spine_barrier_t)));
    ಯದಿ (init_barrier != nullptr) {
        init_barrier_is_shared_mem = ಸತ್ಯ;
    } ಅನ್ಯಥಾ {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = ನೂತನ spine_barrier_t[spine_init_barrier_ಎಣಿಕೆ];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_ಎಣಿಕೆ, 2);
}

spine_env_info::~spine_env_info() {
    ಯದಿ (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } ಅನ್ಯಥಾ {
        delete[] init_barrier;
    }

    init_barrier               = nullptr;
    init_barrier_is_shared_mem = ಅಸತ್ಯ;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
