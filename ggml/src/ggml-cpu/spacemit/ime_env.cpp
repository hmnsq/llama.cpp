#അന്തര്ഭാവ "ime_env.h"

#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "spine_mem_pool.h"

#അന്തര്ഭാവ <fcntl.h>
#അന്തര്ഭാവ <unistd.h>

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <array>
#അന്തര്ഭാവ <cctype>
#അന്തര്ഭാവ <fstream>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <thread>
#അന്തര്ഭാവ <unordered_map>

നാമാകാശ ggml::cpu::riscv64_spacemit {
തർക്കം spine_core_info::get_spine_core_info(std::vector<spine_core_info> & result) {
    സ്ഥിതം std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    result.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   വരി;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    തർക്കം     has_processor     = അസത്യം;
    തർക്കം     has_marchid       = അസത്യം;

    എങ്കിൽ (!file.is_open()) {
        തിരികെ അസത്യം;
    }

    ആകുമ്പോൾ (std::getline(file, വരി)) {
        എങ്കിൽ (വരി.substr(0, 9) == "processor") {
            എങ്കിൽ (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = വരി.find(':');
            എങ്കിൽ (colon_pos != std::string::npos) {
                current_processor = std::stoi(വരി.substr(colon_pos + 1));
                has_processor     = സത്യം;
            }

            has_marchid = അസത്യം;
        } അല്ലെങ്കിൽ എങ്കിൽ (വരി.substr(0, 7) == "marchid") {
            size_t colon_pos = വരി.find(':');
            എങ്കിൽ (colon_pos != std::string::npos) {
                std::string marchid_str = വരി.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), isspace), marchid_str.end());
                current_marchid = std::stoull(marchid_str, nullptr, 16);
                has_marchid     = സത്യം;
            }
        }
    }

    എങ്കിൽ (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    എങ്കിൽ (has_processor && has_marchid) {
        വേണ്ടി (auto & cpu_info : cpu_info_list) {
            എങ്കിൽ (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                auto core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                result.push_back(core_info);
            }
        }
    }

    തിരികെ has_processor && has_marchid;
}

നാമാകാശ {
uint16_t hex_string_to_u16(സ്ഥിരം std::string & hex_str) {
    ശ്രമിക്കുക {
        size_t pos = 0;
        എങ്കിൽ (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        unsigned long result = std::stoul(hex_str.substr(pos), nullptr, 16);
        എങ്കിൽ (result > std::numeric_limits<uint16_t>::max()) {
            എറിയുക std::out_of_range("Converted value is out of range for uint16_t");
        }
        തിരികെ static_cast<uint16_t>(result);
    } പിടിക്കുക (സ്ഥിരം std::invalid_argument & e) {
        എറിയുക std::invalid_argument("Invalid hexadecimal string");
    } പിടിക്കുക (സ്ഥിരം std::out_of_range & e) {
        എറിയുക;
    }
}

സ്ഥിരം അക്ഷരം * spine_mem_pool_backend_to_string(spine_mem_pool_backend backend) {
    തിരഞ്ഞെടുപ്പ് (backend) {
        അവസ്ഥ spine_mem_pool_backend::none:
            തിരികെ "NONE";
        അവസ്ഥ spine_mem_pool_backend::posix_memalign:
            തിരികെ "POSIX";
        അവസ്ഥ spine_mem_pool_backend::transparent_hugepage:
            തിരികെ "HPAGE";
        അവസ്ഥ spine_mem_pool_backend::hugetlb_1g:
            തിരികെ "HPAGE1GB";
    }

    തിരികെ "unknown";
}

spine_mem_pool_backend parse_mem_backend(സ്ഥിരം അക്ഷരം * mem_backend_str) {
    എങ്കിൽ (mem_backend_str == nullptr || mem_backend_str[0] == '\0') {
        തിരികെ spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned അക്ഷരം ch) { തിരികെ static_cast<അക്ഷരം>(std::tolower(ch)); });

    എങ്കിൽ (value == "none") {
        തിരികെ spine_mem_pool_backend::none;
    }

    എങ്കിൽ (value == "posix") {
        തിരികെ spine_mem_pool_backend::posix_memalign;
    }

    എങ്കിൽ (value == "hpage") {
        തിരികെ spine_mem_pool_backend::transparent_hugepage;
    }

    എങ്കിൽ (value == "hpage1gb") {
        തിരികെ spine_mem_pool_backend::hugetlb_1g;
    }

    എറിയുക std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    അക്കം_cores = static_cast<പൂർണ്ണസംഖ്യ>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    എങ്കിൽ (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    എങ്കിൽ (core_info_list.size() == 0) {
        അക്ഷരം * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        എങ്കിൽ (spine_core_arch_str != nullptr) {
            auto arch_id = hex_string_to_u16(spine_core_arch_str);
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < അക്കം_cores; i++) {
                auto core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    എങ്കിൽ (core_info_list.size() == 0) {
        എറിയുക std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    അക്ഷരം * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    എങ്കിൽ (spine_perfer_core_arch_str != nullptr && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_to_u16(spine_perfer_core_arch_str) };
    }

    അക്ഷരം *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<പൂർണ്ണസംഖ്യ> perfer_core_id_vec;
    എങ്കിൽ (spine_perfer_core_id_str != nullptr && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        ആകുമ്പോൾ ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(അക്കം_cores);
    എങ്കിൽ (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        വേണ്ടി (auto & core_info : core_info_list) {
            auto core_arch_id   = core_info.arch_id;
            auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
            എങ്കിൽ (core_arch_head == 0xA) {
                അക്കം_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_mask |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } അല്ലെങ്കിൽ {
        വേണ്ടി (auto & core_info : core_info_list) {
            auto core_arch_id = core_info.arch_id;
            എങ്കിൽ (core_arch_id == perfer_core_arch_id) {
                അക്കം_perfer_cores++;
                cpu_mask |= (1ULL << core_info.core_id);

                auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
                എങ്കിൽ (core_arch_head == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        എങ്കിൽ (അക്കം_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    എങ്കിൽ (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_mask         = 0;
        അക്കം_perfer_cores = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ core_id : perfer_core_id_vec) {
            എങ്കിൽ (core_id < 0 || core_id >= അക്കം_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           അക്കം_cores - 1);
            }
            auto core_info    = core_info_list[core_id];
            auto core_arch_id = core_info.arch_id;
            എങ്കിൽ (core_arch_id == perfer_core_arch_id) {
                cpu_mask |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } അല്ലെങ്കിൽ {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        വേണ്ടി (പൂർണ്ണസംഖ്യ core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        അക്കം_perfer_cores = static_cast<പൂർണ്ണസംഖ്യ>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    അക്ഷരം * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    auto   user_disable_tcm      = spine_disable_tcm_str != nullptr && strcmp(spine_disable_tcm_str, "0") != 0;

    എങ്കിൽ (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        എങ്കിൽ (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.available;
            tcm_blk_size = tcm_info.blk_size;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_size, tcm_info.blk_അക്കം, tcm_info.is_fake_tcm);

            വേണ്ടി (auto & core_info : core_info_list) {
                auto core_arch_head = (uint16_t) (core_info.arch_id) >> 12;
                എങ്കിൽ (core_arch_head != 0xA) {
                    aicpu_id_offset++;
                } അല്ലെങ്കിൽ {
                    നിർത്തുക;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        അക്കം_cores, അക്കം_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_to_string(mem_backend), cpu_mask, aicpu_id_offset);

    സ്ഥിരം size_t init_barrier_size = വലുപ്പം(spine_barrier_t) * spine_init_barrier_എണ്ണം;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_size, alignof(spine_barrier_t)));
    എങ്കിൽ (init_barrier != nullptr) {
        init_barrier_is_shared_mem = സത്യം;
    } അല്ലെങ്കിൽ {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = പുതിയ spine_barrier_t[spine_init_barrier_എണ്ണം];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_എണ്ണം, 2);
}

spine_env_info::~spine_env_info() {
    എങ്കിൽ (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } അല്ലെങ്കിൽ {
        delete[] init_barrier;
    }

    init_barrier               = nullptr;
    init_barrier_is_shared_mem = അസത്യം;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
