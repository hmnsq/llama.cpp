#ਅਨ੍ਤਰ੍ਭਾਵ "ime_env.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "spine_mem_pool.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <fcntl.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <array>
#ਅਨ੍ਤਰ੍ਭਾਵ <cctype>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <thread>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>

ਨਾਮਾਕਾਸ਼ ggml::cpu::riscv64_spacemit {
ਤਾਰਕਿਕ spine_core_info::get_spine_core_info(std::vector<spine_core_info> & ਨਤੀਜਾ) {
    ਸਥਿਰ std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    ਨਤੀਜਾ.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   ਸਤਰ;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    ਤਾਰਕਿਕ     has_processor     = ਝੂਠ;
    ਤਾਰਕਿਕ     has_marchid       = ਝੂਠ;

    ਜੇ (!file.is_open()) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜਦੋਂ (std::getline(file, ਸਤਰ)) {
        ਜੇ (ਸਤਰ.substr(0, 9) == "processor") {
            ਜੇ (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = ਸਤਰ.find(':');
            ਜੇ (colon_pos != std::string::npos) {
                current_processor = std::stoi(ਸਤਰ.substr(colon_pos + 1));
                has_processor     = ਸੱਚ;
            }

            has_marchid = ਝੂਠ;
        } ਨਹੀਂਤਾਂ ਜੇ (ਸਤਰ.substr(0, 7) == "marchid") {
            size_t colon_pos = ਸਤਰ.find(':');
            ਜੇ (colon_pos != std::string::npos) {
                std::string marchid_str = ਸਤਰ.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), ਖਾਲੀ_ਥਾਂ_ਹੈ), marchid_str.end());
                current_marchid = std::stoull(marchid_str, ਖਾਲੀ_ਸੂਚਕ, 16);
                has_marchid     = ਸੱਚ;
            }
        }
    }

    ਜੇ (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    ਜੇ (has_processor && has_marchid) {
        ਲਈ (ਸਵੈਚਾਲਿਤ & cpu_info : cpu_info_list) {
            ਜੇ (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                ਸਵੈਚਾਲਿਤ core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                ਨਤੀਜਾ.push_back(core_info);
            }
        }
    }

    ਵਾਪਸ has_processor && has_marchid;
}

ਨਾਮਾਕਾਸ਼ {
uint16_t hex_string_ਮੰਜ਼ਿਲ_u16(ਅਚਲ std::string & hex_str) {
    ਕੋਸ਼ਿਸ਼ {
        size_t pos = 0;
        ਜੇ (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਨਤੀਜਾ = std::stoul(hex_str.substr(pos), ਖਾਲੀ_ਸੂਚਕ, 16);
        ਜੇ (ਨਤੀਜਾ > std::numeric_limits<uint16_t>::max()) {
            ਸੁੱਟੋ std::out_of_range("Converted value is out of range for uint16_t");
        }
        ਵਾਪਸ static_cast<uint16_t>(ਨਤੀਜਾ);
    } ਫੜੋ (ਅਚਲ std::invalid_argument & e) {
        ਸੁੱਟੋ std::invalid_argument("Invalid hexadecimal string");
    } ਫੜੋ (ਅਚਲ std::out_of_range & e) {
        ਸੁੱਟੋ;
    }
}

ਅਚਲ ਅੱਖਰ * spine_mem_pool_backend_ਮੰਜ਼ਿਲ_string(spine_mem_pool_backend backend) {
    ਚੋਣ (backend) {
        ਹਾਲਤ spine_mem_pool_backend::none:
            ਵਾਪਸ "NONE";
        ਹਾਲਤ spine_mem_pool_backend::posix_memalign:
            ਵਾਪਸ "POSIX";
        ਹਾਲਤ spine_mem_pool_backend::transparent_hugepage:
            ਵਾਪਸ "HPAGE";
        ਹਾਲਤ spine_mem_pool_backend::hugetlb_1g:
            ਵਾਪਸ "HPAGE1GB";
    }

    ਵਾਪਸ "unknown";
}

spine_mem_pool_backend parse_mem_backend(ਅਚਲ ਅੱਖਰ * mem_backend_str) {
    ਜੇ (mem_backend_str == ਖਾਲੀ_ਸੂਚਕ || mem_backend_str[0] == '\0') {
        ਵਾਪਸ spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ ch) { ਵਾਪਸ static_cast<ਅੱਖਰ>(std::tolower(ch)); });

    ਜੇ (value == "none") {
        ਵਾਪਸ spine_mem_pool_backend::none;
    }

    ਜੇ (value == "posix") {
        ਵਾਪਸ spine_mem_pool_backend::posix_memalign;
    }

    ਜੇ (value == "hpage") {
        ਵਾਪਸ spine_mem_pool_backend::transparent_hugepage;
    }

    ਜੇ (value == "hpage1gb") {
        ਵਾਪਸ spine_mem_pool_backend::hugetlb_1g;
    }

    ਸੁੱਟੋ std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    ਅੰਕ_cores = static_cast<ਪੂਰਨ>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    ਜੇ (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    ਜੇ (core_info_list.size() == 0) {
        ਅੱਖਰ * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        ਜੇ (spine_core_arch_str != ਖਾਲੀ_ਸੂਚਕ) {
            ਸਵੈਚਾਲਿਤ arch_id = hex_string_ਮੰਜ਼ਿਲ_u16(spine_core_arch_str);
            ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_cores; i++) {
                ਸਵੈਚਾਲਿਤ core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    ਜੇ (core_info_list.size() == 0) {
        ਸੁੱਟੋ std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    ਅੱਖਰ * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    ਜੇ (spine_perfer_core_arch_str != ਖਾਲੀ_ਸੂਚਕ && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_ਮੰਜ਼ਿਲ_u16(spine_perfer_core_arch_str) };
    }

    ਅੱਖਰ *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<ਪੂਰਨ> perfer_core_id_vec;
    ਜੇ (spine_perfer_core_id_str != ਖਾਲੀ_ਸੂਚਕ && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        ਜਦੋਂ ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(ਅੰਕ_cores);
    ਜੇ (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        ਲਈ (ਸਵੈਚਾਲਿਤ & core_info : core_info_list) {
            ਸਵੈਚਾਲਿਤ core_arch_id   = core_info.arch_id;
            ਸਵੈਚਾਲਿਤ core_arch_ਸਿਰ = (uint16_t) (core_arch_id) >> 12;
            ਜੇ (core_arch_ਸਿਰ == 0xA) {
                ਅੰਕ_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_ਨਕਾਬ |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } ਨਹੀਂਤਾਂ {
        ਲਈ (ਸਵੈਚਾਲਿਤ & core_info : core_info_list) {
            ਸਵੈਚਾਲਿਤ core_arch_id = core_info.arch_id;
            ਜੇ (core_arch_id == perfer_core_arch_id) {
                ਅੰਕ_perfer_cores++;
                cpu_ਨਕਾਬ |= (1ULL << core_info.core_id);

                ਸਵੈਚਾਲਿਤ core_arch_ਸਿਰ = (uint16_t) (core_arch_id) >> 12;
                ਜੇ (core_arch_ਸਿਰ == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        ਜੇ (ਅੰਕ_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    ਜੇ (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_ਨਕਾਬ         = 0;
        ਅੰਕ_perfer_cores = 0;
        ਲਈ (ਪੂਰਨ core_id : perfer_core_id_vec) {
            ਜੇ (core_id < 0 || core_id >= ਅੰਕ_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           ਅੰਕ_cores - 1);
            }
            ਸਵੈਚਾਲਿਤ core_info    = core_info_list[core_id];
            ਸਵੈਚਾਲਿਤ core_arch_id = core_info.arch_id;
            ਜੇ (core_arch_id == perfer_core_arch_id) {
                cpu_ਨਕਾਬ |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } ਨਹੀਂਤਾਂ {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        ਲਈ (ਪੂਰਨ core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        ਅੰਕ_perfer_cores = static_cast<ਪੂਰਨ>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    ਅੱਖਰ * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    ਸਵੈਚਾਲਿਤ   user_disable_tcm      = spine_disable_tcm_str != ਖਾਲੀ_ਸੂਚਕ && strcmp(spine_disable_tcm_str, "0") != 0;

    ਜੇ (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        ਜੇ (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.ਉਪਲਬਧ;
            tcm_blk_ਮਾਪ = tcm_info.blk_ਮਾਪ;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_ਮਾਪ, tcm_info.blk_ਅੰਕ, tcm_info.is_fake_tcm);

            ਲਈ (ਸਵੈਚਾਲਿਤ & core_info : core_info_list) {
                ਸਵੈਚਾਲਿਤ core_arch_ਸਿਰ = (uint16_t) (core_info.arch_id) >> 12;
                ਜੇ (core_arch_ਸਿਰ != 0xA) {
                    aicpu_id_ਵਿੱਥ++;
                } ਨਹੀਂਤਾਂ {
                    ਰੋਕੋ;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        ਅੰਕ_cores, ਅੰਕ_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_ਮੰਜ਼ਿਲ_string(mem_backend), cpu_ਨਕਾਬ, aicpu_id_ਵਿੱਥ);

    ਅਚਲ size_t init_barrier_ਮਾਪ = ਆਕਾਰ(spine_barrier_t) * spine_init_barrier_ਗਿਣਤੀ;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_ਮਾਪ, alignof(spine_barrier_t)));
    ਜੇ (init_barrier != ਖਾਲੀ_ਸੂਚਕ) {
        init_barrier_is_shared_mem = ਸੱਚ;
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = ਨਵਾਂ spine_barrier_t[spine_init_barrier_ਗਿਣਤੀ];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_ਗਿਣਤੀ, 2);
}

spine_env_info::~spine_env_info() {
    ਜੇ (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } ਨਹੀਂਤਾਂ {
        ਮਿਟਾਓ[] init_barrier;
    }

    init_barrier               = ਖਾਲੀ_ਸੂਚਕ;
    init_barrier_is_shared_mem = ਝੂਠ;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
