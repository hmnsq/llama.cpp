#ଅନ୍ତର୍ଭାଵ "ime_env.h"

#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "spine_mem_pool.h"

#ଅନ୍ତର୍ଭାଵ <fcntl.h>
#ଅନ୍ତର୍ଭାଵ <unistd.h>

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <array>
#ଅନ୍ତର୍ଭାଵ <cctype>
#ଅନ୍ତର୍ଭାଵ <fstream>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <thread>
#ଅନ୍ତର୍ଭାଵ <unordered_map>

ନାମାକାଶ ggml::cpu::riscv64_spacemit {
ତାର୍କିକ spine_core_info::get_spine_core_info(std::vector<spine_core_info> & result) {
    ସ୍ଥିର std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    result.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   ଧାଡ଼ି;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    ତାର୍କିକ     has_processor     = ମିଥ୍ୟା;
    ତାର୍କିକ     has_marchid       = ମିଥ୍ୟା;

    ଯଦି (!file.is_open()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯେତେବେଳେ (std::getline(file, ଧାଡ଼ି)) {
        ଯଦି (ଧାଡ଼ି.substr(0, 9) == "processor") {
            ଯଦି (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = ଧାଡ଼ି.find(':');
            ଯଦି (colon_pos != std::string::npos) {
                current_processor = std::stoi(ଧାଡ଼ି.substr(colon_pos + 1));
                has_processor     = ସତ୍ୟ;
            }

            has_marchid = ମିଥ୍ୟା;
        } ଅନ୍ୟଥା ଯଦି (ଧାଡ଼ି.substr(0, 7) == "marchid") {
            size_t colon_pos = ଧାଡ଼ି.find(':');
            ଯଦି (colon_pos != std::string::npos) {
                std::string marchid_str = ଧାଡ଼ି.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), isspace), marchid_str.end());
                current_marchid = std::stoull(marchid_str, nullptr, 16);
                has_marchid     = ସତ୍ୟ;
            }
        }
    }

    ଯଦି (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    ଯଦି (has_processor && has_marchid) {
        ପାଇଁ (auto & cpu_info : cpu_info_list) {
            ଯଦି (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                auto core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                result.push_back(core_info);
            }
        }
    }

    ଫେରନ୍ତୁ has_processor && has_marchid;
}

ନାମାକାଶ {
uint16_t hex_string_to_u16(ଅଚଳ std::string & hex_str) {
    ଚେଷ୍ଟା {
        size_t pos = 0;
        ଯଦି (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        unsigned long result = std::stoul(hex_str.substr(pos), nullptr, 16);
        ଯଦି (result > std::numeric_limits<uint16_t>::max()) {
            ଫିଙ୍ଗ std::out_of_range("Converted value is out of range for uint16_t");
        }
        ଫେରନ୍ତୁ static_cast<uint16_t>(result);
    } ଧର (ଅଚଳ std::invalid_argument & e) {
        ଫିଙ୍ଗ std::invalid_argument("Invalid hexadecimal string");
    } ଧର (ଅଚଳ std::out_of_range & e) {
        ଫିଙ୍ଗ;
    }
}

ଅଚଳ ଅକ୍ଷର * spine_mem_pool_backend_to_string(spine_mem_pool_backend backend) {
    ବାଛ (backend) {
        ସ୍ଥିତି spine_mem_pool_backend::none:
            ଫେରନ୍ତୁ "NONE";
        ସ୍ଥିତି spine_mem_pool_backend::posix_memalign:
            ଫେରନ୍ତୁ "POSIX";
        ସ୍ଥିତି spine_mem_pool_backend::transparent_hugepage:
            ଫେରନ୍ତୁ "HPAGE";
        ସ୍ଥିତି spine_mem_pool_backend::hugetlb_1g:
            ଫେରନ୍ତୁ "HPAGE1GB";
    }

    ଫେରନ୍ତୁ "unknown";
}

spine_mem_pool_backend parse_mem_backend(ଅଚଳ ଅକ୍ଷର * mem_backend_str) {
    ଯଦି (mem_backend_str == nullptr || mem_backend_str[0] == '\0') {
        ଫେରନ୍ତୁ spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned ଅକ୍ଷର ch) { ଫେରନ୍ତୁ static_cast<ଅକ୍ଷର>(std::tolower(ch)); });

    ଯଦି (value == "none") {
        ଫେରନ୍ତୁ spine_mem_pool_backend::none;
    }

    ଯଦି (value == "posix") {
        ଫେରନ୍ତୁ spine_mem_pool_backend::posix_memalign;
    }

    ଯଦି (value == "hpage") {
        ଫେରନ୍ତୁ spine_mem_pool_backend::transparent_hugepage;
    }

    ଯଦି (value == "hpage1gb") {
        ଫେରନ୍ତୁ spine_mem_pool_backend::hugetlb_1g;
    }

    ଫିଙ୍ଗ std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    ଅଙ୍କ_cores = static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    ଯଦି (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    ଯଦି (core_info_list.size() == 0) {
        ଅକ୍ଷର * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        ଯଦି (spine_core_arch_str != nullptr) {
            auto arch_id = hex_string_to_u16(spine_core_arch_str);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଅଙ୍କ_cores; i++) {
                auto core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    ଯଦି (core_info_list.size() == 0) {
        ଫିଙ୍ଗ std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    ଅକ୍ଷର * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    ଯଦି (spine_perfer_core_arch_str != nullptr && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_to_u16(spine_perfer_core_arch_str) };
    }

    ଅକ୍ଷର *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> perfer_core_id_vec;
    ଯଦି (spine_perfer_core_id_str != nullptr && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        ଯେତେବେଳେ ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(ଅଙ୍କ_cores);
    ଯଦି (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        ପାଇଁ (auto & core_info : core_info_list) {
            auto core_arch_id   = core_info.arch_id;
            auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
            ଯଦି (core_arch_head == 0xA) {
                ଅଙ୍କ_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_mask |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } ଅନ୍ୟଥା {
        ପାଇଁ (auto & core_info : core_info_list) {
            auto core_arch_id = core_info.arch_id;
            ଯଦି (core_arch_id == perfer_core_arch_id) {
                ଅଙ୍କ_perfer_cores++;
                cpu_mask |= (1ULL << core_info.core_id);

                auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
                ଯଦି (core_arch_head == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        ଯଦି (ଅଙ୍କ_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    ଯଦି (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_mask         = 0;
        ଅଙ୍କ_perfer_cores = 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା core_id : perfer_core_id_vec) {
            ଯଦି (core_id < 0 || core_id >= ଅଙ୍କ_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           ଅଙ୍କ_cores - 1);
            }
            auto core_info    = core_info_list[core_id];
            auto core_arch_id = core_info.arch_id;
            ଯଦି (core_arch_id == perfer_core_arch_id) {
                cpu_mask |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } ଅନ୍ୟଥା {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        ଅଙ୍କ_perfer_cores = static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    ଅକ୍ଷର * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    auto   user_disable_tcm      = spine_disable_tcm_str != nullptr && strcmp(spine_disable_tcm_str, "0") != 0;

    ଯଦି (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        ଯଦି (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.available;
            tcm_blk_size = tcm_info.blk_size;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_size, tcm_info.blk_ଅଙ୍କ, tcm_info.is_fake_tcm);

            ପାଇଁ (auto & core_info : core_info_list) {
                auto core_arch_head = (uint16_t) (core_info.arch_id) >> 12;
                ଯଦି (core_arch_head != 0xA) {
                    aicpu_id_offset++;
                } ଅନ୍ୟଥା {
                    ଭାଙ୍ଗ;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        ଅଙ୍କ_cores, ଅଙ୍କ_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_to_string(mem_backend), cpu_mask, aicpu_id_offset);

    ଅଚଳ size_t init_barrier_size = ଆକାର(spine_barrier_t) * spine_init_barrier_ଗଣନା;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_size, alignof(spine_barrier_t)));
    ଯଦି (init_barrier != nullptr) {
        init_barrier_is_shared_mem = ସତ୍ୟ;
    } ଅନ୍ୟଥା {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = ନୂତନ spine_barrier_t[spine_init_barrier_ଗଣନା];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_ଗଣନା, 2);
}

spine_env_info::~spine_env_info() {
    ଯଦି (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } ଅନ୍ୟଥା {
        delete[] init_barrier;
    }

    init_barrier               = nullptr;
    init_barrier_is_shared_mem = ମିଥ୍ୟା;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
