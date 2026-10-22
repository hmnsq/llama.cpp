#அந்தர்பாவ "ime_env.h"

#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "spine_mem_pool.h"

#அந்தர்பாவ <fcntl.h>
#அந்தர்பாவ <unistd.h>

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <array>
#அந்தர்பாவ <cctype>
#அந்தர்பாவ <fstream>
#அந்தர்பாவ <string>
#அந்தர்பாவ <thread>
#அந்தர்பாவ <unordered_map>

நாமாகாஶ ggml::cpu::riscv64_spacemit {
தருக்க spine_core_info::get_spine_core_info(std::vector<spine_core_info> & result) {
    நிலையான std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    result.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   வரி;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    தருக்க     has_processor     = பொய்;
    தருக்க     has_marchid       = பொய்;

    எனில் (!file.is_open()) {
        திரும்பு பொய்;
    }

    போது (std::getline(file, வரி)) {
        எனில் (வரி.substr(0, 9) == "processor") {
            எனில் (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = வரி.find(':');
            எனில் (colon_pos != std::string::npos) {
                current_processor = std::stoi(வரி.substr(colon_pos + 1));
                has_processor     = உண்மை;
            }

            has_marchid = பொய்;
        } இல்லையெனில் எனில் (வரி.substr(0, 7) == "marchid") {
            size_t colon_pos = வரி.find(':');
            எனில் (colon_pos != std::string::npos) {
                std::string marchid_str = வரி.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), isspace), marchid_str.end());
                current_marchid = std::stoull(marchid_str, nullptr, 16);
                has_marchid     = உண்மை;
            }
        }
    }

    எனில் (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    எனில் (has_processor && has_marchid) {
        க்கு (auto & cpu_info : cpu_info_list) {
            எனில் (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                auto core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                result.push_back(core_info);
            }
        }
    }

    திரும்பு has_processor && has_marchid;
}

நாமாகாஶ {
uint16_t hex_string_to_u16(மாறிலி std::string & hex_str) {
    முயல் {
        size_t pos = 0;
        எனில் (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        unsigned long result = std::stoul(hex_str.substr(pos), nullptr, 16);
        எனில் (result > std::numeric_limits<uint16_t>::max()) {
            எறி std::out_of_range("Converted value is out of range for uint16_t");
        }
        திரும்பு static_cast<uint16_t>(result);
    } பிடி (மாறிலி std::invalid_argument & e) {
        எறி std::invalid_argument("Invalid hexadecimal string");
    } பிடி (மாறிலி std::out_of_range & e) {
        எறி;
    }
}

மாறிலி எழுத்து * spine_mem_pool_backend_to_string(spine_mem_pool_backend backend) {
    தேர்வு (backend) {
        நிலை spine_mem_pool_backend::none:
            திரும்பு "NONE";
        நிலை spine_mem_pool_backend::posix_memalign:
            திரும்பு "POSIX";
        நிலை spine_mem_pool_backend::transparent_hugepage:
            திரும்பு "HPAGE";
        நிலை spine_mem_pool_backend::hugetlb_1g:
            திரும்பு "HPAGE1GB";
    }

    திரும்பு "unknown";
}

spine_mem_pool_backend parse_mem_backend(மாறிலி எழுத்து * mem_backend_str) {
    எனில் (mem_backend_str == nullptr || mem_backend_str[0] == '\0') {
        திரும்பு spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned எழுத்து ch) { திரும்பு static_cast<எழுத்து>(std::tolower(ch)); });

    எனில் (value == "none") {
        திரும்பு spine_mem_pool_backend::none;
    }

    எனில் (value == "posix") {
        திரும்பு spine_mem_pool_backend::posix_memalign;
    }

    எனில் (value == "hpage") {
        திரும்பு spine_mem_pool_backend::transparent_hugepage;
    }

    எனில் (value == "hpage1gb") {
        திரும்பு spine_mem_pool_backend::hugetlb_1g;
    }

    எறி std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    இலக்கம்_cores = static_cast<முழு>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    எனில் (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        க்கு (முழு i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    எனில் (core_info_list.size() == 0) {
        எழுத்து * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        எனில் (spine_core_arch_str != nullptr) {
            auto arch_id = hex_string_to_u16(spine_core_arch_str);
            க்கு (முழு i = 0; i < இலக்கம்_cores; i++) {
                auto core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    எனில் (core_info_list.size() == 0) {
        எறி std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    எழுத்து * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    எனில் (spine_perfer_core_arch_str != nullptr && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_to_u16(spine_perfer_core_arch_str) };
    }

    எழுத்து *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<முழு> perfer_core_id_vec;
    எனில் (spine_perfer_core_id_str != nullptr && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        போது ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(இலக்கம்_cores);
    எனில் (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        க்கு (auto & core_info : core_info_list) {
            auto core_arch_id   = core_info.arch_id;
            auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
            எனில் (core_arch_head == 0xA) {
                இலக்கம்_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_mask |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } இல்லையெனில் {
        க்கு (auto & core_info : core_info_list) {
            auto core_arch_id = core_info.arch_id;
            எனில் (core_arch_id == perfer_core_arch_id) {
                இலக்கம்_perfer_cores++;
                cpu_mask |= (1ULL << core_info.core_id);

                auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
                எனில் (core_arch_head == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        எனில் (இலக்கம்_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    எனில் (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_mask         = 0;
        இலக்கம்_perfer_cores = 0;
        க்கு (முழு core_id : perfer_core_id_vec) {
            எனில் (core_id < 0 || core_id >= இலக்கம்_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           இலக்கம்_cores - 1);
            }
            auto core_info    = core_info_list[core_id];
            auto core_arch_id = core_info.arch_id;
            எனில் (core_arch_id == perfer_core_arch_id) {
                cpu_mask |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } இல்லையெனில் {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        க்கு (முழு core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        இலக்கம்_perfer_cores = static_cast<முழு>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    எழுத்து * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    auto   user_disable_tcm      = spine_disable_tcm_str != nullptr && strcmp(spine_disable_tcm_str, "0") != 0;

    எனில் (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        எனில் (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.available;
            tcm_blk_size = tcm_info.blk_size;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_size, tcm_info.blk_இலக்கம், tcm_info.is_fake_tcm);

            க்கு (auto & core_info : core_info_list) {
                auto core_arch_head = (uint16_t) (core_info.arch_id) >> 12;
                எனில் (core_arch_head != 0xA) {
                    aicpu_id_offset++;
                } இல்லையெனில் {
                    நிறுத்து;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        இலக்கம்_cores, இலக்கம்_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_to_string(mem_backend), cpu_mask, aicpu_id_offset);

    மாறிலி size_t init_barrier_size = அளவு(spine_barrier_t) * spine_init_barrier_மொத்தம்;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_size, alignof(spine_barrier_t)));
    எனில் (init_barrier != nullptr) {
        init_barrier_is_shared_mem = உண்மை;
    } இல்லையெனில் {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = புதிய spine_barrier_t[spine_init_barrier_மொத்தம்];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_மொத்தம், 2);
}

spine_env_info::~spine_env_info() {
    எனில் (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } இல்லையெனில் {
        delete[] init_barrier;
    }

    init_barrier               = nullptr;
    init_barrier_is_shared_mem = பொய்;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
