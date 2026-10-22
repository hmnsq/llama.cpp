#समावेश "ime_env.h"

#समावेश "ggml-impl.h"
#समावेश "spine_mem_pool.h"

#समावेश <fcntl.h>
#समावेश <unistd.h>

#समावेश <algorithm>
#समावेश <array>
#समावेश <cctype>
#समावेश <fstream>
#समावेश <string>
#समावेश <thread>
#समावेश <unordered_map>

namespace ggml::cpu::riscv64_spacemit {
bool spine_core_info::get_spine_core_info(std::vector<spine_core_info> & परिणाम) {
    अटल std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    परिणाम.सफा();
    std::अख_प्रवाह file("/proc/cpuinfo");
    std::string   पंक्ति;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    bool     has_processor     = false;
    bool     has_marchid       = false;

    अगर (!file.is_open()) {
        वापस false;
    }

    जबतक (std::लो_पंक्ति(file, पंक्ति)) {
        अगर (पंक्ति.substr(0, 9) == "processor") {
            अगर (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            माप_प्रकार colon_pos = पंक्ति.find(':');
            अगर (colon_pos != std::string::npos) {
                current_processor = std::stoi(पंक्ति.substr(colon_pos + 1));
                has_processor     = true;
            }

            has_marchid = false;
        } अन्यथा अगर (पंक्ति.substr(0, 7) == "marchid") {
            माप_प्रकार colon_pos = पंक्ति.find(':');
            अगर (colon_pos != std::string::npos) {
                std::string marchid_str = पंक्ति.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), है_खाली), marchid_str.end());
                current_marchid = std::stoull(marchid_str, nullptr, 16);
                has_marchid     = true;
            }
        }
    }

    अगर (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    अगर (has_processor && has_marchid) {
        क्रम (स्वतः & cpu_info : cpu_info_list) {
            अगर (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                स्वतः core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                परिणाम.push_back(core_info);
            }
        }
    }

    वापस has_processor && has_marchid;
}

namespace {
uint16_t hex_string_to_u16(स्थिर std::string & hex_str) {
    प्रयास {
        माप_प्रकार pos = 0;
        अगर (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        अचिन्हित दीर्घ परिणाम = std::stoul(hex_str.substr(pos), nullptr, 16);
        अगर (परिणाम > std::numeric_limits<uint16_t>::max()) {
            फेंको std::out_of_range("Converted value is out of range for uint16_t");
        }
        वापस static_cast<uint16_t>(परिणाम);
    } पकड़ो (स्थिर std::invalid_argument & e) {
        फेंको std::invalid_argument("Invalid hexadecimal string");
    } पकड़ो (स्थिर std::out_of_range & e) {
        फेंको;
    }
}

स्थिर अक्षर * spine_mem_pool_backend_to_string(spine_mem_pool_backend backend) {
    चयन (backend) {
        हाल spine_mem_pool_backend::none:
            वापस "NONE";
        हाल spine_mem_pool_backend::posix_memalign:
            वापस "POSIX";
        हाल spine_mem_pool_backend::transparent_hugepage:
            वापस "HPAGE";
        हाल spine_mem_pool_backend::hugetlb_1g:
            वापस "HPAGE1GB";
    }

    वापस "unknown";
}

spine_mem_pool_backend parse_mem_backend(स्थिर अक्षर * mem_backend_str) {
    अगर (mem_backend_str == nullptr || mem_backend_str[0] == '\0') {
        वापस spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](अचिन्हित अक्षर ch) { वापस static_cast<अक्षर>(std::छोटे(ch)); });

    अगर (value == "none") {
        वापस spine_mem_pool_backend::none;
    }

    अगर (value == "posix") {
        वापस spine_mem_pool_backend::posix_memalign;
    }

    अगर (value == "hpage") {
        वापस spine_mem_pool_backend::transparent_hugepage;
    }

    अगर (value == "hpage1gb") {
        वापस spine_mem_pool_backend::hugetlb_1g;
    }

    फेंको std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    अंक_cores = static_cast<पूर्णांक>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    अगर (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        क्रम (पूर्णांक i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    अगर (core_info_list.size() == 0) {
        अक्षर * spine_core_arch_str = दो_पर्या("SPACEMIT_CORE_ARCH");
        अगर (spine_core_arch_str != nullptr) {
            स्वतः arch_id = hex_string_to_u16(spine_core_arch_str);
            क्रम (पूर्णांक i = 0; i < अंक_cores; i++) {
                स्वतः core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    अगर (core_info_list.size() == 0) {
        फेंको std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    अक्षर * spine_perfer_core_arch_str = दो_पर्या("SPACEMIT_PERFER_CORE_ARCH");
    अगर (spine_perfer_core_arch_str != nullptr && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_to_u16(spine_perfer_core_arch_str) };
    }

    अक्षर *           spine_perfer_core_id_str = दो_पर्या("SPACEMIT_PERFER_CORE_ID");
    std::vector<पूर्णांक> perfer_core_id_vec;
    अगर (spine_perfer_core_id_str != nullptr && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        माप_प्रकार      start = 0;
        माप_प्रकार      end   = 0;
        जबतक ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(अंक_cores);
    अगर (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        क्रम (स्वतः & core_info : core_info_list) {
            स्वतः core_arch_id   = core_info.arch_id;
            स्वतः core_arch_शीर्ष = (uint16_t) (core_arch_id) >> 12;
            अगर (core_arch_शीर्ष == 0xA) {
                अंक_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_mask |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } अन्यथा {
        क्रम (स्वतः & core_info : core_info_list) {
            स्वतः core_arch_id = core_info.arch_id;
            अगर (core_arch_id == perfer_core_arch_id) {
                अंक_perfer_cores++;
                cpu_mask |= (1ULL << core_info.core_id);

                स्वतः core_arch_शीर्ष = (uint16_t) (core_arch_id) >> 12;
                अगर (core_arch_शीर्ष == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        अगर (अंक_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    अगर (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.सफा();
        cpu_mask         = 0;
        अंक_perfer_cores = 0;
        क्रम (पूर्णांक core_id : perfer_core_id_vec) {
            अगर (core_id < 0 || core_id >= अंक_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           अंक_cores - 1);
            }
            स्वतः core_info    = core_info_list[core_id];
            स्वतः core_arch_id = core_info.arch_id;
            अगर (core_arch_id == perfer_core_arch_id) {
                cpu_mask |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } अन्यथा {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        क्रम (पूर्णांक core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        अंक_perfer_cores = static_cast<पूर्णांक>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(दो_पर्या("SPACEMIT_MEM_BACKEND"));
    अक्षर * spine_disable_tcm_str = दो_पर्या("SPACEMIT_DISABLE_TCM");
    स्वतः   user_disable_tcm      = spine_disable_tcm_str != nullptr && म_भेद(spine_disable_tcm_str, "0") != 0;

    अगर (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        अगर (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.available;
            tcm_blk_आकार = tcm_info.blk_आकार;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_आकार, tcm_info.blk_अंक, tcm_info.is_fake_tcm);

            क्रम (स्वतः & core_info : core_info_list) {
                स्वतः core_arch_शीर्ष = (uint16_t) (core_info.arch_id) >> 12;
                अगर (core_arch_शीर्ष != 0xA) {
                    aicpu_id_offset++;
                } अन्यथा {
                    अवरोध;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        अंक_cores, अंक_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_to_string(mem_backend), cpu_mask, aicpu_id_offset);

    स्थिर माप_प्रकार init_barrier_आकार = माप(spine_barrier_t) * spine_init_barrier_गिनती;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_आकार, alignof(spine_barrier_t)));
    अगर (init_barrier != nullptr) {
        init_barrier_is_shared_mem = true;
    } अन्यथा {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = नया spine_barrier_t[spine_init_barrier_गिनती];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_गिनती, 2);
}

spine_env_info::~spine_env_info() {
    अगर (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } अन्यथा {
        मिटाओ[] init_barrier;
    }

    init_barrier               = nullptr;
    init_barrier_is_shared_mem = false;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
