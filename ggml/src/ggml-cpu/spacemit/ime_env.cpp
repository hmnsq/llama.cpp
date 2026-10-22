#अन्तर्भाव "ime_env.h"

#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "spine_mem_pool.h"

#अन्तर्भाव <fcntl.h>
#अन्तर्भाव <unistd.h>

#अन्तर्भाव <algorithm>
#अन्तर्भाव <array>
#अन्तर्भाव <cctype>
#अन्तर्भाव <fstream>
#अन्तर्भाव <string>
#अन्तर्भाव <thread>
#अन्तर्भाव <unordered_map>

नामाकाश ggml::cpu::riscv64_spacemit {
तार्किक spine_core_info::get_spine_core_info(std::vector<spine_core_info> & result) {
    स्थिर std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    result.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   पङ्क्ति;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    तार्किक     has_processor     = असत्य;
    तार्किक     has_marchid       = असत्य;

    यदि (!file.is_open()) {
        फर्काऊ असत्य;
    }

    जबसम्म (std::getline(file, पङ्क्ति)) {
        यदि (पङ्क्ति.substr(0, 9) == "processor") {
            यदि (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            माप_प्रकार colon_pos = पङ्क्ति.find(':');
            यदि (colon_pos != std::string::npos) {
                current_processor = std::stoi(पङ्क्ति.substr(colon_pos + 1));
                has_processor     = सत्य;
            }

            has_marchid = असत्य;
        } अन्यथा यदि (पङ्क्ति.substr(0, 7) == "marchid") {
            माप_प्रकार colon_pos = पङ्क्ति.find(':');
            यदि (colon_pos != std::string::npos) {
                std::string marchid_str = पङ्क्ति.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), है_खाली), marchid_str.end());
                current_marchid = std::stoull(marchid_str, nullptr, 16);
                has_marchid     = सत्य;
            }
        }
    }

    यदि (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    यदि (has_processor && has_marchid) {
        लागि (स्वतः & cpu_info : cpu_info_list) {
            यदि (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                स्वतः core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                result.push_back(core_info);
            }
        }
    }

    फर्काऊ has_processor && has_marchid;
}

नामाकाश {
uint16_t hex_string_to_u16(अचर std::string & hex_str) {
    यत्न {
        माप_प्रकार pos = 0;
        यदि (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        अचिन्हित दीर्घ result = std::stoul(hex_str.substr(pos), nullptr, 16);
        यदि (result > std::numeric_limits<uint16_t>::max()) {
            क्षिप std::out_of_range("Converted value is out of range for uint16_t");
        }
        फर्काऊ static_cast<uint16_t>(result);
    } गृहाण (अचर std::invalid_argument & e) {
        क्षिप std::invalid_argument("Invalid hexadecimal string");
    } गृहाण (अचर std::out_of_range & e) {
        क्षिप;
    }
}

अचर वर्ण * spine_mem_pool_backend_to_string(spine_mem_pool_backend backend) {
    विकल्प (backend) {
        पक्ष spine_mem_pool_backend::none:
            फर्काऊ "NONE";
        पक्ष spine_mem_pool_backend::posix_memalign:
            फर्काऊ "POSIX";
        पक्ष spine_mem_pool_backend::transparent_hugepage:
            फर्काऊ "HPAGE";
        पक्ष spine_mem_pool_backend::hugetlb_1g:
            फर्काऊ "HPAGE1GB";
    }

    फर्काऊ "unknown";
}

spine_mem_pool_backend parse_mem_backend(अचर वर्ण * mem_backend_str) {
    यदि (mem_backend_str == nullptr || mem_backend_str[0] == '\0') {
        फर्काऊ spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](अचिन्हित वर्ण ch) { फर्काऊ static_cast<वर्ण>(std::छोटे(ch)); });

    यदि (value == "none") {
        फर्काऊ spine_mem_pool_backend::none;
    }

    यदि (value == "posix") {
        फर्काऊ spine_mem_pool_backend::posix_memalign;
    }

    यदि (value == "hpage") {
        फर्काऊ spine_mem_pool_backend::transparent_hugepage;
    }

    यदि (value == "hpage1gb") {
        फर्काऊ spine_mem_pool_backend::hugetlb_1g;
    }

    क्षिप std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    अङ्क_cores = static_cast<पूर्णाङ्क>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    यदि (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        लागि (पूर्णाङ्क i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    यदि (core_info_list.size() == 0) {
        वर्ण * spine_core_arch_str = दो_पर्या("SPACEMIT_CORE_ARCH");
        यदि (spine_core_arch_str != nullptr) {
            स्वतः arch_id = hex_string_to_u16(spine_core_arch_str);
            लागि (पूर्णाङ्क i = 0; i < अङ्क_cores; i++) {
                स्वतः core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    यदि (core_info_list.size() == 0) {
        क्षिप std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    वर्ण * spine_perfer_core_arch_str = दो_पर्या("SPACEMIT_PERFER_CORE_ARCH");
    यदि (spine_perfer_core_arch_str != nullptr && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_to_u16(spine_perfer_core_arch_str) };
    }

    वर्ण *           spine_perfer_core_id_str = दो_पर्या("SPACEMIT_PERFER_CORE_ID");
    std::vector<पूर्णाङ्क> perfer_core_id_vec;
    यदि (spine_perfer_core_id_str != nullptr && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        माप_प्रकार      start = 0;
        माप_प्रकार      end   = 0;
        जबसम्म ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(अङ्क_cores);
    यदि (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        लागि (स्वतः & core_info : core_info_list) {
            स्वतः core_arch_id   = core_info.arch_id;
            स्वतः core_arch_head = (uint16_t) (core_arch_id) >> 12;
            यदि (core_arch_head == 0xA) {
                अङ्क_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_mask |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } अन्यथा {
        लागि (स्वतः & core_info : core_info_list) {
            स्वतः core_arch_id = core_info.arch_id;
            यदि (core_arch_id == perfer_core_arch_id) {
                अङ्क_perfer_cores++;
                cpu_mask |= (1ULL << core_info.core_id);

                स्वतः core_arch_head = (uint16_t) (core_arch_id) >> 12;
                यदि (core_arch_head == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        यदि (अङ्क_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    यदि (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_mask         = 0;
        अङ्क_perfer_cores = 0;
        लागि (पूर्णाङ्क core_id : perfer_core_id_vec) {
            यदि (core_id < 0 || core_id >= अङ्क_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           अङ्क_cores - 1);
            }
            स्वतः core_info    = core_info_list[core_id];
            स्वतः core_arch_id = core_info.arch_id;
            यदि (core_arch_id == perfer_core_arch_id) {
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
        लागि (पूर्णाङ्क core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        अङ्क_perfer_cores = static_cast<पूर्णाङ्क>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(दो_पर्या("SPACEMIT_MEM_BACKEND"));
    वर्ण * spine_disable_tcm_str = दो_पर्या("SPACEMIT_DISABLE_TCM");
    स्वतः   user_disable_tcm      = spine_disable_tcm_str != nullptr && म_भेद(spine_disable_tcm_str, "0") != 0;

    यदि (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        यदि (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.available;
            tcm_blk_size = tcm_info.blk_size;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_size, tcm_info.blk_अङ्क, tcm_info.is_fake_tcm);

            लागि (स्वतः & core_info : core_info_list) {
                स्वतः core_arch_head = (uint16_t) (core_info.arch_id) >> 12;
                यदि (core_arch_head != 0xA) {
                    aicpu_id_offset++;
                } अन्यथा {
                    रोक;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        अङ्क_cores, अङ्क_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_to_string(mem_backend), cpu_mask, aicpu_id_offset);

    अचर माप_प्रकार init_barrier_size = परिमाण(spine_barrier_t) * spine_init_barrier_गन्ती;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_size, alignof(spine_barrier_t)));
    यदि (init_barrier != nullptr) {
        init_barrier_is_shared_mem = सत्य;
    } अन्यथा {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = नूतन spine_barrier_t[spine_init_barrier_गन्ती];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_गन्ती, 2);
}

spine_env_info::~spine_env_info() {
    यदि (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } अन्यथा {
        delete[] init_barrier;
    }

    init_barrier               = nullptr;
    init_barrier_is_shared_mem = असत्य;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
