#অন্তর্ভাব "ime_env.h"

#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "spine_mem_pool.h"

#অন্তর্ভাব <fcntl.h>
#অন্তর্ভাব <unistd.h>

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <array>
#অন্তর্ভাব <cctype>
#অন্তর্ভাব <fstream>
#অন্তর্ভাব <string>
#অন্তর্ভাব <thread>
#অন্তর্ভাব <unordered_map>

নামাকাশ ggml::cpu::riscv64_spacemit {
যৌক্তিক spine_core_info::get_spine_core_info(std::vector<spine_core_info> & ফলাফল) {
    স্থির std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    ফলাফল.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   সারি;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    যৌক্তিক     has_processor     = মিথ্যা;
    যৌক্তিক     has_marchid       = মিথ্যা;

    যদি (!file.is_open()) {
        ফেরত মিথ্যা;
    }

    যখন (std::getline(file, সারি)) {
        যদি (সারি.substr(0, 9) == "processor") {
            যদি (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = সারি.find(':');
            যদি (colon_pos != std::string::npos) {
                current_processor = std::stoi(সারি.substr(colon_pos + 1));
                has_processor     = সত্য;
            }

            has_marchid = মিথ্যা;
        } অন্যথা যদি (সারি.substr(0, 7) == "marchid") {
            size_t colon_pos = সারি.find(':');
            যদি (colon_pos != std::string::npos) {
                std::string marchid_str = সারি.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), isspace), marchid_str.end());
                current_marchid = std::stoull(marchid_str, nullptr, 16);
                has_marchid     = সত্য;
            }
        }
    }

    যদি (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    যদি (has_processor && has_marchid) {
        জন্য (auto & cpu_info : cpu_info_list) {
            যদি (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                auto core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                ফলাফল.push_back(core_info);
            }
        }
    }

    ফেরত has_processor && has_marchid;
}

নামাকাশ {
uint16_t hex_string_to_u16(ধ্রুবক std::string & hex_str) {
    চেষ্টা {
        size_t pos = 0;
        যদি (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        unsigned long ফলাফল = std::stoul(hex_str.substr(pos), nullptr, 16);
        যদি (ফলাফল > std::numeric_limits<uint16_t>::max()) {
            ছোড়ো std::out_of_range("Converted value is out of range for uint16_t");
        }
        ফেরত static_cast<uint16_t>(ফলাফল);
    } ধরো (ধ্রুবক std::invalid_argument & e) {
        ছোড়ো std::invalid_argument("Invalid hexadecimal string");
    } ধরো (ধ্রুবক std::out_of_range & e) {
        ছোড়ো;
    }
}

ধ্রুবক অক্ষর * spine_mem_pool_backend_to_string(spine_mem_pool_backend backend) {
    নির্বাচন (backend) {
        ক্ষেত্র spine_mem_pool_backend::none:
            ফেরত "NONE";
        ক্ষেত্র spine_mem_pool_backend::posix_memalign:
            ফেরত "POSIX";
        ক্ষেত্র spine_mem_pool_backend::transparent_hugepage:
            ফেরত "HPAGE";
        ক্ষেত্র spine_mem_pool_backend::hugetlb_1g:
            ফেরত "HPAGE1GB";
    }

    ফেরত "unknown";
}

spine_mem_pool_backend parse_mem_backend(ধ্রুবক অক্ষর * mem_backend_str) {
    যদি (mem_backend_str == nullptr || mem_backend_str[0] == '\0') {
        ফেরত spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned অক্ষর ch) { ফেরত static_cast<অক্ষর>(std::tolower(ch)); });

    যদি (value == "none") {
        ফেরত spine_mem_pool_backend::none;
    }

    যদি (value == "posix") {
        ফেরত spine_mem_pool_backend::posix_memalign;
    }

    যদি (value == "hpage") {
        ফেরত spine_mem_pool_backend::transparent_hugepage;
    }

    যদি (value == "hpage1gb") {
        ফেরত spine_mem_pool_backend::hugetlb_1g;
    }

    ছোড়ো std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    অঙ্ক_cores = static_cast<পূর্ণসংখ্যা>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    যদি (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    যদি (core_info_list.size() == 0) {
        অক্ষর * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        যদি (spine_core_arch_str != nullptr) {
            auto arch_id = hex_string_to_u16(spine_core_arch_str);
            জন্য (পূর্ণসংখ্যা i = 0; i < অঙ্ক_cores; i++) {
                auto core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    যদি (core_info_list.size() == 0) {
        ছোড়ো std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    অক্ষর * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    যদি (spine_perfer_core_arch_str != nullptr && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_to_u16(spine_perfer_core_arch_str) };
    }

    অক্ষর *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<পূর্ণসংখ্যা> perfer_core_id_vec;
    যদি (spine_perfer_core_id_str != nullptr && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        যখন ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(অঙ্ক_cores);
    যদি (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        জন্য (auto & core_info : core_info_list) {
            auto core_arch_id   = core_info.arch_id;
            auto core_arch_শীর্ষ = (uint16_t) (core_arch_id) >> 12;
            যদি (core_arch_শীর্ষ == 0xA) {
                অঙ্ক_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_mask |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } অন্যথা {
        জন্য (auto & core_info : core_info_list) {
            auto core_arch_id = core_info.arch_id;
            যদি (core_arch_id == perfer_core_arch_id) {
                অঙ্ক_perfer_cores++;
                cpu_mask |= (1ULL << core_info.core_id);

                auto core_arch_শীর্ষ = (uint16_t) (core_arch_id) >> 12;
                যদি (core_arch_শীর্ষ == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        যদি (অঙ্ক_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    যদি (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_mask         = 0;
        অঙ্ক_perfer_cores = 0;
        জন্য (পূর্ণসংখ্যা core_id : perfer_core_id_vec) {
            যদি (core_id < 0 || core_id >= অঙ্ক_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           অঙ্ক_cores - 1);
            }
            auto core_info    = core_info_list[core_id];
            auto core_arch_id = core_info.arch_id;
            যদি (core_arch_id == perfer_core_arch_id) {
                cpu_mask |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } অন্যথা {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        জন্য (পূর্ণসংখ্যা core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        অঙ্ক_perfer_cores = static_cast<পূর্ণসংখ্যা>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    অক্ষর * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    auto   user_disable_tcm      = spine_disable_tcm_str != nullptr && strcmp(spine_disable_tcm_str, "0") != 0;

    যদি (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        যদি (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.available;
            tcm_blk_আকার = tcm_info.blk_আকার;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_আকার, tcm_info.blk_অঙ্ক, tcm_info.is_fake_tcm);

            জন্য (auto & core_info : core_info_list) {
                auto core_arch_শীর্ষ = (uint16_t) (core_info.arch_id) >> 12;
                যদি (core_arch_শীর্ষ != 0xA) {
                    aicpu_id_offset++;
                } অন্যথা {
                    বিরতি;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        অঙ্ক_cores, অঙ্ক_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_to_string(mem_backend), cpu_mask, aicpu_id_offset);

    ধ্রুবক size_t init_barrier_আকার = আকার(spine_barrier_t) * spine_init_barrier_গুনতি;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_আকার, alignof(spine_barrier_t)));
    যদি (init_barrier != nullptr) {
        init_barrier_is_shared_mem = সত্য;
    } অন্যথা {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = নতুন spine_barrier_t[spine_init_barrier_গুনতি];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_গুনতি, 2);
}

spine_env_info::~spine_env_info() {
    যদি (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } অন্যথা {
        delete[] init_barrier;
    }

    init_barrier               = nullptr;
    init_barrier_is_shared_mem = মিথ্যা;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
