#شامل "ime_env.h"

#شامل "ggml-impl.h"
#شامل "spine_mem_pool.h"

#شامل <fcntl.h>
#شامل <unistd.h>

#شامل <algorithm>
#شامل <array>
#شامل <cctype>
#شامل <fstream>
#شامل <string>
#شامل <thread>
#شامل <unordered_map>

نامخلا ggml::cpu::riscv64_spacemit {
منطقی spine_core_info::get_spine_core_info(std::vector<spine_core_info> & نتیجہ) {
    ساکن std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    نتیجہ.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   سطر;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    منطقی     has_processor     = جھوٹ;
    منطقی     has_marchid       = جھوٹ;

    اگر (!file.is_open()) {
        واپس جھوٹ;
    }

    جب (std::getline(file, سطر)) {
        اگر (سطر.substr(0, 9) == "processor") {
            اگر (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = سطر.find(':');
            اگر (colon_pos != std::string::npos) {
                current_processor = std::stoi(سطر.substr(colon_pos + 1));
                has_processor     = سچ;
            }

            has_marchid = جھوٹ;
        } ورنہ اگر (سطر.substr(0, 7) == "marchid") {
            size_t colon_pos = سطر.find(':');
            اگر (colon_pos != std::string::npos) {
                std::string marchid_str = سطر.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), خلا_ہے), marchid_str.end());
                current_marchid = std::stoull(marchid_str, خالی_اشاریہ, 16);
                has_marchid     = سچ;
            }
        }
    }

    اگر (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    اگر (has_processor && has_marchid) {
        برائے (خودکار & cpu_info : cpu_info_list) {
            اگر (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                خودکار core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                نتیجہ.push_back(core_info);
            }
        }
    }

    واپس has_processor && has_marchid;
}

نامخلا {
uint16_t hex_string_مقصد_u16(ثابت std::string & hex_str) {
    کوشش {
        size_t pos = 0;
        اگر (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        غیرعلامتی طویل نتیجہ = std::stoul(hex_str.substr(pos), خالی_اشاریہ, 16);
        اگر (نتیجہ > std::numeric_limits<uint16_t>::max()) {
            پھینکو std::out_of_range("Converted value is out of range for uint16_t");
        }
        واپس static_cast<uint16_t>(نتیجہ);
    } پکڑو (ثابت std::invalid_argument & e) {
        پھینکو std::invalid_argument("Invalid hexadecimal string");
    } پکڑو (ثابت std::out_of_range & e) {
        پھینکو;
    }
}

ثابت حرف * spine_mem_pool_backend_مقصد_string(spine_mem_pool_backend backend) {
    انتخاب (backend) {
        حالت spine_mem_pool_backend::none:
            واپس "NONE";
        حالت spine_mem_pool_backend::posix_memalign:
            واپس "POSIX";
        حالت spine_mem_pool_backend::transparent_hugepage:
            واپس "HPAGE";
        حالت spine_mem_pool_backend::hugetlb_1g:
            واپس "HPAGE1GB";
    }

    واپس "unknown";
}

spine_mem_pool_backend parse_mem_backend(ثابت حرف * mem_backend_str) {
    اگر (mem_backend_str == خالی_اشاریہ || mem_backend_str[0] == '\0') {
        واپس spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](غیرعلامتی حرف ch) { واپس static_cast<حرف>(std::tolower(ch)); });

    اگر (value == "none") {
        واپس spine_mem_pool_backend::none;
    }

    اگر (value == "posix") {
        واپس spine_mem_pool_backend::posix_memalign;
    }

    اگر (value == "hpage") {
        واپس spine_mem_pool_backend::transparent_hugepage;
    }

    اگر (value == "hpage1gb") {
        واپس spine_mem_pool_backend::hugetlb_1g;
    }

    پھینکو std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    عدد_cores = static_cast<صحیح>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    اگر (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        برائے (صحیح i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    اگر (core_info_list.size() == 0) {
        حرف * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        اگر (spine_core_arch_str != خالی_اشاریہ) {
            خودکار arch_id = hex_string_مقصد_u16(spine_core_arch_str);
            برائے (صحیح i = 0; i < عدد_cores; i++) {
                خودکار core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    اگر (core_info_list.size() == 0) {
        پھینکو std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    حرف * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    اگر (spine_perfer_core_arch_str != خالی_اشاریہ && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_مقصد_u16(spine_perfer_core_arch_str) };
    }

    حرف *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<صحیح> perfer_core_id_vec;
    اگر (spine_perfer_core_id_str != خالی_اشاریہ && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        جب ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(عدد_cores);
    اگر (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        برائے (خودکار & core_info : core_info_list) {
            خودکار core_arch_id   = core_info.arch_id;
            خودکار core_arch_سر = (uint16_t) (core_arch_id) >> 12;
            اگر (core_arch_سر == 0xA) {
                عدد_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_نقاب |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } ورنہ {
        برائے (خودکار & core_info : core_info_list) {
            خودکار core_arch_id = core_info.arch_id;
            اگر (core_arch_id == perfer_core_arch_id) {
                عدد_perfer_cores++;
                cpu_نقاب |= (1ULL << core_info.core_id);

                خودکار core_arch_سر = (uint16_t) (core_arch_id) >> 12;
                اگر (core_arch_سر == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        اگر (عدد_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    اگر (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_نقاب         = 0;
        عدد_perfer_cores = 0;
        برائے (صحیح core_id : perfer_core_id_vec) {
            اگر (core_id < 0 || core_id >= عدد_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           عدد_cores - 1);
            }
            خودکار core_info    = core_info_list[core_id];
            خودکار core_arch_id = core_info.arch_id;
            اگر (core_arch_id == perfer_core_arch_id) {
                cpu_نقاب |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } ورنہ {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        برائے (صحیح core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        عدد_perfer_cores = static_cast<صحیح>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    حرف * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    خودکار   user_disable_tcm      = spine_disable_tcm_str != خالی_اشاریہ && strcmp(spine_disable_tcm_str, "0") != 0;

    اگر (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        اگر (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.دستیاب;
            tcm_blk_سائز = tcm_info.blk_سائز;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_سائز, tcm_info.blk_عدد, tcm_info.is_fake_tcm);

            برائے (خودکار & core_info : core_info_list) {
                خودکار core_arch_سر = (uint16_t) (core_info.arch_id) >> 12;
                اگر (core_arch_سر != 0xA) {
                    aicpu_id_انحراف++;
                } ورنہ {
                    توڑو;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        عدد_cores, عدد_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_مقصد_string(mem_backend), cpu_نقاب, aicpu_id_انحراف);

    ثابت size_t init_barrier_سائز = حجم(spine_barrier_t) * spine_init_barrier_گنتی;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_سائز, alignof(spine_barrier_t)));
    اگر (init_barrier != خالی_اشاریہ) {
        init_barrier_is_shared_mem = سچ;
    } ورنہ {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = نیا spine_barrier_t[spine_init_barrier_گنتی];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_گنتی, 2);
}

spine_env_info::~spine_env_info() {
    اگر (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } ورنہ {
        حذف[] init_barrier;
    }

    init_barrier               = خالی_اشاریہ;
    init_barrier_is_shared_mem = جھوٹ;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
