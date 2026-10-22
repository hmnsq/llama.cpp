#ضمن "ime_env.h"

#ضمن "ggml-impl.h"
#ضمن "spine_mem_pool.h"

#ضمن <fcntl.h>
#ضمن <unistd.h>

#ضمن <algorithm>
#ضمن <array>
#ضمن <cctype>
#ضمن <fstream>
#ضمن <string>
#ضمن <thread>
#ضمن <unordered_map>

نطاق ggml::cpu::riscv64_spacemit {
منطقي spine_core_info::get_spine_core_info(std::vector<spine_core_info> & پایله) {
    ساكن std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    پایله.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   کرښه;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    منطقي     has_processor     = خطأ;
    منطقي     has_marchid       = خطأ;

    إذا (!file.is_open()) {
        أرجع خطأ;
    }

    طالما (std::getline(file, کرښه)) {
        إذا (کرښه.substr(0, 9) == "processor") {
            إذا (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = کرښه.find(':');
            إذا (colon_pos != std::string::npos) {
                current_processor = std::stoi(کرښه.substr(colon_pos + 1));
                has_processor     = ریښتیا;
            }

            has_marchid = خطأ;
        } وإلا إذا (کرښه.substr(0, 7) == "marchid") {
            size_t colon_pos = کرښه.find(':');
            إذا (colon_pos != std::string::npos) {
                std::string marchid_str = کرښه.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), که_تشه), marchid_str.end());
                current_marchid = std::stoull(marchid_str, تش_اشاره, 16);
                has_marchid     = ریښتیا;
            }
        }
    }

    إذا (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    إذا (has_processor && has_marchid) {
        لكل (خپلکاری & cpu_info : cpu_info_list) {
            إذا (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                خپلکاری core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                پایله.push_back(core_info);
            }
        }
    }

    أرجع has_processor && has_marchid;
}

نطاق {
uint16_t hex_string_منزل_u16(ثابت std::string & hex_str) {
    حاول {
        size_t pos = 0;
        إذا (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        بینښه اوږد پایله = std::stoul(hex_str.substr(pos), تش_اشاره, 16);
        إذا (پایله > std::numeric_limits<uint16_t>::max()) {
            ارم std::out_of_range("Converted value is out of range for uint16_t");
        }
        أرجع static_cast<uint16_t>(پایله);
    } امسك (ثابت std::invalid_argument & e) {
        ارم std::invalid_argument("Invalid hexadecimal string");
    } امسك (ثابت std::out_of_range & e) {
        ارم;
    }
}

ثابت حرف * spine_mem_pool_backend_منزل_string(spine_mem_pool_backend backend) {
    بدل (backend) {
        حالة spine_mem_pool_backend::none:
            أرجع "NONE";
        حالة spine_mem_pool_backend::posix_memalign:
            أرجع "POSIX";
        حالة spine_mem_pool_backend::transparent_hugepage:
            أرجع "HPAGE";
        حالة spine_mem_pool_backend::hugetlb_1g:
            أرجع "HPAGE1GB";
    }

    أرجع "unknown";
}

spine_mem_pool_backend parse_mem_backend(ثابت حرف * mem_backend_str) {
    إذا (mem_backend_str == تش_اشاره || mem_backend_str[0] == '\0') {
        أرجع spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](بینښه حرف ch) { أرجع static_cast<حرف>(std::tolower(ch)); });

    إذا (value == "none") {
        أرجع spine_mem_pool_backend::none;
    }

    إذا (value == "posix") {
        أرجع spine_mem_pool_backend::posix_memalign;
    }

    إذا (value == "hpage") {
        أرجع spine_mem_pool_backend::transparent_hugepage;
    }

    إذا (value == "hpage1gb") {
        أرجع spine_mem_pool_backend::hugetlb_1g;
    }

    ارم std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    عدد_cores = static_cast<صحيح>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    إذا (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        لكل (صحيح i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    إذا (core_info_list.size() == 0) {
        حرف * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        إذا (spine_core_arch_str != تش_اشاره) {
            خپلکاری arch_id = hex_string_منزل_u16(spine_core_arch_str);
            لكل (صحيح i = 0; i < عدد_cores; i++) {
                خپلکاری core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    إذا (core_info_list.size() == 0) {
        ارم std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    حرف * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    إذا (spine_perfer_core_arch_str != تش_اشاره && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_منزل_u16(spine_perfer_core_arch_str) };
    }

    حرف *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<صحيح> perfer_core_id_vec;
    إذا (spine_perfer_core_id_str != تش_اشاره && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        طالما ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(عدد_cores);
    إذا (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        لكل (خپلکاری & core_info : core_info_list) {
            خپلکاری core_arch_id   = core_info.arch_id;
            خپلکاری core_arch_سر = (uint16_t) (core_arch_id) >> 12;
            إذا (core_arch_سر == 0xA) {
                عدد_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_ماسک |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } وإلا {
        لكل (خپلکاری & core_info : core_info_list) {
            خپلکاری core_arch_id = core_info.arch_id;
            إذا (core_arch_id == perfer_core_arch_id) {
                عدد_perfer_cores++;
                cpu_ماسک |= (1ULL << core_info.core_id);

                خپلکاری core_arch_سر = (uint16_t) (core_arch_id) >> 12;
                إذا (core_arch_سر == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        إذا (عدد_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    إذا (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_ماسک         = 0;
        عدد_perfer_cores = 0;
        لكل (صحيح core_id : perfer_core_id_vec) {
            إذا (core_id < 0 || core_id >= عدد_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           عدد_cores - 1);
            }
            خپلکاری core_info    = core_info_list[core_id];
            خپلکاری core_arch_id = core_info.arch_id;
            إذا (core_arch_id == perfer_core_arch_id) {
                cpu_ماسک |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } وإلا {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        لكل (صحيح core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        عدد_perfer_cores = static_cast<صحيح>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    حرف * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    خپلکاری   user_disable_tcm      = spine_disable_tcm_str != تش_اشاره && strcmp(spine_disable_tcm_str, "0") != 0;

    إذا (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        إذا (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.شته;
            tcm_blk_اندازه = tcm_info.blk_اندازه;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_اندازه, tcm_info.blk_عدد, tcm_info.is_fake_tcm);

            لكل (خپلکاری & core_info : core_info_list) {
                خپلکاری core_arch_سر = (uint16_t) (core_info.arch_id) >> 12;
                إذا (core_arch_سر != 0xA) {
                    aicpu_id_دننه_واټن++;
                } وإلا {
                    اكسر;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        عدد_cores, عدد_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_منزل_string(mem_backend), cpu_ماسک, aicpu_id_دننه_واټن);

    ثابت size_t init_barrier_اندازه = حجم(spine_barrier_t) * spine_init_barrier_شمېر;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_اندازه, alignof(spine_barrier_t)));
    إذا (init_barrier != تش_اشاره) {
        init_barrier_is_shared_mem = ریښتیا;
    } وإلا {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = جديد spine_barrier_t[spine_init_barrier_شمېر];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_شمېر, 2);
}

spine_env_info::~spine_env_info() {
    إذا (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } وإلا {
        ړنګول[] init_barrier;
    }

    init_barrier               = تش_اشاره;
    init_barrier_is_shared_mem = خطأ;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
