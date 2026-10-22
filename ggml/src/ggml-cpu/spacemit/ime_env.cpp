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

فضاینام ggml::cpu::riscv64_spacemit {
منطقی spine_core_info::get_spine_core_info(std::vector<spine_core_info> & result) {
    ایستا std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    result.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   سطر;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    منطقی     has_processor     = نادرست;
    منطقی     has_marchid       = نادرست;

    اگر (!file.is_open()) {
        بازگشت نادرست;
    }

    وقتی (std::getline(file, سطر)) {
        اگر (سطر.substr(0, 9) == "processor") {
            اگر (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = سطر.find(':');
            اگر (colon_pos != std::string::npos) {
                current_processor = std::stoi(سطر.substr(colon_pos + 1));
                has_processor     = درست;
            }

            has_marchid = نادرست;
        } وگرنه اگر (سطر.substr(0, 7) == "marchid") {
            size_t colon_pos = سطر.find(':');
            اگر (colon_pos != std::string::npos) {
                std::string marchid_str = سطر.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), isspace), marchid_str.end());
                current_marchid = std::stoull(marchid_str, nullptr, 16);
                has_marchid     = درست;
            }
        }
    }

    اگر (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    اگر (has_processor && has_marchid) {
        برای (auto & cpu_info : cpu_info_list) {
            اگر (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                auto core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                result.push_back(core_info);
            }
        }
    }

    بازگشت has_processor && has_marchid;
}

فضاینام {
uint16_t hex_string_to_u16(ثابت std::string & hex_str) {
    تلاش {
        size_t pos = 0;
        اگر (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        unsigned long result = std::stoul(hex_str.substr(pos), nullptr, 16);
        اگر (result > std::numeric_limits<uint16_t>::max()) {
            پرتابکن std::out_of_range("Converted value is out of range for uint16_t");
        }
        بازگشت static_cast<uint16_t>(result);
    } بگیر (ثابت std::invalid_argument & e) {
        پرتابکن std::invalid_argument("Invalid hexadecimal string");
    } بگیر (ثابت std::out_of_range & e) {
        پرتابکن;
    }
}

ثابت نویسه * spine_mem_pool_backend_to_string(spine_mem_pool_backend backend) {
    انتخاب (backend) {
        حالت spine_mem_pool_backend::none:
            بازگشت "NONE";
        حالت spine_mem_pool_backend::posix_memalign:
            بازگشت "POSIX";
        حالت spine_mem_pool_backend::transparent_hugepage:
            بازگشت "HPAGE";
        حالت spine_mem_pool_backend::hugetlb_1g:
            بازگشت "HPAGE1GB";
    }

    بازگشت "unknown";
}

spine_mem_pool_backend parse_mem_backend(ثابت نویسه * mem_backend_str) {
    اگر (mem_backend_str == nullptr || mem_backend_str[0] == '\0') {
        بازگشت spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned نویسه ch) { بازگشت static_cast<نویسه>(std::tolower(ch)); });

    اگر (value == "none") {
        بازگشت spine_mem_pool_backend::none;
    }

    اگر (value == "posix") {
        بازگشت spine_mem_pool_backend::posix_memalign;
    }

    اگر (value == "hpage") {
        بازگشت spine_mem_pool_backend::transparent_hugepage;
    }

    اگر (value == "hpage1gb") {
        بازگشت spine_mem_pool_backend::hugetlb_1g;
    }

    پرتابکن std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    رقم_cores = static_cast<صحیح>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    اگر (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        برای (صحیح i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    اگر (core_info_list.size() == 0) {
        نویسه * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        اگر (spine_core_arch_str != nullptr) {
            auto arch_id = hex_string_to_u16(spine_core_arch_str);
            برای (صحیح i = 0; i < رقم_cores; i++) {
                auto core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    اگر (core_info_list.size() == 0) {
        پرتابکن std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    نویسه * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    اگر (spine_perfer_core_arch_str != nullptr && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_to_u16(spine_perfer_core_arch_str) };
    }

    نویسه *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<صحیح> perfer_core_id_vec;
    اگر (spine_perfer_core_id_str != nullptr && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        وقتی ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(رقم_cores);
    اگر (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        برای (auto & core_info : core_info_list) {
            auto core_arch_id   = core_info.arch_id;
            auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
            اگر (core_arch_head == 0xA) {
                رقم_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_mask |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } وگرنه {
        برای (auto & core_info : core_info_list) {
            auto core_arch_id = core_info.arch_id;
            اگر (core_arch_id == perfer_core_arch_id) {
                رقم_perfer_cores++;
                cpu_mask |= (1ULL << core_info.core_id);

                auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
                اگر (core_arch_head == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        اگر (رقم_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    اگر (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_mask         = 0;
        رقم_perfer_cores = 0;
        برای (صحیح core_id : perfer_core_id_vec) {
            اگر (core_id < 0 || core_id >= رقم_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           رقم_cores - 1);
            }
            auto core_info    = core_info_list[core_id];
            auto core_arch_id = core_info.arch_id;
            اگر (core_arch_id == perfer_core_arch_id) {
                cpu_mask |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } وگرنه {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        برای (صحیح core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        رقم_perfer_cores = static_cast<صحیح>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    نویسه * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    auto   user_disable_tcm      = spine_disable_tcm_str != nullptr && strcmp(spine_disable_tcm_str, "0") != 0;

    اگر (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        اگر (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.available;
            tcm_blk_size = tcm_info.blk_size;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_size, tcm_info.blk_رقم, tcm_info.is_fake_tcm);

            برای (auto & core_info : core_info_list) {
                auto core_arch_head = (uint16_t) (core_info.arch_id) >> 12;
                اگر (core_arch_head != 0xA) {
                    aicpu_id_offset++;
                } وگرنه {
                    بشکن;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        رقم_cores, رقم_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_to_string(mem_backend), cpu_mask, aicpu_id_offset);

    ثابت size_t init_barrier_size = اندازه(spine_barrier_t) * spine_init_barrier_شمار;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_size, alignof(spine_barrier_t)));
    اگر (init_barrier != nullptr) {
        init_barrier_is_shared_mem = درست;
    } وگرنه {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = جدید spine_barrier_t[spine_init_barrier_شمار];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_شمار, 2);
}

spine_env_info::~spine_env_info() {
    اگر (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } وگرنه {
        delete[] init_barrier;
    }

    init_barrier               = nullptr;
    init_barrier_is_shared_mem = نادرست;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
