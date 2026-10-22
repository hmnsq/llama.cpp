#అన్తర్భావ "ime_env.h"

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "spine_mem_pool.h"

#అన్తర్భావ <fcntl.h>
#అన్తర్భావ <unistd.h>

#అన్తర్భావ <algorithm>
#అన్తర్భావ <array>
#అన్తర్భావ <cctype>
#అన్తర్భావ <fstream>
#అన్తర్భావ <string>
#అన్తర్భావ <thread>
#అన్తర్భావ <unordered_map>

నామాకాశ ggml::cpu::riscv64_spacemit {
తార్కిక spine_core_info::get_spine_core_info(std::vector<spine_core_info> & ఫలితం) {
    స్థిర std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    ఫలితం.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   పంక్తి;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    తార్కిక     has_processor     = అబద్ధం;
    తార్కిక     has_marchid       = అబద్ధం;

    ఒకవేళ (!file.is_open()) {
        తిరిగి అబద్ధం;
    }

    అయితే (std::getline(file, పంక్తి)) {
        ఒకవేళ (పంక్తి.substr(0, 9) == "processor") {
            ఒకవేళ (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = పంక్తి.find(':');
            ఒకవేళ (colon_pos != std::string::npos) {
                current_processor = std::stoi(పంక్తి.substr(colon_pos + 1));
                has_processor     = నిజం;
            }

            has_marchid = అబద్ధం;
        } లేకపోతే ఒకవేళ (పంక్తి.substr(0, 7) == "marchid") {
            size_t colon_pos = పంక్తి.find(':');
            ఒకవేళ (colon_pos != std::string::npos) {
                std::string marchid_str = పంక్తి.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), ఖాళీనా), marchid_str.end());
                current_marchid = std::stoull(marchid_str, శూన్యసూచిక, 16);
                has_marchid     = నిజం;
            }
        }
    }

    ఒకవేళ (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    ఒకవేళ (has_processor && has_marchid) {
        కొరకు (స్వయంచాలక & cpu_info : cpu_info_list) {
            ఒకవేళ (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                స్వయంచాలక core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                ఫలితం.push_back(core_info);
            }
        }
    }

    తిరిగి has_processor && has_marchid;
}

నామాకాశ {
uint16_t hex_string_గమ్యం_u16(స్థిరరాశి std::string & hex_str) {
    ప్రయత్నించు {
        size_t pos = 0;
        ఒకవేళ (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        గుర్తులేని దీర్ఘ ఫలితం = std::stoul(hex_str.substr(pos), శూన్యసూచిక, 16);
        ఒకవేళ (ఫలితం > std::numeric_limits<uint16_t>::max()) {
            విసురు std::out_of_range("Converted value is out of range for uint16_t");
        }
        తిరిగి static_cast<uint16_t>(ఫలితం);
    } పట్టుకో (స్థిరరాశి std::invalid_argument & e) {
        విసురు std::invalid_argument("Invalid hexadecimal string");
    } పట్టుకో (స్థిరరాశి std::out_of_range & e) {
        విసురు;
    }
}

స్థిరరాశి అక్షరం * spine_mem_pool_backend_గమ్యం_string(spine_mem_pool_backend backend) {
    ఎంపిక (backend) {
        స్థితి spine_mem_pool_backend::none:
            తిరిగి "NONE";
        స్థితి spine_mem_pool_backend::posix_memalign:
            తిరిగి "POSIX";
        స్థితి spine_mem_pool_backend::transparent_hugepage:
            తిరిగి "HPAGE";
        స్థితి spine_mem_pool_backend::hugetlb_1g:
            తిరిగి "HPAGE1GB";
    }

    తిరిగి "unknown";
}

spine_mem_pool_backend parse_mem_backend(స్థిరరాశి అక్షరం * mem_backend_str) {
    ఒకవేళ (mem_backend_str == శూన్యసూచిక || mem_backend_str[0] == '\0') {
        తిరిగి spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](గుర్తులేని అక్షరం ch) { తిరిగి static_cast<అక్షరం>(std::tolower(ch)); });

    ఒకవేళ (value == "none") {
        తిరిగి spine_mem_pool_backend::none;
    }

    ఒకవేళ (value == "posix") {
        తిరిగి spine_mem_pool_backend::posix_memalign;
    }

    ఒకవేళ (value == "hpage") {
        తిరిగి spine_mem_pool_backend::transparent_hugepage;
    }

    ఒకవేళ (value == "hpage1gb") {
        తిరిగి spine_mem_pool_backend::hugetlb_1g;
    }

    విసురు std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    అంకె_cores = static_cast<పూర్ణాంకం>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    ఒకవేళ (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    ఒకవేళ (core_info_list.size() == 0) {
        అక్షరం * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        ఒకవేళ (spine_core_arch_str != శూన్యసూచిక) {
            స్వయంచాలక arch_id = hex_string_గమ్యం_u16(spine_core_arch_str);
            కొరకు (పూర్ణాంకం i = 0; i < అంకె_cores; i++) {
                స్వయంచాలక core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    ఒకవేళ (core_info_list.size() == 0) {
        విసురు std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    అక్షరం * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    ఒకవేళ (spine_perfer_core_arch_str != శూన్యసూచిక && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_గమ్యం_u16(spine_perfer_core_arch_str) };
    }

    అక్షరం *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<పూర్ణాంకం> perfer_core_id_vec;
    ఒకవేళ (spine_perfer_core_id_str != శూన్యసూచిక && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        అయితే ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(అంకె_cores);
    ఒకవేళ (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        కొరకు (స్వయంచాలక & core_info : core_info_list) {
            స్వయంచాలక core_arch_id   = core_info.arch_id;
            స్వయంచాలక core_arch_తల = (uint16_t) (core_arch_id) >> 12;
            ఒకవేళ (core_arch_తల == 0xA) {
                అంకె_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_ముసుగు |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } లేకపోతే {
        కొరకు (స్వయంచాలక & core_info : core_info_list) {
            స్వయంచాలక core_arch_id = core_info.arch_id;
            ఒకవేళ (core_arch_id == perfer_core_arch_id) {
                అంకె_perfer_cores++;
                cpu_ముసుగు |= (1ULL << core_info.core_id);

                స్వయంచాలక core_arch_తల = (uint16_t) (core_arch_id) >> 12;
                ఒకవేళ (core_arch_తల == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        ఒకవేళ (అంకె_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    ఒకవేళ (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_ముసుగు         = 0;
        అంకె_perfer_cores = 0;
        కొరకు (పూర్ణాంకం core_id : perfer_core_id_vec) {
            ఒకవేళ (core_id < 0 || core_id >= అంకె_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           అంకె_cores - 1);
            }
            స్వయంచాలక core_info    = core_info_list[core_id];
            స్వయంచాలక core_arch_id = core_info.arch_id;
            ఒకవేళ (core_arch_id == perfer_core_arch_id) {
                cpu_ముసుగు |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } లేకపోతే {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        కొరకు (పూర్ణాంకం core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        అంకె_perfer_cores = static_cast<పూర్ణాంకం>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    అక్షరం * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    స్వయంచాలక   user_disable_tcm      = spine_disable_tcm_str != శూన్యసూచిక && strcmp(spine_disable_tcm_str, "0") != 0;

    ఒకవేళ (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        ఒకవేళ (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.అందుబాటు;
            tcm_blk_కొలత = tcm_info.blk_కొలత;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_కొలత, tcm_info.blk_అంకె, tcm_info.is_fake_tcm);

            కొరకు (స్వయంచాలక & core_info : core_info_list) {
                స్వయంచాలక core_arch_తల = (uint16_t) (core_info.arch_id) >> 12;
                ఒకవేళ (core_arch_తల != 0xA) {
                    aicpu_id_స్థానభ్రంశం++;
                } లేకపోతే {
                    ఆపు;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        అంకె_cores, అంకె_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_గమ్యం_string(mem_backend), cpu_ముసుగు, aicpu_id_స్థానభ్రంశం);

    స్థిరరాశి size_t init_barrier_కొలత = పరిమాణం(spine_barrier_t) * spine_init_barrier_లెక్క;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_కొలత, alignof(spine_barrier_t)));
    ఒకవేళ (init_barrier != శూన్యసూచిక) {
        init_barrier_is_shared_mem = నిజం;
    } లేకపోతే {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = కొత్త spine_barrier_t[spine_init_barrier_లెక్క];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_లెక్క, 2);
}

spine_env_info::~spine_env_info() {
    ఒకవేళ (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } లేకపోతే {
        తొలగించు[] init_barrier;
    }

    init_barrier               = శూన్యసూచిక;
    init_barrier_is_shared_mem = అబద్ధం;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
