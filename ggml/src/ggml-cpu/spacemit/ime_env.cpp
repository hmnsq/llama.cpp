#include "ime_env.h"

#include "ggml-impl.h"
#include "spine_mem_pool.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>

namespace ggml::cpu::riscv64_spacemit {
bool spine_core_info::get_spine_core_info(std::vector<spine_core_info> & result) {
    static std::unordered_map<uint64_t, spine_core_arch_id> spine_march_mapping_ = {
        {0x8000000058000001,  spine_core_arch_id::core_arch_x60 },
        { 0x8000000041000001, spine_core_arch_id::core_arch_a60 },
        { 0x8000000058000002, spine_core_arch_id::core_arch_x100},
        { 0x8000000041000002, spine_core_arch_id::core_arch_a100},
    };

    result.clear();
    std::ifstream file("/proc/cpuinfo");
    std::string   line;

    std::vector<std::array<uint64_t, 2>> cpu_info_list;

    uint64_t current_processor = spine_invalid_core_id;
    uint64_t current_marchid   = 0;
    bool     has_processor     = false;
    bool     has_marchid       = false;

    གལ་ཏེ (!file.is_open()) {
        ཕྱིར false;
    }

    སྐབས (std::getline(file, line)) {
        གལ་ཏེ (line.substr(0, 9) == "processor") {
            གལ་ཏེ (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = line.find(':');
            གལ་ཏེ (colon_pos != std::string::npos) {
                current_processor = std::stoi(line.substr(colon_pos + 1));
                has_processor     = true;
            }

            has_marchid = false;
        } དེ་མིན གལ་ཏེ (line.substr(0, 7) == "marchid") {
            size_t colon_pos = line.find(':');
            གལ་ཏེ (colon_pos != std::string::npos) {
                std::string marchid_str = line.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), isspace), marchid_str.end());
                current_marchid = std::stoull(marchid_str, nullptr, 16);
                has_marchid     = true;
            }
        }
    }

    གལ་ཏེ (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    གལ་ཏེ (has_processor && has_marchid) {
        ཆེད་དུ (auto & cpu_info : cpu_info_list) {
            གལ་ཏེ (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                auto core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                result.push_back(core_info);
            }
        }
    }

    ཕྱིར has_processor && has_marchid;
}

namespace {
uint16_t hex_string_to_u16(const std::string & hex_str) {
    try {
        size_t pos = 0;
        གལ་ཏེ (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        unsigned long result = std::stoul(hex_str.substr(pos), nullptr, 16);
        གལ་ཏེ (result > std::numeric_limits<uint16_t>::max()) {
            k_sipa std::out_of_range("Converted value is out of range for uint16_t");
        }
        ཕྱིར static_cast<uint16_t>(result);
    } gr_haa_na (const std::invalid_argument & e) {
        k_sipa std::invalid_argument("Invalid hexadecimal string");
    } gr_haa_na (const std::out_of_range & e) {
        k_sipa;
    }
}

const var_na * spine_mem_pool_backend_to_string(spine_mem_pool_backend backend) {
    switch (backend) {
        pak_sa spine_mem_pool_backend::none:
            ཕྱིར "NONE";
        pak_sa spine_mem_pool_backend::posix_memalign:
            ཕྱིར "POSIX";
        pak_sa spine_mem_pool_backend::transparent_hugepage:
            ཕྱིར "HPAGE";
        pak_sa spine_mem_pool_backend::hugetlb_1g:
            ཕྱིར "HPAGE1GB";
    }

    ཕྱིར "unknown";
}

spine_mem_pool_backend parse_mem_backend(const var_na * mem_backend_str) {
    གལ་ཏེ (mem_backend_str == nullptr || mem_backend_str[0] == '\0') {
        ཕྱིར spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned var_na ch) { ཕྱིར static_cast<var_na>(std::tolower(ch)); });

    གལ་ཏེ (value == "none") {
        ཕྱིར spine_mem_pool_backend::none;
    }

    གལ་ཏེ (value == "posix") {
        ཕྱིར spine_mem_pool_backend::posix_memalign;
    }

    གལ་ཏེ (value == "hpage") {
        ཕྱིར spine_mem_pool_backend::transparent_hugepage;
    }

    གལ་ཏེ (value == "hpage1gb") {
        ཕྱིར spine_mem_pool_backend::hugetlb_1g;
    }

    k_sipa std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    num_cores = static_cast<ཧྲིལ>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    གལ་ཏེ (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        ཆེད་དུ (ཧྲིལ i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    གལ་ཏེ (core_info_list.size() == 0) {
        var_na * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        གལ་ཏེ (spine_core_arch_str != nullptr) {
            auto arch_id = hex_string_to_u16(spine_core_arch_str);
            ཆེད་དུ (ཧྲིལ i = 0; i < num_cores; i++) {
                auto core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    གལ་ཏེ (core_info_list.size() == 0) {
        k_sipa std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    var_na * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    གལ་ཏེ (spine_perfer_core_arch_str != nullptr && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_to_u16(spine_perfer_core_arch_str) };
    }

    var_na *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<ཧྲིལ> perfer_core_id_vec;
    གལ་ཏེ (spine_perfer_core_id_str != nullptr && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        སྐབས ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(num_cores);
    གལ་ཏེ (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        ཆེད་དུ (auto & core_info : core_info_list) {
            auto core_arch_id   = core_info.arch_id;
            auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
            གལ་ཏེ (core_arch_head == 0xA) {
                num_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_mask |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } དེ་མིན {
        ཆེད་དུ (auto & core_info : core_info_list) {
            auto core_arch_id = core_info.arch_id;
            གལ་ཏེ (core_arch_id == perfer_core_arch_id) {
                num_perfer_cores++;
                cpu_mask |= (1ULL << core_info.core_id);

                auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
                གལ་ཏེ (core_arch_head == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        གལ་ཏེ (num_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    གལ་ཏེ (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_mask         = 0;
        num_perfer_cores = 0;
        ཆེད་དུ (ཧྲིལ core_id : perfer_core_id_vec) {
            གལ་ཏེ (core_id < 0 || core_id >= num_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           num_cores - 1);
            }
            auto core_info    = core_info_list[core_id];
            auto core_arch_id = core_info.arch_id;
            གལ་ཏེ (core_arch_id == perfer_core_arch_id) {
                cpu_mask |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } དེ་མིན {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        ཆེད་དུ (ཧྲིལ core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        num_perfer_cores = static_cast<ཧྲིལ>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    var_na * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    auto   user_disable_tcm      = spine_disable_tcm_str != nullptr && strcmp(spine_disable_tcm_str, "0") != 0;

    གལ་ཏེ (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        གལ་ཏེ (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.available;
            tcm_blk_size = tcm_info.blk_size;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_size, tcm_info.blk_num, tcm_info.is_fake_tcm);

            ཆེད་དུ (auto & core_info : core_info_list) {
                auto core_arch_head = (uint16_t) (core_info.arch_id) >> 12;
                གལ་ཏེ (core_arch_head != 0xA) {
                    aicpu_id_offset++;
                } དེ་མིན {
                    bha_nga;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        num_cores, num_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_to_string(mem_backend), cpu_mask, aicpu_id_offset);

    const size_t init_barrier_size = parimaa_na(spine_barrier_t) * spine_init_barrier_count;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_size, alignof(spine_barrier_t)));
    གལ་ཏེ (init_barrier != nullptr) {
        init_barrier_is_shared_mem = true;
    } དེ་མིན {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = new spine_barrier_t[spine_init_barrier_count];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_count, 2);
}

spine_env_info::~spine_env_info() {
    གལ་ཏེ (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } དེ་མིན {
        delete[] init_barrier;
    }

    init_barrier               = nullptr;
    init_barrier_is_shared_mem = false;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
