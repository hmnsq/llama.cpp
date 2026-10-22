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

    если (!file.is_open()) {
        вернуть false;
    }

    пока (std::getline(file, line)) {
        если (line.substr(0, 9) == "processor") {
            если (has_processor && has_marchid) {
                cpu_info_list.push_back({ current_processor, current_marchid });
            }

            size_t colon_pos = line.find(':');
            если (colon_pos != std::string::npos) {
                current_processor = std::stoi(line.substr(colon_pos + 1));
                has_processor     = true;
            }

            has_marchid = false;
        } иначе если (line.substr(0, 7) == "marchid") {
            size_t colon_pos = line.find(':');
            если (colon_pos != std::string::npos) {
                std::string marchid_str = line.substr(colon_pos + 1);
                marchid_str.erase(std::remove_if(marchid_str.begin(), marchid_str.end(), isspace), marchid_str.end());
                current_marchid = std::stoull(marchid_str, nullptr, 16);
                has_marchid     = true;
            }
        }
    }

    если (has_processor && has_marchid) {
        cpu_info_list.push_back({ current_processor, current_marchid });
    }

    если (has_processor && has_marchid) {
        для (auto & cpu_info : cpu_info_list) {
            если (cpu_info[0] != spine_invalid_core_id &&
                spine_march_mapping_.find(cpu_info[1]) != spine_march_mapping_.end()) {
                auto core_info    = spine_core_info();
                core_info.core_id = cpu_info[0];
                core_info.arch_id = spine_core_arch_id(spine_march_mapping_[cpu_info[1]]);

                result.push_back(core_info);
            }
        }
    }

    вернуть has_processor && has_marchid;
}

namespace {
uint16_t hex_string_to_u16(const std::string & hex_str) {
    try {
        size_t pos = 0;
        если (hex_str.substr(0, 2) == "0x" || hex_str.substr(0, 2) == "0X") {
            pos = 2;
        }
        unsigned long result = std::stoul(hex_str.substr(pos), nullptr, 16);
        если (result > std::numeric_limits<uint16_t>::max()) {
            throw std::out_of_range("Converted value is out of range for uint16_t");
        }
        вернуть static_cast<uint16_t>(result);
    } catch (const std::invalid_argument & e) {
        throw std::invalid_argument("Invalid hexadecimal string");
    } catch (const std::out_of_range & e) {
        throw;
    }
}

const char * spine_mem_pool_backend_to_string(spine_mem_pool_backend backend) {
    switch (backend) {
        case spine_mem_pool_backend::none:
            вернуть "NONE";
        case spine_mem_pool_backend::posix_memalign:
            вернуть "POSIX";
        case spine_mem_pool_backend::transparent_hugepage:
            вернуть "HPAGE";
        case spine_mem_pool_backend::hugetlb_1g:
            вернуть "HPAGE1GB";
    }

    вернуть "unknown";
}

spine_mem_pool_backend parse_mem_backend(const char * mem_backend_str) {
    если (mem_backend_str == nullptr || mem_backend_str[0] == '\0') {
        вернуть spine_mem_pool_backend::transparent_hugepage;
    }

    std::string value(mem_backend_str);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { вернуть static_cast<char>(std::tolower(ch)); });

    если (value == "none") {
        вернуть spine_mem_pool_backend::none;
    }

    если (value == "posix") {
        вернуть spine_mem_pool_backend::posix_memalign;
    }

    если (value == "hpage") {
        вернуть spine_mem_pool_backend::transparent_hugepage;
    }

    если (value == "hpage1gb") {
        вернуть spine_mem_pool_backend::hugetlb_1g;
    }

    throw std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + value + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
}  // namespace

spine_env_info::spine_env_info() {
    num_cores = static_cast<целое>(std::thread::hardware_concurrency());
    spine_core_info::get_spine_core_info(core_info_list);

    // special for x60 K1
    если (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
        для (целое i = 0; i < 4; i++) {
            core_info_list[i].arch_id = spine_core_arch_id::core_arch_a60;
        }
    }

    // special for qemu
    если (core_info_list.size() == 0) {
        char * spine_core_arch_str = getenv("SPACEMIT_CORE_ARCH");
        если (spine_core_arch_str != nullptr) {
            auto arch_id = hex_string_to_u16(spine_core_arch_str);
            для (целое i = 0; i < num_cores; i++) {
                auto core_info    = spine_core_info();
                core_info.core_id = i;
                core_info.arch_id = spine_core_arch_id{ arch_id };
                core_info_list.push_back(core_info);
            }
        }
    }

    если (core_info_list.size() == 0) {
        throw std::runtime_error(
            "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
    }

    char * spine_perfer_core_arch_str = getenv("SPACEMIT_PERFER_CORE_ARCH");
    если (spine_perfer_core_arch_str != nullptr && spine_perfer_core_arch_str != "") {
        perfer_core_arch_id = spine_core_arch_id{ hex_string_to_u16(spine_perfer_core_arch_str) };
    }

    char *           spine_perfer_core_id_str = getenv("SPACEMIT_PERFER_CORE_ID");
    std::vector<целое> perfer_core_id_vec;
    если (spine_perfer_core_id_str != nullptr && spine_perfer_core_id_str != "") {
        std::string perfer_core_id_str(spine_perfer_core_id_str);
        size_t      start = 0;
        size_t      end   = 0;
        пока ((end = perfer_core_id_str.find(',', start)) != std::string::npos) {
            std::string core_id_substr = perfer_core_id_str.substr(start, end - start);
            perfer_core_id_vec.push_back(std::stoi(core_id_substr));
            start = end + 1;
        }
        std::string core_id_substr = perfer_core_id_str.substr(start);
        perfer_core_id_vec.push_back(std::stoi(core_id_substr));
    }

    perfer_core_ids.reserve(num_cores);
    если (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
        для (auto & core_info : core_info_list) {
            auto core_arch_id   = core_info.arch_id;
            auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
            если (core_arch_head == 0xA) {
                num_perfer_cores++;
                perfer_core_arch_id = core_arch_id;
                cpu_mask |= (1ULL << core_info.core_id);
                perfer_core_ids.push_back(core_info.core_id);
            }
        }
    } иначе {
        для (auto & core_info : core_info_list) {
            auto core_arch_id = core_info.arch_id;
            если (core_arch_id == perfer_core_arch_id) {
                num_perfer_cores++;
                cpu_mask |= (1ULL << core_info.core_id);

                auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
                если (core_arch_head == 0xA) {
                    perfer_core_ids.push_back(core_info.core_id);
                }
            }
        }
        если (num_perfer_cores == 0) {
            GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                       (uint16_t) perfer_core_arch_id);
        }
    }

    если (perfer_core_id_vec.size() > 0) {
        perfer_core_ids.clear();
        cpu_mask         = 0;
        num_perfer_cores = 0;
        для (целое core_id : perfer_core_id_vec) {
            если (core_id < 0 || core_id >= num_cores) {
                GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", core_id,
                           num_cores - 1);
            }
            auto core_info    = core_info_list[core_id];
            auto core_arch_id = core_info.arch_id;
            если (core_arch_id == perfer_core_arch_id) {
                cpu_mask |= (1ULL << core_id);
                perfer_core_ids.push_back(core_id);
            } иначе {
                GGML_ABORT(
                    "core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                    "SPACEMIT_PERFER_CORE_ARCH %x\n",
                    core_id, (uint16_t) core_arch_id, (uint16_t) perfer_core_arch_id);
            }
        }
        std::string perfer_core_id_vec_str;
        для (целое core_id : perfer_core_id_vec) {
            perfer_core_id_vec_str += std::to_string(core_id) + ",";
        }
        perfer_core_id_vec_str.pop_back();
        GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", perfer_core_id_vec_str.c_str());
        num_perfer_cores = static_cast<целое>(perfer_core_id_vec.size());
    }

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    mem_backend                  = parse_mem_backend(getenv("SPACEMIT_MEM_BACKEND"));
    char * spine_disable_tcm_str = getenv("SPACEMIT_DISABLE_TCM");
    auto   user_disable_tcm      = spine_disable_tcm_str != nullptr && strcmp(spine_disable_tcm_str, "0") != 0;

    если (!user_disable_tcm) {
        spine_mem_pool_tcm_info tcm_info;
        если (spine_mem_pool_tcm_init(&tcm_info)) {
            use_tcm      = tcm_info.available;
            tcm_blk_size = tcm_info.blk_size;
            GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                           tcm_info.blk_size, tcm_info.blk_num, tcm_info.is_fake_tcm);

            для (auto & core_info : core_info_list) {
                auto core_arch_head = (uint16_t) (core_info.arch_id) >> 12;
                если (core_arch_head != 0xA) {
                    aicpu_id_offset++;
                } иначе {
                    break;
                }
            }
        }
    }

    GGML_LOG_DEBUG(
        "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
        "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
        num_cores, num_perfer_cores, (uint16_t) perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
        spine_mem_pool_backend_to_string(mem_backend), cpu_mask, aicpu_id_offset);

    const size_t init_barrier_size = sizeof(spine_barrier_t) * spine_init_barrier_count;
    init_barrier =
        static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(init_barrier_size, alignof(spine_barrier_t)));
    если (init_barrier != nullptr) {
        init_barrier_is_shared_mem = true;
    } иначе {
        GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                      __func__);
        init_barrier = new spine_barrier_t[spine_init_barrier_count];
    }

    spine_barrier_init(init_barrier, spine_init_barrier_count, 2);
}

spine_env_info::~spine_env_info() {
    если (init_barrier_is_shared_mem) {
        spine_mem_pool_shared_mem_free(init_barrier);
    } иначе {
        delete[] init_barrier;
    }

    init_barrier               = nullptr;
    init_barrier_is_shared_mem = false;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit
