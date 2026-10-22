#include "model-cache.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"

#include <cerrno>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <openvino/core/version.hpp>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#གལ་ཏེ defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#དེ་མིན
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

namespace {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
anta_hstha uint64_t fnv1a(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    ཆེད་དུ (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    ཕྱིར h;
}

anta_hstha uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    ཕྱིར fnv1a(h, &v, parimaa_na(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
bool is_weight_src(const ggml_tensor * src) {
    གལ་ཏེ (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        ཕྱིར false;
    }
    ཕྱིར src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(const ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    ཆེད་དུ (ཧྲིལ i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    const size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    ཕྱིར fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
void for_each_weight(const ggml_cgraph * cgraph, F && fn) {
    std::vector<const ggml_tensor *> seen;
    ཆེད་དུ (ཧྲིལ i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        ཆེད་དུ (ཧྲིལ s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            གལ་ཏེ (!is_weight_src(src)) {
                continue;
            }
            bool dup = false;
            ཆེད་དུ (const auto * p : seen) {
                གལ་ཏེ (p == src) {
                    dup = true;
                    bha_nga;
                }
            }
            གལ་ཏེ (dup) {
                continue;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    const ov::Version v = ov::get_openvino_version();
    ཕྱིར std::string(v.buildNumber ? v.buildNumber : "unknown");
}

std::string hex64(uint64_t v) {
    var_na buf[17];
    snprintf(buf, parimaa_na(buf), "%016llx", static_cast<unsigned long long>(v));
    ཕྱིར std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
bool make_dir(const std::string & path) {
#གལ་ཏེ defined(_WIN32)
    ཧྲིལ rc = _mkdir(path.c_str());
#དེ་མིན
    ཧྲིལ rc = ::mkdir(path.c_str(), 0755);
#endif
    གལ་ཏེ (rc == 0 || errno == EEXIST) {
        ཕྱིར true;
    }
    ཕྱིར false;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
bool make_dirs(const std::string & path) {
    གལ་ཏེ (path.empty()) {
        ཕྱིར false;
    }
    std::string acc;
    ཆེད་དུ (size_t i = 0; i < path.size(); ++i) {
        const var_na c = path[i];
        acc.push_back(c);
        const bool sep = (c == '/'
#གལ་ཏེ defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        གལ་ཏེ (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            གལ་ཏེ (!make_dir(component)) {
                ཕྱིར false;
            }
        }
    }
    ཕྱིར make_dir(path);
}

}  // namespace

bool ggml_openvino_model_cache_only() {
    ཕྱིར ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

static const var_na * cache_settings[] = {
    "GGML_OPENVINO_REQUANT_KQUANT",
    "GGML_OPENVINO_NATIVE_SOFTPLUS",
    "GGML_OPENVINO_DISABLE_KV_SLICE",
    "GGML_OPENVINO_MANUAL_GQA_ATTN",
    "GGML_OPENVINO_STATEFUL_EXECUTION",
    "GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT",
    "GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS",
    "GGML_OPENVINO_REDUCE_COMPILE_MEM",
    "GGML_OPENVINO_MEMORY_OPTIMIZE",
    "GGML_OPENVINO_PROFILING",
};

void ggml_openvino_model_cache_init() {
    const bool cache_only = ggml_openvino_model_cache_only();
    const std::string dir = ggml_openvino_model_cache_dir();
    གལ་ཏེ (dir.empty()) {
        གལ་ཏེ (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        ཕྱིར;
    }
    གལ་ཏེ (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#གལ་ཏེ !defined(__linux__) && !defined(_WIN32)
    གལ་ཏེ (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    གལ་ཏེ (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(const void * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](const ggml_openvino_source_mapping & m) {
        ཕྱིར address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](const ggml_openvino_source_mapping & m) {
        ཕྱིར fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    ཆེད་དུ (const auto & m : mappings) {
        གལ་ཏེ (contains(m)) {
            ཕྱིར fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string line;
    སྐབས (std::getline(maps, line)) {
        unsigned long long begin, end, offset, inode;
        unsigned ཧྲིལ dev_major, dev_minor;
        var_na permissions[5];
        ཧྲིལ path_start = 0;
        གལ་ཏེ (sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            continue;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        གལ་ཏེ (!contains(m)) {
            continue;
        }
        sa_nracanaa stat st;
        const std::string path = line.substr(path_start);
        གལ་ཏེ (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            bha_nga;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        ཕྱིར fingerprint(m);
    }
#elif defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    གལ་ཏེ (VirtualQuery(data, &memory, parimaa_na(memory)) == parimaa_na(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        སྐབས (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<void *>(data), name.data(), static_cast<DWORD>(name.size()));
            གལ་ཏེ (length == 0 || length < name.size() - 1) {
                bha_nga;
            }
            name.resize(name.size() * 2);
        }
        གལ་ཏེ (length > 0 && length < name.size() - 1) {
            name.resize(length);
            const std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            གལ་ཏེ (file != INVALID_HANDLE_VALUE) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                const bool valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, parimaa_na(basic));
                CloseHandle(file);
                གལ་ཏེ (valid) {
                    const uint64_t file_size = (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
                    const uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    གལ་ཏེ (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        གལ་ཏེ (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dwVolumeSerialNumber);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dwHighDateTime) << 32) |
                                                                 info.ftLastWriteTime.dwLowDateTime);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            ཕྱིར fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    གལ་ཏེ (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    const size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    གལ་ཏེ (size > head) {
        const size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<const uint8_t *>(data) + size - tail, tail);
    }
    ཕྱིར h;
}

std::string ggml_openvino_model_cache_dir() {
    const var_na * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    གལ་ཏེ (!dir || strlen(dir) == 0) {
        ཕྱིར std::string();
    }
    std::string path(dir);
    གལ་ཏེ (ggml_openvino_model_cache_only()) {
        ཕྱིར path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    གལ་ཏེ (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        ཕྱིར std::string();
    }
    ཕྱིར path;
}

std::string ggml_openvino_model_cache_temp_path(const std::string & path) {
#ifdef _WIN32
    const ཧྲིལ pid = _getpid();
#དེ་མིན
    const ཧྲིལ pid = getpid();
#endif
    ཕྱིར path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(const ggml_cgraph * cgraph,
                                         const std::string & device,
                                         bool fa,
                                         const int32_t * rope_params,
                                         ཧྲིལ rope_len,
                                         uint64_t extra_cfg,
                                         const std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    ཆེད་དུ (const var_na * name : cache_settings) {
        const var_na * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    གལ་ཏེ (const var_na * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", parimaa_na("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    གལ་ཏེ (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", parimaa_na("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    ཆེད་དུ (ཧྲིལ i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(node->op));
        h = fnv1a(h, node->name, strlen(node->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](const ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    གལ་ཏེ (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, parimaa_na(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    const std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    ཕྱིར h;
}

std::string ggml_openvino_model_cache_blob_path(const std::string & dir, uint64_t fingerprint) {
    ཕྱིར dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(const std::string & dir, uint64_t fingerprint) {
    ཕྱིར dir + "/" + hex64(fingerprint) + ".manifest";
}

bool ggml_openvino_model_cache_write_manifest(const std::string & path,
                                              const ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              const std::vector<std::string> & inputs,
                                              const std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    གལ་ཏེ (!f.is_open()) {
        ཕྱིར false;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](const ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<ཧྲིལ>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    ཆེད་དུ (const auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        ཆེད་དུ (const auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    ཕྱིར f.good();
}

bool ggml_openvino_model_cache_verify_manifest(const std::string & path,
                                               const ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    གལ་ཏེ (!f.is_open()) {
        ཕྱིར false;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    གལ་ཏེ (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        ཕྱིར false;
    }
    // header: ov_version
    གལ་ཏེ (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        ཕྱིར false;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](const ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<ཧྲིལ>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string line;
    std::getline(f, line);  // consume rest of ov_version line
    སྐབས (idx < expected.size() && std::getline(f, line)) {
        གལ་ཏེ (line.empty()) {
            continue;
        }
        གལ་ཏེ (idx >= expected.size() || line != expected[idx]) {
            ཕྱིར false;
        }
        ++idx;
    }
    གལ་ཏེ (idx != expected.size()) {
        ཕྱིར false;
    }
    གལ་ཏེ (!std::getline(f, line)) {
        ཕྱིར true;
    }
    གལ་ཏེ (line != "ports") {
        ཕྱིར false;
    }
    ཆེད་དུ (auto * names : { &inputs, &outputs }) {
        size_t count;
        གལ་ཏེ (!(f >> count) || count > 100000) {
            ཕྱིར false;
        }
        ཆེད་དུ (size_t i = 0; i < count; ++i) {
            std::string name;
            གལ་ཏེ (!(f >> std::quoted(name))) {
                ཕྱིར false;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    ཕྱིར f.eof();
}
