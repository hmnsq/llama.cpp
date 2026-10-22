#ಅನ್ತರ್ಭಾವ "model-cache.h"

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino-extra.h"

#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <iomanip>
#ಅನ್ತರ್ಭಾವ <limits>
#ಅನ್ತರ್ಭಾವ <sstream>
#ಅನ್ತರ್ಭಾವ <openvino/core/version.hpp>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <vector>

#ಯದಿ defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#ಅನ್ಯಥಾ
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

ನಾಮಾಕಾಶ {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
ಅನ್ತಃಸ್ಥ uint64_t fnv1a(uint64_t h, ಧ್ರುವ ಶೂನ್ಯ * data, size_t n) {
    ಧ್ರುವ uint8_t * p = static_cast<ಧ್ರುವ uint8_t *>(data);
    ಪ್ರತಿ (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    ಪ್ರತ್ಯಾಗಮ h;
}

ಅನ್ತಃಸ್ಥ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    ಪ್ರತ್ಯಾಗಮ fnv1a(h, &v, ಪರಿಮಾಣ(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
ತಾರ್ಕಿಕ is_weight_src(ಧ್ರುವ ggml_tensor * src) {
    ಯದಿ (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(ಧ್ರುವ ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ಧ್ರುವ size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    ಪ್ರತ್ಯಾಗಮ fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
ಶೂನ್ಯ for_each_weight(ಧ್ರುವ ggml_cgraph * cgraph, F && fn) {
    std::vector<ಧ್ರುವ ggml_tensor *> seen;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; ++i) {
        ಧ್ರುವ ggml_tensor * ಗಂಟು = cgraph->nodes[i];
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < GGML_MAX_SRC; ++s) {
            ಧ್ರುವ ggml_tensor * src = ಗಂಟು->src[s];
            ಯದಿ (!is_weight_src(src)) {
                ಅನುವರ್ತ;
            }
            ತಾರ್ಕಿಕ dup = ಅಸತ್ಯ;
            ಪ್ರತಿ (ಧ್ರುವ auto * p : seen) {
                ಯದಿ (p == src) {
                    dup = ಸತ್ಯ;
                    ಭಙ್ಗ;
                }
            }
            ಯದಿ (dup) {
                ಅನುವರ್ತ;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    ಧ್ರುವ ov::Version v = ov::get_openvino_version();
    ಪ್ರತ್ಯಾಗಮ std::string(v.build_ಸಂಖ್ಯೆ ? v.build_ಸಂಖ್ಯೆ : "unknown");
}

std::string hex64(uint64_t v) {
    ವರ್ಣ buf[17];
    snprintf(buf, ಪರಿಮಾಣ(buf), "%016llx", static_cast<unsigned long long>(v));
    ಪ್ರತ್ಯಾಗಮ std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
ತಾರ್ಕಿಕ make_dir(ಧ್ರುವ std::string & path) {
#ಯದಿ defined(_WIN32)
    ಪೂರ್ಣಾಙ್ಕ rc = _mkdir(path.c_str());
#ಅನ್ಯಥಾ
    ಪೂರ್ಣಾಙ್ಕ rc = ::mkdir(path.c_str(), 0755);
#endif
    ಯದಿ (rc == 0 || errno == EEXIST) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
ತಾರ್ಕಿಕ make_dirs(ಧ್ರುವ std::string & path) {
    ಯದಿ (path.empty()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    std::string acc;
    ಪ್ರತಿ (size_t i = 0; i < path.size(); ++i) {
        ಧ್ರುವ ವರ್ಣ c = path[i];
        acc.push_back(c);
        ಧ್ರುವ ತಾರ್ಕಿಕ sep = (c == '/'
#ಯದಿ defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        ಯದಿ (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            ಯದಿ (!make_dir(component)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ make_dir(path);
}

}  // namespace

ತಾರ್ಕಿಕ ggml_openvino_model_cache_only() {
    ಪ್ರತ್ಯಾಗಮ ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * cache_settings[] = {
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

ಶೂನ್ಯ ggml_openvino_model_cache_init() {
    ಧ್ರುವ ತಾರ್ಕಿಕ cache_only = ggml_openvino_model_cache_only();
    ಧ್ರುವ std::string dir = ggml_openvino_model_cache_dir();
    ಯದಿ (dir.empty()) {
        ಯದಿ (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯದಿ (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#ಯದಿ !defined(__linux__) && !defined(_WIN32)
    ಯದಿ (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    ಯದಿ (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(ಧ್ರುವ ಶೂನ್ಯ * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    ಧ್ರುವ uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](ಧ್ರುವ ggml_openvino_source_mapping & m) {
        ಪ್ರತ್ಯಾಗಮ address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](ಧ್ರುವ ggml_openvino_source_mapping & m) {
        ಪ್ರತ್ಯಾಗಮ fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    ಪ್ರತಿ (ಧ್ರುವ auto & m : mappings) {
        ಯದಿ (contains(m)) {
            ಪ್ರತ್ಯಾಗಮ fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string ಸಾಲು;
    ಯಾವತ್ (std::getline(maps, ಸಾಲು)) {
        unsigned long long begin, end, offset, inode;
        unsigned ಪೂರ್ಣಾಙ್ಕ dev_major, dev_minor;
        ವರ್ಣ permissions[5];
        ಪೂರ್ಣಾಙ್ಕ path_start = 0;
        ಯದಿ (sscanf(ಸಾಲು.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            ಅನುವರ್ತ;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        ಯದಿ (!contains(m)) {
            ಅನುವರ್ತ;
        }
        ಸಂರಚನಾ stat st;
        ಧ್ರುವ std::string path = ಸಾಲು.substr(path_start);
        ಯದಿ (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            ಭಙ್ಗ;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        ಪ್ರತ್ಯಾಗಮ fingerprint(m);
    }
#ನೋಚೇದ್ಯದಿ defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    ಯದಿ (VirtualQuery(data, &memory, ಪರಿಮಾಣ(memory)) == ಪರಿಮಾಣ(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        ಯಾವತ್ (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<ಶೂನ್ಯ *>(data), name.data(), static_cast<DWORD>(name.size()));
            ಯದಿ (length == 0 || length < name.size() - 1) {
                ಭಙ್ಗ;
            }
            name.resize(name.size() * 2);
        }
        ಯದಿ (length > 0 && length < name.size() - 1) {
            name.resize(length);
            ಧ್ರುವ std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            ಯದಿ (file != INVALID_HANDLE_ಮೌಲ್ಯ) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ಧ್ರುವ ತಾರ್ಕಿಕ valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, ಪರಿಮಾಣ(basic));
                CloseHandle(file);
                ಯದಿ (valid) {
                    ಧ್ರುವ uint64_t file_size = (uint64_t(info.n_File_Size_ಮೇಲಿನ) << 32) | info.n_File_Size_ಕೆಳಗಿನ;
                    ಧ್ರುವ uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    ಯದಿ (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        ಯದಿ (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_ಸಂಖ್ಯೆ);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_Index_ಮೇಲಿನ) << 32) | info.n_File_Index_ಕೆಳಗಿನ);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_ಮೇಲಿನ_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_ಕೆಳಗಿನ_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            ಪ್ರತ್ಯಾಗಮ fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    ಯದಿ (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    ಧ್ರುವ size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    ಯದಿ (size > head) {
        ಧ್ರುವ size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<ಧ್ರುವ uint8_t *>(data) + size - tail, tail);
    }
    ಪ್ರತ್ಯಾಗಮ h;
}

std::string ggml_openvino_model_cache_dir() {
    ಧ್ರುವ ವರ್ಣ * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    ಯದಿ (!dir || strlen(dir) == 0) {
        ಪ್ರತ್ಯಾಗಮ std::string();
    }
    std::string path(dir);
    ಯದಿ (ggml_openvino_model_cache_only()) {
        ಪ್ರತ್ಯಾಗಮ path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    ಯದಿ (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        ಪ್ರತ್ಯಾಗಮ std::string();
    }
    ಪ್ರತ್ಯಾಗಮ path;
}

std::string ggml_openvino_model_cache_temp_path(ಧ್ರುವ std::string & path) {
#ifdef _WIN32
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ pid = _getpid();
#ಅನ್ಯಥಾ
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ pid = getpid();
#endif
    ಪ್ರತ್ಯಾಗಮ path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(ಧ್ರುವ ggml_cgraph * cgraph,
                                         ಧ್ರುವ std::string & device,
                                         ತಾರ್ಕಿಕ fa,
                                         ಧ್ರುವ int32_t * rope_params,
                                         ಪೂರ್ಣಾಙ್ಕ rope_len,
                                         uint64_t extra_cfg,
                                         ಧ್ರುವ std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    ಪ್ರತಿ (ಧ್ರುವ ವರ್ಣ * name : cache_settings) {
        ಧ್ರುವ ವರ್ಣ * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    ಯದಿ (ಧ್ರುವ ವರ್ಣ * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", ಪರಿಮಾಣ("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    ಯದಿ (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", ಪರಿಮಾಣ("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; ++i) {
        ಧ್ರುವ ggml_tensor * ಗಂಟು = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(ಗಂಟು->op));
        h = fnv1a(h, ಗಂಟು->name, strlen(ಗಂಟು->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](ಧ್ರುವ ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    ಯದಿ (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, ಪರಿಮಾಣ(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ಧ್ರುವ std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    ಪ್ರತ್ಯಾಗಮ h;
}

std::string ggml_openvino_model_cache_blob_path(ಧ್ರುವ std::string & dir, uint64_t fingerprint) {
    ಪ್ರತ್ಯಾಗಮ dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(ಧ್ರುವ std::string & dir, uint64_t fingerprint) {
    ಪ್ರತ್ಯಾಗಮ dir + "/" + hex64(fingerprint) + ".manifest";
}

ತಾರ್ಕಿಕ ggml_openvino_model_cache_write_manifest(ಧ್ರುವ std::string & path,
                                              ಧ್ರುವ ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ಧ್ರುವ std::vector<std::string> & inputs,
                                              ಧ್ರುವ std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    ಯದಿ (!f.is_open()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](ಧ್ರುವ ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<ಪೂರ್ಣಾಙ್ಕ>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    ಪ್ರತಿ (ಧ್ರುವ auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        ಪ್ರತಿ (ಧ್ರುವ auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    ಪ್ರತ್ಯಾಗಮ f.good();
}

ತಾರ್ಕಿಕ ggml_openvino_model_cache_verify_manifest(ಧ್ರುವ std::string & path,
                                               ಧ್ರುವ ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    ಯದಿ (!f.is_open()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    ಯದಿ (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    // header: ov_version
    ಯದಿ (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](ಧ್ರುವ ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<ಪೂರ್ಣಾಙ್ಕ>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string ಸಾಲು;
    std::getline(f, ಸಾಲು);  // consume rest of ov_version line
    ಯಾವತ್ (idx < expected.size() && std::getline(f, ಸಾಲು)) {
        ಯದಿ (ಸಾಲು.empty()) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (idx >= expected.size() || ಸಾಲು != expected[idx]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ++idx;
    }
    ಯದಿ (idx != expected.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!std::getline(f, ಸಾಲು)) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಯದಿ (ಸಾಲು != "ports") {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತಿ (auto * names : { &inputs, &outputs }) {
        size_t count;
        ಯದಿ (!(f >> count) || count > 100000) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತಿ (size_t i = 0; i < count; ++i) {
            std::string name;
            ಯದಿ (!(f >> std::quoted(name))) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    ಪ್ರತ್ಯಾಗಮ f.eof();
}
