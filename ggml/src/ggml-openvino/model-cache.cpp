#അന്തര്ഭാവ "model-cache.h"

#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-openvino-extra.h"

#അന്തര്ഭാവ <cerrno>
#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cstdio>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <fstream>
#അന്തര്ഭാവ <iomanip>
#അന്തര്ഭാവ <limits>
#അന്തര്ഭാവ <sstream>
#അന്തര്ഭാവ <openvino/core/version.hpp>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <sys/stat.h>
#അന്തര്ഭാവ <sys/types.h>
#അന്തര്ഭാവ <vector>

#എങ്കിൽ defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#അല്ലെങ്കിൽ
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

നാമാകാശ {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
വരിയിൽ uint64_t fnv1a(uint64_t h, സ്ഥിരം ശൂന്യം * data, size_t n) {
    സ്ഥിരം uint8_t * p = static_cast<സ്ഥിരം uint8_t *>(data);
    വേണ്ടി (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    തിരികെ h;
}

വരിയിൽ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    തിരികെ fnv1a(h, &v, വലുപ്പം(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
തർക്കം is_weight_src(സ്ഥിരം ggml_tensor * src) {
    എങ്കിൽ (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        തിരികെ അസത്യം;
    }
    തിരികെ src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(സ്ഥിരം ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    സ്ഥിരം size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    തിരികെ fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
ശൂന്യം for_each_weight(സ്ഥിരം ggml_cgraph * cgraph, F && fn) {
    std::vector<സ്ഥിരം ggml_tensor *> seen;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; ++i) {
        സ്ഥിരം ggml_tensor * കണ്ണി = cgraph->nodes[i];
        വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < GGML_MAX_SRC; ++s) {
            സ്ഥിരം ggml_tensor * src = കണ്ണി->src[s];
            എങ്കിൽ (!is_weight_src(src)) {
                തുടരുക;
            }
            തർക്കം dup = അസത്യം;
            വേണ്ടി (സ്ഥിരം auto * p : seen) {
                എങ്കിൽ (p == src) {
                    dup = സത്യം;
                    നിർത്തുക;
                }
            }
            എങ്കിൽ (dup) {
                തുടരുക;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    സ്ഥിരം ov::Version v = ov::get_openvino_version();
    തിരികെ std::string(v.build_സംഖ്യ ? v.build_സംഖ്യ : "unknown");
}

std::string hex64(uint64_t v) {
    അക്ഷരം buf[17];
    snprintf(buf, വലുപ്പം(buf), "%016llx", static_cast<unsigned long long>(v));
    തിരികെ std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
തർക്കം make_dir(സ്ഥിരം std::string & path) {
#എങ്കിൽ defined(_WIN32)
    പൂർണ്ണസംഖ്യ rc = _mkdir(path.c_str());
#അല്ലെങ്കിൽ
    പൂർണ്ണസംഖ്യ rc = ::mkdir(path.c_str(), 0755);
#endif
    എങ്കിൽ (rc == 0 || errno == EEXIST) {
        തിരികെ സത്യം;
    }
    തിരികെ അസത്യം;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
തർക്കം make_dirs(സ്ഥിരം std::string & path) {
    എങ്കിൽ (path.empty()) {
        തിരികെ അസത്യം;
    }
    std::string acc;
    വേണ്ടി (size_t i = 0; i < path.size(); ++i) {
        സ്ഥിരം അക്ഷരം c = path[i];
        acc.push_back(c);
        സ്ഥിരം തർക്കം sep = (c == '/'
#എങ്കിൽ defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        എങ്കിൽ (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            എങ്കിൽ (!make_dir(component)) {
                തിരികെ അസത്യം;
            }
        }
    }
    തിരികെ make_dir(path);
}

}  // namespace

തർക്കം ggml_openvino_model_cache_only() {
    തിരികെ ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * cache_settings[] = {
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

ശൂന്യം ggml_openvino_model_cache_init() {
    സ്ഥിരം തർക്കം cache_only = ggml_openvino_model_cache_only();
    സ്ഥിരം std::string dir = ggml_openvino_model_cache_dir();
    എങ്കിൽ (dir.empty()) {
        എങ്കിൽ (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        തിരികെ;
    }
    എങ്കിൽ (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#എങ്കിൽ !defined(__linux__) && !defined(_WIN32)
    എങ്കിൽ (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    എങ്കിൽ (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(സ്ഥിരം ശൂന്യം * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    സ്ഥിരം uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](സ്ഥിരം ggml_openvino_source_mapping & m) {
        തിരികെ address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](സ്ഥിരം ggml_openvino_source_mapping & m) {
        തിരികെ fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    വേണ്ടി (സ്ഥിരം auto & m : mappings) {
        എങ്കിൽ (contains(m)) {
            തിരികെ fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string വരി;
    ആകുമ്പോൾ (std::getline(maps, വരി)) {
        unsigned long long begin, end, offset, inode;
        unsigned പൂർണ്ണസംഖ്യ dev_major, dev_minor;
        അക്ഷരം permissions[5];
        പൂർണ്ണസംഖ്യ path_start = 0;
        എങ്കിൽ (sscanf(വരി.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            തുടരുക;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        എങ്കിൽ (!contains(m)) {
            തുടരുക;
        }
        ഘടന stat st;
        സ്ഥിരം std::string path = വരി.substr(path_start);
        എങ്കിൽ (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            നിർത്തുക;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        തിരികെ fingerprint(m);
    }
#അല്ലെങ്കിലെങ്കിൽ defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    എങ്കിൽ (VirtualQuery(data, &memory, വലുപ്പം(memory)) == വലുപ്പം(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        ആകുമ്പോൾ (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<ശൂന്യം *>(data), name.data(), static_cast<DWORD>(name.size()));
            എങ്കിൽ (length == 0 || length < name.size() - 1) {
                നിർത്തുക;
            }
            name.resize(name.size() * 2);
        }
        എങ്കിൽ (length > 0 && length < name.size() - 1) {
            name.resize(length);
            സ്ഥിരം std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            എങ്കിൽ (file != INVALID_HANDLE_മൂല്യം) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                സ്ഥിരം തർക്കം valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, വലുപ്പം(basic));
                CloseHandle(file);
                എങ്കിൽ (valid) {
                    സ്ഥിരം uint64_t file_size = (uint64_t(info.n_File_Size_മുകളിലെ) << 32) | info.n_File_Size_താഴെയുള്ള;
                    സ്ഥിരം uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    എങ്കിൽ (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        എങ്കിൽ (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_സംഖ്യ);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_Index_മുകളിലെ) << 32) | info.n_File_Index_താഴെയുള്ള);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_മുകളിലെ_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_താഴെയുള്ള_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            തിരികെ fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    എങ്കിൽ (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    സ്ഥിരം size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    എങ്കിൽ (size > head) {
        സ്ഥിരം size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<സ്ഥിരം uint8_t *>(data) + size - tail, tail);
    }
    തിരികെ h;
}

std::string ggml_openvino_model_cache_dir() {
    സ്ഥിരം അക്ഷരം * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    എങ്കിൽ (!dir || strlen(dir) == 0) {
        തിരികെ std::string();
    }
    std::string path(dir);
    എങ്കിൽ (ggml_openvino_model_cache_only()) {
        തിരികെ path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    എങ്കിൽ (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        തിരികെ std::string();
    }
    തിരികെ path;
}

std::string ggml_openvino_model_cache_temp_path(സ്ഥിരം std::string & path) {
#ifdef _WIN32
    സ്ഥിരം പൂർണ്ണസംഖ്യ pid = _getpid();
#അല്ലെങ്കിൽ
    സ്ഥിരം പൂർണ്ണസംഖ്യ pid = getpid();
#endif
    തിരികെ path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(സ്ഥിരം ggml_cgraph * cgraph,
                                         സ്ഥിരം std::string & device,
                                         തർക്കം fa,
                                         സ്ഥിരം int32_t * rope_params,
                                         പൂർണ്ണസംഖ്യ rope_len,
                                         uint64_t extra_cfg,
                                         സ്ഥിരം std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    വേണ്ടി (സ്ഥിരം അക്ഷരം * name : cache_settings) {
        സ്ഥിരം അക്ഷരം * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    എങ്കിൽ (സ്ഥിരം അക്ഷരം * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", വലുപ്പം("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    എങ്കിൽ (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", വലുപ്പം("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; ++i) {
        സ്ഥിരം ggml_tensor * കണ്ണി = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(കണ്ണി->op));
        h = fnv1a(h, കണ്ണി->name, strlen(കണ്ണി->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](സ്ഥിരം ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    എങ്കിൽ (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, വലുപ്പം(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    സ്ഥിരം std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    തിരികെ h;
}

std::string ggml_openvino_model_cache_blob_path(സ്ഥിരം std::string & dir, uint64_t fingerprint) {
    തിരികെ dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(സ്ഥിരം std::string & dir, uint64_t fingerprint) {
    തിരികെ dir + "/" + hex64(fingerprint) + ".manifest";
}

തർക്കം ggml_openvino_model_cache_write_manifest(സ്ഥിരം std::string & path,
                                              സ്ഥിരം ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              സ്ഥിരം std::vector<std::string> & inputs,
                                              സ്ഥിരം std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    എങ്കിൽ (!f.is_open()) {
        തിരികെ അസത്യം;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](സ്ഥിരം ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<പൂർണ്ണസംഖ്യ>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    വേണ്ടി (സ്ഥിരം auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        വേണ്ടി (സ്ഥിരം auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    തിരികെ f.good();
}

തർക്കം ggml_openvino_model_cache_verify_manifest(സ്ഥിരം std::string & path,
                                               സ്ഥിരം ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    എങ്കിൽ (!f.is_open()) {
        തിരികെ അസത്യം;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    എങ്കിൽ (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        തിരികെ അസത്യം;
    }
    // header: ov_version
    എങ്കിൽ (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        തിരികെ അസത്യം;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](സ്ഥിരം ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<പൂർണ്ണസംഖ്യ>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string വരി;
    std::getline(f, വരി);  // consume rest of ov_version line
    ആകുമ്പോൾ (idx < expected.size() && std::getline(f, വരി)) {
        എങ്കിൽ (വരി.empty()) {
            തുടരുക;
        }
        എങ്കിൽ (idx >= expected.size() || വരി != expected[idx]) {
            തിരികെ അസത്യം;
        }
        ++idx;
    }
    എങ്കിൽ (idx != expected.size()) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!std::getline(f, വരി)) {
        തിരികെ സത്യം;
    }
    എങ്കിൽ (വരി != "ports") {
        തിരികെ അസത്യം;
    }
    വേണ്ടി (auto * names : { &inputs, &outputs }) {
        size_t count;
        എങ്കിൽ (!(f >> count) || count > 100000) {
            തിരികെ അസത്യം;
        }
        വേണ്ടി (size_t i = 0; i < count; ++i) {
            std::string name;
            എങ്കിൽ (!(f >> std::quoted(name))) {
                തിരികെ അസത്യം;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    തിരികെ f.eof();
}
