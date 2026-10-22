#ضمن "model-cache.h"

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-openvino-extra.h"

#ضمن <cerrno>
#ضمن <algorithm>
#ضمن <cstdio>
#ضمن <cstring>
#ضمن <fstream>
#ضمن <iomanip>
#ضمن <limits>
#ضمن <sstream>
#ضمن <openvino/core/version.hpp>
#ضمن <string>
#ضمن <sys/stat.h>
#ضمن <sys/types.h>
#ضمن <vector>

#ܐܢ defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#ܐܚܪܢܐ
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

نطاق {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
مضمن uint64_t fnv1a(uint64_t h, ثابت فراغ * data, size_t n) {
    ثابت uint8_t * p = static_cast<ثابت uint8_t *>(data);
    لكل (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    ܦܢܐ h;
}

مضمن uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    ܦܢܐ fnv1a(h, &v, حجم(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
منطقي is_weight_src(ثابت ggml_tensor * src) {
    ܐܢ (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        ܦܢܐ ܕܓܠܐ;
    }
    ܦܢܐ src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(ثابت ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ثابت size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    ܦܢܐ fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
فراغ for_each_weight(ثابت ggml_cgraph * cgraph, F && fn) {
    std::vector<ثابت ggml_tensor *> seen;
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * node = cgraph->nodes[i];
        لكل (صحيح s = 0; s < GGML_MAX_SRC; ++s) {
            ثابت ggml_tensor * src = node->src[s];
            ܐܢ (!is_weight_src(src)) {
                استمر;
            }
            منطقي dup = ܕܓܠܐ;
            لكل (ثابت auto * p : seen) {
                ܐܢ (p == src) {
                    dup = ܫܪܪܐ;
                    اكسر;
                }
            }
            ܐܢ (dup) {
                استمر;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    ثابت ov::Version v = ov::get_openvino_version();
    ܦܢܐ std::string(v.buildNumber ? v.buildNumber : "unknown");
}

std::string hex64(uint64_t v) {
    حرف buf[17];
    snprintf(buf, حجم(buf), "%016llx", static_cast<unsigned long long>(v));
    ܦܢܐ std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
منطقي make_dir(ثابت std::string & path) {
#ܐܢ defined(_WIN32)
    صحيح rc = _mkdir(path.c_str());
#ܐܚܪܢܐ
    صحيح rc = ::mkdir(path.c_str(), 0755);
#endif
    ܐܢ (rc == 0 || errno == EEXIST) {
        ܦܢܐ ܫܪܪܐ;
    }
    ܦܢܐ ܕܓܠܐ;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
منطقي make_dirs(ثابت std::string & path) {
    ܐܢ (path.empty()) {
        ܦܢܐ ܕܓܠܐ;
    }
    std::string acc;
    لكل (size_t i = 0; i < path.size(); ++i) {
        ثابت حرف c = path[i];
        acc.push_back(c);
        ثابت منطقي sep = (c == '/'
#ܐܢ defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        ܐܢ (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            ܐܢ (!make_dir(component)) {
                ܦܢܐ ܕܓܠܐ;
            }
        }
    }
    ܦܢܐ make_dir(path);
}

}  // namespace

منطقي ggml_openvino_model_cache_only() {
    ܦܢܐ ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

ساكن ثابت حرف * cache_settings[] = {
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

فراغ ggml_openvino_model_cache_init() {
    ثابت منطقي cache_only = ggml_openvino_model_cache_only();
    ثابت std::string dir = ggml_openvino_model_cache_dir();
    ܐܢ (dir.empty()) {
        ܐܢ (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        ܦܢܐ;
    }
    ܐܢ (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#ܐܢ !defined(__linux__) && !defined(_WIN32)
    ܐܢ (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    ܐܢ (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(ثابت فراغ * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    ثابت uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](ثابت ggml_openvino_source_mapping & m) {
        ܦܢܐ address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](ثابت ggml_openvino_source_mapping & m) {
        ܦܢܐ fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    لكل (ثابت auto & m : mappings) {
        ܐܢ (contains(m)) {
            ܦܢܐ fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string line;
    ܟܡܐ (std::getline(maps, line)) {
        unsigned long long begin, end, offset, inode;
        unsigned صحيح dev_major, dev_minor;
        حرف permissions[5];
        صحيح path_start = 0;
        ܐܢ (sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            استمر;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        ܐܢ (!contains(m)) {
            استمر;
        }
        بنية stat st;
        ثابت std::string path = line.substr(path_start);
        ܐܢ (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            اكسر;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        ܦܢܐ fingerprint(m);
    }
#وإلاإذا defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    ܐܢ (VirtualQuery(data, &memory, حجم(memory)) == حجم(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        ܟܡܐ (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<فراغ *>(data), name.data(), static_cast<DWORD>(name.size()));
            ܐܢ (length == 0 || length < name.size() - 1) {
                اكسر;
            }
            name.resize(name.size() * 2);
        }
        ܐܢ (length > 0 && length < name.size() - 1) {
            name.resize(length);
            ثابت std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            ܐܢ (file != INVALID_HANDLE_VALUE) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ثابت منطقي valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, حجم(basic));
                CloseHandle(file);
                ܐܢ (valid) {
                    ثابت uint64_t file_size = (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
                    ثابت uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    ܐܢ (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        ܐܢ (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dwVolumeSerialNumber);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dwHighDateTime) << 32) |
                                                                 info.ftLastWriteTime.dwLowDateTime);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            ܦܢܐ fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    ܐܢ (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    ثابت size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    ܐܢ (size > head) {
        ثابت size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<ثابت uint8_t *>(data) + size - tail, tail);
    }
    ܦܢܐ h;
}

std::string ggml_openvino_model_cache_dir() {
    ثابت حرف * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    ܐܢ (!dir || strlen(dir) == 0) {
        ܦܢܐ std::string();
    }
    std::string path(dir);
    ܐܢ (ggml_openvino_model_cache_only()) {
        ܦܢܐ path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    ܐܢ (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        ܦܢܐ std::string();
    }
    ܦܢܐ path;
}

std::string ggml_openvino_model_cache_temp_path(ثابت std::string & path) {
#ifdef _WIN32
    ثابت صحيح pid = _getpid();
#ܐܚܪܢܐ
    ثابت صحيح pid = getpid();
#endif
    ܦܢܐ path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(ثابت ggml_cgraph * cgraph,
                                         ثابت std::string & device,
                                         منطقي fa,
                                         ثابت int32_t * rope_params,
                                         صحيح rope_len,
                                         uint64_t extra_cfg,
                                         ثابت std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    لكل (ثابت حرف * name : cache_settings) {
        ثابت حرف * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    ܐܢ (ثابت حرف * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", حجم("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    ܐܢ (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", حجم("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * node = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(node->op));
        h = fnv1a(h, node->name, strlen(node->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](ثابت ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    ܐܢ (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, حجم(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ثابت std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    ܦܢܐ h;
}

std::string ggml_openvino_model_cache_blob_path(ثابت std::string & dir, uint64_t fingerprint) {
    ܦܢܐ dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(ثابت std::string & dir, uint64_t fingerprint) {
    ܦܢܐ dir + "/" + hex64(fingerprint) + ".manifest";
}

منطقي ggml_openvino_model_cache_write_manifest(ثابت std::string & path,
                                              ثابت ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ثابت std::vector<std::string> & inputs,
                                              ثابت std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    ܐܢ (!f.is_open()) {
        ܦܢܐ ܕܓܠܐ;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](ثابت ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<صحيح>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    لكل (ثابت auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        لكل (ثابت auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    ܦܢܐ f.good();
}

منطقي ggml_openvino_model_cache_verify_manifest(ثابت std::string & path,
                                               ثابت ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    ܐܢ (!f.is_open()) {
        ܦܢܐ ܕܓܠܐ;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    ܐܢ (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        ܦܢܐ ܕܓܠܐ;
    }
    // header: ov_version
    ܐܢ (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        ܦܢܐ ܕܓܠܐ;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](ثابت ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<صحيح>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string line;
    std::getline(f, line);  // consume rest of ov_version line
    ܟܡܐ (idx < expected.size() && std::getline(f, line)) {
        ܐܢ (line.empty()) {
            استمر;
        }
        ܐܢ (idx >= expected.size() || line != expected[idx]) {
            ܦܢܐ ܕܓܠܐ;
        }
        ++idx;
    }
    ܐܢ (idx != expected.size()) {
        ܦܢܐ ܕܓܠܐ;
    }
    ܐܢ (!std::getline(f, line)) {
        ܦܢܐ ܫܪܪܐ;
    }
    ܐܢ (line != "ports") {
        ܦܢܐ ܕܓܠܐ;
    }
    لكل (auto * names : { &inputs, &outputs }) {
        size_t count;
        ܐܢ (!(f >> count) || count > 100000) {
            ܦܢܐ ܕܓܠܐ;
        }
        لكل (size_t i = 0; i < count; ++i) {
            std::string name;
            ܐܢ (!(f >> std::quoted(name))) {
                ܦܢܐ ܕܓܠܐ;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    ܦܢܐ f.eof();
}
