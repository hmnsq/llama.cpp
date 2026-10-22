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

#אם defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#אחרת
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

نطاق {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
مضمن uint64_t fnv1a(uint64_t h, ثابت فراغ * data, size_t n) {
    ثابت uint8_t * p = static_cast<ثابت uint8_t *>(data);
    עבור (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    החזר h;
}

مضمن uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    החזר fnv1a(h, &v, حجم(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
منطقي is_weight_src(ثابت ggml_tensor * src) {
    אם (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        החזר שקר;
    }
    החזר src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(ثابت ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    עבור (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ثابت size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    החזר fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
فراغ for_each_weight(ثابت ggml_cgraph * cgraph, F && fn) {
    std::vector<ثابت ggml_tensor *> seen;
    עבור (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * node = cgraph->nodes[i];
        עבור (صحيح s = 0; s < GGML_MAX_SRC; ++s) {
            ثابت ggml_tensor * src = node->src[s];
            אם (!is_weight_src(src)) {
                استمر;
            }
            منطقي dup = שקר;
            עבור (ثابت auto * p : seen) {
                אם (p == src) {
                    dup = אמת;
                    اكسر;
                }
            }
            אם (dup) {
                استمر;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    ثابت ov::Version v = ov::get_openvino_version();
    החזר std::string(v.buildNumber ? v.buildNumber : "unknown");
}

std::string hex64(uint64_t v) {
    حرف buf[17];
    snprintf(buf, حجم(buf), "%016llx", static_cast<unsigned long long>(v));
    החזר std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
منطقي make_dir(ثابت std::string & path) {
#אם defined(_WIN32)
    صحيح rc = _mkdir(path.c_str());
#אחרת
    صحيح rc = ::mkdir(path.c_str(), 0755);
#endif
    אם (rc == 0 || errno == EEXIST) {
        החזר אמת;
    }
    החזר שקר;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
منطقي make_dirs(ثابت std::string & path) {
    אם (path.empty()) {
        החזר שקר;
    }
    std::string acc;
    עבור (size_t i = 0; i < path.size(); ++i) {
        ثابت حرف c = path[i];
        acc.push_back(c);
        ثابت منطقي sep = (c == '/'
#אם defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        אם (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            אם (!make_dir(component)) {
                החזר שקר;
            }
        }
    }
    החזר make_dir(path);
}

}  // namespace

منطقي ggml_openvino_model_cache_only() {
    החזר ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
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
    אם (dir.empty()) {
        אם (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        החזר;
    }
    אם (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#אם !defined(__linux__) && !defined(_WIN32)
    אם (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    אם (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(ثابت فراغ * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    ثابت uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](ثابت ggml_openvino_source_mapping & m) {
        החזר address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](ثابت ggml_openvino_source_mapping & m) {
        החזר fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    עבור (ثابت auto & m : mappings) {
        אם (contains(m)) {
            החזר fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string line;
    כלעוד (std::getline(maps, line)) {
        unsigned long long begin, end, offset, inode;
        unsigned صحيح dev_major, dev_minor;
        حرف permissions[5];
        صحيح path_start = 0;
        אם (sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            استمر;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        אם (!contains(m)) {
            استمر;
        }
        بنية stat st;
        ثابت std::string path = line.substr(path_start);
        אם (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
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
        החזר fingerprint(m);
    }
#وإلاإذا defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    אם (VirtualQuery(data, &memory, حجم(memory)) == حجم(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        כלעוד (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<فراغ *>(data), name.data(), static_cast<DWORD>(name.size()));
            אם (length == 0 || length < name.size() - 1) {
                اكسر;
            }
            name.resize(name.size() * 2);
        }
        אם (length > 0 && length < name.size() - 1) {
            name.resize(length);
            ثابت std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            אם (file != INVALID_HANDLE_VALUE) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ثابت منطقي valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, حجم(basic));
                CloseHandle(file);
                אם (valid) {
                    ثابت uint64_t file_size = (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
                    ثابت uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    אם (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        אם (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dwVolumeSerialNumber);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dwHighDateTime) << 32) |
                                                                 info.ftLastWriteTime.dwLowDateTime);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            החזר fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    אם (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    ثابت size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    אם (size > head) {
        ثابت size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<ثابت uint8_t *>(data) + size - tail, tail);
    }
    החזר h;
}

std::string ggml_openvino_model_cache_dir() {
    ثابت حرف * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    אם (!dir || strlen(dir) == 0) {
        החזר std::string();
    }
    std::string path(dir);
    אם (ggml_openvino_model_cache_only()) {
        החזר path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    אם (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        החזר std::string();
    }
    החזר path;
}

std::string ggml_openvino_model_cache_temp_path(ثابت std::string & path) {
#ifdef _WIN32
    ثابت صحيح pid = _getpid();
#אחרת
    ثابت صحيح pid = getpid();
#endif
    החזר path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
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
    עבור (ثابت حرف * name : cache_settings) {
        ثابت حرف * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    אם (ثابت حرف * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", حجم("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    אם (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", حجم("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    עבור (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * node = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(node->op));
        h = fnv1a(h, node->name, strlen(node->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](ثابت ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    אם (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, حجم(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ثابت std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    החזר h;
}

std::string ggml_openvino_model_cache_blob_path(ثابت std::string & dir, uint64_t fingerprint) {
    החזר dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(ثابت std::string & dir, uint64_t fingerprint) {
    החזר dir + "/" + hex64(fingerprint) + ".manifest";
}

منطقي ggml_openvino_model_cache_write_manifest(ثابت std::string & path,
                                              ثابت ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ثابت std::vector<std::string> & inputs,
                                              ثابت std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    אם (!f.is_open()) {
        החזר שקר;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](ثابت ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<صحيح>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    עבור (ثابت auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        עבור (ثابت auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    החזר f.good();
}

منطقي ggml_openvino_model_cache_verify_manifest(ثابت std::string & path,
                                               ثابت ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    אם (!f.is_open()) {
        החזר שקר;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    אם (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        החזר שקר;
    }
    // header: ov_version
    אם (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        החזר שקר;
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
    כלעוד (idx < expected.size() && std::getline(f, line)) {
        אם (line.empty()) {
            استمر;
        }
        אם (idx >= expected.size() || line != expected[idx]) {
            החזר שקר;
        }
        ++idx;
    }
    אם (idx != expected.size()) {
        החזר שקר;
    }
    אם (!std::getline(f, line)) {
        החזר אמת;
    }
    אם (line != "ports") {
        החזר שקר;
    }
    עבור (auto * names : { &inputs, &outputs }) {
        size_t count;
        אם (!(f >> count) || count > 100000) {
            החזר שקר;
        }
        עבור (size_t i = 0; i < count; ++i) {
            std::string name;
            אם (!(f >> std::quoted(name))) {
                החזר שקר;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    החזר f.eof();
}
