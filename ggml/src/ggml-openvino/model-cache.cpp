#অন্তর্ভাব "model-cache.h"

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-openvino-extra.h"

#অন্তর্ভাব <cerrno>
#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cstdio>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <fstream>
#অন্তর্ভাব <iomanip>
#অন্তর্ভাব <limits>
#অন্তর্ভাব <sstream>
#অন্তর্ভাব <openvino/core/version.hpp>
#অন্তর্ভাব <string>
#অন্তর্ভাব <sys/stat.h>
#অন্তর্ভাব <sys/types.h>
#অন্তর্ভাব <vector>

#যদি defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#অন্যথা
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

নামাকাশ {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
অন্তঃরেখ uint64_t fnv1a(uint64_t h, ধ্রুবক শূন্য * তথ্য, size_t n) {
    ধ্রুবক uint8_t * p = static_cast<ধ্রুবক uint8_t *>(তথ্য);
    জন্য (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    ফেরত h;
}

অন্তঃরেখ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    ফেরত fnv1a(h, &v, আকার(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
যৌক্তিক is_weight_src(ধ্রুবক ggml_tensor * src) {
    যদি (src == nullptr || src->view_src != nullptr || src->বাফার == nullptr) {
        ফেরত মিথ্যা;
    }
    ফেরত src->বাফার->usage == GGML_BACKEND_বাফার_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(ধ্রুবক ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ধ্রুবক size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    ফেরত fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
শূন্য for_each_weight(ধ্রুবক ggml_cgraph * cgraph, F && fn) {
    std::vector<ধ্রুবক ggml_tensor *> seen;
    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_nodes; ++i) {
        ধ্রুবক ggml_tensor * গ্রন্থি = cgraph->nodes[i];
        জন্য (পূর্ণসংখ্যা s = 0; s < GGML_MAX_SRC; ++s) {
            ধ্রুবক ggml_tensor * src = গ্রন্থি->src[s];
            যদি (!is_weight_src(src)) {
                চালু;
            }
            যৌক্তিক dup = মিথ্যা;
            জন্য (ধ্রুবক auto * p : seen) {
                যদি (p == src) {
                    dup = সত্য;
                    বিরতি;
                }
            }
            যদি (dup) {
                চালু;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    ধ্রুবক ov::Version v = ov::get_openvino_version();
    ফেরত std::string(v.build_সংখ্যা ? v.build_সংখ্যা : "unknown");
}

std::string hex64(uint64_t v) {
    অক্ষর buf[17];
    snprintf(buf, আকার(buf), "%016llx", static_cast<unsigned long long>(v));
    ফেরত std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
যৌক্তিক make_dir(ধ্রুবক std::string & path) {
#যদি defined(_WIN32)
    পূর্ণসংখ্যা rc = _mkdir(path.c_str());
#অন্যথা
    পূর্ণসংখ্যা rc = ::mkdir(path.c_str(), 0755);
#endif
    যদি (rc == 0 || errno == EEXIST) {
        ফেরত সত্য;
    }
    ফেরত মিথ্যা;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
যৌক্তিক make_dirs(ধ্রুবক std::string & path) {
    যদি (path.empty()) {
        ফেরত মিথ্যা;
    }
    std::string acc;
    জন্য (size_t i = 0; i < path.size(); ++i) {
        ধ্রুবক অক্ষর c = path[i];
        acc.push_back(c);
        ধ্রুবক যৌক্তিক sep = (c == '/'
#যদি defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        যদি (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            যদি (!make_dir(component)) {
                ফেরত মিথ্যা;
            }
        }
    }
    ফেরত make_dir(path);
}

}  // namespace

যৌক্তিক ggml_openvino_model_cache_only() {
    ফেরত ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

স্থির ধ্রুবক অক্ষর * cache_settings[] = {
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

শূন্য ggml_openvino_model_cache_init() {
    ধ্রুবক যৌক্তিক cache_only = ggml_openvino_model_cache_only();
    ধ্রুবক std::string dir = ggml_openvino_model_cache_dir();
    যদি (dir.empty()) {
        যদি (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        ফেরত;
    }
    যদি (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#যদি !defined(__linux__) && !defined(_WIN32)
    যদি (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    যদি (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(ধ্রুবক শূন্য * তথ্য, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    ধ্রুবক uintptr_t address = reinterpret_cast<uintptr_t>(তথ্য);
    auto contains = [&](ধ্রুবক ggml_openvino_source_mapping & m) {
        ফেরত address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](ধ্রুবক ggml_openvino_source_mapping & m) {
        ফেরত fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    জন্য (ধ্রুবক auto & m : mappings) {
        যদি (contains(m)) {
            ফেরত fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string সারি;
    যখন (std::getline(maps, সারি)) {
        unsigned long long begin, end, offset, inode;
        unsigned পূর্ণসংখ্যা dev_major, dev_minor;
        অক্ষর permissions[5];
        পূর্ণসংখ্যা path_start = 0;
        যদি (sscanf(সারি.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            চালু;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        যদি (!contains(m)) {
            চালু;
        }
        কাঠামো stat st;
        ধ্রুবক std::string path = সারি.substr(path_start);
        যদি (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            বিরতি;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        ফেরত fingerprint(m);
    }
#নতুবা_যদি defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    যদি (VirtualQuery(তথ্য, &memory, আকার(memory)) == আকার(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD দৈর্ঘ্য = 0;
        যখন (name.size() <= 32768) {
            দৈর্ঘ্য = GetMappedFileNameW(GetCurrentProcess(), const_cast<শূন্য *>(তথ্য), name.তথ্য(), static_cast<DWORD>(name.size()));
            যদি (দৈর্ঘ্য == 0 || দৈর্ঘ্য < name.size() - 1) {
                বিরতি;
            }
            name.resize(name.size() * 2);
        }
        যদি (দৈর্ঘ্য > 0 && দৈর্ঘ্য < name.size() - 1) {
            name.resize(দৈর্ঘ্য);
            ধ্রুবক std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            যদি (file != INVALID_HANDLE_মান) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ধ্রুবক যৌক্তিক valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, আকার(basic));
                CloseHandle(file);
                যদি (valid) {
                    ধ্রুবক uint64_t file_size = (uint64_t(info.n_File_আকার_উপরের) << 32) | info.n_File_আকার_নিচের;
                    ধ্রুবক uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    যদি (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        যদি (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_সংখ্যা);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_সূচক_উপরের) << 32) | info.n_File_সূচক_নিচের);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_উপরের_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_নিচের_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            ফেরত fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    যদি (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    ধ্রুবক size_t শীর্ষ = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, তথ্য, শীর্ষ);
    যদি (size > শীর্ষ) {
        ধ্রুবক size_t লেজ = std::min(size - শীর্ষ, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<ধ্রুবক uint8_t *>(তথ্য) + size - লেজ, লেজ);
    }
    ফেরত h;
}

std::string ggml_openvino_model_cache_dir() {
    ধ্রুবক অক্ষর * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    যদি (!dir || strlen(dir) == 0) {
        ফেরত std::string();
    }
    std::string path(dir);
    যদি (ggml_openvino_model_cache_only()) {
        ফেরত path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    যদি (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        ফেরত std::string();
    }
    ফেরত path;
}

std::string ggml_openvino_model_cache_অস্থায়ী_path(ধ্রুবক std::string & path) {
#ifdef _WIN32
    ধ্রুবক পূর্ণসংখ্যা pid = _getpid();
#অন্যথা
    ধ্রুবক পূর্ণসংখ্যা pid = getpid();
#endif
    ফেরত path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(ধ্রুবক ggml_cgraph * cgraph,
                                         ধ্রুবক std::string & device,
                                         যৌক্তিক fa,
                                         ধ্রুবক int32_t * rope_params,
                                         পূর্ণসংখ্যা rope_len,
                                         uint64_t extra_cfg,
                                         ধ্রুবক std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.তথ্য(), graph_signature.size());
    জন্য (ধ্রুবক অক্ষর * name : cache_settings) {
        ধ্রুবক অক্ষর * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    যদি (ধ্রুবক অক্ষর * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", আকার("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    যদি (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", আকার("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_nodes; ++i) {
        ধ্রুবক ggml_tensor * গ্রন্থি = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(গ্রন্থি->op));
        h = fnv1a(h, গ্রন্থি->name, strlen(গ্রন্থি->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](ধ্রুবক ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.তথ্য(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    যদি (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, আকার(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ধ্রুবক std::string ver = ov_version_string();
    h = fnv1a(h, ver.তথ্য(), ver.size());

    ফেরত h;
}

std::string ggml_openvino_model_cache_blob_path(ধ্রুবক std::string & dir, uint64_t fingerprint) {
    ফেরত dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(ধ্রুবক std::string & dir, uint64_t fingerprint) {
    ফেরত dir + "/" + hex64(fingerprint) + ".manifest";
}

যৌক্তিক ggml_openvino_model_cache_write_manifest(ধ্রুবক std::string & path,
                                              ধ্রুবক ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ধ্রুবক std::vector<std::string> & inputs,
                                              ধ্রুবক std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    যদি (!f.is_open()) {
        ফেরত মিথ্যা;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](ধ্রুবক ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<পূর্ণসংখ্যা>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    জন্য (ধ্রুবক auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        জন্য (ধ্রুবক auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    ফেরত f.good();
}

যৌক্তিক ggml_openvino_model_cache_verify_manifest(ধ্রুবক std::string & path,
                                               ধ্রুবক ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    যদি (!f.is_open()) {
        ফেরত মিথ্যা;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    যদি (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        ফেরত মিথ্যা;
    }
    // header: ov_version
    যদি (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        ফেরত মিথ্যা;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](ধ্রুবক ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<পূর্ণসংখ্যা>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string সারি;
    std::getline(f, সারি);  // consume rest of ov_version line
    যখন (idx < expected.size() && std::getline(f, সারি)) {
        যদি (সারি.empty()) {
            চালু;
        }
        যদি (idx >= expected.size() || সারি != expected[idx]) {
            ফেরত মিথ্যা;
        }
        ++idx;
    }
    যদি (idx != expected.size()) {
        ফেরত মিথ্যা;
    }
    যদি (!std::getline(f, সারি)) {
        ফেরত সত্য;
    }
    যদি (সারি != "ports") {
        ফেরত মিথ্যা;
    }
    জন্য (auto * names : { &inputs, &outputs }) {
        size_t count;
        যদি (!(f >> count) || count > 100000) {
            ফেরত মিথ্যা;
        }
        জন্য (size_t i = 0; i < count; ++i) {
            std::string name;
            যদি (!(f >> std::quoted(name))) {
                ফেরত মিথ্যা;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    ফেরত f.eof();
}
