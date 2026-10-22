#অন্তর্ভাৱ "model-cache.h"

#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-backend.h"
#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml-openvino-extra.h"

#অন্তর্ভাৱ <cerrno>
#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <cstdio>
#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <fstream>
#অন্তর্ভাৱ <iomanip>
#অন্তর্ভাৱ <limits>
#অন্তর্ভাৱ <sstream>
#অন্তর্ভাৱ <openvino/core/version.hpp>
#অন্তর্ভাৱ <string>
#অন্তর্ভাৱ <sys/stat.h>
#অন্তর্ভাৱ <sys/types.h>
#অন্তর্ভাৱ <vector>

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
অন্তঃস্থ uint64_t fnv1a(uint64_t h, ধ্রুৱ শূন্য * data, size_t n) {
    ধ্রুৱ uint8_t * p = static_cast<ধ্রুৱ uint8_t *>(data);
    বাবে (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    ঘূৰাই h;
}

অন্তঃস্থ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    ঘূৰাই fnv1a(h, &v, পরিমাণ(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
তার্কিক is_weight_src(ধ্রুৱ ggml_tensor * src) {
    যদি (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        ঘূৰাই মিছা;
    }
    ঘূৰাই src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(ধ্রুৱ ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ধ্রুৱ size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    ঘূৰাই fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
শূন্য for_each_weight(ধ্রুৱ ggml_cgraph * cgraph, F && fn) {
    std::vector<ধ্রুৱ ggml_tensor *> seen;
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; ++i) {
        ধ্রুৱ ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];
        বাবে (পূর্ণাঙ্ক s = 0; s < GGML_MAX_SRC; ++s) {
            ধ্রুৱ ggml_tensor * src = গ্ৰন্থি->src[s];
            যদি (!is_weight_src(src)) {
                অনুৱর্ত;
            }
            তার্কিক dup = মিছা;
            বাবে (ধ্রুৱ auto * p : seen) {
                যদি (p == src) {
                    dup = সঁচা;
                    ভঙ্গ;
                }
            }
            যদি (dup) {
                অনুৱর্ত;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    ধ্রুৱ ov::Version v = ov::get_openvino_version();
    ঘূৰাই std::string(v.build_সংখ্যা ? v.build_সংখ্যা : "unknown");
}

std::string hex64(uint64_t v) {
    ৱর্ণ buf[17];
    snprintf(buf, পরিমাণ(buf), "%016llx", static_cast<unsigned long long>(v));
    ঘূৰাই std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
তার্কিক make_dir(ধ্রুৱ std::string & path) {
#যদি defined(_WIN32)
    পূর্ণাঙ্ক rc = _mkdir(path.c_str());
#অন্যথা
    পূর্ণাঙ্ক rc = ::mkdir(path.c_str(), 0755);
#endif
    যদি (rc == 0 || errno == EEXIST) {
        ঘূৰাই সঁচা;
    }
    ঘূৰাই মিছা;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
তার্কিক make_dirs(ধ্রুৱ std::string & path) {
    যদি (path.empty()) {
        ঘূৰাই মিছা;
    }
    std::string acc;
    বাবে (size_t i = 0; i < path.size(); ++i) {
        ধ্রুৱ ৱর্ণ c = path[i];
        acc.push_back(c);
        ধ্রুৱ তার্কিক sep = (c == '/'
#যদি defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        যদি (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            যদি (!make_dir(component)) {
                ঘূৰাই মিছা;
            }
        }
    }
    ঘূৰাই make_dir(path);
}

}  // namespace

তার্কিক ggml_openvino_model_cache_only() {
    ঘূৰাই ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

স্থির ধ্রুৱ ৱর্ণ * cache_settings[] = {
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
    ধ্রুৱ তার্কিক cache_only = ggml_openvino_model_cache_only();
    ধ্রুৱ std::string dir = ggml_openvino_model_cache_dir();
    যদি (dir.empty()) {
        যদি (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        ঘূৰাই;
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

uint64_t ggml_openvino_source_fingerprint(ধ্রুৱ শূন্য * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    ধ্রুৱ uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](ধ্রুৱ ggml_openvino_source_mapping & m) {
        ঘূৰাই address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](ধ্রুৱ ggml_openvino_source_mapping & m) {
        ঘূৰাই fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    বাবে (ধ্রুৱ auto & m : mappings) {
        যদি (contains(m)) {
            ঘূৰাই fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string সাৰি;
    যেতিয়ালৈ (std::getline(maps, সাৰি)) {
        unsigned long long begin, end, offset, inode;
        unsigned পূর্ণাঙ্ক dev_major, dev_minor;
        ৱর্ণ permissions[5];
        পূর্ণাঙ্ক path_start = 0;
        যদি (sscanf(সাৰি.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            অনুৱর্ত;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        যদি (!contains(m)) {
            অনুৱর্ত;
        }
        সংরচনা stat st;
        ধ্রুৱ std::string path = সাৰি.substr(path_start);
        যদি (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            ভঙ্গ;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        ঘূৰাই fingerprint(m);
    }
#নোচেদ্যদি defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    যদি (VirtualQuery(data, &memory, পরিমাণ(memory)) == পরিমাণ(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        যেতিয়ালৈ (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<শূন্য *>(data), name.data(), static_cast<DWORD>(name.size()));
            যদি (length == 0 || length < name.size() - 1) {
                ভঙ্গ;
            }
            name.resize(name.size() * 2);
        }
        যদি (length > 0 && length < name.size() - 1) {
            name.resize(length);
            ধ্রুৱ std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            যদি (file != INVALID_HANDLE_মান) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ধ্রুৱ তার্কিক valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, পরিমাণ(basic));
                CloseHandle(file);
                যদি (valid) {
                    ধ্রুৱ uint64_t file_size = (uint64_t(info.n_File_Size_উপৰেৰ) << 32) | info.n_File_Size_নিচেৰ;
                    ধ্রুৱ uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    যদি (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        যদি (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_সংখ্যা);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_Index_উপৰেৰ) << 32) | info.n_File_Index_নিচেৰ);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_উপৰেৰ_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_নিচেৰ_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            ঘূৰাই fingerprint(m);
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
    ধ্রুৱ size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    যদি (size > head) {
        ধ্রুৱ size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<ধ্রুৱ uint8_t *>(data) + size - tail, tail);
    }
    ঘূৰাই h;
}

std::string ggml_openvino_model_cache_dir() {
    ধ্রুৱ ৱর্ণ * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    যদি (!dir || strlen(dir) == 0) {
        ঘূৰাই std::string();
    }
    std::string path(dir);
    যদি (ggml_openvino_model_cache_only()) {
        ঘূৰাই path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    যদি (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        ঘূৰাই std::string();
    }
    ঘূৰাই path;
}

std::string ggml_openvino_model_cache_temp_path(ধ্রুৱ std::string & path) {
#ifdef _WIN32
    ধ্রুৱ পূর্ণাঙ্ক pid = _getpid();
#অন্যথা
    ধ্রুৱ পূর্ণাঙ্ক pid = getpid();
#endif
    ঘূৰাই path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(ধ্রুৱ ggml_cgraph * cgraph,
                                         ধ্রুৱ std::string & device,
                                         তার্কিক fa,
                                         ধ্রুৱ int32_t * rope_params,
                                         পূর্ণাঙ্ক rope_len,
                                         uint64_t extra_cfg,
                                         ধ্রুৱ std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    বাবে (ধ্রুৱ ৱর্ণ * name : cache_settings) {
        ধ্রুৱ ৱর্ণ * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    যদি (ধ্রুৱ ৱর্ণ * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", পরিমাণ("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    যদি (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", পরিমাণ("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; ++i) {
        ধ্রুৱ ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(গ্ৰন্থি->op));
        h = fnv1a(h, গ্ৰন্থি->name, strlen(গ্ৰন্থি->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](ধ্রুৱ ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    যদি (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, পরিমাণ(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ধ্রুৱ std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    ঘূৰাই h;
}

std::string ggml_openvino_model_cache_blob_path(ধ্রুৱ std::string & dir, uint64_t fingerprint) {
    ঘূৰাই dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(ধ্রুৱ std::string & dir, uint64_t fingerprint) {
    ঘূৰাই dir + "/" + hex64(fingerprint) + ".manifest";
}

তার্কিক ggml_openvino_model_cache_write_manifest(ধ্রুৱ std::string & path,
                                              ধ্রুৱ ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ধ্রুৱ std::vector<std::string> & inputs,
                                              ধ্রুৱ std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    যদি (!f.is_open()) {
        ঘূৰাই মিছা;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](ধ্রুৱ ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<পূর্ণাঙ্ক>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    বাবে (ধ্রুৱ auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        বাবে (ধ্রুৱ auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    ঘূৰাই f.good();
}

তার্কিক ggml_openvino_model_cache_verify_manifest(ধ্রুৱ std::string & path,
                                               ধ্রুৱ ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    যদি (!f.is_open()) {
        ঘূৰাই মিছা;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    যদি (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        ঘূৰাই মিছা;
    }
    // header: ov_version
    যদি (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        ঘূৰাই মিছা;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](ধ্রুৱ ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<পূর্ণাঙ্ক>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string সাৰি;
    std::getline(f, সাৰি);  // consume rest of ov_version line
    যেতিয়ালৈ (idx < expected.size() && std::getline(f, সাৰি)) {
        যদি (সাৰি.empty()) {
            অনুৱর্ত;
        }
        যদি (idx >= expected.size() || সাৰি != expected[idx]) {
            ঘূৰাই মিছা;
        }
        ++idx;
    }
    যদি (idx != expected.size()) {
        ঘূৰাই মিছা;
    }
    যদি (!std::getline(f, সাৰি)) {
        ঘূৰাই সঁচা;
    }
    যদি (সাৰি != "ports") {
        ঘূৰাই মিছা;
    }
    বাবে (auto * names : { &inputs, &outputs }) {
        size_t count;
        যদি (!(f >> count) || count > 100000) {
            ঘূৰাই মিছা;
        }
        বাবে (size_t i = 0; i < count; ++i) {
            std::string name;
            যদি (!(f >> std::quoted(name))) {
                ঘূৰাই মিছা;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    ঘূৰাই f.eof();
}
