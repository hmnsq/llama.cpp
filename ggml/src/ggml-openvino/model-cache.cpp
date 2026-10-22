#ଅନ୍ତର୍ଭାଵ "model-cache.h"

#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-openvino-extra.h"

#ଅନ୍ତର୍ଭାଵ <cerrno>
#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cstdio>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <fstream>
#ଅନ୍ତର୍ଭାଵ <iomanip>
#ଅନ୍ତର୍ଭାଵ <limits>
#ଅନ୍ତର୍ଭାଵ <sstream>
#ଅନ୍ତର୍ଭାଵ <openvino/core/version.hpp>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <sys/stat.h>
#ଅନ୍ତର୍ଭାଵ <sys/types.h>
#ଅନ୍ତର୍ଭାଵ <vector>

#ଯଦି defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#ଅନ୍ୟଥା
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

ନାମାକାଶ {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
ଧାଡ଼ିଭିତରେ uint64_t fnv1a(uint64_t h, ଅଚଳ ଶୂନ୍ୟ * data, size_t n) {
    ଅଚଳ uint8_t * p = static_cast<ଅଚଳ uint8_t *>(data);
    ପାଇଁ (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    ଫେରନ୍ତୁ h;
}

ଧାଡ଼ିଭିତରେ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    ଫେରନ୍ତୁ fnv1a(h, &v, ଆକାର(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
ତାର୍କିକ is_weight_src(ଅଚଳ ggml_tensor * src) {
    ଯଦି (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(ଅଚଳ ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ଅଚଳ size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    ଫେରନ୍ତୁ fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
ଶୂନ୍ୟ for_each_weight(ଅଚଳ ggml_cgraph * cgraph, F && fn) {
    std::vector<ଅଚଳ ggml_tensor *> seen;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; ++i) {
        ଅଚଳ ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < GGML_MAX_SRC; ++s) {
            ଅଚଳ ggml_tensor * src = ଗଣ୍ଠି->src[s];
            ଯଦି (!is_weight_src(src)) {
                ଜାରି;
            }
            ତାର୍କିକ dup = ମିଥ୍ୟା;
            ପାଇଁ (ଅଚଳ auto * p : seen) {
                ଯଦି (p == src) {
                    dup = ସତ୍ୟ;
                    ଭାଙ୍ଗ;
                }
            }
            ଯଦି (dup) {
                ଜାରି;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    ଅଚଳ ov::Version v = ov::get_openvino_version();
    ଫେରନ୍ତୁ std::string(v.build_ସଂଖ୍ୟା ? v.build_ସଂଖ୍ୟା : "unknown");
}

std::string hex64(uint64_t v) {
    ଅକ୍ଷର buf[17];
    snprintf(buf, ଆକାର(buf), "%016llx", static_cast<unsigned long long>(v));
    ଫେରନ୍ତୁ std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
ତାର୍କିକ make_dir(ଅଚଳ std::string & path) {
#ଯଦି defined(_WIN32)
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା rc = _mkdir(path.c_str());
#ଅନ୍ୟଥା
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା rc = ::mkdir(path.c_str(), 0755);
#endif
    ଯଦି (rc == 0 || errno == EEXIST) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
ତାର୍କିକ make_dirs(ଅଚଳ std::string & path) {
    ଯଦି (path.empty()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    std::string acc;
    ପାଇଁ (size_t i = 0; i < path.size(); ++i) {
        ଅଚଳ ଅକ୍ଷର c = path[i];
        acc.push_back(c);
        ଅଚଳ ତାର୍କିକ sep = (c == '/'
#ଯଦି defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        ଯଦି (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            ଯଦି (!make_dir(component)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
    }
    ଫେରନ୍ତୁ make_dir(path);
}

}  // namespace

ତାର୍କିକ ggml_openvino_model_cache_only() {
    ଫେରନ୍ତୁ ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * cache_settings[] = {
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

ଶୂନ୍ୟ ggml_openvino_model_cache_init() {
    ଅଚଳ ତାର୍କିକ cache_only = ggml_openvino_model_cache_only();
    ଅଚଳ std::string dir = ggml_openvino_model_cache_dir();
    ଯଦି (dir.empty()) {
        ଯଦି (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        ଫେରନ୍ତୁ;
    }
    ଯଦି (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#ଯଦି !defined(__linux__) && !defined(_WIN32)
    ଯଦି (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    ଯଦି (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(ଅଚଳ ଶୂନ୍ୟ * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    ଅଚଳ uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](ଅଚଳ ggml_openvino_source_mapping & m) {
        ଫେରନ୍ତୁ address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](ଅଚଳ ggml_openvino_source_mapping & m) {
        ଫେରନ୍ତୁ fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    ପାଇଁ (ଅଚଳ auto & m : mappings) {
        ଯଦି (contains(m)) {
            ଫେରନ୍ତୁ fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string ଧାଡ଼ି;
    ଯେତେବେଳେ (std::getline(maps, ଧାଡ଼ି)) {
        unsigned long long begin, end, offset, inode;
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା dev_major, dev_minor;
        ଅକ୍ଷର permissions[5];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା path_start = 0;
        ଯଦି (sscanf(ଧାଡ଼ି.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            ଜାରି;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        ଯଦି (!contains(m)) {
            ଜାରି;
        }
        ସଂରଚନା stat st;
        ଅଚଳ std::string path = ଧାଡ଼ି.substr(path_start);
        ଯଦି (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            ଭାଙ୍ଗ;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        ଫେରନ୍ତୁ fingerprint(m);
    }
#ନଚେତଯଦି defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    ଯଦି (VirtualQuery(data, &memory, ଆକାର(memory)) == ଆକାର(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        ଯେତେବେଳେ (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<ଶୂନ୍ୟ *>(data), name.data(), static_cast<DWORD>(name.size()));
            ଯଦି (length == 0 || length < name.size() - 1) {
                ଭାଙ୍ଗ;
            }
            name.resize(name.size() * 2);
        }
        ଯଦି (length > 0 && length < name.size() - 1) {
            name.resize(length);
            ଅଚଳ std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            ଯଦି (file != INVALID_HANDLE_ମୂଲ୍ୟ) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ଅଚଳ ତାର୍କିକ valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, ଆକାର(basic));
                CloseHandle(file);
                ଯଦି (valid) {
                    ଅଚଳ uint64_t file_size = (uint64_t(info.n_File_Size_ଉପର) << 32) | info.n_File_Size_ତଳ;
                    ଅଚଳ uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    ଯଦି (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        ଯଦି (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_ସଂଖ୍ୟା);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_Index_ଉପର) << 32) | info.n_File_Index_ତଳ);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_ଉପର_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_ତଳ_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            ଫେରନ୍ତୁ fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    ଯଦି (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    ଅଚଳ size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    ଯଦି (size > head) {
        ଅଚଳ size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<ଅଚଳ uint8_t *>(data) + size - tail, tail);
    }
    ଫେରନ୍ତୁ h;
}

std::string ggml_openvino_model_cache_dir() {
    ଅଚଳ ଅକ୍ଷର * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    ଯଦି (!dir || strlen(dir) == 0) {
        ଫେରନ୍ତୁ std::string();
    }
    std::string path(dir);
    ଯଦି (ggml_openvino_model_cache_only()) {
        ଫେରନ୍ତୁ path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    ଯଦି (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        ଫେରନ୍ତୁ std::string();
    }
    ଫେରନ୍ତୁ path;
}

std::string ggml_openvino_model_cache_temp_path(ଅଚଳ std::string & path) {
#ifdef _WIN32
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pid = _getpid();
#ଅନ୍ୟଥା
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pid = getpid();
#endif
    ଫେରନ୍ତୁ path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(ଅଚଳ ggml_cgraph * cgraph,
                                         ଅଚଳ std::string & device,
                                         ତାର୍କିକ fa,
                                         ଅଚଳ int32_t * rope_params,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା rope_len,
                                         uint64_t extra_cfg,
                                         ଅଚଳ std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    ପାଇଁ (ଅଚଳ ଅକ୍ଷର * name : cache_settings) {
        ଅଚଳ ଅକ୍ଷର * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    ଯଦି (ଅଚଳ ଅକ୍ଷର * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", ଆକାର("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    ଯଦି (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", ଆକାର("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; ++i) {
        ଅଚଳ ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(ଗଣ୍ଠି->op));
        h = fnv1a(h, ଗଣ୍ଠି->name, strlen(ଗଣ୍ଠି->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](ଅଚଳ ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    ଯଦି (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, ଆକାର(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ଅଚଳ std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    ଫେରନ୍ତୁ h;
}

std::string ggml_openvino_model_cache_blob_path(ଅଚଳ std::string & dir, uint64_t fingerprint) {
    ଫେରନ୍ତୁ dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(ଅଚଳ std::string & dir, uint64_t fingerprint) {
    ଫେରନ୍ତୁ dir + "/" + hex64(fingerprint) + ".manifest";
}

ତାର୍କିକ ggml_openvino_model_cache_write_manifest(ଅଚଳ std::string & path,
                                              ଅଚଳ ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ଅଚଳ std::vector<std::string> & inputs,
                                              ଅଚଳ std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    ଯଦି (!f.is_open()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](ଅଚଳ ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    ପାଇଁ (ଅଚଳ auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        ପାଇଁ (ଅଚଳ auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    ଫେରନ୍ତୁ f.good();
}

ତାର୍କିକ ggml_openvino_model_cache_verify_manifest(ଅଚଳ std::string & path,
                                               ଅଚଳ ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    ଯଦି (!f.is_open()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    ଯଦି (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    // header: ov_version
    ଯଦି (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](ଅଚଳ ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string ଧାଡ଼ି;
    std::getline(f, ଧାଡ଼ି);  // consume rest of ov_version line
    ଯେତେବେଳେ (idx < expected.size() && std::getline(f, ଧାଡ଼ି)) {
        ଯଦି (ଧାଡ଼ି.empty()) {
            ଜାରି;
        }
        ଯଦି (idx >= expected.size() || ଧାଡ଼ି != expected[idx]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ++idx;
    }
    ଯଦି (idx != expected.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!std::getline(f, ଧାଡ଼ି)) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଯଦି (ଧାଡ଼ି != "ports") {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ପାଇଁ (auto * names : { &inputs, &outputs }) {
        size_t count;
        ଯଦି (!(f >> count) || count > 100000) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ପାଇଁ (size_t i = 0; i < count; ++i) {
            std::string name;
            ଯଦି (!(f >> std::quoted(name))) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    ଫେରନ୍ତୁ f.eof();
}
