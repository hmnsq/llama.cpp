#अन्तर्भाव "model-cache.h"

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-openvino-extra.h"

#अन्तर्भाव <cerrno>
#अन्तर्भाव <algorithm>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstring>
#अन्तर्भाव <fstream>
#अन्तर्भाव <iomanip>
#अन्तर्भाव <limits>
#अन्तर्भाव <sstream>
#अन्तर्भाव <openvino/core/version.hpp>
#अन्तर्भाव <string>
#अन्तर्भाव <sys/stat.h>
#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <vector>

#जइ defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#अण्णहा
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

नामाकाश {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
अन्तःस्थ uint64_t fnv1a(uint64_t h, ध्रुव शून्य * data, size_t n) {
    ध्रुव uint8_t * p = static_cast<ध्रुव uint8_t *>(data);
    प्रति (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    प्रत्यागम h;
}

अन्तःस्थ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    प्रत्यागम fnv1a(h, &v, परिमाण(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
तार्किक is_weight_src(ध्रुव ggml_tensor * src) {
    जइ (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        प्रत्यागम मिच्छ;
    }
    प्रत्यागम src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(ध्रुव ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ध्रुव size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    प्रत्यागम fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
शून्य for_each_weight(ध्रुव ggml_cgraph * cgraph, F && fn) {
    std::vector<ध्रुव ggml_tensor *> seen;
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; ++i) {
        ध्रुव ggml_tensor * ग्रन्थिः = cgraph->nodes[i];
        प्रति (पूर्णाङ्क s = 0; s < GGML_MAX_SRC; ++s) {
            ध्रुव ggml_tensor * src = ग्रन्थिः->src[s];
            जइ (!is_weight_src(src)) {
                अनुवर्त;
            }
            तार्किक dup = मिच्छ;
            प्रति (ध्रुव auto * p : seen) {
                जइ (p == src) {
                    dup = सच्च;
                    भङ्ग;
                }
            }
            जइ (dup) {
                अनुवर्त;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    ध्रुव ov::Version v = ov::get_openvino_version();
    प्रत्यागम std::string(v.build_सङ्ख्या ? v.build_सङ्ख्या : "unknown");
}

std::string hex64(uint64_t v) {
    वर्ण buf[17];
    snprintf(buf, परिमाण(buf), "%016llx", static_cast<unsigned long long>(v));
    प्रत्यागम std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
तार्किक make_dir(ध्रुव std::string & path) {
#जइ defined(_WIN32)
    पूर्णाङ्क rc = _mkdir(path.c_str());
#अण्णहा
    पूर्णाङ्क rc = ::mkdir(path.c_str(), 0755);
#endif
    जइ (rc == 0 || errno == EEXIST) {
        प्रत्यागम सच्च;
    }
    प्रत्यागम मिच्छ;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
तार्किक make_dirs(ध्रुव std::string & path) {
    जइ (path.empty()) {
        प्रत्यागम मिच्छ;
    }
    std::string acc;
    प्रति (size_t i = 0; i < path.size(); ++i) {
        ध्रुव वर्ण c = path[i];
        acc.push_back(c);
        ध्रुव तार्किक sep = (c == '/'
#जइ defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        जइ (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            जइ (!make_dir(component)) {
                प्रत्यागम मिच्छ;
            }
        }
    }
    प्रत्यागम make_dir(path);
}

}  // namespace

तार्किक ggml_openvino_model_cache_only() {
    प्रत्यागम ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

स्थिर ध्रुव वर्ण * cache_settings[] = {
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

शून्य ggml_openvino_model_cache_init() {
    ध्रुव तार्किक cache_only = ggml_openvino_model_cache_only();
    ध्रुव std::string dir = ggml_openvino_model_cache_dir();
    जइ (dir.empty()) {
        जइ (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        प्रत्यागम;
    }
    जइ (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#जइ !defined(__linux__) && !defined(_WIN32)
    जइ (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    जइ (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(ध्रुव शून्य * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    ध्रुव uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](ध्रुव ggml_openvino_source_mapping & m) {
        प्रत्यागम address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](ध्रुव ggml_openvino_source_mapping & m) {
        प्रत्यागम fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    प्रति (ध्रुव auto & m : mappings) {
        जइ (contains(m)) {
            प्रत्यागम fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string पङ्क्तिः;
    जाव (std::getline(maps, पङ्क्तिः)) {
        unsigned long long begin, end, offset, inode;
        unsigned पूर्णाङ्क dev_major, dev_minor;
        वर्ण permissions[5];
        पूर्णाङ्क path_start = 0;
        जइ (sscanf(पङ्क्तिः.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            अनुवर्त;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        जइ (!contains(m)) {
            अनुवर्त;
        }
        संरचना stat st;
        ध्रुव std::string path = पङ्क्तिः.substr(path_start);
        जइ (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            भङ्ग;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        प्रत्यागम fingerprint(m);
    }
#नोचेद्यदि defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    जइ (VirtualQuery(data, &memory, परिमाण(memory)) == परिमाण(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        जाव (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<शून्य *>(data), name.data(), static_cast<DWORD>(name.size()));
            जइ (length == 0 || length < name.size() - 1) {
                भङ्ग;
            }
            name.resize(name.size() * 2);
        }
        जइ (length > 0 && length < name.size() - 1) {
            name.resize(length);
            ध्रुव std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            जइ (file != INVALID_HANDLE_मानम्) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ध्रुव तार्किक valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, परिमाण(basic));
                CloseHandle(file);
                जइ (valid) {
                    ध्रुव uint64_t file_size = (uint64_t(info.n_File_Size_उच्चः) << 32) | info.n_File_Size_नीचः;
                    ध्रुव uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    जइ (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        जइ (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_सङ्ख्या);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_Index_उच्चः) << 32) | info.n_File_Index_नीचः);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_उच्चः_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_नीचः_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            प्रत्यागम fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    जइ (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    ध्रुव size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    जइ (size > head) {
        ध्रुव size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<ध्रुव uint8_t *>(data) + size - tail, tail);
    }
    प्रत्यागम h;
}

std::string ggml_openvino_model_cache_dir() {
    ध्रुव वर्ण * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    जइ (!dir || strlen(dir) == 0) {
        प्रत्यागम std::string();
    }
    std::string path(dir);
    जइ (ggml_openvino_model_cache_only()) {
        प्रत्यागम path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    जइ (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        प्रत्यागम std::string();
    }
    प्रत्यागम path;
}

std::string ggml_openvino_model_cache_temp_path(ध्रुव std::string & path) {
#ifdef _WIN32
    ध्रुव पूर्णाङ्क pid = _getpid();
#अण्णहा
    ध्रुव पूर्णाङ्क pid = getpid();
#endif
    प्रत्यागम path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(ध्रुव ggml_cgraph * cgraph,
                                         ध्रुव std::string & device,
                                         तार्किक fa,
                                         ध्रुव int32_t * rope_params,
                                         पूर्णाङ्क rope_len,
                                         uint64_t extra_cfg,
                                         ध्रुव std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    प्रति (ध्रुव वर्ण * name : cache_settings) {
        ध्रुव वर्ण * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    जइ (ध्रुव वर्ण * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", परिमाण("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    जइ (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", परिमाण("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; ++i) {
        ध्रुव ggml_tensor * ग्रन्थिः = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(ग्रन्थिः->op));
        h = fnv1a(h, ग्रन्थिः->name, strlen(ग्रन्थिः->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](ध्रुव ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    जइ (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, परिमाण(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ध्रुव std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    प्रत्यागम h;
}

std::string ggml_openvino_model_cache_blob_path(ध्रुव std::string & dir, uint64_t fingerprint) {
    प्रत्यागम dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(ध्रुव std::string & dir, uint64_t fingerprint) {
    प्रत्यागम dir + "/" + hex64(fingerprint) + ".manifest";
}

तार्किक ggml_openvino_model_cache_write_manifest(ध्रुव std::string & path,
                                              ध्रुव ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ध्रुव std::vector<std::string> & inputs,
                                              ध्रुव std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    जइ (!f.is_open()) {
        प्रत्यागम मिच्छ;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](ध्रुव ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<पूर्णाङ्क>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    प्रति (ध्रुव auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        प्रति (ध्रुव auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    प्रत्यागम f.good();
}

तार्किक ggml_openvino_model_cache_verify_manifest(ध्रुव std::string & path,
                                               ध्रुव ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    जइ (!f.is_open()) {
        प्रत्यागम मिच्छ;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    जइ (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        प्रत्यागम मिच्छ;
    }
    // header: ov_version
    जइ (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        प्रत्यागम मिच्छ;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](ध्रुव ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<पूर्णाङ्क>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string पङ्क्तिः;
    std::getline(f, पङ्क्तिः);  // consume rest of ov_version line
    जाव (idx < expected.size() && std::getline(f, पङ्क्तिः)) {
        जइ (पङ्क्तिः.empty()) {
            अनुवर्त;
        }
        जइ (idx >= expected.size() || पङ्क्तिः != expected[idx]) {
            प्रत्यागम मिच्छ;
        }
        ++idx;
    }
    जइ (idx != expected.size()) {
        प्रत्यागम मिच्छ;
    }
    जइ (!std::getline(f, पङ्क्तिः)) {
        प्रत्यागम सच्च;
    }
    जइ (पङ्क्तिः != "ports") {
        प्रत्यागम मिच्छ;
    }
    प्रति (auto * names : { &inputs, &outputs }) {
        size_t count;
        जइ (!(f >> count) || count > 100000) {
            प्रत्यागम मिच्छ;
        }
        प्रति (size_t i = 0; i < count; ++i) {
            std::string name;
            जइ (!(f >> std::quoted(name))) {
                प्रत्यागम मिच्छ;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    प्रत्यागम f.eof();
}
