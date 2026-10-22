#அந்தர்பாவ "model-cache.h"

#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-openvino-extra.h"

#அந்தர்பாவ <cerrno>
#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cstdio>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <fstream>
#அந்தர்பாவ <iomanip>
#அந்தர்பாவ <limits>
#அந்தர்பாவ <sstream>
#அந்தர்பாவ <openvino/core/version.hpp>
#அந்தர்பாவ <string>
#அந்தர்பாவ <sys/stat.h>
#அந்தர்பாவ <sys/types.h>
#அந்தர்பாவ <vector>

#எனில் defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#இல்லையெனில்
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

நாமாகாஶ {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
உள்வரி uint64_t fnv1a(uint64_t h, மாறிலி வெறுமை * data, size_t n) {
    மாறிலி uint8_t * p = static_cast<மாறிலி uint8_t *>(data);
    க்கு (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    திரும்பு h;
}

உள்வரி uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    திரும்பு fnv1a(h, &v, அளவு(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
தருக்க is_weight_src(மாறிலி ggml_tensor * src) {
    எனில் (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        திரும்பு பொய்;
    }
    திரும்பு src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(மாறிலி ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    க்கு (முழு i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    மாறிலி size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    திரும்பு fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
வெறுமை for_each_weight(மாறிலி ggml_cgraph * cgraph, F && fn) {
    std::vector<மாறிலி ggml_tensor *> seen;
    க்கு (முழு i = 0; i < cgraph->n_nodes; ++i) {
        மாறிலி ggml_tensor * முனை = cgraph->nodes[i];
        க்கு (முழு s = 0; s < GGML_MAX_SRC; ++s) {
            மாறிலி ggml_tensor * src = முனை->src[s];
            எனில் (!is_weight_src(src)) {
                தொடர்;
            }
            தருக்க dup = பொய்;
            க்கு (மாறிலி auto * p : seen) {
                எனில் (p == src) {
                    dup = உண்மை;
                    நிறுத்து;
                }
            }
            எனில் (dup) {
                தொடர்;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    மாறிலி ov::Version v = ov::get_openvino_version();
    திரும்பு std::string(v.build_எண் ? v.build_எண் : "unknown");
}

std::string hex64(uint64_t v) {
    எழுத்து buf[17];
    snprintf(buf, அளவு(buf), "%016llx", static_cast<unsigned long long>(v));
    திரும்பு std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
தருக்க make_dir(மாறிலி std::string & path) {
#எனில் defined(_WIN32)
    முழு rc = _mkdir(path.c_str());
#இல்லையெனில்
    முழு rc = ::mkdir(path.c_str(), 0755);
#endif
    எனில் (rc == 0 || errno == EEXIST) {
        திரும்பு உண்மை;
    }
    திரும்பு பொய்;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
தருக்க make_dirs(மாறிலி std::string & path) {
    எனில் (path.empty()) {
        திரும்பு பொய்;
    }
    std::string acc;
    க்கு (size_t i = 0; i < path.size(); ++i) {
        மாறிலி எழுத்து c = path[i];
        acc.push_back(c);
        மாறிலி தருக்க sep = (c == '/'
#எனில் defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        எனில் (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            எனில் (!make_dir(component)) {
                திரும்பு பொய்;
            }
        }
    }
    திரும்பு make_dir(path);
}

}  // namespace

தருக்க ggml_openvino_model_cache_only() {
    திரும்பு ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

நிலையான மாறிலி எழுத்து * cache_settings[] = {
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

வெறுமை ggml_openvino_model_cache_init() {
    மாறிலி தருக்க cache_only = ggml_openvino_model_cache_only();
    மாறிலி std::string dir = ggml_openvino_model_cache_dir();
    எனில் (dir.empty()) {
        எனில் (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        திரும்பு;
    }
    எனில் (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#எனில் !defined(__linux__) && !defined(_WIN32)
    எனில் (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    எனில் (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(மாறிலி வெறுமை * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    மாறிலி uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](மாறிலி ggml_openvino_source_mapping & m) {
        திரும்பு address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](மாறிலி ggml_openvino_source_mapping & m) {
        திரும்பு fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    க்கு (மாறிலி auto & m : mappings) {
        எனில் (contains(m)) {
            திரும்பு fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string வரி;
    போது (std::getline(maps, வரி)) {
        unsigned long long begin, end, offset, inode;
        unsigned முழு dev_major, dev_minor;
        எழுத்து permissions[5];
        முழு path_start = 0;
        எனில் (sscanf(வரி.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            தொடர்;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        எனில் (!contains(m)) {
            தொடர்;
        }
        கட்டமைப்பு stat st;
        மாறிலி std::string path = வரி.substr(path_start);
        எனில் (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            நிறுத்து;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        திரும்பு fingerprint(m);
    }
#அல்லதெனில் defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    எனில் (VirtualQuery(data, &memory, அளவு(memory)) == அளவு(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        போது (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<வெறுமை *>(data), name.data(), static_cast<DWORD>(name.size()));
            எனில் (length == 0 || length < name.size() - 1) {
                நிறுத்து;
            }
            name.resize(name.size() * 2);
        }
        எனில் (length > 0 && length < name.size() - 1) {
            name.resize(length);
            மாறிலி std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            எனில் (file != INVALID_HANDLE_மதிப்பு) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                மாறிலி தருக்க valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, அளவு(basic));
                CloseHandle(file);
                எனில் (valid) {
                    மாறிலி uint64_t file_size = (uint64_t(info.n_File_Size_உயர்வு) << 32) | info.n_File_Size_கீழ்;
                    மாறிலி uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    எனில் (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        எனில் (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_எண்);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_Index_உயர்வு) << 32) | info.n_File_Index_கீழ்);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_உயர்வு_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_கீழ்_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            திரும்பு fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    எனில் (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    மாறிலி size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    எனில் (size > head) {
        மாறிலி size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<மாறிலி uint8_t *>(data) + size - tail, tail);
    }
    திரும்பு h;
}

std::string ggml_openvino_model_cache_dir() {
    மாறிலி எழுத்து * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    எனில் (!dir || strlen(dir) == 0) {
        திரும்பு std::string();
    }
    std::string path(dir);
    எனில் (ggml_openvino_model_cache_only()) {
        திரும்பு path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    எனில் (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        திரும்பு std::string();
    }
    திரும்பு path;
}

std::string ggml_openvino_model_cache_temp_path(மாறிலி std::string & path) {
#ifdef _WIN32
    மாறிலி முழு pid = _getpid();
#இல்லையெனில்
    மாறிலி முழு pid = getpid();
#endif
    திரும்பு path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(மாறிலி ggml_cgraph * cgraph,
                                         மாறிலி std::string & device,
                                         தருக்க fa,
                                         மாறிலி int32_t * rope_params,
                                         முழு rope_len,
                                         uint64_t extra_cfg,
                                         மாறிலி std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    க்கு (மாறிலி எழுத்து * name : cache_settings) {
        மாறிலி எழுத்து * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    எனில் (மாறிலி எழுத்து * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", அளவு("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    எனில் (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", அளவு("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    க்கு (முழு i = 0; i < cgraph->n_nodes; ++i) {
        மாறிலி ggml_tensor * முனை = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(முனை->op));
        h = fnv1a(h, முனை->name, strlen(முனை->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](மாறிலி ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    எனில் (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, அளவு(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    மாறிலி std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    திரும்பு h;
}

std::string ggml_openvino_model_cache_blob_path(மாறிலி std::string & dir, uint64_t fingerprint) {
    திரும்பு dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(மாறிலி std::string & dir, uint64_t fingerprint) {
    திரும்பு dir + "/" + hex64(fingerprint) + ".manifest";
}

தருக்க ggml_openvino_model_cache_write_manifest(மாறிலி std::string & path,
                                              மாறிலி ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              மாறிலி std::vector<std::string> & inputs,
                                              மாறிலி std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    எனில் (!f.is_open()) {
        திரும்பு பொய்;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](மாறிலி ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<முழு>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    க்கு (மாறிலி auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        க்கு (மாறிலி auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    திரும்பு f.good();
}

தருக்க ggml_openvino_model_cache_verify_manifest(மாறிலி std::string & path,
                                               மாறிலி ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    எனில் (!f.is_open()) {
        திரும்பு பொய்;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    எனில் (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        திரும்பு பொய்;
    }
    // header: ov_version
    எனில் (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        திரும்பு பொய்;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](மாறிலி ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<முழு>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string வரி;
    std::getline(f, வரி);  // consume rest of ov_version line
    போது (idx < expected.size() && std::getline(f, வரி)) {
        எனில் (வரி.empty()) {
            தொடர்;
        }
        எனில் (idx >= expected.size() || வரி != expected[idx]) {
            திரும்பு பொய்;
        }
        ++idx;
    }
    எனில் (idx != expected.size()) {
        திரும்பு பொய்;
    }
    எனில் (!std::getline(f, வரி)) {
        திரும்பு உண்மை;
    }
    எனில் (வரி != "ports") {
        திரும்பு பொய்;
    }
    க்கு (auto * names : { &inputs, &outputs }) {
        size_t count;
        எனில் (!(f >> count) || count > 100000) {
            திரும்பு பொய்;
        }
        க்கு (size_t i = 0; i < count; ++i) {
            std::string name;
            எனில் (!(f >> std::quoted(name))) {
                திரும்பு பொய்;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    திரும்பு f.eof();
}
