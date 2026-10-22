#شامل "model-cache.h"

#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"
#شامل "ggml-openvino-extra.h"

#شامل <cerrno>
#شامل <algorithm>
#شامل <cstdio>
#شامل <cstring>
#شامل <fstream>
#شامل <iomanip>
#شامل <limits>
#شامل <sstream>
#شامل <openvino/core/version.hpp>
#شامل <string>
#شامل <sys/stat.h>
#شامل <sys/types.h>
#شامل <vector>

#اگر defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#وگرنه
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

فضاینام {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
درونخطی uint64_t fnv1a(uint64_t h, ثابت تهی * data, size_t n) {
    ثابت uint8_t * p = static_cast<ثابت uint8_t *>(data);
    برای (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    بازگشت h;
}

درونخطی uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    بازگشت fnv1a(h, &v, اندازه(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
منطقی is_weight_src(ثابت ggml_tensor * src) {
    اگر (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        بازگشت نادرست;
    }
    بازگشت src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(ثابت ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    برای (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ثابت size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    بازگشت fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
تهی for_each_weight(ثابت ggml_cgraph * cgraph, F && fn) {
    std::vector<ثابت ggml_tensor *> seen;
    برای (صحیح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * گره = cgraph->nodes[i];
        برای (صحیح s = 0; s < GGML_MAX_SRC; ++s) {
            ثابت ggml_tensor * src = گره->src[s];
            اگر (!is_weight_src(src)) {
                ادامه;
            }
            منطقی dup = نادرست;
            برای (ثابت auto * p : seen) {
                اگر (p == src) {
                    dup = درست;
                    بشکن;
                }
            }
            اگر (dup) {
                ادامه;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    ثابت ov::Version v = ov::get_openvino_version();
    بازگشت std::string(v.build_عدد ? v.build_عدد : "unknown");
}

std::string hex64(uint64_t v) {
    نویسه buf[17];
    snprintf(buf, اندازه(buf), "%016llx", static_cast<unsigned long long>(v));
    بازگشت std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
منطقی make_dir(ثابت std::string & path) {
#اگر defined(_WIN32)
    صحیح rc = _mkdir(path.c_str());
#وگرنه
    صحیح rc = ::mkdir(path.c_str(), 0755);
#endif
    اگر (rc == 0 || errno == EEXIST) {
        بازگشت درست;
    }
    بازگشت نادرست;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
منطقی make_dirs(ثابت std::string & path) {
    اگر (path.empty()) {
        بازگشت نادرست;
    }
    std::string acc;
    برای (size_t i = 0; i < path.size(); ++i) {
        ثابت نویسه c = path[i];
        acc.push_back(c);
        ثابت منطقی sep = (c == '/'
#اگر defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        اگر (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            اگر (!make_dir(component)) {
                بازگشت نادرست;
            }
        }
    }
    بازگشت make_dir(path);
}

}  // namespace

منطقی ggml_openvino_model_cache_only() {
    بازگشت ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

ایستا ثابت نویسه * cache_settings[] = {
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

تهی ggml_openvino_model_cache_init() {
    ثابت منطقی cache_only = ggml_openvino_model_cache_only();
    ثابت std::string dir = ggml_openvino_model_cache_dir();
    اگر (dir.empty()) {
        اگر (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        بازگشت;
    }
    اگر (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#اگر !defined(__linux__) && !defined(_WIN32)
    اگر (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    اگر (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(ثابت تهی * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    ثابت uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](ثابت ggml_openvino_source_mapping & m) {
        بازگشت address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](ثابت ggml_openvino_source_mapping & m) {
        بازگشت fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    برای (ثابت auto & m : mappings) {
        اگر (contains(m)) {
            بازگشت fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string سطر;
    وقتی (std::getline(maps, سطر)) {
        unsigned long long begin, end, offset, inode;
        unsigned صحیح dev_major, dev_minor;
        نویسه permissions[5];
        صحیح path_start = 0;
        اگر (sscanf(سطر.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            ادامه;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        اگر (!contains(m)) {
            ادامه;
        }
        ساختار stat st;
        ثابت std::string path = سطر.substr(path_start);
        اگر (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            بشکن;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        بازگشت fingerprint(m);
    }
#وگرنهاگر defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    اگر (VirtualQuery(data, &memory, اندازه(memory)) == اندازه(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        وقتی (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<تهی *>(data), name.data(), static_cast<DWORD>(name.size()));
            اگر (length == 0 || length < name.size() - 1) {
                بشکن;
            }
            name.resize(name.size() * 2);
        }
        اگر (length > 0 && length < name.size() - 1) {
            name.resize(length);
            ثابت std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            اگر (file != INVALID_HANDLE_مقدار) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ثابت منطقی valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, اندازه(basic));
                CloseHandle(file);
                اگر (valid) {
                    ثابت uint64_t file_size = (uint64_t(info.n_File_Size_بالا) << 32) | info.n_File_Size_کم;
                    ثابت uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    اگر (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        اگر (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_عدد);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_Index_بالا) << 32) | info.n_File_Index_کم);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_بالا_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_کم_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            بازگشت fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    اگر (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    ثابت size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    اگر (size > head) {
        ثابت size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<ثابت uint8_t *>(data) + size - tail, tail);
    }
    بازگشت h;
}

std::string ggml_openvino_model_cache_dir() {
    ثابت نویسه * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    اگر (!dir || strlen(dir) == 0) {
        بازگشت std::string();
    }
    std::string path(dir);
    اگر (ggml_openvino_model_cache_only()) {
        بازگشت path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    اگر (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        بازگشت std::string();
    }
    بازگشت path;
}

std::string ggml_openvino_model_cache_temp_path(ثابت std::string & path) {
#ifdef _WIN32
    ثابت صحیح pid = _getpid();
#وگرنه
    ثابت صحیح pid = getpid();
#endif
    بازگشت path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(ثابت ggml_cgraph * cgraph,
                                         ثابت std::string & device,
                                         منطقی fa,
                                         ثابت int32_t * rope_params,
                                         صحیح rope_len,
                                         uint64_t extra_cfg,
                                         ثابت std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    برای (ثابت نویسه * name : cache_settings) {
        ثابت نویسه * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    اگر (ثابت نویسه * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", اندازه("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    اگر (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", اندازه("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    برای (صحیح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * گره = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(گره->op));
        h = fnv1a(h, گره->name, strlen(گره->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](ثابت ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    اگر (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, اندازه(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ثابت std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    بازگشت h;
}

std::string ggml_openvino_model_cache_blob_path(ثابت std::string & dir, uint64_t fingerprint) {
    بازگشت dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(ثابت std::string & dir, uint64_t fingerprint) {
    بازگشت dir + "/" + hex64(fingerprint) + ".manifest";
}

منطقی ggml_openvino_model_cache_write_manifest(ثابت std::string & path,
                                              ثابت ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ثابت std::vector<std::string> & inputs,
                                              ثابت std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    اگر (!f.is_open()) {
        بازگشت نادرست;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](ثابت ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<صحیح>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    برای (ثابت auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        برای (ثابت auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    بازگشت f.good();
}

منطقی ggml_openvino_model_cache_verify_manifest(ثابت std::string & path,
                                               ثابت ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    اگر (!f.is_open()) {
        بازگشت نادرست;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    اگر (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        بازگشت نادرست;
    }
    // header: ov_version
    اگر (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        بازگشت نادرست;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](ثابت ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<صحیح>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string سطر;
    std::getline(f, سطر);  // consume rest of ov_version line
    وقتی (idx < expected.size() && std::getline(f, سطر)) {
        اگر (سطر.empty()) {
            ادامه;
        }
        اگر (idx >= expected.size() || سطر != expected[idx]) {
            بازگشت نادرست;
        }
        ++idx;
    }
    اگر (idx != expected.size()) {
        بازگشت نادرست;
    }
    اگر (!std::getline(f, سطر)) {
        بازگشت درست;
    }
    اگر (سطر != "ports") {
        بازگشت نادرست;
    }
    برای (auto * names : { &inputs, &outputs }) {
        size_t count;
        اگر (!(f >> count) || count > 100000) {
            بازگشت نادرست;
        }
        برای (size_t i = 0; i < count; ++i) {
            std::string name;
            اگر (!(f >> std::quoted(name))) {
                بازگشت نادرست;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    بازگشت f.eof();
}
