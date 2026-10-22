#અન્તર્ભાવ "model-cache.h"

#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-openvino-extra.h"

#અન્તર્ભાવ <cerrno>
#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cstdio>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <fstream>
#અન્તર્ભાવ <iomanip>
#અન્તર્ભાવ <limits>
#અન્તર્ભાવ <sstream>
#અન્તર્ભાવ <openvino/core/version.hpp>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <sys/stat.h>
#અન્તર્ભાવ <sys/types.h>
#અન્તર્ભાવ <vector>

#જો defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#નહીંતર
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

નામાકાશ {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
અંતઃરેખ uint64_t fnv1a(uint64_t h, અચલ શૂન્ય * data, size_t n) {
    અચલ uint8_t * p = static_cast<અચલ uint8_t *>(data);
    માટે (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    પાછા h;
}

અંતઃરેખ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    પાછા fnv1a(h, &v, કદ(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
તાર્કિક is_weight_src(અચલ ggml_tensor * src) {
    જો (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        પાછા ખોટું;
    }
    પાછા src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(અચલ ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    અચલ size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    પાછા fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
શૂન્ય for_each_weight(અચલ ggml_cgraph * cgraph, F && fn) {
    std::vector<અચલ ggml_tensor *> seen;
    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; ++i) {
        અચલ ggml_tensor * ગાંઠ = cgraph->nodes[i];
        માટે (પૂર્ણાંક s = 0; s < GGML_MAX_SRC; ++s) {
            અચલ ggml_tensor * src = ગાંઠ->src[s];
            જો (!is_weight_src(src)) {
                ચાલુ;
            }
            તાર્કિક dup = ખોટું;
            માટે (અચલ auto * p : seen) {
                જો (p == src) {
                    dup = સાચું;
                    તોડો;
                }
            }
            જો (dup) {
                ચાલુ;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    અચલ ov::Version v = ov::get_openvino_version();
    પાછા std::string(v.build_સંખ્યા ? v.build_સંખ્યા : "unknown");
}

std::string hex64(uint64_t v) {
    અક્ષર buf[17];
    snprintf(buf, કદ(buf), "%016llx", static_cast<unsigned long long>(v));
    પાછા std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
તાર્કિક make_dir(અચલ std::string & path) {
#જો defined(_WIN32)
    પૂર્ણાંક rc = _mkdir(path.c_str());
#નહીંતર
    પૂર્ણાંક rc = ::mkdir(path.c_str(), 0755);
#endif
    જો (rc == 0 || errno == EEXIST) {
        પાછા સાચું;
    }
    પાછા ખોટું;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
તાર્કિક make_dirs(અચલ std::string & path) {
    જો (path.empty()) {
        પાછા ખોટું;
    }
    std::string acc;
    માટે (size_t i = 0; i < path.size(); ++i) {
        અચલ અક્ષર c = path[i];
        acc.push_back(c);
        અચલ તાર્કિક sep = (c == '/'
#જો defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        જો (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            જો (!make_dir(component)) {
                પાછા ખોટું;
            }
        }
    }
    પાછા make_dir(path);
}

}  // namespace

તાર્કિક ggml_openvino_model_cache_only() {
    પાછા ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

સ્થિર અચલ અક્ષર * cache_settings[] = {
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

શૂન્ય ggml_openvino_model_cache_init() {
    અચલ તાર્કિક cache_only = ggml_openvino_model_cache_only();
    અચલ std::string dir = ggml_openvino_model_cache_dir();
    જો (dir.empty()) {
        જો (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        પાછા;
    }
    જો (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#જો !defined(__linux__) && !defined(_WIN32)
    જો (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    જો (cache_only) {
        auto & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(અચલ શૂન્ય * data, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    અચલ uintptr_t address = reinterpret_cast<uintptr_t>(data);
    auto contains = [&](અચલ ggml_openvino_source_mapping & m) {
        પાછા address >= m.begin && address < m.end && size <= m.end - address;
    };
    auto fingerprint = [&](અચલ ggml_openvino_source_mapping & m) {
        પાછા fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    માટે (અચલ auto & m : mappings) {
        જો (contains(m)) {
            પાછા fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string લીટી;
    જ્યારે (std::getline(maps, લીટી)) {
        unsigned long long begin, end, offset, inode;
        unsigned પૂર્ણાંક dev_major, dev_minor;
        અક્ષર permissions[5];
        પૂર્ણાંક path_start = 0;
        જો (sscanf(લીટી.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            ચાલુ;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        જો (!contains(m)) {
            ચાલુ;
        }
        સંરચના stat st;
        અચલ std::string path = લીટી.substr(path_start);
        જો (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            તોડો;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        પાછા fingerprint(m);
    }
#અથવાજો defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    જો (VirtualQuery(data, &memory, કદ(memory)) == કદ(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        જ્યારે (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<શૂન્ય *>(data), name.data(), static_cast<DWORD>(name.size()));
            જો (length == 0 || length < name.size() - 1) {
                તોડો;
            }
            name.resize(name.size() * 2);
        }
        જો (length > 0 && length < name.size() - 1) {
            name.resize(length);
            અચલ std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            જો (file != INVALID_HANDLE_કિંમત) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                અચલ તાર્કિક valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, કદ(basic));
                CloseHandle(file);
                જો (valid) {
                    અચલ uint64_t file_size = (uint64_t(info.n_File_Size_ઉપલું) << 32) | info.n_File_Size_નીચલું;
                    અચલ uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    જો (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        જો (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_સંખ્યા);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_Index_ઉપલું) << 32) | info.n_File_Index_નીચલું);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_ઉપલું_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_નીચલું_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            પાછા fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    જો (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    અચલ size_t head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    જો (size > head) {
        અચલ size_t tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<અચલ uint8_t *>(data) + size - tail, tail);
    }
    પાછા h;
}

std::string ggml_openvino_model_cache_dir() {
    અચલ અક્ષર * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    જો (!dir || strlen(dir) == 0) {
        પાછા std::string();
    }
    std::string path(dir);
    જો (ggml_openvino_model_cache_only()) {
        પાછા path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    જો (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        પાછા std::string();
    }
    પાછા path;
}

std::string ggml_openvino_model_cache_temp_path(અચલ std::string & path) {
#ifdef _WIN32
    અચલ પૂર્ણાંક pid = _getpid();
#નહીંતર
    અચલ પૂર્ણાંક pid = getpid();
#endif
    પાછા path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(અચલ ggml_cgraph * cgraph,
                                         અચલ std::string & device,
                                         તાર્કિક fa,
                                         અચલ int32_t * rope_params,
                                         પૂર્ણાંક rope_len,
                                         uint64_t extra_cfg,
                                         અચલ std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    માટે (અચલ અક્ષર * name : cache_settings) {
        અચલ અક્ષર * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, strlen(value) + 1);
    }
    જો (અચલ અક્ષર * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", કદ("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, strlen(debug_nodes) + 1);
    }
    જો (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", કદ("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; ++i) {
        અચલ ggml_tensor * ગાંઠ = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(ગાંઠ->op));
        h = fnv1a(h, ગાંઠ->name, strlen(ગાંઠ->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](અચલ ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    જો (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, કદ(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    અચલ std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    પાછા h;
}

std::string ggml_openvino_model_cache_blob_path(અચલ std::string & dir, uint64_t fingerprint) {
    પાછા dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(અચલ std::string & dir, uint64_t fingerprint) {
    પાછા dir + "/" + hex64(fingerprint) + ".manifest";
}

તાર્કિક ggml_openvino_model_cache_write_manifest(અચલ std::string & path,
                                              અચલ ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              અચલ std::vector<std::string> & inputs,
                                              અચલ std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    જો (!f.is_open()) {
        પાછા ખોટું;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](અચલ ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<પૂર્ણાંક>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    માટે (અચલ auto * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        માટે (અચલ auto & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    પાછા f.good();
}

તાર્કિક ggml_openvino_model_cache_verify_manifest(અચલ std::string & path,
                                               અચલ ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    જો (!f.is_open()) {
        પાછા ખોટું;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    જો (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        પાછા ખોટું;
    }
    // header: ov_version
    જો (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        પાછા ખોટું;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](અચલ ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<પૂર્ણાંક>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string લીટી;
    std::getline(f, લીટી);  // consume rest of ov_version line
    જ્યારે (idx < expected.size() && std::getline(f, લીટી)) {
        જો (લીટી.empty()) {
            ચાલુ;
        }
        જો (idx >= expected.size() || લીટી != expected[idx]) {
            પાછા ખોટું;
        }
        ++idx;
    }
    જો (idx != expected.size()) {
        પાછા ખોટું;
    }
    જો (!std::getline(f, લીટી)) {
        પાછા સાચું;
    }
    જો (લીટી != "ports") {
        પાછા ખોટું;
    }
    માટે (auto * names : { &inputs, &outputs }) {
        size_t count;
        જો (!(f >> count) || count > 100000) {
            પાછા ખોટું;
        }
        માટે (size_t i = 0; i < count; ++i) {
            std::string name;
            જો (!(f >> std::quoted(name))) {
                પાછા ખોટું;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    પાછા f.eof();
}
