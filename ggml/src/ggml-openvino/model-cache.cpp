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

#यदि defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#अन्यथा
#    include <unistd.h>
#endif
#ifdef __linux__
#    include <sys/sysmacros.h>
#endif

नामाकाश {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
अन्तःस्थ uint64_t fnv1a(uint64_t h, अचर शून्य * data, माप_प्रकार n) {
    अचर uint8_t * p = static_cast<अचर uint8_t *>(data);
    लागि (माप_प्रकार i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    फर्काऊ h;
}

अन्तःस्थ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    फर्काऊ fnv1a(h, &v, परिमाण(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr माप_प्रकार WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
तार्किक is_weight_src(अचर ggml_tensor * src) {
    यदि (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        फर्काऊ असत्य;
    }
    फर्काऊ src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(अचर ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, म_माप(t->name));
    लागि (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    अचर माप_प्रकार nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    फर्काऊ fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
ढाँचा <typename F>
शून्य for_each_weight(अचर ggml_cgraph * cgraph, F && fn) {
    std::vector<अचर ggml_tensor *> seen;
    लागि (पूर्णाङ्क i = 0; i < cgraph->n_nodes; ++i) {
        अचर ggml_tensor * गाँठो = cgraph->nodes[i];
        लागि (पूर्णाङ्क s = 0; s < GGML_MAX_SRC; ++s) {
            अचर ggml_tensor * src = गाँठो->src[s];
            यदि (!is_weight_src(src)) {
                जारी;
            }
            तार्किक dup = असत्य;
            लागि (अचर स्वतः * p : seen) {
                यदि (p == src) {
                    dup = सत्य;
                    रोक;
                }
            }
            यदि (dup) {
                जारी;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    अचर ov::Version v = ov::get_openvino_version();
    फर्काऊ std::string(v.build_सङ्ख्या ? v.build_सङ्ख्या : "unknown");
}

std::string hex64(uint64_t v) {
    वर्ण buf[17];
    snprintf(buf, परिमाण(buf), "%016llx", static_cast<अचिन्हित दीर्घ दीर्घ>(v));
    फर्काऊ std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
तार्किक make_dir(अचर std::string & path) {
#यदि defined(_WIN32)
    पूर्णाङ्क rc = _mkdir(path.c_str());
#अन्यथा
    पूर्णाङ्क rc = ::सूची_गढ़ो(path.c_str(), 0755);
#endif
    यदि (rc == 0 || त्रुटि_सं == EEXIST) {
        फर्काऊ सत्य;
    }
    फर्काऊ असत्य;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
तार्किक make_dirs(अचर std::string & path) {
    यदि (path.empty()) {
        फर्काऊ असत्य;
    }
    std::string acc;
    लागि (माप_प्रकार i = 0; i < path.size(); ++i) {
        अचर वर्ण c = path[i];
        acc.push_back(c);
        अचर तार्किक sep = (c == '/'
#यदि defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        यदि (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            यदि (!make_dir(component)) {
                फर्काऊ असत्य;
            }
        }
    }
    फर्काऊ make_dir(path);
}

}  // namespace

तार्किक ggml_openvino_model_cache_only() {
    फर्काऊ ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

स्थिर अचर वर्ण * cache_settings[] = {
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
    अचर तार्किक cache_only = ggml_openvino_model_cache_only();
    अचर std::string dir = ggml_openvino_model_cache_dir();
    यदि (dir.empty()) {
        यदि (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        फर्काऊ;
    }
    यदि (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#यदि !defined(__linux__) && !defined(_WIN32)
    यदि (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    यदि (cache_only) {
        स्वतः & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(अचर शून्य * data, माप_प्रकार size, std::vector<ggml_openvino_source_mapping> & mappings) {
    अचर uintptr_t address = reinterpret_cast<uintptr_t>(data);
    स्वतः contains = [&](अचर ggml_openvino_source_mapping & m) {
        फर्काऊ address >= m.begin && address < m.end && size <= m.end - address;
    };
    स्वतः fingerprint = [&](अचर ggml_openvino_source_mapping & m) {
        फर्काऊ fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    लागि (अचर स्वतः & m : mappings) {
        यदि (contains(m)) {
            फर्काऊ fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string पङ्क्ति;
    जबसम्म (std::getline(maps, पङ्क्ति)) {
        अचिन्हित दीर्घ दीर्घ begin, end, offset, inode;
        अचिन्हित पूर्णाङ्क dev_major, dev_minor;
        वर्ण permissions[5];
        पूर्णाङ्क path_start = 0;
        यदि (माला_पूछो(पङ्क्ति.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            जारी;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        यदि (!contains(m)) {
            जारी;
        }
        संरचना stat st;
        अचर std::string path = पङ्क्ति.substr(path_start);
        यदि (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            रोक;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        फर्काऊ fingerprint(m);
    }
#अथवायदि defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    यदि (VirtualQuery(data, &memory, परिमाण(memory)) == परिमाण(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        जबसम्म (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<शून्य *>(data), name.data(), static_cast<DWORD>(name.size()));
            यदि (length == 0 || length < name.size() - 1) {
                रोक;
            }
            name.resize(name.size() * 2);
        }
        यदि (length > 0 && length < name.size() - 1) {
            name.resize(length);
            अचर std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            यदि (file != INVALID_HANDLE_मान) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                अचर तार्किक valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, परिमाण(basic));
                CloseHandle(file);
                यदि (valid) {
                    अचर uint64_t file_size = (uint64_t(info.n_File_Size_माथिल्लो) << 32) | info.n_File_Size_तल्लो;
                    अचर uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    यदि (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        यदि (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_सङ्ख्या);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_Index_माथिल्लो) << 32) | info.n_File_Index_तल्लो);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_माथिल्लो_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_तल्लो_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            फर्काऊ fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    यदि (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    अचर माप_प्रकार head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    यदि (size > head) {
        अचर माप_प्रकार tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<अचर uint8_t *>(data) + size - tail, tail);
    }
    फर्काऊ h;
}

std::string ggml_openvino_model_cache_dir() {
    अचर वर्ण * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    यदि (!dir || म_माप(dir) == 0) {
        फर्काऊ std::string();
    }
    std::string path(dir);
    यदि (ggml_openvino_model_cache_only()) {
        फर्काऊ path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    यदि (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), त्रुटि_सं);
        फर्काऊ std::string();
    }
    फर्काऊ path;
}

std::string ggml_openvino_model_cache_temp_path(अचर std::string & path) {
#ifdef _WIN32
    अचर पूर्णाङ्क pid = _getpid();
#अन्यथा
    अचर पूर्णाङ्क pid = getpid();
#endif
    फर्काऊ path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(अचर ggml_cgraph * cgraph,
                                         अचर std::string & device,
                                         तार्किक fa,
                                         अचर int32_t * rope_params,
                                         पूर्णाङ्क rope_len,
                                         uint64_t extra_cfg,
                                         अचर std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    लागि (अचर वर्ण * name : cache_settings) {
        अचर वर्ण * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, म_माप(value) + 1);
    }
    यदि (अचर वर्ण * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", परिमाण("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, म_माप(debug_nodes) + 1);
    }
    यदि (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", परिमाण("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    लागि (पूर्णाङ्क i = 0; i < cgraph->n_nodes; ++i) {
        अचर ggml_tensor * गाँठो = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(गाँठो->op));
        h = fnv1a(h, गाँठो->name, म_माप(गाँठो->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](अचर ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    यदि (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, परिमाण(int32_t) * static_cast<माप_प्रकार>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    अचर std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    फर्काऊ h;
}

std::string ggml_openvino_model_cache_blob_path(अचर std::string & dir, uint64_t fingerprint) {
    फर्काऊ dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(अचर std::string & dir, uint64_t fingerprint) {
    फर्काऊ dir + "/" + hex64(fingerprint) + ".manifest";
}

तार्किक ggml_openvino_model_cache_write_manifest(अचर std::string & path,
                                              अचर ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              अचर std::vector<std::string> & inputs,
                                              अचर std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    यदि (!f.is_open()) {
        फर्काऊ असत्य;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](अचर ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<पूर्णाङ्क>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    लागि (अचर स्वतः * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        लागि (अचर स्वतः & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    फर्काऊ f.good();
}

तार्किक ggml_openvino_model_cache_verify_manifest(अचर std::string & path,
                                               अचर ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    यदि (!f.is_open()) {
        फर्काऊ असत्य;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    यदि (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        फर्काऊ असत्य;
    }
    // header: ov_version
    यदि (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        फर्काऊ असत्य;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](अचर ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<पूर्णाङ्क>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    माप_प्रकार idx = 0;
    std::string पङ्क्ति;
    std::getline(f, पङ्क्ति);  // consume rest of ov_version line
    जबसम्म (idx < expected.size() && std::getline(f, पङ्क्ति)) {
        यदि (पङ्क्ति.empty()) {
            जारी;
        }
        यदि (idx >= expected.size() || पङ्क्ति != expected[idx]) {
            फर्काऊ असत्य;
        }
        ++idx;
    }
    यदि (idx != expected.size()) {
        फर्काऊ असत्य;
    }
    यदि (!std::getline(f, पङ्क्ति)) {
        फर्काऊ सत्य;
    }
    यदि (पङ्क्ति != "ports") {
        फर्काऊ असत्य;
    }
    लागि (स्वतः * names : { &inputs, &outputs }) {
        माप_प्रकार count;
        यदि (!(f >> count) || count > 100000) {
            फर्काऊ असत्य;
        }
        लागि (माप_प्रकार i = 0; i < count; ++i) {
            std::string name;
            यदि (!(f >> std::quoted(name))) {
                फर्काऊ असत्य;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    फर्काऊ f.eof();
}
