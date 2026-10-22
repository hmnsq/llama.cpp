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

#जर defined(_WIN32)
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
ओळीत uint64_t fnv1a(uint64_t h, स्थिरांक शून्य * data, माप_प्रकार n) {
    स्थिरांक uint8_t * p = static_cast<स्थिरांक uint8_t *>(data);
    साठी (माप_प्रकार i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    परत h;
}

ओळीत uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    परत fnv1a(h, &v, आकार(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr माप_प्रकार WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
तार्किक is_weight_src(स्थिरांक ggml_tensor * src) {
    जर (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        परत असत्य;
    }
    परत src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(स्थिरांक ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, म_माप(t->name));
    साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    स्थिरांक माप_प्रकार nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    परत fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
ढाँचा <typename F>
शून्य for_each_weight(स्थिरांक ggml_cgraph * cgraph, F && fn) {
    std::vector<स्थिरांक ggml_tensor *> seen;
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; ++i) {
        स्थिरांक ggml_tensor * गाठ = cgraph->nodes[i];
        साठी (पूर्णांक s = 0; s < GGML_MAX_SRC; ++s) {
            स्थिरांक ggml_tensor * src = गाठ->src[s];
            जर (!is_weight_src(src)) {
                सुरूठेवा;
            }
            तार्किक dup = असत्य;
            साठी (स्थिरांक स्वतः * p : seen) {
                जर (p == src) {
                    dup = सत्य;
                    थांबा;
                }
            }
            जर (dup) {
                सुरूठेवा;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    स्थिरांक ov::Version v = ov::get_openvino_version();
    परत std::string(v.build_संख्या ? v.build_संख्या : "unknown");
}

std::string hex64(uint64_t v) {
    अक्षर buf[17];
    snprintf(buf, आकार(buf), "%016llx", static_cast<अचिन्हित दीर्घ दीर्घ>(v));
    परत std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
तार्किक make_dir(स्थिरांक std::string & path) {
#जर defined(_WIN32)
    पूर्णांक rc = _mkdir(path.c_str());
#अन्यथा
    पूर्णांक rc = ::सूची_गढ़ो(path.c_str(), 0755);
#endif
    जर (rc == 0 || त्रुटि_सं == EEXIST) {
        परत सत्य;
    }
    परत असत्य;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
तार्किक make_dirs(स्थिरांक std::string & path) {
    जर (path.empty()) {
        परत असत्य;
    }
    std::string acc;
    साठी (माप_प्रकार i = 0; i < path.size(); ++i) {
        स्थिरांक अक्षर c = path[i];
        acc.push_back(c);
        स्थिरांक तार्किक sep = (c == '/'
#जर defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        जर (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            जर (!make_dir(component)) {
                परत असत्य;
            }
        }
    }
    परत make_dir(path);
}

}  // namespace

तार्किक ggml_openvino_model_cache_only() {
    परत ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

स्थिर स्थिरांक अक्षर * cache_settings[] = {
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
    स्थिरांक तार्किक cache_only = ggml_openvino_model_cache_only();
    स्थिरांक std::string dir = ggml_openvino_model_cache_dir();
    जर (dir.empty()) {
        जर (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        परत;
    }
    जर (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#जर !defined(__linux__) && !defined(_WIN32)
    जर (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#endif
    जर (cache_only) {
        स्वतः & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(स्थिरांक शून्य * data, माप_प्रकार size, std::vector<ggml_openvino_source_mapping> & mappings) {
    स्थिरांक uintptr_t address = reinterpret_cast<uintptr_t>(data);
    स्वतः contains = [&](स्थिरांक ggml_openvino_source_mapping & m) {
        परत address >= m.begin && address < m.end && size <= m.end - address;
    };
    स्वतः fingerprint = [&](स्थिरांक ggml_openvino_source_mapping & m) {
        परत fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    साठी (स्थिरांक स्वतः & m : mappings) {
        जर (contains(m)) {
            परत fingerprint(m);
        }
    }
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string ओळ;
    पर्यंत (std::getline(maps, ओळ)) {
        अचिन्हित दीर्घ दीर्घ begin, end, offset, inode;
        अचिन्हित पूर्णांक dev_major, dev_minor;
        अक्षर permissions[5];
        पूर्णांक path_start = 0;
        जर (माला_पूछो(ओळ.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            सुरूठेवा;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        जर (!contains(m)) {
            सुरूठेवा;
        }
        संरचना stat st;
        स्थिरांक std::string path = ओळ.substr(path_start);
        जर (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            थांबा;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        परत fingerprint(m);
    }
#किंवाजर defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    जर (VirtualQuery(data, &memory, आकार(memory)) == आकार(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD length = 0;
        पर्यंत (name.size() <= 32768) {
            length = GetMappedFileNameW(GetCurrentProcess(), const_cast<शून्य *>(data), name.data(), static_cast<DWORD>(name.size()));
            जर (length == 0 || length < name.size() - 1) {
                थांबा;
            }
            name.resize(name.size() * 2);
        }
        जर (length > 0 && length < name.size() - 1) {
            name.resize(length);
            स्थिरांक std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            जर (file != INVALID_HANDLE_मूल्य) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                स्थिरांक तार्किक valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, आकार(basic));
                CloseHandle(file);
                जर (valid) {
                    स्थिरांक uint64_t file_size = (uint64_t(info.n_File_Size_वरचा) << 32) | info.n_File_Size_खालचा;
                    स्थिरांक uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    जर (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        जर (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_संख्या);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_Index_वरचा) << 32) | info.n_File_Index_खालचा);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_वरचा_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_खालचा_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            परत fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#endif
    जर (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    स्थिरांक माप_प्रकार head = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, data, head);
    जर (size > head) {
        स्थिरांक माप_प्रकार tail = std::min(size - head, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<स्थिरांक uint8_t *>(data) + size - tail, tail);
    }
    परत h;
}

std::string ggml_openvino_model_cache_dir() {
    स्थिरांक अक्षर * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    जर (!dir || म_माप(dir) == 0) {
        परत std::string();
    }
    std::string path(dir);
    जर (ggml_openvino_model_cache_only()) {
        परत path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    जर (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), त्रुटि_सं);
        परत std::string();
    }
    परत path;
}

std::string ggml_openvino_model_cache_temp_path(स्थिरांक std::string & path) {
#ifdef _WIN32
    स्थिरांक पूर्णांक pid = _getpid();
#अन्यथा
    स्थिरांक पूर्णांक pid = getpid();
#endif
    परत path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(स्थिरांक ggml_cgraph * cgraph,
                                         स्थिरांक std::string & device,
                                         तार्किक fa,
                                         स्थिरांक int32_t * rope_params,
                                         पूर्णांक rope_len,
                                         uint64_t extra_cfg,
                                         स्थिरांक std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.data(), graph_signature.size());
    साठी (स्थिरांक अक्षर * name : cache_settings) {
        स्थिरांक अक्षर * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, म_माप(value) + 1);
    }
    जर (स्थिरांक अक्षर * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", आकार("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, म_माप(debug_nodes) + 1);
    }
    जर (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", आकार("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; ++i) {
        स्थिरांक ggml_tensor * गाठ = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(गाठ->op));
        h = fnv1a(h, गाठ->name, म_माप(गाठ->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](स्थिरांक ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    जर (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, आकार(int32_t) * static_cast<माप_प्रकार>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    स्थिरांक std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    परत h;
}

std::string ggml_openvino_model_cache_blob_path(स्थिरांक std::string & dir, uint64_t fingerprint) {
    परत dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(स्थिरांक std::string & dir, uint64_t fingerprint) {
    परत dir + "/" + hex64(fingerprint) + ".manifest";
}

तार्किक ggml_openvino_model_cache_write_manifest(स्थिरांक std::string & path,
                                              स्थिरांक ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              स्थिरांक std::vector<std::string> & inputs,
                                              स्थिरांक std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    जर (!f.is_open()) {
        परत असत्य;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](स्थिरांक ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<पूर्णांक>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    साठी (स्थिरांक स्वतः * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        साठी (स्थिरांक स्वतः & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    परत f.good();
}

तार्किक ggml_openvino_model_cache_verify_manifest(स्थिरांक std::string & path,
                                               स्थिरांक ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    जर (!f.is_open()) {
        परत असत्य;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    जर (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        परत असत्य;
    }
    // header: ov_version
    जर (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        परत असत्य;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](स्थिरांक ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<पूर्णांक>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    माप_प्रकार idx = 0;
    std::string ओळ;
    std::getline(f, ओळ);  // consume rest of ov_version line
    पर्यंत (idx < expected.size() && std::getline(f, ओळ)) {
        जर (ओळ.empty()) {
            सुरूठेवा;
        }
        जर (idx >= expected.size() || ओळ != expected[idx]) {
            परत असत्य;
        }
        ++idx;
    }
    जर (idx != expected.size()) {
        परत असत्य;
    }
    जर (!std::getline(f, ओळ)) {
        परत सत्य;
    }
    जर (ओळ != "ports") {
        परत असत्य;
    }
    साठी (स्वतः * names : { &inputs, &outputs }) {
        माप_प्रकार count;
        जर (!(f >> count) || count > 100000) {
            परत असत्य;
        }
        साठी (माप_प्रकार i = 0; i < count; ++i) {
            std::string name;
            जर (!(f >> std::quoted(name))) {
                परत असत्य;
            }
            names->push_back(name);
        }
    }
    f >> std::ws;
    परत f.eof();
}
