#समावेश "model-cache.h"

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-openvino-extra.h"

#समावेश <cerrno>
#समावेश <algorithm>
#समावेश <cstdio>
#समावेश <cstring>
#समावेश <fstream>
#समावेश <iomanip>
#समावेश <limits>
#समावेश <sstream>
#समावेश <openvino/core/version.hpp>
#समावेश <string>
#समावेश <sys/stat.h>
#समावेश <sys/types.h>
#समावेश <vector>

#अगर defined(_WIN32)
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
#पूर्ण_अगर
#अगर_घोषित __linux__
#    include <sys/sysmacros.h>
#पूर्ण_अगर

namespace {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
अंतरभूत uint64_t fnv1a(uint64_t h, स्थिर व्योम * तथ्य, माप_प्रकार n) {
    स्थिर uint8_t * p = static_cast<स्थिर uint8_t *>(तथ्य);
    क्रम (माप_प्रकार i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    वापस h;
}

अंतरभूत uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    वापस fnv1a(h, &v, माप(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr माप_प्रकार WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
bool is_weight_src(स्थिर ggml_tensor * src) {
    अगर (src == nullptr || src->view_src != nullptr || src->बफर == nullptr) {
        वापस false;
    }
    वापस src->बफर->usage == GGML_BACKEND_बफर_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(स्थिर ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, म_माप(t->name));
    क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    स्थिर माप_प्रकार nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    वापस fnv1a_u64(h, ggml_backend_openvino_weight_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
ढाँचा <typename F>
व्योम for_each_weight(स्थिर ggml_cgraph * cgraph, F && fn) {
    std::vector<स्थिर ggml_tensor *> seen;
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; ++i) {
        स्थिर ggml_tensor * गाँठ = cgraph->nodes[i];
        क्रम (पूर्णांक s = 0; s < GGML_MAX_SRC; ++s) {
            स्थिर ggml_tensor * src = गाँठ->src[s];
            अगर (!is_weight_src(src)) {
                जारी;
            }
            bool dup = false;
            क्रम (स्थिर स्वतः * p : seen) {
                अगर (p == src) {
                    dup = true;
                    अवरोध;
                }
            }
            अगर (dup) {
                जारी;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    स्थिर ov::Version v = ov::get_openvino_version();
    वापस std::string(v.build_संख्या ? v.build_संख्या : "unknown");
}

std::string hex64(uint64_t v) {
    अक्षर buf[17];
    snprintf(buf, माप(buf), "%016llx", static_cast<अचिन्हित दीर्घ दीर्घ>(v));
    वापस std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
bool make_dir(स्थिर std::string & path) {
#अगर defined(_WIN32)
    पूर्णांक rc = _mkdir(path.c_str());
#अन्यथा
    पूर्णांक rc = ::सूची_गढ़ो(path.c_str(), 0755);
#पूर्ण_अगर
    अगर (rc == 0 || त्रुटि_सं == EEXIST) {
        वापस true;
    }
    वापस false;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
bool make_dirs(स्थिर std::string & path) {
    अगर (path.empty()) {
        वापस false;
    }
    std::string acc;
    क्रम (माप_प्रकार i = 0; i < path.size(); ++i) {
        स्थिर अक्षर c = path[i];
        acc.push_back(c);
        स्थिर bool sep = (c == '/'
#अगर defined(_WIN32)
                          || c == '\\'
#पूर्ण_अगर
        );
        // Create each intermediate component (skip a leading "/" root).
        अगर (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            अगर (!make_dir(component)) {
                वापस false;
            }
        }
    }
    वापस make_dir(path);
}

}  // namespace

bool ggml_openvino_model_cache_only() {
    वापस ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

अटल स्थिर अक्षर * cache_settings[] = {
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

व्योम ggml_openvino_model_cache_init() {
    स्थिर bool cache_only = ggml_openvino_model_cache_only();
    स्थिर std::string dir = ggml_openvino_model_cache_dir();
    अगर (dir.empty()) {
        अगर (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        वापस;
    }
    अगर (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#अगर !defined(__linux__) && !defined(_WIN32)
    अगर (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#पूर्ण_अगर
    अगर (cache_only) {
        स्वतः & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(स्थिर व्योम * तथ्य, माप_प्रकार size, std::vector<ggml_openvino_source_mapping> & mappings) {
    स्थिर uintptr_t address = reinterpret_cast<uintptr_t>(तथ्य);
    स्वतः contains = [&](स्थिर ggml_openvino_source_mapping & m) {
        वापस address >= m.begin && address < m.end && size <= m.end - address;
    };
    स्वतः fingerprint = [&](स्थिर ggml_openvino_source_mapping & m) {
        वापस fnv1a_u64(m.identity, m.offset + address - m.begin);
    };
    क्रम (स्थिर स्वतः & m : mappings) {
        अगर (contains(m)) {
            वापस fingerprint(m);
        }
    }
#अगर_घोषित __linux__
    std::अख_प्रवाह maps("/proc/self/maps");
    std::string पंक्ति;
    जबतक (std::लो_पंक्ति(maps, पंक्ति)) {
        अचिन्हित दीर्घ दीर्घ begin, end, offset, inode;
        अचिन्हित पूर्णांक dev_major, dev_minor;
        अक्षर permissions[5];
        पूर्णांक path_start = 0;
        अगर (माला_पूछो(पंक्ति.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &offset, &dev_major, &dev_minor, &inode, &path_start) != 7 || inode == 0) {
            जारी;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), offset, FNV_OFFSET};
        अगर (!contains(m)) {
            जारी;
        }
        काष्ठा stat st;
        स्थिर std::string path = पंक्ति.substr(path_start);
        अगर (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            अवरोध;
        }
        m.identity = fnv1a_u64(m.identity, st.st_dev);
        m.identity = fnv1a_u64(m.identity, st.st_ino);
        m.identity = fnv1a_u64(m.identity, st.st_size);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_mtim.tv_nsec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_sec);
        m.identity = fnv1a_u64(m.identity, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        वापस fingerprint(m);
    }
#या_अगर defined(_WIN32)
    MEMORY_BASIC_INFORMATION memory;
    अगर (VirtualQuery(तथ्य, &memory, माप(memory)) == माप(memory) && memory.Type == MEM_MAPPED) {
        std::wstring name(MAX_PATH, L'\0');
        DWORD लंबाई = 0;
        जबतक (name.size() <= 32768) {
            लंबाई = GetMappedFileNameW(GetCurrentProcess(), const_cast<व्योम *>(तथ्य), name.तथ्य(), static_cast<DWORD>(name.size()));
            अगर (लंबाई == 0 || लंबाई < name.size() - 1) {
                अवरोध;
            }
            name.resize(name.size() * 2);
        }
        अगर (लंबाई > 0 && लंबाई < name.size() - 1) {
            name.resize(लंबाई);
            स्थिर std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            अगर (file != INVALID_HANDLE_मूल्य) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                स्थिर bool valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, माप(basic));
                CloseHandle(file);
                अगर (valid) {
                    स्थिर uint64_t file_size = (uint64_t(info.n_File_आकार_ऊपरी) << 32) | info.n_File_आकार_निचला;
                    स्थिर uintptr_t begin = reinterpret_cast<uintptr_t>(memory.AllocationBase);
                    अगर (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_OFFSET, "win32", 5)};
                        अगर (contains(m)) {
                            m.identity = fnv1a_u64(m.identity, info.dw_Volume_Serial_संख्या);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.n_File_सूचकांक_ऊपरी) << 32) | info.n_File_सूचकांक_निचला);
                            m.identity = fnv1a_u64(m.identity, file_size);
                            m.identity = fnv1a_u64(m.identity, (uint64_t(info.ftLastWriteTime.dw_ऊपरी_Date_Time) << 32) |
                                                                 info.ftLastWriteTime.dw_निचला_Date_Time);
                            m.identity = fnv1a_u64(m.identity, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            वापस fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#पूर्ण_अगर
    अगर (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_OFFSET;
    स्थिर माप_प्रकार शीर्ष = std::min(size, WEIGHT_SAMPLE_BYTES);
    h = fnv1a(h, तथ्य, शीर्ष);
    अगर (size > शीर्ष) {
        स्थिर माप_प्रकार पुच्छ = std::min(size - शीर्ष, WEIGHT_SAMPLE_BYTES);
        h = fnv1a(h, static_cast<स्थिर uint8_t *>(तथ्य) + size - पुच्छ, पुच्छ);
    }
    वापस h;
}

std::string ggml_openvino_model_cache_dir() {
    स्थिर अक्षर * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    अगर (!dir || म_माप(dir) == 0) {
        वापस std::string();
    }
    std::string path(dir);
    अगर (ggml_openvino_model_cache_only()) {
        वापस path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    अगर (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), त्रुटि_सं);
        वापस std::string();
    }
    वापस path;
}

std::string ggml_openvino_model_cache_अस्थायी_path(स्थिर std::string & path) {
#अगर_घोषित _WIN32
    स्थिर पूर्णांक pid = _getpid();
#अन्यथा
    स्थिर पूर्णांक pid = getpid();
#पूर्ण_अगर
    वापस path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(स्थिर ggml_cgraph * cgraph,
                                         स्थिर std::string & device,
                                         bool fa,
                                         स्थिर int32_t * rope_params,
                                         पूर्णांक rope_len,
                                         uint64_t extra_cfg,
                                         स्थिर std::string & graph_signature) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.तथ्य(), graph_signature.size());
    क्रम (स्थिर अक्षर * name : cache_settings) {
        स्थिर अक्षर * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, म_माप(value) + 1);
    }
    अगर (स्थिर अक्षर * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", माप("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, म_माप(debug_nodes) + 1);
    }
    अगर (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", माप("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; ++i) {
        स्थिर ggml_tensor * गाँठ = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(गाँठ->op));
        h = fnv1a(h, गाँठ->name, म_माप(गाँठ->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](स्थिर ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.तथ्य(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    अगर (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, माप(int32_t) * static_cast<माप_प्रकार>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    स्थिर std::string ver = ov_version_string();
    h = fnv1a(h, ver.तथ्य(), ver.size());

    वापस h;
}

std::string ggml_openvino_model_cache_blob_path(स्थिर std::string & dir, uint64_t fingerprint) {
    वापस dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(स्थिर std::string & dir, uint64_t fingerprint) {
    वापस dir + "/" + hex64(fingerprint) + ".manifest";
}

bool ggml_openvino_model_cache_write_manifest(स्थिर std::string & path,
                                              स्थिर ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              स्थिर std::vector<std::string> & inputs,
                                              स्थिर std::vector<std::string> & outputs) {
    std::बख_प्रवाह f(path, std::पनप::छाँटो);
    अगर (!f.is_open()) {
        वापस false;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](स्थिर ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<पूर्णांक>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    क्रम (स्थिर स्वतः * names : { &inputs, &outputs }) {
        f << names->size() << '\n';
        क्रम (स्थिर स्वतः & name : *names) {
            f << std::quoted(name) << '\n';
        }
    }
    वापस f.अच्छा();
}

bool ggml_openvino_model_cache_verify_manifest(स्थिर std::string & path,
                                               स्थिर ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::अख_प्रवाह f(path);
    अगर (!f.is_open()) {
        वापस false;
    }
    std::string tag;
    std::string val;
    // header: fingerprint
    अगर (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        वापस false;
    }
    // header: ov_version
    अगर (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        वापस false;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](स्थिर ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<पूर्णांक>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    माप_प्रकार idx = 0;
    std::string पंक्ति;
    std::लो_पंक्ति(f, पंक्ति);  // consume rest of ov_version line
    जबतक (idx < expected.size() && std::लो_पंक्ति(f, पंक्ति)) {
        अगर (पंक्ति.empty()) {
            जारी;
        }
        अगर (idx >= expected.size() || पंक्ति != expected[idx]) {
            वापस false;
        }
        ++idx;
    }
    अगर (idx != expected.size()) {
        वापस false;
    }
    अगर (!std::लो_पंक्ति(f, पंक्ति)) {
        वापस true;
    }
    अगर (पंक्ति != "ports") {
        वापस false;
    }
    क्रम (स्वतः * names : { &inputs, &outputs }) {
        माप_प्रकार count;
        अगर (!(f >> count) || count > 100000) {
            वापस false;
        }
        क्रम (माप_प्रकार i = 0; i < count; ++i) {
            std::string name;
            अगर (!(f >> std::quoted(name))) {
                वापस false;
            }
            names->push_back(name);
        }
    }
    f >> std::खाली;
    वापस f.खप();
}
