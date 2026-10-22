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
#    define WIN32_LEAN_AND_माध्यम्
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#अन्यथा
#    include <unistd.h>
#यदि_अन्तः
#यदि_परिभाषितम् __linux__
#    include <sys/sysmacros.h>
#यदि_अन्तः

नामाकाश {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
अन्तःस्थ uint64_t fnv1a(uint64_t h, ध्रुव शून्य * दत्त, size_t n) {
    ध्रुव uint8_t * p = static_cast<ध्रुव uint8_t *>(दत्त);
    प्रति (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    प्रत्यागम h;
}

अन्तःस्थ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    प्रत्यागम fnv1a(h, &v, परिमाण(v));
}

constexpr uint64_t FNV_विस्थापनम् = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t WEIGHT_SAMPLE_अष्टकानि = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
तार्किक is_weight_src(ध्रुव ggml_tensor * src) {
    यदि (src == शून्यसूचकः || src->view_src != शून्यसूचकः || src->बफर == शून्यसूचकः) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम src->बफर->usage == GGML_BACKEND_बफर_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t weight_fingerprint(ध्रुव ggml_tensor * t) {
    uint64_t h = FNV_विस्थापनम्;
    h = fnv1a(h, t->name, वाक्यदैर्घ्यम्(t->name));
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
प्रतिरूपम् <प्रकारनाम F>
शून्य for_each_weight(ध्रुव ggml_cgraph * cgraph, F && fn) {
    std::vector<ध्रुव ggml_tensor *> आलोकितम्;
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; ++i) {
        ध्रुव ggml_tensor * ग्रन्थिः = cgraph->nodes[i];
        प्रति (पूर्णाङ्क s = 0; s < GGML_MAX_SRC; ++s) {
            ध्रुव ggml_tensor * src = ग्रन्थिः->src[s];
            यदि (!is_weight_src(src)) {
                अनुवर्त;
            }
            तार्किक dup = असत्य;
            प्रति (ध्रुव स्वचालित * p : आलोकितम्) {
                यदि (p == src) {
                    dup = सत्य;
                    भङ्ग;
                }
            }
            यदि (dup) {
                अनुवर्त;
            }
            आलोकितम्.push_back(src);
            fn(src);
        }
    }
}

std::string ov_संस्करणम्_string() {
    ध्रुव ov::संस्करणम् v = ov::get_openvino_संस्करणम्();
    प्रत्यागम std::string(v.build_सङ्ख्या ? v.build_सङ्ख्या : "unknown");
}

std::string hex64(uint64_t v) {
    वर्ण buf[17];
    snprintf(buf, परिमाण(buf), "%016llx", static_cast<अचिह्नित दीर्घ दीर्घ>(v));
    प्रत्यागम std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
तार्किक make_dir(ध्रुव std::string & path) {
#यदि defined(_WIN32)
    पूर्णाङ्क rc = _mkdir(path.c_str());
#अन्यथा
    पूर्णाङ्क rc = ::mkdir(path.c_str(), 0755);
#यदि_अन्तः
    यदि (rc == 0 || errno == EEXIST) {
        प्रत्यागम सत्य;
    }
    प्रत्यागम असत्य;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
तार्किक make_dirs(ध्रुव std::string & path) {
    यदि (path.रिक्तानि()) {
        प्रत्यागम असत्य;
    }
    std::string acc;
    प्रति (size_t i = 0; i < path.size(); ++i) {
        ध्रुव वर्ण c = path[i];
        acc.push_back(c);
        ध्रुव तार्किक sep = (c == '/'
#यदि defined(_WIN32)
                          || c == '\\'
#यदि_अन्तः
        );
        // Create each intermediate component (skip a leading "/" root).
        यदि (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            यदि (!make_dir(component)) {
                प्रत्यागम असत्य;
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
    यदि (dir.रिक्तानि()) {
        यदि (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        प्रत्यागम;
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
#यदि_अन्तः
    यदि (cache_only) {
        स्वचालित & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_source_fingerprint(ध्रुव शून्य * दत्त, size_t size, std::vector<ggml_openvino_source_mapping> & mappings) {
    ध्रुव uintptr_t address = reinterpret_cast<uintptr_t>(दत्त);
    स्वचालित अन्तर्भवति = [&](ध्रुव ggml_openvino_source_mapping & m) {
        प्रत्यागम address >= m.begin && address < m.end && size <= m.end - address;
    };
    स्वचालित fingerprint = [&](ध्रुव ggml_openvino_source_mapping & m) {
        प्रत्यागम fnv1a_u64(m.तटस्थः, m.विस्थापनम् + address - m.begin);
    };
    प्रति (ध्रुव स्वचालित & m : mappings) {
        यदि (अन्तर्भवति(m)) {
            प्रत्यागम fingerprint(m);
        }
    }
#यदि_परिभाषितम् __linux__
    std::ifstream maps("/proc/self/maps");
    std::string पङ्क्तिः;
    यावत् (std::getline(maps, पङ्क्तिः)) {
        अचिह्नित दीर्घ दीर्घ begin, end, विस्थापनम्, inode;
        अचिह्नित पूर्णाङ्क dev_major, dev_minor;
        वर्ण permissions[5];
        पूर्णाङ्क मार्गः_start = 0;
        यदि (वाक्यात्_पठ(पङ्क्तिः.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &विस्थापनम्, &dev_major, &dev_minor, &inode, &मार्गः_start) != 7 || inode == 0) {
            अनुवर्त;
        }
        ggml_openvino_source_mapping m{uintptr_t(begin), uintptr_t(end), विस्थापनम्, FNV_विस्थापनम्};
        यदि (!अन्तर्भवति(m)) {
            अनुवर्त;
        }
        संरचना stat st;
        ध्रुव std::string path = पङ्क्तिः.substr(मार्गः_start);
        यदि (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            भङ्ग;
        }
        m.तटस्थः = fnv1a_u64(m.तटस्थः, st.st_dev);
        m.तटस्थः = fnv1a_u64(m.तटस्थः, st.st_ino);
        m.तटस्थः = fnv1a_u64(m.तटस्थः, st.st_size);
        m.तटस्थः = fnv1a_u64(m.तटस्थः, st.st_mtim.tv_sec);
        m.तटस्थः = fnv1a_u64(m.तटस्थः, st.st_mtim.tv_nsec);
        m.तटस्थः = fnv1a_u64(m.तटस्थः, st.st_ctim.tv_sec);
        m.तटस्थः = fnv1a_u64(m.तटस्थः, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        प्रत्यागम fingerprint(m);
    }
#नोचेद्यदि defined(_WIN32)
    स्मृतिः_BASIC_INFORMATION स्मृतिः;
    यदि (VirtualQuery(दत्त, &स्मृतिः, परिमाण(स्मृतिः)) == परिमाण(स्मृतिः) && स्मृतिः.Type == MEM_MAPPED) {
        std::wstring name(MAX_मार्गः, L'\0');
        DWORD दीर्घता = 0;
        यावत् (name.size() <= 32768) {
            दीर्घता = GetMappedFileNameW(GetCurrentProcess(), const_cast<शून्य *>(दत्त), name.दत्त(), static_cast<DWORD>(name.size()));
            यदि (दीर्घता == 0 || दीर्घता < name.size() - 1) {
                भङ्ग;
            }
            name.resize(name.size() * 2);
        }
        यदि (दीर्घता > 0 && दीर्घता < name.size() - 1) {
            name.resize(दीर्घता);
            ध्रुव std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_गुणाः,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, शून्यसूचकः,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_प्रसामान्यम्, शून्यसूचकः);
            यदि (file != INVALID_HANDLE_मानम्) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ध्रुव तार्किक valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, परिमाण(basic));
                CloseHandle(file);
                यदि (valid) {
                    ध्रुव uint64_t file_size = (uint64_t(info.n_File_आकार_उच्चः) << 32) | info.n_File_आकार_नीचः;
                    ध्रुव uintptr_t begin = reinterpret_cast<uintptr_t>(स्मृतिः.आवण्टनम्_आधारः);
                    यदि (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_source_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_विस्थापनम्, "win32", 5)};
                        यदि (अन्तर्भवति(m)) {
                            m.तटस्थः = fnv1a_u64(m.तटस्थः, info.dw_Volume_Serial_सङ्ख्या);
                            m.तटस्थः = fnv1a_u64(m.तटस्थः, (uint64_t(info.n_File_सूचकांक_उच्चः) << 32) | info.n_File_सूचकांक_नीचः);
                            m.तटस्थः = fnv1a_u64(m.तटस्थः, file_size);
                            m.तटस्थः = fnv1a_u64(m.तटस्थः, (uint64_t(info.ft_चरमः_Write_Time.dw_उच्चः_Date_Time) << 32) |
                                                                 info.ft_चरमः_Write_Time.dw_नीचः_Date_Time);
                            m.तटस्थः = fnv1a_u64(m.तटस्थः, static_cast<uint64_t>(basic.ChangeTime.QuadPart));
                            mappings.push_back(m);
                            प्रत्यागम fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#यदि_अन्तः
    यदि (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_विस्थापनम्;
    ध्रुव size_t शीर्ष = std::min(size, WEIGHT_SAMPLE_अष्टकानि);
    h = fnv1a(h, दत्त, शीर्ष);
    यदि (size > शीर्ष) {
        ध्रुव size_t पुच्छ = std::min(size - शीर्ष, WEIGHT_SAMPLE_अष्टकानि);
        h = fnv1a(h, static_cast<ध्रुव uint8_t *>(दत्त) + size - पुच्छ, पुच्छ);
    }
    प्रत्यागम h;
}

std::string ggml_openvino_model_cache_dir() {
    ध्रुव वर्ण * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    यदि (!dir || वाक्यदैर्घ्यम्(dir) == 0) {
        प्रत्यागम std::string();
    }
    std::string path(dir);
    यदि (ggml_openvino_model_cache_only()) {
        प्रत्यागम path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    यदि (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        प्रत्यागम std::string();
    }
    प्रत्यागम path;
}

std::string ggml_openvino_model_cache_अस्थायी_मार्गः(ध्रुव std::string & path) {
#यदि_परिभाषितम् _WIN32
    ध्रुव पूर्णाङ्क pid = _getpid();
#अन्यथा
    ध्रुव पूर्णाङ्क pid = getpid();
#यदि_अन्तः
    प्रत्यागम path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_time_us());
}

uint64_t ggml_openvino_model_fingerprint(ध्रुव ggml_cgraph * cgraph,
                                         ध्रुव std::string & device,
                                         तार्किक fa,
                                         ध्रुव int32_t * rope_params,
                                         पूर्णाङ्क rope_len,
                                         uint64_t extra_cfg,
                                         ध्रुव std::string & graph_signature) {
    uint64_t h = FNV_विस्थापनम्;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, graph_signature.दत्त(), graph_signature.size());
    प्रति (ध्रुव वर्ण * name : cache_settings) {
        ध्रुव वर्ण * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, वाक्यदैर्घ्यम्(value) + 1);
    }
    यदि (ध्रुव वर्ण * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", परिमाण("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, वाक्यदैर्घ्यम्(debug_nodes) + 1);
    }
    यदि (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", परिमाण("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; ++i) {
        ध्रुव ggml_tensor * ग्रन्थिः = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(ग्रन्थिः->op));
        h = fnv1a(h, ग्रन्थिः->name, वाक्यदैर्घ्यम्(ग्रन्थिः->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](ध्रुव ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.दत्त(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    यदि (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, परिमाण(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ध्रुव std::string ver = ov_संस्करणम्_string();
    h = fnv1a(h, ver.दत्त(), ver.size());

    प्रत्यागम h;
}

std::string ggml_openvino_model_cache_blob_मार्गः(ध्रुव std::string & dir, uint64_t fingerprint) {
    प्रत्यागम dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_मार्गः(ध्रुव std::string & dir, uint64_t fingerprint) {
    प्रत्यागम dir + "/" + hex64(fingerprint) + ".manifest";
}

तार्किक ggml_openvino_model_cache_write_manifest(ध्रुव std::string & path,
                                              ध्रुव ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ध्रुव std::vector<std::string> & inputs,
                                              ध्रुव std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    यदि (!f.is_open()) {
        प्रत्यागम असत्य;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_संस्करणम्_string() << "\n";
    for_each_weight(cgraph, [&](ध्रुव ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<पूर्णाङ्क>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    प्रति (ध्रुव स्वचालित * नामानि : { &inputs, &outputs }) {
        f << नामानि->size() << '\n';
        प्रति (ध्रुव स्वचालित & name : *नामानि) {
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
    यदि (!f.is_open()) {
        प्रत्यागम असत्य;
    }
    std::string चिह्नकम्;
    std::string val;
    // header: fingerprint
    यदि (!(f >> चिह्नकम् >> val) || चिह्नकम् != "fingerprint" || val != hex64(fingerprint)) {
        प्रत्यागम असत्य;
    }
    // header: ov_version
    यदि (!(f >> चिह्नकम् >> val) || चिह्नकम् != "ov_version" || val != ov_संस्करणम्_string()) {
        प्रत्यागम असत्य;
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
    यावत् (idx < expected.size() && std::getline(f, पङ्क्तिः)) {
        यदि (पङ्क्तिः.रिक्तानि()) {
            अनुवर्त;
        }
        यदि (idx >= expected.size() || पङ्क्तिः != expected[idx]) {
            प्रत्यागम असत्य;
        }
        ++idx;
    }
    यदि (idx != expected.size()) {
        प्रत्यागम असत्य;
    }
    यदि (!std::getline(f, पङ्क्तिः)) {
        प्रत्यागम सत्य;
    }
    यदि (पङ्क्तिः != "ports") {
        प्रत्यागम असत्य;
    }
    प्रति (स्वचालित * नामानि : { &inputs, &outputs }) {
        size_t count;
        यदि (!(f >> count) || count > 100000) {
            प्रत्यागम असत्य;
        }
        प्रति (size_t i = 0; i < count; ++i) {
            std::string name;
            यदि (!(f >> std::quoted(name))) {
                प्रत्यागम असत्य;
            }
            नामानि->push_back(name);
        }
    }
    f >> std::ws;
    प्रत्यागम f.eof();
}
