#ضمن "model-cache.h"

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-openvino-extra.h"

#ضمن <cerrno>
#ضمن <algorithm>
#ضمن <cstdio>
#ضمن <cstring>
#ضمن <fstream>
#ضمن <iomanip>
#ضمن <limits>
#ضمن <sstream>
#ضمن <openvino/core/version.hpp>
#ضمن <string>
#ضمن <sys/stat.h>
#ضمن <sys/types.h>
#ضمن <vector>

#إذا defined(_WIN32)
#    define WIN32_LEAN_AND_سراسري
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#وإلا
#    include <unistd.h>
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل __linux__
#    include <sys/sysmacros.h>
#جيڪڏهن_پورو

نطاق {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
مضمن uint64_t fnv1a(uint64_t h, ثابت فراغ * معلومات, size_t n) {
    ثابت uint8_t * p = static_cast<ثابت uint8_t *>(معلومات);
    لكل (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    أرجع h;
}

مضمن uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    أرجع fnv1a(h, &v, حجم(v));
}

constexpr uint64_t FNV_وٿي = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t وزن_SAMPLE_بائيٽون = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
منطقي is_وزن_src(ثابت ggml_tensor * src) {
    إذا (src == خالي_اشارو || src->view_src != خالي_اشارو || src->بفر == خالي_اشارو) {
        أرجع خطأ;
    }
    أرجع src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t وزن_fingerprint(ثابت ggml_tensor * t) {
    uint64_t h = FNV_وٿي;
    h = fnv1a(h, t->name, لڙي_ڊيگهه(t->name));
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ثابت size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    أرجع fnv1a_u64(h, ggml_backend_openvino_وزن_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
سانچو <قسم_نالو F>
فراغ for_each_وزن(ثابت ggml_cgraph * cgraph, F && fn) {
    std::vector<ثابت ggml_tensor *> ڏٺل;
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * ڳنڍ = cgraph->nodes[i];
        لكل (صحيح s = 0; s < GGML_MAX_SRC; ++s) {
            ثابت ggml_tensor * src = ڳنڍ->src[s];
            إذا (!is_وزن_src(src)) {
                استمر;
            }
            منطقي dup = خطأ;
            لكل (ثابت خودڪار * p : ڏٺل) {
                إذا (p == src) {
                    dup = سچ;
                    اكسر;
                }
            }
            إذا (dup) {
                استمر;
            }
            ڏٺل.push_back(src);
            fn(src);
        }
    }
}

std::string ov_نسخو_string() {
    ثابت ov::نسخو v = ov::get_openvino_نسخو();
    أرجع std::string(v.build_عدد ? v.build_عدد : "unknown");
}

std::string hex64(uint64_t v) {
    حرف buf[17];
    snprintf(buf, حجم(buf), "%016llx", static_cast<بنانشان ڊگهو ڊگهو>(v));
    أرجع std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
منطقي make_dir(ثابت std::string & path) {
#إذا defined(_WIN32)
    صحيح rc = _mkdir(path.c_str());
#وإلا
    صحيح rc = ::mkdir(path.c_str(), 0755);
#جيڪڏهن_پورو
    إذا (rc == 0 || errno == EEXIST) {
        أرجع سچ;
    }
    أرجع خطأ;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
منطقي make_dirs(ثابت std::string & path) {
    إذا (path.خالي_جايون()) {
        أرجع خطأ;
    }
    std::string acc;
    لكل (size_t i = 0; i < path.size(); ++i) {
        ثابت حرف c = path[i];
        acc.push_back(c);
        ثابت منطقي sep = (c == '/'
#إذا defined(_WIN32)
                          || c == '\\'
#جيڪڏهن_پورو
        );
        // Create each intermediate component (skip a leading "/" root).
        إذا (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            إذا (!make_dir(component)) {
                أرجع خطأ;
            }
        }
    }
    أرجع make_dir(path);
}

}  // namespace

منطقي ggml_openvino_model_cache_only() {
    أرجع ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

ساكن ثابت حرف * cache_settings[] = {
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

فراغ ggml_openvino_model_cache_init() {
    ثابت منطقي cache_only = ggml_openvino_model_cache_only();
    ثابت std::string dir = ggml_openvino_model_cache_dir();
    إذا (dir.خالي_جايون()) {
        إذا (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        أرجع;
    }
    إذا (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#إذا !defined(__linux__) && !defined(_WIN32)
    إذا (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#جيڪڏهن_پورو
    إذا (cache_only) {
        خودڪار & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_ذريعو_fingerprint(ثابت فراغ * معلومات, size_t size, std::vector<ggml_openvino_ذريعو_mapping> & mappings) {
    ثابت uintptr_t address = reinterpret_cast<uintptr_t>(معلومات);
    خودڪار شامل_آهي = [&](ثابت ggml_openvino_ذريعو_mapping & m) {
        أرجع address >= m.begin && address < m.end && size <= m.end - address;
    };
    خودڪار fingerprint = [&](ثابت ggml_openvino_ذريعو_mapping & m) {
        أرجع fnv1a_u64(m.سڃاڻپ_عنصر, m.وٿي + address - m.begin);
    };
    لكل (ثابت خودڪار & m : mappings) {
        إذا (شامل_آهي(m)) {
            أرجع fingerprint(m);
        }
    }
#جيڪڏهن_وصف_ٿيل __linux__
    std::ifstream maps("/proc/self/maps");
    std::string سٽ;
    طالما (std::getline(maps, سٽ)) {
        بنانشان ڊگهو ڊگهو begin, end, وٿي, inode;
        بنانشان صحيح dev_major, dev_minor;
        حرف permissions[5];
        صحيح رستو_شروعات = 0;
        إذا (لڙي_مان_پڙهو(سٽ.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &وٿي, &dev_major, &dev_minor, &inode, &رستو_شروعات) != 7 || inode == 0) {
            استمر;
        }
        ggml_openvino_ذريعو_mapping m{uintptr_t(begin), uintptr_t(end), وٿي, FNV_وٿي};
        إذا (!شامل_آهي(m)) {
            استمر;
        }
        بنية stat st;
        ثابت std::string path = سٽ.substr(رستو_شروعات);
        إذا (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            اكسر;
        }
        m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, st.st_dev);
        m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, st.st_ino);
        m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, st.st_size);
        m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, st.st_mtim.tv_sec);
        m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, st.st_mtim.tv_nsec);
        m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, st.st_ctim.tv_sec);
        m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        أرجع fingerprint(m);
    }
#وإلاإذا defined(_WIN32)
    يادگيري_BASIC_INFORMATION يادگيري;
    إذا (VirtualQuery(معلومات, &يادگيري, حجم(يادگيري)) == حجم(يادگيري) && يادگيري.Type == MEM_MAPPED) {
        std::wstring name(MAX_رستو, L'\0');
        DWORD ڊيگهائي = 0;
        طالما (name.size() <= 32768) {
            ڊيگهائي = GetMappedFileNameW(GetCurrentProcess(), const_cast<فراغ *>(معلومات), name.معلومات(), static_cast<DWORD>(name.size()));
            إذا (ڊيگهائي == 0 || ڊيگهائي < name.size() - 1) {
                اكسر;
            }
            name.resize(name.size() * 2);
        }
        إذا (ڊيگهائي > 0 && ڊيگهائي < name.size() - 1) {
            name.resize(ڊيگهائي);
            ثابت std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_خاصيتون,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, خالي_اشارو,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_عام, خالي_اشارو);
            إذا (file != INVALID_HANDLE_قدر) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ثابت منطقي valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, حجم(basic));
                CloseHandle(file);
                إذا (valid) {
                    ثابت uint64_t file_size = (uint64_t(info.n_File_سائيز_مٿاهون) << 32) | info.n_File_سائيز_هيٺيون;
                    ثابت uintptr_t begin = reinterpret_cast<uintptr_t>(يادگيري.ورهاست_بنياد);
                    إذا (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_ذريعو_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_وٿي, "win32", 5)};
                        إذا (شامل_آهي(m)) {
                            m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, info.dw_Volume_Serial_عدد);
                            m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, (uint64_t(info.n_File_اشاريو_مٿاهون) << 32) | info.n_File_اشاريو_هيٺيون);
                            m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, file_size);
                            m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, (uint64_t(info.ft_آخري_Write_وقت.dw_مٿاهون_Date_وقت) << 32) |
                                                                 info.ft_آخري_Write_وقت.dw_هيٺيون_Date_وقت);
                            m.سڃاڻپ_عنصر = fnv1a_u64(m.سڃاڻپ_عنصر, static_cast<uint64_t>(basic.Change_وقت.QuadPart));
                            mappings.push_back(m);
                            أرجع fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#جيڪڏهن_پورو
    إذا (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_وٿي;
    ثابت size_t سر = std::min(size, وزن_SAMPLE_بائيٽون);
    h = fnv1a(h, معلومات, سر);
    إذا (size > سر) {
        ثابت size_t پڇ = std::min(size - سر, وزن_SAMPLE_بائيٽون);
        h = fnv1a(h, static_cast<ثابت uint8_t *>(معلومات) + size - پڇ, پڇ);
    }
    أرجع h;
}

std::string ggml_openvino_model_cache_dir() {
    ثابت حرف * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    إذا (!dir || لڙي_ڊيگهه(dir) == 0) {
        أرجع std::string();
    }
    std::string path(dir);
    إذا (ggml_openvino_model_cache_only()) {
        أرجع path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    إذا (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        أرجع std::string();
    }
    أرجع path;
}

std::string ggml_openvino_model_cache_عارضي_رستو(ثابت std::string & path) {
#جيڪڏهن_وصف_ٿيل _WIN32
    ثابت صحيح pid = _getpid();
#وإلا
    ثابت صحيح pid = getpid();
#جيڪڏهن_پورو
    أرجع path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_وقت_us());
}

uint64_t ggml_openvino_model_fingerprint(ثابت ggml_cgraph * cgraph,
                                         ثابت std::string & device,
                                         منطقي fa,
                                         ثابت int32_t * rope_params,
                                         صحيح rope_len,
                                         uint64_t extra_cfg,
                                         ثابت std::string & گراف_signature) {
    uint64_t h = FNV_وٿي;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, گراف_signature.معلومات(), گراف_signature.size());
    لكل (ثابت حرف * name : cache_settings) {
        ثابت حرف * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, لڙي_ڊيگهه(value) + 1);
    }
    إذا (ثابت حرف * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", حجم("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, لڙي_ڊيگهه(debug_nodes) + 1);
    }
    إذا (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", حجم("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * ڳنڍ = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(ڳنڍ->op));
        h = fnv1a(h, ڳنڍ->name, لڙي_ڊيگهه(ڳنڍ->name));
    }

    // Weights: the model identity.
    for_each_وزن(cgraph, [&](ثابت ggml_tensor * t) { h = fnv1a_u64(h, وزن_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.معلومات(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    إذا (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, حجم(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ثابت std::string ver = ov_نسخو_string();
    h = fnv1a(h, ver.معلومات(), ver.size());

    أرجع h;
}

std::string ggml_openvino_model_cache_blob_رستو(ثابت std::string & dir, uint64_t fingerprint) {
    أرجع dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_رستو(ثابت std::string & dir, uint64_t fingerprint) {
    أرجع dir + "/" + hex64(fingerprint) + ".manifest";
}

منطقي ggml_openvino_model_cache_write_manifest(ثابت std::string & path,
                                              ثابت ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ثابت std::vector<std::string> & inputs,
                                              ثابت std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    إذا (!f.is_open()) {
        أرجع خطأ;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_نسخو_string() << "\n";
    for_each_وزن(cgraph, [&](ثابت ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<صحيح>(t->type) << " " << hex64(وزن_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    لكل (ثابت خودڪار * نالا : { &inputs, &outputs }) {
        f << نالا->size() << '\n';
        لكل (ثابت خودڪار & name : *نالا) {
            f << std::quoted(name) << '\n';
        }
    }
    أرجع f.good();
}

منطقي ggml_openvino_model_cache_verify_manifest(ثابت std::string & path,
                                               ثابت ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    إذا (!f.is_open()) {
        أرجع خطأ;
    }
    std::string ٽيگ;
    std::string val;
    // header: fingerprint
    إذا (!(f >> ٽيگ >> val) || ٽيگ != "fingerprint" || val != hex64(fingerprint)) {
        أرجع خطأ;
    }
    // header: ov_version
    إذا (!(f >> ٽيگ >> val) || ٽيگ != "ov_version" || val != ov_نسخو_string()) {
        أرجع خطأ;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_وزن(cgraph, [&](ثابت ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<صحيح>(t->type)) + " " + hex64(وزن_fingerprint(t)));
    });

    size_t idx = 0;
    std::string سٽ;
    std::getline(f, سٽ);  // consume rest of ov_version line
    طالما (idx < expected.size() && std::getline(f, سٽ)) {
        إذا (سٽ.خالي_جايون()) {
            استمر;
        }
        إذا (idx >= expected.size() || سٽ != expected[idx]) {
            أرجع خطأ;
        }
        ++idx;
    }
    إذا (idx != expected.size()) {
        أرجع خطأ;
    }
    إذا (!std::getline(f, سٽ)) {
        أرجع سچ;
    }
    إذا (سٽ != "ports") {
        أرجع خطأ;
    }
    لكل (خودڪار * نالا : { &inputs, &outputs }) {
        size_t count;
        إذا (!(f >> count) || count > 100000) {
            أرجع خطأ;
        }
        لكل (size_t i = 0; i < count; ++i) {
            std::string name;
            إذا (!(f >> std::quoted(name))) {
                أرجع خطأ;
            }
            نالا->push_back(name);
        }
    }
    f >> std::ws;
    أرجع f.eof();
}
