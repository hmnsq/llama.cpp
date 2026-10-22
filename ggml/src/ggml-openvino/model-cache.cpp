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
#    define WIN32_LEAN_AND_میانگین
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#وإلا
#    include <unistd.h>
#پایان_اگر
#اگر_تعریف_شده __linux__
#    include <sys/sysmacros.h>
#پایان_اگر

نطاق {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
مضمن uint64_t fnv1a(uint64_t h, ثابت فراغ * داده, size_t n) {
    ثابت uint8_t * p = static_cast<ثابت uint8_t *>(داده);
    لكل (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    أرجع h;
}

مضمن uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    أرجع fnv1a(h, &v, حجم(v));
}

constexpr uint64_t FNV_جابه_جایی = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t وزن_SAMPLE_بایتها = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
منطقي is_وزن_src(ثابت ggml_tensor * src) {
    إذا (src == اشاره_تهی || src->view_src != اشاره_تهی || src->میانگیر == اشاره_تهی) {
        أرجع خطأ;
    }
    أرجع src->میانگیر->usage == GGML_BACKEND_میانگیر_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t وزن_fingerprint(ثابت ggml_tensor * t) {
    uint64_t h = FNV_جابه_جایی;
    h = fnv1a(h, t->name, طول_رشته(t->name));
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
قالب <نام_نوع F>
فراغ for_each_وزن(ثابت ggml_cgraph * cgraph, F && fn) {
    std::vector<ثابت ggml_tensor *> دیده;
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * گره = cgraph->nodes[i];
        لكل (صحيح s = 0; s < GGML_MAX_SRC; ++s) {
            ثابت ggml_tensor * src = گره->src[s];
            إذا (!is_وزن_src(src)) {
                استمر;
            }
            منطقي dup = خطأ;
            لكل (ثابت خودکار * p : دیده) {
                إذا (p == src) {
                    dup = درست;
                    اكسر;
                }
            }
            إذا (dup) {
                استمر;
            }
            دیده.push_back(src);
            fn(src);
        }
    }
}

std::string ov_نسخه_string() {
    ثابت ov::نسخه v = ov::get_openvino_نسخه();
    أرجع std::string(v.build_عدد ? v.build_عدد : "unknown");
}

std::string hex64(uint64_t v) {
    حرف buf[17];
    snprintf(buf, حجم(buf), "%016llx", static_cast<بیعلامت دراز دراز>(v));
    أرجع std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
منطقي make_dir(ثابت std::string & path) {
#إذا defined(_WIN32)
    صحيح rc = _mkdir(path.c_str());
#وإلا
    صحيح rc = ::mkdir(path.c_str(), 0755);
#پایان_اگر
    إذا (rc == 0 || errno == EEXIST) {
        أرجع درست;
    }
    أرجع خطأ;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
منطقي make_dirs(ثابت std::string & path) {
    إذا (path.خالی()) {
        أرجع خطأ;
    }
    std::string acc;
    لكل (size_t i = 0; i < path.size(); ++i) {
        ثابت حرف c = path[i];
        acc.push_back(c);
        ثابت منطقي sep = (c == '/'
#إذا defined(_WIN32)
                          || c == '\\'
#پایان_اگر
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
    إذا (dir.خالی()) {
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
#پایان_اگر
    إذا (cache_only) {
        خودکار & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_منبع_fingerprint(ثابت فراغ * داده, size_t size, std::vector<ggml_openvino_منبع_mapping> & mappings) {
    ثابت uintptr_t address = reinterpret_cast<uintptr_t>(داده);
    خودکار شامل_است = [&](ثابت ggml_openvino_منبع_mapping & m) {
        أرجع address >= m.begin && address < m.end && size <= m.end - address;
    };
    خودکار fingerprint = [&](ثابت ggml_openvino_منبع_mapping & m) {
        أرجع fnv1a_u64(m.عنصر_خنثی, m.جابه_جایی + address - m.begin);
    };
    لكل (ثابت خودکار & m : mappings) {
        إذا (شامل_است(m)) {
            أرجع fingerprint(m);
        }
    }
#اگر_تعریف_شده __linux__
    std::ifstream maps("/proc/self/maps");
    std::string سطر;
    طالما (std::getline(maps, سطر)) {
        بیعلامت دراز دراز begin, end, جابه_جایی, inode;
        بیعلامت صحيح dev_major, dev_minor;
        حرف permissions[5];
        صحيح مسیر_شروع = 0;
        إذا (از_رشته_بخوان(سطر.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &جابه_جایی, &dev_major, &dev_minor, &inode, &مسیر_شروع) != 7 || inode == 0) {
            استمر;
        }
        ggml_openvino_منبع_mapping m{uintptr_t(begin), uintptr_t(end), جابه_جایی, FNV_جابه_جایی};
        إذا (!شامل_است(m)) {
            استمر;
        }
        بنية stat st;
        ثابت std::string path = سطر.substr(مسیر_شروع);
        إذا (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            اكسر;
        }
        m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, st.st_dev);
        m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, st.st_ino);
        m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, st.st_size);
        m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, st.st_mtim.tv_sec);
        m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, st.st_mtim.tv_nsec);
        m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, st.st_ctim.tv_sec);
        m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        أرجع fingerprint(m);
    }
#وإلاإذا defined(_WIN32)
    حافظه_BASIC_INFORMATION حافظه;
    إذا (VirtualQuery(داده, &حافظه, حجم(حافظه)) == حجم(حافظه) && حافظه.Type == MEM_MAPPED) {
        std::wstring name(MAX_مسیر, L'\0');
        DWORD درازی = 0;
        طالما (name.size() <= 32768) {
            درازی = GetMappedFileNameW(GetCurrentProcess(), const_cast<فراغ *>(داده), name.داده(), static_cast<DWORD>(name.size()));
            إذا (درازی == 0 || درازی < name.size() - 1) {
                اكسر;
            }
            name.resize(name.size() * 2);
        }
        إذا (درازی > 0 && درازی < name.size() - 1) {
            name.resize(درازی);
            ثابت std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_صفات,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, اشاره_تهی,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_نرمال, اشاره_تهی);
            إذا (file != INVALID_HANDLE_مقدار) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ثابت منطقي valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, حجم(basic));
                CloseHandle(file);
                إذا (valid) {
                    ثابت uint64_t file_size = (uint64_t(info.n_File_اندازه_بالا) << 32) | info.n_File_اندازه_کم;
                    ثابت uintptr_t begin = reinterpret_cast<uintptr_t>(حافظه.تخصیص_پایه);
                    إذا (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_منبع_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_جابه_جایی, "win32", 5)};
                        إذا (شامل_است(m)) {
                            m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, info.dw_Volume_Serial_عدد);
                            m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, (uint64_t(info.n_File_شاخص_بالا) << 32) | info.n_File_شاخص_کم);
                            m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, file_size);
                            m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, (uint64_t(info.ft_آخرین_Write_زمان.dw_بالا_Date_زمان) << 32) |
                                                                 info.ft_آخرین_Write_زمان.dw_کم_Date_زمان);
                            m.عنصر_خنثی = fnv1a_u64(m.عنصر_خنثی, static_cast<uint64_t>(basic.Change_زمان.QuadPart));
                            mappings.push_back(m);
                            أرجع fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#پایان_اگر
    إذا (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_جابه_جایی;
    ثابت size_t سر = std::min(size, وزن_SAMPLE_بایتها);
    h = fnv1a(h, داده, سر);
    إذا (size > سر) {
        ثابت size_t دم = std::min(size - سر, وزن_SAMPLE_بایتها);
        h = fnv1a(h, static_cast<ثابت uint8_t *>(داده) + size - دم, دم);
    }
    أرجع h;
}

std::string ggml_openvino_model_cache_dir() {
    ثابت حرف * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    إذا (!dir || طول_رشته(dir) == 0) {
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

std::string ggml_openvino_model_cache_موقت_مسیر(ثابت std::string & path) {
#اگر_تعریف_شده _WIN32
    ثابت صحيح pid = _getpid();
#وإلا
    ثابت صحيح pid = getpid();
#پایان_اگر
    أرجع path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_زمان_us());
}

uint64_t ggml_openvino_model_fingerprint(ثابت ggml_cgraph * cgraph,
                                         ثابت std::string & device,
                                         منطقي fa,
                                         ثابت int32_t * rope_params,
                                         صحيح rope_len,
                                         uint64_t extra_cfg,
                                         ثابت std::string & گراف_signature) {
    uint64_t h = FNV_جابه_جایی;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, گراف_signature.داده(), گراف_signature.size());
    لكل (ثابت حرف * name : cache_settings) {
        ثابت حرف * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, طول_رشته(value) + 1);
    }
    إذا (ثابت حرف * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", حجم("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, طول_رشته(debug_nodes) + 1);
    }
    إذا (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", حجم("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * گره = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(گره->op));
        h = fnv1a(h, گره->name, طول_رشته(گره->name));
    }

    // Weights: the model identity.
    for_each_وزن(cgraph, [&](ثابت ggml_tensor * t) { h = fnv1a_u64(h, وزن_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.داده(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    إذا (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, حجم(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ثابت std::string ver = ov_نسخه_string();
    h = fnv1a(h, ver.داده(), ver.size());

    أرجع h;
}

std::string ggml_openvino_model_cache_blob_مسیر(ثابت std::string & dir, uint64_t fingerprint) {
    أرجع dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_مسیر(ثابت std::string & dir, uint64_t fingerprint) {
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
    f << "ov_version " << ov_نسخه_string() << "\n";
    for_each_وزن(cgraph, [&](ثابت ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<صحيح>(t->type) << " " << hex64(وزن_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    لكل (ثابت خودکار * نامها : { &inputs, &outputs }) {
        f << نامها->size() << '\n';
        لكل (ثابت خودکار & name : *نامها) {
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
    std::string برچسب;
    std::string val;
    // header: fingerprint
    إذا (!(f >> برچسب >> val) || برچسب != "fingerprint" || val != hex64(fingerprint)) {
        أرجع خطأ;
    }
    // header: ov_version
    إذا (!(f >> برچسب >> val) || برچسب != "ov_version" || val != ov_نسخه_string()) {
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
    std::string سطر;
    std::getline(f, سطر);  // consume rest of ov_version line
    طالما (idx < expected.size() && std::getline(f, سطر)) {
        إذا (سطر.خالی()) {
            استمر;
        }
        إذا (idx >= expected.size() || سطر != expected[idx]) {
            أرجع خطأ;
        }
        ++idx;
    }
    إذا (idx != expected.size()) {
        أرجع خطأ;
    }
    إذا (!std::getline(f, سطر)) {
        أرجع درست;
    }
    إذا (سطر != "ports") {
        أرجع خطأ;
    }
    لكل (خودکار * نامها : { &inputs, &outputs }) {
        size_t count;
        إذا (!(f >> count) || count > 100000) {
            أرجع خطأ;
        }
        لكل (size_t i = 0; i < count; ++i) {
            std::string name;
            إذا (!(f >> std::quoted(name))) {
                أرجع خطأ;
            }
            نامها->push_back(name);
        }
    }
    f >> std::ws;
    أرجع f.eof();
}
