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
#    define WIN32_LEAN_AND_منځنۍ
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#وإلا
#    include <unistd.h>
#د_که_پای
#که_تعریف_شوی __linux__
#    include <sys/sysmacros.h>
#د_که_پای

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

constexpr uint64_t FNV_دننه_واټن = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t وزن_SAMPLE_بایټونه = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
منطقي is_وزن_src(ثابت ggml_tensor * src) {
    إذا (src == تش_اشاره || src->view_src != تش_اشاره || src->بفر == تش_اشاره) {
        أرجع خطأ;
    }
    أرجع src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t وزن_fingerprint(ثابت ggml_tensor * t) {
    uint64_t h = FNV_دننه_واټن;
    h = fnv1a(h, t->name, مزي_اوږدوالی(t->name));
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
کينډۍ <ډول_نوم F>
فراغ for_each_وزن(ثابت ggml_cgraph * cgraph, F && fn) {
    std::vector<ثابت ggml_tensor *> کتل_شوی;
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * غوټه = cgraph->nodes[i];
        لكل (صحيح s = 0; s < GGML_MAX_SRC; ++s) {
            ثابت ggml_tensor * src = غوټه->src[s];
            إذا (!is_وزن_src(src)) {
                استمر;
            }
            منطقي dup = خطأ;
            لكل (ثابت خپلکاری * p : کتل_شوی) {
                إذا (p == src) {
                    dup = ریښتیا;
                    اكسر;
                }
            }
            إذا (dup) {
                استمر;
            }
            کتل_شوی.push_back(src);
            fn(src);
        }
    }
}

std::string ov_نسخه_string() {
    ثابت ov::نسخه v = ov::get_openvino_نسخه();
    أرجع std::string(v.build_شمېره ? v.build_شمېره : "unknown");
}

std::string hex64(uint64_t v) {
    حرف buf[17];
    snprintf(buf, حجم(buf), "%016llx", static_cast<بینښه اوږد اوږد>(v));
    أرجع std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
منطقي make_dir(ثابت std::string & path) {
#إذا defined(_WIN32)
    صحيح rc = _mkdir(path.c_str());
#وإلا
    صحيح rc = ::mkdir(path.c_str(), 0755);
#د_که_پای
    إذا (rc == 0 || errno == EEXIST) {
        أرجع ریښتیا;
    }
    أرجع خطأ;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
منطقي make_dirs(ثابت std::string & path) {
    إذا (path.تشه()) {
        أرجع خطأ;
    }
    std::string acc;
    لكل (size_t i = 0; i < path.size(); ++i) {
        ثابت حرف c = path[i];
        acc.push_back(c);
        ثابت منطقي sep = (c == '/'
#إذا defined(_WIN32)
                          || c == '\\'
#د_که_پای
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
    إذا (dir.تشه()) {
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
#د_که_پای
    إذا (cache_only) {
        خپلکاری & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_سرچینه_fingerprint(ثابت فراغ * معلومات, size_t size, std::vector<ggml_openvino_سرچینه_mapping> & mappings) {
    ثابت uintptr_t address = reinterpret_cast<uintptr_t>(معلومات);
    خپلکاری لري = [&](ثابت ggml_openvino_سرچینه_mapping & m) {
        أرجع address >= m.begin && address < m.end && size <= m.end - address;
    };
    خپلکاری fingerprint = [&](ثابت ggml_openvino_سرچینه_mapping & m) {
        أرجع fnv1a_u64(m.بې_اغېزه_عنصر, m.دننه_واټن + address - m.begin);
    };
    لكل (ثابت خپلکاری & m : mappings) {
        إذا (لري(m)) {
            أرجع fingerprint(m);
        }
    }
#که_تعریف_شوی __linux__
    std::ifstream maps("/proc/self/maps");
    std::string کرښه;
    طالما (std::getline(maps, کرښه)) {
        بینښه اوږد اوږد begin, end, دننه_واټن, inode;
        بینښه صحيح dev_major, dev_minor;
        حرف permissions[5];
        صحيح لار_پیل = 0;
        إذا (له_مزي_ولوله(کرښه.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &دننه_واټن, &dev_major, &dev_minor, &inode, &لار_پیل) != 7 || inode == 0) {
            استمر;
        }
        ggml_openvino_سرچینه_mapping m{uintptr_t(begin), uintptr_t(end), دننه_واټن, FNV_دننه_واټن};
        إذا (!لري(m)) {
            استمر;
        }
        بنية stat st;
        ثابت std::string path = کرښه.substr(لار_پیل);
        إذا (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            اكسر;
        }
        m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, st.st_dev);
        m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, st.st_ino);
        m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, st.st_size);
        m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, st.st_mtim.tv_sec);
        m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, st.st_mtim.tv_nsec);
        m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, st.st_ctim.tv_sec);
        m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        أرجع fingerprint(m);
    }
#وإلاإذا defined(_WIN32)
    حافظه_BASIC_INFORMATION حافظه;
    إذا (VirtualQuery(معلومات, &حافظه, حجم(حافظه)) == حجم(حافظه) && حافظه.Type == MEM_MAPPED) {
        std::wstring name(MAX_لار, L'\0');
        DWORD اوږدتیا = 0;
        طالما (name.size() <= 32768) {
            اوږدتیا = GetMappedFileNameW(GetCurrentProcess(), const_cast<فراغ *>(معلومات), name.معلومات(), static_cast<DWORD>(name.size()));
            إذا (اوږدتیا == 0 || اوږدتیا < name.size() - 1) {
                اكسر;
            }
            name.resize(name.size() * 2);
        }
        إذا (اوږدتیا > 0 && اوږدتیا < name.size() - 1) {
            name.resize(اوږدتیا);
            ثابت std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ځانګړتیاوې,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, تش_اشاره,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_نورمال, تش_اشاره);
            إذا (file != INVALID_HANDLE_ارزښت) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ثابت منطقي valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, حجم(basic));
                CloseHandle(file);
                إذا (valid) {
                    ثابت uint64_t file_size = (uint64_t(info.n_File_اندازه_لوړ) << 32) | info.n_File_اندازه_ټیټ;
                    ثابت uintptr_t begin = reinterpret_cast<uintptr_t>(حافظه.ویش_بنسټ);
                    إذا (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_سرچینه_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_دننه_واټن, "win32", 5)};
                        إذا (لري(m)) {
                            m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, info.dw_Volume_Serial_شمېره);
                            m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, (uint64_t(info.n_File_شاخص_لوړ) << 32) | info.n_File_شاخص_ټیټ);
                            m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, file_size);
                            m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, (uint64_t(info.ft_وروستی_Write_وخت.dw_لوړ_Date_وخت) << 32) |
                                                                 info.ft_وروستی_Write_وخت.dw_ټیټ_Date_وخت);
                            m.بې_اغېزه_عنصر = fnv1a_u64(m.بې_اغېزه_عنصر, static_cast<uint64_t>(basic.Change_وخت.QuadPart));
                            mappings.push_back(m);
                            أرجع fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#د_که_پای
    إذا (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_دننه_واټن;
    ثابت size_t سر = std::min(size, وزن_SAMPLE_بایټونه);
    h = fnv1a(h, معلومات, سر);
    إذا (size > سر) {
        ثابت size_t لکۍ = std::min(size - سر, وزن_SAMPLE_بایټونه);
        h = fnv1a(h, static_cast<ثابت uint8_t *>(معلومات) + size - لکۍ, لکۍ);
    }
    أرجع h;
}

std::string ggml_openvino_model_cache_dir() {
    ثابت حرف * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    إذا (!dir || مزي_اوږدوالی(dir) == 0) {
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

std::string ggml_openvino_model_cache_لنډمهاله_لار(ثابت std::string & path) {
#که_تعریف_شوی _WIN32
    ثابت صحيح pid = _getpid();
#وإلا
    ثابت صحيح pid = getpid();
#د_که_پای
    أرجع path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_وخت_us());
}

uint64_t ggml_openvino_model_fingerprint(ثابت ggml_cgraph * cgraph,
                                         ثابت std::string & device,
                                         منطقي fa,
                                         ثابت int32_t * rope_params,
                                         صحيح rope_len,
                                         uint64_t extra_cfg,
                                         ثابت std::string & ګراف_signature) {
    uint64_t h = FNV_دننه_واټن;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, ګراف_signature.معلومات(), ګراف_signature.size());
    لكل (ثابت حرف * name : cache_settings) {
        ثابت حرف * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, مزي_اوږدوالی(value) + 1);
    }
    إذا (ثابت حرف * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", حجم("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, مزي_اوږدوالی(debug_nodes) + 1);
    }
    إذا (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", حجم("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * غوټه = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(غوټه->op));
        h = fnv1a(h, غوټه->name, مزي_اوږدوالی(غوټه->name));
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
    ثابت std::string ver = ov_نسخه_string();
    h = fnv1a(h, ver.معلومات(), ver.size());

    أرجع h;
}

std::string ggml_openvino_model_cache_blob_لار(ثابت std::string & dir, uint64_t fingerprint) {
    أرجع dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_لار(ثابت std::string & dir, uint64_t fingerprint) {
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
    لكل (ثابت خپلکاری * نومونه : { &inputs, &outputs }) {
        f << نومونه->size() << '\n';
        لكل (ثابت خپلکاری & name : *نومونه) {
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
    std::string نښان;
    std::string val;
    // header: fingerprint
    إذا (!(f >> نښان >> val) || نښان != "fingerprint" || val != hex64(fingerprint)) {
        أرجع خطأ;
    }
    // header: ov_version
    إذا (!(f >> نښان >> val) || نښان != "ov_version" || val != ov_نسخه_string()) {
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
    std::string کرښه;
    std::getline(f, کرښه);  // consume rest of ov_version line
    طالما (idx < expected.size() && std::getline(f, کرښه)) {
        إذا (کرښه.تشه()) {
            استمر;
        }
        إذا (idx >= expected.size() || کرښه != expected[idx]) {
            أرجع خطأ;
        }
        ++idx;
    }
    إذا (idx != expected.size()) {
        أرجع خطأ;
    }
    إذا (!std::getline(f, کرښه)) {
        أرجع ریښتیا;
    }
    إذا (کرښه != "ports") {
        أرجع خطأ;
    }
    لكل (خپلکاری * نومونه : { &inputs, &outputs }) {
        size_t count;
        إذا (!(f >> count) || count > 100000) {
            أرجع خطأ;
        }
        لكل (size_t i = 0; i < count; ++i) {
            std::string name;
            إذا (!(f >> std::quoted(name))) {
                أرجع خطأ;
            }
            نومونه->push_back(name);
        }
    }
    f >> std::ws;
    أرجع f.eof();
}
