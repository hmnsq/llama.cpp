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
#    define WIN32_LEAN_AND_اوسط
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#ورنہ
#    include <unistd.h>
#اگر_ختم
#اگر_تعریف_شدہ __linux__
#    include <sys/sysmacros.h>
#اگر_ختم

نامخلا {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
اندرونی uint64_t fnv1a(uint64_t h, ثابت خالی * ڈیٹا, size_t n) {
    ثابت uint8_t * p = static_cast<ثابت uint8_t *>(ڈیٹا);
    برائے (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    واپس h;
}

اندرونی uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    واپس fnv1a(h, &v, حجم(v));
}

constexpr uint64_t FNV_انحراف = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t وزن_SAMPLE_بائٹیں = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
منطقی is_وزن_src(ثابت ggml_tensor * src) {
    اگر (src == خالی_اشاریہ || src->view_src != خالی_اشاریہ || src->بفر == خالی_اشاریہ) {
        واپس جھوٹ;
    }
    واپس src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t وزن_fingerprint(ثابت ggml_tensor * t) {
    uint64_t h = FNV_انحراف;
    h = fnv1a(h, t->name, لڑی_لمبائی(t->name));
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ثابت size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    واپس fnv1a_u64(h, ggml_backend_openvino_وزن_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
سانچہ <قسم_نام F>
خالی for_each_وزن(ثابت ggml_cgraph * cgraph, F && fn) {
    std::vector<ثابت ggml_tensor *> نظر_آیا;
    برائے (صحیح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * گرہ = cgraph->nodes[i];
        برائے (صحیح s = 0; s < GGML_MAX_SRC; ++s) {
            ثابت ggml_tensor * src = گرہ->src[s];
            اگر (!is_وزن_src(src)) {
                جاری;
            }
            منطقی dup = جھوٹ;
            برائے (ثابت خودکار * p : نظر_آیا) {
                اگر (p == src) {
                    dup = سچ;
                    توڑو;
                }
            }
            اگر (dup) {
                جاری;
            }
            نظر_آیا.push_back(src);
            fn(src);
        }
    }
}

std::string ov_نسخہ_string() {
    ثابت ov::نسخہ v = ov::get_openvino_نسخہ();
    واپس std::string(v.build_نمبر ? v.build_نمبر : "unknown");
}

std::string hex64(uint64_t v) {
    حرف buf[17];
    snprintf(buf, حجم(buf), "%016llx", static_cast<غیرعلامتی طویل طویل>(v));
    واپس std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
منطقی make_dir(ثابت std::string & path) {
#اگر defined(_WIN32)
    صحیح rc = _mkdir(path.c_str());
#ورنہ
    صحیح rc = ::mkdir(path.c_str(), 0755);
#اگر_ختم
    اگر (rc == 0 || errno == EEXIST) {
        واپس سچ;
    }
    واپس جھوٹ;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
منطقی make_dirs(ثابت std::string & path) {
    اگر (path.خالی_خانے()) {
        واپس جھوٹ;
    }
    std::string acc;
    برائے (size_t i = 0; i < path.size(); ++i) {
        ثابت حرف c = path[i];
        acc.push_back(c);
        ثابت منطقی sep = (c == '/'
#اگر defined(_WIN32)
                          || c == '\\'
#اگر_ختم
        );
        // Create each intermediate component (skip a leading "/" root).
        اگر (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            اگر (!make_dir(component)) {
                واپس جھوٹ;
            }
        }
    }
    واپس make_dir(path);
}

}  // namespace

منطقی ggml_openvino_model_cache_only() {
    واپس ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

ساکن ثابت حرف * cache_settings[] = {
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

خالی ggml_openvino_model_cache_init() {
    ثابت منطقی cache_only = ggml_openvino_model_cache_only();
    ثابت std::string dir = ggml_openvino_model_cache_dir();
    اگر (dir.خالی_خانے()) {
        اگر (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        واپس;
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
#اگر_ختم
    اگر (cache_only) {
        خودکار & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_ماخذ_fingerprint(ثابت خالی * ڈیٹا, size_t size, std::vector<ggml_openvino_ماخذ_mapping> & mappings) {
    ثابت uintptr_t address = reinterpret_cast<uintptr_t>(ڈیٹا);
    خودکار شامل_ہے = [&](ثابت ggml_openvino_ماخذ_mapping & m) {
        واپس address >= m.begin && address < m.end && size <= m.end - address;
    };
    خودکار fingerprint = [&](ثابت ggml_openvino_ماخذ_mapping & m) {
        واپس fnv1a_u64(m.شناختی_عنصر, m.انحراف + address - m.begin);
    };
    برائے (ثابت خودکار & m : mappings) {
        اگر (شامل_ہے(m)) {
            واپس fingerprint(m);
        }
    }
#اگر_تعریف_شدہ __linux__
    std::ifstream maps("/proc/self/maps");
    std::string سطر;
    جب (std::getline(maps, سطر)) {
        غیرعلامتی طویل طویل begin, end, انحراف, inode;
        غیرعلامتی صحیح dev_major, dev_minor;
        حرف permissions[5];
        صحیح راستہ_آغاز = 0;
        اگر (لڑی_سے_پڑھو(سطر.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &انحراف, &dev_major, &dev_minor, &inode, &راستہ_آغاز) != 7 || inode == 0) {
            جاری;
        }
        ggml_openvino_ماخذ_mapping m{uintptr_t(begin), uintptr_t(end), انحراف, FNV_انحراف};
        اگر (!شامل_ہے(m)) {
            جاری;
        }
        ساخت stat st;
        ثابت std::string path = سطر.substr(راستہ_آغاز);
        اگر (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            توڑو;
        }
        m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, st.st_dev);
        m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, st.st_ino);
        m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, st.st_size);
        m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, st.st_mtim.tv_sec);
        m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, st.st_mtim.tv_nsec);
        m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, st.st_ctim.tv_sec);
        m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        واپس fingerprint(m);
    }
#وگرنہ_اگر defined(_WIN32)
    حافظہ_BASIC_INFORMATION حافظہ;
    اگر (VirtualQuery(ڈیٹا, &حافظہ, حجم(حافظہ)) == حجم(حافظہ) && حافظہ.Type == MEM_MAPPED) {
        std::wstring name(MAX_راستہ, L'\0');
        DWORD لمبائی = 0;
        جب (name.size() <= 32768) {
            لمبائی = GetMappedFileNameW(GetCurrentProcess(), const_cast<خالی *>(ڈیٹا), name.ڈیٹا(), static_cast<DWORD>(name.size()));
            اگر (لمبائی == 0 || لمبائی < name.size() - 1) {
                توڑو;
            }
            name.resize(name.size() * 2);
        }
        اگر (لمبائی > 0 && لمبائی < name.size() - 1) {
            name.resize(لمبائی);
            ثابت std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_اوصاف,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, خالی_اشاریہ,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_معمول, خالی_اشاریہ);
            اگر (file != INVALID_HANDLE_قدر) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ثابت منطقی valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, حجم(basic));
                CloseHandle(file);
                اگر (valid) {
                    ثابت uint64_t file_size = (uint64_t(info.n_File_سائز_بلند) << 32) | info.n_File_سائز_کم;
                    ثابت uintptr_t begin = reinterpret_cast<uintptr_t>(حافظہ.تخصیص_بنیاد);
                    اگر (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_ماخذ_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_انحراف, "win32", 5)};
                        اگر (شامل_ہے(m)) {
                            m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, info.dw_Volume_Serial_نمبر);
                            m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, (uint64_t(info.n_File_اشاریہ_بلند) << 32) | info.n_File_اشاریہ_کم);
                            m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, file_size);
                            m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, (uint64_t(info.ft_آخری_Write_وقت.dw_بلند_Date_وقت) << 32) |
                                                                 info.ft_آخری_Write_وقت.dw_کم_Date_وقت);
                            m.شناختی_عنصر = fnv1a_u64(m.شناختی_عنصر, static_cast<uint64_t>(basic.Change_وقت.QuadPart));
                            mappings.push_back(m);
                            واپس fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#اگر_ختم
    اگر (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_انحراف;
    ثابت size_t سر = std::min(size, وزن_SAMPLE_بائٹیں);
    h = fnv1a(h, ڈیٹا, سر);
    اگر (size > سر) {
        ثابت size_t دم = std::min(size - سر, وزن_SAMPLE_بائٹیں);
        h = fnv1a(h, static_cast<ثابت uint8_t *>(ڈیٹا) + size - دم, دم);
    }
    واپس h;
}

std::string ggml_openvino_model_cache_dir() {
    ثابت حرف * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    اگر (!dir || لڑی_لمبائی(dir) == 0) {
        واپس std::string();
    }
    std::string path(dir);
    اگر (ggml_openvino_model_cache_only()) {
        واپس path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    اگر (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        واپس std::string();
    }
    واپس path;
}

std::string ggml_openvino_model_cache_عارضی_راستہ(ثابت std::string & path) {
#اگر_تعریف_شدہ _WIN32
    ثابت صحیح pid = _getpid();
#ورنہ
    ثابت صحیح pid = getpid();
#اگر_ختم
    واپس path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_وقت_us());
}

uint64_t ggml_openvino_model_fingerprint(ثابت ggml_cgraph * cgraph,
                                         ثابت std::string & device,
                                         منطقی fa,
                                         ثابت int32_t * rope_params,
                                         صحیح rope_len,
                                         uint64_t extra_cfg,
                                         ثابت std::string & گراف_signature) {
    uint64_t h = FNV_انحراف;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, گراف_signature.ڈیٹا(), گراف_signature.size());
    برائے (ثابت حرف * name : cache_settings) {
        ثابت حرف * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, لڑی_لمبائی(value) + 1);
    }
    اگر (ثابت حرف * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", حجم("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, لڑی_لمبائی(debug_nodes) + 1);
    }
    اگر (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", حجم("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    برائے (صحیح i = 0; i < cgraph->n_nodes; ++i) {
        ثابت ggml_tensor * گرہ = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(گرہ->op));
        h = fnv1a(h, گرہ->name, لڑی_لمبائی(گرہ->name));
    }

    // Weights: the model identity.
    for_each_وزن(cgraph, [&](ثابت ggml_tensor * t) { h = fnv1a_u64(h, وزن_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.ڈیٹا(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    اگر (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, حجم(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ثابت std::string ver = ov_نسخہ_string();
    h = fnv1a(h, ver.ڈیٹا(), ver.size());

    واپس h;
}

std::string ggml_openvino_model_cache_blob_راستہ(ثابت std::string & dir, uint64_t fingerprint) {
    واپس dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_راستہ(ثابت std::string & dir, uint64_t fingerprint) {
    واپس dir + "/" + hex64(fingerprint) + ".manifest";
}

منطقی ggml_openvino_model_cache_write_manifest(ثابت std::string & path,
                                              ثابت ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ثابت std::vector<std::string> & inputs,
                                              ثابت std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    اگر (!f.is_open()) {
        واپس جھوٹ;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_نسخہ_string() << "\n";
    for_each_وزن(cgraph, [&](ثابت ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<صحیح>(t->type) << " " << hex64(وزن_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    برائے (ثابت خودکار * نام : { &inputs, &outputs }) {
        f << نام->size() << '\n';
        برائے (ثابت خودکار & name : *نام) {
            f << std::quoted(name) << '\n';
        }
    }
    واپس f.good();
}

منطقی ggml_openvino_model_cache_verify_manifest(ثابت std::string & path,
                                               ثابت ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    اگر (!f.is_open()) {
        واپس جھوٹ;
    }
    std::string لیبل;
    std::string val;
    // header: fingerprint
    اگر (!(f >> لیبل >> val) || لیبل != "fingerprint" || val != hex64(fingerprint)) {
        واپس جھوٹ;
    }
    // header: ov_version
    اگر (!(f >> لیبل >> val) || لیبل != "ov_version" || val != ov_نسخہ_string()) {
        واپس جھوٹ;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_وزن(cgraph, [&](ثابت ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<صحیح>(t->type)) + " " + hex64(وزن_fingerprint(t)));
    });

    size_t idx = 0;
    std::string سطر;
    std::getline(f, سطر);  // consume rest of ov_version line
    جب (idx < expected.size() && std::getline(f, سطر)) {
        اگر (سطر.خالی_خانے()) {
            جاری;
        }
        اگر (idx >= expected.size() || سطر != expected[idx]) {
            واپس جھوٹ;
        }
        ++idx;
    }
    اگر (idx != expected.size()) {
        واپس جھوٹ;
    }
    اگر (!std::getline(f, سطر)) {
        واپس سچ;
    }
    اگر (سطر != "ports") {
        واپس جھوٹ;
    }
    برائے (خودکار * نام : { &inputs, &outputs }) {
        size_t count;
        اگر (!(f >> count) || count > 100000) {
            واپس جھوٹ;
        }
        برائے (size_t i = 0; i < count; ++i) {
            std::string name;
            اگر (!(f >> std::quoted(name))) {
                واپس جھوٹ;
            }
            نام->push_back(name);
        }
    }
    f >> std::ws;
    واپس f.eof();
}
