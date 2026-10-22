#ਅਨ੍ਤਰ੍ਭਾਵ "model-cache.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino-extra.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cerrno>
#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <iomanip>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits>
#ਅਨ੍ਤਰ੍ਭਾਵ <sstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/version.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/stat.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਜੇ defined(_WIN32)
#    define WIN32_LEAN_AND_ਔਸਤ
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#ਨਹੀਂਤਾਂ
#    include <unistd.h>
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __linux__
#    include <sys/sysmacros.h>
#ਜੇ_ਅੰਤ

ਨਾਮਾਕਾਸ਼ {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
ਸਤਰਅੰਦਰ uint64_t fnv1a(uint64_t h, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t n) {
    ਅਚਲ uint8_t * p = static_cast<ਅਚਲ uint8_t *>(ਅੰਕੜੇ);
    ਲਈ (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    ਵਾਪਸ h;
}

ਸਤਰਅੰਦਰ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    ਵਾਪਸ fnv1a(h, &v, ਆਕਾਰ(v));
}

constexpr uint64_t FNV_ਵਿੱਥ = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t ਭਾਰ_SAMPLE_ਬਾਈਟਾਂ = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
ਤਾਰਕਿਕ is_ਭਾਰ_src(ਅਚਲ ggml_tensor * src) {
    ਜੇ (src == ਖਾਲੀ_ਸੂਚਕ || src->view_src != ਖਾਲੀ_ਸੂਚਕ || src->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ src->ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t ਭਾਰ_fingerprint(ਅਚਲ ggml_tensor * t) {
    uint64_t h = FNV_ਵਿੱਥ;
    h = fnv1a(h, t->name, ਲੜੀ_ਲੰਬਾਈ(t->name));
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ਅਚਲ size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    ਵਾਪਸ fnv1a_u64(h, ggml_backend_openvino_ਭਾਰ_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ F>
ਖਾਲੀ for_each_ਭਾਰ(ਅਚਲ ggml_cgraph * cgraph, F && fn) {
    std::vector<ਅਚਲ ggml_tensor *> ਵੇਖਿਆ;
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; ++i) {
        ਅਚਲ ggml_tensor * ਗੰਢ = cgraph->nodes[i];
        ਲਈ (ਪੂਰਨ s = 0; s < GGML_MAX_SRC; ++s) {
            ਅਚਲ ggml_tensor * src = ਗੰਢ->src[s];
            ਜੇ (!is_ਭਾਰ_src(src)) {
                ਜਾਰੀ;
            }
            ਤਾਰਕਿਕ dup = ਝੂਠ;
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * p : ਵੇਖਿਆ) {
                ਜੇ (p == src) {
                    dup = ਸੱਚ;
                    ਰੋਕੋ;
                }
            }
            ਜੇ (dup) {
                ਜਾਰੀ;
            }
            ਵੇਖਿਆ.push_back(src);
            fn(src);
        }
    }
}

std::string ov_ਸੰਸਕਰਣ_string() {
    ਅਚਲ ov::ਸੰਸਕਰਣ v = ov::get_openvino_ਸੰਸਕਰਣ();
    ਵਾਪਸ std::string(v.build_ਸੰਖਿਆ ? v.build_ਸੰਖਿਆ : "unknown");
}

std::string hex64(uint64_t v) {
    ਅੱਖਰ buf[17];
    snprintf(buf, ਆਕਾਰ(buf), "%016llx", static_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ>(v));
    ਵਾਪਸ std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
ਤਾਰਕਿਕ make_dir(ਅਚਲ std::string & path) {
#ਜੇ defined(_WIN32)
    ਪੂਰਨ rc = _mkdir(path.c_str());
#ਨਹੀਂਤਾਂ
    ਪੂਰਨ rc = ::mkdir(path.c_str(), 0755);
#ਜੇ_ਅੰਤ
    ਜੇ (rc == 0 || errno == EEXIST) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਵਾਪਸ ਝੂਠ;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
ਤਾਰਕਿਕ make_dirs(ਅਚਲ std::string & path) {
    ਜੇ (path.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਵਾਪਸ ਝੂਠ;
    }
    std::string acc;
    ਲਈ (size_t i = 0; i < path.size(); ++i) {
        ਅਚਲ ਅੱਖਰ c = path[i];
        acc.push_back(c);
        ਅਚਲ ਤਾਰਕਿਕ sep = (c == '/'
#ਜੇ defined(_WIN32)
                          || c == '\\'
#ਜੇ_ਅੰਤ
        );
        // Create each intermediate component (skip a leading "/" root).
        ਜੇ (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            ਜੇ (!make_dir(component)) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
    }
    ਵਾਪਸ make_dir(path);
}

}  // namespace

ਤਾਰਕਿਕ ggml_openvino_model_cache_only() {
    ਵਾਪਸ ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * cache_settings[] = {
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

ਖਾਲੀ ggml_openvino_model_cache_init() {
    ਅਚਲ ਤਾਰਕਿਕ cache_only = ggml_openvino_model_cache_only();
    ਅਚਲ std::string dir = ggml_openvino_model_cache_dir();
    ਜੇ (dir.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਜੇ (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        ਵਾਪਸ;
    }
    ਜੇ (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#ਜੇ !defined(__linux__) && !defined(_WIN32)
    ਜੇ (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#ਜੇ_ਅੰਤ
    ਜੇ (cache_only) {
        ਸਵੈਚਾਲਿਤ & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_ਸਰੋਤ_fingerprint(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size, std::vector<ggml_openvino_ਸਰੋਤ_mapping> & mappings) {
    ਅਚਲ uintptr_t address = reinterpret_cast<uintptr_t>(ਅੰਕੜੇ);
    ਸਵੈਚਾਲਿਤ ਸ਼ਾਮਲ_ਹੈ = [&](ਅਚਲ ggml_openvino_ਸਰੋਤ_mapping & m) {
        ਵਾਪਸ address >= m.begin && address < m.end && size <= m.end - address;
    };
    ਸਵੈਚਾਲਿਤ fingerprint = [&](ਅਚਲ ggml_openvino_ਸਰੋਤ_mapping & m) {
        ਵਾਪਸ fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, m.ਵਿੱਥ + address - m.begin);
    };
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & m : mappings) {
        ਜੇ (ਸ਼ਾਮਲ_ਹੈ(m)) {
            ਵਾਪਸ fingerprint(m);
        }
    }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __linux__
    std::ifstream maps("/proc/self/maps");
    std::string ਸਤਰ;
    ਜਦੋਂ (std::getline(maps, ਸਤਰ)) {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ begin, end, ਵਿੱਥ, inode;
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ dev_major, dev_minor;
        ਅੱਖਰ permissions[5];
        ਪੂਰਨ ਰਸਤਾ_ਸ਼ੁਰੂ = 0;
        ਜੇ (ਲੜੀ_ਤੋਂ_ਪੜ੍ਹੋ(ਸਤਰ.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &ਵਿੱਥ, &dev_major, &dev_minor, &inode, &ਰਸਤਾ_ਸ਼ੁਰੂ) != 7 || inode == 0) {
            ਜਾਰੀ;
        }
        ggml_openvino_ਸਰੋਤ_mapping m{uintptr_t(begin), uintptr_t(end), ਵਿੱਥ, FNV_ਵਿੱਥ};
        ਜੇ (!ਸ਼ਾਮਲ_ਹੈ(m)) {
            ਜਾਰੀ;
        }
        ਬਣਤਰ stat st;
        ਅਚਲ std::string path = ਸਤਰ.substr(ਰਸਤਾ_ਸ਼ੁਰੂ);
        ਜੇ (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            ਰੋਕੋ;
        }
        m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, st.st_dev);
        m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, st.st_ino);
        m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, st.st_size);
        m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, st.st_mtim.tv_sec);
        m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, st.st_mtim.tv_nsec);
        m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, st.st_ctim.tv_sec);
        m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        ਵਾਪਸ fingerprint(m);
    }
#ਜਾਂਜੇ defined(_WIN32)
    ਸਮ੍ਰਿਤੀ_BASIC_INFORMATION ਸਮ੍ਰਿਤੀ;
    ਜੇ (VirtualQuery(ਅੰਕੜੇ, &ਸਮ੍ਰਿਤੀ, ਆਕਾਰ(ਸਮ੍ਰਿਤੀ)) == ਆਕਾਰ(ਸਮ੍ਰਿਤੀ) && ਸਮ੍ਰਿਤੀ.Type == MEM_MAPPED) {
        std::wstring name(MAX_ਰਸਤਾ, L'\0');
        DWORD ਲੰਮਾਈ = 0;
        ਜਦੋਂ (name.size() <= 32768) {
            ਲੰਮਾਈ = GetMappedFileNameW(GetCurrentProcess(), const_cast<ਖਾਲੀ *>(ਅੰਕੜੇ), name.ਅੰਕੜੇ(), static_cast<DWORD>(name.size()));
            ਜੇ (ਲੰਮਾਈ == 0 || ਲੰਮਾਈ < name.size() - 1) {
                ਰੋਕੋ;
            }
            name.resize(name.size() * 2);
        }
        ਜੇ (ਲੰਮਾਈ > 0 && ਲੰਮਾਈ < name.size() - 1) {
            name.resize(ਲੰਮਾਈ);
            ਅਚਲ std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ਗੁਣ,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, ਖਾਲੀ_ਸੂਚਕ,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_ਸਾਧਾਰਨ, ਖਾਲੀ_ਸੂਚਕ);
            ਜੇ (file != INVALID_HANDLE_ਮੁੱਲ) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ਅਚਲ ਤਾਰਕਿਕ valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, ਆਕਾਰ(basic));
                CloseHandle(file);
                ਜੇ (valid) {
                    ਅਚਲ uint64_t file_size = (uint64_t(info.n_File_ਮਾਪ_ਉੱਚਾ) << 32) | info.n_File_ਮਾਪ_ਨੀਵਾਂ;
                    ਅਚਲ uintptr_t begin = reinterpret_cast<uintptr_t>(ਸਮ੍ਰਿਤੀ.ਵੰਡ_ਅਧਾਰ);
                    ਜੇ (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_ਸਰੋਤ_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_ਵਿੱਥ, "win32", 5)};
                        ਜੇ (ਸ਼ਾਮਲ_ਹੈ(m)) {
                            m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, info.dw_Volume_Serial_ਸੰਖਿਆ);
                            m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, (uint64_t(info.n_File_ਸੂਚਕਾਂਕ_ਉੱਚਾ) << 32) | info.n_File_ਸੂਚਕਾਂਕ_ਨੀਵਾਂ);
                            m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, file_size);
                            m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, (uint64_t(info.ft_ਆਖਰੀ_Write_ਸਮਾਂ.dw_ਉੱਚਾ_Date_ਸਮਾਂ) << 32) |
                                                                 info.ft_ਆਖਰੀ_Write_ਸਮਾਂ.dw_ਨੀਵਾਂ_Date_ਸਮਾਂ);
                            m.ਪਛਾਣ_ਤੱਤ = fnv1a_u64(m.ਪਛਾਣ_ਤੱਤ, static_cast<uint64_t>(basic.Change_ਸਮਾਂ.QuadPart));
                            mappings.push_back(m);
                            ਵਾਪਸ fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#ਜੇ_ਅੰਤ
    ਜੇ (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_ਵਿੱਥ;
    ਅਚਲ size_t ਸਿਰ = std::min(size, ਭਾਰ_SAMPLE_ਬਾਈਟਾਂ);
    h = fnv1a(h, ਅੰਕੜੇ, ਸਿਰ);
    ਜੇ (size > ਸਿਰ) {
        ਅਚਲ size_t ਪੂਛ = std::min(size - ਸਿਰ, ਭਾਰ_SAMPLE_ਬਾਈਟਾਂ);
        h = fnv1a(h, static_cast<ਅਚਲ uint8_t *>(ਅੰਕੜੇ) + size - ਪੂਛ, ਪੂਛ);
    }
    ਵਾਪਸ h;
}

std::string ggml_openvino_model_cache_dir() {
    ਅਚਲ ਅੱਖਰ * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    ਜੇ (!dir || ਲੜੀ_ਲੰਬਾਈ(dir) == 0) {
        ਵਾਪਸ std::string();
    }
    std::string path(dir);
    ਜੇ (ggml_openvino_model_cache_only()) {
        ਵਾਪਸ path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    ਜੇ (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        ਵਾਪਸ std::string();
    }
    ਵਾਪਸ path;
}

std::string ggml_openvino_model_cache_ਆਰਜ਼ੀ_ਰਸਤਾ(ਅਚਲ std::string & path) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    ਅਚਲ ਪੂਰਨ pid = _getpid();
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਪੂਰਨ pid = getpid();
#ਜੇ_ਅੰਤ
    ਵਾਪਸ path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_ਸਮਾਂ_us());
}

uint64_t ggml_openvino_model_fingerprint(ਅਚਲ ggml_cgraph * cgraph,
                                         ਅਚਲ std::string & device,
                                         ਤਾਰਕਿਕ fa,
                                         ਅਚਲ int32_t * rope_params,
                                         ਪੂਰਨ rope_len,
                                         uint64_t extra_cfg,
                                         ਅਚਲ std::string & ਗ੍ਰਾਫ_signature) {
    uint64_t h = FNV_ਵਿੱਥ;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, ਗ੍ਰਾਫ_signature.ਅੰਕੜੇ(), ਗ੍ਰਾਫ_signature.size());
    ਲਈ (ਅਚਲ ਅੱਖਰ * name : cache_settings) {
        ਅਚਲ ਅੱਖਰ * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, ਲੜੀ_ਲੰਬਾਈ(value) + 1);
    }
    ਜੇ (ਅਚਲ ਅੱਖਰ * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", ਆਕਾਰ("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, ਲੜੀ_ਲੰਬਾਈ(debug_nodes) + 1);
    }
    ਜੇ (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", ਆਕਾਰ("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; ++i) {
        ਅਚਲ ggml_tensor * ਗੰਢ = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(ਗੰਢ->op));
        h = fnv1a(h, ਗੰਢ->name, ਲੜੀ_ਲੰਬਾਈ(ਗੰਢ->name));
    }

    // Weights: the model identity.
    for_each_ਭਾਰ(cgraph, [&](ਅਚਲ ggml_tensor * t) { h = fnv1a_u64(h, ਭਾਰ_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.ਅੰਕੜੇ(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    ਜੇ (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, ਆਕਾਰ(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ਅਚਲ std::string ver = ov_ਸੰਸਕਰਣ_string();
    h = fnv1a(h, ver.ਅੰਕੜੇ(), ver.size());

    ਵਾਪਸ h;
}

std::string ggml_openvino_model_cache_blob_ਰਸਤਾ(ਅਚਲ std::string & dir, uint64_t fingerprint) {
    ਵਾਪਸ dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_ਰਸਤਾ(ਅਚਲ std::string & dir, uint64_t fingerprint) {
    ਵਾਪਸ dir + "/" + hex64(fingerprint) + ".manifest";
}

ਤਾਰਕਿਕ ggml_openvino_model_cache_write_manifest(ਅਚਲ std::string & path,
                                              ਅਚਲ ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ਅਚਲ std::vector<std::string> & inputs,
                                              ਅਚਲ std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    ਜੇ (!f.is_open()) {
        ਵਾਪਸ ਝੂਠ;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_ਸੰਸਕਰਣ_string() << "\n";
    for_each_ਭਾਰ(cgraph, [&](ਅਚਲ ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<ਪੂਰਨ>(t->type) << " " << hex64(ਭਾਰ_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * ਨਾਂ : { &inputs, &outputs }) {
        f << ਨਾਂ->size() << '\n';
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & name : *ਨਾਂ) {
            f << std::quoted(name) << '\n';
        }
    }
    ਵਾਪਸ f.good();
}

ਤਾਰਕਿਕ ggml_openvino_model_cache_verify_manifest(ਅਚਲ std::string & path,
                                               ਅਚਲ ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    ਜੇ (!f.is_open()) {
        ਵਾਪਸ ਝੂਠ;
    }
    std::string ਟੈਗ;
    std::string val;
    // header: fingerprint
    ਜੇ (!(f >> ਟੈਗ >> val) || ਟੈਗ != "fingerprint" || val != hex64(fingerprint)) {
        ਵਾਪਸ ਝੂਠ;
    }
    // header: ov_version
    ਜੇ (!(f >> ਟੈਗ >> val) || ਟੈਗ != "ov_version" || val != ov_ਸੰਸਕਰਣ_string()) {
        ਵਾਪਸ ਝੂਠ;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_ਭਾਰ(cgraph, [&](ਅਚਲ ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<ਪੂਰਨ>(t->type)) + " " + hex64(ਭਾਰ_fingerprint(t)));
    });

    size_t idx = 0;
    std::string ਸਤਰ;
    std::getline(f, ਸਤਰ);  // consume rest of ov_version line
    ਜਦੋਂ (idx < expected.size() && std::getline(f, ਸਤਰ)) {
        ਜੇ (ਸਤਰ.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਜਾਰੀ;
        }
        ਜੇ (idx >= expected.size() || ਸਤਰ != expected[idx]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ++idx;
    }
    ਜੇ (idx != expected.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!std::getline(f, ਸਤਰ)) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਜੇ (ਸਤਰ != "ports") {
        ਵਾਪਸ ਝੂਠ;
    }
    ਲਈ (ਸਵੈਚਾਲਿਤ * ਨਾਂ : { &inputs, &outputs }) {
        size_t count;
        ਜੇ (!(f >> count) || count > 100000) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਲਈ (size_t i = 0; i < count; ++i) {
            std::string name;
            ਜੇ (!(f >> std::quoted(name))) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਨਾਂ->push_back(name);
        }
    }
    f >> std::ws;
    ਵਾਪਸ f.eof();
}
