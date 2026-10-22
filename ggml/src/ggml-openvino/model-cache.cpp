#ಅನ್ತರ್ಭಾವ "model-cache.h"

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino-extra.h"

#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <iomanip>
#ಅನ್ತರ್ಭಾವ <limits>
#ಅನ್ತರ್ಭಾವ <sstream>
#ಅನ್ತರ್ಭಾವ <openvino/core/version.hpp>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <vector>

#ಆದರೆ defined(_WIN32)
#    define WIN32_LEAN_AND_ಸರಾಸರಿ
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#ಇಲ್ಲದಿದ್ದರೆ
#    include <unistd.h>
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __linux__
#    include <sys/sysmacros.h>
#ಆದರೆ_ಅಂತ್ಯ

ನಾಮಾಕಾಶ {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
ಸಾಲೊಳಗೆ uint64_t fnv1a(uint64_t h, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t n) {
    ಸ್ಥಿರಾಂಕ uint8_t * p = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(ದತ್ತಾಂಶ);
    ಗಾಗಿ (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    ಹಿಂತಿರುಗಿ h;
}

ಸಾಲೊಳಗೆ uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    ಹಿಂತಿರುಗಿ fnv1a(h, &v, ಗಾತ್ರ(v));
}

constexpr uint64_t FNV_ಸ್ಥಾನಾಂತರ = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t ತೂಕ_SAMPLE_ಬೈಟುಗಳು = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
ತಾರ್ಕಿಕ is_ತೂಕ_src(ಸ್ಥಿರಾಂಕ ggml_tensor * src) {
    ಆದರೆ (src == ಶೂನ್ಯಸೂಚಕ || src->view_src != ಶೂನ್ಯಸೂಚಕ || src->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ src->ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t ತೂಕ_fingerprint(ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    uint64_t h = FNV_ಸ್ಥಾನಾಂತರ;
    h = fnv1a(h, t->name, ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(t->name));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    ಸ್ಥಿರಾಂಕ size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    ಹಿಂತಿರುಗಿ fnv1a_u64(h, ggml_backend_openvino_ತೂಕ_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು F>
ಶೂನ್ಯ for_each_ತೂಕ(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, F && fn) {
    std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> ನೋಡಿದ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; ++i) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು = cgraph->nodes[i];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < GGML_MAX_SRC; ++s) {
            ಸ್ಥಿರಾಂಕ ggml_tensor * src = ಗಂಟು->src[s];
            ಆದರೆ (!is_ತೂಕ_src(src)) {
                ಮುಂದುವರಿ;
            }
            ತಾರ್ಕಿಕ dup = ಸುಳ್ಳು;
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * p : ನೋಡಿದ) {
                ಆದರೆ (p == src) {
                    dup = ನಿಜ;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (dup) {
                ಮುಂದುವರಿ;
            }
            ನೋಡಿದ.push_back(src);
            fn(src);
        }
    }
}

std::string ov_ಆವೃತ್ತಿ_string() {
    ಸ್ಥಿರಾಂಕ ov::ಆವೃತ್ತಿ v = ov::get_openvino_ಆವೃತ್ತಿ();
    ಹಿಂತಿರುಗಿ std::string(v.build_ಸಂಖ್ಯೆ ? v.build_ಸಂಖ್ಯೆ : "unknown");
}

std::string hex64(uint64_t v) {
    ಅಕ್ಷರ buf[17];
    snprintf(buf, ಗಾತ್ರ(buf), "%016llx", static_cast<ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ>(v));
    ಹಿಂತಿರುಗಿ std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
ತಾರ್ಕಿಕ make_dir(ಸ್ಥಿರಾಂಕ std::string & path) {
#ಆದರೆ defined(_WIN32)
    ಪೂರ್ಣಾಂಕ rc = _mkdir(path.c_str());
#ಇಲ್ಲದಿದ್ದರೆ
    ಪೂರ್ಣಾಂಕ rc = ::mkdir(path.c_str(), 0755);
#ಆದರೆ_ಅಂತ್ಯ
    ಆದರೆ (rc == 0 || errno == EEXIST) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
ತಾರ್ಕಿಕ make_dirs(ಸ್ಥಿರಾಂಕ std::string & path) {
    ಆದರೆ (path.ಖಾಲಿಗಳು()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    std::string acc;
    ಗಾಗಿ (size_t i = 0; i < path.size(); ++i) {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ c = path[i];
        acc.push_back(c);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ sep = (c == '/'
#ಆದರೆ defined(_WIN32)
                          || c == '\\'
#ಆದರೆ_ಅಂತ್ಯ
        );
        // Create each intermediate component (skip a leading "/" root).
        ಆದರೆ (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            ಆದರೆ (!make_dir(component)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    }
    ಹಿಂತಿರುಗಿ make_dir(path);
}

}  // namespace

ತಾರ್ಕಿಕ ggml_openvino_model_cache_only() {
    ಹಿಂತಿರುಗಿ ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cache_settings[] = {
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

ಶೂನ್ಯ ggml_openvino_model_cache_init() {
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cache_only = ggml_openvino_model_cache_only();
    ಸ್ಥಿರಾಂಕ std::string dir = ggml_openvino_model_cache_dir();
    ಆದರೆ (dir.ಖಾಲಿಗಳು()) {
        ಆದರೆ (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#ಆದರೆ !defined(__linux__) && !defined(_WIN32)
    ಆದರೆ (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಆದರೆ (cache_only) {
        ಸ್ವಯಂಚಾಲಿತ & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_ಮೂಲ_fingerprint(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size, std::vector<ggml_openvino_ಮೂಲ_mapping> & mappings) {
    ಸ್ಥಿರಾಂಕ uintptr_t address = reinterpret_cast<uintptr_t>(ದತ್ತಾಂಶ);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಗೊಂಡಿದೆ = [&](ಸ್ಥಿರಾಂಕ ggml_openvino_ಮೂಲ_mapping & m) {
        ಹಿಂತಿರುಗಿ address >= m.begin && address < m.end && size <= m.end - address;
    };
    ಸ್ವಯಂಚಾಲಿತ fingerprint = [&](ಸ್ಥಿರಾಂಕ ggml_openvino_ಮೂಲ_mapping & m) {
        ಹಿಂತಿರುಗಿ fnv1a_u64(m.ತಟಸ್ಥ, m.ಸ್ಥಾನಾಂತರ + address - m.begin);
    };
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & m : mappings) {
        ಆದರೆ (ಒಳಗೊಂಡಿದೆ(m)) {
            ಹಿಂತಿರುಗಿ fingerprint(m);
        }
    }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __linux__
    std::ifstream maps("/proc/self/maps");
    std::string ಸಾಲು;
    ಆಗ (std::getline(maps, ಸಾಲು)) {
        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ begin, end, ಸ್ಥಾನಾಂತರ, inode;
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ dev_major, dev_minor;
        ಅಕ್ಷರ permissions[5];
        ಪೂರ್ಣಾಂಕ ಮಾರ್ಗ_ಆರಂಭ = 0;
        ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಓದು(ಸಾಲು.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &ಸ್ಥಾನಾಂತರ, &dev_major, &dev_minor, &inode, &ಮಾರ್ಗ_ಆರಂಭ) != 7 || inode == 0) {
            ಮುಂದುವರಿ;
        }
        ggml_openvino_ಮೂಲ_mapping m{uintptr_t(begin), uintptr_t(end), ಸ್ಥಾನಾಂತರ, FNV_ಸ್ಥಾನಾಂತರ};
        ಆದರೆ (!ಒಳಗೊಂಡಿದೆ(m)) {
            ಮುಂದುವರಿ;
        }
        ರಚನೆ stat st;
        ಸ್ಥಿರಾಂಕ std::string path = ಸಾಲು.substr(ಮಾರ್ಗ_ಆರಂಭ);
        ಆದರೆ (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            ನಿಲ್ಲಿಸು;
        }
        m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, st.st_dev);
        m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, st.st_ino);
        m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, st.st_size);
        m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, st.st_mtim.tv_sec);
        m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, st.st_mtim.tv_nsec);
        m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, st.st_ctim.tv_sec);
        m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        ಹಿಂತಿರುಗಿ fingerprint(m);
    }
#ಅಥವಾಆದರೆ defined(_WIN32)
    ಸ್ಮರಣೆ_BASIC_INFORMATION ಸ್ಮರಣೆ;
    ಆದರೆ (VirtualQuery(ದತ್ತಾಂಶ, &ಸ್ಮರಣೆ, ಗಾತ್ರ(ಸ್ಮರಣೆ)) == ಗಾತ್ರ(ಸ್ಮರಣೆ) && ಸ್ಮರಣೆ.Type == MEM_MAPPED) {
        std::wstring name(MAX_ಮಾರ್ಗ, L'\0');
        DWORD ಉದ್ದಳತೆ = 0;
        ಆಗ (name.size() <= 32768) {
            ಉದ್ದಳತೆ = GetMappedFileNameW(GetCurrentProcess(), const_cast<ಶೂನ್ಯ *>(ದತ್ತಾಂಶ), name.ದತ್ತಾಂಶ(), static_cast<DWORD>(name.size()));
            ಆದರೆ (ಉದ್ದಳತೆ == 0 || ಉದ್ದಳತೆ < name.size() - 1) {
                ನಿಲ್ಲಿಸು;
            }
            name.resize(name.size() * 2);
        }
        ಆದರೆ (ಉದ್ದಳತೆ > 0 && ಉದ್ದಳತೆ < name.size() - 1) {
            name.resize(ಉದ್ದಳತೆ);
            ಸ್ಥಿರಾಂಕ std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_ಗುಣಲಕ್ಷಣಗಳು,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, ಶೂನ್ಯಸೂಚಕ,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_ಸಾಮಾನ್ಯ, ಶೂನ್ಯಸೂಚಕ);
            ಆದರೆ (file != INVALID_HANDLE_ಮೌಲ್ಯ) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, ಗಾತ್ರ(basic));
                CloseHandle(file);
                ಆದರೆ (valid) {
                    ಸ್ಥಿರಾಂಕ uint64_t file_size = (uint64_t(info.n_File_ಗಾತ್ರ_ಮೇಲಿನ) << 32) | info.n_File_ಗಾತ್ರ_ಕೆಳಗಿನ;
                    ಸ್ಥಿರಾಂಕ uintptr_t begin = reinterpret_cast<uintptr_t>(ಸ್ಮರಣೆ.ಹಂಚಿಕೆ_ತಳ);
                    ಆದರೆ (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_ಮೂಲ_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_ಸ್ಥಾನಾಂತರ, "win32", 5)};
                        ಆದರೆ (ಒಳಗೊಂಡಿದೆ(m)) {
                            m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, info.dw_Volume_Serial_ಸಂಖ್ಯೆ);
                            m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, (uint64_t(info.n_File_ಸೂಚ್ಯಂಕ_ಮೇಲಿನ) << 32) | info.n_File_ಸೂಚ್ಯಂಕ_ಕೆಳಗಿನ);
                            m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, file_size);
                            m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, (uint64_t(info.ft_ಕೊನೆಯ_Write_ಸಮಯ.dw_ಮೇಲಿನ_Date_ಸಮಯ) << 32) |
                                                                 info.ft_ಕೊನೆಯ_Write_ಸಮಯ.dw_ಕೆಳಗಿನ_Date_ಸಮಯ);
                            m.ತಟಸ್ಥ = fnv1a_u64(m.ತಟಸ್ಥ, static_cast<uint64_t>(basic.Change_ಸಮಯ.QuadPart));
                            mappings.push_back(m);
                            ಹಿಂತಿರುಗಿ fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಆದರೆ (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ size_t ತಲೆ = std::min(size, ತೂಕ_SAMPLE_ಬೈಟುಗಳು);
    h = fnv1a(h, ದತ್ತಾಂಶ, ತಲೆ);
    ಆದರೆ (size > ತಲೆ) {
        ಸ್ಥಿರಾಂಕ size_t ಬಾಲ = std::min(size - ತಲೆ, ತೂಕ_SAMPLE_ಬೈಟುಗಳು);
        h = fnv1a(h, static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(ದತ್ತಾಂಶ) + size - ಬಾಲ, ಬಾಲ);
    }
    ಹಿಂತಿರುಗಿ h;
}

std::string ggml_openvino_model_cache_dir() {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    ಆದರೆ (!dir || ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(dir) == 0) {
        ಹಿಂತಿರುಗಿ std::string();
    }
    std::string path(dir);
    ಆದರೆ (ggml_openvino_model_cache_only()) {
        ಹಿಂತಿರುಗಿ path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    ಆದರೆ (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        ಹಿಂತಿರುಗಿ std::string();
    }
    ಹಿಂತಿರುಗಿ path;
}

std::string ggml_openvino_model_cache_ತಾತ್ಕಾಲಿಕ_ಮಾರ್ಗ(ಸ್ಥಿರಾಂಕ std::string & path) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pid = _getpid();
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pid = getpid();
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_ಸಮಯ_us());
}

uint64_t ggml_openvino_model_fingerprint(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph,
                                         ಸ್ಥಿರಾಂಕ std::string & device,
                                         ತಾರ್ಕಿಕ fa,
                                         ಸ್ಥಿರಾಂಕ int32_t * rope_params,
                                         ಪೂರ್ಣಾಂಕ rope_len,
                                         uint64_t extra_cfg,
                                         ಸ್ಥಿರಾಂಕ std::string & ನಕ್ಷೆ_signature) {
    uint64_t h = FNV_ಸ್ಥಾನಾಂತರ;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, ನಕ್ಷೆ_signature.ದತ್ತಾಂಶ(), ನಕ್ಷೆ_signature.size());
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name : cache_settings) {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(value) + 1);
    }
    ಆದರೆ (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", ಗಾತ್ರ("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(debug_nodes) + 1);
    }
    ಆದರೆ (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", ಗಾತ್ರ("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; ++i) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(ಗಂಟು->op));
        h = fnv1a(h, ಗಂಟು->name, ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(ಗಂಟು->name));
    }

    // Weights: the model identity.
    for_each_ತೂಕ(cgraph, [&](ಸ್ಥಿರಾಂಕ ggml_tensor * t) { h = fnv1a_u64(h, ತೂಕ_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.ದತ್ತಾಂಶ(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    ಆದರೆ (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, ಗಾತ್ರ(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    ಸ್ಥಿರಾಂಕ std::string ver = ov_ಆವೃತ್ತಿ_string();
    h = fnv1a(h, ver.ದತ್ತಾಂಶ(), ver.size());

    ಹಿಂತಿರುಗಿ h;
}

std::string ggml_openvino_model_cache_blob_ಮಾರ್ಗ(ಸ್ಥಿರಾಂಕ std::string & dir, uint64_t fingerprint) {
    ಹಿಂತಿರುಗಿ dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_ಮಾರ್ಗ(ಸ್ಥಿರಾಂಕ std::string & dir, uint64_t fingerprint) {
    ಹಿಂತಿರುಗಿ dir + "/" + hex64(fingerprint) + ".manifest";
}

ತಾರ್ಕಿಕ ggml_openvino_model_cache_write_manifest(ಸ್ಥಿರಾಂಕ std::string & path,
                                              ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              ಸ್ಥಿರಾಂಕ std::vector<std::string> & inputs,
                                              ಸ್ಥಿರಾಂಕ std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    ಆದರೆ (!f.is_open()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_ಆವೃತ್ತಿ_string() << "\n";
    for_each_ತೂಕ(cgraph, [&](ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<ಪೂರ್ಣಾಂಕ>(t->type) << " " << hex64(ತೂಕ_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ಹೆಸರುಗಳು : { &inputs, &outputs }) {
        f << ಹೆಸರುಗಳು->size() << '\n';
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & name : *ಹೆಸರುಗಳು) {
            f << std::quoted(name) << '\n';
        }
    }
    ಹಿಂತಿರುಗಿ f.good();
}

ತಾರ್ಕಿಕ ggml_openvino_model_cache_verify_manifest(ಸ್ಥಿರಾಂಕ std::string & path,
                                               ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    ಆದರೆ (!f.is_open()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    std::string ಟ್ಯಾಗ್;
    std::string val;
    // header: fingerprint
    ಆದರೆ (!(f >> ಟ್ಯಾಗ್ >> val) || ಟ್ಯಾಗ್ != "fingerprint" || val != hex64(fingerprint)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // header: ov_version
    ಆದರೆ (!(f >> ಟ್ಯಾಗ್ >> val) || ಟ್ಯಾಗ್ != "ov_version" || val != ov_ಆವೃತ್ತಿ_string()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_ತೂಕ(cgraph, [&](ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<ಪೂರ್ಣಾಂಕ>(t->type)) + " " + hex64(ತೂಕ_fingerprint(t)));
    });

    size_t idx = 0;
    std::string ಸಾಲು;
    std::getline(f, ಸಾಲು);  // consume rest of ov_version line
    ಆಗ (idx < expected.size() && std::getline(f, ಸಾಲು)) {
        ಆದರೆ (ಸಾಲು.ಖಾಲಿಗಳು()) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (idx >= expected.size() || ಸಾಲು != expected[idx]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ++idx;
    }
    ಆದರೆ (idx != expected.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!std::getline(f, ಸಾಲು)) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಆದರೆ (ಸಾಲು != "ports") {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ * ಹೆಸರುಗಳು : { &inputs, &outputs }) {
        size_t count;
        ಆದರೆ (!(f >> count) || count > 100000) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಗಾಗಿ (size_t i = 0; i < count; ++i) {
            std::string name;
            ಆದರೆ (!(f >> std::quoted(name))) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಹೆಸರುಗಳು->push_back(name);
        }
    }
    f >> std::ws;
    ಹಿಂತಿರುಗಿ f.eof();
}
