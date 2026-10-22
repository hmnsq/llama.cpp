#అన్తర్భావ "model-cache.h"

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-openvino-extra.h"

#అన్తర్భావ <cerrno>
#అన్తర్భావ <algorithm>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstring>
#అన్తర్భావ <fstream>
#అన్తర్భావ <iomanip>
#అన్తర్భావ <limits>
#అన్తర్భావ <sstream>
#అన్తర్భావ <openvino/core/version.hpp>
#అన్తర్భావ <string>
#అన్తర్భావ <sys/stat.h>
#అన్తర్భావ <sys/types.h>
#అన్తర్భావ <vector>

#ఒకవేళ defined(_WIN32)
#    define WIN32_LEAN_AND_సగటు
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <direct.h>
#    include <process.h>
#లేకపోతే
#    include <unistd.h>
#అయితే_ముగింపు
#నిర్వచితమైతే __linux__
#    include <sys/sysmacros.h>
#అయితే_ముగింపు

నామాకాశ {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
అంతఃవరుస uint64_t fnv1a(uint64_t h, స్థిరరాశి శూన్య * దత్తాంశం, size_t n) {
    స్థిరరాశి uint8_t * p = static_cast<స్థిరరాశి uint8_t *>(దత్తాంశం);
    కొరకు (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    తిరిగి h;
}

అంతఃవరుస uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    తిరిగి fnv1a(h, &v, పరిమాణం(v));
}

constexpr uint64_t FNV_స్థానభ్రంశం = 0xcbf29ce484222325ull;

// Fallback when source-file identity is unavailable outside cache-only mode.
constexpr size_t బరువు_SAMPLE_బైట్లు = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
తార్కిక is_బరువు_src(స్థిరరాశి ggml_tensor * src) {
    ఒకవేళ (src == శూన్యసూచిక || src->view_src != శూన్యసూచిక || src->బఫరు == శూన్యసూచిక) {
        తిరిగి అబద్ధం;
    }
    తిరిగి src->బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Weight metadata and source identity; do not read repacked or unloaded buffers.
uint64_t బరువు_fingerprint(స్థిరరాశి ggml_tensor * t) {
    uint64_t h = FNV_స్థానభ్రంశం;
    h = fnv1a(h, t->name, అక్షరశ్రేణి_పొడవు(t->name));
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    స్థిరరాశి size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    తిరిగి fnv1a_u64(h, ggml_backend_openvino_బరువు_fingerprint(t));
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
మూస <రకం_పేరు F>
శూన్య for_each_బరువు(స్థిరరాశి ggml_cgraph * cgraph, F && fn) {
    std::vector<స్థిరరాశి ggml_tensor *> చూసిన;
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; ++i) {
        స్థిరరాశి ggml_tensor * కణుపు = cgraph->nodes[i];
        కొరకు (పూర్ణాంకం s = 0; s < GGML_MAX_SRC; ++s) {
            స్థిరరాశి ggml_tensor * src = కణుపు->src[s];
            ఒకవేళ (!is_బరువు_src(src)) {
                కొనసాగు;
            }
            తార్కిక dup = అబద్ధం;
            కొరకు (స్థిరరాశి స్వయంచాలక * p : చూసిన) {
                ఒకవేళ (p == src) {
                    dup = నిజం;
                    ఆపు;
                }
            }
            ఒకవేళ (dup) {
                కొనసాగు;
            }
            చూసిన.push_back(src);
            fn(src);
        }
    }
}

std::string ov_సంస్కరణ_string() {
    స్థిరరాశి ov::సంస్కరణ v = ov::get_openvino_సంస్కరణ();
    తిరిగి std::string(v.build_సంఖ్య ? v.build_సంఖ్య : "unknown");
}

std::string hex64(uint64_t v) {
    అక్షరం buf[17];
    snprintf(buf, పరిమాణం(buf), "%016llx", static_cast<గుర్తులేని దీర్ఘ దీర్ఘ>(v));
    తిరిగి std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
తార్కిక make_dir(స్థిరరాశి std::string & path) {
#ఒకవేళ defined(_WIN32)
    పూర్ణాంకం rc = _mkdir(path.c_str());
#లేకపోతే
    పూర్ణాంకం rc = ::mkdir(path.c_str(), 0755);
#అయితే_ముగింపు
    ఒకవేళ (rc == 0 || errno == EEXIST) {
        తిరిగి నిజం;
    }
    తిరిగి అబద్ధం;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
తార్కిక make_dirs(స్థిరరాశి std::string & path) {
    ఒకవేళ (path.ఖాళీలు()) {
        తిరిగి అబద్ధం;
    }
    std::string acc;
    కొరకు (size_t i = 0; i < path.size(); ++i) {
        స్థిరరాశి అక్షరం c = path[i];
        acc.push_back(c);
        స్థిరరాశి తార్కిక sep = (c == '/'
#ఒకవేళ defined(_WIN32)
                          || c == '\\'
#అయితే_ముగింపు
        );
        // Create each intermediate component (skip a leading "/" root).
        ఒకవేళ (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            ఒకవేళ (!make_dir(component)) {
                తిరిగి అబద్ధం;
            }
        }
    }
    తిరిగి make_dir(path);
}

}  // namespace

తార్కిక ggml_openvino_model_cache_only() {
    తిరిగి ggml_openvino_getenv_int("GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY") != 0;
}

స్థిర స్థిరరాశి అక్షరం * cache_settings[] = {
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

శూన్య ggml_openvino_model_cache_init() {
    స్థిరరాశి తార్కిక cache_only = ggml_openvino_model_cache_only();
    స్థిరరాశి std::string dir = ggml_openvino_model_cache_dir();
    ఒకవేళ (dir.ఖాళీలు()) {
        ఒకవేళ (cache_only) {
            GGML_ABORT("ggml-openvino: cache-only mode requires GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
        }
        తిరిగి;
    }
    ఒకవేళ (cache_only && (ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE") ||
                       ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK"))) {
        GGML_ABORT("ggml-openvino: cache-only mode requires dynamic CPU/GPU execution with caching and without fallback");
    }
#ఒకవేళ !defined(__linux__) && !defined(_WIN32)
    ఒకవేళ (cache_only) {
        GGML_ABORT("ggml-openvino: cache-only mmap identification requires Linux or Windows");
    }
#అయితే_ముగింపు
    ఒకవేళ (cache_only) {
        స్వయంచాలక & config = ggml_openvino_get_device_config();
        config.environment_variables.erase("GGML_OPENVINO_SPILL_DIR");
        config.environment_variables["GGML_OPENVINO_RELEASE_WEIGHTS"] = "0";
    }
}

uint64_t ggml_openvino_ఆధారం_fingerprint(స్థిరరాశి శూన్య * దత్తాంశం, size_t size, std::vector<ggml_openvino_ఆధారం_mapping> & mappings) {
    స్థిరరాశి uintptr_t address = reinterpret_cast<uintptr_t>(దత్తాంశం);
    స్వయంచాలక కలిగియుంది = [&](స్థిరరాశి ggml_openvino_ఆధారం_mapping & m) {
        తిరిగి address >= m.begin && address < m.end && size <= m.end - address;
    };
    స్వయంచాలక fingerprint = [&](స్థిరరాశి ggml_openvino_ఆధారం_mapping & m) {
        తిరిగి fnv1a_u64(m.తటస్థం, m.స్థానభ్రంశం + address - m.begin);
    };
    కొరకు (స్థిరరాశి స్వయంచాలక & m : mappings) {
        ఒకవేళ (కలిగియుంది(m)) {
            తిరిగి fingerprint(m);
        }
    }
#నిర్వచితమైతే __linux__
    std::ifstream maps("/proc/self/maps");
    std::string పంక్తి;
    అయితే (std::getline(maps, పంక్తి)) {
        గుర్తులేని దీర్ఘ దీర్ఘ begin, end, స్థానభ్రంశం, inode;
        గుర్తులేని పూర్ణాంకం dev_major, dev_minor;
        అక్షరం permissions[5];
        పూర్ణాంకం మార్గం_ప్రారంభం = 0;
        ఒకవేళ (అక్షరశ్రేణి_చదువు(పంక్తి.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &begin, &end, permissions,
                   &స్థానభ్రంశం, &dev_major, &dev_minor, &inode, &మార్గం_ప్రారంభం) != 7 || inode == 0) {
            కొనసాగు;
        }
        ggml_openvino_ఆధారం_mapping m{uintptr_t(begin), uintptr_t(end), స్థానభ్రంశం, FNV_స్థానభ్రంశం};
        ఒకవేళ (!కలిగియుంది(m)) {
            కొనసాగు;
        }
        నిర్మాణం stat st;
        స్థిరరాశి std::string path = పంక్తి.substr(మార్గం_ప్రారంభం);
        ఒకవేళ (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || uint64_t(st.st_ino) != inode ||
            major(st.st_dev) != dev_major || minor(st.st_dev) != dev_minor) {
            ఆపు;
        }
        m.తటస్థం = fnv1a_u64(m.తటస్థం, st.st_dev);
        m.తటస్థం = fnv1a_u64(m.తటస్థం, st.st_ino);
        m.తటస్థం = fnv1a_u64(m.తటస్థం, st.st_size);
        m.తటస్థం = fnv1a_u64(m.తటస్థం, st.st_mtim.tv_sec);
        m.తటస్థం = fnv1a_u64(m.తటస్థం, st.st_mtim.tv_nsec);
        m.తటస్థం = fnv1a_u64(m.తటస్థం, st.st_ctim.tv_sec);
        m.తటస్థం = fnv1a_u64(m.తటస్థం, st.st_ctim.tv_nsec);
        mappings.push_back(m);
        తిరిగి fingerprint(m);
    }
#లేదావేళ defined(_WIN32)
    జ్ఞాపకం_BASIC_INFORMATION జ్ఞాపకం;
    ఒకవేళ (VirtualQuery(దత్తాంశం, &జ్ఞాపకం, పరిమాణం(జ్ఞాపకం)) == పరిమాణం(జ్ఞాపకం) && జ్ఞాపకం.Type == MEM_MAPPED) {
        std::wstring name(MAX_మార్గం, L'\0');
        DWORD నిడివి = 0;
        అయితే (name.size() <= 32768) {
            నిడివి = GetMappedFileNameW(GetCurrentProcess(), const_cast<శూన్య *>(దత్తాంశం), name.దత్తాంశం(), static_cast<DWORD>(name.size()));
            ఒకవేళ (నిడివి == 0 || నిడివి < name.size() - 1) {
                ఆపు;
            }
            name.resize(name.size() * 2);
        }
        ఒకవేళ (నిడివి > 0 && నిడివి < name.size() - 1) {
            name.resize(నిడివి);
            స్థిరరాశి std::wstring path = L"\\\\?\\GLOBALROOT" + name;
            HANDLE file = CreateFileW(path.c_str(), FILE_READ_లక్షణాలు,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, శూన్యసూచిక,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_సాధారణ, శూన్యసూచిక);
            ఒకవేళ (file != INVALID_HANDLE_విలువ) {
                BY_HANDLE_FILE_INFORMATION info;
                FILE_BASIC_INFO basic;
                స్థిరరాశి తార్కిక valid = GetFileInformationByHandle(file, &info) &&
                                   GetFileInformationByHandleEx(file, FileBasicInfo, &basic, పరిమాణం(basic));
                CloseHandle(file);
                ఒకవేళ (valid) {
                    స్థిరరాశి uint64_t file_size = (uint64_t(info.n_File_కొలత_ఎగువ) << 32) | info.n_File_కొలత_దిగువ;
                    స్థిరరాశి uintptr_t begin = reinterpret_cast<uintptr_t>(జ్ఞాపకం.కేటాయింపు_పునాది);
                    ఒకవేళ (file_size <= std::numeric_limits<uintptr_t>::max() - begin) {
                        ggml_openvino_ఆధారం_mapping m{begin, begin + static_cast<uintptr_t>(file_size), 0,
                                                       fnv1a(FNV_స్థానభ్రంశం, "win32", 5)};
                        ఒకవేళ (కలిగియుంది(m)) {
                            m.తటస్థం = fnv1a_u64(m.తటస్థం, info.dw_Volume_Serial_సంఖ్య);
                            m.తటస్థం = fnv1a_u64(m.తటస్థం, (uint64_t(info.n_File_సూచీ_ఎగువ) << 32) | info.n_File_సూచీ_దిగువ);
                            m.తటస్థం = fnv1a_u64(m.తటస్థం, file_size);
                            m.తటస్థం = fnv1a_u64(m.తటస్థం, (uint64_t(info.ft_చివరి_Write_సమయం.dw_ఎగువ_Date_సమయం) << 32) |
                                                                 info.ft_చివరి_Write_సమయం.dw_దిగువ_Date_సమయం);
                            m.తటస్థం = fnv1a_u64(m.తటస్థం, static_cast<uint64_t>(basic.Change_సమయం.QuadPart));
                            mappings.push_back(m);
                            తిరిగి fingerprint(m);
                        }
                    }
                }
            }
        }
    }
#అయితే_ముగింపు
    ఒకవేళ (ggml_openvino_model_cache_only()) {
        GGML_ABORT("ggml-openvino: could not identify mapped GGUF weight; use --load-mode mmap");
    }
    uint64_t h = FNV_స్థానభ్రంశం;
    స్థిరరాశి size_t తల = std::min(size, బరువు_SAMPLE_బైట్లు);
    h = fnv1a(h, దత్తాంశం, తల);
    ఒకవేళ (size > తల) {
        స్థిరరాశి size_t తోక = std::min(size - తల, బరువు_SAMPLE_బైట్లు);
        h = fnv1a(h, static_cast<స్థిరరాశి uint8_t *>(దత్తాంశం) + size - తోక, తోక);
    }
    తిరిగి h;
}

std::string ggml_openvino_model_cache_dir() {
    స్థిరరాశి అక్షరం * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    ఒకవేళ (!dir || అక్షరశ్రేణి_పొడవు(dir) == 0) {
        తిరిగి std::string();
    }
    std::string path(dir);
    ఒకవేళ (ggml_openvino_model_cache_only()) {
        తిరిగి path;
    }
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    ఒకవేళ (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        తిరిగి std::string();
    }
    తిరిగి path;
}

std::string ggml_openvino_model_cache_తాత్కాలిక_మార్గం(స్థిరరాశి std::string & path) {
#నిర్వచితమైతే _WIN32
    స్థిరరాశి పూర్ణాంకం pid = _getpid();
#లేకపోతే
    స్థిరరాశి పూర్ణాంకం pid = getpid();
#అయితే_ముగింపు
    తిరిగి path + ".tmp." + std::to_string(pid) + "." + std::to_string(ggml_సమయం_us());
}

uint64_t ggml_openvino_model_fingerprint(స్థిరరాశి ggml_cgraph * cgraph,
                                         స్థిరరాశి std::string & device,
                                         తార్కిక fa,
                                         స్థిరరాశి int32_t * rope_params,
                                         పూర్ణాంకం rope_len,
                                         uint64_t extra_cfg,
                                         స్థిరరాశి std::string & లేఖాచిత్రం_signature) {
    uint64_t h = FNV_స్థానభ్రంశం;
    h = fnv1a_u64(h, 2);
    h = fnv1a(h, లేఖాచిత్రం_signature.దత్తాంశం(), లేఖాచిత్రం_signature.size());
    కొరకు (స్థిరరాశి అక్షరం * name : cache_settings) {
        స్థిరరాశి అక్షరం * value = ggml_openvino_getenv_str(name, "");
        h = fnv1a(h, value, అక్షరశ్రేణి_పొడవు(value) + 1);
    }
    ఒకవేళ (స్థిరరాశి అక్షరం * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        h = fnv1a(h, "GGML_OPENVINO_DEBUG_NODE", పరిమాణం("GGML_OPENVINO_DEBUG_NODE"));
        h = fnv1a(h, debug_nodes, అక్షరశ్రేణి_పొడవు(debug_nodes) + 1);
    }
    ఒకవేళ (ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) == 0) {
        h = fnv1a(h, "GGML_OPENVINO_MOE_OP=0", పరిమాణం("GGML_OPENVINO_MOE_OP=0"));
    }

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; ++i) {
        స్థిరరాశి ggml_tensor * కణుపు = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(కణుపు->op));
        h = fnv1a(h, కణుపు->name, అక్షరశ్రేణి_పొడవు(కణుపు->name));
    }

    // Weights: the model identity.
    for_each_బరువు(cgraph, [&](స్థిరరాశి ggml_tensor * t) { h = fnv1a_u64(h, బరువు_fingerprint(t)); });

    // Device, model parameters, and backend configuration.
    h = fnv1a(h, device.దత్తాంశం(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    ఒకవేళ (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, పరిమాణం(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    స్థిరరాశి std::string ver = ov_సంస్కరణ_string();
    h = fnv1a(h, ver.దత్తాంశం(), ver.size());

    తిరిగి h;
}

std::string ggml_openvino_model_cache_blob_మార్గం(స్థిరరాశి std::string & dir, uint64_t fingerprint) {
    తిరిగి dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_మార్గం(స్థిరరాశి std::string & dir, uint64_t fingerprint) {
    తిరిగి dir + "/" + hex64(fingerprint) + ".manifest";
}

తార్కిక ggml_openvino_model_cache_write_manifest(స్థిరరాశి std::string & path,
                                              స్థిరరాశి ggml_cgraph * cgraph,
                                              uint64_t fingerprint,
                                              స్థిరరాశి std::vector<std::string> & inputs,
                                              స్థిరరాశి std::vector<std::string> & outputs) {
    std::ofstream f(path, std::ios::trunc);
    ఒకవేళ (!f.is_open()) {
        తిరిగి అబద్ధం;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_సంస్కరణ_string() << "\n";
    for_each_బరువు(cgraph, [&](స్థిరరాశి ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<పూర్ణాంకం>(t->type) << " " << hex64(బరువు_fingerprint(t)) << "\n";
    });
    f << "ports\n";
    కొరకు (స్థిరరాశి స్వయంచాలక * పేర్లు : { &inputs, &outputs }) {
        f << పేర్లు->size() << '\n';
        కొరకు (స్థిరరాశి స్వయంచాలక & name : *పేర్లు) {
            f << std::quoted(name) << '\n';
        }
    }
    తిరిగి f.good();
}

తార్కిక ggml_openvino_model_cache_verify_manifest(స్థిరరాశి std::string & path,
                                               స్థిరరాశి ggml_cgraph * cgraph,
                                               uint64_t fingerprint,
                                               std::vector<std::string> & inputs,
                                               std::vector<std::string> & outputs) {
    std::ifstream f(path);
    ఒకవేళ (!f.is_open()) {
        తిరిగి అబద్ధం;
    }
    std::string ట్యాగ్;
    std::string val;
    // header: fingerprint
    ఒకవేళ (!(f >> ట్యాగ్ >> val) || ట్యాగ్ != "fingerprint" || val != hex64(fingerprint)) {
        తిరిగి అబద్ధం;
    }
    // header: ov_version
    ఒకవేళ (!(f >> ట్యాగ్ >> val) || ట్యాగ్ != "ov_version" || val != ov_సంస్కరణ_string()) {
        తిరిగి అబద్ధం;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_బరువు(cgraph, [&](స్థిరరాశి ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<పూర్ణాంకం>(t->type)) + " " + hex64(బరువు_fingerprint(t)));
    });

    size_t idx = 0;
    std::string పంక్తి;
    std::getline(f, పంక్తి);  // consume rest of ov_version line
    అయితే (idx < expected.size() && std::getline(f, పంక్తి)) {
        ఒకవేళ (పంక్తి.ఖాళీలు()) {
            కొనసాగు;
        }
        ఒకవేళ (idx >= expected.size() || పంక్తి != expected[idx]) {
            తిరిగి అబద్ధం;
        }
        ++idx;
    }
    ఒకవేళ (idx != expected.size()) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!std::getline(f, పంక్తి)) {
        తిరిగి నిజం;
    }
    ఒకవేళ (పంక్తి != "ports") {
        తిరిగి అబద్ధం;
    }
    కొరకు (స్వయంచాలక * పేర్లు : { &inputs, &outputs }) {
        size_t count;
        ఒకవేళ (!(f >> count) || count > 100000) {
            తిరిగి అబద్ధం;
        }
        కొరకు (size_t i = 0; i < count; ++i) {
            std::string name;
            ఒకవేళ (!(f >> std::quoted(name))) {
                తిరిగి అబద్ధం;
            }
            పేర్లు->push_back(name);
        }
    }
    f >> std::ws;
    తిరిగి f.eof();
}
