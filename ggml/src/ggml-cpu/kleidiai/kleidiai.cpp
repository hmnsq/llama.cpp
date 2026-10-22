// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#ಅನ್ತರ್ಭಾವ <arm_neon.h>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cfloat>
#ಅನ್ತರ್ಭಾವ <cctype>
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <stdexcept>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <climits>
#ಅನ್ತರ್ಭಾವ <charconv>
#ಅನ್ತರ್ಭಾವ <system_error>
#ಯದಿ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <asm/hwcap.h>
#ಅನ್ತರ್ಭಾವ <dirent.h>
#ಅನ್ತರ್ಭಾವ <sys/auxv.h>
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ನೋಚೇದ್ಯದಿ defined(__APPLE__)
#ಅನ್ತರ್ಭಾವ <sys/sysctl.h>
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#endif

#ಅನ್ತರ್ಭಾವ "kleidiai.h"

#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-feats.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"
#ಅನ್ತರ್ಭಾವ "ggml-threading.h"
#ಅನ್ತರ್ಭಾವ "traits.h"

#ಅನ್ತರ್ಭಾವ "kernels.h"

#ಅನ್ತರ್ಭಾವ "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
ಸ್ಥಿರ constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
ಸ್ಥಿರ constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
ಸ್ಥಿರ constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
ಸ್ಥಿರ constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

ಸಂರಚನಾ ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    ಪೂರ್ಣಾಙ್ಕ sme_thread_cap; // <= 0 means "SME disabled/unknown"
    ಪೂರ್ಣಾಙ್ಕ thread_hint;    // <= 0 means "no hint"
    ಪೂರ್ಣಾಙ್ಕ chunk_multiplier;
} ಸ್ಥಿರ ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ is_sme_family(cpu_feature f) {
    ಪ್ರತ್ಯಾಗಮ (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ* cpu_feature_to_string(cpu_feature f) {
    ಯದಿ (f == CPU_FEATURE_NONE) {
        ಪ್ರತ್ಯಾಗಮ "NONE";
    } ಅನ್ಯಥಾ ಯದಿ ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        ಪ್ರತ್ಯಾಗಮ "SME2";
    } ಅನ್ಯಥಾ ಯದಿ ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        ಪ್ರತ್ಯಾಗಮ "SME";
    } ಅನ್ಯಥಾ ಯದಿ ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        ಪ್ರತ್ಯಾಗಮ "SVE";
    }
    ಅನ್ಯಥಾ ಯದಿ ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        ಪ್ರತ್ಯಾಗಮ "I8MM";
    } ಅನ್ಯಥಾ ಯದಿ ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        ಪ್ರತ್ಯಾಗಮ "DOTPROD";
    }
    ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ "UNKNOWN";
    }
}

#ಯದಿ defined(__linux__) && defined(__aarch64__)
ಸ್ಥಿರ ತಾರ್ಕಿಕ parse_cpu_dir_name(ಧ್ರುವ ವರ್ಣ* name, size_t* cpu) {
    ಯದಿ (strncmp(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ವರ್ಣ* first = name + 3;
    ಧ್ರುವ ವರ್ಣ* last = name + strlen(name);

    size_t value = 0;
    ಧ್ರುವ auto [end, ec] = std::from_chars(first, last, value, 10);

    ಯದಿ (ec != std::errc{} || end != last) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    *cpu = value;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    ಯದಿ (dir == nullptr) {
        ಪ್ರತ್ಯಾಗಮ cpus;
    }

    ಯಾವತ್ (dirent * entry = readdir(dir)) {
        size_t cpu = 0;
        ಯದಿ (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    ಪ್ರತ್ಯಾಗಮ cpus;
}
#endif

#ಯದಿ defined(__APPLE__) && defined(__aarch64__)
ಸ್ಥಿರ ತಾರ್ಕಿಕ apple_sme_counted_perf_level(std::string name) {
    ಪ್ರತಿ (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (ವರ್ಣ) std::tolower((unsigned ವರ್ಣ) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    ಪ್ರತ್ಯಾಗಮ name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

ಸ್ಥಿರ ಶೂನ್ಯ add_smcus_from_smidr(uint64_t smidr, size_t & ಅಂಕೆ_private, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ಧ್ರುವ uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ಧ್ರುವ uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ಧ್ರುವ size_t shared_ಎಣಿಕೆ = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ಧ್ರುವ uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ಧ್ರುವ uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ಧ್ರುವ uint32_t id = (affinity2 << 12) | affinity;

    ಯದಿ (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    ವಿಕಲ್ಪ (sh) {
        ಪಕ್ಷ 2: // private SMCU
            ++ಅಂಕೆ_private;
            ಭಙ್ಗ;
        ಪಕ್ಷ 3: // shared SMCU
            ಯದಿ (shared_counts[id] < shared_ಎಣಿಕೆ) {
                shared_counts[id] = shared_ಎಣಿಕೆ;
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ 0:
            ಯದಿ (id == 0) {
                ++ಅಂಕೆ_private;
            } ಅನ್ಯಥಾ ಯದಿ (shared_counts[id] < shared_ಎಣಿಕೆ) {
                shared_counts[id] = shared_ಎಣಿಕೆ;
            }
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
    }
}

ಸ್ಥಿರ size_t detect_ಅಂಕೆ_smcus() {
    ಧ್ರುವ auto runtime_feat = ggml_feats_get_arch64_runtime();
    ಯದಿ (!runtime_feat.has_sme) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

#ಯದಿ defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t ಅಂಕೆ_private = 0;
    std::map<uint32_t, size_t> shared_counts;

    ಧ್ರುವ std::vector<size_t> cpus = detect_cpu_ids();
    ಪ್ರತಿ (ಧ್ರುವ size_t cpu : cpus) {
        ಧ್ರುವ std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        ಯದಿ (!file.is_open()) {
            ಅನುವರ್ತ;
        }

        uint64_t smidr = 0;
        ಯದಿ (!(file >> std::hex >> smidr)) {
            ಅನುವರ್ತ;
        }

        add_smcus_from_smidr(smidr, ಅಂಕೆ_private, shared_counts);
    }

    size_t total = ಅಂಕೆ_private;
    ಪ್ರತಿ (ಧ್ರುವ auto & entry : shared_counts) {
        total += entry.second;
    }
    ಪ್ರತ್ಯಾಗಮ total;

#ನೋಚೇದ್ಯದಿ defined(__APPLE__) && defined(__aarch64__)
    ಪೂರ್ಣಾಙ್ಕ perf_levels = 0;
    size_t size = ಪರಿಮಾಣ(perf_levels);
    ಯದಿ (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != ಪರಿಮಾಣ(perf_levels) || perf_levels <= 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    size_t units = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < perf_levels; ++i) {
        ವರ್ಣ ಕೀಲಿ[64] = {};
        ಪೂರ್ಣಾಙ್ಕ physical_cpus = 0;
        ಪೂರ್ಣಾಙ್ಕ cpus_per_l2 = 0;

        snprintf(ಕೀಲಿ, ಪರಿಮಾಣ(ಕೀಲಿ), "hw.perflevel%d.physicalcpu", i);
        size = ಪರಿಮಾಣ(physical_cpus);
        ಯದಿ (sysctlbyname(ಕೀಲಿ, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != ಪರಿಮಾಣ(physical_cpus) || physical_cpus <= 0) {
            ಅನುವರ್ತ;
        }

        snprintf(ಕೀಲಿ, ಪರಿಮಾಣ(ಕೀಲಿ), "hw.perflevel%d.cpusperl2", i);
        size = ಪರಿಮಾಣ(cpus_per_l2);
        ಯದಿ (sysctlbyname(ಕೀಲಿ, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != ಪರಿಮಾಣ(cpus_per_l2) || cpus_per_l2 <= 0) {
            ಅನುವರ್ತ;
        }

        snprintf(ಕೀಲಿ, ಪರಿಮಾಣ(ಕೀಲಿ), "hw.perflevel%d.name", i);
        size = 0;
        ಯದಿ (sysctlbyname(ಕೀಲಿ, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            ಅನುವರ್ತ;
        }

        std::string name(size, '\0');
        ಯದಿ (sysctlbyname(ಕೀಲಿ, &name[0], &size, nullptr, 0) != 0) {
            ಅನುವರ್ತ;
        }
        name.resize(size);
        ಯಾವತ್ (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        ಯದಿ (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    ಪ್ರತ್ಯಾಗಮ units;

#ನೋಚೇದ್ಯದಿ defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    ಪ್ರತ್ಯಾಗಮ 0;

#ಅನ್ಯಥಾ
    ಪ್ರತ್ಯಾಗಮ 0;
#endif
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ parse_uint_env(ಧ್ರುವ ವರ್ಣ *s, ಧ್ರುವ ವರ್ಣ *name, ತಾರ್ಕಿಕ *ok) {
    ಯದಿ (!s) { *ok = ಅಸತ್ಯ; ಪ್ರತ್ಯಾಗಮ 0; }
    ವರ್ಣ *end = nullptr;
    long v = strtol(s, &end, 10);
    ಯದಿ (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = ಅಸತ್ಯ;
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ಯದಿ (v < 0 || v > INT_MAX) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = ಅಸತ್ಯ;
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    *ok = ಸತ್ಯ;
    ಪ್ರತ್ಯಾಗಮ (ಪೂರ್ಣಾಙ್ಕ)v;
}

ಸ್ಥಿರ ಶೂನ್ಯ init_kleidiai_context(ಶೂನ್ಯ) {
    ggml_critical_section_start();
    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಅಸತ್ಯ;

    ಯದಿ (!initialized) {
        initialized = ಸತ್ಯ;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ಧ್ರುವ ವರ್ಣ *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ಧ್ರುವ ವರ್ಣ *env_threads     = getenv("GGML_TOTAL_THREADS");
        ಧ್ರುವ ವರ್ಣ *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ಧ್ರುವ auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        ಯದಿ (env_threads) {
            ತಾರ್ಕಿಕ ok = ಅಸತ್ಯ;
            ಪೂರ್ಣಾಙ್ಕ hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            ಯದಿ (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        ಯದಿ (env_chunk_mult) {
            ತಾರ್ಕಿಕ ok = ಅಸತ್ಯ;
            ಪೂರ್ಣಾಙ್ಕ multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            ಯದಿ (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        ಪೂರ್ಣಾಙ್ಕ sme_cores = 0;
        ತಾರ್ಕಿಕ sme_env_ok = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ sme_env_set = (env_sme != nullptr);

        ಧ್ರುವ ತಾರ್ಕಿಕ has_supported_sme_family = runtime_feat.has_sme;
        ತಾರ್ಕಿಕ sme_cap_detected = ಅಸತ್ಯ;

        ಯದಿ (has_supported_sme_family) {
            detected_smcus = detect_ಅಂಕೆ_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (ಪೂರ್ಣಾಙ್ಕ)detected_smcus : 1;

            ಯದಿ (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        ಯದಿ (sme_env_set) {
            ತಾರ್ಕಿಕ ok = ಅಸತ್ಯ;
            ಪೂರ್ಣಾಙ್ಕ v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            ಯದಿ (ok) {
                ಯದಿ (has_supported_sme_family) {
                    sme_cores = v;
                } ಅನ್ಯಥಾ {
                    ಯದಿ (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } ಅನ್ಯಥಾ {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        ಯದಿ (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            ಯದಿ (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        ಯದಿ (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (ಪೂರ್ಣಾಙ್ಕ)ctx.features);
        } ಅನ್ಯಥಾ {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        ಯದಿ (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (ಪೂರ್ಣಾಙ್ಕ)ctx.features);
        } ಅನ್ಯಥಾ {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        ಯದಿ (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (ಪೂರ್ಣಾಙ್ಕ) ctx.features);
        } ಅನ್ಯಥಾ {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        ಯದಿ (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (ಪೂರ್ಣಾಙ್ಕ)ctx.features);
        } ಅನ್ಯಥಾ {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        ಧ್ರುವ ತಾರ್ಕಿಕ has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        ಯದಿ (has_selected_sme_family_kernel) {
            ಯದಿ (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } ಅನ್ಯಥಾ ಯದಿ (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } ಅನ್ಯಥಾ {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } ಅನ್ಯಥಾ {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_end();
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ kleidiai_sme_thread_cap() {
    ಪ್ರತ್ಯಾಗಮ ctx.sme_thread_cap;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t align_up(size_t value, size_t alignment) {
    ಯದಿ (alignment == 0) {
        ಪ್ರತ್ಯಾಗಮ value;
    }
    ಧ್ರುವ size_t remainder = value % alignment;
    ಪ್ರತ್ಯಾಗಮ remainder == 0 ? value : value + (alignment - remainder);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t gcd_size(size_t a, size_t b) {
    ಯಾವತ್ (b != 0) {
        ಧ್ರುವ size_t t = a % b;
        a = b;
        b = t;
    }
    ಪ್ರತ್ಯಾಗಮ a;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ lcm_size(size_t a, size_t b, size_t & result) {
    ಯದಿ (a == 0 || b == 0) {
        result = 0;
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ size_t g = gcd_size(a, b);
    ಧ್ರುವ size_t q = a / g;
    ಯದಿ (q > SIZE_MAX / b) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    result = q * b;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t ceil_div_size(size_t a, size_t b) {
    ಪ್ರತ್ಯಾಗಮ b == 0 ? 0 : (a + b - 1) / b;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t kleidiai_chunk_cols(size_t n, ಪೂರ್ಣಾಙ್ಕ nth_total, ತಾರ್ಕಿಕ disable_chunking, size_t n_step) {
    ಧ್ರುವ size_t multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ಧ್ರುವ size_t ಭಾಜಕ = std::max<size_t>(1, (size_t) nth_total * multiplier);
    ಧ್ರುವ size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, ಭಾಜಕ)), n_step);
    ಪ್ರತ್ಯಾಗಮ chunk_cols ? chunk_cols : n_step;
}

ಸಂರಚನಾ kleidiai_block_args {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    ವಿಕಲ್ಪ (rhs_type) {
        ಪಕ್ಷ GGML_TYPE_Q4_0:
            ಪ್ರತ್ಯಾಗಮ { QK4_0, QK4_0, QK4_0 };
        ಪಕ್ಷ GGML_TYPE_Q4_K:
            ಪ್ರತ್ಯಾಗಮ { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಪ್ರತ್ಯಾಗಮ { 0, 0, QK8_0 };
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ { 0, 0, 0 };
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ kleidiai_pack_fallback_allowed() {
    ಯದಿ (ctx.sme_thread_cap <= 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (ctx.thread_hint <= 0) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ctx.thread_hint > ctx.sme_thread_cap;
}

ಸಂರಚನಾ kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_ಎಣಿಕೆ;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ kleidiai_weight_header * kleidiai_weight_header_from_ptr(ಶೂನ್ಯ * data) {
    ಪ್ರತ್ಯಾಗಮ reinterpret_cast<kleidiai_weight_header *>(data);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ kleidiai_weight_header * kleidiai_weight_header_from_ptr(ಧ್ರುವ ಶೂನ್ಯ * data) {
    ಪ್ರತ್ಯಾಗಮ reinterpret_cast<ಧ್ರುವ kleidiai_weight_header *>(data);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ kleidiai_is_weight_header_valid(ಧ್ರುವ kleidiai_weight_header * header) {
    ಯದಿ (!header) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (header->slot_ಎಣಿಕೆ == 0 || header->slot_ಎಣಿಕೆ > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, ಪೂರ್ಣಾಙ್ಕ slot) {
    ಯದಿ (!kleidiai_is_weight_header_valid(header)) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯದಿ (slot < 0 || slot >= header->slot_ಎಣಿಕೆ) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪ್ರತ್ಯಾಗಮ reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ uint8_t * kleidiai_weight_slot_ptr(ಧ್ರುವ kleidiai_weight_header * header, ಪೂರ್ಣಾಙ್ಕ slot) {
    ಯದಿ (!kleidiai_is_weight_header_valid(header)) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯದಿ (slot < 0 || slot >= header->slot_ಎಣಿಕೆ) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪ್ರತ್ಯಾಗಮ reinterpret_cast<ಧ್ರುವ uint8_t *>(header) + header->offsets[slot];
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    ಪ್ರತ್ಯಾಗಮ ctx.kernels_q4;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    ಪ್ರತ್ಯಾಗಮ ctx.kernels_q4k;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    ಪ್ರತ್ಯಾಗಮ ctx.kernels_q8;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    ಪ್ರತ್ಯಾಗಮ ctx.kernels_f32;
}

template <typename SelectFallback>
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    ಪೂರ್ಣಾಙ್ಕ count = 0;
    ಯದಿ (!primary) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    out[count++] = primary;

    ಯದಿ (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        ಪ್ರತ್ಯಾಗಮ count;
    }

    ಯದಿ (is_sme_family(primary->required_cpu)) {
        ಧ್ರುವ cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        ಯದಿ (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            ಯದಿ (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ count;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ kleidiai_collect_kernel_chain(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    ಪ್ರತ್ಯಾಗಮ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ಪ್ರತ್ಯಾಗಮ ggml_kleidiai_select_kernels(mask, op); });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    ಪ್ರತ್ಯಾಗಮ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ಪ್ರತ್ಯಾಗಮ ggml_kleidiai_select_kernels_q4_0(mask); });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    ಪ್ರತ್ಯಾಗಮ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ಪ್ರತ್ಯಾಗಮ ggml_kleidiai_select_kernels_q4_k(mask); });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    ಪ್ರತ್ಯಾಗಮ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ಪ್ರತ್ಯಾಗಮ ggml_kleidiai_select_kernels_q8_0(mask); });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    ಪ್ರತ್ಯಾಗಮ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ಪ್ರತ್ಯಾಗಮ ggml_kleidiai_select_kernels_f32(mask); });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int64_t ggml_ne(ಧ್ರುವ ggml_tensor * tensor, ಪೂರ್ಣಾಙ್ಕ dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    ಪ್ರತ್ಯಾಗಮ tensor->ne[dim];
}

ನಾಮಾಕಾಶ ggml::cpu::kleidiai {

ಸ್ಥಿರ size_t round_down(size_t x, size_t y) {
    ಪ್ರತ್ಯಾಗಮ y == 0 ? x : x - (x % y);
}

ಸ್ಥಿರ ಶೂನ್ಯ transpose_f32kxn_f16nxk(size_t n, size_t k, ಭಿನ್ನ * dst, ಧ್ರುವ uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / ಪರಿಮಾಣ(uint16_t);
    size_t dst_stride = n;

    ಪ್ರತಿ (size_t k_idx = 0; k_idx < k; ++k_idx) {
        ಪ್ರತಿ (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

ವರ್ಗ tensor_traits : ಸಾರ್ವಜನಿಕ ggml::cpu::tensor_traits {
    ತಾರ್ಕಿಕ work_size(ಪೂರ್ಣಾಙ್ಕ /* n_threads */, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op, size_t & size) override {
        ಯದಿ (op->op != GGML_OP_MUL_MAT) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ slot_ಎಣಿಕೆ = kleidiai_collect_kernel_chain(op, kernel_chain);
        ಯದಿ (slot_ಎಣಿಕೆ == 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ತಾರ್ಕಿಕ is_gemv = op->src[1]->ne[1] == 1;
        ಧ್ರುವ size_t k = op->src[0]->ne[0];
        ಧ್ರುವ size_t n = op->src[0]->ne[1];
        ಧ್ರುವ size_t m = op->src[1]->ne[1];

        ಯದಿ (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ಧ್ರುವ kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            ಧ್ರುವ size_t qk = block_args.pack_bl;

            size_t cursor = 0;
            ತಾರ್ಕಿಕ any_slot = ಅಸತ್ಯ;

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ slot = 0; slot < slot_ಎಣಿಕೆ; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                ಯದಿ (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                ಧ್ರುವ size_t mr = kernel->get_mr();
                ಧ್ರುವ size_t kr = kernel->get_kr();
                ಧ್ರುವ size_t sr = kernel->get_sr();

                ಧ್ರುವ size_t packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = ಸತ್ಯ;
            }

            ಯದಿ (!any_slot) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            size = cursor;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            ಯದಿ (is_gemv && op->src[1]->nb[0] == (int64_t) ಪರಿಮಾಣ(ಭಿನ್ನ) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }

            size_t cursor = 0;
            ತಾರ್ಕಿಕ any_slot = ಅಸತ್ಯ;

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ slot = 0; slot < slot_ಎಣಿಕೆ; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                ಯದಿ (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                ಧ್ರುವ size_t mr = kernel->get_mr();
                ಧ್ರುವ size_t kr = kernel->get_kr();
                ಧ್ರುವ size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
                any_slot = ಸತ್ಯ;
            }

            ಯದಿ (!any_slot) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            size = cursor;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (op->src[0]->type == GGML_TYPE_F16) {
            ಧ್ರುವ int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ಧ್ರುವ int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ಧ್ರುವ int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            ತಾರ್ಕಿಕ any_slot = ಅಸತ್ಯ;

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ slot = 0; slot < slot_ಎಣಿಕೆ; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                ಯದಿ (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                ಧ್ರುವ size_t mr = kernel->get_mr();
                ಧ್ರುವ size_t kr = kernel->get_kr();
                ಧ್ರುವ size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = ಸತ್ಯ;
            }

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ slot = 0; slot < slot_ಎಣಿಕೆ; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                ಯದಿ (!kernel || !kernels->rhs_info.packed_size_ex) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * ಪರಿಮಾಣ(ಭಿನ್ನ);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * ಪರಿಮಾಣ(ಭಿನ್ನ);

            ಯದಿ (!any_slot) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            size = cursor;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ compute_forward(ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst) override {
        ಯದಿ (dst->op == GGML_OP_MUL_MAT) {
            ಯದಿ (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                ಪ್ರತ್ಯಾಗಮ compute_forward_qx(params, dst);
            } ಅನ್ಯಥಾ ಯದಿ (dst->src[0]->type == GGML_TYPE_F32) {
                ಪ್ರತ್ಯಾಗಮ compute_forward_f32(params, dst);
            } ಅನ್ಯಥಾ ಯದಿ (dst->src[0]->type == GGML_TYPE_F16) {
                ಪ್ರತ್ಯಾಗಮ compute_forward_fp16(params, dst);
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ compute_forward_f32(ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
        ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

        ಯದಿ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        ಯದಿ (!kernels) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ size_t k = ne00;
        ಧ್ರುವ size_t m = ne11;
        ಧ್ರುವ size_t n = ne01;
        ಧ್ರುವ ತಾರ್ಕಿಕ use_gemv = m == 1 && src1->nb[0] == (int64_t) ಪರಿಮಾಣ(ಭಿನ್ನ) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        ಯದಿ (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_size_ex || !lhs_info->pack_func_ex)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ಧ್ರುವ ತಾರ್ಕಿಕ has_header = kleidiai_is_weight_header_valid(header);

        ಧ್ರುವ uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<ಧ್ರುವ uint8_t *>(src0->data);
        ಯದಿ (!rhs_base) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = params->nth > 0 ? params->nth : 1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ith = params->ith;

        ಧ್ರುವ size_t mr = kernel->get_mr();
        ಧ್ರುವ size_t kr = kernel->get_kr();
        ಧ್ರುವ size_t sr = kernel->get_sr();

        ಧ್ರುವ size_t lhs_packed_size = use_gemv ? 0 : lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        ಯದಿ (!use_gemv) {
            GGML_ASSERT(lhs_packed_size <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ಧ್ರುವ size_t dst_stride = dst->nb[1];
        ಧ್ರುವ size_t n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        ಧ್ರುವ ತಾರ್ಕಿಕ disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) INT_MAX);

        ಧ್ರುವ int64_t batch_size = ne12 * ne13;
        ಪ್ರತಿ (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ಧ್ರುವ int64_t i12 = batch_idx % ne12;
            ಧ್ರುವ int64_t i13 = batch_idx / ne12;
            ಧ್ರುವ uint8_t * lhs_batch_base = static_cast<ಧ್ರುವ uint8_t *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];

            ಯದಿ (!use_gemv) {
                ಧ್ರುವ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                ಧ್ರುವ int64_t use_threads = std::min<int64_t>(nth, max_threads);

                ಯದಿ (ith < use_threads) {
                    ಧ್ರುವ int64_t ಅಂಕೆ_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), mr);
                    ಧ್ರುವ int64_t ಅಂಕೆ_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * ಅಂಕೆ_m_per_thread0;

                    ಧ್ರುವ int64_t m_start = (int64_t)ith * ಅಂಕೆ_m_per_thread0;
                    ಧ್ರುವ int64_t m_ಎಣಿಕೆ = (ith == use_threads - 1) ? ಅಂಕೆ_m_per_thread_N_1 : ಅಂಕೆ_m_per_thread0;

                    ಧ್ರುವ size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ಧ್ರುವ size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ಧ್ರುವ size_t row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_ಎಣಿಕೆ;
                    int64_t cur       = m_start;

                    ಯಾವತ್ (remaining > 0) {
                        ಧ್ರುವ int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        ಧ್ರುವ size_t src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        ಧ್ರುವ ಶೂನ್ಯ * src_ptr = lhs_batch_base + src_off;
                        ಧ್ರುವ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        ಶೂನ್ಯ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            ಯದಿ (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ಧ್ರುವ size_t chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (size_t) INT_MAX);

            ಪೂರ್ಣಾಙ್ಕ current_col = ggml_threadpool_chunk_add(params->threadpool, (ಪೂರ್ಣಾಙ್ಕ) chunk_cols);
            ಯಾವತ್ ((size_t) current_col < n) {
                ಧ್ರುವ size_t n_start = (size_t) current_col;
                ಧ್ರುವ size_t n_to_process = std::min(chunk_cols, n - n_start);

                ಯದಿ (n_to_process > 0) {
                    ಧ್ರುವ size_t lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ಧ್ರುವ size_t rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ಧ್ರುವ size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    ಧ್ರುವ ಶೂನ್ಯ * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    ಧ್ರುವ ಶೂನ್ಯ * rhs_ptr = rhs_base + rhs_packed_offset;
                    ಭಿನ್ನ * dst_ptr      = reinterpret_cast<ಭಿನ್ನ *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          ಪರಿಮಾಣ(ಭಿನ್ನ),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (ಪೂರ್ಣಾಙ್ಕ) chunk_cols);
            }

            ಯದಿ (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ compute_forward_fp16(ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst) {
        ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
        ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        ಯದಿ (!kernels) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ತಾರ್ಕಿಕ is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        ಯದಿ (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth = params->nth;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ith = params->ith;

        ಧ್ರುವ int64_t lhs_batch_size0 = ne12;
        ಧ್ರುವ int64_t rhs_batch_size0 = ne02;
        ಧ್ರುವ int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        ಧ್ರುವ int64_t r = lhs_batch_size0 / rhs_batch_size0;

        ಧ್ರುವ int64_t m_group = ne11;
        ಧ್ರುವ int64_t m       = m_group;
        ಧ್ರುವ int64_t n       = ne01;
        ಧ್ರುವ int64_t k       = ne00;

        ಧ್ರುವ size_t lhs_stride = src1->nb[1];
        ಧ್ರುವ size_t rhs_stride = src0->nb[1];
        ಧ್ರುವ size_t dst_stride = dst->nb[1];

        ಧ್ರುವ int64_t mr = (int64_t) kernel->get_mr();
        ಧ್ರುವ int64_t nr = (int64_t) kernel->get_nr();
        ಧ್ರುವ int64_t kr = (int64_t) kernel->get_kr();
        ಧ್ರುವ int64_t sr = (int64_t) kernel->get_sr();

        ಧ್ರುವ size_t lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        ಧ್ರುವ size_t rhs_packed_size = kernels->rhs_info.packed_size_ex(n, k, nr, kr, 0);
        ಧ್ರುವ size_t kxn_size        = k * n * ಪರಿಮಾಣ(ಭಿನ್ನ);
        ಧ್ರುವ size_t bias_size       = n * ಪರಿಮಾಣ(ಭಿನ್ನ);

        ಧ್ರುವ size_t wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        ಪ್ರತಿ (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ಧ್ರುವ int64_t rhs_batch_idx = batch_idx / r;
            ಧ್ರುವ uint8_t * rhs_batch_base = static_cast<ಧ್ರುವ uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ಧ್ರುವ int64_t m_roundup_mr = kai_roundup(m, mr);
                ಧ್ರುವ int64_t ಅಂಕೆ_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                ಯದಿ (ith < ಅಂಕೆ_threads) {
                    ಧ್ರುವ int64_t ಅಂಕೆ_m_per_thread0   = round_down((size_t)(m_roundup_mr / ಅಂಕೆ_threads), (size_t)mr);
                    ಧ್ರುವ int64_t ಅಂಕೆ_m_per_thread_N_1 = m - (ಅಂಕೆ_threads - 1) * ಅಂಕೆ_m_per_thread0;

                    ಧ್ರುವ int64_t m_start = ith * ಅಂಕೆ_m_per_thread0;
                    ಧ್ರುವ int64_t m_ಎಣಿಕೆ = (ith == ಅಂಕೆ_threads - 1) ? ಅಂಕೆ_m_per_thread_N_1 : ಅಂಕೆ_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ಧ್ರುವ size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ಧ್ರುವ size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ಧ್ರುವ size_t row_stride_bytes = (next_block_off - base_packed_off) / (size_t)mr;

                    int64_t remaining = m_ಎಣಿಕೆ;
                    int64_t cur       = m_start;

                    ಯಾವತ್ (remaining > 0) {
                        ಧ್ರುವ int64_t row_in_group = cur;
                        ಧ್ರುವ int64_t avail        = m_group - row_in_group;
                        ಧ್ರುವ int64_t take         = std::min(avail, remaining);

                        ಧ್ರುವ uint8_t * lhs_batch_base = static_cast<ಧ್ರುವ uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        ಧ್ರುವ ಶೂನ್ಯ * src_ptr = lhs_batch_base + (size_t)row_in_group * lhs_stride;
                        ಧ್ರುವ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        ಶೂನ್ಯ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            ಯದಿ (ith == 0) {
                memset(bias, 0, (size_t)n * ಪರಿಮಾಣ(ಭಿನ್ನ));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<ಭಿನ್ನ *>(rhs_kxn),
                                        reinterpret_cast<ಧ್ರುವ uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * ಪರಿಮಾಣ(ಭಿನ್ನ),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ಧ್ರುವ int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t ಅಂಕೆ_threads_n = KAI_MIN(n / n_step, nth);
                ಯದಿ (ಅಂಕೆ_threads_n <= 0) {
                    ಅಂಕೆ_threads_n = 1;
                }

                ಯದಿ (ith < ಅಂಕೆ_threads_n) {
                    ಧ್ರುವ int64_t ಅಂಕೆ_n_per_thread0   = round_down((size_t)(n / ಅಂಕೆ_threads_n), (size_t)n_step);
                    ಧ್ರುವ int64_t ಅಂಕೆ_n_per_thread_N_1 = n - (ಅಂಕೆ_threads_n - 1) * ಅಂಕೆ_n_per_thread0;

                    ಧ್ರುವ int64_t n_start      = ith * ಅಂಕೆ_n_per_thread0;
                    ಧ್ರುವ int64_t n_to_process = (ith == ಅಂಕೆ_threads_n - 1) ? ಅಂಕೆ_n_per_thread_N_1 : ಅಂಕೆ_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ಧ್ರುವ size_t lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ಧ್ರುವ size_t rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ಧ್ರುವ size_t dst_offset         = kernel->get_dst_offset((size_t)0, (size_t)n_start, dst_stride);

                    ಧ್ರುವ ಶೂನ್ಯ * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ಧ್ರುವ ಶೂನ್ಯ * rhs_ptr = rhs_packed + rhs_packed_offset;
                    ಭಿನ್ನ * dst_ptr      = reinterpret_cast<ಭಿನ್ನ *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, ಪರಿಮಾಣ(ಭಿನ್ನ), -FLT_MAX, FLT_MAX);
                }
            }

            ಯದಿ (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ compute_forward_qx(ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
        ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

        ಧ್ರುವ kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ಧ್ರುವ ತಾರ್ಕಿಕ has_header = kleidiai_is_weight_header_valid(header);
        ಧ್ರುವ ತಾರ್ಕಿಕ is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        auto weight_for_slot = [&](ಪೂರ್ಣಾಙ್ಕ slot_index, size_t & size_out) -> ಧ್ರುವ uint8_t * {
            ಯದಿ (slot_index < 0 || slot_index >= slot_total) {
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
            ಯದಿ (has_header) {
                ಯದಿ (slot_index < header->slot_ಎಣಿಕೆ) {
                    size_out = static_cast<size_t>(header->sizes[slot_index]);
                    ಪ್ರತ್ಯಾಗಮ kleidiai_weight_slot_ptr(header, slot_index);
                }
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
            ಯದಿ (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                ಪ್ರತ್ಯಾಗಮ static_cast<ಧ್ರುವ uint8_t *>(src0->data);
            }
            ಪ್ರತ್ಯಾಗಮ nullptr;
        };

        ಸಂರಚನಾ runtime_slot {
            ಪೂರ್ಣಾಙ್ಕ slot_index;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            size_t mr;
            size_t nr;
            size_t kr;
            size_t sr;
            size_t n_step;
            size_t lhs_packed_size;
            size_t lhs_offset;
            size_t lhs_bl;
            size_t rhs_bl;
            size_t pack_bl;
            size_t lhs_packed_offset0;
            ಪೂರ್ಣಾಙ್ಕ assigned_threads;
            ಪೂರ್ಣಾಙ್ಕ thread_begin;
            ಪೂರ್ಣಾಙ್ಕ thread_end;
            ಧ್ರುವ uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        ಪೂರ್ಣಾಙ್ಕ runtime_ಎಣಿಕೆ = 0;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ slot = 0; slot < slot_total && runtime_ಎಣಿಕೆ < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            ಯದಿ (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                ಅನುವರ್ತ;
            }

            size_t rhs_size = 0;
            ಧ್ರುವ uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            ಯದಿ (!rhs_ptr || rhs_size == 0) {
                ಅನುವರ್ತ;
            }

            ಧ್ರುವ kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

            runtime[runtime_ಎಣಿಕೆ] = {
                slot,
                kernels,
                kinfo,
                linfo,
                kinfo->get_mr(),
                kinfo->get_nr(),
                kinfo->get_kr(),
                kinfo->get_sr(),
                kinfo->get_n_step(),
                0,
                0,
                block_args.lhs_bl,
                block_args.rhs_bl,
                block_args.pack_bl,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_ಎಣಿಕೆ;
        }

        ಯದಿ (runtime_ಎಣಿಕೆ == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nth_total = params->nth > 0 ? params->nth : 1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ith_total = params->ith;

        ಪೂರ್ಣಾಙ್ಕ sme_slot = -1;
        ಪೂರ್ಣಾಙ್ಕ non_sme_slot = -1;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            ಯದಿ (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                ಭಙ್ಗ;
            }
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            ಯದಿ (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                ಭಙ್ಗ;
            }
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sme_cap_limit = ctx.sme_thread_cap;
        ಧ್ರುವ ತಾರ್ಕಿಕ use_hybrid = sme_cap_limit > 0 &&
                                 runtime_ಎಣಿಕೆ > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_cols_per_thread = 0;
        ಯದಿ (runtime_ಎಣಿಕೆ > 0 && nth_total > 0) {
            min_cols_per_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        ಧ್ರುವ ತಾರ್ಕಿಕ too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        ಧ್ರುವ ತಾರ್ಕಿಕ hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        ಯದಿ (!hybrid_enabled) {
            ಪೂರ್ಣಾಙ್ಕ chosen_slot = 0;
            ಯದಿ (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } ಅನ್ಯಥಾ ಯದಿ (runtime_ಎಣಿಕೆ > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            ಯದಿ (chosen_slot != 0 && chosen_slot < runtime_ಎಣಿಕೆ) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_ಎಣಿಕೆ = runtime_ಎಣಿಕೆ > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            ಯದಿ (runtime_ಎಣಿಕೆ > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        ಧ್ರುವ ತಾರ್ಕಿಕ sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        ಪೂರ್ಣಾಙ್ಕ sme_cap = sme_single_only ? nth_total : kleidiai_sme_thread_cap();
        ಯದಿ (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        ಪೂರ್ಣಾಙ್ಕ threads_remaining = nth_total;
        ಯದಿ (sme_slot != -1) {
            ಪೂರ್ಣಾಙ್ಕ sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        ಪೂರ್ಣಾಙ್ಕ fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        ಪೂರ್ಣಾಙ್ಕ fallback_ಎಣಿಕೆ = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            ಯದಿ (i == sme_slot) {
                ಅನುವರ್ತ;
            }
            fallback_indices[fallback_ಎಣಿಕೆ++] = i;
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ fi = 0; fi < fallback_ಎಣಿಕೆ; ++fi) {
            ಯದಿ (threads_remaining <= 0) {
                ಭಙ್ಗ;
            }
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ slot_index = fallback_indices[fi];
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ slots_ಎಡ = fallback_ಎಣಿಕೆ - fi;
            ಪೂರ್ಣಾಙ್ಕ share = (threads_remaining + slots_ಎಡ - 1) / slots_ಎಡ;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        ಪೂರ್ಣಾಙ್ಕ thread_cursor = 0;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ active_threads = thread_cursor;

        ಪೂರ್ಣಾಙ್ಕ local_slot = -1;
        ಪೂರ್ಣಾಙ್ಕ local_ith  = 0;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            ಯದಿ (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                ಭಙ್ಗ;
            }
        }
        ಧ್ರುವ size_t k = ne00;
        ಧ್ರುವ size_t m = ne11;
        ಧ್ರುವ size_t n = ne01;

        size_t cursor = 0;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_step = 1;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            ಯದಿ (runtime[i].assigned_threads == 0) {
                ಅನುವರ್ತ;
            }
            size_t next_step = 0;
            ಯದಿ (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        ಧ್ರುವ ತಾರ್ಕಿಕ disable_chunking = ggml_is_numa();
        ಧ್ರುವ size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ಧ್ರುವ size_t chunk_ಭಾಜಕ = (active_threads == 1 || disable_chunking) ? (size_t) active_threads : (size_t) active_threads * chunk_multiplier;
        size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, chunk_ಭಾಜಕ)), common_step);
        ಯದಿ (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ಧ್ರುವ size_t nchunk_size = std::max<size_t>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (size_t)INT_MAX);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nchunk = (ಪೂರ್ಣಾಙ್ಕ)nchunk_size;
        ಧ್ರುವ size_t dst_stride = dst->nb[1];

        auto run_chunk = [&](runtime_slot & slot, size_t global_start, size_t cols, uint8_t * dst_batch_base) {
            ಧ್ರುವ size_t rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            ಧ್ರುವ size_t dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            ಧ್ರುವ uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            ಧ್ರುವ uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            ಭಿನ್ನ * dst_ptr         = reinterpret_cast<ಭಿನ್ನ *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       ಪರಿಮಾಣ(ಭಿನ್ನ),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        ಪ್ರತಿ (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ಧ್ರುವ uint8_t * lhs_batch_base = static_cast<ಧ್ರುವ uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            ಯದಿ (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                ಧ್ರುವ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                ಧ್ರುವ int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                ಯದಿ (local_ith < use_threads) {
                    ಧ್ರುವ int64_t ಅಂಕೆ_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), slot.mr);
                    ಧ್ರುವ int64_t ಅಂಕೆ_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * ಅಂಕೆ_m_per_thread0;

                    ಧ್ರುವ int64_t m_start = (int64_t)local_ith * ಅಂಕೆ_m_per_thread0;
                    ಧ್ರುವ int64_t m_ಎಣಿಕೆ = (local_ith == use_threads - 1) ? ಅಂಕೆ_m_per_thread_N_1 : ಅಂಕೆ_m_per_thread0;

                    ಧ್ರುವ size_t base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ಧ್ರುವ size_t next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ಧ್ರುವ size_t row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_ಎಣಿಕೆ;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    ಯಾವತ್ (remaining > 0) {
                        ಧ್ರುವ int64_t row_in_group = cur;
                        ಧ್ರುವ int64_t avail        = (int64_t)m - row_in_group;
                        ಧ್ರುವ int64_t take         = std::min(avail, remaining);

                        ಧ್ರುವ size_t src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        ಧ್ರುವ ಶೂನ್ಯ * src_ptr = lhs_batch_base + src_off;
                        ಧ್ರುವ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        ಶೂನ್ಯ * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            ಯದಿ (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            ಯದಿ (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                ಪೂರ್ಣಾಙ್ಕ current_chunk = ith_total;
                ಯಾವತ್ (current_chunk < nchunk) {
                    ಧ್ರುವ size_t global_start = (size_t)current_chunk * chunk_cols;
                    ಯದಿ (global_start >= n) {
                        ಭಙ್ಗ;
                    }

                    ಧ್ರುವ size_t cols = std::min(chunk_cols, n - global_start);
                    ಯದಿ (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            ಯದಿ (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

ಸಾರ್ವಜನಿಕ:
    ಪೂರ್ಣಾಙ್ಕ repack(ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ಧ್ರುವ size_t n = tensor->ne[1];
        ಧ್ರುವ size_t k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        ಯದಿ (!header) {
            ಪ್ರತ್ಯಾಗಮ -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_ಎಣಿಕೆ = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        size_t cursor = ಪರಿಮಾಣ(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ಧ್ರುವ ತಾರ್ಕಿಕ want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ಧ್ರುವ ತಾರ್ಕಿಕ want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ಧ್ರುವ ತಾರ್ಕಿಕ want_f32 = tensor->type == GGML_TYPE_F32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ಧ್ರುವ ತಾರ್ಕಿಕ allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<ಭಿನ್ನ>  scales;
        std::vector<ಭಿನ್ನ>  bias;
        std::vector<ಭಿನ್ನ> row_f32;

        ಯದಿ (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ಧ್ರುವ size_t row_stride = tensor->nb[1];
            ಧ್ರುವ size_t k_blocks   = (k + QK8_0 - 1) / QK8_0;

            ಯದಿ (tensor->type == GGML_TYPE_Q6_K) {
                ಯದಿ (k % QK_K != 0 || k % QK8_0 != 0) {
                    ಪ್ರತ್ಯಾಗಮ -1;
                }
                row_f32.resize(k);
            }

            ಪ್ರತಿ (size_t row = 0; row < n; ++row) {
                ಧ್ರುವ uint8_t * row_data = static_cast<ಧ್ರುವ uint8_t *>(data) + row * row_stride;

                ಯದಿ (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<ಧ್ರುವ block_q6_K *>(row_data), row_f32.data(), k);
                }

                ಭಿನ್ನ max_abs = 0.0f;
                ಯದಿ (tensor->type == GGML_TYPE_Q6_K) {
                    ಪ್ರತಿ (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::fabs(row_f32[linear_idx]));
                    }
                } ಅನ್ಯಥಾ {
                    ಧ್ರುವ block_q8_0 * row_blocks = reinterpret_cast<ಧ್ರುವ block_q8_0 *>(row_data);
                    ಪ್ರತಿ (size_t block = 0; block < k_blocks; ++block) {
                        ಧ್ರುವ block_q8_0 & blk = row_blocks[block];
                        ಧ್ರುವ ಭಿನ್ನ d = GGML_FP16_TO_FP32(blk.d);
                        ಪ್ರತಿ (size_t l = 0; l < QK8_0; ++l) {
                            ಧ್ರುವ size_t linear_idx = block * QK8_0 + l;
                            ಯದಿ (linear_idx >= k) {
                                ಭಙ್ಗ;
                            }
                            ಧ್ರುವ ಭಿನ್ನ value = d * static_cast<ಭಿನ್ನ>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                ಭಿನ್ನ scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                ಧ್ರುವ ಭಿನ್ನ inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                ಯದಿ (tensor->type == GGML_TYPE_Q6_K) {
                    ಪ್ರತಿ (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } ಅನ್ಯಥಾ {
                    ಧ್ರುವ block_q8_0 * row_blocks = reinterpret_cast<ಧ್ರುವ block_q8_0 *>(row_data);
                    ಪ್ರತಿ (size_t block = 0; block < k_blocks; ++block) {
                        ಧ್ರುವ block_q8_0 & blk = row_blocks[block];
                        ಧ್ರುವ ಭಿನ್ನ d = GGML_FP16_TO_FP32(blk.d);
                        ಪ್ರತಿ (size_t l = 0; l < QK8_0; ++l) {
                            ಧ್ರುವ size_t linear_idx = block * QK8_0 + l;
                            ಯದಿ (linear_idx >= k) {
                                ಭಙ್ಗ;
                            }
                            ಧ್ರುವ ಭಿನ್ನ value = d * static_cast<ಭಿನ್ನ>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        ಯದಿ (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ಯದಿ (!allow_fallback && slot > 0) {
                ಭಙ್ಗ;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            ಯದಿ (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                ಅನುವರ್ತ;
            }

            ಧ್ರುವ size_t nr = kernel->get_nr();
            ಧ್ರುವ size_t kr = kernel->get_kr();
            ಧ್ರುವ size_t sr = kernel->get_sr();
            ಧ್ರುವ ggml_type rhs_type = kernels->rhs_type;
            ಯದಿ (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            ಧ್ರುವ size_t block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            ಯದಿ (block_len == SIZE_MAX) {
                ಅನುವರ್ತ;
            }

            ಧ್ರುವ size_t packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            ಧ್ರುವ size_t aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            ಯದಿ (rhs_type == GGML_TYPE_Q4_0) {
                ಸಂರಚನಾ kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ಧ್ರುವ uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } ಅನ್ಯಥಾ ಯದಿ (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       data, nullptr, nullptr, dst_ptr, 0, nullptr);
            } ಅನ್ಯಥಾ ಯದಿ (rhs_type == GGML_TYPE_Q8_0) {
                ಸಂರಚನಾ kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } ಅನ್ಯಥಾ ಯದಿ (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
            } ಅನ್ಯಥಾ {
                ಅನುವರ್ತ;
            }

            header->offsets[header->slot_ಎಣಿಕೆ] = aligned_cursor;
            header->sizes[header->slot_ಎಣಿಕೆ]   = packed_size;
            ++header->slot_ಎಣಿಕೆ;

            cursor = aligned_cursor + packed_size;
        }

        ಯದಿ (header->slot_ಎಣಿಕೆ == 0) {
            header->magic   = 0;
            header->version = 0;
            memcpy(tensor->data, data, data_size);
        }

        ಪ್ರತ್ಯಾಗಮ 0;
    }
};

ಸ್ಥಿರ ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, ಸಂರಚನಾ ggml_tensor *) {
    ಸ್ಥಿರ tensor_traits traits;
    ಪ್ರತ್ಯಾಗಮ &traits;
}
}  // namespace ggml::cpu::kleidiai

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, ಸಂರಚನಾ ggml_tensor * tensor) {
    tensor->extra = (ಶೂನ್ಯ *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, ಸಂರಚನಾ ggml_tensor * tensor,
                                                       ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    ಪ್ರತ್ಯಾಗಮ "CPU_KLEIDIAI";
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    ಯದಿ (buffer == nullptr) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    ಪ್ರತ್ಯಾಗಮ buffer;
}

ಸ್ಥಿರ size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    ಪ್ರತ್ಯಾಗಮ TENSOR_ALIGNMENT;
}

ಸ್ಥಿರ size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    ಯದಿ (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ggml_nbytes(tensor);
    }

    ಧ್ರುವ size_t n = tensor->ne[1];
    ಧ್ರುವ size_t k = tensor->ne[0];

    size_t cursor = ಪರಿಮಾಣ(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ಧ್ರುವ ತಾರ್ಕಿಕ want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ಧ್ರುವ ತಾರ್ಕಿಕ want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ಧ್ರುವ ತಾರ್ಕಿಕ want_f32 = tensor->type == GGML_TYPE_F32;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ಧ್ರುವ ತಾರ್ಕಿಕ allow_fallback = kleidiai_pack_fallback_allowed();

    size_t slot_ಎಣಿಕೆ = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ slot = 0; slot < slot_total; ++slot) {
        ಯದಿ (!allow_fallback && slot > 0) {
            ಭಙ್ಗ;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        ಯದಿ (!kernels) {
            ಅನುವರ್ತ;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        ಯದಿ (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ ggml_type rhs_type = kernels->rhs_type;
        ಧ್ರುವ size_t block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        ಯದಿ (block_len == SIZE_MAX) {
            ಅನುವರ್ತ;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_ಎಣಿಕೆ;
    }

    ಯದಿ (slot_ಎಣಿಕೆ == 0) {
        ಪ್ರತ್ಯಾಗಮ ggml_nbytes(tensor);
    }

    ಪ್ರತ್ಯಾಗಮ std::max(cursor, ggml_nbytes(tensor));
}

ನಾಮಾಕಾಶ ggml::cpu::kleidiai {
ವರ್ಗ extra_buffer_type : ggml::cpu::extra_buffer_type {
    ತಾರ್ಕಿಕ supports_op(ggml_backend_dev_t, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        ಯದಿ (op->op != GGML_OP_MUL_MAT) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        ಧ್ರುವ ತಾರ್ಕಿಕ src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft->context == ಏತತ್ &&
            slot_total > 0;

        ಯದಿ ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            ಯದಿ (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಯದಿ (op->src[0]->type == GGML_TYPE_Q4_K) {
                ಯದಿ (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಪ್ರತ್ಯಾಗಮ ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == ಪರಿಮಾಣ(ಭಿನ್ನ) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * ಪರಿಮಾಣ(ಭಿನ್ನ));
            }

            ಯದಿ (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                ಯದಿ (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಯದಿ (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) override {
        ಯದಿ (op->op == GGML_OP_MUL_MAT) {
            ಯದಿ (op->src[0]->buffer && op->src[0]->buffer->buft->context == ಏತತ್) {
                ಪ್ರತ್ಯಾಗಮ (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } ಅನ್ಯಥಾ {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                ಯದಿ (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ> warned(ಅಸತ್ಯ);
                    ಯದಿ (!warned.exchange(ಸತ್ಯ)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                ಯದಿ (op->src[0]->type != GGML_TYPE_F16) {
                    ಪ್ರತ್ಯಾಗಮ nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                ಯದಿ (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    ಯದಿ ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        ಪ್ರತ್ಯಾಗಮ nullptr;
                    }
                    ಪ್ರತ್ಯಾಗಮ ggml::cpu::kleidiai::get_tensor_traits(ರಿಕ್ತ, ರಿಕ್ತ);
                }
            }
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml::cpu::kleidiai::extra_buffer_type ctx;
    ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .get_alloc_size_n     = */ ರಿಕ್ತ,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_cpu_buffer_type_kleidiai;
}
