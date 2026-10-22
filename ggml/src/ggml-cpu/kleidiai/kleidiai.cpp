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
#ಆದರೆ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <asm/hwcap.h>
#ಅನ್ತರ್ಭಾವ <dirent.h>
#ಅನ್ತರ್ಭಾವ <sys/auxv.h>
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅಥವಾಆದರೆ defined(__APPLE__)
#ಅನ್ತರ್ಭಾವ <sys/sysctl.h>
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಆದರೆ_ಅಂತ್ಯ

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

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_CPP
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
ಸ್ಥಿರ constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
ಸ್ಥಿರ constexpr uint16_t GGML_KLEIDIAI_PACK_ಆವೃತ್ತಿ     = 1;
ಸ್ಥಿರ constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
ಸ್ಥಿರ constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

ರಚನೆ ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    ಪೂರ್ಣಾಂಕ sme_thread_cap; // <= 0 means "SME disabled/unknown"
    ಪೂರ್ಣಾಂಕ thread_hint;    // <= 0 means "no hint"
    ಪೂರ್ಣಾಂಕ chunk_multiplier;
} ಸ್ಥಿರ ctx = { CPU_FEATURE_NONE, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, 0, -1, 4 };

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ is_sme_family(cpu_feature f) {
    ಹಿಂತಿರುಗಿ (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* cpu_feature_ಗಮ್ಯ_string(cpu_feature f) {
    ಆದರೆ (f == CPU_FEATURE_NONE) {
        ಹಿಂತಿರುಗಿ "NONE";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        ಹಿಂತಿರುಗಿ "SME2";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        ಹಿಂತಿರುಗಿ "SME";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        ಹಿಂತಿರುಗಿ "SVE";
    }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        ಹಿಂತಿರುಗಿ "I8MM";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        ಹಿಂತಿರುಗಿ "DOTPROD";
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ "UNKNOWN";
    }
}

#ಆದರೆ defined(__linux__) && defined(__aarch64__)
ಸ್ಥಿರ ತಾರ್ಕಿಕ parse_cpu_dir_name(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* name, size_t* cpu) {
    ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* ಮೊದಲ = name + 3;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* ಕೊನೆಯ = name + ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(name);

    size_t value = 0;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ [end, ec] = std::from_chars(ಮೊದಲ, ಕೊನೆಯ, value, 10);

    ಆದರೆ (ec != std::errc{} || end != ಕೊನೆಯ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    *cpu = value;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    ಆದರೆ (dir == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ cpus;
    }

    ಆಗ (dirent * ನಮೂದು = readdir(dir)) {
        size_t cpu = 0;
        ಆದರೆ (parse_cpu_dir_name(ನಮೂದು->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    ಹಿಂತಿರುಗಿ cpus;
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__APPLE__) && defined(__aarch64__)
ಸ್ಥಿರ ತಾರ್ಕಿಕ apple_sme_counted_perf_level(std::string name) {
    ಗಾಗಿ (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (ಅಕ್ಷರ) std::tolower((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    ಹಿಂತಿರುಗಿ name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಶೂನ್ಯ ಕೂಡಿಸು_smcus_from_smidr(uint64_t smidr, size_t & ಅಂಕೆ_ಖಾಸಗಿ, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ಸ್ಥಿರಾಂಕ uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ಸ್ಥಿರಾಂಕ uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ಸ್ಥಿರಾಂಕ size_t shared_ಎಣಿಕೆ = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ಸ್ಥಿರಾಂಕ uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ಸ್ಥಿರಾಂಕ uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ಸ್ಥಿರಾಂಕ uint32_t id = (affinity2 << 12) | affinity;

    ಆದರೆ (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    ಆಯ್ಕೆ (sh) {
        ಸ್ಥಿತಿ 2: // private SMCU
            ++ಅಂಕೆ_ಖಾಸಗಿ;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: // shared SMCU
            ಆದರೆ (shared_counts[id] < shared_ಎಣಿಕೆ) {
                shared_counts[id] = shared_ಎಣಿಕೆ;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0:
            ಆದರೆ (id == 0) {
                ++ಅಂಕೆ_ಖಾಸಗಿ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (shared_counts[id] < shared_ಎಣಿಕೆ) {
                shared_counts[id] = shared_ಎಣಿಕೆ;
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }
}

ಸ್ಥಿರ size_t detect_ಅಂಕೆ_smcus() {
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ runtime_feat = ggml_feats_get_arch64_runtime();
    ಆದರೆ (!runtime_feat.has_sme) {
        ಹಿಂತಿರುಗಿ 0;
    }

#ಆದರೆ defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t ಅಂಕೆ_ಖಾಸಗಿ = 0;
    std::map<uint32_t, size_t> shared_counts;

    ಸ್ಥಿರಾಂಕ std::vector<size_t> cpus = detect_cpu_ids();
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ size_t cpu : cpus) {
        ಸ್ಥಿರಾಂಕ std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        ಆದರೆ (!file.is_open()) {
            ಮುಂದುವರಿ;
        }

        uint64_t smidr = 0;
        ಆದರೆ (!(file >> std::hex >> smidr)) {
            ಮುಂದುವರಿ;
        }

        ಕೂಡಿಸು_smcus_from_smidr(smidr, ಅಂಕೆ_ಖಾಸಗಿ, shared_counts);
    }

    size_t ಒಟ್ಟು = ಅಂಕೆ_ಖಾಸಗಿ;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ನಮೂದು : shared_counts) {
        ಒಟ್ಟು += ನಮೂದು.second;
    }
    ಹಿಂತಿರುಗಿ ಒಟ್ಟು;

#ಅಥವಾಆದರೆ defined(__APPLE__) && defined(__aarch64__)
    ಪೂರ್ಣಾಂಕ perf_ಮಟ್ಟಗಳು = 0;
    size_t size = ಗಾತ್ರ(perf_ಮಟ್ಟಗಳು);
    ಆದರೆ (sysctlbyname("hw.nperflevels", &perf_ಮಟ್ಟಗಳು, &size, ಶೂನ್ಯಸೂಚಕ, 0) != 0 ||
        size != ಗಾತ್ರ(perf_ಮಟ್ಟಗಳು) || perf_ಮಟ್ಟಗಳು <= 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    size_t units = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < perf_ಮಟ್ಟಗಳು; ++i) {
        ಅಕ್ಷರ ಕೀಲಿ[64] = {};
        ಪೂರ್ಣಾಂಕ physical_cpus = 0;
        ಪೂರ್ಣಾಂಕ cpus_ಪ್ರತಿ_l2 = 0;

        snprintf(ಕೀಲಿ, ಗಾತ್ರ(ಕೀಲಿ), "hw.perflevel%d.physicalcpu", i);
        size = ಗಾತ್ರ(physical_cpus);
        ಆದರೆ (sysctlbyname(ಕೀಲಿ, &physical_cpus, &size, ಶೂನ್ಯಸೂಚಕ, 0) != 0 ||
            size != ಗಾತ್ರ(physical_cpus) || physical_cpus <= 0) {
            ಮುಂದುವರಿ;
        }

        snprintf(ಕೀಲಿ, ಗಾತ್ರ(ಕೀಲಿ), "hw.perflevel%d.cpusperl2", i);
        size = ಗಾತ್ರ(cpus_ಪ್ರತಿ_l2);
        ಆದರೆ (sysctlbyname(ಕೀಲಿ, &cpus_ಪ್ರತಿ_l2, &size, ಶೂನ್ಯಸೂಚಕ, 0) != 0 ||
            size != ಗಾತ್ರ(cpus_ಪ್ರತಿ_l2) || cpus_ಪ್ರತಿ_l2 <= 0) {
            ಮುಂದುವರಿ;
        }

        snprintf(ಕೀಲಿ, ಗಾತ್ರ(ಕೀಲಿ), "hw.perflevel%d.name", i);
        size = 0;
        ಆದರೆ (sysctlbyname(ಕೀಲಿ, ಶೂನ್ಯಸೂಚಕ, &size, ಶೂನ್ಯಸೂಚಕ, 0) != 0 || size == 0) {
            ಮುಂದುವರಿ;
        }

        std::string name(size, '\0');
        ಆದರೆ (sysctlbyname(ಕೀಲಿ, &name[0], &size, ಶೂನ್ಯಸೂಚಕ, 0) != 0) {
            ಮುಂದುವರಿ;
        }
        name.resize(size);
        ಆಗ (!name.ಖಾಲಿಗಳು() && name.back() == '\0') {
            name.pop_back();
        }

        ಆದರೆ (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_ಪ್ರತಿ_l2 - 1) / cpus_ಪ್ರತಿ_l2);
        }
    }

    ಹಿಂತಿರುಗಿ units;

#ಅಥವಾಆದರೆ defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    ಹಿಂತಿರುಗಿ 0;

#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ parse_uint_env(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *s, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *name, ತಾರ್ಕಿಕ *ok) {
    ಆದರೆ (!s) { *ok = ಸುಳ್ಳು; ಹಿಂತಿರುಗಿ 0; }
    ಅಕ್ಷರ *end = ಶೂನ್ಯಸೂಚಕ;
    ದೀರ್ಘ v = strtol(s, &end, 10);
    ಆದರೆ (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = ಸುಳ್ಳು;
        ಹಿಂತಿರುಗಿ 0;
    }
    ಆದರೆ (v < 0 || v > ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = ಸುಳ್ಳು;
        ಹಿಂತಿರುಗಿ 0;
    }
    *ok = ನಿಜ;
    ಹಿಂತಿರುಗಿ (ಪೂರ್ಣಾಂಕ)v;
}

ಸ್ಥಿರ ಶೂನ್ಯ init_kleidiai_context(ಶೂನ್ಯ) {
    ggml_critical_section_ಆರಂಭ();
    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;

    ಆದರೆ (!initialized) {
        initialized = ನಿಜ;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *env_ಎಳೆಗಳು     = getenv("GGML_TOTAL_THREADS");
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        ಆದರೆ (env_ಎಳೆಗಳು) {
            ತಾರ್ಕಿಕ ok = ಸುಳ್ಳು;
            ಪೂರ್ಣಾಂಕ hint = parse_uint_env(env_ಎಳೆಗಳು, "GGML_TOTAL_THREADS", &ok);
            ಆದರೆ (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        ಆದರೆ (env_chunk_mult) {
            ತಾರ್ಕಿಕ ok = ಸುಳ್ಳು;
            ಪೂರ್ಣಾಂಕ multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            ಆದರೆ (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        ಪೂರ್ಣಾಂಕ sme_cores = 0;
        ತಾರ್ಕಿಕ sme_env_ok = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ sme_env_ಗಣ = (env_sme != ಶೂನ್ಯಸೂಚಕ);

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_supported_sme_family = runtime_feat.has_sme;
        ತಾರ್ಕಿಕ sme_cap_detected = ಸುಳ್ಳು;

        ಆದರೆ (has_supported_sme_family) {
            detected_smcus = detect_ಅಂಕೆ_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (ಪೂರ್ಣಾಂಕ)detected_smcus : 1;

            ಆದರೆ (!sme_env_ಗಣ && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        ಆದರೆ (sme_env_ಗಣ) {
            ತಾರ್ಕಿಕ ok = ಸುಳ್ಳು;
            ಪೂರ್ಣಾಂಕ v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            ಆದರೆ (ok) {
                ಆದರೆ (has_supported_sme_family) {
                    sme_cores = v;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಆದರೆ (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        ಆದರೆ (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            ಆದರೆ (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        ಆದರೆ (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (ಪೂರ್ಣಾಂಕ)ctx.features);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_ಗಮ್ಯ_string(ctx.kernels_q4->required_cpu));
        }

        ಆದರೆ (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (ಪೂರ್ಣಾಂಕ)ctx.features);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_ಗಮ್ಯ_string(ctx.kernels_q8->required_cpu));
        }

        ಆದರೆ (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (ಪೂರ್ಣಾಂಕ) ctx.features);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_ಗಮ್ಯ_string(ctx.kernels_q4k->required_cpu));
        }

        ಆದರೆ (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (ಪೂರ್ಣಾಂಕ)ctx.features);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_ಗಮ್ಯ_string(ctx.kernels_f32->required_cpu));
        }

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        ಆದರೆ (has_selected_sme_family_kernel) {
            ಆದರೆ (sme_env_ಗಣ && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_ಕೊನೆ();
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ kleidiai_sme_thread_cap() {
    ಹಿಂತಿರುಗಿ ctx.sme_thread_cap;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t align_ಮೇಲೆ(size_t value, size_t alignment) {
    ಆದರೆ (alignment == 0) {
        ಹಿಂತಿರುಗಿ value;
    }
    ಸ್ಥಿರಾಂಕ size_t remainder = value % alignment;
    ಹಿಂತಿರುಗಿ remainder == 0 ? value : value + (alignment - remainder);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_ಗಾತ್ರ(size_t a, size_t b) {
    ಆಗ (b != 0) {
        ಸ್ಥಿರಾಂಕ size_t t = a % b;
        a = b;
        b = t;
    }
    ಹಿಂತಿರುಗಿ a;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ lcm_ಗಾತ್ರ(size_t a, size_t b, size_t & ಫಲಿತಾಂಶ) {
    ಆದರೆ (a == 0 || b == 0) {
        ಫಲಿತಾಂಶ = 0;
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ size_t g = ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_ಗಾತ್ರ(a, b);
    ಸ್ಥಿರಾಂಕ size_t q = a / g;
    ಆದರೆ (q > SIZE_MAX / b) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಫಲಿತಾಂಶ = q * b;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t ceil_div_ಗಾತ್ರ(size_t a, size_t b) {
    ಹಿಂತಿರುಗಿ b == 0 ? 0 : (a + b - 1) / b;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t kleidiai_chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ(size_t n, ಪೂರ್ಣಾಂಕ nth_ಒಟ್ಟು, ತಾರ್ಕಿಕ disable_chunking, size_t n_ಹೆಜ್ಜೆ) {
    ಸ್ಥಿರಾಂಕ size_t multiplier = (nth_ಒಟ್ಟು == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ಸ್ಥಿರಾಂಕ size_t ಭಾಜಕ = std::max<size_t>(1, (size_t) nth_ಒಟ್ಟು * multiplier);
    ಸ್ಥಿರಾಂಕ size_t chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = align_ಮೇಲೆ(std::max<size_t>(1, ceil_div_ಗಾತ್ರ(n, ಭಾಜಕ)), n_ಹೆಜ್ಜೆ);
    ಹಿಂತಿರುಗಿ chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ ? chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ : n_ಹೆಜ್ಜೆ;
}

ರಚನೆ kleidiai_ಖಂಡ_ವಾದಗಳು {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ kleidiai_ಖಂಡ_ವಾದಗಳು kleidiai_get_ಖಂಡ_ವಾದಗಳು(ggml_type rhs_type) {
    ಆಯ್ಕೆ (rhs_type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಹಿಂತಿರುಗಿ { QK4_0, QK4_0, QK4_0 };
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಹಿಂತಿರುಗಿ { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ { 0, 0, QK8_0 };
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ { 0, 0, 0 };
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ kleidiai_pack_fallback_allowed() {
    ಆದರೆ (ctx.sme_thread_cap <= 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ctx.thread_hint <= 0) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಹಿಂತಿರುಗಿ ctx.thread_hint > ctx.sme_thread_cap;
}

ರಚನೆ kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ {
    uint32_t magic;
    uint16_t ಆವೃತ್ತಿ_2;
    uint16_t ಸ್ಥಾನ_ಎಣಿಕೆ;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ * kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ_from_ptr(ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ಹಿಂತಿರುಗಿ reinterpret_cast<kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ *>(ದತ್ತಾಂಶ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ * kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ_from_ptr(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ಹಿಂತಿರುಗಿ reinterpret_cast<ಸ್ಥಿರಾಂಕ kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ *>(ದತ್ತಾಂಶ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ kleidiai_is_ತೂಕ_ಶೀರ್ಷಿಕೆ_valid(ಸ್ಥಿರಾಂಕ kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ * ಶೀರ್ಷಿಕೆ) {
    ಆದರೆ (!ಶೀರ್ಷಿಕೆ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ಶೀರ್ಷಿಕೆ->magic != GGML_KLEIDIAI_PACK_MAGIC || ಶೀರ್ಷಿಕೆ->ಆವೃತ್ತಿ_2 != GGML_KLEIDIAI_PACK_ಆವೃತ್ತಿ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ಶೀರ್ಷಿಕೆ->ಸ್ಥಾನ_ಎಣಿಕೆ == 0 || ಶೀರ್ಷಿಕೆ->ಸ್ಥಾನ_ಎಣಿಕೆ > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint8_t * kleidiai_ತೂಕ_ಸ್ಥಾನ_ptr(kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ * ಶೀರ್ಷಿಕೆ, ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ) {
    ಆದರೆ (!kleidiai_is_ತೂಕ_ಶೀರ್ಷಿಕೆ_valid(ಶೀರ್ಷಿಕೆ)) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (ಸ್ಥಾನ < 0 || ಸ್ಥಾನ >= ಶೀರ್ಷಿಕೆ->ಸ್ಥಾನ_ಎಣಿಕೆ) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ reinterpret_cast<uint8_t *>(ಶೀರ್ಷಿಕೆ) + ಶೀರ್ಷಿಕೆ->offsets[ಸ್ಥಾನ];
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ uint8_t * kleidiai_ತೂಕ_ಸ್ಥಾನ_ptr(ಸ್ಥಿರಾಂಕ kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ * ಶೀರ್ಷಿಕೆ, ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ) {
    ಆದರೆ (!kleidiai_is_ತೂಕ_ಶೀರ್ಷಿಕೆ_valid(ಶೀರ್ಷಿಕೆ)) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (ಸ್ಥಾನ < 0 || ಸ್ಥಾನ >= ಶೀರ್ಷಿಕೆ->ಸ್ಥಾನ_ಎಣಿಕೆ) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(ಶೀರ್ಷಿಕೆ) + ಶೀರ್ಷಿಕೆ->offsets[ಸ್ಥಾನ];
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    ಹಿಂತಿರುಗಿ ctx.kernels_q4;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    ಹಿಂತಿರುಗಿ ctx.kernels_q4k;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    ಹಿಂತಿರುಗಿ ctx.kernels_q8;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    ಹಿಂತಿರುಗಿ ctx.kernels_f32;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು SelectFallback>
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ಹೊರಹೋಗು,
        SelectFallback select_fallback) {
    ಪೂರ್ಣಾಂಕ count = 0;
    ಆದರೆ (!primary) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಹೊರಹೋಗು[count++] = primary;

    ಆದರೆ (primary->rhs_info.repack_ಬಹುಲಕ == RHS_REPACK_SINGLE_ONLY) {
        ಹಿಂತಿರುಗಿ count;
    }

    ಆದರೆ (is_sme_family(primary->required_cpu)) {
        ಸ್ಥಿರಾಂಕ cpu_feature fallback_ಮುಖವಾಡ = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        ಆದರೆ (fallback_ಮುಖವಾಡ != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_ಮುಖವಾಡ);
            ಆದರೆ (fallback && fallback != primary &&
                fallback->rhs_info.repack_ಬಹುಲಕ != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                ಹೊರಹೋಗು[count++] = fallback;
            }
        }
    }

    ಹಿಂತಿರುಗಿ count;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ kleidiai_collect_kernel_chain(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ಹೊರಹೋಗು) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    ಹಿಂತಿರುಗಿ kleidiai_collect_kernel_chain_common(primary, ctx.features, ಹೊರಹೋಗು,
        [&](cpu_feature ಮುಖವಾಡ) { ಹಿಂತಿರುಗಿ ggml_kleidiai_select_kernels(ಮುಖವಾಡ, op); });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ಹೊರಹೋಗು) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    ಹಿಂತಿರುಗಿ kleidiai_collect_kernel_chain_common(primary, ctx.features, ಹೊರಹೋಗು,
        [&](cpu_feature ಮುಖವಾಡ) { ಹಿಂತಿರುಗಿ ggml_kleidiai_select_kernels_q4_0(ಮುಖವಾಡ); });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ಹೊರಹೋಗು) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    ಹಿಂತಿರುಗಿ kleidiai_collect_kernel_chain_common(primary, ctx.features, ಹೊರಹೋಗು,
        [&](cpu_feature ಮುಖವಾಡ) { ಹಿಂತಿರುಗಿ ggml_kleidiai_select_kernels_q4_k(ಮುಖವಾಡ); });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ಹೊರಹೋಗು) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    ಹಿಂತಿರುಗಿ kleidiai_collect_kernel_chain_common(primary, ctx.features, ಹೊರಹೋಗು,
        [&](cpu_feature ಮುಖವಾಡ) { ಹಿಂತಿರುಗಿ ggml_kleidiai_select_kernels_q8_0(ಮುಖವಾಡ); });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ಹೊರಹೋಗು) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    ಹಿಂತಿರುಗಿ kleidiai_collect_kernel_chain_common(primary, ctx.features, ಹೊರಹೋಗು,
        [&](cpu_feature ಮುಖವಾಡ) { ಹಿಂತಿರುಗಿ ggml_kleidiai_select_kernels_f32(ಮುಖವಾಡ); });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t ggml_ne(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    ಹಿಂತಿರುಗಿ tensor->ne[dim];
}

ನಾಮಾಕಾಶ ggml::cpu::kleidiai {

ಸ್ಥಿರ size_t ಸುತ್ತು_ಕೆಳಕ್ಕೆ(size_t x, size_t y) {
    ಹಿಂತಿರುಗಿ y == 0 ? x : x - (x % y);
}

ಸ್ಥಿರ ಶೂನ್ಯ transpose_f32kxn_f16nxk(size_t n, size_t k, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / ಗಾತ್ರ(uint16_t);
    size_t dst_stride = n;

    ಗಾಗಿ (size_t k_idx = 0; k_idx < k; ++k_idx) {
        ಗಾಗಿ (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

ವರ್ಗ tensor_traits : ಸಾರ್ವಜನಿಕ ggml::cpu::tensor_traits {
    ತಾರ್ಕಿಕ work_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ /* n_threads */, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, size_t & size) override {
        ಆದರೆ (op->op != GGML_OP_MUL_MAT) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ_ಎಣಿಕೆ = kleidiai_collect_kernel_chain(op, kernel_chain);
        ಆದರೆ (ಸ್ಥಾನ_ಎಣಿಕೆ == 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_gemv = op->src[1]->ne[1] == 1;
        ಸ್ಥಿರಾಂಕ size_t k = op->src[0]->ne[0];
        ಸ್ಥಿರಾಂಕ size_t n = op->src[0]->ne[1];
        ಸ್ಥಿರಾಂಕ size_t m = op->src[1]->ne[1];

        ಆದರೆ (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ಸ್ಥಿರಾಂಕ kleidiai_ಖಂಡ_ವಾದಗಳು ಖಂಡ_ವಾದಗಳು = kleidiai_get_ಖಂಡ_ವಾದಗಳು(kernel_chain[0]->rhs_type);
            ಸ್ಥಿರಾಂಕ size_t qk = ಖಂಡ_ವಾದಗಳು.pack_bl;

            size_t cursor = 0;
            ತಾರ್ಕಿಕ any_ಸ್ಥಾನ = ಸುಳ್ಳು;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = 0; ಸ್ಥಾನ < ಸ್ಥಾನ_ಎಣಿಕೆ; ++ಸ್ಥಾನ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[ಸ್ಥಾನ];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                ಆದರೆ (!lhs_info || !lhs_info->packed_ಗಾತ್ರ_ex || !kernel) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಸ್ಥಿರಾಂಕ size_t mr = kernel->get_mr();
                ಸ್ಥಿರಾಂಕ size_t kr = kernel->get_kr();
                ಸ್ಥಿರಾಂಕ size_t sr = kernel->get_sr();

                ಸ್ಥಿರಾಂಕ size_t packed = lhs_info->packed_ಗಾತ್ರ_ex(m, k, qk, mr, kr, sr);

                cursor = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_ಸ್ಥಾನ = ನಿಜ;
            }

            ಆದರೆ (!any_ಸ್ಥಾನ) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            size = cursor;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : ಶೂನ್ಯಸೂಚಕ;
            ಆದರೆ (is_gemv && op->src[1]->nb[0] == (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ) && gemv_kernel &&
                gemv_kernel->get_lhs_ಸ್ಥಾನಾಂತರ_ex && gemv_kernel->get_rhs_packed_ಸ್ಥಾನಾಂತರ_ex &&
                gemv_kernel->ಸತತ_kernel_ex && gemv_kernel->get_dst_ಸ್ಥಾನಾಂತರ) {
                size = 0;
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }

            size_t cursor = 0;
            ತಾರ್ಕಿಕ any_ಸ್ಥಾನ = ಸುಳ್ಳು;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = 0; ಸ್ಥಾನ < ಸ್ಥಾನ_ಎಣಿಕೆ; ++ಸ್ಥಾನ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[ಸ್ಥಾನ];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                ಆದರೆ (!lhs_info || !lhs_info->packed_ಗಾತ್ರ_ex || !kernel) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಸ್ಥಿರಾಂಕ size_t mr = kernel->get_mr();
                ಸ್ಥಿರಾಂಕ size_t kr = kernel->get_kr();
                ಸ್ಥಿರಾಂಕ size_t sr = kernel->get_sr();

                cursor  = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_ಗಾತ್ರ_ex(m, k, 0, mr, kr, sr);
                any_ಸ್ಥಾನ = ನಿಜ;
            }

            ಆದರೆ (!any_ಸ್ಥಾನ) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            size = cursor;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (op->src[0]->type == GGML_TYPE_F16) {
            ಸ್ಥಿರಾಂಕ int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ಸ್ಥಿರಾಂಕ int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ಸ್ಥಿರಾಂಕ int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            ತಾರ್ಕಿಕ any_ಸ್ಥಾನ = ಸುಳ್ಳು;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = 0; ಸ್ಥಾನ < ಸ್ಥಾನ_ಎಣಿಕೆ; ++ಸ್ಥಾನ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[ಸ್ಥಾನ];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                ಆದರೆ (!lhs_info || !lhs_info->packed_ಗಾತ್ರ_ex || !kernels->rhs_info.packed_ಗಾತ್ರ_ex || !kernel) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಸ್ಥಿರಾಂಕ size_t mr = kernel->get_mr();
                ಸ್ಥಿರಾಂಕ size_t kr = kernel->get_kr();
                ಸ್ಥಿರಾಂಕ size_t sr = kernel->get_sr();

                cursor  = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_ಗಾತ್ರ_ex(m * r, k, 0, mr, kr, sr);
                any_ಸ್ಥಾನ = ನಿಜ;
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = 0; ಸ್ಥಾನ < ಸ್ಥಾನ_ಎಣಿಕೆ; ++ಸ್ಥಾನ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[ಸ್ಥಾನ];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                ಆದರೆ (!kernel || !kernels->rhs_info.packed_ಗಾತ್ರ_ex) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                cursor  = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_ಗಾತ್ರ_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * ಗಾತ್ರ(ದಶಮಾಂಶ);
            cursor  = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * ಗಾತ್ರ(ದಶಮಾಂಶ);

            ಆದರೆ (!any_ಸ್ಥಾನ) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            size = cursor;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ತಾರ್ಕಿಕ compute_forward(ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst) override {
        ಆದರೆ (dst->op == GGML_OP_MUL_MAT) {
            ಆದರೆ (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                ಹಿಂತಿರುಗಿ compute_forward_qx(params, dst);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->src[0]->type == GGML_TYPE_F32) {
                ಹಿಂತಿರುಗಿ compute_forward_f32(params, dst);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->src[0]->type == GGML_TYPE_F16) {
                ಹಿಂತಿರುಗಿ compute_forward_fp16(params, dst);
            }
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ತಾರ್ಕಿಕ compute_forward_f32(ggml_compute_params * params, ರಚನೆ ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

        ಆದರೆ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        ಆದರೆ (!kernels) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ size_t k = ne00;
        ಸ್ಥಿರಾಂಕ size_t m = ne11;
        ಸ್ಥಿರಾಂಕ size_t n = ne01;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_gemv = m == 1 && src1->nb[0] == (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                              kernels->gemv.get_lhs_ಸ್ಥಾನಾಂತರ_ex &&
                              kernels->gemv.get_rhs_packed_ಸ್ಥಾನಾಂತರ_ex &&
                              kernels->gemv.ಸತತ_kernel_ex &&
                              kernels->gemv.get_dst_ಸ್ಥಾನಾಂತರ;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        ಆದರೆ (!kernel || !kernel->get_lhs_ಸ್ಥಾನಾಂತರ_ex ||
            !kernel->get_rhs_packed_ಸ್ಥಾನಾಂತರ_ex || !kernel->ಸತತ_kernel_ex || !kernel->get_dst_ಸ್ಥಾನಾಂತರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (!use_gemv && (!lhs_info || !lhs_info->get_ಸ್ಥಾನಾಂತರ || !lhs_info->get_packed_ಸ್ಥಾನಾಂತರ_ex ||
                          !lhs_info->packed_ಗಾತ್ರ_ex || !lhs_info->pack_func_ex)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ * ಶೀರ್ಷಿಕೆ = kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ_from_ptr(src0->ದತ್ತಾಂಶ);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_ಶೀರ್ಷಿಕೆ = kleidiai_is_ತೂಕ_ಶೀರ್ಷಿಕೆ_valid(ಶೀರ್ಷಿಕೆ);

        ಸ್ಥಿರಾಂಕ uint8_t * rhs_ತಳ = has_ಶೀರ್ಷಿಕೆ ? kleidiai_ತೂಕ_ಸ್ಥಾನ_ptr(ಶೀರ್ಷಿಕೆ, 0)
                                              : static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(src0->ದತ್ತಾಂಶ);
        ಆದರೆ (!rhs_ತಳ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth > 0 ? params->nth : 1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;

        ಸ್ಥಿರಾಂಕ size_t mr = kernel->get_mr();
        ಸ್ಥಿರಾಂಕ size_t kr = kernel->get_kr();
        ಸ್ಥಿರಾಂಕ size_t sr = kernel->get_sr();

        ಸ್ಥಿರಾಂಕ size_t lhs_packed_ಗಾತ್ರ = use_gemv ? 0 : lhs_info->packed_ಗಾತ್ರ_ex(m, k, 0, mr, kr, sr);
        ಆದರೆ (!use_gemv) {
            GGML_ASSERT(lhs_packed_ಗಾತ್ರ <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ಸ್ಥಿರಾಂಕ size_t dst_stride = dst->nb[1];
        ಸ್ಥಿರಾಂಕ size_t n_ಹೆಜ್ಜೆ = kernel->get_n_ಹೆಜ್ಜೆ() ? kernel->get_n_ಹೆಜ್ಜೆ() : 1;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ);

        ಸ್ಥಿರಾಂಕ int64_t batch_ಗಾತ್ರ = ne12 * ne13;
        ಗಾಗಿ (int64_t batch_idx = 0; batch_idx < batch_ಗಾತ್ರ; ++batch_idx) {
            ಸ್ಥಿರಾಂಕ int64_t i12 = batch_idx % ne12;
            ಸ್ಥಿರಾಂಕ int64_t i13 = batch_idx / ne12;
            ಸ್ಥಿರಾಂಕ uint8_t * lhs_batch_ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(src1->ದತ್ತಾಂಶ) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_ತಳ = static_cast<uint8_t *>(dst->ದತ್ತಾಂಶ) + i12 * dst->nb[2] + i13 * dst->nb[3];

            ಆದರೆ (!use_gemv) {
                ಸ್ಥಿರಾಂಕ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_ಎಳೆಗಳು = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_ಎಳೆಗಳು = std::max<int64_t>(1, max_ಎಳೆಗಳು);
                ಸ್ಥಿರಾಂಕ int64_t use_ಎಳೆಗಳು = std::min<int64_t>(nth, max_ಎಳೆಗಳು);

                ಆದರೆ (ith < use_ಎಳೆಗಳು) {
                    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_m_ಪ್ರತಿ_thread0   = ಸುತ್ತು_ಕೆಳಕ್ಕೆ((size_t)(m_roundup_mr / use_ಎಳೆಗಳು), mr);
                    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_m_ಪ್ರತಿ_thread_N_1 = (int64_t)m - (use_ಎಳೆಗಳು - 1) * ಅಂಕೆ_m_ಪ್ರತಿ_thread0;

                    ಸ್ಥಿರಾಂಕ int64_t m_ಆರಂಭ = (int64_t)ith * ಅಂಕೆ_m_ಪ್ರತಿ_thread0;
                    ಸ್ಥಿರಾಂಕ int64_t m_ಎಣಿಕೆ = (ith == use_ಎಳೆಗಳು - 1) ? ಅಂಕೆ_m_ಪ್ರತಿ_thread_N_1 : ಅಂಕೆ_m_ಪ್ರತಿ_thread0;

                    ಸ್ಥಿರಾಂಕ size_t ತಳ_packed_off  = lhs_info->get_packed_ಸ್ಥಾನಾಂತರ_ex(m_ಆರಂಭ, k, 0, mr, kr, sr);
                    ಸ್ಥಿರಾಂಕ size_t ಮುಂದಿನ_ಖಂಡ_off   = lhs_info->get_packed_ಸ್ಥಾನಾಂತರ_ex(m_ಆರಂಭ + mr, k, 0, mr, kr, sr);
                    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride_ಬೈಟುಗಳು = mr ? (ಮುಂದಿನ_ಖಂಡ_off - ತಳ_packed_off) / mr : 0;

                    int64_t ಉಳಿದ = m_ಎಣಿಕೆ;
                    int64_t cur       = m_ಆರಂಭ;

                    ಆಗ (ಉಳಿದ > 0) {
                        ಸ್ಥಿರಾಂಕ int64_t take = std::min<int64_t>((int64_t)m - cur, ಉಳಿದ);
                        ಸ್ಥಿರಾಂಕ size_t src_off = lhs_info->get_ಸ್ಥಾನಾಂತರ(cur, src1->nb[1]);
                        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src_ptr = lhs_batch_ತಳ + src_off;
                        ಸ್ಥಿರಾಂಕ size_t dst_off = ತಳ_packed_off + (size_t)(cur - m_ಆರಂಭ) * ಅಡ್ಡಸಾಲು_stride_ಬೈಟುಗಳು;
                        ಶೂನ್ಯ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        ಉಳಿದ -= take;
                    }
                }
            }

            ಆದರೆ (ith == 0) {
                ggml_threadpool_chunk_ಗಣ(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ಸ್ಥಿರಾಂಕ size_t chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = kleidiai_chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ(n, nth, disable_chunking, n_ಹೆಜ್ಜೆ);
            GGML_ASSERT(chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ <= (size_t) ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ);

            ಪೂರ್ಣಾಂಕ current_ಲಂಬಸಾಲು = ggml_threadpool_chunk_ಕೂಡಿಸು(params->threadpool, (ಪೂರ್ಣಾಂಕ) chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
            ಆಗ ((size_t) current_ಲಂಬಸಾಲು < n) {
                ಸ್ಥಿರಾಂಕ size_t n_ಆರಂಭ = (size_t) current_ಲಂಬಸಾಲು;
                ಸ್ಥಿರಾಂಕ size_t n_ಗಮ್ಯ_process = std::min(chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n - n_ಆರಂಭ);

                ಆದರೆ (n_ಗಮ್ಯ_process > 0) {
                    ಸ್ಥಿರಾಂಕ size_t lhs_ಸ್ಥಾನಾಂತರ = use_gemv ? kernel->get_lhs_ಸ್ಥಾನಾಂತರ_ex(0, k, 0)
                                                       : lhs_info->get_packed_ಸ್ಥಾನಾಂತರ_ex(0, k, 0, mr, kr, sr);
                    ಸ್ಥಿರಾಂಕ size_t rhs_packed_ಸ್ಥಾನಾಂತರ = kernel->get_rhs_packed_ಸ್ಥಾನಾಂತರ_ex(n_ಆರಂಭ, k, 0);
                    ಸ್ಥಿರಾಂಕ size_t dst_ಸ್ಥಾನಾಂತರ        = kernel->get_dst_ಸ್ಥಾನಾಂತರ(0, n_ಆರಂಭ, dst_stride);

                    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * lhs_ptr = use_gemv ? lhs_batch_ತಳ + lhs_ಸ್ಥಾನಾಂತರ
                                                    : lhs_packed + lhs_ಸ್ಥಾನಾಂತರ;
                    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * rhs_ptr = rhs_ತಳ + rhs_packed_ಸ್ಥಾನಾಂತರ;
                    ದಶಮಾಂಶ * dst_ptr      = reinterpret_cast<ದಶಮಾಂಶ *>(dst_batch_ತಳ + dst_ಸ್ಥಾನಾಂತರ);

                    kernel->ಸತತ_kernel_ex(m, n_ಗಮ್ಯ_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          ಗಾತ್ರ(ದಶಮಾಂಶ),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_ಲಂಬಸಾಲು = ggml_threadpool_chunk_ಕೂಡಿಸು(params->threadpool, (ಪೂರ್ಣಾಂಕ) chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
            }

            ಆದರೆ (batch_idx != batch_ಗಾತ್ರ - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ compute_forward_fp16(ggml_compute_params * params, ರಚನೆ ggml_tensor * dst) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        ಆದರೆ (!kernels) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        ಆದರೆ (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_ಸ್ಥಾನಾಂತರ_ex || !kernel->get_rhs_packed_ಸ್ಥಾನಾಂತರ_ex || !kernel->ಸತತ_kernel_ex) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;

        ಸ್ಥಿರಾಂಕ int64_t lhs_batch_size0 = ne12;
        ಸ್ಥಿರಾಂಕ int64_t rhs_batch_size0 = ne02;
        ಸ್ಥಿರಾಂಕ int64_t batch_ಗಾತ್ರ      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        ಸ್ಥಿರಾಂಕ int64_t r = lhs_batch_size0 / rhs_batch_size0;

        ಸ್ಥಿರಾಂಕ int64_t m_ಗುಂಪು = ne11;
        ಸ್ಥಿರಾಂಕ int64_t m       = m_ಗುಂಪು;
        ಸ್ಥಿರಾಂಕ int64_t n       = ne01;
        ಸ್ಥಿರಾಂಕ int64_t k       = ne00;

        ಸ್ಥಿರಾಂಕ size_t lhs_stride = src1->nb[1];
        ಸ್ಥಿರಾಂಕ size_t rhs_stride = src0->nb[1];
        ಸ್ಥಿರಾಂಕ size_t dst_stride = dst->nb[1];

        ಸ್ಥಿರಾಂಕ int64_t mr = (int64_t) kernel->get_mr();
        ಸ್ಥಿರಾಂಕ int64_t nr = (int64_t) kernel->get_nr();
        ಸ್ಥಿರಾಂಕ int64_t kr = (int64_t) kernel->get_kr();
        ಸ್ಥಿರಾಂಕ int64_t sr = (int64_t) kernel->get_sr();

        ಸ್ಥಿರಾಂಕ size_t lhs_packed_ಗಾತ್ರ = lhs_info->packed_ಗಾತ್ರ_ex(m, k, 0, mr, kr, sr);
        ಸ್ಥಿರಾಂಕ size_t rhs_packed_ಗಾತ್ರ = kernels->rhs_info.packed_ಗಾತ್ರ_ex(n, k, nr, kr, 0);
        ಸ್ಥಿರಾಂಕ size_t kxn_ಗಾತ್ರ        = k * n * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ size_t bias_ಗಾತ್ರ       = n * ಗಾತ್ರ(ದಶಮಾಂಶ);

        ಸ್ಥಿರಾಂಕ size_t wsize_required = lhs_packed_ಗಾತ್ರ + rhs_packed_ಗಾತ್ರ + kxn_ಗಾತ್ರ + bias_ಗಾತ್ರ;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_ಗಾತ್ರ;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_ಗಾತ್ರ;
        uint8_t * bias       = rhs_kxn + kxn_ಗಾತ್ರ;

        ಗಾಗಿ (int64_t batch_idx = 0; batch_idx < batch_ಗಾತ್ರ; ++batch_idx) {
            ಸ್ಥಿರಾಂಕ int64_t rhs_batch_idx = batch_idx / r;
            ಸ್ಥಿರಾಂಕ uint8_t * rhs_batch_ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(src0->ದತ್ತಾಂಶ) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_ತಳ = static_cast<uint8_t *>(dst->ದತ್ತಾಂಶ) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ಸ್ಥಿರಾಂಕ int64_t m_roundup_mr = kai_roundup(m, mr);
                ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಎಳೆಗಳು  = KAI_MIN(m_roundup_mr / mr, nth);

                ಆದರೆ (ith < ಅಂಕೆ_ಎಳೆಗಳು) {
                    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_m_ಪ್ರತಿ_thread0   = ಸುತ್ತು_ಕೆಳಕ್ಕೆ((size_t)(m_roundup_mr / ಅಂಕೆ_ಎಳೆಗಳು), (size_t)mr);
                    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_m_ಪ್ರತಿ_thread_N_1 = m - (ಅಂಕೆ_ಎಳೆಗಳು - 1) * ಅಂಕೆ_m_ಪ್ರತಿ_thread0;

                    ಸ್ಥಿರಾಂಕ int64_t m_ಆರಂಭ = ith * ಅಂಕೆ_m_ಪ್ರತಿ_thread0;
                    ಸ್ಥಿರಾಂಕ int64_t m_ಎಣಿಕೆ = (ith == ಅಂಕೆ_ಎಳೆಗಳು - 1) ? ಅಂಕೆ_m_ಪ್ರತಿ_thread_N_1 : ಅಂಕೆ_m_ಪ್ರತಿ_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ಸ್ಥಿರಾಂಕ size_t ತಳ_packed_off  = lhs_info->get_packed_ಸ್ಥಾನಾಂತರ_ex(m_ಆರಂಭ, k, 0, mr, kr, sr);
                    ಸ್ಥಿರಾಂಕ size_t ಮುಂದಿನ_ಖಂಡ_off   = lhs_info->get_packed_ಸ್ಥಾನಾಂತರ_ex(m_ಆರಂಭ + mr, k, 0, mr, kr, sr);
                    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride_ಬೈಟುಗಳು = (ಮುಂದಿನ_ಖಂಡ_off - ತಳ_packed_off) / (size_t)mr;

                    int64_t ಉಳಿದ = m_ಎಣಿಕೆ;
                    int64_t cur       = m_ಆರಂಭ;

                    ಆಗ (ಉಳಿದ > 0) {
                        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಒಳ_ಗುಂಪು = cur;
                        ಸ್ಥಿರಾಂಕ int64_t avail        = m_ಗುಂಪು - ಅಡ್ಡಸಾಲು_ಒಳ_ಗುಂಪು;
                        ಸ್ಥಿರಾಂಕ int64_t take         = std::min(avail, ಉಳಿದ);

                        ಸ್ಥಿರಾಂಕ uint8_t * lhs_batch_ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(src1->ದತ್ತಾಂಶ) + batch_idx * src1->nb[2];
                        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src_ptr = lhs_batch_ತಳ + (size_t)ಅಡ್ಡಸಾಲು_ಒಳ_ಗುಂಪು * lhs_stride;
                        ಸ್ಥಿರಾಂಕ size_t dst_off = ತಳ_packed_off + (size_t)(cur - m_ಆರಂಭ) * ಅಡ್ಡಸಾಲು_stride_ಬೈಟುಗಳು;
                        ಶೂನ್ಯ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        ಉಳಿದ -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            ಆದರೆ (ith == 0) {
                memset(bias, 0, (size_t)n * ಗಾತ್ರ(ದಶಮಾಂಶ));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<ದಶಮಾಂಶ *>(rhs_kxn),
                                        reinterpret_cast<ಸ್ಥಿರಾಂಕ uint16_t *>(rhs_batch_ತಳ),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * ಗಾತ್ರ(ದಶಮಾಂಶ),
                             rhs_kxn, bias, ಶೂನ್ಯಸೂಚಕ, rhs_packed, 0, ಶೂನ್ಯಸೂಚಕ);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ಸ್ಥಿರಾಂಕ int64_t n_ಹೆಜ್ಜೆ  = (int64_t) kernel->get_n_ಹೆಜ್ಜೆ();
                int64_t ಅಂಕೆ_ಎಳೆಗಳು_n = KAI_MIN(n / n_ಹೆಜ್ಜೆ, nth);
                ಆದರೆ (ಅಂಕೆ_ಎಳೆಗಳು_n <= 0) {
                    ಅಂಕೆ_ಎಳೆಗಳು_n = 1;
                }

                ಆದರೆ (ith < ಅಂಕೆ_ಎಳೆಗಳು_n) {
                    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_n_ಪ್ರತಿ_thread0   = ಸುತ್ತು_ಕೆಳಕ್ಕೆ((size_t)(n / ಅಂಕೆ_ಎಳೆಗಳು_n), (size_t)n_ಹೆಜ್ಜೆ);
                    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_n_ಪ್ರತಿ_thread_N_1 = n - (ಅಂಕೆ_ಎಳೆಗಳು_n - 1) * ಅಂಕೆ_n_ಪ್ರತಿ_thread0;

                    ಸ್ಥಿರಾಂಕ int64_t n_ಆರಂಭ      = ith * ಅಂಕೆ_n_ಪ್ರತಿ_thread0;
                    ಸ್ಥಿರಾಂಕ int64_t n_ಗಮ್ಯ_process = (ith == ಅಂಕೆ_ಎಳೆಗಳು_n - 1) ? ಅಂಕೆ_n_ಪ್ರತಿ_thread_N_1 : ಅಂಕೆ_n_ಪ್ರತಿ_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ಸ್ಥಿರಾಂಕ size_t lhs_packed_offset0 = lhs_info->get_packed_ಸ್ಥಾನಾಂತರ_ex(0, k, 0, mr, kr, sr);
                    ಸ್ಥಿರಾಂಕ size_t rhs_packed_ಸ್ಥಾನಾಂತರ  = kernel->get_rhs_packed_ಸ್ಥಾನಾಂತರ_ex(n_ಆರಂಭ, k, 0);
                    ಸ್ಥಿರಾಂಕ size_t dst_ಸ್ಥಾನಾಂತರ         = kernel->get_dst_ಸ್ಥಾನಾಂತರ((size_t)0, (size_t)n_ಆರಂಭ, dst_stride);

                    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * rhs_ptr = rhs_packed + rhs_packed_ಸ್ಥಾನಾಂತರ;
                    ದಶಮಾಂಶ * dst_ptr      = reinterpret_cast<ದಶಮಾಂಶ *>(dst_batch_ತಳ + dst_ಸ್ಥಾನಾಂತರ);

                    kernel->ಸತತ_kernel_ex(m, n_ಗಮ್ಯ_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, ಗಾತ್ರ(ದಶಮಾಂಶ), -FLT_MAX, FLT_MAX);
                }
            }

            ಆದರೆ (batch_idx != batch_ಗಾತ್ರ - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ compute_forward_qx(ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

        ಸ್ಥಿರಾಂಕ kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ * ಶೀರ್ಷಿಕೆ = kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ_from_ptr(src0->ದತ್ತಾಂಶ);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_ಶೀರ್ಷಿಕೆ = kleidiai_is_ತೂಕ_ಶೀರ್ಷಿಕೆ_valid(ಶೀರ್ಷಿಕೆ);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ_ಒಟ್ಟು = kleidiai_collect_kernel_chain(dst, kernel_chain);

        ಸ್ವಯಂಚಾಲಿತ ತೂಕ_for_ಸ್ಥಾನ = [&](ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ_ಸೂಚ್ಯಂಕ, size_t & ಗಾತ್ರ_ಹೊರಹೋಗು) -> ಸ್ಥಿರಾಂಕ uint8_t * {
            ಆದರೆ (ಸ್ಥಾನ_ಸೂಚ್ಯಂಕ < 0 || ಸ್ಥಾನ_ಸೂಚ್ಯಂಕ >= ಸ್ಥಾನ_ಒಟ್ಟು) {
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
            ಆದರೆ (has_ಶೀರ್ಷಿಕೆ) {
                ಆದರೆ (ಸ್ಥಾನ_ಸೂಚ್ಯಂಕ < ಶೀರ್ಷಿಕೆ->ಸ್ಥಾನ_ಎಣಿಕೆ) {
                    ಗಾತ್ರ_ಹೊರಹೋಗು = static_cast<size_t>(ಶೀರ್ಷಿಕೆ->sizes[ಸ್ಥಾನ_ಸೂಚ್ಯಂಕ]);
                    ಹಿಂತಿರುಗಿ kleidiai_ತೂಕ_ಸ್ಥಾನ_ptr(ಶೀರ್ಷಿಕೆ, ಸ್ಥಾನ_ಸೂಚ್ಯಂಕ);
                }
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
            ಆದರೆ (ಸ್ಥಾನ_ಸೂಚ್ಯಂಕ == 0) {
                ಗಾತ್ರ_ಹೊರಹೋಗು = ggml_nbytes(src0);
                ಹಿಂತಿರುಗಿ static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(src0->ದತ್ತಾಂಶ);
            }
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        };

        ರಚನೆ runtime_ಸ್ಥಾನ {
            ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ_ಸೂಚ್ಯಂಕ;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            size_t mr;
            size_t nr;
            size_t kr;
            size_t sr;
            size_t n_ಹೆಜ್ಜೆ;
            size_t lhs_packed_ಗಾತ್ರ;
            size_t lhs_ಸ್ಥಾನಾಂತರ;
            size_t lhs_bl;
            size_t rhs_bl;
            size_t pack_bl;
            size_t lhs_packed_offset0;
            ಪೂರ್ಣಾಂಕ assigned_ಎಳೆಗಳು;
            ಪೂರ್ಣಾಂಕ thread_begin;
            ಪೂರ್ಣಾಂಕ thread_ಕೊನೆ;
            ಸ್ಥಿರಾಂಕ uint8_t * rhs_ತಳ;
        };

        std::array<runtime_ಸ್ಥಾನ, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        ಪೂರ್ಣಾಂಕ runtime_ಎಣಿಕೆ = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = 0; ಸ್ಥಾನ < ಸ್ಥಾನ_ಒಟ್ಟು && runtime_ಎಣಿಕೆ < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++ಸ್ಥಾನ) {
            ggml_kleidiai_kernels * kernels = kernel_chain[ಸ್ಥಾನ];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            ಆದರೆ (!kinfo || !linfo || !linfo->packed_ಗಾತ್ರ_ex || !linfo->pack_func_ex || !linfo->get_ಸ್ಥಾನಾಂತರ ||
                !kinfo->get_rhs_packed_ಸ್ಥಾನಾಂತರ_ex || !kinfo->ಸತತ_kernel_ex || !kinfo->get_dst_ಸ್ಥಾನಾಂತರ) {
                ಮುಂದುವರಿ;
            }

            size_t rhs_ಗಾತ್ರ = 0;
            ಸ್ಥಿರಾಂಕ uint8_t * rhs_ptr = ತೂಕ_for_ಸ್ಥಾನ(ಸ್ಥಾನ, rhs_ಗಾತ್ರ);
            ಆದರೆ (!rhs_ptr || rhs_ಗಾತ್ರ == 0) {
                ಮುಂದುವರಿ;
            }

            ಸ್ಥಿರಾಂಕ kleidiai_ಖಂಡ_ವಾದಗಳು ಖಂಡ_ವಾದಗಳು = kleidiai_get_ಖಂಡ_ವಾದಗಳು(kernels->rhs_type);

            runtime[runtime_ಎಣಿಕೆ] = {
                ಸ್ಥಾನ,
                kernels,
                kinfo,
                linfo,
                kinfo->get_mr(),
                kinfo->get_nr(),
                kinfo->get_kr(),
                kinfo->get_sr(),
                kinfo->get_n_ಹೆಜ್ಜೆ(),
                0,
                0,
                ಖಂಡ_ವಾದಗಳು.lhs_bl,
                ಖಂಡ_ವಾದಗಳು.rhs_bl,
                ಖಂಡ_ವಾದಗಳು.pack_bl,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_ಎಣಿಕೆ;
        }

        ಆದರೆ (runtime_ಎಣಿಕೆ == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth_ಒಟ್ಟು = params->nth > 0 ? params->nth : 1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith_ಒಟ್ಟು = params->ith;

        ಪೂರ್ಣಾಂಕ sme_ಸ್ಥಾನ = -1;
        ಪೂರ್ಣಾಂಕ non_sme_ಸ್ಥಾನ = -1;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            ಆದರೆ (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_ಸ್ಥಾನ = i;
                ನಿಲ್ಲಿಸು;
            }
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            ಆದರೆ (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_ಸ್ಥಾನ = i;
                ನಿಲ್ಲಿಸು;
            }
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sme_cap_ಎಲ್ಲೆ = ctx.sme_thread_cap;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_hybrid = sme_cap_ಎಲ್ಲೆ > 0 &&
                                 runtime_ಎಣಿಕೆ > 1 &&
                                 nth_ಒಟ್ಟು > sme_cap_ಎಲ್ಲೆ;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_thread = 0;
        ಆದರೆ (runtime_ಎಣಿಕೆ > 0 && nth_ಒಟ್ಟು > 0) {
            min_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_ಒಟ್ಟು);
        }
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ too_small_for_hybrid = (min_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_thread < 2) || (ne11 < 128);

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        ಆದರೆ (!hybrid_enabled) {
            ಪೂರ್ಣಾಂಕ chosen_ಸ್ಥಾನ = 0;
            ಆದರೆ (too_small_for_hybrid && sme_ಸ್ಥಾನ != -1) {
                chosen_ಸ್ಥಾನ = nth_ಒಟ್ಟು > sme_cap_ಎಲ್ಲೆ && non_sme_ಸ್ಥಾನ != -1 ? non_sme_ಸ್ಥಾನ : sme_ಸ್ಥಾನ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (runtime_ಎಣಿಕೆ > 1 && ctx.sme_thread_cap > 0 && nth_ಒಟ್ಟು > ctx.sme_thread_cap) {
                chosen_ಸ್ಥಾನ = 1;
            }
            ಆದರೆ (chosen_ಸ್ಥಾನ != 0 && chosen_ಸ್ಥಾನ < runtime_ಎಣಿಕೆ) {
                runtime[0] = runtime[chosen_ಸ್ಥಾನ];
                runtime[0].assigned_ಎಳೆಗಳು = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_ಕೊನೆ = 0;
            }
            runtime_ಎಣಿಕೆ = runtime_ಎಣಿಕೆ > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_ಸ್ಥಾನ = -1;
            ಆದರೆ (runtime_ಎಣಿಕೆ > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_ಸ್ಥಾನ = 0;
            }
        }

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ sme_single_only = sme_ಸ್ಥಾನ != -1 &&
                                     runtime[sme_ಸ್ಥಾನ].kernels->rhs_info.repack_ಬಹುಲಕ == RHS_REPACK_SINGLE_ONLY;
        ಪೂರ್ಣಾಂಕ sme_cap = sme_single_only ? nth_ಒಟ್ಟು : kleidiai_sme_thread_cap();
        ಆದರೆ (sme_cap < 0) {
            sme_cap = nth_ಒಟ್ಟು;
        }
        sme_cap = std::min(sme_cap, nth_ಒಟ್ಟು);

        ಪೂರ್ಣಾಂಕ ಎಳೆಗಳು_ಉಳಿದ = nth_ಒಟ್ಟು;
        ಆದರೆ (sme_ಸ್ಥಾನ != -1) {
            ಪೂರ್ಣಾಂಕ sme_ಎಳೆಗಳು = std::min(std::max(sme_cap, 0), ಎಳೆಗಳು_ಉಳಿದ);
            runtime[sme_ಸ್ಥಾನ].assigned_ಎಳೆಗಳು = sme_ಎಳೆಗಳು;
            ಎಳೆಗಳು_ಉಳಿದ -= sme_ಎಳೆಗಳು;
        }

        ಪೂರ್ಣಾಂಕ fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        ಪೂರ್ಣಾಂಕ fallback_ಎಣಿಕೆ = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            ಆದರೆ (i == sme_ಸ್ಥಾನ) {
                ಮುಂದುವರಿ;
            }
            fallback_indices[fallback_ಎಣಿಕೆ++] = i;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ fi = 0; fi < fallback_ಎಣಿಕೆ; ++fi) {
            ಆದರೆ (ಎಳೆಗಳು_ಉಳಿದ <= 0) {
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ_ಸೂಚ್ಯಂಕ = fallback_indices[fi];
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ slots_ಎಡ = fallback_ಎಣಿಕೆ - fi;
            ಪೂರ್ಣಾಂಕ share = (ಎಳೆಗಳು_ಉಳಿದ + slots_ಎಡ - 1) / slots_ಎಡ;
            share     = std::min(share, ಎಳೆಗಳು_ಉಳಿದ);
            runtime[ಸ್ಥಾನ_ಸೂಚ್ಯಂಕ].assigned_ಎಳೆಗಳು = share;
            ಎಳೆಗಳು_ಉಳಿದ -= share;
        }

        ಪೂರ್ಣಾಂಕ thread_cursor = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_ಎಳೆಗಳು;
            runtime[i].thread_ಕೊನೆ = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸಕ್ರಿಯ_ಎಳೆಗಳು = thread_cursor;

        ಪೂರ್ಣಾಂಕ local_ಸ್ಥಾನ = -1;
        ಪೂರ್ಣಾಂಕ local_ith  = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            ಆದರೆ (ith_ಒಟ್ಟು >= runtime[i].thread_begin && ith_ಒಟ್ಟು < runtime[i].thread_ಕೊನೆ) {
                local_ಸ್ಥಾನ = i;
                local_ith  = ith_ಒಟ್ಟು - runtime[i].thread_begin;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಸ್ಥಿರಾಂಕ size_t k = ne00;
        ಸ್ಥಿರಾಂಕ size_t m = ne11;
        ಸ್ಥಿರಾಂಕ size_t n = ne01;

        size_t cursor = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            runtime[i].lhs_packed_ಗಾತ್ರ = runtime[i].lhs_info->packed_ಗಾತ್ರ_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_ಸ್ಥಾನಾಂತರ = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_ಸ್ಥಾನಾಂತರ_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_ಗಾತ್ರ;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_ಹೆಜ್ಜೆ = 1;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < runtime_ಎಣಿಕೆ; ++i) {
            ಆದರೆ (runtime[i].assigned_ಎಳೆಗಳು == 0) {
                ಮುಂದುವರಿ;
            }
            size_t ಮುಂದಿನ_ಹೆಜ್ಜೆ = 0;
            ಆದರೆ (!lcm_ಗಾತ್ರ(common_ಹೆಜ್ಜೆ, runtime[i].n_ಹೆಜ್ಜೆ ? runtime[i].n_ಹೆಜ್ಜೆ : 1, ಮುಂದಿನ_ಹೆಜ್ಜೆ)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            common_ಹೆಜ್ಜೆ = ಮುಂದಿನ_ಹೆಜ್ಜೆ;
        }
        GGML_ASSERT(common_ಹೆಜ್ಜೆ > 0);

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ disable_chunking = ggml_is_numa();
        ಸ್ಥಿರಾಂಕ size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ಸ್ಥಿರಾಂಕ size_t chunk_ಭಾಜಕ = (ಸಕ್ರಿಯ_ಎಳೆಗಳು == 1 || disable_chunking) ? (size_t) ಸಕ್ರಿಯ_ಎಳೆಗಳು : (size_t) ಸಕ್ರಿಯ_ಎಳೆಗಳು * chunk_multiplier;
        size_t chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = align_ಮೇಲೆ(std::max<size_t>(1, ceil_div_ಗಾತ್ರ(n, chunk_ಭಾಜಕ)), common_ಹೆಜ್ಜೆ);
        ಆದರೆ (chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ == 0) {
            chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = common_ಹೆಜ್ಜೆ;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ಸ್ಥಿರಾಂಕ size_t nchunk_ಗಾತ್ರ = std::max<size_t>(1, ceil_div_ಗಾತ್ರ(n, chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ));
        GGML_ASSERT(nchunk_ಗಾತ್ರ <= (size_t)ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchunk = (ಪೂರ್ಣಾಂಕ)nchunk_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ size_t dst_stride = dst->nb[1];

        ಸ್ವಯಂಚಾಲಿತ ಸತತ_chunk = [&](runtime_ಸ್ಥಾನ & ಸ್ಥಾನ, size_t global_ಆರಂಭ, size_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, uint8_t * dst_batch_ತಳ) {
            ಸ್ಥಿರಾಂಕ size_t rhs_packed_ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನ.kernel->get_rhs_packed_ಸ್ಥಾನಾಂತರ_ex(global_ಆರಂಭ, k, ಸ್ಥಾನ.rhs_bl);
            ಸ್ಥಿರಾಂಕ size_t dst_ಸ್ಥಾನಾಂತರ        = ಸ್ಥಾನ.kernel->get_dst_ಸ್ಥಾನಾಂತರ(0, global_ಆರಂಭ, dst_stride);

            ಸ್ಥಿರಾಂಕ uint8_t * lhs_ptr = scratch + ಸ್ಥಾನ.lhs_ಸ್ಥಾನಾಂತರ + ಸ್ಥಾನ.lhs_packed_offset0;
            ಸ್ಥಿರಾಂಕ uint8_t * rhs_ptr = ಸ್ಥಾನ.rhs_ತಳ + rhs_packed_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ * dst_ptr         = reinterpret_cast<ದಶಮಾಂಶ *>(dst_batch_ತಳ + dst_ಸ್ಥಾನಾಂತರ);

            ಸ್ಥಾನ.kernel->ಸತತ_kernel_ex(m, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, k, ಸ್ಥಾನ.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       ಗಾತ್ರ(ದಶಮಾಂಶ),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        ಗಾಗಿ (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ಸ್ಥಿರಾಂಕ uint8_t * lhs_batch_ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(src1->ದತ್ತಾಂಶ) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_ತಳ = static_cast<uint8_t *>(dst->ದತ್ತಾಂಶ) + batch_idx * dst->nb[2];

            ಆದರೆ (local_ಸ್ಥಾನ != -1) {
                runtime_ಸ್ಥಾನ & ಸ್ಥಾನ = runtime[local_ಸ್ಥಾನ];
                ಸ್ಥಿರಾಂಕ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)ಸ್ಥಾನ.mr);
                int64_t max_ಎಳೆಗಳು = ಸ್ಥಾನ.mr ? (m_roundup_mr / (int64_t)ಸ್ಥಾನ.mr) : ಸ್ಥಾನ.assigned_ಎಳೆಗಳು;
                max_ಎಳೆಗಳು = std::max<int64_t>(1, max_ಎಳೆಗಳು);
                ಸ್ಥಿರಾಂಕ int64_t use_ಎಳೆಗಳು = std::min<int64_t>(ಸ್ಥಾನ.assigned_ಎಳೆಗಳು, max_ಎಳೆಗಳು);

                ಆದರೆ (local_ith < use_ಎಳೆಗಳು) {
                    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_m_ಪ್ರತಿ_thread0   = ಸುತ್ತು_ಕೆಳಕ್ಕೆ((size_t)(m_roundup_mr / use_ಎಳೆಗಳು), ಸ್ಥಾನ.mr);
                    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_m_ಪ್ರತಿ_thread_N_1 = (int64_t)m - (use_ಎಳೆಗಳು - 1) * ಅಂಕೆ_m_ಪ್ರತಿ_thread0;

                    ಸ್ಥಿರಾಂಕ int64_t m_ಆರಂಭ = (int64_t)local_ith * ಅಂಕೆ_m_ಪ್ರತಿ_thread0;
                    ಸ್ಥಿರಾಂಕ int64_t m_ಎಣಿಕೆ = (local_ith == use_ಎಳೆಗಳು - 1) ? ಅಂಕೆ_m_ಪ್ರತಿ_thread_N_1 : ಅಂಕೆ_m_ಪ್ರತಿ_thread0;

                    ಸ್ಥಿರಾಂಕ size_t ತಳ_packed_off  = ಸ್ಥಾನ.lhs_info->get_packed_ಸ್ಥಾನಾಂತರ_ex(m_ಆರಂಭ, k, ಸ್ಥಾನ.lhs_bl, ಸ್ಥಾನ.mr, ಸ್ಥಾನ.kr, ಸ್ಥಾನ.sr);
                    ಸ್ಥಿರಾಂಕ size_t ಮುಂದಿನ_ಖಂಡ_off   = ಸ್ಥಾನ.lhs_info->get_packed_ಸ್ಥಾನಾಂತರ_ex(m_ಆರಂಭ + ಸ್ಥಾನ.mr, k, ಸ್ಥಾನ.lhs_bl, ಸ್ಥಾನ.mr, ಸ್ಥಾನ.kr, ಸ್ಥಾನ.sr);
                    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride_ಬೈಟುಗಳು = ಸ್ಥಾನ.mr ? (ಮುಂದಿನ_ಖಂಡ_off - ತಳ_packed_off) / ಸ್ಥಾನ.mr : 0;

                    int64_t ಉಳಿದ = m_ಎಣಿಕೆ;
                    int64_t cur       = m_ಆರಂಭ;

                    uint8_t * lhs_packed = scratch + ಸ್ಥಾನ.lhs_ಸ್ಥಾನಾಂತರ;
                    ಆಗ (ಉಳಿದ > 0) {
                        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಒಳ_ಗುಂಪು = cur;
                        ಸ್ಥಿರಾಂಕ int64_t avail        = (int64_t)m - ಅಡ್ಡಸಾಲು_ಒಳ_ಗುಂಪು;
                        ಸ್ಥಿರಾಂಕ int64_t take         = std::min(avail, ಉಳಿದ);

                        ಸ್ಥಿರಾಂಕ size_t src_off = ಸ್ಥಾನ.lhs_info->get_ಸ್ಥಾನಾಂತರ(ಅಡ್ಡಸಾಲು_ಒಳ_ಗುಂಪು, src1->nb[1]);
                        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src_ptr = lhs_batch_ತಳ + src_off;
                        ಸ್ಥಿರಾಂಕ size_t dst_off = ತಳ_packed_off + (size_t)(cur - m_ಆರಂಭ) * ಅಡ್ಡಸಾಲು_stride_ಬೈಟುಗಳು;
                        ಶೂನ್ಯ * dst_ptr       = lhs_packed + dst_off;

                        ಸ್ಥಾನ.lhs_info->pack_func_ex(take, k, ಸ್ಥಾನ.lhs_bl, ಸ್ಥಾನ.mr, ಸ್ಥಾನ.kr, ಸ್ಥಾನ.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        ಉಳಿದ -= take;
                    }
                }
            }

            ಆದರೆ (ith_ಒಟ್ಟು == 0) {
                ggml_threadpool_chunk_ಗಣ(params->threadpool, ಸಕ್ರಿಯ_ಎಳೆಗಳು);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            ಆದರೆ (local_ಸ್ಥಾನ != -1) {
                runtime_ಸ್ಥಾನ & ಸ್ಥಾನ = runtime[local_ಸ್ಥಾನ];
                ಪೂರ್ಣಾಂಕ current_chunk = ith_ಒಟ್ಟು;
                ಆಗ (current_chunk < nchunk) {
                    ಸ್ಥಿರಾಂಕ size_t global_ಆರಂಭ = (size_t)current_chunk * chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                    ಆದರೆ (global_ಆರಂಭ >= n) {
                        ನಿಲ್ಲಿಸು;
                    }

                    ಸ್ಥಿರಾಂಕ size_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 = std::min(chunk_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, n - global_ಆರಂಭ);
                    ಆದರೆ (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        ಸತತ_chunk(ಸ್ಥಾನ, global_ಆರಂಭ, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, dst_batch_ತಳ);
                    }

                    current_chunk = ggml_threadpool_chunk_ಕೂಡಿಸು(params->threadpool, 1);
                }
            }

            ಆದರೆ (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

ಸಾರ್ವಜನಿಕ:
    ಪೂರ್ಣಾಂಕ repack(ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ಸ್ಥಿರಾಂಕ size_t n = tensor->ne[1];
        ಸ್ಥಿರಾಂಕ size_t k = tensor->ne[0];

        kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ * ಶೀರ್ಷಿಕೆ = kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ_from_ptr(tensor->ದತ್ತಾಂಶ);
        ಆದರೆ (!ಶೀರ್ಷಿಕೆ) {
            ಹಿಂತಿರುಗಿ -1;
        }

        ಶೀರ್ಷಿಕೆ->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        ಶೀರ್ಷಿಕೆ->ಆವೃತ್ತಿ_2    = GGML_KLEIDIAI_PACK_ಆವೃತ್ತಿ;
        ಶೀರ್ಷಿಕೆ->ಸ್ಥಾನ_ಎಣಿಕೆ = 0;

        uint8_t * ತಳ_ptr = static_cast<uint8_t *>(tensor->ದತ್ತಾಂಶ);
        size_t cursor = ಗಾತ್ರ(kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ);
        cursor = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ want_f32 = tensor->type == GGML_TYPE_F32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ_ಒಟ್ಟು = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<ದಶಮಾಂಶ>  scales;
        std::vector<ದಶಮಾಂಶ>  bias;
        std::vector<ದಶಮಾಂಶ> ಅಡ್ಡಸಾಲು_f32;

        ಆದರೆ (want_q8 && ಸ್ಥಾನ_ಒಟ್ಟು > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = tensor->nb[1];
            ಸ್ಥಿರಾಂಕ size_t k_ಖಂಡಗಳು   = (k + QK8_0 - 1) / QK8_0;

            ಆದರೆ (tensor->type == GGML_TYPE_Q6_K) {
                ಆದರೆ (k % QK_K != 0 || k % QK8_0 != 0) {
                    ಹಿಂತಿರುಗಿ -1;
                }
                ಅಡ್ಡಸಾಲು_f32.resize(k);
            }

            ಗಾಗಿ (size_t ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < n; ++ಅಡ್ಡಸಾಲು) {
                ಸ್ಥಿರಾಂಕ uint8_t * ಅಡ್ಡಸಾಲು_ದತ್ತಾಂಶ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(ದತ್ತಾಂಶ) + ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_stride;

                ಆದರೆ (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_ಅಡ್ಡಸಾಲು_q6_K(reinterpret_cast<ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K *>(ಅಡ್ಡಸಾಲು_ದತ್ತಾಂಶ), ಅಡ್ಡಸಾಲು_f32.ದತ್ತಾಂಶ(), k);
                }

                ದಶಮಾಂಶ max_abs = 0.0f;
                ಆದರೆ (tensor->type == GGML_TYPE_Q6_K) {
                    ಗಾಗಿ (size_t ರೇಖೀಯ_idx = 0; ರೇಖೀಯ_idx < k; ++ರೇಖೀಯ_idx) {
                        max_abs = std::max(max_abs, std::fabs(ಅಡ್ಡಸಾಲು_f32[ರೇಖೀಯ_idx]));
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * ಅಡ್ಡಸಾಲು_ಖಂಡಗಳು = reinterpret_cast<ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *>(ಅಡ್ಡಸಾಲು_ದತ್ತಾಂಶ);
                    ಗಾಗಿ (size_t ಖಂಡ = 0; ಖಂಡ < k_ಖಂಡಗಳು; ++ಖಂಡ) {
                        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 & blk = ಅಡ್ಡಸಾಲು_ಖಂಡಗಳು[ಖಂಡ];
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_FP16_ಗಮ್ಯ_FP32(blk.d);
                        ಗಾಗಿ (size_t l = 0; l < QK8_0; ++l) {
                            ಸ್ಥಿರಾಂಕ size_t ರೇಖೀಯ_idx = ಖಂಡ * QK8_0 + l;
                            ಆದರೆ (ರೇಖೀಯ_idx >= k) {
                                ನಿಲ್ಲಿಸು;
                            }
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ value = d * static_cast<ದಶಮಾಂಶ>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                ದಶಮಾಂಶ scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[ಅಡ್ಡಸಾಲು] = scale;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                ಆದರೆ (tensor->type == GGML_TYPE_Q6_K) {
                    ಗಾಗಿ (size_t ರೇಖೀಯ_idx = 0; ರೇಖೀಯ_idx < k; ++ರೇಖೀಯ_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(ಅಡ್ಡಸಾಲು_f32[ರೇಖೀಯ_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[ಅಡ್ಡಸಾಲು * k + ರೇಖೀಯ_idx] = static_cast<int8_t>(q);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * ಅಡ್ಡಸಾಲು_ಖಂಡಗಳು = reinterpret_cast<ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *>(ಅಡ್ಡಸಾಲು_ದತ್ತಾಂಶ);
                    ಗಾಗಿ (size_t ಖಂಡ = 0; ಖಂಡ < k_ಖಂಡಗಳು; ++ಖಂಡ) {
                        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 & blk = ಅಡ್ಡಸಾಲು_ಖಂಡಗಳು[ಖಂಡ];
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_FP16_ಗಮ್ಯ_FP32(blk.d);
                        ಗಾಗಿ (size_t l = 0; l < QK8_0; ++l) {
                            ಸ್ಥಿರಾಂಕ size_t ರೇಖೀಯ_idx = ಖಂಡ * QK8_0 + l;
                            ಆದರೆ (ರೇಖೀಯ_idx >= k) {
                                ನಿಲ್ಲಿಸು;
                            }
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ value = d * static_cast<ದಶಮಾಂಶ>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[ಅಡ್ಡಸಾಲು * k + ರೇಖೀಯ_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        ಆದರೆ (want_f32 && ಸ್ಥಾನ_ಒಟ್ಟು > 0) {
            bias.resize(n, 0.0f);
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = 0; ಸ್ಥಾನ < ಸ್ಥಾನ_ಒಟ್ಟು && ಸ್ಥಾನ < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++ಸ್ಥಾನ) {
            ಆದರೆ (!allow_fallback && ಸ್ಥಾನ > 0) {
                ನಿಲ್ಲಿಸು;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[ಸ್ಥಾನ];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            ಆದರೆ (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_ಗಾತ್ರ_ex || !kernel) {
                ಮುಂದುವರಿ;
            }

            ಸ್ಥಿರಾಂಕ size_t nr = kernel->get_nr();
            ಸ್ಥಿರಾಂಕ size_t kr = kernel->get_kr();
            ಸ್ಥಿರಾಂಕ size_t sr = kernel->get_sr();
            ಸ್ಥಿರಾಂಕ ggml_type rhs_type = kernels->rhs_type;
            ಆದರೆ (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            ಸ್ಥಿರಾಂಕ size_t ಖಂಡ_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            ಆದರೆ (ಖಂಡ_len == SIZE_MAX) {
                ಮುಂದುವರಿ;
            }

            ಸ್ಥಿರಾಂಕ size_t packed_ಗಾತ್ರ = rhs_info->packed_ಗಾತ್ರ_ex(n, k, nr, kr, ಖಂಡ_len);
            ಸ್ಥಿರಾಂಕ size_t aligned_cursor = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = ತಳ_ptr + aligned_cursor;

            ಆದರೆ (rhs_type == GGML_TYPE_Q4_0) {
                ರಚನೆ kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_ಬಿಂದು = 1;
                params.rhs_zero_ಬಿಂದು = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(ದತ್ತಾಂಶ), ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ,
                                       dst_ptr, 0, &params);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       ದತ್ತಾಂಶ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst_ptr, 0, ಶೂನ್ಯಸೂಚಕ);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (rhs_type == GGML_TYPE_Q8_0) {
                ರಚನೆ kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_ಬಿಂದು = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.ದತ್ತಾಂಶ(), ಶೂನ್ಯಸೂಚಕ, scales.ದತ್ತಾಂಶ(),
                                       dst_ptr, 0, &params);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       ದತ್ತಾಂಶ, bias.ದತ್ತಾಂಶ(), ಶೂನ್ಯಸೂಚಕ,
                                       dst_ptr, 0, ಶೂನ್ಯಸೂಚಕ);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಮುಂದುವರಿ;
            }

            ಶೀರ್ಷಿಕೆ->offsets[ಶೀರ್ಷಿಕೆ->ಸ್ಥಾನ_ಎಣಿಕೆ] = aligned_cursor;
            ಶೀರ್ಷಿಕೆ->sizes[ಶೀರ್ಷಿಕೆ->ಸ್ಥಾನ_ಎಣಿಕೆ]   = packed_ಗಾತ್ರ;
            ++ಶೀರ್ಷಿಕೆ->ಸ್ಥಾನ_ಎಣಿಕೆ;

            cursor = aligned_cursor + packed_ಗಾತ್ರ;
        }

        ಆದರೆ (ಶೀರ್ಷಿಕೆ->ಸ್ಥಾನ_ಎಣಿಕೆ == 0) {
            ಶೀರ್ಷಿಕೆ->magic   = 0;
            ಶೀರ್ಷಿಕೆ->ಆವೃತ್ತಿ_2 = 0;
            ಸ್ಮರಣೆ_ನಕಲು(tensor->ದತ್ತಾಂಶ, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
        }

        ಹಿಂತಿರುಗಿ 0;
    }
};

ಸ್ಥಿರ ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_ಬಫರ್_t, ರಚನೆ ggml_tensor *) {
    ಸ್ಥಿರ tensor_traits traits;
    ಹಿಂತಿರುಗಿ &traits;
}
}  // namespace ggml::cpu::kleidiai

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_cpu_kleidiai_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ರಚನೆ ggml_tensor * tensor) {
    tensor->extra = (ಶೂನ್ಯ *) ggml::cpu::kleidiai::get_tensor_traits(ಬಫರ್, tensor);

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_kleidiai_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ರಚನೆ ggml_tensor * tensor,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ಸ್ವಯಂಚಾಲಿತ tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    ಸ್ವಯಂಚಾಲಿತ OK            = tensor_traits->repack(tensor, ದತ್ತಾಂಶ, size);

    GGML_ASSERT(OK == 0);
    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cpu_kleidiai_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ಬಳಸದ(buft);
    ಹಿಂತಿರುಗಿ "CPU_KLEIDIAI";
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_cpu_kleidiai_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_buft_alloc_ಬಫರ್(ggml_backend_cpu_ಬಫರ್_type(), size);

    ಆದರೆ (ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಬಫರ್->buft              = buft;
    ಬಫರ್->iface.init_tensor = ggml_backend_cpu_kleidiai_ಬಫರ್_init_tensor;
    ಬಫರ್->iface.ಗಣ_tensor  = ggml_backend_cpu_kleidiai_ಬಫರ್_ಗಣ_tensor;
    ಬಫರ್->iface.get_tensor  = ಶೂನ್ಯಸೂಚಕ;
    ಬಫರ್->iface.cpy_tensor  = ಶೂನ್ಯಸೂಚಕ;
    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

ಸ್ಥಿರ size_t ggml_backend_cpu_kleidiai_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ಬಳಸದ(buft);
    ಹಿಂತಿರುಗಿ TENSOR_ALIGNMENT;
}

ಸ್ಥಿರ size_t ggml_backend_cpu_kleidiai_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    GGML_ಬಳಸದ(buft);

    ಆದರೆ (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ggml_nbytes(tensor);
    }

    ಸ್ಥಿರಾಂಕ size_t n = tensor->ne[1];
    ಸ್ಥಿರಾಂಕ size_t k = tensor->ne[0];

    size_t cursor = ಗಾತ್ರ(kleidiai_ತೂಕ_ಶೀರ್ಷಿಕೆ);
    cursor = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ want_f32 = tensor->type == GGML_TYPE_F32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ_ಒಟ್ಟು = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ allow_fallback = kleidiai_pack_fallback_allowed();

    size_t ಸ್ಥಾನ_ಎಣಿಕೆ = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = 0; ಸ್ಥಾನ < ಸ್ಥಾನ_ಒಟ್ಟು; ++ಸ್ಥಾನ) {
        ಆದರೆ (!allow_fallback && ಸ್ಥಾನ > 0) {
            ನಿಲ್ಲಿಸು;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[ಸ್ಥಾನ];
        ಆದರೆ (!kernels) {
            ಮುಂದುವರಿ;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        ಆದರೆ (!kernel || !rhs_info || !rhs_info->packed_ಗಾತ್ರ_ex) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ggml_type rhs_type = kernels->rhs_type;
        ಸ್ಥಿರಾಂಕ size_t ಖಂಡ_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        ಆದರೆ (ಖಂಡ_len == SIZE_MAX) {
            ಮುಂದುವರಿ;
        }

        cursor = align_ಮೇಲೆ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_ಗಾತ್ರ_ex(n, k, kernel->get_nr(), kernel->get_kr(), ಖಂಡ_len);
        ++ಸ್ಥಾನ_ಎಣಿಕೆ;
    }

    ಆದರೆ (ಸ್ಥಾನ_ಎಣಿಕೆ == 0) {
        ಹಿಂತಿರುಗಿ ggml_nbytes(tensor);
    }

    ಹಿಂತಿರುಗಿ std::max(cursor, ggml_nbytes(tensor));
}

ನಾಮಾಕಾಶ ggml::cpu::kleidiai {
ವರ್ಗ extra_ಬಫರ್_type : ggml::cpu::extra_ಬಫರ್_type {
    ತಾರ್ಕಿಕ supports_op(ggml_backend_dev_t, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        ಆದರೆ (op->op != GGML_OP_MUL_MAT) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ_ಒಟ್ಟು = kleidiai_collect_kernel_chain(op, kernel_chain);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src0_is_kleidiai =
            op->src[0]->ಬಫರ್ &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->ಬಫರ್->buft->context == ಇದು &&
            ಸ್ಥಾನ_ಒಟ್ಟು > 0;

        ಆದರೆ ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            ಆದರೆ (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == ಶೂನ್ಯಸೂಚಕ) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == ಶೂನ್ಯಸೂಚಕ) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == ಶೂನ್ಯಸೂಚಕ) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == ಶೂನ್ಯಸೂಚಕ) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == ಶೂನ್ಯಸೂಚಕ) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (op->src[1]->ಬಫರ್ && !ggml_backend_buft_is_host(op->src[1]->ಬಫರ್->buft)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆದರೆ (op->src[0]->type == GGML_TYPE_Q4_K) {
                ಆದರೆ (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಹಿಂತಿರುಗಿ ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ));
            }

            ಆದರೆ (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                ಆದರೆ (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆದರೆ (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) override {
        ಆದರೆ (op->op == GGML_OP_MUL_MAT) {
            ಆದರೆ (op->src[0]->ಬಫರ್ && op->src[0]->ಬಫರ್->buft->context == ಇದು) {
                ಹಿಂತಿರುಗಿ (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } ಇಲ್ಲದಿದ್ದರೆ {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                ಆದರೆ (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ> warned(ಸುಳ್ಳು);
                    ಆದರೆ (!warned.exchange(ನಿಜ)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                ಆದರೆ (op->src[0]->type != GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ_ಒಟ್ಟು = kleidiai_collect_kernel_chain(op, kernel_chain);
                ಆದರೆ (ಸ್ಥಾನ_ಒಟ್ಟು > 0 && op->src[1]->ne[1] > 1) {
                    ಆದರೆ ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
                    }
                    ಹಿಂತಿರುಗಿ ggml::cpu::kleidiai::get_tensor_traits(ಖಾಲಿ, ಖಾಲಿ);
                }
            }
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_ಬಫರ್_type_t ggml_backend_cpu_kleidiai_ಬಫರ್_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml::cpu::kleidiai::extra_ಬಫರ್_type ctx;
    ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_type ggml_backend_cpu_ಬಫರ್_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_ಬಫರ್_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_ಬಫರ್_type_alloc_ಬಫರ್,
                           /* .alloc_buffer_n       = */ ಶೂನ್ಯಸೂಚಕ,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_ಬಫರ್_type_get_alignment,
                           /* .get_max_size         = */ ಶೂನ್ಯಸೂಚಕ,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
                           /* .get_alloc_size_n     = */ ಖಾಲಿ,
                           /* .is_host              = */ ಶೂನ್ಯಸೂಚಕ,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    ಹಿಂತಿರುಗಿ &ggml_backend_cpu_ಬಫರ್_type_kleidiai;
}
