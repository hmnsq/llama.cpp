// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#ਅਨ੍ਤਰ੍ਭਾਵ <arm_neon.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>
#ਅਨ੍ਤਰ੍ਭਾਵ <cfloat>
#ਅਨ੍ਤਰ੍ਭਾਵ <cctype>
#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdexcept>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <array>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <climits>
#ਅਨ੍ਤਰ੍ਭਾਵ <charconv>
#ਅਨ੍ਤਰ੍ਭਾਵ <system_error>
#ਜੇ defined(__linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <asm/hwcap.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <dirent.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/auxv.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/stat.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਜਾਂਜੇ defined(__APPLE__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/sysctl.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/types.h>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ "kleidiai.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-feats.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-threading.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "traits.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "kernels.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "kai/kai_common.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_CPP
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

ਸਥਿਰ constexpr ਪੂਰਨ      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
ਸਥਿਰ constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
ਸਥਿਰ constexpr uint16_t GGML_KLEIDIAI_PACK_ਸੰਸਕਰਣ     = 1;
ਸਥਿਰ constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
ਸਥਿਰ constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

ਬਣਤਰ ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    ਪੂਰਨ sme_thread_cap; // <= 0 means "SME disabled/unknown"
    ਪੂਰਨ thread_hint;    // <= 0 means "no hint"
    ਪੂਰਨ chunk_multiplier;
} ਸਥਿਰ ctx = { CPU_FEATURE_NONE, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, 0, -1, 4 };

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ is_sme_family(cpu_feature f) {
    ਵਾਪਸ (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ* cpu_feature_ਮੰਜ਼ਿਲ_string(cpu_feature f) {
    ਜੇ (f == CPU_FEATURE_NONE) {
        ਵਾਪਸ "NONE";
    } ਨਹੀਂਤਾਂ ਜੇ ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        ਵਾਪਸ "SME2";
    } ਨਹੀਂਤਾਂ ਜੇ ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        ਵਾਪਸ "SME";
    } ਨਹੀਂਤਾਂ ਜੇ ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        ਵਾਪਸ "SVE";
    }
    ਨਹੀਂਤਾਂ ਜੇ ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        ਵਾਪਸ "I8MM";
    } ਨਹੀਂਤਾਂ ਜੇ ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        ਵਾਪਸ "DOTPROD";
    }
    ਨਹੀਂਤਾਂ {
        ਵਾਪਸ "UNKNOWN";
    }
}

#ਜੇ defined(__linux__) && defined(__aarch64__)
ਸਥਿਰ ਤਾਰਕਿਕ parse_cpu_dir_name(ਅਚਲ ਅੱਖਰ* name, size_t* cpu) {
    ਜੇ (ਲੜੀ_ਤੁਲਨਾ(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ* ਪਹਿਲਾ = name + 3;
    ਅਚਲ ਅੱਖਰ* ਆਖਰੀ = name + ਲੜੀ_ਲੰਬਾਈ(name);

    size_t value = 0;
    ਅਚਲ ਸਵੈਚਾਲਿਤ [end, ec] = std::from_chars(ਪਹਿਲਾ, ਆਖਰੀ, value, 10);

    ਜੇ (ec != std::errc{} || end != ਆਖਰੀ) {
        ਵਾਪਸ ਝੂਠ;
    }

    *cpu = value;
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    ਜੇ (dir == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ cpus;
    }

    ਜਦੋਂ (dirent * ਇੰਦਰਾਜ = readdir(dir)) {
        size_t cpu = 0;
        ਜੇ (parse_cpu_dir_name(ਇੰਦਰਾਜ->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    ਵਾਪਸ cpus;
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__APPLE__) && defined(__aarch64__)
ਸਥਿਰ ਤਾਰਕਿਕ apple_sme_counted_perf_level(std::string name) {
    ਲਈ (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (ਅੱਖਰ) std::tolower((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    ਵਾਪਸ name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਖਾਲੀ ਜੋੜੋ_smcus_from_smidr(uint64_t smidr, size_t & ਅੰਕ_ਨਿੱਜੀ, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ਅਚਲ uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ਅਚਲ uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ਅਚਲ size_t shared_ਗਿਣਤੀ = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ਅਚਲ uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ਅਚਲ uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ਅਚਲ uint32_t id = (affinity2 << 12) | affinity;

    ਜੇ (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    ਚੋਣ (sh) {
        ਹਾਲਤ 2: // private SMCU
            ++ਅੰਕ_ਨਿੱਜੀ;
            ਰੋਕੋ;
        ਹਾਲਤ 3: // shared SMCU
            ਜੇ (shared_counts[id] < shared_ਗਿਣਤੀ) {
                shared_counts[id] = shared_ਗਿਣਤੀ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ 0:
            ਜੇ (id == 0) {
                ++ਅੰਕ_ਨਿੱਜੀ;
            } ਨਹੀਂਤਾਂ ਜੇ (shared_counts[id] < shared_ਗਿਣਤੀ) {
                shared_counts[id] = shared_ਗਿਣਤੀ;
            }
            ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
    }
}

ਸਥਿਰ size_t detect_ਅੰਕ_smcus() {
    ਅਚਲ ਸਵੈਚਾਲਿਤ runtime_feat = ggml_feats_get_arch64_runtime();
    ਜੇ (!runtime_feat.has_sme) {
        ਵਾਪਸ 0;
    }

#ਜੇ defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t ਅੰਕ_ਨਿੱਜੀ = 0;
    std::map<uint32_t, size_t> shared_counts;

    ਅਚਲ std::vector<size_t> cpus = detect_cpu_ids();
    ਲਈ (ਅਚਲ size_t cpu : cpus) {
        ਅਚਲ std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        ਜੇ (!file.is_open()) {
            ਜਾਰੀ;
        }

        uint64_t smidr = 0;
        ਜੇ (!(file >> std::hex >> smidr)) {
            ਜਾਰੀ;
        }

        ਜੋੜੋ_smcus_from_smidr(smidr, ਅੰਕ_ਨਿੱਜੀ, shared_counts);
    }

    size_t ਕੁੱਲ = ਅੰਕ_ਨਿੱਜੀ;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਇੰਦਰਾਜ : shared_counts) {
        ਕੁੱਲ += ਇੰਦਰਾਜ.second;
    }
    ਵਾਪਸ ਕੁੱਲ;

#ਜਾਂਜੇ defined(__APPLE__) && defined(__aarch64__)
    ਪੂਰਨ perf_ਪੱਧਰ = 0;
    size_t size = ਆਕਾਰ(perf_ਪੱਧਰ);
    ਜੇ (sysctlbyname("hw.nperflevels", &perf_ਪੱਧਰ, &size, ਖਾਲੀ_ਸੂਚਕ, 0) != 0 ||
        size != ਆਕਾਰ(perf_ਪੱਧਰ) || perf_ਪੱਧਰ <= 0) {
        ਵਾਪਸ 0;
    }

    size_t units = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < perf_ਪੱਧਰ; ++i) {
        ਅੱਖਰ ਕੁੰਜੀ[64] = {};
        ਪੂਰਨ physical_cpus = 0;
        ਪੂਰਨ cpus_ਪ੍ਰਤੀ_l2 = 0;

        snprintf(ਕੁੰਜੀ, ਆਕਾਰ(ਕੁੰਜੀ), "hw.perflevel%d.physicalcpu", i);
        size = ਆਕਾਰ(physical_cpus);
        ਜੇ (sysctlbyname(ਕੁੰਜੀ, &physical_cpus, &size, ਖਾਲੀ_ਸੂਚਕ, 0) != 0 ||
            size != ਆਕਾਰ(physical_cpus) || physical_cpus <= 0) {
            ਜਾਰੀ;
        }

        snprintf(ਕੁੰਜੀ, ਆਕਾਰ(ਕੁੰਜੀ), "hw.perflevel%d.cpusperl2", i);
        size = ਆਕਾਰ(cpus_ਪ੍ਰਤੀ_l2);
        ਜੇ (sysctlbyname(ਕੁੰਜੀ, &cpus_ਪ੍ਰਤੀ_l2, &size, ਖਾਲੀ_ਸੂਚਕ, 0) != 0 ||
            size != ਆਕਾਰ(cpus_ਪ੍ਰਤੀ_l2) || cpus_ਪ੍ਰਤੀ_l2 <= 0) {
            ਜਾਰੀ;
        }

        snprintf(ਕੁੰਜੀ, ਆਕਾਰ(ਕੁੰਜੀ), "hw.perflevel%d.name", i);
        size = 0;
        ਜੇ (sysctlbyname(ਕੁੰਜੀ, ਖਾਲੀ_ਸੂਚਕ, &size, ਖਾਲੀ_ਸੂਚਕ, 0) != 0 || size == 0) {
            ਜਾਰੀ;
        }

        std::string name(size, '\0');
        ਜੇ (sysctlbyname(ਕੁੰਜੀ, &name[0], &size, ਖਾਲੀ_ਸੂਚਕ, 0) != 0) {
            ਜਾਰੀ;
        }
        name.resize(size);
        ਜਦੋਂ (!name.ਖਾਲੀ_ਥਾਵਾਂ() && name.back() == '\0') {
            name.pop_back();
        }

        ਜੇ (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_ਪ੍ਰਤੀ_l2 - 1) / cpus_ਪ੍ਰਤੀ_l2);
        }
    }

    ਵਾਪਸ units;

#ਜਾਂਜੇ defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    ਵਾਪਸ 0;

#ਨਹੀਂਤਾਂ
    ਵਾਪਸ 0;
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ਪੂਰਨ parse_uint_env(ਅਚਲ ਅੱਖਰ *s, ਅਚਲ ਅੱਖਰ *name, ਤਾਰਕਿਕ *ok) {
    ਜੇ (!s) { *ok = ਝੂਠ; ਵਾਪਸ 0; }
    ਅੱਖਰ *end = ਖਾਲੀ_ਸੂਚਕ;
    ਲੰਮਾ v = strtol(s, &end, 10);
    ਜੇ (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = ਝੂਠ;
        ਵਾਪਸ 0;
    }
    ਜੇ (v < 0 || v > ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = ਝੂਠ;
        ਵਾਪਸ 0;
    }
    *ok = ਸੱਚ;
    ਵਾਪਸ (ਪੂਰਨ)v;
}

ਸਥਿਰ ਖਾਲੀ init_kleidiai_context(ਖਾਲੀ) {
    ggml_critical_section_ਸ਼ੁਰੂ();
    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;

    ਜੇ (!initialized) {
        initialized = ਸੱਚ;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ਅਚਲ ਅੱਖਰ *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ਅਚਲ ਅੱਖਰ *env_ਧਾਗੇ     = getenv("GGML_TOTAL_THREADS");
        ਅਚਲ ਅੱਖਰ *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ਅਚਲ ਸਵੈਚਾਲਿਤ runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        ਜੇ (env_ਧਾਗੇ) {
            ਤਾਰਕਿਕ ok = ਝੂਠ;
            ਪੂਰਨ hint = parse_uint_env(env_ਧਾਗੇ, "GGML_TOTAL_THREADS", &ok);
            ਜੇ (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        ਜੇ (env_chunk_mult) {
            ਤਾਰਕਿਕ ok = ਝੂਠ;
            ਪੂਰਨ multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            ਜੇ (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        ਪੂਰਨ sme_cores = 0;
        ਤਾਰਕਿਕ sme_env_ok = ਝੂਠ;
        ਤਾਰਕਿਕ sme_env_ਸੈੱਟ = (env_sme != ਖਾਲੀ_ਸੂਚਕ);

        ਅਚਲ ਤਾਰਕਿਕ has_supported_sme_family = runtime_feat.has_sme;
        ਤਾਰਕਿਕ sme_cap_detected = ਝੂਠ;

        ਜੇ (has_supported_sme_family) {
            detected_smcus = detect_ਅੰਕ_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (ਪੂਰਨ)detected_smcus : 1;

            ਜੇ (!sme_env_ਸੈੱਟ && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        ਜੇ (sme_env_ਸੈੱਟ) {
            ਤਾਰਕਿਕ ok = ਝੂਠ;
            ਪੂਰਨ v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            ਜੇ (ok) {
                ਜੇ (has_supported_sme_family) {
                    sme_cores = v;
                } ਨਹੀਂਤਾਂ {
                    ਜੇ (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } ਨਹੀਂਤਾਂ {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        ਜੇ (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            ਜੇ (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        ਜੇ (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (ਪੂਰਨ)ctx.features);
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_ਮੰਜ਼ਿਲ_string(ctx.kernels_q4->required_cpu));
        }

        ਜੇ (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (ਪੂਰਨ)ctx.features);
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_ਮੰਜ਼ਿਲ_string(ctx.kernels_q8->required_cpu));
        }

        ਜੇ (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (ਪੂਰਨ) ctx.features);
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_ਮੰਜ਼ਿਲ_string(ctx.kernels_q4k->required_cpu));
        }

        ਜੇ (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (ਪੂਰਨ)ctx.features);
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_ਮੰਜ਼ਿਲ_string(ctx.kernels_f32->required_cpu));
        }

        ਅਚਲ ਤਾਰਕਿਕ has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        ਜੇ (has_selected_sme_family_kernel) {
            ਜੇ (sme_env_ਸੈੱਟ && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } ਨਹੀਂਤਾਂ ਜੇ (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } ਨਹੀਂਤਾਂ {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_ਸਿਰਾ();
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ kleidiai_sme_thread_cap() {
    ਵਾਪਸ ctx.sme_thread_cap;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t align_ਉੱਪਰ(size_t value, size_t alignment) {
    ਜੇ (alignment == 0) {
        ਵਾਪਸ value;
    }
    ਅਚਲ size_t remainder = value % alignment;
    ਵਾਪਸ remainder == 0 ? value : value + (alignment - remainder);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_ਮਾਪ(size_t a, size_t b) {
    ਜਦੋਂ (b != 0) {
        ਅਚਲ size_t t = a % b;
        a = b;
        b = t;
    }
    ਵਾਪਸ a;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ lcm_ਮਾਪ(size_t a, size_t b, size_t & ਨਤੀਜਾ) {
    ਜੇ (a == 0 || b == 0) {
        ਨਤੀਜਾ = 0;
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ size_t g = ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_ਮਾਪ(a, b);
    ਅਚਲ size_t q = a / g;
    ਜੇ (q > SIZE_MAX / b) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਨਤੀਜਾ = q * b;
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t ceil_div_ਮਾਪ(size_t a, size_t b) {
    ਵਾਪਸ b == 0 ? 0 : (a + b - 1) / b;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t kleidiai_chunk_ਖਾਨੇ_ਗਿਣਤੀ(size_t n, ਪੂਰਨ nth_ਕੁੱਲ, ਤਾਰਕਿਕ disable_chunking, size_t n_ਕਦਮ) {
    ਅਚਲ size_t multiplier = (nth_ਕੁੱਲ == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ਅਚਲ size_t ਭਾਜਕ = std::max<size_t>(1, (size_t) nth_ਕੁੱਲ * multiplier);
    ਅਚਲ size_t chunk_ਖਾਨੇ_ਗਿਣਤੀ = align_ਉੱਪਰ(std::max<size_t>(1, ceil_div_ਮਾਪ(n, ਭਾਜਕ)), n_ਕਦਮ);
    ਵਾਪਸ chunk_ਖਾਨੇ_ਗਿਣਤੀ ? chunk_ਖਾਨੇ_ਗਿਣਤੀ : n_ਕਦਮ;
}

ਬਣਤਰ kleidiai_ਖੰਡ_ਦਲੀਲਾਂ {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ kleidiai_ਖੰਡ_ਦਲੀਲਾਂ kleidiai_get_ਖੰਡ_ਦਲੀਲਾਂ(ggml_type rhs_type) {
    ਚੋਣ (rhs_type) {
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਵਾਪਸ { QK4_0, QK4_0, QK4_0 };
        ਹਾਲਤ GGML_TYPE_Q4_K:
            ਵਾਪਸ { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ { 0, 0, QK8_0 };
        ਮੂਲ:
            ਵਾਪਸ { 0, 0, 0 };
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ kleidiai_pack_fallback_allowed() {
    ਜੇ (ctx.sme_thread_cap <= 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ctx.thread_hint <= 0) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਵਾਪਸ ctx.thread_hint > ctx.sme_thread_cap;
}

ਬਣਤਰ kleidiai_ਭਾਰ_ਸਿਰਲੇਖ {
    uint32_t magic;
    uint16_t ਸੰਸਕਰਣ_2;
    uint16_t ਥਾਂ_ਗਿਣਤੀ;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

ਸਥਿਰ ਸਤਰਅੰਦਰ kleidiai_ਭਾਰ_ਸਿਰਲੇਖ * kleidiai_ਭਾਰ_ਸਿਰਲੇਖ_from_ptr(ਖਾਲੀ * ਅੰਕੜੇ) {
    ਵਾਪਸ reinterpret_cast<kleidiai_ਭਾਰ_ਸਿਰਲੇਖ *>(ਅੰਕੜੇ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ kleidiai_ਭਾਰ_ਸਿਰਲੇਖ * kleidiai_ਭਾਰ_ਸਿਰਲੇਖ_from_ptr(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ) {
    ਵਾਪਸ reinterpret_cast<ਅਚਲ kleidiai_ਭਾਰ_ਸਿਰਲੇਖ *>(ਅੰਕੜੇ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ kleidiai_is_ਭਾਰ_ਸਿਰਲੇਖ_valid(ਅਚਲ kleidiai_ਭਾਰ_ਸਿਰਲੇਖ * ਸਿਰਲੇਖ) {
    ਜੇ (!ਸਿਰਲੇਖ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ਸਿਰਲੇਖ->magic != GGML_KLEIDIAI_PACK_MAGIC || ਸਿਰਲੇਖ->ਸੰਸਕਰਣ_2 != GGML_KLEIDIAI_PACK_ਸੰਸਕਰਣ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ਸਿਰਲੇਖ->ਥਾਂ_ਗਿਣਤੀ == 0 || ਸਿਰਲੇਖ->ਥਾਂ_ਗਿਣਤੀ > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint8_t * kleidiai_ਭਾਰ_ਥਾਂ_ptr(kleidiai_ਭਾਰ_ਸਿਰਲੇਖ * ਸਿਰਲੇਖ, ਪੂਰਨ ਥਾਂ) {
    ਜੇ (!kleidiai_is_ਭਾਰ_ਸਿਰਲੇਖ_valid(ਸਿਰਲੇਖ)) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (ਥਾਂ < 0 || ਥਾਂ >= ਸਿਰਲੇਖ->ਥਾਂ_ਗਿਣਤੀ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ reinterpret_cast<uint8_t *>(ਸਿਰਲੇਖ) + ਸਿਰਲੇਖ->offsets[ਥਾਂ];
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ uint8_t * kleidiai_ਭਾਰ_ਥਾਂ_ptr(ਅਚਲ kleidiai_ਭਾਰ_ਸਿਰਲੇਖ * ਸਿਰਲੇਖ, ਪੂਰਨ ਥਾਂ) {
    ਜੇ (!kleidiai_is_ਭਾਰ_ਸਿਰਲੇਖ_valid(ਸਿਰਲੇਖ)) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (ਥਾਂ < 0 || ਥਾਂ >= ਸਿਰਲੇਖ->ਥਾਂ_ਗਿਣਤੀ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ reinterpret_cast<ਅਚਲ uint8_t *>(ਸਿਰਲੇਖ) + ਸਿਰਲੇਖ->offsets[ਥਾਂ];
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    ਵਾਪਸ ctx.kernels_q4;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    ਵਾਪਸ ctx.kernels_q4k;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    ਵਾਪਸ ctx.kernels_q8;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    ਵਾਪਸ ctx.kernels_f32;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ SelectFallback>
ਸਥਿਰ ਪੂਰਨ kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ਬਾਹਰਲਾ,
        SelectFallback select_fallback) {
    ਪੂਰਨ count = 0;
    ਜੇ (!primary) {
        ਵਾਪਸ 0;
    }
    ਬਾਹਰਲਾ[count++] = primary;

    ਜੇ (primary->rhs_info.repack_ਬਹੁਲਕ == RHS_REPACK_SINGLE_ONLY) {
        ਵਾਪਸ count;
    }

    ਜੇ (is_sme_family(primary->required_cpu)) {
        ਅਚਲ cpu_feature fallback_ਨਕਾਬ = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        ਜੇ (fallback_ਨਕਾਬ != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_ਨਕਾਬ);
            ਜੇ (fallback && fallback != primary &&
                fallback->rhs_info.repack_ਬਹੁਲਕ != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                ਬਾਹਰਲਾ[count++] = fallback;
            }
        }
    }

    ਵਾਪਸ count;
}

ਸਥਿਰ ਪੂਰਨ kleidiai_collect_kernel_chain(ਅਚਲ ਬਣਤਰ ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ਬਾਹਰਲਾ) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    ਵਾਪਸ kleidiai_collect_kernel_chain_common(primary, ctx.features, ਬਾਹਰਲਾ,
        [&](cpu_feature ਨਕਾਬ) { ਵਾਪਸ ggml_kleidiai_select_kernels(ਨਕਾਬ, op); });
}

ਸਥਿਰ ਪੂਰਨ kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ਬਾਹਰਲਾ) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    ਵਾਪਸ kleidiai_collect_kernel_chain_common(primary, ctx.features, ਬਾਹਰਲਾ,
        [&](cpu_feature ਨਕਾਬ) { ਵਾਪਸ ggml_kleidiai_select_kernels_q4_0(ਨਕਾਬ); });
}

ਸਥਿਰ ਪੂਰਨ kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ਬਾਹਰਲਾ) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    ਵਾਪਸ kleidiai_collect_kernel_chain_common(primary, ctx.features, ਬਾਹਰਲਾ,
        [&](cpu_feature ਨਕਾਬ) { ਵਾਪਸ ggml_kleidiai_select_kernels_q4_k(ਨਕਾਬ); });
}

ਸਥਿਰ ਪੂਰਨ kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ਬਾਹਰਲਾ) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    ਵਾਪਸ kleidiai_collect_kernel_chain_common(primary, ctx.features, ਬਾਹਰਲਾ,
        [&](cpu_feature ਨਕਾਬ) { ਵਾਪਸ ggml_kleidiai_select_kernels_q8_0(ਨਕਾਬ); });
}

ਸਥਿਰ ਪੂਰਨ kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & ਬਾਹਰਲਾ) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    ਵਾਪਸ kleidiai_collect_kernel_chain_common(primary, ctx.features, ਬਾਹਰਲਾ,
        [&](cpu_feature ਨਕਾਬ) { ਵਾਪਸ ggml_kleidiai_select_kernels_f32(ਨਕਾਬ); });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t ggml_ne(ਅਚਲ ggml_tensor * tensor, ਪੂਰਨ dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    ਵਾਪਸ tensor->ne[dim];
}

ਨਾਮਾਕਾਸ਼ ggml::cpu::kleidiai {

ਸਥਿਰ size_t ਗੇੜਾ_ਥੱਲੇ(size_t x, size_t y) {
    ਵਾਪਸ y == 0 ? x : x - (x % y);
}

ਸਥਿਰ ਖਾਲੀ transpose_f32kxn_f16nxk(size_t n, size_t k, ਦਸ਼ਮਲਵ * dst, ਅਚਲ uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / ਆਕਾਰ(uint16_t);
    size_t dst_stride = n;

    ਲਈ (size_t k_idx = 0; k_idx < k; ++k_idx) {
        ਲਈ (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

ਵਰਗ tensor_traits : ਜਨਤਕ ggml::cpu::tensor_traits {
    ਤਾਰਕਿਕ work_ਮਾਪ(ਪੂਰਨ /* n_threads */, ਅਚਲ ਬਣਤਰ ggml_tensor * op, size_t & size) override {
        ਜੇ (op->op != GGML_OP_MUL_MAT) {
            ਵਾਪਸ ਝੂਠ;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ਅਚਲ ਪੂਰਨ ਥਾਂ_ਗਿਣਤੀ = kleidiai_collect_kernel_chain(op, kernel_chain);
        ਜੇ (ਥਾਂ_ਗਿਣਤੀ == 0) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਤਾਰਕਿਕ is_gemv = op->src[1]->ne[1] == 1;
        ਅਚਲ size_t k = op->src[0]->ne[0];
        ਅਚਲ size_t n = op->src[0]->ne[1];
        ਅਚਲ size_t m = op->src[1]->ne[1];

        ਜੇ (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ਅਚਲ kleidiai_ਖੰਡ_ਦਲੀਲਾਂ ਖੰਡ_ਦਲੀਲਾਂ = kleidiai_get_ਖੰਡ_ਦਲੀਲਾਂ(kernel_chain[0]->rhs_type);
            ਅਚਲ size_t qk = ਖੰਡ_ਦਲੀਲਾਂ.pack_bl;

            size_t cursor = 0;
            ਤਾਰਕਿਕ any_ਥਾਂ = ਝੂਠ;

            ਲਈ (ਪੂਰਨ ਥਾਂ = 0; ਥਾਂ < ਥਾਂ_ਗਿਣਤੀ; ++ਥਾਂ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[ਥਾਂ];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                ਜੇ (!lhs_info || !lhs_info->packed_ਮਾਪ_ex || !kernel) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਅਚਲ size_t mr = kernel->get_mr();
                ਅਚਲ size_t kr = kernel->get_kr();
                ਅਚਲ size_t sr = kernel->get_sr();

                ਅਚਲ size_t packed = lhs_info->packed_ਮਾਪ_ex(m, k, qk, mr, kr, sr);

                cursor = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_ਥਾਂ = ਸੱਚ;
            }

            ਜੇ (!any_ਥਾਂ) {
                ਵਾਪਸ ਝੂਠ;
            }

            size = cursor;
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : ਖਾਲੀ_ਸੂਚਕ;
            ਜੇ (is_gemv && op->src[1]->nb[0] == (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ) && gemv_kernel &&
                gemv_kernel->get_lhs_ਵਿੱਥ_ex && gemv_kernel->get_rhs_packed_ਵਿੱਥ_ex &&
                gemv_kernel->ਲੜੀਵਾਰ_kernel_ex && gemv_kernel->get_dst_ਵਿੱਥ) {
                size = 0;
                ਵਾਪਸ ਸੱਚ;
            }

            size_t cursor = 0;
            ਤਾਰਕਿਕ any_ਥਾਂ = ਝੂਠ;

            ਲਈ (ਪੂਰਨ ਥਾਂ = 0; ਥਾਂ < ਥਾਂ_ਗਿਣਤੀ; ++ਥਾਂ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[ਥਾਂ];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                ਜੇ (!lhs_info || !lhs_info->packed_ਮਾਪ_ex || !kernel) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਅਚਲ size_t mr = kernel->get_mr();
                ਅਚਲ size_t kr = kernel->get_kr();
                ਅਚਲ size_t sr = kernel->get_sr();

                cursor  = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_ਮਾਪ_ex(m, k, 0, mr, kr, sr);
                any_ਥਾਂ = ਸੱਚ;
            }

            ਜੇ (!any_ਥਾਂ) {
                ਵਾਪਸ ਝੂਠ;
            }

            size = cursor;
            ਵਾਪਸ ਸੱਚ;
        }

        ਜੇ (op->src[0]->type == GGML_TYPE_F16) {
            ਅਚਲ int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ਅਚਲ int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ਅਚਲ int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            ਤਾਰਕਿਕ any_ਥਾਂ = ਝੂਠ;

            ਲਈ (ਪੂਰਨ ਥਾਂ = 0; ਥਾਂ < ਥਾਂ_ਗਿਣਤੀ; ++ਥਾਂ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[ਥਾਂ];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                ਜੇ (!lhs_info || !lhs_info->packed_ਮਾਪ_ex || !kernels->rhs_info.packed_ਮਾਪ_ex || !kernel) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਅਚਲ size_t mr = kernel->get_mr();
                ਅਚਲ size_t kr = kernel->get_kr();
                ਅਚਲ size_t sr = kernel->get_sr();

                cursor  = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_ਮਾਪ_ex(m * r, k, 0, mr, kr, sr);
                any_ਥਾਂ = ਸੱਚ;
            }

            ਲਈ (ਪੂਰਨ ਥਾਂ = 0; ਥਾਂ < ਥਾਂ_ਗਿਣਤੀ; ++ਥਾਂ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[ਥਾਂ];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                ਜੇ (!kernel || !kernels->rhs_info.packed_ਮਾਪ_ex) {
                    ਵਾਪਸ ਝੂਠ;
                }
                cursor  = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_ਮਾਪ_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            cursor  = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

            ਜੇ (!any_ਥਾਂ) {
                ਵਾਪਸ ਝੂਠ;
            }

            size = cursor;
            ਵਾਪਸ ਸੱਚ;
        }

        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ compute_forward(ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst) override {
        ਜੇ (dst->op == GGML_OP_MUL_MAT) {
            ਜੇ (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                ਵਾਪਸ compute_forward_qx(params, dst);
            } ਨਹੀਂਤਾਂ ਜੇ (dst->src[0]->type == GGML_TYPE_F32) {
                ਵਾਪਸ compute_forward_f32(params, dst);
            } ਨਹੀਂਤਾਂ ਜੇ (dst->src[0]->type == GGML_TYPE_F16) {
                ਵਾਪਸ compute_forward_fp16(params, dst);
            }
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ compute_forward_f32(ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        ਅਚਲ ggml_tensor * src0 = dst->src[0];
        ਅਚਲ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

        ਜੇ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        ਜੇ (!kernels) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ size_t k = ne00;
        ਅਚਲ size_t m = ne11;
        ਅਚਲ size_t n = ne01;
        ਅਚਲ ਤਾਰਕਿਕ use_gemv = m == 1 && src1->nb[0] == (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                              kernels->gemv.get_lhs_ਵਿੱਥ_ex &&
                              kernels->gemv.get_rhs_packed_ਵਿੱਥ_ex &&
                              kernels->gemv.ਲੜੀਵਾਰ_kernel_ex &&
                              kernels->gemv.get_dst_ਵਿੱਥ;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        ਜੇ (!kernel || !kernel->get_lhs_ਵਿੱਥ_ex ||
            !kernel->get_rhs_packed_ਵਿੱਥ_ex || !kernel->ਲੜੀਵਾਰ_kernel_ex || !kernel->get_dst_ਵਿੱਥ) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (!use_gemv && (!lhs_info || !lhs_info->get_ਵਿੱਥ || !lhs_info->get_packed_ਵਿੱਥ_ex ||
                          !lhs_info->packed_ਮਾਪ_ex || !lhs_info->pack_func_ex)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ kleidiai_ਭਾਰ_ਸਿਰਲੇਖ * ਸਿਰਲੇਖ = kleidiai_ਭਾਰ_ਸਿਰਲੇਖ_from_ptr(src0->ਅੰਕੜੇ);
        ਅਚਲ ਤਾਰਕਿਕ has_ਸਿਰਲੇਖ = kleidiai_is_ਭਾਰ_ਸਿਰਲੇਖ_valid(ਸਿਰਲੇਖ);

        ਅਚਲ uint8_t * rhs_ਅਧਾਰ = has_ਸਿਰਲੇਖ ? kleidiai_ਭਾਰ_ਥਾਂ_ptr(ਸਿਰਲੇਖ, 0)
                                              : static_cast<ਅਚਲ uint8_t *>(src0->ਅੰਕੜੇ);
        ਜੇ (!rhs_ਅਧਾਰ) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਪੂਰਨ nth = params->nth > 0 ? params->nth : 1;
        ਅਚਲ ਪੂਰਨ ith = params->ith;

        ਅਚਲ size_t mr = kernel->get_mr();
        ਅਚਲ size_t kr = kernel->get_kr();
        ਅਚਲ size_t sr = kernel->get_sr();

        ਅਚਲ size_t lhs_packed_ਮਾਪ = use_gemv ? 0 : lhs_info->packed_ਮਾਪ_ex(m, k, 0, mr, kr, sr);
        ਜੇ (!use_gemv) {
            GGML_ASSERT(lhs_packed_ਮਾਪ <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ਅਚਲ size_t dst_stride = dst->nb[1];
        ਅਚਲ size_t n_ਕਦਮ = kernel->get_n_ਕਦਮ() ? kernel->get_n_ਕਦਮ() : 1;
        ਅਚਲ ਤਾਰਕਿਕ disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ);

        ਅਚਲ int64_t batch_ਮਾਪ = ne12 * ne13;
        ਲਈ (int64_t batch_idx = 0; batch_idx < batch_ਮਾਪ; ++batch_idx) {
            ਅਚਲ int64_t i12 = batch_idx % ne12;
            ਅਚਲ int64_t i13 = batch_idx / ne12;
            ਅਚਲ uint8_t * lhs_batch_ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(src1->ਅੰਕੜੇ) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_ਅਧਾਰ = static_cast<uint8_t *>(dst->ਅੰਕੜੇ) + i12 * dst->nb[2] + i13 * dst->nb[3];

            ਜੇ (!use_gemv) {
                ਅਚਲ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_ਧਾਗੇ = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_ਧਾਗੇ = std::max<int64_t>(1, max_ਧਾਗੇ);
                ਅਚਲ int64_t use_ਧਾਗੇ = std::min<int64_t>(nth, max_ਧਾਗੇ);

                ਜੇ (ith < use_ਧਾਗੇ) {
                    ਅਚਲ int64_t ਅੰਕ_m_ਪ੍ਰਤੀ_thread0   = ਗੇੜਾ_ਥੱਲੇ((size_t)(m_roundup_mr / use_ਧਾਗੇ), mr);
                    ਅਚਲ int64_t ਅੰਕ_m_ਪ੍ਰਤੀ_thread_N_1 = (int64_t)m - (use_ਧਾਗੇ - 1) * ਅੰਕ_m_ਪ੍ਰਤੀ_thread0;

                    ਅਚਲ int64_t m_ਸ਼ੁਰੂ = (int64_t)ith * ਅੰਕ_m_ਪ੍ਰਤੀ_thread0;
                    ਅਚਲ int64_t m_ਗਿਣਤੀ = (ith == use_ਧਾਗੇ - 1) ? ਅੰਕ_m_ਪ੍ਰਤੀ_thread_N_1 : ਅੰਕ_m_ਪ੍ਰਤੀ_thread0;

                    ਅਚਲ size_t ਅਧਾਰ_packed_off  = lhs_info->get_packed_ਵਿੱਥ_ex(m_ਸ਼ੁਰੂ, k, 0, mr, kr, sr);
                    ਅਚਲ size_t ਅਗਲਾ_ਖੰਡ_off   = lhs_info->get_packed_ਵਿੱਥ_ex(m_ਸ਼ੁਰੂ + mr, k, 0, mr, kr, sr);
                    ਅਚਲ size_t ਪੰਕਤੀ_stride_ਬਾਈਟਾਂ = mr ? (ਅਗਲਾ_ਖੰਡ_off - ਅਧਾਰ_packed_off) / mr : 0;

                    int64_t ਬਚਿਆ = m_ਗਿਣਤੀ;
                    int64_t cur       = m_ਸ਼ੁਰੂ;

                    ਜਦੋਂ (ਬਚਿਆ > 0) {
                        ਅਚਲ int64_t take = std::min<int64_t>((int64_t)m - cur, ਬਚਿਆ);
                        ਅਚਲ size_t src_off = lhs_info->get_ਵਿੱਥ(cur, src1->nb[1]);
                        ਅਚਲ ਖਾਲੀ * src_ptr = lhs_batch_ਅਧਾਰ + src_off;
                        ਅਚਲ size_t dst_off = ਅਧਾਰ_packed_off + (size_t)(cur - m_ਸ਼ੁਰੂ) * ਪੰਕਤੀ_stride_ਬਾਈਟਾਂ;
                        ਖਾਲੀ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        ਬਚਿਆ -= take;
                    }
                }
            }

            ਜੇ (ith == 0) {
                ggml_threadpool_chunk_ਸੈੱਟ(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ਅਚਲ size_t chunk_ਖਾਨੇ_ਗਿਣਤੀ = kleidiai_chunk_ਖਾਨੇ_ਗਿਣਤੀ(n, nth, disable_chunking, n_ਕਦਮ);
            GGML_ASSERT(chunk_ਖਾਨੇ_ਗਿਣਤੀ <= (size_t) ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ);

            ਪੂਰਨ current_ਖਾਨਾ = ggml_threadpool_chunk_ਜੋੜੋ(params->threadpool, (ਪੂਰਨ) chunk_ਖਾਨੇ_ਗਿਣਤੀ);
            ਜਦੋਂ ((size_t) current_ਖਾਨਾ < n) {
                ਅਚਲ size_t n_ਸ਼ੁਰੂ = (size_t) current_ਖਾਨਾ;
                ਅਚਲ size_t n_ਮੰਜ਼ਿਲ_process = std::min(chunk_ਖਾਨੇ_ਗਿਣਤੀ, n - n_ਸ਼ੁਰੂ);

                ਜੇ (n_ਮੰਜ਼ਿਲ_process > 0) {
                    ਅਚਲ size_t lhs_ਵਿੱਥ = use_gemv ? kernel->get_lhs_ਵਿੱਥ_ex(0, k, 0)
                                                       : lhs_info->get_packed_ਵਿੱਥ_ex(0, k, 0, mr, kr, sr);
                    ਅਚਲ size_t rhs_packed_ਵਿੱਥ = kernel->get_rhs_packed_ਵਿੱਥ_ex(n_ਸ਼ੁਰੂ, k, 0);
                    ਅਚਲ size_t dst_ਵਿੱਥ        = kernel->get_dst_ਵਿੱਥ(0, n_ਸ਼ੁਰੂ, dst_stride);

                    ਅਚਲ ਖਾਲੀ * lhs_ptr = use_gemv ? lhs_batch_ਅਧਾਰ + lhs_ਵਿੱਥ
                                                    : lhs_packed + lhs_ਵਿੱਥ;
                    ਅਚਲ ਖਾਲੀ * rhs_ptr = rhs_ਅਧਾਰ + rhs_packed_ਵਿੱਥ;
                    ਦਸ਼ਮਲਵ * dst_ptr      = reinterpret_cast<ਦਸ਼ਮਲਵ *>(dst_batch_ਅਧਾਰ + dst_ਵਿੱਥ);

                    kernel->ਲੜੀਵਾਰ_kernel_ex(m, n_ਮੰਜ਼ਿਲ_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_ਖਾਨਾ = ggml_threadpool_chunk_ਜੋੜੋ(params->threadpool, (ਪੂਰਨ) chunk_ਖਾਨੇ_ਗਿਣਤੀ);
            }

            ਜੇ (batch_idx != batch_ਮਾਪ - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ compute_forward_fp16(ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst) {
        ਅਚਲ ggml_tensor * src0 = dst->src[0];
        ਅਚਲ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        ਜੇ (!kernels) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਤਾਰਕਿਕ is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        ਜੇ (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_ਵਿੱਥ_ex || !kernel->get_rhs_packed_ਵਿੱਥ_ex || !kernel->ਲੜੀਵਾਰ_kernel_ex) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਪੂਰਨ nth = params->nth;
        ਅਚਲ ਪੂਰਨ ith = params->ith;

        ਅਚਲ int64_t lhs_batch_size0 = ne12;
        ਅਚਲ int64_t rhs_batch_size0 = ne02;
        ਅਚਲ int64_t batch_ਮਾਪ      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        ਅਚਲ int64_t r = lhs_batch_size0 / rhs_batch_size0;

        ਅਚਲ int64_t m_ਸਮੂਹ = ne11;
        ਅਚਲ int64_t m       = m_ਸਮੂਹ;
        ਅਚਲ int64_t n       = ne01;
        ਅਚਲ int64_t k       = ne00;

        ਅਚਲ size_t lhs_stride = src1->nb[1];
        ਅਚਲ size_t rhs_stride = src0->nb[1];
        ਅਚਲ size_t dst_stride = dst->nb[1];

        ਅਚਲ int64_t mr = (int64_t) kernel->get_mr();
        ਅਚਲ int64_t nr = (int64_t) kernel->get_nr();
        ਅਚਲ int64_t kr = (int64_t) kernel->get_kr();
        ਅਚਲ int64_t sr = (int64_t) kernel->get_sr();

        ਅਚਲ size_t lhs_packed_ਮਾਪ = lhs_info->packed_ਮਾਪ_ex(m, k, 0, mr, kr, sr);
        ਅਚਲ size_t rhs_packed_ਮਾਪ = kernels->rhs_info.packed_ਮਾਪ_ex(n, k, nr, kr, 0);
        ਅਚਲ size_t kxn_ਮਾਪ        = k * n * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ size_t bias_ਮਾਪ       = n * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        ਅਚਲ size_t wsize_required = lhs_packed_ਮਾਪ + rhs_packed_ਮਾਪ + kxn_ਮਾਪ + bias_ਮਾਪ;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_ਮਾਪ;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_ਮਾਪ;
        uint8_t * bias       = rhs_kxn + kxn_ਮਾਪ;

        ਲਈ (int64_t batch_idx = 0; batch_idx < batch_ਮਾਪ; ++batch_idx) {
            ਅਚਲ int64_t rhs_batch_idx = batch_idx / r;
            ਅਚਲ uint8_t * rhs_batch_ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(src0->ਅੰਕੜੇ) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_ਅਧਾਰ = static_cast<uint8_t *>(dst->ਅੰਕੜੇ) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ਅਚਲ int64_t m_roundup_mr = kai_roundup(m, mr);
                ਅਚਲ int64_t ਅੰਕ_ਧਾਗੇ  = KAI_MIN(m_roundup_mr / mr, nth);

                ਜੇ (ith < ਅੰਕ_ਧਾਗੇ) {
                    ਅਚਲ int64_t ਅੰਕ_m_ਪ੍ਰਤੀ_thread0   = ਗੇੜਾ_ਥੱਲੇ((size_t)(m_roundup_mr / ਅੰਕ_ਧਾਗੇ), (size_t)mr);
                    ਅਚਲ int64_t ਅੰਕ_m_ਪ੍ਰਤੀ_thread_N_1 = m - (ਅੰਕ_ਧਾਗੇ - 1) * ਅੰਕ_m_ਪ੍ਰਤੀ_thread0;

                    ਅਚਲ int64_t m_ਸ਼ੁਰੂ = ith * ਅੰਕ_m_ਪ੍ਰਤੀ_thread0;
                    ਅਚਲ int64_t m_ਗਿਣਤੀ = (ith == ਅੰਕ_ਧਾਗੇ - 1) ? ਅੰਕ_m_ਪ੍ਰਤੀ_thread_N_1 : ਅੰਕ_m_ਪ੍ਰਤੀ_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ਅਚਲ size_t ਅਧਾਰ_packed_off  = lhs_info->get_packed_ਵਿੱਥ_ex(m_ਸ਼ੁਰੂ, k, 0, mr, kr, sr);
                    ਅਚਲ size_t ਅਗਲਾ_ਖੰਡ_off   = lhs_info->get_packed_ਵਿੱਥ_ex(m_ਸ਼ੁਰੂ + mr, k, 0, mr, kr, sr);
                    ਅਚਲ size_t ਪੰਕਤੀ_stride_ਬਾਈਟਾਂ = (ਅਗਲਾ_ਖੰਡ_off - ਅਧਾਰ_packed_off) / (size_t)mr;

                    int64_t ਬਚਿਆ = m_ਗਿਣਤੀ;
                    int64_t cur       = m_ਸ਼ੁਰੂ;

                    ਜਦੋਂ (ਬਚਿਆ > 0) {
                        ਅਚਲ int64_t ਪੰਕਤੀ_ਵਿੱਚ_ਸਮੂਹ = cur;
                        ਅਚਲ int64_t avail        = m_ਸਮੂਹ - ਪੰਕਤੀ_ਵਿੱਚ_ਸਮੂਹ;
                        ਅਚਲ int64_t take         = std::min(avail, ਬਚਿਆ);

                        ਅਚਲ uint8_t * lhs_batch_ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(src1->ਅੰਕੜੇ) + batch_idx * src1->nb[2];
                        ਅਚਲ ਖਾਲੀ * src_ptr = lhs_batch_ਅਧਾਰ + (size_t)ਪੰਕਤੀ_ਵਿੱਚ_ਸਮੂਹ * lhs_stride;
                        ਅਚਲ size_t dst_off = ਅਧਾਰ_packed_off + (size_t)(cur - m_ਸ਼ੁਰੂ) * ਪੰਕਤੀ_stride_ਬਾਈਟਾਂ;
                        ਖਾਲੀ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        ਬਚਿਆ -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            ਜੇ (ith == 0) {
                memset(bias, 0, (size_t)n * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<ਦਸ਼ਮਲਵ *>(rhs_kxn),
                                        reinterpret_cast<ਅਚਲ uint16_t *>(rhs_batch_ਅਧਾਰ),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                             rhs_kxn, bias, ਖਾਲੀ_ਸੂਚਕ, rhs_packed, 0, ਖਾਲੀ_ਸੂਚਕ);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ਅਚਲ int64_t n_ਕਦਮ  = (int64_t) kernel->get_n_ਕਦਮ();
                int64_t ਅੰਕ_ਧਾਗੇ_n = KAI_MIN(n / n_ਕਦਮ, nth);
                ਜੇ (ਅੰਕ_ਧਾਗੇ_n <= 0) {
                    ਅੰਕ_ਧਾਗੇ_n = 1;
                }

                ਜੇ (ith < ਅੰਕ_ਧਾਗੇ_n) {
                    ਅਚਲ int64_t ਅੰਕ_n_ਪ੍ਰਤੀ_thread0   = ਗੇੜਾ_ਥੱਲੇ((size_t)(n / ਅੰਕ_ਧਾਗੇ_n), (size_t)n_ਕਦਮ);
                    ਅਚਲ int64_t ਅੰਕ_n_ਪ੍ਰਤੀ_thread_N_1 = n - (ਅੰਕ_ਧਾਗੇ_n - 1) * ਅੰਕ_n_ਪ੍ਰਤੀ_thread0;

                    ਅਚਲ int64_t n_ਸ਼ੁਰੂ      = ith * ਅੰਕ_n_ਪ੍ਰਤੀ_thread0;
                    ਅਚਲ int64_t n_ਮੰਜ਼ਿਲ_process = (ith == ਅੰਕ_ਧਾਗੇ_n - 1) ? ਅੰਕ_n_ਪ੍ਰਤੀ_thread_N_1 : ਅੰਕ_n_ਪ੍ਰਤੀ_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ਅਚਲ size_t lhs_packed_offset0 = lhs_info->get_packed_ਵਿੱਥ_ex(0, k, 0, mr, kr, sr);
                    ਅਚਲ size_t rhs_packed_ਵਿੱਥ  = kernel->get_rhs_packed_ਵਿੱਥ_ex(n_ਸ਼ੁਰੂ, k, 0);
                    ਅਚਲ size_t dst_ਵਿੱਥ         = kernel->get_dst_ਵਿੱਥ((size_t)0, (size_t)n_ਸ਼ੁਰੂ, dst_stride);

                    ਅਚਲ ਖਾਲੀ * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ਅਚਲ ਖਾਲੀ * rhs_ptr = rhs_packed + rhs_packed_ਵਿੱਥ;
                    ਦਸ਼ਮਲਵ * dst_ptr      = reinterpret_cast<ਦਸ਼ਮਲਵ *>(dst_batch_ਅਧਾਰ + dst_ਵਿੱਥ);

                    kernel->ਲੜੀਵਾਰ_kernel_ex(m, n_ਮੰਜ਼ਿਲ_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, ਆਕਾਰ(ਦਸ਼ਮਲਵ), -FLT_MAX, FLT_MAX);
                }
            }

            ਜੇ (batch_idx != batch_ਮਾਪ - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ compute_forward_qx(ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ਅਚਲ ggml_tensor * src0 = dst->src[0];
        ਅਚਲ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

        ਅਚਲ kleidiai_ਭਾਰ_ਸਿਰਲੇਖ * ਸਿਰਲੇਖ = kleidiai_ਭਾਰ_ਸਿਰਲੇਖ_from_ptr(src0->ਅੰਕੜੇ);
        ਅਚਲ ਤਾਰਕਿਕ has_ਸਿਰਲੇਖ = kleidiai_is_ਭਾਰ_ਸਿਰਲੇਖ_valid(ਸਿਰਲੇਖ);
        ਅਚਲ ਤਾਰਕਿਕ is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ਅਚਲ ਪੂਰਨ ਥਾਂ_ਕੁੱਲ = kleidiai_collect_kernel_chain(dst, kernel_chain);

        ਸਵੈਚਾਲਿਤ ਭਾਰ_for_ਥਾਂ = [&](ਪੂਰਨ ਥਾਂ_ਸੂਚਕਾਂਕ, size_t & ਮਾਪ_ਬਾਹਰਲਾ) -> ਅਚਲ uint8_t * {
            ਜੇ (ਥਾਂ_ਸੂਚਕਾਂਕ < 0 || ਥਾਂ_ਸੂਚਕਾਂਕ >= ਥਾਂ_ਕੁੱਲ) {
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
            ਜੇ (has_ਸਿਰਲੇਖ) {
                ਜੇ (ਥਾਂ_ਸੂਚਕਾਂਕ < ਸਿਰਲੇਖ->ਥਾਂ_ਗਿਣਤੀ) {
                    ਮਾਪ_ਬਾਹਰਲਾ = static_cast<size_t>(ਸਿਰਲੇਖ->sizes[ਥਾਂ_ਸੂਚਕਾਂਕ]);
                    ਵਾਪਸ kleidiai_ਭਾਰ_ਥਾਂ_ptr(ਸਿਰਲੇਖ, ਥਾਂ_ਸੂਚਕਾਂਕ);
                }
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
            ਜੇ (ਥਾਂ_ਸੂਚਕਾਂਕ == 0) {
                ਮਾਪ_ਬਾਹਰਲਾ = ggml_nbytes(src0);
                ਵਾਪਸ static_cast<ਅਚਲ uint8_t *>(src0->ਅੰਕੜੇ);
            }
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        };

        ਬਣਤਰ runtime_ਥਾਂ {
            ਪੂਰਨ ਥਾਂ_ਸੂਚਕਾਂਕ;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            size_t mr;
            size_t nr;
            size_t kr;
            size_t sr;
            size_t n_ਕਦਮ;
            size_t lhs_packed_ਮਾਪ;
            size_t lhs_ਵਿੱਥ;
            size_t lhs_bl;
            size_t rhs_bl;
            size_t pack_bl;
            size_t lhs_packed_offset0;
            ਪੂਰਨ assigned_ਧਾਗੇ;
            ਪੂਰਨ thread_begin;
            ਪੂਰਨ thread_ਸਿਰਾ;
            ਅਚਲ uint8_t * rhs_ਅਧਾਰ;
        };

        std::array<runtime_ਥਾਂ, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        ਪੂਰਨ runtime_ਗਿਣਤੀ = 0;

        ਲਈ (ਪੂਰਨ ਥਾਂ = 0; ਥਾਂ < ਥਾਂ_ਕੁੱਲ && runtime_ਗਿਣਤੀ < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++ਥਾਂ) {
            ggml_kleidiai_kernels * kernels = kernel_chain[ਥਾਂ];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            ਜੇ (!kinfo || !linfo || !linfo->packed_ਮਾਪ_ex || !linfo->pack_func_ex || !linfo->get_ਵਿੱਥ ||
                !kinfo->get_rhs_packed_ਵਿੱਥ_ex || !kinfo->ਲੜੀਵਾਰ_kernel_ex || !kinfo->get_dst_ਵਿੱਥ) {
                ਜਾਰੀ;
            }

            size_t rhs_ਮਾਪ = 0;
            ਅਚਲ uint8_t * rhs_ptr = ਭਾਰ_for_ਥਾਂ(ਥਾਂ, rhs_ਮਾਪ);
            ਜੇ (!rhs_ptr || rhs_ਮਾਪ == 0) {
                ਜਾਰੀ;
            }

            ਅਚਲ kleidiai_ਖੰਡ_ਦਲੀਲਾਂ ਖੰਡ_ਦਲੀਲਾਂ = kleidiai_get_ਖੰਡ_ਦਲੀਲਾਂ(kernels->rhs_type);

            runtime[runtime_ਗਿਣਤੀ] = {
                ਥਾਂ,
                kernels,
                kinfo,
                linfo,
                kinfo->get_mr(),
                kinfo->get_nr(),
                kinfo->get_kr(),
                kinfo->get_sr(),
                kinfo->get_n_ਕਦਮ(),
                0,
                0,
                ਖੰਡ_ਦਲੀਲਾਂ.lhs_bl,
                ਖੰਡ_ਦਲੀਲਾਂ.rhs_bl,
                ਖੰਡ_ਦਲੀਲਾਂ.pack_bl,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_ਗਿਣਤੀ;
        }

        ਜੇ (runtime_ਗਿਣਤੀ == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਪੂਰਨ nth_ਕੁੱਲ = params->nth > 0 ? params->nth : 1;
        ਅਚਲ ਪੂਰਨ ith_ਕੁੱਲ = params->ith;

        ਪੂਰਨ sme_ਥਾਂ = -1;
        ਪੂਰਨ non_sme_ਥਾਂ = -1;
        ਲਈ (ਪੂਰਨ i = 0; i < runtime_ਗਿਣਤੀ; ++i) {
            ਜੇ (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_ਥਾਂ = i;
                ਰੋਕੋ;
            }
        }

        ਲਈ (ਪੂਰਨ i = 0; i < runtime_ਗਿਣਤੀ; ++i) {
            ਜੇ (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_ਥਾਂ = i;
                ਰੋਕੋ;
            }
        }

        ਅਚਲ ਪੂਰਨ sme_cap_ਹੱਦਬੰਦੀ = ctx.sme_thread_cap;
        ਅਚਲ ਤਾਰਕਿਕ use_hybrid = sme_cap_ਹੱਦਬੰਦੀ > 0 &&
                                 runtime_ਗਿਣਤੀ > 1 &&
                                 nth_ਕੁੱਲ > sme_cap_ਹੱਦਬੰਦੀ;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_thread = 0;
        ਜੇ (runtime_ਗਿਣਤੀ > 0 && nth_ਕੁੱਲ > 0) {
            min_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_ਕੁੱਲ);
        }
        ਅਚਲ ਤਾਰਕਿਕ too_small_for_hybrid = (min_ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_thread < 2) || (ne11 < 128);

        ਅਚਲ ਤਾਰਕਿਕ hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        ਜੇ (!hybrid_enabled) {
            ਪੂਰਨ chosen_ਥਾਂ = 0;
            ਜੇ (too_small_for_hybrid && sme_ਥਾਂ != -1) {
                chosen_ਥਾਂ = nth_ਕੁੱਲ > sme_cap_ਹੱਦਬੰਦੀ && non_sme_ਥਾਂ != -1 ? non_sme_ਥਾਂ : sme_ਥਾਂ;
            } ਨਹੀਂਤਾਂ ਜੇ (runtime_ਗਿਣਤੀ > 1 && ctx.sme_thread_cap > 0 && nth_ਕੁੱਲ > ctx.sme_thread_cap) {
                chosen_ਥਾਂ = 1;
            }
            ਜੇ (chosen_ਥਾਂ != 0 && chosen_ਥਾਂ < runtime_ਗਿਣਤੀ) {
                runtime[0] = runtime[chosen_ਥਾਂ];
                runtime[0].assigned_ਧਾਗੇ = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_ਸਿਰਾ = 0;
            }
            runtime_ਗਿਣਤੀ = runtime_ਗਿਣਤੀ > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_ਥਾਂ = -1;
            ਜੇ (runtime_ਗਿਣਤੀ > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_ਥਾਂ = 0;
            }
        }

        ਅਚਲ ਤਾਰਕਿਕ sme_single_only = sme_ਥਾਂ != -1 &&
                                     runtime[sme_ਥਾਂ].kernels->rhs_info.repack_ਬਹੁਲਕ == RHS_REPACK_SINGLE_ONLY;
        ਪੂਰਨ sme_cap = sme_single_only ? nth_ਕੁੱਲ : kleidiai_sme_thread_cap();
        ਜੇ (sme_cap < 0) {
            sme_cap = nth_ਕੁੱਲ;
        }
        sme_cap = std::min(sme_cap, nth_ਕੁੱਲ);

        ਪੂਰਨ ਧਾਗੇ_ਬਚਿਆ = nth_ਕੁੱਲ;
        ਜੇ (sme_ਥਾਂ != -1) {
            ਪੂਰਨ sme_ਧਾਗੇ = std::min(std::max(sme_cap, 0), ਧਾਗੇ_ਬਚਿਆ);
            runtime[sme_ਥਾਂ].assigned_ਧਾਗੇ = sme_ਧਾਗੇ;
            ਧਾਗੇ_ਬਚਿਆ -= sme_ਧਾਗੇ;
        }

        ਪੂਰਨ fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        ਪੂਰਨ fallback_ਗਿਣਤੀ = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        ਲਈ (ਪੂਰਨ i = 0; i < runtime_ਗਿਣਤੀ; ++i) {
            ਜੇ (i == sme_ਥਾਂ) {
                ਜਾਰੀ;
            }
            fallback_indices[fallback_ਗਿਣਤੀ++] = i;
        }

        ਲਈ (ਪੂਰਨ fi = 0; fi < fallback_ਗਿਣਤੀ; ++fi) {
            ਜੇ (ਧਾਗੇ_ਬਚਿਆ <= 0) {
                ਰੋਕੋ;
            }
            ਅਚਲ ਪੂਰਨ ਥਾਂ_ਸੂਚਕਾਂਕ = fallback_indices[fi];
            ਅਚਲ ਪੂਰਨ slots_ਖੱਬਾ = fallback_ਗਿਣਤੀ - fi;
            ਪੂਰਨ share = (ਧਾਗੇ_ਬਚਿਆ + slots_ਖੱਬਾ - 1) / slots_ਖੱਬਾ;
            share     = std::min(share, ਧਾਗੇ_ਬਚਿਆ);
            runtime[ਥਾਂ_ਸੂਚਕਾਂਕ].assigned_ਧਾਗੇ = share;
            ਧਾਗੇ_ਬਚਿਆ -= share;
        }

        ਪੂਰਨ thread_cursor = 0;
        ਲਈ (ਪੂਰਨ i = 0; i < runtime_ਗਿਣਤੀ; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_ਧਾਗੇ;
            runtime[i].thread_ਸਿਰਾ = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ਅਚਲ ਪੂਰਨ ਸਰਗਰਮ_ਧਾਗੇ = thread_cursor;

        ਪੂਰਨ local_ਥਾਂ = -1;
        ਪੂਰਨ local_ith  = 0;
        ਲਈ (ਪੂਰਨ i = 0; i < runtime_ਗਿਣਤੀ; ++i) {
            ਜੇ (ith_ਕੁੱਲ >= runtime[i].thread_begin && ith_ਕੁੱਲ < runtime[i].thread_ਸਿਰਾ) {
                local_ਥਾਂ = i;
                local_ith  = ith_ਕੁੱਲ - runtime[i].thread_begin;
                ਰੋਕੋ;
            }
        }
        ਅਚਲ size_t k = ne00;
        ਅਚਲ size_t m = ne11;
        ਅਚਲ size_t n = ne01;

        size_t cursor = 0;
        ਲਈ (ਪੂਰਨ i = 0; i < runtime_ਗਿਣਤੀ; ++i) {
            runtime[i].lhs_packed_ਮਾਪ = runtime[i].lhs_info->packed_ਮਾਪ_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_ਵਿੱਥ = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_ਵਿੱਥ_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_ਮਾਪ;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_ਕਦਮ = 1;
        ਲਈ (ਪੂਰਨ i = 0; i < runtime_ਗਿਣਤੀ; ++i) {
            ਜੇ (runtime[i].assigned_ਧਾਗੇ == 0) {
                ਜਾਰੀ;
            }
            size_t ਅਗਲਾ_ਕਦਮ = 0;
            ਜੇ (!lcm_ਮਾਪ(common_ਕਦਮ, runtime[i].n_ਕਦਮ ? runtime[i].n_ਕਦਮ : 1, ਅਗਲਾ_ਕਦਮ)) {
                ਵਾਪਸ ਝੂਠ;
            }
            common_ਕਦਮ = ਅਗਲਾ_ਕਦਮ;
        }
        GGML_ASSERT(common_ਕਦਮ > 0);

        ਅਚਲ ਤਾਰਕਿਕ disable_chunking = ggml_is_numa();
        ਅਚਲ size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ਅਚਲ size_t chunk_ਭਾਜਕ = (ਸਰਗਰਮ_ਧਾਗੇ == 1 || disable_chunking) ? (size_t) ਸਰਗਰਮ_ਧਾਗੇ : (size_t) ਸਰਗਰਮ_ਧਾਗੇ * chunk_multiplier;
        size_t chunk_ਖਾਨੇ_ਗਿਣਤੀ = align_ਉੱਪਰ(std::max<size_t>(1, ceil_div_ਮਾਪ(n, chunk_ਭਾਜਕ)), common_ਕਦਮ);
        ਜੇ (chunk_ਖਾਨੇ_ਗਿਣਤੀ == 0) {
            chunk_ਖਾਨੇ_ਗਿਣਤੀ = common_ਕਦਮ;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ਅਚਲ size_t nchunk_ਮਾਪ = std::max<size_t>(1, ceil_div_ਮਾਪ(n, chunk_ਖਾਨੇ_ਗਿਣਤੀ));
        GGML_ASSERT(nchunk_ਮਾਪ <= (size_t)ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ);
        ਅਚਲ ਪੂਰਨ nchunk = (ਪੂਰਨ)nchunk_ਮਾਪ;
        ਅਚਲ size_t dst_stride = dst->nb[1];

        ਸਵੈਚਾਲਿਤ ਲੜੀਵਾਰ_chunk = [&](runtime_ਥਾਂ & ਥਾਂ, size_t global_ਸ਼ੁਰੂ, size_t ਖਾਨੇ_ਗਿਣਤੀ_2, uint8_t * dst_batch_ਅਧਾਰ) {
            ਅਚਲ size_t rhs_packed_ਵਿੱਥ = ਥਾਂ.kernel->get_rhs_packed_ਵਿੱਥ_ex(global_ਸ਼ੁਰੂ, k, ਥਾਂ.rhs_bl);
            ਅਚਲ size_t dst_ਵਿੱਥ        = ਥਾਂ.kernel->get_dst_ਵਿੱਥ(0, global_ਸ਼ੁਰੂ, dst_stride);

            ਅਚਲ uint8_t * lhs_ptr = scratch + ਥਾਂ.lhs_ਵਿੱਥ + ਥਾਂ.lhs_packed_offset0;
            ਅਚਲ uint8_t * rhs_ptr = ਥਾਂ.rhs_ਅਧਾਰ + rhs_packed_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ * dst_ptr         = reinterpret_cast<ਦਸ਼ਮਲਵ *>(dst_batch_ਅਧਾਰ + dst_ਵਿੱਥ);

            ਥਾਂ.kernel->ਲੜੀਵਾਰ_kernel_ex(m, ਖਾਨੇ_ਗਿਣਤੀ_2, k, ਥਾਂ.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        ਲਈ (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ਅਚਲ uint8_t * lhs_batch_ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(src1->ਅੰਕੜੇ) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_ਅਧਾਰ = static_cast<uint8_t *>(dst->ਅੰਕੜੇ) + batch_idx * dst->nb[2];

            ਜੇ (local_ਥਾਂ != -1) {
                runtime_ਥਾਂ & ਥਾਂ = runtime[local_ਥਾਂ];
                ਅਚਲ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)ਥਾਂ.mr);
                int64_t max_ਧਾਗੇ = ਥਾਂ.mr ? (m_roundup_mr / (int64_t)ਥਾਂ.mr) : ਥਾਂ.assigned_ਧਾਗੇ;
                max_ਧਾਗੇ = std::max<int64_t>(1, max_ਧਾਗੇ);
                ਅਚਲ int64_t use_ਧਾਗੇ = std::min<int64_t>(ਥਾਂ.assigned_ਧਾਗੇ, max_ਧਾਗੇ);

                ਜੇ (local_ith < use_ਧਾਗੇ) {
                    ਅਚਲ int64_t ਅੰਕ_m_ਪ੍ਰਤੀ_thread0   = ਗੇੜਾ_ਥੱਲੇ((size_t)(m_roundup_mr / use_ਧਾਗੇ), ਥਾਂ.mr);
                    ਅਚਲ int64_t ਅੰਕ_m_ਪ੍ਰਤੀ_thread_N_1 = (int64_t)m - (use_ਧਾਗੇ - 1) * ਅੰਕ_m_ਪ੍ਰਤੀ_thread0;

                    ਅਚਲ int64_t m_ਸ਼ੁਰੂ = (int64_t)local_ith * ਅੰਕ_m_ਪ੍ਰਤੀ_thread0;
                    ਅਚਲ int64_t m_ਗਿਣਤੀ = (local_ith == use_ਧਾਗੇ - 1) ? ਅੰਕ_m_ਪ੍ਰਤੀ_thread_N_1 : ਅੰਕ_m_ਪ੍ਰਤੀ_thread0;

                    ਅਚਲ size_t ਅਧਾਰ_packed_off  = ਥਾਂ.lhs_info->get_packed_ਵਿੱਥ_ex(m_ਸ਼ੁਰੂ, k, ਥਾਂ.lhs_bl, ਥਾਂ.mr, ਥਾਂ.kr, ਥਾਂ.sr);
                    ਅਚਲ size_t ਅਗਲਾ_ਖੰਡ_off   = ਥਾਂ.lhs_info->get_packed_ਵਿੱਥ_ex(m_ਸ਼ੁਰੂ + ਥਾਂ.mr, k, ਥਾਂ.lhs_bl, ਥਾਂ.mr, ਥਾਂ.kr, ਥਾਂ.sr);
                    ਅਚਲ size_t ਪੰਕਤੀ_stride_ਬਾਈਟਾਂ = ਥਾਂ.mr ? (ਅਗਲਾ_ਖੰਡ_off - ਅਧਾਰ_packed_off) / ਥਾਂ.mr : 0;

                    int64_t ਬਚਿਆ = m_ਗਿਣਤੀ;
                    int64_t cur       = m_ਸ਼ੁਰੂ;

                    uint8_t * lhs_packed = scratch + ਥਾਂ.lhs_ਵਿੱਥ;
                    ਜਦੋਂ (ਬਚਿਆ > 0) {
                        ਅਚਲ int64_t ਪੰਕਤੀ_ਵਿੱਚ_ਸਮੂਹ = cur;
                        ਅਚਲ int64_t avail        = (int64_t)m - ਪੰਕਤੀ_ਵਿੱਚ_ਸਮੂਹ;
                        ਅਚਲ int64_t take         = std::min(avail, ਬਚਿਆ);

                        ਅਚਲ size_t src_off = ਥਾਂ.lhs_info->get_ਵਿੱਥ(ਪੰਕਤੀ_ਵਿੱਚ_ਸਮੂਹ, src1->nb[1]);
                        ਅਚਲ ਖਾਲੀ * src_ptr = lhs_batch_ਅਧਾਰ + src_off;
                        ਅਚਲ size_t dst_off = ਅਧਾਰ_packed_off + (size_t)(cur - m_ਸ਼ੁਰੂ) * ਪੰਕਤੀ_stride_ਬਾਈਟਾਂ;
                        ਖਾਲੀ * dst_ptr       = lhs_packed + dst_off;

                        ਥਾਂ.lhs_info->pack_func_ex(take, k, ਥਾਂ.lhs_bl, ਥਾਂ.mr, ਥਾਂ.kr, ਥਾਂ.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        ਬਚਿਆ -= take;
                    }
                }
            }

            ਜੇ (ith_ਕੁੱਲ == 0) {
                ggml_threadpool_chunk_ਸੈੱਟ(params->threadpool, ਸਰਗਰਮ_ਧਾਗੇ);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            ਜੇ (local_ਥਾਂ != -1) {
                runtime_ਥਾਂ & ਥਾਂ = runtime[local_ਥਾਂ];
                ਪੂਰਨ current_chunk = ith_ਕੁੱਲ;
                ਜਦੋਂ (current_chunk < nchunk) {
                    ਅਚਲ size_t global_ਸ਼ੁਰੂ = (size_t)current_chunk * chunk_ਖਾਨੇ_ਗਿਣਤੀ;
                    ਜੇ (global_ਸ਼ੁਰੂ >= n) {
                        ਰੋਕੋ;
                    }

                    ਅਚਲ size_t ਖਾਨੇ_ਗਿਣਤੀ_2 = std::min(chunk_ਖਾਨੇ_ਗਿਣਤੀ, n - global_ਸ਼ੁਰੂ);
                    ਜੇ (ਖਾਨੇ_ਗਿਣਤੀ_2 > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        ਲੜੀਵਾਰ_chunk(ਥਾਂ, global_ਸ਼ੁਰੂ, ਖਾਨੇ_ਗਿਣਤੀ_2, dst_batch_ਅਧਾਰ);
                    }

                    current_chunk = ggml_threadpool_chunk_ਜੋੜੋ(params->threadpool, 1);
                }
            }

            ਜੇ (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ਵਾਪਸ ਸੱਚ;
    }

ਜਨਤਕ:
    ਪੂਰਨ repack(ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ਅਚਲ size_t n = tensor->ne[1];
        ਅਚਲ size_t k = tensor->ne[0];

        kleidiai_ਭਾਰ_ਸਿਰਲੇਖ * ਸਿਰਲੇਖ = kleidiai_ਭਾਰ_ਸਿਰਲੇਖ_from_ptr(tensor->ਅੰਕੜੇ);
        ਜੇ (!ਸਿਰਲੇਖ) {
            ਵਾਪਸ -1;
        }

        ਸਿਰਲੇਖ->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        ਸਿਰਲੇਖ->ਸੰਸਕਰਣ_2    = GGML_KLEIDIAI_PACK_ਸੰਸਕਰਣ;
        ਸਿਰਲੇਖ->ਥਾਂ_ਗਿਣਤੀ = 0;

        uint8_t * ਅਧਾਰ_ptr = static_cast<uint8_t *>(tensor->ਅੰਕੜੇ);
        size_t cursor = ਆਕਾਰ(kleidiai_ਭਾਰ_ਸਿਰਲੇਖ);
        cursor = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ਅਚਲ ਤਾਰਕਿਕ want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ਅਚਲ ਤਾਰਕਿਕ want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ਅਚਲ ਤਾਰਕਿਕ want_f32 = tensor->type == GGML_TYPE_F32;
        ਅਚਲ ਪੂਰਨ ਥਾਂ_ਕੁੱਲ = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ਅਚਲ ਤਾਰਕਿਕ allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<ਦਸ਼ਮਲਵ>  scales;
        std::vector<ਦਸ਼ਮਲਵ>  bias;
        std::vector<ਦਸ਼ਮਲਵ> ਪੰਕਤੀ_f32;

        ਜੇ (want_q8 && ਥਾਂ_ਕੁੱਲ > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ਅਚਲ size_t ਪੰਕਤੀ_stride = tensor->nb[1];
            ਅਚਲ size_t k_ਖੰਡਾਂ   = (k + QK8_0 - 1) / QK8_0;

            ਜੇ (tensor->type == GGML_TYPE_Q6_K) {
                ਜੇ (k % QK_K != 0 || k % QK8_0 != 0) {
                    ਵਾਪਸ -1;
                }
                ਪੰਕਤੀ_f32.resize(k);
            }

            ਲਈ (size_t ਪੰਕਤੀ = 0; ਪੰਕਤੀ < n; ++ਪੰਕਤੀ) {
                ਅਚਲ uint8_t * ਪੰਕਤੀ_ਅੰਕੜੇ = static_cast<ਅਚਲ uint8_t *>(ਅੰਕੜੇ) + ਪੰਕਤੀ * ਪੰਕਤੀ_stride;

                ਜੇ (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_ਪੰਕਤੀ_q6_K(reinterpret_cast<ਅਚਲ ਖੰਡ_q6_K *>(ਪੰਕਤੀ_ਅੰਕੜੇ), ਪੰਕਤੀ_f32.ਅੰਕੜੇ(), k);
                }

                ਦਸ਼ਮਲਵ max_abs = 0.0f;
                ਜੇ (tensor->type == GGML_TYPE_Q6_K) {
                    ਲਈ (size_t ਰੇਖਿਕ_idx = 0; ਰੇਖਿਕ_idx < k; ++ਰੇਖਿਕ_idx) {
                        max_abs = std::max(max_abs, std::fabs(ਪੰਕਤੀ_f32[ਰੇਖਿਕ_idx]));
                    }
                } ਨਹੀਂਤਾਂ {
                    ਅਚਲ ਖੰਡ_q8_0 * ਪੰਕਤੀ_ਖੰਡਾਂ = reinterpret_cast<ਅਚਲ ਖੰਡ_q8_0 *>(ਪੰਕਤੀ_ਅੰਕੜੇ);
                    ਲਈ (size_t ਖੰਡ = 0; ਖੰਡ < k_ਖੰਡਾਂ; ++ਖੰਡ) {
                        ਅਚਲ ਖੰਡ_q8_0 & blk = ਪੰਕਤੀ_ਖੰਡਾਂ[ਖੰਡ];
                        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(blk.d);
                        ਲਈ (size_t l = 0; l < QK8_0; ++l) {
                            ਅਚਲ size_t ਰੇਖਿਕ_idx = ਖੰਡ * QK8_0 + l;
                            ਜੇ (ਰੇਖਿਕ_idx >= k) {
                                ਰੋਕੋ;
                            }
                            ਅਚਲ ਦਸ਼ਮਲਵ value = d * static_cast<ਦਸ਼ਮਲਵ>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                ਦਸ਼ਮਲਵ scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[ਪੰਕਤੀ] = scale;
                ਅਚਲ ਦਸ਼ਮਲਵ inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                ਜੇ (tensor->type == GGML_TYPE_Q6_K) {
                    ਲਈ (size_t ਰੇਖਿਕ_idx = 0; ਰੇਖਿਕ_idx < k; ++ਰੇਖਿਕ_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(ਪੰਕਤੀ_f32[ਰੇਖਿਕ_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[ਪੰਕਤੀ * k + ਰੇਖਿਕ_idx] = static_cast<int8_t>(q);
                    }
                } ਨਹੀਂਤਾਂ {
                    ਅਚਲ ਖੰਡ_q8_0 * ਪੰਕਤੀ_ਖੰਡਾਂ = reinterpret_cast<ਅਚਲ ਖੰਡ_q8_0 *>(ਪੰਕਤੀ_ਅੰਕੜੇ);
                    ਲਈ (size_t ਖੰਡ = 0; ਖੰਡ < k_ਖੰਡਾਂ; ++ਖੰਡ) {
                        ਅਚਲ ਖੰਡ_q8_0 & blk = ਪੰਕਤੀ_ਖੰਡਾਂ[ਖੰਡ];
                        ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(blk.d);
                        ਲਈ (size_t l = 0; l < QK8_0; ++l) {
                            ਅਚਲ size_t ਰੇਖਿਕ_idx = ਖੰਡ * QK8_0 + l;
                            ਜੇ (ਰੇਖਿਕ_idx >= k) {
                                ਰੋਕੋ;
                            }
                            ਅਚਲ ਦਸ਼ਮਲਵ value = d * static_cast<ਦਸ਼ਮਲਵ>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[ਪੰਕਤੀ * k + ਰੇਖਿਕ_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        ਜੇ (want_f32 && ਥਾਂ_ਕੁੱਲ > 0) {
            bias.resize(n, 0.0f);
        }

        ਲਈ (ਪੂਰਨ ਥਾਂ = 0; ਥਾਂ < ਥਾਂ_ਕੁੱਲ && ਥਾਂ < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++ਥਾਂ) {
            ਜੇ (!allow_fallback && ਥਾਂ > 0) {
                ਰੋਕੋ;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[ਥਾਂ];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            ਜੇ (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_ਮਾਪ_ex || !kernel) {
                ਜਾਰੀ;
            }

            ਅਚਲ size_t nr = kernel->get_nr();
            ਅਚਲ size_t kr = kernel->get_kr();
            ਅਚਲ size_t sr = kernel->get_sr();
            ਅਚਲ ggml_type rhs_type = kernels->rhs_type;
            ਜੇ (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            ਅਚਲ size_t ਖੰਡ_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            ਜੇ (ਖੰਡ_len == SIZE_MAX) {
                ਜਾਰੀ;
            }

            ਅਚਲ size_t packed_ਮਾਪ = rhs_info->packed_ਮਾਪ_ex(n, k, nr, kr, ਖੰਡ_len);
            ਅਚਲ size_t aligned_cursor = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = ਅਧਾਰ_ptr + aligned_cursor;

            ਜੇ (rhs_type == GGML_TYPE_Q4_0) {
                ਬਣਤਰ kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_ਬਿੰਦੂ = 1;
                params.rhs_zero_ਬਿੰਦੂ = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ਅਚਲ uint8_t *>(ਅੰਕੜੇ), ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ,
                                       dst_ptr, 0, &params);
            } ਨਹੀਂਤਾਂ ਜੇ (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       ਅੰਕੜੇ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, dst_ptr, 0, ਖਾਲੀ_ਸੂਚਕ);
            } ਨਹੀਂਤਾਂ ਜੇ (rhs_type == GGML_TYPE_Q8_0) {
                ਬਣਤਰ kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_ਬਿੰਦੂ = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.ਅੰਕੜੇ(), ਖਾਲੀ_ਸੂਚਕ, scales.ਅੰਕੜੇ(),
                                       dst_ptr, 0, &params);
            } ਨਹੀਂਤਾਂ ਜੇ (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       ਅੰਕੜੇ, bias.ਅੰਕੜੇ(), ਖਾਲੀ_ਸੂਚਕ,
                                       dst_ptr, 0, ਖਾਲੀ_ਸੂਚਕ);
            } ਨਹੀਂਤਾਂ {
                ਜਾਰੀ;
            }

            ਸਿਰਲੇਖ->offsets[ਸਿਰਲੇਖ->ਥਾਂ_ਗਿਣਤੀ] = aligned_cursor;
            ਸਿਰਲੇਖ->sizes[ਸਿਰਲੇਖ->ਥਾਂ_ਗਿਣਤੀ]   = packed_ਮਾਪ;
            ++ਸਿਰਲੇਖ->ਥਾਂ_ਗਿਣਤੀ;

            cursor = aligned_cursor + packed_ਮਾਪ;
        }

        ਜੇ (ਸਿਰਲੇਖ->ਥਾਂ_ਗਿਣਤੀ == 0) {
            ਸਿਰਲੇਖ->magic   = 0;
            ਸਿਰਲੇਖ->ਸੰਸਕਰਣ_2 = 0;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(tensor->ਅੰਕੜੇ, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
        }

        ਵਾਪਸ 0;
    }
};

ਸਥਿਰ ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_ਬਫਰ_t, ਬਣਤਰ ggml_tensor *) {
    ਸਥਿਰ tensor_traits traits;
    ਵਾਪਸ &traits;
}
}  // namespace ggml::cpu::kleidiai

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_cpu_kleidiai_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor) {
    tensor->extra = (ਖਾਲੀ *) ggml::cpu::kleidiai::get_tensor_traits(ਬਫਰ, tensor);

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_kleidiai_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor,
                                                       ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(ਵਿੱਥ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ਸਵੈਚਾਲਿਤ tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    ਸਵੈਚਾਲਿਤ OK            = tensor_traits->repack(tensor, ਅੰਕੜੇ, size);

    GGML_ASSERT(OK == 0);
    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cpu_kleidiai_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ਅਣਵਰਤਿਆ(buft);
    ਵਾਪਸ "CPU_KLEIDIAI";
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_cpu_kleidiai_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_buft_alloc_ਬਫਰ(ggml_backend_cpu_ਬਫਰ_type(), size);

    ਜੇ (ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਬਫਰ->buft              = buft;
    ਬਫਰ->iface.init_tensor = ggml_backend_cpu_kleidiai_ਬਫਰ_init_tensor;
    ਬਫਰ->iface.ਸੈੱਟ_tensor  = ggml_backend_cpu_kleidiai_ਬਫਰ_ਸੈੱਟ_tensor;
    ਬਫਰ->iface.get_tensor  = ਖਾਲੀ_ਸੂਚਕ;
    ਬਫਰ->iface.cpy_tensor  = ਖਾਲੀ_ਸੂਚਕ;
    ਵਾਪਸ ਬਫਰ;
}

ਸਥਿਰ size_t ggml_backend_cpu_kleidiai_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ਅਣਵਰਤਿਆ(buft);
    ਵਾਪਸ TENSOR_ALIGNMENT;
}

ਸਥਿਰ size_t ggml_backend_cpu_kleidiai_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ਅਣਵਰਤਿਆ(buft);

    ਜੇ (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        ਵਾਪਸ ggml_nbytes(tensor);
    }

    ਅਚਲ size_t n = tensor->ne[1];
    ਅਚਲ size_t k = tensor->ne[0];

    size_t cursor = ਆਕਾਰ(kleidiai_ਭਾਰ_ਸਿਰਲੇਖ);
    cursor = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ਅਚਲ ਤਾਰਕਿਕ want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ਅਚਲ ਤਾਰਕਿਕ want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ਅਚਲ ਤਾਰਕਿਕ want_f32 = tensor->type == GGML_TYPE_F32;
    ਅਚਲ ਪੂਰਨ ਥਾਂ_ਕੁੱਲ = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ਅਚਲ ਤਾਰਕਿਕ allow_fallback = kleidiai_pack_fallback_allowed();

    size_t ਥਾਂ_ਗਿਣਤੀ = 0;
    ਲਈ (ਪੂਰਨ ਥਾਂ = 0; ਥਾਂ < ਥਾਂ_ਕੁੱਲ; ++ਥਾਂ) {
        ਜੇ (!allow_fallback && ਥਾਂ > 0) {
            ਰੋਕੋ;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[ਥਾਂ];
        ਜੇ (!kernels) {
            ਜਾਰੀ;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        ਜੇ (!kernel || !rhs_info || !rhs_info->packed_ਮਾਪ_ex) {
            ਜਾਰੀ;
        }

        ਅਚਲ ggml_type rhs_type = kernels->rhs_type;
        ਅਚਲ size_t ਖੰਡ_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        ਜੇ (ਖੰਡ_len == SIZE_MAX) {
            ਜਾਰੀ;
        }

        cursor = align_ਉੱਪਰ(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_ਮਾਪ_ex(n, k, kernel->get_nr(), kernel->get_kr(), ਖੰਡ_len);
        ++ਥਾਂ_ਗਿਣਤੀ;
    }

    ਜੇ (ਥਾਂ_ਗਿਣਤੀ == 0) {
        ਵਾਪਸ ggml_nbytes(tensor);
    }

    ਵਾਪਸ std::max(cursor, ggml_nbytes(tensor));
}

ਨਾਮਾਕਾਸ਼ ggml::cpu::kleidiai {
ਵਰਗ extra_ਬਫਰ_type : ggml::cpu::extra_ਬਫਰ_type {
    ਤਾਰਕਿਕ supports_op(ggml_backend_dev_t, ਅਚਲ ਬਣਤਰ ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        ਜੇ (op->op != GGML_OP_MUL_MAT) {
            ਵਾਪਸ ਝੂਠ;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ਅਚਲ ਪੂਰਨ ਥਾਂ_ਕੁੱਲ = kleidiai_collect_kernel_chain(op, kernel_chain);
        ਅਚਲ ਤਾਰਕਿਕ src0_is_kleidiai =
            op->src[0]->ਬਫਰ &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->ਬਫਰ->buft->context == ਇਹ &&
            ਥਾਂ_ਕੁੱਲ > 0;

        ਜੇ ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            ਜੇ (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == ਖਾਲੀ_ਸੂਚਕ) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == ਖਾਲੀ_ਸੂਚਕ) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == ਖਾਲੀ_ਸੂਚਕ) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == ਖਾਲੀ_ਸੂਚਕ) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == ਖਾਲੀ_ਸੂਚਕ) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (op->src[1]->ਬਫਰ && !ggml_backend_buft_is_host(op->src[1]->ਬਫਰ->buft)) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਜੇ (op->src[0]->type == GGML_TYPE_Q4_K) {
                ਜੇ (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਵਾਪਸ ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            }

            ਜੇ (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                ਜੇ (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    ਵਾਪਸ ਸੱਚ;
                }
                ਵਾਪਸ ਝੂਠ;
            }

            ਜੇ (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਵਾਪਸ ਸੱਚ;
        }

        ਵਾਪਸ ਝੂਠ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ਅਚਲ ਬਣਤਰ ggml_tensor * op) override {
        ਜੇ (op->op == GGML_OP_MUL_MAT) {
            ਜੇ (op->src[0]->ਬਫਰ && op->src[0]->ਬਫਰ->buft->context == ਇਹ) {
                ਵਾਪਸ (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } ਨਹੀਂਤਾਂ {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                ਜੇ (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    ਸਥਿਰ std::atomic<ਤਾਰਕਿਕ> warned(ਝੂਠ);
                    ਜੇ (!warned.exchange(ਸੱਚ)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                ਜੇ (op->src[0]->type != GGML_TYPE_F16) {
                    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ਅਚਲ ਪੂਰਨ ਥਾਂ_ਕੁੱਲ = kleidiai_collect_kernel_chain(op, kernel_chain);
                ਜੇ (ਥਾਂ_ਕੁੱਲ > 0 && op->src[1]->ne[1] > 1) {
                    ਜੇ ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
                    }
                    ਵਾਪਸ ggml::cpu::kleidiai::get_tensor_traits(ਨਿਰਾ, ਨਿਰਾ);
                }
            }
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_ਬਫਰ_type_t ggml_backend_cpu_kleidiai_ਬਫਰ_type(ਖਾਲੀ) {
    ਸਥਿਰ ggml::cpu::kleidiai::extra_ਬਫਰ_type ctx;
    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_cpu_ਬਫਰ_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_ਬਫਰ_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_ਬਫਰ_type_alloc_ਬਫਰ,
                           /* .alloc_buffer_n       = */ ਖਾਲੀ_ਸੂਚਕ,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_ਬਫਰ_type_get_alignment,
                           /* .get_max_size         = */ ਖਾਲੀ_ਸੂਚਕ,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_ਬਫਰ_type_get_alloc_ਮਾਪ,
                           /* .get_alloc_size_n     = */ ਨਿਰਾ,
                           /* .is_host              = */ ਖਾਲੀ_ਸੂਚਕ,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    ਵਾਪਸ &ggml_backend_cpu_ਬਫਰ_type_kleidiai;
}
