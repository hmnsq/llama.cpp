// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#ଅନ୍ତର୍ଭାଵ <arm_neon.h>
#ଅନ୍ତର୍ଭାଵ <cassert>
#ଅନ୍ତର୍ଭାଵ <cstdio>
#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <atomic>
#ଅନ୍ତର୍ଭାଵ <cfloat>
#ଅନ୍ତର୍ଭାଵ <cctype>
#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <stdexcept>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <vector>
#ଅନ୍ତର୍ଭାଵ <array>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <fstream>
#ଅନ୍ତର୍ଭାଵ <map>
#ଅନ୍ତର୍ଭାଵ <iostream>
#ଅନ୍ତର୍ଭାଵ <climits>
#ଅନ୍ତର୍ଭାଵ <charconv>
#ଅନ୍ତର୍ଭାଵ <system_error>
#ଯଦି defined(__linux__)
#ଅନ୍ତର୍ଭାଵ <asm/hwcap.h>
#ଅନ୍ତର୍ଭାଵ <dirent.h>
#ଅନ୍ତର୍ଭାଵ <sys/auxv.h>
#ଅନ୍ତର୍ଭାଵ <sys/types.h>
#ଅନ୍ତର୍ଭାଵ <sys/stat.h>
#ଅନ୍ତର୍ଭାଵ <unistd.h>
#ନଚେତଯଦି defined(__APPLE__)
#ଅନ୍ତର୍ଭାଵ <sys/sysctl.h>
#ଅନ୍ତର୍ଭାଵ <sys/types.h>
#endif

#ଅନ୍ତର୍ଭାଵ "kleidiai.h"

#ଅନ୍ତର୍ଭାଵ "ggml-cpu.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-feats.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-quants.h"
#ଅନ୍ତର୍ଭାଵ "ggml-threading.h"
#ଅନ୍ତର୍ଭାଵ "traits.h"

#ଅନ୍ତର୍ଭାଵ "kernels.h"

#ଅନ୍ତର୍ଭାଵ "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"

ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
ସ୍ଥିର constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
ସ୍ଥିର constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
ସ୍ଥିର constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
ସ୍ଥିର constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

ସଂରଚନା ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା sme_thread_cap; // <= 0 means "SME disabled/unknown"
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_hint;    // <= 0 means "no hint"
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_multiplier;
} ସ୍ଥିର ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ is_sme_family(cpu_feature f) {
    ଫେରନ୍ତୁ (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର* cpu_feature_to_string(cpu_feature f) {
    ଯଦି (f == CPU_FEATURE_NONE) {
        ଫେରନ୍ତୁ "NONE";
    } ଅନ୍ୟଥା ଯଦି ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        ଫେରନ୍ତୁ "SME2";
    } ଅନ୍ୟଥା ଯଦି ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        ଫେରନ୍ତୁ "SME";
    } ଅନ୍ୟଥା ଯଦି ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        ଫେରନ୍ତୁ "SVE";
    }
    ଅନ୍ୟଥା ଯଦି ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        ଫେରନ୍ତୁ "I8MM";
    } ଅନ୍ୟଥା ଯଦି ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        ଫେରନ୍ତୁ "DOTPROD";
    }
    ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ "UNKNOWN";
    }
}

#ଯଦି defined(__linux__) && defined(__aarch64__)
ସ୍ଥିର ତାର୍କିକ parse_cpu_dir_name(ଅଚଳ ଅକ୍ଷର* name, size_t* cpu) {
    ଯଦି (strncmp(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର* first = name + 3;
    ଅଚଳ ଅକ୍ଷର* last = name + strlen(name);

    size_t value = 0;
    ଅଚଳ auto [end, ec] = std::from_chars(first, last, value, 10);

    ଯଦି (ec != std::errc{} || end != last) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    *cpu = value;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    ଯଦି (dir == nullptr) {
        ଫେରନ୍ତୁ cpus;
    }

    ଯେତେବେଳେ (dirent * entry = readdir(dir)) {
        size_t cpu = 0;
        ଯଦି (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    ଫେରନ୍ତୁ cpus;
}
#endif

#ଯଦି defined(__APPLE__) && defined(__aarch64__)
ସ୍ଥିର ତାର୍କିକ apple_sme_counted_perf_level(std::string name) {
    ପାଇଁ (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (ଅକ୍ଷର) std::tolower((unsigned ଅକ୍ଷର) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    ଫେରନ୍ତୁ name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

ସ୍ଥିର ଶୂନ୍ୟ add_smcus_from_smidr(uint64_t smidr, size_t & ଅଙ୍କ_private, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ଅଚଳ uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ଅଚଳ uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ଅଚଳ size_t shared_ଗଣନା = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ଅଚଳ uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ଅଚଳ uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ଅଚଳ uint32_t id = (affinity2 << 12) | affinity;

    ଯଦି (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    ବାଛ (sh) {
        ସ୍ଥିତି 2: // private SMCU
            ++ଅଙ୍କ_private;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 3: // shared SMCU
            ଯଦି (shared_counts[id] < shared_ଗଣନା) {
                shared_counts[id] = shared_ଗଣନା;
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 0:
            ଯଦି (id == 0) {
                ++ଅଙ୍କ_private;
            } ଅନ୍ୟଥା ଯଦି (shared_counts[id] < shared_ଗଣନା) {
                shared_counts[id] = shared_ଗଣନା;
            }
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଭାଙ୍ଗ;
    }
}

ସ୍ଥିର size_t detect_ଅଙ୍କ_smcus() {
    ଅଚଳ auto runtime_feat = ggml_feats_get_arch64_runtime();
    ଯଦି (!runtime_feat.has_sme) {
        ଫେରନ୍ତୁ 0;
    }

#ଯଦି defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t ଅଙ୍କ_private = 0;
    std::map<uint32_t, size_t> shared_counts;

    ଅଚଳ std::vector<size_t> cpus = detect_cpu_ids();
    ପାଇଁ (ଅଚଳ size_t cpu : cpus) {
        ଅଚଳ std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        ଯଦି (!file.is_open()) {
            ଜାରି;
        }

        uint64_t smidr = 0;
        ଯଦି (!(file >> std::hex >> smidr)) {
            ଜାରି;
        }

        add_smcus_from_smidr(smidr, ଅଙ୍କ_private, shared_counts);
    }

    size_t total = ଅଙ୍କ_private;
    ପାଇଁ (ଅଚଳ auto & entry : shared_counts) {
        total += entry.second;
    }
    ଫେରନ୍ତୁ total;

#ନଚେତଯଦି defined(__APPLE__) && defined(__aarch64__)
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା perf_levels = 0;
    size_t size = ଆକାର(perf_levels);
    ଯଦି (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != ଆକାର(perf_levels) || perf_levels <= 0) {
        ଫେରନ୍ତୁ 0;
    }

    size_t units = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < perf_levels; ++i) {
        ଅକ୍ଷର ଚାବି[64] = {};
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା physical_cpus = 0;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpus_per_l2 = 0;

        snprintf(ଚାବି, ଆକାର(ଚାବି), "hw.perflevel%d.physicalcpu", i);
        size = ଆକାର(physical_cpus);
        ଯଦି (sysctlbyname(ଚାବି, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != ଆକାର(physical_cpus) || physical_cpus <= 0) {
            ଜାରି;
        }

        snprintf(ଚାବି, ଆକାର(ଚାବି), "hw.perflevel%d.cpusperl2", i);
        size = ଆକାର(cpus_per_l2);
        ଯଦି (sysctlbyname(ଚାବି, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != ଆକାର(cpus_per_l2) || cpus_per_l2 <= 0) {
            ଜାରି;
        }

        snprintf(ଚାବି, ଆକାର(ଚାବି), "hw.perflevel%d.name", i);
        size = 0;
        ଯଦି (sysctlbyname(ଚାବି, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            ଜାରି;
        }

        std::string name(size, '\0');
        ଯଦି (sysctlbyname(ଚାବି, &name[0], &size, nullptr, 0) != 0) {
            ଜାରି;
        }
        name.resize(size);
        ଯେତେବେଳେ (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        ଯଦି (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    ଫେରନ୍ତୁ units;

#ନଚେତଯଦି defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    ଫେରନ୍ତୁ 0;

#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ 0;
#endif
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା parse_uint_env(ଅଚଳ ଅକ୍ଷର *s, ଅଚଳ ଅକ୍ଷର *name, ତାର୍କିକ *ok) {
    ଯଦି (!s) { *ok = ମିଥ୍ୟା; ଫେରନ୍ତୁ 0; }
    ଅକ୍ଷର *end = nullptr;
    long v = strtol(s, &end, 10);
    ଯଦି (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = ମିଥ୍ୟା;
        ଫେରନ୍ତୁ 0;
    }
    ଯଦି (v < 0 || v > INT_MAX) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = ମିଥ୍ୟା;
        ଫେରନ୍ତୁ 0;
    }
    *ok = ସତ୍ୟ;
    ଫେରନ୍ତୁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)v;
}

ସ୍ଥିର ଶୂନ୍ୟ init_kleidiai_context(ଶୂନ୍ୟ) {
    ggml_critical_section_start();
    ସ୍ଥିର ତାର୍କିକ initialized = ମିଥ୍ୟା;

    ଯଦି (!initialized) {
        initialized = ସତ୍ୟ;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ଅଚଳ ଅକ୍ଷର *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ଅଚଳ ଅକ୍ଷର *env_threads     = getenv("GGML_TOTAL_THREADS");
        ଅଚଳ ଅକ୍ଷର *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ଅଚଳ auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        ଯଦି (env_threads) {
            ତାର୍କିକ ok = ମିଥ୍ୟା;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            ଯଦି (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        ଯଦି (env_chunk_mult) {
            ତାର୍କିକ ok = ମିଥ୍ୟା;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            ଯଦି (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା sme_cores = 0;
        ତାର୍କିକ sme_env_ok = ମିଥ୍ୟା;
        ତାର୍କିକ sme_env_set = (env_sme != nullptr);

        ଅଚଳ ତାର୍କିକ has_supported_sme_family = runtime_feat.has_sme;
        ତାର୍କିକ sme_cap_detected = ମିଥ୍ୟା;

        ଯଦି (has_supported_sme_family) {
            detected_smcus = detect_ଅଙ୍କ_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)detected_smcus : 1;

            ଯଦି (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        ଯଦି (sme_env_set) {
            ତାର୍କିକ ok = ମିଥ୍ୟା;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            ଯଦି (ok) {
                ଯଦି (has_supported_sme_family) {
                    sme_cores = v;
                } ଅନ୍ୟଥା {
                    ଯଦି (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } ଅନ୍ୟଥା {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        ଯଦି (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            ଯଦି (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        ଯଦି (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ctx.features);
        } ଅନ୍ୟଥା {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        ଯଦି (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ctx.features);
        } ଅନ୍ୟଥା {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        ଯଦି (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ctx.features);
        } ଅନ୍ୟଥା {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        ଯଦି (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ctx.features);
        } ଅନ୍ୟଥା {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        ଅଚଳ ତାର୍କିକ has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        ଯଦି (has_selected_sme_family_kernel) {
            ଯଦି (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } ଅନ୍ୟଥା ଯଦି (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } ଅନ୍ୟଥା {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } ଅନ୍ୟଥା {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_end();
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kleidiai_sme_thread_cap() {
    ଫେରନ୍ତୁ ctx.sme_thread_cap;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t align_up(size_t value, size_t alignment) {
    ଯଦି (alignment == 0) {
        ଫେରନ୍ତୁ value;
    }
    ଅଚଳ size_t remainder = value % alignment;
    ଫେରନ୍ତୁ remainder == 0 ? value : value + (alignment - remainder);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t gcd_size(size_t a, size_t b) {
    ଯେତେବେଳେ (b != 0) {
        ଅଚଳ size_t t = a % b;
        a = b;
        b = t;
    }
    ଫେରନ୍ତୁ a;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ lcm_size(size_t a, size_t b, size_t & result) {
    ଯଦି (a == 0 || b == 0) {
        result = 0;
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ size_t g = gcd_size(a, b);
    ଅଚଳ size_t q = a / g;
    ଯଦି (q > SIZE_MAX / b) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    result = q * b;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t ceil_div_size(size_t a, size_t b) {
    ଫେରନ୍ତୁ b == 0 ? 0 : (a + b - 1) / b;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t kleidiai_chunk_cols(size_t n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth_total, ତାର୍କିକ disable_chunking, size_t n_step) {
    ଅଚଳ size_t multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ଅଚଳ size_t ଭାଜକ = std::max<size_t>(1, (size_t) nth_total * multiplier);
    ଅଚଳ size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, ଭାଜକ)), n_step);
    ଫେରନ୍ତୁ chunk_cols ? chunk_cols : n_step;
}

ସଂରଚନା kleidiai_block_args {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    ବାଛ (rhs_type) {
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            ଫେରନ୍ତୁ { QK4_0, QK4_0, QK4_0 };
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            ଫେରନ୍ତୁ { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            ଫେରନ୍ତୁ { 0, 0, QK8_0 };
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ { 0, 0, 0 };
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ kleidiai_pack_fallback_allowed() {
    ଯଦି (ctx.sme_thread_cap <= 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (ctx.thread_hint <= 0) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଫେରନ୍ତୁ ctx.thread_hint > ctx.sme_thread_cap;
}

ସଂରଚନା kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_ଗଣନା;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ kleidiai_weight_header * kleidiai_weight_header_from_ptr(ଶୂନ୍ୟ * data) {
    ଫେରନ୍ତୁ reinterpret_cast<kleidiai_weight_header *>(data);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ kleidiai_weight_header * kleidiai_weight_header_from_ptr(ଅଚଳ ଶୂନ୍ୟ * data) {
    ଫେରନ୍ତୁ reinterpret_cast<ଅଚଳ kleidiai_weight_header *>(data);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ kleidiai_is_weight_header_valid(ଅଚଳ kleidiai_weight_header * header) {
    ଯଦି (!header) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (header->slot_ଗଣନା == 0 || header->slot_ଗଣନା > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot) {
    ଯଦି (!kleidiai_is_weight_header_valid(header)) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଯଦି (slot < 0 || slot >= header->slot_ଗଣନା) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଫେରନ୍ତୁ reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ uint8_t * kleidiai_weight_slot_ptr(ଅଚଳ kleidiai_weight_header * header, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot) {
    ଯଦି (!kleidiai_is_weight_header_valid(header)) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଯଦି (slot < 0 || slot >= header->slot_ଗଣନା) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଫେରନ୍ତୁ reinterpret_cast<ଅଚଳ uint8_t *>(header) + header->offsets[slot];
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    ଫେରନ୍ତୁ ctx.kernels_q4;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    ଫେରନ୍ତୁ ctx.kernels_q4k;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    ଫେରନ୍ତୁ ctx.kernels_q8;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    ଫେରନ୍ତୁ ctx.kernels_f32;
}

template <typename SelectFallback>
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା count = 0;
    ଯଦି (!primary) {
        ଫେରନ୍ତୁ 0;
    }
    out[count++] = primary;

    ଯଦି (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        ଫେରନ୍ତୁ count;
    }

    ଯଦି (is_sme_family(primary->required_cpu)) {
        ଅଚଳ cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        ଯଦି (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            ଯଦି (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    ଫେରନ୍ତୁ count;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା kleidiai_collect_kernel_chain(ଅଚଳ ସଂରଚନା ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    ଫେରନ୍ତୁ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ଫେରନ୍ତୁ ggml_kleidiai_select_kernels(mask, op); });
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    ଫେରନ୍ତୁ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ଫେରନ୍ତୁ ggml_kleidiai_select_kernels_q4_0(mask); });
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    ଫେରନ୍ତୁ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ଫେରନ୍ତୁ ggml_kleidiai_select_kernels_q4_k(mask); });
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    ଫେରନ୍ତୁ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ଫେରନ୍ତୁ ggml_kleidiai_select_kernels_q8_0(mask); });
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    ଫେରନ୍ତୁ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ଫେରନ୍ତୁ ggml_kleidiai_select_kernels_f32(mask); });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int64_t ggml_ne(ଅଚଳ ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    ଫେରନ୍ତୁ tensor->ne[dim];
}

ନାମାକାଶ ggml::cpu::kleidiai {

ସ୍ଥିର size_t round_down(size_t x, size_t y) {
    ଫେରନ୍ତୁ y == 0 ? x : x - (x % y);
}

ସ୍ଥିର ଶୂନ୍ୟ transpose_f32kxn_f16nxk(size_t n, size_t k, ଦଶମିକ * dst, ଅଚଳ uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / ଆକାର(uint16_t);
    size_t dst_stride = n;

    ପାଇଁ (size_t k_idx = 0; k_idx < k; ++k_idx) {
        ପାଇଁ (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

ଶ୍ରେଣୀ tensor_traits : ସର୍ବସାଧାରଣ ggml::cpu::tensor_traits {
    ତାର୍କିକ work_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା /* n_threads */, ଅଚଳ ସଂରଚନା ggml_tensor * op, size_t & size) override {
        ଯଦି (op->op != GGML_OP_MUL_MAT) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot_ଗଣନା = kleidiai_collect_kernel_chain(op, kernel_chain);
        ଯଦି (slot_ଗଣନା == 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ତାର୍କିକ is_gemv = op->src[1]->ne[1] == 1;
        ଅଚଳ size_t k = op->src[0]->ne[0];
        ଅଚଳ size_t n = op->src[0]->ne[1];
        ଅଚଳ size_t m = op->src[1]->ne[1];

        ଯଦି (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ଅଚଳ kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            ଅଚଳ size_t qk = block_args.pack_bl;

            size_t cursor = 0;
            ତାର୍କିକ any_slot = ମିଥ୍ୟା;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot = 0; slot < slot_ଗଣନା; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                ଯଦି (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }

                ଅଚଳ size_t mr = kernel->get_mr();
                ଅଚଳ size_t kr = kernel->get_kr();
                ଅଚଳ size_t sr = kernel->get_sr();

                ଅଚଳ size_t packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = ସତ୍ୟ;
            }

            ଯଦି (!any_slot) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            size = cursor;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଯଦି (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            ଯଦି (is_gemv && op->src[1]->nb[0] == (int64_t) ଆକାର(ଦଶମିକ) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }

            size_t cursor = 0;
            ତାର୍କିକ any_slot = ମିଥ୍ୟା;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot = 0; slot < slot_ଗଣନା; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                ଯଦି (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }

                ଅଚଳ size_t mr = kernel->get_mr();
                ଅଚଳ size_t kr = kernel->get_kr();
                ଅଚଳ size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
                any_slot = ସତ୍ୟ;
            }

            ଯଦି (!any_slot) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            size = cursor;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଯଦି (op->src[0]->type == GGML_TYPE_F16) {
            ଅଚଳ int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ଅଚଳ int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ଅଚଳ int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            ତାର୍କିକ any_slot = ମିଥ୍ୟା;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot = 0; slot < slot_ଗଣନା; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                ଯଦି (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }

                ଅଚଳ size_t mr = kernel->get_mr();
                ଅଚଳ size_t kr = kernel->get_kr();
                ଅଚଳ size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = ସତ୍ୟ;
            }

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot = 0; slot < slot_ଗଣନା; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                ଯଦି (!kernel || !kernels->rhs_info.packed_size_ex) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * ଆକାର(ଦଶମିକ);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * ଆକାର(ଦଶମିକ);

            ଯଦି (!any_slot) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            size = cursor;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ତାର୍କିକ compute_forward(ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst) override {
        ଯଦି (dst->op == GGML_OP_MUL_MAT) {
            ଯଦି (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                ଫେରନ୍ତୁ compute_forward_qx(params, dst);
            } ଅନ୍ୟଥା ଯଦି (dst->src[0]->type == GGML_TYPE_F32) {
                ଫେରନ୍ତୁ compute_forward_f32(params, dst);
            } ଅନ୍ୟଥା ଯଦି (dst->src[0]->type == GGML_TYPE_F16) {
                ଫେରନ୍ତୁ compute_forward_fp16(params, dst);
            }
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ତାର୍କିକ compute_forward_f32(ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        ଅଚଳ ggml_tensor * src0 = dst->src[0];
        ଅଚଳ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

        ଯଦି (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        ଯଦି (!kernels) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ size_t k = ne00;
        ଅଚଳ size_t m = ne11;
        ଅଚଳ size_t n = ne01;
        ଅଚଳ ତାର୍କିକ use_gemv = m == 1 && src1->nb[0] == (int64_t) ଆକାର(ଦଶମିକ) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        ଯଦି (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_size_ex || !lhs_info->pack_func_ex)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ଅଚଳ ତାର୍କିକ has_header = kleidiai_is_weight_header_valid(header);

        ଅଚଳ uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<ଅଚଳ uint8_t *>(src0->data);
        ଯଦି (!rhs_base) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth > 0 ? params->nth : 1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;

        ଅଚଳ size_t mr = kernel->get_mr();
        ଅଚଳ size_t kr = kernel->get_kr();
        ଅଚଳ size_t sr = kernel->get_sr();

        ଅଚଳ size_t lhs_packed_size = use_gemv ? 0 : lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        ଯଦି (!use_gemv) {
            GGML_ASSERT(lhs_packed_size <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ଅଚଳ size_t dst_stride = dst->nb[1];
        ଅଚଳ size_t n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        ଅଚଳ ତାର୍କିକ disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) INT_MAX);

        ଅଚଳ int64_t batch_size = ne12 * ne13;
        ପାଇଁ (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ଅଚଳ int64_t i12 = batch_idx % ne12;
            ଅଚଳ int64_t i13 = batch_idx / ne12;
            ଅଚଳ uint8_t * lhs_batch_base = static_cast<ଅଚଳ uint8_t *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];

            ଯଦି (!use_gemv) {
                ଅଚଳ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                ଅଚଳ int64_t use_threads = std::min<int64_t>(nth, max_threads);

                ଯଦି (ith < use_threads) {
                    ଅଚଳ int64_t ଅଙ୍କ_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), mr);
                    ଅଚଳ int64_t ଅଙ୍କ_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * ଅଙ୍କ_m_per_thread0;

                    ଅଚଳ int64_t m_start = (int64_t)ith * ଅଙ୍କ_m_per_thread0;
                    ଅଚଳ int64_t m_ଗଣନା = (ith == use_threads - 1) ? ଅଙ୍କ_m_per_thread_N_1 : ଅଙ୍କ_m_per_thread0;

                    ଅଚଳ size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ଅଚଳ size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ଅଚଳ size_t row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_ଗଣନା;
                    int64_t cur       = m_start;

                    ଯେତେବେଳେ (remaining > 0) {
                        ଅଚଳ int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        ଅଚଳ size_t src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        ଅଚଳ ଶୂନ୍ୟ * src_ptr = lhs_batch_base + src_off;
                        ଅଚଳ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        ଶୂନ୍ୟ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            ଯଦି (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ଅଚଳ size_t chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (size_t) INT_MAX);

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା current_col = ggml_threadpool_chunk_add(params->threadpool, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) chunk_cols);
            ଯେତେବେଳେ ((size_t) current_col < n) {
                ଅଚଳ size_t n_start = (size_t) current_col;
                ଅଚଳ size_t n_to_process = std::min(chunk_cols, n - n_start);

                ଯଦି (n_to_process > 0) {
                    ଅଚଳ size_t lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ଅଚଳ size_t rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ଅଚଳ size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    ଅଚଳ ଶୂନ୍ୟ * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    ଅଚଳ ଶୂନ୍ୟ * rhs_ptr = rhs_base + rhs_packed_offset;
                    ଦଶମିକ * dst_ptr      = reinterpret_cast<ଦଶମିକ *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          ଆକାର(ଦଶମିକ),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) chunk_cols);
            }

            ଯଦି (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ତାର୍କିକ compute_forward_fp16(ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst) {
        ଅଚଳ ggml_tensor * src0 = dst->src[0];
        ଅଚଳ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        ଯଦି (!kernels) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ତାର୍କିକ is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        ଯଦି (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;

        ଅଚଳ int64_t lhs_batch_size0 = ne12;
        ଅଚଳ int64_t rhs_batch_size0 = ne02;
        ଅଚଳ int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        ଅଚଳ int64_t r = lhs_batch_size0 / rhs_batch_size0;

        ଅଚଳ int64_t m_group = ne11;
        ଅଚଳ int64_t m       = m_group;
        ଅଚଳ int64_t n       = ne01;
        ଅଚଳ int64_t k       = ne00;

        ଅଚଳ size_t lhs_stride = src1->nb[1];
        ଅଚଳ size_t rhs_stride = src0->nb[1];
        ଅଚଳ size_t dst_stride = dst->nb[1];

        ଅଚଳ int64_t mr = (int64_t) kernel->get_mr();
        ଅଚଳ int64_t nr = (int64_t) kernel->get_nr();
        ଅଚଳ int64_t kr = (int64_t) kernel->get_kr();
        ଅଚଳ int64_t sr = (int64_t) kernel->get_sr();

        ଅଚଳ size_t lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        ଅଚଳ size_t rhs_packed_size = kernels->rhs_info.packed_size_ex(n, k, nr, kr, 0);
        ଅଚଳ size_t kxn_size        = k * n * ଆକାର(ଦଶମିକ);
        ଅଚଳ size_t bias_size       = n * ଆକାର(ଦଶମିକ);

        ଅଚଳ size_t wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        ପାଇଁ (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ଅଚଳ int64_t rhs_batch_idx = batch_idx / r;
            ଅଚଳ uint8_t * rhs_batch_base = static_cast<ଅଚଳ uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ଅଚଳ int64_t m_roundup_mr = kai_roundup(m, mr);
                ଅଚଳ int64_t ଅଙ୍କ_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                ଯଦି (ith < ଅଙ୍କ_threads) {
                    ଅଚଳ int64_t ଅଙ୍କ_m_per_thread0   = round_down((size_t)(m_roundup_mr / ଅଙ୍କ_threads), (size_t)mr);
                    ଅଚଳ int64_t ଅଙ୍କ_m_per_thread_N_1 = m - (ଅଙ୍କ_threads - 1) * ଅଙ୍କ_m_per_thread0;

                    ଅଚଳ int64_t m_start = ith * ଅଙ୍କ_m_per_thread0;
                    ଅଚଳ int64_t m_ଗଣନା = (ith == ଅଙ୍କ_threads - 1) ? ଅଙ୍କ_m_per_thread_N_1 : ଅଙ୍କ_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ଅଚଳ size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ଅଚଳ size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ଅଚଳ size_t row_stride_bytes = (next_block_off - base_packed_off) / (size_t)mr;

                    int64_t remaining = m_ଗଣନା;
                    int64_t cur       = m_start;

                    ଯେତେବେଳେ (remaining > 0) {
                        ଅଚଳ int64_t row_in_group = cur;
                        ଅଚଳ int64_t avail        = m_group - row_in_group;
                        ଅଚଳ int64_t take         = std::min(avail, remaining);

                        ଅଚଳ uint8_t * lhs_batch_base = static_cast<ଅଚଳ uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        ଅଚଳ ଶୂନ୍ୟ * src_ptr = lhs_batch_base + (size_t)row_in_group * lhs_stride;
                        ଅଚଳ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        ଶୂନ୍ୟ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            ଯଦି (ith == 0) {
                memset(bias, 0, (size_t)n * ଆକାର(ଦଶମିକ));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<ଦଶମିକ *>(rhs_kxn),
                                        reinterpret_cast<ଅଚଳ uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * ଆକାର(ଦଶମିକ),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ଅଚଳ int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t ଅଙ୍କ_threads_n = KAI_MIN(n / n_step, nth);
                ଯଦି (ଅଙ୍କ_threads_n <= 0) {
                    ଅଙ୍କ_threads_n = 1;
                }

                ଯଦି (ith < ଅଙ୍କ_threads_n) {
                    ଅଚଳ int64_t ଅଙ୍କ_n_per_thread0   = round_down((size_t)(n / ଅଙ୍କ_threads_n), (size_t)n_step);
                    ଅଚଳ int64_t ଅଙ୍କ_n_per_thread_N_1 = n - (ଅଙ୍କ_threads_n - 1) * ଅଙ୍କ_n_per_thread0;

                    ଅଚଳ int64_t n_start      = ith * ଅଙ୍କ_n_per_thread0;
                    ଅଚଳ int64_t n_to_process = (ith == ଅଙ୍କ_threads_n - 1) ? ଅଙ୍କ_n_per_thread_N_1 : ଅଙ୍କ_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ଅଚଳ size_t lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ଅଚଳ size_t rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ଅଚଳ size_t dst_offset         = kernel->get_dst_offset((size_t)0, (size_t)n_start, dst_stride);

                    ଅଚଳ ଶୂନ୍ୟ * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ଅଚଳ ଶୂନ୍ୟ * rhs_ptr = rhs_packed + rhs_packed_offset;
                    ଦଶମିକ * dst_ptr      = reinterpret_cast<ଦଶମିକ *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, ଆକାର(ଦଶମିକ), -FLT_MAX, FLT_MAX);
                }
            }

            ଯଦି (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ତାର୍କିକ compute_forward_qx(ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ଅଚଳ ggml_tensor * src0 = dst->src[0];
        ଅଚଳ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

        ଅଚଳ kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ଅଚଳ ତାର୍କିକ has_header = kleidiai_is_weight_header_valid(header);
        ଅଚଳ ତାର୍କିକ is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        auto weight_for_slot = [&](ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot_index, size_t & size_out) -> ଅଚଳ uint8_t * {
            ଯଦି (slot_index < 0 || slot_index >= slot_total) {
                ଫେରନ୍ତୁ nullptr;
            }
            ଯଦି (has_header) {
                ଯଦି (slot_index < header->slot_ଗଣନା) {
                    size_out = static_cast<size_t>(header->sizes[slot_index]);
                    ଫେରନ୍ତୁ kleidiai_weight_slot_ptr(header, slot_index);
                }
                ଫେରନ୍ତୁ nullptr;
            }
            ଯଦି (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                ଫେରନ୍ତୁ static_cast<ଅଚଳ uint8_t *>(src0->data);
            }
            ଫେରନ୍ତୁ nullptr;
        };

        ସଂରଚନା runtime_slot {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot_index;
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
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା assigned_threads;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_begin;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_end;
            ଅଚଳ uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା runtime_ଗଣନା = 0;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot = 0; slot < slot_total && runtime_ଗଣନା < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            ଯଦି (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                ଜାରି;
            }

            size_t rhs_size = 0;
            ଅଚଳ uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            ଯଦି (!rhs_ptr || rhs_size == 0) {
                ଜାରି;
            }

            ଅଚଳ kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

            runtime[runtime_ଗଣନା] = {
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
            ++runtime_ଗଣନା;
        }

        ଯଦି (runtime_ଗଣନା == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth_total = params->nth > 0 ? params->nth : 1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith_total = params->ith;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା sme_slot = -1;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା non_sme_slot = -1;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < runtime_ଗଣନା; ++i) {
            ଯଦି (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                ଭାଙ୍ଗ;
            }
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < runtime_ଗଣନା; ++i) {
            ଯଦି (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                ଭାଙ୍ଗ;
            }
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sme_cap_limit = ctx.sme_thread_cap;
        ଅଚଳ ତାର୍କିକ use_hybrid = sme_cap_limit > 0 &&
                                 runtime_ଗଣନା > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_cols_per_thread = 0;
        ଯଦି (runtime_ଗଣନା > 0 && nth_total > 0) {
            min_cols_per_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        ଅଚଳ ତାର୍କିକ too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        ଅଚଳ ତାର୍କିକ hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        ଯଦି (!hybrid_enabled) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା chosen_slot = 0;
            ଯଦି (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } ଅନ୍ୟଥା ଯଦି (runtime_ଗଣନା > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            ଯଦି (chosen_slot != 0 && chosen_slot < runtime_ଗଣନା) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_ଗଣନା = runtime_ଗଣନା > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            ଯଦି (runtime_ଗଣନା > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        ଅଚଳ ତାର୍କିକ sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା sme_cap = sme_single_only ? nth_total : kleidiai_sme_thread_cap();
        ଯଦି (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା threads_remaining = nth_total;
        ଯଦି (sme_slot != -1) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା fallback_ଗଣନା = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < runtime_ଗଣନା; ++i) {
            ଯଦି (i == sme_slot) {
                ଜାରି;
            }
            fallback_indices[fallback_ଗଣନା++] = i;
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା fi = 0; fi < fallback_ଗଣନା; ++fi) {
            ଯଦି (threads_remaining <= 0) {
                ଭାଙ୍ଗ;
            }
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot_index = fallback_indices[fi];
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା slots_ବାମ = fallback_ଗଣନା - fi;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା share = (threads_remaining + slots_ବାମ - 1) / slots_ବାମ;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_cursor = 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < runtime_ଗଣନା; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା active_threads = thread_cursor;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା local_slot = -1;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା local_ith  = 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < runtime_ଗଣନା; ++i) {
            ଯଦି (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                ଭାଙ୍ଗ;
            }
        }
        ଅଚଳ size_t k = ne00;
        ଅଚଳ size_t m = ne11;
        ଅଚଳ size_t n = ne01;

        size_t cursor = 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < runtime_ଗଣନା; ++i) {
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_step = 1;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < runtime_ଗଣନା; ++i) {
            ଯଦି (runtime[i].assigned_threads == 0) {
                ଜାରି;
            }
            size_t next_step = 0;
            ଯଦି (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        ଅଚଳ ତାର୍କିକ disable_chunking = ggml_is_numa();
        ଅଚଳ size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ଅଚଳ size_t chunk_ଭାଜକ = (active_threads == 1 || disable_chunking) ? (size_t) active_threads : (size_t) active_threads * chunk_multiplier;
        size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, chunk_ଭାଜକ)), common_step);
        ଯଦି (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ଅଚଳ size_t nchunk_size = std::max<size_t>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (size_t)INT_MAX);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nchunk = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)nchunk_size;
        ଅଚଳ size_t dst_stride = dst->nb[1];

        auto run_chunk = [&](runtime_slot & slot, size_t global_start, size_t cols, uint8_t * dst_batch_base) {
            ଅଚଳ size_t rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            ଅଚଳ size_t dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            ଅଚଳ uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            ଅଚଳ uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            ଦଶମିକ * dst_ptr         = reinterpret_cast<ଦଶମିକ *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       ଆକାର(ଦଶମିକ),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        ପାଇଁ (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ଅଚଳ uint8_t * lhs_batch_base = static_cast<ଅଚଳ uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            ଯଦି (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                ଅଚଳ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                ଅଚଳ int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                ଯଦି (local_ith < use_threads) {
                    ଅଚଳ int64_t ଅଙ୍କ_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), slot.mr);
                    ଅଚଳ int64_t ଅଙ୍କ_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * ଅଙ୍କ_m_per_thread0;

                    ଅଚଳ int64_t m_start = (int64_t)local_ith * ଅଙ୍କ_m_per_thread0;
                    ଅଚଳ int64_t m_ଗଣନା = (local_ith == use_threads - 1) ? ଅଙ୍କ_m_per_thread_N_1 : ଅଙ୍କ_m_per_thread0;

                    ଅଚଳ size_t base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ଅଚଳ size_t next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ଅଚଳ size_t row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_ଗଣନା;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    ଯେତେବେଳେ (remaining > 0) {
                        ଅଚଳ int64_t row_in_group = cur;
                        ଅଚଳ int64_t avail        = (int64_t)m - row_in_group;
                        ଅଚଳ int64_t take         = std::min(avail, remaining);

                        ଅଚଳ size_t src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        ଅଚଳ ଶୂନ୍ୟ * src_ptr = lhs_batch_base + src_off;
                        ଅଚଳ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        ଶୂନ୍ୟ * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            ଯଦି (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            ଯଦି (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା current_chunk = ith_total;
                ଯେତେବେଳେ (current_chunk < nchunk) {
                    ଅଚଳ size_t global_start = (size_t)current_chunk * chunk_cols;
                    ଯଦି (global_start >= n) {
                        ଭାଙ୍ଗ;
                    }

                    ଅଚଳ size_t cols = std::min(chunk_cols, n - global_start);
                    ଯଦି (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            ଯଦି (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

ସର୍ବସାଧାରଣ:
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା repack(ସଂରଚନା ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ଅଚଳ size_t n = tensor->ne[1];
        ଅଚଳ size_t k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        ଯଦି (!header) {
            ଫେରନ୍ତୁ -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_ଗଣନା = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        size_t cursor = ଆକାର(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ଅଚଳ ତାର୍କିକ want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ଅଚଳ ତାର୍କିକ want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ଅଚଳ ତାର୍କିକ want_f32 = tensor->type == GGML_TYPE_F32;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ଅଚଳ ତାର୍କିକ allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<ଦଶମିକ>  scales;
        std::vector<ଦଶମିକ>  bias;
        std::vector<ଦଶମିକ> row_f32;

        ଯଦି (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ଅଚଳ size_t row_stride = tensor->nb[1];
            ଅଚଳ size_t k_blocks   = (k + QK8_0 - 1) / QK8_0;

            ଯଦି (tensor->type == GGML_TYPE_Q6_K) {
                ଯଦି (k % QK_K != 0 || k % QK8_0 != 0) {
                    ଫେରନ୍ତୁ -1;
                }
                row_f32.resize(k);
            }

            ପାଇଁ (size_t row = 0; row < n; ++row) {
                ଅଚଳ uint8_t * row_data = static_cast<ଅଚଳ uint8_t *>(data) + row * row_stride;

                ଯଦି (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<ଅଚଳ block_q6_K *>(row_data), row_f32.data(), k);
                }

                ଦଶମିକ max_abs = 0.0f;
                ଯଦି (tensor->type == GGML_TYPE_Q6_K) {
                    ପାଇଁ (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::fabs(row_f32[linear_idx]));
                    }
                } ଅନ୍ୟଥା {
                    ଅଚଳ block_q8_0 * row_blocks = reinterpret_cast<ଅଚଳ block_q8_0 *>(row_data);
                    ପାଇଁ (size_t block = 0; block < k_blocks; ++block) {
                        ଅଚଳ block_q8_0 & blk = row_blocks[block];
                        ଅଚଳ ଦଶମିକ d = GGML_FP16_TO_FP32(blk.d);
                        ପାଇଁ (size_t l = 0; l < QK8_0; ++l) {
                            ଅଚଳ size_t linear_idx = block * QK8_0 + l;
                            ଯଦି (linear_idx >= k) {
                                ଭାଙ୍ଗ;
                            }
                            ଅଚଳ ଦଶମିକ value = d * static_cast<ଦଶମିକ>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                ଦଶମିକ scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                ଅଚଳ ଦଶମିକ inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                ଯଦି (tensor->type == GGML_TYPE_Q6_K) {
                    ପାଇଁ (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } ଅନ୍ୟଥା {
                    ଅଚଳ block_q8_0 * row_blocks = reinterpret_cast<ଅଚଳ block_q8_0 *>(row_data);
                    ପାଇଁ (size_t block = 0; block < k_blocks; ++block) {
                        ଅଚଳ block_q8_0 & blk = row_blocks[block];
                        ଅଚଳ ଦଶମିକ d = GGML_FP16_TO_FP32(blk.d);
                        ପାଇଁ (size_t l = 0; l < QK8_0; ++l) {
                            ଅଚଳ size_t linear_idx = block * QK8_0 + l;
                            ଯଦି (linear_idx >= k) {
                                ଭାଙ୍ଗ;
                            }
                            ଅଚଳ ଦଶମିକ value = d * static_cast<ଦଶମିକ>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        ଯଦି (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ଯଦି (!allow_fallback && slot > 0) {
                ଭାଙ୍ଗ;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            ଯଦି (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                ଜାରି;
            }

            ଅଚଳ size_t nr = kernel->get_nr();
            ଅଚଳ size_t kr = kernel->get_kr();
            ଅଚଳ size_t sr = kernel->get_sr();
            ଅଚଳ ggml_type rhs_type = kernels->rhs_type;
            ଯଦି (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            ଅଚଳ size_t block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            ଯଦି (block_len == SIZE_MAX) {
                ଜାରି;
            }

            ଅଚଳ size_t packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            ଅଚଳ size_t aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            ଯଦି (rhs_type == GGML_TYPE_Q4_0) {
                ସଂରଚନା kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ଅଚଳ uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } ଅନ୍ୟଥା ଯଦି (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       data, nullptr, nullptr, dst_ptr, 0, nullptr);
            } ଅନ୍ୟଥା ଯଦି (rhs_type == GGML_TYPE_Q8_0) {
                ସଂରଚନା kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } ଅନ୍ୟଥା ଯଦି (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
            } ଅନ୍ୟଥା {
                ଜାରି;
            }

            header->offsets[header->slot_ଗଣନା] = aligned_cursor;
            header->sizes[header->slot_ଗଣନା]   = packed_size;
            ++header->slot_ଗଣନା;

            cursor = aligned_cursor + packed_size;
        }

        ଯଦି (header->slot_ଗଣନା == 0) {
            header->magic   = 0;
            header->version = 0;
            memcpy(tensor->data, data, data_size);
        }

        ଫେରନ୍ତୁ 0;
    }
};

ସ୍ଥିର ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, ସଂରଚନା ggml_tensor *) {
    ସ୍ଥିର tensor_traits traits;
    ଫେରନ୍ତୁ &traits;
}
}  // namespace ggml::cpu::kleidiai

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, ସଂରଚନା ggml_tensor * tensor) {
    tensor->extra = (ଶୂନ୍ୟ *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, ସଂରଚନା ggml_tensor * tensor,
                                                       ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    ଫେରନ୍ତୁ "CPU_KLEIDIAI";
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    ଯଦି (buffer == nullptr) {
        ଫେରନ୍ତୁ nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    ଫେରନ୍ତୁ buffer;
}

ସ୍ଥିର size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    ଫେରନ୍ତୁ TENSOR_ALIGNMENT;
}

ସ୍ଥିର size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ସଂରଚନା ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    ଯଦି (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ggml_nbytes(tensor);
    }

    ଅଚଳ size_t n = tensor->ne[1];
    ଅଚଳ size_t k = tensor->ne[0];

    size_t cursor = ଆକାର(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ଅଚଳ ତାର୍କିକ want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ଅଚଳ ତାର୍କିକ want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ଅଚଳ ତାର୍କିକ want_f32 = tensor->type == GGML_TYPE_F32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ଅଚଳ ତାର୍କିକ allow_fallback = kleidiai_pack_fallback_allowed();

    size_t slot_ଗଣନା = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot = 0; slot < slot_total; ++slot) {
        ଯଦି (!allow_fallback && slot > 0) {
            ଭାଙ୍ଗ;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        ଯଦି (!kernels) {
            ଜାରି;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        ଯଦି (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            ଜାରି;
        }

        ଅଚଳ ggml_type rhs_type = kernels->rhs_type;
        ଅଚଳ size_t block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        ଯଦି (block_len == SIZE_MAX) {
            ଜାରି;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_ଗଣନା;
    }

    ଯଦି (slot_ଗଣନା == 0) {
        ଫେରନ୍ତୁ ggml_nbytes(tensor);
    }

    ଫେରନ୍ତୁ std::max(cursor, ggml_nbytes(tensor));
}

ନାମାକାଶ ggml::cpu::kleidiai {
ଶ୍ରେଣୀ extra_buffer_type : ggml::cpu::extra_buffer_type {
    ତାର୍କିକ supports_op(ggml_backend_dev_t, ଅଚଳ ସଂରଚନା ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        ଯଦି (op->op != GGML_OP_MUL_MAT) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        ଅଚଳ ତାର୍କିକ src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft->context == ଏହି &&
            slot_total > 0;

        ଯଦି ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            ଯଦି (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଯଦି (op->src[0]->type == GGML_TYPE_Q4_K) {
                ଯଦି (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
                ଫେରନ୍ତୁ ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == ଆକାର(ଦଶମିକ) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * ଆକାର(ଦଶମିକ));
            }

            ଯଦି (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                ଯଦି (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    ଫେରନ୍ତୁ ସତ୍ୟ;
                }
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଯଦି (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ଅଚଳ ସଂରଚନା ggml_tensor * op) override {
        ଯଦି (op->op == GGML_OP_MUL_MAT) {
            ଯଦି (op->src[0]->buffer && op->src[0]->buffer->buft->context == ଏହି) {
                ଫେରନ୍ତୁ (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } ଅନ୍ୟଥା {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                ଯଦି (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    ସ୍ଥିର std::atomic<ତାର୍କିକ> warned(ମିଥ୍ୟା);
                    ଯଦି (!warned.exchange(ସତ୍ୟ)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                ଯଦି (op->src[0]->type != GGML_TYPE_F16) {
                    ଫେରନ୍ତୁ nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                ଯଦି (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    ଯଦି ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        ଫେରନ୍ତୁ nullptr;
                    }
                    ଫେରନ୍ତୁ ggml::cpu::kleidiai::get_tensor_traits(ଖାଲି, ଖାଲି);
                }
            }
        }
        ଫେରନ୍ତୁ nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(ଶୂନ୍ୟ) {
    ସ୍ଥିର ggml::cpu::kleidiai::extra_buffer_type ctx;
    ସ୍ଥିର ସଂରଚନା ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .get_alloc_size_n     = */ ଖାଲି,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    ଫେରନ୍ତୁ &ggml_backend_cpu_buffer_type_kleidiai;
}
