// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#అన్తర్భావ <arm_neon.h>
#అన్తర్భావ <cassert>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <atomic>
#అన్తర్భావ <cfloat>
#అన్తర్భావ <cctype>
#అన్తర్భావ <algorithm>
#అన్తర్భావ <cmath>
#అన్తర్భావ <stdexcept>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <string>
#అన్తర్భావ <vector>
#అన్తర్భావ <array>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <fstream>
#అన్తర్భావ <map>
#అన్తర్భావ <iostream>
#అన్తర్భావ <climits>
#అన్తర్భావ <charconv>
#అన్తర్భావ <system_error>
#ఒకవేళ defined(__linux__)
#అన్తర్భావ <asm/hwcap.h>
#అన్తర్భావ <dirent.h>
#అన్తర్భావ <sys/auxv.h>
#అన్తర్భావ <sys/types.h>
#అన్తర్భావ <sys/stat.h>
#అన్తర్భావ <unistd.h>
#లేదావేళ defined(__APPLE__)
#అన్తర్భావ <sys/sysctl.h>
#అన్తర్భావ <sys/types.h>
#అయితే_ముగింపు

#అన్తర్భావ "kleidiai.h"

#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "ggml-cpu-impl.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-feats.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-quants.h"
#అన్తర్భావ "ggml-threading.h"
#అన్తర్భావ "traits.h"

#అన్తర్భావ "kernels.h"

#అన్తర్భావ "kai/kai_common.h"

#నిర్వచించు GGML_COMMON_DECL_CPP
#అన్తర్భావ "ggml-common.h"

స్థిర constexpr పూర్ణాంకం      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
స్థిర constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
స్థిర constexpr uint16_t GGML_KLEIDIAI_PACK_సంస్కరణ     = 1;
స్థిర constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
స్థిర constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

నిర్మాణం ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    పూర్ణాంకం sme_thread_cap; // <= 0 means "SME disabled/unknown"
    పూర్ణాంకం thread_hint;    // <= 0 means "no hint"
    పూర్ణాంకం chunk_multiplier;
} స్థిర ctx = { CPU_FEATURE_NONE, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, 0, -1, 4 };

స్థిర అంతఃవరుస తార్కిక is_sme_family(cpu_feature f) {
    తిరిగి (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

స్థిర స్థిరరాశి అక్షరం* cpu_feature_గమ్యం_string(cpu_feature f) {
    ఒకవేళ (f == CPU_FEATURE_NONE) {
        తిరిగి "NONE";
    } లేకపోతే ఒకవేళ ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        తిరిగి "SME2";
    } లేకపోతే ఒకవేళ ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        తిరిగి "SME";
    } లేకపోతే ఒకవేళ ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        తిరిగి "SVE";
    }
    లేకపోతే ఒకవేళ ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        తిరిగి "I8MM";
    } లేకపోతే ఒకవేళ ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        తిరిగి "DOTPROD";
    }
    లేకపోతే {
        తిరిగి "UNKNOWN";
    }
}

#ఒకవేళ defined(__linux__) && defined(__aarch64__)
స్థిర తార్కిక parse_cpu_dir_name(స్థిరరాశి అక్షరం* name, size_t* cpu) {
    ఒకవేళ (అక్షరశ్రేణి_పోలిక(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం* మొదటి = name + 3;
    స్థిరరాశి అక్షరం* చివరి = name + అక్షరశ్రేణి_పొడవు(name);

    size_t value = 0;
    స్థిరరాశి స్వయంచాలక [end, ec] = std::from_chars(మొదటి, చివరి, value, 10);

    ఒకవేళ (ec != std::errc{} || end != చివరి) {
        తిరిగి అబద్ధం;
    }

    *cpu = value;
    తిరిగి నిజం;
}

స్థిర std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    ఒకవేళ (dir == శూన్యసూచిక) {
        తిరిగి cpus;
    }

    అయితే (dirent * నమోదు_2 = readdir(dir)) {
        size_t cpu = 0;
        ఒకవేళ (parse_cpu_dir_name(నమోదు_2->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    తిరిగి cpus;
}
#అయితే_ముగింపు

#ఒకవేళ defined(__APPLE__) && defined(__aarch64__)
స్థిర తార్కిక apple_sme_counted_perf_level(std::string name) {
    కొరకు (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (అక్షరం) std::tolower((గుర్తులేని అక్షరం) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    తిరిగి name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#అయితే_ముగింపు

స్థిర శూన్య కూడు_smcus_from_smidr(uint64_t smidr, size_t & అంకె_వ్యక్తిగత, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    స్థిరరాశి uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    స్థిరరాశి uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    స్థిరరాశి size_t shared_లెక్క = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    స్థిరరాశి uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    స్థిరరాశి uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    స్థిరరాశి uint32_t id = (affinity2 << 12) | affinity;

    ఒకవేళ (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    ఎంపిక (sh) {
        స్థితి 2: // private SMCU
            ++అంకె_వ్యక్తిగత;
            ఆపు;
        స్థితి 3: // shared SMCU
            ఒకవేళ (shared_counts[id] < shared_లెక్క) {
                shared_counts[id] = shared_లెక్క;
            }
            ఆపు;
        స్థితి 0:
            ఒకవేళ (id == 0) {
                ++అంకె_వ్యక్తిగత;
            } లేకపోతే ఒకవేళ (shared_counts[id] < shared_లెక్క) {
                shared_counts[id] = shared_లెక్క;
            }
            ఆపు;
        అప్రమేయ:
            ఆపు;
    }
}

స్థిర size_t detect_అంకె_smcus() {
    స్థిరరాశి స్వయంచాలక runtime_feat = ggml_feats_get_arch64_runtime();
    ఒకవేళ (!runtime_feat.has_sme) {
        తిరిగి 0;
    }

#ఒకవేళ defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t అంకె_వ్యక్తిగత = 0;
    std::map<uint32_t, size_t> shared_counts;

    స్థిరరాశి std::vector<size_t> cpus = detect_cpu_ids();
    కొరకు (స్థిరరాశి size_t cpu : cpus) {
        స్థిరరాశి std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        ఒకవేళ (!file.is_open()) {
            కొనసాగు;
        }

        uint64_t smidr = 0;
        ఒకవేళ (!(file >> std::hex >> smidr)) {
            కొనసాగు;
        }

        కూడు_smcus_from_smidr(smidr, అంకె_వ్యక్తిగత, shared_counts);
    }

    size_t మొత్తం = అంకె_వ్యక్తిగత;
    కొరకు (స్థిరరాశి స్వయంచాలక & నమోదు_2 : shared_counts) {
        మొత్తం += నమోదు_2.second;
    }
    తిరిగి మొత్తం;

#లేదావేళ defined(__APPLE__) && defined(__aarch64__)
    పూర్ణాంకం perf_స్థాయిలు = 0;
    size_t size = పరిమాణం(perf_స్థాయిలు);
    ఒకవేళ (sysctlbyname("hw.nperflevels", &perf_స్థాయిలు, &size, శూన్యసూచిక, 0) != 0 ||
        size != పరిమాణం(perf_స్థాయిలు) || perf_స్థాయిలు <= 0) {
        తిరిగి 0;
    }

    size_t units = 0;
    కొరకు (పూర్ణాంకం i = 0; i < perf_స్థాయిలు; ++i) {
        అక్షరం కీలకం[64] = {};
        పూర్ణాంకం physical_cpus = 0;
        పూర్ణాంకం cpus_చొప్పున_l2 = 0;

        snprintf(కీలకం, పరిమాణం(కీలకం), "hw.perflevel%d.physicalcpu", i);
        size = పరిమాణం(physical_cpus);
        ఒకవేళ (sysctlbyname(కీలకం, &physical_cpus, &size, శూన్యసూచిక, 0) != 0 ||
            size != పరిమాణం(physical_cpus) || physical_cpus <= 0) {
            కొనసాగు;
        }

        snprintf(కీలకం, పరిమాణం(కీలకం), "hw.perflevel%d.cpusperl2", i);
        size = పరిమాణం(cpus_చొప్పున_l2);
        ఒకవేళ (sysctlbyname(కీలకం, &cpus_చొప్పున_l2, &size, శూన్యసూచిక, 0) != 0 ||
            size != పరిమాణం(cpus_చొప్పున_l2) || cpus_చొప్పున_l2 <= 0) {
            కొనసాగు;
        }

        snprintf(కీలకం, పరిమాణం(కీలకం), "hw.perflevel%d.name", i);
        size = 0;
        ఒకవేళ (sysctlbyname(కీలకం, శూన్యసూచిక, &size, శూన్యసూచిక, 0) != 0 || size == 0) {
            కొనసాగు;
        }

        std::string name(size, '\0');
        ఒకవేళ (sysctlbyname(కీలకం, &name[0], &size, శూన్యసూచిక, 0) != 0) {
            కొనసాగు;
        }
        name.resize(size);
        అయితే (!name.ఖాళీలు() && name.back() == '\0') {
            name.pop_back();
        }

        ఒకవేళ (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_చొప్పున_l2 - 1) / cpus_చొప్పున_l2);
        }
    }

    తిరిగి units;

#లేదావేళ defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    తిరిగి 0;

#లేకపోతే
    తిరిగి 0;
#అయితే_ముగింపు
}

స్థిర పూర్ణాంకం parse_uint_env(స్థిరరాశి అక్షరం *s, స్థిరరాశి అక్షరం *name, తార్కిక *ok) {
    ఒకవేళ (!s) { *ok = అబద్ధం; తిరిగి 0; }
    అక్షరం *end = శూన్యసూచిక;
    దీర్ఘ v = strtol(s, &end, 10);
    ఒకవేళ (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = అబద్ధం;
        తిరిగి 0;
    }
    ఒకవేళ (v < 0 || v > గరిష్ఠ_పూర్ణాంకం) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = అబద్ధం;
        తిరిగి 0;
    }
    *ok = నిజం;
    తిరిగి (పూర్ణాంకం)v;
}

స్థిర శూన్య init_kleidiai_context(శూన్య) {
    ggml_critical_section_ప్రారంభం();
    స్థిర తార్కిక initialized = అబద్ధం;

    ఒకవేళ (!initialized) {
        initialized = నిజం;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        స్థిరరాశి అక్షరం *env_sme         = getenv("GGML_KLEIDIAI_SME");
        స్థిరరాశి అక్షరం *env_దారాలు     = getenv("GGML_TOTAL_THREADS");
        స్థిరరాశి అక్షరం *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        స్థిరరాశి స్వయంచాలక runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        ఒకవేళ (env_దారాలు) {
            తార్కిక ok = అబద్ధం;
            పూర్ణాంకం hint = parse_uint_env(env_దారాలు, "GGML_TOTAL_THREADS", &ok);
            ఒకవేళ (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        ఒకవేళ (env_chunk_mult) {
            తార్కిక ok = అబద్ధం;
            పూర్ణాంకం multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            ఒకవేళ (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        పూర్ణాంకం sme_cores = 0;
        తార్కిక sme_env_ok = అబద్ధం;
        తార్కిక sme_env_సమితి = (env_sme != శూన్యసూచిక);

        స్థిరరాశి తార్కిక has_supported_sme_family = runtime_feat.has_sme;
        తార్కిక sme_cap_detected = అబద్ధం;

        ఒకవేళ (has_supported_sme_family) {
            detected_smcus = detect_అంకె_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (పూర్ణాంకం)detected_smcus : 1;

            ఒకవేళ (!sme_env_సమితి && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        ఒకవేళ (sme_env_సమితి) {
            తార్కిక ok = అబద్ధం;
            పూర్ణాంకం v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            ఒకవేళ (ok) {
                ఒకవేళ (has_supported_sme_family) {
                    sme_cores = v;
                } లేకపోతే {
                    ఒకవేళ (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } లేకపోతే {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        ఒకవేళ (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            ఒకవేళ (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        ఒకవేళ (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (పూర్ణాంకం)ctx.features);
        } లేకపోతే {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_గమ్యం_string(ctx.kernels_q4->required_cpu));
        }

        ఒకవేళ (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (పూర్ణాంకం)ctx.features);
        } లేకపోతే {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_గమ్యం_string(ctx.kernels_q8->required_cpu));
        }

        ఒకవేళ (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (పూర్ణాంకం) ctx.features);
        } లేకపోతే {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_గమ్యం_string(ctx.kernels_q4k->required_cpu));
        }

        ఒకవేళ (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (పూర్ణాంకం)ctx.features);
        } లేకపోతే {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_గమ్యం_string(ctx.kernels_f32->required_cpu));
        }

        స్థిరరాశి తార్కిక has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        ఒకవేళ (has_selected_sme_family_kernel) {
            ఒకవేళ (sme_env_సమితి && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } లేకపోతే ఒకవేళ (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } లేకపోతే {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } లేకపోతే {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_చివర();
}

స్థిర అంతఃవరుస పూర్ణాంకం kleidiai_sme_thread_cap() {
    తిరిగి ctx.sme_thread_cap;
}

స్థిర అంతఃవరుస size_t align_పైకి(size_t value, size_t alignment) {
    ఒకవేళ (alignment == 0) {
        తిరిగి value;
    }
    స్థిరరాశి size_t remainder = value % alignment;
    తిరిగి remainder == 0 ? value : value + (alignment - remainder);
}

స్థిర అంతఃవరుస size_t గరిష్ఠ_సామాన్య_భాజకం_కొలత(size_t a, size_t b) {
    అయితే (b != 0) {
        స్థిరరాశి size_t t = a % b;
        a = b;
        b = t;
    }
    తిరిగి a;
}

స్థిర అంతఃవరుస తార్కిక lcm_కొలత(size_t a, size_t b, size_t & ఫలితం) {
    ఒకవేళ (a == 0 || b == 0) {
        ఫలితం = 0;
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి size_t g = గరిష్ఠ_సామాన్య_భాజకం_కొలత(a, b);
    స్థిరరాశి size_t q = a / g;
    ఒకవేళ (q > SIZE_MAX / b) {
        తిరిగి అబద్ధం;
    }
    ఫలితం = q * b;
    తిరిగి నిజం;
}

స్థిర అంతఃవరుస size_t ceil_div_కొలత(size_t a, size_t b) {
    తిరిగి b == 0 ? 0 : (a + b - 1) / b;
}

స్థిర అంతఃవరుస size_t kleidiai_chunk_నిలువుల_సంఖ్య(size_t n, పూర్ణాంకం nth_మొత్తం, తార్కిక disable_chunking, size_t n_అడుగు) {
    స్థిరరాశి size_t multiplier = (nth_మొత్తం == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    స్థిరరాశి size_t భాజకం = std::max<size_t>(1, (size_t) nth_మొత్తం * multiplier);
    స్థిరరాశి size_t chunk_నిలువుల_సంఖ్య = align_పైకి(std::max<size_t>(1, ceil_div_కొలత(n, భాజకం)), n_అడుగు);
    తిరిగి chunk_నిలువుల_సంఖ్య ? chunk_నిలువుల_సంఖ్య : n_అడుగు;
}

నిర్మాణం kleidiai_ఖండం_వాదనలు {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

స్థిర అంతఃవరుస kleidiai_ఖండం_వాదనలు kleidiai_get_ఖండం_వాదనలు(ggml_type rhs_type) {
    ఎంపిక (rhs_type) {
        స్థితి GGML_TYPE_Q4_0:
            తిరిగి { QK4_0, QK4_0, QK4_0 };
        స్థితి GGML_TYPE_Q4_K:
            తిరిగి { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి { 0, 0, QK8_0 };
        అప్రమేయ:
            తిరిగి { 0, 0, 0 };
    }
}

స్థిర అంతఃవరుస తార్కిక kleidiai_pack_fallback_allowed() {
    ఒకవేళ (ctx.sme_thread_cap <= 0) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (ctx.thread_hint <= 0) {
        తిరిగి నిజం;
    }
    తిరిగి ctx.thread_hint > ctx.sme_thread_cap;
}

నిర్మాణం kleidiai_బరువు_శీర్షిక {
    uint32_t magic;
    uint16_t సంస్కరణ_2;
    uint16_t స్థానం_లెక్క;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

స్థిర అంతఃవరుస kleidiai_బరువు_శీర్షిక * kleidiai_బరువు_శీర్షిక_from_ptr(శూన్య * దత్తాంశం) {
    తిరిగి reinterpret_cast<kleidiai_బరువు_శీర్షిక *>(దత్తాంశం);
}

స్థిర అంతఃవరుస స్థిరరాశి kleidiai_బరువు_శీర్షిక * kleidiai_బరువు_శీర్షిక_from_ptr(స్థిరరాశి శూన్య * దత్తాంశం) {
    తిరిగి reinterpret_cast<స్థిరరాశి kleidiai_బరువు_శీర్షిక *>(దత్తాంశం);
}

స్థిర అంతఃవరుస తార్కిక kleidiai_is_బరువు_శీర్షిక_valid(స్థిరరాశి kleidiai_బరువు_శీర్షిక * శీర్షిక) {
    ఒకవేళ (!శీర్షిక) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (శీర్షిక->magic != GGML_KLEIDIAI_PACK_MAGIC || శీర్షిక->సంస్కరణ_2 != GGML_KLEIDIAI_PACK_సంస్కరణ) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (శీర్షిక->స్థానం_లెక్క == 0 || శీర్షిక->స్థానం_లెక్క > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

స్థిర అంతఃవరుస uint8_t * kleidiai_బరువు_స్థానం_ptr(kleidiai_బరువు_శీర్షిక * శీర్షిక, పూర్ణాంకం స్థానం) {
    ఒకవేళ (!kleidiai_is_బరువు_శీర్షిక_valid(శీర్షిక)) {
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (స్థానం < 0 || స్థానం >= శీర్షిక->స్థానం_లెక్క) {
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి reinterpret_cast<uint8_t *>(శీర్షిక) + శీర్షిక->offsets[స్థానం];
}

స్థిర అంతఃవరుస స్థిరరాశి uint8_t * kleidiai_బరువు_స్థానం_ptr(స్థిరరాశి kleidiai_బరువు_శీర్షిక * శీర్షిక, పూర్ణాంకం స్థానం) {
    ఒకవేళ (!kleidiai_is_బరువు_శీర్షిక_valid(శీర్షిక)) {
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (స్థానం < 0 || స్థానం >= శీర్షిక->స్థానం_లెక్క) {
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి reinterpret_cast<స్థిరరాశి uint8_t *>(శీర్షిక) + శీర్షిక->offsets[స్థానం];
}

స్థిర అంతఃవరుస ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    తిరిగి ctx.kernels_q4;
}

స్థిర అంతఃవరుస ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    తిరిగి ctx.kernels_q4k;
}

స్థిర అంతఃవరుస ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    తిరిగి ctx.kernels_q8;
}

స్థిర అంతఃవరుస ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    తిరిగి ctx.kernels_f32;
}

మూస <రకం_పేరు SelectFallback>
స్థిర పూర్ణాంకం kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & బహిర్గతం,
        SelectFallback select_fallback) {
    పూర్ణాంకం count = 0;
    ఒకవేళ (!primary) {
        తిరిగి 0;
    }
    బహిర్గతం[count++] = primary;

    ఒకవేళ (primary->rhs_info.repack_బహుళకం == RHS_REPACK_SINGLE_ONLY) {
        తిరిగి count;
    }

    ఒకవేళ (is_sme_family(primary->required_cpu)) {
        స్థిరరాశి cpu_feature fallback_ముసుగు = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        ఒకవేళ (fallback_ముసుగు != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_ముసుగు);
            ఒకవేళ (fallback && fallback != primary &&
                fallback->rhs_info.repack_బహుళకం != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                బహిర్గతం[count++] = fallback;
            }
        }
    }

    తిరిగి count;
}

స్థిర పూర్ణాంకం kleidiai_collect_kernel_chain(స్థిరరాశి నిర్మాణం ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & బహిర్గతం) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    తిరిగి kleidiai_collect_kernel_chain_common(primary, ctx.features, బహిర్గతం,
        [&](cpu_feature ముసుగు) { తిరిగి ggml_kleidiai_select_kernels(ముసుగు, op); });
}

స్థిర పూర్ణాంకం kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & బహిర్గతం) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    తిరిగి kleidiai_collect_kernel_chain_common(primary, ctx.features, బహిర్గతం,
        [&](cpu_feature ముసుగు) { తిరిగి ggml_kleidiai_select_kernels_q4_0(ముసుగు); });
}

స్థిర పూర్ణాంకం kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & బహిర్గతం) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    తిరిగి kleidiai_collect_kernel_chain_common(primary, ctx.features, బహిర్గతం,
        [&](cpu_feature ముసుగు) { తిరిగి ggml_kleidiai_select_kernels_q4_k(ముసుగు); });
}

స్థిర పూర్ణాంకం kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & బహిర్గతం) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    తిరిగి kleidiai_collect_kernel_chain_common(primary, ctx.features, బహిర్గతం,
        [&](cpu_feature ముసుగు) { తిరిగి ggml_kleidiai_select_kernels_q8_0(ముసుగు); });
}

స్థిర పూర్ణాంకం kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & బహిర్గతం) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    తిరిగి kleidiai_collect_kernel_chain_common(primary, ctx.features, బహిర్గతం,
        [&](cpu_feature ముసుగు) { తిరిగి ggml_kleidiai_select_kernels_f32(ముసుగు); });
}

స్థిర అంతఃవరుస int64_t ggml_ne(స్థిరరాశి ggml_tensor * tensor, పూర్ణాంకం dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    తిరిగి tensor->ne[dim];
}

నామాకాశ ggml::cpu::kleidiai {

స్థిర size_t విడత_కిందికి(size_t x, size_t y) {
    తిరిగి y == 0 ? x : x - (x % y);
}

స్థిర శూన్య transpose_f32kxn_f16nxk(size_t n, size_t k, దశాంశం * dst, స్థిరరాశి uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / పరిమాణం(uint16_t);
    size_t dst_stride = n;

    కొరకు (size_t k_idx = 0; k_idx < k; ++k_idx) {
        కొరకు (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

తరగతి tensor_traits : బహిరంగ ggml::cpu::tensor_traits {
    తార్కిక work_కొలత(పూర్ణాంకం /* n_threads */, స్థిరరాశి నిర్మాణం ggml_tensor * op, size_t & size) override {
        ఒకవేళ (op->op != GGML_OP_MUL_MAT) {
            తిరిగి అబద్ధం;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        స్థిరరాశి పూర్ణాంకం స్థానం_లెక్క = kleidiai_collect_kernel_chain(op, kernel_chain);
        ఒకవేళ (స్థానం_లెక్క == 0) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి తార్కిక is_gemv = op->src[1]->ne[1] == 1;
        స్థిరరాశి size_t k = op->src[0]->ne[0];
        స్థిరరాశి size_t n = op->src[0]->ne[1];
        స్థిరరాశి size_t m = op->src[1]->ne[1];

        ఒకవేళ (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            స్థిరరాశి kleidiai_ఖండం_వాదనలు ఖండం_వాదనలు = kleidiai_get_ఖండం_వాదనలు(kernel_chain[0]->rhs_type);
            స్థిరరాశి size_t qk = ఖండం_వాదనలు.pack_bl;

            size_t cursor = 0;
            తార్కిక any_స్థానం = అబద్ధం;

            కొరకు (పూర్ణాంకం స్థానం = 0; స్థానం < స్థానం_లెక్క; ++స్థానం) {
                ggml_kleidiai_kernels * kernels = kernel_chain[స్థానం];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                ఒకవేళ (!lhs_info || !lhs_info->packed_కొలత_ex || !kernel) {
                    తిరిగి అబద్ధం;
                }

                స్థిరరాశి size_t mr = kernel->get_mr();
                స్థిరరాశి size_t kr = kernel->get_kr();
                స్థిరరాశి size_t sr = kernel->get_sr();

                స్థిరరాశి size_t packed = lhs_info->packed_కొలత_ex(m, k, qk, mr, kr, sr);

                cursor = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_స్థానం = నిజం;
            }

            ఒకవేళ (!any_స్థానం) {
                తిరిగి అబద్ధం;
            }

            size = cursor;
            తిరిగి నిజం;
        }

        ఒకవేళ (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : శూన్యసూచిక;
            ఒకవేళ (is_gemv && op->src[1]->nb[0] == (int64_t) పరిమాణం(దశాంశం) && gemv_kernel &&
                gemv_kernel->get_lhs_స్థానభ్రంశం_ex && gemv_kernel->get_rhs_packed_స్థానభ్రంశం_ex &&
                gemv_kernel->నిరంతరం_kernel_ex && gemv_kernel->get_dst_స్థానభ్రంశం) {
                size = 0;
                తిరిగి నిజం;
            }

            size_t cursor = 0;
            తార్కిక any_స్థానం = అబద్ధం;

            కొరకు (పూర్ణాంకం స్థానం = 0; స్థానం < స్థానం_లెక్క; ++స్థానం) {
                ggml_kleidiai_kernels * kernels = kernel_chain[స్థానం];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                ఒకవేళ (!lhs_info || !lhs_info->packed_కొలత_ex || !kernel) {
                    తిరిగి అబద్ధం;
                }

                స్థిరరాశి size_t mr = kernel->get_mr();
                స్థిరరాశి size_t kr = kernel->get_kr();
                స్థిరరాశి size_t sr = kernel->get_sr();

                cursor  = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_కొలత_ex(m, k, 0, mr, kr, sr);
                any_స్థానం = నిజం;
            }

            ఒకవేళ (!any_స్థానం) {
                తిరిగి అబద్ధం;
            }

            size = cursor;
            తిరిగి నిజం;
        }

        ఒకవేళ (op->src[0]->type == GGML_TYPE_F16) {
            స్థిరరాశి int64_t lhs_batch_size0 = op->src[1]->ne[2];
            స్థిరరాశి int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            స్థిరరాశి int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            తార్కిక any_స్థానం = అబద్ధం;

            కొరకు (పూర్ణాంకం స్థానం = 0; స్థానం < స్థానం_లెక్క; ++స్థానం) {
                ggml_kleidiai_kernels * kernels = kernel_chain[స్థానం];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                ఒకవేళ (!lhs_info || !lhs_info->packed_కొలత_ex || !kernels->rhs_info.packed_కొలత_ex || !kernel) {
                    తిరిగి అబద్ధం;
                }

                స్థిరరాశి size_t mr = kernel->get_mr();
                స్థిరరాశి size_t kr = kernel->get_kr();
                స్థిరరాశి size_t sr = kernel->get_sr();

                cursor  = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_కొలత_ex(m * r, k, 0, mr, kr, sr);
                any_స్థానం = నిజం;
            }

            కొరకు (పూర్ణాంకం స్థానం = 0; స్థానం < స్థానం_లెక్క; ++స్థానం) {
                ggml_kleidiai_kernels * kernels = kernel_chain[స్థానం];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                ఒకవేళ (!kernel || !kernels->rhs_info.packed_కొలత_ex) {
                    తిరిగి అబద్ధం;
                }
                cursor  = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_కొలత_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * పరిమాణం(దశాంశం);
            cursor  = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * పరిమాణం(దశాంశం);

            ఒకవేళ (!any_స్థానం) {
                తిరిగి అబద్ధం;
            }

            size = cursor;
            తిరిగి నిజం;
        }

        తిరిగి అబద్ధం;
    }

    తార్కిక compute_forward(నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst) override {
        ఒకవేళ (dst->op == GGML_OP_MUL_MAT) {
            ఒకవేళ (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                తిరిగి compute_forward_qx(params, dst);
            } లేకపోతే ఒకవేళ (dst->src[0]->type == GGML_TYPE_F32) {
                తిరిగి compute_forward_f32(params, dst);
            } లేకపోతే ఒకవేళ (dst->src[0]->type == GGML_TYPE_F16) {
                తిరిగి compute_forward_fp16(params, dst);
            }
        }
        తిరిగి అబద్ధం;
    }

    తార్కిక compute_forward_f32(ggml_compute_params * params, నిర్మాణం ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        స్థిరరాశి ggml_tensor * src0 = dst->src[0];
        స్థిరరాశి ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ద్విమాన_OP_LOCALS

        ఒకవేళ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        ఒకవేళ (!kernels) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి size_t k = ne00;
        స్థిరరాశి size_t m = ne11;
        స్థిరరాశి size_t n = ne01;
        స్థిరరాశి తార్కిక use_gemv = m == 1 && src1->nb[0] == (int64_t) పరిమాణం(దశాంశం) &&
                              kernels->gemv.get_lhs_స్థానభ్రంశం_ex &&
                              kernels->gemv.get_rhs_packed_స్థానభ్రంశం_ex &&
                              kernels->gemv.నిరంతరం_kernel_ex &&
                              kernels->gemv.get_dst_స్థానభ్రంశం;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        ఒకవేళ (!kernel || !kernel->get_lhs_స్థానభ్రంశం_ex ||
            !kernel->get_rhs_packed_స్థానభ్రంశం_ex || !kernel->నిరంతరం_kernel_ex || !kernel->get_dst_స్థానభ్రంశం) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (!use_gemv && (!lhs_info || !lhs_info->get_స్థానభ్రంశం || !lhs_info->get_packed_స్థానభ్రంశం_ex ||
                          !lhs_info->packed_కొలత_ex || !lhs_info->pack_func_ex)) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి kleidiai_బరువు_శీర్షిక * శీర్షిక = kleidiai_బరువు_శీర్షిక_from_ptr(src0->దత్తాంశం);
        స్థిరరాశి తార్కిక has_శీర్షిక = kleidiai_is_బరువు_శీర్షిక_valid(శీర్షిక);

        స్థిరరాశి uint8_t * rhs_పునాది = has_శీర్షిక ? kleidiai_బరువు_స్థానం_ptr(శీర్షిక, 0)
                                              : static_cast<స్థిరరాశి uint8_t *>(src0->దత్తాంశం);
        ఒకవేళ (!rhs_పునాది) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి పూర్ణాంకం nth = params->nth > 0 ? params->nth : 1;
        స్థిరరాశి పూర్ణాంకం ith = params->ith;

        స్థిరరాశి size_t mr = kernel->get_mr();
        స్థిరరాశి size_t kr = kernel->get_kr();
        స్థిరరాశి size_t sr = kernel->get_sr();

        స్థిరరాశి size_t lhs_packed_కొలత = use_gemv ? 0 : lhs_info->packed_కొలత_ex(m, k, 0, mr, kr, sr);
        ఒకవేళ (!use_gemv) {
            GGML_ASSERT(lhs_packed_కొలత <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        స్థిరరాశి size_t dst_stride = dst->nb[1];
        స్థిరరాశి size_t n_అడుగు = kernel->get_n_అడుగు() ? kernel->get_n_అడుగు() : 1;
        స్థిరరాశి తార్కిక disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) గరిష్ఠ_పూర్ణాంకం);

        స్థిరరాశి int64_t batch_కొలత = ne12 * ne13;
        కొరకు (int64_t batch_idx = 0; batch_idx < batch_కొలత; ++batch_idx) {
            స్థిరరాశి int64_t i12 = batch_idx % ne12;
            స్థిరరాశి int64_t i13 = batch_idx / ne12;
            స్థిరరాశి uint8_t * lhs_batch_పునాది = static_cast<స్థిరరాశి uint8_t *>(src1->దత్తాంశం) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_పునాది = static_cast<uint8_t *>(dst->దత్తాంశం) + i12 * dst->nb[2] + i13 * dst->nb[3];

            ఒకవేళ (!use_gemv) {
                స్థిరరాశి int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_దారాలు = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_దారాలు = std::max<int64_t>(1, max_దారాలు);
                స్థిరరాశి int64_t use_దారాలు = std::min<int64_t>(nth, max_దారాలు);

                ఒకవేళ (ith < use_దారాలు) {
                    స్థిరరాశి int64_t అంకె_m_చొప్పున_thread0   = విడత_కిందికి((size_t)(m_roundup_mr / use_దారాలు), mr);
                    స్థిరరాశి int64_t అంకె_m_చొప్పున_thread_N_1 = (int64_t)m - (use_దారాలు - 1) * అంకె_m_చొప్పున_thread0;

                    స్థిరరాశి int64_t m_ప్రారంభం = (int64_t)ith * అంకె_m_చొప్పున_thread0;
                    స్థిరరాశి int64_t m_లెక్క = (ith == use_దారాలు - 1) ? అంకె_m_చొప్పున_thread_N_1 : అంకె_m_చొప్పున_thread0;

                    స్థిరరాశి size_t పునాది_packed_off  = lhs_info->get_packed_స్థానభ్రంశం_ex(m_ప్రారంభం, k, 0, mr, kr, sr);
                    స్థిరరాశి size_t తదుపరి_ఖండం_off   = lhs_info->get_packed_స్థానభ్రంశం_ex(m_ప్రారంభం + mr, k, 0, mr, kr, sr);
                    స్థిరరాశి size_t అడ్డువరుస_stride_బైట్లు = mr ? (తదుపరి_ఖండం_off - పునాది_packed_off) / mr : 0;

                    int64_t మిగిలిన = m_లెక్క;
                    int64_t cur       = m_ప్రారంభం;

                    అయితే (మిగిలిన > 0) {
                        స్థిరరాశి int64_t take = std::min<int64_t>((int64_t)m - cur, మిగిలిన);
                        స్థిరరాశి size_t src_off = lhs_info->get_స్థానభ్రంశం(cur, src1->nb[1]);
                        స్థిరరాశి శూన్య * src_ptr = lhs_batch_పునాది + src_off;
                        స్థిరరాశి size_t dst_off = పునాది_packed_off + (size_t)(cur - m_ప్రారంభం) * అడ్డువరుస_stride_బైట్లు;
                        శూన్య * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        మిగిలిన -= take;
                    }
                }
            }

            ఒకవేళ (ith == 0) {
                ggml_threadpool_chunk_సమితి(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            స్థిరరాశి size_t chunk_నిలువుల_సంఖ్య = kleidiai_chunk_నిలువుల_సంఖ్య(n, nth, disable_chunking, n_అడుగు);
            GGML_ASSERT(chunk_నిలువుల_సంఖ్య <= (size_t) గరిష్ఠ_పూర్ణాంకం);

            పూర్ణాంకం current_నిలువు = ggml_threadpool_chunk_కూడు(params->threadpool, (పూర్ణాంకం) chunk_నిలువుల_సంఖ్య);
            అయితే ((size_t) current_నిలువు < n) {
                స్థిరరాశి size_t n_ప్రారంభం = (size_t) current_నిలువు;
                స్థిరరాశి size_t n_గమ్యం_process = std::min(chunk_నిలువుల_సంఖ్య, n - n_ప్రారంభం);

                ఒకవేళ (n_గమ్యం_process > 0) {
                    స్థిరరాశి size_t lhs_స్థానభ్రంశం = use_gemv ? kernel->get_lhs_స్థానభ్రంశం_ex(0, k, 0)
                                                       : lhs_info->get_packed_స్థానభ్రంశం_ex(0, k, 0, mr, kr, sr);
                    స్థిరరాశి size_t rhs_packed_స్థానభ్రంశం = kernel->get_rhs_packed_స్థానభ్రంశం_ex(n_ప్రారంభం, k, 0);
                    స్థిరరాశి size_t dst_స్థానభ్రంశం        = kernel->get_dst_స్థానభ్రంశం(0, n_ప్రారంభం, dst_stride);

                    స్థిరరాశి శూన్య * lhs_ptr = use_gemv ? lhs_batch_పునాది + lhs_స్థానభ్రంశం
                                                    : lhs_packed + lhs_స్థానభ్రంశం;
                    స్థిరరాశి శూన్య * rhs_ptr = rhs_పునాది + rhs_packed_స్థానభ్రంశం;
                    దశాంశం * dst_ptr      = reinterpret_cast<దశాంశం *>(dst_batch_పునాది + dst_స్థానభ్రంశం);

                    kernel->నిరంతరం_kernel_ex(m, n_గమ్యం_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          పరిమాణం(దశాంశం),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_నిలువు = ggml_threadpool_chunk_కూడు(params->threadpool, (పూర్ణాంకం) chunk_నిలువుల_సంఖ్య);
            }

            ఒకవేళ (batch_idx != batch_కొలత - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        తిరిగి నిజం;
    }

    తార్కిక compute_forward_fp16(ggml_compute_params * params, నిర్మాణం ggml_tensor * dst) {
        స్థిరరాశి ggml_tensor * src0 = dst->src[0];
        స్థిరరాశి ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ద్విమాన_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        ఒకవేళ (!kernels) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి తార్కిక is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        ఒకవేళ (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_స్థానభ్రంశం_ex || !kernel->get_rhs_packed_స్థానభ్రంశం_ex || !kernel->నిరంతరం_kernel_ex) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి పూర్ణాంకం nth = params->nth;
        స్థిరరాశి పూర్ణాంకం ith = params->ith;

        స్థిరరాశి int64_t lhs_batch_size0 = ne12;
        స్థిరరాశి int64_t rhs_batch_size0 = ne02;
        స్థిరరాశి int64_t batch_కొలత      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        స్థిరరాశి int64_t r = lhs_batch_size0 / rhs_batch_size0;

        స్థిరరాశి int64_t m_సమూహం = ne11;
        స్థిరరాశి int64_t m       = m_సమూహం;
        స్థిరరాశి int64_t n       = ne01;
        స్థిరరాశి int64_t k       = ne00;

        స్థిరరాశి size_t lhs_stride = src1->nb[1];
        స్థిరరాశి size_t rhs_stride = src0->nb[1];
        స్థిరరాశి size_t dst_stride = dst->nb[1];

        స్థిరరాశి int64_t mr = (int64_t) kernel->get_mr();
        స్థిరరాశి int64_t nr = (int64_t) kernel->get_nr();
        స్థిరరాశి int64_t kr = (int64_t) kernel->get_kr();
        స్థిరరాశి int64_t sr = (int64_t) kernel->get_sr();

        స్థిరరాశి size_t lhs_packed_కొలత = lhs_info->packed_కొలత_ex(m, k, 0, mr, kr, sr);
        స్థిరరాశి size_t rhs_packed_కొలత = kernels->rhs_info.packed_కొలత_ex(n, k, nr, kr, 0);
        స్థిరరాశి size_t kxn_కొలత        = k * n * పరిమాణం(దశాంశం);
        స్థిరరాశి size_t bias_కొలత       = n * పరిమాణం(దశాంశం);

        స్థిరరాశి size_t wsize_required = lhs_packed_కొలత + rhs_packed_కొలత + kxn_కొలత + bias_కొలత;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_కొలత;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_కొలత;
        uint8_t * bias       = rhs_kxn + kxn_కొలత;

        కొరకు (int64_t batch_idx = 0; batch_idx < batch_కొలత; ++batch_idx) {
            స్థిరరాశి int64_t rhs_batch_idx = batch_idx / r;
            స్థిరరాశి uint8_t * rhs_batch_పునాది = static_cast<స్థిరరాశి uint8_t *>(src0->దత్తాంశం) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_పునాది = static_cast<uint8_t *>(dst->దత్తాంశం) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                స్థిరరాశి int64_t m_roundup_mr = kai_roundup(m, mr);
                స్థిరరాశి int64_t అంకె_దారాలు  = KAI_MIN(m_roundup_mr / mr, nth);

                ఒకవేళ (ith < అంకె_దారాలు) {
                    స్థిరరాశి int64_t అంకె_m_చొప్పున_thread0   = విడత_కిందికి((size_t)(m_roundup_mr / అంకె_దారాలు), (size_t)mr);
                    స్థిరరాశి int64_t అంకె_m_చొప్పున_thread_N_1 = m - (అంకె_దారాలు - 1) * అంకె_m_చొప్పున_thread0;

                    స్థిరరాశి int64_t m_ప్రారంభం = ith * అంకె_m_చొప్పున_thread0;
                    స్థిరరాశి int64_t m_లెక్క = (ith == అంకె_దారాలు - 1) ? అంకె_m_చొప్పున_thread_N_1 : అంకె_m_చొప్పున_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    స్థిరరాశి size_t పునాది_packed_off  = lhs_info->get_packed_స్థానభ్రంశం_ex(m_ప్రారంభం, k, 0, mr, kr, sr);
                    స్థిరరాశి size_t తదుపరి_ఖండం_off   = lhs_info->get_packed_స్థానభ్రంశం_ex(m_ప్రారంభం + mr, k, 0, mr, kr, sr);
                    స్థిరరాశి size_t అడ్డువరుస_stride_బైట్లు = (తదుపరి_ఖండం_off - పునాది_packed_off) / (size_t)mr;

                    int64_t మిగిలిన = m_లెక్క;
                    int64_t cur       = m_ప్రారంభం;

                    అయితే (మిగిలిన > 0) {
                        స్థిరరాశి int64_t అడ్డువరుస_లోపల_సమూహం = cur;
                        స్థిరరాశి int64_t avail        = m_సమూహం - అడ్డువరుస_లోపల_సమూహం;
                        స్థిరరాశి int64_t take         = std::min(avail, మిగిలిన);

                        స్థిరరాశి uint8_t * lhs_batch_పునాది = static_cast<స్థిరరాశి uint8_t *>(src1->దత్తాంశం) + batch_idx * src1->nb[2];
                        స్థిరరాశి శూన్య * src_ptr = lhs_batch_పునాది + (size_t)అడ్డువరుస_లోపల_సమూహం * lhs_stride;
                        స్థిరరాశి size_t dst_off = పునాది_packed_off + (size_t)(cur - m_ప్రారంభం) * అడ్డువరుస_stride_బైట్లు;
                        శూన్య * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        మిగిలిన -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            ఒకవేళ (ith == 0) {
                memset(bias, 0, (size_t)n * పరిమాణం(దశాంశం));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<దశాంశం *>(rhs_kxn),
                                        reinterpret_cast<స్థిరరాశి uint16_t *>(rhs_batch_పునాది),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * పరిమాణం(దశాంశం),
                             rhs_kxn, bias, శూన్యసూచిక, rhs_packed, 0, శూన్యసూచిక);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                స్థిరరాశి int64_t n_అడుగు  = (int64_t) kernel->get_n_అడుగు();
                int64_t అంకె_దారాలు_n = KAI_MIN(n / n_అడుగు, nth);
                ఒకవేళ (అంకె_దారాలు_n <= 0) {
                    అంకె_దారాలు_n = 1;
                }

                ఒకవేళ (ith < అంకె_దారాలు_n) {
                    స్థిరరాశి int64_t అంకె_n_చొప్పున_thread0   = విడత_కిందికి((size_t)(n / అంకె_దారాలు_n), (size_t)n_అడుగు);
                    స్థిరరాశి int64_t అంకె_n_చొప్పున_thread_N_1 = n - (అంకె_దారాలు_n - 1) * అంకె_n_చొప్పున_thread0;

                    స్థిరరాశి int64_t n_ప్రారంభం      = ith * అంకె_n_చొప్పున_thread0;
                    స్థిరరాశి int64_t n_గమ్యం_process = (ith == అంకె_దారాలు_n - 1) ? అంకె_n_చొప్పున_thread_N_1 : అంకె_n_చొప్పున_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    స్థిరరాశి size_t lhs_packed_offset0 = lhs_info->get_packed_స్థానభ్రంశం_ex(0, k, 0, mr, kr, sr);
                    స్థిరరాశి size_t rhs_packed_స్థానభ్రంశం  = kernel->get_rhs_packed_స్థానభ్రంశం_ex(n_ప్రారంభం, k, 0);
                    స్థిరరాశి size_t dst_స్థానభ్రంశం         = kernel->get_dst_స్థానభ్రంశం((size_t)0, (size_t)n_ప్రారంభం, dst_stride);

                    స్థిరరాశి శూన్య * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    స్థిరరాశి శూన్య * rhs_ptr = rhs_packed + rhs_packed_స్థానభ్రంశం;
                    దశాంశం * dst_ptr      = reinterpret_cast<దశాంశం *>(dst_batch_పునాది + dst_స్థానభ్రంశం);

                    kernel->నిరంతరం_kernel_ex(m, n_గమ్యం_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, పరిమాణం(దశాంశం), -FLT_MAX, FLT_MAX);
                }
            }

            ఒకవేళ (batch_idx != batch_కొలత - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        తిరిగి నిజం;
    }

    తార్కిక compute_forward_qx(నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        స్థిరరాశి ggml_tensor * src0 = dst->src[0];
        స్థిరరాశి ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ద్విమాన_OP_LOCALS

        స్థిరరాశి kleidiai_బరువు_శీర్షిక * శీర్షిక = kleidiai_బరువు_శీర్షిక_from_ptr(src0->దత్తాంశం);
        స్థిరరాశి తార్కిక has_శీర్షిక = kleidiai_is_బరువు_శీర్షిక_valid(శీర్షిక);
        స్థిరరాశి తార్కిక is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        స్థిరరాశి పూర్ణాంకం స్థానం_మొత్తం = kleidiai_collect_kernel_chain(dst, kernel_chain);

        స్వయంచాలక బరువు_for_స్థానం = [&](పూర్ణాంకం స్థానం_సూచీ, size_t & కొలత_బహిర్గతం) -> స్థిరరాశి uint8_t * {
            ఒకవేళ (స్థానం_సూచీ < 0 || స్థానం_సూచీ >= స్థానం_మొత్తం) {
                తిరిగి శూన్యసూచిక;
            }
            ఒకవేళ (has_శీర్షిక) {
                ఒకవేళ (స్థానం_సూచీ < శీర్షిక->స్థానం_లెక్క) {
                    కొలత_బహిర్గతం = static_cast<size_t>(శీర్షిక->sizes[స్థానం_సూచీ]);
                    తిరిగి kleidiai_బరువు_స్థానం_ptr(శీర్షిక, స్థానం_సూచీ);
                }
                తిరిగి శూన్యసూచిక;
            }
            ఒకవేళ (స్థానం_సూచీ == 0) {
                కొలత_బహిర్గతం = ggml_nbytes(src0);
                తిరిగి static_cast<స్థిరరాశి uint8_t *>(src0->దత్తాంశం);
            }
            తిరిగి శూన్యసూచిక;
        };

        నిర్మాణం runtime_స్థానం {
            పూర్ణాంకం స్థానం_సూచీ;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            size_t mr;
            size_t nr;
            size_t kr;
            size_t sr;
            size_t n_అడుగు;
            size_t lhs_packed_కొలత;
            size_t lhs_స్థానభ్రంశం;
            size_t lhs_bl;
            size_t rhs_bl;
            size_t pack_bl;
            size_t lhs_packed_offset0;
            పూర్ణాంకం assigned_దారాలు;
            పూర్ణాంకం thread_begin;
            పూర్ణాంకం thread_చివర;
            స్థిరరాశి uint8_t * rhs_పునాది;
        };

        std::array<runtime_స్థానం, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        పూర్ణాంకం runtime_లెక్క = 0;

        కొరకు (పూర్ణాంకం స్థానం = 0; స్థానం < స్థానం_మొత్తం && runtime_లెక్క < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++స్థానం) {
            ggml_kleidiai_kernels * kernels = kernel_chain[స్థానం];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            ఒకవేళ (!kinfo || !linfo || !linfo->packed_కొలత_ex || !linfo->pack_func_ex || !linfo->get_స్థానభ్రంశం ||
                !kinfo->get_rhs_packed_స్థానభ్రంశం_ex || !kinfo->నిరంతరం_kernel_ex || !kinfo->get_dst_స్థానభ్రంశం) {
                కొనసాగు;
            }

            size_t rhs_కొలత = 0;
            స్థిరరాశి uint8_t * rhs_ptr = బరువు_for_స్థానం(స్థానం, rhs_కొలత);
            ఒకవేళ (!rhs_ptr || rhs_కొలత == 0) {
                కొనసాగు;
            }

            స్థిరరాశి kleidiai_ఖండం_వాదనలు ఖండం_వాదనలు = kleidiai_get_ఖండం_వాదనలు(kernels->rhs_type);

            runtime[runtime_లెక్క] = {
                స్థానం,
                kernels,
                kinfo,
                linfo,
                kinfo->get_mr(),
                kinfo->get_nr(),
                kinfo->get_kr(),
                kinfo->get_sr(),
                kinfo->get_n_అడుగు(),
                0,
                0,
                ఖండం_వాదనలు.lhs_bl,
                ఖండం_వాదనలు.rhs_bl,
                ఖండం_వాదనలు.pack_bl,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_లెక్క;
        }

        ఒకవేళ (runtime_లెక్క == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి పూర్ణాంకం nth_మొత్తం = params->nth > 0 ? params->nth : 1;
        స్థిరరాశి పూర్ణాంకం ith_మొత్తం = params->ith;

        పూర్ణాంకం sme_స్థానం = -1;
        పూర్ణాంకం non_sme_స్థానం = -1;
        కొరకు (పూర్ణాంకం i = 0; i < runtime_లెక్క; ++i) {
            ఒకవేళ (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_స్థానం = i;
                ఆపు;
            }
        }

        కొరకు (పూర్ణాంకం i = 0; i < runtime_లెక్క; ++i) {
            ఒకవేళ (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_స్థానం = i;
                ఆపు;
            }
        }

        స్థిరరాశి పూర్ణాంకం sme_cap_హద్దు = ctx.sme_thread_cap;
        స్థిరరాశి తార్కిక use_hybrid = sme_cap_హద్దు > 0 &&
                                 runtime_లెక్క > 1 &&
                                 nth_మొత్తం > sme_cap_హద్దు;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_నిలువుల_సంఖ్య_చొప్పున_thread = 0;
        ఒకవేళ (runtime_లెక్క > 0 && nth_మొత్తం > 0) {
            min_నిలువుల_సంఖ్య_చొప్పున_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_మొత్తం);
        }
        స్థిరరాశి తార్కిక too_small_for_hybrid = (min_నిలువుల_సంఖ్య_చొప్పున_thread < 2) || (ne11 < 128);

        స్థిరరాశి తార్కిక hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        ఒకవేళ (!hybrid_enabled) {
            పూర్ణాంకం chosen_స్థానం = 0;
            ఒకవేళ (too_small_for_hybrid && sme_స్థానం != -1) {
                chosen_స్థానం = nth_మొత్తం > sme_cap_హద్దు && non_sme_స్థానం != -1 ? non_sme_స్థానం : sme_స్థానం;
            } లేకపోతే ఒకవేళ (runtime_లెక్క > 1 && ctx.sme_thread_cap > 0 && nth_మొత్తం > ctx.sme_thread_cap) {
                chosen_స్థానం = 1;
            }
            ఒకవేళ (chosen_స్థానం != 0 && chosen_స్థానం < runtime_లెక్క) {
                runtime[0] = runtime[chosen_స్థానం];
                runtime[0].assigned_దారాలు = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_చివర = 0;
            }
            runtime_లెక్క = runtime_లెక్క > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_స్థానం = -1;
            ఒకవేళ (runtime_లెక్క > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_స్థానం = 0;
            }
        }

        స్థిరరాశి తార్కిక sme_single_only = sme_స్థానం != -1 &&
                                     runtime[sme_స్థానం].kernels->rhs_info.repack_బహుళకం == RHS_REPACK_SINGLE_ONLY;
        పూర్ణాంకం sme_cap = sme_single_only ? nth_మొత్తం : kleidiai_sme_thread_cap();
        ఒకవేళ (sme_cap < 0) {
            sme_cap = nth_మొత్తం;
        }
        sme_cap = std::min(sme_cap, nth_మొత్తం);

        పూర్ణాంకం దారాలు_మిగిలిన = nth_మొత్తం;
        ఒకవేళ (sme_స్థానం != -1) {
            పూర్ణాంకం sme_దారాలు = std::min(std::max(sme_cap, 0), దారాలు_మిగిలిన);
            runtime[sme_స్థానం].assigned_దారాలు = sme_దారాలు;
            దారాలు_మిగిలిన -= sme_దారాలు;
        }

        పూర్ణాంకం fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        పూర్ణాంకం fallback_లెక్క = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        కొరకు (పూర్ణాంకం i = 0; i < runtime_లెక్క; ++i) {
            ఒకవేళ (i == sme_స్థానం) {
                కొనసాగు;
            }
            fallback_indices[fallback_లెక్క++] = i;
        }

        కొరకు (పూర్ణాంకం fi = 0; fi < fallback_లెక్క; ++fi) {
            ఒకవేళ (దారాలు_మిగిలిన <= 0) {
                ఆపు;
            }
            స్థిరరాశి పూర్ణాంకం స్థానం_సూచీ = fallback_indices[fi];
            స్థిరరాశి పూర్ణాంకం slots_ఎడమ = fallback_లెక్క - fi;
            పూర్ణాంకం share = (దారాలు_మిగిలిన + slots_ఎడమ - 1) / slots_ఎడమ;
            share     = std::min(share, దారాలు_మిగిలిన);
            runtime[స్థానం_సూచీ].assigned_దారాలు = share;
            దారాలు_మిగిలిన -= share;
        }

        పూర్ణాంకం thread_cursor = 0;
        కొరకు (పూర్ణాంకం i = 0; i < runtime_లెక్క; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_దారాలు;
            runtime[i].thread_చివర = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        స్థిరరాశి పూర్ణాంకం చురుకైన_దారాలు = thread_cursor;

        పూర్ణాంకం local_స్థానం = -1;
        పూర్ణాంకం local_ith  = 0;
        కొరకు (పూర్ణాంకం i = 0; i < runtime_లెక్క; ++i) {
            ఒకవేళ (ith_మొత్తం >= runtime[i].thread_begin && ith_మొత్తం < runtime[i].thread_చివర) {
                local_స్థానం = i;
                local_ith  = ith_మొత్తం - runtime[i].thread_begin;
                ఆపు;
            }
        }
        స్థిరరాశి size_t k = ne00;
        స్థిరరాశి size_t m = ne11;
        స్థిరరాశి size_t n = ne01;

        size_t cursor = 0;
        కొరకు (పూర్ణాంకం i = 0; i < runtime_లెక్క; ++i) {
            runtime[i].lhs_packed_కొలత = runtime[i].lhs_info->packed_కొలత_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_స్థానభ్రంశం = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_స్థానభ్రంశం_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_కొలత;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_అడుగు = 1;
        కొరకు (పూర్ణాంకం i = 0; i < runtime_లెక్క; ++i) {
            ఒకవేళ (runtime[i].assigned_దారాలు == 0) {
                కొనసాగు;
            }
            size_t తదుపరి_అడుగు = 0;
            ఒకవేళ (!lcm_కొలత(common_అడుగు, runtime[i].n_అడుగు ? runtime[i].n_అడుగు : 1, తదుపరి_అడుగు)) {
                తిరిగి అబద్ధం;
            }
            common_అడుగు = తదుపరి_అడుగు;
        }
        GGML_ASSERT(common_అడుగు > 0);

        స్థిరరాశి తార్కిక disable_chunking = ggml_is_numa();
        స్థిరరాశి size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        స్థిరరాశి size_t chunk_భాజకం = (చురుకైన_దారాలు == 1 || disable_chunking) ? (size_t) చురుకైన_దారాలు : (size_t) చురుకైన_దారాలు * chunk_multiplier;
        size_t chunk_నిలువుల_సంఖ్య = align_పైకి(std::max<size_t>(1, ceil_div_కొలత(n, chunk_భాజకం)), common_అడుగు);
        ఒకవేళ (chunk_నిలువుల_సంఖ్య == 0) {
            chunk_నిలువుల_సంఖ్య = common_అడుగు;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        స్థిరరాశి size_t nchunk_కొలత = std::max<size_t>(1, ceil_div_కొలత(n, chunk_నిలువుల_సంఖ్య));
        GGML_ASSERT(nchunk_కొలత <= (size_t)గరిష్ఠ_పూర్ణాంకం);
        స్థిరరాశి పూర్ణాంకం nchunk = (పూర్ణాంకం)nchunk_కొలత;
        స్థిరరాశి size_t dst_stride = dst->nb[1];

        స్వయంచాలక నిరంతరం_chunk = [&](runtime_స్థానం & స్థానం, size_t global_ప్రారంభం, size_t నిలువుల_సంఖ్య_2, uint8_t * dst_batch_పునాది) {
            స్థిరరాశి size_t rhs_packed_స్థానభ్రంశం = స్థానం.kernel->get_rhs_packed_స్థానభ్రంశం_ex(global_ప్రారంభం, k, స్థానం.rhs_bl);
            స్థిరరాశి size_t dst_స్థానభ్రంశం        = స్థానం.kernel->get_dst_స్థానభ్రంశం(0, global_ప్రారంభం, dst_stride);

            స్థిరరాశి uint8_t * lhs_ptr = scratch + స్థానం.lhs_స్థానభ్రంశం + స్థానం.lhs_packed_offset0;
            స్థిరరాశి uint8_t * rhs_ptr = స్థానం.rhs_పునాది + rhs_packed_స్థానభ్రంశం;
            దశాంశం * dst_ptr         = reinterpret_cast<దశాంశం *>(dst_batch_పునాది + dst_స్థానభ్రంశం);

            స్థానం.kernel->నిరంతరం_kernel_ex(m, నిలువుల_సంఖ్య_2, k, స్థానం.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       పరిమాణం(దశాంశం),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        కొరకు (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            స్థిరరాశి uint8_t * lhs_batch_పునాది = static_cast<స్థిరరాశి uint8_t *>(src1->దత్తాంశం) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_పునాది = static_cast<uint8_t *>(dst->దత్తాంశం) + batch_idx * dst->nb[2];

            ఒకవేళ (local_స్థానం != -1) {
                runtime_స్థానం & స్థానం = runtime[local_స్థానం];
                స్థిరరాశి int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)స్థానం.mr);
                int64_t max_దారాలు = స్థానం.mr ? (m_roundup_mr / (int64_t)స్థానం.mr) : స్థానం.assigned_దారాలు;
                max_దారాలు = std::max<int64_t>(1, max_దారాలు);
                స్థిరరాశి int64_t use_దారాలు = std::min<int64_t>(స్థానం.assigned_దారాలు, max_దారాలు);

                ఒకవేళ (local_ith < use_దారాలు) {
                    స్థిరరాశి int64_t అంకె_m_చొప్పున_thread0   = విడత_కిందికి((size_t)(m_roundup_mr / use_దారాలు), స్థానం.mr);
                    స్థిరరాశి int64_t అంకె_m_చొప్పున_thread_N_1 = (int64_t)m - (use_దారాలు - 1) * అంకె_m_చొప్పున_thread0;

                    స్థిరరాశి int64_t m_ప్రారంభం = (int64_t)local_ith * అంకె_m_చొప్పున_thread0;
                    స్థిరరాశి int64_t m_లెక్క = (local_ith == use_దారాలు - 1) ? అంకె_m_చొప్పున_thread_N_1 : అంకె_m_చొప్పున_thread0;

                    స్థిరరాశి size_t పునాది_packed_off  = స్థానం.lhs_info->get_packed_స్థానభ్రంశం_ex(m_ప్రారంభం, k, స్థానం.lhs_bl, స్థానం.mr, స్థానం.kr, స్థానం.sr);
                    స్థిరరాశి size_t తదుపరి_ఖండం_off   = స్థానం.lhs_info->get_packed_స్థానభ్రంశం_ex(m_ప్రారంభం + స్థానం.mr, k, స్థానం.lhs_bl, స్థానం.mr, స్థానం.kr, స్థానం.sr);
                    స్థిరరాశి size_t అడ్డువరుస_stride_బైట్లు = స్థానం.mr ? (తదుపరి_ఖండం_off - పునాది_packed_off) / స్థానం.mr : 0;

                    int64_t మిగిలిన = m_లెక్క;
                    int64_t cur       = m_ప్రారంభం;

                    uint8_t * lhs_packed = scratch + స్థానం.lhs_స్థానభ్రంశం;
                    అయితే (మిగిలిన > 0) {
                        స్థిరరాశి int64_t అడ్డువరుస_లోపల_సమూహం = cur;
                        స్థిరరాశి int64_t avail        = (int64_t)m - అడ్డువరుస_లోపల_సమూహం;
                        స్థిరరాశి int64_t take         = std::min(avail, మిగిలిన);

                        స్థిరరాశి size_t src_off = స్థానం.lhs_info->get_స్థానభ్రంశం(అడ్డువరుస_లోపల_సమూహం, src1->nb[1]);
                        స్థిరరాశి శూన్య * src_ptr = lhs_batch_పునాది + src_off;
                        స్థిరరాశి size_t dst_off = పునాది_packed_off + (size_t)(cur - m_ప్రారంభం) * అడ్డువరుస_stride_బైట్లు;
                        శూన్య * dst_ptr       = lhs_packed + dst_off;

                        స్థానం.lhs_info->pack_func_ex(take, k, స్థానం.lhs_bl, స్థానం.mr, స్థానం.kr, స్థానం.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        మిగిలిన -= take;
                    }
                }
            }

            ఒకవేళ (ith_మొత్తం == 0) {
                ggml_threadpool_chunk_సమితి(params->threadpool, చురుకైన_దారాలు);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            ఒకవేళ (local_స్థానం != -1) {
                runtime_స్థానం & స్థానం = runtime[local_స్థానం];
                పూర్ణాంకం current_chunk = ith_మొత్తం;
                అయితే (current_chunk < nchunk) {
                    స్థిరరాశి size_t global_ప్రారంభం = (size_t)current_chunk * chunk_నిలువుల_సంఖ్య;
                    ఒకవేళ (global_ప్రారంభం >= n) {
                        ఆపు;
                    }

                    స్థిరరాశి size_t నిలువుల_సంఖ్య_2 = std::min(chunk_నిలువుల_సంఖ్య, n - global_ప్రారంభం);
                    ఒకవేళ (నిలువుల_సంఖ్య_2 > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        నిరంతరం_chunk(స్థానం, global_ప్రారంభం, నిలువుల_సంఖ్య_2, dst_batch_పునాది);
                    }

                    current_chunk = ggml_threadpool_chunk_కూడు(params->threadpool, 1);
                }
            }

            ఒకవేళ (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        తిరిగి నిజం;
    }

బహిరంగ:
    పూర్ణాంకం repack(నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        స్థిరరాశి size_t n = tensor->ne[1];
        స్థిరరాశి size_t k = tensor->ne[0];

        kleidiai_బరువు_శీర్షిక * శీర్షిక = kleidiai_బరువు_శీర్షిక_from_ptr(tensor->దత్తాంశం);
        ఒకవేళ (!శీర్షిక) {
            తిరిగి -1;
        }

        శీర్షిక->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        శీర్షిక->సంస్కరణ_2    = GGML_KLEIDIAI_PACK_సంస్కరణ;
        శీర్షిక->స్థానం_లెక్క = 0;

        uint8_t * పునాది_ptr = static_cast<uint8_t *>(tensor->దత్తాంశం);
        size_t cursor = పరిమాణం(kleidiai_బరువు_శీర్షిక);
        cursor = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        స్థిరరాశి తార్కిక want_q4k = tensor->type == GGML_TYPE_Q4_K;
        స్థిరరాశి తార్కిక want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        స్థిరరాశి తార్కిక want_f32 = tensor->type == GGML_TYPE_F32;
        స్థిరరాశి పూర్ణాంకం స్థానం_మొత్తం = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        స్థిరరాశి తార్కిక allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<దశాంశం>  scales;
        std::vector<దశాంశం>  bias;
        std::vector<దశాంశం> అడ్డువరుస_f32;

        ఒకవేళ (want_q8 && స్థానం_మొత్తం > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            స్థిరరాశి size_t అడ్డువరుస_stride = tensor->nb[1];
            స్థిరరాశి size_t k_ఖండాలు   = (k + QK8_0 - 1) / QK8_0;

            ఒకవేళ (tensor->type == GGML_TYPE_Q6_K) {
                ఒకవేళ (k % QK_K != 0 || k % QK8_0 != 0) {
                    తిరిగి -1;
                }
                అడ్డువరుస_f32.resize(k);
            }

            కొరకు (size_t అడ్డువరుస = 0; అడ్డువరుస < n; ++అడ్డువరుస) {
                స్థిరరాశి uint8_t * అడ్డువరుస_దత్తాంశం = static_cast<స్థిరరాశి uint8_t *>(దత్తాంశం) + అడ్డువరుస * అడ్డువరుస_stride;

                ఒకవేళ (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_అడ్డువరుస_q6_K(reinterpret_cast<స్థిరరాశి ఖండం_q6_K *>(అడ్డువరుస_దత్తాంశం), అడ్డువరుస_f32.దత్తాంశం(), k);
                }

                దశాంశం max_abs = 0.0f;
                ఒకవేళ (tensor->type == GGML_TYPE_Q6_K) {
                    కొరకు (size_t రేఖీయ_idx = 0; రేఖీయ_idx < k; ++రేఖీయ_idx) {
                        max_abs = std::max(max_abs, std::fabs(అడ్డువరుస_f32[రేఖీయ_idx]));
                    }
                } లేకపోతే {
                    స్థిరరాశి ఖండం_q8_0 * అడ్డువరుస_ఖండాలు = reinterpret_cast<స్థిరరాశి ఖండం_q8_0 *>(అడ్డువరుస_దత్తాంశం);
                    కొరకు (size_t ఖండం = 0; ఖండం < k_ఖండాలు; ++ఖండం) {
                        స్థిరరాశి ఖండం_q8_0 & blk = అడ్డువరుస_ఖండాలు[ఖండం];
                        స్థిరరాశి దశాంశం d = GGML_FP16_గమ్యం_FP32(blk.d);
                        కొరకు (size_t l = 0; l < QK8_0; ++l) {
                            స్థిరరాశి size_t రేఖీయ_idx = ఖండం * QK8_0 + l;
                            ఒకవేళ (రేఖీయ_idx >= k) {
                                ఆపు;
                            }
                            స్థిరరాశి దశాంశం value = d * static_cast<దశాంశం>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                దశాంశం scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[అడ్డువరుస] = scale;
                స్థిరరాశి దశాంశం inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                ఒకవేళ (tensor->type == GGML_TYPE_Q6_K) {
                    కొరకు (size_t రేఖీయ_idx = 0; రేఖీయ_idx < k; ++రేఖీయ_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(అడ్డువరుస_f32[రేఖీయ_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[అడ్డువరుస * k + రేఖీయ_idx] = static_cast<int8_t>(q);
                    }
                } లేకపోతే {
                    స్థిరరాశి ఖండం_q8_0 * అడ్డువరుస_ఖండాలు = reinterpret_cast<స్థిరరాశి ఖండం_q8_0 *>(అడ్డువరుస_దత్తాంశం);
                    కొరకు (size_t ఖండం = 0; ఖండం < k_ఖండాలు; ++ఖండం) {
                        స్థిరరాశి ఖండం_q8_0 & blk = అడ్డువరుస_ఖండాలు[ఖండం];
                        స్థిరరాశి దశాంశం d = GGML_FP16_గమ్యం_FP32(blk.d);
                        కొరకు (size_t l = 0; l < QK8_0; ++l) {
                            స్థిరరాశి size_t రేఖీయ_idx = ఖండం * QK8_0 + l;
                            ఒకవేళ (రేఖీయ_idx >= k) {
                                ఆపు;
                            }
                            స్థిరరాశి దశాంశం value = d * static_cast<దశాంశం>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[అడ్డువరుస * k + రేఖీయ_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        ఒకవేళ (want_f32 && స్థానం_మొత్తం > 0) {
            bias.resize(n, 0.0f);
        }

        కొరకు (పూర్ణాంకం స్థానం = 0; స్థానం < స్థానం_మొత్తం && స్థానం < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++స్థానం) {
            ఒకవేళ (!allow_fallback && స్థానం > 0) {
                ఆపు;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[స్థానం];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            ఒకవేళ (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_కొలత_ex || !kernel) {
                కొనసాగు;
            }

            స్థిరరాశి size_t nr = kernel->get_nr();
            స్థిరరాశి size_t kr = kernel->get_kr();
            స్థిరరాశి size_t sr = kernel->get_sr();
            స్థిరరాశి ggml_type rhs_type = kernels->rhs_type;
            ఒకవేళ (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            స్థిరరాశి size_t ఖండం_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            ఒకవేళ (ఖండం_len == SIZE_MAX) {
                కొనసాగు;
            }

            స్థిరరాశి size_t packed_కొలత = rhs_info->packed_కొలత_ex(n, k, nr, kr, ఖండం_len);
            స్థిరరాశి size_t aligned_cursor = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = పునాది_ptr + aligned_cursor;

            ఒకవేళ (rhs_type == GGML_TYPE_Q4_0) {
                నిర్మాణం kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_బిందువు = 1;
                params.rhs_zero_బిందువు = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<స్థిరరాశి uint8_t *>(దత్తాంశం), శూన్యసూచిక, శూన్యసూచిక,
                                       dst_ptr, 0, &params);
            } లేకపోతే ఒకవేళ (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       దత్తాంశం, శూన్యసూచిక, శూన్యసూచిక, dst_ptr, 0, శూన్యసూచిక);
            } లేకపోతే ఒకవేళ (rhs_type == GGML_TYPE_Q8_0) {
                నిర్మాణం kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_బిందువు = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.దత్తాంశం(), శూన్యసూచిక, scales.దత్తాంశం(),
                                       dst_ptr, 0, &params);
            } లేకపోతే ఒకవేళ (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       దత్తాంశం, bias.దత్తాంశం(), శూన్యసూచిక,
                                       dst_ptr, 0, శూన్యసూచిక);
            } లేకపోతే {
                కొనసాగు;
            }

            శీర్షిక->offsets[శీర్షిక->స్థానం_లెక్క] = aligned_cursor;
            శీర్షిక->sizes[శీర్షిక->స్థానం_లెక్క]   = packed_కొలత;
            ++శీర్షిక->స్థానం_లెక్క;

            cursor = aligned_cursor + packed_కొలత;
        }

        ఒకవేళ (శీర్షిక->స్థానం_లెక్క == 0) {
            శీర్షిక->magic   = 0;
            శీర్షిక->సంస్కరణ_2 = 0;
            జ్ఞాపకం_నకలు(tensor->దత్తాంశం, దత్తాంశం, దత్తాంశం_కొలత);
        }

        తిరిగి 0;
    }
};

స్థిర ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_బఫరు_t, నిర్మాణం ggml_tensor *) {
    స్థిర tensor_traits traits;
    తిరిగి &traits;
}
}  // namespace ggml::cpu::kleidiai

స్థిర గణన ggml_స్థితి ggml_backend_cpu_kleidiai_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor) {
    tensor->extra = (శూన్య *) ggml::cpu::kleidiai::get_tensor_traits(బఫరు, tensor);

    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర శూన్య ggml_backend_cpu_kleidiai_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor,
                                                       స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(స్థానభ్రంశం == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    స్వయంచాలక tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    స్వయంచాలక OK            = tensor_traits->repack(tensor, దత్తాంశం, size);

    GGML_ASSERT(OK == 0);
    GGML_వాడని(బఫరు);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cpu_kleidiai_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    GGML_వాడని(buft);
    తిరిగి "CPU_KLEIDIAI";
}

స్థిర ggml_backend_బఫరు_t ggml_backend_cpu_kleidiai_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    ggml_backend_బఫరు_t బఫరు = ggml_backend_buft_alloc_బఫరు(ggml_backend_cpu_బఫరు_type(), size);

    ఒకవేళ (బఫరు == శూన్యసూచిక) {
        తిరిగి శూన్యసూచిక;
    }

    బఫరు->buft              = buft;
    బఫరు->iface.init_tensor = ggml_backend_cpu_kleidiai_బఫరు_init_tensor;
    బఫరు->iface.సమితి_tensor  = ggml_backend_cpu_kleidiai_బఫరు_సమితి_tensor;
    బఫరు->iface.get_tensor  = శూన్యసూచిక;
    బఫరు->iface.cpy_tensor  = శూన్యసూచిక;
    తిరిగి బఫరు;
}

స్థిర size_t ggml_backend_cpu_kleidiai_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    GGML_వాడని(buft);
    తిరిగి TENSOR_ALIGNMENT;
}

స్థిర size_t ggml_backend_cpu_kleidiai_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    GGML_వాడని(buft);

    ఒకవేళ (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        తిరిగి ggml_nbytes(tensor);
    }

    స్థిరరాశి size_t n = tensor->ne[1];
    స్థిరరాశి size_t k = tensor->ne[0];

    size_t cursor = పరిమాణం(kleidiai_బరువు_శీర్షిక);
    cursor = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    స్థిరరాశి తార్కిక want_q4k = tensor->type == GGML_TYPE_Q4_K;
    స్థిరరాశి తార్కిక want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    స్థిరరాశి తార్కిక want_f32 = tensor->type == GGML_TYPE_F32;
    స్థిరరాశి పూర్ణాంకం స్థానం_మొత్తం = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    స్థిరరాశి తార్కిక allow_fallback = kleidiai_pack_fallback_allowed();

    size_t స్థానం_లెక్క = 0;
    కొరకు (పూర్ణాంకం స్థానం = 0; స్థానం < స్థానం_మొత్తం; ++స్థానం) {
        ఒకవేళ (!allow_fallback && స్థానం > 0) {
            ఆపు;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[స్థానం];
        ఒకవేళ (!kernels) {
            కొనసాగు;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        ఒకవేళ (!kernel || !rhs_info || !rhs_info->packed_కొలత_ex) {
            కొనసాగు;
        }

        స్థిరరాశి ggml_type rhs_type = kernels->rhs_type;
        స్థిరరాశి size_t ఖండం_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        ఒకవేళ (ఖండం_len == SIZE_MAX) {
            కొనసాగు;
        }

        cursor = align_పైకి(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_కొలత_ex(n, k, kernel->get_nr(), kernel->get_kr(), ఖండం_len);
        ++స్థానం_లెక్క;
    }

    ఒకవేళ (స్థానం_లెక్క == 0) {
        తిరిగి ggml_nbytes(tensor);
    }

    తిరిగి std::max(cursor, ggml_nbytes(tensor));
}

నామాకాశ ggml::cpu::kleidiai {
తరగతి extra_బఫరు_type : ggml::cpu::extra_బఫరు_type {
    తార్కిక supports_op(ggml_backend_dev_t, స్థిరరాశి నిర్మాణం ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        ఒకవేళ (op->op != GGML_OP_MUL_MAT) {
            తిరిగి అబద్ధం;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        స్థిరరాశి పూర్ణాంకం స్థానం_మొత్తం = kleidiai_collect_kernel_chain(op, kernel_chain);
        స్థిరరాశి తార్కిక src0_is_kleidiai =
            op->src[0]->బఫరు &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->బఫరు->buft->context == ఇది &&
            స్థానం_మొత్తం > 0;

        ఒకవేళ ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            ఒకవేళ (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == శూన్యసూచిక) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == శూన్యసూచిక) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == శూన్యసూచిక) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == శూన్యసూచిక) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == శూన్యసూచిక) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (op->src[1]->బఫరు && !ggml_backend_buft_is_host(op->src[1]->బఫరు->buft)) {
                తిరిగి అబద్ధం;
            }

            ఒకవేళ (op->src[0]->type == GGML_TYPE_Q4_K) {
                ఒకవేళ (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    తిరిగి అబద్ధం;
                }
                తిరిగి ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == పరిమాణం(దశాంశం) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * పరిమాణం(దశాంశం));
            }

            ఒకవేళ (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                ఒకవేళ (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    తిరిగి నిజం;
                }
                తిరిగి అబద్ధం;
            }

            ఒకవేళ (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                తిరిగి అబద్ధం;
            }

            తిరిగి నిజం;
        }

        తిరిగి అబద్ధం;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(స్థిరరాశి నిర్మాణం ggml_tensor * op) override {
        ఒకవేళ (op->op == GGML_OP_MUL_MAT) {
            ఒకవేళ (op->src[0]->బఫరు && op->src[0]->బఫరు->buft->context == ఇది) {
                తిరిగి (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } లేకపోతే {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                ఒకవేళ (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    స్థిర std::atomic<తార్కిక> warned(అబద్ధం);
                    ఒకవేళ (!warned.exchange(నిజం)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                ఒకవేళ (op->src[0]->type != GGML_TYPE_F16) {
                    తిరిగి శూన్యసూచిక;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                స్థిరరాశి పూర్ణాంకం స్థానం_మొత్తం = kleidiai_collect_kernel_chain(op, kernel_chain);
                ఒకవేళ (స్థానం_మొత్తం > 0 && op->src[1]->ne[1] > 1) {
                    ఒకవేళ ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        తిరిగి శూన్యసూచిక;
                    }
                    తిరిగి ggml::cpu::kleidiai::get_tensor_traits(ఖాళీ, ఖాళీ);
                }
            }
        }
        తిరిగి శూన్యసూచిక;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_బఫరు_type_t ggml_backend_cpu_kleidiai_బఫరు_type(శూన్య) {
    స్థిర ggml::cpu::kleidiai::extra_బఫరు_type ctx;
    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_cpu_బఫరు_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_బఫరు_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_బఫరు_type_alloc_బఫరు,
                           /* .alloc_buffer_n       = */ శూన్యసూచిక,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_బఫరు_type_get_alignment,
                           /* .get_max_size         = */ శూన్యసూచిక,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_బఫరు_type_get_alloc_కొలత,
                           /* .get_alloc_size_n     = */ ఖాళీ,
                           /* .is_host              = */ శూన్యసూచిక,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    తిరిగి &ggml_backend_cpu_బఫరు_type_kleidiai;
}
