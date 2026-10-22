// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#അന്തര്ഭാവ <arm_neon.h>
#അന്തര്ഭാവ <cassert>
#അന്തര്ഭാവ <cstdio>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <atomic>
#അന്തര്ഭാവ <cfloat>
#അന്തര്ഭാവ <cctype>
#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <stdexcept>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <string.h>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <vector>
#അന്തര്ഭാവ <array>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <fstream>
#അന്തര്ഭാവ <map>
#അന്തര്ഭാവ <iostream>
#അന്തര്ഭാവ <climits>
#അന്തര്ഭാവ <charconv>
#അന്തര്ഭാവ <system_error>
#എങ്കിൽ defined(__linux__)
#അന്തര്ഭാവ <asm/hwcap.h>
#അന്തര്ഭാവ <dirent.h>
#അന്തര്ഭാവ <sys/auxv.h>
#അന്തര്ഭാവ <sys/types.h>
#അന്തര്ഭാവ <sys/stat.h>
#അന്തര്ഭാവ <unistd.h>
#അല്ലെങ്കിലെങ്കിൽ defined(__APPLE__)
#അന്തര്ഭാവ <sys/sysctl.h>
#അന്തര്ഭാവ <sys/types.h>
#endif

#അന്തര്ഭാവ "kleidiai.h"

#അന്തര്ഭാവ "ggml-cpu.h"
#അന്തര്ഭാവ "ggml-cpu-impl.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-feats.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-quants.h"
#അന്തര്ഭാവ "ggml-threading.h"
#അന്തര്ഭാവ "traits.h"

#അന്തര്ഭാവ "kernels.h"

#അന്തര്ഭാവ "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#അന്തര്ഭാവ "ggml-common.h"

സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
സ്ഥിതം constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
സ്ഥിതം constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
സ്ഥിതം constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
സ്ഥിതം constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

ഘടന ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    പൂർണ്ണസംഖ്യ sme_thread_cap; // <= 0 means "SME disabled/unknown"
    പൂർണ്ണസംഖ്യ thread_hint;    // <= 0 means "no hint"
    പൂർണ്ണസംഖ്യ chunk_multiplier;
} സ്ഥിതം ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

സ്ഥിതം വരിയിൽ തർക്കം is_sme_family(cpu_feature f) {
    തിരികെ (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

സ്ഥിതം സ്ഥിരം അക്ഷരം* cpu_feature_to_string(cpu_feature f) {
    എങ്കിൽ (f == CPU_FEATURE_NONE) {
        തിരികെ "NONE";
    } അല്ലെങ്കിൽ എങ്കിൽ ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        തിരികെ "SME2";
    } അല്ലെങ്കിൽ എങ്കിൽ ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        തിരികെ "SME";
    } അല്ലെങ്കിൽ എങ്കിൽ ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        തിരികെ "SVE";
    }
    അല്ലെങ്കിൽ എങ്കിൽ ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        തിരികെ "I8MM";
    } അല്ലെങ്കിൽ എങ്കിൽ ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        തിരികെ "DOTPROD";
    }
    അല്ലെങ്കിൽ {
        തിരികെ "UNKNOWN";
    }
}

#എങ്കിൽ defined(__linux__) && defined(__aarch64__)
സ്ഥിതം തർക്കം parse_cpu_dir_name(സ്ഥിരം അക്ഷരം* name, size_t* cpu) {
    എങ്കിൽ (strncmp(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം അക്ഷരം* first = name + 3;
    സ്ഥിരം അക്ഷരം* last = name + strlen(name);

    size_t value = 0;
    സ്ഥിരം auto [end, ec] = std::from_chars(first, last, value, 10);

    എങ്കിൽ (ec != std::errc{} || end != last) {
        തിരികെ അസത്യം;
    }

    *cpu = value;
    തിരികെ സത്യം;
}

സ്ഥിതം std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    എങ്കിൽ (dir == nullptr) {
        തിരികെ cpus;
    }

    ആകുമ്പോൾ (dirent * entry = readdir(dir)) {
        size_t cpu = 0;
        എങ്കിൽ (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    തിരികെ cpus;
}
#endif

#എങ്കിൽ defined(__APPLE__) && defined(__aarch64__)
സ്ഥിതം തർക്കം apple_sme_counted_perf_level(std::string name) {
    വേണ്ടി (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (അക്ഷരം) std::tolower((unsigned അക്ഷരം) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    തിരികെ name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

സ്ഥിതം ശൂന്യം add_smcus_from_smidr(uint64_t smidr, size_t & അക്കം_private, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    സ്ഥിരം uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    സ്ഥിരം uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    സ്ഥിരം size_t shared_എണ്ണം = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    സ്ഥിരം uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    സ്ഥിരം uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    സ്ഥിരം uint32_t id = (affinity2 << 12) | affinity;

    എങ്കിൽ (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    തിരഞ്ഞെടുപ്പ് (sh) {
        അവസ്ഥ 2: // private SMCU
            ++അക്കം_private;
            നിർത്തുക;
        അവസ്ഥ 3: // shared SMCU
            എങ്കിൽ (shared_counts[id] < shared_എണ്ണം) {
                shared_counts[id] = shared_എണ്ണം;
            }
            നിർത്തുക;
        അവസ്ഥ 0:
            എങ്കിൽ (id == 0) {
                ++അക്കം_private;
            } അല്ലെങ്കിൽ എങ്കിൽ (shared_counts[id] < shared_എണ്ണം) {
                shared_counts[id] = shared_എണ്ണം;
            }
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            നിർത്തുക;
    }
}

സ്ഥിതം size_t detect_അക്കം_smcus() {
    സ്ഥിരം auto runtime_feat = ggml_feats_get_arch64_runtime();
    എങ്കിൽ (!runtime_feat.has_sme) {
        തിരികെ 0;
    }

#എങ്കിൽ defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t അക്കം_private = 0;
    std::map<uint32_t, size_t> shared_counts;

    സ്ഥിരം std::vector<size_t> cpus = detect_cpu_ids();
    വേണ്ടി (സ്ഥിരം size_t cpu : cpus) {
        സ്ഥിരം std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        എങ്കിൽ (!file.is_open()) {
            തുടരുക;
        }

        uint64_t smidr = 0;
        എങ്കിൽ (!(file >> std::hex >> smidr)) {
            തുടരുക;
        }

        add_smcus_from_smidr(smidr, അക്കം_private, shared_counts);
    }

    size_t total = അക്കം_private;
    വേണ്ടി (സ്ഥിരം auto & entry : shared_counts) {
        total += entry.second;
    }
    തിരികെ total;

#അല്ലെങ്കിലെങ്കിൽ defined(__APPLE__) && defined(__aarch64__)
    പൂർണ്ണസംഖ്യ perf_levels = 0;
    size_t size = വലുപ്പം(perf_levels);
    എങ്കിൽ (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != വലുപ്പം(perf_levels) || perf_levels <= 0) {
        തിരികെ 0;
    }

    size_t units = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < perf_levels; ++i) {
        അക്ഷരം താക്കോൽ[64] = {};
        പൂർണ്ണസംഖ്യ physical_cpus = 0;
        പൂർണ്ണസംഖ്യ cpus_per_l2 = 0;

        snprintf(താക്കോൽ, വലുപ്പം(താക്കോൽ), "hw.perflevel%d.physicalcpu", i);
        size = വലുപ്പം(physical_cpus);
        എങ്കിൽ (sysctlbyname(താക്കോൽ, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != വലുപ്പം(physical_cpus) || physical_cpus <= 0) {
            തുടരുക;
        }

        snprintf(താക്കോൽ, വലുപ്പം(താക്കോൽ), "hw.perflevel%d.cpusperl2", i);
        size = വലുപ്പം(cpus_per_l2);
        എങ്കിൽ (sysctlbyname(താക്കോൽ, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != വലുപ്പം(cpus_per_l2) || cpus_per_l2 <= 0) {
            തുടരുക;
        }

        snprintf(താക്കോൽ, വലുപ്പം(താക്കോൽ), "hw.perflevel%d.name", i);
        size = 0;
        എങ്കിൽ (sysctlbyname(താക്കോൽ, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            തുടരുക;
        }

        std::string name(size, '\0');
        എങ്കിൽ (sysctlbyname(താക്കോൽ, &name[0], &size, nullptr, 0) != 0) {
            തുടരുക;
        }
        name.resize(size);
        ആകുമ്പോൾ (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        എങ്കിൽ (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    തിരികെ units;

#അല്ലെങ്കിലെങ്കിൽ defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    തിരികെ 0;

#അല്ലെങ്കിൽ
    തിരികെ 0;
#endif
}

സ്ഥിതം പൂർണ്ണസംഖ്യ parse_uint_env(സ്ഥിരം അക്ഷരം *s, സ്ഥിരം അക്ഷരം *name, തർക്കം *ok) {
    എങ്കിൽ (!s) { *ok = അസത്യം; തിരികെ 0; }
    അക്ഷരം *end = nullptr;
    long v = strtol(s, &end, 10);
    എങ്കിൽ (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = അസത്യം;
        തിരികെ 0;
    }
    എങ്കിൽ (v < 0 || v > INT_MAX) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = അസത്യം;
        തിരികെ 0;
    }
    *ok = സത്യം;
    തിരികെ (പൂർണ്ണസംഖ്യ)v;
}

സ്ഥിതം ശൂന്യം init_kleidiai_context(ശൂന്യം) {
    ggml_critical_section_start();
    സ്ഥിതം തർക്കം initialized = അസത്യം;

    എങ്കിൽ (!initialized) {
        initialized = സത്യം;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        സ്ഥിരം അക്ഷരം *env_sme         = getenv("GGML_KLEIDIAI_SME");
        സ്ഥിരം അക്ഷരം *env_threads     = getenv("GGML_TOTAL_THREADS");
        സ്ഥിരം അക്ഷരം *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        സ്ഥിരം auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        എങ്കിൽ (env_threads) {
            തർക്കം ok = അസത്യം;
            പൂർണ്ണസംഖ്യ hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            എങ്കിൽ (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        എങ്കിൽ (env_chunk_mult) {
            തർക്കം ok = അസത്യം;
            പൂർണ്ണസംഖ്യ multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            എങ്കിൽ (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        പൂർണ്ണസംഖ്യ sme_cores = 0;
        തർക്കം sme_env_ok = അസത്യം;
        തർക്കം sme_env_set = (env_sme != nullptr);

        സ്ഥിരം തർക്കം has_supported_sme_family = runtime_feat.has_sme;
        തർക്കം sme_cap_detected = അസത്യം;

        എങ്കിൽ (has_supported_sme_family) {
            detected_smcus = detect_അക്കം_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (പൂർണ്ണസംഖ്യ)detected_smcus : 1;

            എങ്കിൽ (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        എങ്കിൽ (sme_env_set) {
            തർക്കം ok = അസത്യം;
            പൂർണ്ണസംഖ്യ v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            എങ്കിൽ (ok) {
                എങ്കിൽ (has_supported_sme_family) {
                    sme_cores = v;
                } അല്ലെങ്കിൽ {
                    എങ്കിൽ (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } അല്ലെങ്കിൽ {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        എങ്കിൽ (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            എങ്കിൽ (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        എങ്കിൽ (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (പൂർണ്ണസംഖ്യ)ctx.features);
        } അല്ലെങ്കിൽ {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        എങ്കിൽ (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (പൂർണ്ണസംഖ്യ)ctx.features);
        } അല്ലെങ്കിൽ {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        എങ്കിൽ (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (പൂർണ്ണസംഖ്യ) ctx.features);
        } അല്ലെങ്കിൽ {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        എങ്കിൽ (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (പൂർണ്ണസംഖ്യ)ctx.features);
        } അല്ലെങ്കിൽ {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        സ്ഥിരം തർക്കം has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        എങ്കിൽ (has_selected_sme_family_kernel) {
            എങ്കിൽ (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } അല്ലെങ്കിൽ എങ്കിൽ (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } അല്ലെങ്കിൽ {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } അല്ലെങ്കിൽ {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_end();
}

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ kleidiai_sme_thread_cap() {
    തിരികെ ctx.sme_thread_cap;
}

സ്ഥിതം വരിയിൽ size_t align_up(size_t value, size_t alignment) {
    എങ്കിൽ (alignment == 0) {
        തിരികെ value;
    }
    സ്ഥിരം size_t remainder = value % alignment;
    തിരികെ remainder == 0 ? value : value + (alignment - remainder);
}

സ്ഥിതം വരിയിൽ size_t gcd_size(size_t a, size_t b) {
    ആകുമ്പോൾ (b != 0) {
        സ്ഥിരം size_t t = a % b;
        a = b;
        b = t;
    }
    തിരികെ a;
}

സ്ഥിതം വരിയിൽ തർക്കം lcm_size(size_t a, size_t b, size_t & result) {
    എങ്കിൽ (a == 0 || b == 0) {
        result = 0;
        തിരികെ അസത്യം;
    }
    സ്ഥിരം size_t g = gcd_size(a, b);
    സ്ഥിരം size_t q = a / g;
    എങ്കിൽ (q > SIZE_MAX / b) {
        തിരികെ അസത്യം;
    }
    result = q * b;
    തിരികെ സത്യം;
}

സ്ഥിതം വരിയിൽ size_t ceil_div_size(size_t a, size_t b) {
    തിരികെ b == 0 ? 0 : (a + b - 1) / b;
}

സ്ഥിതം വരിയിൽ size_t kleidiai_chunk_cols(size_t n, പൂർണ്ണസംഖ്യ nth_total, തർക്കം disable_chunking, size_t n_step) {
    സ്ഥിരം size_t multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    സ്ഥിരം size_t ഹാരകം = std::max<size_t>(1, (size_t) nth_total * multiplier);
    സ്ഥിരം size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, ഹാരകം)), n_step);
    തിരികെ chunk_cols ? chunk_cols : n_step;
}

ഘടന kleidiai_block_args {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

സ്ഥിതം വരിയിൽ kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    തിരഞ്ഞെടുപ്പ് (rhs_type) {
        അവസ്ഥ GGML_TYPE_Q4_0:
            തിരികെ { QK4_0, QK4_0, QK4_0 };
        അവസ്ഥ GGML_TYPE_Q4_K:
            തിരികെ { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        അവസ്ഥ GGML_TYPE_Q8_0:
            തിരികെ { 0, 0, QK8_0 };
        സ്ഥിരസ്ഥിതി:
            തിരികെ { 0, 0, 0 };
    }
}

സ്ഥിതം വരിയിൽ തർക്കം kleidiai_pack_fallback_allowed() {
    എങ്കിൽ (ctx.sme_thread_cap <= 0) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (ctx.thread_hint <= 0) {
        തിരികെ സത്യം;
    }
    തിരികെ ctx.thread_hint > ctx.sme_thread_cap;
}

ഘടന kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_എണ്ണം;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

സ്ഥിതം വരിയിൽ kleidiai_weight_header * kleidiai_weight_header_from_ptr(ശൂന്യം * data) {
    തിരികെ reinterpret_cast<kleidiai_weight_header *>(data);
}

സ്ഥിതം വരിയിൽ സ്ഥിരം kleidiai_weight_header * kleidiai_weight_header_from_ptr(സ്ഥിരം ശൂന്യം * data) {
    തിരികെ reinterpret_cast<സ്ഥിരം kleidiai_weight_header *>(data);
}

സ്ഥിതം വരിയിൽ തർക്കം kleidiai_is_weight_header_valid(സ്ഥിരം kleidiai_weight_header * header) {
    എങ്കിൽ (!header) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (header->slot_എണ്ണം == 0 || header->slot_എണ്ണം > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        തിരികെ അസത്യം;
    }
    തിരികെ സത്യം;
}

സ്ഥിതം വരിയിൽ uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, പൂർണ്ണസംഖ്യ slot) {
    എങ്കിൽ (!kleidiai_is_weight_header_valid(header)) {
        തിരികെ nullptr;
    }
    എങ്കിൽ (slot < 0 || slot >= header->slot_എണ്ണം) {
        തിരികെ nullptr;
    }
    തിരികെ reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

സ്ഥിതം വരിയിൽ സ്ഥിരം uint8_t * kleidiai_weight_slot_ptr(സ്ഥിരം kleidiai_weight_header * header, പൂർണ്ണസംഖ്യ slot) {
    എങ്കിൽ (!kleidiai_is_weight_header_valid(header)) {
        തിരികെ nullptr;
    }
    എങ്കിൽ (slot < 0 || slot >= header->slot_എണ്ണം) {
        തിരികെ nullptr;
    }
    തിരികെ reinterpret_cast<സ്ഥിരം uint8_t *>(header) + header->offsets[slot];
}

സ്ഥിതം വരിയിൽ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    തിരികെ ctx.kernels_q4;
}

സ്ഥിതം വരിയിൽ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    തിരികെ ctx.kernels_q4k;
}

സ്ഥിതം വരിയിൽ ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    തിരികെ ctx.kernels_q8;
}

സ്ഥിതം വരിയിൽ ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    തിരികെ ctx.kernels_f32;
}

template <typename SelectFallback>
സ്ഥിതം പൂർണ്ണസംഖ്യ kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    പൂർണ്ണസംഖ്യ count = 0;
    എങ്കിൽ (!primary) {
        തിരികെ 0;
    }
    out[count++] = primary;

    എങ്കിൽ (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        തിരികെ count;
    }

    എങ്കിൽ (is_sme_family(primary->required_cpu)) {
        സ്ഥിരം cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        എങ്കിൽ (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            എങ്കിൽ (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    തിരികെ count;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ kleidiai_collect_kernel_chain(സ്ഥിരം ഘടന ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    തിരികെ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { തിരികെ ggml_kleidiai_select_kernels(mask, op); });
}

സ്ഥിതം പൂർണ്ണസംഖ്യ kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    തിരികെ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { തിരികെ ggml_kleidiai_select_kernels_q4_0(mask); });
}

സ്ഥിതം പൂർണ്ണസംഖ്യ kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    തിരികെ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { തിരികെ ggml_kleidiai_select_kernels_q4_k(mask); });
}

സ്ഥിതം പൂർണ്ണസംഖ്യ kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    തിരികെ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { തിരികെ ggml_kleidiai_select_kernels_q8_0(mask); });
}

സ്ഥിതം പൂർണ്ണസംഖ്യ kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    തിരികെ kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { തിരികെ ggml_kleidiai_select_kernels_f32(mask); });
}

സ്ഥിതം വരിയിൽ int64_t ggml_ne(സ്ഥിരം ggml_tensor * tensor, പൂർണ്ണസംഖ്യ dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    തിരികെ tensor->ne[dim];
}

നാമാകാശ ggml::cpu::kleidiai {

സ്ഥിതം size_t round_down(size_t x, size_t y) {
    തിരികെ y == 0 ? x : x - (x % y);
}

സ്ഥിതം ശൂന്യം transpose_f32kxn_f16nxk(size_t n, size_t k, ദശാംശം * dst, സ്ഥിരം uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / വലുപ്പം(uint16_t);
    size_t dst_stride = n;

    വേണ്ടി (size_t k_idx = 0; k_idx < k; ++k_idx) {
        വേണ്ടി (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

വർഗ്ഗം tensor_traits : പൊതു ggml::cpu::tensor_traits {
    തർക്കം work_size(പൂർണ്ണസംഖ്യ /* n_threads */, സ്ഥിരം ഘടന ggml_tensor * op, size_t & size) override {
        എങ്കിൽ (op->op != GGML_OP_MUL_MAT) {
            തിരികെ അസത്യം;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        സ്ഥിരം പൂർണ്ണസംഖ്യ slot_എണ്ണം = kleidiai_collect_kernel_chain(op, kernel_chain);
        എങ്കിൽ (slot_എണ്ണം == 0) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം തർക്കം is_gemv = op->src[1]->ne[1] == 1;
        സ്ഥിരം size_t k = op->src[0]->ne[0];
        സ്ഥിരം size_t n = op->src[0]->ne[1];
        സ്ഥിരം size_t m = op->src[1]->ne[1];

        എങ്കിൽ (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            സ്ഥിരം kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            സ്ഥിരം size_t qk = block_args.pack_bl;

            size_t cursor = 0;
            തർക്കം any_slot = അസത്യം;

            വേണ്ടി (പൂർണ്ണസംഖ്യ slot = 0; slot < slot_എണ്ണം; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                എങ്കിൽ (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    തിരികെ അസത്യം;
                }

                സ്ഥിരം size_t mr = kernel->get_mr();
                സ്ഥിരം size_t kr = kernel->get_kr();
                സ്ഥിരം size_t sr = kernel->get_sr();

                സ്ഥിരം size_t packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = സത്യം;
            }

            എങ്കിൽ (!any_slot) {
                തിരികെ അസത്യം;
            }

            size = cursor;
            തിരികെ സത്യം;
        }

        എങ്കിൽ (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            എങ്കിൽ (is_gemv && op->src[1]->nb[0] == (int64_t) വലുപ്പം(ദശാംശം) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                തിരികെ സത്യം;
            }

            size_t cursor = 0;
            തർക്കം any_slot = അസത്യം;

            വേണ്ടി (പൂർണ്ണസംഖ്യ slot = 0; slot < slot_എണ്ണം; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                എങ്കിൽ (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    തിരികെ അസത്യം;
                }

                സ്ഥിരം size_t mr = kernel->get_mr();
                സ്ഥിരം size_t kr = kernel->get_kr();
                സ്ഥിരം size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
                any_slot = സത്യം;
            }

            എങ്കിൽ (!any_slot) {
                തിരികെ അസത്യം;
            }

            size = cursor;
            തിരികെ സത്യം;
        }

        എങ്കിൽ (op->src[0]->type == GGML_TYPE_F16) {
            സ്ഥിരം int64_t lhs_batch_size0 = op->src[1]->ne[2];
            സ്ഥിരം int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            സ്ഥിരം int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            തർക്കം any_slot = അസത്യം;

            വേണ്ടി (പൂർണ്ണസംഖ്യ slot = 0; slot < slot_എണ്ണം; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                എങ്കിൽ (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    തിരികെ അസത്യം;
                }

                സ്ഥിരം size_t mr = kernel->get_mr();
                സ്ഥിരം size_t kr = kernel->get_kr();
                സ്ഥിരം size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = സത്യം;
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ slot = 0; slot < slot_എണ്ണം; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                എങ്കിൽ (!kernel || !kernels->rhs_info.packed_size_ex) {
                    തിരികെ അസത്യം;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * വലുപ്പം(ദശാംശം);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * വലുപ്പം(ദശാംശം);

            എങ്കിൽ (!any_slot) {
                തിരികെ അസത്യം;
            }

            size = cursor;
            തിരികെ സത്യം;
        }

        തിരികെ അസത്യം;
    }

    തർക്കം compute_forward(ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst) override {
        എങ്കിൽ (dst->op == GGML_OP_MUL_MAT) {
            എങ്കിൽ (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                തിരികെ compute_forward_qx(params, dst);
            } അല്ലെങ്കിൽ എങ്കിൽ (dst->src[0]->type == GGML_TYPE_F32) {
                തിരികെ compute_forward_f32(params, dst);
            } അല്ലെങ്കിൽ എങ്കിൽ (dst->src[0]->type == GGML_TYPE_F16) {
                തിരികെ compute_forward_fp16(params, dst);
            }
        }
        തിരികെ അസത്യം;
    }

    തർക്കം compute_forward_f32(ggml_compute_params * params, ഘടന ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        സ്ഥിരം ggml_tensor * src0 = dst->src[0];
        സ്ഥിരം ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ദ്വിമാന_OP_LOCALS

        എങ്കിൽ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            തിരികെ അസത്യം;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        എങ്കിൽ (!kernels) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം size_t k = ne00;
        സ്ഥിരം size_t m = ne11;
        സ്ഥിരം size_t n = ne01;
        സ്ഥിരം തർക്കം use_gemv = m == 1 && src1->nb[0] == (int64_t) വലുപ്പം(ദശാംശം) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        എങ്കിൽ (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_size_ex || !lhs_info->pack_func_ex)) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        സ്ഥിരം തർക്കം has_header = kleidiai_is_weight_header_valid(header);

        സ്ഥിരം uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<സ്ഥിരം uint8_t *>(src0->data);
        എങ്കിൽ (!rhs_base) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth > 0 ? params->nth : 1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;

        സ്ഥിരം size_t mr = kernel->get_mr();
        സ്ഥിരം size_t kr = kernel->get_kr();
        സ്ഥിരം size_t sr = kernel->get_sr();

        സ്ഥിരം size_t lhs_packed_size = use_gemv ? 0 : lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        എങ്കിൽ (!use_gemv) {
            GGML_ASSERT(lhs_packed_size <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        സ്ഥിരം size_t dst_stride = dst->nb[1];
        സ്ഥിരം size_t n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        സ്ഥിരം തർക്കം disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) INT_MAX);

        സ്ഥിരം int64_t batch_size = ne12 * ne13;
        വേണ്ടി (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            സ്ഥിരം int64_t i12 = batch_idx % ne12;
            സ്ഥിരം int64_t i13 = batch_idx / ne12;
            സ്ഥിരം uint8_t * lhs_batch_base = static_cast<സ്ഥിരം uint8_t *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];

            എങ്കിൽ (!use_gemv) {
                സ്ഥിരം int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                സ്ഥിരം int64_t use_threads = std::min<int64_t>(nth, max_threads);

                എങ്കിൽ (ith < use_threads) {
                    സ്ഥിരം int64_t അക്കം_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), mr);
                    സ്ഥിരം int64_t അക്കം_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * അക്കം_m_per_thread0;

                    സ്ഥിരം int64_t m_start = (int64_t)ith * അക്കം_m_per_thread0;
                    സ്ഥിരം int64_t m_എണ്ണം = (ith == use_threads - 1) ? അക്കം_m_per_thread_N_1 : അക്കം_m_per_thread0;

                    സ്ഥിരം size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    സ്ഥിരം size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    സ്ഥിരം size_t row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_എണ്ണം;
                    int64_t cur       = m_start;

                    ആകുമ്പോൾ (remaining > 0) {
                        സ്ഥിരം int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        സ്ഥിരം size_t src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        സ്ഥിരം ശൂന്യം * src_ptr = lhs_batch_base + src_off;
                        സ്ഥിരം size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        ശൂന്യം * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            എങ്കിൽ (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            സ്ഥിരം size_t chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (size_t) INT_MAX);

            പൂർണ്ണസംഖ്യ current_col = ggml_threadpool_chunk_add(params->threadpool, (പൂർണ്ണസംഖ്യ) chunk_cols);
            ആകുമ്പോൾ ((size_t) current_col < n) {
                സ്ഥിരം size_t n_start = (size_t) current_col;
                സ്ഥിരം size_t n_to_process = std::min(chunk_cols, n - n_start);

                എങ്കിൽ (n_to_process > 0) {
                    സ്ഥിരം size_t lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    സ്ഥിരം size_t rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    സ്ഥിരം size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    സ്ഥിരം ശൂന്യം * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    സ്ഥിരം ശൂന്യം * rhs_ptr = rhs_base + rhs_packed_offset;
                    ദശാംശം * dst_ptr      = reinterpret_cast<ദശാംശം *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          വലുപ്പം(ദശാംശം),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (പൂർണ്ണസംഖ്യ) chunk_cols);
            }

            എങ്കിൽ (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        തിരികെ സത്യം;
    }

    തർക്കം compute_forward_fp16(ggml_compute_params * params, ഘടന ggml_tensor * dst) {
        സ്ഥിരം ggml_tensor * src0 = dst->src[0];
        സ്ഥിരം ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ദ്വിമാന_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        എങ്കിൽ (!kernels) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം തർക്കം is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        എങ്കിൽ (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;

        സ്ഥിരം int64_t lhs_batch_size0 = ne12;
        സ്ഥിരം int64_t rhs_batch_size0 = ne02;
        സ്ഥിരം int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        സ്ഥിരം int64_t r = lhs_batch_size0 / rhs_batch_size0;

        സ്ഥിരം int64_t m_group = ne11;
        സ്ഥിരം int64_t m       = m_group;
        സ്ഥിരം int64_t n       = ne01;
        സ്ഥിരം int64_t k       = ne00;

        സ്ഥിരം size_t lhs_stride = src1->nb[1];
        സ്ഥിരം size_t rhs_stride = src0->nb[1];
        സ്ഥിരം size_t dst_stride = dst->nb[1];

        സ്ഥിരം int64_t mr = (int64_t) kernel->get_mr();
        സ്ഥിരം int64_t nr = (int64_t) kernel->get_nr();
        സ്ഥിരം int64_t kr = (int64_t) kernel->get_kr();
        സ്ഥിരം int64_t sr = (int64_t) kernel->get_sr();

        സ്ഥിരം size_t lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        സ്ഥിരം size_t rhs_packed_size = kernels->rhs_info.packed_size_ex(n, k, nr, kr, 0);
        സ്ഥിരം size_t kxn_size        = k * n * വലുപ്പം(ദശാംശം);
        സ്ഥിരം size_t bias_size       = n * വലുപ്പം(ദശാംശം);

        സ്ഥിരം size_t wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        വേണ്ടി (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            സ്ഥിരം int64_t rhs_batch_idx = batch_idx / r;
            സ്ഥിരം uint8_t * rhs_batch_base = static_cast<സ്ഥിരം uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                സ്ഥിരം int64_t m_roundup_mr = kai_roundup(m, mr);
                സ്ഥിരം int64_t അക്കം_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                എങ്കിൽ (ith < അക്കം_threads) {
                    സ്ഥിരം int64_t അക്കം_m_per_thread0   = round_down((size_t)(m_roundup_mr / അക്കം_threads), (size_t)mr);
                    സ്ഥിരം int64_t അക്കം_m_per_thread_N_1 = m - (അക്കം_threads - 1) * അക്കം_m_per_thread0;

                    സ്ഥിരം int64_t m_start = ith * അക്കം_m_per_thread0;
                    സ്ഥിരം int64_t m_എണ്ണം = (ith == അക്കം_threads - 1) ? അക്കം_m_per_thread_N_1 : അക്കം_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    സ്ഥിരം size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    സ്ഥിരം size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    സ്ഥിരം size_t row_stride_bytes = (next_block_off - base_packed_off) / (size_t)mr;

                    int64_t remaining = m_എണ്ണം;
                    int64_t cur       = m_start;

                    ആകുമ്പോൾ (remaining > 0) {
                        സ്ഥിരം int64_t row_in_group = cur;
                        സ്ഥിരം int64_t avail        = m_group - row_in_group;
                        സ്ഥിരം int64_t take         = std::min(avail, remaining);

                        സ്ഥിരം uint8_t * lhs_batch_base = static_cast<സ്ഥിരം uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        സ്ഥിരം ശൂന്യം * src_ptr = lhs_batch_base + (size_t)row_in_group * lhs_stride;
                        സ്ഥിരം size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        ശൂന്യം * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            എങ്കിൽ (ith == 0) {
                memset(bias, 0, (size_t)n * വലുപ്പം(ദശാംശം));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<ദശാംശം *>(rhs_kxn),
                                        reinterpret_cast<സ്ഥിരം uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * വലുപ്പം(ദശാംശം),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                സ്ഥിരം int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t അക്കം_threads_n = KAI_MIN(n / n_step, nth);
                എങ്കിൽ (അക്കം_threads_n <= 0) {
                    അക്കം_threads_n = 1;
                }

                എങ്കിൽ (ith < അക്കം_threads_n) {
                    സ്ഥിരം int64_t അക്കം_n_per_thread0   = round_down((size_t)(n / അക്കം_threads_n), (size_t)n_step);
                    സ്ഥിരം int64_t അക്കം_n_per_thread_N_1 = n - (അക്കം_threads_n - 1) * അക്കം_n_per_thread0;

                    സ്ഥിരം int64_t n_start      = ith * അക്കം_n_per_thread0;
                    സ്ഥിരം int64_t n_to_process = (ith == അക്കം_threads_n - 1) ? അക്കം_n_per_thread_N_1 : അക്കം_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    സ്ഥിരം size_t lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    സ്ഥിരം size_t rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    സ്ഥിരം size_t dst_offset         = kernel->get_dst_offset((size_t)0, (size_t)n_start, dst_stride);

                    സ്ഥിരം ശൂന്യം * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    സ്ഥിരം ശൂന്യം * rhs_ptr = rhs_packed + rhs_packed_offset;
                    ദശാംശം * dst_ptr      = reinterpret_cast<ദശാംശം *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, വലുപ്പം(ദശാംശം), -FLT_MAX, FLT_MAX);
                }
            }

            എങ്കിൽ (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        തിരികെ സത്യം;
    }

    തർക്കം compute_forward_qx(ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        സ്ഥിരം ggml_tensor * src0 = dst->src[0];
        സ്ഥിരം ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ദ്വിമാന_OP_LOCALS

        സ്ഥിരം kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        സ്ഥിരം തർക്കം has_header = kleidiai_is_weight_header_valid(header);
        സ്ഥിരം തർക്കം is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        സ്ഥിരം പൂർണ്ണസംഖ്യ slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        auto weight_for_slot = [&](പൂർണ്ണസംഖ്യ slot_index, size_t & size_out) -> സ്ഥിരം uint8_t * {
            എങ്കിൽ (slot_index < 0 || slot_index >= slot_total) {
                തിരികെ nullptr;
            }
            എങ്കിൽ (has_header) {
                എങ്കിൽ (slot_index < header->slot_എണ്ണം) {
                    size_out = static_cast<size_t>(header->sizes[slot_index]);
                    തിരികെ kleidiai_weight_slot_ptr(header, slot_index);
                }
                തിരികെ nullptr;
            }
            എങ്കിൽ (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                തിരികെ static_cast<സ്ഥിരം uint8_t *>(src0->data);
            }
            തിരികെ nullptr;
        };

        ഘടന runtime_slot {
            പൂർണ്ണസംഖ്യ slot_index;
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
            പൂർണ്ണസംഖ്യ assigned_threads;
            പൂർണ്ണസംഖ്യ thread_begin;
            പൂർണ്ണസംഖ്യ thread_end;
            സ്ഥിരം uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        പൂർണ്ണസംഖ്യ runtime_എണ്ണം = 0;

        വേണ്ടി (പൂർണ്ണസംഖ്യ slot = 0; slot < slot_total && runtime_എണ്ണം < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            എങ്കിൽ (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                തുടരുക;
            }

            size_t rhs_size = 0;
            സ്ഥിരം uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            എങ്കിൽ (!rhs_ptr || rhs_size == 0) {
                തുടരുക;
            }

            സ്ഥിരം kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

            runtime[runtime_എണ്ണം] = {
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
            ++runtime_എണ്ണം;
        }

        എങ്കിൽ (runtime_എണ്ണം == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            തിരികെ അസത്യം;
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ nth_total = params->nth > 0 ? params->nth : 1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ith_total = params->ith;

        പൂർണ്ണസംഖ്യ sme_slot = -1;
        പൂർണ്ണസംഖ്യ non_sme_slot = -1;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < runtime_എണ്ണം; ++i) {
            എങ്കിൽ (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                നിർത്തുക;
            }
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < runtime_എണ്ണം; ++i) {
            എങ്കിൽ (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                നിർത്തുക;
            }
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ sme_cap_limit = ctx.sme_thread_cap;
        സ്ഥിരം തർക്കം use_hybrid = sme_cap_limit > 0 &&
                                 runtime_എണ്ണം > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_cols_per_thread = 0;
        എങ്കിൽ (runtime_എണ്ണം > 0 && nth_total > 0) {
            min_cols_per_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        സ്ഥിരം തർക്കം too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        സ്ഥിരം തർക്കം hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        എങ്കിൽ (!hybrid_enabled) {
            പൂർണ്ണസംഖ്യ chosen_slot = 0;
            എങ്കിൽ (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } അല്ലെങ്കിൽ എങ്കിൽ (runtime_എണ്ണം > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            എങ്കിൽ (chosen_slot != 0 && chosen_slot < runtime_എണ്ണം) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_എണ്ണം = runtime_എണ്ണം > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            എങ്കിൽ (runtime_എണ്ണം > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        സ്ഥിരം തർക്കം sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        പൂർണ്ണസംഖ്യ sme_cap = sme_single_only ? nth_total : kleidiai_sme_thread_cap();
        എങ്കിൽ (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        പൂർണ്ണസംഖ്യ threads_remaining = nth_total;
        എങ്കിൽ (sme_slot != -1) {
            പൂർണ്ണസംഖ്യ sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        പൂർണ്ണസംഖ്യ fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        പൂർണ്ണസംഖ്യ fallback_എണ്ണം = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < runtime_എണ്ണം; ++i) {
            എങ്കിൽ (i == sme_slot) {
                തുടരുക;
            }
            fallback_indices[fallback_എണ്ണം++] = i;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ fi = 0; fi < fallback_എണ്ണം; ++fi) {
            എങ്കിൽ (threads_remaining <= 0) {
                നിർത്തുക;
            }
            സ്ഥിരം പൂർണ്ണസംഖ്യ slot_index = fallback_indices[fi];
            സ്ഥിരം പൂർണ്ണസംഖ്യ slots_ഇടത് = fallback_എണ്ണം - fi;
            പൂർണ്ണസംഖ്യ share = (threads_remaining + slots_ഇടത് - 1) / slots_ഇടത്;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        പൂർണ്ണസംഖ്യ thread_cursor = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < runtime_എണ്ണം; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        സ്ഥിരം പൂർണ്ണസംഖ്യ active_threads = thread_cursor;

        പൂർണ്ണസംഖ്യ local_slot = -1;
        പൂർണ്ണസംഖ്യ local_ith  = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < runtime_എണ്ണം; ++i) {
            എങ്കിൽ (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                നിർത്തുക;
            }
        }
        സ്ഥിരം size_t k = ne00;
        സ്ഥിരം size_t m = ne11;
        സ്ഥിരം size_t n = ne01;

        size_t cursor = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < runtime_എണ്ണം; ++i) {
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_step = 1;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < runtime_എണ്ണം; ++i) {
            എങ്കിൽ (runtime[i].assigned_threads == 0) {
                തുടരുക;
            }
            size_t next_step = 0;
            എങ്കിൽ (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                തിരികെ അസത്യം;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        സ്ഥിരം തർക്കം disable_chunking = ggml_is_numa();
        സ്ഥിരം size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        സ്ഥിരം size_t chunk_ഹാരകം = (active_threads == 1 || disable_chunking) ? (size_t) active_threads : (size_t) active_threads * chunk_multiplier;
        size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, chunk_ഹാരകം)), common_step);
        എങ്കിൽ (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        സ്ഥിരം size_t nchunk_size = std::max<size_t>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (size_t)INT_MAX);
        സ്ഥിരം പൂർണ്ണസംഖ്യ nchunk = (പൂർണ്ണസംഖ്യ)nchunk_size;
        സ്ഥിരം size_t dst_stride = dst->nb[1];

        auto run_chunk = [&](runtime_slot & slot, size_t global_start, size_t cols, uint8_t * dst_batch_base) {
            സ്ഥിരം size_t rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            സ്ഥിരം size_t dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            സ്ഥിരം uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            സ്ഥിരം uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            ദശാംശം * dst_ptr         = reinterpret_cast<ദശാംശം *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       വലുപ്പം(ദശാംശം),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        വേണ്ടി (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            സ്ഥിരം uint8_t * lhs_batch_base = static_cast<സ്ഥിരം uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            എങ്കിൽ (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                സ്ഥിരം int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                സ്ഥിരം int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                എങ്കിൽ (local_ith < use_threads) {
                    സ്ഥിരം int64_t അക്കം_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), slot.mr);
                    സ്ഥിരം int64_t അക്കം_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * അക്കം_m_per_thread0;

                    സ്ഥിരം int64_t m_start = (int64_t)local_ith * അക്കം_m_per_thread0;
                    സ്ഥിരം int64_t m_എണ്ണം = (local_ith == use_threads - 1) ? അക്കം_m_per_thread_N_1 : അക്കം_m_per_thread0;

                    സ്ഥിരം size_t base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    സ്ഥിരം size_t next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    സ്ഥിരം size_t row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_എണ്ണം;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    ആകുമ്പോൾ (remaining > 0) {
                        സ്ഥിരം int64_t row_in_group = cur;
                        സ്ഥിരം int64_t avail        = (int64_t)m - row_in_group;
                        സ്ഥിരം int64_t take         = std::min(avail, remaining);

                        സ്ഥിരം size_t src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        സ്ഥിരം ശൂന്യം * src_ptr = lhs_batch_base + src_off;
                        സ്ഥിരം size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        ശൂന്യം * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            എങ്കിൽ (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            എങ്കിൽ (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                പൂർണ്ണസംഖ്യ current_chunk = ith_total;
                ആകുമ്പോൾ (current_chunk < nchunk) {
                    സ്ഥിരം size_t global_start = (size_t)current_chunk * chunk_cols;
                    എങ്കിൽ (global_start >= n) {
                        നിർത്തുക;
                    }

                    സ്ഥിരം size_t cols = std::min(chunk_cols, n - global_start);
                    എങ്കിൽ (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            എങ്കിൽ (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        തിരികെ സത്യം;
    }

പൊതു:
    പൂർണ്ണസംഖ്യ repack(ഘടന ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        സ്ഥിരം size_t n = tensor->ne[1];
        സ്ഥിരം size_t k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        എങ്കിൽ (!header) {
            തിരികെ -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_എണ്ണം = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        size_t cursor = വലുപ്പം(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        സ്ഥിരം തർക്കം want_q4k = tensor->type == GGML_TYPE_Q4_K;
        സ്ഥിരം തർക്കം want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        സ്ഥിരം തർക്കം want_f32 = tensor->type == GGML_TYPE_F32;
        സ്ഥിരം പൂർണ്ണസംഖ്യ slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        സ്ഥിരം തർക്കം allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<ദശാംശം>  scales;
        std::vector<ദശാംശം>  bias;
        std::vector<ദശാംശം> row_f32;

        എങ്കിൽ (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            സ്ഥിരം size_t row_stride = tensor->nb[1];
            സ്ഥിരം size_t k_blocks   = (k + QK8_0 - 1) / QK8_0;

            എങ്കിൽ (tensor->type == GGML_TYPE_Q6_K) {
                എങ്കിൽ (k % QK_K != 0 || k % QK8_0 != 0) {
                    തിരികെ -1;
                }
                row_f32.resize(k);
            }

            വേണ്ടി (size_t row = 0; row < n; ++row) {
                സ്ഥിരം uint8_t * row_data = static_cast<സ്ഥിരം uint8_t *>(data) + row * row_stride;

                എങ്കിൽ (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<സ്ഥിരം block_q6_K *>(row_data), row_f32.data(), k);
                }

                ദശാംശം max_abs = 0.0f;
                എങ്കിൽ (tensor->type == GGML_TYPE_Q6_K) {
                    വേണ്ടി (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::fabs(row_f32[linear_idx]));
                    }
                } അല്ലെങ്കിൽ {
                    സ്ഥിരം block_q8_0 * row_blocks = reinterpret_cast<സ്ഥിരം block_q8_0 *>(row_data);
                    വേണ്ടി (size_t block = 0; block < k_blocks; ++block) {
                        സ്ഥിരം block_q8_0 & blk = row_blocks[block];
                        സ്ഥിരം ദശാംശം d = GGML_FP16_TO_FP32(blk.d);
                        വേണ്ടി (size_t l = 0; l < QK8_0; ++l) {
                            സ്ഥിരം size_t linear_idx = block * QK8_0 + l;
                            എങ്കിൽ (linear_idx >= k) {
                                നിർത്തുക;
                            }
                            സ്ഥിരം ദശാംശം value = d * static_cast<ദശാംശം>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                ദശാംശം scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                സ്ഥിരം ദശാംശം inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                എങ്കിൽ (tensor->type == GGML_TYPE_Q6_K) {
                    വേണ്ടി (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } അല്ലെങ്കിൽ {
                    സ്ഥിരം block_q8_0 * row_blocks = reinterpret_cast<സ്ഥിരം block_q8_0 *>(row_data);
                    വേണ്ടി (size_t block = 0; block < k_blocks; ++block) {
                        സ്ഥിരം block_q8_0 & blk = row_blocks[block];
                        സ്ഥിരം ദശാംശം d = GGML_FP16_TO_FP32(blk.d);
                        വേണ്ടി (size_t l = 0; l < QK8_0; ++l) {
                            സ്ഥിരം size_t linear_idx = block * QK8_0 + l;
                            എങ്കിൽ (linear_idx >= k) {
                                നിർത്തുക;
                            }
                            സ്ഥിരം ദശാംശം value = d * static_cast<ദശാംശം>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        എങ്കിൽ (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            എങ്കിൽ (!allow_fallback && slot > 0) {
                നിർത്തുക;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            എങ്കിൽ (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                തുടരുക;
            }

            സ്ഥിരം size_t nr = kernel->get_nr();
            സ്ഥിരം size_t kr = kernel->get_kr();
            സ്ഥിരം size_t sr = kernel->get_sr();
            സ്ഥിരം ggml_type rhs_type = kernels->rhs_type;
            എങ്കിൽ (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            സ്ഥിരം size_t block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            എങ്കിൽ (block_len == SIZE_MAX) {
                തുടരുക;
            }

            സ്ഥിരം size_t packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            സ്ഥിരം size_t aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            എങ്കിൽ (rhs_type == GGML_TYPE_Q4_0) {
                ഘടന kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<സ്ഥിരം uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } അല്ലെങ്കിൽ എങ്കിൽ (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       data, nullptr, nullptr, dst_ptr, 0, nullptr);
            } അല്ലെങ്കിൽ എങ്കിൽ (rhs_type == GGML_TYPE_Q8_0) {
                ഘടന kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } അല്ലെങ്കിൽ എങ്കിൽ (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
            } അല്ലെങ്കിൽ {
                തുടരുക;
            }

            header->offsets[header->slot_എണ്ണം] = aligned_cursor;
            header->sizes[header->slot_എണ്ണം]   = packed_size;
            ++header->slot_എണ്ണം;

            cursor = aligned_cursor + packed_size;
        }

        എങ്കിൽ (header->slot_എണ്ണം == 0) {
            header->magic   = 0;
            header->version = 0;
            memcpy(tensor->data, data, data_size);
        }

        തിരികെ 0;
    }
};

സ്ഥിതം ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, ഘടന ggml_tensor *) {
    സ്ഥിതം tensor_traits traits;
    തിരികെ &traits;
}
}  // namespace ggml::cpu::kleidiai

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, ഘടന ggml_tensor * tensor) {
    tensor->extra = (ശൂന്യം *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    തിരികെ GGML_STATUS_SUCCESS;
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, ഘടന ggml_tensor * tensor,
                                                       സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    തിരികെ "CPU_KLEIDIAI";
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    എങ്കിൽ (buffer == nullptr) {
        തിരികെ nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    തിരികെ buffer;
}

സ്ഥിതം size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    തിരികെ TENSOR_ALIGNMENT;
}

സ്ഥിതം size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ഘടന ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    എങ്കിൽ (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        തിരികെ ggml_nbytes(tensor);
    }

    സ്ഥിരം size_t n = tensor->ne[1];
    സ്ഥിരം size_t k = tensor->ne[0];

    size_t cursor = വലുപ്പം(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    സ്ഥിരം തർക്കം want_q4k = tensor->type == GGML_TYPE_Q4_K;
    സ്ഥിരം തർക്കം want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    സ്ഥിരം തർക്കം want_f32 = tensor->type == GGML_TYPE_F32;
    സ്ഥിരം പൂർണ്ണസംഖ്യ slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    സ്ഥിരം തർക്കം allow_fallback = kleidiai_pack_fallback_allowed();

    size_t slot_എണ്ണം = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ slot = 0; slot < slot_total; ++slot) {
        എങ്കിൽ (!allow_fallback && slot > 0) {
            നിർത്തുക;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        എങ്കിൽ (!kernels) {
            തുടരുക;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        എങ്കിൽ (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            തുടരുക;
        }

        സ്ഥിരം ggml_type rhs_type = kernels->rhs_type;
        സ്ഥിരം size_t block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        എങ്കിൽ (block_len == SIZE_MAX) {
            തുടരുക;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_എണ്ണം;
    }

    എങ്കിൽ (slot_എണ്ണം == 0) {
        തിരികെ ggml_nbytes(tensor);
    }

    തിരികെ std::max(cursor, ggml_nbytes(tensor));
}

നാമാകാശ ggml::cpu::kleidiai {
വർഗ്ഗം extra_buffer_type : ggml::cpu::extra_buffer_type {
    തർക്കം supports_op(ggml_backend_dev_t, സ്ഥിരം ഘടന ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        എങ്കിൽ (op->op != GGML_OP_MUL_MAT) {
            തിരികെ അസത്യം;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        സ്ഥിരം പൂർണ്ണസംഖ്യ slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        സ്ഥിരം തർക്കം src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft->context == ഇത് &&
            slot_total > 0;

        എങ്കിൽ ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            എങ്കിൽ (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                തിരികെ അസത്യം;
            }

            എങ്കിൽ (op->src[0]->type == GGML_TYPE_Q4_K) {
                എങ്കിൽ (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    തിരികെ അസത്യം;
                }
                തിരികെ ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == വലുപ്പം(ദശാംശം) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * വലുപ്പം(ദശാംശം));
            }

            എങ്കിൽ (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                എങ്കിൽ (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    തിരികെ സത്യം;
                }
                തിരികെ അസത്യം;
            }

            എങ്കിൽ (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                തിരികെ അസത്യം;
            }

            തിരികെ സത്യം;
        }

        തിരികെ അസത്യം;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(സ്ഥിരം ഘടന ggml_tensor * op) override {
        എങ്കിൽ (op->op == GGML_OP_MUL_MAT) {
            എങ്കിൽ (op->src[0]->buffer && op->src[0]->buffer->buft->context == ഇത്) {
                തിരികെ (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } അല്ലെങ്കിൽ {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                എങ്കിൽ (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    സ്ഥിതം std::atomic<തർക്കം> warned(അസത്യം);
                    എങ്കിൽ (!warned.exchange(സത്യം)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                എങ്കിൽ (op->src[0]->type != GGML_TYPE_F16) {
                    തിരികെ nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                സ്ഥിരം പൂർണ്ണസംഖ്യ slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                എങ്കിൽ (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    എങ്കിൽ ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        തിരികെ nullptr;
                    }
                    തിരികെ ggml::cpu::kleidiai::get_tensor_traits(ശൂന്യ, ശൂന്യ);
                }
            }
        }
        തിരികെ nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(ശൂന്യം) {
    സ്ഥിതം ggml::cpu::kleidiai::extra_buffer_type ctx;
    സ്ഥിതം ഘടന ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .get_alloc_size_n     = */ ശൂന്യ,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    തിരികെ &ggml_backend_cpu_buffer_type_kleidiai;
}
