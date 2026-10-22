// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#અન્તર્ભાવ <arm_neon.h>
#અન્તર્ભાવ <cassert>
#અન્તર્ભાવ <cstdio>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <atomic>
#અન્તર્ભાવ <cfloat>
#અન્તર્ભાવ <cctype>
#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <stdexcept>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <vector>
#અન્તર્ભાવ <array>
#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <fstream>
#અન્તર્ભાવ <map>
#અન્તર્ભાવ <iostream>
#અન્તર્ભાવ <climits>
#અન્તર્ભાવ <charconv>
#અન્તર્ભાવ <system_error>
#જો defined(__linux__)
#અન્તર્ભાવ <asm/hwcap.h>
#અન્તર્ભાવ <dirent.h>
#અન્તર્ભાવ <sys/auxv.h>
#અન્તર્ભાવ <sys/types.h>
#અન્તર્ભાવ <sys/stat.h>
#અન્તર્ભાવ <unistd.h>
#અથવાજો defined(__APPLE__)
#અન્તર્ભાવ <sys/sysctl.h>
#અન્તર્ભાવ <sys/types.h>
#endif

#અન્તર્ભાવ "kleidiai.h"

#અન્તર્ભાવ "ggml-cpu.h"
#અન્તર્ભાવ "ggml-cpu-impl.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-feats.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-quants.h"
#અન્તર્ભાવ "ggml-threading.h"
#અન્તર્ભાવ "traits.h"

#અન્તર્ભાવ "kernels.h"

#અન્તર્ભાવ "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#અન્તર્ભાવ "ggml-common.h"

સ્થિર constexpr પૂર્ણાંક      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
સ્થિર constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
સ્થિર constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
સ્થિર constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
સ્થિર constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

સંરચના ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    પૂર્ણાંક sme_thread_cap; // <= 0 means "SME disabled/unknown"
    પૂર્ણાંક thread_hint;    // <= 0 means "no hint"
    પૂર્ણાંક chunk_multiplier;
} સ્થિર ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

સ્થિર અંતઃરેખ તાર્કિક is_sme_family(cpu_feature f) {
    પાછા (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

સ્થિર અચલ અક્ષર* cpu_feature_to_string(cpu_feature f) {
    જો (f == CPU_FEATURE_NONE) {
        પાછા "NONE";
    } નહીંતર જો ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        પાછા "SME2";
    } નહીંતર જો ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        પાછા "SME";
    } નહીંતર જો ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        પાછા "SVE";
    }
    નહીંતર જો ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        પાછા "I8MM";
    } નહીંતર જો ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        પાછા "DOTPROD";
    }
    નહીંતર {
        પાછા "UNKNOWN";
    }
}

#જો defined(__linux__) && defined(__aarch64__)
સ્થિર તાર્કિક parse_cpu_dir_name(અચલ અક્ષર* name, size_t* cpu) {
    જો (strncmp(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        પાછા ખોટું;
    }

    અચલ અક્ષર* first = name + 3;
    અચલ અક્ષર* last = name + strlen(name);

    size_t value = 0;
    અચલ auto [end, ec] = std::from_chars(first, last, value, 10);

    જો (ec != std::errc{} || end != last) {
        પાછા ખોટું;
    }

    *cpu = value;
    પાછા સાચું;
}

સ્થિર std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    જો (dir == nullptr) {
        પાછા cpus;
    }

    જ્યારે (dirent * entry = readdir(dir)) {
        size_t cpu = 0;
        જો (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    પાછા cpus;
}
#endif

#જો defined(__APPLE__) && defined(__aarch64__)
સ્થિર તાર્કિક apple_sme_counted_perf_level(std::string name) {
    માટે (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (અક્ષર) std::tolower((unsigned અક્ષર) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    પાછા name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

સ્થિર શૂન્ય add_smcus_from_smidr(uint64_t smidr, size_t & અંક_private, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    અચલ uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    અચલ uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    અચલ size_t shared_ગણતરી = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    અચલ uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    અચલ uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    અચલ uint32_t id = (affinity2 << 12) | affinity;

    જો (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    પસંદગી (sh) {
        સ્થિતિ 2: // private SMCU
            ++અંક_private;
            તોડો;
        સ્થિતિ 3: // shared SMCU
            જો (shared_counts[id] < shared_ગણતરી) {
                shared_counts[id] = shared_ગણતરી;
            }
            તોડો;
        સ્થિતિ 0:
            જો (id == 0) {
                ++અંક_private;
            } નહીંતર જો (shared_counts[id] < shared_ગણતરી) {
                shared_counts[id] = shared_ગણતરી;
            }
            તોડો;
        મૂળભૂત:
            તોડો;
    }
}

સ્થિર size_t detect_અંક_smcus() {
    અચલ auto runtime_feat = ggml_feats_get_arch64_runtime();
    જો (!runtime_feat.has_sme) {
        પાછા 0;
    }

#જો defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t અંક_private = 0;
    std::map<uint32_t, size_t> shared_counts;

    અચલ std::vector<size_t> cpus = detect_cpu_ids();
    માટે (અચલ size_t cpu : cpus) {
        અચલ std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        જો (!file.is_open()) {
            ચાલુ;
        }

        uint64_t smidr = 0;
        જો (!(file >> std::hex >> smidr)) {
            ચાલુ;
        }

        add_smcus_from_smidr(smidr, અંક_private, shared_counts);
    }

    size_t total = અંક_private;
    માટે (અચલ auto & entry : shared_counts) {
        total += entry.second;
    }
    પાછા total;

#અથવાજો defined(__APPLE__) && defined(__aarch64__)
    પૂર્ણાંક perf_levels = 0;
    size_t size = કદ(perf_levels);
    જો (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != કદ(perf_levels) || perf_levels <= 0) {
        પાછા 0;
    }

    size_t units = 0;
    માટે (પૂર્ણાંક i = 0; i < perf_levels; ++i) {
        અક્ષર ચાવી[64] = {};
        પૂર્ણાંક physical_cpus = 0;
        પૂર્ણાંક cpus_per_l2 = 0;

        snprintf(ચાવી, કદ(ચાવી), "hw.perflevel%d.physicalcpu", i);
        size = કદ(physical_cpus);
        જો (sysctlbyname(ચાવી, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != કદ(physical_cpus) || physical_cpus <= 0) {
            ચાલુ;
        }

        snprintf(ચાવી, કદ(ચાવી), "hw.perflevel%d.cpusperl2", i);
        size = કદ(cpus_per_l2);
        જો (sysctlbyname(ચાવી, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != કદ(cpus_per_l2) || cpus_per_l2 <= 0) {
            ચાલુ;
        }

        snprintf(ચાવી, કદ(ચાવી), "hw.perflevel%d.name", i);
        size = 0;
        જો (sysctlbyname(ચાવી, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            ચાલુ;
        }

        std::string name(size, '\0');
        જો (sysctlbyname(ચાવી, &name[0], &size, nullptr, 0) != 0) {
            ચાલુ;
        }
        name.resize(size);
        જ્યારે (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        જો (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    પાછા units;

#અથવાજો defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    પાછા 0;

#નહીંતર
    પાછા 0;
#endif
}

સ્થિર પૂર્ણાંક parse_uint_env(અચલ અક્ષર *s, અચલ અક્ષર *name, તાર્કિક *ok) {
    જો (!s) { *ok = ખોટું; પાછા 0; }
    અક્ષર *end = nullptr;
    long v = strtol(s, &end, 10);
    જો (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = ખોટું;
        પાછા 0;
    }
    જો (v < 0 || v > INT_MAX) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = ખોટું;
        પાછા 0;
    }
    *ok = સાચું;
    પાછા (પૂર્ણાંક)v;
}

સ્થિર શૂન્ય init_kleidiai_context(શૂન્ય) {
    ggml_critical_section_start();
    સ્થિર તાર્કિક initialized = ખોટું;

    જો (!initialized) {
        initialized = સાચું;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        અચલ અક્ષર *env_sme         = getenv("GGML_KLEIDIAI_SME");
        અચલ અક્ષર *env_threads     = getenv("GGML_TOTAL_THREADS");
        અચલ અક્ષર *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        અચલ auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        જો (env_threads) {
            તાર્કિક ok = ખોટું;
            પૂર્ણાંક hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            જો (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        જો (env_chunk_mult) {
            તાર્કિક ok = ખોટું;
            પૂર્ણાંક multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            જો (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        પૂર્ણાંક sme_cores = 0;
        તાર્કિક sme_env_ok = ખોટું;
        તાર્કિક sme_env_set = (env_sme != nullptr);

        અચલ તાર્કિક has_supported_sme_family = runtime_feat.has_sme;
        તાર્કિક sme_cap_detected = ખોટું;

        જો (has_supported_sme_family) {
            detected_smcus = detect_અંક_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (પૂર્ણાંક)detected_smcus : 1;

            જો (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        જો (sme_env_set) {
            તાર્કિક ok = ખોટું;
            પૂર્ણાંક v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            જો (ok) {
                જો (has_supported_sme_family) {
                    sme_cores = v;
                } નહીંતર {
                    જો (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } નહીંતર {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        જો (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            જો (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        જો (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (પૂર્ણાંક)ctx.features);
        } નહીંતર {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        જો (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (પૂર્ણાંક)ctx.features);
        } નહીંતર {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        જો (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (પૂર્ણાંક) ctx.features);
        } નહીંતર {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        જો (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (પૂર્ણાંક)ctx.features);
        } નહીંતર {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        અચલ તાર્કિક has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        જો (has_selected_sme_family_kernel) {
            જો (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } નહીંતર જો (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } નહીંતર {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } નહીંતર {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_end();
}

સ્થિર અંતઃરેખ પૂર્ણાંક kleidiai_sme_thread_cap() {
    પાછા ctx.sme_thread_cap;
}

સ્થિર અંતઃરેખ size_t align_up(size_t value, size_t alignment) {
    જો (alignment == 0) {
        પાછા value;
    }
    અચલ size_t remainder = value % alignment;
    પાછા remainder == 0 ? value : value + (alignment - remainder);
}

સ્થિર અંતઃરેખ size_t gcd_size(size_t a, size_t b) {
    જ્યારે (b != 0) {
        અચલ size_t t = a % b;
        a = b;
        b = t;
    }
    પાછા a;
}

સ્થિર અંતઃરેખ તાર્કિક lcm_size(size_t a, size_t b, size_t & result) {
    જો (a == 0 || b == 0) {
        result = 0;
        પાછા ખોટું;
    }
    અચલ size_t g = gcd_size(a, b);
    અચલ size_t q = a / g;
    જો (q > SIZE_MAX / b) {
        પાછા ખોટું;
    }
    result = q * b;
    પાછા સાચું;
}

સ્થિર અંતઃરેખ size_t ceil_div_size(size_t a, size_t b) {
    પાછા b == 0 ? 0 : (a + b - 1) / b;
}

સ્થિર અંતઃરેખ size_t kleidiai_chunk_cols(size_t n, પૂર્ણાંક nth_total, તાર્કિક disable_chunking, size_t n_step) {
    અચલ size_t multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    અચલ size_t ભાજક = std::max<size_t>(1, (size_t) nth_total * multiplier);
    અચલ size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, ભાજક)), n_step);
    પાછા chunk_cols ? chunk_cols : n_step;
}

સંરચના kleidiai_block_args {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

સ્થિર અંતઃરેખ kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    પસંદગી (rhs_type) {
        સ્થિતિ GGML_TYPE_Q4_0:
            પાછા { QK4_0, QK4_0, QK4_0 };
        સ્થિતિ GGML_TYPE_Q4_K:
            પાછા { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        સ્થિતિ GGML_TYPE_Q8_0:
            પાછા { 0, 0, QK8_0 };
        મૂળભૂત:
            પાછા { 0, 0, 0 };
    }
}

સ્થિર અંતઃરેખ તાર્કિક kleidiai_pack_fallback_allowed() {
    જો (ctx.sme_thread_cap <= 0) {
        પાછા ખોટું;
    }
    જો (ctx.thread_hint <= 0) {
        પાછા સાચું;
    }
    પાછા ctx.thread_hint > ctx.sme_thread_cap;
}

સંરચના kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_ગણતરી;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

સ્થિર અંતઃરેખ kleidiai_weight_header * kleidiai_weight_header_from_ptr(શૂન્ય * data) {
    પાછા reinterpret_cast<kleidiai_weight_header *>(data);
}

સ્થિર અંતઃરેખ અચલ kleidiai_weight_header * kleidiai_weight_header_from_ptr(અચલ શૂન્ય * data) {
    પાછા reinterpret_cast<અચલ kleidiai_weight_header *>(data);
}

સ્થિર અંતઃરેખ તાર્કિક kleidiai_is_weight_header_valid(અચલ kleidiai_weight_header * header) {
    જો (!header) {
        પાછા ખોટું;
    }
    જો (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        પાછા ખોટું;
    }
    જો (header->slot_ગણતરી == 0 || header->slot_ગણતરી > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        પાછા ખોટું;
    }
    પાછા સાચું;
}

સ્થિર અંતઃરેખ uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, પૂર્ણાંક slot) {
    જો (!kleidiai_is_weight_header_valid(header)) {
        પાછા nullptr;
    }
    જો (slot < 0 || slot >= header->slot_ગણતરી) {
        પાછા nullptr;
    }
    પાછા reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

સ્થિર અંતઃરેખ અચલ uint8_t * kleidiai_weight_slot_ptr(અચલ kleidiai_weight_header * header, પૂર્ણાંક slot) {
    જો (!kleidiai_is_weight_header_valid(header)) {
        પાછા nullptr;
    }
    જો (slot < 0 || slot >= header->slot_ગણતરી) {
        પાછા nullptr;
    }
    પાછા reinterpret_cast<અચલ uint8_t *>(header) + header->offsets[slot];
}

સ્થિર અંતઃરેખ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    પાછા ctx.kernels_q4;
}

સ્થિર અંતઃરેખ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    પાછા ctx.kernels_q4k;
}

સ્થિર અંતઃરેખ ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    પાછા ctx.kernels_q8;
}

સ્થિર અંતઃરેખ ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    પાછા ctx.kernels_f32;
}

template <typename SelectFallback>
સ્થિર પૂર્ણાંક kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    પૂર્ણાંક count = 0;
    જો (!primary) {
        પાછા 0;
    }
    out[count++] = primary;

    જો (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        પાછા count;
    }

    જો (is_sme_family(primary->required_cpu)) {
        અચલ cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        જો (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            જો (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    પાછા count;
}

સ્થિર પૂર્ણાંક kleidiai_collect_kernel_chain(અચલ સંરચના ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    પાછા kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { પાછા ggml_kleidiai_select_kernels(mask, op); });
}

સ્થિર પૂર્ણાંક kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    પાછા kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { પાછા ggml_kleidiai_select_kernels_q4_0(mask); });
}

સ્થિર પૂર્ણાંક kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    પાછા kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { પાછા ggml_kleidiai_select_kernels_q4_k(mask); });
}

સ્થિર પૂર્ણાંક kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    પાછા kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { પાછા ggml_kleidiai_select_kernels_q8_0(mask); });
}

સ્થિર પૂર્ણાંક kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    પાછા kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { પાછા ggml_kleidiai_select_kernels_f32(mask); });
}

સ્થિર અંતઃરેખ int64_t ggml_ne(અચલ ggml_tensor * tensor, પૂર્ણાંક dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    પાછા tensor->ne[dim];
}

નામાકાશ ggml::cpu::kleidiai {

સ્થિર size_t round_down(size_t x, size_t y) {
    પાછા y == 0 ? x : x - (x % y);
}

સ્થિર શૂન્ય transpose_f32kxn_f16nxk(size_t n, size_t k, દશાંશ * dst, અચલ uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / કદ(uint16_t);
    size_t dst_stride = n;

    માટે (size_t k_idx = 0; k_idx < k; ++k_idx) {
        માટે (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

વર્ગ tensor_traits : જાહેર ggml::cpu::tensor_traits {
    તાર્કિક work_size(પૂર્ણાંક /* n_threads */, અચલ સંરચના ggml_tensor * op, size_t & size) override {
        જો (op->op != GGML_OP_MUL_MAT) {
            પાછા ખોટું;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        અચલ પૂર્ણાંક slot_ગણતરી = kleidiai_collect_kernel_chain(op, kernel_chain);
        જો (slot_ગણતરી == 0) {
            પાછા ખોટું;
        }

        અચલ તાર્કિક is_gemv = op->src[1]->ne[1] == 1;
        અચલ size_t k = op->src[0]->ne[0];
        અચલ size_t n = op->src[0]->ne[1];
        અચલ size_t m = op->src[1]->ne[1];

        જો (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            અચલ kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            અચલ size_t qk = block_args.pack_bl;

            size_t cursor = 0;
            તાર્કિક any_slot = ખોટું;

            માટે (પૂર્ણાંક slot = 0; slot < slot_ગણતરી; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                જો (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    પાછા ખોટું;
                }

                અચલ size_t mr = kernel->get_mr();
                અચલ size_t kr = kernel->get_kr();
                અચલ size_t sr = kernel->get_sr();

                અચલ size_t packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = સાચું;
            }

            જો (!any_slot) {
                પાછા ખોટું;
            }

            size = cursor;
            પાછા સાચું;
        }

        જો (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            જો (is_gemv && op->src[1]->nb[0] == (int64_t) કદ(દશાંશ) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                પાછા સાચું;
            }

            size_t cursor = 0;
            તાર્કિક any_slot = ખોટું;

            માટે (પૂર્ણાંક slot = 0; slot < slot_ગણતરી; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                જો (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    પાછા ખોટું;
                }

                અચલ size_t mr = kernel->get_mr();
                અચલ size_t kr = kernel->get_kr();
                અચલ size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
                any_slot = સાચું;
            }

            જો (!any_slot) {
                પાછા ખોટું;
            }

            size = cursor;
            પાછા સાચું;
        }

        જો (op->src[0]->type == GGML_TYPE_F16) {
            અચલ int64_t lhs_batch_size0 = op->src[1]->ne[2];
            અચલ int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            અચલ int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            તાર્કિક any_slot = ખોટું;

            માટે (પૂર્ણાંક slot = 0; slot < slot_ગણતરી; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                જો (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    પાછા ખોટું;
                }

                અચલ size_t mr = kernel->get_mr();
                અચલ size_t kr = kernel->get_kr();
                અચલ size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = સાચું;
            }

            માટે (પૂર્ણાંક slot = 0; slot < slot_ગણતરી; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                જો (!kernel || !kernels->rhs_info.packed_size_ex) {
                    પાછા ખોટું;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * કદ(દશાંશ);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * કદ(દશાંશ);

            જો (!any_slot) {
                પાછા ખોટું;
            }

            size = cursor;
            પાછા સાચું;
        }

        પાછા ખોટું;
    }

    તાર્કિક compute_forward(સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst) override {
        જો (dst->op == GGML_OP_MUL_MAT) {
            જો (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                પાછા compute_forward_qx(params, dst);
            } નહીંતર જો (dst->src[0]->type == GGML_TYPE_F32) {
                પાછા compute_forward_f32(params, dst);
            } નહીંતર જો (dst->src[0]->type == GGML_TYPE_F16) {
                પાછા compute_forward_fp16(params, dst);
            }
        }
        પાછા ખોટું;
    }

    તાર્કિક compute_forward_f32(ggml_compute_params * params, સંરચના ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        અચલ ggml_tensor * src0 = dst->src[0];
        અચલ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

        જો (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            પાછા ખોટું;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        જો (!kernels) {
            પાછા ખોટું;
        }

        અચલ size_t k = ne00;
        અચલ size_t m = ne11;
        અચલ size_t n = ne01;
        અચલ તાર્કિક use_gemv = m == 1 && src1->nb[0] == (int64_t) કદ(દશાંશ) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        જો (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            પાછા ખોટું;
        }

        જો (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_size_ex || !lhs_info->pack_func_ex)) {
            પાછા ખોટું;
        }

        અચલ kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        અચલ તાર્કિક has_header = kleidiai_is_weight_header_valid(header);

        અચલ uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<અચલ uint8_t *>(src0->data);
        જો (!rhs_base) {
            પાછા ખોટું;
        }

        અચલ પૂર્ણાંક nth = params->nth > 0 ? params->nth : 1;
        અચલ પૂર્ણાંક ith = params->ith;

        અચલ size_t mr = kernel->get_mr();
        અચલ size_t kr = kernel->get_kr();
        અચલ size_t sr = kernel->get_sr();

        અચલ size_t lhs_packed_size = use_gemv ? 0 : lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        જો (!use_gemv) {
            GGML_ASSERT(lhs_packed_size <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        અચલ size_t dst_stride = dst->nb[1];
        અચલ size_t n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        અચલ તાર્કિક disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) INT_MAX);

        અચલ int64_t batch_size = ne12 * ne13;
        માટે (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            અચલ int64_t i12 = batch_idx % ne12;
            અચલ int64_t i13 = batch_idx / ne12;
            અચલ uint8_t * lhs_batch_base = static_cast<અચલ uint8_t *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];

            જો (!use_gemv) {
                અચલ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                અચલ int64_t use_threads = std::min<int64_t>(nth, max_threads);

                જો (ith < use_threads) {
                    અચલ int64_t અંક_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), mr);
                    અચલ int64_t અંક_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * અંક_m_per_thread0;

                    અચલ int64_t m_start = (int64_t)ith * અંક_m_per_thread0;
                    અચલ int64_t m_ગણતરી = (ith == use_threads - 1) ? અંક_m_per_thread_N_1 : અંક_m_per_thread0;

                    અચલ size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    અચલ size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    અચલ size_t row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_ગણતરી;
                    int64_t cur       = m_start;

                    જ્યારે (remaining > 0) {
                        અચલ int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        અચલ size_t src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        અચલ શૂન્ય * src_ptr = lhs_batch_base + src_off;
                        અચલ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        શૂન્ય * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            જો (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            અચલ size_t chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (size_t) INT_MAX);

            પૂર્ણાંક current_col = ggml_threadpool_chunk_add(params->threadpool, (પૂર્ણાંક) chunk_cols);
            જ્યારે ((size_t) current_col < n) {
                અચલ size_t n_start = (size_t) current_col;
                અચલ size_t n_to_process = std::min(chunk_cols, n - n_start);

                જો (n_to_process > 0) {
                    અચલ size_t lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    અચલ size_t rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    અચલ size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    અચલ શૂન્ય * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    અચલ શૂન્ય * rhs_ptr = rhs_base + rhs_packed_offset;
                    દશાંશ * dst_ptr      = reinterpret_cast<દશાંશ *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          કદ(દશાંશ),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (પૂર્ણાંક) chunk_cols);
            }

            જો (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        પાછા સાચું;
    }

    તાર્કિક compute_forward_fp16(ggml_compute_params * params, સંરચના ggml_tensor * dst) {
        અચલ ggml_tensor * src0 = dst->src[0];
        અચલ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        જો (!kernels) {
            પાછા ખોટું;
        }

        અચલ તાર્કિક is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        જો (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            પાછા ખોટું;
        }

        અચલ પૂર્ણાંક nth = params->nth;
        અચલ પૂર્ણાંક ith = params->ith;

        અચલ int64_t lhs_batch_size0 = ne12;
        અચલ int64_t rhs_batch_size0 = ne02;
        અચલ int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        અચલ int64_t r = lhs_batch_size0 / rhs_batch_size0;

        અચલ int64_t m_group = ne11;
        અચલ int64_t m       = m_group;
        અચલ int64_t n       = ne01;
        અચલ int64_t k       = ne00;

        અચલ size_t lhs_stride = src1->nb[1];
        અચલ size_t rhs_stride = src0->nb[1];
        અચલ size_t dst_stride = dst->nb[1];

        અચલ int64_t mr = (int64_t) kernel->get_mr();
        અચલ int64_t nr = (int64_t) kernel->get_nr();
        અચલ int64_t kr = (int64_t) kernel->get_kr();
        અચલ int64_t sr = (int64_t) kernel->get_sr();

        અચલ size_t lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        અચલ size_t rhs_packed_size = kernels->rhs_info.packed_size_ex(n, k, nr, kr, 0);
        અચલ size_t kxn_size        = k * n * કદ(દશાંશ);
        અચલ size_t bias_size       = n * કદ(દશાંશ);

        અચલ size_t wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        માટે (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            અચલ int64_t rhs_batch_idx = batch_idx / r;
            અચલ uint8_t * rhs_batch_base = static_cast<અચલ uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                અચલ int64_t m_roundup_mr = kai_roundup(m, mr);
                અચલ int64_t અંક_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                જો (ith < અંક_threads) {
                    અચલ int64_t અંક_m_per_thread0   = round_down((size_t)(m_roundup_mr / અંક_threads), (size_t)mr);
                    અચલ int64_t અંક_m_per_thread_N_1 = m - (અંક_threads - 1) * અંક_m_per_thread0;

                    અચલ int64_t m_start = ith * અંક_m_per_thread0;
                    અચલ int64_t m_ગણતરી = (ith == અંક_threads - 1) ? અંક_m_per_thread_N_1 : અંક_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    અચલ size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    અચલ size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    અચલ size_t row_stride_bytes = (next_block_off - base_packed_off) / (size_t)mr;

                    int64_t remaining = m_ગણતરી;
                    int64_t cur       = m_start;

                    જ્યારે (remaining > 0) {
                        અચલ int64_t row_in_group = cur;
                        અચલ int64_t avail        = m_group - row_in_group;
                        અચલ int64_t take         = std::min(avail, remaining);

                        અચલ uint8_t * lhs_batch_base = static_cast<અચલ uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        અચલ શૂન્ય * src_ptr = lhs_batch_base + (size_t)row_in_group * lhs_stride;
                        અચલ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        શૂન્ય * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            જો (ith == 0) {
                memset(bias, 0, (size_t)n * કદ(દશાંશ));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<દશાંશ *>(rhs_kxn),
                                        reinterpret_cast<અચલ uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * કદ(દશાંશ),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                અચલ int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t અંક_threads_n = KAI_MIN(n / n_step, nth);
                જો (અંક_threads_n <= 0) {
                    અંક_threads_n = 1;
                }

                જો (ith < અંક_threads_n) {
                    અચલ int64_t અંક_n_per_thread0   = round_down((size_t)(n / અંક_threads_n), (size_t)n_step);
                    અચલ int64_t અંક_n_per_thread_N_1 = n - (અંક_threads_n - 1) * અંક_n_per_thread0;

                    અચલ int64_t n_start      = ith * અંક_n_per_thread0;
                    અચલ int64_t n_to_process = (ith == અંક_threads_n - 1) ? અંક_n_per_thread_N_1 : અંક_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    અચલ size_t lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    અચલ size_t rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    અચલ size_t dst_offset         = kernel->get_dst_offset((size_t)0, (size_t)n_start, dst_stride);

                    અચલ શૂન્ય * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    અચલ શૂન્ય * rhs_ptr = rhs_packed + rhs_packed_offset;
                    દશાંશ * dst_ptr      = reinterpret_cast<દશાંશ *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, કદ(દશાંશ), -FLT_MAX, FLT_MAX);
                }
            }

            જો (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        પાછા સાચું;
    }

    તાર્કિક compute_forward_qx(સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        અચલ ggml_tensor * src0 = dst->src[0];
        અચલ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

        અચલ kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        અચલ તાર્કિક has_header = kleidiai_is_weight_header_valid(header);
        અચલ તાર્કિક is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        અચલ પૂર્ણાંક slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        auto weight_for_slot = [&](પૂર્ણાંક slot_index, size_t & size_out) -> અચલ uint8_t * {
            જો (slot_index < 0 || slot_index >= slot_total) {
                પાછા nullptr;
            }
            જો (has_header) {
                જો (slot_index < header->slot_ગણતરી) {
                    size_out = static_cast<size_t>(header->sizes[slot_index]);
                    પાછા kleidiai_weight_slot_ptr(header, slot_index);
                }
                પાછા nullptr;
            }
            જો (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                પાછા static_cast<અચલ uint8_t *>(src0->data);
            }
            પાછા nullptr;
        };

        સંરચના runtime_slot {
            પૂર્ણાંક slot_index;
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
            પૂર્ણાંક assigned_threads;
            પૂર્ણાંક thread_begin;
            પૂર્ણાંક thread_end;
            અચલ uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        પૂર્ણાંક runtime_ગણતરી = 0;

        માટે (પૂર્ણાંક slot = 0; slot < slot_total && runtime_ગણતરી < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            જો (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                ચાલુ;
            }

            size_t rhs_size = 0;
            અચલ uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            જો (!rhs_ptr || rhs_size == 0) {
                ચાલુ;
            }

            અચલ kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

            runtime[runtime_ગણતરી] = {
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
            ++runtime_ગણતરી;
        }

        જો (runtime_ગણતરી == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            પાછા ખોટું;
        }

        અચલ પૂર્ણાંક nth_total = params->nth > 0 ? params->nth : 1;
        અચલ પૂર્ણાંક ith_total = params->ith;

        પૂર્ણાંક sme_slot = -1;
        પૂર્ણાંક non_sme_slot = -1;
        માટે (પૂર્ણાંક i = 0; i < runtime_ગણતરી; ++i) {
            જો (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                તોડો;
            }
        }

        માટે (પૂર્ણાંક i = 0; i < runtime_ગણતરી; ++i) {
            જો (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                તોડો;
            }
        }

        અચલ પૂર્ણાંક sme_cap_limit = ctx.sme_thread_cap;
        અચલ તાર્કિક use_hybrid = sme_cap_limit > 0 &&
                                 runtime_ગણતરી > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_cols_per_thread = 0;
        જો (runtime_ગણતરી > 0 && nth_total > 0) {
            min_cols_per_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        અચલ તાર્કિક too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        અચલ તાર્કિક hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        જો (!hybrid_enabled) {
            પૂર્ણાંક chosen_slot = 0;
            જો (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } નહીંતર જો (runtime_ગણતરી > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            જો (chosen_slot != 0 && chosen_slot < runtime_ગણતરી) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_ગણતરી = runtime_ગણતરી > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            જો (runtime_ગણતરી > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        અચલ તાર્કિક sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        પૂર્ણાંક sme_cap = sme_single_only ? nth_total : kleidiai_sme_thread_cap();
        જો (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        પૂર્ણાંક threads_remaining = nth_total;
        જો (sme_slot != -1) {
            પૂર્ણાંક sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        પૂર્ણાંક fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        પૂર્ણાંક fallback_ગણતરી = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        માટે (પૂર્ણાંક i = 0; i < runtime_ગણતરી; ++i) {
            જો (i == sme_slot) {
                ચાલુ;
            }
            fallback_indices[fallback_ગણતરી++] = i;
        }

        માટે (પૂર્ણાંક fi = 0; fi < fallback_ગણતરી; ++fi) {
            જો (threads_remaining <= 0) {
                તોડો;
            }
            અચલ પૂર્ણાંક slot_index = fallback_indices[fi];
            અચલ પૂર્ણાંક slots_ડાબું = fallback_ગણતરી - fi;
            પૂર્ણાંક share = (threads_remaining + slots_ડાબું - 1) / slots_ડાબું;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        પૂર્ણાંક thread_cursor = 0;
        માટે (પૂર્ણાંક i = 0; i < runtime_ગણતરી; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        અચલ પૂર્ણાંક active_threads = thread_cursor;

        પૂર્ણાંક local_slot = -1;
        પૂર્ણાંક local_ith  = 0;
        માટે (પૂર્ણાંક i = 0; i < runtime_ગણતરી; ++i) {
            જો (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                તોડો;
            }
        }
        અચલ size_t k = ne00;
        અચલ size_t m = ne11;
        અચલ size_t n = ne01;

        size_t cursor = 0;
        માટે (પૂર્ણાંક i = 0; i < runtime_ગણતરી; ++i) {
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_step = 1;
        માટે (પૂર્ણાંક i = 0; i < runtime_ગણતરી; ++i) {
            જો (runtime[i].assigned_threads == 0) {
                ચાલુ;
            }
            size_t next_step = 0;
            જો (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                પાછા ખોટું;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        અચલ તાર્કિક disable_chunking = ggml_is_numa();
        અચલ size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        અચલ size_t chunk_ભાજક = (active_threads == 1 || disable_chunking) ? (size_t) active_threads : (size_t) active_threads * chunk_multiplier;
        size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, chunk_ભાજક)), common_step);
        જો (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        અચલ size_t nchunk_size = std::max<size_t>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (size_t)INT_MAX);
        અચલ પૂર્ણાંક nchunk = (પૂર્ણાંક)nchunk_size;
        અચલ size_t dst_stride = dst->nb[1];

        auto run_chunk = [&](runtime_slot & slot, size_t global_start, size_t cols, uint8_t * dst_batch_base) {
            અચલ size_t rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            અચલ size_t dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            અચલ uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            અચલ uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            દશાંશ * dst_ptr         = reinterpret_cast<દશાંશ *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       કદ(દશાંશ),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        માટે (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            અચલ uint8_t * lhs_batch_base = static_cast<અચલ uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            જો (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                અચલ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                અચલ int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                જો (local_ith < use_threads) {
                    અચલ int64_t અંક_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), slot.mr);
                    અચલ int64_t અંક_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * અંક_m_per_thread0;

                    અચલ int64_t m_start = (int64_t)local_ith * અંક_m_per_thread0;
                    અચલ int64_t m_ગણતરી = (local_ith == use_threads - 1) ? અંક_m_per_thread_N_1 : અંક_m_per_thread0;

                    અચલ size_t base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    અચલ size_t next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    અચલ size_t row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_ગણતરી;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    જ્યારે (remaining > 0) {
                        અચલ int64_t row_in_group = cur;
                        અચલ int64_t avail        = (int64_t)m - row_in_group;
                        અચલ int64_t take         = std::min(avail, remaining);

                        અચલ size_t src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        અચલ શૂન્ય * src_ptr = lhs_batch_base + src_off;
                        અચલ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        શૂન્ય * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            જો (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            જો (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                પૂર્ણાંક current_chunk = ith_total;
                જ્યારે (current_chunk < nchunk) {
                    અચલ size_t global_start = (size_t)current_chunk * chunk_cols;
                    જો (global_start >= n) {
                        તોડો;
                    }

                    અચલ size_t cols = std::min(chunk_cols, n - global_start);
                    જો (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            જો (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        પાછા સાચું;
    }

જાહેર:
    પૂર્ણાંક repack(સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        અચલ size_t n = tensor->ne[1];
        અચલ size_t k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        જો (!header) {
            પાછા -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_ગણતરી = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        size_t cursor = કદ(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        અચલ તાર્કિક want_q4k = tensor->type == GGML_TYPE_Q4_K;
        અચલ તાર્કિક want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        અચલ તાર્કિક want_f32 = tensor->type == GGML_TYPE_F32;
        અચલ પૂર્ણાંક slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        અચલ તાર્કિક allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<દશાંશ>  scales;
        std::vector<દશાંશ>  bias;
        std::vector<દશાંશ> row_f32;

        જો (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            અચલ size_t row_stride = tensor->nb[1];
            અચલ size_t k_blocks   = (k + QK8_0 - 1) / QK8_0;

            જો (tensor->type == GGML_TYPE_Q6_K) {
                જો (k % QK_K != 0 || k % QK8_0 != 0) {
                    પાછા -1;
                }
                row_f32.resize(k);
            }

            માટે (size_t row = 0; row < n; ++row) {
                અચલ uint8_t * row_data = static_cast<અચલ uint8_t *>(data) + row * row_stride;

                જો (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<અચલ block_q6_K *>(row_data), row_f32.data(), k);
                }

                દશાંશ max_abs = 0.0f;
                જો (tensor->type == GGML_TYPE_Q6_K) {
                    માટે (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::fabs(row_f32[linear_idx]));
                    }
                } નહીંતર {
                    અચલ block_q8_0 * row_blocks = reinterpret_cast<અચલ block_q8_0 *>(row_data);
                    માટે (size_t block = 0; block < k_blocks; ++block) {
                        અચલ block_q8_0 & blk = row_blocks[block];
                        અચલ દશાંશ d = GGML_FP16_TO_FP32(blk.d);
                        માટે (size_t l = 0; l < QK8_0; ++l) {
                            અચલ size_t linear_idx = block * QK8_0 + l;
                            જો (linear_idx >= k) {
                                તોડો;
                            }
                            અચલ દશાંશ value = d * static_cast<દશાંશ>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                દશાંશ scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                અચલ દશાંશ inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                જો (tensor->type == GGML_TYPE_Q6_K) {
                    માટે (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } નહીંતર {
                    અચલ block_q8_0 * row_blocks = reinterpret_cast<અચલ block_q8_0 *>(row_data);
                    માટે (size_t block = 0; block < k_blocks; ++block) {
                        અચલ block_q8_0 & blk = row_blocks[block];
                        અચલ દશાંશ d = GGML_FP16_TO_FP32(blk.d);
                        માટે (size_t l = 0; l < QK8_0; ++l) {
                            અચલ size_t linear_idx = block * QK8_0 + l;
                            જો (linear_idx >= k) {
                                તોડો;
                            }
                            અચલ દશાંશ value = d * static_cast<દશાંશ>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        જો (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
        }

        માટે (પૂર્ણાંક slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            જો (!allow_fallback && slot > 0) {
                તોડો;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            જો (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                ચાલુ;
            }

            અચલ size_t nr = kernel->get_nr();
            અચલ size_t kr = kernel->get_kr();
            અચલ size_t sr = kernel->get_sr();
            અચલ ggml_type rhs_type = kernels->rhs_type;
            જો (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            અચલ size_t block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            જો (block_len == SIZE_MAX) {
                ચાલુ;
            }

            અચલ size_t packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            અચલ size_t aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            જો (rhs_type == GGML_TYPE_Q4_0) {
                સંરચના kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<અચલ uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } નહીંતર જો (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       data, nullptr, nullptr, dst_ptr, 0, nullptr);
            } નહીંતર જો (rhs_type == GGML_TYPE_Q8_0) {
                સંરચના kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } નહીંતર જો (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
            } નહીંતર {
                ચાલુ;
            }

            header->offsets[header->slot_ગણતરી] = aligned_cursor;
            header->sizes[header->slot_ગણતરી]   = packed_size;
            ++header->slot_ગણતરી;

            cursor = aligned_cursor + packed_size;
        }

        જો (header->slot_ગણતરી == 0) {
            header->magic   = 0;
            header->version = 0;
            memcpy(tensor->data, data, data_size);
        }

        પાછા 0;
    }
};

સ્થિર ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, સંરચના ggml_tensor *) {
    સ્થિર tensor_traits traits;
    પાછા &traits;
}
}  // namespace ggml::cpu::kleidiai

સ્થિર ગણના ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, સંરચના ggml_tensor * tensor) {
    tensor->extra = (શૂન્ય *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    પાછા GGML_STATUS_SUCCESS;
}

સ્થિર શૂન્ય ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, સંરચના ggml_tensor * tensor,
                                                       અચલ શૂન્ય * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

સ્થિર અચલ અક્ષર * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    પાછા "CPU_KLEIDIAI";
}

સ્થિર ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    જો (buffer == nullptr) {
        પાછા nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    પાછા buffer;
}

સ્થિર size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    પાછા TENSOR_ALIGNMENT;
}

સ્થિર size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ સંરચના ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    જો (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        પાછા ggml_nbytes(tensor);
    }

    અચલ size_t n = tensor->ne[1];
    અચલ size_t k = tensor->ne[0];

    size_t cursor = કદ(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    અચલ તાર્કિક want_q4k = tensor->type == GGML_TYPE_Q4_K;
    અચલ તાર્કિક want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    અચલ તાર્કિક want_f32 = tensor->type == GGML_TYPE_F32;
    અચલ પૂર્ણાંક slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    અચલ તાર્કિક allow_fallback = kleidiai_pack_fallback_allowed();

    size_t slot_ગણતરી = 0;
    માટે (પૂર્ણાંક slot = 0; slot < slot_total; ++slot) {
        જો (!allow_fallback && slot > 0) {
            તોડો;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        જો (!kernels) {
            ચાલુ;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        જો (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            ચાલુ;
        }

        અચલ ggml_type rhs_type = kernels->rhs_type;
        અચલ size_t block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        જો (block_len == SIZE_MAX) {
            ચાલુ;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_ગણતરી;
    }

    જો (slot_ગણતરી == 0) {
        પાછા ggml_nbytes(tensor);
    }

    પાછા std::max(cursor, ggml_nbytes(tensor));
}

નામાકાશ ggml::cpu::kleidiai {
વર્ગ extra_buffer_type : ggml::cpu::extra_buffer_type {
    તાર્કિક supports_op(ggml_backend_dev_t, અચલ સંરચના ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        જો (op->op != GGML_OP_MUL_MAT) {
            પાછા ખોટું;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        અચલ પૂર્ણાંક slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        અચલ તાર્કિક src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft->context == આ &&
            slot_total > 0;

        જો ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            જો (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                પાછા ખોટું;
            }
            જો (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                પાછા ખોટું;
            }
            જો (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                પાછા ખોટું;
            }
            જો (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                પાછા ખોટું;
            }
            જો (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                પાછા ખોટું;
            }
            જો (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                પાછા ખોટું;
            }

            જો (op->src[0]->type == GGML_TYPE_Q4_K) {
                જો (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    પાછા ખોટું;
                }
                પાછા ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == કદ(દશાંશ) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * કદ(દશાંશ));
            }

            જો (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                જો (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    પાછા સાચું;
                }
                પાછા ખોટું;
            }

            જો (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                પાછા ખોટું;
            }

            પાછા સાચું;
        }

        પાછા ખોટું;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(અચલ સંરચના ggml_tensor * op) override {
        જો (op->op == GGML_OP_MUL_MAT) {
            જો (op->src[0]->buffer && op->src[0]->buffer->buft->context == આ) {
                પાછા (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } નહીંતર {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                જો (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    સ્થિર std::atomic<તાર્કિક> warned(ખોટું);
                    જો (!warned.exchange(સાચું)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                જો (op->src[0]->type != GGML_TYPE_F16) {
                    પાછા nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                અચલ પૂર્ણાંક slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                જો (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    જો ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        પાછા nullptr;
                    }
                    પાછા ggml::cpu::kleidiai::get_tensor_traits(ખાલી, ખાલી);
                }
            }
        }
        પાછા nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(શૂન્ય) {
    સ્થિર ggml::cpu::kleidiai::extra_buffer_type ctx;
    સ્થિર સંરચના ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .get_alloc_size_n     = */ ખાલી,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    પાછા &ggml_backend_cpu_buffer_type_kleidiai;
}
