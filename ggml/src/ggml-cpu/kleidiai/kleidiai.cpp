// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#அந்தர்பாவ <arm_neon.h>
#அந்தர்பாவ <cassert>
#அந்தர்பாவ <cstdio>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <atomic>
#அந்தர்பாவ <cfloat>
#அந்தர்பாவ <cctype>
#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cmath>
#அந்தர்பாவ <stdexcept>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <string.h>
#அந்தர்பாவ <string>
#அந்தர்பாவ <vector>
#அந்தர்பாவ <array>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <fstream>
#அந்தர்பாவ <map>
#அந்தர்பாவ <iostream>
#அந்தர்பாவ <climits>
#அந்தர்பாவ <charconv>
#அந்தர்பாவ <system_error>
#எனில் defined(__linux__)
#அந்தர்பாவ <asm/hwcap.h>
#அந்தர்பாவ <dirent.h>
#அந்தர்பாவ <sys/auxv.h>
#அந்தர்பாவ <sys/types.h>
#அந்தர்பாவ <sys/stat.h>
#அந்தர்பாவ <unistd.h>
#அல்லதெனில் defined(__APPLE__)
#அந்தர்பாவ <sys/sysctl.h>
#அந்தர்பாவ <sys/types.h>
#endif

#அந்தர்பாவ "kleidiai.h"

#அந்தர்பாவ "ggml-cpu.h"
#அந்தர்பாவ "ggml-cpu-impl.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-feats.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-quants.h"
#அந்தர்பாவ "ggml-threading.h"
#அந்தர்பாவ "traits.h"

#அந்தர்பாவ "kernels.h"

#அந்தர்பாவ "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#அந்தர்பாவ "ggml-common.h"

நிலையான constexpr முழு      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
நிலையான constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
நிலையான constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
நிலையான constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
நிலையான constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

கட்டமைப்பு ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    முழு sme_thread_cap; // <= 0 means "SME disabled/unknown"
    முழு thread_hint;    // <= 0 means "no hint"
    முழு chunk_multiplier;
} நிலையான ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

நிலையான உள்வரி தருக்க is_sme_family(cpu_feature f) {
    திரும்பு (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

நிலையான மாறிலி எழுத்து* cpu_feature_to_string(cpu_feature f) {
    எனில் (f == CPU_FEATURE_NONE) {
        திரும்பு "NONE";
    } இல்லையெனில் எனில் ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        திரும்பு "SME2";
    } இல்லையெனில் எனில் ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        திரும்பு "SME";
    } இல்லையெனில் எனில் ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        திரும்பு "SVE";
    }
    இல்லையெனில் எனில் ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        திரும்பு "I8MM";
    } இல்லையெனில் எனில் ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        திரும்பு "DOTPROD";
    }
    இல்லையெனில் {
        திரும்பு "UNKNOWN";
    }
}

#எனில் defined(__linux__) && defined(__aarch64__)
நிலையான தருக்க parse_cpu_dir_name(மாறிலி எழுத்து* name, size_t* cpu) {
    எனில் (strncmp(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து* first = name + 3;
    மாறிலி எழுத்து* last = name + strlen(name);

    size_t value = 0;
    மாறிலி auto [end, ec] = std::from_chars(first, last, value, 10);

    எனில் (ec != std::errc{} || end != last) {
        திரும்பு பொய்;
    }

    *cpu = value;
    திரும்பு உண்மை;
}

நிலையான std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    எனில் (dir == nullptr) {
        திரும்பு cpus;
    }

    போது (dirent * entry = readdir(dir)) {
        size_t cpu = 0;
        எனில் (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    திரும்பு cpus;
}
#endif

#எனில் defined(__APPLE__) && defined(__aarch64__)
நிலையான தருக்க apple_sme_counted_perf_level(std::string name) {
    க்கு (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (எழுத்து) std::tolower((unsigned எழுத்து) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    திரும்பு name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

நிலையான வெறுமை add_smcus_from_smidr(uint64_t smidr, size_t & இலக்கம்_private, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    மாறிலி uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    மாறிலி uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    மாறிலி size_t shared_மொத்தம் = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    மாறிலி uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    மாறிலி uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    மாறிலி uint32_t id = (affinity2 << 12) | affinity;

    எனில் (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    தேர்வு (sh) {
        நிலை 2: // private SMCU
            ++இலக்கம்_private;
            நிறுத்து;
        நிலை 3: // shared SMCU
            எனில் (shared_counts[id] < shared_மொத்தம்) {
                shared_counts[id] = shared_மொத்தம்;
            }
            நிறுத்து;
        நிலை 0:
            எனில் (id == 0) {
                ++இலக்கம்_private;
            } இல்லையெனில் எனில் (shared_counts[id] < shared_மொத்தம்) {
                shared_counts[id] = shared_மொத்தம்;
            }
            நிறுத்து;
        இயல்பு:
            நிறுத்து;
    }
}

நிலையான size_t detect_இலக்கம்_smcus() {
    மாறிலி auto runtime_feat = ggml_feats_get_arch64_runtime();
    எனில் (!runtime_feat.has_sme) {
        திரும்பு 0;
    }

#எனில் defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t இலக்கம்_private = 0;
    std::map<uint32_t, size_t> shared_counts;

    மாறிலி std::vector<size_t> cpus = detect_cpu_ids();
    க்கு (மாறிலி size_t cpu : cpus) {
        மாறிலி std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        எனில் (!file.is_open()) {
            தொடர்;
        }

        uint64_t smidr = 0;
        எனில் (!(file >> std::hex >> smidr)) {
            தொடர்;
        }

        add_smcus_from_smidr(smidr, இலக்கம்_private, shared_counts);
    }

    size_t total = இலக்கம்_private;
    க்கு (மாறிலி auto & entry : shared_counts) {
        total += entry.second;
    }
    திரும்பு total;

#அல்லதெனில் defined(__APPLE__) && defined(__aarch64__)
    முழு perf_levels = 0;
    size_t size = அளவு(perf_levels);
    எனில் (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != அளவு(perf_levels) || perf_levels <= 0) {
        திரும்பு 0;
    }

    size_t units = 0;
    க்கு (முழு i = 0; i < perf_levels; ++i) {
        எழுத்து சாவி[64] = {};
        முழு physical_cpus = 0;
        முழு cpus_per_l2 = 0;

        snprintf(சாவி, அளவு(சாவி), "hw.perflevel%d.physicalcpu", i);
        size = அளவு(physical_cpus);
        எனில் (sysctlbyname(சாவி, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != அளவு(physical_cpus) || physical_cpus <= 0) {
            தொடர்;
        }

        snprintf(சாவி, அளவு(சாவி), "hw.perflevel%d.cpusperl2", i);
        size = அளவு(cpus_per_l2);
        எனில் (sysctlbyname(சாவி, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != அளவு(cpus_per_l2) || cpus_per_l2 <= 0) {
            தொடர்;
        }

        snprintf(சாவி, அளவு(சாவி), "hw.perflevel%d.name", i);
        size = 0;
        எனில் (sysctlbyname(சாவி, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            தொடர்;
        }

        std::string name(size, '\0');
        எனில் (sysctlbyname(சாவி, &name[0], &size, nullptr, 0) != 0) {
            தொடர்;
        }
        name.resize(size);
        போது (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        எனில் (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    திரும்பு units;

#அல்லதெனில் defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    திரும்பு 0;

#இல்லையெனில்
    திரும்பு 0;
#endif
}

நிலையான முழு parse_uint_env(மாறிலி எழுத்து *s, மாறிலி எழுத்து *name, தருக்க *ok) {
    எனில் (!s) { *ok = பொய்; திரும்பு 0; }
    எழுத்து *end = nullptr;
    long v = strtol(s, &end, 10);
    எனில் (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = பொய்;
        திரும்பு 0;
    }
    எனில் (v < 0 || v > INT_MAX) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = பொய்;
        திரும்பு 0;
    }
    *ok = உண்மை;
    திரும்பு (முழு)v;
}

நிலையான வெறுமை init_kleidiai_context(வெறுமை) {
    ggml_critical_section_start();
    நிலையான தருக்க initialized = பொய்;

    எனில் (!initialized) {
        initialized = உண்மை;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        மாறிலி எழுத்து *env_sme         = getenv("GGML_KLEIDIAI_SME");
        மாறிலி எழுத்து *env_threads     = getenv("GGML_TOTAL_THREADS");
        மாறிலி எழுத்து *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        மாறிலி auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        எனில் (env_threads) {
            தருக்க ok = பொய்;
            முழு hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            எனில் (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        எனில் (env_chunk_mult) {
            தருக்க ok = பொய்;
            முழு multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            எனில் (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        முழு sme_cores = 0;
        தருக்க sme_env_ok = பொய்;
        தருக்க sme_env_set = (env_sme != nullptr);

        மாறிலி தருக்க has_supported_sme_family = runtime_feat.has_sme;
        தருக்க sme_cap_detected = பொய்;

        எனில் (has_supported_sme_family) {
            detected_smcus = detect_இலக்கம்_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (முழு)detected_smcus : 1;

            எனில் (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        எனில் (sme_env_set) {
            தருக்க ok = பொய்;
            முழு v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            எனில் (ok) {
                எனில் (has_supported_sme_family) {
                    sme_cores = v;
                } இல்லையெனில் {
                    எனில் (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } இல்லையெனில் {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        எனில் (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            எனில் (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        எனில் (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (முழு)ctx.features);
        } இல்லையெனில் {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        எனில் (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (முழு)ctx.features);
        } இல்லையெனில் {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        எனில் (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (முழு) ctx.features);
        } இல்லையெனில் {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        எனில் (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (முழு)ctx.features);
        } இல்லையெனில் {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        மாறிலி தருக்க has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        எனில் (has_selected_sme_family_kernel) {
            எனில் (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } இல்லையெனில் எனில் (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } இல்லையெனில் {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } இல்லையெனில் {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_end();
}

நிலையான உள்வரி முழு kleidiai_sme_thread_cap() {
    திரும்பு ctx.sme_thread_cap;
}

நிலையான உள்வரி size_t align_up(size_t value, size_t alignment) {
    எனில் (alignment == 0) {
        திரும்பு value;
    }
    மாறிலி size_t remainder = value % alignment;
    திரும்பு remainder == 0 ? value : value + (alignment - remainder);
}

நிலையான உள்வரி size_t gcd_size(size_t a, size_t b) {
    போது (b != 0) {
        மாறிலி size_t t = a % b;
        a = b;
        b = t;
    }
    திரும்பு a;
}

நிலையான உள்வரி தருக்க lcm_size(size_t a, size_t b, size_t & result) {
    எனில் (a == 0 || b == 0) {
        result = 0;
        திரும்பு பொய்;
    }
    மாறிலி size_t g = gcd_size(a, b);
    மாறிலி size_t q = a / g;
    எனில் (q > SIZE_MAX / b) {
        திரும்பு பொய்;
    }
    result = q * b;
    திரும்பு உண்மை;
}

நிலையான உள்வரி size_t ceil_div_size(size_t a, size_t b) {
    திரும்பு b == 0 ? 0 : (a + b - 1) / b;
}

நிலையான உள்வரி size_t kleidiai_chunk_cols(size_t n, முழு nth_total, தருக்க disable_chunking, size_t n_step) {
    மாறிலி size_t multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    மாறிலி size_t வகுஎண் = std::max<size_t>(1, (size_t) nth_total * multiplier);
    மாறிலி size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, வகுஎண்)), n_step);
    திரும்பு chunk_cols ? chunk_cols : n_step;
}

கட்டமைப்பு kleidiai_block_args {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

நிலையான உள்வரி kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    தேர்வு (rhs_type) {
        நிலை GGML_TYPE_Q4_0:
            திரும்பு { QK4_0, QK4_0, QK4_0 };
        நிலை GGML_TYPE_Q4_K:
            திரும்பு { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        நிலை GGML_TYPE_Q8_0:
            திரும்பு { 0, 0, QK8_0 };
        இயல்பு:
            திரும்பு { 0, 0, 0 };
    }
}

நிலையான உள்வரி தருக்க kleidiai_pack_fallback_allowed() {
    எனில் (ctx.sme_thread_cap <= 0) {
        திரும்பு பொய்;
    }
    எனில் (ctx.thread_hint <= 0) {
        திரும்பு உண்மை;
    }
    திரும்பு ctx.thread_hint > ctx.sme_thread_cap;
}

கட்டமைப்பு kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_மொத்தம்;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

நிலையான உள்வரி kleidiai_weight_header * kleidiai_weight_header_from_ptr(வெறுமை * data) {
    திரும்பு reinterpret_cast<kleidiai_weight_header *>(data);
}

நிலையான உள்வரி மாறிலி kleidiai_weight_header * kleidiai_weight_header_from_ptr(மாறிலி வெறுமை * data) {
    திரும்பு reinterpret_cast<மாறிலி kleidiai_weight_header *>(data);
}

நிலையான உள்வரி தருக்க kleidiai_is_weight_header_valid(மாறிலி kleidiai_weight_header * header) {
    எனில் (!header) {
        திரும்பு பொய்;
    }
    எனில் (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        திரும்பு பொய்;
    }
    எனில் (header->slot_மொத்தம் == 0 || header->slot_மொத்தம் > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        திரும்பு பொய்;
    }
    திரும்பு உண்மை;
}

நிலையான உள்வரி uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, முழு slot) {
    எனில் (!kleidiai_is_weight_header_valid(header)) {
        திரும்பு nullptr;
    }
    எனில் (slot < 0 || slot >= header->slot_மொத்தம்) {
        திரும்பு nullptr;
    }
    திரும்பு reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

நிலையான உள்வரி மாறிலி uint8_t * kleidiai_weight_slot_ptr(மாறிலி kleidiai_weight_header * header, முழு slot) {
    எனில் (!kleidiai_is_weight_header_valid(header)) {
        திரும்பு nullptr;
    }
    எனில் (slot < 0 || slot >= header->slot_மொத்தம்) {
        திரும்பு nullptr;
    }
    திரும்பு reinterpret_cast<மாறிலி uint8_t *>(header) + header->offsets[slot];
}

நிலையான உள்வரி ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    திரும்பு ctx.kernels_q4;
}

நிலையான உள்வரி ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    திரும்பு ctx.kernels_q4k;
}

நிலையான உள்வரி ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    திரும்பு ctx.kernels_q8;
}

நிலையான உள்வரி ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    திரும்பு ctx.kernels_f32;
}

template <typename SelectFallback>
நிலையான முழு kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    முழு count = 0;
    எனில் (!primary) {
        திரும்பு 0;
    }
    out[count++] = primary;

    எனில் (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        திரும்பு count;
    }

    எனில் (is_sme_family(primary->required_cpu)) {
        மாறிலி cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        எனில் (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            எனில் (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    திரும்பு count;
}

நிலையான முழு kleidiai_collect_kernel_chain(மாறிலி கட்டமைப்பு ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    திரும்பு kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { திரும்பு ggml_kleidiai_select_kernels(mask, op); });
}

நிலையான முழு kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    திரும்பு kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { திரும்பு ggml_kleidiai_select_kernels_q4_0(mask); });
}

நிலையான முழு kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    திரும்பு kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { திரும்பு ggml_kleidiai_select_kernels_q4_k(mask); });
}

நிலையான முழு kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    திரும்பு kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { திரும்பு ggml_kleidiai_select_kernels_q8_0(mask); });
}

நிலையான முழு kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    திரும்பு kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { திரும்பு ggml_kleidiai_select_kernels_f32(mask); });
}

நிலையான உள்வரி int64_t ggml_ne(மாறிலி ggml_tensor * tensor, முழு dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    திரும்பு tensor->ne[dim];
}

நாமாகாஶ ggml::cpu::kleidiai {

நிலையான size_t round_down(size_t x, size_t y) {
    திரும்பு y == 0 ? x : x - (x % y);
}

நிலையான வெறுமை transpose_f32kxn_f16nxk(size_t n, size_t k, தசம * dst, மாறிலி uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / அளவு(uint16_t);
    size_t dst_stride = n;

    க்கு (size_t k_idx = 0; k_idx < k; ++k_idx) {
        க்கு (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

வகுப்பு tensor_traits : பொது ggml::cpu::tensor_traits {
    தருக்க work_size(முழு /* n_threads */, மாறிலி கட்டமைப்பு ggml_tensor * op, size_t & size) override {
        எனில் (op->op != GGML_OP_MUL_MAT) {
            திரும்பு பொய்;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        மாறிலி முழு slot_மொத்தம் = kleidiai_collect_kernel_chain(op, kernel_chain);
        எனில் (slot_மொத்தம் == 0) {
            திரும்பு பொய்;
        }

        மாறிலி தருக்க is_gemv = op->src[1]->ne[1] == 1;
        மாறிலி size_t k = op->src[0]->ne[0];
        மாறிலி size_t n = op->src[0]->ne[1];
        மாறிலி size_t m = op->src[1]->ne[1];

        எனில் (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            மாறிலி kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            மாறிலி size_t qk = block_args.pack_bl;

            size_t cursor = 0;
            தருக்க any_slot = பொய்;

            க்கு (முழு slot = 0; slot < slot_மொத்தம்; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                எனில் (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    திரும்பு பொய்;
                }

                மாறிலி size_t mr = kernel->get_mr();
                மாறிலி size_t kr = kernel->get_kr();
                மாறிலி size_t sr = kernel->get_sr();

                மாறிலி size_t packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = உண்மை;
            }

            எனில் (!any_slot) {
                திரும்பு பொய்;
            }

            size = cursor;
            திரும்பு உண்மை;
        }

        எனில் (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            எனில் (is_gemv && op->src[1]->nb[0] == (int64_t) அளவு(தசம) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                திரும்பு உண்மை;
            }

            size_t cursor = 0;
            தருக்க any_slot = பொய்;

            க்கு (முழு slot = 0; slot < slot_மொத்தம்; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                எனில் (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    திரும்பு பொய்;
                }

                மாறிலி size_t mr = kernel->get_mr();
                மாறிலி size_t kr = kernel->get_kr();
                மாறிலி size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
                any_slot = உண்மை;
            }

            எனில் (!any_slot) {
                திரும்பு பொய்;
            }

            size = cursor;
            திரும்பு உண்மை;
        }

        எனில் (op->src[0]->type == GGML_TYPE_F16) {
            மாறிலி int64_t lhs_batch_size0 = op->src[1]->ne[2];
            மாறிலி int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            மாறிலி int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            தருக்க any_slot = பொய்;

            க்கு (முழு slot = 0; slot < slot_மொத்தம்; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                எனில் (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    திரும்பு பொய்;
                }

                மாறிலி size_t mr = kernel->get_mr();
                மாறிலி size_t kr = kernel->get_kr();
                மாறிலி size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = உண்மை;
            }

            க்கு (முழு slot = 0; slot < slot_மொத்தம்; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                எனில் (!kernel || !kernels->rhs_info.packed_size_ex) {
                    திரும்பு பொய்;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * அளவு(தசம);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * அளவு(தசம);

            எனில் (!any_slot) {
                திரும்பு பொய்;
            }

            size = cursor;
            திரும்பு உண்மை;
        }

        திரும்பு பொய்;
    }

    தருக்க compute_forward(கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst) override {
        எனில் (dst->op == GGML_OP_MUL_MAT) {
            எனில் (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                திரும்பு compute_forward_qx(params, dst);
            } இல்லையெனில் எனில் (dst->src[0]->type == GGML_TYPE_F32) {
                திரும்பு compute_forward_f32(params, dst);
            } இல்லையெனில் எனில் (dst->src[0]->type == GGML_TYPE_F16) {
                திரும்பு compute_forward_fp16(params, dst);
            }
        }
        திரும்பு பொய்;
    }

    தருக்க compute_forward_f32(ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        மாறிலி ggml_tensor * src0 = dst->src[0];
        மாறிலி ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_இருமம்_OP_LOCALS

        எனில் (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            திரும்பு பொய்;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        எனில் (!kernels) {
            திரும்பு பொய்;
        }

        மாறிலி size_t k = ne00;
        மாறிலி size_t m = ne11;
        மாறிலி size_t n = ne01;
        மாறிலி தருக்க use_gemv = m == 1 && src1->nb[0] == (int64_t) அளவு(தசம) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        எனில் (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            திரும்பு பொய்;
        }

        எனில் (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_size_ex || !lhs_info->pack_func_ex)) {
            திரும்பு பொய்;
        }

        மாறிலி kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        மாறிலி தருக்க has_header = kleidiai_is_weight_header_valid(header);

        மாறிலி uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<மாறிலி uint8_t *>(src0->data);
        எனில் (!rhs_base) {
            திரும்பு பொய்;
        }

        மாறிலி முழு nth = params->nth > 0 ? params->nth : 1;
        மாறிலி முழு ith = params->ith;

        மாறிலி size_t mr = kernel->get_mr();
        மாறிலி size_t kr = kernel->get_kr();
        மாறிலி size_t sr = kernel->get_sr();

        மாறிலி size_t lhs_packed_size = use_gemv ? 0 : lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        எனில் (!use_gemv) {
            GGML_ASSERT(lhs_packed_size <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        மாறிலி size_t dst_stride = dst->nb[1];
        மாறிலி size_t n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        மாறிலி தருக்க disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) INT_MAX);

        மாறிலி int64_t batch_size = ne12 * ne13;
        க்கு (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            மாறிலி int64_t i12 = batch_idx % ne12;
            மாறிலி int64_t i13 = batch_idx / ne12;
            மாறிலி uint8_t * lhs_batch_base = static_cast<மாறிலி uint8_t *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];

            எனில் (!use_gemv) {
                மாறிலி int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                மாறிலி int64_t use_threads = std::min<int64_t>(nth, max_threads);

                எனில் (ith < use_threads) {
                    மாறிலி int64_t இலக்கம்_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), mr);
                    மாறிலி int64_t இலக்கம்_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * இலக்கம்_m_per_thread0;

                    மாறிலி int64_t m_start = (int64_t)ith * இலக்கம்_m_per_thread0;
                    மாறிலி int64_t m_மொத்தம் = (ith == use_threads - 1) ? இலக்கம்_m_per_thread_N_1 : இலக்கம்_m_per_thread0;

                    மாறிலி size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    மாறிலி size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    மாறிலி size_t row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_மொத்தம்;
                    int64_t cur       = m_start;

                    போது (remaining > 0) {
                        மாறிலி int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        மாறிலி size_t src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        மாறிலி வெறுமை * src_ptr = lhs_batch_base + src_off;
                        மாறிலி size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        வெறுமை * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            எனில் (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            மாறிலி size_t chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (size_t) INT_MAX);

            முழு current_col = ggml_threadpool_chunk_add(params->threadpool, (முழு) chunk_cols);
            போது ((size_t) current_col < n) {
                மாறிலி size_t n_start = (size_t) current_col;
                மாறிலி size_t n_to_process = std::min(chunk_cols, n - n_start);

                எனில் (n_to_process > 0) {
                    மாறிலி size_t lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    மாறிலி size_t rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    மாறிலி size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    மாறிலி வெறுமை * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    மாறிலி வெறுமை * rhs_ptr = rhs_base + rhs_packed_offset;
                    தசம * dst_ptr      = reinterpret_cast<தசம *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          அளவு(தசம),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (முழு) chunk_cols);
            }

            எனில் (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        திரும்பு உண்மை;
    }

    தருக்க compute_forward_fp16(ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst) {
        மாறிலி ggml_tensor * src0 = dst->src[0];
        மாறிலி ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_இருமம்_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        எனில் (!kernels) {
            திரும்பு பொய்;
        }

        மாறிலி தருக்க is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        எனில் (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            திரும்பு பொய்;
        }

        மாறிலி முழு nth = params->nth;
        மாறிலி முழு ith = params->ith;

        மாறிலி int64_t lhs_batch_size0 = ne12;
        மாறிலி int64_t rhs_batch_size0 = ne02;
        மாறிலி int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        மாறிலி int64_t r = lhs_batch_size0 / rhs_batch_size0;

        மாறிலி int64_t m_group = ne11;
        மாறிலி int64_t m       = m_group;
        மாறிலி int64_t n       = ne01;
        மாறிலி int64_t k       = ne00;

        மாறிலி size_t lhs_stride = src1->nb[1];
        மாறிலி size_t rhs_stride = src0->nb[1];
        மாறிலி size_t dst_stride = dst->nb[1];

        மாறிலி int64_t mr = (int64_t) kernel->get_mr();
        மாறிலி int64_t nr = (int64_t) kernel->get_nr();
        மாறிலி int64_t kr = (int64_t) kernel->get_kr();
        மாறிலி int64_t sr = (int64_t) kernel->get_sr();

        மாறிலி size_t lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        மாறிலி size_t rhs_packed_size = kernels->rhs_info.packed_size_ex(n, k, nr, kr, 0);
        மாறிலி size_t kxn_size        = k * n * அளவு(தசம);
        மாறிலி size_t bias_size       = n * அளவு(தசம);

        மாறிலி size_t wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        க்கு (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            மாறிலி int64_t rhs_batch_idx = batch_idx / r;
            மாறிலி uint8_t * rhs_batch_base = static_cast<மாறிலி uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                மாறிலி int64_t m_roundup_mr = kai_roundup(m, mr);
                மாறிலி int64_t இலக்கம்_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                எனில் (ith < இலக்கம்_threads) {
                    மாறிலி int64_t இலக்கம்_m_per_thread0   = round_down((size_t)(m_roundup_mr / இலக்கம்_threads), (size_t)mr);
                    மாறிலி int64_t இலக்கம்_m_per_thread_N_1 = m - (இலக்கம்_threads - 1) * இலக்கம்_m_per_thread0;

                    மாறிலி int64_t m_start = ith * இலக்கம்_m_per_thread0;
                    மாறிலி int64_t m_மொத்தம் = (ith == இலக்கம்_threads - 1) ? இலக்கம்_m_per_thread_N_1 : இலக்கம்_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    மாறிலி size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    மாறிலி size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    மாறிலி size_t row_stride_bytes = (next_block_off - base_packed_off) / (size_t)mr;

                    int64_t remaining = m_மொத்தம்;
                    int64_t cur       = m_start;

                    போது (remaining > 0) {
                        மாறிலி int64_t row_in_group = cur;
                        மாறிலி int64_t avail        = m_group - row_in_group;
                        மாறிலி int64_t take         = std::min(avail, remaining);

                        மாறிலி uint8_t * lhs_batch_base = static_cast<மாறிலி uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        மாறிலி வெறுமை * src_ptr = lhs_batch_base + (size_t)row_in_group * lhs_stride;
                        மாறிலி size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        வெறுமை * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            எனில் (ith == 0) {
                memset(bias, 0, (size_t)n * அளவு(தசம));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<தசம *>(rhs_kxn),
                                        reinterpret_cast<மாறிலி uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * அளவு(தசம),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                மாறிலி int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t இலக்கம்_threads_n = KAI_MIN(n / n_step, nth);
                எனில் (இலக்கம்_threads_n <= 0) {
                    இலக்கம்_threads_n = 1;
                }

                எனில் (ith < இலக்கம்_threads_n) {
                    மாறிலி int64_t இலக்கம்_n_per_thread0   = round_down((size_t)(n / இலக்கம்_threads_n), (size_t)n_step);
                    மாறிலி int64_t இலக்கம்_n_per_thread_N_1 = n - (இலக்கம்_threads_n - 1) * இலக்கம்_n_per_thread0;

                    மாறிலி int64_t n_start      = ith * இலக்கம்_n_per_thread0;
                    மாறிலி int64_t n_to_process = (ith == இலக்கம்_threads_n - 1) ? இலக்கம்_n_per_thread_N_1 : இலக்கம்_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    மாறிலி size_t lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    மாறிலி size_t rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    மாறிலி size_t dst_offset         = kernel->get_dst_offset((size_t)0, (size_t)n_start, dst_stride);

                    மாறிலி வெறுமை * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    மாறிலி வெறுமை * rhs_ptr = rhs_packed + rhs_packed_offset;
                    தசம * dst_ptr      = reinterpret_cast<தசம *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, அளவு(தசம), -FLT_MAX, FLT_MAX);
                }
            }

            எனில் (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        திரும்பு உண்மை;
    }

    தருக்க compute_forward_qx(கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        மாறிலி ggml_tensor * src0 = dst->src[0];
        மாறிலி ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_இருமம்_OP_LOCALS

        மாறிலி kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        மாறிலி தருக்க has_header = kleidiai_is_weight_header_valid(header);
        மாறிலி தருக்க is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        மாறிலி முழு slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        auto weight_for_slot = [&](முழு slot_index, size_t & size_out) -> மாறிலி uint8_t * {
            எனில் (slot_index < 0 || slot_index >= slot_total) {
                திரும்பு nullptr;
            }
            எனில் (has_header) {
                எனில் (slot_index < header->slot_மொத்தம்) {
                    size_out = static_cast<size_t>(header->sizes[slot_index]);
                    திரும்பு kleidiai_weight_slot_ptr(header, slot_index);
                }
                திரும்பு nullptr;
            }
            எனில் (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                திரும்பு static_cast<மாறிலி uint8_t *>(src0->data);
            }
            திரும்பு nullptr;
        };

        கட்டமைப்பு runtime_slot {
            முழு slot_index;
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
            முழு assigned_threads;
            முழு thread_begin;
            முழு thread_end;
            மாறிலி uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        முழு runtime_மொத்தம் = 0;

        க்கு (முழு slot = 0; slot < slot_total && runtime_மொத்தம் < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            எனில் (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                தொடர்;
            }

            size_t rhs_size = 0;
            மாறிலி uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            எனில் (!rhs_ptr || rhs_size == 0) {
                தொடர்;
            }

            மாறிலி kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

            runtime[runtime_மொத்தம்] = {
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
            ++runtime_மொத்தம்;
        }

        எனில் (runtime_மொத்தம் == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            திரும்பு பொய்;
        }

        மாறிலி முழு nth_total = params->nth > 0 ? params->nth : 1;
        மாறிலி முழு ith_total = params->ith;

        முழு sme_slot = -1;
        முழு non_sme_slot = -1;
        க்கு (முழு i = 0; i < runtime_மொத்தம்; ++i) {
            எனில் (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                நிறுத்து;
            }
        }

        க்கு (முழு i = 0; i < runtime_மொத்தம்; ++i) {
            எனில் (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                நிறுத்து;
            }
        }

        மாறிலி முழு sme_cap_limit = ctx.sme_thread_cap;
        மாறிலி தருக்க use_hybrid = sme_cap_limit > 0 &&
                                 runtime_மொத்தம் > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_cols_per_thread = 0;
        எனில் (runtime_மொத்தம் > 0 && nth_total > 0) {
            min_cols_per_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        மாறிலி தருக்க too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        மாறிலி தருக்க hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        எனில் (!hybrid_enabled) {
            முழு chosen_slot = 0;
            எனில் (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } இல்லையெனில் எனில் (runtime_மொத்தம் > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            எனில் (chosen_slot != 0 && chosen_slot < runtime_மொத்தம்) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_மொத்தம் = runtime_மொத்தம் > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            எனில் (runtime_மொத்தம் > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        மாறிலி தருக்க sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        முழு sme_cap = sme_single_only ? nth_total : kleidiai_sme_thread_cap();
        எனில் (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        முழு threads_remaining = nth_total;
        எனில் (sme_slot != -1) {
            முழு sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        முழு fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        முழு fallback_மொத்தம் = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        க்கு (முழு i = 0; i < runtime_மொத்தம்; ++i) {
            எனில் (i == sme_slot) {
                தொடர்;
            }
            fallback_indices[fallback_மொத்தம்++] = i;
        }

        க்கு (முழு fi = 0; fi < fallback_மொத்தம்; ++fi) {
            எனில் (threads_remaining <= 0) {
                நிறுத்து;
            }
            மாறிலி முழு slot_index = fallback_indices[fi];
            மாறிலி முழு slots_இடது = fallback_மொத்தம் - fi;
            முழு share = (threads_remaining + slots_இடது - 1) / slots_இடது;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        முழு thread_cursor = 0;
        க்கு (முழு i = 0; i < runtime_மொத்தம்; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        மாறிலி முழு active_threads = thread_cursor;

        முழு local_slot = -1;
        முழு local_ith  = 0;
        க்கு (முழு i = 0; i < runtime_மொத்தம்; ++i) {
            எனில் (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                நிறுத்து;
            }
        }
        மாறிலி size_t k = ne00;
        மாறிலி size_t m = ne11;
        மாறிலி size_t n = ne01;

        size_t cursor = 0;
        க்கு (முழு i = 0; i < runtime_மொத்தம்; ++i) {
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_step = 1;
        க்கு (முழு i = 0; i < runtime_மொத்தம்; ++i) {
            எனில் (runtime[i].assigned_threads == 0) {
                தொடர்;
            }
            size_t next_step = 0;
            எனில் (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                திரும்பு பொய்;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        மாறிலி தருக்க disable_chunking = ggml_is_numa();
        மாறிலி size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        மாறிலி size_t chunk_வகுஎண் = (active_threads == 1 || disable_chunking) ? (size_t) active_threads : (size_t) active_threads * chunk_multiplier;
        size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, chunk_வகுஎண்)), common_step);
        எனில் (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        மாறிலி size_t nchunk_size = std::max<size_t>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (size_t)INT_MAX);
        மாறிலி முழு nchunk = (முழு)nchunk_size;
        மாறிலி size_t dst_stride = dst->nb[1];

        auto run_chunk = [&](runtime_slot & slot, size_t global_start, size_t cols, uint8_t * dst_batch_base) {
            மாறிலி size_t rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            மாறிலி size_t dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            மாறிலி uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            மாறிலி uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            தசம * dst_ptr         = reinterpret_cast<தசம *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       அளவு(தசம),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        க்கு (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            மாறிலி uint8_t * lhs_batch_base = static_cast<மாறிலி uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            எனில் (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                மாறிலி int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                மாறிலி int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                எனில் (local_ith < use_threads) {
                    மாறிலி int64_t இலக்கம்_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), slot.mr);
                    மாறிலி int64_t இலக்கம்_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * இலக்கம்_m_per_thread0;

                    மாறிலி int64_t m_start = (int64_t)local_ith * இலக்கம்_m_per_thread0;
                    மாறிலி int64_t m_மொத்தம் = (local_ith == use_threads - 1) ? இலக்கம்_m_per_thread_N_1 : இலக்கம்_m_per_thread0;

                    மாறிலி size_t base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    மாறிலி size_t next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    மாறிலி size_t row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_மொத்தம்;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    போது (remaining > 0) {
                        மாறிலி int64_t row_in_group = cur;
                        மாறிலி int64_t avail        = (int64_t)m - row_in_group;
                        மாறிலி int64_t take         = std::min(avail, remaining);

                        மாறிலி size_t src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        மாறிலி வெறுமை * src_ptr = lhs_batch_base + src_off;
                        மாறிலி size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        வெறுமை * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            எனில் (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            எனில் (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                முழு current_chunk = ith_total;
                போது (current_chunk < nchunk) {
                    மாறிலி size_t global_start = (size_t)current_chunk * chunk_cols;
                    எனில் (global_start >= n) {
                        நிறுத்து;
                    }

                    மாறிலி size_t cols = std::min(chunk_cols, n - global_start);
                    எனில் (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            எனில் (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        திரும்பு உண்மை;
    }

பொது:
    முழு repack(கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        மாறிலி size_t n = tensor->ne[1];
        மாறிலி size_t k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        எனில் (!header) {
            திரும்பு -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_மொத்தம் = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        size_t cursor = அளவு(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        மாறிலி தருக்க want_q4k = tensor->type == GGML_TYPE_Q4_K;
        மாறிலி தருக்க want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        மாறிலி தருக்க want_f32 = tensor->type == GGML_TYPE_F32;
        மாறிலி முழு slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        மாறிலி தருக்க allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<தசம>  scales;
        std::vector<தசம>  bias;
        std::vector<தசம> row_f32;

        எனில் (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            மாறிலி size_t row_stride = tensor->nb[1];
            மாறிலி size_t k_blocks   = (k + QK8_0 - 1) / QK8_0;

            எனில் (tensor->type == GGML_TYPE_Q6_K) {
                எனில் (k % QK_K != 0 || k % QK8_0 != 0) {
                    திரும்பு -1;
                }
                row_f32.resize(k);
            }

            க்கு (size_t row = 0; row < n; ++row) {
                மாறிலி uint8_t * row_data = static_cast<மாறிலி uint8_t *>(data) + row * row_stride;

                எனில் (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<மாறிலி block_q6_K *>(row_data), row_f32.data(), k);
                }

                தசம max_abs = 0.0f;
                எனில் (tensor->type == GGML_TYPE_Q6_K) {
                    க்கு (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::fabs(row_f32[linear_idx]));
                    }
                } இல்லையெனில் {
                    மாறிலி block_q8_0 * row_blocks = reinterpret_cast<மாறிலி block_q8_0 *>(row_data);
                    க்கு (size_t block = 0; block < k_blocks; ++block) {
                        மாறிலி block_q8_0 & blk = row_blocks[block];
                        மாறிலி தசம d = GGML_FP16_TO_FP32(blk.d);
                        க்கு (size_t l = 0; l < QK8_0; ++l) {
                            மாறிலி size_t linear_idx = block * QK8_0 + l;
                            எனில் (linear_idx >= k) {
                                நிறுத்து;
                            }
                            மாறிலி தசம value = d * static_cast<தசம>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                தசம scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                மாறிலி தசம inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                எனில் (tensor->type == GGML_TYPE_Q6_K) {
                    க்கு (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } இல்லையெனில் {
                    மாறிலி block_q8_0 * row_blocks = reinterpret_cast<மாறிலி block_q8_0 *>(row_data);
                    க்கு (size_t block = 0; block < k_blocks; ++block) {
                        மாறிலி block_q8_0 & blk = row_blocks[block];
                        மாறிலி தசம d = GGML_FP16_TO_FP32(blk.d);
                        க்கு (size_t l = 0; l < QK8_0; ++l) {
                            மாறிலி size_t linear_idx = block * QK8_0 + l;
                            எனில் (linear_idx >= k) {
                                நிறுத்து;
                            }
                            மாறிலி தசம value = d * static_cast<தசம>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        எனில் (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
        }

        க்கு (முழு slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            எனில் (!allow_fallback && slot > 0) {
                நிறுத்து;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            எனில் (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                தொடர்;
            }

            மாறிலி size_t nr = kernel->get_nr();
            மாறிலி size_t kr = kernel->get_kr();
            மாறிலி size_t sr = kernel->get_sr();
            மாறிலி ggml_type rhs_type = kernels->rhs_type;
            எனில் (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            மாறிலி size_t block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            எனில் (block_len == SIZE_MAX) {
                தொடர்;
            }

            மாறிலி size_t packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            மாறிலி size_t aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            எனில் (rhs_type == GGML_TYPE_Q4_0) {
                கட்டமைப்பு kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<மாறிலி uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } இல்லையெனில் எனில் (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       data, nullptr, nullptr, dst_ptr, 0, nullptr);
            } இல்லையெனில் எனில் (rhs_type == GGML_TYPE_Q8_0) {
                கட்டமைப்பு kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } இல்லையெனில் எனில் (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
            } இல்லையெனில் {
                தொடர்;
            }

            header->offsets[header->slot_மொத்தம்] = aligned_cursor;
            header->sizes[header->slot_மொத்தம்]   = packed_size;
            ++header->slot_மொத்தம்;

            cursor = aligned_cursor + packed_size;
        }

        எனில் (header->slot_மொத்தம் == 0) {
            header->magic   = 0;
            header->version = 0;
            memcpy(tensor->data, data, data_size);
        }

        திரும்பு 0;
    }
};

நிலையான ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, கட்டமைப்பு ggml_tensor *) {
    நிலையான tensor_traits traits;
    திரும்பு &traits;
}
}  // namespace ggml::cpu::kleidiai

நிலையான எண்ணிக்கை ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, கட்டமைப்பு ggml_tensor * tensor) {
    tensor->extra = (வெறுமை *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    திரும்பு GGML_STATUS_SUCCESS;
}

நிலையான வெறுமை ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, கட்டமைப்பு ggml_tensor * tensor,
                                                       மாறிலி வெறுமை * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

நிலையான மாறிலி எழுத்து * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    திரும்பு "CPU_KLEIDIAI";
}

நிலையான ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    எனில் (buffer == nullptr) {
        திரும்பு nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    திரும்பு buffer;
}

நிலையான size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    திரும்பு TENSOR_ALIGNMENT;
}

நிலையான size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, மாறிலி கட்டமைப்பு ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    எனில் (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        திரும்பு ggml_nbytes(tensor);
    }

    மாறிலி size_t n = tensor->ne[1];
    மாறிலி size_t k = tensor->ne[0];

    size_t cursor = அளவு(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    மாறிலி தருக்க want_q4k = tensor->type == GGML_TYPE_Q4_K;
    மாறிலி தருக்க want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    மாறிலி தருக்க want_f32 = tensor->type == GGML_TYPE_F32;
    மாறிலி முழு slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    மாறிலி தருக்க allow_fallback = kleidiai_pack_fallback_allowed();

    size_t slot_மொத்தம் = 0;
    க்கு (முழு slot = 0; slot < slot_total; ++slot) {
        எனில் (!allow_fallback && slot > 0) {
            நிறுத்து;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        எனில் (!kernels) {
            தொடர்;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        எனில் (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            தொடர்;
        }

        மாறிலி ggml_type rhs_type = kernels->rhs_type;
        மாறிலி size_t block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        எனில் (block_len == SIZE_MAX) {
            தொடர்;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_மொத்தம்;
    }

    எனில் (slot_மொத்தம் == 0) {
        திரும்பு ggml_nbytes(tensor);
    }

    திரும்பு std::max(cursor, ggml_nbytes(tensor));
}

நாமாகாஶ ggml::cpu::kleidiai {
வகுப்பு extra_buffer_type : ggml::cpu::extra_buffer_type {
    தருக்க supports_op(ggml_backend_dev_t, மாறிலி கட்டமைப்பு ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        எனில் (op->op != GGML_OP_MUL_MAT) {
            திரும்பு பொய்;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        மாறிலி முழு slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        மாறிலி தருக்க src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft->context == இது &&
            slot_total > 0;

        எனில் ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            எனில் (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                திரும்பு பொய்;
            }
            எனில் (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                திரும்பு பொய்;
            }
            எனில் (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                திரும்பு பொய்;
            }
            எனில் (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                திரும்பு பொய்;
            }
            எனில் (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                திரும்பு பொய்;
            }
            எனில் (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                திரும்பு பொய்;
            }

            எனில் (op->src[0]->type == GGML_TYPE_Q4_K) {
                எனில் (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    திரும்பு பொய்;
                }
                திரும்பு ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == அளவு(தசம) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * அளவு(தசம));
            }

            எனில் (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                எனில் (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    திரும்பு உண்மை;
                }
                திரும்பு பொய்;
            }

            எனில் (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                திரும்பு பொய்;
            }

            திரும்பு உண்மை;
        }

        திரும்பு பொய்;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(மாறிலி கட்டமைப்பு ggml_tensor * op) override {
        எனில் (op->op == GGML_OP_MUL_MAT) {
            எனில் (op->src[0]->buffer && op->src[0]->buffer->buft->context == இது) {
                திரும்பு (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } இல்லையெனில் {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                எனில் (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    நிலையான std::atomic<தருக்க> warned(பொய்);
                    எனில் (!warned.exchange(உண்மை)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                எனில் (op->src[0]->type != GGML_TYPE_F16) {
                    திரும்பு nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                மாறிலி முழு slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                எனில் (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    எனில் ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        திரும்பு nullptr;
                    }
                    திரும்பு ggml::cpu::kleidiai::get_tensor_traits(வெற்று, வெற்று);
                }
            }
        }
        திரும்பு nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(வெறுமை) {
    நிலையான ggml::cpu::kleidiai::extra_buffer_type ctx;
    நிலையான கட்டமைப்பு ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .get_alloc_size_n     = */ வெற்று,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    திரும்பு &ggml_backend_cpu_buffer_type_kleidiai;
}
