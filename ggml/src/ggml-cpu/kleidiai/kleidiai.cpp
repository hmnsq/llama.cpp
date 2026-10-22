// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#ضمن <arm_neon.h>
#ضمن <cassert>
#ضمن <cstdio>
#ضمن <cstdlib>
#ضمن <atomic>
#ضمن <cfloat>
#ضمن <cctype>
#ضمن <algorithm>
#ضمن <cmath>
#ضمن <stdexcept>
#ضمن <stdint.h>
#ضمن <string.h>
#ضمن <string>
#ضمن <vector>
#ضمن <array>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <fstream>
#ضمن <map>
#ضمن <iostream>
#ضمن <climits>
#ضمن <charconv>
#ضمن <system_error>
#אם defined(__linux__)
#ضمن <asm/hwcap.h>
#ضمن <dirent.h>
#ضمن <sys/auxv.h>
#ضمن <sys/types.h>
#ضمن <sys/stat.h>
#ضمن <unistd.h>
#وإلاإذا defined(__APPLE__)
#ضمن <sys/sysctl.h>
#ضمن <sys/types.h>
#endif

#ضمن "kleidiai.h"

#ضمن "ggml-cpu.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-feats.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-quants.h"
#ضمن "ggml-threading.h"
#ضمن "traits.h"

#ضمن "kernels.h"

#ضمن "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#ضمن "ggml-common.h"

ساكن constexpr صحيح      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
ساكن constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
ساكن constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
ساكن constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
ساكن constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

بنية ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    صحيح sme_thread_cap; // <= 0 means "SME disabled/unknown"
    صحيح thread_hint;    // <= 0 means "no hint"
    صحيح chunk_multiplier;
} ساكن ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

ساكن مضمن منطقي is_sme_family(cpu_feature f) {
    החזר (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

ساكن ثابت حرف* cpu_feature_to_string(cpu_feature f) {
    אם (f == CPU_FEATURE_NONE) {
        החזר "NONE";
    } אחרת אם ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        החזר "SME2";
    } אחרת אם ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        החזר "SME";
    } אחרת אם ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        החזר "SVE";
    }
    אחרת אם ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        החזר "I8MM";
    } אחרת אם ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        החזר "DOTPROD";
    }
    אחרת {
        החזר "UNKNOWN";
    }
}

#אם defined(__linux__) && defined(__aarch64__)
ساكن منطقي parse_cpu_dir_name(ثابت حرف* name, size_t* cpu) {
    אם (strncmp(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        החזר שקר;
    }

    ثابت حرف* first = name + 3;
    ثابت حرف* last = name + strlen(name);

    size_t value = 0;
    ثابت auto [end, ec] = std::from_chars(first, last, value, 10);

    אם (ec != std::errc{} || end != last) {
        החזר שקר;
    }

    *cpu = value;
    החזר אמת;
}

ساكن std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    אם (dir == nullptr) {
        החזר cpus;
    }

    כלעוד (dirent * entry = readdir(dir)) {
        size_t cpu = 0;
        אם (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    החזר cpus;
}
#endif

#אם defined(__APPLE__) && defined(__aarch64__)
ساكن منطقي apple_sme_counted_perf_level(std::string name) {
    עבור (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (حرف) std::tolower((unsigned حرف) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    החזר name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

ساكن فراغ add_smcus_from_smidr(uint64_t smidr, size_t & num_private, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ثابت uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ثابت uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ثابت size_t shared_count = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ثابت uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ثابت uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ثابت uint32_t id = (affinity2 << 12) | affinity;

    אם (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    بدل (sh) {
        حالة 2: // private SMCU
            ++num_private;
            اكسر;
        حالة 3: // shared SMCU
            אם (shared_counts[id] < shared_count) {
                shared_counts[id] = shared_count;
            }
            اكسر;
        حالة 0:
            אם (id == 0) {
                ++num_private;
            } אחרת אם (shared_counts[id] < shared_count) {
                shared_counts[id] = shared_count;
            }
            اكسر;
        افتراضي:
            اكسر;
    }
}

ساكن size_t detect_num_smcus() {
    ثابت auto runtime_feat = ggml_feats_get_arch64_runtime();
    אם (!runtime_feat.has_sme) {
        החזר 0;
    }

#אם defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t num_private = 0;
    std::map<uint32_t, size_t> shared_counts;

    ثابت std::vector<size_t> cpus = detect_cpu_ids();
    עבור (ثابت size_t cpu : cpus) {
        ثابت std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        אם (!file.is_open()) {
            استمر;
        }

        uint64_t smidr = 0;
        אם (!(file >> std::hex >> smidr)) {
            استمر;
        }

        add_smcus_from_smidr(smidr, num_private, shared_counts);
    }

    size_t total = num_private;
    עבור (ثابت auto & entry : shared_counts) {
        total += entry.second;
    }
    החזר total;

#وإلاإذا defined(__APPLE__) && defined(__aarch64__)
    صحيح perf_levels = 0;
    size_t size = حجم(perf_levels);
    אם (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != حجم(perf_levels) || perf_levels <= 0) {
        החזר 0;
    }

    size_t units = 0;
    עבור (صحيح i = 0; i < perf_levels; ++i) {
        حرف key[64] = {};
        صحيح physical_cpus = 0;
        صحيح cpus_per_l2 = 0;

        snprintf(key, حجم(key), "hw.perflevel%d.physicalcpu", i);
        size = حجم(physical_cpus);
        אם (sysctlbyname(key, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != حجم(physical_cpus) || physical_cpus <= 0) {
            استمر;
        }

        snprintf(key, حجم(key), "hw.perflevel%d.cpusperl2", i);
        size = حجم(cpus_per_l2);
        אם (sysctlbyname(key, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != حجم(cpus_per_l2) || cpus_per_l2 <= 0) {
            استمر;
        }

        snprintf(key, حجم(key), "hw.perflevel%d.name", i);
        size = 0;
        אם (sysctlbyname(key, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            استمر;
        }

        std::string name(size, '\0');
        אם (sysctlbyname(key, &name[0], &size, nullptr, 0) != 0) {
            استمر;
        }
        name.resize(size);
        כלעוד (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        אם (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    החזר units;

#وإلاإذا defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    החזר 0;

#אחרת
    החזר 0;
#endif
}

ساكن صحيح parse_uint_env(ثابت حرف *s, ثابت حرف *name, منطقي *ok) {
    אם (!s) { *ok = שקר; החזר 0; }
    حرف *end = nullptr;
    long v = strtol(s, &end, 10);
    אם (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = שקר;
        החזר 0;
    }
    אם (v < 0 || v > INT_MAX) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = שקר;
        החזר 0;
    }
    *ok = אמת;
    החזר (صحيح)v;
}

ساكن فراغ init_kleidiai_context(فراغ) {
    ggml_critical_section_start();
    ساكن منطقي initialized = שקר;

    אם (!initialized) {
        initialized = אמת;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ثابت حرف *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ثابت حرف *env_threads     = getenv("GGML_TOTAL_THREADS");
        ثابت حرف *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ثابت auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        אם (env_threads) {
            منطقي ok = שקר;
            صحيح hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            אם (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        אם (env_chunk_mult) {
            منطقي ok = שקר;
            صحيح multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            אם (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        صحيح sme_cores = 0;
        منطقي sme_env_ok = שקר;
        منطقي sme_env_set = (env_sme != nullptr);

        ثابت منطقي has_supported_sme_family = runtime_feat.has_sme;
        منطقي sme_cap_detected = שקר;

        אם (has_supported_sme_family) {
            detected_smcus = detect_num_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (صحيح)detected_smcus : 1;

            אם (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        אם (sme_env_set) {
            منطقي ok = שקר;
            صحيح v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            אם (ok) {
                אם (has_supported_sme_family) {
                    sme_cores = v;
                } אחרת {
                    אם (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } אחרת {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        אם (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            אם (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        אם (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (صحيح)ctx.features);
        } אחרת {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        אם (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (صحيح)ctx.features);
        } אחרת {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        אם (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (صحيح) ctx.features);
        } אחרת {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        אם (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (صحيح)ctx.features);
        } אחרת {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        ثابت منطقي has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        אם (has_selected_sme_family_kernel) {
            אם (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } אחרת אם (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } אחרת {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } אחרת {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_end();
}

ساكن مضمن صحيح kleidiai_sme_thread_cap() {
    החזר ctx.sme_thread_cap;
}

ساكن مضمن size_t align_up(size_t value, size_t alignment) {
    אם (alignment == 0) {
        החזר value;
    }
    ثابت size_t remainder = value % alignment;
    החזר remainder == 0 ? value : value + (alignment - remainder);
}

ساكن مضمن size_t gcd_size(size_t a, size_t b) {
    כלעוד (b != 0) {
        ثابت size_t t = a % b;
        a = b;
        b = t;
    }
    החזר a;
}

ساكن مضمن منطقي lcm_size(size_t a, size_t b, size_t & result) {
    אם (a == 0 || b == 0) {
        result = 0;
        החזר שקר;
    }
    ثابت size_t g = gcd_size(a, b);
    ثابت size_t q = a / g;
    אם (q > SIZE_MAX / b) {
        החזר שקר;
    }
    result = q * b;
    החזר אמת;
}

ساكن مضمن size_t ceil_div_size(size_t a, size_t b) {
    החזר b == 0 ? 0 : (a + b - 1) / b;
}

ساكن مضمن size_t kleidiai_chunk_cols(size_t n, صحيح nth_total, منطقي disable_chunking, size_t n_step) {
    ثابت size_t multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ثابت size_t divisor = std::max<size_t>(1, (size_t) nth_total * multiplier);
    ثابت size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, divisor)), n_step);
    החזר chunk_cols ? chunk_cols : n_step;
}

بنية kleidiai_block_args {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

ساكن مضمن kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    بدل (rhs_type) {
        حالة GGML_TYPE_Q4_0:
            החזר { QK4_0, QK4_0, QK4_0 };
        حالة GGML_TYPE_Q4_K:
            החזר { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        حالة GGML_TYPE_Q8_0:
            החזר { 0, 0, QK8_0 };
        افتراضي:
            החזר { 0, 0, 0 };
    }
}

ساكن مضمن منطقي kleidiai_pack_fallback_allowed() {
    אם (ctx.sme_thread_cap <= 0) {
        החזר שקר;
    }
    אם (ctx.thread_hint <= 0) {
        החזר אמת;
    }
    החזר ctx.thread_hint > ctx.sme_thread_cap;
}

بنية kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_count;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

ساكن مضمن kleidiai_weight_header * kleidiai_weight_header_from_ptr(فراغ * data) {
    החזר reinterpret_cast<kleidiai_weight_header *>(data);
}

ساكن مضمن ثابت kleidiai_weight_header * kleidiai_weight_header_from_ptr(ثابت فراغ * data) {
    החזר reinterpret_cast<ثابت kleidiai_weight_header *>(data);
}

ساكن مضمن منطقي kleidiai_is_weight_header_valid(ثابت kleidiai_weight_header * header) {
    אם (!header) {
        החזר שקר;
    }
    אם (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        החזר שקר;
    }
    אם (header->slot_count == 0 || header->slot_count > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        החזר שקר;
    }
    החזר אמת;
}

ساكن مضمن uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, صحيح slot) {
    אם (!kleidiai_is_weight_header_valid(header)) {
        החזר nullptr;
    }
    אם (slot < 0 || slot >= header->slot_count) {
        החזר nullptr;
    }
    החזר reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

ساكن مضمن ثابت uint8_t * kleidiai_weight_slot_ptr(ثابت kleidiai_weight_header * header, صحيح slot) {
    אם (!kleidiai_is_weight_header_valid(header)) {
        החזר nullptr;
    }
    אם (slot < 0 || slot >= header->slot_count) {
        החזר nullptr;
    }
    החזר reinterpret_cast<ثابت uint8_t *>(header) + header->offsets[slot];
}

ساكن مضمن ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    החזר ctx.kernels_q4;
}

ساكن مضمن ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    החזר ctx.kernels_q4k;
}

ساكن مضمن ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    החזר ctx.kernels_q8;
}

ساكن مضمن ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    החזר ctx.kernels_f32;
}

template <typename SelectFallback>
ساكن صحيح kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    صحيح count = 0;
    אם (!primary) {
        החזר 0;
    }
    out[count++] = primary;

    אם (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        החזר count;
    }

    אם (is_sme_family(primary->required_cpu)) {
        ثابت cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        אם (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            אם (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    החזר count;
}

ساكن صحيح kleidiai_collect_kernel_chain(ثابت بنية ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    החזר kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { החזר ggml_kleidiai_select_kernels(mask, op); });
}

ساكن صحيح kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    החזר kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { החזר ggml_kleidiai_select_kernels_q4_0(mask); });
}

ساكن صحيح kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    החזר kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { החזר ggml_kleidiai_select_kernels_q4_k(mask); });
}

ساكن صحيح kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    החזר kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { החזר ggml_kleidiai_select_kernels_q8_0(mask); });
}

ساكن صحيح kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    החזר kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { החזר ggml_kleidiai_select_kernels_f32(mask); });
}

ساكن مضمن int64_t ggml_ne(ثابت ggml_tensor * tensor, صحيح dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    החזר tensor->ne[dim];
}

نطاق ggml::cpu::kleidiai {

ساكن size_t round_down(size_t x, size_t y) {
    החזר y == 0 ? x : x - (x % y);
}

ساكن فراغ transpose_f32kxn_f16nxk(size_t n, size_t k, عائم * dst, ثابت uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / حجم(uint16_t);
    size_t dst_stride = n;

    עבור (size_t k_idx = 0; k_idx < k; ++k_idx) {
        עבור (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

מחלקה tensor_traits : عام ggml::cpu::tensor_traits {
    منطقي work_size(صحيح /* n_threads */, ثابت بنية ggml_tensor * op, size_t & size) override {
        אם (op->op != GGML_OP_MUL_MAT) {
            החזר שקר;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحيح slot_count = kleidiai_collect_kernel_chain(op, kernel_chain);
        אם (slot_count == 0) {
            החזר שקר;
        }

        ثابت منطقي is_gemv = op->src[1]->ne[1] == 1;
        ثابت size_t k = op->src[0]->ne[0];
        ثابت size_t n = op->src[0]->ne[1];
        ثابت size_t m = op->src[1]->ne[1];

        אם (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ثابت kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            ثابت size_t qk = block_args.pack_bl;

            size_t cursor = 0;
            منطقي any_slot = שקר;

            עבור (صحيح slot = 0; slot < slot_count; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                אם (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    החזר שקר;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                ثابت size_t packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = אמת;
            }

            אם (!any_slot) {
                החזר שקר;
            }

            size = cursor;
            החזר אמת;
        }

        אם (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            אם (is_gemv && op->src[1]->nb[0] == (int64_t) حجم(عائم) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                החזר אמת;
            }

            size_t cursor = 0;
            منطقي any_slot = שקר;

            עבור (صحيح slot = 0; slot < slot_count; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                אם (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    החזר שקר;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
                any_slot = אמת;
            }

            אם (!any_slot) {
                החזר שקר;
            }

            size = cursor;
            החזר אמת;
        }

        אם (op->src[0]->type == GGML_TYPE_F16) {
            ثابت int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ثابت int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ثابت int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            منطقي any_slot = שקר;

            עבור (صحيح slot = 0; slot < slot_count; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                אם (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    החזר שקר;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = אמת;
            }

            עבור (صحيح slot = 0; slot < slot_count; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                אם (!kernel || !kernels->rhs_info.packed_size_ex) {
                    החזר שקר;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * حجم(عائم);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * حجم(عائم);

            אם (!any_slot) {
                החזר שקר;
            }

            size = cursor;
            החזר אמת;
        }

        החזר שקר;
    }

    منطقي compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * dst) override {
        אם (dst->op == GGML_OP_MUL_MAT) {
            אם (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                החזר compute_forward_qx(params, dst);
            } אחרת אם (dst->src[0]->type == GGML_TYPE_F32) {
                החזר compute_forward_f32(params, dst);
            } אחרת אם (dst->src[0]->type == GGML_TYPE_F16) {
                החזר compute_forward_fp16(params, dst);
            }
        }
        החזר שקר;
    }

    منطقي compute_forward_f32(ggml_compute_params * params, بنية ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        אם (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            החזר שקר;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        אם (!kernels) {
            החזר שקר;
        }

        ثابت size_t k = ne00;
        ثابت size_t m = ne11;
        ثابت size_t n = ne01;
        ثابت منطقي use_gemv = m == 1 && src1->nb[0] == (int64_t) حجم(عائم) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        אם (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            החזר שקר;
        }

        אם (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_size_ex || !lhs_info->pack_func_ex)) {
            החזר שקר;
        }

        ثابت kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ثابت منطقي has_header = kleidiai_is_weight_header_valid(header);

        ثابت uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<ثابت uint8_t *>(src0->data);
        אם (!rhs_base) {
            החזר שקר;
        }

        ثابت صحيح nth = params->nth > 0 ? params->nth : 1;
        ثابت صحيح ith = params->ith;

        ثابت size_t mr = kernel->get_mr();
        ثابت size_t kr = kernel->get_kr();
        ثابت size_t sr = kernel->get_sr();

        ثابت size_t lhs_packed_size = use_gemv ? 0 : lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        אם (!use_gemv) {
            GGML_ASSERT(lhs_packed_size <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ثابت size_t dst_stride = dst->nb[1];
        ثابت size_t n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        ثابت منطقي disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) INT_MAX);

        ثابت int64_t batch_size = ne12 * ne13;
        עבור (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ثابت int64_t i12 = batch_idx % ne12;
            ثابت int64_t i13 = batch_idx / ne12;
            ثابت uint8_t * lhs_batch_base = static_cast<ثابت uint8_t *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];

            אם (!use_gemv) {
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                ثابت int64_t use_threads = std::min<int64_t>(nth, max_threads);

                אם (ith < use_threads) {
                    ثابت int64_t num_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), mr);
                    ثابت int64_t num_m_per_threadN_1 = (int64_t)m - (use_threads - 1) * num_m_per_thread0;

                    ثابت int64_t m_start = (int64_t)ith * num_m_per_thread0;
                    ثابت int64_t m_count = (ith == use_threads - 1) ? num_m_per_threadN_1 : num_m_per_thread0;

                    ثابت size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ثابت size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ثابت size_t row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_count;
                    int64_t cur       = m_start;

                    כלעוד (remaining > 0) {
                        ثابت int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        ثابت size_t src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        ثابت فراغ * src_ptr = lhs_batch_base + src_off;
                        ثابت size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            אם (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ثابت size_t chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (size_t) INT_MAX);

            صحيح current_col = ggml_threadpool_chunk_add(params->threadpool, (صحيح) chunk_cols);
            כלעוד ((size_t) current_col < n) {
                ثابت size_t n_start = (size_t) current_col;
                ثابت size_t n_to_process = std::min(chunk_cols, n - n_start);

                אם (n_to_process > 0) {
                    ثابت size_t lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ثابت size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    ثابت فراغ * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    ثابت فراغ * rhs_ptr = rhs_base + rhs_packed_offset;
                    عائم * dst_ptr      = reinterpret_cast<عائم *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          حجم(عائم),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (صحيح) chunk_cols);
            }

            אם (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        החזר אמת;
    }

    منطقي compute_forward_fp16(ggml_compute_params * params, بنية ggml_tensor * dst) {
        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        אם (!kernels) {
            החזר שקר;
        }

        ثابت منطقي is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        אם (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            החזר שקר;
        }

        ثابت صحيح nth = params->nth;
        ثابت صحيح ith = params->ith;

        ثابت int64_t lhs_batch_size0 = ne12;
        ثابت int64_t rhs_batch_size0 = ne02;
        ثابت int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        ثابت int64_t r = lhs_batch_size0 / rhs_batch_size0;

        ثابت int64_t m_group = ne11;
        ثابت int64_t m       = m_group;
        ثابت int64_t n       = ne01;
        ثابت int64_t k       = ne00;

        ثابت size_t lhs_stride = src1->nb[1];
        ثابت size_t rhs_stride = src0->nb[1];
        ثابت size_t dst_stride = dst->nb[1];

        ثابت int64_t mr = (int64_t) kernel->get_mr();
        ثابت int64_t nr = (int64_t) kernel->get_nr();
        ثابت int64_t kr = (int64_t) kernel->get_kr();
        ثابت int64_t sr = (int64_t) kernel->get_sr();

        ثابت size_t lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        ثابت size_t rhs_packed_size = kernels->rhs_info.packed_size_ex(n, k, nr, kr, 0);
        ثابت size_t kxn_size        = k * n * حجم(عائم);
        ثابت size_t bias_size       = n * حجم(عائم);

        ثابت size_t wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        עבור (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ثابت int64_t rhs_batch_idx = batch_idx / r;
            ثابت uint8_t * rhs_batch_base = static_cast<ثابت uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ثابت int64_t m_roundup_mr = kai_roundup(m, mr);
                ثابت int64_t num_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                אם (ith < num_threads) {
                    ثابت int64_t num_m_per_thread0   = round_down((size_t)(m_roundup_mr / num_threads), (size_t)mr);
                    ثابت int64_t num_m_per_threadN_1 = m - (num_threads - 1) * num_m_per_thread0;

                    ثابت int64_t m_start = ith * num_m_per_thread0;
                    ثابت int64_t m_count = (ith == num_threads - 1) ? num_m_per_threadN_1 : num_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ثابت size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ثابت size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ثابت size_t row_stride_bytes = (next_block_off - base_packed_off) / (size_t)mr;

                    int64_t remaining = m_count;
                    int64_t cur       = m_start;

                    כלעוד (remaining > 0) {
                        ثابت int64_t row_in_group = cur;
                        ثابت int64_t avail        = m_group - row_in_group;
                        ثابت int64_t take         = std::min(avail, remaining);

                        ثابت uint8_t * lhs_batch_base = static_cast<ثابت uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        ثابت فراغ * src_ptr = lhs_batch_base + (size_t)row_in_group * lhs_stride;
                        ثابت size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            אם (ith == 0) {
                memset(bias, 0, (size_t)n * حجم(عائم));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<عائم *>(rhs_kxn),
                                        reinterpret_cast<ثابت uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * حجم(عائم),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ثابت int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t num_threads_n = KAI_MIN(n / n_step, nth);
                אם (num_threads_n <= 0) {
                    num_threads_n = 1;
                }

                אם (ith < num_threads_n) {
                    ثابت int64_t num_n_per_thread0   = round_down((size_t)(n / num_threads_n), (size_t)n_step);
                    ثابت int64_t num_n_per_threadN_1 = n - (num_threads_n - 1) * num_n_per_thread0;

                    ثابت int64_t n_start      = ith * num_n_per_thread0;
                    ثابت int64_t n_to_process = (ith == num_threads_n - 1) ? num_n_per_threadN_1 : num_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ثابت size_t lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ثابت size_t dst_offset         = kernel->get_dst_offset((size_t)0, (size_t)n_start, dst_stride);

                    ثابت فراغ * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ثابت فراغ * rhs_ptr = rhs_packed + rhs_packed_offset;
                    عائم * dst_ptr      = reinterpret_cast<عائم *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, حجم(عائم), -FLT_MAX, FLT_MAX);
                }
            }

            אם (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        החזר אמת;
    }

    منطقي compute_forward_qx(بنية ggml_compute_params * params, بنية ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        ثابت kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ثابت منطقي has_header = kleidiai_is_weight_header_valid(header);
        ثابت منطقي is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحيح slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        auto weight_for_slot = [&](صحيح slot_index, size_t & size_out) -> ثابت uint8_t * {
            אם (slot_index < 0 || slot_index >= slot_total) {
                החזר nullptr;
            }
            אם (has_header) {
                אם (slot_index < header->slot_count) {
                    size_out = static_cast<size_t>(header->sizes[slot_index]);
                    החזר kleidiai_weight_slot_ptr(header, slot_index);
                }
                החזר nullptr;
            }
            אם (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                החזר static_cast<ثابت uint8_t *>(src0->data);
            }
            החזר nullptr;
        };

        بنية runtime_slot {
            صحيح slot_index;
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
            صحيح assigned_threads;
            صحيح thread_begin;
            صحيح thread_end;
            ثابت uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        صحيح runtime_count = 0;

        עבור (صحيح slot = 0; slot < slot_total && runtime_count < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            אם (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                استمر;
            }

            size_t rhs_size = 0;
            ثابت uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            אם (!rhs_ptr || rhs_size == 0) {
                استمر;
            }

            ثابت kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

            runtime[runtime_count] = {
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
            ++runtime_count;
        }

        אם (runtime_count == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            החזר שקר;
        }

        ثابت صحيح nth_total = params->nth > 0 ? params->nth : 1;
        ثابت صحيح ith_total = params->ith;

        صحيح sme_slot = -1;
        صحيح non_sme_slot = -1;
        עבור (صحيح i = 0; i < runtime_count; ++i) {
            אם (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                اكسر;
            }
        }

        עבור (صحيح i = 0; i < runtime_count; ++i) {
            אם (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                اكسر;
            }
        }

        ثابت صحيح sme_cap_limit = ctx.sme_thread_cap;
        ثابت منطقي use_hybrid = sme_cap_limit > 0 &&
                                 runtime_count > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_cols_per_thread = 0;
        אם (runtime_count > 0 && nth_total > 0) {
            min_cols_per_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        ثابت منطقي too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        ثابت منطقي hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        אם (!hybrid_enabled) {
            صحيح chosen_slot = 0;
            אם (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } אחרת אם (runtime_count > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            אם (chosen_slot != 0 && chosen_slot < runtime_count) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_count = runtime_count > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            אם (runtime_count > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        ثابت منطقي sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        صحيح sme_cap = sme_single_only ? nth_total : kleidiai_sme_thread_cap();
        אם (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        صحيح threads_remaining = nth_total;
        אם (sme_slot != -1) {
            صحيح sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        صحيح fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        صحيح fallback_count = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        עבור (صحيح i = 0; i < runtime_count; ++i) {
            אם (i == sme_slot) {
                استمر;
            }
            fallback_indices[fallback_count++] = i;
        }

        עבור (صحيح fi = 0; fi < fallback_count; ++fi) {
            אם (threads_remaining <= 0) {
                اكسر;
            }
            ثابت صحيح slot_index = fallback_indices[fi];
            ثابت صحيح slots_left = fallback_count - fi;
            صحيح share = (threads_remaining + slots_left - 1) / slots_left;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        صحيح thread_cursor = 0;
        עבור (صحيح i = 0; i < runtime_count; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ثابت صحيح active_threads = thread_cursor;

        صحيح local_slot = -1;
        صحيح local_ith  = 0;
        עבור (صحيح i = 0; i < runtime_count; ++i) {
            אם (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                اكسر;
            }
        }
        ثابت size_t k = ne00;
        ثابت size_t m = ne11;
        ثابت size_t n = ne01;

        size_t cursor = 0;
        עבור (صحيح i = 0; i < runtime_count; ++i) {
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_step = 1;
        עבור (صحيح i = 0; i < runtime_count; ++i) {
            אם (runtime[i].assigned_threads == 0) {
                استمر;
            }
            size_t next_step = 0;
            אם (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                החזר שקר;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        ثابت منطقي disable_chunking = ggml_is_numa();
        ثابت size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ثابت size_t chunk_divisor = (active_threads == 1 || disable_chunking) ? (size_t) active_threads : (size_t) active_threads * chunk_multiplier;
        size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, chunk_divisor)), common_step);
        אם (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ثابت size_t nchunk_size = std::max<size_t>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (size_t)INT_MAX);
        ثابت صحيح nchunk = (صحيح)nchunk_size;
        ثابت size_t dst_stride = dst->nb[1];

        auto run_chunk = [&](runtime_slot & slot, size_t global_start, size_t cols, uint8_t * dst_batch_base) {
            ثابت size_t rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            ثابت size_t dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            ثابت uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            ثابت uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            عائم * dst_ptr         = reinterpret_cast<عائم *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       حجم(عائم),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        עבור (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ثابت uint8_t * lhs_batch_base = static_cast<ثابت uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            אם (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                ثابت int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                אם (local_ith < use_threads) {
                    ثابت int64_t num_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), slot.mr);
                    ثابت int64_t num_m_per_threadN_1 = (int64_t)m - (use_threads - 1) * num_m_per_thread0;

                    ثابت int64_t m_start = (int64_t)local_ith * num_m_per_thread0;
                    ثابت int64_t m_count = (local_ith == use_threads - 1) ? num_m_per_threadN_1 : num_m_per_thread0;

                    ثابت size_t base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ثابت size_t next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ثابت size_t row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_count;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    כלעוד (remaining > 0) {
                        ثابت int64_t row_in_group = cur;
                        ثابت int64_t avail        = (int64_t)m - row_in_group;
                        ثابت int64_t take         = std::min(avail, remaining);

                        ثابت size_t src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        ثابت فراغ * src_ptr = lhs_batch_base + src_off;
                        ثابت size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            אם (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            אם (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                صحيح current_chunk = ith_total;
                כלעוד (current_chunk < nchunk) {
                    ثابت size_t global_start = (size_t)current_chunk * chunk_cols;
                    אם (global_start >= n) {
                        اكسر;
                    }

                    ثابت size_t cols = std::min(chunk_cols, n - global_start);
                    אם (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            אם (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        החזר אמת;
    }

عام:
    صحيح repack(بنية ggml_tensor * tensor, ثابت فراغ * data, size_t data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ثابت size_t n = tensor->ne[1];
        ثابت size_t k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        אם (!header) {
            החזר -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_count = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        size_t cursor = حجم(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت منطقي want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ثابت منطقي want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ثابت منطقي want_f32 = tensor->type == GGML_TYPE_F32;
        ثابت صحيح slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ثابت منطقي allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<عائم>  scales;
        std::vector<عائم>  bias;
        std::vector<عائم> row_f32;

        אם (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ثابت size_t row_stride = tensor->nb[1];
            ثابت size_t k_blocks   = (k + QK8_0 - 1) / QK8_0;

            אם (tensor->type == GGML_TYPE_Q6_K) {
                אם (k % QK_K != 0 || k % QK8_0 != 0) {
                    החזר -1;
                }
                row_f32.resize(k);
            }

            עבור (size_t row = 0; row < n; ++row) {
                ثابت uint8_t * row_data = static_cast<ثابت uint8_t *>(data) + row * row_stride;

                אם (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<ثابت block_q6_K *>(row_data), row_f32.data(), k);
                }

                عائم max_abs = 0.0f;
                אם (tensor->type == GGML_TYPE_Q6_K) {
                    עבור (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::fabs(row_f32[linear_idx]));
                    }
                } אחרת {
                    ثابت block_q8_0 * row_blocks = reinterpret_cast<ثابت block_q8_0 *>(row_data);
                    עבור (size_t block = 0; block < k_blocks; ++block) {
                        ثابت block_q8_0 & blk = row_blocks[block];
                        ثابت عائم d = GGML_FP16_TO_FP32(blk.d);
                        עבור (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t linear_idx = block * QK8_0 + l;
                            אם (linear_idx >= k) {
                                اكسر;
                            }
                            ثابت عائم value = d * static_cast<عائم>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                عائم scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                ثابت عائم inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                אם (tensor->type == GGML_TYPE_Q6_K) {
                    עבור (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } אחרת {
                    ثابت block_q8_0 * row_blocks = reinterpret_cast<ثابت block_q8_0 *>(row_data);
                    עבור (size_t block = 0; block < k_blocks; ++block) {
                        ثابت block_q8_0 & blk = row_blocks[block];
                        ثابت عائم d = GGML_FP16_TO_FP32(blk.d);
                        עבור (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t linear_idx = block * QK8_0 + l;
                            אם (linear_idx >= k) {
                                اكسر;
                            }
                            ثابت عائم value = d * static_cast<عائم>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        אם (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
        }

        עבור (صحيح slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            אם (!allow_fallback && slot > 0) {
                اكسر;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            אם (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                استمر;
            }

            ثابت size_t nr = kernel->get_nr();
            ثابت size_t kr = kernel->get_kr();
            ثابت size_t sr = kernel->get_sr();
            ثابت ggml_type rhs_type = kernels->rhs_type;
            אם (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            ثابت size_t block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            אם (block_len == SIZE_MAX) {
                استمر;
            }

            ثابت size_t packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            ثابت size_t aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            אם (rhs_type == GGML_TYPE_Q4_0) {
                بنية kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ثابت uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } אחרת אם (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       data, nullptr, nullptr, dst_ptr, 0, nullptr);
            } אחרת אם (rhs_type == GGML_TYPE_Q8_0) {
                بنية kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } אחרת אם (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
            } אחרת {
                استمر;
            }

            header->offsets[header->slot_count] = aligned_cursor;
            header->sizes[header->slot_count]   = packed_size;
            ++header->slot_count;

            cursor = aligned_cursor + packed_size;
        }

        אם (header->slot_count == 0) {
            header->magic   = 0;
            header->version = 0;
            memcpy(tensor->data, data, data_size);
        }

        החזר 0;
    }
};

ساكن ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, بنية ggml_tensor *) {
    ساكن tensor_traits traits;
    החזר &traits;
}
}  // namespace ggml::cpu::kleidiai

ساكن تعداد ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, بنية ggml_tensor * tensor) {
    tensor->extra = (فراغ *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    החזר GGML_STATUS_SUCCESS;
}

ساكن فراغ ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, بنية ggml_tensor * tensor,
                                                       ثابت فراغ * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

ساكن ثابت حرف * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    החזר "CPU_KLEIDIAI";
}

ساكن ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    אם (buffer == nullptr) {
        החזר nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    החזר buffer;
}

ساكن size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    החזר TENSOR_ALIGNMENT;
}

ساكن size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت بنية ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    אם (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        החזר ggml_nbytes(tensor);
    }

    ثابت size_t n = tensor->ne[1];
    ثابت size_t k = tensor->ne[0];

    size_t cursor = حجم(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ثابت منطقي want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ثابت منطقي want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ثابت منطقي want_f32 = tensor->type == GGML_TYPE_F32;
    ثابت صحيح slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ثابت منطقي allow_fallback = kleidiai_pack_fallback_allowed();

    size_t slot_count = 0;
    עבור (صحيح slot = 0; slot < slot_total; ++slot) {
        אם (!allow_fallback && slot > 0) {
            اكسر;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        אם (!kernels) {
            استمر;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        אם (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            استمر;
        }

        ثابت ggml_type rhs_type = kernels->rhs_type;
        ثابت size_t block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        אם (block_len == SIZE_MAX) {
            استمر;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_count;
    }

    אם (slot_count == 0) {
        החזר ggml_nbytes(tensor);
    }

    החזר std::max(cursor, ggml_nbytes(tensor));
}

نطاق ggml::cpu::kleidiai {
מחלקה extra_buffer_type : ggml::cpu::extra_buffer_type {
    منطقي supports_op(ggml_backend_dev_t, ثابت بنية ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        אם (op->op != GGML_OP_MUL_MAT) {
            החזר שקר;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحيح slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        ثابت منطقي src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft->context == هذا &&
            slot_total > 0;

        אם ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            אם (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                החזר שקר;
            }
            אם (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                החזר שקר;
            }
            אם (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                החזר שקר;
            }
            אם (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                החזר שקר;
            }
            אם (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                החזר שקר;
            }
            אם (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                החזר שקר;
            }

            אם (op->src[0]->type == GGML_TYPE_Q4_K) {
                אם (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    החזר שקר;
                }
                החזר ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == حجم(عائم) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * حجم(عائم));
            }

            אם (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                אם (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    החזר אמת;
                }
                החזר שקר;
            }

            אם (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                החזר שקר;
            }

            החזר אמת;
        }

        החזר שקר;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت بنية ggml_tensor * op) override {
        אם (op->op == GGML_OP_MUL_MAT) {
            אם (op->src[0]->buffer && op->src[0]->buffer->buft->context == هذا) {
                החזר (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } אחרת {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                אם (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    ساكن std::atomic<منطقي> warned(שקר);
                    אם (!warned.exchange(אמת)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                אם (op->src[0]->type != GGML_TYPE_F16) {
                    החזר nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ثابت صحيح slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                אם (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    אם ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        החזר nullptr;
                    }
                    החזר ggml::cpu::kleidiai::get_tensor_traits(عدم, عدم);
                }
            }
        }
        החזר nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(فراغ) {
    ساكن ggml::cpu::kleidiai::extra_buffer_type ctx;
    ساكن بنية ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .get_alloc_size_n     = */ عدم,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    החזר &ggml_backend_cpu_buffer_type_kleidiai;
}
