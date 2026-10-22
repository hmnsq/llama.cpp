// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#شامل <arm_neon.h>
#شامل <cassert>
#شامل <cstdio>
#شامل <cstdlib>
#شامل <atomic>
#شامل <cfloat>
#شامل <cctype>
#شامل <algorithm>
#شامل <cmath>
#شامل <stdexcept>
#شامل <stdint.h>
#شامل <string.h>
#شامل <string>
#شامل <vector>
#شامل <array>
#شامل <cstddef>
#شامل <cstdint>
#شامل <fstream>
#شامل <map>
#شامل <iostream>
#شامل <climits>
#شامل <charconv>
#شامل <system_error>
#اگر defined(__linux__)
#شامل <asm/hwcap.h>
#شامل <dirent.h>
#شامل <sys/auxv.h>
#شامل <sys/types.h>
#شامل <sys/stat.h>
#شامل <unistd.h>
#وگرنهاگر defined(__APPLE__)
#شامل <sys/sysctl.h>
#شامل <sys/types.h>
#endif

#شامل "kleidiai.h"

#شامل "ggml-cpu.h"
#شامل "ggml-cpu-impl.h"
#شامل "ggml-impl.h"
#شامل "ggml-feats.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-quants.h"
#شامل "ggml-threading.h"
#شامل "traits.h"

#شامل "kernels.h"

#شامل "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#شامل "ggml-common.h"

ایستا constexpr صحیح      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
ایستا constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
ایستا constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
ایستا constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
ایستا constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

ساختار ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    صحیح sme_thread_cap; // <= 0 means "SME disabled/unknown"
    صحیح thread_hint;    // <= 0 means "no hint"
    صحیح chunk_multiplier;
} ایستا ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

ایستا درونخطی منطقی is_sme_family(cpu_feature f) {
    بازگشت (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

ایستا ثابت نویسه* cpu_feature_to_string(cpu_feature f) {
    اگر (f == CPU_FEATURE_NONE) {
        بازگشت "NONE";
    } وگرنه اگر ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        بازگشت "SME2";
    } وگرنه اگر ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        بازگشت "SME";
    } وگرنه اگر ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        بازگشت "SVE";
    }
    وگرنه اگر ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        بازگشت "I8MM";
    } وگرنه اگر ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        بازگشت "DOTPROD";
    }
    وگرنه {
        بازگشت "UNKNOWN";
    }
}

#اگر defined(__linux__) && defined(__aarch64__)
ایستا منطقی parse_cpu_dir_name(ثابت نویسه* name, size_t* cpu) {
    اگر (strncmp(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        بازگشت نادرست;
    }

    ثابت نویسه* first = name + 3;
    ثابت نویسه* last = name + strlen(name);

    size_t value = 0;
    ثابت auto [end, ec] = std::from_chars(first, last, value, 10);

    اگر (ec != std::errc{} || end != last) {
        بازگشت نادرست;
    }

    *cpu = value;
    بازگشت درست;
}

ایستا std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    اگر (dir == nullptr) {
        بازگشت cpus;
    }

    وقتی (dirent * entry = readdir(dir)) {
        size_t cpu = 0;
        اگر (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    بازگشت cpus;
}
#endif

#اگر defined(__APPLE__) && defined(__aarch64__)
ایستا منطقی apple_sme_counted_perf_level(std::string name) {
    برای (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (نویسه) std::tolower((unsigned نویسه) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    بازگشت name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

ایستا تهی add_smcus_from_smidr(uint64_t smidr, size_t & رقم_private, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ثابت uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ثابت uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ثابت size_t shared_شمار = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ثابت uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ثابت uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ثابت uint32_t id = (affinity2 << 12) | affinity;

    اگر (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    انتخاب (sh) {
        حالت 2: // private SMCU
            ++رقم_private;
            بشکن;
        حالت 3: // shared SMCU
            اگر (shared_counts[id] < shared_شمار) {
                shared_counts[id] = shared_شمار;
            }
            بشکن;
        حالت 0:
            اگر (id == 0) {
                ++رقم_private;
            } وگرنه اگر (shared_counts[id] < shared_شمار) {
                shared_counts[id] = shared_شمار;
            }
            بشکن;
        پیشفرض:
            بشکن;
    }
}

ایستا size_t detect_رقم_smcus() {
    ثابت auto runtime_feat = ggml_feats_get_arch64_runtime();
    اگر (!runtime_feat.has_sme) {
        بازگشت 0;
    }

#اگر defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t رقم_private = 0;
    std::map<uint32_t, size_t> shared_counts;

    ثابت std::vector<size_t> cpus = detect_cpu_ids();
    برای (ثابت size_t cpu : cpus) {
        ثابت std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        اگر (!file.is_open()) {
            ادامه;
        }

        uint64_t smidr = 0;
        اگر (!(file >> std::hex >> smidr)) {
            ادامه;
        }

        add_smcus_from_smidr(smidr, رقم_private, shared_counts);
    }

    size_t total = رقم_private;
    برای (ثابت auto & entry : shared_counts) {
        total += entry.second;
    }
    بازگشت total;

#وگرنهاگر defined(__APPLE__) && defined(__aarch64__)
    صحیح perf_levels = 0;
    size_t size = اندازه(perf_levels);
    اگر (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != اندازه(perf_levels) || perf_levels <= 0) {
        بازگشت 0;
    }

    size_t units = 0;
    برای (صحیح i = 0; i < perf_levels; ++i) {
        نویسه کلید[64] = {};
        صحیح physical_cpus = 0;
        صحیح cpus_per_l2 = 0;

        snprintf(کلید, اندازه(کلید), "hw.perflevel%d.physicalcpu", i);
        size = اندازه(physical_cpus);
        اگر (sysctlbyname(کلید, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != اندازه(physical_cpus) || physical_cpus <= 0) {
            ادامه;
        }

        snprintf(کلید, اندازه(کلید), "hw.perflevel%d.cpusperl2", i);
        size = اندازه(cpus_per_l2);
        اگر (sysctlbyname(کلید, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != اندازه(cpus_per_l2) || cpus_per_l2 <= 0) {
            ادامه;
        }

        snprintf(کلید, اندازه(کلید), "hw.perflevel%d.name", i);
        size = 0;
        اگر (sysctlbyname(کلید, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            ادامه;
        }

        std::string name(size, '\0');
        اگر (sysctlbyname(کلید, &name[0], &size, nullptr, 0) != 0) {
            ادامه;
        }
        name.resize(size);
        وقتی (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        اگر (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    بازگشت units;

#وگرنهاگر defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    بازگشت 0;

#وگرنه
    بازگشت 0;
#endif
}

ایستا صحیح parse_uint_env(ثابت نویسه *s, ثابت نویسه *name, منطقی *ok) {
    اگر (!s) { *ok = نادرست; بازگشت 0; }
    نویسه *end = nullptr;
    long v = strtol(s, &end, 10);
    اگر (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = نادرست;
        بازگشت 0;
    }
    اگر (v < 0 || v > INT_MAX) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = نادرست;
        بازگشت 0;
    }
    *ok = درست;
    بازگشت (صحیح)v;
}

ایستا تهی init_kleidiai_context(تهی) {
    ggml_critical_section_start();
    ایستا منطقی initialized = نادرست;

    اگر (!initialized) {
        initialized = درست;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ثابت نویسه *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ثابت نویسه *env_threads     = getenv("GGML_TOTAL_THREADS");
        ثابت نویسه *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ثابت auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        اگر (env_threads) {
            منطقی ok = نادرست;
            صحیح hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            اگر (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        اگر (env_chunk_mult) {
            منطقی ok = نادرست;
            صحیح multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            اگر (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        صحیح sme_cores = 0;
        منطقی sme_env_ok = نادرست;
        منطقی sme_env_set = (env_sme != nullptr);

        ثابت منطقی has_supported_sme_family = runtime_feat.has_sme;
        منطقی sme_cap_detected = نادرست;

        اگر (has_supported_sme_family) {
            detected_smcus = detect_رقم_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (صحیح)detected_smcus : 1;

            اگر (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        اگر (sme_env_set) {
            منطقی ok = نادرست;
            صحیح v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            اگر (ok) {
                اگر (has_supported_sme_family) {
                    sme_cores = v;
                } وگرنه {
                    اگر (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } وگرنه {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        اگر (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            اگر (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        اگر (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (صحیح)ctx.features);
        } وگرنه {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        اگر (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (صحیح)ctx.features);
        } وگرنه {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        اگر (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (صحیح) ctx.features);
        } وگرنه {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        اگر (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (صحیح)ctx.features);
        } وگرنه {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        ثابت منطقی has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        اگر (has_selected_sme_family_kernel) {
            اگر (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } وگرنه اگر (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } وگرنه {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } وگرنه {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_end();
}

ایستا درونخطی صحیح kleidiai_sme_thread_cap() {
    بازگشت ctx.sme_thread_cap;
}

ایستا درونخطی size_t align_up(size_t value, size_t alignment) {
    اگر (alignment == 0) {
        بازگشت value;
    }
    ثابت size_t remainder = value % alignment;
    بازگشت remainder == 0 ? value : value + (alignment - remainder);
}

ایستا درونخطی size_t gcd_size(size_t a, size_t b) {
    وقتی (b != 0) {
        ثابت size_t t = a % b;
        a = b;
        b = t;
    }
    بازگشت a;
}

ایستا درونخطی منطقی lcm_size(size_t a, size_t b, size_t & result) {
    اگر (a == 0 || b == 0) {
        result = 0;
        بازگشت نادرست;
    }
    ثابت size_t g = gcd_size(a, b);
    ثابت size_t q = a / g;
    اگر (q > SIZE_MAX / b) {
        بازگشت نادرست;
    }
    result = q * b;
    بازگشت درست;
}

ایستا درونخطی size_t ceil_div_size(size_t a, size_t b) {
    بازگشت b == 0 ? 0 : (a + b - 1) / b;
}

ایستا درونخطی size_t kleidiai_chunk_cols(size_t n, صحیح nth_total, منطقی disable_chunking, size_t n_step) {
    ثابت size_t multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ثابت size_t مقسوم_علیه = std::max<size_t>(1, (size_t) nth_total * multiplier);
    ثابت size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, مقسوم_علیه)), n_step);
    بازگشت chunk_cols ? chunk_cols : n_step;
}

ساختار kleidiai_block_args {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

ایستا درونخطی kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    انتخاب (rhs_type) {
        حالت GGML_TYPE_Q4_0:
            بازگشت { QK4_0, QK4_0, QK4_0 };
        حالت GGML_TYPE_Q4_K:
            بازگشت { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        حالت GGML_TYPE_Q8_0:
            بازگشت { 0, 0, QK8_0 };
        پیشفرض:
            بازگشت { 0, 0, 0 };
    }
}

ایستا درونخطی منطقی kleidiai_pack_fallback_allowed() {
    اگر (ctx.sme_thread_cap <= 0) {
        بازگشت نادرست;
    }
    اگر (ctx.thread_hint <= 0) {
        بازگشت درست;
    }
    بازگشت ctx.thread_hint > ctx.sme_thread_cap;
}

ساختار kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_شمار;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

ایستا درونخطی kleidiai_weight_header * kleidiai_weight_header_from_ptr(تهی * data) {
    بازگشت reinterpret_cast<kleidiai_weight_header *>(data);
}

ایستا درونخطی ثابت kleidiai_weight_header * kleidiai_weight_header_from_ptr(ثابت تهی * data) {
    بازگشت reinterpret_cast<ثابت kleidiai_weight_header *>(data);
}

ایستا درونخطی منطقی kleidiai_is_weight_header_valid(ثابت kleidiai_weight_header * header) {
    اگر (!header) {
        بازگشت نادرست;
    }
    اگر (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        بازگشت نادرست;
    }
    اگر (header->slot_شمار == 0 || header->slot_شمار > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        بازگشت نادرست;
    }
    بازگشت درست;
}

ایستا درونخطی uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, صحیح slot) {
    اگر (!kleidiai_is_weight_header_valid(header)) {
        بازگشت nullptr;
    }
    اگر (slot < 0 || slot >= header->slot_شمار) {
        بازگشت nullptr;
    }
    بازگشت reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

ایستا درونخطی ثابت uint8_t * kleidiai_weight_slot_ptr(ثابت kleidiai_weight_header * header, صحیح slot) {
    اگر (!kleidiai_is_weight_header_valid(header)) {
        بازگشت nullptr;
    }
    اگر (slot < 0 || slot >= header->slot_شمار) {
        بازگشت nullptr;
    }
    بازگشت reinterpret_cast<ثابت uint8_t *>(header) + header->offsets[slot];
}

ایستا درونخطی ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    بازگشت ctx.kernels_q4;
}

ایستا درونخطی ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    بازگشت ctx.kernels_q4k;
}

ایستا درونخطی ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    بازگشت ctx.kernels_q8;
}

ایستا درونخطی ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    بازگشت ctx.kernels_f32;
}

template <typename SelectFallback>
ایستا صحیح kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    صحیح count = 0;
    اگر (!primary) {
        بازگشت 0;
    }
    out[count++] = primary;

    اگر (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        بازگشت count;
    }

    اگر (is_sme_family(primary->required_cpu)) {
        ثابت cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        اگر (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            اگر (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    بازگشت count;
}

ایستا صحیح kleidiai_collect_kernel_chain(ثابت ساختار ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    بازگشت kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { بازگشت ggml_kleidiai_select_kernels(mask, op); });
}

ایستا صحیح kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    بازگشت kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { بازگشت ggml_kleidiai_select_kernels_q4_0(mask); });
}

ایستا صحیح kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    بازگشت kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { بازگشت ggml_kleidiai_select_kernels_q4_k(mask); });
}

ایستا صحیح kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    بازگشت kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { بازگشت ggml_kleidiai_select_kernels_q8_0(mask); });
}

ایستا صحیح kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    بازگشت kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { بازگشت ggml_kleidiai_select_kernels_f32(mask); });
}

ایستا درونخطی int64_t ggml_ne(ثابت ggml_tensor * tensor, صحیح dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    بازگشت tensor->ne[dim];
}

فضاینام ggml::cpu::kleidiai {

ایستا size_t round_down(size_t x, size_t y) {
    بازگشت y == 0 ? x : x - (x % y);
}

ایستا تهی transpose_f32kxn_f16nxk(size_t n, size_t k, اعشاری * dst, ثابت uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / اندازه(uint16_t);
    size_t dst_stride = n;

    برای (size_t k_idx = 0; k_idx < k; ++k_idx) {
        برای (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

رده tensor_traits : عمومی ggml::cpu::tensor_traits {
    منطقی work_size(صحیح /* n_threads */, ثابت ساختار ggml_tensor * op, size_t & size) override {
        اگر (op->op != GGML_OP_MUL_MAT) {
            بازگشت نادرست;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحیح slot_شمار = kleidiai_collect_kernel_chain(op, kernel_chain);
        اگر (slot_شمار == 0) {
            بازگشت نادرست;
        }

        ثابت منطقی is_gemv = op->src[1]->ne[1] == 1;
        ثابت size_t k = op->src[0]->ne[0];
        ثابت size_t n = op->src[0]->ne[1];
        ثابت size_t m = op->src[1]->ne[1];

        اگر (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ثابت kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            ثابت size_t qk = block_args.pack_bl;

            size_t cursor = 0;
            منطقی any_slot = نادرست;

            برای (صحیح slot = 0; slot < slot_شمار; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                اگر (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    بازگشت نادرست;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                ثابت size_t packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = درست;
            }

            اگر (!any_slot) {
                بازگشت نادرست;
            }

            size = cursor;
            بازگشت درست;
        }

        اگر (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            اگر (is_gemv && op->src[1]->nb[0] == (int64_t) اندازه(اعشاری) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                بازگشت درست;
            }

            size_t cursor = 0;
            منطقی any_slot = نادرست;

            برای (صحیح slot = 0; slot < slot_شمار; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                اگر (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    بازگشت نادرست;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
                any_slot = درست;
            }

            اگر (!any_slot) {
                بازگشت نادرست;
            }

            size = cursor;
            بازگشت درست;
        }

        اگر (op->src[0]->type == GGML_TYPE_F16) {
            ثابت int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ثابت int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ثابت int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            منطقی any_slot = نادرست;

            برای (صحیح slot = 0; slot < slot_شمار; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                اگر (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    بازگشت نادرست;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = درست;
            }

            برای (صحیح slot = 0; slot < slot_شمار; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                اگر (!kernel || !kernels->rhs_info.packed_size_ex) {
                    بازگشت نادرست;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * اندازه(اعشاری);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * اندازه(اعشاری);

            اگر (!any_slot) {
                بازگشت نادرست;
            }

            size = cursor;
            بازگشت درست;
        }

        بازگشت نادرست;
    }

    منطقی compute_forward(ساختار ggml_compute_params * params, ساختار ggml_tensor * dst) override {
        اگر (dst->op == GGML_OP_MUL_MAT) {
            اگر (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                بازگشت compute_forward_qx(params, dst);
            } وگرنه اگر (dst->src[0]->type == GGML_TYPE_F32) {
                بازگشت compute_forward_f32(params, dst);
            } وگرنه اگر (dst->src[0]->type == GGML_TYPE_F16) {
                بازگشت compute_forward_fp16(params, dst);
            }
        }
        بازگشت نادرست;
    }

    منطقی compute_forward_f32(ggml_compute_params * params, ساختار ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_دودویی_OP_LOCALS

        اگر (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        اگر (!kernels) {
            بازگشت نادرست;
        }

        ثابت size_t k = ne00;
        ثابت size_t m = ne11;
        ثابت size_t n = ne01;
        ثابت منطقی use_gemv = m == 1 && src1->nb[0] == (int64_t) اندازه(اعشاری) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        اگر (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            بازگشت نادرست;
        }

        اگر (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_size_ex || !lhs_info->pack_func_ex)) {
            بازگشت نادرست;
        }

        ثابت kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ثابت منطقی has_header = kleidiai_is_weight_header_valid(header);

        ثابت uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<ثابت uint8_t *>(src0->data);
        اگر (!rhs_base) {
            بازگشت نادرست;
        }

        ثابت صحیح nth = params->nth > 0 ? params->nth : 1;
        ثابت صحیح ith = params->ith;

        ثابت size_t mr = kernel->get_mr();
        ثابت size_t kr = kernel->get_kr();
        ثابت size_t sr = kernel->get_sr();

        ثابت size_t lhs_packed_size = use_gemv ? 0 : lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        اگر (!use_gemv) {
            GGML_ASSERT(lhs_packed_size <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ثابت size_t dst_stride = dst->nb[1];
        ثابت size_t n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        ثابت منطقی disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) INT_MAX);

        ثابت int64_t batch_size = ne12 * ne13;
        برای (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ثابت int64_t i12 = batch_idx % ne12;
            ثابت int64_t i13 = batch_idx / ne12;
            ثابت uint8_t * lhs_batch_base = static_cast<ثابت uint8_t *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];

            اگر (!use_gemv) {
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                ثابت int64_t use_threads = std::min<int64_t>(nth, max_threads);

                اگر (ith < use_threads) {
                    ثابت int64_t رقم_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), mr);
                    ثابت int64_t رقم_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * رقم_m_per_thread0;

                    ثابت int64_t m_start = (int64_t)ith * رقم_m_per_thread0;
                    ثابت int64_t m_شمار = (ith == use_threads - 1) ? رقم_m_per_thread_N_1 : رقم_m_per_thread0;

                    ثابت size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ثابت size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ثابت size_t row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_شمار;
                    int64_t cur       = m_start;

                    وقتی (remaining > 0) {
                        ثابت int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        ثابت size_t src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        ثابت تهی * src_ptr = lhs_batch_base + src_off;
                        ثابت size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        تهی * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            اگر (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ثابت size_t chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (size_t) INT_MAX);

            صحیح current_col = ggml_threadpool_chunk_add(params->threadpool, (صحیح) chunk_cols);
            وقتی ((size_t) current_col < n) {
                ثابت size_t n_start = (size_t) current_col;
                ثابت size_t n_to_process = std::min(chunk_cols, n - n_start);

                اگر (n_to_process > 0) {
                    ثابت size_t lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ثابت size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    ثابت تهی * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    ثابت تهی * rhs_ptr = rhs_base + rhs_packed_offset;
                    اعشاری * dst_ptr      = reinterpret_cast<اعشاری *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          اندازه(اعشاری),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (صحیح) chunk_cols);
            }

            اگر (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        بازگشت درست;
    }

    منطقی compute_forward_fp16(ggml_compute_params * params, ساختار ggml_tensor * dst) {
        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_دودویی_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        اگر (!kernels) {
            بازگشت نادرست;
        }

        ثابت منطقی is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        اگر (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            بازگشت نادرست;
        }

        ثابت صحیح nth = params->nth;
        ثابت صحیح ith = params->ith;

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
        ثابت size_t kxn_size        = k * n * اندازه(اعشاری);
        ثابت size_t bias_size       = n * اندازه(اعشاری);

        ثابت size_t wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        برای (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ثابت int64_t rhs_batch_idx = batch_idx / r;
            ثابت uint8_t * rhs_batch_base = static_cast<ثابت uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ثابت int64_t m_roundup_mr = kai_roundup(m, mr);
                ثابت int64_t رقم_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                اگر (ith < رقم_threads) {
                    ثابت int64_t رقم_m_per_thread0   = round_down((size_t)(m_roundup_mr / رقم_threads), (size_t)mr);
                    ثابت int64_t رقم_m_per_thread_N_1 = m - (رقم_threads - 1) * رقم_m_per_thread0;

                    ثابت int64_t m_start = ith * رقم_m_per_thread0;
                    ثابت int64_t m_شمار = (ith == رقم_threads - 1) ? رقم_m_per_thread_N_1 : رقم_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ثابت size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ثابت size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ثابت size_t row_stride_bytes = (next_block_off - base_packed_off) / (size_t)mr;

                    int64_t remaining = m_شمار;
                    int64_t cur       = m_start;

                    وقتی (remaining > 0) {
                        ثابت int64_t row_in_group = cur;
                        ثابت int64_t avail        = m_group - row_in_group;
                        ثابت int64_t take         = std::min(avail, remaining);

                        ثابت uint8_t * lhs_batch_base = static_cast<ثابت uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        ثابت تهی * src_ptr = lhs_batch_base + (size_t)row_in_group * lhs_stride;
                        ثابت size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        تهی * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            اگر (ith == 0) {
                memset(bias, 0, (size_t)n * اندازه(اعشاری));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<اعشاری *>(rhs_kxn),
                                        reinterpret_cast<ثابت uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * اندازه(اعشاری),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ثابت int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t رقم_threads_n = KAI_MIN(n / n_step, nth);
                اگر (رقم_threads_n <= 0) {
                    رقم_threads_n = 1;
                }

                اگر (ith < رقم_threads_n) {
                    ثابت int64_t رقم_n_per_thread0   = round_down((size_t)(n / رقم_threads_n), (size_t)n_step);
                    ثابت int64_t رقم_n_per_thread_N_1 = n - (رقم_threads_n - 1) * رقم_n_per_thread0;

                    ثابت int64_t n_start      = ith * رقم_n_per_thread0;
                    ثابت int64_t n_to_process = (ith == رقم_threads_n - 1) ? رقم_n_per_thread_N_1 : رقم_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ثابت size_t lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ثابت size_t dst_offset         = kernel->get_dst_offset((size_t)0, (size_t)n_start, dst_stride);

                    ثابت تهی * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ثابت تهی * rhs_ptr = rhs_packed + rhs_packed_offset;
                    اعشاری * dst_ptr      = reinterpret_cast<اعشاری *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, اندازه(اعشاری), -FLT_MAX, FLT_MAX);
                }
            }

            اگر (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        بازگشت درست;
    }

    منطقی compute_forward_qx(ساختار ggml_compute_params * params, ساختار ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_دودویی_OP_LOCALS

        ثابت kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ثابت منطقی has_header = kleidiai_is_weight_header_valid(header);
        ثابت منطقی is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحیح slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        auto weight_for_slot = [&](صحیح slot_index, size_t & size_out) -> ثابت uint8_t * {
            اگر (slot_index < 0 || slot_index >= slot_total) {
                بازگشت nullptr;
            }
            اگر (has_header) {
                اگر (slot_index < header->slot_شمار) {
                    size_out = static_cast<size_t>(header->sizes[slot_index]);
                    بازگشت kleidiai_weight_slot_ptr(header, slot_index);
                }
                بازگشت nullptr;
            }
            اگر (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                بازگشت static_cast<ثابت uint8_t *>(src0->data);
            }
            بازگشت nullptr;
        };

        ساختار runtime_slot {
            صحیح slot_index;
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
            صحیح assigned_threads;
            صحیح thread_begin;
            صحیح thread_end;
            ثابت uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        صحیح runtime_شمار = 0;

        برای (صحیح slot = 0; slot < slot_total && runtime_شمار < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            اگر (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                ادامه;
            }

            size_t rhs_size = 0;
            ثابت uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            اگر (!rhs_ptr || rhs_size == 0) {
                ادامه;
            }

            ثابت kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

            runtime[runtime_شمار] = {
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
            ++runtime_شمار;
        }

        اگر (runtime_شمار == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            بازگشت نادرست;
        }

        ثابت صحیح nth_total = params->nth > 0 ? params->nth : 1;
        ثابت صحیح ith_total = params->ith;

        صحیح sme_slot = -1;
        صحیح non_sme_slot = -1;
        برای (صحیح i = 0; i < runtime_شمار; ++i) {
            اگر (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                بشکن;
            }
        }

        برای (صحیح i = 0; i < runtime_شمار; ++i) {
            اگر (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                بشکن;
            }
        }

        ثابت صحیح sme_cap_limit = ctx.sme_thread_cap;
        ثابت منطقی use_hybrid = sme_cap_limit > 0 &&
                                 runtime_شمار > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_cols_per_thread = 0;
        اگر (runtime_شمار > 0 && nth_total > 0) {
            min_cols_per_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        ثابت منطقی too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        ثابت منطقی hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        اگر (!hybrid_enabled) {
            صحیح chosen_slot = 0;
            اگر (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } وگرنه اگر (runtime_شمار > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            اگر (chosen_slot != 0 && chosen_slot < runtime_شمار) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_شمار = runtime_شمار > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            اگر (runtime_شمار > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        ثابت منطقی sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        صحیح sme_cap = sme_single_only ? nth_total : kleidiai_sme_thread_cap();
        اگر (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        صحیح threads_remaining = nth_total;
        اگر (sme_slot != -1) {
            صحیح sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        صحیح fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        صحیح fallback_شمار = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        برای (صحیح i = 0; i < runtime_شمار; ++i) {
            اگر (i == sme_slot) {
                ادامه;
            }
            fallback_indices[fallback_شمار++] = i;
        }

        برای (صحیح fi = 0; fi < fallback_شمار; ++fi) {
            اگر (threads_remaining <= 0) {
                بشکن;
            }
            ثابت صحیح slot_index = fallback_indices[fi];
            ثابت صحیح slots_چپ = fallback_شمار - fi;
            صحیح share = (threads_remaining + slots_چپ - 1) / slots_چپ;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        صحیح thread_cursor = 0;
        برای (صحیح i = 0; i < runtime_شمار; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ثابت صحیح active_threads = thread_cursor;

        صحیح local_slot = -1;
        صحیح local_ith  = 0;
        برای (صحیح i = 0; i < runtime_شمار; ++i) {
            اگر (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                بشکن;
            }
        }
        ثابت size_t k = ne00;
        ثابت size_t m = ne11;
        ثابت size_t n = ne01;

        size_t cursor = 0;
        برای (صحیح i = 0; i < runtime_شمار; ++i) {
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_step = 1;
        برای (صحیح i = 0; i < runtime_شمار; ++i) {
            اگر (runtime[i].assigned_threads == 0) {
                ادامه;
            }
            size_t next_step = 0;
            اگر (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                بازگشت نادرست;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        ثابت منطقی disable_chunking = ggml_is_numa();
        ثابت size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ثابت size_t chunk_مقسوم_علیه = (active_threads == 1 || disable_chunking) ? (size_t) active_threads : (size_t) active_threads * chunk_multiplier;
        size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, chunk_مقسوم_علیه)), common_step);
        اگر (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ثابت size_t nchunk_size = std::max<size_t>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (size_t)INT_MAX);
        ثابت صحیح nchunk = (صحیح)nchunk_size;
        ثابت size_t dst_stride = dst->nb[1];

        auto run_chunk = [&](runtime_slot & slot, size_t global_start, size_t cols, uint8_t * dst_batch_base) {
            ثابت size_t rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            ثابت size_t dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            ثابت uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            ثابت uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            اعشاری * dst_ptr         = reinterpret_cast<اعشاری *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       اندازه(اعشاری),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        برای (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ثابت uint8_t * lhs_batch_base = static_cast<ثابت uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            اگر (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                ثابت int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                اگر (local_ith < use_threads) {
                    ثابت int64_t رقم_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), slot.mr);
                    ثابت int64_t رقم_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * رقم_m_per_thread0;

                    ثابت int64_t m_start = (int64_t)local_ith * رقم_m_per_thread0;
                    ثابت int64_t m_شمار = (local_ith == use_threads - 1) ? رقم_m_per_thread_N_1 : رقم_m_per_thread0;

                    ثابت size_t base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ثابت size_t next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ثابت size_t row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_شمار;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    وقتی (remaining > 0) {
                        ثابت int64_t row_in_group = cur;
                        ثابت int64_t avail        = (int64_t)m - row_in_group;
                        ثابت int64_t take         = std::min(avail, remaining);

                        ثابت size_t src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        ثابت تهی * src_ptr = lhs_batch_base + src_off;
                        ثابت size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        تهی * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            اگر (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            اگر (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                صحیح current_chunk = ith_total;
                وقتی (current_chunk < nchunk) {
                    ثابت size_t global_start = (size_t)current_chunk * chunk_cols;
                    اگر (global_start >= n) {
                        بشکن;
                    }

                    ثابت size_t cols = std::min(chunk_cols, n - global_start);
                    اگر (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            اگر (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        بازگشت درست;
    }

عمومی:
    صحیح repack(ساختار ggml_tensor * tensor, ثابت تهی * data, size_t data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ثابت size_t n = tensor->ne[1];
        ثابت size_t k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        اگر (!header) {
            بازگشت -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_شمار = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        size_t cursor = اندازه(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت منطقی want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ثابت منطقی want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ثابت منطقی want_f32 = tensor->type == GGML_TYPE_F32;
        ثابت صحیح slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ثابت منطقی allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<اعشاری>  scales;
        std::vector<اعشاری>  bias;
        std::vector<اعشاری> row_f32;

        اگر (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ثابت size_t row_stride = tensor->nb[1];
            ثابت size_t k_blocks   = (k + QK8_0 - 1) / QK8_0;

            اگر (tensor->type == GGML_TYPE_Q6_K) {
                اگر (k % QK_K != 0 || k % QK8_0 != 0) {
                    بازگشت -1;
                }
                row_f32.resize(k);
            }

            برای (size_t row = 0; row < n; ++row) {
                ثابت uint8_t * row_data = static_cast<ثابت uint8_t *>(data) + row * row_stride;

                اگر (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<ثابت block_q6_K *>(row_data), row_f32.data(), k);
                }

                اعشاری max_abs = 0.0f;
                اگر (tensor->type == GGML_TYPE_Q6_K) {
                    برای (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::fabs(row_f32[linear_idx]));
                    }
                } وگرنه {
                    ثابت block_q8_0 * row_blocks = reinterpret_cast<ثابت block_q8_0 *>(row_data);
                    برای (size_t block = 0; block < k_blocks; ++block) {
                        ثابت block_q8_0 & blk = row_blocks[block];
                        ثابت اعشاری d = GGML_FP16_TO_FP32(blk.d);
                        برای (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t linear_idx = block * QK8_0 + l;
                            اگر (linear_idx >= k) {
                                بشکن;
                            }
                            ثابت اعشاری value = d * static_cast<اعشاری>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                اعشاری scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                ثابت اعشاری inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                اگر (tensor->type == GGML_TYPE_Q6_K) {
                    برای (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } وگرنه {
                    ثابت block_q8_0 * row_blocks = reinterpret_cast<ثابت block_q8_0 *>(row_data);
                    برای (size_t block = 0; block < k_blocks; ++block) {
                        ثابت block_q8_0 & blk = row_blocks[block];
                        ثابت اعشاری d = GGML_FP16_TO_FP32(blk.d);
                        برای (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t linear_idx = block * QK8_0 + l;
                            اگر (linear_idx >= k) {
                                بشکن;
                            }
                            ثابت اعشاری value = d * static_cast<اعشاری>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        اگر (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
        }

        برای (صحیح slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            اگر (!allow_fallback && slot > 0) {
                بشکن;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            اگر (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                ادامه;
            }

            ثابت size_t nr = kernel->get_nr();
            ثابت size_t kr = kernel->get_kr();
            ثابت size_t sr = kernel->get_sr();
            ثابت ggml_type rhs_type = kernels->rhs_type;
            اگر (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            ثابت size_t block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            اگر (block_len == SIZE_MAX) {
                ادامه;
            }

            ثابت size_t packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            ثابت size_t aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            اگر (rhs_type == GGML_TYPE_Q4_0) {
                ساختار kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ثابت uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } وگرنه اگر (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       data, nullptr, nullptr, dst_ptr, 0, nullptr);
            } وگرنه اگر (rhs_type == GGML_TYPE_Q8_0) {
                ساختار kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } وگرنه اگر (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
            } وگرنه {
                ادامه;
            }

            header->offsets[header->slot_شمار] = aligned_cursor;
            header->sizes[header->slot_شمار]   = packed_size;
            ++header->slot_شمار;

            cursor = aligned_cursor + packed_size;
        }

        اگر (header->slot_شمار == 0) {
            header->magic   = 0;
            header->version = 0;
            memcpy(tensor->data, data, data_size);
        }

        بازگشت 0;
    }
};

ایستا ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, ساختار ggml_tensor *) {
    ایستا tensor_traits traits;
    بازگشت &traits;
}
}  // namespace ggml::cpu::kleidiai

ایستا شمارش ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, ساختار ggml_tensor * tensor) {
    tensor->extra = (تهی *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    بازگشت GGML_STATUS_SUCCESS;
}

ایستا تهی ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, ساختار ggml_tensor * tensor,
                                                       ثابت تهی * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

ایستا ثابت نویسه * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    بازگشت "CPU_KLEIDIAI";
}

ایستا ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    اگر (buffer == nullptr) {
        بازگشت nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    بازگشت buffer;
}

ایستا size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    بازگشت TENSOR_ALIGNMENT;
}

ایستا size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ساختار ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    اگر (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        بازگشت ggml_nbytes(tensor);
    }

    ثابت size_t n = tensor->ne[1];
    ثابت size_t k = tensor->ne[0];

    size_t cursor = اندازه(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ثابت منطقی want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ثابت منطقی want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ثابت منطقی want_f32 = tensor->type == GGML_TYPE_F32;
    ثابت صحیح slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ثابت منطقی allow_fallback = kleidiai_pack_fallback_allowed();

    size_t slot_شمار = 0;
    برای (صحیح slot = 0; slot < slot_total; ++slot) {
        اگر (!allow_fallback && slot > 0) {
            بشکن;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        اگر (!kernels) {
            ادامه;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        اگر (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            ادامه;
        }

        ثابت ggml_type rhs_type = kernels->rhs_type;
        ثابت size_t block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        اگر (block_len == SIZE_MAX) {
            ادامه;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_شمار;
    }

    اگر (slot_شمار == 0) {
        بازگشت ggml_nbytes(tensor);
    }

    بازگشت std::max(cursor, ggml_nbytes(tensor));
}

فضاینام ggml::cpu::kleidiai {
رده extra_buffer_type : ggml::cpu::extra_buffer_type {
    منطقی supports_op(ggml_backend_dev_t, ثابت ساختار ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        اگر (op->op != GGML_OP_MUL_MAT) {
            بازگشت نادرست;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحیح slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        ثابت منطقی src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft->context == این &&
            slot_total > 0;

        اگر ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            اگر (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                بازگشت نادرست;
            }
            اگر (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                بازگشت نادرست;
            }
            اگر (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                بازگشت نادرست;
            }
            اگر (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                بازگشت نادرست;
            }
            اگر (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                بازگشت نادرست;
            }
            اگر (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                بازگشت نادرست;
            }

            اگر (op->src[0]->type == GGML_TYPE_Q4_K) {
                اگر (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    بازگشت نادرست;
                }
                بازگشت ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == اندازه(اعشاری) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * اندازه(اعشاری));
            }

            اگر (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                اگر (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    بازگشت درست;
                }
                بازگشت نادرست;
            }

            اگر (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                بازگشت نادرست;
            }

            بازگشت درست;
        }

        بازگشت نادرست;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت ساختار ggml_tensor * op) override {
        اگر (op->op == GGML_OP_MUL_MAT) {
            اگر (op->src[0]->buffer && op->src[0]->buffer->buft->context == این) {
                بازگشت (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } وگرنه {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                اگر (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    ایستا std::atomic<منطقی> warned(نادرست);
                    اگر (!warned.exchange(درست)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                اگر (op->src[0]->type != GGML_TYPE_F16) {
                    بازگشت nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ثابت صحیح slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                اگر (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    اگر ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        بازگشت nullptr;
                    }
                    بازگشت ggml::cpu::kleidiai::get_tensor_traits(پوچ, پوچ);
                }
            }
        }
        بازگشت nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(تهی) {
    ایستا ggml::cpu::kleidiai::extra_buffer_type ctx;
    ایستا ساختار ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .get_alloc_size_n     = */ پوچ,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    بازگشت &ggml_backend_cpu_buffer_type_kleidiai;
}
