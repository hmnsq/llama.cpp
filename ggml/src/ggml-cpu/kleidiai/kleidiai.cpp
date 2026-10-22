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
#وگرنہ_اگر defined(__APPLE__)
#شامل <sys/sysctl.h>
#شامل <sys/types.h>
#اگر_ختم

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

#تعریف_کرو GGML_COMMON_DECL_CPP
#شامل "ggml-common.h"

ساکن constexpr صحیح      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
ساکن constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
ساکن constexpr uint16_t GGML_KLEIDIAI_PACK_نسخہ     = 1;
ساکن constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
ساکن constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

ساخت ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    صحیح sme_thread_cap; // <= 0 means "SME disabled/unknown"
    صحیح thread_hint;    // <= 0 means "no hint"
    صحیح chunk_multiplier;
} ساکن ctx = { CPU_FEATURE_NONE, خالی_اشاریہ, خالی_اشاریہ, خالی_اشاریہ, خالی_اشاریہ, 0, -1, 4 };

ساکن اندرونی منطقی is_sme_family(cpu_feature f) {
    واپس (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

ساکن ثابت حرف* cpu_feature_مقصد_string(cpu_feature f) {
    اگر (f == CPU_FEATURE_NONE) {
        واپس "NONE";
    } ورنہ اگر ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        واپس "SME2";
    } ورنہ اگر ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        واپس "SME";
    } ورنہ اگر ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        واپس "SVE";
    }
    ورنہ اگر ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        واپس "I8MM";
    } ورنہ اگر ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        واپس "DOTPROD";
    }
    ورنہ {
        واپس "UNKNOWN";
    }
}

#اگر defined(__linux__) && defined(__aarch64__)
ساکن منطقی parse_cpu_dir_name(ثابت حرف* name, size_t* cpu) {
    اگر (لڑی_موازنہ(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        واپس جھوٹ;
    }

    ثابت حرف* پہلا = name + 3;
    ثابت حرف* آخری = name + لڑی_لمبائی(name);

    size_t value = 0;
    ثابت خودکار [end, ec] = std::from_chars(پہلا, آخری, value, 10);

    اگر (ec != std::errc{} || end != آخری) {
        واپس جھوٹ;
    }

    *cpu = value;
    واپس سچ;
}

ساکن std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    اگر (dir == خالی_اشاریہ) {
        واپس cpus;
    }

    جب (dirent * اندراج_2 = readdir(dir)) {
        size_t cpu = 0;
        اگر (parse_cpu_dir_name(اندراج_2->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    واپس cpus;
}
#اگر_ختم

#اگر defined(__APPLE__) && defined(__aarch64__)
ساکن منطقی apple_sme_counted_perf_level(std::string name) {
    برائے (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (حرف) std::tolower((غیرعلامتی حرف) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    واپس name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#اگر_ختم

ساکن خالی جوڑو_smcus_from_smidr(uint64_t smidr, size_t & عدد_نجی, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ثابت uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ثابت uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ثابت size_t shared_گنتی = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ثابت uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ثابت uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ثابت uint32_t id = (affinity2 << 12) | affinity;

    اگر (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    انتخاب (sh) {
        حالت 2: // private SMCU
            ++عدد_نجی;
            توڑو;
        حالت 3: // shared SMCU
            اگر (shared_counts[id] < shared_گنتی) {
                shared_counts[id] = shared_گنتی;
            }
            توڑو;
        حالت 0:
            اگر (id == 0) {
                ++عدد_نجی;
            } ورنہ اگر (shared_counts[id] < shared_گنتی) {
                shared_counts[id] = shared_گنتی;
            }
            توڑو;
        طےشدہ:
            توڑو;
    }
}

ساکن size_t detect_عدد_smcus() {
    ثابت خودکار runtime_feat = ggml_feats_get_arch64_runtime();
    اگر (!runtime_feat.has_sme) {
        واپس 0;
    }

#اگر defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t عدد_نجی = 0;
    std::map<uint32_t, size_t> shared_counts;

    ثابت std::vector<size_t> cpus = detect_cpu_ids();
    برائے (ثابت size_t cpu : cpus) {
        ثابت std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        اگر (!file.is_open()) {
            جاری;
        }

        uint64_t smidr = 0;
        اگر (!(file >> std::hex >> smidr)) {
            جاری;
        }

        جوڑو_smcus_from_smidr(smidr, عدد_نجی, shared_counts);
    }

    size_t کل = عدد_نجی;
    برائے (ثابت خودکار & اندراج_2 : shared_counts) {
        کل += اندراج_2.second;
    }
    واپس کل;

#وگرنہ_اگر defined(__APPLE__) && defined(__aarch64__)
    صحیح perf_سطحیں = 0;
    size_t size = حجم(perf_سطحیں);
    اگر (sysctlbyname("hw.nperflevels", &perf_سطحیں, &size, خالی_اشاریہ, 0) != 0 ||
        size != حجم(perf_سطحیں) || perf_سطحیں <= 0) {
        واپس 0;
    }

    size_t units = 0;
    برائے (صحیح i = 0; i < perf_سطحیں; ++i) {
        حرف کلید[64] = {};
        صحیح physical_cpus = 0;
        صحیح cpus_فی_l2 = 0;

        snprintf(کلید, حجم(کلید), "hw.perflevel%d.physicalcpu", i);
        size = حجم(physical_cpus);
        اگر (sysctlbyname(کلید, &physical_cpus, &size, خالی_اشاریہ, 0) != 0 ||
            size != حجم(physical_cpus) || physical_cpus <= 0) {
            جاری;
        }

        snprintf(کلید, حجم(کلید), "hw.perflevel%d.cpusperl2", i);
        size = حجم(cpus_فی_l2);
        اگر (sysctlbyname(کلید, &cpus_فی_l2, &size, خالی_اشاریہ, 0) != 0 ||
            size != حجم(cpus_فی_l2) || cpus_فی_l2 <= 0) {
            جاری;
        }

        snprintf(کلید, حجم(کلید), "hw.perflevel%d.name", i);
        size = 0;
        اگر (sysctlbyname(کلید, خالی_اشاریہ, &size, خالی_اشاریہ, 0) != 0 || size == 0) {
            جاری;
        }

        std::string name(size, '\0');
        اگر (sysctlbyname(کلید, &name[0], &size, خالی_اشاریہ, 0) != 0) {
            جاری;
        }
        name.resize(size);
        جب (!name.خالی_خانے() && name.back() == '\0') {
            name.pop_back();
        }

        اگر (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_فی_l2 - 1) / cpus_فی_l2);
        }
    }

    واپس units;

#وگرنہ_اگر defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    واپس 0;

#ورنہ
    واپس 0;
#اگر_ختم
}

ساکن صحیح parse_uint_env(ثابت حرف *s, ثابت حرف *name, منطقی *ok) {
    اگر (!s) { *ok = جھوٹ; واپس 0; }
    حرف *end = خالی_اشاریہ;
    طویل v = strtol(s, &end, 10);
    اگر (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = جھوٹ;
        واپس 0;
    }
    اگر (v < 0 || v > زیادہ_سے_زیادہ_عدد) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = جھوٹ;
        واپس 0;
    }
    *ok = سچ;
    واپس (صحیح)v;
}

ساکن خالی init_kleidiai_context(خالی) {
    ggml_critical_section_آغاز();
    ساکن منطقی initialized = جھوٹ;

    اگر (!initialized) {
        initialized = سچ;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ثابت حرف *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ثابت حرف *env_دھاگے     = getenv("GGML_TOTAL_THREADS");
        ثابت حرف *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ثابت خودکار runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        اگر (env_دھاگے) {
            منطقی ok = جھوٹ;
            صحیح hint = parse_uint_env(env_دھاگے, "GGML_TOTAL_THREADS", &ok);
            اگر (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        اگر (env_chunk_mult) {
            منطقی ok = جھوٹ;
            صحیح multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            اگر (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        صحیح sme_cores = 0;
        منطقی sme_env_ok = جھوٹ;
        منطقی sme_env_سیٹ = (env_sme != خالی_اشاریہ);

        ثابت منطقی has_supported_sme_family = runtime_feat.has_sme;
        منطقی sme_cap_detected = جھوٹ;

        اگر (has_supported_sme_family) {
            detected_smcus = detect_عدد_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (صحیح)detected_smcus : 1;

            اگر (!sme_env_سیٹ && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        اگر (sme_env_سیٹ) {
            منطقی ok = جھوٹ;
            صحیح v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            اگر (ok) {
                اگر (has_supported_sme_family) {
                    sme_cores = v;
                } ورنہ {
                    اگر (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } ورنہ {
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
        } ورنہ {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_مقصد_string(ctx.kernels_q4->required_cpu));
        }

        اگر (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (صحیح)ctx.features);
        } ورنہ {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_مقصد_string(ctx.kernels_q8->required_cpu));
        }

        اگر (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (صحیح) ctx.features);
        } ورنہ {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_مقصد_string(ctx.kernels_q4k->required_cpu));
        }

        اگر (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (صحیح)ctx.features);
        } ورنہ {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_مقصد_string(ctx.kernels_f32->required_cpu));
        }

        ثابت منطقی has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        اگر (has_selected_sme_family_kernel) {
            اگر (sme_env_سیٹ && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } ورنہ اگر (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } ورنہ {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } ورنہ {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_انتہا();
}

ساکن اندرونی صحیح kleidiai_sme_thread_cap() {
    واپس ctx.sme_thread_cap;
}

ساکن اندرونی size_t align_اوپر(size_t value, size_t alignment) {
    اگر (alignment == 0) {
        واپس value;
    }
    ثابت size_t remainder = value % alignment;
    واپس remainder == 0 ? value : value + (alignment - remainder);
}

ساکن اندرونی size_t عاد_اعظم_سائز(size_t a, size_t b) {
    جب (b != 0) {
        ثابت size_t t = a % b;
        a = b;
        b = t;
    }
    واپس a;
}

ساکن اندرونی منطقی lcm_سائز(size_t a, size_t b, size_t & نتیجہ) {
    اگر (a == 0 || b == 0) {
        نتیجہ = 0;
        واپس جھوٹ;
    }
    ثابت size_t g = عاد_اعظم_سائز(a, b);
    ثابت size_t q = a / g;
    اگر (q > SIZE_MAX / b) {
        واپس جھوٹ;
    }
    نتیجہ = q * b;
    واپس سچ;
}

ساکن اندرونی size_t ceil_div_سائز(size_t a, size_t b) {
    واپس b == 0 ? 0 : (a + b - 1) / b;
}

ساکن اندرونی size_t kleidiai_chunk_ستون_تعداد(size_t n, صحیح nth_کل, منطقی disable_chunking, size_t n_قدم) {
    ثابت size_t multiplier = (nth_کل == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ثابت size_t مقسوم_علیہ = std::max<size_t>(1, (size_t) nth_کل * multiplier);
    ثابت size_t chunk_ستون_تعداد = align_اوپر(std::max<size_t>(1, ceil_div_سائز(n, مقسوم_علیہ)), n_قدم);
    واپس chunk_ستون_تعداد ? chunk_ستون_تعداد : n_قدم;
}

ساخت kleidiai_بلاک_دلائل {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

ساکن اندرونی kleidiai_بلاک_دلائل kleidiai_get_بلاک_دلائل(ggml_type rhs_type) {
    انتخاب (rhs_type) {
        حالت GGML_TYPE_Q4_0:
            واپس { QK4_0, QK4_0, QK4_0 };
        حالت GGML_TYPE_Q4_K:
            واپس { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        حالت GGML_TYPE_Q8_0:
            واپس { 0, 0, QK8_0 };
        طےشدہ:
            واپس { 0, 0, 0 };
    }
}

ساکن اندرونی منطقی kleidiai_pack_fallback_allowed() {
    اگر (ctx.sme_thread_cap <= 0) {
        واپس جھوٹ;
    }
    اگر (ctx.thread_hint <= 0) {
        واپس سچ;
    }
    واپس ctx.thread_hint > ctx.sme_thread_cap;
}

ساخت kleidiai_وزن_سرخی {
    uint32_t magic;
    uint16_t نسخہ_2;
    uint16_t خانہ_گنتی;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

ساکن اندرونی kleidiai_وزن_سرخی * kleidiai_وزن_سرخی_from_ptr(خالی * ڈیٹا) {
    واپس reinterpret_cast<kleidiai_وزن_سرخی *>(ڈیٹا);
}

ساکن اندرونی ثابت kleidiai_وزن_سرخی * kleidiai_وزن_سرخی_from_ptr(ثابت خالی * ڈیٹا) {
    واپس reinterpret_cast<ثابت kleidiai_وزن_سرخی *>(ڈیٹا);
}

ساکن اندرونی منطقی kleidiai_is_وزن_سرخی_valid(ثابت kleidiai_وزن_سرخی * سرخی) {
    اگر (!سرخی) {
        واپس جھوٹ;
    }
    اگر (سرخی->magic != GGML_KLEIDIAI_PACK_MAGIC || سرخی->نسخہ_2 != GGML_KLEIDIAI_PACK_نسخہ) {
        واپس جھوٹ;
    }
    اگر (سرخی->خانہ_گنتی == 0 || سرخی->خانہ_گنتی > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        واپس جھوٹ;
    }
    واپس سچ;
}

ساکن اندرونی uint8_t * kleidiai_وزن_خانہ_ptr(kleidiai_وزن_سرخی * سرخی, صحیح خانہ) {
    اگر (!kleidiai_is_وزن_سرخی_valid(سرخی)) {
        واپس خالی_اشاریہ;
    }
    اگر (خانہ < 0 || خانہ >= سرخی->خانہ_گنتی) {
        واپس خالی_اشاریہ;
    }
    واپس reinterpret_cast<uint8_t *>(سرخی) + سرخی->offsets[خانہ];
}

ساکن اندرونی ثابت uint8_t * kleidiai_وزن_خانہ_ptr(ثابت kleidiai_وزن_سرخی * سرخی, صحیح خانہ) {
    اگر (!kleidiai_is_وزن_سرخی_valid(سرخی)) {
        واپس خالی_اشاریہ;
    }
    اگر (خانہ < 0 || خانہ >= سرخی->خانہ_گنتی) {
        واپس خالی_اشاریہ;
    }
    واپس reinterpret_cast<ثابت uint8_t *>(سرخی) + سرخی->offsets[خانہ];
}

ساکن اندرونی ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    واپس ctx.kernels_q4;
}

ساکن اندرونی ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    واپس ctx.kernels_q4k;
}

ساکن اندرونی ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    واپس ctx.kernels_q8;
}

ساکن اندرونی ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    واپس ctx.kernels_f32;
}

سانچہ <قسم_نام SelectFallback>
ساکن صحیح kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & برآمد,
        SelectFallback select_fallback) {
    صحیح count = 0;
    اگر (!primary) {
        واپس 0;
    }
    برآمد[count++] = primary;

    اگر (primary->rhs_info.repack_منوال == RHS_REPACK_SINGLE_ONLY) {
        واپس count;
    }

    اگر (is_sme_family(primary->required_cpu)) {
        ثابت cpu_feature fallback_نقاب = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        اگر (fallback_نقاب != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_نقاب);
            اگر (fallback && fallback != primary &&
                fallback->rhs_info.repack_منوال != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                برآمد[count++] = fallback;
            }
        }
    }

    واپس count;
}

ساکن صحیح kleidiai_collect_kernel_chain(ثابت ساخت ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & برآمد) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    واپس kleidiai_collect_kernel_chain_common(primary, ctx.features, برآمد,
        [&](cpu_feature نقاب) { واپس ggml_kleidiai_select_kernels(نقاب, op); });
}

ساکن صحیح kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & برآمد) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    واپس kleidiai_collect_kernel_chain_common(primary, ctx.features, برآمد,
        [&](cpu_feature نقاب) { واپس ggml_kleidiai_select_kernels_q4_0(نقاب); });
}

ساکن صحیح kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & برآمد) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    واپس kleidiai_collect_kernel_chain_common(primary, ctx.features, برآمد,
        [&](cpu_feature نقاب) { واپس ggml_kleidiai_select_kernels_q4_k(نقاب); });
}

ساکن صحیح kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & برآمد) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    واپس kleidiai_collect_kernel_chain_common(primary, ctx.features, برآمد,
        [&](cpu_feature نقاب) { واپس ggml_kleidiai_select_kernels_q8_0(نقاب); });
}

ساکن صحیح kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & برآمد) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    واپس kleidiai_collect_kernel_chain_common(primary, ctx.features, برآمد,
        [&](cpu_feature نقاب) { واپس ggml_kleidiai_select_kernels_f32(نقاب); });
}

ساکن اندرونی int64_t ggml_ne(ثابت ggml_tensor * tensor, صحیح dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    واپس tensor->ne[dim];
}

نامخلا ggml::cpu::kleidiai {

ساکن size_t باری_نیچے_کو(size_t x, size_t y) {
    واپس y == 0 ? x : x - (x % y);
}

ساکن خالی transpose_f32kxn_f16nxk(size_t n, size_t k, اعشاری * dst, ثابت uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / حجم(uint16_t);
    size_t dst_stride = n;

    برائے (size_t k_idx = 0; k_idx < k; ++k_idx) {
        برائے (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

جماعت tensor_traits : عوامی ggml::cpu::tensor_traits {
    منطقی work_سائز(صحیح /* n_threads */, ثابت ساخت ggml_tensor * op, size_t & size) override {
        اگر (op->op != GGML_OP_MUL_MAT) {
            واپس جھوٹ;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحیح خانہ_گنتی = kleidiai_collect_kernel_chain(op, kernel_chain);
        اگر (خانہ_گنتی == 0) {
            واپس جھوٹ;
        }

        ثابت منطقی is_gemv = op->src[1]->ne[1] == 1;
        ثابت size_t k = op->src[0]->ne[0];
        ثابت size_t n = op->src[0]->ne[1];
        ثابت size_t m = op->src[1]->ne[1];

        اگر (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ثابت kleidiai_بلاک_دلائل بلاک_دلائل = kleidiai_get_بلاک_دلائل(kernel_chain[0]->rhs_type);
            ثابت size_t qk = بلاک_دلائل.pack_bl;

            size_t cursor = 0;
            منطقی any_خانہ = جھوٹ;

            برائے (صحیح خانہ = 0; خانہ < خانہ_گنتی; ++خانہ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[خانہ];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                اگر (!lhs_info || !lhs_info->packed_سائز_ex || !kernel) {
                    واپس جھوٹ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                ثابت size_t packed = lhs_info->packed_سائز_ex(m, k, qk, mr, kr, sr);

                cursor = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_خانہ = سچ;
            }

            اگر (!any_خانہ) {
                واپس جھوٹ;
            }

            size = cursor;
            واپس سچ;
        }

        اگر (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : خالی_اشاریہ;
            اگر (is_gemv && op->src[1]->nb[0] == (int64_t) حجم(اعشاری) && gemv_kernel &&
                gemv_kernel->get_lhs_انحراف_ex && gemv_kernel->get_rhs_packed_انحراف_ex &&
                gemv_kernel->تسلسل_kernel_ex && gemv_kernel->get_dst_انحراف) {
                size = 0;
                واپس سچ;
            }

            size_t cursor = 0;
            منطقی any_خانہ = جھوٹ;

            برائے (صحیح خانہ = 0; خانہ < خانہ_گنتی; ++خانہ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[خانہ];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                اگر (!lhs_info || !lhs_info->packed_سائز_ex || !kernel) {
                    واپس جھوٹ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_سائز_ex(m, k, 0, mr, kr, sr);
                any_خانہ = سچ;
            }

            اگر (!any_خانہ) {
                واپس جھوٹ;
            }

            size = cursor;
            واپس سچ;
        }

        اگر (op->src[0]->type == GGML_TYPE_F16) {
            ثابت int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ثابت int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ثابت int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            منطقی any_خانہ = جھوٹ;

            برائے (صحیح خانہ = 0; خانہ < خانہ_گنتی; ++خانہ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[خانہ];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                اگر (!lhs_info || !lhs_info->packed_سائز_ex || !kernels->rhs_info.packed_سائز_ex || !kernel) {
                    واپس جھوٹ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_سائز_ex(m * r, k, 0, mr, kr, sr);
                any_خانہ = سچ;
            }

            برائے (صحیح خانہ = 0; خانہ < خانہ_گنتی; ++خانہ) {
                ggml_kleidiai_kernels * kernels = kernel_chain[خانہ];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                اگر (!kernel || !kernels->rhs_info.packed_سائز_ex) {
                    واپس جھوٹ;
                }
                cursor  = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_سائز_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * حجم(اعشاری);
            cursor  = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * حجم(اعشاری);

            اگر (!any_خانہ) {
                واپس جھوٹ;
            }

            size = cursor;
            واپس سچ;
        }

        واپس جھوٹ;
    }

    منطقی compute_forward(ساخت ggml_compute_params * params, ساخت ggml_tensor * dst) override {
        اگر (dst->op == GGML_OP_MUL_MAT) {
            اگر (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                واپس compute_forward_qx(params, dst);
            } ورنہ اگر (dst->src[0]->type == GGML_TYPE_F32) {
                واپس compute_forward_f32(params, dst);
            } ورنہ اگر (dst->src[0]->type == GGML_TYPE_F16) {
                واپس compute_forward_fp16(params, dst);
            }
        }
        واپس جھوٹ;
    }

    منطقی compute_forward_f32(ggml_compute_params * params, ساخت ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ثنائی_OP_LOCALS

        اگر (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        اگر (!kernels) {
            واپس جھوٹ;
        }

        ثابت size_t k = ne00;
        ثابت size_t m = ne11;
        ثابت size_t n = ne01;
        ثابت منطقی use_gemv = m == 1 && src1->nb[0] == (int64_t) حجم(اعشاری) &&
                              kernels->gemv.get_lhs_انحراف_ex &&
                              kernels->gemv.get_rhs_packed_انحراف_ex &&
                              kernels->gemv.تسلسل_kernel_ex &&
                              kernels->gemv.get_dst_انحراف;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        اگر (!kernel || !kernel->get_lhs_انحراف_ex ||
            !kernel->get_rhs_packed_انحراف_ex || !kernel->تسلسل_kernel_ex || !kernel->get_dst_انحراف) {
            واپس جھوٹ;
        }

        اگر (!use_gemv && (!lhs_info || !lhs_info->get_انحراف || !lhs_info->get_packed_انحراف_ex ||
                          !lhs_info->packed_سائز_ex || !lhs_info->pack_func_ex)) {
            واپس جھوٹ;
        }

        ثابت kleidiai_وزن_سرخی * سرخی = kleidiai_وزن_سرخی_from_ptr(src0->ڈیٹا);
        ثابت منطقی has_سرخی = kleidiai_is_وزن_سرخی_valid(سرخی);

        ثابت uint8_t * rhs_بنیاد = has_سرخی ? kleidiai_وزن_خانہ_ptr(سرخی, 0)
                                              : static_cast<ثابت uint8_t *>(src0->ڈیٹا);
        اگر (!rhs_بنیاد) {
            واپس جھوٹ;
        }

        ثابت صحیح nth = params->nth > 0 ? params->nth : 1;
        ثابت صحیح ith = params->ith;

        ثابت size_t mr = kernel->get_mr();
        ثابت size_t kr = kernel->get_kr();
        ثابت size_t sr = kernel->get_sr();

        ثابت size_t lhs_packed_سائز = use_gemv ? 0 : lhs_info->packed_سائز_ex(m, k, 0, mr, kr, sr);
        اگر (!use_gemv) {
            GGML_ASSERT(lhs_packed_سائز <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ثابت size_t dst_stride = dst->nb[1];
        ثابت size_t n_قدم = kernel->get_n_قدم() ? kernel->get_n_قدم() : 1;
        ثابت منطقی disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) زیادہ_سے_زیادہ_عدد);

        ثابت int64_t batch_سائز = ne12 * ne13;
        برائے (int64_t batch_idx = 0; batch_idx < batch_سائز; ++batch_idx) {
            ثابت int64_t i12 = batch_idx % ne12;
            ثابت int64_t i13 = batch_idx / ne12;
            ثابت uint8_t * lhs_batch_بنیاد = static_cast<ثابت uint8_t *>(src1->ڈیٹا) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_بنیاد = static_cast<uint8_t *>(dst->ڈیٹا) + i12 * dst->nb[2] + i13 * dst->nb[3];

            اگر (!use_gemv) {
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_دھاگے = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_دھاگے = std::max<int64_t>(1, max_دھاگے);
                ثابت int64_t use_دھاگے = std::min<int64_t>(nth, max_دھاگے);

                اگر (ith < use_دھاگے) {
                    ثابت int64_t عدد_m_فی_thread0   = باری_نیچے_کو((size_t)(m_roundup_mr / use_دھاگے), mr);
                    ثابت int64_t عدد_m_فی_thread_N_1 = (int64_t)m - (use_دھاگے - 1) * عدد_m_فی_thread0;

                    ثابت int64_t m_آغاز = (int64_t)ith * عدد_m_فی_thread0;
                    ثابت int64_t m_گنتی = (ith == use_دھاگے - 1) ? عدد_m_فی_thread_N_1 : عدد_m_فی_thread0;

                    ثابت size_t بنیاد_packed_off  = lhs_info->get_packed_انحراف_ex(m_آغاز, k, 0, mr, kr, sr);
                    ثابت size_t اگلا_بلاک_off   = lhs_info->get_packed_انحراف_ex(m_آغاز + mr, k, 0, mr, kr, sr);
                    ثابت size_t صف_stride_بائٹیں = mr ? (اگلا_بلاک_off - بنیاد_packed_off) / mr : 0;

                    int64_t بقایا = m_گنتی;
                    int64_t cur       = m_آغاز;

                    جب (بقایا > 0) {
                        ثابت int64_t take = std::min<int64_t>((int64_t)m - cur, بقایا);
                        ثابت size_t src_off = lhs_info->get_انحراف(cur, src1->nb[1]);
                        ثابت خالی * src_ptr = lhs_batch_بنیاد + src_off;
                        ثابت size_t dst_off = بنیاد_packed_off + (size_t)(cur - m_آغاز) * صف_stride_بائٹیں;
                        خالی * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        بقایا -= take;
                    }
                }
            }

            اگر (ith == 0) {
                ggml_threadpool_chunk_سیٹ(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ثابت size_t chunk_ستون_تعداد = kleidiai_chunk_ستون_تعداد(n, nth, disable_chunking, n_قدم);
            GGML_ASSERT(chunk_ستون_تعداد <= (size_t) زیادہ_سے_زیادہ_عدد);

            صحیح current_ستون = ggml_threadpool_chunk_جوڑو(params->threadpool, (صحیح) chunk_ستون_تعداد);
            جب ((size_t) current_ستون < n) {
                ثابت size_t n_آغاز = (size_t) current_ستون;
                ثابت size_t n_مقصد_process = std::min(chunk_ستون_تعداد, n - n_آغاز);

                اگر (n_مقصد_process > 0) {
                    ثابت size_t lhs_انحراف = use_gemv ? kernel->get_lhs_انحراف_ex(0, k, 0)
                                                       : lhs_info->get_packed_انحراف_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_انحراف = kernel->get_rhs_packed_انحراف_ex(n_آغاز, k, 0);
                    ثابت size_t dst_انحراف        = kernel->get_dst_انحراف(0, n_آغاز, dst_stride);

                    ثابت خالی * lhs_ptr = use_gemv ? lhs_batch_بنیاد + lhs_انحراف
                                                    : lhs_packed + lhs_انحراف;
                    ثابت خالی * rhs_ptr = rhs_بنیاد + rhs_packed_انحراف;
                    اعشاری * dst_ptr      = reinterpret_cast<اعشاری *>(dst_batch_بنیاد + dst_انحراف);

                    kernel->تسلسل_kernel_ex(m, n_مقصد_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          حجم(اعشاری),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_ستون = ggml_threadpool_chunk_جوڑو(params->threadpool, (صحیح) chunk_ستون_تعداد);
            }

            اگر (batch_idx != batch_سائز - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        واپس سچ;
    }

    منطقی compute_forward_fp16(ggml_compute_params * params, ساخت ggml_tensor * dst) {
        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ثنائی_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        اگر (!kernels) {
            واپس جھوٹ;
        }

        ثابت منطقی is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        اگر (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_انحراف_ex || !kernel->get_rhs_packed_انحراف_ex || !kernel->تسلسل_kernel_ex) {
            واپس جھوٹ;
        }

        ثابت صحیح nth = params->nth;
        ثابت صحیح ith = params->ith;

        ثابت int64_t lhs_batch_size0 = ne12;
        ثابت int64_t rhs_batch_size0 = ne02;
        ثابت int64_t batch_سائز      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        ثابت int64_t r = lhs_batch_size0 / rhs_batch_size0;

        ثابت int64_t m_گروہ = ne11;
        ثابت int64_t m       = m_گروہ;
        ثابت int64_t n       = ne01;
        ثابت int64_t k       = ne00;

        ثابت size_t lhs_stride = src1->nb[1];
        ثابت size_t rhs_stride = src0->nb[1];
        ثابت size_t dst_stride = dst->nb[1];

        ثابت int64_t mr = (int64_t) kernel->get_mr();
        ثابت int64_t nr = (int64_t) kernel->get_nr();
        ثابت int64_t kr = (int64_t) kernel->get_kr();
        ثابت int64_t sr = (int64_t) kernel->get_sr();

        ثابت size_t lhs_packed_سائز = lhs_info->packed_سائز_ex(m, k, 0, mr, kr, sr);
        ثابت size_t rhs_packed_سائز = kernels->rhs_info.packed_سائز_ex(n, k, nr, kr, 0);
        ثابت size_t kxn_سائز        = k * n * حجم(اعشاری);
        ثابت size_t bias_سائز       = n * حجم(اعشاری);

        ثابت size_t wsize_required = lhs_packed_سائز + rhs_packed_سائز + kxn_سائز + bias_سائز;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_سائز;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_سائز;
        uint8_t * bias       = rhs_kxn + kxn_سائز;

        برائے (int64_t batch_idx = 0; batch_idx < batch_سائز; ++batch_idx) {
            ثابت int64_t rhs_batch_idx = batch_idx / r;
            ثابت uint8_t * rhs_batch_بنیاد = static_cast<ثابت uint8_t *>(src0->ڈیٹا) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_بنیاد = static_cast<uint8_t *>(dst->ڈیٹا) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ثابت int64_t m_roundup_mr = kai_roundup(m, mr);
                ثابت int64_t عدد_دھاگے  = KAI_MIN(m_roundup_mr / mr, nth);

                اگر (ith < عدد_دھاگے) {
                    ثابت int64_t عدد_m_فی_thread0   = باری_نیچے_کو((size_t)(m_roundup_mr / عدد_دھاگے), (size_t)mr);
                    ثابت int64_t عدد_m_فی_thread_N_1 = m - (عدد_دھاگے - 1) * عدد_m_فی_thread0;

                    ثابت int64_t m_آغاز = ith * عدد_m_فی_thread0;
                    ثابت int64_t m_گنتی = (ith == عدد_دھاگے - 1) ? عدد_m_فی_thread_N_1 : عدد_m_فی_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ثابت size_t بنیاد_packed_off  = lhs_info->get_packed_انحراف_ex(m_آغاز, k, 0, mr, kr, sr);
                    ثابت size_t اگلا_بلاک_off   = lhs_info->get_packed_انحراف_ex(m_آغاز + mr, k, 0, mr, kr, sr);
                    ثابت size_t صف_stride_بائٹیں = (اگلا_بلاک_off - بنیاد_packed_off) / (size_t)mr;

                    int64_t بقایا = m_گنتی;
                    int64_t cur       = m_آغاز;

                    جب (بقایا > 0) {
                        ثابت int64_t صف_اندر_گروہ = cur;
                        ثابت int64_t avail        = m_گروہ - صف_اندر_گروہ;
                        ثابت int64_t take         = std::min(avail, بقایا);

                        ثابت uint8_t * lhs_batch_بنیاد = static_cast<ثابت uint8_t *>(src1->ڈیٹا) + batch_idx * src1->nb[2];
                        ثابت خالی * src_ptr = lhs_batch_بنیاد + (size_t)صف_اندر_گروہ * lhs_stride;
                        ثابت size_t dst_off = بنیاد_packed_off + (size_t)(cur - m_آغاز) * صف_stride_بائٹیں;
                        خالی * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        بقایا -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            اگر (ith == 0) {
                memset(bias, 0, (size_t)n * حجم(اعشاری));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<اعشاری *>(rhs_kxn),
                                        reinterpret_cast<ثابت uint16_t *>(rhs_batch_بنیاد),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * حجم(اعشاری),
                             rhs_kxn, bias, خالی_اشاریہ, rhs_packed, 0, خالی_اشاریہ);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ثابت int64_t n_قدم  = (int64_t) kernel->get_n_قدم();
                int64_t عدد_دھاگے_n = KAI_MIN(n / n_قدم, nth);
                اگر (عدد_دھاگے_n <= 0) {
                    عدد_دھاگے_n = 1;
                }

                اگر (ith < عدد_دھاگے_n) {
                    ثابت int64_t عدد_n_فی_thread0   = باری_نیچے_کو((size_t)(n / عدد_دھاگے_n), (size_t)n_قدم);
                    ثابت int64_t عدد_n_فی_thread_N_1 = n - (عدد_دھاگے_n - 1) * عدد_n_فی_thread0;

                    ثابت int64_t n_آغاز      = ith * عدد_n_فی_thread0;
                    ثابت int64_t n_مقصد_process = (ith == عدد_دھاگے_n - 1) ? عدد_n_فی_thread_N_1 : عدد_n_فی_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ثابت size_t lhs_packed_offset0 = lhs_info->get_packed_انحراف_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_انحراف  = kernel->get_rhs_packed_انحراف_ex(n_آغاز, k, 0);
                    ثابت size_t dst_انحراف         = kernel->get_dst_انحراف((size_t)0, (size_t)n_آغاز, dst_stride);

                    ثابت خالی * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ثابت خالی * rhs_ptr = rhs_packed + rhs_packed_انحراف;
                    اعشاری * dst_ptr      = reinterpret_cast<اعشاری *>(dst_batch_بنیاد + dst_انحراف);

                    kernel->تسلسل_kernel_ex(m, n_مقصد_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, حجم(اعشاری), -FLT_MAX, FLT_MAX);
                }
            }

            اگر (batch_idx != batch_سائز - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        واپس سچ;
    }

    منطقی compute_forward_qx(ساخت ggml_compute_params * params, ساخت ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ثنائی_OP_LOCALS

        ثابت kleidiai_وزن_سرخی * سرخی = kleidiai_وزن_سرخی_from_ptr(src0->ڈیٹا);
        ثابت منطقی has_سرخی = kleidiai_is_وزن_سرخی_valid(سرخی);
        ثابت منطقی is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحیح خانہ_کل = kleidiai_collect_kernel_chain(dst, kernel_chain);

        خودکار وزن_for_خانہ = [&](صحیح خانہ_اشاریہ, size_t & سائز_برآمد) -> ثابت uint8_t * {
            اگر (خانہ_اشاریہ < 0 || خانہ_اشاریہ >= خانہ_کل) {
                واپس خالی_اشاریہ;
            }
            اگر (has_سرخی) {
                اگر (خانہ_اشاریہ < سرخی->خانہ_گنتی) {
                    سائز_برآمد = static_cast<size_t>(سرخی->sizes[خانہ_اشاریہ]);
                    واپس kleidiai_وزن_خانہ_ptr(سرخی, خانہ_اشاریہ);
                }
                واپس خالی_اشاریہ;
            }
            اگر (خانہ_اشاریہ == 0) {
                سائز_برآمد = ggml_nbytes(src0);
                واپس static_cast<ثابت uint8_t *>(src0->ڈیٹا);
            }
            واپس خالی_اشاریہ;
        };

        ساخت runtime_خانہ {
            صحیح خانہ_اشاریہ;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            size_t mr;
            size_t nr;
            size_t kr;
            size_t sr;
            size_t n_قدم;
            size_t lhs_packed_سائز;
            size_t lhs_انحراف;
            size_t lhs_bl;
            size_t rhs_bl;
            size_t pack_bl;
            size_t lhs_packed_offset0;
            صحیح assigned_دھاگے;
            صحیح thread_begin;
            صحیح thread_انتہا;
            ثابت uint8_t * rhs_بنیاد;
        };

        std::array<runtime_خانہ, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        صحیح runtime_گنتی = 0;

        برائے (صحیح خانہ = 0; خانہ < خانہ_کل && runtime_گنتی < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++خانہ) {
            ggml_kleidiai_kernels * kernels = kernel_chain[خانہ];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            اگر (!kinfo || !linfo || !linfo->packed_سائز_ex || !linfo->pack_func_ex || !linfo->get_انحراف ||
                !kinfo->get_rhs_packed_انحراف_ex || !kinfo->تسلسل_kernel_ex || !kinfo->get_dst_انحراف) {
                جاری;
            }

            size_t rhs_سائز = 0;
            ثابت uint8_t * rhs_ptr = وزن_for_خانہ(خانہ, rhs_سائز);
            اگر (!rhs_ptr || rhs_سائز == 0) {
                جاری;
            }

            ثابت kleidiai_بلاک_دلائل بلاک_دلائل = kleidiai_get_بلاک_دلائل(kernels->rhs_type);

            runtime[runtime_گنتی] = {
                خانہ,
                kernels,
                kinfo,
                linfo,
                kinfo->get_mr(),
                kinfo->get_nr(),
                kinfo->get_kr(),
                kinfo->get_sr(),
                kinfo->get_n_قدم(),
                0,
                0,
                بلاک_دلائل.lhs_bl,
                بلاک_دلائل.rhs_bl,
                بلاک_دلائل.pack_bl,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_گنتی;
        }

        اگر (runtime_گنتی == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            واپس جھوٹ;
        }

        ثابت صحیح nth_کل = params->nth > 0 ? params->nth : 1;
        ثابت صحیح ith_کل = params->ith;

        صحیح sme_خانہ = -1;
        صحیح non_sme_خانہ = -1;
        برائے (صحیح i = 0; i < runtime_گنتی; ++i) {
            اگر (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_خانہ = i;
                توڑو;
            }
        }

        برائے (صحیح i = 0; i < runtime_گنتی; ++i) {
            اگر (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_خانہ = i;
                توڑو;
            }
        }

        ثابت صحیح sme_cap_حد_بندی = ctx.sme_thread_cap;
        ثابت منطقی use_hybrid = sme_cap_حد_بندی > 0 &&
                                 runtime_گنتی > 1 &&
                                 nth_کل > sme_cap_حد_بندی;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_ستون_تعداد_فی_thread = 0;
        اگر (runtime_گنتی > 0 && nth_کل > 0) {
            min_ستون_تعداد_فی_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_کل);
        }
        ثابت منطقی too_small_for_hybrid = (min_ستون_تعداد_فی_thread < 2) || (ne11 < 128);

        ثابت منطقی hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        اگر (!hybrid_enabled) {
            صحیح chosen_خانہ = 0;
            اگر (too_small_for_hybrid && sme_خانہ != -1) {
                chosen_خانہ = nth_کل > sme_cap_حد_بندی && non_sme_خانہ != -1 ? non_sme_خانہ : sme_خانہ;
            } ورنہ اگر (runtime_گنتی > 1 && ctx.sme_thread_cap > 0 && nth_کل > ctx.sme_thread_cap) {
                chosen_خانہ = 1;
            }
            اگر (chosen_خانہ != 0 && chosen_خانہ < runtime_گنتی) {
                runtime[0] = runtime[chosen_خانہ];
                runtime[0].assigned_دھاگے = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_انتہا = 0;
            }
            runtime_گنتی = runtime_گنتی > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_خانہ = -1;
            اگر (runtime_گنتی > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_خانہ = 0;
            }
        }

        ثابت منطقی sme_single_only = sme_خانہ != -1 &&
                                     runtime[sme_خانہ].kernels->rhs_info.repack_منوال == RHS_REPACK_SINGLE_ONLY;
        صحیح sme_cap = sme_single_only ? nth_کل : kleidiai_sme_thread_cap();
        اگر (sme_cap < 0) {
            sme_cap = nth_کل;
        }
        sme_cap = std::min(sme_cap, nth_کل);

        صحیح دھاگے_بقایا = nth_کل;
        اگر (sme_خانہ != -1) {
            صحیح sme_دھاگے = std::min(std::max(sme_cap, 0), دھاگے_بقایا);
            runtime[sme_خانہ].assigned_دھاگے = sme_دھاگے;
            دھاگے_بقایا -= sme_دھاگے;
        }

        صحیح fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        صحیح fallback_گنتی = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        برائے (صحیح i = 0; i < runtime_گنتی; ++i) {
            اگر (i == sme_خانہ) {
                جاری;
            }
            fallback_indices[fallback_گنتی++] = i;
        }

        برائے (صحیح fi = 0; fi < fallback_گنتی; ++fi) {
            اگر (دھاگے_بقایا <= 0) {
                توڑو;
            }
            ثابت صحیح خانہ_اشاریہ = fallback_indices[fi];
            ثابت صحیح slots_بایاں = fallback_گنتی - fi;
            صحیح share = (دھاگے_بقایا + slots_بایاں - 1) / slots_بایاں;
            share     = std::min(share, دھاگے_بقایا);
            runtime[خانہ_اشاریہ].assigned_دھاگے = share;
            دھاگے_بقایا -= share;
        }

        صحیح thread_cursor = 0;
        برائے (صحیح i = 0; i < runtime_گنتی; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_دھاگے;
            runtime[i].thread_انتہا = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ثابت صحیح فعال_دھاگے = thread_cursor;

        صحیح local_خانہ = -1;
        صحیح local_ith  = 0;
        برائے (صحیح i = 0; i < runtime_گنتی; ++i) {
            اگر (ith_کل >= runtime[i].thread_begin && ith_کل < runtime[i].thread_انتہا) {
                local_خانہ = i;
                local_ith  = ith_کل - runtime[i].thread_begin;
                توڑو;
            }
        }
        ثابت size_t k = ne00;
        ثابت size_t m = ne11;
        ثابت size_t n = ne01;

        size_t cursor = 0;
        برائے (صحیح i = 0; i < runtime_گنتی; ++i) {
            runtime[i].lhs_packed_سائز = runtime[i].lhs_info->packed_سائز_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_انحراف = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_انحراف_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_سائز;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_قدم = 1;
        برائے (صحیح i = 0; i < runtime_گنتی; ++i) {
            اگر (runtime[i].assigned_دھاگے == 0) {
                جاری;
            }
            size_t اگلا_قدم = 0;
            اگر (!lcm_سائز(common_قدم, runtime[i].n_قدم ? runtime[i].n_قدم : 1, اگلا_قدم)) {
                واپس جھوٹ;
            }
            common_قدم = اگلا_قدم;
        }
        GGML_ASSERT(common_قدم > 0);

        ثابت منطقی disable_chunking = ggml_is_numa();
        ثابت size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ثابت size_t chunk_مقسوم_علیہ = (فعال_دھاگے == 1 || disable_chunking) ? (size_t) فعال_دھاگے : (size_t) فعال_دھاگے * chunk_multiplier;
        size_t chunk_ستون_تعداد = align_اوپر(std::max<size_t>(1, ceil_div_سائز(n, chunk_مقسوم_علیہ)), common_قدم);
        اگر (chunk_ستون_تعداد == 0) {
            chunk_ستون_تعداد = common_قدم;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ثابت size_t nchunk_سائز = std::max<size_t>(1, ceil_div_سائز(n, chunk_ستون_تعداد));
        GGML_ASSERT(nchunk_سائز <= (size_t)زیادہ_سے_زیادہ_عدد);
        ثابت صحیح nchunk = (صحیح)nchunk_سائز;
        ثابت size_t dst_stride = dst->nb[1];

        خودکار تسلسل_chunk = [&](runtime_خانہ & خانہ, size_t global_آغاز, size_t ستون_تعداد_2, uint8_t * dst_batch_بنیاد) {
            ثابت size_t rhs_packed_انحراف = خانہ.kernel->get_rhs_packed_انحراف_ex(global_آغاز, k, خانہ.rhs_bl);
            ثابت size_t dst_انحراف        = خانہ.kernel->get_dst_انحراف(0, global_آغاز, dst_stride);

            ثابت uint8_t * lhs_ptr = scratch + خانہ.lhs_انحراف + خانہ.lhs_packed_offset0;
            ثابت uint8_t * rhs_ptr = خانہ.rhs_بنیاد + rhs_packed_انحراف;
            اعشاری * dst_ptr         = reinterpret_cast<اعشاری *>(dst_batch_بنیاد + dst_انحراف);

            خانہ.kernel->تسلسل_kernel_ex(m, ستون_تعداد_2, k, خانہ.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       حجم(اعشاری),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        برائے (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ثابت uint8_t * lhs_batch_بنیاد = static_cast<ثابت uint8_t *>(src1->ڈیٹا) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_بنیاد = static_cast<uint8_t *>(dst->ڈیٹا) + batch_idx * dst->nb[2];

            اگر (local_خانہ != -1) {
                runtime_خانہ & خانہ = runtime[local_خانہ];
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)خانہ.mr);
                int64_t max_دھاگے = خانہ.mr ? (m_roundup_mr / (int64_t)خانہ.mr) : خانہ.assigned_دھاگے;
                max_دھاگے = std::max<int64_t>(1, max_دھاگے);
                ثابت int64_t use_دھاگے = std::min<int64_t>(خانہ.assigned_دھاگے, max_دھاگے);

                اگر (local_ith < use_دھاگے) {
                    ثابت int64_t عدد_m_فی_thread0   = باری_نیچے_کو((size_t)(m_roundup_mr / use_دھاگے), خانہ.mr);
                    ثابت int64_t عدد_m_فی_thread_N_1 = (int64_t)m - (use_دھاگے - 1) * عدد_m_فی_thread0;

                    ثابت int64_t m_آغاز = (int64_t)local_ith * عدد_m_فی_thread0;
                    ثابت int64_t m_گنتی = (local_ith == use_دھاگے - 1) ? عدد_m_فی_thread_N_1 : عدد_m_فی_thread0;

                    ثابت size_t بنیاد_packed_off  = خانہ.lhs_info->get_packed_انحراف_ex(m_آغاز, k, خانہ.lhs_bl, خانہ.mr, خانہ.kr, خانہ.sr);
                    ثابت size_t اگلا_بلاک_off   = خانہ.lhs_info->get_packed_انحراف_ex(m_آغاز + خانہ.mr, k, خانہ.lhs_bl, خانہ.mr, خانہ.kr, خانہ.sr);
                    ثابت size_t صف_stride_بائٹیں = خانہ.mr ? (اگلا_بلاک_off - بنیاد_packed_off) / خانہ.mr : 0;

                    int64_t بقایا = m_گنتی;
                    int64_t cur       = m_آغاز;

                    uint8_t * lhs_packed = scratch + خانہ.lhs_انحراف;
                    جب (بقایا > 0) {
                        ثابت int64_t صف_اندر_گروہ = cur;
                        ثابت int64_t avail        = (int64_t)m - صف_اندر_گروہ;
                        ثابت int64_t take         = std::min(avail, بقایا);

                        ثابت size_t src_off = خانہ.lhs_info->get_انحراف(صف_اندر_گروہ, src1->nb[1]);
                        ثابت خالی * src_ptr = lhs_batch_بنیاد + src_off;
                        ثابت size_t dst_off = بنیاد_packed_off + (size_t)(cur - m_آغاز) * صف_stride_بائٹیں;
                        خالی * dst_ptr       = lhs_packed + dst_off;

                        خانہ.lhs_info->pack_func_ex(take, k, خانہ.lhs_bl, خانہ.mr, خانہ.kr, خانہ.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        بقایا -= take;
                    }
                }
            }

            اگر (ith_کل == 0) {
                ggml_threadpool_chunk_سیٹ(params->threadpool, فعال_دھاگے);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            اگر (local_خانہ != -1) {
                runtime_خانہ & خانہ = runtime[local_خانہ];
                صحیح current_chunk = ith_کل;
                جب (current_chunk < nchunk) {
                    ثابت size_t global_آغاز = (size_t)current_chunk * chunk_ستون_تعداد;
                    اگر (global_آغاز >= n) {
                        توڑو;
                    }

                    ثابت size_t ستون_تعداد_2 = std::min(chunk_ستون_تعداد, n - global_آغاز);
                    اگر (ستون_تعداد_2 > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        تسلسل_chunk(خانہ, global_آغاز, ستون_تعداد_2, dst_batch_بنیاد);
                    }

                    current_chunk = ggml_threadpool_chunk_جوڑو(params->threadpool, 1);
                }
            }

            اگر (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        واپس سچ;
    }

عوامی:
    صحیح repack(ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t ڈیٹا_سائز) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ثابت size_t n = tensor->ne[1];
        ثابت size_t k = tensor->ne[0];

        kleidiai_وزن_سرخی * سرخی = kleidiai_وزن_سرخی_from_ptr(tensor->ڈیٹا);
        اگر (!سرخی) {
            واپس -1;
        }

        سرخی->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        سرخی->نسخہ_2    = GGML_KLEIDIAI_PACK_نسخہ;
        سرخی->خانہ_گنتی = 0;

        uint8_t * بنیاد_ptr = static_cast<uint8_t *>(tensor->ڈیٹا);
        size_t cursor = حجم(kleidiai_وزن_سرخی);
        cursor = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت منطقی want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ثابت منطقی want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ثابت منطقی want_f32 = tensor->type == GGML_TYPE_F32;
        ثابت صحیح خانہ_کل = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ثابت منطقی allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<اعشاری>  scales;
        std::vector<اعشاری>  bias;
        std::vector<اعشاری> صف_f32;

        اگر (want_q8 && خانہ_کل > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ثابت size_t صف_stride = tensor->nb[1];
            ثابت size_t k_بلاکیں   = (k + QK8_0 - 1) / QK8_0;

            اگر (tensor->type == GGML_TYPE_Q6_K) {
                اگر (k % QK_K != 0 || k % QK8_0 != 0) {
                    واپس -1;
                }
                صف_f32.resize(k);
            }

            برائے (size_t صف = 0; صف < n; ++صف) {
                ثابت uint8_t * صف_ڈیٹا = static_cast<ثابت uint8_t *>(ڈیٹا) + صف * صف_stride;

                اگر (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_صف_q6_K(reinterpret_cast<ثابت بلاک_q6_K *>(صف_ڈیٹا), صف_f32.ڈیٹا(), k);
                }

                اعشاری max_abs = 0.0f;
                اگر (tensor->type == GGML_TYPE_Q6_K) {
                    برائے (size_t خطی_idx = 0; خطی_idx < k; ++خطی_idx) {
                        max_abs = std::max(max_abs, std::fabs(صف_f32[خطی_idx]));
                    }
                } ورنہ {
                    ثابت بلاک_q8_0 * صف_بلاکیں = reinterpret_cast<ثابت بلاک_q8_0 *>(صف_ڈیٹا);
                    برائے (size_t بلاک = 0; بلاک < k_بلاکیں; ++بلاک) {
                        ثابت بلاک_q8_0 & blk = صف_بلاکیں[بلاک];
                        ثابت اعشاری d = GGML_FP16_مقصد_FP32(blk.d);
                        برائے (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t خطی_idx = بلاک * QK8_0 + l;
                            اگر (خطی_idx >= k) {
                                توڑو;
                            }
                            ثابت اعشاری value = d * static_cast<اعشاری>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                اعشاری scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[صف] = scale;
                ثابت اعشاری inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                اگر (tensor->type == GGML_TYPE_Q6_K) {
                    برائے (size_t خطی_idx = 0; خطی_idx < k; ++خطی_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(صف_f32[خطی_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[صف * k + خطی_idx] = static_cast<int8_t>(q);
                    }
                } ورنہ {
                    ثابت بلاک_q8_0 * صف_بلاکیں = reinterpret_cast<ثابت بلاک_q8_0 *>(صف_ڈیٹا);
                    برائے (size_t بلاک = 0; بلاک < k_بلاکیں; ++بلاک) {
                        ثابت بلاک_q8_0 & blk = صف_بلاکیں[بلاک];
                        ثابت اعشاری d = GGML_FP16_مقصد_FP32(blk.d);
                        برائے (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t خطی_idx = بلاک * QK8_0 + l;
                            اگر (خطی_idx >= k) {
                                توڑو;
                            }
                            ثابت اعشاری value = d * static_cast<اعشاری>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[صف * k + خطی_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        اگر (want_f32 && خانہ_کل > 0) {
            bias.resize(n, 0.0f);
        }

        برائے (صحیح خانہ = 0; خانہ < خانہ_کل && خانہ < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++خانہ) {
            اگر (!allow_fallback && خانہ > 0) {
                توڑو;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[خانہ];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            اگر (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_سائز_ex || !kernel) {
                جاری;
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
            ثابت size_t بلاک_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            اگر (بلاک_len == SIZE_MAX) {
                جاری;
            }

            ثابت size_t packed_سائز = rhs_info->packed_سائز_ex(n, k, nr, kr, بلاک_len);
            ثابت size_t aligned_cursor = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = بنیاد_ptr + aligned_cursor;

            اگر (rhs_type == GGML_TYPE_Q4_0) {
                ساخت kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_نقطہ = 1;
                params.rhs_zero_نقطہ = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ثابت uint8_t *>(ڈیٹا), خالی_اشاریہ, خالی_اشاریہ,
                                       dst_ptr, 0, &params);
            } ورنہ اگر (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       ڈیٹا, خالی_اشاریہ, خالی_اشاریہ, dst_ptr, 0, خالی_اشاریہ);
            } ورنہ اگر (rhs_type == GGML_TYPE_Q8_0) {
                ساخت kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_نقطہ = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.ڈیٹا(), خالی_اشاریہ, scales.ڈیٹا(),
                                       dst_ptr, 0, &params);
            } ورنہ اگر (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       ڈیٹا, bias.ڈیٹا(), خالی_اشاریہ,
                                       dst_ptr, 0, خالی_اشاریہ);
            } ورنہ {
                جاری;
            }

            سرخی->offsets[سرخی->خانہ_گنتی] = aligned_cursor;
            سرخی->sizes[سرخی->خانہ_گنتی]   = packed_سائز;
            ++سرخی->خانہ_گنتی;

            cursor = aligned_cursor + packed_سائز;
        }

        اگر (سرخی->خانہ_گنتی == 0) {
            سرخی->magic   = 0;
            سرخی->نسخہ_2 = 0;
            حافظہ_نقل(tensor->ڈیٹا, ڈیٹا, ڈیٹا_سائز);
        }

        واپس 0;
    }
};

ساکن ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_بفر_t, ساخت ggml_tensor *) {
    ساکن tensor_traits traits;
    واپس &traits;
}
}  // namespace ggml::cpu::kleidiai

ساکن شمار ggml_حالت ggml_backend_cpu_kleidiai_بفر_init_tensor(ggml_backend_بفر_t بفر, ساخت ggml_tensor * tensor) {
    tensor->extra = (خالی *) ggml::cpu::kleidiai::get_tensor_traits(بفر, tensor);

    واپس GGML_حالت_SUCCESS;
}

ساکن خالی ggml_backend_cpu_kleidiai_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر, ساخت ggml_tensor * tensor,
                                                       ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    GGML_ASSERT(انحراف == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    خودکار tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    خودکار OK            = tensor_traits->repack(tensor, ڈیٹا, size);

    GGML_ASSERT(OK == 0);
    GGML_غیر_مستعمل(بفر);
}

ساکن ثابت حرف * ggml_backend_cpu_kleidiai_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    GGML_غیر_مستعمل(buft);
    واپس "CPU_KLEIDIAI";
}

ساکن ggml_backend_بفر_t ggml_backend_cpu_kleidiai_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(ggml_backend_cpu_بفر_type(), size);

    اگر (بفر == خالی_اشاریہ) {
        واپس خالی_اشاریہ;
    }

    بفر->buft              = buft;
    بفر->iface.init_tensor = ggml_backend_cpu_kleidiai_بفر_init_tensor;
    بفر->iface.سیٹ_tensor  = ggml_backend_cpu_kleidiai_بفر_سیٹ_tensor;
    بفر->iface.get_tensor  = خالی_اشاریہ;
    بفر->iface.cpy_tensor  = خالی_اشاریہ;
    واپس بفر;
}

ساکن size_t ggml_backend_cpu_kleidiai_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    GGML_غیر_مستعمل(buft);
    واپس TENSOR_ALIGNMENT;
}

ساکن size_t ggml_backend_cpu_kleidiai_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ساخت ggml_tensor * tensor) {
    GGML_غیر_مستعمل(buft);

    اگر (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        واپس ggml_nbytes(tensor);
    }

    ثابت size_t n = tensor->ne[1];
    ثابت size_t k = tensor->ne[0];

    size_t cursor = حجم(kleidiai_وزن_سرخی);
    cursor = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ثابت منطقی want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ثابت منطقی want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ثابت منطقی want_f32 = tensor->type == GGML_TYPE_F32;
    ثابت صحیح خانہ_کل = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ثابت منطقی allow_fallback = kleidiai_pack_fallback_allowed();

    size_t خانہ_گنتی = 0;
    برائے (صحیح خانہ = 0; خانہ < خانہ_کل; ++خانہ) {
        اگر (!allow_fallback && خانہ > 0) {
            توڑو;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[خانہ];
        اگر (!kernels) {
            جاری;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        اگر (!kernel || !rhs_info || !rhs_info->packed_سائز_ex) {
            جاری;
        }

        ثابت ggml_type rhs_type = kernels->rhs_type;
        ثابت size_t بلاک_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        اگر (بلاک_len == SIZE_MAX) {
            جاری;
        }

        cursor = align_اوپر(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_سائز_ex(n, k, kernel->get_nr(), kernel->get_kr(), بلاک_len);
        ++خانہ_گنتی;
    }

    اگر (خانہ_گنتی == 0) {
        واپس ggml_nbytes(tensor);
    }

    واپس std::max(cursor, ggml_nbytes(tensor));
}

نامخلا ggml::cpu::kleidiai {
جماعت extra_بفر_type : ggml::cpu::extra_بفر_type {
    منطقی supports_op(ggml_backend_dev_t, ثابت ساخت ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        اگر (op->op != GGML_OP_MUL_MAT) {
            واپس جھوٹ;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحیح خانہ_کل = kleidiai_collect_kernel_chain(op, kernel_chain);
        ثابت منطقی src0_is_kleidiai =
            op->src[0]->بفر &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->بفر->buft->context == یہ &&
            خانہ_کل > 0;

        اگر ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            اگر (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == خالی_اشاریہ) {
                واپس جھوٹ;
            }
            اگر (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == خالی_اشاریہ) {
                واپس جھوٹ;
            }
            اگر (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == خالی_اشاریہ) {
                واپس جھوٹ;
            }
            اگر (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == خالی_اشاریہ) {
                واپس جھوٹ;
            }
            اگر (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == خالی_اشاریہ) {
                واپس جھوٹ;
            }
            اگر (op->src[1]->بفر && !ggml_backend_buft_is_host(op->src[1]->بفر->buft)) {
                واپس جھوٹ;
            }

            اگر (op->src[0]->type == GGML_TYPE_Q4_K) {
                اگر (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    واپس جھوٹ;
                }
                واپس ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == حجم(اعشاری) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * حجم(اعشاری));
            }

            اگر (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                اگر (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    واپس سچ;
                }
                واپس جھوٹ;
            }

            اگر (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                واپس جھوٹ;
            }

            واپس سچ;
        }

        واپس جھوٹ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت ساخت ggml_tensor * op) override {
        اگر (op->op == GGML_OP_MUL_MAT) {
            اگر (op->src[0]->بفر && op->src[0]->بفر->buft->context == یہ) {
                واپس (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } ورنہ {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                اگر (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    ساکن std::atomic<منطقی> warned(جھوٹ);
                    اگر (!warned.exchange(سچ)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                اگر (op->src[0]->type != GGML_TYPE_F16) {
                    واپس خالی_اشاریہ;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ثابت صحیح خانہ_کل = kleidiai_collect_kernel_chain(op, kernel_chain);
                اگر (خانہ_کل > 0 && op->src[1]->ne[1] > 1) {
                    اگر ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        واپس خالی_اشاریہ;
                    }
                    واپس ggml::cpu::kleidiai::get_tensor_traits(تہی, تہی);
                }
            }
        }
        واپس خالی_اشاریہ;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_بفر_type_t ggml_backend_cpu_kleidiai_بفر_type(خالی) {
    ساکن ggml::cpu::kleidiai::extra_بفر_type ctx;
    ساکن ساخت ggml_backend_بفر_type ggml_backend_cpu_بفر_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_بفر_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_بفر_type_alloc_بفر,
                           /* .alloc_buffer_n       = */ خالی_اشاریہ,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_بفر_type_get_alignment,
                           /* .get_max_size         = */ خالی_اشاریہ,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_بفر_type_get_alloc_سائز,
                           /* .get_alloc_size_n     = */ تہی,
                           /* .is_host              = */ خالی_اشاریہ,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    واپس &ggml_backend_cpu_بفر_type_kleidiai;
}
