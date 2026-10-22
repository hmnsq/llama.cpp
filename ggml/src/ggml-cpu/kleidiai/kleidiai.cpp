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
#إذا defined(__linux__)
#ضمن <asm/hwcap.h>
#ضمن <dirent.h>
#ضمن <sys/auxv.h>
#ضمن <sys/types.h>
#ضمن <sys/stat.h>
#ضمن <unistd.h>
#وإلاإذا defined(__APPLE__)
#ضمن <sys/sysctl.h>
#ضمن <sys/types.h>
#د_که_پای

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

#تعریف_کړه GGML_COMMON_DECL_CPP
#ضمن "ggml-common.h"

ساكن constexpr صحيح      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
ساكن constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
ساكن constexpr uint16_t GGML_KLEIDIAI_PACK_نسخه     = 1;
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
} ساكن ctx = { CPU_FEATURE_NONE, تش_اشاره, تش_اشاره, تش_اشاره, تش_اشاره, 0, -1, 4 };

ساكن مضمن منطقي is_sme_family(cpu_feature f) {
    أرجع (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

ساكن ثابت حرف* cpu_feature_منزل_string(cpu_feature f) {
    إذا (f == CPU_FEATURE_NONE) {
        أرجع "NONE";
    } وإلا إذا ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        أرجع "SME2";
    } وإلا إذا ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        أرجع "SME";
    } وإلا إذا ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        أرجع "SVE";
    }
    وإلا إذا ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        أرجع "I8MM";
    } وإلا إذا ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        أرجع "DOTPROD";
    }
    وإلا {
        أرجع "UNKNOWN";
    }
}

#إذا defined(__linux__) && defined(__aarch64__)
ساكن منطقي parse_cpu_dir_name(ثابت حرف* name, size_t* cpu) {
    إذا (مزي_پرتله(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        أرجع خطأ;
    }

    ثابت حرف* لومړی = name + 3;
    ثابت حرف* وروستی = name + مزي_اوږدوالی(name);

    size_t value = 0;
    ثابت خپلکاری [end, ec] = std::from_chars(لومړی, وروستی, value, 10);

    إذا (ec != std::errc{} || end != وروستی) {
        أرجع خطأ;
    }

    *cpu = value;
    أرجع ریښتیا;
}

ساكن std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    إذا (dir == تش_اشاره) {
        أرجع cpus;
    }

    طالما (dirent * ثبت = readdir(dir)) {
        size_t cpu = 0;
        إذا (parse_cpu_dir_name(ثبت->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    أرجع cpus;
}
#د_که_پای

#إذا defined(__APPLE__) && defined(__aarch64__)
ساكن منطقي apple_sme_counted_perf_level(std::string name) {
    لكل (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (حرف) std::tolower((بینښه حرف) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    أرجع name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#د_که_پای

ساكن فراغ جمع_کړه_smcus_from_smidr(uint64_t smidr, size_t & عدد_شخصي, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ثابت uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ثابت uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ثابت size_t shared_شمېر = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ثابت uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ثابت uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ثابت uint32_t id = (affinity2 << 12) | affinity;

    إذا (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    بدل (sh) {
        حالة 2: // private SMCU
            ++عدد_شخصي;
            اكسر;
        حالة 3: // shared SMCU
            إذا (shared_counts[id] < shared_شمېر) {
                shared_counts[id] = shared_شمېر;
            }
            اكسر;
        حالة 0:
            إذا (id == 0) {
                ++عدد_شخصي;
            } وإلا إذا (shared_counts[id] < shared_شمېر) {
                shared_counts[id] = shared_شمېر;
            }
            اكسر;
        افتراضي:
            اكسر;
    }
}

ساكن size_t detect_عدد_smcus() {
    ثابت خپلکاری runtime_feat = ggml_feats_get_arch64_runtime();
    إذا (!runtime_feat.has_sme) {
        أرجع 0;
    }

#إذا defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t عدد_شخصي = 0;
    std::map<uint32_t, size_t> shared_counts;

    ثابت std::vector<size_t> cpus = detect_cpu_ids();
    لكل (ثابت size_t cpu : cpus) {
        ثابت std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        إذا (!file.is_open()) {
            استمر;
        }

        uint64_t smidr = 0;
        إذا (!(file >> std::hex >> smidr)) {
            استمر;
        }

        جمع_کړه_smcus_from_smidr(smidr, عدد_شخصي, shared_counts);
    }

    size_t ټول = عدد_شخصي;
    لكل (ثابت خپلکاری & ثبت : shared_counts) {
        ټول += ثبت.second;
    }
    أرجع ټول;

#وإلاإذا defined(__APPLE__) && defined(__aarch64__)
    صحيح perf_پوړونه = 0;
    size_t size = حجم(perf_پوړونه);
    إذا (sysctlbyname("hw.nperflevels", &perf_پوړونه, &size, تش_اشاره, 0) != 0 ||
        size != حجم(perf_پوړونه) || perf_پوړونه <= 0) {
        أرجع 0;
    }

    size_t units = 0;
    لكل (صحيح i = 0; i < perf_پوړونه; ++i) {
        حرف کلی[64] = {};
        صحيح physical_cpus = 0;
        صحيح cpus_فی_l2 = 0;

        snprintf(کلی, حجم(کلی), "hw.perflevel%d.physicalcpu", i);
        size = حجم(physical_cpus);
        إذا (sysctlbyname(کلی, &physical_cpus, &size, تش_اشاره, 0) != 0 ||
            size != حجم(physical_cpus) || physical_cpus <= 0) {
            استمر;
        }

        snprintf(کلی, حجم(کلی), "hw.perflevel%d.cpusperl2", i);
        size = حجم(cpus_فی_l2);
        إذا (sysctlbyname(کلی, &cpus_فی_l2, &size, تش_اشاره, 0) != 0 ||
            size != حجم(cpus_فی_l2) || cpus_فی_l2 <= 0) {
            استمر;
        }

        snprintf(کلی, حجم(کلی), "hw.perflevel%d.name", i);
        size = 0;
        إذا (sysctlbyname(کلی, تش_اشاره, &size, تش_اشاره, 0) != 0 || size == 0) {
            استمر;
        }

        std::string name(size, '\0');
        إذا (sysctlbyname(کلی, &name[0], &size, تش_اشاره, 0) != 0) {
            استمر;
        }
        name.resize(size);
        طالما (!name.تشه() && name.back() == '\0') {
            name.pop_back();
        }

        إذا (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_فی_l2 - 1) / cpus_فی_l2);
        }
    }

    أرجع units;

#وإلاإذا defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    أرجع 0;

#وإلا
    أرجع 0;
#د_که_پای
}

ساكن صحيح parse_uint_env(ثابت حرف *s, ثابت حرف *name, منطقي *ok) {
    إذا (!s) { *ok = خطأ; أرجع 0; }
    حرف *end = تش_اشاره;
    اوږد v = strtol(s, &end, 10);
    إذا (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = خطأ;
        أرجع 0;
    }
    إذا (v < 0 || v > لوی_عدد) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = خطأ;
        أرجع 0;
    }
    *ok = ریښتیا;
    أرجع (صحيح)v;
}

ساكن فراغ init_kleidiai_context(فراغ) {
    ggml_critical_section_پیل();
    ساكن منطقي initialized = خطأ;

    إذا (!initialized) {
        initialized = ریښتیا;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ثابت حرف *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ثابت حرف *env_تارونه     = getenv("GGML_TOTAL_THREADS");
        ثابت حرف *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ثابت خپلکاری runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        إذا (env_تارونه) {
            منطقي ok = خطأ;
            صحيح hint = parse_uint_env(env_تارونه, "GGML_TOTAL_THREADS", &ok);
            إذا (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        إذا (env_chunk_mult) {
            منطقي ok = خطأ;
            صحيح multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            إذا (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        صحيح sme_cores = 0;
        منطقي sme_env_ok = خطأ;
        منطقي sme_env_سېټ = (env_sme != تش_اشاره);

        ثابت منطقي has_supported_sme_family = runtime_feat.has_sme;
        منطقي sme_cap_detected = خطأ;

        إذا (has_supported_sme_family) {
            detected_smcus = detect_عدد_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (صحيح)detected_smcus : 1;

            إذا (!sme_env_سېټ && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        إذا (sme_env_سېټ) {
            منطقي ok = خطأ;
            صحيح v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            إذا (ok) {
                إذا (has_supported_sme_family) {
                    sme_cores = v;
                } وإلا {
                    إذا (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } وإلا {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        إذا (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            إذا (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        إذا (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (صحيح)ctx.features);
        } وإلا {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_منزل_string(ctx.kernels_q4->required_cpu));
        }

        إذا (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (صحيح)ctx.features);
        } وإلا {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_منزل_string(ctx.kernels_q8->required_cpu));
        }

        إذا (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (صحيح) ctx.features);
        } وإلا {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_منزل_string(ctx.kernels_q4k->required_cpu));
        }

        إذا (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (صحيح)ctx.features);
        } وإلا {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_منزل_string(ctx.kernels_f32->required_cpu));
        }

        ثابت منطقي has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        إذا (has_selected_sme_family_kernel) {
            إذا (sme_env_سېټ && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } وإلا إذا (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } وإلا {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } وإلا {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_پای_ټکی();
}

ساكن مضمن صحيح kleidiai_sme_thread_cap() {
    أرجع ctx.sme_thread_cap;
}

ساكن مضمن size_t align_پورته(size_t value, size_t alignment) {
    إذا (alignment == 0) {
        أرجع value;
    }
    ثابت size_t remainder = value % alignment;
    أرجع remainder == 0 ? value : value + (alignment - remainder);
}

ساكن مضمن size_t لوی_ګډ_وېشونکی_اندازه(size_t a, size_t b) {
    طالما (b != 0) {
        ثابت size_t t = a % b;
        a = b;
        b = t;
    }
    أرجع a;
}

ساكن مضمن منطقي lcm_اندازه(size_t a, size_t b, size_t & پایله) {
    إذا (a == 0 || b == 0) {
        پایله = 0;
        أرجع خطأ;
    }
    ثابت size_t g = لوی_ګډ_وېشونکی_اندازه(a, b);
    ثابت size_t q = a / g;
    إذا (q > SIZE_MAX / b) {
        أرجع خطأ;
    }
    پایله = q * b;
    أرجع ریښتیا;
}

ساكن مضمن size_t ceil_div_اندازه(size_t a, size_t b) {
    أرجع b == 0 ? 0 : (a + b - 1) / b;
}

ساكن مضمن size_t kleidiai_chunk_ستون_شمېر(size_t n, صحيح nth_ټول, منطقي disable_chunking, size_t n_ګام) {
    ثابت size_t multiplier = (nth_ټول == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ثابت size_t وېشونکی = std::max<size_t>(1, (size_t) nth_ټول * multiplier);
    ثابت size_t chunk_ستون_شمېر = align_پورته(std::max<size_t>(1, ceil_div_اندازه(n, وېشونکی)), n_ګام);
    أرجع chunk_ستون_شمېر ? chunk_ستون_شمېر : n_ګام;
}

بنية kleidiai_بلاک_دلیلونه {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

ساكن مضمن kleidiai_بلاک_دلیلونه kleidiai_get_بلاک_دلیلونه(ggml_type rhs_type) {
    بدل (rhs_type) {
        حالة GGML_TYPE_Q4_0:
            أرجع { QK4_0, QK4_0, QK4_0 };
        حالة GGML_TYPE_Q4_K:
            أرجع { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        حالة GGML_TYPE_Q8_0:
            أرجع { 0, 0, QK8_0 };
        افتراضي:
            أرجع { 0, 0, 0 };
    }
}

ساكن مضمن منطقي kleidiai_pack_fallback_allowed() {
    إذا (ctx.sme_thread_cap <= 0) {
        أرجع خطأ;
    }
    إذا (ctx.thread_hint <= 0) {
        أرجع ریښتیا;
    }
    أرجع ctx.thread_hint > ctx.sme_thread_cap;
}

بنية kleidiai_وزن_سرلیک {
    uint32_t magic;
    uint16_t نسخه_2;
    uint16_t خونه_شمېر;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

ساكن مضمن kleidiai_وزن_سرلیک * kleidiai_وزن_سرلیک_from_ptr(فراغ * معلومات) {
    أرجع reinterpret_cast<kleidiai_وزن_سرلیک *>(معلومات);
}

ساكن مضمن ثابت kleidiai_وزن_سرلیک * kleidiai_وزن_سرلیک_from_ptr(ثابت فراغ * معلومات) {
    أرجع reinterpret_cast<ثابت kleidiai_وزن_سرلیک *>(معلومات);
}

ساكن مضمن منطقي kleidiai_is_وزن_سرلیک_valid(ثابت kleidiai_وزن_سرلیک * سرلیک) {
    إذا (!سرلیک) {
        أرجع خطأ;
    }
    إذا (سرلیک->magic != GGML_KLEIDIAI_PACK_MAGIC || سرلیک->نسخه_2 != GGML_KLEIDIAI_PACK_نسخه) {
        أرجع خطأ;
    }
    إذا (سرلیک->خونه_شمېر == 0 || سرلیک->خونه_شمېر > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        أرجع خطأ;
    }
    أرجع ریښتیا;
}

ساكن مضمن uint8_t * kleidiai_وزن_خونه_ptr(kleidiai_وزن_سرلیک * سرلیک, صحيح خونه) {
    إذا (!kleidiai_is_وزن_سرلیک_valid(سرلیک)) {
        أرجع تش_اشاره;
    }
    إذا (خونه < 0 || خونه >= سرلیک->خونه_شمېر) {
        أرجع تش_اشاره;
    }
    أرجع reinterpret_cast<uint8_t *>(سرلیک) + سرلیک->offsets[خونه];
}

ساكن مضمن ثابت uint8_t * kleidiai_وزن_خونه_ptr(ثابت kleidiai_وزن_سرلیک * سرلیک, صحيح خونه) {
    إذا (!kleidiai_is_وزن_سرلیک_valid(سرلیک)) {
        أرجع تش_اشاره;
    }
    إذا (خونه < 0 || خونه >= سرلیک->خونه_شمېر) {
        أرجع تش_اشاره;
    }
    أرجع reinterpret_cast<ثابت uint8_t *>(سرلیک) + سرلیک->offsets[خونه];
}

ساكن مضمن ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    أرجع ctx.kernels_q4;
}

ساكن مضمن ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    أرجع ctx.kernels_q4k;
}

ساكن مضمن ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    أرجع ctx.kernels_q8;
}

ساكن مضمن ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    أرجع ctx.kernels_f32;
}

کينډۍ <ډول_نوم SelectFallback>
ساكن صحيح kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بهر,
        SelectFallback select_fallback) {
    صحيح count = 0;
    إذا (!primary) {
        أرجع 0;
    }
    بهر[count++] = primary;

    إذا (primary->rhs_info.repack_ډېرپېښ == RHS_REPACK_SINGLE_ONLY) {
        أرجع count;
    }

    إذا (is_sme_family(primary->required_cpu)) {
        ثابت cpu_feature fallback_ماسک = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        إذا (fallback_ماسک != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_ماسک);
            إذا (fallback && fallback != primary &&
                fallback->rhs_info.repack_ډېرپېښ != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                بهر[count++] = fallback;
            }
        }
    }

    أرجع count;
}

ساكن صحيح kleidiai_collect_kernel_chain(ثابت بنية ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بهر) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, بهر,
        [&](cpu_feature ماسک) { أرجع ggml_kleidiai_select_kernels(ماسک, op); });
}

ساكن صحيح kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بهر) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, بهر,
        [&](cpu_feature ماسک) { أرجع ggml_kleidiai_select_kernels_q4_0(ماسک); });
}

ساكن صحيح kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بهر) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, بهر,
        [&](cpu_feature ماسک) { أرجع ggml_kleidiai_select_kernels_q4_k(ماسک); });
}

ساكن صحيح kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بهر) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, بهر,
        [&](cpu_feature ماسک) { أرجع ggml_kleidiai_select_kernels_q8_0(ماسک); });
}

ساكن صحيح kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بهر) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, بهر,
        [&](cpu_feature ماسک) { أرجع ggml_kleidiai_select_kernels_f32(ماسک); });
}

ساكن مضمن int64_t ggml_ne(ثابت ggml_tensor * tensor, صحيح dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    أرجع tensor->ne[dim];
}

نطاق ggml::cpu::kleidiai {

ساكن size_t پړاو_ښکته(size_t x, size_t y) {
    أرجع y == 0 ? x : x - (x % y);
}

ساكن فراغ transpose_f32kxn_f16nxk(size_t n, size_t k, عائم * dst, ثابت uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / حجم(uint16_t);
    size_t dst_stride = n;

    لكل (size_t k_idx = 0; k_idx < k; ++k_idx) {
        لكل (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

صنف tensor_traits : عام ggml::cpu::tensor_traits {
    منطقي work_اندازه(صحيح /* n_threads */, ثابت بنية ggml_tensor * op, size_t & size) override {
        إذا (op->op != GGML_OP_MUL_MAT) {
            أرجع خطأ;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحيح خونه_شمېر = kleidiai_collect_kernel_chain(op, kernel_chain);
        إذا (خونه_شمېر == 0) {
            أرجع خطأ;
        }

        ثابت منطقي is_gemv = op->src[1]->ne[1] == 1;
        ثابت size_t k = op->src[0]->ne[0];
        ثابت size_t n = op->src[0]->ne[1];
        ثابت size_t m = op->src[1]->ne[1];

        إذا (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ثابت kleidiai_بلاک_دلیلونه بلاک_دلیلونه = kleidiai_get_بلاک_دلیلونه(kernel_chain[0]->rhs_type);
            ثابت size_t qk = بلاک_دلیلونه.pack_bl;

            size_t cursor = 0;
            منطقي any_خونه = خطأ;

            لكل (صحيح خونه = 0; خونه < خونه_شمېر; ++خونه) {
                ggml_kleidiai_kernels * kernels = kernel_chain[خونه];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                إذا (!lhs_info || !lhs_info->packed_اندازه_ex || !kernel) {
                    أرجع خطأ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                ثابت size_t packed = lhs_info->packed_اندازه_ex(m, k, qk, mr, kr, sr);

                cursor = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_خونه = ریښتیا;
            }

            إذا (!any_خونه) {
                أرجع خطأ;
            }

            size = cursor;
            أرجع ریښتیا;
        }

        إذا (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : تش_اشاره;
            إذا (is_gemv && op->src[1]->nb[0] == (int64_t) حجم(عائم) && gemv_kernel &&
                gemv_kernel->get_lhs_دننه_واټن_ex && gemv_kernel->get_rhs_packed_دننه_واټن_ex &&
                gemv_kernel->پرلپسې_kernel_ex && gemv_kernel->get_dst_دننه_واټن) {
                size = 0;
                أرجع ریښتیا;
            }

            size_t cursor = 0;
            منطقي any_خونه = خطأ;

            لكل (صحيح خونه = 0; خونه < خونه_شمېر; ++خونه) {
                ggml_kleidiai_kernels * kernels = kernel_chain[خونه];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                إذا (!lhs_info || !lhs_info->packed_اندازه_ex || !kernel) {
                    أرجع خطأ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_اندازه_ex(m, k, 0, mr, kr, sr);
                any_خونه = ریښتیا;
            }

            إذا (!any_خونه) {
                أرجع خطأ;
            }

            size = cursor;
            أرجع ریښتیا;
        }

        إذا (op->src[0]->type == GGML_TYPE_F16) {
            ثابت int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ثابت int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ثابت int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            منطقي any_خونه = خطأ;

            لكل (صحيح خونه = 0; خونه < خونه_شمېر; ++خونه) {
                ggml_kleidiai_kernels * kernels = kernel_chain[خونه];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                إذا (!lhs_info || !lhs_info->packed_اندازه_ex || !kernels->rhs_info.packed_اندازه_ex || !kernel) {
                    أرجع خطأ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_اندازه_ex(m * r, k, 0, mr, kr, sr);
                any_خونه = ریښتیا;
            }

            لكل (صحيح خونه = 0; خونه < خونه_شمېر; ++خونه) {
                ggml_kleidiai_kernels * kernels = kernel_chain[خونه];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                إذا (!kernel || !kernels->rhs_info.packed_اندازه_ex) {
                    أرجع خطأ;
                }
                cursor  = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_اندازه_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * حجم(عائم);
            cursor  = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * حجم(عائم);

            إذا (!any_خونه) {
                أرجع خطأ;
            }

            size = cursor;
            أرجع ریښتیا;
        }

        أرجع خطأ;
    }

    منطقي compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * dst) override {
        إذا (dst->op == GGML_OP_MUL_MAT) {
            إذا (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                أرجع compute_forward_qx(params, dst);
            } وإلا إذا (dst->src[0]->type == GGML_TYPE_F32) {
                أرجع compute_forward_f32(params, dst);
            } وإلا إذا (dst->src[0]->type == GGML_TYPE_F16) {
                أرجع compute_forward_fp16(params, dst);
            }
        }
        أرجع خطأ;
    }

    منطقي compute_forward_f32(ggml_compute_params * params, بنية ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_دوګونی_OP_LOCALS

        إذا (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        إذا (!kernels) {
            أرجع خطأ;
        }

        ثابت size_t k = ne00;
        ثابت size_t m = ne11;
        ثابت size_t n = ne01;
        ثابت منطقي use_gemv = m == 1 && src1->nb[0] == (int64_t) حجم(عائم) &&
                              kernels->gemv.get_lhs_دننه_واټن_ex &&
                              kernels->gemv.get_rhs_packed_دننه_واټن_ex &&
                              kernels->gemv.پرلپسې_kernel_ex &&
                              kernels->gemv.get_dst_دننه_واټن;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        إذا (!kernel || !kernel->get_lhs_دننه_واټن_ex ||
            !kernel->get_rhs_packed_دننه_واټن_ex || !kernel->پرلپسې_kernel_ex || !kernel->get_dst_دننه_واټن) {
            أرجع خطأ;
        }

        إذا (!use_gemv && (!lhs_info || !lhs_info->get_دننه_واټن || !lhs_info->get_packed_دننه_واټن_ex ||
                          !lhs_info->packed_اندازه_ex || !lhs_info->pack_func_ex)) {
            أرجع خطأ;
        }

        ثابت kleidiai_وزن_سرلیک * سرلیک = kleidiai_وزن_سرلیک_from_ptr(src0->معلومات);
        ثابت منطقي has_سرلیک = kleidiai_is_وزن_سرلیک_valid(سرلیک);

        ثابت uint8_t * rhs_بنسټ = has_سرلیک ? kleidiai_وزن_خونه_ptr(سرلیک, 0)
                                              : static_cast<ثابت uint8_t *>(src0->معلومات);
        إذا (!rhs_بنسټ) {
            أرجع خطأ;
        }

        ثابت صحيح nth = params->nth > 0 ? params->nth : 1;
        ثابت صحيح ith = params->ith;

        ثابت size_t mr = kernel->get_mr();
        ثابت size_t kr = kernel->get_kr();
        ثابت size_t sr = kernel->get_sr();

        ثابت size_t lhs_packed_اندازه = use_gemv ? 0 : lhs_info->packed_اندازه_ex(m, k, 0, mr, kr, sr);
        إذا (!use_gemv) {
            GGML_ASSERT(lhs_packed_اندازه <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ثابت size_t dst_stride = dst->nb[1];
        ثابت size_t n_ګام = kernel->get_n_ګام() ? kernel->get_n_ګام() : 1;
        ثابت منطقي disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) لوی_عدد);

        ثابت int64_t batch_اندازه = ne12 * ne13;
        لكل (int64_t batch_idx = 0; batch_idx < batch_اندازه; ++batch_idx) {
            ثابت int64_t i12 = batch_idx % ne12;
            ثابت int64_t i13 = batch_idx / ne12;
            ثابت uint8_t * lhs_batch_بنسټ = static_cast<ثابت uint8_t *>(src1->معلومات) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_بنسټ = static_cast<uint8_t *>(dst->معلومات) + i12 * dst->nb[2] + i13 * dst->nb[3];

            إذا (!use_gemv) {
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_تارونه = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_تارونه = std::max<int64_t>(1, max_تارونه);
                ثابت int64_t use_تارونه = std::min<int64_t>(nth, max_تارونه);

                إذا (ith < use_تارونه) {
                    ثابت int64_t عدد_m_فی_thread0   = پړاو_ښکته((size_t)(m_roundup_mr / use_تارونه), mr);
                    ثابت int64_t عدد_m_فی_thread_N_1 = (int64_t)m - (use_تارونه - 1) * عدد_m_فی_thread0;

                    ثابت int64_t m_پیل = (int64_t)ith * عدد_m_فی_thread0;
                    ثابت int64_t m_شمېر = (ith == use_تارونه - 1) ? عدد_m_فی_thread_N_1 : عدد_m_فی_thread0;

                    ثابت size_t بنسټ_packed_off  = lhs_info->get_packed_دننه_واټن_ex(m_پیل, k, 0, mr, kr, sr);
                    ثابت size_t بل_بلاک_off   = lhs_info->get_packed_دننه_واټن_ex(m_پیل + mr, k, 0, mr, kr, sr);
                    ثابت size_t لیکه_stride_بایټونه = mr ? (بل_بلاک_off - بنسټ_packed_off) / mr : 0;

                    int64_t پاتې_شوی = m_شمېر;
                    int64_t cur       = m_پیل;

                    طالما (پاتې_شوی > 0) {
                        ثابت int64_t take = std::min<int64_t>((int64_t)m - cur, پاتې_شوی);
                        ثابت size_t src_off = lhs_info->get_دننه_واټن(cur, src1->nb[1]);
                        ثابت فراغ * src_ptr = lhs_batch_بنسټ + src_off;
                        ثابت size_t dst_off = بنسټ_packed_off + (size_t)(cur - m_پیل) * لیکه_stride_بایټونه;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        پاتې_شوی -= take;
                    }
                }
            }

            إذا (ith == 0) {
                ggml_threadpool_chunk_سېټ(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ثابت size_t chunk_ستون_شمېر = kleidiai_chunk_ستون_شمېر(n, nth, disable_chunking, n_ګام);
            GGML_ASSERT(chunk_ستون_شمېر <= (size_t) لوی_عدد);

            صحيح current_ستون = ggml_threadpool_chunk_جمع_کړه(params->threadpool, (صحيح) chunk_ستون_شمېر);
            طالما ((size_t) current_ستون < n) {
                ثابت size_t n_پیل = (size_t) current_ستون;
                ثابت size_t n_منزل_process = std::min(chunk_ستون_شمېر, n - n_پیل);

                إذا (n_منزل_process > 0) {
                    ثابت size_t lhs_دننه_واټن = use_gemv ? kernel->get_lhs_دننه_واټن_ex(0, k, 0)
                                                       : lhs_info->get_packed_دننه_واټن_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_دننه_واټن = kernel->get_rhs_packed_دننه_واټن_ex(n_پیل, k, 0);
                    ثابت size_t dst_دننه_واټن        = kernel->get_dst_دننه_واټن(0, n_پیل, dst_stride);

                    ثابت فراغ * lhs_ptr = use_gemv ? lhs_batch_بنسټ + lhs_دننه_واټن
                                                    : lhs_packed + lhs_دننه_واټن;
                    ثابت فراغ * rhs_ptr = rhs_بنسټ + rhs_packed_دننه_واټن;
                    عائم * dst_ptr      = reinterpret_cast<عائم *>(dst_batch_بنسټ + dst_دننه_واټن);

                    kernel->پرلپسې_kernel_ex(m, n_منزل_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          حجم(عائم),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_ستون = ggml_threadpool_chunk_جمع_کړه(params->threadpool, (صحيح) chunk_ستون_شمېر);
            }

            إذا (batch_idx != batch_اندازه - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        أرجع ریښتیا;
    }

    منطقي compute_forward_fp16(ggml_compute_params * params, بنية ggml_tensor * dst) {
        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_دوګونی_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        إذا (!kernels) {
            أرجع خطأ;
        }

        ثابت منطقي is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        إذا (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_دننه_واټن_ex || !kernel->get_rhs_packed_دننه_واټن_ex || !kernel->پرلپسې_kernel_ex) {
            أرجع خطأ;
        }

        ثابت صحيح nth = params->nth;
        ثابت صحيح ith = params->ith;

        ثابت int64_t lhs_batch_size0 = ne12;
        ثابت int64_t rhs_batch_size0 = ne02;
        ثابت int64_t batch_اندازه      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        ثابت int64_t r = lhs_batch_size0 / rhs_batch_size0;

        ثابت int64_t m_ډله = ne11;
        ثابت int64_t m       = m_ډله;
        ثابت int64_t n       = ne01;
        ثابت int64_t k       = ne00;

        ثابت size_t lhs_stride = src1->nb[1];
        ثابت size_t rhs_stride = src0->nb[1];
        ثابت size_t dst_stride = dst->nb[1];

        ثابت int64_t mr = (int64_t) kernel->get_mr();
        ثابت int64_t nr = (int64_t) kernel->get_nr();
        ثابت int64_t kr = (int64_t) kernel->get_kr();
        ثابت int64_t sr = (int64_t) kernel->get_sr();

        ثابت size_t lhs_packed_اندازه = lhs_info->packed_اندازه_ex(m, k, 0, mr, kr, sr);
        ثابت size_t rhs_packed_اندازه = kernels->rhs_info.packed_اندازه_ex(n, k, nr, kr, 0);
        ثابت size_t kxn_اندازه        = k * n * حجم(عائم);
        ثابت size_t bias_اندازه       = n * حجم(عائم);

        ثابت size_t wsize_required = lhs_packed_اندازه + rhs_packed_اندازه + kxn_اندازه + bias_اندازه;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_اندازه;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_اندازه;
        uint8_t * bias       = rhs_kxn + kxn_اندازه;

        لكل (int64_t batch_idx = 0; batch_idx < batch_اندازه; ++batch_idx) {
            ثابت int64_t rhs_batch_idx = batch_idx / r;
            ثابت uint8_t * rhs_batch_بنسټ = static_cast<ثابت uint8_t *>(src0->معلومات) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_بنسټ = static_cast<uint8_t *>(dst->معلومات) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ثابت int64_t m_roundup_mr = kai_roundup(m, mr);
                ثابت int64_t عدد_تارونه  = KAI_MIN(m_roundup_mr / mr, nth);

                إذا (ith < عدد_تارونه) {
                    ثابت int64_t عدد_m_فی_thread0   = پړاو_ښکته((size_t)(m_roundup_mr / عدد_تارونه), (size_t)mr);
                    ثابت int64_t عدد_m_فی_thread_N_1 = m - (عدد_تارونه - 1) * عدد_m_فی_thread0;

                    ثابت int64_t m_پیل = ith * عدد_m_فی_thread0;
                    ثابت int64_t m_شمېر = (ith == عدد_تارونه - 1) ? عدد_m_فی_thread_N_1 : عدد_m_فی_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ثابت size_t بنسټ_packed_off  = lhs_info->get_packed_دننه_واټن_ex(m_پیل, k, 0, mr, kr, sr);
                    ثابت size_t بل_بلاک_off   = lhs_info->get_packed_دننه_واټن_ex(m_پیل + mr, k, 0, mr, kr, sr);
                    ثابت size_t لیکه_stride_بایټونه = (بل_بلاک_off - بنسټ_packed_off) / (size_t)mr;

                    int64_t پاتې_شوی = m_شمېر;
                    int64_t cur       = m_پیل;

                    طالما (پاتې_شوی > 0) {
                        ثابت int64_t لیکه_دننه_ډله = cur;
                        ثابت int64_t avail        = m_ډله - لیکه_دننه_ډله;
                        ثابت int64_t take         = std::min(avail, پاتې_شوی);

                        ثابت uint8_t * lhs_batch_بنسټ = static_cast<ثابت uint8_t *>(src1->معلومات) + batch_idx * src1->nb[2];
                        ثابت فراغ * src_ptr = lhs_batch_بنسټ + (size_t)لیکه_دننه_ډله * lhs_stride;
                        ثابت size_t dst_off = بنسټ_packed_off + (size_t)(cur - m_پیل) * لیکه_stride_بایټونه;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        پاتې_شوی -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            إذا (ith == 0) {
                memset(bias, 0, (size_t)n * حجم(عائم));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<عائم *>(rhs_kxn),
                                        reinterpret_cast<ثابت uint16_t *>(rhs_batch_بنسټ),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * حجم(عائم),
                             rhs_kxn, bias, تش_اشاره, rhs_packed, 0, تش_اشاره);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ثابت int64_t n_ګام  = (int64_t) kernel->get_n_ګام();
                int64_t عدد_تارونه_n = KAI_MIN(n / n_ګام, nth);
                إذا (عدد_تارونه_n <= 0) {
                    عدد_تارونه_n = 1;
                }

                إذا (ith < عدد_تارونه_n) {
                    ثابت int64_t عدد_n_فی_thread0   = پړاو_ښکته((size_t)(n / عدد_تارونه_n), (size_t)n_ګام);
                    ثابت int64_t عدد_n_فی_thread_N_1 = n - (عدد_تارونه_n - 1) * عدد_n_فی_thread0;

                    ثابت int64_t n_پیل      = ith * عدد_n_فی_thread0;
                    ثابت int64_t n_منزل_process = (ith == عدد_تارونه_n - 1) ? عدد_n_فی_thread_N_1 : عدد_n_فی_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ثابت size_t lhs_packed_offset0 = lhs_info->get_packed_دننه_واټن_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_دننه_واټن  = kernel->get_rhs_packed_دننه_واټن_ex(n_پیل, k, 0);
                    ثابت size_t dst_دننه_واټن         = kernel->get_dst_دننه_واټن((size_t)0, (size_t)n_پیل, dst_stride);

                    ثابت فراغ * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ثابت فراغ * rhs_ptr = rhs_packed + rhs_packed_دننه_واټن;
                    عائم * dst_ptr      = reinterpret_cast<عائم *>(dst_batch_بنسټ + dst_دننه_واټن);

                    kernel->پرلپسې_kernel_ex(m, n_منزل_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, حجم(عائم), -FLT_MAX, FLT_MAX);
                }
            }

            إذا (batch_idx != batch_اندازه - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        أرجع ریښتیا;
    }

    منطقي compute_forward_qx(بنية ggml_compute_params * params, بنية ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_دوګونی_OP_LOCALS

        ثابت kleidiai_وزن_سرلیک * سرلیک = kleidiai_وزن_سرلیک_from_ptr(src0->معلومات);
        ثابت منطقي has_سرلیک = kleidiai_is_وزن_سرلیک_valid(سرلیک);
        ثابت منطقي is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحيح خونه_ټول = kleidiai_collect_kernel_chain(dst, kernel_chain);

        خپلکاری وزن_for_خونه = [&](صحيح خونه_شاخص, size_t & اندازه_بهر) -> ثابت uint8_t * {
            إذا (خونه_شاخص < 0 || خونه_شاخص >= خونه_ټول) {
                أرجع تش_اشاره;
            }
            إذا (has_سرلیک) {
                إذا (خونه_شاخص < سرلیک->خونه_شمېر) {
                    اندازه_بهر = static_cast<size_t>(سرلیک->sizes[خونه_شاخص]);
                    أرجع kleidiai_وزن_خونه_ptr(سرلیک, خونه_شاخص);
                }
                أرجع تش_اشاره;
            }
            إذا (خونه_شاخص == 0) {
                اندازه_بهر = ggml_nbytes(src0);
                أرجع static_cast<ثابت uint8_t *>(src0->معلومات);
            }
            أرجع تش_اشاره;
        };

        بنية runtime_خونه {
            صحيح خونه_شاخص;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            size_t mr;
            size_t nr;
            size_t kr;
            size_t sr;
            size_t n_ګام;
            size_t lhs_packed_اندازه;
            size_t lhs_دننه_واټن;
            size_t lhs_bl;
            size_t rhs_bl;
            size_t pack_bl;
            size_t lhs_packed_offset0;
            صحيح assigned_تارونه;
            صحيح thread_begin;
            صحيح thread_پای_ټکی;
            ثابت uint8_t * rhs_بنسټ;
        };

        std::array<runtime_خونه, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        صحيح runtime_شمېر = 0;

        لكل (صحيح خونه = 0; خونه < خونه_ټول && runtime_شمېر < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++خونه) {
            ggml_kleidiai_kernels * kernels = kernel_chain[خونه];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            إذا (!kinfo || !linfo || !linfo->packed_اندازه_ex || !linfo->pack_func_ex || !linfo->get_دننه_واټن ||
                !kinfo->get_rhs_packed_دننه_واټن_ex || !kinfo->پرلپسې_kernel_ex || !kinfo->get_dst_دننه_واټن) {
                استمر;
            }

            size_t rhs_اندازه = 0;
            ثابت uint8_t * rhs_ptr = وزن_for_خونه(خونه, rhs_اندازه);
            إذا (!rhs_ptr || rhs_اندازه == 0) {
                استمر;
            }

            ثابت kleidiai_بلاک_دلیلونه بلاک_دلیلونه = kleidiai_get_بلاک_دلیلونه(kernels->rhs_type);

            runtime[runtime_شمېر] = {
                خونه,
                kernels,
                kinfo,
                linfo,
                kinfo->get_mr(),
                kinfo->get_nr(),
                kinfo->get_kr(),
                kinfo->get_sr(),
                kinfo->get_n_ګام(),
                0,
                0,
                بلاک_دلیلونه.lhs_bl,
                بلاک_دلیلونه.rhs_bl,
                بلاک_دلیلونه.pack_bl,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_شمېر;
        }

        إذا (runtime_شمېر == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            أرجع خطأ;
        }

        ثابت صحيح nth_ټول = params->nth > 0 ? params->nth : 1;
        ثابت صحيح ith_ټول = params->ith;

        صحيح sme_خونه = -1;
        صحيح non_sme_خونه = -1;
        لكل (صحيح i = 0; i < runtime_شمېر; ++i) {
            إذا (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_خونه = i;
                اكسر;
            }
        }

        لكل (صحيح i = 0; i < runtime_شمېر; ++i) {
            إذا (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_خونه = i;
                اكسر;
            }
        }

        ثابت صحيح sme_cap_حد = ctx.sme_thread_cap;
        ثابت منطقي use_hybrid = sme_cap_حد > 0 &&
                                 runtime_شمېر > 1 &&
                                 nth_ټول > sme_cap_حد;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_ستون_شمېر_فی_thread = 0;
        إذا (runtime_شمېر > 0 && nth_ټول > 0) {
            min_ستون_شمېر_فی_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_ټول);
        }
        ثابت منطقي too_small_for_hybrid = (min_ستون_شمېر_فی_thread < 2) || (ne11 < 128);

        ثابت منطقي hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        إذا (!hybrid_enabled) {
            صحيح chosen_خونه = 0;
            إذا (too_small_for_hybrid && sme_خونه != -1) {
                chosen_خونه = nth_ټول > sme_cap_حد && non_sme_خونه != -1 ? non_sme_خونه : sme_خونه;
            } وإلا إذا (runtime_شمېر > 1 && ctx.sme_thread_cap > 0 && nth_ټول > ctx.sme_thread_cap) {
                chosen_خونه = 1;
            }
            إذا (chosen_خونه != 0 && chosen_خونه < runtime_شمېر) {
                runtime[0] = runtime[chosen_خونه];
                runtime[0].assigned_تارونه = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_پای_ټکی = 0;
            }
            runtime_شمېر = runtime_شمېر > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_خونه = -1;
            إذا (runtime_شمېر > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_خونه = 0;
            }
        }

        ثابت منطقي sme_single_only = sme_خونه != -1 &&
                                     runtime[sme_خونه].kernels->rhs_info.repack_ډېرپېښ == RHS_REPACK_SINGLE_ONLY;
        صحيح sme_cap = sme_single_only ? nth_ټول : kleidiai_sme_thread_cap();
        إذا (sme_cap < 0) {
            sme_cap = nth_ټول;
        }
        sme_cap = std::min(sme_cap, nth_ټول);

        صحيح تارونه_پاتې_شوی = nth_ټول;
        إذا (sme_خونه != -1) {
            صحيح sme_تارونه = std::min(std::max(sme_cap, 0), تارونه_پاتې_شوی);
            runtime[sme_خونه].assigned_تارونه = sme_تارونه;
            تارونه_پاتې_شوی -= sme_تارونه;
        }

        صحيح fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        صحيح fallback_شمېر = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        لكل (صحيح i = 0; i < runtime_شمېر; ++i) {
            إذا (i == sme_خونه) {
                استمر;
            }
            fallback_indices[fallback_شمېر++] = i;
        }

        لكل (صحيح fi = 0; fi < fallback_شمېر; ++fi) {
            إذا (تارونه_پاتې_شوی <= 0) {
                اكسر;
            }
            ثابت صحيح خونه_شاخص = fallback_indices[fi];
            ثابت صحيح slots_کیڼ = fallback_شمېر - fi;
            صحيح share = (تارونه_پاتې_شوی + slots_کیڼ - 1) / slots_کیڼ;
            share     = std::min(share, تارونه_پاتې_شوی);
            runtime[خونه_شاخص].assigned_تارونه = share;
            تارونه_پاتې_شوی -= share;
        }

        صحيح thread_cursor = 0;
        لكل (صحيح i = 0; i < runtime_شمېر; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_تارونه;
            runtime[i].thread_پای_ټکی = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ثابت صحيح فعال_تارونه = thread_cursor;

        صحيح local_خونه = -1;
        صحيح local_ith  = 0;
        لكل (صحيح i = 0; i < runtime_شمېر; ++i) {
            إذا (ith_ټول >= runtime[i].thread_begin && ith_ټول < runtime[i].thread_پای_ټکی) {
                local_خونه = i;
                local_ith  = ith_ټول - runtime[i].thread_begin;
                اكسر;
            }
        }
        ثابت size_t k = ne00;
        ثابت size_t m = ne11;
        ثابت size_t n = ne01;

        size_t cursor = 0;
        لكل (صحيح i = 0; i < runtime_شمېر; ++i) {
            runtime[i].lhs_packed_اندازه = runtime[i].lhs_info->packed_اندازه_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_دننه_واټن = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_دننه_واټن_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_اندازه;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_ګام = 1;
        لكل (صحيح i = 0; i < runtime_شمېر; ++i) {
            إذا (runtime[i].assigned_تارونه == 0) {
                استمر;
            }
            size_t بل_ګام = 0;
            إذا (!lcm_اندازه(common_ګام, runtime[i].n_ګام ? runtime[i].n_ګام : 1, بل_ګام)) {
                أرجع خطأ;
            }
            common_ګام = بل_ګام;
        }
        GGML_ASSERT(common_ګام > 0);

        ثابت منطقي disable_chunking = ggml_is_numa();
        ثابت size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ثابت size_t chunk_وېشونکی = (فعال_تارونه == 1 || disable_chunking) ? (size_t) فعال_تارونه : (size_t) فعال_تارونه * chunk_multiplier;
        size_t chunk_ستون_شمېر = align_پورته(std::max<size_t>(1, ceil_div_اندازه(n, chunk_وېشونکی)), common_ګام);
        إذا (chunk_ستون_شمېر == 0) {
            chunk_ستون_شمېر = common_ګام;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ثابت size_t nchunk_اندازه = std::max<size_t>(1, ceil_div_اندازه(n, chunk_ستون_شمېر));
        GGML_ASSERT(nchunk_اندازه <= (size_t)لوی_عدد);
        ثابت صحيح nchunk = (صحيح)nchunk_اندازه;
        ثابت size_t dst_stride = dst->nb[1];

        خپلکاری پرلپسې_chunk = [&](runtime_خونه & خونه, size_t global_پیل, size_t ستون_شمېر_2, uint8_t * dst_batch_بنسټ) {
            ثابت size_t rhs_packed_دننه_واټن = خونه.kernel->get_rhs_packed_دننه_واټن_ex(global_پیل, k, خونه.rhs_bl);
            ثابت size_t dst_دننه_واټن        = خونه.kernel->get_dst_دننه_واټن(0, global_پیل, dst_stride);

            ثابت uint8_t * lhs_ptr = scratch + خونه.lhs_دننه_واټن + خونه.lhs_packed_offset0;
            ثابت uint8_t * rhs_ptr = خونه.rhs_بنسټ + rhs_packed_دننه_واټن;
            عائم * dst_ptr         = reinterpret_cast<عائم *>(dst_batch_بنسټ + dst_دننه_واټن);

            خونه.kernel->پرلپسې_kernel_ex(m, ستون_شمېر_2, k, خونه.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       حجم(عائم),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        لكل (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ثابت uint8_t * lhs_batch_بنسټ = static_cast<ثابت uint8_t *>(src1->معلومات) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_بنسټ = static_cast<uint8_t *>(dst->معلومات) + batch_idx * dst->nb[2];

            إذا (local_خونه != -1) {
                runtime_خونه & خونه = runtime[local_خونه];
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)خونه.mr);
                int64_t max_تارونه = خونه.mr ? (m_roundup_mr / (int64_t)خونه.mr) : خونه.assigned_تارونه;
                max_تارونه = std::max<int64_t>(1, max_تارونه);
                ثابت int64_t use_تارونه = std::min<int64_t>(خونه.assigned_تارونه, max_تارونه);

                إذا (local_ith < use_تارونه) {
                    ثابت int64_t عدد_m_فی_thread0   = پړاو_ښکته((size_t)(m_roundup_mr / use_تارونه), خونه.mr);
                    ثابت int64_t عدد_m_فی_thread_N_1 = (int64_t)m - (use_تارونه - 1) * عدد_m_فی_thread0;

                    ثابت int64_t m_پیل = (int64_t)local_ith * عدد_m_فی_thread0;
                    ثابت int64_t m_شمېر = (local_ith == use_تارونه - 1) ? عدد_m_فی_thread_N_1 : عدد_m_فی_thread0;

                    ثابت size_t بنسټ_packed_off  = خونه.lhs_info->get_packed_دننه_واټن_ex(m_پیل, k, خونه.lhs_bl, خونه.mr, خونه.kr, خونه.sr);
                    ثابت size_t بل_بلاک_off   = خونه.lhs_info->get_packed_دننه_واټن_ex(m_پیل + خونه.mr, k, خونه.lhs_bl, خونه.mr, خونه.kr, خونه.sr);
                    ثابت size_t لیکه_stride_بایټونه = خونه.mr ? (بل_بلاک_off - بنسټ_packed_off) / خونه.mr : 0;

                    int64_t پاتې_شوی = m_شمېر;
                    int64_t cur       = m_پیل;

                    uint8_t * lhs_packed = scratch + خونه.lhs_دننه_واټن;
                    طالما (پاتې_شوی > 0) {
                        ثابت int64_t لیکه_دننه_ډله = cur;
                        ثابت int64_t avail        = (int64_t)m - لیکه_دننه_ډله;
                        ثابت int64_t take         = std::min(avail, پاتې_شوی);

                        ثابت size_t src_off = خونه.lhs_info->get_دننه_واټن(لیکه_دننه_ډله, src1->nb[1]);
                        ثابت فراغ * src_ptr = lhs_batch_بنسټ + src_off;
                        ثابت size_t dst_off = بنسټ_packed_off + (size_t)(cur - m_پیل) * لیکه_stride_بایټونه;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        خونه.lhs_info->pack_func_ex(take, k, خونه.lhs_bl, خونه.mr, خونه.kr, خونه.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        پاتې_شوی -= take;
                    }
                }
            }

            إذا (ith_ټول == 0) {
                ggml_threadpool_chunk_سېټ(params->threadpool, فعال_تارونه);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            إذا (local_خونه != -1) {
                runtime_خونه & خونه = runtime[local_خونه];
                صحيح current_chunk = ith_ټول;
                طالما (current_chunk < nchunk) {
                    ثابت size_t global_پیل = (size_t)current_chunk * chunk_ستون_شمېر;
                    إذا (global_پیل >= n) {
                        اكسر;
                    }

                    ثابت size_t ستون_شمېر_2 = std::min(chunk_ستون_شمېر, n - global_پیل);
                    إذا (ستون_شمېر_2 > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        پرلپسې_chunk(خونه, global_پیل, ستون_شمېر_2, dst_batch_بنسټ);
                    }

                    current_chunk = ggml_threadpool_chunk_جمع_کړه(params->threadpool, 1);
                }
            }

            إذا (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        أرجع ریښتیا;
    }

عام:
    صحيح repack(بنية ggml_tensor * tensor, ثابت فراغ * معلومات, size_t معلومات_اندازه) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ثابت size_t n = tensor->ne[1];
        ثابت size_t k = tensor->ne[0];

        kleidiai_وزن_سرلیک * سرلیک = kleidiai_وزن_سرلیک_from_ptr(tensor->معلومات);
        إذا (!سرلیک) {
            أرجع -1;
        }

        سرلیک->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        سرلیک->نسخه_2    = GGML_KLEIDIAI_PACK_نسخه;
        سرلیک->خونه_شمېر = 0;

        uint8_t * بنسټ_ptr = static_cast<uint8_t *>(tensor->معلومات);
        size_t cursor = حجم(kleidiai_وزن_سرلیک);
        cursor = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت منطقي want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ثابت منطقي want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ثابت منطقي want_f32 = tensor->type == GGML_TYPE_F32;
        ثابت صحيح خونه_ټول = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ثابت منطقي allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<عائم>  scales;
        std::vector<عائم>  bias;
        std::vector<عائم> لیکه_f32;

        إذا (want_q8 && خونه_ټول > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ثابت size_t لیکه_stride = tensor->nb[1];
            ثابت size_t k_بلاکونه   = (k + QK8_0 - 1) / QK8_0;

            إذا (tensor->type == GGML_TYPE_Q6_K) {
                إذا (k % QK_K != 0 || k % QK8_0 != 0) {
                    أرجع -1;
                }
                لیکه_f32.resize(k);
            }

            لكل (size_t لیکه = 0; لیکه < n; ++لیکه) {
                ثابت uint8_t * لیکه_معلومات = static_cast<ثابت uint8_t *>(معلومات) + لیکه * لیکه_stride;

                إذا (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_لیکه_q6_K(reinterpret_cast<ثابت بلاک_q6_K *>(لیکه_معلومات), لیکه_f32.معلومات(), k);
                }

                عائم max_abs = 0.0f;
                إذا (tensor->type == GGML_TYPE_Q6_K) {
                    لكل (size_t خطي_idx = 0; خطي_idx < k; ++خطي_idx) {
                        max_abs = std::max(max_abs, std::fabs(لیکه_f32[خطي_idx]));
                    }
                } وإلا {
                    ثابت بلاک_q8_0 * لیکه_بلاکونه = reinterpret_cast<ثابت بلاک_q8_0 *>(لیکه_معلومات);
                    لكل (size_t بلاک = 0; بلاک < k_بلاکونه; ++بلاک) {
                        ثابت بلاک_q8_0 & blk = لیکه_بلاکونه[بلاک];
                        ثابت عائم d = GGML_FP16_منزل_FP32(blk.d);
                        لكل (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t خطي_idx = بلاک * QK8_0 + l;
                            إذا (خطي_idx >= k) {
                                اكسر;
                            }
                            ثابت عائم value = d * static_cast<عائم>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                عائم scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[لیکه] = scale;
                ثابت عائم inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                إذا (tensor->type == GGML_TYPE_Q6_K) {
                    لكل (size_t خطي_idx = 0; خطي_idx < k; ++خطي_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(لیکه_f32[خطي_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[لیکه * k + خطي_idx] = static_cast<int8_t>(q);
                    }
                } وإلا {
                    ثابت بلاک_q8_0 * لیکه_بلاکونه = reinterpret_cast<ثابت بلاک_q8_0 *>(لیکه_معلومات);
                    لكل (size_t بلاک = 0; بلاک < k_بلاکونه; ++بلاک) {
                        ثابت بلاک_q8_0 & blk = لیکه_بلاکونه[بلاک];
                        ثابت عائم d = GGML_FP16_منزل_FP32(blk.d);
                        لكل (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t خطي_idx = بلاک * QK8_0 + l;
                            إذا (خطي_idx >= k) {
                                اكسر;
                            }
                            ثابت عائم value = d * static_cast<عائم>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[لیکه * k + خطي_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        إذا (want_f32 && خونه_ټول > 0) {
            bias.resize(n, 0.0f);
        }

        لكل (صحيح خونه = 0; خونه < خونه_ټول && خونه < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++خونه) {
            إذا (!allow_fallback && خونه > 0) {
                اكسر;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[خونه];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            إذا (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_اندازه_ex || !kernel) {
                استمر;
            }

            ثابت size_t nr = kernel->get_nr();
            ثابت size_t kr = kernel->get_kr();
            ثابت size_t sr = kernel->get_sr();
            ثابت ggml_type rhs_type = kernels->rhs_type;
            إذا (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            ثابت size_t بلاک_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            إذا (بلاک_len == SIZE_MAX) {
                استمر;
            }

            ثابت size_t packed_اندازه = rhs_info->packed_اندازه_ex(n, k, nr, kr, بلاک_len);
            ثابت size_t aligned_cursor = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = بنسټ_ptr + aligned_cursor;

            إذا (rhs_type == GGML_TYPE_Q4_0) {
                بنية kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_ټکی = 1;
                params.rhs_zero_ټکی = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ثابت uint8_t *>(معلومات), تش_اشاره, تش_اشاره,
                                       dst_ptr, 0, &params);
            } وإلا إذا (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       معلومات, تش_اشاره, تش_اشاره, dst_ptr, 0, تش_اشاره);
            } وإلا إذا (rhs_type == GGML_TYPE_Q8_0) {
                بنية kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_ټکی = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.معلومات(), تش_اشاره, scales.معلومات(),
                                       dst_ptr, 0, &params);
            } وإلا إذا (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       معلومات, bias.معلومات(), تش_اشاره,
                                       dst_ptr, 0, تش_اشاره);
            } وإلا {
                استمر;
            }

            سرلیک->offsets[سرلیک->خونه_شمېر] = aligned_cursor;
            سرلیک->sizes[سرلیک->خونه_شمېر]   = packed_اندازه;
            ++سرلیک->خونه_شمېر;

            cursor = aligned_cursor + packed_اندازه;
        }

        إذا (سرلیک->خونه_شمېر == 0) {
            سرلیک->magic   = 0;
            سرلیک->نسخه_2 = 0;
            حافظه_نقل(tensor->معلومات, معلومات, معلومات_اندازه);
        }

        أرجع 0;
    }
};

ساكن ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_بفر_t, بنية ggml_tensor *) {
    ساكن tensor_traits traits;
    أرجع &traits;
}
}  // namespace ggml::cpu::kleidiai

ساكن تعداد ggml_حالت ggml_backend_cpu_kleidiai_بفر_init_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor) {
    tensor->extra = (فراغ *) ggml::cpu::kleidiai::get_tensor_traits(بفر, tensor);

    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_cpu_kleidiai_بفر_سېټ_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor,
                                                       ثابت فراغ * معلومات, size_t دننه_واټن, size_t size) {
    GGML_ASSERT(دننه_واټن == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    خپلکاری tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    خپلکاری OK            = tensor_traits->repack(tensor, معلومات, size);

    GGML_ASSERT(OK == 0);
    GGML_ناکارول_شوی(بفر);
}

ساكن ثابت حرف * ggml_backend_cpu_kleidiai_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    GGML_ناکارول_شوی(buft);
    أرجع "CPU_KLEIDIAI";
}

ساكن ggml_backend_بفر_t ggml_backend_cpu_kleidiai_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(ggml_backend_cpu_بفر_type(), size);

    إذا (بفر == تش_اشاره) {
        أرجع تش_اشاره;
    }

    بفر->buft              = buft;
    بفر->iface.init_tensor = ggml_backend_cpu_kleidiai_بفر_init_tensor;
    بفر->iface.سېټ_tensor  = ggml_backend_cpu_kleidiai_بفر_سېټ_tensor;
    بفر->iface.get_tensor  = تش_اشاره;
    بفر->iface.cpy_tensor  = تش_اشاره;
    أرجع بفر;
}

ساكن size_t ggml_backend_cpu_kleidiai_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    GGML_ناکارول_شوی(buft);
    أرجع TENSOR_ALIGNMENT;
}

ساكن size_t ggml_backend_cpu_kleidiai_بفر_type_get_alloc_اندازه(ggml_backend_بفر_type_t buft, ثابت بنية ggml_tensor * tensor) {
    GGML_ناکارول_شوی(buft);

    إذا (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        أرجع ggml_nbytes(tensor);
    }

    ثابت size_t n = tensor->ne[1];
    ثابت size_t k = tensor->ne[0];

    size_t cursor = حجم(kleidiai_وزن_سرلیک);
    cursor = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ثابت منطقي want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ثابت منطقي want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ثابت منطقي want_f32 = tensor->type == GGML_TYPE_F32;
    ثابت صحيح خونه_ټول = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ثابت منطقي allow_fallback = kleidiai_pack_fallback_allowed();

    size_t خونه_شمېر = 0;
    لكل (صحيح خونه = 0; خونه < خونه_ټول; ++خونه) {
        إذا (!allow_fallback && خونه > 0) {
            اكسر;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[خونه];
        إذا (!kernels) {
            استمر;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        إذا (!kernel || !rhs_info || !rhs_info->packed_اندازه_ex) {
            استمر;
        }

        ثابت ggml_type rhs_type = kernels->rhs_type;
        ثابت size_t بلاک_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        إذا (بلاک_len == SIZE_MAX) {
            استمر;
        }

        cursor = align_پورته(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_اندازه_ex(n, k, kernel->get_nr(), kernel->get_kr(), بلاک_len);
        ++خونه_شمېر;
    }

    إذا (خونه_شمېر == 0) {
        أرجع ggml_nbytes(tensor);
    }

    أرجع std::max(cursor, ggml_nbytes(tensor));
}

نطاق ggml::cpu::kleidiai {
صنف extra_بفر_type : ggml::cpu::extra_بفر_type {
    منطقي supports_op(ggml_backend_dev_t, ثابت بنية ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        إذا (op->op != GGML_OP_MUL_MAT) {
            أرجع خطأ;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحيح خونه_ټول = kleidiai_collect_kernel_chain(op, kernel_chain);
        ثابت منطقي src0_is_kleidiai =
            op->src[0]->بفر &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->بفر->buft->context == هذا &&
            خونه_ټول > 0;

        إذا ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            إذا (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == تش_اشاره) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == تش_اشاره) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == تش_اشاره) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == تش_اشاره) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == تش_اشاره) {
                أرجع خطأ;
            }
            إذا (op->src[1]->بفر && !ggml_backend_buft_is_host(op->src[1]->بفر->buft)) {
                أرجع خطأ;
            }

            إذا (op->src[0]->type == GGML_TYPE_Q4_K) {
                إذا (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    أرجع خطأ;
                }
                أرجع ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == حجم(عائم) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * حجم(عائم));
            }

            إذا (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                إذا (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    أرجع ریښتیا;
                }
                أرجع خطأ;
            }

            إذا (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                أرجع خطأ;
            }

            أرجع ریښتیا;
        }

        أرجع خطأ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت بنية ggml_tensor * op) override {
        إذا (op->op == GGML_OP_MUL_MAT) {
            إذا (op->src[0]->بفر && op->src[0]->بفر->buft->context == هذا) {
                أرجع (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } وإلا {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                إذا (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    ساكن std::atomic<منطقي> warned(خطأ);
                    إذا (!warned.exchange(ریښتیا)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                إذا (op->src[0]->type != GGML_TYPE_F16) {
                    أرجع تش_اشاره;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ثابت صحيح خونه_ټول = kleidiai_collect_kernel_chain(op, kernel_chain);
                إذا (خونه_ټول > 0 && op->src[1]->ne[1] > 1) {
                    إذا ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        أرجع تش_اشاره;
                    }
                    أرجع ggml::cpu::kleidiai::get_tensor_traits(عدم, عدم);
                }
            }
        }
        أرجع تش_اشاره;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_بفر_type_t ggml_backend_cpu_kleidiai_بفر_type(فراغ) {
    ساكن ggml::cpu::kleidiai::extra_بفر_type ctx;
    ساكن بنية ggml_backend_بفر_type ggml_backend_cpu_بفر_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_بفر_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_بفر_type_alloc_بفر,
                           /* .alloc_buffer_n       = */ تش_اشاره,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_بفر_type_get_alignment,
                           /* .get_max_size         = */ تش_اشاره,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_بفر_type_get_alloc_اندازه,
                           /* .get_alloc_size_n     = */ عدم,
                           /* .is_host              = */ تش_اشاره,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    أرجع &ggml_backend_cpu_بفر_type_kleidiai;
}
