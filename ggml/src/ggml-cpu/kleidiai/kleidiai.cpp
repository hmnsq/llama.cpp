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
#پایان_اگر

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

#تعریف_کن GGML_COMMON_DECL_CPP
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
} ساكن ctx = { CPU_FEATURE_NONE, اشاره_تهی, اشاره_تهی, اشاره_تهی, اشاره_تهی, 0, -1, 4 };

ساكن مضمن منطقي is_sme_family(cpu_feature f) {
    أرجع (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

ساكن ثابت حرف* cpu_feature_مقصد_string(cpu_feature f) {
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
    إذا (مقایسه_رشته(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        أرجع خطأ;
    }

    ثابت حرف* اولی = name + 3;
    ثابت حرف* آخرین = name + طول_رشته(name);

    size_t value = 0;
    ثابت خودکار [end, ec] = std::from_chars(اولی, آخرین, value, 10);

    إذا (ec != std::errc{} || end != آخرین) {
        أرجع خطأ;
    }

    *cpu = value;
    أرجع درست;
}

ساكن std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    إذا (dir == اشاره_تهی) {
        أرجع cpus;
    }

    طالما (dirent * مدخل = readdir(dir)) {
        size_t cpu = 0;
        إذا (parse_cpu_dir_name(مدخل->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    أرجع cpus;
}
#پایان_اگر

#إذا defined(__APPLE__) && defined(__aarch64__)
ساكن منطقي apple_sme_counted_perf_level(std::string name) {
    لكل (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (حرف) std::tolower((بیعلامت حرف) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    أرجع name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#پایان_اگر

ساكن فراغ جمع_کن_smcus_from_smidr(uint64_t smidr, size_t & رقم_خصوصی, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ثابت uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ثابت uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ثابت size_t shared_شمار = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ثابت uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ثابت uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ثابت uint32_t id = (affinity2 << 12) | affinity;

    إذا (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    بدل (sh) {
        حالة 2: // private SMCU
            ++رقم_خصوصی;
            اكسر;
        حالة 3: // shared SMCU
            إذا (shared_counts[id] < shared_شمار) {
                shared_counts[id] = shared_شمار;
            }
            اكسر;
        حالة 0:
            إذا (id == 0) {
                ++رقم_خصوصی;
            } وإلا إذا (shared_counts[id] < shared_شمار) {
                shared_counts[id] = shared_شمار;
            }
            اكسر;
        افتراضي:
            اكسر;
    }
}

ساكن size_t detect_رقم_smcus() {
    ثابت خودکار runtime_feat = ggml_feats_get_arch64_runtime();
    إذا (!runtime_feat.has_sme) {
        أرجع 0;
    }

#إذا defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t رقم_خصوصی = 0;
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

        جمع_کن_smcus_from_smidr(smidr, رقم_خصوصی, shared_counts);
    }

    size_t مجموع = رقم_خصوصی;
    لكل (ثابت خودکار & مدخل : shared_counts) {
        مجموع += مدخل.second;
    }
    أرجع مجموع;

#وإلاإذا defined(__APPLE__) && defined(__aarch64__)
    صحيح perf_سطوح = 0;
    size_t size = حجم(perf_سطوح);
    إذا (sysctlbyname("hw.nperflevels", &perf_سطوح, &size, اشاره_تهی, 0) != 0 ||
        size != حجم(perf_سطوح) || perf_سطوح <= 0) {
        أرجع 0;
    }

    size_t units = 0;
    لكل (صحيح i = 0; i < perf_سطوح; ++i) {
        حرف کلید[64] = {};
        صحيح physical_cpus = 0;
        صحيح cpus_در_هر_l2 = 0;

        snprintf(کلید, حجم(کلید), "hw.perflevel%d.physicalcpu", i);
        size = حجم(physical_cpus);
        إذا (sysctlbyname(کلید, &physical_cpus, &size, اشاره_تهی, 0) != 0 ||
            size != حجم(physical_cpus) || physical_cpus <= 0) {
            استمر;
        }

        snprintf(کلید, حجم(کلید), "hw.perflevel%d.cpusperl2", i);
        size = حجم(cpus_در_هر_l2);
        إذا (sysctlbyname(کلید, &cpus_در_هر_l2, &size, اشاره_تهی, 0) != 0 ||
            size != حجم(cpus_در_هر_l2) || cpus_در_هر_l2 <= 0) {
            استمر;
        }

        snprintf(کلید, حجم(کلید), "hw.perflevel%d.name", i);
        size = 0;
        إذا (sysctlbyname(کلید, اشاره_تهی, &size, اشاره_تهی, 0) != 0 || size == 0) {
            استمر;
        }

        std::string name(size, '\0');
        إذا (sysctlbyname(کلید, &name[0], &size, اشاره_تهی, 0) != 0) {
            استمر;
        }
        name.resize(size);
        طالما (!name.خالی() && name.back() == '\0') {
            name.pop_back();
        }

        إذا (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_در_هر_l2 - 1) / cpus_در_هر_l2);
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
#پایان_اگر
}

ساكن صحيح parse_uint_env(ثابت حرف *s, ثابت حرف *name, منطقي *ok) {
    إذا (!s) { *ok = خطأ; أرجع 0; }
    حرف *end = اشاره_تهی;
    دراز v = strtol(s, &end, 10);
    إذا (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = خطأ;
        أرجع 0;
    }
    إذا (v < 0 || v > بیشترین_صحیح) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = خطأ;
        أرجع 0;
    }
    *ok = درست;
    أرجع (صحيح)v;
}

ساكن فراغ init_kleidiai_context(فراغ) {
    ggml_critical_section_شروع();
    ساكن منطقي initialized = خطأ;

    إذا (!initialized) {
        initialized = درست;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ثابت حرف *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ثابت حرف *env_نخها     = getenv("GGML_TOTAL_THREADS");
        ثابت حرف *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ثابت خودکار runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        إذا (env_نخها) {
            منطقي ok = خطأ;
            صحيح hint = parse_uint_env(env_نخها, "GGML_TOTAL_THREADS", &ok);
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
        منطقي sme_env_مجموعه = (env_sme != اشاره_تهی);

        ثابت منطقي has_supported_sme_family = runtime_feat.has_sme;
        منطقي sme_cap_detected = خطأ;

        إذا (has_supported_sme_family) {
            detected_smcus = detect_رقم_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (صحيح)detected_smcus : 1;

            إذا (!sme_env_مجموعه && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        إذا (sme_env_مجموعه) {
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
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_مقصد_string(ctx.kernels_q4->required_cpu));
        }

        إذا (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (صحيح)ctx.features);
        } وإلا {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_مقصد_string(ctx.kernels_q8->required_cpu));
        }

        إذا (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (صحيح) ctx.features);
        } وإلا {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_مقصد_string(ctx.kernels_q4k->required_cpu));
        }

        إذا (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (صحيح)ctx.features);
        } وإلا {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_مقصد_string(ctx.kernels_f32->required_cpu));
        }

        ثابت منطقي has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        إذا (has_selected_sme_family_kernel) {
            إذا (sme_env_مجموعه && sme_env_ok && sme_cores > 0) {
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

    ggml_critical_section_انتها();
}

ساكن مضمن صحيح kleidiai_sme_thread_cap() {
    أرجع ctx.sme_thread_cap;
}

ساكن مضمن size_t align_به_بالا(size_t value, size_t alignment) {
    إذا (alignment == 0) {
        أرجع value;
    }
    ثابت size_t remainder = value % alignment;
    أرجع remainder == 0 ? value : value + (alignment - remainder);
}

ساكن مضمن size_t بزرگترین_مقسوم_علیه_اندازه(size_t a, size_t b) {
    طالما (b != 0) {
        ثابت size_t t = a % b;
        a = b;
        b = t;
    }
    أرجع a;
}

ساكن مضمن منطقي lcm_اندازه(size_t a, size_t b, size_t & نتیجه) {
    إذا (a == 0 || b == 0) {
        نتیجه = 0;
        أرجع خطأ;
    }
    ثابت size_t g = بزرگترین_مقسوم_علیه_اندازه(a, b);
    ثابت size_t q = a / g;
    إذا (q > SIZE_MAX / b) {
        أرجع خطأ;
    }
    نتیجه = q * b;
    أرجع درست;
}

ساكن مضمن size_t ceil_div_اندازه(size_t a, size_t b) {
    أرجع b == 0 ? 0 : (a + b - 1) / b;
}

ساكن مضمن size_t kleidiai_chunk_تعداد_ستون(size_t n, صحيح nth_مجموع, منطقي disable_chunking, size_t n_گام) {
    ثابت size_t multiplier = (nth_مجموع == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ثابت size_t مقسوم_علیه = std::max<size_t>(1, (size_t) nth_مجموع * multiplier);
    ثابت size_t chunk_تعداد_ستون = align_به_بالا(std::max<size_t>(1, ceil_div_اندازه(n, مقسوم_علیه)), n_گام);
    أرجع chunk_تعداد_ستون ? chunk_تعداد_ستون : n_گام;
}

بنية kleidiai_بلوک_آرگومانها {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

ساكن مضمن kleidiai_بلوک_آرگومانها kleidiai_get_بلوک_آرگومانها(ggml_type rhs_type) {
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
        أرجع درست;
    }
    أرجع ctx.thread_hint > ctx.sme_thread_cap;
}

بنية kleidiai_وزن_سرآیند {
    uint32_t magic;
    uint16_t نسخه_2;
    uint16_t جایگاه_شمار;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

ساكن مضمن kleidiai_وزن_سرآیند * kleidiai_وزن_سرآیند_from_ptr(فراغ * داده) {
    أرجع reinterpret_cast<kleidiai_وزن_سرآیند *>(داده);
}

ساكن مضمن ثابت kleidiai_وزن_سرآیند * kleidiai_وزن_سرآیند_from_ptr(ثابت فراغ * داده) {
    أرجع reinterpret_cast<ثابت kleidiai_وزن_سرآیند *>(داده);
}

ساكن مضمن منطقي kleidiai_is_وزن_سرآیند_valid(ثابت kleidiai_وزن_سرآیند * سرآیند) {
    إذا (!سرآیند) {
        أرجع خطأ;
    }
    إذا (سرآیند->magic != GGML_KLEIDIAI_PACK_MAGIC || سرآیند->نسخه_2 != GGML_KLEIDIAI_PACK_نسخه) {
        أرجع خطأ;
    }
    إذا (سرآیند->جایگاه_شمار == 0 || سرآیند->جایگاه_شمار > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        أرجع خطأ;
    }
    أرجع درست;
}

ساكن مضمن uint8_t * kleidiai_وزن_جایگاه_ptr(kleidiai_وزن_سرآیند * سرآیند, صحيح جایگاه) {
    إذا (!kleidiai_is_وزن_سرآیند_valid(سرآیند)) {
        أرجع اشاره_تهی;
    }
    إذا (جایگاه < 0 || جایگاه >= سرآیند->جایگاه_شمار) {
        أرجع اشاره_تهی;
    }
    أرجع reinterpret_cast<uint8_t *>(سرآیند) + سرآیند->offsets[جایگاه];
}

ساكن مضمن ثابت uint8_t * kleidiai_وزن_جایگاه_ptr(ثابت kleidiai_وزن_سرآیند * سرآیند, صحيح جایگاه) {
    إذا (!kleidiai_is_وزن_سرآیند_valid(سرآیند)) {
        أرجع اشاره_تهی;
    }
    إذا (جایگاه < 0 || جایگاه >= سرآیند->جایگاه_شمار) {
        أرجع اشاره_تهی;
    }
    أرجع reinterpret_cast<ثابت uint8_t *>(سرآیند) + سرآیند->offsets[جایگاه];
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

قالب <نام_نوع SelectFallback>
ساكن صحيح kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بیرون,
        SelectFallback select_fallback) {
    صحيح count = 0;
    إذا (!primary) {
        أرجع 0;
    }
    بیرون[count++] = primary;

    إذا (primary->rhs_info.repack_نما == RHS_REPACK_SINGLE_ONLY) {
        أرجع count;
    }

    إذا (is_sme_family(primary->required_cpu)) {
        ثابت cpu_feature fallback_ماسک = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        إذا (fallback_ماسک != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_ماسک);
            إذا (fallback && fallback != primary &&
                fallback->rhs_info.repack_نما != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                بیرون[count++] = fallback;
            }
        }
    }

    أرجع count;
}

ساكن صحيح kleidiai_collect_kernel_chain(ثابت بنية ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بیرون) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, بیرون,
        [&](cpu_feature ماسک) { أرجع ggml_kleidiai_select_kernels(ماسک, op); });
}

ساكن صحيح kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بیرون) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, بیرون,
        [&](cpu_feature ماسک) { أرجع ggml_kleidiai_select_kernels_q4_0(ماسک); });
}

ساكن صحيح kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بیرون) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, بیرون,
        [&](cpu_feature ماسک) { أرجع ggml_kleidiai_select_kernels_q4_k(ماسک); });
}

ساكن صحيح kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بیرون) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, بیرون,
        [&](cpu_feature ماسک) { أرجع ggml_kleidiai_select_kernels_q8_0(ماسک); });
}

ساكن صحيح kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & بیرون) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, بیرون,
        [&](cpu_feature ماسک) { أرجع ggml_kleidiai_select_kernels_f32(ماسک); });
}

ساكن مضمن int64_t ggml_ne(ثابت ggml_tensor * tensor, صحيح dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    أرجع tensor->ne[dim];
}

نطاق ggml::cpu::kleidiai {

ساكن size_t دور_پایین(size_t x, size_t y) {
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
        ثابت صحيح جایگاه_شمار = kleidiai_collect_kernel_chain(op, kernel_chain);
        إذا (جایگاه_شمار == 0) {
            أرجع خطأ;
        }

        ثابت منطقي is_gemv = op->src[1]->ne[1] == 1;
        ثابت size_t k = op->src[0]->ne[0];
        ثابت size_t n = op->src[0]->ne[1];
        ثابت size_t m = op->src[1]->ne[1];

        إذا (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ثابت kleidiai_بلوک_آرگومانها بلوک_آرگومانها = kleidiai_get_بلوک_آرگومانها(kernel_chain[0]->rhs_type);
            ثابت size_t qk = بلوک_آرگومانها.pack_bl;

            size_t cursor = 0;
            منطقي any_جایگاه = خطأ;

            لكل (صحيح جایگاه = 0; جایگاه < جایگاه_شمار; ++جایگاه) {
                ggml_kleidiai_kernels * kernels = kernel_chain[جایگاه];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                إذا (!lhs_info || !lhs_info->packed_اندازه_ex || !kernel) {
                    أرجع خطأ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                ثابت size_t packed = lhs_info->packed_اندازه_ex(m, k, qk, mr, kr, sr);

                cursor = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_جایگاه = درست;
            }

            إذا (!any_جایگاه) {
                أرجع خطأ;
            }

            size = cursor;
            أرجع درست;
        }

        إذا (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : اشاره_تهی;
            إذا (is_gemv && op->src[1]->nb[0] == (int64_t) حجم(عائم) && gemv_kernel &&
                gemv_kernel->get_lhs_جابه_جایی_ex && gemv_kernel->get_rhs_packed_جابه_جایی_ex &&
                gemv_kernel->پیاپی_kernel_ex && gemv_kernel->get_dst_جابه_جایی) {
                size = 0;
                أرجع درست;
            }

            size_t cursor = 0;
            منطقي any_جایگاه = خطأ;

            لكل (صحيح جایگاه = 0; جایگاه < جایگاه_شمار; ++جایگاه) {
                ggml_kleidiai_kernels * kernels = kernel_chain[جایگاه];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                إذا (!lhs_info || !lhs_info->packed_اندازه_ex || !kernel) {
                    أرجع خطأ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_اندازه_ex(m, k, 0, mr, kr, sr);
                any_جایگاه = درست;
            }

            إذا (!any_جایگاه) {
                أرجع خطأ;
            }

            size = cursor;
            أرجع درست;
        }

        إذا (op->src[0]->type == GGML_TYPE_F16) {
            ثابت int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ثابت int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ثابت int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            منطقي any_جایگاه = خطأ;

            لكل (صحيح جایگاه = 0; جایگاه < جایگاه_شمار; ++جایگاه) {
                ggml_kleidiai_kernels * kernels = kernel_chain[جایگاه];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                إذا (!lhs_info || !lhs_info->packed_اندازه_ex || !kernels->rhs_info.packed_اندازه_ex || !kernel) {
                    أرجع خطأ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_اندازه_ex(m * r, k, 0, mr, kr, sr);
                any_جایگاه = درست;
            }

            لكل (صحيح جایگاه = 0; جایگاه < جایگاه_شمار; ++جایگاه) {
                ggml_kleidiai_kernels * kernels = kernel_chain[جایگاه];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                إذا (!kernel || !kernels->rhs_info.packed_اندازه_ex) {
                    أرجع خطأ;
                }
                cursor  = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_اندازه_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * حجم(عائم);
            cursor  = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * حجم(عائم);

            إذا (!any_جایگاه) {
                أرجع خطأ;
            }

            size = cursor;
            أرجع درست;
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

        GGML_TENSOR_دودویی_OP_LOCALS

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
                              kernels->gemv.get_lhs_جابه_جایی_ex &&
                              kernels->gemv.get_rhs_packed_جابه_جایی_ex &&
                              kernels->gemv.پیاپی_kernel_ex &&
                              kernels->gemv.get_dst_جابه_جایی;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        إذا (!kernel || !kernel->get_lhs_جابه_جایی_ex ||
            !kernel->get_rhs_packed_جابه_جایی_ex || !kernel->پیاپی_kernel_ex || !kernel->get_dst_جابه_جایی) {
            أرجع خطأ;
        }

        إذا (!use_gemv && (!lhs_info || !lhs_info->get_جابه_جایی || !lhs_info->get_packed_جابه_جایی_ex ||
                          !lhs_info->packed_اندازه_ex || !lhs_info->pack_func_ex)) {
            أرجع خطأ;
        }

        ثابت kleidiai_وزن_سرآیند * سرآیند = kleidiai_وزن_سرآیند_from_ptr(src0->داده);
        ثابت منطقي has_سرآیند = kleidiai_is_وزن_سرآیند_valid(سرآیند);

        ثابت uint8_t * rhs_پایه = has_سرآیند ? kleidiai_وزن_جایگاه_ptr(سرآیند, 0)
                                              : static_cast<ثابت uint8_t *>(src0->داده);
        إذا (!rhs_پایه) {
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
        ثابت size_t n_گام = kernel->get_n_گام() ? kernel->get_n_گام() : 1;
        ثابت منطقي disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) بیشترین_صحیح);

        ثابت int64_t batch_اندازه = ne12 * ne13;
        لكل (int64_t batch_idx = 0; batch_idx < batch_اندازه; ++batch_idx) {
            ثابت int64_t i12 = batch_idx % ne12;
            ثابت int64_t i13 = batch_idx / ne12;
            ثابت uint8_t * lhs_batch_پایه = static_cast<ثابت uint8_t *>(src1->داده) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_پایه = static_cast<uint8_t *>(dst->داده) + i12 * dst->nb[2] + i13 * dst->nb[3];

            إذا (!use_gemv) {
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_نخها = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_نخها = std::max<int64_t>(1, max_نخها);
                ثابت int64_t use_نخها = std::min<int64_t>(nth, max_نخها);

                إذا (ith < use_نخها) {
                    ثابت int64_t رقم_m_در_هر_thread0   = دور_پایین((size_t)(m_roundup_mr / use_نخها), mr);
                    ثابت int64_t رقم_m_در_هر_thread_N_1 = (int64_t)m - (use_نخها - 1) * رقم_m_در_هر_thread0;

                    ثابت int64_t m_شروع = (int64_t)ith * رقم_m_در_هر_thread0;
                    ثابت int64_t m_شمار = (ith == use_نخها - 1) ? رقم_m_در_هر_thread_N_1 : رقم_m_در_هر_thread0;

                    ثابت size_t پایه_packed_off  = lhs_info->get_packed_جابه_جایی_ex(m_شروع, k, 0, mr, kr, sr);
                    ثابت size_t بعدی_بلوک_off   = lhs_info->get_packed_جابه_جایی_ex(m_شروع + mr, k, 0, mr, kr, sr);
                    ثابت size_t ردیف_stride_بایتها = mr ? (بعدی_بلوک_off - پایه_packed_off) / mr : 0;

                    int64_t مانده = m_شمار;
                    int64_t cur       = m_شروع;

                    طالما (مانده > 0) {
                        ثابت int64_t take = std::min<int64_t>((int64_t)m - cur, مانده);
                        ثابت size_t src_off = lhs_info->get_جابه_جایی(cur, src1->nb[1]);
                        ثابت فراغ * src_ptr = lhs_batch_پایه + src_off;
                        ثابت size_t dst_off = پایه_packed_off + (size_t)(cur - m_شروع) * ردیف_stride_بایتها;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        مانده -= take;
                    }
                }
            }

            إذا (ith == 0) {
                ggml_threadpool_chunk_مجموعه(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ثابت size_t chunk_تعداد_ستون = kleidiai_chunk_تعداد_ستون(n, nth, disable_chunking, n_گام);
            GGML_ASSERT(chunk_تعداد_ستون <= (size_t) بیشترین_صحیح);

            صحيح current_ستون = ggml_threadpool_chunk_جمع_کن(params->threadpool, (صحيح) chunk_تعداد_ستون);
            طالما ((size_t) current_ستون < n) {
                ثابت size_t n_شروع = (size_t) current_ستون;
                ثابت size_t n_مقصد_process = std::min(chunk_تعداد_ستون, n - n_شروع);

                إذا (n_مقصد_process > 0) {
                    ثابت size_t lhs_جابه_جایی = use_gemv ? kernel->get_lhs_جابه_جایی_ex(0, k, 0)
                                                       : lhs_info->get_packed_جابه_جایی_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_جابه_جایی = kernel->get_rhs_packed_جابه_جایی_ex(n_شروع, k, 0);
                    ثابت size_t dst_جابه_جایی        = kernel->get_dst_جابه_جایی(0, n_شروع, dst_stride);

                    ثابت فراغ * lhs_ptr = use_gemv ? lhs_batch_پایه + lhs_جابه_جایی
                                                    : lhs_packed + lhs_جابه_جایی;
                    ثابت فراغ * rhs_ptr = rhs_پایه + rhs_packed_جابه_جایی;
                    عائم * dst_ptr      = reinterpret_cast<عائم *>(dst_batch_پایه + dst_جابه_جایی);

                    kernel->پیاپی_kernel_ex(m, n_مقصد_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          حجم(عائم),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_ستون = ggml_threadpool_chunk_جمع_کن(params->threadpool, (صحيح) chunk_تعداد_ستون);
            }

            إذا (batch_idx != batch_اندازه - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        أرجع درست;
    }

    منطقي compute_forward_fp16(ggml_compute_params * params, بنية ggml_tensor * dst) {
        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_دودویی_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        إذا (!kernels) {
            أرجع خطأ;
        }

        ثابت منطقي is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        إذا (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_جابه_جایی_ex || !kernel->get_rhs_packed_جابه_جایی_ex || !kernel->پیاپی_kernel_ex) {
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

        ثابت int64_t m_گروه = ne11;
        ثابت int64_t m       = m_گروه;
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
            ثابت uint8_t * rhs_batch_پایه = static_cast<ثابت uint8_t *>(src0->داده) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_پایه = static_cast<uint8_t *>(dst->داده) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ثابت int64_t m_roundup_mr = kai_roundup(m, mr);
                ثابت int64_t رقم_نخها  = KAI_MIN(m_roundup_mr / mr, nth);

                إذا (ith < رقم_نخها) {
                    ثابت int64_t رقم_m_در_هر_thread0   = دور_پایین((size_t)(m_roundup_mr / رقم_نخها), (size_t)mr);
                    ثابت int64_t رقم_m_در_هر_thread_N_1 = m - (رقم_نخها - 1) * رقم_m_در_هر_thread0;

                    ثابت int64_t m_شروع = ith * رقم_m_در_هر_thread0;
                    ثابت int64_t m_شمار = (ith == رقم_نخها - 1) ? رقم_m_در_هر_thread_N_1 : رقم_m_در_هر_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ثابت size_t پایه_packed_off  = lhs_info->get_packed_جابه_جایی_ex(m_شروع, k, 0, mr, kr, sr);
                    ثابت size_t بعدی_بلوک_off   = lhs_info->get_packed_جابه_جایی_ex(m_شروع + mr, k, 0, mr, kr, sr);
                    ثابت size_t ردیف_stride_بایتها = (بعدی_بلوک_off - پایه_packed_off) / (size_t)mr;

                    int64_t مانده = m_شمار;
                    int64_t cur       = m_شروع;

                    طالما (مانده > 0) {
                        ثابت int64_t ردیف_درون_گروه = cur;
                        ثابت int64_t avail        = m_گروه - ردیف_درون_گروه;
                        ثابت int64_t take         = std::min(avail, مانده);

                        ثابت uint8_t * lhs_batch_پایه = static_cast<ثابت uint8_t *>(src1->داده) + batch_idx * src1->nb[2];
                        ثابت فراغ * src_ptr = lhs_batch_پایه + (size_t)ردیف_درون_گروه * lhs_stride;
                        ثابت size_t dst_off = پایه_packed_off + (size_t)(cur - m_شروع) * ردیف_stride_بایتها;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        مانده -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            إذا (ith == 0) {
                memset(bias, 0, (size_t)n * حجم(عائم));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<عائم *>(rhs_kxn),
                                        reinterpret_cast<ثابت uint16_t *>(rhs_batch_پایه),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * حجم(عائم),
                             rhs_kxn, bias, اشاره_تهی, rhs_packed, 0, اشاره_تهی);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ثابت int64_t n_گام  = (int64_t) kernel->get_n_گام();
                int64_t رقم_نخها_n = KAI_MIN(n / n_گام, nth);
                إذا (رقم_نخها_n <= 0) {
                    رقم_نخها_n = 1;
                }

                إذا (ith < رقم_نخها_n) {
                    ثابت int64_t رقم_n_در_هر_thread0   = دور_پایین((size_t)(n / رقم_نخها_n), (size_t)n_گام);
                    ثابت int64_t رقم_n_در_هر_thread_N_1 = n - (رقم_نخها_n - 1) * رقم_n_در_هر_thread0;

                    ثابت int64_t n_شروع      = ith * رقم_n_در_هر_thread0;
                    ثابت int64_t n_مقصد_process = (ith == رقم_نخها_n - 1) ? رقم_n_در_هر_thread_N_1 : رقم_n_در_هر_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ثابت size_t lhs_packed_offset0 = lhs_info->get_packed_جابه_جایی_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_جابه_جایی  = kernel->get_rhs_packed_جابه_جایی_ex(n_شروع, k, 0);
                    ثابت size_t dst_جابه_جایی         = kernel->get_dst_جابه_جایی((size_t)0, (size_t)n_شروع, dst_stride);

                    ثابت فراغ * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ثابت فراغ * rhs_ptr = rhs_packed + rhs_packed_جابه_جایی;
                    عائم * dst_ptr      = reinterpret_cast<عائم *>(dst_batch_پایه + dst_جابه_جایی);

                    kernel->پیاپی_kernel_ex(m, n_مقصد_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, حجم(عائم), -FLT_MAX, FLT_MAX);
                }
            }

            إذا (batch_idx != batch_اندازه - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        أرجع درست;
    }

    منطقي compute_forward_qx(بنية ggml_compute_params * params, بنية ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_دودویی_OP_LOCALS

        ثابت kleidiai_وزن_سرآیند * سرآیند = kleidiai_وزن_سرآیند_from_ptr(src0->داده);
        ثابت منطقي has_سرآیند = kleidiai_is_وزن_سرآیند_valid(سرآیند);
        ثابت منطقي is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحيح جایگاه_مجموع = kleidiai_collect_kernel_chain(dst, kernel_chain);

        خودکار وزن_for_جایگاه = [&](صحيح جایگاه_شاخص, size_t & اندازه_بیرون) -> ثابت uint8_t * {
            إذا (جایگاه_شاخص < 0 || جایگاه_شاخص >= جایگاه_مجموع) {
                أرجع اشاره_تهی;
            }
            إذا (has_سرآیند) {
                إذا (جایگاه_شاخص < سرآیند->جایگاه_شمار) {
                    اندازه_بیرون = static_cast<size_t>(سرآیند->sizes[جایگاه_شاخص]);
                    أرجع kleidiai_وزن_جایگاه_ptr(سرآیند, جایگاه_شاخص);
                }
                أرجع اشاره_تهی;
            }
            إذا (جایگاه_شاخص == 0) {
                اندازه_بیرون = ggml_nbytes(src0);
                أرجع static_cast<ثابت uint8_t *>(src0->داده);
            }
            أرجع اشاره_تهی;
        };

        بنية runtime_جایگاه {
            صحيح جایگاه_شاخص;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            size_t mr;
            size_t nr;
            size_t kr;
            size_t sr;
            size_t n_گام;
            size_t lhs_packed_اندازه;
            size_t lhs_جابه_جایی;
            size_t lhs_bl;
            size_t rhs_bl;
            size_t pack_bl;
            size_t lhs_packed_offset0;
            صحيح assigned_نخها;
            صحيح thread_begin;
            صحيح thread_انتها;
            ثابت uint8_t * rhs_پایه;
        };

        std::array<runtime_جایگاه, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        صحيح runtime_شمار = 0;

        لكل (صحيح جایگاه = 0; جایگاه < جایگاه_مجموع && runtime_شمار < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++جایگاه) {
            ggml_kleidiai_kernels * kernels = kernel_chain[جایگاه];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            إذا (!kinfo || !linfo || !linfo->packed_اندازه_ex || !linfo->pack_func_ex || !linfo->get_جابه_جایی ||
                !kinfo->get_rhs_packed_جابه_جایی_ex || !kinfo->پیاپی_kernel_ex || !kinfo->get_dst_جابه_جایی) {
                استمر;
            }

            size_t rhs_اندازه = 0;
            ثابت uint8_t * rhs_ptr = وزن_for_جایگاه(جایگاه, rhs_اندازه);
            إذا (!rhs_ptr || rhs_اندازه == 0) {
                استمر;
            }

            ثابت kleidiai_بلوک_آرگومانها بلوک_آرگومانها = kleidiai_get_بلوک_آرگومانها(kernels->rhs_type);

            runtime[runtime_شمار] = {
                جایگاه,
                kernels,
                kinfo,
                linfo,
                kinfo->get_mr(),
                kinfo->get_nr(),
                kinfo->get_kr(),
                kinfo->get_sr(),
                kinfo->get_n_گام(),
                0,
                0,
                بلوک_آرگومانها.lhs_bl,
                بلوک_آرگومانها.rhs_bl,
                بلوک_آرگومانها.pack_bl,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_شمار;
        }

        إذا (runtime_شمار == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            أرجع خطأ;
        }

        ثابت صحيح nth_مجموع = params->nth > 0 ? params->nth : 1;
        ثابت صحيح ith_مجموع = params->ith;

        صحيح sme_جایگاه = -1;
        صحيح non_sme_جایگاه = -1;
        لكل (صحيح i = 0; i < runtime_شمار; ++i) {
            إذا (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_جایگاه = i;
                اكسر;
            }
        }

        لكل (صحيح i = 0; i < runtime_شمار; ++i) {
            إذا (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_جایگاه = i;
                اكسر;
            }
        }

        ثابت صحيح sme_cap_حد = ctx.sme_thread_cap;
        ثابت منطقي use_hybrid = sme_cap_حد > 0 &&
                                 runtime_شمار > 1 &&
                                 nth_مجموع > sme_cap_حد;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_تعداد_ستون_در_هر_thread = 0;
        إذا (runtime_شمار > 0 && nth_مجموع > 0) {
            min_تعداد_ستون_در_هر_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_مجموع);
        }
        ثابت منطقي too_small_for_hybrid = (min_تعداد_ستون_در_هر_thread < 2) || (ne11 < 128);

        ثابت منطقي hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        إذا (!hybrid_enabled) {
            صحيح chosen_جایگاه = 0;
            إذا (too_small_for_hybrid && sme_جایگاه != -1) {
                chosen_جایگاه = nth_مجموع > sme_cap_حد && non_sme_جایگاه != -1 ? non_sme_جایگاه : sme_جایگاه;
            } وإلا إذا (runtime_شمار > 1 && ctx.sme_thread_cap > 0 && nth_مجموع > ctx.sme_thread_cap) {
                chosen_جایگاه = 1;
            }
            إذا (chosen_جایگاه != 0 && chosen_جایگاه < runtime_شمار) {
                runtime[0] = runtime[chosen_جایگاه];
                runtime[0].assigned_نخها = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_انتها = 0;
            }
            runtime_شمار = runtime_شمار > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_جایگاه = -1;
            إذا (runtime_شمار > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_جایگاه = 0;
            }
        }

        ثابت منطقي sme_single_only = sme_جایگاه != -1 &&
                                     runtime[sme_جایگاه].kernels->rhs_info.repack_نما == RHS_REPACK_SINGLE_ONLY;
        صحيح sme_cap = sme_single_only ? nth_مجموع : kleidiai_sme_thread_cap();
        إذا (sme_cap < 0) {
            sme_cap = nth_مجموع;
        }
        sme_cap = std::min(sme_cap, nth_مجموع);

        صحيح نخها_مانده = nth_مجموع;
        إذا (sme_جایگاه != -1) {
            صحيح sme_نخها = std::min(std::max(sme_cap, 0), نخها_مانده);
            runtime[sme_جایگاه].assigned_نخها = sme_نخها;
            نخها_مانده -= sme_نخها;
        }

        صحيح fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        صحيح fallback_شمار = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        لكل (صحيح i = 0; i < runtime_شمار; ++i) {
            إذا (i == sme_جایگاه) {
                استمر;
            }
            fallback_indices[fallback_شمار++] = i;
        }

        لكل (صحيح fi = 0; fi < fallback_شمار; ++fi) {
            إذا (نخها_مانده <= 0) {
                اكسر;
            }
            ثابت صحيح جایگاه_شاخص = fallback_indices[fi];
            ثابت صحيح slots_چپ = fallback_شمار - fi;
            صحيح share = (نخها_مانده + slots_چپ - 1) / slots_چپ;
            share     = std::min(share, نخها_مانده);
            runtime[جایگاه_شاخص].assigned_نخها = share;
            نخها_مانده -= share;
        }

        صحيح thread_cursor = 0;
        لكل (صحيح i = 0; i < runtime_شمار; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_نخها;
            runtime[i].thread_انتها = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ثابت صحيح فعال_نخها = thread_cursor;

        صحيح local_جایگاه = -1;
        صحيح local_ith  = 0;
        لكل (صحيح i = 0; i < runtime_شمار; ++i) {
            إذا (ith_مجموع >= runtime[i].thread_begin && ith_مجموع < runtime[i].thread_انتها) {
                local_جایگاه = i;
                local_ith  = ith_مجموع - runtime[i].thread_begin;
                اكسر;
            }
        }
        ثابت size_t k = ne00;
        ثابت size_t m = ne11;
        ثابت size_t n = ne01;

        size_t cursor = 0;
        لكل (صحيح i = 0; i < runtime_شمار; ++i) {
            runtime[i].lhs_packed_اندازه = runtime[i].lhs_info->packed_اندازه_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_جابه_جایی = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_جابه_جایی_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_اندازه;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_گام = 1;
        لكل (صحيح i = 0; i < runtime_شمار; ++i) {
            إذا (runtime[i].assigned_نخها == 0) {
                استمر;
            }
            size_t بعدی_گام = 0;
            إذا (!lcm_اندازه(common_گام, runtime[i].n_گام ? runtime[i].n_گام : 1, بعدی_گام)) {
                أرجع خطأ;
            }
            common_گام = بعدی_گام;
        }
        GGML_ASSERT(common_گام > 0);

        ثابت منطقي disable_chunking = ggml_is_numa();
        ثابت size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ثابت size_t chunk_مقسوم_علیه = (فعال_نخها == 1 || disable_chunking) ? (size_t) فعال_نخها : (size_t) فعال_نخها * chunk_multiplier;
        size_t chunk_تعداد_ستون = align_به_بالا(std::max<size_t>(1, ceil_div_اندازه(n, chunk_مقسوم_علیه)), common_گام);
        إذا (chunk_تعداد_ستون == 0) {
            chunk_تعداد_ستون = common_گام;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ثابت size_t nchunk_اندازه = std::max<size_t>(1, ceil_div_اندازه(n, chunk_تعداد_ستون));
        GGML_ASSERT(nchunk_اندازه <= (size_t)بیشترین_صحیح);
        ثابت صحيح nchunk = (صحيح)nchunk_اندازه;
        ثابت size_t dst_stride = dst->nb[1];

        خودکار پیاپی_chunk = [&](runtime_جایگاه & جایگاه, size_t global_شروع, size_t تعداد_ستون_2, uint8_t * dst_batch_پایه) {
            ثابت size_t rhs_packed_جابه_جایی = جایگاه.kernel->get_rhs_packed_جابه_جایی_ex(global_شروع, k, جایگاه.rhs_bl);
            ثابت size_t dst_جابه_جایی        = جایگاه.kernel->get_dst_جابه_جایی(0, global_شروع, dst_stride);

            ثابت uint8_t * lhs_ptr = scratch + جایگاه.lhs_جابه_جایی + جایگاه.lhs_packed_offset0;
            ثابت uint8_t * rhs_ptr = جایگاه.rhs_پایه + rhs_packed_جابه_جایی;
            عائم * dst_ptr         = reinterpret_cast<عائم *>(dst_batch_پایه + dst_جابه_جایی);

            جایگاه.kernel->پیاپی_kernel_ex(m, تعداد_ستون_2, k, جایگاه.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       حجم(عائم),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        لكل (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ثابت uint8_t * lhs_batch_پایه = static_cast<ثابت uint8_t *>(src1->داده) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_پایه = static_cast<uint8_t *>(dst->داده) + batch_idx * dst->nb[2];

            إذا (local_جایگاه != -1) {
                runtime_جایگاه & جایگاه = runtime[local_جایگاه];
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)جایگاه.mr);
                int64_t max_نخها = جایگاه.mr ? (m_roundup_mr / (int64_t)جایگاه.mr) : جایگاه.assigned_نخها;
                max_نخها = std::max<int64_t>(1, max_نخها);
                ثابت int64_t use_نخها = std::min<int64_t>(جایگاه.assigned_نخها, max_نخها);

                إذا (local_ith < use_نخها) {
                    ثابت int64_t رقم_m_در_هر_thread0   = دور_پایین((size_t)(m_roundup_mr / use_نخها), جایگاه.mr);
                    ثابت int64_t رقم_m_در_هر_thread_N_1 = (int64_t)m - (use_نخها - 1) * رقم_m_در_هر_thread0;

                    ثابت int64_t m_شروع = (int64_t)local_ith * رقم_m_در_هر_thread0;
                    ثابت int64_t m_شمار = (local_ith == use_نخها - 1) ? رقم_m_در_هر_thread_N_1 : رقم_m_در_هر_thread0;

                    ثابت size_t پایه_packed_off  = جایگاه.lhs_info->get_packed_جابه_جایی_ex(m_شروع, k, جایگاه.lhs_bl, جایگاه.mr, جایگاه.kr, جایگاه.sr);
                    ثابت size_t بعدی_بلوک_off   = جایگاه.lhs_info->get_packed_جابه_جایی_ex(m_شروع + جایگاه.mr, k, جایگاه.lhs_bl, جایگاه.mr, جایگاه.kr, جایگاه.sr);
                    ثابت size_t ردیف_stride_بایتها = جایگاه.mr ? (بعدی_بلوک_off - پایه_packed_off) / جایگاه.mr : 0;

                    int64_t مانده = m_شمار;
                    int64_t cur       = m_شروع;

                    uint8_t * lhs_packed = scratch + جایگاه.lhs_جابه_جایی;
                    طالما (مانده > 0) {
                        ثابت int64_t ردیف_درون_گروه = cur;
                        ثابت int64_t avail        = (int64_t)m - ردیف_درون_گروه;
                        ثابت int64_t take         = std::min(avail, مانده);

                        ثابت size_t src_off = جایگاه.lhs_info->get_جابه_جایی(ردیف_درون_گروه, src1->nb[1]);
                        ثابت فراغ * src_ptr = lhs_batch_پایه + src_off;
                        ثابت size_t dst_off = پایه_packed_off + (size_t)(cur - m_شروع) * ردیف_stride_بایتها;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        جایگاه.lhs_info->pack_func_ex(take, k, جایگاه.lhs_bl, جایگاه.mr, جایگاه.kr, جایگاه.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        مانده -= take;
                    }
                }
            }

            إذا (ith_مجموع == 0) {
                ggml_threadpool_chunk_مجموعه(params->threadpool, فعال_نخها);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            إذا (local_جایگاه != -1) {
                runtime_جایگاه & جایگاه = runtime[local_جایگاه];
                صحيح current_chunk = ith_مجموع;
                طالما (current_chunk < nchunk) {
                    ثابت size_t global_شروع = (size_t)current_chunk * chunk_تعداد_ستون;
                    إذا (global_شروع >= n) {
                        اكسر;
                    }

                    ثابت size_t تعداد_ستون_2 = std::min(chunk_تعداد_ستون, n - global_شروع);
                    إذا (تعداد_ستون_2 > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        پیاپی_chunk(جایگاه, global_شروع, تعداد_ستون_2, dst_batch_پایه);
                    }

                    current_chunk = ggml_threadpool_chunk_جمع_کن(params->threadpool, 1);
                }
            }

            إذا (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        أرجع درست;
    }

عام:
    صحيح repack(بنية ggml_tensor * tensor, ثابت فراغ * داده, size_t داده_اندازه) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ثابت size_t n = tensor->ne[1];
        ثابت size_t k = tensor->ne[0];

        kleidiai_وزن_سرآیند * سرآیند = kleidiai_وزن_سرآیند_from_ptr(tensor->داده);
        إذا (!سرآیند) {
            أرجع -1;
        }

        سرآیند->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        سرآیند->نسخه_2    = GGML_KLEIDIAI_PACK_نسخه;
        سرآیند->جایگاه_شمار = 0;

        uint8_t * پایه_ptr = static_cast<uint8_t *>(tensor->داده);
        size_t cursor = حجم(kleidiai_وزن_سرآیند);
        cursor = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت منطقي want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ثابت منطقي want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ثابت منطقي want_f32 = tensor->type == GGML_TYPE_F32;
        ثابت صحيح جایگاه_مجموع = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ثابت منطقي allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<عائم>  scales;
        std::vector<عائم>  bias;
        std::vector<عائم> ردیف_f32;

        إذا (want_q8 && جایگاه_مجموع > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ثابت size_t ردیف_stride = tensor->nb[1];
            ثابت size_t k_بلوکها   = (k + QK8_0 - 1) / QK8_0;

            إذا (tensor->type == GGML_TYPE_Q6_K) {
                إذا (k % QK_K != 0 || k % QK8_0 != 0) {
                    أرجع -1;
                }
                ردیف_f32.resize(k);
            }

            لكل (size_t ردیف = 0; ردیف < n; ++ردیف) {
                ثابت uint8_t * ردیف_داده = static_cast<ثابت uint8_t *>(داده) + ردیف * ردیف_stride;

                إذا (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_ردیف_q6_K(reinterpret_cast<ثابت بلوک_q6_K *>(ردیف_داده), ردیف_f32.داده(), k);
                }

                عائم max_abs = 0.0f;
                إذا (tensor->type == GGML_TYPE_Q6_K) {
                    لكل (size_t خطی_idx = 0; خطی_idx < k; ++خطی_idx) {
                        max_abs = std::max(max_abs, std::fabs(ردیف_f32[خطی_idx]));
                    }
                } وإلا {
                    ثابت بلوک_q8_0 * ردیف_بلوکها = reinterpret_cast<ثابت بلوک_q8_0 *>(ردیف_داده);
                    لكل (size_t بلوک = 0; بلوک < k_بلوکها; ++بلوک) {
                        ثابت بلوک_q8_0 & blk = ردیف_بلوکها[بلوک];
                        ثابت عائم d = GGML_FP16_مقصد_FP32(blk.d);
                        لكل (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t خطی_idx = بلوک * QK8_0 + l;
                            إذا (خطی_idx >= k) {
                                اكسر;
                            }
                            ثابت عائم value = d * static_cast<عائم>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                عائم scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[ردیف] = scale;
                ثابت عائم inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                إذا (tensor->type == GGML_TYPE_Q6_K) {
                    لكل (size_t خطی_idx = 0; خطی_idx < k; ++خطی_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(ردیف_f32[خطی_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[ردیف * k + خطی_idx] = static_cast<int8_t>(q);
                    }
                } وإلا {
                    ثابت بلوک_q8_0 * ردیف_بلوکها = reinterpret_cast<ثابت بلوک_q8_0 *>(ردیف_داده);
                    لكل (size_t بلوک = 0; بلوک < k_بلوکها; ++بلوک) {
                        ثابت بلوک_q8_0 & blk = ردیف_بلوکها[بلوک];
                        ثابت عائم d = GGML_FP16_مقصد_FP32(blk.d);
                        لكل (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t خطی_idx = بلوک * QK8_0 + l;
                            إذا (خطی_idx >= k) {
                                اكسر;
                            }
                            ثابت عائم value = d * static_cast<عائم>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[ردیف * k + خطی_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        إذا (want_f32 && جایگاه_مجموع > 0) {
            bias.resize(n, 0.0f);
        }

        لكل (صحيح جایگاه = 0; جایگاه < جایگاه_مجموع && جایگاه < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++جایگاه) {
            إذا (!allow_fallback && جایگاه > 0) {
                اكسر;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[جایگاه];
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
            ثابت size_t بلوک_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            إذا (بلوک_len == SIZE_MAX) {
                استمر;
            }

            ثابت size_t packed_اندازه = rhs_info->packed_اندازه_ex(n, k, nr, kr, بلوک_len);
            ثابت size_t aligned_cursor = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = پایه_ptr + aligned_cursor;

            إذا (rhs_type == GGML_TYPE_Q4_0) {
                بنية kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_نقطه = 1;
                params.rhs_zero_نقطه = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ثابت uint8_t *>(داده), اشاره_تهی, اشاره_تهی,
                                       dst_ptr, 0, &params);
            } وإلا إذا (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       داده, اشاره_تهی, اشاره_تهی, dst_ptr, 0, اشاره_تهی);
            } وإلا إذا (rhs_type == GGML_TYPE_Q8_0) {
                بنية kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_نقطه = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.داده(), اشاره_تهی, scales.داده(),
                                       dst_ptr, 0, &params);
            } وإلا إذا (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       داده, bias.داده(), اشاره_تهی,
                                       dst_ptr, 0, اشاره_تهی);
            } وإلا {
                استمر;
            }

            سرآیند->offsets[سرآیند->جایگاه_شمار] = aligned_cursor;
            سرآیند->sizes[سرآیند->جایگاه_شمار]   = packed_اندازه;
            ++سرآیند->جایگاه_شمار;

            cursor = aligned_cursor + packed_اندازه;
        }

        إذا (سرآیند->جایگاه_شمار == 0) {
            سرآیند->magic   = 0;
            سرآیند->نسخه_2 = 0;
            کپی_حافظه(tensor->داده, داده, داده_اندازه);
        }

        أرجع 0;
    }
};

ساكن ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_میانگیر_t, بنية ggml_tensor *) {
    ساكن tensor_traits traits;
    أرجع &traits;
}
}  // namespace ggml::cpu::kleidiai

ساكن تعداد ggml_وضعیت ggml_backend_cpu_kleidiai_میانگیر_init_tensor(ggml_backend_میانگیر_t میانگیر, بنية ggml_tensor * tensor) {
    tensor->extra = (فراغ *) ggml::cpu::kleidiai::get_tensor_traits(میانگیر, tensor);

    أرجع GGML_وضعیت_SUCCESS;
}

ساكن فراغ ggml_backend_cpu_kleidiai_میانگیر_مجموعه_tensor(ggml_backend_میانگیر_t میانگیر, بنية ggml_tensor * tensor,
                                                       ثابت فراغ * داده, size_t جابه_جایی, size_t size) {
    GGML_ASSERT(جابه_جایی == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    خودکار tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    خودکار OK            = tensor_traits->repack(tensor, داده, size);

    GGML_ASSERT(OK == 0);
    GGML_استفادهنشده(میانگیر);
}

ساكن ثابت حرف * ggml_backend_cpu_kleidiai_میانگیر_type_get_name(ggml_backend_میانگیر_type_t buft) {
    GGML_استفادهنشده(buft);
    أرجع "CPU_KLEIDIAI";
}

ساكن ggml_backend_میانگیر_t ggml_backend_cpu_kleidiai_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft, size_t size) {
    ggml_backend_میانگیر_t میانگیر = ggml_backend_buft_alloc_میانگیر(ggml_backend_cpu_میانگیر_type(), size);

    إذا (میانگیر == اشاره_تهی) {
        أرجع اشاره_تهی;
    }

    میانگیر->buft              = buft;
    میانگیر->iface.init_tensor = ggml_backend_cpu_kleidiai_میانگیر_init_tensor;
    میانگیر->iface.مجموعه_tensor  = ggml_backend_cpu_kleidiai_میانگیر_مجموعه_tensor;
    میانگیر->iface.get_tensor  = اشاره_تهی;
    میانگیر->iface.cpy_tensor  = اشاره_تهی;
    أرجع میانگیر;
}

ساكن size_t ggml_backend_cpu_kleidiai_میانگیر_type_get_alignment(ggml_backend_میانگیر_type_t buft) {
    GGML_استفادهنشده(buft);
    أرجع TENSOR_ALIGNMENT;
}

ساكن size_t ggml_backend_cpu_kleidiai_میانگیر_type_get_alloc_اندازه(ggml_backend_میانگیر_type_t buft, ثابت بنية ggml_tensor * tensor) {
    GGML_استفادهنشده(buft);

    إذا (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        أرجع ggml_nbytes(tensor);
    }

    ثابت size_t n = tensor->ne[1];
    ثابت size_t k = tensor->ne[0];

    size_t cursor = حجم(kleidiai_وزن_سرآیند);
    cursor = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ثابت منطقي want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ثابت منطقي want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ثابت منطقي want_f32 = tensor->type == GGML_TYPE_F32;
    ثابت صحيح جایگاه_مجموع = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ثابت منطقي allow_fallback = kleidiai_pack_fallback_allowed();

    size_t جایگاه_شمار = 0;
    لكل (صحيح جایگاه = 0; جایگاه < جایگاه_مجموع; ++جایگاه) {
        إذا (!allow_fallback && جایگاه > 0) {
            اكسر;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[جایگاه];
        إذا (!kernels) {
            استمر;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        إذا (!kernel || !rhs_info || !rhs_info->packed_اندازه_ex) {
            استمر;
        }

        ثابت ggml_type rhs_type = kernels->rhs_type;
        ثابت size_t بلوک_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        إذا (بلوک_len == SIZE_MAX) {
            استمر;
        }

        cursor = align_به_بالا(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_اندازه_ex(n, k, kernel->get_nr(), kernel->get_kr(), بلوک_len);
        ++جایگاه_شمار;
    }

    إذا (جایگاه_شمار == 0) {
        أرجع ggml_nbytes(tensor);
    }

    أرجع std::max(cursor, ggml_nbytes(tensor));
}

نطاق ggml::cpu::kleidiai {
صنف extra_میانگیر_type : ggml::cpu::extra_میانگیر_type {
    منطقي supports_op(ggml_backend_dev_t, ثابت بنية ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        إذا (op->op != GGML_OP_MUL_MAT) {
            أرجع خطأ;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحيح جایگاه_مجموع = kleidiai_collect_kernel_chain(op, kernel_chain);
        ثابت منطقي src0_is_kleidiai =
            op->src[0]->میانگیر &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->میانگیر->buft->context == هذا &&
            جایگاه_مجموع > 0;

        إذا ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            إذا (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == اشاره_تهی) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == اشاره_تهی) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == اشاره_تهی) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == اشاره_تهی) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == اشاره_تهی) {
                أرجع خطأ;
            }
            إذا (op->src[1]->میانگیر && !ggml_backend_buft_is_host(op->src[1]->میانگیر->buft)) {
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
                    أرجع درست;
                }
                أرجع خطأ;
            }

            إذا (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                أرجع خطأ;
            }

            أرجع درست;
        }

        أرجع خطأ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ثابت بنية ggml_tensor * op) override {
        إذا (op->op == GGML_OP_MUL_MAT) {
            إذا (op->src[0]->میانگیر && op->src[0]->میانگیر->buft->context == هذا) {
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
                    إذا (!warned.exchange(درست)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                إذا (op->src[0]->type != GGML_TYPE_F16) {
                    أرجع اشاره_تهی;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ثابت صحيح جایگاه_مجموع = kleidiai_collect_kernel_chain(op, kernel_chain);
                إذا (جایگاه_مجموع > 0 && op->src[1]->ne[1] > 1) {
                    إذا ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        أرجع اشاره_تهی;
                    }
                    أرجع ggml::cpu::kleidiai::get_tensor_traits(عدم, عدم);
                }
            }
        }
        أرجع اشاره_تهی;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_میانگیر_type_t ggml_backend_cpu_kleidiai_میانگیر_type(فراغ) {
    ساكن ggml::cpu::kleidiai::extra_میانگیر_type ctx;
    ساكن بنية ggml_backend_میانگیر_type ggml_backend_cpu_میانگیر_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_میانگیر_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_میانگیر_type_alloc_میانگیر,
                           /* .alloc_buffer_n       = */ اشاره_تهی,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_میانگیر_type_get_alignment,
                           /* .get_max_size         = */ اشاره_تهی,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_میانگیر_type_get_alloc_اندازه,
                           /* .get_alloc_size_n     = */ عدم,
                           /* .is_host              = */ اشاره_تهی,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    أرجع &ggml_backend_cpu_میانگیر_type_kleidiai;
}
