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
#جے_انت

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

#تعریف_کرو GGML_COMMON_DECL_CPP
#ضمن "ggml-common.h"

ساكن constexpr صحيح      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
ساكن constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
ساكن constexpr uint16_t GGML_KLEIDIAI_PACK_نسخہ     = 1;
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
} ساكن ctx = { CPU_FEATURE_NONE, خالی_سوچک, خالی_سوچک, خالی_سوچک, خالی_سوچک, 0, -1, 4 };

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
    إذا (لڑی_مقابلہ(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        أرجع خطأ;
    }

    ثابت حرف* پہلا = name + 3;
    ثابت حرف* آخری = name + لڑی_لمبائی(name);

    size_t value = 0;
    ثابت خودکار [end, ec] = std::from_chars(پہلا, آخری, value, 10);

    إذا (ec != std::errc{} || end != آخری) {
        أرجع خطأ;
    }

    *cpu = value;
    أرجع سچ;
}

ساكن std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    إذا (dir == خالی_سوچک) {
        أرجع cpus;
    }

    طالما (dirent * اندراج = readdir(dir)) {
        size_t cpu = 0;
        إذا (parse_cpu_dir_name(اندراج->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    أرجع cpus;
}
#جے_انت

#إذا defined(__APPLE__) && defined(__aarch64__)
ساكن منطقي apple_sme_counted_perf_level(std::string name) {
    لكل (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (حرف) std::tolower((بنانشانی حرف) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    أرجع name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#جے_انت

ساكن فراغ جوڑو_smcus_from_smidr(uint64_t smidr, size_t & عدد_نجی, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ثابت uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ثابت uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ثابت size_t shared_گنتی = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ثابت uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ثابت uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ثابت uint32_t id = (affinity2 << 12) | affinity;

    إذا (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    بدل (sh) {
        حالة 2: // private SMCU
            ++عدد_نجی;
            اكسر;
        حالة 3: // shared SMCU
            إذا (shared_counts[id] < shared_گنتی) {
                shared_counts[id] = shared_گنتی;
            }
            اكسر;
        حالة 0:
            إذا (id == 0) {
                ++عدد_نجی;
            } وإلا إذا (shared_counts[id] < shared_گنتی) {
                shared_counts[id] = shared_گنتی;
            }
            اكسر;
        افتراضي:
            اكسر;
    }
}

ساكن size_t detect_عدد_smcus() {
    ثابت خودکار runtime_feat = ggml_feats_get_arch64_runtime();
    إذا (!runtime_feat.has_sme) {
        أرجع 0;
    }

#إذا defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t عدد_نجی = 0;
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

        جوڑو_smcus_from_smidr(smidr, عدد_نجی, shared_counts);
    }

    size_t کل = عدد_نجی;
    لكل (ثابت خودکار & اندراج : shared_counts) {
        کل += اندراج.second;
    }
    أرجع کل;

#وإلاإذا defined(__APPLE__) && defined(__aarch64__)
    صحيح perf_پدھراں = 0;
    size_t size = حجم(perf_پدھراں);
    إذا (sysctlbyname("hw.nperflevels", &perf_پدھراں, &size, خالی_سوچک, 0) != 0 ||
        size != حجم(perf_پدھراں) || perf_پدھراں <= 0) {
        أرجع 0;
    }

    size_t units = 0;
    لكل (صحيح i = 0; i < perf_پدھراں; ++i) {
        حرف کلید[64] = {};
        صحيح physical_cpus = 0;
        صحيح cpus_فی_l2 = 0;

        snprintf(کلید, حجم(کلید), "hw.perflevel%d.physicalcpu", i);
        size = حجم(physical_cpus);
        إذا (sysctlbyname(کلید, &physical_cpus, &size, خالی_سوچک, 0) != 0 ||
            size != حجم(physical_cpus) || physical_cpus <= 0) {
            استمر;
        }

        snprintf(کلید, حجم(کلید), "hw.perflevel%d.cpusperl2", i);
        size = حجم(cpus_فی_l2);
        إذا (sysctlbyname(کلید, &cpus_فی_l2, &size, خالی_سوچک, 0) != 0 ||
            size != حجم(cpus_فی_l2) || cpus_فی_l2 <= 0) {
            استمر;
        }

        snprintf(کلید, حجم(کلید), "hw.perflevel%d.name", i);
        size = 0;
        إذا (sysctlbyname(کلید, خالی_سوچک, &size, خالی_سوچک, 0) != 0 || size == 0) {
            استمر;
        }

        std::string name(size, '\0');
        إذا (sysctlbyname(کلید, &name[0], &size, خالی_سوچک, 0) != 0) {
            استمر;
        }
        name.resize(size);
        طالما (!name.خالی_تھاواں() && name.back() == '\0') {
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
#جے_انت
}

ساكن صحيح parse_uint_env(ثابت حرف *s, ثابت حرف *name, منطقي *ok) {
    إذا (!s) { *ok = خطأ; أرجع 0; }
    حرف *end = خالی_سوچک;
    لما v = strtol(s, &end, 10);
    إذا (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = خطأ;
        أرجع 0;
    }
    إذا (v < 0 || v > ودھ_توں_ودھ_عدد) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = خطأ;
        أرجع 0;
    }
    *ok = سچ;
    أرجع (صحيح)v;
}

ساكن فراغ init_kleidiai_context(فراغ) {
    ggml_critical_section_شروع();
    ساكن منطقي initialized = خطأ;

    إذا (!initialized) {
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

        إذا (env_دھاگے) {
            منطقي ok = خطأ;
            صحيح hint = parse_uint_env(env_دھاگے, "GGML_TOTAL_THREADS", &ok);
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
        منطقي sme_env_سیٹ = (env_sme != خالی_سوچک);

        ثابت منطقي has_supported_sme_family = runtime_feat.has_sme;
        منطقي sme_cap_detected = خطأ;

        إذا (has_supported_sme_family) {
            detected_smcus = detect_عدد_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (صحيح)detected_smcus : 1;

            إذا (!sme_env_سیٹ && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        إذا (sme_env_سیٹ) {
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
            إذا (sme_env_سیٹ && sme_env_ok && sme_cores > 0) {
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

    ggml_critical_section_سرا();
}

ساكن مضمن صحيح kleidiai_sme_thread_cap() {
    أرجع ctx.sme_thread_cap;
}

ساكن مضمن size_t align_اتے(size_t value, size_t alignment) {
    إذا (alignment == 0) {
        أرجع value;
    }
    ثابت size_t remainder = value % alignment;
    أرجع remainder == 0 ? value : value + (alignment - remainder);
}

ساكن مضمن size_t عاد_اعظم_ناپ(size_t a, size_t b) {
    طالما (b != 0) {
        ثابت size_t t = a % b;
        a = b;
        b = t;
    }
    أرجع a;
}

ساكن مضمن منطقي lcm_ناپ(size_t a, size_t b, size_t & نتیجہ) {
    إذا (a == 0 || b == 0) {
        نتیجہ = 0;
        أرجع خطأ;
    }
    ثابت size_t g = عاد_اعظم_ناپ(a, b);
    ثابت size_t q = a / g;
    إذا (q > SIZE_MAX / b) {
        أرجع خطأ;
    }
    نتیجہ = q * b;
    أرجع سچ;
}

ساكن مضمن size_t ceil_div_ناپ(size_t a, size_t b) {
    أرجع b == 0 ? 0 : (a + b - 1) / b;
}

ساكن مضمن size_t kleidiai_chunk_خانے_گنتی(size_t n, صحيح nth_کل, منطقي disable_chunking, size_t n_قدم) {
    ثابت size_t multiplier = (nth_کل == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ثابت size_t مقسوم_علیہ = std::max<size_t>(1, (size_t) nth_کل * multiplier);
    ثابت size_t chunk_خانے_گنتی = align_اتے(std::max<size_t>(1, ceil_div_ناپ(n, مقسوم_علیہ)), n_قدم);
    أرجع chunk_خانے_گنتی ? chunk_خانے_گنتی : n_قدم;
}

بنية kleidiai_بلاک_دلیلاں {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

ساكن مضمن kleidiai_بلاک_دلیلاں kleidiai_get_بلاک_دلیلاں(ggml_type rhs_type) {
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
        أرجع سچ;
    }
    أرجع ctx.thread_hint > ctx.sme_thread_cap;
}

بنية kleidiai_بھار_سرخی {
    uint32_t magic;
    uint16_t نسخہ_2;
    uint16_t تھاں_گنتی;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

ساكن مضمن kleidiai_بھار_سرخی * kleidiai_بھار_سرخی_from_ptr(فراغ * انکڑے) {
    أرجع reinterpret_cast<kleidiai_بھار_سرخی *>(انکڑے);
}

ساكن مضمن ثابت kleidiai_بھار_سرخی * kleidiai_بھار_سرخی_from_ptr(ثابت فراغ * انکڑے) {
    أرجع reinterpret_cast<ثابت kleidiai_بھار_سرخی *>(انکڑے);
}

ساكن مضمن منطقي kleidiai_is_بھار_سرخی_valid(ثابت kleidiai_بھار_سرخی * سرخی) {
    إذا (!سرخی) {
        أرجع خطأ;
    }
    إذا (سرخی->magic != GGML_KLEIDIAI_PACK_MAGIC || سرخی->نسخہ_2 != GGML_KLEIDIAI_PACK_نسخہ) {
        أرجع خطأ;
    }
    إذا (سرخی->تھاں_گنتی == 0 || سرخی->تھاں_گنتی > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        أرجع خطأ;
    }
    أرجع سچ;
}

ساكن مضمن uint8_t * kleidiai_بھار_تھاں_ptr(kleidiai_بھار_سرخی * سرخی, صحيح تھاں) {
    إذا (!kleidiai_is_بھار_سرخی_valid(سرخی)) {
        أرجع خالی_سوچک;
    }
    إذا (تھاں < 0 || تھاں >= سرخی->تھاں_گنتی) {
        أرجع خالی_سوچک;
    }
    أرجع reinterpret_cast<uint8_t *>(سرخی) + سرخی->offsets[تھاں];
}

ساكن مضمن ثابت uint8_t * kleidiai_بھار_تھاں_ptr(ثابت kleidiai_بھار_سرخی * سرخی, صحيح تھاں) {
    إذا (!kleidiai_is_بھار_سرخی_valid(سرخی)) {
        أرجع خالی_سوچک;
    }
    إذا (تھاں < 0 || تھاں >= سرخی->تھاں_گنتی) {
        أرجع خالی_سوچک;
    }
    أرجع reinterpret_cast<ثابت uint8_t *>(سرخی) + سرخی->offsets[تھاں];
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

سانچا <قسم_ناں SelectFallback>
ساكن صحيح kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & باہرلا,
        SelectFallback select_fallback) {
    صحيح count = 0;
    إذا (!primary) {
        أرجع 0;
    }
    باہرلا[count++] = primary;

    إذا (primary->rhs_info.repack_منوال == RHS_REPACK_SINGLE_ONLY) {
        أرجع count;
    }

    إذا (is_sme_family(primary->required_cpu)) {
        ثابت cpu_feature fallback_نقاب = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        إذا (fallback_نقاب != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_نقاب);
            إذا (fallback && fallback != primary &&
                fallback->rhs_info.repack_منوال != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                باہرلا[count++] = fallback;
            }
        }
    }

    أرجع count;
}

ساكن صحيح kleidiai_collect_kernel_chain(ثابت بنية ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & باہرلا) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, باہرلا,
        [&](cpu_feature نقاب) { أرجع ggml_kleidiai_select_kernels(نقاب, op); });
}

ساكن صحيح kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & باہرلا) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, باہرلا,
        [&](cpu_feature نقاب) { أرجع ggml_kleidiai_select_kernels_q4_0(نقاب); });
}

ساكن صحيح kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & باہرلا) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, باہرلا,
        [&](cpu_feature نقاب) { أرجع ggml_kleidiai_select_kernels_q4_k(نقاب); });
}

ساكن صحيح kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & باہرلا) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, باہرلا,
        [&](cpu_feature نقاب) { أرجع ggml_kleidiai_select_kernels_q8_0(نقاب); });
}

ساكن صحيح kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & باہرلا) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    أرجع kleidiai_collect_kernel_chain_common(primary, ctx.features, باہرلا,
        [&](cpu_feature نقاب) { أرجع ggml_kleidiai_select_kernels_f32(نقاب); });
}

ساكن مضمن int64_t ggml_ne(ثابت ggml_tensor * tensor, صحيح dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    أرجع tensor->ne[dim];
}

نطاق ggml::cpu::kleidiai {

ساكن size_t گیڑا_تھلے(size_t x, size_t y) {
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
    منطقي work_ناپ(صحيح /* n_threads */, ثابت بنية ggml_tensor * op, size_t & size) override {
        إذا (op->op != GGML_OP_MUL_MAT) {
            أرجع خطأ;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحيح تھاں_گنتی = kleidiai_collect_kernel_chain(op, kernel_chain);
        إذا (تھاں_گنتی == 0) {
            أرجع خطأ;
        }

        ثابت منطقي is_gemv = op->src[1]->ne[1] == 1;
        ثابت size_t k = op->src[0]->ne[0];
        ثابت size_t n = op->src[0]->ne[1];
        ثابت size_t m = op->src[1]->ne[1];

        إذا (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ثابت kleidiai_بلاک_دلیلاں بلاک_دلیلاں = kleidiai_get_بلاک_دلیلاں(kernel_chain[0]->rhs_type);
            ثابت size_t qk = بلاک_دلیلاں.pack_bl;

            size_t cursor = 0;
            منطقي any_تھاں = خطأ;

            لكل (صحيح تھاں = 0; تھاں < تھاں_گنتی; ++تھاں) {
                ggml_kleidiai_kernels * kernels = kernel_chain[تھاں];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                إذا (!lhs_info || !lhs_info->packed_ناپ_ex || !kernel) {
                    أرجع خطأ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                ثابت size_t packed = lhs_info->packed_ناپ_ex(m, k, qk, mr, kr, sr);

                cursor = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_تھاں = سچ;
            }

            إذا (!any_تھاں) {
                أرجع خطأ;
            }

            size = cursor;
            أرجع سچ;
        }

        إذا (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : خالی_سوچک;
            إذا (is_gemv && op->src[1]->nb[0] == (int64_t) حجم(عائم) && gemv_kernel &&
                gemv_kernel->get_lhs_وتھ_ex && gemv_kernel->get_rhs_packed_وتھ_ex &&
                gemv_kernel->لگاتار_kernel_ex && gemv_kernel->get_dst_وتھ) {
                size = 0;
                أرجع سچ;
            }

            size_t cursor = 0;
            منطقي any_تھاں = خطأ;

            لكل (صحيح تھاں = 0; تھاں < تھاں_گنتی; ++تھاں) {
                ggml_kleidiai_kernels * kernels = kernel_chain[تھاں];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                إذا (!lhs_info || !lhs_info->packed_ناپ_ex || !kernel) {
                    أرجع خطأ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_ناپ_ex(m, k, 0, mr, kr, sr);
                any_تھاں = سچ;
            }

            إذا (!any_تھاں) {
                أرجع خطأ;
            }

            size = cursor;
            أرجع سچ;
        }

        إذا (op->src[0]->type == GGML_TYPE_F16) {
            ثابت int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ثابت int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ثابت int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            منطقي any_تھاں = خطأ;

            لكل (صحيح تھاں = 0; تھاں < تھاں_گنتی; ++تھاں) {
                ggml_kleidiai_kernels * kernels = kernel_chain[تھاں];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                إذا (!lhs_info || !lhs_info->packed_ناپ_ex || !kernels->rhs_info.packed_ناپ_ex || !kernel) {
                    أرجع خطأ;
                }

                ثابت size_t mr = kernel->get_mr();
                ثابت size_t kr = kernel->get_kr();
                ثابت size_t sr = kernel->get_sr();

                cursor  = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_ناپ_ex(m * r, k, 0, mr, kr, sr);
                any_تھاں = سچ;
            }

            لكل (صحيح تھاں = 0; تھاں < تھاں_گنتی; ++تھاں) {
                ggml_kleidiai_kernels * kernels = kernel_chain[تھاں];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                إذا (!kernel || !kernels->rhs_info.packed_ناپ_ex) {
                    أرجع خطأ;
                }
                cursor  = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_ناپ_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * حجم(عائم);
            cursor  = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * حجم(عائم);

            إذا (!any_تھاں) {
                أرجع خطأ;
            }

            size = cursor;
            أرجع سچ;
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

        GGML_TENSOR_ثنائی_OP_LOCALS

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
                              kernels->gemv.get_lhs_وتھ_ex &&
                              kernels->gemv.get_rhs_packed_وتھ_ex &&
                              kernels->gemv.لگاتار_kernel_ex &&
                              kernels->gemv.get_dst_وتھ;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        إذا (!kernel || !kernel->get_lhs_وتھ_ex ||
            !kernel->get_rhs_packed_وتھ_ex || !kernel->لگاتار_kernel_ex || !kernel->get_dst_وتھ) {
            أرجع خطأ;
        }

        إذا (!use_gemv && (!lhs_info || !lhs_info->get_وتھ || !lhs_info->get_packed_وتھ_ex ||
                          !lhs_info->packed_ناپ_ex || !lhs_info->pack_func_ex)) {
            أرجع خطأ;
        }

        ثابت kleidiai_بھار_سرخی * سرخی = kleidiai_بھار_سرخی_from_ptr(src0->انکڑے);
        ثابت منطقي has_سرخی = kleidiai_is_بھار_سرخی_valid(سرخی);

        ثابت uint8_t * rhs_ادھار = has_سرخی ? kleidiai_بھار_تھاں_ptr(سرخی, 0)
                                              : static_cast<ثابت uint8_t *>(src0->انکڑے);
        إذا (!rhs_ادھار) {
            أرجع خطأ;
        }

        ثابت صحيح nth = params->nth > 0 ? params->nth : 1;
        ثابت صحيح ith = params->ith;

        ثابت size_t mr = kernel->get_mr();
        ثابت size_t kr = kernel->get_kr();
        ثابت size_t sr = kernel->get_sr();

        ثابت size_t lhs_packed_ناپ = use_gemv ? 0 : lhs_info->packed_ناپ_ex(m, k, 0, mr, kr, sr);
        إذا (!use_gemv) {
            GGML_ASSERT(lhs_packed_ناپ <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ثابت size_t dst_stride = dst->nb[1];
        ثابت size_t n_قدم = kernel->get_n_قدم() ? kernel->get_n_قدم() : 1;
        ثابت منطقي disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) ودھ_توں_ودھ_عدد);

        ثابت int64_t batch_ناپ = ne12 * ne13;
        لكل (int64_t batch_idx = 0; batch_idx < batch_ناپ; ++batch_idx) {
            ثابت int64_t i12 = batch_idx % ne12;
            ثابت int64_t i13 = batch_idx / ne12;
            ثابت uint8_t * lhs_batch_ادھار = static_cast<ثابت uint8_t *>(src1->انکڑے) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_ادھار = static_cast<uint8_t *>(dst->انکڑے) + i12 * dst->nb[2] + i13 * dst->nb[3];

            إذا (!use_gemv) {
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_دھاگے = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_دھاگے = std::max<int64_t>(1, max_دھاگے);
                ثابت int64_t use_دھاگے = std::min<int64_t>(nth, max_دھاگے);

                إذا (ith < use_دھاگے) {
                    ثابت int64_t عدد_m_فی_thread0   = گیڑا_تھلے((size_t)(m_roundup_mr / use_دھاگے), mr);
                    ثابت int64_t عدد_m_فی_thread_N_1 = (int64_t)m - (use_دھاگے - 1) * عدد_m_فی_thread0;

                    ثابت int64_t m_شروع = (int64_t)ith * عدد_m_فی_thread0;
                    ثابت int64_t m_گنتی = (ith == use_دھاگے - 1) ? عدد_m_فی_thread_N_1 : عدد_m_فی_thread0;

                    ثابت size_t ادھار_packed_off  = lhs_info->get_packed_وتھ_ex(m_شروع, k, 0, mr, kr, sr);
                    ثابت size_t اگلا_بلاک_off   = lhs_info->get_packed_وتھ_ex(m_شروع + mr, k, 0, mr, kr, sr);
                    ثابت size_t صف_stride_بائٹاں = mr ? (اگلا_بلاک_off - ادھار_packed_off) / mr : 0;

                    int64_t بچیا = m_گنتی;
                    int64_t cur       = m_شروع;

                    طالما (بچیا > 0) {
                        ثابت int64_t take = std::min<int64_t>((int64_t)m - cur, بچیا);
                        ثابت size_t src_off = lhs_info->get_وتھ(cur, src1->nb[1]);
                        ثابت فراغ * src_ptr = lhs_batch_ادھار + src_off;
                        ثابت size_t dst_off = ادھار_packed_off + (size_t)(cur - m_شروع) * صف_stride_بائٹاں;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        بچیا -= take;
                    }
                }
            }

            إذا (ith == 0) {
                ggml_threadpool_chunk_سیٹ(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ثابت size_t chunk_خانے_گنتی = kleidiai_chunk_خانے_گنتی(n, nth, disable_chunking, n_قدم);
            GGML_ASSERT(chunk_خانے_گنتی <= (size_t) ودھ_توں_ودھ_عدد);

            صحيح current_خانہ = ggml_threadpool_chunk_جوڑو(params->threadpool, (صحيح) chunk_خانے_گنتی);
            طالما ((size_t) current_خانہ < n) {
                ثابت size_t n_شروع = (size_t) current_خانہ;
                ثابت size_t n_منزل_process = std::min(chunk_خانے_گنتی, n - n_شروع);

                إذا (n_منزل_process > 0) {
                    ثابت size_t lhs_وتھ = use_gemv ? kernel->get_lhs_وتھ_ex(0, k, 0)
                                                       : lhs_info->get_packed_وتھ_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_وتھ = kernel->get_rhs_packed_وتھ_ex(n_شروع, k, 0);
                    ثابت size_t dst_وتھ        = kernel->get_dst_وتھ(0, n_شروع, dst_stride);

                    ثابت فراغ * lhs_ptr = use_gemv ? lhs_batch_ادھار + lhs_وتھ
                                                    : lhs_packed + lhs_وتھ;
                    ثابت فراغ * rhs_ptr = rhs_ادھار + rhs_packed_وتھ;
                    عائم * dst_ptr      = reinterpret_cast<عائم *>(dst_batch_ادھار + dst_وتھ);

                    kernel->لگاتار_kernel_ex(m, n_منزل_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          حجم(عائم),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_خانہ = ggml_threadpool_chunk_جوڑو(params->threadpool, (صحيح) chunk_خانے_گنتی);
            }

            إذا (batch_idx != batch_ناپ - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        أرجع سچ;
    }

    منطقي compute_forward_fp16(ggml_compute_params * params, بنية ggml_tensor * dst) {
        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ثنائی_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        إذا (!kernels) {
            أرجع خطأ;
        }

        ثابت منطقي is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        إذا (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_وتھ_ex || !kernel->get_rhs_packed_وتھ_ex || !kernel->لگاتار_kernel_ex) {
            أرجع خطأ;
        }

        ثابت صحيح nth = params->nth;
        ثابت صحيح ith = params->ith;

        ثابت int64_t lhs_batch_size0 = ne12;
        ثابت int64_t rhs_batch_size0 = ne02;
        ثابت int64_t batch_ناپ      = lhs_batch_size0;

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

        ثابت size_t lhs_packed_ناپ = lhs_info->packed_ناپ_ex(m, k, 0, mr, kr, sr);
        ثابت size_t rhs_packed_ناپ = kernels->rhs_info.packed_ناپ_ex(n, k, nr, kr, 0);
        ثابت size_t kxn_ناپ        = k * n * حجم(عائم);
        ثابت size_t bias_ناپ       = n * حجم(عائم);

        ثابت size_t wsize_required = lhs_packed_ناپ + rhs_packed_ناپ + kxn_ناپ + bias_ناپ;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_ناپ;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_ناپ;
        uint8_t * bias       = rhs_kxn + kxn_ناپ;

        لكل (int64_t batch_idx = 0; batch_idx < batch_ناپ; ++batch_idx) {
            ثابت int64_t rhs_batch_idx = batch_idx / r;
            ثابت uint8_t * rhs_batch_ادھار = static_cast<ثابت uint8_t *>(src0->انکڑے) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_ادھار = static_cast<uint8_t *>(dst->انکڑے) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ثابت int64_t m_roundup_mr = kai_roundup(m, mr);
                ثابت int64_t عدد_دھاگے  = KAI_MIN(m_roundup_mr / mr, nth);

                إذا (ith < عدد_دھاگے) {
                    ثابت int64_t عدد_m_فی_thread0   = گیڑا_تھلے((size_t)(m_roundup_mr / عدد_دھاگے), (size_t)mr);
                    ثابت int64_t عدد_m_فی_thread_N_1 = m - (عدد_دھاگے - 1) * عدد_m_فی_thread0;

                    ثابت int64_t m_شروع = ith * عدد_m_فی_thread0;
                    ثابت int64_t m_گنتی = (ith == عدد_دھاگے - 1) ? عدد_m_فی_thread_N_1 : عدد_m_فی_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ثابت size_t ادھار_packed_off  = lhs_info->get_packed_وتھ_ex(m_شروع, k, 0, mr, kr, sr);
                    ثابت size_t اگلا_بلاک_off   = lhs_info->get_packed_وتھ_ex(m_شروع + mr, k, 0, mr, kr, sr);
                    ثابت size_t صف_stride_بائٹاں = (اگلا_بلاک_off - ادھار_packed_off) / (size_t)mr;

                    int64_t بچیا = m_گنتی;
                    int64_t cur       = m_شروع;

                    طالما (بچیا > 0) {
                        ثابت int64_t صف_وچ_گروہ = cur;
                        ثابت int64_t avail        = m_گروہ - صف_وچ_گروہ;
                        ثابت int64_t take         = std::min(avail, بچیا);

                        ثابت uint8_t * lhs_batch_ادھار = static_cast<ثابت uint8_t *>(src1->انکڑے) + batch_idx * src1->nb[2];
                        ثابت فراغ * src_ptr = lhs_batch_ادھار + (size_t)صف_وچ_گروہ * lhs_stride;
                        ثابت size_t dst_off = ادھار_packed_off + (size_t)(cur - m_شروع) * صف_stride_بائٹاں;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        بچیا -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            إذا (ith == 0) {
                memset(bias, 0, (size_t)n * حجم(عائم));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<عائم *>(rhs_kxn),
                                        reinterpret_cast<ثابت uint16_t *>(rhs_batch_ادھار),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * حجم(عائم),
                             rhs_kxn, bias, خالی_سوچک, rhs_packed, 0, خالی_سوچک);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ثابت int64_t n_قدم  = (int64_t) kernel->get_n_قدم();
                int64_t عدد_دھاگے_n = KAI_MIN(n / n_قدم, nth);
                إذا (عدد_دھاگے_n <= 0) {
                    عدد_دھاگے_n = 1;
                }

                إذا (ith < عدد_دھاگے_n) {
                    ثابت int64_t عدد_n_فی_thread0   = گیڑا_تھلے((size_t)(n / عدد_دھاگے_n), (size_t)n_قدم);
                    ثابت int64_t عدد_n_فی_thread_N_1 = n - (عدد_دھاگے_n - 1) * عدد_n_فی_thread0;

                    ثابت int64_t n_شروع      = ith * عدد_n_فی_thread0;
                    ثابت int64_t n_منزل_process = (ith == عدد_دھاگے_n - 1) ? عدد_n_فی_thread_N_1 : عدد_n_فی_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ثابت size_t lhs_packed_offset0 = lhs_info->get_packed_وتھ_ex(0, k, 0, mr, kr, sr);
                    ثابت size_t rhs_packed_وتھ  = kernel->get_rhs_packed_وتھ_ex(n_شروع, k, 0);
                    ثابت size_t dst_وتھ         = kernel->get_dst_وتھ((size_t)0, (size_t)n_شروع, dst_stride);

                    ثابت فراغ * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ثابت فراغ * rhs_ptr = rhs_packed + rhs_packed_وتھ;
                    عائم * dst_ptr      = reinterpret_cast<عائم *>(dst_batch_ادھار + dst_وتھ);

                    kernel->لگاتار_kernel_ex(m, n_منزل_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, حجم(عائم), -FLT_MAX, FLT_MAX);
                }
            }

            إذا (batch_idx != batch_ناپ - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        أرجع سچ;
    }

    منطقي compute_forward_qx(بنية ggml_compute_params * params, بنية ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ثابت ggml_tensor * src0 = dst->src[0];
        ثابت ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_ثنائی_OP_LOCALS

        ثابت kleidiai_بھار_سرخی * سرخی = kleidiai_بھار_سرخی_from_ptr(src0->انکڑے);
        ثابت منطقي has_سرخی = kleidiai_is_بھار_سرخی_valid(سرخی);
        ثابت منطقي is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت صحيح تھاں_کل = kleidiai_collect_kernel_chain(dst, kernel_chain);

        خودکار بھار_for_تھاں = [&](صحيح تھاں_اشاریہ, size_t & ناپ_باہرلا) -> ثابت uint8_t * {
            إذا (تھاں_اشاریہ < 0 || تھاں_اشاریہ >= تھاں_کل) {
                أرجع خالی_سوچک;
            }
            إذا (has_سرخی) {
                إذا (تھاں_اشاریہ < سرخی->تھاں_گنتی) {
                    ناپ_باہرلا = static_cast<size_t>(سرخی->sizes[تھاں_اشاریہ]);
                    أرجع kleidiai_بھار_تھاں_ptr(سرخی, تھاں_اشاریہ);
                }
                أرجع خالی_سوچک;
            }
            إذا (تھاں_اشاریہ == 0) {
                ناپ_باہرلا = ggml_nbytes(src0);
                أرجع static_cast<ثابت uint8_t *>(src0->انکڑے);
            }
            أرجع خالی_سوچک;
        };

        بنية runtime_تھاں {
            صحيح تھاں_اشاریہ;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            size_t mr;
            size_t nr;
            size_t kr;
            size_t sr;
            size_t n_قدم;
            size_t lhs_packed_ناپ;
            size_t lhs_وتھ;
            size_t lhs_bl;
            size_t rhs_bl;
            size_t pack_bl;
            size_t lhs_packed_offset0;
            صحيح assigned_دھاگے;
            صحيح thread_begin;
            صحيح thread_سرا;
            ثابت uint8_t * rhs_ادھار;
        };

        std::array<runtime_تھاں, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        صحيح runtime_گنتی = 0;

        لكل (صحيح تھاں = 0; تھاں < تھاں_کل && runtime_گنتی < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++تھاں) {
            ggml_kleidiai_kernels * kernels = kernel_chain[تھاں];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            إذا (!kinfo || !linfo || !linfo->packed_ناپ_ex || !linfo->pack_func_ex || !linfo->get_وتھ ||
                !kinfo->get_rhs_packed_وتھ_ex || !kinfo->لگاتار_kernel_ex || !kinfo->get_dst_وتھ) {
                استمر;
            }

            size_t rhs_ناپ = 0;
            ثابت uint8_t * rhs_ptr = بھار_for_تھاں(تھاں, rhs_ناپ);
            إذا (!rhs_ptr || rhs_ناپ == 0) {
                استمر;
            }

            ثابت kleidiai_بلاک_دلیلاں بلاک_دلیلاں = kleidiai_get_بلاک_دلیلاں(kernels->rhs_type);

            runtime[runtime_گنتی] = {
                تھاں,
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
                بلاک_دلیلاں.lhs_bl,
                بلاک_دلیلاں.rhs_bl,
                بلاک_دلیلاں.pack_bl,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_گنتی;
        }

        إذا (runtime_گنتی == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            أرجع خطأ;
        }

        ثابت صحيح nth_کل = params->nth > 0 ? params->nth : 1;
        ثابت صحيح ith_کل = params->ith;

        صحيح sme_تھاں = -1;
        صحيح non_sme_تھاں = -1;
        لكل (صحيح i = 0; i < runtime_گنتی; ++i) {
            إذا (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_تھاں = i;
                اكسر;
            }
        }

        لكل (صحيح i = 0; i < runtime_گنتی; ++i) {
            إذا (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_تھاں = i;
                اكسر;
            }
        }

        ثابت صحيح sme_cap_حد_بندی = ctx.sme_thread_cap;
        ثابت منطقي use_hybrid = sme_cap_حد_بندی > 0 &&
                                 runtime_گنتی > 1 &&
                                 nth_کل > sme_cap_حد_بندی;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_خانے_گنتی_فی_thread = 0;
        إذا (runtime_گنتی > 0 && nth_کل > 0) {
            min_خانے_گنتی_فی_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_کل);
        }
        ثابت منطقي too_small_for_hybrid = (min_خانے_گنتی_فی_thread < 2) || (ne11 < 128);

        ثابت منطقي hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        إذا (!hybrid_enabled) {
            صحيح chosen_تھاں = 0;
            إذا (too_small_for_hybrid && sme_تھاں != -1) {
                chosen_تھاں = nth_کل > sme_cap_حد_بندی && non_sme_تھاں != -1 ? non_sme_تھاں : sme_تھاں;
            } وإلا إذا (runtime_گنتی > 1 && ctx.sme_thread_cap > 0 && nth_کل > ctx.sme_thread_cap) {
                chosen_تھاں = 1;
            }
            إذا (chosen_تھاں != 0 && chosen_تھاں < runtime_گنتی) {
                runtime[0] = runtime[chosen_تھاں];
                runtime[0].assigned_دھاگے = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_سرا = 0;
            }
            runtime_گنتی = runtime_گنتی > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_تھاں = -1;
            إذا (runtime_گنتی > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_تھاں = 0;
            }
        }

        ثابت منطقي sme_single_only = sme_تھاں != -1 &&
                                     runtime[sme_تھاں].kernels->rhs_info.repack_منوال == RHS_REPACK_SINGLE_ONLY;
        صحيح sme_cap = sme_single_only ? nth_کل : kleidiai_sme_thread_cap();
        إذا (sme_cap < 0) {
            sme_cap = nth_کل;
        }
        sme_cap = std::min(sme_cap, nth_کل);

        صحيح دھاگے_بچیا = nth_کل;
        إذا (sme_تھاں != -1) {
            صحيح sme_دھاگے = std::min(std::max(sme_cap, 0), دھاگے_بچیا);
            runtime[sme_تھاں].assigned_دھاگے = sme_دھاگے;
            دھاگے_بچیا -= sme_دھاگے;
        }

        صحيح fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        صحيح fallback_گنتی = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        لكل (صحيح i = 0; i < runtime_گنتی; ++i) {
            إذا (i == sme_تھاں) {
                استمر;
            }
            fallback_indices[fallback_گنتی++] = i;
        }

        لكل (صحيح fi = 0; fi < fallback_گنتی; ++fi) {
            إذا (دھاگے_بچیا <= 0) {
                اكسر;
            }
            ثابت صحيح تھاں_اشاریہ = fallback_indices[fi];
            ثابت صحيح slots_بایاں = fallback_گنتی - fi;
            صحيح share = (دھاگے_بچیا + slots_بایاں - 1) / slots_بایاں;
            share     = std::min(share, دھاگے_بچیا);
            runtime[تھاں_اشاریہ].assigned_دھاگے = share;
            دھاگے_بچیا -= share;
        }

        صحيح thread_cursor = 0;
        لكل (صحيح i = 0; i < runtime_گنتی; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_دھاگے;
            runtime[i].thread_سرا = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ثابت صحيح سرگرم_دھاگے = thread_cursor;

        صحيح local_تھاں = -1;
        صحيح local_ith  = 0;
        لكل (صحيح i = 0; i < runtime_گنتی; ++i) {
            إذا (ith_کل >= runtime[i].thread_begin && ith_کل < runtime[i].thread_سرا) {
                local_تھاں = i;
                local_ith  = ith_کل - runtime[i].thread_begin;
                اكسر;
            }
        }
        ثابت size_t k = ne00;
        ثابت size_t m = ne11;
        ثابت size_t n = ne01;

        size_t cursor = 0;
        لكل (صحيح i = 0; i < runtime_گنتی; ++i) {
            runtime[i].lhs_packed_ناپ = runtime[i].lhs_info->packed_ناپ_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_وتھ = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_وتھ_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_ناپ;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_قدم = 1;
        لكل (صحيح i = 0; i < runtime_گنتی; ++i) {
            إذا (runtime[i].assigned_دھاگے == 0) {
                استمر;
            }
            size_t اگلا_قدم = 0;
            إذا (!lcm_ناپ(common_قدم, runtime[i].n_قدم ? runtime[i].n_قدم : 1, اگلا_قدم)) {
                أرجع خطأ;
            }
            common_قدم = اگلا_قدم;
        }
        GGML_ASSERT(common_قدم > 0);

        ثابت منطقي disable_chunking = ggml_is_numa();
        ثابت size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ثابت size_t chunk_مقسوم_علیہ = (سرگرم_دھاگے == 1 || disable_chunking) ? (size_t) سرگرم_دھاگے : (size_t) سرگرم_دھاگے * chunk_multiplier;
        size_t chunk_خانے_گنتی = align_اتے(std::max<size_t>(1, ceil_div_ناپ(n, chunk_مقسوم_علیہ)), common_قدم);
        إذا (chunk_خانے_گنتی == 0) {
            chunk_خانے_گنتی = common_قدم;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ثابت size_t nchunk_ناپ = std::max<size_t>(1, ceil_div_ناپ(n, chunk_خانے_گنتی));
        GGML_ASSERT(nchunk_ناپ <= (size_t)ودھ_توں_ودھ_عدد);
        ثابت صحيح nchunk = (صحيح)nchunk_ناپ;
        ثابت size_t dst_stride = dst->nb[1];

        خودکار لگاتار_chunk = [&](runtime_تھاں & تھاں, size_t global_شروع, size_t خانے_گنتی_2, uint8_t * dst_batch_ادھار) {
            ثابت size_t rhs_packed_وتھ = تھاں.kernel->get_rhs_packed_وتھ_ex(global_شروع, k, تھاں.rhs_bl);
            ثابت size_t dst_وتھ        = تھاں.kernel->get_dst_وتھ(0, global_شروع, dst_stride);

            ثابت uint8_t * lhs_ptr = scratch + تھاں.lhs_وتھ + تھاں.lhs_packed_offset0;
            ثابت uint8_t * rhs_ptr = تھاں.rhs_ادھار + rhs_packed_وتھ;
            عائم * dst_ptr         = reinterpret_cast<عائم *>(dst_batch_ادھار + dst_وتھ);

            تھاں.kernel->لگاتار_kernel_ex(m, خانے_گنتی_2, k, تھاں.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       حجم(عائم),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        لكل (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ثابت uint8_t * lhs_batch_ادھار = static_cast<ثابت uint8_t *>(src1->انکڑے) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_ادھار = static_cast<uint8_t *>(dst->انکڑے) + batch_idx * dst->nb[2];

            إذا (local_تھاں != -1) {
                runtime_تھاں & تھاں = runtime[local_تھاں];
                ثابت int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)تھاں.mr);
                int64_t max_دھاگے = تھاں.mr ? (m_roundup_mr / (int64_t)تھاں.mr) : تھاں.assigned_دھاگے;
                max_دھاگے = std::max<int64_t>(1, max_دھاگے);
                ثابت int64_t use_دھاگے = std::min<int64_t>(تھاں.assigned_دھاگے, max_دھاگے);

                إذا (local_ith < use_دھاگے) {
                    ثابت int64_t عدد_m_فی_thread0   = گیڑا_تھلے((size_t)(m_roundup_mr / use_دھاگے), تھاں.mr);
                    ثابت int64_t عدد_m_فی_thread_N_1 = (int64_t)m - (use_دھاگے - 1) * عدد_m_فی_thread0;

                    ثابت int64_t m_شروع = (int64_t)local_ith * عدد_m_فی_thread0;
                    ثابت int64_t m_گنتی = (local_ith == use_دھاگے - 1) ? عدد_m_فی_thread_N_1 : عدد_m_فی_thread0;

                    ثابت size_t ادھار_packed_off  = تھاں.lhs_info->get_packed_وتھ_ex(m_شروع, k, تھاں.lhs_bl, تھاں.mr, تھاں.kr, تھاں.sr);
                    ثابت size_t اگلا_بلاک_off   = تھاں.lhs_info->get_packed_وتھ_ex(m_شروع + تھاں.mr, k, تھاں.lhs_bl, تھاں.mr, تھاں.kr, تھاں.sr);
                    ثابت size_t صف_stride_بائٹاں = تھاں.mr ? (اگلا_بلاک_off - ادھار_packed_off) / تھاں.mr : 0;

                    int64_t بچیا = m_گنتی;
                    int64_t cur       = m_شروع;

                    uint8_t * lhs_packed = scratch + تھاں.lhs_وتھ;
                    طالما (بچیا > 0) {
                        ثابت int64_t صف_وچ_گروہ = cur;
                        ثابت int64_t avail        = (int64_t)m - صف_وچ_گروہ;
                        ثابت int64_t take         = std::min(avail, بچیا);

                        ثابت size_t src_off = تھاں.lhs_info->get_وتھ(صف_وچ_گروہ, src1->nb[1]);
                        ثابت فراغ * src_ptr = lhs_batch_ادھار + src_off;
                        ثابت size_t dst_off = ادھار_packed_off + (size_t)(cur - m_شروع) * صف_stride_بائٹاں;
                        فراغ * dst_ptr       = lhs_packed + dst_off;

                        تھاں.lhs_info->pack_func_ex(take, k, تھاں.lhs_bl, تھاں.mr, تھاں.kr, تھاں.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        بچیا -= take;
                    }
                }
            }

            إذا (ith_کل == 0) {
                ggml_threadpool_chunk_سیٹ(params->threadpool, سرگرم_دھاگے);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            إذا (local_تھاں != -1) {
                runtime_تھاں & تھاں = runtime[local_تھاں];
                صحيح current_chunk = ith_کل;
                طالما (current_chunk < nchunk) {
                    ثابت size_t global_شروع = (size_t)current_chunk * chunk_خانے_گنتی;
                    إذا (global_شروع >= n) {
                        اكسر;
                    }

                    ثابت size_t خانے_گنتی_2 = std::min(chunk_خانے_گنتی, n - global_شروع);
                    إذا (خانے_گنتی_2 > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        لگاتار_chunk(تھاں, global_شروع, خانے_گنتی_2, dst_batch_ادھار);
                    }

                    current_chunk = ggml_threadpool_chunk_جوڑو(params->threadpool, 1);
                }
            }

            إذا (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        أرجع سچ;
    }

عام:
    صحيح repack(بنية ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t انکڑے_ناپ) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ثابت size_t n = tensor->ne[1];
        ثابت size_t k = tensor->ne[0];

        kleidiai_بھار_سرخی * سرخی = kleidiai_بھار_سرخی_from_ptr(tensor->انکڑے);
        إذا (!سرخی) {
            أرجع -1;
        }

        سرخی->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        سرخی->نسخہ_2    = GGML_KLEIDIAI_PACK_نسخہ;
        سرخی->تھاں_گنتی = 0;

        uint8_t * ادھار_ptr = static_cast<uint8_t *>(tensor->انکڑے);
        size_t cursor = حجم(kleidiai_بھار_سرخی);
        cursor = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ثابت منطقي want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ثابت منطقي want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ثابت منطقي want_f32 = tensor->type == GGML_TYPE_F32;
        ثابت صحيح تھاں_کل = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ثابت منطقي allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<عائم>  scales;
        std::vector<عائم>  bias;
        std::vector<عائم> صف_f32;

        إذا (want_q8 && تھاں_کل > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ثابت size_t صف_stride = tensor->nb[1];
            ثابت size_t k_بلاکاں   = (k + QK8_0 - 1) / QK8_0;

            إذا (tensor->type == GGML_TYPE_Q6_K) {
                إذا (k % QK_K != 0 || k % QK8_0 != 0) {
                    أرجع -1;
                }
                صف_f32.resize(k);
            }

            لكل (size_t صف = 0; صف < n; ++صف) {
                ثابت uint8_t * صف_انکڑے = static_cast<ثابت uint8_t *>(انکڑے) + صف * صف_stride;

                إذا (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_صف_q6_K(reinterpret_cast<ثابت بلاک_q6_K *>(صف_انکڑے), صف_f32.انکڑے(), k);
                }

                عائم max_abs = 0.0f;
                إذا (tensor->type == GGML_TYPE_Q6_K) {
                    لكل (size_t خطی_idx = 0; خطی_idx < k; ++خطی_idx) {
                        max_abs = std::max(max_abs, std::fabs(صف_f32[خطی_idx]));
                    }
                } وإلا {
                    ثابت بلاک_q8_0 * صف_بلاکاں = reinterpret_cast<ثابت بلاک_q8_0 *>(صف_انکڑے);
                    لكل (size_t بلاک = 0; بلاک < k_بلاکاں; ++بلاک) {
                        ثابت بلاک_q8_0 & blk = صف_بلاکاں[بلاک];
                        ثابت عائم d = GGML_FP16_منزل_FP32(blk.d);
                        لكل (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t خطی_idx = بلاک * QK8_0 + l;
                            إذا (خطی_idx >= k) {
                                اكسر;
                            }
                            ثابت عائم value = d * static_cast<عائم>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                عائم scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[صف] = scale;
                ثابت عائم inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                إذا (tensor->type == GGML_TYPE_Q6_K) {
                    لكل (size_t خطی_idx = 0; خطی_idx < k; ++خطی_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(صف_f32[خطی_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[صف * k + خطی_idx] = static_cast<int8_t>(q);
                    }
                } وإلا {
                    ثابت بلاک_q8_0 * صف_بلاکاں = reinterpret_cast<ثابت بلاک_q8_0 *>(صف_انکڑے);
                    لكل (size_t بلاک = 0; بلاک < k_بلاکاں; ++بلاک) {
                        ثابت بلاک_q8_0 & blk = صف_بلاکاں[بلاک];
                        ثابت عائم d = GGML_FP16_منزل_FP32(blk.d);
                        لكل (size_t l = 0; l < QK8_0; ++l) {
                            ثابت size_t خطی_idx = بلاک * QK8_0 + l;
                            إذا (خطی_idx >= k) {
                                اكسر;
                            }
                            ثابت عائم value = d * static_cast<عائم>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[صف * k + خطی_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        إذا (want_f32 && تھاں_کل > 0) {
            bias.resize(n, 0.0f);
        }

        لكل (صحيح تھاں = 0; تھاں < تھاں_کل && تھاں < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++تھاں) {
            إذا (!allow_fallback && تھاں > 0) {
                اكسر;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[تھاں];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            إذا (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_ناپ_ex || !kernel) {
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

            ثابت size_t packed_ناپ = rhs_info->packed_ناپ_ex(n, k, nr, kr, بلاک_len);
            ثابت size_t aligned_cursor = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = ادھار_ptr + aligned_cursor;

            إذا (rhs_type == GGML_TYPE_Q4_0) {
                بنية kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_نقطہ = 1;
                params.rhs_zero_نقطہ = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ثابت uint8_t *>(انکڑے), خالی_سوچک, خالی_سوچک,
                                       dst_ptr, 0, &params);
            } وإلا إذا (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       انکڑے, خالی_سوچک, خالی_سوچک, dst_ptr, 0, خالی_سوچک);
            } وإلا إذا (rhs_type == GGML_TYPE_Q8_0) {
                بنية kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_نقطہ = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.انکڑے(), خالی_سوچک, scales.انکڑے(),
                                       dst_ptr, 0, &params);
            } وإلا إذا (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       انکڑے, bias.انکڑے(), خالی_سوچک,
                                       dst_ptr, 0, خالی_سوچک);
            } وإلا {
                استمر;
            }

            سرخی->offsets[سرخی->تھاں_گنتی] = aligned_cursor;
            سرخی->sizes[سرخی->تھاں_گنتی]   = packed_ناپ;
            ++سرخی->تھاں_گنتی;

            cursor = aligned_cursor + packed_ناپ;
        }

        إذا (سرخی->تھاں_گنتی == 0) {
            سرخی->magic   = 0;
            سرخی->نسخہ_2 = 0;
            حافظہ_نقل(tensor->انکڑے, انکڑے, انکڑے_ناپ);
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

ساكن فراغ ggml_backend_cpu_kleidiai_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor,
                                                       ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    GGML_ASSERT(وتھ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    خودکار tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    خودکار OK            = tensor_traits->repack(tensor, انکڑے, size);

    GGML_ASSERT(OK == 0);
    GGML_بن_ورتیا(بفر);
}

ساكن ثابت حرف * ggml_backend_cpu_kleidiai_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    GGML_بن_ورتیا(buft);
    أرجع "CPU_KLEIDIAI";
}

ساكن ggml_backend_بفر_t ggml_backend_cpu_kleidiai_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(ggml_backend_cpu_بفر_type(), size);

    إذا (بفر == خالی_سوچک) {
        أرجع خالی_سوچک;
    }

    بفر->buft              = buft;
    بفر->iface.init_tensor = ggml_backend_cpu_kleidiai_بفر_init_tensor;
    بفر->iface.سیٹ_tensor  = ggml_backend_cpu_kleidiai_بفر_سیٹ_tensor;
    بفر->iface.get_tensor  = خالی_سوچک;
    بفر->iface.cpy_tensor  = خالی_سوچک;
    أرجع بفر;
}

ساكن size_t ggml_backend_cpu_kleidiai_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    GGML_بن_ورتیا(buft);
    أرجع TENSOR_ALIGNMENT;
}

ساكن size_t ggml_backend_cpu_kleidiai_بفر_type_get_alloc_ناپ(ggml_backend_بفر_type_t buft, ثابت بنية ggml_tensor * tensor) {
    GGML_بن_ورتیا(buft);

    إذا (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        أرجع ggml_nbytes(tensor);
    }

    ثابت size_t n = tensor->ne[1];
    ثابت size_t k = tensor->ne[0];

    size_t cursor = حجم(kleidiai_بھار_سرخی);
    cursor = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ثابت منطقي want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ثابت منطقي want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ثابت منطقي want_f32 = tensor->type == GGML_TYPE_F32;
    ثابت صحيح تھاں_کل = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ثابت منطقي allow_fallback = kleidiai_pack_fallback_allowed();

    size_t تھاں_گنتی = 0;
    لكل (صحيح تھاں = 0; تھاں < تھاں_کل; ++تھاں) {
        إذا (!allow_fallback && تھاں > 0) {
            اكسر;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[تھاں];
        إذا (!kernels) {
            استمر;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        إذا (!kernel || !rhs_info || !rhs_info->packed_ناپ_ex) {
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

        cursor = align_اتے(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_ناپ_ex(n, k, kernel->get_nr(), kernel->get_kr(), بلاک_len);
        ++تھاں_گنتی;
    }

    إذا (تھاں_گنتی == 0) {
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
        ثابت صحيح تھاں_کل = kleidiai_collect_kernel_chain(op, kernel_chain);
        ثابت منطقي src0_is_kleidiai =
            op->src[0]->بفر &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->بفر->buft->context == هذا &&
            تھاں_کل > 0;

        إذا ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            إذا (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == خالی_سوچک) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == خالی_سوچک) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == خالی_سوچک) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == خالی_سوچک) {
                أرجع خطأ;
            }
            إذا (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == خالی_سوچک) {
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
                    أرجع سچ;
                }
                أرجع خطأ;
            }

            إذا (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                أرجع خطأ;
            }

            أرجع سچ;
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
                    إذا (!warned.exchange(سچ)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                إذا (op->src[0]->type != GGML_TYPE_F16) {
                    أرجع خالی_سوچک;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ثابت صحيح تھاں_کل = kleidiai_collect_kernel_chain(op, kernel_chain);
                إذا (تھاں_کل > 0 && op->src[1]->ne[1] > 1) {
                    إذا ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        أرجع خالی_سوچک;
                    }
                    أرجع ggml::cpu::kleidiai::get_tensor_traits(عدم, عدم);
                }
            }
        }
        أرجع خالی_سوچک;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_بفر_type_t ggml_backend_cpu_kleidiai_بفر_type(فراغ) {
    ساكن ggml::cpu::kleidiai::extra_بفر_type ctx;
    ساكن بنية ggml_backend_بفر_type ggml_backend_cpu_بفر_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_بفر_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_بفر_type_alloc_بفر,
                           /* .alloc_buffer_n       = */ خالی_سوچک,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_بفر_type_get_alignment,
                           /* .get_max_size         = */ خالی_سوچک,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_بفر_type_get_alloc_ناپ,
                           /* .get_alloc_size_n     = */ عدم,
                           /* .is_host              = */ خالی_سوچک,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    أرجع &ggml_backend_cpu_بفر_type_kleidiai;
}
