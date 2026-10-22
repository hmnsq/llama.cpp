// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#অন্তর্ভাৱ <arm_neon.h>
#অন্তর্ভাৱ <cassert>
#অন্তর্ভাৱ <cstdio>
#অন্তর্ভাৱ <cstdlib>
#অন্তর্ভাৱ <atomic>
#অন্তর্ভাৱ <cfloat>
#অন্তর্ভাৱ <cctype>
#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <cmath>
#অন্তর্ভাৱ <stdexcept>
#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <string.h>
#অন্তর্ভাৱ <string>
#অন্তর্ভাৱ <vector>
#অন্তর্ভাৱ <array>
#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <fstream>
#অন্তর্ভাৱ <map>
#অন্তর্ভাৱ <iostream>
#অন্তর্ভাৱ <climits>
#অন্তর্ভাৱ <charconv>
#অন্তর্ভাৱ <system_error>
#যদি defined(__linux__)
#অন্তর্ভাৱ <asm/hwcap.h>
#অন্তর্ভাৱ <dirent.h>
#অন্তর্ভাৱ <sys/auxv.h>
#অন্তর্ভাৱ <sys/types.h>
#অন্তর্ভাৱ <sys/stat.h>
#অন্তর্ভাৱ <unistd.h>
#নোচেদ্যদি defined(__APPLE__)
#অন্তর্ভাৱ <sys/sysctl.h>
#অন্তর্ভাৱ <sys/types.h>
#endif

#অন্তর্ভাৱ "kleidiai.h"

#অন্তর্ভাৱ "ggml-cpu.h"
#অন্তর্ভাৱ "ggml-cpu-impl.h"
#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml-feats.h"
#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-quants.h"
#অন্তর্ভাৱ "ggml-threading.h"
#অন্তর্ভাৱ "traits.h"

#অন্তর্ভাৱ "kernels.h"

#অন্তর্ভাৱ "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#অন্তর্ভাৱ "ggml-common.h"

স্থির constexpr পূর্ণাঙ্ক      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
স্থির constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
স্থির constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
স্থির constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
স্থির constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

সংরচনা ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    পূর্ণাঙ্ক sme_thread_cap; // <= 0 means "SME disabled/unknown"
    পূর্ণাঙ্ক thread_hint;    // <= 0 means "no hint"
    পূর্ণাঙ্ক chunk_multiplier;
} স্থির ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

স্থির অন্তঃস্থ তার্কিক is_sme_family(cpu_feature f) {
    ঘূৰাই (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

স্থির ধ্রুৱ ৱর্ণ* cpu_feature_to_string(cpu_feature f) {
    যদি (f == CPU_FEATURE_NONE) {
        ঘূৰাই "NONE";
    } অন্যথা যদি ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        ঘূৰাই "SME2";
    } অন্যথা যদি ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        ঘূৰাই "SME";
    } অন্যথা যদি ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        ঘূৰাই "SVE";
    }
    অন্যথা যদি ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        ঘূৰাই "I8MM";
    } অন্যথা যদি ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        ঘূৰাই "DOTPROD";
    }
    অন্যথা {
        ঘূৰাই "UNKNOWN";
    }
}

#যদি defined(__linux__) && defined(__aarch64__)
স্থির তার্কিক parse_cpu_dir_name(ধ্রুৱ ৱর্ণ* name, size_t* cpu) {
    যদি (strncmp(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ ৱর্ণ* first = name + 3;
    ধ্রুৱ ৱর্ণ* last = name + strlen(name);

    size_t value = 0;
    ধ্রুৱ auto [end, ec] = std::from_chars(first, last, value, 10);

    যদি (ec != std::errc{} || end != last) {
        ঘূৰাই মিছা;
    }

    *cpu = value;
    ঘূৰাই সঁচা;
}

স্থির std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    যদি (dir == nullptr) {
        ঘূৰাই cpus;
    }

    যেতিয়ালৈ (dirent * entry = readdir(dir)) {
        size_t cpu = 0;
        যদি (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    ঘূৰাই cpus;
}
#endif

#যদি defined(__APPLE__) && defined(__aarch64__)
স্থির তার্কিক apple_sme_counted_perf_level(std::string name) {
    বাবে (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (ৱর্ণ) std::tolower((unsigned ৱর্ণ) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    ঘূৰাই name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

স্থির শূন্য add_smcus_from_smidr(uint64_t smidr, size_t & অঙ্ক_private, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ধ্রুৱ uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ধ্রুৱ uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ধ্রুৱ size_t shared_গুনতি = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ধ্রুৱ uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ধ্রুৱ uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ধ্রুৱ uint32_t id = (affinity2 << 12) | affinity;

    যদি (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    ৱিকল্প (sh) {
        পক্ষ 2: // private SMCU
            ++অঙ্ক_private;
            ভঙ্গ;
        পক্ষ 3: // shared SMCU
            যদি (shared_counts[id] < shared_গুনতি) {
                shared_counts[id] = shared_গুনতি;
            }
            ভঙ্গ;
        পক্ষ 0:
            যদি (id == 0) {
                ++অঙ্ক_private;
            } অন্যথা যদি (shared_counts[id] < shared_গুনতি) {
                shared_counts[id] = shared_গুনতি;
            }
            ভঙ্গ;
        স্ৱভাৱ:
            ভঙ্গ;
    }
}

স্থির size_t detect_অঙ্ক_smcus() {
    ধ্রুৱ auto runtime_feat = ggml_feats_get_arch64_runtime();
    যদি (!runtime_feat.has_sme) {
        ঘূৰাই 0;
    }

#যদি defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t অঙ্ক_private = 0;
    std::map<uint32_t, size_t> shared_counts;

    ধ্রুৱ std::vector<size_t> cpus = detect_cpu_ids();
    বাবে (ধ্রুৱ size_t cpu : cpus) {
        ধ্রুৱ std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        যদি (!file.is_open()) {
            অনুৱর্ত;
        }

        uint64_t smidr = 0;
        যদি (!(file >> std::hex >> smidr)) {
            অনুৱর্ত;
        }

        add_smcus_from_smidr(smidr, অঙ্ক_private, shared_counts);
    }

    size_t total = অঙ্ক_private;
    বাবে (ধ্রুৱ auto & entry : shared_counts) {
        total += entry.second;
    }
    ঘূৰাই total;

#নোচেদ্যদি defined(__APPLE__) && defined(__aarch64__)
    পূর্ণাঙ্ক perf_levels = 0;
    size_t size = পরিমাণ(perf_levels);
    যদি (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != পরিমাণ(perf_levels) || perf_levels <= 0) {
        ঘূৰাই 0;
    }

    size_t units = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < perf_levels; ++i) {
        ৱর্ণ চাবি[64] = {};
        পূর্ণাঙ্ক physical_cpus = 0;
        পূর্ণাঙ্ক cpus_per_l2 = 0;

        snprintf(চাবি, পরিমাণ(চাবি), "hw.perflevel%d.physicalcpu", i);
        size = পরিমাণ(physical_cpus);
        যদি (sysctlbyname(চাবি, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != পরিমাণ(physical_cpus) || physical_cpus <= 0) {
            অনুৱর্ত;
        }

        snprintf(চাবি, পরিমাণ(চাবি), "hw.perflevel%d.cpusperl2", i);
        size = পরিমাণ(cpus_per_l2);
        যদি (sysctlbyname(চাবি, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != পরিমাণ(cpus_per_l2) || cpus_per_l2 <= 0) {
            অনুৱর্ত;
        }

        snprintf(চাবি, পরিমাণ(চাবি), "hw.perflevel%d.name", i);
        size = 0;
        যদি (sysctlbyname(চাবি, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            অনুৱর্ত;
        }

        std::string name(size, '\0');
        যদি (sysctlbyname(চাবি, &name[0], &size, nullptr, 0) != 0) {
            অনুৱর্ত;
        }
        name.resize(size);
        যেতিয়ালৈ (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        যদি (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    ঘূৰাই units;

#নোচেদ্যদি defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    ঘূৰাই 0;

#অন্যথা
    ঘূৰাই 0;
#endif
}

স্থির পূর্ণাঙ্ক parse_uint_env(ধ্রুৱ ৱর্ণ *s, ধ্রুৱ ৱর্ণ *name, তার্কিক *ok) {
    যদি (!s) { *ok = মিছা; ঘূৰাই 0; }
    ৱর্ণ *end = nullptr;
    long v = strtol(s, &end, 10);
    যদি (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = মিছা;
        ঘূৰাই 0;
    }
    যদি (v < 0 || v > INT_MAX) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = মিছা;
        ঘূৰাই 0;
    }
    *ok = সঁচা;
    ঘূৰাই (পূর্ণাঙ্ক)v;
}

স্থির শূন্য init_kleidiai_context(শূন্য) {
    ggml_critical_section_start();
    স্থির তার্কিক initialized = মিছা;

    যদি (!initialized) {
        initialized = সঁচা;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ধ্রুৱ ৱর্ণ *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ধ্রুৱ ৱর্ণ *env_threads     = getenv("GGML_TOTAL_THREADS");
        ধ্রুৱ ৱর্ণ *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ধ্রুৱ auto runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        যদি (env_threads) {
            তার্কিক ok = মিছা;
            পূর্ণাঙ্ক hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            যদি (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        যদি (env_chunk_mult) {
            তার্কিক ok = মিছা;
            পূর্ণাঙ্ক multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            যদি (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        পূর্ণাঙ্ক sme_cores = 0;
        তার্কিক sme_env_ok = মিছা;
        তার্কিক sme_env_set = (env_sme != nullptr);

        ধ্রুৱ তার্কিক has_supported_sme_family = runtime_feat.has_sme;
        তার্কিক sme_cap_detected = মিছা;

        যদি (has_supported_sme_family) {
            detected_smcus = detect_অঙ্ক_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (পূর্ণাঙ্ক)detected_smcus : 1;

            যদি (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        যদি (sme_env_set) {
            তার্কিক ok = মিছা;
            পূর্ণাঙ্ক v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            যদি (ok) {
                যদি (has_supported_sme_family) {
                    sme_cores = v;
                } অন্যথা {
                    যদি (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } অন্যথা {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        যদি (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            যদি (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        যদি (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (পূর্ণাঙ্ক)ctx.features);
        } অন্যথা {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        যদি (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (পূর্ণাঙ্ক)ctx.features);
        } অন্যথা {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        যদি (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (পূর্ণাঙ্ক) ctx.features);
        } অন্যথা {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        যদি (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (পূর্ণাঙ্ক)ctx.features);
        } অন্যথা {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        ধ্রুৱ তার্কিক has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        যদি (has_selected_sme_family_kernel) {
            যদি (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } অন্যথা যদি (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } অন্যথা {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } অন্যথা {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_end();
}

স্থির অন্তঃস্থ পূর্ণাঙ্ক kleidiai_sme_thread_cap() {
    ঘূৰাই ctx.sme_thread_cap;
}

স্থির অন্তঃস্থ size_t align_up(size_t value, size_t alignment) {
    যদি (alignment == 0) {
        ঘূৰাই value;
    }
    ধ্রুৱ size_t remainder = value % alignment;
    ঘূৰাই remainder == 0 ? value : value + (alignment - remainder);
}

স্থির অন্তঃস্থ size_t gcd_size(size_t a, size_t b) {
    যেতিয়ালৈ (b != 0) {
        ধ্রুৱ size_t t = a % b;
        a = b;
        b = t;
    }
    ঘূৰাই a;
}

স্থির অন্তঃস্থ তার্কিক lcm_size(size_t a, size_t b, size_t & result) {
    যদি (a == 0 || b == 0) {
        result = 0;
        ঘূৰাই মিছা;
    }
    ধ্রুৱ size_t g = gcd_size(a, b);
    ধ্রুৱ size_t q = a / g;
    যদি (q > SIZE_MAX / b) {
        ঘূৰাই মিছা;
    }
    result = q * b;
    ঘূৰাই সঁচা;
}

স্থির অন্তঃস্থ size_t ceil_div_size(size_t a, size_t b) {
    ঘূৰাই b == 0 ? 0 : (a + b - 1) / b;
}

স্থির অন্তঃস্থ size_t kleidiai_chunk_cols(size_t n, পূর্ণাঙ্ক nth_total, তার্কিক disable_chunking, size_t n_step) {
    ধ্রুৱ size_t multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ধ্রুৱ size_t ভাজক = std::max<size_t>(1, (size_t) nth_total * multiplier);
    ধ্রুৱ size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, ভাজক)), n_step);
    ঘূৰাই chunk_cols ? chunk_cols : n_step;
}

সংরচনা kleidiai_block_args {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

স্থির অন্তঃস্থ kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    ৱিকল্প (rhs_type) {
        পক্ষ GGML_TYPE_Q4_0:
            ঘূৰাই { QK4_0, QK4_0, QK4_0 };
        পক্ষ GGML_TYPE_Q4_K:
            ঘূৰাই { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        পক্ষ GGML_TYPE_Q8_0:
            ঘূৰাই { 0, 0, QK8_0 };
        স্ৱভাৱ:
            ঘূৰাই { 0, 0, 0 };
    }
}

স্থির অন্তঃস্থ তার্কিক kleidiai_pack_fallback_allowed() {
    যদি (ctx.sme_thread_cap <= 0) {
        ঘূৰাই মিছা;
    }
    যদি (ctx.thread_hint <= 0) {
        ঘূৰাই সঁচা;
    }
    ঘূৰাই ctx.thread_hint > ctx.sme_thread_cap;
}

সংরচনা kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_গুনতি;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

স্থির অন্তঃস্থ kleidiai_weight_header * kleidiai_weight_header_from_ptr(শূন্য * data) {
    ঘূৰাই reinterpret_cast<kleidiai_weight_header *>(data);
}

স্থির অন্তঃস্থ ধ্রুৱ kleidiai_weight_header * kleidiai_weight_header_from_ptr(ধ্রুৱ শূন্য * data) {
    ঘূৰাই reinterpret_cast<ধ্রুৱ kleidiai_weight_header *>(data);
}

স্থির অন্তঃস্থ তার্কিক kleidiai_is_weight_header_valid(ধ্রুৱ kleidiai_weight_header * header) {
    যদি (!header) {
        ঘূৰাই মিছা;
    }
    যদি (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        ঘূৰাই মিছা;
    }
    যদি (header->slot_গুনতি == 0 || header->slot_গুনতি > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        ঘূৰাই মিছা;
    }
    ঘূৰাই সঁচা;
}

স্থির অন্তঃস্থ uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, পূর্ণাঙ্ক slot) {
    যদি (!kleidiai_is_weight_header_valid(header)) {
        ঘূৰাই nullptr;
    }
    যদি (slot < 0 || slot >= header->slot_গুনতি) {
        ঘূৰাই nullptr;
    }
    ঘূৰাই reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

স্থির অন্তঃস্থ ধ্রুৱ uint8_t * kleidiai_weight_slot_ptr(ধ্রুৱ kleidiai_weight_header * header, পূর্ণাঙ্ক slot) {
    যদি (!kleidiai_is_weight_header_valid(header)) {
        ঘূৰাই nullptr;
    }
    যদি (slot < 0 || slot >= header->slot_গুনতি) {
        ঘূৰাই nullptr;
    }
    ঘূৰাই reinterpret_cast<ধ্রুৱ uint8_t *>(header) + header->offsets[slot];
}

স্থির অন্তঃস্থ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    ঘূৰাই ctx.kernels_q4;
}

স্থির অন্তঃস্থ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    ঘূৰাই ctx.kernels_q4k;
}

স্থির অন্তঃস্থ ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    ঘূৰাই ctx.kernels_q8;
}

স্থির অন্তঃস্থ ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    ঘূৰাই ctx.kernels_f32;
}

template <typename SelectFallback>
স্থির পূর্ণাঙ্ক kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    পূর্ণাঙ্ক count = 0;
    যদি (!primary) {
        ঘূৰাই 0;
    }
    out[count++] = primary;

    যদি (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        ঘূৰাই count;
    }

    যদি (is_sme_family(primary->required_cpu)) {
        ধ্রুৱ cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        যদি (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            যদি (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    ঘূৰাই count;
}

স্থির পূর্ণাঙ্ক kleidiai_collect_kernel_chain(ধ্রুৱ সংরচনা ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    ঘূৰাই kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ঘূৰাই ggml_kleidiai_select_kernels(mask, op); });
}

স্থির পূর্ণাঙ্ক kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    ঘূৰাই kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ঘূৰাই ggml_kleidiai_select_kernels_q4_0(mask); });
}

স্থির পূর্ণাঙ্ক kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    ঘূৰাই kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ঘূৰাই ggml_kleidiai_select_kernels_q4_k(mask); });
}

স্থির পূর্ণাঙ্ক kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    ঘূৰাই kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ঘূৰাই ggml_kleidiai_select_kernels_q8_0(mask); });
}

স্থির পূর্ণাঙ্ক kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    ঘূৰাই kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { ঘূৰাই ggml_kleidiai_select_kernels_f32(mask); });
}

স্থির অন্তঃস্থ int64_t ggml_ne(ধ্রুৱ ggml_tensor * tensor, পূর্ণাঙ্ক dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    ঘূৰাই tensor->ne[dim];
}

নামাকাশ ggml::cpu::kleidiai {

স্থির size_t round_down(size_t x, size_t y) {
    ঘূৰাই y == 0 ? x : x - (x % y);
}

স্থির শূন্য transpose_f32kxn_f16nxk(size_t n, size_t k, ভিন্ন * dst, ধ্রুৱ uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / পরিমাণ(uint16_t);
    size_t dst_stride = n;

    বাবে (size_t k_idx = 0; k_idx < k; ++k_idx) {
        বাবে (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

শ্ৰেণী tensor_traits : সার্ৱজনিক ggml::cpu::tensor_traits {
    তার্কিক work_size(পূর্ণাঙ্ক /* n_threads */, ধ্রুৱ সংরচনা ggml_tensor * op, size_t & size) override {
        যদি (op->op != GGML_OP_MUL_MAT) {
            ঘূৰাই মিছা;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ধ্রুৱ পূর্ণাঙ্ক slot_গুনতি = kleidiai_collect_kernel_chain(op, kernel_chain);
        যদি (slot_গুনতি == 0) {
            ঘূৰাই মিছা;
        }

        ধ্রুৱ তার্কিক is_gemv = op->src[1]->ne[1] == 1;
        ধ্রুৱ size_t k = op->src[0]->ne[0];
        ধ্রুৱ size_t n = op->src[0]->ne[1];
        ধ্রুৱ size_t m = op->src[1]->ne[1];

        যদি (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ধ্রুৱ kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            ধ্রুৱ size_t qk = block_args.pack_bl;

            size_t cursor = 0;
            তার্কিক any_slot = মিছা;

            বাবে (পূর্ণাঙ্ক slot = 0; slot < slot_গুনতি; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                যদি (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    ঘূৰাই মিছা;
                }

                ধ্রুৱ size_t mr = kernel->get_mr();
                ধ্রুৱ size_t kr = kernel->get_kr();
                ধ্রুৱ size_t sr = kernel->get_sr();

                ধ্রুৱ size_t packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = সঁচা;
            }

            যদি (!any_slot) {
                ঘূৰাই মিছা;
            }

            size = cursor;
            ঘূৰাই সঁচা;
        }

        যদি (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            যদি (is_gemv && op->src[1]->nb[0] == (int64_t) পরিমাণ(ভিন্ন) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                ঘূৰাই সঁচা;
            }

            size_t cursor = 0;
            তার্কিক any_slot = মিছা;

            বাবে (পূর্ণাঙ্ক slot = 0; slot < slot_গুনতি; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                যদি (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    ঘূৰাই মিছা;
                }

                ধ্রুৱ size_t mr = kernel->get_mr();
                ধ্রুৱ size_t kr = kernel->get_kr();
                ধ্রুৱ size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
                any_slot = সঁচা;
            }

            যদি (!any_slot) {
                ঘূৰাই মিছা;
            }

            size = cursor;
            ঘূৰাই সঁচা;
        }

        যদি (op->src[0]->type == GGML_TYPE_F16) {
            ধ্রুৱ int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ধ্রুৱ int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ধ্রুৱ int64_t r = lhs_batch_size0 / rhs_batch_size0;

            size_t cursor = 0;
            তার্কিক any_slot = মিছা;

            বাবে (পূর্ণাঙ্ক slot = 0; slot < slot_গুনতি; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                যদি (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    ঘূৰাই মিছা;
                }

                ধ্রুৱ size_t mr = kernel->get_mr();
                ধ্রুৱ size_t kr = kernel->get_kr();
                ধ্রুৱ size_t sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = সঁচা;
            }

            বাবে (পূর্ণাঙ্ক slot = 0; slot < slot_গুনতি; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                যদি (!kernel || !kernels->rhs_info.packed_size_ex) {
                    ঘূৰাই মিছা;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * পরিমাণ(ভিন্ন);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * পরিমাণ(ভিন্ন);

            যদি (!any_slot) {
                ঘূৰাই মিছা;
            }

            size = cursor;
            ঘূৰাই সঁচা;
        }

        ঘূৰাই মিছা;
    }

    তার্কিক compute_forward(সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst) override {
        যদি (dst->op == GGML_OP_MUL_MAT) {
            যদি (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                ঘূৰাই compute_forward_qx(params, dst);
            } অন্যথা যদি (dst->src[0]->type == GGML_TYPE_F32) {
                ঘূৰাই compute_forward_f32(params, dst);
            } অন্যথা যদি (dst->src[0]->type == GGML_TYPE_F16) {
                ঘূৰাই compute_forward_fp16(params, dst);
            }
        }
        ঘূৰাই মিছা;
    }

    তার্কিক compute_forward_f32(ggml_compute_params * params, সংরচনা ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        ধ্রুৱ ggml_tensor * src0 = dst->src[0];
        ধ্রুৱ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_দ্বিমিক_OP_LOCALS

        যদি (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            ঘূৰাই মিছা;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        যদি (!kernels) {
            ঘূৰাই মিছা;
        }

        ধ্রুৱ size_t k = ne00;
        ধ্রুৱ size_t m = ne11;
        ধ্রুৱ size_t n = ne01;
        ধ্রুৱ তার্কিক use_gemv = m == 1 && src1->nb[0] == (int64_t) পরিমাণ(ভিন্ন) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        যদি (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            ঘূৰাই মিছা;
        }

        যদি (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_size_ex || !lhs_info->pack_func_ex)) {
            ঘূৰাই মিছা;
        }

        ধ্রুৱ kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ধ্রুৱ তার্কিক has_header = kleidiai_is_weight_header_valid(header);

        ধ্রুৱ uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<ধ্রুৱ uint8_t *>(src0->data);
        যদি (!rhs_base) {
            ঘূৰাই মিছা;
        }

        ধ্রুৱ পূর্ণাঙ্ক nth = params->nth > 0 ? params->nth : 1;
        ধ্রুৱ পূর্ণাঙ্ক ith = params->ith;

        ধ্রুৱ size_t mr = kernel->get_mr();
        ধ্রুৱ size_t kr = kernel->get_kr();
        ধ্রুৱ size_t sr = kernel->get_sr();

        ধ্রুৱ size_t lhs_packed_size = use_gemv ? 0 : lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        যদি (!use_gemv) {
            GGML_ASSERT(lhs_packed_size <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ধ্রুৱ size_t dst_stride = dst->nb[1];
        ধ্রুৱ size_t n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        ধ্রুৱ তার্কিক disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) INT_MAX);

        ধ্রুৱ int64_t batch_size = ne12 * ne13;
        বাবে (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ধ্রুৱ int64_t i12 = batch_idx % ne12;
            ধ্রুৱ int64_t i13 = batch_idx / ne12;
            ধ্রুৱ uint8_t * lhs_batch_base = static_cast<ধ্রুৱ uint8_t *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];

            যদি (!use_gemv) {
                ধ্রুৱ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                ধ্রুৱ int64_t use_threads = std::min<int64_t>(nth, max_threads);

                যদি (ith < use_threads) {
                    ধ্রুৱ int64_t অঙ্ক_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), mr);
                    ধ্রুৱ int64_t অঙ্ক_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * অঙ্ক_m_per_thread0;

                    ধ্রুৱ int64_t m_start = (int64_t)ith * অঙ্ক_m_per_thread0;
                    ধ্রুৱ int64_t m_গুনতি = (ith == use_threads - 1) ? অঙ্ক_m_per_thread_N_1 : অঙ্ক_m_per_thread0;

                    ধ্রুৱ size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ধ্রুৱ size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ধ্রুৱ size_t row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_গুনতি;
                    int64_t cur       = m_start;

                    যেতিয়ালৈ (remaining > 0) {
                        ধ্রুৱ int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        ধ্রুৱ size_t src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        ধ্রুৱ শূন্য * src_ptr = lhs_batch_base + src_off;
                        ধ্রুৱ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        শূন্য * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            যদি (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ধ্রুৱ size_t chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (size_t) INT_MAX);

            পূর্ণাঙ্ক current_col = ggml_threadpool_chunk_add(params->threadpool, (পূর্ণাঙ্ক) chunk_cols);
            যেতিয়ালৈ ((size_t) current_col < n) {
                ধ্রুৱ size_t n_start = (size_t) current_col;
                ধ্রুৱ size_t n_to_process = std::min(chunk_cols, n - n_start);

                যদি (n_to_process > 0) {
                    ধ্রুৱ size_t lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ধ্রুৱ size_t rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ধ্রুৱ size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    ধ্রুৱ শূন্য * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    ধ্রুৱ শূন্য * rhs_ptr = rhs_base + rhs_packed_offset;
                    ভিন্ন * dst_ptr      = reinterpret_cast<ভিন্ন *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          পরিমাণ(ভিন্ন),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (পূর্ণাঙ্ক) chunk_cols);
            }

            যদি (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ঘূৰাই সঁচা;
    }

    তার্কিক compute_forward_fp16(ggml_compute_params * params, সংরচনা ggml_tensor * dst) {
        ধ্রুৱ ggml_tensor * src0 = dst->src[0];
        ধ্রুৱ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_দ্বিমিক_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        যদি (!kernels) {
            ঘূৰাই মিছা;
        }

        ধ্রুৱ তার্কিক is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        যদি (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            ঘূৰাই মিছা;
        }

        ধ্রুৱ পূর্ণাঙ্ক nth = params->nth;
        ধ্রুৱ পূর্ণাঙ্ক ith = params->ith;

        ধ্রুৱ int64_t lhs_batch_size0 = ne12;
        ধ্রুৱ int64_t rhs_batch_size0 = ne02;
        ধ্রুৱ int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        ধ্রুৱ int64_t r = lhs_batch_size0 / rhs_batch_size0;

        ধ্রুৱ int64_t m_group = ne11;
        ধ্রুৱ int64_t m       = m_group;
        ধ্রুৱ int64_t n       = ne01;
        ধ্রুৱ int64_t k       = ne00;

        ধ্রুৱ size_t lhs_stride = src1->nb[1];
        ধ্রুৱ size_t rhs_stride = src0->nb[1];
        ধ্রুৱ size_t dst_stride = dst->nb[1];

        ধ্রুৱ int64_t mr = (int64_t) kernel->get_mr();
        ধ্রুৱ int64_t nr = (int64_t) kernel->get_nr();
        ধ্রুৱ int64_t kr = (int64_t) kernel->get_kr();
        ধ্রুৱ int64_t sr = (int64_t) kernel->get_sr();

        ধ্রুৱ size_t lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        ধ্রুৱ size_t rhs_packed_size = kernels->rhs_info.packed_size_ex(n, k, nr, kr, 0);
        ধ্রুৱ size_t kxn_size        = k * n * পরিমাণ(ভিন্ন);
        ধ্রুৱ size_t bias_size       = n * পরিমাণ(ভিন্ন);

        ধ্রুৱ size_t wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        বাবে (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ধ্রুৱ int64_t rhs_batch_idx = batch_idx / r;
            ধ্রুৱ uint8_t * rhs_batch_base = static_cast<ধ্রুৱ uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ধ্রুৱ int64_t m_roundup_mr = kai_roundup(m, mr);
                ধ্রুৱ int64_t অঙ্ক_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                যদি (ith < অঙ্ক_threads) {
                    ধ্রুৱ int64_t অঙ্ক_m_per_thread0   = round_down((size_t)(m_roundup_mr / অঙ্ক_threads), (size_t)mr);
                    ধ্রুৱ int64_t অঙ্ক_m_per_thread_N_1 = m - (অঙ্ক_threads - 1) * অঙ্ক_m_per_thread0;

                    ধ্রুৱ int64_t m_start = ith * অঙ্ক_m_per_thread0;
                    ধ্রুৱ int64_t m_গুনতি = (ith == অঙ্ক_threads - 1) ? অঙ্ক_m_per_thread_N_1 : অঙ্ক_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ধ্রুৱ size_t base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ধ্রুৱ size_t next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ধ্রুৱ size_t row_stride_bytes = (next_block_off - base_packed_off) / (size_t)mr;

                    int64_t remaining = m_গুনতি;
                    int64_t cur       = m_start;

                    যেতিয়ালৈ (remaining > 0) {
                        ধ্রুৱ int64_t row_in_group = cur;
                        ধ্রুৱ int64_t avail        = m_group - row_in_group;
                        ধ্রুৱ int64_t take         = std::min(avail, remaining);

                        ধ্রুৱ uint8_t * lhs_batch_base = static_cast<ধ্রুৱ uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        ধ্রুৱ শূন্য * src_ptr = lhs_batch_base + (size_t)row_in_group * lhs_stride;
                        ধ্রুৱ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        শূন্য * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            যদি (ith == 0) {
                memset(bias, 0, (size_t)n * পরিমাণ(ভিন্ন));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<ভিন্ন *>(rhs_kxn),
                                        reinterpret_cast<ধ্রুৱ uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * পরিমাণ(ভিন্ন),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ধ্রুৱ int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t অঙ্ক_threads_n = KAI_MIN(n / n_step, nth);
                যদি (অঙ্ক_threads_n <= 0) {
                    অঙ্ক_threads_n = 1;
                }

                যদি (ith < অঙ্ক_threads_n) {
                    ধ্রুৱ int64_t অঙ্ক_n_per_thread0   = round_down((size_t)(n / অঙ্ক_threads_n), (size_t)n_step);
                    ধ্রুৱ int64_t অঙ্ক_n_per_thread_N_1 = n - (অঙ্ক_threads_n - 1) * অঙ্ক_n_per_thread0;

                    ধ্রুৱ int64_t n_start      = ith * অঙ্ক_n_per_thread0;
                    ধ্রুৱ int64_t n_to_process = (ith == অঙ্ক_threads_n - 1) ? অঙ্ক_n_per_thread_N_1 : অঙ্ক_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ধ্রুৱ size_t lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ধ্রুৱ size_t rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ধ্রুৱ size_t dst_offset         = kernel->get_dst_offset((size_t)0, (size_t)n_start, dst_stride);

                    ধ্রুৱ শূন্য * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ধ্রুৱ শূন্য * rhs_ptr = rhs_packed + rhs_packed_offset;
                    ভিন্ন * dst_ptr      = reinterpret_cast<ভিন্ন *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, পরিমাণ(ভিন্ন), -FLT_MAX, FLT_MAX);
                }
            }

            যদি (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ঘূৰাই সঁচা;
    }

    তার্কিক compute_forward_qx(সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ধ্রুৱ ggml_tensor * src0 = dst->src[0];
        ধ্রুৱ ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_দ্বিমিক_OP_LOCALS

        ধ্রুৱ kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ধ্রুৱ তার্কিক has_header = kleidiai_is_weight_header_valid(header);
        ধ্রুৱ তার্কিক is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ধ্রুৱ পূর্ণাঙ্ক slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        auto weight_for_slot = [&](পূর্ণাঙ্ক slot_index, size_t & size_out) -> ধ্রুৱ uint8_t * {
            যদি (slot_index < 0 || slot_index >= slot_total) {
                ঘূৰাই nullptr;
            }
            যদি (has_header) {
                যদি (slot_index < header->slot_গুনতি) {
                    size_out = static_cast<size_t>(header->sizes[slot_index]);
                    ঘূৰাই kleidiai_weight_slot_ptr(header, slot_index);
                }
                ঘূৰাই nullptr;
            }
            যদি (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                ঘূৰাই static_cast<ধ্রুৱ uint8_t *>(src0->data);
            }
            ঘূৰাই nullptr;
        };

        সংরচনা runtime_slot {
            পূর্ণাঙ্ক slot_index;
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
            পূর্ণাঙ্ক assigned_threads;
            পূর্ণাঙ্ক thread_begin;
            পূর্ণাঙ্ক thread_end;
            ধ্রুৱ uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        পূর্ণাঙ্ক runtime_গুনতি = 0;

        বাবে (পূর্ণাঙ্ক slot = 0; slot < slot_total && runtime_গুনতি < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            যদি (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                অনুৱর্ত;
            }

            size_t rhs_size = 0;
            ধ্রুৱ uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            যদি (!rhs_ptr || rhs_size == 0) {
                অনুৱর্ত;
            }

            ধ্রুৱ kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

            runtime[runtime_গুনতি] = {
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
            ++runtime_গুনতি;
        }

        যদি (runtime_গুনতি == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            ঘূৰাই মিছা;
        }

        ধ্রুৱ পূর্ণাঙ্ক nth_total = params->nth > 0 ? params->nth : 1;
        ধ্রুৱ পূর্ণাঙ্ক ith_total = params->ith;

        পূর্ণাঙ্ক sme_slot = -1;
        পূর্ণাঙ্ক non_sme_slot = -1;
        বাবে (পূর্ণাঙ্ক i = 0; i < runtime_গুনতি; ++i) {
            যদি (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                ভঙ্গ;
            }
        }

        বাবে (পূর্ণাঙ্ক i = 0; i < runtime_গুনতি; ++i) {
            যদি (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                ভঙ্গ;
            }
        }

        ধ্রুৱ পূর্ণাঙ্ক sme_cap_limit = ctx.sme_thread_cap;
        ধ্রুৱ তার্কিক use_hybrid = sme_cap_limit > 0 &&
                                 runtime_গুনতি > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_cols_per_thread = 0;
        যদি (runtime_গুনতি > 0 && nth_total > 0) {
            min_cols_per_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        ধ্রুৱ তার্কিক too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        ধ্রুৱ তার্কিক hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        যদি (!hybrid_enabled) {
            পূর্ণাঙ্ক chosen_slot = 0;
            যদি (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } অন্যথা যদি (runtime_গুনতি > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            যদি (chosen_slot != 0 && chosen_slot < runtime_গুনতি) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_গুনতি = runtime_গুনতি > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            যদি (runtime_গুনতি > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        ধ্রুৱ তার্কিক sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        পূর্ণাঙ্ক sme_cap = sme_single_only ? nth_total : kleidiai_sme_thread_cap();
        যদি (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        পূর্ণাঙ্ক threads_remaining = nth_total;
        যদি (sme_slot != -1) {
            পূর্ণাঙ্ক sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        পূর্ণাঙ্ক fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        পূর্ণাঙ্ক fallback_গুনতি = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        বাবে (পূর্ণাঙ্ক i = 0; i < runtime_গুনতি; ++i) {
            যদি (i == sme_slot) {
                অনুৱর্ত;
            }
            fallback_indices[fallback_গুনতি++] = i;
        }

        বাবে (পূর্ণাঙ্ক fi = 0; fi < fallback_গুনতি; ++fi) {
            যদি (threads_remaining <= 0) {
                ভঙ্গ;
            }
            ধ্রুৱ পূর্ণাঙ্ক slot_index = fallback_indices[fi];
            ধ্রুৱ পূর্ণাঙ্ক slots_বাম = fallback_গুনতি - fi;
            পূর্ণাঙ্ক share = (threads_remaining + slots_বাম - 1) / slots_বাম;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        পূর্ণাঙ্ক thread_cursor = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < runtime_গুনতি; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ধ্রুৱ পূর্ণাঙ্ক active_threads = thread_cursor;

        পূর্ণাঙ্ক local_slot = -1;
        পূর্ণাঙ্ক local_ith  = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < runtime_গুনতি; ++i) {
            যদি (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                ভঙ্গ;
            }
        }
        ধ্রুৱ size_t k = ne00;
        ধ্রুৱ size_t m = ne11;
        ধ্রুৱ size_t n = ne01;

        size_t cursor = 0;
        বাবে (পূর্ণাঙ্ক i = 0; i < runtime_গুনতি; ++i) {
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_step = 1;
        বাবে (পূর্ণাঙ্ক i = 0; i < runtime_গুনতি; ++i) {
            যদি (runtime[i].assigned_threads == 0) {
                অনুৱর্ত;
            }
            size_t next_step = 0;
            যদি (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                ঘূৰাই মিছা;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        ধ্রুৱ তার্কিক disable_chunking = ggml_is_numa();
        ধ্রুৱ size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ধ্রুৱ size_t chunk_ভাজক = (active_threads == 1 || disable_chunking) ? (size_t) active_threads : (size_t) active_threads * chunk_multiplier;
        size_t chunk_cols = align_up(std::max<size_t>(1, ceil_div_size(n, chunk_ভাজক)), common_step);
        যদি (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ধ্রুৱ size_t nchunk_size = std::max<size_t>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (size_t)INT_MAX);
        ধ্রুৱ পূর্ণাঙ্ক nchunk = (পূর্ণাঙ্ক)nchunk_size;
        ধ্রুৱ size_t dst_stride = dst->nb[1];

        auto run_chunk = [&](runtime_slot & slot, size_t global_start, size_t cols, uint8_t * dst_batch_base) {
            ধ্রুৱ size_t rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            ধ্রুৱ size_t dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            ধ্রুৱ uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            ধ্রুৱ uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            ভিন্ন * dst_ptr         = reinterpret_cast<ভিন্ন *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       পরিমাণ(ভিন্ন),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        বাবে (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ধ্রুৱ uint8_t * lhs_batch_base = static_cast<ধ্রুৱ uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            যদি (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                ধ্রুৱ int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                ধ্রুৱ int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                যদি (local_ith < use_threads) {
                    ধ্রুৱ int64_t অঙ্ক_m_per_thread0   = round_down((size_t)(m_roundup_mr / use_threads), slot.mr);
                    ধ্রুৱ int64_t অঙ্ক_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * অঙ্ক_m_per_thread0;

                    ধ্রুৱ int64_t m_start = (int64_t)local_ith * অঙ্ক_m_per_thread0;
                    ধ্রুৱ int64_t m_গুনতি = (local_ith == use_threads - 1) ? অঙ্ক_m_per_thread_N_1 : অঙ্ক_m_per_thread0;

                    ধ্রুৱ size_t base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ধ্রুৱ size_t next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ধ্রুৱ size_t row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_গুনতি;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    যেতিয়ালৈ (remaining > 0) {
                        ধ্রুৱ int64_t row_in_group = cur;
                        ধ্রুৱ int64_t avail        = (int64_t)m - row_in_group;
                        ধ্রুৱ int64_t take         = std::min(avail, remaining);

                        ধ্রুৱ size_t src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        ধ্রুৱ শূন্য * src_ptr = lhs_batch_base + src_off;
                        ধ্রুৱ size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        শূন্য * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            যদি (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            যদি (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                পূর্ণাঙ্ক current_chunk = ith_total;
                যেতিয়ালৈ (current_chunk < nchunk) {
                    ধ্রুৱ size_t global_start = (size_t)current_chunk * chunk_cols;
                    যদি (global_start >= n) {
                        ভঙ্গ;
                    }

                    ধ্রুৱ size_t cols = std::min(chunk_cols, n - global_start);
                    যদি (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            যদি (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        ঘূৰাই সঁচা;
    }

সার্ৱজনিক:
    পূর্ণাঙ্ক repack(সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ধ্রুৱ size_t n = tensor->ne[1];
        ধ্রুৱ size_t k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        যদি (!header) {
            ঘূৰাই -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_গুনতি = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        size_t cursor = পরিমাণ(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ধ্রুৱ তার্কিক want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ধ্রুৱ তার্কিক want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ধ্রুৱ তার্কিক want_f32 = tensor->type == GGML_TYPE_F32;
        ধ্রুৱ পূর্ণাঙ্ক slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ধ্রুৱ তার্কিক allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<ভিন্ন>  scales;
        std::vector<ভিন্ন>  bias;
        std::vector<ভিন্ন> row_f32;

        যদি (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ধ্রুৱ size_t row_stride = tensor->nb[1];
            ধ্রুৱ size_t k_blocks   = (k + QK8_0 - 1) / QK8_0;

            যদি (tensor->type == GGML_TYPE_Q6_K) {
                যদি (k % QK_K != 0 || k % QK8_0 != 0) {
                    ঘূৰাই -1;
                }
                row_f32.resize(k);
            }

            বাবে (size_t row = 0; row < n; ++row) {
                ধ্রুৱ uint8_t * row_data = static_cast<ধ্রুৱ uint8_t *>(data) + row * row_stride;

                যদি (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<ধ্রুৱ block_q6_K *>(row_data), row_f32.data(), k);
                }

                ভিন্ন max_abs = 0.0f;
                যদি (tensor->type == GGML_TYPE_Q6_K) {
                    বাবে (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::fabs(row_f32[linear_idx]));
                    }
                } অন্যথা {
                    ধ্রুৱ block_q8_0 * row_blocks = reinterpret_cast<ধ্রুৱ block_q8_0 *>(row_data);
                    বাবে (size_t block = 0; block < k_blocks; ++block) {
                        ধ্রুৱ block_q8_0 & blk = row_blocks[block];
                        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(blk.d);
                        বাবে (size_t l = 0; l < QK8_0; ++l) {
                            ধ্রুৱ size_t linear_idx = block * QK8_0 + l;
                            যদি (linear_idx >= k) {
                                ভঙ্গ;
                            }
                            ধ্রুৱ ভিন্ন value = d * static_cast<ভিন্ন>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                ভিন্ন scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                ধ্রুৱ ভিন্ন inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                যদি (tensor->type == GGML_TYPE_Q6_K) {
                    বাবে (size_t linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } অন্যথা {
                    ধ্রুৱ block_q8_0 * row_blocks = reinterpret_cast<ধ্রুৱ block_q8_0 *>(row_data);
                    বাবে (size_t block = 0; block < k_blocks; ++block) {
                        ধ্রুৱ block_q8_0 & blk = row_blocks[block];
                        ধ্রুৱ ভিন্ন d = GGML_FP16_TO_FP32(blk.d);
                        বাবে (size_t l = 0; l < QK8_0; ++l) {
                            ধ্রুৱ size_t linear_idx = block * QK8_0 + l;
                            যদি (linear_idx >= k) {
                                ভঙ্গ;
                            }
                            ধ্রুৱ ভিন্ন value = d * static_cast<ভিন্ন>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        যদি (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
        }

        বাবে (পূর্ণাঙ্ক slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            যদি (!allow_fallback && slot > 0) {
                ভঙ্গ;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            যদি (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                অনুৱর্ত;
            }

            ধ্রুৱ size_t nr = kernel->get_nr();
            ধ্রুৱ size_t kr = kernel->get_kr();
            ধ্রুৱ size_t sr = kernel->get_sr();
            ধ্রুৱ ggml_type rhs_type = kernels->rhs_type;
            যদি (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            ধ্রুৱ size_t block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            যদি (block_len == SIZE_MAX) {
                অনুৱর্ত;
            }

            ধ্রুৱ size_t packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            ধ্রুৱ size_t aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            যদি (rhs_type == GGML_TYPE_Q4_0) {
                সংরচনা kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ধ্রুৱ uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } অন্যথা যদি (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       data, nullptr, nullptr, dst_ptr, 0, nullptr);
            } অন্যথা যদি (rhs_type == GGML_TYPE_Q8_0) {
                সংরচনা kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } অন্যথা যদি (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
            } অন্যথা {
                অনুৱর্ত;
            }

            header->offsets[header->slot_গুনতি] = aligned_cursor;
            header->sizes[header->slot_গুনতি]   = packed_size;
            ++header->slot_গুনতি;

            cursor = aligned_cursor + packed_size;
        }

        যদি (header->slot_গুনতি == 0) {
            header->magic   = 0;
            header->version = 0;
            memcpy(tensor->data, data, data_size);
        }

        ঘূৰাই 0;
    }
};

স্থির ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, সংরচনা ggml_tensor *) {
    স্থির tensor_traits traits;
    ঘূৰাই &traits;
}
}  // namespace ggml::cpu::kleidiai

স্থির গণনা ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, সংরচনা ggml_tensor * tensor) {
    tensor->extra = (শূন্য *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    ঘূৰাই GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, সংরচনা ggml_tensor * tensor,
                                                       ধ্রুৱ শূন্য * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    ঘূৰাই "CPU_KLEIDIAI";
}

স্থির ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    যদি (buffer == nullptr) {
        ঘূৰাই nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    ঘূৰাই buffer;
}

স্থির size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    ঘূৰাই TENSOR_ALIGNMENT;
}

স্থির size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ধ্রুৱ সংরচনা ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    যদি (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        ঘূৰাই ggml_nbytes(tensor);
    }

    ধ্রুৱ size_t n = tensor->ne[1];
    ধ্রুৱ size_t k = tensor->ne[0];

    size_t cursor = পরিমাণ(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ধ্রুৱ তার্কিক want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ধ্রুৱ তার্কিক want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ধ্রুৱ তার্কিক want_f32 = tensor->type == GGML_TYPE_F32;
    ধ্রুৱ পূর্ণাঙ্ক slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ধ্রুৱ তার্কিক allow_fallback = kleidiai_pack_fallback_allowed();

    size_t slot_গুনতি = 0;
    বাবে (পূর্ণাঙ্ক slot = 0; slot < slot_total; ++slot) {
        যদি (!allow_fallback && slot > 0) {
            ভঙ্গ;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        যদি (!kernels) {
            অনুৱর্ত;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        যদি (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            অনুৱর্ত;
        }

        ধ্রুৱ ggml_type rhs_type = kernels->rhs_type;
        ধ্রুৱ size_t block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        যদি (block_len == SIZE_MAX) {
            অনুৱর্ত;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_গুনতি;
    }

    যদি (slot_গুনতি == 0) {
        ঘূৰাই ggml_nbytes(tensor);
    }

    ঘূৰাই std::max(cursor, ggml_nbytes(tensor));
}

নামাকাশ ggml::cpu::kleidiai {
শ্ৰেণী extra_buffer_type : ggml::cpu::extra_buffer_type {
    তার্কিক supports_op(ggml_backend_dev_t, ধ্রুৱ সংরচনা ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        যদি (op->op != GGML_OP_MUL_MAT) {
            ঘূৰাই মিছা;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ধ্রুৱ পূর্ণাঙ্ক slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        ধ্রুৱ তার্কিক src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft->context == এতত্ &&
            slot_total > 0;

        যদি ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            যদি (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                ঘূৰাই মিছা;
            }
            যদি (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                ঘূৰাই মিছা;
            }
            যদি (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                ঘূৰাই মিছা;
            }
            যদি (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                ঘূৰাই মিছা;
            }
            যদি (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                ঘূৰাই মিছা;
            }
            যদি (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                ঘূৰাই মিছা;
            }

            যদি (op->src[0]->type == GGML_TYPE_Q4_K) {
                যদি (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    ঘূৰাই মিছা;
                }
                ঘূৰাই ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == পরিমাণ(ভিন্ন) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * পরিমাণ(ভিন্ন));
            }

            যদি (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                যদি (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    ঘূৰাই সঁচা;
                }
                ঘূৰাই মিছা;
            }

            যদি (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                ঘূৰাই মিছা;
            }

            ঘূৰাই সঁচা;
        }

        ঘূৰাই মিছা;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ধ্রুৱ সংরচনা ggml_tensor * op) override {
        যদি (op->op == GGML_OP_MUL_MAT) {
            যদি (op->src[0]->buffer && op->src[0]->buffer->buft->context == এতত্) {
                ঘূৰাই (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } অন্যথা {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                যদি (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    স্থির std::atomic<তার্কিক> warned(মিছা);
                    যদি (!warned.exchange(সঁচা)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                যদি (op->src[0]->type != GGML_TYPE_F16) {
                    ঘূৰাই nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ধ্রুৱ পূর্ণাঙ্ক slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                যদি (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    যদি ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        ঘূৰাই nullptr;
                    }
                    ঘূৰাই ggml::cpu::kleidiai::get_tensor_traits(রিক্ত, রিক্ত);
                }
            }
        }
        ঘূৰাই nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(শূন্য) {
    স্থির ggml::cpu::kleidiai::extra_buffer_type ctx;
    স্থির সংরচনা ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .get_alloc_size_n     = */ রিক্ত,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    ঘূৰাই &ggml_backend_cpu_buffer_type_kleidiai;
}
