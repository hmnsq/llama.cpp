// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#अन्तर्भाव <arm_neon.h>
#अन्तर्भाव <cassert>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <atomic>
#अन्तर्भाव <cfloat>
#अन्तर्भाव <cctype>
#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>
#अन्तर्भाव <stdexcept>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <string>
#अन्तर्भाव <vector>
#अन्तर्भाव <array>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <fstream>
#अन्तर्भाव <map>
#अन्तर्भाव <iostream>
#अन्तर्भाव <climits>
#अन्तर्भाव <charconv>
#अन्तर्भाव <system_error>
#यदि defined(__linux__)
#अन्तर्भाव <asm/hwcap.h>
#अन्तर्भाव <dirent.h>
#अन्तर्भाव <sys/auxv.h>
#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <sys/stat.h>
#अन्तर्भाव <unistd.h>
#नोचेद्यदि defined(__APPLE__)
#अन्तर्भाव <sys/sysctl.h>
#अन्तर्भाव <sys/types.h>
#endif

#अन्तर्भाव "kleidiai.h"

#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "ggml-cpu-impl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-feats.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-quants.h"
#अन्तर्भाव "ggml-threading.h"
#अन्तर्भाव "traits.h"

#अन्तर्भाव "kernels.h"

#अन्तर्भाव "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#अन्तर्भाव "ggml-common.h"

स्थिर constexpr पूर्णाङ्क      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
स्थिर constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
स्थिर constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
स्थिर constexpr माप_प्रकार   GGML_KLEIDIAI_PACK_ALIGN       = 64;
स्थिर constexpr माप_प्रकार   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

संरचना ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    पूर्णाङ्क sme_thread_cap; // <= 0 means "SME disabled/unknown"
    पूर्णाङ्क thread_hint;    // <= 0 means "no hint"
    पूर्णाङ्क chunk_multiplier;
} स्थिर ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

स्थिर अन्तःस्थ तार्किक is_sme_family(cpu_feature f) {
    प्रत्यागम (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

स्थिर ध्रुव वर्ण* cpu_feature_to_string(cpu_feature f) {
    यदि (f == CPU_FEATURE_NONE) {
        प्रत्यागम "NONE";
    } अन्यथा यदि ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        प्रत्यागम "SME2";
    } अन्यथा यदि ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        प्रत्यागम "SME";
    } अन्यथा यदि ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        प्रत्यागम "SVE";
    }
    अन्यथा यदि ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        प्रत्यागम "I8MM";
    } अन्यथा यदि ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        प्रत्यागम "DOTPROD";
    }
    अन्यथा {
        प्रत्यागम "UNKNOWN";
    }
}

#यदि defined(__linux__) && defined(__aarch64__)
स्थिर तार्किक parse_cpu_dir_name(ध्रुव वर्ण* name, माप_प्रकार* cpu) {
    यदि (म_भेदन(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        प्रत्यागम असत्य;
    }

    ध्रुव वर्ण* first = name + 3;
    ध्रुव वर्ण* last = name + म_माप(name);

    माप_प्रकार value = 0;
    ध्रुव स्वतः [end, ec] = std::from_chars(first, last, value, 10);

    यदि (ec != std::errc{} || end != last) {
        प्रत्यागम असत्य;
    }

    *cpu = value;
    प्रत्यागम सत्य;
}

स्थिर std::vector<माप_प्रकार> detect_cpu_ids() {
    std::vector<माप_प्रकार> cpus;

    सूची * dir = सूची_खोलो("/sys/devices/system/cpu");
    यदि (dir == nullptr) {
        प्रत्यागम cpus;
    }

    यावत् (dirent * entry = सूची_पढ़ो(dir)) {
        माप_प्रकार cpu = 0;
        यदि (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    बंद_सूची(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    प्रत्यागम cpus;
}
#endif

#यदि defined(__APPLE__) && defined(__aarch64__)
स्थिर तार्किक apple_sme_counted_perf_level(std::string name) {
    प्रति (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (वर्ण) std::छोटे((अचिन्हित वर्ण) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    प्रत्यागम name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

स्थिर शून्य add_smcus_from_smidr(uint64_t smidr, माप_प्रकार & अंक_private, std::map<uint32_t, माप_प्रकार> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ध्रुव uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ध्रुव uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ध्रुव माप_प्रकार shared_गिनती = nsmc == 0xF ? 1 : (माप_प्रकार)nsmc + 1;
    ध्रुव uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ध्रुव uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ध्रुव uint32_t id = (affinity2 << 12) | affinity;

    यदि (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    विकल्प (sh) {
        पक्ष 2: // private SMCU
            ++अंक_private;
            भङ्ग;
        पक्ष 3: // shared SMCU
            यदि (shared_counts[id] < shared_गिनती) {
                shared_counts[id] = shared_गिनती;
            }
            भङ्ग;
        पक्ष 0:
            यदि (id == 0) {
                ++अंक_private;
            } अन्यथा यदि (shared_counts[id] < shared_गिनती) {
                shared_counts[id] = shared_गिनती;
            }
            भङ्ग;
        स्वभाव:
            भङ्ग;
    }
}

स्थिर माप_प्रकार detect_अंक_smcus() {
    ध्रुव स्वतः runtime_feat = ggml_feats_get_arch64_runtime();
    यदि (!runtime_feat.has_sme) {
        प्रत्यागम 0;
    }

#यदि defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    माप_प्रकार अंक_private = 0;
    std::map<uint32_t, माप_प्रकार> shared_counts;

    ध्रुव std::vector<माप_प्रकार> cpus = detect_cpu_ids();
    प्रति (ध्रुव माप_प्रकार cpu : cpus) {
        ध्रुव std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        यदि (!file.is_open()) {
            अनुवर्त;
        }

        uint64_t smidr = 0;
        यदि (!(file >> std::hex >> smidr)) {
            अनुवर्त;
        }

        add_smcus_from_smidr(smidr, अंक_private, shared_counts);
    }

    माप_प्रकार total = अंक_private;
    प्रति (ध्रुव स्वतः & entry : shared_counts) {
        total += entry.second;
    }
    प्रत्यागम total;

#नोचेद्यदि defined(__APPLE__) && defined(__aarch64__)
    पूर्णाङ्क perf_levels = 0;
    माप_प्रकार size = परिमाण(perf_levels);
    यदि (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != परिमाण(perf_levels) || perf_levels <= 0) {
        प्रत्यागम 0;
    }

    माप_प्रकार units = 0;
    प्रति (पूर्णाङ्क i = 0; i < perf_levels; ++i) {
        वर्ण कुंजी[64] = {};
        पूर्णाङ्क physical_cpus = 0;
        पूर्णाङ्क cpus_per_l2 = 0;

        snprintf(कुंजी, परिमाण(कुंजी), "hw.perflevel%d.physicalcpu", i);
        size = परिमाण(physical_cpus);
        यदि (sysctlbyname(कुंजी, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != परिमाण(physical_cpus) || physical_cpus <= 0) {
            अनुवर्त;
        }

        snprintf(कुंजी, परिमाण(कुंजी), "hw.perflevel%d.cpusperl2", i);
        size = परिमाण(cpus_per_l2);
        यदि (sysctlbyname(कुंजी, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != परिमाण(cpus_per_l2) || cpus_per_l2 <= 0) {
            अनुवर्त;
        }

        snprintf(कुंजी, परिमाण(कुंजी), "hw.perflevel%d.name", i);
        size = 0;
        यदि (sysctlbyname(कुंजी, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            अनुवर्त;
        }

        std::string name(size, '\0');
        यदि (sysctlbyname(कुंजी, &name[0], &size, nullptr, 0) != 0) {
            अनुवर्त;
        }
        name.resize(size);
        यावत् (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        यदि (apple_sme_counted_perf_level(name)) {
            units += (माप_प्रकार) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    प्रत्यागम units;

#नोचेद्यदि defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    प्रत्यागम 0;

#अन्यथा
    प्रत्यागम 0;
#endif
}

स्थिर पूर्णाङ्क parse_uint_env(ध्रुव वर्ण *s, ध्रुव वर्ण *name, तार्किक *ok) {
    यदि (!s) { *ok = असत्य; प्रत्यागम 0; }
    वर्ण *end = nullptr;
    दीर्घ v = म_से_दीर्घ(s, &end, 10);
    यदि (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = असत्य;
        प्रत्यागम 0;
    }
    यदि (v < 0 || v > पूर्णांक_उच्च) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = असत्य;
        प्रत्यागम 0;
    }
    *ok = सत्य;
    प्रत्यागम (पूर्णाङ्क)v;
}

स्थिर शून्य init_kleidiai_context(शून्य) {
    ggml_critical_section_start();
    स्थिर तार्किक initialized = असत्य;

    यदि (!initialized) {
        initialized = सत्य;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        ध्रुव वर्ण *env_sme         = दो_पर्या("GGML_KLEIDIAI_SME");
        ध्रुव वर्ण *env_threads     = दो_पर्या("GGML_TOTAL_THREADS");
        ध्रुव वर्ण *env_chunk_mult  = दो_पर्या("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ध्रुव स्वतः runtime_feat = ggml_feats_get_arch64_runtime();

        माप_प्रकार detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        यदि (env_threads) {
            तार्किक ok = असत्य;
            पूर्णाङ्क hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            यदि (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        यदि (env_chunk_mult) {
            तार्किक ok = असत्य;
            पूर्णाङ्क multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            यदि (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        पूर्णाङ्क sme_cores = 0;
        तार्किक sme_env_ok = असत्य;
        तार्किक sme_env_set = (env_sme != nullptr);

        ध्रुव तार्किक has_supported_sme_family = runtime_feat.has_sme;
        तार्किक sme_cap_detected = असत्य;

        यदि (has_supported_sme_family) {
            detected_smcus = detect_अंक_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (पूर्णाङ्क)detected_smcus : 1;

            यदि (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        यदि (sme_env_set) {
            तार्किक ok = असत्य;
            पूर्णाङ्क v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            यदि (ok) {
                यदि (has_supported_sme_family) {
                    sme_cores = v;
                } अन्यथा {
                    यदि (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } अन्यथा {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        यदि (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            यदि (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        यदि (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (पूर्णाङ्क)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        यदि (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (पूर्णाङ्क)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        यदि (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (पूर्णाङ्क) ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        यदि (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (पूर्णाङ्क)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        ध्रुव तार्किक has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        यदि (has_selected_sme_family_kernel) {
            यदि (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } अन्यथा यदि (sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime-detected SME thread cap=%d)\n", sme_cores);
            } अन्यथा {
                GGML_LOG_INFO("kleidiai: SME enabled (runtime SME detected, conservative thread cap=%d)\n", sme_cores);
            }
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: SME disabled\n");
        }
    }

    ggml_critical_section_end();
}

स्थिर अन्तःस्थ पूर्णाङ्क kleidiai_sme_thread_cap() {
    प्रत्यागम ctx.sme_thread_cap;
}

स्थिर अन्तःस्थ माप_प्रकार align_up(माप_प्रकार value, माप_प्रकार alignment) {
    यदि (alignment == 0) {
        प्रत्यागम value;
    }
    ध्रुव माप_प्रकार remainder = value % alignment;
    प्रत्यागम remainder == 0 ? value : value + (alignment - remainder);
}

स्थिर अन्तःस्थ माप_प्रकार gcd_size(माप_प्रकार a, माप_प्रकार b) {
    यावत् (b != 0) {
        ध्रुव माप_प्रकार t = a % b;
        a = b;
        b = t;
    }
    प्रत्यागम a;
}

स्थिर अन्तःस्थ तार्किक lcm_size(माप_प्रकार a, माप_प्रकार b, माप_प्रकार & result) {
    यदि (a == 0 || b == 0) {
        result = 0;
        प्रत्यागम असत्य;
    }
    ध्रुव माप_प्रकार g = gcd_size(a, b);
    ध्रुव माप_प्रकार q = a / g;
    यदि (q > SIZE_MAX / b) {
        प्रत्यागम असत्य;
    }
    result = q * b;
    प्रत्यागम सत्य;
}

स्थिर अन्तःस्थ माप_प्रकार ceil_div_size(माप_प्रकार a, माप_प्रकार b) {
    प्रत्यागम b == 0 ? 0 : (a + b - 1) / b;
}

स्थिर अन्तःस्थ माप_प्रकार kleidiai_chunk_cols(माप_प्रकार n, पूर्णाङ्क nth_total, तार्किक disable_chunking, माप_प्रकार n_step) {
    ध्रुव माप_प्रकार multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<माप_प्रकार>(1, (माप_प्रकार) ctx.chunk_multiplier);
    ध्रुव माप_प्रकार भाजक = std::max<माप_प्रकार>(1, (माप_प्रकार) nth_total * multiplier);
    ध्रुव माप_प्रकार chunk_cols = align_up(std::max<माप_प्रकार>(1, ceil_div_size(n, भाजक)), n_step);
    प्रत्यागम chunk_cols ? chunk_cols : n_step;
}

संरचना kleidiai_block_args {
    माप_प्रकार lhs_bl;
    माप_प्रकार rhs_bl;
    माप_प्रकार pack_bl;
};

स्थिर अन्तःस्थ kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    विकल्प (rhs_type) {
        पक्ष GGML_TYPE_Q4_0:
            प्रत्यागम { QK4_0, QK4_0, QK4_0 };
        पक्ष GGML_TYPE_Q4_K:
            प्रत्यागम { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        पक्ष GGML_TYPE_Q8_0:
            प्रत्यागम { 0, 0, QK8_0 };
        स्वभाव:
            प्रत्यागम { 0, 0, 0 };
    }
}

स्थिर अन्तःस्थ तार्किक kleidiai_pack_fallback_allowed() {
    यदि (ctx.sme_thread_cap <= 0) {
        प्रत्यागम असत्य;
    }
    यदि (ctx.thread_hint <= 0) {
        प्रत्यागम सत्य;
    }
    प्रत्यागम ctx.thread_hint > ctx.sme_thread_cap;
}

संरचना kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_गिनती;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

स्थिर अन्तःस्थ kleidiai_weight_header * kleidiai_weight_header_from_ptr(शून्य * data) {
    प्रत्यागम reinterpret_cast<kleidiai_weight_header *>(data);
}

स्थिर अन्तःस्थ ध्रुव kleidiai_weight_header * kleidiai_weight_header_from_ptr(ध्रुव शून्य * data) {
    प्रत्यागम reinterpret_cast<ध्रुव kleidiai_weight_header *>(data);
}

स्थिर अन्तःस्थ तार्किक kleidiai_is_weight_header_valid(ध्रुव kleidiai_weight_header * header) {
    यदि (!header) {
        प्रत्यागम असत्य;
    }
    यदि (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        प्रत्यागम असत्य;
    }
    यदि (header->slot_गिनती == 0 || header->slot_गिनती > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

स्थिर अन्तःस्थ uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, पूर्णाङ्क slot) {
    यदि (!kleidiai_is_weight_header_valid(header)) {
        प्रत्यागम nullptr;
    }
    यदि (slot < 0 || slot >= header->slot_गिनती) {
        प्रत्यागम nullptr;
    }
    प्रत्यागम reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

स्थिर अन्तःस्थ ध्रुव uint8_t * kleidiai_weight_slot_ptr(ध्रुव kleidiai_weight_header * header, पूर्णाङ्क slot) {
    यदि (!kleidiai_is_weight_header_valid(header)) {
        प्रत्यागम nullptr;
    }
    यदि (slot < 0 || slot >= header->slot_गिनती) {
        प्रत्यागम nullptr;
    }
    प्रत्यागम reinterpret_cast<ध्रुव uint8_t *>(header) + header->offsets[slot];
}

स्थिर अन्तःस्थ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    प्रत्यागम ctx.kernels_q4;
}

स्थिर अन्तःस्थ ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    प्रत्यागम ctx.kernels_q4k;
}

स्थिर अन्तःस्थ ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    प्रत्यागम ctx.kernels_q8;
}

स्थिर अन्तःस्थ ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    प्रत्यागम ctx.kernels_f32;
}

ढाँचा <typename SelectFallback>
स्थिर पूर्णाङ्क kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    पूर्णाङ्क count = 0;
    यदि (!primary) {
        प्रत्यागम 0;
    }
    out[count++] = primary;

    यदि (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        प्रत्यागम count;
    }

    यदि (is_sme_family(primary->required_cpu)) {
        ध्रुव cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        यदि (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            यदि (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    प्रत्यागम count;
}

स्थिर पूर्णाङ्क kleidiai_collect_kernel_chain(ध्रुव संरचना ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    प्रत्यागम kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { प्रत्यागम ggml_kleidiai_select_kernels(mask, op); });
}

स्थिर पूर्णाङ्क kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    प्रत्यागम kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { प्रत्यागम ggml_kleidiai_select_kernels_q4_0(mask); });
}

स्थिर पूर्णाङ्क kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    प्रत्यागम kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { प्रत्यागम ggml_kleidiai_select_kernels_q4_k(mask); });
}

स्थिर पूर्णाङ्क kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    प्रत्यागम kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { प्रत्यागम ggml_kleidiai_select_kernels_q8_0(mask); });
}

स्थिर पूर्णाङ्क kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    प्रत्यागम kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { प्रत्यागम ggml_kleidiai_select_kernels_f32(mask); });
}

स्थिर अन्तःस्थ int64_t ggml_ne(ध्रुव ggml_tensor * tensor, पूर्णाङ्क dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    प्रत्यागम tensor->ne[dim];
}

नामाकाश ggml::cpu::kleidiai {

स्थिर माप_प्रकार round_down(माप_प्रकार x, माप_प्रकार y) {
    प्रत्यागम y == 0 ? x : x - (x % y);
}

स्थिर शून्य transpose_f32kxn_f16nxk(माप_प्रकार n, माप_प्रकार k, भिन्न * dst, ध्रुव uint16_t * src, माप_प्रकार rhs_stride) {
    माप_प्रकार src_stride = rhs_stride / परिमाण(uint16_t);
    माप_प्रकार dst_stride = n;

    प्रति (माप_प्रकार k_idx = 0; k_idx < k; ++k_idx) {
        प्रति (माप_प्रकार n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

वर्ग tensor_traits : सार्वजनिक ggml::cpu::tensor_traits {
    तार्किक work_size(पूर्णाङ्क /* n_threads */, ध्रुव संरचना ggml_tensor * op, माप_प्रकार & size) override {
        यदि (op->op != GGML_OP_MUL_MAT) {
            प्रत्यागम असत्य;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ध्रुव पूर्णाङ्क slot_गिनती = kleidiai_collect_kernel_chain(op, kernel_chain);
        यदि (slot_गिनती == 0) {
            प्रत्यागम असत्य;
        }

        ध्रुव तार्किक is_gemv = op->src[1]->ne[1] == 1;
        ध्रुव माप_प्रकार k = op->src[0]->ne[0];
        ध्रुव माप_प्रकार n = op->src[0]->ne[1];
        ध्रुव माप_प्रकार m = op->src[1]->ne[1];

        यदि (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ध्रुव kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            ध्रुव माप_प्रकार qk = block_args.pack_bl;

            माप_प्रकार cursor = 0;
            तार्किक any_slot = असत्य;

            प्रति (पूर्णाङ्क slot = 0; slot < slot_गिनती; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                यदि (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    प्रत्यागम असत्य;
                }

                ध्रुव माप_प्रकार mr = kernel->get_mr();
                ध्रुव माप_प्रकार kr = kernel->get_kr();
                ध्रुव माप_प्रकार sr = kernel->get_sr();

                ध्रुव माप_प्रकार packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = सत्य;
            }

            यदि (!any_slot) {
                प्रत्यागम असत्य;
            }

            size = cursor;
            प्रत्यागम सत्य;
        }

        यदि (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            यदि (is_gemv && op->src[1]->nb[0] == (int64_t) परिमाण(भिन्न) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                प्रत्यागम सत्य;
            }

            माप_प्रकार cursor = 0;
            तार्किक any_slot = असत्य;

            प्रति (पूर्णाङ्क slot = 0; slot < slot_गिनती; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                यदि (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    प्रत्यागम असत्य;
                }

                ध्रुव माप_प्रकार mr = kernel->get_mr();
                ध्रुव माप_प्रकार kr = kernel->get_kr();
                ध्रुव माप_प्रकार sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
                any_slot = सत्य;
            }

            यदि (!any_slot) {
                प्रत्यागम असत्य;
            }

            size = cursor;
            प्रत्यागम सत्य;
        }

        यदि (op->src[0]->type == GGML_TYPE_F16) {
            ध्रुव int64_t lhs_batch_size0 = op->src[1]->ne[2];
            ध्रुव int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            ध्रुव int64_t r = lhs_batch_size0 / rhs_batch_size0;

            माप_प्रकार cursor = 0;
            तार्किक any_slot = असत्य;

            प्रति (पूर्णाङ्क slot = 0; slot < slot_गिनती; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                यदि (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    प्रत्यागम असत्य;
                }

                ध्रुव माप_प्रकार mr = kernel->get_mr();
                ध्रुव माप_प्रकार kr = kernel->get_kr();
                ध्रुव माप_प्रकार sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = सत्य;
            }

            प्रति (पूर्णाङ्क slot = 0; slot < slot_गिनती; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                यदि (!kernel || !kernels->rhs_info.packed_size_ex) {
                    प्रत्यागम असत्य;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * परिमाण(भिन्न);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * परिमाण(भिन्न);

            यदि (!any_slot) {
                प्रत्यागम असत्य;
            }

            size = cursor;
            प्रत्यागम सत्य;
        }

        प्रत्यागम असत्य;
    }

    तार्किक compute_forward(संरचना ggml_compute_params * params, संरचना ggml_tensor * dst) override {
        यदि (dst->op == GGML_OP_MUL_MAT) {
            यदि (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                प्रत्यागम compute_forward_qx(params, dst);
            } अन्यथा यदि (dst->src[0]->type == GGML_TYPE_F32) {
                प्रत्यागम compute_forward_f32(params, dst);
            } अन्यथा यदि (dst->src[0]->type == GGML_TYPE_F16) {
                प्रत्यागम compute_forward_fp16(params, dst);
            }
        }
        प्रत्यागम असत्य;
    }

    तार्किक compute_forward_f32(ggml_compute_params * params, संरचना ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        ध्रुव ggml_tensor * src0 = dst->src[0];
        ध्रुव ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        यदि (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        यदि (!kernels) {
            प्रत्यागम असत्य;
        }

        ध्रुव माप_प्रकार k = ne00;
        ध्रुव माप_प्रकार m = ne11;
        ध्रुव माप_प्रकार n = ne01;
        ध्रुव तार्किक use_gemv = m == 1 && src1->nb[0] == (int64_t) परिमाण(भिन्न) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        यदि (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            प्रत्यागम असत्य;
        }

        यदि (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_size_ex || !lhs_info->pack_func_ex)) {
            प्रत्यागम असत्य;
        }

        ध्रुव kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ध्रुव तार्किक has_header = kleidiai_is_weight_header_valid(header);

        ध्रुव uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<ध्रुव uint8_t *>(src0->data);
        यदि (!rhs_base) {
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क nth = params->nth > 0 ? params->nth : 1;
        ध्रुव पूर्णाङ्क ith = params->ith;

        ध्रुव माप_प्रकार mr = kernel->get_mr();
        ध्रुव माप_प्रकार kr = kernel->get_kr();
        ध्रुव माप_प्रकार sr = kernel->get_sr();

        ध्रुव माप_प्रकार lhs_packed_size = use_gemv ? 0 : lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        यदि (!use_gemv) {
            GGML_ASSERT(lhs_packed_size <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ध्रुव माप_प्रकार dst_stride = dst->nb[1];
        ध्रुव माप_प्रकार n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        ध्रुव तार्किक disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (माप_प्रकार) पूर्णांक_उच्च);

        ध्रुव int64_t batch_size = ne12 * ne13;
        प्रति (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ध्रुव int64_t i12 = batch_idx % ne12;
            ध्रुव int64_t i13 = batch_idx / ne12;
            ध्रुव uint8_t * lhs_batch_base = static_cast<ध्रुव uint8_t *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];

            यदि (!use_gemv) {
                ध्रुव int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                ध्रुव int64_t use_threads = std::min<int64_t>(nth, max_threads);

                यदि (ith < use_threads) {
                    ध्रुव int64_t अंक_m_per_thread0   = round_down((माप_प्रकार)(m_roundup_mr / use_threads), mr);
                    ध्रुव int64_t अंक_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * अंक_m_per_thread0;

                    ध्रुव int64_t m_start = (int64_t)ith * अंक_m_per_thread0;
                    ध्रुव int64_t m_गिनती = (ith == use_threads - 1) ? अंक_m_per_thread_N_1 : अंक_m_per_thread0;

                    ध्रुव माप_प्रकार base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ध्रुव माप_प्रकार next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ध्रुव माप_प्रकार row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_गिनती;
                    int64_t cur       = m_start;

                    यावत् (remaining > 0) {
                        ध्रुव int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        ध्रुव माप_प्रकार src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        ध्रुव शून्य * src_ptr = lhs_batch_base + src_off;
                        ध्रुव माप_प्रकार dst_off = base_packed_off + (माप_प्रकार)(cur - m_start) * row_stride_bytes;
                        शून्य * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            यदि (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ध्रुव माप_प्रकार chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (माप_प्रकार) पूर्णांक_उच्च);

            पूर्णाङ्क current_col = ggml_threadpool_chunk_add(params->threadpool, (पूर्णाङ्क) chunk_cols);
            यावत् ((माप_प्रकार) current_col < n) {
                ध्रुव माप_प्रकार n_start = (माप_प्रकार) current_col;
                ध्रुव माप_प्रकार n_to_process = std::min(chunk_cols, n - n_start);

                यदि (n_to_process > 0) {
                    ध्रुव माप_प्रकार lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ध्रुव माप_प्रकार rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ध्रुव माप_प्रकार dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    ध्रुव शून्य * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    ध्रुव शून्य * rhs_ptr = rhs_base + rhs_packed_offset;
                    भिन्न * dst_ptr      = reinterpret_cast<भिन्न *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          परिमाण(भिन्न),
                                          -भग्न_उच्च,
                                          भग्न_उच्च);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (पूर्णाङ्क) chunk_cols);
            }

            यदि (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        प्रत्यागम सत्य;
    }

    तार्किक compute_forward_fp16(ggml_compute_params * params, संरचना ggml_tensor * dst) {
        ध्रुव ggml_tensor * src0 = dst->src[0];
        ध्रुव ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        यदि (!kernels) {
            प्रत्यागम असत्य;
        }

        ध्रुव तार्किक is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        यदि (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क nth = params->nth;
        ध्रुव पूर्णाङ्क ith = params->ith;

        ध्रुव int64_t lhs_batch_size0 = ne12;
        ध्रुव int64_t rhs_batch_size0 = ne02;
        ध्रुव int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        ध्रुव int64_t r = lhs_batch_size0 / rhs_batch_size0;

        ध्रुव int64_t m_group = ne11;
        ध्रुव int64_t m       = m_group;
        ध्रुव int64_t n       = ne01;
        ध्रुव int64_t k       = ne00;

        ध्रुव माप_प्रकार lhs_stride = src1->nb[1];
        ध्रुव माप_प्रकार rhs_stride = src0->nb[1];
        ध्रुव माप_प्रकार dst_stride = dst->nb[1];

        ध्रुव int64_t mr = (int64_t) kernel->get_mr();
        ध्रुव int64_t nr = (int64_t) kernel->get_nr();
        ध्रुव int64_t kr = (int64_t) kernel->get_kr();
        ध्रुव int64_t sr = (int64_t) kernel->get_sr();

        ध्रुव माप_प्रकार lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        ध्रुव माप_प्रकार rhs_packed_size = kernels->rhs_info.packed_size_ex(n, k, nr, kr, 0);
        ध्रुव माप_प्रकार kxn_size        = k * n * परिमाण(भिन्न);
        ध्रुव माप_प्रकार bias_size       = n * परिमाण(भिन्न);

        ध्रुव माप_प्रकार wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        प्रति (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ध्रुव int64_t rhs_batch_idx = batch_idx / r;
            ध्रुव uint8_t * rhs_batch_base = static_cast<ध्रुव uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ध्रुव int64_t m_roundup_mr = kai_roundup(m, mr);
                ध्रुव int64_t अंक_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                यदि (ith < अंक_threads) {
                    ध्रुव int64_t अंक_m_per_thread0   = round_down((माप_प्रकार)(m_roundup_mr / अंक_threads), (माप_प्रकार)mr);
                    ध्रुव int64_t अंक_m_per_thread_N_1 = m - (अंक_threads - 1) * अंक_m_per_thread0;

                    ध्रुव int64_t m_start = ith * अंक_m_per_thread0;
                    ध्रुव int64_t m_गिनती = (ith == अंक_threads - 1) ? अंक_m_per_thread_N_1 : अंक_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ध्रुव माप_प्रकार base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    ध्रुव माप_प्रकार next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    ध्रुव माप_प्रकार row_stride_bytes = (next_block_off - base_packed_off) / (माप_प्रकार)mr;

                    int64_t remaining = m_गिनती;
                    int64_t cur       = m_start;

                    यावत् (remaining > 0) {
                        ध्रुव int64_t row_in_group = cur;
                        ध्रुव int64_t avail        = m_group - row_in_group;
                        ध्रुव int64_t take         = std::min(avail, remaining);

                        ध्रुव uint8_t * lhs_batch_base = static_cast<ध्रुव uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        ध्रुव शून्य * src_ptr = lhs_batch_base + (माप_प्रकार)row_in_group * lhs_stride;
                        ध्रुव माप_प्रकार dst_off = base_packed_off + (माप_प्रकार)(cur - m_start) * row_stride_bytes;
                        शून्य * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            यदि (ith == 0) {
                स_रखो(bias, 0, (माप_प्रकार)n * परिमाण(भिन्न));
                transpose_f32kxn_f16nxk((माप_प्रकार)n, (माप_प्रकार)k,
                                        reinterpret_cast<भिन्न *>(rhs_kxn),
                                        reinterpret_cast<ध्रुव uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * परिमाण(भिन्न),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ध्रुव int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t अंक_threads_n = KAI_MIN(n / n_step, nth);
                यदि (अंक_threads_n <= 0) {
                    अंक_threads_n = 1;
                }

                यदि (ith < अंक_threads_n) {
                    ध्रुव int64_t अंक_n_per_thread0   = round_down((माप_प्रकार)(n / अंक_threads_n), (माप_प्रकार)n_step);
                    ध्रुव int64_t अंक_n_per_thread_N_1 = n - (अंक_threads_n - 1) * अंक_n_per_thread0;

                    ध्रुव int64_t n_start      = ith * अंक_n_per_thread0;
                    ध्रुव int64_t n_to_process = (ith == अंक_threads_n - 1) ? अंक_n_per_thread_N_1 : अंक_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ध्रुव माप_प्रकार lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    ध्रुव माप_प्रकार rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    ध्रुव माप_प्रकार dst_offset         = kernel->get_dst_offset((माप_प्रकार)0, (माप_प्रकार)n_start, dst_stride);

                    ध्रुव शून्य * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ध्रुव शून्य * rhs_ptr = rhs_packed + rhs_packed_offset;
                    भिन्न * dst_ptr      = reinterpret_cast<भिन्न *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, परिमाण(भिन्न), -भग्न_उच्च, भग्न_उच्च);
                }
            }

            यदि (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        प्रत्यागम सत्य;
    }

    तार्किक compute_forward_qx(संरचना ggml_compute_params * params, संरचना ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        ध्रुव ggml_tensor * src0 = dst->src[0];
        ध्रुव ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        ध्रुव kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        ध्रुव तार्किक has_header = kleidiai_is_weight_header_valid(header);
        ध्रुव तार्किक is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ध्रुव पूर्णाङ्क slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        स्वतः weight_for_slot = [&](पूर्णाङ्क slot_index, माप_प्रकार & size_out) -> ध्रुव uint8_t * {
            यदि (slot_index < 0 || slot_index >= slot_total) {
                प्रत्यागम nullptr;
            }
            यदि (has_header) {
                यदि (slot_index < header->slot_गिनती) {
                    size_out = static_cast<माप_प्रकार>(header->sizes[slot_index]);
                    प्रत्यागम kleidiai_weight_slot_ptr(header, slot_index);
                }
                प्रत्यागम nullptr;
            }
            यदि (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                प्रत्यागम static_cast<ध्रुव uint8_t *>(src0->data);
            }
            प्रत्यागम nullptr;
        };

        संरचना runtime_slot {
            पूर्णाङ्क slot_index;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            माप_प्रकार mr;
            माप_प्रकार nr;
            माप_प्रकार kr;
            माप_प्रकार sr;
            माप_प्रकार n_step;
            माप_प्रकार lhs_packed_size;
            माप_प्रकार lhs_offset;
            माप_प्रकार lhs_bl;
            माप_प्रकार rhs_bl;
            माप_प्रकार pack_bl;
            माप_प्रकार lhs_packed_offset0;
            पूर्णाङ्क assigned_threads;
            पूर्णाङ्क thread_begin;
            पूर्णाङ्क thread_end;
            ध्रुव uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        पूर्णाङ्क runtime_गिनती = 0;

        प्रति (पूर्णाङ्क slot = 0; slot < slot_total && runtime_गिनती < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            यदि (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                अनुवर्त;
            }

            माप_प्रकार rhs_size = 0;
            ध्रुव uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            यदि (!rhs_ptr || rhs_size == 0) {
                अनुवर्त;
            }

            ध्रुव kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

            runtime[runtime_गिनती] = {
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
            ++runtime_गिनती;
        }

        यदि (runtime_गिनती == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क nth_total = params->nth > 0 ? params->nth : 1;
        ध्रुव पूर्णाङ्क ith_total = params->ith;

        पूर्णाङ्क sme_slot = -1;
        पूर्णाङ्क non_sme_slot = -1;
        प्रति (पूर्णाङ्क i = 0; i < runtime_गिनती; ++i) {
            यदि (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                भङ्ग;
            }
        }

        प्रति (पूर्णाङ्क i = 0; i < runtime_गिनती; ++i) {
            यदि (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                भङ्ग;
            }
        }

        ध्रुव पूर्णाङ्क sme_cap_limit = ctx.sme_thread_cap;
        ध्रुव तार्किक use_hybrid = sme_cap_limit > 0 &&
                                 runtime_गिनती > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        माप_प्रकार min_cols_per_thread = 0;
        यदि (runtime_गिनती > 0 && nth_total > 0) {
            min_cols_per_thread = (माप_प्रकार) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        ध्रुव तार्किक too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        ध्रुव तार्किक hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        यदि (!hybrid_enabled) {
            पूर्णाङ्क chosen_slot = 0;
            यदि (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } अन्यथा यदि (runtime_गिनती > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            यदि (chosen_slot != 0 && chosen_slot < runtime_गिनती) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_गिनती = runtime_गिनती > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            यदि (runtime_गिनती > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        ध्रुव तार्किक sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        पूर्णाङ्क sme_cap = sme_single_only ? nth_total : kleidiai_sme_thread_cap();
        यदि (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        पूर्णाङ्क threads_remaining = nth_total;
        यदि (sme_slot != -1) {
            पूर्णाङ्क sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        पूर्णाङ्क fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        पूर्णाङ्क fallback_गिनती = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        प्रति (पूर्णाङ्क i = 0; i < runtime_गिनती; ++i) {
            यदि (i == sme_slot) {
                अनुवर्त;
            }
            fallback_indices[fallback_गिनती++] = i;
        }

        प्रति (पूर्णाङ्क fi = 0; fi < fallback_गिनती; ++fi) {
            यदि (threads_remaining <= 0) {
                भङ्ग;
            }
            ध्रुव पूर्णाङ्क slot_index = fallback_indices[fi];
            ध्रुव पूर्णाङ्क slots_बायाँ = fallback_गिनती - fi;
            पूर्णाङ्क share = (threads_remaining + slots_बायाँ - 1) / slots_बायाँ;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        पूर्णाङ्क thread_cursor = 0;
        प्रति (पूर्णाङ्क i = 0; i < runtime_गिनती; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ध्रुव पूर्णाङ्क active_threads = thread_cursor;

        पूर्णाङ्क local_slot = -1;
        पूर्णाङ्क local_ith  = 0;
        प्रति (पूर्णाङ्क i = 0; i < runtime_गिनती; ++i) {
            यदि (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                भङ्ग;
            }
        }
        ध्रुव माप_प्रकार k = ne00;
        ध्रुव माप_प्रकार m = ne11;
        ध्रुव माप_प्रकार n = ne01;

        माप_प्रकार cursor = 0;
        प्रति (पूर्णाङ्क i = 0; i < runtime_गिनती; ++i) {
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        माप_प्रकार common_step = 1;
        प्रति (पूर्णाङ्क i = 0; i < runtime_गिनती; ++i) {
            यदि (runtime[i].assigned_threads == 0) {
                अनुवर्त;
            }
            माप_प्रकार next_step = 0;
            यदि (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                प्रत्यागम असत्य;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        ध्रुव तार्किक disable_chunking = ggml_is_numa();
        ध्रुव माप_प्रकार chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ध्रुव माप_प्रकार chunk_भाजक = (active_threads == 1 || disable_chunking) ? (माप_प्रकार) active_threads : (माप_प्रकार) active_threads * chunk_multiplier;
        माप_प्रकार chunk_cols = align_up(std::max<माप_प्रकार>(1, ceil_div_size(n, chunk_भाजक)), common_step);
        यदि (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ध्रुव माप_प्रकार nchunk_size = std::max<माप_प्रकार>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (माप_प्रकार)पूर्णांक_उच्च);
        ध्रुव पूर्णाङ्क nchunk = (पूर्णाङ्क)nchunk_size;
        ध्रुव माप_प्रकार dst_stride = dst->nb[1];

        स्वतः run_chunk = [&](runtime_slot & slot, माप_प्रकार global_start, माप_प्रकार cols, uint8_t * dst_batch_base) {
            ध्रुव माप_प्रकार rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            ध्रुव माप_प्रकार dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            ध्रुव uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            ध्रुव uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            भिन्न * dst_ptr         = reinterpret_cast<भिन्न *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       परिमाण(भिन्न),
                                       -भग्न_उच्च,
                                       भग्न_उच्च);
        };

        प्रति (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ध्रुव uint8_t * lhs_batch_base = static_cast<ध्रुव uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            यदि (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                ध्रुव int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                ध्रुव int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                यदि (local_ith < use_threads) {
                    ध्रुव int64_t अंक_m_per_thread0   = round_down((माप_प्रकार)(m_roundup_mr / use_threads), slot.mr);
                    ध्रुव int64_t अंक_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * अंक_m_per_thread0;

                    ध्रुव int64_t m_start = (int64_t)local_ith * अंक_m_per_thread0;
                    ध्रुव int64_t m_गिनती = (local_ith == use_threads - 1) ? अंक_m_per_thread_N_1 : अंक_m_per_thread0;

                    ध्रुव माप_प्रकार base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ध्रुव माप_प्रकार next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ध्रुव माप_प्रकार row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_गिनती;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    यावत् (remaining > 0) {
                        ध्रुव int64_t row_in_group = cur;
                        ध्रुव int64_t avail        = (int64_t)m - row_in_group;
                        ध्रुव int64_t take         = std::min(avail, remaining);

                        ध्रुव माप_प्रकार src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        ध्रुव शून्य * src_ptr = lhs_batch_base + src_off;
                        ध्रुव माप_प्रकार dst_off = base_packed_off + (माप_प्रकार)(cur - m_start) * row_stride_bytes;
                        शून्य * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            यदि (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            यदि (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                पूर्णाङ्क current_chunk = ith_total;
                यावत् (current_chunk < nchunk) {
                    ध्रुव माप_प्रकार global_start = (माप_प्रकार)current_chunk * chunk_cols;
                    यदि (global_start >= n) {
                        भङ्ग;
                    }

                    ध्रुव माप_प्रकार cols = std::min(chunk_cols, n - global_start);
                    यदि (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            यदि (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        प्रत्यागम सत्य;
    }

सार्वजनिक:
    पूर्णाङ्क repack(संरचना ggml_tensor * tensor, ध्रुव शून्य * data, माप_प्रकार data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ध्रुव माप_प्रकार n = tensor->ne[1];
        ध्रुव माप_प्रकार k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        यदि (!header) {
            प्रत्यागम -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_गिनती = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        माप_प्रकार cursor = परिमाण(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ध्रुव तार्किक want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ध्रुव तार्किक want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ध्रुव तार्किक want_f32 = tensor->type == GGML_TYPE_F32;
        ध्रुव पूर्णाङ्क slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ध्रुव तार्किक allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<भिन्न>  scales;
        std::vector<भिन्न>  bias;
        std::vector<भिन्न> row_f32;

        यदि (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ध्रुव माप_प्रकार row_stride = tensor->nb[1];
            ध्रुव माप_प्रकार k_blocks   = (k + QK8_0 - 1) / QK8_0;

            यदि (tensor->type == GGML_TYPE_Q6_K) {
                यदि (k % QK_K != 0 || k % QK8_0 != 0) {
                    प्रत्यागम -1;
                }
                row_f32.resize(k);
            }

            प्रति (माप_प्रकार row = 0; row < n; ++row) {
                ध्रुव uint8_t * row_data = static_cast<ध्रुव uint8_t *>(data) + row * row_stride;

                यदि (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<ध्रुव block_q6_K *>(row_data), row_f32.data(), k);
                }

                भिन्न max_abs = 0.0f;
                यदि (tensor->type == GGML_TYPE_Q6_K) {
                    प्रति (माप_प्रकार linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::भ_असल(row_f32[linear_idx]));
                    }
                } अन्यथा {
                    ध्रुव block_q8_0 * row_blocks = reinterpret_cast<ध्रुव block_q8_0 *>(row_data);
                    प्रति (माप_प्रकार block = 0; block < k_blocks; ++block) {
                        ध्रुव block_q8_0 & blk = row_blocks[block];
                        ध्रुव भिन्न d = GGML_FP16_TO_FP32(blk.d);
                        प्रति (माप_प्रकार l = 0; l < QK8_0; ++l) {
                            ध्रुव माप_प्रकार linear_idx = block * QK8_0 + l;
                            यदि (linear_idx >= k) {
                                भङ्ग;
                            }
                            ध्रुव भिन्न value = d * static_cast<भिन्न>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::भ_असल(value));
                        }
                    }
                }

                भिन्न scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                ध्रुव भिन्न inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                यदि (tensor->type == GGML_TYPE_Q6_K) {
                    प्रति (माप_प्रकार linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } अन्यथा {
                    ध्रुव block_q8_0 * row_blocks = reinterpret_cast<ध्रुव block_q8_0 *>(row_data);
                    प्रति (माप_प्रकार block = 0; block < k_blocks; ++block) {
                        ध्रुव block_q8_0 & blk = row_blocks[block];
                        ध्रुव भिन्न d = GGML_FP16_TO_FP32(blk.d);
                        प्रति (माप_प्रकार l = 0; l < QK8_0; ++l) {
                            ध्रुव माप_प्रकार linear_idx = block * QK8_0 + l;
                            यदि (linear_idx >= k) {
                                भङ्ग;
                            }
                            ध्रुव भिन्न value = d * static_cast<भिन्न>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        यदि (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
        }

        प्रति (पूर्णाङ्क slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            यदि (!allow_fallback && slot > 0) {
                भङ्ग;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            यदि (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                अनुवर्त;
            }

            ध्रुव माप_प्रकार nr = kernel->get_nr();
            ध्रुव माप_प्रकार kr = kernel->get_kr();
            ध्रुव माप_प्रकार sr = kernel->get_sr();
            ध्रुव ggml_type rhs_type = kernels->rhs_type;
            यदि (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            ध्रुव माप_प्रकार block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            यदि (block_len == SIZE_MAX) {
                अनुवर्त;
            }

            ध्रुव माप_प्रकार packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            ध्रुव माप_प्रकार aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            यदि (rhs_type == GGML_TYPE_Q4_0) {
                संरचना kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ध्रुव uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } अन्यथा यदि (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       data, nullptr, nullptr, dst_ptr, 0, nullptr);
            } अन्यथा यदि (rhs_type == GGML_TYPE_Q8_0) {
                संरचना kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } अन्यथा यदि (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
            } अन्यथा {
                अनुवर्त;
            }

            header->offsets[header->slot_गिनती] = aligned_cursor;
            header->sizes[header->slot_गिनती]   = packed_size;
            ++header->slot_गिनती;

            cursor = aligned_cursor + packed_size;
        }

        यदि (header->slot_गिनती == 0) {
            header->magic   = 0;
            header->version = 0;
            स_नकल(tensor->data, data, data_size);
        }

        प्रत्यागम 0;
    }
};

स्थिर ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, संरचना ggml_tensor *) {
    स्थिर tensor_traits traits;
    प्रत्यागम &traits;
}
}  // namespace ggml::cpu::kleidiai

स्थिर गणना ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor) {
    tensor->extra = (शून्य *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    प्रत्यागम GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor,
                                                       ध्रुव शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    स्वतः tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    स्वतः OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

स्थिर ध्रुव वर्ण * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    प्रत्यागम "CPU_KLEIDIAI";
}

स्थिर ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, माप_प्रकार size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    यदि (buffer == nullptr) {
        प्रत्यागम nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    प्रत्यागम buffer;
}

स्थिर माप_प्रकार ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    प्रत्यागम TENSOR_ALIGNMENT;
}

स्थिर माप_प्रकार ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ध्रुव संरचना ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    यदि (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        प्रत्यागम ggml_nbytes(tensor);
    }

    ध्रुव माप_प्रकार n = tensor->ne[1];
    ध्रुव माप_प्रकार k = tensor->ne[0];

    माप_प्रकार cursor = परिमाण(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ध्रुव तार्किक want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ध्रुव तार्किक want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ध्रुव तार्किक want_f32 = tensor->type == GGML_TYPE_F32;
    ध्रुव पूर्णाङ्क slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ध्रुव तार्किक allow_fallback = kleidiai_pack_fallback_allowed();

    माप_प्रकार slot_गिनती = 0;
    प्रति (पूर्णाङ्क slot = 0; slot < slot_total; ++slot) {
        यदि (!allow_fallback && slot > 0) {
            भङ्ग;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        यदि (!kernels) {
            अनुवर्त;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        यदि (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            अनुवर्त;
        }

        ध्रुव ggml_type rhs_type = kernels->rhs_type;
        ध्रुव माप_प्रकार block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        यदि (block_len == SIZE_MAX) {
            अनुवर्त;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_गिनती;
    }

    यदि (slot_गिनती == 0) {
        प्रत्यागम ggml_nbytes(tensor);
    }

    प्रत्यागम std::max(cursor, ggml_nbytes(tensor));
}

नामाकाश ggml::cpu::kleidiai {
वर्ग extra_buffer_type : ggml::cpu::extra_buffer_type {
    तार्किक supports_op(ggml_backend_dev_t, ध्रुव संरचना ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        यदि (op->op != GGML_OP_MUL_MAT) {
            प्रत्यागम असत्य;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ध्रुव पूर्णाङ्क slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        ध्रुव तार्किक src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft->context == एतत् &&
            slot_total > 0;

        यदि ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            यदि (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                प्रत्यागम असत्य;
            }
            यदि (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                प्रत्यागम असत्य;
            }
            यदि (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                प्रत्यागम असत्य;
            }
            यदि (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                प्रत्यागम असत्य;
            }
            यदि (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                प्रत्यागम असत्य;
            }
            यदि (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                प्रत्यागम असत्य;
            }

            यदि (op->src[0]->type == GGML_TYPE_Q4_K) {
                यदि (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    प्रत्यागम असत्य;
                }
                प्रत्यागम ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == परिमाण(भिन्न) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * परिमाण(भिन्न));
            }

            यदि (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                यदि (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    प्रत्यागम सत्य;
                }
                प्रत्यागम असत्य;
            }

            यदि (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                प्रत्यागम असत्य;
            }

            प्रत्यागम सत्य;
        }

        प्रत्यागम असत्य;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ध्रुव संरचना ggml_tensor * op) override {
        यदि (op->op == GGML_OP_MUL_MAT) {
            यदि (op->src[0]->buffer && op->src[0]->buffer->buft->context == एतत्) {
                प्रत्यागम (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } अन्यथा {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                यदि (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    स्थिर std::atomic<तार्किक> warned(असत्य);
                    यदि (!warned.exchange(सत्य)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                यदि (op->src[0]->type != GGML_TYPE_F16) {
                    प्रत्यागम nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ध्रुव पूर्णाङ्क slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                यदि (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    यदि ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        प्रत्यागम nullptr;
                    }
                    प्रत्यागम ggml::cpu::kleidiai::get_tensor_traits(रिक्त, रिक्त);
                }
            }
        }
        प्रत्यागम nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(शून्य) {
    स्थिर ggml::cpu::kleidiai::extra_buffer_type ctx;
    स्थिर संरचना ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .get_alloc_size_n     = */ रिक्त,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    प्रत्यागम &ggml_backend_cpu_buffer_type_kleidiai;
}
