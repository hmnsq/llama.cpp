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
#जर defined(__linux__)
#अन्तर्भाव <asm/hwcap.h>
#अन्तर्भाव <dirent.h>
#अन्तर्भाव <sys/auxv.h>
#अन्तर्भाव <sys/types.h>
#अन्तर्भाव <sys/stat.h>
#अन्तर्भाव <unistd.h>
#किंवाजर defined(__APPLE__)
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

स्थिर constexpr पूर्णांक      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
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
    पूर्णांक sme_thread_cap; // <= 0 means "SME disabled/unknown"
    पूर्णांक thread_hint;    // <= 0 means "no hint"
    पूर्णांक chunk_multiplier;
} स्थिर ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

स्थिर ओळीत तार्किक is_sme_family(cpu_feature f) {
    परत (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

स्थिर स्थिरांक अक्षर* cpu_feature_to_string(cpu_feature f) {
    जर (f == CPU_FEATURE_NONE) {
        परत "NONE";
    } अन्यथा जर ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        परत "SME2";
    } अन्यथा जर ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        परत "SME";
    } अन्यथा जर ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        परत "SVE";
    }
    अन्यथा जर ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        परत "I8MM";
    } अन्यथा जर ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        परत "DOTPROD";
    }
    अन्यथा {
        परत "UNKNOWN";
    }
}

#जर defined(__linux__) && defined(__aarch64__)
स्थिर तार्किक parse_cpu_dir_name(स्थिरांक अक्षर* name, माप_प्रकार* cpu) {
    जर (म_भेदन(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        परत असत्य;
    }

    स्थिरांक अक्षर* first = name + 3;
    स्थिरांक अक्षर* last = name + म_माप(name);

    माप_प्रकार value = 0;
    स्थिरांक स्वतः [end, ec] = std::from_chars(first, last, value, 10);

    जर (ec != std::errc{} || end != last) {
        परत असत्य;
    }

    *cpu = value;
    परत सत्य;
}

स्थिर std::vector<माप_प्रकार> detect_cpu_ids() {
    std::vector<माप_प्रकार> cpus;

    सूची * dir = सूची_खोलो("/sys/devices/system/cpu");
    जर (dir == nullptr) {
        परत cpus;
    }

    पर्यंत (dirent * entry = सूची_पढ़ो(dir)) {
        माप_प्रकार cpu = 0;
        जर (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    बंद_सूची(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    परत cpus;
}
#endif

#जर defined(__APPLE__) && defined(__aarch64__)
स्थिर तार्किक apple_sme_counted_perf_level(std::string name) {
    साठी (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (अक्षर) std::छोटे((अचिन्हित अक्षर) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    परत name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#endif

स्थिर शून्य add_smcus_from_smidr(uint64_t smidr, माप_प्रकार & अंक_private, std::map<uint32_t, माप_प्रकार> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    स्थिरांक uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    स्थिरांक uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    स्थिरांक माप_प्रकार shared_मोजणी = nsmc == 0xF ? 1 : (माप_प्रकार)nsmc + 1;
    स्थिरांक uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    स्थिरांक uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    स्थिरांक uint32_t id = (affinity2 << 12) | affinity;

    जर (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    निवड (sh) {
        स्थिती 2: // private SMCU
            ++अंक_private;
            थांबा;
        स्थिती 3: // shared SMCU
            जर (shared_counts[id] < shared_मोजणी) {
                shared_counts[id] = shared_मोजणी;
            }
            थांबा;
        स्थिती 0:
            जर (id == 0) {
                ++अंक_private;
            } अन्यथा जर (shared_counts[id] < shared_मोजणी) {
                shared_counts[id] = shared_मोजणी;
            }
            थांबा;
        पूर्वनिर्धारित:
            थांबा;
    }
}

स्थिर माप_प्रकार detect_अंक_smcus() {
    स्थिरांक स्वतः runtime_feat = ggml_feats_get_arch64_runtime();
    जर (!runtime_feat.has_sme) {
        परत 0;
    }

#जर defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    माप_प्रकार अंक_private = 0;
    std::map<uint32_t, माप_प्रकार> shared_counts;

    स्थिरांक std::vector<माप_प्रकार> cpus = detect_cpu_ids();
    साठी (स्थिरांक माप_प्रकार cpu : cpus) {
        स्थिरांक std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::ifstream file(path);
        जर (!file.is_open()) {
            सुरूठेवा;
        }

        uint64_t smidr = 0;
        जर (!(file >> std::hex >> smidr)) {
            सुरूठेवा;
        }

        add_smcus_from_smidr(smidr, अंक_private, shared_counts);
    }

    माप_प्रकार total = अंक_private;
    साठी (स्थिरांक स्वतः & entry : shared_counts) {
        total += entry.second;
    }
    परत total;

#किंवाजर defined(__APPLE__) && defined(__aarch64__)
    पूर्णांक perf_levels = 0;
    माप_प्रकार size = आकार(perf_levels);
    जर (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != आकार(perf_levels) || perf_levels <= 0) {
        परत 0;
    }

    माप_प्रकार units = 0;
    साठी (पूर्णांक i = 0; i < perf_levels; ++i) {
        अक्षर किल्ली[64] = {};
        पूर्णांक physical_cpus = 0;
        पूर्णांक cpus_per_l2 = 0;

        snprintf(किल्ली, आकार(किल्ली), "hw.perflevel%d.physicalcpu", i);
        size = आकार(physical_cpus);
        जर (sysctlbyname(किल्ली, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != आकार(physical_cpus) || physical_cpus <= 0) {
            सुरूठेवा;
        }

        snprintf(किल्ली, आकार(किल्ली), "hw.perflevel%d.cpusperl2", i);
        size = आकार(cpus_per_l2);
        जर (sysctlbyname(किल्ली, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != आकार(cpus_per_l2) || cpus_per_l2 <= 0) {
            सुरूठेवा;
        }

        snprintf(किल्ली, आकार(किल्ली), "hw.perflevel%d.name", i);
        size = 0;
        जर (sysctlbyname(किल्ली, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            सुरूठेवा;
        }

        std::string name(size, '\0');
        जर (sysctlbyname(किल्ली, &name[0], &size, nullptr, 0) != 0) {
            सुरूठेवा;
        }
        name.resize(size);
        पर्यंत (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        जर (apple_sme_counted_perf_level(name)) {
            units += (माप_प्रकार) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    परत units;

#किंवाजर defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    परत 0;

#अन्यथा
    परत 0;
#endif
}

स्थिर पूर्णांक parse_uint_env(स्थिरांक अक्षर *s, स्थिरांक अक्षर *name, तार्किक *ok) {
    जर (!s) { *ok = असत्य; परत 0; }
    अक्षर *end = nullptr;
    दीर्घ v = म_से_दीर्घ(s, &end, 10);
    जर (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = असत्य;
        परत 0;
    }
    जर (v < 0 || v > पूर्णांक_उच्च) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = असत्य;
        परत 0;
    }
    *ok = सत्य;
    परत (पूर्णांक)v;
}

स्थिर शून्य init_kleidiai_context(शून्य) {
    ggml_critical_section_start();
    स्थिर तार्किक initialized = असत्य;

    जर (!initialized) {
        initialized = सत्य;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        स्थिरांक अक्षर *env_sme         = दो_पर्या("GGML_KLEIDIAI_SME");
        स्थिरांक अक्षर *env_threads     = दो_पर्या("GGML_TOTAL_THREADS");
        स्थिरांक अक्षर *env_chunk_mult  = दो_पर्या("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        स्थिरांक स्वतः runtime_feat = ggml_feats_get_arch64_runtime();

        माप_प्रकार detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        जर (env_threads) {
            तार्किक ok = असत्य;
            पूर्णांक hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            जर (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        जर (env_chunk_mult) {
            तार्किक ok = असत्य;
            पूर्णांक multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            जर (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        पूर्णांक sme_cores = 0;
        तार्किक sme_env_ok = असत्य;
        तार्किक sme_env_set = (env_sme != nullptr);

        स्थिरांक तार्किक has_supported_sme_family = runtime_feat.has_sme;
        तार्किक sme_cap_detected = असत्य;

        जर (has_supported_sme_family) {
            detected_smcus = detect_अंक_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (पूर्णांक)detected_smcus : 1;

            जर (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        जर (sme_env_set) {
            तार्किक ok = असत्य;
            पूर्णांक v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            जर (ok) {
                जर (has_supported_sme_family) {
                    sme_cores = v;
                } अन्यथा {
                    जर (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } अन्यथा {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        जर (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            जर (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        जर (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (पूर्णांक)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        जर (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (पूर्णांक)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        जर (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (पूर्णांक) ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        जर (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (पूर्णांक)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        स्थिरांक तार्किक has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        जर (has_selected_sme_family_kernel) {
            जर (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } अन्यथा जर (sme_cap_detected) {
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

स्थिर ओळीत पूर्णांक kleidiai_sme_thread_cap() {
    परत ctx.sme_thread_cap;
}

स्थिर ओळीत माप_प्रकार align_up(माप_प्रकार value, माप_प्रकार alignment) {
    जर (alignment == 0) {
        परत value;
    }
    स्थिरांक माप_प्रकार remainder = value % alignment;
    परत remainder == 0 ? value : value + (alignment - remainder);
}

स्थिर ओळीत माप_प्रकार gcd_size(माप_प्रकार a, माप_प्रकार b) {
    पर्यंत (b != 0) {
        स्थिरांक माप_प्रकार t = a % b;
        a = b;
        b = t;
    }
    परत a;
}

स्थिर ओळीत तार्किक lcm_size(माप_प्रकार a, माप_प्रकार b, माप_प्रकार & result) {
    जर (a == 0 || b == 0) {
        result = 0;
        परत असत्य;
    }
    स्थिरांक माप_प्रकार g = gcd_size(a, b);
    स्थिरांक माप_प्रकार q = a / g;
    जर (q > SIZE_MAX / b) {
        परत असत्य;
    }
    result = q * b;
    परत सत्य;
}

स्थिर ओळीत माप_प्रकार ceil_div_size(माप_प्रकार a, माप_प्रकार b) {
    परत b == 0 ? 0 : (a + b - 1) / b;
}

स्थिर ओळीत माप_प्रकार kleidiai_chunk_cols(माप_प्रकार n, पूर्णांक nth_total, तार्किक disable_chunking, माप_प्रकार n_step) {
    स्थिरांक माप_प्रकार multiplier = (nth_total == 1 || disable_chunking) ? 1 : std::max<माप_प्रकार>(1, (माप_प्रकार) ctx.chunk_multiplier);
    स्थिरांक माप_प्रकार भाजक = std::max<माप_प्रकार>(1, (माप_प्रकार) nth_total * multiplier);
    स्थिरांक माप_प्रकार chunk_cols = align_up(std::max<माप_प्रकार>(1, ceil_div_size(n, भाजक)), n_step);
    परत chunk_cols ? chunk_cols : n_step;
}

संरचना kleidiai_block_args {
    माप_प्रकार lhs_bl;
    माप_प्रकार rhs_bl;
    माप_प्रकार pack_bl;
};

स्थिर ओळीत kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    निवड (rhs_type) {
        स्थिती GGML_TYPE_Q4_0:
            परत { QK4_0, QK4_0, QK4_0 };
        स्थिती GGML_TYPE_Q4_K:
            परत { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        स्थिती GGML_TYPE_Q8_0:
            परत { 0, 0, QK8_0 };
        पूर्वनिर्धारित:
            परत { 0, 0, 0 };
    }
}

स्थिर ओळीत तार्किक kleidiai_pack_fallback_allowed() {
    जर (ctx.sme_thread_cap <= 0) {
        परत असत्य;
    }
    जर (ctx.thread_hint <= 0) {
        परत सत्य;
    }
    परत ctx.thread_hint > ctx.sme_thread_cap;
}

संरचना kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_मोजणी;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

स्थिर ओळीत kleidiai_weight_header * kleidiai_weight_header_from_ptr(शून्य * data) {
    परत reinterpret_cast<kleidiai_weight_header *>(data);
}

स्थिर ओळीत स्थिरांक kleidiai_weight_header * kleidiai_weight_header_from_ptr(स्थिरांक शून्य * data) {
    परत reinterpret_cast<स्थिरांक kleidiai_weight_header *>(data);
}

स्थिर ओळीत तार्किक kleidiai_is_weight_header_valid(स्थिरांक kleidiai_weight_header * header) {
    जर (!header) {
        परत असत्य;
    }
    जर (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        परत असत्य;
    }
    जर (header->slot_मोजणी == 0 || header->slot_मोजणी > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        परत असत्य;
    }
    परत सत्य;
}

स्थिर ओळीत uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, पूर्णांक slot) {
    जर (!kleidiai_is_weight_header_valid(header)) {
        परत nullptr;
    }
    जर (slot < 0 || slot >= header->slot_मोजणी) {
        परत nullptr;
    }
    परत reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

स्थिर ओळीत स्थिरांक uint8_t * kleidiai_weight_slot_ptr(स्थिरांक kleidiai_weight_header * header, पूर्णांक slot) {
    जर (!kleidiai_is_weight_header_valid(header)) {
        परत nullptr;
    }
    जर (slot < 0 || slot >= header->slot_मोजणी) {
        परत nullptr;
    }
    परत reinterpret_cast<स्थिरांक uint8_t *>(header) + header->offsets[slot];
}

स्थिर ओळीत ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    परत ctx.kernels_q4;
}

स्थिर ओळीत ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    परत ctx.kernels_q4k;
}

स्थिर ओळीत ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    परत ctx.kernels_q8;
}

स्थिर ओळीत ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    परत ctx.kernels_f32;
}

ढाँचा <typename SelectFallback>
स्थिर पूर्णांक kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    पूर्णांक count = 0;
    जर (!primary) {
        परत 0;
    }
    out[count++] = primary;

    जर (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        परत count;
    }

    जर (is_sme_family(primary->required_cpu)) {
        स्थिरांक cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        जर (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            जर (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    परत count;
}

स्थिर पूर्णांक kleidiai_collect_kernel_chain(स्थिरांक संरचना ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    परत kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { परत ggml_kleidiai_select_kernels(mask, op); });
}

स्थिर पूर्णांक kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    परत kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { परत ggml_kleidiai_select_kernels_q4_0(mask); });
}

स्थिर पूर्णांक kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    परत kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { परत ggml_kleidiai_select_kernels_q4_k(mask); });
}

स्थिर पूर्णांक kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    परत kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { परत ggml_kleidiai_select_kernels_q8_0(mask); });
}

स्थिर पूर्णांक kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    परत kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { परत ggml_kleidiai_select_kernels_f32(mask); });
}

स्थिर ओळीत int64_t ggml_ne(स्थिरांक ggml_tensor * tensor, पूर्णांक dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    परत tensor->ne[dim];
}

नामाकाश ggml::cpu::kleidiai {

स्थिर माप_प्रकार round_down(माप_प्रकार x, माप_प्रकार y) {
    परत y == 0 ? x : x - (x % y);
}

स्थिर शून्य transpose_f32kxn_f16nxk(माप_प्रकार n, माप_प्रकार k, दशांश * dst, स्थिरांक uint16_t * src, माप_प्रकार rhs_stride) {
    माप_प्रकार src_stride = rhs_stride / आकार(uint16_t);
    माप_प्रकार dst_stride = n;

    साठी (माप_प्रकार k_idx = 0; k_idx < k; ++k_idx) {
        साठी (माप_प्रकार n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

वर्ग tensor_traits : सार्वजनिक ggml::cpu::tensor_traits {
    तार्किक work_size(पूर्णांक /* n_threads */, स्थिरांक संरचना ggml_tensor * op, माप_प्रकार & size) override {
        जर (op->op != GGML_OP_MUL_MAT) {
            परत असत्य;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        स्थिरांक पूर्णांक slot_मोजणी = kleidiai_collect_kernel_chain(op, kernel_chain);
        जर (slot_मोजणी == 0) {
            परत असत्य;
        }

        स्थिरांक तार्किक is_gemv = op->src[1]->ne[1] == 1;
        स्थिरांक माप_प्रकार k = op->src[0]->ne[0];
        स्थिरांक माप_प्रकार n = op->src[0]->ne[1];
        स्थिरांक माप_प्रकार m = op->src[1]->ne[1];

        जर (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            स्थिरांक kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            स्थिरांक माप_प्रकार qk = block_args.pack_bl;

            माप_प्रकार cursor = 0;
            तार्किक any_slot = असत्य;

            साठी (पूर्णांक slot = 0; slot < slot_मोजणी; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                जर (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    परत असत्य;
                }

                स्थिरांक माप_प्रकार mr = kernel->get_mr();
                स्थिरांक माप_प्रकार kr = kernel->get_kr();
                स्थिरांक माप_प्रकार sr = kernel->get_sr();

                स्थिरांक माप_प्रकार packed = lhs_info->packed_size_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = सत्य;
            }

            जर (!any_slot) {
                परत असत्य;
            }

            size = cursor;
            परत सत्य;
        }

        जर (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            जर (is_gemv && op->src[1]->nb[0] == (int64_t) आकार(दशांश) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                परत सत्य;
            }

            माप_प्रकार cursor = 0;
            तार्किक any_slot = असत्य;

            साठी (पूर्णांक slot = 0; slot < slot_मोजणी; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                जर (!lhs_info || !lhs_info->packed_size_ex || !kernel) {
                    परत असत्य;
                }

                स्थिरांक माप_प्रकार mr = kernel->get_mr();
                स्थिरांक माप_प्रकार kr = kernel->get_kr();
                स्थिरांक माप_प्रकार sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
                any_slot = सत्य;
            }

            जर (!any_slot) {
                परत असत्य;
            }

            size = cursor;
            परत सत्य;
        }

        जर (op->src[0]->type == GGML_TYPE_F16) {
            स्थिरांक int64_t lhs_batch_size0 = op->src[1]->ne[2];
            स्थिरांक int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            स्थिरांक int64_t r = lhs_batch_size0 / rhs_batch_size0;

            माप_प्रकार cursor = 0;
            तार्किक any_slot = असत्य;

            साठी (पूर्णांक slot = 0; slot < slot_मोजणी; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                जर (!lhs_info || !lhs_info->packed_size_ex || !kernels->rhs_info.packed_size_ex || !kernel) {
                    परत असत्य;
                }

                स्थिरांक माप_प्रकार mr = kernel->get_mr();
                स्थिरांक माप_प्रकार kr = kernel->get_kr();
                स्थिरांक माप_प्रकार sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_size_ex(m * r, k, 0, mr, kr, sr);
                any_slot = सत्य;
            }

            साठी (पूर्णांक slot = 0; slot < slot_मोजणी; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                जर (!kernel || !kernels->rhs_info.packed_size_ex) {
                    परत असत्य;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * आकार(दशांश);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * आकार(दशांश);

            जर (!any_slot) {
                परत असत्य;
            }

            size = cursor;
            परत सत्य;
        }

        परत असत्य;
    }

    तार्किक compute_forward(संरचना ggml_compute_params * params, संरचना ggml_tensor * dst) override {
        जर (dst->op == GGML_OP_MUL_MAT) {
            जर (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                परत compute_forward_qx(params, dst);
            } अन्यथा जर (dst->src[0]->type == GGML_TYPE_F32) {
                परत compute_forward_f32(params, dst);
            } अन्यथा जर (dst->src[0]->type == GGML_TYPE_F16) {
                परत compute_forward_fp16(params, dst);
            }
        }
        परत असत्य;
    }

    तार्किक compute_forward_f32(ggml_compute_params * params, संरचना ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        स्थिरांक ggml_tensor * src0 = dst->src[0];
        स्थिरांक ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_द्विमान_OP_LOCALS

        जर (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            परत असत्य;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        जर (!kernels) {
            परत असत्य;
        }

        स्थिरांक माप_प्रकार k = ne00;
        स्थिरांक माप_प्रकार m = ne11;
        स्थिरांक माप_प्रकार n = ne01;
        स्थिरांक तार्किक use_gemv = m == 1 && src1->nb[0] == (int64_t) आकार(दशांश) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        जर (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            परत असत्य;
        }

        जर (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_size_ex || !lhs_info->pack_func_ex)) {
            परत असत्य;
        }

        स्थिरांक kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        स्थिरांक तार्किक has_header = kleidiai_is_weight_header_valid(header);

        स्थिरांक uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<स्थिरांक uint8_t *>(src0->data);
        जर (!rhs_base) {
            परत असत्य;
        }

        स्थिरांक पूर्णांक nth = params->nth > 0 ? params->nth : 1;
        स्थिरांक पूर्णांक ith = params->ith;

        स्थिरांक माप_प्रकार mr = kernel->get_mr();
        स्थिरांक माप_प्रकार kr = kernel->get_kr();
        स्थिरांक माप_प्रकार sr = kernel->get_sr();

        स्थिरांक माप_प्रकार lhs_packed_size = use_gemv ? 0 : lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        जर (!use_gemv) {
            GGML_ASSERT(lhs_packed_size <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        स्थिरांक माप_प्रकार dst_stride = dst->nb[1];
        स्थिरांक माप_प्रकार n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        स्थिरांक तार्किक disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (माप_प्रकार) पूर्णांक_उच्च);

        स्थिरांक int64_t batch_size = ne12 * ne13;
        साठी (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            स्थिरांक int64_t i12 = batch_idx % ne12;
            स्थिरांक int64_t i13 = batch_idx / ne12;
            स्थिरांक uint8_t * lhs_batch_base = static_cast<स्थिरांक uint8_t *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + i12 * dst->nb[2] + i13 * dst->nb[3];

            जर (!use_gemv) {
                स्थिरांक int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                स्थिरांक int64_t use_threads = std::min<int64_t>(nth, max_threads);

                जर (ith < use_threads) {
                    स्थिरांक int64_t अंक_m_per_thread0   = round_down((माप_प्रकार)(m_roundup_mr / use_threads), mr);
                    स्थिरांक int64_t अंक_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * अंक_m_per_thread0;

                    स्थिरांक int64_t m_start = (int64_t)ith * अंक_m_per_thread0;
                    स्थिरांक int64_t m_मोजणी = (ith == use_threads - 1) ? अंक_m_per_thread_N_1 : अंक_m_per_thread0;

                    स्थिरांक माप_प्रकार base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    स्थिरांक माप_प्रकार next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    स्थिरांक माप_प्रकार row_stride_bytes = mr ? (next_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_मोजणी;
                    int64_t cur       = m_start;

                    पर्यंत (remaining > 0) {
                        स्थिरांक int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        स्थिरांक माप_प्रकार src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        स्थिरांक शून्य * src_ptr = lhs_batch_base + src_off;
                        स्थिरांक माप_प्रकार dst_off = base_packed_off + (माप_प्रकार)(cur - m_start) * row_stride_bytes;
                        शून्य * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            जर (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            स्थिरांक माप_प्रकार chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (माप_प्रकार) पूर्णांक_उच्च);

            पूर्णांक current_col = ggml_threadpool_chunk_add(params->threadpool, (पूर्णांक) chunk_cols);
            पर्यंत ((माप_प्रकार) current_col < n) {
                स्थिरांक माप_प्रकार n_start = (माप_प्रकार) current_col;
                स्थिरांक माप_प्रकार n_to_process = std::min(chunk_cols, n - n_start);

                जर (n_to_process > 0) {
                    स्थिरांक माप_प्रकार lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    स्थिरांक माप_प्रकार rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    स्थिरांक माप_प्रकार dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    स्थिरांक शून्य * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    स्थिरांक शून्य * rhs_ptr = rhs_base + rhs_packed_offset;
                    दशांश * dst_ptr      = reinterpret_cast<दशांश *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          आकार(दशांश),
                                          -भग्न_उच्च,
                                          भग्न_उच्च);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (पूर्णांक) chunk_cols);
            }

            जर (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        परत सत्य;
    }

    तार्किक compute_forward_fp16(ggml_compute_params * params, संरचना ggml_tensor * dst) {
        स्थिरांक ggml_tensor * src0 = dst->src[0];
        स्थिरांक ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_द्विमान_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        जर (!kernels) {
            परत असत्य;
        }

        स्थिरांक तार्किक is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        जर (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            परत असत्य;
        }

        स्थिरांक पूर्णांक nth = params->nth;
        स्थिरांक पूर्णांक ith = params->ith;

        स्थिरांक int64_t lhs_batch_size0 = ne12;
        स्थिरांक int64_t rhs_batch_size0 = ne02;
        स्थिरांक int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        स्थिरांक int64_t r = lhs_batch_size0 / rhs_batch_size0;

        स्थिरांक int64_t m_group = ne11;
        स्थिरांक int64_t m       = m_group;
        स्थिरांक int64_t n       = ne01;
        स्थिरांक int64_t k       = ne00;

        स्थिरांक माप_प्रकार lhs_stride = src1->nb[1];
        स्थिरांक माप_प्रकार rhs_stride = src0->nb[1];
        स्थिरांक माप_प्रकार dst_stride = dst->nb[1];

        स्थिरांक int64_t mr = (int64_t) kernel->get_mr();
        स्थिरांक int64_t nr = (int64_t) kernel->get_nr();
        स्थिरांक int64_t kr = (int64_t) kernel->get_kr();
        स्थिरांक int64_t sr = (int64_t) kernel->get_sr();

        स्थिरांक माप_प्रकार lhs_packed_size = lhs_info->packed_size_ex(m, k, 0, mr, kr, sr);
        स्थिरांक माप_प्रकार rhs_packed_size = kernels->rhs_info.packed_size_ex(n, k, nr, kr, 0);
        स्थिरांक माप_प्रकार kxn_size        = k * n * आकार(दशांश);
        स्थिरांक माप_प्रकार bias_size       = n * आकार(दशांश);

        स्थिरांक माप_प्रकार wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        साठी (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            स्थिरांक int64_t rhs_batch_idx = batch_idx / r;
            स्थिरांक uint8_t * rhs_batch_base = static_cast<स्थिरांक uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                स्थिरांक int64_t m_roundup_mr = kai_roundup(m, mr);
                स्थिरांक int64_t अंक_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                जर (ith < अंक_threads) {
                    स्थिरांक int64_t अंक_m_per_thread0   = round_down((माप_प्रकार)(m_roundup_mr / अंक_threads), (माप_प्रकार)mr);
                    स्थिरांक int64_t अंक_m_per_thread_N_1 = m - (अंक_threads - 1) * अंक_m_per_thread0;

                    स्थिरांक int64_t m_start = ith * अंक_m_per_thread0;
                    स्थिरांक int64_t m_मोजणी = (ith == अंक_threads - 1) ? अंक_m_per_thread_N_1 : अंक_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    स्थिरांक माप_प्रकार base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    स्थिरांक माप_प्रकार next_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    स्थिरांक माप_प्रकार row_stride_bytes = (next_block_off - base_packed_off) / (माप_प्रकार)mr;

                    int64_t remaining = m_मोजणी;
                    int64_t cur       = m_start;

                    पर्यंत (remaining > 0) {
                        स्थिरांक int64_t row_in_group = cur;
                        स्थिरांक int64_t avail        = m_group - row_in_group;
                        स्थिरांक int64_t take         = std::min(avail, remaining);

                        स्थिरांक uint8_t * lhs_batch_base = static_cast<स्थिरांक uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        स्थिरांक शून्य * src_ptr = lhs_batch_base + (माप_प्रकार)row_in_group * lhs_stride;
                        स्थिरांक माप_प्रकार dst_off = base_packed_off + (माप_प्रकार)(cur - m_start) * row_stride_bytes;
                        शून्य * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            जर (ith == 0) {
                स_रखो(bias, 0, (माप_प्रकार)n * आकार(दशांश));
                transpose_f32kxn_f16nxk((माप_प्रकार)n, (माप_प्रकार)k,
                                        reinterpret_cast<दशांश *>(rhs_kxn),
                                        reinterpret_cast<स्थिरांक uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * आकार(दशांश),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                स्थिरांक int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t अंक_threads_n = KAI_MIN(n / n_step, nth);
                जर (अंक_threads_n <= 0) {
                    अंक_threads_n = 1;
                }

                जर (ith < अंक_threads_n) {
                    स्थिरांक int64_t अंक_n_per_thread0   = round_down((माप_प्रकार)(n / अंक_threads_n), (माप_प्रकार)n_step);
                    स्थिरांक int64_t अंक_n_per_thread_N_1 = n - (अंक_threads_n - 1) * अंक_n_per_thread0;

                    स्थिरांक int64_t n_start      = ith * अंक_n_per_thread0;
                    स्थिरांक int64_t n_to_process = (ith == अंक_threads_n - 1) ? अंक_n_per_thread_N_1 : अंक_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    स्थिरांक माप_प्रकार lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    स्थिरांक माप_प्रकार rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    स्थिरांक माप_प्रकार dst_offset         = kernel->get_dst_offset((माप_प्रकार)0, (माप_प्रकार)n_start, dst_stride);

                    स्थिरांक शून्य * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    स्थिरांक शून्य * rhs_ptr = rhs_packed + rhs_packed_offset;
                    दशांश * dst_ptr      = reinterpret_cast<दशांश *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, आकार(दशांश), -भग्न_उच्च, भग्न_उच्च);
                }
            }

            जर (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        परत सत्य;
    }

    तार्किक compute_forward_qx(संरचना ggml_compute_params * params, संरचना ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        स्थिरांक ggml_tensor * src0 = dst->src[0];
        स्थिरांक ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_द्विमान_OP_LOCALS

        स्थिरांक kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->data);
        स्थिरांक तार्किक has_header = kleidiai_is_weight_header_valid(header);
        स्थिरांक तार्किक is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        स्थिरांक पूर्णांक slot_total = kleidiai_collect_kernel_chain(dst, kernel_chain);

        स्वतः weight_for_slot = [&](पूर्णांक slot_index, माप_प्रकार & size_out) -> स्थिरांक uint8_t * {
            जर (slot_index < 0 || slot_index >= slot_total) {
                परत nullptr;
            }
            जर (has_header) {
                जर (slot_index < header->slot_मोजणी) {
                    size_out = static_cast<माप_प्रकार>(header->sizes[slot_index]);
                    परत kleidiai_weight_slot_ptr(header, slot_index);
                }
                परत nullptr;
            }
            जर (slot_index == 0) {
                size_out = ggml_nbytes(src0);
                परत static_cast<स्थिरांक uint8_t *>(src0->data);
            }
            परत nullptr;
        };

        संरचना runtime_slot {
            पूर्णांक slot_index;
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
            पूर्णांक assigned_threads;
            पूर्णांक thread_begin;
            पूर्णांक thread_end;
            स्थिरांक uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        पूर्णांक runtime_मोजणी = 0;

        साठी (पूर्णांक slot = 0; slot < slot_total && runtime_मोजणी < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            जर (!kinfo || !linfo || !linfo->packed_size_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                सुरूठेवा;
            }

            माप_प्रकार rhs_size = 0;
            स्थिरांक uint8_t * rhs_ptr = weight_for_slot(slot, rhs_size);
            जर (!rhs_ptr || rhs_size == 0) {
                सुरूठेवा;
            }

            स्थिरांक kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

            runtime[runtime_मोजणी] = {
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
            ++runtime_मोजणी;
        }

        जर (runtime_मोजणी == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            परत असत्य;
        }

        स्थिरांक पूर्णांक nth_total = params->nth > 0 ? params->nth : 1;
        स्थिरांक पूर्णांक ith_total = params->ith;

        पूर्णांक sme_slot = -1;
        पूर्णांक non_sme_slot = -1;
        साठी (पूर्णांक i = 0; i < runtime_मोजणी; ++i) {
            जर (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                थांबा;
            }
        }

        साठी (पूर्णांक i = 0; i < runtime_मोजणी; ++i) {
            जर (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                थांबा;
            }
        }

        स्थिरांक पूर्णांक sme_cap_limit = ctx.sme_thread_cap;
        स्थिरांक तार्किक use_hybrid = sme_cap_limit > 0 &&
                                 runtime_मोजणी > 1 &&
                                 nth_total > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        माप_प्रकार min_cols_per_thread = 0;
        जर (runtime_मोजणी > 0 && nth_total > 0) {
            min_cols_per_thread = (माप_प्रकार) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
        }
        स्थिरांक तार्किक too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        स्थिरांक तार्किक hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        जर (!hybrid_enabled) {
            पूर्णांक chosen_slot = 0;
            जर (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } अन्यथा जर (runtime_मोजणी > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            जर (chosen_slot != 0 && chosen_slot < runtime_मोजणी) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_मोजणी = runtime_मोजणी > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            जर (runtime_मोजणी > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        स्थिरांक तार्किक sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        पूर्णांक sme_cap = sme_single_only ? nth_total : kleidiai_sme_thread_cap();
        जर (sme_cap < 0) {
            sme_cap = nth_total;
        }
        sme_cap = std::min(sme_cap, nth_total);

        पूर्णांक threads_remaining = nth_total;
        जर (sme_slot != -1) {
            पूर्णांक sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        पूर्णांक fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        पूर्णांक fallback_मोजणी = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        साठी (पूर्णांक i = 0; i < runtime_मोजणी; ++i) {
            जर (i == sme_slot) {
                सुरूठेवा;
            }
            fallback_indices[fallback_मोजणी++] = i;
        }

        साठी (पूर्णांक fi = 0; fi < fallback_मोजणी; ++fi) {
            जर (threads_remaining <= 0) {
                थांबा;
            }
            स्थिरांक पूर्णांक slot_index = fallback_indices[fi];
            स्थिरांक पूर्णांक slots_डावा = fallback_मोजणी - fi;
            पूर्णांक share = (threads_remaining + slots_डावा - 1) / slots_डावा;
            share     = std::min(share, threads_remaining);
            runtime[slot_index].assigned_threads = share;
            threads_remaining -= share;
        }

        पूर्णांक thread_cursor = 0;
        साठी (पूर्णांक i = 0; i < runtime_मोजणी; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        स्थिरांक पूर्णांक active_threads = thread_cursor;

        पूर्णांक local_slot = -1;
        पूर्णांक local_ith  = 0;
        साठी (पूर्णांक i = 0; i < runtime_मोजणी; ++i) {
            जर (ith_total >= runtime[i].thread_begin && ith_total < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_total - runtime[i].thread_begin;
                थांबा;
            }
        }
        स्थिरांक माप_प्रकार k = ne00;
        स्थिरांक माप_प्रकार m = ne11;
        स्थिरांक माप_प्रकार n = ne01;

        माप_प्रकार cursor = 0;
        साठी (पूर्णांक i = 0; i < runtime_मोजणी; ++i) {
            runtime[i].lhs_packed_size = runtime[i].lhs_info->packed_size_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_size;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        माप_प्रकार common_step = 1;
        साठी (पूर्णांक i = 0; i < runtime_मोजणी; ++i) {
            जर (runtime[i].assigned_threads == 0) {
                सुरूठेवा;
            }
            माप_प्रकार next_step = 0;
            जर (!lcm_size(common_step, runtime[i].n_step ? runtime[i].n_step : 1, next_step)) {
                परत असत्य;
            }
            common_step = next_step;
        }
        GGML_ASSERT(common_step > 0);

        स्थिरांक तार्किक disable_chunking = ggml_is_numa();
        स्थिरांक माप_प्रकार chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        स्थिरांक माप_प्रकार chunk_भाजक = (active_threads == 1 || disable_chunking) ? (माप_प्रकार) active_threads : (माप_प्रकार) active_threads * chunk_multiplier;
        माप_प्रकार chunk_cols = align_up(std::max<माप_प्रकार>(1, ceil_div_size(n, chunk_भाजक)), common_step);
        जर (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        स्थिरांक माप_प्रकार nchunk_size = std::max<माप_प्रकार>(1, ceil_div_size(n, chunk_cols));
        GGML_ASSERT(nchunk_size <= (माप_प्रकार)पूर्णांक_उच्च);
        स्थिरांक पूर्णांक nchunk = (पूर्णांक)nchunk_size;
        स्थिरांक माप_प्रकार dst_stride = dst->nb[1];

        स्वतः run_chunk = [&](runtime_slot & slot, माप_प्रकार global_start, माप_प्रकार cols, uint8_t * dst_batch_base) {
            स्थिरांक माप_प्रकार rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            स्थिरांक माप_प्रकार dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            स्थिरांक uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            स्थिरांक uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            दशांश * dst_ptr         = reinterpret_cast<दशांश *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       आकार(दशांश),
                                       -भग्न_उच्च,
                                       भग्न_उच्च);
        };

        साठी (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            स्थिरांक uint8_t * lhs_batch_base = static_cast<स्थिरांक uint8_t *>(src1->data) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            जर (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                स्थिरांक int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                स्थिरांक int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                जर (local_ith < use_threads) {
                    स्थिरांक int64_t अंक_m_per_thread0   = round_down((माप_प्रकार)(m_roundup_mr / use_threads), slot.mr);
                    स्थिरांक int64_t अंक_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * अंक_m_per_thread0;

                    स्थिरांक int64_t m_start = (int64_t)local_ith * अंक_m_per_thread0;
                    स्थिरांक int64_t m_मोजणी = (local_ith == use_threads - 1) ? अंक_m_per_thread_N_1 : अंक_m_per_thread0;

                    स्थिरांक माप_प्रकार base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    स्थिरांक माप_प्रकार next_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    स्थिरांक माप_प्रकार row_stride_bytes = slot.mr ? (next_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_मोजणी;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    पर्यंत (remaining > 0) {
                        स्थिरांक int64_t row_in_group = cur;
                        स्थिरांक int64_t avail        = (int64_t)m - row_in_group;
                        स्थिरांक int64_t take         = std::min(avail, remaining);

                        स्थिरांक माप_प्रकार src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        स्थिरांक शून्य * src_ptr = lhs_batch_base + src_off;
                        स्थिरांक माप_प्रकार dst_off = base_packed_off + (माप_प्रकार)(cur - m_start) * row_stride_bytes;
                        शून्य * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            जर (ith_total == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            जर (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                पूर्णांक current_chunk = ith_total;
                पर्यंत (current_chunk < nchunk) {
                    स्थिरांक माप_प्रकार global_start = (माप_प्रकार)current_chunk * chunk_cols;
                    जर (global_start >= n) {
                        थांबा;
                    }

                    स्थिरांक माप_प्रकार cols = std::min(chunk_cols, n - global_start);
                    जर (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            जर (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        परत सत्य;
    }

सार्वजनिक:
    पूर्णांक repack(संरचना ggml_tensor * tensor, स्थिरांक शून्य * data, माप_प्रकार data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        स्थिरांक माप_प्रकार n = tensor->ne[1];
        स्थिरांक माप_प्रकार k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->data);
        जर (!header) {
            परत -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_मोजणी = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->data);
        माप_प्रकार cursor = आकार(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        स्थिरांक तार्किक want_q4k = tensor->type == GGML_TYPE_Q4_K;
        स्थिरांक तार्किक want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        स्थिरांक तार्किक want_f32 = tensor->type == GGML_TYPE_F32;
        स्थिरांक पूर्णांक slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        स्थिरांक तार्किक allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<दशांश>  scales;
        std::vector<दशांश>  bias;
        std::vector<दशांश> row_f32;

        जर (want_q8 && slot_total > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            स्थिरांक माप_प्रकार row_stride = tensor->nb[1];
            स्थिरांक माप_प्रकार k_blocks   = (k + QK8_0 - 1) / QK8_0;

            जर (tensor->type == GGML_TYPE_Q6_K) {
                जर (k % QK_K != 0 || k % QK8_0 != 0) {
                    परत -1;
                }
                row_f32.resize(k);
            }

            साठी (माप_प्रकार row = 0; row < n; ++row) {
                स्थिरांक uint8_t * row_data = static_cast<स्थिरांक uint8_t *>(data) + row * row_stride;

                जर (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<स्थिरांक block_q6_K *>(row_data), row_f32.data(), k);
                }

                दशांश max_abs = 0.0f;
                जर (tensor->type == GGML_TYPE_Q6_K) {
                    साठी (माप_प्रकार linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::भ_असल(row_f32[linear_idx]));
                    }
                } अन्यथा {
                    स्थिरांक block_q8_0 * row_blocks = reinterpret_cast<स्थिरांक block_q8_0 *>(row_data);
                    साठी (माप_प्रकार block = 0; block < k_blocks; ++block) {
                        स्थिरांक block_q8_0 & blk = row_blocks[block];
                        स्थिरांक दशांश d = GGML_FP16_TO_FP32(blk.d);
                        साठी (माप_प्रकार l = 0; l < QK8_0; ++l) {
                            स्थिरांक माप_प्रकार linear_idx = block * QK8_0 + l;
                            जर (linear_idx >= k) {
                                थांबा;
                            }
                            स्थिरांक दशांश value = d * static_cast<दशांश>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::भ_असल(value));
                        }
                    }
                }

                दशांश scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                स्थिरांक दशांश inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                जर (tensor->type == GGML_TYPE_Q6_K) {
                    साठी (माप_प्रकार linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } अन्यथा {
                    स्थिरांक block_q8_0 * row_blocks = reinterpret_cast<स्थिरांक block_q8_0 *>(row_data);
                    साठी (माप_प्रकार block = 0; block < k_blocks; ++block) {
                        स्थिरांक block_q8_0 & blk = row_blocks[block];
                        स्थिरांक दशांश d = GGML_FP16_TO_FP32(blk.d);
                        साठी (माप_प्रकार l = 0; l < QK8_0; ++l) {
                            स्थिरांक माप_प्रकार linear_idx = block * QK8_0 + l;
                            जर (linear_idx >= k) {
                                थांबा;
                            }
                            स्थिरांक दशांश value = d * static_cast<दशांश>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        जर (want_f32 && slot_total > 0) {
            bias.resize(n, 0.0f);
        }

        साठी (पूर्णांक slot = 0; slot < slot_total && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            जर (!allow_fallback && slot > 0) {
                थांबा;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            जर (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_size_ex || !kernel) {
                सुरूठेवा;
            }

            स्थिरांक माप_प्रकार nr = kernel->get_nr();
            स्थिरांक माप_प्रकार kr = kernel->get_kr();
            स्थिरांक माप_प्रकार sr = kernel->get_sr();
            स्थिरांक ggml_type rhs_type = kernels->rhs_type;
            जर (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            स्थिरांक माप_प्रकार block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            जर (block_len == SIZE_MAX) {
                सुरूठेवा;
            }

            स्थिरांक माप_प्रकार packed_size = rhs_info->packed_size_ex(n, k, nr, kr, block_len);
            स्थिरांक माप_प्रकार aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            जर (rhs_type == GGML_TYPE_Q4_0) {
                संरचना kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<स्थिरांक uint8_t *>(data), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } अन्यथा जर (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       data, nullptr, nullptr, dst_ptr, 0, nullptr);
            } अन्यथा जर (rhs_type == GGML_TYPE_Q8_0) {
                संरचना kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.data(), nullptr, scales.data(),
                                       dst_ptr, 0, &params);
            } अन्यथा जर (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       data, bias.data(), nullptr,
                                       dst_ptr, 0, nullptr);
            } अन्यथा {
                सुरूठेवा;
            }

            header->offsets[header->slot_मोजणी] = aligned_cursor;
            header->sizes[header->slot_मोजणी]   = packed_size;
            ++header->slot_मोजणी;

            cursor = aligned_cursor + packed_size;
        }

        जर (header->slot_मोजणी == 0) {
            header->magic   = 0;
            header->version = 0;
            स_नकल(tensor->data, data, data_size);
        }

        परत 0;
    }
};

स्थिर ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, संरचना ggml_tensor *) {
    स्थिर tensor_traits traits;
    परत &traits;
}
}  // namespace ggml::cpu::kleidiai

स्थिर गणना ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor) {
    tensor->extra = (शून्य *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    परत GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor,
                                                       स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    स्वतः tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    स्वतः OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

स्थिर स्थिरांक अक्षर * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    परत "CPU_KLEIDIAI";
}

स्थिर ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, माप_प्रकार size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    जर (buffer == nullptr) {
        परत nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    परत buffer;
}

स्थिर माप_प्रकार ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    परत TENSOR_ALIGNMENT;
}

स्थिर माप_प्रकार ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, स्थिरांक संरचना ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    जर (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        परत ggml_nbytes(tensor);
    }

    स्थिरांक माप_प्रकार n = tensor->ne[1];
    स्थिरांक माप_प्रकार k = tensor->ne[0];

    माप_प्रकार cursor = आकार(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    स्थिरांक तार्किक want_q4k = tensor->type == GGML_TYPE_Q4_K;
    स्थिरांक तार्किक want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    स्थिरांक तार्किक want_f32 = tensor->type == GGML_TYPE_F32;
    स्थिरांक पूर्णांक slot_total = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    स्थिरांक तार्किक allow_fallback = kleidiai_pack_fallback_allowed();

    माप_प्रकार slot_मोजणी = 0;
    साठी (पूर्णांक slot = 0; slot < slot_total; ++slot) {
        जर (!allow_fallback && slot > 0) {
            थांबा;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        जर (!kernels) {
            सुरूठेवा;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        जर (!kernel || !rhs_info || !rhs_info->packed_size_ex) {
            सुरूठेवा;
        }

        स्थिरांक ggml_type rhs_type = kernels->rhs_type;
        स्थिरांक माप_प्रकार block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        जर (block_len == SIZE_MAX) {
            सुरूठेवा;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_size_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_मोजणी;
    }

    जर (slot_मोजणी == 0) {
        परत ggml_nbytes(tensor);
    }

    परत std::max(cursor, ggml_nbytes(tensor));
}

नामाकाश ggml::cpu::kleidiai {
वर्ग extra_buffer_type : ggml::cpu::extra_buffer_type {
    तार्किक supports_op(ggml_backend_dev_t, स्थिरांक संरचना ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        जर (op->op != GGML_OP_MUL_MAT) {
            परत असत्य;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        स्थिरांक पूर्णांक slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
        स्थिरांक तार्किक src0_is_kleidiai =
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft->context == हे &&
            slot_total > 0;

        जर ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            जर (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                परत असत्य;
            }
            जर (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                परत असत्य;
            }
            जर (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                परत असत्य;
            }
            जर (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                परत असत्य;
            }
            जर (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                परत असत्य;
            }
            जर (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                परत असत्य;
            }

            जर (op->src[0]->type == GGML_TYPE_Q4_K) {
                जर (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    परत असत्य;
                }
                परत ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == आकार(दशांश) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * आकार(दशांश));
            }

            जर (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                जर (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    परत सत्य;
                }
                परत असत्य;
            }

            जर (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                परत असत्य;
            }

            परत सत्य;
        }

        परत असत्य;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(स्थिरांक संरचना ggml_tensor * op) override {
        जर (op->op == GGML_OP_MUL_MAT) {
            जर (op->src[0]->buffer && op->src[0]->buffer->buft->context == हे) {
                परत (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } अन्यथा {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                जर (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    स्थिर std::atomic<तार्किक> warned(असत्य);
                    जर (!warned.exchange(सत्य)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                जर (op->src[0]->type != GGML_TYPE_F16) {
                    परत nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                स्थिरांक पूर्णांक slot_total = kleidiai_collect_kernel_chain(op, kernel_chain);
                जर (slot_total > 0 && op->src[1]->ne[1] > 1) {
                    जर ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        परत nullptr;
                    }
                    परत ggml::cpu::kleidiai::get_tensor_traits(रिक्त, रिक्त);
                }
            }
        }
        परत nullptr;
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

    परत &ggml_backend_cpu_buffer_type_kleidiai;
}
