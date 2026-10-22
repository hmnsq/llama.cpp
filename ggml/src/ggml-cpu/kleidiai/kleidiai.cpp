// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#समावेश <arm_neon.h>
#समावेश <cassert>
#समावेश <cstdio>
#समावेश <cstdlib>
#समावेश <atomic>
#समावेश <cfloat>
#समावेश <cctype>
#समावेश <algorithm>
#समावेश <cmath>
#समावेश <stdexcept>
#समावेश <stdint.h>
#समावेश <string.h>
#समावेश <string>
#समावेश <vector>
#समावेश <array>
#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <fstream>
#समावेश <map>
#समावेश <iostream>
#समावेश <climits>
#समावेश <charconv>
#समावेश <system_error>
#अगर defined(__linux__)
#समावेश <asm/hwcap.h>
#समावेश <dirent.h>
#समावेश <sys/auxv.h>
#समावेश <sys/types.h>
#समावेश <sys/stat.h>
#समावेश <unistd.h>
#या_अगर defined(__APPLE__)
#समावेश <sys/sysctl.h>
#समावेश <sys/types.h>
#पूर्ण_अगर

#समावेश "kleidiai.h"

#समावेश "ggml-cpu.h"
#समावेश "ggml-cpu-impl.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-feats.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-quants.h"
#समावेश "ggml-threading.h"
#समावेश "traits.h"

#समावेश "kernels.h"

#समावेश "kai/kai_common.h"

#घोषणा GGML_COMMON_DECL_CPP
#समावेश "ggml-common.h"

अटल constexpr पूर्णांक      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
अटल constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
अटल constexpr uint16_t GGML_KLEIDIAI_PACK_VERSION     = 1;
अटल constexpr माप_प्रकार   GGML_KLEIDIAI_PACK_ALIGN       = 64;
अटल constexpr माप_प्रकार   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");

काष्ठा ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels_q4;
    ggml_kleidiai_kernels * kernels_q4k;
    ggml_kleidiai_kernels * kernels_q8;
    ggml_kleidiai_kernels * kernels_f32;
    पूर्णांक sme_thread_cap; // <= 0 means "SME disabled/unknown"
    पूर्णांक thread_hint;    // <= 0 means "no hint"
    पूर्णांक chunk_multiplier;
} अटल ctx = { CPU_FEATURE_NONE, nullptr, nullptr, nullptr, nullptr, 0, -1, 4 };

अटल अंतरभूत bool is_sme_family(cpu_feature f) {
    वापस (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

अटल स्थिर अक्षर* cpu_feature_to_string(cpu_feature f) {
    अगर (f == CPU_FEATURE_NONE) {
        वापस "NONE";
    } अन्यथा अगर ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) {
        वापस "SME2";
    } अन्यथा अगर ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) {
        वापस "SME";
    } अन्यथा अगर ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) {
        वापस "SVE";
    }
    अन्यथा अगर ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) {
        वापस "I8MM";
    } अन्यथा अगर ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) {
        वापस "DOTPROD";
    }
    अन्यथा {
        वापस "UNKNOWN";
    }
}

#अगर defined(__linux__) && defined(__aarch64__)
अटल bool parse_cpu_dir_name(स्थिर अक्षर* name, माप_प्रकार* cpu) {
    अगर (म_भेदन(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        वापस false;
    }

    स्थिर अक्षर* first = name + 3;
    स्थिर अक्षर* last = name + म_माप(name);

    माप_प्रकार value = 0;
    स्थिर स्वतः [end, ec] = std::from_chars(first, last, value, 10);

    अगर (ec != std::errc{} || end != last) {
        वापस false;
    }

    *cpu = value;
    वापस true;
}

अटल std::vector<माप_प्रकार> detect_cpu_ids() {
    std::vector<माप_प्रकार> cpus;

    सूची * dir = सूची_खोलो("/sys/devices/system/cpu");
    अगर (dir == nullptr) {
        वापस cpus;
    }

    जबतक (dirent * entry = सूची_पढ़ो(dir)) {
        माप_प्रकार cpu = 0;
        अगर (parse_cpu_dir_name(entry->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    बंद_सूची(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    वापस cpus;
}
#पूर्ण_अगर

#अगर defined(__APPLE__) && defined(__aarch64__)
अटल bool apple_sme_counted_perf_level(std::string name) {
    क्रम (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (अक्षर) std::छोटे((अचिन्हित अक्षर) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    वापस name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#पूर्ण_अगर

अटल व्योम add_smcus_from_smidr(uint64_t smidr, माप_प्रकार & अंक_private, std::map<uint32_t, माप_प्रकार> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    स्थिर uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    स्थिर uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    स्थिर माप_प्रकार shared_गिनती = nsmc == 0xF ? 1 : (माप_प्रकार)nsmc + 1;
    स्थिर uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    स्थिर uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    स्थिर uint32_t id = (affinity2 << 12) | affinity;

    अगर (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    चयन (sh) {
        हाल 2: // private SMCU
            ++अंक_private;
            अवरोध;
        हाल 3: // shared SMCU
            अगर (shared_counts[id] < shared_गिनती) {
                shared_counts[id] = shared_गिनती;
            }
            अवरोध;
        हाल 0:
            अगर (id == 0) {
                ++अंक_private;
            } अन्यथा अगर (shared_counts[id] < shared_गिनती) {
                shared_counts[id] = shared_गिनती;
            }
            अवरोध;
        शेष:
            अवरोध;
    }
}

अटल माप_प्रकार detect_अंक_smcus() {
    स्थिर स्वतः runtime_feat = ggml_feats_get_arch64_runtime();
    अगर (!runtime_feat.has_sme) {
        वापस 0;
    }

#अगर defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    माप_प्रकार अंक_private = 0;
    std::map<uint32_t, माप_प्रकार> shared_counts;

    स्थिर std::vector<माप_प्रकार> cpus = detect_cpu_ids();
    क्रम (स्थिर माप_प्रकार cpu : cpus) {
        स्थिर std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
            "/regs/identification/smidr_el1";

        std::अख_प्रवाह file(path);
        अगर (!file.is_open()) {
            जारी;
        }

        uint64_t smidr = 0;
        अगर (!(file >> std::षष्ठ >> smidr)) {
            जारी;
        }

        add_smcus_from_smidr(smidr, अंक_private, shared_counts);
    }

    माप_प्रकार योग = अंक_private;
    क्रम (स्थिर स्वतः & entry : shared_counts) {
        योग += entry.second;
    }
    वापस योग;

#या_अगर defined(__APPLE__) && defined(__aarch64__)
    पूर्णांक perf_levels = 0;
    माप_प्रकार size = माप(perf_levels);
    अगर (sysctlbyname("hw.nperflevels", &perf_levels, &size, nullptr, 0) != 0 ||
        size != माप(perf_levels) || perf_levels <= 0) {
        वापस 0;
    }

    माप_प्रकार units = 0;
    क्रम (पूर्णांक i = 0; i < perf_levels; ++i) {
        अक्षर कुंजी[64] = {};
        पूर्णांक physical_cpus = 0;
        पूर्णांक cpus_per_l2 = 0;

        snprintf(कुंजी, माप(कुंजी), "hw.perflevel%d.physicalcpu", i);
        size = माप(physical_cpus);
        अगर (sysctlbyname(कुंजी, &physical_cpus, &size, nullptr, 0) != 0 ||
            size != माप(physical_cpus) || physical_cpus <= 0) {
            जारी;
        }

        snprintf(कुंजी, माप(कुंजी), "hw.perflevel%d.cpusperl2", i);
        size = माप(cpus_per_l2);
        अगर (sysctlbyname(कुंजी, &cpus_per_l2, &size, nullptr, 0) != 0 ||
            size != माप(cpus_per_l2) || cpus_per_l2 <= 0) {
            जारी;
        }

        snprintf(कुंजी, माप(कुंजी), "hw.perflevel%d.name", i);
        size = 0;
        अगर (sysctlbyname(कुंजी, nullptr, &size, nullptr, 0) != 0 || size == 0) {
            जारी;
        }

        std::string name(size, '\0');
        अगर (sysctlbyname(कुंजी, &name[0], &size, nullptr, 0) != 0) {
            जारी;
        }
        name.resize(size);
        जबतक (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        अगर (apple_sme_counted_perf_level(name)) {
            units += (माप_प्रकार) ((physical_cpus + cpus_per_l2 - 1) / cpus_per_l2);
        }
    }

    वापस units;

#या_अगर defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    // No verified Windows arm64 SMCU detection path yet. Return unknown and use
    // GGML_KLEIDIAI_SME=N as a diagnostics/debug override for SME thread cap
    // calibration until a detection mechanism is verified on real hardware.
    वापस 0;

#अन्यथा
    वापस 0;
#पूर्ण_अगर
}

अटल पूर्णांक parse_uint_env(स्थिर अक्षर *s, स्थिर अक्षर *name, bool *ok) {
    अगर (!s) { *ok = false; वापस 0; }
    अक्षर *end = nullptr;
    दीर्घ v = म_से_दीर्घ(s, &end, 10);
    अगर (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = false;
        वापस 0;
    }
    अगर (v < 0 || v > पूर्णांक_उच्च) {
        GGML_LOG_WARN("kleidiai: out-of-range %s='%s'\n", name, s);
        *ok = false;
        वापस 0;
    }
    *ok = true;
    वापस (पूर्णांक)v;
}

अटल व्योम init_kleidiai_context(व्योम) {
    ggml_critical_section_start();
    अटल bool initialized = false;

    अगर (!initialized) {
        initialized = true;

        // Optional diagnostics/debug overrides; production defaults come from runtime detection.
        स्थिर अक्षर *env_sme         = दो_पर्या("GGML_KLEIDIAI_SME");
        स्थिर अक्षर *env_threads     = दो_पर्या("GGML_TOTAL_THREADS");
        स्थिर अक्षर *env_chunk_mult  = दो_पर्या("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        स्थिर स्वतः runtime_feat = ggml_feats_get_arch64_runtime();

        माप_प्रकार detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        अगर (env_threads) {
            bool ok = false;
            पूर्णांक hint = parse_uint_env(env_threads, "GGML_TOTAL_THREADS", &ok);
            अगर (ok && hint > 0) {
                ctx.thread_hint = hint;
            }
        }

        अगर (env_chunk_mult) {
            bool ok = false;
            पूर्णांक multiplier = parse_uint_env(env_chunk_mult, "GGML_KLEIDIAI_CHUNK_MULTIPLIER", &ok);
            अगर (ok && multiplier > 0) {
                ctx.chunk_multiplier = multiplier;
            }
        }

        पूर्णांक sme_cores = 0;
        bool sme_env_ok = false;
        bool sme_env_set = (env_sme != nullptr);

        स्थिर bool has_supported_sme_family = runtime_feat.has_sme;
        bool sme_cap_detected = false;

        अगर (has_supported_sme_family) {
            detected_smcus = detect_अंक_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (पूर्णांक)detected_smcus : 1;

            अगर (!sme_env_set && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        अगर (sme_env_set) {
            bool ok = false;
            पूर्णांक v = parse_uint_env(env_sme, "GGML_KLEIDIAI_SME", &ok);
            sme_env_ok = ok;

            अगर (ok) {
                अगर (has_supported_sme_family) {
                    sme_cores = v;
                } अन्यथा {
                    अगर (v > 0) {
                        GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME=%d but SME is not supported on this CPU; disabling SME-family kernels\n", v);
                    }
                    sme_cores = 0;
                }
            } अन्यथा {
                GGML_LOG_WARN("kleidiai: GGML_KLEIDIAI_SME set but parsing failed; using automatic SME thread cap\n");
            }
        }

        अगर (sme_cores > 0 && has_supported_sme_family) {
            ctx.features |= CPU_FEATURE_SME;
            अगर (runtime_feat.has_sme2) {
                ctx.features |= CPU_FEATURE_SME2;
            }
        }

        // Kernel selection
        ctx.kernels_q4  = ggml_kleidiai_select_kernels_q4_0(ctx.features);
        ctx.kernels_q4k = ggml_kleidiai_select_kernels_q4_k(ctx.features);
        ctx.kernels_q8  = ggml_kleidiai_select_kernels_q8_0(ctx.features);
        ctx.kernels_f32 = ggml_kleidiai_select_kernels_f32(ctx.features);

        अगर (!ctx.kernels_q4) {
            GGML_LOG_INFO("kleidiai: no compatible q4 kernels found for CPU features mask %d\n", (पूर्णांक)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4->required_cpu));
        }

        अगर (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (पूर्णांक)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q8->required_cpu));
        }

        अगर (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (पूर्णांक) ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_to_string(ctx.kernels_q4k->required_cpu));
        }

        अगर (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (पूर्णांक)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_to_string(ctx.kernels_f32->required_cpu));
        }

        स्थिर bool has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        अगर (has_selected_sme_family_kernel) {
            अगर (sme_env_set && sme_env_ok && sme_cores > 0) {
                GGML_LOG_INFO("kleidiai: SME enabled (GGML_KLEIDIAI_SME=%d debug override)\n", sme_cores);
            } अन्यथा अगर (sme_cap_detected) {
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

अटल अंतरभूत पूर्णांक kleidiai_sme_thread_cap() {
    वापस ctx.sme_thread_cap;
}

अटल अंतरभूत माप_प्रकार align_up(माप_प्रकार value, माप_प्रकार alignment) {
    अगर (alignment == 0) {
        वापस value;
    }
    स्थिर माप_प्रकार remainder = value % alignment;
    वापस remainder == 0 ? value : value + (alignment - remainder);
}

अटल अंतरभूत माप_प्रकार gcd_आकार(माप_प्रकार a, माप_प्रकार b) {
    जबतक (b != 0) {
        स्थिर माप_प्रकार t = a % b;
        a = b;
        b = t;
    }
    वापस a;
}

अटल अंतरभूत bool lcm_आकार(माप_प्रकार a, माप_प्रकार b, माप_प्रकार & परिणाम) {
    अगर (a == 0 || b == 0) {
        परिणाम = 0;
        वापस false;
    }
    स्थिर माप_प्रकार g = gcd_आकार(a, b);
    स्थिर माप_प्रकार q = a / g;
    अगर (q > SIZE_MAX / b) {
        वापस false;
    }
    परिणाम = q * b;
    वापस true;
}

अटल अंतरभूत माप_प्रकार ceil_div_आकार(माप_प्रकार a, माप_प्रकार b) {
    वापस b == 0 ? 0 : (a + b - 1) / b;
}

अटल अंतरभूत माप_प्रकार kleidiai_chunk_cols(माप_प्रकार n, पूर्णांक nth_योग, bool disable_chunking, माप_प्रकार n_step) {
    स्थिर माप_प्रकार multiplier = (nth_योग == 1 || disable_chunking) ? 1 : std::max<माप_प्रकार>(1, (माप_प्रकार) ctx.chunk_multiplier);
    स्थिर माप_प्रकार भाजक = std::max<माप_प्रकार>(1, (माप_प्रकार) nth_योग * multiplier);
    स्थिर माप_प्रकार chunk_cols = align_up(std::max<माप_प्रकार>(1, ceil_div_आकार(n, भाजक)), n_step);
    वापस chunk_cols ? chunk_cols : n_step;
}

काष्ठा kleidiai_block_args {
    माप_प्रकार lhs_bl;
    माप_प्रकार rhs_bl;
    माप_प्रकार pack_bl;
};

अटल अंतरभूत kleidiai_block_args kleidiai_get_block_args(ggml_type rhs_type) {
    चयन (rhs_type) {
        हाल GGML_TYPE_Q4_0:
            वापस { QK4_0, QK4_0, QK4_0 };
        हाल GGML_TYPE_Q4_K:
            वापस { GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL, GGML_KLEIDIAI_Q4_K_BL };
        हाल GGML_TYPE_Q8_0:
            वापस { 0, 0, QK8_0 };
        शेष:
            वापस { 0, 0, 0 };
    }
}

अटल अंतरभूत bool kleidiai_pack_fallback_allowed() {
    अगर (ctx.sme_thread_cap <= 0) {
        वापस false;
    }
    अगर (ctx.thread_hint <= 0) {
        वापस true;
    }
    वापस ctx.thread_hint > ctx.sme_thread_cap;
}

काष्ठा kleidiai_weight_header {
    uint32_t magic;
    uint16_t version;
    uint16_t slot_गिनती;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

अटल अंतरभूत kleidiai_weight_header * kleidiai_weight_header_from_ptr(व्योम * तथ्य) {
    वापस reinterpret_cast<kleidiai_weight_header *>(तथ्य);
}

अटल अंतरभूत स्थिर kleidiai_weight_header * kleidiai_weight_header_from_ptr(स्थिर व्योम * तथ्य) {
    वापस reinterpret_cast<स्थिर kleidiai_weight_header *>(तथ्य);
}

अटल अंतरभूत bool kleidiai_is_weight_header_valid(स्थिर kleidiai_weight_header * header) {
    अगर (!header) {
        वापस false;
    }
    अगर (header->magic != GGML_KLEIDIAI_PACK_MAGIC || header->version != GGML_KLEIDIAI_PACK_VERSION) {
        वापस false;
    }
    अगर (header->slot_गिनती == 0 || header->slot_गिनती > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        वापस false;
    }
    वापस true;
}

अटल अंतरभूत uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_header * header, पूर्णांक slot) {
    अगर (!kleidiai_is_weight_header_valid(header)) {
        वापस nullptr;
    }
    अगर (slot < 0 || slot >= header->slot_गिनती) {
        वापस nullptr;
    }
    वापस reinterpret_cast<uint8_t *>(header) + header->offsets[slot];
}

अटल अंतरभूत स्थिर uint8_t * kleidiai_weight_slot_ptr(स्थिर kleidiai_weight_header * header, पूर्णांक slot) {
    अगर (!kleidiai_is_weight_header_valid(header)) {
        वापस nullptr;
    }
    अगर (slot < 0 || slot >= header->slot_गिनती) {
        वापस nullptr;
    }
    वापस reinterpret_cast<स्थिर uint8_t *>(header) + header->offsets[slot];
}

अटल अंतरभूत ggml_kleidiai_kernels * kleidiai_primary_kernel_q4() {
    वापस ctx.kernels_q4;
}

अटल अंतरभूत ggml_kleidiai_kernels * kleidiai_primary_kernel_q4k() {
    वापस ctx.kernels_q4k;
}

अटल अंतरभूत ggml_kleidiai_kernels * kleidiai_primary_kernel_q8() {
    वापस ctx.kernels_q8;
}

अटल अंतरभूत ggml_kleidiai_kernels * kleidiai_primary_kernel_f32() {
    वापस ctx.kernels_f32;
}

ढाँचा <typename SelectFallback>
अटल पूर्णांक kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out,
        SelectFallback select_fallback) {
    पूर्णांक count = 0;
    अगर (!primary) {
        वापस 0;
    }
    out[count++] = primary;

    अगर (primary->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY) {
        वापस count;
    }

    अगर (is_sme_family(primary->required_cpu)) {
        स्थिर cpu_feature fallback_mask = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        अगर (fallback_mask != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_mask);
            अगर (fallback && fallback != primary &&
                fallback->rhs_info.repack_mode != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                out[count++] = fallback;
            }
        }
    }

    वापस count;
}

अटल पूर्णांक kleidiai_collect_kernel_chain(स्थिर काष्ठा ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    वापस kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { वापस ggml_kleidiai_select_kernels(mask, op); });
}

अटल पूर्णांक kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    वापस kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { वापस ggml_kleidiai_select_kernels_q4_0(mask); });
}

अटल पूर्णांक kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    वापस kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { वापस ggml_kleidiai_select_kernels_q4_k(mask); });
}

अटल पूर्णांक kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    वापस kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { वापस ggml_kleidiai_select_kernels_q8_0(mask); });
}

अटल पूर्णांक kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & out) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    वापस kleidiai_collect_kernel_chain_common(primary, ctx.features, out,
        [&](cpu_feature mask) { वापस ggml_kleidiai_select_kernels_f32(mask); });
}

अटल अंतरभूत int64_t ggml_ne(स्थिर ggml_tensor * tensor, पूर्णांक dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    वापस tensor->ne[dim];
}

namespace ggml::cpu::kleidiai {

अटल माप_प्रकार round_down(माप_प्रकार x, माप_प्रकार y) {
    वापस y == 0 ? x : x - (x % y);
}

अटल व्योम transpose_f32kxn_f16nxk(माप_प्रकार n, माप_प्रकार k, भग्न * dst, स्थिर uint16_t * src, माप_प्रकार rhs_stride) {
    माप_प्रकार src_stride = rhs_stride / माप(uint16_t);
    माप_प्रकार dst_stride = n;

    क्रम (माप_प्रकार k_idx = 0; k_idx < k; ++k_idx) {
        क्रम (माप_प्रकार n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

श्रेणी tensor_traits : खुला ggml::cpu::tensor_traits {
    bool work_आकार(पूर्णांक /* n_threads */, स्थिर काष्ठा ggml_tensor * op, माप_प्रकार & size) override {
        अगर (op->op != GGML_OP_MUL_MAT) {
            वापस false;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        स्थिर पूर्णांक slot_गिनती = kleidiai_collect_kernel_chain(op, kernel_chain);
        अगर (slot_गिनती == 0) {
            वापस false;
        }

        स्थिर bool is_gemv = op->src[1]->ne[1] == 1;
        स्थिर माप_प्रकार k = op->src[0]->ne[0];
        स्थिर माप_प्रकार n = op->src[0]->ne[1];
        स्थिर माप_प्रकार m = op->src[1]->ne[1];

        अगर (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            स्थिर kleidiai_block_args block_args = kleidiai_get_block_args(kernel_chain[0]->rhs_type);
            स्थिर माप_प्रकार qk = block_args.pack_bl;

            माप_प्रकार cursor = 0;
            bool any_slot = false;

            क्रम (पूर्णांक slot = 0; slot < slot_गिनती; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                अगर (!lhs_info || !lhs_info->packed_आकार_ex || !kernel) {
                    वापस false;
                }

                स्थिर माप_प्रकार mr = kernel->get_mr();
                स्थिर माप_प्रकार kr = kernel->get_kr();
                स्थिर माप_प्रकार sr = kernel->get_sr();

                स्थिर माप_प्रकार packed = lhs_info->packed_आकार_ex(m, k, qk, mr, kr, sr);

                cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += packed;
                any_slot = true;
            }

            अगर (!any_slot) {
                वापस false;
            }

            size = cursor;
            वापस true;
        }

        अगर (op->src[0]->type == GGML_TYPE_F32) {
            ggml_kleidiai_kernels * primary     = kernel_chain[0];
            kernel_info *           gemv_kernel = primary ? &primary->gemv : nullptr;
            अगर (is_gemv && op->src[1]->nb[0] == (int64_t) माप(भग्न) && gemv_kernel &&
                gemv_kernel->get_lhs_offset_ex && gemv_kernel->get_rhs_packed_offset_ex &&
                gemv_kernel->run_kernel_ex && gemv_kernel->get_dst_offset) {
                size = 0;
                वापस true;
            }

            माप_प्रकार cursor = 0;
            bool any_slot = false;

            क्रम (पूर्णांक slot = 0; slot < slot_गिनती; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                अगर (!lhs_info || !lhs_info->packed_आकार_ex || !kernel) {
                    वापस false;
                }

                स्थिर माप_प्रकार mr = kernel->get_mr();
                स्थिर माप_प्रकार kr = kernel->get_kr();
                स्थिर माप_प्रकार sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_आकार_ex(m, k, 0, mr, kr, sr);
                any_slot = true;
            }

            अगर (!any_slot) {
                वापस false;
            }

            size = cursor;
            वापस true;
        }

        अगर (op->src[0]->type == GGML_TYPE_F16) {
            स्थिर int64_t lhs_batch_size0 = op->src[1]->ne[2];
            स्थिर int64_t rhs_batch_size0 = op->src[0]->ne[2];
            GGML_ASSERT(rhs_batch_size0 > 0);
            स्थिर int64_t r = lhs_batch_size0 / rhs_batch_size0;

            माप_प्रकार cursor = 0;
            bool any_slot = false;

            क्रम (पूर्णांक slot = 0; slot < slot_गिनती; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                अगर (!lhs_info || !lhs_info->packed_आकार_ex || !kernels->rhs_info.packed_आकार_ex || !kernel) {
                    वापस false;
                }

                स्थिर माप_प्रकार mr = kernel->get_mr();
                स्थिर माप_प्रकार kr = kernel->get_kr();
                स्थिर माप_प्रकार sr = kernel->get_sr();

                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_आकार_ex(m * r, k, 0, mr, kr, sr);
                any_slot = true;
            }

            क्रम (पूर्णांक slot = 0; slot < slot_गिनती; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                अगर (!kernel || !kernels->rhs_info.packed_आकार_ex) {
                    वापस false;
                }
                cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_आकार_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * माप(भग्न);
            cursor  = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += n * माप(भग्न);

            अगर (!any_slot) {
                वापस false;
            }

            size = cursor;
            वापस true;
        }

        वापस false;
    }

    bool compute_forward(काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst) override {
        अगर (dst->op == GGML_OP_MUL_MAT) {
            अगर (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K) {
                वापस compute_forward_qx(params, dst);
            } अन्यथा अगर (dst->src[0]->type == GGML_TYPE_F32) {
                वापस compute_forward_f32(params, dst);
            } अन्यथा अगर (dst->src[0]->type == GGML_TYPE_F16) {
                वापस compute_forward_fp16(params, dst);
            }
        }
        वापस false;
    }

    bool compute_forward_f32(ggml_compute_params * params, काष्ठा ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);

        स्थिर ggml_tensor * src0 = dst->src[0];
        स्थिर ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        अगर (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            वापस false;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        अगर (!kernels) {
            वापस false;
        }

        स्थिर माप_प्रकार k = ne00;
        स्थिर माप_प्रकार m = ne11;
        स्थिर माप_प्रकार n = ne01;
        स्थिर bool use_gemv = m == 1 && src1->nb[0] == (int64_t) माप(भग्न) &&
                              kernels->gemv.get_lhs_offset_ex &&
                              kernels->gemv.get_rhs_packed_offset_ex &&
                              kernels->gemv.run_kernel_ex &&
                              kernels->gemv.get_dst_offset;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        अगर (!kernel || !kernel->get_lhs_offset_ex ||
            !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex || !kernel->get_dst_offset) {
            वापस false;
        }

        अगर (!use_gemv && (!lhs_info || !lhs_info->get_offset || !lhs_info->get_packed_offset_ex ||
                          !lhs_info->packed_आकार_ex || !lhs_info->pack_func_ex)) {
            वापस false;
        }

        स्थिर kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->तथ्य);
        स्थिर bool has_header = kleidiai_is_weight_header_valid(header);

        स्थिर uint8_t * rhs_base = has_header ? kleidiai_weight_slot_ptr(header, 0)
                                              : static_cast<स्थिर uint8_t *>(src0->तथ्य);
        अगर (!rhs_base) {
            वापस false;
        }

        स्थिर पूर्णांक nth = params->nth > 0 ? params->nth : 1;
        स्थिर पूर्णांक ith = params->ith;

        स्थिर माप_प्रकार mr = kernel->get_mr();
        स्थिर माप_प्रकार kr = kernel->get_kr();
        स्थिर माप_प्रकार sr = kernel->get_sr();

        स्थिर माप_प्रकार lhs_packed_आकार = use_gemv ? 0 : lhs_info->packed_आकार_ex(m, k, 0, mr, kr, sr);
        अगर (!use_gemv) {
            GGML_ASSERT(lhs_packed_आकार <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        स्थिर माप_प्रकार dst_stride = dst->nb[1];
        स्थिर माप_प्रकार n_step = kernel->get_n_step() ? kernel->get_n_step() : 1;
        स्थिर bool disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (माप_प्रकार) पूर्णांक_उच्च);

        स्थिर int64_t batch_आकार = ne12 * ne13;
        क्रम (int64_t batch_idx = 0; batch_idx < batch_आकार; ++batch_idx) {
            स्थिर int64_t i12 = batch_idx % ne12;
            स्थिर int64_t i13 = batch_idx / ne12;
            स्थिर uint8_t * lhs_batch_base = static_cast<स्थिर uint8_t *>(src1->तथ्य) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->तथ्य) + i12 * dst->nb[2] + i13 * dst->nb[3];

            अगर (!use_gemv) {
                स्थिर int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_threads = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_threads = std::max<int64_t>(1, max_threads);
                स्थिर int64_t use_threads = std::min<int64_t>(nth, max_threads);

                अगर (ith < use_threads) {
                    स्थिर int64_t अंक_m_per_thread0   = round_down((माप_प्रकार)(m_roundup_mr / use_threads), mr);
                    स्थिर int64_t अंक_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * अंक_m_per_thread0;

                    स्थिर int64_t m_start = (int64_t)ith * अंक_m_per_thread0;
                    स्थिर int64_t m_गिनती = (ith == use_threads - 1) ? अंक_m_per_thread_N_1 : अंक_m_per_thread0;

                    स्थिर माप_प्रकार base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    स्थिर माप_प्रकार अग्र_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    स्थिर माप_प्रकार row_stride_bytes = mr ? (अग्र_block_off - base_packed_off) / mr : 0;

                    int64_t remaining = m_गिनती;
                    int64_t cur       = m_start;

                    जबतक (remaining > 0) {
                        स्थिर int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        स्थिर माप_प्रकार src_off = lhs_info->get_offset(cur, src1->nb[1]);
                        स्थिर व्योम * src_ptr = lhs_batch_base + src_off;
                        स्थिर माप_प्रकार dst_off = base_packed_off + (माप_प्रकार)(cur - m_start) * row_stride_bytes;
                        व्योम * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            अगर (ith == 0) {
                ggml_threadpool_chunk_set(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            स्थिर माप_प्रकार chunk_cols = kleidiai_chunk_cols(n, nth, disable_chunking, n_step);
            GGML_ASSERT(chunk_cols <= (माप_प्रकार) पूर्णांक_उच्च);

            पूर्णांक current_col = ggml_threadpool_chunk_add(params->threadpool, (पूर्णांक) chunk_cols);
            जबतक ((माप_प्रकार) current_col < n) {
                स्थिर माप_प्रकार n_start = (माप_प्रकार) current_col;
                स्थिर माप_प्रकार n_to_process = std::min(chunk_cols, n - n_start);

                अगर (n_to_process > 0) {
                    स्थिर माप_प्रकार lhs_offset = use_gemv ? kernel->get_lhs_offset_ex(0, k, 0)
                                                       : lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    स्थिर माप_प्रकार rhs_packed_offset = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    स्थिर माप_प्रकार dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

                    स्थिर व्योम * lhs_ptr = use_gemv ? lhs_batch_base + lhs_offset
                                                    : lhs_packed + lhs_offset;
                    स्थिर व्योम * rhs_ptr = rhs_base + rhs_packed_offset;
                    भग्न * dst_ptr      = reinterpret_cast<भग्न *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          माप(भग्न),
                                          -भग्न_उच्च,
                                          भग्न_उच्च);
                }

                current_col = ggml_threadpool_chunk_add(params->threadpool, (पूर्णांक) chunk_cols);
            }

            अगर (batch_idx != batch_आकार - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        वापस true;
    }

    bool compute_forward_fp16(ggml_compute_params * params, काष्ठा ggml_tensor * dst) {
        स्थिर ggml_tensor * src0 = dst->src[0];
        स्थिर ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        अगर (!kernels) {
            वापस false;
        }

        स्थिर bool is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        अगर (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_offset_ex || !kernel->get_rhs_packed_offset_ex || !kernel->run_kernel_ex) {
            वापस false;
        }

        स्थिर पूर्णांक nth = params->nth;
        स्थिर पूर्णांक ith = params->ith;

        स्थिर int64_t lhs_batch_size0 = ne12;
        स्थिर int64_t rhs_batch_size0 = ne02;
        स्थिर int64_t batch_आकार      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        स्थिर int64_t r = lhs_batch_size0 / rhs_batch_size0;

        स्थिर int64_t m_group = ne11;
        स्थिर int64_t m       = m_group;
        स्थिर int64_t n       = ne01;
        स्थिर int64_t k       = ne00;

        स्थिर माप_प्रकार lhs_stride = src1->nb[1];
        स्थिर माप_प्रकार rhs_stride = src0->nb[1];
        स्थिर माप_प्रकार dst_stride = dst->nb[1];

        स्थिर int64_t mr = (int64_t) kernel->get_mr();
        स्थिर int64_t nr = (int64_t) kernel->get_nr();
        स्थिर int64_t kr = (int64_t) kernel->get_kr();
        स्थिर int64_t sr = (int64_t) kernel->get_sr();

        स्थिर माप_प्रकार lhs_packed_आकार = lhs_info->packed_आकार_ex(m, k, 0, mr, kr, sr);
        स्थिर माप_प्रकार rhs_packed_आकार = kernels->rhs_info.packed_आकार_ex(n, k, nr, kr, 0);
        स्थिर माप_प्रकार kxn_आकार        = k * n * माप(भग्न);
        स्थिर माप_प्रकार bias_आकार       = n * माप(भग्न);

        स्थिर माप_प्रकार wsize_required = lhs_packed_आकार + rhs_packed_आकार + kxn_आकार + bias_आकार;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_आकार;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_आकार;
        uint8_t * bias       = rhs_kxn + kxn_आकार;

        क्रम (int64_t batch_idx = 0; batch_idx < batch_आकार; ++batch_idx) {
            स्थिर int64_t rhs_batch_idx = batch_idx / r;
            स्थिर uint8_t * rhs_batch_base = static_cast<स्थिर uint8_t *>(src0->तथ्य) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->तथ्य) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                स्थिर int64_t m_roundup_mr = kai_roundup(m, mr);
                स्थिर int64_t अंक_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                अगर (ith < अंक_threads) {
                    स्थिर int64_t अंक_m_per_thread0   = round_down((माप_प्रकार)(m_roundup_mr / अंक_threads), (माप_प्रकार)mr);
                    स्थिर int64_t अंक_m_per_thread_N_1 = m - (अंक_threads - 1) * अंक_m_per_thread0;

                    स्थिर int64_t m_start = ith * अंक_m_per_thread0;
                    स्थिर int64_t m_गिनती = (ith == अंक_threads - 1) ? अंक_m_per_thread_N_1 : अंक_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    स्थिर माप_प्रकार base_packed_off  = lhs_info->get_packed_offset_ex(m_start, k, 0, mr, kr, sr);
                    स्थिर माप_प्रकार अग्र_block_off   = lhs_info->get_packed_offset_ex(m_start + mr, k, 0, mr, kr, sr);
                    स्थिर माप_प्रकार row_stride_bytes = (अग्र_block_off - base_packed_off) / (माप_प्रकार)mr;

                    int64_t remaining = m_गिनती;
                    int64_t cur       = m_start;

                    जबतक (remaining > 0) {
                        स्थिर int64_t row_in_group = cur;
                        स्थिर int64_t avail        = m_group - row_in_group;
                        स्थिर int64_t take         = std::min(avail, remaining);

                        स्थिर uint8_t * lhs_batch_base = static_cast<स्थिर uint8_t *>(src1->तथ्य) + batch_idx * src1->nb[2];
                        स्थिर व्योम * src_ptr = lhs_batch_base + (माप_प्रकार)row_in_group * lhs_stride;
                        स्थिर माप_प्रकार dst_off = base_packed_off + (माप_प्रकार)(cur - m_start) * row_stride_bytes;
                        व्योम * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            अगर (ith == 0) {
                स_रखो(bias, 0, (माप_प्रकार)n * माप(भग्न));
                transpose_f32kxn_f16nxk((माप_प्रकार)n, (माप_प्रकार)k,
                                        reinterpret_cast<भग्न *>(rhs_kxn),
                                        reinterpret_cast<स्थिर uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * माप(भग्न),
                             rhs_kxn, bias, nullptr, rhs_packed, 0, nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                स्थिर int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t अंक_threads_n = KAI_MIN(n / n_step, nth);
                अगर (अंक_threads_n <= 0) {
                    अंक_threads_n = 1;
                }

                अगर (ith < अंक_threads_n) {
                    स्थिर int64_t अंक_n_per_thread0   = round_down((माप_प्रकार)(n / अंक_threads_n), (माप_प्रकार)n_step);
                    स्थिर int64_t अंक_n_per_thread_N_1 = n - (अंक_threads_n - 1) * अंक_n_per_thread0;

                    स्थिर int64_t n_start      = ith * अंक_n_per_thread0;
                    स्थिर int64_t n_to_process = (ith == अंक_threads_n - 1) ? अंक_n_per_thread_N_1 : अंक_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    स्थिर माप_प्रकार lhs_packed_offset0 = lhs_info->get_packed_offset_ex(0, k, 0, mr, kr, sr);
                    स्थिर माप_प्रकार rhs_packed_offset  = kernel->get_rhs_packed_offset_ex(n_start, k, 0);
                    स्थिर माप_प्रकार dst_offset         = kernel->get_dst_offset((माप_प्रकार)0, (माप_प्रकार)n_start, dst_stride);

                    स्थिर व्योम * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    स्थिर व्योम * rhs_ptr = rhs_packed + rhs_packed_offset;
                    भग्न * dst_ptr      = reinterpret_cast<भग्न *>(dst_batch_base + dst_offset);

                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, माप(भग्न), -भग्न_उच्च, भग्न_उच्च);
                }
            }

            अगर (batch_idx != batch_आकार - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        वापस true;
    }

    bool compute_forward_qx(काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q4_K ||
                    dst->src[0]->type == GGML_TYPE_Q8_0 || dst->src[0]->type == GGML_TYPE_Q6_K);

        स्थिर ggml_tensor * src0 = dst->src[0];
        स्थिर ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_द्विआधारी_OP_LOCALS

        स्थिर kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(src0->तथ्य);
        स्थिर bool has_header = kleidiai_is_weight_header_valid(header);
        स्थिर bool is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        स्थिर पूर्णांक slot_योग = kleidiai_collect_kernel_chain(dst, kernel_chain);

        स्वतः weight_for_slot = [&](पूर्णांक slot_सूचकांक, माप_प्रकार & आकार_out) -> स्थिर uint8_t * {
            अगर (slot_सूचकांक < 0 || slot_सूचकांक >= slot_योग) {
                वापस nullptr;
            }
            अगर (has_header) {
                अगर (slot_सूचकांक < header->slot_गिनती) {
                    आकार_out = static_cast<माप_प्रकार>(header->sizes[slot_सूचकांक]);
                    वापस kleidiai_weight_slot_ptr(header, slot_सूचकांक);
                }
                वापस nullptr;
            }
            अगर (slot_सूचकांक == 0) {
                आकार_out = ggml_nbytes(src0);
                वापस static_cast<स्थिर uint8_t *>(src0->तथ्य);
            }
            वापस nullptr;
        };

        काष्ठा runtime_slot {
            पूर्णांक slot_सूचकांक;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            माप_प्रकार mr;
            माप_प्रकार nr;
            माप_प्रकार kr;
            माप_प्रकार sr;
            माप_प्रकार n_step;
            माप_प्रकार lhs_packed_आकार;
            माप_प्रकार lhs_offset;
            माप_प्रकार lhs_bl;
            माप_प्रकार rhs_bl;
            माप_प्रकार pack_bl;
            माप_प्रकार lhs_packed_offset0;
            पूर्णांक assigned_threads;
            पूर्णांक thread_begin;
            पूर्णांक thread_end;
            स्थिर uint8_t * rhs_base;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        पूर्णांक runtime_गिनती = 0;

        क्रम (पूर्णांक slot = 0; slot < slot_योग && runtime_गिनती < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            अगर (!kinfo || !linfo || !linfo->packed_आकार_ex || !linfo->pack_func_ex || !linfo->get_offset ||
                !kinfo->get_rhs_packed_offset_ex || !kinfo->run_kernel_ex || !kinfo->get_dst_offset) {
                जारी;
            }

            माप_प्रकार rhs_आकार = 0;
            स्थिर uint8_t * rhs_ptr = weight_for_slot(slot, rhs_आकार);
            अगर (!rhs_ptr || rhs_आकार == 0) {
                जारी;
            }

            स्थिर kleidiai_block_args block_args = kleidiai_get_block_args(kernels->rhs_type);

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

        अगर (runtime_गिनती == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            वापस false;
        }

        स्थिर पूर्णांक nth_योग = params->nth > 0 ? params->nth : 1;
        स्थिर पूर्णांक ith_योग = params->ith;

        पूर्णांक sme_slot = -1;
        पूर्णांक non_sme_slot = -1;
        क्रम (पूर्णांक i = 0; i < runtime_गिनती; ++i) {
            अगर (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                अवरोध;
            }
        }

        क्रम (पूर्णांक i = 0; i < runtime_गिनती; ++i) {
            अगर (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                अवरोध;
            }
        }

        स्थिर पूर्णांक sme_cap_limit = ctx.sme_thread_cap;
        स्थिर bool use_hybrid = sme_cap_limit > 0 &&
                                 runtime_गिनती > 1 &&
                                 nth_योग > sme_cap_limit;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        माप_प्रकार min_cols_per_thread = 0;
        अगर (runtime_गिनती > 0 && nth_योग > 0) {
            min_cols_per_thread = (माप_प्रकार) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_योग);
        }
        स्थिर bool too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

        स्थिर bool hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        अगर (!hybrid_enabled) {
            पूर्णांक chosen_slot = 0;
            अगर (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_योग > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } अन्यथा अगर (runtime_गिनती > 1 && ctx.sme_thread_cap > 0 && nth_योग > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            अगर (chosen_slot != 0 && chosen_slot < runtime_गिनती) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_threads = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_end = 0;
            }
            runtime_गिनती = runtime_गिनती > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            अगर (runtime_गिनती > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        स्थिर bool sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_mode == RHS_REPACK_SINGLE_ONLY;
        पूर्णांक sme_cap = sme_single_only ? nth_योग : kleidiai_sme_thread_cap();
        अगर (sme_cap < 0) {
            sme_cap = nth_योग;
        }
        sme_cap = std::min(sme_cap, nth_योग);

        पूर्णांक threads_remaining = nth_योग;
        अगर (sme_slot != -1) {
            पूर्णांक sme_threads = std::min(std::max(sme_cap, 0), threads_remaining);
            runtime[sme_slot].assigned_threads = sme_threads;
            threads_remaining -= sme_threads;
        }

        पूर्णांक fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        पूर्णांक fallback_गिनती = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        क्रम (पूर्णांक i = 0; i < runtime_गिनती; ++i) {
            अगर (i == sme_slot) {
                जारी;
            }
            fallback_indices[fallback_गिनती++] = i;
        }

        क्रम (पूर्णांक fi = 0; fi < fallback_गिनती; ++fi) {
            अगर (threads_remaining <= 0) {
                अवरोध;
            }
            स्थिर पूर्णांक slot_सूचकांक = fallback_indices[fi];
            स्थिर पूर्णांक slots_बायाँ = fallback_गिनती - fi;
            पूर्णांक share = (threads_remaining + slots_बायाँ - 1) / slots_बायाँ;
            share     = std::min(share, threads_remaining);
            runtime[slot_सूचकांक].assigned_threads = share;
            threads_remaining -= share;
        }

        पूर्णांक thread_cursor = 0;
        क्रम (पूर्णांक i = 0; i < runtime_गिनती; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_threads;
            runtime[i].thread_end = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        स्थिर पूर्णांक active_threads = thread_cursor;

        पूर्णांक local_slot = -1;
        पूर्णांक local_ith  = 0;
        क्रम (पूर्णांक i = 0; i < runtime_गिनती; ++i) {
            अगर (ith_योग >= runtime[i].thread_begin && ith_योग < runtime[i].thread_end) {
                local_slot = i;
                local_ith  = ith_योग - runtime[i].thread_begin;
                अवरोध;
            }
        }
        स्थिर माप_प्रकार k = ne00;
        स्थिर माप_प्रकार m = ne11;
        स्थिर माप_प्रकार n = ne01;

        माप_प्रकार cursor = 0;
        क्रम (पूर्णांक i = 0; i < runtime_गिनती; ++i) {
            runtime[i].lhs_packed_आकार = runtime[i].lhs_info->packed_आकार_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_offset = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_offset_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_आकार;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        माप_प्रकार common_step = 1;
        क्रम (पूर्णांक i = 0; i < runtime_गिनती; ++i) {
            अगर (runtime[i].assigned_threads == 0) {
                जारी;
            }
            माप_प्रकार अग्र_step = 0;
            अगर (!lcm_आकार(common_step, runtime[i].n_step ? runtime[i].n_step : 1, अग्र_step)) {
                वापस false;
            }
            common_step = अग्र_step;
        }
        GGML_ASSERT(common_step > 0);

        स्थिर bool disable_chunking = ggml_is_numa();
        स्थिर माप_प्रकार chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        स्थिर माप_प्रकार chunk_भाजक = (active_threads == 1 || disable_chunking) ? (माप_प्रकार) active_threads : (माप_प्रकार) active_threads * chunk_multiplier;
        माप_प्रकार chunk_cols = align_up(std::max<माप_प्रकार>(1, ceil_div_आकार(n, chunk_भाजक)), common_step);
        अगर (chunk_cols == 0) {
            chunk_cols = common_step;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        स्थिर माप_प्रकार nchunk_आकार = std::max<माप_प्रकार>(1, ceil_div_आकार(n, chunk_cols));
        GGML_ASSERT(nchunk_आकार <= (माप_प्रकार)पूर्णांक_उच्च);
        स्थिर पूर्णांक nchunk = (पूर्णांक)nchunk_आकार;
        स्थिर माप_प्रकार dst_stride = dst->nb[1];

        स्वतः run_chunk = [&](runtime_slot & slot, माप_प्रकार global_start, माप_प्रकार cols, uint8_t * dst_batch_base) {
            स्थिर माप_प्रकार rhs_packed_offset = slot.kernel->get_rhs_packed_offset_ex(global_start, k, slot.rhs_bl);
            स्थिर माप_प्रकार dst_offset        = slot.kernel->get_dst_offset(0, global_start, dst_stride);

            स्थिर uint8_t * lhs_ptr = scratch + slot.lhs_offset + slot.lhs_packed_offset0;
            स्थिर uint8_t * rhs_ptr = slot.rhs_base + rhs_packed_offset;
            भग्न * dst_ptr         = reinterpret_cast<भग्न *>(dst_batch_base + dst_offset);

            slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       माप(भग्न),
                                       -भग्न_उच्च,
                                       भग्न_उच्च);
        };

        क्रम (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            स्थिर uint8_t * lhs_batch_base = static_cast<स्थिर uint8_t *>(src1->तथ्य) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->तथ्य) + batch_idx * dst->nb[2];

            अगर (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                स्थिर int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_threads = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_threads;
                max_threads = std::max<int64_t>(1, max_threads);
                स्थिर int64_t use_threads = std::min<int64_t>(slot.assigned_threads, max_threads);

                अगर (local_ith < use_threads) {
                    स्थिर int64_t अंक_m_per_thread0   = round_down((माप_प्रकार)(m_roundup_mr / use_threads), slot.mr);
                    स्थिर int64_t अंक_m_per_thread_N_1 = (int64_t)m - (use_threads - 1) * अंक_m_per_thread0;

                    स्थिर int64_t m_start = (int64_t)local_ith * अंक_m_per_thread0;
                    स्थिर int64_t m_गिनती = (local_ith == use_threads - 1) ? अंक_m_per_thread_N_1 : अंक_m_per_thread0;

                    स्थिर माप_प्रकार base_packed_off  = slot.lhs_info->get_packed_offset_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    स्थिर माप_प्रकार अग्र_block_off   = slot.lhs_info->get_packed_offset_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    स्थिर माप_प्रकार row_stride_bytes = slot.mr ? (अग्र_block_off - base_packed_off) / slot.mr : 0;

                    int64_t remaining = m_गिनती;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_offset;
                    जबतक (remaining > 0) {
                        स्थिर int64_t row_in_group = cur;
                        स्थिर int64_t avail        = (int64_t)m - row_in_group;
                        स्थिर int64_t take         = std::min(avail, remaining);

                        स्थिर माप_प्रकार src_off = slot.lhs_info->get_offset(row_in_group, src1->nb[1]);
                        स्थिर व्योम * src_ptr = lhs_batch_base + src_off;
                        स्थिर माप_प्रकार dst_off = base_packed_off + (माप_प्रकार)(cur - m_start) * row_stride_bytes;
                        व्योम * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            अगर (ith_योग == 0) {
                ggml_threadpool_chunk_set(params->threadpool, active_threads);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            अगर (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                पूर्णांक current_chunk = ith_योग;
                जबतक (current_chunk < nchunk) {
                    स्थिर माप_प्रकार global_start = (माप_प्रकार)current_chunk * chunk_cols;
                    अगर (global_start >= n) {
                        अवरोध;
                    }

                    स्थिर माप_प्रकार cols = std::min(chunk_cols, n - global_start);
                    अगर (cols > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        run_chunk(slot, global_start, cols, dst_batch_base);
                    }

                    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                }
            }

            अगर (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        वापस true;
    }

खुला:
    पूर्णांक repack(काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        स्थिर माप_प्रकार n = tensor->ne[1];
        स्थिर माप_प्रकार k = tensor->ne[0];

        kleidiai_weight_header * header = kleidiai_weight_header_from_ptr(tensor->तथ्य);
        अगर (!header) {
            वापस -1;
        }

        header->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        header->version    = GGML_KLEIDIAI_PACK_VERSION;
        header->slot_गिनती = 0;

        uint8_t * base_ptr = static_cast<uint8_t *>(tensor->तथ्य);
        माप_प्रकार cursor = माप(kleidiai_weight_header);
        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        स्थिर bool want_q4k = tensor->type == GGML_TYPE_Q4_K;
        स्थिर bool want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        स्थिर bool want_f32 = tensor->type == GGML_TYPE_F32;
        स्थिर पूर्णांक slot_योग = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        स्थिर bool allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<भग्न>  scales;
        std::vector<भग्न>  bias;
        std::vector<भग्न> row_f32;

        अगर (want_q8 && slot_योग > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            स्थिर माप_प्रकार row_stride = tensor->nb[1];
            स्थिर माप_प्रकार k_blocks   = (k + QK8_0 - 1) / QK8_0;

            अगर (tensor->type == GGML_TYPE_Q6_K) {
                अगर (k % QK_K != 0 || k % QK8_0 != 0) {
                    वापस -1;
                }
                row_f32.resize(k);
            }

            क्रम (माप_प्रकार row = 0; row < n; ++row) {
                स्थिर uint8_t * row_तथ्य = static_cast<स्थिर uint8_t *>(तथ्य) + row * row_stride;

                अगर (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<स्थिर block_q6_K *>(row_तथ्य), row_f32.तथ्य(), k);
                }

                भग्न max_abs = 0.0f;
                अगर (tensor->type == GGML_TYPE_Q6_K) {
                    क्रम (माप_प्रकार linear_idx = 0; linear_idx < k; ++linear_idx) {
                        max_abs = std::max(max_abs, std::भ_असल(row_f32[linear_idx]));
                    }
                } अन्यथा {
                    स्थिर block_q8_0 * row_blocks = reinterpret_cast<स्थिर block_q8_0 *>(row_तथ्य);
                    क्रम (माप_प्रकार block = 0; block < k_blocks; ++block) {
                        स्थिर block_q8_0 & blk = row_blocks[block];
                        स्थिर भग्न d = GGML_FP16_TO_FP32(blk.d);
                        क्रम (माप_प्रकार l = 0; l < QK8_0; ++l) {
                            स्थिर माप_प्रकार linear_idx = block * QK8_0 + l;
                            अगर (linear_idx >= k) {
                                अवरोध;
                            }
                            स्थिर भग्न value = d * static_cast<भग्न>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::भ_असल(value));
                        }
                    }
                }

                भग्न scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                स्थिर भग्न inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                अगर (tensor->type == GGML_TYPE_Q6_K) {
                    क्रम (माप_प्रकार linear_idx = 0; linear_idx < k; ++linear_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[linear_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                    }
                } अन्यथा {
                    स्थिर block_q8_0 * row_blocks = reinterpret_cast<स्थिर block_q8_0 *>(row_तथ्य);
                    क्रम (माप_प्रकार block = 0; block < k_blocks; ++block) {
                        स्थिर block_q8_0 & blk = row_blocks[block];
                        स्थिर भग्न d = GGML_FP16_TO_FP32(blk.d);
                        क्रम (माप_प्रकार l = 0; l < QK8_0; ++l) {
                            स्थिर माप_प्रकार linear_idx = block * QK8_0 + l;
                            अगर (linear_idx >= k) {
                                अवरोध;
                            }
                            स्थिर भग्न value = d * static_cast<भग्न>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + linear_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        अगर (want_f32 && slot_योग > 0) {
            bias.resize(n, 0.0f);
        }

        क्रम (पूर्णांक slot = 0; slot < slot_योग && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            अगर (!allow_fallback && slot > 0) {
                अवरोध;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            अगर (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_आकार_ex || !kernel) {
                जारी;
            }

            स्थिर माप_प्रकार nr = kernel->get_nr();
            स्थिर माप_प्रकार kr = kernel->get_kr();
            स्थिर माप_प्रकार sr = kernel->get_sr();
            स्थिर ggml_type rhs_type = kernels->rhs_type;
            अगर (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            स्थिर माप_प्रकार block_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            अगर (block_len == SIZE_MAX) {
                जारी;
            }

            स्थिर माप_प्रकार packed_आकार = rhs_info->packed_आकार_ex(n, k, nr, kr, block_len);
            स्थिर माप_प्रकार aligned_cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = base_ptr + aligned_cursor;

            अगर (rhs_type == GGML_TYPE_Q4_0) {
                काष्ठा kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<स्थिर uint8_t *>(तथ्य), nullptr, nullptr,
                                       dst_ptr, 0, &params);
            } अन्यथा अगर (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       तथ्य, nullptr, nullptr, dst_ptr, 0, nullptr);
            } अन्यथा अगर (rhs_type == GGML_TYPE_Q8_0) {
                काष्ठा kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.तथ्य(), nullptr, scales.तथ्य(),
                                       dst_ptr, 0, &params);
            } अन्यथा अगर (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       तथ्य, bias.तथ्य(), nullptr,
                                       dst_ptr, 0, nullptr);
            } अन्यथा {
                जारी;
            }

            header->offsets[header->slot_गिनती] = aligned_cursor;
            header->sizes[header->slot_गिनती]   = packed_आकार;
            ++header->slot_गिनती;

            cursor = aligned_cursor + packed_आकार;
        }

        अगर (header->slot_गिनती == 0) {
            header->magic   = 0;
            header->version = 0;
            स_नकल(tensor->तथ्य, तथ्य, तथ्य_आकार);
        }

        वापस 0;
    }
};

अटल ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_बफर_t, काष्ठा ggml_tensor *) {
    अटल tensor_traits traits;
    वापस &traits;
}
}  // namespace ggml::cpu::kleidiai

अटल क्रमागत ggml_status ggml_backend_cpu_kleidiai_बफर_init_tensor(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor) {
    tensor->extra = (व्योम *) ggml::cpu::kleidiai::get_tensor_traits(बफर, tensor);

    वापस GGML_STATUS_SUCCESS;
}

अटल व्योम ggml_backend_cpu_kleidiai_बफर_set_tensor(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor,
                                                       स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    स्वतः tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    स्वतः OK            = tensor_traits->repack(tensor, तथ्य, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(बफर);
}

अटल स्थिर अक्षर * ggml_backend_cpu_kleidiai_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    GGML_UNUSED(buft);
    वापस "CPU_KLEIDIAI";
}

अटल ggml_backend_बफर_t ggml_backend_cpu_kleidiai_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    ggml_backend_बफर_t बफर = ggml_backend_buft_alloc_बफर(ggml_backend_cpu_बफर_type(), size);

    अगर (बफर == nullptr) {
        वापस nullptr;
    }

    बफर->buft              = buft;
    बफर->iface.init_tensor = ggml_backend_cpu_kleidiai_बफर_init_tensor;
    बफर->iface.set_tensor  = ggml_backend_cpu_kleidiai_बफर_set_tensor;
    बफर->iface.get_tensor  = nullptr;
    बफर->iface.cpy_tensor  = nullptr;
    वापस बफर;
}

अटल माप_प्रकार ggml_backend_cpu_kleidiai_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    GGML_UNUSED(buft);
    वापस TENSOR_ALIGNMENT;
}

अटल माप_प्रकार ggml_backend_cpu_kleidiai_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर काष्ठा ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    अगर (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        वापस ggml_nbytes(tensor);
    }

    स्थिर माप_प्रकार n = tensor->ne[1];
    स्थिर माप_प्रकार k = tensor->ne[0];

    माप_प्रकार cursor = माप(kleidiai_weight_header);
    cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    स्थिर bool want_q4k = tensor->type == GGML_TYPE_Q4_K;
    स्थिर bool want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    स्थिर bool want_f32 = tensor->type == GGML_TYPE_F32;
    स्थिर पूर्णांक slot_योग = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    स्थिर bool allow_fallback = kleidiai_pack_fallback_allowed();

    माप_प्रकार slot_गिनती = 0;
    क्रम (पूर्णांक slot = 0; slot < slot_योग; ++slot) {
        अगर (!allow_fallback && slot > 0) {
            अवरोध;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        अगर (!kernels) {
            जारी;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        अगर (!kernel || !rhs_info || !rhs_info->packed_आकार_ex) {
            जारी;
        }

        स्थिर ggml_type rhs_type = kernels->rhs_type;
        स्थिर माप_प्रकार block_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        अगर (block_len == SIZE_MAX) {
            जारी;
        }

        cursor = align_up(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_आकार_ex(n, k, kernel->get_nr(), kernel->get_kr(), block_len);
        ++slot_गिनती;
    }

    अगर (slot_गिनती == 0) {
        वापस ggml_nbytes(tensor);
    }

    वापस std::max(cursor, ggml_nbytes(tensor));
}

namespace ggml::cpu::kleidiai {
श्रेणी extra_बफर_type : ggml::cpu::extra_बफर_type {
    bool supports_op(ggml_backend_dev_t, स्थिर काष्ठा ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        अगर (op->op != GGML_OP_MUL_MAT) {
            वापस false;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        स्थिर पूर्णांक slot_योग = kleidiai_collect_kernel_chain(op, kernel_chain);
        स्थिर bool src0_is_kleidiai =
            op->src[0]->बफर &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->बफर->buft->context == यह &&
            slot_योग > 0;

        अगर ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            अगर (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == nullptr) {
                वापस false;
            }
            अगर (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == nullptr) {
                वापस false;
            }
            अगर (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == nullptr) {
                वापस false;
            }
            अगर (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == nullptr) {
                वापस false;
            }
            अगर (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == nullptr) {
                वापस false;
            }
            अगर (op->src[1]->बफर && !ggml_backend_buft_is_host(op->src[1]->बफर->buft)) {
                वापस false;
            }

            अगर (op->src[0]->type == GGML_TYPE_Q4_K) {
                अगर (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    वापस false;
                }
                वापस ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1 &&
                       op->src[1]->nb[0] == माप(भग्न) &&
                       (op->src[1]->ne[1] == 1 || op->src[1]->nb[1] == op->src[1]->ne[0] * माप(भग्न));
            }

            अगर (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                अगर (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_ne(op->src[1], 3) == 1) {
                    वापस true;
                }
                वापस false;
            }

            अगर (op->op != GGML_OP_MUL_MAT || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                वापस false;
            }

            वापस true;
        }

        वापस false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(स्थिर काष्ठा ggml_tensor * op) override {
        अगर (op->op == GGML_OP_MUL_MAT) {
            अगर (op->src[0]->बफर && op->src[0]->बफर->buft->context == यह) {
                वापस (ggml::cpu::tensor_traits *) op->src[0]->extra;
            } अन्यथा {
                // KleidiAI only has kernels for Q4_0, Q4_K, Q8_0 and Q6_K. For a quantized weight of any
                // other type (K-quants, IQ) it declines the op and returns nullptr below, so
                // KleidiAI does not accelerate it. Another CPU backend may still take the op,
                // and this can run during graph planning, so the message says what KleidiAI
                // did rather than what ends up executing. Warn once per process.
                अगर (ggml_is_quantized(op->src[0]->type) &&
                    op->src[0]->type != GGML_TYPE_Q4_0 && op->src[0]->type != GGML_TYPE_Q4_K &&
                    op->src[0]->type != GGML_TYPE_Q8_0 && op->src[0]->type != GGML_TYPE_Q6_K) {
                    अटल std::atomic<bool> warned(false);
                    अगर (!warned.exchange(true)) {
                        GGML_LOG_WARN("kleidiai: no kernel for tensor type %s, not accelerated by KleidiAI "
                                      "(kernels available for Q4_0, Q4_K, Q8_0 and Q6_K)\n",
                                      ggml_type_name(op->src[0]->type));
                    }
                }
                अगर (op->src[0]->type != GGML_TYPE_F16) {
                    वापस nullptr;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                स्थिर पूर्णांक slot_योग = kleidiai_collect_kernel_chain(op, kernel_chain);
                अगर (slot_योग > 0 && op->src[1]->ne[1] > 1) {
                    अगर ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        वापस nullptr;
                    }
                    वापस ggml::cpu::kleidiai::get_tensor_traits(शून्य, शून्य);
                }
            }
        }
        वापस nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_बफर_type_t ggml_backend_cpu_kleidiai_बफर_type(व्योम) {
    अटल ggml::cpu::kleidiai::extra_बफर_type ctx;
    अटल काष्ठा ggml_backend_बफर_type ggml_backend_cpu_बफर_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_बफर_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_बफर_type_alloc_बफर,
                           /* .alloc_buffer_n       = */ nullptr,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_बफर_type_get_alignment,
                           /* .get_max_size         = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_बफर_type_get_alloc_आकार,
                           /* .get_alloc_size_n     = */ शून्य,
                           /* .is_host              = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    वापस &ggml_backend_cpu_बफर_type_kleidiai;
}
