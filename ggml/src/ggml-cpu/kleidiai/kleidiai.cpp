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
#यदि_अन्तः

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

#परिभाषय GGML_COMMON_DECL_CPP
#अन्तर्भाव "ggml-common.h"

स्थिर constexpr पूर्णाङ्क      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2;
स्थिर constexpr uint32_t GGML_KLEIDIAI_PACK_MAGIC       = 0x4b4c4149; // "KLAI"
स्थिर constexpr uint16_t GGML_KLEIDIAI_PACK_संस्करणम्     = 1;
स्थिर constexpr size_t   GGML_KLEIDIAI_PACK_ALIGN       = 64;
स्थिर constexpr size_t   GGML_KLEIDIAI_Q4_K_BL          = QK_K / 8;

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
} स्थिर ctx = { CPU_FEATURE_NONE, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, शून्यसूचकः, 0, -1, 4 };

स्थिर अन्तःस्थ तार्किक is_sme_family(cpu_feature f) {
    प्रत्यागम (f & (CPU_FEATURE_SME | CPU_FEATURE_SME2)) != CPU_FEATURE_NONE;
}

स्थिर ध्रुव वर्ण* cpu_feature_गन्तव्यम्_string(cpu_feature f) {
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
स्थिर तार्किक parse_cpu_dir_name(ध्रुव वर्ण* name, size_t* cpu) {
    यदि (वाक्ये_तुलय(name, "cpu", 3) != 0 ||
        name[3] < '0' || name[3] > '9') {
        प्रत्यागम असत्य;
    }

    ध्रुव वर्ण* first = name + 3;
    ध्रुव वर्ण* चरमः = name + वाक्यदैर्घ्यम्(name);

    size_t value = 0;
    ध्रुव स्वचालित [end, ec] = std::from_chars(first, चरमः, value, 10);

    यदि (ec != std::errc{} || end != चरमः) {
        प्रत्यागम असत्य;
    }

    *cpu = value;
    प्रत्यागम सत्य;
}

स्थिर std::vector<size_t> detect_cpu_ids() {
    std::vector<size_t> cpus;

    DIR * dir = opendir("/sys/devices/system/cpu");
    यदि (dir == शून्यसूचकः) {
        प्रत्यागम cpus;
    }

    यावत् (dirent * प्रविष्टिः = readdir(dir)) {
        size_t cpu = 0;
        यदि (parse_cpu_dir_name(प्रविष्टिः->d_name, &cpu)) {
            cpus.push_back(cpu);
        }
    }
    closedir(dir);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    प्रत्यागम cpus;
}
#यदि_अन्तः

#यदि defined(__APPLE__) && defined(__aarch64__)
स्थिर तार्किक apple_sme_counted_perf_level(std::string name) {
    प्रति (std::string::size_type i = 0; i < name.size(); ++i) {
        name[i] = (वर्ण) std::tolower((अचिह्नित वर्ण) name[i]);
    }

    // Conservative ceiling: only count perf-level names observed to provide full SME throughput.
    // Future names should be calibrated here before they raise the automatic SME thread cap.
    प्रत्यागम name.find("super") != std::string::npos ||
           name.find("performance") != std::string::npos;
}
#यदि_अन्तः

स्थिर शून्य सङ्कलय_smcus_from_smidr(uint64_t smidr, size_t & अङ्कः_वैयक्तिकम्, std::map<uint32_t, size_t> & shared_counts) {
    // Arm ARM: SMIDR_EL1. SH==0 is implementation-defined; keep the existing
    // conservative policy and only treat zero affinity as private.
    ध्रुव uint32_t sh = (uint32_t)((smidr >> 13) & 0x3);
    ध्रुव uint32_t nsmc = (uint32_t)((smidr >> 56) & 0xF);
    ध्रुव size_t shared_गणनम् = nsmc == 0xF ? 1 : (size_t)nsmc + 1;
    ध्रुव uint32_t affinity = (uint32_t)(smidr & 0xFFFu);
    ध्रुव uint32_t affinity2 = (uint32_t)((smidr >> 32) & 0xFFFFFu);
    ध्रुव uint32_t id = (affinity2 << 12) | affinity;

    यदि (nsmc == 0xF) {
        GGML_LOG_WARN("kleidiai: NSMC detected as 0xF indicating reseved value, setting min safe shared SMCU count to 1");
    }

    विकल्प (sh) {
        पक्ष 2: // private SMCU
            ++अङ्कः_वैयक्तिकम्;
            भङ्ग;
        पक्ष 3: // shared SMCU
            यदि (shared_counts[id] < shared_गणनम्) {
                shared_counts[id] = shared_गणनम्;
            }
            भङ्ग;
        पक्ष 0:
            यदि (id == 0) {
                ++अङ्कः_वैयक्तिकम्;
            } अन्यथा यदि (shared_counts[id] < shared_गणनम्) {
                shared_counts[id] = shared_गणनम्;
            }
            भङ्ग;
        स्वभाव:
            भङ्ग;
    }
}

स्थिर size_t detect_अङ्कः_smcus() {
    ध्रुव स्वचालित runtime_feat = ggml_feats_get_arch64_runtime();
    यदि (!runtime_feat.has_sme) {
        प्रत्यागम 0;
    }

#यदि defined(__linux__) && defined(__aarch64__)
    // Linux/aarch64: Best-effort count of Streaming Mode Compute Units (SMCUs) via SMIDR_EL1 sysfs.
    size_t अङ्कः_वैयक्तिकम् = 0;
    std::map<uint32_t, size_t> shared_counts;

    ध्रुव std::vector<size_t> cpus = detect_cpu_ids();
    प्रति (ध्रुव size_t cpu : cpus) {
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

        सङ्कलय_smcus_from_smidr(smidr, अङ्कः_वैयक्तिकम्, shared_counts);
    }

    size_t योग = अङ्कः_वैयक्तिकम्;
    प्रति (ध्रुव स्वचालित & प्रविष्टिः : shared_counts) {
        योग += प्रविष्टिः.second;
    }
    प्रत्यागम योग;

#नोचेद्यदि defined(__APPLE__) && defined(__aarch64__)
    पूर्णाङ्क perf_स्तराः = 0;
    size_t size = परिमाण(perf_स्तराः);
    यदि (sysctlbyname("hw.nperflevels", &perf_स्तराः, &size, शून्यसूचकः, 0) != 0 ||
        size != परिमाण(perf_स्तराः) || perf_स्तराः <= 0) {
        प्रत्यागम 0;
    }

    size_t units = 0;
    प्रति (पूर्णाङ्क i = 0; i < perf_स्तराः; ++i) {
        वर्ण कुञ्चिका[64] = {};
        पूर्णाङ्क physical_cpus = 0;
        पूर्णाङ्क cpus_प्रत्येकम्_l2 = 0;

        snprintf(कुञ्चिका, परिमाण(कुञ्चिका), "hw.perflevel%d.physicalcpu", i);
        size = परिमाण(physical_cpus);
        यदि (sysctlbyname(कुञ्चिका, &physical_cpus, &size, शून्यसूचकः, 0) != 0 ||
            size != परिमाण(physical_cpus) || physical_cpus <= 0) {
            अनुवर्त;
        }

        snprintf(कुञ्चिका, परिमाण(कुञ्चिका), "hw.perflevel%d.cpusperl2", i);
        size = परिमाण(cpus_प्रत्येकम्_l2);
        यदि (sysctlbyname(कुञ्चिका, &cpus_प्रत्येकम्_l2, &size, शून्यसूचकः, 0) != 0 ||
            size != परिमाण(cpus_प्रत्येकम्_l2) || cpus_प्रत्येकम्_l2 <= 0) {
            अनुवर्त;
        }

        snprintf(कुञ्चिका, परिमाण(कुञ्चिका), "hw.perflevel%d.name", i);
        size = 0;
        यदि (sysctlbyname(कुञ्चिका, शून्यसूचकः, &size, शून्यसूचकः, 0) != 0 || size == 0) {
            अनुवर्त;
        }

        std::string name(size, '\0');
        यदि (sysctlbyname(कुञ्चिका, &name[0], &size, शून्यसूचकः, 0) != 0) {
            अनुवर्त;
        }
        name.resize(size);
        यावत् (!name.रिक्तानि() && name.back() == '\0') {
            name.pop_back();
        }

        यदि (apple_sme_counted_perf_level(name)) {
            units += (size_t) ((physical_cpus + cpus_प्रत्येकम्_l2 - 1) / cpus_प्रत्येकम्_l2);
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
#यदि_अन्तः
}

स्थिर पूर्णाङ्क parse_uint_env(ध्रुव वर्ण *s, ध्रुव वर्ण *name, तार्किक *ok) {
    यदि (!s) { *ok = असत्य; प्रत्यागम 0; }
    वर्ण *end = शून्यसूचकः;
    दीर्घ v = strtol(s, &end, 10);
    यदि (end == s || *end != '\0') {
        GGML_LOG_WARN("kleidiai: invalid %s='%s' (expected integer)\n", name, s);
        *ok = असत्य;
        प्रत्यागम 0;
    }
    यदि (v < 0 || v > अधिकतम_पूर्णाङ्कः) {
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
        ध्रुव वर्ण *env_sme         = getenv("GGML_KLEIDIAI_SME");
        ध्रुव वर्ण *env_तन्तवः     = getenv("GGML_TOTAL_THREADS");
        ध्रुव वर्ण *env_chunk_mult  = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER");

        ध्रुव स्वचालित runtime_feat = ggml_feats_get_arch64_runtime();

        size_t detected_smcus = 0;

        // The fixed-width SVE kernels require 32-byte vectors, including the Q8 shared RHS layout.
        ctx.features  = (runtime_feat.has_dotprod  ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (runtime_feat.has_i8mm     ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (runtime_feat.has_svei8mm  ? CPU_FEATURE_SVEI8MM : CPU_FEATURE_NONE) |
                        (runtime_feat.has_fp16     ? CPU_FEATURE_FP16    : CPU_FEATURE_NONE) |
                        (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);

        यदि (env_तन्तवः) {
            तार्किक ok = असत्य;
            पूर्णाङ्क hint = parse_uint_env(env_तन्तवः, "GGML_TOTAL_THREADS", &ok);
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
        तार्किक sme_env_समुच्चयः = (env_sme != शून्यसूचकः);

        ध्रुव तार्किक has_supported_sme_family = runtime_feat.has_sme;
        तार्किक sme_cap_detected = असत्य;

        यदि (has_supported_sme_family) {
            detected_smcus = detect_अङ्कः_smcus();
            sme_cap_detected = detected_smcus > 0;
            // Some platforms expose SME without exposing a calibrated SMCU count.
            // Use one SME thread as the conservative default; add platform SMCU detection to raise it.
            sme_cores = sme_cap_detected ? (पूर्णाङ्क)detected_smcus : 1;

            यदि (!sme_env_समुच्चयः && !sme_cap_detected) {
                GGML_LOG_INFO("kleidiai: SME detected; SMCU count unavailable, using conservative SME thread cap=1\n");
            }
        }

        // Runtime-detect SME support and available SMCUs first. The detected SMCU
        // count is used as the SME thread cap, and GGML_KLEIDIAI_SME can debug-override that:
        //   - unset: use runtime detection.
        //   - 0:     disable SME-family kernels.
        //   - N > 0: use N as the SME thread cap, if an SME-family kernel is selectable.
        यदि (sme_env_समुच्चयः) {
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
            GGML_LOG_INFO("kleidiai: primary q4 kernel feature %s\n", cpu_feature_गन्तव्यम्_string(ctx.kernels_q4->required_cpu));
        }

        यदि (!ctx.kernels_q8) {
            GGML_LOG_INFO("kleidiai: no compatible q8 kernels found for CPU features mask %d\n", (पूर्णाङ्क)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q8 kernel feature %s\n", cpu_feature_गन्तव्यम्_string(ctx.kernels_q8->required_cpu));
        }

        यदि (!ctx.kernels_q4k) {
            GGML_LOG_INFO("kleidiai: no compatible q4_k kernels found for CPU features mask %d\n", (पूर्णाङ्क) ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary q4_k kernel feature %s\n", cpu_feature_गन्तव्यम्_string(ctx.kernels_q4k->required_cpu));
        }

        यदि (!ctx.kernels_f32) {
            GGML_LOG_INFO("kleidiai: no compatible f32 kernels found for CPU features mask %d\n", (पूर्णाङ्क)ctx.features);
        } अन्यथा {
            GGML_LOG_INFO("kleidiai: primary f32 kernel feature %s\n", cpu_feature_गन्तव्यम्_string(ctx.kernels_f32->required_cpu));
        }

        ध्रुव तार्किक has_selected_sme_family_kernel =
            (ctx.kernels_q4  && is_sme_family(ctx.kernels_q4->required_cpu)) ||
            (ctx.kernels_q4k && is_sme_family(ctx.kernels_q4k->required_cpu)) ||
            (ctx.kernels_q8  && is_sme_family(ctx.kernels_q8->required_cpu)) ||
            (ctx.kernels_f32 && is_sme_family(ctx.kernels_f32->required_cpu));
        ctx.sme_thread_cap = has_selected_sme_family_kernel ? sme_cores : 0;

        यदि (has_selected_sme_family_kernel) {
            यदि (sme_env_समुच्चयः && sme_env_ok && sme_cores > 0) {
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

    ggml_critical_section_अन्तः();
}

स्थिर अन्तःस्थ पूर्णाङ्क kleidiai_sme_thread_cap() {
    प्रत्यागम ctx.sme_thread_cap;
}

स्थिर अन्तःस्थ size_t align_ऊर्ध्वम्(size_t value, size_t alignment) {
    यदि (alignment == 0) {
        प्रत्यागम value;
    }
    ध्रुव size_t remainder = value % alignment;
    प्रत्यागम remainder == 0 ? value : value + (alignment - remainder);
}

स्थिर अन्तःस्थ size_t gcd_आकार(size_t a, size_t b) {
    यावत् (b != 0) {
        ध्रुव size_t t = a % b;
        a = b;
        b = t;
    }
    प्रत्यागम a;
}

स्थिर अन्तःस्थ तार्किक lcm_आकार(size_t a, size_t b, size_t & परिणाम) {
    यदि (a == 0 || b == 0) {
        परिणाम = 0;
        प्रत्यागम असत्य;
    }
    ध्रुव size_t g = gcd_आकार(a, b);
    ध्रुव size_t q = a / g;
    यदि (q > SIZE_MAX / b) {
        प्रत्यागम असत्य;
    }
    परिणाम = q * b;
    प्रत्यागम सत्य;
}

स्थिर अन्तःस्थ size_t ceil_div_आकार(size_t a, size_t b) {
    प्रत्यागम b == 0 ? 0 : (a + b - 1) / b;
}

स्थिर अन्तःस्थ size_t kleidiai_chunk_स्तम्भसङ्ख्या(size_t n, पूर्णाङ्क nth_योग, तार्किक disable_chunking, size_t n_चरणः) {
    ध्रुव size_t multiplier = (nth_योग == 1 || disable_chunking) ? 1 : std::max<size_t>(1, (size_t) ctx.chunk_multiplier);
    ध्रुव size_t भाजकः = std::max<size_t>(1, (size_t) nth_योग * multiplier);
    ध्रुव size_t chunk_स्तम्भसङ्ख्या = align_ऊर्ध्वम्(std::max<size_t>(1, ceil_div_आकार(n, भाजकः)), n_चरणः);
    प्रत्यागम chunk_स्तम्भसङ्ख्या ? chunk_स्तम्भसङ्ख्या : n_चरणः;
}

संरचना kleidiai_खण्डम्_तर्काः {
    size_t lhs_bl;
    size_t rhs_bl;
    size_t pack_bl;
};

स्थिर अन्तःस्थ kleidiai_खण्डम्_तर्काः kleidiai_get_खण्डम्_तर्काः(ggml_type rhs_type) {
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

संरचना kleidiai_weight_शीर्षकम् {
    uint32_t magic;
    uint16_t संस्करणम्_2;
    uint16_t slot_गणनम्;
    uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
    uint64_t sizes[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
};

स्थिर अन्तःस्थ kleidiai_weight_शीर्षकम् * kleidiai_weight_शीर्षकम्_from_ptr(शून्य * दत्त) {
    प्रत्यागम reinterpret_cast<kleidiai_weight_शीर्षकम् *>(दत्त);
}

स्थिर अन्तःस्थ ध्रुव kleidiai_weight_शीर्षकम् * kleidiai_weight_शीर्षकम्_from_ptr(ध्रुव शून्य * दत्त) {
    प्रत्यागम reinterpret_cast<ध्रुव kleidiai_weight_शीर्षकम् *>(दत्त);
}

स्थिर अन्तःस्थ तार्किक kleidiai_is_weight_शीर्षकम्_valid(ध्रुव kleidiai_weight_शीर्षकम् * शीर्षकम्) {
    यदि (!शीर्षकम्) {
        प्रत्यागम असत्य;
    }
    यदि (शीर्षकम्->magic != GGML_KLEIDIAI_PACK_MAGIC || शीर्षकम्->संस्करणम्_2 != GGML_KLEIDIAI_PACK_संस्करणम्) {
        प्रत्यागम असत्य;
    }
    यदि (शीर्षकम्->slot_गणनम् == 0 || शीर्षकम्->slot_गणनम् > GGML_KLEIDIAI_MAX_KERNEL_SLOTS) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

स्थिर अन्तःस्थ uint8_t * kleidiai_weight_slot_ptr(kleidiai_weight_शीर्षकम् * शीर्षकम्, पूर्णाङ्क slot) {
    यदि (!kleidiai_is_weight_शीर्षकम्_valid(शीर्षकम्)) {
        प्रत्यागम शून्यसूचकः;
    }
    यदि (slot < 0 || slot >= शीर्षकम्->slot_गणनम्) {
        प्रत्यागम शून्यसूचकः;
    }
    प्रत्यागम reinterpret_cast<uint8_t *>(शीर्षकम्) + शीर्षकम्->offsets[slot];
}

स्थिर अन्तःस्थ ध्रुव uint8_t * kleidiai_weight_slot_ptr(ध्रुव kleidiai_weight_शीर्षकम् * शीर्षकम्, पूर्णाङ्क slot) {
    यदि (!kleidiai_is_weight_शीर्षकम्_valid(शीर्षकम्)) {
        प्रत्यागम शून्यसूचकः;
    }
    यदि (slot < 0 || slot >= शीर्षकम्->slot_गणनम्) {
        प्रत्यागम शून्यसूचकः;
    }
    प्रत्यागम reinterpret_cast<ध्रुव uint8_t *>(शीर्षकम्) + शीर्षकम्->offsets[slot];
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

प्रतिरूपम् <प्रकारनाम SelectFallback>
स्थिर पूर्णाङ्क kleidiai_collect_kernel_chain_common(
        ggml_kleidiai_kernels * primary,
        cpu_feature features,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & बहिर्गमनम्,
        SelectFallback select_fallback) {
    पूर्णाङ्क count = 0;
    यदि (!primary) {
        प्रत्यागम 0;
    }
    बहिर्गमनम्[count++] = primary;

    यदि (primary->rhs_info.repack_बहुलकम् == RHS_REPACK_SINGLE_ONLY) {
        प्रत्यागम count;
    }

    यदि (is_sme_family(primary->required_cpu)) {
        ध्रुव cpu_feature fallback_आवरणम् = static_cast<cpu_feature>(features & ~(CPU_FEATURE_SME | CPU_FEATURE_SME2));
        यदि (fallback_आवरणम् != CPU_FEATURE_NONE) {
            ggml_kleidiai_kernels * fallback = select_fallback(fallback_आवरणम्);
            यदि (fallback && fallback != primary &&
                fallback->rhs_info.repack_बहुलकम् != RHS_REPACK_SINGLE_ONLY &&
                fallback->lhs_type == primary->lhs_type &&
                fallback->rhs_type == primary->rhs_type &&
                fallback->op_type  == primary->op_type) {
                बहिर्गमनम्[count++] = fallback;
            }
        }
    }

    प्रत्यागम count;
}

स्थिर पूर्णाङ्क kleidiai_collect_kernel_chain(ध्रुव संरचना ggml_tensor * op,
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & बहिर्गमनम्) {
    ggml_kleidiai_kernels * primary = ggml_kleidiai_select_kernels(ctx.features, op);
    प्रत्यागम kleidiai_collect_kernel_chain_common(primary, ctx.features, बहिर्गमनम्,
        [&](cpu_feature आवरणम्) { प्रत्यागम ggml_kleidiai_select_kernels(आवरणम्, op); });
}

स्थिर पूर्णाङ्क kleidiai_collect_q4_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & बहिर्गमनम्) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4();
    प्रत्यागम kleidiai_collect_kernel_chain_common(primary, ctx.features, बहिर्गमनम्,
        [&](cpu_feature आवरणम्) { प्रत्यागम ggml_kleidiai_select_kernels_q4_0(आवरणम्); });
}

स्थिर पूर्णाङ्क kleidiai_collect_q4k_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & बहिर्गमनम्) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q4k();
    प्रत्यागम kleidiai_collect_kernel_chain_common(primary, ctx.features, बहिर्गमनम्,
        [&](cpu_feature आवरणम्) { प्रत्यागम ggml_kleidiai_select_kernels_q4_k(आवरणम्); });
}

स्थिर पूर्णाङ्क kleidiai_collect_q8_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & बहिर्गमनम्) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_q8();
    प्रत्यागम kleidiai_collect_kernel_chain_common(primary, ctx.features, बहिर्गमनम्,
        [&](cpu_feature आवरणम्) { प्रत्यागम ggml_kleidiai_select_kernels_q8_0(आवरणम्); });
}

स्थिर पूर्णाङ्क kleidiai_collect_f32_chain(std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> & बहिर्गमनम्) {
    ggml_kleidiai_kernels * primary = kleidiai_primary_kernel_f32();
    प्रत्यागम kleidiai_collect_kernel_chain_common(primary, ctx.features, बहिर्गमनम्,
        [&](cpu_feature आवरणम्) { प्रत्यागम ggml_kleidiai_select_kernels_f32(आवरणम्); });
}

स्थिर अन्तःस्थ int64_t ggml_ne(ध्रुव ggml_tensor * tensor, पूर्णाङ्क dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    प्रत्यागम tensor->ne[dim];
}

नामाकाश ggml::cpu::kleidiai {

स्थिर size_t आवर्तनम्_अधस्तात्(size_t x, size_t y) {
    प्रत्यागम y == 0 ? x : x - (x % y);
}

स्थिर शून्य transpose_f32kxn_f16nxk(size_t n, size_t k, भिन्न * dst, ध्रुव uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / परिमाण(uint16_t);
    size_t dst_stride = n;

    प्रति (size_t k_idx = 0; k_idx < k; ++k_idx) {
        प्रति (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

वर्ग tensor_traits : सार्वजनिक ggml::cpu::tensor_traits {
    तार्किक work_आकार(पूर्णाङ्क /* n_threads */, ध्रुव संरचना ggml_tensor * op, size_t & size) override {
        यदि (op->op != GGML_OP_MUL_MAT) {
            प्रत्यागम असत्य;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ध्रुव पूर्णाङ्क slot_गणनम् = kleidiai_collect_kernel_chain(op, kernel_chain);
        यदि (slot_गणनम् == 0) {
            प्रत्यागम असत्य;
        }

        ध्रुव तार्किक is_gemv = op->src[1]->ne[1] == 1;
        ध्रुव size_t k = op->src[0]->ne[0];
        ध्रुव size_t n = op->src[0]->ne[1];
        ध्रुव size_t m = op->src[1]->ne[1];

        यदि (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
            op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K) {
            ध्रुव kleidiai_खण्डम्_तर्काः खण्डम्_तर्काः = kleidiai_get_खण्डम्_तर्काः(kernel_chain[0]->rhs_type);
            ध्रुव size_t qk = खण्डम्_तर्काः.pack_bl;

            size_t cursor = 0;
            तार्किक any_slot = असत्य;

            प्रति (पूर्णाङ्क slot = 0; slot < slot_गणनम्; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;

                यदि (!lhs_info || !lhs_info->packed_आकार_ex || !kernel) {
                    प्रत्यागम असत्य;
                }

                ध्रुव size_t mr = kernel->get_mr();
                ध्रुव size_t kr = kernel->get_kr();
                ध्रुव size_t sr = kernel->get_sr();

                ध्रुव size_t packed = lhs_info->packed_आकार_ex(m, k, qk, mr, kr, sr);

                cursor = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);
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
            kernel_info *           gemv_kernel = primary ? &primary->gemv : शून्यसूचकः;
            यदि (is_gemv && op->src[1]->nb[0] == (int64_t) परिमाण(भिन्न) && gemv_kernel &&
                gemv_kernel->get_lhs_विस्थापनम्_ex && gemv_kernel->get_rhs_packed_विस्थापनम्_ex &&
                gemv_kernel->सततम्_kernel_ex && gemv_kernel->get_dst_विस्थापनम्) {
                size = 0;
                प्रत्यागम सत्य;
            }

            size_t cursor = 0;
            तार्किक any_slot = असत्य;

            प्रति (पूर्णाङ्क slot = 0; slot < slot_गणनम्; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;
                kernel_info * kernel        = &kernels->gemm;

                यदि (!lhs_info || !lhs_info->packed_आकार_ex || !kernel) {
                    प्रत्यागम असत्य;
                }

                ध्रुव size_t mr = kernel->get_mr();
                ध्रुव size_t kr = kernel->get_kr();
                ध्रुव size_t sr = kernel->get_sr();

                cursor  = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_आकार_ex(m, k, 0, mr, kr, sr);
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

            size_t cursor = 0;
            तार्किक any_slot = असत्य;

            प्रति (पूर्णाङ्क slot = 0; slot < slot_गणनम्; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
                kernel_info * kernel        = is_gemv ? &kernels->gemv : &kernels->gemm;
                यदि (!lhs_info || !lhs_info->packed_आकार_ex || !kernels->rhs_info.packed_आकार_ex || !kernel) {
                    प्रत्यागम असत्य;
                }

                ध्रुव size_t mr = kernel->get_mr();
                ध्रुव size_t kr = kernel->get_kr();
                ध्रुव size_t sr = kernel->get_sr();

                cursor  = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += lhs_info->packed_आकार_ex(m * r, k, 0, mr, kr, sr);
                any_slot = सत्य;
            }

            प्रति (पूर्णाङ्क slot = 0; slot < slot_गणनम्; ++slot) {
                ggml_kleidiai_kernels * kernels = kernel_chain[slot];
                kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
                यदि (!kernel || !kernels->rhs_info.packed_आकार_ex) {
                    प्रत्यागम असत्य;
                }
                cursor  = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);
                cursor += kernels->rhs_info.packed_आकार_ex(n, k, kernel->get_nr(), kernel->get_kr(), 0);
            }

            cursor  = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            cursor += k * n * परिमाण(भिन्न);
            cursor  = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);
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

        GGML_TENSOR_द्विमान_OP_LOCALS

        यदि (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }

        ggml_kleidiai_kernels * kernels = kleidiai_primary_kernel_f32();
        यदि (!kernels) {
            प्रत्यागम असत्य;
        }

        ध्रुव size_t k = ne00;
        ध्रुव size_t m = ne11;
        ध्रुव size_t n = ne01;
        ध्रुव तार्किक use_gemv = m == 1 && src1->nb[0] == (int64_t) परिमाण(भिन्न) &&
                              kernels->gemv.get_lhs_विस्थापनम्_ex &&
                              kernels->gemv.get_rhs_packed_विस्थापनम्_ex &&
                              kernels->gemv.सततम्_kernel_ex &&
                              kernels->gemv.get_dst_विस्थापनम्;

        kernel_info * kernel        = use_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = &kernels->gemm_lhs_info;

        यदि (!kernel || !kernel->get_lhs_विस्थापनम्_ex ||
            !kernel->get_rhs_packed_विस्थापनम्_ex || !kernel->सततम्_kernel_ex || !kernel->get_dst_विस्थापनम्) {
            प्रत्यागम असत्य;
        }

        यदि (!use_gemv && (!lhs_info || !lhs_info->get_विस्थापनम् || !lhs_info->get_packed_विस्थापनम्_ex ||
                          !lhs_info->packed_आकार_ex || !lhs_info->pack_func_ex)) {
            प्रत्यागम असत्य;
        }

        ध्रुव kleidiai_weight_शीर्षकम् * शीर्षकम् = kleidiai_weight_शीर्षकम्_from_ptr(src0->दत्त);
        ध्रुव तार्किक has_शीर्षकम् = kleidiai_is_weight_शीर्षकम्_valid(शीर्षकम्);

        ध्रुव uint8_t * rhs_आधारः = has_शीर्षकम् ? kleidiai_weight_slot_ptr(शीर्षकम्, 0)
                                              : static_cast<ध्रुव uint8_t *>(src0->दत्त);
        यदि (!rhs_आधारः) {
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क nth = params->nth > 0 ? params->nth : 1;
        ध्रुव पूर्णाङ्क ith = params->ith;

        ध्रुव size_t mr = kernel->get_mr();
        ध्रुव size_t kr = kernel->get_kr();
        ध्रुव size_t sr = kernel->get_sr();

        ध्रुव size_t lhs_packed_आकार = use_gemv ? 0 : lhs_info->packed_आकार_ex(m, k, 0, mr, kr, sr);
        यदि (!use_gemv) {
            GGML_ASSERT(lhs_packed_आकार <= params->wsize);
        }

        uint8_t * lhs_packed   = static_cast<uint8_t *>(params->wdata);
        ध्रुव size_t dst_stride = dst->nb[1];
        ध्रुव size_t n_चरणः = kernel->get_n_चरणः() ? kernel->get_n_चरणः() : 1;
        ध्रुव तार्किक disable_chunking = ggml_is_numa();
        GGML_ASSERT(n <= (size_t) अधिकतम_पूर्णाङ्कः);

        ध्रुव int64_t batch_आकार = ne12 * ne13;
        प्रति (int64_t batch_idx = 0; batch_idx < batch_आकार; ++batch_idx) {
            ध्रुव int64_t i12 = batch_idx % ne12;
            ध्रुव int64_t i13 = batch_idx / ne12;
            ध्रुव uint8_t * lhs_batch_आधारः = static_cast<ध्रुव uint8_t *>(src1->दत्त) + i12 * src1->nb[2] + i13 * src1->nb[3];
            uint8_t * dst_batch_आधारः = static_cast<uint8_t *>(dst->दत्त) + i12 * dst->nb[2] + i13 * dst->nb[3];

            यदि (!use_gemv) {
                ध्रुव int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)mr);
                int64_t max_तन्तवः = mr ? (m_roundup_mr / (int64_t)mr) : nth;
                max_तन्तवः = std::max<int64_t>(1, max_तन्तवः);
                ध्रुव int64_t use_तन्तवः = std::min<int64_t>(nth, max_तन्तवः);

                यदि (ith < use_तन्तवः) {
                    ध्रुव int64_t अङ्कः_m_प्रत्येकम्_thread0   = आवर्तनम्_अधस्तात्((size_t)(m_roundup_mr / use_तन्तवः), mr);
                    ध्रुव int64_t अङ्कः_m_प्रत्येकम्_thread_N_1 = (int64_t)m - (use_तन्तवः - 1) * अङ्कः_m_प्रत्येकम्_thread0;

                    ध्रुव int64_t m_start = (int64_t)ith * अङ्कः_m_प्रत्येकम्_thread0;
                    ध्रुव int64_t m_गणनम् = (ith == use_तन्तवः - 1) ? अङ्कः_m_प्रत्येकम्_thread_N_1 : अङ्कः_m_प्रत्येकम्_thread0;

                    ध्रुव size_t आधारः_packed_off  = lhs_info->get_packed_विस्थापनम्_ex(m_start, k, 0, mr, kr, sr);
                    ध्रुव size_t अग्र_खण्डम्_off   = lhs_info->get_packed_विस्थापनम्_ex(m_start + mr, k, 0, mr, kr, sr);
                    ध्रुव size_t row_stride_अष्टकानि = mr ? (अग्र_खण्डम्_off - आधारः_packed_off) / mr : 0;

                    int64_t remaining = m_गणनम्;
                    int64_t cur       = m_start;

                    यावत् (remaining > 0) {
                        ध्रुव int64_t take = std::min<int64_t>((int64_t)m - cur, remaining);
                        ध्रुव size_t src_off = lhs_info->get_विस्थापनम्(cur, src1->nb[1]);
                        ध्रुव शून्य * src_ptr = lhs_batch_आधारः + src_off;
                        ध्रुव size_t dst_off = आधारः_packed_off + (size_t)(cur - m_start) * row_stride_अष्टकानि;
                        शून्य * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            यदि (ith == 0) {
                ggml_threadpool_chunk_समुच्चयः(params->threadpool, 0);
            }

            ggml_barrier(params->threadpool);

            ध्रुव size_t chunk_स्तम्भसङ्ख्या = kleidiai_chunk_स्तम्भसङ्ख्या(n, nth, disable_chunking, n_चरणः);
            GGML_ASSERT(chunk_स्तम्भसङ्ख्या <= (size_t) अधिकतम_पूर्णाङ्कः);

            पूर्णाङ्क current_col = ggml_threadpool_chunk_सङ्कलय(params->threadpool, (पूर्णाङ्क) chunk_स्तम्भसङ्ख्या);
            यावत् ((size_t) current_col < n) {
                ध्रुव size_t n_start = (size_t) current_col;
                ध्रुव size_t n_गन्तव्यम्_process = std::min(chunk_स्तम्भसङ्ख्या, n - n_start);

                यदि (n_गन्तव्यम्_process > 0) {
                    ध्रुव size_t lhs_विस्थापनम् = use_gemv ? kernel->get_lhs_विस्थापनम्_ex(0, k, 0)
                                                       : lhs_info->get_packed_विस्थापनम्_ex(0, k, 0, mr, kr, sr);
                    ध्रुव size_t rhs_packed_विस्थापनम् = kernel->get_rhs_packed_विस्थापनम्_ex(n_start, k, 0);
                    ध्रुव size_t dst_विस्थापनम्        = kernel->get_dst_विस्थापनम्(0, n_start, dst_stride);

                    ध्रुव शून्य * lhs_ptr = use_gemv ? lhs_batch_आधारः + lhs_विस्थापनम्
                                                    : lhs_packed + lhs_विस्थापनम्;
                    ध्रुव शून्य * rhs_ptr = rhs_आधारः + rhs_packed_विस्थापनम्;
                    भिन्न * dst_ptr      = reinterpret_cast<भिन्न *>(dst_batch_आधारः + dst_विस्थापनम्);

                    kernel->सततम्_kernel_ex(m, n_गन्तव्यम्_process, k, use_gemv ? src1->nb[1] : 0,
                                          lhs_ptr,
                                          rhs_ptr,
                                          dst_ptr,
                                          dst_stride,
                                          परिमाण(भिन्न),
                                          -FLT_MAX,
                                          FLT_MAX);
                }

                current_col = ggml_threadpool_chunk_सङ्कलय(params->threadpool, (पूर्णाङ्क) chunk_स्तम्भसङ्ख्या);
            }

            यदि (batch_idx != batch_आकार - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        प्रत्यागम सत्य;
    }

    तार्किक compute_forward_fp16(ggml_compute_params * params, संरचना ggml_tensor * dst) {
        ध्रुव ggml_tensor * src0 = dst->src[0];
        ध्रुव ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_द्विमान_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        यदि (!kernels) {
            प्रत्यागम असत्य;
        }

        ध्रुव तार्किक is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);
        यदि (!kernels->rhs_info.pack_func_ex ||
            !kernel->get_lhs_विस्थापनम्_ex || !kernel->get_rhs_packed_विस्थापनम्_ex || !kernel->सततम्_kernel_ex) {
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क nth = params->nth;
        ध्रुव पूर्णाङ्क ith = params->ith;

        ध्रुव int64_t lhs_batch_size0 = ne12;
        ध्रुव int64_t rhs_batch_size0 = ne02;
        ध्रुव int64_t batch_आकार      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        ध्रुव int64_t r = lhs_batch_size0 / rhs_batch_size0;

        ध्रुव int64_t m_गणः = ne11;
        ध्रुव int64_t m       = m_गणः;
        ध्रुव int64_t n       = ne01;
        ध्रुव int64_t k       = ne00;

        ध्रुव size_t lhs_stride = src1->nb[1];
        ध्रुव size_t rhs_stride = src0->nb[1];
        ध्रुव size_t dst_stride = dst->nb[1];

        ध्रुव int64_t mr = (int64_t) kernel->get_mr();
        ध्रुव int64_t nr = (int64_t) kernel->get_nr();
        ध्रुव int64_t kr = (int64_t) kernel->get_kr();
        ध्रुव int64_t sr = (int64_t) kernel->get_sr();

        ध्रुव size_t lhs_packed_आकार = lhs_info->packed_आकार_ex(m, k, 0, mr, kr, sr);
        ध्रुव size_t rhs_packed_आकार = kernels->rhs_info.packed_आकार_ex(n, k, nr, kr, 0);
        ध्रुव size_t kxn_आकार        = k * n * परिमाण(भिन्न);
        ध्रुव size_t bias_आकार       = n * परिमाण(भिन्न);

        ध्रुव size_t wsize_required = lhs_packed_आकार + rhs_packed_आकार + kxn_आकार + bias_आकार;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_आकार;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_आकार;
        uint8_t * bias       = rhs_kxn + kxn_आकार;

        प्रति (int64_t batch_idx = 0; batch_idx < batch_आकार; ++batch_idx) {
            ध्रुव int64_t rhs_batch_idx = batch_idx / r;
            ध्रुव uint8_t * rhs_batch_आधारः = static_cast<ध्रुव uint8_t *>(src0->दत्त) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_आधारः = static_cast<uint8_t *>(dst->दत्त) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                ध्रुव int64_t m_roundup_mr = kai_roundup(m, mr);
                ध्रुव int64_t अङ्कः_तन्तवः  = KAI_MIN(m_roundup_mr / mr, nth);

                यदि (ith < अङ्कः_तन्तवः) {
                    ध्रुव int64_t अङ्कः_m_प्रत्येकम्_thread0   = आवर्तनम्_अधस्तात्((size_t)(m_roundup_mr / अङ्कः_तन्तवः), (size_t)mr);
                    ध्रुव int64_t अङ्कः_m_प्रत्येकम्_thread_N_1 = m - (अङ्कः_तन्तवः - 1) * अङ्कः_m_प्रत्येकम्_thread0;

                    ध्रुव int64_t m_start = ith * अङ्कः_m_प्रत्येकम्_thread0;
                    ध्रुव int64_t m_गणनम् = (ith == अङ्कः_तन्तवः - 1) ? अङ्कः_m_प्रत्येकम्_thread_N_1 : अङ्कः_m_प्रत्येकम्_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    ध्रुव size_t आधारः_packed_off  = lhs_info->get_packed_विस्थापनम्_ex(m_start, k, 0, mr, kr, sr);
                    ध्रुव size_t अग्र_खण्डम्_off   = lhs_info->get_packed_विस्थापनम्_ex(m_start + mr, k, 0, mr, kr, sr);
                    ध्रुव size_t row_stride_अष्टकानि = (अग्र_खण्डम्_off - आधारः_packed_off) / (size_t)mr;

                    int64_t remaining = m_गणनम्;
                    int64_t cur       = m_start;

                    यावत् (remaining > 0) {
                        ध्रुव int64_t row_अन्तर्_गणः = cur;
                        ध्रुव int64_t avail        = m_गणः - row_अन्तर्_गणः;
                        ध्रुव int64_t take         = std::min(avail, remaining);

                        ध्रुव uint8_t * lhs_batch_आधारः = static_cast<ध्रुव uint8_t *>(src1->दत्त) + batch_idx * src1->nb[2];
                        ध्रुव शून्य * src_ptr = lhs_batch_आधारः + (size_t)row_अन्तर्_गणः * lhs_stride;
                        ध्रुव size_t dst_off = आधारः_packed_off + (size_t)(cur - m_start) * row_stride_अष्टकानि;
                        शून्य * dst_ptr       = lhs_packed + dst_off;

                        lhs_info->pack_func_ex(take, k, 0, mr, kr, sr, 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            यदि (ith == 0) {
                स्मृतिं_पूरय(bias, 0, (size_t)n * परिमाण(भिन्न));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<भिन्न *>(rhs_kxn),
                                        reinterpret_cast<ध्रुव uint16_t *>(rhs_batch_आधारः),
                                        rhs_stride);

                kernels->rhs_info.pack_func_ex(1, n, k, nr, kr, sr, 0, n * परिमाण(भिन्न),
                             rhs_kxn, bias, शून्यसूचकः, rhs_packed, 0, शून्यसूचकः);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                ध्रुव int64_t n_चरणः  = (int64_t) kernel->get_n_चरणः();
                int64_t अङ्कः_तन्तवः_n = KAI_MIN(n / n_चरणः, nth);
                यदि (अङ्कः_तन्तवः_n <= 0) {
                    अङ्कः_तन्तवः_n = 1;
                }

                यदि (ith < अङ्कः_तन्तवः_n) {
                    ध्रुव int64_t अङ्कः_n_प्रत्येकम्_thread0   = आवर्तनम्_अधस्तात्((size_t)(n / अङ्कः_तन्तवः_n), (size_t)n_चरणः);
                    ध्रुव int64_t अङ्कः_n_प्रत्येकम्_thread_N_1 = n - (अङ्कः_तन्तवः_n - 1) * अङ्कः_n_प्रत्येकम्_thread0;

                    ध्रुव int64_t n_start      = ith * अङ्कः_n_प्रत्येकम्_thread0;
                    ध्रुव int64_t n_गन्तव्यम्_process = (ith == अङ्कः_तन्तवः_n - 1) ? अङ्कः_n_प्रत्येकम्_thread_N_1 : अङ्कः_n_प्रत्येकम्_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    ध्रुव size_t lhs_packed_offset0 = lhs_info->get_packed_विस्थापनम्_ex(0, k, 0, mr, kr, sr);
                    ध्रुव size_t rhs_packed_विस्थापनम्  = kernel->get_rhs_packed_विस्थापनम्_ex(n_start, k, 0);
                    ध्रुव size_t dst_विस्थापनम्         = kernel->get_dst_विस्थापनम्((size_t)0, (size_t)n_start, dst_stride);

                    ध्रुव शून्य * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    ध्रुव शून्य * rhs_ptr = rhs_packed + rhs_packed_विस्थापनम्;
                    भिन्न * dst_ptr      = reinterpret_cast<भिन्न *>(dst_batch_आधारः + dst_विस्थापनम्);

                    kernel->सततम्_kernel_ex(m, n_गन्तव्यम्_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, परिमाण(भिन्न), -FLT_MAX, FLT_MAX);
                }
            }

            यदि (batch_idx != batch_आकार - 1) {
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

        GGML_TENSOR_द्विमान_OP_LOCALS

        ध्रुव kleidiai_weight_शीर्षकम् * शीर्षकम् = kleidiai_weight_शीर्षकम्_from_ptr(src0->दत्त);
        ध्रुव तार्किक has_शीर्षकम् = kleidiai_is_weight_शीर्षकम्_valid(शीर्षकम्);
        ध्रुव तार्किक is_gemv = src1->ne[1] == 1;
        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ध्रुव पूर्णाङ्क slot_योग = kleidiai_collect_kernel_chain(dst, kernel_chain);

        स्वचालित weight_for_slot = [&](पूर्णाङ्क slot_सूचकांक, size_t & आकार_बहिर्गमनम्) -> ध्रुव uint8_t * {
            यदि (slot_सूचकांक < 0 || slot_सूचकांक >= slot_योग) {
                प्रत्यागम शून्यसूचकः;
            }
            यदि (has_शीर्षकम्) {
                यदि (slot_सूचकांक < शीर्षकम्->slot_गणनम्) {
                    आकार_बहिर्गमनम् = static_cast<size_t>(शीर्षकम्->sizes[slot_सूचकांक]);
                    प्रत्यागम kleidiai_weight_slot_ptr(शीर्षकम्, slot_सूचकांक);
                }
                प्रत्यागम शून्यसूचकः;
            }
            यदि (slot_सूचकांक == 0) {
                आकार_बहिर्गमनम् = ggml_nbytes(src0);
                प्रत्यागम static_cast<ध्रुव uint8_t *>(src0->दत्त);
            }
            प्रत्यागम शून्यसूचकः;
        };

        संरचना runtime_slot {
            पूर्णाङ्क slot_सूचकांक;
            ggml_kleidiai_kernels * kernels;
            kernel_info * kernel;
            lhs_packing_info * lhs_info;
            size_t mr;
            size_t nr;
            size_t kr;
            size_t sr;
            size_t n_चरणः;
            size_t lhs_packed_आकार;
            size_t lhs_विस्थापनम्;
            size_t lhs_bl;
            size_t rhs_bl;
            size_t pack_bl;
            size_t lhs_packed_offset0;
            पूर्णाङ्क assigned_तन्तवः;
            पूर्णाङ्क thread_begin;
            पूर्णाङ्क thread_अन्तः;
            ध्रुव uint8_t * rhs_आधारः;
        };

        std::array<runtime_slot, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> runtime{};
        पूर्णाङ्क runtime_गणनम् = 0;

        प्रति (पूर्णाङ्क slot = 0; slot < slot_योग && runtime_गणनम् < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kinfo      = is_gemv ? &kernels->gemv : &kernels->gemm;
            lhs_packing_info * linfo = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
            यदि (!kinfo || !linfo || !linfo->packed_आकार_ex || !linfo->pack_func_ex || !linfo->get_विस्थापनम् ||
                !kinfo->get_rhs_packed_विस्थापनम्_ex || !kinfo->सततम्_kernel_ex || !kinfo->get_dst_विस्थापनम्) {
                अनुवर्त;
            }

            size_t rhs_आकार = 0;
            ध्रुव uint8_t * rhs_ptr = weight_for_slot(slot, rhs_आकार);
            यदि (!rhs_ptr || rhs_आकार == 0) {
                अनुवर्त;
            }

            ध्रुव kleidiai_खण्डम्_तर्काः खण्डम्_तर्काः = kleidiai_get_खण्डम्_तर्काः(kernels->rhs_type);

            runtime[runtime_गणनम्] = {
                slot,
                kernels,
                kinfo,
                linfo,
                kinfo->get_mr(),
                kinfo->get_nr(),
                kinfo->get_kr(),
                kinfo->get_sr(),
                kinfo->get_n_चरणः(),
                0,
                0,
                खण्डम्_तर्काः.lhs_bl,
                खण्डम्_तर्काः.rhs_bl,
                खण्डम्_तर्काः.pack_bl,
                0,
                0,
                0,
                0,
                rhs_ptr
            };
            ++runtime_गणनम्;
        }

        यदि (runtime_गणनम् == 0) {
            GGML_LOG_WARN("kleidiai: no runtime kernel slot available for supported op %s\n", dst->name);
            प्रत्यागम असत्य;
        }

        ध्रुव पूर्णाङ्क nth_योग = params->nth > 0 ? params->nth : 1;
        ध्रुव पूर्णाङ्क ith_योग = params->ith;

        पूर्णाङ्क sme_slot = -1;
        पूर्णाङ्क non_sme_slot = -1;
        प्रति (पूर्णाङ्क i = 0; i < runtime_गणनम्; ++i) {
            यदि (is_sme_family(runtime[i].kernels->required_cpu)) {
                sme_slot = i;
                भङ्ग;
            }
        }

        प्रति (पूर्णाङ्क i = 0; i < runtime_गणनम्; ++i) {
            यदि (!is_sme_family(runtime[i].kernels->required_cpu)) {
                non_sme_slot = i;
                भङ्ग;
            }
        }

        ध्रुव पूर्णाङ्क sme_cap_सीमा = ctx.sme_thread_cap;
        ध्रुव तार्किक use_hybrid = sme_cap_सीमा > 0 &&
                                 runtime_गणनम् > 1 &&
                                 nth_योग > sme_cap_सीमा;
        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
        // If rows are small or average columns per thread are small, keep single-slot.
        size_t min_स्तम्भसङ्ख्या_प्रत्येकम्_thread = 0;
        यदि (runtime_गणनम् > 0 && nth_योग > 0) {
            min_स्तम्भसङ्ख्या_प्रत्येकम्_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_योग);
        }
        ध्रुव तार्किक too_small_for_hybrid = (min_स्तम्भसङ्ख्या_प्रत्येकम्_thread < 2) || (ne11 < 128);

        ध्रुव तार्किक hybrid_enabled = use_hybrid && !too_small_for_hybrid;

        यदि (!hybrid_enabled) {
            पूर्णाङ्क chosen_slot = 0;
            यदि (too_small_for_hybrid && sme_slot != -1) {
                chosen_slot = nth_योग > sme_cap_सीमा && non_sme_slot != -1 ? non_sme_slot : sme_slot;
            } अन्यथा यदि (runtime_गणनम् > 1 && ctx.sme_thread_cap > 0 && nth_योग > ctx.sme_thread_cap) {
                chosen_slot = 1;
            }
            यदि (chosen_slot != 0 && chosen_slot < runtime_गणनम्) {
                runtime[0] = runtime[chosen_slot];
                runtime[0].assigned_तन्तवः = 0;
                runtime[0].thread_begin = 0;
                runtime[0].thread_अन्तः = 0;
            }
            runtime_गणनम् = runtime_गणनम् > 0 ? 1 : 0;

            // Recompute SME slot based on the collapsed runtime[0]
            sme_slot = -1;
            यदि (runtime_गणनम् > 0 &&
                is_sme_family(runtime[0].kernels->required_cpu)) {
                sme_slot = 0;
            }
        }

        ध्रुव तार्किक sme_single_only = sme_slot != -1 &&
                                     runtime[sme_slot].kernels->rhs_info.repack_बहुलकम् == RHS_REPACK_SINGLE_ONLY;
        पूर्णाङ्क sme_cap = sme_single_only ? nth_योग : kleidiai_sme_thread_cap();
        यदि (sme_cap < 0) {
            sme_cap = nth_योग;
        }
        sme_cap = std::min(sme_cap, nth_योग);

        पूर्णाङ्क तन्तवः_remaining = nth_योग;
        यदि (sme_slot != -1) {
            पूर्णाङ्क sme_तन्तवः = std::min(std::max(sme_cap, 0), तन्तवः_remaining);
            runtime[sme_slot].assigned_तन्तवः = sme_तन्तवः;
            तन्तवः_remaining -= sme_तन्तवः;
        }

        पूर्णाङ्क fallback_indices[GGML_KLEIDIAI_MAX_KERNEL_SLOTS];
        पूर्णाङ्क fallback_गणनम् = 0;
        // The current hybrid chain is bounded to SME + one non-SME fallback slot.
        GGML_ASSERT(GGML_KLEIDIAI_MAX_KERNEL_SLOTS == 2);
        प्रति (पूर्णाङ्क i = 0; i < runtime_गणनम्; ++i) {
            यदि (i == sme_slot) {
                अनुवर्त;
            }
            fallback_indices[fallback_गणनम्++] = i;
        }

        प्रति (पूर्णाङ्क fi = 0; fi < fallback_गणनम्; ++fi) {
            यदि (तन्तवः_remaining <= 0) {
                भङ्ग;
            }
            ध्रुव पूर्णाङ्क slot_सूचकांक = fallback_indices[fi];
            ध्रुव पूर्णाङ्क slots_वामः = fallback_गणनम् - fi;
            पूर्णाङ्क share = (तन्तवः_remaining + slots_वामः - 1) / slots_वामः;
            share     = std::min(share, तन्तवः_remaining);
            runtime[slot_सूचकांक].assigned_तन्तवः = share;
            तन्तवः_remaining -= share;
        }

        पूर्णाङ्क thread_cursor = 0;
        प्रति (पूर्णाङ्क i = 0; i < runtime_गणनम्; ++i) {
            runtime[i].thread_begin = thread_cursor;
            thread_cursor += runtime[i].assigned_तन्तवः;
            runtime[i].thread_अन्तः = thread_cursor;
        }
        GGML_ASSERT(thread_cursor > 0);
        ध्रुव पूर्णाङ्क सक्रिय_तन्तवः = thread_cursor;

        पूर्णाङ्क local_slot = -1;
        पूर्णाङ्क local_ith  = 0;
        प्रति (पूर्णाङ्क i = 0; i < runtime_गणनम्; ++i) {
            यदि (ith_योग >= runtime[i].thread_begin && ith_योग < runtime[i].thread_अन्तः) {
                local_slot = i;
                local_ith  = ith_योग - runtime[i].thread_begin;
                भङ्ग;
            }
        }
        ध्रुव size_t k = ne00;
        ध्रुव size_t m = ne11;
        ध्रुव size_t n = ne01;

        size_t cursor = 0;
        प्रति (पूर्णाङ्क i = 0; i < runtime_गणनम्; ++i) {
            runtime[i].lhs_packed_आकार = runtime[i].lhs_info->packed_आकार_ex(m, k, runtime[i].pack_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);
            runtime[i].lhs_विस्थापनम् = cursor;
            runtime[i].lhs_packed_offset0 = runtime[i].lhs_info->get_packed_विस्थापनम्_ex(0, k, runtime[i].lhs_bl, runtime[i].mr, runtime[i].kr, runtime[i].sr);
            cursor += runtime[i].lhs_packed_आकार;
        }

        GGML_ASSERT(cursor <= params->wsize);
        uint8_t * scratch = static_cast<uint8_t *>(params->wdata);

        size_t common_चरणः = 1;
        प्रति (पूर्णाङ्क i = 0; i < runtime_गणनम्; ++i) {
            यदि (runtime[i].assigned_तन्तवः == 0) {
                अनुवर्त;
            }
            size_t अग्र_चरणः = 0;
            यदि (!lcm_आकार(common_चरणः, runtime[i].n_चरणः ? runtime[i].n_चरणः : 1, अग्र_चरणः)) {
                प्रत्यागम असत्य;
            }
            common_चरणः = अग्र_चरणः;
        }
        GGML_ASSERT(common_चरणः > 0);

        ध्रुव तार्किक disable_chunking = ggml_is_numa();
        ध्रुव size_t chunk_multiplier = std::max(1, ctx.chunk_multiplier);
        ध्रुव size_t chunk_भाजकः = (सक्रिय_तन्तवः == 1 || disable_chunking) ? (size_t) सक्रिय_तन्तवः : (size_t) सक्रिय_तन्तवः * chunk_multiplier;
        size_t chunk_स्तम्भसङ्ख्या = align_ऊर्ध्वम्(std::max<size_t>(1, ceil_div_आकार(n, chunk_भाजकः)), common_चरणः);
        यदि (chunk_स्तम्भसङ्ख्या == 0) {
            chunk_स्तम्भसङ्ख्या = common_चरणः;
        }
        // If common_step is larger than n, the loop below runs one valid tail chunk
        // with cols == n.
        ध्रुव size_t nchunk_आकार = std::max<size_t>(1, ceil_div_आकार(n, chunk_स्तम्भसङ्ख्या));
        GGML_ASSERT(nchunk_आकार <= (size_t)अधिकतम_पूर्णाङ्कः);
        ध्रुव पूर्णाङ्क nchunk = (पूर्णाङ्क)nchunk_आकार;
        ध्रुव size_t dst_stride = dst->nb[1];

        स्वचालित सततम्_chunk = [&](runtime_slot & slot, size_t global_start, size_t स्तम्भसङ्ख्या_2, uint8_t * dst_batch_आधारः) {
            ध्रुव size_t rhs_packed_विस्थापनम् = slot.kernel->get_rhs_packed_विस्थापनम्_ex(global_start, k, slot.rhs_bl);
            ध्रुव size_t dst_विस्थापनम्        = slot.kernel->get_dst_विस्थापनम्(0, global_start, dst_stride);

            ध्रुव uint8_t * lhs_ptr = scratch + slot.lhs_विस्थापनम् + slot.lhs_packed_offset0;
            ध्रुव uint8_t * rhs_ptr = slot.rhs_आधारः + rhs_packed_विस्थापनम्;
            भिन्न * dst_ptr         = reinterpret_cast<भिन्न *>(dst_batch_आधारः + dst_विस्थापनम्);

            slot.kernel->सततम्_kernel_ex(m, स्तम्भसङ्ख्या_2, k, slot.rhs_bl,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride,
                                       परिमाण(भिन्न),
                                       -FLT_MAX,
                                       FLT_MAX);
        };

        प्रति (int64_t batch_idx = 0; batch_idx < ne12; ++batch_idx) {
            ध्रुव uint8_t * lhs_batch_आधारः = static_cast<ध्रुव uint8_t *>(src1->दत्त) + batch_idx * src1->nb[2];
            uint8_t * dst_batch_आधारः = static_cast<uint8_t *>(dst->दत्त) + batch_idx * dst->nb[2];

            यदि (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                ध्रुव int64_t m_roundup_mr = kai_roundup((int64_t)m, (int64_t)slot.mr);
                int64_t max_तन्तवः = slot.mr ? (m_roundup_mr / (int64_t)slot.mr) : slot.assigned_तन्तवः;
                max_तन्तवः = std::max<int64_t>(1, max_तन्तवः);
                ध्रुव int64_t use_तन्तवः = std::min<int64_t>(slot.assigned_तन्तवः, max_तन्तवः);

                यदि (local_ith < use_तन्तवः) {
                    ध्रुव int64_t अङ्कः_m_प्रत्येकम्_thread0   = आवर्तनम्_अधस्तात्((size_t)(m_roundup_mr / use_तन्तवः), slot.mr);
                    ध्रुव int64_t अङ्कः_m_प्रत्येकम्_thread_N_1 = (int64_t)m - (use_तन्तवः - 1) * अङ्कः_m_प्रत्येकम्_thread0;

                    ध्रुव int64_t m_start = (int64_t)local_ith * अङ्कः_m_प्रत्येकम्_thread0;
                    ध्रुव int64_t m_गणनम् = (local_ith == use_तन्तवः - 1) ? अङ्कः_m_प्रत्येकम्_thread_N_1 : अङ्कः_m_प्रत्येकम्_thread0;

                    ध्रुव size_t आधारः_packed_off  = slot.lhs_info->get_packed_विस्थापनम्_ex(m_start, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ध्रुव size_t अग्र_खण्डम्_off   = slot.lhs_info->get_packed_विस्थापनम्_ex(m_start + slot.mr, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr);
                    ध्रुव size_t row_stride_अष्टकानि = slot.mr ? (अग्र_खण्डम्_off - आधारः_packed_off) / slot.mr : 0;

                    int64_t remaining = m_गणनम्;
                    int64_t cur       = m_start;

                    uint8_t * lhs_packed = scratch + slot.lhs_विस्थापनम्;
                    यावत् (remaining > 0) {
                        ध्रुव int64_t row_अन्तर्_गणः = cur;
                        ध्रुव int64_t avail        = (int64_t)m - row_अन्तर्_गणः;
                        ध्रुव int64_t take         = std::min(avail, remaining);

                        ध्रुव size_t src_off = slot.lhs_info->get_विस्थापनम्(row_अन्तर्_गणः, src1->nb[1]);
                        ध्रुव शून्य * src_ptr = lhs_batch_आधारः + src_off;
                        ध्रुव size_t dst_off = आधारः_packed_off + (size_t)(cur - m_start) * row_stride_अष्टकानि;
                        शून्य * dst_ptr       = lhs_packed + dst_off;

                        slot.lhs_info->pack_func_ex(take, k, slot.lhs_bl, slot.mr, slot.kr, slot.sr, 0, src_ptr, src1->nb[1], dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            यदि (ith_योग == 0) {
                ggml_threadpool_chunk_समुच्चयः(params->threadpool, सक्रिय_तन्तवः);
            }

            // Publishes both LHS packing and the initialized dynamic chunk queue.
            ggml_barrier(params->threadpool);

            यदि (local_slot != -1) {
                runtime_slot & slot = runtime[local_slot];
                पूर्णाङ्क current_chunk = ith_योग;
                यावत् (current_chunk < nchunk) {
                    ध्रुव size_t global_start = (size_t)current_chunk * chunk_स्तम्भसङ्ख्या;
                    यदि (global_start >= n) {
                        भङ्ग;
                    }

                    ध्रुव size_t स्तम्भसङ्ख्या_2 = std::min(chunk_स्तम्भसङ्ख्या, n - global_start);
                    यदि (स्तम्भसङ्ख्या_2 > 0) {
                        // KleidiAI GEMM/GEMV kernels accept arbitrary final tail widths;
                        // only non-tail chunks are guaranteed to be n_step-aligned.
                        सततम्_chunk(slot, global_start, स्तम्भसङ्ख्या_2, dst_batch_आधारः);
                    }

                    current_chunk = ggml_threadpool_chunk_सङ्कलय(params->threadpool, 1);
                }
            }

            यदि (batch_idx != ne12 - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        प्रत्यागम सत्य;
    }

सार्वजनिक:
    पूर्णाङ्क repack(संरचना ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t दत्त_आकार) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_K ||
                    tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K ||
                    tensor->type == GGML_TYPE_F32);
        ध्रुव size_t n = tensor->ne[1];
        ध्रुव size_t k = tensor->ne[0];

        kleidiai_weight_शीर्षकम् * शीर्षकम् = kleidiai_weight_शीर्षकम्_from_ptr(tensor->दत्त);
        यदि (!शीर्षकम्) {
            प्रत्यागम -1;
        }

        शीर्षकम्->magic      = GGML_KLEIDIAI_PACK_MAGIC;
        शीर्षकम्->संस्करणम्_2    = GGML_KLEIDIAI_PACK_संस्करणम्;
        शीर्षकम्->slot_गणनम् = 0;

        uint8_t * आधारः_ptr = static_cast<uint8_t *>(tensor->दत्त);
        size_t cursor = परिमाण(kleidiai_weight_शीर्षकम्);
        cursor = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ध्रुव तार्किक want_q4k = tensor->type == GGML_TYPE_Q4_K;
        ध्रुव तार्किक want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
        ध्रुव तार्किक want_f32 = tensor->type == GGML_TYPE_F32;
        ध्रुव पूर्णाङ्क slot_योग = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                        : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                        : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                                   : kleidiai_collect_q4_chain(kernel_chain);
        ध्रुव तार्किक allow_fallback = kleidiai_pack_fallback_allowed();

        std::vector<int8_t> qdata;
        std::vector<भिन्न>  scales;
        std::vector<भिन्न>  bias;
        std::vector<भिन्न> row_f32;

        यदि (want_q8 && slot_योग > 0) {
            qdata.resize(n * k, 0);
            scales.resize(n, 0.0f);

            ध्रुव size_t row_stride = tensor->nb[1];
            ध्रुव size_t k_खण्डानि   = (k + QK8_0 - 1) / QK8_0;

            यदि (tensor->type == GGML_TYPE_Q6_K) {
                यदि (k % QK_K != 0 || k % QK8_0 != 0) {
                    प्रत्यागम -1;
                }
                row_f32.resize(k);
            }

            प्रति (size_t row = 0; row < n; ++row) {
                ध्रुव uint8_t * row_दत्त = static_cast<ध्रुव uint8_t *>(दत्त) + row * row_stride;

                यदि (tensor->type == GGML_TYPE_Q6_K) {
                    dequantize_row_q6_K(reinterpret_cast<ध्रुव खण्डम्_q6_K *>(row_दत्त), row_f32.दत्त(), k);
                }

                भिन्न max_abs = 0.0f;
                यदि (tensor->type == GGML_TYPE_Q6_K) {
                    प्रति (size_t रैखिकम्_idx = 0; रैखिकम्_idx < k; ++रैखिकम्_idx) {
                        max_abs = std::max(max_abs, std::fabs(row_f32[रैखिकम्_idx]));
                    }
                } अन्यथा {
                    ध्रुव खण्डम्_q8_0 * row_खण्डानि = reinterpret_cast<ध्रुव खण्डम्_q8_0 *>(row_दत्त);
                    प्रति (size_t खण्डम् = 0; खण्डम् < k_खण्डानि; ++खण्डम्) {
                        ध्रुव खण्डम्_q8_0 & blk = row_खण्डानि[खण्डम्];
                        ध्रुव भिन्न d = GGML_FP16_गन्तव्यम्_FP32(blk.d);
                        प्रति (size_t l = 0; l < QK8_0; ++l) {
                            ध्रुव size_t रैखिकम्_idx = खण्डम् * QK8_0 + l;
                            यदि (रैखिकम्_idx >= k) {
                                भङ्ग;
                            }
                            ध्रुव भिन्न value = d * static_cast<भिन्न>(blk.qs[l]);
                            max_abs = std::max(max_abs, std::fabs(value));
                        }
                    }
                }

                भिन्न scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
                scales[row] = scale;
                ध्रुव भिन्न inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

                यदि (tensor->type == GGML_TYPE_Q6_K) {
                    प्रति (size_t रैखिकम्_idx = 0; रैखिकम्_idx < k; ++रैखिकम्_idx) {
                        int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(row_f32[रैखिकम्_idx] * inv_scale)) : 0;
                        q = std::clamp(q, -127, 127);
                        qdata[row * k + रैखिकम्_idx] = static_cast<int8_t>(q);
                    }
                } अन्यथा {
                    ध्रुव खण्डम्_q8_0 * row_खण्डानि = reinterpret_cast<ध्रुव खण्डम्_q8_0 *>(row_दत्त);
                    प्रति (size_t खण्डम् = 0; खण्डम् < k_खण्डानि; ++खण्डम्) {
                        ध्रुव खण्डम्_q8_0 & blk = row_खण्डानि[खण्डम्];
                        ध्रुव भिन्न d = GGML_FP16_गन्तव्यम्_FP32(blk.d);
                        प्रति (size_t l = 0; l < QK8_0; ++l) {
                            ध्रुव size_t रैखिकम्_idx = खण्डम् * QK8_0 + l;
                            यदि (रैखिकम्_idx >= k) {
                                भङ्ग;
                            }
                            ध्रुव भिन्न value = d * static_cast<भिन्न>(blk.qs[l]);
                            int32_t q = scale > 0.0f ? static_cast<int32_t>(std::lround(value * inv_scale)) : 0;
                            q = std::clamp(q, -127, 127);
                            qdata[row * k + रैखिकम्_idx] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }

        यदि (want_f32 && slot_योग > 0) {
            bias.resize(n, 0.0f);
        }

        प्रति (पूर्णाङ्क slot = 0; slot < slot_योग && slot < GGML_KLEIDIAI_MAX_KERNEL_SLOTS; ++slot) {
            यदि (!allow_fallback && slot > 0) {
                भङ्ग;
            }
            ggml_kleidiai_kernels * kernels = kernel_chain[slot];
            kernel_info * kernel = &kernels->gemm;
            rhs_packing_info * rhs_info = &kernels->rhs_info;
            यदि (!rhs_info || !rhs_info->pack_func_ex || !rhs_info->packed_आकार_ex || !kernel) {
                अनुवर्त;
            }

            ध्रुव size_t nr = kernel->get_nr();
            ध्रुव size_t kr = kernel->get_kr();
            ध्रुव size_t sr = kernel->get_sr();
            ध्रुव ggml_type rhs_type = kernels->rhs_type;
            यदि (rhs_type == GGML_TYPE_Q8_0 && (kernels->required_cpu & CPU_FEATURE_SVE)) {
                GGML_ASSERT(nr == kernels->gemv.get_nr());
                GGML_ASSERT(kr == kernels->gemv.get_kr());
                GGML_ASSERT(sr == kernels->gemv.get_sr());
            }
            ध्रुव size_t खण्डम्_len = rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                     rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                     rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                     rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
            यदि (खण्डम्_len == SIZE_MAX) {
                अनुवर्त;
            }

            ध्रुव size_t packed_आकार = rhs_info->packed_आकार_ex(n, k, nr, kr, खण्डम्_len);
            ध्रुव size_t aligned_cursor = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);

            uint8_t * dst_ptr = आधारः_ptr + aligned_cursor;

            यदि (rhs_type == GGML_TYPE_Q4_0) {
                संरचना kai_rhs_pack_qs4cxs1s0_param params;
                params.lhs_zero_point = 1;
                params.rhs_zero_point = 8;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, QK4_0, 0,
                                       static_cast<ध्रुव uint8_t *>(दत्त), शून्यसूचकः, शून्यसूचकः,
                                       dst_ptr, 0, &params);
            } अन्यथा यदि (rhs_type == GGML_TYPE_Q4_K) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, GGML_KLEIDIAI_Q4_K_BL, tensor->nb[1],
                                       दत्त, शून्यसूचकः, शून्यसूचकः, dst_ptr, 0, शून्यसूचकः);
            } अन्यथा यदि (rhs_type == GGML_TYPE_Q8_0) {
                संरचना kai_rhs_pack_qsi8cx_params params;
                params.lhs_zero_point = 1;
                params.scale_multiplier = 1.0f;
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, 0,
                                       qdata.दत्त(), शून्यसूचकः, scales.दत्त(),
                                       dst_ptr, 0, &params);
            } अन्यथा यदि (rhs_type == GGML_TYPE_F32) {
                rhs_info->pack_func_ex(1, n, k, nr, kr, sr, 0, tensor->nb[1],
                                       दत्त, bias.दत्त(), शून्यसूचकः,
                                       dst_ptr, 0, शून्यसूचकः);
            } अन्यथा {
                अनुवर्त;
            }

            शीर्षकम्->offsets[शीर्षकम्->slot_गणनम्] = aligned_cursor;
            शीर्षकम्->sizes[शीर्षकम्->slot_गणनम्]   = packed_आकार;
            ++शीर्षकम्->slot_गणनम्;

            cursor = aligned_cursor + packed_आकार;
        }

        यदि (शीर्षकम्->slot_गणनम् == 0) {
            शीर्षकम्->magic   = 0;
            शीर्षकम्->संस्करणम्_2 = 0;
            स्मृतिं_प्रतिलिख(tensor->दत्त, दत्त, दत्त_आकार);
        }

        प्रत्यागम 0;
    }
};

स्थिर ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_बफर_t, संरचना ggml_tensor *) {
    स्थिर tensor_traits traits;
    प्रत्यागम &traits;
}
}  // namespace ggml::cpu::kleidiai

स्थिर गणना ggml_स्थितिः ggml_backend_cpu_kleidiai_बफर_init_tensor(ggml_backend_बफर_t बफर, संरचना ggml_tensor * tensor) {
    tensor->extra = (शून्य *) ggml::cpu::kleidiai::get_tensor_traits(बफर, tensor);

    प्रत्यागम GGML_स्थितिः_SUCCESS;
}

स्थिर शून्य ggml_backend_cpu_kleidiai_बफर_समुच्चयः_tensor(ggml_backend_बफर_t बफर, संरचना ggml_tensor * tensor,
                                                       ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    GGML_ASSERT(विस्थापनम् == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    स्वचालित tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    स्वचालित OK            = tensor_traits->repack(tensor, दत्त, size);

    GGML_ASSERT(OK == 0);
    GGML_अप्रयुक्तम्(बफर);
}

स्थिर ध्रुव वर्ण * ggml_backend_cpu_kleidiai_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    GGML_अप्रयुक्तम्(buft);
    प्रत्यागम "CPU_KLEIDIAI";
}

स्थिर ggml_backend_बफर_t ggml_backend_cpu_kleidiai_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, size_t size) {
    ggml_backend_बफर_t बफर = ggml_backend_buft_alloc_बफर(ggml_backend_cpu_बफर_type(), size);

    यदि (बफर == शून्यसूचकः) {
        प्रत्यागम शून्यसूचकः;
    }

    बफर->buft              = buft;
    बफर->iface.init_tensor = ggml_backend_cpu_kleidiai_बफर_init_tensor;
    बफर->iface.समुच्चयः_tensor  = ggml_backend_cpu_kleidiai_बफर_समुच्चयः_tensor;
    बफर->iface.get_tensor  = शून्यसूचकः;
    बफर->iface.cpy_tensor  = शून्यसूचकः;
    प्रत्यागम बफर;
}

स्थिर size_t ggml_backend_cpu_kleidiai_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    GGML_अप्रयुक्तम्(buft);
    प्रत्यागम TENSOR_ALIGNMENT;
}

स्थिर size_t ggml_backend_cpu_kleidiai_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, ध्रुव संरचना ggml_tensor * tensor) {
    GGML_अप्रयुक्तम्(buft);

    यदि (tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_Q4_K &&
        tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_Q6_K &&
        tensor->type != GGML_TYPE_F32) {
        प्रत्यागम ggml_nbytes(tensor);
    }

    ध्रुव size_t n = tensor->ne[1];
    ध्रुव size_t k = tensor->ne[0];

    size_t cursor = परिमाण(kleidiai_weight_शीर्षकम्);
    cursor = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);

    std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
    ध्रुव तार्किक want_q4k = tensor->type == GGML_TYPE_Q4_K;
    ध्रुव तार्किक want_q8 = tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q6_K;
    ध्रुव तार्किक want_f32 = tensor->type == GGML_TYPE_F32;
    ध्रुव पूर्णाङ्क slot_योग = want_f32 ? kleidiai_collect_f32_chain(kernel_chain)
                                    : want_q8 ? kleidiai_collect_q8_chain(kernel_chain)
                                    : want_q4k ? kleidiai_collect_q4k_chain(kernel_chain)
                                               : kleidiai_collect_q4_chain(kernel_chain);
    ध्रुव तार्किक allow_fallback = kleidiai_pack_fallback_allowed();

    size_t slot_गणनम् = 0;
    प्रति (पूर्णाङ्क slot = 0; slot < slot_योग; ++slot) {
        यदि (!allow_fallback && slot > 0) {
            भङ्ग;
        }
        ggml_kleidiai_kernels * kernels = kernel_chain[slot];
        यदि (!kernels) {
            अनुवर्त;
        }
        kernel_info * kernel = &kernels->gemm;
        rhs_packing_info * rhs_info = &kernels->rhs_info;
        यदि (!kernel || !rhs_info || !rhs_info->packed_आकार_ex) {
            अनुवर्त;
        }

        ध्रुव ggml_type rhs_type = kernels->rhs_type;
        ध्रुव size_t खण्डम्_len = rhs_type == GGML_TYPE_Q4_0 ? QK4_0 :
                                 rhs_type == GGML_TYPE_Q4_K ? GGML_KLEIDIAI_Q4_K_BL :
                                 rhs_type == GGML_TYPE_Q8_0 ? QK8_0 :
                                 rhs_type == GGML_TYPE_F32 ? 0 : SIZE_MAX;
        यदि (खण्डम्_len == SIZE_MAX) {
            अनुवर्त;
        }

        cursor = align_ऊर्ध्वम्(cursor, GGML_KLEIDIAI_PACK_ALIGN);
        cursor += rhs_info->packed_आकार_ex(n, k, kernel->get_nr(), kernel->get_kr(), खण्डम्_len);
        ++slot_गणनम्;
    }

    यदि (slot_गणनम् == 0) {
        प्रत्यागम ggml_nbytes(tensor);
    }

    प्रत्यागम std::max(cursor, ggml_nbytes(tensor));
}

नामाकाश ggml::cpu::kleidiai {
वर्ग extra_बफर_type : ggml::cpu::extra_बफर_type {
    तार्किक supports_op(ggml_backend_dev_t, ध्रुव संरचना ggml_tensor * op) override {
        // GET_ROWS needs the original weights in a native buffer.
        यदि (op->op != GGML_OP_MUL_MAT) {
            प्रत्यागम असत्य;
        }

        std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
        ध्रुव पूर्णाङ्क slot_योग = kleidiai_collect_kernel_chain(op, kernel_chain);
        ध्रुव तार्किक src0_is_kleidiai =
            op->src[0]->बफर &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->बफर->buft->context == एतत् &&
            slot_योग > 0;

        यदि ((op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_F32) &&
            src0_is_kleidiai) {
            यदि (op->src[0]->type == GGML_TYPE_Q4_0 && ctx.kernels_q4 == शून्यसूचकः) {
                प्रत्यागम असत्य;
            }
            यदि (op->src[0]->type == GGML_TYPE_Q8_0 && ctx.kernels_q8 == शून्यसूचकः) {
                प्रत्यागम असत्य;
            }
            यदि (op->src[0]->type == GGML_TYPE_Q6_K && ctx.kernels_q8 == शून्यसूचकः) {
                प्रत्यागम असत्य;
            }
            यदि (op->src[0]->type == GGML_TYPE_Q4_K && ctx.kernels_q4k == शून्यसूचकः) {
                प्रत्यागम असत्य;
            }
            यदि (op->src[0]->type == GGML_TYPE_F32 && ctx.kernels_f32 == शून्यसूचकः) {
                प्रत्यागम असत्य;
            }
            यदि (op->src[1]->बफर && !ggml_backend_buft_is_host(op->src[1]->बफर->buft)) {
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
            यदि (op->src[0]->बफर && op->src[0]->बफर->buft->context == एतत्) {
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
                    प्रत्यागम शून्यसूचकः;
                }
                std::array<ggml_kleidiai_kernels *, GGML_KLEIDIAI_MAX_KERNEL_SLOTS> kernel_chain;
                ध्रुव पूर्णाङ्क slot_योग = kleidiai_collect_kernel_chain(op, kernel_chain);
                यदि (slot_योग > 0 && op->src[1]->ne[1] > 1) {
                    यदि ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                        (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                        प्रत्यागम शून्यसूचकः;
                    }
                    प्रत्यागम ggml::cpu::kleidiai::get_tensor_traits(रिक्त, रिक्त);
                }
            }
        }
        प्रत्यागम शून्यसूचकः;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_बफर_type_t ggml_backend_cpu_kleidiai_बफर_type(शून्य) {
    स्थिर ggml::cpu::kleidiai::extra_बफर_type ctx;
    स्थिर संरचना ggml_backend_बफर_type ggml_backend_cpu_बफर_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_kleidiai_बफर_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_kleidiai_बफर_type_alloc_बफर,
                           /* .alloc_buffer_n       = */ शून्यसूचकः,
                           /* .get_alignment        = */ ggml_backend_cpu_kleidiai_बफर_type_get_alignment,
                           /* .get_max_size         = */ शून्यसूचकः,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_kleidiai_बफर_type_get_alloc_आकार,
                           /* .get_alloc_size_n     = */ रिक्त,
                           /* .is_host              = */ शून्यसूचकः,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    प्रत्यागम &ggml_backend_cpu_बफर_type_kleidiai;
}
