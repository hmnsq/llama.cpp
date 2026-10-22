#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "repack.h"
#ಅನ್ತರ್ಭಾವ "traits.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "amx/amx.h"

#ಅನ್ತರ್ಭಾವ <cctype>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <vector>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_HBM
#    include "hbm.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(_WIN32)
#    define WIN32_LEAN_AND_ಸರಾಸರಿ
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#ಇಲ್ಲದಿದ್ದರೆ
#    include <unistd.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#ಆದರೆ_ಅಂತ್ಯ

// ggml-backend interface

std::vector<ggml_backend_ಬಫರ್_type_t> & ggml_backend_cpu_get_extra_ಬಫರ್_types() {
    ಸ್ಥಿರ std::vector<ggml_backend_ಬಫರ್_type_t> bufts = []() {
        std::vector<ggml_backend_ಬಫರ್_type_t> bufts;

#ಆದರೆ defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        ಆದರೆ (ggml_backend_amx_ಬಫರ್_type()) {
            bufts.push_back(ggml_backend_amx_ಬಫರ್_type());
        }
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_RISCV64_SPACEMIT
        ಆದರೆ (ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_ಬಫರ್_type());
        }
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_KLEIDIAI
        ಆದರೆ (ggml_backend_cpu_kleidiai_ಬಫರ್_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_ಬಫರ್_type());
        }
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_REPACK
        ಆದರೆ (ggml_backend_cpu_repack_ಬಫರ್_type()) {
            bufts.push_back(ggml_backend_cpu_repack_ಬಫರ್_type());
        }
#ಆದರೆ_ಅಂತ್ಯ

        ಹಿಂತಿರುಗಿ bufts;
    }();

    ಹಿಂತಿರುಗಿ bufts;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    ಸ್ಥಿರ std::vector<ggml_backend_ಬಫರ್_type_t> extra_bufts = [] {
        std::vector<ggml_backend_ಬಫರ್_type_t> bufts = ggml_backend_cpu_get_extra_ಬಫರ್_types();
        bufts.push_back(ಶೂನ್ಯಸೂಚಕ);
        ಹಿಂತಿರುಗಿ bufts;
    }();

    ಹಿಂತಿರುಗಿ extra_bufts.ದತ್ತಾಂಶ();

    GGML_ಬಳಸದ(device);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cpu_is_extra_ಬಫರ್_type(ggml_backend_ಬಫರ್_type_t buft) {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ * extra : ggml_backend_cpu_get_extra_ಬಫರ್_types()) {
        ಆದರೆ (extra == buft) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

// CPU backend - backend (stream)

ರಚನೆ ggml_backend_cpu_context {
    ಪೂರ್ಣಾಂಕ                 n_ಎಳೆಗಳು;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_ದತ್ತಾಂಶ;
    size_t              work_ಗಾತ್ರ;

    ggml_abort_callback abort_callback;
    ಶೂನ್ಯ *              abort_callback_ದತ್ತಾಂಶ;

    ತಾರ್ಕಿಕ                use_ref;  // use reference implementation
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ "CPU";

    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_free(ggml_backend_t backend) {
    ರಚನೆ ggml_backend_cpu_context * cpu_ctx = (ರಚನೆ ggml_backend_cpu_context *)backend->context;
    ಅಳಿಸು[] cpu_ctx->work_ದತ್ತಾಂಶ;
    ಅಳಿಸು cpu_ctx;
    ಅಳಿಸು backend;
}

ರಚನೆ ggml_backend_plan_cpu {
    ರಚನೆ ggml_cplan cplan;
    ರಚನೆ ggml_cgraph cgraph;
};

ಸ್ಥಿರ ggml_backend_ನಕ್ಷೆ_plan_t ggml_backend_cpu_ನಕ್ಷೆ_plan_create(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph) {
    ರಚನೆ ggml_backend_cpu_context * cpu_ctx = (ರಚನೆ ggml_backend_cpu_context *)backend->context;

    ರಚನೆ ggml_backend_plan_cpu * cpu_plan = ಹೊಸ ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_ನಕ್ಷೆ_plan(cgraph, cpu_ctx->n_ಎಳೆಗಳು, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    ಆದರೆ (cpu_plan->cplan.work_ಗಾತ್ರ > 0) {
        cpu_plan->cplan.work_ದತ್ತಾಂಶ = ಹೊಸ uint8_t[cpu_plan->cplan.work_ಗಾತ್ರ];
        ಆದರೆ (cpu_plan->cplan.work_ದತ್ತಾಂಶ == ಖಾಲಿ) {
            ಅಳಿಸು cpu_plan;
            ಹಿಂತಿರುಗಿ ಖಾಲಿ;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_ದತ್ತಾಂಶ = cpu_ctx->abort_callback_ದತ್ತಾಂಶ;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    ಹಿಂತಿರುಗಿ cpu_plan;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_ನಕ್ಷೆ_plan_free(ggml_backend_t backend, ggml_backend_ನಕ್ಷೆ_plan_t plan) {
    ರಚನೆ ggml_backend_plan_cpu * cpu_plan = (ರಚನೆ ggml_backend_plan_cpu *)plan;

    ಅಳಿಸು[] cpu_plan->cplan.work_ದತ್ತಾಂಶ;
    ಅಳಿಸು cpu_plan;

    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_cpu_ನಕ್ಷೆ_plan_compute(ggml_backend_t backend, ggml_backend_ನಕ್ಷೆ_plan_t plan) {
    ರಚನೆ ggml_backend_plan_cpu * cpu_plan = (ರಚನೆ ggml_backend_plan_cpu *)plan;

    ಹಿಂತಿರುಗಿ ggml_ನಕ್ಷೆ_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_cpu_ನಕ್ಷೆ_compute(ggml_backend_t backend, ರಚನೆ ggml_cgraph * cgraph) {
    ರಚನೆ ggml_backend_cpu_context * cpu_ctx = (ರಚನೆ ggml_backend_cpu_context *)backend->context;

    ರಚನೆ ggml_cplan cplan = ggml_ನಕ್ಷೆ_plan(cgraph, cpu_ctx->n_ಎಳೆಗಳು, cpu_ctx->threadpool);

    ಆದರೆ (cpu_ctx->work_ಗಾತ್ರ < cplan.work_ಗಾತ್ರ) {
        ಅಳಿಸು[] cpu_ctx->work_ದತ್ತಾಂಶ;
        cpu_ctx->work_ದತ್ತಾಂಶ = ಹೊಸ uint8_t[cplan.work_ಗಾತ್ರ];
        ಆದರೆ (cpu_ctx->work_ದತ್ತಾಂಶ == ಖಾಲಿ) {
            cpu_ctx->work_ಗಾತ್ರ = 0;
            ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_ALLOC_FAILED;
        }
        cpu_ctx->work_ಗಾತ್ರ = cplan.work_ಗಾತ್ರ;
    }
    cplan.work_ದತ್ತಾಂಶ = (uint8_t *)cpu_ctx->work_ದತ್ತಾಂಶ;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_ದತ್ತಾಂಶ = cpu_ctx->abort_callback_ದತ್ತಾಂಶ;
    cplan.use_ref             = cpu_ctx->use_ref;

    ಹಿಂತಿರುಗಿ ggml_ನಕ್ಷೆ_compute(cgraph, &cplan);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ ಖಾಲಿ,
    /* .get_tensor_async        = */ ಖಾಲಿ,
    /* .set_tensor_2d_async     = */ ಖಾಲಿ,
    /* .get_tensor_2d_async     = */ ಖಾಲಿ,
    /* .cpy_tensor_async        = */ ಖಾಲಿ,
    /* .synchronize             = */ ಖಾಲಿ,
    /* .graph_plan_create       = */ ggml_backend_cpu_ನಕ್ಷೆ_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_ನಕ್ಷೆ_plan_free,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ggml_backend_cpu_ನಕ್ಷೆ_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ಖಾಲಿ,
    /* .event_wait              = */ ಖಾಲಿ,
    /* .graph_optimize          = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_cpu_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    ಹಿಂತಿರುಗಿ &guid;
}

ggml_backend_t ggml_backend_cpu_init(ಶೂನ್ಯ) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    ರಚನೆ ggml_backend_cpu_context * ctx = ಹೊಸ ggml_backend_cpu_context;
    ಆದರೆ (ctx == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ctx->n_ಎಳೆಗಳು           = GGML_DEFAULT_N_ಎಳೆಗಳು;
    ctx->threadpool          = ಖಾಲಿ;
    ctx->work_ದತ್ತಾಂಶ           = ಖಾಲಿ;
    ctx->work_ಗಾತ್ರ           = 0;
    ctx->abort_callback      = ಖಾಲಿ;
    ctx->abort_callback_ದತ್ತಾಂಶ = ಖಾಲಿ;
    ctx->use_ref             = ಸುಳ್ಳು;

    ggml_backend_t cpu_backend = ಹೊಸ ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    ಆದರೆ (cpu_backend == ಖಾಲಿ) {
        ಅಳಿಸು ctx;
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ cpu_backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_cpu(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಖಾಲಿ && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

ಶೂನ್ಯ ggml_backend_cpu_ಗಣ_n_ಎಳೆಗಳು(ggml_backend_t backend_cpu, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ರಚನೆ ggml_backend_cpu_context * ctx = (ರಚನೆ ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
}

ಶೂನ್ಯ ggml_backend_cpu_ಗಣ_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ರಚನೆ ggml_backend_cpu_context * ctx = (ರಚನೆ ggml_backend_cpu_context *)backend_cpu->context;

    ಆದರೆ (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

ಶೂನ್ಯ ggml_backend_cpu_ಗಣ_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, ಶೂನ್ಯ * abort_callback_ದತ್ತಾಂಶ) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ರಚನೆ ggml_backend_cpu_context * ctx = (ರಚನೆ ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_ದತ್ತಾಂಶ = abort_callback_ದತ್ತಾಂಶ;
}

ಶೂನ್ಯ ggml_backend_cpu_ಗಣ_use_ref(ggml_backend_t backend_cpu, ತಾರ್ಕಿಕ use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ರಚನೆ ggml_backend_cpu_context * ctx = (ರಚನೆ ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

ರಚನೆ ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __APPLE__
        size_t len = 0;
        ಆದರೆ (!sysctlbyname("machdep.cpu.brand_string", ಖಾಲಿ, &len, ಖಾಲಿ, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, ಖಾಲಿ, 0); // NOLINT
        }
#ಅಥವಾಆದರೆ defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        ಆದರೆ (f) {
            ಅಕ್ಷರ buf[1024];
            ಆಗ (fgets(buf, ಗಾತ್ರ(buf), f)) {
                ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(buf, "model name", 10) == 0) {
                    ಅಕ್ಷರ * p = strchr(buf, ':');
                    ಆದರೆ (p) {
                        p++;
                        ಆಗ (std::ಖಾಲಿಯೇ(*p)) {
                            p++;
                        }
                        ಆಗ (std::ಖಾಲಿಯೇ(p[ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(p) - 1])) {
                            p[ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(p) - 1] = '\0';
                        }
                        description = p;
                        ನಿಲ್ಲಿಸು;
                    }
                }
            }
            fclose(f);
        }
#ಅಥವಾಆದರೆ defined(_WIN32)
        HKEY h_ಕೀಲಿ;
        ಆದರೆ (Reg_Open_ಕೀಲಿ_Ex(HKEY_LOCAL_MACHINE,
                        ಪಠ್ಯ("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        ಕೀಲಿ_READ,
                        &h_ಕೀಲಿ) == ERROR_SUCCESS) {
            DWORD cpu_brand_ಗಾತ್ರ = 0;
            ಆದರೆ (Reg_Query_ಮೌಲ್ಯ_Ex_A(h_ಕೀಲಿ,
                                "ProcessorNameString",
                                ಖಾಲಿ,
                                ಖಾಲಿ,
                                ಖಾಲಿ,
                                &cpu_brand_ಗಾತ್ರ) == ERROR_SUCCESS) {
                description.resize(cpu_brand_ಗಾತ್ರ);
                ಆದರೆ (Reg_Query_ಮೌಲ್ಯ_Ex_A(h_ಕೀಲಿ,
                                    "ProcessorNameString",
                                    ಖಾಲಿ,
                                    ಖಾಲಿ,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_ಗಾತ್ರ) == ERROR_SUCCESS) {
                    ಆದರೆ (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_ಕೀಲಿ(h_ಕೀಲಿ);
        }
#ಆದರೆ_ಅಂತ್ಯ
    }
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ "CPU";

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    ರಚನೆ ggml_backend_cpu_device_context * ctx = (ರಚನೆ ggml_backend_cpu_device_context *)dev->context;

    ಹಿಂತಿರುಗಿ ctx->description.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    MEMORYSTATUSEX ಸ್ಥಿತಿ_2;
    ಸ್ಥಿತಿ_2.dw_ಉದ್ದಳತೆ = ಗಾತ್ರ(ಸ್ಥಿತಿ_2);
    Global_ಸ್ಮರಣೆ_ಸ್ಥಿತಿ_Ex(&ಸ್ಥಿತಿ_2);
    *ಒಟ್ಟು = ಸ್ಥಿತಿ_2.ull_ಒಟ್ಟು_Phys;
    *ಬಿಡುಗಡೆ = ಸ್ಥಿತಿ_2.ullAvailPhys;
#ಇಲ್ಲದಿದ್ದರೆ
    ದೀರ್ಘ ಪುಟಗಳು = sysconf(_SC_PHYS_PAGES);
    ದೀರ್ಘ ಪುಟ_ಗಾತ್ರ = sysconf(_SC_PAGE_SIZE);
    *ಒಟ್ಟು = ಪುಟಗಳು * ಪುಟ_ಗಾತ್ರ;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *ಬಿಡುಗಡೆ = *ಒಟ್ಟು;
#ಆದರೆ_ಅಂತ್ಯ // _WIN32

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, ರಚನೆ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);
    props->caps = {
        /* .async                 = */ ಸುಳ್ಳು,
        /* .host_buffer           = */ ಸುಳ್ಳು,
        /* .buffer_from_host_ptr  = */ ನಿಜ,
        /* .events                = */ ಸುಳ್ಳು,
        /* .mmap_support          = */ ನಿಜ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ಹಿಂತಿರುಗಿ ggml_backend_cpu_init();

    GGML_ಬಳಸದ(dev);
    GGML_ಬಳಸದ(params);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_cpu_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ ggml_backend_cpu_ಬಫರ್_type();

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_cpu_device_ಬಫರ್_from_host_ptr(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ ggml_backend_cpu_ಬಫರ್_from_ptr(ptr, size);

    GGML_ಬಳಸದ(dev);
    GGML_ಬಳಸದ(max_tensor_ಗಾತ್ರ);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = op->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = op->src[1];

    ಆದರೆ (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ || op->op == GGML_OP_TRANSPOSE) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
        ಆದರೆ (op->src[i] && op->src[i]->ಬಫರ್ &&
            ggml_backend_cpu_is_extra_ಬಫರ್_type(op->src[i]->ಬಫರ್->buft)) {
            ಸ್ವಯಂಚಾಲಿತ * buf_extra = (ggml::cpu::extra_ಬಫರ್_type *) op->src[i]->ಬಫರ್->buft->context;
            ಹಿಂತಿರುಗಿ buf_extra->supports_op(dev, op);
        }
    }

    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_CPY:
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            ಹಿಂತಿರುಗಿ
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ಆದರೆ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                ಹಿಂತಿರುಗಿ src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            ಹಿಂತಿರುಗಿ src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ: {
            ಆದರೆ (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ದಶಮಾಂಶ max_bias = 0.0f;

            ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) op->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

            ಹಿಂತಿರುಗಿ max_bias == 0.0f;
        }
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
            ಹಿಂತಿರುಗಿ (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_CONV_2D:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]);
        ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
            ಹಿಂತಿರುಗಿ ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ನಿಜ;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_ಬಫರ್_type(buft);
    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_ಸ್ಮರಣೆ,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ಖಾಲಿ,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_ಬಫರ್_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ ಖಾಲಿ,
    /* .event_new            = */ ಖಾಲಿ,
    /* .event_free           = */ ಖಾಲಿ,
    /* .event_synchronize    = */ ಖಾಲಿ,
};

// CPU backend - backend (reg)

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ "CPU";

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ size_t ggml_backend_cpu_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ 1;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ಸೂಚ್ಯಂಕ == 0);

    ಸ್ಥಿರ ggml_backend_cpu_device_context ctx;
    ಸ್ಥಿರ ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
ಸ್ಥಿರ ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    ಸ್ಥಿರ std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        ಆದರೆ (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        ಆದರೆ (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        ಆದರೆ (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        ಆದರೆ (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        ಆದರೆ (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        ಆದರೆ (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        ಆದರೆ (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        ಆದರೆ (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        ಆದರೆ (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        ಆದರೆ (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        ಆದರೆ (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        ಆದರೆ (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        ಆದರೆ (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        ಆದರೆ (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        ಆದರೆ (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        ಆದರೆ (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        ಆದರೆ (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        ಆದರೆ (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        ಆದರೆ (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        ಆದರೆ (ggml_cpu_get_sve_cnt() > 0) {
            ಸ್ಥಿರ std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        ಆದರೆ (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        ಆದರೆ (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        ಆದರೆ (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        ಆದರೆ (ggml_cpu_get_rvv_vlen() > 0) {
            ಸ್ಥಿರ std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        ಆದರೆ (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        ಆದರೆ (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        ಆದರೆ (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        ಆದರೆ (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #ಆದರೆ_ಅಂತ್ಯ
    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #ಆದರೆ_ಅಂತ್ಯ
    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #ಆದರೆ_ಅಂತ್ಯ
    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #ಆದರೆ_ಅಂತ್ಯ
    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #ಆದರೆ_ಅಂತ್ಯ

        features.push_back({ ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ });

        ಹಿಂತಿರುಗಿ features;
    }();

    ಹಿಂತಿರುಗಿ features.ದತ್ತಾಂಶ();

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_cpu_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಆದರೆ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_ಗಣ_n_ಎಳೆಗಳು_t fct = ggml_backend_cpu_ಗಣ_n_ಎಳೆಗಳು;
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)fct;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)fct;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_get_features") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_cpu_get_features;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_cpu_ಗಣ_abort_callback;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_numa_init;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_is_numa;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_cpu_ಗಣ_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    ಆದರೆ (strcmp(name, "ggml_threadpool_new") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_threadpool_new;
    }
    ಆದರೆ (strcmp(name, "ggml_threadpool_free") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_threadpool_free;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_cpu_ಗಣ_threadpool;
    }

    ಹಿಂತಿರುಗಿ ಖಾಲಿ;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_ವಿಳಾಸ,
};

ggml_backend_reg_t ggml_backend_cpu_reg(ಶೂನ್ಯ) {
    // init CPU feature detection
    ggml_cpu_init();

    ಸ್ಥಿರ ರಚನೆ ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ ಖಾಲಿ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
