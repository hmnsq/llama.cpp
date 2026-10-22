#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "repack.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "traits.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "amx/amx.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cctype>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_HBM
#    include "hbm.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#ਜੇ_ਅੰਤ

#ਜੇ defined(_WIN32)
#    define WIN32_LEAN_AND_ਔਸਤ
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#ਨਹੀਂਤਾਂ
#    include <unistd.h>
#ਜੇ_ਅੰਤ

#ਜੇ defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#ਜੇ_ਅੰਤ

// ggml-backend interface

std::vector<ggml_backend_ਬਫਰ_type_t> & ggml_backend_cpu_get_extra_ਬਫਰ_types() {
    ਸਥਿਰ std::vector<ggml_backend_ਬਫਰ_type_t> bufts = []() {
        std::vector<ggml_backend_ਬਫਰ_type_t> bufts;

#ਜੇ defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        ਜੇ (ggml_backend_amx_ਬਫਰ_type()) {
            bufts.push_back(ggml_backend_amx_ਬਫਰ_type());
        }
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_RISCV64_SPACEMIT
        ਜੇ (ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_ਬਫਰ_type());
        }
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_KLEIDIAI
        ਜੇ (ggml_backend_cpu_kleidiai_ਬਫਰ_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_ਬਫਰ_type());
        }
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_REPACK
        ਜੇ (ggml_backend_cpu_repack_ਬਫਰ_type()) {
            bufts.push_back(ggml_backend_cpu_repack_ਬਫਰ_type());
        }
#ਜੇ_ਅੰਤ

        ਵਾਪਸ bufts;
    }();

    ਵਾਪਸ bufts;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    ਸਥਿਰ std::vector<ggml_backend_ਬਫਰ_type_t> extra_bufts = [] {
        std::vector<ggml_backend_ਬਫਰ_type_t> bufts = ggml_backend_cpu_get_extra_ਬਫਰ_types();
        bufts.push_back(ਖਾਲੀ_ਸੂਚਕ);
        ਵਾਪਸ bufts;
    }();

    ਵਾਪਸ extra_bufts.ਅੰਕੜੇ();

    GGML_ਅਣਵਰਤਿਆ(device);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cpu_is_extra_ਬਫਰ_type(ggml_backend_ਬਫਰ_type_t buft) {
    ਲਈ (ਸਵੈਚਾਲਿਤ * extra : ggml_backend_cpu_get_extra_ਬਫਰ_types()) {
        ਜੇ (extra == buft) {
            ਵਾਪਸ ਸੱਚ;
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

// CPU backend - backend (stream)

ਬਣਤਰ ggml_backend_cpu_context {
    ਪੂਰਨ                 n_ਧਾਗੇ;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_ਅੰਕੜੇ;
    size_t              work_ਮਾਪ;

    ggml_abort_callback abort_callback;
    ਖਾਲੀ *              abort_callback_ਅੰਕੜੇ;

    ਤਾਰਕਿਕ                use_ref;  // use reference implementation
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    ਵਾਪਸ "CPU";

    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_free(ggml_backend_t backend) {
    ਬਣਤਰ ggml_backend_cpu_context * cpu_ctx = (ਬਣਤਰ ggml_backend_cpu_context *)backend->context;
    ਮਿਟਾਓ[] cpu_ctx->work_ਅੰਕੜੇ;
    ਮਿਟਾਓ cpu_ctx;
    ਮਿਟਾਓ backend;
}

ਬਣਤਰ ggml_backend_plan_cpu {
    ਬਣਤਰ ggml_cplan cplan;
    ਬਣਤਰ ggml_cgraph cgraph;
};

ਸਥਿਰ ggml_backend_ਗ੍ਰਾਫ_plan_t ggml_backend_cpu_ਗ੍ਰਾਫ_plan_create(ggml_backend_t backend, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph) {
    ਬਣਤਰ ggml_backend_cpu_context * cpu_ctx = (ਬਣਤਰ ggml_backend_cpu_context *)backend->context;

    ਬਣਤਰ ggml_backend_plan_cpu * cpu_plan = ਨਵਾਂ ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_ਗ੍ਰਾਫ_plan(cgraph, cpu_ctx->n_ਧਾਗੇ, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    ਜੇ (cpu_plan->cplan.work_ਮਾਪ > 0) {
        cpu_plan->cplan.work_ਅੰਕੜੇ = ਨਵਾਂ uint8_t[cpu_plan->cplan.work_ਮਾਪ];
        ਜੇ (cpu_plan->cplan.work_ਅੰਕੜੇ == ਨਿਰਾ) {
            ਮਿਟਾਓ cpu_plan;
            ਵਾਪਸ ਨਿਰਾ;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_ਅੰਕੜੇ = cpu_ctx->abort_callback_ਅੰਕੜੇ;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    ਵਾਪਸ cpu_plan;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_ਗ੍ਰਾਫ_plan_free(ggml_backend_t backend, ggml_backend_ਗ੍ਰਾਫ_plan_t plan) {
    ਬਣਤਰ ggml_backend_plan_cpu * cpu_plan = (ਬਣਤਰ ggml_backend_plan_cpu *)plan;

    ਮਿਟਾਓ[] cpu_plan->cplan.work_ਅੰਕੜੇ;
    ਮਿਟਾਓ cpu_plan;

    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_cpu_ਗ੍ਰਾਫ_plan_compute(ggml_backend_t backend, ggml_backend_ਗ੍ਰਾਫ_plan_t plan) {
    ਬਣਤਰ ggml_backend_plan_cpu * cpu_plan = (ਬਣਤਰ ggml_backend_plan_cpu *)plan;

    ਵਾਪਸ ggml_ਗ੍ਰਾਫ_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_cpu_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * cgraph) {
    ਬਣਤਰ ggml_backend_cpu_context * cpu_ctx = (ਬਣਤਰ ggml_backend_cpu_context *)backend->context;

    ਬਣਤਰ ggml_cplan cplan = ggml_ਗ੍ਰਾਫ_plan(cgraph, cpu_ctx->n_ਧਾਗੇ, cpu_ctx->threadpool);

    ਜੇ (cpu_ctx->work_ਮਾਪ < cplan.work_ਮਾਪ) {
        ਮਿਟਾਓ[] cpu_ctx->work_ਅੰਕੜੇ;
        cpu_ctx->work_ਅੰਕੜੇ = ਨਵਾਂ uint8_t[cplan.work_ਮਾਪ];
        ਜੇ (cpu_ctx->work_ਅੰਕੜੇ == ਨਿਰਾ) {
            cpu_ctx->work_ਮਾਪ = 0;
            ਵਾਪਸ GGML_ਸਥਿਤੀ_ALLOC_FAILED;
        }
        cpu_ctx->work_ਮਾਪ = cplan.work_ਮਾਪ;
    }
    cplan.work_ਅੰਕੜੇ = (uint8_t *)cpu_ctx->work_ਅੰਕੜੇ;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_ਅੰਕੜੇ = cpu_ctx->abort_callback_ਅੰਕੜੇ;
    cplan.use_ref             = cpu_ctx->use_ref;

    ਵਾਪਸ ggml_ਗ੍ਰਾਫ_compute(cgraph, &cplan);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ ਨਿਰਾ,
    /* .get_tensor_async        = */ ਨਿਰਾ,
    /* .set_tensor_2d_async     = */ ਨਿਰਾ,
    /* .get_tensor_2d_async     = */ ਨਿਰਾ,
    /* .cpy_tensor_async        = */ ਨਿਰਾ,
    /* .synchronize             = */ ਨਿਰਾ,
    /* .graph_plan_create       = */ ggml_backend_cpu_ਗ੍ਰਾਫ_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_ਗ੍ਰਾਫ_plan_free,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ggml_backend_cpu_ਗ੍ਰਾਫ_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ਨਿਰਾ,
    /* .event_wait              = */ ਨਿਰਾ,
    /* .graph_optimize          = */ ਨਿਰਾ,
};

ਸਥਿਰ ggml_guid_t ggml_backend_cpu_guid(ਖਾਲੀ) {
    ਸਥਿਰ ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    ਵਾਪਸ &guid;
}

ggml_backend_t ggml_backend_cpu_init(ਖਾਲੀ) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    ਬਣਤਰ ggml_backend_cpu_context * ctx = ਨਵਾਂ ggml_backend_cpu_context;
    ਜੇ (ctx == ਨਿਰਾ) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ctx->n_ਧਾਗੇ           = GGML_DEFAULT_N_ਧਾਗੇ;
    ctx->threadpool          = ਨਿਰਾ;
    ctx->work_ਅੰਕੜੇ           = ਨਿਰਾ;
    ctx->work_ਮਾਪ           = 0;
    ctx->abort_callback      = ਨਿਰਾ;
    ctx->abort_callback_ਅੰਕੜੇ = ਨਿਰਾ;
    ctx->use_ref             = ਝੂਠ;

    ggml_backend_t cpu_backend = ਨਵਾਂ ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    ਜੇ (cpu_backend == ਨਿਰਾ) {
        ਮਿਟਾਓ ctx;
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ cpu_backend;
}

ਤਾਰਕਿਕ ggml_backend_is_cpu(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਨਿਰਾ && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

ਖਾਲੀ ggml_backend_cpu_ਸੈੱਟ_n_ਧਾਗੇ(ggml_backend_t backend_cpu, ਪੂਰਨ n_ਧਾਗੇ) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ਬਣਤਰ ggml_backend_cpu_context * ctx = (ਬਣਤਰ ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_ਧਾਗੇ = n_ਧਾਗੇ;
}

ਖਾਲੀ ggml_backend_cpu_ਸੈੱਟ_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ਬਣਤਰ ggml_backend_cpu_context * ctx = (ਬਣਤਰ ggml_backend_cpu_context *)backend_cpu->context;

    ਜੇ (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

ਖਾਲੀ ggml_backend_cpu_ਸੈੱਟ_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, ਖਾਲੀ * abort_callback_ਅੰਕੜੇ) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ਬਣਤਰ ggml_backend_cpu_context * ctx = (ਬਣਤਰ ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_ਅੰਕੜੇ = abort_callback_ਅੰਕੜੇ;
}

ਖਾਲੀ ggml_backend_cpu_ਸੈੱਟ_use_ref(ggml_backend_t backend_cpu, ਤਾਰਕਿਕ use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ਬਣਤਰ ggml_backend_cpu_context * ctx = (ਬਣਤਰ ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

ਬਣਤਰ ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __APPLE__
        size_t len = 0;
        ਜੇ (!sysctlbyname("machdep.cpu.brand_string", ਨਿਰਾ, &len, ਨਿਰਾ, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, ਨਿਰਾ, 0); // NOLINT
        }
#ਜਾਂਜੇ defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        ਜੇ (f) {
            ਅੱਖਰ buf[1024];
            ਜਦੋਂ (fgets(buf, ਆਕਾਰ(buf), f)) {
                ਜੇ (ਲੜੀ_ਤੁਲਨਾ(buf, "model name", 10) == 0) {
                    ਅੱਖਰ * p = strchr(buf, ':');
                    ਜੇ (p) {
                        p++;
                        ਜਦੋਂ (std::ਖਾਲੀ_ਥਾਂ_ਹੈ(*p)) {
                            p++;
                        }
                        ਜਦੋਂ (std::ਖਾਲੀ_ਥਾਂ_ਹੈ(p[ਲੜੀ_ਲੰਬਾਈ(p) - 1])) {
                            p[ਲੜੀ_ਲੰਬਾਈ(p) - 1] = '\0';
                        }
                        description = p;
                        ਰੋਕੋ;
                    }
                }
            }
            fclose(f);
        }
#ਜਾਂਜੇ defined(_WIN32)
        HKEY h_ਕੁੰਜੀ;
        ਜੇ (Reg_Open_ਕੁੰਜੀ_Ex(HKEY_LOCAL_MACHINE,
                        ਪਾਠ("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        ਕੁੰਜੀ_READ,
                        &h_ਕੁੰਜੀ) == ERROR_SUCCESS) {
            DWORD cpu_brand_ਮਾਪ = 0;
            ਜੇ (Reg_Query_ਮੁੱਲ_Ex_A(h_ਕੁੰਜੀ,
                                "ProcessorNameString",
                                ਨਿਰਾ,
                                ਨਿਰਾ,
                                ਨਿਰਾ,
                                &cpu_brand_ਮਾਪ) == ERROR_SUCCESS) {
                description.resize(cpu_brand_ਮਾਪ);
                ਜੇ (Reg_Query_ਮੁੱਲ_Ex_A(h_ਕੁੰਜੀ,
                                    "ProcessorNameString",
                                    ਨਿਰਾ,
                                    ਨਿਰਾ,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_ਮਾਪ) == ERROR_SUCCESS) {
                    ਜੇ (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_ਕੁੰਜੀ(h_ਕੁੰਜੀ);
        }
#ਜੇ_ਅੰਤ
    }
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    ਵਾਪਸ "CPU";

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    ਬਣਤਰ ggml_backend_cpu_device_context * ctx = (ਬਣਤਰ ggml_backend_cpu_device_context *)dev->context;

    ਵਾਪਸ ctx->description.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    MEMORYSTATUSEX ਸਥਿਤੀ;
    ਸਥਿਤੀ.dw_ਲੰਮਾਈ = ਆਕਾਰ(ਸਥਿਤੀ);
    Global_ਸਮ੍ਰਿਤੀ_ਸਥਿਤੀ_Ex(&ਸਥਿਤੀ);
    *ਕੁੱਲ = ਸਥਿਤੀ.ull_ਕੁੱਲ_Phys;
    *ਮੁਕਤ = ਸਥਿਤੀ.ullAvailPhys;
#ਨਹੀਂਤਾਂ
    ਲੰਮਾ ਪੰਨੇ = sysconf(_SC_PHYS_PAGES);
    ਲੰਮਾ ਪੰਨਾ_ਮਾਪ = sysconf(_SC_PAGE_SIZE);
    *ਕੁੱਲ = ਪੰਨੇ * ਪੰਨਾ_ਮਾਪ;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *ਮੁਕਤ = *ਕੁੱਲ;
#ਜੇ_ਅੰਤ // _WIN32

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, ਬਣਤਰ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);
    props->caps = {
        /* .async                 = */ ਝੂਠ,
        /* .host_buffer           = */ ਝੂਠ,
        /* .buffer_from_host_ptr  = */ ਸੱਚ,
        /* .events                = */ ਝੂਠ,
        /* .mmap_support          = */ ਸੱਚ,
    };
}

ਸਥਿਰ ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    ਵਾਪਸ ggml_backend_cpu_init();

    GGML_ਅਣਵਰਤਿਆ(dev);
    GGML_ਅਣਵਰਤਿਆ(params);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_cpu_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ ggml_backend_cpu_ਬਫਰ_type();

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_cpu_device_ਬਫਰ_from_host_ptr(ggml_backend_dev_t dev, ਖਾਲੀ * ptr, size_t size, size_t max_tensor_ਮਾਪ) {
    ਵਾਪਸ ggml_backend_cpu_ਬਫਰ_from_ptr(ptr, size);

    GGML_ਅਣਵਰਤਿਆ(dev);
    GGML_ਅਣਵਰਤਿਆ(max_tensor_ਮਾਪ);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];

    ਜੇ (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ || op->op == GGML_OP_TRANSPOSE) {
        ਵਾਪਸ ਸੱਚ;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਜੇ (op->src[i] && op->src[i]->ਬਫਰ &&
            ggml_backend_cpu_is_extra_ਬਫਰ_type(op->src[i]->ਬਫਰ->buft)) {
            ਸਵੈਚਾਲਿਤ * buf_extra = (ggml::cpu::extra_ਬਫਰ_type *) op->src[i]->ਬਫਰ->buft->context;
            ਵਾਪਸ buf_extra->supports_op(dev, op);
        }
    }

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_CPY:
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            ਵਾਪਸ
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        ਹਾਲਤ GGML_OP_MUL_MAT:
            ਜੇ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                ਵਾਪਸ src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            ਵਾਪਸ src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ: {
            ਜੇ (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਦਸ਼ਮਲਵ max_bias = 0.0f;

            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) op->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

            ਵਾਪਸ max_bias == 0.0f;
        }
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_ਪਿੱਠ:
            ਵਾਪਸ src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ:
            ਵਾਪਸ src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
            ਵਾਪਸ (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_CONV_2D:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]);
        ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            ਵਾਪਸ ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        ਮੂਲ:
            ਵਾਪਸ ਸੱਚ;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_ਬਫਰ_type(buft);
    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ਨਿਰਾ,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_ਬਫਰ_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ ਨਿਰਾ,
    /* .event_new            = */ ਨਿਰਾ,
    /* .event_free           = */ ਨਿਰਾ,
    /* .event_synchronize    = */ ਨਿਰਾ,
};

// CPU backend - backend (reg)

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    ਵਾਪਸ "CPU";

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ size_t ggml_backend_cpu_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ਵਾਪਸ 1;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ਸੂਚਕਾਂਕ == 0);

    ਸਥਿਰ ggml_backend_cpu_device_context ctx;
    ਸਥਿਰ ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    ਵਾਪਸ &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
ਸਥਿਰ ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    ਸਥਿਰ std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        ਜੇ (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        ਜੇ (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        ਜੇ (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        ਜੇ (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        ਜੇ (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        ਜੇ (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        ਜੇ (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        ਜੇ (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        ਜੇ (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        ਜੇ (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        ਜੇ (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        ਜੇ (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        ਜੇ (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        ਜੇ (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        ਜੇ (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        ਜੇ (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        ਜੇ (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        ਜੇ (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        ਜੇ (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        ਜੇ (ggml_cpu_get_sve_cnt() > 0) {
            ਸਥਿਰ std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        ਜੇ (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        ਜੇ (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        ਜੇ (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        ਜੇ (ggml_cpu_get_rvv_vlen() > 0) {
            ਸਥਿਰ std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        ਜੇ (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        ਜੇ (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        ਜੇ (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        ਜੇ (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #ਜੇ_ਅੰਤ
    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #ਜੇ_ਅੰਤ
    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #ਜੇ_ਅੰਤ
    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #ਜੇ_ਅੰਤ
    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #ਜੇ_ਅੰਤ

        features.push_back({ ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ });

        ਵਾਪਸ features;
    }();

    ਵਾਪਸ features.ਅੰਕੜੇ();

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_cpu_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name) {
    ਜੇ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_ਸੈੱਟ_n_ਧਾਗੇ_t fct = ggml_backend_cpu_ਸੈੱਟ_n_ਧਾਗੇ;
        ਵਾਪਸ (ਖਾਲੀ *)fct;
    }
    ਜੇ (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        ਵਾਪਸ (ਖਾਲੀ *)fct;
    }
    ਜੇ (strcmp(name, "ggml_backend_get_features") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_cpu_get_features;
    }
    ਜੇ (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_cpu_ਸੈੱਟ_abort_callback;
    }
    ਜੇ (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_numa_init;
    }
    ਜੇ (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_is_numa;
    }
    ਜੇ (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_cpu_ਸੈੱਟ_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    ਜੇ (strcmp(name, "ggml_threadpool_new") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_threadpool_new;
    }
    ਜੇ (strcmp(name, "ggml_threadpool_free") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_threadpool_free;
    }
    ਜੇ (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_cpu_ਸੈੱਟ_threadpool;
    }

    ਵਾਪਸ ਨਿਰਾ;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_ਪਤਾ,
};

ggml_backend_reg_t ggml_backend_cpu_reg(ਖਾਲੀ) {
    // init CPU feature detection
    ggml_cpu_init();

    ਸਥਿਰ ਬਣਤਰ ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ ਨਿਰਾ,
    };

    ਵਾਪਸ &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
