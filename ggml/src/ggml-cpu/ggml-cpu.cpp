#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "repack.h"
#అన్తర్భావ "traits.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "amx/amx.h"

#అన్తర్భావ <cctype>
#అన్తర్భావ <string>
#అన్తర్భావ <vector>

#నిర్వచితమైతే GGML_USE_CPU_HBM
#    include "hbm.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#అయితే_ముగింపు

#ఒకవేళ defined(_WIN32)
#    define WIN32_LEAN_AND_సగటు
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#లేకపోతే
#    include <unistd.h>
#అయితే_ముగింపు

#ఒకవేళ defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#అయితే_ముగింపు

// ggml-backend interface

std::vector<ggml_backend_బఫరు_type_t> & ggml_backend_cpu_get_extra_బఫరు_types() {
    స్థిర std::vector<ggml_backend_బఫరు_type_t> bufts = []() {
        std::vector<ggml_backend_బఫరు_type_t> bufts;

#ఒకవేళ defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        ఒకవేళ (ggml_backend_amx_బఫరు_type()) {
            bufts.push_back(ggml_backend_amx_బఫరు_type());
        }
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_CPU_RISCV64_SPACEMIT
        ఒకవేళ (ggml_backend_cpu_riscv64_spacemit_బఫరు_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_బఫరు_type());
        }
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_CPU_KLEIDIAI
        ఒకవేళ (ggml_backend_cpu_kleidiai_బఫరు_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_బఫరు_type());
        }
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_CPU_REPACK
        ఒకవేళ (ggml_backend_cpu_repack_బఫరు_type()) {
            bufts.push_back(ggml_backend_cpu_repack_బఫరు_type());
        }
#అయితే_ముగింపు

        తిరిగి bufts;
    }();

    తిరిగి bufts;
}

స్థిర ggml_backend_బఫరు_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    స్థిర std::vector<ggml_backend_బఫరు_type_t> extra_bufts = [] {
        std::vector<ggml_backend_బఫరు_type_t> bufts = ggml_backend_cpu_get_extra_బఫరు_types();
        bufts.push_back(శూన్యసూచిక);
        తిరిగి bufts;
    }();

    తిరిగి extra_bufts.దత్తాంశం();

    GGML_వాడని(device);
}

స్థిర తార్కిక ggml_backend_cpu_is_extra_బఫరు_type(ggml_backend_బఫరు_type_t buft) {
    కొరకు (స్వయంచాలక * extra : ggml_backend_cpu_get_extra_బఫరు_types()) {
        ఒకవేళ (extra == buft) {
            తిరిగి నిజం;
        }
    }
    తిరిగి అబద్ధం;
}

// CPU backend - backend (stream)

నిర్మాణం ggml_backend_cpu_context {
    పూర్ణాంకం                 n_దారాలు;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_దత్తాంశం;
    size_t              work_కొలత;

    ggml_abort_callback abort_callback;
    శూన్య *              abort_callback_దత్తాంశం;

    తార్కిక                use_ref;  // use reference implementation
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    తిరిగి "CPU";

    GGML_వాడని(backend);
}

స్థిర శూన్య ggml_backend_cpu_free(ggml_backend_t backend) {
    నిర్మాణం ggml_backend_cpu_context * cpu_ctx = (నిర్మాణం ggml_backend_cpu_context *)backend->context;
    తొలగించు[] cpu_ctx->work_దత్తాంశం;
    తొలగించు cpu_ctx;
    తొలగించు backend;
}

నిర్మాణం ggml_backend_plan_cpu {
    నిర్మాణం ggml_cplan cplan;
    నిర్మాణం ggml_cgraph cgraph;
};

స్థిర ggml_backend_లేఖాచిత్రం_plan_t ggml_backend_cpu_లేఖాచిత్రం_plan_create(ggml_backend_t backend, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph) {
    నిర్మాణం ggml_backend_cpu_context * cpu_ctx = (నిర్మాణం ggml_backend_cpu_context *)backend->context;

    నిర్మాణం ggml_backend_plan_cpu * cpu_plan = కొత్త ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_లేఖాచిత్రం_plan(cgraph, cpu_ctx->n_దారాలు, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    ఒకవేళ (cpu_plan->cplan.work_కొలత > 0) {
        cpu_plan->cplan.work_దత్తాంశం = కొత్త uint8_t[cpu_plan->cplan.work_కొలత];
        ఒకవేళ (cpu_plan->cplan.work_దత్తాంశం == ఖాళీ) {
            తొలగించు cpu_plan;
            తిరిగి ఖాళీ;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_దత్తాంశం = cpu_ctx->abort_callback_దత్తాంశం;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    తిరిగి cpu_plan;
}

స్థిర శూన్య ggml_backend_cpu_లేఖాచిత్రం_plan_free(ggml_backend_t backend, ggml_backend_లేఖాచిత్రం_plan_t plan) {
    నిర్మాణం ggml_backend_plan_cpu * cpu_plan = (నిర్మాణం ggml_backend_plan_cpu *)plan;

    తొలగించు[] cpu_plan->cplan.work_దత్తాంశం;
    తొలగించు cpu_plan;

    GGML_వాడని(backend);
}

స్థిర గణన ggml_స్థితి ggml_backend_cpu_లేఖాచిత్రం_plan_compute(ggml_backend_t backend, ggml_backend_లేఖాచిత్రం_plan_t plan) {
    నిర్మాణం ggml_backend_plan_cpu * cpu_plan = (నిర్మాణం ggml_backend_plan_cpu *)plan;

    తిరిగి ggml_లేఖాచిత్రం_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_వాడని(backend);
}

స్థిర గణన ggml_స్థితి ggml_backend_cpu_లేఖాచిత్రం_compute(ggml_backend_t backend, నిర్మాణం ggml_cgraph * cgraph) {
    నిర్మాణం ggml_backend_cpu_context * cpu_ctx = (నిర్మాణం ggml_backend_cpu_context *)backend->context;

    నిర్మాణం ggml_cplan cplan = ggml_లేఖాచిత్రం_plan(cgraph, cpu_ctx->n_దారాలు, cpu_ctx->threadpool);

    ఒకవేళ (cpu_ctx->work_కొలత < cplan.work_కొలత) {
        తొలగించు[] cpu_ctx->work_దత్తాంశం;
        cpu_ctx->work_దత్తాంశం = కొత్త uint8_t[cplan.work_కొలత];
        ఒకవేళ (cpu_ctx->work_దత్తాంశం == ఖాళీ) {
            cpu_ctx->work_కొలత = 0;
            తిరిగి GGML_స్థితి_ALLOC_FAILED;
        }
        cpu_ctx->work_కొలత = cplan.work_కొలత;
    }
    cplan.work_దత్తాంశం = (uint8_t *)cpu_ctx->work_దత్తాంశం;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_దత్తాంశం = cpu_ctx->abort_callback_దత్తాంశం;
    cplan.use_ref             = cpu_ctx->use_ref;

    తిరిగి ggml_లేఖాచిత్రం_compute(cgraph, &cplan);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ ఖాళీ,
    /* .get_tensor_async        = */ ఖాళీ,
    /* .set_tensor_2d_async     = */ ఖాళీ,
    /* .get_tensor_2d_async     = */ ఖాళీ,
    /* .cpy_tensor_async        = */ ఖాళీ,
    /* .synchronize             = */ ఖాళీ,
    /* .graph_plan_create       = */ ggml_backend_cpu_లేఖాచిత్రం_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_లేఖాచిత్రం_plan_free,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ggml_backend_cpu_లేఖాచిత్రం_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_లేఖాచిత్రం_compute,
    /* .event_record            = */ ఖాళీ,
    /* .event_wait              = */ ఖాళీ,
    /* .graph_optimize          = */ ఖాళీ,
};

స్థిర ggml_guid_t ggml_backend_cpu_guid(శూన్య) {
    స్థిర ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    తిరిగి &guid;
}

ggml_backend_t ggml_backend_cpu_init(శూన్య) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    నిర్మాణం ggml_backend_cpu_context * ctx = కొత్త ggml_backend_cpu_context;
    ఒకవేళ (ctx == ఖాళీ) {
        తిరిగి ఖాళీ;
    }

    ctx->n_దారాలు           = GGML_DEFAULT_N_దారాలు;
    ctx->threadpool          = ఖాళీ;
    ctx->work_దత్తాంశం           = ఖాళీ;
    ctx->work_కొలత           = 0;
    ctx->abort_callback      = ఖాళీ;
    ctx->abort_callback_దత్తాంశం = ఖాళీ;
    ctx->use_ref             = అబద్ధం;

    ggml_backend_t cpu_backend = కొత్త ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    ఒకవేళ (cpu_backend == ఖాళీ) {
        తొలగించు ctx;
        తిరిగి ఖాళీ;
    }

    తిరిగి cpu_backend;
}

తార్కిక ggml_backend_is_cpu(ggml_backend_t backend) {
    తిరిగి backend != ఖాళీ && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

శూన్య ggml_backend_cpu_సమితి_n_దారాలు(ggml_backend_t backend_cpu, పూర్ణాంకం n_దారాలు) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    నిర్మాణం ggml_backend_cpu_context * ctx = (నిర్మాణం ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_దారాలు = n_దారాలు;
}

శూన్య ggml_backend_cpu_సమితి_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    నిర్మాణం ggml_backend_cpu_context * ctx = (నిర్మాణం ggml_backend_cpu_context *)backend_cpu->context;

    ఒకవేళ (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

శూన్య ggml_backend_cpu_సమితి_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, శూన్య * abort_callback_దత్తాంశం) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    నిర్మాణం ggml_backend_cpu_context * ctx = (నిర్మాణం ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_దత్తాంశం = abort_callback_దత్తాంశం;
}

శూన్య ggml_backend_cpu_సమితి_use_ref(ggml_backend_t backend_cpu, తార్కిక use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    నిర్మాణం ggml_backend_cpu_context * ctx = (నిర్మాణం ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

నిర్మాణం ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#నిర్వచితమైతే __APPLE__
        size_t len = 0;
        ఒకవేళ (!sysctlbyname("machdep.cpu.brand_string", ఖాళీ, &len, ఖాళీ, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, ఖాళీ, 0); // NOLINT
        }
#లేదావేళ defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        ఒకవేళ (f) {
            అక్షరం buf[1024];
            అయితే (fgets(buf, పరిమాణం(buf), f)) {
                ఒకవేళ (అక్షరశ్రేణి_పోలిక(buf, "model name", 10) == 0) {
                    అక్షరం * p = strchr(buf, ':');
                    ఒకవేళ (p) {
                        p++;
                        అయితే (std::ఖాళీనా(*p)) {
                            p++;
                        }
                        అయితే (std::ఖాళీనా(p[అక్షరశ్రేణి_పొడవు(p) - 1])) {
                            p[అక్షరశ్రేణి_పొడవు(p) - 1] = '\0';
                        }
                        description = p;
                        ఆపు;
                    }
                }
            }
            fclose(f);
        }
#లేదావేళ defined(_WIN32)
        HKEY h_కీలకం;
        ఒకవేళ (Reg_Open_కీలకం_Ex(HKEY_LOCAL_MACHINE,
                        పాఠ్యం("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        కీలకం_READ,
                        &h_కీలకం) == ERROR_SUCCESS) {
            DWORD cpu_brand_కొలత = 0;
            ఒకవేళ (Reg_Query_విలువ_Ex_A(h_కీలకం,
                                "ProcessorNameString",
                                ఖాళీ,
                                ఖాళీ,
                                ఖాళీ,
                                &cpu_brand_కొలత) == ERROR_SUCCESS) {
                description.resize(cpu_brand_కొలత);
                ఒకవేళ (Reg_Query_విలువ_Ex_A(h_కీలకం,
                                    "ProcessorNameString",
                                    ఖాళీ,
                                    ఖాళీ,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_కొలత) == ERROR_SUCCESS) {
                    ఒకవేళ (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_కీలకం(h_కీలకం);
        }
#అయితే_ముగింపు
    }
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    తిరిగి "CPU";

    GGML_వాడని(dev);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    నిర్మాణం ggml_backend_cpu_device_context * ctx = (నిర్మాణం ggml_backend_cpu_device_context *)dev->context;

    తిరిగి ctx->description.c_str();
}

స్థిర శూన్య ggml_backend_cpu_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
#నిర్వచితమైతే _WIN32
    MEMORYSTATUSEX స్థితి_2;
    స్థితి_2.dw_నిడివి = పరిమాణం(స్థితి_2);
    Global_జ్ఞాపకం_స్థితి_Ex(&స్థితి_2);
    *మొత్తం = స్థితి_2.ull_మొత్తం_Phys;
    *విడుదల = స్థితి_2.ullAvailPhys;
#లేకపోతే
    దీర్ఘ పుటలు = sysconf(_SC_PHYS_PAGES);
    దీర్ఘ పుట_కొలత = sysconf(_SC_PAGE_SIZE);
    *మొత్తం = పుటలు * పుట_కొలత;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *విడుదల = *మొత్తం;
#అయితే_ముగింపు // _WIN32

    GGML_వాడని(dev);
}

స్థిర గణన ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    తిరిగి GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, నిర్మాణం ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);
    props->caps = {
        /* .async                 = */ అబద్ధం,
        /* .host_buffer           = */ అబద్ధం,
        /* .buffer_from_host_ptr  = */ నిజం,
        /* .events                = */ అబద్ధం,
        /* .mmap_support          = */ నిజం,
    };
}

స్థిర ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    తిరిగి ggml_backend_cpu_init();

    GGML_వాడని(dev);
    GGML_వాడని(params);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_cpu_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    తిరిగి ggml_backend_cpu_బఫరు_type();

    GGML_వాడని(dev);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_cpu_device_బఫరు_from_host_ptr(ggml_backend_dev_t dev, శూన్య * ptr, size_t size, size_t max_tensor_కొలత) {
    తిరిగి ggml_backend_cpu_బఫరు_from_ptr(ptr, size);

    GGML_వాడని(dev);
    GGML_వాడని(max_tensor_కొలత);
}

స్థిర తార్కిక ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];

    ఒకవేళ (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_క్రమమార్పు || op->op == GGML_OP_TRANSPOSE) {
        తిరిగి నిజం;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        ఒకవేళ (op->src[i] && op->src[i]->బఫరు &&
            ggml_backend_cpu_is_extra_బఫరు_type(op->src[i]->బఫరు->buft)) {
            స్వయంచాలక * buf_extra = (ggml::cpu::extra_బఫరు_type *) op->src[i]->బఫరు->buft->context;
            తిరిగి buf_extra->supports_op(dev, op);
        }
    }

    ఎంపిక (op->op) {
        స్థితి GGML_OP_CPY:
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            తిరిగి
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        స్థితి GGML_OP_MUL_MAT:
            ఒకవేళ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                తిరిగి src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            తిరిగి src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        స్థితి GGML_OP_SOFT_MAX_వెనుక: {
            ఒకవేళ (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                తిరిగి అబద్ధం;
            }
            దశాంశం max_bias = 0.0f;

            జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) op->op_params + 1, పరిమాణం(దశాంశం));

            తిరిగి max_bias == 0.0f;
        }
        స్థితి GGML_OP_IM2_నిలువు_వెనుక:
            తిరిగి src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక:
            తిరిగి src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        స్థితి GGML_OP_బహిర్గతం_PROD:
            తిరిగి (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_CONV_2D:
            తిరిగి ggml_is_contiguous(op->src[0]);
        స్థితి GGML_OP_SSM_పరిశీలన:
            తిరిగి ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        అప్రమేయ:
            తిరిగి నిజం;
    }
}

స్థిర తార్కిక ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    తిరిగి ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_బఫరు_type(buft);
    GGML_వాడని(dev);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_జ్ఞాపకం,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ఖాళీ,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_బఫరు_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ ఖాళీ,
    /* .event_new            = */ ఖాళీ,
    /* .event_free           = */ ఖాళీ,
    /* .event_synchronize    = */ ఖాళీ,
};

// CPU backend - backend (reg)

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    తిరిగి "CPU";

    GGML_వాడని(reg);
}

స్థిర size_t ggml_backend_cpu_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    తిరిగి 1;

    GGML_వాడని(reg);
}

స్థిర ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t సూచీ) {
    GGML_ASSERT(సూచీ == 0);

    స్థిర ggml_backend_cpu_device_context ctx;
    స్థిర ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    తిరిగి &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
స్థిర ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    స్థిర std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        ఒకవేళ (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        ఒకవేళ (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        ఒకవేళ (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        ఒకవేళ (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        ఒకవేళ (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        ఒకవేళ (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        ఒకవేళ (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        ఒకవేళ (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        ఒకవేళ (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        ఒకవేళ (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        ఒకవేళ (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        ఒకవేళ (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        ఒకవేళ (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        ఒకవేళ (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        ఒకవేళ (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        ఒకవేళ (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        ఒకవేళ (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        ఒకవేళ (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        ఒకవేళ (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        ఒకవేళ (ggml_cpu_get_sve_cnt() > 0) {
            స్థిర std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        ఒకవేళ (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        ఒకవేళ (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        ఒకవేళ (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        ఒకవేళ (ggml_cpu_get_rvv_vlen() > 0) {
            స్థిర std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        ఒకవేళ (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        ఒకవేళ (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        ఒకవేళ (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        ఒకవేళ (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #నిర్వచితమైతే GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #అయితే_ముగింపు
    #నిర్వచితమైతే GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #అయితే_ముగింపు
    #నిర్వచితమైతే GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #అయితే_ముగింపు
    #నిర్వచితమైతే GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #అయితే_ముగింపు
    #నిర్వచితమైతే GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #అయితే_ముగింపు

        features.push_back({ శూన్యసూచిక, శూన్యసూచిక });

        తిరిగి features;
    }();

    తిరిగి features.దత్తాంశం();

    GGML_వాడని(reg);
}

స్థిర శూన్య * ggml_backend_cpu_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name) {
    ఒకవేళ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_సమితి_n_దారాలు_t fct = ggml_backend_cpu_సమితి_n_దారాలు;
        తిరిగి (శూన్య *)fct;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        తిరిగి (శూన్య *)fct;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_get_features") == 0) {
        తిరిగి (శూన్య *)ggml_backend_cpu_get_features;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        తిరిగి (శూన్య *)ggml_backend_cpu_సమితి_abort_callback;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        తిరిగి (శూన్య *)ggml_numa_init;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        తిరిగి (శూన్య *)ggml_is_numa;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        తిరిగి (శూన్య *)ggml_backend_cpu_సమితి_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    ఒకవేళ (strcmp(name, "ggml_threadpool_new") == 0) {
        తిరిగి (శూన్య *)ggml_threadpool_new;
    }
    ఒకవేళ (strcmp(name, "ggml_threadpool_free") == 0) {
        తిరిగి (శూన్య *)ggml_threadpool_free;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        తిరిగి (శూన్య *)ggml_backend_cpu_సమితి_threadpool;
    }

    తిరిగి ఖాళీ;

    GGML_వాడని(reg);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_లెక్క,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_చిరునామా,
};

ggml_backend_reg_t ggml_backend_cpu_reg(శూన్య) {
    // init CPU feature detection
    ggml_cpu_init();

    స్థిర నిర్మాణం ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ ఖాళీ,
    };

    తిరిగి &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
