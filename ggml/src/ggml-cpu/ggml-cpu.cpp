#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "repack.h"
#अन्तर्भाव "traits.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "amx/amx.h"

#अन्तर्भाव <cctype>
#अन्तर्भाव <string>
#अन्तर्भाव <vector>

#यदि_परिभाषितम् GGML_USE_CPU_HBM
#    include "hbm.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#यदि_अन्तः

#यदि defined(_WIN32)
#    define WIN32_LEAN_AND_माध्यम्
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#अन्यथा
#    include <unistd.h>
#यदि_अन्तः

#यदि defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#यदि_अन्तः

// ggml-backend interface

std::vector<ggml_backend_बफर_type_t> & ggml_backend_cpu_get_extra_बफर_types() {
    स्थिर std::vector<ggml_backend_बफर_type_t> bufts = []() {
        std::vector<ggml_backend_बफर_type_t> bufts;

#यदि defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        यदि (ggml_backend_amx_बफर_type()) {
            bufts.push_back(ggml_backend_amx_बफर_type());
        }
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_CPU_RISCV64_SPACEMIT
        यदि (ggml_backend_cpu_riscv64_spacemit_बफर_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_बफर_type());
        }
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_CPU_KLEIDIAI
        यदि (ggml_backend_cpu_kleidiai_बफर_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_बफर_type());
        }
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_CPU_REPACK
        यदि (ggml_backend_cpu_repack_बफर_type()) {
            bufts.push_back(ggml_backend_cpu_repack_बफर_type());
        }
#यदि_अन्तः

        प्रत्यागम bufts;
    }();

    प्रत्यागम bufts;
}

स्थिर ggml_backend_बफर_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    स्थिर std::vector<ggml_backend_बफर_type_t> extra_bufts = [] {
        std::vector<ggml_backend_बफर_type_t> bufts = ggml_backend_cpu_get_extra_बफर_types();
        bufts.push_back(शून्यसूचकः);
        प्रत्यागम bufts;
    }();

    प्रत्यागम extra_bufts.दत्त();

    GGML_अप्रयुक्तम्(device);
}

स्थिर तार्किक ggml_backend_cpu_is_extra_बफर_type(ggml_backend_बफर_type_t buft) {
    प्रति (स्वचालित * extra : ggml_backend_cpu_get_extra_बफर_types()) {
        यदि (extra == buft) {
            प्रत्यागम सत्य;
        }
    }
    प्रत्यागम असत्य;
}

// CPU backend - backend (stream)

संरचना ggml_backend_cpu_context {
    पूर्णाङ्क                 n_तन्तवः;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_दत्त;
    size_t              work_आकार;

    ggml_abort_callback abort_callback;
    शून्य *              abort_callback_दत्त;

    तार्किक                use_ref;  // use reference implementation
};

स्थिर ध्रुव वर्ण * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    प्रत्यागम "CPU";

    GGML_अप्रयुक्तम्(backend);
}

स्थिर शून्य ggml_backend_cpu_free(ggml_backend_t backend) {
    संरचना ggml_backend_cpu_context * cpu_ctx = (संरचना ggml_backend_cpu_context *)backend->context;
    विलोपय[] cpu_ctx->work_दत्त;
    विलोपय cpu_ctx;
    विलोपय backend;
}

संरचना ggml_backend_plan_cpu {
    संरचना ggml_cplan cplan;
    संरचना ggml_cgraph cgraph;
};

स्थिर ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, ध्रुव संरचना ggml_cgraph * cgraph) {
    संरचना ggml_backend_cpu_context * cpu_ctx = (संरचना ggml_backend_cpu_context *)backend->context;

    संरचना ggml_backend_plan_cpu * cpu_plan = नूतन ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_तन्तवः, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    यदि (cpu_plan->cplan.work_आकार > 0) {
        cpu_plan->cplan.work_दत्त = नूतन uint8_t[cpu_plan->cplan.work_आकार];
        यदि (cpu_plan->cplan.work_दत्त == रिक्त) {
            विलोपय cpu_plan;
            प्रत्यागम रिक्त;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_दत्त = cpu_ctx->abort_callback_दत्त;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    प्रत्यागम cpu_plan;
}

स्थिर शून्य ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    संरचना ggml_backend_plan_cpu * cpu_plan = (संरचना ggml_backend_plan_cpu *)plan;

    विलोपय[] cpu_plan->cplan.work_दत्त;
    विलोपय cpu_plan;

    GGML_अप्रयुक्तम्(backend);
}

स्थिर गणना ggml_स्थितिः ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    संरचना ggml_backend_plan_cpu * cpu_plan = (संरचना ggml_backend_plan_cpu *)plan;

    प्रत्यागम ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_अप्रयुक्तम्(backend);
}

स्थिर गणना ggml_स्थितिः ggml_backend_cpu_graph_compute(ggml_backend_t backend, संरचना ggml_cgraph * cgraph) {
    संरचना ggml_backend_cpu_context * cpu_ctx = (संरचना ggml_backend_cpu_context *)backend->context;

    संरचना ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_तन्तवः, cpu_ctx->threadpool);

    यदि (cpu_ctx->work_आकार < cplan.work_आकार) {
        विलोपय[] cpu_ctx->work_दत्त;
        cpu_ctx->work_दत्त = नूतन uint8_t[cplan.work_आकार];
        यदि (cpu_ctx->work_दत्त == रिक्त) {
            cpu_ctx->work_आकार = 0;
            प्रत्यागम GGML_स्थितिः_ALLOC_FAILED;
        }
        cpu_ctx->work_आकार = cplan.work_आकार;
    }
    cplan.work_दत्त = (uint8_t *)cpu_ctx->work_दत्त;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_दत्त = cpu_ctx->abort_callback_दत्त;
    cplan.use_ref             = cpu_ctx->use_ref;

    प्रत्यागम ggml_graph_compute(cgraph, &cplan);
}

स्थिर ध्रुव संरचना ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ रिक्त,
    /* .get_tensor_async        = */ रिक्त,
    /* .set_tensor_2d_async     = */ रिक्त,
    /* .get_tensor_2d_async     = */ रिक्त,
    /* .cpy_tensor_async        = */ रिक्त,
    /* .synchronize             = */ रिक्त,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ रिक्त,
    /* .event_wait              = */ रिक्त,
    /* .graph_optimize          = */ रिक्त,
};

स्थिर ggml_guid_t ggml_backend_cpu_guid(शून्य) {
    स्थिर ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    प्रत्यागम &guid;
}

ggml_backend_t ggml_backend_cpu_init(शून्य) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    संरचना ggml_backend_cpu_context * ctx = नूतन ggml_backend_cpu_context;
    यदि (ctx == रिक्त) {
        प्रत्यागम रिक्त;
    }

    ctx->n_तन्तवः           = GGML_DEFAULT_N_तन्तवः;
    ctx->threadpool          = रिक्त;
    ctx->work_दत्त           = रिक्त;
    ctx->work_आकार           = 0;
    ctx->abort_callback      = रिक्त;
    ctx->abort_callback_दत्त = रिक्त;
    ctx->use_ref             = असत्य;

    ggml_backend_t cpu_backend = नूतन ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    यदि (cpu_backend == रिक्त) {
        विलोपय ctx;
        प्रत्यागम रिक्त;
    }

    प्रत्यागम cpu_backend;
}

तार्किक ggml_backend_is_cpu(ggml_backend_t backend) {
    प्रत्यागम backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

शून्य ggml_backend_cpu_समुच्चयः_n_तन्तवः(ggml_backend_t backend_cpu, पूर्णाङ्क n_तन्तवः) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    संरचना ggml_backend_cpu_context * ctx = (संरचना ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_तन्तवः = n_तन्तवः;
}

शून्य ggml_backend_cpu_समुच्चयः_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    संरचना ggml_backend_cpu_context * ctx = (संरचना ggml_backend_cpu_context *)backend_cpu->context;

    यदि (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

शून्य ggml_backend_cpu_समुच्चयः_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, शून्य * abort_callback_दत्त) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    संरचना ggml_backend_cpu_context * ctx = (संरचना ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_दत्त = abort_callback_दत्त;
}

शून्य ggml_backend_cpu_समुच्चयः_use_ref(ggml_backend_t backend_cpu, तार्किक use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    संरचना ggml_backend_cpu_context * ctx = (संरचना ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

संरचना ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#यदि_परिभाषितम् __APPLE__
        size_t len = 0;
        यदि (!sysctlbyname("machdep.cpu.brand_string", रिक्त, &len, रिक्त, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, रिक्त, 0); // NOLINT
        }
#नोचेद्यदि defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        यदि (f) {
            वर्ण buf[1024];
            यावत् (fgets(buf, परिमाण(buf), f)) {
                यदि (वाक्ये_तुलय(buf, "model name", 10) == 0) {
                    वर्ण * p = strchr(buf, ':');
                    यदि (p) {
                        p++;
                        यावत् (std::रिक्तम्_वा(*p)) {
                            p++;
                        }
                        यावत् (std::रिक्तम्_वा(p[वाक्यदैर्घ्यम्(p) - 1])) {
                            p[वाक्यदैर्घ्यम्(p) - 1] = '\0';
                        }
                        description = p;
                        भङ्ग;
                    }
                }
            }
            fclose(f);
        }
#नोचेद्यदि defined(_WIN32)
        HKEY h_कुञ्चिका;
        यदि (Reg_Open_कुञ्चिका_Ex(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        कुञ्चिका_READ,
                        &h_कुञ्चिका) == ERROR_SUCCESS) {
            DWORD cpu_brand_आकार = 0;
            यदि (Reg_Query_मानम्_Ex_A(h_कुञ्चिका,
                                "ProcessorNameString",
                                रिक्त,
                                रिक्त,
                                रिक्त,
                                &cpu_brand_आकार) == ERROR_SUCCESS) {
                description.resize(cpu_brand_आकार);
                यदि (Reg_Query_मानम्_Ex_A(h_कुञ्चिका,
                                    "ProcessorNameString",
                                    रिक्त,
                                    रिक्त,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_आकार) == ERROR_SUCCESS) {
                    यदि (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_कुञ्चिका(h_कुञ्चिका);
        }
#यदि_अन्तः
    }
};

स्थिर ध्रुव वर्ण * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    प्रत्यागम "CPU";

    GGML_अप्रयुक्तम्(dev);
}

स्थिर ध्रुव वर्ण * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    संरचना ggml_backend_cpu_device_context * ctx = (संरचना ggml_backend_cpu_device_context *)dev->context;

    प्रत्यागम ctx->description.c_str();
}

स्थिर शून्य ggml_backend_cpu_device_get_स्मृतिः(ggml_backend_dev_t dev, size_t * मुक्त, size_t * योग) {
#यदि_परिभाषितम् _WIN32
    MEMORYSTATUSEX स्थितिः;
    स्थितिः.dw_दीर्घता = परिमाण(स्थितिः);
    Global_स्मृतिः_स्थितिः_Ex(&स्थितिः);
    *योग = स्थितिः.ull_योग_Phys;
    *मुक्त = स्थितिः.ullAvailPhys;
#अन्यथा
    दीर्घ पृष्ठानि = sysconf(_SC_PHYS_PAGES);
    दीर्घ page_आकार = sysconf(_SC_PAGE_SIZE);
    *योग = पृष्ठानि * page_आकार;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *मुक्त = *योग;
#यदि_अन्तः // _WIN32

    GGML_अप्रयुक्तम्(dev);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    प्रत्यागम GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_अप्रयुक्तम्(dev);
}

स्थिर शून्य ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_स्मृतिः(dev, &props->स्मृतिः_free, &props->स्मृतिः_योग);
    props->caps = {
        /* .async                 = */ असत्य,
        /* .host_buffer           = */ असत्य,
        /* .buffer_from_host_ptr  = */ सत्य,
        /* .events                = */ असत्य,
        /* .mmap_support          = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    प्रत्यागम ggml_backend_cpu_init();

    GGML_अप्रयुक्तम्(dev);
    GGML_अप्रयुक्तम्(params);
}

स्थिर ggml_backend_बफर_type_t ggml_backend_cpu_device_get_बफर_type(ggml_backend_dev_t dev) {
    प्रत्यागम ggml_backend_cpu_बफर_type();

    GGML_अप्रयुक्तम्(dev);
}

स्थिर ggml_backend_बफर_t ggml_backend_cpu_device_बफर_from_host_ptr(ggml_backend_dev_t dev, शून्य * ptr, size_t size, size_t max_tensor_आकार) {
    प्रत्यागम ggml_backend_cpu_बफर_from_ptr(ptr, size);

    GGML_अप्रयुक्तम्(dev);
    GGML_अप्रयुक्तम्(max_tensor_आकार);
}

स्थिर तार्किक ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, ध्रुव संरचना ggml_tensor * op) {
    ध्रुव संरचना ggml_tensor * src0 = op->src[0];
    ध्रुव संरचना ggml_tensor * src1 = op->src[1];

    यदि (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        प्रत्यागम सत्य;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
        यदि (op->src[i] && op->src[i]->बफर &&
            ggml_backend_cpu_is_extra_बफर_type(op->src[i]->बफर->buft)) {
            स्वचालित * buf_extra = (ggml::cpu::extra_बफर_type *) op->src[i]->बफर->buft->context;
            प्रत्यागम buf_extra->supports_op(dev, op);
        }
    }

    विकल्प (op->op) {
        पक्ष GGML_OP_CPY:
        पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
            प्रत्यागम
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        पक्ष GGML_OP_MUL_MAT:
            यदि (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                प्रत्यागम src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            प्रत्यागम src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        पक्ष GGML_OP_SOFT_MAX_पृष्ठभागः: {
            यदि (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                प्रत्यागम असत्य;
            }
            भिन्न max_bias = 0.0f;

            स्मृतिं_प्रतिलिख(&max_bias, (ध्रुव भिन्न *) op->op_params + 1, परिमाण(भिन्न));

            प्रत्यागम max_bias == 0.0f;
        }
        पक्ष GGML_OP_IM2_COL_पृष्ठभागः:
            प्रत्यागम src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        पक्ष GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः:
            प्रत्यागम src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        पक्ष GGML_OP_बहिर्गमनम्_PROD:
            प्रत्यागम (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_CONV_2D:
            प्रत्यागम ggml_is_contiguous(op->src[0]);
        पक्ष GGML_OP_SSM_अवलोकनम्:
            प्रत्यागम ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        स्वभाव:
            प्रत्यागम सत्य;
    }
}

स्थिर तार्किक ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    प्रत्यागम ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_बफर_type(buft);
    GGML_अप्रयुक्तम्(dev);
}

स्थिर ध्रुव संरचना ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_स्मृतिः,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_बफर_type,
    /* .get_host_buffer_type = */ रिक्त,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_बफर_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ रिक्त,
    /* .event_new            = */ रिक्त,
    /* .event_free           = */ रिक्त,
    /* .event_synchronize    = */ रिक्त,
};

// CPU backend - backend (reg)

स्थिर ध्रुव वर्ण * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    प्रत्यागम "CPU";

    GGML_अप्रयुक्तम्(reg);
}

स्थिर size_t ggml_backend_cpu_reg_get_device_गणनम्(ggml_backend_reg_t reg) {
    प्रत्यागम 1;

    GGML_अप्रयुक्तम्(reg);
}

स्थिर ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t सूचकांक) {
    GGML_ASSERT(सूचकांक == 0);

    स्थिर ggml_backend_cpu_device_context ctx;
    स्थिर ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    प्रत्यागम &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
स्थिर ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    स्थिर std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        यदि (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        यदि (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        यदि (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        यदि (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        यदि (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        यदि (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        यदि (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        यदि (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        यदि (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        यदि (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        यदि (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        यदि (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        यदि (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        यदि (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        यदि (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        यदि (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        यदि (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        यदि (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        यदि (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        यदि (ggml_cpu_get_sve_cnt() > 0) {
            स्थिर std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        यदि (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        यदि (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        यदि (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        यदि (ggml_cpu_get_rvv_vlen() > 0) {
            स्थिर std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        यदि (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        यदि (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        यदि (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        यदि (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #यदि_परिभाषितम् GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #यदि_अन्तः
    #यदि_परिभाषितम् GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #यदि_अन्तः
    #यदि_परिभाषितम् GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #यदि_अन्तः
    #यदि_परिभाषितम् GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #यदि_अन्तः
    #यदि_परिभाषितम् GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #यदि_अन्तः

        features.push_back({ शून्यसूचकः, शून्यसूचकः });

        प्रत्यागम features;
    }();

    प्रत्यागम features.दत्त();

    GGML_अप्रयुक्तम्(reg);
}

स्थिर शून्य * ggml_backend_cpu_get_proc_स्थानाङ्कः(ggml_backend_reg_t reg, ध्रुव वर्ण * name) {
    यदि (वाक्यं_तुलय(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_समुच्चयः_n_तन्तवः_t fct = ggml_backend_cpu_समुच्चयः_n_तन्तवः;
        प्रत्यागम (शून्य *)fct;
    }
    यदि (वाक्यं_तुलय(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        प्रत्यागम (शून्य *)fct;
    }
    यदि (वाक्यं_तुलय(name, "ggml_backend_get_features") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_cpu_get_features;
    }
    यदि (वाक्यं_तुलय(name, "ggml_backend_set_abort_callback") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_cpu_समुच्चयः_abort_callback;
    }
    यदि (वाक्यं_तुलय(name, "ggml_backend_cpu_numa_init") == 0) {
        प्रत्यागम (शून्य *)ggml_numa_init;
    }
    यदि (वाक्यं_तुलय(name, "ggml_backend_cpu_is_numa") == 0) {
        प्रत्यागम (शून्य *)ggml_is_numa;
    }
    यदि (वाक्यं_तुलय(name, "ggml_backend_cpu_set_use_ref") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_cpu_समुच्चयः_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    यदि (वाक्यं_तुलय(name, "ggml_threadpool_new") == 0) {
        प्रत्यागम (शून्य *)ggml_threadpool_new;
    }
    यदि (वाक्यं_तुलय(name, "ggml_threadpool_free") == 0) {
        प्रत्यागम (शून्य *)ggml_threadpool_free;
    }
    यदि (वाक्यं_तुलय(name, "ggml_backend_cpu_set_threadpool") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_cpu_समुच्चयः_threadpool;
    }

    प्रत्यागम रिक्त;

    GGML_अप्रयुक्तम्(reg);
}

स्थिर ध्रुव संरचना ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_गणनम्,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_स्थानाङ्कः,
};

ggml_backend_reg_t ggml_backend_cpu_reg(शून्य) {
    // init CPU feature detection
    ggml_cpu_init();

    स्थिर संरचना ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_संस्करणम्,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ रिक्त,
    };

    प्रत्यागम &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
