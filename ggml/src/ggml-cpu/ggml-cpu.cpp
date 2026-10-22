#समावेश "ggml-backend.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-cpu.h"
#समावेश "repack.h"
#समावेश "traits.h"
#समावेश "ggml-impl.h"
#समावेश "amx/amx.h"

#समावेश <cctype>
#समावेश <string>
#समावेश <vector>

#अगर_घोषित GGML_USE_CPU_HBM
#    include "hbm.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#पूर्ण_अगर

#अगर defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#अन्यथा
#    include <unistd.h>
#पूर्ण_अगर

#अगर defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#पूर्ण_अगर

// ggml-backend interface

std::vector<ggml_backend_बफर_type_t> & ggml_backend_cpu_get_extra_बफर_types() {
    अटल std::vector<ggml_backend_बफर_type_t> bufts = []() {
        std::vector<ggml_backend_बफर_type_t> bufts;

#अगर defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        अगर (ggml_backend_amx_बफर_type()) {
            bufts.push_back(ggml_backend_amx_बफर_type());
        }
#पूर्ण_अगर

#अगर_घोषित GGML_USE_CPU_RISCV64_SPACEMIT
        अगर (ggml_backend_cpu_riscv64_spacemit_बफर_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_बफर_type());
        }
#पूर्ण_अगर

#अगर_घोषित GGML_USE_CPU_KLEIDIAI
        अगर (ggml_backend_cpu_kleidiai_बफर_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_बफर_type());
        }
#पूर्ण_अगर

#अगर_घोषित GGML_USE_CPU_REPACK
        अगर (ggml_backend_cpu_repack_बफर_type()) {
            bufts.push_back(ggml_backend_cpu_repack_बफर_type());
        }
#पूर्ण_अगर

        वापस bufts;
    }();

    वापस bufts;
}

अटल ggml_backend_बफर_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    अटल std::vector<ggml_backend_बफर_type_t> extra_bufts = [] {
        std::vector<ggml_backend_बफर_type_t> bufts = ggml_backend_cpu_get_extra_बफर_types();
        bufts.push_back(nullptr);
        वापस bufts;
    }();

    वापस extra_bufts.तथ्य();

    GGML_UNUSED(device);
}

अटल bool ggml_backend_cpu_is_extra_बफर_type(ggml_backend_बफर_type_t buft) {
    क्रम (स्वतः * extra : ggml_backend_cpu_get_extra_बफर_types()) {
        अगर (extra == buft) {
            वापस true;
        }
    }
    वापस false;
}

// CPU backend - backend (stream)

काष्ठा ggml_backend_cpu_context {
    पूर्णांक                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_तथ्य;
    माप_प्रकार              work_आकार;

    ggml_abort_callback abort_callback;
    व्योम *              abort_callback_तथ्य;

    bool                use_ref;  // use reference implementation
};

अटल स्थिर अक्षर * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    वापस "CPU";

    GGML_UNUSED(backend);
}

अटल व्योम ggml_backend_cpu_free(ggml_backend_t backend) {
    काष्ठा ggml_backend_cpu_context * cpu_ctx = (काष्ठा ggml_backend_cpu_context *)backend->context;
    मिटाओ[] cpu_ctx->work_तथ्य;
    मिटाओ cpu_ctx;
    मिटाओ backend;
}

काष्ठा ggml_backend_plan_cpu {
    काष्ठा ggml_cplan cplan;
    काष्ठा ggml_cgraph cgraph;
};

अटल ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, स्थिर काष्ठा ggml_cgraph * cgraph) {
    काष्ठा ggml_backend_cpu_context * cpu_ctx = (काष्ठा ggml_backend_cpu_context *)backend->context;

    काष्ठा ggml_backend_plan_cpu * cpu_plan = नया ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    अगर (cpu_plan->cplan.work_आकार > 0) {
        cpu_plan->cplan.work_तथ्य = नया uint8_t[cpu_plan->cplan.work_आकार];
        अगर (cpu_plan->cplan.work_तथ्य == शून्य) {
            मिटाओ cpu_plan;
            वापस शून्य;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_तथ्य = cpu_ctx->abort_callback_तथ्य;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    वापस cpu_plan;
}

अटल व्योम ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    काष्ठा ggml_backend_plan_cpu * cpu_plan = (काष्ठा ggml_backend_plan_cpu *)plan;

    मिटाओ[] cpu_plan->cplan.work_तथ्य;
    मिटाओ cpu_plan;

    GGML_UNUSED(backend);
}

अटल क्रमागत ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    काष्ठा ggml_backend_plan_cpu * cpu_plan = (काष्ठा ggml_backend_plan_cpu *)plan;

    वापस ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

अटल क्रमागत ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, काष्ठा ggml_cgraph * cgraph) {
    काष्ठा ggml_backend_cpu_context * cpu_ctx = (काष्ठा ggml_backend_cpu_context *)backend->context;

    काष्ठा ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    अगर (cpu_ctx->work_आकार < cplan.work_आकार) {
        मिटाओ[] cpu_ctx->work_तथ्य;
        cpu_ctx->work_तथ्य = नया uint8_t[cplan.work_आकार];
        अगर (cpu_ctx->work_तथ्य == शून्य) {
            cpu_ctx->work_आकार = 0;
            वापस GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_आकार = cplan.work_आकार;
    }
    cplan.work_तथ्य = (uint8_t *)cpu_ctx->work_तथ्य;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_तथ्य = cpu_ctx->abort_callback_तथ्य;
    cplan.use_ref             = cpu_ctx->use_ref;

    वापस ggml_graph_compute(cgraph, &cplan);
}

अटल स्थिर काष्ठा ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ शून्य,
    /* .get_tensor_async        = */ शून्य,
    /* .set_tensor_2d_async     = */ शून्य,
    /* .get_tensor_2d_async     = */ शून्य,
    /* .cpy_tensor_async        = */ शून्य,
    /* .synchronize             = */ शून्य,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ शून्य,
    /* .event_wait              = */ शून्य,
    /* .graph_optimize          = */ शून्य,
};

अटल ggml_guid_t ggml_backend_cpu_guid(व्योम) {
    अटल ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    वापस &guid;
}

ggml_backend_t ggml_backend_cpu_init(व्योम) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    काष्ठा ggml_backend_cpu_context * ctx = नया ggml_backend_cpu_context;
    अगर (ctx == शून्य) {
        वापस शून्य;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = शून्य;
    ctx->work_तथ्य           = शून्य;
    ctx->work_आकार           = 0;
    ctx->abort_callback      = शून्य;
    ctx->abort_callback_तथ्य = शून्य;
    ctx->use_ref             = false;

    ggml_backend_t cpu_backend = नया ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    अगर (cpu_backend == शून्य) {
        मिटाओ ctx;
        वापस शून्य;
    }

    वापस cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    वापस backend != शून्य && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

व्योम ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, पूर्णांक n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    काष्ठा ggml_backend_cpu_context * ctx = (काष्ठा ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

व्योम ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    काष्ठा ggml_backend_cpu_context * ctx = (काष्ठा ggml_backend_cpu_context *)backend_cpu->context;

    अगर (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

व्योम ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, व्योम * abort_callback_तथ्य) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    काष्ठा ggml_backend_cpu_context * ctx = (काष्ठा ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_तथ्य = abort_callback_तथ्य;
}

व्योम ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    काष्ठा ggml_backend_cpu_context * ctx = (काष्ठा ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

काष्ठा ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#अगर_घोषित __APPLE__
        माप_प्रकार len = 0;
        अगर (!sysctlbyname("machdep.cpu.brand_string", शून्य, &len, शून्य, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, शून्य, 0); // NOLINT
        }
#या_अगर defined(__linux__)
        खाता * f = ख_खोलो("/proc/cpuinfo", "r");
        अगर (f) {
            अक्षर buf[1024];
            जबतक (ख_माला_लो(buf, माप(buf), f)) {
                अगर (म_भेदन(buf, "model name", 10) == 0) {
                    अक्षर * p = म_अक्षर(buf, ':');
                    अगर (p) {
                        p++;
                        जबतक (std::है_खाली(*p)) {
                            p++;
                        }
                        जबतक (std::है_खाली(p[म_माप(p) - 1])) {
                            p[म_माप(p) - 1] = '\0';
                        }
                        description = p;
                        अवरोध;
                    }
                }
            }
            ख_बंद(f);
        }
#या_अगर defined(_WIN32)
        HKEY h_कुंजी;
        अगर (Reg_Open_कुंजी_Ex(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        कुंजी_READ,
                        &h_कुंजी) == ERROR_SUCCESS) {
            DWORD cpu_brand_आकार = 0;
            अगर (Reg_Query_मूल्य_Ex_A(h_कुंजी,
                                "ProcessorNameString",
                                शून्य,
                                शून्य,
                                शून्य,
                                &cpu_brand_आकार) == ERROR_SUCCESS) {
                description.resize(cpu_brand_आकार);
                अगर (Reg_Query_मूल्य_Ex_A(h_कुंजी,
                                    "ProcessorNameString",
                                    शून्य,
                                    शून्य,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_आकार) == ERROR_SUCCESS) {
                    अगर (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_कुंजी(h_कुंजी);
        }
#पूर्ण_अगर
    }
};

अटल स्थिर अक्षर * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    वापस "CPU";

    GGML_UNUSED(dev);
}

अटल स्थिर अक्षर * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    काष्ठा ggml_backend_cpu_device_context * ctx = (काष्ठा ggml_backend_cpu_device_context *)dev->context;

    वापस ctx->description.c_str();
}

अटल व्योम ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
#अगर_घोषित _WIN32
    MEMORYSTATUSEX status;
    status.dw_लंबाई = माप(status);
    GlobalMemoryStatusEx(&status);
    *योग = status.ull_योग_Phys;
    *मुक्त = status.ullAvailPhys;
#अन्यथा
    दीर्घ pages = sysconf(_SC_PHYS_PAGES);
    दीर्घ page_आकार = sysconf(_SC_PAGE_SIZE);
    *योग = pages * page_आकार;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *मुक्त = *योग;
#पूर्ण_अगर // _WIN32

    GGML_UNUSED(dev);
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    वापस GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, काष्ठा ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_योग);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

अटल ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    वापस ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

अटल ggml_backend_बफर_type_t ggml_backend_cpu_device_get_बफर_type(ggml_backend_dev_t dev) {
    वापस ggml_backend_cpu_बफर_type();

    GGML_UNUSED(dev);
}

अटल ggml_backend_बफर_t ggml_backend_cpu_device_बफर_from_host_ptr(ggml_backend_dev_t dev, व्योम * ptr, माप_प्रकार size, माप_प्रकार max_tensor_आकार) {
    वापस ggml_backend_cpu_बफर_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_आकार);
}

अटल bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, स्थिर काष्ठा ggml_tensor * op) {
    स्थिर काष्ठा ggml_tensor * src0 = op->src[0];
    स्थिर काष्ठा ggml_tensor * src1 = op->src[1];

    अगर (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        वापस true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    क्रम (पूर्णांक i = 0; i < 4; i++) {
        अगर (op->src[i] && op->src[i]->बफर &&
            ggml_backend_cpu_is_extra_बफर_type(op->src[i]->बफर->buft)) {
            स्वतः * buf_extra = (ggml::cpu::extra_बफर_type *) op->src[i]->बफर->buft->context;
            वापस buf_extra->supports_op(dev, op);
        }
    }

    चयन (op->op) {
        हाल GGML_OP_CPY:
        हाल GGML_OP_SET_ROWS:
            वापस
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        हाल GGML_OP_MUL_MAT:
            अगर (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                वापस src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            वापस src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        हाल GGML_OP_SOFT_MAX_BACK: {
            अगर (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                वापस false;
            }
            भग्न max_bias = 0.0f;

            स_नकल(&max_bias, (स्थिर भग्न *) op->op_params + 1, माप(भग्न));

            वापस max_bias == 0.0f;
        }
        हाल GGML_OP_IM2COL_BACK:
            वापस src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        हाल GGML_OP_GET_ROWS_BACK:
            वापस src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        हाल GGML_OP_OUT_PROD:
            वापस (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        हाल GGML_OP_CONV_2D:
            वापस ggml_is_contiguous(op->src[0]);
        हाल GGML_OP_SSM_SCAN:
            वापस ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        शेष:
            वापस true;
    }
}

अटल bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    वापस ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_बफर_type(buft);
    GGML_UNUSED(dev);
}

अटल स्थिर काष्ठा ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_बफर_type,
    /* .get_host_buffer_type = */ शून्य,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_बफर_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ शून्य,
    /* .event_new            = */ शून्य,
    /* .event_free           = */ शून्य,
    /* .event_synchronize    = */ शून्य,
};

// CPU backend - backend (reg)

अटल स्थिर अक्षर * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    वापस "CPU";

    GGML_UNUSED(reg);
}

अटल माप_प्रकार ggml_backend_cpu_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    वापस 1;

    GGML_UNUSED(reg);
}

अटल ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    GGML_ASSERT(सूचकांक == 0);

    अटल ggml_backend_cpu_device_context ctx;
    अटल ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    वापस &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
अटल ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    अटल std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        अगर (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        अगर (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        अगर (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        अगर (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        अगर (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        अगर (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        अगर (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        अगर (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        अगर (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        अगर (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        अगर (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        अगर (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        अगर (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        अगर (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        अगर (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        अगर (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        अगर (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        अगर (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        अगर (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        अगर (ggml_cpu_get_sve_cnt() > 0) {
            अटल std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        अगर (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        अगर (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        अगर (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        अगर (ggml_cpu_get_rvv_vlen() > 0) {
            अटल std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        अगर (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        अगर (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        अगर (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        अगर (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #अगर_घोषित GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #पूर्ण_अगर
    #अगर_घोषित GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #पूर्ण_अगर
    #अगर_घोषित GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #पूर्ण_अगर
    #अगर_घोषित GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #पूर्ण_अगर
    #अगर_घोषित GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #पूर्ण_अगर

        features.push_back({ nullptr, nullptr });

        वापस features;
    }();

    वापस features.तथ्य();

    GGML_UNUSED(reg);
}

अटल व्योम * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर * name) {
    अगर (म_भेद(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        वापस (व्योम *)fct;
    }
    अगर (म_भेद(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        वापस (व्योम *)fct;
    }
    अगर (म_भेद(name, "ggml_backend_get_features") == 0) {
        वापस (व्योम *)ggml_backend_cpu_get_features;
    }
    अगर (म_भेद(name, "ggml_backend_set_abort_callback") == 0) {
        वापस (व्योम *)ggml_backend_cpu_set_abort_callback;
    }
    अगर (म_भेद(name, "ggml_backend_cpu_numa_init") == 0) {
        वापस (व्योम *)ggml_numa_init;
    }
    अगर (म_भेद(name, "ggml_backend_cpu_is_numa") == 0) {
        वापस (व्योम *)ggml_is_numa;
    }
    अगर (म_भेद(name, "ggml_backend_cpu_set_use_ref") == 0) {
        वापस (व्योम *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    अगर (म_भेद(name, "ggml_threadpool_new") == 0) {
        वापस (व्योम *)ggml_threadpool_new;
    }
    अगर (म_भेद(name, "ggml_threadpool_free") == 0) {
        वापस (व्योम *)ggml_threadpool_free;
    }
    अगर (म_भेद(name, "ggml_backend_cpu_set_threadpool") == 0) {
        वापस (व्योम *)ggml_backend_cpu_set_threadpool;
    }

    वापस शून्य;

    GGML_UNUSED(reg);
}

अटल स्थिर काष्ठा ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_गिनती,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(व्योम) {
    // init CPU feature detection
    ggml_cpu_init();

    अटल काष्ठा ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ शून्य,
    };

    वापस &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
