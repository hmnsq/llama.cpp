#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-cpu.h"
#അന്തര്ഭാവ "repack.h"
#അന്തര്ഭാവ "traits.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "amx/amx.h"

#അന്തര്ഭാവ <cctype>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#എങ്കിൽ defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#അല്ലെങ്കിൽ
#    include <unistd.h>
#endif

#എങ്കിൽ defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    സ്ഥിതം std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#എങ്കിൽ defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        എങ്കിൽ (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        എങ്കിൽ (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        എങ്കിൽ (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        എങ്കിൽ (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        തിരികെ bufts;
    }();

    തിരികെ bufts;
}

സ്ഥിതം ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    സ്ഥിതം std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        തിരികെ bufts;
    }();

    തിരികെ extra_bufts.data();

    GGML_UNUSED(device);
}

സ്ഥിതം തർക്കം ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    വേണ്ടി (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        എങ്കിൽ (extra == buft) {
            തിരികെ സത്യം;
        }
    }
    തിരികെ അസത്യം;
}

// CPU backend - backend (stream)

ഘടന ggml_backend_cpu_context {
    പൂർണ്ണസംഖ്യ                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    ശൂന്യം *              abort_callback_data;

    തർക്കം                use_ref;  // use reference implementation
};

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    തിരികെ "CPU";

    GGML_UNUSED(backend);
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_free(ggml_backend_t backend) {
    ഘടന ggml_backend_cpu_context * cpu_ctx = (ഘടന ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

ഘടന ggml_backend_plan_cpu {
    ഘടന ggml_cplan cplan;
    ഘടന ggml_cgraph cgraph;
};

സ്ഥിതം ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, സ്ഥിരം ഘടന ggml_cgraph * cgraph) {
    ഘടന ggml_backend_cpu_context * cpu_ctx = (ഘടന ggml_backend_cpu_context *)backend->context;

    ഘടന ggml_backend_plan_cpu * cpu_plan = പുതിയ ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    എങ്കിൽ (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = പുതിയ uint8_t[cpu_plan->cplan.work_size];
        എങ്കിൽ (cpu_plan->cplan.work_data == ശൂന്യ) {
            delete cpu_plan;
            തിരികെ ശൂന്യ;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    തിരികെ cpu_plan;
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ഘടന ggml_backend_plan_cpu * cpu_plan = (ഘടന ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ഘടന ggml_backend_plan_cpu * cpu_plan = (ഘടന ggml_backend_plan_cpu *)plan;

    തിരികെ ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, ഘടന ggml_cgraph * cgraph) {
    ഘടന ggml_backend_cpu_context * cpu_ctx = (ഘടന ggml_backend_cpu_context *)backend->context;

    ഘടന ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    എങ്കിൽ (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = പുതിയ uint8_t[cplan.work_size];
        എങ്കിൽ (cpu_ctx->work_data == ശൂന്യ) {
            cpu_ctx->work_size = 0;
            തിരികെ GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    തിരികെ ggml_graph_compute(cgraph, &cplan);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ ശൂന്യ,
    /* .get_tensor_async        = */ ശൂന്യ,
    /* .set_tensor_2d_async     = */ ശൂന്യ,
    /* .get_tensor_2d_async     = */ ശൂന്യ,
    /* .cpy_tensor_async        = */ ശൂന്യ,
    /* .synchronize             = */ ശൂന്യ,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ ശൂന്യ,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ ശൂന്യ,
    /* .event_wait              = */ ശൂന്യ,
    /* .graph_optimize          = */ ശൂന്യ,
};

സ്ഥിതം ggml_guid_t ggml_backend_cpu_guid(ശൂന്യം) {
    സ്ഥിതം ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    തിരികെ &guid;
}

ggml_backend_t ggml_backend_cpu_init(ശൂന്യം) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    ഘടന ggml_backend_cpu_context * ctx = പുതിയ ggml_backend_cpu_context;
    എങ്കിൽ (ctx == ശൂന്യ) {
        തിരികെ ശൂന്യ;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = ശൂന്യ;
    ctx->work_data           = ശൂന്യ;
    ctx->work_size           = 0;
    ctx->abort_callback      = ശൂന്യ;
    ctx->abort_callback_data = ശൂന്യ;
    ctx->use_ref             = അസത്യം;

    ggml_backend_t cpu_backend = പുതിയ ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    എങ്കിൽ (cpu_backend == ശൂന്യ) {
        delete ctx;
        തിരികെ ശൂന്യ;
    }

    തിരികെ cpu_backend;
}

തർക്കം ggml_backend_is_cpu(ggml_backend_t backend) {
    തിരികെ backend != ശൂന്യ && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

ശൂന്യം ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, പൂർണ്ണസംഖ്യ n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ഘടന ggml_backend_cpu_context * ctx = (ഘടന ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

ശൂന്യം ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ഘടന ggml_backend_cpu_context * ctx = (ഘടന ggml_backend_cpu_context *)backend_cpu->context;

    എങ്കിൽ (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

ശൂന്യം ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, ശൂന്യം * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ഘടന ggml_backend_cpu_context * ctx = (ഘടന ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

ശൂന്യം ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, തർക്കം use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ഘടന ggml_backend_cpu_context * ctx = (ഘടന ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

ഘടന ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        എങ്കിൽ (!sysctlbyname("machdep.cpu.brand_string", ശൂന്യ, &len, ശൂന്യ, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, ശൂന്യ, 0); // NOLINT
        }
#അല്ലെങ്കിലെങ്കിൽ defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        എങ്കിൽ (f) {
            അക്ഷരം buf[1024];
            ആകുമ്പോൾ (fgets(buf, വലുപ്പം(buf), f)) {
                എങ്കിൽ (strncmp(buf, "model name", 10) == 0) {
                    അക്ഷരം * p = strchr(buf, ':');
                    എങ്കിൽ (p) {
                        p++;
                        ആകുമ്പോൾ (std::isspace(*p)) {
                            p++;
                        }
                        ആകുമ്പോൾ (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        നിർത്തുക;
                    }
                }
            }
            fclose(f);
        }
#അല്ലെങ്കിലെങ്കിൽ defined(_WIN32)
        HKEY h_താക്കോൽ;
        എങ്കിൽ (Reg_Open_താക്കോൽ_Ex(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        താക്കോൽ_READ,
                        &h_താക്കോൽ) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            എങ്കിൽ (Reg_Query_മൂല്യം_Ex_A(h_താക്കോൽ,
                                "ProcessorNameString",
                                ശൂന്യ,
                                ശൂന്യ,
                                ശൂന്യ,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                എങ്കിൽ (Reg_Query_മൂല്യം_Ex_A(h_താക്കോൽ,
                                    "ProcessorNameString",
                                    ശൂന്യ,
                                    ശൂന്യ,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    എങ്കിൽ (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_താക്കോൽ(h_താക്കോൽ);
        }
#endif
    }
};

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    തിരികെ "CPU";

    GGML_UNUSED(dev);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    ഘടന ggml_backend_cpu_device_context * ctx = (ഘടന ggml_backend_cpu_device_context *)dev->context;

    തിരികെ ctx->description.c_str();
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * സ്വതന്ത്രം, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = വലുപ്പം(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *സ്വതന്ത്രം = status.ullAvailPhys;
#അല്ലെങ്കിൽ
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *സ്വതന്ത്രം = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

സ്ഥിതം എണ്ണൽ ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    തിരികെ GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, ഘടന ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ അസത്യം,
        /* .host_buffer           = */ അസത്യം,
        /* .buffer_from_host_ptr  = */ സത്യം,
        /* .events                = */ അസത്യം,
        /* .mmap_support          = */ സത്യം,
    };
}

സ്ഥിതം ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, സ്ഥിരം അക്ഷരം * params) {
    തിരികെ ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    തിരികെ ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ശൂന്യം * ptr, size_t size, size_t max_tensor_size) {
    തിരികെ ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

സ്ഥിതം തർക്കം ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];

    എങ്കിൽ (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        തിരികെ സത്യം;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
        എങ്കിൽ (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            തിരികെ buf_extra->supports_op(dev, op);
        }
    }

    തിരഞ്ഞെടുപ്പ് (op->op) {
        അവസ്ഥ GGML_OP_CPY:
        അവസ്ഥ GGML_OP_SET_ROWS:
            തിരികെ
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        അവസ്ഥ GGML_OP_MUL_MAT:
            എങ്കിൽ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                തിരികെ src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            തിരികെ src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        അവസ്ഥ GGML_OP_SOFT_MAX_BACK: {
            എങ്കിൽ (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                തിരികെ അസത്യം;
            }
            ദശാംശം max_bias = 0.0f;

            memcpy(&max_bias, (സ്ഥിരം ദശാംശം *) op->op_params + 1, വലുപ്പം(ദശാംശം));

            തിരികെ max_bias == 0.0f;
        }
        അവസ്ഥ GGML_OP_IM2COL_BACK:
            തിരികെ src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        അവസ്ഥ GGML_OP_GET_ROWS_BACK:
            തിരികെ src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        അവസ്ഥ GGML_OP_OUT_PROD:
            തിരികെ (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        അവസ്ഥ GGML_OP_CONV_2D:
            തിരികെ ggml_is_contiguous(op->src[0]);
        അവസ്ഥ GGML_OP_SSM_SCAN:
            തിരികെ ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        സ്ഥിരസ്ഥിതി:
            തിരികെ സത്യം;
    }
}

സ്ഥിതം തർക്കം ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    തിരികെ ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ ശൂന്യ,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ ശൂന്യ,
    /* .event_new            = */ ശൂന്യ,
    /* .event_free           = */ ശൂന്യ,
    /* .event_synchronize    = */ ശൂന്യ,
};

// CPU backend - backend (reg)

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    തിരികെ "CPU";

    GGML_UNUSED(reg);
}

സ്ഥിതം size_t ggml_backend_cpu_reg_get_device_എണ്ണം(ggml_backend_reg_t reg) {
    തിരികെ 1;

    GGML_UNUSED(reg);
}

സ്ഥിതം ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    സ്ഥിതം ggml_backend_cpu_device_context ctx;
    സ്ഥിതം ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    തിരികെ &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
സ്ഥിതം ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    സ്ഥിതം std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        എങ്കിൽ (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        എങ്കിൽ (ggml_cpu_get_sve_cnt() > 0) {
            സ്ഥിതം std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        എങ്കിൽ (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        എങ്കിൽ (ggml_cpu_get_rvv_vlen() > 0) {
            സ്ഥിതം std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        എങ്കിൽ (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        എങ്കിൽ (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        തിരികെ features;
    }();

    തിരികെ features.data();

    GGML_UNUSED(reg);
}

സ്ഥിതം ശൂന്യം * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, സ്ഥിരം അക്ഷരം * name) {
    എങ്കിൽ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        തിരികെ (ശൂന്യം *)fct;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        തിരികെ (ശൂന്യം *)fct;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_get_features") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_cpu_get_features;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_cpu_set_abort_callback;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        തിരികെ (ശൂന്യം *)ggml_numa_init;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        തിരികെ (ശൂന്യം *)ggml_is_numa;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    എങ്കിൽ (strcmp(name, "ggml_threadpool_new") == 0) {
        തിരികെ (ശൂന്യം *)ggml_threadpool_new;
    }
    എങ്കിൽ (strcmp(name, "ggml_threadpool_free") == 0) {
        തിരികെ (ശൂന്യം *)ggml_threadpool_free;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_cpu_set_threadpool;
    }

    തിരികെ ശൂന്യ;

    GGML_UNUSED(reg);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_എണ്ണം,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(ശൂന്യം) {
    // init CPU feature detection
    ggml_cpu_init();

    സ്ഥിതം ഘടന ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ ശൂന്യ,
    };

    തിരികെ &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
