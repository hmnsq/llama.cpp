#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu.h"
#ଅନ୍ତର୍ଭାଵ "repack.h"
#ଅନ୍ତର୍ଭାଵ "traits.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "amx/amx.h"

#ଅନ୍ତର୍ଭାଵ <cctype>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#ଯଦି defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#ଅନ୍ୟଥା
#    include <unistd.h>
#endif

#ଯଦି defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    ସ୍ଥିର std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#ଯଦି defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        ଯଦି (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        ଯଦି (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        ଯଦି (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        ଯଦି (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        ଫେରନ୍ତୁ bufts;
    }();

    ଫେରନ୍ତୁ bufts;
}

ସ୍ଥିର ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    ସ୍ଥିର std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        ଫେରନ୍ତୁ bufts;
    }();

    ଫେରନ୍ତୁ extra_bufts.data();

    GGML_UNUSED(device);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    ପାଇଁ (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        ଯଦି (extra == buft) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

// CPU backend - backend (stream)

ସଂରଚନା ggml_backend_cpu_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    ଶୂନ୍ୟ *              abort_callback_data;

    ତାର୍କିକ                use_ref;  // use reference implementation
};

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    ଫେରନ୍ତୁ "CPU";

    GGML_UNUSED(backend);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_cpu_free(ggml_backend_t backend) {
    ସଂରଚନା ggml_backend_cpu_context * cpu_ctx = (ସଂରଚନା ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

ସଂରଚନା ggml_backend_plan_cpu {
    ସଂରଚନା ggml_cplan cplan;
    ସଂରଚନା ggml_cgraph cgraph;
};

ସ୍ଥିର ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph) {
    ସଂରଚନା ggml_backend_cpu_context * cpu_ctx = (ସଂରଚନା ggml_backend_cpu_context *)backend->context;

    ସଂରଚନା ggml_backend_plan_cpu * cpu_plan = ନୂତନ ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    ଯଦି (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = ନୂତନ uint8_t[cpu_plan->cplan.work_size];
        ଯଦି (cpu_plan->cplan.work_data == ଖାଲି) {
            delete cpu_plan;
            ଫେରନ୍ତୁ ଖାଲି;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    ଫେରନ୍ତୁ cpu_plan;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ସଂରଚନା ggml_backend_plan_cpu * cpu_plan = (ସଂରଚନା ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ସଂରଚନା ggml_backend_plan_cpu * cpu_plan = (ସଂରଚନା ggml_backend_plan_cpu *)plan;

    ଫେରନ୍ତୁ ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, ସଂରଚନା ggml_cgraph * cgraph) {
    ସଂରଚନା ggml_backend_cpu_context * cpu_ctx = (ସଂରଚନା ggml_backend_cpu_context *)backend->context;

    ସଂରଚନା ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    ଯଦି (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = ନୂତନ uint8_t[cplan.work_size];
        ଯଦି (cpu_ctx->work_data == ଖାଲି) {
            cpu_ctx->work_size = 0;
            ଫେରନ୍ତୁ GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    ଫେରନ୍ତୁ ggml_graph_compute(cgraph, &cplan);
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ ଖାଲି,
    /* .get_tensor_async        = */ ଖାଲି,
    /* .set_tensor_2d_async     = */ ଖାଲି,
    /* .get_tensor_2d_async     = */ ଖାଲି,
    /* .cpy_tensor_async        = */ ଖାଲି,
    /* .synchronize             = */ ଖାଲି,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ ଖାଲି,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ ଖାଲି,
    /* .event_wait              = */ ଖାଲି,
    /* .graph_optimize          = */ ଖାଲି,
};

ସ୍ଥିର ggml_guid_t ggml_backend_cpu_guid(ଶୂନ୍ୟ) {
    ସ୍ଥିର ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    ଫେରନ୍ତୁ &guid;
}

ggml_backend_t ggml_backend_cpu_init(ଶୂନ୍ୟ) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    ସଂରଚନା ggml_backend_cpu_context * ctx = ନୂତନ ggml_backend_cpu_context;
    ଯଦି (ctx == ଖାଲି) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = ଖାଲି;
    ctx->work_data           = ଖାଲି;
    ctx->work_size           = 0;
    ctx->abort_callback      = ଖାଲି;
    ctx->abort_callback_data = ଖାଲି;
    ctx->use_ref             = ମିଥ୍ୟା;

    ggml_backend_t cpu_backend = ନୂତନ ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    ଯଦି (cpu_backend == ଖାଲି) {
        delete ctx;
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଫେରନ୍ତୁ cpu_backend;
}

ତାର୍କିକ ggml_backend_is_cpu(ggml_backend_t backend) {
    ଫେରନ୍ତୁ backend != ଖାଲି && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

ଶୂନ୍ୟ ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ସଂରଚନା ggml_backend_cpu_context * ctx = (ସଂରଚନା ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

ଶୂନ୍ୟ ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ସଂରଚନା ggml_backend_cpu_context * ctx = (ସଂରଚନା ggml_backend_cpu_context *)backend_cpu->context;

    ଯଦି (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

ଶୂନ୍ୟ ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, ଶୂନ୍ୟ * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ସଂରଚନା ggml_backend_cpu_context * ctx = (ସଂରଚନା ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

ଶୂନ୍ୟ ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, ତାର୍କିକ use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ସଂରଚନା ggml_backend_cpu_context * ctx = (ସଂରଚନା ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

ସଂରଚନା ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        ଯଦି (!sysctlbyname("machdep.cpu.brand_string", ଖାଲି, &len, ଖାଲି, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, ଖାଲି, 0); // NOLINT
        }
#ନଚେତଯଦି defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        ଯଦି (f) {
            ଅକ୍ଷର buf[1024];
            ଯେତେବେଳେ (fgets(buf, ଆକାର(buf), f)) {
                ଯଦି (strncmp(buf, "model name", 10) == 0) {
                    ଅକ୍ଷର * p = strchr(buf, ':');
                    ଯଦି (p) {
                        p++;
                        ଯେତେବେଳେ (std::isspace(*p)) {
                            p++;
                        }
                        ଯେତେବେଳେ (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        ଭାଙ୍ଗ;
                    }
                }
            }
            fclose(f);
        }
#ନଚେତଯଦି defined(_WIN32)
        HKEY h_ଚାବି;
        ଯଦି (Reg_Open_ଚାବି_Ex(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        ଚାବି_READ,
                        &h_ଚାବି) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            ଯଦି (Reg_Query_ମୂଲ୍ୟ_Ex_A(h_ଚାବି,
                                "ProcessorNameString",
                                ଖାଲି,
                                ଖାଲି,
                                ଖାଲି,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                ଯଦି (Reg_Query_ମୂଲ୍ୟ_Ex_A(h_ଚାବି,
                                    "ProcessorNameString",
                                    ଖାଲି,
                                    ଖାଲି,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    ଯଦି (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_ଚାବି(h_ଚାବି);
        }
#endif
    }
};

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ "CPU";

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    ସଂରଚନା ggml_backend_cpu_device_context * ctx = (ସଂରଚନା ggml_backend_cpu_device_context *)dev->context;

    ଫେରନ୍ତୁ ctx->description.c_str();
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * ମୁକ୍ତ, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = ଆକାର(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *ମୁକ୍ତ = status.ullAvailPhys;
#ଅନ୍ୟଥା
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *ମୁକ୍ତ = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଗଣନା ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, ସଂରଚନା ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ମିଥ୍ୟା,
        /* .host_buffer           = */ ମିଥ୍ୟା,
        /* .buffer_from_host_ptr  = */ ସତ୍ୟ,
        /* .events                = */ ମିଥ୍ୟା,
        /* .mmap_support          = */ ସତ୍ୟ,
    };
}

ସ୍ଥିର ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params) {
    ଫେରନ୍ତୁ ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ଶୂନ୍ୟ * ptr, size_t size, size_t max_tensor_size) {
    ଫେରନ୍ତୁ ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];

    ଯଦି (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; i++) {
        ଯଦି (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            ଫେରନ୍ତୁ buf_extra->supports_op(dev, op);
        }
    }

    ବାଛ (op->op) {
        ସ୍ଥିତି GGML_OP_CPY:
        ସ୍ଥିତି GGML_OP_SET_ROWS:
            ଫେରନ୍ତୁ
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        ସ୍ଥିତି GGML_OP_MUL_MAT:
            ଯଦି (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                ଫେରନ୍ତୁ src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            ଫେରନ୍ତୁ src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        ସ୍ଥିତି GGML_OP_SOFT_MAX_BACK: {
            ଯଦି (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଦଶମିକ max_bias = 0.0f;

            memcpy(&max_bias, (ଅଚଳ ଦଶମିକ *) op->op_params + 1, ଆକାର(ଦଶମିକ));

            ଫେରନ୍ତୁ max_bias == 0.0f;
        }
        ସ୍ଥିତି GGML_OP_IM2COL_BACK:
            ଫେରନ୍ତୁ src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        ସ୍ଥିତି GGML_OP_GET_ROWS_BACK:
            ଫେରନ୍ତୁ src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        ସ୍ଥିତି GGML_OP_OUT_PROD:
            ଫେରନ୍ତୁ (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ସ୍ଥିତି GGML_OP_CONV_2D:
            ଫେରନ୍ତୁ ggml_is_contiguous(op->src[0]);
        ସ୍ଥିତି GGML_OP_SSM_SCAN:
            ଫେରନ୍ତୁ ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ସତ୍ୟ;
    }
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ ଖାଲି,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ ଖାଲି,
    /* .event_new            = */ ଖାଲି,
    /* .event_free           = */ ଖାଲି,
    /* .event_synchronize    = */ ଖାଲି,
};

// CPU backend - backend (reg)

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ "CPU";

    GGML_UNUSED(reg);
}

ସ୍ଥିର size_t ggml_backend_cpu_reg_get_device_ଗଣନା(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ 1;

    GGML_UNUSED(reg);
}

ସ୍ଥିର ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ସ୍ଥିର ggml_backend_cpu_device_context ctx;
    ସ୍ଥିର ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    ଫେରନ୍ତୁ &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
ସ୍ଥିର ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    ସ୍ଥିର std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        ଯଦି (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        ଯଦି (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        ଯଦି (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        ଯଦି (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        ଯଦି (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        ଯଦି (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        ଯଦି (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        ଯଦି (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        ଯଦି (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        ଯଦି (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        ଯଦି (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        ଯଦି (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        ଯଦି (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        ଯଦି (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        ଯଦି (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        ଯଦି (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        ଯଦି (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        ଯଦି (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        ଯଦି (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        ଯଦି (ggml_cpu_get_sve_cnt() > 0) {
            ସ୍ଥିର std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        ଯଦି (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        ଯଦି (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        ଯଦି (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        ଯଦି (ggml_cpu_get_rvv_vlen() > 0) {
            ସ୍ଥିର std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        ଯଦି (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        ଯଦି (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        ଯଦି (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        ଯଦି (ggml_cpu_has_llamafile()) {
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

        ଫେରନ୍ତୁ features;
    }();

    ଫେରନ୍ତୁ features.data();

    GGML_UNUSED(reg);
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, ଅଚଳ ଅକ୍ଷର * name) {
    ଯଦି (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)fct;
    }
    ଯଦି (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)fct;
    }
    ଯଦି (strcmp(name, "ggml_backend_get_features") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_cpu_get_features;
    }
    ଯଦି (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_cpu_set_abort_callback;
    }
    ଯଦି (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_numa_init;
    }
    ଯଦି (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_is_numa;
    }
    ଯଦି (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    ଯଦି (strcmp(name, "ggml_threadpool_new") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_threadpool_new;
    }
    ଯଦି (strcmp(name, "ggml_threadpool_free") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_threadpool_free;
    }
    ଯଦି (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_cpu_set_threadpool;
    }

    ଫେରନ୍ତୁ ଖାଲି;

    GGML_UNUSED(reg);
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_ଗଣନା,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(ଶୂନ୍ୟ) {
    // init CPU feature detection
    ggml_cpu_init();

    ସ୍ଥିର ସଂରଚନା ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ ଖାଲି,
    };

    ଫେରନ୍ତୁ &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
