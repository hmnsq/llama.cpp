#ضمن "ggml-backend.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cpu.h"
#ضمن "repack.h"
#ضمن "traits.h"
#ضمن "ggml-impl.h"
#ضمن "amx/amx.h"

#ضمن <cctype>
#ضمن <string>
#ضمن <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#ܐܢ defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#ܐܚܪܢܐ
#    include <unistd.h>
#endif

#ܐܢ defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    ساكن std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#ܐܢ defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        ܐܢ (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        ܐܢ (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        ܐܢ (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        ܐܢ (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        ܦܢܐ bufts;
    }();

    ܦܢܐ bufts;
}

ساكن ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    ساكن std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        ܦܢܐ bufts;
    }();

    ܦܢܐ extra_bufts.data();

    GGML_UNUSED(device);
}

ساكن منطقي ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    لكل (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        ܐܢ (extra == buft) {
            ܦܢܐ ܫܪܪܐ;
        }
    }
    ܦܢܐ ܕܓܠܐ;
}

// CPU backend - backend (stream)

بنية ggml_backend_cpu_context {
    صحيح                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    فراغ *              abort_callback_data;

    منطقي                use_ref;  // use reference implementation
};

ساكن ثابت حرف * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    ܦܢܐ "CPU";

    GGML_UNUSED(backend);
}

ساكن فراغ ggml_backend_cpu_free(ggml_backend_t backend) {
    بنية ggml_backend_cpu_context * cpu_ctx = (بنية ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

بنية ggml_backend_plan_cpu {
    بنية ggml_cplan cplan;
    بنية ggml_cgraph cgraph;
};

ساكن ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, ثابت بنية ggml_cgraph * cgraph) {
    بنية ggml_backend_cpu_context * cpu_ctx = (بنية ggml_backend_cpu_context *)backend->context;

    بنية ggml_backend_plan_cpu * cpu_plan = جديد ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    ܐܢ (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = جديد uint8_t[cpu_plan->cplan.work_size];
        ܐܢ (cpu_plan->cplan.work_data == عدم) {
            delete cpu_plan;
            ܦܢܐ عدم;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    ܦܢܐ cpu_plan;
}

ساكن فراغ ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    بنية ggml_backend_plan_cpu * cpu_plan = (بنية ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

ساكن تعداد ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    بنية ggml_backend_plan_cpu * cpu_plan = (بنية ggml_backend_plan_cpu *)plan;

    ܦܢܐ ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

ساكن تعداد ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    بنية ggml_backend_cpu_context * cpu_ctx = (بنية ggml_backend_cpu_context *)backend->context;

    بنية ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    ܐܢ (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = جديد uint8_t[cplan.work_size];
        ܐܢ (cpu_ctx->work_data == عدم) {
            cpu_ctx->work_size = 0;
            ܦܢܐ GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    ܦܢܐ ggml_graph_compute(cgraph, &cplan);
}

ساكن ثابت بنية ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ عدم,
    /* .get_tensor_async        = */ عدم,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ عدم,
    /* .synchronize             = */ عدم,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ عدم,
    /* .event_wait              = */ عدم,
    /* .graph_optimize          = */ عدم,
};

ساكن ggml_guid_t ggml_backend_cpu_guid(فراغ) {
    ساكن ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    ܦܢܐ &guid;
}

ggml_backend_t ggml_backend_cpu_init(فراغ) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    بنية ggml_backend_cpu_context * ctx = جديد ggml_backend_cpu_context;
    ܐܢ (ctx == عدم) {
        ܦܢܐ عدم;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = عدم;
    ctx->work_data           = عدم;
    ctx->work_size           = 0;
    ctx->abort_callback      = عدم;
    ctx->abort_callback_data = عدم;
    ctx->use_ref             = ܕܓܠܐ;

    ggml_backend_t cpu_backend = جديد ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    ܐܢ (cpu_backend == عدم) {
        delete ctx;
        ܦܢܐ عدم;
    }

    ܦܢܐ cpu_backend;
}

منطقي ggml_backend_is_cpu(ggml_backend_t backend) {
    ܦܢܐ backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

فراغ ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, صحيح n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    بنية ggml_backend_cpu_context * ctx = (بنية ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

فراغ ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    بنية ggml_backend_cpu_context * ctx = (بنية ggml_backend_cpu_context *)backend_cpu->context;

    ܐܢ (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

فراغ ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, فراغ * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    بنية ggml_backend_cpu_context * ctx = (بنية ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

فراغ ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, منطقي use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    بنية ggml_backend_cpu_context * ctx = (بنية ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

بنية ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        ܐܢ (!sysctlbyname("machdep.cpu.brand_string", عدم, &len, عدم, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, عدم, 0); // NOLINT
        }
#وإلاإذا defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        ܐܢ (f) {
            حرف buf[1024];
            ܟܡܐ (fgets(buf, حجم(buf), f)) {
                ܐܢ (strncmp(buf, "model name", 10) == 0) {
                    حرف * p = strchr(buf, ':');
                    ܐܢ (p) {
                        p++;
                        ܟܡܐ (std::isspace(*p)) {
                            p++;
                        }
                        ܟܡܐ (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        اكسر;
                    }
                }
            }
            fclose(f);
        }
#وإلاإذا defined(_WIN32)
        HKEY hKey;
        ܐܢ (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            ܐܢ (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                عدم,
                                عدم,
                                عدم,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                ܐܢ (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    عدم,
                                    عدم,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    ܐܢ (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

ساكن ثابت حرف * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    ܦܢܐ "CPU";

    GGML_UNUSED(dev);
}

ساكن ثابت حرف * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    بنية ggml_backend_cpu_device_context * ctx = (بنية ggml_backend_cpu_device_context *)dev->context;

    ܦܢܐ ctx->description.c_str();
}

ساكن فراغ ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * حرر, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = حجم(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *حرر = status.ullAvailPhys;
#ܐܚܪܢܐ
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *حرر = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    ܦܢܐ GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

ساكن فراغ ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ܕܓܠܐ,
        /* .host_buffer           = */ ܕܓܠܐ,
        /* .buffer_from_host_ptr  = */ ܫܪܪܐ,
        /* .events                = */ ܕܓܠܐ,
        /* .mmap_support          = */ ܫܪܪܐ,
    };
}

ساكن ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    ܦܢܐ ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

ساكن ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    ܦܢܐ ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

ساكن ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, فراغ * ptr, size_t size, size_t max_tensor_size) {
    ܦܢܐ ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

ساكن منطقي ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];

    ܐܢ (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        ܦܢܐ ܫܪܪܐ;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    لكل (صحيح i = 0; i < 4; i++) {
        ܐܢ (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            ܦܢܐ buf_extra->supports_op(dev, op);
        }
    }

    بدل (op->op) {
        حالة GGML_OP_CPY:
        حالة GGML_OP_SET_ROWS:
            ܦܢܐ
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        حالة GGML_OP_MUL_MAT:
            ܐܢ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                ܦܢܐ src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            ܦܢܐ src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        حالة GGML_OP_SOFT_MAX_BACK: {
            ܐܢ (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                ܦܢܐ ܕܓܠܐ;
            }
            عائم max_bias = 0.0f;

            memcpy(&max_bias, (ثابت عائم *) op->op_params + 1, حجم(عائم));

            ܦܢܐ max_bias == 0.0f;
        }
        حالة GGML_OP_IM2COL_BACK:
            ܦܢܐ src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        حالة GGML_OP_GET_ROWS_BACK:
            ܦܢܐ src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        حالة GGML_OP_OUT_PROD:
            ܦܢܐ (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        حالة GGML_OP_CONV_2D:
            ܦܢܐ ggml_is_contiguous(op->src[0]);
        حالة GGML_OP_SSM_SCAN:
            ܦܢܐ ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        افتراضي:
            ܦܢܐ ܫܪܪܐ;
    }
}

ساكن منطقي ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ܦܢܐ ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

ساكن ثابت بنية ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ عدم,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ عدم,
    /* .event_new            = */ عدم,
    /* .event_free           = */ عدم,
    /* .event_synchronize    = */ عدم,
};

// CPU backend - backend (reg)

ساكن ثابت حرف * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    ܦܢܐ "CPU";

    GGML_UNUSED(reg);
}

ساكن size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    ܦܢܐ 1;

    GGML_UNUSED(reg);
}

ساكن ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ساكن ggml_backend_cpu_device_context ctx;
    ساكن ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    ܦܢܐ &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
ساكن ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    ساكن std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        ܐܢ (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        ܐܢ (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        ܐܢ (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        ܐܢ (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        ܐܢ (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        ܐܢ (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        ܐܢ (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        ܐܢ (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        ܐܢ (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        ܐܢ (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        ܐܢ (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        ܐܢ (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        ܐܢ (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        ܐܢ (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        ܐܢ (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        ܐܢ (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        ܐܢ (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        ܐܢ (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        ܐܢ (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        ܐܢ (ggml_cpu_get_sve_cnt() > 0) {
            ساكن std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        ܐܢ (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        ܐܢ (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        ܐܢ (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        ܐܢ (ggml_cpu_get_rvv_vlen() > 0) {
            ساكن std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        ܐܢ (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        ܐܢ (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        ܐܢ (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        ܐܢ (ggml_cpu_has_llamafile()) {
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

        ܦܢܐ features;
    }();

    ܦܢܐ features.data();

    GGML_UNUSED(reg);
}

ساكن فراغ * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, ثابت حرف * name) {
    ܐܢ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        ܦܢܐ (فراغ *)fct;
    }
    ܐܢ (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        ܦܢܐ (فراغ *)fct;
    }
    ܐܢ (strcmp(name, "ggml_backend_get_features") == 0) {
        ܦܢܐ (فراغ *)ggml_backend_cpu_get_features;
    }
    ܐܢ (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        ܦܢܐ (فراغ *)ggml_backend_cpu_set_abort_callback;
    }
    ܐܢ (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        ܦܢܐ (فراغ *)ggml_numa_init;
    }
    ܐܢ (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        ܦܢܐ (فراغ *)ggml_is_numa;
    }
    ܐܢ (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        ܦܢܐ (فراغ *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    ܐܢ (strcmp(name, "ggml_threadpool_new") == 0) {
        ܦܢܐ (فراغ *)ggml_threadpool_new;
    }
    ܐܢ (strcmp(name, "ggml_threadpool_free") == 0) {
        ܦܢܐ (فراغ *)ggml_threadpool_free;
    }
    ܐܢ (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        ܦܢܐ (فراغ *)ggml_backend_cpu_set_threadpool;
    }

    ܦܢܐ عدم;

    GGML_UNUSED(reg);
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(فراغ) {
    // init CPU feature detection
    ggml_cpu_init();

    ساكن بنية ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ عدم,
    };

    ܦܢܐ &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
