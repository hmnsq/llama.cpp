#شامل "ggml-backend.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-cpu.h"
#شامل "repack.h"
#شامل "traits.h"
#شامل "ggml-impl.h"
#شامل "amx/amx.h"

#شامل <cctype>
#شامل <string>
#شامل <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#اگر defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#وگرنه
#    include <unistd.h>
#endif

#اگر defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    ایستا std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#اگر defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        اگر (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        اگر (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        اگر (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        اگر (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        بازگشت bufts;
    }();

    بازگشت bufts;
}

ایستا ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    ایستا std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        بازگشت bufts;
    }();

    بازگشت extra_bufts.data();

    GGML_UNUSED(device);
}

ایستا منطقی ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    برای (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        اگر (extra == buft) {
            بازگشت درست;
        }
    }
    بازگشت نادرست;
}

// CPU backend - backend (stream)

ساختار ggml_backend_cpu_context {
    صحیح                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    تهی *              abort_callback_data;

    منطقی                use_ref;  // use reference implementation
};

ایستا ثابت نویسه * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    بازگشت "CPU";

    GGML_UNUSED(backend);
}

ایستا تهی ggml_backend_cpu_free(ggml_backend_t backend) {
    ساختار ggml_backend_cpu_context * cpu_ctx = (ساختار ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

ساختار ggml_backend_plan_cpu {
    ساختار ggml_cplan cplan;
    ساختار ggml_cgraph cgraph;
};

ایستا ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, ثابت ساختار ggml_cgraph * cgraph) {
    ساختار ggml_backend_cpu_context * cpu_ctx = (ساختار ggml_backend_cpu_context *)backend->context;

    ساختار ggml_backend_plan_cpu * cpu_plan = جدید ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    اگر (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = جدید uint8_t[cpu_plan->cplan.work_size];
        اگر (cpu_plan->cplan.work_data == پوچ) {
            delete cpu_plan;
            بازگشت پوچ;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    بازگشت cpu_plan;
}

ایستا تهی ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ساختار ggml_backend_plan_cpu * cpu_plan = (ساختار ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

ایستا شمارش ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ساختار ggml_backend_plan_cpu * cpu_plan = (ساختار ggml_backend_plan_cpu *)plan;

    بازگشت ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

ایستا شمارش ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, ساختار ggml_cgraph * cgraph) {
    ساختار ggml_backend_cpu_context * cpu_ctx = (ساختار ggml_backend_cpu_context *)backend->context;

    ساختار ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    اگر (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = جدید uint8_t[cplan.work_size];
        اگر (cpu_ctx->work_data == پوچ) {
            cpu_ctx->work_size = 0;
            بازگشت GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    بازگشت ggml_graph_compute(cgraph, &cplan);
}

ایستا ثابت ساختار ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ پوچ,
    /* .get_tensor_async        = */ پوچ,
    /* .set_tensor_2d_async     = */ پوچ,
    /* .get_tensor_2d_async     = */ پوچ,
    /* .cpy_tensor_async        = */ پوچ,
    /* .synchronize             = */ پوچ,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ پوچ,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ پوچ,
    /* .event_wait              = */ پوچ,
    /* .graph_optimize          = */ پوچ,
};

ایستا ggml_guid_t ggml_backend_cpu_guid(تهی) {
    ایستا ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    بازگشت &guid;
}

ggml_backend_t ggml_backend_cpu_init(تهی) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    ساختار ggml_backend_cpu_context * ctx = جدید ggml_backend_cpu_context;
    اگر (ctx == پوچ) {
        بازگشت پوچ;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = پوچ;
    ctx->work_data           = پوچ;
    ctx->work_size           = 0;
    ctx->abort_callback      = پوچ;
    ctx->abort_callback_data = پوچ;
    ctx->use_ref             = نادرست;

    ggml_backend_t cpu_backend = جدید ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    اگر (cpu_backend == پوچ) {
        delete ctx;
        بازگشت پوچ;
    }

    بازگشت cpu_backend;
}

منطقی ggml_backend_is_cpu(ggml_backend_t backend) {
    بازگشت backend != پوچ && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

تهی ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, صحیح n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ساختار ggml_backend_cpu_context * ctx = (ساختار ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

تهی ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ساختار ggml_backend_cpu_context * ctx = (ساختار ggml_backend_cpu_context *)backend_cpu->context;

    اگر (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

تهی ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, تهی * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ساختار ggml_backend_cpu_context * ctx = (ساختار ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

تهی ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, منطقی use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ساختار ggml_backend_cpu_context * ctx = (ساختار ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

ساختار ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        اگر (!sysctlbyname("machdep.cpu.brand_string", پوچ, &len, پوچ, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, پوچ, 0); // NOLINT
        }
#وگرنهاگر defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        اگر (f) {
            نویسه buf[1024];
            وقتی (fgets(buf, اندازه(buf), f)) {
                اگر (strncmp(buf, "model name", 10) == 0) {
                    نویسه * p = strchr(buf, ':');
                    اگر (p) {
                        p++;
                        وقتی (std::isspace(*p)) {
                            p++;
                        }
                        وقتی (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        بشکن;
                    }
                }
            }
            fclose(f);
        }
#وگرنهاگر defined(_WIN32)
        HKEY h_کلید;
        اگر (Reg_Open_کلید_Ex(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        کلید_READ,
                        &h_کلید) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            اگر (Reg_Query_مقدار_Ex_A(h_کلید,
                                "ProcessorNameString",
                                پوچ,
                                پوچ,
                                پوچ,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                اگر (Reg_Query_مقدار_Ex_A(h_کلید,
                                    "ProcessorNameString",
                                    پوچ,
                                    پوچ,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    اگر (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_کلید(h_کلید);
        }
#endif
    }
};

ایستا ثابت نویسه * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    بازگشت "CPU";

    GGML_UNUSED(dev);
}

ایستا ثابت نویسه * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    ساختار ggml_backend_cpu_device_context * ctx = (ساختار ggml_backend_cpu_device_context *)dev->context;

    بازگشت ctx->description.c_str();
}

ایستا تهی ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * آزاد, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = اندازه(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *آزاد = status.ullAvailPhys;
#وگرنه
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *آزاد = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

ایستا شمارش ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    بازگشت GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

ایستا تهی ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, ساختار ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ نادرست,
        /* .host_buffer           = */ نادرست,
        /* .buffer_from_host_ptr  = */ درست,
        /* .events                = */ نادرست,
        /* .mmap_support          = */ درست,
    };
}

ایستا ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, ثابت نویسه * params) {
    بازگشت ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

ایستا ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    بازگشت ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

ایستا ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, تهی * ptr, size_t size, size_t max_tensor_size) {
    بازگشت ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

ایستا منطقی ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];

    اگر (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        بازگشت درست;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    برای (صحیح i = 0; i < 4; i++) {
        اگر (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            بازگشت buf_extra->supports_op(dev, op);
        }
    }

    انتخاب (op->op) {
        حالت GGML_OP_CPY:
        حالت GGML_OP_SET_ROWS:
            بازگشت
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        حالت GGML_OP_MUL_MAT:
            اگر (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                بازگشت src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            بازگشت src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        حالت GGML_OP_SOFT_MAX_BACK: {
            اگر (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                بازگشت نادرست;
            }
            اعشاری max_bias = 0.0f;

            memcpy(&max_bias, (ثابت اعشاری *) op->op_params + 1, اندازه(اعشاری));

            بازگشت max_bias == 0.0f;
        }
        حالت GGML_OP_IM2COL_BACK:
            بازگشت src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        حالت GGML_OP_GET_ROWS_BACK:
            بازگشت src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        حالت GGML_OP_OUT_PROD:
            بازگشت (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        حالت GGML_OP_CONV_2D:
            بازگشت ggml_is_contiguous(op->src[0]);
        حالت GGML_OP_SSM_SCAN:
            بازگشت ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        پیشفرض:
            بازگشت درست;
    }
}

ایستا منطقی ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    بازگشت ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

ایستا ثابت ساختار ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ پوچ,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ پوچ,
    /* .event_new            = */ پوچ,
    /* .event_free           = */ پوچ,
    /* .event_synchronize    = */ پوچ,
};

// CPU backend - backend (reg)

ایستا ثابت نویسه * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    بازگشت "CPU";

    GGML_UNUSED(reg);
}

ایستا size_t ggml_backend_cpu_reg_get_device_شمار(ggml_backend_reg_t reg) {
    بازگشت 1;

    GGML_UNUSED(reg);
}

ایستا ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ایستا ggml_backend_cpu_device_context ctx;
    ایستا ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    بازگشت &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
ایستا ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    ایستا std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        اگر (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        اگر (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        اگر (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        اگر (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        اگر (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        اگر (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        اگر (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        اگر (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        اگر (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        اگر (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        اگر (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        اگر (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        اگر (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        اگر (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        اگر (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        اگر (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        اگر (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        اگر (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        اگر (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        اگر (ggml_cpu_get_sve_cnt() > 0) {
            ایستا std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        اگر (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        اگر (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        اگر (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        اگر (ggml_cpu_get_rvv_vlen() > 0) {
            ایستا std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        اگر (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        اگر (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        اگر (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        اگر (ggml_cpu_has_llamafile()) {
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

        بازگشت features;
    }();

    بازگشت features.data();

    GGML_UNUSED(reg);
}

ایستا تهی * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, ثابت نویسه * name) {
    اگر (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        بازگشت (تهی *)fct;
    }
    اگر (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        بازگشت (تهی *)fct;
    }
    اگر (strcmp(name, "ggml_backend_get_features") == 0) {
        بازگشت (تهی *)ggml_backend_cpu_get_features;
    }
    اگر (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        بازگشت (تهی *)ggml_backend_cpu_set_abort_callback;
    }
    اگر (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        بازگشت (تهی *)ggml_numa_init;
    }
    اگر (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        بازگشت (تهی *)ggml_is_numa;
    }
    اگر (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        بازگشت (تهی *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    اگر (strcmp(name, "ggml_threadpool_new") == 0) {
        بازگشت (تهی *)ggml_threadpool_new;
    }
    اگر (strcmp(name, "ggml_threadpool_free") == 0) {
        بازگشت (تهی *)ggml_threadpool_free;
    }
    اگر (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        بازگشت (تهی *)ggml_backend_cpu_set_threadpool;
    }

    بازگشت پوچ;

    GGML_UNUSED(reg);
}

ایستا ثابت ساختار ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_شمار,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(تهی) {
    // init CPU feature detection
    ggml_cpu_init();

    ایستا ساختار ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ پوچ,
    };

    بازگشت &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
