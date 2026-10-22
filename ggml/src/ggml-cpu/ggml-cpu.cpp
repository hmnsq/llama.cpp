#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-cpu.h"
#অন্তর্ভাব "repack.h"
#অন্তর্ভাব "traits.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "amx/amx.h"

#অন্তর্ভাব <cctype>
#অন্তর্ভাব <string>
#অন্তর্ভাব <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#যদি defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#অন্যথা
#    include <unistd.h>
#endif

#যদি defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    স্থির std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#যদি defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        যদি (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        যদি (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        যদি (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        যদি (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        প্রত্যাগম bufts;
    }();

    প্রত্যাগম bufts;
}

স্থির ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    স্থির std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        প্রত্যাগম bufts;
    }();

    প্রত্যাগম extra_bufts.data();

    GGML_UNUSED(device);
}

স্থির তার্কিক ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    প্রতি (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        যদি (extra == buft) {
            প্রত্যাগম সত্য;
        }
    }
    প্রত্যাগম অসত্য;
}

// CPU backend - backend (stream)

সংরচনা ggml_backend_cpu_context {
    পূর্ণাঙ্ক                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    শূন্য *              abort_callback_data;

    তার্কিক                use_ref;  // use reference implementation
};

স্থির ধ্রুব বর্ণ * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    প্রত্যাগম "CPU";

    GGML_UNUSED(backend);
}

স্থির শূন্য ggml_backend_cpu_free(ggml_backend_t backend) {
    সংরচনা ggml_backend_cpu_context * cpu_ctx = (সংরচনা ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

সংরচনা ggml_backend_plan_cpu {
    সংরচনা ggml_cplan cplan;
    সংরচনা ggml_cgraph cgraph;
};

স্থির ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, ধ্রুব সংরচনা ggml_cgraph * cgraph) {
    সংরচনা ggml_backend_cpu_context * cpu_ctx = (সংরচনা ggml_backend_cpu_context *)backend->context;

    সংরচনা ggml_backend_plan_cpu * cpu_plan = নূতন ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    যদি (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = নূতন uint8_t[cpu_plan->cplan.work_size];
        যদি (cpu_plan->cplan.work_data == রিক্ত) {
            delete cpu_plan;
            প্রত্যাগম রিক্ত;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    প্রত্যাগম cpu_plan;
}

স্থির শূন্য ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    সংরচনা ggml_backend_plan_cpu * cpu_plan = (সংরচনা ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

স্থির গণনা ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    সংরচনা ggml_backend_plan_cpu * cpu_plan = (সংরচনা ggml_backend_plan_cpu *)plan;

    প্রত্যাগম ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

স্থির গণনা ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, সংরচনা ggml_cgraph * cgraph) {
    সংরচনা ggml_backend_cpu_context * cpu_ctx = (সংরচনা ggml_backend_cpu_context *)backend->context;

    সংরচনা ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    যদি (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = নূতন uint8_t[cplan.work_size];
        যদি (cpu_ctx->work_data == রিক্ত) {
            cpu_ctx->work_size = 0;
            প্রত্যাগম GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    প্রত্যাগম ggml_graph_compute(cgraph, &cplan);
}

স্থির ধ্রুব সংরচনা ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ রিক্ত,
    /* .get_tensor_async        = */ রিক্ত,
    /* .set_tensor_2d_async     = */ রিক্ত,
    /* .get_tensor_2d_async     = */ রিক্ত,
    /* .cpy_tensor_async        = */ রিক্ত,
    /* .synchronize             = */ রিক্ত,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ রিক্ত,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ রিক্ত,
    /* .event_wait              = */ রিক্ত,
    /* .graph_optimize          = */ রিক্ত,
};

স্থির ggml_guid_t ggml_backend_cpu_guid(শূন্য) {
    স্থির ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    প্রত্যাগম &guid;
}

ggml_backend_t ggml_backend_cpu_init(শূন্য) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    সংরচনা ggml_backend_cpu_context * ctx = নূতন ggml_backend_cpu_context;
    যদি (ctx == রিক্ত) {
        প্রত্যাগম রিক্ত;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = রিক্ত;
    ctx->work_data           = রিক্ত;
    ctx->work_size           = 0;
    ctx->abort_callback      = রিক্ত;
    ctx->abort_callback_data = রিক্ত;
    ctx->use_ref             = অসত্য;

    ggml_backend_t cpu_backend = নূতন ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    যদি (cpu_backend == রিক্ত) {
        delete ctx;
        প্রত্যাগম রিক্ত;
    }

    প্রত্যাগম cpu_backend;
}

তার্কিক ggml_backend_is_cpu(ggml_backend_t backend) {
    প্রত্যাগম backend != রিক্ত && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

শূন্য ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, পূর্ণাঙ্ক n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    সংরচনা ggml_backend_cpu_context * ctx = (সংরচনা ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

শূন্য ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    সংরচনা ggml_backend_cpu_context * ctx = (সংরচনা ggml_backend_cpu_context *)backend_cpu->context;

    যদি (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

শূন্য ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, শূন্য * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    সংরচনা ggml_backend_cpu_context * ctx = (সংরচনা ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

শূন্য ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, তার্কিক use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    সংরচনা ggml_backend_cpu_context * ctx = (সংরচনা ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

সংরচনা ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        যদি (!sysctlbyname("machdep.cpu.brand_string", রিক্ত, &len, রিক্ত, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, রিক্ত, 0); // NOLINT
        }
#নোচেদ্যদি defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        যদি (f) {
            বর্ণ buf[1024];
            যাবত্ (fgets(buf, পরিমাণ(buf), f)) {
                যদি (strncmp(buf, "model name", 10) == 0) {
                    বর্ণ * p = strchr(buf, ':');
                    যদি (p) {
                        p++;
                        যাবত্ (std::isspace(*p)) {
                            p++;
                        }
                        যাবত্ (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        ভঙ্গ;
                    }
                }
            }
            fclose(f);
        }
#নোচেদ্যদি defined(_WIN32)
        HKEY h_চাবি;
        যদি (Reg_Open_চাবি_Ex(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        চাবি_READ,
                        &h_চাবি) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            যদি (Reg_Query_মান_Ex_A(h_চাবি,
                                "ProcessorNameString",
                                রিক্ত,
                                রিক্ত,
                                রিক্ত,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                যদি (Reg_Query_মান_Ex_A(h_চাবি,
                                    "ProcessorNameString",
                                    রিক্ত,
                                    রিক্ত,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    যদি (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_চাবি(h_চাবি);
        }
#endif
    }
};

স্থির ধ্রুব বর্ণ * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    প্রত্যাগম "CPU";

    GGML_UNUSED(dev);
}

স্থির ধ্রুব বর্ণ * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    সংরচনা ggml_backend_cpu_device_context * ctx = (সংরচনা ggml_backend_cpu_device_context *)dev->context;

    প্রত্যাগম ctx->description.c_str();
}

স্থির শূন্য ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = পরিমাণ(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *মুক্ত = status.ullAvailPhys;
#অন্যথা
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *মুক্ত = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    প্রত্যাগম GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, সংরচনা ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ অসত্য,
        /* .host_buffer           = */ অসত্য,
        /* .buffer_from_host_ptr  = */ সত্য,
        /* .events                = */ অসত্য,
        /* .mmap_support          = */ সত্য,
    };
}

স্থির ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, ধ্রুব বর্ণ * params) {
    প্রত্যাগম ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

স্থির ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    প্রত্যাগম ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

স্থির ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, শূন্য * ptr, size_t size, size_t max_tensor_size) {
    প্রত্যাগম ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

স্থির তার্কিক ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];

    যদি (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        প্রত্যাগম সত্য;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    প্রতি (পূর্ণাঙ্ক i = 0; i < 4; i++) {
        যদি (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            প্রত্যাগম buf_extra->supports_op(dev, op);
        }
    }

    বিকল্প (op->op) {
        পক্ষ GGML_OP_CPY:
        পক্ষ GGML_OP_SET_ROWS:
            প্রত্যাগম
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        পক্ষ GGML_OP_MUL_MAT:
            যদি (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                প্রত্যাগম src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            প্রত্যাগম src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        পক্ষ GGML_OP_SOFT_MAX_BACK: {
            যদি (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                প্রত্যাগম অসত্য;
            }
            ভিন্ন max_bias = 0.0f;

            memcpy(&max_bias, (ধ্রুব ভিন্ন *) op->op_params + 1, পরিমাণ(ভিন্ন));

            প্রত্যাগম max_bias == 0.0f;
        }
        পক্ষ GGML_OP_IM2COL_BACK:
            প্রত্যাগম src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        পক্ষ GGML_OP_GET_ROWS_BACK:
            প্রত্যাগম src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        পক্ষ GGML_OP_OUT_PROD:
            প্রত্যাগম (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        পক্ষ GGML_OP_CONV_2D:
            প্রত্যাগম ggml_is_contiguous(op->src[0]);
        পক্ষ GGML_OP_SSM_SCAN:
            প্রত্যাগম ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        স্বভাব:
            প্রত্যাগম সত্য;
    }
}

স্থির তার্কিক ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    প্রত্যাগম ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

স্থির ধ্রুব সংরচনা ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ রিক্ত,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ রিক্ত,
    /* .event_new            = */ রিক্ত,
    /* .event_free           = */ রিক্ত,
    /* .event_synchronize    = */ রিক্ত,
};

// CPU backend - backend (reg)

স্থির ধ্রুব বর্ণ * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    প্রত্যাগম "CPU";

    GGML_UNUSED(reg);
}

স্থির size_t ggml_backend_cpu_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    প্রত্যাগম 1;

    GGML_UNUSED(reg);
}

স্থির ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    স্থির ggml_backend_cpu_device_context ctx;
    স্থির ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    প্রত্যাগম &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
স্থির ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    স্থির std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        যদি (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        যদি (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        যদি (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        যদি (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        যদি (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        যদি (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        যদি (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        যদি (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        যদি (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        যদি (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        যদি (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        যদি (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        যদি (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        যদি (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        যদি (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        যদি (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        যদি (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        যদি (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        যদি (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        যদি (ggml_cpu_get_sve_cnt() > 0) {
            স্থির std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        যদি (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        যদি (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        যদি (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        যদি (ggml_cpu_get_rvv_vlen() > 0) {
            স্থির std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        যদি (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        যদি (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        যদি (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        যদি (ggml_cpu_has_llamafile()) {
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

        প্রত্যাগম features;
    }();

    প্রত্যাগম features.data();

    GGML_UNUSED(reg);
}

স্থির শূন্য * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, ধ্রুব বর্ণ * name) {
    যদি (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        প্রত্যাগম (শূন্য *)fct;
    }
    যদি (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        প্রত্যাগম (শূন্য *)fct;
    }
    যদি (strcmp(name, "ggml_backend_get_features") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_cpu_get_features;
    }
    যদি (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_cpu_set_abort_callback;
    }
    যদি (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        প্রত্যাগম (শূন্য *)ggml_numa_init;
    }
    যদি (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        প্রত্যাগম (শূন্য *)ggml_is_numa;
    }
    যদি (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    যদি (strcmp(name, "ggml_threadpool_new") == 0) {
        প্রত্যাগম (শূন্য *)ggml_threadpool_new;
    }
    যদি (strcmp(name, "ggml_threadpool_free") == 0) {
        প্রত্যাগম (শূন্য *)ggml_threadpool_free;
    }
    যদি (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        প্রত্যাগম (শূন্য *)ggml_backend_cpu_set_threadpool;
    }

    প্রত্যাগম রিক্ত;

    GGML_UNUSED(reg);
}

স্থির ধ্রুব সংরচনা ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(শূন্য) {
    // init CPU feature detection
    ggml_cpu_init();

    স্থির সংরচনা ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ রিক্ত,
    };

    প্রত্যাগম &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
