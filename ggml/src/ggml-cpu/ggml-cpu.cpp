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

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#ಯದಿ defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#ಅನ್ಯಥಾ
#    include <unistd.h>
#endif

#ಯದಿ defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    ಸ್ಥಿರ std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#ಯದಿ defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        ಯದಿ (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        ಯದಿ (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        ಯದಿ (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        ಯದಿ (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        ಪ್ರತ್ಯಾಗಮ bufts;
    }();

    ಪ್ರತ್ಯಾಗಮ bufts;
}

ಸ್ಥಿರ ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    ಸ್ಥಿರ std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        ಪ್ರತ್ಯಾಗಮ bufts;
    }();

    ಪ್ರತ್ಯಾಗಮ extra_bufts.data();

    GGML_UNUSED(device);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    ಪ್ರತಿ (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        ಯದಿ (extra == buft) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

// CPU backend - backend (stream)

ಸಂರಚನಾ ggml_backend_cpu_context {
    ಪೂರ್ಣಾಙ್ಕ                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    ಶೂನ್ಯ *              abort_callback_data;

    ತಾರ್ಕಿಕ                use_ref;  // use reference implementation
};

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ "CPU";

    GGML_UNUSED(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_free(ggml_backend_t backend) {
    ಸಂರಚನಾ ggml_backend_cpu_context * cpu_ctx = (ಸಂರಚನಾ ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

ಸಂರಚನಾ ggml_backend_plan_cpu {
    ಸಂರಚನಾ ggml_cplan cplan;
    ಸಂರಚನಾ ggml_cgraph cgraph;
};

ಸ್ಥಿರ ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph) {
    ಸಂರಚನಾ ggml_backend_cpu_context * cpu_ctx = (ಸಂರಚನಾ ggml_backend_cpu_context *)backend->context;

    ಸಂರಚನಾ ggml_backend_plan_cpu * cpu_plan = ನೂತನ ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    ಯದಿ (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = ನೂತನ uint8_t[cpu_plan->cplan.work_size];
        ಯದಿ (cpu_plan->cplan.work_data == ರಿಕ್ತ) {
            delete cpu_plan;
            ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    ಪ್ರತ್ಯಾಗಮ cpu_plan;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ಸಂರಚನಾ ggml_backend_plan_cpu * cpu_plan = (ಸಂರಚನಾ ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ಸಂರಚನಾ ggml_backend_plan_cpu * cpu_plan = (ಸಂರಚನಾ ggml_backend_plan_cpu *)plan;

    ಪ್ರತ್ಯಾಗಮ ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, ಸಂರಚನಾ ggml_cgraph * cgraph) {
    ಸಂರಚನಾ ggml_backend_cpu_context * cpu_ctx = (ಸಂರಚನಾ ggml_backend_cpu_context *)backend->context;

    ಸಂರಚನಾ ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    ಯದಿ (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = ನೂತನ uint8_t[cplan.work_size];
        ಯದಿ (cpu_ctx->work_data == ರಿಕ್ತ) {
            cpu_ctx->work_size = 0;
            ಪ್ರತ್ಯಾಗಮ GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    ಪ್ರತ್ಯಾಗಮ ggml_graph_compute(cgraph, &cplan);
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ ರಿಕ್ತ,
    /* .get_tensor_async        = */ ರಿಕ್ತ,
    /* .set_tensor_2d_async     = */ ರಿಕ್ತ,
    /* .get_tensor_2d_async     = */ ರಿಕ್ತ,
    /* .cpy_tensor_async        = */ ರಿಕ್ತ,
    /* .synchronize             = */ ರಿಕ್ತ,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ ರಿಕ್ತ,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ ರಿಕ್ತ,
    /* .event_wait              = */ ರಿಕ್ತ,
    /* .graph_optimize          = */ ರಿಕ್ತ,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_cpu_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    ಪ್ರತ್ಯಾಗಮ &guid;
}

ggml_backend_t ggml_backend_cpu_init(ಶೂನ್ಯ) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    ಸಂರಚನಾ ggml_backend_cpu_context * ctx = ನೂತನ ggml_backend_cpu_context;
    ಯದಿ (ctx == ರಿಕ್ತ) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = ರಿಕ್ತ;
    ctx->work_data           = ರಿಕ್ತ;
    ctx->work_size           = 0;
    ctx->abort_callback      = ರಿಕ್ತ;
    ctx->abort_callback_data = ರಿಕ್ತ;
    ctx->use_ref             = ಅಸತ್ಯ;

    ggml_backend_t cpu_backend = ನೂತನ ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    ಯದಿ (cpu_backend == ರಿಕ್ತ) {
        delete ctx;
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತ್ಯಾಗಮ cpu_backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_cpu(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ backend != ರಿಕ್ತ && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

ಶೂನ್ಯ ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, ಪೂರ್ಣಾಙ್ಕ n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ಸಂರಚನಾ ggml_backend_cpu_context * ctx = (ಸಂರಚನಾ ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

ಶೂನ್ಯ ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ಸಂರಚನಾ ggml_backend_cpu_context * ctx = (ಸಂರಚನಾ ggml_backend_cpu_context *)backend_cpu->context;

    ಯದಿ (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

ಶೂನ್ಯ ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, ಶೂನ್ಯ * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ಸಂರಚನಾ ggml_backend_cpu_context * ctx = (ಸಂರಚನಾ ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

ಶೂನ್ಯ ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, ತಾರ್ಕಿಕ use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ಸಂರಚನಾ ggml_backend_cpu_context * ctx = (ಸಂರಚನಾ ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

ಸಂರಚನಾ ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        ಯದಿ (!sysctlbyname("machdep.cpu.brand_string", ರಿಕ್ತ, &len, ರಿಕ್ತ, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, ರಿಕ್ತ, 0); // NOLINT
        }
#ನೋಚೇದ್ಯದಿ defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        ಯದಿ (f) {
            ವರ್ಣ buf[1024];
            ಯಾವತ್ (fgets(buf, ಪರಿಮಾಣ(buf), f)) {
                ಯದಿ (strncmp(buf, "model name", 10) == 0) {
                    ವರ್ಣ * p = strchr(buf, ':');
                    ಯದಿ (p) {
                        p++;
                        ಯಾವತ್ (std::isspace(*p)) {
                            p++;
                        }
                        ಯಾವತ್ (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        ಭಙ್ಗ;
                    }
                }
            }
            fclose(f);
        }
#ನೋಚೇದ್ಯದಿ defined(_WIN32)
        HKEY h_ಕೀಲಿ;
        ಯದಿ (Reg_Open_ಕೀಲಿ_Ex(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        ಕೀಲಿ_READ,
                        &h_ಕೀಲಿ) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            ಯದಿ (Reg_Query_ಮೌಲ್ಯ_Ex_A(h_ಕೀಲಿ,
                                "ProcessorNameString",
                                ರಿಕ್ತ,
                                ರಿಕ್ತ,
                                ರಿಕ್ತ,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                ಯದಿ (Reg_Query_ಮೌಲ್ಯ_Ex_A(h_ಕೀಲಿ,
                                    "ProcessorNameString",
                                    ರಿಕ್ತ,
                                    ರಿಕ್ತ,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    ಯದಿ (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_ಕೀಲಿ(h_ಕೀಲಿ);
        }
#endif
    }
};

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ "CPU";

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    ಸಂರಚನಾ ggml_backend_cpu_device_context * ctx = (ಸಂರಚನಾ ggml_backend_cpu_device_context *)dev->context;

    ಪ್ರತ್ಯಾಗಮ ctx->description.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * ಮುಕ್ತ, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = ಪರಿಮಾಣ(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *ಮುಕ್ತ = status.ullAvailPhys;
#ಅನ್ಯಥಾ
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *ಮುಕ್ತ = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಗಣನಾ ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, ಸಂರಚನಾ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ಅಸತ್ಯ,
        /* .host_buffer           = */ ಅಸತ್ಯ,
        /* .buffer_from_host_ptr  = */ ಸತ್ಯ,
        /* .events                = */ ಅಸತ್ಯ,
        /* .mmap_support          = */ ಸತ್ಯ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, ಧ್ರುವ ವರ್ಣ * params) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_size) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];

    ಯದಿ (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; i++) {
        ಯದಿ (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            ಪ್ರತ್ಯಾಗಮ buf_extra->supports_op(dev, op);
        }
    }

    ವಿಕಲ್ಪ (op->op) {
        ಪಕ್ಷ GGML_OP_CPY:
        ಪಕ್ಷ GGML_OP_SET_ROWS:
            ಪ್ರತ್ಯಾಗಮ
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        ಪಕ್ಷ GGML_OP_MUL_MAT:
            ಯದಿ (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                ಪ್ರತ್ಯಾಗಮ src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            ಪ್ರತ್ಯಾಗಮ src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        ಪಕ್ಷ GGML_OP_SOFT_MAX_BACK: {
            ಯದಿ (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಭಿನ್ನ max_bias = 0.0f;

            memcpy(&max_bias, (ಧ್ರುವ ಭಿನ್ನ *) op->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));

            ಪ್ರತ್ಯಾಗಮ max_bias == 0.0f;
        }
        ಪಕ್ಷ GGML_OP_IM2COL_BACK:
            ಪ್ರತ್ಯಾಗಮ src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        ಪಕ್ಷ GGML_OP_GET_ROWS_BACK:
            ಪ್ರತ್ಯಾಗಮ src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        ಪಕ್ಷ GGML_OP_OUT_PROD:
            ಪ್ರತ್ಯಾಗಮ (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_CONV_2D:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]);
        ಪಕ್ಷ GGML_OP_SSM_SCAN:
            ಪ್ರತ್ಯಾಗಮ ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ ರಿಕ್ತ,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ ರಿಕ್ತ,
    /* .event_new            = */ ರಿಕ್ತ,
    /* .event_free           = */ ರಿಕ್ತ,
    /* .event_synchronize    = */ ರಿಕ್ತ,
};

// CPU backend - backend (reg)

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    ಪ್ರತ್ಯಾಗಮ "CPU";

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ size_t ggml_backend_cpu_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಪ್ರತ್ಯಾಗಮ 1;

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ಸ್ಥಿರ ggml_backend_cpu_device_context ctx;
    ಸ್ಥಿರ ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
ಸ್ಥಿರ ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    ಸ್ಥಿರ std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        ಯದಿ (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        ಯದಿ (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        ಯದಿ (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        ಯದಿ (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        ಯದಿ (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        ಯದಿ (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        ಯದಿ (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        ಯದಿ (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        ಯದಿ (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        ಯದಿ (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        ಯದಿ (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        ಯದಿ (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        ಯದಿ (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        ಯದಿ (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        ಯದಿ (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        ಯದಿ (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        ಯದಿ (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        ಯದಿ (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        ಯದಿ (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        ಯದಿ (ggml_cpu_get_sve_cnt() > 0) {
            ಸ್ಥಿರ std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        ಯದಿ (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        ಯದಿ (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        ಯದಿ (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        ಯದಿ (ggml_cpu_get_rvv_vlen() > 0) {
            ಸ್ಥಿರ std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        ಯದಿ (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        ಯದಿ (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        ಯದಿ (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        ಯದಿ (ggml_cpu_has_llamafile()) {
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

        ಪ್ರತ್ಯಾಗಮ features;
    }();

    ಪ್ರತ್ಯಾಗಮ features.data();

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, ಧ್ರುವ ವರ್ಣ * name) {
    ಯದಿ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)fct;
    }
    ಯದಿ (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)fct;
    }
    ಯದಿ (strcmp(name, "ggml_backend_get_features") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_cpu_get_features;
    }
    ಯದಿ (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_cpu_set_abort_callback;
    }
    ಯದಿ (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_numa_init;
    }
    ಯದಿ (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_is_numa;
    }
    ಯದಿ (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    ಯದಿ (strcmp(name, "ggml_threadpool_new") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_threadpool_new;
    }
    ಯದಿ (strcmp(name, "ggml_threadpool_free") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_threadpool_free;
    }
    ಯದಿ (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_cpu_set_threadpool;
    }

    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(ಶೂನ್ಯ) {
    // init CPU feature detection
    ggml_cpu_init();

    ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ ರಿಕ್ತ,
    };

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
