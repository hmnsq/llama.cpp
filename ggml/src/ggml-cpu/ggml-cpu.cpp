#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-cpu.h"
#அந்தர்பாவ "repack.h"
#அந்தர்பாவ "traits.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "amx/amx.h"

#அந்தர்பாவ <cctype>
#அந்தர்பாவ <string>
#அந்தர்பாவ <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#எனில் defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#இல்லையெனில்
#    include <unistd.h>
#endif

#எனில் defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    நிலையான std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#எனில் defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        எனில் (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        எனில் (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        எனில் (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        எனில் (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        திரும்பு bufts;
    }();

    திரும்பு bufts;
}

நிலையான ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    நிலையான std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        திரும்பு bufts;
    }();

    திரும்பு extra_bufts.data();

    GGML_UNUSED(device);
}

நிலையான தருக்க ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    க்கு (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        எனில் (extra == buft) {
            திரும்பு உண்மை;
        }
    }
    திரும்பு பொய்;
}

// CPU backend - backend (stream)

கட்டமைப்பு ggml_backend_cpu_context {
    முழு                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    வெறுமை *              abort_callback_data;

    தருக்க                use_ref;  // use reference implementation
};

நிலையான மாறிலி எழுத்து * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    திரும்பு "CPU";

    GGML_UNUSED(backend);
}

நிலையான வெறுமை ggml_backend_cpu_free(ggml_backend_t backend) {
    கட்டமைப்பு ggml_backend_cpu_context * cpu_ctx = (கட்டமைப்பு ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

கட்டமைப்பு ggml_backend_plan_cpu {
    கட்டமைப்பு ggml_cplan cplan;
    கட்டமைப்பு ggml_cgraph cgraph;
};

நிலையான ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, மாறிலி கட்டமைப்பு ggml_cgraph * cgraph) {
    கட்டமைப்பு ggml_backend_cpu_context * cpu_ctx = (கட்டமைப்பு ggml_backend_cpu_context *)backend->context;

    கட்டமைப்பு ggml_backend_plan_cpu * cpu_plan = புதிய ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    எனில் (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = புதிய uint8_t[cpu_plan->cplan.work_size];
        எனில் (cpu_plan->cplan.work_data == வெற்று) {
            delete cpu_plan;
            திரும்பு வெற்று;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    திரும்பு cpu_plan;
}

நிலையான வெறுமை ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    கட்டமைப்பு ggml_backend_plan_cpu * cpu_plan = (கட்டமைப்பு ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    கட்டமைப்பு ggml_backend_plan_cpu * cpu_plan = (கட்டமைப்பு ggml_backend_plan_cpu *)plan;

    திரும்பு ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, கட்டமைப்பு ggml_cgraph * cgraph) {
    கட்டமைப்பு ggml_backend_cpu_context * cpu_ctx = (கட்டமைப்பு ggml_backend_cpu_context *)backend->context;

    கட்டமைப்பு ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    எனில் (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = புதிய uint8_t[cplan.work_size];
        எனில் (cpu_ctx->work_data == வெற்று) {
            cpu_ctx->work_size = 0;
            திரும்பு GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    திரும்பு ggml_graph_compute(cgraph, &cplan);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ வெற்று,
    /* .get_tensor_async        = */ வெற்று,
    /* .set_tensor_2d_async     = */ வெற்று,
    /* .get_tensor_2d_async     = */ வெற்று,
    /* .cpy_tensor_async        = */ வெற்று,
    /* .synchronize             = */ வெற்று,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ வெற்று,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ வெற்று,
    /* .event_wait              = */ வெற்று,
    /* .graph_optimize          = */ வெற்று,
};

நிலையான ggml_guid_t ggml_backend_cpu_guid(வெறுமை) {
    நிலையான ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    திரும்பு &guid;
}

ggml_backend_t ggml_backend_cpu_init(வெறுமை) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    கட்டமைப்பு ggml_backend_cpu_context * ctx = புதிய ggml_backend_cpu_context;
    எனில் (ctx == வெற்று) {
        திரும்பு வெற்று;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = வெற்று;
    ctx->work_data           = வெற்று;
    ctx->work_size           = 0;
    ctx->abort_callback      = வெற்று;
    ctx->abort_callback_data = வெற்று;
    ctx->use_ref             = பொய்;

    ggml_backend_t cpu_backend = புதிய ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    எனில் (cpu_backend == வெற்று) {
        delete ctx;
        திரும்பு வெற்று;
    }

    திரும்பு cpu_backend;
}

தருக்க ggml_backend_is_cpu(ggml_backend_t backend) {
    திரும்பு backend != வெற்று && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

வெறுமை ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, முழு n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    கட்டமைப்பு ggml_backend_cpu_context * ctx = (கட்டமைப்பு ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

வெறுமை ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    கட்டமைப்பு ggml_backend_cpu_context * ctx = (கட்டமைப்பு ggml_backend_cpu_context *)backend_cpu->context;

    எனில் (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

வெறுமை ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, வெறுமை * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    கட்டமைப்பு ggml_backend_cpu_context * ctx = (கட்டமைப்பு ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

வெறுமை ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, தருக்க use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    கட்டமைப்பு ggml_backend_cpu_context * ctx = (கட்டமைப்பு ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

கட்டமைப்பு ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        எனில் (!sysctlbyname("machdep.cpu.brand_string", வெற்று, &len, வெற்று, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, வெற்று, 0); // NOLINT
        }
#அல்லதெனில் defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        எனில் (f) {
            எழுத்து buf[1024];
            போது (fgets(buf, அளவு(buf), f)) {
                எனில் (strncmp(buf, "model name", 10) == 0) {
                    எழுத்து * p = strchr(buf, ':');
                    எனில் (p) {
                        p++;
                        போது (std::isspace(*p)) {
                            p++;
                        }
                        போது (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        நிறுத்து;
                    }
                }
            }
            fclose(f);
        }
#அல்லதெனில் defined(_WIN32)
        HKEY h_சாவி;
        எனில் (Reg_Open_சாவி_Ex(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        சாவி_READ,
                        &h_சாவி) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            எனில் (Reg_Query_மதிப்பு_Ex_A(h_சாவி,
                                "ProcessorNameString",
                                வெற்று,
                                வெற்று,
                                வெற்று,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                எனில் (Reg_Query_மதிப்பு_Ex_A(h_சாவி,
                                    "ProcessorNameString",
                                    வெற்று,
                                    வெற்று,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    எனில் (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_சாவி(h_சாவி);
        }
#endif
    }
};

நிலையான மாறிலி எழுத்து * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    திரும்பு "CPU";

    GGML_UNUSED(dev);
}

நிலையான மாறிலி எழுத்து * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    கட்டமைப்பு ggml_backend_cpu_device_context * ctx = (கட்டமைப்பு ggml_backend_cpu_device_context *)dev->context;

    திரும்பு ctx->description.c_str();
}

நிலையான வெறுமை ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * விடுவி, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = அளவு(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *விடுவி = status.ullAvailPhys;
#இல்லையெனில்
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *விடுவி = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

நிலையான எண்ணிக்கை ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    திரும்பு GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, கட்டமைப்பு ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ பொய்,
        /* .host_buffer           = */ பொய்,
        /* .buffer_from_host_ptr  = */ உண்மை,
        /* .events                = */ பொய்,
        /* .mmap_support          = */ உண்மை,
    };
}

நிலையான ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, மாறிலி எழுத்து * params) {
    திரும்பு ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

நிலையான ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    திரும்பு ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

நிலையான ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, வெறுமை * ptr, size_t size, size_t max_tensor_size) {
    திரும்பு ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

நிலையான தருக்க ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];

    எனில் (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        திரும்பு உண்மை;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    க்கு (முழு i = 0; i < 4; i++) {
        எனில் (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            திரும்பு buf_extra->supports_op(dev, op);
        }
    }

    தேர்வு (op->op) {
        நிலை GGML_OP_CPY:
        நிலை GGML_OP_SET_ROWS:
            திரும்பு
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        நிலை GGML_OP_MUL_MAT:
            எனில் (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                திரும்பு src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            திரும்பு src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        நிலை GGML_OP_SOFT_MAX_BACK: {
            எனில் (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                திரும்பு பொய்;
            }
            தசம max_bias = 0.0f;

            memcpy(&max_bias, (மாறிலி தசம *) op->op_params + 1, அளவு(தசம));

            திரும்பு max_bias == 0.0f;
        }
        நிலை GGML_OP_IM2COL_BACK:
            திரும்பு src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        நிலை GGML_OP_GET_ROWS_BACK:
            திரும்பு src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        நிலை GGML_OP_OUT_PROD:
            திரும்பு (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        நிலை GGML_OP_CONV_2D:
            திரும்பு ggml_is_contiguous(op->src[0]);
        நிலை GGML_OP_SSM_SCAN:
            திரும்பு ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        இயல்பு:
            திரும்பு உண்மை;
    }
}

நிலையான தருக்க ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    திரும்பு ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ வெற்று,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ வெற்று,
    /* .event_new            = */ வெற்று,
    /* .event_free           = */ வெற்று,
    /* .event_synchronize    = */ வெற்று,
};

// CPU backend - backend (reg)

நிலையான மாறிலி எழுத்து * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    திரும்பு "CPU";

    GGML_UNUSED(reg);
}

நிலையான size_t ggml_backend_cpu_reg_get_device_மொத்தம்(ggml_backend_reg_t reg) {
    திரும்பு 1;

    GGML_UNUSED(reg);
}

நிலையான ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    நிலையான ggml_backend_cpu_device_context ctx;
    நிலையான ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    திரும்பு &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
நிலையான ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    நிலையான std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        எனில் (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        எனில் (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        எனில் (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        எனில் (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        எனில் (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        எனில் (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        எனில் (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        எனில் (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        எனில் (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        எனில் (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        எனில் (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        எனில் (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        எனில் (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        எனில் (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        எனில் (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        எனில் (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        எனில் (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        எனில் (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        எனில் (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        எனில் (ggml_cpu_get_sve_cnt() > 0) {
            நிலையான std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        எனில் (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        எனில் (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        எனில் (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        எனில் (ggml_cpu_get_rvv_vlen() > 0) {
            நிலையான std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        எனில் (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        எனில் (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        எனில் (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        எனில் (ggml_cpu_has_llamafile()) {
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

        திரும்பு features;
    }();

    திரும்பு features.data();

    GGML_UNUSED(reg);
}

நிலையான வெறுமை * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, மாறிலி எழுத்து * name) {
    எனில் (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        திரும்பு (வெறுமை *)fct;
    }
    எனில் (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        திரும்பு (வெறுமை *)fct;
    }
    எனில் (strcmp(name, "ggml_backend_get_features") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_cpu_get_features;
    }
    எனில் (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_cpu_set_abort_callback;
    }
    எனில் (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        திரும்பு (வெறுமை *)ggml_numa_init;
    }
    எனில் (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        திரும்பு (வெறுமை *)ggml_is_numa;
    }
    எனில் (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    எனில் (strcmp(name, "ggml_threadpool_new") == 0) {
        திரும்பு (வெறுமை *)ggml_threadpool_new;
    }
    எனில் (strcmp(name, "ggml_threadpool_free") == 0) {
        திரும்பு (வெறுமை *)ggml_threadpool_free;
    }
    எனில் (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_cpu_set_threadpool;
    }

    திரும்பு வெற்று;

    GGML_UNUSED(reg);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_மொத்தம்,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(வெறுமை) {
    // init CPU feature detection
    ggml_cpu_init();

    நிலையான கட்டமைப்பு ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ வெற்று,
    };

    திரும்பு &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
