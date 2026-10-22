#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-cpu.h"
#અન્તર્ભાવ "repack.h"
#અન્તર્ભાવ "traits.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "amx/amx.h"

#અન્તર્ભાવ <cctype>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#જો defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#નહીંતર
#    include <unistd.h>
#endif

#જો defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    સ્થિર std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#જો defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        જો (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        જો (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        જો (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        જો (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        પાછા bufts;
    }();

    પાછા bufts;
}

સ્થિર ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    સ્થિર std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        પાછા bufts;
    }();

    પાછા extra_bufts.data();

    GGML_UNUSED(device);
}

સ્થિર તાર્કિક ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    માટે (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        જો (extra == buft) {
            પાછા સાચું;
        }
    }
    પાછા ખોટું;
}

// CPU backend - backend (stream)

સંરચના ggml_backend_cpu_context {
    પૂર્ણાંક                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    શૂન્ય *              abort_callback_data;

    તાર્કિક                use_ref;  // use reference implementation
};

સ્થિર અચલ અક્ષર * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    પાછા "CPU";

    GGML_UNUSED(backend);
}

સ્થિર શૂન્ય ggml_backend_cpu_free(ggml_backend_t backend) {
    સંરચના ggml_backend_cpu_context * cpu_ctx = (સંરચના ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

સંરચના ggml_backend_plan_cpu {
    સંરચના ggml_cplan cplan;
    સંરચના ggml_cgraph cgraph;
};

સ્થિર ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, અચલ સંરચના ggml_cgraph * cgraph) {
    સંરચના ggml_backend_cpu_context * cpu_ctx = (સંરચના ggml_backend_cpu_context *)backend->context;

    સંરચના ggml_backend_plan_cpu * cpu_plan = નવું ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    જો (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = નવું uint8_t[cpu_plan->cplan.work_size];
        જો (cpu_plan->cplan.work_data == ખાલી) {
            delete cpu_plan;
            પાછા ખાલી;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    પાછા cpu_plan;
}

સ્થિર શૂન્ય ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    સંરચના ggml_backend_plan_cpu * cpu_plan = (સંરચના ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

સ્થિર ગણના ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    સંરચના ggml_backend_plan_cpu * cpu_plan = (સંરચના ggml_backend_plan_cpu *)plan;

    પાછા ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

સ્થિર ગણના ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, સંરચના ggml_cgraph * cgraph) {
    સંરચના ggml_backend_cpu_context * cpu_ctx = (સંરચના ggml_backend_cpu_context *)backend->context;

    સંરચના ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    જો (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = નવું uint8_t[cplan.work_size];
        જો (cpu_ctx->work_data == ખાલી) {
            cpu_ctx->work_size = 0;
            પાછા GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    પાછા ggml_graph_compute(cgraph, &cplan);
}

સ્થિર અચલ સંરચના ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ ખાલી,
    /* .get_tensor_async        = */ ખાલી,
    /* .set_tensor_2d_async     = */ ખાલી,
    /* .get_tensor_2d_async     = */ ખાલી,
    /* .cpy_tensor_async        = */ ખાલી,
    /* .synchronize             = */ ખાલી,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ ખાલી,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ ખાલી,
    /* .event_wait              = */ ખાલી,
    /* .graph_optimize          = */ ખાલી,
};

સ્થિર ggml_guid_t ggml_backend_cpu_guid(શૂન્ય) {
    સ્થિર ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    પાછા &guid;
}

ggml_backend_t ggml_backend_cpu_init(શૂન્ય) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    સંરચના ggml_backend_cpu_context * ctx = નવું ggml_backend_cpu_context;
    જો (ctx == ખાલી) {
        પાછા ખાલી;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = ખાલી;
    ctx->work_data           = ખાલી;
    ctx->work_size           = 0;
    ctx->abort_callback      = ખાલી;
    ctx->abort_callback_data = ખાલી;
    ctx->use_ref             = ખોટું;

    ggml_backend_t cpu_backend = નવું ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    જો (cpu_backend == ખાલી) {
        delete ctx;
        પાછા ખાલી;
    }

    પાછા cpu_backend;
}

તાર્કિક ggml_backend_is_cpu(ggml_backend_t backend) {
    પાછા backend != ખાલી && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

શૂન્ય ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, પૂર્ણાંક n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    સંરચના ggml_backend_cpu_context * ctx = (સંરચના ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

શૂન્ય ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    સંરચના ggml_backend_cpu_context * ctx = (સંરચના ggml_backend_cpu_context *)backend_cpu->context;

    જો (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

શૂન્ય ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, શૂન્ય * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    સંરચના ggml_backend_cpu_context * ctx = (સંરચના ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

શૂન્ય ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, તાર્કિક use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    સંરચના ggml_backend_cpu_context * ctx = (સંરચના ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

સંરચના ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        જો (!sysctlbyname("machdep.cpu.brand_string", ખાલી, &len, ખાલી, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, ખાલી, 0); // NOLINT
        }
#અથવાજો defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        જો (f) {
            અક્ષર buf[1024];
            જ્યારે (fgets(buf, કદ(buf), f)) {
                જો (strncmp(buf, "model name", 10) == 0) {
                    અક્ષર * p = strchr(buf, ':');
                    જો (p) {
                        p++;
                        જ્યારે (std::isspace(*p)) {
                            p++;
                        }
                        જ્યારે (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        તોડો;
                    }
                }
            }
            fclose(f);
        }
#અથવાજો defined(_WIN32)
        HKEY h_ચાવી;
        જો (Reg_Open_ચાવી_Ex(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        ચાવી_READ,
                        &h_ચાવી) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            જો (Reg_Query_કિંમત_Ex_A(h_ચાવી,
                                "ProcessorNameString",
                                ખાલી,
                                ખાલી,
                                ખાલી,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                જો (Reg_Query_કિંમત_Ex_A(h_ચાવી,
                                    "ProcessorNameString",
                                    ખાલી,
                                    ખાલી,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    જો (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_ચાવી(h_ચાવી);
        }
#endif
    }
};

સ્થિર અચલ અક્ષર * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    પાછા "CPU";

    GGML_UNUSED(dev);
}

સ્થિર અચલ અક્ષર * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    સંરચના ggml_backend_cpu_device_context * ctx = (સંરચના ggml_backend_cpu_device_context *)dev->context;

    પાછા ctx->description.c_str();
}

સ્થિર શૂન્ય ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = કદ(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *મુક્ત = status.ullAvailPhys;
#નહીંતર
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *મુક્ત = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

સ્થિર ગણના ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    પાછા GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, સંરચના ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ખોટું,
        /* .host_buffer           = */ ખોટું,
        /* .buffer_from_host_ptr  = */ સાચું,
        /* .events                = */ ખોટું,
        /* .mmap_support          = */ સાચું,
    };
}

સ્થિર ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, અચલ અક્ષર * params) {
    પાછા ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    પાછા ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

સ્થિર ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, શૂન્ય * ptr, size_t size, size_t max_tensor_size) {
    પાછા ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

સ્થિર તાર્કિક ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];

    જો (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        પાછા સાચું;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    માટે (પૂર્ણાંક i = 0; i < 4; i++) {
        જો (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            પાછા buf_extra->supports_op(dev, op);
        }
    }

    પસંદગી (op->op) {
        સ્થિતિ GGML_OP_CPY:
        સ્થિતિ GGML_OP_SET_ROWS:
            પાછા
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        સ્થિતિ GGML_OP_MUL_MAT:
            જો (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                પાછા src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            પાછા src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        સ્થિતિ GGML_OP_SOFT_MAX_BACK: {
            જો (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                પાછા ખોટું;
            }
            દશાંશ max_bias = 0.0f;

            memcpy(&max_bias, (અચલ દશાંશ *) op->op_params + 1, કદ(દશાંશ));

            પાછા max_bias == 0.0f;
        }
        સ્થિતિ GGML_OP_IM2COL_BACK:
            પાછા src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        સ્થિતિ GGML_OP_GET_ROWS_BACK:
            પાછા src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        સ્થિતિ GGML_OP_OUT_PROD:
            પાછા (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        સ્થિતિ GGML_OP_CONV_2D:
            પાછા ggml_is_contiguous(op->src[0]);
        સ્થિતિ GGML_OP_SSM_SCAN:
            પાછા ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        મૂળભૂત:
            પાછા સાચું;
    }
}

સ્થિર તાર્કિક ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    પાછા ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

સ્થિર અચલ સંરચના ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ ખાલી,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ ખાલી,
    /* .event_new            = */ ખાલી,
    /* .event_free           = */ ખાલી,
    /* .event_synchronize    = */ ખાલી,
};

// CPU backend - backend (reg)

સ્થિર અચલ અક્ષર * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    પાછા "CPU";

    GGML_UNUSED(reg);
}

સ્થિર size_t ggml_backend_cpu_reg_get_device_ગણતરી(ggml_backend_reg_t reg) {
    પાછા 1;

    GGML_UNUSED(reg);
}

સ્થિર ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    સ્થિર ggml_backend_cpu_device_context ctx;
    સ્થિર ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    પાછા &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
સ્થિર ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    સ્થિર std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        જો (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        જો (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        જો (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        જો (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        જો (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        જો (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        જો (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        જો (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        જો (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        જો (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        જો (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        જો (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        જો (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        જો (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        જો (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        જો (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        જો (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        જો (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        જો (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        જો (ggml_cpu_get_sve_cnt() > 0) {
            સ્થિર std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        જો (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        જો (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        જો (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        જો (ggml_cpu_get_rvv_vlen() > 0) {
            સ્થિર std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        જો (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        જો (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        જો (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        જો (ggml_cpu_has_llamafile()) {
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

        પાછા features;
    }();

    પાછા features.data();

    GGML_UNUSED(reg);
}

સ્થિર શૂન્ય * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, અચલ અક્ષર * name) {
    જો (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        પાછા (શૂન્ય *)fct;
    }
    જો (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        પાછા (શૂન્ય *)fct;
    }
    જો (strcmp(name, "ggml_backend_get_features") == 0) {
        પાછા (શૂન્ય *)ggml_backend_cpu_get_features;
    }
    જો (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        પાછા (શૂન્ય *)ggml_backend_cpu_set_abort_callback;
    }
    જો (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        પાછા (શૂન્ય *)ggml_numa_init;
    }
    જો (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        પાછા (શૂન્ય *)ggml_is_numa;
    }
    જો (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        પાછા (શૂન્ય *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    જો (strcmp(name, "ggml_threadpool_new") == 0) {
        પાછા (શૂન્ય *)ggml_threadpool_new;
    }
    જો (strcmp(name, "ggml_threadpool_free") == 0) {
        પાછા (શૂન્ય *)ggml_threadpool_free;
    }
    જો (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        પાછા (શૂન્ય *)ggml_backend_cpu_set_threadpool;
    }

    પાછા ખાલી;

    GGML_UNUSED(reg);
}

સ્થિર અચલ સંરચના ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_ગણતરી,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(શૂન્ય) {
    // init CPU feature detection
    ggml_cpu_init();

    સ્થિર સંરચના ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ ખાલી,
    };

    પાછા &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
