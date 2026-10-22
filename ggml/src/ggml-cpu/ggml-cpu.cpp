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

#جيڪڏهن_وصف_ٿيل GGML_USE_CPU_HBM
#    include "hbm.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#جيڪڏهن_پورو

#إذا defined(_WIN32)
#    define WIN32_LEAN_AND_سراسري
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#وإلا
#    include <unistd.h>
#جيڪڏهن_پورو

#إذا defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#جيڪڏهن_پورو

// ggml-backend interface

std::vector<ggml_backend_بفر_type_t> & ggml_backend_cpu_get_extra_بفر_types() {
    ساكن std::vector<ggml_backend_بفر_type_t> bufts = []() {
        std::vector<ggml_backend_بفر_type_t> bufts;

#إذا defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        إذا (ggml_backend_amx_بفر_type()) {
            bufts.push_back(ggml_backend_amx_بفر_type());
        }
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_CPU_RISCV64_SPACEMIT
        إذا (ggml_backend_cpu_riscv64_spacemit_بفر_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_بفر_type());
        }
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_CPU_KLEIDIAI
        إذا (ggml_backend_cpu_kleidiai_بفر_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_بفر_type());
        }
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_CPU_REPACK
        إذا (ggml_backend_cpu_repack_بفر_type()) {
            bufts.push_back(ggml_backend_cpu_repack_بفر_type());
        }
#جيڪڏهن_پورو

        أرجع bufts;
    }();

    أرجع bufts;
}

ساكن ggml_backend_بفر_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    ساكن std::vector<ggml_backend_بفر_type_t> extra_bufts = [] {
        std::vector<ggml_backend_بفر_type_t> bufts = ggml_backend_cpu_get_extra_بفر_types();
        bufts.push_back(خالي_اشارو);
        أرجع bufts;
    }();

    أرجع extra_bufts.معلومات();

    GGML_اڻ_استعمال_ٿيل(device);
}

ساكن منطقي ggml_backend_cpu_is_extra_بفر_type(ggml_backend_بفر_type_t buft) {
    لكل (خودڪار * extra : ggml_backend_cpu_get_extra_بفر_types()) {
        إذا (extra == buft) {
            أرجع سچ;
        }
    }
    أرجع خطأ;
}

// CPU backend - backend (stream)

بنية ggml_backend_cpu_context {
    صحيح                 n_ڌاڳا;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_معلومات;
    size_t              work_سائيز;

    ggml_abort_callback abort_callback;
    فراغ *              abort_callback_معلومات;

    منطقي                use_ref;  // use reference implementation
};

ساكن ثابت حرف * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    أرجع "CPU";

    GGML_اڻ_استعمال_ٿيل(backend);
}

ساكن فراغ ggml_backend_cpu_free(ggml_backend_t backend) {
    بنية ggml_backend_cpu_context * cpu_ctx = (بنية ggml_backend_cpu_context *)backend->context;
    ميٽيو[] cpu_ctx->work_معلومات;
    ميٽيو cpu_ctx;
    ميٽيو backend;
}

بنية ggml_backend_plan_cpu {
    بنية ggml_cplan cplan;
    بنية ggml_cgraph cgraph;
};

ساكن ggml_backend_گراف_plan_t ggml_backend_cpu_گراف_plan_create(ggml_backend_t backend, ثابت بنية ggml_cgraph * cgraph) {
    بنية ggml_backend_cpu_context * cpu_ctx = (بنية ggml_backend_cpu_context *)backend->context;

    بنية ggml_backend_plan_cpu * cpu_plan = جديد ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_گراف_plan(cgraph, cpu_ctx->n_ڌاڳا, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    إذا (cpu_plan->cplan.work_سائيز > 0) {
        cpu_plan->cplan.work_معلومات = جديد uint8_t[cpu_plan->cplan.work_سائيز];
        إذا (cpu_plan->cplan.work_معلومات == عدم) {
            ميٽيو cpu_plan;
            أرجع عدم;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_معلومات = cpu_ctx->abort_callback_معلومات;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    أرجع cpu_plan;
}

ساكن فراغ ggml_backend_cpu_گراف_plan_free(ggml_backend_t backend, ggml_backend_گراف_plan_t plan) {
    بنية ggml_backend_plan_cpu * cpu_plan = (بنية ggml_backend_plan_cpu *)plan;

    ميٽيو[] cpu_plan->cplan.work_معلومات;
    ميٽيو cpu_plan;

    GGML_اڻ_استعمال_ٿيل(backend);
}

ساكن تعداد ggml_حالت ggml_backend_cpu_گراف_plan_compute(ggml_backend_t backend, ggml_backend_گراف_plan_t plan) {
    بنية ggml_backend_plan_cpu * cpu_plan = (بنية ggml_backend_plan_cpu *)plan;

    أرجع ggml_گراف_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_اڻ_استعمال_ٿيل(backend);
}

ساكن تعداد ggml_حالت ggml_backend_cpu_گراف_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    بنية ggml_backend_cpu_context * cpu_ctx = (بنية ggml_backend_cpu_context *)backend->context;

    بنية ggml_cplan cplan = ggml_گراف_plan(cgraph, cpu_ctx->n_ڌاڳا, cpu_ctx->threadpool);

    إذا (cpu_ctx->work_سائيز < cplan.work_سائيز) {
        ميٽيو[] cpu_ctx->work_معلومات;
        cpu_ctx->work_معلومات = جديد uint8_t[cplan.work_سائيز];
        إذا (cpu_ctx->work_معلومات == عدم) {
            cpu_ctx->work_سائيز = 0;
            أرجع GGML_حالت_ALLOC_FAILED;
        }
        cpu_ctx->work_سائيز = cplan.work_سائيز;
    }
    cplan.work_معلومات = (uint8_t *)cpu_ctx->work_معلومات;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_معلومات = cpu_ctx->abort_callback_معلومات;
    cplan.use_ref             = cpu_ctx->use_ref;

    أرجع ggml_گراف_compute(cgraph, &cplan);
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
    /* .graph_plan_create       = */ ggml_backend_cpu_گراف_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_گراف_plan_free,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ ggml_backend_cpu_گراف_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_گراف_compute,
    /* .event_record            = */ عدم,
    /* .event_wait              = */ عدم,
    /* .graph_optimize          = */ عدم,
};

ساكن ggml_guid_t ggml_backend_cpu_guid(فراغ) {
    ساكن ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    أرجع &guid;
}

ggml_backend_t ggml_backend_cpu_init(فراغ) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    بنية ggml_backend_cpu_context * ctx = جديد ggml_backend_cpu_context;
    إذا (ctx == عدم) {
        أرجع عدم;
    }

    ctx->n_ڌاڳا           = GGML_DEFAULT_N_ڌاڳا;
    ctx->threadpool          = عدم;
    ctx->work_معلومات           = عدم;
    ctx->work_سائيز           = 0;
    ctx->abort_callback      = عدم;
    ctx->abort_callback_معلومات = عدم;
    ctx->use_ref             = خطأ;

    ggml_backend_t cpu_backend = جديد ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    إذا (cpu_backend == عدم) {
        ميٽيو ctx;
        أرجع عدم;
    }

    أرجع cpu_backend;
}

منطقي ggml_backend_is_cpu(ggml_backend_t backend) {
    أرجع backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

فراغ ggml_backend_cpu_سيٽ_n_ڌاڳا(ggml_backend_t backend_cpu, صحيح n_ڌاڳا) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    بنية ggml_backend_cpu_context * ctx = (بنية ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_ڌاڳا = n_ڌاڳا;
}

فراغ ggml_backend_cpu_سيٽ_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    بنية ggml_backend_cpu_context * ctx = (بنية ggml_backend_cpu_context *)backend_cpu->context;

    إذا (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

فراغ ggml_backend_cpu_سيٽ_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, فراغ * abort_callback_معلومات) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    بنية ggml_backend_cpu_context * ctx = (بنية ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_معلومات = abort_callback_معلومات;
}

فراغ ggml_backend_cpu_سيٽ_use_ref(ggml_backend_t backend_cpu, منطقي use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    بنية ggml_backend_cpu_context * ctx = (بنية ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

بنية ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#جيڪڏهن_وصف_ٿيل __APPLE__
        size_t len = 0;
        إذا (!sysctlbyname("machdep.cpu.brand_string", عدم, &len, عدم, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, عدم, 0); // NOLINT
        }
#وإلاإذا defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        إذا (f) {
            حرف buf[1024];
            طالما (fgets(buf, حجم(buf), f)) {
                إذا (لڙي_ڀيٽ(buf, "model name", 10) == 0) {
                    حرف * p = strchr(buf, ':');
                    إذا (p) {
                        p++;
                        طالما (std::خال_آهي(*p)) {
                            p++;
                        }
                        طالما (std::خال_آهي(p[لڙي_ڊيگهه(p) - 1])) {
                            p[لڙي_ڊيگهه(p) - 1] = '\0';
                        }
                        description = p;
                        اكسر;
                    }
                }
            }
            fclose(f);
        }
#وإلاإذا defined(_WIN32)
        HKEY h_ڪنجي;
        إذا (Reg_Open_ڪنجي_Ex(HKEY_LOCAL_MACHINE,
                        متن("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        ڪنجي_READ,
                        &h_ڪنجي) == ERROR_SUCCESS) {
            DWORD cpu_brand_سائيز = 0;
            إذا (Reg_Query_قدر_Ex_A(h_ڪنجي,
                                "ProcessorNameString",
                                عدم,
                                عدم,
                                عدم,
                                &cpu_brand_سائيز) == ERROR_SUCCESS) {
                description.resize(cpu_brand_سائيز);
                إذا (Reg_Query_قدر_Ex_A(h_ڪنجي,
                                    "ProcessorNameString",
                                    عدم,
                                    عدم,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_سائيز) == ERROR_SUCCESS) {
                    إذا (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_ڪنجي(h_ڪنجي);
        }
#جيڪڏهن_پورو
    }
};

ساكن ثابت حرف * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    أرجع "CPU";

    GGML_اڻ_استعمال_ٿيل(dev);
}

ساكن ثابت حرف * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    بنية ggml_backend_cpu_device_context * ctx = (بنية ggml_backend_cpu_device_context *)dev->context;

    أرجع ctx->description.c_str();
}

ساكن فراغ ggml_backend_cpu_device_get_يادگيري(ggml_backend_dev_t dev, size_t * حرر, size_t * ڪل) {
#جيڪڏهن_وصف_ٿيل _WIN32
    MEMORYSTATUSEX حالت;
    حالت.dw_ڊيگهائي = حجم(حالت);
    Global_يادگيري_حالت_Ex(&حالت);
    *ڪل = حالت.ull_ڪل_Phys;
    *حرر = حالت.ullAvailPhys;
#وإلا
    ڊگهو صفحا = sysconf(_SC_PHYS_PAGES);
    ڊگهو صفحو_سائيز = sysconf(_SC_PAGE_SIZE);
    *ڪل = صفحا * صفحو_سائيز;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *حرر = *ڪل;
#جيڪڏهن_پورو // _WIN32

    GGML_اڻ_استعمال_ٿيل(dev);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    أرجع GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_اڻ_استعمال_ٿيل(dev);
}

ساكن فراغ ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_يادگيري(dev, &props->يادگيري_free, &props->يادگيري_ڪل);
    props->caps = {
        /* .async                 = */ خطأ,
        /* .host_buffer           = */ خطأ,
        /* .buffer_from_host_ptr  = */ سچ,
        /* .events                = */ خطأ,
        /* .mmap_support          = */ سچ,
    };
}

ساكن ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    أرجع ggml_backend_cpu_init();

    GGML_اڻ_استعمال_ٿيل(dev);
    GGML_اڻ_استعمال_ٿيل(params);
}

ساكن ggml_backend_بفر_type_t ggml_backend_cpu_device_get_بفر_type(ggml_backend_dev_t dev) {
    أرجع ggml_backend_cpu_بفر_type();

    GGML_اڻ_استعمال_ٿيل(dev);
}

ساكن ggml_backend_بفر_t ggml_backend_cpu_device_بفر_from_host_ptr(ggml_backend_dev_t dev, فراغ * ptr, size_t size, size_t max_tensor_سائيز) {
    أرجع ggml_backend_cpu_بفر_from_ptr(ptr, size);

    GGML_اڻ_استعمال_ٿيل(dev);
    GGML_اڻ_استعمال_ٿيل(max_tensor_سائيز);
}

ساكن منطقي ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, ثابت بنية ggml_tensor * op) {
    ثابت بنية ggml_tensor * src0 = op->src[0];
    ثابت بنية ggml_tensor * src1 = op->src[1];

    إذا (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_ترتيب_بدلايو || op->op == GGML_OP_TRANSPOSE) {
        أرجع سچ;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    لكل (صحيح i = 0; i < 4; i++) {
        إذا (op->src[i] && op->src[i]->بفر &&
            ggml_backend_cpu_is_extra_بفر_type(op->src[i]->بفر->buft)) {
            خودڪار * buf_extra = (ggml::cpu::extra_بفر_type *) op->src[i]->بفر->buft->context;
            أرجع buf_extra->supports_op(dev, op);
        }
    }

    بدل (op->op) {
        حالة GGML_OP_CPY:
        حالة GGML_OP_سيٽ_صفون:
            أرجع
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        حالة GGML_OP_MUL_MAT:
            إذا (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                أرجع src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            أرجع src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        حالة GGML_OP_SOFT_MAX_پٺ: {
            إذا (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                أرجع خطأ;
            }
            عائم max_bias = 0.0f;

            يادگيري_نقل(&max_bias, (ثابت عائم *) op->op_params + 1, حجم(عائم));

            أرجع max_bias == 0.0f;
        }
        حالة GGML_OP_IM2_کانو_پٺ:
            أرجع src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        حالة GGML_OP_GET_صفون_پٺ:
            أرجع src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        حالة GGML_OP_ٻاهريون_PROD:
            أرجع (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        حالة GGML_OP_CONV_2D:
            أرجع ggml_is_contiguous(op->src[0]);
        حالة GGML_OP_SSM_جاچ:
            أرجع ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        افتراضي:
            أرجع سچ;
    }
}

ساكن منطقي ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    أرجع ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_بفر_type(buft);
    GGML_اڻ_استعمال_ٿيل(dev);
}

ساكن ثابت بنية ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_يادگيري,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_بفر_type,
    /* .get_host_buffer_type = */ عدم,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_بفر_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ عدم,
    /* .event_new            = */ عدم,
    /* .event_free           = */ عدم,
    /* .event_synchronize    = */ عدم,
};

// CPU backend - backend (reg)

ساكن ثابت حرف * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    أرجع "CPU";

    GGML_اڻ_استعمال_ٿيل(reg);
}

ساكن size_t ggml_backend_cpu_reg_get_device_ڳڻپ(ggml_backend_reg_t reg) {
    أرجع 1;

    GGML_اڻ_استعمال_ٿيل(reg);
}

ساكن ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t اشاريو) {
    GGML_ASSERT(اشاريو == 0);

    ساكن ggml_backend_cpu_device_context ctx;
    ساكن ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    أرجع &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
ساكن ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    ساكن std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        إذا (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        إذا (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        إذا (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        إذا (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        إذا (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        إذا (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        إذا (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        إذا (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        إذا (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        إذا (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        إذا (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        إذا (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        إذا (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        إذا (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        إذا (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        إذا (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        إذا (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        إذا (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        إذا (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        إذا (ggml_cpu_get_sve_cnt() > 0) {
            ساكن std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        إذا (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        إذا (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        إذا (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        إذا (ggml_cpu_get_rvv_vlen() > 0) {
            ساكن std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        إذا (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        إذا (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        إذا (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        إذا (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #جيڪڏهن_وصف_ٿيل GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #جيڪڏهن_پورو
    #جيڪڏهن_وصف_ٿيل GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #جيڪڏهن_پورو
    #جيڪڏهن_وصف_ٿيل GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #جيڪڏهن_پورو
    #جيڪڏهن_وصف_ٿيل GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #جيڪڏهن_پورو
    #جيڪڏهن_وصف_ٿيل GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #جيڪڏهن_پورو

        features.push_back({ خالي_اشارو, خالي_اشارو });

        أرجع features;
    }();

    أرجع features.معلومات();

    GGML_اڻ_استعمال_ٿيل(reg);
}

ساكن فراغ * ggml_backend_cpu_get_proc_پتو(ggml_backend_reg_t reg, ثابت حرف * name) {
    إذا (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_سيٽ_n_ڌاڳا_t fct = ggml_backend_cpu_سيٽ_n_ڌاڳا;
        أرجع (فراغ *)fct;
    }
    إذا (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        أرجع (فراغ *)fct;
    }
    إذا (strcmp(name, "ggml_backend_get_features") == 0) {
        أرجع (فراغ *)ggml_backend_cpu_get_features;
    }
    إذا (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        أرجع (فراغ *)ggml_backend_cpu_سيٽ_abort_callback;
    }
    إذا (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        أرجع (فراغ *)ggml_numa_init;
    }
    إذا (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        أرجع (فراغ *)ggml_is_numa;
    }
    إذا (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        أرجع (فراغ *)ggml_backend_cpu_سيٽ_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    إذا (strcmp(name, "ggml_threadpool_new") == 0) {
        أرجع (فراغ *)ggml_threadpool_new;
    }
    إذا (strcmp(name, "ggml_threadpool_free") == 0) {
        أرجع (فراغ *)ggml_threadpool_free;
    }
    إذا (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        أرجع (فراغ *)ggml_backend_cpu_سيٽ_threadpool;
    }

    أرجع عدم;

    GGML_اڻ_استعمال_ٿيل(reg);
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_ڳڻپ,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_پتو,
};

ggml_backend_reg_t ggml_backend_cpu_reg(فراغ) {
    // init CPU feature detection
    ggml_cpu_init();

    ساكن بنية ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخو,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ عدم,
    };

    أرجع &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
