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

#اگر_تعریف_شدہ GGML_USE_CPU_HBM
#    include "hbm.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#اگر_ختم

#اگر defined(_WIN32)
#    define WIN32_LEAN_AND_اوسط
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#ورنہ
#    include <unistd.h>
#اگر_ختم

#اگر defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#اگر_ختم

// ggml-backend interface

std::vector<ggml_backend_بفر_type_t> & ggml_backend_cpu_get_extra_بفر_types() {
    ساکن std::vector<ggml_backend_بفر_type_t> bufts = []() {
        std::vector<ggml_backend_بفر_type_t> bufts;

#اگر defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        اگر (ggml_backend_amx_بفر_type()) {
            bufts.push_back(ggml_backend_amx_بفر_type());
        }
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_CPU_RISCV64_SPACEMIT
        اگر (ggml_backend_cpu_riscv64_spacemit_بفر_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_بفر_type());
        }
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_CPU_KLEIDIAI
        اگر (ggml_backend_cpu_kleidiai_بفر_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_بفر_type());
        }
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_CPU_REPACK
        اگر (ggml_backend_cpu_repack_بفر_type()) {
            bufts.push_back(ggml_backend_cpu_repack_بفر_type());
        }
#اگر_ختم

        واپس bufts;
    }();

    واپس bufts;
}

ساکن ggml_backend_بفر_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    ساکن std::vector<ggml_backend_بفر_type_t> extra_bufts = [] {
        std::vector<ggml_backend_بفر_type_t> bufts = ggml_backend_cpu_get_extra_بفر_types();
        bufts.push_back(خالی_اشاریہ);
        واپس bufts;
    }();

    واپس extra_bufts.ڈیٹا();

    GGML_غیر_مستعمل(device);
}

ساکن منطقی ggml_backend_cpu_is_extra_بفر_type(ggml_backend_بفر_type_t buft) {
    برائے (خودکار * extra : ggml_backend_cpu_get_extra_بفر_types()) {
        اگر (extra == buft) {
            واپس سچ;
        }
    }
    واپس جھوٹ;
}

// CPU backend - backend (stream)

ساخت ggml_backend_cpu_context {
    صحیح                 n_دھاگے;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_ڈیٹا;
    size_t              work_سائز;

    ggml_abort_callback abort_callback;
    خالی *              abort_callback_ڈیٹا;

    منطقی                use_ref;  // use reference implementation
};

ساکن ثابت حرف * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    واپس "CPU";

    GGML_غیر_مستعمل(backend);
}

ساکن خالی ggml_backend_cpu_free(ggml_backend_t backend) {
    ساخت ggml_backend_cpu_context * cpu_ctx = (ساخت ggml_backend_cpu_context *)backend->context;
    حذف[] cpu_ctx->work_ڈیٹا;
    حذف cpu_ctx;
    حذف backend;
}

ساخت ggml_backend_plan_cpu {
    ساخت ggml_cplan cplan;
    ساخت ggml_cgraph cgraph;
};

ساکن ggml_backend_گراف_plan_t ggml_backend_cpu_گراف_plan_create(ggml_backend_t backend, ثابت ساخت ggml_cgraph * cgraph) {
    ساخت ggml_backend_cpu_context * cpu_ctx = (ساخت ggml_backend_cpu_context *)backend->context;

    ساخت ggml_backend_plan_cpu * cpu_plan = نیا ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_گراف_plan(cgraph, cpu_ctx->n_دھاگے, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    اگر (cpu_plan->cplan.work_سائز > 0) {
        cpu_plan->cplan.work_ڈیٹا = نیا uint8_t[cpu_plan->cplan.work_سائز];
        اگر (cpu_plan->cplan.work_ڈیٹا == تہی) {
            حذف cpu_plan;
            واپس تہی;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_ڈیٹا = cpu_ctx->abort_callback_ڈیٹا;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    واپس cpu_plan;
}

ساکن خالی ggml_backend_cpu_گراف_plan_free(ggml_backend_t backend, ggml_backend_گراف_plan_t plan) {
    ساخت ggml_backend_plan_cpu * cpu_plan = (ساخت ggml_backend_plan_cpu *)plan;

    حذف[] cpu_plan->cplan.work_ڈیٹا;
    حذف cpu_plan;

    GGML_غیر_مستعمل(backend);
}

ساکن شمار ggml_حالت ggml_backend_cpu_گراف_plan_compute(ggml_backend_t backend, ggml_backend_گراف_plan_t plan) {
    ساخت ggml_backend_plan_cpu * cpu_plan = (ساخت ggml_backend_plan_cpu *)plan;

    واپس ggml_گراف_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_غیر_مستعمل(backend);
}

ساکن شمار ggml_حالت ggml_backend_cpu_گراف_compute(ggml_backend_t backend, ساخت ggml_cgraph * cgraph) {
    ساخت ggml_backend_cpu_context * cpu_ctx = (ساخت ggml_backend_cpu_context *)backend->context;

    ساخت ggml_cplan cplan = ggml_گراف_plan(cgraph, cpu_ctx->n_دھاگے, cpu_ctx->threadpool);

    اگر (cpu_ctx->work_سائز < cplan.work_سائز) {
        حذف[] cpu_ctx->work_ڈیٹا;
        cpu_ctx->work_ڈیٹا = نیا uint8_t[cplan.work_سائز];
        اگر (cpu_ctx->work_ڈیٹا == تہی) {
            cpu_ctx->work_سائز = 0;
            واپس GGML_حالت_ALLOC_FAILED;
        }
        cpu_ctx->work_سائز = cplan.work_سائز;
    }
    cplan.work_ڈیٹا = (uint8_t *)cpu_ctx->work_ڈیٹا;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_ڈیٹا = cpu_ctx->abort_callback_ڈیٹا;
    cplan.use_ref             = cpu_ctx->use_ref;

    واپس ggml_گراف_compute(cgraph, &cplan);
}

ساکن ثابت ساخت ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ تہی,
    /* .get_tensor_async        = */ تہی,
    /* .set_tensor_2d_async     = */ تہی,
    /* .get_tensor_2d_async     = */ تہی,
    /* .cpy_tensor_async        = */ تہی,
    /* .synchronize             = */ تہی,
    /* .graph_plan_create       = */ ggml_backend_cpu_گراف_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_گراف_plan_free,
    /* .graph_plan_update       = */ تہی,
    /* .graph_plan_compute      = */ ggml_backend_cpu_گراف_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_گراف_compute,
    /* .event_record            = */ تہی,
    /* .event_wait              = */ تہی,
    /* .graph_optimize          = */ تہی,
};

ساکن ggml_guid_t ggml_backend_cpu_guid(خالی) {
    ساکن ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    واپس &guid;
}

ggml_backend_t ggml_backend_cpu_init(خالی) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    ساخت ggml_backend_cpu_context * ctx = نیا ggml_backend_cpu_context;
    اگر (ctx == تہی) {
        واپس تہی;
    }

    ctx->n_دھاگے           = GGML_DEFAULT_N_دھاگے;
    ctx->threadpool          = تہی;
    ctx->work_ڈیٹا           = تہی;
    ctx->work_سائز           = 0;
    ctx->abort_callback      = تہی;
    ctx->abort_callback_ڈیٹا = تہی;
    ctx->use_ref             = جھوٹ;

    ggml_backend_t cpu_backend = نیا ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    اگر (cpu_backend == تہی) {
        حذف ctx;
        واپس تہی;
    }

    واپس cpu_backend;
}

منطقی ggml_backend_is_cpu(ggml_backend_t backend) {
    واپس backend != تہی && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

خالی ggml_backend_cpu_سیٹ_n_دھاگے(ggml_backend_t backend_cpu, صحیح n_دھاگے) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ساخت ggml_backend_cpu_context * ctx = (ساخت ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_دھاگے = n_دھاگے;
}

خالی ggml_backend_cpu_سیٹ_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ساخت ggml_backend_cpu_context * ctx = (ساخت ggml_backend_cpu_context *)backend_cpu->context;

    اگر (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

خالی ggml_backend_cpu_سیٹ_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, خالی * abort_callback_ڈیٹا) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ساخت ggml_backend_cpu_context * ctx = (ساخت ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_ڈیٹا = abort_callback_ڈیٹا;
}

خالی ggml_backend_cpu_سیٹ_use_ref(ggml_backend_t backend_cpu, منطقی use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    ساخت ggml_backend_cpu_context * ctx = (ساخت ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

ساخت ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#اگر_تعریف_شدہ __APPLE__
        size_t len = 0;
        اگر (!sysctlbyname("machdep.cpu.brand_string", تہی, &len, تہی, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, تہی, 0); // NOLINT
        }
#وگرنہ_اگر defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        اگر (f) {
            حرف buf[1024];
            جب (fgets(buf, حجم(buf), f)) {
                اگر (لڑی_موازنہ(buf, "model name", 10) == 0) {
                    حرف * p = strchr(buf, ':');
                    اگر (p) {
                        p++;
                        جب (std::خلا_ہے(*p)) {
                            p++;
                        }
                        جب (std::خلا_ہے(p[لڑی_لمبائی(p) - 1])) {
                            p[لڑی_لمبائی(p) - 1] = '\0';
                        }
                        description = p;
                        توڑو;
                    }
                }
            }
            fclose(f);
        }
#وگرنہ_اگر defined(_WIN32)
        HKEY h_کلید;
        اگر (Reg_Open_کلید_Ex(HKEY_LOCAL_MACHINE,
                        متن("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        کلید_READ,
                        &h_کلید) == ERROR_SUCCESS) {
            DWORD cpu_brand_سائز = 0;
            اگر (Reg_Query_قدر_Ex_A(h_کلید,
                                "ProcessorNameString",
                                تہی,
                                تہی,
                                تہی,
                                &cpu_brand_سائز) == ERROR_SUCCESS) {
                description.resize(cpu_brand_سائز);
                اگر (Reg_Query_قدر_Ex_A(h_کلید,
                                    "ProcessorNameString",
                                    تہی,
                                    تہی,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_سائز) == ERROR_SUCCESS) {
                    اگر (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            Reg_Close_کلید(h_کلید);
        }
#اگر_ختم
    }
};

ساکن ثابت حرف * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    واپس "CPU";

    GGML_غیر_مستعمل(dev);
}

ساکن ثابت حرف * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    ساخت ggml_backend_cpu_device_context * ctx = (ساخت ggml_backend_cpu_device_context *)dev->context;

    واپس ctx->description.c_str();
}

ساکن خالی ggml_backend_cpu_device_get_حافظہ(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل) {
#اگر_تعریف_شدہ _WIN32
    MEMORYSTATUSEX حالت_2;
    حالت_2.dw_لمبائی = حجم(حالت_2);
    Global_حافظہ_حالت_Ex(&حالت_2);
    *کل = حالت_2.ull_کل_Phys;
    *آزاد = حالت_2.ullAvailPhys;
#ورنہ
    طویل صفحات = sysconf(_SC_PHYS_PAGES);
    طویل صفحہ_سائز = sysconf(_SC_PAGE_SIZE);
    *کل = صفحات * صفحہ_سائز;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *آزاد = *کل;
#اگر_ختم // _WIN32

    GGML_غیر_مستعمل(dev);
}

ساکن شمار ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    واپس GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_غیر_مستعمل(dev);
}

ساکن خالی ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, ساخت ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);
    props->caps = {
        /* .async                 = */ جھوٹ,
        /* .host_buffer           = */ جھوٹ,
        /* .buffer_from_host_ptr  = */ سچ,
        /* .events                = */ جھوٹ,
        /* .mmap_support          = */ سچ,
    };
}

ساکن ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    واپس ggml_backend_cpu_init();

    GGML_غیر_مستعمل(dev);
    GGML_غیر_مستعمل(params);
}

ساکن ggml_backend_بفر_type_t ggml_backend_cpu_device_get_بفر_type(ggml_backend_dev_t dev) {
    واپس ggml_backend_cpu_بفر_type();

    GGML_غیر_مستعمل(dev);
}

ساکن ggml_backend_بفر_t ggml_backend_cpu_device_بفر_from_host_ptr(ggml_backend_dev_t dev, خالی * ptr, size_t size, size_t max_tensor_سائز) {
    واپس ggml_backend_cpu_بفر_from_ptr(ptr, size);

    GGML_غیر_مستعمل(dev);
    GGML_غیر_مستعمل(max_tensor_سائز);
}

ساکن منطقی ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, ثابت ساخت ggml_tensor * op) {
    ثابت ساخت ggml_tensor * src0 = op->src[0];
    ثابت ساخت ggml_tensor * src1 = op->src[1];

    اگر (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_ترتیب_بدلو || op->op == GGML_OP_TRANSPOSE) {
        واپس سچ;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    برائے (صحیح i = 0; i < 4; i++) {
        اگر (op->src[i] && op->src[i]->بفر &&
            ggml_backend_cpu_is_extra_بفر_type(op->src[i]->بفر->buft)) {
            خودکار * buf_extra = (ggml::cpu::extra_بفر_type *) op->src[i]->بفر->buft->context;
            واپس buf_extra->supports_op(dev, op);
        }
    }

    انتخاب (op->op) {
        حالت GGML_OP_CPY:
        حالت GGML_OP_سیٹ_صفیں:
            واپس
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
                واپس src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            واپس src1->type == GGML_TYPE_F32 ||
                   src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type ||
                   (src1->type == GGML_TYPE_BF16 && ggml_get_type_traits_cpu(src0->type)->vec_dot_type == GGML_TYPE_F32);
        حالت GGML_OP_SOFT_MAX_عقب: {
            اگر (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                واپس جھوٹ;
            }
            اعشاری max_bias = 0.0f;

            حافظہ_نقل(&max_bias, (ثابت اعشاری *) op->op_params + 1, حجم(اعشاری));

            واپس max_bias == 0.0f;
        }
        حالت GGML_OP_IM2_ستون_عقب:
            واپس src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        حالت GGML_OP_GET_صفیں_عقب:
            واپس src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        حالت GGML_OP_برآمد_PROD:
            واپس (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        حالت GGML_OP_CONV_2D:
            واپس ggml_is_contiguous(op->src[0]);
        حالت GGML_OP_SSM_جائزہ:
            واپس ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        طےشدہ:
            واپس سچ;
    }
}

ساکن منطقی ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    واپس ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_بفر_type(buft);
    GGML_غیر_مستعمل(dev);
}

ساکن ثابت ساخت ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_بفر_type,
    /* .get_host_buffer_type = */ تہی,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_بفر_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ تہی,
    /* .event_new            = */ تہی,
    /* .event_free           = */ تہی,
    /* .event_synchronize    = */ تہی,
};

// CPU backend - backend (reg)

ساکن ثابت حرف * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    واپس "CPU";

    GGML_غیر_مستعمل(reg);
}

ساکن size_t ggml_backend_cpu_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    واپس 1;

    GGML_غیر_مستعمل(reg);
}

ساکن ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t اشاریہ) {
    GGML_ASSERT(اشاریہ == 0);

    ساکن ggml_backend_cpu_device_context ctx;
    ساکن ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    واپس &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
ساکن ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    ساکن std::vector<ggml_backend_feature> features = []() {
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
            ساکن std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
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
            ساکن std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
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
    #اگر_تعریف_شدہ GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #اگر_ختم
    #اگر_تعریف_شدہ GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #اگر_ختم
    #اگر_تعریف_شدہ GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #اگر_ختم
    #اگر_تعریف_شدہ GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #اگر_ختم
    #اگر_تعریف_شدہ GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #اگر_ختم

        features.push_back({ خالی_اشاریہ, خالی_اشاریہ });

        واپس features;
    }();

    واپس features.ڈیٹا();

    GGML_غیر_مستعمل(reg);
}

ساکن خالی * ggml_backend_cpu_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    اگر (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_سیٹ_n_دھاگے_t fct = ggml_backend_cpu_سیٹ_n_دھاگے;
        واپس (خالی *)fct;
    }
    اگر (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        واپس (خالی *)fct;
    }
    اگر (strcmp(name, "ggml_backend_get_features") == 0) {
        واپس (خالی *)ggml_backend_cpu_get_features;
    }
    اگر (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        واپس (خالی *)ggml_backend_cpu_سیٹ_abort_callback;
    }
    اگر (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        واپس (خالی *)ggml_numa_init;
    }
    اگر (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        واپس (خالی *)ggml_is_numa;
    }
    اگر (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        واپس (خالی *)ggml_backend_cpu_سیٹ_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    اگر (strcmp(name, "ggml_threadpool_new") == 0) {
        واپس (خالی *)ggml_threadpool_new;
    }
    اگر (strcmp(name, "ggml_threadpool_free") == 0) {
        واپس (خالی *)ggml_threadpool_free;
    }
    اگر (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        واپس (خالی *)ggml_backend_cpu_سیٹ_threadpool;
    }

    واپس تہی;

    GGML_غیر_مستعمل(reg);
}

ساکن ثابت ساخت ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_گنتی,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_پتہ,
};

ggml_backend_reg_t ggml_backend_cpu_reg(خالی) {
    // init CPU feature detection
    ggml_cpu_init();

    ساکن ساخت ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخہ,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ تہی,
    };

    واپس &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
