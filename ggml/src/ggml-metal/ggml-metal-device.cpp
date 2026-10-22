#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-device.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-tuning.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>

ਬਣਤਰ ggml_metal_device_deleter {
    ਖਾਲੀ ਸੰਚਾਲਕ()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

ggml_metal_device_t ggml_metal_device_get(ਪੂਰਨ device, ਪੂਰਨ n_devices) {
    ਸਥਿਰ std::vector<ggml_metal_device_ptr> devs;

    devs.emplace_back(ggml_metal_device_init(device, n_devices));

    ਵਾਪਸ devs.back().get();
}

ਬਣਤਰ ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> ਅੰਕੜੇ;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(ਖਾਲੀ) {
    ggml_metal_pipelines_t res = ਨਵਾਂ ggml_metal_pipelines();

    ਵਾਪਸ res;
}

ਖਾਲੀ ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    ਜੇ (!ppls) {
        ਵਾਪਸ;
    }

    ਲਈ (ਸਵੈਚਾਲਿਤ it = ppls->ਅੰਕੜੇ.begin(); it != ppls->ਅੰਕੜੇ.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    ਮਿਟਾਓ ppls;
}

ਖਾਲੀ ggml_metal_pipelines_ਜੋੜੋ(ggml_metal_pipelines_t ppls, ਅਚਲ ਅੱਖਰ * name, ggml_metal_pipeline_t pipeline) {
    ppls->ਅੰਕੜੇ[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ਅਚਲ ਅੱਖਰ * name) {
    ਜੇ (ppls->ਅੰਕੜੇ.find(name) == ppls->ਅੰਕੜੇ.end()) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਵਾਪਸ ppls->ਅੰਕੜੇ[name];
}

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਅਧਾਰ(ggml_metal_library_t lib, ggml_op op) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਅੱਖਰ * op_str = "undefined";
    ਚੋਣ (op) {
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID: op_str = "add_id"; ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    };

    snprintf(ਅਧਾਰ, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    ਅਚਲ ਅੱਖਰ * pool_str = "undefined";
    ਚੋਣ (op_pool) {
        ਹਾਲਤ GGML_OP_POOL_AVG: pool_str = "avg"; ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_MAX: pool_str = "max"; ਰੋਕੋ;
        ਮੂਲ: GGML_ASSERT(ਝੂਠ && "not implemented");
    };

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, ਆਕਾਰ(ਅਧਾਰ), "kernel_pool_1d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, ਆਕਾਰ(name), "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    ਅਚਲ ਅੱਖਰ * pool_str = "undefined";
    ਚੋਣ (op_pool) {
        ਹਾਲਤ GGML_OP_POOL_AVG: pool_str = "avg"; ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_MAX: pool_str = "max"; ਰੋਕੋ;
        ਮੂਲ: GGML_ASSERT(ਝੂਠ && "not implemented");
    };

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_ਪੰਕਤੀਆਂ(ggml_metal_library_t lib, ggml_type tsrc) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਸਵੈਚਾਲਿਤ tsrc = op->src[0]->type;
    ਅਚਲ ਸਵੈਚਾਲਿਤ tidx = op->src[1]->type;
    ਅਚਲ ਸਵੈਚਾਲਿਤ tdst = op->type;

    snprintf(ਅਧਾਰ, 256, "kernel_set_rows_%s_%s_%s", ggml_type_name(tsrc), ggml_type_name(tidx), ggml_type_name(tdst));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਪੂਰਨ n = op->src[0]->ne[0];

    snprintf(ਅਧਾਰ, 256, "kernel_diag_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_n=%d", ਅਧਾਰ, n);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    res.nsg  = 1;
    res.smem = 0;

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat(ggml_metal_library_t lib, ggml_type tsrc) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat(ggml_metal_library_t lib, ggml_type tsrc) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_concat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਪੂਰਨ op_ਅੰਕ = -1;

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_SCALE:      op_ਅੰਕ = OP_UNARY_ਅੰਕ_SCALE;      ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FILL:       op_ਅੰਕ = OP_UNARY_ਅੰਕ_FILL;       ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CLAMP:      op_ਅੰਕ = OP_UNARY_ਅੰਕ_CLAMP;      ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQR:        op_ਅੰਕ = OP_UNARY_ਅੰਕ_SQR;        ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQRT:       op_ਅੰਕ = OP_UNARY_ਅੰਕ_SQRT;       ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SIN:        op_ਅੰਕ = OP_UNARY_ਅੰਕ_SIN;        ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_COS:        op_ਅੰਕ = OP_UNARY_ਅੰਕ_COS;        ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LOG:        op_ਅੰਕ = OP_UNARY_ਅੰਕ_LOG;        ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LEAKY_RELU: op_ਅੰਕ = OP_UNARY_ਅੰਕ_LEAKY_RELU; ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UNARY:
            ਚੋਣ (ggml_get_unary_op(op)) {
                ਹਾਲਤ GGML_UNARY_OP_TANH:        op_ਅੰਕ = OP_UNARY_ਅੰਕ_TANH;        ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_RELU:        op_ਅੰਕ = OP_UNARY_ਅੰਕ_RELU;        ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SIGMOID:     op_ਅੰਕ = OP_UNARY_ਅੰਕ_SIGMOID;     ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_GELU:        op_ਅੰਕ = OP_UNARY_ਅੰਕ_GELU;        ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:    op_ਅੰਕ = OP_UNARY_ਅੰਕ_GELU_ERF;    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:  op_ਅੰਕ = OP_UNARY_ਅੰਕ_GELU_ਤੇਜ਼;  ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SILU:        op_ਅੰਕ = OP_UNARY_ਅੰਕ_SILU;        ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ELU:         op_ਅੰਕ = OP_UNARY_ਅੰਕ_ELU;         ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_NEG:         op_ਅੰਕ = OP_UNARY_ਅੰਕ_NEG;         ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ABS:         op_ਅੰਕ = OP_UNARY_ਅੰਕ_ABS;         ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SGN:         op_ਅੰਕ = OP_UNARY_ਅੰਕ_SGN;         ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:        op_ਅੰਕ = OP_UNARY_ਅੰਕ_ਕਦਮ;        ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:   op_ਅੰਕ = OP_UNARY_ਅੰਕ_HARDSWISH;   ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID: op_ਅੰਕ = OP_UNARY_ਅੰਕ_HARDSIGMOID; ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_EXP:         op_ਅੰਕ = OP_UNARY_ਅੰਕ_EXP;         ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:    op_ਅੰਕ = OP_UNARY_ਅੰਕ_SOFTPLUS;    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_EXPM1:       op_ਅੰਕ = OP_UNARY_ਅੰਕ_EXPM1;       ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_FLOOR:       op_ਅੰਕ = OP_UNARY_ਅੰਕ_FLOOR;       ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_CEIL:        op_ਅੰਕ = OP_UNARY_ਅੰਕ_CEIL;        ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:       op_ਅੰਕ = OP_UNARY_ਅੰਕ_ਗੇੜਾ;       ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_TRUNC:       op_ਅੰਕ = OP_UNARY_ਅੰਕ_TRUNC;       ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_XIELU:       op_ਅੰਕ = OP_UNARY_ਅੰਕ_XIELU;       ਰੋਕੋ;
                ਮੂਲ: GGML_ABORT("fatal error");
            } ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    };

    ਅਚਲ ਅੱਖਰ * t0_str = ggml_type_name(op->src[0]->type);
    ਅਚਲ ਅੱਖਰ * t_str  = ggml_type_name(op->type);

    ਅਚਲ ਤਾਰਕਿਕ is_c4 = op->src[0]->ne[0] % 4 == 0;
    ਅਚਲ ਤਾਰਕਿਕ is_cnt = ggml_is_contiguous(op->src[0]) && ggml_nelements(op) < 32768;

    snprintf(ਅਧਾਰ, 256, "kernel_unary_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_cnt=%d", ਅਧਾਰ, op_ਅੰਕ, is_cnt);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, op_ਅੰਕ, FC_UNARY + 0);
        ggml_metal_cv_ਸੈੱਟ_bool (cv, is_cnt, FC_UNARY + 1);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_cnt;

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਅੱਖਰ * op_str = "undefined";
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_GLU:
            ਚੋਣ (ggml_get_glu_op(op)) {
                ਹਾਲਤ GGML_GLU_OP_REGLU:        op_str = "reglu";        ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU:        op_str = "geglu";        ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:  op_str = "geglu_quick";  ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP: op_str = "swiglu_clamp"; ਰੋਕੋ;
                ਮੂਲ: GGML_ABORT("fatal error");
            } ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    };

    snprintf(ਅਧਾਰ, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਜੋੜ(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ਜੋੜ);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_op_sum_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਜੋੜ_ਪੰਕਤੀਆਂ(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਪੂਰਨ op_ਅੰਕ = -1;

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ: op_ਅੰਕ = OP_ਜੋੜ_ਪੰਕਤੀਆਂ_ਅੰਕ_ਜੋੜ_ਪੰਕਤੀਆਂ; ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਔਸਤ:     op_ਅੰਕ = OP_ਜੋੜ_ਪੰਕਤੀਆਂ_ਅੰਕ_ਔਸਤ;     ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    };

    ਅਚਲ ਅੱਖਰ * t0_str = ggml_type_name(op->src[0]->type);
    ਅਚਲ ਅੱਖਰ * t_str  = ggml_type_name(op->type);

    ਅਚਲ ਤਾਰਕਿਕ is_c4 = op->src[0]->ne[0] % 4 == 0;

    snprintf(ਅਧਾਰ, 256, "kernel_sum_rows_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d", ਅਧਾਰ, op_ਅੰਕ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, op_ਅੰਕ, FC_ਜੋੜ_ਪੰਕਤੀਆਂ + 0);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਜੇ (is_c4) {
        res.smem *= 4;
    }

    res.c4  = is_c4;

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_cumsum_blk_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_ਜੋੜੋ(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_cumsum_add_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_TRI);
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_ਮਾਪ(op->src[0]->type));

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਅੱਖਰ * op_str = "tri";
    ਅਚਲ ਪੂਰਨ ttype = op->op_params[0];

    snprintf(ਅਧਾਰ, 256, "kernel_%s_%s_%d", op_str, ggml_type_name(op->src[0]->type), ttype);

    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਅੱਖਰ * suffix = "";

    ਜੇ (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    ਅਚਲ ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(ਅਧਾਰ, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    res.smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer(
        ggml_metal_library_t lib,
        ਅਚਲ ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ int16_t nh = op->src[0]->ne[1];

    snprintf(ਅਧਾਰ, 256, "kernel_lightning_indexer_%s", ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s_nh=%d", ਅਧਾਰ, nh);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, nh, FC_LIGHTNING_INDEXER + 0);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਅੱਖਰ name[256];
    ਅਚਲ ਅੱਖਰ * ਅਧਾਰ = ਖਾਲੀ_ਸੂਚਕ;

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_DSV4_HC_COMB:
            ਅਧਾਰ = "kernel_dsv4_hc_comb_f32";
            snprintf(name, 256, "%s", ਅਧਾਰ);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_PRE:
            ਅਧਾਰ = ggml_get_op_params_i32(op, 1) != 0 ? "kernel_dsv4_hc_pre_gated_f32" : "kernel_dsv4_hc_pre_f32";
            snprintf(name, 256, "%s_n_hc=%d", ਅਧਾਰ, (ਪੂਰਨ) op->src[0]->ne[1]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_POST:
            ਅਧਾਰ = op->src[3] ? "kernel_dsv4_hc_post_f32" : "kernel_dsv4_hc_post_nocomb_f32";
            snprintf(name, 256, "%s", ਅਧਾਰ);
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("fatal error");
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ਖਾਲੀ_ਸੂਚਕ;

        ਜੇ (op->op == GGML_OP_DSV4_HC_PRE) {
            cv = ggml_metal_cv_init();
            ggml_metal_cv_ਸੈੱਟ_int32(cv, (int32_t) op->src[0]->ne[1], FC_DSV4_HC + 0);
        }

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ਜੇ (cv) {
            ggml_metal_cv_free(cv);
        }
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv(
        ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, int32_t nc, ਤਾਰਕਿਕ use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਅੱਖਰ * suffix = "";

    ਜੇ (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(ਅਧਾਰ, 256, "kernel_ssm_conv_%s_%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_nc=%d_silu=%d", ਅਧਾਰ, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ਸੈੱਟ_bool(cv, use_silu, FC_SSM_CONV + 1);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, nc,      FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched(
        ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, ਪੂਰਨ ssm_conv_bs, int32_t nc, ਤਾਰਕਿਕ use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਅੱਖਰ * suffix = "";
    ਜੇ (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(ਅਧਾਰ, 256, "kernel_ssm_conv_%s_%s_batched%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_ssm_conv_bs=%d_nc=%d_silu=%d", ਅਧਾਰ, ssm_conv_bs, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, ssm_conv_bs, FC_SSM_CONV + 0);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, use_silu,     FC_SSM_CONV + 1);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, nc,          FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_ਜਾਂਚ(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, ਤਾਰਕਿਕ ਪੂਛ)  {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਪੂਰਨ nsg = (ne00 + 31)/32;

    snprintf(ਅਧਾਰ, 256, "kernel_ssm_scan_%s%s", ggml_type_name(op->src[0]->type), ਪੂਛ ? "_tail" : "");
    snprintf(name, 256, "%s_nsg=%d", ਅਧਾਰ, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (32 + 2) floats
    res.smem = GGML_PAD((32 + 2)*ਆਕਾਰ(ਦਸ਼ਮਲਵ)*nsg, 16);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_ਜਾਂਚ_ssd_mma(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op)  {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_ssm_scan_ssd_mma_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    // acs/exp(acs)/state-decay vectors + dtX + SAM rows + two 8x8 tiles per simdgroup
    res.smem = (3*OP_SSM_ਜਾਂਚ_SSD_CS +
                OP_SSM_ਜਾਂਚ_SSD_CS*OP_SSM_ਜਾਂਚ_SSD_HD +
                OP_SSM_ਜਾਂਚ_SSD_NSG*8*OP_SSM_ਜਾਂਚ_SSD_CS +
                OP_SSM_ਜਾਂਚ_SSD_NSG*2*8*8)*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ int64_t C = op->ne[0];
    ਅਚਲ int64_t H = op->src[0]->ne[1];

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(ਅਧਾਰ, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(ਅਧਾਰ, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    // v is src[2], dimensions: S_v = ne[0], H = ne[1]
    ਅਚਲ ਪੂਰਨ ne20 = op->src[2]->ne[0]; // S_v
    ਅਚਲ ਪੂਰਨ ne21 = op->src[2]->ne[1]; // H
    ਅਚਲ ਪੂਰਨ ne30 = op->src[3]->ne[0]; // G
    // state is src[5], 4D [S_v, S_v, H_v, n_seqs] (s0 only); K is op param 0.
    ਅਚਲ ਪੂਰਨ K = ggml_get_op_params_i32(op, 0);

    ਅਚਲ ਪੂਰਨ nsg = op->src[2]->ne[0]/32;

    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->ne[0] == ne20 * ne21);
    GGML_ASSERT(ne20 % 32 == 0);

    snprintf(ਅਧਾਰ, 256, "kernel_gated_delta_net_%s_%d", ggml_type_name(op->src[0]->type), nsg);
    snprintf(name, 256, "%s_ne20=%d_ne30=%d_K=%d", ਅਧਾਰ, ne20, ne30, K);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, ne20, FC_GATED_DELTA_NET + 0);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, ne30, FC_GATED_DELTA_NET + 1);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, K,    FC_GATED_DELTA_NET + 2);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg = nsg;

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਪੂਰਨ nsg = 8;
    ਅਚਲ ਪੂਰਨ n   = op->src[1]->ne[1];
    ਅਚਲ ਪੂਰਨ k   = op->src[1]->ne[0];

    snprintf(ਅਧਾਰ, 256, "kernel_solve_tri_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d_n=%d_k=%d", ਅਧਾਰ, nsg, n, k);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, nsg, FC_SOLVE_TRI + 0);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, n,   FC_SOLVE_TRI + 1);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, k,   FC_SOLVE_TRI + 2);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = GGML_PAD(GGML_PAD(n, 32)*nsg*ਆਕਾਰ(ਦਸ਼ਮਲਵ), 16);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, ਪੂਰਨ nsg, ਪੂਰਨ nxpsg, ਪੂਰਨ r1ptg) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ggml_type tsrc0 = op->src[0]->type;
    ਅਚਲ ggml_type tsrc1 = op->src[1]->type;
    ਅਚਲ ਪੂਰਨ       ne12  = op->src[1]->ne[2];
    ਅਚਲ ਪੂਰਨ       r2    = ne12 / op->src[0]->ne[2];
    ਅਚਲ ਪੂਰਨ       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    snprintf(ਅਧਾਰ, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d_ne12=%d_r2=%d_r3=%d", ਅਧਾਰ, nsg, nxpsg, ne12, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, nxpsg,          FC_MUL_MV + 1);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, (int16_t) r2,   FC_MUL_MV + 3);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, (int16_t) r3,   FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

// threadgroups needed to fill the GPU
// TODO: dedup
ਸਥਿਰ constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

// few-row MMA mat-mul: 8x8 simdgroup matrices for the src1 rows of ggml_metal_mul_mat_use_mma
ਸਥਿਰ constexpr ਪੂਰਨ     GGML_METAL_MMA_TILE          = 8;
ਸਥਿਰ constexpr ਪੂਰਨ     GGML_METAL_MMA_NT_MAX        = 4;
ਸਥਿਰ constexpr ਪੂਰਨ     GGML_METAL_MMA_SMEM_MAX      = 16384;

// simdgroups per threadgroup for at most FEW_ROWS, at most MANY_ROWS, and more src0 rows
ਸਥਿਰ constexpr int64_t GGML_METAL_MMA_FEW_ਪੰਕਤੀਆਂ      = 64;
ਸਥਿਰ constexpr int64_t GGML_METAL_MMA_MANY_ਪੰਕਤੀਆਂ     = 6144;
ਸਥਿਰ constexpr ਪੂਰਨ     GGML_METAL_MMA_NSG_FEW_ਪੰਕਤੀਆਂ  = 32;
ਸਥਿਰ constexpr ਪੂਰਨ     GGML_METAL_MMA_NSG_MID_ਪੰਕਤੀਆਂ  = 16;
ਸਥਿਰ constexpr ਪੂਰਨ     GGML_METAL_MMA_NSG_MANY_ਪੰਕਤੀਆਂ = 8;

ਬਣਤਰ ggml_metal_mma_tiling {
    ਪੂਰਨ nsg; // simdgroups per threadgroup, each over a slice of K
    ਪੂਰਨ nt;  // 8-row src0 tiles per threadgroup
    ਪੂਰਨ rt;  // 8-row src1 tiles per threadgroup
};

// halves n while it is above limit
ਸਥਿਰ ਪੂਰਨ ggml_metal_halve_ਮੰਜ਼ਿਲ_ਹੱਦਬੰਦੀ(ਪੂਰਨ n, int64_t ਹੱਦਬੰਦੀ) {
    ਜਦੋਂ (n > 1 && n > ਹੱਦਬੰਦੀ) {
        n /= 2;
    }
    ਵਾਪਸ n;
}

ਸਥਿਰ size_t ggml_metal_mul_mv_mma_smem(ਪੂਰਨ nsg, ਪੂਰਨ nt, ਪੂਰਨ rt) {
    // one 8x8 float simdgroup matrix per output tile
    constexpr size_t tile_ਬਾਈਟਾਂ = 8*8*ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਵਾਪਸ (size_t) nsg*nt*rt*tile_ਬਾਈਟਾਂ;
}

// fewer src0 rows need more simdgroups per threadgroup (a finer K split) to fill the GPU.
// the tiles become narrower when the K-slice reduction buffer is too large or when too few threadgroups fill the GPU.
ਸਥਿਰ ggml_metal_mma_tiling ggml_metal_op_mul_mat_mma_tiling(ਅਚਲ ggml_tensor * op) {
    ਅਚਲ ggml_type type = op->src[0]->type;
    ਅਚਲ int64_t   ne00 = op->src[0]->ne[0];
    ਅਚਲ int64_t   ne01 = op->src[0]->ne[1];

    ggml_metal_mma_tiling res;
    res.rt = ggml_metal_mul_mv_mma_rt(op);

    ਅਚਲ int64_t n_ਕਦਮਾਂ = ne00/ggml_metal_mul_mv_mma_k_ਕਦਮ(type, res.rt);

    ਪੂਰਨ nsg = GGML_METAL_MMA_NSG_MANY_ਪੰਕਤੀਆਂ;
    ਜੇ (ne01 <= GGML_METAL_MMA_FEW_ਪੰਕਤੀਆਂ) {
        nsg = GGML_METAL_MMA_NSG_FEW_ਪੰਕਤੀਆਂ;
    } ਨਹੀਂਤਾਂ ਜੇ (ne01 <= GGML_METAL_MMA_MANY_ਪੰਕਤੀਆਂ) {
        nsg = GGML_METAL_MMA_NSG_MID_ਪੰਕਤੀਆਂ;
    }
    res.nsg = ggml_metal_halve_ਮੰਜ਼ਿਲ_ਹੱਦਬੰਦੀ(nsg, n_ਕਦਮਾਂ);

    ਅਚਲ int64_t nt_smem   = GGML_METAL_MMA_SMEM_MAX/(int64_t) ggml_metal_mul_mv_mma_smem(res.nsg, 1, res.rt);
    ਅਚਲ int64_t nt_ਪੰਕਤੀਆਂ   = ne01/(GGML_METAL_MIN_THREADGROUPS*GGML_METAL_MMA_TILE);
    res.nt = ggml_metal_halve_ਮੰਜ਼ਿਲ_ਹੱਦਬੰਦੀ(GGML_METAL_MMA_NT_MAX, std::min(nt_smem, nt_ਪੰਕਤੀਆਂ));

    ਵਾਪਸ res;
}

// the pipeline for the tiling, with fewer simdgroups when the device cannot run that many threads
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, ਤਾਰਕਿਕ ਜੋੜੋ_2) {
    ggml_metal_mma_tiling tiling = ggml_metal_op_mul_mat_mma_tiling(op);

    ਸਵੈਚਾਲਿਤ pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, ਜੋੜੋ_2);
    ਜਦੋਂ (tiling.nsg > 1 && ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(pipeline) < tiling.nsg*32) {
        tiling.nsg /= 2;
        pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, ਜੋੜੋ_2);
    }

    ਵਾਪਸ pipeline;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, ਪੂਰਨ nsg, ਪੂਰਨ nt, ਪੂਰਨ rt, ਤਾਰਕਿਕ ਜੋੜੋ_2) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ggml_type tsrc0 = op->src[0]->type;
    ਅਚਲ ggml_type tsrc1 = op->src[1]->type;
    ਅਚਲ ਪੂਰਨ       ne12  = op->src[1]->ne[2];
    ਅਚਲ ਪੂਰਨ       r2    = ne12 / op->src[0]->ne[2];
    ਅਚਲ ਪੂਰਨ       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    // the specialized kernels unroll over a compile-time row length
    ਅਚਲ ਪੂਰਨ ne00 = ggml_metal_mul_mv_mma_ਕਿਸਮ(tsrc0, rt) != GGML_METAL_MMA_ਕਿਸਮ_GEN ? op->src[0]->ne[0] : 0;

    snprintf(ਅਧਾਰ, 256, "kernel_mul_mv_mma_%s_%s_nt%d_rt%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), nt, rt);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_ne00=%d_add=%d", ਅਧਾਰ, nsg, ne12, r2, r3, ne00, ਜੋੜੋ_2);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int32(cv, ne00,           FC_MUL_MV_MMA + 4);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, nsg,            FC_MUL_MV_MMA + 0);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, (int16_t) ne12, FC_MUL_MV_MMA + 1);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, (int16_t) r2,   FC_MUL_MV_MMA + 2);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, (int16_t) r3,   FC_MUL_MV_MMA + 3);
        ggml_metal_cv_ਸੈੱਟ_bool (cv, ਜੋੜੋ_2,            FC_MUL_MV_MMA + 5);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = ggml_metal_mul_mv_mma_smem(nsg, nt, rt);

    res.nr0 = GGML_METAL_MMA_TILE*nt;
    res.nr1 = GGML_METAL_MMA_TILE*rt;

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ggml_type tsrc0 = op->src[0]->type;
    ਅਚਲ ggml_type tsrc1 = op->src[1]->type;

    ਅਚਲ ਤਾਰਕਿਕ bc_inp = op->src[0]->ne[0] % 32 != 0;

    constexpr ਪੂਰਨ NRA = SZ_SIMDGROUP * N_MM_ਖੰਡ_Y * N_MM_SIMD_ਸਮੂਹ_Y;
    constexpr ਪੂਰਨ NRB = SZ_SIMDGROUP * N_MM_ਖੰਡ_X * N_MM_SIMD_ਸਮੂਹ_X;

    ਅਚਲ ਤਾਰਕਿਕ has_tensor = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_tensor;

    ਅਚਲ ਤਾਰਕਿਕ bc_ਬਾਹਰਲਾ = has_tensor
        ? (op->ne[0] % NRA != 0 || op->ne[1] % NRB != 0)
        : (op->ne[0] % 64  != 0 || op->ne[1] % 32  != 0);

    GGML_ASSERT(op->src[1]->ne[2] <= INT16_MAX && op->src[1]->ne[3] <= INT16_MAX);
    ਅਚਲ int16_t ne12 = (int16_t) op->src[1]->ne[2];
    ਅਚਲ int16_t ne13 = (int16_t) op->src[1]->ne[3];
    ਅਚਲ int16_t r2   = (int16_t) (ne12 / op->src[0]->ne[2]);
    ਅਚਲ int16_t r3   = (int16_t) (ne13 / op->src[0]->ne[3]);

    snprintf(ਅਧਾਰ, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_bco=%d_ne12=%d_ne13=%d_r2=%d_r3=%d",
             ਅਧਾਰ, bc_inp, bc_ਬਾਹਰਲਾ, ne12, ne13, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, bc_ਬਾਹਰਲਾ, FC_MUL_MM + 1);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, ne12,  FC_MUL_MM + 2);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, ne13,  FC_MUL_MM + 3);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, r2,    FC_MUL_MM + 4);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, r3,    FC_MUL_MM + 5);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਜੇ (has_tensor) {
        res.nr0 = NRA;
        res.nr1 = NRB;

        ਅਚਲ size_t smem_a = NRA * N_MM_NK_ਕੁੱਲ * ਆਕਾਰ(ggml_fp16_t);
        res.smem = smem_a;
    } ਨਹੀਂਤਾਂ {
        res.nr0 = 64;
        res.nr1 = 32;

        res.smem = bc_ਬਾਹਰਲਾ ? 8192 : (4096 + 2048);
    }

    res.nsg = N_MM_SIMD_ਸਮੂਹ_X * N_MM_SIMD_ਸਮੂਹ_Y;

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਪੂਰਨ nsg = 0; // number of simdgroups
    ਪੂਰਨ nr0 = 0; // number of src0 rows per simdgroup
    ਪੂਰਨ nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    ਅਚਲ ggml_type tsrc0 = op->src[0]->type;
    ਅਚਲ ggml_type tsrc1 = op->src[1]->type;

    ਅਚਲ ਅੱਖਰ * suffix = "";

    ਤਾਰਕਿਕ ਵੰਡੋ = ਝੂਠ;

    // use custom matrix x vector kernel
    ਚੋਣ (tsrc0) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ਜੇ (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } ਨਹੀਂਤਾਂ {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ)*N_R0_Q8_0;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XXS_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XS_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;

                // split the rows across threads when there are fewer than 32 chunks per row
                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_XXS_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_S_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_S_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_S_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_M_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (ਪੂਰਨ) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    GGML_ASSERT(ne12 <= INT16_MAX && ne13 <= INT16_MAX);
    ਅਚਲ int16_t r2 = (int16_t) (ne12 / ne02);
    ਅਚਲ int16_t r3 = (int16_t) (ne13 / ne03);

    snprintf(ਅਧਾਰ, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_split=%d", ਅਧਾਰ, nsg, ne12, r2, r3, ਵੰਡੋ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, r2,             FC_MUL_MV + 3);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, r3,             FC_MUL_MV + 4);
        ggml_metal_cv_ਸੈੱਟ_bool (cv, ਵੰਡੋ,          FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_mul_mm_id_amax_part_f32");
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    res.smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_mul_mm_id_amax_f32");
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    res.smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, ਪੂਰਨ ne02, ਪੂਰਨ ne20) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s_ne02=%d", ਅਧਾਰ, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ਸੈੱਟ_int32(cv, ne02, FC_MUL_MM + 7);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    // per expert: entry count + base offset, followed by the staging of the ids of ntg tokens
    res.smem = 2*(size_t) ne02*ਆਕਾਰ(uint32_t) + (size_t) ne02*ne20*ਆਕਾਰ(uint16_t);
    res.smem = GGML_PAD(res.smem, 16);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ggml_type tsrc0 = op->src[0]->type;
    ਅਚਲ ggml_type tsrc1 = op->src[1]->type;

    ਅਚਲ ਤਾਰਕਿਕ bc_inp = op->src[0]->ne[0] % 32 != 0;

    // src1 prec [TAG_GGML_PREC]
    ਅਚਲ ਤਾਰਕਿਕ amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

    snprintf(ਅਧਾਰ, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_amax=%d", ਅਧਾਰ, bc_inp, amax);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, amax,   FC_MUL_MM + 6);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 8192;

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਪੂਰਨ nsg = 0; // number of simdgroups
    ਪੂਰਨ nr0 = 0; // number of src0 rows per simdgroup
    ਪੂਰਨ nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    ਅਚਲ ggml_type tsrc0 = op->src[0]->type;
    ਅਚਲ ggml_type tsrc1 = op->src[1]->type;

    ਅਚਲ ਅੱਖਰ * suffix = "";

    ਤਾਰਕਿਕ ਵੰਡੋ = ਝੂਠ;

        // use custom matrix x vector kernel
    ਚੋਣ (tsrc0) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ)*N_R0_Q8_0;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XXS_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XS_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;

                // split the rows across threads when there are fewer than 32 chunks per row
                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_XXS_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_S_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_S_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_S_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;

                ਅਚਲ ਪੂਰਨ nb32 = ne00/32;
                ਜੇ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_M_ਵੰਡੋ;
                    ਵੰਡੋ = ਸੱਚ;
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (ਪੂਰਨ)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(ਅਧਾਰ, 256, "kernel_mul_mv_id_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_split=%d", ਅਧਾਰ, nsg, ਵੰਡੋ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, 1,   FC_MUL_MV + 2);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, 1,   FC_MUL_MV + 3);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, 1,   FC_MUL_MV + 4);
        ggml_metal_cv_ਸੈੱਟ_bool (cv, ਵੰਡੋ, FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_ਮਾਪ(op->src[0]->type));

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    res.smem = 32*(ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(int32_t));

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part(ggml_metal_library_t lib) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, "kernel_argmax_part_f32");
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, "kernel_argmax_part_f32", "kernel_argmax_part_f32", ਖਾਲੀ_ਸੂਚਕ);
    }

    res.smem = 32*(ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(int32_t));

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final(ggml_metal_library_t lib) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, "kernel_argmax_final_f32");
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, "kernel_argmax_final_f32", "kernel_argmax_final_f32", ਖਾਲੀ_ਸੂਚਕ);
    }

    res.smem = 32*(ਆਕਾਰ(ਦਸ਼ਮਲਵ) + ਆਕਾਰ(int32_t));

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ARGSORT);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ = (ggml_ਛਾਂਟੋ_ਕ੍ਰਮ) op->op_params[0];

    ਅਚਲ ਅੱਖਰ * ਕ੍ਰਮ_str = "undefined";
    ਚੋਣ (ਕ੍ਰਮ) {
        ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC:  ਕ੍ਰਮ_str = "asc";  ਰੋਕੋ;
        ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC: ਕ੍ਰਮ_str = "desc"; ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    };

    snprintf(ਅਧਾਰ, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), ਕ੍ਰਮ_str);
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_ਮਿਲਾਓ(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ARGSORT);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ = (ggml_ਛਾਂਟੋ_ਕ੍ਰਮ) op->op_params[0];

    ਅਚਲ ਅੱਖਰ * ਕ੍ਰਮ_str = "undefined";
    ਚੋਣ (ਕ੍ਰਮ) {
        ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC:  ਕ੍ਰਮ_str = "asc";  ਰੋਕੋ;
        ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC: ਕ੍ਰਮ_str = "desc"; ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    };

    snprintf(ਅਧਾਰ, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), ਕ੍ਰਮ_str);
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht(ggml_metal_library_t lib, ਪੂਰਨ n, ggml_type tsrc) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_fwht_%s_%d", ggml_type_name(tsrc), n);
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

// note: reuse the argsort kernel for the bitonic top_k fallback
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਸਿਖਰ_k(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ਸਿਖਰ_K);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    // note: the top_k kernel is always descending order
    ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ = GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC;

    ਅਚਲ ਅੱਖਰ * ਕ੍ਰਮ_str = "undefined";
    ਚੋਣ (ਕ੍ਰਮ) {
        ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC:  ਕ੍ਰਮ_str = "asc";  ਰੋਕੋ;
        ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC: ਕ੍ਰਮ_str = "desc"; ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    };

    snprintf(ਅਧਾਰ, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), ਕ੍ਰਮ_str);
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਸਿਖਰ_k_radix(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ਸਿਖਰ_K);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_top_k_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਸਿਖਰ_k_ਮਿਲਾਓ(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ਸਿਖਰ_K);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ = GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC;

    ਅਚਲ ਅੱਖਰ * ਕ੍ਰਮ_str = "undefined";
    ਚੋਣ (ਕ੍ਰਮ) {
        ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC:  ਕ੍ਰਮ_str = "asc";  ਰੋਕੋ;
        ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC: ਕ੍ਰਮ_str = "desc"; ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    };

    snprintf(ਅਧਾਰ, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), ਕ੍ਰਮ_str);
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe(
        ggml_metal_library_t lib, int32_t n_expert, int32_t ਸਿਖਰ_k, ਤਾਰਕਿਕ with_norm) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_topk_moe_f32");
    snprintf(name, 256, "%s_n_expert=%d_top_k=%d_with_norm=%d", ਅਧਾਰ, n_expert, ਸਿਖਰ_k, with_norm ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ਸੈੱਟ_bool (cv, with_norm, FC_TOPK_MOE + 0);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, n_expert,  FC_TOPK_MOE + 1);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, ਸਿਖਰ_k,     FC_TOPK_MOE + 2);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce(ggml_metal_library_t lib, int32_t n_expert_ਵਰਤਿਆ) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_moe_reduce_f32");
    snprintf(name, 256, "%s_n_expert_used=%d", ਅਧਾਰ, n_expert_ਵਰਤਿਆ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ਸੈੱਟ_int32(cv, n_expert_ਵਰਤਿਆ, FC_MOE_REDUCE + 0);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ਅਚਲ ਬਣਤਰ ggml_tensor * op,
        ਤਾਰਕਿਕ    has_ਨਕਾਬ,
        int32_t ncpsg) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ਅਣਵਰਤਿਆ(op);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_%s",
            "flash_attn_ext_pad");

    snprintf(name, 256, "%s_mask=%d_ncpsg=%d",
            ਅਧਾਰ,
            has_ਨਕਾਬ,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_ਨਕਾਬ,  FC_FLASH_ATTN_EXT_PAD + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_PAD + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_PAD + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_PAD + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_PAD + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_PAD + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_PAD + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_PAD + 23);
        //ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_PAD + 24);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ਅੱਖਰ ਅਧਾਰ[256];

    snprintf(ਅਧਾਰ, 256, "kernel_flash_attn_ext_kv_%s_f16", ggml_type_name(op->src[1]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, ਅਧਾਰ);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, ਅਧਾਰ, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ਅਚਲ ਬਣਤਰ ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ਅਣਵਰਤਿਆ(op);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_%s",
            "flash_attn_ext_blk");

    snprintf(name, 256, "%s_nqptg=%d_ncpsg=%d",
            ਅਧਾਰ,
            nqptg,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        //ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_BLK + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_BLK + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_BLK + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_BLK + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_BLK + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_BLK + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_BLK + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_BLK + 23);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ਅਚਲ ggml_tensor * op,
        ਤਾਰਕਿਕ has_ਨਕਾਬ,
        ਤਾਰਕਿਕ has_sinks,
        ਤਾਰਕਿਕ has_bias,
        ਤਾਰਕਿਕ has_scap) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ int32_t dk = (int32_t) op->src[1]->ne[0];
    ਅਚਲ int32_t dv = (int32_t) op->src[2]->ne[0];

    ਅਚਲ ਅੱਖਰ * type = op->src[1]->type == GGML_TYPE_BF16 ? "bf16" : "f16";

    snprintf(ਅਧਾਰ, 256, "kernel_flash_attn_ext_tensor_%s_dk%d_dv%d", type, dk, dv);
    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d", ਅਧਾਰ, has_ਨਕਾਬ, has_sinks, has_bias, has_scap);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_ਨਕਾਬ,  FC_FLASH_ATTN_EXT_TENSOR + 0);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_TENSOR + 1);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_TENSOR + 2);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_TENSOR + 3);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ਅਚਲ ggml_tensor * op,
        ਤਾਰਕਿਕ    has_ਨਕਾਬ,
        ਤਾਰਕਿਕ    has_sinks,
        ਤਾਰਕਿਕ    has_bias,
        ਤਾਰਕਿਕ    has_scap,
        ਤਾਰਕਿਕ    has_kvpad,
        int32_t nsg,
        ਤਾਰਕਿਕ    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ int32_t dk = (int32_t) op->src[1]->ne[0];
    ਅਚਲ int32_t dv = (int32_t) op->src[2]->ne[0];

    ਅਚਲ ਅੱਖਰ * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    // do bounds checks for the mask?
    ਅਚਲ ਤਾਰਕਿਕ bc_ਨਕਾਬ = op->src[3] && (op->src[3]->ne[1] % 8 != 0);

    snprintf(ਅਧਾਰ, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            type,
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
            ਅਧਾਰ,
            has_ਨਕਾਬ,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            bc_ਨਕਾਬ,
            ns10,
            ns20,
            nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_ਨਕਾਬ,  FC_FLASH_ATTN_EXT + 0);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);

        ggml_metal_cv_ਸੈੱਟ_bool(cv, bc_ਨਕਾਬ, FC_FLASH_ATTN_EXT + 10);

        ggml_metal_cv_ਸੈੱਟ_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);
    ਪ੍ਰਤਿਜ੍ਞਾ(op->src[3]);

    ਅੱਖਰ name[256];

    snprintf(name, 256, "kernel_flash_attn_ext_vec_idx");

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    GGML_ਅਣਵਰਤਿਆ(op);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ਅਚਲ ggml_tensor * op,
        ਤਾਰਕਿਕ    has_ਨਕਾਬ,
        ਤਾਰਕਿਕ    has_sinks,
        ਤਾਰਕਿਕ    has_bias,
        ਤਾਰਕਿਕ    has_scap,
        ਤਾਰਕਿਕ    has_kvpad,
        ਤਾਰਕਿਕ    has_sparse,
        int32_t nqpsg,
        int32_t ne,
        int32_t nsg,
        int32_t nwg,
        ਤਾਰਕਿਕ    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ int32_t dk = (int32_t) op->src[1]->ne[0];
    ਅਚਲ int32_t dv = (int32_t) op->src[2]->ne[0];

    ਅਚਲ ਅੱਖਰ * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    ਅੱਖਰ qne_suffix[16] = {0};
    ਜੇ (!(nqpsg == 1 && ne == ggml_metal_tuning::fa_vec_baseline_ne(dk, dv))) {
        snprintf(qne_suffix, ਆਕਾਰ(qne_suffix), "_q%d_ne%d", nqpsg, ne);
    }

    snprintf(ਅਧਾਰ, 256, "kernel_%s_%s_dk%d_dv%d%s",
            "flash_attn_ext_vec",
            type,
            dk,
            dv,
            qne_suffix);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_sparse=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            ਅਧਾਰ,
            has_ਨਕਾਬ,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            has_sparse,
            ns10,
            ns20,
            nsg, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_ਨਕਾਬ,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_kvpad,  FC_FLASH_ATTN_EXT_VEC + 4);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, has_sparse, FC_FLASH_ATTN_EXT_VEC + 5);

        ggml_metal_cv_ਸੈੱਟ_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ਅਚਲ ggml_tensor * op,
        int32_t dv,
        int32_t nwg) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "%s_dv=%d_nwg=%d", ਅਧਾਰ, dv, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        ggml_metal_cv_ਸੈੱਟ_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;

    GGML_ਅਣਵਰਤਿਆ(op);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, int32_t n_fuse) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਪੂਰਨ op_ਅੰਕ = -1;

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_ਜੋੜੋ: op_ਅੰਕ = 0; ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SUB: op_ਅੰਕ = 1; ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL: op_ਅੰਕ = 2; ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIV: op_ਅੰਕ = 3; ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    };

    ਅਚਲ ਅੱਖਰ * t0_str = ggml_type_name(op->src[0]->type);
    ਅਚਲ ਅੱਖਰ * t1_str = ggml_type_name(op->src[1]->type);
    ਅਚਲ ਅੱਖਰ * t_str  = ggml_type_name(op->type);

    ਅਚਲ ਤਾਰਕਿਕ is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    ਅਚਲ ਤਾਰਕਿਕ is_cb = op->src[0]->ne[0] != op->src[1]->ne[0];
    ਅਚਲ ਤਾਰਕਿਕ is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(ਅਧਾਰ, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d_cb=%d", ਅਧਾਰ, op_ਅੰਕ, n_fuse, is_rb, is_cb);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, op_ਅੰਕ, FC_BIN + 0);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, n_fuse, FC_BIN + 1);
        ggml_metal_cv_ਸੈੱਟ_bool (cv, is_rb,  FC_BIN + 2);
        ggml_metal_cv_ਸੈੱਟ_bool (cv, is_cb,  FC_BIN + 3);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_rb;

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one(ggml_metal_library_t lib, ggml_op op) {
    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਪੂਰਨ op_ਅੰਕ = -1;

    ਚੋਣ (op) {
        ਹਾਲਤ GGML_OP_ਜੋੜੋ: op_ਅੰਕ = 0; ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SUB: op_ਅੰਕ = 1; ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL: op_ਅੰਕ = 2; ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIV: op_ਅੰਕ = 3; ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    };

    snprintf(ਅਧਾਰ, 256, "kernel_bin_fuse_%s_%s_%s", "f32", "f32", "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", ਅਧਾਰ, op_ਅੰਕ, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, op_ਅੰਕ, FC_BIN + 0);
        ggml_metal_cv_ਸੈੱਟ_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_ਸੈੱਟ_bool (cv, ਝੂਠ,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_L2_NORM);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਤਾਰਕਿਕ is_c4 = op->src[0]->ne[0] % 4 == 0;

    ਅਚਲ ਅੱਖਰ * t0_str = ggml_type_name(op->src[0]->type);
    ਅਚਲ ਅੱਖਰ * t_str  = ggml_type_name(op->type);

    snprintf(ਅਧਾਰ, 256, "kernel_l2_norm_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    res.c4   = is_c4;
    res.smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਸਮੂਹ_norm(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ਸਮੂਹ_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    res.smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, ਪੂਰਨ n_fuse) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਅੱਖਰ * suffix = "";
    ਜੇ (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_NORM:
            ਚੋਣ (n_fuse) {
                ਹਾਲਤ 1: snprintf(ਅਧਾਰ, 256, "kernel_norm_f32%s", suffix);         ਰੋਕੋ;
                ਹਾਲਤ 2: snprintf(ਅਧਾਰ, 256, "kernel_norm_mul_f32%s", suffix);     ਰੋਕੋ;
                ਹਾਲਤ 3: snprintf(ਅਧਾਰ, 256, "kernel_norm_mul_add_f32%s", suffix); ਰੋਕੋ;
                ਮੂਲ: GGML_ABORT("fatal error");
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM:
            ਚੋਣ (n_fuse) {
                ਹਾਲਤ 1: snprintf(ਅਧਾਰ, 256, "kernel_rms_norm_f32%s", suffix);         ਰੋਕੋ;
                ਹਾਲਤ 2: snprintf(ਅਧਾਰ, 256, "kernel_rms_norm_mul_f32%s", suffix);     ਰੋਕੋ;
                ਹਾਲਤ 3: snprintf(ਅਧਾਰ, 256, "kernel_rms_norm_mul_add_f32%s", suffix); ਰੋਕੋ;
                ਮੂਲ: GGML_ABORT("fatal error");
            } ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ਸੈੱਟ_bool(cv, ਝੂਠ, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]));

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਅੱਖਰ * suffix = "";
    ਜੇ (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_NORM:     snprintf(ਅਧਾਰ, 256, "kernel_norm_mul_f32%s", suffix);     ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM: snprintf(ਅਧਾਰ, 256, "kernel_rms_norm_mul_f32%s", suffix); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s_use_scale", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ਸੈੱਟ_bool(cv, ਸੱਚ, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ROPE || op->op == GGML_OP_ROPE_ਪਿੱਠ);

    ਅਚਲ ਤਾਰਕਿਕ is_ਪਿੱਠ = op->op == GGML_OP_ROPE_ਪਿੱਠ;

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਪੂਰਨ mode = ((ਅਚਲ int32_t *) op->op_params)[2];

    ਅਚਲ ਤਾਰਕਿਕ is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    ਅਚਲ ਤਾਰਕਿਕ is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    ਅਚਲ ਤਾਰਕਿਕ is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ਅਚਲ ਤਾਰਕਿਕ is_vision = mode == GGML_ROPE_TYPE_VISION;

    ਜੇ (is_neox) {
        snprintf(ਅਧਾਰ, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } ਨਹੀਂਤਾਂ ਜੇ ((is_mrope || is_imrope) && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(ਅਧਾਰ, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } ਨਹੀਂਤਾਂ ਜੇ (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(ਅਧਾਰ, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } ਨਹੀਂਤਾਂ {
        snprintf(ਅਧਾਰ, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s_imrope=%d_is_back=%d", ਅਧਾਰ, is_imrope ? 1 : 0, is_ਪਿੱਠ ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_bool(cv, is_imrope, FC_ROPE + 0);
        ggml_metal_cv_ਸੈੱਟ_bool(cv, is_ਪਿੱਠ,   FC_ROPE + 1);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_IM2_ਖਾਨਾ);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    ਅਚਲ ਤਾਰਕਿਕ is_2D = ((ਅਚਲ int32_t *)(op->op_params))[6] == 1;
    ਅਚਲ int64_t KH = is_2D ? ne01 : 1;
    ਅਚਲ int64_t KW = ne00;

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਜੇ (KH*KW <= 1024) {
        snprintf(ਅਧਾਰ, 256, "kernel_im2col_%s", ggml_type_name(op->type));
    } ਨਹੀਂਤਾਂ {
        snprintf(ਅਧਾਰ, 256, "kernel_im2col_ext_%s", ggml_type_name(op->type));
    }
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_COL2IM_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_col2im_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake(ggml_metal_library_t lib, ਗਿਣਤੀ ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_snake_%s", ggml_type_name(type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_conv_transpose_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_CONV_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_conv_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op, ਤਾਰਕਿਕ tiled) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_CONV_2D_DW);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_conv_2d_dw%s_%s_%s",
             tiled ? "_tiled" : "",
             ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_CONV_3D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_conv_3d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_UPSCALE);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ int32_t ਬਹੁਲਕ_flags = ggml_get_op_params_i32(op, 0);
    ਅਚਲ ggml_scale_ਬਹੁਲਕ mode = (ggml_scale_ਬਹੁਲਕ) (ਬਹੁਲਕ_flags & 0xFF);

    ਅਚਲ ਤਾਰਕਿਕ antialias = (ਬਹੁਲਕ_flags & GGML_SCALE_FLAG_ANTIALIAS);

    ਜੇ (mode == GGML_SCALE_ਬਹੁਲਕ_BILINEAR) {
        snprintf(ਅਧਾਰ, 256, "kernel_upscale_bilinear_%s", ggml_type_name(op->src[0]->type));
    } ਨਹੀਂਤਾਂ ਜੇ (mode == GGML_SCALE_ਬਹੁਲਕ_BICUBIC) {
        snprintf(ਅਧਾਰ, 256, "kernel_upscale_bicubic_%s", ggml_type_name(op->src[0]->type));
    } ਨਹੀਂਤਾਂ {
        snprintf(ਅਧਾਰ, 256, "kernel_upscale_nearest_%s", ggml_type_name(op->src[0]->type));
    }
    snprintf(name, 256, "%s_aa=%d", ਅਧਾਰ, antialias);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_bool(cv, antialias, FC_UPSCALE + 0);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ROLL);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_roll_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_PAD);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਅਚਲ ਤਾਰਕਿਕ circular = ggml_get_op_params_i32(op, 8) != 0;

    snprintf(ਅਧਾਰ, 256, "kernel_pad_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_circular=%d", ਅਧਾਰ, circular);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_bool(cv, circular, FC_PAD + 0);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_PAD_REFLECT_1D);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ARANGE);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_ਕਦਮ_adamw(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_OPT_ਕਦਮ_ADAMW);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_ਕਦਮ_sgd(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_OPT_ਕਦਮ_SGD);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_ਪਿੱਠ(ggml_metal_library_t lib, ਅਚਲ ggml_tensor * op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_SILU_ਪਿੱਠ);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_silu_back_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, ਅਚਲ ggml_tensor *  op) {
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    snprintf(ਅਧਾਰ, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", ਅਧਾਰ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, ਖਾਲੀ_ਸੂਚਕ);
    }

    ਵਾਪਸ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਗਿਣਤੀ_equal(ggml_metal_library_t lib, ਅਚਲ ggml_tensor *  op) {
    ਪ੍ਰਤਿਜ੍ਞਾ(op->op == GGML_OP_ਗਿਣਤੀ_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    ਅੱਖਰ ਅਧਾਰ[256];
    ਅੱਖਰ name[256];

    ਪੂਰਨ nsg = 1;
    ਜਦੋਂ (32*nsg < ne00 && nsg < 32) {
        nsg *= 2;
    }

    snprintf(ਅਧਾਰ, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", ਅਧਾਰ, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ਜੇ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ਸੈੱਟ_int16(cv, nsg, FC_ਗਿਣਤੀ_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, ਅਧਾਰ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32 * ਆਕਾਰ(int32_t);
    res.nsg  = nsg;

    ਵਾਪਸ res;
}
