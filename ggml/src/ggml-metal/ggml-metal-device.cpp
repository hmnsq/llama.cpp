#ಅನ್ತರ್ಭಾವ "ggml-metal-device.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-common.h"

#ಅನ್ತರ್ಭಾವ "ggml-metal-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-tuning.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <unordered_map>

ರಚನೆ ggml_metal_device_deleter {
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

ggml_metal_device_t ggml_metal_device_get(ಪೂರ್ಣಾಂಕ device, ಪೂರ್ಣಾಂಕ n_devices) {
    ಸ್ಥಿರ std::vector<ggml_metal_device_ptr> devs;

    devs.emplace_back(ggml_metal_device_init(device, n_devices));

    ಹಿಂತಿರುಗಿ devs.back().get();
}

ರಚನೆ ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> ದತ್ತಾಂಶ;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(ಶೂನ್ಯ) {
    ggml_metal_pipelines_t res = ಹೊಸ ggml_metal_pipelines();

    ಹಿಂತಿರುಗಿ res;
}

ಶೂನ್ಯ ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    ಆದರೆ (!ppls) {
        ಹಿಂತಿರುಗಿ;
    }

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ it = ppls->ದತ್ತಾಂಶ.begin(); it != ppls->ದತ್ತಾಂಶ.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    ಅಳಿಸು ppls;
}

ಶೂನ್ಯ ggml_metal_pipelines_ಕೂಡಿಸು(ggml_metal_pipelines_t ppls, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name, ggml_metal_pipeline_t pipeline) {
    ppls->ದತ್ತಾಂಶ[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಆದರೆ (ppls->ದತ್ತಾಂಶ.find(name) == ppls->ದತ್ತಾಂಶ.end()) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಹಿಂತಿರುಗಿ ppls->ದತ್ತಾಂಶ[name];
}

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ತಳ(ggml_metal_library_t lib, ggml_op op) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * op_str = "undefined";
    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID: op_str = "add_id"; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    };

    snprintf(ತಳ, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * pool_str = "undefined";
    ಆಯ್ಕೆ (op_pool) {
        ಸ್ಥಿತಿ GGML_OP_POOL_AVG: pool_str = "avg"; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_MAX: pool_str = "max"; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ASSERT(ಸುಳ್ಳು && "not implemented");
    };

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, ಗಾತ್ರ(ತಳ), "kernel_pool_1d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, ಗಾತ್ರ(name), "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * pool_str = "undefined";
    ಆಯ್ಕೆ (op_pool) {
        ಸ್ಥಿತಿ GGML_OP_POOL_AVG: pool_str = "avg"; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_MAX: pool_str = "max"; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ASSERT(ಸುಳ್ಳು && "not implemented");
    };

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_ಅಡ್ಡಸಾಲುಗಳು(ggml_metal_library_t lib, ggml_type tsrc) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ tsrc = op->src[0]->type;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ tidx = op->src[1]->type;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ tdst = op->type;

    snprintf(ತಳ, 256, "kernel_set_rows_%s_%s_%s", ggml_type_name(tsrc), ggml_type_name(tidx), ggml_type_name(tdst));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n = op->src[0]->ne[0];

    snprintf(ತಳ, 256, "kernel_diag_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_n=%d", ತಳ, n);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    res.nsg  = 1;
    res.smem = 0;

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat(ggml_metal_library_t lib, ggml_type tsrc) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat(ggml_metal_library_t lib, ggml_type tsrc) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_concat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಪೂರ್ಣಾಂಕ op_ಅಂಕೆ = -1;

    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_SCALE:      op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_SCALE;      ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FILL:       op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_FILL;       ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CLAMP:      op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_CLAMP;      ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQR:        op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_SQR;        ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQRT:       op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_SQRT;       ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SIN:        op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_SIN;        ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_COS:        op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_COS;        ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LOG:        op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_LOG;        ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU: op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_LEAKY_RELU; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            ಆಯ್ಕೆ (ggml_get_unary_op(op)) {
                ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:        op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_TANH;        ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:        op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_RELU;        ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:     op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_SIGMOID;     ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:        op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_GELU;        ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:    op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_GELU_ERF;    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:  op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_GELU_ಶೀಘ್ರ;  ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:        op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_SILU;        ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:         op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_ELU;         ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:         op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_NEG;         ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:         op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_ABS;         ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:         op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_SGN;         ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:        op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_ಹೆಜ್ಜೆ;        ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:   op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_HARDSWISH;   ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID: op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_HARDSIGMOID; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:         op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_EXP;         ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:    op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_SOFTPLUS;    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:       op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_EXPM1;       ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:       op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_FLOOR;       ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:        op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_CEIL;        ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:       op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_ಸುತ್ತು;       ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:       op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_TRUNC;       ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:       op_ಅಂಕೆ = OP_UNARY_ಅಂಕೆ_XIELU;       ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    };

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * t0_str = ggml_type_name(op->src[0]->type);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * t_str  = ggml_type_name(op->type);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_c4 = op->src[0]->ne[0] % 4 == 0;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_cnt = ggml_is_contiguous(op->src[0]) && ggml_nelements(op) < 32768;

    snprintf(ತಳ, 256, "kernel_unary_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_cnt=%d", ತಳ, op_ಅಂಕೆ, is_cnt);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, op_ಅಂಕೆ, FC_UNARY + 0);
        ggml_metal_cv_ಗಣ_bool (cv, is_cnt, FC_UNARY + 1);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_cnt;

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * op_str = "undefined";
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಆಯ್ಕೆ (ggml_get_glu_op(op)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:        op_str = "reglu";        ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:        op_str = "geglu";        ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:  op_str = "geglu_quick";  ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP: op_str = "swiglu_clamp"; ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    };

    snprintf(ತಳ, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ಮೊತ್ತ(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ಮೊತ್ತ);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_op_sum_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op->src[0]));

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಪೂರ್ಣಾಂಕ op_ಅಂಕೆ = -1;

    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು: op_ಅಂಕೆ = OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ಅಂಕೆ_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:     op_ಅಂಕೆ = OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ಅಂಕೆ_ಸರಾಸರಿ;     ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    };

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * t0_str = ggml_type_name(op->src[0]->type);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * t_str  = ggml_type_name(op->type);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_c4 = op->src[0]->ne[0] % 4 == 0;

    snprintf(ತಳ, 256, "kernel_sum_rows_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d", ತಳ, op_ಅಂಕೆ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, op_ಅಂಕೆ, FC_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು + 0);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಆದರೆ (is_c4) {
        res.smem *= 4;
    }

    res.c4  = is_c4;

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_cumsum_blk_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_ಕೂಡಿಸು(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_cumsum_add_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_TRI);
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[0]->type));

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * op_str = "tri";
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ttype = op->op_params[0];

    snprintf(ತಳ, 256, "kernel_%s_%s_%d", op_str, ggml_type_name(op->src[0]->type), ttype);

    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * suffix = "";

    ಆದರೆ (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    ಸ್ಥಿರಾಂಕ ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(ತಳ, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    res.smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ int16_t nh = op->src[0]->ne[1];

    snprintf(ತಳ, 256, "kernel_lightning_indexer_%s", ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s_nh=%d", ತಳ, nh);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, nh, FC_LIGHTNING_INDEXER + 0);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಅಕ್ಷರ name[256];
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ = ಶೂನ್ಯಸೂಚಕ;

    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_COMB:
            ತಳ = "kernel_dsv4_hc_comb_f32";
            snprintf(name, 256, "%s", ತಳ);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_PRE:
            ತಳ = ggml_get_op_params_i32(op, 1) != 0 ? "kernel_dsv4_hc_pre_gated_f32" : "kernel_dsv4_hc_pre_f32";
            snprintf(name, 256, "%s_n_hc=%d", ತಳ, (ಪೂರ್ಣಾಂಕ) op->src[0]->ne[1]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_POST:
            ತಳ = op->src[3] ? "kernel_dsv4_hc_post_f32" : "kernel_dsv4_hc_post_nocomb_f32";
            snprintf(name, 256, "%s", ತಳ);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fatal error");
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ಶೂನ್ಯಸೂಚಕ;

        ಆದರೆ (op->op == GGML_OP_DSV4_HC_PRE) {
            cv = ggml_metal_cv_init();
            ggml_metal_cv_ಗಣ_int32(cv, (int32_t) op->src[0]->ne[1], FC_DSV4_HC + 0);
        }

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ಆದರೆ (cv) {
            ggml_metal_cv_free(cv);
        }
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv(
        ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, int32_t nc, ತಾರ್ಕಿಕ use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * suffix = "";

    ಆದರೆ (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(ತಳ, 256, "kernel_ssm_conv_%s_%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_nc=%d_silu=%d", ತಳ, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ಗಣ_bool(cv, use_silu, FC_SSM_CONV + 1);
        ggml_metal_cv_ಗಣ_int32(cv, nc,      FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched(
        ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, ಪೂರ್ಣಾಂಕ ssm_conv_bs, int32_t nc, ತಾರ್ಕಿಕ use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * suffix = "";
    ಆದರೆ (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(ತಳ, 256, "kernel_ssm_conv_%s_%s_batched%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_ssm_conv_bs=%d_nc=%d_silu=%d", ತಳ, ssm_conv_bs, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, ssm_conv_bs, FC_SSM_CONV + 0);
        ggml_metal_cv_ಗಣ_bool(cv, use_silu,     FC_SSM_CONV + 1);
        ggml_metal_cv_ಗಣ_int32(cv, nc,          FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_ಪರಿಶೀಲನೆ(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, ತಾರ್ಕಿಕ ಬಾಲ)  {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nsg = (ne00 + 31)/32;

    snprintf(ತಳ, 256, "kernel_ssm_scan_%s%s", ggml_type_name(op->src[0]->type), ಬಾಲ ? "_tail" : "");
    snprintf(name, 256, "%s_nsg=%d", ತಳ, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (32 + 2) floats
    res.smem = GGML_PAD((32 + 2)*ಗಾತ್ರ(ದಶಮಾಂಶ)*nsg, 16);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_ಪರಿಶೀಲನೆ_ssd_mma(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op)  {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_ssm_scan_ssd_mma_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    // acs/exp(acs)/state-decay vectors + dtX + SAM rows + two 8x8 tiles per simdgroup
    res.smem = (3*OP_SSM_ಪರಿಶೀಲನೆ_SSD_CS +
                OP_SSM_ಪರಿಶೀಲನೆ_SSD_CS*OP_SSM_ಪರಿಶೀಲನೆ_SSD_HD +
                OP_SSM_ಪರಿಶೀಲನೆ_SSD_NSG*8*OP_SSM_ಪರಿಶೀಲನೆ_SSD_CS +
                OP_SSM_ಪರಿಶೀಲನೆ_SSD_NSG*2*8*8)*ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ int64_t C = op->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H = op->src[0]->ne[1];

    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(ತಳ, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(ತಳ, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    // v is src[2], dimensions: S_v = ne[0], H = ne[1]
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne20 = op->src[2]->ne[0]; // S_v
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne21 = op->src[2]->ne[1]; // H
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne30 = op->src[3]->ne[0]; // G
    // state is src[5], 4D [S_v, S_v, H_v, n_seqs] (s0 only); K is op param 0.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K = ggml_get_op_params_i32(op, 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nsg = op->src[2]->ne[0]/32;

    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->ne[0] == ne20 * ne21);
    GGML_ASSERT(ne20 % 32 == 0);

    snprintf(ತಳ, 256, "kernel_gated_delta_net_%s_%d", ggml_type_name(op->src[0]->type), nsg);
    snprintf(name, 256, "%s_ne20=%d_ne30=%d_K=%d", ತಳ, ne20, ne30, K);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, ne20, FC_GATED_DELTA_NET + 0);
        ggml_metal_cv_ಗಣ_int16(cv, ne30, FC_GATED_DELTA_NET + 1);
        ggml_metal_cv_ಗಣ_int16(cv, K,    FC_GATED_DELTA_NET + 2);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg = nsg;

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nsg = 8;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n   = op->src[1]->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k   = op->src[1]->ne[0];

    snprintf(ತಳ, 256, "kernel_solve_tri_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d_n=%d_k=%d", ತಳ, nsg, n, k);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, nsg, FC_SOLVE_TRI + 0);
        ggml_metal_cv_ಗಣ_int16(cv, n,   FC_SOLVE_TRI + 1);
        ggml_metal_cv_ಗಣ_int16(cv, k,   FC_SOLVE_TRI + 2);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = GGML_PAD(GGML_PAD(n, 32)*nsg*ಗಾತ್ರ(ದಶಮಾಂಶ), 16);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, ಪೂರ್ಣಾಂಕ nsg, ಪೂರ್ಣಾಂಕ nxpsg, ಪೂರ್ಣಾಂಕ r1ptg) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ggml_type tsrc0 = op->src[0]->type;
    ಸ್ಥಿರಾಂಕ ggml_type tsrc1 = op->src[1]->type;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne12  = op->src[1]->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       r2    = ne12 / op->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    snprintf(ತಳ, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d_ne12=%d_r2=%d_r3=%d", ತಳ, nsg, nxpsg, ne12, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_ಗಣ_int16(cv, nxpsg,          FC_MUL_MV + 1);
        ggml_metal_cv_ಗಣ_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_ಗಣ_int16(cv, (int16_t) r2,   FC_MUL_MV + 3);
        ggml_metal_cv_ಗಣ_int16(cv, (int16_t) r3,   FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

// threadgroups needed to fill the GPU
// TODO: dedup
ಸ್ಥಿರ constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

// few-row MMA mat-mul: 8x8 simdgroup matrices for the src1 rows of ggml_metal_mul_mat_use_mma
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ     GGML_METAL_MMA_TILE          = 8;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ     GGML_METAL_MMA_NT_MAX        = 4;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ     GGML_METAL_MMA_SMEM_MAX      = 16384;

// simdgroups per threadgroup for at most FEW_ROWS, at most MANY_ROWS, and more src0 rows
ಸ್ಥಿರ constexpr int64_t GGML_METAL_MMA_FEW_ಅಡ್ಡಸಾಲುಗಳು      = 64;
ಸ್ಥಿರ constexpr int64_t GGML_METAL_MMA_MANY_ಅಡ್ಡಸಾಲುಗಳು     = 6144;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ     GGML_METAL_MMA_NSG_FEW_ಅಡ್ಡಸಾಲುಗಳು  = 32;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ     GGML_METAL_MMA_NSG_MID_ಅಡ್ಡಸಾಲುಗಳು  = 16;
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ     GGML_METAL_MMA_NSG_MANY_ಅಡ್ಡಸಾಲುಗಳು = 8;

ರಚನೆ ggml_metal_mma_tiling {
    ಪೂರ್ಣಾಂಕ nsg; // simdgroups per threadgroup, each over a slice of K
    ಪೂರ್ಣಾಂಕ nt;  // 8-row src0 tiles per threadgroup
    ಪೂರ್ಣಾಂಕ rt;  // 8-row src1 tiles per threadgroup
};

// halves n while it is above limit
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_metal_halve_ಗಮ್ಯ_ಎಲ್ಲೆ(ಪೂರ್ಣಾಂಕ n, int64_t ಎಲ್ಲೆ) {
    ಆಗ (n > 1 && n > ಎಲ್ಲೆ) {
        n /= 2;
    }
    ಹಿಂತಿರುಗಿ n;
}

ಸ್ಥಿರ size_t ggml_metal_mul_mv_mma_smem(ಪೂರ್ಣಾಂಕ nsg, ಪೂರ್ಣಾಂಕ nt, ಪೂರ್ಣಾಂಕ rt) {
    // one 8x8 float simdgroup matrix per output tile
    constexpr size_t tile_ಬೈಟುಗಳು = 8*8*ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಹಿಂತಿರುಗಿ (size_t) nsg*nt*rt*tile_ಬೈಟುಗಳು;
}

// fewer src0 rows need more simdgroups per threadgroup (a finer K split) to fill the GPU.
// the tiles become narrower when the K-slice reduction buffer is too large or when too few threadgroups fill the GPU.
ಸ್ಥಿರ ggml_metal_mma_tiling ggml_metal_op_mul_mat_mma_tiling(ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_type type = op->src[0]->type;
    ಸ್ಥಿರಾಂಕ int64_t   ne00 = op->src[0]->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t   ne01 = op->src[0]->ne[1];

    ggml_metal_mma_tiling res;
    res.rt = ggml_metal_mul_mv_mma_rt(op);

    ಸ್ಥಿರಾಂಕ int64_t n_ಹೆಜ್ಜೆಗಳು = ne00/ggml_metal_mul_mv_mma_k_ಹೆಜ್ಜೆ(type, res.rt);

    ಪೂರ್ಣಾಂಕ nsg = GGML_METAL_MMA_NSG_MANY_ಅಡ್ಡಸಾಲುಗಳು;
    ಆದರೆ (ne01 <= GGML_METAL_MMA_FEW_ಅಡ್ಡಸಾಲುಗಳು) {
        nsg = GGML_METAL_MMA_NSG_FEW_ಅಡ್ಡಸಾಲುಗಳು;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ne01 <= GGML_METAL_MMA_MANY_ಅಡ್ಡಸಾಲುಗಳು) {
        nsg = GGML_METAL_MMA_NSG_MID_ಅಡ್ಡಸಾಲುಗಳು;
    }
    res.nsg = ggml_metal_halve_ಗಮ್ಯ_ಎಲ್ಲೆ(nsg, n_ಹೆಜ್ಜೆಗಳು);

    ಸ್ಥಿರಾಂಕ int64_t nt_smem   = GGML_METAL_MMA_SMEM_MAX/(int64_t) ggml_metal_mul_mv_mma_smem(res.nsg, 1, res.rt);
    ಸ್ಥಿರಾಂಕ int64_t nt_ಅಡ್ಡಸಾಲುಗಳು   = ne01/(GGML_METAL_MIN_THREADGROUPS*GGML_METAL_MMA_TILE);
    res.nt = ggml_metal_halve_ಗಮ್ಯ_ಎಲ್ಲೆ(GGML_METAL_MMA_NT_MAX, std::min(nt_smem, nt_ಅಡ್ಡಸಾಲುಗಳು));

    ಹಿಂತಿರುಗಿ res;
}

// the pipeline for the tiling, with fewer simdgroups when the device cannot run that many threads
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, ತಾರ್ಕಿಕ ಕೂಡಿಸು_2) {
    ggml_metal_mma_tiling tiling = ggml_metal_op_mul_mat_mma_tiling(op);

    ಸ್ವಯಂಚಾಲಿತ pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, ಕೂಡಿಸು_2);
    ಆಗ (tiling.nsg > 1 && ggml_metal_pipeline_max_theads_ಪ್ರತಿ_threadgroup(pipeline) < tiling.nsg*32) {
        tiling.nsg /= 2;
        pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, ಕೂಡಿಸು_2);
    }

    ಹಿಂತಿರುಗಿ pipeline;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, ಪೂರ್ಣಾಂಕ nsg, ಪೂರ್ಣಾಂಕ nt, ಪೂರ್ಣಾಂಕ rt, ತಾರ್ಕಿಕ ಕೂಡಿಸು_2) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ggml_type tsrc0 = op->src[0]->type;
    ಸ್ಥಿರಾಂಕ ggml_type tsrc1 = op->src[1]->type;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ne12  = op->src[1]->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       r2    = ne12 / op->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    // the specialized kernels unroll over a compile-time row length
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00 = ggml_metal_mul_mv_mma_ಬಗೆ(tsrc0, rt) != GGML_METAL_MMA_ಬಗೆ_GEN ? op->src[0]->ne[0] : 0;

    snprintf(ತಳ, 256, "kernel_mul_mv_mma_%s_%s_nt%d_rt%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), nt, rt);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_ne00=%d_add=%d", ತಳ, nsg, ne12, r2, r3, ne00, ಕೂಡಿಸು_2);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int32(cv, ne00,           FC_MUL_MV_MMA + 4);
        ggml_metal_cv_ಗಣ_int16(cv, nsg,            FC_MUL_MV_MMA + 0);
        ggml_metal_cv_ಗಣ_int16(cv, (int16_t) ne12, FC_MUL_MV_MMA + 1);
        ggml_metal_cv_ಗಣ_int16(cv, (int16_t) r2,   FC_MUL_MV_MMA + 2);
        ggml_metal_cv_ಗಣ_int16(cv, (int16_t) r3,   FC_MUL_MV_MMA + 3);
        ggml_metal_cv_ಗಣ_bool (cv, ಕೂಡಿಸು_2,            FC_MUL_MV_MMA + 5);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = ggml_metal_mul_mv_mma_smem(nsg, nt, rt);

    res.nr0 = GGML_METAL_MMA_TILE*nt;
    res.nr1 = GGML_METAL_MMA_TILE*rt;

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ggml_type tsrc0 = op->src[0]->type;
    ಸ್ಥಿರಾಂಕ ggml_type tsrc1 = op->src[1]->type;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ bc_inp = op->src[0]->ne[0] % 32 != 0;

    constexpr ಪೂರ್ಣಾಂಕ NRA = SZ_SIMDGROUP * N_MM_ಖಂಡ_Y * N_MM_SIMD_ಗುಂಪು_Y;
    constexpr ಪೂರ್ಣಾಂಕ NRB = SZ_SIMDGROUP * N_MM_ಖಂಡ_X * N_MM_SIMD_ಗುಂಪು_X;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_tensor = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_tensor;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ bc_ಹೊರಹೋಗು = has_tensor
        ? (op->ne[0] % NRA != 0 || op->ne[1] % NRB != 0)
        : (op->ne[0] % 64  != 0 || op->ne[1] % 32  != 0);

    GGML_ASSERT(op->src[1]->ne[2] <= INT16_MAX && op->src[1]->ne[3] <= INT16_MAX);
    ಸ್ಥಿರಾಂಕ int16_t ne12 = (int16_t) op->src[1]->ne[2];
    ಸ್ಥಿರಾಂಕ int16_t ne13 = (int16_t) op->src[1]->ne[3];
    ಸ್ಥಿರಾಂಕ int16_t r2   = (int16_t) (ne12 / op->src[0]->ne[2]);
    ಸ್ಥಿರಾಂಕ int16_t r3   = (int16_t) (ne13 / op->src[0]->ne[3]);

    snprintf(ತಳ, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_bco=%d_ne12=%d_ne13=%d_r2=%d_r3=%d",
             ತಳ, bc_inp, bc_ಹೊರಹೋಗು, ne12, ne13, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_ಗಣ_bool(cv, bc_ಹೊರಹೋಗು, FC_MUL_MM + 1);
        ggml_metal_cv_ಗಣ_int16(cv, ne12,  FC_MUL_MM + 2);
        ggml_metal_cv_ಗಣ_int16(cv, ne13,  FC_MUL_MM + 3);
        ggml_metal_cv_ಗಣ_int16(cv, r2,    FC_MUL_MM + 4);
        ggml_metal_cv_ಗಣ_int16(cv, r3,    FC_MUL_MM + 5);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಆದರೆ (has_tensor) {
        res.nr0 = NRA;
        res.nr1 = NRB;

        ಸ್ಥಿರಾಂಕ size_t smem_a = NRA * N_MM_NK_ಒಟ್ಟು * ಗಾತ್ರ(ggml_fp16_t);
        res.smem = smem_a;
    } ಇಲ್ಲದಿದ್ದರೆ {
        res.nr0 = 64;
        res.nr1 = 32;

        res.smem = bc_ಹೊರಹೋಗು ? 8192 : (4096 + 2048);
    }

    res.nsg = N_MM_SIMD_ಗುಂಪು_X * N_MM_SIMD_ಗುಂಪು_Y;

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಪೂರ್ಣಾಂಕ nsg = 0; // number of simdgroups
    ಪೂರ್ಣಾಂಕ nr0 = 0; // number of src0 rows per simdgroup
    ಪೂರ್ಣಾಂಕ nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    ಸ್ಥಿರಾಂಕ ggml_type tsrc0 = op->src[0]->type;
    ಸ್ಥಿರಾಂಕ ggml_type tsrc1 = op->src[1]->type;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * suffix = "";

    ತಾರ್ಕಿಕ ವಿಭಜಿಸು = ಸುಳ್ಳು;

    // use custom matrix x vector kernel
    ಆಯ್ಕೆ (tsrc0) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ಆದರೆ (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } ಇಲ್ಲದಿದ್ದರೆ {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ)*N_R0_Q8_0;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XXS_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XS_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;

                // split the rows across threads when there are fewer than 32 chunks per row
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_XXS_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_S_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_S_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_S_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_M_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (ಪೂರ್ಣಾಂಕ) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    GGML_ASSERT(ne12 <= INT16_MAX && ne13 <= INT16_MAX);
    ಸ್ಥಿರಾಂಕ int16_t r2 = (int16_t) (ne12 / ne02);
    ಸ್ಥಿರಾಂಕ int16_t r3 = (int16_t) (ne13 / ne03);

    snprintf(ತಳ, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_split=%d", ತಳ, nsg, ne12, r2, r3, ವಿಭಜಿಸು);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_ಗಣ_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_ಗಣ_int16(cv, r2,             FC_MUL_MV + 3);
        ggml_metal_cv_ಗಣ_int16(cv, r3,             FC_MUL_MV + 4);
        ggml_metal_cv_ಗಣ_bool (cv, ವಿಭಜಿಸು,          FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_mul_mm_id_amax_part_f32");
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    res.smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_mul_mm_id_amax_f32");
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    res.smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, ಪೂರ್ಣಾಂಕ ne02, ಪೂರ್ಣಾಂಕ ne20) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s_ne02=%d", ತಳ, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ಗಣ_int32(cv, ne02, FC_MUL_MM + 7);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    // per expert: entry count + base offset, followed by the staging of the ids of ntg tokens
    res.smem = 2*(size_t) ne02*ಗಾತ್ರ(uint32_t) + (size_t) ne02*ne20*ಗಾತ್ರ(uint16_t);
    res.smem = GGML_PAD(res.smem, 16);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ggml_type tsrc0 = op->src[0]->type;
    ಸ್ಥಿರಾಂಕ ggml_type tsrc1 = op->src[1]->type;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ bc_inp = op->src[0]->ne[0] % 32 != 0;

    // src1 prec [TAG_GGML_PREC]
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

    snprintf(ತಳ, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_amax=%d", ತಳ, bc_inp, amax);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_ಗಣ_bool(cv, amax,   FC_MUL_MM + 6);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 8192;

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಪೂರ್ಣಾಂಕ nsg = 0; // number of simdgroups
    ಪೂರ್ಣಾಂಕ nr0 = 0; // number of src0 rows per simdgroup
    ಪೂರ್ಣಾಂಕ nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    ಸ್ಥಿರಾಂಕ ggml_type tsrc0 = op->src[0]->type;
    ಸ್ಥಿರಾಂಕ ggml_type tsrc1 = op->src[1]->type;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * suffix = "";

    ತಾರ್ಕಿಕ ವಿಭಜಿಸು = ಸುಳ್ಳು;

        // use custom matrix x vector kernel
    ಆಯ್ಕೆ (tsrc0) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ)*N_R0_Q8_0;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XXS_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XS_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;

                // split the rows across threads when there are fewer than 32 chunks per row
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_XXS_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_S_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_S_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_S_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb32 = ne00/32;
                ಆದರೆ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_M_ವಿಭಜಿಸು;
                    ವಿಭಜಿಸು = ನಿಜ;
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (ಪೂರ್ಣಾಂಕ)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(ತಳ, 256, "kernel_mul_mv_id_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_split=%d", ತಳ, nsg, ವಿಭಜಿಸು);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_ಗಣ_int16(cv, 1,   FC_MUL_MV + 2);
        ggml_metal_cv_ಗಣ_int16(cv, 1,   FC_MUL_MV + 3);
        ggml_metal_cv_ಗಣ_int16(cv, 1,   FC_MUL_MV + 4);
        ggml_metal_cv_ಗಣ_bool (cv, ವಿಭಜಿಸು, FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[0]->type));

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    res.smem = 32*(ಗಾತ್ರ(ದಶಮಾಂಶ) + ಗಾತ್ರ(int32_t));

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part(ggml_metal_library_t lib) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, "kernel_argmax_part_f32");
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, "kernel_argmax_part_f32", "kernel_argmax_part_f32", ಶೂನ್ಯಸೂಚಕ);
    }

    res.smem = 32*(ಗಾತ್ರ(ದಶಮಾಂಶ) + ಗಾತ್ರ(int32_t));

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final(ggml_metal_library_t lib) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, "kernel_argmax_final_f32");
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, "kernel_argmax_final_f32", "kernel_argmax_final_f32", ಶೂನ್ಯಸೂಚಕ);
    }

    res.smem = 32*(ಗಾತ್ರ(ದಶಮಾಂಶ) + ಗಾತ್ರ(int32_t));

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ARGSORT);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ = (ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ) op->op_params[0];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕ್ರಮ_str = "undefined";
    ಆಯ್ಕೆ (ಕ್ರಮ) {
        ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC:  ಕ್ರಮ_str = "asc";  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC: ಕ್ರಮ_str = "desc"; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    };

    snprintf(ತಳ, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), ಕ್ರಮ_str);
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_ವಿಲೀನ(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ARGSORT);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ = (ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ) op->op_params[0];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕ್ರಮ_str = "undefined";
    ಆಯ್ಕೆ (ಕ್ರಮ) {
        ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC:  ಕ್ರಮ_str = "asc";  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC: ಕ್ರಮ_str = "desc"; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    };

    snprintf(ತಳ, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), ಕ್ರಮ_str);
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht(ggml_metal_library_t lib, ಪೂರ್ಣಾಂಕ n, ggml_type tsrc) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_fwht_%s_%d", ggml_type_name(tsrc), n);
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

// note: reuse the argsort kernel for the bitonic top_k fallback
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ತುದಿ_k(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ತುದಿ_K);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    // note: the top_k kernel is always descending order
    ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ = GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕ್ರಮ_str = "undefined";
    ಆಯ್ಕೆ (ಕ್ರಮ) {
        ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC:  ಕ್ರಮ_str = "asc";  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC: ಕ್ರಮ_str = "desc"; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    };

    snprintf(ತಳ, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), ಕ್ರಮ_str);
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ತುದಿ_k_radix(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ತುದಿ_K);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_top_k_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ತುದಿ_k_ವಿಲೀನ(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ತುದಿ_K);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ = GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕ್ರಮ_str = "undefined";
    ಆಯ್ಕೆ (ಕ್ರಮ) {
        ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC:  ಕ್ರಮ_str = "asc";  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC: ಕ್ರಮ_str = "desc"; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    };

    snprintf(ತಳ, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), ಕ್ರಮ_str);
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe(
        ggml_metal_library_t lib, int32_t n_expert, int32_t ತುದಿ_k, ತಾರ್ಕಿಕ with_norm) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_topk_moe_f32");
    snprintf(name, 256, "%s_n_expert=%d_top_k=%d_with_norm=%d", ತಳ, n_expert, ತುದಿ_k, with_norm ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ಗಣ_bool (cv, with_norm, FC_TOPK_MOE + 0);
        ggml_metal_cv_ಗಣ_int32(cv, n_expert,  FC_TOPK_MOE + 1);
        ggml_metal_cv_ಗಣ_int32(cv, ತುದಿ_k,     FC_TOPK_MOE + 2);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce(ggml_metal_library_t lib, int32_t n_expert_ಬಳಸಿದ) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_moe_reduce_f32");
    snprintf(name, 256, "%s_n_expert_used=%d", ತಳ, n_expert_ಬಳಸಿದ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ಗಣ_int32(cv, n_expert_ಬಳಸಿದ, FC_MOE_REDUCE + 0);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
        ತಾರ್ಕಿಕ    has_ಮುಖವಾಡ,
        int32_t ncpsg) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ಬಳಸದ(op);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_%s",
            "flash_attn_ext_pad");

    snprintf(name, 256, "%s_mask=%d_ncpsg=%d",
            ತಳ,
            has_ಮುಖವಾಡ,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_bool(cv, has_ಮುಖವಾಡ,  FC_FLASH_ATTN_EXT_PAD + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_PAD + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_PAD + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_PAD + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_PAD + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_PAD + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_PAD + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_PAD + 23);
        //ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_PAD + 24);
        ggml_metal_cv_ಗಣ_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ಅಕ್ಷರ ತಳ[256];

    snprintf(ತಳ, 256, "kernel_flash_attn_ext_kv_%s_f16", ggml_type_name(op->src[1]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, ತಳ);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, ತಳ, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ಬಳಸದ(op);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_%s",
            "flash_attn_ext_blk");

    snprintf(name, 256, "%s_nqptg=%d_ncpsg=%d",
            ತಳ,
            nqptg,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        //ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_BLK + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_BLK + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_BLK + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_BLK + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_BLK + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_BLK + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_BLK + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_BLK + 23);
        ggml_metal_cv_ಗಣ_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
        ggml_metal_cv_ಗಣ_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ggml_tensor * op,
        ತಾರ್ಕಿಕ has_ಮುಖವಾಡ,
        ತಾರ್ಕಿಕ has_sinks,
        ತಾರ್ಕಿಕ has_bias,
        ತಾರ್ಕಿಕ has_scap) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ int32_t dk = (int32_t) op->src[1]->ne[0];
    ಸ್ಥಿರಾಂಕ int32_t dv = (int32_t) op->src[2]->ne[0];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * type = op->src[1]->type == GGML_TYPE_BF16 ? "bf16" : "f16";

    snprintf(ತಳ, 256, "kernel_flash_attn_ext_tensor_%s_dk%d_dv%d", type, dk, dv);
    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d", ತಳ, has_ಮುಖವಾಡ, has_sinks, has_bias, has_scap);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_bool(cv, has_ಮುಖವಾಡ,  FC_FLASH_ATTN_EXT_TENSOR + 0);
        ggml_metal_cv_ಗಣ_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_TENSOR + 1);
        ggml_metal_cv_ಗಣ_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_TENSOR + 2);
        ggml_metal_cv_ಗಣ_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_TENSOR + 3);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ggml_tensor * op,
        ತಾರ್ಕಿಕ    has_ಮುಖವಾಡ,
        ತಾರ್ಕಿಕ    has_sinks,
        ತಾರ್ಕಿಕ    has_bias,
        ತಾರ್ಕಿಕ    has_scap,
        ತಾರ್ಕಿಕ    has_kvpad,
        int32_t nsg,
        ತಾರ್ಕಿಕ    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ int32_t dk = (int32_t) op->src[1]->ne[0];
    ಸ್ಥಿರಾಂಕ int32_t dv = (int32_t) op->src[2]->ne[0];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    // do bounds checks for the mask?
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ bc_ಮುಖವಾಡ = op->src[3] && (op->src[3]->ne[1] % 8 != 0);

    snprintf(ತಳ, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            type,
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
            ತಳ,
            has_ಮುಖವಾಡ,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            bc_ಮುಖವಾಡ,
            ns10,
            ns20,
            nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_bool(cv, has_ಮುಖವಾಡ,  FC_FLASH_ATTN_EXT + 0);
        ggml_metal_cv_ಗಣ_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        ggml_metal_cv_ಗಣ_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
        ggml_metal_cv_ಗಣ_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);
        ggml_metal_cv_ಗಣ_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);

        ggml_metal_cv_ಗಣ_bool(cv, bc_ಮುಖವಾಡ, FC_FLASH_ATTN_EXT + 10);

        ggml_metal_cv_ಗಣ_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        ggml_metal_cv_ಗಣ_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        ggml_metal_cv_ಗಣ_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);
    ಪ್ರತಿಜ್ಞಾ(op->src[3]);

    ಅಕ್ಷರ name[256];

    snprintf(name, 256, "kernel_flash_attn_ext_vec_idx");

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, ಶೂನ್ಯಸೂಚಕ);
    }

    GGML_ಬಳಸದ(op);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ggml_tensor * op,
        ತಾರ್ಕಿಕ    has_ಮುಖವಾಡ,
        ತಾರ್ಕಿಕ    has_sinks,
        ತಾರ್ಕಿಕ    has_bias,
        ತಾರ್ಕಿಕ    has_scap,
        ತಾರ್ಕಿಕ    has_kvpad,
        ತಾರ್ಕಿಕ    has_sparse,
        int32_t nqpsg,
        int32_t ne,
        int32_t nsg,
        int32_t nwg,
        ತಾರ್ಕಿಕ    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ int32_t dk = (int32_t) op->src[1]->ne[0];
    ಸ್ಥಿರಾಂಕ int32_t dv = (int32_t) op->src[2]->ne[0];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    ಅಕ್ಷರ qne_suffix[16] = {0};
    ಆದರೆ (!(nqpsg == 1 && ne == ggml_metal_tuning::fa_vec_baseline_ne(dk, dv))) {
        snprintf(qne_suffix, ಗಾತ್ರ(qne_suffix), "_q%d_ne%d", nqpsg, ne);
    }

    snprintf(ತಳ, 256, "kernel_%s_%s_dk%d_dv%d%s",
            "flash_attn_ext_vec",
            type,
            dk,
            dv,
            qne_suffix);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_sparse=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            ತಳ,
            has_ಮುಖವಾಡ,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            has_sparse,
            ns10,
            ns20,
            nsg, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_bool(cv, has_ಮುಖವಾಡ,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_ಗಣ_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_ಗಣ_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_ಗಣ_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_ಗಣ_bool(cv, has_kvpad,  FC_FLASH_ATTN_EXT_VEC + 4);
        ggml_metal_cv_ಗಣ_bool(cv, has_sparse, FC_FLASH_ATTN_EXT_VEC + 5);

        ggml_metal_cv_ಗಣ_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_ಗಣ_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_ಗಣ_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_ಗಣ_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ggml_tensor * op,
        int32_t dv,
        int32_t nwg) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_FLASH_ATTN_EXT);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "%s_dv=%d_nwg=%d", ತಳ, dv, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        ggml_metal_cv_ಗಣ_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;

    GGML_ಬಳಸದ(op);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, int32_t n_fuse) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಪೂರ್ಣಾಂಕ op_ಅಂಕೆ = -1;

    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು: op_ಅಂಕೆ = 0; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SUB: op_ಅಂಕೆ = 1; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL: op_ಅಂಕೆ = 2; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIV: op_ಅಂಕೆ = 3; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    };

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * t0_str = ggml_type_name(op->src[0]->type);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * t1_str = ggml_type_name(op->src[1]->type);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * t_str  = ggml_type_name(op->type);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_cb = op->src[0]->ne[0] != op->src[1]->ne[0];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(ತಳ, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d_cb=%d", ತಳ, op_ಅಂಕೆ, n_fuse, is_rb, is_cb);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, op_ಅಂಕೆ, FC_BIN + 0);
        ggml_metal_cv_ಗಣ_int16(cv, n_fuse, FC_BIN + 1);
        ggml_metal_cv_ಗಣ_bool (cv, is_rb,  FC_BIN + 2);
        ggml_metal_cv_ಗಣ_bool (cv, is_cb,  FC_BIN + 3);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_rb;

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one(ggml_metal_library_t lib, ggml_op op) {
    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಪೂರ್ಣಾಂಕ op_ಅಂಕೆ = -1;

    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು: op_ಅಂಕೆ = 0; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SUB: op_ಅಂಕೆ = 1; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL: op_ಅಂಕೆ = 2; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIV: op_ಅಂಕೆ = 3; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    };

    snprintf(ತಳ, 256, "kernel_bin_fuse_%s_%s_%s", "f32", "f32", "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", ತಳ, op_ಅಂಕೆ, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, op_ಅಂಕೆ, FC_BIN + 0);
        ggml_metal_cv_ಗಣ_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_ಗಣ_bool (cv, ಸುಳ್ಳು,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_L2_NORM);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_c4 = op->src[0]->ne[0] % 4 == 0;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * t0_str = ggml_type_name(op->src[0]->type);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * t_str  = ggml_type_name(op->type);

    snprintf(ತಳ, 256, "kernel_l2_norm_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    res.c4   = is_c4;
    res.smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ಗುಂಪು_norm(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ಗುಂಪು_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    res.smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, ಪೂರ್ಣಾಂಕ n_fuse) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op->src[0]));

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * suffix = "";
    ಆದರೆ (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_NORM:
            ಆಯ್ಕೆ (n_fuse) {
                ಸ್ಥಿತಿ 1: snprintf(ತಳ, 256, "kernel_norm_f32%s", suffix);         ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 2: snprintf(ತಳ, 256, "kernel_norm_mul_f32%s", suffix);     ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 3: snprintf(ತಳ, 256, "kernel_norm_mul_add_f32%s", suffix); ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            ಆಯ್ಕೆ (n_fuse) {
                ಸ್ಥಿತಿ 1: snprintf(ತಳ, 256, "kernel_rms_norm_f32%s", suffix);         ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 2: snprintf(ತಳ, 256, "kernel_rms_norm_mul_f32%s", suffix);     ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 3: snprintf(ತಳ, 256, "kernel_rms_norm_mul_add_f32%s", suffix); ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ಗಣ_bool(cv, ಸುಳ್ಳು, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op->src[0]));

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * suffix = "";
    ಆದರೆ (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_NORM:     snprintf(ತಳ, 256, "kernel_norm_mul_f32%s", suffix);     ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM: snprintf(ತಳ, 256, "kernel_rms_norm_mul_f32%s", suffix); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s_use_scale", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_ಗಣ_bool(cv, ನಿಜ, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ROPE || op->op == GGML_OP_ROPE_ಹಿಂಭಾಗ);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ಹಿಂಭಾಗ = op->op == GGML_OP_ROPE_ಹಿಂಭಾಗ;

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[2];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_vision = mode == GGML_ROPE_TYPE_VISION;

    ಆದರೆ (is_neox) {
        snprintf(ತಳ, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((is_mrope || is_imrope) && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(ತಳ, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(ತಳ, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } ಇಲ್ಲದಿದ್ದರೆ {
        snprintf(ತಳ, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s_imrope=%d_is_back=%d", ತಳ, is_imrope ? 1 : 0, is_ಹಿಂಭಾಗ ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_bool(cv, is_imrope, FC_ROPE + 0);
        ggml_metal_cv_ಗಣ_bool(cv, is_ಹಿಂಭಾಗ,   FC_ROPE + 1);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_IM2_ಲಂಬಸಾಲು);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_2D = ((ಸ್ಥಿರಾಂಕ int32_t *)(op->op_params))[6] == 1;
    ಸ್ಥಿರಾಂಕ int64_t KH = is_2D ? ne01 : 1;
    ಸ್ಥಿರಾಂಕ int64_t KW = ne00;

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಆದರೆ (KH*KW <= 1024) {
        snprintf(ತಳ, 256, "kernel_im2col_%s", ggml_type_name(op->type));
    } ಇಲ್ಲದಿದ್ದರೆ {
        snprintf(ತಳ, 256, "kernel_im2col_ext_%s", ggml_type_name(op->type));
    }
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_COL2IM_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_col2im_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake(ggml_metal_library_t lib, ಎಣಿಕೆ ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_snake_%s", ggml_type_name(type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_conv_transpose_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_CONV_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_conv_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op, ತಾರ್ಕಿಕ tiled) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_CONV_2D_DW);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_conv_2d_dw%s_%s_%s",
             tiled ? "_tiled" : "",
             ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_CONV_3D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_conv_3d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_UPSCALE);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ int32_t ಬಹುಲಕ_flags = ggml_get_op_params_i32(op, 0);
    ಸ್ಥಿರಾಂಕ ggml_scale_ಬಹುಲಕ mode = (ggml_scale_ಬಹುಲಕ) (ಬಹುಲಕ_flags & 0xFF);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ antialias = (ಬಹುಲಕ_flags & GGML_SCALE_FLAG_ANTIALIAS);

    ಆದರೆ (mode == GGML_SCALE_ಬಹುಲಕ_BILINEAR) {
        snprintf(ತಳ, 256, "kernel_upscale_bilinear_%s", ggml_type_name(op->src[0]->type));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mode == GGML_SCALE_ಬಹುಲಕ_BICUBIC) {
        snprintf(ತಳ, 256, "kernel_upscale_bicubic_%s", ggml_type_name(op->src[0]->type));
    } ಇಲ್ಲದಿದ್ದರೆ {
        snprintf(ತಳ, 256, "kernel_upscale_nearest_%s", ggml_type_name(op->src[0]->type));
    }
    snprintf(name, 256, "%s_aa=%d", ತಳ, antialias);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_bool(cv, antialias, FC_UPSCALE + 0);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ROLL);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_roll_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_PAD);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ circular = ggml_get_op_params_i32(op, 8) != 0;

    snprintf(ತಳ, 256, "kernel_pad_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_circular=%d", ತಳ, circular);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_bool(cv, circular, FC_PAD + 0);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_PAD_REFLECT_1D);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ARANGE);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_ಹೆಜ್ಜೆ_adamw(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_ಹೆಜ್ಜೆ_sgd(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_ಹಿಂಭಾಗ(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_SILU_ಹಿಂಭಾಗ);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_silu_back_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor *  op) {
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    snprintf(ತಳ, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", ತಳ);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, ಶೂನ್ಯಸೂಚಕ);
    }

    ಹಿಂತಿರುಗಿ res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ಎಣಿಕೆ_equal(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ggml_tensor *  op) {
    ಪ್ರತಿಜ್ಞಾ(op->op == GGML_OP_ಎಣಿಕೆ_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    ಅಕ್ಷರ ತಳ[256];
    ಅಕ್ಷರ name[256];

    ಪೂರ್ಣಾಂಕ nsg = 1;
    ಆಗ (32*nsg < ne00 && nsg < 32) {
        nsg *= 2;
    }

    snprintf(ತಳ, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", ತಳ, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ಆದರೆ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_ಗಣ_int16(cv, nsg, FC_ಎಣಿಕೆ_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, ತಳ, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32 * ಗಾತ್ರ(int32_t);
    res.nsg  = nsg;

    ಹಿಂತಿರುಗಿ res;
}
