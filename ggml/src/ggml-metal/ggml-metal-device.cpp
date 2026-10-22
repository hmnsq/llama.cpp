#अन्तर्भाव "ggml-metal-device.h"
#अन्तर्भाव "ggml-metal-common.h"

#अन्तर्भाव "ggml-metal-impl.h"
#अन्तर्भाव "ggml-metal-tuning.h"

#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cassert>
#अन्तर्भाव <memory>
#अन्तर्भाव <string>
#अन्तर्भाव <unordered_map>

संरचना ggml_metal_device_deleter {
    शून्य संक्रियकः()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

प्रकारनिर्देश std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

ggml_metal_device_t ggml_metal_device_get(पूर्णाङ्क device, पूर्णाङ्क n_devices) {
    स्थिर std::vector<ggml_metal_device_ptr> devs;

    devs.emplace_back(ggml_metal_device_init(device, n_devices));

    प्रत्यागम devs.back().get();
}

संरचना ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> दत्त;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(शून्य) {
    ggml_metal_pipelines_t res = नूतन ggml_metal_pipelines();

    प्रत्यागम res;
}

शून्य ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    यदि (!ppls) {
        प्रत्यागम;
    }

    प्रति (स्वचालित it = ppls->दत्त.begin(); it != ppls->दत्त.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    विलोपय ppls;
}

शून्य ggml_metal_pipelines_सङ्कलय(ggml_metal_pipelines_t ppls, ध्रुव वर्ण * name, ggml_metal_pipeline_t pipeline) {
    ppls->दत्त[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ध्रुव वर्ण * name) {
    यदि (ppls->दत्त.find(name) == ppls->दत्त.end()) {
        प्रत्यागम शून्यसूचकः;
    }

    प्रत्यागम ppls->दत्त[name];
}

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_आधारः(ggml_metal_library_t lib, ggml_op op) {
    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव वर्ण * op_str = "undefined";
    विकल्प (op) {
        पक्ष GGML_OP_सङ्कलय_ID: op_str = "add_id"; भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    };

    snprintf(आधारः, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d(ggml_metal_library_t lib, ध्रुव ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    ध्रुव वर्ण * pool_str = "undefined";
    विकल्प (op_pool) {
        पक्ष GGML_OP_POOL_AVG: pool_str = "avg"; भङ्ग;
        पक्ष GGML_OP_POOL_MAX: pool_str = "max"; भङ्ग;
        स्वभाव: GGML_ASSERT(असत्य && "not implemented");
    };

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, परिमाण(आधारः), "kernel_pool_1d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, परिमाण(name), "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_metal_library_t lib, ध्रुव ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    ध्रुव वर्ण * pool_str = "undefined";
    विकल्प (op_pool) {
        पक्ष GGML_OP_POOL_AVG: pool_str = "avg"; भङ्ग;
        पक्ष GGML_OP_POOL_MAX: pool_str = "max"; भङ्ग;
        स्वभाव: GGML_ASSERT(असत्य && "not implemented");
    };

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_पङ्क्तिकाः(ggml_metal_library_t lib, ggml_type tsrc) {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_समुच्चयः_पङ्क्तिकाः(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव स्वचालित tsrc = op->src[0]->type;
    ध्रुव स्वचालित tidx = op->src[1]->type;
    ध्रुव स्वचालित tdst = op->type;

    snprintf(आधारः, 256, "kernel_set_rows_%s_%s_%s", ggml_type_name(tsrc), ggml_type_name(tidx), ggml_type_name(tdst));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव पूर्णाङ्क n = op->src[0]->ne[0];

    snprintf(आधारः, 256, "kernel_diag_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_n=%d", आधारः, n);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    res.nsg  = 1;
    res.smem = 0;

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat(ggml_metal_library_t lib, ggml_type tsrc) {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat(ggml_metal_library_t lib, ggml_type tsrc) {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_concat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    वर्ण आधारः[256];
    वर्ण name[256];

    पूर्णाङ्क op_अङ्कः = -1;

    विकल्प (op->op) {
        पक्ष GGML_OP_SCALE:      op_अङ्कः = OP_UNARY_अङ्कः_SCALE;      भङ्ग;
        पक्ष GGML_OP_FILL:       op_अङ्कः = OP_UNARY_अङ्कः_FILL;       भङ्ग;
        पक्ष GGML_OP_CLAMP:      op_अङ्कः = OP_UNARY_अङ्कः_CLAMP;      भङ्ग;
        पक्ष GGML_OP_SQR:        op_अङ्कः = OP_UNARY_अङ्कः_SQR;        भङ्ग;
        पक्ष GGML_OP_SQRT:       op_अङ्कः = OP_UNARY_अङ्कः_SQRT;       भङ्ग;
        पक्ष GGML_OP_SIN:        op_अङ्कः = OP_UNARY_अङ्कः_SIN;        भङ्ग;
        पक्ष GGML_OP_COS:        op_अङ्कः = OP_UNARY_अङ्कः_COS;        भङ्ग;
        पक्ष GGML_OP_LOG:        op_अङ्कः = OP_UNARY_अङ्कः_LOG;        भङ्ग;
        पक्ष GGML_OP_LEAKY_RELU: op_अङ्कः = OP_UNARY_अङ्कः_LEAKY_RELU; भङ्ग;
        पक्ष GGML_OP_UNARY:
            विकल्प (ggml_get_unary_op(op)) {
                पक्ष GGML_UNARY_OP_TANH:        op_अङ्कः = OP_UNARY_अङ्कः_TANH;        भङ्ग;
                पक्ष GGML_UNARY_OP_RELU:        op_अङ्कः = OP_UNARY_अङ्कः_RELU;        भङ्ग;
                पक्ष GGML_UNARY_OP_SIGMOID:     op_अङ्कः = OP_UNARY_अङ्कः_SIGMOID;     भङ्ग;
                पक्ष GGML_UNARY_OP_GELU:        op_अङ्कः = OP_UNARY_अङ्कः_GELU;        भङ्ग;
                पक्ष GGML_UNARY_OP_GELU_ERF:    op_अङ्कः = OP_UNARY_अङ्कः_GELU_ERF;    भङ्ग;
                पक्ष GGML_UNARY_OP_GELU_शीघ्र:  op_अङ्कः = OP_UNARY_अङ्कः_GELU_शीघ्र;  भङ्ग;
                पक्ष GGML_UNARY_OP_SILU:        op_अङ्कः = OP_UNARY_अङ्कः_SILU;        भङ्ग;
                पक्ष GGML_UNARY_OP_ELU:         op_अङ्कः = OP_UNARY_अङ्कः_ELU;         भङ्ग;
                पक्ष GGML_UNARY_OP_NEG:         op_अङ्कः = OP_UNARY_अङ्कः_NEG;         भङ्ग;
                पक्ष GGML_UNARY_OP_ABS:         op_अङ्कः = OP_UNARY_अङ्कः_ABS;         भङ्ग;
                पक्ष GGML_UNARY_OP_SGN:         op_अङ्कः = OP_UNARY_अङ्कः_SGN;         भङ्ग;
                पक्ष GGML_UNARY_OP_चरणः:        op_अङ्कः = OP_UNARY_अङ्कः_चरणः;        भङ्ग;
                पक्ष GGML_UNARY_OP_HARDSWISH:   op_अङ्कः = OP_UNARY_अङ्कः_HARDSWISH;   भङ्ग;
                पक्ष GGML_UNARY_OP_HARDSIGMOID: op_अङ्कः = OP_UNARY_अङ्कः_HARDSIGMOID; भङ्ग;
                पक्ष GGML_UNARY_OP_EXP:         op_अङ्कः = OP_UNARY_अङ्कः_EXP;         भङ्ग;
                पक्ष GGML_UNARY_OP_SOFTPLUS:    op_अङ्कः = OP_UNARY_अङ्कः_SOFTPLUS;    भङ्ग;
                पक्ष GGML_UNARY_OP_EXPM1:       op_अङ्कः = OP_UNARY_अङ्कः_EXPM1;       भङ्ग;
                पक्ष GGML_UNARY_OP_FLOOR:       op_अङ्कः = OP_UNARY_अङ्कः_FLOOR;       भङ्ग;
                पक्ष GGML_UNARY_OP_CEIL:        op_अङ्कः = OP_UNARY_अङ्कः_CEIL;        भङ्ग;
                पक्ष GGML_UNARY_OP_आवर्तनम्:       op_अङ्कः = OP_UNARY_अङ्कः_आवर्तनम्;       भङ्ग;
                पक्ष GGML_UNARY_OP_TRUNC:       op_अङ्कः = OP_UNARY_अङ्कः_TRUNC;       भङ्ग;
                पक्ष GGML_UNARY_OP_XIELU:       op_अङ्कः = OP_UNARY_अङ्कः_XIELU;       भङ्ग;
                स्वभाव: GGML_ABORT("fatal error");
            } भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    };

    ध्रुव वर्ण * t0_str = ggml_type_name(op->src[0]->type);
    ध्रुव वर्ण * t_str  = ggml_type_name(op->type);

    ध्रुव तार्किक is_c4 = op->src[0]->ne[0] % 4 == 0;
    ध्रुव तार्किक is_cnt = ggml_is_contiguous(op->src[0]) && ggml_nelements(op) < 32768;

    snprintf(आधारः, 256, "kernel_unary_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_cnt=%d", आधारः, op_अङ्कः, is_cnt);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, op_अङ्कः, FC_UNARY + 0);
        ggml_metal_cv_समुच्चयः_bool (cv, is_cnt, FC_UNARY + 1);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_cnt;

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव वर्ण * op_str = "undefined";
    विकल्प (op->op) {
        पक्ष GGML_OP_GLU:
            विकल्प (ggml_get_glu_op(op)) {
                पक्ष GGML_GLU_OP_REGLU:        op_str = "reglu";        भङ्ग;
                पक्ष GGML_GLU_OP_GEGLU:        op_str = "geglu";        भङ्ग;
                पक्ष GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       भङ्ग;
                पक्ष GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   भङ्ग;
                पक्ष GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    भङ्ग;
                पक्ष GGML_GLU_OP_GEGLU_शीघ्र:  op_str = "geglu_quick";  भङ्ग;
                पक्ष GGML_GLU_OP_SWIGLU_CLAMP: op_str = "swiglu_clamp"; भङ्ग;
                स्वभाव: GGML_ABORT("fatal error");
            } भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    };

    snprintf(आधारः, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_योग(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_योग);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_op_sum_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_योग_पङ्क्तिकाः(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(op->src[0]));

    वर्ण आधारः[256];
    वर्ण name[256];

    पूर्णाङ्क op_अङ्कः = -1;

    विकल्प (op->op) {
        पक्ष GGML_OP_योग_पङ्क्तिकाः: op_अङ्कः = OP_योग_पङ्क्तिकाः_अङ्कः_योग_पङ्क्तिकाः; भङ्ग;
        पक्ष GGML_OP_माध्यम्:     op_अङ्कः = OP_योग_पङ्क्तिकाः_अङ्कः_माध्यम्;     भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    };

    ध्रुव वर्ण * t0_str = ggml_type_name(op->src[0]->type);
    ध्रुव वर्ण * t_str  = ggml_type_name(op->type);

    ध्रुव तार्किक is_c4 = op->src[0]->ne[0] % 4 == 0;

    snprintf(आधारः, 256, "kernel_sum_rows_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d", आधारः, op_अङ्कः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, op_अङ्कः, FC_योग_पङ्क्तिकाः + 0);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*परिमाण(भिन्न);

    यदि (is_c4) {
        res.smem *= 4;
    }

    res.c4  = is_c4;

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_cumsum_blk_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_सङ्कलय(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_cumsum_add_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_TRI);
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_आकार(op->src[0]->type));

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव वर्ण * op_str = "tri";
    ध्रुव पूर्णाङ्क ttype = op->op_params[0];

    snprintf(आधारः, 256, "kernel_%s_%s_%d", op_str, ggml_type_name(op->src[0]->type), ttype);

    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव वर्ण * suffix = "";

    यदि (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    ध्रुव ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(आधारः, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    res.smem = 32*परिमाण(भिन्न);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer(
        ggml_metal_library_t lib,
        ध्रुव ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव int16_t nh = op->src[0]->ne[1];

    snprintf(आधारः, 256, "kernel_lightning_indexer_%s", ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s_nh=%d", आधारः, nh);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, nh, FC_LIGHTNING_INDEXER + 0);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    वर्ण name[256];
    ध्रुव वर्ण * आधारः = शून्यसूचकः;

    विकल्प (op->op) {
        पक्ष GGML_OP_DSV4_HC_COMB:
            आधारः = "kernel_dsv4_hc_comb_f32";
            snprintf(name, 256, "%s", आधारः);
            भङ्ग;
        पक्ष GGML_OP_DSV4_HC_PRE:
            आधारः = ggml_get_op_params_i32(op, 1) != 0 ? "kernel_dsv4_hc_pre_gated_f32" : "kernel_dsv4_hc_pre_f32";
            snprintf(name, 256, "%s_n_hc=%d", आधारः, (पूर्णाङ्क) op->src[0]->ne[1]);
            भङ्ग;
        पक्ष GGML_OP_DSV4_HC_POST:
            आधारः = op->src[3] ? "kernel_dsv4_hc_post_f32" : "kernel_dsv4_hc_post_nocomb_f32";
            snprintf(name, 256, "%s", आधारः);
            भङ्ग;
        स्वभाव:
            GGML_ABORT("fatal error");
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = शून्यसूचकः;

        यदि (op->op == GGML_OP_DSV4_HC_PRE) {
            cv = ggml_metal_cv_init();
            ggml_metal_cv_समुच्चयः_int32(cv, (int32_t) op->src[0]->ne[1], FC_DSV4_HC + 0);
        }

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        यदि (cv) {
            ggml_metal_cv_free(cv);
        }
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv(
        ggml_metal_library_t lib, ध्रुव ggml_tensor * op, int32_t nc, तार्किक use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव वर्ण * suffix = "";

    यदि (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(आधारः, 256, "kernel_ssm_conv_%s_%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_nc=%d_silu=%d", आधारः, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_समुच्चयः_bool(cv, use_silu, FC_SSM_CONV + 1);
        ggml_metal_cv_समुच्चयः_int32(cv, nc,      FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched(
        ggml_metal_library_t lib, ध्रुव ggml_tensor * op, पूर्णाङ्क ssm_conv_bs, int32_t nc, तार्किक use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव वर्ण * suffix = "";
    यदि (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(आधारः, 256, "kernel_ssm_conv_%s_%s_batched%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_ssm_conv_bs=%d_nc=%d_silu=%d", आधारः, ssm_conv_bs, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, ssm_conv_bs, FC_SSM_CONV + 0);
        ggml_metal_cv_समुच्चयः_bool(cv, use_silu,     FC_SSM_CONV + 1);
        ggml_metal_cv_समुच्चयः_int32(cv, nc,          FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_अवलोकनम्(ggml_metal_library_t lib, ध्रुव ggml_tensor * op, तार्किक पुच्छ)  {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव पूर्णाङ्क nsg = (ne00 + 31)/32;

    snprintf(आधारः, 256, "kernel_ssm_scan_%s%s", ggml_type_name(op->src[0]->type), पुच्छ ? "_tail" : "");
    snprintf(name, 256, "%s_nsg=%d", आधारः, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (32 + 2) floats
    res.smem = GGML_PAD((32 + 2)*परिमाण(भिन्न)*nsg, 16);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_अवलोकनम्_ssd_mma(ggml_metal_library_t lib, ध्रुव ggml_tensor * op)  {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_ssm_scan_ssd_mma_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    // acs/exp(acs)/state-decay vectors + dtX + SAM rows + two 8x8 tiles per simdgroup
    res.smem = (3*OP_SSM_अवलोकनम्_SSD_CS +
                OP_SSM_अवलोकनम्_SSD_CS*OP_SSM_अवलोकनम्_SSD_HD +
                OP_SSM_अवलोकनम्_SSD_NSG*8*OP_SSM_अवलोकनम्_SSD_CS +
                OP_SSM_अवलोकनम्_SSD_NSG*2*8*8)*परिमाण(भिन्न);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव int64_t C = op->ne[0];
    ध्रुव int64_t H = op->src[0]->ne[1];

    विकल्प (op->op) {
        पक्ष GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(आधारः, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } भङ्ग;
        पक्ष GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(आधारः, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } भङ्ग;
        स्वभाव:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    वर्ण आधारः[256];
    वर्ण name[256];

    // v is src[2], dimensions: S_v = ne[0], H = ne[1]
    ध्रुव पूर्णाङ्क ne20 = op->src[2]->ne[0]; // S_v
    ध्रुव पूर्णाङ्क ne21 = op->src[2]->ne[1]; // H
    ध्रुव पूर्णाङ्क ne30 = op->src[3]->ne[0]; // G
    // state is src[5], 4D [S_v, S_v, H_v, n_seqs] (s0 only); K is op param 0.
    ध्रुव पूर्णाङ्क K = ggml_get_op_params_i32(op, 0);

    ध्रुव पूर्णाङ्क nsg = op->src[2]->ne[0]/32;

    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->ne[0] == ne20 * ne21);
    GGML_ASSERT(ne20 % 32 == 0);

    snprintf(आधारः, 256, "kernel_gated_delta_net_%s_%d", ggml_type_name(op->src[0]->type), nsg);
    snprintf(name, 256, "%s_ne20=%d_ne30=%d_K=%d", आधारः, ne20, ne30, K);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, ne20, FC_GATED_DELTA_NET + 0);
        ggml_metal_cv_समुच्चयः_int16(cv, ne30, FC_GATED_DELTA_NET + 1);
        ggml_metal_cv_समुच्चयः_int16(cv, K,    FC_GATED_DELTA_NET + 2);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg = nsg;

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव पूर्णाङ्क nsg = 8;
    ध्रुव पूर्णाङ्क n   = op->src[1]->ne[1];
    ध्रुव पूर्णाङ्क k   = op->src[1]->ne[0];

    snprintf(आधारः, 256, "kernel_solve_tri_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d_n=%d_k=%d", आधारः, nsg, n, k);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, nsg, FC_SOLVE_TRI + 0);
        ggml_metal_cv_समुच्चयः_int16(cv, n,   FC_SOLVE_TRI + 1);
        ggml_metal_cv_समुच्चयः_int16(cv, k,   FC_SOLVE_TRI + 2);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = GGML_PAD(GGML_PAD(n, 32)*nsg*परिमाण(भिन्न), 16);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, ध्रुव ggml_tensor * op, पूर्णाङ्क nsg, पूर्णाङ्क nxpsg, पूर्णाङ्क r1ptg) {
    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव ggml_type tsrc0 = op->src[0]->type;
    ध्रुव ggml_type tsrc1 = op->src[1]->type;
    ध्रुव पूर्णाङ्क       ne12  = op->src[1]->ne[2];
    ध्रुव पूर्णाङ्क       r2    = ne12 / op->src[0]->ne[2];
    ध्रुव पूर्णाङ्क       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    snprintf(आधारः, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d_ne12=%d_r2=%d_r3=%d", आधारः, nsg, nxpsg, ne12, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_समुच्चयः_int16(cv, nxpsg,          FC_MUL_MV + 1);
        ggml_metal_cv_समुच्चयः_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_समुच्चयः_int16(cv, (int16_t) r2,   FC_MUL_MV + 3);
        ggml_metal_cv_समुच्चयः_int16(cv, (int16_t) r3,   FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

// threadgroups needed to fill the GPU
// TODO: dedup
स्थिर constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

// few-row MMA mat-mul: 8x8 simdgroup matrices for the src1 rows of ggml_metal_mul_mat_use_mma
स्थिर constexpr पूर्णाङ्क     GGML_METAL_MMA_TILE          = 8;
स्थिर constexpr पूर्णाङ्क     GGML_METAL_MMA_NT_MAX        = 4;
स्थिर constexpr पूर्णाङ्क     GGML_METAL_MMA_SMEM_MAX      = 16384;

// simdgroups per threadgroup for at most FEW_ROWS, at most MANY_ROWS, and more src0 rows
स्थिर constexpr int64_t GGML_METAL_MMA_FEW_पङ्क्तिकाः      = 64;
स्थिर constexpr int64_t GGML_METAL_MMA_MANY_पङ्क्तिकाः     = 6144;
स्थिर constexpr पूर्णाङ्क     GGML_METAL_MMA_NSG_FEW_पङ्क्तिकाः  = 32;
स्थिर constexpr पूर्णाङ्क     GGML_METAL_MMA_NSG_MID_पङ्क्तिकाः  = 16;
स्थिर constexpr पूर्णाङ्क     GGML_METAL_MMA_NSG_MANY_पङ्क्तिकाः = 8;

संरचना ggml_metal_mma_tiling {
    पूर्णाङ्क nsg; // simdgroups per threadgroup, each over a slice of K
    पूर्णाङ्क nt;  // 8-row src0 tiles per threadgroup
    पूर्णाङ्क rt;  // 8-row src1 tiles per threadgroup
};

// halves n while it is above limit
स्थिर पूर्णाङ्क ggml_metal_halve_गन्तव्यम्_सीमा(पूर्णाङ्क n, int64_t सीमा) {
    यावत् (n > 1 && n > सीमा) {
        n /= 2;
    }
    प्रत्यागम n;
}

स्थिर size_t ggml_metal_mul_mv_mma_smem(पूर्णाङ्क nsg, पूर्णाङ्क nt, पूर्णाङ्क rt) {
    // one 8x8 float simdgroup matrix per output tile
    constexpr size_t tile_अष्टकानि = 8*8*परिमाण(भिन्न);
    प्रत्यागम (size_t) nsg*nt*rt*tile_अष्टकानि;
}

// fewer src0 rows need more simdgroups per threadgroup (a finer K split) to fill the GPU.
// the tiles become narrower when the K-slice reduction buffer is too large or when too few threadgroups fill the GPU.
स्थिर ggml_metal_mma_tiling ggml_metal_op_mul_mat_mma_tiling(ध्रुव ggml_tensor * op) {
    ध्रुव ggml_type type = op->src[0]->type;
    ध्रुव int64_t   ne00 = op->src[0]->ne[0];
    ध्रुव int64_t   ne01 = op->src[0]->ne[1];

    ggml_metal_mma_tiling res;
    res.rt = ggml_metal_mul_mv_mma_rt(op);

    ध्रुव int64_t n_चरणाः = ne00/ggml_metal_mul_mv_mma_k_चरणः(type, res.rt);

    पूर्णाङ्क nsg = GGML_METAL_MMA_NSG_MANY_पङ्क्तिकाः;
    यदि (ne01 <= GGML_METAL_MMA_FEW_पङ्क्तिकाः) {
        nsg = GGML_METAL_MMA_NSG_FEW_पङ्क्तिकाः;
    } अन्यथा यदि (ne01 <= GGML_METAL_MMA_MANY_पङ्क्तिकाः) {
        nsg = GGML_METAL_MMA_NSG_MID_पङ्क्तिकाः;
    }
    res.nsg = ggml_metal_halve_गन्तव्यम्_सीमा(nsg, n_चरणाः);

    ध्रुव int64_t nt_smem   = GGML_METAL_MMA_SMEM_MAX/(int64_t) ggml_metal_mul_mv_mma_smem(res.nsg, 1, res.rt);
    ध्रुव int64_t nt_पङ्क्तिकाः   = ne01/(GGML_METAL_MIN_THREADGROUPS*GGML_METAL_MMA_TILE);
    res.nt = ggml_metal_halve_गन्तव्यम्_सीमा(GGML_METAL_MMA_NT_MAX, std::min(nt_smem, nt_पङ्क्तिकाः));

    प्रत्यागम res;
}

// the pipeline for the tiling, with fewer simdgroups when the device cannot run that many threads
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto(ggml_metal_library_t lib, ध्रुव ggml_tensor * op, तार्किक सङ्कलय_2) {
    ggml_metal_mma_tiling tiling = ggml_metal_op_mul_mat_mma_tiling(op);

    स्वचालित pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, सङ्कलय_2);
    यावत् (tiling.nsg > 1 && ggml_metal_pipeline_max_theads_प्रत्येकम्_threadgroup(pipeline) < tiling.nsg*32) {
        tiling.nsg /= 2;
        pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, सङ्कलय_2);
    }

    प्रत्यागम pipeline;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma(ggml_metal_library_t lib, ध्रुव ggml_tensor * op, पूर्णाङ्क nsg, पूर्णाङ्क nt, पूर्णाङ्क rt, तार्किक सङ्कलय_2) {
    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव ggml_type tsrc0 = op->src[0]->type;
    ध्रुव ggml_type tsrc1 = op->src[1]->type;
    ध्रुव पूर्णाङ्क       ne12  = op->src[1]->ne[2];
    ध्रुव पूर्णाङ्क       r2    = ne12 / op->src[0]->ne[2];
    ध्रुव पूर्णाङ्क       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    // the specialized kernels unroll over a compile-time row length
    ध्रुव पूर्णाङ्क ne00 = ggml_metal_mul_mv_mma_विधा(tsrc0, rt) != GGML_METAL_MMA_विधा_GEN ? op->src[0]->ne[0] : 0;

    snprintf(आधारः, 256, "kernel_mul_mv_mma_%s_%s_nt%d_rt%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), nt, rt);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_ne00=%d_add=%d", आधारः, nsg, ne12, r2, r3, ne00, सङ्कलय_2);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int32(cv, ne00,           FC_MUL_MV_MMA + 4);
        ggml_metal_cv_समुच्चयः_int16(cv, nsg,            FC_MUL_MV_MMA + 0);
        ggml_metal_cv_समुच्चयः_int16(cv, (int16_t) ne12, FC_MUL_MV_MMA + 1);
        ggml_metal_cv_समुच्चयः_int16(cv, (int16_t) r2,   FC_MUL_MV_MMA + 2);
        ggml_metal_cv_समुच्चयः_int16(cv, (int16_t) r3,   FC_MUL_MV_MMA + 3);
        ggml_metal_cv_समुच्चयः_bool (cv, सङ्कलय_2,            FC_MUL_MV_MMA + 5);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = ggml_metal_mul_mv_mma_smem(nsg, nt, rt);

    res.nr0 = GGML_METAL_MMA_TILE*nt;
    res.nr1 = GGML_METAL_MMA_TILE*rt;

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव ggml_type tsrc0 = op->src[0]->type;
    ध्रुव ggml_type tsrc1 = op->src[1]->type;

    ध्रुव तार्किक bc_inp = op->src[0]->ne[0] % 32 != 0;

    constexpr पूर्णाङ्क NRA = SZ_SIMDGROUP * N_MM_खण्डम्_Y * N_MM_SIMD_गणः_Y;
    constexpr पूर्णाङ्क NRB = SZ_SIMDGROUP * N_MM_खण्डम्_X * N_MM_SIMD_गणः_X;

    ध्रुव तार्किक has_tensor = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_tensor;

    ध्रुव तार्किक bc_बहिर्गमनम् = has_tensor
        ? (op->ne[0] % NRA != 0 || op->ne[1] % NRB != 0)
        : (op->ne[0] % 64  != 0 || op->ne[1] % 32  != 0);

    GGML_ASSERT(op->src[1]->ne[2] <= INT16_MAX && op->src[1]->ne[3] <= INT16_MAX);
    ध्रुव int16_t ne12 = (int16_t) op->src[1]->ne[2];
    ध्रुव int16_t ne13 = (int16_t) op->src[1]->ne[3];
    ध्रुव int16_t r2   = (int16_t) (ne12 / op->src[0]->ne[2]);
    ध्रुव int16_t r3   = (int16_t) (ne13 / op->src[0]->ne[3]);

    snprintf(आधारः, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_bco=%d_ne12=%d_ne13=%d_r2=%d_r3=%d",
             आधारः, bc_inp, bc_बहिर्गमनम्, ne12, ne13, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_समुच्चयः_bool(cv, bc_बहिर्गमनम्, FC_MUL_MM + 1);
        ggml_metal_cv_समुच्चयः_int16(cv, ne12,  FC_MUL_MM + 2);
        ggml_metal_cv_समुच्चयः_int16(cv, ne13,  FC_MUL_MM + 3);
        ggml_metal_cv_समुच्चयः_int16(cv, r2,    FC_MUL_MM + 4);
        ggml_metal_cv_समुच्चयः_int16(cv, r3,    FC_MUL_MM + 5);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    यदि (has_tensor) {
        res.nr0 = NRA;
        res.nr1 = NRB;

        ध्रुव size_t smem_a = NRA * N_MM_NK_योग * परिमाण(ggml_fp16_t);
        res.smem = smem_a;
    } अन्यथा {
        res.nr0 = 64;
        res.nr1 = 32;

        res.smem = bc_बहिर्गमनम् ? 8192 : (4096 + 2048);
    }

    res.nsg = N_MM_SIMD_गणः_X * N_MM_SIMD_गणः_Y;

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    वर्ण आधारः[256];
    वर्ण name[256];

    पूर्णाङ्क nsg = 0; // number of simdgroups
    पूर्णाङ्क nr0 = 0; // number of src0 rows per simdgroup
    पूर्णाङ्क nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    ध्रुव ggml_type tsrc0 = op->src[0]->type;
    ध्रुव ggml_type tsrc1 = op->src[1]->type;

    ध्रुव वर्ण * suffix = "";

    तार्किक विभज = असत्य;

    // use custom matrix x vector kernel
    विकल्प (tsrc0) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
            {
                यदि (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } अन्यथा {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    smem = 32*परिमाण(भिन्न)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
            } भङ्ग;
        पक्ष GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } भङ्ग;
        पक्ष GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } भङ्ग;
        पक्ष GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } भङ्ग;
        पक्ष GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } भङ्ग;
        पक्ष GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } भङ्ग;
        पक्ष GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } भङ्ग;
        पक्ष GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*परिमाण(भिन्न)*N_R0_Q8_0;
            } भङ्ग;
        पक्ष GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*परिमाण(भिन्न);
            } भङ्ग;
        पक्ष GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } भङ्ग;
        पक्ष GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } भङ्ग;
        पक्ष GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } भङ्ग;
        पक्ष GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } भङ्ग;
        पक्ष GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } भङ्ग;
        पक्ष GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XXS_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XS_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;

                // split the rows across threads when there are fewer than 32 chunks per row
                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_XXS_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_S_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_S_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_S_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_M_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*परिमाण(भिन्न);
            } भङ्ग;
        पक्ष GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*परिमाण(भिन्न);
            } भङ्ग;
        पक्ष GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } भङ्ग;
        स्वभाव:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (पूर्णाङ्क) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    GGML_ASSERT(ne12 <= INT16_MAX && ne13 <= INT16_MAX);
    ध्रुव int16_t r2 = (int16_t) (ne12 / ne02);
    ध्रुव int16_t r3 = (int16_t) (ne13 / ne03);

    snprintf(आधारः, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_split=%d", आधारः, nsg, ne12, r2, r3, विभज);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_समुच्चयः_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_समुच्चयः_int16(cv, r2,             FC_MUL_MV + 3);
        ggml_metal_cv_समुच्चयः_int16(cv, r3,             FC_MUL_MV + 4);
        ggml_metal_cv_समुच्चयः_bool (cv, विभज,          FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib) {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_mul_mm_id_amax_part_f32");
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    res.smem = 32*परिमाण(भिन्न);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib) {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_mul_mm_id_amax_f32");
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    res.smem = 32*परिमाण(भिन्न);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, पूर्णाङ्क ne02, पूर्णाङ्क ne20) {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s_ne02=%d", आधारः, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_समुच्चयः_int32(cv, ne02, FC_MUL_MM + 7);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    // per expert: entry count + base offset, followed by the staging of the ids of ntg tokens
    res.smem = 2*(size_t) ne02*परिमाण(uint32_t) + (size_t) ne02*ne20*परिमाण(uint16_t);
    res.smem = GGML_PAD(res.smem, 16);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव ggml_type tsrc0 = op->src[0]->type;
    ध्रुव ggml_type tsrc1 = op->src[1]->type;

    ध्रुव तार्किक bc_inp = op->src[0]->ne[0] % 32 != 0;

    // src1 prec [TAG_GGML_PREC]
    ध्रुव तार्किक amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

    snprintf(आधारः, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_amax=%d", आधारः, bc_inp, amax);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_समुच्चयः_bool(cv, amax,   FC_MUL_MM + 6);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 8192;

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    वर्ण आधारः[256];
    वर्ण name[256];

    पूर्णाङ्क nsg = 0; // number of simdgroups
    पूर्णाङ्क nr0 = 0; // number of src0 rows per simdgroup
    पूर्णाङ्क nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    ध्रुव ggml_type tsrc0 = op->src[0]->type;
    ध्रुव ggml_type tsrc1 = op->src[1]->type;

    ध्रुव वर्ण * suffix = "";

    तार्किक विभज = असत्य;

        // use custom matrix x vector kernel
    विकल्प (tsrc0) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = 32*परिमाण(भिन्न)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } भङ्ग;
        पक्ष GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } भङ्ग;
        पक्ष GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } भङ्ग;
        पक्ष GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } भङ्ग;
        पक्ष GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } भङ्ग;
        पक्ष GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } भङ्ग;
        पक्ष GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } भङ्ग;
        पक्ष GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*परिमाण(भिन्न)*N_R0_Q8_0;
            } भङ्ग;
        पक्ष GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*परिमाण(भिन्न);
            } भङ्ग;
        पक्ष GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } भङ्ग;
        पक्ष GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } भङ्ग;
        पक्ष GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } भङ्ग;
        पक्ष GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } भङ्ग;
        पक्ष GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } भङ्ग;
        पक्ष GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XXS_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XS_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;

                // split the rows across threads when there are fewer than 32 chunks per row
                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_XXS_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_S_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_S_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_S_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;

                ध्रुव पूर्णाङ्क nb32 = ne00/32;
                यदि (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_M_विभज;
                    विभज = सत्य;
                }
            } भङ्ग;
        पक्ष GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*परिमाण(भिन्न);
            } भङ्ग;
        पक्ष GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*परिमाण(भिन्न);
            } भङ्ग;
        पक्ष GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } भङ्ग;
        स्वभाव:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (पूर्णाङ्क)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(आधारः, 256, "kernel_mul_mv_id_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_split=%d", आधारः, nsg, विभज);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_समुच्चयः_int16(cv, 1,   FC_MUL_MV + 2);
        ggml_metal_cv_समुच्चयः_int16(cv, 1,   FC_MUL_MV + 3);
        ggml_metal_cv_समुच्चयः_int16(cv, 1,   FC_MUL_MV + 4);
        ggml_metal_cv_समुच्चयः_bool (cv, विभज, FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_आकार(op->src[0]->type));

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    res.smem = 32*(परिमाण(भिन्न) + परिमाण(int32_t));

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part(ggml_metal_library_t lib) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, "kernel_argmax_part_f32");
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, "kernel_argmax_part_f32", "kernel_argmax_part_f32", शून्यसूचकः);
    }

    res.smem = 32*(परिमाण(भिन्न) + परिमाण(int32_t));

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final(ggml_metal_library_t lib) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, "kernel_argmax_final_f32");
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, "kernel_argmax_final_f32", "kernel_argmax_final_f32", शून्यसूचकः);
    }

    res.smem = 32*(परिमाण(भिन्न) + परिमाण(int32_t));

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_ARGSORT);

    वर्ण आधारः[256];
    वर्ण name[256];

    ggml_क्रमय_क्रमः क्रमः = (ggml_क्रमय_क्रमः) op->op_params[0];

    ध्रुव वर्ण * क्रमः_str = "undefined";
    विकल्प (क्रमः) {
        पक्ष GGML_क्रमय_क्रमः_ASC:  क्रमः_str = "asc";  भङ्ग;
        पक्ष GGML_क्रमय_क्रमः_DESC: क्रमः_str = "desc"; भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    };

    snprintf(आधारः, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), क्रमः_str);
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_ARGSORT);

    वर्ण आधारः[256];
    वर्ण name[256];

    ggml_क्रमय_क्रमः क्रमः = (ggml_क्रमय_क्रमः) op->op_params[0];

    ध्रुव वर्ण * क्रमः_str = "undefined";
    विकल्प (क्रमः) {
        पक्ष GGML_क्रमय_क्रमः_ASC:  क्रमः_str = "asc";  भङ्ग;
        पक्ष GGML_क्रमय_क्रमः_DESC: क्रमः_str = "desc"; भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    };

    snprintf(आधारः, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), क्रमः_str);
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht(ggml_metal_library_t lib, पूर्णाङ्क n, ggml_type tsrc) {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_fwht_%s_%d", ggml_type_name(tsrc), n);
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

// note: reuse the argsort kernel for the bitonic top_k fallback
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शिखरम्_k(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_शिखरम्_K);

    वर्ण आधारः[256];
    वर्ण name[256];

    // note: the top_k kernel is always descending order
    ggml_क्रमय_क्रमः क्रमः = GGML_क्रमय_क्रमः_DESC;

    ध्रुव वर्ण * क्रमः_str = "undefined";
    विकल्प (क्रमः) {
        पक्ष GGML_क्रमय_क्रमः_ASC:  क्रमः_str = "asc";  भङ्ग;
        पक्ष GGML_क्रमय_क्रमः_DESC: क्रमः_str = "desc"; भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    };

    snprintf(आधारः, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), क्रमः_str);
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शिखरम्_k_radix(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_शिखरम्_K);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_top_k_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शिखरम्_k_merge(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_शिखरम्_K);

    वर्ण आधारः[256];
    वर्ण name[256];

    ggml_क्रमय_क्रमः क्रमः = GGML_क्रमय_क्रमः_DESC;

    ध्रुव वर्ण * क्रमः_str = "undefined";
    विकल्प (क्रमः) {
        पक्ष GGML_क्रमय_क्रमः_ASC:  क्रमः_str = "asc";  भङ्ग;
        पक्ष GGML_क्रमय_क्रमः_DESC: क्रमः_str = "desc"; भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    };

    snprintf(आधारः, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), क्रमः_str);
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe(
        ggml_metal_library_t lib, int32_t n_expert, int32_t शिखरम्_k, तार्किक with_norm) {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_topk_moe_f32");
    snprintf(name, 256, "%s_n_expert=%d_top_k=%d_with_norm=%d", आधारः, n_expert, शिखरम्_k, with_norm ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_समुच्चयः_bool (cv, with_norm, FC_TOPK_MOE + 0);
        ggml_metal_cv_समुच्चयः_int32(cv, n_expert,  FC_TOPK_MOE + 1);
        ggml_metal_cv_समुच्चयः_int32(cv, शिखरम्_k,     FC_TOPK_MOE + 2);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce(ggml_metal_library_t lib, int32_t n_expert_प्रयुक्तम्) {
    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_moe_reduce_f32");
    snprintf(name, 256, "%s_n_expert_used=%d", आधारः, n_expert_प्रयुक्तम्);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_समुच्चयः_int32(cv, n_expert_प्रयुक्तम्, FC_MOE_REDUCE + 0);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ध्रुव संरचना ggml_tensor * op,
        तार्किक    has_आवरणम्,
        int32_t ncpsg) {
    प्रतिज्ञा(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_अप्रयुक्तम्(op);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_%s",
            "flash_attn_ext_pad");

    snprintf(name, 256, "%s_mask=%d_ncpsg=%d",
            आधारः,
            has_आवरणम्,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_bool(cv, has_आवरणम्,  FC_FLASH_ATTN_EXT_PAD + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_PAD + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_PAD + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_PAD + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_PAD + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_PAD + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_PAD + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_PAD + 23);
        //ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_PAD + 24);
        ggml_metal_cv_समुच्चयः_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_FLASH_ATTN_EXT);

    वर्ण आधारः[256];

    snprintf(आधारः, 256, "kernel_flash_attn_ext_kv_%s_f16", ggml_type_name(op->src[1]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, आधारः);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, आधारः, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ध्रुव संरचना ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg) {
    प्रतिज्ञा(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_अप्रयुक्तम्(op);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_%s",
            "flash_attn_ext_blk");

    snprintf(name, 256, "%s_nqptg=%d_ncpsg=%d",
            आधारः,
            nqptg,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        //ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_BLK + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_BLK + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_BLK + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_BLK + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_BLK + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_BLK + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_BLK + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_BLK + 23);
        ggml_metal_cv_समुच्चयः_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
        ggml_metal_cv_समुच्चयः_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ध्रुव ggml_tensor * op,
        तार्किक has_आवरणम्,
        तार्किक has_sinks,
        तार्किक has_bias,
        तार्किक has_scap) {
    प्रतिज्ञा(op->op == GGML_OP_FLASH_ATTN_EXT);

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव int32_t dk = (int32_t) op->src[1]->ne[0];
    ध्रुव int32_t dv = (int32_t) op->src[2]->ne[0];

    ध्रुव वर्ण * type = op->src[1]->type == GGML_TYPE_BF16 ? "bf16" : "f16";

    snprintf(आधारः, 256, "kernel_flash_attn_ext_tensor_%s_dk%d_dv%d", type, dk, dv);
    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d", आधारः, has_आवरणम्, has_sinks, has_bias, has_scap);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_bool(cv, has_आवरणम्,  FC_FLASH_ATTN_EXT_TENSOR + 0);
        ggml_metal_cv_समुच्चयः_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_TENSOR + 1);
        ggml_metal_cv_समुच्चयः_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_TENSOR + 2);
        ggml_metal_cv_समुच्चयः_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_TENSOR + 3);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ध्रुव ggml_tensor * op,
        तार्किक    has_आवरणम्,
        तार्किक    has_sinks,
        तार्किक    has_bias,
        तार्किक    has_scap,
        तार्किक    has_kvpad,
        int32_t nsg,
        तार्किक    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    प्रतिज्ञा(op->op == GGML_OP_FLASH_ATTN_EXT);

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव int32_t dk = (int32_t) op->src[1]->ne[0];
    ध्रुव int32_t dv = (int32_t) op->src[2]->ne[0];

    ध्रुव वर्ण * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    // do bounds checks for the mask?
    ध्रुव तार्किक bc_आवरणम् = op->src[3] && (op->src[3]->ne[1] % 8 != 0);

    snprintf(आधारः, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            type,
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
            आधारः,
            has_आवरणम्,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            bc_आवरणम्,
            ns10,
            ns20,
            nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_bool(cv, has_आवरणम्,  FC_FLASH_ATTN_EXT + 0);
        ggml_metal_cv_समुच्चयः_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        ggml_metal_cv_समुच्चयः_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
        ggml_metal_cv_समुच्चयः_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);
        ggml_metal_cv_समुच्चयः_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);

        ggml_metal_cv_समुच्चयः_bool(cv, bc_आवरणम्, FC_FLASH_ATTN_EXT + 10);

        ggml_metal_cv_समुच्चयः_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        ggml_metal_cv_समुच्चयः_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        ggml_metal_cv_समुच्चयः_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_FLASH_ATTN_EXT);
    प्रतिज्ञा(op->src[3]);

    वर्ण name[256];

    snprintf(name, 256, "kernel_flash_attn_ext_vec_idx");

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, शून्यसूचकः);
    }

    GGML_अप्रयुक्तम्(op);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ध्रुव ggml_tensor * op,
        तार्किक    has_आवरणम्,
        तार्किक    has_sinks,
        तार्किक    has_bias,
        तार्किक    has_scap,
        तार्किक    has_kvpad,
        तार्किक    has_sparse,
        int32_t nqpsg,
        int32_t ne,
        int32_t nsg,
        int32_t nwg,
        तार्किक    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    प्रतिज्ञा(op->op == GGML_OP_FLASH_ATTN_EXT);

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव int32_t dk = (int32_t) op->src[1]->ne[0];
    ध्रुव int32_t dv = (int32_t) op->src[2]->ne[0];

    ध्रुव वर्ण * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    वर्ण qne_suffix[16] = {0};
    यदि (!(nqpsg == 1 && ne == ggml_metal_tuning::fa_vec_baseline_ne(dk, dv))) {
        snprintf(qne_suffix, परिमाण(qne_suffix), "_q%d_ne%d", nqpsg, ne);
    }

    snprintf(आधारः, 256, "kernel_%s_%s_dk%d_dv%d%s",
            "flash_attn_ext_vec",
            type,
            dk,
            dv,
            qne_suffix);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_sparse=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            आधारः,
            has_आवरणम्,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            has_sparse,
            ns10,
            ns20,
            nsg, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_bool(cv, has_आवरणम्,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_समुच्चयः_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_समुच्चयः_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_समुच्चयः_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_समुच्चयः_bool(cv, has_kvpad,  FC_FLASH_ATTN_EXT_VEC + 4);
        ggml_metal_cv_समुच्चयः_bool(cv, has_sparse, FC_FLASH_ATTN_EXT_VEC + 5);

        ggml_metal_cv_समुच्चयः_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_समुच्चयः_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_समुच्चयः_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_समुच्चयः_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ध्रुव ggml_tensor * op,
        int32_t dv,
        int32_t nwg) {
    प्रतिज्ञा(op->op == GGML_OP_FLASH_ATTN_EXT);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "%s_dv=%d_nwg=%d", आधारः, dv, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        ggml_metal_cv_समुच्चयः_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;

    GGML_अप्रयुक्तम्(op);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, ध्रुव ggml_tensor * op, int32_t n_fuse) {
    वर्ण आधारः[256];
    वर्ण name[256];

    पूर्णाङ्क op_अङ्कः = -1;

    विकल्प (op->op) {
        पक्ष GGML_OP_सङ्कलय: op_अङ्कः = 0; भङ्ग;
        पक्ष GGML_OP_SUB: op_अङ्कः = 1; भङ्ग;
        पक्ष GGML_OP_MUL: op_अङ्कः = 2; भङ्ग;
        पक्ष GGML_OP_DIV: op_अङ्कः = 3; भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    };

    ध्रुव वर्ण * t0_str = ggml_type_name(op->src[0]->type);
    ध्रुव वर्ण * t1_str = ggml_type_name(op->src[1]->type);
    ध्रुव वर्ण * t_str  = ggml_type_name(op->type);

    ध्रुव तार्किक is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    ध्रुव तार्किक is_cb = op->src[0]->ne[0] != op->src[1]->ne[0];
    ध्रुव तार्किक is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(आधारः, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d_cb=%d", आधारः, op_अङ्कः, n_fuse, is_rb, is_cb);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, op_अङ्कः, FC_BIN + 0);
        ggml_metal_cv_समुच्चयः_int16(cv, n_fuse, FC_BIN + 1);
        ggml_metal_cv_समुच्चयः_bool (cv, is_rb,  FC_BIN + 2);
        ggml_metal_cv_समुच्चयः_bool (cv, is_cb,  FC_BIN + 3);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_rb;

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one(ggml_metal_library_t lib, ggml_op op) {
    वर्ण आधारः[256];
    वर्ण name[256];

    पूर्णाङ्क op_अङ्कः = -1;

    विकल्प (op) {
        पक्ष GGML_OP_सङ्कलय: op_अङ्कः = 0; भङ्ग;
        पक्ष GGML_OP_SUB: op_अङ्कः = 1; भङ्ग;
        पक्ष GGML_OP_MUL: op_अङ्कः = 2; भङ्ग;
        पक्ष GGML_OP_DIV: op_अङ्कः = 3; भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    };

    snprintf(आधारः, 256, "kernel_bin_fuse_%s_%s_%s", "f32", "f32", "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", आधारः, op_अङ्कः, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, op_अङ्कः, FC_BIN + 0);
        ggml_metal_cv_समुच्चयः_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_समुच्चयः_bool (cv, असत्य,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_L2_NORM);

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव तार्किक is_c4 = op->src[0]->ne[0] % 4 == 0;

    ध्रुव वर्ण * t0_str = ggml_type_name(op->src[0]->type);
    ध्रुव वर्ण * t_str  = ggml_type_name(op->type);

    snprintf(आधारः, 256, "kernel_l2_norm_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    res.c4   = is_c4;
    res.smem = 32*परिमाण(भिन्न);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_गणः_norm(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_गणः_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    res.smem = 32*परिमाण(भिन्न);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, ध्रुव ggml_tensor * op, पूर्णाङ्क n_fuse) {
    प्रतिज्ञा(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(op->src[0]));

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव वर्ण * suffix = "";
    यदि (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    विकल्प (op->op) {
        पक्ष GGML_OP_NORM:
            विकल्प (n_fuse) {
                पक्ष 1: snprintf(आधारः, 256, "kernel_norm_f32%s", suffix);         भङ्ग;
                पक्ष 2: snprintf(आधारः, 256, "kernel_norm_mul_f32%s", suffix);     भङ्ग;
                पक्ष 3: snprintf(आधारः, 256, "kernel_norm_mul_add_f32%s", suffix); भङ्ग;
                स्वभाव: GGML_ABORT("fatal error");
            } भङ्ग;
        पक्ष GGML_OP_RMS_NORM:
            विकल्प (n_fuse) {
                पक्ष 1: snprintf(आधारः, 256, "kernel_rms_norm_f32%s", suffix);         भङ्ग;
                पक्ष 2: snprintf(आधारः, 256, "kernel_rms_norm_mul_f32%s", suffix);     भङ्ग;
                पक्ष 3: snprintf(आधारः, 256, "kernel_rms_norm_mul_add_f32%s", suffix); भङ्ग;
                स्वभाव: GGML_ABORT("fatal error");
            } भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_समुच्चयः_bool(cv, असत्य, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*परिमाण(भिन्न);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(op->src[0]));

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव वर्ण * suffix = "";
    यदि (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    विकल्प (op->op) {
        पक्ष GGML_OP_NORM:     snprintf(आधारः, 256, "kernel_norm_mul_f32%s", suffix);     भङ्ग;
        पक्ष GGML_OP_RMS_NORM: snprintf(आधारः, 256, "kernel_rms_norm_mul_f32%s", suffix); भङ्ग;
        स्वभाव: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s_use_scale", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_समुच्चयः_bool(cv, सत्य, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*परिमाण(भिन्न);

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_ROPE || op->op == GGML_OP_ROPE_पृष्ठभागः);

    ध्रुव तार्किक is_पृष्ठभागः = op->op == GGML_OP_ROPE_पृष्ठभागः;

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव पूर्णाङ्क mode = ((ध्रुव int32_t *) op->op_params)[2];

    ध्रुव तार्किक is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    ध्रुव तार्किक is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    ध्रुव तार्किक is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ध्रुव तार्किक is_vision = mode == GGML_ROPE_TYPE_VISION;

    यदि (is_neox) {
        snprintf(आधारः, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } अन्यथा यदि ((is_mrope || is_imrope) && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(आधारः, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } अन्यथा यदि (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(आधारः, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } अन्यथा {
        snprintf(आधारः, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s_imrope=%d_is_back=%d", आधारः, is_imrope ? 1 : 0, is_पृष्ठभागः ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_bool(cv, is_imrope, FC_ROPE + 0);
        ggml_metal_cv_समुच्चयः_bool(cv, is_पृष्ठभागः,   FC_ROPE + 1);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_IM2COL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    ध्रुव तार्किक is_2D = ((ध्रुव int32_t *)(op->op_params))[6] == 1;
    ध्रुव int64_t KH = is_2D ? ne01 : 1;
    ध्रुव int64_t KW = ne00;

    वर्ण आधारः[256];
    वर्ण name[256];

    यदि (KH*KW <= 1024) {
        snprintf(आधारः, 256, "kernel_im2col_%s", ggml_type_name(op->type));
    } अन्यथा {
        snprintf(आधारः, 256, "kernel_im2col_ext_%s", ggml_type_name(op->type));
    }
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_COL2IM_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_col2im_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake(ggml_metal_library_t lib, गणना ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_snake_%s", ggml_type_name(type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_conv_transpose_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_CONV_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_conv_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw(ggml_metal_library_t lib, ध्रुव ggml_tensor * op, तार्किक tiled) {
    प्रतिज्ञा(op->op == GGML_OP_CONV_2D_DW);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_conv_2d_dw%s_%s_%s",
             tiled ? "_tiled" : "",
             ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_CONV_3D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_conv_3d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_UPSCALE);

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव int32_t बहुलकम्_flags = ggml_get_op_params_i32(op, 0);
    ध्रुव ggml_scale_बहुलकम् mode = (ggml_scale_बहुलकम्) (बहुलकम्_flags & 0xFF);

    ध्रुव तार्किक antialias = (बहुलकम्_flags & GGML_SCALE_FLAG_ANTIALIAS);

    यदि (mode == GGML_SCALE_बहुलकम्_BILINEAR) {
        snprintf(आधारः, 256, "kernel_upscale_bilinear_%s", ggml_type_name(op->src[0]->type));
    } अन्यथा यदि (mode == GGML_SCALE_बहुलकम्_BICUBIC) {
        snprintf(आधारः, 256, "kernel_upscale_bicubic_%s", ggml_type_name(op->src[0]->type));
    } अन्यथा {
        snprintf(आधारः, 256, "kernel_upscale_nearest_%s", ggml_type_name(op->src[0]->type));
    }
    snprintf(name, 256, "%s_aa=%d", आधारः, antialias);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_bool(cv, antialias, FC_UPSCALE + 0);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_ROLL);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_roll_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_PAD);

    वर्ण आधारः[256];
    वर्ण name[256];

    ध्रुव तार्किक circular = ggml_get_op_params_i32(op, 8) != 0;

    snprintf(आधारः, 256, "kernel_pad_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_circular=%d", आधारः, circular);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_bool(cv, circular, FC_PAD + 0);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_PAD_REFLECT_1D);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_ARANGE);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_चरणः_adamw(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_OPT_चरणः_ADAMW);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_चरणः_sgd(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_OPT_चरणः_SGD);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_पृष्ठभागः(ggml_metal_library_t lib, ध्रुव ggml_tensor * op) {
    प्रतिज्ञा(op->op == GGML_OP_SILU_पृष्ठभागः);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_silu_back_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, ध्रुव ggml_tensor *  op) {
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    वर्ण आधारः[256];
    वर्ण name[256];

    snprintf(आधारः, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", आधारः);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, शून्यसूचकः);
    }

    प्रत्यागम res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_गणनम्_equal(ggml_metal_library_t lib, ध्रुव ggml_tensor *  op) {
    प्रतिज्ञा(op->op == GGML_OP_गणनम्_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    वर्ण आधारः[256];
    वर्ण name[256];

    पूर्णाङ्क nsg = 1;
    यावत् (32*nsg < ne00 && nsg < 32) {
        nsg *= 2;
    }

    snprintf(आधारः, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", आधारः, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    यदि (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_समुच्चयः_int16(cv, nsg, FC_गणनम्_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, आधारः, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32 * परिमाण(int32_t);
    res.nsg  = nsg;

    प्रत्यागम res;
}
