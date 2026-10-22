#అన్తర్భావ "ggml-metal-device.h"
#అన్తర్భావ "ggml-metal-common.h"

#అన్తర్భావ "ggml-metal-impl.h"
#అన్తర్భావ "ggml-metal-tuning.h"

#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <cassert>
#అన్తర్భావ <memory>
#అన్తర్భావ <string>
#అన్తర్భావ <unordered_map>

నిర్మాణం ggml_metal_device_deleter {
    శూన్య క్రియాకారకం()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

రకనిర్వచనం std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

ggml_metal_device_t ggml_metal_device_get(పూర్ణాంకం device, పూర్ణాంకం n_devices) {
    స్థిర std::vector<ggml_metal_device_ptr> devs;

    devs.emplace_back(ggml_metal_device_init(device, n_devices));

    తిరిగి devs.back().get();
}

నిర్మాణం ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> దత్తాంశం;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(శూన్య) {
    ggml_metal_pipelines_t res = కొత్త ggml_metal_pipelines();

    తిరిగి res;
}

శూన్య ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    ఒకవేళ (!ppls) {
        తిరిగి;
    }

    కొరకు (స్వయంచాలక it = ppls->దత్తాంశం.begin(); it != ppls->దత్తాంశం.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    తొలగించు ppls;
}

శూన్య ggml_metal_pipelines_కూడు(ggml_metal_pipelines_t ppls, స్థిరరాశి అక్షరం * name, ggml_metal_pipeline_t pipeline) {
    ppls->దత్తాంశం[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, స్థిరరాశి అక్షరం * name) {
    ఒకవేళ (ppls->దత్తాంశం.find(name) == ppls->దత్తాంశం.end()) {
        తిరిగి శూన్యసూచిక;
    }

    తిరిగి ppls->దత్తాంశం[name];
}

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_పునాది(ggml_metal_library_t lib, ggml_op op) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి అక్షరం * op_str = "undefined";
    ఎంపిక (op) {
        స్థితి GGML_OP_కూడు_ID: op_str = "add_id"; ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    };

    snprintf(పునాది, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    స్థిరరాశి అక్షరం * pool_str = "undefined";
    ఎంపిక (op_pool) {
        స్థితి GGML_OP_POOL_AVG: pool_str = "avg"; ఆపు;
        స్థితి GGML_OP_POOL_MAX: pool_str = "max"; ఆపు;
        అప్రమేయ: GGML_ASSERT(అబద్ధం && "not implemented");
    };

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, పరిమాణం(పునాది), "kernel_pool_1d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, పరిమాణం(name), "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    స్థిరరాశి అక్షరం * pool_str = "undefined";
    ఎంపిక (op_pool) {
        స్థితి GGML_OP_POOL_AVG: pool_str = "avg"; ఆపు;
        స్థితి GGML_OP_POOL_MAX: pool_str = "max"; ఆపు;
        అప్రమేయ: GGML_ASSERT(అబద్ధం && "not implemented");
    };

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_అడ్డువరుసలు(ggml_metal_library_t lib, ggml_type tsrc) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_సమితి_అడ్డువరుసలు(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి స్వయంచాలక tsrc = op->src[0]->type;
    స్థిరరాశి స్వయంచాలక tidx = op->src[1]->type;
    స్థిరరాశి స్వయంచాలక tdst = op->type;

    snprintf(పునాది, 256, "kernel_set_rows_%s_%s_%s", ggml_type_name(tsrc), ggml_type_name(tidx), ggml_type_name(tdst));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి పూర్ణాంకం n = op->src[0]->ne[0];

    snprintf(పునాది, 256, "kernel_diag_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_n=%d", పునాది, n);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    res.nsg  = 1;
    res.smem = 0;

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat(ggml_metal_library_t lib, ggml_type tsrc) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat(ggml_metal_library_t lib, ggml_type tsrc) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_concat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    పూర్ణాంకం op_అంకె = -1;

    ఎంపిక (op->op) {
        స్థితి GGML_OP_SCALE:      op_అంకె = OP_UNARY_అంకె_SCALE;      ఆపు;
        స్థితి GGML_OP_FILL:       op_అంకె = OP_UNARY_అంకె_FILL;       ఆపు;
        స్థితి GGML_OP_CLAMP:      op_అంకె = OP_UNARY_అంకె_CLAMP;      ఆపు;
        స్థితి GGML_OP_SQR:        op_అంకె = OP_UNARY_అంకె_SQR;        ఆపు;
        స్థితి GGML_OP_SQRT:       op_అంకె = OP_UNARY_అంకె_SQRT;       ఆపు;
        స్థితి GGML_OP_SIN:        op_అంకె = OP_UNARY_అంకె_SIN;        ఆపు;
        స్థితి GGML_OP_COS:        op_అంకె = OP_UNARY_అంకె_COS;        ఆపు;
        స్థితి GGML_OP_LOG:        op_అంకె = OP_UNARY_అంకె_LOG;        ఆపు;
        స్థితి GGML_OP_LEAKY_RELU: op_అంకె = OP_UNARY_అంకె_LEAKY_RELU; ఆపు;
        స్థితి GGML_OP_UNARY:
            ఎంపిక (ggml_get_unary_op(op)) {
                స్థితి GGML_UNARY_OP_TANH:        op_అంకె = OP_UNARY_అంకె_TANH;        ఆపు;
                స్థితి GGML_UNARY_OP_RELU:        op_అంకె = OP_UNARY_అంకె_RELU;        ఆపు;
                స్థితి GGML_UNARY_OP_SIGMOID:     op_అంకె = OP_UNARY_అంకె_SIGMOID;     ఆపు;
                స్థితి GGML_UNARY_OP_GELU:        op_అంకె = OP_UNARY_అంకె_GELU;        ఆపు;
                స్థితి GGML_UNARY_OP_GELU_ERF:    op_అంకె = OP_UNARY_అంకె_GELU_ERF;    ఆపు;
                స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:  op_అంకె = OP_UNARY_అంకె_GELU_శీఘ్ర;  ఆపు;
                స్థితి GGML_UNARY_OP_SILU:        op_అంకె = OP_UNARY_అంకె_SILU;        ఆపు;
                స్థితి GGML_UNARY_OP_ELU:         op_అంకె = OP_UNARY_అంకె_ELU;         ఆపు;
                స్థితి GGML_UNARY_OP_NEG:         op_అంకె = OP_UNARY_అంకె_NEG;         ఆపు;
                స్థితి GGML_UNARY_OP_ABS:         op_అంకె = OP_UNARY_అంకె_ABS;         ఆపు;
                స్థితి GGML_UNARY_OP_SGN:         op_అంకె = OP_UNARY_అంకె_SGN;         ఆపు;
                స్థితి GGML_UNARY_OP_అడుగు:        op_అంకె = OP_UNARY_అంకె_అడుగు;        ఆపు;
                స్థితి GGML_UNARY_OP_HARDSWISH:   op_అంకె = OP_UNARY_అంకె_HARDSWISH;   ఆపు;
                స్థితి GGML_UNARY_OP_HARDSIGMOID: op_అంకె = OP_UNARY_అంకె_HARDSIGMOID; ఆపు;
                స్థితి GGML_UNARY_OP_EXP:         op_అంకె = OP_UNARY_అంకె_EXP;         ఆపు;
                స్థితి GGML_UNARY_OP_SOFTPLUS:    op_అంకె = OP_UNARY_అంకె_SOFTPLUS;    ఆపు;
                స్థితి GGML_UNARY_OP_EXPM1:       op_అంకె = OP_UNARY_అంకె_EXPM1;       ఆపు;
                స్థితి GGML_UNARY_OP_FLOOR:       op_అంకె = OP_UNARY_అంకె_FLOOR;       ఆపు;
                స్థితి GGML_UNARY_OP_CEIL:        op_అంకె = OP_UNARY_అంకె_CEIL;        ఆపు;
                స్థితి GGML_UNARY_OP_విడత:       op_అంకె = OP_UNARY_అంకె_విడత;       ఆపు;
                స్థితి GGML_UNARY_OP_TRUNC:       op_అంకె = OP_UNARY_అంకె_TRUNC;       ఆపు;
                స్థితి GGML_UNARY_OP_XIELU:       op_అంకె = OP_UNARY_అంకె_XIELU;       ఆపు;
                అప్రమేయ: GGML_ABORT("fatal error");
            } ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    };

    స్థిరరాశి అక్షరం * t0_str = ggml_type_name(op->src[0]->type);
    స్థిరరాశి అక్షరం * t_str  = ggml_type_name(op->type);

    స్థిరరాశి తార్కిక is_c4 = op->src[0]->ne[0] % 4 == 0;
    స్థిరరాశి తార్కిక is_cnt = ggml_is_contiguous(op->src[0]) && ggml_nelements(op) < 32768;

    snprintf(పునాది, 256, "kernel_unary_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_cnt=%d", పునాది, op_అంకె, is_cnt);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, op_అంకె, FC_UNARY + 0);
        ggml_metal_cv_సమితి_bool (cv, is_cnt, FC_UNARY + 1);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_cnt;

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి అక్షరం * op_str = "undefined";
    ఎంపిక (op->op) {
        స్థితి GGML_OP_GLU:
            ఎంపిక (ggml_get_glu_op(op)) {
                స్థితి GGML_GLU_OP_REGLU:        op_str = "reglu";        ఆపు;
                స్థితి GGML_GLU_OP_GEGLU:        op_str = "geglu";        ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   ఆపు;
                స్థితి GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    ఆపు;
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:  op_str = "geglu_quick";  ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP: op_str = "swiglu_clamp"; ఆపు;
                అప్రమేయ: GGML_ABORT("fatal error");
            } ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    };

    snprintf(పునాది, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_కూడిక(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_కూడిక);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_op_sum_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_కూడిక_అడ్డువరుసలు(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(op->src[0]));

    అక్షరం పునాది[256];
    అక్షరం name[256];

    పూర్ణాంకం op_అంకె = -1;

    ఎంపిక (op->op) {
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు: op_అంకె = OP_కూడిక_అడ్డువరుసలు_అంకె_కూడిక_అడ్డువరుసలు; ఆపు;
        స్థితి GGML_OP_సగటు:     op_అంకె = OP_కూడిక_అడ్డువరుసలు_అంకె_సగటు;     ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    };

    స్థిరరాశి అక్షరం * t0_str = ggml_type_name(op->src[0]->type);
    స్థిరరాశి అక్షరం * t_str  = ggml_type_name(op->type);

    స్థిరరాశి తార్కిక is_c4 = op->src[0]->ne[0] % 4 == 0;

    snprintf(పునాది, 256, "kernel_sum_rows_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d", పునాది, op_అంకె);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, op_అంకె, FC_కూడిక_అడ్డువరుసలు + 0);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*పరిమాణం(దశాంశం);

    ఒకవేళ (is_c4) {
        res.smem *= 4;
    }

    res.c4  = is_c4;

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_cumsum_blk_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_కూడు(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_cumsum_add_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_TRI);
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_కొలత(op->src[0]->type));

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి అక్షరం * op_str = "tri";
    స్థిరరాశి పూర్ణాంకం ttype = op->op_params[0];

    snprintf(పునాది, 256, "kernel_%s_%s_%d", op_str, ggml_type_name(op->src[0]->type), ttype);

    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి అక్షరం * suffix = "";

    ఒకవేళ (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    స్థిరరాశి ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(పునాది, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    res.smem = 32*పరిమాణం(దశాంశం);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer(
        ggml_metal_library_t lib,
        స్థిరరాశి ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి int16_t nh = op->src[0]->ne[1];

    snprintf(పునాది, 256, "kernel_lightning_indexer_%s", ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s_nh=%d", పునాది, nh);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, nh, FC_LIGHTNING_INDEXER + 0);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    అక్షరం name[256];
    స్థిరరాశి అక్షరం * పునాది = శూన్యసూచిక;

    ఎంపిక (op->op) {
        స్థితి GGML_OP_DSV4_HC_COMB:
            పునాది = "kernel_dsv4_hc_comb_f32";
            snprintf(name, 256, "%s", పునాది);
            ఆపు;
        స్థితి GGML_OP_DSV4_HC_PRE:
            పునాది = ggml_get_op_params_i32(op, 1) != 0 ? "kernel_dsv4_hc_pre_gated_f32" : "kernel_dsv4_hc_pre_f32";
            snprintf(name, 256, "%s_n_hc=%d", పునాది, (పూర్ణాంకం) op->src[0]->ne[1]);
            ఆపు;
        స్థితి GGML_OP_DSV4_HC_POST:
            పునాది = op->src[3] ? "kernel_dsv4_hc_post_f32" : "kernel_dsv4_hc_post_nocomb_f32";
            snprintf(name, 256, "%s", పునాది);
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("fatal error");
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = శూన్యసూచిక;

        ఒకవేళ (op->op == GGML_OP_DSV4_HC_PRE) {
            cv = ggml_metal_cv_init();
            ggml_metal_cv_సమితి_int32(cv, (int32_t) op->src[0]->ne[1], FC_DSV4_HC + 0);
        }

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ఒకవేళ (cv) {
            ggml_metal_cv_free(cv);
        }
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv(
        ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, int32_t nc, తార్కిక use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి అక్షరం * suffix = "";

    ఒకవేళ (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(పునాది, 256, "kernel_ssm_conv_%s_%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_nc=%d_silu=%d", పునాది, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_సమితి_bool(cv, use_silu, FC_SSM_CONV + 1);
        ggml_metal_cv_సమితి_int32(cv, nc,      FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched(
        ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, పూర్ణాంకం ssm_conv_bs, int32_t nc, తార్కిక use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి అక్షరం * suffix = "";
    ఒకవేళ (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(పునాది, 256, "kernel_ssm_conv_%s_%s_batched%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_ssm_conv_bs=%d_nc=%d_silu=%d", పునాది, ssm_conv_bs, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, ssm_conv_bs, FC_SSM_CONV + 0);
        ggml_metal_cv_సమితి_bool(cv, use_silu,     FC_SSM_CONV + 1);
        ggml_metal_cv_సమితి_int32(cv, nc,          FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_పరిశీలన(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, తార్కిక తోక)  {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి పూర్ణాంకం nsg = (ne00 + 31)/32;

    snprintf(పునాది, 256, "kernel_ssm_scan_%s%s", ggml_type_name(op->src[0]->type), తోక ? "_tail" : "");
    snprintf(name, 256, "%s_nsg=%d", పునాది, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (32 + 2) floats
    res.smem = GGML_PAD((32 + 2)*పరిమాణం(దశాంశం)*nsg, 16);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_పరిశీలన_ssd_mma(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op)  {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_ssm_scan_ssd_mma_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    // acs/exp(acs)/state-decay vectors + dtX + SAM rows + two 8x8 tiles per simdgroup
    res.smem = (3*OP_SSM_పరిశీలన_SSD_CS +
                OP_SSM_పరిశీలన_SSD_CS*OP_SSM_పరిశీలన_SSD_HD +
                OP_SSM_పరిశీలన_SSD_NSG*8*OP_SSM_పరిశీలన_SSD_CS +
                OP_SSM_పరిశీలన_SSD_NSG*2*8*8)*పరిమాణం(దశాంశం);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి int64_t C = op->ne[0];
    స్థిరరాశి int64_t H = op->src[0]->ne[1];

    ఎంపిక (op->op) {
        స్థితి GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(పునాది, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } ఆపు;
        స్థితి GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(పునాది, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } ఆపు;
        అప్రమేయ:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    // v is src[2], dimensions: S_v = ne[0], H = ne[1]
    స్థిరరాశి పూర్ణాంకం ne20 = op->src[2]->ne[0]; // S_v
    స్థిరరాశి పూర్ణాంకం ne21 = op->src[2]->ne[1]; // H
    స్థిరరాశి పూర్ణాంకం ne30 = op->src[3]->ne[0]; // G
    // state is src[5], 4D [S_v, S_v, H_v, n_seqs] (s0 only); K is op param 0.
    స్థిరరాశి పూర్ణాంకం K = ggml_get_op_params_i32(op, 0);

    స్థిరరాశి పూర్ణాంకం nsg = op->src[2]->ne[0]/32;

    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->ne[0] == ne20 * ne21);
    GGML_ASSERT(ne20 % 32 == 0);

    snprintf(పునాది, 256, "kernel_gated_delta_net_%s_%d", ggml_type_name(op->src[0]->type), nsg);
    snprintf(name, 256, "%s_ne20=%d_ne30=%d_K=%d", పునాది, ne20, ne30, K);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, ne20, FC_GATED_DELTA_NET + 0);
        ggml_metal_cv_సమితి_int16(cv, ne30, FC_GATED_DELTA_NET + 1);
        ggml_metal_cv_సమితి_int16(cv, K,    FC_GATED_DELTA_NET + 2);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg = nsg;

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి పూర్ణాంకం nsg = 8;
    స్థిరరాశి పూర్ణాంకం n   = op->src[1]->ne[1];
    స్థిరరాశి పూర్ణాంకం k   = op->src[1]->ne[0];

    snprintf(పునాది, 256, "kernel_solve_tri_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d_n=%d_k=%d", పునాది, nsg, n, k);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, nsg, FC_SOLVE_TRI + 0);
        ggml_metal_cv_సమితి_int16(cv, n,   FC_SOLVE_TRI + 1);
        ggml_metal_cv_సమితి_int16(cv, k,   FC_SOLVE_TRI + 2);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = GGML_PAD(GGML_PAD(n, 32)*nsg*పరిమాణం(దశాంశం), 16);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, పూర్ణాంకం nsg, పూర్ణాంకం nxpsg, పూర్ణాంకం r1ptg) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి ggml_type tsrc0 = op->src[0]->type;
    స్థిరరాశి ggml_type tsrc1 = op->src[1]->type;
    స్థిరరాశి పూర్ణాంకం       ne12  = op->src[1]->ne[2];
    స్థిరరాశి పూర్ణాంకం       r2    = ne12 / op->src[0]->ne[2];
    స్థిరరాశి పూర్ణాంకం       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    snprintf(పునాది, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d_ne12=%d_r2=%d_r3=%d", పునాది, nsg, nxpsg, ne12, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_సమితి_int16(cv, nxpsg,          FC_MUL_MV + 1);
        ggml_metal_cv_సమితి_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_సమితి_int16(cv, (int16_t) r2,   FC_MUL_MV + 3);
        ggml_metal_cv_సమితి_int16(cv, (int16_t) r3,   FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

// threadgroups needed to fill the GPU
// TODO: dedup
స్థిర constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

// few-row MMA mat-mul: 8x8 simdgroup matrices for the src1 rows of ggml_metal_mul_mat_use_mma
స్థిర constexpr పూర్ణాంకం     GGML_METAL_MMA_TILE          = 8;
స్థిర constexpr పూర్ణాంకం     GGML_METAL_MMA_NT_MAX        = 4;
స్థిర constexpr పూర్ణాంకం     GGML_METAL_MMA_SMEM_MAX      = 16384;

// simdgroups per threadgroup for at most FEW_ROWS, at most MANY_ROWS, and more src0 rows
స్థిర constexpr int64_t GGML_METAL_MMA_FEW_అడ్డువరుసలు      = 64;
స్థిర constexpr int64_t GGML_METAL_MMA_MANY_అడ్డువరుసలు     = 6144;
స్థిర constexpr పూర్ణాంకం     GGML_METAL_MMA_NSG_FEW_అడ్డువరుసలు  = 32;
స్థిర constexpr పూర్ణాంకం     GGML_METAL_MMA_NSG_MID_అడ్డువరుసలు  = 16;
స్థిర constexpr పూర్ణాంకం     GGML_METAL_MMA_NSG_MANY_అడ్డువరుసలు = 8;

నిర్మాణం ggml_metal_mma_tiling {
    పూర్ణాంకం nsg; // simdgroups per threadgroup, each over a slice of K
    పూర్ణాంకం nt;  // 8-row src0 tiles per threadgroup
    పూర్ణాంకం rt;  // 8-row src1 tiles per threadgroup
};

// halves n while it is above limit
స్థిర పూర్ణాంకం ggml_metal_halve_గమ్యం_హద్దు(పూర్ణాంకం n, int64_t హద్దు) {
    అయితే (n > 1 && n > హద్దు) {
        n /= 2;
    }
    తిరిగి n;
}

స్థిర size_t ggml_metal_mul_mv_mma_smem(పూర్ణాంకం nsg, పూర్ణాంకం nt, పూర్ణాంకం rt) {
    // one 8x8 float simdgroup matrix per output tile
    constexpr size_t tile_బైట్లు = 8*8*పరిమాణం(దశాంశం);
    తిరిగి (size_t) nsg*nt*rt*tile_బైట్లు;
}

// fewer src0 rows need more simdgroups per threadgroup (a finer K split) to fill the GPU.
// the tiles become narrower when the K-slice reduction buffer is too large or when too few threadgroups fill the GPU.
స్థిర ggml_metal_mma_tiling ggml_metal_op_mul_mat_mma_tiling(స్థిరరాశి ggml_tensor * op) {
    స్థిరరాశి ggml_type type = op->src[0]->type;
    స్థిరరాశి int64_t   ne00 = op->src[0]->ne[0];
    స్థిరరాశి int64_t   ne01 = op->src[0]->ne[1];

    ggml_metal_mma_tiling res;
    res.rt = ggml_metal_mul_mv_mma_rt(op);

    స్థిరరాశి int64_t n_అడుగులు = ne00/ggml_metal_mul_mv_mma_k_అడుగు(type, res.rt);

    పూర్ణాంకం nsg = GGML_METAL_MMA_NSG_MANY_అడ్డువరుసలు;
    ఒకవేళ (ne01 <= GGML_METAL_MMA_FEW_అడ్డువరుసలు) {
        nsg = GGML_METAL_MMA_NSG_FEW_అడ్డువరుసలు;
    } లేకపోతే ఒకవేళ (ne01 <= GGML_METAL_MMA_MANY_అడ్డువరుసలు) {
        nsg = GGML_METAL_MMA_NSG_MID_అడ్డువరుసలు;
    }
    res.nsg = ggml_metal_halve_గమ్యం_హద్దు(nsg, n_అడుగులు);

    స్థిరరాశి int64_t nt_smem   = GGML_METAL_MMA_SMEM_MAX/(int64_t) ggml_metal_mul_mv_mma_smem(res.nsg, 1, res.rt);
    స్థిరరాశి int64_t nt_అడ్డువరుసలు   = ne01/(GGML_METAL_MIN_THREADGROUPS*GGML_METAL_MMA_TILE);
    res.nt = ggml_metal_halve_గమ్యం_హద్దు(GGML_METAL_MMA_NT_MAX, std::min(nt_smem, nt_అడ్డువరుసలు));

    తిరిగి res;
}

// the pipeline for the tiling, with fewer simdgroups when the device cannot run that many threads
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, తార్కిక కూడు_2) {
    ggml_metal_mma_tiling tiling = ggml_metal_op_mul_mat_mma_tiling(op);

    స్వయంచాలక pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, కూడు_2);
    అయితే (tiling.nsg > 1 && ggml_metal_pipeline_max_theads_చొప్పున_threadgroup(pipeline) < tiling.nsg*32) {
        tiling.nsg /= 2;
        pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, కూడు_2);
    }

    తిరిగి pipeline;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, పూర్ణాంకం nsg, పూర్ణాంకం nt, పూర్ణాంకం rt, తార్కిక కూడు_2) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి ggml_type tsrc0 = op->src[0]->type;
    స్థిరరాశి ggml_type tsrc1 = op->src[1]->type;
    స్థిరరాశి పూర్ణాంకం       ne12  = op->src[1]->ne[2];
    స్థిరరాశి పూర్ణాంకం       r2    = ne12 / op->src[0]->ne[2];
    స్థిరరాశి పూర్ణాంకం       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    // the specialized kernels unroll over a compile-time row length
    స్థిరరాశి పూర్ణాంకం ne00 = ggml_metal_mul_mv_mma_రకం(tsrc0, rt) != GGML_METAL_MMA_రకం_GEN ? op->src[0]->ne[0] : 0;

    snprintf(పునాది, 256, "kernel_mul_mv_mma_%s_%s_nt%d_rt%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), nt, rt);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_ne00=%d_add=%d", పునాది, nsg, ne12, r2, r3, ne00, కూడు_2);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int32(cv, ne00,           FC_MUL_MV_MMA + 4);
        ggml_metal_cv_సమితి_int16(cv, nsg,            FC_MUL_MV_MMA + 0);
        ggml_metal_cv_సమితి_int16(cv, (int16_t) ne12, FC_MUL_MV_MMA + 1);
        ggml_metal_cv_సమితి_int16(cv, (int16_t) r2,   FC_MUL_MV_MMA + 2);
        ggml_metal_cv_సమితి_int16(cv, (int16_t) r3,   FC_MUL_MV_MMA + 3);
        ggml_metal_cv_సమితి_bool (cv, కూడు_2,            FC_MUL_MV_MMA + 5);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = ggml_metal_mul_mv_mma_smem(nsg, nt, rt);

    res.nr0 = GGML_METAL_MMA_TILE*nt;
    res.nr1 = GGML_METAL_MMA_TILE*rt;

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి ggml_type tsrc0 = op->src[0]->type;
    స్థిరరాశి ggml_type tsrc1 = op->src[1]->type;

    స్థిరరాశి తార్కిక bc_inp = op->src[0]->ne[0] % 32 != 0;

    constexpr పూర్ణాంకం NRA = SZ_SIMDGROUP * N_MM_ఖండం_Y * N_MM_SIMD_సమూహం_Y;
    constexpr పూర్ణాంకం NRB = SZ_SIMDGROUP * N_MM_ఖండం_X * N_MM_SIMD_సమూహం_X;

    స్థిరరాశి తార్కిక has_tensor = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_tensor;

    స్థిరరాశి తార్కిక bc_బహిర్గతం = has_tensor
        ? (op->ne[0] % NRA != 0 || op->ne[1] % NRB != 0)
        : (op->ne[0] % 64  != 0 || op->ne[1] % 32  != 0);

    GGML_ASSERT(op->src[1]->ne[2] <= INT16_MAX && op->src[1]->ne[3] <= INT16_MAX);
    స్థిరరాశి int16_t ne12 = (int16_t) op->src[1]->ne[2];
    స్థిరరాశి int16_t ne13 = (int16_t) op->src[1]->ne[3];
    స్థిరరాశి int16_t r2   = (int16_t) (ne12 / op->src[0]->ne[2]);
    స్థిరరాశి int16_t r3   = (int16_t) (ne13 / op->src[0]->ne[3]);

    snprintf(పునాది, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_bco=%d_ne12=%d_ne13=%d_r2=%d_r3=%d",
             పునాది, bc_inp, bc_బహిర్గతం, ne12, ne13, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_సమితి_bool(cv, bc_బహిర్గతం, FC_MUL_MM + 1);
        ggml_metal_cv_సమితి_int16(cv, ne12,  FC_MUL_MM + 2);
        ggml_metal_cv_సమితి_int16(cv, ne13,  FC_MUL_MM + 3);
        ggml_metal_cv_సమితి_int16(cv, r2,    FC_MUL_MM + 4);
        ggml_metal_cv_సమితి_int16(cv, r3,    FC_MUL_MM + 5);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    ఒకవేళ (has_tensor) {
        res.nr0 = NRA;
        res.nr1 = NRB;

        స్థిరరాశి size_t smem_a = NRA * N_MM_NK_మొత్తం * పరిమాణం(ggml_fp16_t);
        res.smem = smem_a;
    } లేకపోతే {
        res.nr0 = 64;
        res.nr1 = 32;

        res.smem = bc_బహిర్గతం ? 8192 : (4096 + 2048);
    }

    res.nsg = N_MM_SIMD_సమూహం_X * N_MM_SIMD_సమూహం_Y;

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    పూర్ణాంకం nsg = 0; // number of simdgroups
    పూర్ణాంకం nr0 = 0; // number of src0 rows per simdgroup
    పూర్ణాంకం nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    స్థిరరాశి ggml_type tsrc0 = op->src[0]->type;
    స్థిరరాశి ggml_type tsrc1 = op->src[1]->type;

    స్థిరరాశి అక్షరం * suffix = "";

    తార్కిక విభజించు = అబద్ధం;

    // use custom matrix x vector kernel
    ఎంపిక (tsrc0) {
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
            {
                ఒకవేళ (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } లేకపోతే {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    smem = 32*పరిమాణం(దశాంశం)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
            } ఆపు;
        స్థితి GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } ఆపు;
        స్థితి GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } ఆపు;
        స్థితి GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } ఆపు;
        స్థితి GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } ఆపు;
        స్థితి GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } ఆపు;
        స్థితి GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*పరిమాణం(దశాంశం)*N_R0_Q8_0;
            } ఆపు;
        స్థితి GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*పరిమాణం(దశాంశం);
            } ఆపు;
        స్థితి GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } ఆపు;
        స్థితి GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } ఆపు;
        స్థితి GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } ఆపు;
        స్థితి GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } ఆపు;
        స్థితి GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } ఆపు;
        స్థితి GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XXS_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XS_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;

                // split the rows across threads when there are fewer than 32 chunks per row
                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_XXS_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_S_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_S_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_S_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_M_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*పరిమాణం(దశాంశం);
            } ఆపు;
        స్థితి GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*పరిమాణం(దశాంశం);
            } ఆపు;
        స్థితి GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } ఆపు;
        అప్రమేయ:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (పూర్ణాంకం) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    GGML_ASSERT(ne12 <= INT16_MAX && ne13 <= INT16_MAX);
    స్థిరరాశి int16_t r2 = (int16_t) (ne12 / ne02);
    స్థిరరాశి int16_t r3 = (int16_t) (ne13 / ne03);

    snprintf(పునాది, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_split=%d", పునాది, nsg, ne12, r2, r3, విభజించు);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_సమితి_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_సమితి_int16(cv, r2,             FC_MUL_MV + 3);
        ggml_metal_cv_సమితి_int16(cv, r3,             FC_MUL_MV + 4);
        ggml_metal_cv_సమితి_bool (cv, విభజించు,          FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_mul_mm_id_amax_part_f32");
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    res.smem = 32*పరిమాణం(దశాంశం);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_mul_mm_id_amax_f32");
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    res.smem = 32*పరిమాణం(దశాంశం);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, పూర్ణాంకం ne02, పూర్ణాంకం ne20) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s_ne02=%d", పునాది, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_సమితి_int32(cv, ne02, FC_MUL_MM + 7);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    // per expert: entry count + base offset, followed by the staging of the ids of ntg tokens
    res.smem = 2*(size_t) ne02*పరిమాణం(uint32_t) + (size_t) ne02*ne20*పరిమాణం(uint16_t);
    res.smem = GGML_PAD(res.smem, 16);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి ggml_type tsrc0 = op->src[0]->type;
    స్థిరరాశి ggml_type tsrc1 = op->src[1]->type;

    స్థిరరాశి తార్కిక bc_inp = op->src[0]->ne[0] % 32 != 0;

    // src1 prec [TAG_GGML_PREC]
    స్థిరరాశి తార్కిక amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

    snprintf(పునాది, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_amax=%d", పునాది, bc_inp, amax);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_సమితి_bool(cv, amax,   FC_MUL_MM + 6);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 8192;

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    పూర్ణాంకం nsg = 0; // number of simdgroups
    పూర్ణాంకం nr0 = 0; // number of src0 rows per simdgroup
    పూర్ణాంకం nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    స్థిరరాశి ggml_type tsrc0 = op->src[0]->type;
    స్థిరరాశి ggml_type tsrc1 = op->src[1]->type;

    స్థిరరాశి అక్షరం * suffix = "";

    తార్కిక విభజించు = అబద్ధం;

        // use custom matrix x vector kernel
    ఎంపిక (tsrc0) {
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = 32*పరిమాణం(దశాంశం)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } ఆపు;
        స్థితి GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } ఆపు;
        స్థితి GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } ఆపు;
        స్థితి GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } ఆపు;
        స్థితి GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } ఆపు;
        స్థితి GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } ఆపు;
        స్థితి GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*పరిమాణం(దశాంశం)*N_R0_Q8_0;
            } ఆపు;
        స్థితి GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*పరిమాణం(దశాంశం);
            } ఆపు;
        స్థితి GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } ఆపు;
        స్థితి GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } ఆపు;
        స్థితి GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } ఆపు;
        స్థితి GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } ఆపు;
        స్థితి GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } ఆపు;
        స్థితి GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XXS_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XS_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;

                // split the rows across threads when there are fewer than 32 chunks per row
                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_XXS_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_S_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_S_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_S_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;

                స్థిరరాశి పూర్ణాంకం nb32 = ne00/32;
                ఒకవేళ (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_M_విభజించు;
                    విభజించు = నిజం;
                }
            } ఆపు;
        స్థితి GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*పరిమాణం(దశాంశం);
            } ఆపు;
        స్థితి GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*పరిమాణం(దశాంశం);
            } ఆపు;
        స్థితి GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } ఆపు;
        అప్రమేయ:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (పూర్ణాంకం)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(పునాది, 256, "kernel_mul_mv_id_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_split=%d", పునాది, nsg, విభజించు);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_సమితి_int16(cv, 1,   FC_MUL_MV + 2);
        ggml_metal_cv_సమితి_int16(cv, 1,   FC_MUL_MV + 3);
        ggml_metal_cv_సమితి_int16(cv, 1,   FC_MUL_MV + 4);
        ggml_metal_cv_సమితి_bool (cv, విభజించు, FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_కొలత(op->src[0]->type));

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    res.smem = 32*(పరిమాణం(దశాంశం) + పరిమాణం(int32_t));

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part(ggml_metal_library_t lib) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, "kernel_argmax_part_f32");
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, "kernel_argmax_part_f32", "kernel_argmax_part_f32", శూన్యసూచిక);
    }

    res.smem = 32*(పరిమాణం(దశాంశం) + పరిమాణం(int32_t));

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final(ggml_metal_library_t lib) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, "kernel_argmax_final_f32");
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, "kernel_argmax_final_f32", "kernel_argmax_final_f32", శూన్యసూచిక);
    }

    res.smem = 32*(పరిమాణం(దశాంశం) + పరిమాణం(int32_t));

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_ARGSORT);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    ggml_క్రమీకరించు_క్రమం క్రమం = (ggml_క్రమీకరించు_క్రమం) op->op_params[0];

    స్థిరరాశి అక్షరం * క్రమం_str = "undefined";
    ఎంపిక (క్రమం) {
        స్థితి GGML_క్రమీకరించు_క్రమం_ASC:  క్రమం_str = "asc";  ఆపు;
        స్థితి GGML_క్రమీకరించు_క్రమం_DESC: క్రమం_str = "desc"; ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    };

    snprintf(పునాది, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), క్రమం_str);
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_విలీనం(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_ARGSORT);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    ggml_క్రమీకరించు_క్రమం క్రమం = (ggml_క్రమీకరించు_క్రమం) op->op_params[0];

    స్థిరరాశి అక్షరం * క్రమం_str = "undefined";
    ఎంపిక (క్రమం) {
        స్థితి GGML_క్రమీకరించు_క్రమం_ASC:  క్రమం_str = "asc";  ఆపు;
        స్థితి GGML_క్రమీకరించు_క్రమం_DESC: క్రమం_str = "desc"; ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    };

    snprintf(పునాది, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), క్రమం_str);
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht(ggml_metal_library_t lib, పూర్ణాంకం n, ggml_type tsrc) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_fwht_%s_%d", ggml_type_name(tsrc), n);
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

// note: reuse the argsort kernel for the bitonic top_k fallback
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_శిఖరం_k(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_శిఖరం_K);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    // note: the top_k kernel is always descending order
    ggml_క్రమీకరించు_క్రమం క్రమం = GGML_క్రమీకరించు_క్రమం_DESC;

    స్థిరరాశి అక్షరం * క్రమం_str = "undefined";
    ఎంపిక (క్రమం) {
        స్థితి GGML_క్రమీకరించు_క్రమం_ASC:  క్రమం_str = "asc";  ఆపు;
        స్థితి GGML_క్రమీకరించు_క్రమం_DESC: క్రమం_str = "desc"; ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    };

    snprintf(పునాది, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), క్రమం_str);
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_శిఖరం_k_radix(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_శిఖరం_K);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_top_k_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_శిఖరం_k_విలీనం(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_శిఖరం_K);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    ggml_క్రమీకరించు_క్రమం క్రమం = GGML_క్రమీకరించు_క్రమం_DESC;

    స్థిరరాశి అక్షరం * క్రమం_str = "undefined";
    ఎంపిక (క్రమం) {
        స్థితి GGML_క్రమీకరించు_క్రమం_ASC:  క్రమం_str = "asc";  ఆపు;
        స్థితి GGML_క్రమీకరించు_క్రమం_DESC: క్రమం_str = "desc"; ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    };

    snprintf(పునాది, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), క్రమం_str);
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe(
        ggml_metal_library_t lib, int32_t n_expert, int32_t శిఖరం_k, తార్కిక with_norm) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_topk_moe_f32");
    snprintf(name, 256, "%s_n_expert=%d_top_k=%d_with_norm=%d", పునాది, n_expert, శిఖరం_k, with_norm ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_సమితి_bool (cv, with_norm, FC_TOPK_MOE + 0);
        ggml_metal_cv_సమితి_int32(cv, n_expert,  FC_TOPK_MOE + 1);
        ggml_metal_cv_సమితి_int32(cv, శిఖరం_k,     FC_TOPK_MOE + 2);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce(ggml_metal_library_t lib, int32_t n_expert_వాడిన) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_moe_reduce_f32");
    snprintf(name, 256, "%s_n_expert_used=%d", పునాది, n_expert_వాడిన);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_సమితి_int32(cv, n_expert_వాడిన, FC_MOE_REDUCE + 0);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        స్థిరరాశి నిర్మాణం ggml_tensor * op,
        తార్కిక    has_ముసుగు,
        int32_t ncpsg) {
    ప్రతిజ్ఞా(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_వాడని(op);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_%s",
            "flash_attn_ext_pad");

    snprintf(name, 256, "%s_mask=%d_ncpsg=%d",
            పునాది,
            has_ముసుగు,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_bool(cv, has_ముసుగు,  FC_FLASH_ATTN_EXT_PAD + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_PAD + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_PAD + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_PAD + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_PAD + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_PAD + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_PAD + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_PAD + 23);
        //ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_PAD + 24);
        ggml_metal_cv_సమితి_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_FLASH_ATTN_EXT);

    అక్షరం పునాది[256];

    snprintf(పునాది, 256, "kernel_flash_attn_ext_kv_%s_f16", ggml_type_name(op->src[1]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, పునాది);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, పునాది, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        స్థిరరాశి నిర్మాణం ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg) {
    ప్రతిజ్ఞా(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_వాడని(op);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_%s",
            "flash_attn_ext_blk");

    snprintf(name, 256, "%s_nqptg=%d_ncpsg=%d",
            పునాది,
            nqptg,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        //ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_BLK + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_BLK + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_BLK + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_BLK + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_BLK + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_BLK + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_BLK + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_BLK + 23);
        ggml_metal_cv_సమితి_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
        ggml_metal_cv_సమితి_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        స్థిరరాశి ggml_tensor * op,
        తార్కిక has_ముసుగు,
        తార్కిక has_sinks,
        తార్కిక has_bias,
        తార్కిక has_scap) {
    ప్రతిజ్ఞా(op->op == GGML_OP_FLASH_ATTN_EXT);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి int32_t dk = (int32_t) op->src[1]->ne[0];
    స్థిరరాశి int32_t dv = (int32_t) op->src[2]->ne[0];

    స్థిరరాశి అక్షరం * type = op->src[1]->type == GGML_TYPE_BF16 ? "bf16" : "f16";

    snprintf(పునాది, 256, "kernel_flash_attn_ext_tensor_%s_dk%d_dv%d", type, dk, dv);
    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d", పునాది, has_ముసుగు, has_sinks, has_bias, has_scap);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_bool(cv, has_ముసుగు,  FC_FLASH_ATTN_EXT_TENSOR + 0);
        ggml_metal_cv_సమితి_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_TENSOR + 1);
        ggml_metal_cv_సమితి_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_TENSOR + 2);
        ggml_metal_cv_సమితి_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_TENSOR + 3);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        స్థిరరాశి ggml_tensor * op,
        తార్కిక    has_ముసుగు,
        తార్కిక    has_sinks,
        తార్కిక    has_bias,
        తార్కిక    has_scap,
        తార్కిక    has_kvpad,
        int32_t nsg,
        తార్కిక    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    ప్రతిజ్ఞా(op->op == GGML_OP_FLASH_ATTN_EXT);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి int32_t dk = (int32_t) op->src[1]->ne[0];
    స్థిరరాశి int32_t dv = (int32_t) op->src[2]->ne[0];

    స్థిరరాశి అక్షరం * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    // do bounds checks for the mask?
    స్థిరరాశి తార్కిక bc_ముసుగు = op->src[3] && (op->src[3]->ne[1] % 8 != 0);

    snprintf(పునాది, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            type,
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
            పునాది,
            has_ముసుగు,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            bc_ముసుగు,
            ns10,
            ns20,
            nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_bool(cv, has_ముసుగు,  FC_FLASH_ATTN_EXT + 0);
        ggml_metal_cv_సమితి_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        ggml_metal_cv_సమితి_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
        ggml_metal_cv_సమితి_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);
        ggml_metal_cv_సమితి_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);

        ggml_metal_cv_సమితి_bool(cv, bc_ముసుగు, FC_FLASH_ATTN_EXT + 10);

        ggml_metal_cv_సమితి_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        ggml_metal_cv_సమితి_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        ggml_metal_cv_సమితి_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_FLASH_ATTN_EXT);
    ప్రతిజ్ఞా(op->src[3]);

    అక్షరం name[256];

    snprintf(name, 256, "kernel_flash_attn_ext_vec_idx");

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, శూన్యసూచిక);
    }

    GGML_వాడని(op);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        స్థిరరాశి ggml_tensor * op,
        తార్కిక    has_ముసుగు,
        తార్కిక    has_sinks,
        తార్కిక    has_bias,
        తార్కిక    has_scap,
        తార్కిక    has_kvpad,
        తార్కిక    has_sparse,
        int32_t nqpsg,
        int32_t ne,
        int32_t nsg,
        int32_t nwg,
        తార్కిక    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    ప్రతిజ్ఞా(op->op == GGML_OP_FLASH_ATTN_EXT);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి int32_t dk = (int32_t) op->src[1]->ne[0];
    స్థిరరాశి int32_t dv = (int32_t) op->src[2]->ne[0];

    స్థిరరాశి అక్షరం * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    అక్షరం qne_suffix[16] = {0};
    ఒకవేళ (!(nqpsg == 1 && ne == ggml_metal_tuning::fa_vec_baseline_ne(dk, dv))) {
        snprintf(qne_suffix, పరిమాణం(qne_suffix), "_q%d_ne%d", nqpsg, ne);
    }

    snprintf(పునాది, 256, "kernel_%s_%s_dk%d_dv%d%s",
            "flash_attn_ext_vec",
            type,
            dk,
            dv,
            qne_suffix);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_sparse=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            పునాది,
            has_ముసుగు,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            has_sparse,
            ns10,
            ns20,
            nsg, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_bool(cv, has_ముసుగు,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_సమితి_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_సమితి_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_సమితి_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_సమితి_bool(cv, has_kvpad,  FC_FLASH_ATTN_EXT_VEC + 4);
        ggml_metal_cv_సమితి_bool(cv, has_sparse, FC_FLASH_ATTN_EXT_VEC + 5);

        ggml_metal_cv_సమితి_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_సమితి_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_సమితి_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_సమితి_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        స్థిరరాశి ggml_tensor * op,
        int32_t dv,
        int32_t nwg) {
    ప్రతిజ్ఞా(op->op == GGML_OP_FLASH_ATTN_EXT);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "%s_dv=%d_nwg=%d", పునాది, dv, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        ggml_metal_cv_సమితి_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;

    GGML_వాడని(op);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, int32_t n_fuse) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    పూర్ణాంకం op_అంకె = -1;

    ఎంపిక (op->op) {
        స్థితి GGML_OP_కూడు: op_అంకె = 0; ఆపు;
        స్థితి GGML_OP_SUB: op_అంకె = 1; ఆపు;
        స్థితి GGML_OP_MUL: op_అంకె = 2; ఆపు;
        స్థితి GGML_OP_DIV: op_అంకె = 3; ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    };

    స్థిరరాశి అక్షరం * t0_str = ggml_type_name(op->src[0]->type);
    స్థిరరాశి అక్షరం * t1_str = ggml_type_name(op->src[1]->type);
    స్థిరరాశి అక్షరం * t_str  = ggml_type_name(op->type);

    స్థిరరాశి తార్కిక is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    స్థిరరాశి తార్కిక is_cb = op->src[0]->ne[0] != op->src[1]->ne[0];
    స్థిరరాశి తార్కిక is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(పునాది, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d_cb=%d", పునాది, op_అంకె, n_fuse, is_rb, is_cb);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, op_అంకె, FC_BIN + 0);
        ggml_metal_cv_సమితి_int16(cv, n_fuse, FC_BIN + 1);
        ggml_metal_cv_సమితి_bool (cv, is_rb,  FC_BIN + 2);
        ggml_metal_cv_సమితి_bool (cv, is_cb,  FC_BIN + 3);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_rb;

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one(ggml_metal_library_t lib, ggml_op op) {
    అక్షరం పునాది[256];
    అక్షరం name[256];

    పూర్ణాంకం op_అంకె = -1;

    ఎంపిక (op) {
        స్థితి GGML_OP_కూడు: op_అంకె = 0; ఆపు;
        స్థితి GGML_OP_SUB: op_అంకె = 1; ఆపు;
        స్థితి GGML_OP_MUL: op_అంకె = 2; ఆపు;
        స్థితి GGML_OP_DIV: op_అంకె = 3; ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    };

    snprintf(పునాది, 256, "kernel_bin_fuse_%s_%s_%s", "f32", "f32", "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", పునాది, op_అంకె, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, op_అంకె, FC_BIN + 0);
        ggml_metal_cv_సమితి_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_సమితి_bool (cv, అబద్ధం,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_L2_NORM);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి తార్కిక is_c4 = op->src[0]->ne[0] % 4 == 0;

    స్థిరరాశి అక్షరం * t0_str = ggml_type_name(op->src[0]->type);
    స్థిరరాశి అక్షరం * t_str  = ggml_type_name(op->type);

    snprintf(పునాది, 256, "kernel_l2_norm_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    res.c4   = is_c4;
    res.smem = 32*పరిమాణం(దశాంశం);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_సమూహం_norm(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_సమూహం_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    res.smem = 32*పరిమాణం(దశాంశం);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, పూర్ణాంకం n_fuse) {
    ప్రతిజ్ఞా(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(op->src[0]));

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి అక్షరం * suffix = "";
    ఒకవేళ (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    ఎంపిక (op->op) {
        స్థితి GGML_OP_NORM:
            ఎంపిక (n_fuse) {
                స్థితి 1: snprintf(పునాది, 256, "kernel_norm_f32%s", suffix);         ఆపు;
                స్థితి 2: snprintf(పునాది, 256, "kernel_norm_mul_f32%s", suffix);     ఆపు;
                స్థితి 3: snprintf(పునాది, 256, "kernel_norm_mul_add_f32%s", suffix); ఆపు;
                అప్రమేయ: GGML_ABORT("fatal error");
            } ఆపు;
        స్థితి GGML_OP_RMS_NORM:
            ఎంపిక (n_fuse) {
                స్థితి 1: snprintf(పునాది, 256, "kernel_rms_norm_f32%s", suffix);         ఆపు;
                స్థితి 2: snprintf(పునాది, 256, "kernel_rms_norm_mul_f32%s", suffix);     ఆపు;
                స్థితి 3: snprintf(పునాది, 256, "kernel_rms_norm_mul_add_f32%s", suffix); ఆపు;
                అప్రమేయ: GGML_ABORT("fatal error");
            } ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_సమితి_bool(cv, అబద్ధం, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*పరిమాణం(దశాంశం);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(op->src[0]));

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి అక్షరం * suffix = "";
    ఒకవేళ (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    ఎంపిక (op->op) {
        స్థితి GGML_OP_NORM:     snprintf(పునాది, 256, "kernel_norm_mul_f32%s", suffix);     ఆపు;
        స్థితి GGML_OP_RMS_NORM: snprintf(పునాది, 256, "kernel_rms_norm_mul_f32%s", suffix); ఆపు;
        అప్రమేయ: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s_use_scale", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_సమితి_bool(cv, నిజం, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*పరిమాణం(దశాంశం);

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_ROPE || op->op == GGML_OP_ROPE_వెనుక);

    స్థిరరాశి తార్కిక is_వెనుక = op->op == GGML_OP_ROPE_వెనుక;

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి పూర్ణాంకం mode = ((స్థిరరాశి int32_t *) op->op_params)[2];

    స్థిరరాశి తార్కిక is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    స్థిరరాశి తార్కిక is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    స్థిరరాశి తార్కిక is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    స్థిరరాశి తార్కిక is_vision = mode == GGML_ROPE_TYPE_VISION;

    ఒకవేళ (is_neox) {
        snprintf(పునాది, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } లేకపోతే ఒకవేళ ((is_mrope || is_imrope) && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(పునాది, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } లేకపోతే ఒకవేళ (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(పునాది, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } లేకపోతే {
        snprintf(పునాది, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s_imrope=%d_is_back=%d", పునాది, is_imrope ? 1 : 0, is_వెనుక ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_bool(cv, is_imrope, FC_ROPE + 0);
        ggml_metal_cv_సమితి_bool(cv, is_వెనుక,   FC_ROPE + 1);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_IM2_నిలువు);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    స్థిరరాశి తార్కిక is_2D = ((స్థిరరాశి int32_t *)(op->op_params))[6] == 1;
    స్థిరరాశి int64_t KH = is_2D ? ne01 : 1;
    స్థిరరాశి int64_t KW = ne00;

    అక్షరం పునాది[256];
    అక్షరం name[256];

    ఒకవేళ (KH*KW <= 1024) {
        snprintf(పునాది, 256, "kernel_im2col_%s", ggml_type_name(op->type));
    } లేకపోతే {
        snprintf(పునాది, 256, "kernel_im2col_ext_%s", ggml_type_name(op->type));
    }
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_COL2IM_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_col2im_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake(ggml_metal_library_t lib, గణన ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_snake_%s", ggml_type_name(type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_conv_transpose_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_CONV_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_conv_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op, తార్కిక tiled) {
    ప్రతిజ్ఞా(op->op == GGML_OP_CONV_2D_DW);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_conv_2d_dw%s_%s_%s",
             tiled ? "_tiled" : "",
             ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_CONV_3D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_conv_3d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_UPSCALE);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి int32_t బహుళకం_flags = ggml_get_op_params_i32(op, 0);
    స్థిరరాశి ggml_scale_బహుళకం mode = (ggml_scale_బహుళకం) (బహుళకం_flags & 0xFF);

    స్థిరరాశి తార్కిక antialias = (బహుళకం_flags & GGML_SCALE_FLAG_ANTIALIAS);

    ఒకవేళ (mode == GGML_SCALE_బహుళకం_BILINEAR) {
        snprintf(పునాది, 256, "kernel_upscale_bilinear_%s", ggml_type_name(op->src[0]->type));
    } లేకపోతే ఒకవేళ (mode == GGML_SCALE_బహుళకం_BICUBIC) {
        snprintf(పునాది, 256, "kernel_upscale_bicubic_%s", ggml_type_name(op->src[0]->type));
    } లేకపోతే {
        snprintf(పునాది, 256, "kernel_upscale_nearest_%s", ggml_type_name(op->src[0]->type));
    }
    snprintf(name, 256, "%s_aa=%d", పునాది, antialias);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_bool(cv, antialias, FC_UPSCALE + 0);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_ROLL);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_roll_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_PAD);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    స్థిరరాశి తార్కిక circular = ggml_get_op_params_i32(op, 8) != 0;

    snprintf(పునాది, 256, "kernel_pad_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_circular=%d", పునాది, circular);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_bool(cv, circular, FC_PAD + 0);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_PAD_REFLECT_1D);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_ARANGE);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_అడుగు_adamw(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_OPT_అడుగు_ADAMW);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_అడుగు_sgd(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_OPT_అడుగు_SGD);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_వెనుక(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor * op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_SILU_వెనుక);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_silu_back_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor *  op) {
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    అక్షరం పునాది[256];
    అక్షరం name[256];

    snprintf(పునాది, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", పునాది);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, శూన్యసూచిక);
    }

    తిరిగి res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_లెక్క_equal(ggml_metal_library_t lib, స్థిరరాశి ggml_tensor *  op) {
    ప్రతిజ్ఞా(op->op == GGML_OP_లెక్క_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    అక్షరం పునాది[256];
    అక్షరం name[256];

    పూర్ణాంకం nsg = 1;
    అయితే (32*nsg < ne00 && nsg < 32) {
        nsg *= 2;
    }

    snprintf(పునాది, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", పునాది, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    ఒకవేళ (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_సమితి_int16(cv, nsg, FC_లెక్క_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, పునాది, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32 * పరిమాణం(int32_t);
    res.nsg  = nsg;

    తిరిగి res;
}
