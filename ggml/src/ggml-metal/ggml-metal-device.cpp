#অন্তর্ভাব "ggml-metal-device.h"
#অন্তর্ভাব "ggml-metal-common.h"

#অন্তর্ভাব "ggml-metal-impl.h"
#অন্তর্ভাব "ggml-metal-tuning.h"

#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <cassert>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <string>
#অন্তর্ভাব <unordered_map>

কাঠামো ggml_metal_device_deleter {
    শূন্য operator()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

প্রকারলেখ std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

ggml_metal_device_t ggml_metal_device_get(পূর্ণসংখ্যা device, পূর্ণসংখ্যা n_devices) {
    স্থির std::vector<ggml_metal_device_ptr> devs;

    devs.emplace_back(ggml_metal_device_init(device, n_devices));

    ফেরত devs.back().get();
}

কাঠামো ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> তথ্য;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(শূন্য) {
    ggml_metal_pipelines_t res = নতুন ggml_metal_pipelines();

    ফেরত res;
}

শূন্য ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    যদি (!ppls) {
        ফেরত;
    }

    জন্য (auto it = ppls->তথ্য.begin(); it != ppls->তথ্য.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    delete ppls;
}

শূন্য ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, ধ্রুবক অক্ষর * name, ggml_metal_pipeline_t pipeline) {
    ppls->তথ্য[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ধ্রুবক অক্ষর * name) {
    যদি (ppls->তথ্য.find(name) == ppls->তথ্য.end()) {
        ফেরত nullptr;
    }

    ফেরত ppls->তথ্য[name];
}

কাঠামো ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base(ggml_metal_library_t lib, ggml_op op) {
    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক অক্ষর * op_str = "undefined";
    নির্বাচন (op) {
        ক্ষেত্র GGML_OP_ADD_ID: op_str = "add_id"; বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    ধ্রুবক অক্ষর * pool_str = "undefined";
    নির্বাচন (op_pool) {
        ক্ষেত্র GGML_OP_POOL_AVG: pool_str = "avg"; বিরতি;
        ক্ষেত্র GGML_OP_POOL_MAX: pool_str = "max"; বিরতি;
        পূর্বনির্ধারিত: GGML_ASSERT(মিথ্যা && "not implemented");
    };

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, আকার(base), "kernel_pool_1d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, আকার(name), "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    ধ্রুবক অক্ষর * pool_str = "undefined";
    নির্বাচন (op_pool) {
        ক্ষেত্র GGML_OP_POOL_AVG: pool_str = "avg"; বিরতি;
        ক্ষেত্র GGML_OP_POOL_MAX: pool_str = "max"; বিরতি;
        পূর্বনির্ধারিত: GGML_ASSERT(মিথ্যা && "not implemented");
    };

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows(ggml_metal_library_t lib, ggml_type tsrc) {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক auto tsrc = op->src[0]->type;
    ধ্রুবক auto tidx = op->src[1]->type;
    ধ্রুবক auto tdst = op->type;

    snprintf(base, 256, "kernel_set_rows_%s_%s_%s", ggml_type_name(tsrc), ggml_type_name(tidx), ggml_type_name(tdst));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক পূর্ণসংখ্যা n = op->src[0]->ne[0];

    snprintf(base, 256, "kernel_diag_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_n=%d", base, n);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.nsg  = 1;
    res.smem = 0;

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat(ggml_metal_library_t lib, ggml_type tsrc) {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat(ggml_metal_library_t lib, ggml_type tsrc) {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_concat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    অক্ষর base[256];
    অক্ষর name[256];

    পূর্ণসংখ্যা op_অঙ্ক = -1;

    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_SCALE:      op_অঙ্ক = OP_UNARY_অঙ্ক_SCALE;      বিরতি;
        ক্ষেত্র GGML_OP_FILL:       op_অঙ্ক = OP_UNARY_অঙ্ক_FILL;       বিরতি;
        ক্ষেত্র GGML_OP_CLAMP:      op_অঙ্ক = OP_UNARY_অঙ্ক_CLAMP;      বিরতি;
        ক্ষেত্র GGML_OP_SQR:        op_অঙ্ক = OP_UNARY_অঙ্ক_SQR;        বিরতি;
        ক্ষেত্র GGML_OP_SQRT:       op_অঙ্ক = OP_UNARY_অঙ্ক_SQRT;       বিরতি;
        ক্ষেত্র GGML_OP_SIN:        op_অঙ্ক = OP_UNARY_অঙ্ক_SIN;        বিরতি;
        ক্ষেত্র GGML_OP_COS:        op_অঙ্ক = OP_UNARY_অঙ্ক_COS;        বিরতি;
        ক্ষেত্র GGML_OP_LOG:        op_অঙ্ক = OP_UNARY_অঙ্ক_LOG;        বিরতি;
        ক্ষেত্র GGML_OP_LEAKY_RELU: op_অঙ্ক = OP_UNARY_অঙ্ক_LEAKY_RELU; বিরতি;
        ক্ষেত্র GGML_OP_UNARY:
            নির্বাচন (ggml_get_unary_op(op)) {
                ক্ষেত্র GGML_UNARY_OP_TANH:        op_অঙ্ক = OP_UNARY_অঙ্ক_TANH;        বিরতি;
                ক্ষেত্র GGML_UNARY_OP_RELU:        op_অঙ্ক = OP_UNARY_অঙ্ক_RELU;        বিরতি;
                ক্ষেত্র GGML_UNARY_OP_SIGMOID:     op_অঙ্ক = OP_UNARY_অঙ্ক_SIGMOID;     বিরতি;
                ক্ষেত্র GGML_UNARY_OP_GELU:        op_অঙ্ক = OP_UNARY_অঙ্ক_GELU;        বিরতি;
                ক্ষেত্র GGML_UNARY_OP_GELU_ERF:    op_অঙ্ক = OP_UNARY_অঙ্ক_GELU_ERF;    বিরতি;
                ক্ষেত্র GGML_UNARY_OP_GELU_QUICK:  op_অঙ্ক = OP_UNARY_অঙ্ক_GELU_QUICK;  বিরতি;
                ক্ষেত্র GGML_UNARY_OP_SILU:        op_অঙ্ক = OP_UNARY_অঙ্ক_SILU;        বিরতি;
                ক্ষেত্র GGML_UNARY_OP_ELU:         op_অঙ্ক = OP_UNARY_অঙ্ক_ELU;         বিরতি;
                ক্ষেত্র GGML_UNARY_OP_NEG:         op_অঙ্ক = OP_UNARY_অঙ্ক_NEG;         বিরতি;
                ক্ষেত্র GGML_UNARY_OP_ABS:         op_অঙ্ক = OP_UNARY_অঙ্ক_ABS;         বিরতি;
                ক্ষেত্র GGML_UNARY_OP_SGN:         op_অঙ্ক = OP_UNARY_অঙ্ক_SGN;         বিরতি;
                ক্ষেত্র GGML_UNARY_OP_STEP:        op_অঙ্ক = OP_UNARY_অঙ্ক_STEP;        বিরতি;
                ক্ষেত্র GGML_UNARY_OP_HARDSWISH:   op_অঙ্ক = OP_UNARY_অঙ্ক_HARDSWISH;   বিরতি;
                ক্ষেত্র GGML_UNARY_OP_HARDSIGMOID: op_অঙ্ক = OP_UNARY_অঙ্ক_HARDSIGMOID; বিরতি;
                ক্ষেত্র GGML_UNARY_OP_EXP:         op_অঙ্ক = OP_UNARY_অঙ্ক_EXP;         বিরতি;
                ক্ষেত্র GGML_UNARY_OP_SOFTPLUS:    op_অঙ্ক = OP_UNARY_অঙ্ক_SOFTPLUS;    বিরতি;
                ক্ষেত্র GGML_UNARY_OP_EXPM1:       op_অঙ্ক = OP_UNARY_অঙ্ক_EXPM1;       বিরতি;
                ক্ষেত্র GGML_UNARY_OP_FLOOR:       op_অঙ্ক = OP_UNARY_অঙ্ক_FLOOR;       বিরতি;
                ক্ষেত্র GGML_UNARY_OP_CEIL:        op_অঙ্ক = OP_UNARY_অঙ্ক_CEIL;        বিরতি;
                ক্ষেত্র GGML_UNARY_OP_ROUND:       op_অঙ্ক = OP_UNARY_অঙ্ক_ROUND;       বিরতি;
                ক্ষেত্র GGML_UNARY_OP_TRUNC:       op_অঙ্ক = OP_UNARY_অঙ্ক_TRUNC;       বিরতি;
                ক্ষেত্র GGML_UNARY_OP_XIELU:       op_অঙ্ক = OP_UNARY_অঙ্ক_XIELU;       বিরতি;
                পূর্বনির্ধারিত: GGML_ABORT("fatal error");
            } বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    };

    ধ্রুবক অক্ষর * t0_str = ggml_type_name(op->src[0]->type);
    ধ্রুবক অক্ষর * t_str  = ggml_type_name(op->type);

    ধ্রুবক যৌক্তিক is_c4 = op->src[0]->ne[0] % 4 == 0;
    ধ্রুবক যৌক্তিক is_cnt = ggml_is_contiguous(op->src[0]) && ggml_nelements(op) < 32768;

    snprintf(base, 256, "kernel_unary_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_cnt=%d", base, op_অঙ্ক, is_cnt);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_অঙ্ক, FC_UNARY + 0);
        ggml_metal_cv_set_bool (cv, is_cnt, FC_UNARY + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_cnt;

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক অক্ষর * op_str = "undefined";
    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_GLU:
            নির্বাচন (ggml_get_glu_op(op)) {
                ক্ষেত্র GGML_GLU_OP_REGLU:        op_str = "reglu";        বিরতি;
                ক্ষেত্র GGML_GLU_OP_GEGLU:        op_str = "geglu";        বিরতি;
                ক্ষেত্র GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       বিরতি;
                ক্ষেত্র GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   বিরতি;
                ক্ষেত্র GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    বিরতি;
                ক্ষেত্র GGML_GLU_OP_GEGLU_QUICK:  op_str = "geglu_quick";  বিরতি;
                ক্ষেত্র GGML_GLU_OP_SWIGLU_CLAMP: op_str = "swiglu_clamp"; বিরতি;
                পূর্বনির্ধারিত: GGML_ABORT("fatal error");
            } বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_যোগ(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_যোগ);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_op_sum_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_যোগ_rows(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    অক্ষর base[256];
    অক্ষর name[256];

    পূর্ণসংখ্যা op_অঙ্ক = -1;

    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_যোগ_ROWS: op_অঙ্ক = OP_যোগ_ROWS_অঙ্ক_যোগ_ROWS; বিরতি;
        ক্ষেত্র GGML_OP_MEAN:     op_অঙ্ক = OP_যোগ_ROWS_অঙ্ক_MEAN;     বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    };

    ধ্রুবক অক্ষর * t0_str = ggml_type_name(op->src[0]->type);
    ধ্রুবক অক্ষর * t_str  = ggml_type_name(op->type);

    ধ্রুবক যৌক্তিক is_c4 = op->src[0]->ne[0] % 4 == 0;

    snprintf(base, 256, "kernel_sum_rows_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d", base, op_অঙ্ক);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_অঙ্ক, FC_যোগ_ROWS + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*আকার(দশমিক);

    যদি (is_c4) {
        res.smem *= 4;
    }

    res.c4  = is_c4;

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_cumsum_blk_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_cumsum_add_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_TRI);
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_আকার(op->src[0]->type));

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক অক্ষর * op_str = "tri";
    ধ্রুবক পূর্ণসংখ্যা ttype = op->op_params[0];

    snprintf(base, 256, "kernel_%s_%s_%d", op_str, ggml_type_name(op->src[0]->type), ttype);

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক অক্ষর * suffix = "";

    যদি (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    ধ্রুবক ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(base, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*আকার(দশমিক);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer(
        ggml_metal_library_t lib,
        ধ্রুবক ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক int16_t nh = op->src[0]->ne[1];

    snprintf(base, 256, "kernel_lightning_indexer_%s", ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s_nh=%d", base, nh);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nh, FC_LIGHTNING_INDEXER + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    অক্ষর name[256];
    ধ্রুবক অক্ষর * base = nullptr;

    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_DSV4_HC_COMB:
            base = "kernel_dsv4_hc_comb_f32";
            snprintf(name, 256, "%s", base);
            বিরতি;
        ক্ষেত্র GGML_OP_DSV4_HC_PRE:
            base = ggml_get_op_params_i32(op, 1) != 0 ? "kernel_dsv4_hc_pre_gated_f32" : "kernel_dsv4_hc_pre_f32";
            snprintf(name, 256, "%s_n_hc=%d", base, (পূর্ণসংখ্যা) op->src[0]->ne[1]);
            বিরতি;
        ক্ষেত্র GGML_OP_DSV4_HC_POST:
            base = op->src[3] ? "kernel_dsv4_hc_post_f32" : "kernel_dsv4_hc_post_nocomb_f32";
            snprintf(name, 256, "%s", base);
            বিরতি;
        পূর্বনির্ধারিত:
            GGML_ABORT("fatal error");
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = nullptr;

        যদি (op->op == GGML_OP_DSV4_HC_PRE) {
            cv = ggml_metal_cv_init();
            ggml_metal_cv_set_int32(cv, (int32_t) op->src[0]->ne[1], FC_DSV4_HC + 0);
        }

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        যদি (cv) {
            ggml_metal_cv_free(cv);
        }
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv(
        ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, int32_t nc, যৌক্তিক use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক অক্ষর * suffix = "";

    যদি (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_nc=%d_silu=%d", base, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_bool(cv, use_silu, FC_SSM_CONV + 1);
        ggml_metal_cv_set_int32(cv, nc,      FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched(
        ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, পূর্ণসংখ্যা ssm_conv_bs, int32_t nc, যৌক্তিক use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক অক্ষর * suffix = "";
    যদি (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s_batched%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_ssm_conv_bs=%d_nc=%d_silu=%d", base, ssm_conv_bs, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ssm_conv_bs, FC_SSM_CONV + 0);
        ggml_metal_cv_set_bool(cv, use_silu,     FC_SSM_CONV + 1);
        ggml_metal_cv_set_int32(cv, nc,          FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, যৌক্তিক লেজ)  {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক পূর্ণসংখ্যা nsg = (ne00 + 31)/32;

    snprintf(base, 256, "kernel_ssm_scan_%s%s", ggml_type_name(op->src[0]->type), লেজ ? "_tail" : "");
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (32 + 2) floats
    res.smem = GGML_PAD((32 + 2)*আকার(দশমিক)*nsg, 16);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan_ssd_mma(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op)  {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_ssm_scan_ssd_mma_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    // acs/exp(acs)/state-decay vectors + dtX + SAM rows + two 8x8 tiles per simdgroup
    res.smem = (3*OP_SSM_SCAN_SSD_CS +
                OP_SSM_SCAN_SSD_CS*OP_SSM_SCAN_SSD_HD +
                OP_SSM_SCAN_SSD_NSG*8*OP_SSM_SCAN_SSD_CS +
                OP_SSM_SCAN_SSD_NSG*2*8*8)*আকার(দশমিক);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক int64_t C = op->ne[0];
    ধ্রুবক int64_t H = op->src[0]->ne[1];

    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } বিরতি;
        ক্ষেত্র GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } বিরতি;
        পূর্বনির্ধারিত:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    অক্ষর base[256];
    অক্ষর name[256];

    // v is src[2], dimensions: S_v = ne[0], H = ne[1]
    ধ্রুবক পূর্ণসংখ্যা ne20 = op->src[2]->ne[0]; // S_v
    ধ্রুবক পূর্ণসংখ্যা ne21 = op->src[2]->ne[1]; // H
    ধ্রুবক পূর্ণসংখ্যা ne30 = op->src[3]->ne[0]; // G
    // state is src[5], 4D [S_v, S_v, H_v, n_seqs] (s0 only); K is op param 0.
    ধ্রুবক পূর্ণসংখ্যা K = ggml_get_op_params_i32(op, 0);

    ধ্রুবক পূর্ণসংখ্যা nsg = op->src[2]->ne[0]/32;

    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->ne[0] == ne20 * ne21);
    GGML_ASSERT(ne20 % 32 == 0);

    snprintf(base, 256, "kernel_gated_delta_net_%s_%d", ggml_type_name(op->src[0]->type), nsg);
    snprintf(name, 256, "%s_ne20=%d_ne30=%d_K=%d", base, ne20, ne30, K);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ne20, FC_GATED_DELTA_NET + 0);
        ggml_metal_cv_set_int16(cv, ne30, FC_GATED_DELTA_NET + 1);
        ggml_metal_cv_set_int16(cv, K,    FC_GATED_DELTA_NET + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg = nsg;

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক পূর্ণসংখ্যা nsg = 8;
    ধ্রুবক পূর্ণসংখ্যা n   = op->src[1]->ne[1];
    ধ্রুবক পূর্ণসংখ্যা k   = op->src[1]->ne[0];

    snprintf(base, 256, "kernel_solve_tri_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d_n=%d_k=%d", base, nsg, n, k);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_SOLVE_TRI + 0);
        ggml_metal_cv_set_int16(cv, n,   FC_SOLVE_TRI + 1);
        ggml_metal_cv_set_int16(cv, k,   FC_SOLVE_TRI + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = GGML_PAD(GGML_PAD(n, 32)*nsg*আকার(দশমিক), 16);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, পূর্ণসংখ্যা nsg, পূর্ণসংখ্যা nxpsg, পূর্ণসংখ্যা r1ptg) {
    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক ggml_type tsrc0 = op->src[0]->type;
    ধ্রুবক ggml_type tsrc1 = op->src[1]->type;
    ধ্রুবক পূর্ণসংখ্যা       ne12  = op->src[1]->ne[2];
    ধ্রুবক পূর্ণসংখ্যা       r2    = ne12 / op->src[0]->ne[2];
    ধ্রুবক পূর্ণসংখ্যা       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d_ne12=%d_r2=%d_r3=%d", base, nsg, nxpsg, ne12, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, nxpsg,          FC_MUL_MV + 1);
        ggml_metal_cv_set_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, (int16_t) r2,   FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, (int16_t) r3,   FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

// threadgroups needed to fill the GPU
// TODO: dedup
স্থির constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

// few-row MMA mat-mul: 8x8 simdgroup matrices for the src1 rows of ggml_metal_mul_mat_use_mma
স্থির constexpr পূর্ণসংখ্যা     GGML_METAL_MMA_TILE          = 8;
স্থির constexpr পূর্ণসংখ্যা     GGML_METAL_MMA_NT_MAX        = 4;
স্থির constexpr পূর্ণসংখ্যা     GGML_METAL_MMA_SMEM_MAX      = 16384;

// simdgroups per threadgroup for at most FEW_ROWS, at most MANY_ROWS, and more src0 rows
স্থির constexpr int64_t GGML_METAL_MMA_FEW_ROWS      = 64;
স্থির constexpr int64_t GGML_METAL_MMA_MANY_ROWS     = 6144;
স্থির constexpr পূর্ণসংখ্যা     GGML_METAL_MMA_NSG_FEW_ROWS  = 32;
স্থির constexpr পূর্ণসংখ্যা     GGML_METAL_MMA_NSG_MID_ROWS  = 16;
স্থির constexpr পূর্ণসংখ্যা     GGML_METAL_MMA_NSG_MANY_ROWS = 8;

কাঠামো ggml_metal_mma_tiling {
    পূর্ণসংখ্যা nsg; // simdgroups per threadgroup, each over a slice of K
    পূর্ণসংখ্যা nt;  // 8-row src0 tiles per threadgroup
    পূর্ণসংখ্যা rt;  // 8-row src1 tiles per threadgroup
};

// halves n while it is above limit
স্থির পূর্ণসংখ্যা ggml_metal_halve_to_limit(পূর্ণসংখ্যা n, int64_t limit) {
    যখন (n > 1 && n > limit) {
        n /= 2;
    }
    ফেরত n;
}

স্থির size_t ggml_metal_mul_mv_mma_smem(পূর্ণসংখ্যা nsg, পূর্ণসংখ্যা nt, পূর্ণসংখ্যা rt) {
    // one 8x8 float simdgroup matrix per output tile
    constexpr size_t tile_bytes = 8*8*আকার(দশমিক);
    ফেরত (size_t) nsg*nt*rt*tile_bytes;
}

// fewer src0 rows need more simdgroups per threadgroup (a finer K split) to fill the GPU.
// the tiles become narrower when the K-slice reduction buffer is too large or when too few threadgroups fill the GPU.
স্থির ggml_metal_mma_tiling ggml_metal_op_mul_mat_mma_tiling(ধ্রুবক ggml_tensor * op) {
    ধ্রুবক ggml_type type = op->src[0]->type;
    ধ্রুবক int64_t   ne00 = op->src[0]->ne[0];
    ধ্রুবক int64_t   ne01 = op->src[0]->ne[1];

    ggml_metal_mma_tiling res;
    res.rt = ggml_metal_mul_mv_mma_rt(op);

    ধ্রুবক int64_t n_steps = ne00/ggml_metal_mul_mv_mma_k_step(type, res.rt);

    পূর্ণসংখ্যা nsg = GGML_METAL_MMA_NSG_MANY_ROWS;
    যদি (ne01 <= GGML_METAL_MMA_FEW_ROWS) {
        nsg = GGML_METAL_MMA_NSG_FEW_ROWS;
    } অন্যথা যদি (ne01 <= GGML_METAL_MMA_MANY_ROWS) {
        nsg = GGML_METAL_MMA_NSG_MID_ROWS;
    }
    res.nsg = ggml_metal_halve_to_limit(nsg, n_steps);

    ধ্রুবক int64_t nt_smem   = GGML_METAL_MMA_SMEM_MAX/(int64_t) ggml_metal_mul_mv_mma_smem(res.nsg, 1, res.rt);
    ধ্রুবক int64_t nt_rows   = ne01/(GGML_METAL_MIN_THREADGROUPS*GGML_METAL_MMA_TILE);
    res.nt = ggml_metal_halve_to_limit(GGML_METAL_MMA_NT_MAX, std::min(nt_smem, nt_rows));

    ফেরত res;
}

// the pipeline for the tiling, with fewer simdgroups when the device cannot run that many threads
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, যৌক্তিক add) {
    ggml_metal_mma_tiling tiling = ggml_metal_op_mul_mat_mma_tiling(op);

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, add);
    যখন (tiling.nsg > 1 && ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < tiling.nsg*32) {
        tiling.nsg /= 2;
        pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, add);
    }

    ফেরত pipeline;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, পূর্ণসংখ্যা nsg, পূর্ণসংখ্যা nt, পূর্ণসংখ্যা rt, যৌক্তিক add) {
    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক ggml_type tsrc0 = op->src[0]->type;
    ধ্রুবক ggml_type tsrc1 = op->src[1]->type;
    ধ্রুবক পূর্ণসংখ্যা       ne12  = op->src[1]->ne[2];
    ধ্রুবক পূর্ণসংখ্যা       r2    = ne12 / op->src[0]->ne[2];
    ধ্রুবক পূর্ণসংখ্যা       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    // the specialized kernels unroll over a compile-time row length
    ধ্রুবক পূর্ণসংখ্যা ne00 = ggml_metal_mul_mv_mma_kind(tsrc0, rt) != GGML_METAL_MMA_KIND_GEN ? op->src[0]->ne[0] : 0;

    snprintf(base, 256, "kernel_mul_mv_mma_%s_%s_nt%d_rt%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), nt, rt);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_ne00=%d_add=%d", base, nsg, ne12, r2, r3, ne00, add);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int32(cv, ne00,           FC_MUL_MV_MMA + 4);
        ggml_metal_cv_set_int16(cv, nsg,            FC_MUL_MV_MMA + 0);
        ggml_metal_cv_set_int16(cv, (int16_t) ne12, FC_MUL_MV_MMA + 1);
        ggml_metal_cv_set_int16(cv, (int16_t) r2,   FC_MUL_MV_MMA + 2);
        ggml_metal_cv_set_int16(cv, (int16_t) r3,   FC_MUL_MV_MMA + 3);
        ggml_metal_cv_set_bool (cv, add,            FC_MUL_MV_MMA + 5);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = ggml_metal_mul_mv_mma_smem(nsg, nt, rt);

    res.nr0 = GGML_METAL_MMA_TILE*nt;
    res.nr1 = GGML_METAL_MMA_TILE*rt;

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক ggml_type tsrc0 = op->src[0]->type;
    ধ্রুবক ggml_type tsrc1 = op->src[1]->type;

    ধ্রুবক যৌক্তিক bc_inp = op->src[0]->ne[0] % 32 != 0;

    constexpr পূর্ণসংখ্যা NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;
    constexpr পূর্ণসংখ্যা NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;

    ধ্রুবক যৌক্তিক has_tensor = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_tensor;

    ধ্রুবক যৌক্তিক bc_out = has_tensor
        ? (op->ne[0] % NRA != 0 || op->ne[1] % NRB != 0)
        : (op->ne[0] % 64  != 0 || op->ne[1] % 32  != 0);

    GGML_ASSERT(op->src[1]->ne[2] <= INT16_MAX && op->src[1]->ne[3] <= INT16_MAX);
    ধ্রুবক int16_t ne12 = (int16_t) op->src[1]->ne[2];
    ধ্রুবক int16_t ne13 = (int16_t) op->src[1]->ne[3];
    ধ্রুবক int16_t r2   = (int16_t) (ne12 / op->src[0]->ne[2]);
    ধ্রুবক int16_t r3   = (int16_t) (ne13 / op->src[0]->ne[3]);

    snprintf(base, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_bco=%d_ne12=%d_ne13=%d_r2=%d_r3=%d",
             base, bc_inp, bc_out, ne12, ne13, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_set_bool(cv, bc_out, FC_MUL_MM + 1);
        ggml_metal_cv_set_int16(cv, ne12,  FC_MUL_MM + 2);
        ggml_metal_cv_set_int16(cv, ne13,  FC_MUL_MM + 3);
        ggml_metal_cv_set_int16(cv, r2,    FC_MUL_MM + 4);
        ggml_metal_cv_set_int16(cv, r3,    FC_MUL_MM + 5);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    যদি (has_tensor) {
        res.nr0 = NRA;
        res.nr1 = NRB;

        ধ্রুবক size_t smem_a = NRA * N_MM_NK_যোগ * আকার(ggml_fp16_t);
        res.smem = smem_a;
    } অন্যথা {
        res.nr0 = 64;
        res.nr1 = 32;

        res.smem = bc_out ? 8192 : (4096 + 2048);
    }

    res.nsg = N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    অক্ষর base[256];
    অক্ষর name[256];

    পূর্ণসংখ্যা nsg = 0; // number of simdgroups
    পূর্ণসংখ্যা nr0 = 0; // number of src0 rows per simdgroup
    পূর্ণসংখ্যা nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    ধ্রুবক ggml_type tsrc0 = op->src[0]->type;
    ধ্রুবক ggml_type tsrc1 = op->src[1]->type;

    ধ্রুবক অক্ষর * suffix = "";

    যৌক্তিক split = মিথ্যা;

    // use custom matrix x vector kernel
    নির্বাচন (tsrc0) {
        ক্ষেত্র GGML_TYPE_F32:
        ক্ষেত্র GGML_TYPE_F16:
        ক্ষেত্র GGML_TYPE_BF16:
            {
                যদি (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } অন্যথা {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    smem = 32*আকার(দশমিক)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*আকার(দশমিক)*N_R0_Q8_0;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*আকার(দশমিক);
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XXS_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XS_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;

                // split the rows across threads when there are fewer than 32 chunks per row
                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_XXS_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_S_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_S_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_S_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_M_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*আকার(দশমিক);
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*আকার(দশমিক);
            } বিরতি;
        ক্ষেত্র GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } বিরতি;
        পূর্বনির্ধারিত:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (পূর্ণসংখ্যা) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    GGML_ASSERT(ne12 <= INT16_MAX && ne13 <= INT16_MAX);
    ধ্রুবক int16_t r2 = (int16_t) (ne12 / ne02);
    ধ্রুবক int16_t r3 = (int16_t) (ne13 / ne03);

    snprintf(base, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_split=%d", base, nsg, ne12, r2, r3, split);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, r2,             FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, r3,             FC_MUL_MV + 4);
        ggml_metal_cv_set_bool (cv, split,          FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib) {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_mul_mm_id_amax_part_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*আকার(দশমিক);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib) {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_mul_mm_id_amax_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*আকার(দশমিক);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, পূর্ণসংখ্যা ne02, পূর্ণসংখ্যা ne20) {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s_ne02=%d", base, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int32(cv, ne02, FC_MUL_MM + 7);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    // per expert: entry count + base offset, followed by the staging of the ids of ntg tokens
    res.smem = 2*(size_t) ne02*আকার(uint32_t) + (size_t) ne02*ne20*আকার(uint16_t);
    res.smem = GGML_PAD(res.smem, 16);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক ggml_type tsrc0 = op->src[0]->type;
    ধ্রুবক ggml_type tsrc1 = op->src[1]->type;

    ধ্রুবক যৌক্তিক bc_inp = op->src[0]->ne[0] % 32 != 0;

    // src1 prec [TAG_GGML_PREC]
    ধ্রুবক যৌক্তিক amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

    snprintf(base, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_amax=%d", base, bc_inp, amax);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_set_bool(cv, amax,   FC_MUL_MM + 6);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 8192;

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    অক্ষর base[256];
    অক্ষর name[256];

    পূর্ণসংখ্যা nsg = 0; // number of simdgroups
    পূর্ণসংখ্যা nr0 = 0; // number of src0 rows per simdgroup
    পূর্ণসংখ্যা nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    ধ্রুবক ggml_type tsrc0 = op->src[0]->type;
    ধ্রুবক ggml_type tsrc1 = op->src[1]->type;

    ধ্রুবক অক্ষর * suffix = "";

    যৌক্তিক split = মিথ্যা;

        // use custom matrix x vector kernel
    নির্বাচন (tsrc0) {
        ক্ষেত্র GGML_TYPE_F32:
        ক্ষেত্র GGML_TYPE_F16:
        ক্ষেত্র GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = 32*আকার(দশমিক)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*আকার(দশমিক)*N_R0_Q8_0;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*আকার(দশমিক);
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XXS_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_XS_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;

                // split the rows across threads when there are fewer than 32 chunks per row
                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_XXS_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ3_S_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ2_S_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_S_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;

                ধ্রুবক পূর্ণসংখ্যা nb32 = ne00/32;
                যদি (nb32 < 32 && (32 % nb32) == 0) {
                    nr0 = N_R0_IQ1_M_SPLIT;
                    split = সত্য;
                }
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*আকার(দশমিক);
            } বিরতি;
        ক্ষেত্র GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*আকার(দশমিক);
            } বিরতি;
        ক্ষেত্র GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } বিরতি;
        পূর্বনির্ধারিত:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (পূর্ণসংখ্যা)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_split=%d", base, nsg, split);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 4);
        ggml_metal_cv_set_bool (cv, split, FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_আকার(op->src[0]->type));

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*(আকার(দশমিক) + আকার(int32_t));

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part(ggml_metal_library_t lib) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, "kernel_argmax_part_f32");
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, "kernel_argmax_part_f32", "kernel_argmax_part_f32", nullptr);
    }

    res.smem = 32*(আকার(দশমিক) + আকার(int32_t));

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final(ggml_metal_library_t lib) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, "kernel_argmax_final_f32");
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, "kernel_argmax_final_f32", "kernel_argmax_final_f32", nullptr);
    }

    res.smem = 32*(আকার(দশমিক) + আকার(int32_t));

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_ARGSORT);

    অক্ষর base[256];
    অক্ষর name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    ধ্রুবক অক্ষর * order_str = "undefined";
    নির্বাচন (order) {
        ক্ষেত্র GGML_SORT_ORDER_ASC:  order_str = "asc";  বিরতি;
        ক্ষেত্র GGML_SORT_ORDER_DESC: order_str = "desc"; বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_ARGSORT);

    অক্ষর base[256];
    অক্ষর name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    ধ্রুবক অক্ষর * order_str = "undefined";
    নির্বাচন (order) {
        ক্ষেত্র GGML_SORT_ORDER_ASC:  order_str = "asc";  বিরতি;
        ক্ষেত্র GGML_SORT_ORDER_DESC: order_str = "desc"; বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht(ggml_metal_library_t lib, পূর্ণসংখ্যা n, ggml_type tsrc) {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_fwht_%s_%d", ggml_type_name(tsrc), n);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

// note: reuse the argsort kernel for the bitonic top_k fallback
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_শীর্ষ_k(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_শীর্ষ_K);

    অক্ষর base[256];
    অক্ষর name[256];

    // note: the top_k kernel is always descending order
    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    ধ্রুবক অক্ষর * order_str = "undefined";
    নির্বাচন (order) {
        ক্ষেত্র GGML_SORT_ORDER_ASC:  order_str = "asc";  বিরতি;
        ক্ষেত্র GGML_SORT_ORDER_DESC: order_str = "desc"; বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_শীর্ষ_k_radix(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_শীর্ষ_K);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_top_k_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_শীর্ষ_k_merge(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_শীর্ষ_K);

    অক্ষর base[256];
    অক্ষর name[256];

    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    ধ্রুবক অক্ষর * order_str = "undefined";
    নির্বাচন (order) {
        ক্ষেত্র GGML_SORT_ORDER_ASC:  order_str = "asc";  বিরতি;
        ক্ষেত্র GGML_SORT_ORDER_DESC: order_str = "desc"; বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe(
        ggml_metal_library_t lib, int32_t n_expert, int32_t শীর্ষ_k, যৌক্তিক with_norm) {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_topk_moe_f32");
    snprintf(name, 256, "%s_n_expert=%d_top_k=%d_with_norm=%d", base, n_expert, শীর্ষ_k, with_norm ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_bool (cv, with_norm, FC_TOPK_MOE + 0);
        ggml_metal_cv_set_int32(cv, n_expert,  FC_TOPK_MOE + 1);
        ggml_metal_cv_set_int32(cv, শীর্ষ_k,     FC_TOPK_MOE + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce(ggml_metal_library_t lib, int32_t n_expert_used) {
    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_moe_reduce_f32");
    snprintf(name, 256, "%s_n_expert_used=%d", base, n_expert_used);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int32(cv, n_expert_used, FC_MOE_REDUCE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ধ্রুবক কাঠামো ggml_tensor * op,
        যৌক্তিক    has_mask,
        int32_t ncpsg) {
    প্রতিজ্ঞা(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_pad");

    snprintf(name, 256, "%s_mask=%d_ncpsg=%d",
            base,
            has_mask,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_PAD + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_PAD + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_PAD + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_PAD + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_PAD + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_PAD + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_PAD + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_PAD + 23);
        //ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_PAD + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_FLASH_ATTN_EXT);

    অক্ষর base[256];

    snprintf(base, 256, "kernel_flash_attn_ext_kv_%s_f16", ggml_type_name(op->src[1]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, base);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ধ্রুবক কাঠামো ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg) {
    প্রতিজ্ঞা(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_blk");

    snprintf(name, 256, "%s_nqptg=%d_ncpsg=%d",
            base,
            nqptg,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        //ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_BLK + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_BLK + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_BLK + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_BLK + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_BLK + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_BLK + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_BLK + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_BLK + 23);
        ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ধ্রুবক ggml_tensor * op,
        যৌক্তিক has_mask,
        যৌক্তিক has_sinks,
        যৌক্তিক has_bias,
        যৌক্তিক has_scap) {
    প্রতিজ্ঞা(op->op == GGML_OP_FLASH_ATTN_EXT);

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক int32_t dk = (int32_t) op->src[1]->ne[0];
    ধ্রুবক int32_t dv = (int32_t) op->src[2]->ne[0];

    ধ্রুবক অক্ষর * type = op->src[1]->type == GGML_TYPE_BF16 ? "bf16" : "f16";

    snprintf(base, 256, "kernel_flash_attn_ext_tensor_%s_dk%d_dv%d", type, dk, dv);
    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d", base, has_mask, has_sinks, has_bias, has_scap);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_TENSOR + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_TENSOR + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_TENSOR + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_TENSOR + 3);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ধ্রুবক ggml_tensor * op,
        যৌক্তিক    has_mask,
        যৌক্তিক    has_sinks,
        যৌক্তিক    has_bias,
        যৌক্তিক    has_scap,
        যৌক্তিক    has_kvpad,
        int32_t nsg,
        যৌক্তিক    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    প্রতিজ্ঞা(op->op == GGML_OP_FLASH_ATTN_EXT);

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক int32_t dk = (int32_t) op->src[1]->ne[0];
    ধ্রুবক int32_t dv = (int32_t) op->src[2]->ne[0];

    ধ্রুবক অক্ষর * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    // do bounds checks for the mask?
    ধ্রুবক যৌক্তিক bc_mask = op->src[3] && (op->src[3]->ne[1] % 8 != 0);

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            type,
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            bc_mask,
            ns10,
            ns20,
            nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);

        ggml_metal_cv_set_bool(cv, bc_mask, FC_FLASH_ATTN_EXT + 10);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_FLASH_ATTN_EXT);
    প্রতিজ্ঞা(op->src[3]);

    অক্ষর name[256];

    snprintf(name, 256, "kernel_flash_attn_ext_vec_idx");

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    GGML_UNUSED(op);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ধ্রুবক ggml_tensor * op,
        যৌক্তিক    has_mask,
        যৌক্তিক    has_sinks,
        যৌক্তিক    has_bias,
        যৌক্তিক    has_scap,
        যৌক্তিক    has_kvpad,
        যৌক্তিক    has_sparse,
        int32_t nqpsg,
        int32_t ne,
        int32_t nsg,
        int32_t nwg,
        যৌক্তিক    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    প্রতিজ্ঞা(op->op == GGML_OP_FLASH_ATTN_EXT);

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক int32_t dk = (int32_t) op->src[1]->ne[0];
    ধ্রুবক int32_t dv = (int32_t) op->src[2]->ne[0];

    ধ্রুবক অক্ষর * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    অক্ষর qne_suffix[16] = {0};
    যদি (!(nqpsg == 1 && ne == ggml_metal_tuning::fa_vec_baseline_ne(dk, dv))) {
        snprintf(qne_suffix, আকার(qne_suffix), "_q%d_ne%d", nqpsg, ne);
    }

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d%s",
            "flash_attn_ext_vec",
            type,
            dk,
            dv,
            qne_suffix);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_sparse=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            has_sparse,
            ns10,
            ns20,
            nsg, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad,  FC_FLASH_ATTN_EXT_VEC + 4);
        ggml_metal_cv_set_bool(cv, has_sparse, FC_FLASH_ATTN_EXT_VEC + 5);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ধ্রুবক ggml_tensor * op,
        int32_t dv,
        int32_t nwg) {
    প্রতিজ্ঞা(op->op == GGML_OP_FLASH_ATTN_EXT);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "%s_dv=%d_nwg=%d", base, dv, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        ggml_metal_cv_set_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;

    GGML_UNUSED(op);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, int32_t n_fuse) {
    অক্ষর base[256];
    অক্ষর name[256];

    পূর্ণসংখ্যা op_অঙ্ক = -1;

    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_ADD: op_অঙ্ক = 0; বিরতি;
        ক্ষেত্র GGML_OP_SUB: op_অঙ্ক = 1; বিরতি;
        ক্ষেত্র GGML_OP_MUL: op_অঙ্ক = 2; বিরতি;
        ক্ষেত্র GGML_OP_DIV: op_অঙ্ক = 3; বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    };

    ধ্রুবক অক্ষর * t0_str = ggml_type_name(op->src[0]->type);
    ধ্রুবক অক্ষর * t1_str = ggml_type_name(op->src[1]->type);
    ধ্রুবক অক্ষর * t_str  = ggml_type_name(op->type);

    ধ্রুবক যৌক্তিক is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    ধ্রুবক যৌক্তিক is_cb = op->src[0]->ne[0] != op->src[1]->ne[0];
    ধ্রুবক যৌক্তিক is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d_cb=%d", base, op_অঙ্ক, n_fuse, is_rb, is_cb);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_অঙ্ক, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, n_fuse, FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, is_rb,  FC_BIN + 2);
        ggml_metal_cv_set_bool (cv, is_cb,  FC_BIN + 3);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_rb;

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one(ggml_metal_library_t lib, ggml_op op) {
    অক্ষর base[256];
    অক্ষর name[256];

    পূর্ণসংখ্যা op_অঙ্ক = -1;

    নির্বাচন (op) {
        ক্ষেত্র GGML_OP_ADD: op_অঙ্ক = 0; বিরতি;
        ক্ষেত্র GGML_OP_SUB: op_অঙ্ক = 1; বিরতি;
        ক্ষেত্র GGML_OP_MUL: op_অঙ্ক = 2; বিরতি;
        ক্ষেত্র GGML_OP_DIV: op_অঙ্ক = 3; বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s", "f32", "f32", "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", base, op_অঙ্ক, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_অঙ্ক, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, মিথ্যা,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_L2_NORM);

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক যৌক্তিক is_c4 = op->src[0]->ne[0] % 4 == 0;

    ধ্রুবক অক্ষর * t0_str = ggml_type_name(op->src[0]->type);
    ধ্রুবক অক্ষর * t_str  = ggml_type_name(op->type);

    snprintf(base, 256, "kernel_l2_norm_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.c4   = is_c4;
    res.smem = 32*আকার(দশমিক);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_GROUP_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*আকার(দশমিক);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, পূর্ণসংখ্যা n_fuse) {
    প্রতিজ্ঞা(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক অক্ষর * suffix = "";
    যদি (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_NORM:
            নির্বাচন (n_fuse) {
                ক্ষেত্র 1: snprintf(base, 256, "kernel_norm_f32%s", suffix);         বিরতি;
                ক্ষেত্র 2: snprintf(base, 256, "kernel_norm_mul_f32%s", suffix);     বিরতি;
                ক্ষেত্র 3: snprintf(base, 256, "kernel_norm_mul_add_f32%s", suffix); বিরতি;
                পূর্বনির্ধারিত: GGML_ABORT("fatal error");
            } বিরতি;
        ক্ষেত্র GGML_OP_RMS_NORM:
            নির্বাচন (n_fuse) {
                ক্ষেত্র 1: snprintf(base, 256, "kernel_rms_norm_f32%s", suffix);         বিরতি;
                ক্ষেত্র 2: snprintf(base, 256, "kernel_rms_norm_mul_f32%s", suffix);     বিরতি;
                ক্ষেত্র 3: snprintf(base, 256, "kernel_rms_norm_mul_add_f32%s", suffix); বিরতি;
                পূর্বনির্ধারিত: GGML_ABORT("fatal error");
            } বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_bool(cv, মিথ্যা, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*আকার(দশমিক);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক অক্ষর * suffix = "";
    যদি (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_NORM:     snprintf(base, 256, "kernel_norm_mul_f32%s", suffix);     বিরতি;
        ক্ষেত্র GGML_OP_RMS_NORM: snprintf(base, 256, "kernel_rms_norm_mul_f32%s", suffix); বিরতি;
        পূর্বনির্ধারিত: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s_use_scale", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_bool(cv, সত্য, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*আকার(দশমিক);

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_ROPE || op->op == GGML_OP_ROPE_BACK);

    ধ্রুবক যৌক্তিক is_back = op->op == GGML_OP_ROPE_BACK;

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক পূর্ণসংখ্যা mode = ((ধ্রুবক int32_t *) op->op_params)[2];

    ধ্রুবক যৌক্তিক is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    ধ্রুবক যৌক্তিক is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    ধ্রুবক যৌক্তিক is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    ধ্রুবক যৌক্তিক is_vision = mode == GGML_ROPE_TYPE_VISION;

    যদি (is_neox) {
        snprintf(base, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } অন্যথা যদি ((is_mrope || is_imrope) && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } অন্যথা যদি (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } অন্যথা {
        snprintf(base, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s_imrope=%d_is_back=%d", base, is_imrope ? 1 : 0, is_back ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, is_imrope, FC_ROPE + 0);
        ggml_metal_cv_set_bool(cv, is_back,   FC_ROPE + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_IM2COL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    ধ্রুবক যৌক্তিক is_2D = ((ধ্রুবক int32_t *)(op->op_params))[6] == 1;
    ধ্রুবক int64_t KH = is_2D ? ne01 : 1;
    ধ্রুবক int64_t KW = ne00;

    অক্ষর base[256];
    অক্ষর name[256];

    যদি (KH*KW <= 1024) {
        snprintf(base, 256, "kernel_im2col_%s", ggml_type_name(op->type));
    } অন্যথা {
        snprintf(base, 256, "kernel_im2col_ext_%s", ggml_type_name(op->type));
    }
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_COL2IM_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_col2im_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake(ggml_metal_library_t lib, গণনা ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_snake_%s", ggml_type_name(type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_conv_transpose_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_CONV_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_conv_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op, যৌক্তিক tiled) {
    প্রতিজ্ঞা(op->op == GGML_OP_CONV_2D_DW);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_conv_2d_dw%s_%s_%s",
             tiled ? "_tiled" : "",
             ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_CONV_3D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_conv_3d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_UPSCALE);

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক int32_t mode_flags = ggml_get_op_params_i32(op, 0);
    ধ্রুবক ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    ধ্রুবক যৌক্তিক antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);

    যদি (mode == GGML_SCALE_MODE_BILINEAR) {
        snprintf(base, 256, "kernel_upscale_bilinear_%s", ggml_type_name(op->src[0]->type));
    } অন্যথা যদি (mode == GGML_SCALE_MODE_BICUBIC) {
        snprintf(base, 256, "kernel_upscale_bicubic_%s", ggml_type_name(op->src[0]->type));
    } অন্যথা {
        snprintf(base, 256, "kernel_upscale_nearest_%s", ggml_type_name(op->src[0]->type));
    }
    snprintf(name, 256, "%s_aa=%d", base, antialias);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, antialias, FC_UPSCALE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_ROLL);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_roll_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_PAD);

    অক্ষর base[256];
    অক্ষর name[256];

    ধ্রুবক যৌক্তিক circular = ggml_get_op_params_i32(op, 8) != 0;

    snprintf(base, 256, "kernel_pad_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_circular=%d", base, circular);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, circular, FC_PAD + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_PAD_REFLECT_1D);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_ARANGE);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_OPT_STEP_ADAMW);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_OPT_STEP_SGD);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back(ggml_metal_library_t lib, ধ্রুবক ggml_tensor * op) {
    প্রতিজ্ঞা(op->op == GGML_OP_SILU_BACK);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_silu_back_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, ধ্রুবক ggml_tensor *  op) {
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    অক্ষর base[256];
    অক্ষর name[256];

    snprintf(base, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    ফেরত res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_গুনতি_equal(ggml_metal_library_t lib, ধ্রুবক ggml_tensor *  op) {
    প্রতিজ্ঞা(op->op == GGML_OP_গুনতি_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    অক্ষর base[256];
    অক্ষর name[256];

    পূর্ণসংখ্যা nsg = 1;
    যখন (32*nsg < ne00 && nsg < 32) {
        nsg *= 2;
    }

    snprintf(base, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    যদি (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_গুনতি_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32 * আকার(int32_t);
    res.nsg  = nsg;

    ফেরত res;
}
