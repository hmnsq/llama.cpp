#اگر_تعریف_نشده HTP_OPNODE_H
#تعریف_کن HTP_OPNODE_H

#تعریف_کن GGML_COMMON_IMPL_CPP
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-common.h"

#ضمن <algorithm>
#ضمن <string>
#ضمن <vector>
#ضمن <memory>
#ضمن <stdio.h>
#ضمن "htp-ops.h"
#ضمن "htp/matmul-ops.h"
#ضمن "htp/flash-attn-ops.h"
#ضمن "htp/unary-ops.h"
#ضمن "htp/binary-ops.h"
#ضمن "htp/allreduce-ops.h"
#ضمن "htp/ssm-conv.h"
#ضمن "htp/gated-delta-net-ops.h"
#ضمن "htp/softmax-ops.h"
#ضمن "htp/argsort-ops.h"
#ضمن "htp/get-rows-ops.h"

بنية htp_opnode {
    ggml_tensor * گره   { nullptr };
    htp_op_کد   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::وکتور<ggml_tensor *>                fused;
    std::وکتور<std::shared_ptr<ggml_tensor>> dummy;

    std::وکتور<ثابت ggml_tensor *> inputs;
    std::وکتور<ثابت ggml_tensor *> outputs;
    std::string                      name;

    صحيح n_فعال_src(ثابت ggml_tensor * t) ثابت {
        إذا (!t) أرجع 0;
        لكل (صحيح i = GGML_MAX_SRC - 1; i >= 0; i--) {
            إذا (t->src[i]) {
                أرجع i + 1;
            }
        }
        أرجع 0;
    }

    فراغ init(ggml_tensor * گره) {
        this->گره = گره;
        إذا (this->گره) {
            إذا (this->گره->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->گره, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.افزودن_پشت(this->گره->src[1]);
            } وإلا {
                this->name = ggml_op_desc(this->گره);

                // Build inputs (preserving optional nullptrs)
                صحيح n_inputs = n_فعال_src(this->گره);
                this->inputs.resize(n_inputs, nullptr);
                لكل (صحيح i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->گره->src[i];
                }
            }

            // Build outputs
            this->outputs.افزودن_پشت(this->dst());
        }
    }

    htp_opnode(htp_op_کد opcode = HTP_OP_INVALID, ggml_tensor * گره = nullptr) : opcode(opcode) {
        init(گره);
    }

    ggml_op             op()   ثابت { أرجع گره->op; }
    ثابت ggml_tensor * src0() ثابت { أرجع گره->src[0]; }
    ثابت ggml_tensor * src1() ثابت { أرجع گره->src[1]; }
    ثابت ggml_tensor * dst()  ثابت { أرجع outputs.خالی() ? گره : outputs.پشت(); }

    ggml_tensor * جمع_کن_dummy(ثابت ggml_tensor & t) {
        dummy.افزودن_پشت(std::make_shared<ggml_tensor>(t));
        أرجع dummy.پشت().get();
    }

    فراغ جمع_کن_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.افزودن_پشت(t);

        name += "+";
        name += ggml_op_desc(t);

        إذا (extra_dst) {
            outputs.افزودن_پشت(t);
        } وإلا {
            outputs.clear();
            outputs.افزودن_پشت(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.انتها(), t), inputs.انتها());

        // Append new inputs from t, preserving middle nullptrs
        صحيح n_inputs = n_فعال_src(t);
        لكل (صحيح i = 0; i < n_inputs; i++) {
            ثابت خودکار * src = t->src[i];
            إذا (!src) {
                inputs.افزودن_پشت(nullptr);
            } وإلا إذا (src != گره &&
                       std::پیدا_کن(fused.begin(), fused.انتها(), src) == fused.انتها() &&
                       std::پیدا_کن(inputs.begin(), inputs.انتها(), src) == inputs.انتها()) {
                inputs.افزودن_پشت(src);
            }
        }
    }

    ثابت std::وکتور<ثابت ggml_tensor *> & get_inputs() ثابت {
        أرجع inputs;
    }

    ثابت std::وکتور<ثابت ggml_tensor *> & get_outputs() ثابت {
        أرجع outputs;
    }

    std::string op_name() ثابت {
        أرجع name;
    }

    bool is_خالی() ثابت {
        أرجع ggml_op_is_خالی(گره->op);
    }

    bool stackable() ثابت {
        بدل (this->op()) {
            حالة GGML_OP_MUL_MAT:
                إذا (ggml_get_op_params_i32(this->گره, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    أرجع false;
                }
                أرجع ggml_is_quantized(this->src0()->type);
            حالة GGML_OP_MUL_MAT_ID:
                أرجع ggml_is_quantized(this->src0()->type);
            افتراضي:
                أرجع false;
        }
    }

    bool same_درون_داد(ثابت htp_opnode& n) ثابت {
        أرجع n.src1() == this->src1();
    }
};

بنية htp_opformat {
    حرف strides[64 * GGML_MAX_SRC];
    حرف dims[64 * GGML_MAX_SRC];
    حرف types[16 * GGML_MAX_SRC];
    حرف buffs[64 * GGML_MAX_SRC];
    حرف نامها[64 * GGML_MAX_SRC];
    حرف kparams[128];

    صحيح format_tensor_dims(حرف * str, size_t max_اندازه, ثابت بنية ggml_tensor * t) {
        إذا (!t) {
            أرجع snprintf(str, max_اندازه, "NONE");
        }
        إذا (t->ne[2] == 1 && t->ne[3] == 1) {
            أرجع snprintf(str, max_اندازه, "%d:%d", (صحيح) t->ne[0], (صحيح) t->ne[1]);
        } وإلا {
            أرجع snprintf(str, max_اندازه, "%d:%d:%d:%d", (صحيح) t->ne[0], (صحيح) t->ne[1], (صحيح) t->ne[2], (صحيح) t->ne[3]);
        }
    }

    فراغ format_op_dims(حرف * str, size_t max_اندازه, ثابت htp_opnode & گره) {
        حرف * p = str;
        حرف * p_انتها = str + max_اندازه;
        خودکار inputs = گره.get_inputs();

        إذا (!inputs.خالی()) {
            p += std::min((size_t)format_tensor_dims(p, p_انتها - p, inputs[0]), (size_t)(p_انتها - p));

            لكل (size_t i = 1; i < inputs.اندازه(); i++) {
                إذا (p < p_انتها) {
                    p += std::min((size_t)snprintf(p, p_انتها - p, " x "), (size_t)(p_انتها - p));
                }
                إذا (p < p_انتها) {
                    p += std::min((size_t)format_tensor_dims(p, p_انتها - p, inputs[i]), (size_t)(p_انتها - p));
                }
            }

            إذا (p < p_انتها) {
                p += std::min((size_t)snprintf(p, p_انتها - p, " -> "), (size_t)(p_انتها - p));
            }
        }

        حرف self[64];
        format_tensor_dims(self, حجم(self), گره.dst());
        إذا (p < p_انتها) {
            p += std::min((size_t)snprintf(p, p_انتها - p, "%s", self), (size_t)(p_انتها - p));
        }
    }

    صحيح format_tensor_strides(حرف * str, size_t max_اندازه, ثابت بنية ggml_tensor * t) {
        إذا (!t) {
            أرجع snprintf(str, max_اندازه, "NONE");
        }
        ثابت حرف * c = ggml_is_contiguous(t) ? "" : "!";

        إذا (t->ne[2] == 1 && t->ne[3] == 1) {
            أرجع snprintf(str, max_اندازه, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } وإلا {
            أرجع snprintf(str, max_اندازه, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    فراغ format_op_strides(حرف * str, size_t max_اندازه, ثابت htp_opnode & گره) {
        حرف * p = str;
        حرف * p_انتها = str + max_اندازه;
        خودکار inputs = گره.get_inputs();

        إذا (!inputs.خالی()) {
            p += std::min((size_t)format_tensor_strides(p, p_انتها - p, inputs[0]), (size_t)(p_انتها - p));

            لكل (size_t i = 1; i < inputs.اندازه(); i++) {
                إذا (p < p_انتها) {
                    p += std::min((size_t)snprintf(p, p_انتها - p, " x "), (size_t)(p_انتها - p));
                }
                إذا (p < p_انتها) {
                    p += std::min((size_t)format_tensor_strides(p, p_انتها - p, inputs[i]), (size_t)(p_انتها - p));
                }
            }

            إذا (p < p_انتها) {
                p += std::min((size_t)snprintf(p, p_انتها - p, " -> "), (size_t)(p_انتها - p));
            }
        }

        حرف self[64];
        format_tensor_strides(self, حجم(self), گره.dst());
        إذا (p < p_انتها) {
            p += std::min((size_t)snprintf(p, p_انتها - p, "%s", self), (size_t)(p_انتها - p));
        }
    }

    فراغ format_op_types(حرف * str, size_t max_اندازه, ثابت htp_opnode & گره) {
        حرف * p = str;
        حرف * p_انتها = str + max_اندازه;
        خودکار inputs = گره.get_inputs();

        إذا (!inputs.خالی()) {
            إذا (p < p_انتها) {
                p += std::min((size_t)snprintf(p, p_انتها - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_انتها - p));
            }

            لكل (size_t i = 1; i < inputs.اندازه(); i++) {
                إذا (p < p_انتها) {
                    p += std::min((size_t)snprintf(p, p_انتها - p, " x "), (size_t)(p_انتها - p));
                }
                إذا (p < p_انتها) {
                    p += std::min((size_t)snprintf(p, p_انتها - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_انتها - p));
                }
            }

            إذا (p < p_انتها) {
                p += std::min((size_t)snprintf(p, p_انتها - p, " -> "), (size_t)(p_انتها - p));
            }
        }

        إذا (p < p_انتها) {
            p += std::min((size_t)snprintf(p, p_انتها - p, "%s", ggml_type_name(گره.dst()->type)), (size_t)(p_انتها - p));
        }
    }

    ثابت حرف * tensor_buff_name(ثابت بنية ggml_tensor * t) {
        إذا (t && t->میانگیر) {
            أرجع ggml_backend_میانگیر_name(t->میانگیر);
        }
        أرجع "NONE";
    }

    فراغ format_op_buffs(حرف * str, size_t max_اندازه, ثابت htp_opnode & گره) {
        حرف * p = str;
        حرف * p_انتها = str + max_اندازه;
        خودکار inputs = گره.get_inputs();

        إذا (!inputs.خالی()) {
            إذا (p < p_انتها) {
                p += std::min((size_t)snprintf(p, p_انتها - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_انتها - p));
            }

            لكل (size_t i = 1; i < inputs.اندازه(); i++) {
                إذا (p < p_انتها) {
                    p += std::min((size_t)snprintf(p, p_انتها - p, " x "), (size_t)(p_انتها - p));
                }
                إذا (p < p_انتها) {
                    p += std::min((size_t)snprintf(p, p_انتها - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_انتها - p));
                }
            }

            إذا (p < p_انتها) {
                p += std::min((size_t)snprintf(p, p_انتها - p, " -> "), (size_t)(p_انتها - p));
            }
        }

        إذا (p < p_انتها) {
            p += std::min((size_t)snprintf(p, p_انتها - p, "%s", tensor_buff_name(گره.dst())), (size_t)(p_انتها - p));
        }
    }

    فراغ format_op_نامها(حرف * str, size_t max_اندازه, ثابت htp_opnode & گره) {
        حرف * p = str;
        حرف * p_انتها = str + max_اندازه;
        خودکار inputs = گره.get_inputs();

        إذا (!inputs.خالی()) {
            إذا (p < p_انتها) {
                p += std::min((size_t)snprintf(p, p_انتها - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_انتها - p));
            }

            لكل (size_t i = 1; i < inputs.اندازه(); i++) {
                إذا (p < p_انتها) {
                    p += std::min((size_t)snprintf(p, p_انتها - p, " x "), (size_t)(p_انتها - p));
                }
                إذا (p < p_انتها) {
                    p += std::min((size_t)snprintf(p, p_انتها - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_انتها - p));
                }
            }

            إذا (p < p_انتها) {
                p += std::min((size_t)snprintf(p, p_انتها - p, " -> "), (size_t)(p_انتها - p));
            }
        }

        إذا (p < p_انتها) {
            p += std::min((size_t)snprintf(p, p_انتها - p, "%s", گره.dst()->name), (size_t)(p_انتها - p));
        }
    }
    فراغ format_kernel_params(حرف * str, size_t max_اندازه, ثابت htp_opnode & گره) {
        إذا (گره.opcode == HTP_OP_MUL_MAT || گره.opcode == HTP_OP_MUL_MAT_ID ||
            گره.opcode == HTP_OP_MUL_MAT_NX || گره.opcode == HTP_OP_MUL_MAT_ID_NX ||
            گره.opcode == HTP_OP_MUL_MAT_جمع_کن) {
            ثابت خودکار * kparams = (ثابت بنية htp_mm_kernel_params *) گره.kernel_params;
            ثابت حرف * مسیر = "unknown";
            int32_t type = kparams->kernel_type;
            إذا (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                مسیر = (kparams->tile_اندازه > 0) ? "hmx-tiled" : "hmx-flat";
            } وإلا إذا (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                مسیر = "hvx-flat";
            } وإلا إذا (type == HTP_MM_KERNEL_HVX_QUANT_ردیف || type == HTP_MM_KERNEL_HVX_QUANT_بلوک) {
                مسیر = "hvx-tiled";
            }
            snprintf(str, max_اندازه, "%s vtcm %d", مسیر, (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (گره.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ثابت خودکار * kparams = (ثابت بنية htp_fa_kernel_params *) گره.kernel_params;
            ثابت حرف * مسیر = "unknown";
            int32_t type = kparams->kernel_type;
            إذا (type == HTP_FA_KERNEL_HMX) {
                مسیر = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } وإلا إذا (type == HTP_FA_KERNEL_HVX) {
                مسیر = "hvx";
            }
            snprintf(str, max_اندازه, "%s vtcm %d", مسیر, (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (htp_op_is_unary(گره.opcode)) {
            ثابت خودکار * kparams = (ثابت بنية htp_unary_kernel_params *) گره.kernel_params;
            snprintf(str, max_اندازه, "%s vtcm %d", kparams->ستون_tile ? "wide-row" : "row-block", (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (گره.opcode == HTP_OP_MDEV_گروه && گره.گره) {
            snprintf(str, max_اندازه, "idx %d count %d", (صحيح) گره.گره->op_params[0], (صحيح) گره.dst()->ne[1]);
        } وإلا إذا ((گره.opcode == HTP_OP_FENCE || گره.opcode == HTP_OP_CPY_FENCE) && گره.گره) {
            snprintf(str, max_اندازه, "seq 0x%x", (uint32_t) گره.گره->op_params[0]);
        } وإلا إذا (گره.opcode == HTP_OP_ALLREDUCE && گره.گره) {
            snprintf(str, max_اندازه, "seq 0x%x -> 0x%x", (uint32_t) گره.گره->op_params[0], (uint32_t) گره.گره->op_params[1]);
        } وإلا إذا (گره.opcode == HTP_OP_SSM_CONV) {
            ثابت خودکار * kparams = (ثابت بنية htp_ssm_conv_kernel_params *) گره.kernel_params;
            snprintf(str, max_اندازه, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (گره.opcode == HTP_OP_SOFTMAX) {
            ثابت خودکار * kparams = (ثابت بنية htp_softmax_kernel_params *) گره.kernel_params;
            snprintf(str, max_اندازه, "k%d nth %d vtcm %d", (صحيح) kparams->kernel_id, (صحيح) kparams->n_نخها, (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (گره.opcode == HTP_OP_GATED_DELTA_NET) {
            ثابت خودکار * kparams = (ثابت بنية htp_gdn_kernel_params *) گره.kernel_params;
            ثابت حرف * مسیر = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_اندازه, "%s-%s vtcm %u", مسیر, kparams->kda ? "kda" : "scalar",
                     (بیعلامت صحيح) (kparams->vtcm_اندازه ? kparams->vtcm_اندازه : kparams->vtcm_در_هر_thread * kparams->n_نخها));
        } وإلا إذا (گره.opcode == HTP_OP_MUL || گره.opcode == HTP_OP_جمع_کن || گره.opcode == HTP_OP_جمع_کن_ID ||
                   گره.opcode == HTP_OP_SUB || گره.opcode == HTP_OP_DIV) {
            ثابت خودکار * kparams = (ثابت بنية htp_دودویی_kernel_params *) گره.kernel_params;
            snprintf(str, max_اندازه, "vtcm %u", (بیعلامت صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (گره.opcode == HTP_OP_ARGSORT || گره.opcode == HTP_OP_بالاترین_K) {
            ثابت خودکار * kparams = (ثابت بنية htp_مرتب_سازی_kernel_params *) گره.kernel_params;
            snprintf(str, max_اندازه, "%s nth %d nchk %d chk %d vtcm %d",
                     گره.opcode == HTP_OP_بالاترین_K ? "top_k" : "argsort",
                     (صحيح) kparams->n_نخها, (صحيح) kparams->n_chunks,
                     (صحيح) kparams->chunk_elems, (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (گره.opcode == HTP_OP_GET_ردیفها) {
            ثابت خودکار * kparams = (ثابت بنية htp_get_ردیفها_kernel_params *) گره.kernel_params;
            ثابت حرف * ktype_str = "unknown";
            بدل (kparams->kernel_type) {
                حالة HTP_GET_ردیفها_KERNEL_SAMETYPE: ktype_str = "sametype"; اكسر;
                حالة HTP_GET_ردیفها_KERNEL_TILED:    ktype_str = "tiled"; اكسر;
                حالة HTP_GET_ردیفها_KERNEL_FLAT:     ktype_str = "flat"; اكسر;
            }
            snprintf(str, max_اندازه, "%s%s vtcm %u", ktype_str, kparams->n_نخها > 1 ? "-multi" : "", (بیعلامت صحيح) kparams->vtcm_اندازه);
        } وإلا {
            snprintf(str, max_اندازه, "----");
        }

    }

    فراغ format(ثابت htp_opnode & گره) {
        format_op_dims(dims, حجم(dims), گره);
        format_op_strides(strides, حجم(strides), گره);
        format_op_types(types, حجم(types), گره);
        format_op_buffs(buffs, حجم(buffs), گره);
        format_op_نامها(نامها, حجم(نامها), گره);
        format_kernel_params(kparams, حجم(kparams), گره);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        نامها[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ثابت htp_opnode & گره) { format(گره); }
};

#پایان_اگر // HTP_OPNODE_H
