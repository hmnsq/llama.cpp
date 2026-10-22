#که_نه_تعریف_شوی HTP_OPNODE_H
#تعریف_کړه HTP_OPNODE_H

#تعریف_کړه GGML_COMMON_IMPL_CPP
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
    ggml_tensor * غوټه   { nullptr };
    htp_op_کوډ   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::ویکټور<ggml_tensor *>                fused;
    std::ویکټور<std::shared_ptr<ggml_tensor>> dummy;

    std::ویکټور<ثابت ggml_tensor *> inputs;
    std::ویکټور<ثابت ggml_tensor *> outputs;
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

    فراغ init(ggml_tensor * غوټه) {
        this->غوټه = غوټه;
        إذا (this->غوټه) {
            إذا (this->غوټه->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->غوټه, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ټېل_شا(this->غوټه->src[1]);
            } وإلا {
                this->name = ggml_op_desc(this->غوټه);

                // Build inputs (preserving optional nullptrs)
                صحيح n_inputs = n_فعال_src(this->غوټه);
                this->inputs.resize(n_inputs, nullptr);
                لكل (صحيح i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->غوټه->src[i];
                }
            }

            // Build outputs
            this->outputs.ټېل_شا(this->dst());
        }
    }

    htp_opnode(htp_op_کوډ opcode = HTP_OP_INVALID, ggml_tensor * غوټه = nullptr) : opcode(opcode) {
        init(غوټه);
    }

    ggml_op             op()   ثابت { أرجع غوټه->op; }
    ثابت ggml_tensor * src0() ثابت { أرجع غوټه->src[0]; }
    ثابت ggml_tensor * src1() ثابت { أرجع غوټه->src[1]; }
    ثابت ggml_tensor * dst()  ثابت { أرجع outputs.تشه() ? غوټه : outputs.شا(); }

    ggml_tensor * جمع_کړه_dummy(ثابت ggml_tensor & t) {
        dummy.ټېل_شا(std::make_shared<ggml_tensor>(t));
        أرجع dummy.شا().get();
    }

    فراغ جمع_کړه_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ټېل_شا(t);

        name += "+";
        name += ggml_op_desc(t);

        إذا (extra_dst) {
            outputs.ټېل_شا(t);
        } وإلا {
            outputs.clear();
            outputs.ټېل_شا(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.پای_ټکی(), t), inputs.پای_ټکی());

        // Append new inputs from t, preserving middle nullptrs
        صحيح n_inputs = n_فعال_src(t);
        لكل (صحيح i = 0; i < n_inputs; i++) {
            ثابت خپلکاری * src = t->src[i];
            إذا (!src) {
                inputs.ټېل_شا(nullptr);
            } وإلا إذا (src != غوټه &&
                       std::پیدا_کړه(fused.begin(), fused.پای_ټکی(), src) == fused.پای_ټکی() &&
                       std::پیدا_کړه(inputs.begin(), inputs.پای_ټکی(), src) == inputs.پای_ټکی()) {
                inputs.ټېل_شا(src);
            }
        }
    }

    ثابت std::ویکټور<ثابت ggml_tensor *> & get_inputs() ثابت {
        أرجع inputs;
    }

    ثابت std::ویکټور<ثابت ggml_tensor *> & get_outputs() ثابت {
        أرجع outputs;
    }

    std::string op_name() ثابت {
        أرجع name;
    }

    bool is_تشه() ثابت {
        أرجع ggml_op_is_تشه(غوټه->op);
    }

    bool stackable() ثابت {
        بدل (this->op()) {
            حالة GGML_OP_MUL_MAT:
                إذا (ggml_get_op_params_i32(this->غوټه, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    أرجع false;
                }
                أرجع ggml_is_quantized(this->src0()->type);
            حالة GGML_OP_MUL_MAT_ID:
                أرجع ggml_is_quantized(this->src0()->type);
            افتراضي:
                أرجع false;
        }
    }

    bool same_ننوت(ثابت htp_opnode& n) ثابت {
        أرجع n.src1() == this->src1();
    }
};

بنية htp_opformat {
    حرف strides[64 * GGML_MAX_SRC];
    حرف dims[64 * GGML_MAX_SRC];
    حرف types[16 * GGML_MAX_SRC];
    حرف buffs[64 * GGML_MAX_SRC];
    حرف نومونه[64 * GGML_MAX_SRC];
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

    فراغ format_op_dims(حرف * str, size_t max_اندازه, ثابت htp_opnode & غوټه) {
        حرف * p = str;
        حرف * p_پای_ټکی = str + max_اندازه;
        خپلکاری inputs = غوټه.get_inputs();

        إذا (!inputs.تشه()) {
            p += std::min((size_t)format_tensor_dims(p, p_پای_ټکی - p, inputs[0]), (size_t)(p_پای_ټکی - p));

            لكل (size_t i = 1; i < inputs.اندازه(); i++) {
                إذا (p < p_پای_ټکی) {
                    p += std::min((size_t)snprintf(p, p_پای_ټکی - p, " x "), (size_t)(p_پای_ټکی - p));
                }
                إذا (p < p_پای_ټکی) {
                    p += std::min((size_t)format_tensor_dims(p, p_پای_ټکی - p, inputs[i]), (size_t)(p_پای_ټکی - p));
                }
            }

            إذا (p < p_پای_ټکی) {
                p += std::min((size_t)snprintf(p, p_پای_ټکی - p, " -> "), (size_t)(p_پای_ټکی - p));
            }
        }

        حرف self[64];
        format_tensor_dims(self, حجم(self), غوټه.dst());
        إذا (p < p_پای_ټکی) {
            p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", self), (size_t)(p_پای_ټکی - p));
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

    فراغ format_op_strides(حرف * str, size_t max_اندازه, ثابت htp_opnode & غوټه) {
        حرف * p = str;
        حرف * p_پای_ټکی = str + max_اندازه;
        خپلکاری inputs = غوټه.get_inputs();

        إذا (!inputs.تشه()) {
            p += std::min((size_t)format_tensor_strides(p, p_پای_ټکی - p, inputs[0]), (size_t)(p_پای_ټکی - p));

            لكل (size_t i = 1; i < inputs.اندازه(); i++) {
                إذا (p < p_پای_ټکی) {
                    p += std::min((size_t)snprintf(p, p_پای_ټکی - p, " x "), (size_t)(p_پای_ټکی - p));
                }
                إذا (p < p_پای_ټکی) {
                    p += std::min((size_t)format_tensor_strides(p, p_پای_ټکی - p, inputs[i]), (size_t)(p_پای_ټکی - p));
                }
            }

            إذا (p < p_پای_ټکی) {
                p += std::min((size_t)snprintf(p, p_پای_ټکی - p, " -> "), (size_t)(p_پای_ټکی - p));
            }
        }

        حرف self[64];
        format_tensor_strides(self, حجم(self), غوټه.dst());
        إذا (p < p_پای_ټکی) {
            p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", self), (size_t)(p_پای_ټکی - p));
        }
    }

    فراغ format_op_types(حرف * str, size_t max_اندازه, ثابت htp_opnode & غوټه) {
        حرف * p = str;
        حرف * p_پای_ټکی = str + max_اندازه;
        خپلکاری inputs = غوټه.get_inputs();

        إذا (!inputs.تشه()) {
            إذا (p < p_پای_ټکی) {
                p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_پای_ټکی - p));
            }

            لكل (size_t i = 1; i < inputs.اندازه(); i++) {
                إذا (p < p_پای_ټکی) {
                    p += std::min((size_t)snprintf(p, p_پای_ټکی - p, " x "), (size_t)(p_پای_ټکی - p));
                }
                إذا (p < p_پای_ټکی) {
                    p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_پای_ټکی - p));
                }
            }

            إذا (p < p_پای_ټکی) {
                p += std::min((size_t)snprintf(p, p_پای_ټکی - p, " -> "), (size_t)(p_پای_ټکی - p));
            }
        }

        إذا (p < p_پای_ټکی) {
            p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", ggml_type_name(غوټه.dst()->type)), (size_t)(p_پای_ټکی - p));
        }
    }

    ثابت حرف * tensor_buff_name(ثابت بنية ggml_tensor * t) {
        إذا (t && t->بفر) {
            أرجع ggml_backend_بفر_name(t->بفر);
        }
        أرجع "NONE";
    }

    فراغ format_op_buffs(حرف * str, size_t max_اندازه, ثابت htp_opnode & غوټه) {
        حرف * p = str;
        حرف * p_پای_ټکی = str + max_اندازه;
        خپلکاری inputs = غوټه.get_inputs();

        إذا (!inputs.تشه()) {
            إذا (p < p_پای_ټکی) {
                p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_پای_ټکی - p));
            }

            لكل (size_t i = 1; i < inputs.اندازه(); i++) {
                إذا (p < p_پای_ټکی) {
                    p += std::min((size_t)snprintf(p, p_پای_ټکی - p, " x "), (size_t)(p_پای_ټکی - p));
                }
                إذا (p < p_پای_ټکی) {
                    p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_پای_ټکی - p));
                }
            }

            إذا (p < p_پای_ټکی) {
                p += std::min((size_t)snprintf(p, p_پای_ټکی - p, " -> "), (size_t)(p_پای_ټکی - p));
            }
        }

        إذا (p < p_پای_ټکی) {
            p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", tensor_buff_name(غوټه.dst())), (size_t)(p_پای_ټکی - p));
        }
    }

    فراغ format_op_نومونه(حرف * str, size_t max_اندازه, ثابت htp_opnode & غوټه) {
        حرف * p = str;
        حرف * p_پای_ټکی = str + max_اندازه;
        خپلکاری inputs = غوټه.get_inputs();

        إذا (!inputs.تشه()) {
            إذا (p < p_پای_ټکی) {
                p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_پای_ټکی - p));
            }

            لكل (size_t i = 1; i < inputs.اندازه(); i++) {
                إذا (p < p_پای_ټکی) {
                    p += std::min((size_t)snprintf(p, p_پای_ټکی - p, " x "), (size_t)(p_پای_ټکی - p));
                }
                إذا (p < p_پای_ټکی) {
                    p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_پای_ټکی - p));
                }
            }

            إذا (p < p_پای_ټکی) {
                p += std::min((size_t)snprintf(p, p_پای_ټکی - p, " -> "), (size_t)(p_پای_ټکی - p));
            }
        }

        إذا (p < p_پای_ټکی) {
            p += std::min((size_t)snprintf(p, p_پای_ټکی - p, "%s", غوټه.dst()->name), (size_t)(p_پای_ټکی - p));
        }
    }
    فراغ format_kernel_params(حرف * str, size_t max_اندازه, ثابت htp_opnode & غوټه) {
        إذا (غوټه.opcode == HTP_OP_MUL_MAT || غوټه.opcode == HTP_OP_MUL_MAT_ID ||
            غوټه.opcode == HTP_OP_MUL_MAT_NX || غوټه.opcode == HTP_OP_MUL_MAT_ID_NX ||
            غوټه.opcode == HTP_OP_MUL_MAT_جمع_کړه) {
            ثابت خپلکاری * kparams = (ثابت بنية htp_mm_kernel_params *) غوټه.kernel_params;
            ثابت حرف * لار = "unknown";
            int32_t type = kparams->kernel_type;
            إذا (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                لار = (kparams->tile_اندازه > 0) ? "hmx-tiled" : "hmx-flat";
            } وإلا إذا (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                لار = "hvx-flat";
            } وإلا إذا (type == HTP_MM_KERNEL_HVX_QUANT_لیکه || type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
                لار = "hvx-tiled";
            }
            snprintf(str, max_اندازه, "%s vtcm %d", لار, (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (غوټه.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ثابت خپلکاری * kparams = (ثابت بنية htp_fa_kernel_params *) غوټه.kernel_params;
            ثابت حرف * لار = "unknown";
            int32_t type = kparams->kernel_type;
            إذا (type == HTP_FA_KERNEL_HMX) {
                لار = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } وإلا إذا (type == HTP_FA_KERNEL_HVX) {
                لار = "hvx";
            }
            snprintf(str, max_اندازه, "%s vtcm %d", لار, (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (htp_op_is_unary(غوټه.opcode)) {
            ثابت خپلکاری * kparams = (ثابت بنية htp_unary_kernel_params *) غوټه.kernel_params;
            snprintf(str, max_اندازه, "%s vtcm %d", kparams->ستون_tile ? "wide-row" : "row-block", (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (غوټه.opcode == HTP_OP_MDEV_ډله && غوټه.غوټه) {
            snprintf(str, max_اندازه, "idx %d count %d", (صحيح) غوټه.غوټه->op_params[0], (صحيح) غوټه.dst()->ne[1]);
        } وإلا إذا ((غوټه.opcode == HTP_OP_FENCE || غوټه.opcode == HTP_OP_CPY_FENCE) && غوټه.غوټه) {
            snprintf(str, max_اندازه, "seq 0x%x", (uint32_t) غوټه.غوټه->op_params[0]);
        } وإلا إذا (غوټه.opcode == HTP_OP_ALLREDUCE && غوټه.غوټه) {
            snprintf(str, max_اندازه, "seq 0x%x -> 0x%x", (uint32_t) غوټه.غوټه->op_params[0], (uint32_t) غوټه.غوټه->op_params[1]);
        } وإلا إذا (غوټه.opcode == HTP_OP_SSM_CONV) {
            ثابت خپلکاری * kparams = (ثابت بنية htp_ssm_conv_kernel_params *) غوټه.kernel_params;
            snprintf(str, max_اندازه, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (غوټه.opcode == HTP_OP_SOFTMAX) {
            ثابت خپلکاری * kparams = (ثابت بنية htp_softmax_kernel_params *) غوټه.kernel_params;
            snprintf(str, max_اندازه, "k%d nth %d vtcm %d", (صحيح) kparams->kernel_id, (صحيح) kparams->n_تارونه, (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (غوټه.opcode == HTP_OP_GATED_DELTA_NET) {
            ثابت خپلکاری * kparams = (ثابت بنية htp_gdn_kernel_params *) غوټه.kernel_params;
            ثابت حرف * لار = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_اندازه, "%s-%s vtcm %u", لار, kparams->kda ? "kda" : "scalar",
                     (بینښه صحيح) (kparams->vtcm_اندازه ? kparams->vtcm_اندازه : kparams->vtcm_فی_thread * kparams->n_تارونه));
        } وإلا إذا (غوټه.opcode == HTP_OP_MUL || غوټه.opcode == HTP_OP_جمع_کړه || غوټه.opcode == HTP_OP_جمع_کړه_ID ||
                   غوټه.opcode == HTP_OP_SUB || غوټه.opcode == HTP_OP_DIV) {
            ثابت خپلکاری * kparams = (ثابت بنية htp_دوګونی_kernel_params *) غوټه.kernel_params;
            snprintf(str, max_اندازه, "vtcm %u", (بینښه صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (غوټه.opcode == HTP_OP_ARGSORT || غوټه.opcode == HTP_OP_سر_K) {
            ثابت خپلکاری * kparams = (ثابت بنية htp_ترتیبول_kernel_params *) غوټه.kernel_params;
            snprintf(str, max_اندازه, "%s nth %d nchk %d chk %d vtcm %d",
                     غوټه.opcode == HTP_OP_سر_K ? "top_k" : "argsort",
                     (صحيح) kparams->n_تارونه, (صحيح) kparams->n_chunks,
                     (صحيح) kparams->chunk_elems, (صحيح) kparams->vtcm_اندازه);
        } وإلا إذا (غوټه.opcode == HTP_OP_GET_لیکې) {
            ثابت خپلکاری * kparams = (ثابت بنية htp_get_لیکې_kernel_params *) غوټه.kernel_params;
            ثابت حرف * ktype_str = "unknown";
            بدل (kparams->kernel_type) {
                حالة HTP_GET_لیکې_KERNEL_SAMETYPE: ktype_str = "sametype"; اكسر;
                حالة HTP_GET_لیکې_KERNEL_TILED:    ktype_str = "tiled"; اكسر;
                حالة HTP_GET_لیکې_KERNEL_FLAT:     ktype_str = "flat"; اكسر;
            }
            snprintf(str, max_اندازه, "%s%s vtcm %u", ktype_str, kparams->n_تارونه > 1 ? "-multi" : "", (بینښه صحيح) kparams->vtcm_اندازه);
        } وإلا {
            snprintf(str, max_اندازه, "----");
        }

    }

    فراغ format(ثابت htp_opnode & غوټه) {
        format_op_dims(dims, حجم(dims), غوټه);
        format_op_strides(strides, حجم(strides), غوټه);
        format_op_types(types, حجم(types), غوټه);
        format_op_buffs(buffs, حجم(buffs), غوټه);
        format_op_نومونه(نومونه, حجم(نومونه), غوټه);
        format_kernel_params(kparams, حجم(kparams), غوټه);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        نومونه[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ثابت htp_opnode & غوټه) { format(غوټه); }
};

#د_که_پای // HTP_OPNODE_H
