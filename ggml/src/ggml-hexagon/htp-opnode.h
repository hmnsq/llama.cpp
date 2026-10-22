#جے_بنا_تعریف HTP_OPNODE_H
#تعریف_کرو HTP_OPNODE_H

#تعریف_کرو GGML_COMMON_IMPL_CPP
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
    ggml_tensor * گرہ   { nullptr };
    htp_op_کوڈ   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::سمتیہ<ggml_tensor *>                fused;
    std::سمتیہ<std::shared_ptr<ggml_tensor>> dummy;

    std::سمتیہ<ثابت ggml_tensor *> inputs;
    std::سمتیہ<ثابت ggml_tensor *> outputs;
    std::string                      name;

    صحيح n_سرگرم_src(ثابت ggml_tensor * t) ثابت {
        إذا (!t) أرجع 0;
        لكل (صحيح i = GGML_MAX_SRC - 1; i >= 0; i--) {
            إذا (t->src[i]) {
                أرجع i + 1;
            }
        }
        أرجع 0;
    }

    فراغ init(ggml_tensor * گرہ) {
        this->گرہ = گرہ;
        إذا (this->گرہ) {
            إذا (this->گرہ->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->گرہ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ڈالو_پٹھ(this->گرہ->src[1]);
            } وإلا {
                this->name = ggml_op_desc(this->گرہ);

                // Build inputs (preserving optional nullptrs)
                صحيح n_inputs = n_سرگرم_src(this->گرہ);
                this->inputs.resize(n_inputs, nullptr);
                لكل (صحيح i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->گرہ->src[i];
                }
            }

            // Build outputs
            this->outputs.ڈالو_پٹھ(this->dst());
        }
    }

    htp_opnode(htp_op_کوڈ opcode = HTP_OP_INVALID, ggml_tensor * گرہ = nullptr) : opcode(opcode) {
        init(گرہ);
    }

    ggml_op             op()   ثابت { أرجع گرہ->op; }
    ثابت ggml_tensor * src0() ثابت { أرجع گرہ->src[0]; }
    ثابت ggml_tensor * src1() ثابت { أرجع گرہ->src[1]; }
    ثابت ggml_tensor * dst()  ثابت { أرجع outputs.خالی_تھاواں() ? گرہ : outputs.پٹھ(); }

    ggml_tensor * جوڑو_dummy(ثابت ggml_tensor & t) {
        dummy.ڈالو_پٹھ(std::make_shared<ggml_tensor>(t));
        أرجع dummy.پٹھ().get();
    }

    فراغ جوڑو_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ڈالو_پٹھ(t);

        name += "+";
        name += ggml_op_desc(t);

        إذا (extra_dst) {
            outputs.ڈالو_پٹھ(t);
        } وإلا {
            outputs.clear();
            outputs.ڈالو_پٹھ(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.سرا(), t), inputs.سرا());

        // Append new inputs from t, preserving middle nullptrs
        صحيح n_inputs = n_سرگرم_src(t);
        لكل (صحيح i = 0; i < n_inputs; i++) {
            ثابت خودکار * src = t->src[i];
            إذا (!src) {
                inputs.ڈالو_پٹھ(nullptr);
            } وإلا إذا (src != گرہ &&
                       std::لبھو(fused.begin(), fused.سرا(), src) == fused.سرا() &&
                       std::لبھو(inputs.begin(), inputs.سرا(), src) == inputs.سرا()) {
                inputs.ڈالو_پٹھ(src);
            }
        }
    }

    ثابت std::سمتیہ<ثابت ggml_tensor *> & get_inputs() ثابت {
        أرجع inputs;
    }

    ثابت std::سمتیہ<ثابت ggml_tensor *> & get_outputs() ثابت {
        أرجع outputs;
    }

    std::string op_name() ثابت {
        أرجع name;
    }

    bool is_خالی_تھاواں() ثابت {
        أرجع ggml_op_is_خالی_تھاواں(گرہ->op);
    }

    bool stackable() ثابت {
        بدل (this->op()) {
            حالة GGML_OP_MUL_MAT:
                إذا (ggml_get_op_params_i32(this->گرہ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    أرجع false;
                }
                أرجع ggml_is_quantized(this->src0()->type);
            حالة GGML_OP_MUL_MAT_ID:
                أرجع ggml_is_quantized(this->src0()->type);
            افتراضي:
                أرجع false;
        }
    }

    bool same_داخلہ(ثابت htp_opnode& n) ثابت {
        أرجع n.src1() == this->src1();
    }
};

بنية htp_opformat {
    حرف strides[64 * GGML_MAX_SRC];
    حرف dims[64 * GGML_MAX_SRC];
    حرف types[16 * GGML_MAX_SRC];
    حرف buffs[64 * GGML_MAX_SRC];
    حرف ناں[64 * GGML_MAX_SRC];
    حرف kparams[128];

    صحيح format_tensor_dims(حرف * str, size_t max_ناپ, ثابت بنية ggml_tensor * t) {
        إذا (!t) {
            أرجع snprintf(str, max_ناپ, "NONE");
        }
        إذا (t->ne[2] == 1 && t->ne[3] == 1) {
            أرجع snprintf(str, max_ناپ, "%d:%d", (صحيح) t->ne[0], (صحيح) t->ne[1]);
        } وإلا {
            أرجع snprintf(str, max_ناپ, "%d:%d:%d:%d", (صحيح) t->ne[0], (صحيح) t->ne[1], (صحيح) t->ne[2], (صحيح) t->ne[3]);
        }
    }

    فراغ format_op_dims(حرف * str, size_t max_ناپ, ثابت htp_opnode & گرہ) {
        حرف * p = str;
        حرف * p_سرا = str + max_ناپ;
        خودکار inputs = گرہ.get_inputs();

        إذا (!inputs.خالی_تھاواں()) {
            p += std::min((size_t)format_tensor_dims(p, p_سرا - p, inputs[0]), (size_t)(p_سرا - p));

            لكل (size_t i = 1; i < inputs.ناپ(); i++) {
                إذا (p < p_سرا) {
                    p += std::min((size_t)snprintf(p, p_سرا - p, " x "), (size_t)(p_سرا - p));
                }
                إذا (p < p_سرا) {
                    p += std::min((size_t)format_tensor_dims(p, p_سرا - p, inputs[i]), (size_t)(p_سرا - p));
                }
            }

            إذا (p < p_سرا) {
                p += std::min((size_t)snprintf(p, p_سرا - p, " -> "), (size_t)(p_سرا - p));
            }
        }

        حرف self[64];
        format_tensor_dims(self, حجم(self), گرہ.dst());
        إذا (p < p_سرا) {
            p += std::min((size_t)snprintf(p, p_سرا - p, "%s", self), (size_t)(p_سرا - p));
        }
    }

    صحيح format_tensor_strides(حرف * str, size_t max_ناپ, ثابت بنية ggml_tensor * t) {
        إذا (!t) {
            أرجع snprintf(str, max_ناپ, "NONE");
        }
        ثابت حرف * c = ggml_is_contiguous(t) ? "" : "!";

        إذا (t->ne[2] == 1 && t->ne[3] == 1) {
            أرجع snprintf(str, max_ناپ, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } وإلا {
            أرجع snprintf(str, max_ناپ, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    فراغ format_op_strides(حرف * str, size_t max_ناپ, ثابت htp_opnode & گرہ) {
        حرف * p = str;
        حرف * p_سرا = str + max_ناپ;
        خودکار inputs = گرہ.get_inputs();

        إذا (!inputs.خالی_تھاواں()) {
            p += std::min((size_t)format_tensor_strides(p, p_سرا - p, inputs[0]), (size_t)(p_سرا - p));

            لكل (size_t i = 1; i < inputs.ناپ(); i++) {
                إذا (p < p_سرا) {
                    p += std::min((size_t)snprintf(p, p_سرا - p, " x "), (size_t)(p_سرا - p));
                }
                إذا (p < p_سرا) {
                    p += std::min((size_t)format_tensor_strides(p, p_سرا - p, inputs[i]), (size_t)(p_سرا - p));
                }
            }

            إذا (p < p_سرا) {
                p += std::min((size_t)snprintf(p, p_سرا - p, " -> "), (size_t)(p_سرا - p));
            }
        }

        حرف self[64];
        format_tensor_strides(self, حجم(self), گرہ.dst());
        إذا (p < p_سرا) {
            p += std::min((size_t)snprintf(p, p_سرا - p, "%s", self), (size_t)(p_سرا - p));
        }
    }

    فراغ format_op_types(حرف * str, size_t max_ناپ, ثابت htp_opnode & گرہ) {
        حرف * p = str;
        حرف * p_سرا = str + max_ناپ;
        خودکار inputs = گرہ.get_inputs();

        إذا (!inputs.خالی_تھاواں()) {
            إذا (p < p_سرا) {
                p += std::min((size_t)snprintf(p, p_سرا - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_سرا - p));
            }

            لكل (size_t i = 1; i < inputs.ناپ(); i++) {
                إذا (p < p_سرا) {
                    p += std::min((size_t)snprintf(p, p_سرا - p, " x "), (size_t)(p_سرا - p));
                }
                إذا (p < p_سرا) {
                    p += std::min((size_t)snprintf(p, p_سرا - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_سرا - p));
                }
            }

            إذا (p < p_سرا) {
                p += std::min((size_t)snprintf(p, p_سرا - p, " -> "), (size_t)(p_سرا - p));
            }
        }

        إذا (p < p_سرا) {
            p += std::min((size_t)snprintf(p, p_سرا - p, "%s", ggml_type_name(گرہ.dst()->type)), (size_t)(p_سرا - p));
        }
    }

    ثابت حرف * tensor_buff_name(ثابت بنية ggml_tensor * t) {
        إذا (t && t->بفر) {
            أرجع ggml_backend_بفر_name(t->بفر);
        }
        أرجع "NONE";
    }

    فراغ format_op_buffs(حرف * str, size_t max_ناپ, ثابت htp_opnode & گرہ) {
        حرف * p = str;
        حرف * p_سرا = str + max_ناپ;
        خودکار inputs = گرہ.get_inputs();

        إذا (!inputs.خالی_تھاواں()) {
            إذا (p < p_سرا) {
                p += std::min((size_t)snprintf(p, p_سرا - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_سرا - p));
            }

            لكل (size_t i = 1; i < inputs.ناپ(); i++) {
                إذا (p < p_سرا) {
                    p += std::min((size_t)snprintf(p, p_سرا - p, " x "), (size_t)(p_سرا - p));
                }
                إذا (p < p_سرا) {
                    p += std::min((size_t)snprintf(p, p_سرا - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_سرا - p));
                }
            }

            إذا (p < p_سرا) {
                p += std::min((size_t)snprintf(p, p_سرا - p, " -> "), (size_t)(p_سرا - p));
            }
        }

        إذا (p < p_سرا) {
            p += std::min((size_t)snprintf(p, p_سرا - p, "%s", tensor_buff_name(گرہ.dst())), (size_t)(p_سرا - p));
        }
    }

    فراغ format_op_ناں(حرف * str, size_t max_ناپ, ثابت htp_opnode & گرہ) {
        حرف * p = str;
        حرف * p_سرا = str + max_ناپ;
        خودکار inputs = گرہ.get_inputs();

        إذا (!inputs.خالی_تھاواں()) {
            إذا (p < p_سرا) {
                p += std::min((size_t)snprintf(p, p_سرا - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_سرا - p));
            }

            لكل (size_t i = 1; i < inputs.ناپ(); i++) {
                إذا (p < p_سرا) {
                    p += std::min((size_t)snprintf(p, p_سرا - p, " x "), (size_t)(p_سرا - p));
                }
                إذا (p < p_سرا) {
                    p += std::min((size_t)snprintf(p, p_سرا - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_سرا - p));
                }
            }

            إذا (p < p_سرا) {
                p += std::min((size_t)snprintf(p, p_سرا - p, " -> "), (size_t)(p_سرا - p));
            }
        }

        إذا (p < p_سرا) {
            p += std::min((size_t)snprintf(p, p_سرا - p, "%s", گرہ.dst()->name), (size_t)(p_سرا - p));
        }
    }
    فراغ format_kernel_params(حرف * str, size_t max_ناپ, ثابت htp_opnode & گرہ) {
        إذا (گرہ.opcode == HTP_OP_MUL_MAT || گرہ.opcode == HTP_OP_MUL_MAT_ID ||
            گرہ.opcode == HTP_OP_MUL_MAT_NX || گرہ.opcode == HTP_OP_MUL_MAT_ID_NX ||
            گرہ.opcode == HTP_OP_MUL_MAT_جوڑو) {
            ثابت خودکار * kparams = (ثابت بنية htp_mm_kernel_params *) گرہ.kernel_params;
            ثابت حرف * رستہ = "unknown";
            int32_t type = kparams->kernel_type;
            إذا (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                رستہ = (kparams->tile_ناپ > 0) ? "hmx-tiled" : "hmx-flat";
            } وإلا إذا (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                رستہ = "hvx-flat";
            } وإلا إذا (type == HTP_MM_KERNEL_HVX_QUANT_صف || type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
                رستہ = "hvx-tiled";
            }
            snprintf(str, max_ناپ, "%s vtcm %d", رستہ, (صحيح) kparams->vtcm_ناپ);
        } وإلا إذا (گرہ.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ثابت خودکار * kparams = (ثابت بنية htp_fa_kernel_params *) گرہ.kernel_params;
            ثابت حرف * رستہ = "unknown";
            int32_t type = kparams->kernel_type;
            إذا (type == HTP_FA_KERNEL_HMX) {
                رستہ = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } وإلا إذا (type == HTP_FA_KERNEL_HVX) {
                رستہ = "hvx";
            }
            snprintf(str, max_ناپ, "%s vtcm %d", رستہ, (صحيح) kparams->vtcm_ناپ);
        } وإلا إذا (htp_op_is_unary(گرہ.opcode)) {
            ثابت خودکار * kparams = (ثابت بنية htp_unary_kernel_params *) گرہ.kernel_params;
            snprintf(str, max_ناپ, "%s vtcm %d", kparams->خانہ_tile ? "wide-row" : "row-block", (صحيح) kparams->vtcm_ناپ);
        } وإلا إذا (گرہ.opcode == HTP_OP_MDEV_گروہ && گرہ.گرہ) {
            snprintf(str, max_ناپ, "idx %d count %d", (صحيح) گرہ.گرہ->op_params[0], (صحيح) گرہ.dst()->ne[1]);
        } وإلا إذا ((گرہ.opcode == HTP_OP_FENCE || گرہ.opcode == HTP_OP_CPY_FENCE) && گرہ.گرہ) {
            snprintf(str, max_ناپ, "seq 0x%x", (uint32_t) گرہ.گرہ->op_params[0]);
        } وإلا إذا (گرہ.opcode == HTP_OP_ALLREDUCE && گرہ.گرہ) {
            snprintf(str, max_ناپ, "seq 0x%x -> 0x%x", (uint32_t) گرہ.گرہ->op_params[0], (uint32_t) گرہ.گرہ->op_params[1]);
        } وإلا إذا (گرہ.opcode == HTP_OP_SSM_CONV) {
            ثابت خودکار * kparams = (ثابت بنية htp_ssm_conv_kernel_params *) گرہ.kernel_params;
            snprintf(str, max_ناپ, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (صحيح) kparams->vtcm_ناپ);
        } وإلا إذا (گرہ.opcode == HTP_OP_SOFTMAX) {
            ثابت خودکار * kparams = (ثابت بنية htp_softmax_kernel_params *) گرہ.kernel_params;
            snprintf(str, max_ناپ, "k%d nth %d vtcm %d", (صحيح) kparams->kernel_id, (صحيح) kparams->n_دھاگے, (صحيح) kparams->vtcm_ناپ);
        } وإلا إذا (گرہ.opcode == HTP_OP_GATED_DELTA_NET) {
            ثابت خودکار * kparams = (ثابت بنية htp_gdn_kernel_params *) گرہ.kernel_params;
            ثابت حرف * رستہ = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_ناپ, "%s-%s vtcm %u", رستہ, kparams->kda ? "kda" : "scalar",
                     (بنانشانی صحيح) (kparams->vtcm_ناپ ? kparams->vtcm_ناپ : kparams->vtcm_فی_thread * kparams->n_دھاگے));
        } وإلا إذا (گرہ.opcode == HTP_OP_MUL || گرہ.opcode == HTP_OP_جوڑو || گرہ.opcode == HTP_OP_جوڑو_ID ||
                   گرہ.opcode == HTP_OP_SUB || گرہ.opcode == HTP_OP_DIV) {
            ثابت خودکار * kparams = (ثابت بنية htp_ثنائی_kernel_params *) گرہ.kernel_params;
            snprintf(str, max_ناپ, "vtcm %u", (بنانشانی صحيح) kparams->vtcm_ناپ);
        } وإلا إذا (گرہ.opcode == HTP_OP_ARGSORT || گرہ.opcode == HTP_OP_چوٹی_K) {
            ثابت خودکار * kparams = (ثابت بنية htp_ترتیب_دیو_kernel_params *) گرہ.kernel_params;
            snprintf(str, max_ناپ, "%s nth %d nchk %d chk %d vtcm %d",
                     گرہ.opcode == HTP_OP_چوٹی_K ? "top_k" : "argsort",
                     (صحيح) kparams->n_دھاگے, (صحيح) kparams->n_chunks,
                     (صحيح) kparams->chunk_elems, (صحيح) kparams->vtcm_ناپ);
        } وإلا إذا (گرہ.opcode == HTP_OP_GET_صفاں) {
            ثابت خودکار * kparams = (ثابت بنية htp_get_صفاں_kernel_params *) گرہ.kernel_params;
            ثابت حرف * ktype_str = "unknown";
            بدل (kparams->kernel_type) {
                حالة HTP_GET_صفاں_KERNEL_SAMETYPE: ktype_str = "sametype"; اكسر;
                حالة HTP_GET_صفاں_KERNEL_TILED:    ktype_str = "tiled"; اكسر;
                حالة HTP_GET_صفاں_KERNEL_FLAT:     ktype_str = "flat"; اكسر;
            }
            snprintf(str, max_ناپ, "%s%s vtcm %u", ktype_str, kparams->n_دھاگے > 1 ? "-multi" : "", (بنانشانی صحيح) kparams->vtcm_ناپ);
        } وإلا {
            snprintf(str, max_ناپ, "----");
        }

    }

    فراغ format(ثابت htp_opnode & گرہ) {
        format_op_dims(dims, حجم(dims), گرہ);
        format_op_strides(strides, حجم(strides), گرہ);
        format_op_types(types, حجم(types), گرہ);
        format_op_buffs(buffs, حجم(buffs), گرہ);
        format_op_ناں(ناں, حجم(ناں), گرہ);
        format_kernel_params(kparams, حجم(kparams), گرہ);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        ناں[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ثابت htp_opnode & گرہ) { format(گرہ); }
};

#جے_انت // HTP_OPNODE_H
