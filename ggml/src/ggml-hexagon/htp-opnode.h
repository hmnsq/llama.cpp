#جيڪڏهن_وصف_نه_ٿيل HTP_OPNODE_H
#وصف_ڪريو HTP_OPNODE_H

#وصف_ڪريو GGML_COMMON_IMPL_CPP
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
    ggml_tensor * ڳنڍ   { nullptr };
    htp_op_ڪوڊ   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::ويڪٽر<ggml_tensor *>                fused;
    std::ويڪٽر<std::shared_ptr<ggml_tensor>> dummy;

    std::ويڪٽر<ثابت ggml_tensor *> inputs;
    std::ويڪٽر<ثابت ggml_tensor *> outputs;
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

    فراغ init(ggml_tensor * ڳنڍ) {
        this->ڳنڍ = ڳنڍ;
        إذا (this->ڳنڍ) {
            إذا (this->ڳنڍ->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->ڳنڍ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ڌڪو_پٺ(this->ڳنڍ->src[1]);
            } وإلا {
                this->name = ggml_op_desc(this->ڳنڍ);

                // Build inputs (preserving optional nullptrs)
                صحيح n_inputs = n_سرگرم_src(this->ڳنڍ);
                this->inputs.resize(n_inputs, nullptr);
                لكل (صحيح i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->ڳنڍ->src[i];
                }
            }

            // Build outputs
            this->outputs.ڌڪو_پٺ(this->dst());
        }
    }

    htp_opnode(htp_op_ڪوڊ opcode = HTP_OP_INVALID, ggml_tensor * ڳنڍ = nullptr) : opcode(opcode) {
        init(ڳنڍ);
    }

    ggml_op             op()   ثابت { أرجع ڳنڍ->op; }
    ثابت ggml_tensor * src0() ثابت { أرجع ڳنڍ->src[0]; }
    ثابت ggml_tensor * src1() ثابت { أرجع ڳنڍ->src[1]; }
    ثابت ggml_tensor * dst()  ثابت { أرجع outputs.خالي_جايون() ? ڳنڍ : outputs.پٺ(); }

    ggml_tensor * جوڙيو_dummy(ثابت ggml_tensor & t) {
        dummy.ڌڪو_پٺ(std::make_shared<ggml_tensor>(t));
        أرجع dummy.پٺ().get();
    }

    فراغ جوڙيو_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ڌڪو_پٺ(t);

        name += "+";
        name += ggml_op_desc(t);

        إذا (extra_dst) {
            outputs.ڌڪو_پٺ(t);
        } وإلا {
            outputs.clear();
            outputs.ڌڪو_پٺ(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.پڇاڙي(), t), inputs.پڇاڙي());

        // Append new inputs from t, preserving middle nullptrs
        صحيح n_inputs = n_سرگرم_src(t);
        لكل (صحيح i = 0; i < n_inputs; i++) {
            ثابت خودڪار * src = t->src[i];
            إذا (!src) {
                inputs.ڌڪو_پٺ(nullptr);
            } وإلا إذا (src != ڳنڍ &&
                       std::ڳوليو(fused.begin(), fused.پڇاڙي(), src) == fused.پڇاڙي() &&
                       std::ڳوليو(inputs.begin(), inputs.پڇاڙي(), src) == inputs.پڇاڙي()) {
                inputs.ڌڪو_پٺ(src);
            }
        }
    }

    ثابت std::ويڪٽر<ثابت ggml_tensor *> & get_inputs() ثابت {
        أرجع inputs;
    }

    ثابت std::ويڪٽر<ثابت ggml_tensor *> & get_outputs() ثابت {
        أرجع outputs;
    }

    std::string op_name() ثابت {
        أرجع name;
    }

    bool is_خالي_جايون() ثابت {
        أرجع ggml_op_is_خالي_جايون(ڳنڍ->op);
    }

    bool stackable() ثابت {
        بدل (this->op()) {
            حالة GGML_OP_MUL_MAT:
                إذا (ggml_get_op_params_i32(this->ڳنڍ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    أرجع false;
                }
                أرجع ggml_is_quantized(this->src0()->type);
            حالة GGML_OP_MUL_MAT_ID:
                أرجع ggml_is_quantized(this->src0()->type);
            افتراضي:
                أرجع false;
        }
    }

    bool same_داخلا(ثابت htp_opnode& n) ثابت {
        أرجع n.src1() == this->src1();
    }
};

بنية htp_opformat {
    حرف strides[64 * GGML_MAX_SRC];
    حرف dims[64 * GGML_MAX_SRC];
    حرف types[16 * GGML_MAX_SRC];
    حرف buffs[64 * GGML_MAX_SRC];
    حرف نالا[64 * GGML_MAX_SRC];
    حرف kparams[128];

    صحيح format_tensor_dims(حرف * str, size_t max_سائيز, ثابت بنية ggml_tensor * t) {
        إذا (!t) {
            أرجع snprintf(str, max_سائيز, "NONE");
        }
        إذا (t->ne[2] == 1 && t->ne[3] == 1) {
            أرجع snprintf(str, max_سائيز, "%d:%d", (صحيح) t->ne[0], (صحيح) t->ne[1]);
        } وإلا {
            أرجع snprintf(str, max_سائيز, "%d:%d:%d:%d", (صحيح) t->ne[0], (صحيح) t->ne[1], (صحيح) t->ne[2], (صحيح) t->ne[3]);
        }
    }

    فراغ format_op_dims(حرف * str, size_t max_سائيز, ثابت htp_opnode & ڳنڍ) {
        حرف * p = str;
        حرف * p_پڇاڙي = str + max_سائيز;
        خودڪار inputs = ڳنڍ.get_inputs();

        إذا (!inputs.خالي_جايون()) {
            p += std::min((size_t)format_tensor_dims(p, p_پڇاڙي - p, inputs[0]), (size_t)(p_پڇاڙي - p));

            لكل (size_t i = 1; i < inputs.سائيز(); i++) {
                إذا (p < p_پڇاڙي) {
                    p += std::min((size_t)snprintf(p, p_پڇاڙي - p, " x "), (size_t)(p_پڇاڙي - p));
                }
                إذا (p < p_پڇاڙي) {
                    p += std::min((size_t)format_tensor_dims(p, p_پڇاڙي - p, inputs[i]), (size_t)(p_پڇاڙي - p));
                }
            }

            إذا (p < p_پڇاڙي) {
                p += std::min((size_t)snprintf(p, p_پڇاڙي - p, " -> "), (size_t)(p_پڇاڙي - p));
            }
        }

        حرف self[64];
        format_tensor_dims(self, حجم(self), ڳنڍ.dst());
        إذا (p < p_پڇاڙي) {
            p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", self), (size_t)(p_پڇاڙي - p));
        }
    }

    صحيح format_tensor_strides(حرف * str, size_t max_سائيز, ثابت بنية ggml_tensor * t) {
        إذا (!t) {
            أرجع snprintf(str, max_سائيز, "NONE");
        }
        ثابت حرف * c = ggml_is_contiguous(t) ? "" : "!";

        إذا (t->ne[2] == 1 && t->ne[3] == 1) {
            أرجع snprintf(str, max_سائيز, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } وإلا {
            أرجع snprintf(str, max_سائيز, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    فراغ format_op_strides(حرف * str, size_t max_سائيز, ثابت htp_opnode & ڳنڍ) {
        حرف * p = str;
        حرف * p_پڇاڙي = str + max_سائيز;
        خودڪار inputs = ڳنڍ.get_inputs();

        إذا (!inputs.خالي_جايون()) {
            p += std::min((size_t)format_tensor_strides(p, p_پڇاڙي - p, inputs[0]), (size_t)(p_پڇاڙي - p));

            لكل (size_t i = 1; i < inputs.سائيز(); i++) {
                إذا (p < p_پڇاڙي) {
                    p += std::min((size_t)snprintf(p, p_پڇاڙي - p, " x "), (size_t)(p_پڇاڙي - p));
                }
                إذا (p < p_پڇاڙي) {
                    p += std::min((size_t)format_tensor_strides(p, p_پڇاڙي - p, inputs[i]), (size_t)(p_پڇاڙي - p));
                }
            }

            إذا (p < p_پڇاڙي) {
                p += std::min((size_t)snprintf(p, p_پڇاڙي - p, " -> "), (size_t)(p_پڇاڙي - p));
            }
        }

        حرف self[64];
        format_tensor_strides(self, حجم(self), ڳنڍ.dst());
        إذا (p < p_پڇاڙي) {
            p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", self), (size_t)(p_پڇاڙي - p));
        }
    }

    فراغ format_op_types(حرف * str, size_t max_سائيز, ثابت htp_opnode & ڳنڍ) {
        حرف * p = str;
        حرف * p_پڇاڙي = str + max_سائيز;
        خودڪار inputs = ڳنڍ.get_inputs();

        إذا (!inputs.خالي_جايون()) {
            إذا (p < p_پڇاڙي) {
                p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_پڇاڙي - p));
            }

            لكل (size_t i = 1; i < inputs.سائيز(); i++) {
                إذا (p < p_پڇاڙي) {
                    p += std::min((size_t)snprintf(p, p_پڇاڙي - p, " x "), (size_t)(p_پڇاڙي - p));
                }
                إذا (p < p_پڇاڙي) {
                    p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_پڇاڙي - p));
                }
            }

            إذا (p < p_پڇاڙي) {
                p += std::min((size_t)snprintf(p, p_پڇاڙي - p, " -> "), (size_t)(p_پڇاڙي - p));
            }
        }

        إذا (p < p_پڇاڙي) {
            p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", ggml_type_name(ڳنڍ.dst()->type)), (size_t)(p_پڇاڙي - p));
        }
    }

    ثابت حرف * tensor_buff_name(ثابت بنية ggml_tensor * t) {
        إذا (t && t->بفر) {
            أرجع ggml_backend_بفر_name(t->بفر);
        }
        أرجع "NONE";
    }

    فراغ format_op_buffs(حرف * str, size_t max_سائيز, ثابت htp_opnode & ڳنڍ) {
        حرف * p = str;
        حرف * p_پڇاڙي = str + max_سائيز;
        خودڪار inputs = ڳنڍ.get_inputs();

        إذا (!inputs.خالي_جايون()) {
            إذا (p < p_پڇاڙي) {
                p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_پڇاڙي - p));
            }

            لكل (size_t i = 1; i < inputs.سائيز(); i++) {
                إذا (p < p_پڇاڙي) {
                    p += std::min((size_t)snprintf(p, p_پڇاڙي - p, " x "), (size_t)(p_پڇاڙي - p));
                }
                إذا (p < p_پڇاڙي) {
                    p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_پڇاڙي - p));
                }
            }

            إذا (p < p_پڇاڙي) {
                p += std::min((size_t)snprintf(p, p_پڇاڙي - p, " -> "), (size_t)(p_پڇاڙي - p));
            }
        }

        إذا (p < p_پڇاڙي) {
            p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", tensor_buff_name(ڳنڍ.dst())), (size_t)(p_پڇاڙي - p));
        }
    }

    فراغ format_op_نالا(حرف * str, size_t max_سائيز, ثابت htp_opnode & ڳنڍ) {
        حرف * p = str;
        حرف * p_پڇاڙي = str + max_سائيز;
        خودڪار inputs = ڳنڍ.get_inputs();

        إذا (!inputs.خالي_جايون()) {
            إذا (p < p_پڇاڙي) {
                p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_پڇاڙي - p));
            }

            لكل (size_t i = 1; i < inputs.سائيز(); i++) {
                إذا (p < p_پڇاڙي) {
                    p += std::min((size_t)snprintf(p, p_پڇاڙي - p, " x "), (size_t)(p_پڇاڙي - p));
                }
                إذا (p < p_پڇاڙي) {
                    p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_پڇاڙي - p));
                }
            }

            إذا (p < p_پڇاڙي) {
                p += std::min((size_t)snprintf(p, p_پڇاڙي - p, " -> "), (size_t)(p_پڇاڙي - p));
            }
        }

        إذا (p < p_پڇاڙي) {
            p += std::min((size_t)snprintf(p, p_پڇاڙي - p, "%s", ڳنڍ.dst()->name), (size_t)(p_پڇاڙي - p));
        }
    }
    فراغ format_kernel_params(حرف * str, size_t max_سائيز, ثابت htp_opnode & ڳنڍ) {
        إذا (ڳنڍ.opcode == HTP_OP_MUL_MAT || ڳنڍ.opcode == HTP_OP_MUL_MAT_ID ||
            ڳنڍ.opcode == HTP_OP_MUL_MAT_NX || ڳنڍ.opcode == HTP_OP_MUL_MAT_ID_NX ||
            ڳنڍ.opcode == HTP_OP_MUL_MAT_جوڙيو) {
            ثابت خودڪار * kparams = (ثابت بنية htp_mm_kernel_params *) ڳنڍ.kernel_params;
            ثابت حرف * رستو = "unknown";
            int32_t type = kparams->kernel_type;
            إذا (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                رستو = (kparams->tile_سائيز > 0) ? "hmx-tiled" : "hmx-flat";
            } وإلا إذا (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                رستو = "hvx-flat";
            } وإلا إذا (type == HTP_MM_KERNEL_HVX_QUANT_صف || type == HTP_MM_KERNEL_HVX_QUANT_بلاڪ) {
                رستو = "hvx-tiled";
            }
            snprintf(str, max_سائيز, "%s vtcm %d", رستو, (صحيح) kparams->vtcm_سائيز);
        } وإلا إذا (ڳنڍ.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ثابت خودڪار * kparams = (ثابت بنية htp_fa_kernel_params *) ڳنڍ.kernel_params;
            ثابت حرف * رستو = "unknown";
            int32_t type = kparams->kernel_type;
            إذا (type == HTP_FA_KERNEL_HMX) {
                رستو = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } وإلا إذا (type == HTP_FA_KERNEL_HVX) {
                رستو = "hvx";
            }
            snprintf(str, max_سائيز, "%s vtcm %d", رستو, (صحيح) kparams->vtcm_سائيز);
        } وإلا إذا (htp_op_is_unary(ڳنڍ.opcode)) {
            ثابت خودڪار * kparams = (ثابت بنية htp_unary_kernel_params *) ڳنڍ.kernel_params;
            snprintf(str, max_سائيز, "%s vtcm %d", kparams->کانو_tile ? "wide-row" : "row-block", (صحيح) kparams->vtcm_سائيز);
        } وإلا إذا (ڳنڍ.opcode == HTP_OP_MDEV_گروهه && ڳنڍ.ڳنڍ) {
            snprintf(str, max_سائيز, "idx %d count %d", (صحيح) ڳنڍ.ڳنڍ->op_params[0], (صحيح) ڳنڍ.dst()->ne[1]);
        } وإلا إذا ((ڳنڍ.opcode == HTP_OP_FENCE || ڳنڍ.opcode == HTP_OP_CPY_FENCE) && ڳنڍ.ڳنڍ) {
            snprintf(str, max_سائيز, "seq 0x%x", (uint32_t) ڳنڍ.ڳنڍ->op_params[0]);
        } وإلا إذا (ڳنڍ.opcode == HTP_OP_ALLREDUCE && ڳنڍ.ڳنڍ) {
            snprintf(str, max_سائيز, "seq 0x%x -> 0x%x", (uint32_t) ڳنڍ.ڳنڍ->op_params[0], (uint32_t) ڳنڍ.ڳنڍ->op_params[1]);
        } وإلا إذا (ڳنڍ.opcode == HTP_OP_SSM_CONV) {
            ثابت خودڪار * kparams = (ثابت بنية htp_ssm_conv_kernel_params *) ڳنڍ.kernel_params;
            snprintf(str, max_سائيز, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (صحيح) kparams->vtcm_سائيز);
        } وإلا إذا (ڳنڍ.opcode == HTP_OP_SOFTMAX) {
            ثابت خودڪار * kparams = (ثابت بنية htp_softmax_kernel_params *) ڳنڍ.kernel_params;
            snprintf(str, max_سائيز, "k%d nth %d vtcm %d", (صحيح) kparams->kernel_id, (صحيح) kparams->n_ڌاڳا, (صحيح) kparams->vtcm_سائيز);
        } وإلا إذا (ڳنڍ.opcode == HTP_OP_GATED_DELTA_NET) {
            ثابت خودڪار * kparams = (ثابت بنية htp_gdn_kernel_params *) ڳنڍ.kernel_params;
            ثابت حرف * رستو = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_سائيز, "%s-%s vtcm %u", رستو, kparams->kda ? "kda" : "scalar",
                     (بنانشان صحيح) (kparams->vtcm_سائيز ? kparams->vtcm_سائيز : kparams->vtcm_في_thread * kparams->n_ڌاڳا));
        } وإلا إذا (ڳنڍ.opcode == HTP_OP_MUL || ڳنڍ.opcode == HTP_OP_جوڙيو || ڳنڍ.opcode == HTP_OP_جوڙيو_ID ||
                   ڳنڍ.opcode == HTP_OP_SUB || ڳنڍ.opcode == HTP_OP_DIV) {
            ثابت خودڪار * kparams = (ثابت بنية htp_ٻٽو_kernel_params *) ڳنڍ.kernel_params;
            snprintf(str, max_سائيز, "vtcm %u", (بنانشان صحيح) kparams->vtcm_سائيز);
        } وإلا إذا (ڳنڍ.opcode == HTP_OP_ARGSORT || ڳنڍ.opcode == HTP_OP_چوٽي_K) {
            ثابت خودڪار * kparams = (ثابت بنية htp_ترتيب_ڏيو_kernel_params *) ڳنڍ.kernel_params;
            snprintf(str, max_سائيز, "%s nth %d nchk %d chk %d vtcm %d",
                     ڳنڍ.opcode == HTP_OP_چوٽي_K ? "top_k" : "argsort",
                     (صحيح) kparams->n_ڌاڳا, (صحيح) kparams->n_chunks,
                     (صحيح) kparams->chunk_elems, (صحيح) kparams->vtcm_سائيز);
        } وإلا إذا (ڳنڍ.opcode == HTP_OP_GET_صفون) {
            ثابت خودڪار * kparams = (ثابت بنية htp_get_صفون_kernel_params *) ڳنڍ.kernel_params;
            ثابت حرف * ktype_str = "unknown";
            بدل (kparams->kernel_type) {
                حالة HTP_GET_صفون_KERNEL_SAMETYPE: ktype_str = "sametype"; اكسر;
                حالة HTP_GET_صفون_KERNEL_TILED:    ktype_str = "tiled"; اكسر;
                حالة HTP_GET_صفون_KERNEL_FLAT:     ktype_str = "flat"; اكسر;
            }
            snprintf(str, max_سائيز, "%s%s vtcm %u", ktype_str, kparams->n_ڌاڳا > 1 ? "-multi" : "", (بنانشان صحيح) kparams->vtcm_سائيز);
        } وإلا {
            snprintf(str, max_سائيز, "----");
        }

    }

    فراغ format(ثابت htp_opnode & ڳنڍ) {
        format_op_dims(dims, حجم(dims), ڳنڍ);
        format_op_strides(strides, حجم(strides), ڳنڍ);
        format_op_types(types, حجم(types), ڳنڍ);
        format_op_buffs(buffs, حجم(buffs), ڳنڍ);
        format_op_نالا(نالا, حجم(نالا), ڳنڍ);
        format_kernel_params(kparams, حجم(kparams), ڳنڍ);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        نالا[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ثابت htp_opnode & ڳنڍ) { format(ڳنڍ); }
};

#جيڪڏهن_پورو // HTP_OPNODE_H
