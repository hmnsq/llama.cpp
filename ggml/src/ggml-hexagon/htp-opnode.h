#اگر_غیر_تعریف_شدہ HTP_OPNODE_H
#تعریف_کرو HTP_OPNODE_H

#تعریف_کرو GGML_COMMON_IMPL_CPP
#شامل "ggml-backend-impl.h"
#شامل "ggml-common.h"

#شامل <algorithm>
#شامل <string>
#شامل <vector>
#شامل <memory>
#شامل <stdio.h>
#شامل "htp-ops.h"
#شامل "htp/matmul-ops.h"
#شامل "htp/flash-attn-ops.h"
#شامل "htp/unary-ops.h"
#شامل "htp/binary-ops.h"
#شامل "htp/allreduce-ops.h"
#شامل "htp/ssm-conv.h"
#شامل "htp/gated-delta-net-ops.h"
#شامل "htp/softmax-ops.h"
#شامل "htp/argsort-ops.h"
#شامل "htp/get-rows-ops.h"

ساخت htp_opnode {
    ggml_tensor * گرہ   { nullptr };
    htp_op_رمز   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::سمتیہ<ggml_tensor *>                fused;
    std::سمتیہ<std::shared_ptr<ggml_tensor>> dummy;

    std::سمتیہ<ثابت ggml_tensor *> inputs;
    std::سمتیہ<ثابت ggml_tensor *> outputs;
    std::string                      name;

    صحیح n_فعال_src(ثابت ggml_tensor * t) ثابت {
        اگر (!t) واپس 0;
        برائے (صحیح i = GGML_MAX_SRC - 1; i >= 0; i--) {
            اگر (t->src[i]) {
                واپس i + 1;
            }
        }
        واپس 0;
    }

    خالی init(ggml_tensor * گرہ) {
        this->گرہ = گرہ;
        اگر (this->گرہ) {
            اگر (this->گرہ->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->گرہ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ڈالو_عقب(this->گرہ->src[1]);
            } ورنہ {
                this->name = ggml_op_desc(this->گرہ);

                // Build inputs (preserving optional nullptrs)
                صحیح n_inputs = n_فعال_src(this->گرہ);
                this->inputs.resize(n_inputs, nullptr);
                برائے (صحیح i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->گرہ->src[i];
                }
            }

            // Build outputs
            this->outputs.ڈالو_عقب(this->dst());
        }
    }

    htp_opnode(htp_op_رمز opcode = HTP_OP_INVALID, ggml_tensor * گرہ = nullptr) : opcode(opcode) {
        init(گرہ);
    }

    ggml_op             op()   ثابت { واپس گرہ->op; }
    ثابت ggml_tensor * src0() ثابت { واپس گرہ->src[0]; }
    ثابت ggml_tensor * src1() ثابت { واپس گرہ->src[1]; }
    ثابت ggml_tensor * dst()  ثابت { واپس outputs.خالی_خانے() ? گرہ : outputs.عقب(); }

    ggml_tensor * جوڑو_dummy(ثابت ggml_tensor & t) {
        dummy.ڈالو_عقب(std::make_shared<ggml_tensor>(t));
        واپس dummy.عقب().get();
    }

    خالی جوڑو_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ڈالو_عقب(t);

        name += "+";
        name += ggml_op_desc(t);

        اگر (extra_dst) {
            outputs.ڈالو_عقب(t);
        } ورنہ {
            outputs.clear();
            outputs.ڈالو_عقب(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.انتہا(), t), inputs.انتہا());

        // Append new inputs from t, preserving middle nullptrs
        صحیح n_inputs = n_فعال_src(t);
        برائے (صحیح i = 0; i < n_inputs; i++) {
            ثابت خودکار * src = t->src[i];
            اگر (!src) {
                inputs.ڈالو_عقب(nullptr);
            } ورنہ اگر (src != گرہ &&
                       std::ڈھونڈو(fused.begin(), fused.انتہا(), src) == fused.انتہا() &&
                       std::ڈھونڈو(inputs.begin(), inputs.انتہا(), src) == inputs.انتہا()) {
                inputs.ڈالو_عقب(src);
            }
        }
    }

    ثابت std::سمتیہ<ثابت ggml_tensor *> & get_inputs() ثابت {
        واپس inputs;
    }

    ثابت std::سمتیہ<ثابت ggml_tensor *> & get_outputs() ثابت {
        واپس outputs;
    }

    std::string op_name() ثابت {
        واپس name;
    }

    bool is_خالی_خانے() ثابت {
        واپس ggml_op_is_خالی_خانے(گرہ->op);
    }

    bool stackable() ثابت {
        انتخاب (this->op()) {
            حالت GGML_OP_MUL_MAT:
                اگر (ggml_get_op_params_i32(this->گرہ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    واپس false;
                }
                واپس ggml_is_quantized(this->src0()->type);
            حالت GGML_OP_MUL_MAT_ID:
                واپس ggml_is_quantized(this->src0()->type);
            طےشدہ:
                واپس false;
        }
    }

    bool same_داخلہ(ثابت htp_opnode& n) ثابت {
        واپس n.src1() == this->src1();
    }
};

ساخت htp_opformat {
    حرف strides[64 * GGML_MAX_SRC];
    حرف dims[64 * GGML_MAX_SRC];
    حرف types[16 * GGML_MAX_SRC];
    حرف buffs[64 * GGML_MAX_SRC];
    حرف نام[64 * GGML_MAX_SRC];
    حرف kparams[128];

    صحیح format_tensor_dims(حرف * str, size_t max_سائز, ثابت ساخت ggml_tensor * t) {
        اگر (!t) {
            واپس snprintf(str, max_سائز, "NONE");
        }
        اگر (t->ne[2] == 1 && t->ne[3] == 1) {
            واپس snprintf(str, max_سائز, "%d:%d", (صحیح) t->ne[0], (صحیح) t->ne[1]);
        } ورنہ {
            واپس snprintf(str, max_سائز, "%d:%d:%d:%d", (صحیح) t->ne[0], (صحیح) t->ne[1], (صحیح) t->ne[2], (صحیح) t->ne[3]);
        }
    }

    خالی format_op_dims(حرف * str, size_t max_سائز, ثابت htp_opnode & گرہ) {
        حرف * p = str;
        حرف * p_انتہا = str + max_سائز;
        خودکار inputs = گرہ.get_inputs();

        اگر (!inputs.خالی_خانے()) {
            p += std::min((size_t)format_tensor_dims(p, p_انتہا - p, inputs[0]), (size_t)(p_انتہا - p));

            برائے (size_t i = 1; i < inputs.سائز(); i++) {
                اگر (p < p_انتہا) {
                    p += std::min((size_t)snprintf(p, p_انتہا - p, " x "), (size_t)(p_انتہا - p));
                }
                اگر (p < p_انتہا) {
                    p += std::min((size_t)format_tensor_dims(p, p_انتہا - p, inputs[i]), (size_t)(p_انتہا - p));
                }
            }

            اگر (p < p_انتہا) {
                p += std::min((size_t)snprintf(p, p_انتہا - p, " -> "), (size_t)(p_انتہا - p));
            }
        }

        حرف self[64];
        format_tensor_dims(self, حجم(self), گرہ.dst());
        اگر (p < p_انتہا) {
            p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", self), (size_t)(p_انتہا - p));
        }
    }

    صحیح format_tensor_strides(حرف * str, size_t max_سائز, ثابت ساخت ggml_tensor * t) {
        اگر (!t) {
            واپس snprintf(str, max_سائز, "NONE");
        }
        ثابت حرف * c = ggml_is_contiguous(t) ? "" : "!";

        اگر (t->ne[2] == 1 && t->ne[3] == 1) {
            واپس snprintf(str, max_سائز, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } ورنہ {
            واپس snprintf(str, max_سائز, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    خالی format_op_strides(حرف * str, size_t max_سائز, ثابت htp_opnode & گرہ) {
        حرف * p = str;
        حرف * p_انتہا = str + max_سائز;
        خودکار inputs = گرہ.get_inputs();

        اگر (!inputs.خالی_خانے()) {
            p += std::min((size_t)format_tensor_strides(p, p_انتہا - p, inputs[0]), (size_t)(p_انتہا - p));

            برائے (size_t i = 1; i < inputs.سائز(); i++) {
                اگر (p < p_انتہا) {
                    p += std::min((size_t)snprintf(p, p_انتہا - p, " x "), (size_t)(p_انتہا - p));
                }
                اگر (p < p_انتہا) {
                    p += std::min((size_t)format_tensor_strides(p, p_انتہا - p, inputs[i]), (size_t)(p_انتہا - p));
                }
            }

            اگر (p < p_انتہا) {
                p += std::min((size_t)snprintf(p, p_انتہا - p, " -> "), (size_t)(p_انتہا - p));
            }
        }

        حرف self[64];
        format_tensor_strides(self, حجم(self), گرہ.dst());
        اگر (p < p_انتہا) {
            p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", self), (size_t)(p_انتہا - p));
        }
    }

    خالی format_op_types(حرف * str, size_t max_سائز, ثابت htp_opnode & گرہ) {
        حرف * p = str;
        حرف * p_انتہا = str + max_سائز;
        خودکار inputs = گرہ.get_inputs();

        اگر (!inputs.خالی_خانے()) {
            اگر (p < p_انتہا) {
                p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_انتہا - p));
            }

            برائے (size_t i = 1; i < inputs.سائز(); i++) {
                اگر (p < p_انتہا) {
                    p += std::min((size_t)snprintf(p, p_انتہا - p, " x "), (size_t)(p_انتہا - p));
                }
                اگر (p < p_انتہا) {
                    p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_انتہا - p));
                }
            }

            اگر (p < p_انتہا) {
                p += std::min((size_t)snprintf(p, p_انتہا - p, " -> "), (size_t)(p_انتہا - p));
            }
        }

        اگر (p < p_انتہا) {
            p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", ggml_type_name(گرہ.dst()->type)), (size_t)(p_انتہا - p));
        }
    }

    ثابت حرف * tensor_buff_name(ثابت ساخت ggml_tensor * t) {
        اگر (t && t->بفر) {
            واپس ggml_backend_بفر_name(t->بفر);
        }
        واپس "NONE";
    }

    خالی format_op_buffs(حرف * str, size_t max_سائز, ثابت htp_opnode & گرہ) {
        حرف * p = str;
        حرف * p_انتہا = str + max_سائز;
        خودکار inputs = گرہ.get_inputs();

        اگر (!inputs.خالی_خانے()) {
            اگر (p < p_انتہا) {
                p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_انتہا - p));
            }

            برائے (size_t i = 1; i < inputs.سائز(); i++) {
                اگر (p < p_انتہا) {
                    p += std::min((size_t)snprintf(p, p_انتہا - p, " x "), (size_t)(p_انتہا - p));
                }
                اگر (p < p_انتہا) {
                    p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_انتہا - p));
                }
            }

            اگر (p < p_انتہا) {
                p += std::min((size_t)snprintf(p, p_انتہا - p, " -> "), (size_t)(p_انتہا - p));
            }
        }

        اگر (p < p_انتہا) {
            p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", tensor_buff_name(گرہ.dst())), (size_t)(p_انتہا - p));
        }
    }

    خالی format_op_نام(حرف * str, size_t max_سائز, ثابت htp_opnode & گرہ) {
        حرف * p = str;
        حرف * p_انتہا = str + max_سائز;
        خودکار inputs = گرہ.get_inputs();

        اگر (!inputs.خالی_خانے()) {
            اگر (p < p_انتہا) {
                p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_انتہا - p));
            }

            برائے (size_t i = 1; i < inputs.سائز(); i++) {
                اگر (p < p_انتہا) {
                    p += std::min((size_t)snprintf(p, p_انتہا - p, " x "), (size_t)(p_انتہا - p));
                }
                اگر (p < p_انتہا) {
                    p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_انتہا - p));
                }
            }

            اگر (p < p_انتہا) {
                p += std::min((size_t)snprintf(p, p_انتہا - p, " -> "), (size_t)(p_انتہا - p));
            }
        }

        اگر (p < p_انتہا) {
            p += std::min((size_t)snprintf(p, p_انتہا - p, "%s", گرہ.dst()->name), (size_t)(p_انتہا - p));
        }
    }
    خالی format_kernel_params(حرف * str, size_t max_سائز, ثابت htp_opnode & گرہ) {
        اگر (گرہ.opcode == HTP_OP_MUL_MAT || گرہ.opcode == HTP_OP_MUL_MAT_ID ||
            گرہ.opcode == HTP_OP_MUL_MAT_NX || گرہ.opcode == HTP_OP_MUL_MAT_ID_NX ||
            گرہ.opcode == HTP_OP_MUL_MAT_جوڑو) {
            ثابت خودکار * kparams = (ثابت ساخت htp_mm_kernel_params *) گرہ.kernel_params;
            ثابت حرف * راستہ = "unknown";
            int32_t type = kparams->kernel_type;
            اگر (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                راستہ = (kparams->tile_سائز > 0) ? "hmx-tiled" : "hmx-flat";
            } ورنہ اگر (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                راستہ = "hvx-flat";
            } ورنہ اگر (type == HTP_MM_KERNEL_HVX_QUANT_صف || type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
                راستہ = "hvx-tiled";
            }
            snprintf(str, max_سائز, "%s vtcm %d", راستہ, (صحیح) kparams->vtcm_سائز);
        } ورنہ اگر (گرہ.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ثابت خودکار * kparams = (ثابت ساخت htp_fa_kernel_params *) گرہ.kernel_params;
            ثابت حرف * راستہ = "unknown";
            int32_t type = kparams->kernel_type;
            اگر (type == HTP_FA_KERNEL_HMX) {
                راستہ = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } ورنہ اگر (type == HTP_FA_KERNEL_HVX) {
                راستہ = "hvx";
            }
            snprintf(str, max_سائز, "%s vtcm %d", راستہ, (صحیح) kparams->vtcm_سائز);
        } ورنہ اگر (htp_op_is_unary(گرہ.opcode)) {
            ثابت خودکار * kparams = (ثابت ساخت htp_unary_kernel_params *) گرہ.kernel_params;
            snprintf(str, max_سائز, "%s vtcm %d", kparams->ستون_tile ? "wide-row" : "row-block", (صحیح) kparams->vtcm_سائز);
        } ورنہ اگر (گرہ.opcode == HTP_OP_MDEV_گروہ && گرہ.گرہ) {
            snprintf(str, max_سائز, "idx %d count %d", (صحیح) گرہ.گرہ->op_params[0], (صحیح) گرہ.dst()->ne[1]);
        } ورنہ اگر ((گرہ.opcode == HTP_OP_FENCE || گرہ.opcode == HTP_OP_CPY_FENCE) && گرہ.گرہ) {
            snprintf(str, max_سائز, "seq 0x%x", (uint32_t) گرہ.گرہ->op_params[0]);
        } ورنہ اگر (گرہ.opcode == HTP_OP_ALLREDUCE && گرہ.گرہ) {
            snprintf(str, max_سائز, "seq 0x%x -> 0x%x", (uint32_t) گرہ.گرہ->op_params[0], (uint32_t) گرہ.گرہ->op_params[1]);
        } ورنہ اگر (گرہ.opcode == HTP_OP_SSM_CONV) {
            ثابت خودکار * kparams = (ثابت ساخت htp_ssm_conv_kernel_params *) گرہ.kernel_params;
            snprintf(str, max_سائز, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (صحیح) kparams->vtcm_سائز);
        } ورنہ اگر (گرہ.opcode == HTP_OP_SOFTMAX) {
            ثابت خودکار * kparams = (ثابت ساخت htp_softmax_kernel_params *) گرہ.kernel_params;
            snprintf(str, max_سائز, "k%d nth %d vtcm %d", (صحیح) kparams->kernel_id, (صحیح) kparams->n_دھاگے, (صحیح) kparams->vtcm_سائز);
        } ورنہ اگر (گرہ.opcode == HTP_OP_GATED_DELTA_NET) {
            ثابت خودکار * kparams = (ثابت ساخت htp_gdn_kernel_params *) گرہ.kernel_params;
            ثابت حرف * راستہ = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_سائز, "%s-%s vtcm %u", راستہ, kparams->kda ? "kda" : "scalar",
                     (غیرعلامتی صحیح) (kparams->vtcm_سائز ? kparams->vtcm_سائز : kparams->vtcm_فی_thread * kparams->n_دھاگے));
        } ورنہ اگر (گرہ.opcode == HTP_OP_MUL || گرہ.opcode == HTP_OP_جوڑو || گرہ.opcode == HTP_OP_جوڑو_ID ||
                   گرہ.opcode == HTP_OP_SUB || گرہ.opcode == HTP_OP_DIV) {
            ثابت خودکار * kparams = (ثابت ساخت htp_ثنائی_kernel_params *) گرہ.kernel_params;
            snprintf(str, max_سائز, "vtcm %u", (غیرعلامتی صحیح) kparams->vtcm_سائز);
        } ورنہ اگر (گرہ.opcode == HTP_OP_ARGSORT || گرہ.opcode == HTP_OP_چوٹی_K) {
            ثابت خودکار * kparams = (ثابت ساخت htp_ترتیب_دو_kernel_params *) گرہ.kernel_params;
            snprintf(str, max_سائز, "%s nth %d nchk %d chk %d vtcm %d",
                     گرہ.opcode == HTP_OP_چوٹی_K ? "top_k" : "argsort",
                     (صحیح) kparams->n_دھاگے, (صحیح) kparams->n_chunks,
                     (صحیح) kparams->chunk_elems, (صحیح) kparams->vtcm_سائز);
        } ورنہ اگر (گرہ.opcode == HTP_OP_GET_صفیں) {
            ثابت خودکار * kparams = (ثابت ساخت htp_get_صفیں_kernel_params *) گرہ.kernel_params;
            ثابت حرف * ktype_str = "unknown";
            انتخاب (kparams->kernel_type) {
                حالت HTP_GET_صفیں_KERNEL_SAMETYPE: ktype_str = "sametype"; توڑو;
                حالت HTP_GET_صفیں_KERNEL_TILED:    ktype_str = "tiled"; توڑو;
                حالت HTP_GET_صفیں_KERNEL_FLAT:     ktype_str = "flat"; توڑو;
            }
            snprintf(str, max_سائز, "%s%s vtcm %u", ktype_str, kparams->n_دھاگے > 1 ? "-multi" : "", (غیرعلامتی صحیح) kparams->vtcm_سائز);
        } ورنہ {
            snprintf(str, max_سائز, "----");
        }

    }

    خالی format(ثابت htp_opnode & گرہ) {
        format_op_dims(dims, حجم(dims), گرہ);
        format_op_strides(strides, حجم(strides), گرہ);
        format_op_types(types, حجم(types), گرہ);
        format_op_buffs(buffs, حجم(buffs), گرہ);
        format_op_نام(نام, حجم(نام), گرہ);
        format_kernel_params(kparams, حجم(kparams), گرہ);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        نام[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ثابت htp_opnode & گرہ) { format(گرہ); }
};

#اگر_ختم // HTP_OPNODE_H
