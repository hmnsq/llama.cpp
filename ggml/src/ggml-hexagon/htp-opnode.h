#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
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

ساختار htp_opnode {
    ggml_tensor * گره   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<ثابت ggml_tensor *> inputs;
    std::vector<ثابت ggml_tensor *> outputs;
    std::string                      name;

    صحیح n_active_src(ثابت ggml_tensor * t) ثابت {
        اگر (!t) بازگشت 0;
        برای (صحیح i = GGML_MAX_SRC - 1; i >= 0; i--) {
            اگر (t->src[i]) {
                بازگشت i + 1;
            }
        }
        بازگشت 0;
    }

    تهی init(ggml_tensor * گره) {
        this->گره = گره;
        اگر (this->گره) {
            اگر (this->گره->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->گره, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.افزودن_back(this->گره->src[1]);
            } وگرنه {
                this->name = ggml_op_desc(this->گره);

                // Build inputs (preserving optional nullptrs)
                صحیح n_inputs = n_active_src(this->گره);
                this->inputs.resize(n_inputs, nullptr);
                برای (صحیح i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->گره->src[i];
                }
            }

            // Build outputs
            this->outputs.افزودن_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * گره = nullptr) : opcode(opcode) {
        init(گره);
    }

    ggml_op             op()   ثابت { بازگشت گره->op; }
    ثابت ggml_tensor * src0() ثابت { بازگشت گره->src[0]; }
    ثابت ggml_tensor * src1() ثابت { بازگشت گره->src[1]; }
    ثابت ggml_tensor * dst()  ثابت { بازگشت outputs.empty() ? گره : outputs.back(); }

    ggml_tensor * add_dummy(ثابت ggml_tensor & t) {
        dummy.افزودن_back(std::make_shared<ggml_tensor>(t));
        بازگشت dummy.back().get();
    }

    تهی add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.افزودن_back(t);

        name += "+";
        name += ggml_op_desc(t);

        اگر (extra_dst) {
            outputs.افزودن_back(t);
        } وگرنه {
            outputs.clear();
            outputs.افزودن_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        صحیح n_inputs = n_active_src(t);
        برای (صحیح i = 0; i < n_inputs; i++) {
            ثابت auto * src = t->src[i];
            اگر (!src) {
                inputs.افزودن_back(nullptr);
            } وگرنه اگر (src != گره &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.افزودن_back(src);
            }
        }
    }

    ثابت std::vector<ثابت ggml_tensor *> & get_inputs() ثابت {
        بازگشت inputs;
    }

    ثابت std::vector<ثابت ggml_tensor *> & get_outputs() ثابت {
        بازگشت outputs;
    }

    std::string op_name() ثابت {
        بازگشت name;
    }

    bool is_empty() ثابت {
        بازگشت ggml_op_is_empty(گره->op);
    }

    bool stackable() ثابت {
        انتخاب (this->op()) {
            حالت GGML_OP_MUL_MAT:
                اگر (ggml_get_op_params_i32(this->گره, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    بازگشت false;
                }
                بازگشت ggml_is_quantized(this->src0()->type);
            حالت GGML_OP_MUL_MAT_ID:
                بازگشت ggml_is_quantized(this->src0()->type);
            پیشفرض:
                بازگشت false;
        }
    }

    bool same_درون_داد(ثابت htp_opnode& n) ثابت {
        بازگشت n.src1() == this->src1();
    }
};

ساختار htp_opformat {
    نویسه strides[64 * GGML_MAX_SRC];
    نویسه dims[64 * GGML_MAX_SRC];
    نویسه types[16 * GGML_MAX_SRC];
    نویسه buffs[64 * GGML_MAX_SRC];
    نویسه names[64 * GGML_MAX_SRC];
    نویسه kparams[128];

    صحیح format_tensor_dims(نویسه * str, size_t max_size, ثابت ساختار ggml_tensor * t) {
        اگر (!t) {
            بازگشت snprintf(str, max_size, "NONE");
        }
        اگر (t->ne[2] == 1 && t->ne[3] == 1) {
            بازگشت snprintf(str, max_size, "%d:%d", (صحیح) t->ne[0], (صحیح) t->ne[1]);
        } وگرنه {
            بازگشت snprintf(str, max_size, "%d:%d:%d:%d", (صحیح) t->ne[0], (صحیح) t->ne[1], (صحیح) t->ne[2], (صحیح) t->ne[3]);
        }
    }

    تهی format_op_dims(نویسه * str, size_t max_size, ثابت htp_opnode & گره) {
        نویسه * p = str;
        نویسه * p_end = str + max_size;
        auto inputs = گره.get_inputs();

        اگر (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            برای (size_t i = 1; i < inputs.size(); i++) {
                اگر (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                اگر (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            اگر (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        نویسه self[64];
        format_tensor_dims(self, اندازه(self), گره.dst());
        اگر (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    صحیح format_tensor_strides(نویسه * str, size_t max_size, ثابت ساختار ggml_tensor * t) {
        اگر (!t) {
            بازگشت snprintf(str, max_size, "NONE");
        }
        ثابت نویسه * c = ggml_is_contiguous(t) ? "" : "!";

        اگر (t->ne[2] == 1 && t->ne[3] == 1) {
            بازگشت snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } وگرنه {
            بازگشت snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    تهی format_op_strides(نویسه * str, size_t max_size, ثابت htp_opnode & گره) {
        نویسه * p = str;
        نویسه * p_end = str + max_size;
        auto inputs = گره.get_inputs();

        اگر (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            برای (size_t i = 1; i < inputs.size(); i++) {
                اگر (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                اگر (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            اگر (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        نویسه self[64];
        format_tensor_strides(self, اندازه(self), گره.dst());
        اگر (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    تهی format_op_types(نویسه * str, size_t max_size, ثابت htp_opnode & گره) {
        نویسه * p = str;
        نویسه * p_end = str + max_size;
        auto inputs = گره.get_inputs();

        اگر (!inputs.empty()) {
            اگر (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            برای (size_t i = 1; i < inputs.size(); i++) {
                اگر (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                اگر (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            اگر (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        اگر (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(گره.dst()->type)), (size_t)(p_end - p));
        }
    }

    ثابت نویسه * tensor_buff_name(ثابت ساختار ggml_tensor * t) {
        اگر (t && t->buffer) {
            بازگشت ggml_backend_buffer_name(t->buffer);
        }
        بازگشت "NONE";
    }

    تهی format_op_buffs(نویسه * str, size_t max_size, ثابت htp_opnode & گره) {
        نویسه * p = str;
        نویسه * p_end = str + max_size;
        auto inputs = گره.get_inputs();

        اگر (!inputs.empty()) {
            اگر (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            برای (size_t i = 1; i < inputs.size(); i++) {
                اگر (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                اگر (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            اگر (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        اگر (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(گره.dst())), (size_t)(p_end - p));
        }
    }

    تهی format_op_names(نویسه * str, size_t max_size, ثابت htp_opnode & گره) {
        نویسه * p = str;
        نویسه * p_end = str + max_size;
        auto inputs = گره.get_inputs();

        اگر (!inputs.empty()) {
            اگر (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            برای (size_t i = 1; i < inputs.size(); i++) {
                اگر (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                اگر (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            اگر (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        اگر (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", گره.dst()->name), (size_t)(p_end - p));
        }
    }
    تهی format_kernel_params(نویسه * str, size_t max_size, ثابت htp_opnode & گره) {
        اگر (گره.opcode == HTP_OP_MUL_MAT || گره.opcode == HTP_OP_MUL_MAT_ID ||
            گره.opcode == HTP_OP_MUL_MAT_NX || گره.opcode == HTP_OP_MUL_MAT_ID_NX ||
            گره.opcode == HTP_OP_MUL_MAT_ADD) {
            ثابت auto * kparams = (ثابت ساختار htp_mm_kernel_params *) گره.kernel_params;
            ثابت نویسه * path = "unknown";
            int32_t type = kparams->kernel_type;
            اگر (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } وگرنه اگر (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } وگرنه اگر (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (صحیح) kparams->vtcm_size);
        } وگرنه اگر (گره.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ثابت auto * kparams = (ثابت ساختار htp_fa_kernel_params *) گره.kernel_params;
            ثابت نویسه * path = "unknown";
            int32_t type = kparams->kernel_type;
            اگر (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } وگرنه اگر (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (صحیح) kparams->vtcm_size);
        } وگرنه اگر (htp_op_is_unary(گره.opcode)) {
            ثابت auto * kparams = (ثابت ساختار htp_unary_kernel_params *) گره.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (صحیح) kparams->vtcm_size);
        } وگرنه اگر (گره.opcode == HTP_OP_MDEV_GROUP && گره.گره) {
            snprintf(str, max_size, "idx %d count %d", (صحیح) گره.گره->op_params[0], (صحیح) گره.dst()->ne[1]);
        } وگرنه اگر ((گره.opcode == HTP_OP_FENCE || گره.opcode == HTP_OP_CPY_FENCE) && گره.گره) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) گره.گره->op_params[0]);
        } وگرنه اگر (گره.opcode == HTP_OP_ALLREDUCE && گره.گره) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) گره.گره->op_params[0], (uint32_t) گره.گره->op_params[1]);
        } وگرنه اگر (گره.opcode == HTP_OP_SSM_CONV) {
            ثابت auto * kparams = (ثابت ساختار htp_ssm_conv_kernel_params *) گره.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (صحیح) kparams->vtcm_size);
        } وگرنه اگر (گره.opcode == HTP_OP_SOFTMAX) {
            ثابت auto * kparams = (ثابت ساختار htp_softmax_kernel_params *) گره.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (صحیح) kparams->kernel_id, (صحیح) kparams->n_threads, (صحیح) kparams->vtcm_size);
        } وگرنه اگر (گره.opcode == HTP_OP_GATED_DELTA_NET) {
            ثابت auto * kparams = (ثابت ساختار htp_gdn_kernel_params *) گره.kernel_params;
            ثابت نویسه * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned صحیح) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } وگرنه اگر (گره.opcode == HTP_OP_MUL || گره.opcode == HTP_OP_ADD || گره.opcode == HTP_OP_ADD_ID ||
                   گره.opcode == HTP_OP_SUB || گره.opcode == HTP_OP_DIV) {
            ثابت auto * kparams = (ثابت ساختار htp_دودویی_kernel_params *) گره.kernel_params;
            snprintf(str, max_size, "vtcm %u", (unsigned صحیح) kparams->vtcm_size);
        } وگرنه اگر (گره.opcode == HTP_OP_ARGSORT || گره.opcode == HTP_OP_بالاترین_K) {
            ثابت auto * kparams = (ثابت ساختار htp_sort_kernel_params *) گره.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     گره.opcode == HTP_OP_بالاترین_K ? "top_k" : "argsort",
                     (صحیح) kparams->n_threads, (صحیح) kparams->n_chunks,
                     (صحیح) kparams->chunk_elems, (صحیح) kparams->vtcm_size);
        } وگرنه اگر (گره.opcode == HTP_OP_GET_ROWS) {
            ثابت auto * kparams = (ثابت ساختار htp_get_rows_kernel_params *) گره.kernel_params;
            ثابت نویسه * ktype_str = "unknown";
            انتخاب (kparams->kernel_type) {
                حالت HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; بشکن;
                حالت HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; بشکن;
                حالت HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; بشکن;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned صحیح) kparams->vtcm_size);
        } وگرنه {
            snprintf(str, max_size, "----");
        }

    }

    تهی format(ثابت htp_opnode & گره) {
        format_op_dims(dims, اندازه(dims), گره);
        format_op_strides(strides, اندازه(strides), گره);
        format_op_types(types, اندازه(types), گره);
        format_op_buffs(buffs, اندازه(buffs), گره);
        format_op_names(names, اندازه(names), گره);
        format_kernel_params(kparams, اندازه(kparams), گره);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ثابت htp_opnode & گره) { format(گره); }
};

#endif // HTP_OPNODE_H
