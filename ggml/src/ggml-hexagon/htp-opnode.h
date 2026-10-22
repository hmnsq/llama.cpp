#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
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
    ggml_tensor * node   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<ثابت ggml_tensor *> inputs;
    std::vector<ثابت ggml_tensor *> outputs;
    std::string                      name;

    صحيح n_active_src(ثابت ggml_tensor * t) ثابت {
        אם (!t) החזר 0;
        עבור (صحيح i = GGML_MAX_SRC - 1; i >= 0; i--) {
            אם (t->src[i]) {
                החזר i + 1;
            }
        }
        החזר 0;
    }

    فراغ init(ggml_tensor * node) {
        this->node = node;
        אם (this->node) {
            אם (this->node->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.push_back(this->node->src[1]);
            } אחרת {
                this->name = ggml_op_desc(this->node);

                // Build inputs (preserving optional nullptrs)
                صحيح n_inputs = n_active_src(this->node);
                this->inputs.resize(n_inputs, nullptr);
                עבור (صحيح i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->node->src[i];
                }
            }

            // Build outputs
            this->outputs.push_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * node = nullptr) : opcode(opcode) {
        init(node);
    }

    ggml_op             op()   ثابت { החזר node->op; }
    ثابت ggml_tensor * src0() ثابت { החזר node->src[0]; }
    ثابت ggml_tensor * src1() ثابت { החזר node->src[1]; }
    ثابت ggml_tensor * dst()  ثابت { החזר outputs.empty() ? node : outputs.back(); }

    ggml_tensor * add_dummy(ثابت ggml_tensor & t) {
        dummy.push_back(std::make_shared<ggml_tensor>(t));
        החזר dummy.back().get();
    }

    فراغ add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.push_back(t);

        name += "+";
        name += ggml_op_desc(t);

        אם (extra_dst) {
            outputs.push_back(t);
        } אחרת {
            outputs.clear();
            outputs.push_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        صحيح n_inputs = n_active_src(t);
        עבור (صحيح i = 0; i < n_inputs; i++) {
            ثابت auto * src = t->src[i];
            אם (!src) {
                inputs.push_back(nullptr);
            } אחרת אם (src != node &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.push_back(src);
            }
        }
    }

    ثابت std::vector<ثابت ggml_tensor *> & get_inputs() ثابت {
        החזר inputs;
    }

    ثابت std::vector<ثابت ggml_tensor *> & get_outputs() ثابت {
        החזר outputs;
    }

    std::string op_name() ثابت {
        החזר name;
    }

    bool is_empty() ثابت {
        החזר ggml_op_is_empty(node->op);
    }

    bool stackable() ثابت {
        بدل (this->op()) {
            حالة GGML_OP_MUL_MAT:
                אם (ggml_get_op_params_i32(this->node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    החזר false;
                }
                החזר ggml_is_quantized(this->src0()->type);
            حالة GGML_OP_MUL_MAT_ID:
                החזר ggml_is_quantized(this->src0()->type);
            افتراضي:
                החזר false;
        }
    }

    bool same_input(ثابت htp_opnode& n) ثابت {
        החזר n.src1() == this->src1();
    }
};

بنية htp_opformat {
    حرف strides[64 * GGML_MAX_SRC];
    حرف dims[64 * GGML_MAX_SRC];
    حرف types[16 * GGML_MAX_SRC];
    حرف buffs[64 * GGML_MAX_SRC];
    حرف names[64 * GGML_MAX_SRC];
    حرف kparams[128];

    صحيح format_tensor_dims(حرف * str, size_t max_size, ثابت بنية ggml_tensor * t) {
        אם (!t) {
            החזר snprintf(str, max_size, "NONE");
        }
        אם (t->ne[2] == 1 && t->ne[3] == 1) {
            החזר snprintf(str, max_size, "%d:%d", (صحيح) t->ne[0], (صحيح) t->ne[1]);
        } אחרת {
            החזר snprintf(str, max_size, "%d:%d:%d:%d", (صحيح) t->ne[0], (صحيح) t->ne[1], (صحيح) t->ne[2], (صحيح) t->ne[3]);
        }
    }

    فراغ format_op_dims(حرف * str, size_t max_size, ثابت htp_opnode & node) {
        حرف * p = str;
        حرف * p_end = str + max_size;
        auto inputs = node.get_inputs();

        אם (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            עבור (size_t i = 1; i < inputs.size(); i++) {
                אם (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                אם (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            אם (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        حرف self[64];
        format_tensor_dims(self, حجم(self), node.dst());
        אם (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    صحيح format_tensor_strides(حرف * str, size_t max_size, ثابت بنية ggml_tensor * t) {
        אם (!t) {
            החזר snprintf(str, max_size, "NONE");
        }
        ثابت حرف * c = ggml_is_contiguous(t) ? "" : "!";

        אם (t->ne[2] == 1 && t->ne[3] == 1) {
            החזר snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } אחרת {
            החזר snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    فراغ format_op_strides(حرف * str, size_t max_size, ثابت htp_opnode & node) {
        حرف * p = str;
        حرف * p_end = str + max_size;
        auto inputs = node.get_inputs();

        אם (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            עבור (size_t i = 1; i < inputs.size(); i++) {
                אם (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                אם (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            אם (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        حرف self[64];
        format_tensor_strides(self, حجم(self), node.dst());
        אם (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    فراغ format_op_types(حرف * str, size_t max_size, ثابت htp_opnode & node) {
        حرف * p = str;
        حرف * p_end = str + max_size;
        auto inputs = node.get_inputs();

        אם (!inputs.empty()) {
            אם (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            עבור (size_t i = 1; i < inputs.size(); i++) {
                אם (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                אם (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            אם (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        אם (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(node.dst()->type)), (size_t)(p_end - p));
        }
    }

    ثابت حرف * tensor_buff_name(ثابت بنية ggml_tensor * t) {
        אם (t && t->buffer) {
            החזר ggml_backend_buffer_name(t->buffer);
        }
        החזר "NONE";
    }

    فراغ format_op_buffs(حرف * str, size_t max_size, ثابت htp_opnode & node) {
        حرف * p = str;
        حرف * p_end = str + max_size;
        auto inputs = node.get_inputs();

        אם (!inputs.empty()) {
            אם (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            עבור (size_t i = 1; i < inputs.size(); i++) {
                אם (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                אם (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            אם (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        אם (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(node.dst())), (size_t)(p_end - p));
        }
    }

    فراغ format_op_names(حرف * str, size_t max_size, ثابت htp_opnode & node) {
        حرف * p = str;
        حرف * p_end = str + max_size;
        auto inputs = node.get_inputs();

        אם (!inputs.empty()) {
            אם (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            עבור (size_t i = 1; i < inputs.size(); i++) {
                אם (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                אם (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            אם (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        אם (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", node.dst()->name), (size_t)(p_end - p));
        }
    }
    فراغ format_kernel_params(حرف * str, size_t max_size, ثابت htp_opnode & node) {
        אם (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID ||
            node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX ||
            node.opcode == HTP_OP_MUL_MAT_ADD) {
            ثابت auto * kparams = (ثابت بنية htp_mm_kernel_params *) node.kernel_params;
            ثابت حرف * path = "unknown";
            int32_t type = kparams->kernel_type;
            אם (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } אחרת אם (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } אחרת אם (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (صحيح) kparams->vtcm_size);
        } אחרת אם (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ثابت auto * kparams = (ثابت بنية htp_fa_kernel_params *) node.kernel_params;
            ثابت حرف * path = "unknown";
            int32_t type = kparams->kernel_type;
            אם (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } אחרת אם (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (صحيح) kparams->vtcm_size);
        } אחרת אם (htp_op_is_unary(node.opcode)) {
            ثابت auto * kparams = (ثابت بنية htp_unary_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (صحيح) kparams->vtcm_size);
        } אחרת אם (node.opcode == HTP_OP_MDEV_GROUP && node.node) {
            snprintf(str, max_size, "idx %d count %d", (صحيح) node.node->op_params[0], (صحيح) node.dst()->ne[1]);
        } אחרת אם ((node.opcode == HTP_OP_FENCE || node.opcode == HTP_OP_CPY_FENCE) && node.node) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) node.node->op_params[0]);
        } אחרת אם (node.opcode == HTP_OP_ALLREDUCE && node.node) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) node.node->op_params[0], (uint32_t) node.node->op_params[1]);
        } אחרת אם (node.opcode == HTP_OP_SSM_CONV) {
            ثابت auto * kparams = (ثابت بنية htp_ssm_conv_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (صحيح) kparams->vtcm_size);
        } אחרת אם (node.opcode == HTP_OP_SOFTMAX) {
            ثابت auto * kparams = (ثابت بنية htp_softmax_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (صحيح) kparams->kernel_id, (صحيح) kparams->n_threads, (صحيح) kparams->vtcm_size);
        } אחרת אם (node.opcode == HTP_OP_GATED_DELTA_NET) {
            ثابت auto * kparams = (ثابت بنية htp_gdn_kernel_params *) node.kernel_params;
            ثابت حرف * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned صحيح) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } אחרת אם (node.opcode == HTP_OP_MUL || node.opcode == HTP_OP_ADD || node.opcode == HTP_OP_ADD_ID ||
                   node.opcode == HTP_OP_SUB || node.opcode == HTP_OP_DIV) {
            ثابت auto * kparams = (ثابت بنية htp_binary_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "vtcm %u", (unsigned صحيح) kparams->vtcm_size);
        } אחרת אם (node.opcode == HTP_OP_ARGSORT || node.opcode == HTP_OP_TOP_K) {
            ثابت auto * kparams = (ثابت بنية htp_sort_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     node.opcode == HTP_OP_TOP_K ? "top_k" : "argsort",
                     (صحيح) kparams->n_threads, (صحيح) kparams->n_chunks,
                     (صحيح) kparams->chunk_elems, (صحيح) kparams->vtcm_size);
        } אחרת אם (node.opcode == HTP_OP_GET_ROWS) {
            ثابت auto * kparams = (ثابت بنية htp_get_rows_kernel_params *) node.kernel_params;
            ثابت حرف * ktype_str = "unknown";
            بدل (kparams->kernel_type) {
                حالة HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; اكسر;
                حالة HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; اكسر;
                حالة HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; اكسر;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned صحيح) kparams->vtcm_size);
        } אחרת {
            snprintf(str, max_size, "----");
        }

    }

    فراغ format(ثابت htp_opnode & node) {
        format_op_dims(dims, حجم(dims), node);
        format_op_strides(strides, حجم(strides), node);
        format_op_types(types, حجم(types), node);
        format_op_buffs(buffs, حجم(buffs), node);
        format_op_names(names, حجم(names), node);
        format_kernel_params(kparams, حجم(kparams), node);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ثابت htp_opnode & node) { format(node); }
};

#endif // HTP_OPNODE_H
