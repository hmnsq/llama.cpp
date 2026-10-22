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
    ggml_tensor * عقدة   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<ثابت ggml_tensor *> inputs;
    std::vector<ثابت ggml_tensor *> outputs;
    std::string                      name;

    صحيح n_active_src(ثابت ggml_tensor * t) ثابت {
        إذا (!t) إرجاع 0;
        لكل (صحيح i = GGML_MAX_SRC - 1; i >= 0; i--) {
            إذا (t->src[i]) {
                إرجاع i + 1;
            }
        }
        إرجاع 0;
    }

    فراغ init(ggml_tensor * عقدة) {
        this->عقدة = عقدة;
        إذا (this->عقدة) {
            إذا (this->عقدة->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->عقدة, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ادفع_back(this->عقدة->src[1]);
            } وإلا {
                this->name = ggml_op_desc(this->عقدة);

                // Build inputs (preserving optional nullptrs)
                صحيح n_inputs = n_active_src(this->عقدة);
                this->inputs.resize(n_inputs, nullptr);
                لكل (صحيح i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->عقدة->src[i];
                }
            }

            // Build outputs
            this->outputs.ادفع_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * عقدة = nullptr) : opcode(opcode) {
        init(عقدة);
    }

    ggml_op             op()   ثابت { إرجاع عقدة->op; }
    ثابت ggml_tensor * src0() ثابت { إرجاع عقدة->src[0]; }
    ثابت ggml_tensor * src1() ثابت { إرجاع عقدة->src[1]; }
    ثابت ggml_tensor * dst()  ثابت { إرجاع outputs.empty() ? عقدة : outputs.back(); }

    ggml_tensor * add_dummy(ثابت ggml_tensor & t) {
        dummy.ادفع_back(std::make_shared<ggml_tensor>(t));
        إرجاع dummy.back().get();
    }

    فراغ add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ادفع_back(t);

        name += "+";
        name += ggml_op_desc(t);

        إذا (extra_dst) {
            outputs.ادفع_back(t);
        } وإلا {
            outputs.clear();
            outputs.ادفع_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        صحيح n_inputs = n_active_src(t);
        لكل (صحيح i = 0; i < n_inputs; i++) {
            ثابت auto * src = t->src[i];
            إذا (!src) {
                inputs.ادفع_back(nullptr);
            } وإلا إذا (src != عقدة &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.ادفع_back(src);
            }
        }
    }

    ثابت std::vector<ثابت ggml_tensor *> & get_inputs() ثابت {
        إرجاع inputs;
    }

    ثابت std::vector<ثابت ggml_tensor *> & get_outputs() ثابت {
        إرجاع outputs;
    }

    std::string op_name() ثابت {
        إرجاع name;
    }

    bool is_empty() ثابت {
        إرجاع ggml_op_is_empty(عقدة->op);
    }

    bool stackable() ثابت {
        بدل (this->op()) {
            حالة GGML_OP_MUL_MAT:
                إذا (ggml_get_op_params_i32(this->عقدة, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    إرجاع false;
                }
                إرجاع ggml_is_quantized(this->src0()->type);
            حالة GGML_OP_MUL_MAT_ID:
                إرجاع ggml_is_quantized(this->src0()->type);
            افتراضي:
                إرجاع false;
        }
    }

    bool same_إدخال(ثابت htp_opnode& n) ثابت {
        إرجاع n.src1() == this->src1();
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
        إذا (!t) {
            إرجاع snprintf(str, max_size, "NONE");
        }
        إذا (t->ne[2] == 1 && t->ne[3] == 1) {
            إرجاع snprintf(str, max_size, "%d:%d", (صحيح) t->ne[0], (صحيح) t->ne[1]);
        } وإلا {
            إرجاع snprintf(str, max_size, "%d:%d:%d:%d", (صحيح) t->ne[0], (صحيح) t->ne[1], (صحيح) t->ne[2], (صحيح) t->ne[3]);
        }
    }

    فراغ format_op_dims(حرف * str, size_t max_size, ثابت htp_opnode & عقدة) {
        حرف * p = str;
        حرف * p_end = str + max_size;
        auto inputs = عقدة.get_inputs();

        إذا (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            لكل (size_t i = 1; i < inputs.size(); i++) {
                إذا (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                إذا (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            إذا (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        حرف self[64];
        format_tensor_dims(self, حجم(self), عقدة.dst());
        إذا (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    صحيح format_tensor_strides(حرف * str, size_t max_size, ثابت بنية ggml_tensor * t) {
        إذا (!t) {
            إرجاع snprintf(str, max_size, "NONE");
        }
        ثابت حرف * c = ggml_is_contiguous(t) ? "" : "!";

        إذا (t->ne[2] == 1 && t->ne[3] == 1) {
            إرجاع snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } وإلا {
            إرجاع snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    فراغ format_op_strides(حرف * str, size_t max_size, ثابت htp_opnode & عقدة) {
        حرف * p = str;
        حرف * p_end = str + max_size;
        auto inputs = عقدة.get_inputs();

        إذا (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            لكل (size_t i = 1; i < inputs.size(); i++) {
                إذا (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                إذا (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            إذا (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        حرف self[64];
        format_tensor_strides(self, حجم(self), عقدة.dst());
        إذا (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    فراغ format_op_types(حرف * str, size_t max_size, ثابت htp_opnode & عقدة) {
        حرف * p = str;
        حرف * p_end = str + max_size;
        auto inputs = عقدة.get_inputs();

        إذا (!inputs.empty()) {
            إذا (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            لكل (size_t i = 1; i < inputs.size(); i++) {
                إذا (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                إذا (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            إذا (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        إذا (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(عقدة.dst()->type)), (size_t)(p_end - p));
        }
    }

    ثابت حرف * tensor_buff_name(ثابت بنية ggml_tensor * t) {
        إذا (t && t->buffer) {
            إرجاع ggml_backend_buffer_name(t->buffer);
        }
        إرجاع "NONE";
    }

    فراغ format_op_buffs(حرف * str, size_t max_size, ثابت htp_opnode & عقدة) {
        حرف * p = str;
        حرف * p_end = str + max_size;
        auto inputs = عقدة.get_inputs();

        إذا (!inputs.empty()) {
            إذا (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            لكل (size_t i = 1; i < inputs.size(); i++) {
                إذا (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                إذا (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            إذا (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        إذا (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(عقدة.dst())), (size_t)(p_end - p));
        }
    }

    فراغ format_op_names(حرف * str, size_t max_size, ثابت htp_opnode & عقدة) {
        حرف * p = str;
        حرف * p_end = str + max_size;
        auto inputs = عقدة.get_inputs();

        إذا (!inputs.empty()) {
            إذا (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            لكل (size_t i = 1; i < inputs.size(); i++) {
                إذا (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                إذا (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            إذا (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        إذا (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", عقدة.dst()->name), (size_t)(p_end - p));
        }
    }
    فراغ format_kernel_params(حرف * str, size_t max_size, ثابت htp_opnode & عقدة) {
        إذا (عقدة.opcode == HTP_OP_MUL_MAT || عقدة.opcode == HTP_OP_MUL_MAT_ID ||
            عقدة.opcode == HTP_OP_MUL_MAT_NX || عقدة.opcode == HTP_OP_MUL_MAT_ID_NX ||
            عقدة.opcode == HTP_OP_MUL_MAT_ADD) {
            ثابت auto * kparams = (ثابت بنية htp_mm_kernel_params *) عقدة.kernel_params;
            ثابت حرف * path = "unknown";
            int32_t type = kparams->kernel_type;
            إذا (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } وإلا إذا (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } وإلا إذا (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (صحيح) kparams->vtcm_size);
        } وإلا إذا (عقدة.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ثابت auto * kparams = (ثابت بنية htp_fa_kernel_params *) عقدة.kernel_params;
            ثابت حرف * path = "unknown";
            int32_t type = kparams->kernel_type;
            إذا (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } وإلا إذا (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (صحيح) kparams->vtcm_size);
        } وإلا إذا (htp_op_is_unary(عقدة.opcode)) {
            ثابت auto * kparams = (ثابت بنية htp_unary_kernel_params *) عقدة.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (صحيح) kparams->vtcm_size);
        } وإلا إذا (عقدة.opcode == HTP_OP_MDEV_GROUP && عقدة.عقدة) {
            snprintf(str, max_size, "idx %d count %d", (صحيح) عقدة.عقدة->op_params[0], (صحيح) عقدة.dst()->ne[1]);
        } وإلا إذا ((عقدة.opcode == HTP_OP_FENCE || عقدة.opcode == HTP_OP_CPY_FENCE) && عقدة.عقدة) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) عقدة.عقدة->op_params[0]);
        } وإلا إذا (عقدة.opcode == HTP_OP_ALLREDUCE && عقدة.عقدة) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) عقدة.عقدة->op_params[0], (uint32_t) عقدة.عقدة->op_params[1]);
        } وإلا إذا (عقدة.opcode == HTP_OP_SSM_CONV) {
            ثابت auto * kparams = (ثابت بنية htp_ssm_conv_kernel_params *) عقدة.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (صحيح) kparams->vtcm_size);
        } وإلا إذا (عقدة.opcode == HTP_OP_SOFTMAX) {
            ثابت auto * kparams = (ثابت بنية htp_softmax_kernel_params *) عقدة.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (صحيح) kparams->kernel_id, (صحيح) kparams->n_threads, (صحيح) kparams->vtcm_size);
        } وإلا إذا (عقدة.opcode == HTP_OP_GATED_DELTA_NET) {
            ثابت auto * kparams = (ثابت بنية htp_gdn_kernel_params *) عقدة.kernel_params;
            ثابت حرف * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned صحيح) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } وإلا إذا (عقدة.opcode == HTP_OP_MUL || عقدة.opcode == HTP_OP_ADD || عقدة.opcode == HTP_OP_ADD_ID ||
                   عقدة.opcode == HTP_OP_SUB || عقدة.opcode == HTP_OP_DIV) {
            ثابت auto * kparams = (ثابت بنية htp_ثنائي_kernel_params *) عقدة.kernel_params;
            snprintf(str, max_size, "vtcm %u", (unsigned صحيح) kparams->vtcm_size);
        } وإلا إذا (عقدة.opcode == HTP_OP_ARGSORT || عقدة.opcode == HTP_OP_قمة_K) {
            ثابت auto * kparams = (ثابت بنية htp_sort_kernel_params *) عقدة.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     عقدة.opcode == HTP_OP_قمة_K ? "top_k" : "argsort",
                     (صحيح) kparams->n_threads, (صحيح) kparams->n_chunks,
                     (صحيح) kparams->chunk_elems, (صحيح) kparams->vtcm_size);
        } وإلا إذا (عقدة.opcode == HTP_OP_GET_ROWS) {
            ثابت auto * kparams = (ثابت بنية htp_get_rows_kernel_params *) عقدة.kernel_params;
            ثابت حرف * ktype_str = "unknown";
            بدل (kparams->kernel_type) {
                حالة HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; اكسر;
                حالة HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; اكسر;
                حالة HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; اكسر;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned صحيح) kparams->vtcm_size);
        } وإلا {
            snprintf(str, max_size, "----");
        }

    }

    فراغ format(ثابت htp_opnode & عقدة) {
        format_op_dims(dims, حجم(dims), عقدة);
        format_op_strides(strides, حجم(strides), عقدة);
        format_op_types(types, حجم(types), عقدة);
        format_op_buffs(buffs, حجم(buffs), عقدة);
        format_op_names(names, حجم(names), عقدة);
        format_kernel_params(kparams, حجم(kparams), عقدة);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ثابت htp_opnode & عقدة) { format(عقدة); }
};

#endif // HTP_OPNODE_H
