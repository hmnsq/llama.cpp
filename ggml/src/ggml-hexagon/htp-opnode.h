#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-common.h"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <string>
#অন্তর্ভাব <vector>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp/matmul-ops.h"
#অন্তর্ভাব "htp/flash-attn-ops.h"
#অন্তর্ভাব "htp/unary-ops.h"
#অন্তর্ভাব "htp/binary-ops.h"
#অন্তর্ভাব "htp/allreduce-ops.h"
#অন্তর্ভাব "htp/ssm-conv.h"
#অন্তর্ভাব "htp/gated-delta-net-ops.h"
#অন্তর্ভাব "htp/softmax-ops.h"
#অন্তর্ভাব "htp/argsort-ops.h"
#অন্তর্ভাব "htp/get-rows-ops.h"

কাঠামো htp_opnode {
    ggml_tensor * গ্রন্থি   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<ধ্রুবক ggml_tensor *> inputs;
    std::vector<ধ্রুবক ggml_tensor *> outputs;
    std::string                      name;

    পূর্ণসংখ্যা n_active_src(ধ্রুবক ggml_tensor * t) ধ্রুবক {
        যদি (!t) ফেরত 0;
        জন্য (পূর্ণসংখ্যা i = GGML_MAX_SRC - 1; i >= 0; i--) {
            যদি (t->src[i]) {
                ফেরত i + 1;
            }
        }
        ফেরত 0;
    }

    শূন্য init(ggml_tensor * গ্রন্থি) {
        this->গ্রন্থি = গ্রন্থি;
        যদি (this->গ্রন্থি) {
            যদি (this->গ্রন্থি->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->গ্রন্থি, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ঠেলো_back(this->গ্রন্থি->src[1]);
            } অন্যথা {
                this->name = ggml_op_desc(this->গ্রন্থি);

                // Build inputs (preserving optional nullptrs)
                পূর্ণসংখ্যা n_inputs = n_active_src(this->গ্রন্থি);
                this->inputs.resize(n_inputs, nullptr);
                জন্য (পূর্ণসংখ্যা i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->গ্রন্থি->src[i];
                }
            }

            // Build outputs
            this->outputs.ঠেলো_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * গ্রন্থি = nullptr) : opcode(opcode) {
        init(গ্রন্থি);
    }

    ggml_op             op()   ধ্রুবক { ফেরত গ্রন্থি->op; }
    ধ্রুবক ggml_tensor * src0() ধ্রুবক { ফেরত গ্রন্থি->src[0]; }
    ধ্রুবক ggml_tensor * src1() ধ্রুবক { ফেরত গ্রন্থি->src[1]; }
    ধ্রুবক ggml_tensor * dst()  ধ্রুবক { ফেরত outputs.empty() ? গ্রন্থি : outputs.back(); }

    ggml_tensor * add_dummy(ধ্রুবক ggml_tensor & t) {
        dummy.ঠেলো_back(std::make_shared<ggml_tensor>(t));
        ফেরত dummy.back().get();
    }

    শূন্য add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ঠেলো_back(t);

        name += "+";
        name += ggml_op_desc(t);

        যদি (extra_dst) {
            outputs.ঠেলো_back(t);
        } অন্যথা {
            outputs.clear();
            outputs.ঠেলো_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        পূর্ণসংখ্যা n_inputs = n_active_src(t);
        জন্য (পূর্ণসংখ্যা i = 0; i < n_inputs; i++) {
            ধ্রুবক auto * src = t->src[i];
            যদি (!src) {
                inputs.ঠেলো_back(nullptr);
            } অন্যথা যদি (src != গ্রন্থি &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.ঠেলো_back(src);
            }
        }
    }

    ধ্রুবক std::vector<ধ্রুবক ggml_tensor *> & get_inputs() ধ্রুবক {
        ফেরত inputs;
    }

    ধ্রুবক std::vector<ধ্রুবক ggml_tensor *> & get_outputs() ধ্রুবক {
        ফেরত outputs;
    }

    std::string op_name() ধ্রুবক {
        ফেরত name;
    }

    bool is_empty() ধ্রুবক {
        ফেরত ggml_op_is_empty(গ্রন্থি->op);
    }

    bool stackable() ধ্রুবক {
        নির্বাচন (this->op()) {
            ক্ষেত্র GGML_OP_MUL_MAT:
                যদি (ggml_get_op_params_i32(this->গ্রন্থি, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    ফেরত false;
                }
                ফেরত ggml_is_quantized(this->src0()->type);
            ক্ষেত্র GGML_OP_MUL_MAT_ID:
                ফেরত ggml_is_quantized(this->src0()->type);
            পূর্বনির্ধারিত:
                ফেরত false;
        }
    }

    bool same_প্রবেশ(ধ্রুবক htp_opnode& n) ধ্রুবক {
        ফেরত n.src1() == this->src1();
    }
};

কাঠামো htp_opformat {
    অক্ষর strides[64 * GGML_MAX_SRC];
    অক্ষর dims[64 * GGML_MAX_SRC];
    অক্ষর types[16 * GGML_MAX_SRC];
    অক্ষর buffs[64 * GGML_MAX_SRC];
    অক্ষর names[64 * GGML_MAX_SRC];
    অক্ষর kparams[128];

    পূর্ণসংখ্যা format_tensor_dims(অক্ষর * str, size_t max_আকার, ধ্রুবক কাঠামো ggml_tensor * t) {
        যদি (!t) {
            ফেরত snprintf(str, max_আকার, "NONE");
        }
        যদি (t->ne[2] == 1 && t->ne[3] == 1) {
            ফেরত snprintf(str, max_আকার, "%d:%d", (পূর্ণসংখ্যা) t->ne[0], (পূর্ণসংখ্যা) t->ne[1]);
        } অন্যথা {
            ফেরত snprintf(str, max_আকার, "%d:%d:%d:%d", (পূর্ণসংখ্যা) t->ne[0], (পূর্ণসংখ্যা) t->ne[1], (পূর্ণসংখ্যা) t->ne[2], (পূর্ণসংখ্যা) t->ne[3]);
        }
    }

    শূন্য format_op_dims(অক্ষর * str, size_t max_আকার, ধ্রুবক htp_opnode & গ্রন্থি) {
        অক্ষর * p = str;
        অক্ষর * p_end = str + max_আকার;
        auto inputs = গ্রন্থি.get_inputs();

        যদি (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            জন্য (size_t i = 1; i < inputs.আকার_2(); i++) {
                যদি (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                যদি (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            যদি (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        অক্ষর self[64];
        format_tensor_dims(self, আকার(self), গ্রন্থি.dst());
        যদি (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    পূর্ণসংখ্যা format_tensor_strides(অক্ষর * str, size_t max_আকার, ধ্রুবক কাঠামো ggml_tensor * t) {
        যদি (!t) {
            ফেরত snprintf(str, max_আকার, "NONE");
        }
        ধ্রুবক অক্ষর * c = ggml_is_contiguous(t) ? "" : "!";

        যদি (t->ne[2] == 1 && t->ne[3] == 1) {
            ফেরত snprintf(str, max_আকার, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } অন্যথা {
            ফেরত snprintf(str, max_আকার, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    শূন্য format_op_strides(অক্ষর * str, size_t max_আকার, ধ্রুবক htp_opnode & গ্রন্থি) {
        অক্ষর * p = str;
        অক্ষর * p_end = str + max_আকার;
        auto inputs = গ্রন্থি.get_inputs();

        যদি (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            জন্য (size_t i = 1; i < inputs.আকার_2(); i++) {
                যদি (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                যদি (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            যদি (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        অক্ষর self[64];
        format_tensor_strides(self, আকার(self), গ্রন্থি.dst());
        যদি (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    শূন্য format_op_types(অক্ষর * str, size_t max_আকার, ধ্রুবক htp_opnode & গ্রন্থি) {
        অক্ষর * p = str;
        অক্ষর * p_end = str + max_আকার;
        auto inputs = গ্রন্থি.get_inputs();

        যদি (!inputs.empty()) {
            যদি (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            জন্য (size_t i = 1; i < inputs.আকার_2(); i++) {
                যদি (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                যদি (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            যদি (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        যদি (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(গ্রন্থি.dst()->type)), (size_t)(p_end - p));
        }
    }

    ধ্রুবক অক্ষর * tensor_buff_name(ধ্রুবক কাঠামো ggml_tensor * t) {
        যদি (t && t->বাফার) {
            ফেরত ggml_backend_বাফার_name(t->বাফার);
        }
        ফেরত "NONE";
    }

    শূন্য format_op_buffs(অক্ষর * str, size_t max_আকার, ধ্রুবক htp_opnode & গ্রন্থি) {
        অক্ষর * p = str;
        অক্ষর * p_end = str + max_আকার;
        auto inputs = গ্রন্থি.get_inputs();

        যদি (!inputs.empty()) {
            যদি (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            জন্য (size_t i = 1; i < inputs.আকার_2(); i++) {
                যদি (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                যদি (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            যদি (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        যদি (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(গ্রন্থি.dst())), (size_t)(p_end - p));
        }
    }

    শূন্য format_op_names(অক্ষর * str, size_t max_আকার, ধ্রুবক htp_opnode & গ্রন্থি) {
        অক্ষর * p = str;
        অক্ষর * p_end = str + max_আকার;
        auto inputs = গ্রন্থি.get_inputs();

        যদি (!inputs.empty()) {
            যদি (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            জন্য (size_t i = 1; i < inputs.আকার_2(); i++) {
                যদি (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                যদি (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            যদি (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        যদি (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", গ্রন্থি.dst()->name), (size_t)(p_end - p));
        }
    }
    শূন্য format_kernel_params(অক্ষর * str, size_t max_আকার, ধ্রুবক htp_opnode & গ্রন্থি) {
        যদি (গ্রন্থি.opcode == HTP_OP_MUL_MAT || গ্রন্থি.opcode == HTP_OP_MUL_MAT_ID ||
            গ্রন্থি.opcode == HTP_OP_MUL_MAT_NX || গ্রন্থি.opcode == HTP_OP_MUL_MAT_ID_NX ||
            গ্রন্থি.opcode == HTP_OP_MUL_MAT_ADD) {
            ধ্রুবক auto * kparams = (ধ্রুবক কাঠামো htp_mm_kernel_params *) গ্রন্থি.kernel_params;
            ধ্রুবক অক্ষর * path = "unknown";
            int32_t type = kparams->kernel_type;
            যদি (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_আকার > 0) ? "hmx-tiled" : "hmx-flat";
            } অন্যথা যদি (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } অন্যথা যদি (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_আকার, "%s vtcm %d", path, (পূর্ণসংখ্যা) kparams->vtcm_আকার);
        } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ধ্রুবক auto * kparams = (ধ্রুবক কাঠামো htp_fa_kernel_params *) গ্রন্থি.kernel_params;
            ধ্রুবক অক্ষর * path = "unknown";
            int32_t type = kparams->kernel_type;
            যদি (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } অন্যথা যদি (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_আকার, "%s vtcm %d", path, (পূর্ণসংখ্যা) kparams->vtcm_আকার);
        } অন্যথা যদি (htp_op_is_unary(গ্রন্থি.opcode)) {
            ধ্রুবক auto * kparams = (ধ্রুবক কাঠামো htp_unary_kernel_params *) গ্রন্থি.kernel_params;
            snprintf(str, max_আকার, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (পূর্ণসংখ্যা) kparams->vtcm_আকার);
        } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_MDEV_GROUP && গ্রন্থি.গ্রন্থি) {
            snprintf(str, max_আকার, "idx %d count %d", (পূর্ণসংখ্যা) গ্রন্থি.গ্রন্থি->op_params[0], (পূর্ণসংখ্যা) গ্রন্থি.dst()->ne[1]);
        } অন্যথা যদি ((গ্রন্থি.opcode == HTP_OP_FENCE || গ্রন্থি.opcode == HTP_OP_CPY_FENCE) && গ্রন্থি.গ্রন্থি) {
            snprintf(str, max_আকার, "seq 0x%x", (uint32_t) গ্রন্থি.গ্রন্থি->op_params[0]);
        } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_ALLREDUCE && গ্রন্থি.গ্রন্থি) {
            snprintf(str, max_আকার, "seq 0x%x -> 0x%x", (uint32_t) গ্রন্থি.গ্রন্থি->op_params[0], (uint32_t) গ্রন্থি.গ্রন্থি->op_params[1]);
        } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_SSM_CONV) {
            ধ্রুবক auto * kparams = (ধ্রুবক কাঠামো htp_ssm_conv_kernel_params *) গ্রন্থি.kernel_params;
            snprintf(str, max_আকার, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (পূর্ণসংখ্যা) kparams->vtcm_আকার);
        } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_SOFTMAX) {
            ধ্রুবক auto * kparams = (ধ্রুবক কাঠামো htp_softmax_kernel_params *) গ্রন্থি.kernel_params;
            snprintf(str, max_আকার, "k%d nth %d vtcm %d", (পূর্ণসংখ্যা) kparams->kernel_id, (পূর্ণসংখ্যা) kparams->n_threads, (পূর্ণসংখ্যা) kparams->vtcm_আকার);
        } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_GATED_DELTA_NET) {
            ধ্রুবক auto * kparams = (ধ্রুবক কাঠামো htp_gdn_kernel_params *) গ্রন্থি.kernel_params;
            ধ্রুবক অক্ষর * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_আকার, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned পূর্ণসংখ্যা) (kparams->vtcm_আকার ? kparams->vtcm_আকার : kparams->vtcm_per_thread * kparams->n_threads));
        } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_MUL || গ্রন্থি.opcode == HTP_OP_ADD || গ্রন্থি.opcode == HTP_OP_ADD_ID ||
                   গ্রন্থি.opcode == HTP_OP_SUB || গ্রন্থি.opcode == HTP_OP_DIV) {
            ধ্রুবক auto * kparams = (ধ্রুবক কাঠামো htp_দ্বিমিক_kernel_params *) গ্রন্থি.kernel_params;
            snprintf(str, max_আকার, "vtcm %u", (unsigned পূর্ণসংখ্যা) kparams->vtcm_আকার);
        } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_ARGSORT || গ্রন্থি.opcode == HTP_OP_শীর্ষ_K) {
            ধ্রুবক auto * kparams = (ধ্রুবক কাঠামো htp_sort_kernel_params *) গ্রন্থি.kernel_params;
            snprintf(str, max_আকার, "%s nth %d nchk %d chk %d vtcm %d",
                     গ্রন্থি.opcode == HTP_OP_শীর্ষ_K ? "top_k" : "argsort",
                     (পূর্ণসংখ্যা) kparams->n_threads, (পূর্ণসংখ্যা) kparams->n_chunks,
                     (পূর্ণসংখ্যা) kparams->chunk_elems, (পূর্ণসংখ্যা) kparams->vtcm_আকার);
        } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_GET_ROWS) {
            ধ্রুবক auto * kparams = (ধ্রুবক কাঠামো htp_get_rows_kernel_params *) গ্রন্থি.kernel_params;
            ধ্রুবক অক্ষর * ktype_str = "unknown";
            নির্বাচন (kparams->kernel_type) {
                ক্ষেত্র HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; বিরতি;
                ক্ষেত্র HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; বিরতি;
                ক্ষেত্র HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; বিরতি;
            }
            snprintf(str, max_আকার, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned পূর্ণসংখ্যা) kparams->vtcm_আকার);
        } অন্যথা {
            snprintf(str, max_আকার, "----");
        }

    }

    শূন্য format(ধ্রুবক htp_opnode & গ্রন্থি) {
        format_op_dims(dims, আকার(dims), গ্রন্থি);
        format_op_strides(strides, আকার(strides), গ্রন্থি);
        format_op_types(types, আকার(types), গ্রন্থি);
        format_op_buffs(buffs, আকার(buffs), গ্রন্থি);
        format_op_names(names, আকার(names), গ্রন্থি);
        format_kernel_params(kparams, আকার(kparams), গ্রন্থি);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ধ্রুবক htp_opnode & গ্রন্থি) { format(গ্রন্থি); }
};

#endif // HTP_OPNODE_H
