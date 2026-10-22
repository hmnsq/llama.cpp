#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <vector>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/matmul-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/flash-attn-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/unary-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/binary-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/allreduce-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/ssm-conv.h"
#ଅନ୍ତର୍ଭାଵ "htp/gated-delta-net-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/softmax-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/argsort-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/get-rows-ops.h"

ସଂରଚନା htp_opnode {
    ggml_tensor * ଗଣ୍ଠି   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<ଅଚଳ ggml_tensor *> inputs;
    std::vector<ଅଚଳ ggml_tensor *> outputs;
    std::string                      name;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_active_src(ଅଚଳ ggml_tensor * t) ଅଚଳ {
        ଯଦି (!t) ଫେରନ୍ତୁ 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = GGML_MAX_SRC - 1; i >= 0; i--) {
            ଯଦି (t->src[i]) {
                ଫେରନ୍ତୁ i + 1;
            }
        }
        ଫେରନ୍ତୁ 0;
    }

    ଶୂନ୍ୟ init(ggml_tensor * ଗଣ୍ଠି) {
        this->ଗଣ୍ଠି = ଗଣ୍ଠି;
        ଯଦି (this->ଗଣ୍ଠି) {
            ଯଦି (this->ଗଣ୍ଠି->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->ଗଣ୍ଠି, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ଠେଲ_back(this->ଗଣ୍ଠି->src[1]);
            } ଅନ୍ୟଥା {
                this->name = ggml_op_desc(this->ଗଣ୍ଠି);

                // Build inputs (preserving optional nullptrs)
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_inputs = n_active_src(this->ଗଣ୍ଠି);
                this->inputs.resize(n_inputs, nullptr);
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->ଗଣ୍ଠି->src[i];
                }
            }

            // Build outputs
            this->outputs.ଠେଲ_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * ଗଣ୍ଠି = nullptr) : opcode(opcode) {
        init(ଗଣ୍ଠି);
    }

    ggml_op             op()   ଅଚଳ { ଫେରନ୍ତୁ ଗଣ୍ଠି->op; }
    ଅଚଳ ggml_tensor * src0() ଅଚଳ { ଫେରନ୍ତୁ ଗଣ୍ଠି->src[0]; }
    ଅଚଳ ggml_tensor * src1() ଅଚଳ { ଫେରନ୍ତୁ ଗଣ୍ଠି->src[1]; }
    ଅଚଳ ggml_tensor * dst()  ଅଚଳ { ଫେରନ୍ତୁ outputs.empty() ? ଗଣ୍ଠି : outputs.back(); }

    ggml_tensor * add_dummy(ଅଚଳ ggml_tensor & t) {
        dummy.ଠେଲ_back(std::make_shared<ggml_tensor>(t));
        ଫେରନ୍ତୁ dummy.back().get();
    }

    ଶୂନ୍ୟ add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ଠେଲ_back(t);

        name += "+";
        name += ggml_op_desc(t);

        ଯଦି (extra_dst) {
            outputs.ଠେଲ_back(t);
        } ଅନ୍ୟଥା {
            outputs.clear();
            outputs.ଠେଲ_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_inputs = n_active_src(t);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_inputs; i++) {
            ଅଚଳ auto * src = t->src[i];
            ଯଦି (!src) {
                inputs.ଠେଲ_back(nullptr);
            } ଅନ୍ୟଥା ଯଦି (src != ଗଣ୍ଠି &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.ଠେଲ_back(src);
            }
        }
    }

    ଅଚଳ std::vector<ଅଚଳ ggml_tensor *> & get_inputs() ଅଚଳ {
        ଫେରନ୍ତୁ inputs;
    }

    ଅଚଳ std::vector<ଅଚଳ ggml_tensor *> & get_outputs() ଅଚଳ {
        ଫେରନ୍ତୁ outputs;
    }

    std::string op_name() ଅଚଳ {
        ଫେରନ୍ତୁ name;
    }

    bool is_empty() ଅଚଳ {
        ଫେରନ୍ତୁ ggml_op_is_empty(ଗଣ୍ଠି->op);
    }

    bool stackable() ଅଚଳ {
        ବାଛ (this->op()) {
            ସ୍ଥିତି GGML_OP_MUL_MAT:
                ଯଦି (ggml_get_op_params_i32(this->ଗଣ୍ଠି, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    ଫେରନ୍ତୁ false;
                }
                ଫେରନ୍ତୁ ggml_is_quantized(this->src0()->type);
            ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
                ଫେରନ୍ତୁ ggml_is_quantized(this->src0()->type);
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଫେରନ୍ତୁ false;
        }
    }

    bool same_ନିବେଶ(ଅଚଳ htp_opnode& n) ଅଚଳ {
        ଫେରନ୍ତୁ n.src1() == this->src1();
    }
};

ସଂରଚନା htp_opformat {
    ଅକ୍ଷର strides[64 * GGML_MAX_SRC];
    ଅକ୍ଷର dims[64 * GGML_MAX_SRC];
    ଅକ୍ଷର types[16 * GGML_MAX_SRC];
    ଅକ୍ଷର buffs[64 * GGML_MAX_SRC];
    ଅକ୍ଷର names[64 * GGML_MAX_SRC];
    ଅକ୍ଷର kparams[128];

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା format_tensor_dims(ଅକ୍ଷର * str, size_t max_size, ଅଚଳ ସଂରଚନା ggml_tensor * t) {
        ଯଦି (!t) {
            ଫେରନ୍ତୁ snprintf(str, max_size, "NONE");
        }
        ଯଦି (t->ne[2] == 1 && t->ne[3] == 1) {
            ଫେରନ୍ତୁ snprintf(str, max_size, "%d:%d", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) t->ne[0], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) t->ne[1]);
        } ଅନ୍ୟଥା {
            ଫେରନ୍ତୁ snprintf(str, max_size, "%d:%d:%d:%d", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) t->ne[0], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) t->ne[1], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) t->ne[2], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) t->ne[3]);
        }
    }

    ଶୂନ୍ୟ format_op_dims(ଅକ୍ଷର * str, size_t max_size, ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଅକ୍ଷର * p = str;
        ଅକ୍ଷର * p_end = str + max_size;
        auto inputs = ଗଣ୍ଠି.get_inputs();

        ଯଦି (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            ପାଇଁ (size_t i = 1; i < inputs.size(); i++) {
                ଯଦି (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ଯଦି (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            ଯଦି (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ଅକ୍ଷର self[64];
        format_tensor_dims(self, ଆକାର(self), ଗଣ୍ଠି.dst());
        ଯଦି (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା format_tensor_strides(ଅକ୍ଷର * str, size_t max_size, ଅଚଳ ସଂରଚନା ggml_tensor * t) {
        ଯଦି (!t) {
            ଫେରନ୍ତୁ snprintf(str, max_size, "NONE");
        }
        ଅଚଳ ଅକ୍ଷର * c = ggml_is_contiguous(t) ? "" : "!";

        ଯଦି (t->ne[2] == 1 && t->ne[3] == 1) {
            ଫେରନ୍ତୁ snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } ଅନ୍ୟଥା {
            ଫେରନ୍ତୁ snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    ଶୂନ୍ୟ format_op_strides(ଅକ୍ଷର * str, size_t max_size, ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଅକ୍ଷର * p = str;
        ଅକ୍ଷର * p_end = str + max_size;
        auto inputs = ଗଣ୍ଠି.get_inputs();

        ଯଦି (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            ପାଇଁ (size_t i = 1; i < inputs.size(); i++) {
                ଯଦି (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ଯଦି (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            ଯଦି (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ଅକ୍ଷର self[64];
        format_tensor_strides(self, ଆକାର(self), ଗଣ୍ଠି.dst());
        ଯଦି (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    ଶୂନ୍ୟ format_op_types(ଅକ୍ଷର * str, size_t max_size, ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଅକ୍ଷର * p = str;
        ଅକ୍ଷର * p_end = str + max_size;
        auto inputs = ଗଣ୍ଠି.get_inputs();

        ଯଦି (!inputs.empty()) {
            ଯଦି (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            ପାଇଁ (size_t i = 1; i < inputs.size(); i++) {
                ଯଦି (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ଯଦି (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            ଯଦି (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ଯଦି (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(ଗଣ୍ଠି.dst()->type)), (size_t)(p_end - p));
        }
    }

    ଅଚଳ ଅକ୍ଷର * tensor_buff_name(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
        ଯଦି (t && t->buffer) {
            ଫେରନ୍ତୁ ggml_backend_buffer_name(t->buffer);
        }
        ଫେରନ୍ତୁ "NONE";
    }

    ଶୂନ୍ୟ format_op_buffs(ଅକ୍ଷର * str, size_t max_size, ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଅକ୍ଷର * p = str;
        ଅକ୍ଷର * p_end = str + max_size;
        auto inputs = ଗଣ୍ଠି.get_inputs();

        ଯଦି (!inputs.empty()) {
            ଯଦି (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            ପାଇଁ (size_t i = 1; i < inputs.size(); i++) {
                ଯଦି (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ଯଦି (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            ଯଦି (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ଯଦି (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(ଗଣ୍ଠି.dst())), (size_t)(p_end - p));
        }
    }

    ଶୂନ୍ୟ format_op_names(ଅକ୍ଷର * str, size_t max_size, ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଅକ୍ଷର * p = str;
        ଅକ୍ଷର * p_end = str + max_size;
        auto inputs = ଗଣ୍ଠି.get_inputs();

        ଯଦି (!inputs.empty()) {
            ଯଦି (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            ପାଇଁ (size_t i = 1; i < inputs.size(); i++) {
                ଯଦି (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ଯଦି (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            ଯଦି (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ଯଦି (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ଗଣ୍ଠି.dst()->name), (size_t)(p_end - p));
        }
    }
    ଶୂନ୍ୟ format_kernel_params(ଅକ୍ଷର * str, size_t max_size, ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT || ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT_ID ||
            ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT_NX || ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT_ID_NX ||
            ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT_ADD) {
            ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) ଗଣ୍ଠି.kernel_params;
            ଅଚଳ ଅକ୍ଷର * path = "unknown";
            int32_t type = kparams->kernel_type;
            ଯଦି (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } ଅନ୍ୟଥା ଯଦି (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } ଅନ୍ୟଥା ଯଦି (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->vtcm_size);
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_fa_kernel_params *) ଗଣ୍ଠି.kernel_params;
            ଅଚଳ ଅକ୍ଷର * path = "unknown";
            int32_t type = kparams->kernel_type;
            ଯଦି (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } ଅନ୍ୟଥା ଯଦି (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->vtcm_size);
        } ଅନ୍ୟଥା ଯଦି (htp_op_is_unary(ଗଣ୍ଠି.opcode)) {
            ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_unary_kernel_params *) ଗଣ୍ଠି.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->vtcm_size);
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_MDEV_GROUP && ଗଣ୍ଠି.ଗଣ୍ଠି) {
            snprintf(str, max_size, "idx %d count %d", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଗଣ୍ଠି.ଗଣ୍ଠି->op_params[0], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଗଣ୍ଠି.dst()->ne[1]);
        } ଅନ୍ୟଥା ଯଦି ((ଗଣ୍ଠି.opcode == HTP_OP_FENCE || ଗଣ୍ଠି.opcode == HTP_OP_CPY_FENCE) && ଗଣ୍ଠି.ଗଣ୍ଠି) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) ଗଣ୍ଠି.ଗଣ୍ଠି->op_params[0]);
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_ALLREDUCE && ଗଣ୍ଠି.ଗଣ୍ଠି) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) ଗଣ୍ଠି.ଗଣ୍ଠି->op_params[0], (uint32_t) ଗଣ୍ଠି.ଗଣ୍ଠି->op_params[1]);
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_SSM_CONV) {
            ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_ssm_conv_kernel_params *) ଗଣ୍ଠି.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->vtcm_size);
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_SOFTMAX) {
            ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_softmax_kernel_params *) ଗଣ୍ଠି.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->kernel_id, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->n_threads, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->vtcm_size);
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_GATED_DELTA_NET) {
            ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_gdn_kernel_params *) ଗଣ୍ଠି.kernel_params;
            ଅଚଳ ଅକ୍ଷର * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_MUL || ଗଣ୍ଠି.opcode == HTP_OP_ADD || ଗଣ୍ଠି.opcode == HTP_OP_ADD_ID ||
                   ଗଣ୍ଠି.opcode == HTP_OP_SUB || ଗଣ୍ଠି.opcode == HTP_OP_DIV) {
            ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_ଦ୍ୱିମିକ_kernel_params *) ଗଣ୍ଠି.kernel_params;
            snprintf(str, max_size, "vtcm %u", (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->vtcm_size);
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_ARGSORT || ଗଣ୍ଠି.opcode == HTP_OP_ଶୀର୍ଷ_K) {
            ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_sort_kernel_params *) ଗଣ୍ଠି.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     ଗଣ୍ଠି.opcode == HTP_OP_ଶୀର୍ଷ_K ? "top_k" : "argsort",
                     (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->n_threads, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->n_chunks,
                     (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->chunk_elems, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->vtcm_size);
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_GET_ROWS) {
            ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_get_rows_kernel_params *) ଗଣ୍ଠି.kernel_params;
            ଅଚଳ ଅକ୍ଷର * ktype_str = "unknown";
            ବାଛ (kparams->kernel_type) {
                ସ୍ଥିତି HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; ଭାଙ୍ଗ;
                ସ୍ଥିତି HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; ଭାଙ୍ଗ;
                ସ୍ଥିତି HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; ଭାଙ୍ଗ;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kparams->vtcm_size);
        } ଅନ୍ୟଥା {
            snprintf(str, max_size, "----");
        }

    }

    ଶୂନ୍ୟ format(ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        format_op_dims(dims, ଆକାର(dims), ଗଣ୍ଠି);
        format_op_strides(strides, ଆକାର(strides), ଗଣ୍ଠି);
        format_op_types(types, ଆକାର(types), ଗଣ୍ଠି);
        format_op_buffs(buffs, ଆକାର(buffs), ଗଣ୍ଠି);
        format_op_names(names, ଆକାର(names), ଗଣ୍ଠି);
        format_kernel_params(kparams, ଆକାର(kparams), ଗଣ୍ଠି);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ଅଚଳ htp_opnode & ଗଣ୍ଠି) { format(ଗଣ୍ଠି); }
};

#endif // HTP_OPNODE_H
