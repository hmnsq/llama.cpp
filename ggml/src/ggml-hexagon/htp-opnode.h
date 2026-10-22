#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/matmul-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/flash-attn-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/unary-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/binary-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/allreduce-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/ssm-conv.h"
#ಅನ್ತರ್ಭಾವ "htp/gated-delta-net-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/softmax-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/argsort-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/get-rows-ops.h"

ಸಂರಚನಾ htp_opnode {
    ggml_tensor * ಗಂಟು   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<ಧ್ರುವ ggml_tensor *> inputs;
    std::vector<ಧ್ರುವ ggml_tensor *> outputs;
    std::string                      name;

    ಪೂರ್ಣಾಙ್ಕ n_active_src(ಧ್ರುವ ggml_tensor * t) ಧ್ರುವ {
        ಯದಿ (!t) ಪ್ರತ್ಯಾಗಮ 0;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = GGML_MAX_SRC - 1; i >= 0; i--) {
            ಯದಿ (t->src[i]) {
                ಪ್ರತ್ಯಾಗಮ i + 1;
            }
        }
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಶೂನ್ಯ init(ggml_tensor * ಗಂಟು) {
        this->ಗಂಟು = ಗಂಟು;
        ಯದಿ (this->ಗಂಟು) {
            ಯದಿ (this->ಗಂಟು->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->ಗಂಟು, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ತಳ್ಳು_back(this->ಗಂಟು->src[1]);
            } ಅನ್ಯಥಾ {
                this->name = ggml_op_desc(this->ಗಂಟು);

                // Build inputs (preserving optional nullptrs)
                ಪೂರ್ಣಾಙ್ಕ n_inputs = n_active_src(this->ಗಂಟು);
                this->inputs.resize(n_inputs, nullptr);
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->ಗಂಟು->src[i];
                }
            }

            // Build outputs
            this->outputs.ತಳ್ಳು_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * ಗಂಟು = nullptr) : opcode(opcode) {
        init(ಗಂಟು);
    }

    ggml_op             op()   ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ ಗಂಟು->op; }
    ಧ್ರುವ ggml_tensor * src0() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ ಗಂಟು->src[0]; }
    ಧ್ರುವ ggml_tensor * src1() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ ಗಂಟು->src[1]; }
    ಧ್ರುವ ggml_tensor * dst()  ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ outputs.empty() ? ಗಂಟು : outputs.back(); }

    ggml_tensor * add_dummy(ಧ್ರುವ ggml_tensor & t) {
        dummy.ತಳ್ಳು_back(std::make_shared<ggml_tensor>(t));
        ಪ್ರತ್ಯಾಗಮ dummy.back().get();
    }

    ಶೂನ್ಯ add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ತಳ್ಳು_back(t);

        name += "+";
        name += ggml_op_desc(t);

        ಯದಿ (extra_dst) {
            outputs.ತಳ್ಳು_back(t);
        } ಅನ್ಯಥಾ {
            outputs.clear();
            outputs.ತಳ್ಳು_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        ಪೂರ್ಣಾಙ್ಕ n_inputs = n_active_src(t);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_inputs; i++) {
            ಧ್ರುವ auto * src = t->src[i];
            ಯದಿ (!src) {
                inputs.ತಳ್ಳು_back(nullptr);
            } ಅನ್ಯಥಾ ಯದಿ (src != ಗಂಟು &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.ತಳ್ಳು_back(src);
            }
        }
    }

    ಧ್ರುವ std::vector<ಧ್ರುವ ggml_tensor *> & get_inputs() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ inputs;
    }

    ಧ್ರುವ std::vector<ಧ್ರುವ ggml_tensor *> & get_outputs() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ outputs;
    }

    std::string op_name() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ name;
    }

    bool is_empty() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ ggml_op_is_empty(ಗಂಟು->op);
    }

    bool stackable() ಧ್ರುವ {
        ವಿಕಲ್ಪ (this->op()) {
            ಪಕ್ಷ GGML_OP_MUL_MAT:
                ಯದಿ (ggml_get_op_params_i32(this->ಗಂಟು, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    ಪ್ರತ್ಯಾಗಮ false;
                }
                ಪ್ರತ್ಯಾಗಮ ggml_is_quantized(this->src0()->type);
            ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
                ಪ್ರತ್ಯಾಗಮ ggml_is_quantized(this->src0()->type);
            ಸ್ವಭಾವ:
                ಪ್ರತ್ಯಾಗಮ false;
        }
    }

    bool same_ಒಳಹರಿವು(ಧ್ರುವ htp_opnode& n) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ n.src1() == this->src1();
    }
};

ಸಂರಚನಾ htp_opformat {
    ವರ್ಣ strides[64 * GGML_MAX_SRC];
    ವರ್ಣ dims[64 * GGML_MAX_SRC];
    ವರ್ಣ types[16 * GGML_MAX_SRC];
    ವರ್ಣ buffs[64 * GGML_MAX_SRC];
    ವರ್ಣ names[64 * GGML_MAX_SRC];
    ವರ್ಣ kparams[128];

    ಪೂರ್ಣಾಙ್ಕ format_tensor_dims(ವರ್ಣ * str, size_t max_size, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
        ಯದಿ (!t) {
            ಪ್ರತ್ಯಾಗಮ snprintf(str, max_size, "NONE");
        }
        ಯದಿ (t->ne[2] == 1 && t->ne[3] == 1) {
            ಪ್ರತ್ಯಾಗಮ snprintf(str, max_size, "%d:%d", (ಪೂರ್ಣಾಙ್ಕ) t->ne[0], (ಪೂರ್ಣಾಙ್ಕ) t->ne[1]);
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ snprintf(str, max_size, "%d:%d:%d:%d", (ಪೂರ್ಣಾಙ್ಕ) t->ne[0], (ಪೂರ್ಣಾಙ್ಕ) t->ne[1], (ಪೂರ್ಣಾಙ್ಕ) t->ne[2], (ಪೂರ್ಣಾಙ್ಕ) t->ne[3]);
        }
    }

    ಶೂನ್ಯ format_op_dims(ವರ್ಣ * str, size_t max_size, ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ವರ್ಣ * p = str;
        ವರ್ಣ * p_end = str + max_size;
        auto inputs = ಗಂಟು.get_inputs();

        ಯದಿ (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            ಪ್ರತಿ (size_t i = 1; i < inputs.size(); i++) {
                ಯದಿ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ಯದಿ (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            ಯದಿ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ವರ್ಣ self[64];
        format_tensor_dims(self, ಪರಿಮಾಣ(self), ಗಂಟು.dst());
        ಯದಿ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    ಪೂರ್ಣಾಙ್ಕ format_tensor_strides(ವರ್ಣ * str, size_t max_size, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
        ಯದಿ (!t) {
            ಪ್ರತ್ಯಾಗಮ snprintf(str, max_size, "NONE");
        }
        ಧ್ರುವ ವರ್ಣ * c = ggml_is_contiguous(t) ? "" : "!";

        ಯದಿ (t->ne[2] == 1 && t->ne[3] == 1) {
            ಪ್ರತ್ಯಾಗಮ snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    ಶೂನ್ಯ format_op_strides(ವರ್ಣ * str, size_t max_size, ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ವರ್ಣ * p = str;
        ವರ್ಣ * p_end = str + max_size;
        auto inputs = ಗಂಟು.get_inputs();

        ಯದಿ (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            ಪ್ರತಿ (size_t i = 1; i < inputs.size(); i++) {
                ಯದಿ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ಯದಿ (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            ಯದಿ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ವರ್ಣ self[64];
        format_tensor_strides(self, ಪರಿಮಾಣ(self), ಗಂಟು.dst());
        ಯದಿ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    ಶೂನ್ಯ format_op_types(ವರ್ಣ * str, size_t max_size, ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ವರ್ಣ * p = str;
        ವರ್ಣ * p_end = str + max_size;
        auto inputs = ಗಂಟು.get_inputs();

        ಯದಿ (!inputs.empty()) {
            ಯದಿ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            ಪ್ರತಿ (size_t i = 1; i < inputs.size(); i++) {
                ಯದಿ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ಯದಿ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            ಯದಿ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ಯದಿ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(ಗಂಟು.dst()->type)), (size_t)(p_end - p));
        }
    }

    ಧ್ರುವ ವರ್ಣ * tensor_buff_name(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
        ಯದಿ (t && t->buffer) {
            ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_name(t->buffer);
        }
        ಪ್ರತ್ಯಾಗಮ "NONE";
    }

    ಶೂನ್ಯ format_op_buffs(ವರ್ಣ * str, size_t max_size, ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ವರ್ಣ * p = str;
        ವರ್ಣ * p_end = str + max_size;
        auto inputs = ಗಂಟು.get_inputs();

        ಯದಿ (!inputs.empty()) {
            ಯದಿ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            ಪ್ರತಿ (size_t i = 1; i < inputs.size(); i++) {
                ಯದಿ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ಯದಿ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            ಯದಿ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ಯದಿ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(ಗಂಟು.dst())), (size_t)(p_end - p));
        }
    }

    ಶೂನ್ಯ format_op_names(ವರ್ಣ * str, size_t max_size, ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ವರ್ಣ * p = str;
        ವರ್ಣ * p_end = str + max_size;
        auto inputs = ಗಂಟು.get_inputs();

        ಯದಿ (!inputs.empty()) {
            ಯದಿ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            ಪ್ರತಿ (size_t i = 1; i < inputs.size(); i++) {
                ಯದಿ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ಯದಿ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            ಯದಿ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ಯದಿ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ಗಂಟು.dst()->name), (size_t)(p_end - p));
        }
    }
    ಶೂನ್ಯ format_kernel_params(ವರ್ಣ * str, size_t max_size, ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ಯದಿ (ಗಂಟು.opcode == HTP_OP_MUL_MAT || ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID ||
            ಗಂಟು.opcode == HTP_OP_MUL_MAT_NX || ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID_NX ||
            ಗಂಟು.opcode == HTP_OP_MUL_MAT_ADD) {
            ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params *) ಗಂಟು.kernel_params;
            ಧ್ರುವ ವರ್ಣ * path = "unknown";
            int32_t type = kparams->kernel_type;
            ಯದಿ (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } ಅನ್ಯಥಾ ಯದಿ (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } ಅನ್ಯಥಾ ಯದಿ (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (ಪೂರ್ಣಾಙ್ಕ) kparams->vtcm_size);
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_fa_kernel_params *) ಗಂಟು.kernel_params;
            ಧ್ರುವ ವರ್ಣ * path = "unknown";
            int32_t type = kparams->kernel_type;
            ಯದಿ (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } ಅನ್ಯಥಾ ಯದಿ (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (ಪೂರ್ಣಾಙ್ಕ) kparams->vtcm_size);
        } ಅನ್ಯಥಾ ಯದಿ (htp_op_is_unary(ಗಂಟು.opcode)) {
            ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_unary_kernel_params *) ಗಂಟು.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (ಪೂರ್ಣಾಙ್ಕ) kparams->vtcm_size);
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_MDEV_GROUP && ಗಂಟು.ಗಂಟು) {
            snprintf(str, max_size, "idx %d count %d", (ಪೂರ್ಣಾಙ್ಕ) ಗಂಟು.ಗಂಟು->op_params[0], (ಪೂರ್ಣಾಙ್ಕ) ಗಂಟು.dst()->ne[1]);
        } ಅನ್ಯಥಾ ಯದಿ ((ಗಂಟು.opcode == HTP_OP_FENCE || ಗಂಟು.opcode == HTP_OP_CPY_FENCE) && ಗಂಟು.ಗಂಟು) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) ಗಂಟು.ಗಂಟು->op_params[0]);
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_ALLREDUCE && ಗಂಟು.ಗಂಟು) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) ಗಂಟು.ಗಂಟು->op_params[0], (uint32_t) ಗಂಟು.ಗಂಟು->op_params[1]);
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_SSM_CONV) {
            ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_ssm_conv_kernel_params *) ಗಂಟು.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (ಪೂರ್ಣಾಙ್ಕ) kparams->vtcm_size);
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_SOFTMAX) {
            ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_softmax_kernel_params *) ಗಂಟು.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (ಪೂರ್ಣಾಙ್ಕ) kparams->kernel_id, (ಪೂರ್ಣಾಙ್ಕ) kparams->n_threads, (ಪೂರ್ಣಾಙ್ಕ) kparams->vtcm_size);
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_GATED_DELTA_NET) {
            ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_gdn_kernel_params *) ಗಂಟು.kernel_params;
            ಧ್ರುವ ವರ್ಣ * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned ಪೂರ್ಣಾಙ್ಕ) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_MUL || ಗಂಟು.opcode == HTP_OP_ADD || ಗಂಟು.opcode == HTP_OP_ADD_ID ||
                   ಗಂಟು.opcode == HTP_OP_SUB || ಗಂಟು.opcode == HTP_OP_DIV) {
            ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_ದ್ವಿಮಾನ_kernel_params *) ಗಂಟು.kernel_params;
            snprintf(str, max_size, "vtcm %u", (unsigned ಪೂರ್ಣಾಙ್ಕ) kparams->vtcm_size);
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_ARGSORT || ಗಂಟು.opcode == HTP_OP_ತುದಿ_K) {
            ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_sort_kernel_params *) ಗಂಟು.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     ಗಂಟು.opcode == HTP_OP_ತುದಿ_K ? "top_k" : "argsort",
                     (ಪೂರ್ಣಾಙ್ಕ) kparams->n_threads, (ಪೂರ್ಣಾಙ್ಕ) kparams->n_chunks,
                     (ಪೂರ್ಣಾಙ್ಕ) kparams->chunk_elems, (ಪೂರ್ಣಾಙ್ಕ) kparams->vtcm_size);
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_GET_ROWS) {
            ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_get_rows_kernel_params *) ಗಂಟು.kernel_params;
            ಧ್ರುವ ವರ್ಣ * ktype_str = "unknown";
            ವಿಕಲ್ಪ (kparams->kernel_type) {
                ಪಕ್ಷ HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; ಭಙ್ಗ;
                ಪಕ್ಷ HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; ಭಙ್ಗ;
                ಪಕ್ಷ HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; ಭಙ್ಗ;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned ಪೂರ್ಣಾಙ್ಕ) kparams->vtcm_size);
        } ಅನ್ಯಥಾ {
            snprintf(str, max_size, "----");
        }

    }

    ಶೂನ್ಯ format(ಧ್ರುವ htp_opnode & ಗಂಟು) {
        format_op_dims(dims, ಪರಿಮಾಣ(dims), ಗಂಟು);
        format_op_strides(strides, ಪರಿಮಾಣ(strides), ಗಂಟು);
        format_op_types(types, ಪರಿಮಾಣ(types), ಗಂಟು);
        format_op_buffs(buffs, ಪರಿಮಾಣ(buffs), ಗಂಟು);
        format_op_names(names, ಪರಿಮಾಣ(names), ಗಂಟು);
        format_kernel_params(kparams, ಪರಿಮಾಣ(kparams), ಗಂಟು);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ಧ್ರುವ htp_opnode & ಗಂಟು) { format(ಗಂಟು); }
};

#endif // HTP_OPNODE_H
