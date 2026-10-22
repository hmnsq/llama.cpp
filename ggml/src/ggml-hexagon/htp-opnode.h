#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-common.h"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <vector>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp/matmul-ops.h"
#അന്തര്ഭാവ "htp/flash-attn-ops.h"
#അന്തര്ഭാവ "htp/unary-ops.h"
#അന്തര്ഭാവ "htp/binary-ops.h"
#അന്തര്ഭാവ "htp/allreduce-ops.h"
#അന്തര്ഭാവ "htp/ssm-conv.h"
#അന്തര്ഭാവ "htp/gated-delta-net-ops.h"
#അന്തര്ഭാവ "htp/softmax-ops.h"
#അന്തര്ഭാവ "htp/argsort-ops.h"
#അന്തര്ഭാവ "htp/get-rows-ops.h"

ഘടന htp_opnode {
    ggml_tensor * കണ്ണി   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<സ്ഥിരം ggml_tensor *> inputs;
    std::vector<സ്ഥിരം ggml_tensor *> outputs;
    std::string                      name;

    പൂർണ്ണസംഖ്യ n_active_src(സ്ഥിരം ggml_tensor * t) സ്ഥിരം {
        എങ്കിൽ (!t) തിരികെ 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = GGML_MAX_SRC - 1; i >= 0; i--) {
            എങ്കിൽ (t->src[i]) {
                തിരികെ i + 1;
            }
        }
        തിരികെ 0;
    }

    ശൂന്യം init(ggml_tensor * കണ്ണി) {
        this->കണ്ണി = കണ്ണി;
        എങ്കിൽ (this->കണ്ണി) {
            എങ്കിൽ (this->കണ്ണി->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->കണ്ണി, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.തള്ളുക_back(this->കണ്ണി->src[1]);
            } അല്ലെങ്കിൽ {
                this->name = ggml_op_desc(this->കണ്ണി);

                // Build inputs (preserving optional nullptrs)
                പൂർണ്ണസംഖ്യ n_inputs = n_active_src(this->കണ്ണി);
                this->inputs.resize(n_inputs, nullptr);
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->കണ്ണി->src[i];
                }
            }

            // Build outputs
            this->outputs.തള്ളുക_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * കണ്ണി = nullptr) : opcode(opcode) {
        init(കണ്ണി);
    }

    ggml_op             op()   സ്ഥിരം { തിരികെ കണ്ണി->op; }
    സ്ഥിരം ggml_tensor * src0() സ്ഥിരം { തിരികെ കണ്ണി->src[0]; }
    സ്ഥിരം ggml_tensor * src1() സ്ഥിരം { തിരികെ കണ്ണി->src[1]; }
    സ്ഥിരം ggml_tensor * dst()  സ്ഥിരം { തിരികെ outputs.empty() ? കണ്ണി : outputs.back(); }

    ggml_tensor * add_dummy(സ്ഥിരം ggml_tensor & t) {
        dummy.തള്ളുക_back(std::make_shared<ggml_tensor>(t));
        തിരികെ dummy.back().get();
    }

    ശൂന്യം add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.തള്ളുക_back(t);

        name += "+";
        name += ggml_op_desc(t);

        എങ്കിൽ (extra_dst) {
            outputs.തള്ളുക_back(t);
        } അല്ലെങ്കിൽ {
            outputs.clear();
            outputs.തള്ളുക_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        പൂർണ്ണസംഖ്യ n_inputs = n_active_src(t);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_inputs; i++) {
            സ്ഥിരം auto * src = t->src[i];
            എങ്കിൽ (!src) {
                inputs.തള്ളുക_back(nullptr);
            } അല്ലെങ്കിൽ എങ്കിൽ (src != കണ്ണി &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.തള്ളുക_back(src);
            }
        }
    }

    സ്ഥിരം std::vector<സ്ഥിരം ggml_tensor *> & get_inputs() സ്ഥിരം {
        തിരികെ inputs;
    }

    സ്ഥിരം std::vector<സ്ഥിരം ggml_tensor *> & get_outputs() സ്ഥിരം {
        തിരികെ outputs;
    }

    std::string op_name() സ്ഥിരം {
        തിരികെ name;
    }

    bool is_empty() സ്ഥിരം {
        തിരികെ ggml_op_is_empty(കണ്ണി->op);
    }

    bool stackable() സ്ഥിരം {
        തിരഞ്ഞെടുപ്പ് (this->op()) {
            അവസ്ഥ GGML_OP_MUL_MAT:
                എങ്കിൽ (ggml_get_op_params_i32(this->കണ്ണി, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    തിരികെ false;
                }
                തിരികെ ggml_is_quantized(this->src0()->type);
            അവസ്ഥ GGML_OP_MUL_MAT_ID:
                തിരികെ ggml_is_quantized(this->src0()->type);
            സ്ഥിരസ്ഥിതി:
                തിരികെ false;
        }
    }

    bool same_നിവേശം(സ്ഥിരം htp_opnode& n) സ്ഥിരം {
        തിരികെ n.src1() == this->src1();
    }
};

ഘടന htp_opformat {
    അക്ഷരം strides[64 * GGML_MAX_SRC];
    അക്ഷരം dims[64 * GGML_MAX_SRC];
    അക്ഷരം types[16 * GGML_MAX_SRC];
    അക്ഷരം buffs[64 * GGML_MAX_SRC];
    അക്ഷരം names[64 * GGML_MAX_SRC];
    അക്ഷരം kparams[128];

    പൂർണ്ണസംഖ്യ format_tensor_dims(അക്ഷരം * str, size_t max_size, സ്ഥിരം ഘടന ggml_tensor * t) {
        എങ്കിൽ (!t) {
            തിരികെ snprintf(str, max_size, "NONE");
        }
        എങ്കിൽ (t->ne[2] == 1 && t->ne[3] == 1) {
            തിരികെ snprintf(str, max_size, "%d:%d", (പൂർണ്ണസംഖ്യ) t->ne[0], (പൂർണ്ണസംഖ്യ) t->ne[1]);
        } അല്ലെങ്കിൽ {
            തിരികെ snprintf(str, max_size, "%d:%d:%d:%d", (പൂർണ്ണസംഖ്യ) t->ne[0], (പൂർണ്ണസംഖ്യ) t->ne[1], (പൂർണ്ണസംഖ്യ) t->ne[2], (പൂർണ്ണസംഖ്യ) t->ne[3]);
        }
    }

    ശൂന്യം format_op_dims(അക്ഷരം * str, size_t max_size, സ്ഥിരം htp_opnode & കണ്ണി) {
        അക്ഷരം * p = str;
        അക്ഷരം * p_end = str + max_size;
        auto inputs = കണ്ണി.get_inputs();

        എങ്കിൽ (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            വേണ്ടി (size_t i = 1; i < inputs.size(); i++) {
                എങ്കിൽ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                എങ്കിൽ (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            എങ്കിൽ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        അക്ഷരം self[64];
        format_tensor_dims(self, വലുപ്പം(self), കണ്ണി.dst());
        എങ്കിൽ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    പൂർണ്ണസംഖ്യ format_tensor_strides(അക്ഷരം * str, size_t max_size, സ്ഥിരം ഘടന ggml_tensor * t) {
        എങ്കിൽ (!t) {
            തിരികെ snprintf(str, max_size, "NONE");
        }
        സ്ഥിരം അക്ഷരം * c = ggml_is_contiguous(t) ? "" : "!";

        എങ്കിൽ (t->ne[2] == 1 && t->ne[3] == 1) {
            തിരികെ snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } അല്ലെങ്കിൽ {
            തിരികെ snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    ശൂന്യം format_op_strides(അക്ഷരം * str, size_t max_size, സ്ഥിരം htp_opnode & കണ്ണി) {
        അക്ഷരം * p = str;
        അക്ഷരം * p_end = str + max_size;
        auto inputs = കണ്ണി.get_inputs();

        എങ്കിൽ (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            വേണ്ടി (size_t i = 1; i < inputs.size(); i++) {
                എങ്കിൽ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                എങ്കിൽ (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            എങ്കിൽ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        അക്ഷരം self[64];
        format_tensor_strides(self, വലുപ്പം(self), കണ്ണി.dst());
        എങ്കിൽ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    ശൂന്യം format_op_types(അക്ഷരം * str, size_t max_size, സ്ഥിരം htp_opnode & കണ്ണി) {
        അക്ഷരം * p = str;
        അക്ഷരം * p_end = str + max_size;
        auto inputs = കണ്ണി.get_inputs();

        എങ്കിൽ (!inputs.empty()) {
            എങ്കിൽ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            വേണ്ടി (size_t i = 1; i < inputs.size(); i++) {
                എങ്കിൽ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                എങ്കിൽ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            എങ്കിൽ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        എങ്കിൽ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(കണ്ണി.dst()->type)), (size_t)(p_end - p));
        }
    }

    സ്ഥിരം അക്ഷരം * tensor_buff_name(സ്ഥിരം ഘടന ggml_tensor * t) {
        എങ്കിൽ (t && t->buffer) {
            തിരികെ ggml_backend_buffer_name(t->buffer);
        }
        തിരികെ "NONE";
    }

    ശൂന്യം format_op_buffs(അക്ഷരം * str, size_t max_size, സ്ഥിരം htp_opnode & കണ്ണി) {
        അക്ഷരം * p = str;
        അക്ഷരം * p_end = str + max_size;
        auto inputs = കണ്ണി.get_inputs();

        എങ്കിൽ (!inputs.empty()) {
            എങ്കിൽ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            വേണ്ടി (size_t i = 1; i < inputs.size(); i++) {
                എങ്കിൽ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                എങ്കിൽ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            എങ്കിൽ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        എങ്കിൽ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(കണ്ണി.dst())), (size_t)(p_end - p));
        }
    }

    ശൂന്യം format_op_names(അക്ഷരം * str, size_t max_size, സ്ഥിരം htp_opnode & കണ്ണി) {
        അക്ഷരം * p = str;
        അക്ഷരം * p_end = str + max_size;
        auto inputs = കണ്ണി.get_inputs();

        എങ്കിൽ (!inputs.empty()) {
            എങ്കിൽ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            വേണ്ടി (size_t i = 1; i < inputs.size(); i++) {
                എങ്കിൽ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                എങ്കിൽ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            എങ്കിൽ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        എങ്കിൽ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", കണ്ണി.dst()->name), (size_t)(p_end - p));
        }
    }
    ശൂന്യം format_kernel_params(അക്ഷരം * str, size_t max_size, സ്ഥിരം htp_opnode & കണ്ണി) {
        എങ്കിൽ (കണ്ണി.opcode == HTP_OP_MUL_MAT || കണ്ണി.opcode == HTP_OP_MUL_MAT_ID ||
            കണ്ണി.opcode == HTP_OP_MUL_MAT_NX || കണ്ണി.opcode == HTP_OP_MUL_MAT_ID_NX ||
            കണ്ണി.opcode == HTP_OP_MUL_MAT_ADD) {
            സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_mm_kernel_params *) കണ്ണി.kernel_params;
            സ്ഥിരം അക്ഷരം * path = "unknown";
            int32_t type = kparams->kernel_type;
            എങ്കിൽ (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } അല്ലെങ്കിൽ എങ്കിൽ (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } അല്ലെങ്കിൽ എങ്കിൽ (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (പൂർണ്ണസംഖ്യ) kparams->vtcm_size);
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_FLASH_ATTN_EXT) {
            സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_fa_kernel_params *) കണ്ണി.kernel_params;
            സ്ഥിരം അക്ഷരം * path = "unknown";
            int32_t type = kparams->kernel_type;
            എങ്കിൽ (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } അല്ലെങ്കിൽ എങ്കിൽ (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (പൂർണ്ണസംഖ്യ) kparams->vtcm_size);
        } അല്ലെങ്കിൽ എങ്കിൽ (htp_op_is_unary(കണ്ണി.opcode)) {
            സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_unary_kernel_params *) കണ്ണി.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (പൂർണ്ണസംഖ്യ) kparams->vtcm_size);
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_MDEV_GROUP && കണ്ണി.കണ്ണി) {
            snprintf(str, max_size, "idx %d count %d", (പൂർണ്ണസംഖ്യ) കണ്ണി.കണ്ണി->op_params[0], (പൂർണ്ണസംഖ്യ) കണ്ണി.dst()->ne[1]);
        } അല്ലെങ്കിൽ എങ്കിൽ ((കണ്ണി.opcode == HTP_OP_FENCE || കണ്ണി.opcode == HTP_OP_CPY_FENCE) && കണ്ണി.കണ്ണി) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) കണ്ണി.കണ്ണി->op_params[0]);
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_ALLREDUCE && കണ്ണി.കണ്ണി) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) കണ്ണി.കണ്ണി->op_params[0], (uint32_t) കണ്ണി.കണ്ണി->op_params[1]);
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_SSM_CONV) {
            സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_ssm_conv_kernel_params *) കണ്ണി.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (പൂർണ്ണസംഖ്യ) kparams->vtcm_size);
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_SOFTMAX) {
            സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_softmax_kernel_params *) കണ്ണി.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (പൂർണ്ണസംഖ്യ) kparams->kernel_id, (പൂർണ്ണസംഖ്യ) kparams->n_threads, (പൂർണ്ണസംഖ്യ) kparams->vtcm_size);
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_GATED_DELTA_NET) {
            സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_gdn_kernel_params *) കണ്ണി.kernel_params;
            സ്ഥിരം അക്ഷരം * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned പൂർണ്ണസംഖ്യ) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_MUL || കണ്ണി.opcode == HTP_OP_ADD || കണ്ണി.opcode == HTP_OP_ADD_ID ||
                   കണ്ണി.opcode == HTP_OP_SUB || കണ്ണി.opcode == HTP_OP_DIV) {
            സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_ദ്വിമാന_kernel_params *) കണ്ണി.kernel_params;
            snprintf(str, max_size, "vtcm %u", (unsigned പൂർണ്ണസംഖ്യ) kparams->vtcm_size);
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_ARGSORT || കണ്ണി.opcode == HTP_OP_മുകൾ_K) {
            സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_sort_kernel_params *) കണ്ണി.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     കണ്ണി.opcode == HTP_OP_മുകൾ_K ? "top_k" : "argsort",
                     (പൂർണ്ണസംഖ്യ) kparams->n_threads, (പൂർണ്ണസംഖ്യ) kparams->n_chunks,
                     (പൂർണ്ണസംഖ്യ) kparams->chunk_elems, (പൂർണ്ണസംഖ്യ) kparams->vtcm_size);
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_GET_ROWS) {
            സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_get_rows_kernel_params *) കണ്ണി.kernel_params;
            സ്ഥിരം അക്ഷരം * ktype_str = "unknown";
            തിരഞ്ഞെടുപ്പ് (kparams->kernel_type) {
                അവസ്ഥ HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; നിർത്തുക;
                അവസ്ഥ HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; നിർത്തുക;
                അവസ്ഥ HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; നിർത്തുക;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned പൂർണ്ണസംഖ്യ) kparams->vtcm_size);
        } അല്ലെങ്കിൽ {
            snprintf(str, max_size, "----");
        }

    }

    ശൂന്യം format(സ്ഥിരം htp_opnode & കണ്ണി) {
        format_op_dims(dims, വലുപ്പം(dims), കണ്ണി);
        format_op_strides(strides, വലുപ്പം(strides), കണ്ണി);
        format_op_types(types, വലുപ്പം(types), കണ്ണി);
        format_op_buffs(buffs, വലുപ്പം(buffs), കണ്ണി);
        format_op_names(names, വലുപ്പം(names), കണ്ണി);
        format_kernel_params(kparams, വലുപ്പം(kparams), കണ്ണി);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(സ്ഥിരം htp_opnode & കണ്ണി) { format(കണ്ണി); }
};

#endif // HTP_OPNODE_H
