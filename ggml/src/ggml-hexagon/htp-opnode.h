#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-common.h"

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <vector>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp/matmul-ops.h"
#અન્તર્ભાવ "htp/flash-attn-ops.h"
#અન્તર્ભાવ "htp/unary-ops.h"
#અન્તર્ભાવ "htp/binary-ops.h"
#અન્તર્ભાવ "htp/allreduce-ops.h"
#અન્તર્ભાવ "htp/ssm-conv.h"
#અન્તર્ભાવ "htp/gated-delta-net-ops.h"
#અન્તર્ભાવ "htp/softmax-ops.h"
#અન્તર્ભાવ "htp/argsort-ops.h"
#અન્તર્ભાવ "htp/get-rows-ops.h"

સંરચના htp_opnode {
    ggml_tensor * ગાંઠ   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<અચલ ggml_tensor *> inputs;
    std::vector<અચલ ggml_tensor *> outputs;
    std::string                      name;

    પૂર્ણાંક n_active_src(અચલ ggml_tensor * t) અચલ {
        જો (!t) પાછા 0;
        માટે (પૂર્ણાંક i = GGML_MAX_SRC - 1; i >= 0; i--) {
            જો (t->src[i]) {
                પાછા i + 1;
            }
        }
        પાછા 0;
    }

    શૂન્ય init(ggml_tensor * ગાંઠ) {
        this->ગાંઠ = ગાંઠ;
        જો (this->ગાંઠ) {
            જો (this->ગાંઠ->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->ગાંઠ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ધકેલો_back(this->ગાંઠ->src[1]);
            } નહીંતર {
                this->name = ggml_op_desc(this->ગાંઠ);

                // Build inputs (preserving optional nullptrs)
                પૂર્ણાંક n_inputs = n_active_src(this->ગાંઠ);
                this->inputs.resize(n_inputs, nullptr);
                માટે (પૂર્ણાંક i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->ગાંઠ->src[i];
                }
            }

            // Build outputs
            this->outputs.ધકેલો_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * ગાંઠ = nullptr) : opcode(opcode) {
        init(ગાંઠ);
    }

    ggml_op             op()   અચલ { પાછા ગાંઠ->op; }
    અચલ ggml_tensor * src0() અચલ { પાછા ગાંઠ->src[0]; }
    અચલ ggml_tensor * src1() અચલ { પાછા ગાંઠ->src[1]; }
    અચલ ggml_tensor * dst()  અચલ { પાછા outputs.empty() ? ગાંઠ : outputs.back(); }

    ggml_tensor * add_dummy(અચલ ggml_tensor & t) {
        dummy.ધકેલો_back(std::make_shared<ggml_tensor>(t));
        પાછા dummy.back().get();
    }

    શૂન્ય add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ધકેલો_back(t);

        name += "+";
        name += ggml_op_desc(t);

        જો (extra_dst) {
            outputs.ધકેલો_back(t);
        } નહીંતર {
            outputs.clear();
            outputs.ધકેલો_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        પૂર્ણાંક n_inputs = n_active_src(t);
        માટે (પૂર્ણાંક i = 0; i < n_inputs; i++) {
            અચલ auto * src = t->src[i];
            જો (!src) {
                inputs.ધકેલો_back(nullptr);
            } નહીંતર જો (src != ગાંઠ &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.ધકેલો_back(src);
            }
        }
    }

    અચલ std::vector<અચલ ggml_tensor *> & get_inputs() અચલ {
        પાછા inputs;
    }

    અચલ std::vector<અચલ ggml_tensor *> & get_outputs() અચલ {
        પાછા outputs;
    }

    std::string op_name() અચલ {
        પાછા name;
    }

    bool is_empty() અચલ {
        પાછા ggml_op_is_empty(ગાંઠ->op);
    }

    bool stackable() અચલ {
        પસંદગી (this->op()) {
            સ્થિતિ GGML_OP_MUL_MAT:
                જો (ggml_get_op_params_i32(this->ગાંઠ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    પાછા false;
                }
                પાછા ggml_is_quantized(this->src0()->type);
            સ્થિતિ GGML_OP_MUL_MAT_ID:
                પાછા ggml_is_quantized(this->src0()->type);
            મૂળભૂત:
                પાછા false;
        }
    }

    bool same_આગત(અચલ htp_opnode& n) અચલ {
        પાછા n.src1() == this->src1();
    }
};

સંરચના htp_opformat {
    અક્ષર strides[64 * GGML_MAX_SRC];
    અક્ષર dims[64 * GGML_MAX_SRC];
    અક્ષર types[16 * GGML_MAX_SRC];
    અક્ષર buffs[64 * GGML_MAX_SRC];
    અક્ષર names[64 * GGML_MAX_SRC];
    અક્ષર kparams[128];

    પૂર્ણાંક format_tensor_dims(અક્ષર * str, size_t max_size, અચલ સંરચના ggml_tensor * t) {
        જો (!t) {
            પાછા snprintf(str, max_size, "NONE");
        }
        જો (t->ne[2] == 1 && t->ne[3] == 1) {
            પાછા snprintf(str, max_size, "%d:%d", (પૂર્ણાંક) t->ne[0], (પૂર્ણાંક) t->ne[1]);
        } નહીંતર {
            પાછા snprintf(str, max_size, "%d:%d:%d:%d", (પૂર્ણાંક) t->ne[0], (પૂર્ણાંક) t->ne[1], (પૂર્ણાંક) t->ne[2], (પૂર્ણાંક) t->ne[3]);
        }
    }

    શૂન્ય format_op_dims(અક્ષર * str, size_t max_size, અચલ htp_opnode & ગાંઠ) {
        અક્ષર * p = str;
        અક્ષર * p_end = str + max_size;
        auto inputs = ગાંઠ.get_inputs();

        જો (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            માટે (size_t i = 1; i < inputs.size(); i++) {
                જો (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                જો (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            જો (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        અક્ષર self[64];
        format_tensor_dims(self, કદ(self), ગાંઠ.dst());
        જો (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    પૂર્ણાંક format_tensor_strides(અક્ષર * str, size_t max_size, અચલ સંરચના ggml_tensor * t) {
        જો (!t) {
            પાછા snprintf(str, max_size, "NONE");
        }
        અચલ અક્ષર * c = ggml_is_contiguous(t) ? "" : "!";

        જો (t->ne[2] == 1 && t->ne[3] == 1) {
            પાછા snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } નહીંતર {
            પાછા snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    શૂન્ય format_op_strides(અક્ષર * str, size_t max_size, અચલ htp_opnode & ગાંઠ) {
        અક્ષર * p = str;
        અક્ષર * p_end = str + max_size;
        auto inputs = ગાંઠ.get_inputs();

        જો (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            માટે (size_t i = 1; i < inputs.size(); i++) {
                જો (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                જો (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            જો (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        અક્ષર self[64];
        format_tensor_strides(self, કદ(self), ગાંઠ.dst());
        જો (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    શૂન્ય format_op_types(અક્ષર * str, size_t max_size, અચલ htp_opnode & ગાંઠ) {
        અક્ષર * p = str;
        અક્ષર * p_end = str + max_size;
        auto inputs = ગાંઠ.get_inputs();

        જો (!inputs.empty()) {
            જો (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            માટે (size_t i = 1; i < inputs.size(); i++) {
                જો (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                જો (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            જો (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        જો (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(ગાંઠ.dst()->type)), (size_t)(p_end - p));
        }
    }

    અચલ અક્ષર * tensor_buff_name(અચલ સંરચના ggml_tensor * t) {
        જો (t && t->buffer) {
            પાછા ggml_backend_buffer_name(t->buffer);
        }
        પાછા "NONE";
    }

    શૂન્ય format_op_buffs(અક્ષર * str, size_t max_size, અચલ htp_opnode & ગાંઠ) {
        અક્ષર * p = str;
        અક્ષર * p_end = str + max_size;
        auto inputs = ગાંઠ.get_inputs();

        જો (!inputs.empty()) {
            જો (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            માટે (size_t i = 1; i < inputs.size(); i++) {
                જો (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                જો (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            જો (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        જો (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(ગાંઠ.dst())), (size_t)(p_end - p));
        }
    }

    શૂન્ય format_op_names(અક્ષર * str, size_t max_size, અચલ htp_opnode & ગાંઠ) {
        અક્ષર * p = str;
        અક્ષર * p_end = str + max_size;
        auto inputs = ગાંઠ.get_inputs();

        જો (!inputs.empty()) {
            જો (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            માટે (size_t i = 1; i < inputs.size(); i++) {
                જો (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                જો (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            જો (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        જો (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ગાંઠ.dst()->name), (size_t)(p_end - p));
        }
    }
    શૂન્ય format_kernel_params(અક્ષર * str, size_t max_size, અચલ htp_opnode & ગાંઠ) {
        જો (ગાંઠ.opcode == HTP_OP_MUL_MAT || ગાંઠ.opcode == HTP_OP_MUL_MAT_ID ||
            ગાંઠ.opcode == HTP_OP_MUL_MAT_NX || ગાંઠ.opcode == HTP_OP_MUL_MAT_ID_NX ||
            ગાંઠ.opcode == HTP_OP_MUL_MAT_ADD) {
            અચલ auto * kparams = (અચલ સંરચના htp_mm_kernel_params *) ગાંઠ.kernel_params;
            અચલ અક્ષર * path = "unknown";
            int32_t type = kparams->kernel_type;
            જો (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } નહીંતર જો (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } નહીંતર જો (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (પૂર્ણાંક) kparams->vtcm_size);
        } નહીંતર જો (ગાંઠ.opcode == HTP_OP_FLASH_ATTN_EXT) {
            અચલ auto * kparams = (અચલ સંરચના htp_fa_kernel_params *) ગાંઠ.kernel_params;
            અચલ અક્ષર * path = "unknown";
            int32_t type = kparams->kernel_type;
            જો (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } નહીંતર જો (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (પૂર્ણાંક) kparams->vtcm_size);
        } નહીંતર જો (htp_op_is_unary(ગાંઠ.opcode)) {
            અચલ auto * kparams = (અચલ સંરચના htp_unary_kernel_params *) ગાંઠ.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (પૂર્ણાંક) kparams->vtcm_size);
        } નહીંતર જો (ગાંઠ.opcode == HTP_OP_MDEV_GROUP && ગાંઠ.ગાંઠ) {
            snprintf(str, max_size, "idx %d count %d", (પૂર્ણાંક) ગાંઠ.ગાંઠ->op_params[0], (પૂર્ણાંક) ગાંઠ.dst()->ne[1]);
        } નહીંતર જો ((ગાંઠ.opcode == HTP_OP_FENCE || ગાંઠ.opcode == HTP_OP_CPY_FENCE) && ગાંઠ.ગાંઠ) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) ગાંઠ.ગાંઠ->op_params[0]);
        } નહીંતર જો (ગાંઠ.opcode == HTP_OP_ALLREDUCE && ગાંઠ.ગાંઠ) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) ગાંઠ.ગાંઠ->op_params[0], (uint32_t) ગાંઠ.ગાંઠ->op_params[1]);
        } નહીંતર જો (ગાંઠ.opcode == HTP_OP_SSM_CONV) {
            અચલ auto * kparams = (અચલ સંરચના htp_ssm_conv_kernel_params *) ગાંઠ.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (પૂર્ણાંક) kparams->vtcm_size);
        } નહીંતર જો (ગાંઠ.opcode == HTP_OP_SOFTMAX) {
            અચલ auto * kparams = (અચલ સંરચના htp_softmax_kernel_params *) ગાંઠ.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (પૂર્ણાંક) kparams->kernel_id, (પૂર્ણાંક) kparams->n_threads, (પૂર્ણાંક) kparams->vtcm_size);
        } નહીંતર જો (ગાંઠ.opcode == HTP_OP_GATED_DELTA_NET) {
            અચલ auto * kparams = (અચલ સંરચના htp_gdn_kernel_params *) ગાંઠ.kernel_params;
            અચલ અક્ષર * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned પૂર્ણાંક) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } નહીંતર જો (ગાંઠ.opcode == HTP_OP_MUL || ગાંઠ.opcode == HTP_OP_ADD || ગાંઠ.opcode == HTP_OP_ADD_ID ||
                   ગાંઠ.opcode == HTP_OP_SUB || ગાંઠ.opcode == HTP_OP_DIV) {
            અચલ auto * kparams = (અચલ સંરચના htp_દ્વિઅંકી_kernel_params *) ગાંઠ.kernel_params;
            snprintf(str, max_size, "vtcm %u", (unsigned પૂર્ણાંક) kparams->vtcm_size);
        } નહીંતર જો (ગાંઠ.opcode == HTP_OP_ARGSORT || ગાંઠ.opcode == HTP_OP_ટોચ_K) {
            અચલ auto * kparams = (અચલ સંરચના htp_sort_kernel_params *) ગાંઠ.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     ગાંઠ.opcode == HTP_OP_ટોચ_K ? "top_k" : "argsort",
                     (પૂર્ણાંક) kparams->n_threads, (પૂર્ણાંક) kparams->n_chunks,
                     (પૂર્ણાંક) kparams->chunk_elems, (પૂર્ણાંક) kparams->vtcm_size);
        } નહીંતર જો (ગાંઠ.opcode == HTP_OP_GET_ROWS) {
            અચલ auto * kparams = (અચલ સંરચના htp_get_rows_kernel_params *) ગાંઠ.kernel_params;
            અચલ અક્ષર * ktype_str = "unknown";
            પસંદગી (kparams->kernel_type) {
                સ્થિતિ HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; તોડો;
                સ્થિતિ HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; તોડો;
                સ્થિતિ HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; તોડો;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned પૂર્ણાંક) kparams->vtcm_size);
        } નહીંતર {
            snprintf(str, max_size, "----");
        }

    }

    શૂન્ય format(અચલ htp_opnode & ગાંઠ) {
        format_op_dims(dims, કદ(dims), ગાંઠ);
        format_op_strides(strides, કદ(strides), ગાંઠ);
        format_op_types(types, કદ(types), ગાંઠ);
        format_op_buffs(buffs, કદ(buffs), ગાંઠ);
        format_op_names(names, કદ(names), ગાંઠ);
        format_kernel_params(kparams, કદ(kparams), ગાંઠ);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(અચલ htp_opnode & ગાંઠ) { format(ગાંઠ); }
};

#endif // HTP_OPNODE_H
