#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-common.h"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <string>
#அந்தர்பாவ <vector>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp/matmul-ops.h"
#அந்தர்பாவ "htp/flash-attn-ops.h"
#அந்தர்பாவ "htp/unary-ops.h"
#அந்தர்பாவ "htp/binary-ops.h"
#அந்தர்பாவ "htp/allreduce-ops.h"
#அந்தர்பாவ "htp/ssm-conv.h"
#அந்தர்பாவ "htp/gated-delta-net-ops.h"
#அந்தர்பாவ "htp/softmax-ops.h"
#அந்தர்பாவ "htp/argsort-ops.h"
#அந்தர்பாவ "htp/get-rows-ops.h"

கட்டமைப்பு htp_opnode {
    ggml_tensor * முனை   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<மாறிலி ggml_tensor *> inputs;
    std::vector<மாறிலி ggml_tensor *> outputs;
    std::string                      name;

    முழு n_active_src(மாறிலி ggml_tensor * t) மாறிலி {
        எனில் (!t) திரும்பு 0;
        க்கு (முழு i = GGML_MAX_SRC - 1; i >= 0; i--) {
            எனில் (t->src[i]) {
                திரும்பு i + 1;
            }
        }
        திரும்பு 0;
    }

    வெறுமை init(ggml_tensor * முனை) {
        this->முனை = முனை;
        எனில் (this->முனை) {
            எனில் (this->முனை->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->முனை, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.தள்ளு_back(this->முனை->src[1]);
            } இல்லையெனில் {
                this->name = ggml_op_desc(this->முனை);

                // Build inputs (preserving optional nullptrs)
                முழு n_inputs = n_active_src(this->முனை);
                this->inputs.resize(n_inputs, nullptr);
                க்கு (முழு i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->முனை->src[i];
                }
            }

            // Build outputs
            this->outputs.தள்ளு_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * முனை = nullptr) : opcode(opcode) {
        init(முனை);
    }

    ggml_op             op()   மாறிலி { திரும்பு முனை->op; }
    மாறிலி ggml_tensor * src0() மாறிலி { திரும்பு முனை->src[0]; }
    மாறிலி ggml_tensor * src1() மாறிலி { திரும்பு முனை->src[1]; }
    மாறிலி ggml_tensor * dst()  மாறிலி { திரும்பு outputs.empty() ? முனை : outputs.back(); }

    ggml_tensor * add_dummy(மாறிலி ggml_tensor & t) {
        dummy.தள்ளு_back(std::make_shared<ggml_tensor>(t));
        திரும்பு dummy.back().get();
    }

    வெறுமை add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.தள்ளு_back(t);

        name += "+";
        name += ggml_op_desc(t);

        எனில் (extra_dst) {
            outputs.தள்ளு_back(t);
        } இல்லையெனில் {
            outputs.clear();
            outputs.தள்ளு_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        முழு n_inputs = n_active_src(t);
        க்கு (முழு i = 0; i < n_inputs; i++) {
            மாறிலி auto * src = t->src[i];
            எனில் (!src) {
                inputs.தள்ளு_back(nullptr);
            } இல்லையெனில் எனில் (src != முனை &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.தள்ளு_back(src);
            }
        }
    }

    மாறிலி std::vector<மாறிலி ggml_tensor *> & get_inputs() மாறிலி {
        திரும்பு inputs;
    }

    மாறிலி std::vector<மாறிலி ggml_tensor *> & get_outputs() மாறிலி {
        திரும்பு outputs;
    }

    std::string op_name() மாறிலி {
        திரும்பு name;
    }

    bool is_empty() மாறிலி {
        திரும்பு ggml_op_is_empty(முனை->op);
    }

    bool stackable() மாறிலி {
        தேர்வு (this->op()) {
            நிலை GGML_OP_MUL_MAT:
                எனில் (ggml_get_op_params_i32(this->முனை, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    திரும்பு false;
                }
                திரும்பு ggml_is_quantized(this->src0()->type);
            நிலை GGML_OP_MUL_MAT_ID:
                திரும்பு ggml_is_quantized(this->src0()->type);
            இயல்பு:
                திரும்பு false;
        }
    }

    bool same_உள்ளீடு(மாறிலி htp_opnode& n) மாறிலி {
        திரும்பு n.src1() == this->src1();
    }
};

கட்டமைப்பு htp_opformat {
    எழுத்து strides[64 * GGML_MAX_SRC];
    எழுத்து dims[64 * GGML_MAX_SRC];
    எழுத்து types[16 * GGML_MAX_SRC];
    எழுத்து buffs[64 * GGML_MAX_SRC];
    எழுத்து names[64 * GGML_MAX_SRC];
    எழுத்து kparams[128];

    முழு format_tensor_dims(எழுத்து * str, size_t max_size, மாறிலி கட்டமைப்பு ggml_tensor * t) {
        எனில் (!t) {
            திரும்பு snprintf(str, max_size, "NONE");
        }
        எனில் (t->ne[2] == 1 && t->ne[3] == 1) {
            திரும்பு snprintf(str, max_size, "%d:%d", (முழு) t->ne[0], (முழு) t->ne[1]);
        } இல்லையெனில் {
            திரும்பு snprintf(str, max_size, "%d:%d:%d:%d", (முழு) t->ne[0], (முழு) t->ne[1], (முழு) t->ne[2], (முழு) t->ne[3]);
        }
    }

    வெறுமை format_op_dims(எழுத்து * str, size_t max_size, மாறிலி htp_opnode & முனை) {
        எழுத்து * p = str;
        எழுத்து * p_end = str + max_size;
        auto inputs = முனை.get_inputs();

        எனில் (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            க்கு (size_t i = 1; i < inputs.size(); i++) {
                எனில் (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                எனில் (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            எனில் (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        எழுத்து self[64];
        format_tensor_dims(self, அளவு(self), முனை.dst());
        எனில் (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    முழு format_tensor_strides(எழுத்து * str, size_t max_size, மாறிலி கட்டமைப்பு ggml_tensor * t) {
        எனில் (!t) {
            திரும்பு snprintf(str, max_size, "NONE");
        }
        மாறிலி எழுத்து * c = ggml_is_contiguous(t) ? "" : "!";

        எனில் (t->ne[2] == 1 && t->ne[3] == 1) {
            திரும்பு snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } இல்லையெனில் {
            திரும்பு snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    வெறுமை format_op_strides(எழுத்து * str, size_t max_size, மாறிலி htp_opnode & முனை) {
        எழுத்து * p = str;
        எழுத்து * p_end = str + max_size;
        auto inputs = முனை.get_inputs();

        எனில் (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            க்கு (size_t i = 1; i < inputs.size(); i++) {
                எனில் (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                எனில் (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            எனில் (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        எழுத்து self[64];
        format_tensor_strides(self, அளவு(self), முனை.dst());
        எனில் (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    வெறுமை format_op_types(எழுத்து * str, size_t max_size, மாறிலி htp_opnode & முனை) {
        எழுத்து * p = str;
        எழுத்து * p_end = str + max_size;
        auto inputs = முனை.get_inputs();

        எனில் (!inputs.empty()) {
            எனில் (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            க்கு (size_t i = 1; i < inputs.size(); i++) {
                எனில் (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                எனில் (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            எனில் (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        எனில் (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(முனை.dst()->type)), (size_t)(p_end - p));
        }
    }

    மாறிலி எழுத்து * tensor_buff_name(மாறிலி கட்டமைப்பு ggml_tensor * t) {
        எனில் (t && t->buffer) {
            திரும்பு ggml_backend_buffer_name(t->buffer);
        }
        திரும்பு "NONE";
    }

    வெறுமை format_op_buffs(எழுத்து * str, size_t max_size, மாறிலி htp_opnode & முனை) {
        எழுத்து * p = str;
        எழுத்து * p_end = str + max_size;
        auto inputs = முனை.get_inputs();

        எனில் (!inputs.empty()) {
            எனில் (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            க்கு (size_t i = 1; i < inputs.size(); i++) {
                எனில் (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                எனில் (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            எனில் (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        எனில் (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(முனை.dst())), (size_t)(p_end - p));
        }
    }

    வெறுமை format_op_names(எழுத்து * str, size_t max_size, மாறிலி htp_opnode & முனை) {
        எழுத்து * p = str;
        எழுத்து * p_end = str + max_size;
        auto inputs = முனை.get_inputs();

        எனில் (!inputs.empty()) {
            எனில் (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            க்கு (size_t i = 1; i < inputs.size(); i++) {
                எனில் (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                எனில் (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            எனில் (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        எனில் (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", முனை.dst()->name), (size_t)(p_end - p));
        }
    }
    வெறுமை format_kernel_params(எழுத்து * str, size_t max_size, மாறிலி htp_opnode & முனை) {
        எனில் (முனை.opcode == HTP_OP_MUL_MAT || முனை.opcode == HTP_OP_MUL_MAT_ID ||
            முனை.opcode == HTP_OP_MUL_MAT_NX || முனை.opcode == HTP_OP_MUL_MAT_ID_NX ||
            முனை.opcode == HTP_OP_MUL_MAT_ADD) {
            மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_mm_kernel_params *) முனை.kernel_params;
            மாறிலி எழுத்து * path = "unknown";
            int32_t type = kparams->kernel_type;
            எனில் (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } இல்லையெனில் எனில் (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } இல்லையெனில் எனில் (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (முழு) kparams->vtcm_size);
        } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_FLASH_ATTN_EXT) {
            மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_fa_kernel_params *) முனை.kernel_params;
            மாறிலி எழுத்து * path = "unknown";
            int32_t type = kparams->kernel_type;
            எனில் (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } இல்லையெனில் எனில் (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (முழு) kparams->vtcm_size);
        } இல்லையெனில் எனில் (htp_op_is_unary(முனை.opcode)) {
            மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_unary_kernel_params *) முனை.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (முழு) kparams->vtcm_size);
        } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_MDEV_GROUP && முனை.முனை) {
            snprintf(str, max_size, "idx %d count %d", (முழு) முனை.முனை->op_params[0], (முழு) முனை.dst()->ne[1]);
        } இல்லையெனில் எனில் ((முனை.opcode == HTP_OP_FENCE || முனை.opcode == HTP_OP_CPY_FENCE) && முனை.முனை) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) முனை.முனை->op_params[0]);
        } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_ALLREDUCE && முனை.முனை) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) முனை.முனை->op_params[0], (uint32_t) முனை.முனை->op_params[1]);
        } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_SSM_CONV) {
            மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_ssm_conv_kernel_params *) முனை.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (முழு) kparams->vtcm_size);
        } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_SOFTMAX) {
            மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_softmax_kernel_params *) முனை.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (முழு) kparams->kernel_id, (முழு) kparams->n_threads, (முழு) kparams->vtcm_size);
        } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_GATED_DELTA_NET) {
            மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_gdn_kernel_params *) முனை.kernel_params;
            மாறிலி எழுத்து * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned முழு) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_MUL || முனை.opcode == HTP_OP_ADD || முனை.opcode == HTP_OP_ADD_ID ||
                   முனை.opcode == HTP_OP_SUB || முனை.opcode == HTP_OP_DIV) {
            மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_இருமம்_kernel_params *) முனை.kernel_params;
            snprintf(str, max_size, "vtcm %u", (unsigned முழு) kparams->vtcm_size);
        } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_ARGSORT || முனை.opcode == HTP_OP_உச்சி_K) {
            மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_sort_kernel_params *) முனை.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     முனை.opcode == HTP_OP_உச்சி_K ? "top_k" : "argsort",
                     (முழு) kparams->n_threads, (முழு) kparams->n_chunks,
                     (முழு) kparams->chunk_elems, (முழு) kparams->vtcm_size);
        } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_GET_ROWS) {
            மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_get_rows_kernel_params *) முனை.kernel_params;
            மாறிலி எழுத்து * ktype_str = "unknown";
            தேர்வு (kparams->kernel_type) {
                நிலை HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; நிறுத்து;
                நிலை HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; நிறுத்து;
                நிலை HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; நிறுத்து;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned முழு) kparams->vtcm_size);
        } இல்லையெனில் {
            snprintf(str, max_size, "----");
        }

    }

    வெறுமை format(மாறிலி htp_opnode & முனை) {
        format_op_dims(dims, அளவு(dims), முனை);
        format_op_strides(strides, அளவு(strides), முனை);
        format_op_types(types, அளவு(types), முனை);
        format_op_buffs(buffs, அளவு(buffs), முனை);
        format_op_names(names, அளவு(names), முனை);
        format_kernel_params(kparams, அளவு(kparams), முனை);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(மாறிலி htp_opnode & முனை) { format(முனை); }
};

#endif // HTP_OPNODE_H
