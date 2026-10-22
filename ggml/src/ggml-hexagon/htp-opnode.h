#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <string>
#अन्तर्भाव <vector>
#अन्तर्भाव <memory>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp/matmul-ops.h"
#अन्तर्भाव "htp/flash-attn-ops.h"
#अन्तर्भाव "htp/unary-ops.h"
#अन्तर्भाव "htp/binary-ops.h"
#अन्तर्भाव "htp/allreduce-ops.h"
#अन्तर्भाव "htp/ssm-conv.h"
#अन्तर्भाव "htp/gated-delta-net-ops.h"
#अन्तर्भाव "htp/softmax-ops.h"
#अन्तर्भाव "htp/argsort-ops.h"
#अन्तर्भाव "htp/get-rows-ops.h"

संरचना htp_opnode {
    ggml_tensor * गाँठ   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<ध्रुव ggml_tensor *> inputs;
    std::vector<ध्रुव ggml_tensor *> outputs;
    std::string                      name;

    पूर्णाङ्क n_active_src(ध्रुव ggml_tensor * t) ध्रुव {
        यदि (!t) प्रत्यागम 0;
        प्रति (पूर्णाङ्क i = GGML_MAX_SRC - 1; i >= 0; i--) {
            यदि (t->src[i]) {
                प्रत्यागम i + 1;
            }
        }
        प्रत्यागम 0;
    }

    शून्य init(ggml_tensor * गाँठ) {
        this->गाँठ = गाँठ;
        यदि (this->गाँठ) {
            यदि (this->गाँठ->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->गाँठ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.धकेलो_back(this->गाँठ->src[1]);
            } अन्यथा {
                this->name = ggml_op_desc(this->गाँठ);

                // Build inputs (preserving optional nullptrs)
                पूर्णाङ्क n_inputs = n_active_src(this->गाँठ);
                this->inputs.resize(n_inputs, nullptr);
                प्रति (पूर्णाङ्क i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->गाँठ->src[i];
                }
            }

            // Build outputs
            this->outputs.धकेलो_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * गाँठ = nullptr) : opcode(opcode) {
        init(गाँठ);
    }

    ggml_op             op()   ध्रुव { प्रत्यागम गाँठ->op; }
    ध्रुव ggml_tensor * src0() ध्रुव { प्रत्यागम गाँठ->src[0]; }
    ध्रुव ggml_tensor * src1() ध्रुव { प्रत्यागम गाँठ->src[1]; }
    ध्रुव ggml_tensor * dst()  ध्रुव { प्रत्यागम outputs.empty() ? गाँठ : outputs.back(); }

    ggml_tensor * add_dummy(ध्रुव ggml_tensor & t) {
        dummy.धकेलो_back(std::make_shared<ggml_tensor>(t));
        प्रत्यागम dummy.back().get();
    }

    शून्य add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.धकेलो_back(t);

        name += "+";
        name += ggml_op_desc(t);

        यदि (extra_dst) {
            outputs.धकेलो_back(t);
        } अन्यथा {
            outputs.clear();
            outputs.धकेलो_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::हटाओ(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        पूर्णाङ्क n_inputs = n_active_src(t);
        प्रति (पूर्णाङ्क i = 0; i < n_inputs; i++) {
            ध्रुव स्वतः * src = t->src[i];
            यदि (!src) {
                inputs.धकेलो_back(nullptr);
            } अन्यथा यदि (src != गाँठ &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.धकेलो_back(src);
            }
        }
    }

    ध्रुव std::vector<ध्रुव ggml_tensor *> & get_inputs() ध्रुव {
        प्रत्यागम inputs;
    }

    ध्रुव std::vector<ध्रुव ggml_tensor *> & get_outputs() ध्रुव {
        प्रत्यागम outputs;
    }

    std::string op_name() ध्रुव {
        प्रत्यागम name;
    }

    bool is_empty() ध्रुव {
        प्रत्यागम ggml_op_is_empty(गाँठ->op);
    }

    bool stackable() ध्रुव {
        विकल्प (this->op()) {
            पक्ष GGML_OP_MUL_MAT:
                यदि (ggml_get_op_params_i32(this->गाँठ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    प्रत्यागम false;
                }
                प्रत्यागम ggml_is_quantized(this->src0()->type);
            पक्ष GGML_OP_MUL_MAT_ID:
                प्रत्यागम ggml_is_quantized(this->src0()->type);
            स्वभाव:
                प्रत्यागम false;
        }
    }

    bool same_निवेश(ध्रुव htp_opnode& n) ध्रुव {
        प्रत्यागम n.src1() == this->src1();
    }
};

संरचना htp_opformat {
    वर्ण strides[64 * GGML_MAX_SRC];
    वर्ण dims[64 * GGML_MAX_SRC];
    वर्ण types[16 * GGML_MAX_SRC];
    वर्ण buffs[64 * GGML_MAX_SRC];
    वर्ण names[64 * GGML_MAX_SRC];
    वर्ण kparams[128];

    पूर्णाङ्क format_tensor_dims(वर्ण * str, माप_प्रकार max_size, ध्रुव संरचना ggml_tensor * t) {
        यदि (!t) {
            प्रत्यागम snprintf(str, max_size, "NONE");
        }
        यदि (t->ne[2] == 1 && t->ne[3] == 1) {
            प्रत्यागम snprintf(str, max_size, "%d:%d", (पूर्णाङ्क) t->ne[0], (पूर्णाङ्क) t->ne[1]);
        } अन्यथा {
            प्रत्यागम snprintf(str, max_size, "%d:%d:%d:%d", (पूर्णाङ्क) t->ne[0], (पूर्णाङ्क) t->ne[1], (पूर्णाङ्क) t->ne[2], (पूर्णाङ्क) t->ne[3]);
        }
    }

    शून्य format_op_dims(वर्ण * str, माप_प्रकार max_size, ध्रुव htp_opnode & गाँठ) {
        वर्ण * p = str;
        वर्ण * p_end = str + max_size;
        स्वतः inputs = गाँठ.get_inputs();

        यदि (!inputs.empty()) {
            p += std::min((माप_प्रकार)format_tensor_dims(p, p_end - p, inputs[0]), (माप_प्रकार)(p_end - p));

            प्रति (माप_प्रकार i = 1; i < inputs.size(); i++) {
                यदि (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                यदि (p < p_end) {
                    p += std::min((माप_प्रकार)format_tensor_dims(p, p_end - p, inputs[i]), (माप_प्रकार)(p_end - p));
                }
            }

            यदि (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        वर्ण self[64];
        format_tensor_dims(self, परिमाण(self), गाँठ.dst());
        यदि (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", self), (माप_प्रकार)(p_end - p));
        }
    }

    पूर्णाङ्क format_tensor_strides(वर्ण * str, माप_प्रकार max_size, ध्रुव संरचना ggml_tensor * t) {
        यदि (!t) {
            प्रत्यागम snprintf(str, max_size, "NONE");
        }
        ध्रुव वर्ण * c = ggml_is_contiguous(t) ? "" : "!";

        यदि (t->ne[2] == 1 && t->ne[3] == 1) {
            प्रत्यागम snprintf(str, max_size, "%zu:%zu%s", (माप_प्रकार) t->nb[0], (माप_प्रकार) t->nb[1], c);
        } अन्यथा {
            प्रत्यागम snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (माप_प्रकार) t->nb[0], (माप_प्रकार) t->nb[1], (माप_प्रकार) t->nb[2], (माप_प्रकार) t->nb[3], c);
        }
    }

    शून्य format_op_strides(वर्ण * str, माप_प्रकार max_size, ध्रुव htp_opnode & गाँठ) {
        वर्ण * p = str;
        वर्ण * p_end = str + max_size;
        स्वतः inputs = गाँठ.get_inputs();

        यदि (!inputs.empty()) {
            p += std::min((माप_प्रकार)format_tensor_strides(p, p_end - p, inputs[0]), (माप_प्रकार)(p_end - p));

            प्रति (माप_प्रकार i = 1; i < inputs.size(); i++) {
                यदि (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                यदि (p < p_end) {
                    p += std::min((माप_प्रकार)format_tensor_strides(p, p_end - p, inputs[i]), (माप_प्रकार)(p_end - p));
                }
            }

            यदि (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        वर्ण self[64];
        format_tensor_strides(self, परिमाण(self), गाँठ.dst());
        यदि (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", self), (माप_प्रकार)(p_end - p));
        }
    }

    शून्य format_op_types(वर्ण * str, माप_प्रकार max_size, ध्रुव htp_opnode & गाँठ) {
        वर्ण * p = str;
        वर्ण * p_end = str + max_size;
        स्वतः inputs = गाँठ.get_inputs();

        यदि (!inputs.empty()) {
            यदि (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (माप_प्रकार)(p_end - p));
            }

            प्रति (माप_प्रकार i = 1; i < inputs.size(); i++) {
                यदि (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                यदि (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (माप_प्रकार)(p_end - p));
                }
            }

            यदि (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        यदि (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", ggml_type_name(गाँठ.dst()->type)), (माप_प्रकार)(p_end - p));
        }
    }

    ध्रुव वर्ण * tensor_buff_name(ध्रुव संरचना ggml_tensor * t) {
        यदि (t && t->buffer) {
            प्रत्यागम ggml_backend_buffer_name(t->buffer);
        }
        प्रत्यागम "NONE";
    }

    शून्य format_op_buffs(वर्ण * str, माप_प्रकार max_size, ध्रुव htp_opnode & गाँठ) {
        वर्ण * p = str;
        वर्ण * p_end = str + max_size;
        स्वतः inputs = गाँठ.get_inputs();

        यदि (!inputs.empty()) {
            यदि (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (माप_प्रकार)(p_end - p));
            }

            प्रति (माप_प्रकार i = 1; i < inputs.size(); i++) {
                यदि (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                यदि (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (माप_प्रकार)(p_end - p));
                }
            }

            यदि (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        यदि (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", tensor_buff_name(गाँठ.dst())), (माप_प्रकार)(p_end - p));
        }
    }

    शून्य format_op_names(वर्ण * str, माप_प्रकार max_size, ध्रुव htp_opnode & गाँठ) {
        वर्ण * p = str;
        वर्ण * p_end = str + max_size;
        स्वतः inputs = गाँठ.get_inputs();

        यदि (!inputs.empty()) {
            यदि (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (माप_प्रकार)(p_end - p));
            }

            प्रति (माप_प्रकार i = 1; i < inputs.size(); i++) {
                यदि (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                यदि (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (माप_प्रकार)(p_end - p));
                }
            }

            यदि (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        यदि (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", गाँठ.dst()->name), (माप_प्रकार)(p_end - p));
        }
    }
    शून्य format_kernel_params(वर्ण * str, माप_प्रकार max_size, ध्रुव htp_opnode & गाँठ) {
        यदि (गाँठ.opcode == HTP_OP_MUL_MAT || गाँठ.opcode == HTP_OP_MUL_MAT_ID ||
            गाँठ.opcode == HTP_OP_MUL_MAT_NX || गाँठ.opcode == HTP_OP_MUL_MAT_ID_NX ||
            गाँठ.opcode == HTP_OP_MUL_MAT_ADD) {
            ध्रुव स्वतः * kparams = (ध्रुव संरचना htp_mm_kernel_params *) गाँठ.kernel_params;
            ध्रुव वर्ण * path = "unknown";
            int32_t type = kparams->kernel_type;
            यदि (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } अन्यथा यदि (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } अन्यथा यदि (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (पूर्णाङ्क) kparams->vtcm_size);
        } अन्यथा यदि (गाँठ.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ध्रुव स्वतः * kparams = (ध्रुव संरचना htp_fa_kernel_params *) गाँठ.kernel_params;
            ध्रुव वर्ण * path = "unknown";
            int32_t type = kparams->kernel_type;
            यदि (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } अन्यथा यदि (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (पूर्णाङ्क) kparams->vtcm_size);
        } अन्यथा यदि (htp_op_is_unary(गाँठ.opcode)) {
            ध्रुव स्वतः * kparams = (ध्रुव संरचना htp_unary_kernel_params *) गाँठ.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (पूर्णाङ्क) kparams->vtcm_size);
        } अन्यथा यदि (गाँठ.opcode == HTP_OP_MDEV_GROUP && गाँठ.गाँठ) {
            snprintf(str, max_size, "idx %d count %d", (पूर्णाङ्क) गाँठ.गाँठ->op_params[0], (पूर्णाङ्क) गाँठ.dst()->ne[1]);
        } अन्यथा यदि ((गाँठ.opcode == HTP_OP_FENCE || गाँठ.opcode == HTP_OP_CPY_FENCE) && गाँठ.गाँठ) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) गाँठ.गाँठ->op_params[0]);
        } अन्यथा यदि (गाँठ.opcode == HTP_OP_ALLREDUCE && गाँठ.गाँठ) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) गाँठ.गाँठ->op_params[0], (uint32_t) गाँठ.गाँठ->op_params[1]);
        } अन्यथा यदि (गाँठ.opcode == HTP_OP_SSM_CONV) {
            ध्रुव स्वतः * kparams = (ध्रुव संरचना htp_ssm_conv_kernel_params *) गाँठ.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (पूर्णाङ्क) kparams->vtcm_size);
        } अन्यथा यदि (गाँठ.opcode == HTP_OP_SOFTMAX) {
            ध्रुव स्वतः * kparams = (ध्रुव संरचना htp_softmax_kernel_params *) गाँठ.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (पूर्णाङ्क) kparams->kernel_id, (पूर्णाङ्क) kparams->n_threads, (पूर्णाङ्क) kparams->vtcm_size);
        } अन्यथा यदि (गाँठ.opcode == HTP_OP_GATED_DELTA_NET) {
            ध्रुव स्वतः * kparams = (ध्रुव संरचना htp_gdn_kernel_params *) गाँठ.kernel_params;
            ध्रुव वर्ण * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (अचिन्हित पूर्णाङ्क) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } अन्यथा यदि (गाँठ.opcode == HTP_OP_MUL || गाँठ.opcode == HTP_OP_ADD || गाँठ.opcode == HTP_OP_ADD_ID ||
                   गाँठ.opcode == HTP_OP_SUB || गाँठ.opcode == HTP_OP_DIV) {
            ध्रुव स्वतः * kparams = (ध्रुव संरचना htp_द्विआधारी_kernel_params *) गाँठ.kernel_params;
            snprintf(str, max_size, "vtcm %u", (अचिन्हित पूर्णाङ्क) kparams->vtcm_size);
        } अन्यथा यदि (गाँठ.opcode == HTP_OP_ARGSORT || गाँठ.opcode == HTP_OP_शीर्ष_K) {
            ध्रुव स्वतः * kparams = (ध्रुव संरचना htp_sort_kernel_params *) गाँठ.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     गाँठ.opcode == HTP_OP_शीर्ष_K ? "top_k" : "argsort",
                     (पूर्णाङ्क) kparams->n_threads, (पूर्णाङ्क) kparams->n_chunks,
                     (पूर्णाङ्क) kparams->chunk_elems, (पूर्णाङ्क) kparams->vtcm_size);
        } अन्यथा यदि (गाँठ.opcode == HTP_OP_GET_ROWS) {
            ध्रुव स्वतः * kparams = (ध्रुव संरचना htp_get_rows_kernel_params *) गाँठ.kernel_params;
            ध्रुव वर्ण * ktype_str = "unknown";
            विकल्प (kparams->kernel_type) {
                पक्ष HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; भङ्ग;
                पक्ष HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; भङ्ग;
                पक्ष HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; भङ्ग;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (अचिन्हित पूर्णाङ्क) kparams->vtcm_size);
        } अन्यथा {
            snprintf(str, max_size, "----");
        }

    }

    शून्य format(ध्रुव htp_opnode & गाँठ) {
        format_op_dims(dims, परिमाण(dims), गाँठ);
        format_op_strides(strides, परिमाण(strides), गाँठ);
        format_op_types(types, परिमाण(types), गाँठ);
        format_op_buffs(buffs, परिमाण(buffs), गाँठ);
        format_op_names(names, परिमाण(names), गाँठ);
        format_kernel_params(kparams, परिमाण(kparams), गाँठ);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ध्रुव htp_opnode & गाँठ) { format(गाँठ); }
};

#endif // HTP_OPNODE_H
