#अगर_अघोषित HTP_OPNODE_H
#घोषणा HTP_OPNODE_H

#घोषणा GGML_COMMON_IMPL_CPP
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-common.h"

#समावेश <algorithm>
#समावेश <string>
#समावेश <vector>
#समावेश <memory>
#समावेश <stdio.h>
#समावेश "htp-ops.h"
#समावेश "htp/matmul-ops.h"
#समावेश "htp/flash-attn-ops.h"
#समावेश "htp/unary-ops.h"
#समावेश "htp/binary-ops.h"
#समावेश "htp/allreduce-ops.h"
#समावेश "htp/ssm-conv.h"
#समावेश "htp/gated-delta-net-ops.h"
#समावेश "htp/softmax-ops.h"
#समावेश "htp/argsort-ops.h"
#समावेश "htp/get-rows-ops.h"

काष्ठा htp_opnode {
    ggml_tensor * गाँठ   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<स्थिर ggml_tensor *> inputs;
    std::vector<स्थिर ggml_tensor *> outputs;
    std::string                      name;

    पूर्णांक n_active_src(स्थिर ggml_tensor * t) स्थिर {
        अगर (!t) वापस 0;
        क्रम (पूर्णांक i = GGML_MAX_SRC - 1; i >= 0; i--) {
            अगर (t->src[i]) {
                वापस i + 1;
            }
        }
        वापस 0;
    }

    व्योम init(ggml_tensor * गाँठ) {
        this->गाँठ = गाँठ;
        अगर (this->गाँठ) {
            अगर (this->गाँठ->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->गाँठ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.धकेलो_back(this->गाँठ->src[1]);
            } अन्यथा {
                this->name = ggml_op_desc(this->गाँठ);

                // Build inputs (preserving optional nullptrs)
                पूर्णांक n_inputs = n_active_src(this->गाँठ);
                this->inputs.resize(n_inputs, nullptr);
                क्रम (पूर्णांक i = 0; i < n_inputs; i++) {
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

    ggml_op             op()   स्थिर { वापस गाँठ->op; }
    स्थिर ggml_tensor * src0() स्थिर { वापस गाँठ->src[0]; }
    स्थिर ggml_tensor * src1() स्थिर { वापस गाँठ->src[1]; }
    स्थिर ggml_tensor * dst()  स्थिर { वापस outputs.empty() ? गाँठ : outputs.back(); }

    ggml_tensor * add_dummy(स्थिर ggml_tensor & t) {
        dummy.धकेलो_back(std::make_shared<ggml_tensor>(t));
        वापस dummy.back().get();
    }

    व्योम add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.धकेलो_back(t);

        name += "+";
        name += ggml_op_desc(t);

        अगर (extra_dst) {
            outputs.धकेलो_back(t);
        } अन्यथा {
            outputs.clear();
            outputs.धकेलो_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::हटाओ(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        पूर्णांक n_inputs = n_active_src(t);
        क्रम (पूर्णांक i = 0; i < n_inputs; i++) {
            स्थिर स्वतः * src = t->src[i];
            अगर (!src) {
                inputs.धकेलो_back(nullptr);
            } अन्यथा अगर (src != गाँठ &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.धकेलो_back(src);
            }
        }
    }

    स्थिर std::vector<स्थिर ggml_tensor *> & get_inputs() स्थिर {
        वापस inputs;
    }

    स्थिर std::vector<स्थिर ggml_tensor *> & get_outputs() स्थिर {
        वापस outputs;
    }

    std::string op_name() स्थिर {
        वापस name;
    }

    bool is_empty() स्थिर {
        वापस ggml_op_is_empty(गाँठ->op);
    }

    bool stackable() स्थिर {
        चयन (this->op()) {
            हाल GGML_OP_MUL_MAT:
                अगर (ggml_get_op_params_i32(this->गाँठ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    वापस false;
                }
                वापस ggml_is_quantized(this->src0()->type);
            हाल GGML_OP_MUL_MAT_ID:
                वापस ggml_is_quantized(this->src0()->type);
            शेष:
                वापस false;
        }
    }

    bool same_निवेश(स्थिर htp_opnode& n) स्थिर {
        वापस n.src1() == this->src1();
    }
};

काष्ठा htp_opformat {
    अक्षर strides[64 * GGML_MAX_SRC];
    अक्षर dims[64 * GGML_MAX_SRC];
    अक्षर types[16 * GGML_MAX_SRC];
    अक्षर buffs[64 * GGML_MAX_SRC];
    अक्षर names[64 * GGML_MAX_SRC];
    अक्षर kparams[128];

    पूर्णांक format_tensor_dims(अक्षर * str, माप_प्रकार max_आकार, स्थिर काष्ठा ggml_tensor * t) {
        अगर (!t) {
            वापस snprintf(str, max_आकार, "NONE");
        }
        अगर (t->ne[2] == 1 && t->ne[3] == 1) {
            वापस snprintf(str, max_आकार, "%d:%d", (पूर्णांक) t->ne[0], (पूर्णांक) t->ne[1]);
        } अन्यथा {
            वापस snprintf(str, max_आकार, "%d:%d:%d:%d", (पूर्णांक) t->ne[0], (पूर्णांक) t->ne[1], (पूर्णांक) t->ne[2], (पूर्णांक) t->ne[3]);
        }
    }

    व्योम format_op_dims(अक्षर * str, माप_प्रकार max_आकार, स्थिर htp_opnode & गाँठ) {
        अक्षर * p = str;
        अक्षर * p_end = str + max_आकार;
        स्वतः inputs = गाँठ.get_inputs();

        अगर (!inputs.empty()) {
            p += std::min((माप_प्रकार)format_tensor_dims(p, p_end - p, inputs[0]), (माप_प्रकार)(p_end - p));

            क्रम (माप_प्रकार i = 1; i < inputs.आकार(); i++) {
                अगर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                अगर (p < p_end) {
                    p += std::min((माप_प्रकार)format_tensor_dims(p, p_end - p, inputs[i]), (माप_प्रकार)(p_end - p));
                }
            }

            अगर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        अक्षर self[64];
        format_tensor_dims(self, माप(self), गाँठ.dst());
        अगर (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", self), (माप_प्रकार)(p_end - p));
        }
    }

    पूर्णांक format_tensor_strides(अक्षर * str, माप_प्रकार max_आकार, स्थिर काष्ठा ggml_tensor * t) {
        अगर (!t) {
            वापस snprintf(str, max_आकार, "NONE");
        }
        स्थिर अक्षर * c = ggml_is_contiguous(t) ? "" : "!";

        अगर (t->ne[2] == 1 && t->ne[3] == 1) {
            वापस snprintf(str, max_आकार, "%zu:%zu%s", (माप_प्रकार) t->nb[0], (माप_प्रकार) t->nb[1], c);
        } अन्यथा {
            वापस snprintf(str, max_आकार, "%zu:%zu:%zu:%zu%s", (माप_प्रकार) t->nb[0], (माप_प्रकार) t->nb[1], (माप_प्रकार) t->nb[2], (माप_प्रकार) t->nb[3], c);
        }
    }

    व्योम format_op_strides(अक्षर * str, माप_प्रकार max_आकार, स्थिर htp_opnode & गाँठ) {
        अक्षर * p = str;
        अक्षर * p_end = str + max_आकार;
        स्वतः inputs = गाँठ.get_inputs();

        अगर (!inputs.empty()) {
            p += std::min((माप_प्रकार)format_tensor_strides(p, p_end - p, inputs[0]), (माप_प्रकार)(p_end - p));

            क्रम (माप_प्रकार i = 1; i < inputs.आकार(); i++) {
                अगर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                अगर (p < p_end) {
                    p += std::min((माप_प्रकार)format_tensor_strides(p, p_end - p, inputs[i]), (माप_प्रकार)(p_end - p));
                }
            }

            अगर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        अक्षर self[64];
        format_tensor_strides(self, माप(self), गाँठ.dst());
        अगर (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", self), (माप_प्रकार)(p_end - p));
        }
    }

    व्योम format_op_types(अक्षर * str, माप_प्रकार max_आकार, स्थिर htp_opnode & गाँठ) {
        अक्षर * p = str;
        अक्षर * p_end = str + max_आकार;
        स्वतः inputs = गाँठ.get_inputs();

        अगर (!inputs.empty()) {
            अगर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (माप_प्रकार)(p_end - p));
            }

            क्रम (माप_प्रकार i = 1; i < inputs.आकार(); i++) {
                अगर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                अगर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (माप_प्रकार)(p_end - p));
                }
            }

            अगर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        अगर (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", ggml_type_name(गाँठ.dst()->type)), (माप_प्रकार)(p_end - p));
        }
    }

    स्थिर अक्षर * tensor_buff_name(स्थिर काष्ठा ggml_tensor * t) {
        अगर (t && t->बफर) {
            वापस ggml_backend_बफर_name(t->बफर);
        }
        वापस "NONE";
    }

    व्योम format_op_buffs(अक्षर * str, माप_प्रकार max_आकार, स्थिर htp_opnode & गाँठ) {
        अक्षर * p = str;
        अक्षर * p_end = str + max_आकार;
        स्वतः inputs = गाँठ.get_inputs();

        अगर (!inputs.empty()) {
            अगर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (माप_प्रकार)(p_end - p));
            }

            क्रम (माप_प्रकार i = 1; i < inputs.आकार(); i++) {
                अगर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                अगर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (माप_प्रकार)(p_end - p));
                }
            }

            अगर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        अगर (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", tensor_buff_name(गाँठ.dst())), (माप_प्रकार)(p_end - p));
        }
    }

    व्योम format_op_names(अक्षर * str, माप_प्रकार max_आकार, स्थिर htp_opnode & गाँठ) {
        अक्षर * p = str;
        अक्षर * p_end = str + max_आकार;
        स्वतः inputs = गाँठ.get_inputs();

        अगर (!inputs.empty()) {
            अगर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (माप_प्रकार)(p_end - p));
            }

            क्रम (माप_प्रकार i = 1; i < inputs.आकार(); i++) {
                अगर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                अगर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (माप_प्रकार)(p_end - p));
                }
            }

            अगर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        अगर (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", गाँठ.dst()->name), (माप_प्रकार)(p_end - p));
        }
    }
    व्योम format_kernel_params(अक्षर * str, माप_प्रकार max_आकार, स्थिर htp_opnode & गाँठ) {
        अगर (गाँठ.opcode == HTP_OP_MUL_MAT || गाँठ.opcode == HTP_OP_MUL_MAT_ID ||
            गाँठ.opcode == HTP_OP_MUL_MAT_NX || गाँठ.opcode == HTP_OP_MUL_MAT_ID_NX ||
            गाँठ.opcode == HTP_OP_MUL_MAT_ADD) {
            स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) गाँठ.kernel_params;
            स्थिर अक्षर * path = "unknown";
            int32_t type = kparams->kernel_type;
            अगर (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_आकार > 0) ? "hmx-tiled" : "hmx-flat";
            } अन्यथा अगर (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } अन्यथा अगर (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_आकार, "%s vtcm %d", path, (पूर्णांक) kparams->vtcm_आकार);
        } अन्यथा अगर (गाँठ.opcode == HTP_OP_FLASH_ATTN_EXT) {
            स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_fa_kernel_params *) गाँठ.kernel_params;
            स्थिर अक्षर * path = "unknown";
            int32_t type = kparams->kernel_type;
            अगर (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } अन्यथा अगर (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_आकार, "%s vtcm %d", path, (पूर्णांक) kparams->vtcm_आकार);
        } अन्यथा अगर (htp_op_is_unary(गाँठ.opcode)) {
            स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_unary_kernel_params *) गाँठ.kernel_params;
            snprintf(str, max_आकार, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (पूर्णांक) kparams->vtcm_आकार);
        } अन्यथा अगर (गाँठ.opcode == HTP_OP_MDEV_GROUP && गाँठ.गाँठ) {
            snprintf(str, max_आकार, "idx %d count %d", (पूर्णांक) गाँठ.गाँठ->op_params[0], (पूर्णांक) गाँठ.dst()->ne[1]);
        } अन्यथा अगर ((गाँठ.opcode == HTP_OP_FENCE || गाँठ.opcode == HTP_OP_CPY_FENCE) && गाँठ.गाँठ) {
            snprintf(str, max_आकार, "seq 0x%x", (uint32_t) गाँठ.गाँठ->op_params[0]);
        } अन्यथा अगर (गाँठ.opcode == HTP_OP_ALLREDUCE && गाँठ.गाँठ) {
            snprintf(str, max_आकार, "seq 0x%x -> 0x%x", (uint32_t) गाँठ.गाँठ->op_params[0], (uint32_t) गाँठ.गाँठ->op_params[1]);
        } अन्यथा अगर (गाँठ.opcode == HTP_OP_SSM_CONV) {
            स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_ssm_conv_kernel_params *) गाँठ.kernel_params;
            snprintf(str, max_आकार, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (पूर्णांक) kparams->vtcm_आकार);
        } अन्यथा अगर (गाँठ.opcode == HTP_OP_SOFTMAX) {
            स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_softmax_kernel_params *) गाँठ.kernel_params;
            snprintf(str, max_आकार, "k%d nth %d vtcm %d", (पूर्णांक) kparams->kernel_id, (पूर्णांक) kparams->n_threads, (पूर्णांक) kparams->vtcm_आकार);
        } अन्यथा अगर (गाँठ.opcode == HTP_OP_GATED_DELTA_NET) {
            स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_gdn_kernel_params *) गाँठ.kernel_params;
            स्थिर अक्षर * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_आकार, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (अचिन्हित पूर्णांक) (kparams->vtcm_आकार ? kparams->vtcm_आकार : kparams->vtcm_per_thread * kparams->n_threads));
        } अन्यथा अगर (गाँठ.opcode == HTP_OP_MUL || गाँठ.opcode == HTP_OP_ADD || गाँठ.opcode == HTP_OP_ADD_ID ||
                   गाँठ.opcode == HTP_OP_SUB || गाँठ.opcode == HTP_OP_DIV) {
            स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_द्विआधारी_kernel_params *) गाँठ.kernel_params;
            snprintf(str, max_आकार, "vtcm %u", (अचिन्हित पूर्णांक) kparams->vtcm_आकार);
        } अन्यथा अगर (गाँठ.opcode == HTP_OP_ARGSORT || गाँठ.opcode == HTP_OP_शीर्ष_K) {
            स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_sort_kernel_params *) गाँठ.kernel_params;
            snprintf(str, max_आकार, "%s nth %d nchk %d chk %d vtcm %d",
                     गाँठ.opcode == HTP_OP_शीर्ष_K ? "top_k" : "argsort",
                     (पूर्णांक) kparams->n_threads, (पूर्णांक) kparams->n_chunks,
                     (पूर्णांक) kparams->chunk_elems, (पूर्णांक) kparams->vtcm_आकार);
        } अन्यथा अगर (गाँठ.opcode == HTP_OP_GET_ROWS) {
            स्थिर स्वतः * kparams = (स्थिर काष्ठा htp_get_rows_kernel_params *) गाँठ.kernel_params;
            स्थिर अक्षर * ktype_str = "unknown";
            चयन (kparams->kernel_type) {
                हाल HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; अवरोध;
                हाल HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; अवरोध;
                हाल HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; अवरोध;
            }
            snprintf(str, max_आकार, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (अचिन्हित पूर्णांक) kparams->vtcm_आकार);
        } अन्यथा {
            snprintf(str, max_आकार, "----");
        }

    }

    व्योम format(स्थिर htp_opnode & गाँठ) {
        format_op_dims(dims, माप(dims), गाँठ);
        format_op_strides(strides, माप(strides), गाँठ);
        format_op_types(types, माप(types), गाँठ);
        format_op_buffs(buffs, माप(buffs), गाँठ);
        format_op_names(names, माप(names), गाँठ);
        format_kernel_params(kparams, माप(kparams), गाँठ);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(स्थिर htp_opnode & गाँठ) { format(गाँठ); }
};

#पूर्ण_अगर // HTP_OPNODE_H
