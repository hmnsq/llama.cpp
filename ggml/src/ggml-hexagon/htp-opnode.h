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
    ggml_tensor * गाठ   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<स्थिरांक ggml_tensor *> inputs;
    std::vector<स्थिरांक ggml_tensor *> outputs;
    std::string                      name;

    पूर्णांक n_active_src(स्थिरांक ggml_tensor * t) स्थिरांक {
        जर (!t) परत 0;
        साठी (पूर्णांक i = GGML_MAX_SRC - 1; i >= 0; i--) {
            जर (t->src[i]) {
                परत i + 1;
            }
        }
        परत 0;
    }

    शून्य init(ggml_tensor * गाठ) {
        this->गाठ = गाठ;
        जर (this->गाठ) {
            जर (this->गाठ->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->गाठ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ढकला_back(this->गाठ->src[1]);
            } अन्यथा {
                this->name = ggml_op_desc(this->गाठ);

                // Build inputs (preserving optional nullptrs)
                पूर्णांक n_inputs = n_active_src(this->गाठ);
                this->inputs.resize(n_inputs, nullptr);
                साठी (पूर्णांक i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->गाठ->src[i];
                }
            }

            // Build outputs
            this->outputs.ढकला_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * गाठ = nullptr) : opcode(opcode) {
        init(गाठ);
    }

    ggml_op             op()   स्थिरांक { परत गाठ->op; }
    स्थिरांक ggml_tensor * src0() स्थिरांक { परत गाठ->src[0]; }
    स्थिरांक ggml_tensor * src1() स्थिरांक { परत गाठ->src[1]; }
    स्थिरांक ggml_tensor * dst()  स्थिरांक { परत outputs.empty() ? गाठ : outputs.back(); }

    ggml_tensor * add_dummy(स्थिरांक ggml_tensor & t) {
        dummy.ढकला_back(std::make_shared<ggml_tensor>(t));
        परत dummy.back().get();
    }

    शून्य add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ढकला_back(t);

        name += "+";
        name += ggml_op_desc(t);

        जर (extra_dst) {
            outputs.ढकला_back(t);
        } अन्यथा {
            outputs.clear();
            outputs.ढकला_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::हटाओ(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        पूर्णांक n_inputs = n_active_src(t);
        साठी (पूर्णांक i = 0; i < n_inputs; i++) {
            स्थिरांक स्वतः * src = t->src[i];
            जर (!src) {
                inputs.ढकला_back(nullptr);
            } अन्यथा जर (src != गाठ &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.ढकला_back(src);
            }
        }
    }

    स्थिरांक std::vector<स्थिरांक ggml_tensor *> & get_inputs() स्थिरांक {
        परत inputs;
    }

    स्थिरांक std::vector<स्थिरांक ggml_tensor *> & get_outputs() स्थिरांक {
        परत outputs;
    }

    std::string op_name() स्थिरांक {
        परत name;
    }

    bool is_empty() स्थिरांक {
        परत ggml_op_is_empty(गाठ->op);
    }

    bool stackable() स्थिरांक {
        निवड (this->op()) {
            स्थिती GGML_OP_MUL_MAT:
                जर (ggml_get_op_params_i32(this->गाठ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    परत false;
                }
                परत ggml_is_quantized(this->src0()->type);
            स्थिती GGML_OP_MUL_MAT_ID:
                परत ggml_is_quantized(this->src0()->type);
            पूर्वनिर्धारित:
                परत false;
        }
    }

    bool same_आदान(स्थिरांक htp_opnode& n) स्थिरांक {
        परत n.src1() == this->src1();
    }
};

संरचना htp_opformat {
    अक्षर strides[64 * GGML_MAX_SRC];
    अक्षर dims[64 * GGML_MAX_SRC];
    अक्षर types[16 * GGML_MAX_SRC];
    अक्षर buffs[64 * GGML_MAX_SRC];
    अक्षर names[64 * GGML_MAX_SRC];
    अक्षर kparams[128];

    पूर्णांक format_tensor_dims(अक्षर * str, माप_प्रकार max_size, स्थिरांक संरचना ggml_tensor * t) {
        जर (!t) {
            परत snprintf(str, max_size, "NONE");
        }
        जर (t->ne[2] == 1 && t->ne[3] == 1) {
            परत snprintf(str, max_size, "%d:%d", (पूर्णांक) t->ne[0], (पूर्णांक) t->ne[1]);
        } अन्यथा {
            परत snprintf(str, max_size, "%d:%d:%d:%d", (पूर्णांक) t->ne[0], (पूर्णांक) t->ne[1], (पूर्णांक) t->ne[2], (पूर्णांक) t->ne[3]);
        }
    }

    शून्य format_op_dims(अक्षर * str, माप_प्रकार max_size, स्थिरांक htp_opnode & गाठ) {
        अक्षर * p = str;
        अक्षर * p_end = str + max_size;
        स्वतः inputs = गाठ.get_inputs();

        जर (!inputs.empty()) {
            p += std::min((माप_प्रकार)format_tensor_dims(p, p_end - p, inputs[0]), (माप_प्रकार)(p_end - p));

            साठी (माप_प्रकार i = 1; i < inputs.size(); i++) {
                जर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                जर (p < p_end) {
                    p += std::min((माप_प्रकार)format_tensor_dims(p, p_end - p, inputs[i]), (माप_प्रकार)(p_end - p));
                }
            }

            जर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        अक्षर self[64];
        format_tensor_dims(self, आकार(self), गाठ.dst());
        जर (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", self), (माप_प्रकार)(p_end - p));
        }
    }

    पूर्णांक format_tensor_strides(अक्षर * str, माप_प्रकार max_size, स्थिरांक संरचना ggml_tensor * t) {
        जर (!t) {
            परत snprintf(str, max_size, "NONE");
        }
        स्थिरांक अक्षर * c = ggml_is_contiguous(t) ? "" : "!";

        जर (t->ne[2] == 1 && t->ne[3] == 1) {
            परत snprintf(str, max_size, "%zu:%zu%s", (माप_प्रकार) t->nb[0], (माप_प्रकार) t->nb[1], c);
        } अन्यथा {
            परत snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (माप_प्रकार) t->nb[0], (माप_प्रकार) t->nb[1], (माप_प्रकार) t->nb[2], (माप_प्रकार) t->nb[3], c);
        }
    }

    शून्य format_op_strides(अक्षर * str, माप_प्रकार max_size, स्थिरांक htp_opnode & गाठ) {
        अक्षर * p = str;
        अक्षर * p_end = str + max_size;
        स्वतः inputs = गाठ.get_inputs();

        जर (!inputs.empty()) {
            p += std::min((माप_प्रकार)format_tensor_strides(p, p_end - p, inputs[0]), (माप_प्रकार)(p_end - p));

            साठी (माप_प्रकार i = 1; i < inputs.size(); i++) {
                जर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                जर (p < p_end) {
                    p += std::min((माप_प्रकार)format_tensor_strides(p, p_end - p, inputs[i]), (माप_प्रकार)(p_end - p));
                }
            }

            जर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        अक्षर self[64];
        format_tensor_strides(self, आकार(self), गाठ.dst());
        जर (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", self), (माप_प्रकार)(p_end - p));
        }
    }

    शून्य format_op_types(अक्षर * str, माप_प्रकार max_size, स्थिरांक htp_opnode & गाठ) {
        अक्षर * p = str;
        अक्षर * p_end = str + max_size;
        स्वतः inputs = गाठ.get_inputs();

        जर (!inputs.empty()) {
            जर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (माप_प्रकार)(p_end - p));
            }

            साठी (माप_प्रकार i = 1; i < inputs.size(); i++) {
                जर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                जर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (माप_प्रकार)(p_end - p));
                }
            }

            जर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        जर (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", ggml_type_name(गाठ.dst()->type)), (माप_प्रकार)(p_end - p));
        }
    }

    स्थिरांक अक्षर * tensor_buff_name(स्थिरांक संरचना ggml_tensor * t) {
        जर (t && t->buffer) {
            परत ggml_backend_buffer_name(t->buffer);
        }
        परत "NONE";
    }

    शून्य format_op_buffs(अक्षर * str, माप_प्रकार max_size, स्थिरांक htp_opnode & गाठ) {
        अक्षर * p = str;
        अक्षर * p_end = str + max_size;
        स्वतः inputs = गाठ.get_inputs();

        जर (!inputs.empty()) {
            जर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (माप_प्रकार)(p_end - p));
            }

            साठी (माप_प्रकार i = 1; i < inputs.size(); i++) {
                जर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                जर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (माप_प्रकार)(p_end - p));
                }
            }

            जर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        जर (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", tensor_buff_name(गाठ.dst())), (माप_प्रकार)(p_end - p));
        }
    }

    शून्य format_op_names(अक्षर * str, माप_प्रकार max_size, स्थिरांक htp_opnode & गाठ) {
        अक्षर * p = str;
        अक्षर * p_end = str + max_size;
        स्वतः inputs = गाठ.get_inputs();

        जर (!inputs.empty()) {
            जर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (माप_प्रकार)(p_end - p));
            }

            साठी (माप_प्रकार i = 1; i < inputs.size(); i++) {
                जर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, " x "), (माप_प्रकार)(p_end - p));
                }
                जर (p < p_end) {
                    p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (माप_प्रकार)(p_end - p));
                }
            }

            जर (p < p_end) {
                p += std::min((माप_प्रकार)snprintf(p, p_end - p, " -> "), (माप_प्रकार)(p_end - p));
            }
        }

        जर (p < p_end) {
            p += std::min((माप_प्रकार)snprintf(p, p_end - p, "%s", गाठ.dst()->name), (माप_प्रकार)(p_end - p));
        }
    }
    शून्य format_kernel_params(अक्षर * str, माप_प्रकार max_size, स्थिरांक htp_opnode & गाठ) {
        जर (गाठ.opcode == HTP_OP_MUL_MAT || गाठ.opcode == HTP_OP_MUL_MAT_ID ||
            गाठ.opcode == HTP_OP_MUL_MAT_NX || गाठ.opcode == HTP_OP_MUL_MAT_ID_NX ||
            गाठ.opcode == HTP_OP_MUL_MAT_ADD) {
            स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_mm_kernel_params *) गाठ.kernel_params;
            स्थिरांक अक्षर * path = "unknown";
            int32_t type = kparams->kernel_type;
            जर (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } अन्यथा जर (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } अन्यथा जर (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (पूर्णांक) kparams->vtcm_size);
        } अन्यथा जर (गाठ.opcode == HTP_OP_FLASH_ATTN_EXT) {
            स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_fa_kernel_params *) गाठ.kernel_params;
            स्थिरांक अक्षर * path = "unknown";
            int32_t type = kparams->kernel_type;
            जर (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } अन्यथा जर (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (पूर्णांक) kparams->vtcm_size);
        } अन्यथा जर (htp_op_is_unary(गाठ.opcode)) {
            स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_unary_kernel_params *) गाठ.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (पूर्णांक) kparams->vtcm_size);
        } अन्यथा जर (गाठ.opcode == HTP_OP_MDEV_GROUP && गाठ.गाठ) {
            snprintf(str, max_size, "idx %d count %d", (पूर्णांक) गाठ.गाठ->op_params[0], (पूर्णांक) गाठ.dst()->ne[1]);
        } अन्यथा जर ((गाठ.opcode == HTP_OP_FENCE || गाठ.opcode == HTP_OP_CPY_FENCE) && गाठ.गाठ) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) गाठ.गाठ->op_params[0]);
        } अन्यथा जर (गाठ.opcode == HTP_OP_ALLREDUCE && गाठ.गाठ) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) गाठ.गाठ->op_params[0], (uint32_t) गाठ.गाठ->op_params[1]);
        } अन्यथा जर (गाठ.opcode == HTP_OP_SSM_CONV) {
            स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_ssm_conv_kernel_params *) गाठ.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (पूर्णांक) kparams->vtcm_size);
        } अन्यथा जर (गाठ.opcode == HTP_OP_SOFTMAX) {
            स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_softmax_kernel_params *) गाठ.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (पूर्णांक) kparams->kernel_id, (पूर्णांक) kparams->n_threads, (पूर्णांक) kparams->vtcm_size);
        } अन्यथा जर (गाठ.opcode == HTP_OP_GATED_DELTA_NET) {
            स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_gdn_kernel_params *) गाठ.kernel_params;
            स्थिरांक अक्षर * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (अचिन्हित पूर्णांक) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } अन्यथा जर (गाठ.opcode == HTP_OP_MUL || गाठ.opcode == HTP_OP_ADD || गाठ.opcode == HTP_OP_ADD_ID ||
                   गाठ.opcode == HTP_OP_SUB || गाठ.opcode == HTP_OP_DIV) {
            स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_द्विमान_kernel_params *) गाठ.kernel_params;
            snprintf(str, max_size, "vtcm %u", (अचिन्हित पूर्णांक) kparams->vtcm_size);
        } अन्यथा जर (गाठ.opcode == HTP_OP_ARGSORT || गाठ.opcode == HTP_OP_शीर्ष_K) {
            स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_sort_kernel_params *) गाठ.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     गाठ.opcode == HTP_OP_शीर्ष_K ? "top_k" : "argsort",
                     (पूर्णांक) kparams->n_threads, (पूर्णांक) kparams->n_chunks,
                     (पूर्णांक) kparams->chunk_elems, (पूर्णांक) kparams->vtcm_size);
        } अन्यथा जर (गाठ.opcode == HTP_OP_GET_ROWS) {
            स्थिरांक स्वतः * kparams = (स्थिरांक संरचना htp_get_rows_kernel_params *) गाठ.kernel_params;
            स्थिरांक अक्षर * ktype_str = "unknown";
            निवड (kparams->kernel_type) {
                स्थिती HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; थांबा;
                स्थिती HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; थांबा;
                स्थिती HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; थांबा;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (अचिन्हित पूर्णांक) kparams->vtcm_size);
        } अन्यथा {
            snprintf(str, max_size, "----");
        }

    }

    शून्य format(स्थिरांक htp_opnode & गाठ) {
        format_op_dims(dims, आकार(dims), गाठ);
        format_op_strides(strides, आकार(strides), गाठ);
        format_op_types(types, आकार(types), गाठ);
        format_op_buffs(buffs, आकार(buffs), गाठ);
        format_op_names(names, आकार(names), गाठ);
        format_kernel_params(kparams, आकार(kparams), गाठ);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(स्थिरांक htp_opnode & गाठ) { format(गाठ); }
};

#endif // HTP_OPNODE_H
