#यदि_अपरिभाषितम् HTP_OPNODE_H
#परिभाषय HTP_OPNODE_H

#परिभाषय GGML_COMMON_IMPL_CPP
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
    ggml_tensor * ग्रन्थिः   { nullptr };
    htp_op_संहिता   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::सदिशम्<ggml_tensor *>                fused;
    std::सदिशम्<std::shared_ptr<ggml_tensor>> dummy;

    std::सदिशम्<ध्रुव ggml_tensor *> inputs;
    std::सदिशम्<ध्रुव ggml_tensor *> outputs;
    std::string                      name;

    पूर्णाङ्क n_सक्रिय_src(ध्रुव ggml_tensor * t) ध्रुव {
        यदि (!t) प्रत्यागम 0;
        प्रति (पूर्णाङ्क i = GGML_MAX_SRC - 1; i >= 0; i--) {
            यदि (t->src[i]) {
                प्रत्यागम i + 1;
            }
        }
        प्रत्यागम 0;
    }

    शून्य init(ggml_tensor * ग्रन्थिः) {
        this->ग्रन्थिः = ग्रन्थिः;
        यदि (this->ग्रन्थिः) {
            यदि (this->ग्रन्थिः->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->ग्रन्थिः, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.प्रक्षिप_पृष्ठभागः(this->ग्रन्थिः->src[1]);
            } अन्यथा {
                this->name = ggml_op_desc(this->ग्रन्थिः);

                // Build inputs (preserving optional nullptrs)
                पूर्णाङ्क n_inputs = n_सक्रिय_src(this->ग्रन्थिः);
                this->inputs.resize(n_inputs, nullptr);
                प्रति (पूर्णाङ्क i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->ग्रन्थिः->src[i];
                }
            }

            // Build outputs
            this->outputs.प्रक्षिप_पृष्ठभागः(this->dst());
        }
    }

    htp_opnode(htp_op_संहिता opcode = HTP_OP_INVALID, ggml_tensor * ग्रन्थिः = nullptr) : opcode(opcode) {
        init(ग्रन्थिः);
    }

    ggml_op             op()   ध्रुव { प्रत्यागम ग्रन्थिः->op; }
    ध्रुव ggml_tensor * src0() ध्रुव { प्रत्यागम ग्रन्थिः->src[0]; }
    ध्रुव ggml_tensor * src1() ध्रुव { प्रत्यागम ग्रन्थिः->src[1]; }
    ध्रुव ggml_tensor * dst()  ध्रुव { प्रत्यागम outputs.रिक्तानि() ? ग्रन्थिः : outputs.पृष्ठभागः(); }

    ggml_tensor * सङ्कलय_dummy(ध्रुव ggml_tensor & t) {
        dummy.प्रक्षिप_पृष्ठभागः(std::make_shared<ggml_tensor>(t));
        प्रत्यागम dummy.पृष्ठभागः().get();
    }

    शून्य सङ्कलय_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.प्रक्षिप_पृष्ठभागः(t);

        name += "+";
        name += ggml_op_desc(t);

        यदि (extra_dst) {
            outputs.प्रक्षिप_पृष्ठभागः(t);
        } अन्यथा {
            outputs.clear();
            outputs.प्रक्षिप_पृष्ठभागः(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.अन्तः_2(), t), inputs.अन्तः_2());

        // Append new inputs from t, preserving middle nullptrs
        पूर्णाङ्क n_inputs = n_सक्रिय_src(t);
        प्रति (पूर्णाङ्क i = 0; i < n_inputs; i++) {
            ध्रुव स्वचालित * src = t->src[i];
            यदि (!src) {
                inputs.प्रक्षिप_पृष्ठभागः(nullptr);
            } अन्यथा यदि (src != ग्रन्थिः &&
                       std::find(fused.begin(), fused.अन्तः_2(), src) == fused.अन्तः_2() &&
                       std::find(inputs.begin(), inputs.अन्तः_2(), src) == inputs.अन्तः_2()) {
                inputs.प्रक्षिप_पृष्ठभागः(src);
            }
        }
    }

    ध्रुव std::सदिशम्<ध्रुव ggml_tensor *> & get_inputs() ध्रुव {
        प्रत्यागम inputs;
    }

    ध्रुव std::सदिशम्<ध्रुव ggml_tensor *> & get_outputs() ध्रुव {
        प्रत्यागम outputs;
    }

    std::string op_name() ध्रुव {
        प्रत्यागम name;
    }

    bool is_रिक्तानि() ध्रुव {
        प्रत्यागम ggml_op_is_रिक्तानि(ग्रन्थिः->op);
    }

    bool stackable() ध्रुव {
        विकल्प (this->op()) {
            पक्ष GGML_OP_MUL_MAT:
                यदि (ggml_get_op_params_i32(this->ग्रन्थिः, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    प्रत्यागम false;
                }
                प्रत्यागम ggml_is_quantized(this->src0()->type);
            पक्ष GGML_OP_MUL_MAT_ID:
                प्रत्यागम ggml_is_quantized(this->src0()->type);
            स्वभाव:
                प्रत्यागम false;
        }
    }

    bool same_निवेशः(ध्रुव htp_opnode& n) ध्रुव {
        प्रत्यागम n.src1() == this->src1();
    }
};

संरचना htp_opformat {
    वर्ण strides[64 * GGML_MAX_SRC];
    वर्ण dims[64 * GGML_MAX_SRC];
    वर्ण types[16 * GGML_MAX_SRC];
    वर्ण buffs[64 * GGML_MAX_SRC];
    वर्ण नामानि[64 * GGML_MAX_SRC];
    वर्ण kparams[128];

    पूर्णाङ्क format_tensor_dims(वर्ण * str, size_t max_आकार, ध्रुव संरचना ggml_tensor * t) {
        यदि (!t) {
            प्रत्यागम snprintf(str, max_आकार, "NONE");
        }
        यदि (t->ne[2] == 1 && t->ne[3] == 1) {
            प्रत्यागम snprintf(str, max_आकार, "%d:%d", (पूर्णाङ्क) t->ne[0], (पूर्णाङ्क) t->ne[1]);
        } अन्यथा {
            प्रत्यागम snprintf(str, max_आकार, "%d:%d:%d:%d", (पूर्णाङ्क) t->ne[0], (पूर्णाङ्क) t->ne[1], (पूर्णाङ्क) t->ne[2], (पूर्णाङ्क) t->ne[3]);
        }
    }

    शून्य format_op_dims(वर्ण * str, size_t max_आकार, ध्रुव htp_opnode & ग्रन्थिः) {
        वर्ण * p = str;
        वर्ण * p_अन्तः = str + max_आकार;
        स्वचालित inputs = ग्रन्थिः.get_inputs();

        यदि (!inputs.रिक्तानि()) {
            p += std::min((size_t)format_tensor_dims(p, p_अन्तः - p, inputs[0]), (size_t)(p_अन्तः - p));

            प्रति (size_t i = 1; i < inputs.आकार(); i++) {
                यदि (p < p_अन्तः) {
                    p += std::min((size_t)snprintf(p, p_अन्तः - p, " x "), (size_t)(p_अन्तः - p));
                }
                यदि (p < p_अन्तः) {
                    p += std::min((size_t)format_tensor_dims(p, p_अन्तः - p, inputs[i]), (size_t)(p_अन्तः - p));
                }
            }

            यदि (p < p_अन्तः) {
                p += std::min((size_t)snprintf(p, p_अन्तः - p, " -> "), (size_t)(p_अन्तः - p));
            }
        }

        वर्ण self[64];
        format_tensor_dims(self, परिमाण(self), ग्रन्थिः.dst());
        यदि (p < p_अन्तः) {
            p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", self), (size_t)(p_अन्तः - p));
        }
    }

    पूर्णाङ्क format_tensor_strides(वर्ण * str, size_t max_आकार, ध्रुव संरचना ggml_tensor * t) {
        यदि (!t) {
            प्रत्यागम snprintf(str, max_आकार, "NONE");
        }
        ध्रुव वर्ण * c = ggml_is_contiguous(t) ? "" : "!";

        यदि (t->ne[2] == 1 && t->ne[3] == 1) {
            प्रत्यागम snprintf(str, max_आकार, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } अन्यथा {
            प्रत्यागम snprintf(str, max_आकार, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    शून्य format_op_strides(वर्ण * str, size_t max_आकार, ध्रुव htp_opnode & ग्रन्थिः) {
        वर्ण * p = str;
        वर्ण * p_अन्तः = str + max_आकार;
        स्वचालित inputs = ग्रन्थिः.get_inputs();

        यदि (!inputs.रिक्तानि()) {
            p += std::min((size_t)format_tensor_strides(p, p_अन्तः - p, inputs[0]), (size_t)(p_अन्तः - p));

            प्रति (size_t i = 1; i < inputs.आकार(); i++) {
                यदि (p < p_अन्तः) {
                    p += std::min((size_t)snprintf(p, p_अन्तः - p, " x "), (size_t)(p_अन्तः - p));
                }
                यदि (p < p_अन्तः) {
                    p += std::min((size_t)format_tensor_strides(p, p_अन्तः - p, inputs[i]), (size_t)(p_अन्तः - p));
                }
            }

            यदि (p < p_अन्तः) {
                p += std::min((size_t)snprintf(p, p_अन्तः - p, " -> "), (size_t)(p_अन्तः - p));
            }
        }

        वर्ण self[64];
        format_tensor_strides(self, परिमाण(self), ग्रन्थिः.dst());
        यदि (p < p_अन्तः) {
            p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", self), (size_t)(p_अन्तः - p));
        }
    }

    शून्य format_op_types(वर्ण * str, size_t max_आकार, ध्रुव htp_opnode & ग्रन्थिः) {
        वर्ण * p = str;
        वर्ण * p_अन्तः = str + max_आकार;
        स्वचालित inputs = ग्रन्थिः.get_inputs();

        यदि (!inputs.रिक्तानि()) {
            यदि (p < p_अन्तः) {
                p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_अन्तः - p));
            }

            प्रति (size_t i = 1; i < inputs.आकार(); i++) {
                यदि (p < p_अन्तः) {
                    p += std::min((size_t)snprintf(p, p_अन्तः - p, " x "), (size_t)(p_अन्तः - p));
                }
                यदि (p < p_अन्तः) {
                    p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_अन्तः - p));
                }
            }

            यदि (p < p_अन्तः) {
                p += std::min((size_t)snprintf(p, p_अन्तः - p, " -> "), (size_t)(p_अन्तः - p));
            }
        }

        यदि (p < p_अन्तः) {
            p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", ggml_type_name(ग्रन्थिः.dst()->type)), (size_t)(p_अन्तः - p));
        }
    }

    ध्रुव वर्ण * tensor_buff_name(ध्रुव संरचना ggml_tensor * t) {
        यदि (t && t->बफर) {
            प्रत्यागम ggml_backend_बफर_name(t->बफर);
        }
        प्रत्यागम "NONE";
    }

    शून्य format_op_buffs(वर्ण * str, size_t max_आकार, ध्रुव htp_opnode & ग्रन्थिः) {
        वर्ण * p = str;
        वर्ण * p_अन्तः = str + max_आकार;
        स्वचालित inputs = ग्रन्थिः.get_inputs();

        यदि (!inputs.रिक्तानि()) {
            यदि (p < p_अन्तः) {
                p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_अन्तः - p));
            }

            प्रति (size_t i = 1; i < inputs.आकार(); i++) {
                यदि (p < p_अन्तः) {
                    p += std::min((size_t)snprintf(p, p_अन्तः - p, " x "), (size_t)(p_अन्तः - p));
                }
                यदि (p < p_अन्तः) {
                    p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_अन्तः - p));
                }
            }

            यदि (p < p_अन्तः) {
                p += std::min((size_t)snprintf(p, p_अन्तः - p, " -> "), (size_t)(p_अन्तः - p));
            }
        }

        यदि (p < p_अन्तः) {
            p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", tensor_buff_name(ग्रन्थिः.dst())), (size_t)(p_अन्तः - p));
        }
    }

    शून्य format_op_नामानि(वर्ण * str, size_t max_आकार, ध्रुव htp_opnode & ग्रन्थिः) {
        वर्ण * p = str;
        वर्ण * p_अन्तः = str + max_आकार;
        स्वचालित inputs = ग्रन्थिः.get_inputs();

        यदि (!inputs.रिक्तानि()) {
            यदि (p < p_अन्तः) {
                p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_अन्तः - p));
            }

            प्रति (size_t i = 1; i < inputs.आकार(); i++) {
                यदि (p < p_अन्तः) {
                    p += std::min((size_t)snprintf(p, p_अन्तः - p, " x "), (size_t)(p_अन्तः - p));
                }
                यदि (p < p_अन्तः) {
                    p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_अन्तः - p));
                }
            }

            यदि (p < p_अन्तः) {
                p += std::min((size_t)snprintf(p, p_अन्तः - p, " -> "), (size_t)(p_अन्तः - p));
            }
        }

        यदि (p < p_अन्तः) {
            p += std::min((size_t)snprintf(p, p_अन्तः - p, "%s", ग्रन्थिः.dst()->name), (size_t)(p_अन्तः - p));
        }
    }
    शून्य format_kernel_params(वर्ण * str, size_t max_आकार, ध्रुव htp_opnode & ग्रन्थिः) {
        यदि (ग्रन्थिः.opcode == HTP_OP_MUL_MAT || ग्रन्थिः.opcode == HTP_OP_MUL_MAT_ID ||
            ग्रन्थिः.opcode == HTP_OP_MUL_MAT_NX || ग्रन्थिः.opcode == HTP_OP_MUL_MAT_ID_NX ||
            ग्रन्थिः.opcode == HTP_OP_MUL_MAT_सङ्कलय) {
            ध्रुव स्वचालित * kparams = (ध्रुव संरचना htp_mm_kernel_params *) ग्रन्थिः.kernel_params;
            ध्रुव वर्ण * मार्गः = "unknown";
            int32_t type = kparams->kernel_type;
            यदि (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                मार्गः = (kparams->tile_आकार > 0) ? "hmx-tiled" : "hmx-flat";
            } अन्यथा यदि (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                मार्गः = "hvx-flat";
            } अन्यथा यदि (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_खण्डम्) {
                मार्गः = "hvx-tiled";
            }
            snprintf(str, max_आकार, "%s vtcm %d", मार्गः, (पूर्णाङ्क) kparams->vtcm_आकार);
        } अन्यथा यदि (ग्रन्थिः.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ध्रुव स्वचालित * kparams = (ध्रुव संरचना htp_fa_kernel_params *) ग्रन्थिः.kernel_params;
            ध्रुव वर्ण * मार्गः = "unknown";
            int32_t type = kparams->kernel_type;
            यदि (type == HTP_FA_KERNEL_HMX) {
                मार्गः = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } अन्यथा यदि (type == HTP_FA_KERNEL_HVX) {
                मार्गः = "hvx";
            }
            snprintf(str, max_आकार, "%s vtcm %d", मार्गः, (पूर्णाङ्क) kparams->vtcm_आकार);
        } अन्यथा यदि (htp_op_is_unary(ग्रन्थिः.opcode)) {
            ध्रुव स्वचालित * kparams = (ध्रुव संरचना htp_unary_kernel_params *) ग्रन्थिः.kernel_params;
            snprintf(str, max_आकार, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (पूर्णाङ्क) kparams->vtcm_आकार);
        } अन्यथा यदि (ग्रन्थिः.opcode == HTP_OP_MDEV_गणः && ग्रन्थिः.ग्रन्थिः) {
            snprintf(str, max_आकार, "idx %d count %d", (पूर्णाङ्क) ग्रन्थिः.ग्रन्थिः->op_params[0], (पूर्णाङ्क) ग्रन्थिः.dst()->ne[1]);
        } अन्यथा यदि ((ग्रन्थिः.opcode == HTP_OP_FENCE || ग्रन्थिः.opcode == HTP_OP_CPY_FENCE) && ग्रन्थिः.ग्रन्थिः) {
            snprintf(str, max_आकार, "seq 0x%x", (uint32_t) ग्रन्थिः.ग्रन्थिः->op_params[0]);
        } अन्यथा यदि (ग्रन्थिः.opcode == HTP_OP_ALLREDUCE && ग्रन्थिः.ग्रन्थिः) {
            snprintf(str, max_आकार, "seq 0x%x -> 0x%x", (uint32_t) ग्रन्थिः.ग्रन्थिः->op_params[0], (uint32_t) ग्रन्थिः.ग्रन्थिः->op_params[1]);
        } अन्यथा यदि (ग्रन्थिः.opcode == HTP_OP_SSM_CONV) {
            ध्रुव स्वचालित * kparams = (ध्रुव संरचना htp_ssm_conv_kernel_params *) ग्रन्थिः.kernel_params;
            snprintf(str, max_आकार, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (पूर्णाङ्क) kparams->vtcm_आकार);
        } अन्यथा यदि (ग्रन्थिः.opcode == HTP_OP_SOFTMAX) {
            ध्रुव स्वचालित * kparams = (ध्रुव संरचना htp_softmax_kernel_params *) ग्रन्थिः.kernel_params;
            snprintf(str, max_आकार, "k%d nth %d vtcm %d", (पूर्णाङ्क) kparams->kernel_id, (पूर्णाङ्क) kparams->n_तन्तवः, (पूर्णाङ्क) kparams->vtcm_आकार);
        } अन्यथा यदि (ग्रन्थिः.opcode == HTP_OP_GATED_DELTA_NET) {
            ध्रुव स्वचालित * kparams = (ध्रुव संरचना htp_gdn_kernel_params *) ग्रन्थिः.kernel_params;
            ध्रुव वर्ण * मार्गः = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_आकार, "%s-%s vtcm %u", मार्गः, kparams->kda ? "kda" : "scalar",
                     (अचिह्नित पूर्णाङ्क) (kparams->vtcm_आकार ? kparams->vtcm_आकार : kparams->vtcm_प्रत्येकम्_thread * kparams->n_तन्तवः));
        } अन्यथा यदि (ग्रन्थिः.opcode == HTP_OP_MUL || ग्रन्थिः.opcode == HTP_OP_सङ्कलय || ग्रन्थिः.opcode == HTP_OP_सङ्कलय_ID ||
                   ग्रन्थिः.opcode == HTP_OP_SUB || ग्रन्थिः.opcode == HTP_OP_DIV) {
            ध्रुव स्वचालित * kparams = (ध्रुव संरचना htp_द्विमान_kernel_params *) ग्रन्थिः.kernel_params;
            snprintf(str, max_आकार, "vtcm %u", (अचिह्नित पूर्णाङ्क) kparams->vtcm_आकार);
        } अन्यथा यदि (ग्रन्थिः.opcode == HTP_OP_ARGSORT || ग्रन्थिः.opcode == HTP_OP_शिखरम्_K) {
            ध्रुव स्वचालित * kparams = (ध्रुव संरचना htp_क्रमय_kernel_params *) ग्रन्थिः.kernel_params;
            snprintf(str, max_आकार, "%s nth %d nchk %d chk %d vtcm %d",
                     ग्रन्थिः.opcode == HTP_OP_शिखरम्_K ? "top_k" : "argsort",
                     (पूर्णाङ्क) kparams->n_तन्तवः, (पूर्णाङ्क) kparams->n_chunks,
                     (पूर्णाङ्क) kparams->chunk_elems, (पूर्णाङ्क) kparams->vtcm_आकार);
        } अन्यथा यदि (ग्रन्थिः.opcode == HTP_OP_GET_पङ्क्तिकाः) {
            ध्रुव स्वचालित * kparams = (ध्रुव संरचना htp_get_पङ्क्तिकाः_kernel_params *) ग्रन्थिः.kernel_params;
            ध्रुव वर्ण * ktype_str = "unknown";
            विकल्प (kparams->kernel_type) {
                पक्ष HTP_GET_पङ्क्तिकाः_KERNEL_SAMETYPE: ktype_str = "sametype"; भङ्ग;
                पक्ष HTP_GET_पङ्क्तिकाः_KERNEL_TILED:    ktype_str = "tiled"; भङ्ग;
                पक्ष HTP_GET_पङ्क्तिकाः_KERNEL_FLAT:     ktype_str = "flat"; भङ्ग;
            }
            snprintf(str, max_आकार, "%s%s vtcm %u", ktype_str, kparams->n_तन्तवः > 1 ? "-multi" : "", (अचिह्नित पूर्णाङ्क) kparams->vtcm_आकार);
        } अन्यथा {
            snprintf(str, max_आकार, "----");
        }

    }

    शून्य format(ध्रुव htp_opnode & ग्रन्थिः) {
        format_op_dims(dims, परिमाण(dims), ग्रन्थिः);
        format_op_strides(strides, परिमाण(strides), ग्रन्थिः);
        format_op_types(types, परिमाण(types), ग्रन्थिः);
        format_op_buffs(buffs, परिमाण(buffs), ग्रन्थिः);
        format_op_नामानि(नामानि, परिमाण(नामानि), ग्रन्थिः);
        format_kernel_params(kparams, परिमाण(kparams), ग्रन्थिः);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        नामानि[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ध्रुव htp_opnode & ग्रन्थिः) { format(ग्रन्थिः); }
};

#यदि_अन्तः // HTP_OPNODE_H
