#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_OPNODE_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_OPNODE_H

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_IMPL_CPP
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

ರಚನೆ htp_opnode {
    ggml_tensor * ಗಂಟು   { nullptr };
    htp_op_ಸಂಕೇತ   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::ಸದಿಶ<ggml_tensor *>                fused;
    std::ಸದಿಶ<std::shared_ptr<ggml_tensor>> dummy;

    std::ಸದಿಶ<ಸ್ಥಿರಾಂಕ ggml_tensor *> inputs;
    std::ಸದಿಶ<ಸ್ಥಿರಾಂಕ ggml_tensor *> outputs;
    std::string                      name;

    ಪೂರ್ಣಾಂಕ n_ಸಕ್ರಿಯ_src(ಸ್ಥಿರಾಂಕ ggml_tensor * t) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ (!t) ಹಿಂತಿರುಗಿ 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = GGML_MAX_SRC - 1; i >= 0; i--) {
            ಆದರೆ (t->src[i]) {
                ಹಿಂತಿರುಗಿ i + 1;
            }
        }
        ಹಿಂತಿರುಗಿ 0;
    }

    ಶೂನ್ಯ init(ggml_tensor * ಗಂಟು) {
        this->ಗಂಟು = ಗಂಟು;
        ಆದರೆ (this->ಗಂಟು) {
            ಆದರೆ (this->ಗಂಟು->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->ಗಂಟು, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ತಳ್ಳು_ಹಿಂಭಾಗ(this->ಗಂಟು->src[1]);
            } ಇಲ್ಲದಿದ್ದರೆ {
                this->name = ggml_op_desc(this->ಗಂಟು);

                // Build inputs (preserving optional nullptrs)
                ಪೂರ್ಣಾಂಕ n_inputs = n_ಸಕ್ರಿಯ_src(this->ಗಂಟು);
                this->inputs.resize(n_inputs, nullptr);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->ಗಂಟು->src[i];
                }
            }

            // Build outputs
            this->outputs.ತಳ್ಳು_ಹಿಂಭಾಗ(this->dst());
        }
    }

    htp_opnode(htp_op_ಸಂಕೇತ opcode = HTP_OP_INVALID, ggml_tensor * ಗಂಟು = nullptr) : opcode(opcode) {
        init(ಗಂಟು);
    }

    ggml_op             op()   ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ಗಂಟು->op; }
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ಗಂಟು->src[0]; }
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ಗಂಟು->src[1]; }
    ಸ್ಥಿರಾಂಕ ggml_tensor * dst()  ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ outputs.ಖಾಲಿಗಳು() ? ಗಂಟು : outputs.ಹಿಂಭಾಗ(); }

    ggml_tensor * ಕೂಡಿಸು_dummy(ಸ್ಥಿರಾಂಕ ggml_tensor & t) {
        dummy.ತಳ್ಳು_ಹಿಂಭಾಗ(std::make_shared<ggml_tensor>(t));
        ಹಿಂತಿರುಗಿ dummy.ಹಿಂಭಾಗ().get();
    }

    ಶೂನ್ಯ ಕೂಡಿಸು_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ತಳ್ಳು_ಹಿಂಭಾಗ(t);

        name += "+";
        name += ggml_op_desc(t);

        ಆದರೆ (extra_dst) {
            outputs.ತಳ್ಳು_ಹಿಂಭಾಗ(t);
        } ಇಲ್ಲದಿದ್ದರೆ {
            outputs.clear();
            outputs.ತಳ್ಳು_ಹಿಂಭಾಗ(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.ಕೊನೆ(), t), inputs.ಕೊನೆ());

        // Append new inputs from t, preserving middle nullptrs
        ಪೂರ್ಣಾಂಕ n_inputs = n_ಸಕ್ರಿಯ_src(t);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_inputs; i++) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src = t->src[i];
            ಆದರೆ (!src) {
                inputs.ತಳ್ಳು_ಹಿಂಭಾಗ(nullptr);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src != ಗಂಟು &&
                       std::ಕಂಡುಹಿಡಿ(fused.begin(), fused.ಕೊನೆ(), src) == fused.ಕೊನೆ() &&
                       std::ಕಂಡುಹಿಡಿ(inputs.begin(), inputs.ಕೊನೆ(), src) == inputs.ಕೊನೆ()) {
                inputs.ತಳ್ಳು_ಹಿಂಭಾಗ(src);
            }
        }
    }

    ಸ್ಥಿರಾಂಕ std::ಸದಿಶ<ಸ್ಥಿರಾಂಕ ggml_tensor *> & get_inputs() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ inputs;
    }

    ಸ್ಥಿರಾಂಕ std::ಸದಿಶ<ಸ್ಥಿರಾಂಕ ggml_tensor *> & get_outputs() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ outputs;
    }

    std::string op_name() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ name;
    }

    bool is_ಖಾಲಿಗಳು() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ggml_op_is_ಖಾಲಿಗಳು(ಗಂಟು->op);
    }

    bool stackable() ಸ್ಥಿರಾಂಕ {
        ಆಯ್ಕೆ (this->op()) {
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
                ಆದರೆ (ggml_get_op_params_i32(this->ಗಂಟು, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    ಹಿಂತಿರುಗಿ false;
                }
                ಹಿಂತಿರುಗಿ ggml_is_quantized(this->src0()->type);
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
                ಹಿಂತಿರುಗಿ ggml_is_quantized(this->src0()->type);
            ಪೂರ್ವನಿಯೋಜಿತ:
                ಹಿಂತಿರುಗಿ false;
        }
    }

    bool same_ಒಳಹರಿವು(ಸ್ಥಿರಾಂಕ htp_opnode& n) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ n.src1() == this->src1();
    }
};

ರಚನೆ htp_opformat {
    ಅಕ್ಷರ strides[64 * GGML_MAX_SRC];
    ಅಕ್ಷರ dims[64 * GGML_MAX_SRC];
    ಅಕ್ಷರ types[16 * GGML_MAX_SRC];
    ಅಕ್ಷರ buffs[64 * GGML_MAX_SRC];
    ಅಕ್ಷರ ಹೆಸರುಗಳು[64 * GGML_MAX_SRC];
    ಅಕ್ಷರ kparams[128];

    ಪೂರ್ಣಾಂಕ format_tensor_dims(ಅಕ್ಷರ * str, size_t max_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
        ಆದರೆ (!t) {
            ಹಿಂತಿರುಗಿ snprintf(str, max_ಗಾತ್ರ, "NONE");
        }
        ಆದರೆ (t->ne[2] == 1 && t->ne[3] == 1) {
            ಹಿಂತಿರುಗಿ snprintf(str, max_ಗಾತ್ರ, "%d:%d", (ಪೂರ್ಣಾಂಕ) t->ne[0], (ಪೂರ್ಣಾಂಕ) t->ne[1]);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ snprintf(str, max_ಗಾತ್ರ, "%d:%d:%d:%d", (ಪೂರ್ಣಾಂಕ) t->ne[0], (ಪೂರ್ಣಾಂಕ) t->ne[1], (ಪೂರ್ಣಾಂಕ) t->ne[2], (ಪೂರ್ಣಾಂಕ) t->ne[3]);
        }
    }

    ಶೂನ್ಯ format_op_dims(ಅಕ್ಷರ * str, size_t max_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಅಕ್ಷರ * p = str;
        ಅಕ್ಷರ * p_ಕೊನೆ = str + max_ಗಾತ್ರ;
        ಸ್ವಯಂಚಾಲಿತ inputs = ಗಂಟು.get_inputs();

        ಆದರೆ (!inputs.ಖಾಲಿಗಳು()) {
            p += std::min((size_t)format_tensor_dims(p, p_ಕೊನೆ - p, inputs[0]), (size_t)(p_ಕೊನೆ - p));

            ಗಾಗಿ (size_t i = 1; i < inputs.ಗಾತ್ರ_2(); i++) {
                ಆದರೆ (p < p_ಕೊನೆ) {
                    p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, " x "), (size_t)(p_ಕೊನೆ - p));
                }
                ಆದರೆ (p < p_ಕೊನೆ) {
                    p += std::min((size_t)format_tensor_dims(p, p_ಕೊನೆ - p, inputs[i]), (size_t)(p_ಕೊನೆ - p));
                }
            }

            ಆದರೆ (p < p_ಕೊನೆ) {
                p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, " -> "), (size_t)(p_ಕೊನೆ - p));
            }
        }

        ಅಕ್ಷರ self[64];
        format_tensor_dims(self, ಗಾತ್ರ(self), ಗಂಟು.dst());
        ಆದರೆ (p < p_ಕೊನೆ) {
            p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", self), (size_t)(p_ಕೊನೆ - p));
        }
    }

    ಪೂರ್ಣಾಂಕ format_tensor_strides(ಅಕ್ಷರ * str, size_t max_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
        ಆದರೆ (!t) {
            ಹಿಂತಿರುಗಿ snprintf(str, max_ಗಾತ್ರ, "NONE");
        }
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * c = ggml_is_contiguous(t) ? "" : "!";

        ಆದರೆ (t->ne[2] == 1 && t->ne[3] == 1) {
            ಹಿಂತಿರುಗಿ snprintf(str, max_ಗಾತ್ರ, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ snprintf(str, max_ಗಾತ್ರ, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    ಶೂನ್ಯ format_op_strides(ಅಕ್ಷರ * str, size_t max_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಅಕ್ಷರ * p = str;
        ಅಕ್ಷರ * p_ಕೊನೆ = str + max_ಗಾತ್ರ;
        ಸ್ವಯಂಚಾಲಿತ inputs = ಗಂಟು.get_inputs();

        ಆದರೆ (!inputs.ಖಾಲಿಗಳು()) {
            p += std::min((size_t)format_tensor_strides(p, p_ಕೊನೆ - p, inputs[0]), (size_t)(p_ಕೊನೆ - p));

            ಗಾಗಿ (size_t i = 1; i < inputs.ಗಾತ್ರ_2(); i++) {
                ಆದರೆ (p < p_ಕೊನೆ) {
                    p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, " x "), (size_t)(p_ಕೊನೆ - p));
                }
                ಆದರೆ (p < p_ಕೊನೆ) {
                    p += std::min((size_t)format_tensor_strides(p, p_ಕೊನೆ - p, inputs[i]), (size_t)(p_ಕೊನೆ - p));
                }
            }

            ಆದರೆ (p < p_ಕೊನೆ) {
                p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, " -> "), (size_t)(p_ಕೊನೆ - p));
            }
        }

        ಅಕ್ಷರ self[64];
        format_tensor_strides(self, ಗಾತ್ರ(self), ಗಂಟು.dst());
        ಆದರೆ (p < p_ಕೊನೆ) {
            p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", self), (size_t)(p_ಕೊನೆ - p));
        }
    }

    ಶೂನ್ಯ format_op_types(ಅಕ್ಷರ * str, size_t max_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಅಕ್ಷರ * p = str;
        ಅಕ್ಷರ * p_ಕೊನೆ = str + max_ಗಾತ್ರ;
        ಸ್ವಯಂಚಾಲಿತ inputs = ಗಂಟು.get_inputs();

        ಆದರೆ (!inputs.ಖಾಲಿಗಳು()) {
            ಆದರೆ (p < p_ಕೊನೆ) {
                p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_ಕೊನೆ - p));
            }

            ಗಾಗಿ (size_t i = 1; i < inputs.ಗಾತ್ರ_2(); i++) {
                ಆದರೆ (p < p_ಕೊನೆ) {
                    p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, " x "), (size_t)(p_ಕೊನೆ - p));
                }
                ಆದರೆ (p < p_ಕೊನೆ) {
                    p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_ಕೊನೆ - p));
                }
            }

            ಆದರೆ (p < p_ಕೊನೆ) {
                p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, " -> "), (size_t)(p_ಕೊನೆ - p));
            }
        }

        ಆದರೆ (p < p_ಕೊನೆ) {
            p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", ggml_type_name(ಗಂಟು.dst()->type)), (size_t)(p_ಕೊನೆ - p));
        }
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * tensor_buff_name(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
        ಆದರೆ (t && t->ಬಫರ್) {
            ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_name(t->ಬಫರ್);
        }
        ಹಿಂತಿರುಗಿ "NONE";
    }

    ಶೂನ್ಯ format_op_buffs(ಅಕ್ಷರ * str, size_t max_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಅಕ್ಷರ * p = str;
        ಅಕ್ಷರ * p_ಕೊನೆ = str + max_ಗಾತ್ರ;
        ಸ್ವಯಂಚಾಲಿತ inputs = ಗಂಟು.get_inputs();

        ಆದರೆ (!inputs.ಖಾಲಿಗಳು()) {
            ಆದರೆ (p < p_ಕೊನೆ) {
                p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_ಕೊನೆ - p));
            }

            ಗಾಗಿ (size_t i = 1; i < inputs.ಗಾತ್ರ_2(); i++) {
                ಆದರೆ (p < p_ಕೊನೆ) {
                    p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, " x "), (size_t)(p_ಕೊನೆ - p));
                }
                ಆದರೆ (p < p_ಕೊನೆ) {
                    p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_ಕೊನೆ - p));
                }
            }

            ಆದರೆ (p < p_ಕೊನೆ) {
                p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, " -> "), (size_t)(p_ಕೊನೆ - p));
            }
        }

        ಆದರೆ (p < p_ಕೊನೆ) {
            p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", tensor_buff_name(ಗಂಟು.dst())), (size_t)(p_ಕೊನೆ - p));
        }
    }

    ಶೂನ್ಯ format_op_ಹೆಸರುಗಳು(ಅಕ್ಷರ * str, size_t max_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಅಕ್ಷರ * p = str;
        ಅಕ್ಷರ * p_ಕೊನೆ = str + max_ಗಾತ್ರ;
        ಸ್ವಯಂಚಾಲಿತ inputs = ಗಂಟು.get_inputs();

        ಆದರೆ (!inputs.ಖಾಲಿಗಳು()) {
            ಆದರೆ (p < p_ಕೊನೆ) {
                p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_ಕೊನೆ - p));
            }

            ಗಾಗಿ (size_t i = 1; i < inputs.ಗಾತ್ರ_2(); i++) {
                ಆದರೆ (p < p_ಕೊನೆ) {
                    p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, " x "), (size_t)(p_ಕೊನೆ - p));
                }
                ಆದರೆ (p < p_ಕೊನೆ) {
                    p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_ಕೊನೆ - p));
                }
            }

            ಆದರೆ (p < p_ಕೊನೆ) {
                p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, " -> "), (size_t)(p_ಕೊನೆ - p));
            }
        }

        ಆದರೆ (p < p_ಕೊನೆ) {
            p += std::min((size_t)snprintf(p, p_ಕೊನೆ - p, "%s", ಗಂಟು.dst()->name), (size_t)(p_ಕೊನೆ - p));
        }
    }
    ಶೂನ್ಯ format_kernel_params(ಅಕ್ಷರ * str, size_t max_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        ಆದರೆ (ಗಂಟು.opcode == HTP_OP_MUL_MAT || ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID ||
            ಗಂಟು.opcode == HTP_OP_MUL_MAT_NX || ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID_NX ||
            ಗಂಟು.opcode == HTP_OP_MUL_MAT_ಕೂಡಿಸು) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_mm_kernel_params *) ಗಂಟು.kernel_params;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಮಾರ್ಗ = "unknown";
            int32_t type = kparams->kernel_type;
            ಆದರೆ (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                ಮಾರ್ಗ = (kparams->tile_ಗಾತ್ರ > 0) ? "hmx-tiled" : "hmx-flat";
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                ಮಾರ್ಗ = "hvx-flat";
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (type == HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು || type == HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ) {
                ಮಾರ್ಗ = "hvx-tiled";
            }
            snprintf(str, max_ಗಾತ್ರ, "%s vtcm %d", ಮಾರ್ಗ, (ಪೂರ್ಣಾಂಕ) kparams->vtcm_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_fa_kernel_params *) ಗಂಟು.kernel_params;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಮಾರ್ಗ = "unknown";
            int32_t type = kparams->kernel_type;
            ಆದರೆ (type == HTP_FA_KERNEL_HMX) {
                ಮಾರ್ಗ = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (type == HTP_FA_KERNEL_HVX) {
                ಮಾರ್ಗ = "hvx";
            }
            snprintf(str, max_ಗಾತ್ರ, "%s vtcm %d", ಮಾರ್ಗ, (ಪೂರ್ಣಾಂಕ) kparams->vtcm_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (htp_op_is_unary(ಗಂಟು.opcode)) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_kernel_params *) ಗಂಟು.kernel_params;
            snprintf(str, max_ಗಾತ್ರ, "%s vtcm %d", kparams->ಲಂಬಸಾಲು_tile ? "wide-row" : "row-block", (ಪೂರ್ಣಾಂಕ) kparams->vtcm_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_MDEV_ಗುಂಪು && ಗಂಟು.ಗಂಟು) {
            snprintf(str, max_ಗಾತ್ರ, "idx %d count %d", (ಪೂರ್ಣಾಂಕ) ಗಂಟು.ಗಂಟು->op_params[0], (ಪೂರ್ಣಾಂಕ) ಗಂಟು.dst()->ne[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((ಗಂಟು.opcode == HTP_OP_FENCE || ಗಂಟು.opcode == HTP_OP_CPY_FENCE) && ಗಂಟು.ಗಂಟು) {
            snprintf(str, max_ಗಾತ್ರ, "seq 0x%x", (uint32_t) ಗಂಟು.ಗಂಟು->op_params[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_ALLREDUCE && ಗಂಟು.ಗಂಟು) {
            snprintf(str, max_ಗಾತ್ರ, "seq 0x%x -> 0x%x", (uint32_t) ಗಂಟು.ಗಂಟು->op_params[0], (uint32_t) ಗಂಟು.ಗಂಟು->op_params[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_SSM_CONV) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ssm_conv_kernel_params *) ಗಂಟು.kernel_params;
            snprintf(str, max_ಗಾತ್ರ, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (ಪೂರ್ಣಾಂಕ) kparams->vtcm_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_SOFTMAX) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_softmax_kernel_params *) ಗಂಟು.kernel_params;
            snprintf(str, max_ಗಾತ್ರ, "k%d nth %d vtcm %d", (ಪೂರ್ಣಾಂಕ) kparams->kernel_id, (ಪೂರ್ಣಾಂಕ) kparams->n_ಎಳೆಗಳು, (ಪೂರ್ಣಾಂಕ) kparams->vtcm_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_GATED_DELTA_NET) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_gdn_kernel_params *) ಗಂಟು.kernel_params;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಮಾರ್ಗ = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_ಗಾತ್ರ, "%s-%s vtcm %u", ಮಾರ್ಗ, kparams->kda ? "kda" : "scalar",
                     (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) (kparams->vtcm_ಗಾತ್ರ ? kparams->vtcm_ಗಾತ್ರ : kparams->vtcm_ಪ್ರತಿ_thread * kparams->n_ಎಳೆಗಳು));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_MUL || ಗಂಟು.opcode == HTP_OP_ಕೂಡಿಸು || ಗಂಟು.opcode == HTP_OP_ಕೂಡಿಸು_ID ||
                   ಗಂಟು.opcode == HTP_OP_SUB || ಗಂಟು.opcode == HTP_OP_DIV) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params *) ಗಂಟು.kernel_params;
            snprintf(str, max_ಗಾತ್ರ, "vtcm %u", (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) kparams->vtcm_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_ARGSORT || ಗಂಟು.opcode == HTP_OP_ತುದಿ_K) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params *) ಗಂಟು.kernel_params;
            snprintf(str, max_ಗಾತ್ರ, "%s nth %d nchk %d chk %d vtcm %d",
                     ಗಂಟು.opcode == HTP_OP_ತುದಿ_K ? "top_k" : "argsort",
                     (ಪೂರ್ಣಾಂಕ) kparams->n_ಎಳೆಗಳು, (ಪೂರ್ಣಾಂಕ) kparams->n_chunks,
                     (ಪೂರ್ಣಾಂಕ) kparams->chunk_elems, (ಪೂರ್ಣಾಂಕ) kparams->vtcm_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು.opcode == HTP_OP_GET_ಅಡ್ಡಸಾಲುಗಳು) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params *) ಗಂಟು.kernel_params;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ktype_str = "unknown";
            ಆಯ್ಕೆ (kparams->kernel_type) {
                ಸ್ಥಿತಿ HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_SAMETYPE: ktype_str = "sametype"; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_TILED:    ktype_str = "tiled"; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_FLAT:     ktype_str = "flat"; ನಿಲ್ಲಿಸು;
            }
            snprintf(str, max_ಗಾತ್ರ, "%s%s vtcm %u", ktype_str, kparams->n_ಎಳೆಗಳು > 1 ? "-multi" : "", (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ) kparams->vtcm_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            snprintf(str, max_ಗಾತ್ರ, "----");
        }

    }

    ಶೂನ್ಯ format(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) {
        format_op_dims(dims, ಗಾತ್ರ(dims), ಗಂಟು);
        format_op_strides(strides, ಗಾತ್ರ(strides), ಗಂಟು);
        format_op_types(types, ಗಾತ್ರ(types), ಗಂಟು);
        format_op_buffs(buffs, ಗಾತ್ರ(buffs), ಗಂಟು);
        format_op_ಹೆಸರುಗಳು(ಹೆಸರುಗಳು, ಗಾತ್ರ(ಹೆಸರುಗಳು), ಗಂಟು);
        format_kernel_params(kparams, ಗಾತ್ರ(kparams), ಗಂಟು);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        ಹೆಸರುಗಳು[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ಸ್ಥಿರಾಂಕ htp_opnode & ಗಂಟು) { format(ಗಂಟು); }
};

#ಆದರೆ_ಅಂತ್ಯ // HTP_OPNODE_H
