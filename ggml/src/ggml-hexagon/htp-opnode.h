#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_OPNODE_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_OPNODE_H

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_CPP
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/matmul-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/flash-attn-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/unary-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/binary-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/allreduce-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/ssm-conv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/gated-delta-net-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/softmax-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/argsort-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/get-rows-ops.h"

ਬਣਤਰ htp_opnode {
    ggml_tensor * ਗੰਢ   { nullptr };
    htp_op_ਕੋਡ   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::ਸਦਿਸ਼<ggml_tensor *>                fused;
    std::ਸਦਿਸ਼<std::shared_ptr<ggml_tensor>> dummy;

    std::ਸਦਿਸ਼<ਅਚਲ ggml_tensor *> inputs;
    std::ਸਦਿਸ਼<ਅਚਲ ggml_tensor *> outputs;
    std::string                      name;

    ਪੂਰਨ n_ਸਰਗਰਮ_src(ਅਚਲ ggml_tensor * t) ਅਚਲ {
        ਜੇ (!t) ਵਾਪਸ 0;
        ਲਈ (ਪੂਰਨ i = GGML_MAX_SRC - 1; i >= 0; i--) {
            ਜੇ (t->src[i]) {
                ਵਾਪਸ i + 1;
            }
        }
        ਵਾਪਸ 0;
    }

    ਖਾਲੀ init(ggml_tensor * ਗੰਢ) {
        this->ਗੰਢ = ਗੰਢ;
        ਜੇ (this->ਗੰਢ) {
            ਜੇ (this->ਗੰਢ->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->ਗੰਢ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.ਧੱਕੋ_ਪਿੱਠ(this->ਗੰਢ->src[1]);
            } ਨਹੀਂਤਾਂ {
                this->name = ggml_op_desc(this->ਗੰਢ);

                // Build inputs (preserving optional nullptrs)
                ਪੂਰਨ n_inputs = n_ਸਰਗਰਮ_src(this->ਗੰਢ);
                this->inputs.resize(n_inputs, nullptr);
                ਲਈ (ਪੂਰਨ i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->ਗੰਢ->src[i];
                }
            }

            // Build outputs
            this->outputs.ਧੱਕੋ_ਪਿੱਠ(this->dst());
        }
    }

    htp_opnode(htp_op_ਕੋਡ opcode = HTP_OP_INVALID, ggml_tensor * ਗੰਢ = nullptr) : opcode(opcode) {
        init(ਗੰਢ);
    }

    ggml_op             op()   ਅਚਲ { ਵਾਪਸ ਗੰਢ->op; }
    ਅਚਲ ggml_tensor * src0() ਅਚਲ { ਵਾਪਸ ਗੰਢ->src[0]; }
    ਅਚਲ ggml_tensor * src1() ਅਚਲ { ਵਾਪਸ ਗੰਢ->src[1]; }
    ਅਚਲ ggml_tensor * dst()  ਅਚਲ { ਵਾਪਸ outputs.ਖਾਲੀ_ਥਾਵਾਂ() ? ਗੰਢ : outputs.ਪਿੱਠ(); }

    ggml_tensor * ਜੋੜੋ_dummy(ਅਚਲ ggml_tensor & t) {
        dummy.ਧੱਕੋ_ਪਿੱਠ(std::make_shared<ggml_tensor>(t));
        ਵਾਪਸ dummy.ਪਿੱਠ().get();
    }

    ਖਾਲੀ ਜੋੜੋ_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.ਧੱਕੋ_ਪਿੱਠ(t);

        name += "+";
        name += ggml_op_desc(t);

        ਜੇ (extra_dst) {
            outputs.ਧੱਕੋ_ਪਿੱਠ(t);
        } ਨਹੀਂਤਾਂ {
            outputs.clear();
            outputs.ਧੱਕੋ_ਪਿੱਠ(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.ਸਿਰਾ(), t), inputs.ਸਿਰਾ());

        // Append new inputs from t, preserving middle nullptrs
        ਪੂਰਨ n_inputs = n_ਸਰਗਰਮ_src(t);
        ਲਈ (ਪੂਰਨ i = 0; i < n_inputs; i++) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * src = t->src[i];
            ਜੇ (!src) {
                inputs.ਧੱਕੋ_ਪਿੱਠ(nullptr);
            } ਨਹੀਂਤਾਂ ਜੇ (src != ਗੰਢ &&
                       std::ਲੱਭੋ(fused.begin(), fused.ਸਿਰਾ(), src) == fused.ਸਿਰਾ() &&
                       std::ਲੱਭੋ(inputs.begin(), inputs.ਸਿਰਾ(), src) == inputs.ਸਿਰਾ()) {
                inputs.ਧੱਕੋ_ਪਿੱਠ(src);
            }
        }
    }

    ਅਚਲ std::ਸਦਿਸ਼<ਅਚਲ ggml_tensor *> & get_inputs() ਅਚਲ {
        ਵਾਪਸ inputs;
    }

    ਅਚਲ std::ਸਦਿਸ਼<ਅਚਲ ggml_tensor *> & get_outputs() ਅਚਲ {
        ਵਾਪਸ outputs;
    }

    std::string op_name() ਅਚਲ {
        ਵਾਪਸ name;
    }

    bool is_ਖਾਲੀ_ਥਾਵਾਂ() ਅਚਲ {
        ਵਾਪਸ ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ->op);
    }

    bool stackable() ਅਚਲ {
        ਚੋਣ (this->op()) {
            ਹਾਲਤ GGML_OP_MUL_MAT:
                ਜੇ (ggml_get_op_params_i32(this->ਗੰਢ, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    ਵਾਪਸ false;
                }
                ਵਾਪਸ ggml_is_quantized(this->src0()->type);
            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                ਵਾਪਸ ggml_is_quantized(this->src0()->type);
            ਮੂਲ:
                ਵਾਪਸ false;
        }
    }

    bool same_ਨਿਵੇਸ਼(ਅਚਲ htp_opnode& n) ਅਚਲ {
        ਵਾਪਸ n.src1() == this->src1();
    }
};

ਬਣਤਰ htp_opformat {
    ਅੱਖਰ strides[64 * GGML_MAX_SRC];
    ਅੱਖਰ dims[64 * GGML_MAX_SRC];
    ਅੱਖਰ types[16 * GGML_MAX_SRC];
    ਅੱਖਰ buffs[64 * GGML_MAX_SRC];
    ਅੱਖਰ ਨਾਂ[64 * GGML_MAX_SRC];
    ਅੱਖਰ kparams[128];

    ਪੂਰਨ format_tensor_dims(ਅੱਖਰ * str, size_t max_ਮਾਪ, ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
        ਜੇ (!t) {
            ਵਾਪਸ snprintf(str, max_ਮਾਪ, "NONE");
        }
        ਜੇ (t->ne[2] == 1 && t->ne[3] == 1) {
            ਵਾਪਸ snprintf(str, max_ਮਾਪ, "%d:%d", (ਪੂਰਨ) t->ne[0], (ਪੂਰਨ) t->ne[1]);
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ snprintf(str, max_ਮਾਪ, "%d:%d:%d:%d", (ਪੂਰਨ) t->ne[0], (ਪੂਰਨ) t->ne[1], (ਪੂਰਨ) t->ne[2], (ਪੂਰਨ) t->ne[3]);
        }
    }

    ਖਾਲੀ format_op_dims(ਅੱਖਰ * str, size_t max_ਮਾਪ, ਅਚਲ htp_opnode & ਗੰਢ) {
        ਅੱਖਰ * p = str;
        ਅੱਖਰ * p_ਸਿਰਾ = str + max_ਮਾਪ;
        ਸਵੈਚਾਲਿਤ inputs = ਗੰਢ.get_inputs();

        ਜੇ (!inputs.ਖਾਲੀ_ਥਾਵਾਂ()) {
            p += std::min((size_t)format_tensor_dims(p, p_ਸਿਰਾ - p, inputs[0]), (size_t)(p_ਸਿਰਾ - p));

            ਲਈ (size_t i = 1; i < inputs.ਮਾਪ(); i++) {
                ਜੇ (p < p_ਸਿਰਾ) {
                    p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, " x "), (size_t)(p_ਸਿਰਾ - p));
                }
                ਜੇ (p < p_ਸਿਰਾ) {
                    p += std::min((size_t)format_tensor_dims(p, p_ਸਿਰਾ - p, inputs[i]), (size_t)(p_ਸਿਰਾ - p));
                }
            }

            ਜੇ (p < p_ਸਿਰਾ) {
                p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, " -> "), (size_t)(p_ਸਿਰਾ - p));
            }
        }

        ਅੱਖਰ self[64];
        format_tensor_dims(self, ਆਕਾਰ(self), ਗੰਢ.dst());
        ਜੇ (p < p_ਸਿਰਾ) {
            p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", self), (size_t)(p_ਸਿਰਾ - p));
        }
    }

    ਪੂਰਨ format_tensor_strides(ਅੱਖਰ * str, size_t max_ਮਾਪ, ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
        ਜੇ (!t) {
            ਵਾਪਸ snprintf(str, max_ਮਾਪ, "NONE");
        }
        ਅਚਲ ਅੱਖਰ * c = ggml_is_contiguous(t) ? "" : "!";

        ਜੇ (t->ne[2] == 1 && t->ne[3] == 1) {
            ਵਾਪਸ snprintf(str, max_ਮਾਪ, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ snprintf(str, max_ਮਾਪ, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    ਖਾਲੀ format_op_strides(ਅੱਖਰ * str, size_t max_ਮਾਪ, ਅਚਲ htp_opnode & ਗੰਢ) {
        ਅੱਖਰ * p = str;
        ਅੱਖਰ * p_ਸਿਰਾ = str + max_ਮਾਪ;
        ਸਵੈਚਾਲਿਤ inputs = ਗੰਢ.get_inputs();

        ਜੇ (!inputs.ਖਾਲੀ_ਥਾਵਾਂ()) {
            p += std::min((size_t)format_tensor_strides(p, p_ਸਿਰਾ - p, inputs[0]), (size_t)(p_ਸਿਰਾ - p));

            ਲਈ (size_t i = 1; i < inputs.ਮਾਪ(); i++) {
                ਜੇ (p < p_ਸਿਰਾ) {
                    p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, " x "), (size_t)(p_ਸਿਰਾ - p));
                }
                ਜੇ (p < p_ਸਿਰਾ) {
                    p += std::min((size_t)format_tensor_strides(p, p_ਸਿਰਾ - p, inputs[i]), (size_t)(p_ਸਿਰਾ - p));
                }
            }

            ਜੇ (p < p_ਸਿਰਾ) {
                p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, " -> "), (size_t)(p_ਸਿਰਾ - p));
            }
        }

        ਅੱਖਰ self[64];
        format_tensor_strides(self, ਆਕਾਰ(self), ਗੰਢ.dst());
        ਜੇ (p < p_ਸਿਰਾ) {
            p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", self), (size_t)(p_ਸਿਰਾ - p));
        }
    }

    ਖਾਲੀ format_op_types(ਅੱਖਰ * str, size_t max_ਮਾਪ, ਅਚਲ htp_opnode & ਗੰਢ) {
        ਅੱਖਰ * p = str;
        ਅੱਖਰ * p_ਸਿਰਾ = str + max_ਮਾਪ;
        ਸਵੈਚਾਲਿਤ inputs = ਗੰਢ.get_inputs();

        ਜੇ (!inputs.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਜੇ (p < p_ਸਿਰਾ) {
                p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_ਸਿਰਾ - p));
            }

            ਲਈ (size_t i = 1; i < inputs.ਮਾਪ(); i++) {
                ਜੇ (p < p_ਸਿਰਾ) {
                    p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, " x "), (size_t)(p_ਸਿਰਾ - p));
                }
                ਜੇ (p < p_ਸਿਰਾ) {
                    p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_ਸਿਰਾ - p));
                }
            }

            ਜੇ (p < p_ਸਿਰਾ) {
                p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, " -> "), (size_t)(p_ਸਿਰਾ - p));
            }
        }

        ਜੇ (p < p_ਸਿਰਾ) {
            p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", ggml_type_name(ਗੰਢ.dst()->type)), (size_t)(p_ਸਿਰਾ - p));
        }
    }

    ਅਚਲ ਅੱਖਰ * tensor_buff_name(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
        ਜੇ (t && t->ਬਫਰ) {
            ਵਾਪਸ ggml_backend_ਬਫਰ_name(t->ਬਫਰ);
        }
        ਵਾਪਸ "NONE";
    }

    ਖਾਲੀ format_op_buffs(ਅੱਖਰ * str, size_t max_ਮਾਪ, ਅਚਲ htp_opnode & ਗੰਢ) {
        ਅੱਖਰ * p = str;
        ਅੱਖਰ * p_ਸਿਰਾ = str + max_ਮਾਪ;
        ਸਵੈਚਾਲਿਤ inputs = ਗੰਢ.get_inputs();

        ਜੇ (!inputs.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਜੇ (p < p_ਸਿਰਾ) {
                p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_ਸਿਰਾ - p));
            }

            ਲਈ (size_t i = 1; i < inputs.ਮਾਪ(); i++) {
                ਜੇ (p < p_ਸਿਰਾ) {
                    p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, " x "), (size_t)(p_ਸਿਰਾ - p));
                }
                ਜੇ (p < p_ਸਿਰਾ) {
                    p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_ਸਿਰਾ - p));
                }
            }

            ਜੇ (p < p_ਸਿਰਾ) {
                p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, " -> "), (size_t)(p_ਸਿਰਾ - p));
            }
        }

        ਜੇ (p < p_ਸਿਰਾ) {
            p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", tensor_buff_name(ਗੰਢ.dst())), (size_t)(p_ਸਿਰਾ - p));
        }
    }

    ਖਾਲੀ format_op_ਨਾਂ(ਅੱਖਰ * str, size_t max_ਮਾਪ, ਅਚਲ htp_opnode & ਗੰਢ) {
        ਅੱਖਰ * p = str;
        ਅੱਖਰ * p_ਸਿਰਾ = str + max_ਮਾਪ;
        ਸਵੈਚਾਲਿਤ inputs = ਗੰਢ.get_inputs();

        ਜੇ (!inputs.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਜੇ (p < p_ਸਿਰਾ) {
                p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_ਸਿਰਾ - p));
            }

            ਲਈ (size_t i = 1; i < inputs.ਮਾਪ(); i++) {
                ਜੇ (p < p_ਸਿਰਾ) {
                    p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, " x "), (size_t)(p_ਸਿਰਾ - p));
                }
                ਜੇ (p < p_ਸਿਰਾ) {
                    p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_ਸਿਰਾ - p));
                }
            }

            ਜੇ (p < p_ਸਿਰਾ) {
                p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, " -> "), (size_t)(p_ਸਿਰਾ - p));
            }
        }

        ਜੇ (p < p_ਸਿਰਾ) {
            p += std::min((size_t)snprintf(p, p_ਸਿਰਾ - p, "%s", ਗੰਢ.dst()->name), (size_t)(p_ਸਿਰਾ - p));
        }
    }
    ਖਾਲੀ format_kernel_params(ਅੱਖਰ * str, size_t max_ਮਾਪ, ਅਚਲ htp_opnode & ਗੰਢ) {
        ਜੇ (ਗੰਢ.opcode == HTP_OP_MUL_MAT || ਗੰਢ.opcode == HTP_OP_MUL_MAT_ID ||
            ਗੰਢ.opcode == HTP_OP_MUL_MAT_NX || ਗੰਢ.opcode == HTP_OP_MUL_MAT_ID_NX ||
            ਗੰਢ.opcode == HTP_OP_MUL_MAT_ਜੋੜੋ) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) ਗੰਢ.kernel_params;
            ਅਚਲ ਅੱਖਰ * ਰਸਤਾ = "unknown";
            int32_t type = kparams->kernel_type;
            ਜੇ (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                ਰਸਤਾ = (kparams->tile_ਮਾਪ > 0) ? "hmx-tiled" : "hmx-flat";
            } ਨਹੀਂਤਾਂ ਜੇ (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                ਰਸਤਾ = "hvx-flat";
            } ਨਹੀਂਤਾਂ ਜੇ (type == HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ || type == HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ) {
                ਰਸਤਾ = "hvx-tiled";
            }
            snprintf(str, max_ਮਾਪ, "%s vtcm %d", ਰਸਤਾ, (ਪੂਰਨ) kparams->vtcm_ਮਾਪ);
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_FLASH_ATTN_EXT) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_fa_kernel_params *) ਗੰਢ.kernel_params;
            ਅਚਲ ਅੱਖਰ * ਰਸਤਾ = "unknown";
            int32_t type = kparams->kernel_type;
            ਜੇ (type == HTP_FA_KERNEL_HMX) {
                ਰਸਤਾ = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } ਨਹੀਂਤਾਂ ਜੇ (type == HTP_FA_KERNEL_HVX) {
                ਰਸਤਾ = "hvx";
            }
            snprintf(str, max_ਮਾਪ, "%s vtcm %d", ਰਸਤਾ, (ਪੂਰਨ) kparams->vtcm_ਮਾਪ);
        } ਨਹੀਂਤਾਂ ਜੇ (htp_op_is_unary(ਗੰਢ.opcode)) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_unary_kernel_params *) ਗੰਢ.kernel_params;
            snprintf(str, max_ਮਾਪ, "%s vtcm %d", kparams->ਖਾਨਾ_tile ? "wide-row" : "row-block", (ਪੂਰਨ) kparams->vtcm_ਮਾਪ);
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_MDEV_ਸਮੂਹ && ਗੰਢ.ਗੰਢ) {
            snprintf(str, max_ਮਾਪ, "idx %d count %d", (ਪੂਰਨ) ਗੰਢ.ਗੰਢ->op_params[0], (ਪੂਰਨ) ਗੰਢ.dst()->ne[1]);
        } ਨਹੀਂਤਾਂ ਜੇ ((ਗੰਢ.opcode == HTP_OP_FENCE || ਗੰਢ.opcode == HTP_OP_CPY_FENCE) && ਗੰਢ.ਗੰਢ) {
            snprintf(str, max_ਮਾਪ, "seq 0x%x", (uint32_t) ਗੰਢ.ਗੰਢ->op_params[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_ALLREDUCE && ਗੰਢ.ਗੰਢ) {
            snprintf(str, max_ਮਾਪ, "seq 0x%x -> 0x%x", (uint32_t) ਗੰਢ.ਗੰਢ->op_params[0], (uint32_t) ਗੰਢ.ਗੰਢ->op_params[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_SSM_CONV) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_ssm_conv_kernel_params *) ਗੰਢ.kernel_params;
            snprintf(str, max_ਮਾਪ, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (ਪੂਰਨ) kparams->vtcm_ਮਾਪ);
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_SOFTMAX) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_softmax_kernel_params *) ਗੰਢ.kernel_params;
            snprintf(str, max_ਮਾਪ, "k%d nth %d vtcm %d", (ਪੂਰਨ) kparams->kernel_id, (ਪੂਰਨ) kparams->n_ਧਾਗੇ, (ਪੂਰਨ) kparams->vtcm_ਮਾਪ);
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_GATED_DELTA_NET) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_gdn_kernel_params *) ਗੰਢ.kernel_params;
            ਅਚਲ ਅੱਖਰ * ਰਸਤਾ = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_ਮਾਪ, "%s-%s vtcm %u", ਰਸਤਾ, kparams->kda ? "kda" : "scalar",
                     (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) (kparams->vtcm_ਮਾਪ ? kparams->vtcm_ਮਾਪ : kparams->vtcm_ਪ੍ਰਤੀ_thread * kparams->n_ਧਾਗੇ));
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_MUL || ਗੰਢ.opcode == HTP_OP_ਜੋੜੋ || ਗੰਢ.opcode == HTP_OP_ਜੋੜੋ_ID ||
                   ਗੰਢ.opcode == HTP_OP_SUB || ਗੰਢ.opcode == HTP_OP_DIV) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params *) ਗੰਢ.kernel_params;
            snprintf(str, max_ਮਾਪ, "vtcm %u", (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) kparams->vtcm_ਮਾਪ);
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_ARGSORT || ਗੰਢ.opcode == HTP_OP_ਸਿਖਰ_K) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params *) ਗੰਢ.kernel_params;
            snprintf(str, max_ਮਾਪ, "%s nth %d nchk %d chk %d vtcm %d",
                     ਗੰਢ.opcode == HTP_OP_ਸਿਖਰ_K ? "top_k" : "argsort",
                     (ਪੂਰਨ) kparams->n_ਧਾਗੇ, (ਪੂਰਨ) kparams->n_chunks,
                     (ਪੂਰਨ) kparams->chunk_elems, (ਪੂਰਨ) kparams->vtcm_ਮਾਪ);
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_GET_ਪੰਕਤੀਆਂ) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params *) ਗੰਢ.kernel_params;
            ਅਚਲ ਅੱਖਰ * ktype_str = "unknown";
            ਚੋਣ (kparams->kernel_type) {
                ਹਾਲਤ HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_SAMETYPE: ktype_str = "sametype"; ਰੋਕੋ;
                ਹਾਲਤ HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_TILED:    ktype_str = "tiled"; ਰੋਕੋ;
                ਹਾਲਤ HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_FLAT:     ktype_str = "flat"; ਰੋਕੋ;
            }
            snprintf(str, max_ਮਾਪ, "%s%s vtcm %u", ktype_str, kparams->n_ਧਾਗੇ > 1 ? "-multi" : "", (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) kparams->vtcm_ਮਾਪ);
        } ਨਹੀਂਤਾਂ {
            snprintf(str, max_ਮਾਪ, "----");
        }

    }

    ਖਾਲੀ format(ਅਚਲ htp_opnode & ਗੰਢ) {
        format_op_dims(dims, ਆਕਾਰ(dims), ਗੰਢ);
        format_op_strides(strides, ਆਕਾਰ(strides), ਗੰਢ);
        format_op_types(types, ਆਕਾਰ(types), ਗੰਢ);
        format_op_buffs(buffs, ਆਕਾਰ(buffs), ਗੰਢ);
        format_op_ਨਾਂ(ਨਾਂ, ਆਕਾਰ(ਨਾਂ), ਗੰਢ);
        format_kernel_params(kparams, ਆਕਾਰ(kparams), ਗੰਢ);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        ਨਾਂ[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(ਅਚਲ htp_opnode & ਗੰਢ) { format(ਗੰਢ); }
};

#ਜੇ_ਅੰਤ // HTP_OPNODE_H
