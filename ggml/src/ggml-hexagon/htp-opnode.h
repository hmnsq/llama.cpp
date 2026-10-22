#నిర్వచితం_కాకపోతే HTP_OPNODE_H
#నిర్వచించు HTP_OPNODE_H

#నిర్వచించు GGML_COMMON_IMPL_CPP
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-common.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <string>
#అన్తర్భావ <vector>
#అన్తర్భావ <memory>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp/matmul-ops.h"
#అన్తర్భావ "htp/flash-attn-ops.h"
#అన్తర్భావ "htp/unary-ops.h"
#అన్తర్భావ "htp/binary-ops.h"
#అన్తర్భావ "htp/allreduce-ops.h"
#అన్తర్భావ "htp/ssm-conv.h"
#అన్తర్భావ "htp/gated-delta-net-ops.h"
#అన్తర్భావ "htp/softmax-ops.h"
#అన్తర్భావ "htp/argsort-ops.h"
#అన్తర్భావ "htp/get-rows-ops.h"

నిర్మాణం htp_opnode {
    ggml_tensor * కణుపు   { nullptr };
    htp_op_సంకేతం   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::సదిశ<ggml_tensor *>                fused;
    std::సదిశ<std::shared_ptr<ggml_tensor>> dummy;

    std::సదిశ<స్థిరరాశి ggml_tensor *> inputs;
    std::సదిశ<స్థిరరాశి ggml_tensor *> outputs;
    std::string                      name;

    పూర్ణాంకం n_చురుకైన_src(స్థిరరాశి ggml_tensor * t) స్థిరరాశి {
        ఒకవేళ (!t) తిరిగి 0;
        కొరకు (పూర్ణాంకం i = GGML_MAX_SRC - 1; i >= 0; i--) {
            ఒకవేళ (t->src[i]) {
                తిరిగి i + 1;
            }
        }
        తిరిగి 0;
    }

    శూన్య init(ggml_tensor * కణుపు) {
        this->కణుపు = కణుపు;
        ఒకవేళ (this->కణుపు) {
            ఒకవేళ (this->కణుపు->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->కణుపు, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.నెట్టు_వెనుక(this->కణుపు->src[1]);
            } లేకపోతే {
                this->name = ggml_op_desc(this->కణుపు);

                // Build inputs (preserving optional nullptrs)
                పూర్ణాంకం n_inputs = n_చురుకైన_src(this->కణుపు);
                this->inputs.resize(n_inputs, nullptr);
                కొరకు (పూర్ణాంకం i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->కణుపు->src[i];
                }
            }

            // Build outputs
            this->outputs.నెట్టు_వెనుక(this->dst());
        }
    }

    htp_opnode(htp_op_సంకేతం opcode = HTP_OP_INVALID, ggml_tensor * కణుపు = nullptr) : opcode(opcode) {
        init(కణుపు);
    }

    ggml_op             op()   స్థిరరాశి { తిరిగి కణుపు->op; }
    స్థిరరాశి ggml_tensor * src0() స్థిరరాశి { తిరిగి కణుపు->src[0]; }
    స్థిరరాశి ggml_tensor * src1() స్థిరరాశి { తిరిగి కణుపు->src[1]; }
    స్థిరరాశి ggml_tensor * dst()  స్థిరరాశి { తిరిగి outputs.ఖాళీలు() ? కణుపు : outputs.వెనుక(); }

    ggml_tensor * కూడు_dummy(స్థిరరాశి ggml_tensor & t) {
        dummy.నెట్టు_వెనుక(std::make_shared<ggml_tensor>(t));
        తిరిగి dummy.వెనుక().get();
    }

    శూన్య కూడు_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.నెట్టు_వెనుక(t);

        name += "+";
        name += ggml_op_desc(t);

        ఒకవేళ (extra_dst) {
            outputs.నెట్టు_వెనుక(t);
        } లేకపోతే {
            outputs.clear();
            outputs.నెట్టు_వెనుక(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.చివర(), t), inputs.చివర());

        // Append new inputs from t, preserving middle nullptrs
        పూర్ణాంకం n_inputs = n_చురుకైన_src(t);
        కొరకు (పూర్ణాంకం i = 0; i < n_inputs; i++) {
            స్థిరరాశి స్వయంచాలక * src = t->src[i];
            ఒకవేళ (!src) {
                inputs.నెట్టు_వెనుక(nullptr);
            } లేకపోతే ఒకవేళ (src != కణుపు &&
                       std::కనుగొను(fused.begin(), fused.చివర(), src) == fused.చివర() &&
                       std::కనుగొను(inputs.begin(), inputs.చివర(), src) == inputs.చివర()) {
                inputs.నెట్టు_వెనుక(src);
            }
        }
    }

    స్థిరరాశి std::సదిశ<స్థిరరాశి ggml_tensor *> & get_inputs() స్థిరరాశి {
        తిరిగి inputs;
    }

    స్థిరరాశి std::సదిశ<స్థిరరాశి ggml_tensor *> & get_outputs() స్థిరరాశి {
        తిరిగి outputs;
    }

    std::string op_name() స్థిరరాశి {
        తిరిగి name;
    }

    bool is_ఖాళీలు() స్థిరరాశి {
        తిరిగి ggml_op_is_ఖాళీలు(కణుపు->op);
    }

    bool stackable() స్థిరరాశి {
        ఎంపిక (this->op()) {
            స్థితి GGML_OP_MUL_MAT:
                ఒకవేళ (ggml_get_op_params_i32(this->కణుపు, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    తిరిగి false;
                }
                తిరిగి ggml_is_quantized(this->src0()->type);
            స్థితి GGML_OP_MUL_MAT_ID:
                తిరిగి ggml_is_quantized(this->src0()->type);
            అప్రమేయ:
                తిరిగి false;
        }
    }

    bool same_ఆదానం(స్థిరరాశి htp_opnode& n) స్థిరరాశి {
        తిరిగి n.src1() == this->src1();
    }
};

నిర్మాణం htp_opformat {
    అక్షరం strides[64 * GGML_MAX_SRC];
    అక్షరం dims[64 * GGML_MAX_SRC];
    అక్షరం types[16 * GGML_MAX_SRC];
    అక్షరం buffs[64 * GGML_MAX_SRC];
    అక్షరం పేర్లు[64 * GGML_MAX_SRC];
    అక్షరం kparams[128];

    పూర్ణాంకం format_tensor_dims(అక్షరం * str, size_t max_కొలత, స్థిరరాశి నిర్మాణం ggml_tensor * t) {
        ఒకవేళ (!t) {
            తిరిగి snprintf(str, max_కొలత, "NONE");
        }
        ఒకవేళ (t->ne[2] == 1 && t->ne[3] == 1) {
            తిరిగి snprintf(str, max_కొలత, "%d:%d", (పూర్ణాంకం) t->ne[0], (పూర్ణాంకం) t->ne[1]);
        } లేకపోతే {
            తిరిగి snprintf(str, max_కొలత, "%d:%d:%d:%d", (పూర్ణాంకం) t->ne[0], (పూర్ణాంకం) t->ne[1], (పూర్ణాంకం) t->ne[2], (పూర్ణాంకం) t->ne[3]);
        }
    }

    శూన్య format_op_dims(అక్షరం * str, size_t max_కొలత, స్థిరరాశి htp_opnode & కణుపు) {
        అక్షరం * p = str;
        అక్షరం * p_చివర = str + max_కొలత;
        స్వయంచాలక inputs = కణుపు.get_inputs();

        ఒకవేళ (!inputs.ఖాళీలు()) {
            p += std::min((size_t)format_tensor_dims(p, p_చివర - p, inputs[0]), (size_t)(p_చివర - p));

            కొరకు (size_t i = 1; i < inputs.కొలత(); i++) {
                ఒకవేళ (p < p_చివర) {
                    p += std::min((size_t)snprintf(p, p_చివర - p, " x "), (size_t)(p_చివర - p));
                }
                ఒకవేళ (p < p_చివర) {
                    p += std::min((size_t)format_tensor_dims(p, p_చివర - p, inputs[i]), (size_t)(p_చివర - p));
                }
            }

            ఒకవేళ (p < p_చివర) {
                p += std::min((size_t)snprintf(p, p_చివర - p, " -> "), (size_t)(p_చివర - p));
            }
        }

        అక్షరం self[64];
        format_tensor_dims(self, పరిమాణం(self), కణుపు.dst());
        ఒకవేళ (p < p_చివర) {
            p += std::min((size_t)snprintf(p, p_చివర - p, "%s", self), (size_t)(p_చివర - p));
        }
    }

    పూర్ణాంకం format_tensor_strides(అక్షరం * str, size_t max_కొలత, స్థిరరాశి నిర్మాణం ggml_tensor * t) {
        ఒకవేళ (!t) {
            తిరిగి snprintf(str, max_కొలత, "NONE");
        }
        స్థిరరాశి అక్షరం * c = ggml_is_contiguous(t) ? "" : "!";

        ఒకవేళ (t->ne[2] == 1 && t->ne[3] == 1) {
            తిరిగి snprintf(str, max_కొలత, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } లేకపోతే {
            తిరిగి snprintf(str, max_కొలత, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    శూన్య format_op_strides(అక్షరం * str, size_t max_కొలత, స్థిరరాశి htp_opnode & కణుపు) {
        అక్షరం * p = str;
        అక్షరం * p_చివర = str + max_కొలత;
        స్వయంచాలక inputs = కణుపు.get_inputs();

        ఒకవేళ (!inputs.ఖాళీలు()) {
            p += std::min((size_t)format_tensor_strides(p, p_చివర - p, inputs[0]), (size_t)(p_చివర - p));

            కొరకు (size_t i = 1; i < inputs.కొలత(); i++) {
                ఒకవేళ (p < p_చివర) {
                    p += std::min((size_t)snprintf(p, p_చివర - p, " x "), (size_t)(p_చివర - p));
                }
                ఒకవేళ (p < p_చివర) {
                    p += std::min((size_t)format_tensor_strides(p, p_చివర - p, inputs[i]), (size_t)(p_చివర - p));
                }
            }

            ఒకవేళ (p < p_చివర) {
                p += std::min((size_t)snprintf(p, p_చివర - p, " -> "), (size_t)(p_చివర - p));
            }
        }

        అక్షరం self[64];
        format_tensor_strides(self, పరిమాణం(self), కణుపు.dst());
        ఒకవేళ (p < p_చివర) {
            p += std::min((size_t)snprintf(p, p_చివర - p, "%s", self), (size_t)(p_చివర - p));
        }
    }

    శూన్య format_op_types(అక్షరం * str, size_t max_కొలత, స్థిరరాశి htp_opnode & కణుపు) {
        అక్షరం * p = str;
        అక్షరం * p_చివర = str + max_కొలత;
        స్వయంచాలక inputs = కణుపు.get_inputs();

        ఒకవేళ (!inputs.ఖాళీలు()) {
            ఒకవేళ (p < p_చివర) {
                p += std::min((size_t)snprintf(p, p_చివర - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_చివర - p));
            }

            కొరకు (size_t i = 1; i < inputs.కొలత(); i++) {
                ఒకవేళ (p < p_చివర) {
                    p += std::min((size_t)snprintf(p, p_చివర - p, " x "), (size_t)(p_చివర - p));
                }
                ఒకవేళ (p < p_చివర) {
                    p += std::min((size_t)snprintf(p, p_చివర - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_చివర - p));
                }
            }

            ఒకవేళ (p < p_చివర) {
                p += std::min((size_t)snprintf(p, p_చివర - p, " -> "), (size_t)(p_చివర - p));
            }
        }

        ఒకవేళ (p < p_చివర) {
            p += std::min((size_t)snprintf(p, p_చివర - p, "%s", ggml_type_name(కణుపు.dst()->type)), (size_t)(p_చివర - p));
        }
    }

    స్థిరరాశి అక్షరం * tensor_buff_name(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
        ఒకవేళ (t && t->బఫరు) {
            తిరిగి ggml_backend_బఫరు_name(t->బఫరు);
        }
        తిరిగి "NONE";
    }

    శూన్య format_op_buffs(అక్షరం * str, size_t max_కొలత, స్థిరరాశి htp_opnode & కణుపు) {
        అక్షరం * p = str;
        అక్షరం * p_చివర = str + max_కొలత;
        స్వయంచాలక inputs = కణుపు.get_inputs();

        ఒకవేళ (!inputs.ఖాళీలు()) {
            ఒకవేళ (p < p_చివర) {
                p += std::min((size_t)snprintf(p, p_చివర - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_చివర - p));
            }

            కొరకు (size_t i = 1; i < inputs.కొలత(); i++) {
                ఒకవేళ (p < p_చివర) {
                    p += std::min((size_t)snprintf(p, p_చివర - p, " x "), (size_t)(p_చివర - p));
                }
                ఒకవేళ (p < p_చివర) {
                    p += std::min((size_t)snprintf(p, p_చివర - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_చివర - p));
                }
            }

            ఒకవేళ (p < p_చివర) {
                p += std::min((size_t)snprintf(p, p_చివర - p, " -> "), (size_t)(p_చివర - p));
            }
        }

        ఒకవేళ (p < p_చివర) {
            p += std::min((size_t)snprintf(p, p_చివర - p, "%s", tensor_buff_name(కణుపు.dst())), (size_t)(p_చివర - p));
        }
    }

    శూన్య format_op_పేర్లు(అక్షరం * str, size_t max_కొలత, స్థిరరాశి htp_opnode & కణుపు) {
        అక్షరం * p = str;
        అక్షరం * p_చివర = str + max_కొలత;
        స్వయంచాలక inputs = కణుపు.get_inputs();

        ఒకవేళ (!inputs.ఖాళీలు()) {
            ఒకవేళ (p < p_చివర) {
                p += std::min((size_t)snprintf(p, p_చివర - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_చివర - p));
            }

            కొరకు (size_t i = 1; i < inputs.కొలత(); i++) {
                ఒకవేళ (p < p_చివర) {
                    p += std::min((size_t)snprintf(p, p_చివర - p, " x "), (size_t)(p_చివర - p));
                }
                ఒకవేళ (p < p_చివర) {
                    p += std::min((size_t)snprintf(p, p_చివర - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_చివర - p));
                }
            }

            ఒకవేళ (p < p_చివర) {
                p += std::min((size_t)snprintf(p, p_చివర - p, " -> "), (size_t)(p_చివర - p));
            }
        }

        ఒకవేళ (p < p_చివర) {
            p += std::min((size_t)snprintf(p, p_చివర - p, "%s", కణుపు.dst()->name), (size_t)(p_చివర - p));
        }
    }
    శూన్య format_kernel_params(అక్షరం * str, size_t max_కొలత, స్థిరరాశి htp_opnode & కణుపు) {
        ఒకవేళ (కణుపు.opcode == HTP_OP_MUL_MAT || కణుపు.opcode == HTP_OP_MUL_MAT_ID ||
            కణుపు.opcode == HTP_OP_MUL_MAT_NX || కణుపు.opcode == HTP_OP_MUL_MAT_ID_NX ||
            కణుపు.opcode == HTP_OP_MUL_MAT_కూడు) {
            స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) కణుపు.kernel_params;
            స్థిరరాశి అక్షరం * మార్గం = "unknown";
            int32_t type = kparams->kernel_type;
            ఒకవేళ (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                మార్గం = (kparams->tile_కొలత > 0) ? "hmx-tiled" : "hmx-flat";
            } లేకపోతే ఒకవేళ (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                మార్గం = "hvx-flat";
            } లేకపోతే ఒకవేళ (type == HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస || type == HTP_MM_KERNEL_HVX_QUANT_ఖండం) {
                మార్గం = "hvx-tiled";
            }
            snprintf(str, max_కొలత, "%s vtcm %d", మార్గం, (పూర్ణాంకం) kparams->vtcm_కొలత);
        } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_FLASH_ATTN_EXT) {
            స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_fa_kernel_params *) కణుపు.kernel_params;
            స్థిరరాశి అక్షరం * మార్గం = "unknown";
            int32_t type = kparams->kernel_type;
            ఒకవేళ (type == HTP_FA_KERNEL_HMX) {
                మార్గం = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } లేకపోతే ఒకవేళ (type == HTP_FA_KERNEL_HVX) {
                మార్గం = "hvx";
            }
            snprintf(str, max_కొలత, "%s vtcm %d", మార్గం, (పూర్ణాంకం) kparams->vtcm_కొలత);
        } లేకపోతే ఒకవేళ (htp_op_is_unary(కణుపు.opcode)) {
            స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_unary_kernel_params *) కణుపు.kernel_params;
            snprintf(str, max_కొలత, "%s vtcm %d", kparams->నిలువు_tile ? "wide-row" : "row-block", (పూర్ణాంకం) kparams->vtcm_కొలత);
        } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_MDEV_సమూహం && కణుపు.కణుపు) {
            snprintf(str, max_కొలత, "idx %d count %d", (పూర్ణాంకం) కణుపు.కణుపు->op_params[0], (పూర్ణాంకం) కణుపు.dst()->ne[1]);
        } లేకపోతే ఒకవేళ ((కణుపు.opcode == HTP_OP_FENCE || కణుపు.opcode == HTP_OP_CPY_FENCE) && కణుపు.కణుపు) {
            snprintf(str, max_కొలత, "seq 0x%x", (uint32_t) కణుపు.కణుపు->op_params[0]);
        } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_ALLREDUCE && కణుపు.కణుపు) {
            snprintf(str, max_కొలత, "seq 0x%x -> 0x%x", (uint32_t) కణుపు.కణుపు->op_params[0], (uint32_t) కణుపు.కణుపు->op_params[1]);
        } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_SSM_CONV) {
            స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_ssm_conv_kernel_params *) కణుపు.kernel_params;
            snprintf(str, max_కొలత, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (పూర్ణాంకం) kparams->vtcm_కొలత);
        } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_SOFTMAX) {
            స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_softmax_kernel_params *) కణుపు.kernel_params;
            snprintf(str, max_కొలత, "k%d nth %d vtcm %d", (పూర్ణాంకం) kparams->kernel_id, (పూర్ణాంకం) kparams->n_దారాలు, (పూర్ణాంకం) kparams->vtcm_కొలత);
        } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_GATED_DELTA_NET) {
            స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_gdn_kernel_params *) కణుపు.kernel_params;
            స్థిరరాశి అక్షరం * మార్గం = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_కొలత, "%s-%s vtcm %u", మార్గం, kparams->kda ? "kda" : "scalar",
                     (గుర్తులేని పూర్ణాంకం) (kparams->vtcm_కొలత ? kparams->vtcm_కొలత : kparams->vtcm_చొప్పున_thread * kparams->n_దారాలు));
        } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_MUL || కణుపు.opcode == HTP_OP_కూడు || కణుపు.opcode == HTP_OP_కూడు_ID ||
                   కణుపు.opcode == HTP_OP_SUB || కణుపు.opcode == HTP_OP_DIV) {
            స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_ద్విమాన_kernel_params *) కణుపు.kernel_params;
            snprintf(str, max_కొలత, "vtcm %u", (గుర్తులేని పూర్ణాంకం) kparams->vtcm_కొలత);
        } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_ARGSORT || కణుపు.opcode == HTP_OP_శిఖరం_K) {
            స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_క్రమీకరించు_kernel_params *) కణుపు.kernel_params;
            snprintf(str, max_కొలత, "%s nth %d nchk %d chk %d vtcm %d",
                     కణుపు.opcode == HTP_OP_శిఖరం_K ? "top_k" : "argsort",
                     (పూర్ణాంకం) kparams->n_దారాలు, (పూర్ణాంకం) kparams->n_chunks,
                     (పూర్ణాంకం) kparams->chunk_elems, (పూర్ణాంకం) kparams->vtcm_కొలత);
        } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_GET_అడ్డువరుసలు) {
            స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params *) కణుపు.kernel_params;
            స్థిరరాశి అక్షరం * ktype_str = "unknown";
            ఎంపిక (kparams->kernel_type) {
                స్థితి HTP_GET_అడ్డువరుసలు_KERNEL_SAMETYPE: ktype_str = "sametype"; ఆపు;
                స్థితి HTP_GET_అడ్డువరుసలు_KERNEL_TILED:    ktype_str = "tiled"; ఆపు;
                స్థితి HTP_GET_అడ్డువరుసలు_KERNEL_FLAT:     ktype_str = "flat"; ఆపు;
            }
            snprintf(str, max_కొలత, "%s%s vtcm %u", ktype_str, kparams->n_దారాలు > 1 ? "-multi" : "", (గుర్తులేని పూర్ణాంకం) kparams->vtcm_కొలత);
        } లేకపోతే {
            snprintf(str, max_కొలత, "----");
        }

    }

    శూన్య format(స్థిరరాశి htp_opnode & కణుపు) {
        format_op_dims(dims, పరిమాణం(dims), కణుపు);
        format_op_strides(strides, పరిమాణం(strides), కణుపు);
        format_op_types(types, పరిమాణం(types), కణుపు);
        format_op_buffs(buffs, పరిమాణం(buffs), కణుపు);
        format_op_పేర్లు(పేర్లు, పరిమాణం(పేర్లు), కణుపు);
        format_kernel_params(kparams, పరిమాణం(kparams), కణుపు);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        పేర్లు[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(స్థిరరాశి htp_opnode & కణుపు) { format(కణుపు); }
};

#అయితే_ముగింపు // HTP_OPNODE_H
