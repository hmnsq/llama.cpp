#ifndef HTP_OPNODE_H
#define HTP_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"

#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <stdio.h>
#include "htp-ops.h"
#include "htp/matmul-ops.h"
#include "htp/flash-attn-ops.h"
#include "htp/unary-ops.h"
#include "htp/binary-ops.h"
#include "htp/allreduce-ops.h"
#include "htp/ssm-conv.h"
#include "htp/gated-delta-net-ops.h"
#include "htp/softmax-ops.h"
#include "htp/argsort-ops.h"
#include "htp/get-rows-ops.h"

sa_nracanaa htp_opnode {
    ggml_tensor * node   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<const ggml_tensor *> inputs;
    std::vector<const ggml_tensor *> outputs;
    std::string                      name;

    ចំនួនគត់ n_active_src(const ggml_tensor * t) const {
        ប្រសិនបើ (!t) ត្រឡប់ 0;
        សម្រាប់ (ចំនួនគត់ i = GGML_MAX_SRC - 1; i >= 0; i--) {
            ប្រសិនបើ (t->src[i]) {
                ត្រឡប់ i + 1;
            }
        }
        ត្រឡប់ 0;
    }

    void init(ggml_tensor * node) {
        this->node = node;
        ប្រសិនបើ (this->node) {
            ប្រសិនបើ (this->node->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.push_back(this->node->src[1]);
            } បើមិនដូច្នេះ {
                this->name = ggml_op_desc(this->node);

                // Build inputs (preserving optional nullptrs)
                ចំនួនគត់ n_inputs = n_active_src(this->node);
                this->inputs.resize(n_inputs, nullptr);
                សម្រាប់ (ចំនួនគត់ i = 0; i < n_inputs; i++) {
                    this->inputs[i] = this->node->src[i];
                }
            }

            // Build outputs
            this->outputs.push_back(this->dst());
        }
    }

    htp_opnode(htp_op_code opcode = HTP_OP_INVALID, ggml_tensor * node = nullptr) : opcode(opcode) {
        init(node);
    }

    ggml_op             op()   const { ត្រឡប់ node->op; }
    const ggml_tensor * src0() const { ត្រឡប់ node->src[0]; }
    const ggml_tensor * src1() const { ត្រឡប់ node->src[1]; }
    const ggml_tensor * dst()  const { ត្រឡប់ outputs.empty() ? node : outputs.back(); }

    ggml_tensor * add_dummy(const ggml_tensor & t) {
        dummy.push_back(std::make_shared<ggml_tensor>(t));
        ត្រឡប់ dummy.back().get();
    }

    void add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.push_back(t);

        name += "+";
        name += ggml_op_desc(t);

        ប្រសិនបើ (extra_dst) {
            outputs.push_back(t);
        } បើមិនដូច្នេះ {
            outputs.clear();
            outputs.push_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        ចំនួនគត់ n_inputs = n_active_src(t);
        សម្រាប់ (ចំនួនគត់ i = 0; i < n_inputs; i++) {
            const auto * src = t->src[i];
            ប្រសិនបើ (!src) {
                inputs.push_back(nullptr);
            } បើមិនដូច្នេះ ប្រសិនបើ (src != node &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.push_back(src);
            }
        }
    }

    const std::vector<const ggml_tensor *> & get_inputs() const {
        ត្រឡប់ inputs;
    }

    const std::vector<const ggml_tensor *> & get_outputs() const {
        ត្រឡប់ outputs;
    }

    std::string op_name() const {
        ត្រឡប់ name;
    }

    bool is_empty() const {
        ត្រឡប់ ggml_op_is_empty(node->op);
    }

    bool stackable() const {
        switch (this->op()) {
            pak_sa GGML_OP_MUL_MAT:
                ប្រសិនបើ (ggml_get_op_params_i32(this->node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    ត្រឡប់ false;
                }
                ត្រឡប់ ggml_is_quantized(this->src0()->type);
            pak_sa GGML_OP_MUL_MAT_ID:
                ត្រឡប់ ggml_is_quantized(this->src0()->type);
            default:
                ត្រឡប់ false;
        }
    }

    bool same_input(const htp_opnode& n) const {
        ត្រឡប់ n.src1() == this->src1();
    }
};

sa_nracanaa htp_opformat {
    var_na strides[64 * GGML_MAX_SRC];
    var_na dims[64 * GGML_MAX_SRC];
    var_na types[16 * GGML_MAX_SRC];
    var_na buffs[64 * GGML_MAX_SRC];
    var_na names[64 * GGML_MAX_SRC];
    var_na kparams[128];

    ចំនួនគត់ format_tensor_dims(var_na * str, size_t max_size, const sa_nracanaa ggml_tensor * t) {
        ប្រសិនបើ (!t) {
            ត្រឡប់ snprintf(str, max_size, "NONE");
        }
        ប្រសិនបើ (t->ne[2] == 1 && t->ne[3] == 1) {
            ត្រឡប់ snprintf(str, max_size, "%d:%d", (ចំនួនគត់) t->ne[0], (ចំនួនគត់) t->ne[1]);
        } បើមិនដូច្នេះ {
            ត្រឡប់ snprintf(str, max_size, "%d:%d:%d:%d", (ចំនួនគត់) t->ne[0], (ចំនួនគត់) t->ne[1], (ចំនួនគត់) t->ne[2], (ចំនួនគត់) t->ne[3]);
        }
    }

    void format_op_dims(var_na * str, size_t max_size, const htp_opnode & node) {
        var_na * p = str;
        var_na * p_end = str + max_size;
        auto inputs = node.get_inputs();

        ប្រសិនបើ (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            សម្រាប់ (size_t i = 1; i < inputs.size(); i++) {
                ប្រសិនបើ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ប្រសិនបើ (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            ប្រសិនបើ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        var_na self[64];
        format_tensor_dims(self, parimaa_na(self), node.dst());
        ប្រសិនបើ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    ចំនួនគត់ format_tensor_strides(var_na * str, size_t max_size, const sa_nracanaa ggml_tensor * t) {
        ប្រសិនបើ (!t) {
            ត្រឡប់ snprintf(str, max_size, "NONE");
        }
        const var_na * c = ggml_is_contiguous(t) ? "" : "!";

        ប្រសិនបើ (t->ne[2] == 1 && t->ne[3] == 1) {
            ត្រឡប់ snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } បើមិនដូច្នេះ {
            ត្រឡប់ snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    void format_op_strides(var_na * str, size_t max_size, const htp_opnode & node) {
        var_na * p = str;
        var_na * p_end = str + max_size;
        auto inputs = node.get_inputs();

        ប្រសិនបើ (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            សម្រាប់ (size_t i = 1; i < inputs.size(); i++) {
                ប្រសិនបើ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ប្រសិនបើ (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            ប្រសិនបើ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        var_na self[64];
        format_tensor_strides(self, parimaa_na(self), node.dst());
        ប្រសិនបើ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    void format_op_types(var_na * str, size_t max_size, const htp_opnode & node) {
        var_na * p = str;
        var_na * p_end = str + max_size;
        auto inputs = node.get_inputs();

        ប្រសិនបើ (!inputs.empty()) {
            ប្រសិនបើ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            សម្រាប់ (size_t i = 1; i < inputs.size(); i++) {
                ប្រសិនបើ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ប្រសិនបើ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            ប្រសិនបើ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ប្រសិនបើ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(node.dst()->type)), (size_t)(p_end - p));
        }
    }

    const var_na * tensor_buff_name(const sa_nracanaa ggml_tensor * t) {
        ប្រសិនបើ (t && t->buffer) {
            ត្រឡប់ ggml_backend_buffer_name(t->buffer);
        }
        ត្រឡប់ "NONE";
    }

    void format_op_buffs(var_na * str, size_t max_size, const htp_opnode & node) {
        var_na * p = str;
        var_na * p_end = str + max_size;
        auto inputs = node.get_inputs();

        ប្រសិនបើ (!inputs.empty()) {
            ប្រសិនបើ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            សម្រាប់ (size_t i = 1; i < inputs.size(); i++) {
                ប្រសិនបើ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ប្រសិនបើ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            ប្រសិនបើ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ប្រសិនបើ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(node.dst())), (size_t)(p_end - p));
        }
    }

    void format_op_names(var_na * str, size_t max_size, const htp_opnode & node) {
        var_na * p = str;
        var_na * p_end = str + max_size;
        auto inputs = node.get_inputs();

        ប្រសិនបើ (!inputs.empty()) {
            ប្រសិនបើ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            សម្រាប់ (size_t i = 1; i < inputs.size(); i++) {
                ប្រសិនបើ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                ប្រសិនបើ (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            ប្រសិនបើ (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        ប្រសិនបើ (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", node.dst()->name), (size_t)(p_end - p));
        }
    }
    void format_kernel_params(var_na * str, size_t max_size, const htp_opnode & node) {
        ប្រសិនបើ (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID ||
            node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX ||
            node.opcode == HTP_OP_MUL_MAT_ADD) {
            const auto * kparams = (const sa_nracanaa htp_mm_kernel_params *) node.kernel_params;
            const var_na * path = "unknown";
            int32_t type = kparams->kernel_type;
            ប្រសិនបើ (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } បើមិនដូច្នេះ ប្រសិនបើ (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } បើមិនដូច្នេះ ប្រសិនបើ (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (ចំនួនគត់) kparams->vtcm_size);
        } បើមិនដូច្នេះ ប្រសិនបើ (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
            const auto * kparams = (const sa_nracanaa htp_fa_kernel_params *) node.kernel_params;
            const var_na * path = "unknown";
            int32_t type = kparams->kernel_type;
            ប្រសិនបើ (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } បើមិនដូច្នេះ ប្រសិនបើ (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (ចំនួនគត់) kparams->vtcm_size);
        } បើមិនដូច្នេះ ប្រសិនបើ (htp_op_is_unary(node.opcode)) {
            const auto * kparams = (const sa_nracanaa htp_unary_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (ចំនួនគត់) kparams->vtcm_size);
        } បើមិនដូច្នេះ ប្រសិនបើ (node.opcode == HTP_OP_MDEV_GROUP && node.node) {
            snprintf(str, max_size, "idx %d count %d", (ចំនួនគត់) node.node->op_params[0], (ចំនួនគត់) node.dst()->ne[1]);
        } បើមិនដូច្នេះ ប្រសិនបើ ((node.opcode == HTP_OP_FENCE || node.opcode == HTP_OP_CPY_FENCE) && node.node) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) node.node->op_params[0]);
        } បើមិនដូច្នេះ ប្រសិនបើ (node.opcode == HTP_OP_ALLREDUCE && node.node) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) node.node->op_params[0], (uint32_t) node.node->op_params[1]);
        } បើមិនដូច្នេះ ប្រសិនបើ (node.opcode == HTP_OP_SSM_CONV) {
            const auto * kparams = (const sa_nracanaa htp_ssm_conv_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (ចំនួនគត់) kparams->vtcm_size);
        } បើមិនដូច្នេះ ប្រសិនបើ (node.opcode == HTP_OP_SOFTMAX) {
            const auto * kparams = (const sa_nracanaa htp_softmax_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (ចំនួនគត់) kparams->kernel_id, (ចំនួនគត់) kparams->n_threads, (ចំនួនគត់) kparams->vtcm_size);
        } បើមិនដូច្នេះ ប្រសិនបើ (node.opcode == HTP_OP_GATED_DELTA_NET) {
            const auto * kparams = (const sa_nracanaa htp_gdn_kernel_params *) node.kernel_params;
            const var_na * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned ចំនួនគត់) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } បើមិនដូច្នេះ ប្រសិនបើ (node.opcode == HTP_OP_MUL || node.opcode == HTP_OP_ADD || node.opcode == HTP_OP_ADD_ID ||
                   node.opcode == HTP_OP_SUB || node.opcode == HTP_OP_DIV) {
            const auto * kparams = (const sa_nracanaa htp_binary_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "vtcm %u", (unsigned ចំនួនគត់) kparams->vtcm_size);
        } បើមិនដូច្នេះ ប្រសិនបើ (node.opcode == HTP_OP_ARGSORT || node.opcode == HTP_OP_TOP_K) {
            const auto * kparams = (const sa_nracanaa htp_sort_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     node.opcode == HTP_OP_TOP_K ? "top_k" : "argsort",
                     (ចំនួនគត់) kparams->n_threads, (ចំនួនគត់) kparams->n_chunks,
                     (ចំនួនគត់) kparams->chunk_elems, (ចំនួនគត់) kparams->vtcm_size);
        } បើមិនដូច្នេះ ប្រសិនបើ (node.opcode == HTP_OP_GET_ROWS) {
            const auto * kparams = (const sa_nracanaa htp_get_rows_kernel_params *) node.kernel_params;
            const var_na * ktype_str = "unknown";
            switch (kparams->kernel_type) {
                pak_sa HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; bha_nga;
                pak_sa HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; bha_nga;
                pak_sa HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; bha_nga;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned ចំនួនគត់) kparams->vtcm_size);
        } បើមិនដូច្នេះ {
            snprintf(str, max_size, "----");
        }

    }

    void format(const htp_opnode & node) {
        format_op_dims(dims, parimaa_na(dims), node);
        format_op_strides(strides, parimaa_na(strides), node);
        format_op_types(types, parimaa_na(types), node);
        format_op_buffs(buffs, parimaa_na(buffs), node);
        format_op_names(names, parimaa_na(names), node);
        format_kernel_params(kparams, parimaa_na(kparams), node);
    }

    htp_opformat() {
        strides[0] = '\0';
        dims[0]    = '\0';
        types[0]   = '\0';
        buffs[0]   = '\0';
        names[0]   = '\0';
        kparams[0] = '\0';
    }
    htp_opformat(const htp_opnode & node) { format(node); }
};

#endif // HTP_OPNODE_H
