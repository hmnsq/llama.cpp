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

struct htp_opnode {
    ggml_tensor * node   { nullptr };
    htp_op_code   opcode { HTP_OP_INVALID };
    int32_t       kernel_params[HTP_OP_MAX_KERN_PARAMS] {0};

    std::vector<ggml_tensor *>                fused;
    std::vector<std::shared_ptr<ggml_tensor>> dummy;

    std::vector<const ggml_tensor *> inputs;
    std::vector<const ggml_tensor *> outputs;
    std::string                      name;

    số_nguyên n_active_src(const ggml_tensor * t) const {
        nếu (!t) trả_về 0;
        cho (số_nguyên i = GGML_MAX_SRC - 1; i >= 0; i--) {
            nếu (t->src[i]) {
                trả_về i + 1;
            }
        }
        trả_về 0;
    }

    void init(ggml_tensor * node) {
        this->node = node;
        nếu (this->node) {
            nếu (this->node->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(this->node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                this->name = "MUL_MAT_HADAMARD";
                this->inputs.push_back(this->node->src[1]);
            } không_thì {
                this->name = ggml_op_desc(this->node);

                // Build inputs (preserving optional nullptrs)
                số_nguyên n_inputs = n_active_src(this->node);
                this->inputs.resize(n_inputs, nullptr);
                cho (số_nguyên i = 0; i < n_inputs; i++) {
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

    ggml_op             op()   const { trả_về node->op; }
    const ggml_tensor * src0() const { trả_về node->src[0]; }
    const ggml_tensor * src1() const { trả_về node->src[1]; }
    const ggml_tensor * dst()  const { trả_về outputs.empty() ? node : outputs.back(); }

    ggml_tensor * add_dummy(const ggml_tensor & t) {
        dummy.push_back(std::make_shared<ggml_tensor>(t));
        trả_về dummy.back().get();
    }

    void add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.push_back(t);

        name += "+";
        name += ggml_op_desc(t);

        nếu (extra_dst) {
            outputs.push_back(t);
        } không_thì {
            outputs.clear();
            outputs.push_back(t);
        }

        // Remove the newly fused intermediate output tensor t from inputs (if it was there)
        inputs.erase(std::remove(inputs.begin(), inputs.end(), t), inputs.end());

        // Append new inputs from t, preserving middle nullptrs
        số_nguyên n_inputs = n_active_src(t);
        cho (số_nguyên i = 0; i < n_inputs; i++) {
            const auto * src = t->src[i];
            nếu (!src) {
                inputs.push_back(nullptr);
            } không_thì nếu (src != node &&
                       std::find(fused.begin(), fused.end(), src) == fused.end() &&
                       std::find(inputs.begin(), inputs.end(), src) == inputs.end()) {
                inputs.push_back(src);
            }
        }
    }

    const std::vector<const ggml_tensor *> & get_inputs() const {
        trả_về inputs;
    }

    const std::vector<const ggml_tensor *> & get_outputs() const {
        trả_về outputs;
    }

    std::string op_name() const {
        trả_về name;
    }

    bool is_empty() const {
        trả_về ggml_op_is_empty(node->op);
    }

    bool stackable() const {
        switch (this->op()) {
            case GGML_OP_MUL_MAT:
                nếu (ggml_get_op_params_i32(this->node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    trả_về false;
                }
                trả_về ggml_is_quantized(this->src0()->type);
            case GGML_OP_MUL_MAT_ID:
                trả_về ggml_is_quantized(this->src0()->type);
            default:
                trả_về false;
        }
    }

    bool same_input(const htp_opnode& n) const {
        trả_về n.src1() == this->src1();
    }
};

struct htp_opformat {
    char strides[64 * GGML_MAX_SRC];
    char dims[64 * GGML_MAX_SRC];
    char types[16 * GGML_MAX_SRC];
    char buffs[64 * GGML_MAX_SRC];
    char names[64 * GGML_MAX_SRC];
    char kparams[128];

    số_nguyên format_tensor_dims(char * str, size_t max_size, const struct ggml_tensor * t) {
        nếu (!t) {
            trả_về snprintf(str, max_size, "NONE");
        }
        nếu (t->ne[2] == 1 && t->ne[3] == 1) {
            trả_về snprintf(str, max_size, "%d:%d", (số_nguyên) t->ne[0], (số_nguyên) t->ne[1]);
        } không_thì {
            trả_về snprintf(str, max_size, "%d:%d:%d:%d", (số_nguyên) t->ne[0], (số_nguyên) t->ne[1], (số_nguyên) t->ne[2], (số_nguyên) t->ne[3]);
        }
    }

    void format_op_dims(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        nếu (!inputs.empty()) {
            p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            cho (size_t i = 1; i < inputs.size(); i++) {
                nếu (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                nếu (p < p_end) {
                    p += std::min((size_t)format_tensor_dims(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            nếu (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        char self[64];
        format_tensor_dims(self, sizeof(self), node.dst());
        nếu (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    số_nguyên format_tensor_strides(char * str, size_t max_size, const struct ggml_tensor * t) {
        nếu (!t) {
            trả_về snprintf(str, max_size, "NONE");
        }
        const char * c = ggml_is_contiguous(t) ? "" : "!";

        nếu (t->ne[2] == 1 && t->ne[3] == 1) {
            trả_về snprintf(str, max_size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } không_thì {
            trả_về snprintf(str, max_size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    void format_op_strides(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        nếu (!inputs.empty()) {
            p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[0]), (size_t)(p_end - p));

            cho (size_t i = 1; i < inputs.size(); i++) {
                nếu (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                nếu (p < p_end) {
                    p += std::min((size_t)format_tensor_strides(p, p_end - p, inputs[i]), (size_t)(p_end - p));
                }
            }

            nếu (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        char self[64];
        format_tensor_strides(self, sizeof(self), node.dst());
        nếu (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", self), (size_t)(p_end - p));
        }
    }

    void format_op_types(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        nếu (!inputs.empty()) {
            nếu (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? ggml_type_name(inputs[0]->type) : "NONE"), (size_t)(p_end - p));
            }

            cho (size_t i = 1; i < inputs.size(); i++) {
                nếu (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                nếu (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? ggml_type_name(inputs[i]->type) : "NONE"), (size_t)(p_end - p));
                }
            }

            nếu (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        nếu (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", ggml_type_name(node.dst()->type)), (size_t)(p_end - p));
        }
    }

    const char * tensor_buff_name(const struct ggml_tensor * t) {
        nếu (t && t->buffer) {
            trả_về ggml_backend_buffer_name(t->buffer);
        }
        trả_về "NONE";
    }

    void format_op_buffs(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        nếu (!inputs.empty()) {
            nếu (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[0])), (size_t)(p_end - p));
            }

            cho (size_t i = 1; i < inputs.size(); i++) {
                nếu (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                nếu (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(inputs[i])), (size_t)(p_end - p));
                }
            }

            nếu (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        nếu (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", tensor_buff_name(node.dst())), (size_t)(p_end - p));
        }
    }

    void format_op_names(char * str, size_t max_size, const htp_opnode & node) {
        char * p = str;
        char * p_end = str + max_size;
        auto inputs = node.get_inputs();

        nếu (!inputs.empty()) {
            nếu (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[0] ? inputs[0]->name : "NONE"), (size_t)(p_end - p));
            }

            cho (size_t i = 1; i < inputs.size(); i++) {
                nếu (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, " x "), (size_t)(p_end - p));
                }
                nếu (p < p_end) {
                    p += std::min((size_t)snprintf(p, p_end - p, "%s", inputs[i] ? inputs[i]->name : "NONE"), (size_t)(p_end - p));
                }
            }

            nếu (p < p_end) {
                p += std::min((size_t)snprintf(p, p_end - p, " -> "), (size_t)(p_end - p));
            }
        }

        nếu (p < p_end) {
            p += std::min((size_t)snprintf(p, p_end - p, "%s", node.dst()->name), (size_t)(p_end - p));
        }
    }
    void format_kernel_params(char * str, size_t max_size, const htp_opnode & node) {
        nếu (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID ||
            node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX ||
            node.opcode == HTP_OP_MUL_MAT_ADD) {
            const auto * kparams = (const struct htp_mm_kernel_params *) node.kernel_params;
            const char * path = "unknown";
            int32_t type = kparams->kernel_type;
            nếu (type == HTP_MM_KERNEL_HMX_2D || type == HTP_MM_KERNEL_HMX_BATCHED) {
                path = (kparams->tile_size > 0) ? "hmx-tiled" : "hmx-flat";
            } không_thì nếu (type == HTP_MM_KERNEL_HVX_F16_F16_VTCM || type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
                       type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM) {
                path = "hvx-flat";
            } không_thì nếu (type == HTP_MM_KERNEL_HVX_QUANT_ROW || type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
                path = "hvx-tiled";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (số_nguyên) kparams->vtcm_size);
        } không_thì nếu (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
            const auto * kparams = (const struct htp_fa_kernel_params *) node.kernel_params;
            const char * path = "unknown";
            int32_t type = kparams->kernel_type;
            nếu (type == HTP_FA_KERNEL_HMX) {
                path = kparams->u.hmx.pipeline ? "hmx-pipe" : "hmx-seq";
            } không_thì nếu (type == HTP_FA_KERNEL_HVX) {
                path = "hvx";
            }
            snprintf(str, max_size, "%s vtcm %d", path, (số_nguyên) kparams->vtcm_size);
        } không_thì nếu (htp_op_is_unary(node.opcode)) {
            const auto * kparams = (const struct htp_unary_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->col_tile ? "wide-row" : "row-block", (số_nguyên) kparams->vtcm_size);
        } không_thì nếu (node.opcode == HTP_OP_MDEV_GROUP && node.node) {
            snprintf(str, max_size, "idx %d count %d", (số_nguyên) node.node->op_params[0], (số_nguyên) node.dst()->ne[1]);
        } không_thì nếu ((node.opcode == HTP_OP_FENCE || node.opcode == HTP_OP_CPY_FENCE) && node.node) {
            snprintf(str, max_size, "seq 0x%x", (uint32_t) node.node->op_params[0]);
        } không_thì nếu (node.opcode == HTP_OP_ALLREDUCE && node.node) {
            snprintf(str, max_size, "seq 0x%x -> 0x%x", (uint32_t) node.node->op_params[0], (uint32_t) node.node->op_params[1]);
        } không_thì nếu (node.opcode == HTP_OP_SSM_CONV) {
            const auto * kparams = (const struct htp_ssm_conv_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s vtcm %d", kparams->n_t == 1 ? "decode" : "prefill", (số_nguyên) kparams->vtcm_size);
        } không_thì nếu (node.opcode == HTP_OP_SOFTMAX) {
            const auto * kparams = (const struct htp_softmax_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "k%d nth %d vtcm %d", (số_nguyên) kparams->kernel_id, (số_nguyên) kparams->n_threads, (số_nguyên) kparams->vtcm_size);
        } không_thì nếu (node.opcode == HTP_OP_GATED_DELTA_NET) {
            const auto * kparams = (const struct htp_gdn_kernel_params *) node.kernel_params;
            const char * path = (kparams->kernel_type == HTP_GDN_KERNEL_HMX_CHUNKED) ? "hmx-chunked" : "hvx-recurrent";
            snprintf(str, max_size, "%s-%s vtcm %u", path, kparams->kda ? "kda" : "scalar",
                     (unsigned số_nguyên) (kparams->vtcm_size ? kparams->vtcm_size : kparams->vtcm_per_thread * kparams->n_threads));
        } không_thì nếu (node.opcode == HTP_OP_MUL || node.opcode == HTP_OP_ADD || node.opcode == HTP_OP_ADD_ID ||
                   node.opcode == HTP_OP_SUB || node.opcode == HTP_OP_DIV) {
            const auto * kparams = (const struct htp_binary_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "vtcm %u", (unsigned số_nguyên) kparams->vtcm_size);
        } không_thì nếu (node.opcode == HTP_OP_ARGSORT || node.opcode == HTP_OP_TOP_K) {
            const auto * kparams = (const struct htp_sort_kernel_params *) node.kernel_params;
            snprintf(str, max_size, "%s nth %d nchk %d chk %d vtcm %d",
                     node.opcode == HTP_OP_TOP_K ? "top_k" : "argsort",
                     (số_nguyên) kparams->n_threads, (số_nguyên) kparams->n_chunks,
                     (số_nguyên) kparams->chunk_elems, (số_nguyên) kparams->vtcm_size);
        } không_thì nếu (node.opcode == HTP_OP_GET_ROWS) {
            const auto * kparams = (const struct htp_get_rows_kernel_params *) node.kernel_params;
            const char * ktype_str = "unknown";
            switch (kparams->kernel_type) {
                case HTP_GET_ROWS_KERNEL_SAMETYPE: ktype_str = "sametype"; break;
                case HTP_GET_ROWS_KERNEL_TILED:    ktype_str = "tiled"; break;
                case HTP_GET_ROWS_KERNEL_FLAT:     ktype_str = "flat"; break;
            }
            snprintf(str, max_size, "%s%s vtcm %u", ktype_str, kparams->n_threads > 1 ? "-multi" : "", (unsigned số_nguyên) kparams->vtcm_size);
        } không_thì {
            snprintf(str, max_size, "----");
        }

    }

    void format(const htp_opnode & node) {
        format_op_dims(dims, sizeof(dims), node);
        format_op_strides(strides, sizeof(strides), node);
        format_op_types(types, sizeof(types), node);
        format_op_buffs(buffs, sizeof(buffs), node);
        format_op_names(names, sizeof(names), node);
        format_kernel_params(kparams, sizeof(kparams), node);
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
