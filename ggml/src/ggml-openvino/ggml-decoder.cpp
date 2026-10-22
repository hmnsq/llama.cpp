#شامل "ggml-decoder.h"

#شامل "ggml-impl.h"
#شامل "ggml-openvino-extra.h"
#شامل "ggml-openvino.h"
#شامل "ggml-quants.h"
#شامل "ggml.h"
#شامل "utils.h"

#شامل <algorithm>
#شامل <cassert>
#شامل <cstddef>
#شامل <cstdint>
#شامل <cstdlib>
#شامل <fstream>
#شامل <iomanip>
#شامل <map>
#شامل <memory>
#شامل <mutex>
#شامل <openvino/core/dimension.hpp>
#شامل <openvino/core/except.hpp>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/partial_shape.hpp>
#شامل <openvino/core/type/bfloat16.hpp>
#شامل <openvino/core/type/element_type.hpp>
#شامل <openvino/core/type/float16.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/runtime/tensor.hpp>
#شامل <ostream>
#شامل <set>
#شامل <stdexcept>
#شامل <string>
#شامل <cstring>
#شامل <unordered_map>
#شامل <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             منطقی is_static,
                             منطقی is_stateful,
                             منطقی model_is_splitted,
                             منطقی is_prefill,
                             صحیح prefill_chunk_سائز) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(جھوٹ),
    m_prefill_chunk_سائز(prefill_chunk_سائز),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    ساکن منطقی printed_پتہ_map = جھوٹ;
    اگر (!printed_پتہ_map) {
        اگر (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_پتہ_map = سچ;
            print_tensor_پتہ_map(cgraph);
        }
    }

    validate_cgraph();

    سیٹ_داخلہ_output();
    compute_گرہ_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    برائے (صحیح گرہ_n = 0; گرہ_n < cgraph->n_nodes; گرہ_n++) {
        m_گرہ_info_list[گرہ_n].گرہ_op_case = compute_op_case(m_گرہ_info_list[گرہ_n].گرہ);
        m_گرہ_info_list[گرہ_n].گرہ_op_type = compute_op_type(m_گرہ_info_list[گرہ_n].گرہ);
    }

    جوڑو_extra_inputs();
}

خالی GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_گرہ_info_list.clear();
    سیٹ_داخلہ_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
منطقی GgmlOvDecoder::is_bound_مقصد(ثابت ggml_cgraph * cgraph) ثابت {
    واپس m_cgraph == cgraph && cgraph->n_nodes > 0 && m_گرہ_info_list.size() == (size_t) cgraph->n_nodes &&
           m_گرہ_info_list.front().گرہ == cgraph->nodes[0] &&
           m_گرہ_info_list.back().گرہ == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = سچ;
    سیٹ_داخلہ_output();
    compute_model_inputs();
    compute_model_outputs();
    برائے (صحیح گرہ_n = 0; گرہ_n < cgraph->n_nodes; گرہ_n++) {
        m_گرہ_info_list[گرہ_n].گرہ_op_case = compute_op_case(m_گرہ_info_list[گرہ_n].گرہ);
        m_گرہ_info_list[گرہ_n].گرہ_op_type = compute_op_type(m_گرہ_info_list[گرہ_n].گرہ);
    }
}

نامخلا {
منطقی is_inplace_op(ثابت ggml_tensor * گرہ) {
    واپس گرہ->op == GGML_OP_سیٹ_صفیں || گرہ->op == GGML_OP_CPY || (گرہ->op == GGML_OP_SCALE && گرہ->view_src);
}

منطقی is_same_shape(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        اگر (a->ne[i] != b->ne[i]) {
            واپس جھوٹ;
        }
    }
    واپس سچ;
}

منطقی is_conv_حالتیں_all_tensor(ثابت ggml_tensor * tensor) {
    واپس tensor != خالی_اشاریہ && لڑی_موازنہ(tensor->name, "conv_states_all", لڑی_لمبائی("conv_states_all")) == 0;
}

منطقی is_بھرا_single_خانہ_writeback(ثابت ggml_tensor * گرہ) {
    واپس گرہ->view_src != خالی_اشاریہ && گرہ->view_src->ne[1] == 1 && گرہ->src[1] != خالی_اشاریہ &&
           گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src == گرہ->view_src &&
           گرہ->src[1]->view_offs == 0 && ggml_nbytes(گرہ->src[1]) == ggml_nbytes(گرہ->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
منطقی is_moe_expert_مجموع_جوڑو(ثابت ggml_tensor * گرہ) {
    std::vector<ثابت ggml_tensor *> leaves;
    ثابت ggml_tensor * cur = گرہ;
    جب (cur->op == GGML_OP_جوڑو) {
        اگر (cur->src[0] == خالی_اشاریہ || cur->src[1] == خالی_اشاریہ) {
            واپس جھوٹ;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ثابت ggml_tensor * بنیاد = خالی_اشاریہ;
    std::set<int64_t> plane_indices;
    برائے (ثابت ggml_tensor * برگ : leaves) {
        اگر (برگ->op != GGML_OP_VIEW || برگ->src[0] == خالی_اشاریہ) {
            واپس جھوٹ;
        }
        ثابت ggml_tensor * برگ_بنیاد = برگ->src[0];
        اگر (بنیاد == خالی_اشاریہ) {
            بنیاد = برگ_بنیاد;
        } ورنہ اگر (برگ_بنیاد != بنیاد) {
            واپس جھوٹ;
        }
        اگر (برگ->ne[0] != بنیاد->ne[0] || برگ->ne[1] != بنیاد->ne[2] || برگ->ne[2] != 1 || برگ->ne[3] != 1 ||
            برگ->nb[1] != بنیاد->nb[2]) {
            واپس جھوٹ;
        }
        اگر (بنیاد->nb[1] == 0 || برگ->view_offs % بنیاد->nb[1] != 0) {
            واپس جھوٹ;
        }
        int64_t plane = static_cast<int64_t>(برگ->view_offs / بنیاد->nb[1]);
        اگر (plane < 0 || plane >= بنیاد->ne[1] || !plane_indices.insert(plane).second) {
            واپس جھوٹ;
        }
    }

    واپس بنیاد != خالی_اشاریہ && بنیاد->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(بنیاد->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor) {
    اگر (tensor == خالی_اشاریہ) {
        واپس "";
    }
    اگر ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, خالی_اشاریہ)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ثابت خودکار * گرہ = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        اگر (گرہ != cgraph->nodes + cgraph->n_nodes) {
            واپس std::string(tensor->name) + "#n" + std::to_string(گرہ - cgraph->nodes);
        }
        ثابت خودکار * برگ = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        اگر (برگ != cgraph->leafs + cgraph->n_leafs) {
            واپس std::string(tensor->name) + "#l" + std::to_string(برگ - cgraph->leafs);
        }
    }
    واپس tensor->name;
}

ساکن std::string get_tensor_ov_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor) {
    واپس GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

ساکن std::string get_tensor_گراف_داخلہ_ov_name(ثابت GgmlOvDecoder * decoder,
                                                  ثابت ggml_cgraph * cgraph,
                                                  ثابت ggml_tensor * tensor,
                                                  ثابت ggml_tensor * op) {
    اگر (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        واپس "inp_pos";
    }
    اگر (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        واپس "embd";
    }
    اگر (GgmlOvDecoder::is_inp_نقاب(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ثابت منطقی is_swa = decoder->is_swa_نقاب(tensor);
        اگر (decoder->is_stateful()) {
            واپس is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        اگر (is_swa) {
            واپس get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    واپس get_tensor_ov_name(cgraph, tensor);
}

خالی GgmlOvDecoder::سیٹ_داخلہ_output() {
    برائے (صحیح گرہ_n = 0; گرہ_n < m_cgraph->n_nodes; گرہ_n++) {
        خودکار * گرہ = m_cgraph->nodes[گرہ_n];

        گرہ_Info current_گرہ_info;
        خودکار گرہ_name = get_tensor_ov_name(m_cgraph, گرہ);

        current_گرہ_info.گرہ = گرہ;
        current_گرہ_info.گرہ_name = گرہ_name;
        current_گرہ_info.گرہ_op_case = 0;
        current_گرہ_info.ڈیٹا_addr = گرہ->ڈیٹا;

        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            خودکار * src = گرہ->src[i];
            اگر (src == خالی_اشاریہ) {
                جاری;
            }
            خودکار src_name = get_tensor_ov_name(m_cgraph, src);
            اگر (src->flags & GGML_TENSOR_FLAG_داخلہ) {
                src_name = get_tensor_گراف_داخلہ_ov_name(یہ, m_cgraph, src, گرہ);
            }
            current_گرہ_info.گرہ_inputs[src_name] = src;
            current_گرہ_info.گرہ_inputs_نام.push_back(src_name);

            اگر (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_گرہ_info.گرہ_inputs_views[src_name])> view_chain;
                خودکار * current = src;

                جب (current != خالی_اشاریہ) {
                    خودکار current_name = get_tensor_ov_name(m_cgraph, current);
                    اگر (current->flags & GGML_TENSOR_FLAG_داخلہ) {
                        current_name = get_tensor_گراف_داخلہ_ov_name(یہ, m_cgraph, current, گرہ);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    اگر (current->src[0] != خالی_اشاریہ && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } ورنہ {
                        توڑو;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_گرہ_info.گرہ_inputs_views[src_name] = view_chain;
            }
        }

        m_گرہ_info_list.push_back(current_گرہ_info);
    }
}

صحیح GgmlOvDecoder::compute_op_case(ثابت ggml_tensor * گرہ) ثابت {
    صحیح op_case = 0;
    انتخاب (گرہ->op) {
    حالت GGML_OP_RESHAPE: {
        اگر (m_naive) {
            توڑو;
        }
        خودکار name = std::string(گرہ->name);
        خودکار * src = گرہ->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        منطقی recurrent_sequence = جھوٹ;
        برائے (صحیح i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ثابت خودکار * صارف = m_cgraph->nodes[i];
            اگر (صارف->op == GGML_OP_MUL_MAT_ID && صارف->src[1] == گرہ) {
                واپس 1;
            } ورنہ اگر (صارف->op == GGML_OP_SSM_CONV) {
                ثابت خودکار * concat = صارف->src[0];
                اگر (concat->op == GGML_OP_CONCAT) {
                    ثابت خودکار * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == گرہ;
                }
            } ورنہ اگر (صارف->op == GGML_OP_UNARY && ggml_get_unary_op(صارف) == GGML_UNARY_OP_SOFTPLUS) {
                ثابت خودکار * biased = صارف->src[0];
                recurrent_sequence = biased->op == GGML_OP_جوڑو && biased->src[0] == گرہ;
            }
        }
        اگر (recurrent_sequence && گرہ->ne[0] == src->ne[0] && گرہ->ne[3] == 1) {
            واپس 6;
        }
        اگر (گرہ->ne[0] == src->ne[0] && گرہ->ne[2] == 1 && گرہ->ne[3] == 1) {
            واپس 5;
        }
        اگر (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == گرہ->ne[0] && src->src[0]->ne[1] == گرہ->ne[1]) {
            op_case = 4;
        } ورنہ اگر (گرہ->ne[0] * گرہ->ne[1] == src->ne[0]) {
            op_case = 1;
        } ورنہ اگر (src->ne[0] * src->ne[1] == گرہ->ne[0]) {
            op_case = 2;
            اگر (src->ne[2] * src->ne[3] == گرہ->ne[1]) {
                op_case = 5;
            }
        } ورنہ اگر (گرہ->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == گرہ->ne[1]) {
            op_case = 3;
        } ورنہ اگر (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } ورنہ اگر (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } ورنہ اگر (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        اگر (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            ثابت خودکار * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            منطقی recurrent = gather->op == GGML_OP_GET_صفیں && is_recurrent_cache(gather->src[0]);
            برائے (صحیح i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ثابت خودکار * صارف = m_cgraph->nodes[i];
                اگر (صارف->op == GGML_OP_GATED_DELTA_NET) {
                    برائے (صحیح j : {3, 4}) {
                        ثابت خودکار * gate = صارف->src[j];
                        اگر (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == گرہ;
                    }
                } ورنہ اگر (صارف->op == GGML_OP_MUL) {
                    برائے (صحیح j = 0; j < 2; ++j) {
                        ثابت خودکار * gate = صارف->src[j];
                        ثابت خودکار * norm = صارف->src[1 - j];
                        اگر (gate->op != GGML_OP_UNARY || gate->src[0] != گرہ) {
                            جاری;
                        }
                        اگر (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            اگر (recurrent) {
                op_case = 9;
            }
        }
        توڑو;
    }
    حالت GGML_OP_ترتیب_بدلو: {
        اگر (گرہ->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } ورنہ اگر (گرہ->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(گرہ->view_src->name);
            صحیح layer = نکالو_layer_from_name(src_name).value();
            اگر (ggml_is_contiguous(گرہ->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                اگر (!is_swa_layer(layer)) {
                    op_case = 3;
                } ورنہ {
                    op_case = 4;
                }
            } ورنہ {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                اگر (!is_swa_layer(layer)) {
                    op_case = 5;
                } ورنہ {
                    op_case = 6;
                }
            }
        } ورنہ {
            // rope'ed query tensor
            op_case = 2;
        }
        توڑو;
    }
    حالت GGML_OP_MUL_MAT: {
        اگر (گرہ->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        توڑو;
    }
    حالت GGML_OP_GET_صفیں: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        اگر (گرہ->src[0]->op == GGML_OP_RESHAPE && گرہ->src[0]->src[0] != خالی_اشاریہ &&
            is_recurrent_cache(گرہ->src[0]->src[0])) {
            ثابت منطقی single_خانہ = گرہ->src[0]->src[0]->ne[1] == 1;
            اگر (گرہ->src[1]->op == GGML_OP_VIEW) {
                op_case = (گرہ->src[1]->view_offs == 0 ? 1 : 2) + (single_خانہ ? 2 : 0);
            } ورنہ اگر (single_خانہ) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        توڑو;
    }
    حالت GGML_OP_ROPE: {
        ثابت صحیح mode = گرہ->op_params[2];
        انتخاب (mode) {
        حالت GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            توڑو;
        }
        حالت GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            توڑو;
        }
        حالت GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            توڑو;
        }
        حالت GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            توڑو;
        }
        طےشدہ:
            op_case = 0;
            توڑو;
        }
        توڑو;
    }
    حالت GGML_OP_VIEW: {
        اگر (!m_model_params.has_rs_rollback && گرہ->src[0] != خالی_اشاریہ &&
            گرہ->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            توڑو;
        }
        اگر (m_is_static && گرہ->src[0] != خالی_اشاریہ &&
            (گرہ->src[0]->op == GGML_OP_GATED_DELTA_NET || گرہ->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            توڑو;
        }
        اگر (گرہ->src[0]->op == GGML_OP_VIEW) {
            خودکار * src = گرہ->src[0];
            اگر (ggml_nelements(گرہ) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            اگر (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            خودکار * src = گرہ->src[0];
            اگر (ggml_nelements(گرہ) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                اگر (گرہ->ne[0] == src->ne[0] && گرہ->ne[1] == src->ne[2] && گرہ->ne[2] == 1 &&
                    گرہ->nb[0] == src->nb[0] && گرہ->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    توڑو;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                صحیح diff_گنتی = 0;
                برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                    اگر (گرہ->ne[i] != src->ne[i]) {
                        diff_گنتی++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    اگر (گرہ->ne[i] > src->ne[i]) {
                        واپس 0;
                    }
                }
                اگر (diff_گنتی >= 1) {
                    op_case = 0;
                }
            }
        }
        توڑو;
    }
    حالت GGML_OP_RMS_NORM: {
        اگر (گرہ->src[0]->op == GGML_OP_VIEW) {
            اگر (is_same_shape(گرہ->src[0]->src[0], گرہ->src[0])) {
                op_case = 1;
            } ورنہ اگر (!m_model_params.has_rs_rollback &&
                       گرہ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } ورنہ اگر (گرہ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        توڑو;
    }
    حالت GGML_OP_POOL_2D: {
        ثابت ggml_op_pool pool_منوال = static_cast<ggml_op_pool>(گرہ->op_params[0]);
        انتخاب (pool_منوال) {
        حالت GGML_OP_POOL_MAX: {
            op_case = 1;
            توڑو;
        }
        حالت GGML_OP_POOL_AVG: {
            op_case = 2;
            توڑو;
        }
        طےشدہ:
            op_case = 0;
            توڑو;
        }
        توڑو;
    }
    حالت GGML_OP_UPSCALE: {
        ثابت int32_t منوال_flags = گرہ->op_params[0];
        ثابت ggml_scale_منوال scale_منوال = static_cast<ggml_scale_منوال>(منوال_flags & 0xFF);
        انتخاب (scale_منوال) {
        حالت GGML_SCALE_منوال_قریب_ترین: {
            op_case = 1;
            توڑو;
        }
        حالت GGML_SCALE_منوال_BILINEAR: {
            op_case = 2;
            توڑو;
        }
        حالت GGML_SCALE_منوال_BICUBIC: {
            op_case = 3;
            توڑو;
        }
        طےشدہ:
            op_case = 0;
            توڑو;
        }
        توڑو;
    }
    حالت GGML_OP_CPY: {
        اگر (گرہ->src[0]->op == GGML_OP_VIEW) {
            اگر (گرہ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                اگر (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_بھرا_single_خانہ_writeback(گرہ) ? 7 : 10;
                } ورنہ {
                    op_case = 1;
                }
            } ورنہ اگر (GgmlOvDecoder::is_conv_state_writeback(گرہ)) {
                op_case = is_بھرا_single_خانہ_writeback(گرہ) ? 8 : 2;
                توڑو;
            } ورنہ اگر (is_conv_حالتیں_all_tensor(گرہ->view_src) && گرہ->src[1] != خالی_اشاریہ &&
                       گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src == گرہ->view_src) {
                op_case = 4;
                توڑو;
            } ورنہ اگر (گرہ->src[0]->src[0]->op == GGML_OP_GET_صفیں && گرہ->src[1] != خالی_اشاریہ &&
                       گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src != خالی_اشاریہ &&
                       is_recurrent_cache(گرہ->src[1]->view_src) && گرہ->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } ورنہ اگر (گرہ->src[0]->op == GGML_OP_GET_صفیں && گرہ->src[1] != خالی_اشاریہ &&
                   گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src != خالی_اشاریہ &&
                   is_recurrent_cache(گرہ->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = گرہ->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } ورنہ اگر (گرہ->src[1] != خالی_اشاریہ && گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src != خالی_اشاریہ) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ثابت ggml_tensor * kv_buf = گرہ->src[1]->view_src;
            اگر (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                برائے (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
                    ثابت ggml_tensor * n = m_cgraph->nodes[i];
                    اگر (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        جاری;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    اگر ((n->src[1] != خالی_اشاریہ && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != خالی_اشاریہ && n->src[2]->view_src == kv_buf)) {
                        اگر (n->src[3] != خالی_اشاریہ) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        توڑو;
                    }
                }
            }
        }
        توڑو;
    }
    حالت GGML_OP_جوڑو: {
        اگر (is_moe_expert_مجموع_جوڑو(گرہ)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        توڑو;
    }
    حالت GGML_OP_SCALE: {
        اگر (گرہ->view_src && گرہ->بفر->usage == GGML_BACKEND_بفر_USAGE_ANY) {
            op_case = گرہ->view_src->ne[1] == 1 ? 2 : 1;
        }
        توڑو;
    }
    حالت GGML_OP_L2_NORM: {
        اگر (std::string(گرہ->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        توڑو;
    }
    حالت GGML_OP_FLASH_ATTN_EXT: {
        اگر (گرہ->src[1] != خالی_اشاریہ && گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src != خالی_اشاریہ) {
            ثابت ggml_tensor * kv_buf = گرہ->src[1]->view_src;
            اگر (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (گرہ->src[3] != خالی_اشاریہ) ? 1 : 2;
            }
        }
        توڑو;
    }
    طےشدہ:
        توڑو;
    }
    واپس op_case;
}

std::optional<صحیح> نکالو_layer_from_name(ثابت std::string & name) {
    size_t pos1 = name.find("_l");
    اگر (pos1 == std::string::npos) {
        واپس std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    اگر (pos2 == std::string::npos) {
        pos2 = name.لمبائی();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    صحیح layer = std::stoi(layer_str);
    واپس layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
ساکن صحیح get_swa_کھڑکی_from_نقاب(ثابت ggml_tensor * نقاب) {
    اگر (نقاب->ڈیٹا == خالی_اشاریہ || !ggml_backend_بفر_is_host(نقاب->بفر)) {
        واپس -1;
    }
    اگر (نقاب->type != GGML_TYPE_F16 && نقاب->type != GGML_TYPE_F32) {
        واپس -1;
    }

    ثابت int64_t n_kv = نقاب->ne[0];
    ثابت int64_t n_علامات = نقاب->ne[1];
    int64_t کھڑکی = 0;

    برائے (int64_t r = 0; r < n_علامات; r++) {
        int64_t kept = 0;
        برائے (int64_t c = 0; c < n_kv; c++) {
            ثابت size_t i = (size_t) r * n_kv + c;
            ثابت اعشاری v = نقاب->type == GGML_TYPE_F16 ? ggml_fp16_مقصد_fp32(((ثابت ggml_fp16_t *) نقاب->ڈیٹا)[i]) :
                                                          ((ثابت اعشاری *) نقاب->ڈیٹا)[i];
            اگر (v > -INFINITY) {
                kept++;
            }
        }
        کھڑکی = std::max(کھڑکی, kept);
    }

    واپس کھڑکی > 0 ? (صحیح) کھڑکی : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, منطقی is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    خودکار get_attention_نمونہ_case = [](ثابت ggml_tensor * گرہ) -> صحیح {
        اگر (گرہ == خالی_اشاریہ) {
            واپس -1;
        }

        انتخاب (گرہ->op) {
        حالت GGML_OP_FLASH_ATTN_EXT:
            اگر (گرہ->src[0] == خالی_اشاریہ || گرہ->src[1] == خالی_اشاریہ) {
                واپس -1;
            }
            انتخاب (گرہ->src[1]->op) {
            حالت GGML_OP_ترتیب_بدلو:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                اگر (گرہ->src[3] != خالی_اشاریہ && گرہ->src[1]->src[0] != خالی_اشاریہ &&
                    گرہ->src[1]->src[0]->op == GGML_OP_VIEW) {
                    واپس 0;
                }
                توڑو;
            حالت GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                اگر (گرہ->src[3] != خالی_اشاریہ && گرہ->src[1]->src[0] != خالی_اشاریہ &&
                    گرہ->src[1]->src[0]->op == GGML_OP_ترتیب_بدلو && گرہ->src[1]->src[0]->src[0] != خالی_اشاریہ &&
                    گرہ->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    واپس 1;
                }
                توڑو;
            حالت GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                اگر (گرہ->src[1]->view_src != خالی_اشاریہ) {
                    اگر (گرہ->src[3] != خالی_اشاریہ) {
                        واپس 4;  // decoder self-attention
                    }
                    واپس 5;      // cross-attention or encoder self-attention
                }
                توڑو;
            طےشدہ:
                توڑو;
            }
            توڑو;
        حالت GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            اگر (گرہ->src[0] != خالی_اشاریہ && گرہ->src[1] != خالی_اشاریہ && گرہ->src[0]->op == GGML_OP_MUL_MAT &&
                گرہ->src[0]->src[0] != خالی_اشاریہ && گرہ->src[0]->src[1] != خالی_اشاریہ &&
                گرہ->src[0]->src[0]->op == GGML_OP_ترتیب_بدلو && گرہ->src[0]->src[0]->src[0] != خالی_اشاریہ &&
                گرہ->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                واپس 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            اگر (گرہ->src[0]->op == GGML_OP_جوڑو && گرہ->src[0]->src[0] != خالی_اشاریہ &&
                گرہ->src[0]->src[0]->op == GGML_OP_MUL_MAT && گرہ->src[0]->src[0]->src[0] != خالی_اشاریہ &&
                گرہ->src[0]->src[0]->src[0]->op == GGML_OP_ترتیب_بدلو) {
                واپس 3;
            }
            توڑو;
        طےشدہ:
            توڑو;
        }

        واپس -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    خودکار get_attention_op_نقاب = [&get_attention_نمونہ_case](ثابت ggml_tensor * گرہ) -> ثابت ggml_tensor * {
        انتخاب (get_attention_نمونہ_case(گرہ)) {
        حالت 0:
        حالت 1:
            واپس گرہ->src[3];
        حالت 2:
        حالت 3:
            واپس گرہ->src[1];
        طےشدہ:
            واپس خالی_اشاریہ;
        }
    };

    // Pre-pass: classify sliding-window vs full-attention layers.
    //
    // An interleaved-SWA model keeps two KV caches and two attention masks, and hands each layer
    // whichever pair matches its attention type. The mask tensor does not say which is which: both
    // are named "attn_inp_kq_mask" by build_attn_inp_kq_mask(), and both carry the same n_kv because
    // llama_kv_cache::get_n_kv() pads occupancy up to a common multiple.
    //
    // The KV cache does say. Each cache allocates cache_k_l<N> once at load time with its own cell
    // count: the windowed cache is sized from the window
    // (PAD(min(size_base, n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256), see
    // llama_kv_cache_iswa), the full-attention one spans the whole context. Read the LEAF buffer
    // behind the VIEW rather than the VIEW itself: the leaf extent is a constant per layer, known
    // from the first graph onwards, while the view grows with context depth and would invert the
    // comparison at shallow depth.
    //
    // Layers whose leaf is smaller than the largest leaf are the windowed ones. When every layer
    // reports the same extent there is no distinction to draw -- either the model has no windowed
    // layers, or the window is at least as large as the context so the two caches coincide, in
    // which case a windowed layer and a full-attention one compute the same thing.
    //
    // Getting this wrong is silent and severe: with the windowed layers classified as
    // full-attention, permute's KV slicing uses attention_size instead of attention_size_swa. The
    // two agree while the context is shorter than the window, then diverge, and the mask add fails
    // shape inference ("Failed to broadcast-merge input shapes") partway into a long prompt.
    {
        std::map<صحیح, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<صحیح, ثابت ggml_tensor *> layer_نقاب;            // layer -> mask it consumes
        int64_t max_extent = 0;

        برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
            ثابت ggml_tensor * نقاب = get_attention_op_نقاب(cgraph->nodes[i]);
            اگر (نقاب == خالی_اشاریہ) {
                جاری;
            }
            ثابت ggml_tensor * cache_k_ترتیب_بدلو = خالی_اشاریہ;
            انتخاب (get_attention_نمونہ_case(cgraph->nodes[i])) {
            حالت 0:  cache_k_ترتیب_بدلو = cgraph->nodes[i]->src[1];                     توڑو;
            حالت 1:  cache_k_ترتیب_بدلو = cgraph->nodes[i]->src[1]->src[0];             توڑو;
            حالت 2:  cache_k_ترتیب_بدلو = cgraph->nodes[i]->src[0]->src[0];             توڑو;
            طےشدہ: cache_k_ترتیب_بدلو = cgraph->nodes[i]->src[0]->src[0]->src[0];     توڑو;
            }
            ثابت ggml_tensor * cache_k_view = cache_k_ترتیب_بدلو->src[0];
            اگر (cache_k_view->op != GGML_OP_VIEW) {
                جاری;
            }
            ثابت ggml_tensor * برگ = cache_k_view->src[0];
            خودکار layer = نکالو_layer_from_name(برگ->name);
            اگر (!layer.has_value()) {
                جاری;
            }
            layer_extent[layer.value()] = برگ->ne[1];
            layer_نقاب[layer.value()] = نقاب;
            max_extent = std::max(max_extent, برگ->ne[1]);
        }

        برائے (ثابت خودکار & [layer, extent] : layer_extent) {
            اگر (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                اگر (model_params.swa_نقاب == خالی_اشاریہ) {
                    model_params.swa_نقاب = layer_نقاب[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        اگر (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string فی_layer;
            برائے (ثابت خودکار & [layer, extent] : layer_extent) {
                فی_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (طویل) max_extent, model_params.swa_layers.size(), فی_layer.c_str());
        }
    }

    منطقی rope_نظر_آیا = جھوٹ;
    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = cgraph->nodes[i];
        ثابت صحیح attention_نمونہ_case = get_attention_نمونہ_case(گرہ);
        اگر (attention_نمونہ_case != -1) {
            ggml_tensor * cache_k_ترتیب_بدلو = خالی_اشاریہ;
            ggml_tensor * نقاب = خالی_اشاریہ;

            انتخاب (attention_نمونہ_case) {
            حالت 0:
                cache_k_ترتیب_بدلو = گرہ->src[1];
                نقاب = گرہ->src[3];
                توڑو;
            حالت 1:
                cache_k_ترتیب_بدلو = گرہ->src[1]->src[0];
                نقاب = گرہ->src[3];
                توڑو;
            حالت 2:
                cache_k_ترتیب_بدلو = گرہ->src[0]->src[0];
                نقاب = گرہ->src[1];
                توڑو;
            حالت 3:
                cache_k_ترتیب_بدلو = گرہ->src[0]->src[0]->src[0];
                نقاب = گرہ->src[1];
                توڑو;
            حالت 4:
            حالت 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                خودکار * cache_k_view = گرہ->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_فی_seq = گرہ->src[0]->ne[1];
                اگر (attention_نمونہ_case == 4) {
                    compute_params.attention_سائز = cache_k_view->ne[1];
                } ورنہ {
                    compute_params.attention_سائز_static = cache_k_view->ne[1];
                }
                جاری;
            }
            طےشدہ:
                توڑو;
            }

            أكد(cache_k_ترتیب_بدلو != خالی_اشاریہ);

            model_params.سر_سائز = cache_k_ترتیب_بدلو->ne[0];
            model_params.n_heads_kv = cache_k_ترتیب_بدلو->ne[2];
            compute_params.داخلہ_len = گرہ->src[0]->ne[1];
            compute_params.token_len_فی_seq = گرہ->src[0]->ne[1];

            خودکار * cache_k_view = cache_k_ترتیب_بدلو->src[0];
            اگر (cache_k_view->op != GGML_OP_VIEW || نقاب == خالی_اشاریہ) {
                جاری;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            صحیح layer = نکالو_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ثابت منطقی layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_بفر_ctx_id = ggml_backend_openvino_بفر_get_ctx_id(cache_k->بفر);
            model_params.n_heads_kv_فی_layer[layer] = cache_k_ترتیب_بدلو->ne[2];
            اگر (layer_is_swa) {
                model_params.ctx_فی_seq_swa = cache_k->ne[1];
            } ورنہ {
                model_params.ctx_فی_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_فعال = نقاب->ne[3];
            خودکار seq_سائز = cache_k->ne[0] * cache_k->ne[1] * ggml_type_سائز(cache_k->type);
            size_t انحراف;
            حافظہ_نقل(&انحراف, cache_k_view->op_params, حجم(size_t));
            compute_params.seq_فعال_آغاز = انحراف / seq_سائز;

            اگر (layer_is_swa) {
                compute_params.attention_سائز_swa = نقاب->ne[0];
                compute_params.swa_کھڑکی = get_swa_کھڑکی_from_نقاب(نقاب);
            } ورنہ {
                compute_params.attention_سائز = نقاب->ne[0];
            }
            اگر (is_static) {
                compute_params.attention_سائز = model_params.ctx_فی_seq;
                compute_params.attention_سائز_swa = model_params.ctx_فی_seq_swa;
                compute_params.token_len_فی_seq = 1;
            }
        }

        اگر (گرہ->op == GGML_OP_MUL_MAT && گرہ->src[0]->op == GGML_OP_ترتیب_بدلو &&
            گرہ->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(گرہ->src[0]->view_src, گرہ->view_src)) {
            اگر (گرہ->src[1]->op == GGML_OP_ترتیب_بدلو && گرہ->src[1]->src[0]->op == GGML_OP_VIEW &&
                گرہ->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_سائز = گرہ->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        اگر (گرہ->op == GGML_OP_TRANSPOSE && گرہ->src[0]->op == GGML_OP_ترتیب_بدلو &&
            گرہ->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_سائز = گرہ->ne[0];
            اگر (is_static) {
                compute_params.attention_سائز = model_params.ctx_فی_seq;
            }
        }
        اگر (گرہ->op == GGML_OP_ROPE) {
            اگر (compute_params.token_len_فی_seq == -1 && گرہ->src[1] != خالی_اشاریہ) {
                compute_params.token_len_فی_seq = ggml_nelements(گرہ->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(حجم(model_params.rope_params) == حجم(int32_t) * 16, "rope_params size");
            اگر (!rope_نظر_آیا) {
                حافظہ_نقل(model_params.rope_params, گرہ->op_params, حجم(int32_t) * 16);
                rope_نظر_آیا = سچ;
            } ورنہ اگر (memcmp(model_params.rope_params, گرہ->op_params, حجم(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = سچ;
            }
        }
        اگر (گرہ->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_سائز = گرہ->src[0]->ne[0];
        }
        اگر (گرہ->op == GGML_OP_SCALE && گرہ->view_src != خالی_اشاریہ && is_recurrent_cache(گرہ->view_src)) {
            اگر (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = گرہ->view_src->ne[1];
            } ورنہ {
                GGML_ASSERT(model_params.n_rs_slots == گرہ->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(گرہ) / گرہ->view_src->ne[0];
            compute_params.cache_rs_reset_idx = گرہ->src[0]->view_offs / گرہ->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        اگر (گرہ->op == GGML_OP_CPY && گرہ->view_src != خالی_اشاریہ && is_recurrent_cache(گرہ->view_src) &&
            گرہ->src[1] != خالی_اشاریہ && گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src == گرہ->view_src) {
            ثابت منطقی is_conv = is_conv_state_writeback(گرہ);
            ثابت منطقی is_gdn = گرہ->src[0]->op == GGML_OP_VIEW && گرہ->src[0]->src[0] != خالی_اشاریہ &&
                                گرہ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ثابت منطقی is_extra = گرہ->src[0]->op == GGML_OP_GET_صفیں;
            ثابت منطقی is_gdn_rollback = is_gdn && is_same_shape(گرہ->src[0], گرہ->src[1]);

            ثابت ggml_tensor * dest_view = گرہ->src[1];
            ثابت ggml_tensor * cache = گرہ->view_src;
            ثابت size_t صف_بائٹیں = cache->ne[0] * ggml_type_سائز(cache->type);
            اگر (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = سچ;
            }
            اگر (صف_بائٹیں > 0 && (is_conv || is_gdn || is_extra) && !is_بھرا_single_خانہ_writeback(گرہ)) {
                ComputeParams::RsWriteback writeback;
                writeback.خانہ_begin = (صحیح) (dest_view->view_offs / صف_بائٹیں);
                اگر (is_conv) {
                    writeback.src_begin = (صحیح) (گرہ->src[0]->view_offs / گرہ->src[0]->view_src->nb[0]);
                } ورنہ اگر (is_gdn_rollback) {
                    writeback.src_begin = (صحیح) (گرہ->src[0]->view_offs / گرہ->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, گرہ)] = writeback;
            }
            اگر ((is_conv || is_gdn) && !is_بھرا_single_خانہ_writeback(گرہ)) {
                compute_params.s_copy_فعال_خانہ_len = (صحیح) dest_view->ne[1];
            }
        }
    }
    اگر (model_params.n_heads_kv == -1) {
        برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
            ثابت خودکار * گرہ = cgraph->nodes[i];
            ثابت ggml_tensor * نقاب = خالی_اشاریہ;
            اگر (گرہ->op == GGML_OP_SOFT_MAX) {
                نقاب = گرہ->src[1];
            } ورنہ اگر (گرہ->op == GGML_OP_FLASH_ATTN_EXT) {
                نقاب = گرہ->src[3];
            } ورنہ {
                جاری;
            }
            اگر (نقاب == خالی_اشاریہ || نقاب->op != GGML_OP_NONE || !(نقاب->flags & GGML_TENSOR_FLAG_داخلہ) ||
                گرہ->src[0] == خالی_اشاریہ) {
                جاری;
            }
            model_params.is_cacheless_attn = سچ;
            model_params.n_seq = 1;
            model_params.ctx_فی_seq = نقاب->ne[0];
            compute_params.داخلہ_len = گرہ->src[0]->ne[1];
            compute_params.token_len_فی_seq = compute_params.داخلہ_len;
            توڑو;
        }
    }

    خودکار * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    اگر (model_params.is_cacheless_attn) {
        برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
            ثابت خودکار * گرہ = cgraph->nodes[i];
            اگر (گرہ->op == GGML_OP_GET_صفیں && is_output_idx(گرہ->src[1], گرہ)) {
                compute_params.output_len = گرہ->src[1]->ne[0];
                توڑو;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    اگر (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_فی_seq * model_params.n_seq;
    واپس {model_params, compute_params};
}

خالی GgmlOvDecoder::validate_cgraph() ثابت {
    اگر (m_model_params.n_seq > 1 && m_is_static == سچ) {
        پھینکو std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_گراف_داخلہ_shape(ثابت ggml_tensor * op,
                                                      ثابت ggml_tensor * داخلہ,
                                                      صحیح dynamic_dim_اشاریہ) ثابت {
    اگر (m_naive) {
        واپس داخلہ != خالی_اشاریہ ? ov::PartialShape{get_shape(داخلہ)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape داخلہ_shape;

    اگر (is_inp_tok(داخلہ, op) || is_inp_pos(داخلہ, op)) {
        // tokens or positions
        صحیح len = m_is_static ? (m_is_prefill ? m_prefill_chunk_سائز : 1) : -1;
        اگر (m_is_static && is_inp_pos(داخلہ, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        داخلہ_shape = ov::PartialShape{1, 1, 1, len};

    } ورنہ اگر (is_output_idx(داخلہ, op)) {
        // output index
        داخلہ_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } ورنہ اگر (is_inp_اوسط(داخلہ, op)) {
        داخلہ_shape = m_is_static ? ov::PartialShape{1, 1, داخلہ->ne[1], m_prefill_chunk_سائز} :
                                    ov::PartialShape{1, 1, -1, -1};

    } ورنہ اگر (is_inp_scale_صفیں(داخلہ, op)) {
        داخلہ_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_سائز : 1) : -1, 1};

    } ورنہ اگر (is_inp_نقاب(داخلہ, op)) {
        //نقاب
        اگر (m_is_static) {
            داخلہ_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_سائز : 1, m_model_params.ctx};
        } ورنہ اگر (m_is_stateful) {
            داخلہ_shape = ov::PartialShape{1, 1, -1, -1};
        } ورنہ {
            داخلہ_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } ورنہ اگر (is_recurrent_cache(داخلہ)) {
        داخلہ_shape = ov::PartialShape{get_shape(داخلہ)};
        اگر (!m_is_static && !m_is_stateful && داخلہ->ne[1] > 1) {
            داخلہ_shape[2] = -1;
        }

    } ورنہ اگر (is_kvcache(داخلہ, op)) {
        // kvcache
        داخلہ_shape = ov::PartialShape{get_shape(داخلہ)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ثابت منطقی is_flat_kv = (داخلہ->ne[1] == 1 && داخلہ->ne[2] == 1 && داخلہ->ne[3] == 1);
        اگر (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            داخلہ_shape[2] = -1;
        }
        اگر (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            أكد(داخلہ_shape.size() == 4 && داخلہ_shape[0] == 1 && داخلہ_shape[1] == 1 &&
                   داخلہ_shape[2].is_dynamic() && داخلہ_shape[3].is_static());
            ثابت صحیح n_heads_kv = get_n_heads_kv_for_tensor(داخلہ);
            أكد(n_heads_kv > 0 && داخلہ_shape[3].get_لمبائی() % n_heads_kv == 0);
            ثابت int64_t combined_dim = داخلہ_shape[3].get_لمبائی();  // n_heads_kv * head_size
            ثابت int64_t سر_سائز = combined_dim / n_heads_kv;
            داخلہ_shape = {داخلہ_shape[0], ov::Dimension::dynamic(), n_heads_kv, سر_سائز};
        }

    } ورنہ اگر (is_kv_idx(داخلہ, op)) {
        // kv update index
        صحیح len = m_is_static ? (m_is_prefill ? m_prefill_chunk_سائز : 1) : -1;
        داخلہ_shape = ov::PartialShape{1, 1, 1, len};

    } ورنہ اگر (is_inp_s_copy(داخلہ, op) || is_s_copy_برگ(داخلہ)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        داخلہ_shape = m_is_static ? ov::PartialShape{get_shape(داخلہ)} : ov::PartialShape{1, 1, 1, -1};

    } ورنہ {
        داخلہ_shape = ov::PartialShape{get_shape(داخلہ)};
    }
    اگر (dynamic_dim_اشاریہ != -1 && m_model_is_splitted) {
        داخلہ_shape[3 - dynamic_dim_اشاریہ] = -1;
    }
    اگر (op->op == GGML_OP_SOFT_MAX && op->src[1] != خالی_اشاریہ && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_داخلہ && op->src[1] == داخلہ) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        اگر (m_is_static) {
            ثابت int64_t seq_فعال = m_is_prefill ? m_prefill_chunk_سائز : 1;
            داخلہ_shape[2] = seq_فعال;
            داخلہ_shape[3] = seq_فعال;
        } ورنہ {
            داخلہ_shape[2] = -1;
            داخلہ_shape[3] = -1;
        }
    }
    واپس داخلہ_shape;
}

منطقی GgmlOvDecoder::is_s_copy_برگ(ثابت ggml_tensor * tensor) ثابت {
    اگر (tensor == خالی_اشاریہ || tensor->op != GGML_OP_NONE || m_cgraph == خالی_اشاریہ) {
        واپس جھوٹ;
    }
    برائے (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت ggml_tensor * گرہ = m_cgraph->nodes[i];
        اگر (گرہ->op != GGML_OP_GET_صفیں || گرہ->src[0] == خالی_اشاریہ || گرہ->src[1] == خالی_اشاریہ) {
            جاری;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ثابت ggml_tensor * idx = گرہ->src[1];
        جب (idx != خالی_اشاریہ && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        اگر (idx != tensor) {
            جاری;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ثابت ggml_tensor * ڈیٹا = گرہ->src[0];
        جب (ڈیٹا != خالی_اشاریہ && (ڈیٹا->op == GGML_OP_VIEW || ڈیٹا->op == GGML_OP_RESHAPE)) {
            ڈیٹا = ڈیٹا->src[0];
        }
        اگر (ڈیٹا != خالی_اشاریہ && is_recurrent_cache(ڈیٹا)) {
            واپس سچ;
        }
    }
    واپس جھوٹ;
}

خالی GgmlOvDecoder::جوڑو_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    خودکار create_1d_داخلہ = [یہ](ثابت std::string & name, int64_t value, منطقی force_parameter = جھوٹ) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    اگر (m_compute_params.attention_سائز != -1) {
        create_1d_داخلہ("attention_size", m_compute_params.attention_سائز);
    }
    اگر (m_compute_params.attention_سائز_static != -1) {
        create_1d_داخلہ("attention_size_static", m_compute_params.attention_سائز_static);
    }
    اگر (m_compute_params.attention_سائز_swa != -1) {
        create_1d_داخلہ("attention_size_swa", m_compute_params.attention_سائز_swa);
    }
    // only the stateful SWA mask consumes this
    اگر (is_stateful() && m_compute_params.swa_کھڑکی != -1) {
        create_1d_داخلہ("swa_window", m_compute_params.swa_کھڑکی);
    }
    create_1d_داخلہ("n_seq_active", m_compute_params.n_seq_فعال);
    create_1d_داخلہ("seq_active_start", m_compute_params.seq_فعال_آغاز);
    create_1d_داخلہ("seq_active_end", m_compute_params.seq_فعال_آغاز + m_compute_params.n_seq_فعال);
    اگر (m_compute_params.token_len_فی_seq != -1) {
        create_1d_داخلہ("token_len_per_seq", m_compute_params.token_len_فی_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    اگر (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_داخلہ("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/سچ);
        create_1d_داخلہ("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/سچ);
    }

    اگر (m_compute_params.s_copy_فعال_خانہ_len != -1) {
        create_1d_داخلہ("s_copy_active_slot_len", m_compute_params.s_copy_فعال_خانہ_len);
        اگر (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_داخلہ("chunk_valid_len", get_static_n_علامات(), /*force_parameter=*/سچ);
        }
    }

    برائے (ثابت خودکار & [گرہ_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_داخلہ("rs_slot_begin_" + گرہ_name, writeback.خانہ_begin);
        اگر (!m_is_static && writeback.src_begin >= 0) {
            create_1d_داخلہ("rs_src_begin_" + گرہ_name, writeback.src_begin);
        }
    }
}

منطقی GgmlOvDecoder::گرہ_is_مستعمل_as_src(ثابت صحیح گرہ_idx) {
    ggml_tensor * گرہ = m_cgraph->nodes[گرہ_idx];
    برائے (صحیح i = گرہ_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_گرہ = m_cgraph->nodes[i];
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            اگر (other_گرہ->src[j] == گرہ) {
                واپس سچ;
            }
        }
    }
    واپس جھوٹ;
}

خالی GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    برائے (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        اگر (گرہ->op == GGML_OP_NONE && گرہ_is_مستعمل_as_src(i)) {
            std::string گرہ_name = get_tensor_ov_name(m_cgraph, گرہ);
            اگر (m_model_weights.find(گرہ_name) == m_model_weights.end()) {
                m_inputs[گرہ_name] = گرہ;
                m_model_inputs[گرہ_name] = {get_ov_type(گرہ),
                                             get_گراف_داخلہ_shape(گرہ, خالی_اشاریہ, m_گرہ_dynamic_dims[گرہ])};
            }
            جاری;
        }
        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            خودکار * src = گرہ->src[i];
            اگر (src == خالی_اشاریہ) {
                جاری;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            اگر (src->flags & GGML_TENSOR_FLAG_داخلہ) {
                src_name = get_tensor_گراف_داخلہ_ov_name(یہ, m_cgraph, src, گرہ);
            }
            اگر (m_model_weights.find(src_name) != m_model_weights.end()) {
                جاری;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            اگر (src->view_src != خالی_اشاریہ &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                جاری;
            }

            منطقی is_intermediate_گرہ = جھوٹ;
            برائے (ثابت خودکار & گرہ_info : m_گرہ_info_list) {
                اگر (گرہ_info.گرہ == src) {
                    is_intermediate_گرہ = سچ;
                    توڑو;
                }
            }
            اگر (is_intermediate_گرہ) {
                جاری;
            }
            اگر (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                جاری;
            }

            m_inputs[src_name] = src;

            ggml_backend_بفر * بفر = src->بفر;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            اگر (بفر->usage == GGML_BACKEND_بفر_USAGE_ANY) {
                اگر (خودکار it = std::find(m_model_params.kv_نام.begin(), m_model_params.kv_نام.end(), src_name);
                    it == m_model_params.kv_نام.end()) {
                    m_model_params.kv_نام.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            جب (src->op == GGML_OP_VIEW && src->src[0] != خالی_اشاریہ) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_گراف_داخلہ_shape(گرہ, src, m_گرہ_dynamic_dims[src])};
        }
    }
}

خالی GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_نام.clear();
    برائے (صحیح گرہ_n = 0; گرہ_n < m_cgraph->n_nodes; گرہ_n++) {
        خودکار * cur_گرہ = m_cgraph->nodes[گرہ_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        اگر (cur_گرہ->op == GGML_OP_NONE || cur_گرہ->op == GGML_OP_VIEW || cur_گرہ->op == GGML_OP_RESHAPE) {
            جاری;
        }
        اگر (::is_inplace_op(cur_گرہ) && ggml_nbytes(cur_گرہ) == 0) {
            جاری;
        }
        خودکار cur_گرہ_use_گنتی = m_cgraph->use_counts[ggml_hash_ڈھونڈو(&m_cgraph->دیکھا_hash_سیٹ, cur_گرہ)];
        اگر (cur_گرہ_use_گنتی == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            اگر (cur_گرہ != خالی_اشاریہ && ::is_inplace_op(cur_گرہ) && ggml_nbytes(cur_گرہ) > 0) {
                cur_گرہ = cur_گرہ->view_src;
            }
        } ورنہ {
            صحیح داخلہ_use_گنتی = 0;
            برائے (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * گرہ = m_cgraph->nodes[i];
                برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                    اگر (گرہ->src[j] != تہی && گرہ->src[j] == cur_گرہ) {
                        داخلہ_use_گنتی++;
                    }
                }
            }
            اگر (داخلہ_use_گنتی == cur_گرہ_use_گنتی) {
                cur_گرہ = خالی_اشاریہ;
            }
        }
        اگر (cur_گرہ != خالی_اشاریہ) {
            std::string cur_گرہ_name = get_tensor_ov_name(m_cgraph, cur_گرہ);
            m_model_outputs[cur_گرہ_name] = cur_گرہ;
            m_model_output_نام.insert(cur_گرہ_name);
        }
    }
}

ثابت ggml_tensor * GgmlOvDecoder::get_tensor_مستعمل_op(ثابت ggml_tensor * tensor) ثابت {
    اگر (tensor == خالی_اشاریہ) {
        واپس خالی_اشاریہ;
    }
    برائے (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت خودکار * گرہ = m_cgraph->nodes[i];
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            اگر (گرہ->src[j] == tensor) {
                واپس گرہ;
            }
        }
    }
    واپس خالی_اشاریہ;
}

ثابت ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ثابت std::string & name) ثابت {
    برائے (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت خودکار * گرہ = m_cgraph->nodes[i];
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ثابت خودکار * src = گرہ->src[j];
            اگر (src == خالی_اشاریہ) {
                توڑو;
            }
            اگر (get_tensor_ov_name(m_cgraph, src) == name) {
                واپس src;
            }
        }
    }
    واپس خالی_اشاریہ;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_نام() ثابت {
    std::map<std::string, std::string> kv_param_res_نام;
    برائے (ثابت خودکار & name : m_model_params.kv_نام) {
        kv_param_res_نام[name] = name;
    }
    واپس kv_param_res_نام;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
ساکن منطقی is_mul_mat_id_expert_وزن(ثابت ggml_tensor * گرہ, صحیح src_اشاریہ) {
    واپس گرہ->op == GGML_OP_MUL_MAT_ID && src_اشاریہ == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_وزن_nodes(ggml_cgraph * cgraph, منطقی naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    خودکار * nodes = cgraph->nodes;
    خودکار n_nodes = cgraph->n_nodes;
    برائے (صحیح گرہ_i = 0; گرہ_i < n_nodes; گرہ_i++) {
        خودکار * گرہ = nodes[گرہ_i];
        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            خودکار * src = گرہ->src[i];
            اگر (src == خالی_اشاریہ) {
                جاری;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * بنیاد = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, بنیاد);
            اگر (is_rope_freqs_وزن(بنیاد, گرہ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_بفر * بفر = بنیاد->بفر;
            اگر (بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS || ggml_is_quantized(بنیاد->type) ||
                is_mul_mat_id_expert_وزن(گرہ, i)) {
                اگر (model_weights.find(src_name) == model_weights.end()) {
                    خودکار وزن_گرہ = create_وزن_گرہ(بنیاد, naive);
                    وزن_گرہ->سیٹ_friendly_name(src_name);
                    model_weights[src_name] = وزن_گرہ;
                }
            }
        }
    }
    واپس model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
ساکن std::mutex g_nonov_وزن_cache_mutex;
ساکن std::unordered_map<ثابت خالی *, std::shared_ptr<ov::Node>> g_nonov_وزن_cache;

std::set<std::string> GgmlOvDecoder::collect_وزن_نام(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> نام;
    برائے (صحیح گرہ_i = 0; گرہ_i < cgraph->n_nodes; گرہ_i++) {
        خودکار * گرہ = cgraph->nodes[گرہ_i];
        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            خودکار * src = گرہ->src[i];
            اگر (src == خالی_اشاریہ) {
                جاری;
            }
            ثابت ggml_tensor * بنیاد = src->view_src ? src->view_src : src;
            std::string src_name(بنیاد->name);
            اگر (is_rope_freqs_وزن(بنیاد, گرہ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_بفر * بفر = بنیاد->بفر;
            اگر (بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS || ggml_is_quantized(بنیاد->type)) {
                نام.insert(src_name);
            }
        }
    }
    واپس نام;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_وزن_گرہ(ggml_tensor * tensor, منطقی naive) {
    ثابت منطقی is_ov_بفر = ggml_backend_بفر_is_openvino(tensor->بفر);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    اگر (tensor->extra) {
        OPENVINO_ASSERT(is_ov_بفر, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        خودکار * extra_بنیاد = static_cast<ggml_openvino_extra_بنیاد *>(tensor->extra);

        اگر (extra_بنیاد->type == ggml_openvino_extra_بنیاد::Type::وزن_2) {
            // F16/F32/BF16 weight with shared-memory constant
            خودکار * وزن_extra = static_cast<ggml_openvino_وزن_extra *>(tensor->extra);
            اگر (وزن_extra->وزن_گرہ) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                واپس وزن_extra->وزن_گرہ;
            }
        } ورنہ اگر (extra_بنیاد->type == ggml_openvino_extra_بنیاد::Type::QUANTIZED_وزن) {
            // Quantized weight with pre-extracted data
            خودکار * quant_extra = static_cast<ggml_openvino_quantized_وزن_extra *>(tensor->extra);
            اگر (quant_extra->وزن_گرہ) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                واپس quant_extra->وزن_گرہ;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    اگر (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        خودکار وزن_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->ڈیٹا);
        خودکار وزن_گرہ = std::make_shared<ov::op::v0::Constant>(وزن_tensor);
        وزن_گرہ->سیٹ_friendly_name(tensor->name);
        واپس وزن_گرہ;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ثابت منطقی cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_بفر &&
                                 !naive && tensor->ڈیٹا != خالی_اشاریہ;
    اگر (cacheable_nonov) {
        std::lock_guard<std::mutex> تالا(g_nonov_وزن_cache_mutex);
        خودکار it = g_nonov_وزن_cache.find(tensor->ڈیٹا);
        اگر (it != g_nonov_وزن_cache.end()) {
            واپس it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    ساکن ثابت std::set<ggml_type> وزن_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    اگر (وزن_types.find(tensor->type) == وزن_types.end()) {
        پھینکو std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    Ov_وزن ov_وزن;
    اگر (ggml_is_quantized(tensor->type)) {
        خودکار use_bias = naive;
        اگر (is_ov_بفر) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_سائز = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_سائز);
            حافظہ_نقل(tmp.ڈیٹا(), tensor->ڈیٹا, raw_سائز);
            ov_وزن = process_وزن_tensor(tensor, tmp.ڈیٹا(), tensor->ڈیٹا, use_bias);
        } ورنہ {
            ov_وزن = process_وزن_tensor(tensor, tensor->ڈیٹا, خالی_اشاریہ, use_bias);
        }
    } ورنہ {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_وزن = process_وزن_tensor(tensor, tensor->ڈیٹا, tensor->ڈیٹا);
    }

    ov_وزن.وزن_گرہ->سیٹ_friendly_name(tensor->name);
    اگر (!is_ov_بفر) {
        اگر (cacheable_nonov) {
            std::lock_guard<std::mutex> تالا(g_nonov_وزن_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            خودکار [it, inserted] = g_nonov_وزن_cache.emplace(tensor->ڈیٹا, ov_وزن.وزن_گرہ);
            واپس it->second;
        }
        واپس ov_وزن.وزن_گرہ;
    }

    ggml_openvino_extra_بنیاد * extra;
    اگر (ov_وزن.is_quantized()) {
        extra = نیا ggml_openvino_quantized_وزن_extra(std::move(ov_وزن.weights), std::move(ov_وزن.scales),
                                                         std::move(ov_وزن.zp), ov_وزن.وزن_گرہ);
    } ورنہ {
        extra = نیا ggml_openvino_وزن_extra(std::move(ov_وزن.weights), ov_وزن.وزن_گرہ);
    }
    ggml_openvino_بفر_رجسٹر_extra(tensor, extra);

    واپس ov_وزن.وزن_گرہ;
}

خالی GgmlOvDecoder::dump_cgraph(ثابت ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    اگر (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        واپس;
    }

    file << "=== GRAPH ===\n";

    // clang-format off
    file << "n_nodes = " << cgraph->n_nodes << "\n";
    file << " " << std::setw(3) << "nodes"
                <<  std::setw(15) << "shape"
                << std::setw(20) << "op"
                << std::setw(20) << "name"
                << std::setw(3) << "    "
                << std::setw(62) << "stride"
                << std::setw(20) << "buffer_type"
                << "\n";
    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = cgraph->nodes[i];

        // Get buffer type name
        ثابت حرف * buf_name = "none";
        ggml_backend_بفر_t buf = گرہ->view_src ? گرہ->view_src->بفر : گرہ->بفر;
        اگر (buf) {
            buf_name = ggml_backend_بفر_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << گرہ->ne[0] << ", "
             << std::setw(5) << گرہ->ne[1] << ", "
             << std::setw(5) << گرہ->ne[2] << ", "
             << std::setw(5) << گرہ->ne[3] << "] "
             << std::بائیں << std::setw(20) << ggml_op_name(گرہ->op) << std::دائیں << " "
             << std::بائیں << std::setw(45) << گرہ->name << std::دائیں
             << std::setw(2) << "[ "
             << std::setw(0) << گرہ->nb[0] << ", "
             << std::setw(5) << گرہ->nb[1] << ", "
             << std::setw(5) << گرہ->nb[2] << ", "
             << std::setw(5) << گرہ->nb[3] << "] "
             << std::دائیں << std::setw(15) << buf_name << std::دائیں
             << "\n";

        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            اگر (خودکار* src = گرہ->src[i]) {
                // Get buffer type name for source
                ثابت حرف * src_buf_name = "none";
                ggml_backend_بفر_t src_buf = src->view_src ? src->view_src->بفر : src->بفر;
                اگر (src_buf) {
                    src_buf_name = ggml_backend_بفر_name(src_buf);
                }

                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::بائیں << std::setw(12) << ggml_op_name(src->op) << std::دائیں;
                file << std::بائیں << std::setw(30) << src->name << std::دائیں
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << std::دائیں << std::setw(15) << src_buf_name << std::دائیں
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    برائے (صحیح i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * گرہ = cgraph->leafs[i];

        // Get buffer type name for leaf
        ثابت حرف * برگ_buf_name = "none";
        ggml_backend_بفر_t برگ_buf = گرہ->view_src ? گرہ->view_src->بفر : گرہ->بفر;
        اگر (برگ_buf) {
            برگ_buf_name = ggml_backend_بفر_name(برگ_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << گرہ->ne[0] << ", "
             << std::setw(5) << گرہ->ne[1] << "] "
             << std::setw(8) << ggml_op_name(گرہ->op) << " "
             << std::setw(16) << ggml_get_name(گرہ)
             << std::setw(20) << برگ_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

خالی print_tensor_پتہ_map(ثابت ggml_cgraph * cgraph) {
    std::map<خالی *, std::vector<std::string>> پتہ_map;
    برائے (صحیح گرہ_n = 0; گرہ_n < cgraph->n_nodes; گرہ_n++) {
        خودکار * گرہ = cgraph->nodes[گرہ_n];
        اگر (گرہ->ڈیٹا) {
            خودکار it = پتہ_map.find(گرہ->ڈیٹا);
            اگر (it == پتہ_map.end()) {
                پتہ_map[گرہ->ڈیٹا] = std::vector<std::string>();
            }
            پتہ_map[گرہ->ڈیٹا].push_back(گرہ->name);
        }
    }
    برائے (ثابت خودکار & pair : پتہ_map) {
        std::خارج << "Address: " << pair.پہلا << '\n';
        برائے (ثابت خودکار & name : pair.second) {
            std::خارج << name << " ; ";
        }
        std::خارج << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ثابت ggml_tensor * tensor) {
    std::vector<size_t> shape;
    برائے (صحیح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    واپس shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(ثابت ggml_tensor * tensor) {
    std::vector<size_t> stride;
    برائے (صحیح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    واپس stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(ثابت ggml_tensor * tensor) {
    انتخاب (tensor->type) {
    حالت GGML_TYPE_F64:
        واپس ov::element::f64;
    حالت GGML_TYPE_F32:
        واپس ov::element::f32;
    حالت GGML_TYPE_F16:
        واپس ov::element::f16;
    حالت GGML_TYPE_BF16:
        واپس ov::element::bf16;
    حالت GGML_TYPE_I8:
        واپس ov::element::i8;
    حالت GGML_TYPE_I16:
        واپس ov::element::i16;
    حالت GGML_TYPE_I32:
        واپس ov::element::i32;
    حالت GGML_TYPE_I64:
        واپس ov::element::i64;
    طےشدہ:
        واپس ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_داخلہ_shape(صحیح گرہ_idx, ثابت std::string & name) ثابت {
    واپس ov::PartialShape(get_shape(m_گرہ_info_list[گرہ_idx].گرہ_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_داخلہ_stride(صحیح گرہ_idx, ثابت std::string & name) ثابت {
    واپس get_stride(m_گرہ_info_list[گرہ_idx].گرہ_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_داخلہ_سائز(صحیح گرہ_idx, ثابت std::string & name) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        واپس it->second.size();
    }
    واپس 0;
}

size_t GgmlOvDecoder::get_view_داخلہ_انحراف(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        اگر (view_اشاریہ < it->second.size()) {
            واپس it->second[view_اشاریہ].second->view_offs;
        }
    }
    واپس 0;
}

size_t GgmlOvDecoder::get_view_داخلہ_src_انحراف(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        اگر (view_اشاریہ < it->second.size()) {
            خودکار * view_tensor = it->second[view_اشاریہ].second;
            اگر (view_tensor && view_tensor->src[0]) {
                واپس view_tensor->src[0]->view_offs;
            }
        }
    }
    واپس 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_داخلہ_stride(صحیح گرہ_idx,
                                                         ثابت std::string & name,
                                                         size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        اگر (view_اشاریہ < it->second.size()) {
            واپس get_stride(it->second[view_اشاریہ].second);
        }
    }
    واپس {};
}

std::vector<size_t> GgmlOvDecoder::get_view_داخلہ_src_stride(صحیح گرہ_idx,
                                                             ثابت std::string & name,
                                                             size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        اگر (view_اشاریہ < it->second.size()) {
            خودکار * view_tensor = it->second[view_اشاریہ].second;
            اگر (view_tensor && view_tensor->src[0]) {
                واپس get_stride(view_tensor->src[0]);
            }
        }
    }
    واپس {};
}

ov::Shape GgmlOvDecoder::get_view_داخلہ_ggml_shape(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        اگر (view_اشاریہ < it->second.size()) {
            واپس get_shape(it->second[view_اشاریہ].second);
        }
    }
    واپس {};
}

ov::Shape GgmlOvDecoder::get_view_داخلہ_src_ggml_shape(صحیح گرہ_idx,
                                                       ثابت std::string & name,
                                                       size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        اگر (view_اشاریہ < it->second.size()) {
            خودکار * view_tensor = it->second[view_اشاریہ].second;
            اگر (view_tensor && view_tensor->src[0]) {
                واپس get_shape(view_tensor->src[0]);
            }
        }
    }
    واپس {};
}

ov::PartialShape GgmlOvDecoder::get_view_داخلہ_ov_shape(صحیح گرہ_idx,
                                                        ثابت std::string & name,
                                                        size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        اگر (view_اشاریہ < it->second.size()) {
            خودکار * tensor = it->second[view_اشاریہ].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            خودکار dynamic_it = m_گرہ_dynamic_dims.find(tensor);
            اگر (dynamic_it != m_گرہ_dynamic_dims.end() && dynamic_it->second != -1) {
                صحیح dynamic_dim_اشاریہ = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_اشاریہ] = m_is_static ? get_static_n_علامات() : -1;
            }

            واپس shape;
        }
    }
    واپس {};
}

ov::PartialShape GgmlOvDecoder::get_view_داخلہ_src_ov_shape(صحیح گرہ_idx,
                                                            ثابت std::string & name,
                                                            size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        اگر (view_اشاریہ < it->second.size()) {
            خودکار * view_tensor = it->second[view_اشاریہ].second;
            اگر (view_tensor && view_tensor->src[0]) {
                خودکار * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                خودکار dynamic_it = m_گرہ_dynamic_dims.find(src_tensor);
                اگر (dynamic_it != m_گرہ_dynamic_dims.end() && dynamic_it->second != -1) {
                    صحیح dynamic_dim_اشاریہ = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_اشاریہ] = m_is_static ? get_static_n_علامات() : -1;
                }

                واپس shape;
            }
        }
    }
    واپس {};
}

std::string GgmlOvDecoder::get_view_داخلہ_name(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        اگر (view_اشاریہ < it->second.size()) {
            واپس it->second[view_اشاریہ].پہلا;
        }
    }
    واپس "";
}

std::string GgmlOvDecoder::get_view_داخلہ_src_name(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    اگر (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        اگر (view_اشاریہ < it->second.size()) {
            خودکار * view_tensor = it->second[view_اشاریہ].second;
            اگر (view_tensor && view_tensor->src[0]) {
                واپس get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    واپس "";
}

ov::element::Type GgmlOvDecoder::get_داخلہ_type(صحیح گرہ_idx, ثابت std::string & name) ثابت {
    واپس get_ov_type(m_گرہ_info_list[گرہ_idx].گرہ_inputs.at(name));
}

size_t GgmlOvDecoder::get_داخلہ_سائز() ثابت {
    واپس m_model_inputs.size();
}

size_t GgmlOvDecoder::get_داخلہ_سائز(صحیح گرہ_idx) ثابت {
    واپس m_گرہ_info_list[گرہ_idx].گرہ_inputs_نام.size();
}

std::vector<std::string> GgmlOvDecoder::get_داخلہ_نام(صحیح گرہ_idx) ثابت {
    واپس m_گرہ_info_list[گرہ_idx].گرہ_inputs_نام;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(صحیح گرہ_idx) ثابت {
    خودکار * ggml_tensor = m_گرہ_info_list[گرہ_idx].گرہ;
    واپس ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ثابت صحیح گرہ_idx) ثابت {
    واپس get_ov_type(m_گرہ_info_list[گرہ_idx].گرہ);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(صحیح گرہ_idx) ثابت {
    خودکار * ggml_tensor = m_گرہ_info_list[گرہ_idx].گرہ;
    واپس get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_نام(صحیح گرہ_idx) ثابت {
    خودکار * گرہ = m_گرہ_info_list[گرہ_idx].گرہ;
    اگر (گرہ->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        برائے (صحیح i = گرہ_idx + 1; i < m_cgraph->n_nodes; i++) {
            خودکار * صارف = m_cgraph->nodes[i];
            اگر (صارف->op != GGML_OP_VIEW || صارف->src[0] != گرہ) {
                جاری;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            خودکار & name = صارف->view_offs == 0 ? attn_name : state_name;
            اگر (!name.خالی_خانے()) {
                واپس {m_گرہ_info_list[گرہ_idx].گرہ_name};
            }
            name = get_tensor_ov_name(m_cgraph, صارف);
        }
        اگر (!attn_name.خالی_خانے() && !state_name.خالی_خانے()) {
            واپس {attn_name, state_name};
        }
    }
    واپس {m_گرہ_info_list[گرہ_idx].گرہ_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(صحیح گرہ_idx) ثابت {
    خودکار * گرہ = m_گرہ_info_list[گرہ_idx].گرہ;
    اگر (!::is_inplace_op(گرہ) || گرہ->view_src == خالی_اشاریہ || ggml_nbytes(گرہ) == 0) {
        واپس "";
    }
    ثابت صحیح op_case = m_گرہ_info_list[گرہ_idx].گرہ_op_case;
    اگر (گرہ->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_فعال_خانہ_len == -1) {
        واپس "";
    }
    واپس get_tensor_ov_name(m_cgraph, گرہ->view_src);
}

منطقی GgmlOvDecoder::is_view_like_alias_کا(صحیح گرہ_idx, ثابت std::string & view_src_name) ثابت {
    خودکار * گرہ = m_گرہ_info_list[گرہ_idx].گرہ;
    اگر (گرہ->view_src == خالی_اشاریہ || get_tensor_ov_name(m_cgraph, گرہ->view_src) != view_src_name) {
        واپس جھوٹ;
    }
    واپس گرہ->op == GGML_OP_RESHAPE || گرہ->op == GGML_OP_VIEW;
}

ثابت std::string & GgmlOvDecoder::get_op_name() ثابت {
    ساکن ثابت std::string unknown_name = "UNKNOWN_OP_NAME";
    واپس unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(صحیح گرہ_idx) ثابت {
    خودکار it = m_گرہ_dynamic_dims.find(m_گرہ_info_list[گرہ_idx].گرہ);
    اگر (it == m_گرہ_dynamic_dims.end()) {
        واپس -1;
    }
    واپس it->second;
}

ثابت std::string & GgmlOvDecoder::get_op_name(صحیح گرہ_idx) ثابت {
    واپس m_گرہ_info_list[گرہ_idx].گرہ_name;
}

int32_t * GgmlOvDecoder::get_داخلہ_op_params(صحیح گرہ_idx, ثابت std::string & name) ثابت {
    واپس m_گرہ_info_list[گرہ_idx].گرہ_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(صحیح گرہ_idx) ثابت {
    واپس m_گرہ_info_list[گرہ_idx].گرہ->op_params;
}

size_t GgmlOvDecoder::get_output_op_انحراف(صحیح گرہ_idx) ثابت {
    واپس m_گرہ_info_list[گرہ_idx].گرہ->view_offs;
}

خالی GgmlOvDecoder::دورہ_subgraph(std::function<خالی(std::shared_ptr<GgmlDecoder>, صحیح گرہ_idx)> گرہ_visitor) ثابت {
    برائے (صحیح گرہ_idx = 0; گرہ_idx < m_cgraph->n_nodes; گرہ_idx++) {
        اگر (m_cgraph->nodes[گرہ_idx]->op == GGML_OP_NONE) {
            جاری;
        }
        گرہ_visitor(std::make_shared<GgmlOvDecoder>(*یہ), گرہ_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ثابت ggml_tensor * گرہ) {
    انتخاب (گرہ->op) {
    حالت GGML_OP_UNARY:
        واپس std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(گرہ));
    حالت GGML_OP_GLU:
        واپس std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(گرہ));
    طےشدہ:
        واپس std::string("GGML_OP_") + ggml_op_name(گرہ->op);
    }
}

ثابت std::string & GgmlOvDecoder::get_op_type(صحیح گرہ_idx) ثابت {
    واپس m_گرہ_info_list[گرہ_idx].گرہ_op_type;
}

ثابت std::string & GgmlOvDecoder::get_op_type() ثابت {
    ساکن ثابت std::string unknown_op = "UNKNOWN_GGML_OP";
    واپس unknown_op;
}

خالی GgmlOvDecoder::compute_گرہ_dynamic_dims() {
    خودکار دورہ_گرہ = [&](خودکار && self, ggml_tensor * گرہ) -> خالی {
        اگر (!گرہ) {
            واپس;
        }

        اگر (گرہ->op == GGML_OP_CPY) {
            m_گرہ_dynamic_dims[گرہ] = -1;
        }

        اگر (m_گرہ_dynamic_dims.count(گرہ)) {
            واپس;
        }
        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = گرہ->src[i];
            اگر (src == خالی_اشاریہ) {
                جاری;
            }
            ساخت ggml_tensor * جڑ_src = خالی_اشاریہ;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            اگر (جڑ_src) {
                اگر (is_inp_tok(جڑ_src, گرہ) || is_inp_pos(جڑ_src, گرہ) || is_output_idx(جڑ_src, گرہ)) {
                    m_گرہ_dynamic_dims[جڑ_src] = 0;
                    m_گرہ_dynamic_dims[src] = m_گرہ_dynamic_dims[جڑ_src];
                    جاری;
                }
                self(self, جڑ_src);
                m_گرہ_dynamic_dims[src] = m_گرہ_dynamic_dims[جڑ_src];
            } ورنہ {
                اگر (is_inp_tok(src, گرہ) || is_inp_pos(src, گرہ) || is_output_idx(src, گرہ)) {
                    m_گرہ_dynamic_dims[src] = 0;
                    جاری;
                }
                اگر (is_inp_scale_صفیں(src, گرہ)) {
                    m_گرہ_dynamic_dims[src] = 1;
                    جاری;
                }
                اگر (گرہ->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_گرہ_dynamic_dims[src] = 1;
                    جاری;
                }
                self(self, src);
            }
        }
        انتخاب (گرہ->op) {
        حالت GGML_OP_NONE:
            m_گرہ_dynamic_dims[گرہ] = -1;
            توڑو;
        حالت GGML_OP_GET_صفیں:
            m_گرہ_dynamic_dims[گرہ] = -1;
            اگر (m_گرہ_dynamic_dims[گرہ->src[1]] != -1) {
                خودکار dynamic_dim_idx = m_گرہ_dynamic_dims[گرہ->src[1]];
                اگر (dynamic_dim_idx == 0) {
                    m_گرہ_dynamic_dims[گرہ] = 1;
                } ورنہ {
                    خودکار dynamic_dim_stride = گرہ->src[1]->nb[dynamic_dim_idx] / ggml_type_سائز(گرہ->src[1]->type) *
                                              ggml_type_سائز(گرہ->src[0]->type);
                    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                        اگر (dynamic_dim_stride == گرہ->src[0]->nb[i]) {
                            m_گرہ_dynamic_dims[گرہ] = i;
                            توڑو;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            توڑو;
        حالت GGML_OP_MUL:
        حالت GGML_OP_MUL_MAT:
            m_گرہ_dynamic_dims[گرہ] = -1;
            اگر (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[0]];
            }
            اگر (m_گرہ_dynamic_dims[گرہ->src[1]] != -1) {
                m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[1]];
            }
            توڑو;
        حالت GGML_OP_ترتیب_بدلو:
            m_گرہ_dynamic_dims[گرہ] = -1;
            اگر (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                خودکار dynamic_dim_idx = m_گرہ_dynamic_dims[گرہ->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                    اگر (گرہ->op_params[i] == dynamic_dim_idx) {
                        m_گرہ_dynamic_dims[گرہ] = i;
                        توڑو;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            توڑو;
        حالت GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_گرہ_dynamic_dims[گرہ] = -1;
            اگر (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                اگر (گرہ->src[0]->op == GGML_OP_NONE) {
                    m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[0]];
                    توڑو;
                }
                خودکار dynamic_dim_idx = m_گرہ_dynamic_dims[گرہ->src[0]];
                خودکار dynamic_dim_قدر = گرہ->src[0]->ne[dynamic_dim_idx];
                خودکار dynamic_dim_stride =
                    گرہ->src[0]->nb[dynamic_dim_idx] / ggml_type_سائز(گرہ->src[0]->type) * ggml_type_سائز(گرہ->type);
                برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                    اگر (گرہ->nb[i] == dynamic_dim_stride) {
                        m_گرہ_dynamic_dims[گرہ] = i;
                        توڑو;
                    }
                }
                اگر (m_گرہ_dynamic_dims[گرہ] != -1 && dynamic_dim_قدر != گرہ->ne[m_گرہ_dynamic_dims[گرہ]]) {
                    m_گرہ_dynamic_dims[گرہ] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    اگر (ggml_nelements(گرہ) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      گرہ->name, گرہ->src[0]->name);
                    }
                }
            }
            توڑو;
        }
        حالت GGML_OP_TRANSPOSE:
        حالت GGML_OP_RESHAPE: {
            اگر (is_same_shape(گرہ->src[0], گرہ)) {
                m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[0]];
                توڑو;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_گرہ_dynamic_dims[گرہ] = -1;
            اگر (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                خودکار dynamic_dim_idx = m_گرہ_dynamic_dims[گرہ->src[0]];
                خودکار dynamic_dim_stride = گرہ->src[0]->nb[dynamic_dim_idx];
                برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                    اگر (گرہ->nb[i] == dynamic_dim_stride && گرہ->ne[i] == گرہ->src[0]->ne[dynamic_dim_idx]) {
                        m_گرہ_dynamic_dims[گرہ] = i;
                        توڑو;
                    }
                }
                اگر (m_گرہ_dynamic_dims[گرہ] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", گرہ->name);
                }
            }
            توڑو;
        }
        حالت GGML_OP_FLASH_ATTN_EXT: {
            // Output shape is hard-coded in ggml_flash_attn_ext as:
            //   ne = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] }
            // i.e. output dim 0 <- v dim 0 (head_size, static)
            //      output dim 1 <- q dim 2 (n_heads,   static)
            //      output dim 2 <- q dim 1 (n_tokens,  potentially dynamic)
            //      output dim 3 <- q dim 3 (batch,     static)
            // Using the fixed q-dim -> output-dim mapping table.
            // q is src[0]; the mapping from q's dynamic dim to the output dim is:
            //   q dim 1 -> output dim 2
            //   q dim 2 -> output dim 1
            //   q dim 3 -> output dim 3
            //   q dim 0 -> output dim 0  (head_size axis, unlikely to be dynamic)
            constexpr صحیح q_مقصد_برآمد[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_گرہ_dynamic_dims[گرہ] = -1;
            اگر (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                خودکار q_dynamic_dim = m_گرہ_dynamic_dims[گرہ->src[0]];
                m_گرہ_dynamic_dims[گرہ] = q_مقصد_برآمد[q_dynamic_dim];
            }
            توڑو;
        }
        حالت GGML_OP_CONT:
            m_گرہ_dynamic_dims[گرہ] = -1;
            اگر (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                خودکار dynamic_dim_idx = m_گرہ_dynamic_dims[گرہ->src[0]];
                اگر (ggml_are_same_shape(گرہ, گرہ->src[0])) {
                    m_گرہ_dynamic_dims[گرہ] = dynamic_dim_idx;
                } ورنہ {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_سائز(گرہ->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (گرہ->src[0]->ne[0] / ggml_blck_سائز(گرہ->src[0]->type));
                    برائے (صحیح i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * گرہ->src[0]->ne[i - 1];
                    }

                    خودکار dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_سائز(گرہ->src[0]->type) *
                                              ggml_type_سائز(گرہ->type);
                    صحیح matched_dim_گنتی = 0;
                    صحیح پہلا_matched_dim = -1;
                    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                        اگر (گرہ->nb[i] == dynamic_dim_stride && گرہ->ne[i] == گرہ->src[0]->ne[dynamic_dim_idx]) {
                            اگر (پہلا_matched_dim == -1) {
                                پہلا_matched_dim = i;
                            }
                            m_گرہ_dynamic_dims[گرہ] = i;
                            matched_dim_گنتی++;
                        }
                    }
                    اگر (matched_dim_گنتی > 1 && گرہ->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_گرہ_dynamic_dims[گرہ] = پہلا_matched_dim;
                    } ورنہ اگر (matched_dim_گنتی != 1) {
                        m_گرہ_dynamic_dims[گرہ] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      گرہ->name, گرہ->src[0]->name);
                    }
                }
            }
            توڑو;
        حالت GGML_OP_CONCAT:
            برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                اگر (گرہ->src[0]->ne[i] != گرہ->ne[i]) {
                    m_گرہ_dynamic_dims[گرہ] = i;
                    توڑو;
                }
            }
            توڑو;
        حالت GGML_OP_SSM_CONV:
        حالت GGML_OP_GATED_DELTA_NET:
            m_گرہ_dynamic_dims[گرہ] = 1;
            توڑو;
        حالت GGML_OP_RMS_NORM:
        حالت GGML_OP_L2_NORM:
        حالت GGML_OP_NORM:
        حالت GGML_OP_جوڑو:
        حالت GGML_OP_SUB:
        حالت GGML_OP_GLU:
        حالت GGML_OP_ROPE:
        حالت GGML_OP_SCALE:
        حالت GGML_OP_SOFT_MAX:
        حالت GGML_OP_ARGSORT:
        حالت GGML_OP_جوڑو_ID:
        حالت GGML_OP_UNARY:
        حالت GGML_OP_CUMSUM:
        حالت GGML_OP_FILL:
        حالت GGML_OP_سیٹ:
        حالت GGML_OP_DIAG:
        حالت GGML_OP_TRI:
        حالت GGML_OP_REPEAT:
        حالت GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        حالت GGML_OP_DIV:
        حالت GGML_OP_CLAMP:
        حالت GGML_OP_PAD:
        حالت GGML_OP_UPSCALE:
        حالت GGML_OP_SIN:
        حالت GGML_OP_COS:
        حالت GGML_OP_LOG:
        حالت GGML_OP_ROLL:
            m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[0]];
            توڑو;
        حالت GGML_OP_مجموع_صفیں:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_گرہ_dynamic_dims[گرہ] =
                (m_گرہ_dynamic_dims[گرہ->src[0]] == 0) ? -1 : m_گرہ_dynamic_dims[گرہ->src[0]];
            توڑو;
        حالت GGML_OP_MUL_MAT_ID:
        حالت GGML_OP_SOLVE_TRI:
            m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[1]];
            توڑو;
        حالت GGML_OP_CPY:
        حالت GGML_OP_سیٹ_صفیں:
        حالت GGML_OP_مجموع:
        حالت GGML_OP_اوسط:
            m_گرہ_dynamic_dims[گرہ] = -1;
            توڑو;
        حالت GGML_OP_IM2_ستون: {
            m_گرہ_dynamic_dims[گرہ] = -1;
            اگر (m_گرہ_dynamic_dims[گرہ->src[1]] != -1) {
                ثابت منطقی is_2D = گرہ->op_params[6] == 1;
                ثابت صحیح src_dyn = m_گرہ_dynamic_dims[گرہ->src[1]];
                اگر (is_2D) {
                    اگر (src_dyn == 0) {
                        m_گرہ_dynamic_dims[گرہ] = 1;  // IW -> OW
                    } ورنہ اگر (src_dyn == 1) {
                        m_گرہ_dynamic_dims[گرہ] = 2;  // IH -> OH
                    } ورنہ اگر (src_dyn == 3) {
                        m_گرہ_dynamic_dims[گرہ] = 3;  // N  -> N
                    }
                } ورنہ {
                    اگر (src_dyn == 0) {
                        m_گرہ_dynamic_dims[گرہ] = 1;  // IW -> OW
                    } ورنہ اگر (src_dyn == 2) {
                        m_گرہ_dynamic_dims[گرہ] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                اگر (m_گرہ_dynamic_dims[گرہ] != -1) {
                    OPENVINO_ASSERT(گرہ->src[1]->ne[src_dyn] == گرہ->ne[m_گرہ_dynamic_dims[گرہ]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(گرہ->name) +
                                        " and its src[1]: " + std::string(گرہ->src[1]->name));
                }
            }
            توڑو;
        }
        حالت GGML_OP_IM2_ستون_3_D: {
            m_گرہ_dynamic_dims[گرہ] = -1;
            اگر (m_گرہ_dynamic_dims[گرہ->src[1]] != -1) {
                ثابت صحیح src_dyn = m_گرہ_dynamic_dims[گرہ->src[1]];
                اگر (src_dyn == 0) {
                    m_گرہ_dynamic_dims[گرہ] = 1;  // IW -> OW
                } ورنہ اگر (src_dyn == 1) {
                    m_گرہ_dynamic_dims[گرہ] = 2;  // IH -> OH
                } ورنہ اگر (src_dyn == 3) {
                    m_گرہ_dynamic_dims[گرہ] = 3;  // N  -> N
                }
                اگر (m_گرہ_dynamic_dims[گرہ] != -1) {
                    OPENVINO_ASSERT(گرہ->src[1]->ne[src_dyn] == گرہ->ne[m_گرہ_dynamic_dims[گرہ]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(گرہ->name) +
                                        " and its src[1]: " + std::string(گرہ->src[1]->name));
                }
            }
            توڑو;
        }
        طےشدہ:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(گرہ->op), گرہ->name);
            توڑو;
        }
    };

    برائے (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = m_cgraph->nodes[i];
        دورہ_گرہ(دورہ_گرہ, گرہ);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    اگر (0) {
        برائے (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * گرہ = m_cgraph->nodes[i];
            صحیح dynamic_dim = m_گرہ_dynamic_dims[گرہ];
            std::خارج << "[" << i << "] " << "node_name: " << گرہ->name << " op: " << ggml_op_name(گرہ->op)
                      << " shape: [";
            برائے (صحیح j = 0; j < 4; j++) {
                اگر (j == dynamic_dim) {
                    std::خارج << "*";
                } ورنہ {
                    std::خارج << گرہ->ne[j];
                }
                اگر (j < 3) {
                    std::خارج << ", ";
                }
            }
            std::خارج << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = گرہ->src[j];
                اگر (src == خالی_اشاریہ) {
                    جاری;
                }
                صحیح src_dynamic_dim = m_گرہ_dynamic_dims[src];
                std::خارج << "    [" << j << "] src_name: " << src->name << " [";
                برائے (صحیح k = 0; k < 4; k++) {
                    اگر (k == src_dynamic_dim) {
                        std::خارج << "*";
                    } ورنہ {
                        std::خارج << src->ne[k];
                    }
                    اگر (k < 3) {
                        std::خارج << ", ";
                    }
                }
                std::خارج << "]" << '\n';
            }
            std::خارج << '\n';
        }
    }
}
