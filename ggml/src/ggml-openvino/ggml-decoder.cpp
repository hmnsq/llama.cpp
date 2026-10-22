#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-decoder.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino-extra.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <iomanip>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/dimension.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/except.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/partial_shape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type/bfloat16.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type/element_type.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type/float16.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/tensor.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <ostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <set>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdexcept>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             ਤਾਰਕਿਕ is_static,
                             ਤਾਰਕਿਕ is_stateful,
                             ਤਾਰਕਿਕ model_is_splitted,
                             ਤਾਰਕਿਕ is_prefill,
                             ਪੂਰਨ prefill_chunk_ਮਾਪ) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(ਝੂਠ),
    m_prefill_chunk_ਮਾਪ(prefill_chunk_ਮਾਪ),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    ਸਥਿਰ ਤਾਰਕਿਕ printed_ਪਤਾ_map = ਝੂਠ;
    ਜੇ (!printed_ਪਤਾ_map) {
        ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_ਪਤਾ_map = ਸੱਚ;
            print_tensor_ਪਤਾ_map(cgraph);
        }
    }

    validate_cgraph();

    ਸੈੱਟ_ਨਿਵੇਸ਼_output();
    compute_ਗੰਢ_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    ਲਈ (ਪੂਰਨ ਗੰਢ_n = 0; ਗੰਢ_n < cgraph->n_nodes; ਗੰਢ_n++) {
        m_ਗੰਢ_info_list[ਗੰਢ_n].ਗੰਢ_op_case = compute_op_case(m_ਗੰਢ_info_list[ਗੰਢ_n].ਗੰਢ);
        m_ਗੰਢ_info_list[ਗੰਢ_n].ਗੰਢ_op_type = compute_op_type(m_ਗੰਢ_info_list[ਗੰਢ_n].ਗੰਢ);
    }

    ਜੋੜੋ_extra_inputs();
}

ਖਾਲੀ GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_ਗੰਢ_info_list.clear();
    ਸੈੱਟ_ਨਿਵੇਸ਼_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
ਤਾਰਕਿਕ GgmlOvDecoder::is_bound_ਮੰਜ਼ਿਲ(ਅਚਲ ggml_cgraph * cgraph) ਅਚਲ {
    ਵਾਪਸ m_cgraph == cgraph && cgraph->n_nodes > 0 && m_ਗੰਢ_info_list.size() == (size_t) cgraph->n_nodes &&
           m_ਗੰਢ_info_list.front().ਗੰਢ == cgraph->nodes[0] &&
           m_ਗੰਢ_info_list.back().ਗੰਢ == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = ਸੱਚ;
    ਸੈੱਟ_ਨਿਵੇਸ਼_output();
    compute_model_inputs();
    compute_model_outputs();
    ਲਈ (ਪੂਰਨ ਗੰਢ_n = 0; ਗੰਢ_n < cgraph->n_nodes; ਗੰਢ_n++) {
        m_ਗੰਢ_info_list[ਗੰਢ_n].ਗੰਢ_op_case = compute_op_case(m_ਗੰਢ_info_list[ਗੰਢ_n].ਗੰਢ);
        m_ਗੰਢ_info_list[ਗੰਢ_n].ਗੰਢ_op_type = compute_op_type(m_ਗੰਢ_info_list[ਗੰਢ_n].ਗੰਢ);
    }
}

ਨਾਮਾਕਾਸ਼ {
ਤਾਰਕਿਕ is_inplace_op(ਅਚਲ ggml_tensor * ਗੰਢ) {
    ਵਾਪਸ ਗੰਢ->op == GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ || ਗੰਢ->op == GGML_OP_CPY || (ਗੰਢ->op == GGML_OP_SCALE && ਗੰਢ->view_src);
}

ਤਾਰਕਿਕ is_same_shape(ਅਚਲ ggml_tensor * a, ਅਚਲ ggml_tensor * b) {
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
        ਜੇ (a->ne[i] != b->ne[i]) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ is_conv_ਹਾਲਤਾਂ_all_tensor(ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ tensor != ਖਾਲੀ_ਸੂਚਕ && ਲੜੀ_ਤੁਲਨਾ(tensor->name, "conv_states_all", ਲੜੀ_ਲੰਬਾਈ("conv_states_all")) == 0;
}

ਤਾਰਕਿਕ is_ਭਰਿਆ_single_ਥਾਂ_writeback(ਅਚਲ ggml_tensor * ਗੰਢ) {
    ਵਾਪਸ ਗੰਢ->view_src != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->view_src->ne[1] == 1 && ਗੰਢ->src[1] != ਖਾਲੀ_ਸੂਚਕ &&
           ਗੰਢ->src[1]->op == GGML_OP_VIEW && ਗੰਢ->src[1]->view_src == ਗੰਢ->view_src &&
           ਗੰਢ->src[1]->view_offs == 0 && ggml_nbytes(ਗੰਢ->src[1]) == ggml_nbytes(ਗੰਢ->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
ਤਾਰਕਿਕ is_moe_expert_ਜੋੜ_ਜੋੜੋ(ਅਚਲ ggml_tensor * ਗੰਢ) {
    std::vector<ਅਚਲ ggml_tensor *> leaves;
    ਅਚਲ ggml_tensor * cur = ਗੰਢ;
    ਜਦੋਂ (cur->op == GGML_OP_ਜੋੜੋ) {
        ਜੇ (cur->src[0] == ਖਾਲੀ_ਸੂਚਕ || cur->src[1] == ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ਝੂਠ;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ਅਚਲ ggml_tensor * ਅਧਾਰ = ਖਾਲੀ_ਸੂਚਕ;
    std::set<int64_t> plane_indices;
    ਲਈ (ਅਚਲ ggml_tensor * ਪੱਤੀ : leaves) {
        ਜੇ (ਪੱਤੀ->op != GGML_OP_VIEW || ਪੱਤੀ->src[0] == ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ ggml_tensor * ਪੱਤੀ_ਅਧਾਰ = ਪੱਤੀ->src[0];
        ਜੇ (ਅਧਾਰ == ਖਾਲੀ_ਸੂਚਕ) {
            ਅਧਾਰ = ਪੱਤੀ_ਅਧਾਰ;
        } ਨਹੀਂਤਾਂ ਜੇ (ਪੱਤੀ_ਅਧਾਰ != ਅਧਾਰ) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ਪੱਤੀ->ne[0] != ਅਧਾਰ->ne[0] || ਪੱਤੀ->ne[1] != ਅਧਾਰ->ne[2] || ਪੱਤੀ->ne[2] != 1 || ਪੱਤੀ->ne[3] != 1 ||
            ਪੱਤੀ->nb[1] != ਅਧਾਰ->nb[2]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ਅਧਾਰ->nb[1] == 0 || ਪੱਤੀ->view_offs % ਅਧਾਰ->nb[1] != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
        int64_t plane = static_cast<int64_t>(ਪੱਤੀ->view_offs / ਅਧਾਰ->nb[1]);
        ਜੇ (plane < 0 || plane >= ਅਧਾਰ->ne[1] || !plane_indices.insert(plane).second) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਅਧਾਰ != ਖਾਲੀ_ਸੂਚਕ && ਅਧਾਰ->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(ਅਧਾਰ->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ਅਚਲ ggml_cgraph * cgraph, ਅਚਲ ggml_tensor * tensor) {
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ "";
    }
    ਜੇ ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, ਖਾਲੀ_ਸੂਚਕ)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ਅਚਲ ਸਵੈਚਾਲਿਤ * ਗੰਢ = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        ਜੇ (ਗੰਢ != cgraph->nodes + cgraph->n_nodes) {
            ਵਾਪਸ std::string(tensor->name) + "#n" + std::to_string(ਗੰਢ - cgraph->nodes);
        }
        ਅਚਲ ਸਵੈਚਾਲਿਤ * ਪੱਤੀ = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        ਜੇ (ਪੱਤੀ != cgraph->leafs + cgraph->n_leafs) {
            ਵਾਪਸ std::string(tensor->name) + "#l" + std::to_string(ਪੱਤੀ - cgraph->leafs);
        }
    }
    ਵਾਪਸ tensor->name;
}

ਸਥਿਰ std::string get_tensor_ov_name(ਅਚਲ ggml_cgraph * cgraph, ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

ਸਥਿਰ std::string get_tensor_ਗ੍ਰਾਫ_ਨਿਵੇਸ਼_ov_name(ਅਚਲ GgmlOvDecoder * decoder,
                                                  ਅਚਲ ggml_cgraph * cgraph,
                                                  ਅਚਲ ggml_tensor * tensor,
                                                  ਅਚਲ ggml_tensor * op) {
    ਜੇ (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        ਵਾਪਸ "inp_pos";
    }
    ਜੇ (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        ਵਾਪਸ "embd";
    }
    ਜੇ (GgmlOvDecoder::is_inp_ਨਕਾਬ(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ਅਚਲ ਤਾਰਕਿਕ is_swa = decoder->is_swa_ਨਕਾਬ(tensor);
        ਜੇ (decoder->is_stateful()) {
            ਵਾਪਸ is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        ਜੇ (is_swa) {
            ਵਾਪਸ get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    ਵਾਪਸ get_tensor_ov_name(cgraph, tensor);
}

ਖਾਲੀ GgmlOvDecoder::ਸੈੱਟ_ਨਿਵੇਸ਼_output() {
    ਲਈ (ਪੂਰਨ ਗੰਢ_n = 0; ਗੰਢ_n < m_cgraph->n_nodes; ਗੰਢ_n++) {
        ਸਵੈਚਾਲਿਤ * ਗੰਢ = m_cgraph->nodes[ਗੰਢ_n];

        ਗੰਢ_Info current_ਗੰਢ_info;
        ਸਵੈਚਾਲਿਤ ਗੰਢ_name = get_tensor_ov_name(m_cgraph, ਗੰਢ);

        current_ਗੰਢ_info.ਗੰਢ = ਗੰਢ;
        current_ਗੰਢ_info.ਗੰਢ_name = ਗੰਢ_name;
        current_ਗੰਢ_info.ਗੰਢ_op_case = 0;
        current_ਗੰਢ_info.ਅੰਕੜੇ_addr = ਗੰਢ->ਅੰਕੜੇ;

        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਸਵੈਚਾਲਿਤ * src = ਗੰਢ->src[i];
            ਜੇ (src == ਖਾਲੀ_ਸੂਚਕ) {
                ਜਾਰੀ;
            }
            ਸਵੈਚਾਲਿਤ src_name = get_tensor_ov_name(m_cgraph, src);
            ਜੇ (src->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼) {
                src_name = get_tensor_ਗ੍ਰਾਫ_ਨਿਵੇਸ਼_ov_name(ਇਹ, m_cgraph, src, ਗੰਢ);
            }
            current_ਗੰਢ_info.ਗੰਢ_inputs[src_name] = src;
            current_ਗੰਢ_info.ਗੰਢ_inputs_ਨਾਂ.push_back(src_name);

            ਜੇ (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_ਗੰਢ_info.ਗੰਢ_inputs_views[src_name])> view_chain;
                ਸਵੈਚਾਲਿਤ * current = src;

                ਜਦੋਂ (current != ਖਾਲੀ_ਸੂਚਕ) {
                    ਸਵੈਚਾਲਿਤ current_name = get_tensor_ov_name(m_cgraph, current);
                    ਜੇ (current->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼) {
                        current_name = get_tensor_ਗ੍ਰਾਫ_ਨਿਵੇਸ਼_ov_name(ਇਹ, m_cgraph, current, ਗੰਢ);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    ਜੇ (current->src[0] != ਖਾਲੀ_ਸੂਚਕ && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } ਨਹੀਂਤਾਂ {
                        ਰੋਕੋ;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_ਗੰਢ_info.ਗੰਢ_inputs_views[src_name] = view_chain;
            }
        }

        m_ਗੰਢ_info_list.push_back(current_ਗੰਢ_info);
    }
}

ਪੂਰਨ GgmlOvDecoder::compute_op_case(ਅਚਲ ggml_tensor * ਗੰਢ) ਅਚਲ {
    ਪੂਰਨ op_case = 0;
    ਚੋਣ (ਗੰਢ->op) {
    ਹਾਲਤ GGML_OP_RESHAPE: {
        ਜੇ (m_naive) {
            ਰੋਕੋ;
        }
        ਸਵੈਚਾਲਿਤ name = std::string(ਗੰਢ->name);
        ਸਵੈਚਾਲਿਤ * src = ਗੰਢ->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        ਤਾਰਕਿਕ recurrent_sequence = ਝੂਠ;
        ਲਈ (ਪੂਰਨ i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * ਖਪਤਕਾਰ = m_cgraph->nodes[i];
            ਜੇ (ਖਪਤਕਾਰ->op == GGML_OP_MUL_MAT_ID && ਖਪਤਕਾਰ->src[1] == ਗੰਢ) {
                ਵਾਪਸ 1;
            } ਨਹੀਂਤਾਂ ਜੇ (ਖਪਤਕਾਰ->op == GGML_OP_SSM_CONV) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ * concat = ਖਪਤਕਾਰ->src[0];
                ਜੇ (concat->op == GGML_OP_CONCAT) {
                    ਅਚਲ ਸਵੈਚਾਲਿਤ * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == ਗੰਢ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (ਖਪਤਕਾਰ->op == GGML_OP_UNARY && ggml_get_unary_op(ਖਪਤਕਾਰ) == GGML_UNARY_OP_SOFTPLUS) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ * biased = ਖਪਤਕਾਰ->src[0];
                recurrent_sequence = biased->op == GGML_OP_ਜੋੜੋ && biased->src[0] == ਗੰਢ;
            }
        }
        ਜੇ (recurrent_sequence && ਗੰਢ->ne[0] == src->ne[0] && ਗੰਢ->ne[3] == 1) {
            ਵਾਪਸ 6;
        }
        ਜੇ (ਗੰਢ->ne[0] == src->ne[0] && ਗੰਢ->ne[2] == 1 && ਗੰਢ->ne[3] == 1) {
            ਵਾਪਸ 5;
        }
        ਜੇ (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == ਗੰਢ->ne[0] && src->src[0]->ne[1] == ਗੰਢ->ne[1]) {
            op_case = 4;
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->ne[0] * ਗੰਢ->ne[1] == src->ne[0]) {
            op_case = 1;
        } ਨਹੀਂਤਾਂ ਜੇ (src->ne[0] * src->ne[1] == ਗੰਢ->ne[0]) {
            op_case = 2;
            ਜੇ (src->ne[2] * src->ne[3] == ਗੰਢ->ne[1]) {
                op_case = 5;
            }
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == ਗੰਢ->ne[1]) {
            op_case = 3;
        } ਨਹੀਂਤਾਂ ਜੇ (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } ਨਹੀਂਤਾਂ ਜੇ (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } ਨਹੀਂਤਾਂ ਜੇ (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        ਜੇ (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            ਅਚਲ ਸਵੈਚਾਲਿਤ * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            ਤਾਰਕਿਕ recurrent = gather->op == GGML_OP_GET_ਪੰਕਤੀਆਂ && is_recurrent_cache(gather->src[0]);
            ਲਈ (ਪੂਰਨ i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ * ਖਪਤਕਾਰ = m_cgraph->nodes[i];
                ਜੇ (ਖਪਤਕਾਰ->op == GGML_OP_GATED_DELTA_NET) {
                    ਲਈ (ਪੂਰਨ j : {3, 4}) {
                        ਅਚਲ ਸਵੈਚਾਲਿਤ * gate = ਖਪਤਕਾਰ->src[j];
                        ਜੇ (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == ਗੰਢ;
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (ਖਪਤਕਾਰ->op == GGML_OP_MUL) {
                    ਲਈ (ਪੂਰਨ j = 0; j < 2; ++j) {
                        ਅਚਲ ਸਵੈਚਾਲਿਤ * gate = ਖਪਤਕਾਰ->src[j];
                        ਅਚਲ ਸਵੈਚਾਲਿਤ * norm = ਖਪਤਕਾਰ->src[1 - j];
                        ਜੇ (gate->op != GGML_OP_UNARY || gate->src[0] != ਗੰਢ) {
                            ਜਾਰੀ;
                        }
                        ਜੇ (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            ਜੇ (recurrent) {
                op_case = 9;
            }
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ: {
        ਜੇ (ਗੰਢ->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(ਗੰਢ->view_src->name);
            ਪੂਰਨ layer = ਕੱਢੋ_layer_from_name(src_name).value();
            ਜੇ (ggml_is_contiguous(ਗੰਢ->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                ਜੇ (!is_swa_layer(layer)) {
                    op_case = 3;
                } ਨਹੀਂਤਾਂ {
                    op_case = 4;
                }
            } ਨਹੀਂਤਾਂ {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                ਜੇ (!is_swa_layer(layer)) {
                    op_case = 5;
                } ਨਹੀਂਤਾਂ {
                    op_case = 6;
                }
            }
        } ਨਹੀਂਤਾਂ {
            // rope'ed query tensor
            op_case = 2;
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_MUL_MAT: {
        ਜੇ (ਗੰਢ->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        ਜੇ (ਗੰਢ->src[0]->op == GGML_OP_RESHAPE && ਗੰਢ->src[0]->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
            is_recurrent_cache(ਗੰਢ->src[0]->src[0])) {
            ਅਚਲ ਤਾਰਕਿਕ single_ਥਾਂ = ਗੰਢ->src[0]->src[0]->ne[1] == 1;
            ਜੇ (ਗੰਢ->src[1]->op == GGML_OP_VIEW) {
                op_case = (ਗੰਢ->src[1]->view_offs == 0 ? 1 : 2) + (single_ਥਾਂ ? 2 : 0);
            } ਨਹੀਂਤਾਂ ਜੇ (single_ਥਾਂ) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_ROPE: {
        ਅਚਲ ਪੂਰਨ mode = ਗੰਢ->op_params[2];
        ਚੋਣ (mode) {
        ਹਾਲਤ GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            ਰੋਕੋ;
        }
        ਮੂਲ:
            op_case = 0;
            ਰੋਕੋ;
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_VIEW: {
        ਜੇ (!m_model_params.has_rs_rollback && ਗੰਢ->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
            ਗੰਢ->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            ਰੋਕੋ;
        }
        ਜੇ (m_is_static && ਗੰਢ->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
            (ਗੰਢ->src[0]->op == GGML_OP_GATED_DELTA_NET || ਗੰਢ->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            ਰੋਕੋ;
        }
        ਜੇ (ਗੰਢ->src[0]->op == GGML_OP_VIEW) {
            ਸਵੈਚਾਲਿਤ * src = ਗੰਢ->src[0];
            ਜੇ (ggml_nelements(ਗੰਢ) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            ਜੇ (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            ਸਵੈਚਾਲਿਤ * src = ਗੰਢ->src[0];
            ਜੇ (ggml_nelements(ਗੰਢ) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                ਜੇ (ਗੰਢ->ne[0] == src->ne[0] && ਗੰਢ->ne[1] == src->ne[2] && ਗੰਢ->ne[2] == 1 &&
                    ਗੰਢ->nb[0] == src->nb[0] && ਗੰਢ->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    ਰੋਕੋ;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                ਪੂਰਨ diff_ਗਿਣਤੀ = 0;
                ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
                    ਜੇ (ਗੰਢ->ne[i] != src->ne[i]) {
                        diff_ਗਿਣਤੀ++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    ਜੇ (ਗੰਢ->ne[i] > src->ne[i]) {
                        ਵਾਪਸ 0;
                    }
                }
                ਜੇ (diff_ਗਿਣਤੀ >= 1) {
                    op_case = 0;
                }
            }
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_RMS_NORM: {
        ਜੇ (ਗੰਢ->src[0]->op == GGML_OP_VIEW) {
            ਜੇ (is_same_shape(ਗੰਢ->src[0]->src[0], ਗੰਢ->src[0])) {
                op_case = 1;
            } ਨਹੀਂਤਾਂ ਜੇ (!m_model_params.has_rs_rollback &&
                       ਗੰਢ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_POOL_2D: {
        ਅਚਲ ggml_op_pool pool_ਬਹੁਲਕ = static_cast<ggml_op_pool>(ਗੰਢ->op_params[0]);
        ਚੋਣ (pool_ਬਹੁਲਕ) {
        ਹਾਲਤ GGML_OP_POOL_MAX: {
            op_case = 1;
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_OP_POOL_AVG: {
            op_case = 2;
            ਰੋਕੋ;
        }
        ਮੂਲ:
            op_case = 0;
            ਰੋਕੋ;
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_UPSCALE: {
        ਅਚਲ int32_t ਬਹੁਲਕ_flags = ਗੰਢ->op_params[0];
        ਅਚਲ ggml_scale_ਬਹੁਲਕ scale_ਬਹੁਲਕ = static_cast<ggml_scale_ਬਹੁਲਕ>(ਬਹੁਲਕ_flags & 0xFF);
        ਚੋਣ (scale_ਬਹੁਲਕ) {
        ਹਾਲਤ GGML_SCALE_ਬਹੁਲਕ_ਸਭ_ਤੋਂ_ਨੇੜੇ: {
            op_case = 1;
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_SCALE_ਬਹੁਲਕ_BILINEAR: {
            op_case = 2;
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_SCALE_ਬਹੁਲਕ_BICUBIC: {
            op_case = 3;
            ਰੋਕੋ;
        }
        ਮੂਲ:
            op_case = 0;
            ਰੋਕੋ;
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_CPY: {
        ਜੇ (ਗੰਢ->src[0]->op == GGML_OP_VIEW) {
            ਜੇ (ਗੰਢ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                ਜੇ (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_ਭਰਿਆ_single_ਥਾਂ_writeback(ਗੰਢ) ? 7 : 10;
                } ਨਹੀਂਤਾਂ {
                    op_case = 1;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (GgmlOvDecoder::is_conv_state_writeback(ਗੰਢ)) {
                op_case = is_ਭਰਿਆ_single_ਥਾਂ_writeback(ਗੰਢ) ? 8 : 2;
                ਰੋਕੋ;
            } ਨਹੀਂਤਾਂ ਜੇ (is_conv_ਹਾਲਤਾਂ_all_tensor(ਗੰਢ->view_src) && ਗੰਢ->src[1] != ਖਾਲੀ_ਸੂਚਕ &&
                       ਗੰਢ->src[1]->op == GGML_OP_VIEW && ਗੰਢ->src[1]->view_src == ਗੰਢ->view_src) {
                op_case = 4;
                ਰੋਕੋ;
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->src[0]->src[0]->op == GGML_OP_GET_ਪੰਕਤੀਆਂ && ਗੰਢ->src[1] != ਖਾਲੀ_ਸੂਚਕ &&
                       ਗੰਢ->src[1]->op == GGML_OP_VIEW && ਗੰਢ->src[1]->view_src != ਖਾਲੀ_ਸੂਚਕ &&
                       is_recurrent_cache(ਗੰਢ->src[1]->view_src) && ਗੰਢ->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->src[0]->op == GGML_OP_GET_ਪੰਕਤੀਆਂ && ਗੰਢ->src[1] != ਖਾਲੀ_ਸੂਚਕ &&
                   ਗੰਢ->src[1]->op == GGML_OP_VIEW && ਗੰਢ->src[1]->view_src != ਖਾਲੀ_ਸੂਚਕ &&
                   is_recurrent_cache(ਗੰਢ->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = ਗੰਢ->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->src[1] != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->src[1]->op == GGML_OP_VIEW && ਗੰਢ->src[1]->view_src != ਖਾਲੀ_ਸੂਚਕ) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ਅਚਲ ggml_tensor * kv_buf = ਗੰਢ->src[1]->view_src;
            ਜੇ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                ਲਈ (ਪੂਰਨ i = 0; i < m_cgraph->n_nodes; i++) {
                    ਅਚਲ ggml_tensor * n = m_cgraph->nodes[i];
                    ਜੇ (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        ਜਾਰੀ;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    ਜੇ ((n->src[1] != ਖਾਲੀ_ਸੂਚਕ && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != ਖਾਲੀ_ਸੂਚਕ && n->src[2]->view_src == kv_buf)) {
                        ਜੇ (n->src[3] != ਖਾਲੀ_ਸੂਚਕ) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        ਰੋਕੋ;
                    }
                }
            }
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_ਜੋੜੋ: {
        ਜੇ (is_moe_expert_ਜੋੜ_ਜੋੜੋ(ਗੰਢ)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_SCALE: {
        ਜੇ (ਗੰਢ->view_src && ਗੰਢ->ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_ANY) {
            op_case = ਗੰਢ->view_src->ne[1] == 1 ? 2 : 1;
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_L2_NORM: {
        ਜੇ (std::string(ਗੰਢ->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT: {
        ਜੇ (ਗੰਢ->src[1] != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->src[1]->op == GGML_OP_VIEW && ਗੰਢ->src[1]->view_src != ਖਾਲੀ_ਸੂਚਕ) {
            ਅਚਲ ggml_tensor * kv_buf = ਗੰਢ->src[1]->view_src;
            ਜੇ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (ਗੰਢ->src[3] != ਖਾਲੀ_ਸੂਚਕ) ? 1 : 2;
            }
        }
        ਰੋਕੋ;
    }
    ਮੂਲ:
        ਰੋਕੋ;
    }
    ਵਾਪਸ op_case;
}

std::optional<ਪੂਰਨ> ਕੱਢੋ_layer_from_name(ਅਚਲ std::string & name) {
    size_t pos1 = name.find("_l");
    ਜੇ (pos1 == std::string::npos) {
        ਵਾਪਸ std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    ਜੇ (pos2 == std::string::npos) {
        pos2 = name.ਲੰਮਾਈ();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    ਪੂਰਨ layer = std::stoi(layer_str);
    ਵਾਪਸ layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
ਸਥਿਰ ਪੂਰਨ get_swa_ਖਿੜਕੀ_from_ਨਕਾਬ(ਅਚਲ ggml_tensor * ਨਕਾਬ) {
    ਜੇ (ਨਕਾਬ->ਅੰਕੜੇ == ਖਾਲੀ_ਸੂਚਕ || !ggml_backend_ਬਫਰ_is_host(ਨਕਾਬ->ਬਫਰ)) {
        ਵਾਪਸ -1;
    }
    ਜੇ (ਨਕਾਬ->type != GGML_TYPE_F16 && ਨਕਾਬ->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t n_kv = ਨਕਾਬ->ne[0];
    ਅਚਲ int64_t n_ਟੋਕਨ = ਨਕਾਬ->ne[1];
    int64_t ਖਿੜਕੀ = 0;

    ਲਈ (int64_t r = 0; r < n_ਟੋਕਨ; r++) {
        int64_t kept = 0;
        ਲਈ (int64_t c = 0; c < n_kv; c++) {
            ਅਚਲ size_t i = (size_t) r * n_kv + c;
            ਅਚਲ ਦਸ਼ਮਲਵ v = ਨਕਾਬ->type == GGML_TYPE_F16 ? ggml_fp16_ਮੰਜ਼ਿਲ_fp32(((ਅਚਲ ggml_fp16_t *) ਨਕਾਬ->ਅੰਕੜੇ)[i]) :
                                                          ((ਅਚਲ ਦਸ਼ਮਲਵ *) ਨਕਾਬ->ਅੰਕੜੇ)[i];
            ਜੇ (v > -INFINITY) {
                kept++;
            }
        }
        ਖਿੜਕੀ = std::max(ਖਿੜਕੀ, kept);
    }

    ਵਾਪਸ ਖਿੜਕੀ > 0 ? (ਪੂਰਨ) ਖਿੜਕੀ : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, ਤਾਰਕਿਕ is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    ਸਵੈਚਾਲਿਤ get_attention_ਨਮੂਨਾ_case = [](ਅਚਲ ggml_tensor * ਗੰਢ) -> ਪੂਰਨ {
        ਜੇ (ਗੰਢ == ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ -1;
        }

        ਚੋਣ (ਗੰਢ->op) {
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            ਜੇ (ਗੰਢ->src[0] == ਖਾਲੀ_ਸੂਚਕ || ਗੰਢ->src[1] == ਖਾਲੀ_ਸੂਚਕ) {
                ਵਾਪਸ -1;
            }
            ਚੋਣ (ਗੰਢ->src[1]->op) {
            ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                ਜੇ (ਗੰਢ->src[3] != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->src[1]->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
                    ਗੰਢ->src[1]->src[0]->op == GGML_OP_VIEW) {
                    ਵਾਪਸ 0;
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                ਜੇ (ਗੰਢ->src[3] != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->src[1]->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
                    ਗੰਢ->src[1]->src[0]->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ && ਗੰਢ->src[1]->src[0]->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
                    ਗੰਢ->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    ਵਾਪਸ 1;
                }
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                ਜੇ (ਗੰਢ->src[1]->view_src != ਖਾਲੀ_ਸੂਚਕ) {
                    ਜੇ (ਗੰਢ->src[3] != ਖਾਲੀ_ਸੂਚਕ) {
                        ਵਾਪਸ 4;  // decoder self-attention
                    }
                    ਵਾਪਸ 5;      // cross-attention or encoder self-attention
                }
                ਰੋਕੋ;
            ਮੂਲ:
                ਰੋਕੋ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            ਜੇ (ਗੰਢ->src[0] != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->src[1] != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->src[0]->op == GGML_OP_MUL_MAT &&
                ਗੰਢ->src[0]->src[0] != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->src[0]->src[1] != ਖਾਲੀ_ਸੂਚਕ &&
                ਗੰਢ->src[0]->src[0]->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ && ਗੰਢ->src[0]->src[0]->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
                ਗੰਢ->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                ਵਾਪਸ 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            ਜੇ (ਗੰਢ->src[0]->op == GGML_OP_ਜੋੜੋ && ਗੰਢ->src[0]->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
                ਗੰਢ->src[0]->src[0]->op == GGML_OP_MUL_MAT && ਗੰਢ->src[0]->src[0]->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
                ਗੰਢ->src[0]->src[0]->src[0]->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ) {
                ਵਾਪਸ 3;
            }
            ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
        }

        ਵਾਪਸ -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    ਸਵੈਚਾਲਿਤ get_attention_op_ਨਕਾਬ = [&get_attention_ਨਮੂਨਾ_case](ਅਚਲ ggml_tensor * ਗੰਢ) -> ਅਚਲ ggml_tensor * {
        ਚੋਣ (get_attention_ਨਮੂਨਾ_case(ਗੰਢ)) {
        ਹਾਲਤ 0:
        ਹਾਲਤ 1:
            ਵਾਪਸ ਗੰਢ->src[3];
        ਹਾਲਤ 2:
        ਹਾਲਤ 3:
            ਵਾਪਸ ਗੰਢ->src[1];
        ਮੂਲ:
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
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
        std::map<ਪੂਰਨ, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<ਪੂਰਨ, ਅਚਲ ggml_tensor *> layer_ਨਕਾਬ;            // layer -> mask it consumes
        int64_t max_extent = 0;

        ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
            ਅਚਲ ggml_tensor * ਨਕਾਬ = get_attention_op_ਨਕਾਬ(cgraph->nodes[i]);
            ਜੇ (ਨਕਾਬ == ਖਾਲੀ_ਸੂਚਕ) {
                ਜਾਰੀ;
            }
            ਅਚਲ ggml_tensor * cache_k_ਕ੍ਰਮ_ਬਦਲੋ = ਖਾਲੀ_ਸੂਚਕ;
            ਚੋਣ (get_attention_ਨਮੂਨਾ_case(cgraph->nodes[i])) {
            ਹਾਲਤ 0:  cache_k_ਕ੍ਰਮ_ਬਦਲੋ = cgraph->nodes[i]->src[1];                     ਰੋਕੋ;
            ਹਾਲਤ 1:  cache_k_ਕ੍ਰਮ_ਬਦਲੋ = cgraph->nodes[i]->src[1]->src[0];             ਰੋਕੋ;
            ਹਾਲਤ 2:  cache_k_ਕ੍ਰਮ_ਬਦਲੋ = cgraph->nodes[i]->src[0]->src[0];             ਰੋਕੋ;
            ਮੂਲ: cache_k_ਕ੍ਰਮ_ਬਦਲੋ = cgraph->nodes[i]->src[0]->src[0]->src[0];     ਰੋਕੋ;
            }
            ਅਚਲ ggml_tensor * cache_k_view = cache_k_ਕ੍ਰਮ_ਬਦਲੋ->src[0];
            ਜੇ (cache_k_view->op != GGML_OP_VIEW) {
                ਜਾਰੀ;
            }
            ਅਚਲ ggml_tensor * ਪੱਤੀ = cache_k_view->src[0];
            ਸਵੈਚਾਲਿਤ layer = ਕੱਢੋ_layer_from_name(ਪੱਤੀ->name);
            ਜੇ (!layer.has_value()) {
                ਜਾਰੀ;
            }
            layer_extent[layer.value()] = ਪੱਤੀ->ne[1];
            layer_ਨਕਾਬ[layer.value()] = ਨਕਾਬ;
            max_extent = std::max(max_extent, ਪੱਤੀ->ne[1]);
        }

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & [layer, extent] : layer_extent) {
            ਜੇ (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                ਜੇ (model_params.swa_ਨਕਾਬ == ਖਾਲੀ_ਸੂਚਕ) {
                    model_params.swa_ਨਕਾਬ = layer_ਨਕਾਬ[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string ਪ੍ਰਤੀ_layer;
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & [layer, extent] : layer_extent) {
                ਪ੍ਰਤੀ_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (ਲੰਮਾ) max_extent, model_params.swa_layers.size(), ਪ੍ਰਤੀ_layer.c_str());
        }
    }

    ਤਾਰਕਿਕ rope_ਵੇਖਿਆ = ਝੂਠ;
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ਗੰਢ = cgraph->nodes[i];
        ਅਚਲ ਪੂਰਨ attention_ਨਮੂਨਾ_case = get_attention_ਨਮੂਨਾ_case(ਗੰਢ);
        ਜੇ (attention_ਨਮੂਨਾ_case != -1) {
            ggml_tensor * cache_k_ਕ੍ਰਮ_ਬਦਲੋ = ਖਾਲੀ_ਸੂਚਕ;
            ggml_tensor * ਨਕਾਬ = ਖਾਲੀ_ਸੂਚਕ;

            ਚੋਣ (attention_ਨਮੂਨਾ_case) {
            ਹਾਲਤ 0:
                cache_k_ਕ੍ਰਮ_ਬਦਲੋ = ਗੰਢ->src[1];
                ਨਕਾਬ = ਗੰਢ->src[3];
                ਰੋਕੋ;
            ਹਾਲਤ 1:
                cache_k_ਕ੍ਰਮ_ਬਦਲੋ = ਗੰਢ->src[1]->src[0];
                ਨਕਾਬ = ਗੰਢ->src[3];
                ਰੋਕੋ;
            ਹਾਲਤ 2:
                cache_k_ਕ੍ਰਮ_ਬਦਲੋ = ਗੰਢ->src[0]->src[0];
                ਨਕਾਬ = ਗੰਢ->src[1];
                ਰੋਕੋ;
            ਹਾਲਤ 3:
                cache_k_ਕ੍ਰਮ_ਬਦਲੋ = ਗੰਢ->src[0]->src[0]->src[0];
                ਨਕਾਬ = ਗੰਢ->src[1];
                ਰੋਕੋ;
            ਹਾਲਤ 4:
            ਹਾਲਤ 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                ਸਵੈਚਾਲਿਤ * cache_k_view = ਗੰਢ->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_ਪ੍ਰਤੀ_seq = ਗੰਢ->src[0]->ne[1];
                ਜੇ (attention_ਨਮੂਨਾ_case == 4) {
                    compute_params.attention_ਮਾਪ = cache_k_view->ne[1];
                } ਨਹੀਂਤਾਂ {
                    compute_params.attention_ਮਾਪ_static = cache_k_view->ne[1];
                }
                ਜਾਰੀ;
            }
            ਮੂਲ:
                ਰੋਕੋ;
            }

            ਪ੍ਰਤਿਜ੍ਞਾ(cache_k_ਕ੍ਰਮ_ਬਦਲੋ != ਖਾਲੀ_ਸੂਚਕ);

            model_params.ਸਿਰ_ਮਾਪ = cache_k_ਕ੍ਰਮ_ਬਦਲੋ->ne[0];
            model_params.n_heads_kv = cache_k_ਕ੍ਰਮ_ਬਦਲੋ->ne[2];
            compute_params.ਨਿਵੇਸ਼_len = ਗੰਢ->src[0]->ne[1];
            compute_params.token_len_ਪ੍ਰਤੀ_seq = ਗੰਢ->src[0]->ne[1];

            ਸਵੈਚਾਲਿਤ * cache_k_view = cache_k_ਕ੍ਰਮ_ਬਦਲੋ->src[0];
            ਜੇ (cache_k_view->op != GGML_OP_VIEW || ਨਕਾਬ == ਖਾਲੀ_ਸੂਚਕ) {
                ਜਾਰੀ;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            ਪੂਰਨ layer = ਕੱਢੋ_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ਅਚਲ ਤਾਰਕਿਕ layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_ਬਫਰ_ctx_id = ggml_backend_openvino_ਬਫਰ_get_ctx_id(cache_k->ਬਫਰ);
            model_params.n_heads_kv_ਪ੍ਰਤੀ_layer[layer] = cache_k_ਕ੍ਰਮ_ਬਦਲੋ->ne[2];
            ਜੇ (layer_is_swa) {
                model_params.ctx_ਪ੍ਰਤੀ_seq_swa = cache_k->ne[1];
            } ਨਹੀਂਤਾਂ {
                model_params.ctx_ਪ੍ਰਤੀ_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_ਸਰਗਰਮ = ਨਕਾਬ->ne[3];
            ਸਵੈਚਾਲਿਤ seq_ਮਾਪ = cache_k->ne[0] * cache_k->ne[1] * ggml_type_ਮਾਪ(cache_k->type);
            size_t ਵਿੱਥ;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਵਿੱਥ, cache_k_view->op_params, ਆਕਾਰ(size_t));
            compute_params.seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ = ਵਿੱਥ / seq_ਮਾਪ;

            ਜੇ (layer_is_swa) {
                compute_params.attention_ਮਾਪ_swa = ਨਕਾਬ->ne[0];
                compute_params.swa_ਖਿੜਕੀ = get_swa_ਖਿੜਕੀ_from_ਨਕਾਬ(ਨਕਾਬ);
            } ਨਹੀਂਤਾਂ {
                compute_params.attention_ਮਾਪ = ਨਕਾਬ->ne[0];
            }
            ਜੇ (is_static) {
                compute_params.attention_ਮਾਪ = model_params.ctx_ਪ੍ਰਤੀ_seq;
                compute_params.attention_ਮਾਪ_swa = model_params.ctx_ਪ੍ਰਤੀ_seq_swa;
                compute_params.token_len_ਪ੍ਰਤੀ_seq = 1;
            }
        }

        ਜੇ (ਗੰਢ->op == GGML_OP_MUL_MAT && ਗੰਢ->src[0]->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ &&
            ਗੰਢ->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(ਗੰਢ->src[0]->view_src, ਗੰਢ->view_src)) {
            ਜੇ (ਗੰਢ->src[1]->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ && ਗੰਢ->src[1]->src[0]->op == GGML_OP_VIEW &&
                ਗੰਢ->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_ਮਾਪ = ਗੰਢ->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        ਜੇ (ਗੰਢ->op == GGML_OP_TRANSPOSE && ਗੰਢ->src[0]->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ &&
            ਗੰਢ->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_ਮਾਪ = ਗੰਢ->ne[0];
            ਜੇ (is_static) {
                compute_params.attention_ਮਾਪ = model_params.ctx_ਪ੍ਰਤੀ_seq;
            }
        }
        ਜੇ (ਗੰਢ->op == GGML_OP_ROPE) {
            ਜੇ (compute_params.token_len_ਪ੍ਰਤੀ_seq == -1 && ਗੰਢ->src[1] != ਖਾਲੀ_ਸੂਚਕ) {
                compute_params.token_len_ਪ੍ਰਤੀ_seq = ggml_nelements(ਗੰਢ->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(ਆਕਾਰ(model_params.rope_params) == ਆਕਾਰ(int32_t) * 16, "rope_params size");
            ਜੇ (!rope_ਵੇਖਿਆ) {
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(model_params.rope_params, ਗੰਢ->op_params, ਆਕਾਰ(int32_t) * 16);
                rope_ਵੇਖਿਆ = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (memcmp(model_params.rope_params, ਗੰਢ->op_params, ਆਕਾਰ(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = ਸੱਚ;
            }
        }
        ਜੇ (ਗੰਢ->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_ਮਾਪ = ਗੰਢ->src[0]->ne[0];
        }
        ਜੇ (ਗੰਢ->op == GGML_OP_SCALE && ਗੰਢ->view_src != ਖਾਲੀ_ਸੂਚਕ && is_recurrent_cache(ਗੰਢ->view_src)) {
            ਜੇ (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = ਗੰਢ->view_src->ne[1];
            } ਨਹੀਂਤਾਂ {
                GGML_ASSERT(model_params.n_rs_slots == ਗੰਢ->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(ਗੰਢ) / ਗੰਢ->view_src->ne[0];
            compute_params.cache_rs_reset_idx = ਗੰਢ->src[0]->view_offs / ਗੰਢ->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        ਜੇ (ਗੰਢ->op == GGML_OP_CPY && ਗੰਢ->view_src != ਖਾਲੀ_ਸੂਚਕ && is_recurrent_cache(ਗੰਢ->view_src) &&
            ਗੰਢ->src[1] != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->src[1]->op == GGML_OP_VIEW && ਗੰਢ->src[1]->view_src == ਗੰਢ->view_src) {
            ਅਚਲ ਤਾਰਕਿਕ is_conv = is_conv_state_writeback(ਗੰਢ);
            ਅਚਲ ਤਾਰਕਿਕ is_gdn = ਗੰਢ->src[0]->op == GGML_OP_VIEW && ਗੰਢ->src[0]->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
                                ਗੰਢ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ਅਚਲ ਤਾਰਕਿਕ is_extra = ਗੰਢ->src[0]->op == GGML_OP_GET_ਪੰਕਤੀਆਂ;
            ਅਚਲ ਤਾਰਕਿਕ is_gdn_rollback = is_gdn && is_same_shape(ਗੰਢ->src[0], ਗੰਢ->src[1]);

            ਅਚਲ ggml_tensor * dest_view = ਗੰਢ->src[1];
            ਅਚਲ ggml_tensor * cache = ਗੰਢ->view_src;
            ਅਚਲ size_t ਪੰਕਤੀ_ਬਾਈਟਾਂ = cache->ne[0] * ggml_type_ਮਾਪ(cache->type);
            ਜੇ (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = ਸੱਚ;
            }
            ਜੇ (ਪੰਕਤੀ_ਬਾਈਟਾਂ > 0 && (is_conv || is_gdn || is_extra) && !is_ਭਰਿਆ_single_ਥਾਂ_writeback(ਗੰਢ)) {
                ComputeParams::RsWriteback writeback;
                writeback.ਥਾਂ_begin = (ਪੂਰਨ) (dest_view->view_offs / ਪੰਕਤੀ_ਬਾਈਟਾਂ);
                ਜੇ (is_conv) {
                    writeback.src_begin = (ਪੂਰਨ) (ਗੰਢ->src[0]->view_offs / ਗੰਢ->src[0]->view_src->nb[0]);
                } ਨਹੀਂਤਾਂ ਜੇ (is_gdn_rollback) {
                    writeback.src_begin = (ਪੂਰਨ) (ਗੰਢ->src[0]->view_offs / ਗੰਢ->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, ਗੰਢ)] = writeback;
            }
            ਜੇ ((is_conv || is_gdn) && !is_ਭਰਿਆ_single_ਥਾਂ_writeback(ਗੰਢ)) {
                compute_params.s_copy_ਸਰਗਰਮ_ਥਾਂ_len = (ਪੂਰਨ) dest_view->ne[1];
            }
        }
    }
    ਜੇ (model_params.n_heads_kv == -1) {
        ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * ਗੰਢ = cgraph->nodes[i];
            ਅਚਲ ggml_tensor * ਨਕਾਬ = ਖਾਲੀ_ਸੂਚਕ;
            ਜੇ (ਗੰਢ->op == GGML_OP_SOFT_MAX) {
                ਨਕਾਬ = ਗੰਢ->src[1];
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->op == GGML_OP_FLASH_ATTN_EXT) {
                ਨਕਾਬ = ਗੰਢ->src[3];
            } ਨਹੀਂਤਾਂ {
                ਜਾਰੀ;
            }
            ਜੇ (ਨਕਾਬ == ਖਾਲੀ_ਸੂਚਕ || ਨਕਾਬ->op != GGML_OP_NONE || !(ਨਕਾਬ->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼) ||
                ਗੰਢ->src[0] == ਖਾਲੀ_ਸੂਚਕ) {
                ਜਾਰੀ;
            }
            model_params.is_cacheless_attn = ਸੱਚ;
            model_params.n_seq = 1;
            model_params.ctx_ਪ੍ਰਤੀ_seq = ਨਕਾਬ->ne[0];
            compute_params.ਨਿਵੇਸ਼_len = ਗੰਢ->src[0]->ne[1];
            compute_params.token_len_ਪ੍ਰਤੀ_seq = compute_params.ਨਿਵੇਸ਼_len;
            ਰੋਕੋ;
        }
    }

    ਸਵੈਚਾਲਿਤ * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    ਜੇ (model_params.is_cacheless_attn) {
        ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * ਗੰਢ = cgraph->nodes[i];
            ਜੇ (ਗੰਢ->op == GGML_OP_GET_ਪੰਕਤੀਆਂ && is_output_idx(ਗੰਢ->src[1], ਗੰਢ)) {
                compute_params.output_len = ਗੰਢ->src[1]->ne[0];
                ਰੋਕੋ;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    ਜੇ (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_ਪ੍ਰਤੀ_seq * model_params.n_seq;
    ਵਾਪਸ {model_params, compute_params};
}

ਖਾਲੀ GgmlOvDecoder::validate_cgraph() ਅਚਲ {
    ਜੇ (m_model_params.n_seq > 1 && m_is_static == ਸੱਚ) {
        ਸੁੱਟੋ std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_ਗ੍ਰਾਫ_ਨਿਵੇਸ਼_shape(ਅਚਲ ggml_tensor * op,
                                                      ਅਚਲ ggml_tensor * ਨਿਵੇਸ਼,
                                                      ਪੂਰਨ dynamic_dim_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਜੇ (m_naive) {
        ਵਾਪਸ ਨਿਵੇਸ਼ != ਖਾਲੀ_ਸੂਚਕ ? ov::PartialShape{get_shape(ਨਿਵੇਸ਼)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape ਨਿਵੇਸ਼_shape;

    ਜੇ (is_inp_tok(ਨਿਵੇਸ਼, op) || is_inp_pos(ਨਿਵੇਸ਼, op)) {
        // tokens or positions
        ਪੂਰਨ len = m_is_static ? (m_is_prefill ? m_prefill_chunk_ਮਾਪ : 1) : -1;
        ਜੇ (m_is_static && is_inp_pos(ਨਿਵੇਸ਼, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        ਨਿਵੇਸ਼_shape = ov::PartialShape{1, 1, 1, len};

    } ਨਹੀਂਤਾਂ ਜੇ (is_output_idx(ਨਿਵੇਸ਼, op)) {
        // output index
        ਨਿਵੇਸ਼_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } ਨਹੀਂਤਾਂ ਜੇ (is_inp_ਔਸਤ(ਨਿਵੇਸ਼, op)) {
        ਨਿਵੇਸ਼_shape = m_is_static ? ov::PartialShape{1, 1, ਨਿਵੇਸ਼->ne[1], m_prefill_chunk_ਮਾਪ} :
                                    ov::PartialShape{1, 1, -1, -1};

    } ਨਹੀਂਤਾਂ ਜੇ (is_inp_scale_ਪੰਕਤੀਆਂ(ਨਿਵੇਸ਼, op)) {
        ਨਿਵੇਸ਼_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_ਮਾਪ : 1) : -1, 1};

    } ਨਹੀਂਤਾਂ ਜੇ (is_inp_ਨਕਾਬ(ਨਿਵੇਸ਼, op)) {
        //ਨਕਾਬ
        ਜੇ (m_is_static) {
            ਨਿਵੇਸ਼_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_ਮਾਪ : 1, m_model_params.ctx};
        } ਨਹੀਂਤਾਂ ਜੇ (m_is_stateful) {
            ਨਿਵੇਸ਼_shape = ov::PartialShape{1, 1, -1, -1};
        } ਨਹੀਂਤਾਂ {
            ਨਿਵੇਸ਼_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } ਨਹੀਂਤਾਂ ਜੇ (is_recurrent_cache(ਨਿਵੇਸ਼)) {
        ਨਿਵੇਸ਼_shape = ov::PartialShape{get_shape(ਨਿਵੇਸ਼)};
        ਜੇ (!m_is_static && !m_is_stateful && ਨਿਵੇਸ਼->ne[1] > 1) {
            ਨਿਵੇਸ਼_shape[2] = -1;
        }

    } ਨਹੀਂਤਾਂ ਜੇ (is_kvcache(ਨਿਵੇਸ਼, op)) {
        // kvcache
        ਨਿਵੇਸ਼_shape = ov::PartialShape{get_shape(ਨਿਵੇਸ਼)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ਅਚਲ ਤਾਰਕਿਕ is_flat_kv = (ਨਿਵੇਸ਼->ne[1] == 1 && ਨਿਵੇਸ਼->ne[2] == 1 && ਨਿਵੇਸ਼->ne[3] == 1);
        ਜੇ (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            ਨਿਵੇਸ਼_shape[2] = -1;
        }
        ਜੇ (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            ਪ੍ਰਤਿਜ੍ਞਾ(ਨਿਵੇਸ਼_shape.size() == 4 && ਨਿਵੇਸ਼_shape[0] == 1 && ਨਿਵੇਸ਼_shape[1] == 1 &&
                   ਨਿਵੇਸ਼_shape[2].is_dynamic() && ਨਿਵੇਸ਼_shape[3].is_static());
            ਅਚਲ ਪੂਰਨ n_heads_kv = get_n_heads_kv_for_tensor(ਨਿਵੇਸ਼);
            ਪ੍ਰਤਿਜ੍ਞਾ(n_heads_kv > 0 && ਨਿਵੇਸ਼_shape[3].get_ਲੰਮਾਈ() % n_heads_kv == 0);
            ਅਚਲ int64_t combined_dim = ਨਿਵੇਸ਼_shape[3].get_ਲੰਮਾਈ();  // n_heads_kv * head_size
            ਅਚਲ int64_t ਸਿਰ_ਮਾਪ = combined_dim / n_heads_kv;
            ਨਿਵੇਸ਼_shape = {ਨਿਵੇਸ਼_shape[0], ov::Dimension::dynamic(), n_heads_kv, ਸਿਰ_ਮਾਪ};
        }

    } ਨਹੀਂਤਾਂ ਜੇ (is_kv_idx(ਨਿਵੇਸ਼, op)) {
        // kv update index
        ਪੂਰਨ len = m_is_static ? (m_is_prefill ? m_prefill_chunk_ਮਾਪ : 1) : -1;
        ਨਿਵੇਸ਼_shape = ov::PartialShape{1, 1, 1, len};

    } ਨਹੀਂਤਾਂ ਜੇ (is_inp_s_copy(ਨਿਵੇਸ਼, op) || is_s_copy_ਪੱਤੀ(ਨਿਵੇਸ਼)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        ਨਿਵੇਸ਼_shape = m_is_static ? ov::PartialShape{get_shape(ਨਿਵੇਸ਼)} : ov::PartialShape{1, 1, 1, -1};

    } ਨਹੀਂਤਾਂ {
        ਨਿਵੇਸ਼_shape = ov::PartialShape{get_shape(ਨਿਵੇਸ਼)};
    }
    ਜੇ (dynamic_dim_ਸੂਚਕਾਂਕ != -1 && m_model_is_splitted) {
        ਨਿਵੇਸ਼_shape[3 - dynamic_dim_ਸੂਚਕਾਂਕ] = -1;
    }
    ਜੇ (op->op == GGML_OP_SOFT_MAX && op->src[1] != ਖਾਲੀ_ਸੂਚਕ && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼ && op->src[1] == ਨਿਵੇਸ਼) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        ਜੇ (m_is_static) {
            ਅਚਲ int64_t seq_ਸਰਗਰਮ = m_is_prefill ? m_prefill_chunk_ਮਾਪ : 1;
            ਨਿਵੇਸ਼_shape[2] = seq_ਸਰਗਰਮ;
            ਨਿਵੇਸ਼_shape[3] = seq_ਸਰਗਰਮ;
        } ਨਹੀਂਤਾਂ {
            ਨਿਵੇਸ਼_shape[2] = -1;
            ਨਿਵੇਸ਼_shape[3] = -1;
        }
    }
    ਵਾਪਸ ਨਿਵੇਸ਼_shape;
}

ਤਾਰਕਿਕ GgmlOvDecoder::is_s_copy_ਪੱਤੀ(ਅਚਲ ggml_tensor * tensor) ਅਚਲ {
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ || tensor->op != GGML_OP_NONE || m_cgraph == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਲਈ (ਪੂਰਨ i = 0; i < m_cgraph->n_nodes; i++) {
        ਅਚਲ ggml_tensor * ਗੰਢ = m_cgraph->nodes[i];
        ਜੇ (ਗੰਢ->op != GGML_OP_GET_ਪੰਕਤੀਆਂ || ਗੰਢ->src[0] == ਖਾਲੀ_ਸੂਚਕ || ਗੰਢ->src[1] == ਖਾਲੀ_ਸੂਚਕ) {
            ਜਾਰੀ;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ਅਚਲ ggml_tensor * idx = ਗੰਢ->src[1];
        ਜਦੋਂ (idx != ਖਾਲੀ_ਸੂਚਕ && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        ਜੇ (idx != tensor) {
            ਜਾਰੀ;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ਅਚਲ ggml_tensor * ਅੰਕੜੇ = ਗੰਢ->src[0];
        ਜਦੋਂ (ਅੰਕੜੇ != ਖਾਲੀ_ਸੂਚਕ && (ਅੰਕੜੇ->op == GGML_OP_VIEW || ਅੰਕੜੇ->op == GGML_OP_RESHAPE)) {
            ਅੰਕੜੇ = ਅੰਕੜੇ->src[0];
        }
        ਜੇ (ਅੰਕੜੇ != ਖਾਲੀ_ਸੂਚਕ && is_recurrent_cache(ਅੰਕੜੇ)) {
            ਵਾਪਸ ਸੱਚ;
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

ਖਾਲੀ GgmlOvDecoder::ਜੋੜੋ_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    ਸਵੈਚਾਲਿਤ create_1d_ਨਿਵੇਸ਼ = [ਇਹ](ਅਚਲ std::string & name, int64_t value, ਤਾਰਕਿਕ force_parameter = ਝੂਠ) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    ਜੇ (m_compute_params.attention_ਮਾਪ != -1) {
        create_1d_ਨਿਵੇਸ਼("attention_size", m_compute_params.attention_ਮਾਪ);
    }
    ਜੇ (m_compute_params.attention_ਮਾਪ_static != -1) {
        create_1d_ਨਿਵੇਸ਼("attention_size_static", m_compute_params.attention_ਮਾਪ_static);
    }
    ਜੇ (m_compute_params.attention_ਮਾਪ_swa != -1) {
        create_1d_ਨਿਵੇਸ਼("attention_size_swa", m_compute_params.attention_ਮਾਪ_swa);
    }
    // only the stateful SWA mask consumes this
    ਜੇ (is_stateful() && m_compute_params.swa_ਖਿੜਕੀ != -1) {
        create_1d_ਨਿਵੇਸ਼("swa_window", m_compute_params.swa_ਖਿੜਕੀ);
    }
    create_1d_ਨਿਵੇਸ਼("n_seq_active", m_compute_params.n_seq_ਸਰਗਰਮ);
    create_1d_ਨਿਵੇਸ਼("seq_active_start", m_compute_params.seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ);
    create_1d_ਨਿਵੇਸ਼("seq_active_end", m_compute_params.seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ + m_compute_params.n_seq_ਸਰਗਰਮ);
    ਜੇ (m_compute_params.token_len_ਪ੍ਰਤੀ_seq != -1) {
        create_1d_ਨਿਵੇਸ਼("token_len_per_seq", m_compute_params.token_len_ਪ੍ਰਤੀ_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    ਜੇ (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_ਨਿਵੇਸ਼("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/ਸੱਚ);
        create_1d_ਨਿਵੇਸ਼("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/ਸੱਚ);
    }

    ਜੇ (m_compute_params.s_copy_ਸਰਗਰਮ_ਥਾਂ_len != -1) {
        create_1d_ਨਿਵੇਸ਼("s_copy_active_slot_len", m_compute_params.s_copy_ਸਰਗਰਮ_ਥਾਂ_len);
        ਜੇ (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_ਨਿਵੇਸ਼("chunk_valid_len", get_static_n_ਟੋਕਨ(), /*force_parameter=*/ਸੱਚ);
        }
    }

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & [ਗੰਢ_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_ਨਿਵੇਸ਼("rs_slot_begin_" + ਗੰਢ_name, writeback.ਥਾਂ_begin);
        ਜੇ (!m_is_static && writeback.src_begin >= 0) {
            create_1d_ਨਿਵੇਸ਼("rs_src_begin_" + ਗੰਢ_name, writeback.src_begin);
        }
    }
}

ਤਾਰਕਿਕ GgmlOvDecoder::ਗੰਢ_is_ਵਰਤਿਆ_as_src(ਅਚਲ ਪੂਰਨ ਗੰਢ_idx) {
    ggml_tensor * ਗੰਢ = m_cgraph->nodes[ਗੰਢ_idx];
    ਲਈ (ਪੂਰਨ i = ਗੰਢ_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_ਗੰਢ = m_cgraph->nodes[i];
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਜੇ (other_ਗੰਢ->src[j] == ਗੰਢ) {
                ਵਾਪਸ ਸੱਚ;
            }
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

ਖਾਲੀ GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    ਲਈ (ਪੂਰਨ i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ਗੰਢ = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        ਜੇ (ਗੰਢ->op == GGML_OP_NONE && ਗੰਢ_is_ਵਰਤਿਆ_as_src(i)) {
            std::string ਗੰਢ_name = get_tensor_ov_name(m_cgraph, ਗੰਢ);
            ਜੇ (m_model_weights.find(ਗੰਢ_name) == m_model_weights.end()) {
                m_inputs[ਗੰਢ_name] = ਗੰਢ;
                m_model_inputs[ਗੰਢ_name] = {get_ov_type(ਗੰਢ),
                                             get_ਗ੍ਰਾਫ_ਨਿਵੇਸ਼_shape(ਗੰਢ, ਖਾਲੀ_ਸੂਚਕ, m_ਗੰਢ_dynamic_dims[ਗੰਢ])};
            }
            ਜਾਰੀ;
        }
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਸਵੈਚਾਲਿਤ * src = ਗੰਢ->src[i];
            ਜੇ (src == ਖਾਲੀ_ਸੂਚਕ) {
                ਜਾਰੀ;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            ਜੇ (src->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼) {
                src_name = get_tensor_ਗ੍ਰਾਫ_ਨਿਵੇਸ਼_ov_name(ਇਹ, m_cgraph, src, ਗੰਢ);
            }
            ਜੇ (m_model_weights.find(src_name) != m_model_weights.end()) {
                ਜਾਰੀ;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            ਜੇ (src->view_src != ਖਾਲੀ_ਸੂਚਕ &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                ਜਾਰੀ;
            }

            ਤਾਰਕਿਕ is_intermediate_ਗੰਢ = ਝੂਠ;
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਗੰਢ_info : m_ਗੰਢ_info_list) {
                ਜੇ (ਗੰਢ_info.ਗੰਢ == src) {
                    is_intermediate_ਗੰਢ = ਸੱਚ;
                    ਰੋਕੋ;
                }
            }
            ਜੇ (is_intermediate_ਗੰਢ) {
                ਜਾਰੀ;
            }
            ਜੇ (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                ਜਾਰੀ;
            }

            m_inputs[src_name] = src;

            ggml_backend_ਬਫਰ * ਬਫਰ = src->ਬਫਰ;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            ਜੇ (ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_ANY) {
                ਜੇ (ਸਵੈਚਾਲਿਤ it = std::find(m_model_params.kv_ਨਾਂ.begin(), m_model_params.kv_ਨਾਂ.end(), src_name);
                    it == m_model_params.kv_ਨਾਂ.end()) {
                    m_model_params.kv_ਨਾਂ.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            ਜਦੋਂ (src->op == GGML_OP_VIEW && src->src[0] != ਖਾਲੀ_ਸੂਚਕ) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_ਗ੍ਰਾਫ_ਨਿਵੇਸ਼_shape(ਗੰਢ, src, m_ਗੰਢ_dynamic_dims[src])};
        }
    }
}

ਖਾਲੀ GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_ਨਾਂ.clear();
    ਲਈ (ਪੂਰਨ ਗੰਢ_n = 0; ਗੰਢ_n < m_cgraph->n_nodes; ਗੰਢ_n++) {
        ਸਵੈਚਾਲਿਤ * cur_ਗੰਢ = m_cgraph->nodes[ਗੰਢ_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        ਜੇ (cur_ਗੰਢ->op == GGML_OP_NONE || cur_ਗੰਢ->op == GGML_OP_VIEW || cur_ਗੰਢ->op == GGML_OP_RESHAPE) {
            ਜਾਰੀ;
        }
        ਜੇ (::is_inplace_op(cur_ਗੰਢ) && ggml_nbytes(cur_ਗੰਢ) == 0) {
            ਜਾਰੀ;
        }
        ਸਵੈਚਾਲਿਤ cur_ਗੰਢ_use_ਗਿਣਤੀ = m_cgraph->use_counts[ggml_hash_ਲੱਭੋ(&m_cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ, cur_ਗੰਢ)];
        ਜੇ (cur_ਗੰਢ_use_ਗਿਣਤੀ == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            ਜੇ (cur_ਗੰਢ != ਖਾਲੀ_ਸੂਚਕ && ::is_inplace_op(cur_ਗੰਢ) && ggml_nbytes(cur_ਗੰਢ) > 0) {
                cur_ਗੰਢ = cur_ਗੰਢ->view_src;
            }
        } ਨਹੀਂਤਾਂ {
            ਪੂਰਨ ਨਿਵੇਸ਼_use_ਗਿਣਤੀ = 0;
            ਲਈ (ਪੂਰਨ i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * ਗੰਢ = m_cgraph->nodes[i];
                ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
                    ਜੇ (ਗੰਢ->src[j] != ਨਿਰਾ && ਗੰਢ->src[j] == cur_ਗੰਢ) {
                        ਨਿਵੇਸ਼_use_ਗਿਣਤੀ++;
                    }
                }
            }
            ਜੇ (ਨਿਵੇਸ਼_use_ਗਿਣਤੀ == cur_ਗੰਢ_use_ਗਿਣਤੀ) {
                cur_ਗੰਢ = ਖਾਲੀ_ਸੂਚਕ;
            }
        }
        ਜੇ (cur_ਗੰਢ != ਖਾਲੀ_ਸੂਚਕ) {
            std::string cur_ਗੰਢ_name = get_tensor_ov_name(m_cgraph, cur_ਗੰਢ);
            m_model_outputs[cur_ਗੰਢ_name] = cur_ਗੰਢ;
            m_model_output_ਨਾਂ.insert(cur_ਗੰਢ_name);
        }
    }
}

ਅਚਲ ggml_tensor * GgmlOvDecoder::get_tensor_ਵਰਤਿਆ_op(ਅਚਲ ggml_tensor * tensor) ਅਚਲ {
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਲਈ (ਪੂਰਨ i = 0; i < m_cgraph->n_nodes; i++) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * ਗੰਢ = m_cgraph->nodes[i];
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਜੇ (ਗੰਢ->src[j] == tensor) {
                ਵਾਪਸ ਗੰਢ;
            }
        }
    }
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

ਅਚਲ ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ਅਚਲ std::string & name) ਅਚਲ {
    ਲਈ (ਪੂਰਨ i = 0; i < m_cgraph->n_nodes; i++) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * ਗੰਢ = m_cgraph->nodes[i];
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * src = ਗੰਢ->src[j];
            ਜੇ (src == ਖਾਲੀ_ਸੂਚਕ) {
                ਰੋਕੋ;
            }
            ਜੇ (get_tensor_ov_name(m_cgraph, src) == name) {
                ਵਾਪਸ src;
            }
        }
    }
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_ਨਾਂ() ਅਚਲ {
    std::map<std::string, std::string> kv_param_res_ਨਾਂ;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & name : m_model_params.kv_ਨਾਂ) {
        kv_param_res_ਨਾਂ[name] = name;
    }
    ਵਾਪਸ kv_param_res_ਨਾਂ;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
ਸਥਿਰ ਤਾਰਕਿਕ is_mul_mat_id_expert_ਭਾਰ(ਅਚਲ ggml_tensor * ਗੰਢ, ਪੂਰਨ src_ਸੂਚਕਾਂਕ) {
    ਵਾਪਸ ਗੰਢ->op == GGML_OP_MUL_MAT_ID && src_ਸੂਚਕਾਂਕ == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_ਭਾਰ_nodes(ggml_cgraph * cgraph, ਤਾਰਕਿਕ naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    ਸਵੈਚਾਲਿਤ * nodes = cgraph->nodes;
    ਸਵੈਚਾਲਿਤ n_nodes = cgraph->n_nodes;
    ਲਈ (ਪੂਰਨ ਗੰਢ_i = 0; ਗੰਢ_i < n_nodes; ਗੰਢ_i++) {
        ਸਵੈਚਾਲਿਤ * ਗੰਢ = nodes[ਗੰਢ_i];
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਸਵੈਚਾਲਿਤ * src = ਗੰਢ->src[i];
            ਜੇ (src == ਖਾਲੀ_ਸੂਚਕ) {
                ਜਾਰੀ;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * ਅਧਾਰ = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, ਅਧਾਰ);
            ਜੇ (is_rope_freqs_ਭਾਰ(ਅਧਾਰ, ਗੰਢ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_ਬਫਰ * ਬਫਰ = ਅਧਾਰ->ਬਫਰ;
            ਜੇ (ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS || ggml_is_quantized(ਅਧਾਰ->type) ||
                is_mul_mat_id_expert_ਭਾਰ(ਗੰਢ, i)) {
                ਜੇ (model_weights.find(src_name) == model_weights.end()) {
                    ਸਵੈਚਾਲਿਤ ਭਾਰ_ਗੰਢ = create_ਭਾਰ_ਗੰਢ(ਅਧਾਰ, naive);
                    ਭਾਰ_ਗੰਢ->ਸੈੱਟ_friendly_name(src_name);
                    model_weights[src_name] = ਭਾਰ_ਗੰਢ;
                }
            }
        }
    }
    ਵਾਪਸ model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
ਸਥਿਰ std::mutex g_nonov_ਭਾਰ_cache_mutex;
ਸਥਿਰ std::unordered_map<ਅਚਲ ਖਾਲੀ *, std::shared_ptr<ov::Node>> g_nonov_ਭਾਰ_cache;

std::set<std::string> GgmlOvDecoder::collect_ਭਾਰ_ਨਾਂ(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> ਨਾਂ;
    ਲਈ (ਪੂਰਨ ਗੰਢ_i = 0; ਗੰਢ_i < cgraph->n_nodes; ਗੰਢ_i++) {
        ਸਵੈਚਾਲਿਤ * ਗੰਢ = cgraph->nodes[ਗੰਢ_i];
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਸਵੈਚਾਲਿਤ * src = ਗੰਢ->src[i];
            ਜੇ (src == ਖਾਲੀ_ਸੂਚਕ) {
                ਜਾਰੀ;
            }
            ਅਚਲ ggml_tensor * ਅਧਾਰ = src->view_src ? src->view_src : src;
            std::string src_name(ਅਧਾਰ->name);
            ਜੇ (is_rope_freqs_ਭਾਰ(ਅਧਾਰ, ਗੰਢ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_ਬਫਰ * ਬਫਰ = ਅਧਾਰ->ਬਫਰ;
            ਜੇ (ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS || ggml_is_quantized(ਅਧਾਰ->type)) {
                ਨਾਂ.insert(src_name);
            }
        }
    }
    ਵਾਪਸ ਨਾਂ;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_ਭਾਰ_ਗੰਢ(ggml_tensor * tensor, ਤਾਰਕਿਕ naive) {
    ਅਚਲ ਤਾਰਕਿਕ is_ov_ਬਫਰ = ggml_backend_ਬਫਰ_is_openvino(tensor->ਬਫਰ);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    ਜੇ (tensor->extra) {
        OPENVINO_ASSERT(is_ov_ਬਫਰ, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        ਸਵੈਚਾਲਿਤ * extra_ਅਧਾਰ = static_cast<ggml_openvino_extra_ਅਧਾਰ *>(tensor->extra);

        ਜੇ (extra_ਅਧਾਰ->type == ggml_openvino_extra_ਅਧਾਰ::Type::ਭਾਰ_2) {
            // F16/F32/BF16 weight with shared-memory constant
            ਸਵੈਚਾਲਿਤ * ਭਾਰ_extra = static_cast<ggml_openvino_ਭਾਰ_extra *>(tensor->extra);
            ਜੇ (ਭਾਰ_extra->ਭਾਰ_ਗੰਢ) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                ਵਾਪਸ ਭਾਰ_extra->ਭਾਰ_ਗੰਢ;
            }
        } ਨਹੀਂਤਾਂ ਜੇ (extra_ਅਧਾਰ->type == ggml_openvino_extra_ਅਧਾਰ::Type::QUANTIZED_ਭਾਰ) {
            // Quantized weight with pre-extracted data
            ਸਵੈਚਾਲਿਤ * quant_extra = static_cast<ggml_openvino_quantized_ਭਾਰ_extra *>(tensor->extra);
            ਜੇ (quant_extra->ਭਾਰ_ਗੰਢ) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                ਵਾਪਸ quant_extra->ਭਾਰ_ਗੰਢ;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    ਜੇ (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        ਸਵੈਚਾਲਿਤ ਭਾਰ_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->ਅੰਕੜੇ);
        ਸਵੈਚਾਲਿਤ ਭਾਰ_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(ਭਾਰ_tensor);
        ਭਾਰ_ਗੰਢ->ਸੈੱਟ_friendly_name(tensor->name);
        ਵਾਪਸ ਭਾਰ_ਗੰਢ;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ਅਚਲ ਤਾਰਕਿਕ cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_ਬਫਰ &&
                                 !naive && tensor->ਅੰਕੜੇ != ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (cacheable_nonov) {
        std::lock_guard<std::mutex> ਤਾਲਾ(g_nonov_ਭਾਰ_cache_mutex);
        ਸਵੈਚਾਲਿਤ it = g_nonov_ਭਾਰ_cache.find(tensor->ਅੰਕੜੇ);
        ਜੇ (it != g_nonov_ਭਾਰ_cache.end()) {
            ਵਾਪਸ it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    ਸਥਿਰ ਅਚਲ std::set<ggml_type> ਭਾਰ_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    ਜੇ (ਭਾਰ_types.find(tensor->type) == ਭਾਰ_types.end()) {
        ਸੁੱਟੋ std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    Ov_ਭਾਰ ov_ਭਾਰ;
    ਜੇ (ggml_is_quantized(tensor->type)) {
        ਸਵੈਚਾਲਿਤ use_bias = naive;
        ਜੇ (is_ov_ਬਫਰ) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_ਮਾਪ = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_ਮਾਪ);
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp.ਅੰਕੜੇ(), tensor->ਅੰਕੜੇ, raw_ਮਾਪ);
            ov_ਭਾਰ = process_ਭਾਰ_tensor(tensor, tmp.ਅੰਕੜੇ(), tensor->ਅੰਕੜੇ, use_bias);
        } ਨਹੀਂਤਾਂ {
            ov_ਭਾਰ = process_ਭਾਰ_tensor(tensor, tensor->ਅੰਕੜੇ, ਖਾਲੀ_ਸੂਚਕ, use_bias);
        }
    } ਨਹੀਂਤਾਂ {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_ਭਾਰ = process_ਭਾਰ_tensor(tensor, tensor->ਅੰਕੜੇ, tensor->ਅੰਕੜੇ);
    }

    ov_ਭਾਰ.ਭਾਰ_ਗੰਢ->ਸੈੱਟ_friendly_name(tensor->name);
    ਜੇ (!is_ov_ਬਫਰ) {
        ਜੇ (cacheable_nonov) {
            std::lock_guard<std::mutex> ਤਾਲਾ(g_nonov_ਭਾਰ_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            ਸਵੈਚਾਲਿਤ [it, inserted] = g_nonov_ਭਾਰ_cache.emplace(tensor->ਅੰਕੜੇ, ov_ਭਾਰ.ਭਾਰ_ਗੰਢ);
            ਵਾਪਸ it->second;
        }
        ਵਾਪਸ ov_ਭਾਰ.ਭਾਰ_ਗੰਢ;
    }

    ggml_openvino_extra_ਅਧਾਰ * extra;
    ਜੇ (ov_ਭਾਰ.is_quantized()) {
        extra = ਨਵਾਂ ggml_openvino_quantized_ਭਾਰ_extra(std::move(ov_ਭਾਰ.weights), std::move(ov_ਭਾਰ.scales),
                                                         std::move(ov_ਭਾਰ.zp), ov_ਭਾਰ.ਭਾਰ_ਗੰਢ);
    } ਨਹੀਂਤਾਂ {
        extra = ਨਵਾਂ ggml_openvino_ਭਾਰ_extra(std::move(ov_ਭਾਰ.weights), ov_ਭਾਰ.ਭਾਰ_ਗੰਢ);
    }
    ggml_openvino_ਬਫਰ_ਰਜਿਸਟਰ_extra(tensor, extra);

    ਵਾਪਸ ov_ਭਾਰ.ਭਾਰ_ਗੰਢ;
}

ਖਾਲੀ GgmlOvDecoder::dump_cgraph(ਅਚਲ ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    ਜੇ (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        ਵਾਪਸ;
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
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ਗੰਢ = cgraph->nodes[i];

        // Get buffer type name
        ਅਚਲ ਅੱਖਰ * buf_name = "none";
        ggml_backend_ਬਫਰ_t buf = ਗੰਢ->view_src ? ਗੰਢ->view_src->ਬਫਰ : ਗੰਢ->ਬਫਰ;
        ਜੇ (buf) {
            buf_name = ggml_backend_ਬਫਰ_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ਗੰਢ->ne[0] << ", "
             << std::setw(5) << ਗੰਢ->ne[1] << ", "
             << std::setw(5) << ਗੰਢ->ne[2] << ", "
             << std::setw(5) << ਗੰਢ->ne[3] << "] "
             << std::ਖੱਬੇ << std::setw(20) << ggml_op_name(ਗੰਢ->op) << std::ਸੱਜੇ << " "
             << std::ਖੱਬੇ << std::setw(45) << ਗੰਢ->name << std::ਸੱਜੇ
             << std::setw(2) << "[ "
             << std::setw(0) << ਗੰਢ->nb[0] << ", "
             << std::setw(5) << ਗੰਢ->nb[1] << ", "
             << std::setw(5) << ਗੰਢ->nb[2] << ", "
             << std::setw(5) << ਗੰਢ->nb[3] << "] "
             << std::ਸੱਜੇ << std::setw(15) << buf_name << std::ਸੱਜੇ
             << "\n";

        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਜੇ (ਸਵੈਚਾਲਿਤ* src = ਗੰਢ->src[i]) {
                // Get buffer type name for source
                ਅਚਲ ਅੱਖਰ * src_buf_name = "none";
                ggml_backend_ਬਫਰ_t src_buf = src->view_src ? src->view_src->ਬਫਰ : src->ਬਫਰ;
                ਜੇ (src_buf) {
                    src_buf_name = ggml_backend_ਬਫਰ_name(src_buf);
                }

                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::ਖੱਬੇ << std::setw(12) << ggml_op_name(src->op) << std::ਸੱਜੇ;
                file << std::ਖੱਬੇ << std::setw(30) << src->name << std::ਸੱਜੇ
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << std::ਸੱਜੇ << std::setw(15) << src_buf_name << std::ਸੱਜੇ
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * ਗੰਢ = cgraph->leafs[i];

        // Get buffer type name for leaf
        ਅਚਲ ਅੱਖਰ * ਪੱਤੀ_buf_name = "none";
        ggml_backend_ਬਫਰ_t ਪੱਤੀ_buf = ਗੰਢ->view_src ? ਗੰਢ->view_src->ਬਫਰ : ਗੰਢ->ਬਫਰ;
        ਜੇ (ਪੱਤੀ_buf) {
            ਪੱਤੀ_buf_name = ggml_backend_ਬਫਰ_name(ਪੱਤੀ_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ਗੰਢ->ne[0] << ", "
             << std::setw(5) << ਗੰਢ->ne[1] << "] "
             << std::setw(8) << ggml_op_name(ਗੰਢ->op) << " "
             << std::setw(16) << ggml_get_name(ਗੰਢ)
             << std::setw(20) << ਪੱਤੀ_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

ਖਾਲੀ print_tensor_ਪਤਾ_map(ਅਚਲ ggml_cgraph * cgraph) {
    std::map<ਖਾਲੀ *, std::vector<std::string>> ਪਤਾ_map;
    ਲਈ (ਪੂਰਨ ਗੰਢ_n = 0; ਗੰਢ_n < cgraph->n_nodes; ਗੰਢ_n++) {
        ਸਵੈਚਾਲਿਤ * ਗੰਢ = cgraph->nodes[ਗੰਢ_n];
        ਜੇ (ਗੰਢ->ਅੰਕੜੇ) {
            ਸਵੈਚਾਲਿਤ it = ਪਤਾ_map.find(ਗੰਢ->ਅੰਕੜੇ);
            ਜੇ (it == ਪਤਾ_map.end()) {
                ਪਤਾ_map[ਗੰਢ->ਅੰਕੜੇ] = std::vector<std::string>();
            }
            ਪਤਾ_map[ਗੰਢ->ਅੰਕੜੇ].push_back(ਗੰਢ->name);
        }
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & pair : ਪਤਾ_map) {
        std::ਬਾਹਰ << "Address: " << pair.ਪਹਿਲਾ << '\n';
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & name : pair.second) {
            std::ਬਾਹਰ << name << " ; ";
        }
        std::ਬਾਹਰ << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ਅਚਲ ggml_tensor * tensor) {
    std::vector<size_t> shape;
    ਲਈ (ਪੂਰਨ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    ਵਾਪਸ shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(ਅਚਲ ggml_tensor * tensor) {
    std::vector<size_t> stride;
    ਲਈ (ਪੂਰਨ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    ਵਾਪਸ stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(ਅਚਲ ggml_tensor * tensor) {
    ਚੋਣ (tensor->type) {
    ਹਾਲਤ GGML_TYPE_F64:
        ਵਾਪਸ ov::element::f64;
    ਹਾਲਤ GGML_TYPE_F32:
        ਵਾਪਸ ov::element::f32;
    ਹਾਲਤ GGML_TYPE_F16:
        ਵਾਪਸ ov::element::f16;
    ਹਾਲਤ GGML_TYPE_BF16:
        ਵਾਪਸ ov::element::bf16;
    ਹਾਲਤ GGML_TYPE_I8:
        ਵਾਪਸ ov::element::i8;
    ਹਾਲਤ GGML_TYPE_I16:
        ਵਾਪਸ ov::element::i16;
    ਹਾਲਤ GGML_TYPE_I32:
        ਵਾਪਸ ov::element::i32;
    ਹਾਲਤ GGML_TYPE_I64:
        ਵਾਪਸ ov::element::i64;
    ਮੂਲ:
        ਵਾਪਸ ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_ਨਿਵੇਸ਼_shape(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ {
    ਵਾਪਸ ov::PartialShape(get_shape(m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_ਨਿਵੇਸ਼_stride(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ {
    ਵਾਪਸ get_stride(m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_ਮਾਪ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਵਾਪਸ it->second.size();
    }
    ਵਾਪਸ 0;
}

size_t GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਜੇ (view_ਸੂਚਕਾਂਕ < it->second.size()) {
            ਵਾਪਸ it->second[view_ਸੂਚਕਾਂਕ].second->view_offs;
        }
    }
    ਵਾਪਸ 0;
}

size_t GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_src_ਵਿੱਥ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਜੇ (view_ਸੂਚਕਾਂਕ < it->second.size()) {
            ਸਵੈਚਾਲਿਤ * view_tensor = it->second[view_ਸੂਚਕਾਂਕ].second;
            ਜੇ (view_tensor && view_tensor->src[0]) {
                ਵਾਪਸ view_tensor->src[0]->view_offs;
            }
        }
    }
    ਵਾਪਸ 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_stride(ਪੂਰਨ ਗੰਢ_idx,
                                                         ਅਚਲ std::string & name,
                                                         size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਜੇ (view_ਸੂਚਕਾਂਕ < it->second.size()) {
            ਵਾਪਸ get_stride(it->second[view_ਸੂਚਕਾਂਕ].second);
        }
    }
    ਵਾਪਸ {};
}

std::vector<size_t> GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_src_stride(ਪੂਰਨ ਗੰਢ_idx,
                                                             ਅਚਲ std::string & name,
                                                             size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਜੇ (view_ਸੂਚਕਾਂਕ < it->second.size()) {
            ਸਵੈਚਾਲਿਤ * view_tensor = it->second[view_ਸੂਚਕਾਂਕ].second;
            ਜੇ (view_tensor && view_tensor->src[0]) {
                ਵਾਪਸ get_stride(view_tensor->src[0]);
            }
        }
    }
    ਵਾਪਸ {};
}

ov::Shape GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_ggml_shape(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਜੇ (view_ਸੂਚਕਾਂਕ < it->second.size()) {
            ਵਾਪਸ get_shape(it->second[view_ਸੂਚਕਾਂਕ].second);
        }
    }
    ਵਾਪਸ {};
}

ov::Shape GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_src_ggml_shape(ਪੂਰਨ ਗੰਢ_idx,
                                                       ਅਚਲ std::string & name,
                                                       size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਜੇ (view_ਸੂਚਕਾਂਕ < it->second.size()) {
            ਸਵੈਚਾਲਿਤ * view_tensor = it->second[view_ਸੂਚਕਾਂਕ].second;
            ਜੇ (view_tensor && view_tensor->src[0]) {
                ਵਾਪਸ get_shape(view_tensor->src[0]);
            }
        }
    }
    ਵਾਪਸ {};
}

ov::PartialShape GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_ov_shape(ਪੂਰਨ ਗੰਢ_idx,
                                                        ਅਚਲ std::string & name,
                                                        size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਜੇ (view_ਸੂਚਕਾਂਕ < it->second.size()) {
            ਸਵੈਚਾਲਿਤ * tensor = it->second[view_ਸੂਚਕਾਂਕ].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            ਸਵੈਚਾਲਿਤ dynamic_it = m_ਗੰਢ_dynamic_dims.find(tensor);
            ਜੇ (dynamic_it != m_ਗੰਢ_dynamic_dims.end() && dynamic_it->second != -1) {
                ਪੂਰਨ dynamic_dim_ਸੂਚਕਾਂਕ = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_ਸੂਚਕਾਂਕ] = m_is_static ? get_static_n_ਟੋਕਨ() : -1;
            }

            ਵਾਪਸ shape;
        }
    }
    ਵਾਪਸ {};
}

ov::PartialShape GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_src_ov_shape(ਪੂਰਨ ਗੰਢ_idx,
                                                            ਅਚਲ std::string & name,
                                                            size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਜੇ (view_ਸੂਚਕਾਂਕ < it->second.size()) {
            ਸਵੈਚਾਲਿਤ * view_tensor = it->second[view_ਸੂਚਕਾਂਕ].second;
            ਜੇ (view_tensor && view_tensor->src[0]) {
                ਸਵੈਚਾਲਿਤ * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                ਸਵੈਚਾਲਿਤ dynamic_it = m_ਗੰਢ_dynamic_dims.find(src_tensor);
                ਜੇ (dynamic_it != m_ਗੰਢ_dynamic_dims.end() && dynamic_it->second != -1) {
                    ਪੂਰਨ dynamic_dim_ਸੂਚਕਾਂਕ = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_ਸੂਚਕਾਂਕ] = m_is_static ? get_static_n_ਟੋਕਨ() : -1;
                }

                ਵਾਪਸ shape;
            }
        }
    }
    ਵਾਪਸ {};
}

std::string GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_name(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਜੇ (view_ਸੂਚਕਾਂਕ < it->second.size()) {
            ਵਾਪਸ it->second[view_ਸੂਚਕਾਂਕ].ਪਹਿਲਾ;
        }
    }
    ਵਾਪਸ "";
}

std::string GgmlOvDecoder::get_view_ਨਿਵੇਸ਼_src_name(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.find(name);
    ਜੇ (it != m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_views.end()) {
        ਜੇ (view_ਸੂਚਕਾਂਕ < it->second.size()) {
            ਸਵੈਚਾਲਿਤ * view_tensor = it->second[view_ਸੂਚਕਾਂਕ].second;
            ਜੇ (view_tensor && view_tensor->src[0]) {
                ਵਾਪਸ get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    ਵਾਪਸ "";
}

ov::element::Type GgmlOvDecoder::get_ਨਿਵੇਸ਼_type(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ {
    ਵਾਪਸ get_ov_type(m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs.at(name));
}

size_t GgmlOvDecoder::get_ਨਿਵੇਸ਼_ਮਾਪ() ਅਚਲ {
    ਵਾਪਸ m_model_inputs.size();
}

size_t GgmlOvDecoder::get_ਨਿਵੇਸ਼_ਮਾਪ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਵਾਪਸ m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_ਨਾਂ.size();
}

std::vector<std::string> GgmlOvDecoder::get_ਨਿਵੇਸ਼_ਨਾਂ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਵਾਪਸ m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs_ਨਾਂ;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ * ggml_tensor = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ;
    ਵਾਪਸ ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ਅਚਲ ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਵਾਪਸ get_ov_type(m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ * ggml_tensor = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ;
    ਵਾਪਸ get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_ਨਾਂ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ * ਗੰਢ = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ;
    ਜੇ (ਗੰਢ->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        ਲਈ (ਪੂਰਨ i = ਗੰਢ_idx + 1; i < m_cgraph->n_nodes; i++) {
            ਸਵੈਚਾਲਿਤ * ਖਪਤਕਾਰ = m_cgraph->nodes[i];
            ਜੇ (ਖਪਤਕਾਰ->op != GGML_OP_VIEW || ਖਪਤਕਾਰ->src[0] != ਗੰਢ) {
                ਜਾਰੀ;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            ਸਵੈਚਾਲਿਤ & name = ਖਪਤਕਾਰ->view_offs == 0 ? attn_name : state_name;
            ਜੇ (!name.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਵਾਪਸ {m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_name};
            }
            name = get_tensor_ov_name(m_cgraph, ਖਪਤਕਾਰ);
        }
        ਜੇ (!attn_name.ਖਾਲੀ_ਥਾਵਾਂ() && !state_name.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਵਾਪਸ {attn_name, state_name};
        }
    }
    ਵਾਪਸ {m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ * ਗੰਢ = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ;
    ਜੇ (!::is_inplace_op(ਗੰਢ) || ਗੰਢ->view_src == ਖਾਲੀ_ਸੂਚਕ || ggml_nbytes(ਗੰਢ) == 0) {
        ਵਾਪਸ "";
    }
    ਅਚਲ ਪੂਰਨ op_case = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_op_case;
    ਜੇ (ਗੰਢ->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_ਸਰਗਰਮ_ਥਾਂ_len == -1) {
        ਵਾਪਸ "";
    }
    ਵਾਪਸ get_tensor_ov_name(m_cgraph, ਗੰਢ->view_src);
}

ਤਾਰਕਿਕ GgmlOvDecoder::is_view_like_alias_ਦਾ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & view_src_name) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ * ਗੰਢ = m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ;
    ਜੇ (ਗੰਢ->view_src == ਖਾਲੀ_ਸੂਚਕ || get_tensor_ov_name(m_cgraph, ਗੰਢ->view_src) != view_src_name) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਗੰਢ->op == GGML_OP_RESHAPE || ਗੰਢ->op == GGML_OP_VIEW;
}

ਅਚਲ std::string & GgmlOvDecoder::get_op_name() ਅਚਲ {
    ਸਥਿਰ ਅਚਲ std::string unknown_name = "UNKNOWN_OP_NAME";
    ਵਾਪਸ unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਸਵੈਚਾਲਿਤ it = m_ਗੰਢ_dynamic_dims.find(m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ);
    ਜੇ (it == m_ਗੰਢ_dynamic_dims.end()) {
        ਵਾਪਸ -1;
    }
    ਵਾਪਸ it->second;
}

ਅਚਲ std::string & GgmlOvDecoder::get_op_name(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਵਾਪਸ m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_name;
}

int32_t * GgmlOvDecoder::get_ਨਿਵੇਸ਼_op_params(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ {
    ਵਾਪਸ m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਵਾਪਸ m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ->op_params;
}

size_t GgmlOvDecoder::get_output_op_ਵਿੱਥ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਵਾਪਸ m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ->view_offs;
}

ਖਾਲੀ GgmlOvDecoder::ਦੇਖੋ_subgraph(std::function<ਖਾਲੀ(std::shared_ptr<GgmlDecoder>, ਪੂਰਨ ਗੰਢ_idx)> ਗੰਢ_visitor) ਅਚਲ {
    ਲਈ (ਪੂਰਨ ਗੰਢ_idx = 0; ਗੰਢ_idx < m_cgraph->n_nodes; ਗੰਢ_idx++) {
        ਜੇ (m_cgraph->nodes[ਗੰਢ_idx]->op == GGML_OP_NONE) {
            ਜਾਰੀ;
        }
        ਗੰਢ_visitor(std::make_shared<GgmlOvDecoder>(*ਇਹ), ਗੰਢ_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ਅਚਲ ggml_tensor * ਗੰਢ) {
    ਚੋਣ (ਗੰਢ->op) {
    ਹਾਲਤ GGML_OP_UNARY:
        ਵਾਪਸ std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(ਗੰਢ));
    ਹਾਲਤ GGML_OP_GLU:
        ਵਾਪਸ std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(ਗੰਢ));
    ਮੂਲ:
        ਵਾਪਸ std::string("GGML_OP_") + ggml_op_name(ਗੰਢ->op);
    }
}

ਅਚਲ std::string & GgmlOvDecoder::get_op_type(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ {
    ਵਾਪਸ m_ਗੰਢ_info_list[ਗੰਢ_idx].ਗੰਢ_op_type;
}

ਅਚਲ std::string & GgmlOvDecoder::get_op_type() ਅਚਲ {
    ਸਥਿਰ ਅਚਲ std::string unknown_op = "UNKNOWN_GGML_OP";
    ਵਾਪਸ unknown_op;
}

ਖਾਲੀ GgmlOvDecoder::compute_ਗੰਢ_dynamic_dims() {
    ਸਵੈਚਾਲਿਤ ਦੇਖੋ_ਗੰਢ = [&](ਸਵੈਚਾਲਿਤ && self, ggml_tensor * ਗੰਢ) -> ਖਾਲੀ {
        ਜੇ (!ਗੰਢ) {
            ਵਾਪਸ;
        }

        ਜੇ (ਗੰਢ->op == GGML_OP_CPY) {
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
        }

        ਜੇ (m_ਗੰਢ_dynamic_dims.count(ਗੰਢ)) {
            ਵਾਪਸ;
        }
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = ਗੰਢ->src[i];
            ਜੇ (src == ਖਾਲੀ_ਸੂਚਕ) {
                ਜਾਰੀ;
            }
            ਬਣਤਰ ggml_tensor * ਜੜ੍ਹ_src = ਖਾਲੀ_ਸੂਚਕ;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            ਜੇ (ਜੜ੍ਹ_src) {
                ਜੇ (is_inp_tok(ਜੜ੍ਹ_src, ਗੰਢ) || is_inp_pos(ਜੜ੍ਹ_src, ਗੰਢ) || is_output_idx(ਜੜ੍ਹ_src, ਗੰਢ)) {
                    m_ਗੰਢ_dynamic_dims[ਜੜ੍ਹ_src] = 0;
                    m_ਗੰਢ_dynamic_dims[src] = m_ਗੰਢ_dynamic_dims[ਜੜ੍ਹ_src];
                    ਜਾਰੀ;
                }
                self(self, ਜੜ੍ਹ_src);
                m_ਗੰਢ_dynamic_dims[src] = m_ਗੰਢ_dynamic_dims[ਜੜ੍ਹ_src];
            } ਨਹੀਂਤਾਂ {
                ਜੇ (is_inp_tok(src, ਗੰਢ) || is_inp_pos(src, ਗੰਢ) || is_output_idx(src, ਗੰਢ)) {
                    m_ਗੰਢ_dynamic_dims[src] = 0;
                    ਜਾਰੀ;
                }
                ਜੇ (is_inp_scale_ਪੰਕਤੀਆਂ(src, ਗੰਢ)) {
                    m_ਗੰਢ_dynamic_dims[src] = 1;
                    ਜਾਰੀ;
                }
                ਜੇ (ਗੰਢ->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_ਗੰਢ_dynamic_dims[src] = 1;
                    ਜਾਰੀ;
                }
                self(self, src);
            }
        }
        ਚੋਣ (ਗੰਢ->op) {
        ਹਾਲਤ GGML_OP_NONE:
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[1]] != -1) {
                ਸਵੈਚਾਲਿਤ dynamic_dim_idx = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[1]];
                ਜੇ (dynamic_dim_idx == 0) {
                    m_ਗੰਢ_dynamic_dims[ਗੰਢ] = 1;
                } ਨਹੀਂਤਾਂ {
                    ਸਵੈਚਾਲਿਤ dynamic_dim_stride = ਗੰਢ->src[1]->nb[dynamic_dim_idx] / ggml_type_ਮਾਪ(ਗੰਢ->src[1]->type) *
                                              ggml_type_ਮਾਪ(ਗੰਢ->src[0]->type);
                    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
                        ਜੇ (dynamic_dim_stride == ਗੰਢ->src[0]->nb[i]) {
                            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = i;
                            ਰੋਕੋ;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_MUL_MAT:
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]] != -1) {
                m_ਗੰਢ_dynamic_dims[ਗੰਢ] = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]];
            }
            ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[1]] != -1) {
                m_ਗੰਢ_dynamic_dims[ਗੰਢ] = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[1]];
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]] != -1) {
                ਸਵੈਚਾਲਿਤ dynamic_dim_idx = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
                    ਜੇ (ਗੰਢ->op_params[i] == dynamic_dim_idx) {
                        m_ਗੰਢ_dynamic_dims[ਗੰਢ] = i;
                        ਰੋਕੋ;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]] != -1) {
                ਜੇ (ਗੰਢ->src[0]->op == GGML_OP_NONE) {
                    m_ਗੰਢ_dynamic_dims[ਗੰਢ] = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]];
                    ਰੋਕੋ;
                }
                ਸਵੈਚਾਲਿਤ dynamic_dim_idx = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]];
                ਸਵੈਚਾਲਿਤ dynamic_dim_ਮੁੱਲ = ਗੰਢ->src[0]->ne[dynamic_dim_idx];
                ਸਵੈਚਾਲਿਤ dynamic_dim_stride =
                    ਗੰਢ->src[0]->nb[dynamic_dim_idx] / ggml_type_ਮਾਪ(ਗੰਢ->src[0]->type) * ggml_type_ਮਾਪ(ਗੰਢ->type);
                ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
                    ਜੇ (ਗੰਢ->nb[i] == dynamic_dim_stride) {
                        m_ਗੰਢ_dynamic_dims[ਗੰਢ] = i;
                        ਰੋਕੋ;
                    }
                }
                ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ] != -1 && dynamic_dim_ਮੁੱਲ != ਗੰਢ->ne[m_ਗੰਢ_dynamic_dims[ਗੰਢ]]) {
                    m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    ਜੇ (ggml_nelements(ਗੰਢ) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      ਗੰਢ->name, ਗੰਢ->src[0]->name);
                    }
                }
            }
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_RESHAPE: {
            ਜੇ (is_same_shape(ਗੰਢ->src[0], ਗੰਢ)) {
                m_ਗੰਢ_dynamic_dims[ਗੰਢ] = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]];
                ਰੋਕੋ;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]] != -1) {
                ਸਵੈਚਾਲਿਤ dynamic_dim_idx = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]];
                ਸਵੈਚਾਲਿਤ dynamic_dim_stride = ਗੰਢ->src[0]->nb[dynamic_dim_idx];
                ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
                    ਜੇ (ਗੰਢ->nb[i] == dynamic_dim_stride && ਗੰਢ->ne[i] == ਗੰਢ->src[0]->ne[dynamic_dim_idx]) {
                        m_ਗੰਢ_dynamic_dims[ਗੰਢ] = i;
                        ਰੋਕੋ;
                    }
                }
                ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", ਗੰਢ->name);
                }
            }
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr ਪੂਰਨ q_ਮੰਜ਼ਿਲ_ਬਾਹਰਲਾ[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]] != -1) {
                ਸਵੈਚਾਲਿਤ q_dynamic_dim = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]];
                m_ਗੰਢ_dynamic_dims[ਗੰਢ] = q_ਮੰਜ਼ਿਲ_ਬਾਹਰਲਾ[q_dynamic_dim];
            }
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_OP_CONT:
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]] != -1) {
                ਸਵੈਚਾਲਿਤ dynamic_dim_idx = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]];
                ਜੇ (ggml_are_same_shape(ਗੰਢ, ਗੰਢ->src[0])) {
                    m_ਗੰਢ_dynamic_dims[ਗੰਢ] = dynamic_dim_idx;
                } ਨਹੀਂਤਾਂ {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_ਮਾਪ(ਗੰਢ->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (ਗੰਢ->src[0]->ne[0] / ggml_blck_ਮਾਪ(ਗੰਢ->src[0]->type));
                    ਲਈ (ਪੂਰਨ i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * ਗੰਢ->src[0]->ne[i - 1];
                    }

                    ਸਵੈਚਾਲਿਤ dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_ਮਾਪ(ਗੰਢ->src[0]->type) *
                                              ggml_type_ਮਾਪ(ਗੰਢ->type);
                    ਪੂਰਨ matched_dim_ਗਿਣਤੀ = 0;
                    ਪੂਰਨ ਪਹਿਲਾ_matched_dim = -1;
                    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
                        ਜੇ (ਗੰਢ->nb[i] == dynamic_dim_stride && ਗੰਢ->ne[i] == ਗੰਢ->src[0]->ne[dynamic_dim_idx]) {
                            ਜੇ (ਪਹਿਲਾ_matched_dim == -1) {
                                ਪਹਿਲਾ_matched_dim = i;
                            }
                            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = i;
                            matched_dim_ਗਿਣਤੀ++;
                        }
                    }
                    ਜੇ (matched_dim_ਗਿਣਤੀ > 1 && ਗੰਢ->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_ਗੰਢ_dynamic_dims[ਗੰਢ] = ਪਹਿਲਾ_matched_dim;
                    } ਨਹੀਂਤਾਂ ਜੇ (matched_dim_ਗਿਣਤੀ != 1) {
                        m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      ਗੰਢ->name, ਗੰਢ->src[0]->name);
                    }
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONCAT:
            ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
                ਜੇ (ਗੰਢ->src[0]->ne[i] != ਗੰਢ->ne[i]) {
                    m_ਗੰਢ_dynamic_dims[ਗੰਢ] = i;
                    ਰੋਕੋ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_CONV:
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = 1;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM:
        ਹਾਲਤ GGML_OP_L2_NORM:
        ਹਾਲਤ GGML_OP_NORM:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_SUB:
        ਹਾਲਤ GGML_OP_GLU:
        ਹਾਲਤ GGML_OP_ROPE:
        ਹਾਲਤ GGML_OP_SCALE:
        ਹਾਲਤ GGML_OP_SOFT_MAX:
        ਹਾਲਤ GGML_OP_ARGSORT:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
        ਹਾਲਤ GGML_OP_UNARY:
        ਹਾਲਤ GGML_OP_CUMSUM:
        ਹਾਲਤ GGML_OP_FILL:
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
        ਹਾਲਤ GGML_OP_DIAG:
        ਹਾਲਤ GGML_OP_TRI:
        ਹਾਲਤ GGML_OP_REPEAT:
        ਹਾਲਤ GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        ਹਾਲਤ GGML_OP_DIV:
        ਹਾਲਤ GGML_OP_CLAMP:
        ਹਾਲਤ GGML_OP_PAD:
        ਹਾਲਤ GGML_OP_UPSCALE:
        ਹਾਲਤ GGML_OP_SIN:
        ਹਾਲਤ GGML_OP_COS:
        ਹਾਲਤ GGML_OP_LOG:
        ਹਾਲਤ GGML_OP_ROLL:
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]];
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] =
                (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]] == 0) ? -1 : m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[0]];
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[1]];
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CPY:
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
        ਹਾਲਤ GGML_OP_ਜੋੜ:
        ਹਾਲਤ GGML_OP_ਔਸਤ:
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ: {
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[1]] != -1) {
                ਅਚਲ ਤਾਰਕਿਕ is_2D = ਗੰਢ->op_params[6] == 1;
                ਅਚਲ ਪੂਰਨ src_dyn = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[1]];
                ਜੇ (is_2D) {
                    ਜੇ (src_dyn == 0) {
                        m_ਗੰਢ_dynamic_dims[ਗੰਢ] = 1;  // IW -> OW
                    } ਨਹੀਂਤਾਂ ਜੇ (src_dyn == 1) {
                        m_ਗੰਢ_dynamic_dims[ਗੰਢ] = 2;  // IH -> OH
                    } ਨਹੀਂਤਾਂ ਜੇ (src_dyn == 3) {
                        m_ਗੰਢ_dynamic_dims[ਗੰਢ] = 3;  // N  -> N
                    }
                } ਨਹੀਂਤਾਂ {
                    ਜੇ (src_dyn == 0) {
                        m_ਗੰਢ_dynamic_dims[ਗੰਢ] = 1;  // IW -> OW
                    } ਨਹੀਂਤਾਂ ਜੇ (src_dyn == 2) {
                        m_ਗੰਢ_dynamic_dims[ਗੰਢ] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ] != -1) {
                    OPENVINO_ASSERT(ਗੰਢ->src[1]->ne[src_dyn] == ਗੰਢ->ne[m_ਗੰਢ_dynamic_dims[ਗੰਢ]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(ਗੰਢ->name) +
                                        " and its src[1]: " + std::string(ਗੰਢ->src[1]->name));
                }
            }
            ਰੋਕੋ;
        }
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D: {
            m_ਗੰਢ_dynamic_dims[ਗੰਢ] = -1;
            ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[1]] != -1) {
                ਅਚਲ ਪੂਰਨ src_dyn = m_ਗੰਢ_dynamic_dims[ਗੰਢ->src[1]];
                ਜੇ (src_dyn == 0) {
                    m_ਗੰਢ_dynamic_dims[ਗੰਢ] = 1;  // IW -> OW
                } ਨਹੀਂਤਾਂ ਜੇ (src_dyn == 1) {
                    m_ਗੰਢ_dynamic_dims[ਗੰਢ] = 2;  // IH -> OH
                } ਨਹੀਂਤਾਂ ਜੇ (src_dyn == 3) {
                    m_ਗੰਢ_dynamic_dims[ਗੰਢ] = 3;  // N  -> N
                }
                ਜੇ (m_ਗੰਢ_dynamic_dims[ਗੰਢ] != -1) {
                    OPENVINO_ASSERT(ਗੰਢ->src[1]->ne[src_dyn] == ਗੰਢ->ne[m_ਗੰਢ_dynamic_dims[ਗੰਢ]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(ਗੰਢ->name) +
                                        " and its src[1]: " + std::string(ਗੰਢ->src[1]->name));
                }
            }
            ਰੋਕੋ;
        }
        ਮੂਲ:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(ਗੰਢ->op), ਗੰਢ->name);
            ਰੋਕੋ;
        }
    };

    ਲਈ (ਪੂਰਨ i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ਗੰਢ = m_cgraph->nodes[i];
        ਦੇਖੋ_ਗੰਢ(ਦੇਖੋ_ਗੰਢ, ਗੰਢ);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    ਜੇ (0) {
        ਲਈ (ਪੂਰਨ i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * ਗੰਢ = m_cgraph->nodes[i];
            ਪੂਰਨ dynamic_dim = m_ਗੰਢ_dynamic_dims[ਗੰਢ];
            std::ਬਾਹਰ << "[" << i << "] " << "node_name: " << ਗੰਢ->name << " op: " << ggml_op_name(ਗੰਢ->op)
                      << " shape: [";
            ਲਈ (ਪੂਰਨ j = 0; j < 4; j++) {
                ਜੇ (j == dynamic_dim) {
                    std::ਬਾਹਰ << "*";
                } ਨਹੀਂਤਾਂ {
                    std::ਬਾਹਰ << ਗੰਢ->ne[j];
                }
                ਜੇ (j < 3) {
                    std::ਬਾਹਰ << ", ";
                }
            }
            std::ਬਾਹਰ << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = ਗੰਢ->src[j];
                ਜੇ (src == ਖਾਲੀ_ਸੂਚਕ) {
                    ਜਾਰੀ;
                }
                ਪੂਰਨ src_dynamic_dim = m_ਗੰਢ_dynamic_dims[src];
                std::ਬਾਹਰ << "    [" << j << "] src_name: " << src->name << " [";
                ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
                    ਜੇ (k == src_dynamic_dim) {
                        std::ਬਾਹਰ << "*";
                    } ਨਹੀਂਤਾਂ {
                        std::ਬਾਹਰ << src->ne[k];
                    }
                    ਜੇ (k < 3) {
                        std::ਬਾਹਰ << ", ";
                    }
                }
                std::ਬਾਹਰ << "]" << '\n';
            }
            std::ਬਾਹਰ << '\n';
        }
    }
}
