#ಅನ್ತರ್ಭಾವ "ggml-decoder.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino-extra.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino.h"
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "utils.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <iomanip>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <openvino/core/dimension.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/except.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/partial_shape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/bfloat16.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/element_type.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/float16.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/tensor.hpp>
#ಅನ್ತರ್ಭಾವ <ostream>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <stdexcept>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             ತಾರ್ಕಿಕ is_static,
                             ತಾರ್ಕಿಕ is_stateful,
                             ತಾರ್ಕಿಕ model_is_splitted,
                             ತಾರ್ಕಿಕ is_prefill,
                             ಪೂರ್ಣಾಙ್ಕ prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(ಅಸತ್ಯ),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    ಸ್ಥಿರ ತಾರ್ಕಿಕ printed_address_map = ಅಸತ್ಯ;
    ಯದಿ (!printed_address_map) {
        ಯದಿ (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = ಸತ್ಯ;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_ಒಳಹರಿವು_output();
    compute_ಗಂಟು_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_n = 0; ಗಂಟು_n < cgraph->n_nodes; ಗಂಟು_n++) {
        m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು_op_case = compute_op_case(m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು);
        m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು_op_type = compute_op_type(m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು);
    }

    add_extra_inputs();
}

ಶೂನ್ಯ GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_ಗಂಟು_info_list.clear();
    set_ಒಳಹರಿವು_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
ತಾರ್ಕಿಕ GgmlOvDecoder::is_bound_to(ಧ್ರುವ ggml_cgraph * cgraph) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ m_cgraph == cgraph && cgraph->n_nodes > 0 && m_ಗಂಟು_info_list.size() == (size_t) cgraph->n_nodes &&
           m_ಗಂಟು_info_list.front().ಗಂಟು == cgraph->nodes[0] &&
           m_ಗಂಟು_info_list.back().ಗಂಟು == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = ಸತ್ಯ;
    set_ಒಳಹರಿವು_output();
    compute_model_inputs();
    compute_model_outputs();
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_n = 0; ಗಂಟು_n < cgraph->n_nodes; ಗಂಟು_n++) {
        m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು_op_case = compute_op_case(m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು);
        m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು_op_type = compute_op_type(m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು);
    }
}

ನಾಮಾಕಾಶ {
ತಾರ್ಕಿಕ is_inplace_op(ಧ್ರುವ ggml_tensor * ಗಂಟು) {
    ಪ್ರತ್ಯಾಗಮ ಗಂಟು->op == GGML_OP_SET_ROWS || ಗಂಟು->op == GGML_OP_CPY || (ಗಂಟು->op == GGML_OP_SCALE && ಗಂಟು->view_src);
}

ತಾರ್ಕಿಕ is_same_shape(ಧ್ರುವ ggml_tensor * a, ಧ್ರುವ ggml_tensor * b) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
        ಯದಿ (a->ne[i] != b->ne[i]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ is_conv_states_all_tensor(ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

ತಾರ್ಕಿಕ is_full_single_slot_writeback(ಧ್ರುವ ggml_tensor * ಗಂಟು) {
    ಪ್ರತ್ಯಾಗಮ ಗಂಟು->view_src != nullptr && ಗಂಟು->view_src->ne[1] == 1 && ಗಂಟು->src[1] != nullptr &&
           ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src == ಗಂಟು->view_src &&
           ಗಂಟು->src[1]->view_offs == 0 && ggml_nbytes(ಗಂಟು->src[1]) == ggml_nbytes(ಗಂಟು->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
ತಾರ್ಕಿಕ is_moe_expert_sum_add(ಧ್ರುವ ggml_tensor * ಗಂಟು) {
    std::vector<ಧ್ರುವ ggml_tensor *> leaves;
    ಧ್ರುವ ggml_tensor * cur = ಗಂಟು;
    ಯಾವತ್ (cur->op == GGML_OP_ADD) {
        ಯದಿ (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ಧ್ರುವ ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    ಪ್ರತಿ (ಧ್ರುವ ggml_tensor * leaf : leaves) {
        ಯದಿ (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಧ್ರುವ ggml_tensor * leaf_base = leaf->src[0];
        ಯದಿ (base == nullptr) {
            base = leaf_base;
        } ಅನ್ಯಥಾ ಯದಿ (leaf_base != base) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        ಯದಿ (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ಧ್ರುವ ggml_cgraph * cgraph, ಧ್ರುವ ggml_tensor * tensor) {
    ಯದಿ (tensor == nullptr) {
        ಪ್ರತ್ಯಾಗಮ "";
    }
    ಯದಿ ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ಧ್ರುವ auto * ಗಂಟು = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        ಯದಿ (ಗಂಟು != cgraph->nodes + cgraph->n_nodes) {
            ಪ್ರತ್ಯಾಗಮ std::string(tensor->name) + "#n" + std::to_string(ಗಂಟು - cgraph->nodes);
        }
        ಧ್ರುವ auto * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        ಯದಿ (leaf != cgraph->leafs + cgraph->n_leafs) {
            ಪ್ರತ್ಯಾಗಮ std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    ಪ್ರತ್ಯಾಗಮ tensor->name;
}

ಸ್ಥಿರ std::string get_tensor_ov_name(ಧ್ರುವ ggml_cgraph * cgraph, ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

ಸ್ಥಿರ std::string get_tensor_graph_ಒಳಹರಿವು_ov_name(ಧ್ರುವ GgmlOvDecoder * decoder,
                                                  ಧ್ರುವ ggml_cgraph * cgraph,
                                                  ಧ್ರುವ ggml_tensor * tensor,
                                                  ಧ್ರುವ ggml_tensor * op) {
    ಯದಿ (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        ಪ್ರತ್ಯಾಗಮ "inp_pos";
    }
    ಯದಿ (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        ಪ್ರತ್ಯಾಗಮ "embd";
    }
    ಯದಿ (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ಧ್ರುವ ತಾರ್ಕಿಕ is_swa = decoder->is_swa_mask(tensor);
        ಯದಿ (decoder->is_stateful()) {
            ಪ್ರತ್ಯಾಗಮ is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        ಯದಿ (is_swa) {
            ಪ್ರತ್ಯಾಗಮ get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    ಪ್ರತ್ಯಾಗಮ get_tensor_ov_name(cgraph, tensor);
}

ಶೂನ್ಯ GgmlOvDecoder::set_ಒಳಹರಿವು_output() {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_n = 0; ಗಂಟು_n < m_cgraph->n_nodes; ಗಂಟು_n++) {
        auto * ಗಂಟು = m_cgraph->nodes[ಗಂಟು_n];

        ಗಂಟು_Info current_ಗಂಟು_info;
        auto ಗಂಟು_name = get_tensor_ov_name(m_cgraph, ಗಂಟು);

        current_ಗಂಟು_info.ಗಂಟು = ಗಂಟು;
        current_ಗಂಟು_info.ಗಂಟು_name = ಗಂಟು_name;
        current_ಗಂಟು_info.ಗಂಟು_op_case = 0;
        current_ಗಂಟು_info.data_addr = ಗಂಟು->data;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ಗಂಟು->src[i];
            ಯದಿ (src == nullptr) {
                ಅನುವರ್ತ;
            }
            auto src_name = get_tensor_ov_name(m_cgraph, src);
            ಯದಿ (src->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
                src_name = get_tensor_graph_ಒಳಹರಿವು_ov_name(ಏತತ್, m_cgraph, src, ಗಂಟು);
            }
            current_ಗಂಟು_info.ಗಂಟು_inputs[src_name] = src;
            current_ಗಂಟು_info.ಗಂಟು_inputs_names.push_back(src_name);

            ಯದಿ (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_ಗಂಟು_info.ಗಂಟು_inputs_views[src_name])> view_chain;
                auto * current = src;

                ಯಾವತ್ (current != nullptr) {
                    auto current_name = get_tensor_ov_name(m_cgraph, current);
                    ಯದಿ (current->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
                        current_name = get_tensor_graph_ಒಳಹರಿವು_ov_name(ಏತತ್, m_cgraph, current, ಗಂಟು);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    ಯದಿ (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } ಅನ್ಯಥಾ {
                        ಭಙ್ಗ;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_ಗಂಟು_info.ಗಂಟು_inputs_views[src_name] = view_chain;
            }
        }

        m_ಗಂಟು_info_list.push_back(current_ಗಂಟು_info);
    }
}

ಪೂರ್ಣಾಙ್ಕ GgmlOvDecoder::compute_op_case(ಧ್ರುವ ggml_tensor * ಗಂಟು) ಧ್ರುವ {
    ಪೂರ್ಣಾಙ್ಕ op_case = 0;
    ವಿಕಲ್ಪ (ಗಂಟು->op) {
    ಪಕ್ಷ GGML_OP_RESHAPE: {
        ಯದಿ (m_naive) {
            ಭಙ್ಗ;
        }
        auto name = std::string(ಗಂಟು->name);
        auto * src = ಗಂಟು->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        ತಾರ್ಕಿಕ recurrent_sequence = ಅಸತ್ಯ;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ಧ್ರುವ auto * consumer = m_cgraph->nodes[i];
            ಯದಿ (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == ಗಂಟು) {
                ಪ್ರತ್ಯಾಗಮ 1;
            } ಅನ್ಯಥಾ ಯದಿ (consumer->op == GGML_OP_SSM_CONV) {
                ಧ್ರುವ auto * concat = consumer->src[0];
                ಯದಿ (concat->op == GGML_OP_CONCAT) {
                    ಧ್ರುವ auto * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == ಗಂಟು;
                }
            } ಅನ್ಯಥಾ ಯದಿ (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                ಧ್ರುವ auto * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == ಗಂಟು;
            }
        }
        ಯದಿ (recurrent_sequence && ಗಂಟು->ne[0] == src->ne[0] && ಗಂಟು->ne[3] == 1) {
            ಪ್ರತ್ಯಾಗಮ 6;
        }
        ಯದಿ (ಗಂಟು->ne[0] == src->ne[0] && ಗಂಟು->ne[2] == 1 && ಗಂಟು->ne[3] == 1) {
            ಪ್ರತ್ಯಾಗಮ 5;
        }
        ಯದಿ (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == ಗಂಟು->ne[0] && src->src[0]->ne[1] == ಗಂಟು->ne[1]) {
            op_case = 4;
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು->ne[0] * ಗಂಟು->ne[1] == src->ne[0]) {
            op_case = 1;
        } ಅನ್ಯಥಾ ಯದಿ (src->ne[0] * src->ne[1] == ಗಂಟು->ne[0]) {
            op_case = 2;
            ಯದಿ (src->ne[2] * src->ne[3] == ಗಂಟು->ne[1]) {
                op_case = 5;
            }
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == ಗಂಟು->ne[1]) {
            op_case = 3;
        } ಅನ್ಯಥಾ ಯದಿ (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } ಅನ್ಯಥಾ ಯದಿ (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } ಅನ್ಯಥಾ ಯದಿ (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        ಯದಿ (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            ಧ್ರುವ auto * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            ತಾರ್ಕಿಕ recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ಧ್ರುವ auto * consumer = m_cgraph->nodes[i];
                ಯದಿ (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j : {3, 4}) {
                        ಧ್ರುವ auto * gate = consumer->src[j];
                        ಯದಿ (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == ಗಂಟು;
                    }
                } ಅನ್ಯಥಾ ಯದಿ (consumer->op == GGML_OP_MUL) {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 2; ++j) {
                        ಧ್ರುವ auto * gate = consumer->src[j];
                        ಧ್ರುವ auto * norm = consumer->src[1 - j];
                        ಯದಿ (gate->op != GGML_OP_UNARY || gate->src[0] != ಗಂಟು) {
                            ಅನುವರ್ತ;
                        }
                        ಯದಿ (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            ಯದಿ (recurrent) {
                op_case = 9;
            }
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_PERMUTE: {
        ಯದಿ (ಗಂಟು->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(ಗಂಟು->view_src->name);
            ಪೂರ್ಣಾಙ್ಕ layer = extract_layer_from_name(src_name).value();
            ಯದಿ (ggml_is_contiguous(ಗಂಟು->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                ಯದಿ (!is_swa_layer(layer)) {
                    op_case = 3;
                } ಅನ್ಯಥಾ {
                    op_case = 4;
                }
            } ಅನ್ಯಥಾ {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                ಯದಿ (!is_swa_layer(layer)) {
                    op_case = 5;
                } ಅನ್ಯಥಾ {
                    op_case = 6;
                }
            }
        } ಅನ್ಯಥಾ {
            // rope'ed query tensor
            op_case = 2;
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_MUL_MAT: {
        ಯದಿ (ಗಂಟು->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_GET_ROWS: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        ಯದಿ (ಗಂಟು->src[0]->op == GGML_OP_RESHAPE && ಗಂಟು->src[0]->src[0] != nullptr &&
            is_recurrent_cache(ಗಂಟು->src[0]->src[0])) {
            ಧ್ರುವ ತಾರ್ಕಿಕ single_slot = ಗಂಟು->src[0]->src[0]->ne[1] == 1;
            ಯದಿ (ಗಂಟು->src[1]->op == GGML_OP_VIEW) {
                op_case = (ಗಂಟು->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } ಅನ್ಯಥಾ ಯದಿ (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_ROPE: {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode = ಗಂಟು->op_params[2];
        ವಿಕಲ್ಪ (mode) {
        ಪಕ್ಷ GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            ಭಙ್ಗ;
        }
        ಸ್ವಭಾವ:
            op_case = 0;
            ಭಙ್ಗ;
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_VIEW: {
        ಯದಿ (!m_model_params.has_rs_rollback && ಗಂಟು->src[0] != nullptr &&
            ಗಂಟು->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            ಭಙ್ಗ;
        }
        ಯದಿ (m_is_static && ಗಂಟು->src[0] != nullptr &&
            (ಗಂಟು->src[0]->op == GGML_OP_GATED_DELTA_NET || ಗಂಟು->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            ಭಙ್ಗ;
        }
        ಯದಿ (ಗಂಟು->src[0]->op == GGML_OP_VIEW) {
            auto * src = ಗಂಟು->src[0];
            ಯದಿ (ggml_nelements(ಗಂಟು) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            ಯದಿ (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            auto * src = ಗಂಟು->src[0];
            ಯದಿ (ggml_nelements(ಗಂಟು) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                ಯದಿ (ಗಂಟು->ne[0] == src->ne[0] && ಗಂಟು->ne[1] == src->ne[2] && ಗಂಟು->ne[2] == 1 &&
                    ಗಂಟು->nb[0] == src->nb[0] && ಗಂಟು->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    ಭಙ್ಗ;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                ಪೂರ್ಣಾಙ್ಕ diff_ಎಣಿಕೆ = 0;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
                    ಯದಿ (ಗಂಟು->ne[i] != src->ne[i]) {
                        diff_ಎಣಿಕೆ++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    ಯದಿ (ಗಂಟು->ne[i] > src->ne[i]) {
                        ಪ್ರತ್ಯಾಗಮ 0;
                    }
                }
                ಯದಿ (diff_ಎಣಿಕೆ >= 1) {
                    op_case = 0;
                }
            }
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_RMS_NORM: {
        ಯದಿ (ಗಂಟು->src[0]->op == GGML_OP_VIEW) {
            ಯದಿ (is_same_shape(ಗಂಟು->src[0]->src[0], ಗಂಟು->src[0])) {
                op_case = 1;
            } ಅನ್ಯಥಾ ಯದಿ (!m_model_params.has_rs_rollback &&
                       ಗಂಟು->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_POOL_2D: {
        ಧ್ರುವ ggml_op_pool pool_mode = static_cast<ggml_op_pool>(ಗಂಟು->op_params[0]);
        ವಿಕಲ್ಪ (pool_mode) {
        ಪಕ್ಷ GGML_OP_POOL_MAX: {
            op_case = 1;
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ GGML_OP_POOL_AVG: {
            op_case = 2;
            ಭಙ್ಗ;
        }
        ಸ್ವಭಾವ:
            op_case = 0;
            ಭಙ್ಗ;
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_UPSCALE: {
        ಧ್ರುವ int32_t mode_flags = ಗಂಟು->op_params[0];
        ಧ್ರುವ ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        ವಿಕಲ್ಪ (scale_mode) {
        ಪಕ್ಷ GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            ಭಙ್ಗ;
        }
        ಸ್ವಭಾವ:
            op_case = 0;
            ಭಙ್ಗ;
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_CPY: {
        ಯದಿ (ಗಂಟು->src[0]->op == GGML_OP_VIEW) {
            ಯದಿ (ಗಂಟು->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                ಯದಿ (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(ಗಂಟು) ? 7 : 10;
                } ಅನ್ಯಥಾ {
                    op_case = 1;
                }
            } ಅನ್ಯಥಾ ಯದಿ (GgmlOvDecoder::is_conv_state_writeback(ಗಂಟು)) {
                op_case = is_full_single_slot_writeback(ಗಂಟು) ? 8 : 2;
                ಭಙ್ಗ;
            } ಅನ್ಯಥಾ ಯದಿ (is_conv_states_all_tensor(ಗಂಟು->view_src) && ಗಂಟು->src[1] != nullptr &&
                       ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src == ಗಂಟು->view_src) {
                op_case = 4;
                ಭಙ್ಗ;
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು->src[0]->src[0]->op == GGML_OP_GET_ROWS && ಗಂಟು->src[1] != nullptr &&
                       ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src != nullptr &&
                       is_recurrent_cache(ಗಂಟು->src[1]->view_src) && ಗಂಟು->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು->src[0]->op == GGML_OP_GET_ROWS && ಗಂಟು->src[1] != nullptr &&
                   ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src != nullptr &&
                   is_recurrent_cache(ಗಂಟು->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = ಗಂಟು->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು->src[1] != nullptr && ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ಧ್ರುವ ggml_tensor * kv_buf = ಗಂಟು->src[1]->view_src;
            ಯದಿ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m_cgraph->n_nodes; i++) {
                    ಧ್ರುವ ggml_tensor * n = m_cgraph->nodes[i];
                    ಯದಿ (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        ಅನುವರ್ತ;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    ಯದಿ ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        ಯದಿ (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        ಭಙ್ಗ;
                    }
                }
            }
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_ADD: {
        ಯದಿ (is_moe_expert_sum_add(ಗಂಟು)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_SCALE: {
        ಯದಿ (ಗಂಟು->view_src && ಗಂಟು->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = ಗಂಟು->view_src->ne[1] == 1 ? 2 : 1;
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_L2_NORM: {
        ಯದಿ (std::string(ಗಂಟು->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        ಭಙ್ಗ;
    }
    ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT: {
        ಯದಿ (ಗಂಟು->src[1] != nullptr && ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src != nullptr) {
            ಧ್ರುವ ggml_tensor * kv_buf = ಗಂಟು->src[1]->view_src;
            ಯದಿ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (ಗಂಟು->src[3] != nullptr) ? 1 : 2;
            }
        }
        ಭಙ್ಗ;
    }
    ಸ್ವಭಾವ:
        ಭಙ್ಗ;
    }
    ಪ್ರತ್ಯಾಗಮ op_case;
}

std::optional<ಪೂರ್ಣಾಙ್ಕ> extract_layer_from_name(ಧ್ರುವ std::string & name) {
    size_t pos1 = name.find("_l");
    ಯದಿ (pos1 == std::string::npos) {
        ಪ್ರತ್ಯಾಗಮ std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    ಯದಿ (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    ಪೂರ್ಣಾಙ್ಕ layer = std::stoi(layer_str);
    ಪ್ರತ್ಯಾಗಮ layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ get_swa_window_from_mask(ಧ್ರುವ ggml_tensor * mask) {
    ಯದಿ (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }
    ಯದಿ (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t n_kv = mask->ne[0];
    ಧ್ರುವ int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    ಪ್ರತಿ (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        ಪ್ರತಿ (int64_t c = 0; c < n_kv; c++) {
            ಧ್ರುವ size_t i = (size_t) r * n_kv + c;
            ಧ್ರುವ ಭಿನ್ನ v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((ಧ್ರುವ ggml_fp16_t *) mask->data)[i]) :
                                                          ((ಧ್ರುವ ಭಿನ್ನ *) mask->data)[i];
            ಯದಿ (v > -INFINITY) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    ಪ್ರತ್ಯಾಗಮ window > 0 ? (ಪೂರ್ಣಾಙ್ಕ) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, ತಾರ್ಕಿಕ is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    auto get_attention_pattern_case = [](ಧ್ರುವ ggml_tensor * ಗಂಟು) -> ಪೂರ್ಣಾಙ್ಕ {
        ಯದಿ (ಗಂಟು == nullptr) {
            ಪ್ರತ್ಯಾಗಮ -1;
        }

        ವಿಕಲ್ಪ (ಗಂಟು->op) {
        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
            ಯದಿ (ಗಂಟು->src[0] == nullptr || ಗಂಟು->src[1] == nullptr) {
                ಪ್ರತ್ಯಾಗಮ -1;
            }
            ವಿಕಲ್ಪ (ಗಂಟು->src[1]->op) {
            ಪಕ್ಷ GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                ಯದಿ (ಗಂಟು->src[3] != nullptr && ಗಂಟು->src[1]->src[0] != nullptr &&
                    ಗಂಟು->src[1]->src[0]->op == GGML_OP_VIEW) {
                    ಪ್ರತ್ಯಾಗಮ 0;
                }
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                ಯದಿ (ಗಂಟು->src[3] != nullptr && ಗಂಟು->src[1]->src[0] != nullptr &&
                    ಗಂಟು->src[1]->src[0]->op == GGML_OP_PERMUTE && ಗಂಟು->src[1]->src[0]->src[0] != nullptr &&
                    ಗಂಟು->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    ಪ್ರತ್ಯಾಗಮ 1;
                }
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                ಯದಿ (ಗಂಟು->src[1]->view_src != nullptr) {
                    ಯದಿ (ಗಂಟು->src[3] != nullptr) {
                        ಪ್ರತ್ಯಾಗಮ 4;  // decoder self-attention
                    }
                    ಪ್ರತ್ಯಾಗಮ 5;      // cross-attention or encoder self-attention
                }
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                ಭಙ್ಗ;
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            ಯದಿ (ಗಂಟು->src[0] != nullptr && ಗಂಟು->src[1] != nullptr && ಗಂಟು->src[0]->op == GGML_OP_MUL_MAT &&
                ಗಂಟು->src[0]->src[0] != nullptr && ಗಂಟು->src[0]->src[1] != nullptr &&
                ಗಂಟು->src[0]->src[0]->op == GGML_OP_PERMUTE && ಗಂಟು->src[0]->src[0]->src[0] != nullptr &&
                ಗಂಟು->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                ಪ್ರತ್ಯಾಗಮ 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            ಯದಿ (ಗಂಟು->src[0]->op == GGML_OP_ADD && ಗಂಟು->src[0]->src[0] != nullptr &&
                ಗಂಟು->src[0]->src[0]->op == GGML_OP_MUL_MAT && ಗಂಟು->src[0]->src[0]->src[0] != nullptr &&
                ಗಂಟು->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                ಪ್ರತ್ಯಾಗಮ 3;
            }
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
        }

        ಪ್ರತ್ಯಾಗಮ -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    auto get_attention_op_mask = [&get_attention_pattern_case](ಧ್ರುವ ggml_tensor * ಗಂಟು) -> ಧ್ರುವ ggml_tensor * {
        ವಿಕಲ್ಪ (get_attention_pattern_case(ಗಂಟು)) {
        ಪಕ್ಷ 0:
        ಪಕ್ಷ 1:
            ಪ್ರತ್ಯಾಗಮ ಗಂಟು->src[3];
        ಪಕ್ಷ 2:
        ಪಕ್ಷ 3:
            ಪ್ರತ್ಯಾಗಮ ಗಂಟು->src[1];
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ nullptr;
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
        std::map<ಪೂರ್ಣಾಙ್ಕ, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<ಪೂರ್ಣಾಙ್ಕ, ಧ್ರುವ ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
            ಧ್ರುವ ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            ಯದಿ (mask == nullptr) {
                ಅನುವರ್ತ;
            }
            ಧ್ರುವ ggml_tensor * cache_k_permute = nullptr;
            ವಿಕಲ್ಪ (get_attention_pattern_case(cgraph->nodes[i])) {
            ಪಕ್ಷ 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     ಭಙ್ಗ;
            ಪಕ್ಷ 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             ಭಙ್ಗ;
            ಪಕ್ಷ 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             ಭಙ್ಗ;
            ಸ್ವಭಾವ: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     ಭಙ್ಗ;
            }
            ಧ್ರುವ ggml_tensor * cache_k_view = cache_k_permute->src[0];
            ಯದಿ (cache_k_view->op != GGML_OP_VIEW) {
                ಅನುವರ್ತ;
            }
            ಧ್ರುವ ggml_tensor * leaf = cache_k_view->src[0];
            auto layer = extract_layer_from_name(leaf->name);
            ಯದಿ (!layer.has_value()) {
                ಅನುವರ್ತ;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        ಪ್ರತಿ (ಧ್ರುವ auto & [layer, extent] : layer_extent) {
            ಯದಿ (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                ಯದಿ (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        ಯದಿ (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            ಪ್ರತಿ (ಧ್ರುವ auto & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (long) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    ತಾರ್ಕಿಕ rope_seen = ಅಸತ್ಯ;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = cgraph->nodes[i];
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ attention_pattern_case = get_attention_pattern_case(ಗಂಟು);
        ಯದಿ (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            ವಿಕಲ್ಪ (attention_pattern_case) {
            ಪಕ್ಷ 0:
                cache_k_permute = ಗಂಟು->src[1];
                mask = ಗಂಟು->src[3];
                ಭಙ್ಗ;
            ಪಕ್ಷ 1:
                cache_k_permute = ಗಂಟು->src[1]->src[0];
                mask = ಗಂಟು->src[3];
                ಭಙ್ಗ;
            ಪಕ್ಷ 2:
                cache_k_permute = ಗಂಟು->src[0]->src[0];
                mask = ಗಂಟು->src[1];
                ಭಙ್ಗ;
            ಪಕ್ಷ 3:
                cache_k_permute = ಗಂಟು->src[0]->src[0]->src[0];
                mask = ಗಂಟು->src[1];
                ಭಙ್ಗ;
            ಪಕ್ಷ 4:
            ಪಕ್ಷ 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                auto * cache_k_view = ಗಂಟು->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = ಗಂಟು->src[0]->ne[1];
                ಯದಿ (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } ಅನ್ಯಥಾ {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                ಅನುವರ್ತ;
            }
            ಸ್ವಭಾವ:
                ಭಙ್ಗ;
            }

            ಪ್ರತಿಜ್ಞಾ(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.ಒಳಹರಿವು_len = ಗಂಟು->src[0]->ne[1];
            compute_params.token_len_per_seq = ಗಂಟು->src[0]->ne[1];

            auto * cache_k_view = cache_k_permute->src[0];
            ಯದಿ (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                ಅನುವರ್ತ;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            ಪೂರ್ಣಾಙ್ಕ layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ಧ್ರುವ ತಾರ್ಕಿಕ layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            ಯದಿ (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } ಅನ್ಯಥಾ {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, ಪರಿಮಾಣ(size_t));
            compute_params.seq_active_start = offset / seq_size;

            ಯದಿ (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } ಅನ್ಯಥಾ {
                compute_params.attention_size = mask->ne[0];
            }
            ಯದಿ (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        ಯದಿ (ಗಂಟು->op == GGML_OP_MUL_MAT && ಗಂಟು->src[0]->op == GGML_OP_PERMUTE &&
            ಗಂಟು->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(ಗಂಟು->src[0]->view_src, ಗಂಟು->view_src)) {
            ಯದಿ (ಗಂಟು->src[1]->op == GGML_OP_PERMUTE && ಗಂಟು->src[1]->src[0]->op == GGML_OP_VIEW &&
                ಗಂಟು->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = ಗಂಟು->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        ಯದಿ (ಗಂಟು->op == GGML_OP_TRANSPOSE && ಗಂಟು->src[0]->op == GGML_OP_PERMUTE &&
            ಗಂಟು->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = ಗಂಟು->ne[0];
            ಯದಿ (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        ಯದಿ (ಗಂಟು->op == GGML_OP_ROPE) {
            ಯದಿ (compute_params.token_len_per_seq == -1 && ಗಂಟು->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(ಗಂಟು->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(ಪರಿಮಾಣ(model_params.rope_params) == ಪರಿಮಾಣ(int32_t) * 16, "rope_params size");
            ಯದಿ (!rope_seen) {
                memcpy(model_params.rope_params, ಗಂಟು->op_params, ಪರಿಮಾಣ(int32_t) * 16);
                rope_seen = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (memcmp(model_params.rope_params, ಗಂಟು->op_params, ಪರಿಮಾಣ(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = ಸತ್ಯ;
            }
        }
        ಯದಿ (ಗಂಟು->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = ಗಂಟು->src[0]->ne[0];
        }
        ಯದಿ (ಗಂಟು->op == GGML_OP_SCALE && ಗಂಟು->view_src != nullptr && is_recurrent_cache(ಗಂಟು->view_src)) {
            ಯದಿ (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = ಗಂಟು->view_src->ne[1];
            } ಅನ್ಯಥಾ {
                GGML_ASSERT(model_params.n_rs_slots == ಗಂಟು->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(ಗಂಟು) / ಗಂಟು->view_src->ne[0];
            compute_params.cache_rs_reset_idx = ಗಂಟು->src[0]->view_offs / ಗಂಟು->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        ಯದಿ (ಗಂಟು->op == GGML_OP_CPY && ಗಂಟು->view_src != nullptr && is_recurrent_cache(ಗಂಟು->view_src) &&
            ಗಂಟು->src[1] != nullptr && ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src == ಗಂಟು->view_src) {
            ಧ್ರುವ ತಾರ್ಕಿಕ is_conv = is_conv_state_writeback(ಗಂಟು);
            ಧ್ರುವ ತಾರ್ಕಿಕ is_gdn = ಗಂಟು->src[0]->op == GGML_OP_VIEW && ಗಂಟು->src[0]->src[0] != nullptr &&
                                ಗಂಟು->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ಧ್ರುವ ತಾರ್ಕಿಕ is_extra = ಗಂಟು->src[0]->op == GGML_OP_GET_ROWS;
            ಧ್ರುವ ತಾರ್ಕಿಕ is_gdn_rollback = is_gdn && is_same_shape(ಗಂಟು->src[0], ಗಂಟು->src[1]);

            ಧ್ರುವ ggml_tensor * dest_view = ಗಂಟು->src[1];
            ಧ್ರುವ ggml_tensor * cache = ಗಂಟು->view_src;
            ಧ್ರುವ size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            ಯದಿ (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = ಸತ್ಯ;
            }
            ಯದಿ (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(ಗಂಟು)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (ಪೂರ್ಣಾಙ್ಕ) (dest_view->view_offs / row_bytes);
                ಯದಿ (is_conv) {
                    writeback.src_begin = (ಪೂರ್ಣಾಙ್ಕ) (ಗಂಟು->src[0]->view_offs / ಗಂಟು->src[0]->view_src->nb[0]);
                } ಅನ್ಯಥಾ ಯದಿ (is_gdn_rollback) {
                    writeback.src_begin = (ಪೂರ್ಣಾಙ್ಕ) (ಗಂಟು->src[0]->view_offs / ಗಂಟು->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, ಗಂಟು)] = writeback;
            }
            ಯದಿ ((is_conv || is_gdn) && !is_full_single_slot_writeback(ಗಂಟು)) {
                compute_params.s_copy_active_slot_len = (ಪೂರ್ಣಾಙ್ಕ) dest_view->ne[1];
            }
        }
    }
    ಯದಿ (model_params.n_heads_kv == -1) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
            ಧ್ರುವ auto * ಗಂಟು = cgraph->nodes[i];
            ಧ್ರುವ ggml_tensor * mask = nullptr;
            ಯದಿ (ಗಂಟು->op == GGML_OP_SOFT_MAX) {
                mask = ಗಂಟು->src[1];
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = ಗಂಟು->src[3];
            } ಅನ್ಯಥಾ {
                ಅನುವರ್ತ;
            }
            ಯದಿ (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) ||
                ಗಂಟು->src[0] == nullptr) {
                ಅನುವರ್ತ;
            }
            model_params.is_cacheless_attn = ಸತ್ಯ;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.ಒಳಹರಿವು_len = ಗಂಟು->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.ಒಳಹರಿವು_len;
            ಭಙ್ಗ;
        }
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    ಯದಿ (model_params.is_cacheless_attn) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
            ಧ್ರುವ auto * ಗಂಟು = cgraph->nodes[i];
            ಯದಿ (ಗಂಟು->op == GGML_OP_GET_ROWS && is_output_idx(ಗಂಟು->src[1], ಗಂಟು)) {
                compute_params.output_len = ಗಂಟು->src[1]->ne[0];
                ಭಙ್ಗ;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    ಯದಿ (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    ಪ್ರತ್ಯಾಗಮ {model_params, compute_params};
}

ಶೂನ್ಯ GgmlOvDecoder::validate_cgraph() ಧ್ರುವ {
    ಯದಿ (m_model_params.n_seq > 1 && m_is_static == ಸತ್ಯ) {
        ಕ್ಷಿಪ std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_ಒಳಹರಿವು_shape(ಧ್ರುವ ggml_tensor * op,
                                                      ಧ್ರುವ ggml_tensor * ಒಳಹರಿವು,
                                                      ಪೂರ್ಣಾಙ್ಕ dynamic_dim_index) ಧ್ರುವ {
    ಯದಿ (m_naive) {
        ಪ್ರತ್ಯಾಗಮ ಒಳಹರಿವು != nullptr ? ov::PartialShape{get_shape(ಒಳಹರಿವು)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape ಒಳಹರಿವು_shape;

    ಯದಿ (is_inp_tok(ಒಳಹರಿವು, op) || is_inp_pos(ಒಳಹರಿವು, op)) {
        // tokens or positions
        ಪೂರ್ಣಾಙ್ಕ len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        ಯದಿ (m_is_static && is_inp_pos(ಒಳಹರಿವು, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, 1, len};

    } ಅನ್ಯಥಾ ಯದಿ (is_output_idx(ಒಳಹರಿವು, op)) {
        // output index
        ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } ಅನ್ಯಥಾ ಯದಿ (is_inp_mean(ಒಳಹರಿವು, op)) {
        ಒಳಹರಿವು_shape = m_is_static ? ov::PartialShape{1, 1, ಒಳಹರಿವು->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } ಅನ್ಯಥಾ ಯದಿ (is_inp_scale_rows(ಒಳಹರಿವು, op)) {
        ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } ಅನ್ಯಥಾ ಯದಿ (is_inp_mask(ಒಳಹರಿವು, op)) {
        // mask
        ಯದಿ (m_is_static) {
            ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } ಅನ್ಯಥಾ ಯದಿ (m_is_stateful) {
            ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, -1, -1};
        } ಅನ್ಯಥಾ {
            ಒಳಹರಿವು_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } ಅನ್ಯಥಾ ಯದಿ (is_recurrent_cache(ಒಳಹರಿವು)) {
        ಒಳಹರಿವು_shape = ov::PartialShape{get_shape(ಒಳಹರಿವು)};
        ಯದಿ (!m_is_static && !m_is_stateful && ಒಳಹರಿವು->ne[1] > 1) {
            ಒಳಹರಿವು_shape[2] = -1;
        }

    } ಅನ್ಯಥಾ ಯದಿ (is_kvcache(ಒಳಹರಿವು, op)) {
        // kvcache
        ಒಳಹರಿವು_shape = ov::PartialShape{get_shape(ಒಳಹರಿವು)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ಧ್ರುವ ತಾರ್ಕಿಕ is_flat_kv = (ಒಳಹರಿವು->ne[1] == 1 && ಒಳಹರಿವು->ne[2] == 1 && ಒಳಹರಿವು->ne[3] == 1);
        ಯದಿ (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            ಒಳಹರಿವು_shape[2] = -1;
        }
        ಯದಿ (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            ಪ್ರತಿಜ್ಞಾ(ಒಳಹರಿವು_shape.size() == 4 && ಒಳಹರಿವು_shape[0] == 1 && ಒಳಹರಿವು_shape[1] == 1 &&
                   ಒಳಹರಿವು_shape[2].is_dynamic() && ಒಳಹರಿವು_shape[3].is_static());
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_heads_kv = get_n_heads_kv_for_tensor(ಒಳಹರಿವು);
            ಪ್ರತಿಜ್ಞಾ(n_heads_kv > 0 && ಒಳಹರಿವು_shape[3].get_length() % n_heads_kv == 0);
            ಧ್ರುವ int64_t combined_dim = ಒಳಹರಿವು_shape[3].get_length();  // n_heads_kv * head_size
            ಧ್ರುವ int64_t head_size = combined_dim / n_heads_kv;
            ಒಳಹರಿವು_shape = {ಒಳಹರಿವು_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } ಅನ್ಯಥಾ ಯದಿ (is_kv_idx(ಒಳಹರಿವು, op)) {
        // kv update index
        ಪೂರ್ಣಾಙ್ಕ len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, 1, len};

    } ಅನ್ಯಥಾ ಯದಿ (is_inp_s_copy(ಒಳಹರಿವು, op) || is_s_copy_leaf(ಒಳಹರಿವು)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        ಒಳಹರಿವು_shape = m_is_static ? ov::PartialShape{get_shape(ಒಳಹರಿವು)} : ov::PartialShape{1, 1, 1, -1};

    } ಅನ್ಯಥಾ {
        ಒಳಹರಿವು_shape = ov::PartialShape{get_shape(ಒಳಹರಿವು)};
    }
    ಯದಿ (dynamic_dim_index != -1 && m_model_is_splitted) {
        ಒಳಹರಿವು_shape[3 - dynamic_dim_index] = -1;
    }
    ಯದಿ (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು && op->src[1] == ಒಳಹರಿವು) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        ಯದಿ (m_is_static) {
            ಧ್ರುವ int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            ಒಳಹರಿವು_shape[2] = seq_active;
            ಒಳಹರಿವು_shape[3] = seq_active;
        } ಅನ್ಯಥಾ {
            ಒಳಹರಿವು_shape[2] = -1;
            ಒಳಹರಿವು_shape[3] = -1;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಒಳಹರಿವು_shape;
}

ತಾರ್ಕಿಕ GgmlOvDecoder::is_s_copy_leaf(ಧ್ರುವ ggml_tensor * tensor) ಧ್ರುವ {
    ಯದಿ (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m_cgraph->n_nodes; i++) {
        ಧ್ರುವ ggml_tensor * ಗಂಟು = m_cgraph->nodes[i];
        ಯದಿ (ಗಂಟು->op != GGML_OP_GET_ROWS || ಗಂಟು->src[0] == nullptr || ಗಂಟು->src[1] == nullptr) {
            ಅನುವರ್ತ;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ಧ್ರುವ ggml_tensor * idx = ಗಂಟು->src[1];
        ಯಾವತ್ (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        ಯದಿ (idx != tensor) {
            ಅನುವರ್ತ;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ಧ್ರುವ ggml_tensor * data = ಗಂಟು->src[0];
        ಯಾವತ್ (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        ಯದಿ (data != nullptr && is_recurrent_cache(data)) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ಶೂನ್ಯ GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_ಒಳಹರಿವು = [ಏತತ್](ಧ್ರುವ std::string & name, int64_t value, ತಾರ್ಕಿಕ force_parameter = ಅಸತ್ಯ) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    ಯದಿ (m_compute_params.attention_size != -1) {
        create_1d_ಒಳಹರಿವು("attention_size", m_compute_params.attention_size);
    }
    ಯದಿ (m_compute_params.attention_size_static != -1) {
        create_1d_ಒಳಹರಿವು("attention_size_static", m_compute_params.attention_size_static);
    }
    ಯದಿ (m_compute_params.attention_size_swa != -1) {
        create_1d_ಒಳಹರಿವು("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    ಯದಿ (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_ಒಳಹರಿವು("swa_window", m_compute_params.swa_window);
    }
    create_1d_ಒಳಹರಿವು("n_seq_active", m_compute_params.n_seq_active);
    create_1d_ಒಳಹರಿವು("seq_active_start", m_compute_params.seq_active_start);
    create_1d_ಒಳಹರಿವು("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    ಯದಿ (m_compute_params.token_len_per_seq != -1) {
        create_1d_ಒಳಹರಿವು("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    ಯದಿ (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_ಒಳಹರಿವು("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/ಸತ್ಯ);
        create_1d_ಒಳಹರಿವು("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/ಸತ್ಯ);
    }

    ಯದಿ (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_ಒಳಹರಿವು("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        ಯದಿ (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_ಒಳಹರಿವು("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/ಸತ್ಯ);
        }
    }

    ಪ್ರತಿ (ಧ್ರುವ auto & [ಗಂಟು_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_ಒಳಹರಿವು("rs_slot_begin_" + ಗಂಟು_name, writeback.slot_begin);
        ಯದಿ (!m_is_static && writeback.src_begin >= 0) {
            create_1d_ಒಳಹರಿವು("rs_src_begin_" + ಗಂಟು_name, writeback.src_begin);
        }
    }
}

ತಾರ್ಕಿಕ GgmlOvDecoder::ಗಂಟು_is_used_as_src(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ggml_tensor * ಗಂಟು = m_cgraph->nodes[ಗಂಟು_idx];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = ಗಂಟು_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_ಗಂಟು = m_cgraph->nodes[i];
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಯದಿ (other_ಗಂಟು->src[j] == ಗಂಟು) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ಶೂನ್ಯ GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        ಯದಿ (ಗಂಟು->op == GGML_OP_NONE && ಗಂಟು_is_used_as_src(i)) {
            std::string ಗಂಟು_name = get_tensor_ov_name(m_cgraph, ಗಂಟು);
            ಯದಿ (m_model_weights.find(ಗಂಟು_name) == m_model_weights.end()) {
                m_inputs[ಗಂಟು_name] = ಗಂಟು;
                m_model_inputs[ಗಂಟು_name] = {get_ov_type(ಗಂಟು),
                                             get_graph_ಒಳಹರಿವು_shape(ಗಂಟು, nullptr, m_ಗಂಟು_dynamic_dims[ಗಂಟು])};
            }
            ಅನುವರ್ತ;
        }
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ಗಂಟು->src[i];
            ಯದಿ (src == nullptr) {
                ಅನುವರ್ತ;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            ಯದಿ (src->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
                src_name = get_tensor_graph_ಒಳಹರಿವು_ov_name(ಏತತ್, m_cgraph, src, ಗಂಟು);
            }
            ಯದಿ (m_model_weights.find(src_name) != m_model_weights.end()) {
                ಅನುವರ್ತ;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            ಯದಿ (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                ಅನುವರ್ತ;
            }

            ತಾರ್ಕಿಕ is_intermediate_ಗಂಟು = ಅಸತ್ಯ;
            ಪ್ರತಿ (ಧ್ರುವ auto & ಗಂಟು_info : m_ಗಂಟು_info_list) {
                ಯದಿ (ಗಂಟು_info.ಗಂಟು == src) {
                    is_intermediate_ಗಂಟು = ಸತ್ಯ;
                    ಭಙ್ಗ;
                }
            }
            ಯದಿ (is_intermediate_ಗಂಟು) {
                ಅನುವರ್ತ;
            }
            ಯದಿ (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                ಅನುವರ್ತ;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            ಯದಿ (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                ಯದಿ (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            ಯಾವತ್ (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_ಒಳಹರಿವು_shape(ಗಂಟು, src, m_ಗಂಟು_dynamic_dims[src])};
        }
    }
}

ಶೂನ್ಯ GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_n = 0; ಗಂಟು_n < m_cgraph->n_nodes; ಗಂಟು_n++) {
        auto * cur_ಗಂಟು = m_cgraph->nodes[ಗಂಟು_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        ಯದಿ (cur_ಗಂಟು->op == GGML_OP_NONE || cur_ಗಂಟು->op == GGML_OP_VIEW || cur_ಗಂಟು->op == GGML_OP_RESHAPE) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (::is_inplace_op(cur_ಗಂಟು) && ggml_nbytes(cur_ಗಂಟು) == 0) {
            ಅನುವರ್ತ;
        }
        auto cur_ಗಂಟು_use_ಎಣಿಕೆ = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_ಗಂಟು)];
        ಯದಿ (cur_ಗಂಟು_use_ಎಣಿಕೆ == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            ಯದಿ (cur_ಗಂಟು != nullptr && ::is_inplace_op(cur_ಗಂಟು) && ggml_nbytes(cur_ಗಂಟು) > 0) {
                cur_ಗಂಟು = cur_ಗಂಟು->view_src;
            }
        } ಅನ್ಯಥಾ {
            ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_use_ಎಣಿಕೆ = 0;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * ಗಂಟು = m_cgraph->nodes[i];
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
                    ಯದಿ (ಗಂಟು->src[j] != ರಿಕ್ತ && ಗಂಟು->src[j] == cur_ಗಂಟು) {
                        ಒಳಹರಿವು_use_ಎಣಿಕೆ++;
                    }
                }
            }
            ಯದಿ (ಒಳಹರಿವು_use_ಎಣಿಕೆ == cur_ಗಂಟು_use_ಎಣಿಕೆ) {
                cur_ಗಂಟು = nullptr;
            }
        }
        ಯದಿ (cur_ಗಂಟು != nullptr) {
            std::string cur_ಗಂಟು_name = get_tensor_ov_name(m_cgraph, cur_ಗಂಟು);
            m_model_outputs[cur_ಗಂಟು_name] = cur_ಗಂಟು;
            m_model_output_names.insert(cur_ಗಂಟು_name);
        }
    }
}

ಧ್ರುವ ggml_tensor * GgmlOvDecoder::get_tensor_used_op(ಧ್ರುವ ggml_tensor * tensor) ಧ್ರುವ {
    ಯದಿ (tensor == nullptr) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m_cgraph->n_nodes; i++) {
        ಧ್ರುವ auto * ಗಂಟು = m_cgraph->nodes[i];
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಯದಿ (ಗಂಟು->src[j] == tensor) {
                ಪ್ರತ್ಯಾಗಮ ಗಂಟು;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ nullptr;
}

ಧ್ರುವ ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ಧ್ರುವ std::string & name) ಧ್ರುವ {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m_cgraph->n_nodes; i++) {
        ಧ್ರುವ auto * ಗಂಟು = m_cgraph->nodes[i];
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಧ್ರುವ auto * src = ಗಂಟು->src[j];
            ಯದಿ (src == nullptr) {
                ಭಙ್ಗ;
            }
            ಯದಿ (get_tensor_ov_name(m_cgraph, src) == name) {
                ಪ್ರತ್ಯಾಗಮ src;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() ಧ್ರುವ {
    std::map<std::string, std::string> kv_param_res_names;
    ಪ್ರತಿ (ಧ್ರುವ auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    ಪ್ರತ್ಯಾಗಮ kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mul_mat_id_expert_weight(ಧ್ರುವ ggml_tensor * ಗಂಟು, ಪೂರ್ಣಾಙ್ಕ src_index) {
    ಪ್ರತ್ಯಾಗಮ ಗಂಟು->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, ತಾರ್ಕಿಕ naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_i = 0; ಗಂಟು_i < n_nodes; ಗಂಟು_i++) {
        auto * ಗಂಟು = nodes[ಗಂಟು_i];
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ಗಂಟು->src[i];
            ಯದಿ (src == nullptr) {
                ಅನುವರ್ತ;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            ಯದಿ (is_rope_freqs_weight(base, ಗಂಟು)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            ಯದಿ (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(ಗಂಟು, i)) {
                ಯದಿ (model_weights.find(src_name) == model_weights.end()) {
                    auto weight_ಗಂಟು = create_weight_ಗಂಟು(base, naive);
                    weight_ಗಂಟು->set_friendly_name(src_name);
                    model_weights[src_name] = weight_ಗಂಟು;
                }
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
ಸ್ಥಿರ std::mutex g_nonov_weight_cache_mutex;
ಸ್ಥಿರ std::unordered_map<ಧ್ರುವ ಶೂನ್ಯ *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_i = 0; ಗಂಟು_i < cgraph->n_nodes; ಗಂಟು_i++) {
        auto * ಗಂಟು = cgraph->nodes[ಗಂಟು_i];
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ಗಂಟು->src[i];
            ಯದಿ (src == nullptr) {
                ಅನುವರ್ತ;
            }
            ಧ್ರುವ ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            ಯದಿ (is_rope_freqs_weight(base, ಗಂಟು)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            ಯದಿ (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_ಗಂಟು(ggml_tensor * tensor, ತಾರ್ಕಿಕ naive) {
    ಧ್ರುವ ತಾರ್ಕಿಕ is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    ಯದಿ (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        ಯದಿ (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            ಯದಿ (weight_extra->weight_ಗಂಟು) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                ಪ್ರತ್ಯಾಗಮ weight_extra->weight_ಗಂಟು;
            }
        } ಅನ್ಯಥಾ ಯದಿ (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            ಯದಿ (quant_extra->weight_ಗಂಟು) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                ಪ್ರತ್ಯಾಗಮ quant_extra->weight_ಗಂಟು;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    ಯದಿ (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        auto weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        auto weight_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_ಗಂಟು->set_friendly_name(tensor->name);
        ಪ್ರತ್ಯಾಗಮ weight_ಗಂಟು;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ಧ್ರುವ ತಾರ್ಕಿಕ cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    ಯದಿ (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        auto it = g_nonov_weight_cache.find(tensor->data);
        ಯದಿ (it != g_nonov_weight_cache.end()) {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    ಸ್ಥಿರ ಧ್ರುವ std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    ಯದಿ (weight_types.find(tensor->type) == weight_types.end()) {
        ಕ್ಷಿಪ std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    ಯದಿ (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        ಯದಿ (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } ಅನ್ಯಥಾ {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } ಅನ್ಯಥಾ {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_ಗಂಟು->set_friendly_name(tensor->name);
    ಯದಿ (!is_ov_buffer) {
        ಯದಿ (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            auto [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_ಗಂಟು);
            ಪ್ರತ್ಯಾಗಮ it->second;
        }
        ಪ್ರತ್ಯಾಗಮ ov_weight.weight_ಗಂಟು;
    }

    ggml_openvino_extra_base * extra;
    ಯದಿ (ov_weight.is_quantized()) {
        extra = ನೂತನ ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_ಗಂಟು);
    } ಅನ್ಯಥಾ {
        extra = ನೂತನ ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_ಗಂಟು);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    ಪ್ರತ್ಯಾಗಮ ov_weight.weight_ಗಂಟು;
}

ಶೂನ್ಯ GgmlOvDecoder::dump_cgraph(ಧ್ರುವ ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    ಯದಿ (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        ಪ್ರತ್ಯಾಗಮ;
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
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        // Get buffer type name
        ಧ್ರುವ ವರ್ಣ * buf_name = "none";
        ggml_backend_buffer_t buf = ಗಂಟು->view_src ? ಗಂಟು->view_src->buffer : ಗಂಟು->buffer;
        ಯದಿ (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ಗಂಟು->ne[0] << ", "
             << std::setw(5) << ಗಂಟು->ne[1] << ", "
             << std::setw(5) << ಗಂಟು->ne[2] << ", "
             << std::setw(5) << ಗಂಟು->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(ಗಂಟು->op) << std::right << " "
             << std::left << std::setw(45) << ಗಂಟು->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << ಗಂಟು->nb[0] << ", "
             << std::setw(5) << ಗಂಟು->nb[1] << ", "
             << std::setw(5) << ಗಂಟು->nb[2] << ", "
             << std::setw(5) << ಗಂಟು->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಯದಿ (auto* src = ಗಂಟು->src[i]) {
                // Get buffer type name for source
                ಧ್ರುವ ವರ್ಣ * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                ಯದಿ (src_buf) {
                    src_buf_name = ggml_backend_buffer_name(src_buf);
                }

                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::left << std::setw(12) << ggml_op_name(src->op) << std::right;
                file << std::left << std::setw(30) << src->name << std::right
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << std::right << std::setw(15) << src_buf_name << std::right
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * ಗಂಟು = cgraph->leafs[i];

        // Get buffer type name for leaf
        ಧ್ರುವ ವರ್ಣ * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = ಗಂಟು->view_src ? ಗಂಟು->view_src->buffer : ಗಂಟು->buffer;
        ಯದಿ (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ಗಂಟು->ne[0] << ", "
             << std::setw(5) << ಗಂಟು->ne[1] << "] "
             << std::setw(8) << ggml_op_name(ಗಂಟು->op) << " "
             << std::setw(16) << ggml_get_name(ಗಂಟು)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

ಶೂನ್ಯ print_tensor_address_map(ಧ್ರುವ ggml_cgraph * cgraph) {
    std::map<ಶೂನ್ಯ *, std::vector<std::string>> address_map;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_n = 0; ಗಂಟು_n < cgraph->n_nodes; ಗಂಟು_n++) {
        auto * ಗಂಟು = cgraph->nodes[ಗಂಟು_n];
        ಯದಿ (ಗಂಟು->data) {
            auto it = address_map.find(ಗಂಟು->data);
            ಯದಿ (it == address_map.end()) {
                address_map[ಗಂಟು->data] = std::vector<std::string>();
            }
            address_map[ಗಂಟು->data].push_back(ಗಂಟು->name);
        }
    }
    ಪ್ರತಿ (ಧ್ರುವ auto & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        ಪ್ರತಿ (ಧ್ರುವ auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ಧ್ರುವ ggml_tensor * tensor) {
    std::vector<size_t> shape;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    ಪ್ರತ್ಯಾಗಮ shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(ಧ್ರುವ ggml_tensor * tensor) {
    std::vector<size_t> stride;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    ಪ್ರತ್ಯಾಗಮ stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(ಧ್ರುವ ggml_tensor * tensor) {
    ವಿಕಲ್ಪ (tensor->type) {
    ಪಕ್ಷ GGML_TYPE_F64:
        ಪ್ರತ್ಯಾಗಮ ov::element::f64;
    ಪಕ್ಷ GGML_TYPE_F32:
        ಪ್ರತ್ಯಾಗಮ ov::element::f32;
    ಪಕ್ಷ GGML_TYPE_F16:
        ಪ್ರತ್ಯಾಗಮ ov::element::f16;
    ಪಕ್ಷ GGML_TYPE_BF16:
        ಪ್ರತ್ಯಾಗಮ ov::element::bf16;
    ಪಕ್ಷ GGML_TYPE_I8:
        ಪ್ರತ್ಯಾಗಮ ov::element::i8;
    ಪಕ್ಷ GGML_TYPE_I16:
        ಪ್ರತ್ಯಾಗಮ ov::element::i16;
    ಪಕ್ಷ GGML_TYPE_I32:
        ಪ್ರತ್ಯಾಗಮ ov::element::i32;
    ಪಕ್ಷ GGML_TYPE_I64:
        ಪ್ರತ್ಯಾಗಮ ov::element::i64;
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_ಒಳಹರಿವು_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ ov::PartialShape(get_shape(m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ get_stride(m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_ಒಳಹರಿವು_size(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಪ್ರತ್ಯಾಗಮ it->second.size();
    }
    ಪ್ರತ್ಯಾಗಮ 0;
}

size_t GgmlOvDecoder::get_view_ಒಳಹರಿವು_offset(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಯದಿ (view_index < it->second.size()) {
            ಪ್ರತ್ಯಾಗಮ it->second[view_index].second->view_offs;
        }
    }
    ಪ್ರತ್ಯಾಗಮ 0;
}

size_t GgmlOvDecoder::get_view_ಒಳಹರಿವು_src_offset(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಯದಿ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            ಯದಿ (view_tensor && view_tensor->src[0]) {
                ಪ್ರತ್ಯಾಗಮ view_tensor->src[0]->view_offs;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                         ಧ್ರುವ std::string & name,
                                                         size_t view_index) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಯದಿ (view_index < it->second.size()) {
            ಪ್ರತ್ಯಾಗಮ get_stride(it->second[view_index].second);
        }
    }
    ಪ್ರತ್ಯಾಗಮ {};
}

std::vector<size_t> GgmlOvDecoder::get_view_ಒಳಹರಿವು_src_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                             ಧ್ರುವ std::string & name,
                                                             size_t view_index) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಯದಿ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            ಯದಿ (view_tensor && view_tensor->src[0]) {
                ಪ್ರತ್ಯಾಗಮ get_stride(view_tensor->src[0]);
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ {};
}

ov::Shape GgmlOvDecoder::get_view_ಒಳಹರಿವು_ggml_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಯದಿ (view_index < it->second.size()) {
            ಪ್ರತ್ಯಾಗಮ get_shape(it->second[view_index].second);
        }
    }
    ಪ್ರತ್ಯಾಗಮ {};
}

ov::Shape GgmlOvDecoder::get_view_ಒಳಹರಿವು_src_ggml_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                       ಧ್ರುವ std::string & name,
                                                       size_t view_index) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಯದಿ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            ಯದಿ (view_tensor && view_tensor->src[0]) {
                ಪ್ರತ್ಯಾಗಮ get_shape(view_tensor->src[0]);
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ {};
}

ov::PartialShape GgmlOvDecoder::get_view_ಒಳಹರಿವು_ov_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                        ಧ್ರುವ std::string & name,
                                                        size_t view_index) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಯದಿ (view_index < it->second.size()) {
            auto * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            auto dynamic_it = m_ಗಂಟು_dynamic_dims.find(tensor);
            ಯದಿ (dynamic_it != m_ಗಂಟು_dynamic_dims.end() && dynamic_it->second != -1) {
                ಪೂರ್ಣಾಙ್ಕ dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            ಪ್ರತ್ಯಾಗಮ shape;
        }
    }
    ಪ್ರತ್ಯಾಗಮ {};
}

ov::PartialShape GgmlOvDecoder::get_view_ಒಳಹರಿವು_src_ov_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                            ಧ್ರುವ std::string & name,
                                                            size_t view_index) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಯದಿ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            ಯದಿ (view_tensor && view_tensor->src[0]) {
                auto * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                auto dynamic_it = m_ಗಂಟು_dynamic_dims.find(src_tensor);
                ಯದಿ (dynamic_it != m_ಗಂಟು_dynamic_dims.end() && dynamic_it->second != -1) {
                    ಪೂರ್ಣಾಙ್ಕ dynamic_dim_index = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
                }

                ಪ್ರತ್ಯಾಗಮ shape;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ {};
}

std::string GgmlOvDecoder::get_view_ಒಳಹರಿವು_name(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಯದಿ (view_index < it->second.size()) {
            ಪ್ರತ್ಯಾಗಮ it->second[view_index].first;
        }
    }
    ಪ್ರತ್ಯಾಗಮ "";
}

std::string GgmlOvDecoder::get_view_ಒಳಹರಿವು_src_name(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ {
    auto it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಯದಿ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಯದಿ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            ಯದಿ (view_tensor && view_tensor->src[0]) {
                ಪ್ರತ್ಯಾಗಮ get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ "";
}

ov::element::Type GgmlOvDecoder::get_ಒಳಹರಿವು_type(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ get_ov_type(m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs.at(name));
}

size_t GgmlOvDecoder::get_ಒಳಹರಿವು_size() ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ m_model_inputs.size();
}

size_t GgmlOvDecoder::get_ಒಳಹರಿವು_size(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_ಒಳಹರಿವು_names(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    auto * ggml_tensor = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು;
    ಪ್ರತ್ಯಾಗಮ ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ get_ov_type(m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    auto * ggml_tensor = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು;
    ಪ್ರತ್ಯಾಗಮ get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    auto * ಗಂಟು = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು;
    ಯದಿ (ಗಂಟು->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = ಗಂಟು_idx + 1; i < m_cgraph->n_nodes; i++) {
            auto * consumer = m_cgraph->nodes[i];
            ಯದಿ (consumer->op != GGML_OP_VIEW || consumer->src[0] != ಗಂಟು) {
                ಅನುವರ್ತ;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            auto & name = consumer->view_offs == 0 ? attn_name : state_name;
            ಯದಿ (!name.empty()) {
                ಪ್ರತ್ಯಾಗಮ {m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        ಯದಿ (!attn_name.empty() && !state_name.empty()) {
            ಪ್ರತ್ಯಾಗಮ {attn_name, state_name};
        }
    }
    ಪ್ರತ್ಯಾಗಮ {m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    auto * ಗಂಟು = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು;
    ಯದಿ (!::is_inplace_op(ಗಂಟು) || ಗಂಟು->view_src == nullptr || ggml_nbytes(ಗಂಟು) == 0) {
        ಪ್ರತ್ಯಾಗಮ "";
    }
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ op_case = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_op_case;
    ಯದಿ (ಗಂಟು->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        ಪ್ರತ್ಯಾಗಮ "";
    }
    ಪ್ರತ್ಯಾಗಮ get_tensor_ov_name(m_cgraph, ಗಂಟು->view_src);
}

ತಾರ್ಕಿಕ GgmlOvDecoder::is_view_like_alias_of(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & view_src_name) ಧ್ರುವ {
    auto * ಗಂಟು = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು;
    ಯದಿ (ಗಂಟು->view_src == nullptr || get_tensor_ov_name(m_cgraph, ಗಂಟು->view_src) != view_src_name) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಗಂಟು->op == GGML_OP_RESHAPE || ಗಂಟು->op == GGML_OP_VIEW;
}

ಧ್ರುವ std::string & GgmlOvDecoder::get_op_name() ಧ್ರುವ {
    ಸ್ಥಿರ ಧ್ರುವ std::string unknown_name = "UNKNOWN_OP_NAME";
    ಪ್ರತ್ಯಾಗಮ unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    auto it = m_ಗಂಟು_dynamic_dims.find(m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು);
    ಯದಿ (it == m_ಗಂಟು_dynamic_dims.end()) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }
    ಪ್ರತ್ಯಾಗಮ it->second;
}

ಧ್ರುವ std::string & GgmlOvDecoder::get_op_name(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_name;
}

int32_t * GgmlOvDecoder::get_ಒಳಹರಿವು_op_params(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು->op_params;
}

size_t GgmlOvDecoder::get_output_op_offset(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು->view_offs;
}

ಶೂನ್ಯ GgmlOvDecoder::visit_subgraph(std::function<ಶೂನ್ಯ(std::shared_ptr<GgmlDecoder>, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx)> ಗಂಟು_visitor) ಧ್ರುವ {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx = 0; ಗಂಟು_idx < m_cgraph->n_nodes; ಗಂಟು_idx++) {
        ಯದಿ (m_cgraph->nodes[ಗಂಟು_idx]->op == GGML_OP_NONE) {
            ಅನುವರ್ತ;
        }
        ಗಂಟು_visitor(std::make_shared<GgmlOvDecoder>(*ಏತತ್), ಗಂಟು_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ಧ್ರುವ ggml_tensor * ಗಂಟು) {
    ವಿಕಲ್ಪ (ಗಂಟು->op) {
    ಪಕ್ಷ GGML_OP_UNARY:
        ಪ್ರತ್ಯಾಗಮ std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(ಗಂಟು));
    ಪಕ್ಷ GGML_OP_GLU:
        ಪ್ರತ್ಯಾಗಮ std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(ಗಂಟು));
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ std::string("GGML_OP_") + ggml_op_name(ಗಂಟು->op);
    }
}

ಧ್ರುವ std::string & GgmlOvDecoder::get_op_type(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_op_type;
}

ಧ್ರುವ std::string & GgmlOvDecoder::get_op_type() ಧ್ರುವ {
    ಸ್ಥಿರ ಧ್ರುವ std::string unknown_op = "UNKNOWN_GGML_OP";
    ಪ್ರತ್ಯಾಗಮ unknown_op;
}

ಶೂನ್ಯ GgmlOvDecoder::compute_ಗಂಟು_dynamic_dims() {
    auto visit_ಗಂಟು = [&](auto && self, ggml_tensor * ಗಂಟು) -> ಶೂನ್ಯ {
        ಯದಿ (!ಗಂಟು) {
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಯದಿ (ಗಂಟು->op == GGML_OP_CPY) {
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
        }

        ಯದಿ (m_ಗಂಟು_dynamic_dims.count(ಗಂಟು)) {
            ಪ್ರತ್ಯಾಗಮ;
        }
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = ಗಂಟು->src[i];
            ಯದಿ (src == nullptr) {
                ಅನುವರ್ತ;
            }
            ಸಂರಚನಾ ggml_tensor * ಬೇರು_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            ಯದಿ (ಬೇರು_src) {
                ಯದಿ (is_inp_tok(ಬೇರು_src, ಗಂಟು) || is_inp_pos(ಬೇರು_src, ಗಂಟು) || is_output_idx(ಬೇರು_src, ಗಂಟು)) {
                    m_ಗಂಟು_dynamic_dims[ಬೇರು_src] = 0;
                    m_ಗಂಟು_dynamic_dims[src] = m_ಗಂಟು_dynamic_dims[ಬೇರು_src];
                    ಅನುವರ್ತ;
                }
                self(self, ಬೇರು_src);
                m_ಗಂಟು_dynamic_dims[src] = m_ಗಂಟು_dynamic_dims[ಬೇರು_src];
            } ಅನ್ಯಥಾ {
                ಯದಿ (is_inp_tok(src, ಗಂಟು) || is_inp_pos(src, ಗಂಟು) || is_output_idx(src, ಗಂಟು)) {
                    m_ಗಂಟು_dynamic_dims[src] = 0;
                    ಅನುವರ್ತ;
                }
                ಯದಿ (is_inp_scale_rows(src, ಗಂಟು)) {
                    m_ಗಂಟು_dynamic_dims[src] = 1;
                    ಅನುವರ್ತ;
                }
                ಯದಿ (ಗಂಟು->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_ಗಂಟು_dynamic_dims[src] = 1;
                    ಅನುವರ್ತ;
                }
                self(self, src);
            }
        }
        ವಿಕಲ್ಪ (ಗಂಟು->op) {
        ಪಕ್ಷ GGML_OP_NONE:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_GET_ROWS:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]] != -1) {
                auto dynamic_dim_idx = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]];
                ಯದಿ (dynamic_dim_idx == 0) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 1;
                } ಅನ್ಯಥಾ {
                    auto dynamic_dim_stride = ಗಂಟು->src[1]->nb[dynamic_dim_idx] / ggml_type_size(ಗಂಟು->src[1]->type) *
                                              ggml_type_size(ಗಂಟು->src[0]->type);
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
                        ಯದಿ (dynamic_dim_stride == ಗಂಟು->src[0]->nb[i]) {
                            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                            ಭಙ್ಗ;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_MUL:
        ಪಕ್ಷ GGML_OP_MUL_MAT:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
            }
            ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]] != -1) {
                m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]];
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_PERMUTE:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                auto dynamic_dim_idx = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
                    ಯದಿ (ಗಂಟು->op_params[i] == dynamic_dim_idx) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                        ಭಙ್ಗ;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                ಯದಿ (ಗಂಟು->src[0]->op == GGML_OP_NONE) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                    ಭಙ್ಗ;
                }
                auto dynamic_dim_idx = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                auto dynamic_dim_ಮೌಲ್ಯ = ಗಂಟು->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    ಗಂಟು->src[0]->nb[dynamic_dim_idx] / ggml_type_size(ಗಂಟು->src[0]->type) * ggml_type_size(ಗಂಟು->type);
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
                    ಯದಿ (ಗಂಟು->nb[i] == dynamic_dim_stride) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                        ಭಙ್ಗ;
                    }
                }
                ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು] != -1 && dynamic_dim_ಮೌಲ್ಯ != ಗಂಟು->ne[m_ಗಂಟು_dynamic_dims[ಗಂಟು]]) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    ಯದಿ (ggml_nelements(ಗಂಟು) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      ಗಂಟು->name, ಗಂಟು->src[0]->name);
                    }
                }
            }
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ GGML_OP_TRANSPOSE:
        ಪಕ್ಷ GGML_OP_RESHAPE: {
            ಯದಿ (is_same_shape(ಗಂಟು->src[0], ಗಂಟು)) {
                m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                ಭಙ್ಗ;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                auto dynamic_dim_idx = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                auto dynamic_dim_stride = ಗಂಟು->src[0]->nb[dynamic_dim_idx];
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
                    ಯದಿ (ಗಂಟು->nb[i] == dynamic_dim_stride && ಗಂಟು->ne[i] == ಗಂಟು->src[0]->ne[dynamic_dim_idx]) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                        ಭಙ್ಗ;
                    }
                }
                ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", ಗಂಟು->name);
                }
            }
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr ಪೂರ್ಣಾಙ್ಕ q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                auto q_dynamic_dim = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                m_ಗಂಟು_dynamic_dims[ಗಂಟು] = q_to_out[q_dynamic_dim];
            }
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ GGML_OP_CONT:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                auto dynamic_dim_idx = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                ಯದಿ (ggml_are_same_shape(ಗಂಟು, ಗಂಟು->src[0])) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = dynamic_dim_idx;
                } ಅನ್ಯಥಾ {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(ಗಂಟು->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (ಗಂಟು->src[0]->ne[0] / ggml_blck_size(ಗಂಟು->src[0]->type));
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * ಗಂಟು->src[0]->ne[i - 1];
                    }

                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(ಗಂಟು->src[0]->type) *
                                              ggml_type_size(ಗಂಟು->type);
                    ಪೂರ್ಣಾಙ್ಕ matched_dim_ಎಣಿಕೆ = 0;
                    ಪೂರ್ಣಾಙ್ಕ first_matched_dim = -1;
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
                        ಯದಿ (ಗಂಟು->nb[i] == dynamic_dim_stride && ಗಂಟು->ne[i] == ಗಂಟು->src[0]->ne[dynamic_dim_idx]) {
                            ಯದಿ (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                            matched_dim_ಎಣಿಕೆ++;
                        }
                    }
                    ಯದಿ (matched_dim_ಎಣಿಕೆ > 1 && ಗಂಟು->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = first_matched_dim;
                    } ಅನ್ಯಥಾ ಯದಿ (matched_dim_ಎಣಿಕೆ != 1) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      ಗಂಟು->name, ಗಂಟು->src[0]->name);
                    }
                }
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CONCAT:
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
                ಯದಿ (ಗಂಟು->src[0]->ne[i] != ಗಂಟು->ne[i]) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                    ಭಙ್ಗ;
                }
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SSM_CONV:
        ಪಕ್ಷ GGML_OP_GATED_DELTA_NET:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 1;
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_RMS_NORM:
        ಪಕ್ಷ GGML_OP_L2_NORM:
        ಪಕ್ಷ GGML_OP_NORM:
        ಪಕ್ಷ GGML_OP_ADD:
        ಪಕ್ಷ GGML_OP_SUB:
        ಪಕ್ಷ GGML_OP_GLU:
        ಪಕ್ಷ GGML_OP_ROPE:
        ಪಕ್ಷ GGML_OP_SCALE:
        ಪಕ್ಷ GGML_OP_SOFT_MAX:
        ಪಕ್ಷ GGML_OP_ARGSORT:
        ಪಕ್ಷ GGML_OP_ADD_ID:
        ಪಕ್ಷ GGML_OP_UNARY:
        ಪಕ್ಷ GGML_OP_CUMSUM:
        ಪಕ್ಷ GGML_OP_FILL:
        ಪಕ್ಷ GGML_OP_SET:
        ಪಕ್ಷ GGML_OP_DIAG:
        ಪಕ್ಷ GGML_OP_TRI:
        ಪಕ್ಷ GGML_OP_REPEAT:
        ಪಕ್ಷ GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        ಪಕ್ಷ GGML_OP_DIV:
        ಪಕ್ಷ GGML_OP_CLAMP:
        ಪಕ್ಷ GGML_OP_PAD:
        ಪಕ್ಷ GGML_OP_UPSCALE:
        ಪಕ್ಷ GGML_OP_SIN:
        ಪಕ್ಷ GGML_OP_COS:
        ಪಕ್ಷ GGML_OP_LOG:
        ಪಕ್ಷ GGML_OP_ROLL:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] =
                (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] == 0) ? -1 : m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
        ಪಕ್ಷ GGML_OP_SOLVE_TRI:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]];
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CPY:
        ಪಕ್ಷ GGML_OP_SET_ROWS:
        ಪಕ್ಷ GGML_OP_SUM:
        ಪಕ್ಷ GGML_OP_MEAN:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_IM2COL: {
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]] != -1) {
                ಧ್ರುವ ತಾರ್ಕಿಕ is_2D = ಗಂಟು->op_params[6] == 1;
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src_dyn = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]];
                ಯದಿ (is_2D) {
                    ಯದಿ (src_dyn == 0) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 1;  // IW -> OW
                    } ಅನ್ಯಥಾ ಯದಿ (src_dyn == 1) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 2;  // IH -> OH
                    } ಅನ್ಯಥಾ ಯದಿ (src_dyn == 3) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 3;  // N  -> N
                    }
                } ಅನ್ಯಥಾ {
                    ಯದಿ (src_dyn == 0) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 1;  // IW -> OW
                    } ಅನ್ಯಥಾ ಯದಿ (src_dyn == 2) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು] != -1) {
                    OPENVINO_ASSERT(ಗಂಟು->src[1]->ne[src_dyn] == ಗಂಟು->ne[m_ಗಂಟು_dynamic_dims[ಗಂಟು]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(ಗಂಟು->name) +
                                        " and its src[1]: " + std::string(ಗಂಟು->src[1]->name));
                }
            }
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ GGML_OP_IM2COL_3D: {
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]] != -1) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src_dyn = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]];
                ಯದಿ (src_dyn == 0) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 1;  // IW -> OW
                } ಅನ್ಯಥಾ ಯದಿ (src_dyn == 1) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 2;  // IH -> OH
                } ಅನ್ಯಥಾ ಯದಿ (src_dyn == 3) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 3;  // N  -> N
                }
                ಯದಿ (m_ಗಂಟು_dynamic_dims[ಗಂಟು] != -1) {
                    OPENVINO_ASSERT(ಗಂಟು->src[1]->ne[src_dyn] == ಗಂಟು->ne[m_ಗಂಟು_dynamic_dims[ಗಂಟು]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(ಗಂಟು->name) +
                                        " and its src[1]: " + std::string(ಗಂಟು->src[1]->name));
                }
            }
            ಭಙ್ಗ;
        }
        ಸ್ವಭಾವ:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(ಗಂಟು->op), ಗಂಟು->name);
            ಭಙ್ಗ;
        }
    };

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = m_cgraph->nodes[i];
        visit_ಗಂಟು(visit_ಗಂಟು, ಗಂಟು);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    ಯದಿ (0) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * ಗಂಟು = m_cgraph->nodes[i];
            ಪೂರ್ಣಾಙ್ಕ dynamic_dim = m_ಗಂಟು_dynamic_dims[ಗಂಟು];
            std::cout << "[" << i << "] " << "node_name: " << ಗಂಟು->name << " op: " << ggml_op_name(ಗಂಟು->op)
                      << " shape: [";
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; j++) {
                ಯದಿ (j == dynamic_dim) {
                    std::cout << "*";
                } ಅನ್ಯಥಾ {
                    std::cout << ಗಂಟು->ne[j];
                }
                ಯದಿ (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = ಗಂಟು->src[j];
                ಯದಿ (src == nullptr) {
                    ಅನುವರ್ತ;
                }
                ಪೂರ್ಣಾಙ್ಕ src_dynamic_dim = m_ಗಂಟು_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < 4; k++) {
                    ಯದಿ (k == src_dynamic_dim) {
                        std::cout << "*";
                    } ಅನ್ಯಥಾ {
                        std::cout << src->ne[k];
                    }
                    ಯದಿ (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
