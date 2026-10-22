#അന്തര്ഭാവ "ggml-decoder.h"

#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-openvino-extra.h"
#അന്തര്ഭാവ "ggml-openvino.h"
#അന്തര്ഭാവ "ggml-quants.h"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "utils.h"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cassert>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <fstream>
#അന്തര്ഭാവ <iomanip>
#അന്തര്ഭാവ <map>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <mutex>
#അന്തര്ഭാവ <openvino/core/dimension.hpp>
#അന്തര്ഭാവ <openvino/core/except.hpp>
#അന്തര്ഭാവ <openvino/core/node.hpp>
#അന്തര്ഭാവ <openvino/core/partial_shape.hpp>
#അന്തര്ഭാവ <openvino/core/type/bfloat16.hpp>
#അന്തര്ഭാവ <openvino/core/type/element_type.hpp>
#അന്തര്ഭാവ <openvino/core/type/float16.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/runtime/tensor.hpp>
#അന്തര്ഭാവ <ostream>
#അന്തര്ഭാവ <set>
#അന്തര്ഭാവ <stdexcept>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             തർക്കം is_static,
                             തർക്കം is_stateful,
                             തർക്കം model_is_splitted,
                             തർക്കം is_prefill,
                             പൂർണ്ണസംഖ്യ prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(അസത്യം),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    സ്ഥിതം തർക്കം printed_address_map = അസത്യം;
    എങ്കിൽ (!printed_address_map) {
        എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = സത്യം;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_നിവേശം_output();
    compute_കണ്ണി_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    വേണ്ടി (പൂർണ്ണസംഖ്യ കണ്ണി_n = 0; കണ്ണി_n < cgraph->n_nodes; കണ്ണി_n++) {
        m_കണ്ണി_info_list[കണ്ണി_n].കണ്ണി_op_case = compute_op_case(m_കണ്ണി_info_list[കണ്ണി_n].കണ്ണി);
        m_കണ്ണി_info_list[കണ്ണി_n].കണ്ണി_op_type = compute_op_type(m_കണ്ണി_info_list[കണ്ണി_n].കണ്ണി);
    }

    add_extra_inputs();
}

ശൂന്യം GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_കണ്ണി_info_list.clear();
    set_നിവേശം_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
തർക്കം GgmlOvDecoder::is_bound_to(സ്ഥിരം ggml_cgraph * cgraph) സ്ഥിരം {
    തിരികെ m_cgraph == cgraph && cgraph->n_nodes > 0 && m_കണ്ണി_info_list.size() == (size_t) cgraph->n_nodes &&
           m_കണ്ണി_info_list.front().കണ്ണി == cgraph->nodes[0] &&
           m_കണ്ണി_info_list.back().കണ്ണി == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = സത്യം;
    set_നിവേശം_output();
    compute_model_inputs();
    compute_model_outputs();
    വേണ്ടി (പൂർണ്ണസംഖ്യ കണ്ണി_n = 0; കണ്ണി_n < cgraph->n_nodes; കണ്ണി_n++) {
        m_കണ്ണി_info_list[കണ്ണി_n].കണ്ണി_op_case = compute_op_case(m_കണ്ണി_info_list[കണ്ണി_n].കണ്ണി);
        m_കണ്ണി_info_list[കണ്ണി_n].കണ്ണി_op_type = compute_op_type(m_കണ്ണി_info_list[കണ്ണി_n].കണ്ണി);
    }
}

നാമാകാശ {
തർക്കം is_inplace_op(സ്ഥിരം ggml_tensor * കണ്ണി) {
    തിരികെ കണ്ണി->op == GGML_OP_SET_ROWS || കണ്ണി->op == GGML_OP_CPY || (കണ്ണി->op == GGML_OP_SCALE && കണ്ണി->view_src);
}

തർക്കം is_same_shape(സ്ഥിരം ggml_tensor * a, സ്ഥിരം ggml_tensor * b) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
        എങ്കിൽ (a->ne[i] != b->ne[i]) {
            തിരികെ അസത്യം;
        }
    }
    തിരികെ സത്യം;
}

തർക്കം is_conv_states_all_tensor(സ്ഥിരം ggml_tensor * tensor) {
    തിരികെ tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

തർക്കം is_full_single_slot_writeback(സ്ഥിരം ggml_tensor * കണ്ണി) {
    തിരികെ കണ്ണി->view_src != nullptr && കണ്ണി->view_src->ne[1] == 1 && കണ്ണി->src[1] != nullptr &&
           കണ്ണി->src[1]->op == GGML_OP_VIEW && കണ്ണി->src[1]->view_src == കണ്ണി->view_src &&
           കണ്ണി->src[1]->view_offs == 0 && ggml_nbytes(കണ്ണി->src[1]) == ggml_nbytes(കണ്ണി->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
തർക്കം is_moe_expert_sum_add(സ്ഥിരം ggml_tensor * കണ്ണി) {
    std::vector<സ്ഥിരം ggml_tensor *> leaves;
    സ്ഥിരം ggml_tensor * cur = കണ്ണി;
    ആകുമ്പോൾ (cur->op == GGML_OP_ADD) {
        എങ്കിൽ (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            തിരികെ അസത്യം;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    സ്ഥിരം ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    വേണ്ടി (സ്ഥിരം ggml_tensor * leaf : leaves) {
        എങ്കിൽ (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            തിരികെ അസത്യം;
        }
        സ്ഥിരം ggml_tensor * leaf_base = leaf->src[0];
        എങ്കിൽ (base == nullptr) {
            base = leaf_base;
        } അല്ലെങ്കിൽ എങ്കിൽ (leaf_base != base) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            തിരികെ അസത്യം;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        എങ്കിൽ (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            തിരികെ അസത്യം;
        }
    }

    തിരികെ base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(സ്ഥിരം ggml_cgraph * cgraph, സ്ഥിരം ggml_tensor * tensor) {
    എങ്കിൽ (tensor == nullptr) {
        തിരികെ "";
    }
    എങ്കിൽ ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        സ്ഥിരം auto * കണ്ണി = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        എങ്കിൽ (കണ്ണി != cgraph->nodes + cgraph->n_nodes) {
            തിരികെ std::string(tensor->name) + "#n" + std::to_string(കണ്ണി - cgraph->nodes);
        }
        സ്ഥിരം auto * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        എങ്കിൽ (leaf != cgraph->leafs + cgraph->n_leafs) {
            തിരികെ std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    തിരികെ tensor->name;
}

സ്ഥിതം std::string get_tensor_ov_name(സ്ഥിരം ggml_cgraph * cgraph, സ്ഥിരം ggml_tensor * tensor) {
    തിരികെ GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

സ്ഥിതം std::string get_tensor_graph_നിവേശം_ov_name(സ്ഥിരം GgmlOvDecoder * decoder,
                                                  സ്ഥിരം ggml_cgraph * cgraph,
                                                  സ്ഥിരം ggml_tensor * tensor,
                                                  സ്ഥിരം ggml_tensor * op) {
    എങ്കിൽ (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        തിരികെ "inp_pos";
    }
    എങ്കിൽ (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        തിരികെ "embd";
    }
    എങ്കിൽ (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        സ്ഥിരം തർക്കം is_swa = decoder->is_swa_mask(tensor);
        എങ്കിൽ (decoder->is_stateful()) {
            തിരികെ is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        എങ്കിൽ (is_swa) {
            തിരികെ get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    തിരികെ get_tensor_ov_name(cgraph, tensor);
}

ശൂന്യം GgmlOvDecoder::set_നിവേശം_output() {
    വേണ്ടി (പൂർണ്ണസംഖ്യ കണ്ണി_n = 0; കണ്ണി_n < m_cgraph->n_nodes; കണ്ണി_n++) {
        auto * കണ്ണി = m_cgraph->nodes[കണ്ണി_n];

        കണ്ണി_Info current_കണ്ണി_info;
        auto കണ്ണി_name = get_tensor_ov_name(m_cgraph, കണ്ണി);

        current_കണ്ണി_info.കണ്ണി = കണ്ണി;
        current_കണ്ണി_info.കണ്ണി_name = കണ്ണി_name;
        current_കണ്ണി_info.കണ്ണി_op_case = 0;
        current_കണ്ണി_info.data_addr = കണ്ണി->data;

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = കണ്ണി->src[i];
            എങ്കിൽ (src == nullptr) {
                തുടരുക;
            }
            auto src_name = get_tensor_ov_name(m_cgraph, src);
            എങ്കിൽ (src->flags & GGML_TENSOR_FLAG_നിവേശം) {
                src_name = get_tensor_graph_നിവേശം_ov_name(ഇത്, m_cgraph, src, കണ്ണി);
            }
            current_കണ്ണി_info.കണ്ണി_inputs[src_name] = src;
            current_കണ്ണി_info.കണ്ണി_inputs_names.push_back(src_name);

            എങ്കിൽ (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_കണ്ണി_info.കണ്ണി_inputs_views[src_name])> view_chain;
                auto * current = src;

                ആകുമ്പോൾ (current != nullptr) {
                    auto current_name = get_tensor_ov_name(m_cgraph, current);
                    എങ്കിൽ (current->flags & GGML_TENSOR_FLAG_നിവേശം) {
                        current_name = get_tensor_graph_നിവേശം_ov_name(ഇത്, m_cgraph, current, കണ്ണി);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    എങ്കിൽ (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } അല്ലെങ്കിൽ {
                        നിർത്തുക;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_കണ്ണി_info.കണ്ണി_inputs_views[src_name] = view_chain;
            }
        }

        m_കണ്ണി_info_list.push_back(current_കണ്ണി_info);
    }
}

പൂർണ്ണസംഖ്യ GgmlOvDecoder::compute_op_case(സ്ഥിരം ggml_tensor * കണ്ണി) സ്ഥിരം {
    പൂർണ്ണസംഖ്യ op_case = 0;
    തിരഞ്ഞെടുപ്പ് (കണ്ണി->op) {
    അവസ്ഥ GGML_OP_RESHAPE: {
        എങ്കിൽ (m_naive) {
            നിർത്തുക;
        }
        auto name = std::string(കണ്ണി->name);
        auto * src = കണ്ണി->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        തർക്കം recurrent_sequence = അസത്യം;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            സ്ഥിരം auto * consumer = m_cgraph->nodes[i];
            എങ്കിൽ (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == കണ്ണി) {
                തിരികെ 1;
            } അല്ലെങ്കിൽ എങ്കിൽ (consumer->op == GGML_OP_SSM_CONV) {
                സ്ഥിരം auto * concat = consumer->src[0];
                എങ്കിൽ (concat->op == GGML_OP_CONCAT) {
                    സ്ഥിരം auto * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == കണ്ണി;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                സ്ഥിരം auto * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == കണ്ണി;
            }
        }
        എങ്കിൽ (recurrent_sequence && കണ്ണി->ne[0] == src->ne[0] && കണ്ണി->ne[3] == 1) {
            തിരികെ 6;
        }
        എങ്കിൽ (കണ്ണി->ne[0] == src->ne[0] && കണ്ണി->ne[2] == 1 && കണ്ണി->ne[3] == 1) {
            തിരികെ 5;
        }
        എങ്കിൽ (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == കണ്ണി->ne[0] && src->src[0]->ne[1] == കണ്ണി->ne[1]) {
            op_case = 4;
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->ne[0] * കണ്ണി->ne[1] == src->ne[0]) {
            op_case = 1;
        } അല്ലെങ്കിൽ എങ്കിൽ (src->ne[0] * src->ne[1] == കണ്ണി->ne[0]) {
            op_case = 2;
            എങ്കിൽ (src->ne[2] * src->ne[3] == കണ്ണി->ne[1]) {
                op_case = 5;
            }
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == കണ്ണി->ne[1]) {
            op_case = 3;
        } അല്ലെങ്കിൽ എങ്കിൽ (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } അല്ലെങ്കിൽ എങ്കിൽ (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } അല്ലെങ്കിൽ എങ്കിൽ (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        എങ്കിൽ (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            സ്ഥിരം auto * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            തർക്കം recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                സ്ഥിരം auto * consumer = m_cgraph->nodes[i];
                എങ്കിൽ (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j : {3, 4}) {
                        സ്ഥിരം auto * gate = consumer->src[j];
                        എങ്കിൽ (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == കണ്ണി;
                    }
                } അല്ലെങ്കിൽ എങ്കിൽ (consumer->op == GGML_OP_MUL) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 2; ++j) {
                        സ്ഥിരം auto * gate = consumer->src[j];
                        സ്ഥിരം auto * norm = consumer->src[1 - j];
                        എങ്കിൽ (gate->op != GGML_OP_UNARY || gate->src[0] != കണ്ണി) {
                            തുടരുക;
                        }
                        എങ്കിൽ (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            എങ്കിൽ (recurrent) {
                op_case = 9;
            }
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_PERMUTE: {
        എങ്കിൽ (കണ്ണി->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(കണ്ണി->view_src->name);
            പൂർണ്ണസംഖ്യ layer = extract_layer_from_name(src_name).value();
            എങ്കിൽ (ggml_is_contiguous(കണ്ണി->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                എങ്കിൽ (!is_swa_layer(layer)) {
                    op_case = 3;
                } അല്ലെങ്കിൽ {
                    op_case = 4;
                }
            } അല്ലെങ്കിൽ {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                എങ്കിൽ (!is_swa_layer(layer)) {
                    op_case = 5;
                } അല്ലെങ്കിൽ {
                    op_case = 6;
                }
            }
        } അല്ലെങ്കിൽ {
            // rope'ed query tensor
            op_case = 2;
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_MUL_MAT: {
        എങ്കിൽ (കണ്ണി->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_GET_ROWS: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        എങ്കിൽ (കണ്ണി->src[0]->op == GGML_OP_RESHAPE && കണ്ണി->src[0]->src[0] != nullptr &&
            is_recurrent_cache(കണ്ണി->src[0]->src[0])) {
            സ്ഥിരം തർക്കം single_slot = കണ്ണി->src[0]->src[0]->ne[1] == 1;
            എങ്കിൽ (കണ്ണി->src[1]->op == GGML_OP_VIEW) {
                op_case = (കണ്ണി->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } അല്ലെങ്കിൽ എങ്കിൽ (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_ROPE: {
        സ്ഥിരം പൂർണ്ണസംഖ്യ mode = കണ്ണി->op_params[2];
        തിരഞ്ഞെടുപ്പ് (mode) {
        അവസ്ഥ GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            നിർത്തുക;
        }
        അവസ്ഥ GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            നിർത്തുക;
        }
        അവസ്ഥ GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            നിർത്തുക;
        }
        അവസ്ഥ GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            നിർത്തുക;
        }
        സ്ഥിരസ്ഥിതി:
            op_case = 0;
            നിർത്തുക;
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_VIEW: {
        എങ്കിൽ (!m_model_params.has_rs_rollback && കണ്ണി->src[0] != nullptr &&
            കണ്ണി->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            നിർത്തുക;
        }
        എങ്കിൽ (m_is_static && കണ്ണി->src[0] != nullptr &&
            (കണ്ണി->src[0]->op == GGML_OP_GATED_DELTA_NET || കണ്ണി->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            നിർത്തുക;
        }
        എങ്കിൽ (കണ്ണി->src[0]->op == GGML_OP_VIEW) {
            auto * src = കണ്ണി->src[0];
            എങ്കിൽ (ggml_nelements(കണ്ണി) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            എങ്കിൽ (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            auto * src = കണ്ണി->src[0];
            എങ്കിൽ (ggml_nelements(കണ്ണി) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                എങ്കിൽ (കണ്ണി->ne[0] == src->ne[0] && കണ്ണി->ne[1] == src->ne[2] && കണ്ണി->ne[2] == 1 &&
                    കണ്ണി->nb[0] == src->nb[0] && കണ്ണി->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    നിർത്തുക;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                പൂർണ്ണസംഖ്യ diff_എണ്ണം = 0;
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
                    എങ്കിൽ (കണ്ണി->ne[i] != src->ne[i]) {
                        diff_എണ്ണം++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    എങ്കിൽ (കണ്ണി->ne[i] > src->ne[i]) {
                        തിരികെ 0;
                    }
                }
                എങ്കിൽ (diff_എണ്ണം >= 1) {
                    op_case = 0;
                }
            }
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_RMS_NORM: {
        എങ്കിൽ (കണ്ണി->src[0]->op == GGML_OP_VIEW) {
            എങ്കിൽ (is_same_shape(കണ്ണി->src[0]->src[0], കണ്ണി->src[0])) {
                op_case = 1;
            } അല്ലെങ്കിൽ എങ്കിൽ (!m_model_params.has_rs_rollback &&
                       കണ്ണി->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_POOL_2D: {
        സ്ഥിരം ggml_op_pool pool_mode = static_cast<ggml_op_pool>(കണ്ണി->op_params[0]);
        തിരഞ്ഞെടുപ്പ് (pool_mode) {
        അവസ്ഥ GGML_OP_POOL_MAX: {
            op_case = 1;
            നിർത്തുക;
        }
        അവസ്ഥ GGML_OP_POOL_AVG: {
            op_case = 2;
            നിർത്തുക;
        }
        സ്ഥിരസ്ഥിതി:
            op_case = 0;
            നിർത്തുക;
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_UPSCALE: {
        സ്ഥിരം int32_t mode_flags = കണ്ണി->op_params[0];
        സ്ഥിരം ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        തിരഞ്ഞെടുപ്പ് (scale_mode) {
        അവസ്ഥ GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            നിർത്തുക;
        }
        അവസ്ഥ GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            നിർത്തുക;
        }
        അവസ്ഥ GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            നിർത്തുക;
        }
        സ്ഥിരസ്ഥിതി:
            op_case = 0;
            നിർത്തുക;
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_CPY: {
        എങ്കിൽ (കണ്ണി->src[0]->op == GGML_OP_VIEW) {
            എങ്കിൽ (കണ്ണി->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                എങ്കിൽ (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(കണ്ണി) ? 7 : 10;
                } അല്ലെങ്കിൽ {
                    op_case = 1;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (GgmlOvDecoder::is_conv_state_writeback(കണ്ണി)) {
                op_case = is_full_single_slot_writeback(കണ്ണി) ? 8 : 2;
                നിർത്തുക;
            } അല്ലെങ്കിൽ എങ്കിൽ (is_conv_states_all_tensor(കണ്ണി->view_src) && കണ്ണി->src[1] != nullptr &&
                       കണ്ണി->src[1]->op == GGML_OP_VIEW && കണ്ണി->src[1]->view_src == കണ്ണി->view_src) {
                op_case = 4;
                നിർത്തുക;
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->src[0]->src[0]->op == GGML_OP_GET_ROWS && കണ്ണി->src[1] != nullptr &&
                       കണ്ണി->src[1]->op == GGML_OP_VIEW && കണ്ണി->src[1]->view_src != nullptr &&
                       is_recurrent_cache(കണ്ണി->src[1]->view_src) && കണ്ണി->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->src[0]->op == GGML_OP_GET_ROWS && കണ്ണി->src[1] != nullptr &&
                   കണ്ണി->src[1]->op == GGML_OP_VIEW && കണ്ണി->src[1]->view_src != nullptr &&
                   is_recurrent_cache(കണ്ണി->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = കണ്ണി->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->src[1] != nullptr && കണ്ണി->src[1]->op == GGML_OP_VIEW && കണ്ണി->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            സ്ഥിരം ggml_tensor * kv_buf = കണ്ണി->src[1]->view_src;
            എങ്കിൽ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m_cgraph->n_nodes; i++) {
                    സ്ഥിരം ggml_tensor * n = m_cgraph->nodes[i];
                    എങ്കിൽ (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        തുടരുക;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    എങ്കിൽ ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        എങ്കിൽ (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        നിർത്തുക;
                    }
                }
            }
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_ADD: {
        എങ്കിൽ (is_moe_expert_sum_add(കണ്ണി)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_SCALE: {
        എങ്കിൽ (കണ്ണി->view_src && കണ്ണി->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = കണ്ണി->view_src->ne[1] == 1 ? 2 : 1;
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_L2_NORM: {
        എങ്കിൽ (std::string(കണ്ണി->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_OP_FLASH_ATTN_EXT: {
        എങ്കിൽ (കണ്ണി->src[1] != nullptr && കണ്ണി->src[1]->op == GGML_OP_VIEW && കണ്ണി->src[1]->view_src != nullptr) {
            സ്ഥിരം ggml_tensor * kv_buf = കണ്ണി->src[1]->view_src;
            എങ്കിൽ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (കണ്ണി->src[3] != nullptr) ? 1 : 2;
            }
        }
        നിർത്തുക;
    }
    സ്ഥിരസ്ഥിതി:
        നിർത്തുക;
    }
    തിരികെ op_case;
}

std::optional<പൂർണ്ണസംഖ്യ> extract_layer_from_name(സ്ഥിരം std::string & name) {
    size_t pos1 = name.find("_l");
    എങ്കിൽ (pos1 == std::string::npos) {
        തിരികെ std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    എങ്കിൽ (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    പൂർണ്ണസംഖ്യ layer = std::stoi(layer_str);
    തിരികെ layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
സ്ഥിതം പൂർണ്ണസംഖ്യ get_swa_window_from_mask(സ്ഥിരം ggml_tensor * mask) {
    എങ്കിൽ (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        തിരികെ -1;
    }
    എങ്കിൽ (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t n_kv = mask->ne[0];
    സ്ഥിരം int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    വേണ്ടി (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        വേണ്ടി (int64_t c = 0; c < n_kv; c++) {
            സ്ഥിരം size_t i = (size_t) r * n_kv + c;
            സ്ഥിരം ദശാംശം v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((സ്ഥിരം ggml_fp16_t *) mask->data)[i]) :
                                                          ((സ്ഥിരം ദശാംശം *) mask->data)[i];
            എങ്കിൽ (v > -INFINITY) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    തിരികെ window > 0 ? (പൂർണ്ണസംഖ്യ) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, തർക്കം is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    auto get_attention_pattern_case = [](സ്ഥിരം ggml_tensor * കണ്ണി) -> പൂർണ്ണസംഖ്യ {
        എങ്കിൽ (കണ്ണി == nullptr) {
            തിരികെ -1;
        }

        തിരഞ്ഞെടുപ്പ് (കണ്ണി->op) {
        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
            എങ്കിൽ (കണ്ണി->src[0] == nullptr || കണ്ണി->src[1] == nullptr) {
                തിരികെ -1;
            }
            തിരഞ്ഞെടുപ്പ് (കണ്ണി->src[1]->op) {
            അവസ്ഥ GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                എങ്കിൽ (കണ്ണി->src[3] != nullptr && കണ്ണി->src[1]->src[0] != nullptr &&
                    കണ്ണി->src[1]->src[0]->op == GGML_OP_VIEW) {
                    തിരികെ 0;
                }
                നിർത്തുക;
            അവസ്ഥ GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                എങ്കിൽ (കണ്ണി->src[3] != nullptr && കണ്ണി->src[1]->src[0] != nullptr &&
                    കണ്ണി->src[1]->src[0]->op == GGML_OP_PERMUTE && കണ്ണി->src[1]->src[0]->src[0] != nullptr &&
                    കണ്ണി->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    തിരികെ 1;
                }
                നിർത്തുക;
            അവസ്ഥ GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                എങ്കിൽ (കണ്ണി->src[1]->view_src != nullptr) {
                    എങ്കിൽ (കണ്ണി->src[3] != nullptr) {
                        തിരികെ 4;  // decoder self-attention
                    }
                    തിരികെ 5;      // cross-attention or encoder self-attention
                }
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                നിർത്തുക;
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            എങ്കിൽ (കണ്ണി->src[0] != nullptr && കണ്ണി->src[1] != nullptr && കണ്ണി->src[0]->op == GGML_OP_MUL_MAT &&
                കണ്ണി->src[0]->src[0] != nullptr && കണ്ണി->src[0]->src[1] != nullptr &&
                കണ്ണി->src[0]->src[0]->op == GGML_OP_PERMUTE && കണ്ണി->src[0]->src[0]->src[0] != nullptr &&
                കണ്ണി->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                തിരികെ 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            എങ്കിൽ (കണ്ണി->src[0]->op == GGML_OP_ADD && കണ്ണി->src[0]->src[0] != nullptr &&
                കണ്ണി->src[0]->src[0]->op == GGML_OP_MUL_MAT && കണ്ണി->src[0]->src[0]->src[0] != nullptr &&
                കണ്ണി->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                തിരികെ 3;
            }
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            നിർത്തുക;
        }

        തിരികെ -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    auto get_attention_op_mask = [&get_attention_pattern_case](സ്ഥിരം ggml_tensor * കണ്ണി) -> സ്ഥിരം ggml_tensor * {
        തിരഞ്ഞെടുപ്പ് (get_attention_pattern_case(കണ്ണി)) {
        അവസ്ഥ 0:
        അവസ്ഥ 1:
            തിരികെ കണ്ണി->src[3];
        അവസ്ഥ 2:
        അവസ്ഥ 3:
            തിരികെ കണ്ണി->src[1];
        സ്ഥിരസ്ഥിതി:
            തിരികെ nullptr;
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
        std::map<പൂർണ്ണസംഖ്യ, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<പൂർണ്ണസംഖ്യ, സ്ഥിരം ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
            സ്ഥിരം ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            എങ്കിൽ (mask == nullptr) {
                തുടരുക;
            }
            സ്ഥിരം ggml_tensor * cache_k_permute = nullptr;
            തിരഞ്ഞെടുപ്പ് (get_attention_pattern_case(cgraph->nodes[i])) {
            അവസ്ഥ 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     നിർത്തുക;
            അവസ്ഥ 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             നിർത്തുക;
            അവസ്ഥ 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             നിർത്തുക;
            സ്ഥിരസ്ഥിതി: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     നിർത്തുക;
            }
            സ്ഥിരം ggml_tensor * cache_k_view = cache_k_permute->src[0];
            എങ്കിൽ (cache_k_view->op != GGML_OP_VIEW) {
                തുടരുക;
            }
            സ്ഥിരം ggml_tensor * leaf = cache_k_view->src[0];
            auto layer = extract_layer_from_name(leaf->name);
            എങ്കിൽ (!layer.has_value()) {
                തുടരുക;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        വേണ്ടി (സ്ഥിരം auto & [layer, extent] : layer_extent) {
            എങ്കിൽ (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                എങ്കിൽ (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            വേണ്ടി (സ്ഥിരം auto & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (long) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    തർക്കം rope_seen = അസത്യം;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * കണ്ണി = cgraph->nodes[i];
        സ്ഥിരം പൂർണ്ണസംഖ്യ attention_pattern_case = get_attention_pattern_case(കണ്ണി);
        എങ്കിൽ (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            തിരഞ്ഞെടുപ്പ് (attention_pattern_case) {
            അവസ്ഥ 0:
                cache_k_permute = കണ്ണി->src[1];
                mask = കണ്ണി->src[3];
                നിർത്തുക;
            അവസ്ഥ 1:
                cache_k_permute = കണ്ണി->src[1]->src[0];
                mask = കണ്ണി->src[3];
                നിർത്തുക;
            അവസ്ഥ 2:
                cache_k_permute = കണ്ണി->src[0]->src[0];
                mask = കണ്ണി->src[1];
                നിർത്തുക;
            അവസ്ഥ 3:
                cache_k_permute = കണ്ണി->src[0]->src[0]->src[0];
                mask = കണ്ണി->src[1];
                നിർത്തുക;
            അവസ്ഥ 4:
            അവസ്ഥ 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                auto * cache_k_view = കണ്ണി->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = കണ്ണി->src[0]->ne[1];
                എങ്കിൽ (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } അല്ലെങ്കിൽ {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                തുടരുക;
            }
            സ്ഥിരസ്ഥിതി:
                നിർത്തുക;
            }

            പ്രതിജ്ഞാ(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.നിവേശം_len = കണ്ണി->src[0]->ne[1];
            compute_params.token_len_per_seq = കണ്ണി->src[0]->ne[1];

            auto * cache_k_view = cache_k_permute->src[0];
            എങ്കിൽ (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                തുടരുക;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            പൂർണ്ണസംഖ്യ layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            സ്ഥിരം തർക്കം layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            എങ്കിൽ (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } അല്ലെങ്കിൽ {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, വലുപ്പം(size_t));
            compute_params.seq_active_start = offset / seq_size;

            എങ്കിൽ (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } അല്ലെങ്കിൽ {
                compute_params.attention_size = mask->ne[0];
            }
            എങ്കിൽ (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        എങ്കിൽ (കണ്ണി->op == GGML_OP_MUL_MAT && കണ്ണി->src[0]->op == GGML_OP_PERMUTE &&
            കണ്ണി->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(കണ്ണി->src[0]->view_src, കണ്ണി->view_src)) {
            എങ്കിൽ (കണ്ണി->src[1]->op == GGML_OP_PERMUTE && കണ്ണി->src[1]->src[0]->op == GGML_OP_VIEW &&
                കണ്ണി->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = കണ്ണി->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        എങ്കിൽ (കണ്ണി->op == GGML_OP_TRANSPOSE && കണ്ണി->src[0]->op == GGML_OP_PERMUTE &&
            കണ്ണി->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = കണ്ണി->ne[0];
            എങ്കിൽ (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        എങ്കിൽ (കണ്ണി->op == GGML_OP_ROPE) {
            എങ്കിൽ (compute_params.token_len_per_seq == -1 && കണ്ണി->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(കണ്ണി->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(വലുപ്പം(model_params.rope_params) == വലുപ്പം(int32_t) * 16, "rope_params size");
            എങ്കിൽ (!rope_seen) {
                memcpy(model_params.rope_params, കണ്ണി->op_params, വലുപ്പം(int32_t) * 16);
                rope_seen = സത്യം;
            } അല്ലെങ്കിൽ എങ്കിൽ (memcmp(model_params.rope_params, കണ്ണി->op_params, വലുപ്പം(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = സത്യം;
            }
        }
        എങ്കിൽ (കണ്ണി->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = കണ്ണി->src[0]->ne[0];
        }
        എങ്കിൽ (കണ്ണി->op == GGML_OP_SCALE && കണ്ണി->view_src != nullptr && is_recurrent_cache(കണ്ണി->view_src)) {
            എങ്കിൽ (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = കണ്ണി->view_src->ne[1];
            } അല്ലെങ്കിൽ {
                GGML_ASSERT(model_params.n_rs_slots == കണ്ണി->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(കണ്ണി) / കണ്ണി->view_src->ne[0];
            compute_params.cache_rs_reset_idx = കണ്ണി->src[0]->view_offs / കണ്ണി->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        എങ്കിൽ (കണ്ണി->op == GGML_OP_CPY && കണ്ണി->view_src != nullptr && is_recurrent_cache(കണ്ണി->view_src) &&
            കണ്ണി->src[1] != nullptr && കണ്ണി->src[1]->op == GGML_OP_VIEW && കണ്ണി->src[1]->view_src == കണ്ണി->view_src) {
            സ്ഥിരം തർക്കം is_conv = is_conv_state_writeback(കണ്ണി);
            സ്ഥിരം തർക്കം is_gdn = കണ്ണി->src[0]->op == GGML_OP_VIEW && കണ്ണി->src[0]->src[0] != nullptr &&
                                കണ്ണി->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            സ്ഥിരം തർക്കം is_extra = കണ്ണി->src[0]->op == GGML_OP_GET_ROWS;
            സ്ഥിരം തർക്കം is_gdn_rollback = is_gdn && is_same_shape(കണ്ണി->src[0], കണ്ണി->src[1]);

            സ്ഥിരം ggml_tensor * dest_view = കണ്ണി->src[1];
            സ്ഥിരം ggml_tensor * cache = കണ്ണി->view_src;
            സ്ഥിരം size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            എങ്കിൽ (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = സത്യം;
            }
            എങ്കിൽ (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(കണ്ണി)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (പൂർണ്ണസംഖ്യ) (dest_view->view_offs / row_bytes);
                എങ്കിൽ (is_conv) {
                    writeback.src_begin = (പൂർണ്ണസംഖ്യ) (കണ്ണി->src[0]->view_offs / കണ്ണി->src[0]->view_src->nb[0]);
                } അല്ലെങ്കിൽ എങ്കിൽ (is_gdn_rollback) {
                    writeback.src_begin = (പൂർണ്ണസംഖ്യ) (കണ്ണി->src[0]->view_offs / കണ്ണി->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, കണ്ണി)] = writeback;
            }
            എങ്കിൽ ((is_conv || is_gdn) && !is_full_single_slot_writeback(കണ്ണി)) {
                compute_params.s_copy_active_slot_len = (പൂർണ്ണസംഖ്യ) dest_view->ne[1];
            }
        }
    }
    എങ്കിൽ (model_params.n_heads_kv == -1) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
            സ്ഥിരം auto * കണ്ണി = cgraph->nodes[i];
            സ്ഥിരം ggml_tensor * mask = nullptr;
            എങ്കിൽ (കണ്ണി->op == GGML_OP_SOFT_MAX) {
                mask = കണ്ണി->src[1];
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = കണ്ണി->src[3];
            } അല്ലെങ്കിൽ {
                തുടരുക;
            }
            എങ്കിൽ (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_നിവേശം) ||
                കണ്ണി->src[0] == nullptr) {
                തുടരുക;
            }
            model_params.is_cacheless_attn = സത്യം;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.നിവേശം_len = കണ്ണി->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.നിവേശം_len;
            നിർത്തുക;
        }
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    എങ്കിൽ (model_params.is_cacheless_attn) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
            സ്ഥിരം auto * കണ്ണി = cgraph->nodes[i];
            എങ്കിൽ (കണ്ണി->op == GGML_OP_GET_ROWS && is_output_idx(കണ്ണി->src[1], കണ്ണി)) {
                compute_params.output_len = കണ്ണി->src[1]->ne[0];
                നിർത്തുക;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    എങ്കിൽ (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    തിരികെ {model_params, compute_params};
}

ശൂന്യം GgmlOvDecoder::validate_cgraph() സ്ഥിരം {
    എങ്കിൽ (m_model_params.n_seq > 1 && m_is_static == സത്യം) {
        എറിയുക std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_നിവേശം_shape(സ്ഥിരം ggml_tensor * op,
                                                      സ്ഥിരം ggml_tensor * നിവേശം,
                                                      പൂർണ്ണസംഖ്യ dynamic_dim_index) സ്ഥിരം {
    എങ്കിൽ (m_naive) {
        തിരികെ നിവേശം != nullptr ? ov::PartialShape{get_shape(നിവേശം)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape നിവേശം_shape;

    എങ്കിൽ (is_inp_tok(നിവേശം, op) || is_inp_pos(നിവേശം, op)) {
        // tokens or positions
        പൂർണ്ണസംഖ്യ len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        എങ്കിൽ (m_is_static && is_inp_pos(നിവേശം, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        നിവേശം_shape = ov::PartialShape{1, 1, 1, len};

    } അല്ലെങ്കിൽ എങ്കിൽ (is_output_idx(നിവേശം, op)) {
        // output index
        നിവേശം_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } അല്ലെങ്കിൽ എങ്കിൽ (is_inp_mean(നിവേശം, op)) {
        നിവേശം_shape = m_is_static ? ov::PartialShape{1, 1, നിവേശം->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } അല്ലെങ്കിൽ എങ്കിൽ (is_inp_scale_rows(നിവേശം, op)) {
        നിവേശം_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } അല്ലെങ്കിൽ എങ്കിൽ (is_inp_mask(നിവേശം, op)) {
        // mask
        എങ്കിൽ (m_is_static) {
            നിവേശം_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } അല്ലെങ്കിൽ എങ്കിൽ (m_is_stateful) {
            നിവേശം_shape = ov::PartialShape{1, 1, -1, -1};
        } അല്ലെങ്കിൽ {
            നിവേശം_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } അല്ലെങ്കിൽ എങ്കിൽ (is_recurrent_cache(നിവേശം)) {
        നിവേശം_shape = ov::PartialShape{get_shape(നിവേശം)};
        എങ്കിൽ (!m_is_static && !m_is_stateful && നിവേശം->ne[1] > 1) {
            നിവേശം_shape[2] = -1;
        }

    } അല്ലെങ്കിൽ എങ്കിൽ (is_kvcache(നിവേശം, op)) {
        // kvcache
        നിവേശം_shape = ov::PartialShape{get_shape(നിവേശം)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        സ്ഥിരം തർക്കം is_flat_kv = (നിവേശം->ne[1] == 1 && നിവേശം->ne[2] == 1 && നിവേശം->ne[3] == 1);
        എങ്കിൽ (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            നിവേശം_shape[2] = -1;
        }
        എങ്കിൽ (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            പ്രതിജ്ഞാ(നിവേശം_shape.size() == 4 && നിവേശം_shape[0] == 1 && നിവേശം_shape[1] == 1 &&
                   നിവേശം_shape[2].is_dynamic() && നിവേശം_shape[3].is_static());
            സ്ഥിരം പൂർണ്ണസംഖ്യ n_heads_kv = get_n_heads_kv_for_tensor(നിവേശം);
            പ്രതിജ്ഞാ(n_heads_kv > 0 && നിവേശം_shape[3].get_length() % n_heads_kv == 0);
            സ്ഥിരം int64_t combined_dim = നിവേശം_shape[3].get_length();  // n_heads_kv * head_size
            സ്ഥിരം int64_t head_size = combined_dim / n_heads_kv;
            നിവേശം_shape = {നിവേശം_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } അല്ലെങ്കിൽ എങ്കിൽ (is_kv_idx(നിവേശം, op)) {
        // kv update index
        പൂർണ്ണസംഖ്യ len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        നിവേശം_shape = ov::PartialShape{1, 1, 1, len};

    } അല്ലെങ്കിൽ എങ്കിൽ (is_inp_s_copy(നിവേശം, op) || is_s_copy_leaf(നിവേശം)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        നിവേശം_shape = m_is_static ? ov::PartialShape{get_shape(നിവേശം)} : ov::PartialShape{1, 1, 1, -1};

    } അല്ലെങ്കിൽ {
        നിവേശം_shape = ov::PartialShape{get_shape(നിവേശം)};
    }
    എങ്കിൽ (dynamic_dim_index != -1 && m_model_is_splitted) {
        നിവേശം_shape[3 - dynamic_dim_index] = -1;
    }
    എങ്കിൽ (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_നിവേശം && op->src[1] == നിവേശം) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        എങ്കിൽ (m_is_static) {
            സ്ഥിരം int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            നിവേശം_shape[2] = seq_active;
            നിവേശം_shape[3] = seq_active;
        } അല്ലെങ്കിൽ {
            നിവേശം_shape[2] = -1;
            നിവേശം_shape[3] = -1;
        }
    }
    തിരികെ നിവേശം_shape;
}

തർക്കം GgmlOvDecoder::is_s_copy_leaf(സ്ഥിരം ggml_tensor * tensor) സ്ഥിരം {
    എങ്കിൽ (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        തിരികെ അസത്യം;
    }
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m_cgraph->n_nodes; i++) {
        സ്ഥിരം ggml_tensor * കണ്ണി = m_cgraph->nodes[i];
        എങ്കിൽ (കണ്ണി->op != GGML_OP_GET_ROWS || കണ്ണി->src[0] == nullptr || കണ്ണി->src[1] == nullptr) {
            തുടരുക;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        സ്ഥിരം ggml_tensor * idx = കണ്ണി->src[1];
        ആകുമ്പോൾ (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        എങ്കിൽ (idx != tensor) {
            തുടരുക;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        സ്ഥിരം ggml_tensor * data = കണ്ണി->src[0];
        ആകുമ്പോൾ (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        എങ്കിൽ (data != nullptr && is_recurrent_cache(data)) {
            തിരികെ സത്യം;
        }
    }
    തിരികെ അസത്യം;
}

ശൂന്യം GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_നിവേശം = [ഇത്](സ്ഥിരം std::string & name, int64_t value, തർക്കം force_parameter = അസത്യം) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    എങ്കിൽ (m_compute_params.attention_size != -1) {
        create_1d_നിവേശം("attention_size", m_compute_params.attention_size);
    }
    എങ്കിൽ (m_compute_params.attention_size_static != -1) {
        create_1d_നിവേശം("attention_size_static", m_compute_params.attention_size_static);
    }
    എങ്കിൽ (m_compute_params.attention_size_swa != -1) {
        create_1d_നിവേശം("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    എങ്കിൽ (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_നിവേശം("swa_window", m_compute_params.swa_window);
    }
    create_1d_നിവേശം("n_seq_active", m_compute_params.n_seq_active);
    create_1d_നിവേശം("seq_active_start", m_compute_params.seq_active_start);
    create_1d_നിവേശം("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    എങ്കിൽ (m_compute_params.token_len_per_seq != -1) {
        create_1d_നിവേശം("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    എങ്കിൽ (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_നിവേശം("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/സത്യം);
        create_1d_നിവേശം("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/സത്യം);
    }

    എങ്കിൽ (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_നിവേശം("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        എങ്കിൽ (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_നിവേശം("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/സത്യം);
        }
    }

    വേണ്ടി (സ്ഥിരം auto & [കണ്ണി_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_നിവേശം("rs_slot_begin_" + കണ്ണി_name, writeback.slot_begin);
        എങ്കിൽ (!m_is_static && writeback.src_begin >= 0) {
            create_1d_നിവേശം("rs_src_begin_" + കണ്ണി_name, writeback.src_begin);
        }
    }
}

തർക്കം GgmlOvDecoder::കണ്ണി_is_used_as_src(സ്ഥിരം പൂർണ്ണസംഖ്യ കണ്ണി_idx) {
    ggml_tensor * കണ്ണി = m_cgraph->nodes[കണ്ണി_idx];
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = കണ്ണി_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_കണ്ണി = m_cgraph->nodes[i];
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            എങ്കിൽ (other_കണ്ണി->src[j] == കണ്ണി) {
                തിരികെ സത്യം;
            }
        }
    }
    തിരികെ അസത്യം;
}

ശൂന്യം GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * കണ്ണി = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        എങ്കിൽ (കണ്ണി->op == GGML_OP_NONE && കണ്ണി_is_used_as_src(i)) {
            std::string കണ്ണി_name = get_tensor_ov_name(m_cgraph, കണ്ണി);
            എങ്കിൽ (m_model_weights.find(കണ്ണി_name) == m_model_weights.end()) {
                m_inputs[കണ്ണി_name] = കണ്ണി;
                m_model_inputs[കണ്ണി_name] = {get_ov_type(കണ്ണി),
                                             get_graph_നിവേശം_shape(കണ്ണി, nullptr, m_കണ്ണി_dynamic_dims[കണ്ണി])};
            }
            തുടരുക;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = കണ്ണി->src[i];
            എങ്കിൽ (src == nullptr) {
                തുടരുക;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            എങ്കിൽ (src->flags & GGML_TENSOR_FLAG_നിവേശം) {
                src_name = get_tensor_graph_നിവേശം_ov_name(ഇത്, m_cgraph, src, കണ്ണി);
            }
            എങ്കിൽ (m_model_weights.find(src_name) != m_model_weights.end()) {
                തുടരുക;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            എങ്കിൽ (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                തുടരുക;
            }

            തർക്കം is_intermediate_കണ്ണി = അസത്യം;
            വേണ്ടി (സ്ഥിരം auto & കണ്ണി_info : m_കണ്ണി_info_list) {
                എങ്കിൽ (കണ്ണി_info.കണ്ണി == src) {
                    is_intermediate_കണ്ണി = സത്യം;
                    നിർത്തുക;
                }
            }
            എങ്കിൽ (is_intermediate_കണ്ണി) {
                തുടരുക;
            }
            എങ്കിൽ (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                തുടരുക;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            എങ്കിൽ (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                എങ്കിൽ (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            ആകുമ്പോൾ (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_നിവേശം_shape(കണ്ണി, src, m_കണ്ണി_dynamic_dims[src])};
        }
    }
}

ശൂന്യം GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    വേണ്ടി (പൂർണ്ണസംഖ്യ കണ്ണി_n = 0; കണ്ണി_n < m_cgraph->n_nodes; കണ്ണി_n++) {
        auto * cur_കണ്ണി = m_cgraph->nodes[കണ്ണി_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        എങ്കിൽ (cur_കണ്ണി->op == GGML_OP_NONE || cur_കണ്ണി->op == GGML_OP_VIEW || cur_കണ്ണി->op == GGML_OP_RESHAPE) {
            തുടരുക;
        }
        എങ്കിൽ (::is_inplace_op(cur_കണ്ണി) && ggml_nbytes(cur_കണ്ണി) == 0) {
            തുടരുക;
        }
        auto cur_കണ്ണി_use_എണ്ണം = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_കണ്ണി)];
        എങ്കിൽ (cur_കണ്ണി_use_എണ്ണം == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            എങ്കിൽ (cur_കണ്ണി != nullptr && ::is_inplace_op(cur_കണ്ണി) && ggml_nbytes(cur_കണ്ണി) > 0) {
                cur_കണ്ണി = cur_കണ്ണി->view_src;
            }
        } അല്ലെങ്കിൽ {
            പൂർണ്ണസംഖ്യ നിവേശം_use_എണ്ണം = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * കണ്ണി = m_cgraph->nodes[i];
                വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
                    എങ്കിൽ (കണ്ണി->src[j] != ശൂന്യ && കണ്ണി->src[j] == cur_കണ്ണി) {
                        നിവേശം_use_എണ്ണം++;
                    }
                }
            }
            എങ്കിൽ (നിവേശം_use_എണ്ണം == cur_കണ്ണി_use_എണ്ണം) {
                cur_കണ്ണി = nullptr;
            }
        }
        എങ്കിൽ (cur_കണ്ണി != nullptr) {
            std::string cur_കണ്ണി_name = get_tensor_ov_name(m_cgraph, cur_കണ്ണി);
            m_model_outputs[cur_കണ്ണി_name] = cur_കണ്ണി;
            m_model_output_names.insert(cur_കണ്ണി_name);
        }
    }
}

സ്ഥിരം ggml_tensor * GgmlOvDecoder::get_tensor_used_op(സ്ഥിരം ggml_tensor * tensor) സ്ഥിരം {
    എങ്കിൽ (tensor == nullptr) {
        തിരികെ nullptr;
    }
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m_cgraph->n_nodes; i++) {
        സ്ഥിരം auto * കണ്ണി = m_cgraph->nodes[i];
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            എങ്കിൽ (കണ്ണി->src[j] == tensor) {
                തിരികെ കണ്ണി;
            }
        }
    }
    തിരികെ nullptr;
}

സ്ഥിരം ggml_tensor * GgmlOvDecoder::get_tensor_from_name(സ്ഥിരം std::string & name) സ്ഥിരം {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m_cgraph->n_nodes; i++) {
        സ്ഥിരം auto * കണ്ണി = m_cgraph->nodes[i];
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            സ്ഥിരം auto * src = കണ്ണി->src[j];
            എങ്കിൽ (src == nullptr) {
                നിർത്തുക;
            }
            എങ്കിൽ (get_tensor_ov_name(m_cgraph, src) == name) {
                തിരികെ src;
            }
        }
    }
    തിരികെ nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() സ്ഥിരം {
    std::map<std::string, std::string> kv_param_res_names;
    വേണ്ടി (സ്ഥിരം auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    തിരികെ kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
സ്ഥിതം തർക്കം is_mul_mat_id_expert_weight(സ്ഥിരം ggml_tensor * കണ്ണി, പൂർണ്ണസംഖ്യ src_index) {
    തിരികെ കണ്ണി->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, തർക്കം naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    വേണ്ടി (പൂർണ്ണസംഖ്യ കണ്ണി_i = 0; കണ്ണി_i < n_nodes; കണ്ണി_i++) {
        auto * കണ്ണി = nodes[കണ്ണി_i];
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = കണ്ണി->src[i];
            എങ്കിൽ (src == nullptr) {
                തുടരുക;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            എങ്കിൽ (is_rope_freqs_weight(base, കണ്ണി)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            എങ്കിൽ (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(കണ്ണി, i)) {
                എങ്കിൽ (model_weights.find(src_name) == model_weights.end()) {
                    auto weight_കണ്ണി = create_weight_കണ്ണി(base, naive);
                    weight_കണ്ണി->set_friendly_name(src_name);
                    model_weights[src_name] = weight_കണ്ണി;
                }
            }
        }
    }
    തിരികെ model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
സ്ഥിതം std::mutex g_nonov_weight_cache_mutex;
സ്ഥിതം std::unordered_map<സ്ഥിരം ശൂന്യം *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    വേണ്ടി (പൂർണ്ണസംഖ്യ കണ്ണി_i = 0; കണ്ണി_i < cgraph->n_nodes; കണ്ണി_i++) {
        auto * കണ്ണി = cgraph->nodes[കണ്ണി_i];
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = കണ്ണി->src[i];
            എങ്കിൽ (src == nullptr) {
                തുടരുക;
            }
            സ്ഥിരം ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            എങ്കിൽ (is_rope_freqs_weight(base, കണ്ണി)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            എങ്കിൽ (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    തിരികെ names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_കണ്ണി(ggml_tensor * tensor, തർക്കം naive) {
    സ്ഥിരം തർക്കം is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    എങ്കിൽ (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        എങ്കിൽ (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            എങ്കിൽ (weight_extra->weight_കണ്ണി) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                തിരികെ weight_extra->weight_കണ്ണി;
            }
        } അല്ലെങ്കിൽ എങ്കിൽ (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            എങ്കിൽ (quant_extra->weight_കണ്ണി) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                തിരികെ quant_extra->weight_കണ്ണി;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    എങ്കിൽ (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        auto weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        auto weight_കണ്ണി = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_കണ്ണി->set_friendly_name(tensor->name);
        തിരികെ weight_കണ്ണി;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    സ്ഥിരം തർക്കം cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    എങ്കിൽ (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        auto it = g_nonov_weight_cache.find(tensor->data);
        എങ്കിൽ (it != g_nonov_weight_cache.end()) {
            തിരികെ it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    സ്ഥിതം സ്ഥിരം std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    എങ്കിൽ (weight_types.find(tensor->type) == weight_types.end()) {
        എറിയുക std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    എങ്കിൽ (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        എങ്കിൽ (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } അല്ലെങ്കിൽ {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } അല്ലെങ്കിൽ {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_കണ്ണി->set_friendly_name(tensor->name);
    എങ്കിൽ (!is_ov_buffer) {
        എങ്കിൽ (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            auto [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_കണ്ണി);
            തിരികെ it->second;
        }
        തിരികെ ov_weight.weight_കണ്ണി;
    }

    ggml_openvino_extra_base * extra;
    എങ്കിൽ (ov_weight.is_quantized()) {
        extra = പുതിയ ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_കണ്ണി);
    } അല്ലെങ്കിൽ {
        extra = പുതിയ ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_കണ്ണി);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    തിരികെ ov_weight.weight_കണ്ണി;
}

ശൂന്യം GgmlOvDecoder::dump_cgraph(സ്ഥിരം ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    എങ്കിൽ (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        തിരികെ;
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * കണ്ണി = cgraph->nodes[i];

        // Get buffer type name
        സ്ഥിരം അക്ഷരം * buf_name = "none";
        ggml_backend_buffer_t buf = കണ്ണി->view_src ? കണ്ണി->view_src->buffer : കണ്ണി->buffer;
        എങ്കിൽ (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << കണ്ണി->ne[0] << ", "
             << std::setw(5) << കണ്ണി->ne[1] << ", "
             << std::setw(5) << കണ്ണി->ne[2] << ", "
             << std::setw(5) << കണ്ണി->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(കണ്ണി->op) << std::right << " "
             << std::left << std::setw(45) << കണ്ണി->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << കണ്ണി->nb[0] << ", "
             << std::setw(5) << കണ്ണി->nb[1] << ", "
             << std::setw(5) << കണ്ണി->nb[2] << ", "
             << std::setw(5) << കണ്ണി->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            എങ്കിൽ (auto* src = കണ്ണി->src[i]) {
                // Get buffer type name for source
                സ്ഥിരം അക്ഷരം * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                എങ്കിൽ (src_buf) {
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * കണ്ണി = cgraph->leafs[i];

        // Get buffer type name for leaf
        സ്ഥിരം അക്ഷരം * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = കണ്ണി->view_src ? കണ്ണി->view_src->buffer : കണ്ണി->buffer;
        എങ്കിൽ (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << കണ്ണി->ne[0] << ", "
             << std::setw(5) << കണ്ണി->ne[1] << "] "
             << std::setw(8) << ggml_op_name(കണ്ണി->op) << " "
             << std::setw(16) << ggml_get_name(കണ്ണി)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

ശൂന്യം print_tensor_address_map(സ്ഥിരം ggml_cgraph * cgraph) {
    std::map<ശൂന്യം *, std::vector<std::string>> address_map;
    വേണ്ടി (പൂർണ്ണസംഖ്യ കണ്ണി_n = 0; കണ്ണി_n < cgraph->n_nodes; കണ്ണി_n++) {
        auto * കണ്ണി = cgraph->nodes[കണ്ണി_n];
        എങ്കിൽ (കണ്ണി->data) {
            auto it = address_map.find(കണ്ണി->data);
            എങ്കിൽ (it == address_map.end()) {
                address_map[കണ്ണി->data] = std::vector<std::string>();
            }
            address_map[കണ്ണി->data].push_back(കണ്ണി->name);
        }
    }
    വേണ്ടി (സ്ഥിരം auto & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        വേണ്ടി (സ്ഥിരം auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(സ്ഥിരം ggml_tensor * tensor) {
    std::vector<size_t> shape;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    തിരികെ shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(സ്ഥിരം ggml_tensor * tensor) {
    std::vector<size_t> stride;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    തിരികെ stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(സ്ഥിരം ggml_tensor * tensor) {
    തിരഞ്ഞെടുപ്പ് (tensor->type) {
    അവസ്ഥ GGML_TYPE_F64:
        തിരികെ ov::element::f64;
    അവസ്ഥ GGML_TYPE_F32:
        തിരികെ ov::element::f32;
    അവസ്ഥ GGML_TYPE_F16:
        തിരികെ ov::element::f16;
    അവസ്ഥ GGML_TYPE_BF16:
        തിരികെ ov::element::bf16;
    അവസ്ഥ GGML_TYPE_I8:
        തിരികെ ov::element::i8;
    അവസ്ഥ GGML_TYPE_I16:
        തിരികെ ov::element::i16;
    അവസ്ഥ GGML_TYPE_I32:
        തിരികെ ov::element::i32;
    അവസ്ഥ GGML_TYPE_I64:
        തിരികെ ov::element::i64;
    സ്ഥിരസ്ഥിതി:
        തിരികെ ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_നിവേശം_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം {
    തിരികെ ov::PartialShape(get_shape(m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_നിവേശം_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം {
    തിരികെ get_stride(m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_നിവേശം_size(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        തിരികെ it->second.size();
    }
    തിരികെ 0;
}

size_t GgmlOvDecoder::get_view_നിവേശം_offset(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        എങ്കിൽ (view_index < it->second.size()) {
            തിരികെ it->second[view_index].second->view_offs;
        }
    }
    തിരികെ 0;
}

size_t GgmlOvDecoder::get_view_നിവേശം_src_offset(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        എങ്കിൽ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            എങ്കിൽ (view_tensor && view_tensor->src[0]) {
                തിരികെ view_tensor->src[0]->view_offs;
            }
        }
    }
    തിരികെ 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_നിവേശം_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                         സ്ഥിരം std::string & name,
                                                         size_t view_index) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        എങ്കിൽ (view_index < it->second.size()) {
            തിരികെ get_stride(it->second[view_index].second);
        }
    }
    തിരികെ {};
}

std::vector<size_t> GgmlOvDecoder::get_view_നിവേശം_src_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                             സ്ഥിരം std::string & name,
                                                             size_t view_index) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        എങ്കിൽ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            എങ്കിൽ (view_tensor && view_tensor->src[0]) {
                തിരികെ get_stride(view_tensor->src[0]);
            }
        }
    }
    തിരികെ {};
}

ov::Shape GgmlOvDecoder::get_view_നിവേശം_ggml_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        എങ്കിൽ (view_index < it->second.size()) {
            തിരികെ get_shape(it->second[view_index].second);
        }
    }
    തിരികെ {};
}

ov::Shape GgmlOvDecoder::get_view_നിവേശം_src_ggml_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                       സ്ഥിരം std::string & name,
                                                       size_t view_index) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        എങ്കിൽ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            എങ്കിൽ (view_tensor && view_tensor->src[0]) {
                തിരികെ get_shape(view_tensor->src[0]);
            }
        }
    }
    തിരികെ {};
}

ov::PartialShape GgmlOvDecoder::get_view_നിവേശം_ov_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                        സ്ഥിരം std::string & name,
                                                        size_t view_index) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        എങ്കിൽ (view_index < it->second.size()) {
            auto * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            auto dynamic_it = m_കണ്ണി_dynamic_dims.find(tensor);
            എങ്കിൽ (dynamic_it != m_കണ്ണി_dynamic_dims.end() && dynamic_it->second != -1) {
                പൂർണ്ണസംഖ്യ dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            തിരികെ shape;
        }
    }
    തിരികെ {};
}

ov::PartialShape GgmlOvDecoder::get_view_നിവേശം_src_ov_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                            സ്ഥിരം std::string & name,
                                                            size_t view_index) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        എങ്കിൽ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            എങ്കിൽ (view_tensor && view_tensor->src[0]) {
                auto * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                auto dynamic_it = m_കണ്ണി_dynamic_dims.find(src_tensor);
                എങ്കിൽ (dynamic_it != m_കണ്ണി_dynamic_dims.end() && dynamic_it->second != -1) {
                    പൂർണ്ണസംഖ്യ dynamic_dim_index = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
                }

                തിരികെ shape;
            }
        }
    }
    തിരികെ {};
}

std::string GgmlOvDecoder::get_view_നിവേശം_name(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        എങ്കിൽ (view_index < it->second.size()) {
            തിരികെ it->second[view_index].first;
        }
    }
    തിരികെ "";
}

std::string GgmlOvDecoder::get_view_നിവേശം_src_name(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം {
    auto it = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.find(name);
    എങ്കിൽ (it != m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_views.end()) {
        എങ്കിൽ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            എങ്കിൽ (view_tensor && view_tensor->src[0]) {
                തിരികെ get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    തിരികെ "";
}

ov::element::Type GgmlOvDecoder::get_നിവേശം_type(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം {
    തിരികെ get_ov_type(m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs.at(name));
}

size_t GgmlOvDecoder::get_നിവേശം_size() സ്ഥിരം {
    തിരികെ m_model_inputs.size();
}

size_t GgmlOvDecoder::get_നിവേശം_size(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    തിരികെ m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_നിവേശം_names(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    തിരികെ m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    auto * ggml_tensor = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി;
    തിരികെ ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(സ്ഥിരം പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    തിരികെ get_ov_type(m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    auto * ggml_tensor = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി;
    തിരികെ get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    auto * കണ്ണി = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി;
    എങ്കിൽ (കണ്ണി->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = കണ്ണി_idx + 1; i < m_cgraph->n_nodes; i++) {
            auto * consumer = m_cgraph->nodes[i];
            എങ്കിൽ (consumer->op != GGML_OP_VIEW || consumer->src[0] != കണ്ണി) {
                തുടരുക;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            auto & name = consumer->view_offs == 0 ? attn_name : state_name;
            എങ്കിൽ (!name.empty()) {
                തിരികെ {m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        എങ്കിൽ (!attn_name.empty() && !state_name.empty()) {
            തിരികെ {attn_name, state_name};
        }
    }
    തിരികെ {m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    auto * കണ്ണി = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി;
    എങ്കിൽ (!::is_inplace_op(കണ്ണി) || കണ്ണി->view_src == nullptr || ggml_nbytes(കണ്ണി) == 0) {
        തിരികെ "";
    }
    സ്ഥിരം പൂർണ്ണസംഖ്യ op_case = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_op_case;
    എങ്കിൽ (കണ്ണി->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        തിരികെ "";
    }
    തിരികെ get_tensor_ov_name(m_cgraph, കണ്ണി->view_src);
}

തർക്കം GgmlOvDecoder::is_view_like_alias_of(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & view_src_name) സ്ഥിരം {
    auto * കണ്ണി = m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി;
    എങ്കിൽ (കണ്ണി->view_src == nullptr || get_tensor_ov_name(m_cgraph, കണ്ണി->view_src) != view_src_name) {
        തിരികെ അസത്യം;
    }
    തിരികെ കണ്ണി->op == GGML_OP_RESHAPE || കണ്ണി->op == GGML_OP_VIEW;
}

സ്ഥിരം std::string & GgmlOvDecoder::get_op_name() സ്ഥിരം {
    സ്ഥിതം സ്ഥിരം std::string unknown_name = "UNKNOWN_OP_NAME";
    തിരികെ unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    auto it = m_കണ്ണി_dynamic_dims.find(m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി);
    എങ്കിൽ (it == m_കണ്ണി_dynamic_dims.end()) {
        തിരികെ -1;
    }
    തിരികെ it->second;
}

സ്ഥിരം std::string & GgmlOvDecoder::get_op_name(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    തിരികെ m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_name;
}

int32_t * GgmlOvDecoder::get_നിവേശം_op_params(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം {
    തിരികെ m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    തിരികെ m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി->op_params;
}

size_t GgmlOvDecoder::get_output_op_offset(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    തിരികെ m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി->view_offs;
}

ശൂന്യം GgmlOvDecoder::visit_subgraph(std::function<ശൂന്യം(std::shared_ptr<GgmlDecoder>, പൂർണ്ണസംഖ്യ കണ്ണി_idx)> കണ്ണി_visitor) സ്ഥിരം {
    വേണ്ടി (പൂർണ്ണസംഖ്യ കണ്ണി_idx = 0; കണ്ണി_idx < m_cgraph->n_nodes; കണ്ണി_idx++) {
        എങ്കിൽ (m_cgraph->nodes[കണ്ണി_idx]->op == GGML_OP_NONE) {
            തുടരുക;
        }
        കണ്ണി_visitor(std::make_shared<GgmlOvDecoder>(*ഇത്), കണ്ണി_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(സ്ഥിരം ggml_tensor * കണ്ണി) {
    തിരഞ്ഞെടുപ്പ് (കണ്ണി->op) {
    അവസ്ഥ GGML_OP_UNARY:
        തിരികെ std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(കണ്ണി));
    അവസ്ഥ GGML_OP_GLU:
        തിരികെ std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(കണ്ണി));
    സ്ഥിരസ്ഥിതി:
        തിരികെ std::string("GGML_OP_") + ggml_op_name(കണ്ണി->op);
    }
}

സ്ഥിരം std::string & GgmlOvDecoder::get_op_type(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം {
    തിരികെ m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_op_type;
}

സ്ഥിരം std::string & GgmlOvDecoder::get_op_type() സ്ഥിരം {
    സ്ഥിതം സ്ഥിരം std::string unknown_op = "UNKNOWN_GGML_OP";
    തിരികെ unknown_op;
}

ശൂന്യം GgmlOvDecoder::compute_കണ്ണി_dynamic_dims() {
    auto visit_കണ്ണി = [&](auto && self, ggml_tensor * കണ്ണി) -> ശൂന്യം {
        എങ്കിൽ (!കണ്ണി) {
            തിരികെ;
        }

        എങ്കിൽ (കണ്ണി->op == GGML_OP_CPY) {
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
        }

        എങ്കിൽ (m_കണ്ണി_dynamic_dims.count(കണ്ണി)) {
            തിരികെ;
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = കണ്ണി->src[i];
            എങ്കിൽ (src == nullptr) {
                തുടരുക;
            }
            ഘടന ggml_tensor * വേര്_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            എങ്കിൽ (വേര്_src) {
                എങ്കിൽ (is_inp_tok(വേര്_src, കണ്ണി) || is_inp_pos(വേര്_src, കണ്ണി) || is_output_idx(വേര്_src, കണ്ണി)) {
                    m_കണ്ണി_dynamic_dims[വേര്_src] = 0;
                    m_കണ്ണി_dynamic_dims[src] = m_കണ്ണി_dynamic_dims[വേര്_src];
                    തുടരുക;
                }
                self(self, വേര്_src);
                m_കണ്ണി_dynamic_dims[src] = m_കണ്ണി_dynamic_dims[വേര്_src];
            } അല്ലെങ്കിൽ {
                എങ്കിൽ (is_inp_tok(src, കണ്ണി) || is_inp_pos(src, കണ്ണി) || is_output_idx(src, കണ്ണി)) {
                    m_കണ്ണി_dynamic_dims[src] = 0;
                    തുടരുക;
                }
                എങ്കിൽ (is_inp_scale_rows(src, കണ്ണി)) {
                    m_കണ്ണി_dynamic_dims[src] = 1;
                    തുടരുക;
                }
                എങ്കിൽ (കണ്ണി->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_കണ്ണി_dynamic_dims[src] = 1;
                    തുടരുക;
                }
                self(self, src);
            }
        }
        തിരഞ്ഞെടുപ്പ് (കണ്ണി->op) {
        അവസ്ഥ GGML_OP_NONE:
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            നിർത്തുക;
        അവസ്ഥ GGML_OP_GET_ROWS:
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി->src[1]] != -1) {
                auto dynamic_dim_idx = m_കണ്ണി_dynamic_dims[കണ്ണി->src[1]];
                എങ്കിൽ (dynamic_dim_idx == 0) {
                    m_കണ്ണി_dynamic_dims[കണ്ണി] = 1;
                } അല്ലെങ്കിൽ {
                    auto dynamic_dim_stride = കണ്ണി->src[1]->nb[dynamic_dim_idx] / ggml_type_size(കണ്ണി->src[1]->type) *
                                              ggml_type_size(കണ്ണി->src[0]->type);
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
                        എങ്കിൽ (dynamic_dim_stride == കണ്ണി->src[0]->nb[i]) {
                            m_കണ്ണി_dynamic_dims[കണ്ണി] = i;
                            നിർത്തുക;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL:
        അവസ്ഥ GGML_OP_MUL_MAT:
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]] != -1) {
                m_കണ്ണി_dynamic_dims[കണ്ണി] = m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]];
            }
            എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി->src[1]] != -1) {
                m_കണ്ണി_dynamic_dims[കണ്ണി] = m_കണ്ണി_dynamic_dims[കണ്ണി->src[1]];
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_PERMUTE:
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]] != -1) {
                auto dynamic_dim_idx = m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
                    എങ്കിൽ (കണ്ണി->op_params[i] == dynamic_dim_idx) {
                        m_കണ്ണി_dynamic_dims[കണ്ണി] = i;
                        നിർത്തുക;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]] != -1) {
                എങ്കിൽ (കണ്ണി->src[0]->op == GGML_OP_NONE) {
                    m_കണ്ണി_dynamic_dims[കണ്ണി] = m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]];
                    നിർത്തുക;
                }
                auto dynamic_dim_idx = m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]];
                auto dynamic_dim_മൂല്യം = കണ്ണി->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    കണ്ണി->src[0]->nb[dynamic_dim_idx] / ggml_type_size(കണ്ണി->src[0]->type) * ggml_type_size(കണ്ണി->type);
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
                    എങ്കിൽ (കണ്ണി->nb[i] == dynamic_dim_stride) {
                        m_കണ്ണി_dynamic_dims[കണ്ണി] = i;
                        നിർത്തുക;
                    }
                }
                എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി] != -1 && dynamic_dim_മൂല്യം != കണ്ണി->ne[m_കണ്ണി_dynamic_dims[കണ്ണി]]) {
                    m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    എങ്കിൽ (ggml_nelements(കണ്ണി) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      കണ്ണി->name, കണ്ണി->src[0]->name);
                    }
                }
            }
            നിർത്തുക;
        }
        അവസ്ഥ GGML_OP_TRANSPOSE:
        അവസ്ഥ GGML_OP_RESHAPE: {
            എങ്കിൽ (is_same_shape(കണ്ണി->src[0], കണ്ണി)) {
                m_കണ്ണി_dynamic_dims[കണ്ണി] = m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]];
                നിർത്തുക;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]] != -1) {
                auto dynamic_dim_idx = m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]];
                auto dynamic_dim_stride = കണ്ണി->src[0]->nb[dynamic_dim_idx];
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
                    എങ്കിൽ (കണ്ണി->nb[i] == dynamic_dim_stride && കണ്ണി->ne[i] == കണ്ണി->src[0]->ne[dynamic_dim_idx]) {
                        m_കണ്ണി_dynamic_dims[കണ്ണി] = i;
                        നിർത്തുക;
                    }
                }
                എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", കണ്ണി->name);
                }
            }
            നിർത്തുക;
        }
        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr പൂർണ്ണസംഖ്യ q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]] != -1) {
                auto q_dynamic_dim = m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]];
                m_കണ്ണി_dynamic_dims[കണ്ണി] = q_to_out[q_dynamic_dim];
            }
            നിർത്തുക;
        }
        അവസ്ഥ GGML_OP_CONT:
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]] != -1) {
                auto dynamic_dim_idx = m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]];
                എങ്കിൽ (ggml_are_same_shape(കണ്ണി, കണ്ണി->src[0])) {
                    m_കണ്ണി_dynamic_dims[കണ്ണി] = dynamic_dim_idx;
                } അല്ലെങ്കിൽ {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(കണ്ണി->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (കണ്ണി->src[0]->ne[0] / ggml_blck_size(കണ്ണി->src[0]->type));
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * കണ്ണി->src[0]->ne[i - 1];
                    }

                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(കണ്ണി->src[0]->type) *
                                              ggml_type_size(കണ്ണി->type);
                    പൂർണ്ണസംഖ്യ matched_dim_എണ്ണം = 0;
                    പൂർണ്ണസംഖ്യ first_matched_dim = -1;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
                        എങ്കിൽ (കണ്ണി->nb[i] == dynamic_dim_stride && കണ്ണി->ne[i] == കണ്ണി->src[0]->ne[dynamic_dim_idx]) {
                            എങ്കിൽ (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_കണ്ണി_dynamic_dims[കണ്ണി] = i;
                            matched_dim_എണ്ണം++;
                        }
                    }
                    എങ്കിൽ (matched_dim_എണ്ണം > 1 && കണ്ണി->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_കണ്ണി_dynamic_dims[കണ്ണി] = first_matched_dim;
                    } അല്ലെങ്കിൽ എങ്കിൽ (matched_dim_എണ്ണം != 1) {
                        m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      കണ്ണി->name, കണ്ണി->src[0]->name);
                    }
                }
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CONCAT:
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
                എങ്കിൽ (കണ്ണി->src[0]->ne[i] != കണ്ണി->ne[i]) {
                    m_കണ്ണി_dynamic_dims[കണ്ണി] = i;
                    നിർത്തുക;
                }
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SSM_CONV:
        അവസ്ഥ GGML_OP_GATED_DELTA_NET:
            m_കണ്ണി_dynamic_dims[കണ്ണി] = 1;
            നിർത്തുക;
        അവസ്ഥ GGML_OP_RMS_NORM:
        അവസ്ഥ GGML_OP_L2_NORM:
        അവസ്ഥ GGML_OP_NORM:
        അവസ്ഥ GGML_OP_ADD:
        അവസ്ഥ GGML_OP_SUB:
        അവസ്ഥ GGML_OP_GLU:
        അവസ്ഥ GGML_OP_ROPE:
        അവസ്ഥ GGML_OP_SCALE:
        അവസ്ഥ GGML_OP_SOFT_MAX:
        അവസ്ഥ GGML_OP_ARGSORT:
        അവസ്ഥ GGML_OP_ADD_ID:
        അവസ്ഥ GGML_OP_UNARY:
        അവസ്ഥ GGML_OP_CUMSUM:
        അവസ്ഥ GGML_OP_FILL:
        അവസ്ഥ GGML_OP_SET:
        അവസ്ഥ GGML_OP_DIAG:
        അവസ്ഥ GGML_OP_TRI:
        അവസ്ഥ GGML_OP_REPEAT:
        അവസ്ഥ GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        അവസ്ഥ GGML_OP_DIV:
        അവസ്ഥ GGML_OP_CLAMP:
        അവസ്ഥ GGML_OP_PAD:
        അവസ്ഥ GGML_OP_UPSCALE:
        അവസ്ഥ GGML_OP_SIN:
        അവസ്ഥ GGML_OP_COS:
        അവസ്ഥ GGML_OP_LOG:
        അവസ്ഥ GGML_OP_ROLL:
            m_കണ്ണി_dynamic_dims[കണ്ണി] = m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]];
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_കണ്ണി_dynamic_dims[കണ്ണി] =
                (m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]] == 0) ? -1 : m_കണ്ണി_dynamic_dims[കണ്ണി->src[0]];
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
        അവസ്ഥ GGML_OP_SOLVE_TRI:
            m_കണ്ണി_dynamic_dims[കണ്ണി] = m_കണ്ണി_dynamic_dims[കണ്ണി->src[1]];
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CPY:
        അവസ്ഥ GGML_OP_SET_ROWS:
        അവസ്ഥ GGML_OP_SUM:
        അവസ്ഥ GGML_OP_MEAN:
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            നിർത്തുക;
        അവസ്ഥ GGML_OP_IM2COL: {
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി->src[1]] != -1) {
                സ്ഥിരം തർക്കം is_2D = കണ്ണി->op_params[6] == 1;
                സ്ഥിരം പൂർണ്ണസംഖ്യ src_dyn = m_കണ്ണി_dynamic_dims[കണ്ണി->src[1]];
                എങ്കിൽ (is_2D) {
                    എങ്കിൽ (src_dyn == 0) {
                        m_കണ്ണി_dynamic_dims[കണ്ണി] = 1;  // IW -> OW
                    } അല്ലെങ്കിൽ എങ്കിൽ (src_dyn == 1) {
                        m_കണ്ണി_dynamic_dims[കണ്ണി] = 2;  // IH -> OH
                    } അല്ലെങ്കിൽ എങ്കിൽ (src_dyn == 3) {
                        m_കണ്ണി_dynamic_dims[കണ്ണി] = 3;  // N  -> N
                    }
                } അല്ലെങ്കിൽ {
                    എങ്കിൽ (src_dyn == 0) {
                        m_കണ്ണി_dynamic_dims[കണ്ണി] = 1;  // IW -> OW
                    } അല്ലെങ്കിൽ എങ്കിൽ (src_dyn == 2) {
                        m_കണ്ണി_dynamic_dims[കണ്ണി] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി] != -1) {
                    OPENVINO_ASSERT(കണ്ണി->src[1]->ne[src_dyn] == കണ്ണി->ne[m_കണ്ണി_dynamic_dims[കണ്ണി]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(കണ്ണി->name) +
                                        " and its src[1]: " + std::string(കണ്ണി->src[1]->name));
                }
            }
            നിർത്തുക;
        }
        അവസ്ഥ GGML_OP_IM2COL_3D: {
            m_കണ്ണി_dynamic_dims[കണ്ണി] = -1;
            എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി->src[1]] != -1) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ src_dyn = m_കണ്ണി_dynamic_dims[കണ്ണി->src[1]];
                എങ്കിൽ (src_dyn == 0) {
                    m_കണ്ണി_dynamic_dims[കണ്ണി] = 1;  // IW -> OW
                } അല്ലെങ്കിൽ എങ്കിൽ (src_dyn == 1) {
                    m_കണ്ണി_dynamic_dims[കണ്ണി] = 2;  // IH -> OH
                } അല്ലെങ്കിൽ എങ്കിൽ (src_dyn == 3) {
                    m_കണ്ണി_dynamic_dims[കണ്ണി] = 3;  // N  -> N
                }
                എങ്കിൽ (m_കണ്ണി_dynamic_dims[കണ്ണി] != -1) {
                    OPENVINO_ASSERT(കണ്ണി->src[1]->ne[src_dyn] == കണ്ണി->ne[m_കണ്ണി_dynamic_dims[കണ്ണി]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(കണ്ണി->name) +
                                        " and its src[1]: " + std::string(കണ്ണി->src[1]->name));
                }
            }
            നിർത്തുക;
        }
        സ്ഥിരസ്ഥിതി:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(കണ്ണി->op), കണ്ണി->name);
            നിർത്തുക;
        }
    };

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * കണ്ണി = m_cgraph->nodes[i];
        visit_കണ്ണി(visit_കണ്ണി, കണ്ണി);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    എങ്കിൽ (0) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * കണ്ണി = m_cgraph->nodes[i];
            പൂർണ്ണസംഖ്യ dynamic_dim = m_കണ്ണി_dynamic_dims[കണ്ണി];
            std::cout << "[" << i << "] " << "node_name: " << കണ്ണി->name << " op: " << ggml_op_name(കണ്ണി->op)
                      << " shape: [";
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 4; j++) {
                എങ്കിൽ (j == dynamic_dim) {
                    std::cout << "*";
                } അല്ലെങ്കിൽ {
                    std::cout << കണ്ണി->ne[j];
                }
                എങ്കിൽ (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = കണ്ണി->src[j];
                എങ്കിൽ (src == nullptr) {
                    തുടരുക;
                }
                പൂർണ്ണസംഖ്യ src_dynamic_dim = m_കണ്ണി_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < 4; k++) {
                    എങ്കിൽ (k == src_dynamic_dim) {
                        std::cout << "*";
                    } അല്ലെങ്കിൽ {
                        std::cout << src->ne[k];
                    }
                    എങ്കിൽ (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
