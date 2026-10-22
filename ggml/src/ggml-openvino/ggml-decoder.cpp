#ଅନ୍ତର୍ଭାଵ "ggml-decoder.h"

#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-openvino-extra.h"
#ଅନ୍ତର୍ଭାଵ "ggml-openvino.h"
#ଅନ୍ତର୍ଭାଵ "ggml-quants.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "utils.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cassert>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <fstream>
#ଅନ୍ତର୍ଭାଵ <iomanip>
#ଅନ୍ତର୍ଭାଵ <map>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <mutex>
#ଅନ୍ତର୍ଭାଵ <openvino/core/dimension.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/except.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/partial_shape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/type/bfloat16.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/type/element_type.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/type/float16.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/runtime/tensor.hpp>
#ଅନ୍ତର୍ଭାଵ <ostream>
#ଅନ୍ତର୍ଭାଵ <set>
#ଅନ୍ତର୍ଭାଵ <stdexcept>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             ତାର୍କିକ is_static,
                             ତାର୍କିକ is_stateful,
                             ତାର୍କିକ model_is_splitted,
                             ତାର୍କିକ is_prefill,
                             ପୂର୍ଣ୍ଣସଂଖ୍ୟା prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(ମିଥ୍ୟା),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    ସ୍ଥିର ତାର୍କିକ printed_address_map = ମିଥ୍ୟା;
    ଯଦି (!printed_address_map) {
        ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = ସତ୍ୟ;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_ନିବେଶ_output();
    compute_ଗଣ୍ଠି_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_n = 0; ଗଣ୍ଠି_n < cgraph->n_nodes; ଗଣ୍ଠି_n++) {
        m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_n].ଗଣ୍ଠି_op_case = compute_op_case(m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_n].ଗଣ୍ଠି);
        m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_n].ଗଣ୍ଠି_op_type = compute_op_type(m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_n].ଗଣ୍ଠି);
    }

    add_extra_inputs();
}

ଶୂନ୍ୟ GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_ଗଣ୍ଠି_info_list.clear();
    set_ନିବେଶ_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
ତାର୍କିକ GgmlOvDecoder::is_bound_to(ଅଚଳ ggml_cgraph * cgraph) ଅଚଳ {
    ଫେରନ୍ତୁ m_cgraph == cgraph && cgraph->n_nodes > 0 && m_ଗଣ୍ଠି_info_list.size() == (size_t) cgraph->n_nodes &&
           m_ଗଣ୍ଠି_info_list.front().ଗଣ୍ଠି == cgraph->nodes[0] &&
           m_ଗଣ୍ଠି_info_list.back().ଗଣ୍ଠି == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = ସତ୍ୟ;
    set_ନିବେଶ_output();
    compute_model_inputs();
    compute_model_outputs();
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_n = 0; ଗଣ୍ଠି_n < cgraph->n_nodes; ଗଣ୍ଠି_n++) {
        m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_n].ଗଣ୍ଠି_op_case = compute_op_case(m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_n].ଗଣ୍ଠି);
        m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_n].ଗଣ୍ଠି_op_type = compute_op_type(m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_n].ଗଣ୍ଠି);
    }
}

ନାମାକାଶ {
ତାର୍କିକ is_inplace_op(ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ଫେରନ୍ତୁ ଗଣ୍ଠି->op == GGML_OP_SET_ROWS || ଗଣ୍ଠି->op == GGML_OP_CPY || (ଗଣ୍ଠି->op == GGML_OP_SCALE && ଗଣ୍ଠି->view_src);
}

ତାର୍କିକ is_same_shape(ଅଚଳ ggml_tensor * a, ଅଚଳ ggml_tensor * b) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
        ଯଦି (a->ne[i] != b->ne[i]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ is_conv_states_all_tensor(ଅଚଳ ggml_tensor * tensor) {
    ଫେରନ୍ତୁ tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

ତାର୍କିକ is_full_single_slot_writeback(ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ଫେରନ୍ତୁ ଗଣ୍ଠି->view_src != nullptr && ଗଣ୍ଠି->view_src->ne[1] == 1 && ଗଣ୍ଠି->src[1] != nullptr &&
           ଗଣ୍ଠି->src[1]->op == GGML_OP_VIEW && ଗଣ୍ଠି->src[1]->view_src == ଗଣ୍ଠି->view_src &&
           ଗଣ୍ଠି->src[1]->view_offs == 0 && ggml_nbytes(ଗଣ୍ଠି->src[1]) == ggml_nbytes(ଗଣ୍ଠି->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
ତାର୍କିକ is_moe_expert_sum_add(ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    std::vector<ଅଚଳ ggml_tensor *> leaves;
    ଅଚଳ ggml_tensor * cur = ଗଣ୍ଠି;
    ଯେତେବେଳେ (cur->op == GGML_OP_ADD) {
        ଯଦି (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ଅଚଳ ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    ପାଇଁ (ଅଚଳ ggml_tensor * leaf : leaves) {
        ଯଦି (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଅଚଳ ggml_tensor * leaf_base = leaf->src[0];
        ଯଦି (base == nullptr) {
            base = leaf_base;
        } ଅନ୍ୟଥା ଯଦି (leaf_base != base) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        ଯଦି (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ଅଚଳ ggml_cgraph * cgraph, ଅଚଳ ggml_tensor * tensor) {
    ଯଦି (tensor == nullptr) {
        ଫେରନ୍ତୁ "";
    }
    ଯଦି ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ଅଚଳ auto * ଗଣ୍ଠି = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        ଯଦି (ଗଣ୍ଠି != cgraph->nodes + cgraph->n_nodes) {
            ଫେରନ୍ତୁ std::string(tensor->name) + "#n" + std::to_string(ଗଣ୍ଠି - cgraph->nodes);
        }
        ଅଚଳ auto * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        ଯଦି (leaf != cgraph->leafs + cgraph->n_leafs) {
            ଫେରନ୍ତୁ std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    ଫେରନ୍ତୁ tensor->name;
}

ସ୍ଥିର std::string get_tensor_ov_name(ଅଚଳ ggml_cgraph * cgraph, ଅଚଳ ggml_tensor * tensor) {
    ଫେରନ୍ତୁ GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

ସ୍ଥିର std::string get_tensor_graph_ନିବେଶ_ov_name(ଅଚଳ GgmlOvDecoder * decoder,
                                                  ଅଚଳ ggml_cgraph * cgraph,
                                                  ଅଚଳ ggml_tensor * tensor,
                                                  ଅଚଳ ggml_tensor * op) {
    ଯଦି (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        ଫେରନ୍ତୁ "inp_pos";
    }
    ଯଦି (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        ଫେରନ୍ତୁ "embd";
    }
    ଯଦି (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ଅଚଳ ତାର୍କିକ is_swa = decoder->is_swa_mask(tensor);
        ଯଦି (decoder->is_stateful()) {
            ଫେରନ୍ତୁ is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        ଯଦି (is_swa) {
            ଫେରନ୍ତୁ get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    ଫେରନ୍ତୁ get_tensor_ov_name(cgraph, tensor);
}

ଶୂନ୍ୟ GgmlOvDecoder::set_ନିବେଶ_output() {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_n = 0; ଗଣ୍ଠି_n < m_cgraph->n_nodes; ଗଣ୍ଠି_n++) {
        auto * ଗଣ୍ଠି = m_cgraph->nodes[ଗଣ୍ଠି_n];

        ଗଣ୍ଠି_Info current_ଗଣ୍ଠି_info;
        auto ଗଣ୍ଠି_name = get_tensor_ov_name(m_cgraph, ଗଣ୍ଠି);

        current_ଗଣ୍ଠି_info.ଗଣ୍ଠି = ଗଣ୍ଠି;
        current_ଗଣ୍ଠି_info.ଗଣ୍ଠି_name = ଗଣ୍ଠି_name;
        current_ଗଣ୍ଠି_info.ଗଣ୍ଠି_op_case = 0;
        current_ଗଣ୍ଠି_info.data_addr = ଗଣ୍ଠି->data;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ଗଣ୍ଠି->src[i];
            ଯଦି (src == nullptr) {
                ଜାରି;
            }
            auto src_name = get_tensor_ov_name(m_cgraph, src);
            ଯଦି (src->flags & GGML_TENSOR_FLAG_ନିବେଶ) {
                src_name = get_tensor_graph_ନିବେଶ_ov_name(ଏହି, m_cgraph, src, ଗଣ୍ଠି);
            }
            current_ଗଣ୍ଠି_info.ଗଣ୍ଠି_inputs[src_name] = src;
            current_ଗଣ୍ଠି_info.ଗଣ୍ଠି_inputs_names.push_back(src_name);

            ଯଦି (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_ଗଣ୍ଠି_info.ଗଣ୍ଠି_inputs_views[src_name])> view_chain;
                auto * current = src;

                ଯେତେବେଳେ (current != nullptr) {
                    auto current_name = get_tensor_ov_name(m_cgraph, current);
                    ଯଦି (current->flags & GGML_TENSOR_FLAG_ନିବେଶ) {
                        current_name = get_tensor_graph_ନିବେଶ_ov_name(ଏହି, m_cgraph, current, ଗଣ୍ଠି);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    ଯଦି (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } ଅନ୍ୟଥା {
                        ଭାଙ୍ଗ;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_ଗଣ୍ଠି_info.ଗଣ୍ଠି_inputs_views[src_name] = view_chain;
            }
        }

        m_ଗଣ୍ଠି_info_list.push_back(current_ଗଣ୍ଠି_info);
    }
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା GgmlOvDecoder::compute_op_case(ଅଚଳ ggml_tensor * ଗଣ୍ଠି) ଅଚଳ {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_case = 0;
    ବାଛ (ଗଣ୍ଠି->op) {
    ସ୍ଥିତି GGML_OP_RESHAPE: {
        ଯଦି (m_naive) {
            ଭାଙ୍ଗ;
        }
        auto name = std::string(ଗଣ୍ଠି->name);
        auto * src = ଗଣ୍ଠି->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        ତାର୍କିକ recurrent_sequence = ମିଥ୍ୟା;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ଅଚଳ auto * consumer = m_cgraph->nodes[i];
            ଯଦି (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == ଗଣ୍ଠି) {
                ଫେରନ୍ତୁ 1;
            } ଅନ୍ୟଥା ଯଦି (consumer->op == GGML_OP_SSM_CONV) {
                ଅଚଳ auto * concat = consumer->src[0];
                ଯଦି (concat->op == GGML_OP_CONCAT) {
                    ଅଚଳ auto * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == ଗଣ୍ଠି;
                }
            } ଅନ୍ୟଥା ଯଦି (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                ଅଚଳ auto * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == ଗଣ୍ଠି;
            }
        }
        ଯଦି (recurrent_sequence && ଗଣ୍ଠି->ne[0] == src->ne[0] && ଗଣ୍ଠି->ne[3] == 1) {
            ଫେରନ୍ତୁ 6;
        }
        ଯଦି (ଗଣ୍ଠି->ne[0] == src->ne[0] && ଗଣ୍ଠି->ne[2] == 1 && ଗଣ୍ଠି->ne[3] == 1) {
            ଫେରନ୍ତୁ 5;
        }
        ଯଦି (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == ଗଣ୍ଠି->ne[0] && src->src[0]->ne[1] == ଗଣ୍ଠି->ne[1]) {
            op_case = 4;
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->ne[0] * ଗଣ୍ଠି->ne[1] == src->ne[0]) {
            op_case = 1;
        } ଅନ୍ୟଥା ଯଦି (src->ne[0] * src->ne[1] == ଗଣ୍ଠି->ne[0]) {
            op_case = 2;
            ଯଦି (src->ne[2] * src->ne[3] == ଗଣ୍ଠି->ne[1]) {
                op_case = 5;
            }
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == ଗଣ୍ଠି->ne[1]) {
            op_case = 3;
        } ଅନ୍ୟଥା ଯଦି (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } ଅନ୍ୟଥା ଯଦି (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } ଅନ୍ୟଥା ଯଦି (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        ଯଦି (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            ଅଚଳ auto * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            ତାର୍କିକ recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ଅଚଳ auto * consumer = m_cgraph->nodes[i];
                ଯଦି (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j : {3, 4}) {
                        ଅଚଳ auto * gate = consumer->src[j];
                        ଯଦି (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == ଗଣ୍ଠି;
                    }
                } ଅନ୍ୟଥା ଯଦି (consumer->op == GGML_OP_MUL) {
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 2; ++j) {
                        ଅଚଳ auto * gate = consumer->src[j];
                        ଅଚଳ auto * norm = consumer->src[1 - j];
                        ଯଦି (gate->op != GGML_OP_UNARY || gate->src[0] != ଗଣ୍ଠି) {
                            ଜାରି;
                        }
                        ଯଦି (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            ଯଦି (recurrent) {
                op_case = 9;
            }
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_PERMUTE: {
        ଯଦି (ଗଣ୍ଠି->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(ଗଣ୍ଠି->view_src->name);
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା layer = extract_layer_from_name(src_name).value();
            ଯଦି (ggml_is_contiguous(ଗଣ୍ଠି->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                ଯଦି (!is_swa_layer(layer)) {
                    op_case = 3;
                } ଅନ୍ୟଥା {
                    op_case = 4;
                }
            } ଅନ୍ୟଥା {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                ଯଦି (!is_swa_layer(layer)) {
                    op_case = 5;
                } ଅନ୍ୟଥା {
                    op_case = 6;
                }
            }
        } ଅନ୍ୟଥା {
            // rope'ed query tensor
            op_case = 2;
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_MUL_MAT: {
        ଯଦି (ଗଣ୍ଠି->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_GET_ROWS: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        ଯଦି (ଗଣ୍ଠି->src[0]->op == GGML_OP_RESHAPE && ଗଣ୍ଠି->src[0]->src[0] != nullptr &&
            is_recurrent_cache(ଗଣ୍ଠି->src[0]->src[0])) {
            ଅଚଳ ତାର୍କିକ single_slot = ଗଣ୍ଠି->src[0]->src[0]->ne[1] == 1;
            ଯଦି (ଗଣ୍ଠି->src[1]->op == GGML_OP_VIEW) {
                op_case = (ଗଣ୍ଠି->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } ଅନ୍ୟଥା ଯଦି (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_ROPE: {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode = ଗଣ୍ଠି->op_params[2];
        ବାଛ (mode) {
        ସ୍ଥିତି GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            ଭାଙ୍ଗ;
        }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            op_case = 0;
            ଭାଙ୍ଗ;
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_VIEW: {
        ଯଦି (!m_model_params.has_rs_rollback && ଗଣ୍ଠି->src[0] != nullptr &&
            ଗଣ୍ଠି->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            ଭାଙ୍ଗ;
        }
        ଯଦି (m_is_static && ଗଣ୍ଠି->src[0] != nullptr &&
            (ଗଣ୍ଠି->src[0]->op == GGML_OP_GATED_DELTA_NET || ଗଣ୍ଠି->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            ଭାଙ୍ଗ;
        }
        ଯଦି (ଗଣ୍ଠି->src[0]->op == GGML_OP_VIEW) {
            auto * src = ଗଣ୍ଠି->src[0];
            ଯଦି (ggml_nelements(ଗଣ୍ଠି) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            ଯଦି (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            auto * src = ଗଣ୍ଠି->src[0];
            ଯଦି (ggml_nelements(ଗଣ୍ଠି) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                ଯଦି (ଗଣ୍ଠି->ne[0] == src->ne[0] && ଗଣ୍ଠି->ne[1] == src->ne[2] && ଗଣ୍ଠି->ne[2] == 1 &&
                    ଗଣ୍ଠି->nb[0] == src->nb[0] && ଗଣ୍ଠି->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    ଭାଙ୍ଗ;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା diff_ଗଣନା = 0;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
                    ଯଦି (ଗଣ୍ଠି->ne[i] != src->ne[i]) {
                        diff_ଗଣନା++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    ଯଦି (ଗଣ୍ଠି->ne[i] > src->ne[i]) {
                        ଫେରନ୍ତୁ 0;
                    }
                }
                ଯଦି (diff_ଗଣନା >= 1) {
                    op_case = 0;
                }
            }
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_RMS_NORM: {
        ଯଦି (ଗଣ୍ଠି->src[0]->op == GGML_OP_VIEW) {
            ଯଦି (is_same_shape(ଗଣ୍ଠି->src[0]->src[0], ଗଣ୍ଠି->src[0])) {
                op_case = 1;
            } ଅନ୍ୟଥା ଯଦି (!m_model_params.has_rs_rollback &&
                       ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_POOL_2D: {
        ଅଚଳ ggml_op_pool pool_mode = static_cast<ggml_op_pool>(ଗଣ୍ଠି->op_params[0]);
        ବାଛ (pool_mode) {
        ସ୍ଥିତି GGML_OP_POOL_MAX: {
            op_case = 1;
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_OP_POOL_AVG: {
            op_case = 2;
            ଭାଙ୍ଗ;
        }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            op_case = 0;
            ଭାଙ୍ଗ;
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_UPSCALE: {
        ଅଚଳ int32_t mode_flags = ଗଣ୍ଠି->op_params[0];
        ଅଚଳ ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        ବାଛ (scale_mode) {
        ସ୍ଥିତି GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            ଭାଙ୍ଗ;
        }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            op_case = 0;
            ଭାଙ୍ଗ;
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_CPY: {
        ଯଦି (ଗଣ୍ଠି->src[0]->op == GGML_OP_VIEW) {
            ଯଦି (ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                ଯଦି (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(ଗଣ୍ଠି) ? 7 : 10;
                } ଅନ୍ୟଥା {
                    op_case = 1;
                }
            } ଅନ୍ୟଥା ଯଦି (GgmlOvDecoder::is_conv_state_writeback(ଗଣ୍ଠି)) {
                op_case = is_full_single_slot_writeback(ଗଣ୍ଠି) ? 8 : 2;
                ଭାଙ୍ଗ;
            } ଅନ୍ୟଥା ଯଦି (is_conv_states_all_tensor(ଗଣ୍ଠି->view_src) && ଗଣ୍ଠି->src[1] != nullptr &&
                       ଗଣ୍ଠି->src[1]->op == GGML_OP_VIEW && ଗଣ୍ଠି->src[1]->view_src == ଗଣ୍ଠି->view_src) {
                op_case = 4;
                ଭାଙ୍ଗ;
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_GET_ROWS && ଗଣ୍ଠି->src[1] != nullptr &&
                       ଗଣ୍ଠି->src[1]->op == GGML_OP_VIEW && ଗଣ୍ଠି->src[1]->view_src != nullptr &&
                       is_recurrent_cache(ଗଣ୍ଠି->src[1]->view_src) && ଗଣ୍ଠି->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->src[0]->op == GGML_OP_GET_ROWS && ଗଣ୍ଠି->src[1] != nullptr &&
                   ଗଣ୍ଠି->src[1]->op == GGML_OP_VIEW && ଗଣ୍ଠି->src[1]->view_src != nullptr &&
                   is_recurrent_cache(ଗଣ୍ଠି->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = ଗଣ୍ଠି->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->src[1] != nullptr && ଗଣ୍ଠି->src[1]->op == GGML_OP_VIEW && ଗଣ୍ଠି->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ଅଚଳ ggml_tensor * kv_buf = ଗଣ୍ଠି->src[1]->view_src;
            ଯଦି (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m_cgraph->n_nodes; i++) {
                    ଅଚଳ ggml_tensor * n = m_cgraph->nodes[i];
                    ଯଦି (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        ଜାରି;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    ଯଦି ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        ଯଦି (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        ଭାଙ୍ଗ;
                    }
                }
            }
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_ADD: {
        ଯଦି (is_moe_expert_sum_add(ଗଣ୍ଠି)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_SCALE: {
        ଯଦି (ଗଣ୍ଠି->view_src && ଗଣ୍ଠି->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = ଗଣ୍ଠି->view_src->ne[1] == 1 ? 2 : 1;
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_L2_NORM: {
        ଯଦି (std::string(ଗଣ୍ଠି->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT: {
        ଯଦି (ଗଣ୍ଠି->src[1] != nullptr && ଗଣ୍ଠି->src[1]->op == GGML_OP_VIEW && ଗଣ୍ଠି->src[1]->view_src != nullptr) {
            ଅଚଳ ggml_tensor * kv_buf = ଗଣ୍ଠି->src[1]->view_src;
            ଯଦି (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (ଗଣ୍ଠି->src[3] != nullptr) ? 1 : 2;
            }
        }
        ଭାଙ୍ଗ;
    }
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        ଭାଙ୍ଗ;
    }
    ଫେରନ୍ତୁ op_case;
}

std::optional<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> extract_layer_from_name(ଅଚଳ std::string & name) {
    size_t pos1 = name.find("_l");
    ଯଦି (pos1 == std::string::npos) {
        ଫେରନ୍ତୁ std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    ଯଦି (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା layer = std::stoi(layer_str);
    ଫେରନ୍ତୁ layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_swa_window_from_mask(ଅଚଳ ggml_tensor * mask) {
    ଯଦି (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        ଫେରନ୍ତୁ -1;
    }
    ଯଦି (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t n_kv = mask->ne[0];
    ଅଚଳ int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    ପାଇଁ (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        ପାଇଁ (int64_t c = 0; c < n_kv; c++) {
            ଅଚଳ size_t i = (size_t) r * n_kv + c;
            ଅଚଳ ଦଶମିକ v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((ଅଚଳ ggml_fp16_t *) mask->data)[i]) :
                                                          ((ଅଚଳ ଦଶମିକ *) mask->data)[i];
            ଯଦି (v > -INFINITY) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    ଫେରନ୍ତୁ window > 0 ? (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, ତାର୍କିକ is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    auto get_attention_pattern_case = [](ଅଚଳ ggml_tensor * ଗଣ୍ଠି) -> ପୂର୍ଣ୍ଣସଂଖ୍ୟା {
        ଯଦି (ଗଣ୍ଠି == nullptr) {
            ଫେରନ୍ତୁ -1;
        }

        ବାଛ (ଗଣ୍ଠି->op) {
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
            ଯଦି (ଗଣ୍ଠି->src[0] == nullptr || ଗଣ୍ଠି->src[1] == nullptr) {
                ଫେରନ୍ତୁ -1;
            }
            ବାଛ (ଗଣ୍ଠି->src[1]->op) {
            ସ୍ଥିତି GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                ଯଦି (ଗଣ୍ଠି->src[3] != nullptr && ଗଣ୍ଠି->src[1]->src[0] != nullptr &&
                    ଗଣ୍ଠି->src[1]->src[0]->op == GGML_OP_VIEW) {
                    ଫେରନ୍ତୁ 0;
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                ଯଦି (ଗଣ୍ଠି->src[3] != nullptr && ଗଣ୍ଠି->src[1]->src[0] != nullptr &&
                    ଗଣ୍ଠି->src[1]->src[0]->op == GGML_OP_PERMUTE && ଗଣ୍ଠି->src[1]->src[0]->src[0] != nullptr &&
                    ଗଣ୍ଠି->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    ଫେରନ୍ତୁ 1;
                }
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                ଯଦି (ଗଣ୍ଠି->src[1]->view_src != nullptr) {
                    ଯଦି (ଗଣ୍ଠି->src[3] != nullptr) {
                        ଫେରନ୍ତୁ 4;  // decoder self-attention
                    }
                    ଫେରନ୍ତୁ 5;      // cross-attention or encoder self-attention
                }
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଭାଙ୍ଗ;
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            ଯଦି (ଗଣ୍ଠି->src[0] != nullptr && ଗଣ୍ଠି->src[1] != nullptr && ଗଣ୍ଠି->src[0]->op == GGML_OP_MUL_MAT &&
                ଗଣ୍ଠି->src[0]->src[0] != nullptr && ଗଣ୍ଠି->src[0]->src[1] != nullptr &&
                ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_PERMUTE && ଗଣ୍ଠି->src[0]->src[0]->src[0] != nullptr &&
                ଗଣ୍ଠି->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                ଫେରନ୍ତୁ 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            ଯଦି (ଗଣ୍ଠି->src[0]->op == GGML_OP_ADD && ଗଣ୍ଠି->src[0]->src[0] != nullptr &&
                ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_MUL_MAT && ଗଣ୍ଠି->src[0]->src[0]->src[0] != nullptr &&
                ଗଣ୍ଠି->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                ଫେରନ୍ତୁ 3;
            }
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଭାଙ୍ଗ;
        }

        ଫେରନ୍ତୁ -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    auto get_attention_op_mask = [&get_attention_pattern_case](ଅଚଳ ggml_tensor * ଗଣ୍ଠି) -> ଅଚଳ ggml_tensor * {
        ବାଛ (get_attention_pattern_case(ଗଣ୍ଠି)) {
        ସ୍ଥିତି 0:
        ସ୍ଥିତି 1:
            ଫେରନ୍ତୁ ଗଣ୍ଠି->src[3];
        ସ୍ଥିତି 2:
        ସ୍ଥିତି 3:
            ଫେରନ୍ତୁ ଗଣ୍ଠି->src[1];
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ nullptr;
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
        std::map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ଅଚଳ ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
            ଅଚଳ ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            ଯଦି (mask == nullptr) {
                ଜାରି;
            }
            ଅଚଳ ggml_tensor * cache_k_permute = nullptr;
            ବାଛ (get_attention_pattern_case(cgraph->nodes[i])) {
            ସ୍ଥିତି 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     ଭାଙ୍ଗ;
            ସ୍ଥିତି 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             ଭାଙ୍ଗ;
            ସ୍ଥିତି 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     ଭାଙ୍ଗ;
            }
            ଅଚଳ ggml_tensor * cache_k_view = cache_k_permute->src[0];
            ଯଦି (cache_k_view->op != GGML_OP_VIEW) {
                ଜାରି;
            }
            ଅଚଳ ggml_tensor * leaf = cache_k_view->src[0];
            auto layer = extract_layer_from_name(leaf->name);
            ଯଦି (!layer.has_value()) {
                ଜାରି;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        ପାଇଁ (ଅଚଳ auto & [layer, extent] : layer_extent) {
            ଯଦି (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                ଯଦି (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            ପାଇଁ (ଅଚଳ auto & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (long) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    ତାର୍କିକ rope_seen = ମିଥ୍ୟା;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା attention_pattern_case = get_attention_pattern_case(ଗଣ୍ଠି);
        ଯଦି (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            ବାଛ (attention_pattern_case) {
            ସ୍ଥିତି 0:
                cache_k_permute = ଗଣ୍ଠି->src[1];
                mask = ଗଣ୍ଠି->src[3];
                ଭାଙ୍ଗ;
            ସ୍ଥିତି 1:
                cache_k_permute = ଗଣ୍ଠି->src[1]->src[0];
                mask = ଗଣ୍ଠି->src[3];
                ଭାଙ୍ଗ;
            ସ୍ଥିତି 2:
                cache_k_permute = ଗଣ୍ଠି->src[0]->src[0];
                mask = ଗଣ୍ଠି->src[1];
                ଭାଙ୍ଗ;
            ସ୍ଥିତି 3:
                cache_k_permute = ଗଣ୍ଠି->src[0]->src[0]->src[0];
                mask = ଗଣ୍ଠି->src[1];
                ଭାଙ୍ଗ;
            ସ୍ଥିତି 4:
            ସ୍ଥିତି 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                auto * cache_k_view = ଗଣ୍ଠି->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = ଗଣ୍ଠି->src[0]->ne[1];
                ଯଦି (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } ଅନ୍ୟଥା {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                ଜାରି;
            }
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଭାଙ୍ଗ;
            }

            ପ୍ରତିଜ୍ଞା(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.ନିବେଶ_len = ଗଣ୍ଠି->src[0]->ne[1];
            compute_params.token_len_per_seq = ଗଣ୍ଠି->src[0]->ne[1];

            auto * cache_k_view = cache_k_permute->src[0];
            ଯଦି (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                ଜାରି;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ଅଚଳ ତାର୍କିକ layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            ଯଦି (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } ଅନ୍ୟଥା {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, ଆକାର(size_t));
            compute_params.seq_active_start = offset / seq_size;

            ଯଦି (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } ଅନ୍ୟଥା {
                compute_params.attention_size = mask->ne[0];
            }
            ଯଦି (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_MUL_MAT && ଗଣ୍ଠି->src[0]->op == GGML_OP_PERMUTE &&
            ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(ଗଣ୍ଠି->src[0]->view_src, ଗଣ୍ଠି->view_src)) {
            ଯଦି (ଗଣ୍ଠି->src[1]->op == GGML_OP_PERMUTE && ଗଣ୍ଠି->src[1]->src[0]->op == GGML_OP_VIEW &&
                ଗଣ୍ଠି->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = ଗଣ୍ଠି->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_TRANSPOSE && ଗଣ୍ଠି->src[0]->op == GGML_OP_PERMUTE &&
            ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = ଗଣ୍ଠି->ne[0];
            ଯଦି (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_ROPE) {
            ଯଦି (compute_params.token_len_per_seq == -1 && ଗଣ୍ଠି->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(ଗଣ୍ଠି->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(ଆକାର(model_params.rope_params) == ଆକାର(int32_t) * 16, "rope_params size");
            ଯଦି (!rope_seen) {
                memcpy(model_params.rope_params, ଗଣ୍ଠି->op_params, ଆକାର(int32_t) * 16);
                rope_seen = ସତ୍ୟ;
            } ଅନ୍ୟଥା ଯଦି (memcmp(model_params.rope_params, ଗଣ୍ଠି->op_params, ଆକାର(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = ସତ୍ୟ;
            }
        }
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = ଗଣ୍ଠି->src[0]->ne[0];
        }
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_SCALE && ଗଣ୍ଠି->view_src != nullptr && is_recurrent_cache(ଗଣ୍ଠି->view_src)) {
            ଯଦି (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = ଗଣ୍ଠି->view_src->ne[1];
            } ଅନ୍ୟଥା {
                GGML_ASSERT(model_params.n_rs_slots == ଗଣ୍ଠି->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(ଗଣ୍ଠି) / ଗଣ୍ଠି->view_src->ne[0];
            compute_params.cache_rs_reset_idx = ଗଣ୍ଠି->src[0]->view_offs / ଗଣ୍ଠି->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_CPY && ଗଣ୍ଠି->view_src != nullptr && is_recurrent_cache(ଗଣ୍ଠି->view_src) &&
            ଗଣ୍ଠି->src[1] != nullptr && ଗଣ୍ଠି->src[1]->op == GGML_OP_VIEW && ଗଣ୍ଠି->src[1]->view_src == ଗଣ୍ଠି->view_src) {
            ଅଚଳ ତାର୍କିକ is_conv = is_conv_state_writeback(ଗଣ୍ଠି);
            ଅଚଳ ତାର୍କିକ is_gdn = ଗଣ୍ଠି->src[0]->op == GGML_OP_VIEW && ଗଣ୍ଠି->src[0]->src[0] != nullptr &&
                                ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ଅଚଳ ତାର୍କିକ is_extra = ଗଣ୍ଠି->src[0]->op == GGML_OP_GET_ROWS;
            ଅଚଳ ତାର୍କିକ is_gdn_rollback = is_gdn && is_same_shape(ଗଣ୍ଠି->src[0], ଗଣ୍ଠି->src[1]);

            ଅଚଳ ggml_tensor * dest_view = ଗଣ୍ଠି->src[1];
            ଅଚଳ ggml_tensor * cache = ଗଣ୍ଠି->view_src;
            ଅଚଳ size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            ଯଦି (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = ସତ୍ୟ;
            }
            ଯଦି (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(ଗଣ୍ଠି)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (dest_view->view_offs / row_bytes);
                ଯଦି (is_conv) {
                    writeback.src_begin = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (ଗଣ୍ଠି->src[0]->view_offs / ଗଣ୍ଠି->src[0]->view_src->nb[0]);
                } ଅନ୍ୟଥା ଯଦି (is_gdn_rollback) {
                    writeback.src_begin = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (ଗଣ୍ଠି->src[0]->view_offs / ଗଣ୍ଠି->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, ଗଣ୍ଠି)] = writeback;
            }
            ଯଦି ((is_conv || is_gdn) && !is_full_single_slot_writeback(ଗଣ୍ଠି)) {
                compute_params.s_copy_active_slot_len = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) dest_view->ne[1];
            }
        }
    }
    ଯଦି (model_params.n_heads_kv == -1) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
            ଅଚଳ auto * ଗଣ୍ଠି = cgraph->nodes[i];
            ଅଚଳ ggml_tensor * mask = nullptr;
            ଯଦି (ଗଣ୍ଠି->op == GGML_OP_SOFT_MAX) {
                mask = ଗଣ୍ଠି->src[1];
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = ଗଣ୍ଠି->src[3];
            } ଅନ୍ୟଥା {
                ଜାରି;
            }
            ଯଦି (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_ନିବେଶ) ||
                ଗଣ୍ଠି->src[0] == nullptr) {
                ଜାରି;
            }
            model_params.is_cacheless_attn = ସତ୍ୟ;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.ନିବେଶ_len = ଗଣ୍ଠି->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.ନିବେଶ_len;
            ଭାଙ୍ଗ;
        }
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    ଯଦି (model_params.is_cacheless_attn) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
            ଅଚଳ auto * ଗଣ୍ଠି = cgraph->nodes[i];
            ଯଦି (ଗଣ୍ଠି->op == GGML_OP_GET_ROWS && is_output_idx(ଗଣ୍ଠି->src[1], ଗଣ୍ଠି)) {
                compute_params.output_len = ଗଣ୍ଠି->src[1]->ne[0];
                ଭାଙ୍ଗ;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    ଯଦି (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    ଫେରନ୍ତୁ {model_params, compute_params};
}

ଶୂନ୍ୟ GgmlOvDecoder::validate_cgraph() ଅଚଳ {
    ଯଦି (m_model_params.n_seq > 1 && m_is_static == ସତ୍ୟ) {
        ଫିଙ୍ଗ std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_ନିବେଶ_shape(ଅଚଳ ggml_tensor * op,
                                                      ଅଚଳ ggml_tensor * ନିବେଶ,
                                                      ପୂର୍ଣ୍ଣସଂଖ୍ୟା dynamic_dim_index) ଅଚଳ {
    ଯଦି (m_naive) {
        ଫେରନ୍ତୁ ନିବେଶ != nullptr ? ov::PartialShape{get_shape(ନିବେଶ)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape ନିବେଶ_shape;

    ଯଦି (is_inp_tok(ନିବେଶ, op) || is_inp_pos(ନିବେଶ, op)) {
        // tokens or positions
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        ଯଦି (m_is_static && is_inp_pos(ନିବେଶ, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        ନିବେଶ_shape = ov::PartialShape{1, 1, 1, len};

    } ଅନ୍ୟଥା ଯଦି (is_output_idx(ନିବେଶ, op)) {
        // output index
        ନିବେଶ_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } ଅନ୍ୟଥା ଯଦି (is_inp_mean(ନିବେଶ, op)) {
        ନିବେଶ_shape = m_is_static ? ov::PartialShape{1, 1, ନିବେଶ->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } ଅନ୍ୟଥା ଯଦି (is_inp_scale_rows(ନିବେଶ, op)) {
        ନିବେଶ_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } ଅନ୍ୟଥା ଯଦି (is_inp_mask(ନିବେଶ, op)) {
        // mask
        ଯଦି (m_is_static) {
            ନିବେଶ_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } ଅନ୍ୟଥା ଯଦି (m_is_stateful) {
            ନିବେଶ_shape = ov::PartialShape{1, 1, -1, -1};
        } ଅନ୍ୟଥା {
            ନିବେଶ_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } ଅନ୍ୟଥା ଯଦି (is_recurrent_cache(ନିବେଶ)) {
        ନିବେଶ_shape = ov::PartialShape{get_shape(ନିବେଶ)};
        ଯଦି (!m_is_static && !m_is_stateful && ନିବେଶ->ne[1] > 1) {
            ନିବେଶ_shape[2] = -1;
        }

    } ଅନ୍ୟଥା ଯଦି (is_kvcache(ନିବେଶ, op)) {
        // kvcache
        ନିବେଶ_shape = ov::PartialShape{get_shape(ନିବେଶ)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ଅଚଳ ତାର୍କିକ is_flat_kv = (ନିବେଶ->ne[1] == 1 && ନିବେଶ->ne[2] == 1 && ନିବେଶ->ne[3] == 1);
        ଯଦି (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            ନିବେଶ_shape[2] = -1;
        }
        ଯଦି (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            ପ୍ରତିଜ୍ଞା(ନିବେଶ_shape.size() == 4 && ନିବେଶ_shape[0] == 1 && ନିବେଶ_shape[1] == 1 &&
                   ନିବେଶ_shape[2].is_dynamic() && ନିବେଶ_shape[3].is_static());
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_heads_kv = get_n_heads_kv_for_tensor(ନିବେଶ);
            ପ୍ରତିଜ୍ଞା(n_heads_kv > 0 && ନିବେଶ_shape[3].get_length() % n_heads_kv == 0);
            ଅଚଳ int64_t combined_dim = ନିବେଶ_shape[3].get_length();  // n_heads_kv * head_size
            ଅଚଳ int64_t head_size = combined_dim / n_heads_kv;
            ନିବେଶ_shape = {ନିବେଶ_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } ଅନ୍ୟଥା ଯଦି (is_kv_idx(ନିବେଶ, op)) {
        // kv update index
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        ନିବେଶ_shape = ov::PartialShape{1, 1, 1, len};

    } ଅନ୍ୟଥା ଯଦି (is_inp_s_copy(ନିବେଶ, op) || is_s_copy_leaf(ନିବେଶ)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        ନିବେଶ_shape = m_is_static ? ov::PartialShape{get_shape(ନିବେଶ)} : ov::PartialShape{1, 1, 1, -1};

    } ଅନ୍ୟଥା {
        ନିବେଶ_shape = ov::PartialShape{get_shape(ନିବେଶ)};
    }
    ଯଦି (dynamic_dim_index != -1 && m_model_is_splitted) {
        ନିବେଶ_shape[3 - dynamic_dim_index] = -1;
    }
    ଯଦି (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_ନିବେଶ && op->src[1] == ନିବେଶ) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        ଯଦି (m_is_static) {
            ଅଚଳ int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            ନିବେଶ_shape[2] = seq_active;
            ନିବେଶ_shape[3] = seq_active;
        } ଅନ୍ୟଥା {
            ନିବେଶ_shape[2] = -1;
            ନିବେଶ_shape[3] = -1;
        }
    }
    ଫେରନ୍ତୁ ନିବେଶ_shape;
}

ତାର୍କିକ GgmlOvDecoder::is_s_copy_leaf(ଅଚଳ ggml_tensor * tensor) ଅଚଳ {
    ଯଦି (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m_cgraph->n_nodes; i++) {
        ଅଚଳ ggml_tensor * ଗଣ୍ଠି = m_cgraph->nodes[i];
        ଯଦି (ଗଣ୍ଠି->op != GGML_OP_GET_ROWS || ଗଣ୍ଠି->src[0] == nullptr || ଗଣ୍ଠି->src[1] == nullptr) {
            ଜାରି;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ଅଚଳ ggml_tensor * idx = ଗଣ୍ଠି->src[1];
        ଯେତେବେଳେ (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        ଯଦି (idx != tensor) {
            ଜାରି;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ଅଚଳ ggml_tensor * data = ଗଣ୍ଠି->src[0];
        ଯେତେବେଳେ (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        ଯଦି (data != nullptr && is_recurrent_cache(data)) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ଶୂନ୍ୟ GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_ନିବେଶ = [ଏହି](ଅଚଳ std::string & name, int64_t value, ତାର୍କିକ force_parameter = ମିଥ୍ୟା) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    ଯଦି (m_compute_params.attention_size != -1) {
        create_1d_ନିବେଶ("attention_size", m_compute_params.attention_size);
    }
    ଯଦି (m_compute_params.attention_size_static != -1) {
        create_1d_ନିବେଶ("attention_size_static", m_compute_params.attention_size_static);
    }
    ଯଦି (m_compute_params.attention_size_swa != -1) {
        create_1d_ନିବେଶ("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    ଯଦି (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_ନିବେଶ("swa_window", m_compute_params.swa_window);
    }
    create_1d_ନିବେଶ("n_seq_active", m_compute_params.n_seq_active);
    create_1d_ନିବେଶ("seq_active_start", m_compute_params.seq_active_start);
    create_1d_ନିବେଶ("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    ଯଦି (m_compute_params.token_len_per_seq != -1) {
        create_1d_ନିବେଶ("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    ଯଦି (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_ନିବେଶ("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/ସତ୍ୟ);
        create_1d_ନିବେଶ("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/ସତ୍ୟ);
    }

    ଯଦି (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_ନିବେଶ("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        ଯଦି (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_ନିବେଶ("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/ସତ୍ୟ);
        }
    }

    ପାଇଁ (ଅଚଳ auto & [ଗଣ୍ଠି_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_ନିବେଶ("rs_slot_begin_" + ଗଣ୍ଠି_name, writeback.slot_begin);
        ଯଦି (!m_is_static && writeback.src_begin >= 0) {
            create_1d_ନିବେଶ("rs_src_begin_" + ଗଣ୍ଠି_name, writeback.src_begin);
        }
    }
}

ତାର୍କିକ GgmlOvDecoder::ଗଣ୍ଠି_is_used_as_src(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) {
    ggml_tensor * ଗଣ୍ଠି = m_cgraph->nodes[ଗଣ୍ଠି_idx];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = ଗଣ୍ଠି_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_ଗଣ୍ଠି = m_cgraph->nodes[i];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ଯଦି (other_ଗଣ୍ଠି->src[j] == ଗଣ୍ଠି) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ଶୂନ୍ୟ GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ଗଣ୍ଠି = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_NONE && ଗଣ୍ଠି_is_used_as_src(i)) {
            std::string ଗଣ୍ଠି_name = get_tensor_ov_name(m_cgraph, ଗଣ୍ଠି);
            ଯଦି (m_model_weights.find(ଗଣ୍ଠି_name) == m_model_weights.end()) {
                m_inputs[ଗଣ୍ଠି_name] = ଗଣ୍ଠି;
                m_model_inputs[ଗଣ୍ଠି_name] = {get_ov_type(ଗଣ୍ଠି),
                                             get_graph_ନିବେଶ_shape(ଗଣ୍ଠି, nullptr, m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି])};
            }
            ଜାରି;
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ଗଣ୍ଠି->src[i];
            ଯଦି (src == nullptr) {
                ଜାରି;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            ଯଦି (src->flags & GGML_TENSOR_FLAG_ନିବେଶ) {
                src_name = get_tensor_graph_ନିବେଶ_ov_name(ଏହି, m_cgraph, src, ଗଣ୍ଠି);
            }
            ଯଦି (m_model_weights.find(src_name) != m_model_weights.end()) {
                ଜାରି;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            ଯଦି (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                ଜାରି;
            }

            ତାର୍କିକ is_intermediate_ଗଣ୍ଠି = ମିଥ୍ୟା;
            ପାଇଁ (ଅଚଳ auto & ଗଣ୍ଠି_info : m_ଗଣ୍ଠି_info_list) {
                ଯଦି (ଗଣ୍ଠି_info.ଗଣ୍ଠି == src) {
                    is_intermediate_ଗଣ୍ଠି = ସତ୍ୟ;
                    ଭାଙ୍ଗ;
                }
            }
            ଯଦି (is_intermediate_ଗଣ୍ଠି) {
                ଜାରି;
            }
            ଯଦି (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                ଜାରି;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            ଯଦି (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                ଯଦି (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            ଯେତେବେଳେ (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_ନିବେଶ_shape(ଗଣ୍ଠି, src, m_ଗଣ୍ଠି_dynamic_dims[src])};
        }
    }
}

ଶୂନ୍ୟ GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_n = 0; ଗଣ୍ଠି_n < m_cgraph->n_nodes; ଗଣ୍ଠି_n++) {
        auto * cur_ଗଣ୍ଠି = m_cgraph->nodes[ଗଣ୍ଠି_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        ଯଦି (cur_ଗଣ୍ଠି->op == GGML_OP_NONE || cur_ଗଣ୍ଠି->op == GGML_OP_VIEW || cur_ଗଣ୍ଠି->op == GGML_OP_RESHAPE) {
            ଜାରି;
        }
        ଯଦି (::is_inplace_op(cur_ଗଣ୍ଠି) && ggml_nbytes(cur_ଗଣ୍ଠି) == 0) {
            ଜାରି;
        }
        auto cur_ଗଣ୍ଠି_use_ଗଣନା = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_ଗଣ୍ଠି)];
        ଯଦି (cur_ଗଣ୍ଠି_use_ଗଣନା == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            ଯଦି (cur_ଗଣ୍ଠି != nullptr && ::is_inplace_op(cur_ଗଣ୍ଠି) && ggml_nbytes(cur_ଗଣ୍ଠି) > 0) {
                cur_ଗଣ୍ଠି = cur_ଗଣ୍ଠି->view_src;
            }
        } ଅନ୍ୟଥା {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_use_ଗଣନା = 0;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * ଗଣ୍ଠି = m_cgraph->nodes[i];
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
                    ଯଦି (ଗଣ୍ଠି->src[j] != ଖାଲି && ଗଣ୍ଠି->src[j] == cur_ଗଣ୍ଠି) {
                        ନିବେଶ_use_ଗଣନା++;
                    }
                }
            }
            ଯଦି (ନିବେଶ_use_ଗଣନା == cur_ଗଣ୍ଠି_use_ଗଣନା) {
                cur_ଗଣ୍ଠି = nullptr;
            }
        }
        ଯଦି (cur_ଗଣ୍ଠି != nullptr) {
            std::string cur_ଗଣ୍ଠି_name = get_tensor_ov_name(m_cgraph, cur_ଗଣ୍ଠି);
            m_model_outputs[cur_ଗଣ୍ଠି_name] = cur_ଗଣ୍ଠି;
            m_model_output_names.insert(cur_ଗଣ୍ଠି_name);
        }
    }
}

ଅଚଳ ggml_tensor * GgmlOvDecoder::get_tensor_used_op(ଅଚଳ ggml_tensor * tensor) ଅଚଳ {
    ଯଦି (tensor == nullptr) {
        ଫେରନ୍ତୁ nullptr;
    }
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m_cgraph->n_nodes; i++) {
        ଅଚଳ auto * ଗଣ୍ଠି = m_cgraph->nodes[i];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ଯଦି (ଗଣ୍ଠି->src[j] == tensor) {
                ଫେରନ୍ତୁ ଗଣ୍ଠି;
            }
        }
    }
    ଫେରନ୍ତୁ nullptr;
}

ଅଚଳ ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ଅଚଳ std::string & name) ଅଚଳ {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m_cgraph->n_nodes; i++) {
        ଅଚଳ auto * ଗଣ୍ଠି = m_cgraph->nodes[i];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ଅଚଳ auto * src = ଗଣ୍ଠି->src[j];
            ଯଦି (src == nullptr) {
                ଭାଙ୍ଗ;
            }
            ଯଦି (get_tensor_ov_name(m_cgraph, src) == name) {
                ଫେରନ୍ତୁ src;
            }
        }
    }
    ଫେରନ୍ତୁ nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() ଅଚଳ {
    std::map<std::string, std::string> kv_param_res_names;
    ପାଇଁ (ଅଚଳ auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    ଫେରନ୍ତୁ kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
ସ୍ଥିର ତାର୍କିକ is_mul_mat_id_expert_weight(ଅଚଳ ggml_tensor * ଗଣ୍ଠି, ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_index) {
    ଫେରନ୍ତୁ ଗଣ୍ଠି->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, ତାର୍କିକ naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_i = 0; ଗଣ୍ଠି_i < n_nodes; ଗଣ୍ଠି_i++) {
        auto * ଗଣ୍ଠି = nodes[ଗଣ୍ଠି_i];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ଗଣ୍ଠି->src[i];
            ଯଦି (src == nullptr) {
                ଜାରି;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            ଯଦି (is_rope_freqs_weight(base, ଗଣ୍ଠି)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            ଯଦି (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(ଗଣ୍ଠି, i)) {
                ଯଦି (model_weights.find(src_name) == model_weights.end()) {
                    auto weight_ଗଣ୍ଠି = create_weight_ଗଣ୍ଠି(base, naive);
                    weight_ଗଣ୍ଠି->set_friendly_name(src_name);
                    model_weights[src_name] = weight_ଗଣ୍ଠି;
                }
            }
        }
    }
    ଫେରନ୍ତୁ model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
ସ୍ଥିର std::mutex g_nonov_weight_cache_mutex;
ସ୍ଥିର std::unordered_map<ଅଚଳ ଶୂନ୍ୟ *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_i = 0; ଗଣ୍ଠି_i < cgraph->n_nodes; ଗଣ୍ଠି_i++) {
        auto * ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_i];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ଗଣ୍ଠି->src[i];
            ଯଦି (src == nullptr) {
                ଜାରି;
            }
            ଅଚଳ ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            ଯଦି (is_rope_freqs_weight(base, ଗଣ୍ଠି)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            ଯଦି (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    ଫେରନ୍ତୁ names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_ଗଣ୍ଠି(ggml_tensor * tensor, ତାର୍କିକ naive) {
    ଅଚଳ ତାର୍କିକ is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    ଯଦି (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        ଯଦି (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            ଯଦି (weight_extra->weight_ଗଣ୍ଠି) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                ଫେରନ୍ତୁ weight_extra->weight_ଗଣ୍ଠି;
            }
        } ଅନ୍ୟଥା ଯଦି (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            ଯଦି (quant_extra->weight_ଗଣ୍ଠି) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                ଫେରନ୍ତୁ quant_extra->weight_ଗଣ୍ଠି;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    ଯଦି (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        auto weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        auto weight_ଗଣ୍ଠି = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_ଗଣ୍ଠି->set_friendly_name(tensor->name);
        ଫେରନ୍ତୁ weight_ଗଣ୍ଠି;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ଅଚଳ ତାର୍କିକ cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    ଯଦି (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        auto it = g_nonov_weight_cache.find(tensor->data);
        ଯଦି (it != g_nonov_weight_cache.end()) {
            ଫେରନ୍ତୁ it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    ସ୍ଥିର ଅଚଳ std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    ଯଦି (weight_types.find(tensor->type) == weight_types.end()) {
        ଫିଙ୍ଗ std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    ଯଦି (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        ଯଦି (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } ଅନ୍ୟଥା {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } ଅନ୍ୟଥା {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_ଗଣ୍ଠି->set_friendly_name(tensor->name);
    ଯଦି (!is_ov_buffer) {
        ଯଦି (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            auto [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_ଗଣ୍ଠି);
            ଫେରନ୍ତୁ it->second;
        }
        ଫେରନ୍ତୁ ov_weight.weight_ଗଣ୍ଠି;
    }

    ggml_openvino_extra_base * extra;
    ଯଦି (ov_weight.is_quantized()) {
        extra = ନୂତନ ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_ଗଣ୍ଠି);
    } ଅନ୍ୟଥା {
        extra = ନୂତନ ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_ଗଣ୍ଠି);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    ଫେରନ୍ତୁ ov_weight.weight_ଗଣ୍ଠି;
}

ଶୂନ୍ୟ GgmlOvDecoder::dump_cgraph(ଅଚଳ ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    ଯଦି (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        ଫେରନ୍ତୁ;
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
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];

        // Get buffer type name
        ଅଚଳ ଅକ୍ଷର * buf_name = "none";
        ggml_backend_buffer_t buf = ଗଣ୍ଠି->view_src ? ଗଣ୍ଠି->view_src->buffer : ଗଣ୍ଠି->buffer;
        ଯଦି (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ଗଣ୍ଠି->ne[0] << ", "
             << std::setw(5) << ଗଣ୍ଠି->ne[1] << ", "
             << std::setw(5) << ଗଣ୍ଠି->ne[2] << ", "
             << std::setw(5) << ଗଣ୍ଠି->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(ଗଣ୍ଠି->op) << std::right << " "
             << std::left << std::setw(45) << ଗଣ୍ଠି->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << ଗଣ୍ଠି->nb[0] << ", "
             << std::setw(5) << ଗଣ୍ଠି->nb[1] << ", "
             << std::setw(5) << ଗଣ୍ଠି->nb[2] << ", "
             << std::setw(5) << ଗଣ୍ଠି->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            ଯଦି (auto* src = ଗଣ୍ଠି->src[i]) {
                // Get buffer type name for source
                ଅଚଳ ଅକ୍ଷର * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                ଯଦି (src_buf) {
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
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * ଗଣ୍ଠି = cgraph->leafs[i];

        // Get buffer type name for leaf
        ଅଚଳ ଅକ୍ଷର * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = ଗଣ୍ଠି->view_src ? ଗଣ୍ଠି->view_src->buffer : ଗଣ୍ଠି->buffer;
        ଯଦି (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ଗଣ୍ଠି->ne[0] << ", "
             << std::setw(5) << ଗଣ୍ଠି->ne[1] << "] "
             << std::setw(8) << ggml_op_name(ଗଣ୍ଠି->op) << " "
             << std::setw(16) << ggml_get_name(ଗଣ୍ଠି)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

ଶୂନ୍ୟ print_tensor_address_map(ଅଚଳ ggml_cgraph * cgraph) {
    std::map<ଶୂନ୍ୟ *, std::vector<std::string>> address_map;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_n = 0; ଗଣ୍ଠି_n < cgraph->n_nodes; ଗଣ୍ଠି_n++) {
        auto * ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_n];
        ଯଦି (ଗଣ୍ଠି->data) {
            auto it = address_map.find(ଗଣ୍ଠି->data);
            ଯଦି (it == address_map.end()) {
                address_map[ଗଣ୍ଠି->data] = std::vector<std::string>();
            }
            address_map[ଗଣ୍ଠି->data].push_back(ଗଣ୍ଠି->name);
        }
    }
    ପାଇଁ (ଅଚଳ auto & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        ପାଇଁ (ଅଚଳ auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ଅଚଳ ggml_tensor * tensor) {
    std::vector<size_t> shape;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    ଫେରନ୍ତୁ shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(ଅଚଳ ggml_tensor * tensor) {
    std::vector<size_t> stride;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    ଫେରନ୍ତୁ stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(ଅଚଳ ggml_tensor * tensor) {
    ବାଛ (tensor->type) {
    ସ୍ଥିତି GGML_TYPE_F64:
        ଫେରନ୍ତୁ ov::element::f64;
    ସ୍ଥିତି GGML_TYPE_F32:
        ଫେରନ୍ତୁ ov::element::f32;
    ସ୍ଥିତି GGML_TYPE_F16:
        ଫେରନ୍ତୁ ov::element::f16;
    ସ୍ଥିତି GGML_TYPE_BF16:
        ଫେରନ୍ତୁ ov::element::bf16;
    ସ୍ଥିତି GGML_TYPE_I8:
        ଫେରନ୍ତୁ ov::element::i8;
    ସ୍ଥିତି GGML_TYPE_I16:
        ଫେରନ୍ତୁ ov::element::i16;
    ସ୍ଥିତି GGML_TYPE_I32:
        ଫେରନ୍ତୁ ov::element::i32;
    ସ୍ଥିତି GGML_TYPE_I64:
        ଫେରନ୍ତୁ ov::element::i64;
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        ଫେରନ୍ତୁ ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_ନିବେଶ_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ {
    ଫେରନ୍ତୁ ov::PartialShape(get_shape(m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_ନିବେଶ_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ {
    ଫେରନ୍ତୁ get_stride(m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_ନିବେଶ_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଫେରନ୍ତୁ it->second.size();
    }
    ଫେରନ୍ତୁ 0;
}

size_t GgmlOvDecoder::get_view_ନିବେଶ_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଯଦି (view_index < it->second.size()) {
            ଫେରନ୍ତୁ it->second[view_index].second->view_offs;
        }
    }
    ଫେରନ୍ତୁ 0;
}

size_t GgmlOvDecoder::get_view_ନିବେଶ_src_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଯଦି (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            ଯଦି (view_tensor && view_tensor->src[0]) {
                ଫେରନ୍ତୁ view_tensor->src[0]->view_offs;
            }
        }
    }
    ଫେରନ୍ତୁ 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_ନିବେଶ_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                         ଅଚଳ std::string & name,
                                                         size_t view_index) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଯଦି (view_index < it->second.size()) {
            ଫେରନ୍ତୁ get_stride(it->second[view_index].second);
        }
    }
    ଫେରନ୍ତୁ {};
}

std::vector<size_t> GgmlOvDecoder::get_view_ନିବେଶ_src_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                             ଅଚଳ std::string & name,
                                                             size_t view_index) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଯଦି (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            ଯଦି (view_tensor && view_tensor->src[0]) {
                ଫେରନ୍ତୁ get_stride(view_tensor->src[0]);
            }
        }
    }
    ଫେରନ୍ତୁ {};
}

ov::Shape GgmlOvDecoder::get_view_ନିବେଶ_ggml_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଯଦି (view_index < it->second.size()) {
            ଫେରନ୍ତୁ get_shape(it->second[view_index].second);
        }
    }
    ଫେରନ୍ତୁ {};
}

ov::Shape GgmlOvDecoder::get_view_ନିବେଶ_src_ggml_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                       ଅଚଳ std::string & name,
                                                       size_t view_index) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଯଦି (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            ଯଦି (view_tensor && view_tensor->src[0]) {
                ଫେରନ୍ତୁ get_shape(view_tensor->src[0]);
            }
        }
    }
    ଫେରନ୍ତୁ {};
}

ov::PartialShape GgmlOvDecoder::get_view_ନିବେଶ_ov_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                        ଅଚଳ std::string & name,
                                                        size_t view_index) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଯଦି (view_index < it->second.size()) {
            auto * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            auto dynamic_it = m_ଗଣ୍ଠି_dynamic_dims.find(tensor);
            ଯଦି (dynamic_it != m_ଗଣ୍ଠି_dynamic_dims.end() && dynamic_it->second != -1) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            ଫେରନ୍ତୁ shape;
        }
    }
    ଫେରନ୍ତୁ {};
}

ov::PartialShape GgmlOvDecoder::get_view_ନିବେଶ_src_ov_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                            ଅଚଳ std::string & name,
                                                            size_t view_index) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଯଦି (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            ଯଦି (view_tensor && view_tensor->src[0]) {
                auto * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                auto dynamic_it = m_ଗଣ୍ଠି_dynamic_dims.find(src_tensor);
                ଯଦି (dynamic_it != m_ଗଣ୍ଠି_dynamic_dims.end() && dynamic_it->second != -1) {
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dynamic_dim_index = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
                }

                ଫେରନ୍ତୁ shape;
            }
        }
    }
    ଫେରନ୍ତୁ {};
}

std::string GgmlOvDecoder::get_view_ନିବେଶ_name(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଯଦି (view_index < it->second.size()) {
            ଫେରନ୍ତୁ it->second[view_index].first;
        }
    }
    ଫେରନ୍ତୁ "";
}

std::string GgmlOvDecoder::get_view_ନିବେଶ_src_name(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.find(name);
    ଯଦି (it != m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_views.end()) {
        ଯଦି (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            ଯଦି (view_tensor && view_tensor->src[0]) {
                ଫେରନ୍ତୁ get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    ଫେରନ୍ତୁ "";
}

ov::element::Type GgmlOvDecoder::get_ନିବେଶ_type(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ {
    ଫେରନ୍ତୁ get_ov_type(m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs.at(name));
}

size_t GgmlOvDecoder::get_ନିବେଶ_size() ଅଚଳ {
    ଫେରନ୍ତୁ m_model_inputs.size();
}

size_t GgmlOvDecoder::get_ନିବେଶ_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    ଫେରନ୍ତୁ m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_ନିବେଶ_names(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    ଫେରନ୍ତୁ m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    auto * ggml_tensor = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି;
    ଫେରନ୍ତୁ ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    ଫେରନ୍ତୁ get_ov_type(m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    auto * ggml_tensor = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି;
    ଫେରନ୍ତୁ get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    auto * ଗଣ୍ଠି = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି;
    ଯଦି (ଗଣ୍ଠି->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = ଗଣ୍ଠି_idx + 1; i < m_cgraph->n_nodes; i++) {
            auto * consumer = m_cgraph->nodes[i];
            ଯଦି (consumer->op != GGML_OP_VIEW || consumer->src[0] != ଗଣ୍ଠି) {
                ଜାରି;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            auto & name = consumer->view_offs == 0 ? attn_name : state_name;
            ଯଦି (!name.empty()) {
                ଫେରନ୍ତୁ {m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        ଯଦି (!attn_name.empty() && !state_name.empty()) {
            ଫେରନ୍ତୁ {attn_name, state_name};
        }
    }
    ଫେରନ୍ତୁ {m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    auto * ଗଣ୍ଠି = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି;
    ଯଦି (!::is_inplace_op(ଗଣ୍ଠି) || ଗଣ୍ଠି->view_src == nullptr || ggml_nbytes(ଗଣ୍ଠି) == 0) {
        ଫେରନ୍ତୁ "";
    }
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_case = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_op_case;
    ଯଦି (ଗଣ୍ଠି->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        ଫେରନ୍ତୁ "";
    }
    ଫେରନ୍ତୁ get_tensor_ov_name(m_cgraph, ଗଣ୍ଠି->view_src);
}

ତାର୍କିକ GgmlOvDecoder::is_view_like_alias_of(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & view_src_name) ଅଚଳ {
    auto * ଗଣ୍ଠି = m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି;
    ଯଦି (ଗଣ୍ଠି->view_src == nullptr || get_tensor_ov_name(m_cgraph, ଗଣ୍ଠି->view_src) != view_src_name) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ ଗଣ୍ଠି->op == GGML_OP_RESHAPE || ଗଣ୍ଠି->op == GGML_OP_VIEW;
}

ଅଚଳ std::string & GgmlOvDecoder::get_op_name() ଅଚଳ {
    ସ୍ଥିର ଅଚଳ std::string unknown_name = "UNKNOWN_OP_NAME";
    ଫେରନ୍ତୁ unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    auto it = m_ଗଣ୍ଠି_dynamic_dims.find(m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି);
    ଯଦି (it == m_ଗଣ୍ଠି_dynamic_dims.end()) {
        ଫେରନ୍ତୁ -1;
    }
    ଫେରନ୍ତୁ it->second;
}

ଅଚଳ std::string & GgmlOvDecoder::get_op_name(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    ଫେରନ୍ତୁ m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_name;
}

int32_t * GgmlOvDecoder::get_ନିବେଶ_op_params(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ {
    ଫେରନ୍ତୁ m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    ଫେରନ୍ତୁ m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି->op_params;
}

size_t GgmlOvDecoder::get_output_op_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    ଫେରନ୍ତୁ m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି->view_offs;
}

ଶୂନ୍ୟ GgmlOvDecoder::visit_subgraph(std::function<ଶୂନ୍ୟ(std::shared_ptr<GgmlDecoder>, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx)> ଗଣ୍ଠି_visitor) ଅଚଳ {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx = 0; ଗଣ୍ଠି_idx < m_cgraph->n_nodes; ଗଣ୍ଠି_idx++) {
        ଯଦି (m_cgraph->nodes[ଗଣ୍ଠି_idx]->op == GGML_OP_NONE) {
            ଜାରି;
        }
        ଗଣ୍ଠି_visitor(std::make_shared<GgmlOvDecoder>(*ଏହି), ଗଣ୍ଠି_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ବାଛ (ଗଣ୍ଠି->op) {
    ସ୍ଥିତି GGML_OP_UNARY:
        ଫେରନ୍ତୁ std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(ଗଣ୍ଠି));
    ସ୍ଥିତି GGML_OP_GLU:
        ଫେରନ୍ତୁ std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(ଗଣ୍ଠି));
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        ଫେରନ୍ତୁ std::string("GGML_OP_") + ggml_op_name(ଗଣ୍ଠି->op);
    }
}

ଅଚଳ std::string & GgmlOvDecoder::get_op_type(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ {
    ଫେରନ୍ତୁ m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_op_type;
}

ଅଚଳ std::string & GgmlOvDecoder::get_op_type() ଅଚଳ {
    ସ୍ଥିର ଅଚଳ std::string unknown_op = "UNKNOWN_GGML_OP";
    ଫେରନ୍ତୁ unknown_op;
}

ଶୂନ୍ୟ GgmlOvDecoder::compute_ଗଣ୍ଠି_dynamic_dims() {
    auto visit_ଗଣ୍ଠି = [&](auto && self, ggml_tensor * ଗଣ୍ଠି) -> ଶୂନ୍ୟ {
        ଯଦି (!ଗଣ୍ଠି) {
            ଫେରନ୍ତୁ;
        }

        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_CPY) {
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
        }

        ଯଦି (m_ଗଣ୍ଠି_dynamic_dims.count(ଗଣ୍ଠି)) {
            ଫେରନ୍ତୁ;
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = ଗଣ୍ଠି->src[i];
            ଯଦି (src == nullptr) {
                ଜାରି;
            }
            ସଂରଚନା ggml_tensor * ମୂଳ_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            ଯଦି (ମୂଳ_src) {
                ଯଦି (is_inp_tok(ମୂଳ_src, ଗଣ୍ଠି) || is_inp_pos(ମୂଳ_src, ଗଣ୍ଠି) || is_output_idx(ମୂଳ_src, ଗଣ୍ଠି)) {
                    m_ଗଣ୍ଠି_dynamic_dims[ମୂଳ_src] = 0;
                    m_ଗଣ୍ଠି_dynamic_dims[src] = m_ଗଣ୍ଠି_dynamic_dims[ମୂଳ_src];
                    ଜାରି;
                }
                self(self, ମୂଳ_src);
                m_ଗଣ୍ଠି_dynamic_dims[src] = m_ଗଣ୍ଠି_dynamic_dims[ମୂଳ_src];
            } ଅନ୍ୟଥା {
                ଯଦି (is_inp_tok(src, ଗଣ୍ଠି) || is_inp_pos(src, ଗଣ୍ଠି) || is_output_idx(src, ଗଣ୍ଠି)) {
                    m_ଗଣ୍ଠି_dynamic_dims[src] = 0;
                    ଜାରି;
                }
                ଯଦି (is_inp_scale_rows(src, ଗଣ୍ଠି)) {
                    m_ଗଣ୍ଠି_dynamic_dims[src] = 1;
                    ଜାରି;
                }
                ଯଦି (ଗଣ୍ଠି->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_ଗଣ୍ଠି_dynamic_dims[src] = 1;
                    ଜାରି;
                }
                self(self, src);
            }
        }
        ବାଛ (ଗଣ୍ଠି->op) {
        ସ୍ଥିତି GGML_OP_NONE:
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GET_ROWS:
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[1]] != -1) {
                auto dynamic_dim_idx = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[1]];
                ଯଦି (dynamic_dim_idx == 0) {
                    m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = 1;
                } ଅନ୍ୟଥା {
                    auto dynamic_dim_stride = ଗଣ୍ଠି->src[1]->nb[dynamic_dim_idx] / ggml_type_size(ଗଣ୍ଠି->src[1]->type) *
                                              ggml_type_size(ଗଣ୍ଠି->src[0]->type);
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
                        ଯଦି (dynamic_dim_stride == ଗଣ୍ଠି->src[0]->nb[i]) {
                            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = i;
                            ଭାଙ୍ଗ;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL:
        ସ୍ଥିତି GGML_OP_MUL_MAT:
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]] != -1) {
                m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]];
            }
            ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[1]] != -1) {
                m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[1]];
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_PERMUTE:
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]] != -1) {
                auto dynamic_dim_idx = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
                    ଯଦି (ଗଣ୍ଠି->op_params[i] == dynamic_dim_idx) {
                        m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = i;
                        ଭାଙ୍ଗ;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]] != -1) {
                ଯଦି (ଗଣ୍ଠି->src[0]->op == GGML_OP_NONE) {
                    m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]];
                    ଭାଙ୍ଗ;
                }
                auto dynamic_dim_idx = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]];
                auto dynamic_dim_ମୂଲ୍ୟ = ଗଣ୍ଠି->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    ଗଣ୍ଠି->src[0]->nb[dynamic_dim_idx] / ggml_type_size(ଗଣ୍ଠି->src[0]->type) * ggml_type_size(ଗଣ୍ଠି->type);
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
                    ଯଦି (ଗଣ୍ଠି->nb[i] == dynamic_dim_stride) {
                        m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = i;
                        ଭାଙ୍ଗ;
                    }
                }
                ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] != -1 && dynamic_dim_ମୂଲ୍ୟ != ଗଣ୍ଠି->ne[m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି]]) {
                    m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    ଯଦି (ggml_nelements(ଗଣ୍ଠି) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      ଗଣ୍ଠି->name, ଗଣ୍ଠି->src[0]->name);
                    }
                }
            }
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_OP_TRANSPOSE:
        ସ୍ଥିତି GGML_OP_RESHAPE: {
            ଯଦି (is_same_shape(ଗଣ୍ଠି->src[0], ଗଣ୍ଠି)) {
                m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]];
                ଭାଙ୍ଗ;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]] != -1) {
                auto dynamic_dim_idx = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]];
                auto dynamic_dim_stride = ଗଣ୍ଠି->src[0]->nb[dynamic_dim_idx];
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
                    ଯଦି (ଗଣ୍ଠି->nb[i] == dynamic_dim_stride && ଗଣ୍ଠି->ne[i] == ଗଣ୍ଠି->src[0]->ne[dynamic_dim_idx]) {
                        m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = i;
                        ଭାଙ୍ଗ;
                    }
                }
                ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", ଗଣ୍ଠି->name);
                }
            }
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]] != -1) {
                auto q_dynamic_dim = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]];
                m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = q_to_out[q_dynamic_dim];
            }
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_OP_CONT:
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]] != -1) {
                auto dynamic_dim_idx = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]];
                ଯଦି (ggml_are_same_shape(ଗଣ୍ଠି, ଗଣ୍ଠି->src[0])) {
                    m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = dynamic_dim_idx;
                } ଅନ୍ୟଥା {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(ଗଣ୍ଠି->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (ଗଣ୍ଠି->src[0]->ne[0] / ggml_blck_size(ଗଣ୍ଠି->src[0]->type));
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * ଗଣ୍ଠି->src[0]->ne[i - 1];
                    }

                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(ଗଣ୍ଠି->src[0]->type) *
                                              ggml_type_size(ଗଣ୍ଠି->type);
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା matched_dim_ଗଣନା = 0;
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା first_matched_dim = -1;
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
                        ଯଦି (ଗଣ୍ଠି->nb[i] == dynamic_dim_stride && ଗଣ୍ଠି->ne[i] == ଗଣ୍ଠି->src[0]->ne[dynamic_dim_idx]) {
                            ଯଦି (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = i;
                            matched_dim_ଗଣନା++;
                        }
                    }
                    ଯଦି (matched_dim_ଗଣନା > 1 && ଗଣ୍ଠି->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = first_matched_dim;
                    } ଅନ୍ୟଥା ଯଦି (matched_dim_ଗଣନା != 1) {
                        m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      ଗଣ୍ଠି->name, ଗଣ୍ଠି->src[0]->name);
                    }
                }
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONCAT:
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
                ଯଦି (ଗଣ୍ଠି->src[0]->ne[i] != ଗଣ୍ଠି->ne[i]) {
                    m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = i;
                    ଭାଙ୍ଗ;
                }
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SSM_CONV:
        ସ୍ଥିତି GGML_OP_GATED_DELTA_NET:
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = 1;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RMS_NORM:
        ସ୍ଥିତି GGML_OP_L2_NORM:
        ସ୍ଥିତି GGML_OP_NORM:
        ସ୍ଥିତି GGML_OP_ADD:
        ସ୍ଥିତି GGML_OP_SUB:
        ସ୍ଥିତି GGML_OP_GLU:
        ସ୍ଥିତି GGML_OP_ROPE:
        ସ୍ଥିତି GGML_OP_SCALE:
        ସ୍ଥିତି GGML_OP_SOFT_MAX:
        ସ୍ଥିତି GGML_OP_ARGSORT:
        ସ୍ଥିତି GGML_OP_ADD_ID:
        ସ୍ଥିତି GGML_OP_UNARY:
        ସ୍ଥିତି GGML_OP_CUMSUM:
        ସ୍ଥିତି GGML_OP_FILL:
        ସ୍ଥିତି GGML_OP_SET:
        ସ୍ଥିତି GGML_OP_DIAG:
        ସ୍ଥିତି GGML_OP_TRI:
        ସ୍ଥିତି GGML_OP_REPEAT:
        ସ୍ଥିତି GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        ସ୍ଥିତି GGML_OP_DIV:
        ସ୍ଥିତି GGML_OP_CLAMP:
        ସ୍ଥିତି GGML_OP_PAD:
        ସ୍ଥିତି GGML_OP_UPSCALE:
        ସ୍ଥିତି GGML_OP_SIN:
        ସ୍ଥିତି GGML_OP_COS:
        ସ୍ଥିତି GGML_OP_LOG:
        ସ୍ଥିତି GGML_OP_ROLL:
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]];
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] =
                (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]] == 0) ? -1 : m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[0]];
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
        ସ୍ଥିତି GGML_OP_SOLVE_TRI:
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[1]];
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CPY:
        ସ୍ଥିତି GGML_OP_SET_ROWS:
        ସ୍ଥିତି GGML_OP_SUM:
        ସ୍ଥିତି GGML_OP_MEAN:
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_IM2COL: {
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[1]] != -1) {
                ଅଚଳ ତାର୍କିକ is_2D = ଗଣ୍ଠି->op_params[6] == 1;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_dyn = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[1]];
                ଯଦି (is_2D) {
                    ଯଦି (src_dyn == 0) {
                        m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = 1;  // IW -> OW
                    } ଅନ୍ୟଥା ଯଦି (src_dyn == 1) {
                        m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = 2;  // IH -> OH
                    } ଅନ୍ୟଥା ଯଦି (src_dyn == 3) {
                        m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = 3;  // N  -> N
                    }
                } ଅନ୍ୟଥା {
                    ଯଦି (src_dyn == 0) {
                        m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = 1;  // IW -> OW
                    } ଅନ୍ୟଥା ଯଦି (src_dyn == 2) {
                        m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] != -1) {
                    OPENVINO_ASSERT(ଗଣ୍ଠି->src[1]->ne[src_dyn] == ଗଣ୍ଠି->ne[m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(ଗଣ୍ଠି->name) +
                                        " and its src[1]: " + std::string(ଗଣ୍ଠି->src[1]->name));
                }
            }
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି GGML_OP_IM2COL_3D: {
            m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = -1;
            ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[1]] != -1) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_dyn = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି->src[1]];
                ଯଦି (src_dyn == 0) {
                    m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = 1;  // IW -> OW
                } ଅନ୍ୟଥା ଯଦି (src_dyn == 1) {
                    m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = 2;  // IH -> OH
                } ଅନ୍ୟଥା ଯଦି (src_dyn == 3) {
                    m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] = 3;  // N  -> N
                }
                ଯଦି (m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି] != -1) {
                    OPENVINO_ASSERT(ଗଣ୍ଠି->src[1]->ne[src_dyn] == ଗଣ୍ଠି->ne[m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(ଗଣ୍ଠି->name) +
                                        " and its src[1]: " + std::string(ଗଣ୍ଠି->src[1]->name));
                }
            }
            ଭାଙ୍ଗ;
        }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(ଗଣ୍ଠି->op), ଗଣ୍ଠି->name);
            ଭାଙ୍ଗ;
        }
    };

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ଗଣ୍ଠି = m_cgraph->nodes[i];
        visit_ଗଣ୍ଠି(visit_ଗଣ୍ଠି, ଗଣ୍ଠି);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    ଯଦି (0) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * ଗଣ୍ଠି = m_cgraph->nodes[i];
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା dynamic_dim = m_ଗଣ୍ଠି_dynamic_dims[ଗଣ୍ଠି];
            std::cout << "[" << i << "] " << "node_name: " << ଗଣ୍ଠି->name << " op: " << ggml_op_name(ଗଣ୍ଠି->op)
                      << " shape: [";
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 4; j++) {
                ଯଦି (j == dynamic_dim) {
                    std::cout << "*";
                } ଅନ୍ୟଥା {
                    std::cout << ଗଣ୍ଠି->ne[j];
                }
                ଯଦି (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = ଗଣ୍ଠି->src[j];
                ଯଦି (src == nullptr) {
                    ଜାରି;
                }
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_dynamic_dim = m_ଗଣ୍ଠି_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < 4; k++) {
                    ଯଦି (k == src_dynamic_dim) {
                        std::cout << "*";
                    } ଅନ୍ୟଥା {
                        std::cout << src->ne[k];
                    }
                    ଯଦି (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
