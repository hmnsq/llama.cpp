#অন্তর্ভাৱ "ggml-decoder.h"

#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml-openvino-extra.h"
#অন্তর্ভাৱ "ggml-openvino.h"
#অন্তর্ভাৱ "ggml-quants.h"
#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "utils.h"

#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <cassert>
#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <cstdlib>
#অন্তর্ভাৱ <fstream>
#অন্তর্ভাৱ <iomanip>
#অন্তর্ভাৱ <map>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <mutex>
#অন্তর্ভাৱ <openvino/core/dimension.hpp>
#অন্তর্ভাৱ <openvino/core/except.hpp>
#অন্তর্ভাৱ <openvino/core/node.hpp>
#অন্তর্ভাৱ <openvino/core/partial_shape.hpp>
#অন্তর্ভাৱ <openvino/core/type/bfloat16.hpp>
#অন্তর্ভাৱ <openvino/core/type/element_type.hpp>
#অন্তর্ভাৱ <openvino/core/type/float16.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/convert.hpp>
#অন্তর্ভাৱ <openvino/runtime/tensor.hpp>
#অন্তর্ভাৱ <ostream>
#অন্তর্ভাৱ <set>
#অন্তর্ভাৱ <stdexcept>
#অন্তর্ভাৱ <string>
#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <unordered_map>
#অন্তর্ভাৱ <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             তার্কিক is_static,
                             তার্কিক is_stateful,
                             তার্কিক model_is_splitted,
                             তার্কিক is_prefill,
                             পূর্ণাঙ্ক prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(মিছা),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    স্থির তার্কিক printed_address_map = মিছা;
    যদি (!printed_address_map) {
        যদি (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = সঁচা;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_প্ৰবেশ_output();
    compute_গ্ৰন্থি_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    বাবে (পূর্ণাঙ্ক গ্ৰন্থি_n = 0; গ্ৰন্থি_n < cgraph->n_nodes; গ্ৰন্থি_n++) {
        m_গ্ৰন্থি_info_list[গ্ৰন্থি_n].গ্ৰন্থি_op_case = compute_op_case(m_গ্ৰন্থি_info_list[গ্ৰন্থি_n].গ্ৰন্থি);
        m_গ্ৰন্থি_info_list[গ্ৰন্থি_n].গ্ৰন্থি_op_type = compute_op_type(m_গ্ৰন্থি_info_list[গ্ৰন্থি_n].গ্ৰন্থি);
    }

    add_extra_inputs();
}

শূন্য GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_গ্ৰন্থি_info_list.clear();
    set_প্ৰবেশ_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
তার্কিক GgmlOvDecoder::is_bound_to(ধ্রুৱ ggml_cgraph * cgraph) ধ্রুৱ {
    ঘূৰাই m_cgraph == cgraph && cgraph->n_nodes > 0 && m_গ্ৰন্থি_info_list.size() == (size_t) cgraph->n_nodes &&
           m_গ্ৰন্থি_info_list.front().গ্ৰন্থি == cgraph->nodes[0] &&
           m_গ্ৰন্থি_info_list.back().গ্ৰন্থি == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = সঁচা;
    set_প্ৰবেশ_output();
    compute_model_inputs();
    compute_model_outputs();
    বাবে (পূর্ণাঙ্ক গ্ৰন্থি_n = 0; গ্ৰন্থি_n < cgraph->n_nodes; গ্ৰন্থি_n++) {
        m_গ্ৰন্থি_info_list[গ্ৰন্থি_n].গ্ৰন্থি_op_case = compute_op_case(m_গ্ৰন্থি_info_list[গ্ৰন্থি_n].গ্ৰন্থি);
        m_গ্ৰন্থি_info_list[গ্ৰন্থি_n].গ্ৰন্থি_op_type = compute_op_type(m_গ্ৰন্থি_info_list[গ্ৰন্থি_n].গ্ৰন্থি);
    }
}

নামাকাশ {
তার্কিক is_inplace_op(ধ্রুৱ ggml_tensor * গ্ৰন্থি) {
    ঘূৰাই গ্ৰন্থি->op == GGML_OP_SET_ROWS || গ্ৰন্থি->op == GGML_OP_CPY || (গ্ৰন্থি->op == GGML_OP_SCALE && গ্ৰন্থি->view_src);
}

তার্কিক is_same_shape(ধ্রুৱ ggml_tensor * a, ধ্রুৱ ggml_tensor * b) {
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
        যদি (a->ne[i] != b->ne[i]) {
            ঘূৰাই মিছা;
        }
    }
    ঘূৰাই সঁচা;
}

তার্কিক is_conv_states_all_tensor(ধ্রুৱ ggml_tensor * tensor) {
    ঘূৰাই tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

তার্কিক is_full_single_slot_writeback(ধ্রুৱ ggml_tensor * গ্ৰন্থি) {
    ঘূৰাই গ্ৰন্থি->view_src != nullptr && গ্ৰন্থি->view_src->ne[1] == 1 && গ্ৰন্থি->src[1] != nullptr &&
           গ্ৰন্থি->src[1]->op == GGML_OP_VIEW && গ্ৰন্থি->src[1]->view_src == গ্ৰন্থি->view_src &&
           গ্ৰন্থি->src[1]->view_offs == 0 && ggml_nbytes(গ্ৰন্থি->src[1]) == ggml_nbytes(গ্ৰন্থি->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
তার্কিক is_moe_expert_sum_add(ধ্রুৱ ggml_tensor * গ্ৰন্থি) {
    std::vector<ধ্রুৱ ggml_tensor *> leaves;
    ধ্রুৱ ggml_tensor * cur = গ্ৰন্থি;
    যেতিয়ালৈ (cur->op == GGML_OP_ADD) {
        যদি (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            ঘূৰাই মিছা;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ধ্রুৱ ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    বাবে (ধ্রুৱ ggml_tensor * leaf : leaves) {
        যদি (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            ঘূৰাই মিছা;
        }
        ধ্রুৱ ggml_tensor * leaf_base = leaf->src[0];
        যদি (base == nullptr) {
            base = leaf_base;
        } অন্যথা যদি (leaf_base != base) {
            ঘূৰাই মিছা;
        }
        যদি (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            ঘূৰাই মিছা;
        }
        যদি (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            ঘূৰাই মিছা;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        যদি (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            ঘূৰাই মিছা;
        }
    }

    ঘূৰাই base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ধ্রুৱ ggml_cgraph * cgraph, ধ্রুৱ ggml_tensor * tensor) {
    যদি (tensor == nullptr) {
        ঘূৰাই "";
    }
    যদি ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ধ্রুৱ auto * গ্ৰন্থি = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        যদি (গ্ৰন্থি != cgraph->nodes + cgraph->n_nodes) {
            ঘূৰাই std::string(tensor->name) + "#n" + std::to_string(গ্ৰন্থি - cgraph->nodes);
        }
        ধ্রুৱ auto * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        যদি (leaf != cgraph->leafs + cgraph->n_leafs) {
            ঘূৰাই std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    ঘূৰাই tensor->name;
}

স্থির std::string get_tensor_ov_name(ধ্রুৱ ggml_cgraph * cgraph, ধ্রুৱ ggml_tensor * tensor) {
    ঘূৰাই GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

স্থির std::string get_tensor_graph_প্ৰবেশ_ov_name(ধ্রুৱ GgmlOvDecoder * decoder,
                                                  ধ্রুৱ ggml_cgraph * cgraph,
                                                  ধ্রুৱ ggml_tensor * tensor,
                                                  ধ্রুৱ ggml_tensor * op) {
    যদি (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        ঘূৰাই "inp_pos";
    }
    যদি (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        ঘূৰাই "embd";
    }
    যদি (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ধ্রুৱ তার্কিক is_swa = decoder->is_swa_mask(tensor);
        যদি (decoder->is_stateful()) {
            ঘূৰাই is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        যদি (is_swa) {
            ঘূৰাই get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    ঘূৰাই get_tensor_ov_name(cgraph, tensor);
}

শূন্য GgmlOvDecoder::set_প্ৰবেশ_output() {
    বাবে (পূর্ণাঙ্ক গ্ৰন্থি_n = 0; গ্ৰন্থি_n < m_cgraph->n_nodes; গ্ৰন্থি_n++) {
        auto * গ্ৰন্থি = m_cgraph->nodes[গ্ৰন্থি_n];

        গ্ৰন্থি_Info current_গ্ৰন্থি_info;
        auto গ্ৰন্থি_name = get_tensor_ov_name(m_cgraph, গ্ৰন্থি);

        current_গ্ৰন্থি_info.গ্ৰন্থি = গ্ৰন্থি;
        current_গ্ৰন্থি_info.গ্ৰন্থি_name = গ্ৰন্থি_name;
        current_গ্ৰন্থি_info.গ্ৰন্থি_op_case = 0;
        current_গ্ৰন্থি_info.data_addr = গ্ৰন্থি->data;

        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = গ্ৰন্থি->src[i];
            যদি (src == nullptr) {
                অনুৱর্ত;
            }
            auto src_name = get_tensor_ov_name(m_cgraph, src);
            যদি (src->flags & GGML_TENSOR_FLAG_প্ৰবেশ) {
                src_name = get_tensor_graph_প্ৰবেশ_ov_name(এতত্, m_cgraph, src, গ্ৰন্থি);
            }
            current_গ্ৰন্থি_info.গ্ৰন্থি_inputs[src_name] = src;
            current_গ্ৰন্থি_info.গ্ৰন্থি_inputs_names.push_back(src_name);

            যদি (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_গ্ৰন্থি_info.গ্ৰন্থি_inputs_views[src_name])> view_chain;
                auto * current = src;

                যেতিয়ালৈ (current != nullptr) {
                    auto current_name = get_tensor_ov_name(m_cgraph, current);
                    যদি (current->flags & GGML_TENSOR_FLAG_প্ৰবেশ) {
                        current_name = get_tensor_graph_প্ৰবেশ_ov_name(এতত্, m_cgraph, current, গ্ৰন্থি);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    যদি (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } অন্যথা {
                        ভঙ্গ;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_গ্ৰন্থি_info.গ্ৰন্থি_inputs_views[src_name] = view_chain;
            }
        }

        m_গ্ৰন্থি_info_list.push_back(current_গ্ৰন্থি_info);
    }
}

পূর্ণাঙ্ক GgmlOvDecoder::compute_op_case(ধ্রুৱ ggml_tensor * গ্ৰন্থি) ধ্রুৱ {
    পূর্ণাঙ্ক op_case = 0;
    ৱিকল্প (গ্ৰন্থি->op) {
    পক্ষ GGML_OP_RESHAPE: {
        যদি (m_naive) {
            ভঙ্গ;
        }
        auto name = std::string(গ্ৰন্থি->name);
        auto * src = গ্ৰন্থি->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        তার্কিক recurrent_sequence = মিছা;
        বাবে (পূর্ণাঙ্ক i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ধ্রুৱ auto * consumer = m_cgraph->nodes[i];
            যদি (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == গ্ৰন্থি) {
                ঘূৰাই 1;
            } অন্যথা যদি (consumer->op == GGML_OP_SSM_CONV) {
                ধ্রুৱ auto * concat = consumer->src[0];
                যদি (concat->op == GGML_OP_CONCAT) {
                    ধ্রুৱ auto * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == গ্ৰন্থি;
                }
            } অন্যথা যদি (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                ধ্রুৱ auto * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == গ্ৰন্থি;
            }
        }
        যদি (recurrent_sequence && গ্ৰন্থি->ne[0] == src->ne[0] && গ্ৰন্থি->ne[3] == 1) {
            ঘূৰাই 6;
        }
        যদি (গ্ৰন্থি->ne[0] == src->ne[0] && গ্ৰন্থি->ne[2] == 1 && গ্ৰন্থি->ne[3] == 1) {
            ঘূৰাই 5;
        }
        যদি (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == গ্ৰন্থি->ne[0] && src->src[0]->ne[1] == গ্ৰন্থি->ne[1]) {
            op_case = 4;
        } অন্যথা যদি (গ্ৰন্থি->ne[0] * গ্ৰন্থি->ne[1] == src->ne[0]) {
            op_case = 1;
        } অন্যথা যদি (src->ne[0] * src->ne[1] == গ্ৰন্থি->ne[0]) {
            op_case = 2;
            যদি (src->ne[2] * src->ne[3] == গ্ৰন্থি->ne[1]) {
                op_case = 5;
            }
        } অন্যথা যদি (গ্ৰন্থি->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == গ্ৰন্থি->ne[1]) {
            op_case = 3;
        } অন্যথা যদি (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } অন্যথা যদি (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } অন্যথা যদি (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        যদি (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            ধ্রুৱ auto * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            তার্কিক recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            বাবে (পূর্ণাঙ্ক i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ধ্রুৱ auto * consumer = m_cgraph->nodes[i];
                যদি (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    বাবে (পূর্ণাঙ্ক j : {3, 4}) {
                        ধ্রুৱ auto * gate = consumer->src[j];
                        যদি (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == গ্ৰন্থি;
                    }
                } অন্যথা যদি (consumer->op == GGML_OP_MUL) {
                    বাবে (পূর্ণাঙ্ক j = 0; j < 2; ++j) {
                        ধ্রুৱ auto * gate = consumer->src[j];
                        ধ্রুৱ auto * norm = consumer->src[1 - j];
                        যদি (gate->op != GGML_OP_UNARY || gate->src[0] != গ্ৰন্থি) {
                            অনুৱর্ত;
                        }
                        যদি (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            যদি (recurrent) {
                op_case = 9;
            }
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_PERMUTE: {
        যদি (গ্ৰন্থি->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } অন্যথা যদি (গ্ৰন্থি->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(গ্ৰন্থি->view_src->name);
            পূর্ণাঙ্ক layer = extract_layer_from_name(src_name).value();
            যদি (ggml_is_contiguous(গ্ৰন্থি->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                যদি (!is_swa_layer(layer)) {
                    op_case = 3;
                } অন্যথা {
                    op_case = 4;
                }
            } অন্যথা {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                যদি (!is_swa_layer(layer)) {
                    op_case = 5;
                } অন্যথা {
                    op_case = 6;
                }
            }
        } অন্যথা {
            // rope'ed query tensor
            op_case = 2;
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_MUL_MAT: {
        যদি (গ্ৰন্থি->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_GET_ROWS: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        যদি (গ্ৰন্থি->src[0]->op == GGML_OP_RESHAPE && গ্ৰন্থি->src[0]->src[0] != nullptr &&
            is_recurrent_cache(গ্ৰন্থি->src[0]->src[0])) {
            ধ্রুৱ তার্কিক single_slot = গ্ৰন্থি->src[0]->src[0]->ne[1] == 1;
            যদি (গ্ৰন্থি->src[1]->op == GGML_OP_VIEW) {
                op_case = (গ্ৰন্থি->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } অন্যথা যদি (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_ROPE: {
        ধ্রুৱ পূর্ণাঙ্ক mode = গ্ৰন্থি->op_params[2];
        ৱিকল্প (mode) {
        পক্ষ GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            ভঙ্গ;
        }
        পক্ষ GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            ভঙ্গ;
        }
        পক্ষ GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            ভঙ্গ;
        }
        পক্ষ GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            ভঙ্গ;
        }
        স্ৱভাৱ:
            op_case = 0;
            ভঙ্গ;
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_VIEW: {
        যদি (!m_model_params.has_rs_rollback && গ্ৰন্থি->src[0] != nullptr &&
            গ্ৰন্থি->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            ভঙ্গ;
        }
        যদি (m_is_static && গ্ৰন্থি->src[0] != nullptr &&
            (গ্ৰন্থি->src[0]->op == GGML_OP_GATED_DELTA_NET || গ্ৰন্থি->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            ভঙ্গ;
        }
        যদি (গ্ৰন্থি->src[0]->op == GGML_OP_VIEW) {
            auto * src = গ্ৰন্থি->src[0];
            যদি (ggml_nelements(গ্ৰন্থি) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            যদি (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            auto * src = গ্ৰন্থি->src[0];
            যদি (ggml_nelements(গ্ৰন্থি) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                যদি (গ্ৰন্থি->ne[0] == src->ne[0] && গ্ৰন্থি->ne[1] == src->ne[2] && গ্ৰন্থি->ne[2] == 1 &&
                    গ্ৰন্থি->nb[0] == src->nb[0] && গ্ৰন্থি->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    ভঙ্গ;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                পূর্ণাঙ্ক diff_গুনতি = 0;
                বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
                    যদি (গ্ৰন্থি->ne[i] != src->ne[i]) {
                        diff_গুনতি++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    যদি (গ্ৰন্থি->ne[i] > src->ne[i]) {
                        ঘূৰাই 0;
                    }
                }
                যদি (diff_গুনতি >= 1) {
                    op_case = 0;
                }
            }
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_RMS_NORM: {
        যদি (গ্ৰন্থি->src[0]->op == GGML_OP_VIEW) {
            যদি (is_same_shape(গ্ৰন্থি->src[0]->src[0], গ্ৰন্থি->src[0])) {
                op_case = 1;
            } অন্যথা যদি (!m_model_params.has_rs_rollback &&
                       গ্ৰন্থি->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } অন্যথা যদি (গ্ৰন্থি->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_POOL_2D: {
        ধ্রুৱ ggml_op_pool pool_mode = static_cast<ggml_op_pool>(গ্ৰন্থি->op_params[0]);
        ৱিকল্প (pool_mode) {
        পক্ষ GGML_OP_POOL_MAX: {
            op_case = 1;
            ভঙ্গ;
        }
        পক্ষ GGML_OP_POOL_AVG: {
            op_case = 2;
            ভঙ্গ;
        }
        স্ৱভাৱ:
            op_case = 0;
            ভঙ্গ;
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_UPSCALE: {
        ধ্রুৱ int32_t mode_flags = গ্ৰন্থি->op_params[0];
        ধ্রুৱ ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        ৱিকল্প (scale_mode) {
        পক্ষ GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            ভঙ্গ;
        }
        পক্ষ GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            ভঙ্গ;
        }
        পক্ষ GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            ভঙ্গ;
        }
        স্ৱভাৱ:
            op_case = 0;
            ভঙ্গ;
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_CPY: {
        যদি (গ্ৰন্থি->src[0]->op == GGML_OP_VIEW) {
            যদি (গ্ৰন্থি->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                যদি (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(গ্ৰন্থি) ? 7 : 10;
                } অন্যথা {
                    op_case = 1;
                }
            } অন্যথা যদি (GgmlOvDecoder::is_conv_state_writeback(গ্ৰন্থি)) {
                op_case = is_full_single_slot_writeback(গ্ৰন্থি) ? 8 : 2;
                ভঙ্গ;
            } অন্যথা যদি (is_conv_states_all_tensor(গ্ৰন্থি->view_src) && গ্ৰন্থি->src[1] != nullptr &&
                       গ্ৰন্থি->src[1]->op == GGML_OP_VIEW && গ্ৰন্থি->src[1]->view_src == গ্ৰন্থি->view_src) {
                op_case = 4;
                ভঙ্গ;
            } অন্যথা যদি (গ্ৰন্থি->src[0]->src[0]->op == GGML_OP_GET_ROWS && গ্ৰন্থি->src[1] != nullptr &&
                       গ্ৰন্থি->src[1]->op == GGML_OP_VIEW && গ্ৰন্থি->src[1]->view_src != nullptr &&
                       is_recurrent_cache(গ্ৰন্থি->src[1]->view_src) && গ্ৰন্থি->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } অন্যথা যদি (গ্ৰন্থি->src[0]->op == GGML_OP_GET_ROWS && গ্ৰন্থি->src[1] != nullptr &&
                   গ্ৰন্থি->src[1]->op == GGML_OP_VIEW && গ্ৰন্থি->src[1]->view_src != nullptr &&
                   is_recurrent_cache(গ্ৰন্থি->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = গ্ৰন্থি->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } অন্যথা যদি (গ্ৰন্থি->src[1] != nullptr && গ্ৰন্থি->src[1]->op == GGML_OP_VIEW && গ্ৰন্থি->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ধ্রুৱ ggml_tensor * kv_buf = গ্ৰন্থি->src[1]->view_src;
            যদি (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                বাবে (পূর্ণাঙ্ক i = 0; i < m_cgraph->n_nodes; i++) {
                    ধ্রুৱ ggml_tensor * n = m_cgraph->nodes[i];
                    যদি (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        অনুৱর্ত;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    যদি ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        যদি (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        ভঙ্গ;
                    }
                }
            }
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_ADD: {
        যদি (is_moe_expert_sum_add(গ্ৰন্থি)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_SCALE: {
        যদি (গ্ৰন্থি->view_src && গ্ৰন্থি->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = গ্ৰন্থি->view_src->ne[1] == 1 ? 2 : 1;
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_L2_NORM: {
        যদি (std::string(গ্ৰন্থি->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        ভঙ্গ;
    }
    পক্ষ GGML_OP_FLASH_ATTN_EXT: {
        যদি (গ্ৰন্থি->src[1] != nullptr && গ্ৰন্থি->src[1]->op == GGML_OP_VIEW && গ্ৰন্থি->src[1]->view_src != nullptr) {
            ধ্রুৱ ggml_tensor * kv_buf = গ্ৰন্থি->src[1]->view_src;
            যদি (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (গ্ৰন্থি->src[3] != nullptr) ? 1 : 2;
            }
        }
        ভঙ্গ;
    }
    স্ৱভাৱ:
        ভঙ্গ;
    }
    ঘূৰাই op_case;
}

std::optional<পূর্ণাঙ্ক> extract_layer_from_name(ধ্রুৱ std::string & name) {
    size_t pos1 = name.find("_l");
    যদি (pos1 == std::string::npos) {
        ঘূৰাই std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    যদি (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    পূর্ণাঙ্ক layer = std::stoi(layer_str);
    ঘূৰাই layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
স্থির পূর্ণাঙ্ক get_swa_window_from_mask(ধ্রুৱ ggml_tensor * mask) {
    যদি (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        ঘূৰাই -1;
    }
    যদি (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        ঘূৰাই -1;
    }

    ধ্রুৱ int64_t n_kv = mask->ne[0];
    ধ্রুৱ int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    বাবে (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        বাবে (int64_t c = 0; c < n_kv; c++) {
            ধ্রুৱ size_t i = (size_t) r * n_kv + c;
            ধ্রুৱ ভিন্ন v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((ধ্রুৱ ggml_fp16_t *) mask->data)[i]) :
                                                          ((ধ্রুৱ ভিন্ন *) mask->data)[i];
            যদি (v > -INFINITY) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    ঘূৰাই window > 0 ? (পূর্ণাঙ্ক) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, তার্কিক is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    auto get_attention_pattern_case = [](ধ্রুৱ ggml_tensor * গ্ৰন্থি) -> পূর্ণাঙ্ক {
        যদি (গ্ৰন্থি == nullptr) {
            ঘূৰাই -1;
        }

        ৱিকল্প (গ্ৰন্থি->op) {
        পক্ষ GGML_OP_FLASH_ATTN_EXT:
            যদি (গ্ৰন্থি->src[0] == nullptr || গ্ৰন্থি->src[1] == nullptr) {
                ঘূৰাই -1;
            }
            ৱিকল্প (গ্ৰন্থি->src[1]->op) {
            পক্ষ GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                যদি (গ্ৰন্থি->src[3] != nullptr && গ্ৰন্থি->src[1]->src[0] != nullptr &&
                    গ্ৰন্থি->src[1]->src[0]->op == GGML_OP_VIEW) {
                    ঘূৰাই 0;
                }
                ভঙ্গ;
            পক্ষ GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                যদি (গ্ৰন্থি->src[3] != nullptr && গ্ৰন্থি->src[1]->src[0] != nullptr &&
                    গ্ৰন্থি->src[1]->src[0]->op == GGML_OP_PERMUTE && গ্ৰন্থি->src[1]->src[0]->src[0] != nullptr &&
                    গ্ৰন্থি->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    ঘূৰাই 1;
                }
                ভঙ্গ;
            পক্ষ GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                যদি (গ্ৰন্থি->src[1]->view_src != nullptr) {
                    যদি (গ্ৰন্থি->src[3] != nullptr) {
                        ঘূৰাই 4;  // decoder self-attention
                    }
                    ঘূৰাই 5;      // cross-attention or encoder self-attention
                }
                ভঙ্গ;
            স্ৱভাৱ:
                ভঙ্গ;
            }
            ভঙ্গ;
        পক্ষ GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            যদি (গ্ৰন্থি->src[0] != nullptr && গ্ৰন্থি->src[1] != nullptr && গ্ৰন্থি->src[0]->op == GGML_OP_MUL_MAT &&
                গ্ৰন্থি->src[0]->src[0] != nullptr && গ্ৰন্থি->src[0]->src[1] != nullptr &&
                গ্ৰন্থি->src[0]->src[0]->op == GGML_OP_PERMUTE && গ্ৰন্থি->src[0]->src[0]->src[0] != nullptr &&
                গ্ৰন্থি->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                ঘূৰাই 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            যদি (গ্ৰন্থি->src[0]->op == GGML_OP_ADD && গ্ৰন্থি->src[0]->src[0] != nullptr &&
                গ্ৰন্থি->src[0]->src[0]->op == GGML_OP_MUL_MAT && গ্ৰন্থি->src[0]->src[0]->src[0] != nullptr &&
                গ্ৰন্থি->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                ঘূৰাই 3;
            }
            ভঙ্গ;
        স্ৱভাৱ:
            ভঙ্গ;
        }

        ঘূৰাই -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    auto get_attention_op_mask = [&get_attention_pattern_case](ধ্রুৱ ggml_tensor * গ্ৰন্থি) -> ধ্রুৱ ggml_tensor * {
        ৱিকল্প (get_attention_pattern_case(গ্ৰন্থি)) {
        পক্ষ 0:
        পক্ষ 1:
            ঘূৰাই গ্ৰন্থি->src[3];
        পক্ষ 2:
        পক্ষ 3:
            ঘূৰাই গ্ৰন্থি->src[1];
        স্ৱভাৱ:
            ঘূৰাই nullptr;
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
        std::map<পূর্ণাঙ্ক, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<পূর্ণাঙ্ক, ধ্রুৱ ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
            ধ্রুৱ ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            যদি (mask == nullptr) {
                অনুৱর্ত;
            }
            ধ্রুৱ ggml_tensor * cache_k_permute = nullptr;
            ৱিকল্প (get_attention_pattern_case(cgraph->nodes[i])) {
            পক্ষ 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     ভঙ্গ;
            পক্ষ 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             ভঙ্গ;
            পক্ষ 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             ভঙ্গ;
            স্ৱভাৱ: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     ভঙ্গ;
            }
            ধ্রুৱ ggml_tensor * cache_k_view = cache_k_permute->src[0];
            যদি (cache_k_view->op != GGML_OP_VIEW) {
                অনুৱর্ত;
            }
            ধ্রুৱ ggml_tensor * leaf = cache_k_view->src[0];
            auto layer = extract_layer_from_name(leaf->name);
            যদি (!layer.has_value()) {
                অনুৱর্ত;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        বাবে (ধ্রুৱ auto & [layer, extent] : layer_extent) {
            যদি (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                যদি (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        যদি (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            বাবে (ধ্রুৱ auto & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (long) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    তার্কিক rope_seen = মিছা;
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];
        ধ্রুৱ পূর্ণাঙ্ক attention_pattern_case = get_attention_pattern_case(গ্ৰন্থি);
        যদি (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            ৱিকল্প (attention_pattern_case) {
            পক্ষ 0:
                cache_k_permute = গ্ৰন্থি->src[1];
                mask = গ্ৰন্থি->src[3];
                ভঙ্গ;
            পক্ষ 1:
                cache_k_permute = গ্ৰন্থি->src[1]->src[0];
                mask = গ্ৰন্থি->src[3];
                ভঙ্গ;
            পক্ষ 2:
                cache_k_permute = গ্ৰন্থি->src[0]->src[0];
                mask = গ্ৰন্থি->src[1];
                ভঙ্গ;
            পক্ষ 3:
                cache_k_permute = গ্ৰন্থি->src[0]->src[0]->src[0];
                mask = গ্ৰন্থি->src[1];
                ভঙ্গ;
            পক্ষ 4:
            পক্ষ 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                auto * cache_k_view = গ্ৰন্থি->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = গ্ৰন্থি->src[0]->ne[1];
                যদি (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } অন্যথা {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                অনুৱর্ত;
            }
            স্ৱভাৱ:
                ভঙ্গ;
            }

            প্রতিজ্ঞা(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.প্ৰবেশ_len = গ্ৰন্থি->src[0]->ne[1];
            compute_params.token_len_per_seq = গ্ৰন্থি->src[0]->ne[1];

            auto * cache_k_view = cache_k_permute->src[0];
            যদি (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                অনুৱর্ত;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            পূর্ণাঙ্ক layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ধ্রুৱ তার্কিক layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            যদি (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } অন্যথা {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, পরিমাণ(size_t));
            compute_params.seq_active_start = offset / seq_size;

            যদি (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } অন্যথা {
                compute_params.attention_size = mask->ne[0];
            }
            যদি (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        যদি (গ্ৰন্থি->op == GGML_OP_MUL_MAT && গ্ৰন্থি->src[0]->op == GGML_OP_PERMUTE &&
            গ্ৰন্থি->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(গ্ৰন্থি->src[0]->view_src, গ্ৰন্থি->view_src)) {
            যদি (গ্ৰন্থি->src[1]->op == GGML_OP_PERMUTE && গ্ৰন্থি->src[1]->src[0]->op == GGML_OP_VIEW &&
                গ্ৰন্থি->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = গ্ৰন্থি->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        যদি (গ্ৰন্থি->op == GGML_OP_TRANSPOSE && গ্ৰন্থি->src[0]->op == GGML_OP_PERMUTE &&
            গ্ৰন্থি->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = গ্ৰন্থি->ne[0];
            যদি (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        যদি (গ্ৰন্থি->op == GGML_OP_ROPE) {
            যদি (compute_params.token_len_per_seq == -1 && গ্ৰন্থি->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(গ্ৰন্থি->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(পরিমাণ(model_params.rope_params) == পরিমাণ(int32_t) * 16, "rope_params size");
            যদি (!rope_seen) {
                memcpy(model_params.rope_params, গ্ৰন্থি->op_params, পরিমাণ(int32_t) * 16);
                rope_seen = সঁচা;
            } অন্যথা যদি (memcmp(model_params.rope_params, গ্ৰন্থি->op_params, পরিমাণ(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = সঁচা;
            }
        }
        যদি (গ্ৰন্থি->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = গ্ৰন্থি->src[0]->ne[0];
        }
        যদি (গ্ৰন্থি->op == GGML_OP_SCALE && গ্ৰন্থি->view_src != nullptr && is_recurrent_cache(গ্ৰন্থি->view_src)) {
            যদি (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = গ্ৰন্থি->view_src->ne[1];
            } অন্যথা {
                GGML_ASSERT(model_params.n_rs_slots == গ্ৰন্থি->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(গ্ৰন্থি) / গ্ৰন্থি->view_src->ne[0];
            compute_params.cache_rs_reset_idx = গ্ৰন্থি->src[0]->view_offs / গ্ৰন্থি->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        যদি (গ্ৰন্থি->op == GGML_OP_CPY && গ্ৰন্থি->view_src != nullptr && is_recurrent_cache(গ্ৰন্থি->view_src) &&
            গ্ৰন্থি->src[1] != nullptr && গ্ৰন্থি->src[1]->op == GGML_OP_VIEW && গ্ৰন্থি->src[1]->view_src == গ্ৰন্থি->view_src) {
            ধ্রুৱ তার্কিক is_conv = is_conv_state_writeback(গ্ৰন্থি);
            ধ্রুৱ তার্কিক is_gdn = গ্ৰন্থি->src[0]->op == GGML_OP_VIEW && গ্ৰন্থি->src[0]->src[0] != nullptr &&
                                গ্ৰন্থি->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ধ্রুৱ তার্কিক is_extra = গ্ৰন্থি->src[0]->op == GGML_OP_GET_ROWS;
            ধ্রুৱ তার্কিক is_gdn_rollback = is_gdn && is_same_shape(গ্ৰন্থি->src[0], গ্ৰন্থি->src[1]);

            ধ্রুৱ ggml_tensor * dest_view = গ্ৰন্থি->src[1];
            ধ্রুৱ ggml_tensor * cache = গ্ৰন্থি->view_src;
            ধ্রুৱ size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            যদি (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = সঁচা;
            }
            যদি (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(গ্ৰন্থি)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (পূর্ণাঙ্ক) (dest_view->view_offs / row_bytes);
                যদি (is_conv) {
                    writeback.src_begin = (পূর্ণাঙ্ক) (গ্ৰন্থি->src[0]->view_offs / গ্ৰন্থি->src[0]->view_src->nb[0]);
                } অন্যথা যদি (is_gdn_rollback) {
                    writeback.src_begin = (পূর্ণাঙ্ক) (গ্ৰন্থি->src[0]->view_offs / গ্ৰন্থি->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, গ্ৰন্থি)] = writeback;
            }
            যদি ((is_conv || is_gdn) && !is_full_single_slot_writeback(গ্ৰন্থি)) {
                compute_params.s_copy_active_slot_len = (পূর্ণাঙ্ক) dest_view->ne[1];
            }
        }
    }
    যদি (model_params.n_heads_kv == -1) {
        বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
            ধ্রুৱ auto * গ্ৰন্থি = cgraph->nodes[i];
            ধ্রুৱ ggml_tensor * mask = nullptr;
            যদি (গ্ৰন্থি->op == GGML_OP_SOFT_MAX) {
                mask = গ্ৰন্থি->src[1];
            } অন্যথা যদি (গ্ৰন্থি->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = গ্ৰন্থি->src[3];
            } অন্যথা {
                অনুৱর্ত;
            }
            যদি (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_প্ৰবেশ) ||
                গ্ৰন্থি->src[0] == nullptr) {
                অনুৱর্ত;
            }
            model_params.is_cacheless_attn = সঁচা;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.প্ৰবেশ_len = গ্ৰন্থি->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.প্ৰবেশ_len;
            ভঙ্গ;
        }
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    যদি (model_params.is_cacheless_attn) {
        বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
            ধ্রুৱ auto * গ্ৰন্থি = cgraph->nodes[i];
            যদি (গ্ৰন্থি->op == GGML_OP_GET_ROWS && is_output_idx(গ্ৰন্থি->src[1], গ্ৰন্থি)) {
                compute_params.output_len = গ্ৰন্থি->src[1]->ne[0];
                ভঙ্গ;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    যদি (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    ঘূৰাই {model_params, compute_params};
}

শূন্য GgmlOvDecoder::validate_cgraph() ধ্রুৱ {
    যদি (m_model_params.n_seq > 1 && m_is_static == সঁচা) {
        ক্ষিপ std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_প্ৰবেশ_shape(ধ্রুৱ ggml_tensor * op,
                                                      ধ্রুৱ ggml_tensor * প্ৰবেশ,
                                                      পূর্ণাঙ্ক dynamic_dim_index) ধ্রুৱ {
    যদি (m_naive) {
        ঘূৰাই প্ৰবেশ != nullptr ? ov::PartialShape{get_shape(প্ৰবেশ)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape প্ৰবেশ_shape;

    যদি (is_inp_tok(প্ৰবেশ, op) || is_inp_pos(প্ৰবেশ, op)) {
        // tokens or positions
        পূর্ণাঙ্ক len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        যদি (m_is_static && is_inp_pos(প্ৰবেশ, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        প্ৰবেশ_shape = ov::PartialShape{1, 1, 1, len};

    } অন্যথা যদি (is_output_idx(প্ৰবেশ, op)) {
        // output index
        প্ৰবেশ_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } অন্যথা যদি (is_inp_mean(প্ৰবেশ, op)) {
        প্ৰবেশ_shape = m_is_static ? ov::PartialShape{1, 1, প্ৰবেশ->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } অন্যথা যদি (is_inp_scale_rows(প্ৰবেশ, op)) {
        প্ৰবেশ_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } অন্যথা যদি (is_inp_mask(প্ৰবেশ, op)) {
        // mask
        যদি (m_is_static) {
            প্ৰবেশ_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } অন্যথা যদি (m_is_stateful) {
            প্ৰবেশ_shape = ov::PartialShape{1, 1, -1, -1};
        } অন্যথা {
            প্ৰবেশ_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } অন্যথা যদি (is_recurrent_cache(প্ৰবেশ)) {
        প্ৰবেশ_shape = ov::PartialShape{get_shape(প্ৰবেশ)};
        যদি (!m_is_static && !m_is_stateful && প্ৰবেশ->ne[1] > 1) {
            প্ৰবেশ_shape[2] = -1;
        }

    } অন্যথা যদি (is_kvcache(প্ৰবেশ, op)) {
        // kvcache
        প্ৰবেশ_shape = ov::PartialShape{get_shape(প্ৰবেশ)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ধ্রুৱ তার্কিক is_flat_kv = (প্ৰবেশ->ne[1] == 1 && প্ৰবেশ->ne[2] == 1 && প্ৰবেশ->ne[3] == 1);
        যদি (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            প্ৰবেশ_shape[2] = -1;
        }
        যদি (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            প্রতিজ্ঞা(প্ৰবেশ_shape.size() == 4 && প্ৰবেশ_shape[0] == 1 && প্ৰবেশ_shape[1] == 1 &&
                   প্ৰবেশ_shape[2].is_dynamic() && প্ৰবেশ_shape[3].is_static());
            ধ্রুৱ পূর্ণাঙ্ক n_heads_kv = get_n_heads_kv_for_tensor(প্ৰবেশ);
            প্রতিজ্ঞা(n_heads_kv > 0 && প্ৰবেশ_shape[3].get_length() % n_heads_kv == 0);
            ধ্রুৱ int64_t combined_dim = প্ৰবেশ_shape[3].get_length();  // n_heads_kv * head_size
            ধ্রুৱ int64_t head_size = combined_dim / n_heads_kv;
            প্ৰবেশ_shape = {প্ৰবেশ_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } অন্যথা যদি (is_kv_idx(প্ৰবেশ, op)) {
        // kv update index
        পূর্ণাঙ্ক len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        প্ৰবেশ_shape = ov::PartialShape{1, 1, 1, len};

    } অন্যথা যদি (is_inp_s_copy(প্ৰবেশ, op) || is_s_copy_leaf(প্ৰবেশ)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        প্ৰবেশ_shape = m_is_static ? ov::PartialShape{get_shape(প্ৰবেশ)} : ov::PartialShape{1, 1, 1, -1};

    } অন্যথা {
        প্ৰবেশ_shape = ov::PartialShape{get_shape(প্ৰবেশ)};
    }
    যদি (dynamic_dim_index != -1 && m_model_is_splitted) {
        প্ৰবেশ_shape[3 - dynamic_dim_index] = -1;
    }
    যদি (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_প্ৰবেশ && op->src[1] == প্ৰবেশ) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        যদি (m_is_static) {
            ধ্রুৱ int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            প্ৰবেশ_shape[2] = seq_active;
            প্ৰবেশ_shape[3] = seq_active;
        } অন্যথা {
            প্ৰবেশ_shape[2] = -1;
            প্ৰবেশ_shape[3] = -1;
        }
    }
    ঘূৰাই প্ৰবেশ_shape;
}

তার্কিক GgmlOvDecoder::is_s_copy_leaf(ধ্রুৱ ggml_tensor * tensor) ধ্রুৱ {
    যদি (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        ঘূৰাই মিছা;
    }
    বাবে (পূর্ণাঙ্ক i = 0; i < m_cgraph->n_nodes; i++) {
        ধ্রুৱ ggml_tensor * গ্ৰন্থি = m_cgraph->nodes[i];
        যদি (গ্ৰন্থি->op != GGML_OP_GET_ROWS || গ্ৰন্থি->src[0] == nullptr || গ্ৰন্থি->src[1] == nullptr) {
            অনুৱর্ত;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ধ্রুৱ ggml_tensor * idx = গ্ৰন্থি->src[1];
        যেতিয়ালৈ (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        যদি (idx != tensor) {
            অনুৱর্ত;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ধ্রুৱ ggml_tensor * data = গ্ৰন্থি->src[0];
        যেতিয়ালৈ (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        যদি (data != nullptr && is_recurrent_cache(data)) {
            ঘূৰাই সঁচা;
        }
    }
    ঘূৰাই মিছা;
}

শূন্য GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_প্ৰবেশ = [এতত্](ধ্রুৱ std::string & name, int64_t value, তার্কিক force_parameter = মিছা) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    যদি (m_compute_params.attention_size != -1) {
        create_1d_প্ৰবেশ("attention_size", m_compute_params.attention_size);
    }
    যদি (m_compute_params.attention_size_static != -1) {
        create_1d_প্ৰবেশ("attention_size_static", m_compute_params.attention_size_static);
    }
    যদি (m_compute_params.attention_size_swa != -1) {
        create_1d_প্ৰবেশ("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    যদি (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_প্ৰবেশ("swa_window", m_compute_params.swa_window);
    }
    create_1d_প্ৰবেশ("n_seq_active", m_compute_params.n_seq_active);
    create_1d_প্ৰবেশ("seq_active_start", m_compute_params.seq_active_start);
    create_1d_প্ৰবেশ("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    যদি (m_compute_params.token_len_per_seq != -1) {
        create_1d_প্ৰবেশ("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    যদি (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_প্ৰবেশ("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/সঁচা);
        create_1d_প্ৰবেশ("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/সঁচা);
    }

    যদি (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_প্ৰবেশ("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        যদি (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_প্ৰবেশ("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/সঁচা);
        }
    }

    বাবে (ধ্রুৱ auto & [গ্ৰন্থি_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_প্ৰবেশ("rs_slot_begin_" + গ্ৰন্থি_name, writeback.slot_begin);
        যদি (!m_is_static && writeback.src_begin >= 0) {
            create_1d_প্ৰবেশ("rs_src_begin_" + গ্ৰন্থি_name, writeback.src_begin);
        }
    }
}

তার্কিক GgmlOvDecoder::গ্ৰন্থি_is_used_as_src(ধ্রুৱ পূর্ণাঙ্ক গ্ৰন্থি_idx) {
    ggml_tensor * গ্ৰন্থি = m_cgraph->nodes[গ্ৰন্থি_idx];
    বাবে (পূর্ণাঙ্ক i = গ্ৰন্থি_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_গ্ৰন্থি = m_cgraph->nodes[i];
        বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
            যদি (other_গ্ৰন্থি->src[j] == গ্ৰন্থি) {
                ঘূৰাই সঁচা;
            }
        }
    }
    ঘূৰাই মিছা;
}

শূন্য GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    বাবে (পূর্ণাঙ্ক i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * গ্ৰন্থি = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        যদি (গ্ৰন্থি->op == GGML_OP_NONE && গ্ৰন্থি_is_used_as_src(i)) {
            std::string গ্ৰন্থি_name = get_tensor_ov_name(m_cgraph, গ্ৰন্থি);
            যদি (m_model_weights.find(গ্ৰন্থি_name) == m_model_weights.end()) {
                m_inputs[গ্ৰন্থি_name] = গ্ৰন্থি;
                m_model_inputs[গ্ৰন্থি_name] = {get_ov_type(গ্ৰন্থি),
                                             get_graph_প্ৰবেশ_shape(গ্ৰন্থি, nullptr, m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি])};
            }
            অনুৱর্ত;
        }
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = গ্ৰন্থি->src[i];
            যদি (src == nullptr) {
                অনুৱর্ত;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            যদি (src->flags & GGML_TENSOR_FLAG_প্ৰবেশ) {
                src_name = get_tensor_graph_প্ৰবেশ_ov_name(এতত্, m_cgraph, src, গ্ৰন্থি);
            }
            যদি (m_model_weights.find(src_name) != m_model_weights.end()) {
                অনুৱর্ত;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            যদি (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                অনুৱর্ত;
            }

            তার্কিক is_intermediate_গ্ৰন্থি = মিছা;
            বাবে (ধ্রুৱ auto & গ্ৰন্থি_info : m_গ্ৰন্থি_info_list) {
                যদি (গ্ৰন্থি_info.গ্ৰন্থি == src) {
                    is_intermediate_গ্ৰন্থি = সঁচা;
                    ভঙ্গ;
                }
            }
            যদি (is_intermediate_গ্ৰন্থি) {
                অনুৱর্ত;
            }
            যদি (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                অনুৱর্ত;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            যদি (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                যদি (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            যেতিয়ালৈ (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_প্ৰবেশ_shape(গ্ৰন্থি, src, m_গ্ৰন্থি_dynamic_dims[src])};
        }
    }
}

শূন্য GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    বাবে (পূর্ণাঙ্ক গ্ৰন্থি_n = 0; গ্ৰন্থি_n < m_cgraph->n_nodes; গ্ৰন্থি_n++) {
        auto * cur_গ্ৰন্থি = m_cgraph->nodes[গ্ৰন্থি_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        যদি (cur_গ্ৰন্থি->op == GGML_OP_NONE || cur_গ্ৰন্থি->op == GGML_OP_VIEW || cur_গ্ৰন্থি->op == GGML_OP_RESHAPE) {
            অনুৱর্ত;
        }
        যদি (::is_inplace_op(cur_গ্ৰন্থি) && ggml_nbytes(cur_গ্ৰন্থি) == 0) {
            অনুৱর্ত;
        }
        auto cur_গ্ৰন্থি_use_গুনতি = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_গ্ৰন্থি)];
        যদি (cur_গ্ৰন্থি_use_গুনতি == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            যদি (cur_গ্ৰন্থি != nullptr && ::is_inplace_op(cur_গ্ৰন্থি) && ggml_nbytes(cur_গ্ৰন্থি) > 0) {
                cur_গ্ৰন্থি = cur_গ্ৰন্থি->view_src;
            }
        } অন্যথা {
            পূর্ণাঙ্ক প্ৰবেশ_use_গুনতি = 0;
            বাবে (পূর্ণাঙ্ক i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * গ্ৰন্থি = m_cgraph->nodes[i];
                বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
                    যদি (গ্ৰন্থি->src[j] != রিক্ত && গ্ৰন্থি->src[j] == cur_গ্ৰন্থি) {
                        প্ৰবেশ_use_গুনতি++;
                    }
                }
            }
            যদি (প্ৰবেশ_use_গুনতি == cur_গ্ৰন্থি_use_গুনতি) {
                cur_গ্ৰন্থি = nullptr;
            }
        }
        যদি (cur_গ্ৰন্থি != nullptr) {
            std::string cur_গ্ৰন্থি_name = get_tensor_ov_name(m_cgraph, cur_গ্ৰন্থি);
            m_model_outputs[cur_গ্ৰন্থি_name] = cur_গ্ৰন্থি;
            m_model_output_names.insert(cur_গ্ৰন্থি_name);
        }
    }
}

ধ্রুৱ ggml_tensor * GgmlOvDecoder::get_tensor_used_op(ধ্রুৱ ggml_tensor * tensor) ধ্রুৱ {
    যদি (tensor == nullptr) {
        ঘূৰাই nullptr;
    }
    বাবে (পূর্ণাঙ্ক i = 0; i < m_cgraph->n_nodes; i++) {
        ধ্রুৱ auto * গ্ৰন্থি = m_cgraph->nodes[i];
        বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
            যদি (গ্ৰন্থি->src[j] == tensor) {
                ঘূৰাই গ্ৰন্থি;
            }
        }
    }
    ঘূৰাই nullptr;
}

ধ্রুৱ ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ধ্রুৱ std::string & name) ধ্রুৱ {
    বাবে (পূর্ণাঙ্ক i = 0; i < m_cgraph->n_nodes; i++) {
        ধ্রুৱ auto * গ্ৰন্থি = m_cgraph->nodes[i];
        বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
            ধ্রুৱ auto * src = গ্ৰন্থি->src[j];
            যদি (src == nullptr) {
                ভঙ্গ;
            }
            যদি (get_tensor_ov_name(m_cgraph, src) == name) {
                ঘূৰাই src;
            }
        }
    }
    ঘূৰাই nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() ধ্রুৱ {
    std::map<std::string, std::string> kv_param_res_names;
    বাবে (ধ্রুৱ auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    ঘূৰাই kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
স্থির তার্কিক is_mul_mat_id_expert_weight(ধ্রুৱ ggml_tensor * গ্ৰন্থি, পূর্ণাঙ্ক src_index) {
    ঘূৰাই গ্ৰন্থি->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, তার্কিক naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    বাবে (পূর্ণাঙ্ক গ্ৰন্থি_i = 0; গ্ৰন্থি_i < n_nodes; গ্ৰন্থি_i++) {
        auto * গ্ৰন্থি = nodes[গ্ৰন্থি_i];
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = গ্ৰন্থি->src[i];
            যদি (src == nullptr) {
                অনুৱর্ত;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            যদি (is_rope_freqs_weight(base, গ্ৰন্থি)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            যদি (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(গ্ৰন্থি, i)) {
                যদি (model_weights.find(src_name) == model_weights.end()) {
                    auto weight_গ্ৰন্থি = create_weight_গ্ৰন্থি(base, naive);
                    weight_গ্ৰন্থি->set_friendly_name(src_name);
                    model_weights[src_name] = weight_গ্ৰন্থি;
                }
            }
        }
    }
    ঘূৰাই model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
স্থির std::mutex g_nonov_weight_cache_mutex;
স্থির std::unordered_map<ধ্রুৱ শূন্য *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    বাবে (পূর্ণাঙ্ক গ্ৰন্থি_i = 0; গ্ৰন্থি_i < cgraph->n_nodes; গ্ৰন্থি_i++) {
        auto * গ্ৰন্থি = cgraph->nodes[গ্ৰন্থি_i];
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = গ্ৰন্থি->src[i];
            যদি (src == nullptr) {
                অনুৱর্ত;
            }
            ধ্রুৱ ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            যদি (is_rope_freqs_weight(base, গ্ৰন্থি)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            যদি (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    ঘূৰাই names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_গ্ৰন্থি(ggml_tensor * tensor, তার্কিক naive) {
    ধ্রুৱ তার্কিক is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    যদি (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        যদি (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            যদি (weight_extra->weight_গ্ৰন্থি) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                ঘূৰাই weight_extra->weight_গ্ৰন্থি;
            }
        } অন্যথা যদি (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            যদি (quant_extra->weight_গ্ৰন্থি) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                ঘূৰাই quant_extra->weight_গ্ৰন্থি;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    যদি (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        auto weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        auto weight_গ্ৰন্থি = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_গ্ৰন্থি->set_friendly_name(tensor->name);
        ঘূৰাই weight_গ্ৰন্থি;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ধ্রুৱ তার্কিক cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    যদি (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        auto it = g_nonov_weight_cache.find(tensor->data);
        যদি (it != g_nonov_weight_cache.end()) {
            ঘূৰাই it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    স্থির ধ্রুৱ std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    যদি (weight_types.find(tensor->type) == weight_types.end()) {
        ক্ষিপ std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    যদি (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        যদি (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } অন্যথা {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } অন্যথা {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_গ্ৰন্থি->set_friendly_name(tensor->name);
    যদি (!is_ov_buffer) {
        যদি (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            auto [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_গ্ৰন্থি);
            ঘূৰাই it->second;
        }
        ঘূৰাই ov_weight.weight_গ্ৰন্থি;
    }

    ggml_openvino_extra_base * extra;
    যদি (ov_weight.is_quantized()) {
        extra = নূতন ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_গ্ৰন্থি);
    } অন্যথা {
        extra = নূতন ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_গ্ৰন্থি);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    ঘূৰাই ov_weight.weight_গ্ৰন্থি;
}

শূন্য GgmlOvDecoder::dump_cgraph(ধ্রুৱ ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    যদি (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        ঘূৰাই;
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
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];

        // Get buffer type name
        ধ্রুৱ ৱর্ণ * buf_name = "none";
        ggml_backend_buffer_t buf = গ্ৰন্থি->view_src ? গ্ৰন্থি->view_src->buffer : গ্ৰন্থি->buffer;
        যদি (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << গ্ৰন্থি->ne[0] << ", "
             << std::setw(5) << গ্ৰন্থি->ne[1] << ", "
             << std::setw(5) << গ্ৰন্থি->ne[2] << ", "
             << std::setw(5) << গ্ৰন্থি->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(গ্ৰন্থি->op) << std::right << " "
             << std::left << std::setw(45) << গ্ৰন্থি->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << গ্ৰন্থি->nb[0] << ", "
             << std::setw(5) << গ্ৰন্থি->nb[1] << ", "
             << std::setw(5) << গ্ৰন্থি->nb[2] << ", "
             << std::setw(5) << গ্ৰন্থি->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            যদি (auto* src = গ্ৰন্থি->src[i]) {
                // Get buffer type name for source
                ধ্রুৱ ৱর্ণ * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                যদি (src_buf) {
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
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * গ্ৰন্থি = cgraph->leafs[i];

        // Get buffer type name for leaf
        ধ্রুৱ ৱর্ণ * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = গ্ৰন্থি->view_src ? গ্ৰন্থি->view_src->buffer : গ্ৰন্থি->buffer;
        যদি (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << গ্ৰন্থি->ne[0] << ", "
             << std::setw(5) << গ্ৰন্থি->ne[1] << "] "
             << std::setw(8) << ggml_op_name(গ্ৰন্থি->op) << " "
             << std::setw(16) << ggml_get_name(গ্ৰন্থি)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

শূন্য print_tensor_address_map(ধ্রুৱ ggml_cgraph * cgraph) {
    std::map<শূন্য *, std::vector<std::string>> address_map;
    বাবে (পূর্ণাঙ্ক গ্ৰন্থি_n = 0; গ্ৰন্থি_n < cgraph->n_nodes; গ্ৰন্থি_n++) {
        auto * গ্ৰন্থি = cgraph->nodes[গ্ৰন্থি_n];
        যদি (গ্ৰন্থি->data) {
            auto it = address_map.find(গ্ৰন্থি->data);
            যদি (it == address_map.end()) {
                address_map[গ্ৰন্থি->data] = std::vector<std::string>();
            }
            address_map[গ্ৰন্থি->data].push_back(গ্ৰন্থি->name);
        }
    }
    বাবে (ধ্রুৱ auto & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        বাবে (ধ্রুৱ auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ধ্রুৱ ggml_tensor * tensor) {
    std::vector<size_t> shape;
    বাবে (পূর্ণাঙ্ক i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    ঘূৰাই shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(ধ্রুৱ ggml_tensor * tensor) {
    std::vector<size_t> stride;
    বাবে (পূর্ণাঙ্ক i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    ঘূৰাই stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(ধ্রুৱ ggml_tensor * tensor) {
    ৱিকল্প (tensor->type) {
    পক্ষ GGML_TYPE_F64:
        ঘূৰাই ov::element::f64;
    পক্ষ GGML_TYPE_F32:
        ঘূৰাই ov::element::f32;
    পক্ষ GGML_TYPE_F16:
        ঘূৰাই ov::element::f16;
    পক্ষ GGML_TYPE_BF16:
        ঘূৰাই ov::element::bf16;
    পক্ষ GGML_TYPE_I8:
        ঘূৰাই ov::element::i8;
    পক্ষ GGML_TYPE_I16:
        ঘূৰাই ov::element::i16;
    পক্ষ GGML_TYPE_I32:
        ঘূৰাই ov::element::i32;
    পক্ষ GGML_TYPE_I64:
        ঘূৰাই ov::element::i64;
    স্ৱভাৱ:
        ঘূৰাই ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_প্ৰবেশ_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name) ধ্রুৱ {
    ঘূৰাই ov::PartialShape(get_shape(m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_প্ৰবেশ_stride(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name) ধ্রুৱ {
    ঘূৰাই get_stride(m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_প্ৰবেশ_size(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        ঘূৰাই it->second.size();
    }
    ঘূৰাই 0;
}

size_t GgmlOvDecoder::get_view_প্ৰবেশ_offset(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        যদি (view_index < it->second.size()) {
            ঘূৰাই it->second[view_index].second->view_offs;
        }
    }
    ঘূৰাই 0;
}

size_t GgmlOvDecoder::get_view_প্ৰবেশ_src_offset(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        যদি (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            যদি (view_tensor && view_tensor->src[0]) {
                ঘূৰাই view_tensor->src[0]->view_offs;
            }
        }
    }
    ঘূৰাই 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_প্ৰবেশ_stride(পূর্ণাঙ্ক গ্ৰন্থি_idx,
                                                         ধ্রুৱ std::string & name,
                                                         size_t view_index) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        যদি (view_index < it->second.size()) {
            ঘূৰাই get_stride(it->second[view_index].second);
        }
    }
    ঘূৰাই {};
}

std::vector<size_t> GgmlOvDecoder::get_view_প্ৰবেশ_src_stride(পূর্ণাঙ্ক গ্ৰন্থি_idx,
                                                             ধ্রুৱ std::string & name,
                                                             size_t view_index) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        যদি (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            যদি (view_tensor && view_tensor->src[0]) {
                ঘূৰাই get_stride(view_tensor->src[0]);
            }
        }
    }
    ঘূৰাই {};
}

ov::Shape GgmlOvDecoder::get_view_প্ৰবেশ_ggml_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        যদি (view_index < it->second.size()) {
            ঘূৰাই get_shape(it->second[view_index].second);
        }
    }
    ঘূৰাই {};
}

ov::Shape GgmlOvDecoder::get_view_প্ৰবেশ_src_ggml_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx,
                                                       ধ্রুৱ std::string & name,
                                                       size_t view_index) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        যদি (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            যদি (view_tensor && view_tensor->src[0]) {
                ঘূৰাই get_shape(view_tensor->src[0]);
            }
        }
    }
    ঘূৰাই {};
}

ov::PartialShape GgmlOvDecoder::get_view_প্ৰবেশ_ov_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx,
                                                        ধ্রুৱ std::string & name,
                                                        size_t view_index) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        যদি (view_index < it->second.size()) {
            auto * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            auto dynamic_it = m_গ্ৰন্থি_dynamic_dims.find(tensor);
            যদি (dynamic_it != m_গ্ৰন্থি_dynamic_dims.end() && dynamic_it->second != -1) {
                পূর্ণাঙ্ক dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            ঘূৰাই shape;
        }
    }
    ঘূৰাই {};
}

ov::PartialShape GgmlOvDecoder::get_view_প্ৰবেশ_src_ov_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx,
                                                            ধ্রুৱ std::string & name,
                                                            size_t view_index) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        যদি (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            যদি (view_tensor && view_tensor->src[0]) {
                auto * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                auto dynamic_it = m_গ্ৰন্থি_dynamic_dims.find(src_tensor);
                যদি (dynamic_it != m_গ্ৰন্থি_dynamic_dims.end() && dynamic_it->second != -1) {
                    পূর্ণাঙ্ক dynamic_dim_index = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
                }

                ঘূৰাই shape;
            }
        }
    }
    ঘূৰাই {};
}

std::string GgmlOvDecoder::get_view_প্ৰবেশ_name(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        যদি (view_index < it->second.size()) {
            ঘূৰাই it->second[view_index].first;
        }
    }
    ঘূৰাই "";
}

std::string GgmlOvDecoder::get_view_প্ৰবেশ_src_name(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.find(name);
    যদি (it != m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_views.end()) {
        যদি (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            যদি (view_tensor && view_tensor->src[0]) {
                ঘূৰাই get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    ঘূৰাই "";
}

ov::element::Type GgmlOvDecoder::get_প্ৰবেশ_type(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name) ধ্রুৱ {
    ঘূৰাই get_ov_type(m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs.at(name));
}

size_t GgmlOvDecoder::get_প্ৰবেশ_size() ধ্রুৱ {
    ঘূৰাই m_model_inputs.size();
}

size_t GgmlOvDecoder::get_প্ৰবেশ_size(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    ঘূৰাই m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_প্ৰবেশ_names(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    ঘূৰাই m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    auto * ggml_tensor = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি;
    ঘূৰাই ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ধ্রুৱ পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    ঘূৰাই get_ov_type(m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    auto * ggml_tensor = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি;
    ঘূৰাই get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    auto * গ্ৰন্থি = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি;
    যদি (গ্ৰন্থি->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        বাবে (পূর্ণাঙ্ক i = গ্ৰন্থি_idx + 1; i < m_cgraph->n_nodes; i++) {
            auto * consumer = m_cgraph->nodes[i];
            যদি (consumer->op != GGML_OP_VIEW || consumer->src[0] != গ্ৰন্থি) {
                অনুৱর্ত;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            auto & name = consumer->view_offs == 0 ? attn_name : state_name;
            যদি (!name.empty()) {
                ঘূৰাই {m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        যদি (!attn_name.empty() && !state_name.empty()) {
            ঘূৰাই {attn_name, state_name};
        }
    }
    ঘূৰাই {m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    auto * গ্ৰন্থি = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি;
    যদি (!::is_inplace_op(গ্ৰন্থি) || গ্ৰন্থি->view_src == nullptr || ggml_nbytes(গ্ৰন্থি) == 0) {
        ঘূৰাই "";
    }
    ধ্রুৱ পূর্ণাঙ্ক op_case = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_op_case;
    যদি (গ্ৰন্থি->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        ঘূৰাই "";
    }
    ঘূৰাই get_tensor_ov_name(m_cgraph, গ্ৰন্থি->view_src);
}

তার্কিক GgmlOvDecoder::is_view_like_alias_of(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & view_src_name) ধ্রুৱ {
    auto * গ্ৰন্থি = m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি;
    যদি (গ্ৰন্থি->view_src == nullptr || get_tensor_ov_name(m_cgraph, গ্ৰন্থি->view_src) != view_src_name) {
        ঘূৰাই মিছা;
    }
    ঘূৰাই গ্ৰন্থি->op == GGML_OP_RESHAPE || গ্ৰন্থি->op == GGML_OP_VIEW;
}

ধ্রুৱ std::string & GgmlOvDecoder::get_op_name() ধ্রুৱ {
    স্থির ধ্রুৱ std::string unknown_name = "UNKNOWN_OP_NAME";
    ঘূৰাই unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    auto it = m_গ্ৰন্থি_dynamic_dims.find(m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি);
    যদি (it == m_গ্ৰন্থি_dynamic_dims.end()) {
        ঘূৰাই -1;
    }
    ঘূৰাই it->second;
}

ধ্রুৱ std::string & GgmlOvDecoder::get_op_name(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    ঘূৰাই m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_name;
}

int32_t * GgmlOvDecoder::get_প্ৰবেশ_op_params(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name) ধ্রুৱ {
    ঘূৰাই m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    ঘূৰাই m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি->op_params;
}

size_t GgmlOvDecoder::get_output_op_offset(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    ঘূৰাই m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি->view_offs;
}

শূন্য GgmlOvDecoder::visit_subgraph(std::function<শূন্য(std::shared_ptr<GgmlDecoder>, পূর্ণাঙ্ক গ্ৰন্থি_idx)> গ্ৰন্থি_visitor) ধ্রুৱ {
    বাবে (পূর্ণাঙ্ক গ্ৰন্থি_idx = 0; গ্ৰন্থি_idx < m_cgraph->n_nodes; গ্ৰন্থি_idx++) {
        যদি (m_cgraph->nodes[গ্ৰন্থি_idx]->op == GGML_OP_NONE) {
            অনুৱর্ত;
        }
        গ্ৰন্থি_visitor(std::make_shared<GgmlOvDecoder>(*এতত্), গ্ৰন্থি_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ধ্রুৱ ggml_tensor * গ্ৰন্থি) {
    ৱিকল্প (গ্ৰন্থি->op) {
    পক্ষ GGML_OP_UNARY:
        ঘূৰাই std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(গ্ৰন্থি));
    পক্ষ GGML_OP_GLU:
        ঘূৰাই std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(গ্ৰন্থি));
    স্ৱভাৱ:
        ঘূৰাই std::string("GGML_OP_") + ggml_op_name(গ্ৰন্থি->op);
    }
}

ধ্রুৱ std::string & GgmlOvDecoder::get_op_type(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ {
    ঘূৰাই m_গ্ৰন্থি_info_list[গ্ৰন্থি_idx].গ্ৰন্থি_op_type;
}

ধ্রুৱ std::string & GgmlOvDecoder::get_op_type() ধ্রুৱ {
    স্থির ধ্রুৱ std::string unknown_op = "UNKNOWN_GGML_OP";
    ঘূৰাই unknown_op;
}

শূন্য GgmlOvDecoder::compute_গ্ৰন্থি_dynamic_dims() {
    auto visit_গ্ৰন্থি = [&](auto && self, ggml_tensor * গ্ৰন্থি) -> শূন্য {
        যদি (!গ্ৰন্থি) {
            ঘূৰাই;
        }

        যদি (গ্ৰন্থি->op == GGML_OP_CPY) {
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
        }

        যদি (m_গ্ৰন্থি_dynamic_dims.count(গ্ৰন্থি)) {
            ঘূৰাই;
        }
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = গ্ৰন্থি->src[i];
            যদি (src == nullptr) {
                অনুৱর্ত;
            }
            সংরচনা ggml_tensor * মূল_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            যদি (মূল_src) {
                যদি (is_inp_tok(মূল_src, গ্ৰন্থি) || is_inp_pos(মূল_src, গ্ৰন্থি) || is_output_idx(মূল_src, গ্ৰন্থি)) {
                    m_গ্ৰন্থি_dynamic_dims[মূল_src] = 0;
                    m_গ্ৰন্থি_dynamic_dims[src] = m_গ্ৰন্থি_dynamic_dims[মূল_src];
                    অনুৱর্ত;
                }
                self(self, মূল_src);
                m_গ্ৰন্থি_dynamic_dims[src] = m_গ্ৰন্থি_dynamic_dims[মূল_src];
            } অন্যথা {
                যদি (is_inp_tok(src, গ্ৰন্থি) || is_inp_pos(src, গ্ৰন্থি) || is_output_idx(src, গ্ৰন্থি)) {
                    m_গ্ৰন্থি_dynamic_dims[src] = 0;
                    অনুৱর্ত;
                }
                যদি (is_inp_scale_rows(src, গ্ৰন্থি)) {
                    m_গ্ৰন্থি_dynamic_dims[src] = 1;
                    অনুৱর্ত;
                }
                যদি (গ্ৰন্থি->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_গ্ৰন্থি_dynamic_dims[src] = 1;
                    অনুৱর্ত;
                }
                self(self, src);
            }
        }
        ৱিকল্প (গ্ৰন্থি->op) {
        পক্ষ GGML_OP_NONE:
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            ভঙ্গ;
        পক্ষ GGML_OP_GET_ROWS:
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[1]] != -1) {
                auto dynamic_dim_idx = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[1]];
                যদি (dynamic_dim_idx == 0) {
                    m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = 1;
                } অন্যথা {
                    auto dynamic_dim_stride = গ্ৰন্থি->src[1]->nb[dynamic_dim_idx] / ggml_type_size(গ্ৰন্থি->src[1]->type) *
                                              ggml_type_size(গ্ৰন্থি->src[0]->type);
                    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
                        যদি (dynamic_dim_stride == গ্ৰন্থি->src[0]->nb[i]) {
                            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = i;
                            ভঙ্গ;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            ভঙ্গ;
        পক্ষ GGML_OP_MUL:
        পক্ষ GGML_OP_MUL_MAT:
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]] != -1) {
                m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]];
            }
            যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[1]] != -1) {
                m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[1]];
            }
            ভঙ্গ;
        পক্ষ GGML_OP_PERMUTE:
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]] != -1) {
                auto dynamic_dim_idx = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
                    যদি (গ্ৰন্থি->op_params[i] == dynamic_dim_idx) {
                        m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = i;
                        ভঙ্গ;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            ভঙ্গ;
        পক্ষ GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]] != -1) {
                যদি (গ্ৰন্থি->src[0]->op == GGML_OP_NONE) {
                    m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]];
                    ভঙ্গ;
                }
                auto dynamic_dim_idx = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]];
                auto dynamic_dim_মান = গ্ৰন্থি->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    গ্ৰন্থি->src[0]->nb[dynamic_dim_idx] / ggml_type_size(গ্ৰন্থি->src[0]->type) * ggml_type_size(গ্ৰন্থি->type);
                বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
                    যদি (গ্ৰন্থি->nb[i] == dynamic_dim_stride) {
                        m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = i;
                        ভঙ্গ;
                    }
                }
                যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] != -1 && dynamic_dim_মান != গ্ৰন্থি->ne[m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি]]) {
                    m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    যদি (ggml_nelements(গ্ৰন্থি) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      গ্ৰন্থি->name, গ্ৰন্থি->src[0]->name);
                    }
                }
            }
            ভঙ্গ;
        }
        পক্ষ GGML_OP_TRANSPOSE:
        পক্ষ GGML_OP_RESHAPE: {
            যদি (is_same_shape(গ্ৰন্থি->src[0], গ্ৰন্থি)) {
                m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]];
                ভঙ্গ;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]] != -1) {
                auto dynamic_dim_idx = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]];
                auto dynamic_dim_stride = গ্ৰন্থি->src[0]->nb[dynamic_dim_idx];
                বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
                    যদি (গ্ৰন্থি->nb[i] == dynamic_dim_stride && গ্ৰন্থি->ne[i] == গ্ৰন্থি->src[0]->ne[dynamic_dim_idx]) {
                        m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = i;
                        ভঙ্গ;
                    }
                }
                যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", গ্ৰন্থি->name);
                }
            }
            ভঙ্গ;
        }
        পক্ষ GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr পূর্ণাঙ্ক q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]] != -1) {
                auto q_dynamic_dim = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]];
                m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = q_to_out[q_dynamic_dim];
            }
            ভঙ্গ;
        }
        পক্ষ GGML_OP_CONT:
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]] != -1) {
                auto dynamic_dim_idx = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]];
                যদি (ggml_are_same_shape(গ্ৰন্থি, গ্ৰন্থি->src[0])) {
                    m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = dynamic_dim_idx;
                } অন্যথা {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(গ্ৰন্থি->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (গ্ৰন্থি->src[0]->ne[0] / ggml_blck_size(গ্ৰন্থি->src[0]->type));
                    বাবে (পূর্ণাঙ্ক i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * গ্ৰন্থি->src[0]->ne[i - 1];
                    }

                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(গ্ৰন্থি->src[0]->type) *
                                              ggml_type_size(গ্ৰন্থি->type);
                    পূর্ণাঙ্ক matched_dim_গুনতি = 0;
                    পূর্ণাঙ্ক first_matched_dim = -1;
                    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
                        যদি (গ্ৰন্থি->nb[i] == dynamic_dim_stride && গ্ৰন্থি->ne[i] == গ্ৰন্থি->src[0]->ne[dynamic_dim_idx]) {
                            যদি (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = i;
                            matched_dim_গুনতি++;
                        }
                    }
                    যদি (matched_dim_গুনতি > 1 && গ্ৰন্থি->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = first_matched_dim;
                    } অন্যথা যদি (matched_dim_গুনতি != 1) {
                        m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      গ্ৰন্থি->name, গ্ৰন্থি->src[0]->name);
                    }
                }
            }
            ভঙ্গ;
        পক্ষ GGML_OP_CONCAT:
            বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
                যদি (গ্ৰন্থি->src[0]->ne[i] != গ্ৰন্থি->ne[i]) {
                    m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = i;
                    ভঙ্গ;
                }
            }
            ভঙ্গ;
        পক্ষ GGML_OP_SSM_CONV:
        পক্ষ GGML_OP_GATED_DELTA_NET:
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = 1;
            ভঙ্গ;
        পক্ষ GGML_OP_RMS_NORM:
        পক্ষ GGML_OP_L2_NORM:
        পক্ষ GGML_OP_NORM:
        পক্ষ GGML_OP_ADD:
        পক্ষ GGML_OP_SUB:
        পক্ষ GGML_OP_GLU:
        পক্ষ GGML_OP_ROPE:
        পক্ষ GGML_OP_SCALE:
        পক্ষ GGML_OP_SOFT_MAX:
        পক্ষ GGML_OP_ARGSORT:
        পক্ষ GGML_OP_ADD_ID:
        পক্ষ GGML_OP_UNARY:
        পক্ষ GGML_OP_CUMSUM:
        পক্ষ GGML_OP_FILL:
        পক্ষ GGML_OP_SET:
        পক্ষ GGML_OP_DIAG:
        পক্ষ GGML_OP_TRI:
        পক্ষ GGML_OP_REPEAT:
        পক্ষ GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        পক্ষ GGML_OP_DIV:
        পক্ষ GGML_OP_CLAMP:
        পক্ষ GGML_OP_PAD:
        পক্ষ GGML_OP_UPSCALE:
        পক্ষ GGML_OP_SIN:
        পক্ষ GGML_OP_COS:
        পক্ষ GGML_OP_LOG:
        পক্ষ GGML_OP_ROLL:
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]];
            ভঙ্গ;
        পক্ষ GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] =
                (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]] == 0) ? -1 : m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[0]];
            ভঙ্গ;
        পক্ষ GGML_OP_MUL_MAT_ID:
        পক্ষ GGML_OP_SOLVE_TRI:
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[1]];
            ভঙ্গ;
        পক্ষ GGML_OP_CPY:
        পক্ষ GGML_OP_SET_ROWS:
        পক্ষ GGML_OP_SUM:
        পক্ষ GGML_OP_MEAN:
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            ভঙ্গ;
        পক্ষ GGML_OP_IM2COL: {
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[1]] != -1) {
                ধ্রুৱ তার্কিক is_2D = গ্ৰন্থি->op_params[6] == 1;
                ধ্রুৱ পূর্ণাঙ্ক src_dyn = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[1]];
                যদি (is_2D) {
                    যদি (src_dyn == 0) {
                        m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = 1;  // IW -> OW
                    } অন্যথা যদি (src_dyn == 1) {
                        m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = 2;  // IH -> OH
                    } অন্যথা যদি (src_dyn == 3) {
                        m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = 3;  // N  -> N
                    }
                } অন্যথা {
                    যদি (src_dyn == 0) {
                        m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = 1;  // IW -> OW
                    } অন্যথা যদি (src_dyn == 2) {
                        m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] != -1) {
                    OPENVINO_ASSERT(গ্ৰন্থি->src[1]->ne[src_dyn] == গ্ৰন্থি->ne[m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(গ্ৰন্থি->name) +
                                        " and its src[1]: " + std::string(গ্ৰন্থি->src[1]->name));
                }
            }
            ভঙ্গ;
        }
        পক্ষ GGML_OP_IM2COL_3D: {
            m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = -1;
            যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[1]] != -1) {
                ধ্রুৱ পূর্ণাঙ্ক src_dyn = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি->src[1]];
                যদি (src_dyn == 0) {
                    m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = 1;  // IW -> OW
                } অন্যথা যদি (src_dyn == 1) {
                    m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = 2;  // IH -> OH
                } অন্যথা যদি (src_dyn == 3) {
                    m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] = 3;  // N  -> N
                }
                যদি (m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি] != -1) {
                    OPENVINO_ASSERT(গ্ৰন্থি->src[1]->ne[src_dyn] == গ্ৰন্থি->ne[m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(গ্ৰন্থি->name) +
                                        " and its src[1]: " + std::string(গ্ৰন্থি->src[1]->name));
                }
            }
            ভঙ্গ;
        }
        স্ৱভাৱ:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(গ্ৰন্থি->op), গ্ৰন্থি->name);
            ভঙ্গ;
        }
    };

    বাবে (পূর্ণাঙ্ক i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * গ্ৰন্থি = m_cgraph->nodes[i];
        visit_গ্ৰন্থি(visit_গ্ৰন্থি, গ্ৰন্থি);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    যদি (0) {
        বাবে (পূর্ণাঙ্ক i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * গ্ৰন্থি = m_cgraph->nodes[i];
            পূর্ণাঙ্ক dynamic_dim = m_গ্ৰন্থি_dynamic_dims[গ্ৰন্থি];
            std::cout << "[" << i << "] " << "node_name: " << গ্ৰন্থি->name << " op: " << ggml_op_name(গ্ৰন্থি->op)
                      << " shape: [";
            বাবে (পূর্ণাঙ্ক j = 0; j < 4; j++) {
                যদি (j == dynamic_dim) {
                    std::cout << "*";
                } অন্যথা {
                    std::cout << গ্ৰন্থি->ne[j];
                }
                যদি (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = গ্ৰন্থি->src[j];
                যদি (src == nullptr) {
                    অনুৱর্ত;
                }
                পূর্ণাঙ্ক src_dynamic_dim = m_গ্ৰন্থি_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                বাবে (পূর্ণাঙ্ক k = 0; k < 4; k++) {
                    যদি (k == src_dynamic_dim) {
                        std::cout << "*";
                    } অন্যথা {
                        std::cout << src->ne[k];
                    }
                    যদি (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
