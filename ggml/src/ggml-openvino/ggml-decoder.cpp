#अन्तर्भाव "ggml-decoder.h"

#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-openvino-extra.h"
#अन्तर्भाव "ggml-openvino.h"
#अन्तर्भाव "ggml-quants.h"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "utils.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cassert>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <fstream>
#अन्तर्भाव <iomanip>
#अन्तर्भाव <map>
#अन्तर्भाव <memory>
#अन्तर्भाव <mutex>
#अन्तर्भाव <openvino/core/dimension.hpp>
#अन्तर्भाव <openvino/core/except.hpp>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/partial_shape.hpp>
#अन्तर्भाव <openvino/core/type/bfloat16.hpp>
#अन्तर्भाव <openvino/core/type/element_type.hpp>
#अन्तर्भाव <openvino/core/type/float16.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/runtime/tensor.hpp>
#अन्तर्भाव <ostream>
#अन्तर्भाव <set>
#अन्तर्भाव <stdexcept>
#अन्तर्भाव <string>
#अन्तर्भाव <cstring>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             तार्किक is_static,
                             तार्किक is_stateful,
                             तार्किक model_is_splitted,
                             तार्किक is_prefill,
                             पूर्णाङ्क prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(मिच्छ),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    स्थिर तार्किक printed_address_map = मिच्छ;
    जइ (!printed_address_map) {
        जइ (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = सच्च;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_निवेशः_output();
    compute_ग्रन्थिः_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    प्रति (पूर्णाङ्क ग्रन्थिः_n = 0; ग्रन्थिः_n < cgraph->n_nodes; ग्रन्थिः_n++) {
        m_ग्रन्थिः_info_list[ग्रन्थिः_n].ग्रन्थिः_op_case = compute_op_case(m_ग्रन्थिः_info_list[ग्रन्थिः_n].ग्रन्थिः);
        m_ग्रन्थिः_info_list[ग्रन्थिः_n].ग्रन्थिः_op_type = compute_op_type(m_ग्रन्थिः_info_list[ग्रन्थिः_n].ग्रन्थिः);
    }

    add_extra_inputs();
}

शून्य GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_ग्रन्थिः_info_list.clear();
    set_निवेशः_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
तार्किक GgmlOvDecoder::is_bound_to(ध्रुव ggml_cgraph * cgraph) ध्रुव {
    प्रत्यागम m_cgraph == cgraph && cgraph->n_nodes > 0 && m_ग्रन्थिः_info_list.size() == (size_t) cgraph->n_nodes &&
           m_ग्रन्थिः_info_list.front().ग्रन्थिः == cgraph->nodes[0] &&
           m_ग्रन्थिः_info_list.back().ग्रन्थिः == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = सच्च;
    set_निवेशः_output();
    compute_model_inputs();
    compute_model_outputs();
    प्रति (पूर्णाङ्क ग्रन्थिः_n = 0; ग्रन्थिः_n < cgraph->n_nodes; ग्रन्थिः_n++) {
        m_ग्रन्थिः_info_list[ग्रन्थिः_n].ग्रन्थिः_op_case = compute_op_case(m_ग्रन्थिः_info_list[ग्रन्थिः_n].ग्रन्थिः);
        m_ग्रन्थिः_info_list[ग्रन्थिः_n].ग्रन्थिः_op_type = compute_op_type(m_ग्रन्थिः_info_list[ग्रन्थिः_n].ग्रन्थिः);
    }
}

नामाकाश {
तार्किक is_inplace_op(ध्रुव ggml_tensor * ग्रन्थिः) {
    प्रत्यागम ग्रन्थिः->op == GGML_OP_SET_ROWS || ग्रन्थिः->op == GGML_OP_CPY || (ग्रन्थिः->op == GGML_OP_SCALE && ग्रन्थिः->view_src);
}

तार्किक is_same_shape(ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        जइ (a->ne[i] != b->ne[i]) {
            प्रत्यागम मिच्छ;
        }
    }
    प्रत्यागम सच्च;
}

तार्किक is_conv_states_all_tensor(ध्रुव ggml_tensor * tensor) {
    प्रत्यागम tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

तार्किक is_full_single_slot_writeback(ध्रुव ggml_tensor * ग्रन्थिः) {
    प्रत्यागम ग्रन्थिः->view_src != nullptr && ग्रन्थिः->view_src->ne[1] == 1 && ग्रन्थिः->src[1] != nullptr &&
           ग्रन्थिः->src[1]->op == GGML_OP_VIEW && ग्रन्थिः->src[1]->view_src == ग्रन्थिः->view_src &&
           ग्रन्थिः->src[1]->view_offs == 0 && ggml_nbytes(ग्रन्थिः->src[1]) == ggml_nbytes(ग्रन्थिः->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
तार्किक is_moe_expert_sum_add(ध्रुव ggml_tensor * ग्रन्थिः) {
    std::vector<ध्रुव ggml_tensor *> leaves;
    ध्रुव ggml_tensor * cur = ग्रन्थिः;
    जाव (cur->op == GGML_OP_ADD) {
        जइ (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            प्रत्यागम मिच्छ;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ध्रुव ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    प्रति (ध्रुव ggml_tensor * leaf : leaves) {
        जइ (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            प्रत्यागम मिच्छ;
        }
        ध्रुव ggml_tensor * leaf_base = leaf->src[0];
        जइ (base == nullptr) {
            base = leaf_base;
        } अण्णहा जइ (leaf_base != base) {
            प्रत्यागम मिच्छ;
        }
        जइ (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            प्रत्यागम मिच्छ;
        }
        जइ (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            प्रत्यागम मिच्छ;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        जइ (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            प्रत्यागम मिच्छ;
        }
    }

    प्रत्यागम base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ध्रुव ggml_cgraph * cgraph, ध्रुव ggml_tensor * tensor) {
    जइ (tensor == nullptr) {
        प्रत्यागम "";
    }
    जइ ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ध्रुव auto * ग्रन्थिः = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        जइ (ग्रन्थिः != cgraph->nodes + cgraph->n_nodes) {
            प्रत्यागम std::string(tensor->name) + "#n" + std::to_string(ग्रन्थिः - cgraph->nodes);
        }
        ध्रुव auto * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        जइ (leaf != cgraph->leafs + cgraph->n_leafs) {
            प्रत्यागम std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    प्रत्यागम tensor->name;
}

स्थिर std::string get_tensor_ov_name(ध्रुव ggml_cgraph * cgraph, ध्रुव ggml_tensor * tensor) {
    प्रत्यागम GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

स्थिर std::string get_tensor_graph_निवेशः_ov_name(ध्रुव GgmlOvDecoder * decoder,
                                                  ध्रुव ggml_cgraph * cgraph,
                                                  ध्रुव ggml_tensor * tensor,
                                                  ध्रुव ggml_tensor * op) {
    जइ (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        प्रत्यागम "inp_pos";
    }
    जइ (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        प्रत्यागम "embd";
    }
    जइ (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ध्रुव तार्किक is_swa = decoder->is_swa_mask(tensor);
        जइ (decoder->is_stateful()) {
            प्रत्यागम is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        जइ (is_swa) {
            प्रत्यागम get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    प्रत्यागम get_tensor_ov_name(cgraph, tensor);
}

शून्य GgmlOvDecoder::set_निवेशः_output() {
    प्रति (पूर्णाङ्क ग्रन्थिः_n = 0; ग्रन्थिः_n < m_cgraph->n_nodes; ग्रन्थिः_n++) {
        auto * ग्रन्थिः = m_cgraph->nodes[ग्रन्थिः_n];

        ग्रन्थिः_Info current_ग्रन्थिः_info;
        auto ग्रन्थिः_name = get_tensor_ov_name(m_cgraph, ग्रन्थिः);

        current_ग्रन्थिः_info.ग्रन्थिः = ग्रन्थिः;
        current_ग्रन्थिः_info.ग्रन्थिः_name = ग्रन्थिः_name;
        current_ग्रन्थिः_info.ग्रन्थिः_op_case = 0;
        current_ग्रन्थिः_info.data_addr = ग्रन्थिः->data;

        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ग्रन्थिः->src[i];
            जइ (src == nullptr) {
                अनुवर्त;
            }
            auto src_name = get_tensor_ov_name(m_cgraph, src);
            जइ (src->flags & GGML_TENSOR_FLAG_निवेशः) {
                src_name = get_tensor_graph_निवेशः_ov_name(एतत्, m_cgraph, src, ग्रन्थिः);
            }
            current_ग्रन्थिः_info.ग्रन्थिः_inputs[src_name] = src;
            current_ग्रन्थिः_info.ग्रन्थिः_inputs_names.push_back(src_name);

            जइ (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_ग्रन्थिः_info.ग्रन्थिः_inputs_views[src_name])> view_chain;
                auto * current = src;

                जाव (current != nullptr) {
                    auto current_name = get_tensor_ov_name(m_cgraph, current);
                    जइ (current->flags & GGML_TENSOR_FLAG_निवेशः) {
                        current_name = get_tensor_graph_निवेशः_ov_name(एतत्, m_cgraph, current, ग्रन्थिः);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    जइ (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } अण्णहा {
                        भङ्ग;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_ग्रन्थिः_info.ग्रन्थिः_inputs_views[src_name] = view_chain;
            }
        }

        m_ग्रन्थिः_info_list.push_back(current_ग्रन्थिः_info);
    }
}

पूर्णाङ्क GgmlOvDecoder::compute_op_case(ध्रुव ggml_tensor * ग्रन्थिः) ध्रुव {
    पूर्णाङ्क op_case = 0;
    विकल्प (ग्रन्थिः->op) {
    पक्ष GGML_OP_RESHAPE: {
        जइ (m_naive) {
            भङ्ग;
        }
        auto name = std::string(ग्रन्थिः->name);
        auto * src = ग्रन्थिः->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        तार्किक recurrent_sequence = मिच्छ;
        प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ध्रुव auto * consumer = m_cgraph->nodes[i];
            जइ (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == ग्रन्थिः) {
                प्रत्यागम 1;
            } अण्णहा जइ (consumer->op == GGML_OP_SSM_CONV) {
                ध्रुव auto * concat = consumer->src[0];
                जइ (concat->op == GGML_OP_CONCAT) {
                    ध्रुव auto * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == ग्रन्थिः;
                }
            } अण्णहा जइ (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                ध्रुव auto * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == ग्रन्थिः;
            }
        }
        जइ (recurrent_sequence && ग्रन्थिः->ne[0] == src->ne[0] && ग्रन्थिः->ne[3] == 1) {
            प्रत्यागम 6;
        }
        जइ (ग्रन्थिः->ne[0] == src->ne[0] && ग्रन्थिः->ne[2] == 1 && ग्रन्थिः->ne[3] == 1) {
            प्रत्यागम 5;
        }
        जइ (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == ग्रन्थिः->ne[0] && src->src[0]->ne[1] == ग्रन्थिः->ne[1]) {
            op_case = 4;
        } अण्णहा जइ (ग्रन्थिः->ne[0] * ग्रन्थिः->ne[1] == src->ne[0]) {
            op_case = 1;
        } अण्णहा जइ (src->ne[0] * src->ne[1] == ग्रन्थिः->ne[0]) {
            op_case = 2;
            जइ (src->ne[2] * src->ne[3] == ग्रन्थिः->ne[1]) {
                op_case = 5;
            }
        } अण्णहा जइ (ग्रन्थिः->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == ग्रन्थिः->ne[1]) {
            op_case = 3;
        } अण्णहा जइ (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } अण्णहा जइ (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } अण्णहा जइ (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        जइ (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            ध्रुव auto * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            तार्किक recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ध्रुव auto * consumer = m_cgraph->nodes[i];
                जइ (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    प्रति (पूर्णाङ्क j : {3, 4}) {
                        ध्रुव auto * gate = consumer->src[j];
                        जइ (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == ग्रन्थिः;
                    }
                } अण्णहा जइ (consumer->op == GGML_OP_MUL) {
                    प्रति (पूर्णाङ्क j = 0; j < 2; ++j) {
                        ध्रुव auto * gate = consumer->src[j];
                        ध्रुव auto * norm = consumer->src[1 - j];
                        जइ (gate->op != GGML_OP_UNARY || gate->src[0] != ग्रन्थिः) {
                            अनुवर्त;
                        }
                        जइ (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            जइ (recurrent) {
                op_case = 9;
            }
        }
        भङ्ग;
    }
    पक्ष GGML_OP_PERMUTE: {
        जइ (ग्रन्थिः->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } अण्णहा जइ (ग्रन्थिः->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(ग्रन्थिः->view_src->name);
            पूर्णाङ्क layer = extract_layer_from_name(src_name).value();
            जइ (ggml_is_contiguous(ग्रन्थिः->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                जइ (!is_swa_layer(layer)) {
                    op_case = 3;
                } अण्णहा {
                    op_case = 4;
                }
            } अण्णहा {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                जइ (!is_swa_layer(layer)) {
                    op_case = 5;
                } अण्णहा {
                    op_case = 6;
                }
            }
        } अण्णहा {
            // rope'ed query tensor
            op_case = 2;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_MUL_MAT: {
        जइ (ग्रन्थिः->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_GET_ROWS: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        जइ (ग्रन्थिः->src[0]->op == GGML_OP_RESHAPE && ग्रन्थिः->src[0]->src[0] != nullptr &&
            is_recurrent_cache(ग्रन्थिः->src[0]->src[0])) {
            ध्रुव तार्किक single_slot = ग्रन्थिः->src[0]->src[0]->ne[1] == 1;
            जइ (ग्रन्थिः->src[1]->op == GGML_OP_VIEW) {
                op_case = (ग्रन्थिः->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } अण्णहा जइ (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        भङ्ग;
    }
    पक्ष GGML_OP_ROPE: {
        ध्रुव पूर्णाङ्क mode = ग्रन्थिः->op_params[2];
        विकल्प (mode) {
        पक्ष GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            भङ्ग;
        }
        पक्ष GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            भङ्ग;
        }
        पक्ष GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            भङ्ग;
        }
        पक्ष GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            भङ्ग;
        }
        स्वभाव:
            op_case = 0;
            भङ्ग;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_VIEW: {
        जइ (!m_model_params.has_rs_rollback && ग्रन्थिः->src[0] != nullptr &&
            ग्रन्थिः->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            भङ्ग;
        }
        जइ (m_is_static && ग्रन्थिः->src[0] != nullptr &&
            (ग्रन्थिः->src[0]->op == GGML_OP_GATED_DELTA_NET || ग्रन्थिः->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            भङ्ग;
        }
        जइ (ग्रन्थिः->src[0]->op == GGML_OP_VIEW) {
            auto * src = ग्रन्थिः->src[0];
            जइ (ggml_nelements(ग्रन्थिः) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            जइ (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            auto * src = ग्रन्थिः->src[0];
            जइ (ggml_nelements(ग्रन्थिः) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                जइ (ग्रन्थिः->ne[0] == src->ne[0] && ग्रन्थिः->ne[1] == src->ne[2] && ग्रन्थिः->ne[2] == 1 &&
                    ग्रन्थिः->nb[0] == src->nb[0] && ग्रन्थिः->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    भङ्ग;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                पूर्णाङ्क diff_गणनम् = 0;
                प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                    जइ (ग्रन्थिः->ne[i] != src->ne[i]) {
                        diff_गणनम्++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    जइ (ग्रन्थिः->ne[i] > src->ne[i]) {
                        प्रत्यागम 0;
                    }
                }
                जइ (diff_गणनम् >= 1) {
                    op_case = 0;
                }
            }
        }
        भङ्ग;
    }
    पक्ष GGML_OP_RMS_NORM: {
        जइ (ग्रन्थिः->src[0]->op == GGML_OP_VIEW) {
            जइ (is_same_shape(ग्रन्थिः->src[0]->src[0], ग्रन्थिः->src[0])) {
                op_case = 1;
            } अण्णहा जइ (!m_model_params.has_rs_rollback &&
                       ग्रन्थिः->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } अण्णहा जइ (ग्रन्थिः->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        भङ्ग;
    }
    पक्ष GGML_OP_POOL_2D: {
        ध्रुव ggml_op_pool pool_mode = static_cast<ggml_op_pool>(ग्रन्थिः->op_params[0]);
        विकल्प (pool_mode) {
        पक्ष GGML_OP_POOL_MAX: {
            op_case = 1;
            भङ्ग;
        }
        पक्ष GGML_OP_POOL_AVG: {
            op_case = 2;
            भङ्ग;
        }
        स्वभाव:
            op_case = 0;
            भङ्ग;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_UPSCALE: {
        ध्रुव int32_t mode_flags = ग्रन्थिः->op_params[0];
        ध्रुव ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        विकल्प (scale_mode) {
        पक्ष GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            भङ्ग;
        }
        पक्ष GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            भङ्ग;
        }
        पक्ष GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            भङ्ग;
        }
        स्वभाव:
            op_case = 0;
            भङ्ग;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_CPY: {
        जइ (ग्रन्थिः->src[0]->op == GGML_OP_VIEW) {
            जइ (ग्रन्थिः->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                जइ (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(ग्रन्थिः) ? 7 : 10;
                } अण्णहा {
                    op_case = 1;
                }
            } अण्णहा जइ (GgmlOvDecoder::is_conv_state_writeback(ग्रन्थिः)) {
                op_case = is_full_single_slot_writeback(ग्रन्थिः) ? 8 : 2;
                भङ्ग;
            } अण्णहा जइ (is_conv_states_all_tensor(ग्रन्थिः->view_src) && ग्रन्थिः->src[1] != nullptr &&
                       ग्रन्थिः->src[1]->op == GGML_OP_VIEW && ग्रन्थिः->src[1]->view_src == ग्रन्थिः->view_src) {
                op_case = 4;
                भङ्ग;
            } अण्णहा जइ (ग्रन्थिः->src[0]->src[0]->op == GGML_OP_GET_ROWS && ग्रन्थिः->src[1] != nullptr &&
                       ग्रन्थिः->src[1]->op == GGML_OP_VIEW && ग्रन्थिः->src[1]->view_src != nullptr &&
                       is_recurrent_cache(ग्रन्थिः->src[1]->view_src) && ग्रन्थिः->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } अण्णहा जइ (ग्रन्थिः->src[0]->op == GGML_OP_GET_ROWS && ग्रन्थिः->src[1] != nullptr &&
                   ग्रन्थिः->src[1]->op == GGML_OP_VIEW && ग्रन्थिः->src[1]->view_src != nullptr &&
                   is_recurrent_cache(ग्रन्थिः->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = ग्रन्थिः->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } अण्णहा जइ (ग्रन्थिः->src[1] != nullptr && ग्रन्थिः->src[1]->op == GGML_OP_VIEW && ग्रन्थिः->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ध्रुव ggml_tensor * kv_buf = ग्रन्थिः->src[1]->view_src;
            जइ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
                    ध्रुव ggml_tensor * n = m_cgraph->nodes[i];
                    जइ (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        अनुवर्त;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    जइ ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        जइ (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        भङ्ग;
                    }
                }
            }
        }
        भङ्ग;
    }
    पक्ष GGML_OP_ADD: {
        जइ (is_moe_expert_sum_add(ग्रन्थिः)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_SCALE: {
        जइ (ग्रन्थिः->view_src && ग्रन्थिः->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = ग्रन्थिः->view_src->ne[1] == 1 ? 2 : 1;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_L2_NORM: {
        जइ (std::string(ग्रन्थिः->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_FLASH_ATTN_EXT: {
        जइ (ग्रन्थिः->src[1] != nullptr && ग्रन्थिः->src[1]->op == GGML_OP_VIEW && ग्रन्थिः->src[1]->view_src != nullptr) {
            ध्रुव ggml_tensor * kv_buf = ग्रन्थिः->src[1]->view_src;
            जइ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (ग्रन्थिः->src[3] != nullptr) ? 1 : 2;
            }
        }
        भङ्ग;
    }
    स्वभाव:
        भङ्ग;
    }
    प्रत्यागम op_case;
}

std::optional<पूर्णाङ्क> extract_layer_from_name(ध्रुव std::string & name) {
    size_t pos1 = name.find("_l");
    जइ (pos1 == std::string::npos) {
        प्रत्यागम std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    जइ (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    पूर्णाङ्क layer = std::stoi(layer_str);
    प्रत्यागम layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
स्थिर पूर्णाङ्क get_swa_window_from_mask(ध्रुव ggml_tensor * mask) {
    जइ (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        प्रत्यागम -1;
    }
    जइ (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t n_kv = mask->ne[0];
    ध्रुव int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    प्रति (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        प्रति (int64_t c = 0; c < n_kv; c++) {
            ध्रुव size_t i = (size_t) r * n_kv + c;
            ध्रुव भिन्न v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((ध्रुव ggml_fp16_t *) mask->data)[i]) :
                                                          ((ध्रुव भिन्न *) mask->data)[i];
            जइ (v > -INFINITY) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    प्रत्यागम window > 0 ? (पूर्णाङ्क) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, तार्किक is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    auto get_attention_pattern_case = [](ध्रुव ggml_tensor * ग्रन्थिः) -> पूर्णाङ्क {
        जइ (ग्रन्थिः == nullptr) {
            प्रत्यागम -1;
        }

        विकल्प (ग्रन्थिः->op) {
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            जइ (ग्रन्थिः->src[0] == nullptr || ग्रन्थिः->src[1] == nullptr) {
                प्रत्यागम -1;
            }
            विकल्प (ग्रन्थिः->src[1]->op) {
            पक्ष GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                जइ (ग्रन्थिः->src[3] != nullptr && ग्रन्थिः->src[1]->src[0] != nullptr &&
                    ग्रन्थिः->src[1]->src[0]->op == GGML_OP_VIEW) {
                    प्रत्यागम 0;
                }
                भङ्ग;
            पक्ष GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                जइ (ग्रन्थिः->src[3] != nullptr && ग्रन्थिः->src[1]->src[0] != nullptr &&
                    ग्रन्थिः->src[1]->src[0]->op == GGML_OP_PERMUTE && ग्रन्थिः->src[1]->src[0]->src[0] != nullptr &&
                    ग्रन्थिः->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    प्रत्यागम 1;
                }
                भङ्ग;
            पक्ष GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                जइ (ग्रन्थिः->src[1]->view_src != nullptr) {
                    जइ (ग्रन्थिः->src[3] != nullptr) {
                        प्रत्यागम 4;  // decoder self-attention
                    }
                    प्रत्यागम 5;      // cross-attention or encoder self-attention
                }
                भङ्ग;
            स्वभाव:
                भङ्ग;
            }
            भङ्ग;
        पक्ष GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            जइ (ग्रन्थिः->src[0] != nullptr && ग्रन्थिः->src[1] != nullptr && ग्रन्थिः->src[0]->op == GGML_OP_MUL_MAT &&
                ग्रन्थिः->src[0]->src[0] != nullptr && ग्रन्थिः->src[0]->src[1] != nullptr &&
                ग्रन्थिः->src[0]->src[0]->op == GGML_OP_PERMUTE && ग्रन्थिः->src[0]->src[0]->src[0] != nullptr &&
                ग्रन्थिः->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                प्रत्यागम 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            जइ (ग्रन्थिः->src[0]->op == GGML_OP_ADD && ग्रन्थिः->src[0]->src[0] != nullptr &&
                ग्रन्थिः->src[0]->src[0]->op == GGML_OP_MUL_MAT && ग्रन्थिः->src[0]->src[0]->src[0] != nullptr &&
                ग्रन्थिः->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                प्रत्यागम 3;
            }
            भङ्ग;
        स्वभाव:
            भङ्ग;
        }

        प्रत्यागम -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    auto get_attention_op_mask = [&get_attention_pattern_case](ध्रुव ggml_tensor * ग्रन्थिः) -> ध्रुव ggml_tensor * {
        विकल्प (get_attention_pattern_case(ग्रन्थिः)) {
        पक्ष 0:
        पक्ष 1:
            प्रत्यागम ग्रन्थिः->src[3];
        पक्ष 2:
        पक्ष 3:
            प्रत्यागम ग्रन्थिः->src[1];
        स्वभाव:
            प्रत्यागम nullptr;
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
        std::map<पूर्णाङ्क, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<पूर्णाङ्क, ध्रुव ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
            ध्रुव ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            जइ (mask == nullptr) {
                अनुवर्त;
            }
            ध्रुव ggml_tensor * cache_k_permute = nullptr;
            विकल्प (get_attention_pattern_case(cgraph->nodes[i])) {
            पक्ष 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     भङ्ग;
            पक्ष 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             भङ्ग;
            पक्ष 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             भङ्ग;
            स्वभाव: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     भङ्ग;
            }
            ध्रुव ggml_tensor * cache_k_view = cache_k_permute->src[0];
            जइ (cache_k_view->op != GGML_OP_VIEW) {
                अनुवर्त;
            }
            ध्रुव ggml_tensor * leaf = cache_k_view->src[0];
            auto layer = extract_layer_from_name(leaf->name);
            जइ (!layer.has_value()) {
                अनुवर्त;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        प्रति (ध्रुव auto & [layer, extent] : layer_extent) {
            जइ (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                जइ (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        जइ (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            प्रति (ध्रुव auto & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (long) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    तार्किक rope_seen = मिच्छ;
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ग्रन्थिः = cgraph->nodes[i];
        ध्रुव पूर्णाङ्क attention_pattern_case = get_attention_pattern_case(ग्रन्थिः);
        जइ (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            विकल्प (attention_pattern_case) {
            पक्ष 0:
                cache_k_permute = ग्रन्थिः->src[1];
                mask = ग्रन्थिः->src[3];
                भङ्ग;
            पक्ष 1:
                cache_k_permute = ग्रन्थिः->src[1]->src[0];
                mask = ग्रन्थिः->src[3];
                भङ्ग;
            पक्ष 2:
                cache_k_permute = ग्रन्थिः->src[0]->src[0];
                mask = ग्रन्थिः->src[1];
                भङ्ग;
            पक्ष 3:
                cache_k_permute = ग्रन्थिः->src[0]->src[0]->src[0];
                mask = ग्रन्थिः->src[1];
                भङ्ग;
            पक्ष 4:
            पक्ष 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                auto * cache_k_view = ग्रन्थिः->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = ग्रन्थिः->src[0]->ne[1];
                जइ (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } अण्णहा {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                अनुवर्त;
            }
            स्वभाव:
                भङ्ग;
            }

            प्रतिज्ञा(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.निवेशः_len = ग्रन्थिः->src[0]->ne[1];
            compute_params.token_len_per_seq = ग्रन्थिः->src[0]->ne[1];

            auto * cache_k_view = cache_k_permute->src[0];
            जइ (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                अनुवर्त;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            पूर्णाङ्क layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ध्रुव तार्किक layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            जइ (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } अण्णहा {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, परिमाण(size_t));
            compute_params.seq_active_start = offset / seq_size;

            जइ (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } अण्णहा {
                compute_params.attention_size = mask->ne[0];
            }
            जइ (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        जइ (ग्रन्थिः->op == GGML_OP_MUL_MAT && ग्रन्थिः->src[0]->op == GGML_OP_PERMUTE &&
            ग्रन्थिः->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(ग्रन्थिः->src[0]->view_src, ग्रन्थिः->view_src)) {
            जइ (ग्रन्थिः->src[1]->op == GGML_OP_PERMUTE && ग्रन्थिः->src[1]->src[0]->op == GGML_OP_VIEW &&
                ग्रन्थिः->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = ग्रन्थिः->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        जइ (ग्रन्थिः->op == GGML_OP_TRANSPOSE && ग्रन्थिः->src[0]->op == GGML_OP_PERMUTE &&
            ग्रन्थिः->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = ग्रन्थिः->ne[0];
            जइ (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        जइ (ग्रन्थिः->op == GGML_OP_ROPE) {
            जइ (compute_params.token_len_per_seq == -1 && ग्रन्थिः->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(ग्रन्थिः->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(परिमाण(model_params.rope_params) == परिमाण(int32_t) * 16, "rope_params size");
            जइ (!rope_seen) {
                memcpy(model_params.rope_params, ग्रन्थिः->op_params, परिमाण(int32_t) * 16);
                rope_seen = सच्च;
            } अण्णहा जइ (memcmp(model_params.rope_params, ग्रन्थिः->op_params, परिमाण(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = सच्च;
            }
        }
        जइ (ग्रन्थिः->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = ग्रन्थिः->src[0]->ne[0];
        }
        जइ (ग्रन्थिः->op == GGML_OP_SCALE && ग्रन्थिः->view_src != nullptr && is_recurrent_cache(ग्रन्थिः->view_src)) {
            जइ (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = ग्रन्थिः->view_src->ne[1];
            } अण्णहा {
                GGML_ASSERT(model_params.n_rs_slots == ग्रन्थिः->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(ग्रन्थिः) / ग्रन्थिः->view_src->ne[0];
            compute_params.cache_rs_reset_idx = ग्रन्थिः->src[0]->view_offs / ग्रन्थिः->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        जइ (ग्रन्थिः->op == GGML_OP_CPY && ग्रन्थिः->view_src != nullptr && is_recurrent_cache(ग्रन्थिः->view_src) &&
            ग्रन्थिः->src[1] != nullptr && ग्रन्थिः->src[1]->op == GGML_OP_VIEW && ग्रन्थिः->src[1]->view_src == ग्रन्थिः->view_src) {
            ध्रुव तार्किक is_conv = is_conv_state_writeback(ग्रन्थिः);
            ध्रुव तार्किक is_gdn = ग्रन्थिः->src[0]->op == GGML_OP_VIEW && ग्रन्थिः->src[0]->src[0] != nullptr &&
                                ग्रन्थिः->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ध्रुव तार्किक is_extra = ग्रन्थिः->src[0]->op == GGML_OP_GET_ROWS;
            ध्रुव तार्किक is_gdn_rollback = is_gdn && is_same_shape(ग्रन्थिः->src[0], ग्रन्थिः->src[1]);

            ध्रुव ggml_tensor * dest_view = ग्रन्थिः->src[1];
            ध्रुव ggml_tensor * cache = ग्रन्थिः->view_src;
            ध्रुव size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            जइ (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = सच्च;
            }
            जइ (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(ग्रन्थिः)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (पूर्णाङ्क) (dest_view->view_offs / row_bytes);
                जइ (is_conv) {
                    writeback.src_begin = (पूर्णाङ्क) (ग्रन्थिः->src[0]->view_offs / ग्रन्थिः->src[0]->view_src->nb[0]);
                } अण्णहा जइ (is_gdn_rollback) {
                    writeback.src_begin = (पूर्णाङ्क) (ग्रन्थिः->src[0]->view_offs / ग्रन्थिः->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, ग्रन्थिः)] = writeback;
            }
            जइ ((is_conv || is_gdn) && !is_full_single_slot_writeback(ग्रन्थिः)) {
                compute_params.s_copy_active_slot_len = (पूर्णाङ्क) dest_view->ne[1];
            }
        }
    }
    जइ (model_params.n_heads_kv == -1) {
        प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
            ध्रुव auto * ग्रन्थिः = cgraph->nodes[i];
            ध्रुव ggml_tensor * mask = nullptr;
            जइ (ग्रन्थिः->op == GGML_OP_SOFT_MAX) {
                mask = ग्रन्थिः->src[1];
            } अण्णहा जइ (ग्रन्थिः->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = ग्रन्थिः->src[3];
            } अण्णहा {
                अनुवर्त;
            }
            जइ (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_निवेशः) ||
                ग्रन्थिः->src[0] == nullptr) {
                अनुवर्त;
            }
            model_params.is_cacheless_attn = सच्च;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.निवेशः_len = ग्रन्थिः->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.निवेशः_len;
            भङ्ग;
        }
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    जइ (model_params.is_cacheless_attn) {
        प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
            ध्रुव auto * ग्रन्थिः = cgraph->nodes[i];
            जइ (ग्रन्थिः->op == GGML_OP_GET_ROWS && is_output_idx(ग्रन्थिः->src[1], ग्रन्थिः)) {
                compute_params.output_len = ग्रन्थिः->src[1]->ne[0];
                भङ्ग;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    जइ (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    प्रत्यागम {model_params, compute_params};
}

शून्य GgmlOvDecoder::validate_cgraph() ध्रुव {
    जइ (m_model_params.n_seq > 1 && m_is_static == सच्च) {
        क्षिप std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_निवेशः_shape(ध्रुव ggml_tensor * op,
                                                      ध्रुव ggml_tensor * निवेशः,
                                                      पूर्णाङ्क dynamic_dim_index) ध्रुव {
    जइ (m_naive) {
        प्रत्यागम निवेशः != nullptr ? ov::PartialShape{get_shape(निवेशः)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape निवेशः_shape;

    जइ (is_inp_tok(निवेशः, op) || is_inp_pos(निवेशः, op)) {
        // tokens or positions
        पूर्णाङ्क len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        जइ (m_is_static && is_inp_pos(निवेशः, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        निवेशः_shape = ov::PartialShape{1, 1, 1, len};

    } अण्णहा जइ (is_output_idx(निवेशः, op)) {
        // output index
        निवेशः_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } अण्णहा जइ (is_inp_mean(निवेशः, op)) {
        निवेशः_shape = m_is_static ? ov::PartialShape{1, 1, निवेशः->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } अण्णहा जइ (is_inp_scale_rows(निवेशः, op)) {
        निवेशः_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } अण्णहा जइ (is_inp_mask(निवेशः, op)) {
        // mask
        जइ (m_is_static) {
            निवेशः_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } अण्णहा जइ (m_is_stateful) {
            निवेशः_shape = ov::PartialShape{1, 1, -1, -1};
        } अण्णहा {
            निवेशः_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } अण्णहा जइ (is_recurrent_cache(निवेशः)) {
        निवेशः_shape = ov::PartialShape{get_shape(निवेशः)};
        जइ (!m_is_static && !m_is_stateful && निवेशः->ne[1] > 1) {
            निवेशः_shape[2] = -1;
        }

    } अण्णहा जइ (is_kvcache(निवेशः, op)) {
        // kvcache
        निवेशः_shape = ov::PartialShape{get_shape(निवेशः)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ध्रुव तार्किक is_flat_kv = (निवेशः->ne[1] == 1 && निवेशः->ne[2] == 1 && निवेशः->ne[3] == 1);
        जइ (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            निवेशः_shape[2] = -1;
        }
        जइ (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            प्रतिज्ञा(निवेशः_shape.size() == 4 && निवेशः_shape[0] == 1 && निवेशः_shape[1] == 1 &&
                   निवेशः_shape[2].is_dynamic() && निवेशः_shape[3].is_static());
            ध्रुव पूर्णाङ्क n_heads_kv = get_n_heads_kv_for_tensor(निवेशः);
            प्रतिज्ञा(n_heads_kv > 0 && निवेशः_shape[3].get_length() % n_heads_kv == 0);
            ध्रुव int64_t combined_dim = निवेशः_shape[3].get_length();  // n_heads_kv * head_size
            ध्रुव int64_t head_size = combined_dim / n_heads_kv;
            निवेशः_shape = {निवेशः_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } अण्णहा जइ (is_kv_idx(निवेशः, op)) {
        // kv update index
        पूर्णाङ्क len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        निवेशः_shape = ov::PartialShape{1, 1, 1, len};

    } अण्णहा जइ (is_inp_s_copy(निवेशः, op) || is_s_copy_leaf(निवेशः)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        निवेशः_shape = m_is_static ? ov::PartialShape{get_shape(निवेशः)} : ov::PartialShape{1, 1, 1, -1};

    } अण्णहा {
        निवेशः_shape = ov::PartialShape{get_shape(निवेशः)};
    }
    जइ (dynamic_dim_index != -1 && m_model_is_splitted) {
        निवेशः_shape[3 - dynamic_dim_index] = -1;
    }
    जइ (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_निवेशः && op->src[1] == निवेशः) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        जइ (m_is_static) {
            ध्रुव int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            निवेशः_shape[2] = seq_active;
            निवेशः_shape[3] = seq_active;
        } अण्णहा {
            निवेशः_shape[2] = -1;
            निवेशः_shape[3] = -1;
        }
    }
    प्रत्यागम निवेशः_shape;
}

तार्किक GgmlOvDecoder::is_s_copy_leaf(ध्रुव ggml_tensor * tensor) ध्रुव {
    जइ (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        प्रत्यागम मिच्छ;
    }
    प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
        ध्रुव ggml_tensor * ग्रन्थिः = m_cgraph->nodes[i];
        जइ (ग्रन्थिः->op != GGML_OP_GET_ROWS || ग्रन्थिः->src[0] == nullptr || ग्रन्थिः->src[1] == nullptr) {
            अनुवर्त;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ध्रुव ggml_tensor * idx = ग्रन्थिः->src[1];
        जाव (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        जइ (idx != tensor) {
            अनुवर्त;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ध्रुव ggml_tensor * data = ग्रन्थिः->src[0];
        जाव (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        जइ (data != nullptr && is_recurrent_cache(data)) {
            प्रत्यागम सच्च;
        }
    }
    प्रत्यागम मिच्छ;
}

शून्य GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_निवेशः = [एतत्](ध्रुव std::string & name, int64_t value, तार्किक force_parameter = मिच्छ) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    जइ (m_compute_params.attention_size != -1) {
        create_1d_निवेशः("attention_size", m_compute_params.attention_size);
    }
    जइ (m_compute_params.attention_size_static != -1) {
        create_1d_निवेशः("attention_size_static", m_compute_params.attention_size_static);
    }
    जइ (m_compute_params.attention_size_swa != -1) {
        create_1d_निवेशः("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    जइ (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_निवेशः("swa_window", m_compute_params.swa_window);
    }
    create_1d_निवेशः("n_seq_active", m_compute_params.n_seq_active);
    create_1d_निवेशः("seq_active_start", m_compute_params.seq_active_start);
    create_1d_निवेशः("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    जइ (m_compute_params.token_len_per_seq != -1) {
        create_1d_निवेशः("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    जइ (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_निवेशः("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/सच्च);
        create_1d_निवेशः("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/सच्च);
    }

    जइ (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_निवेशः("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        जइ (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_निवेशः("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/सच्च);
        }
    }

    प्रति (ध्रुव auto & [ग्रन्थिः_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_निवेशः("rs_slot_begin_" + ग्रन्थिः_name, writeback.slot_begin);
        जइ (!m_is_static && writeback.src_begin >= 0) {
            create_1d_निवेशः("rs_src_begin_" + ग्रन्थिः_name, writeback.src_begin);
        }
    }
}

तार्किक GgmlOvDecoder::ग्रन्थिः_is_used_as_src(ध्रुव पूर्णाङ्क ग्रन्थिः_idx) {
    ggml_tensor * ग्रन्थिः = m_cgraph->nodes[ग्रन्थिः_idx];
    प्रति (पूर्णाङ्क i = ग्रन्थिः_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_ग्रन्थिः = m_cgraph->nodes[i];
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            जइ (other_ग्रन्थिः->src[j] == ग्रन्थिः) {
                प्रत्यागम सच्च;
            }
        }
    }
    प्रत्यागम मिच्छ;
}

शून्य GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ग्रन्थिः = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        जइ (ग्रन्थिः->op == GGML_OP_NONE && ग्रन्थिः_is_used_as_src(i)) {
            std::string ग्रन्थिः_name = get_tensor_ov_name(m_cgraph, ग्रन्थिः);
            जइ (m_model_weights.find(ग्रन्थिः_name) == m_model_weights.end()) {
                m_inputs[ग्रन्थिः_name] = ग्रन्थिः;
                m_model_inputs[ग्रन्थिः_name] = {get_ov_type(ग्रन्थिः),
                                             get_graph_निवेशः_shape(ग्रन्थिः, nullptr, m_ग्रन्थिः_dynamic_dims[ग्रन्थिः])};
            }
            अनुवर्त;
        }
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ग्रन्थिः->src[i];
            जइ (src == nullptr) {
                अनुवर्त;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            जइ (src->flags & GGML_TENSOR_FLAG_निवेशः) {
                src_name = get_tensor_graph_निवेशः_ov_name(एतत्, m_cgraph, src, ग्रन्थिः);
            }
            जइ (m_model_weights.find(src_name) != m_model_weights.end()) {
                अनुवर्त;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            जइ (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                अनुवर्त;
            }

            तार्किक is_intermediate_ग्रन्थिः = मिच्छ;
            प्रति (ध्रुव auto & ग्रन्थिः_info : m_ग्रन्थिः_info_list) {
                जइ (ग्रन्थिः_info.ग्रन्थिः == src) {
                    is_intermediate_ग्रन्थिः = सच्च;
                    भङ्ग;
                }
            }
            जइ (is_intermediate_ग्रन्थिः) {
                अनुवर्त;
            }
            जइ (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                अनुवर्त;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            जइ (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                जइ (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            जाव (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_निवेशः_shape(ग्रन्थिः, src, m_ग्रन्थिः_dynamic_dims[src])};
        }
    }
}

शून्य GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    प्रति (पूर्णाङ्क ग्रन्थिः_n = 0; ग्रन्थिः_n < m_cgraph->n_nodes; ग्रन्थिः_n++) {
        auto * cur_ग्रन्थिः = m_cgraph->nodes[ग्रन्थिः_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        जइ (cur_ग्रन्थिः->op == GGML_OP_NONE || cur_ग्रन्थिः->op == GGML_OP_VIEW || cur_ग्रन्थिः->op == GGML_OP_RESHAPE) {
            अनुवर्त;
        }
        जइ (::is_inplace_op(cur_ग्रन्थिः) && ggml_nbytes(cur_ग्रन्थिः) == 0) {
            अनुवर्त;
        }
        auto cur_ग्रन्थिः_use_गणनम् = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_ग्रन्थिः)];
        जइ (cur_ग्रन्थिः_use_गणनम् == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            जइ (cur_ग्रन्थिः != nullptr && ::is_inplace_op(cur_ग्रन्थिः) && ggml_nbytes(cur_ग्रन्थिः) > 0) {
                cur_ग्रन्थिः = cur_ग्रन्थिः->view_src;
            }
        } अण्णहा {
            पूर्णाङ्क निवेशः_use_गणनम् = 0;
            प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * ग्रन्थिः = m_cgraph->nodes[i];
                प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
                    जइ (ग्रन्थिः->src[j] != रिक्त && ग्रन्थिः->src[j] == cur_ग्रन्थिः) {
                        निवेशः_use_गणनम्++;
                    }
                }
            }
            जइ (निवेशः_use_गणनम् == cur_ग्रन्थिः_use_गणनम्) {
                cur_ग्रन्थिः = nullptr;
            }
        }
        जइ (cur_ग्रन्थिः != nullptr) {
            std::string cur_ग्रन्थिः_name = get_tensor_ov_name(m_cgraph, cur_ग्रन्थिः);
            m_model_outputs[cur_ग्रन्थिः_name] = cur_ग्रन्थिः;
            m_model_output_names.insert(cur_ग्रन्थिः_name);
        }
    }
}

ध्रुव ggml_tensor * GgmlOvDecoder::get_tensor_used_op(ध्रुव ggml_tensor * tensor) ध्रुव {
    जइ (tensor == nullptr) {
        प्रत्यागम nullptr;
    }
    प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
        ध्रुव auto * ग्रन्थिः = m_cgraph->nodes[i];
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            जइ (ग्रन्थिः->src[j] == tensor) {
                प्रत्यागम ग्रन्थिः;
            }
        }
    }
    प्रत्यागम nullptr;
}

ध्रुव ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ध्रुव std::string & name) ध्रुव {
    प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
        ध्रुव auto * ग्रन्थिः = m_cgraph->nodes[i];
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            ध्रुव auto * src = ग्रन्थिः->src[j];
            जइ (src == nullptr) {
                भङ्ग;
            }
            जइ (get_tensor_ov_name(m_cgraph, src) == name) {
                प्रत्यागम src;
            }
        }
    }
    प्रत्यागम nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() ध्रुव {
    std::map<std::string, std::string> kv_param_res_names;
    प्रति (ध्रुव auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    प्रत्यागम kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
स्थिर तार्किक is_mul_mat_id_expert_weight(ध्रुव ggml_tensor * ग्रन्थिः, पूर्णाङ्क src_index) {
    प्रत्यागम ग्रन्थिः->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, तार्किक naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    प्रति (पूर्णाङ्क ग्रन्थिः_i = 0; ग्रन्थिः_i < n_nodes; ग्रन्थिः_i++) {
        auto * ग्रन्थिः = nodes[ग्रन्थिः_i];
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ग्रन्थिः->src[i];
            जइ (src == nullptr) {
                अनुवर्त;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            जइ (is_rope_freqs_weight(base, ग्रन्थिः)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            जइ (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(ग्रन्थिः, i)) {
                जइ (model_weights.find(src_name) == model_weights.end()) {
                    auto weight_ग्रन्थिः = create_weight_ग्रन्थिः(base, naive);
                    weight_ग्रन्थिः->set_friendly_name(src_name);
                    model_weights[src_name] = weight_ग्रन्थिः;
                }
            }
        }
    }
    प्रत्यागम model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
स्थिर std::mutex g_nonov_weight_cache_mutex;
स्थिर std::unordered_map<ध्रुव शून्य *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    प्रति (पूर्णाङ्क ग्रन्थिः_i = 0; ग्रन्थिः_i < cgraph->n_nodes; ग्रन्थिः_i++) {
        auto * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_i];
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ग्रन्थिः->src[i];
            जइ (src == nullptr) {
                अनुवर्त;
            }
            ध्रुव ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            जइ (is_rope_freqs_weight(base, ग्रन्थिः)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            जइ (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    प्रत्यागम names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_ग्रन्थिः(ggml_tensor * tensor, तार्किक naive) {
    ध्रुव तार्किक is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    जइ (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        जइ (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            जइ (weight_extra->weight_ग्रन्थिः) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                प्रत्यागम weight_extra->weight_ग्रन्थिः;
            }
        } अण्णहा जइ (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            जइ (quant_extra->weight_ग्रन्थिः) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                प्रत्यागम quant_extra->weight_ग्रन्थिः;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    जइ (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        auto weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        auto weight_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_ग्रन्थिः->set_friendly_name(tensor->name);
        प्रत्यागम weight_ग्रन्थिः;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ध्रुव तार्किक cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    जइ (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        auto it = g_nonov_weight_cache.find(tensor->data);
        जइ (it != g_nonov_weight_cache.end()) {
            प्रत्यागम it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    स्थिर ध्रुव std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    जइ (weight_types.find(tensor->type) == weight_types.end()) {
        क्षिप std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    जइ (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        जइ (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } अण्णहा {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } अण्णहा {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_ग्रन्थिः->set_friendly_name(tensor->name);
    जइ (!is_ov_buffer) {
        जइ (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            auto [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_ग्रन्थिः);
            प्रत्यागम it->second;
        }
        प्रत्यागम ov_weight.weight_ग्रन्थिः;
    }

    ggml_openvino_extra_base * extra;
    जइ (ov_weight.is_quantized()) {
        extra = नूतन ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_ग्रन्थिः);
    } अण्णहा {
        extra = नूतन ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_ग्रन्थिः);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    प्रत्यागम ov_weight.weight_ग्रन्थिः;
}

शून्य GgmlOvDecoder::dump_cgraph(ध्रुव ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    जइ (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        प्रत्यागम;
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
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ग्रन्थिः = cgraph->nodes[i];

        // Get buffer type name
        ध्रुव वर्ण * buf_name = "none";
        ggml_backend_buffer_t buf = ग्रन्थिः->view_src ? ग्रन्थिः->view_src->buffer : ग्रन्थिः->buffer;
        जइ (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ग्रन्थिः->ne[0] << ", "
             << std::setw(5) << ग्रन्थिः->ne[1] << ", "
             << std::setw(5) << ग्रन्थिः->ne[2] << ", "
             << std::setw(5) << ग्रन्थिः->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(ग्रन्थिः->op) << std::right << " "
             << std::left << std::setw(45) << ग्रन्थिः->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << ग्रन्थिः->nb[0] << ", "
             << std::setw(5) << ग्रन्थिः->nb[1] << ", "
             << std::setw(5) << ग्रन्थिः->nb[2] << ", "
             << std::setw(5) << ग्रन्थिः->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            जइ (auto* src = ग्रन्थिः->src[i]) {
                // Get buffer type name for source
                ध्रुव वर्ण * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                जइ (src_buf) {
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
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * ग्रन्थिः = cgraph->leafs[i];

        // Get buffer type name for leaf
        ध्रुव वर्ण * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = ग्रन्थिः->view_src ? ग्रन्थिः->view_src->buffer : ग्रन्थिः->buffer;
        जइ (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ग्रन्थिः->ne[0] << ", "
             << std::setw(5) << ग्रन्थिः->ne[1] << "] "
             << std::setw(8) << ggml_op_name(ग्रन्थिः->op) << " "
             << std::setw(16) << ggml_get_name(ग्रन्थिः)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

शून्य print_tensor_address_map(ध्रुव ggml_cgraph * cgraph) {
    std::map<शून्य *, std::vector<std::string>> address_map;
    प्रति (पूर्णाङ्क ग्रन्थिः_n = 0; ग्रन्थिः_n < cgraph->n_nodes; ग्रन्थिः_n++) {
        auto * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_n];
        जइ (ग्रन्थिः->data) {
            auto it = address_map.find(ग्रन्थिः->data);
            जइ (it == address_map.end()) {
                address_map[ग्रन्थिः->data] = std::vector<std::string>();
            }
            address_map[ग्रन्थिः->data].push_back(ग्रन्थिः->name);
        }
    }
    प्रति (ध्रुव auto & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        प्रति (ध्रुव auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ध्रुव ggml_tensor * tensor) {
    std::vector<size_t> shape;
    प्रति (पूर्णाङ्क i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    प्रत्यागम shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(ध्रुव ggml_tensor * tensor) {
    std::vector<size_t> stride;
    प्रति (पूर्णाङ्क i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    प्रत्यागम stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(ध्रुव ggml_tensor * tensor) {
    विकल्प (tensor->type) {
    पक्ष GGML_TYPE_F64:
        प्रत्यागम ov::element::f64;
    पक्ष GGML_TYPE_F32:
        प्रत्यागम ov::element::f32;
    पक्ष GGML_TYPE_F16:
        प्रत्यागम ov::element::f16;
    पक्ष GGML_TYPE_BF16:
        प्रत्यागम ov::element::bf16;
    पक्ष GGML_TYPE_I8:
        प्रत्यागम ov::element::i8;
    पक्ष GGML_TYPE_I16:
        प्रत्यागम ov::element::i16;
    पक्ष GGML_TYPE_I32:
        प्रत्यागम ov::element::i32;
    पक्ष GGML_TYPE_I64:
        प्रत्यागम ov::element::i64;
    स्वभाव:
        प्रत्यागम ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_निवेशः_shape(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name) ध्रुव {
    प्रत्यागम ov::PartialShape(get_shape(m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_निवेशः_stride(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name) ध्रुव {
    प्रत्यागम get_stride(m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_निवेशः_size(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        प्रत्यागम it->second.size();
    }
    प्रत्यागम 0;
}

size_t GgmlOvDecoder::get_view_निवेशः_offset(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_index) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        जइ (view_index < it->second.size()) {
            प्रत्यागम it->second[view_index].second->view_offs;
        }
    }
    प्रत्यागम 0;
}

size_t GgmlOvDecoder::get_view_निवेशः_src_offset(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_index) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        जइ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            जइ (view_tensor && view_tensor->src[0]) {
                प्रत्यागम view_tensor->src[0]->view_offs;
            }
        }
    }
    प्रत्यागम 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_निवेशः_stride(पूर्णाङ्क ग्रन्थिः_idx,
                                                         ध्रुव std::string & name,
                                                         size_t view_index) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        जइ (view_index < it->second.size()) {
            प्रत्यागम get_stride(it->second[view_index].second);
        }
    }
    प्रत्यागम {};
}

std::vector<size_t> GgmlOvDecoder::get_view_निवेशः_src_stride(पूर्णाङ्क ग्रन्थिः_idx,
                                                             ध्रुव std::string & name,
                                                             size_t view_index) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        जइ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            जइ (view_tensor && view_tensor->src[0]) {
                प्रत्यागम get_stride(view_tensor->src[0]);
            }
        }
    }
    प्रत्यागम {};
}

ov::Shape GgmlOvDecoder::get_view_निवेशः_ggml_shape(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_index) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        जइ (view_index < it->second.size()) {
            प्रत्यागम get_shape(it->second[view_index].second);
        }
    }
    प्रत्यागम {};
}

ov::Shape GgmlOvDecoder::get_view_निवेशः_src_ggml_shape(पूर्णाङ्क ग्रन्थिः_idx,
                                                       ध्रुव std::string & name,
                                                       size_t view_index) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        जइ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            जइ (view_tensor && view_tensor->src[0]) {
                प्रत्यागम get_shape(view_tensor->src[0]);
            }
        }
    }
    प्रत्यागम {};
}

ov::PartialShape GgmlOvDecoder::get_view_निवेशः_ov_shape(पूर्णाङ्क ग्रन्थिः_idx,
                                                        ध्रुव std::string & name,
                                                        size_t view_index) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        जइ (view_index < it->second.size()) {
            auto * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            auto dynamic_it = m_ग्रन्थिः_dynamic_dims.find(tensor);
            जइ (dynamic_it != m_ग्रन्थिः_dynamic_dims.end() && dynamic_it->second != -1) {
                पूर्णाङ्क dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            प्रत्यागम shape;
        }
    }
    प्रत्यागम {};
}

ov::PartialShape GgmlOvDecoder::get_view_निवेशः_src_ov_shape(पूर्णाङ्क ग्रन्थिः_idx,
                                                            ध्रुव std::string & name,
                                                            size_t view_index) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        जइ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            जइ (view_tensor && view_tensor->src[0]) {
                auto * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                auto dynamic_it = m_ग्रन्थिः_dynamic_dims.find(src_tensor);
                जइ (dynamic_it != m_ग्रन्थिः_dynamic_dims.end() && dynamic_it->second != -1) {
                    पूर्णाङ्क dynamic_dim_index = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
                }

                प्रत्यागम shape;
            }
        }
    }
    प्रत्यागम {};
}

std::string GgmlOvDecoder::get_view_निवेशः_name(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_index) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        जइ (view_index < it->second.size()) {
            प्रत्यागम it->second[view_index].first;
        }
    }
    प्रत्यागम "";
}

std::string GgmlOvDecoder::get_view_निवेशः_src_name(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_index) ध्रुव {
    auto it = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.find(name);
    जइ (it != m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_views.end()) {
        जइ (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            जइ (view_tensor && view_tensor->src[0]) {
                प्रत्यागम get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    प्रत्यागम "";
}

ov::element::Type GgmlOvDecoder::get_निवेशः_type(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name) ध्रुव {
    प्रत्यागम get_ov_type(m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs.at(name));
}

size_t GgmlOvDecoder::get_निवेशः_size() ध्रुव {
    प्रत्यागम m_model_inputs.size();
}

size_t GgmlOvDecoder::get_निवेशः_size(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    प्रत्यागम m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_निवेशः_names(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    प्रत्यागम m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    auto * ggml_tensor = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः;
    प्रत्यागम ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ध्रुव पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    प्रत्यागम get_ov_type(m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    auto * ggml_tensor = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः;
    प्रत्यागम get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    auto * ग्रन्थिः = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः;
    जइ (ग्रन्थिः->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        प्रति (पूर्णाङ्क i = ग्रन्थिः_idx + 1; i < m_cgraph->n_nodes; i++) {
            auto * consumer = m_cgraph->nodes[i];
            जइ (consumer->op != GGML_OP_VIEW || consumer->src[0] != ग्रन्थिः) {
                अनुवर्त;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            auto & name = consumer->view_offs == 0 ? attn_name : state_name;
            जइ (!name.empty()) {
                प्रत्यागम {m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        जइ (!attn_name.empty() && !state_name.empty()) {
            प्रत्यागम {attn_name, state_name};
        }
    }
    प्रत्यागम {m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    auto * ग्रन्थिः = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः;
    जइ (!::is_inplace_op(ग्रन्थिः) || ग्रन्थिः->view_src == nullptr || ggml_nbytes(ग्रन्थिः) == 0) {
        प्रत्यागम "";
    }
    ध्रुव पूर्णाङ्क op_case = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_op_case;
    जइ (ग्रन्थिः->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        प्रत्यागम "";
    }
    प्रत्यागम get_tensor_ov_name(m_cgraph, ग्रन्थिः->view_src);
}

तार्किक GgmlOvDecoder::is_view_like_alias_of(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & view_src_name) ध्रुव {
    auto * ग्रन्थिः = m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः;
    जइ (ग्रन्थिः->view_src == nullptr || get_tensor_ov_name(m_cgraph, ग्रन्थिः->view_src) != view_src_name) {
        प्रत्यागम मिच्छ;
    }
    प्रत्यागम ग्रन्थिः->op == GGML_OP_RESHAPE || ग्रन्थिः->op == GGML_OP_VIEW;
}

ध्रुव std::string & GgmlOvDecoder::get_op_name() ध्रुव {
    स्थिर ध्रुव std::string unknown_name = "UNKNOWN_OP_NAME";
    प्रत्यागम unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    auto it = m_ग्रन्थिः_dynamic_dims.find(m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः);
    जइ (it == m_ग्रन्थिः_dynamic_dims.end()) {
        प्रत्यागम -1;
    }
    प्रत्यागम it->second;
}

ध्रुव std::string & GgmlOvDecoder::get_op_name(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    प्रत्यागम m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_name;
}

int32_t * GgmlOvDecoder::get_निवेशः_op_params(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name) ध्रुव {
    प्रत्यागम m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    प्रत्यागम m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः->op_params;
}

size_t GgmlOvDecoder::get_output_op_offset(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    प्रत्यागम m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः->view_offs;
}

शून्य GgmlOvDecoder::visit_subgraph(std::function<शून्य(std::shared_ptr<GgmlDecoder>, पूर्णाङ्क ग्रन्थिः_idx)> ग्रन्थिः_visitor) ध्रुव {
    प्रति (पूर्णाङ्क ग्रन्थिः_idx = 0; ग्रन्थिः_idx < m_cgraph->n_nodes; ग्रन्थिः_idx++) {
        जइ (m_cgraph->nodes[ग्रन्थिः_idx]->op == GGML_OP_NONE) {
            अनुवर्त;
        }
        ग्रन्थिः_visitor(std::make_shared<GgmlOvDecoder>(*एतत्), ग्रन्थिः_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ध्रुव ggml_tensor * ग्रन्थिः) {
    विकल्प (ग्रन्थिः->op) {
    पक्ष GGML_OP_UNARY:
        प्रत्यागम std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(ग्रन्थिः));
    पक्ष GGML_OP_GLU:
        प्रत्यागम std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(ग्रन्थिः));
    स्वभाव:
        प्रत्यागम std::string("GGML_OP_") + ggml_op_name(ग्रन्थिः->op);
    }
}

ध्रुव std::string & GgmlOvDecoder::get_op_type(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव {
    प्रत्यागम m_ग्रन्थिः_info_list[ग्रन्थिः_idx].ग्रन्थिः_op_type;
}

ध्रुव std::string & GgmlOvDecoder::get_op_type() ध्रुव {
    स्थिर ध्रुव std::string unknown_op = "UNKNOWN_GGML_OP";
    प्रत्यागम unknown_op;
}

शून्य GgmlOvDecoder::compute_ग्रन्थिः_dynamic_dims() {
    auto visit_ग्रन्थिः = [&](auto && self, ggml_tensor * ग्रन्थिः) -> शून्य {
        जइ (!ग्रन्थिः) {
            प्रत्यागम;
        }

        जइ (ग्रन्थिः->op == GGML_OP_CPY) {
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
        }

        जइ (m_ग्रन्थिः_dynamic_dims.count(ग्रन्थिः)) {
            प्रत्यागम;
        }
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = ग्रन्थिः->src[i];
            जइ (src == nullptr) {
                अनुवर्त;
            }
            संरचना ggml_tensor * मूलम्_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            जइ (मूलम्_src) {
                जइ (is_inp_tok(मूलम्_src, ग्रन्थिः) || is_inp_pos(मूलम्_src, ग्रन्थिः) || is_output_idx(मूलम्_src, ग्रन्थिः)) {
                    m_ग्रन्थिः_dynamic_dims[मूलम्_src] = 0;
                    m_ग्रन्थिः_dynamic_dims[src] = m_ग्रन्थिः_dynamic_dims[मूलम्_src];
                    अनुवर्त;
                }
                self(self, मूलम्_src);
                m_ग्रन्थिः_dynamic_dims[src] = m_ग्रन्थिः_dynamic_dims[मूलम्_src];
            } अण्णहा {
                जइ (is_inp_tok(src, ग्रन्थिः) || is_inp_pos(src, ग्रन्थिः) || is_output_idx(src, ग्रन्थिः)) {
                    m_ग्रन्थिः_dynamic_dims[src] = 0;
                    अनुवर्त;
                }
                जइ (is_inp_scale_rows(src, ग्रन्थिः)) {
                    m_ग्रन्थिः_dynamic_dims[src] = 1;
                    अनुवर्त;
                }
                जइ (ग्रन्थिः->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_ग्रन्थिः_dynamic_dims[src] = 1;
                    अनुवर्त;
                }
                self(self, src);
            }
        }
        विकल्प (ग्रन्थिः->op) {
        पक्ष GGML_OP_NONE:
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            भङ्ग;
        पक्ष GGML_OP_GET_ROWS:
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[1]] != -1) {
                auto dynamic_dim_idx = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[1]];
                जइ (dynamic_dim_idx == 0) {
                    m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = 1;
                } अण्णहा {
                    auto dynamic_dim_stride = ग्रन्थिः->src[1]->nb[dynamic_dim_idx] / ggml_type_size(ग्रन्थिः->src[1]->type) *
                                              ggml_type_size(ग्रन्थिः->src[0]->type);
                    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                        जइ (dynamic_dim_stride == ग्रन्थिः->src[0]->nb[i]) {
                            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = i;
                            भङ्ग;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            भङ्ग;
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_MUL_MAT:
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]] != -1) {
                m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]];
            }
            जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[1]] != -1) {
                m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[1]];
            }
            भङ्ग;
        पक्ष GGML_OP_PERMUTE:
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]] != -1) {
                auto dynamic_dim_idx = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                    जइ (ग्रन्थिः->op_params[i] == dynamic_dim_idx) {
                        m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = i;
                        भङ्ग;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            भङ्ग;
        पक्ष GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]] != -1) {
                जइ (ग्रन्थिः->src[0]->op == GGML_OP_NONE) {
                    m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]];
                    भङ्ग;
                }
                auto dynamic_dim_idx = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]];
                auto dynamic_dim_मानम् = ग्रन्थिः->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    ग्रन्थिः->src[0]->nb[dynamic_dim_idx] / ggml_type_size(ग्रन्थिः->src[0]->type) * ggml_type_size(ग्रन्थिः->type);
                प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                    जइ (ग्रन्थिः->nb[i] == dynamic_dim_stride) {
                        m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = i;
                        भङ्ग;
                    }
                }
                जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] != -1 && dynamic_dim_मानम् != ग्रन्थिः->ne[m_ग्रन्थिः_dynamic_dims[ग्रन्थिः]]) {
                    m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    जइ (ggml_nelements(ग्रन्थिः) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      ग्रन्थिः->name, ग्रन्थिः->src[0]->name);
                    }
                }
            }
            भङ्ग;
        }
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_RESHAPE: {
            जइ (is_same_shape(ग्रन्थिः->src[0], ग्रन्थिः)) {
                m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]];
                भङ्ग;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]] != -1) {
                auto dynamic_dim_idx = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]];
                auto dynamic_dim_stride = ग्रन्थिः->src[0]->nb[dynamic_dim_idx];
                प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                    जइ (ग्रन्थिः->nb[i] == dynamic_dim_stride && ग्रन्थिः->ne[i] == ग्रन्थिः->src[0]->ne[dynamic_dim_idx]) {
                        m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = i;
                        भङ्ग;
                    }
                }
                जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", ग्रन्थिः->name);
                }
            }
            भङ्ग;
        }
        पक्ष GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr पूर्णाङ्क q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]] != -1) {
                auto q_dynamic_dim = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]];
                m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = q_to_out[q_dynamic_dim];
            }
            भङ्ग;
        }
        पक्ष GGML_OP_CONT:
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]] != -1) {
                auto dynamic_dim_idx = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]];
                जइ (ggml_are_same_shape(ग्रन्थिः, ग्रन्थिः->src[0])) {
                    m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = dynamic_dim_idx;
                } अण्णहा {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(ग्रन्थिः->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (ग्रन्थिः->src[0]->ne[0] / ggml_blck_size(ग्रन्थिः->src[0]->type));
                    प्रति (पूर्णाङ्क i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * ग्रन्थिः->src[0]->ne[i - 1];
                    }

                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(ग्रन्थिः->src[0]->type) *
                                              ggml_type_size(ग्रन्थिः->type);
                    पूर्णाङ्क matched_dim_गणनम् = 0;
                    पूर्णाङ्क first_matched_dim = -1;
                    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                        जइ (ग्रन्थिः->nb[i] == dynamic_dim_stride && ग्रन्थिः->ne[i] == ग्रन्थिः->src[0]->ne[dynamic_dim_idx]) {
                            जइ (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = i;
                            matched_dim_गणनम्++;
                        }
                    }
                    जइ (matched_dim_गणनम् > 1 && ग्रन्थिः->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = first_matched_dim;
                    } अण्णहा जइ (matched_dim_गणनम् != 1) {
                        m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      ग्रन्थिः->name, ग्रन्थिः->src[0]->name);
                    }
                }
            }
            भङ्ग;
        पक्ष GGML_OP_CONCAT:
            प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                जइ (ग्रन्थिः->src[0]->ne[i] != ग्रन्थिः->ne[i]) {
                    m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = i;
                    भङ्ग;
                }
            }
            भङ्ग;
        पक्ष GGML_OP_SSM_CONV:
        पक्ष GGML_OP_GATED_DELTA_NET:
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = 1;
            भङ्ग;
        पक्ष GGML_OP_RMS_NORM:
        पक्ष GGML_OP_L2_NORM:
        पक्ष GGML_OP_NORM:
        पक्ष GGML_OP_ADD:
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_GLU:
        पक्ष GGML_OP_ROPE:
        पक्ष GGML_OP_SCALE:
        पक्ष GGML_OP_SOFT_MAX:
        पक्ष GGML_OP_ARGSORT:
        पक्ष GGML_OP_ADD_ID:
        पक्ष GGML_OP_UNARY:
        पक्ष GGML_OP_CUMSUM:
        पक्ष GGML_OP_FILL:
        पक्ष GGML_OP_SET:
        पक्ष GGML_OP_DIAG:
        पक्ष GGML_OP_TRI:
        पक्ष GGML_OP_REPEAT:
        पक्ष GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        पक्ष GGML_OP_DIV:
        पक्ष GGML_OP_CLAMP:
        पक्ष GGML_OP_PAD:
        पक्ष GGML_OP_UPSCALE:
        पक्ष GGML_OP_SIN:
        पक्ष GGML_OP_COS:
        पक्ष GGML_OP_LOG:
        पक्ष GGML_OP_ROLL:
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]];
            भङ्ग;
        पक्ष GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] =
                (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]] == 0) ? -1 : m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[0]];
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT_ID:
        पक्ष GGML_OP_SOLVE_TRI:
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[1]];
            भङ्ग;
        पक्ष GGML_OP_CPY:
        पक्ष GGML_OP_SET_ROWS:
        पक्ष GGML_OP_SUM:
        पक्ष GGML_OP_MEAN:
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            भङ्ग;
        पक्ष GGML_OP_IM2COL: {
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[1]] != -1) {
                ध्रुव तार्किक is_2D = ग्रन्थिः->op_params[6] == 1;
                ध्रुव पूर्णाङ्क src_dyn = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[1]];
                जइ (is_2D) {
                    जइ (src_dyn == 0) {
                        m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = 1;  // IW -> OW
                    } अण्णहा जइ (src_dyn == 1) {
                        m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = 2;  // IH -> OH
                    } अण्णहा जइ (src_dyn == 3) {
                        m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = 3;  // N  -> N
                    }
                } अण्णहा {
                    जइ (src_dyn == 0) {
                        m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = 1;  // IW -> OW
                    } अण्णहा जइ (src_dyn == 2) {
                        m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] != -1) {
                    OPENVINO_ASSERT(ग्रन्थिः->src[1]->ne[src_dyn] == ग्रन्थिः->ne[m_ग्रन्थिः_dynamic_dims[ग्रन्थिः]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(ग्रन्थिः->name) +
                                        " and its src[1]: " + std::string(ग्रन्थिः->src[1]->name));
                }
            }
            भङ्ग;
        }
        पक्ष GGML_OP_IM2COL_3D: {
            m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = -1;
            जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[1]] != -1) {
                ध्रुव पूर्णाङ्क src_dyn = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः->src[1]];
                जइ (src_dyn == 0) {
                    m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = 1;  // IW -> OW
                } अण्णहा जइ (src_dyn == 1) {
                    m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = 2;  // IH -> OH
                } अण्णहा जइ (src_dyn == 3) {
                    m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] = 3;  // N  -> N
                }
                जइ (m_ग्रन्थिः_dynamic_dims[ग्रन्थिः] != -1) {
                    OPENVINO_ASSERT(ग्रन्थिः->src[1]->ne[src_dyn] == ग्रन्थिः->ne[m_ग्रन्थिः_dynamic_dims[ग्रन्थिः]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(ग्रन्थिः->name) +
                                        " and its src[1]: " + std::string(ग्रन्थिः->src[1]->name));
                }
            }
            भङ्ग;
        }
        स्वभाव:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(ग्रन्थिः->op), ग्रन्थिः->name);
            भङ्ग;
        }
    };

    प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ग्रन्थिः = m_cgraph->nodes[i];
        visit_ग्रन्थिः(visit_ग्रन्थिः, ग्रन्थिः);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    जइ (0) {
        प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * ग्रन्थिः = m_cgraph->nodes[i];
            पूर्णाङ्क dynamic_dim = m_ग्रन्थिः_dynamic_dims[ग्रन्थिः];
            std::cout << "[" << i << "] " << "node_name: " << ग्रन्थिः->name << " op: " << ggml_op_name(ग्रन्थिः->op)
                      << " shape: [";
            प्रति (पूर्णाङ्क j = 0; j < 4; j++) {
                जइ (j == dynamic_dim) {
                    std::cout << "*";
                } अण्णहा {
                    std::cout << ग्रन्थिः->ne[j];
                }
                जइ (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = ग्रन्थिः->src[j];
                जइ (src == nullptr) {
                    अनुवर्त;
                }
                पूर्णाङ्क src_dynamic_dim = m_ग्रन्थिः_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                प्रति (पूर्णाङ्क k = 0; k < 4; k++) {
                    जइ (k == src_dynamic_dim) {
                        std::cout << "*";
                    } अण्णहा {
                        std::cout << src->ne[k];
                    }
                    जइ (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
