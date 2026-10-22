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
    m_naive(असत्य),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    स्थिर तार्किक printed_address_map = असत्य;
    यदि (!printed_address_map) {
        यदि (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = सत्य;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_निवेश_output();
    compute_गाँठ_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    प्रति (पूर्णाङ्क गाँठ_n = 0; गाँठ_n < cgraph->n_nodes; गाँठ_n++) {
        m_गाँठ_info_list[गाँठ_n].गाँठ_op_case = compute_op_case(m_गाँठ_info_list[गाँठ_n].गाँठ);
        m_गाँठ_info_list[गाँठ_n].गाँठ_op_type = compute_op_type(m_गाँठ_info_list[गाँठ_n].गाँठ);
    }

    add_extra_inputs();
}

शून्य GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_गाँठ_info_list.clear();
    set_निवेश_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
तार्किक GgmlOvDecoder::is_bound_to(ध्रुव ggml_cgraph * cgraph) ध्रुव {
    प्रत्यागम m_cgraph == cgraph && cgraph->n_nodes > 0 && m_गाँठ_info_list.size() == (माप_प्रकार) cgraph->n_nodes &&
           m_गाँठ_info_list.front().गाँठ == cgraph->nodes[0] &&
           m_गाँठ_info_list.back().गाँठ == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = सत्य;
    set_निवेश_output();
    compute_model_inputs();
    compute_model_outputs();
    प्रति (पूर्णाङ्क गाँठ_n = 0; गाँठ_n < cgraph->n_nodes; गाँठ_n++) {
        m_गाँठ_info_list[गाँठ_n].गाँठ_op_case = compute_op_case(m_गाँठ_info_list[गाँठ_n].गाँठ);
        m_गाँठ_info_list[गाँठ_n].गाँठ_op_type = compute_op_type(m_गाँठ_info_list[गाँठ_n].गाँठ);
    }
}

नामाकाश {
तार्किक is_inplace_op(ध्रुव ggml_tensor * गाँठ) {
    प्रत्यागम गाँठ->op == GGML_OP_SET_ROWS || गाँठ->op == GGML_OP_CPY || (गाँठ->op == GGML_OP_SCALE && गाँठ->view_src);
}

तार्किक is_same_shape(ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        यदि (a->ne[i] != b->ne[i]) {
            प्रत्यागम असत्य;
        }
    }
    प्रत्यागम सत्य;
}

तार्किक is_conv_states_all_tensor(ध्रुव ggml_tensor * tensor) {
    प्रत्यागम tensor != nullptr && म_भेदन(tensor->name, "conv_states_all", म_माप("conv_states_all")) == 0;
}

तार्किक is_full_single_slot_writeback(ध्रुव ggml_tensor * गाँठ) {
    प्रत्यागम गाँठ->view_src != nullptr && गाँठ->view_src->ne[1] == 1 && गाँठ->src[1] != nullptr &&
           गाँठ->src[1]->op == GGML_OP_VIEW && गाँठ->src[1]->view_src == गाँठ->view_src &&
           गाँठ->src[1]->view_offs == 0 && ggml_nbytes(गाँठ->src[1]) == ggml_nbytes(गाँठ->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
तार्किक is_moe_expert_sum_add(ध्रुव ggml_tensor * गाँठ) {
    std::vector<ध्रुव ggml_tensor *> leaves;
    ध्रुव ggml_tensor * cur = गाँठ;
    यावत् (cur->op == GGML_OP_ADD) {
        यदि (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            प्रत्यागम असत्य;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ध्रुव ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    प्रति (ध्रुव ggml_tensor * leaf : leaves) {
        यदि (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            प्रत्यागम असत्य;
        }
        ध्रुव ggml_tensor * leaf_base = leaf->src[0];
        यदि (base == nullptr) {
            base = leaf_base;
        } अन्यथा यदि (leaf_base != base) {
            प्रत्यागम असत्य;
        }
        यदि (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            प्रत्यागम असत्य;
        }
        यदि (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            प्रत्यागम असत्य;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        यदि (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<माप_प्रकार>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ध्रुव ggml_cgraph * cgraph, ध्रुव ggml_tensor * tensor) {
    यदि (tensor == nullptr) {
        प्रत्यागम "";
    }
    यदि ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ध्रुव स्वतः * गाँठ = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        यदि (गाँठ != cgraph->nodes + cgraph->n_nodes) {
            प्रत्यागम std::string(tensor->name) + "#n" + std::to_string(गाँठ - cgraph->nodes);
        }
        ध्रुव स्वतः * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        यदि (leaf != cgraph->leafs + cgraph->n_leafs) {
            प्रत्यागम std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    प्रत्यागम tensor->name;
}

स्थिर std::string get_tensor_ov_name(ध्रुव ggml_cgraph * cgraph, ध्रुव ggml_tensor * tensor) {
    प्रत्यागम GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

स्थिर std::string get_tensor_graph_निवेश_ov_name(ध्रुव GgmlOvDecoder * decoder,
                                                  ध्रुव ggml_cgraph * cgraph,
                                                  ध्रुव ggml_tensor * tensor,
                                                  ध्रुव ggml_tensor * op) {
    यदि (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        प्रत्यागम "inp_pos";
    }
    यदि (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        प्रत्यागम "embd";
    }
    यदि (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ध्रुव तार्किक is_swa = decoder->is_swa_mask(tensor);
        यदि (decoder->is_stateful()) {
            प्रत्यागम is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        यदि (is_swa) {
            प्रत्यागम get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    प्रत्यागम get_tensor_ov_name(cgraph, tensor);
}

शून्य GgmlOvDecoder::set_निवेश_output() {
    प्रति (पूर्णाङ्क गाँठ_n = 0; गाँठ_n < m_cgraph->n_nodes; गाँठ_n++) {
        स्वतः * गाँठ = m_cgraph->nodes[गाँठ_n];

        गाँठ_Info current_गाँठ_info;
        स्वतः गाँठ_name = get_tensor_ov_name(m_cgraph, गाँठ);

        current_गाँठ_info.गाँठ = गाँठ;
        current_गाँठ_info.गाँठ_name = गाँठ_name;
        current_गाँठ_info.गाँठ_op_case = 0;
        current_गाँठ_info.data_addr = गाँठ->data;

        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            स्वतः * src = गाँठ->src[i];
            यदि (src == nullptr) {
                अनुवर्त;
            }
            स्वतः src_name = get_tensor_ov_name(m_cgraph, src);
            यदि (src->flags & GGML_TENSOR_FLAG_निवेश) {
                src_name = get_tensor_graph_निवेश_ov_name(एतत्, m_cgraph, src, गाँठ);
            }
            current_गाँठ_info.गाँठ_inputs[src_name] = src;
            current_गाँठ_info.गाँठ_inputs_names.push_back(src_name);

            यदि (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_गाँठ_info.गाँठ_inputs_views[src_name])> view_chain;
                स्वतः * current = src;

                यावत् (current != nullptr) {
                    स्वतः current_name = get_tensor_ov_name(m_cgraph, current);
                    यदि (current->flags & GGML_TENSOR_FLAG_निवेश) {
                        current_name = get_tensor_graph_निवेश_ov_name(एतत्, m_cgraph, current, गाँठ);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    यदि (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } अन्यथा {
                        भङ्ग;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_गाँठ_info.गाँठ_inputs_views[src_name] = view_chain;
            }
        }

        m_गाँठ_info_list.push_back(current_गाँठ_info);
    }
}

पूर्णाङ्क GgmlOvDecoder::compute_op_case(ध्रुव ggml_tensor * गाँठ) ध्रुव {
    पूर्णाङ्क op_case = 0;
    विकल्प (गाँठ->op) {
    पक्ष GGML_OP_RESHAPE: {
        यदि (m_naive) {
            भङ्ग;
        }
        स्वतः name = std::string(गाँठ->name);
        स्वतः * src = गाँठ->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        तार्किक recurrent_sequence = असत्य;
        प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ध्रुव स्वतः * consumer = m_cgraph->nodes[i];
            यदि (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == गाँठ) {
                प्रत्यागम 1;
            } अन्यथा यदि (consumer->op == GGML_OP_SSM_CONV) {
                ध्रुव स्वतः * concat = consumer->src[0];
                यदि (concat->op == GGML_OP_CONCAT) {
                    ध्रुव स्वतः * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == गाँठ;
                }
            } अन्यथा यदि (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                ध्रुव स्वतः * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == गाँठ;
            }
        }
        यदि (recurrent_sequence && गाँठ->ne[0] == src->ne[0] && गाँठ->ne[3] == 1) {
            प्रत्यागम 6;
        }
        यदि (गाँठ->ne[0] == src->ne[0] && गाँठ->ne[2] == 1 && गाँठ->ne[3] == 1) {
            प्रत्यागम 5;
        }
        यदि (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == गाँठ->ne[0] && src->src[0]->ne[1] == गाँठ->ne[1]) {
            op_case = 4;
        } अन्यथा यदि (गाँठ->ne[0] * गाँठ->ne[1] == src->ne[0]) {
            op_case = 1;
        } अन्यथा यदि (src->ne[0] * src->ne[1] == गाँठ->ne[0]) {
            op_case = 2;
            यदि (src->ne[2] * src->ne[3] == गाँठ->ne[1]) {
                op_case = 5;
            }
        } अन्यथा यदि (गाँठ->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == गाँठ->ne[1]) {
            op_case = 3;
        } अन्यथा यदि (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } अन्यथा यदि (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } अन्यथा यदि (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        यदि (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            ध्रुव स्वतः * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            तार्किक recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ध्रुव स्वतः * consumer = m_cgraph->nodes[i];
                यदि (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    प्रति (पूर्णाङ्क j : {3, 4}) {
                        ध्रुव स्वतः * gate = consumer->src[j];
                        यदि (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == गाँठ;
                    }
                } अन्यथा यदि (consumer->op == GGML_OP_MUL) {
                    प्रति (पूर्णाङ्क j = 0; j < 2; ++j) {
                        ध्रुव स्वतः * gate = consumer->src[j];
                        ध्रुव स्वतः * norm = consumer->src[1 - j];
                        यदि (gate->op != GGML_OP_UNARY || gate->src[0] != गाँठ) {
                            अनुवर्त;
                        }
                        यदि (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            यदि (recurrent) {
                op_case = 9;
            }
        }
        भङ्ग;
    }
    पक्ष GGML_OP_PERMUTE: {
        यदि (गाँठ->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } अन्यथा यदि (गाँठ->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(गाँठ->view_src->name);
            पूर्णाङ्क layer = extract_layer_from_name(src_name).value();
            यदि (ggml_is_contiguous(गाँठ->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                यदि (!is_swa_layer(layer)) {
                    op_case = 3;
                } अन्यथा {
                    op_case = 4;
                }
            } अन्यथा {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                यदि (!is_swa_layer(layer)) {
                    op_case = 5;
                } अन्यथा {
                    op_case = 6;
                }
            }
        } अन्यथा {
            // rope'ed query tensor
            op_case = 2;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_MUL_MAT: {
        यदि (गाँठ->src[1]->op == GGML_OP_SOFT_MAX) {
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
        यदि (गाँठ->src[0]->op == GGML_OP_RESHAPE && गाँठ->src[0]->src[0] != nullptr &&
            is_recurrent_cache(गाँठ->src[0]->src[0])) {
            ध्रुव तार्किक single_slot = गाँठ->src[0]->src[0]->ne[1] == 1;
            यदि (गाँठ->src[1]->op == GGML_OP_VIEW) {
                op_case = (गाँठ->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } अन्यथा यदि (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        भङ्ग;
    }
    पक्ष GGML_OP_ROPE: {
        ध्रुव पूर्णाङ्क mode = गाँठ->op_params[2];
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
        यदि (!m_model_params.has_rs_rollback && गाँठ->src[0] != nullptr &&
            गाँठ->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            भङ्ग;
        }
        यदि (m_is_static && गाँठ->src[0] != nullptr &&
            (गाँठ->src[0]->op == GGML_OP_GATED_DELTA_NET || गाँठ->src[0]->op == GGML_OP_CONCAT)) {
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
        यदि (गाँठ->src[0]->op == GGML_OP_VIEW) {
            स्वतः * src = गाँठ->src[0];
            यदि (ggml_nelements(गाँठ) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            यदि (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            स्वतः * src = गाँठ->src[0];
            यदि (ggml_nelements(गाँठ) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                यदि (गाँठ->ne[0] == src->ne[0] && गाँठ->ne[1] == src->ne[2] && गाँठ->ne[2] == 1 &&
                    गाँठ->nb[0] == src->nb[0] && गाँठ->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    भङ्ग;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                पूर्णाङ्क diff_गिनती = 0;
                प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                    यदि (गाँठ->ne[i] != src->ne[i]) {
                        diff_गिनती++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    यदि (गाँठ->ne[i] > src->ne[i]) {
                        प्रत्यागम 0;
                    }
                }
                यदि (diff_गिनती >= 1) {
                    op_case = 0;
                }
            }
        }
        भङ्ग;
    }
    पक्ष GGML_OP_RMS_NORM: {
        यदि (गाँठ->src[0]->op == GGML_OP_VIEW) {
            यदि (is_same_shape(गाँठ->src[0]->src[0], गाँठ->src[0])) {
                op_case = 1;
            } अन्यथा यदि (!m_model_params.has_rs_rollback &&
                       गाँठ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } अन्यथा यदि (गाँठ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        भङ्ग;
    }
    पक्ष GGML_OP_POOL_2D: {
        ध्रुव ggml_op_pool pool_mode = static_cast<ggml_op_pool>(गाँठ->op_params[0]);
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
        ध्रुव int32_t mode_flags = गाँठ->op_params[0];
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
        यदि (गाँठ->src[0]->op == GGML_OP_VIEW) {
            यदि (गाँठ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                यदि (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(गाँठ) ? 7 : 10;
                } अन्यथा {
                    op_case = 1;
                }
            } अन्यथा यदि (GgmlOvDecoder::is_conv_state_writeback(गाँठ)) {
                op_case = is_full_single_slot_writeback(गाँठ) ? 8 : 2;
                भङ्ग;
            } अन्यथा यदि (is_conv_states_all_tensor(गाँठ->view_src) && गाँठ->src[1] != nullptr &&
                       गाँठ->src[1]->op == GGML_OP_VIEW && गाँठ->src[1]->view_src == गाँठ->view_src) {
                op_case = 4;
                भङ्ग;
            } अन्यथा यदि (गाँठ->src[0]->src[0]->op == GGML_OP_GET_ROWS && गाँठ->src[1] != nullptr &&
                       गाँठ->src[1]->op == GGML_OP_VIEW && गाँठ->src[1]->view_src != nullptr &&
                       is_recurrent_cache(गाँठ->src[1]->view_src) && गाँठ->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } अन्यथा यदि (गाँठ->src[0]->op == GGML_OP_GET_ROWS && गाँठ->src[1] != nullptr &&
                   गाँठ->src[1]->op == GGML_OP_VIEW && गाँठ->src[1]->view_src != nullptr &&
                   is_recurrent_cache(गाँठ->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = गाँठ->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } अन्यथा यदि (गाँठ->src[1] != nullptr && गाँठ->src[1]->op == GGML_OP_VIEW && गाँठ->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ध्रुव ggml_tensor * kv_buf = गाँठ->src[1]->view_src;
            यदि (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
                    ध्रुव ggml_tensor * n = m_cgraph->nodes[i];
                    यदि (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        अनुवर्त;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    यदि ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        यदि (n->src[3] != nullptr) {
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
        यदि (is_moe_expert_sum_add(गाँठ)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_SCALE: {
        यदि (गाँठ->view_src && गाँठ->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = गाँठ->view_src->ne[1] == 1 ? 2 : 1;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_L2_NORM: {
        यदि (std::string(गाँठ->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        भङ्ग;
    }
    पक्ष GGML_OP_FLASH_ATTN_EXT: {
        यदि (गाँठ->src[1] != nullptr && गाँठ->src[1]->op == GGML_OP_VIEW && गाँठ->src[1]->view_src != nullptr) {
            ध्रुव ggml_tensor * kv_buf = गाँठ->src[1]->view_src;
            यदि (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (गाँठ->src[3] != nullptr) ? 1 : 2;
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
    माप_प्रकार pos1 = name.find("_l");
    यदि (pos1 == std::string::npos) {
        प्रत्यागम std::nullopt;
    }
    pos1 += 2;
    माप_प्रकार pos2 = name.find(' ', pos1);
    यदि (pos2 == std::string::npos) {
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
    यदि (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        प्रत्यागम -1;
    }
    यदि (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t n_kv = mask->ne[0];
    ध्रुव int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    प्रति (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        प्रति (int64_t c = 0; c < n_kv; c++) {
            ध्रुव माप_प्रकार i = (माप_प्रकार) r * n_kv + c;
            ध्रुव भिन्न v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((ध्रुव ggml_fp16_t *) mask->data)[i]) :
                                                          ((ध्रुव भिन्न *) mask->data)[i];
            यदि (v > -अनन्त) {
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
    स्वतः get_attention_pattern_case = [](ध्रुव ggml_tensor * गाँठ) -> पूर्णाङ्क {
        यदि (गाँठ == nullptr) {
            प्रत्यागम -1;
        }

        विकल्प (गाँठ->op) {
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            यदि (गाँठ->src[0] == nullptr || गाँठ->src[1] == nullptr) {
                प्रत्यागम -1;
            }
            विकल्प (गाँठ->src[1]->op) {
            पक्ष GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                यदि (गाँठ->src[3] != nullptr && गाँठ->src[1]->src[0] != nullptr &&
                    गाँठ->src[1]->src[0]->op == GGML_OP_VIEW) {
                    प्रत्यागम 0;
                }
                भङ्ग;
            पक्ष GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                यदि (गाँठ->src[3] != nullptr && गाँठ->src[1]->src[0] != nullptr &&
                    गाँठ->src[1]->src[0]->op == GGML_OP_PERMUTE && गाँठ->src[1]->src[0]->src[0] != nullptr &&
                    गाँठ->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    प्रत्यागम 1;
                }
                भङ्ग;
            पक्ष GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                यदि (गाँठ->src[1]->view_src != nullptr) {
                    यदि (गाँठ->src[3] != nullptr) {
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
            यदि (गाँठ->src[0] != nullptr && गाँठ->src[1] != nullptr && गाँठ->src[0]->op == GGML_OP_MUL_MAT &&
                गाँठ->src[0]->src[0] != nullptr && गाँठ->src[0]->src[1] != nullptr &&
                गाँठ->src[0]->src[0]->op == GGML_OP_PERMUTE && गाँठ->src[0]->src[0]->src[0] != nullptr &&
                गाँठ->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                प्रत्यागम 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            यदि (गाँठ->src[0]->op == GGML_OP_ADD && गाँठ->src[0]->src[0] != nullptr &&
                गाँठ->src[0]->src[0]->op == GGML_OP_MUL_MAT && गाँठ->src[0]->src[0]->src[0] != nullptr &&
                गाँठ->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
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
    स्वतः get_attention_op_mask = [&get_attention_pattern_case](ध्रुव ggml_tensor * गाँठ) -> ध्रुव ggml_tensor * {
        विकल्प (get_attention_pattern_case(गाँठ)) {
        पक्ष 0:
        पक्ष 1:
            प्रत्यागम गाँठ->src[3];
        पक्ष 2:
        पक्ष 3:
            प्रत्यागम गाँठ->src[1];
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
            यदि (mask == nullptr) {
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
            यदि (cache_k_view->op != GGML_OP_VIEW) {
                अनुवर्त;
            }
            ध्रुव ggml_tensor * leaf = cache_k_view->src[0];
            स्वतः layer = extract_layer_from_name(leaf->name);
            यदि (!layer.has_value()) {
                अनुवर्त;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        प्रति (ध्रुव स्वतः & [layer, extent] : layer_extent) {
            यदि (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                यदि (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        यदि (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            प्रति (ध्रुव स्वतः & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (दीर्घ) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    तार्किक rope_seen = असत्य;
    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * गाँठ = cgraph->nodes[i];
        ध्रुव पूर्णाङ्क attention_pattern_case = get_attention_pattern_case(गाँठ);
        यदि (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            विकल्प (attention_pattern_case) {
            पक्ष 0:
                cache_k_permute = गाँठ->src[1];
                mask = गाँठ->src[3];
                भङ्ग;
            पक्ष 1:
                cache_k_permute = गाँठ->src[1]->src[0];
                mask = गाँठ->src[3];
                भङ्ग;
            पक्ष 2:
                cache_k_permute = गाँठ->src[0]->src[0];
                mask = गाँठ->src[1];
                भङ्ग;
            पक्ष 3:
                cache_k_permute = गाँठ->src[0]->src[0]->src[0];
                mask = गाँठ->src[1];
                भङ्ग;
            पक्ष 4:
            पक्ष 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                स्वतः * cache_k_view = गाँठ->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = गाँठ->src[0]->ne[1];
                यदि (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } अन्यथा {
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
            compute_params.निवेश_len = गाँठ->src[0]->ne[1];
            compute_params.token_len_per_seq = गाँठ->src[0]->ne[1];

            स्वतः * cache_k_view = cache_k_permute->src[0];
            यदि (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
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
            यदि (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } अन्यथा {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            स्वतः seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            माप_प्रकार offset;
            स_नकल(&offset, cache_k_view->op_params, परिमाण(माप_प्रकार));
            compute_params.seq_active_start = offset / seq_size;

            यदि (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } अन्यथा {
                compute_params.attention_size = mask->ne[0];
            }
            यदि (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        यदि (गाँठ->op == GGML_OP_MUL_MAT && गाँठ->src[0]->op == GGML_OP_PERMUTE &&
            गाँठ->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(गाँठ->src[0]->view_src, गाँठ->view_src)) {
            यदि (गाँठ->src[1]->op == GGML_OP_PERMUTE && गाँठ->src[1]->src[0]->op == GGML_OP_VIEW &&
                गाँठ->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = गाँठ->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        यदि (गाँठ->op == GGML_OP_TRANSPOSE && गाँठ->src[0]->op == GGML_OP_PERMUTE &&
            गाँठ->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = गाँठ->ne[0];
            यदि (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        यदि (गाँठ->op == GGML_OP_ROPE) {
            यदि (compute_params.token_len_per_seq == -1 && गाँठ->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(गाँठ->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(परिमाण(model_params.rope_params) == परिमाण(int32_t) * 16, "rope_params size");
            यदि (!rope_seen) {
                स_नकल(model_params.rope_params, गाँठ->op_params, परिमाण(int32_t) * 16);
                rope_seen = सत्य;
            } अन्यथा यदि (स_भेद(model_params.rope_params, गाँठ->op_params, परिमाण(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = सत्य;
            }
        }
        यदि (गाँठ->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = गाँठ->src[0]->ne[0];
        }
        यदि (गाँठ->op == GGML_OP_SCALE && गाँठ->view_src != nullptr && is_recurrent_cache(गाँठ->view_src)) {
            यदि (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = गाँठ->view_src->ne[1];
            } अन्यथा {
                GGML_ASSERT(model_params.n_rs_slots == गाँठ->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(गाँठ) / गाँठ->view_src->ne[0];
            compute_params.cache_rs_reset_idx = गाँठ->src[0]->view_offs / गाँठ->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        यदि (गाँठ->op == GGML_OP_CPY && गाँठ->view_src != nullptr && is_recurrent_cache(गाँठ->view_src) &&
            गाँठ->src[1] != nullptr && गाँठ->src[1]->op == GGML_OP_VIEW && गाँठ->src[1]->view_src == गाँठ->view_src) {
            ध्रुव तार्किक is_conv = is_conv_state_writeback(गाँठ);
            ध्रुव तार्किक is_gdn = गाँठ->src[0]->op == GGML_OP_VIEW && गाँठ->src[0]->src[0] != nullptr &&
                                गाँठ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ध्रुव तार्किक is_extra = गाँठ->src[0]->op == GGML_OP_GET_ROWS;
            ध्रुव तार्किक is_gdn_rollback = is_gdn && is_same_shape(गाँठ->src[0], गाँठ->src[1]);

            ध्रुव ggml_tensor * dest_view = गाँठ->src[1];
            ध्रुव ggml_tensor * cache = गाँठ->view_src;
            ध्रुव माप_प्रकार row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            यदि (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = सत्य;
            }
            यदि (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(गाँठ)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (पूर्णाङ्क) (dest_view->view_offs / row_bytes);
                यदि (is_conv) {
                    writeback.src_begin = (पूर्णाङ्क) (गाँठ->src[0]->view_offs / गाँठ->src[0]->view_src->nb[0]);
                } अन्यथा यदि (is_gdn_rollback) {
                    writeback.src_begin = (पूर्णाङ्क) (गाँठ->src[0]->view_offs / गाँठ->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, गाँठ)] = writeback;
            }
            यदि ((is_conv || is_gdn) && !is_full_single_slot_writeback(गाँठ)) {
                compute_params.s_copy_active_slot_len = (पूर्णाङ्क) dest_view->ne[1];
            }
        }
    }
    यदि (model_params.n_heads_kv == -1) {
        प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
            ध्रुव स्वतः * गाँठ = cgraph->nodes[i];
            ध्रुव ggml_tensor * mask = nullptr;
            यदि (गाँठ->op == GGML_OP_SOFT_MAX) {
                mask = गाँठ->src[1];
            } अन्यथा यदि (गाँठ->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = गाँठ->src[3];
            } अन्यथा {
                अनुवर्त;
            }
            यदि (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_निवेश) ||
                गाँठ->src[0] == nullptr) {
                अनुवर्त;
            }
            model_params.is_cacheless_attn = सत्य;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.निवेश_len = गाँठ->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.निवेश_len;
            भङ्ग;
        }
    }

    स्वतः * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    यदि (model_params.is_cacheless_attn) {
        प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
            ध्रुव स्वतः * गाँठ = cgraph->nodes[i];
            यदि (गाँठ->op == GGML_OP_GET_ROWS && is_output_idx(गाँठ->src[1], गाँठ)) {
                compute_params.output_len = गाँठ->src[1]->ne[0];
                भङ्ग;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    यदि (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    प्रत्यागम {model_params, compute_params};
}

शून्य GgmlOvDecoder::validate_cgraph() ध्रुव {
    यदि (m_model_params.n_seq > 1 && m_is_static == सत्य) {
        क्षिप std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_निवेश_shape(ध्रुव ggml_tensor * op,
                                                      ध्रुव ggml_tensor * निवेश,
                                                      पूर्णाङ्क dynamic_dim_index) ध्रुव {
    यदि (m_naive) {
        प्रत्यागम निवेश != nullptr ? ov::PartialShape{get_shape(निवेश)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape निवेश_shape;

    यदि (is_inp_tok(निवेश, op) || is_inp_pos(निवेश, op)) {
        // tokens or positions
        पूर्णाङ्क len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        यदि (m_is_static && is_inp_pos(निवेश, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        निवेश_shape = ov::PartialShape{1, 1, 1, len};

    } अन्यथा यदि (is_output_idx(निवेश, op)) {
        // output index
        निवेश_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } अन्यथा यदि (is_inp_mean(निवेश, op)) {
        निवेश_shape = m_is_static ? ov::PartialShape{1, 1, निवेश->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } अन्यथा यदि (is_inp_scale_rows(निवेश, op)) {
        निवेश_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } अन्यथा यदि (is_inp_mask(निवेश, op)) {
        // mask
        यदि (m_is_static) {
            निवेश_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } अन्यथा यदि (m_is_stateful) {
            निवेश_shape = ov::PartialShape{1, 1, -1, -1};
        } अन्यथा {
            निवेश_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } अन्यथा यदि (is_recurrent_cache(निवेश)) {
        निवेश_shape = ov::PartialShape{get_shape(निवेश)};
        यदि (!m_is_static && !m_is_stateful && निवेश->ne[1] > 1) {
            निवेश_shape[2] = -1;
        }

    } अन्यथा यदि (is_kvcache(निवेश, op)) {
        // kvcache
        निवेश_shape = ov::PartialShape{get_shape(निवेश)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ध्रुव तार्किक is_flat_kv = (निवेश->ne[1] == 1 && निवेश->ne[2] == 1 && निवेश->ne[3] == 1);
        यदि (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            निवेश_shape[2] = -1;
        }
        यदि (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            प्रतिज्ञा(निवेश_shape.size() == 4 && निवेश_shape[0] == 1 && निवेश_shape[1] == 1 &&
                   निवेश_shape[2].is_dynamic() && निवेश_shape[3].is_static());
            ध्रुव पूर्णाङ्क n_heads_kv = get_n_heads_kv_for_tensor(निवेश);
            प्रतिज्ञा(n_heads_kv > 0 && निवेश_shape[3].get_length() % n_heads_kv == 0);
            ध्रुव int64_t combined_dim = निवेश_shape[3].get_length();  // n_heads_kv * head_size
            ध्रुव int64_t head_size = combined_dim / n_heads_kv;
            निवेश_shape = {निवेश_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } अन्यथा यदि (is_kv_idx(निवेश, op)) {
        // kv update index
        पूर्णाङ्क len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        निवेश_shape = ov::PartialShape{1, 1, 1, len};

    } अन्यथा यदि (is_inp_s_copy(निवेश, op) || is_s_copy_leaf(निवेश)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        निवेश_shape = m_is_static ? ov::PartialShape{get_shape(निवेश)} : ov::PartialShape{1, 1, 1, -1};

    } अन्यथा {
        निवेश_shape = ov::PartialShape{get_shape(निवेश)};
    }
    यदि (dynamic_dim_index != -1 && m_model_is_splitted) {
        निवेश_shape[3 - dynamic_dim_index] = -1;
    }
    यदि (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_निवेश && op->src[1] == निवेश) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        यदि (m_is_static) {
            ध्रुव int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            निवेश_shape[2] = seq_active;
            निवेश_shape[3] = seq_active;
        } अन्यथा {
            निवेश_shape[2] = -1;
            निवेश_shape[3] = -1;
        }
    }
    प्रत्यागम निवेश_shape;
}

तार्किक GgmlOvDecoder::is_s_copy_leaf(ध्रुव ggml_tensor * tensor) ध्रुव {
    यदि (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        प्रत्यागम असत्य;
    }
    प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
        ध्रुव ggml_tensor * गाँठ = m_cgraph->nodes[i];
        यदि (गाँठ->op != GGML_OP_GET_ROWS || गाँठ->src[0] == nullptr || गाँठ->src[1] == nullptr) {
            अनुवर्त;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ध्रुव ggml_tensor * idx = गाँठ->src[1];
        यावत् (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        यदि (idx != tensor) {
            अनुवर्त;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ध्रुव ggml_tensor * data = गाँठ->src[0];
        यावत् (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        यदि (data != nullptr && is_recurrent_cache(data)) {
            प्रत्यागम सत्य;
        }
    }
    प्रत्यागम असत्य;
}

शून्य GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    स्वतः create_1d_निवेश = [एतत्](ध्रुव std::string & name, int64_t value, तार्किक force_parameter = असत्य) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    यदि (m_compute_params.attention_size != -1) {
        create_1d_निवेश("attention_size", m_compute_params.attention_size);
    }
    यदि (m_compute_params.attention_size_static != -1) {
        create_1d_निवेश("attention_size_static", m_compute_params.attention_size_static);
    }
    यदि (m_compute_params.attention_size_swa != -1) {
        create_1d_निवेश("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    यदि (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_निवेश("swa_window", m_compute_params.swa_window);
    }
    create_1d_निवेश("n_seq_active", m_compute_params.n_seq_active);
    create_1d_निवेश("seq_active_start", m_compute_params.seq_active_start);
    create_1d_निवेश("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    यदि (m_compute_params.token_len_per_seq != -1) {
        create_1d_निवेश("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    यदि (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_निवेश("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/सत्य);
        create_1d_निवेश("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/सत्य);
    }

    यदि (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_निवेश("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        यदि (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_निवेश("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/सत्य);
        }
    }

    प्रति (ध्रुव स्वतः & [गाँठ_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_निवेश("rs_slot_begin_" + गाँठ_name, writeback.slot_begin);
        यदि (!m_is_static && writeback.src_begin >= 0) {
            create_1d_निवेश("rs_src_begin_" + गाँठ_name, writeback.src_begin);
        }
    }
}

तार्किक GgmlOvDecoder::गाँठ_is_used_as_src(ध्रुव पूर्णाङ्क गाँठ_idx) {
    ggml_tensor * गाँठ = m_cgraph->nodes[गाँठ_idx];
    प्रति (पूर्णाङ्क i = गाँठ_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_गाँठ = m_cgraph->nodes[i];
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            यदि (other_गाँठ->src[j] == गाँठ) {
                प्रत्यागम सत्य;
            }
        }
    }
    प्रत्यागम असत्य;
}

शून्य GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * गाँठ = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        यदि (गाँठ->op == GGML_OP_NONE && गाँठ_is_used_as_src(i)) {
            std::string गाँठ_name = get_tensor_ov_name(m_cgraph, गाँठ);
            यदि (m_model_weights.find(गाँठ_name) == m_model_weights.end()) {
                m_inputs[गाँठ_name] = गाँठ;
                m_model_inputs[गाँठ_name] = {get_ov_type(गाँठ),
                                             get_graph_निवेश_shape(गाँठ, nullptr, m_गाँठ_dynamic_dims[गाँठ])};
            }
            अनुवर्त;
        }
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            स्वतः * src = गाँठ->src[i];
            यदि (src == nullptr) {
                अनुवर्त;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            यदि (src->flags & GGML_TENSOR_FLAG_निवेश) {
                src_name = get_tensor_graph_निवेश_ov_name(एतत्, m_cgraph, src, गाँठ);
            }
            यदि (m_model_weights.find(src_name) != m_model_weights.end()) {
                अनुवर्त;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            यदि (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                अनुवर्त;
            }

            तार्किक is_intermediate_गाँठ = असत्य;
            प्रति (ध्रुव स्वतः & गाँठ_info : m_गाँठ_info_list) {
                यदि (गाँठ_info.गाँठ == src) {
                    is_intermediate_गाँठ = सत्य;
                    भङ्ग;
                }
            }
            यदि (is_intermediate_गाँठ) {
                अनुवर्त;
            }
            यदि (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                अनुवर्त;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            यदि (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                यदि (स्वतः it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            यावत् (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_निवेश_shape(गाँठ, src, m_गाँठ_dynamic_dims[src])};
        }
    }
}

शून्य GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    प्रति (पूर्णाङ्क गाँठ_n = 0; गाँठ_n < m_cgraph->n_nodes; गाँठ_n++) {
        स्वतः * cur_गाँठ = m_cgraph->nodes[गाँठ_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        यदि (cur_गाँठ->op == GGML_OP_NONE || cur_गाँठ->op == GGML_OP_VIEW || cur_गाँठ->op == GGML_OP_RESHAPE) {
            अनुवर्त;
        }
        यदि (::is_inplace_op(cur_गाँठ) && ggml_nbytes(cur_गाँठ) == 0) {
            अनुवर्त;
        }
        स्वतः cur_गाँठ_use_गिनती = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_गाँठ)];
        यदि (cur_गाँठ_use_गिनती == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            यदि (cur_गाँठ != nullptr && ::is_inplace_op(cur_गाँठ) && ggml_nbytes(cur_गाँठ) > 0) {
                cur_गाँठ = cur_गाँठ->view_src;
            }
        } अन्यथा {
            पूर्णाङ्क निवेश_use_गिनती = 0;
            प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * गाँठ = m_cgraph->nodes[i];
                प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
                    यदि (गाँठ->src[j] != रिक्त && गाँठ->src[j] == cur_गाँठ) {
                        निवेश_use_गिनती++;
                    }
                }
            }
            यदि (निवेश_use_गिनती == cur_गाँठ_use_गिनती) {
                cur_गाँठ = nullptr;
            }
        }
        यदि (cur_गाँठ != nullptr) {
            std::string cur_गाँठ_name = get_tensor_ov_name(m_cgraph, cur_गाँठ);
            m_model_outputs[cur_गाँठ_name] = cur_गाँठ;
            m_model_output_names.insert(cur_गाँठ_name);
        }
    }
}

ध्रुव ggml_tensor * GgmlOvDecoder::get_tensor_used_op(ध्रुव ggml_tensor * tensor) ध्रुव {
    यदि (tensor == nullptr) {
        प्रत्यागम nullptr;
    }
    प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
        ध्रुव स्वतः * गाँठ = m_cgraph->nodes[i];
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            यदि (गाँठ->src[j] == tensor) {
                प्रत्यागम गाँठ;
            }
        }
    }
    प्रत्यागम nullptr;
}

ध्रुव ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ध्रुव std::string & name) ध्रुव {
    प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
        ध्रुव स्वतः * गाँठ = m_cgraph->nodes[i];
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            ध्रुव स्वतः * src = गाँठ->src[j];
            यदि (src == nullptr) {
                भङ्ग;
            }
            यदि (get_tensor_ov_name(m_cgraph, src) == name) {
                प्रत्यागम src;
            }
        }
    }
    प्रत्यागम nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() ध्रुव {
    std::map<std::string, std::string> kv_param_res_names;
    प्रति (ध्रुव स्वतः & name : m_model_params.kv_names) {
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
स्थिर तार्किक is_mul_mat_id_expert_weight(ध्रुव ggml_tensor * गाँठ, पूर्णाङ्क src_index) {
    प्रत्यागम गाँठ->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, तार्किक naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    स्वतः * nodes = cgraph->nodes;
    स्वतः n_nodes = cgraph->n_nodes;
    प्रति (पूर्णाङ्क गाँठ_i = 0; गाँठ_i < n_nodes; गाँठ_i++) {
        स्वतः * गाँठ = nodes[गाँठ_i];
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            स्वतः * src = गाँठ->src[i];
            यदि (src == nullptr) {
                अनुवर्त;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            यदि (is_rope_freqs_weight(base, गाँठ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            यदि (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(गाँठ, i)) {
                यदि (model_weights.find(src_name) == model_weights.end()) {
                    स्वतः weight_गाँठ = create_weight_गाँठ(base, naive);
                    weight_गाँठ->set_friendly_name(src_name);
                    model_weights[src_name] = weight_गाँठ;
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
    प्रति (पूर्णाङ्क गाँठ_i = 0; गाँठ_i < cgraph->n_nodes; गाँठ_i++) {
        स्वतः * गाँठ = cgraph->nodes[गाँठ_i];
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            स्वतः * src = गाँठ->src[i];
            यदि (src == nullptr) {
                अनुवर्त;
            }
            ध्रुव ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            यदि (is_rope_freqs_weight(base, गाँठ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            यदि (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    प्रत्यागम names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_गाँठ(ggml_tensor * tensor, तार्किक naive) {
    ध्रुव तार्किक is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    यदि (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        स्वतः * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        यदि (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            स्वतः * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            यदि (weight_extra->weight_गाँठ) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                प्रत्यागम weight_extra->weight_गाँठ;
            }
        } अन्यथा यदि (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            स्वतः * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            यदि (quant_extra->weight_गाँठ) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                प्रत्यागम quant_extra->weight_गाँठ;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    यदि (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        स्वतः weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        स्वतः weight_गाँठ = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_गाँठ->set_friendly_name(tensor->name);
        प्रत्यागम weight_गाँठ;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ध्रुव तार्किक cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    यदि (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        स्वतः it = g_nonov_weight_cache.find(tensor->data);
        यदि (it != g_nonov_weight_cache.end()) {
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
    यदि (weight_types.find(tensor->type) == weight_types.end()) {
        क्षिप std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    यदि (ggml_is_quantized(tensor->type)) {
        स्वतः use_bias = naive;
        यदि (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            माप_प्रकार raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            स_नकल(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } अन्यथा {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } अन्यथा {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_गाँठ->set_friendly_name(tensor->name);
    यदि (!is_ov_buffer) {
        यदि (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            स्वतः [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_गाँठ);
            प्रत्यागम it->second;
        }
        प्रत्यागम ov_weight.weight_गाँठ;
    }

    ggml_openvino_extra_base * extra;
    यदि (ov_weight.is_quantized()) {
        extra = नूतन ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_गाँठ);
    } अन्यथा {
        extra = नूतन ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_गाँठ);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    प्रत्यागम ov_weight.weight_गाँठ;
}

शून्य GgmlOvDecoder::dump_cgraph(ध्रुव ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    यदि (!file.is_open()) {
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
        ggml_tensor * गाँठ = cgraph->nodes[i];

        // Get buffer type name
        ध्रुव वर्ण * buf_name = "none";
        ggml_backend_buffer_t buf = गाँठ->view_src ? गाँठ->view_src->buffer : गाँठ->buffer;
        यदि (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << गाँठ->ne[0] << ", "
             << std::setw(5) << गाँठ->ne[1] << ", "
             << std::setw(5) << गाँठ->ne[2] << ", "
             << std::setw(5) << गाँठ->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(गाँठ->op) << std::right << " "
             << std::left << std::setw(45) << गाँठ->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << गाँठ->nb[0] << ", "
             << std::setw(5) << गाँठ->nb[1] << ", "
             << std::setw(5) << गाँठ->nb[2] << ", "
             << std::setw(5) << गाँठ->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            यदि (स्वतः* src = गाँठ->src[i]) {
                // Get buffer type name for source
                ध्रुव वर्ण * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                यदि (src_buf) {
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
        ggml_tensor * गाँठ = cgraph->leafs[i];

        // Get buffer type name for leaf
        ध्रुव वर्ण * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = गाँठ->view_src ? गाँठ->view_src->buffer : गाँठ->buffer;
        यदि (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << गाँठ->ne[0] << ", "
             << std::setw(5) << गाँठ->ne[1] << "] "
             << std::setw(8) << ggml_op_name(गाँठ->op) << " "
             << std::setw(16) << ggml_get_name(गाँठ)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.बंद();
}

शून्य print_tensor_address_map(ध्रुव ggml_cgraph * cgraph) {
    std::map<शून्य *, std::vector<std::string>> address_map;
    प्रति (पूर्णाङ्क गाँठ_n = 0; गाँठ_n < cgraph->n_nodes; गाँठ_n++) {
        स्वतः * गाँठ = cgraph->nodes[गाँठ_n];
        यदि (गाँठ->data) {
            स्वतः it = address_map.find(गाँठ->data);
            यदि (it == address_map.end()) {
                address_map[गाँठ->data] = std::vector<std::string>();
            }
            address_map[गाँठ->data].push_back(गाँठ->name);
        }
    }
    प्रति (ध्रुव स्वतः & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        प्रति (ध्रुव स्वतः & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ध्रुव ggml_tensor * tensor) {
    std::vector<माप_प्रकार> shape;
    प्रति (पूर्णाङ्क i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<माप_प्रकार>(tensor->ne[i]));
    }
    प्रत्यागम shape;
}

std::vector<माप_प्रकार> GgmlOvDecoder::get_stride(ध्रुव ggml_tensor * tensor) {
    std::vector<माप_प्रकार> stride;
    प्रति (पूर्णाङ्क i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<माप_प्रकार>(tensor->nb[i]));
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

ov::PartialShape GgmlOvDecoder::get_निवेश_shape(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name) ध्रुव {
    प्रत्यागम ov::PartialShape(get_shape(m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs.at(name)));
}

std::vector<माप_प्रकार> GgmlOvDecoder::get_निवेश_stride(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name) ध्रुव {
    प्रत्यागम get_stride(m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs.at(name));
}

माप_प्रकार GgmlOvDecoder::get_view_निवेश_size(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        प्रत्यागम it->second.size();
    }
    प्रत्यागम 0;
}

माप_प्रकार GgmlOvDecoder::get_view_निवेश_offset(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name, माप_प्रकार view_index) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        यदि (view_index < it->second.size()) {
            प्रत्यागम it->second[view_index].second->view_offs;
        }
    }
    प्रत्यागम 0;
}

माप_प्रकार GgmlOvDecoder::get_view_निवेश_src_offset(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name, माप_प्रकार view_index) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        यदि (view_index < it->second.size()) {
            स्वतः * view_tensor = it->second[view_index].second;
            यदि (view_tensor && view_tensor->src[0]) {
                प्रत्यागम view_tensor->src[0]->view_offs;
            }
        }
    }
    प्रत्यागम 0;
}

std::vector<माप_प्रकार> GgmlOvDecoder::get_view_निवेश_stride(पूर्णाङ्क गाँठ_idx,
                                                         ध्रुव std::string & name,
                                                         माप_प्रकार view_index) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        यदि (view_index < it->second.size()) {
            प्रत्यागम get_stride(it->second[view_index].second);
        }
    }
    प्रत्यागम {};
}

std::vector<माप_प्रकार> GgmlOvDecoder::get_view_निवेश_src_stride(पूर्णाङ्क गाँठ_idx,
                                                             ध्रुव std::string & name,
                                                             माप_प्रकार view_index) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        यदि (view_index < it->second.size()) {
            स्वतः * view_tensor = it->second[view_index].second;
            यदि (view_tensor && view_tensor->src[0]) {
                प्रत्यागम get_stride(view_tensor->src[0]);
            }
        }
    }
    प्रत्यागम {};
}

ov::Shape GgmlOvDecoder::get_view_निवेश_ggml_shape(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name, माप_प्रकार view_index) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        यदि (view_index < it->second.size()) {
            प्रत्यागम get_shape(it->second[view_index].second);
        }
    }
    प्रत्यागम {};
}

ov::Shape GgmlOvDecoder::get_view_निवेश_src_ggml_shape(पूर्णाङ्क गाँठ_idx,
                                                       ध्रुव std::string & name,
                                                       माप_प्रकार view_index) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        यदि (view_index < it->second.size()) {
            स्वतः * view_tensor = it->second[view_index].second;
            यदि (view_tensor && view_tensor->src[0]) {
                प्रत्यागम get_shape(view_tensor->src[0]);
            }
        }
    }
    प्रत्यागम {};
}

ov::PartialShape GgmlOvDecoder::get_view_निवेश_ov_shape(पूर्णाङ्क गाँठ_idx,
                                                        ध्रुव std::string & name,
                                                        माप_प्रकार view_index) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        यदि (view_index < it->second.size()) {
            स्वतः * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            स्वतः dynamic_it = m_गाँठ_dynamic_dims.find(tensor);
            यदि (dynamic_it != m_गाँठ_dynamic_dims.end() && dynamic_it->second != -1) {
                पूर्णाङ्क dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            प्रत्यागम shape;
        }
    }
    प्रत्यागम {};
}

ov::PartialShape GgmlOvDecoder::get_view_निवेश_src_ov_shape(पूर्णाङ्क गाँठ_idx,
                                                            ध्रुव std::string & name,
                                                            माप_प्रकार view_index) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        यदि (view_index < it->second.size()) {
            स्वतः * view_tensor = it->second[view_index].second;
            यदि (view_tensor && view_tensor->src[0]) {
                स्वतः * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                स्वतः dynamic_it = m_गाँठ_dynamic_dims.find(src_tensor);
                यदि (dynamic_it != m_गाँठ_dynamic_dims.end() && dynamic_it->second != -1) {
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

std::string GgmlOvDecoder::get_view_निवेश_name(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name, माप_प्रकार view_index) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        यदि (view_index < it->second.size()) {
            प्रत्यागम it->second[view_index].first;
        }
    }
    प्रत्यागम "";
}

std::string GgmlOvDecoder::get_view_निवेश_src_name(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name, माप_प्रकार view_index) ध्रुव {
    स्वतः it = m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.find(name);
    यदि (it != m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_views.end()) {
        यदि (view_index < it->second.size()) {
            स्वतः * view_tensor = it->second[view_index].second;
            यदि (view_tensor && view_tensor->src[0]) {
                प्रत्यागम get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    प्रत्यागम "";
}

ov::element::Type GgmlOvDecoder::get_निवेश_type(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name) ध्रुव {
    प्रत्यागम get_ov_type(m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs.at(name));
}

माप_प्रकार GgmlOvDecoder::get_निवेश_size() ध्रुव {
    प्रत्यागम m_model_inputs.size();
}

माप_प्रकार GgmlOvDecoder::get_निवेश_size(पूर्णाङ्क गाँठ_idx) ध्रुव {
    प्रत्यागम m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_निवेश_names(पूर्णाङ्क गाँठ_idx) ध्रुव {
    प्रत्यागम m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(पूर्णाङ्क गाँठ_idx) ध्रुव {
    स्वतः * ggml_tensor = m_गाँठ_info_list[गाँठ_idx].गाँठ;
    प्रत्यागम ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ध्रुव पूर्णाङ्क गाँठ_idx) ध्रुव {
    प्रत्यागम get_ov_type(m_गाँठ_info_list[गाँठ_idx].गाँठ);
}

std::vector<माप_प्रकार> GgmlOvDecoder::get_output_stride(पूर्णाङ्क गाँठ_idx) ध्रुव {
    स्वतः * ggml_tensor = m_गाँठ_info_list[गाँठ_idx].गाँठ;
    प्रत्यागम get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(पूर्णाङ्क गाँठ_idx) ध्रुव {
    स्वतः * गाँठ = m_गाँठ_info_list[गाँठ_idx].गाँठ;
    यदि (गाँठ->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        प्रति (पूर्णाङ्क i = गाँठ_idx + 1; i < m_cgraph->n_nodes; i++) {
            स्वतः * consumer = m_cgraph->nodes[i];
            यदि (consumer->op != GGML_OP_VIEW || consumer->src[0] != गाँठ) {
                अनुवर्त;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            स्वतः & name = consumer->view_offs == 0 ? attn_name : state_name;
            यदि (!name.empty()) {
                प्रत्यागम {m_गाँठ_info_list[गाँठ_idx].गाँठ_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        यदि (!attn_name.empty() && !state_name.empty()) {
            प्रत्यागम {attn_name, state_name};
        }
    }
    प्रत्यागम {m_गाँठ_info_list[गाँठ_idx].गाँठ_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(पूर्णाङ्क गाँठ_idx) ध्रुव {
    स्वतः * गाँठ = m_गाँठ_info_list[गाँठ_idx].गाँठ;
    यदि (!::is_inplace_op(गाँठ) || गाँठ->view_src == nullptr || ggml_nbytes(गाँठ) == 0) {
        प्रत्यागम "";
    }
    ध्रुव पूर्णाङ्क op_case = m_गाँठ_info_list[गाँठ_idx].गाँठ_op_case;
    यदि (गाँठ->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        प्रत्यागम "";
    }
    प्रत्यागम get_tensor_ov_name(m_cgraph, गाँठ->view_src);
}

तार्किक GgmlOvDecoder::is_view_like_alias_of(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & view_src_name) ध्रुव {
    स्वतः * गाँठ = m_गाँठ_info_list[गाँठ_idx].गाँठ;
    यदि (गाँठ->view_src == nullptr || get_tensor_ov_name(m_cgraph, गाँठ->view_src) != view_src_name) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम गाँठ->op == GGML_OP_RESHAPE || गाँठ->op == GGML_OP_VIEW;
}

ध्रुव std::string & GgmlOvDecoder::get_op_name() ध्रुव {
    स्थिर ध्रुव std::string unknown_name = "UNKNOWN_OP_NAME";
    प्रत्यागम unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(पूर्णाङ्क गाँठ_idx) ध्रुव {
    स्वतः it = m_गाँठ_dynamic_dims.find(m_गाँठ_info_list[गाँठ_idx].गाँठ);
    यदि (it == m_गाँठ_dynamic_dims.end()) {
        प्रत्यागम -1;
    }
    प्रत्यागम it->second;
}

ध्रुव std::string & GgmlOvDecoder::get_op_name(पूर्णाङ्क गाँठ_idx) ध्रुव {
    प्रत्यागम m_गाँठ_info_list[गाँठ_idx].गाँठ_name;
}

int32_t * GgmlOvDecoder::get_निवेश_op_params(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name) ध्रुव {
    प्रत्यागम m_गाँठ_info_list[गाँठ_idx].गाँठ_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(पूर्णाङ्क गाँठ_idx) ध्रुव {
    प्रत्यागम m_गाँठ_info_list[गाँठ_idx].गाँठ->op_params;
}

माप_प्रकार GgmlOvDecoder::get_output_op_offset(पूर्णाङ्क गाँठ_idx) ध्रुव {
    प्रत्यागम m_गाँठ_info_list[गाँठ_idx].गाँठ->view_offs;
}

शून्य GgmlOvDecoder::visit_subgraph(std::function<शून्य(std::shared_ptr<GgmlDecoder>, पूर्णाङ्क गाँठ_idx)> गाँठ_visitor) ध्रुव {
    प्रति (पूर्णाङ्क गाँठ_idx = 0; गाँठ_idx < m_cgraph->n_nodes; गाँठ_idx++) {
        यदि (m_cgraph->nodes[गाँठ_idx]->op == GGML_OP_NONE) {
            अनुवर्त;
        }
        गाँठ_visitor(std::make_shared<GgmlOvDecoder>(*एतत्), गाँठ_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ध्रुव ggml_tensor * गाँठ) {
    विकल्प (गाँठ->op) {
    पक्ष GGML_OP_UNARY:
        प्रत्यागम std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(गाँठ));
    पक्ष GGML_OP_GLU:
        प्रत्यागम std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(गाँठ));
    स्वभाव:
        प्रत्यागम std::string("GGML_OP_") + ggml_op_name(गाँठ->op);
    }
}

ध्रुव std::string & GgmlOvDecoder::get_op_type(पूर्णाङ्क गाँठ_idx) ध्रुव {
    प्रत्यागम m_गाँठ_info_list[गाँठ_idx].गाँठ_op_type;
}

ध्रुव std::string & GgmlOvDecoder::get_op_type() ध्रुव {
    स्थिर ध्रुव std::string unknown_op = "UNKNOWN_GGML_OP";
    प्रत्यागम unknown_op;
}

शून्य GgmlOvDecoder::compute_गाँठ_dynamic_dims() {
    स्वतः visit_गाँठ = [&](स्वतः && self, ggml_tensor * गाँठ) -> शून्य {
        यदि (!गाँठ) {
            प्रत्यागम;
        }

        यदि (गाँठ->op == GGML_OP_CPY) {
            m_गाँठ_dynamic_dims[गाँठ] = -1;
        }

        यदि (m_गाँठ_dynamic_dims.count(गाँठ)) {
            प्रत्यागम;
        }
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = गाँठ->src[i];
            यदि (src == nullptr) {
                अनुवर्त;
            }
            संरचना ggml_tensor * मूल_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            यदि (मूल_src) {
                यदि (is_inp_tok(मूल_src, गाँठ) || is_inp_pos(मूल_src, गाँठ) || is_output_idx(मूल_src, गाँठ)) {
                    m_गाँठ_dynamic_dims[मूल_src] = 0;
                    m_गाँठ_dynamic_dims[src] = m_गाँठ_dynamic_dims[मूल_src];
                    अनुवर्त;
                }
                self(self, मूल_src);
                m_गाँठ_dynamic_dims[src] = m_गाँठ_dynamic_dims[मूल_src];
            } अन्यथा {
                यदि (is_inp_tok(src, गाँठ) || is_inp_pos(src, गाँठ) || is_output_idx(src, गाँठ)) {
                    m_गाँठ_dynamic_dims[src] = 0;
                    अनुवर्त;
                }
                यदि (is_inp_scale_rows(src, गाँठ)) {
                    m_गाँठ_dynamic_dims[src] = 1;
                    अनुवर्त;
                }
                यदि (गाँठ->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_गाँठ_dynamic_dims[src] = 1;
                    अनुवर्त;
                }
                self(self, src);
            }
        }
        विकल्प (गाँठ->op) {
        पक्ष GGML_OP_NONE:
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            भङ्ग;
        पक्ष GGML_OP_GET_ROWS:
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            यदि (m_गाँठ_dynamic_dims[गाँठ->src[1]] != -1) {
                स्वतः dynamic_dim_idx = m_गाँठ_dynamic_dims[गाँठ->src[1]];
                यदि (dynamic_dim_idx == 0) {
                    m_गाँठ_dynamic_dims[गाँठ] = 1;
                } अन्यथा {
                    स्वतः dynamic_dim_stride = गाँठ->src[1]->nb[dynamic_dim_idx] / ggml_type_size(गाँठ->src[1]->type) *
                                              ggml_type_size(गाँठ->src[0]->type);
                    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                        यदि (dynamic_dim_stride == गाँठ->src[0]->nb[i]) {
                            m_गाँठ_dynamic_dims[गाँठ] = i;
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
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            यदि (m_गाँठ_dynamic_dims[गाँठ->src[0]] != -1) {
                m_गाँठ_dynamic_dims[गाँठ] = m_गाँठ_dynamic_dims[गाँठ->src[0]];
            }
            यदि (m_गाँठ_dynamic_dims[गाँठ->src[1]] != -1) {
                m_गाँठ_dynamic_dims[गाँठ] = m_गाँठ_dynamic_dims[गाँठ->src[1]];
            }
            भङ्ग;
        पक्ष GGML_OP_PERMUTE:
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            यदि (m_गाँठ_dynamic_dims[गाँठ->src[0]] != -1) {
                स्वतः dynamic_dim_idx = m_गाँठ_dynamic_dims[गाँठ->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                    यदि (गाँठ->op_params[i] == dynamic_dim_idx) {
                        m_गाँठ_dynamic_dims[गाँठ] = i;
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
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            यदि (m_गाँठ_dynamic_dims[गाँठ->src[0]] != -1) {
                यदि (गाँठ->src[0]->op == GGML_OP_NONE) {
                    m_गाँठ_dynamic_dims[गाँठ] = m_गाँठ_dynamic_dims[गाँठ->src[0]];
                    भङ्ग;
                }
                स्वतः dynamic_dim_idx = m_गाँठ_dynamic_dims[गाँठ->src[0]];
                स्वतः dynamic_dim_मूल्य = गाँठ->src[0]->ne[dynamic_dim_idx];
                स्वतः dynamic_dim_stride =
                    गाँठ->src[0]->nb[dynamic_dim_idx] / ggml_type_size(गाँठ->src[0]->type) * ggml_type_size(गाँठ->type);
                प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                    यदि (गाँठ->nb[i] == dynamic_dim_stride) {
                        m_गाँठ_dynamic_dims[गाँठ] = i;
                        भङ्ग;
                    }
                }
                यदि (m_गाँठ_dynamic_dims[गाँठ] != -1 && dynamic_dim_मूल्य != गाँठ->ne[m_गाँठ_dynamic_dims[गाँठ]]) {
                    m_गाँठ_dynamic_dims[गाँठ] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    यदि (ggml_nelements(गाँठ) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      गाँठ->name, गाँठ->src[0]->name);
                    }
                }
            }
            भङ्ग;
        }
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_RESHAPE: {
            यदि (is_same_shape(गाँठ->src[0], गाँठ)) {
                m_गाँठ_dynamic_dims[गाँठ] = m_गाँठ_dynamic_dims[गाँठ->src[0]];
                भङ्ग;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            यदि (m_गाँठ_dynamic_dims[गाँठ->src[0]] != -1) {
                स्वतः dynamic_dim_idx = m_गाँठ_dynamic_dims[गाँठ->src[0]];
                स्वतः dynamic_dim_stride = गाँठ->src[0]->nb[dynamic_dim_idx];
                प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                    यदि (गाँठ->nb[i] == dynamic_dim_stride && गाँठ->ne[i] == गाँठ->src[0]->ne[dynamic_dim_idx]) {
                        m_गाँठ_dynamic_dims[गाँठ] = i;
                        भङ्ग;
                    }
                }
                यदि (m_गाँठ_dynamic_dims[गाँठ] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", गाँठ->name);
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
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            यदि (m_गाँठ_dynamic_dims[गाँठ->src[0]] != -1) {
                स्वतः q_dynamic_dim = m_गाँठ_dynamic_dims[गाँठ->src[0]];
                m_गाँठ_dynamic_dims[गाँठ] = q_to_out[q_dynamic_dim];
            }
            भङ्ग;
        }
        पक्ष GGML_OP_CONT:
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            यदि (m_गाँठ_dynamic_dims[गाँठ->src[0]] != -1) {
                स्वतः dynamic_dim_idx = m_गाँठ_dynamic_dims[गाँठ->src[0]];
                यदि (ggml_are_same_shape(गाँठ, गाँठ->src[0])) {
                    m_गाँठ_dynamic_dims[गाँठ] = dynamic_dim_idx;
                } अन्यथा {
                    माप_प्रकार src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(गाँठ->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (गाँठ->src[0]->ne[0] / ggml_blck_size(गाँठ->src[0]->type));
                    प्रति (पूर्णाङ्क i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * गाँठ->src[0]->ne[i - 1];
                    }

                    स्वतः dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(गाँठ->src[0]->type) *
                                              ggml_type_size(गाँठ->type);
                    पूर्णाङ्क matched_dim_गिनती = 0;
                    पूर्णाङ्क first_matched_dim = -1;
                    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                        यदि (गाँठ->nb[i] == dynamic_dim_stride && गाँठ->ne[i] == गाँठ->src[0]->ne[dynamic_dim_idx]) {
                            यदि (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_गाँठ_dynamic_dims[गाँठ] = i;
                            matched_dim_गिनती++;
                        }
                    }
                    यदि (matched_dim_गिनती > 1 && गाँठ->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_गाँठ_dynamic_dims[गाँठ] = first_matched_dim;
                    } अन्यथा यदि (matched_dim_गिनती != 1) {
                        m_गाँठ_dynamic_dims[गाँठ] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      गाँठ->name, गाँठ->src[0]->name);
                    }
                }
            }
            भङ्ग;
        पक्ष GGML_OP_CONCAT:
            प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
                यदि (गाँठ->src[0]->ne[i] != गाँठ->ne[i]) {
                    m_गाँठ_dynamic_dims[गाँठ] = i;
                    भङ्ग;
                }
            }
            भङ्ग;
        पक्ष GGML_OP_SSM_CONV:
        पक्ष GGML_OP_GATED_DELTA_NET:
            m_गाँठ_dynamic_dims[गाँठ] = 1;
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
            m_गाँठ_dynamic_dims[गाँठ] = m_गाँठ_dynamic_dims[गाँठ->src[0]];
            भङ्ग;
        पक्ष GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_गाँठ_dynamic_dims[गाँठ] =
                (m_गाँठ_dynamic_dims[गाँठ->src[0]] == 0) ? -1 : m_गाँठ_dynamic_dims[गाँठ->src[0]];
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT_ID:
        पक्ष GGML_OP_SOLVE_TRI:
            m_गाँठ_dynamic_dims[गाँठ] = m_गाँठ_dynamic_dims[गाँठ->src[1]];
            भङ्ग;
        पक्ष GGML_OP_CPY:
        पक्ष GGML_OP_SET_ROWS:
        पक्ष GGML_OP_SUM:
        पक्ष GGML_OP_MEAN:
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            भङ्ग;
        पक्ष GGML_OP_IM2COL: {
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            यदि (m_गाँठ_dynamic_dims[गाँठ->src[1]] != -1) {
                ध्रुव तार्किक is_2D = गाँठ->op_params[6] == 1;
                ध्रुव पूर्णाङ्क src_dyn = m_गाँठ_dynamic_dims[गाँठ->src[1]];
                यदि (is_2D) {
                    यदि (src_dyn == 0) {
                        m_गाँठ_dynamic_dims[गाँठ] = 1;  // IW -> OW
                    } अन्यथा यदि (src_dyn == 1) {
                        m_गाँठ_dynamic_dims[गाँठ] = 2;  // IH -> OH
                    } अन्यथा यदि (src_dyn == 3) {
                        m_गाँठ_dynamic_dims[गाँठ] = 3;  // N  -> N
                    }
                } अन्यथा {
                    यदि (src_dyn == 0) {
                        m_गाँठ_dynamic_dims[गाँठ] = 1;  // IW -> OW
                    } अन्यथा यदि (src_dyn == 2) {
                        m_गाँठ_dynamic_dims[गाँठ] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                यदि (m_गाँठ_dynamic_dims[गाँठ] != -1) {
                    OPENVINO_ASSERT(गाँठ->src[1]->ne[src_dyn] == गाँठ->ne[m_गाँठ_dynamic_dims[गाँठ]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(गाँठ->name) +
                                        " and its src[1]: " + std::string(गाँठ->src[1]->name));
                }
            }
            भङ्ग;
        }
        पक्ष GGML_OP_IM2COL_3D: {
            m_गाँठ_dynamic_dims[गाँठ] = -1;
            यदि (m_गाँठ_dynamic_dims[गाँठ->src[1]] != -1) {
                ध्रुव पूर्णाङ्क src_dyn = m_गाँठ_dynamic_dims[गाँठ->src[1]];
                यदि (src_dyn == 0) {
                    m_गाँठ_dynamic_dims[गाँठ] = 1;  // IW -> OW
                } अन्यथा यदि (src_dyn == 1) {
                    m_गाँठ_dynamic_dims[गाँठ] = 2;  // IH -> OH
                } अन्यथा यदि (src_dyn == 3) {
                    m_गाँठ_dynamic_dims[गाँठ] = 3;  // N  -> N
                }
                यदि (m_गाँठ_dynamic_dims[गाँठ] != -1) {
                    OPENVINO_ASSERT(गाँठ->src[1]->ne[src_dyn] == गाँठ->ne[m_गाँठ_dynamic_dims[गाँठ]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(गाँठ->name) +
                                        " and its src[1]: " + std::string(गाँठ->src[1]->name));
                }
            }
            भङ्ग;
        }
        स्वभाव:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(गाँठ->op), गाँठ->name);
            भङ्ग;
        }
    };

    प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * गाँठ = m_cgraph->nodes[i];
        visit_गाँठ(visit_गाँठ, गाँठ);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    यदि (0) {
        प्रति (पूर्णाङ्क i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * गाँठ = m_cgraph->nodes[i];
            पूर्णाङ्क dynamic_dim = m_गाँठ_dynamic_dims[गाँठ];
            std::cout << "[" << i << "] " << "node_name: " << गाँठ->name << " op: " << ggml_op_name(गाँठ->op)
                      << " shape: [";
            प्रति (पूर्णाङ्क j = 0; j < 4; j++) {
                यदि (j == dynamic_dim) {
                    std::cout << "*";
                } अन्यथा {
                    std::cout << गाँठ->ne[j];
                }
                यदि (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = गाँठ->src[j];
                यदि (src == nullptr) {
                    अनुवर्त;
                }
                पूर्णाङ्क src_dynamic_dim = m_गाँठ_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                प्रति (पूर्णाङ्क k = 0; k < 4; k++) {
                    यदि (k == src_dynamic_dim) {
                        std::cout << "*";
                    } अन्यथा {
                        std::cout << src->ne[k];
                    }
                    यदि (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
