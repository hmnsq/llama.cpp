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
                             पूर्णांक prefill_chunk_size) :
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
    जर (!printed_address_map) {
        जर (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = सत्य;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_आदान_output();
    compute_गाठ_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    साठी (पूर्णांक गाठ_n = 0; गाठ_n < cgraph->n_nodes; गाठ_n++) {
        m_गाठ_info_list[गाठ_n].गाठ_op_case = compute_op_case(m_गाठ_info_list[गाठ_n].गाठ);
        m_गाठ_info_list[गाठ_n].गाठ_op_type = compute_op_type(m_गाठ_info_list[गाठ_n].गाठ);
    }

    add_extra_inputs();
}

शून्य GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_गाठ_info_list.clear();
    set_आदान_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
तार्किक GgmlOvDecoder::is_bound_to(स्थिरांक ggml_cgraph * cgraph) स्थिरांक {
    परत m_cgraph == cgraph && cgraph->n_nodes > 0 && m_गाठ_info_list.size() == (माप_प्रकार) cgraph->n_nodes &&
           m_गाठ_info_list.front().गाठ == cgraph->nodes[0] &&
           m_गाठ_info_list.back().गाठ == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = सत्य;
    set_आदान_output();
    compute_model_inputs();
    compute_model_outputs();
    साठी (पूर्णांक गाठ_n = 0; गाठ_n < cgraph->n_nodes; गाठ_n++) {
        m_गाठ_info_list[गाठ_n].गाठ_op_case = compute_op_case(m_गाठ_info_list[गाठ_n].गाठ);
        m_गाठ_info_list[गाठ_n].गाठ_op_type = compute_op_type(m_गाठ_info_list[गाठ_n].गाठ);
    }
}

नामाकाश {
तार्किक is_inplace_op(स्थिरांक ggml_tensor * गाठ) {
    परत गाठ->op == GGML_OP_SET_ROWS || गाठ->op == GGML_OP_CPY || (गाठ->op == GGML_OP_SCALE && गाठ->view_src);
}

तार्किक is_same_shape(स्थिरांक ggml_tensor * a, स्थिरांक ggml_tensor * b) {
    साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
        जर (a->ne[i] != b->ne[i]) {
            परत असत्य;
        }
    }
    परत सत्य;
}

तार्किक is_conv_states_all_tensor(स्थिरांक ggml_tensor * tensor) {
    परत tensor != nullptr && म_भेदन(tensor->name, "conv_states_all", म_माप("conv_states_all")) == 0;
}

तार्किक is_full_single_slot_writeback(स्थिरांक ggml_tensor * गाठ) {
    परत गाठ->view_src != nullptr && गाठ->view_src->ne[1] == 1 && गाठ->src[1] != nullptr &&
           गाठ->src[1]->op == GGML_OP_VIEW && गाठ->src[1]->view_src == गाठ->view_src &&
           गाठ->src[1]->view_offs == 0 && ggml_nbytes(गाठ->src[1]) == ggml_nbytes(गाठ->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
तार्किक is_moe_expert_sum_add(स्थिरांक ggml_tensor * गाठ) {
    std::vector<स्थिरांक ggml_tensor *> leaves;
    स्थिरांक ggml_tensor * cur = गाठ;
    पर्यंत (cur->op == GGML_OP_ADD) {
        जर (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            परत असत्य;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    स्थिरांक ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    साठी (स्थिरांक ggml_tensor * leaf : leaves) {
        जर (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            परत असत्य;
        }
        स्थिरांक ggml_tensor * leaf_base = leaf->src[0];
        जर (base == nullptr) {
            base = leaf_base;
        } अन्यथा जर (leaf_base != base) {
            परत असत्य;
        }
        जर (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            परत असत्य;
        }
        जर (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            परत असत्य;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        जर (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            परत असत्य;
        }
    }

    परत base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<माप_प्रकार>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(स्थिरांक ggml_cgraph * cgraph, स्थिरांक ggml_tensor * tensor) {
    जर (tensor == nullptr) {
        परत "";
    }
    जर ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        स्थिरांक स्वतः * गाठ = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        जर (गाठ != cgraph->nodes + cgraph->n_nodes) {
            परत std::string(tensor->name) + "#n" + std::to_string(गाठ - cgraph->nodes);
        }
        स्थिरांक स्वतः * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        जर (leaf != cgraph->leafs + cgraph->n_leafs) {
            परत std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    परत tensor->name;
}

स्थिर std::string get_tensor_ov_name(स्थिरांक ggml_cgraph * cgraph, स्थिरांक ggml_tensor * tensor) {
    परत GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

स्थिर std::string get_tensor_graph_आदान_ov_name(स्थिरांक GgmlOvDecoder * decoder,
                                                  स्थिरांक ggml_cgraph * cgraph,
                                                  स्थिरांक ggml_tensor * tensor,
                                                  स्थिरांक ggml_tensor * op) {
    जर (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        परत "inp_pos";
    }
    जर (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        परत "embd";
    }
    जर (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        स्थिरांक तार्किक is_swa = decoder->is_swa_mask(tensor);
        जर (decoder->is_stateful()) {
            परत is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        जर (is_swa) {
            परत get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    परत get_tensor_ov_name(cgraph, tensor);
}

शून्य GgmlOvDecoder::set_आदान_output() {
    साठी (पूर्णांक गाठ_n = 0; गाठ_n < m_cgraph->n_nodes; गाठ_n++) {
        स्वतः * गाठ = m_cgraph->nodes[गाठ_n];

        गाठ_Info current_गाठ_info;
        स्वतः गाठ_name = get_tensor_ov_name(m_cgraph, गाठ);

        current_गाठ_info.गाठ = गाठ;
        current_गाठ_info.गाठ_name = गाठ_name;
        current_गाठ_info.गाठ_op_case = 0;
        current_गाठ_info.data_addr = गाठ->data;

        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            स्वतः * src = गाठ->src[i];
            जर (src == nullptr) {
                सुरूठेवा;
            }
            स्वतः src_name = get_tensor_ov_name(m_cgraph, src);
            जर (src->flags & GGML_TENSOR_FLAG_आदान) {
                src_name = get_tensor_graph_आदान_ov_name(हे, m_cgraph, src, गाठ);
            }
            current_गाठ_info.गाठ_inputs[src_name] = src;
            current_गाठ_info.गाठ_inputs_names.push_back(src_name);

            जर (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_गाठ_info.गाठ_inputs_views[src_name])> view_chain;
                स्वतः * current = src;

                पर्यंत (current != nullptr) {
                    स्वतः current_name = get_tensor_ov_name(m_cgraph, current);
                    जर (current->flags & GGML_TENSOR_FLAG_आदान) {
                        current_name = get_tensor_graph_आदान_ov_name(हे, m_cgraph, current, गाठ);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    जर (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } अन्यथा {
                        थांबा;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_गाठ_info.गाठ_inputs_views[src_name] = view_chain;
            }
        }

        m_गाठ_info_list.push_back(current_गाठ_info);
    }
}

पूर्णांक GgmlOvDecoder::compute_op_case(स्थिरांक ggml_tensor * गाठ) स्थिरांक {
    पूर्णांक op_case = 0;
    निवड (गाठ->op) {
    स्थिती GGML_OP_RESHAPE: {
        जर (m_naive) {
            थांबा;
        }
        स्वतः name = std::string(गाठ->name);
        स्वतः * src = गाठ->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        तार्किक recurrent_sequence = असत्य;
        साठी (पूर्णांक i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            स्थिरांक स्वतः * consumer = m_cgraph->nodes[i];
            जर (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == गाठ) {
                परत 1;
            } अन्यथा जर (consumer->op == GGML_OP_SSM_CONV) {
                स्थिरांक स्वतः * concat = consumer->src[0];
                जर (concat->op == GGML_OP_CONCAT) {
                    स्थिरांक स्वतः * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == गाठ;
                }
            } अन्यथा जर (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                स्थिरांक स्वतः * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == गाठ;
            }
        }
        जर (recurrent_sequence && गाठ->ne[0] == src->ne[0] && गाठ->ne[3] == 1) {
            परत 6;
        }
        जर (गाठ->ne[0] == src->ne[0] && गाठ->ne[2] == 1 && गाठ->ne[3] == 1) {
            परत 5;
        }
        जर (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == गाठ->ne[0] && src->src[0]->ne[1] == गाठ->ne[1]) {
            op_case = 4;
        } अन्यथा जर (गाठ->ne[0] * गाठ->ne[1] == src->ne[0]) {
            op_case = 1;
        } अन्यथा जर (src->ne[0] * src->ne[1] == गाठ->ne[0]) {
            op_case = 2;
            जर (src->ne[2] * src->ne[3] == गाठ->ne[1]) {
                op_case = 5;
            }
        } अन्यथा जर (गाठ->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == गाठ->ne[1]) {
            op_case = 3;
        } अन्यथा जर (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } अन्यथा जर (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } अन्यथा जर (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        जर (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            स्थिरांक स्वतः * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            तार्किक recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            साठी (पूर्णांक i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                स्थिरांक स्वतः * consumer = m_cgraph->nodes[i];
                जर (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    साठी (पूर्णांक j : {3, 4}) {
                        स्थिरांक स्वतः * gate = consumer->src[j];
                        जर (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == गाठ;
                    }
                } अन्यथा जर (consumer->op == GGML_OP_MUL) {
                    साठी (पूर्णांक j = 0; j < 2; ++j) {
                        स्थिरांक स्वतः * gate = consumer->src[j];
                        स्थिरांक स्वतः * norm = consumer->src[1 - j];
                        जर (gate->op != GGML_OP_UNARY || gate->src[0] != गाठ) {
                            सुरूठेवा;
                        }
                        जर (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            जर (recurrent) {
                op_case = 9;
            }
        }
        थांबा;
    }
    स्थिती GGML_OP_PERMUTE: {
        जर (गाठ->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } अन्यथा जर (गाठ->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(गाठ->view_src->name);
            पूर्णांक layer = extract_layer_from_name(src_name).value();
            जर (ggml_is_contiguous(गाठ->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                जर (!is_swa_layer(layer)) {
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
                जर (!is_swa_layer(layer)) {
                    op_case = 5;
                } अन्यथा {
                    op_case = 6;
                }
            }
        } अन्यथा {
            // rope'ed query tensor
            op_case = 2;
        }
        थांबा;
    }
    स्थिती GGML_OP_MUL_MAT: {
        जर (गाठ->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        थांबा;
    }
    स्थिती GGML_OP_GET_ROWS: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        जर (गाठ->src[0]->op == GGML_OP_RESHAPE && गाठ->src[0]->src[0] != nullptr &&
            is_recurrent_cache(गाठ->src[0]->src[0])) {
            स्थिरांक तार्किक single_slot = गाठ->src[0]->src[0]->ne[1] == 1;
            जर (गाठ->src[1]->op == GGML_OP_VIEW) {
                op_case = (गाठ->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } अन्यथा जर (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        थांबा;
    }
    स्थिती GGML_OP_ROPE: {
        स्थिरांक पूर्णांक mode = गाठ->op_params[2];
        निवड (mode) {
        स्थिती GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            थांबा;
        }
        स्थिती GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            थांबा;
        }
        स्थिती GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            थांबा;
        }
        स्थिती GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            थांबा;
        }
        पूर्वनिर्धारित:
            op_case = 0;
            थांबा;
        }
        थांबा;
    }
    स्थिती GGML_OP_VIEW: {
        जर (!m_model_params.has_rs_rollback && गाठ->src[0] != nullptr &&
            गाठ->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            थांबा;
        }
        जर (m_is_static && गाठ->src[0] != nullptr &&
            (गाठ->src[0]->op == GGML_OP_GATED_DELTA_NET || गाठ->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            थांबा;
        }
        जर (गाठ->src[0]->op == GGML_OP_VIEW) {
            स्वतः * src = गाठ->src[0];
            जर (ggml_nelements(गाठ) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            जर (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            स्वतः * src = गाठ->src[0];
            जर (ggml_nelements(गाठ) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                जर (गाठ->ne[0] == src->ne[0] && गाठ->ne[1] == src->ne[2] && गाठ->ne[2] == 1 &&
                    गाठ->nb[0] == src->nb[0] && गाठ->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    थांबा;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                पूर्णांक diff_मोजणी = 0;
                साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
                    जर (गाठ->ne[i] != src->ne[i]) {
                        diff_मोजणी++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    जर (गाठ->ne[i] > src->ne[i]) {
                        परत 0;
                    }
                }
                जर (diff_मोजणी >= 1) {
                    op_case = 0;
                }
            }
        }
        थांबा;
    }
    स्थिती GGML_OP_RMS_NORM: {
        जर (गाठ->src[0]->op == GGML_OP_VIEW) {
            जर (is_same_shape(गाठ->src[0]->src[0], गाठ->src[0])) {
                op_case = 1;
            } अन्यथा जर (!m_model_params.has_rs_rollback &&
                       गाठ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } अन्यथा जर (गाठ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        थांबा;
    }
    स्थिती GGML_OP_POOL_2D: {
        स्थिरांक ggml_op_pool pool_mode = static_cast<ggml_op_pool>(गाठ->op_params[0]);
        निवड (pool_mode) {
        स्थिती GGML_OP_POOL_MAX: {
            op_case = 1;
            थांबा;
        }
        स्थिती GGML_OP_POOL_AVG: {
            op_case = 2;
            थांबा;
        }
        पूर्वनिर्धारित:
            op_case = 0;
            थांबा;
        }
        थांबा;
    }
    स्थिती GGML_OP_UPSCALE: {
        स्थिरांक int32_t mode_flags = गाठ->op_params[0];
        स्थिरांक ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        निवड (scale_mode) {
        स्थिती GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            थांबा;
        }
        स्थिती GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            थांबा;
        }
        स्थिती GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            थांबा;
        }
        पूर्वनिर्धारित:
            op_case = 0;
            थांबा;
        }
        थांबा;
    }
    स्थिती GGML_OP_CPY: {
        जर (गाठ->src[0]->op == GGML_OP_VIEW) {
            जर (गाठ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                जर (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(गाठ) ? 7 : 10;
                } अन्यथा {
                    op_case = 1;
                }
            } अन्यथा जर (GgmlOvDecoder::is_conv_state_writeback(गाठ)) {
                op_case = is_full_single_slot_writeback(गाठ) ? 8 : 2;
                थांबा;
            } अन्यथा जर (is_conv_states_all_tensor(गाठ->view_src) && गाठ->src[1] != nullptr &&
                       गाठ->src[1]->op == GGML_OP_VIEW && गाठ->src[1]->view_src == गाठ->view_src) {
                op_case = 4;
                थांबा;
            } अन्यथा जर (गाठ->src[0]->src[0]->op == GGML_OP_GET_ROWS && गाठ->src[1] != nullptr &&
                       गाठ->src[1]->op == GGML_OP_VIEW && गाठ->src[1]->view_src != nullptr &&
                       is_recurrent_cache(गाठ->src[1]->view_src) && गाठ->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } अन्यथा जर (गाठ->src[0]->op == GGML_OP_GET_ROWS && गाठ->src[1] != nullptr &&
                   गाठ->src[1]->op == GGML_OP_VIEW && गाठ->src[1]->view_src != nullptr &&
                   is_recurrent_cache(गाठ->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = गाठ->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } अन्यथा जर (गाठ->src[1] != nullptr && गाठ->src[1]->op == GGML_OP_VIEW && गाठ->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            स्थिरांक ggml_tensor * kv_buf = गाठ->src[1]->view_src;
            जर (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                साठी (पूर्णांक i = 0; i < m_cgraph->n_nodes; i++) {
                    स्थिरांक ggml_tensor * n = m_cgraph->nodes[i];
                    जर (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        सुरूठेवा;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    जर ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        जर (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        थांबा;
                    }
                }
            }
        }
        थांबा;
    }
    स्थिती GGML_OP_ADD: {
        जर (is_moe_expert_sum_add(गाठ)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        थांबा;
    }
    स्थिती GGML_OP_SCALE: {
        जर (गाठ->view_src && गाठ->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = गाठ->view_src->ne[1] == 1 ? 2 : 1;
        }
        थांबा;
    }
    स्थिती GGML_OP_L2_NORM: {
        जर (std::string(गाठ->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        थांबा;
    }
    स्थिती GGML_OP_FLASH_ATTN_EXT: {
        जर (गाठ->src[1] != nullptr && गाठ->src[1]->op == GGML_OP_VIEW && गाठ->src[1]->view_src != nullptr) {
            स्थिरांक ggml_tensor * kv_buf = गाठ->src[1]->view_src;
            जर (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (गाठ->src[3] != nullptr) ? 1 : 2;
            }
        }
        थांबा;
    }
    पूर्वनिर्धारित:
        थांबा;
    }
    परत op_case;
}

std::optional<पूर्णांक> extract_layer_from_name(स्थिरांक std::string & name) {
    माप_प्रकार pos1 = name.find("_l");
    जर (pos1 == std::string::npos) {
        परत std::nullopt;
    }
    pos1 += 2;
    माप_प्रकार pos2 = name.find(' ', pos1);
    जर (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    पूर्णांक layer = std::stoi(layer_str);
    परत layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
स्थिर पूर्णांक get_swa_window_from_mask(स्थिरांक ggml_tensor * mask) {
    जर (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        परत -1;
    }
    जर (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        परत -1;
    }

    स्थिरांक int64_t n_kv = mask->ne[0];
    स्थिरांक int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    साठी (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        साठी (int64_t c = 0; c < n_kv; c++) {
            स्थिरांक माप_प्रकार i = (माप_प्रकार) r * n_kv + c;
            स्थिरांक दशांश v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((स्थिरांक ggml_fp16_t *) mask->data)[i]) :
                                                          ((स्थिरांक दशांश *) mask->data)[i];
            जर (v > -अनन्त) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    परत window > 0 ? (पूर्णांक) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, तार्किक is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    स्वतः get_attention_pattern_case = [](स्थिरांक ggml_tensor * गाठ) -> पूर्णांक {
        जर (गाठ == nullptr) {
            परत -1;
        }

        निवड (गाठ->op) {
        स्थिती GGML_OP_FLASH_ATTN_EXT:
            जर (गाठ->src[0] == nullptr || गाठ->src[1] == nullptr) {
                परत -1;
            }
            निवड (गाठ->src[1]->op) {
            स्थिती GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                जर (गाठ->src[3] != nullptr && गाठ->src[1]->src[0] != nullptr &&
                    गाठ->src[1]->src[0]->op == GGML_OP_VIEW) {
                    परत 0;
                }
                थांबा;
            स्थिती GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                जर (गाठ->src[3] != nullptr && गाठ->src[1]->src[0] != nullptr &&
                    गाठ->src[1]->src[0]->op == GGML_OP_PERMUTE && गाठ->src[1]->src[0]->src[0] != nullptr &&
                    गाठ->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    परत 1;
                }
                थांबा;
            स्थिती GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                जर (गाठ->src[1]->view_src != nullptr) {
                    जर (गाठ->src[3] != nullptr) {
                        परत 4;  // decoder self-attention
                    }
                    परत 5;      // cross-attention or encoder self-attention
                }
                थांबा;
            पूर्वनिर्धारित:
                थांबा;
            }
            थांबा;
        स्थिती GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            जर (गाठ->src[0] != nullptr && गाठ->src[1] != nullptr && गाठ->src[0]->op == GGML_OP_MUL_MAT &&
                गाठ->src[0]->src[0] != nullptr && गाठ->src[0]->src[1] != nullptr &&
                गाठ->src[0]->src[0]->op == GGML_OP_PERMUTE && गाठ->src[0]->src[0]->src[0] != nullptr &&
                गाठ->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                परत 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            जर (गाठ->src[0]->op == GGML_OP_ADD && गाठ->src[0]->src[0] != nullptr &&
                गाठ->src[0]->src[0]->op == GGML_OP_MUL_MAT && गाठ->src[0]->src[0]->src[0] != nullptr &&
                गाठ->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                परत 3;
            }
            थांबा;
        पूर्वनिर्धारित:
            थांबा;
        }

        परत -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    स्वतः get_attention_op_mask = [&get_attention_pattern_case](स्थिरांक ggml_tensor * गाठ) -> स्थिरांक ggml_tensor * {
        निवड (get_attention_pattern_case(गाठ)) {
        स्थिती 0:
        स्थिती 1:
            परत गाठ->src[3];
        स्थिती 2:
        स्थिती 3:
            परत गाठ->src[1];
        पूर्वनिर्धारित:
            परत nullptr;
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
        std::map<पूर्णांक, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<पूर्णांक, स्थिरांक ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
            स्थिरांक ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            जर (mask == nullptr) {
                सुरूठेवा;
            }
            स्थिरांक ggml_tensor * cache_k_permute = nullptr;
            निवड (get_attention_pattern_case(cgraph->nodes[i])) {
            स्थिती 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     थांबा;
            स्थिती 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             थांबा;
            स्थिती 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             थांबा;
            पूर्वनिर्धारित: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     थांबा;
            }
            स्थिरांक ggml_tensor * cache_k_view = cache_k_permute->src[0];
            जर (cache_k_view->op != GGML_OP_VIEW) {
                सुरूठेवा;
            }
            स्थिरांक ggml_tensor * leaf = cache_k_view->src[0];
            स्वतः layer = extract_layer_from_name(leaf->name);
            जर (!layer.has_value()) {
                सुरूठेवा;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        साठी (स्थिरांक स्वतः & [layer, extent] : layer_extent) {
            जर (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                जर (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        जर (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            साठी (स्थिरांक स्वतः & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (दीर्घ) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    तार्किक rope_seen = असत्य;
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * गाठ = cgraph->nodes[i];
        स्थिरांक पूर्णांक attention_pattern_case = get_attention_pattern_case(गाठ);
        जर (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            निवड (attention_pattern_case) {
            स्थिती 0:
                cache_k_permute = गाठ->src[1];
                mask = गाठ->src[3];
                थांबा;
            स्थिती 1:
                cache_k_permute = गाठ->src[1]->src[0];
                mask = गाठ->src[3];
                थांबा;
            स्थिती 2:
                cache_k_permute = गाठ->src[0]->src[0];
                mask = गाठ->src[1];
                थांबा;
            स्थिती 3:
                cache_k_permute = गाठ->src[0]->src[0]->src[0];
                mask = गाठ->src[1];
                थांबा;
            स्थिती 4:
            स्थिती 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                स्वतः * cache_k_view = गाठ->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = गाठ->src[0]->ne[1];
                जर (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } अन्यथा {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                सुरूठेवा;
            }
            पूर्वनिर्धारित:
                थांबा;
            }

            प्रतिज्ञा(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.आदान_len = गाठ->src[0]->ne[1];
            compute_params.token_len_per_seq = गाठ->src[0]->ne[1];

            स्वतः * cache_k_view = cache_k_permute->src[0];
            जर (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                सुरूठेवा;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            पूर्णांक layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            स्थिरांक तार्किक layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            जर (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } अन्यथा {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            स्वतः seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            माप_प्रकार offset;
            स_नकल(&offset, cache_k_view->op_params, आकार(माप_प्रकार));
            compute_params.seq_active_start = offset / seq_size;

            जर (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } अन्यथा {
                compute_params.attention_size = mask->ne[0];
            }
            जर (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        जर (गाठ->op == GGML_OP_MUL_MAT && गाठ->src[0]->op == GGML_OP_PERMUTE &&
            गाठ->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(गाठ->src[0]->view_src, गाठ->view_src)) {
            जर (गाठ->src[1]->op == GGML_OP_PERMUTE && गाठ->src[1]->src[0]->op == GGML_OP_VIEW &&
                गाठ->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = गाठ->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        जर (गाठ->op == GGML_OP_TRANSPOSE && गाठ->src[0]->op == GGML_OP_PERMUTE &&
            गाठ->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = गाठ->ne[0];
            जर (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        जर (गाठ->op == GGML_OP_ROPE) {
            जर (compute_params.token_len_per_seq == -1 && गाठ->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(गाठ->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(आकार(model_params.rope_params) == आकार(int32_t) * 16, "rope_params size");
            जर (!rope_seen) {
                स_नकल(model_params.rope_params, गाठ->op_params, आकार(int32_t) * 16);
                rope_seen = सत्य;
            } अन्यथा जर (स_भेद(model_params.rope_params, गाठ->op_params, आकार(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = सत्य;
            }
        }
        जर (गाठ->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = गाठ->src[0]->ne[0];
        }
        जर (गाठ->op == GGML_OP_SCALE && गाठ->view_src != nullptr && is_recurrent_cache(गाठ->view_src)) {
            जर (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = गाठ->view_src->ne[1];
            } अन्यथा {
                GGML_ASSERT(model_params.n_rs_slots == गाठ->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(गाठ) / गाठ->view_src->ne[0];
            compute_params.cache_rs_reset_idx = गाठ->src[0]->view_offs / गाठ->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        जर (गाठ->op == GGML_OP_CPY && गाठ->view_src != nullptr && is_recurrent_cache(गाठ->view_src) &&
            गाठ->src[1] != nullptr && गाठ->src[1]->op == GGML_OP_VIEW && गाठ->src[1]->view_src == गाठ->view_src) {
            स्थिरांक तार्किक is_conv = is_conv_state_writeback(गाठ);
            स्थिरांक तार्किक is_gdn = गाठ->src[0]->op == GGML_OP_VIEW && गाठ->src[0]->src[0] != nullptr &&
                                गाठ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            स्थिरांक तार्किक is_extra = गाठ->src[0]->op == GGML_OP_GET_ROWS;
            स्थिरांक तार्किक is_gdn_rollback = is_gdn && is_same_shape(गाठ->src[0], गाठ->src[1]);

            स्थिरांक ggml_tensor * dest_view = गाठ->src[1];
            स्थिरांक ggml_tensor * cache = गाठ->view_src;
            स्थिरांक माप_प्रकार row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            जर (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = सत्य;
            }
            जर (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(गाठ)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (पूर्णांक) (dest_view->view_offs / row_bytes);
                जर (is_conv) {
                    writeback.src_begin = (पूर्णांक) (गाठ->src[0]->view_offs / गाठ->src[0]->view_src->nb[0]);
                } अन्यथा जर (is_gdn_rollback) {
                    writeback.src_begin = (पूर्णांक) (गाठ->src[0]->view_offs / गाठ->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, गाठ)] = writeback;
            }
            जर ((is_conv || is_gdn) && !is_full_single_slot_writeback(गाठ)) {
                compute_params.s_copy_active_slot_len = (पूर्णांक) dest_view->ne[1];
            }
        }
    }
    जर (model_params.n_heads_kv == -1) {
        साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
            स्थिरांक स्वतः * गाठ = cgraph->nodes[i];
            स्थिरांक ggml_tensor * mask = nullptr;
            जर (गाठ->op == GGML_OP_SOFT_MAX) {
                mask = गाठ->src[1];
            } अन्यथा जर (गाठ->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = गाठ->src[3];
            } अन्यथा {
                सुरूठेवा;
            }
            जर (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_आदान) ||
                गाठ->src[0] == nullptr) {
                सुरूठेवा;
            }
            model_params.is_cacheless_attn = सत्य;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.आदान_len = गाठ->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.आदान_len;
            थांबा;
        }
    }

    स्वतः * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    जर (model_params.is_cacheless_attn) {
        साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
            स्थिरांक स्वतः * गाठ = cgraph->nodes[i];
            जर (गाठ->op == GGML_OP_GET_ROWS && is_output_idx(गाठ->src[1], गाठ)) {
                compute_params.output_len = गाठ->src[1]->ne[0];
                थांबा;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    जर (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    परत {model_params, compute_params};
}

शून्य GgmlOvDecoder::validate_cgraph() स्थिरांक {
    जर (m_model_params.n_seq > 1 && m_is_static == सत्य) {
        फेका std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_आदान_shape(स्थिरांक ggml_tensor * op,
                                                      स्थिरांक ggml_tensor * आदान,
                                                      पूर्णांक dynamic_dim_index) स्थिरांक {
    जर (m_naive) {
        परत आदान != nullptr ? ov::PartialShape{get_shape(आदान)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape आदान_shape;

    जर (is_inp_tok(आदान, op) || is_inp_pos(आदान, op)) {
        // tokens or positions
        पूर्णांक len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        जर (m_is_static && is_inp_pos(आदान, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        आदान_shape = ov::PartialShape{1, 1, 1, len};

    } अन्यथा जर (is_output_idx(आदान, op)) {
        // output index
        आदान_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } अन्यथा जर (is_inp_mean(आदान, op)) {
        आदान_shape = m_is_static ? ov::PartialShape{1, 1, आदान->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } अन्यथा जर (is_inp_scale_rows(आदान, op)) {
        आदान_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } अन्यथा जर (is_inp_mask(आदान, op)) {
        // mask
        जर (m_is_static) {
            आदान_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } अन्यथा जर (m_is_stateful) {
            आदान_shape = ov::PartialShape{1, 1, -1, -1};
        } अन्यथा {
            आदान_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } अन्यथा जर (is_recurrent_cache(आदान)) {
        आदान_shape = ov::PartialShape{get_shape(आदान)};
        जर (!m_is_static && !m_is_stateful && आदान->ne[1] > 1) {
            आदान_shape[2] = -1;
        }

    } अन्यथा जर (is_kvcache(आदान, op)) {
        // kvcache
        आदान_shape = ov::PartialShape{get_shape(आदान)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        स्थिरांक तार्किक is_flat_kv = (आदान->ne[1] == 1 && आदान->ne[2] == 1 && आदान->ne[3] == 1);
        जर (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            आदान_shape[2] = -1;
        }
        जर (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            प्रतिज्ञा(आदान_shape.size() == 4 && आदान_shape[0] == 1 && आदान_shape[1] == 1 &&
                   आदान_shape[2].is_dynamic() && आदान_shape[3].is_static());
            स्थिरांक पूर्णांक n_heads_kv = get_n_heads_kv_for_tensor(आदान);
            प्रतिज्ञा(n_heads_kv > 0 && आदान_shape[3].get_length() % n_heads_kv == 0);
            स्थिरांक int64_t combined_dim = आदान_shape[3].get_length();  // n_heads_kv * head_size
            स्थिरांक int64_t head_size = combined_dim / n_heads_kv;
            आदान_shape = {आदान_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } अन्यथा जर (is_kv_idx(आदान, op)) {
        // kv update index
        पूर्णांक len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        आदान_shape = ov::PartialShape{1, 1, 1, len};

    } अन्यथा जर (is_inp_s_copy(आदान, op) || is_s_copy_leaf(आदान)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        आदान_shape = m_is_static ? ov::PartialShape{get_shape(आदान)} : ov::PartialShape{1, 1, 1, -1};

    } अन्यथा {
        आदान_shape = ov::PartialShape{get_shape(आदान)};
    }
    जर (dynamic_dim_index != -1 && m_model_is_splitted) {
        आदान_shape[3 - dynamic_dim_index] = -1;
    }
    जर (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_आदान && op->src[1] == आदान) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        जर (m_is_static) {
            स्थिरांक int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            आदान_shape[2] = seq_active;
            आदान_shape[3] = seq_active;
        } अन्यथा {
            आदान_shape[2] = -1;
            आदान_shape[3] = -1;
        }
    }
    परत आदान_shape;
}

तार्किक GgmlOvDecoder::is_s_copy_leaf(स्थिरांक ggml_tensor * tensor) स्थिरांक {
    जर (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        परत असत्य;
    }
    साठी (पूर्णांक i = 0; i < m_cgraph->n_nodes; i++) {
        स्थिरांक ggml_tensor * गाठ = m_cgraph->nodes[i];
        जर (गाठ->op != GGML_OP_GET_ROWS || गाठ->src[0] == nullptr || गाठ->src[1] == nullptr) {
            सुरूठेवा;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        स्थिरांक ggml_tensor * idx = गाठ->src[1];
        पर्यंत (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        जर (idx != tensor) {
            सुरूठेवा;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        स्थिरांक ggml_tensor * data = गाठ->src[0];
        पर्यंत (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        जर (data != nullptr && is_recurrent_cache(data)) {
            परत सत्य;
        }
    }
    परत असत्य;
}

शून्य GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    स्वतः create_1d_आदान = [हे](स्थिरांक std::string & name, int64_t value, तार्किक force_parameter = असत्य) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    जर (m_compute_params.attention_size != -1) {
        create_1d_आदान("attention_size", m_compute_params.attention_size);
    }
    जर (m_compute_params.attention_size_static != -1) {
        create_1d_आदान("attention_size_static", m_compute_params.attention_size_static);
    }
    जर (m_compute_params.attention_size_swa != -1) {
        create_1d_आदान("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    जर (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_आदान("swa_window", m_compute_params.swa_window);
    }
    create_1d_आदान("n_seq_active", m_compute_params.n_seq_active);
    create_1d_आदान("seq_active_start", m_compute_params.seq_active_start);
    create_1d_आदान("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    जर (m_compute_params.token_len_per_seq != -1) {
        create_1d_आदान("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    जर (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_आदान("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/सत्य);
        create_1d_आदान("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/सत्य);
    }

    जर (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_आदान("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        जर (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_आदान("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/सत्य);
        }
    }

    साठी (स्थिरांक स्वतः & [गाठ_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_आदान("rs_slot_begin_" + गाठ_name, writeback.slot_begin);
        जर (!m_is_static && writeback.src_begin >= 0) {
            create_1d_आदान("rs_src_begin_" + गाठ_name, writeback.src_begin);
        }
    }
}

तार्किक GgmlOvDecoder::गाठ_is_used_as_src(स्थिरांक पूर्णांक गाठ_idx) {
    ggml_tensor * गाठ = m_cgraph->nodes[गाठ_idx];
    साठी (पूर्णांक i = गाठ_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_गाठ = m_cgraph->nodes[i];
        साठी (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            जर (other_गाठ->src[j] == गाठ) {
                परत सत्य;
            }
        }
    }
    परत असत्य;
}

शून्य GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    साठी (पूर्णांक i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * गाठ = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        जर (गाठ->op == GGML_OP_NONE && गाठ_is_used_as_src(i)) {
            std::string गाठ_name = get_tensor_ov_name(m_cgraph, गाठ);
            जर (m_model_weights.find(गाठ_name) == m_model_weights.end()) {
                m_inputs[गाठ_name] = गाठ;
                m_model_inputs[गाठ_name] = {get_ov_type(गाठ),
                                             get_graph_आदान_shape(गाठ, nullptr, m_गाठ_dynamic_dims[गाठ])};
            }
            सुरूठेवा;
        }
        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            स्वतः * src = गाठ->src[i];
            जर (src == nullptr) {
                सुरूठेवा;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            जर (src->flags & GGML_TENSOR_FLAG_आदान) {
                src_name = get_tensor_graph_आदान_ov_name(हे, m_cgraph, src, गाठ);
            }
            जर (m_model_weights.find(src_name) != m_model_weights.end()) {
                सुरूठेवा;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            जर (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                सुरूठेवा;
            }

            तार्किक is_intermediate_गाठ = असत्य;
            साठी (स्थिरांक स्वतः & गाठ_info : m_गाठ_info_list) {
                जर (गाठ_info.गाठ == src) {
                    is_intermediate_गाठ = सत्य;
                    थांबा;
                }
            }
            जर (is_intermediate_गाठ) {
                सुरूठेवा;
            }
            जर (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                सुरूठेवा;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            जर (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                जर (स्वतः it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            पर्यंत (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_आदान_shape(गाठ, src, m_गाठ_dynamic_dims[src])};
        }
    }
}

शून्य GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    साठी (पूर्णांक गाठ_n = 0; गाठ_n < m_cgraph->n_nodes; गाठ_n++) {
        स्वतः * cur_गाठ = m_cgraph->nodes[गाठ_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        जर (cur_गाठ->op == GGML_OP_NONE || cur_गाठ->op == GGML_OP_VIEW || cur_गाठ->op == GGML_OP_RESHAPE) {
            सुरूठेवा;
        }
        जर (::is_inplace_op(cur_गाठ) && ggml_nbytes(cur_गाठ) == 0) {
            सुरूठेवा;
        }
        स्वतः cur_गाठ_use_मोजणी = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_गाठ)];
        जर (cur_गाठ_use_मोजणी == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            जर (cur_गाठ != nullptr && ::is_inplace_op(cur_गाठ) && ggml_nbytes(cur_गाठ) > 0) {
                cur_गाठ = cur_गाठ->view_src;
            }
        } अन्यथा {
            पूर्णांक आदान_use_मोजणी = 0;
            साठी (पूर्णांक i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * गाठ = m_cgraph->nodes[i];
                साठी (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
                    जर (गाठ->src[j] != रिक्त && गाठ->src[j] == cur_गाठ) {
                        आदान_use_मोजणी++;
                    }
                }
            }
            जर (आदान_use_मोजणी == cur_गाठ_use_मोजणी) {
                cur_गाठ = nullptr;
            }
        }
        जर (cur_गाठ != nullptr) {
            std::string cur_गाठ_name = get_tensor_ov_name(m_cgraph, cur_गाठ);
            m_model_outputs[cur_गाठ_name] = cur_गाठ;
            m_model_output_names.insert(cur_गाठ_name);
        }
    }
}

स्थिरांक ggml_tensor * GgmlOvDecoder::get_tensor_used_op(स्थिरांक ggml_tensor * tensor) स्थिरांक {
    जर (tensor == nullptr) {
        परत nullptr;
    }
    साठी (पूर्णांक i = 0; i < m_cgraph->n_nodes; i++) {
        स्थिरांक स्वतः * गाठ = m_cgraph->nodes[i];
        साठी (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            जर (गाठ->src[j] == tensor) {
                परत गाठ;
            }
        }
    }
    परत nullptr;
}

स्थिरांक ggml_tensor * GgmlOvDecoder::get_tensor_from_name(स्थिरांक std::string & name) स्थिरांक {
    साठी (पूर्णांक i = 0; i < m_cgraph->n_nodes; i++) {
        स्थिरांक स्वतः * गाठ = m_cgraph->nodes[i];
        साठी (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            स्थिरांक स्वतः * src = गाठ->src[j];
            जर (src == nullptr) {
                थांबा;
            }
            जर (get_tensor_ov_name(m_cgraph, src) == name) {
                परत src;
            }
        }
    }
    परत nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() स्थिरांक {
    std::map<std::string, std::string> kv_param_res_names;
    साठी (स्थिरांक स्वतः & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    परत kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
स्थिर तार्किक is_mul_mat_id_expert_weight(स्थिरांक ggml_tensor * गाठ, पूर्णांक src_index) {
    परत गाठ->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, तार्किक naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    स्वतः * nodes = cgraph->nodes;
    स्वतः n_nodes = cgraph->n_nodes;
    साठी (पूर्णांक गाठ_i = 0; गाठ_i < n_nodes; गाठ_i++) {
        स्वतः * गाठ = nodes[गाठ_i];
        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            स्वतः * src = गाठ->src[i];
            जर (src == nullptr) {
                सुरूठेवा;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            जर (is_rope_freqs_weight(base, गाठ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            जर (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(गाठ, i)) {
                जर (model_weights.find(src_name) == model_weights.end()) {
                    स्वतः weight_गाठ = create_weight_गाठ(base, naive);
                    weight_गाठ->set_friendly_name(src_name);
                    model_weights[src_name] = weight_गाठ;
                }
            }
        }
    }
    परत model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
स्थिर std::mutex g_nonov_weight_cache_mutex;
स्थिर std::unordered_map<स्थिरांक शून्य *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    साठी (पूर्णांक गाठ_i = 0; गाठ_i < cgraph->n_nodes; गाठ_i++) {
        स्वतः * गाठ = cgraph->nodes[गाठ_i];
        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            स्वतः * src = गाठ->src[i];
            जर (src == nullptr) {
                सुरूठेवा;
            }
            स्थिरांक ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            जर (is_rope_freqs_weight(base, गाठ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            जर (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    परत names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_गाठ(ggml_tensor * tensor, तार्किक naive) {
    स्थिरांक तार्किक is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    जर (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        स्वतः * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        जर (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            स्वतः * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            जर (weight_extra->weight_गाठ) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                परत weight_extra->weight_गाठ;
            }
        } अन्यथा जर (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            स्वतः * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            जर (quant_extra->weight_गाठ) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                परत quant_extra->weight_गाठ;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    जर (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        स्वतः weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        स्वतः weight_गाठ = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_गाठ->set_friendly_name(tensor->name);
        परत weight_गाठ;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    स्थिरांक तार्किक cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    जर (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        स्वतः it = g_nonov_weight_cache.find(tensor->data);
        जर (it != g_nonov_weight_cache.end()) {
            परत it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    स्थिर स्थिरांक std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    जर (weight_types.find(tensor->type) == weight_types.end()) {
        फेका std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    जर (ggml_is_quantized(tensor->type)) {
        स्वतः use_bias = naive;
        जर (is_ov_buffer) {
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

    ov_weight.weight_गाठ->set_friendly_name(tensor->name);
    जर (!is_ov_buffer) {
        जर (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            स्वतः [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_गाठ);
            परत it->second;
        }
        परत ov_weight.weight_गाठ;
    }

    ggml_openvino_extra_base * extra;
    जर (ov_weight.is_quantized()) {
        extra = नवीन ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_गाठ);
    } अन्यथा {
        extra = नवीन ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_गाठ);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    परत ov_weight.weight_गाठ;
}

शून्य GgmlOvDecoder::dump_cgraph(स्थिरांक ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    जर (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        परत;
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
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * गाठ = cgraph->nodes[i];

        // Get buffer type name
        स्थिरांक अक्षर * buf_name = "none";
        ggml_backend_buffer_t buf = गाठ->view_src ? गाठ->view_src->buffer : गाठ->buffer;
        जर (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << गाठ->ne[0] << ", "
             << std::setw(5) << गाठ->ne[1] << ", "
             << std::setw(5) << गाठ->ne[2] << ", "
             << std::setw(5) << गाठ->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(गाठ->op) << std::right << " "
             << std::left << std::setw(45) << गाठ->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << गाठ->nb[0] << ", "
             << std::setw(5) << गाठ->nb[1] << ", "
             << std::setw(5) << गाठ->nb[2] << ", "
             << std::setw(5) << गाठ->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            जर (स्वतः* src = गाठ->src[i]) {
                // Get buffer type name for source
                स्थिरांक अक्षर * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                जर (src_buf) {
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
    साठी (पूर्णांक i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * गाठ = cgraph->leafs[i];

        // Get buffer type name for leaf
        स्थिरांक अक्षर * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = गाठ->view_src ? गाठ->view_src->buffer : गाठ->buffer;
        जर (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << गाठ->ne[0] << ", "
             << std::setw(5) << गाठ->ne[1] << "] "
             << std::setw(8) << ggml_op_name(गाठ->op) << " "
             << std::setw(16) << ggml_get_name(गाठ)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.बंद();
}

शून्य print_tensor_address_map(स्थिरांक ggml_cgraph * cgraph) {
    std::map<शून्य *, std::vector<std::string>> address_map;
    साठी (पूर्णांक गाठ_n = 0; गाठ_n < cgraph->n_nodes; गाठ_n++) {
        स्वतः * गाठ = cgraph->nodes[गाठ_n];
        जर (गाठ->data) {
            स्वतः it = address_map.find(गाठ->data);
            जर (it == address_map.end()) {
                address_map[गाठ->data] = std::vector<std::string>();
            }
            address_map[गाठ->data].push_back(गाठ->name);
        }
    }
    साठी (स्थिरांक स्वतः & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        साठी (स्थिरांक स्वतः & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(स्थिरांक ggml_tensor * tensor) {
    std::vector<माप_प्रकार> shape;
    साठी (पूर्णांक i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<माप_प्रकार>(tensor->ne[i]));
    }
    परत shape;
}

std::vector<माप_प्रकार> GgmlOvDecoder::get_stride(स्थिरांक ggml_tensor * tensor) {
    std::vector<माप_प्रकार> stride;
    साठी (पूर्णांक i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<माप_प्रकार>(tensor->nb[i]));
    }
    परत stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(स्थिरांक ggml_tensor * tensor) {
    निवड (tensor->type) {
    स्थिती GGML_TYPE_F64:
        परत ov::element::f64;
    स्थिती GGML_TYPE_F32:
        परत ov::element::f32;
    स्थिती GGML_TYPE_F16:
        परत ov::element::f16;
    स्थिती GGML_TYPE_BF16:
        परत ov::element::bf16;
    स्थिती GGML_TYPE_I8:
        परत ov::element::i8;
    स्थिती GGML_TYPE_I16:
        परत ov::element::i16;
    स्थिती GGML_TYPE_I32:
        परत ov::element::i32;
    स्थिती GGML_TYPE_I64:
        परत ov::element::i64;
    पूर्वनिर्धारित:
        परत ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_आदान_shape(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक {
    परत ov::PartialShape(get_shape(m_गाठ_info_list[गाठ_idx].गाठ_inputs.at(name)));
}

std::vector<माप_प्रकार> GgmlOvDecoder::get_आदान_stride(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक {
    परत get_stride(m_गाठ_info_list[गाठ_idx].गाठ_inputs.at(name));
}

माप_प्रकार GgmlOvDecoder::get_view_आदान_size(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        परत it->second.size();
    }
    परत 0;
}

माप_प्रकार GgmlOvDecoder::get_view_आदान_offset(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        जर (view_index < it->second.size()) {
            परत it->second[view_index].second->view_offs;
        }
    }
    परत 0;
}

माप_प्रकार GgmlOvDecoder::get_view_आदान_src_offset(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        जर (view_index < it->second.size()) {
            स्वतः * view_tensor = it->second[view_index].second;
            जर (view_tensor && view_tensor->src[0]) {
                परत view_tensor->src[0]->view_offs;
            }
        }
    }
    परत 0;
}

std::vector<माप_प्रकार> GgmlOvDecoder::get_view_आदान_stride(पूर्णांक गाठ_idx,
                                                         स्थिरांक std::string & name,
                                                         माप_प्रकार view_index) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        जर (view_index < it->second.size()) {
            परत get_stride(it->second[view_index].second);
        }
    }
    परत {};
}

std::vector<माप_प्रकार> GgmlOvDecoder::get_view_आदान_src_stride(पूर्णांक गाठ_idx,
                                                             स्थिरांक std::string & name,
                                                             माप_प्रकार view_index) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        जर (view_index < it->second.size()) {
            स्वतः * view_tensor = it->second[view_index].second;
            जर (view_tensor && view_tensor->src[0]) {
                परत get_stride(view_tensor->src[0]);
            }
        }
    }
    परत {};
}

ov::Shape GgmlOvDecoder::get_view_आदान_ggml_shape(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        जर (view_index < it->second.size()) {
            परत get_shape(it->second[view_index].second);
        }
    }
    परत {};
}

ov::Shape GgmlOvDecoder::get_view_आदान_src_ggml_shape(पूर्णांक गाठ_idx,
                                                       स्थिरांक std::string & name,
                                                       माप_प्रकार view_index) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        जर (view_index < it->second.size()) {
            स्वतः * view_tensor = it->second[view_index].second;
            जर (view_tensor && view_tensor->src[0]) {
                परत get_shape(view_tensor->src[0]);
            }
        }
    }
    परत {};
}

ov::PartialShape GgmlOvDecoder::get_view_आदान_ov_shape(पूर्णांक गाठ_idx,
                                                        स्थिरांक std::string & name,
                                                        माप_प्रकार view_index) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        जर (view_index < it->second.size()) {
            स्वतः * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            स्वतः dynamic_it = m_गाठ_dynamic_dims.find(tensor);
            जर (dynamic_it != m_गाठ_dynamic_dims.end() && dynamic_it->second != -1) {
                पूर्णांक dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            परत shape;
        }
    }
    परत {};
}

ov::PartialShape GgmlOvDecoder::get_view_आदान_src_ov_shape(पूर्णांक गाठ_idx,
                                                            स्थिरांक std::string & name,
                                                            माप_प्रकार view_index) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        जर (view_index < it->second.size()) {
            स्वतः * view_tensor = it->second[view_index].second;
            जर (view_tensor && view_tensor->src[0]) {
                स्वतः * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                स्वतः dynamic_it = m_गाठ_dynamic_dims.find(src_tensor);
                जर (dynamic_it != m_गाठ_dynamic_dims.end() && dynamic_it->second != -1) {
                    पूर्णांक dynamic_dim_index = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
                }

                परत shape;
            }
        }
    }
    परत {};
}

std::string GgmlOvDecoder::get_view_आदान_name(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        जर (view_index < it->second.size()) {
            परत it->second[view_index].first;
        }
    }
    परत "";
}

std::string GgmlOvDecoder::get_view_आदान_src_name(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक {
    स्वतः it = m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.find(name);
    जर (it != m_गाठ_info_list[गाठ_idx].गाठ_inputs_views.end()) {
        जर (view_index < it->second.size()) {
            स्वतः * view_tensor = it->second[view_index].second;
            जर (view_tensor && view_tensor->src[0]) {
                परत get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    परत "";
}

ov::element::Type GgmlOvDecoder::get_आदान_type(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक {
    परत get_ov_type(m_गाठ_info_list[गाठ_idx].गाठ_inputs.at(name));
}

माप_प्रकार GgmlOvDecoder::get_आदान_size() स्थिरांक {
    परत m_model_inputs.size();
}

माप_प्रकार GgmlOvDecoder::get_आदान_size(पूर्णांक गाठ_idx) स्थिरांक {
    परत m_गाठ_info_list[गाठ_idx].गाठ_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_आदान_names(पूर्णांक गाठ_idx) स्थिरांक {
    परत m_गाठ_info_list[गाठ_idx].गाठ_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(पूर्णांक गाठ_idx) स्थिरांक {
    स्वतः * ggml_tensor = m_गाठ_info_list[गाठ_idx].गाठ;
    परत ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(स्थिरांक पूर्णांक गाठ_idx) स्थिरांक {
    परत get_ov_type(m_गाठ_info_list[गाठ_idx].गाठ);
}

std::vector<माप_प्रकार> GgmlOvDecoder::get_output_stride(पूर्णांक गाठ_idx) स्थिरांक {
    स्वतः * ggml_tensor = m_गाठ_info_list[गाठ_idx].गाठ;
    परत get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(पूर्णांक गाठ_idx) स्थिरांक {
    स्वतः * गाठ = m_गाठ_info_list[गाठ_idx].गाठ;
    जर (गाठ->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        साठी (पूर्णांक i = गाठ_idx + 1; i < m_cgraph->n_nodes; i++) {
            स्वतः * consumer = m_cgraph->nodes[i];
            जर (consumer->op != GGML_OP_VIEW || consumer->src[0] != गाठ) {
                सुरूठेवा;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            स्वतः & name = consumer->view_offs == 0 ? attn_name : state_name;
            जर (!name.empty()) {
                परत {m_गाठ_info_list[गाठ_idx].गाठ_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        जर (!attn_name.empty() && !state_name.empty()) {
            परत {attn_name, state_name};
        }
    }
    परत {m_गाठ_info_list[गाठ_idx].गाठ_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(पूर्णांक गाठ_idx) स्थिरांक {
    स्वतः * गाठ = m_गाठ_info_list[गाठ_idx].गाठ;
    जर (!::is_inplace_op(गाठ) || गाठ->view_src == nullptr || ggml_nbytes(गाठ) == 0) {
        परत "";
    }
    स्थिरांक पूर्णांक op_case = m_गाठ_info_list[गाठ_idx].गाठ_op_case;
    जर (गाठ->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        परत "";
    }
    परत get_tensor_ov_name(m_cgraph, गाठ->view_src);
}

तार्किक GgmlOvDecoder::is_view_like_alias_of(पूर्णांक गाठ_idx, स्थिरांक std::string & view_src_name) स्थिरांक {
    स्वतः * गाठ = m_गाठ_info_list[गाठ_idx].गाठ;
    जर (गाठ->view_src == nullptr || get_tensor_ov_name(m_cgraph, गाठ->view_src) != view_src_name) {
        परत असत्य;
    }
    परत गाठ->op == GGML_OP_RESHAPE || गाठ->op == GGML_OP_VIEW;
}

स्थिरांक std::string & GgmlOvDecoder::get_op_name() स्थिरांक {
    स्थिर स्थिरांक std::string unknown_name = "UNKNOWN_OP_NAME";
    परत unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(पूर्णांक गाठ_idx) स्थिरांक {
    स्वतः it = m_गाठ_dynamic_dims.find(m_गाठ_info_list[गाठ_idx].गाठ);
    जर (it == m_गाठ_dynamic_dims.end()) {
        परत -1;
    }
    परत it->second;
}

स्थिरांक std::string & GgmlOvDecoder::get_op_name(पूर्णांक गाठ_idx) स्थिरांक {
    परत m_गाठ_info_list[गाठ_idx].गाठ_name;
}

int32_t * GgmlOvDecoder::get_आदान_op_params(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक {
    परत m_गाठ_info_list[गाठ_idx].गाठ_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(पूर्णांक गाठ_idx) स्थिरांक {
    परत m_गाठ_info_list[गाठ_idx].गाठ->op_params;
}

माप_प्रकार GgmlOvDecoder::get_output_op_offset(पूर्णांक गाठ_idx) स्थिरांक {
    परत m_गाठ_info_list[गाठ_idx].गाठ->view_offs;
}

शून्य GgmlOvDecoder::visit_subgraph(std::function<शून्य(std::shared_ptr<GgmlDecoder>, पूर्णांक गाठ_idx)> गाठ_visitor) स्थिरांक {
    साठी (पूर्णांक गाठ_idx = 0; गाठ_idx < m_cgraph->n_nodes; गाठ_idx++) {
        जर (m_cgraph->nodes[गाठ_idx]->op == GGML_OP_NONE) {
            सुरूठेवा;
        }
        गाठ_visitor(std::make_shared<GgmlOvDecoder>(*हे), गाठ_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(स्थिरांक ggml_tensor * गाठ) {
    निवड (गाठ->op) {
    स्थिती GGML_OP_UNARY:
        परत std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(गाठ));
    स्थिती GGML_OP_GLU:
        परत std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(गाठ));
    पूर्वनिर्धारित:
        परत std::string("GGML_OP_") + ggml_op_name(गाठ->op);
    }
}

स्थिरांक std::string & GgmlOvDecoder::get_op_type(पूर्णांक गाठ_idx) स्थिरांक {
    परत m_गाठ_info_list[गाठ_idx].गाठ_op_type;
}

स्थिरांक std::string & GgmlOvDecoder::get_op_type() स्थिरांक {
    स्थिर स्थिरांक std::string unknown_op = "UNKNOWN_GGML_OP";
    परत unknown_op;
}

शून्य GgmlOvDecoder::compute_गाठ_dynamic_dims() {
    स्वतः visit_गाठ = [&](स्वतः && self, ggml_tensor * गाठ) -> शून्य {
        जर (!गाठ) {
            परत;
        }

        जर (गाठ->op == GGML_OP_CPY) {
            m_गाठ_dynamic_dims[गाठ] = -1;
        }

        जर (m_गाठ_dynamic_dims.count(गाठ)) {
            परत;
        }
        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = गाठ->src[i];
            जर (src == nullptr) {
                सुरूठेवा;
            }
            संरचना ggml_tensor * मूळ_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            जर (मूळ_src) {
                जर (is_inp_tok(मूळ_src, गाठ) || is_inp_pos(मूळ_src, गाठ) || is_output_idx(मूळ_src, गाठ)) {
                    m_गाठ_dynamic_dims[मूळ_src] = 0;
                    m_गाठ_dynamic_dims[src] = m_गाठ_dynamic_dims[मूळ_src];
                    सुरूठेवा;
                }
                self(self, मूळ_src);
                m_गाठ_dynamic_dims[src] = m_गाठ_dynamic_dims[मूळ_src];
            } अन्यथा {
                जर (is_inp_tok(src, गाठ) || is_inp_pos(src, गाठ) || is_output_idx(src, गाठ)) {
                    m_गाठ_dynamic_dims[src] = 0;
                    सुरूठेवा;
                }
                जर (is_inp_scale_rows(src, गाठ)) {
                    m_गाठ_dynamic_dims[src] = 1;
                    सुरूठेवा;
                }
                जर (गाठ->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_गाठ_dynamic_dims[src] = 1;
                    सुरूठेवा;
                }
                self(self, src);
            }
        }
        निवड (गाठ->op) {
        स्थिती GGML_OP_NONE:
            m_गाठ_dynamic_dims[गाठ] = -1;
            थांबा;
        स्थिती GGML_OP_GET_ROWS:
            m_गाठ_dynamic_dims[गाठ] = -1;
            जर (m_गाठ_dynamic_dims[गाठ->src[1]] != -1) {
                स्वतः dynamic_dim_idx = m_गाठ_dynamic_dims[गाठ->src[1]];
                जर (dynamic_dim_idx == 0) {
                    m_गाठ_dynamic_dims[गाठ] = 1;
                } अन्यथा {
                    स्वतः dynamic_dim_stride = गाठ->src[1]->nb[dynamic_dim_idx] / ggml_type_size(गाठ->src[1]->type) *
                                              ggml_type_size(गाठ->src[0]->type);
                    साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
                        जर (dynamic_dim_stride == गाठ->src[0]->nb[i]) {
                            m_गाठ_dynamic_dims[गाठ] = i;
                            थांबा;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            थांबा;
        स्थिती GGML_OP_MUL:
        स्थिती GGML_OP_MUL_MAT:
            m_गाठ_dynamic_dims[गाठ] = -1;
            जर (m_गाठ_dynamic_dims[गाठ->src[0]] != -1) {
                m_गाठ_dynamic_dims[गाठ] = m_गाठ_dynamic_dims[गाठ->src[0]];
            }
            जर (m_गाठ_dynamic_dims[गाठ->src[1]] != -1) {
                m_गाठ_dynamic_dims[गाठ] = m_गाठ_dynamic_dims[गाठ->src[1]];
            }
            थांबा;
        स्थिती GGML_OP_PERMUTE:
            m_गाठ_dynamic_dims[गाठ] = -1;
            जर (m_गाठ_dynamic_dims[गाठ->src[0]] != -1) {
                स्वतः dynamic_dim_idx = m_गाठ_dynamic_dims[गाठ->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
                    जर (गाठ->op_params[i] == dynamic_dim_idx) {
                        m_गाठ_dynamic_dims[गाठ] = i;
                        थांबा;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            थांबा;
        स्थिती GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_गाठ_dynamic_dims[गाठ] = -1;
            जर (m_गाठ_dynamic_dims[गाठ->src[0]] != -1) {
                जर (गाठ->src[0]->op == GGML_OP_NONE) {
                    m_गाठ_dynamic_dims[गाठ] = m_गाठ_dynamic_dims[गाठ->src[0]];
                    थांबा;
                }
                स्वतः dynamic_dim_idx = m_गाठ_dynamic_dims[गाठ->src[0]];
                स्वतः dynamic_dim_मूल्य = गाठ->src[0]->ne[dynamic_dim_idx];
                स्वतः dynamic_dim_stride =
                    गाठ->src[0]->nb[dynamic_dim_idx] / ggml_type_size(गाठ->src[0]->type) * ggml_type_size(गाठ->type);
                साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
                    जर (गाठ->nb[i] == dynamic_dim_stride) {
                        m_गाठ_dynamic_dims[गाठ] = i;
                        थांबा;
                    }
                }
                जर (m_गाठ_dynamic_dims[गाठ] != -1 && dynamic_dim_मूल्य != गाठ->ne[m_गाठ_dynamic_dims[गाठ]]) {
                    m_गाठ_dynamic_dims[गाठ] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    जर (ggml_nelements(गाठ) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      गाठ->name, गाठ->src[0]->name);
                    }
                }
            }
            थांबा;
        }
        स्थिती GGML_OP_TRANSPOSE:
        स्थिती GGML_OP_RESHAPE: {
            जर (is_same_shape(गाठ->src[0], गाठ)) {
                m_गाठ_dynamic_dims[गाठ] = m_गाठ_dynamic_dims[गाठ->src[0]];
                थांबा;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_गाठ_dynamic_dims[गाठ] = -1;
            जर (m_गाठ_dynamic_dims[गाठ->src[0]] != -1) {
                स्वतः dynamic_dim_idx = m_गाठ_dynamic_dims[गाठ->src[0]];
                स्वतः dynamic_dim_stride = गाठ->src[0]->nb[dynamic_dim_idx];
                साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
                    जर (गाठ->nb[i] == dynamic_dim_stride && गाठ->ne[i] == गाठ->src[0]->ne[dynamic_dim_idx]) {
                        m_गाठ_dynamic_dims[गाठ] = i;
                        थांबा;
                    }
                }
                जर (m_गाठ_dynamic_dims[गाठ] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", गाठ->name);
                }
            }
            थांबा;
        }
        स्थिती GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr पूर्णांक q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_गाठ_dynamic_dims[गाठ] = -1;
            जर (m_गाठ_dynamic_dims[गाठ->src[0]] != -1) {
                स्वतः q_dynamic_dim = m_गाठ_dynamic_dims[गाठ->src[0]];
                m_गाठ_dynamic_dims[गाठ] = q_to_out[q_dynamic_dim];
            }
            थांबा;
        }
        स्थिती GGML_OP_CONT:
            m_गाठ_dynamic_dims[गाठ] = -1;
            जर (m_गाठ_dynamic_dims[गाठ->src[0]] != -1) {
                स्वतः dynamic_dim_idx = m_गाठ_dynamic_dims[गाठ->src[0]];
                जर (ggml_are_same_shape(गाठ, गाठ->src[0])) {
                    m_गाठ_dynamic_dims[गाठ] = dynamic_dim_idx;
                } अन्यथा {
                    माप_प्रकार src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(गाठ->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (गाठ->src[0]->ne[0] / ggml_blck_size(गाठ->src[0]->type));
                    साठी (पूर्णांक i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * गाठ->src[0]->ne[i - 1];
                    }

                    स्वतः dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(गाठ->src[0]->type) *
                                              ggml_type_size(गाठ->type);
                    पूर्णांक matched_dim_मोजणी = 0;
                    पूर्णांक first_matched_dim = -1;
                    साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
                        जर (गाठ->nb[i] == dynamic_dim_stride && गाठ->ne[i] == गाठ->src[0]->ne[dynamic_dim_idx]) {
                            जर (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_गाठ_dynamic_dims[गाठ] = i;
                            matched_dim_मोजणी++;
                        }
                    }
                    जर (matched_dim_मोजणी > 1 && गाठ->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_गाठ_dynamic_dims[गाठ] = first_matched_dim;
                    } अन्यथा जर (matched_dim_मोजणी != 1) {
                        m_गाठ_dynamic_dims[गाठ] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      गाठ->name, गाठ->src[0]->name);
                    }
                }
            }
            थांबा;
        स्थिती GGML_OP_CONCAT:
            साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
                जर (गाठ->src[0]->ne[i] != गाठ->ne[i]) {
                    m_गाठ_dynamic_dims[गाठ] = i;
                    थांबा;
                }
            }
            थांबा;
        स्थिती GGML_OP_SSM_CONV:
        स्थिती GGML_OP_GATED_DELTA_NET:
            m_गाठ_dynamic_dims[गाठ] = 1;
            थांबा;
        स्थिती GGML_OP_RMS_NORM:
        स्थिती GGML_OP_L2_NORM:
        स्थिती GGML_OP_NORM:
        स्थिती GGML_OP_ADD:
        स्थिती GGML_OP_SUB:
        स्थिती GGML_OP_GLU:
        स्थिती GGML_OP_ROPE:
        स्थिती GGML_OP_SCALE:
        स्थिती GGML_OP_SOFT_MAX:
        स्थिती GGML_OP_ARGSORT:
        स्थिती GGML_OP_ADD_ID:
        स्थिती GGML_OP_UNARY:
        स्थिती GGML_OP_CUMSUM:
        स्थिती GGML_OP_FILL:
        स्थिती GGML_OP_SET:
        स्थिती GGML_OP_DIAG:
        स्थिती GGML_OP_TRI:
        स्थिती GGML_OP_REPEAT:
        स्थिती GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        स्थिती GGML_OP_DIV:
        स्थिती GGML_OP_CLAMP:
        स्थिती GGML_OP_PAD:
        स्थिती GGML_OP_UPSCALE:
        स्थिती GGML_OP_SIN:
        स्थिती GGML_OP_COS:
        स्थिती GGML_OP_LOG:
        स्थिती GGML_OP_ROLL:
            m_गाठ_dynamic_dims[गाठ] = m_गाठ_dynamic_dims[गाठ->src[0]];
            थांबा;
        स्थिती GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_गाठ_dynamic_dims[गाठ] =
                (m_गाठ_dynamic_dims[गाठ->src[0]] == 0) ? -1 : m_गाठ_dynamic_dims[गाठ->src[0]];
            थांबा;
        स्थिती GGML_OP_MUL_MAT_ID:
        स्थिती GGML_OP_SOLVE_TRI:
            m_गाठ_dynamic_dims[गाठ] = m_गाठ_dynamic_dims[गाठ->src[1]];
            थांबा;
        स्थिती GGML_OP_CPY:
        स्थिती GGML_OP_SET_ROWS:
        स्थिती GGML_OP_SUM:
        स्थिती GGML_OP_MEAN:
            m_गाठ_dynamic_dims[गाठ] = -1;
            थांबा;
        स्थिती GGML_OP_IM2COL: {
            m_गाठ_dynamic_dims[गाठ] = -1;
            जर (m_गाठ_dynamic_dims[गाठ->src[1]] != -1) {
                स्थिरांक तार्किक is_2D = गाठ->op_params[6] == 1;
                स्थिरांक पूर्णांक src_dyn = m_गाठ_dynamic_dims[गाठ->src[1]];
                जर (is_2D) {
                    जर (src_dyn == 0) {
                        m_गाठ_dynamic_dims[गाठ] = 1;  // IW -> OW
                    } अन्यथा जर (src_dyn == 1) {
                        m_गाठ_dynamic_dims[गाठ] = 2;  // IH -> OH
                    } अन्यथा जर (src_dyn == 3) {
                        m_गाठ_dynamic_dims[गाठ] = 3;  // N  -> N
                    }
                } अन्यथा {
                    जर (src_dyn == 0) {
                        m_गाठ_dynamic_dims[गाठ] = 1;  // IW -> OW
                    } अन्यथा जर (src_dyn == 2) {
                        m_गाठ_dynamic_dims[गाठ] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                जर (m_गाठ_dynamic_dims[गाठ] != -1) {
                    OPENVINO_ASSERT(गाठ->src[1]->ne[src_dyn] == गाठ->ne[m_गाठ_dynamic_dims[गाठ]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(गाठ->name) +
                                        " and its src[1]: " + std::string(गाठ->src[1]->name));
                }
            }
            थांबा;
        }
        स्थिती GGML_OP_IM2COL_3D: {
            m_गाठ_dynamic_dims[गाठ] = -1;
            जर (m_गाठ_dynamic_dims[गाठ->src[1]] != -1) {
                स्थिरांक पूर्णांक src_dyn = m_गाठ_dynamic_dims[गाठ->src[1]];
                जर (src_dyn == 0) {
                    m_गाठ_dynamic_dims[गाठ] = 1;  // IW -> OW
                } अन्यथा जर (src_dyn == 1) {
                    m_गाठ_dynamic_dims[गाठ] = 2;  // IH -> OH
                } अन्यथा जर (src_dyn == 3) {
                    m_गाठ_dynamic_dims[गाठ] = 3;  // N  -> N
                }
                जर (m_गाठ_dynamic_dims[गाठ] != -1) {
                    OPENVINO_ASSERT(गाठ->src[1]->ne[src_dyn] == गाठ->ne[m_गाठ_dynamic_dims[गाठ]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(गाठ->name) +
                                        " and its src[1]: " + std::string(गाठ->src[1]->name));
                }
            }
            थांबा;
        }
        पूर्वनिर्धारित:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(गाठ->op), गाठ->name);
            थांबा;
        }
    };

    साठी (पूर्णांक i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * गाठ = m_cgraph->nodes[i];
        visit_गाठ(visit_गाठ, गाठ);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    जर (0) {
        साठी (पूर्णांक i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * गाठ = m_cgraph->nodes[i];
            पूर्णांक dynamic_dim = m_गाठ_dynamic_dims[गाठ];
            std::cout << "[" << i << "] " << "node_name: " << गाठ->name << " op: " << ggml_op_name(गाठ->op)
                      << " shape: [";
            साठी (पूर्णांक j = 0; j < 4; j++) {
                जर (j == dynamic_dim) {
                    std::cout << "*";
                } अन्यथा {
                    std::cout << गाठ->ne[j];
                }
                जर (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            साठी (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = गाठ->src[j];
                जर (src == nullptr) {
                    सुरूठेवा;
                }
                पूर्णांक src_dynamic_dim = m_गाठ_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                साठी (पूर्णांक k = 0; k < 4; k++) {
                    जर (k == src_dynamic_dim) {
                        std::cout << "*";
                    } अन्यथा {
                        std::cout << src->ne[k];
                    }
                    जर (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
