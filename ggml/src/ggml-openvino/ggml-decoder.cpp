#அந்தர்பாவ "ggml-decoder.h"

#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-openvino-extra.h"
#அந்தர்பாவ "ggml-openvino.h"
#அந்தர்பாவ "ggml-quants.h"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "utils.h"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cassert>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <fstream>
#அந்தர்பாவ <iomanip>
#அந்தர்பாவ <map>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <mutex>
#அந்தர்பாவ <openvino/core/dimension.hpp>
#அந்தர்பாவ <openvino/core/except.hpp>
#அந்தர்பாவ <openvino/core/node.hpp>
#அந்தர்பாவ <openvino/core/partial_shape.hpp>
#அந்தர்பாவ <openvino/core/type/bfloat16.hpp>
#அந்தர்பாவ <openvino/core/type/element_type.hpp>
#அந்தர்பாவ <openvino/core/type/float16.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/runtime/tensor.hpp>
#அந்தர்பாவ <ostream>
#அந்தர்பாவ <set>
#அந்தர்பாவ <stdexcept>
#அந்தர்பாவ <string>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             தருக்க is_static,
                             தருக்க is_stateful,
                             தருக்க model_is_splitted,
                             தருக்க is_prefill,
                             முழு prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(பொய்),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    நிலையான தருக்க printed_address_map = பொய்;
    எனில் (!printed_address_map) {
        எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = உண்மை;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_உள்ளீடு_output();
    compute_முனை_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    க்கு (முழு முனை_n = 0; முனை_n < cgraph->n_nodes; முனை_n++) {
        m_முனை_info_list[முனை_n].முனை_op_case = compute_op_case(m_முனை_info_list[முனை_n].முனை);
        m_முனை_info_list[முனை_n].முனை_op_type = compute_op_type(m_முனை_info_list[முனை_n].முனை);
    }

    add_extra_inputs();
}

வெறுமை GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_முனை_info_list.clear();
    set_உள்ளீடு_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
தருக்க GgmlOvDecoder::is_bound_to(மாறிலி ggml_cgraph * cgraph) மாறிலி {
    திரும்பு m_cgraph == cgraph && cgraph->n_nodes > 0 && m_முனை_info_list.size() == (size_t) cgraph->n_nodes &&
           m_முனை_info_list.front().முனை == cgraph->nodes[0] &&
           m_முனை_info_list.back().முனை == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = உண்மை;
    set_உள்ளீடு_output();
    compute_model_inputs();
    compute_model_outputs();
    க்கு (முழு முனை_n = 0; முனை_n < cgraph->n_nodes; முனை_n++) {
        m_முனை_info_list[முனை_n].முனை_op_case = compute_op_case(m_முனை_info_list[முனை_n].முனை);
        m_முனை_info_list[முனை_n].முனை_op_type = compute_op_type(m_முனை_info_list[முனை_n].முனை);
    }
}

நாமாகாஶ {
தருக்க is_inplace_op(மாறிலி ggml_tensor * முனை) {
    திரும்பு முனை->op == GGML_OP_SET_ROWS || முனை->op == GGML_OP_CPY || (முனை->op == GGML_OP_SCALE && முனை->view_src);
}

தருக்க is_same_shape(மாறிலி ggml_tensor * a, மாறிலி ggml_tensor * b) {
    க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
        எனில் (a->ne[i] != b->ne[i]) {
            திரும்பு பொய்;
        }
    }
    திரும்பு உண்மை;
}

தருக்க is_conv_states_all_tensor(மாறிலி ggml_tensor * tensor) {
    திரும்பு tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

தருக்க is_full_single_slot_writeback(மாறிலி ggml_tensor * முனை) {
    திரும்பு முனை->view_src != nullptr && முனை->view_src->ne[1] == 1 && முனை->src[1] != nullptr &&
           முனை->src[1]->op == GGML_OP_VIEW && முனை->src[1]->view_src == முனை->view_src &&
           முனை->src[1]->view_offs == 0 && ggml_nbytes(முனை->src[1]) == ggml_nbytes(முனை->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
தருக்க is_moe_expert_sum_add(மாறிலி ggml_tensor * முனை) {
    std::vector<மாறிலி ggml_tensor *> leaves;
    மாறிலி ggml_tensor * cur = முனை;
    போது (cur->op == GGML_OP_ADD) {
        எனில் (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            திரும்பு பொய்;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    மாறிலி ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    க்கு (மாறிலி ggml_tensor * leaf : leaves) {
        எனில் (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            திரும்பு பொய்;
        }
        மாறிலி ggml_tensor * leaf_base = leaf->src[0];
        எனில் (base == nullptr) {
            base = leaf_base;
        } இல்லையெனில் எனில் (leaf_base != base) {
            திரும்பு பொய்;
        }
        எனில் (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            திரும்பு பொய்;
        }
        எனில் (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            திரும்பு பொய்;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        எனில் (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            திரும்பு பொய்;
        }
    }

    திரும்பு base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(மாறிலி ggml_cgraph * cgraph, மாறிலி ggml_tensor * tensor) {
    எனில் (tensor == nullptr) {
        திரும்பு "";
    }
    எனில் ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        மாறிலி auto * முனை = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        எனில் (முனை != cgraph->nodes + cgraph->n_nodes) {
            திரும்பு std::string(tensor->name) + "#n" + std::to_string(முனை - cgraph->nodes);
        }
        மாறிலி auto * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        எனில் (leaf != cgraph->leafs + cgraph->n_leafs) {
            திரும்பு std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    திரும்பு tensor->name;
}

நிலையான std::string get_tensor_ov_name(மாறிலி ggml_cgraph * cgraph, மாறிலி ggml_tensor * tensor) {
    திரும்பு GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

நிலையான std::string get_tensor_graph_உள்ளீடு_ov_name(மாறிலி GgmlOvDecoder * decoder,
                                                  மாறிலி ggml_cgraph * cgraph,
                                                  மாறிலி ggml_tensor * tensor,
                                                  மாறிலி ggml_tensor * op) {
    எனில் (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        திரும்பு "inp_pos";
    }
    எனில் (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        திரும்பு "embd";
    }
    எனில் (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        மாறிலி தருக்க is_swa = decoder->is_swa_mask(tensor);
        எனில் (decoder->is_stateful()) {
            திரும்பு is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        எனில் (is_swa) {
            திரும்பு get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    திரும்பு get_tensor_ov_name(cgraph, tensor);
}

வெறுமை GgmlOvDecoder::set_உள்ளீடு_output() {
    க்கு (முழு முனை_n = 0; முனை_n < m_cgraph->n_nodes; முனை_n++) {
        auto * முனை = m_cgraph->nodes[முனை_n];

        முனை_Info current_முனை_info;
        auto முனை_name = get_tensor_ov_name(m_cgraph, முனை);

        current_முனை_info.முனை = முனை;
        current_முனை_info.முனை_name = முனை_name;
        current_முனை_info.முனை_op_case = 0;
        current_முனை_info.data_addr = முனை->data;

        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = முனை->src[i];
            எனில் (src == nullptr) {
                தொடர்;
            }
            auto src_name = get_tensor_ov_name(m_cgraph, src);
            எனில் (src->flags & GGML_TENSOR_FLAG_உள்ளீடு) {
                src_name = get_tensor_graph_உள்ளீடு_ov_name(இது, m_cgraph, src, முனை);
            }
            current_முனை_info.முனை_inputs[src_name] = src;
            current_முனை_info.முனை_inputs_names.push_back(src_name);

            எனில் (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_முனை_info.முனை_inputs_views[src_name])> view_chain;
                auto * current = src;

                போது (current != nullptr) {
                    auto current_name = get_tensor_ov_name(m_cgraph, current);
                    எனில் (current->flags & GGML_TENSOR_FLAG_உள்ளீடு) {
                        current_name = get_tensor_graph_உள்ளீடு_ov_name(இது, m_cgraph, current, முனை);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    எனில் (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } இல்லையெனில் {
                        நிறுத்து;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_முனை_info.முனை_inputs_views[src_name] = view_chain;
            }
        }

        m_முனை_info_list.push_back(current_முனை_info);
    }
}

முழு GgmlOvDecoder::compute_op_case(மாறிலி ggml_tensor * முனை) மாறிலி {
    முழு op_case = 0;
    தேர்வு (முனை->op) {
    நிலை GGML_OP_RESHAPE: {
        எனில் (m_naive) {
            நிறுத்து;
        }
        auto name = std::string(முனை->name);
        auto * src = முனை->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        தருக்க recurrent_sequence = பொய்;
        க்கு (முழு i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            மாறிலி auto * consumer = m_cgraph->nodes[i];
            எனில் (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == முனை) {
                திரும்பு 1;
            } இல்லையெனில் எனில் (consumer->op == GGML_OP_SSM_CONV) {
                மாறிலி auto * concat = consumer->src[0];
                எனில் (concat->op == GGML_OP_CONCAT) {
                    மாறிலி auto * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == முனை;
                }
            } இல்லையெனில் எனில் (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                மாறிலி auto * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == முனை;
            }
        }
        எனில் (recurrent_sequence && முனை->ne[0] == src->ne[0] && முனை->ne[3] == 1) {
            திரும்பு 6;
        }
        எனில் (முனை->ne[0] == src->ne[0] && முனை->ne[2] == 1 && முனை->ne[3] == 1) {
            திரும்பு 5;
        }
        எனில் (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == முனை->ne[0] && src->src[0]->ne[1] == முனை->ne[1]) {
            op_case = 4;
        } இல்லையெனில் எனில் (முனை->ne[0] * முனை->ne[1] == src->ne[0]) {
            op_case = 1;
        } இல்லையெனில் எனில் (src->ne[0] * src->ne[1] == முனை->ne[0]) {
            op_case = 2;
            எனில் (src->ne[2] * src->ne[3] == முனை->ne[1]) {
                op_case = 5;
            }
        } இல்லையெனில் எனில் (முனை->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == முனை->ne[1]) {
            op_case = 3;
        } இல்லையெனில் எனில் (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } இல்லையெனில் எனில் (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } இல்லையெனில் எனில் (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        எனில் (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            மாறிலி auto * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            தருக்க recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            க்கு (முழு i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                மாறிலி auto * consumer = m_cgraph->nodes[i];
                எனில் (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    க்கு (முழு j : {3, 4}) {
                        மாறிலி auto * gate = consumer->src[j];
                        எனில் (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == முனை;
                    }
                } இல்லையெனில் எனில் (consumer->op == GGML_OP_MUL) {
                    க்கு (முழு j = 0; j < 2; ++j) {
                        மாறிலி auto * gate = consumer->src[j];
                        மாறிலி auto * norm = consumer->src[1 - j];
                        எனில் (gate->op != GGML_OP_UNARY || gate->src[0] != முனை) {
                            தொடர்;
                        }
                        எனில் (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            எனில் (recurrent) {
                op_case = 9;
            }
        }
        நிறுத்து;
    }
    நிலை GGML_OP_PERMUTE: {
        எனில் (முனை->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } இல்லையெனில் எனில் (முனை->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(முனை->view_src->name);
            முழு layer = extract_layer_from_name(src_name).value();
            எனில் (ggml_is_contiguous(முனை->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                எனில் (!is_swa_layer(layer)) {
                    op_case = 3;
                } இல்லையெனில் {
                    op_case = 4;
                }
            } இல்லையெனில் {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                எனில் (!is_swa_layer(layer)) {
                    op_case = 5;
                } இல்லையெனில் {
                    op_case = 6;
                }
            }
        } இல்லையெனில் {
            // rope'ed query tensor
            op_case = 2;
        }
        நிறுத்து;
    }
    நிலை GGML_OP_MUL_MAT: {
        எனில் (முனை->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        நிறுத்து;
    }
    நிலை GGML_OP_GET_ROWS: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        எனில் (முனை->src[0]->op == GGML_OP_RESHAPE && முனை->src[0]->src[0] != nullptr &&
            is_recurrent_cache(முனை->src[0]->src[0])) {
            மாறிலி தருக்க single_slot = முனை->src[0]->src[0]->ne[1] == 1;
            எனில் (முனை->src[1]->op == GGML_OP_VIEW) {
                op_case = (முனை->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } இல்லையெனில் எனில் (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        நிறுத்து;
    }
    நிலை GGML_OP_ROPE: {
        மாறிலி முழு mode = முனை->op_params[2];
        தேர்வு (mode) {
        நிலை GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            நிறுத்து;
        }
        நிலை GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            நிறுத்து;
        }
        நிலை GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            நிறுத்து;
        }
        நிலை GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            நிறுத்து;
        }
        இயல்பு:
            op_case = 0;
            நிறுத்து;
        }
        நிறுத்து;
    }
    நிலை GGML_OP_VIEW: {
        எனில் (!m_model_params.has_rs_rollback && முனை->src[0] != nullptr &&
            முனை->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            நிறுத்து;
        }
        எனில் (m_is_static && முனை->src[0] != nullptr &&
            (முனை->src[0]->op == GGML_OP_GATED_DELTA_NET || முனை->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            நிறுத்து;
        }
        எனில் (முனை->src[0]->op == GGML_OP_VIEW) {
            auto * src = முனை->src[0];
            எனில் (ggml_nelements(முனை) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            எனில் (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            auto * src = முனை->src[0];
            எனில் (ggml_nelements(முனை) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                எனில் (முனை->ne[0] == src->ne[0] && முனை->ne[1] == src->ne[2] && முனை->ne[2] == 1 &&
                    முனை->nb[0] == src->nb[0] && முனை->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    நிறுத்து;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                முழு diff_மொத்தம் = 0;
                க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
                    எனில் (முனை->ne[i] != src->ne[i]) {
                        diff_மொத்தம்++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    எனில் (முனை->ne[i] > src->ne[i]) {
                        திரும்பு 0;
                    }
                }
                எனில் (diff_மொத்தம் >= 1) {
                    op_case = 0;
                }
            }
        }
        நிறுத்து;
    }
    நிலை GGML_OP_RMS_NORM: {
        எனில் (முனை->src[0]->op == GGML_OP_VIEW) {
            எனில் (is_same_shape(முனை->src[0]->src[0], முனை->src[0])) {
                op_case = 1;
            } இல்லையெனில் எனில் (!m_model_params.has_rs_rollback &&
                       முனை->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } இல்லையெனில் எனில் (முனை->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        நிறுத்து;
    }
    நிலை GGML_OP_POOL_2D: {
        மாறிலி ggml_op_pool pool_mode = static_cast<ggml_op_pool>(முனை->op_params[0]);
        தேர்வு (pool_mode) {
        நிலை GGML_OP_POOL_MAX: {
            op_case = 1;
            நிறுத்து;
        }
        நிலை GGML_OP_POOL_AVG: {
            op_case = 2;
            நிறுத்து;
        }
        இயல்பு:
            op_case = 0;
            நிறுத்து;
        }
        நிறுத்து;
    }
    நிலை GGML_OP_UPSCALE: {
        மாறிலி int32_t mode_flags = முனை->op_params[0];
        மாறிலி ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        தேர்வு (scale_mode) {
        நிலை GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            நிறுத்து;
        }
        நிலை GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            நிறுத்து;
        }
        நிலை GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            நிறுத்து;
        }
        இயல்பு:
            op_case = 0;
            நிறுத்து;
        }
        நிறுத்து;
    }
    நிலை GGML_OP_CPY: {
        எனில் (முனை->src[0]->op == GGML_OP_VIEW) {
            எனில் (முனை->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                எனில் (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(முனை) ? 7 : 10;
                } இல்லையெனில் {
                    op_case = 1;
                }
            } இல்லையெனில் எனில் (GgmlOvDecoder::is_conv_state_writeback(முனை)) {
                op_case = is_full_single_slot_writeback(முனை) ? 8 : 2;
                நிறுத்து;
            } இல்லையெனில் எனில் (is_conv_states_all_tensor(முனை->view_src) && முனை->src[1] != nullptr &&
                       முனை->src[1]->op == GGML_OP_VIEW && முனை->src[1]->view_src == முனை->view_src) {
                op_case = 4;
                நிறுத்து;
            } இல்லையெனில் எனில் (முனை->src[0]->src[0]->op == GGML_OP_GET_ROWS && முனை->src[1] != nullptr &&
                       முனை->src[1]->op == GGML_OP_VIEW && முனை->src[1]->view_src != nullptr &&
                       is_recurrent_cache(முனை->src[1]->view_src) && முனை->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } இல்லையெனில் எனில் (முனை->src[0]->op == GGML_OP_GET_ROWS && முனை->src[1] != nullptr &&
                   முனை->src[1]->op == GGML_OP_VIEW && முனை->src[1]->view_src != nullptr &&
                   is_recurrent_cache(முனை->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = முனை->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } இல்லையெனில் எனில் (முனை->src[1] != nullptr && முனை->src[1]->op == GGML_OP_VIEW && முனை->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            மாறிலி ggml_tensor * kv_buf = முனை->src[1]->view_src;
            எனில் (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                க்கு (முழு i = 0; i < m_cgraph->n_nodes; i++) {
                    மாறிலி ggml_tensor * n = m_cgraph->nodes[i];
                    எனில் (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        தொடர்;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    எனில் ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        எனில் (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        நிறுத்து;
                    }
                }
            }
        }
        நிறுத்து;
    }
    நிலை GGML_OP_ADD: {
        எனில் (is_moe_expert_sum_add(முனை)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        நிறுத்து;
    }
    நிலை GGML_OP_SCALE: {
        எனில் (முனை->view_src && முனை->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = முனை->view_src->ne[1] == 1 ? 2 : 1;
        }
        நிறுத்து;
    }
    நிலை GGML_OP_L2_NORM: {
        எனில் (std::string(முனை->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        நிறுத்து;
    }
    நிலை GGML_OP_FLASH_ATTN_EXT: {
        எனில் (முனை->src[1] != nullptr && முனை->src[1]->op == GGML_OP_VIEW && முனை->src[1]->view_src != nullptr) {
            மாறிலி ggml_tensor * kv_buf = முனை->src[1]->view_src;
            எனில் (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (முனை->src[3] != nullptr) ? 1 : 2;
            }
        }
        நிறுத்து;
    }
    இயல்பு:
        நிறுத்து;
    }
    திரும்பு op_case;
}

std::optional<முழு> extract_layer_from_name(மாறிலி std::string & name) {
    size_t pos1 = name.find("_l");
    எனில் (pos1 == std::string::npos) {
        திரும்பு std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    எனில் (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    முழு layer = std::stoi(layer_str);
    திரும்பு layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
நிலையான முழு get_swa_window_from_mask(மாறிலி ggml_tensor * mask) {
    எனில் (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        திரும்பு -1;
    }
    எனில் (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    மாறிலி int64_t n_kv = mask->ne[0];
    மாறிலி int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    க்கு (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        க்கு (int64_t c = 0; c < n_kv; c++) {
            மாறிலி size_t i = (size_t) r * n_kv + c;
            மாறிலி தசம v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((மாறிலி ggml_fp16_t *) mask->data)[i]) :
                                                          ((மாறிலி தசம *) mask->data)[i];
            எனில் (v > -INFINITY) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    திரும்பு window > 0 ? (முழு) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, தருக்க is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    auto get_attention_pattern_case = [](மாறிலி ggml_tensor * முனை) -> முழு {
        எனில் (முனை == nullptr) {
            திரும்பு -1;
        }

        தேர்வு (முனை->op) {
        நிலை GGML_OP_FLASH_ATTN_EXT:
            எனில் (முனை->src[0] == nullptr || முனை->src[1] == nullptr) {
                திரும்பு -1;
            }
            தேர்வு (முனை->src[1]->op) {
            நிலை GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                எனில் (முனை->src[3] != nullptr && முனை->src[1]->src[0] != nullptr &&
                    முனை->src[1]->src[0]->op == GGML_OP_VIEW) {
                    திரும்பு 0;
                }
                நிறுத்து;
            நிலை GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                எனில் (முனை->src[3] != nullptr && முனை->src[1]->src[0] != nullptr &&
                    முனை->src[1]->src[0]->op == GGML_OP_PERMUTE && முனை->src[1]->src[0]->src[0] != nullptr &&
                    முனை->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    திரும்பு 1;
                }
                நிறுத்து;
            நிலை GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                எனில் (முனை->src[1]->view_src != nullptr) {
                    எனில் (முனை->src[3] != nullptr) {
                        திரும்பு 4;  // decoder self-attention
                    }
                    திரும்பு 5;      // cross-attention or encoder self-attention
                }
                நிறுத்து;
            இயல்பு:
                நிறுத்து;
            }
            நிறுத்து;
        நிலை GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            எனில் (முனை->src[0] != nullptr && முனை->src[1] != nullptr && முனை->src[0]->op == GGML_OP_MUL_MAT &&
                முனை->src[0]->src[0] != nullptr && முனை->src[0]->src[1] != nullptr &&
                முனை->src[0]->src[0]->op == GGML_OP_PERMUTE && முனை->src[0]->src[0]->src[0] != nullptr &&
                முனை->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                திரும்பு 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            எனில் (முனை->src[0]->op == GGML_OP_ADD && முனை->src[0]->src[0] != nullptr &&
                முனை->src[0]->src[0]->op == GGML_OP_MUL_MAT && முனை->src[0]->src[0]->src[0] != nullptr &&
                முனை->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                திரும்பு 3;
            }
            நிறுத்து;
        இயல்பு:
            நிறுத்து;
        }

        திரும்பு -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    auto get_attention_op_mask = [&get_attention_pattern_case](மாறிலி ggml_tensor * முனை) -> மாறிலி ggml_tensor * {
        தேர்வு (get_attention_pattern_case(முனை)) {
        நிலை 0:
        நிலை 1:
            திரும்பு முனை->src[3];
        நிலை 2:
        நிலை 3:
            திரும்பு முனை->src[1];
        இயல்பு:
            திரும்பு nullptr;
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
        std::map<முழு, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<முழு, மாறிலி ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
            மாறிலி ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            எனில் (mask == nullptr) {
                தொடர்;
            }
            மாறிலி ggml_tensor * cache_k_permute = nullptr;
            தேர்வு (get_attention_pattern_case(cgraph->nodes[i])) {
            நிலை 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     நிறுத்து;
            நிலை 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             நிறுத்து;
            நிலை 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             நிறுத்து;
            இயல்பு: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     நிறுத்து;
            }
            மாறிலி ggml_tensor * cache_k_view = cache_k_permute->src[0];
            எனில் (cache_k_view->op != GGML_OP_VIEW) {
                தொடர்;
            }
            மாறிலி ggml_tensor * leaf = cache_k_view->src[0];
            auto layer = extract_layer_from_name(leaf->name);
            எனில் (!layer.has_value()) {
                தொடர்;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        க்கு (மாறிலி auto & [layer, extent] : layer_extent) {
            எனில் (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                எனில் (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            க்கு (மாறிலி auto & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (long) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    தருக்க rope_seen = பொய்;
    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * முனை = cgraph->nodes[i];
        மாறிலி முழு attention_pattern_case = get_attention_pattern_case(முனை);
        எனில் (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            தேர்வு (attention_pattern_case) {
            நிலை 0:
                cache_k_permute = முனை->src[1];
                mask = முனை->src[3];
                நிறுத்து;
            நிலை 1:
                cache_k_permute = முனை->src[1]->src[0];
                mask = முனை->src[3];
                நிறுத்து;
            நிலை 2:
                cache_k_permute = முனை->src[0]->src[0];
                mask = முனை->src[1];
                நிறுத்து;
            நிலை 3:
                cache_k_permute = முனை->src[0]->src[0]->src[0];
                mask = முனை->src[1];
                நிறுத்து;
            நிலை 4:
            நிலை 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                auto * cache_k_view = முனை->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = முனை->src[0]->ne[1];
                எனில் (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } இல்லையெனில் {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                தொடர்;
            }
            இயல்பு:
                நிறுத்து;
            }

            ப்ரதிஜ்ஞா(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.உள்ளீடு_len = முனை->src[0]->ne[1];
            compute_params.token_len_per_seq = முனை->src[0]->ne[1];

            auto * cache_k_view = cache_k_permute->src[0];
            எனில் (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                தொடர்;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            முழு layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            மாறிலி தருக்க layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            எனில் (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } இல்லையெனில் {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, அளவு(size_t));
            compute_params.seq_active_start = offset / seq_size;

            எனில் (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } இல்லையெனில் {
                compute_params.attention_size = mask->ne[0];
            }
            எனில் (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        எனில் (முனை->op == GGML_OP_MUL_MAT && முனை->src[0]->op == GGML_OP_PERMUTE &&
            முனை->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(முனை->src[0]->view_src, முனை->view_src)) {
            எனில் (முனை->src[1]->op == GGML_OP_PERMUTE && முனை->src[1]->src[0]->op == GGML_OP_VIEW &&
                முனை->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = முனை->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        எனில் (முனை->op == GGML_OP_TRANSPOSE && முனை->src[0]->op == GGML_OP_PERMUTE &&
            முனை->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = முனை->ne[0];
            எனில் (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        எனில் (முனை->op == GGML_OP_ROPE) {
            எனில் (compute_params.token_len_per_seq == -1 && முனை->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(முனை->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(அளவு(model_params.rope_params) == அளவு(int32_t) * 16, "rope_params size");
            எனில் (!rope_seen) {
                memcpy(model_params.rope_params, முனை->op_params, அளவு(int32_t) * 16);
                rope_seen = உண்மை;
            } இல்லையெனில் எனில் (memcmp(model_params.rope_params, முனை->op_params, அளவு(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = உண்மை;
            }
        }
        எனில் (முனை->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = முனை->src[0]->ne[0];
        }
        எனில் (முனை->op == GGML_OP_SCALE && முனை->view_src != nullptr && is_recurrent_cache(முனை->view_src)) {
            எனில் (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = முனை->view_src->ne[1];
            } இல்லையெனில் {
                GGML_ASSERT(model_params.n_rs_slots == முனை->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(முனை) / முனை->view_src->ne[0];
            compute_params.cache_rs_reset_idx = முனை->src[0]->view_offs / முனை->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        எனில் (முனை->op == GGML_OP_CPY && முனை->view_src != nullptr && is_recurrent_cache(முனை->view_src) &&
            முனை->src[1] != nullptr && முனை->src[1]->op == GGML_OP_VIEW && முனை->src[1]->view_src == முனை->view_src) {
            மாறிலி தருக்க is_conv = is_conv_state_writeback(முனை);
            மாறிலி தருக்க is_gdn = முனை->src[0]->op == GGML_OP_VIEW && முனை->src[0]->src[0] != nullptr &&
                                முனை->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            மாறிலி தருக்க is_extra = முனை->src[0]->op == GGML_OP_GET_ROWS;
            மாறிலி தருக்க is_gdn_rollback = is_gdn && is_same_shape(முனை->src[0], முனை->src[1]);

            மாறிலி ggml_tensor * dest_view = முனை->src[1];
            மாறிலி ggml_tensor * cache = முனை->view_src;
            மாறிலி size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            எனில் (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = உண்மை;
            }
            எனில் (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(முனை)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (முழு) (dest_view->view_offs / row_bytes);
                எனில் (is_conv) {
                    writeback.src_begin = (முழு) (முனை->src[0]->view_offs / முனை->src[0]->view_src->nb[0]);
                } இல்லையெனில் எனில் (is_gdn_rollback) {
                    writeback.src_begin = (முழு) (முனை->src[0]->view_offs / முனை->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, முனை)] = writeback;
            }
            எனில் ((is_conv || is_gdn) && !is_full_single_slot_writeback(முனை)) {
                compute_params.s_copy_active_slot_len = (முழு) dest_view->ne[1];
            }
        }
    }
    எனில் (model_params.n_heads_kv == -1) {
        க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
            மாறிலி auto * முனை = cgraph->nodes[i];
            மாறிலி ggml_tensor * mask = nullptr;
            எனில் (முனை->op == GGML_OP_SOFT_MAX) {
                mask = முனை->src[1];
            } இல்லையெனில் எனில் (முனை->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = முனை->src[3];
            } இல்லையெனில் {
                தொடர்;
            }
            எனில் (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_உள்ளீடு) ||
                முனை->src[0] == nullptr) {
                தொடர்;
            }
            model_params.is_cacheless_attn = உண்மை;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.உள்ளீடு_len = முனை->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.உள்ளீடு_len;
            நிறுத்து;
        }
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    எனில் (model_params.is_cacheless_attn) {
        க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
            மாறிலி auto * முனை = cgraph->nodes[i];
            எனில் (முனை->op == GGML_OP_GET_ROWS && is_output_idx(முனை->src[1], முனை)) {
                compute_params.output_len = முனை->src[1]->ne[0];
                நிறுத்து;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    எனில் (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    திரும்பு {model_params, compute_params};
}

வெறுமை GgmlOvDecoder::validate_cgraph() மாறிலி {
    எனில் (m_model_params.n_seq > 1 && m_is_static == உண்மை) {
        எறி std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_உள்ளீடு_shape(மாறிலி ggml_tensor * op,
                                                      மாறிலி ggml_tensor * உள்ளீடு,
                                                      முழு dynamic_dim_index) மாறிலி {
    எனில் (m_naive) {
        திரும்பு உள்ளீடு != nullptr ? ov::PartialShape{get_shape(உள்ளீடு)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape உள்ளீடு_shape;

    எனில் (is_inp_tok(உள்ளீடு, op) || is_inp_pos(உள்ளீடு, op)) {
        // tokens or positions
        முழு len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        எனில் (m_is_static && is_inp_pos(உள்ளீடு, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        உள்ளீடு_shape = ov::PartialShape{1, 1, 1, len};

    } இல்லையெனில் எனில் (is_output_idx(உள்ளீடு, op)) {
        // output index
        உள்ளீடு_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } இல்லையெனில் எனில் (is_inp_mean(உள்ளீடு, op)) {
        உள்ளீடு_shape = m_is_static ? ov::PartialShape{1, 1, உள்ளீடு->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } இல்லையெனில் எனில் (is_inp_scale_rows(உள்ளீடு, op)) {
        உள்ளீடு_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } இல்லையெனில் எனில் (is_inp_mask(உள்ளீடு, op)) {
        // mask
        எனில் (m_is_static) {
            உள்ளீடு_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } இல்லையெனில் எனில் (m_is_stateful) {
            உள்ளீடு_shape = ov::PartialShape{1, 1, -1, -1};
        } இல்லையெனில் {
            உள்ளீடு_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } இல்லையெனில் எனில் (is_recurrent_cache(உள்ளீடு)) {
        உள்ளீடு_shape = ov::PartialShape{get_shape(உள்ளீடு)};
        எனில் (!m_is_static && !m_is_stateful && உள்ளீடு->ne[1] > 1) {
            உள்ளீடு_shape[2] = -1;
        }

    } இல்லையெனில் எனில் (is_kvcache(உள்ளீடு, op)) {
        // kvcache
        உள்ளீடு_shape = ov::PartialShape{get_shape(உள்ளீடு)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        மாறிலி தருக்க is_flat_kv = (உள்ளீடு->ne[1] == 1 && உள்ளீடு->ne[2] == 1 && உள்ளீடு->ne[3] == 1);
        எனில் (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            உள்ளீடு_shape[2] = -1;
        }
        எனில் (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            ப்ரதிஜ்ஞா(உள்ளீடு_shape.size() == 4 && உள்ளீடு_shape[0] == 1 && உள்ளீடு_shape[1] == 1 &&
                   உள்ளீடு_shape[2].is_dynamic() && உள்ளீடு_shape[3].is_static());
            மாறிலி முழு n_heads_kv = get_n_heads_kv_for_tensor(உள்ளீடு);
            ப்ரதிஜ்ஞா(n_heads_kv > 0 && உள்ளீடு_shape[3].get_length() % n_heads_kv == 0);
            மாறிலி int64_t combined_dim = உள்ளீடு_shape[3].get_length();  // n_heads_kv * head_size
            மாறிலி int64_t head_size = combined_dim / n_heads_kv;
            உள்ளீடு_shape = {உள்ளீடு_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } இல்லையெனில் எனில் (is_kv_idx(உள்ளீடு, op)) {
        // kv update index
        முழு len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        உள்ளீடு_shape = ov::PartialShape{1, 1, 1, len};

    } இல்லையெனில் எனில் (is_inp_s_copy(உள்ளீடு, op) || is_s_copy_leaf(உள்ளீடு)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        உள்ளீடு_shape = m_is_static ? ov::PartialShape{get_shape(உள்ளீடு)} : ov::PartialShape{1, 1, 1, -1};

    } இல்லையெனில் {
        உள்ளீடு_shape = ov::PartialShape{get_shape(உள்ளீடு)};
    }
    எனில் (dynamic_dim_index != -1 && m_model_is_splitted) {
        உள்ளீடு_shape[3 - dynamic_dim_index] = -1;
    }
    எனில் (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_உள்ளீடு && op->src[1] == உள்ளீடு) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        எனில் (m_is_static) {
            மாறிலி int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            உள்ளீடு_shape[2] = seq_active;
            உள்ளீடு_shape[3] = seq_active;
        } இல்லையெனில் {
            உள்ளீடு_shape[2] = -1;
            உள்ளீடு_shape[3] = -1;
        }
    }
    திரும்பு உள்ளீடு_shape;
}

தருக்க GgmlOvDecoder::is_s_copy_leaf(மாறிலி ggml_tensor * tensor) மாறிலி {
    எனில் (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        திரும்பு பொய்;
    }
    க்கு (முழு i = 0; i < m_cgraph->n_nodes; i++) {
        மாறிலி ggml_tensor * முனை = m_cgraph->nodes[i];
        எனில் (முனை->op != GGML_OP_GET_ROWS || முனை->src[0] == nullptr || முனை->src[1] == nullptr) {
            தொடர்;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        மாறிலி ggml_tensor * idx = முனை->src[1];
        போது (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        எனில் (idx != tensor) {
            தொடர்;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        மாறிலி ggml_tensor * data = முனை->src[0];
        போது (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        எனில் (data != nullptr && is_recurrent_cache(data)) {
            திரும்பு உண்மை;
        }
    }
    திரும்பு பொய்;
}

வெறுமை GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_உள்ளீடு = [இது](மாறிலி std::string & name, int64_t value, தருக்க force_parameter = பொய்) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    எனில் (m_compute_params.attention_size != -1) {
        create_1d_உள்ளீடு("attention_size", m_compute_params.attention_size);
    }
    எனில் (m_compute_params.attention_size_static != -1) {
        create_1d_உள்ளீடு("attention_size_static", m_compute_params.attention_size_static);
    }
    எனில் (m_compute_params.attention_size_swa != -1) {
        create_1d_உள்ளீடு("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    எனில் (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_உள்ளீடு("swa_window", m_compute_params.swa_window);
    }
    create_1d_உள்ளீடு("n_seq_active", m_compute_params.n_seq_active);
    create_1d_உள்ளீடு("seq_active_start", m_compute_params.seq_active_start);
    create_1d_உள்ளீடு("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    எனில் (m_compute_params.token_len_per_seq != -1) {
        create_1d_உள்ளீடு("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    எனில் (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_உள்ளீடு("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/உண்மை);
        create_1d_உள்ளீடு("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/உண்மை);
    }

    எனில் (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_உள்ளீடு("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        எனில் (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_உள்ளீடு("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/உண்மை);
        }
    }

    க்கு (மாறிலி auto & [முனை_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_உள்ளீடு("rs_slot_begin_" + முனை_name, writeback.slot_begin);
        எனில் (!m_is_static && writeback.src_begin >= 0) {
            create_1d_உள்ளீடு("rs_src_begin_" + முனை_name, writeback.src_begin);
        }
    }
}

தருக்க GgmlOvDecoder::முனை_is_used_as_src(மாறிலி முழு முனை_idx) {
    ggml_tensor * முனை = m_cgraph->nodes[முனை_idx];
    க்கு (முழு i = முனை_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_முனை = m_cgraph->nodes[i];
        க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
            எனில் (other_முனை->src[j] == முனை) {
                திரும்பு உண்மை;
            }
        }
    }
    திரும்பு பொய்;
}

வெறுமை GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    க்கு (முழு i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * முனை = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        எனில் (முனை->op == GGML_OP_NONE && முனை_is_used_as_src(i)) {
            std::string முனை_name = get_tensor_ov_name(m_cgraph, முனை);
            எனில் (m_model_weights.find(முனை_name) == m_model_weights.end()) {
                m_inputs[முனை_name] = முனை;
                m_model_inputs[முனை_name] = {get_ov_type(முனை),
                                             get_graph_உள்ளீடு_shape(முனை, nullptr, m_முனை_dynamic_dims[முனை])};
            }
            தொடர்;
        }
        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = முனை->src[i];
            எனில் (src == nullptr) {
                தொடர்;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            எனில் (src->flags & GGML_TENSOR_FLAG_உள்ளீடு) {
                src_name = get_tensor_graph_உள்ளீடு_ov_name(இது, m_cgraph, src, முனை);
            }
            எனில் (m_model_weights.find(src_name) != m_model_weights.end()) {
                தொடர்;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            எனில் (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                தொடர்;
            }

            தருக்க is_intermediate_முனை = பொய்;
            க்கு (மாறிலி auto & முனை_info : m_முனை_info_list) {
                எனில் (முனை_info.முனை == src) {
                    is_intermediate_முனை = உண்மை;
                    நிறுத்து;
                }
            }
            எனில் (is_intermediate_முனை) {
                தொடர்;
            }
            எனில் (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                தொடர்;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            எனில் (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                எனில் (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            போது (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_உள்ளீடு_shape(முனை, src, m_முனை_dynamic_dims[src])};
        }
    }
}

வெறுமை GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    க்கு (முழு முனை_n = 0; முனை_n < m_cgraph->n_nodes; முனை_n++) {
        auto * cur_முனை = m_cgraph->nodes[முனை_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        எனில் (cur_முனை->op == GGML_OP_NONE || cur_முனை->op == GGML_OP_VIEW || cur_முனை->op == GGML_OP_RESHAPE) {
            தொடர்;
        }
        எனில் (::is_inplace_op(cur_முனை) && ggml_nbytes(cur_முனை) == 0) {
            தொடர்;
        }
        auto cur_முனை_use_மொத்தம் = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_முனை)];
        எனில் (cur_முனை_use_மொத்தம் == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            எனில் (cur_முனை != nullptr && ::is_inplace_op(cur_முனை) && ggml_nbytes(cur_முனை) > 0) {
                cur_முனை = cur_முனை->view_src;
            }
        } இல்லையெனில் {
            முழு உள்ளீடு_use_மொத்தம் = 0;
            க்கு (முழு i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * முனை = m_cgraph->nodes[i];
                க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
                    எனில் (முனை->src[j] != வெற்று && முனை->src[j] == cur_முனை) {
                        உள்ளீடு_use_மொத்தம்++;
                    }
                }
            }
            எனில் (உள்ளீடு_use_மொத்தம் == cur_முனை_use_மொத்தம்) {
                cur_முனை = nullptr;
            }
        }
        எனில் (cur_முனை != nullptr) {
            std::string cur_முனை_name = get_tensor_ov_name(m_cgraph, cur_முனை);
            m_model_outputs[cur_முனை_name] = cur_முனை;
            m_model_output_names.insert(cur_முனை_name);
        }
    }
}

மாறிலி ggml_tensor * GgmlOvDecoder::get_tensor_used_op(மாறிலி ggml_tensor * tensor) மாறிலி {
    எனில் (tensor == nullptr) {
        திரும்பு nullptr;
    }
    க்கு (முழு i = 0; i < m_cgraph->n_nodes; i++) {
        மாறிலி auto * முனை = m_cgraph->nodes[i];
        க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
            எனில் (முனை->src[j] == tensor) {
                திரும்பு முனை;
            }
        }
    }
    திரும்பு nullptr;
}

மாறிலி ggml_tensor * GgmlOvDecoder::get_tensor_from_name(மாறிலி std::string & name) மாறிலி {
    க்கு (முழு i = 0; i < m_cgraph->n_nodes; i++) {
        மாறிலி auto * முனை = m_cgraph->nodes[i];
        க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
            மாறிலி auto * src = முனை->src[j];
            எனில் (src == nullptr) {
                நிறுத்து;
            }
            எனில் (get_tensor_ov_name(m_cgraph, src) == name) {
                திரும்பு src;
            }
        }
    }
    திரும்பு nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() மாறிலி {
    std::map<std::string, std::string> kv_param_res_names;
    க்கு (மாறிலி auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    திரும்பு kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
நிலையான தருக்க is_mul_mat_id_expert_weight(மாறிலி ggml_tensor * முனை, முழு src_index) {
    திரும்பு முனை->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, தருக்க naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    க்கு (முழு முனை_i = 0; முனை_i < n_nodes; முனை_i++) {
        auto * முனை = nodes[முனை_i];
        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = முனை->src[i];
            எனில் (src == nullptr) {
                தொடர்;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            எனில் (is_rope_freqs_weight(base, முனை)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            எனில் (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(முனை, i)) {
                எனில் (model_weights.find(src_name) == model_weights.end()) {
                    auto weight_முனை = create_weight_முனை(base, naive);
                    weight_முனை->set_friendly_name(src_name);
                    model_weights[src_name] = weight_முனை;
                }
            }
        }
    }
    திரும்பு model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
நிலையான std::mutex g_nonov_weight_cache_mutex;
நிலையான std::unordered_map<மாறிலி வெறுமை *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    க்கு (முழு முனை_i = 0; முனை_i < cgraph->n_nodes; முனை_i++) {
        auto * முனை = cgraph->nodes[முனை_i];
        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = முனை->src[i];
            எனில் (src == nullptr) {
                தொடர்;
            }
            மாறிலி ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            எனில் (is_rope_freqs_weight(base, முனை)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            எனில் (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    திரும்பு names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_முனை(ggml_tensor * tensor, தருக்க naive) {
    மாறிலி தருக்க is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    எனில் (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        எனில் (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            எனில் (weight_extra->weight_முனை) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                திரும்பு weight_extra->weight_முனை;
            }
        } இல்லையெனில் எனில் (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            எனில் (quant_extra->weight_முனை) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                திரும்பு quant_extra->weight_முனை;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    எனில் (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        auto weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        auto weight_முனை = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_முனை->set_friendly_name(tensor->name);
        திரும்பு weight_முனை;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    மாறிலி தருக்க cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    எனில் (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        auto it = g_nonov_weight_cache.find(tensor->data);
        எனில் (it != g_nonov_weight_cache.end()) {
            திரும்பு it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    நிலையான மாறிலி std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    எனில் (weight_types.find(tensor->type) == weight_types.end()) {
        எறி std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    எனில் (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        எனில் (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } இல்லையெனில் {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } இல்லையெனில் {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_முனை->set_friendly_name(tensor->name);
    எனில் (!is_ov_buffer) {
        எனில் (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            auto [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_முனை);
            திரும்பு it->second;
        }
        திரும்பு ov_weight.weight_முனை;
    }

    ggml_openvino_extra_base * extra;
    எனில் (ov_weight.is_quantized()) {
        extra = புதிய ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_முனை);
    } இல்லையெனில் {
        extra = புதிய ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_முனை);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    திரும்பு ov_weight.weight_முனை;
}

வெறுமை GgmlOvDecoder::dump_cgraph(மாறிலி ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    எனில் (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        திரும்பு;
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
    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * முனை = cgraph->nodes[i];

        // Get buffer type name
        மாறிலி எழுத்து * buf_name = "none";
        ggml_backend_buffer_t buf = முனை->view_src ? முனை->view_src->buffer : முனை->buffer;
        எனில் (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << முனை->ne[0] << ", "
             << std::setw(5) << முனை->ne[1] << ", "
             << std::setw(5) << முனை->ne[2] << ", "
             << std::setw(5) << முனை->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(முனை->op) << std::right << " "
             << std::left << std::setw(45) << முனை->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << முனை->nb[0] << ", "
             << std::setw(5) << முனை->nb[1] << ", "
             << std::setw(5) << முனை->nb[2] << ", "
             << std::setw(5) << முனை->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            எனில் (auto* src = முனை->src[i]) {
                // Get buffer type name for source
                மாறிலி எழுத்து * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                எனில் (src_buf) {
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
    க்கு (முழு i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * முனை = cgraph->leafs[i];

        // Get buffer type name for leaf
        மாறிலி எழுத்து * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = முனை->view_src ? முனை->view_src->buffer : முனை->buffer;
        எனில் (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << முனை->ne[0] << ", "
             << std::setw(5) << முனை->ne[1] << "] "
             << std::setw(8) << ggml_op_name(முனை->op) << " "
             << std::setw(16) << ggml_get_name(முனை)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

வெறுமை print_tensor_address_map(மாறிலி ggml_cgraph * cgraph) {
    std::map<வெறுமை *, std::vector<std::string>> address_map;
    க்கு (முழு முனை_n = 0; முனை_n < cgraph->n_nodes; முனை_n++) {
        auto * முனை = cgraph->nodes[முனை_n];
        எனில் (முனை->data) {
            auto it = address_map.find(முனை->data);
            எனில் (it == address_map.end()) {
                address_map[முனை->data] = std::vector<std::string>();
            }
            address_map[முனை->data].push_back(முனை->name);
        }
    }
    க்கு (மாறிலி auto & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        க்கு (மாறிலி auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(மாறிலி ggml_tensor * tensor) {
    std::vector<size_t> shape;
    க்கு (முழு i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    திரும்பு shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(மாறிலி ggml_tensor * tensor) {
    std::vector<size_t> stride;
    க்கு (முழு i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    திரும்பு stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(மாறிலி ggml_tensor * tensor) {
    தேர்வு (tensor->type) {
    நிலை GGML_TYPE_F64:
        திரும்பு ov::element::f64;
    நிலை GGML_TYPE_F32:
        திரும்பு ov::element::f32;
    நிலை GGML_TYPE_F16:
        திரும்பு ov::element::f16;
    நிலை GGML_TYPE_BF16:
        திரும்பு ov::element::bf16;
    நிலை GGML_TYPE_I8:
        திரும்பு ov::element::i8;
    நிலை GGML_TYPE_I16:
        திரும்பு ov::element::i16;
    நிலை GGML_TYPE_I32:
        திரும்பு ov::element::i32;
    நிலை GGML_TYPE_I64:
        திரும்பு ov::element::i64;
    இயல்பு:
        திரும்பு ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_உள்ளீடு_shape(முழு முனை_idx, மாறிலி std::string & name) மாறிலி {
    திரும்பு ov::PartialShape(get_shape(m_முனை_info_list[முனை_idx].முனை_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_உள்ளீடு_stride(முழு முனை_idx, மாறிலி std::string & name) மாறிலி {
    திரும்பு get_stride(m_முனை_info_list[முனை_idx].முனை_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_உள்ளீடு_size(முழு முனை_idx, மாறிலி std::string & name) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        திரும்பு it->second.size();
    }
    திரும்பு 0;
}

size_t GgmlOvDecoder::get_view_உள்ளீடு_offset(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        எனில் (view_index < it->second.size()) {
            திரும்பு it->second[view_index].second->view_offs;
        }
    }
    திரும்பு 0;
}

size_t GgmlOvDecoder::get_view_உள்ளீடு_src_offset(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        எனில் (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            எனில் (view_tensor && view_tensor->src[0]) {
                திரும்பு view_tensor->src[0]->view_offs;
            }
        }
    }
    திரும்பு 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_உள்ளீடு_stride(முழு முனை_idx,
                                                         மாறிலி std::string & name,
                                                         size_t view_index) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        எனில் (view_index < it->second.size()) {
            திரும்பு get_stride(it->second[view_index].second);
        }
    }
    திரும்பு {};
}

std::vector<size_t> GgmlOvDecoder::get_view_உள்ளீடு_src_stride(முழு முனை_idx,
                                                             மாறிலி std::string & name,
                                                             size_t view_index) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        எனில் (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            எனில் (view_tensor && view_tensor->src[0]) {
                திரும்பு get_stride(view_tensor->src[0]);
            }
        }
    }
    திரும்பு {};
}

ov::Shape GgmlOvDecoder::get_view_உள்ளீடு_ggml_shape(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        எனில் (view_index < it->second.size()) {
            திரும்பு get_shape(it->second[view_index].second);
        }
    }
    திரும்பு {};
}

ov::Shape GgmlOvDecoder::get_view_உள்ளீடு_src_ggml_shape(முழு முனை_idx,
                                                       மாறிலி std::string & name,
                                                       size_t view_index) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        எனில் (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            எனில் (view_tensor && view_tensor->src[0]) {
                திரும்பு get_shape(view_tensor->src[0]);
            }
        }
    }
    திரும்பு {};
}

ov::PartialShape GgmlOvDecoder::get_view_உள்ளீடு_ov_shape(முழு முனை_idx,
                                                        மாறிலி std::string & name,
                                                        size_t view_index) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        எனில் (view_index < it->second.size()) {
            auto * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            auto dynamic_it = m_முனை_dynamic_dims.find(tensor);
            எனில் (dynamic_it != m_முனை_dynamic_dims.end() && dynamic_it->second != -1) {
                முழு dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            திரும்பு shape;
        }
    }
    திரும்பு {};
}

ov::PartialShape GgmlOvDecoder::get_view_உள்ளீடு_src_ov_shape(முழு முனை_idx,
                                                            மாறிலி std::string & name,
                                                            size_t view_index) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        எனில் (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            எனில் (view_tensor && view_tensor->src[0]) {
                auto * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                auto dynamic_it = m_முனை_dynamic_dims.find(src_tensor);
                எனில் (dynamic_it != m_முனை_dynamic_dims.end() && dynamic_it->second != -1) {
                    முழு dynamic_dim_index = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
                }

                திரும்பு shape;
            }
        }
    }
    திரும்பு {};
}

std::string GgmlOvDecoder::get_view_உள்ளீடு_name(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        எனில் (view_index < it->second.size()) {
            திரும்பு it->second[view_index].first;
        }
    }
    திரும்பு "";
}

std::string GgmlOvDecoder::get_view_உள்ளீடு_src_name(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி {
    auto it = m_முனை_info_list[முனை_idx].முனை_inputs_views.find(name);
    எனில் (it != m_முனை_info_list[முனை_idx].முனை_inputs_views.end()) {
        எனில் (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            எனில் (view_tensor && view_tensor->src[0]) {
                திரும்பு get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    திரும்பு "";
}

ov::element::Type GgmlOvDecoder::get_உள்ளீடு_type(முழு முனை_idx, மாறிலி std::string & name) மாறிலி {
    திரும்பு get_ov_type(m_முனை_info_list[முனை_idx].முனை_inputs.at(name));
}

size_t GgmlOvDecoder::get_உள்ளீடு_size() மாறிலி {
    திரும்பு m_model_inputs.size();
}

size_t GgmlOvDecoder::get_உள்ளீடு_size(முழு முனை_idx) மாறிலி {
    திரும்பு m_முனை_info_list[முனை_idx].முனை_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_உள்ளீடு_names(முழு முனை_idx) மாறிலி {
    திரும்பு m_முனை_info_list[முனை_idx].முனை_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(முழு முனை_idx) மாறிலி {
    auto * ggml_tensor = m_முனை_info_list[முனை_idx].முனை;
    திரும்பு ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(மாறிலி முழு முனை_idx) மாறிலி {
    திரும்பு get_ov_type(m_முனை_info_list[முனை_idx].முனை);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(முழு முனை_idx) மாறிலி {
    auto * ggml_tensor = m_முனை_info_list[முனை_idx].முனை;
    திரும்பு get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(முழு முனை_idx) மாறிலி {
    auto * முனை = m_முனை_info_list[முனை_idx].முனை;
    எனில் (முனை->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        க்கு (முழு i = முனை_idx + 1; i < m_cgraph->n_nodes; i++) {
            auto * consumer = m_cgraph->nodes[i];
            எனில் (consumer->op != GGML_OP_VIEW || consumer->src[0] != முனை) {
                தொடர்;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            auto & name = consumer->view_offs == 0 ? attn_name : state_name;
            எனில் (!name.empty()) {
                திரும்பு {m_முனை_info_list[முனை_idx].முனை_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        எனில் (!attn_name.empty() && !state_name.empty()) {
            திரும்பு {attn_name, state_name};
        }
    }
    திரும்பு {m_முனை_info_list[முனை_idx].முனை_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(முழு முனை_idx) மாறிலி {
    auto * முனை = m_முனை_info_list[முனை_idx].முனை;
    எனில் (!::is_inplace_op(முனை) || முனை->view_src == nullptr || ggml_nbytes(முனை) == 0) {
        திரும்பு "";
    }
    மாறிலி முழு op_case = m_முனை_info_list[முனை_idx].முனை_op_case;
    எனில் (முனை->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        திரும்பு "";
    }
    திரும்பு get_tensor_ov_name(m_cgraph, முனை->view_src);
}

தருக்க GgmlOvDecoder::is_view_like_alias_of(முழு முனை_idx, மாறிலி std::string & view_src_name) மாறிலி {
    auto * முனை = m_முனை_info_list[முனை_idx].முனை;
    எனில் (முனை->view_src == nullptr || get_tensor_ov_name(m_cgraph, முனை->view_src) != view_src_name) {
        திரும்பு பொய்;
    }
    திரும்பு முனை->op == GGML_OP_RESHAPE || முனை->op == GGML_OP_VIEW;
}

மாறிலி std::string & GgmlOvDecoder::get_op_name() மாறிலி {
    நிலையான மாறிலி std::string unknown_name = "UNKNOWN_OP_NAME";
    திரும்பு unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(முழு முனை_idx) மாறிலி {
    auto it = m_முனை_dynamic_dims.find(m_முனை_info_list[முனை_idx].முனை);
    எனில் (it == m_முனை_dynamic_dims.end()) {
        திரும்பு -1;
    }
    திரும்பு it->second;
}

மாறிலி std::string & GgmlOvDecoder::get_op_name(முழு முனை_idx) மாறிலி {
    திரும்பு m_முனை_info_list[முனை_idx].முனை_name;
}

int32_t * GgmlOvDecoder::get_உள்ளீடு_op_params(முழு முனை_idx, மாறிலி std::string & name) மாறிலி {
    திரும்பு m_முனை_info_list[முனை_idx].முனை_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(முழு முனை_idx) மாறிலி {
    திரும்பு m_முனை_info_list[முனை_idx].முனை->op_params;
}

size_t GgmlOvDecoder::get_output_op_offset(முழு முனை_idx) மாறிலி {
    திரும்பு m_முனை_info_list[முனை_idx].முனை->view_offs;
}

வெறுமை GgmlOvDecoder::visit_subgraph(std::function<வெறுமை(std::shared_ptr<GgmlDecoder>, முழு முனை_idx)> முனை_visitor) மாறிலி {
    க்கு (முழு முனை_idx = 0; முனை_idx < m_cgraph->n_nodes; முனை_idx++) {
        எனில் (m_cgraph->nodes[முனை_idx]->op == GGML_OP_NONE) {
            தொடர்;
        }
        முனை_visitor(std::make_shared<GgmlOvDecoder>(*இது), முனை_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(மாறிலி ggml_tensor * முனை) {
    தேர்வு (முனை->op) {
    நிலை GGML_OP_UNARY:
        திரும்பு std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(முனை));
    நிலை GGML_OP_GLU:
        திரும்பு std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(முனை));
    இயல்பு:
        திரும்பு std::string("GGML_OP_") + ggml_op_name(முனை->op);
    }
}

மாறிலி std::string & GgmlOvDecoder::get_op_type(முழு முனை_idx) மாறிலி {
    திரும்பு m_முனை_info_list[முனை_idx].முனை_op_type;
}

மாறிலி std::string & GgmlOvDecoder::get_op_type() மாறிலி {
    நிலையான மாறிலி std::string unknown_op = "UNKNOWN_GGML_OP";
    திரும்பு unknown_op;
}

வெறுமை GgmlOvDecoder::compute_முனை_dynamic_dims() {
    auto visit_முனை = [&](auto && self, ggml_tensor * முனை) -> வெறுமை {
        எனில் (!முனை) {
            திரும்பு;
        }

        எனில் (முனை->op == GGML_OP_CPY) {
            m_முனை_dynamic_dims[முனை] = -1;
        }

        எனில் (m_முனை_dynamic_dims.count(முனை)) {
            திரும்பு;
        }
        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = முனை->src[i];
            எனில் (src == nullptr) {
                தொடர்;
            }
            கட்டமைப்பு ggml_tensor * வேர்_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            எனில் (வேர்_src) {
                எனில் (is_inp_tok(வேர்_src, முனை) || is_inp_pos(வேர்_src, முனை) || is_output_idx(வேர்_src, முனை)) {
                    m_முனை_dynamic_dims[வேர்_src] = 0;
                    m_முனை_dynamic_dims[src] = m_முனை_dynamic_dims[வேர்_src];
                    தொடர்;
                }
                self(self, வேர்_src);
                m_முனை_dynamic_dims[src] = m_முனை_dynamic_dims[வேர்_src];
            } இல்லையெனில் {
                எனில் (is_inp_tok(src, முனை) || is_inp_pos(src, முனை) || is_output_idx(src, முனை)) {
                    m_முனை_dynamic_dims[src] = 0;
                    தொடர்;
                }
                எனில் (is_inp_scale_rows(src, முனை)) {
                    m_முனை_dynamic_dims[src] = 1;
                    தொடர்;
                }
                எனில் (முனை->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_முனை_dynamic_dims[src] = 1;
                    தொடர்;
                }
                self(self, src);
            }
        }
        தேர்வு (முனை->op) {
        நிலை GGML_OP_NONE:
            m_முனை_dynamic_dims[முனை] = -1;
            நிறுத்து;
        நிலை GGML_OP_GET_ROWS:
            m_முனை_dynamic_dims[முனை] = -1;
            எனில் (m_முனை_dynamic_dims[முனை->src[1]] != -1) {
                auto dynamic_dim_idx = m_முனை_dynamic_dims[முனை->src[1]];
                எனில் (dynamic_dim_idx == 0) {
                    m_முனை_dynamic_dims[முனை] = 1;
                } இல்லையெனில் {
                    auto dynamic_dim_stride = முனை->src[1]->nb[dynamic_dim_idx] / ggml_type_size(முனை->src[1]->type) *
                                              ggml_type_size(முனை->src[0]->type);
                    க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
                        எனில் (dynamic_dim_stride == முனை->src[0]->nb[i]) {
                            m_முனை_dynamic_dims[முனை] = i;
                            நிறுத்து;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            நிறுத்து;
        நிலை GGML_OP_MUL:
        நிலை GGML_OP_MUL_MAT:
            m_முனை_dynamic_dims[முனை] = -1;
            எனில் (m_முனை_dynamic_dims[முனை->src[0]] != -1) {
                m_முனை_dynamic_dims[முனை] = m_முனை_dynamic_dims[முனை->src[0]];
            }
            எனில் (m_முனை_dynamic_dims[முனை->src[1]] != -1) {
                m_முனை_dynamic_dims[முனை] = m_முனை_dynamic_dims[முனை->src[1]];
            }
            நிறுத்து;
        நிலை GGML_OP_PERMUTE:
            m_முனை_dynamic_dims[முனை] = -1;
            எனில் (m_முனை_dynamic_dims[முனை->src[0]] != -1) {
                auto dynamic_dim_idx = m_முனை_dynamic_dims[முனை->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
                    எனில் (முனை->op_params[i] == dynamic_dim_idx) {
                        m_முனை_dynamic_dims[முனை] = i;
                        நிறுத்து;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            நிறுத்து;
        நிலை GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_முனை_dynamic_dims[முனை] = -1;
            எனில் (m_முனை_dynamic_dims[முனை->src[0]] != -1) {
                எனில் (முனை->src[0]->op == GGML_OP_NONE) {
                    m_முனை_dynamic_dims[முனை] = m_முனை_dynamic_dims[முனை->src[0]];
                    நிறுத்து;
                }
                auto dynamic_dim_idx = m_முனை_dynamic_dims[முனை->src[0]];
                auto dynamic_dim_மதிப்பு = முனை->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    முனை->src[0]->nb[dynamic_dim_idx] / ggml_type_size(முனை->src[0]->type) * ggml_type_size(முனை->type);
                க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
                    எனில் (முனை->nb[i] == dynamic_dim_stride) {
                        m_முனை_dynamic_dims[முனை] = i;
                        நிறுத்து;
                    }
                }
                எனில் (m_முனை_dynamic_dims[முனை] != -1 && dynamic_dim_மதிப்பு != முனை->ne[m_முனை_dynamic_dims[முனை]]) {
                    m_முனை_dynamic_dims[முனை] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    எனில் (ggml_nelements(முனை) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      முனை->name, முனை->src[0]->name);
                    }
                }
            }
            நிறுத்து;
        }
        நிலை GGML_OP_TRANSPOSE:
        நிலை GGML_OP_RESHAPE: {
            எனில் (is_same_shape(முனை->src[0], முனை)) {
                m_முனை_dynamic_dims[முனை] = m_முனை_dynamic_dims[முனை->src[0]];
                நிறுத்து;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_முனை_dynamic_dims[முனை] = -1;
            எனில் (m_முனை_dynamic_dims[முனை->src[0]] != -1) {
                auto dynamic_dim_idx = m_முனை_dynamic_dims[முனை->src[0]];
                auto dynamic_dim_stride = முனை->src[0]->nb[dynamic_dim_idx];
                க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
                    எனில் (முனை->nb[i] == dynamic_dim_stride && முனை->ne[i] == முனை->src[0]->ne[dynamic_dim_idx]) {
                        m_முனை_dynamic_dims[முனை] = i;
                        நிறுத்து;
                    }
                }
                எனில் (m_முனை_dynamic_dims[முனை] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", முனை->name);
                }
            }
            நிறுத்து;
        }
        நிலை GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr முழு q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_முனை_dynamic_dims[முனை] = -1;
            எனில் (m_முனை_dynamic_dims[முனை->src[0]] != -1) {
                auto q_dynamic_dim = m_முனை_dynamic_dims[முனை->src[0]];
                m_முனை_dynamic_dims[முனை] = q_to_out[q_dynamic_dim];
            }
            நிறுத்து;
        }
        நிலை GGML_OP_CONT:
            m_முனை_dynamic_dims[முனை] = -1;
            எனில் (m_முனை_dynamic_dims[முனை->src[0]] != -1) {
                auto dynamic_dim_idx = m_முனை_dynamic_dims[முனை->src[0]];
                எனில் (ggml_are_same_shape(முனை, முனை->src[0])) {
                    m_முனை_dynamic_dims[முனை] = dynamic_dim_idx;
                } இல்லையெனில் {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(முனை->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (முனை->src[0]->ne[0] / ggml_blck_size(முனை->src[0]->type));
                    க்கு (முழு i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * முனை->src[0]->ne[i - 1];
                    }

                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(முனை->src[0]->type) *
                                              ggml_type_size(முனை->type);
                    முழு matched_dim_மொத்தம் = 0;
                    முழு first_matched_dim = -1;
                    க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
                        எனில் (முனை->nb[i] == dynamic_dim_stride && முனை->ne[i] == முனை->src[0]->ne[dynamic_dim_idx]) {
                            எனில் (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_முனை_dynamic_dims[முனை] = i;
                            matched_dim_மொத்தம்++;
                        }
                    }
                    எனில் (matched_dim_மொத்தம் > 1 && முனை->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_முனை_dynamic_dims[முனை] = first_matched_dim;
                    } இல்லையெனில் எனில் (matched_dim_மொத்தம் != 1) {
                        m_முனை_dynamic_dims[முனை] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      முனை->name, முனை->src[0]->name);
                    }
                }
            }
            நிறுத்து;
        நிலை GGML_OP_CONCAT:
            க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
                எனில் (முனை->src[0]->ne[i] != முனை->ne[i]) {
                    m_முனை_dynamic_dims[முனை] = i;
                    நிறுத்து;
                }
            }
            நிறுத்து;
        நிலை GGML_OP_SSM_CONV:
        நிலை GGML_OP_GATED_DELTA_NET:
            m_முனை_dynamic_dims[முனை] = 1;
            நிறுத்து;
        நிலை GGML_OP_RMS_NORM:
        நிலை GGML_OP_L2_NORM:
        நிலை GGML_OP_NORM:
        நிலை GGML_OP_ADD:
        நிலை GGML_OP_SUB:
        நிலை GGML_OP_GLU:
        நிலை GGML_OP_ROPE:
        நிலை GGML_OP_SCALE:
        நிலை GGML_OP_SOFT_MAX:
        நிலை GGML_OP_ARGSORT:
        நிலை GGML_OP_ADD_ID:
        நிலை GGML_OP_UNARY:
        நிலை GGML_OP_CUMSUM:
        நிலை GGML_OP_FILL:
        நிலை GGML_OP_SET:
        நிலை GGML_OP_DIAG:
        நிலை GGML_OP_TRI:
        நிலை GGML_OP_REPEAT:
        நிலை GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        நிலை GGML_OP_DIV:
        நிலை GGML_OP_CLAMP:
        நிலை GGML_OP_PAD:
        நிலை GGML_OP_UPSCALE:
        நிலை GGML_OP_SIN:
        நிலை GGML_OP_COS:
        நிலை GGML_OP_LOG:
        நிலை GGML_OP_ROLL:
            m_முனை_dynamic_dims[முனை] = m_முனை_dynamic_dims[முனை->src[0]];
            நிறுத்து;
        நிலை GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_முனை_dynamic_dims[முனை] =
                (m_முனை_dynamic_dims[முனை->src[0]] == 0) ? -1 : m_முனை_dynamic_dims[முனை->src[0]];
            நிறுத்து;
        நிலை GGML_OP_MUL_MAT_ID:
        நிலை GGML_OP_SOLVE_TRI:
            m_முனை_dynamic_dims[முனை] = m_முனை_dynamic_dims[முனை->src[1]];
            நிறுத்து;
        நிலை GGML_OP_CPY:
        நிலை GGML_OP_SET_ROWS:
        நிலை GGML_OP_SUM:
        நிலை GGML_OP_MEAN:
            m_முனை_dynamic_dims[முனை] = -1;
            நிறுத்து;
        நிலை GGML_OP_IM2COL: {
            m_முனை_dynamic_dims[முனை] = -1;
            எனில் (m_முனை_dynamic_dims[முனை->src[1]] != -1) {
                மாறிலி தருக்க is_2D = முனை->op_params[6] == 1;
                மாறிலி முழு src_dyn = m_முனை_dynamic_dims[முனை->src[1]];
                எனில் (is_2D) {
                    எனில் (src_dyn == 0) {
                        m_முனை_dynamic_dims[முனை] = 1;  // IW -> OW
                    } இல்லையெனில் எனில் (src_dyn == 1) {
                        m_முனை_dynamic_dims[முனை] = 2;  // IH -> OH
                    } இல்லையெனில் எனில் (src_dyn == 3) {
                        m_முனை_dynamic_dims[முனை] = 3;  // N  -> N
                    }
                } இல்லையெனில் {
                    எனில் (src_dyn == 0) {
                        m_முனை_dynamic_dims[முனை] = 1;  // IW -> OW
                    } இல்லையெனில் எனில் (src_dyn == 2) {
                        m_முனை_dynamic_dims[முனை] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                எனில் (m_முனை_dynamic_dims[முனை] != -1) {
                    OPENVINO_ASSERT(முனை->src[1]->ne[src_dyn] == முனை->ne[m_முனை_dynamic_dims[முனை]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(முனை->name) +
                                        " and its src[1]: " + std::string(முனை->src[1]->name));
                }
            }
            நிறுத்து;
        }
        நிலை GGML_OP_IM2COL_3D: {
            m_முனை_dynamic_dims[முனை] = -1;
            எனில் (m_முனை_dynamic_dims[முனை->src[1]] != -1) {
                மாறிலி முழு src_dyn = m_முனை_dynamic_dims[முனை->src[1]];
                எனில் (src_dyn == 0) {
                    m_முனை_dynamic_dims[முனை] = 1;  // IW -> OW
                } இல்லையெனில் எனில் (src_dyn == 1) {
                    m_முனை_dynamic_dims[முனை] = 2;  // IH -> OH
                } இல்லையெனில் எனில் (src_dyn == 3) {
                    m_முனை_dynamic_dims[முனை] = 3;  // N  -> N
                }
                எனில் (m_முனை_dynamic_dims[முனை] != -1) {
                    OPENVINO_ASSERT(முனை->src[1]->ne[src_dyn] == முனை->ne[m_முனை_dynamic_dims[முனை]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(முனை->name) +
                                        " and its src[1]: " + std::string(முனை->src[1]->name));
                }
            }
            நிறுத்து;
        }
        இயல்பு:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(முனை->op), முனை->name);
            நிறுத்து;
        }
    };

    க்கு (முழு i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * முனை = m_cgraph->nodes[i];
        visit_முனை(visit_முனை, முனை);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    எனில் (0) {
        க்கு (முழு i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * முனை = m_cgraph->nodes[i];
            முழு dynamic_dim = m_முனை_dynamic_dims[முனை];
            std::cout << "[" << i << "] " << "node_name: " << முனை->name << " op: " << ggml_op_name(முனை->op)
                      << " shape: [";
            க்கு (முழு j = 0; j < 4; j++) {
                எனில் (j == dynamic_dim) {
                    std::cout << "*";
                } இல்லையெனில் {
                    std::cout << முனை->ne[j];
                }
                எனில் (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = முனை->src[j];
                எனில் (src == nullptr) {
                    தொடர்;
                }
                முழு src_dynamic_dim = m_முனை_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                க்கு (முழு k = 0; k < 4; k++) {
                    எனில் (k == src_dynamic_dim) {
                        std::cout << "*";
                    } இல்லையெனில் {
                        std::cout << src->ne[k];
                    }
                    எனில் (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
