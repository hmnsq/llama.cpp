#અન્તર્ભાવ "ggml-decoder.h"

#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-openvino-extra.h"
#અન્તર્ભાવ "ggml-openvino.h"
#અન્તર્ભાવ "ggml-quants.h"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "utils.h"

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cassert>
#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <fstream>
#અન્તર્ભાવ <iomanip>
#અન્તર્ભાવ <map>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <openvino/core/dimension.hpp>
#અન્તર્ભાવ <openvino/core/except.hpp>
#અન્તર્ભાવ <openvino/core/node.hpp>
#અન્તર્ભાવ <openvino/core/partial_shape.hpp>
#અન્તર્ભાવ <openvino/core/type/bfloat16.hpp>
#અન્તર્ભાવ <openvino/core/type/element_type.hpp>
#અન્તર્ભાવ <openvino/core/type/float16.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/runtime/tensor.hpp>
#અન્તર્ભાવ <ostream>
#અન્તર્ભાવ <set>
#અન્તર્ભાવ <stdexcept>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <unordered_map>
#અન્તર્ભાવ <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             તાર્કિક is_static,
                             તાર્કિક is_stateful,
                             તાર્કિક model_is_splitted,
                             તાર્કિક is_prefill,
                             પૂર્ણાંક prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(ખોટું),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    સ્થિર તાર્કિક printed_address_map = ખોટું;
    જો (!printed_address_map) {
        જો (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = સાચું;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_આગત_output();
    compute_ગાંઠ_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    માટે (પૂર્ણાંક ગાંઠ_n = 0; ગાંઠ_n < cgraph->n_nodes; ગાંઠ_n++) {
        m_ગાંઠ_info_list[ગાંઠ_n].ગાંઠ_op_case = compute_op_case(m_ગાંઠ_info_list[ગાંઠ_n].ગાંઠ);
        m_ગાંઠ_info_list[ગાંઠ_n].ગાંઠ_op_type = compute_op_type(m_ગાંઠ_info_list[ગાંઠ_n].ગાંઠ);
    }

    add_extra_inputs();
}

શૂન્ય GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_ગાંઠ_info_list.clear();
    set_આગત_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
તાર્કિક GgmlOvDecoder::is_bound_to(અચલ ggml_cgraph * cgraph) અચલ {
    પાછા m_cgraph == cgraph && cgraph->n_nodes > 0 && m_ગાંઠ_info_list.size() == (size_t) cgraph->n_nodes &&
           m_ગાંઠ_info_list.front().ગાંઠ == cgraph->nodes[0] &&
           m_ગાંઠ_info_list.back().ગાંઠ == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = સાચું;
    set_આગત_output();
    compute_model_inputs();
    compute_model_outputs();
    માટે (પૂર્ણાંક ગાંઠ_n = 0; ગાંઠ_n < cgraph->n_nodes; ગાંઠ_n++) {
        m_ગાંઠ_info_list[ગાંઠ_n].ગાંઠ_op_case = compute_op_case(m_ગાંઠ_info_list[ગાંઠ_n].ગાંઠ);
        m_ગાંઠ_info_list[ગાંઠ_n].ગાંઠ_op_type = compute_op_type(m_ગાંઠ_info_list[ગાંઠ_n].ગાંઠ);
    }
}

નામાકાશ {
તાર્કિક is_inplace_op(અચલ ggml_tensor * ગાંઠ) {
    પાછા ગાંઠ->op == GGML_OP_SET_ROWS || ગાંઠ->op == GGML_OP_CPY || (ગાંઠ->op == GGML_OP_SCALE && ગાંઠ->view_src);
}

તાર્કિક is_same_shape(અચલ ggml_tensor * a, અચલ ggml_tensor * b) {
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
        જો (a->ne[i] != b->ne[i]) {
            પાછા ખોટું;
        }
    }
    પાછા સાચું;
}

તાર્કિક is_conv_states_all_tensor(અચલ ggml_tensor * tensor) {
    પાછા tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

તાર્કિક is_full_single_slot_writeback(અચલ ggml_tensor * ગાંઠ) {
    પાછા ગાંઠ->view_src != nullptr && ગાંઠ->view_src->ne[1] == 1 && ગાંઠ->src[1] != nullptr &&
           ગાંઠ->src[1]->op == GGML_OP_VIEW && ગાંઠ->src[1]->view_src == ગાંઠ->view_src &&
           ગાંઠ->src[1]->view_offs == 0 && ggml_nbytes(ગાંઠ->src[1]) == ggml_nbytes(ગાંઠ->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
તાર્કિક is_moe_expert_sum_add(અચલ ggml_tensor * ગાંઠ) {
    std::vector<અચલ ggml_tensor *> leaves;
    અચલ ggml_tensor * cur = ગાંઠ;
    જ્યારે (cur->op == GGML_OP_ADD) {
        જો (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            પાછા ખોટું;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    અચલ ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    માટે (અચલ ggml_tensor * leaf : leaves) {
        જો (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            પાછા ખોટું;
        }
        અચલ ggml_tensor * leaf_base = leaf->src[0];
        જો (base == nullptr) {
            base = leaf_base;
        } નહીંતર જો (leaf_base != base) {
            પાછા ખોટું;
        }
        જો (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            પાછા ખોટું;
        }
        જો (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            પાછા ખોટું;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        જો (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            પાછા ખોટું;
        }
    }

    પાછા base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(અચલ ggml_cgraph * cgraph, અચલ ggml_tensor * tensor) {
    જો (tensor == nullptr) {
        પાછા "";
    }
    જો ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        અચલ auto * ગાંઠ = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        જો (ગાંઠ != cgraph->nodes + cgraph->n_nodes) {
            પાછા std::string(tensor->name) + "#n" + std::to_string(ગાંઠ - cgraph->nodes);
        }
        અચલ auto * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        જો (leaf != cgraph->leafs + cgraph->n_leafs) {
            પાછા std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    પાછા tensor->name;
}

સ્થિર std::string get_tensor_ov_name(અચલ ggml_cgraph * cgraph, અચલ ggml_tensor * tensor) {
    પાછા GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

સ્થિર std::string get_tensor_graph_આગત_ov_name(અચલ GgmlOvDecoder * decoder,
                                                  અચલ ggml_cgraph * cgraph,
                                                  અચલ ggml_tensor * tensor,
                                                  અચલ ggml_tensor * op) {
    જો (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        પાછા "inp_pos";
    }
    જો (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        પાછા "embd";
    }
    જો (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        અચલ તાર્કિક is_swa = decoder->is_swa_mask(tensor);
        જો (decoder->is_stateful()) {
            પાછા is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        જો (is_swa) {
            પાછા get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    પાછા get_tensor_ov_name(cgraph, tensor);
}

શૂન્ય GgmlOvDecoder::set_આગત_output() {
    માટે (પૂર્ણાંક ગાંઠ_n = 0; ગાંઠ_n < m_cgraph->n_nodes; ગાંઠ_n++) {
        auto * ગાંઠ = m_cgraph->nodes[ગાંઠ_n];

        ગાંઠ_Info current_ગાંઠ_info;
        auto ગાંઠ_name = get_tensor_ov_name(m_cgraph, ગાંઠ);

        current_ગાંઠ_info.ગાંઠ = ગાંઠ;
        current_ગાંઠ_info.ગાંઠ_name = ગાંઠ_name;
        current_ગાંઠ_info.ગાંઠ_op_case = 0;
        current_ગાંઠ_info.data_addr = ગાંઠ->data;

        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ગાંઠ->src[i];
            જો (src == nullptr) {
                ચાલુ;
            }
            auto src_name = get_tensor_ov_name(m_cgraph, src);
            જો (src->flags & GGML_TENSOR_FLAG_આગત) {
                src_name = get_tensor_graph_આગત_ov_name(આ, m_cgraph, src, ગાંઠ);
            }
            current_ગાંઠ_info.ગાંઠ_inputs[src_name] = src;
            current_ગાંઠ_info.ગાંઠ_inputs_names.push_back(src_name);

            જો (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_ગાંઠ_info.ગાંઠ_inputs_views[src_name])> view_chain;
                auto * current = src;

                જ્યારે (current != nullptr) {
                    auto current_name = get_tensor_ov_name(m_cgraph, current);
                    જો (current->flags & GGML_TENSOR_FLAG_આગત) {
                        current_name = get_tensor_graph_આગત_ov_name(આ, m_cgraph, current, ગાંઠ);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    જો (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } નહીંતર {
                        તોડો;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_ગાંઠ_info.ગાંઠ_inputs_views[src_name] = view_chain;
            }
        }

        m_ગાંઠ_info_list.push_back(current_ગાંઠ_info);
    }
}

પૂર્ણાંક GgmlOvDecoder::compute_op_case(અચલ ggml_tensor * ગાંઠ) અચલ {
    પૂર્ણાંક op_case = 0;
    પસંદગી (ગાંઠ->op) {
    સ્થિતિ GGML_OP_RESHAPE: {
        જો (m_naive) {
            તોડો;
        }
        auto name = std::string(ગાંઠ->name);
        auto * src = ગાંઠ->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        તાર્કિક recurrent_sequence = ખોટું;
        માટે (પૂર્ણાંક i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            અચલ auto * consumer = m_cgraph->nodes[i];
            જો (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == ગાંઠ) {
                પાછા 1;
            } નહીંતર જો (consumer->op == GGML_OP_SSM_CONV) {
                અચલ auto * concat = consumer->src[0];
                જો (concat->op == GGML_OP_CONCAT) {
                    અચલ auto * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == ગાંઠ;
                }
            } નહીંતર જો (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                અચલ auto * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == ગાંઠ;
            }
        }
        જો (recurrent_sequence && ગાંઠ->ne[0] == src->ne[0] && ગાંઠ->ne[3] == 1) {
            પાછા 6;
        }
        જો (ગાંઠ->ne[0] == src->ne[0] && ગાંઠ->ne[2] == 1 && ગાંઠ->ne[3] == 1) {
            પાછા 5;
        }
        જો (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == ગાંઠ->ne[0] && src->src[0]->ne[1] == ગાંઠ->ne[1]) {
            op_case = 4;
        } નહીંતર જો (ગાંઠ->ne[0] * ગાંઠ->ne[1] == src->ne[0]) {
            op_case = 1;
        } નહીંતર જો (src->ne[0] * src->ne[1] == ગાંઠ->ne[0]) {
            op_case = 2;
            જો (src->ne[2] * src->ne[3] == ગાંઠ->ne[1]) {
                op_case = 5;
            }
        } નહીંતર જો (ગાંઠ->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == ગાંઠ->ne[1]) {
            op_case = 3;
        } નહીંતર જો (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } નહીંતર જો (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } નહીંતર જો (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        જો (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            અચલ auto * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            તાર્કિક recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            માટે (પૂર્ણાંક i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                અચલ auto * consumer = m_cgraph->nodes[i];
                જો (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    માટે (પૂર્ણાંક j : {3, 4}) {
                        અચલ auto * gate = consumer->src[j];
                        જો (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == ગાંઠ;
                    }
                } નહીંતર જો (consumer->op == GGML_OP_MUL) {
                    માટે (પૂર્ણાંક j = 0; j < 2; ++j) {
                        અચલ auto * gate = consumer->src[j];
                        અચલ auto * norm = consumer->src[1 - j];
                        જો (gate->op != GGML_OP_UNARY || gate->src[0] != ગાંઠ) {
                            ચાલુ;
                        }
                        જો (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            જો (recurrent) {
                op_case = 9;
            }
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_PERMUTE: {
        જો (ગાંઠ->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } નહીંતર જો (ગાંઠ->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(ગાંઠ->view_src->name);
            પૂર્ણાંક layer = extract_layer_from_name(src_name).value();
            જો (ggml_is_contiguous(ગાંઠ->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                જો (!is_swa_layer(layer)) {
                    op_case = 3;
                } નહીંતર {
                    op_case = 4;
                }
            } નહીંતર {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                જો (!is_swa_layer(layer)) {
                    op_case = 5;
                } નહીંતર {
                    op_case = 6;
                }
            }
        } નહીંતર {
            // rope'ed query tensor
            op_case = 2;
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_MUL_MAT: {
        જો (ગાંઠ->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_GET_ROWS: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        જો (ગાંઠ->src[0]->op == GGML_OP_RESHAPE && ગાંઠ->src[0]->src[0] != nullptr &&
            is_recurrent_cache(ગાંઠ->src[0]->src[0])) {
            અચલ તાર્કિક single_slot = ગાંઠ->src[0]->src[0]->ne[1] == 1;
            જો (ગાંઠ->src[1]->op == GGML_OP_VIEW) {
                op_case = (ગાંઠ->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } નહીંતર જો (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_ROPE: {
        અચલ પૂર્ણાંક mode = ગાંઠ->op_params[2];
        પસંદગી (mode) {
        સ્થિતિ GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            તોડો;
        }
        સ્થિતિ GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            તોડો;
        }
        સ્થિતિ GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            તોડો;
        }
        સ્થિતિ GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            તોડો;
        }
        મૂળભૂત:
            op_case = 0;
            તોડો;
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_VIEW: {
        જો (!m_model_params.has_rs_rollback && ગાંઠ->src[0] != nullptr &&
            ગાંઠ->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            તોડો;
        }
        જો (m_is_static && ગાંઠ->src[0] != nullptr &&
            (ગાંઠ->src[0]->op == GGML_OP_GATED_DELTA_NET || ગાંઠ->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            તોડો;
        }
        જો (ગાંઠ->src[0]->op == GGML_OP_VIEW) {
            auto * src = ગાંઠ->src[0];
            જો (ggml_nelements(ગાંઠ) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            જો (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            auto * src = ગાંઠ->src[0];
            જો (ggml_nelements(ગાંઠ) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                જો (ગાંઠ->ne[0] == src->ne[0] && ગાંઠ->ne[1] == src->ne[2] && ગાંઠ->ne[2] == 1 &&
                    ગાંઠ->nb[0] == src->nb[0] && ગાંઠ->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    તોડો;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                પૂર્ણાંક diff_ગણતરી = 0;
                માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
                    જો (ગાંઠ->ne[i] != src->ne[i]) {
                        diff_ગણતરી++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    જો (ગાંઠ->ne[i] > src->ne[i]) {
                        પાછા 0;
                    }
                }
                જો (diff_ગણતરી >= 1) {
                    op_case = 0;
                }
            }
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_RMS_NORM: {
        જો (ગાંઠ->src[0]->op == GGML_OP_VIEW) {
            જો (is_same_shape(ગાંઠ->src[0]->src[0], ગાંઠ->src[0])) {
                op_case = 1;
            } નહીંતર જો (!m_model_params.has_rs_rollback &&
                       ગાંઠ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } નહીંતર જો (ગાંઠ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_POOL_2D: {
        અચલ ggml_op_pool pool_mode = static_cast<ggml_op_pool>(ગાંઠ->op_params[0]);
        પસંદગી (pool_mode) {
        સ્થિતિ GGML_OP_POOL_MAX: {
            op_case = 1;
            તોડો;
        }
        સ્થિતિ GGML_OP_POOL_AVG: {
            op_case = 2;
            તોડો;
        }
        મૂળભૂત:
            op_case = 0;
            તોડો;
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_UPSCALE: {
        અચલ int32_t mode_flags = ગાંઠ->op_params[0];
        અચલ ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        પસંદગી (scale_mode) {
        સ્થિતિ GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            તોડો;
        }
        સ્થિતિ GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            તોડો;
        }
        સ્થિતિ GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            તોડો;
        }
        મૂળભૂત:
            op_case = 0;
            તોડો;
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_CPY: {
        જો (ગાંઠ->src[0]->op == GGML_OP_VIEW) {
            જો (ગાંઠ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                જો (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(ગાંઠ) ? 7 : 10;
                } નહીંતર {
                    op_case = 1;
                }
            } નહીંતર જો (GgmlOvDecoder::is_conv_state_writeback(ગાંઠ)) {
                op_case = is_full_single_slot_writeback(ગાંઠ) ? 8 : 2;
                તોડો;
            } નહીંતર જો (is_conv_states_all_tensor(ગાંઠ->view_src) && ગાંઠ->src[1] != nullptr &&
                       ગાંઠ->src[1]->op == GGML_OP_VIEW && ગાંઠ->src[1]->view_src == ગાંઠ->view_src) {
                op_case = 4;
                તોડો;
            } નહીંતર જો (ગાંઠ->src[0]->src[0]->op == GGML_OP_GET_ROWS && ગાંઠ->src[1] != nullptr &&
                       ગાંઠ->src[1]->op == GGML_OP_VIEW && ગાંઠ->src[1]->view_src != nullptr &&
                       is_recurrent_cache(ગાંઠ->src[1]->view_src) && ગાંઠ->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } નહીંતર જો (ગાંઠ->src[0]->op == GGML_OP_GET_ROWS && ગાંઠ->src[1] != nullptr &&
                   ગાંઠ->src[1]->op == GGML_OP_VIEW && ગાંઠ->src[1]->view_src != nullptr &&
                   is_recurrent_cache(ગાંઠ->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = ગાંઠ->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } નહીંતર જો (ગાંઠ->src[1] != nullptr && ગાંઠ->src[1]->op == GGML_OP_VIEW && ગાંઠ->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            અચલ ggml_tensor * kv_buf = ગાંઠ->src[1]->view_src;
            જો (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                માટે (પૂર્ણાંક i = 0; i < m_cgraph->n_nodes; i++) {
                    અચલ ggml_tensor * n = m_cgraph->nodes[i];
                    જો (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        ચાલુ;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    જો ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        જો (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        તોડો;
                    }
                }
            }
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_ADD: {
        જો (is_moe_expert_sum_add(ગાંઠ)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_SCALE: {
        જો (ગાંઠ->view_src && ગાંઠ->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = ગાંઠ->view_src->ne[1] == 1 ? 2 : 1;
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_L2_NORM: {
        જો (std::string(ગાંઠ->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        તોડો;
    }
    સ્થિતિ GGML_OP_FLASH_ATTN_EXT: {
        જો (ગાંઠ->src[1] != nullptr && ગાંઠ->src[1]->op == GGML_OP_VIEW && ગાંઠ->src[1]->view_src != nullptr) {
            અચલ ggml_tensor * kv_buf = ગાંઠ->src[1]->view_src;
            જો (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (ગાંઠ->src[3] != nullptr) ? 1 : 2;
            }
        }
        તોડો;
    }
    મૂળભૂત:
        તોડો;
    }
    પાછા op_case;
}

std::optional<પૂર્ણાંક> extract_layer_from_name(અચલ std::string & name) {
    size_t pos1 = name.find("_l");
    જો (pos1 == std::string::npos) {
        પાછા std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    જો (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    પૂર્ણાંક layer = std::stoi(layer_str);
    પાછા layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
સ્થિર પૂર્ણાંક get_swa_window_from_mask(અચલ ggml_tensor * mask) {
    જો (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        પાછા -1;
    }
    જો (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    અચલ int64_t n_kv = mask->ne[0];
    અચલ int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    માટે (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        માટે (int64_t c = 0; c < n_kv; c++) {
            અચલ size_t i = (size_t) r * n_kv + c;
            અચલ દશાંશ v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((અચલ ggml_fp16_t *) mask->data)[i]) :
                                                          ((અચલ દશાંશ *) mask->data)[i];
            જો (v > -INFINITY) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    પાછા window > 0 ? (પૂર્ણાંક) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, તાર્કિક is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    auto get_attention_pattern_case = [](અચલ ggml_tensor * ગાંઠ) -> પૂર્ણાંક {
        જો (ગાંઠ == nullptr) {
            પાછા -1;
        }

        પસંદગી (ગાંઠ->op) {
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            જો (ગાંઠ->src[0] == nullptr || ગાંઠ->src[1] == nullptr) {
                પાછા -1;
            }
            પસંદગી (ગાંઠ->src[1]->op) {
            સ્થિતિ GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                જો (ગાંઠ->src[3] != nullptr && ગાંઠ->src[1]->src[0] != nullptr &&
                    ગાંઠ->src[1]->src[0]->op == GGML_OP_VIEW) {
                    પાછા 0;
                }
                તોડો;
            સ્થિતિ GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                જો (ગાંઠ->src[3] != nullptr && ગાંઠ->src[1]->src[0] != nullptr &&
                    ગાંઠ->src[1]->src[0]->op == GGML_OP_PERMUTE && ગાંઠ->src[1]->src[0]->src[0] != nullptr &&
                    ગાંઠ->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    પાછા 1;
                }
                તોડો;
            સ્થિતિ GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                જો (ગાંઠ->src[1]->view_src != nullptr) {
                    જો (ગાંઠ->src[3] != nullptr) {
                        પાછા 4;  // decoder self-attention
                    }
                    પાછા 5;      // cross-attention or encoder self-attention
                }
                તોડો;
            મૂળભૂત:
                તોડો;
            }
            તોડો;
        સ્થિતિ GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            જો (ગાંઠ->src[0] != nullptr && ગાંઠ->src[1] != nullptr && ગાંઠ->src[0]->op == GGML_OP_MUL_MAT &&
                ગાંઠ->src[0]->src[0] != nullptr && ગાંઠ->src[0]->src[1] != nullptr &&
                ગાંઠ->src[0]->src[0]->op == GGML_OP_PERMUTE && ગાંઠ->src[0]->src[0]->src[0] != nullptr &&
                ગાંઠ->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                પાછા 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            જો (ગાંઠ->src[0]->op == GGML_OP_ADD && ગાંઠ->src[0]->src[0] != nullptr &&
                ગાંઠ->src[0]->src[0]->op == GGML_OP_MUL_MAT && ગાંઠ->src[0]->src[0]->src[0] != nullptr &&
                ગાંઠ->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                પાછા 3;
            }
            તોડો;
        મૂળભૂત:
            તોડો;
        }

        પાછા -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    auto get_attention_op_mask = [&get_attention_pattern_case](અચલ ggml_tensor * ગાંઠ) -> અચલ ggml_tensor * {
        પસંદગી (get_attention_pattern_case(ગાંઠ)) {
        સ્થિતિ 0:
        સ્થિતિ 1:
            પાછા ગાંઠ->src[3];
        સ્થિતિ 2:
        સ્થિતિ 3:
            પાછા ગાંઠ->src[1];
        મૂળભૂત:
            પાછા nullptr;
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
        std::map<પૂર્ણાંક, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<પૂર્ણાંક, અચલ ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
            અચલ ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            જો (mask == nullptr) {
                ચાલુ;
            }
            અચલ ggml_tensor * cache_k_permute = nullptr;
            પસંદગી (get_attention_pattern_case(cgraph->nodes[i])) {
            સ્થિતિ 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     તોડો;
            સ્થિતિ 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             તોડો;
            સ્થિતિ 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             તોડો;
            મૂળભૂત: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     તોડો;
            }
            અચલ ggml_tensor * cache_k_view = cache_k_permute->src[0];
            જો (cache_k_view->op != GGML_OP_VIEW) {
                ચાલુ;
            }
            અચલ ggml_tensor * leaf = cache_k_view->src[0];
            auto layer = extract_layer_from_name(leaf->name);
            જો (!layer.has_value()) {
                ચાલુ;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        માટે (અચલ auto & [layer, extent] : layer_extent) {
            જો (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                જો (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        જો (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            માટે (અચલ auto & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (long) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    તાર્કિક rope_seen = ખોટું;
    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ગાંઠ = cgraph->nodes[i];
        અચલ પૂર્ણાંક attention_pattern_case = get_attention_pattern_case(ગાંઠ);
        જો (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            પસંદગી (attention_pattern_case) {
            સ્થિતિ 0:
                cache_k_permute = ગાંઠ->src[1];
                mask = ગાંઠ->src[3];
                તોડો;
            સ્થિતિ 1:
                cache_k_permute = ગાંઠ->src[1]->src[0];
                mask = ગાંઠ->src[3];
                તોડો;
            સ્થિતિ 2:
                cache_k_permute = ગાંઠ->src[0]->src[0];
                mask = ગાંઠ->src[1];
                તોડો;
            સ્થિતિ 3:
                cache_k_permute = ગાંઠ->src[0]->src[0]->src[0];
                mask = ગાંઠ->src[1];
                તોડો;
            સ્થિતિ 4:
            સ્થિતિ 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                auto * cache_k_view = ગાંઠ->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = ગાંઠ->src[0]->ne[1];
                જો (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } નહીંતર {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                ચાલુ;
            }
            મૂળભૂત:
                તોડો;
            }

            પ્રતિજ્ઞા(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.આગત_len = ગાંઠ->src[0]->ne[1];
            compute_params.token_len_per_seq = ગાંઠ->src[0]->ne[1];

            auto * cache_k_view = cache_k_permute->src[0];
            જો (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                ચાલુ;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            પૂર્ણાંક layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            અચલ તાર્કિક layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            જો (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } નહીંતર {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, કદ(size_t));
            compute_params.seq_active_start = offset / seq_size;

            જો (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } નહીંતર {
                compute_params.attention_size = mask->ne[0];
            }
            જો (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        જો (ગાંઠ->op == GGML_OP_MUL_MAT && ગાંઠ->src[0]->op == GGML_OP_PERMUTE &&
            ગાંઠ->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(ગાંઠ->src[0]->view_src, ગાંઠ->view_src)) {
            જો (ગાંઠ->src[1]->op == GGML_OP_PERMUTE && ગાંઠ->src[1]->src[0]->op == GGML_OP_VIEW &&
                ગાંઠ->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = ગાંઠ->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        જો (ગાંઠ->op == GGML_OP_TRANSPOSE && ગાંઠ->src[0]->op == GGML_OP_PERMUTE &&
            ગાંઠ->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = ગાંઠ->ne[0];
            જો (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        જો (ગાંઠ->op == GGML_OP_ROPE) {
            જો (compute_params.token_len_per_seq == -1 && ગાંઠ->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(ગાંઠ->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(કદ(model_params.rope_params) == કદ(int32_t) * 16, "rope_params size");
            જો (!rope_seen) {
                memcpy(model_params.rope_params, ગાંઠ->op_params, કદ(int32_t) * 16);
                rope_seen = સાચું;
            } નહીંતર જો (memcmp(model_params.rope_params, ગાંઠ->op_params, કદ(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = સાચું;
            }
        }
        જો (ગાંઠ->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = ગાંઠ->src[0]->ne[0];
        }
        જો (ગાંઠ->op == GGML_OP_SCALE && ગાંઠ->view_src != nullptr && is_recurrent_cache(ગાંઠ->view_src)) {
            જો (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = ગાંઠ->view_src->ne[1];
            } નહીંતર {
                GGML_ASSERT(model_params.n_rs_slots == ગાંઠ->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(ગાંઠ) / ગાંઠ->view_src->ne[0];
            compute_params.cache_rs_reset_idx = ગાંઠ->src[0]->view_offs / ગાંઠ->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        જો (ગાંઠ->op == GGML_OP_CPY && ગાંઠ->view_src != nullptr && is_recurrent_cache(ગાંઠ->view_src) &&
            ગાંઠ->src[1] != nullptr && ગાંઠ->src[1]->op == GGML_OP_VIEW && ગાંઠ->src[1]->view_src == ગાંઠ->view_src) {
            અચલ તાર્કિક is_conv = is_conv_state_writeback(ગાંઠ);
            અચલ તાર્કિક is_gdn = ગાંઠ->src[0]->op == GGML_OP_VIEW && ગાંઠ->src[0]->src[0] != nullptr &&
                                ગાંઠ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            અચલ તાર્કિક is_extra = ગાંઠ->src[0]->op == GGML_OP_GET_ROWS;
            અચલ તાર્કિક is_gdn_rollback = is_gdn && is_same_shape(ગાંઠ->src[0], ગાંઠ->src[1]);

            અચલ ggml_tensor * dest_view = ગાંઠ->src[1];
            અચલ ggml_tensor * cache = ગાંઠ->view_src;
            અચલ size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            જો (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = સાચું;
            }
            જો (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(ગાંઠ)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (પૂર્ણાંક) (dest_view->view_offs / row_bytes);
                જો (is_conv) {
                    writeback.src_begin = (પૂર્ણાંક) (ગાંઠ->src[0]->view_offs / ગાંઠ->src[0]->view_src->nb[0]);
                } નહીંતર જો (is_gdn_rollback) {
                    writeback.src_begin = (પૂર્ણાંક) (ગાંઠ->src[0]->view_offs / ગાંઠ->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, ગાંઠ)] = writeback;
            }
            જો ((is_conv || is_gdn) && !is_full_single_slot_writeback(ગાંઠ)) {
                compute_params.s_copy_active_slot_len = (પૂર્ણાંક) dest_view->ne[1];
            }
        }
    }
    જો (model_params.n_heads_kv == -1) {
        માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
            અચલ auto * ગાંઠ = cgraph->nodes[i];
            અચલ ggml_tensor * mask = nullptr;
            જો (ગાંઠ->op == GGML_OP_SOFT_MAX) {
                mask = ગાંઠ->src[1];
            } નહીંતર જો (ગાંઠ->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = ગાંઠ->src[3];
            } નહીંતર {
                ચાલુ;
            }
            જો (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_આગત) ||
                ગાંઠ->src[0] == nullptr) {
                ચાલુ;
            }
            model_params.is_cacheless_attn = સાચું;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.આગત_len = ગાંઠ->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.આગત_len;
            તોડો;
        }
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    જો (model_params.is_cacheless_attn) {
        માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
            અચલ auto * ગાંઠ = cgraph->nodes[i];
            જો (ગાંઠ->op == GGML_OP_GET_ROWS && is_output_idx(ગાંઠ->src[1], ગાંઠ)) {
                compute_params.output_len = ગાંઠ->src[1]->ne[0];
                તોડો;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    જો (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    પાછા {model_params, compute_params};
}

શૂન્ય GgmlOvDecoder::validate_cgraph() અચલ {
    જો (m_model_params.n_seq > 1 && m_is_static == સાચું) {
        ફેંકો std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_આગત_shape(અચલ ggml_tensor * op,
                                                      અચલ ggml_tensor * આગત,
                                                      પૂર્ણાંક dynamic_dim_index) અચલ {
    જો (m_naive) {
        પાછા આગત != nullptr ? ov::PartialShape{get_shape(આગત)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape આગત_shape;

    જો (is_inp_tok(આગત, op) || is_inp_pos(આગત, op)) {
        // tokens or positions
        પૂર્ણાંક len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        જો (m_is_static && is_inp_pos(આગત, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        આગત_shape = ov::PartialShape{1, 1, 1, len};

    } નહીંતર જો (is_output_idx(આગત, op)) {
        // output index
        આગત_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } નહીંતર જો (is_inp_mean(આગત, op)) {
        આગત_shape = m_is_static ? ov::PartialShape{1, 1, આગત->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } નહીંતર જો (is_inp_scale_rows(આગત, op)) {
        આગત_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } નહીંતર જો (is_inp_mask(આગત, op)) {
        // mask
        જો (m_is_static) {
            આગત_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } નહીંતર જો (m_is_stateful) {
            આગત_shape = ov::PartialShape{1, 1, -1, -1};
        } નહીંતર {
            આગત_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } નહીંતર જો (is_recurrent_cache(આગત)) {
        આગત_shape = ov::PartialShape{get_shape(આગત)};
        જો (!m_is_static && !m_is_stateful && આગત->ne[1] > 1) {
            આગત_shape[2] = -1;
        }

    } નહીંતર જો (is_kvcache(આગત, op)) {
        // kvcache
        આગત_shape = ov::PartialShape{get_shape(આગત)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        અચલ તાર્કિક is_flat_kv = (આગત->ne[1] == 1 && આગત->ne[2] == 1 && આગત->ne[3] == 1);
        જો (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            આગત_shape[2] = -1;
        }
        જો (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            પ્રતિજ્ઞા(આગત_shape.size() == 4 && આગત_shape[0] == 1 && આગત_shape[1] == 1 &&
                   આગત_shape[2].is_dynamic() && આગત_shape[3].is_static());
            અચલ પૂર્ણાંક n_heads_kv = get_n_heads_kv_for_tensor(આગત);
            પ્રતિજ્ઞા(n_heads_kv > 0 && આગત_shape[3].get_length() % n_heads_kv == 0);
            અચલ int64_t combined_dim = આગત_shape[3].get_length();  // n_heads_kv * head_size
            અચલ int64_t head_size = combined_dim / n_heads_kv;
            આગત_shape = {આગત_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } નહીંતર જો (is_kv_idx(આગત, op)) {
        // kv update index
        પૂર્ણાંક len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        આગત_shape = ov::PartialShape{1, 1, 1, len};

    } નહીંતર જો (is_inp_s_copy(આગત, op) || is_s_copy_leaf(આગત)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        આગત_shape = m_is_static ? ov::PartialShape{get_shape(આગત)} : ov::PartialShape{1, 1, 1, -1};

    } નહીંતર {
        આગત_shape = ov::PartialShape{get_shape(આગત)};
    }
    જો (dynamic_dim_index != -1 && m_model_is_splitted) {
        આગત_shape[3 - dynamic_dim_index] = -1;
    }
    જો (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_આગત && op->src[1] == આગત) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        જો (m_is_static) {
            અચલ int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            આગત_shape[2] = seq_active;
            આગત_shape[3] = seq_active;
        } નહીંતર {
            આગત_shape[2] = -1;
            આગત_shape[3] = -1;
        }
    }
    પાછા આગત_shape;
}

તાર્કિક GgmlOvDecoder::is_s_copy_leaf(અચલ ggml_tensor * tensor) અચલ {
    જો (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        પાછા ખોટું;
    }
    માટે (પૂર્ણાંક i = 0; i < m_cgraph->n_nodes; i++) {
        અચલ ggml_tensor * ગાંઠ = m_cgraph->nodes[i];
        જો (ગાંઠ->op != GGML_OP_GET_ROWS || ગાંઠ->src[0] == nullptr || ગાંઠ->src[1] == nullptr) {
            ચાલુ;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        અચલ ggml_tensor * idx = ગાંઠ->src[1];
        જ્યારે (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        જો (idx != tensor) {
            ચાલુ;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        અચલ ggml_tensor * data = ગાંઠ->src[0];
        જ્યારે (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        જો (data != nullptr && is_recurrent_cache(data)) {
            પાછા સાચું;
        }
    }
    પાછા ખોટું;
}

શૂન્ય GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_આગત = [આ](અચલ std::string & name, int64_t value, તાર્કિક force_parameter = ખોટું) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    જો (m_compute_params.attention_size != -1) {
        create_1d_આગત("attention_size", m_compute_params.attention_size);
    }
    જો (m_compute_params.attention_size_static != -1) {
        create_1d_આગત("attention_size_static", m_compute_params.attention_size_static);
    }
    જો (m_compute_params.attention_size_swa != -1) {
        create_1d_આગત("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    જો (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_આગત("swa_window", m_compute_params.swa_window);
    }
    create_1d_આગત("n_seq_active", m_compute_params.n_seq_active);
    create_1d_આગત("seq_active_start", m_compute_params.seq_active_start);
    create_1d_આગત("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    જો (m_compute_params.token_len_per_seq != -1) {
        create_1d_આગત("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    જો (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_આગત("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/સાચું);
        create_1d_આગત("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/સાચું);
    }

    જો (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_આગત("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        જો (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_આગત("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/સાચું);
        }
    }

    માટે (અચલ auto & [ગાંઠ_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_આગત("rs_slot_begin_" + ગાંઠ_name, writeback.slot_begin);
        જો (!m_is_static && writeback.src_begin >= 0) {
            create_1d_આગત("rs_src_begin_" + ગાંઠ_name, writeback.src_begin);
        }
    }
}

તાર્કિક GgmlOvDecoder::ગાંઠ_is_used_as_src(અચલ પૂર્ણાંક ગાંઠ_idx) {
    ggml_tensor * ગાંઠ = m_cgraph->nodes[ગાંઠ_idx];
    માટે (પૂર્ણાંક i = ગાંઠ_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_ગાંઠ = m_cgraph->nodes[i];
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            જો (other_ગાંઠ->src[j] == ગાંઠ) {
                પાછા સાચું;
            }
        }
    }
    પાછા ખોટું;
}

શૂન્ય GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    માટે (પૂર્ણાંક i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ગાંઠ = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        જો (ગાંઠ->op == GGML_OP_NONE && ગાંઠ_is_used_as_src(i)) {
            std::string ગાંઠ_name = get_tensor_ov_name(m_cgraph, ગાંઠ);
            જો (m_model_weights.find(ગાંઠ_name) == m_model_weights.end()) {
                m_inputs[ગાંઠ_name] = ગાંઠ;
                m_model_inputs[ગાંઠ_name] = {get_ov_type(ગાંઠ),
                                             get_graph_આગત_shape(ગાંઠ, nullptr, m_ગાંઠ_dynamic_dims[ગાંઠ])};
            }
            ચાલુ;
        }
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ગાંઠ->src[i];
            જો (src == nullptr) {
                ચાલુ;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            જો (src->flags & GGML_TENSOR_FLAG_આગત) {
                src_name = get_tensor_graph_આગત_ov_name(આ, m_cgraph, src, ગાંઠ);
            }
            જો (m_model_weights.find(src_name) != m_model_weights.end()) {
                ચાલુ;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            જો (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                ચાલુ;
            }

            તાર્કિક is_intermediate_ગાંઠ = ખોટું;
            માટે (અચલ auto & ગાંઠ_info : m_ગાંઠ_info_list) {
                જો (ગાંઠ_info.ગાંઠ == src) {
                    is_intermediate_ગાંઠ = સાચું;
                    તોડો;
                }
            }
            જો (is_intermediate_ગાંઠ) {
                ચાલુ;
            }
            જો (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                ચાલુ;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            જો (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                જો (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            જ્યારે (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_આગત_shape(ગાંઠ, src, m_ગાંઠ_dynamic_dims[src])};
        }
    }
}

શૂન્ય GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    માટે (પૂર્ણાંક ગાંઠ_n = 0; ગાંઠ_n < m_cgraph->n_nodes; ગાંઠ_n++) {
        auto * cur_ગાંઠ = m_cgraph->nodes[ગાંઠ_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        જો (cur_ગાંઠ->op == GGML_OP_NONE || cur_ગાંઠ->op == GGML_OP_VIEW || cur_ગાંઠ->op == GGML_OP_RESHAPE) {
            ચાલુ;
        }
        જો (::is_inplace_op(cur_ગાંઠ) && ggml_nbytes(cur_ગાંઠ) == 0) {
            ચાલુ;
        }
        auto cur_ગાંઠ_use_ગણતરી = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_ગાંઠ)];
        જો (cur_ગાંઠ_use_ગણતરી == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            જો (cur_ગાંઠ != nullptr && ::is_inplace_op(cur_ગાંઠ) && ggml_nbytes(cur_ગાંઠ) > 0) {
                cur_ગાંઠ = cur_ગાંઠ->view_src;
            }
        } નહીંતર {
            પૂર્ણાંક આગત_use_ગણતરી = 0;
            માટે (પૂર્ણાંક i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * ગાંઠ = m_cgraph->nodes[i];
                માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
                    જો (ગાંઠ->src[j] != ખાલી && ગાંઠ->src[j] == cur_ગાંઠ) {
                        આગત_use_ગણતરી++;
                    }
                }
            }
            જો (આગત_use_ગણતરી == cur_ગાંઠ_use_ગણતરી) {
                cur_ગાંઠ = nullptr;
            }
        }
        જો (cur_ગાંઠ != nullptr) {
            std::string cur_ગાંઠ_name = get_tensor_ov_name(m_cgraph, cur_ગાંઠ);
            m_model_outputs[cur_ગાંઠ_name] = cur_ગાંઠ;
            m_model_output_names.insert(cur_ગાંઠ_name);
        }
    }
}

અચલ ggml_tensor * GgmlOvDecoder::get_tensor_used_op(અચલ ggml_tensor * tensor) અચલ {
    જો (tensor == nullptr) {
        પાછા nullptr;
    }
    માટે (પૂર્ણાંક i = 0; i < m_cgraph->n_nodes; i++) {
        અચલ auto * ગાંઠ = m_cgraph->nodes[i];
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            જો (ગાંઠ->src[j] == tensor) {
                પાછા ગાંઠ;
            }
        }
    }
    પાછા nullptr;
}

અચલ ggml_tensor * GgmlOvDecoder::get_tensor_from_name(અચલ std::string & name) અચલ {
    માટે (પૂર્ણાંક i = 0; i < m_cgraph->n_nodes; i++) {
        અચલ auto * ગાંઠ = m_cgraph->nodes[i];
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            અચલ auto * src = ગાંઠ->src[j];
            જો (src == nullptr) {
                તોડો;
            }
            જો (get_tensor_ov_name(m_cgraph, src) == name) {
                પાછા src;
            }
        }
    }
    પાછા nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() અચલ {
    std::map<std::string, std::string> kv_param_res_names;
    માટે (અચલ auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    પાછા kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
સ્થિર તાર્કિક is_mul_mat_id_expert_weight(અચલ ggml_tensor * ગાંઠ, પૂર્ણાંક src_index) {
    પાછા ગાંઠ->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, તાર્કિક naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    માટે (પૂર્ણાંક ગાંઠ_i = 0; ગાંઠ_i < n_nodes; ગાંઠ_i++) {
        auto * ગાંઠ = nodes[ગાંઠ_i];
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ગાંઠ->src[i];
            જો (src == nullptr) {
                ચાલુ;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            જો (is_rope_freqs_weight(base, ગાંઠ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            જો (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(ગાંઠ, i)) {
                જો (model_weights.find(src_name) == model_weights.end()) {
                    auto weight_ગાંઠ = create_weight_ગાંઠ(base, naive);
                    weight_ગાંઠ->set_friendly_name(src_name);
                    model_weights[src_name] = weight_ગાંઠ;
                }
            }
        }
    }
    પાછા model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
સ્થિર std::mutex g_nonov_weight_cache_mutex;
સ્થિર std::unordered_map<અચલ શૂન્ય *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    માટે (પૂર્ણાંક ગાંઠ_i = 0; ગાંઠ_i < cgraph->n_nodes; ગાંઠ_i++) {
        auto * ગાંઠ = cgraph->nodes[ગાંઠ_i];
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = ગાંઠ->src[i];
            જો (src == nullptr) {
                ચાલુ;
            }
            અચલ ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            જો (is_rope_freqs_weight(base, ગાંઠ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            જો (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    પાછા names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_ગાંઠ(ggml_tensor * tensor, તાર્કિક naive) {
    અચલ તાર્કિક is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    જો (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        જો (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            જો (weight_extra->weight_ગાંઠ) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                પાછા weight_extra->weight_ગાંઠ;
            }
        } નહીંતર જો (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            જો (quant_extra->weight_ગાંઠ) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                પાછા quant_extra->weight_ગાંઠ;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    જો (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        auto weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        auto weight_ગાંઠ = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_ગાંઠ->set_friendly_name(tensor->name);
        પાછા weight_ગાંઠ;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    અચલ તાર્કિક cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    જો (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        auto it = g_nonov_weight_cache.find(tensor->data);
        જો (it != g_nonov_weight_cache.end()) {
            પાછા it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    સ્થિર અચલ std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    જો (weight_types.find(tensor->type) == weight_types.end()) {
        ફેંકો std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    જો (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        જો (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } નહીંતર {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } નહીંતર {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_ગાંઠ->set_friendly_name(tensor->name);
    જો (!is_ov_buffer) {
        જો (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            auto [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_ગાંઠ);
            પાછા it->second;
        }
        પાછા ov_weight.weight_ગાંઠ;
    }

    ggml_openvino_extra_base * extra;
    જો (ov_weight.is_quantized()) {
        extra = નવું ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_ગાંઠ);
    } નહીંતર {
        extra = નવું ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_ગાંઠ);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    પાછા ov_weight.weight_ગાંઠ;
}

શૂન્ય GgmlOvDecoder::dump_cgraph(અચલ ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    જો (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        પાછા;
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
    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ગાંઠ = cgraph->nodes[i];

        // Get buffer type name
        અચલ અક્ષર * buf_name = "none";
        ggml_backend_buffer_t buf = ગાંઠ->view_src ? ગાંઠ->view_src->buffer : ગાંઠ->buffer;
        જો (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ગાંઠ->ne[0] << ", "
             << std::setw(5) << ગાંઠ->ne[1] << ", "
             << std::setw(5) << ગાંઠ->ne[2] << ", "
             << std::setw(5) << ગાંઠ->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(ગાંઠ->op) << std::right << " "
             << std::left << std::setw(45) << ગાંઠ->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << ગાંઠ->nb[0] << ", "
             << std::setw(5) << ગાંઠ->nb[1] << ", "
             << std::setw(5) << ગાંઠ->nb[2] << ", "
             << std::setw(5) << ગાંઠ->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            જો (auto* src = ગાંઠ->src[i]) {
                // Get buffer type name for source
                અચલ અક્ષર * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                જો (src_buf) {
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
    માટે (પૂર્ણાંક i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * ગાંઠ = cgraph->leafs[i];

        // Get buffer type name for leaf
        અચલ અક્ષર * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = ગાંઠ->view_src ? ગાંઠ->view_src->buffer : ગાંઠ->buffer;
        જો (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ગાંઠ->ne[0] << ", "
             << std::setw(5) << ગાંઠ->ne[1] << "] "
             << std::setw(8) << ggml_op_name(ગાંઠ->op) << " "
             << std::setw(16) << ggml_get_name(ગાંઠ)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

શૂન્ય print_tensor_address_map(અચલ ggml_cgraph * cgraph) {
    std::map<શૂન્ય *, std::vector<std::string>> address_map;
    માટે (પૂર્ણાંક ગાંઠ_n = 0; ગાંઠ_n < cgraph->n_nodes; ગાંઠ_n++) {
        auto * ગાંઠ = cgraph->nodes[ગાંઠ_n];
        જો (ગાંઠ->data) {
            auto it = address_map.find(ગાંઠ->data);
            જો (it == address_map.end()) {
                address_map[ગાંઠ->data] = std::vector<std::string>();
            }
            address_map[ગાંઠ->data].push_back(ગાંઠ->name);
        }
    }
    માટે (અચલ auto & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        માટે (અચલ auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(અચલ ggml_tensor * tensor) {
    std::vector<size_t> shape;
    માટે (પૂર્ણાંક i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    પાછા shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(અચલ ggml_tensor * tensor) {
    std::vector<size_t> stride;
    માટે (પૂર્ણાંક i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    પાછા stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(અચલ ggml_tensor * tensor) {
    પસંદગી (tensor->type) {
    સ્થિતિ GGML_TYPE_F64:
        પાછા ov::element::f64;
    સ્થિતિ GGML_TYPE_F32:
        પાછા ov::element::f32;
    સ્થિતિ GGML_TYPE_F16:
        પાછા ov::element::f16;
    સ્થિતિ GGML_TYPE_BF16:
        પાછા ov::element::bf16;
    સ્થિતિ GGML_TYPE_I8:
        પાછા ov::element::i8;
    સ્થિતિ GGML_TYPE_I16:
        પાછા ov::element::i16;
    સ્થિતિ GGML_TYPE_I32:
        પાછા ov::element::i32;
    સ્થિતિ GGML_TYPE_I64:
        પાછા ov::element::i64;
    મૂળભૂત:
        પાછા ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_આગત_shape(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ {
    પાછા ov::PartialShape(get_shape(m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_આગત_stride(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ {
    પાછા get_stride(m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_આગત_size(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        પાછા it->second.size();
    }
    પાછા 0;
}

size_t GgmlOvDecoder::get_view_આગત_offset(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        જો (view_index < it->second.size()) {
            પાછા it->second[view_index].second->view_offs;
        }
    }
    પાછા 0;
}

size_t GgmlOvDecoder::get_view_આગત_src_offset(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        જો (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            જો (view_tensor && view_tensor->src[0]) {
                પાછા view_tensor->src[0]->view_offs;
            }
        }
    }
    પાછા 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_આગત_stride(પૂર્ણાંક ગાંઠ_idx,
                                                         અચલ std::string & name,
                                                         size_t view_index) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        જો (view_index < it->second.size()) {
            પાછા get_stride(it->second[view_index].second);
        }
    }
    પાછા {};
}

std::vector<size_t> GgmlOvDecoder::get_view_આગત_src_stride(પૂર્ણાંક ગાંઠ_idx,
                                                             અચલ std::string & name,
                                                             size_t view_index) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        જો (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            જો (view_tensor && view_tensor->src[0]) {
                પાછા get_stride(view_tensor->src[0]);
            }
        }
    }
    પાછા {};
}

ov::Shape GgmlOvDecoder::get_view_આગત_ggml_shape(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        જો (view_index < it->second.size()) {
            પાછા get_shape(it->second[view_index].second);
        }
    }
    પાછા {};
}

ov::Shape GgmlOvDecoder::get_view_આગત_src_ggml_shape(પૂર્ણાંક ગાંઠ_idx,
                                                       અચલ std::string & name,
                                                       size_t view_index) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        જો (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            જો (view_tensor && view_tensor->src[0]) {
                પાછા get_shape(view_tensor->src[0]);
            }
        }
    }
    પાછા {};
}

ov::PartialShape GgmlOvDecoder::get_view_આગત_ov_shape(પૂર્ણાંક ગાંઠ_idx,
                                                        અચલ std::string & name,
                                                        size_t view_index) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        જો (view_index < it->second.size()) {
            auto * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            auto dynamic_it = m_ગાંઠ_dynamic_dims.find(tensor);
            જો (dynamic_it != m_ગાંઠ_dynamic_dims.end() && dynamic_it->second != -1) {
                પૂર્ણાંક dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            પાછા shape;
        }
    }
    પાછા {};
}

ov::PartialShape GgmlOvDecoder::get_view_આગત_src_ov_shape(પૂર્ણાંક ગાંઠ_idx,
                                                            અચલ std::string & name,
                                                            size_t view_index) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        જો (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            જો (view_tensor && view_tensor->src[0]) {
                auto * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                auto dynamic_it = m_ગાંઠ_dynamic_dims.find(src_tensor);
                જો (dynamic_it != m_ગાંઠ_dynamic_dims.end() && dynamic_it->second != -1) {
                    પૂર્ણાંક dynamic_dim_index = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
                }

                પાછા shape;
            }
        }
    }
    પાછા {};
}

std::string GgmlOvDecoder::get_view_આગત_name(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        જો (view_index < it->second.size()) {
            પાછા it->second[view_index].first;
        }
    }
    પાછા "";
}

std::string GgmlOvDecoder::get_view_આગત_src_name(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ {
    auto it = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.find(name);
    જો (it != m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_views.end()) {
        જો (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            જો (view_tensor && view_tensor->src[0]) {
                પાછા get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    પાછા "";
}

ov::element::Type GgmlOvDecoder::get_આગત_type(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ {
    પાછા get_ov_type(m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs.at(name));
}

size_t GgmlOvDecoder::get_આગત_size() અચલ {
    પાછા m_model_inputs.size();
}

size_t GgmlOvDecoder::get_આગત_size(પૂર્ણાંક ગાંઠ_idx) અચલ {
    પાછા m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_આગત_names(પૂર્ણાંક ગાંઠ_idx) અચલ {
    પાછા m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(પૂર્ણાંક ગાંઠ_idx) અચલ {
    auto * ggml_tensor = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ;
    પાછા ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(અચલ પૂર્ણાંક ગાંઠ_idx) અચલ {
    પાછા get_ov_type(m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(પૂર્ણાંક ગાંઠ_idx) અચલ {
    auto * ggml_tensor = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ;
    પાછા get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(પૂર્ણાંક ગાંઠ_idx) અચલ {
    auto * ગાંઠ = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ;
    જો (ગાંઠ->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        માટે (પૂર્ણાંક i = ગાંઠ_idx + 1; i < m_cgraph->n_nodes; i++) {
            auto * consumer = m_cgraph->nodes[i];
            જો (consumer->op != GGML_OP_VIEW || consumer->src[0] != ગાંઠ) {
                ચાલુ;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            auto & name = consumer->view_offs == 0 ? attn_name : state_name;
            જો (!name.empty()) {
                પાછા {m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        જો (!attn_name.empty() && !state_name.empty()) {
            પાછા {attn_name, state_name};
        }
    }
    પાછા {m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(પૂર્ણાંક ગાંઠ_idx) અચલ {
    auto * ગાંઠ = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ;
    જો (!::is_inplace_op(ગાંઠ) || ગાંઠ->view_src == nullptr || ggml_nbytes(ગાંઠ) == 0) {
        પાછા "";
    }
    અચલ પૂર્ણાંક op_case = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_op_case;
    જો (ગાંઠ->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        પાછા "";
    }
    પાછા get_tensor_ov_name(m_cgraph, ગાંઠ->view_src);
}

તાર્કિક GgmlOvDecoder::is_view_like_alias_of(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & view_src_name) અચલ {
    auto * ગાંઠ = m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ;
    જો (ગાંઠ->view_src == nullptr || get_tensor_ov_name(m_cgraph, ગાંઠ->view_src) != view_src_name) {
        પાછા ખોટું;
    }
    પાછા ગાંઠ->op == GGML_OP_RESHAPE || ગાંઠ->op == GGML_OP_VIEW;
}

અચલ std::string & GgmlOvDecoder::get_op_name() અચલ {
    સ્થિર અચલ std::string unknown_name = "UNKNOWN_OP_NAME";
    પાછા unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(પૂર્ણાંક ગાંઠ_idx) અચલ {
    auto it = m_ગાંઠ_dynamic_dims.find(m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ);
    જો (it == m_ગાંઠ_dynamic_dims.end()) {
        પાછા -1;
    }
    પાછા it->second;
}

અચલ std::string & GgmlOvDecoder::get_op_name(પૂર્ણાંક ગાંઠ_idx) અચલ {
    પાછા m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_name;
}

int32_t * GgmlOvDecoder::get_આગત_op_params(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ {
    પાછા m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(પૂર્ણાંક ગાંઠ_idx) અચલ {
    પાછા m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ->op_params;
}

size_t GgmlOvDecoder::get_output_op_offset(પૂર્ણાંક ગાંઠ_idx) અચલ {
    પાછા m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ->view_offs;
}

શૂન્ય GgmlOvDecoder::visit_subgraph(std::function<શૂન્ય(std::shared_ptr<GgmlDecoder>, પૂર્ણાંક ગાંઠ_idx)> ગાંઠ_visitor) અચલ {
    માટે (પૂર્ણાંક ગાંઠ_idx = 0; ગાંઠ_idx < m_cgraph->n_nodes; ગાંઠ_idx++) {
        જો (m_cgraph->nodes[ગાંઠ_idx]->op == GGML_OP_NONE) {
            ચાલુ;
        }
        ગાંઠ_visitor(std::make_shared<GgmlOvDecoder>(*આ), ગાંઠ_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(અચલ ggml_tensor * ગાંઠ) {
    પસંદગી (ગાંઠ->op) {
    સ્થિતિ GGML_OP_UNARY:
        પાછા std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(ગાંઠ));
    સ્થિતિ GGML_OP_GLU:
        પાછા std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(ગાંઠ));
    મૂળભૂત:
        પાછા std::string("GGML_OP_") + ggml_op_name(ગાંઠ->op);
    }
}

અચલ std::string & GgmlOvDecoder::get_op_type(પૂર્ણાંક ગાંઠ_idx) અચલ {
    પાછા m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_op_type;
}

અચલ std::string & GgmlOvDecoder::get_op_type() અચલ {
    સ્થિર અચલ std::string unknown_op = "UNKNOWN_GGML_OP";
    પાછા unknown_op;
}

શૂન્ય GgmlOvDecoder::compute_ગાંઠ_dynamic_dims() {
    auto visit_ગાંઠ = [&](auto && self, ggml_tensor * ગાંઠ) -> શૂન્ય {
        જો (!ગાંઠ) {
            પાછા;
        }

        જો (ગાંઠ->op == GGML_OP_CPY) {
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
        }

        જો (m_ગાંઠ_dynamic_dims.count(ગાંઠ)) {
            પાછા;
        }
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = ગાંઠ->src[i];
            જો (src == nullptr) {
                ચાલુ;
            }
            સંરચના ggml_tensor * મૂળ_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            જો (મૂળ_src) {
                જો (is_inp_tok(મૂળ_src, ગાંઠ) || is_inp_pos(મૂળ_src, ગાંઠ) || is_output_idx(મૂળ_src, ગાંઠ)) {
                    m_ગાંઠ_dynamic_dims[મૂળ_src] = 0;
                    m_ગાંઠ_dynamic_dims[src] = m_ગાંઠ_dynamic_dims[મૂળ_src];
                    ચાલુ;
                }
                self(self, મૂળ_src);
                m_ગાંઠ_dynamic_dims[src] = m_ગાંઠ_dynamic_dims[મૂળ_src];
            } નહીંતર {
                જો (is_inp_tok(src, ગાંઠ) || is_inp_pos(src, ગાંઠ) || is_output_idx(src, ગાંઠ)) {
                    m_ગાંઠ_dynamic_dims[src] = 0;
                    ચાલુ;
                }
                જો (is_inp_scale_rows(src, ગાંઠ)) {
                    m_ગાંઠ_dynamic_dims[src] = 1;
                    ચાલુ;
                }
                જો (ગાંઠ->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_ગાંઠ_dynamic_dims[src] = 1;
                    ચાલુ;
                }
                self(self, src);
            }
        }
        પસંદગી (ગાંઠ->op) {
        સ્થિતિ GGML_OP_NONE:
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            તોડો;
        સ્થિતિ GGML_OP_GET_ROWS:
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            જો (m_ગાંઠ_dynamic_dims[ગાંઠ->src[1]] != -1) {
                auto dynamic_dim_idx = m_ગાંઠ_dynamic_dims[ગાંઠ->src[1]];
                જો (dynamic_dim_idx == 0) {
                    m_ગાંઠ_dynamic_dims[ગાંઠ] = 1;
                } નહીંતર {
                    auto dynamic_dim_stride = ગાંઠ->src[1]->nb[dynamic_dim_idx] / ggml_type_size(ગાંઠ->src[1]->type) *
                                              ggml_type_size(ગાંઠ->src[0]->type);
                    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
                        જો (dynamic_dim_stride == ગાંઠ->src[0]->nb[i]) {
                            m_ગાંઠ_dynamic_dims[ગાંઠ] = i;
                            તોડો;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            તોડો;
        સ્થિતિ GGML_OP_MUL:
        સ્થિતિ GGML_OP_MUL_MAT:
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            જો (m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]] != -1) {
                m_ગાંઠ_dynamic_dims[ગાંઠ] = m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]];
            }
            જો (m_ગાંઠ_dynamic_dims[ગાંઠ->src[1]] != -1) {
                m_ગાંઠ_dynamic_dims[ગાંઠ] = m_ગાંઠ_dynamic_dims[ગાંઠ->src[1]];
            }
            તોડો;
        સ્થિતિ GGML_OP_PERMUTE:
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            જો (m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]] != -1) {
                auto dynamic_dim_idx = m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
                    જો (ગાંઠ->op_params[i] == dynamic_dim_idx) {
                        m_ગાંઠ_dynamic_dims[ગાંઠ] = i;
                        તોડો;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            તોડો;
        સ્થિતિ GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            જો (m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]] != -1) {
                જો (ગાંઠ->src[0]->op == GGML_OP_NONE) {
                    m_ગાંઠ_dynamic_dims[ગાંઠ] = m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]];
                    તોડો;
                }
                auto dynamic_dim_idx = m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]];
                auto dynamic_dim_કિંમત = ગાંઠ->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    ગાંઠ->src[0]->nb[dynamic_dim_idx] / ggml_type_size(ગાંઠ->src[0]->type) * ggml_type_size(ગાંઠ->type);
                માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
                    જો (ગાંઠ->nb[i] == dynamic_dim_stride) {
                        m_ગાંઠ_dynamic_dims[ગાંઠ] = i;
                        તોડો;
                    }
                }
                જો (m_ગાંઠ_dynamic_dims[ગાંઠ] != -1 && dynamic_dim_કિંમત != ગાંઠ->ne[m_ગાંઠ_dynamic_dims[ગાંઠ]]) {
                    m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    જો (ggml_nelements(ગાંઠ) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      ગાંઠ->name, ગાંઠ->src[0]->name);
                    }
                }
            }
            તોડો;
        }
        સ્થિતિ GGML_OP_TRANSPOSE:
        સ્થિતિ GGML_OP_RESHAPE: {
            જો (is_same_shape(ગાંઠ->src[0], ગાંઠ)) {
                m_ગાંઠ_dynamic_dims[ગાંઠ] = m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]];
                તોડો;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            જો (m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]] != -1) {
                auto dynamic_dim_idx = m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]];
                auto dynamic_dim_stride = ગાંઠ->src[0]->nb[dynamic_dim_idx];
                માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
                    જો (ગાંઠ->nb[i] == dynamic_dim_stride && ગાંઠ->ne[i] == ગાંઠ->src[0]->ne[dynamic_dim_idx]) {
                        m_ગાંઠ_dynamic_dims[ગાંઠ] = i;
                        તોડો;
                    }
                }
                જો (m_ગાંઠ_dynamic_dims[ગાંઠ] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", ગાંઠ->name);
                }
            }
            તોડો;
        }
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr પૂર્ણાંક q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            જો (m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]] != -1) {
                auto q_dynamic_dim = m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]];
                m_ગાંઠ_dynamic_dims[ગાંઠ] = q_to_out[q_dynamic_dim];
            }
            તોડો;
        }
        સ્થિતિ GGML_OP_CONT:
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            જો (m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]] != -1) {
                auto dynamic_dim_idx = m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]];
                જો (ggml_are_same_shape(ગાંઠ, ગાંઠ->src[0])) {
                    m_ગાંઠ_dynamic_dims[ગાંઠ] = dynamic_dim_idx;
                } નહીંતર {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(ગાંઠ->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (ગાંઠ->src[0]->ne[0] / ggml_blck_size(ગાંઠ->src[0]->type));
                    માટે (પૂર્ણાંક i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * ગાંઠ->src[0]->ne[i - 1];
                    }

                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(ગાંઠ->src[0]->type) *
                                              ggml_type_size(ગાંઠ->type);
                    પૂર્ણાંક matched_dim_ગણતરી = 0;
                    પૂર્ણાંક first_matched_dim = -1;
                    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
                        જો (ગાંઠ->nb[i] == dynamic_dim_stride && ગાંઠ->ne[i] == ગાંઠ->src[0]->ne[dynamic_dim_idx]) {
                            જો (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_ગાંઠ_dynamic_dims[ગાંઠ] = i;
                            matched_dim_ગણતરી++;
                        }
                    }
                    જો (matched_dim_ગણતરી > 1 && ગાંઠ->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_ગાંઠ_dynamic_dims[ગાંઠ] = first_matched_dim;
                    } નહીંતર જો (matched_dim_ગણતરી != 1) {
                        m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      ગાંઠ->name, ગાંઠ->src[0]->name);
                    }
                }
            }
            તોડો;
        સ્થિતિ GGML_OP_CONCAT:
            માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
                જો (ગાંઠ->src[0]->ne[i] != ગાંઠ->ne[i]) {
                    m_ગાંઠ_dynamic_dims[ગાંઠ] = i;
                    તોડો;
                }
            }
            તોડો;
        સ્થિતિ GGML_OP_SSM_CONV:
        સ્થિતિ GGML_OP_GATED_DELTA_NET:
            m_ગાંઠ_dynamic_dims[ગાંઠ] = 1;
            તોડો;
        સ્થિતિ GGML_OP_RMS_NORM:
        સ્થિતિ GGML_OP_L2_NORM:
        સ્થિતિ GGML_OP_NORM:
        સ્થિતિ GGML_OP_ADD:
        સ્થિતિ GGML_OP_SUB:
        સ્થિતિ GGML_OP_GLU:
        સ્થિતિ GGML_OP_ROPE:
        સ્થિતિ GGML_OP_SCALE:
        સ્થિતિ GGML_OP_SOFT_MAX:
        સ્થિતિ GGML_OP_ARGSORT:
        સ્થિતિ GGML_OP_ADD_ID:
        સ્થિતિ GGML_OP_UNARY:
        સ્થિતિ GGML_OP_CUMSUM:
        સ્થિતિ GGML_OP_FILL:
        સ્થિતિ GGML_OP_SET:
        સ્થિતિ GGML_OP_DIAG:
        સ્થિતિ GGML_OP_TRI:
        સ્થિતિ GGML_OP_REPEAT:
        સ્થિતિ GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        સ્થિતિ GGML_OP_DIV:
        સ્થિતિ GGML_OP_CLAMP:
        સ્થિતિ GGML_OP_PAD:
        સ્થિતિ GGML_OP_UPSCALE:
        સ્થિતિ GGML_OP_SIN:
        સ્થિતિ GGML_OP_COS:
        સ્થિતિ GGML_OP_LOG:
        સ્થિતિ GGML_OP_ROLL:
            m_ગાંઠ_dynamic_dims[ગાંઠ] = m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]];
            તોડો;
        સ્થિતિ GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_ગાંઠ_dynamic_dims[ગાંઠ] =
                (m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]] == 0) ? -1 : m_ગાંઠ_dynamic_dims[ગાંઠ->src[0]];
            તોડો;
        સ્થિતિ GGML_OP_MUL_MAT_ID:
        સ્થિતિ GGML_OP_SOLVE_TRI:
            m_ગાંઠ_dynamic_dims[ગાંઠ] = m_ગાંઠ_dynamic_dims[ગાંઠ->src[1]];
            તોડો;
        સ્થિતિ GGML_OP_CPY:
        સ્થિતિ GGML_OP_SET_ROWS:
        સ્થિતિ GGML_OP_SUM:
        સ્થિતિ GGML_OP_MEAN:
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            તોડો;
        સ્થિતિ GGML_OP_IM2COL: {
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            જો (m_ગાંઠ_dynamic_dims[ગાંઠ->src[1]] != -1) {
                અચલ તાર્કિક is_2D = ગાંઠ->op_params[6] == 1;
                અચલ પૂર્ણાંક src_dyn = m_ગાંઠ_dynamic_dims[ગાંઠ->src[1]];
                જો (is_2D) {
                    જો (src_dyn == 0) {
                        m_ગાંઠ_dynamic_dims[ગાંઠ] = 1;  // IW -> OW
                    } નહીંતર જો (src_dyn == 1) {
                        m_ગાંઠ_dynamic_dims[ગાંઠ] = 2;  // IH -> OH
                    } નહીંતર જો (src_dyn == 3) {
                        m_ગાંઠ_dynamic_dims[ગાંઠ] = 3;  // N  -> N
                    }
                } નહીંતર {
                    જો (src_dyn == 0) {
                        m_ગાંઠ_dynamic_dims[ગાંઠ] = 1;  // IW -> OW
                    } નહીંતર જો (src_dyn == 2) {
                        m_ગાંઠ_dynamic_dims[ગાંઠ] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                જો (m_ગાંઠ_dynamic_dims[ગાંઠ] != -1) {
                    OPENVINO_ASSERT(ગાંઠ->src[1]->ne[src_dyn] == ગાંઠ->ne[m_ગાંઠ_dynamic_dims[ગાંઠ]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(ગાંઠ->name) +
                                        " and its src[1]: " + std::string(ગાંઠ->src[1]->name));
                }
            }
            તોડો;
        }
        સ્થિતિ GGML_OP_IM2COL_3D: {
            m_ગાંઠ_dynamic_dims[ગાંઠ] = -1;
            જો (m_ગાંઠ_dynamic_dims[ગાંઠ->src[1]] != -1) {
                અચલ પૂર્ણાંક src_dyn = m_ગાંઠ_dynamic_dims[ગાંઠ->src[1]];
                જો (src_dyn == 0) {
                    m_ગાંઠ_dynamic_dims[ગાંઠ] = 1;  // IW -> OW
                } નહીંતર જો (src_dyn == 1) {
                    m_ગાંઠ_dynamic_dims[ગાંઠ] = 2;  // IH -> OH
                } નહીંતર જો (src_dyn == 3) {
                    m_ગાંઠ_dynamic_dims[ગાંઠ] = 3;  // N  -> N
                }
                જો (m_ગાંઠ_dynamic_dims[ગાંઠ] != -1) {
                    OPENVINO_ASSERT(ગાંઠ->src[1]->ne[src_dyn] == ગાંઠ->ne[m_ગાંઠ_dynamic_dims[ગાંઠ]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(ગાંઠ->name) +
                                        " and its src[1]: " + std::string(ગાંઠ->src[1]->name));
                }
            }
            તોડો;
        }
        મૂળભૂત:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(ગાંઠ->op), ગાંઠ->name);
            તોડો;
        }
    };

    માટે (પૂર્ણાંક i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ગાંઠ = m_cgraph->nodes[i];
        visit_ગાંઠ(visit_ગાંઠ, ગાંઠ);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    જો (0) {
        માટે (પૂર્ણાંક i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * ગાંઠ = m_cgraph->nodes[i];
            પૂર્ણાંક dynamic_dim = m_ગાંઠ_dynamic_dims[ગાંઠ];
            std::cout << "[" << i << "] " << "node_name: " << ગાંઠ->name << " op: " << ggml_op_name(ગાંઠ->op)
                      << " shape: [";
            માટે (પૂર્ણાંક j = 0; j < 4; j++) {
                જો (j == dynamic_dim) {
                    std::cout << "*";
                } નહીંતર {
                    std::cout << ગાંઠ->ne[j];
                }
                જો (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = ગાંઠ->src[j];
                જો (src == nullptr) {
                    ચાલુ;
                }
                પૂર્ણાંક src_dynamic_dim = m_ગાંઠ_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                માટે (પૂર્ણાંક k = 0; k < 4; k++) {
                    જો (k == src_dynamic_dim) {
                        std::cout << "*";
                    } નહીંતર {
                        std::cout << src->ne[k];
                    }
                    જો (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
