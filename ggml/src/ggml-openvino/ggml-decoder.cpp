#అన్తర్భావ "ggml-decoder.h"

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-openvino-extra.h"
#అన్తర్భావ "ggml-openvino.h"
#అన్తర్భావ "ggml-quants.h"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "utils.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cassert>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <fstream>
#అన్తర్భావ <iomanip>
#అన్తర్భావ <map>
#అన్తర్భావ <memory>
#అన్తర్భావ <mutex>
#అన్తర్భావ <openvino/core/dimension.hpp>
#అన్తర్భావ <openvino/core/except.hpp>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/partial_shape.hpp>
#అన్తర్భావ <openvino/core/type/bfloat16.hpp>
#అన్తర్భావ <openvino/core/type/element_type.hpp>
#అన్తర్భావ <openvino/core/type/float16.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/runtime/tensor.hpp>
#అన్తర్భావ <ostream>
#అన్తర్భావ <set>
#అన్తర్భావ <stdexcept>
#అన్తర్భావ <string>
#అన్తర్భావ <cstring>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             తార్కిక is_static,
                             తార్కిక is_stateful,
                             తార్కిక model_is_splitted,
                             తార్కిక is_prefill,
                             పూర్ణాంకం prefill_chunk_కొలత) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(అబద్ధం),
    m_prefill_chunk_కొలత(prefill_chunk_కొలత),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    స్థిర తార్కిక printed_చిరునామా_map = అబద్ధం;
    ఒకవేళ (!printed_చిరునామా_map) {
        ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_చిరునామా_map = నిజం;
            print_tensor_చిరునామా_map(cgraph);
        }
    }

    validate_cgraph();

    సమితి_ఆదానం_output();
    compute_కణుపు_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    కొరకు (పూర్ణాంకం కణుపు_n = 0; కణుపు_n < cgraph->n_nodes; కణుపు_n++) {
        m_కణుపు_info_list[కణుపు_n].కణుపు_op_case = compute_op_case(m_కణుపు_info_list[కణుపు_n].కణుపు);
        m_కణుపు_info_list[కణుపు_n].కణుపు_op_type = compute_op_type(m_కణుపు_info_list[కణుపు_n].కణుపు);
    }

    కూడు_extra_inputs();
}

శూన్య GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_కణుపు_info_list.clear();
    సమితి_ఆదానం_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
తార్కిక GgmlOvDecoder::is_bound_గమ్యం(స్థిరరాశి ggml_cgraph * cgraph) స్థిరరాశి {
    తిరిగి m_cgraph == cgraph && cgraph->n_nodes > 0 && m_కణుపు_info_list.size() == (size_t) cgraph->n_nodes &&
           m_కణుపు_info_list.front().కణుపు == cgraph->nodes[0] &&
           m_కణుపు_info_list.back().కణుపు == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = నిజం;
    సమితి_ఆదానం_output();
    compute_model_inputs();
    compute_model_outputs();
    కొరకు (పూర్ణాంకం కణుపు_n = 0; కణుపు_n < cgraph->n_nodes; కణుపు_n++) {
        m_కణుపు_info_list[కణుపు_n].కణుపు_op_case = compute_op_case(m_కణుపు_info_list[కణుపు_n].కణుపు);
        m_కణుపు_info_list[కణుపు_n].కణుపు_op_type = compute_op_type(m_కణుపు_info_list[కణుపు_n].కణుపు);
    }
}

నామాకాశ {
తార్కిక is_inplace_op(స్థిరరాశి ggml_tensor * కణుపు) {
    తిరిగి కణుపు->op == GGML_OP_సమితి_అడ్డువరుసలు || కణుపు->op == GGML_OP_CPY || (కణుపు->op == GGML_OP_SCALE && కణుపు->view_src);
}

తార్కిక is_same_shape(స్థిరరాశి ggml_tensor * a, స్థిరరాశి ggml_tensor * b) {
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
        ఒకవేళ (a->ne[i] != b->ne[i]) {
            తిరిగి అబద్ధం;
        }
    }
    తిరిగి నిజం;
}

తార్కిక is_conv_స్థితులు_all_tensor(స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి tensor != శూన్యసూచిక && అక్షరశ్రేణి_పోలిక(tensor->name, "conv_states_all", అక్షరశ్రేణి_పొడవు("conv_states_all")) == 0;
}

తార్కిక is_నిండిన_single_స్థానం_writeback(స్థిరరాశి ggml_tensor * కణుపు) {
    తిరిగి కణుపు->view_src != శూన్యసూచిక && కణుపు->view_src->ne[1] == 1 && కణుపు->src[1] != శూన్యసూచిక &&
           కణుపు->src[1]->op == GGML_OP_VIEW && కణుపు->src[1]->view_src == కణుపు->view_src &&
           కణుపు->src[1]->view_offs == 0 && ggml_nbytes(కణుపు->src[1]) == ggml_nbytes(కణుపు->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
తార్కిక is_moe_expert_కూడిక_కూడు(స్థిరరాశి ggml_tensor * కణుపు) {
    std::vector<స్థిరరాశి ggml_tensor *> leaves;
    స్థిరరాశి ggml_tensor * cur = కణుపు;
    అయితే (cur->op == GGML_OP_కూడు) {
        ఒకవేళ (cur->src[0] == శూన్యసూచిక || cur->src[1] == శూన్యసూచిక) {
            తిరిగి అబద్ధం;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    స్థిరరాశి ggml_tensor * పునాది = శూన్యసూచిక;
    std::set<int64_t> plane_indices;
    కొరకు (స్థిరరాశి ggml_tensor * ఆకు : leaves) {
        ఒకవేళ (ఆకు->op != GGML_OP_VIEW || ఆకు->src[0] == శూన్యసూచిక) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి ggml_tensor * ఆకు_పునాది = ఆకు->src[0];
        ఒకవేళ (పునాది == శూన్యసూచిక) {
            పునాది = ఆకు_పునాది;
        } లేకపోతే ఒకవేళ (ఆకు_పునాది != పునాది) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (ఆకు->ne[0] != పునాది->ne[0] || ఆకు->ne[1] != పునాది->ne[2] || ఆకు->ne[2] != 1 || ఆకు->ne[3] != 1 ||
            ఆకు->nb[1] != పునాది->nb[2]) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (పునాది->nb[1] == 0 || ఆకు->view_offs % పునాది->nb[1] != 0) {
            తిరిగి అబద్ధం;
        }
        int64_t plane = static_cast<int64_t>(ఆకు->view_offs / పునాది->nb[1]);
        ఒకవేళ (plane < 0 || plane >= పునాది->ne[1] || !plane_indices.insert(plane).second) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి పునాది != శూన్యసూచిక && పునాది->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(పునాది->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(స్థిరరాశి ggml_cgraph * cgraph, స్థిరరాశి ggml_tensor * tensor) {
    ఒకవేళ (tensor == శూన్యసూచిక) {
        తిరిగి "";
    }
    ఒకవేళ ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, శూన్యసూచిక)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        స్థిరరాశి స్వయంచాలక * కణుపు = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        ఒకవేళ (కణుపు != cgraph->nodes + cgraph->n_nodes) {
            తిరిగి std::string(tensor->name) + "#n" + std::to_string(కణుపు - cgraph->nodes);
        }
        స్థిరరాశి స్వయంచాలక * ఆకు = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        ఒకవేళ (ఆకు != cgraph->leafs + cgraph->n_leafs) {
            తిరిగి std::string(tensor->name) + "#l" + std::to_string(ఆకు - cgraph->leafs);
        }
    }
    తిరిగి tensor->name;
}

స్థిర std::string get_tensor_ov_name(స్థిరరాశి ggml_cgraph * cgraph, స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

స్థిర std::string get_tensor_లేఖాచిత్రం_ఆదానం_ov_name(స్థిరరాశి GgmlOvDecoder * decoder,
                                                  స్థిరరాశి ggml_cgraph * cgraph,
                                                  స్థిరరాశి ggml_tensor * tensor,
                                                  స్థిరరాశి ggml_tensor * op) {
    ఒకవేళ (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        తిరిగి "inp_pos";
    }
    ఒకవేళ (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        తిరిగి "embd";
    }
    ఒకవేళ (GgmlOvDecoder::is_inp_ముసుగు(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        స్థిరరాశి తార్కిక is_swa = decoder->is_swa_ముసుగు(tensor);
        ఒకవేళ (decoder->is_stateful()) {
            తిరిగి is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        ఒకవేళ (is_swa) {
            తిరిగి get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    తిరిగి get_tensor_ov_name(cgraph, tensor);
}

శూన్య GgmlOvDecoder::సమితి_ఆదానం_output() {
    కొరకు (పూర్ణాంకం కణుపు_n = 0; కణుపు_n < m_cgraph->n_nodes; కణుపు_n++) {
        స్వయంచాలక * కణుపు = m_cgraph->nodes[కణుపు_n];

        కణుపు_Info current_కణుపు_info;
        స్వయంచాలక కణుపు_name = get_tensor_ov_name(m_cgraph, కణుపు);

        current_కణుపు_info.కణుపు = కణుపు;
        current_కణుపు_info.కణుపు_name = కణుపు_name;
        current_కణుపు_info.కణుపు_op_case = 0;
        current_కణుపు_info.దత్తాంశం_addr = కణుపు->దత్తాంశం;

        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            స్వయంచాలక * src = కణుపు->src[i];
            ఒకవేళ (src == శూన్యసూచిక) {
                కొనసాగు;
            }
            స్వయంచాలక src_name = get_tensor_ov_name(m_cgraph, src);
            ఒకవేళ (src->flags & GGML_TENSOR_FLAG_ఆదానం) {
                src_name = get_tensor_లేఖాచిత్రం_ఆదానం_ov_name(ఇది, m_cgraph, src, కణుపు);
            }
            current_కణుపు_info.కణుపు_inputs[src_name] = src;
            current_కణుపు_info.కణుపు_inputs_పేర్లు.push_back(src_name);

            ఒకవేళ (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_కణుపు_info.కణుపు_inputs_views[src_name])> view_chain;
                స్వయంచాలక * current = src;

                అయితే (current != శూన్యసూచిక) {
                    స్వయంచాలక current_name = get_tensor_ov_name(m_cgraph, current);
                    ఒకవేళ (current->flags & GGML_TENSOR_FLAG_ఆదానం) {
                        current_name = get_tensor_లేఖాచిత్రం_ఆదానం_ov_name(ఇది, m_cgraph, current, కణుపు);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    ఒకవేళ (current->src[0] != శూన్యసూచిక && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } లేకపోతే {
                        ఆపు;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_కణుపు_info.కణుపు_inputs_views[src_name] = view_chain;
            }
        }

        m_కణుపు_info_list.push_back(current_కణుపు_info);
    }
}

పూర్ణాంకం GgmlOvDecoder::compute_op_case(స్థిరరాశి ggml_tensor * కణుపు) స్థిరరాశి {
    పూర్ణాంకం op_case = 0;
    ఎంపిక (కణుపు->op) {
    స్థితి GGML_OP_RESHAPE: {
        ఒకవేళ (m_naive) {
            ఆపు;
        }
        స్వయంచాలక name = std::string(కణుపు->name);
        స్వయంచాలక * src = కణుపు->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        తార్కిక recurrent_sequence = అబద్ధం;
        కొరకు (పూర్ణాంకం i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            స్థిరరాశి స్వయంచాలక * వినియోగదారు = m_cgraph->nodes[i];
            ఒకవేళ (వినియోగదారు->op == GGML_OP_MUL_MAT_ID && వినియోగదారు->src[1] == కణుపు) {
                తిరిగి 1;
            } లేకపోతే ఒకవేళ (వినియోగదారు->op == GGML_OP_SSM_CONV) {
                స్థిరరాశి స్వయంచాలక * concat = వినియోగదారు->src[0];
                ఒకవేళ (concat->op == GGML_OP_CONCAT) {
                    స్థిరరాశి స్వయంచాలక * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == కణుపు;
                }
            } లేకపోతే ఒకవేళ (వినియోగదారు->op == GGML_OP_UNARY && ggml_get_unary_op(వినియోగదారు) == GGML_UNARY_OP_SOFTPLUS) {
                స్థిరరాశి స్వయంచాలక * biased = వినియోగదారు->src[0];
                recurrent_sequence = biased->op == GGML_OP_కూడు && biased->src[0] == కణుపు;
            }
        }
        ఒకవేళ (recurrent_sequence && కణుపు->ne[0] == src->ne[0] && కణుపు->ne[3] == 1) {
            తిరిగి 6;
        }
        ఒకవేళ (కణుపు->ne[0] == src->ne[0] && కణుపు->ne[2] == 1 && కణుపు->ne[3] == 1) {
            తిరిగి 5;
        }
        ఒకవేళ (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == కణుపు->ne[0] && src->src[0]->ne[1] == కణుపు->ne[1]) {
            op_case = 4;
        } లేకపోతే ఒకవేళ (కణుపు->ne[0] * కణుపు->ne[1] == src->ne[0]) {
            op_case = 1;
        } లేకపోతే ఒకవేళ (src->ne[0] * src->ne[1] == కణుపు->ne[0]) {
            op_case = 2;
            ఒకవేళ (src->ne[2] * src->ne[3] == కణుపు->ne[1]) {
                op_case = 5;
            }
        } లేకపోతే ఒకవేళ (కణుపు->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == కణుపు->ne[1]) {
            op_case = 3;
        } లేకపోతే ఒకవేళ (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } లేకపోతే ఒకవేళ (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } లేకపోతే ఒకవేళ (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        ఒకవేళ (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            స్థిరరాశి స్వయంచాలక * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            తార్కిక recurrent = gather->op == GGML_OP_GET_అడ్డువరుసలు && is_recurrent_cache(gather->src[0]);
            కొరకు (పూర్ణాంకం i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                స్థిరరాశి స్వయంచాలక * వినియోగదారు = m_cgraph->nodes[i];
                ఒకవేళ (వినియోగదారు->op == GGML_OP_GATED_DELTA_NET) {
                    కొరకు (పూర్ణాంకం j : {3, 4}) {
                        స్థిరరాశి స్వయంచాలక * gate = వినియోగదారు->src[j];
                        ఒకవేళ (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == కణుపు;
                    }
                } లేకపోతే ఒకవేళ (వినియోగదారు->op == GGML_OP_MUL) {
                    కొరకు (పూర్ణాంకం j = 0; j < 2; ++j) {
                        స్థిరరాశి స్వయంచాలక * gate = వినియోగదారు->src[j];
                        స్థిరరాశి స్వయంచాలక * norm = వినియోగదారు->src[1 - j];
                        ఒకవేళ (gate->op != GGML_OP_UNARY || gate->src[0] != కణుపు) {
                            కొనసాగు;
                        }
                        ఒకవేళ (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            ఒకవేళ (recurrent) {
                op_case = 9;
            }
        }
        ఆపు;
    }
    స్థితి GGML_OP_క్రమమార్పు: {
        ఒకవేళ (కణుపు->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } లేకపోతే ఒకవేళ (కణుపు->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(కణుపు->view_src->name);
            పూర్ణాంకం layer = వెలికితీయి_layer_from_name(src_name).value();
            ఒకవేళ (ggml_is_contiguous(కణుపు->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                ఒకవేళ (!is_swa_layer(layer)) {
                    op_case = 3;
                } లేకపోతే {
                    op_case = 4;
                }
            } లేకపోతే {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                ఒకవేళ (!is_swa_layer(layer)) {
                    op_case = 5;
                } లేకపోతే {
                    op_case = 6;
                }
            }
        } లేకపోతే {
            // rope'ed query tensor
            op_case = 2;
        }
        ఆపు;
    }
    స్థితి GGML_OP_MUL_MAT: {
        ఒకవేళ (కణుపు->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        ఆపు;
    }
    స్థితి GGML_OP_GET_అడ్డువరుసలు: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        ఒకవేళ (కణుపు->src[0]->op == GGML_OP_RESHAPE && కణుపు->src[0]->src[0] != శూన్యసూచిక &&
            is_recurrent_cache(కణుపు->src[0]->src[0])) {
            స్థిరరాశి తార్కిక single_స్థానం = కణుపు->src[0]->src[0]->ne[1] == 1;
            ఒకవేళ (కణుపు->src[1]->op == GGML_OP_VIEW) {
                op_case = (కణుపు->src[1]->view_offs == 0 ? 1 : 2) + (single_స్థానం ? 2 : 0);
            } లేకపోతే ఒకవేళ (single_స్థానం) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        ఆపు;
    }
    స్థితి GGML_OP_ROPE: {
        స్థిరరాశి పూర్ణాంకం mode = కణుపు->op_params[2];
        ఎంపిక (mode) {
        స్థితి GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            ఆపు;
        }
        స్థితి GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            ఆపు;
        }
        స్థితి GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            ఆపు;
        }
        స్థితి GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            ఆపు;
        }
        అప్రమేయ:
            op_case = 0;
            ఆపు;
        }
        ఆపు;
    }
    స్థితి GGML_OP_VIEW: {
        ఒకవేళ (!m_model_params.has_rs_rollback && కణుపు->src[0] != శూన్యసూచిక &&
            కణుపు->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            ఆపు;
        }
        ఒకవేళ (m_is_static && కణుపు->src[0] != శూన్యసూచిక &&
            (కణుపు->src[0]->op == GGML_OP_GATED_DELTA_NET || కణుపు->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            ఆపు;
        }
        ఒకవేళ (కణుపు->src[0]->op == GGML_OP_VIEW) {
            స్వయంచాలక * src = కణుపు->src[0];
            ఒకవేళ (ggml_nelements(కణుపు) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            ఒకవేళ (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            స్వయంచాలక * src = కణుపు->src[0];
            ఒకవేళ (ggml_nelements(కణుపు) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                ఒకవేళ (కణుపు->ne[0] == src->ne[0] && కణుపు->ne[1] == src->ne[2] && కణుపు->ne[2] == 1 &&
                    కణుపు->nb[0] == src->nb[0] && కణుపు->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    ఆపు;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                పూర్ణాంకం diff_లెక్క = 0;
                కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
                    ఒకవేళ (కణుపు->ne[i] != src->ne[i]) {
                        diff_లెక్క++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    ఒకవేళ (కణుపు->ne[i] > src->ne[i]) {
                        తిరిగి 0;
                    }
                }
                ఒకవేళ (diff_లెక్క >= 1) {
                    op_case = 0;
                }
            }
        }
        ఆపు;
    }
    స్థితి GGML_OP_RMS_NORM: {
        ఒకవేళ (కణుపు->src[0]->op == GGML_OP_VIEW) {
            ఒకవేళ (is_same_shape(కణుపు->src[0]->src[0], కణుపు->src[0])) {
                op_case = 1;
            } లేకపోతే ఒకవేళ (!m_model_params.has_rs_rollback &&
                       కణుపు->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } లేకపోతే ఒకవేళ (కణుపు->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        ఆపు;
    }
    స్థితి GGML_OP_POOL_2D: {
        స్థిరరాశి ggml_op_pool pool_బహుళకం = static_cast<ggml_op_pool>(కణుపు->op_params[0]);
        ఎంపిక (pool_బహుళకం) {
        స్థితి GGML_OP_POOL_MAX: {
            op_case = 1;
            ఆపు;
        }
        స్థితి GGML_OP_POOL_AVG: {
            op_case = 2;
            ఆపు;
        }
        అప్రమేయ:
            op_case = 0;
            ఆపు;
        }
        ఆపు;
    }
    స్థితి GGML_OP_UPSCALE: {
        స్థిరరాశి int32_t బహుళకం_flags = కణుపు->op_params[0];
        స్థిరరాశి ggml_scale_బహుళకం scale_బహుళకం = static_cast<ggml_scale_బహుళకం>(బహుళకం_flags & 0xFF);
        ఎంపిక (scale_బహుళకం) {
        స్థితి GGML_SCALE_బహుళకం_సమీప: {
            op_case = 1;
            ఆపు;
        }
        స్థితి GGML_SCALE_బహుళకం_BILINEAR: {
            op_case = 2;
            ఆపు;
        }
        స్థితి GGML_SCALE_బహుళకం_BICUBIC: {
            op_case = 3;
            ఆపు;
        }
        అప్రమేయ:
            op_case = 0;
            ఆపు;
        }
        ఆపు;
    }
    స్థితి GGML_OP_CPY: {
        ఒకవేళ (కణుపు->src[0]->op == GGML_OP_VIEW) {
            ఒకవేళ (కణుపు->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                ఒకవేళ (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_నిండిన_single_స్థానం_writeback(కణుపు) ? 7 : 10;
                } లేకపోతే {
                    op_case = 1;
                }
            } లేకపోతే ఒకవేళ (GgmlOvDecoder::is_conv_state_writeback(కణుపు)) {
                op_case = is_నిండిన_single_స్థానం_writeback(కణుపు) ? 8 : 2;
                ఆపు;
            } లేకపోతే ఒకవేళ (is_conv_స్థితులు_all_tensor(కణుపు->view_src) && కణుపు->src[1] != శూన్యసూచిక &&
                       కణుపు->src[1]->op == GGML_OP_VIEW && కణుపు->src[1]->view_src == కణుపు->view_src) {
                op_case = 4;
                ఆపు;
            } లేకపోతే ఒకవేళ (కణుపు->src[0]->src[0]->op == GGML_OP_GET_అడ్డువరుసలు && కణుపు->src[1] != శూన్యసూచిక &&
                       కణుపు->src[1]->op == GGML_OP_VIEW && కణుపు->src[1]->view_src != శూన్యసూచిక &&
                       is_recurrent_cache(కణుపు->src[1]->view_src) && కణుపు->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } లేకపోతే ఒకవేళ (కణుపు->src[0]->op == GGML_OP_GET_అడ్డువరుసలు && కణుపు->src[1] != శూన్యసూచిక &&
                   కణుపు->src[1]->op == GGML_OP_VIEW && కణుపు->src[1]->view_src != శూన్యసూచిక &&
                   is_recurrent_cache(కణుపు->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = కణుపు->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } లేకపోతే ఒకవేళ (కణుపు->src[1] != శూన్యసూచిక && కణుపు->src[1]->op == GGML_OP_VIEW && కణుపు->src[1]->view_src != శూన్యసూచిక) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            స్థిరరాశి ggml_tensor * kv_buf = కణుపు->src[1]->view_src;
            ఒకవేళ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                కొరకు (పూర్ణాంకం i = 0; i < m_cgraph->n_nodes; i++) {
                    స్థిరరాశి ggml_tensor * n = m_cgraph->nodes[i];
                    ఒకవేళ (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        కొనసాగు;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    ఒకవేళ ((n->src[1] != శూన్యసూచిక && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != శూన్యసూచిక && n->src[2]->view_src == kv_buf)) {
                        ఒకవేళ (n->src[3] != శూన్యసూచిక) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        ఆపు;
                    }
                }
            }
        }
        ఆపు;
    }
    స్థితి GGML_OP_కూడు: {
        ఒకవేళ (is_moe_expert_కూడిక_కూడు(కణుపు)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        ఆపు;
    }
    స్థితి GGML_OP_SCALE: {
        ఒకవేళ (కణుపు->view_src && కణుపు->బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_ANY) {
            op_case = కణుపు->view_src->ne[1] == 1 ? 2 : 1;
        }
        ఆపు;
    }
    స్థితి GGML_OP_L2_NORM: {
        ఒకవేళ (std::string(కణుపు->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        ఆపు;
    }
    స్థితి GGML_OP_FLASH_ATTN_EXT: {
        ఒకవేళ (కణుపు->src[1] != శూన్యసూచిక && కణుపు->src[1]->op == GGML_OP_VIEW && కణుపు->src[1]->view_src != శూన్యసూచిక) {
            స్థిరరాశి ggml_tensor * kv_buf = కణుపు->src[1]->view_src;
            ఒకవేళ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (కణుపు->src[3] != శూన్యసూచిక) ? 1 : 2;
            }
        }
        ఆపు;
    }
    అప్రమేయ:
        ఆపు;
    }
    తిరిగి op_case;
}

std::optional<పూర్ణాంకం> వెలికితీయి_layer_from_name(స్థిరరాశి std::string & name) {
    size_t pos1 = name.find("_l");
    ఒకవేళ (pos1 == std::string::npos) {
        తిరిగి std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    ఒకవేళ (pos2 == std::string::npos) {
        pos2 = name.నిడివి();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    పూర్ణాంకం layer = std::stoi(layer_str);
    తిరిగి layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
స్థిర పూర్ణాంకం get_swa_కిటికీ_from_ముసుగు(స్థిరరాశి ggml_tensor * ముసుగు) {
    ఒకవేళ (ముసుగు->దత్తాంశం == శూన్యసూచిక || !ggml_backend_బఫరు_is_host(ముసుగు->బఫరు)) {
        తిరిగి -1;
    }
    ఒకవేళ (ముసుగు->type != GGML_TYPE_F16 && ముసుగు->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t n_kv = ముసుగు->ne[0];
    స్థిరరాశి int64_t n_టోకెన్లు = ముసుగు->ne[1];
    int64_t కిటికీ = 0;

    కొరకు (int64_t r = 0; r < n_టోకెన్లు; r++) {
        int64_t kept = 0;
        కొరకు (int64_t c = 0; c < n_kv; c++) {
            స్థిరరాశి size_t i = (size_t) r * n_kv + c;
            స్థిరరాశి దశాంశం v = ముసుగు->type == GGML_TYPE_F16 ? ggml_fp16_గమ్యం_fp32(((స్థిరరాశి ggml_fp16_t *) ముసుగు->దత్తాంశం)[i]) :
                                                          ((స్థిరరాశి దశాంశం *) ముసుగు->దత్తాంశం)[i];
            ఒకవేళ (v > -INFINITY) {
                kept++;
            }
        }
        కిటికీ = std::max(కిటికీ, kept);
    }

    తిరిగి కిటికీ > 0 ? (పూర్ణాంకం) కిటికీ : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, తార్కిక is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    స్వయంచాలక get_attention_నమూనా_case = [](స్థిరరాశి ggml_tensor * కణుపు) -> పూర్ణాంకం {
        ఒకవేళ (కణుపు == శూన్యసూచిక) {
            తిరిగి -1;
        }

        ఎంపిక (కణుపు->op) {
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            ఒకవేళ (కణుపు->src[0] == శూన్యసూచిక || కణుపు->src[1] == శూన్యసూచిక) {
                తిరిగి -1;
            }
            ఎంపిక (కణుపు->src[1]->op) {
            స్థితి GGML_OP_క్రమమార్పు:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                ఒకవేళ (కణుపు->src[3] != శూన్యసూచిక && కణుపు->src[1]->src[0] != శూన్యసూచిక &&
                    కణుపు->src[1]->src[0]->op == GGML_OP_VIEW) {
                    తిరిగి 0;
                }
                ఆపు;
            స్థితి GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                ఒకవేళ (కణుపు->src[3] != శూన్యసూచిక && కణుపు->src[1]->src[0] != శూన్యసూచిక &&
                    కణుపు->src[1]->src[0]->op == GGML_OP_క్రమమార్పు && కణుపు->src[1]->src[0]->src[0] != శూన్యసూచిక &&
                    కణుపు->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    తిరిగి 1;
                }
                ఆపు;
            స్థితి GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                ఒకవేళ (కణుపు->src[1]->view_src != శూన్యసూచిక) {
                    ఒకవేళ (కణుపు->src[3] != శూన్యసూచిక) {
                        తిరిగి 4;  // decoder self-attention
                    }
                    తిరిగి 5;      // cross-attention or encoder self-attention
                }
                ఆపు;
            అప్రమేయ:
                ఆపు;
            }
            ఆపు;
        స్థితి GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            ఒకవేళ (కణుపు->src[0] != శూన్యసూచిక && కణుపు->src[1] != శూన్యసూచిక && కణుపు->src[0]->op == GGML_OP_MUL_MAT &&
                కణుపు->src[0]->src[0] != శూన్యసూచిక && కణుపు->src[0]->src[1] != శూన్యసూచిక &&
                కణుపు->src[0]->src[0]->op == GGML_OP_క్రమమార్పు && కణుపు->src[0]->src[0]->src[0] != శూన్యసూచిక &&
                కణుపు->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                తిరిగి 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            ఒకవేళ (కణుపు->src[0]->op == GGML_OP_కూడు && కణుపు->src[0]->src[0] != శూన్యసూచిక &&
                కణుపు->src[0]->src[0]->op == GGML_OP_MUL_MAT && కణుపు->src[0]->src[0]->src[0] != శూన్యసూచిక &&
                కణుపు->src[0]->src[0]->src[0]->op == GGML_OP_క్రమమార్పు) {
                తిరిగి 3;
            }
            ఆపు;
        అప్రమేయ:
            ఆపు;
        }

        తిరిగి -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    స్వయంచాలక get_attention_op_ముసుగు = [&get_attention_నమూనా_case](స్థిరరాశి ggml_tensor * కణుపు) -> స్థిరరాశి ggml_tensor * {
        ఎంపిక (get_attention_నమూనా_case(కణుపు)) {
        స్థితి 0:
        స్థితి 1:
            తిరిగి కణుపు->src[3];
        స్థితి 2:
        స్థితి 3:
            తిరిగి కణుపు->src[1];
        అప్రమేయ:
            తిరిగి శూన్యసూచిక;
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
        std::map<పూర్ణాంకం, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<పూర్ణాంకం, స్థిరరాశి ggml_tensor *> layer_ముసుగు;            // layer -> mask it consumes
        int64_t max_extent = 0;

        కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
            స్థిరరాశి ggml_tensor * ముసుగు = get_attention_op_ముసుగు(cgraph->nodes[i]);
            ఒకవేళ (ముసుగు == శూన్యసూచిక) {
                కొనసాగు;
            }
            స్థిరరాశి ggml_tensor * cache_k_క్రమమార్పు = శూన్యసూచిక;
            ఎంపిక (get_attention_నమూనా_case(cgraph->nodes[i])) {
            స్థితి 0:  cache_k_క్రమమార్పు = cgraph->nodes[i]->src[1];                     ఆపు;
            స్థితి 1:  cache_k_క్రమమార్పు = cgraph->nodes[i]->src[1]->src[0];             ఆపు;
            స్థితి 2:  cache_k_క్రమమార్పు = cgraph->nodes[i]->src[0]->src[0];             ఆపు;
            అప్రమేయ: cache_k_క్రమమార్పు = cgraph->nodes[i]->src[0]->src[0]->src[0];     ఆపు;
            }
            స్థిరరాశి ggml_tensor * cache_k_view = cache_k_క్రమమార్పు->src[0];
            ఒకవేళ (cache_k_view->op != GGML_OP_VIEW) {
                కొనసాగు;
            }
            స్థిరరాశి ggml_tensor * ఆకు = cache_k_view->src[0];
            స్వయంచాలక layer = వెలికితీయి_layer_from_name(ఆకు->name);
            ఒకవేళ (!layer.has_value()) {
                కొనసాగు;
            }
            layer_extent[layer.value()] = ఆకు->ne[1];
            layer_ముసుగు[layer.value()] = ముసుగు;
            max_extent = std::max(max_extent, ఆకు->ne[1]);
        }

        కొరకు (స్థిరరాశి స్వయంచాలక & [layer, extent] : layer_extent) {
            ఒకవేళ (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                ఒకవేళ (model_params.swa_ముసుగు == శూన్యసూచిక) {
                    model_params.swa_ముసుగు = layer_ముసుగు[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string చొప్పున_layer;
            కొరకు (స్థిరరాశి స్వయంచాలక & [layer, extent] : layer_extent) {
                చొప్పున_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (దీర్ఘ) max_extent, model_params.swa_layers.size(), చొప్పున_layer.c_str());
        }
    }

    తార్కిక rope_చూసిన = అబద్ధం;
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * కణుపు = cgraph->nodes[i];
        స్థిరరాశి పూర్ణాంకం attention_నమూనా_case = get_attention_నమూనా_case(కణుపు);
        ఒకవేళ (attention_నమూనా_case != -1) {
            ggml_tensor * cache_k_క్రమమార్పు = శూన్యసూచిక;
            ggml_tensor * ముసుగు = శూన్యసూచిక;

            ఎంపిక (attention_నమూనా_case) {
            స్థితి 0:
                cache_k_క్రమమార్పు = కణుపు->src[1];
                ముసుగు = కణుపు->src[3];
                ఆపు;
            స్థితి 1:
                cache_k_క్రమమార్పు = కణుపు->src[1]->src[0];
                ముసుగు = కణుపు->src[3];
                ఆపు;
            స్థితి 2:
                cache_k_క్రమమార్పు = కణుపు->src[0]->src[0];
                ముసుగు = కణుపు->src[1];
                ఆపు;
            స్థితి 3:
                cache_k_క్రమమార్పు = కణుపు->src[0]->src[0]->src[0];
                ముసుగు = కణుపు->src[1];
                ఆపు;
            స్థితి 4:
            స్థితి 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                స్వయంచాలక * cache_k_view = కణుపు->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_చొప్పున_seq = కణుపు->src[0]->ne[1];
                ఒకవేళ (attention_నమూనా_case == 4) {
                    compute_params.attention_కొలత = cache_k_view->ne[1];
                } లేకపోతే {
                    compute_params.attention_కొలత_static = cache_k_view->ne[1];
                }
                కొనసాగు;
            }
            అప్రమేయ:
                ఆపు;
            }

            ప్రతిజ్ఞా(cache_k_క్రమమార్పు != శూన్యసూచిక);

            model_params.తల_కొలత = cache_k_క్రమమార్పు->ne[0];
            model_params.n_heads_kv = cache_k_క్రమమార్పు->ne[2];
            compute_params.ఆదానం_len = కణుపు->src[0]->ne[1];
            compute_params.token_len_చొప్పున_seq = కణుపు->src[0]->ne[1];

            స్వయంచాలక * cache_k_view = cache_k_క్రమమార్పు->src[0];
            ఒకవేళ (cache_k_view->op != GGML_OP_VIEW || ముసుగు == శూన్యసూచిక) {
                కొనసాగు;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            పూర్ణాంకం layer = వెలికితీయి_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            స్థిరరాశి తార్కిక layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_బఫరు_ctx_id = ggml_backend_openvino_బఫరు_get_ctx_id(cache_k->బఫరు);
            model_params.n_heads_kv_చొప్పున_layer[layer] = cache_k_క్రమమార్పు->ne[2];
            ఒకవేళ (layer_is_swa) {
                model_params.ctx_చొప్పున_seq_swa = cache_k->ne[1];
            } లేకపోతే {
                model_params.ctx_చొప్పున_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_చురుకైన = ముసుగు->ne[3];
            స్వయంచాలక seq_కొలత = cache_k->ne[0] * cache_k->ne[1] * ggml_type_కొలత(cache_k->type);
            size_t స్థానభ్రంశం;
            జ్ఞాపకం_నకలు(&స్థానభ్రంశం, cache_k_view->op_params, పరిమాణం(size_t));
            compute_params.seq_చురుకైన_ప్రారంభం = స్థానభ్రంశం / seq_కొలత;

            ఒకవేళ (layer_is_swa) {
                compute_params.attention_కొలత_swa = ముసుగు->ne[0];
                compute_params.swa_కిటికీ = get_swa_కిటికీ_from_ముసుగు(ముసుగు);
            } లేకపోతే {
                compute_params.attention_కొలత = ముసుగు->ne[0];
            }
            ఒకవేళ (is_static) {
                compute_params.attention_కొలత = model_params.ctx_చొప్పున_seq;
                compute_params.attention_కొలత_swa = model_params.ctx_చొప్పున_seq_swa;
                compute_params.token_len_చొప్పున_seq = 1;
            }
        }

        ఒకవేళ (కణుపు->op == GGML_OP_MUL_MAT && కణుపు->src[0]->op == GGML_OP_క్రమమార్పు &&
            కణుపు->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(కణుపు->src[0]->view_src, కణుపు->view_src)) {
            ఒకవేళ (కణుపు->src[1]->op == GGML_OP_క్రమమార్పు && కణుపు->src[1]->src[0]->op == GGML_OP_VIEW &&
                కణుపు->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_కొలత = కణుపు->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        ఒకవేళ (కణుపు->op == GGML_OP_TRANSPOSE && కణుపు->src[0]->op == GGML_OP_క్రమమార్పు &&
            కణుపు->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_కొలత = కణుపు->ne[0];
            ఒకవేళ (is_static) {
                compute_params.attention_కొలత = model_params.ctx_చొప్పున_seq;
            }
        }
        ఒకవేళ (కణుపు->op == GGML_OP_ROPE) {
            ఒకవేళ (compute_params.token_len_చొప్పున_seq == -1 && కణుపు->src[1] != శూన్యసూచిక) {
                compute_params.token_len_చొప్పున_seq = ggml_nelements(కణుపు->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(పరిమాణం(model_params.rope_params) == పరిమాణం(int32_t) * 16, "rope_params size");
            ఒకవేళ (!rope_చూసిన) {
                జ్ఞాపకం_నకలు(model_params.rope_params, కణుపు->op_params, పరిమాణం(int32_t) * 16);
                rope_చూసిన = నిజం;
            } లేకపోతే ఒకవేళ (memcmp(model_params.rope_params, కణుపు->op_params, పరిమాణం(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = నిజం;
            }
        }
        ఒకవేళ (కణుపు->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_కొలత = కణుపు->src[0]->ne[0];
        }
        ఒకవేళ (కణుపు->op == GGML_OP_SCALE && కణుపు->view_src != శూన్యసూచిక && is_recurrent_cache(కణుపు->view_src)) {
            ఒకవేళ (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = కణుపు->view_src->ne[1];
            } లేకపోతే {
                GGML_ASSERT(model_params.n_rs_slots == కణుపు->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(కణుపు) / కణుపు->view_src->ne[0];
            compute_params.cache_rs_reset_idx = కణుపు->src[0]->view_offs / కణుపు->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        ఒకవేళ (కణుపు->op == GGML_OP_CPY && కణుపు->view_src != శూన్యసూచిక && is_recurrent_cache(కణుపు->view_src) &&
            కణుపు->src[1] != శూన్యసూచిక && కణుపు->src[1]->op == GGML_OP_VIEW && కణుపు->src[1]->view_src == కణుపు->view_src) {
            స్థిరరాశి తార్కిక is_conv = is_conv_state_writeback(కణుపు);
            స్థిరరాశి తార్కిక is_gdn = కణుపు->src[0]->op == GGML_OP_VIEW && కణుపు->src[0]->src[0] != శూన్యసూచిక &&
                                కణుపు->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            స్థిరరాశి తార్కిక is_extra = కణుపు->src[0]->op == GGML_OP_GET_అడ్డువరుసలు;
            స్థిరరాశి తార్కిక is_gdn_rollback = is_gdn && is_same_shape(కణుపు->src[0], కణుపు->src[1]);

            స్థిరరాశి ggml_tensor * dest_view = కణుపు->src[1];
            స్థిరరాశి ggml_tensor * cache = కణుపు->view_src;
            స్థిరరాశి size_t అడ్డువరుస_బైట్లు = cache->ne[0] * ggml_type_కొలత(cache->type);
            ఒకవేళ (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = నిజం;
            }
            ఒకవేళ (అడ్డువరుస_బైట్లు > 0 && (is_conv || is_gdn || is_extra) && !is_నిండిన_single_స్థానం_writeback(కణుపు)) {
                ComputeParams::RsWriteback writeback;
                writeback.స్థానం_begin = (పూర్ణాంకం) (dest_view->view_offs / అడ్డువరుస_బైట్లు);
                ఒకవేళ (is_conv) {
                    writeback.src_begin = (పూర్ణాంకం) (కణుపు->src[0]->view_offs / కణుపు->src[0]->view_src->nb[0]);
                } లేకపోతే ఒకవేళ (is_gdn_rollback) {
                    writeback.src_begin = (పూర్ణాంకం) (కణుపు->src[0]->view_offs / కణుపు->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, కణుపు)] = writeback;
            }
            ఒకవేళ ((is_conv || is_gdn) && !is_నిండిన_single_స్థానం_writeback(కణుపు)) {
                compute_params.s_copy_చురుకైన_స్థానం_len = (పూర్ణాంకం) dest_view->ne[1];
            }
        }
    }
    ఒకవేళ (model_params.n_heads_kv == -1) {
        కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
            స్థిరరాశి స్వయంచాలక * కణుపు = cgraph->nodes[i];
            స్థిరరాశి ggml_tensor * ముసుగు = శూన్యసూచిక;
            ఒకవేళ (కణుపు->op == GGML_OP_SOFT_MAX) {
                ముసుగు = కణుపు->src[1];
            } లేకపోతే ఒకవేళ (కణుపు->op == GGML_OP_FLASH_ATTN_EXT) {
                ముసుగు = కణుపు->src[3];
            } లేకపోతే {
                కొనసాగు;
            }
            ఒకవేళ (ముసుగు == శూన్యసూచిక || ముసుగు->op != GGML_OP_NONE || !(ముసుగు->flags & GGML_TENSOR_FLAG_ఆదానం) ||
                కణుపు->src[0] == శూన్యసూచిక) {
                కొనసాగు;
            }
            model_params.is_cacheless_attn = నిజం;
            model_params.n_seq = 1;
            model_params.ctx_చొప్పున_seq = ముసుగు->ne[0];
            compute_params.ఆదానం_len = కణుపు->src[0]->ne[1];
            compute_params.token_len_చొప్పున_seq = compute_params.ఆదానం_len;
            ఆపు;
        }
    }

    స్వయంచాలక * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    ఒకవేళ (model_params.is_cacheless_attn) {
        కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
            స్థిరరాశి స్వయంచాలక * కణుపు = cgraph->nodes[i];
            ఒకవేళ (కణుపు->op == GGML_OP_GET_అడ్డువరుసలు && is_output_idx(కణుపు->src[1], కణుపు)) {
                compute_params.output_len = కణుపు->src[1]->ne[0];
                ఆపు;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    ఒకవేళ (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_చొప్పున_seq * model_params.n_seq;
    తిరిగి {model_params, compute_params};
}

శూన్య GgmlOvDecoder::validate_cgraph() స్థిరరాశి {
    ఒకవేళ (m_model_params.n_seq > 1 && m_is_static == నిజం) {
        విసురు std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_లేఖాచిత్రం_ఆదానం_shape(స్థిరరాశి ggml_tensor * op,
                                                      స్థిరరాశి ggml_tensor * ఆదానం,
                                                      పూర్ణాంకం dynamic_dim_సూచీ) స్థిరరాశి {
    ఒకవేళ (m_naive) {
        తిరిగి ఆదానం != శూన్యసూచిక ? ov::PartialShape{get_shape(ఆదానం)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape ఆదానం_shape;

    ఒకవేళ (is_inp_tok(ఆదానం, op) || is_inp_pos(ఆదానం, op)) {
        // tokens or positions
        పూర్ణాంకం len = m_is_static ? (m_is_prefill ? m_prefill_chunk_కొలత : 1) : -1;
        ఒకవేళ (m_is_static && is_inp_pos(ఆదానం, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        ఆదానం_shape = ov::PartialShape{1, 1, 1, len};

    } లేకపోతే ఒకవేళ (is_output_idx(ఆదానం, op)) {
        // output index
        ఆదానం_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } లేకపోతే ఒకవేళ (is_inp_సగటు(ఆదానం, op)) {
        ఆదానం_shape = m_is_static ? ov::PartialShape{1, 1, ఆదానం->ne[1], m_prefill_chunk_కొలత} :
                                    ov::PartialShape{1, 1, -1, -1};

    } లేకపోతే ఒకవేళ (is_inp_scale_అడ్డువరుసలు(ఆదానం, op)) {
        ఆదానం_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_కొలత : 1) : -1, 1};

    } లేకపోతే ఒకవేళ (is_inp_ముసుగు(ఆదానం, op)) {
        //ముసుగు
        ఒకవేళ (m_is_static) {
            ఆదానం_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_కొలత : 1, m_model_params.ctx};
        } లేకపోతే ఒకవేళ (m_is_stateful) {
            ఆదానం_shape = ov::PartialShape{1, 1, -1, -1};
        } లేకపోతే {
            ఆదానం_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } లేకపోతే ఒకవేళ (is_recurrent_cache(ఆదానం)) {
        ఆదానం_shape = ov::PartialShape{get_shape(ఆదానం)};
        ఒకవేళ (!m_is_static && !m_is_stateful && ఆదానం->ne[1] > 1) {
            ఆదానం_shape[2] = -1;
        }

    } లేకపోతే ఒకవేళ (is_kvcache(ఆదానం, op)) {
        // kvcache
        ఆదానం_shape = ov::PartialShape{get_shape(ఆదానం)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        స్థిరరాశి తార్కిక is_flat_kv = (ఆదానం->ne[1] == 1 && ఆదానం->ne[2] == 1 && ఆదానం->ne[3] == 1);
        ఒకవేళ (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            ఆదానం_shape[2] = -1;
        }
        ఒకవేళ (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            ప్రతిజ్ఞా(ఆదానం_shape.size() == 4 && ఆదానం_shape[0] == 1 && ఆదానం_shape[1] == 1 &&
                   ఆదానం_shape[2].is_dynamic() && ఆదానం_shape[3].is_static());
            స్థిరరాశి పూర్ణాంకం n_heads_kv = get_n_heads_kv_for_tensor(ఆదానం);
            ప్రతిజ్ఞా(n_heads_kv > 0 && ఆదానం_shape[3].get_నిడివి() % n_heads_kv == 0);
            స్థిరరాశి int64_t combined_dim = ఆదానం_shape[3].get_నిడివి();  // n_heads_kv * head_size
            స్థిరరాశి int64_t తల_కొలత = combined_dim / n_heads_kv;
            ఆదానం_shape = {ఆదానం_shape[0], ov::Dimension::dynamic(), n_heads_kv, తల_కొలత};
        }

    } లేకపోతే ఒకవేళ (is_kv_idx(ఆదానం, op)) {
        // kv update index
        పూర్ణాంకం len = m_is_static ? (m_is_prefill ? m_prefill_chunk_కొలత : 1) : -1;
        ఆదానం_shape = ov::PartialShape{1, 1, 1, len};

    } లేకపోతే ఒకవేళ (is_inp_s_copy(ఆదానం, op) || is_s_copy_ఆకు(ఆదానం)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        ఆదానం_shape = m_is_static ? ov::PartialShape{get_shape(ఆదానం)} : ov::PartialShape{1, 1, 1, -1};

    } లేకపోతే {
        ఆదానం_shape = ov::PartialShape{get_shape(ఆదానం)};
    }
    ఒకవేళ (dynamic_dim_సూచీ != -1 && m_model_is_splitted) {
        ఆదానం_shape[3 - dynamic_dim_సూచీ] = -1;
    }
    ఒకవేళ (op->op == GGML_OP_SOFT_MAX && op->src[1] != శూన్యసూచిక && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_ఆదానం && op->src[1] == ఆదానం) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        ఒకవేళ (m_is_static) {
            స్థిరరాశి int64_t seq_చురుకైన = m_is_prefill ? m_prefill_chunk_కొలత : 1;
            ఆదానం_shape[2] = seq_చురుకైన;
            ఆదానం_shape[3] = seq_చురుకైన;
        } లేకపోతే {
            ఆదానం_shape[2] = -1;
            ఆదానం_shape[3] = -1;
        }
    }
    తిరిగి ఆదానం_shape;
}

తార్కిక GgmlOvDecoder::is_s_copy_ఆకు(స్థిరరాశి ggml_tensor * tensor) స్థిరరాశి {
    ఒకవేళ (tensor == శూన్యసూచిక || tensor->op != GGML_OP_NONE || m_cgraph == శూన్యసూచిక) {
        తిరిగి అబద్ధం;
    }
    కొరకు (పూర్ణాంకం i = 0; i < m_cgraph->n_nodes; i++) {
        స్థిరరాశి ggml_tensor * కణుపు = m_cgraph->nodes[i];
        ఒకవేళ (కణుపు->op != GGML_OP_GET_అడ్డువరుసలు || కణుపు->src[0] == శూన్యసూచిక || కణుపు->src[1] == శూన్యసూచిక) {
            కొనసాగు;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        స్థిరరాశి ggml_tensor * idx = కణుపు->src[1];
        అయితే (idx != శూన్యసూచిక && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        ఒకవేళ (idx != tensor) {
            కొనసాగు;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        స్థిరరాశి ggml_tensor * దత్తాంశం = కణుపు->src[0];
        అయితే (దత్తాంశం != శూన్యసూచిక && (దత్తాంశం->op == GGML_OP_VIEW || దత్తాంశం->op == GGML_OP_RESHAPE)) {
            దత్తాంశం = దత్తాంశం->src[0];
        }
        ఒకవేళ (దత్తాంశం != శూన్యసూచిక && is_recurrent_cache(దత్తాంశం)) {
            తిరిగి నిజం;
        }
    }
    తిరిగి అబద్ధం;
}

శూన్య GgmlOvDecoder::కూడు_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    స్వయంచాలక create_1d_ఆదానం = [ఇది](స్థిరరాశి std::string & name, int64_t value, తార్కిక force_parameter = అబద్ధం) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    ఒకవేళ (m_compute_params.attention_కొలత != -1) {
        create_1d_ఆదానం("attention_size", m_compute_params.attention_కొలత);
    }
    ఒకవేళ (m_compute_params.attention_కొలత_static != -1) {
        create_1d_ఆదానం("attention_size_static", m_compute_params.attention_కొలత_static);
    }
    ఒకవేళ (m_compute_params.attention_కొలత_swa != -1) {
        create_1d_ఆదానం("attention_size_swa", m_compute_params.attention_కొలత_swa);
    }
    // only the stateful SWA mask consumes this
    ఒకవేళ (is_stateful() && m_compute_params.swa_కిటికీ != -1) {
        create_1d_ఆదానం("swa_window", m_compute_params.swa_కిటికీ);
    }
    create_1d_ఆదానం("n_seq_active", m_compute_params.n_seq_చురుకైన);
    create_1d_ఆదానం("seq_active_start", m_compute_params.seq_చురుకైన_ప్రారంభం);
    create_1d_ఆదానం("seq_active_end", m_compute_params.seq_చురుకైన_ప్రారంభం + m_compute_params.n_seq_చురుకైన);
    ఒకవేళ (m_compute_params.token_len_చొప్పున_seq != -1) {
        create_1d_ఆదానం("token_len_per_seq", m_compute_params.token_len_చొప్పున_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    ఒకవేళ (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_ఆదానం("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/నిజం);
        create_1d_ఆదానం("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/నిజం);
    }

    ఒకవేళ (m_compute_params.s_copy_చురుకైన_స్థానం_len != -1) {
        create_1d_ఆదానం("s_copy_active_slot_len", m_compute_params.s_copy_చురుకైన_స్థానం_len);
        ఒకవేళ (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_ఆదానం("chunk_valid_len", get_static_n_టోకెన్లు(), /*force_parameter=*/నిజం);
        }
    }

    కొరకు (స్థిరరాశి స్వయంచాలక & [కణుపు_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_ఆదానం("rs_slot_begin_" + కణుపు_name, writeback.స్థానం_begin);
        ఒకవేళ (!m_is_static && writeback.src_begin >= 0) {
            create_1d_ఆదానం("rs_src_begin_" + కణుపు_name, writeback.src_begin);
        }
    }
}

తార్కిక GgmlOvDecoder::కణుపు_is_వాడిన_as_src(స్థిరరాశి పూర్ణాంకం కణుపు_idx) {
    ggml_tensor * కణుపు = m_cgraph->nodes[కణుపు_idx];
    కొరకు (పూర్ణాంకం i = కణుపు_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_కణుపు = m_cgraph->nodes[i];
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            ఒకవేళ (other_కణుపు->src[j] == కణుపు) {
                తిరిగి నిజం;
            }
        }
    }
    తిరిగి అబద్ధం;
}

శూన్య GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    కొరకు (పూర్ణాంకం i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * కణుపు = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        ఒకవేళ (కణుపు->op == GGML_OP_NONE && కణుపు_is_వాడిన_as_src(i)) {
            std::string కణుపు_name = get_tensor_ov_name(m_cgraph, కణుపు);
            ఒకవేళ (m_model_weights.find(కణుపు_name) == m_model_weights.end()) {
                m_inputs[కణుపు_name] = కణుపు;
                m_model_inputs[కణుపు_name] = {get_ov_type(కణుపు),
                                             get_లేఖాచిత్రం_ఆదానం_shape(కణుపు, శూన్యసూచిక, m_కణుపు_dynamic_dims[కణుపు])};
            }
            కొనసాగు;
        }
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            స్వయంచాలక * src = కణుపు->src[i];
            ఒకవేళ (src == శూన్యసూచిక) {
                కొనసాగు;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            ఒకవేళ (src->flags & GGML_TENSOR_FLAG_ఆదానం) {
                src_name = get_tensor_లేఖాచిత్రం_ఆదానం_ov_name(ఇది, m_cgraph, src, కణుపు);
            }
            ఒకవేళ (m_model_weights.find(src_name) != m_model_weights.end()) {
                కొనసాగు;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            ఒకవేళ (src->view_src != శూన్యసూచిక &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                కొనసాగు;
            }

            తార్కిక is_intermediate_కణుపు = అబద్ధం;
            కొరకు (స్థిరరాశి స్వయంచాలక & కణుపు_info : m_కణుపు_info_list) {
                ఒకవేళ (కణుపు_info.కణుపు == src) {
                    is_intermediate_కణుపు = నిజం;
                    ఆపు;
                }
            }
            ఒకవేళ (is_intermediate_కణుపు) {
                కొనసాగు;
            }
            ఒకవేళ (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                కొనసాగు;
            }

            m_inputs[src_name] = src;

            ggml_backend_బఫరు * బఫరు = src->బఫరు;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            ఒకవేళ (బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_ANY) {
                ఒకవేళ (స్వయంచాలక it = std::find(m_model_params.kv_పేర్లు.begin(), m_model_params.kv_పేర్లు.end(), src_name);
                    it == m_model_params.kv_పేర్లు.end()) {
                    m_model_params.kv_పేర్లు.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            అయితే (src->op == GGML_OP_VIEW && src->src[0] != శూన్యసూచిక) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_లేఖాచిత్రం_ఆదానం_shape(కణుపు, src, m_కణుపు_dynamic_dims[src])};
        }
    }
}

శూన్య GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_పేర్లు.clear();
    కొరకు (పూర్ణాంకం కణుపు_n = 0; కణుపు_n < m_cgraph->n_nodes; కణుపు_n++) {
        స్వయంచాలక * cur_కణుపు = m_cgraph->nodes[కణుపు_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        ఒకవేళ (cur_కణుపు->op == GGML_OP_NONE || cur_కణుపు->op == GGML_OP_VIEW || cur_కణుపు->op == GGML_OP_RESHAPE) {
            కొనసాగు;
        }
        ఒకవేళ (::is_inplace_op(cur_కణుపు) && ggml_nbytes(cur_కణుపు) == 0) {
            కొనసాగు;
        }
        స్వయంచాలక cur_కణుపు_use_లెక్క = m_cgraph->use_counts[ggml_hash_కనుగొను(&m_cgraph->సందర్శించిన_hash_సమితి, cur_కణుపు)];
        ఒకవేళ (cur_కణుపు_use_లెక్క == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            ఒకవేళ (cur_కణుపు != శూన్యసూచిక && ::is_inplace_op(cur_కణుపు) && ggml_nbytes(cur_కణుపు) > 0) {
                cur_కణుపు = cur_కణుపు->view_src;
            }
        } లేకపోతే {
            పూర్ణాంకం ఆదానం_use_లెక్క = 0;
            కొరకు (పూర్ణాంకం i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * కణుపు = m_cgraph->nodes[i];
                కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
                    ఒకవేళ (కణుపు->src[j] != ఖాళీ && కణుపు->src[j] == cur_కణుపు) {
                        ఆదానం_use_లెక్క++;
                    }
                }
            }
            ఒకవేళ (ఆదానం_use_లెక్క == cur_కణుపు_use_లెక్క) {
                cur_కణుపు = శూన్యసూచిక;
            }
        }
        ఒకవేళ (cur_కణుపు != శూన్యసూచిక) {
            std::string cur_కణుపు_name = get_tensor_ov_name(m_cgraph, cur_కణుపు);
            m_model_outputs[cur_కణుపు_name] = cur_కణుపు;
            m_model_output_పేర్లు.insert(cur_కణుపు_name);
        }
    }
}

స్థిరరాశి ggml_tensor * GgmlOvDecoder::get_tensor_వాడిన_op(స్థిరరాశి ggml_tensor * tensor) స్థిరరాశి {
    ఒకవేళ (tensor == శూన్యసూచిక) {
        తిరిగి శూన్యసూచిక;
    }
    కొరకు (పూర్ణాంకం i = 0; i < m_cgraph->n_nodes; i++) {
        స్థిరరాశి స్వయంచాలక * కణుపు = m_cgraph->nodes[i];
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            ఒకవేళ (కణుపు->src[j] == tensor) {
                తిరిగి కణుపు;
            }
        }
    }
    తిరిగి శూన్యసూచిక;
}

స్థిరరాశి ggml_tensor * GgmlOvDecoder::get_tensor_from_name(స్థిరరాశి std::string & name) స్థిరరాశి {
    కొరకు (పూర్ణాంకం i = 0; i < m_cgraph->n_nodes; i++) {
        స్థిరరాశి స్వయంచాలక * కణుపు = m_cgraph->nodes[i];
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            స్థిరరాశి స్వయంచాలక * src = కణుపు->src[j];
            ఒకవేళ (src == శూన్యసూచిక) {
                ఆపు;
            }
            ఒకవేళ (get_tensor_ov_name(m_cgraph, src) == name) {
                తిరిగి src;
            }
        }
    }
    తిరిగి శూన్యసూచిక;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_పేర్లు() స్థిరరాశి {
    std::map<std::string, std::string> kv_param_res_పేర్లు;
    కొరకు (స్థిరరాశి స్వయంచాలక & name : m_model_params.kv_పేర్లు) {
        kv_param_res_పేర్లు[name] = name;
    }
    తిరిగి kv_param_res_పేర్లు;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
స్థిర తార్కిక is_mul_mat_id_expert_బరువు(స్థిరరాశి ggml_tensor * కణుపు, పూర్ణాంకం src_సూచీ) {
    తిరిగి కణుపు->op == GGML_OP_MUL_MAT_ID && src_సూచీ == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_బరువు_nodes(ggml_cgraph * cgraph, తార్కిక naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    స్వయంచాలక * nodes = cgraph->nodes;
    స్వయంచాలక n_nodes = cgraph->n_nodes;
    కొరకు (పూర్ణాంకం కణుపు_i = 0; కణుపు_i < n_nodes; కణుపు_i++) {
        స్వయంచాలక * కణుపు = nodes[కణుపు_i];
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            స్వయంచాలక * src = కణుపు->src[i];
            ఒకవేళ (src == శూన్యసూచిక) {
                కొనసాగు;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * పునాది = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, పునాది);
            ఒకవేళ (is_rope_freqs_బరువు(పునాది, కణుపు)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_బఫరు * బఫరు = పునాది->బఫరు;
            ఒకవేళ (బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_WEIGHTS || ggml_is_quantized(పునాది->type) ||
                is_mul_mat_id_expert_బరువు(కణుపు, i)) {
                ఒకవేళ (model_weights.find(src_name) == model_weights.end()) {
                    స్వయంచాలక బరువు_కణుపు = create_బరువు_కణుపు(పునాది, naive);
                    బరువు_కణుపు->సమితి_friendly_name(src_name);
                    model_weights[src_name] = బరువు_కణుపు;
                }
            }
        }
    }
    తిరిగి model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
స్థిర std::mutex g_nonov_బరువు_cache_mutex;
స్థిర std::unordered_map<స్థిరరాశి శూన్య *, std::shared_ptr<ov::Node>> g_nonov_బరువు_cache;

std::set<std::string> GgmlOvDecoder::collect_బరువు_పేర్లు(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> పేర్లు;
    కొరకు (పూర్ణాంకం కణుపు_i = 0; కణుపు_i < cgraph->n_nodes; కణుపు_i++) {
        స్వయంచాలక * కణుపు = cgraph->nodes[కణుపు_i];
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            స్వయంచాలక * src = కణుపు->src[i];
            ఒకవేళ (src == శూన్యసూచిక) {
                కొనసాగు;
            }
            స్థిరరాశి ggml_tensor * పునాది = src->view_src ? src->view_src : src;
            std::string src_name(పునాది->name);
            ఒకవేళ (is_rope_freqs_బరువు(పునాది, కణుపు)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_బఫరు * బఫరు = పునాది->బఫరు;
            ఒకవేళ (బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_WEIGHTS || ggml_is_quantized(పునాది->type)) {
                పేర్లు.insert(src_name);
            }
        }
    }
    తిరిగి పేర్లు;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_బరువు_కణుపు(ggml_tensor * tensor, తార్కిక naive) {
    స్థిరరాశి తార్కిక is_ov_బఫరు = ggml_backend_బఫరు_is_openvino(tensor->బఫరు);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    ఒకవేళ (tensor->extra) {
        OPENVINO_ASSERT(is_ov_బఫరు, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        స్వయంచాలక * extra_పునాది = static_cast<ggml_openvino_extra_పునాది *>(tensor->extra);

        ఒకవేళ (extra_పునాది->type == ggml_openvino_extra_పునాది::Type::బరువు_2) {
            // F16/F32/BF16 weight with shared-memory constant
            స్వయంచాలక * బరువు_extra = static_cast<ggml_openvino_బరువు_extra *>(tensor->extra);
            ఒకవేళ (బరువు_extra->బరువు_కణుపు) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                తిరిగి బరువు_extra->బరువు_కణుపు;
            }
        } లేకపోతే ఒకవేళ (extra_పునాది->type == ggml_openvino_extra_పునాది::Type::QUANTIZED_బరువు) {
            // Quantized weight with pre-extracted data
            స్వయంచాలక * quant_extra = static_cast<ggml_openvino_quantized_బరువు_extra *>(tensor->extra);
            ఒకవేళ (quant_extra->బరువు_కణుపు) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                తిరిగి quant_extra->బరువు_కణుపు;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    ఒకవేళ (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        స్వయంచాలక బరువు_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->దత్తాంశం);
        స్వయంచాలక బరువు_కణుపు = std::make_shared<ov::op::v0::Constant>(బరువు_tensor);
        బరువు_కణుపు->సమితి_friendly_name(tensor->name);
        తిరిగి బరువు_కణుపు;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    స్థిరరాశి తార్కిక cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_బఫరు &&
                                 !naive && tensor->దత్తాంశం != శూన్యసూచిక;
    ఒకవేళ (cacheable_nonov) {
        std::lock_guard<std::mutex> తాళం(g_nonov_బరువు_cache_mutex);
        స్వయంచాలక it = g_nonov_బరువు_cache.find(tensor->దత్తాంశం);
        ఒకవేళ (it != g_nonov_బరువు_cache.end()) {
            తిరిగి it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    స్థిర స్థిరరాశి std::set<ggml_type> బరువు_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    ఒకవేళ (బరువు_types.find(tensor->type) == బరువు_types.end()) {
        విసురు std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    Ov_బరువు ov_బరువు;
    ఒకవేళ (ggml_is_quantized(tensor->type)) {
        స్వయంచాలక use_bias = naive;
        ఒకవేళ (is_ov_బఫరు) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_కొలత = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_కొలత);
            జ్ఞాపకం_నకలు(tmp.దత్తాంశం(), tensor->దత్తాంశం, raw_కొలత);
            ov_బరువు = process_బరువు_tensor(tensor, tmp.దత్తాంశం(), tensor->దత్తాంశం, use_bias);
        } లేకపోతే {
            ov_బరువు = process_బరువు_tensor(tensor, tensor->దత్తాంశం, శూన్యసూచిక, use_bias);
        }
    } లేకపోతే {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_బరువు = process_బరువు_tensor(tensor, tensor->దత్తాంశం, tensor->దత్తాంశం);
    }

    ov_బరువు.బరువు_కణుపు->సమితి_friendly_name(tensor->name);
    ఒకవేళ (!is_ov_బఫరు) {
        ఒకవేళ (cacheable_nonov) {
            std::lock_guard<std::mutex> తాళం(g_nonov_బరువు_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            స్వయంచాలక [it, inserted] = g_nonov_బరువు_cache.emplace(tensor->దత్తాంశం, ov_బరువు.బరువు_కణుపు);
            తిరిగి it->second;
        }
        తిరిగి ov_బరువు.బరువు_కణుపు;
    }

    ggml_openvino_extra_పునాది * extra;
    ఒకవేళ (ov_బరువు.is_quantized()) {
        extra = కొత్త ggml_openvino_quantized_బరువు_extra(std::move(ov_బరువు.weights), std::move(ov_బరువు.scales),
                                                         std::move(ov_బరువు.zp), ov_బరువు.బరువు_కణుపు);
    } లేకపోతే {
        extra = కొత్త ggml_openvino_బరువు_extra(std::move(ov_బరువు.weights), ov_బరువు.బరువు_కణుపు);
    }
    ggml_openvino_బఫరు_రిజిస్టరు_extra(tensor, extra);

    తిరిగి ov_బరువు.బరువు_కణుపు;
}

శూన్య GgmlOvDecoder::dump_cgraph(స్థిరరాశి ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    ఒకవేళ (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        తిరిగి;
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
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * కణుపు = cgraph->nodes[i];

        // Get buffer type name
        స్థిరరాశి అక్షరం * buf_name = "none";
        ggml_backend_బఫరు_t buf = కణుపు->view_src ? కణుపు->view_src->బఫరు : కణుపు->బఫరు;
        ఒకవేళ (buf) {
            buf_name = ggml_backend_బఫరు_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << కణుపు->ne[0] << ", "
             << std::setw(5) << కణుపు->ne[1] << ", "
             << std::setw(5) << కణుపు->ne[2] << ", "
             << std::setw(5) << కణుపు->ne[3] << "] "
             << std::ఎడమ << std::setw(20) << ggml_op_name(కణుపు->op) << std::కుడి << " "
             << std::ఎడమ << std::setw(45) << కణుపు->name << std::కుడి
             << std::setw(2) << "[ "
             << std::setw(0) << కణుపు->nb[0] << ", "
             << std::setw(5) << కణుపు->nb[1] << ", "
             << std::setw(5) << కణుపు->nb[2] << ", "
             << std::setw(5) << కణుపు->nb[3] << "] "
             << std::కుడి << std::setw(15) << buf_name << std::కుడి
             << "\n";

        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            ఒకవేళ (స్వయంచాలక* src = కణుపు->src[i]) {
                // Get buffer type name for source
                స్థిరరాశి అక్షరం * src_buf_name = "none";
                ggml_backend_బఫరు_t src_buf = src->view_src ? src->view_src->బఫరు : src->బఫరు;
                ఒకవేళ (src_buf) {
                    src_buf_name = ggml_backend_బఫరు_name(src_buf);
                }

                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::ఎడమ << std::setw(12) << ggml_op_name(src->op) << std::కుడి;
                file << std::ఎడమ << std::setw(30) << src->name << std::కుడి
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << std::కుడి << std::setw(15) << src_buf_name << std::కుడి
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * కణుపు = cgraph->leafs[i];

        // Get buffer type name for leaf
        స్థిరరాశి అక్షరం * ఆకు_buf_name = "none";
        ggml_backend_బఫరు_t ఆకు_buf = కణుపు->view_src ? కణుపు->view_src->బఫరు : కణుపు->బఫరు;
        ఒకవేళ (ఆకు_buf) {
            ఆకు_buf_name = ggml_backend_బఫరు_name(ఆకు_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << కణుపు->ne[0] << ", "
             << std::setw(5) << కణుపు->ne[1] << "] "
             << std::setw(8) << ggml_op_name(కణుపు->op) << " "
             << std::setw(16) << ggml_get_name(కణుపు)
             << std::setw(20) << ఆకు_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

శూన్య print_tensor_చిరునామా_map(స్థిరరాశి ggml_cgraph * cgraph) {
    std::map<శూన్య *, std::vector<std::string>> చిరునామా_map;
    కొరకు (పూర్ణాంకం కణుపు_n = 0; కణుపు_n < cgraph->n_nodes; కణుపు_n++) {
        స్వయంచాలక * కణుపు = cgraph->nodes[కణుపు_n];
        ఒకవేళ (కణుపు->దత్తాంశం) {
            స్వయంచాలక it = చిరునామా_map.find(కణుపు->దత్తాంశం);
            ఒకవేళ (it == చిరునామా_map.end()) {
                చిరునామా_map[కణుపు->దత్తాంశం] = std::vector<std::string>();
            }
            చిరునామా_map[కణుపు->దత్తాంశం].push_back(కణుపు->name);
        }
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & pair : చిరునామా_map) {
        std::బయటకు << "Address: " << pair.మొదటి << '\n';
        కొరకు (స్థిరరాశి స్వయంచాలక & name : pair.second) {
            std::బయటకు << name << " ; ";
        }
        std::బయటకు << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(స్థిరరాశి ggml_tensor * tensor) {
    std::vector<size_t> shape;
    కొరకు (పూర్ణాంకం i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    తిరిగి shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(స్థిరరాశి ggml_tensor * tensor) {
    std::vector<size_t> stride;
    కొరకు (పూర్ణాంకం i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    తిరిగి stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(స్థిరరాశి ggml_tensor * tensor) {
    ఎంపిక (tensor->type) {
    స్థితి GGML_TYPE_F64:
        తిరిగి ov::element::f64;
    స్థితి GGML_TYPE_F32:
        తిరిగి ov::element::f32;
    స్థితి GGML_TYPE_F16:
        తిరిగి ov::element::f16;
    స్థితి GGML_TYPE_BF16:
        తిరిగి ov::element::bf16;
    స్థితి GGML_TYPE_I8:
        తిరిగి ov::element::i8;
    స్థితి GGML_TYPE_I16:
        తిరిగి ov::element::i16;
    స్థితి GGML_TYPE_I32:
        తిరిగి ov::element::i32;
    స్థితి GGML_TYPE_I64:
        తిరిగి ov::element::i64;
    అప్రమేయ:
        తిరిగి ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_ఆదానం_shape(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి {
    తిరిగి ov::PartialShape(get_shape(m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_ఆదానం_stride(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి {
    తిరిగి get_stride(m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_ఆదానం_కొలత(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        తిరిగి it->second.size();
    }
    తిరిగి 0;
}

size_t GgmlOvDecoder::get_view_ఆదానం_స్థానభ్రంశం(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        ఒకవేళ (view_సూచీ < it->second.size()) {
            తిరిగి it->second[view_సూచీ].second->view_offs;
        }
    }
    తిరిగి 0;
}

size_t GgmlOvDecoder::get_view_ఆదానం_src_స్థానభ్రంశం(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        ఒకవేళ (view_సూచీ < it->second.size()) {
            స్వయంచాలక * view_tensor = it->second[view_సూచీ].second;
            ఒకవేళ (view_tensor && view_tensor->src[0]) {
                తిరిగి view_tensor->src[0]->view_offs;
            }
        }
    }
    తిరిగి 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_ఆదానం_stride(పూర్ణాంకం కణుపు_idx,
                                                         స్థిరరాశి std::string & name,
                                                         size_t view_సూచీ) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        ఒకవేళ (view_సూచీ < it->second.size()) {
            తిరిగి get_stride(it->second[view_సూచీ].second);
        }
    }
    తిరిగి {};
}

std::vector<size_t> GgmlOvDecoder::get_view_ఆదానం_src_stride(పూర్ణాంకం కణుపు_idx,
                                                             స్థిరరాశి std::string & name,
                                                             size_t view_సూచీ) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        ఒకవేళ (view_సూచీ < it->second.size()) {
            స్వయంచాలక * view_tensor = it->second[view_సూచీ].second;
            ఒకవేళ (view_tensor && view_tensor->src[0]) {
                తిరిగి get_stride(view_tensor->src[0]);
            }
        }
    }
    తిరిగి {};
}

ov::Shape GgmlOvDecoder::get_view_ఆదానం_ggml_shape(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        ఒకవేళ (view_సూచీ < it->second.size()) {
            తిరిగి get_shape(it->second[view_సూచీ].second);
        }
    }
    తిరిగి {};
}

ov::Shape GgmlOvDecoder::get_view_ఆదానం_src_ggml_shape(పూర్ణాంకం కణుపు_idx,
                                                       స్థిరరాశి std::string & name,
                                                       size_t view_సూచీ) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        ఒకవేళ (view_సూచీ < it->second.size()) {
            స్వయంచాలక * view_tensor = it->second[view_సూచీ].second;
            ఒకవేళ (view_tensor && view_tensor->src[0]) {
                తిరిగి get_shape(view_tensor->src[0]);
            }
        }
    }
    తిరిగి {};
}

ov::PartialShape GgmlOvDecoder::get_view_ఆదానం_ov_shape(పూర్ణాంకం కణుపు_idx,
                                                        స్థిరరాశి std::string & name,
                                                        size_t view_సూచీ) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        ఒకవేళ (view_సూచీ < it->second.size()) {
            స్వయంచాలక * tensor = it->second[view_సూచీ].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            స్వయంచాలక dynamic_it = m_కణుపు_dynamic_dims.find(tensor);
            ఒకవేళ (dynamic_it != m_కణుపు_dynamic_dims.end() && dynamic_it->second != -1) {
                పూర్ణాంకం dynamic_dim_సూచీ = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_సూచీ] = m_is_static ? get_static_n_టోకెన్లు() : -1;
            }

            తిరిగి shape;
        }
    }
    తిరిగి {};
}

ov::PartialShape GgmlOvDecoder::get_view_ఆదానం_src_ov_shape(పూర్ణాంకం కణుపు_idx,
                                                            స్థిరరాశి std::string & name,
                                                            size_t view_సూచీ) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        ఒకవేళ (view_సూచీ < it->second.size()) {
            స్వయంచాలక * view_tensor = it->second[view_సూచీ].second;
            ఒకవేళ (view_tensor && view_tensor->src[0]) {
                స్వయంచాలక * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                స్వయంచాలక dynamic_it = m_కణుపు_dynamic_dims.find(src_tensor);
                ఒకవేళ (dynamic_it != m_కణుపు_dynamic_dims.end() && dynamic_it->second != -1) {
                    పూర్ణాంకం dynamic_dim_సూచీ = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_సూచీ] = m_is_static ? get_static_n_టోకెన్లు() : -1;
                }

                తిరిగి shape;
            }
        }
    }
    తిరిగి {};
}

std::string GgmlOvDecoder::get_view_ఆదానం_name(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        ఒకవేళ (view_సూచీ < it->second.size()) {
            తిరిగి it->second[view_సూచీ].మొదటి;
        }
    }
    తిరిగి "";
}

std::string GgmlOvDecoder::get_view_ఆదానం_src_name(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.find(name);
    ఒకవేళ (it != m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_views.end()) {
        ఒకవేళ (view_సూచీ < it->second.size()) {
            స్వయంచాలక * view_tensor = it->second[view_సూచీ].second;
            ఒకవేళ (view_tensor && view_tensor->src[0]) {
                తిరిగి get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    తిరిగి "";
}

ov::element::Type GgmlOvDecoder::get_ఆదానం_type(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి {
    తిరిగి get_ov_type(m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs.at(name));
}

size_t GgmlOvDecoder::get_ఆదానం_కొలత() స్థిరరాశి {
    తిరిగి m_model_inputs.size();
}

size_t GgmlOvDecoder::get_ఆదానం_కొలత(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    తిరిగి m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_పేర్లు.size();
}

std::vector<std::string> GgmlOvDecoder::get_ఆదానం_పేర్లు(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    తిరిగి m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs_పేర్లు;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    స్వయంచాలక * ggml_tensor = m_కణుపు_info_list[కణుపు_idx].కణుపు;
    తిరిగి ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(స్థిరరాశి పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    తిరిగి get_ov_type(m_కణుపు_info_list[కణుపు_idx].కణుపు);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    స్వయంచాలక * ggml_tensor = m_కణుపు_info_list[కణుపు_idx].కణుపు;
    తిరిగి get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_పేర్లు(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    స్వయంచాలక * కణుపు = m_కణుపు_info_list[కణుపు_idx].కణుపు;
    ఒకవేళ (కణుపు->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        కొరకు (పూర్ణాంకం i = కణుపు_idx + 1; i < m_cgraph->n_nodes; i++) {
            స్వయంచాలక * వినియోగదారు = m_cgraph->nodes[i];
            ఒకవేళ (వినియోగదారు->op != GGML_OP_VIEW || వినియోగదారు->src[0] != కణుపు) {
                కొనసాగు;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            స్వయంచాలక & name = వినియోగదారు->view_offs == 0 ? attn_name : state_name;
            ఒకవేళ (!name.ఖాళీలు()) {
                తిరిగి {m_కణుపు_info_list[కణుపు_idx].కణుపు_name};
            }
            name = get_tensor_ov_name(m_cgraph, వినియోగదారు);
        }
        ఒకవేళ (!attn_name.ఖాళీలు() && !state_name.ఖాళీలు()) {
            తిరిగి {attn_name, state_name};
        }
    }
    తిరిగి {m_కణుపు_info_list[కణుపు_idx].కణుపు_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    స్వయంచాలక * కణుపు = m_కణుపు_info_list[కణుపు_idx].కణుపు;
    ఒకవేళ (!::is_inplace_op(కణుపు) || కణుపు->view_src == శూన్యసూచిక || ggml_nbytes(కణుపు) == 0) {
        తిరిగి "";
    }
    స్థిరరాశి పూర్ణాంకం op_case = m_కణుపు_info_list[కణుపు_idx].కణుపు_op_case;
    ఒకవేళ (కణుపు->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_చురుకైన_స్థానం_len == -1) {
        తిరిగి "";
    }
    తిరిగి get_tensor_ov_name(m_cgraph, కణుపు->view_src);
}

తార్కిక GgmlOvDecoder::is_view_like_alias_యొక్క(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & view_src_name) స్థిరరాశి {
    స్వయంచాలక * కణుపు = m_కణుపు_info_list[కణుపు_idx].కణుపు;
    ఒకవేళ (కణుపు->view_src == శూన్యసూచిక || get_tensor_ov_name(m_cgraph, కణుపు->view_src) != view_src_name) {
        తిరిగి అబద్ధం;
    }
    తిరిగి కణుపు->op == GGML_OP_RESHAPE || కణుపు->op == GGML_OP_VIEW;
}

స్థిరరాశి std::string & GgmlOvDecoder::get_op_name() స్థిరరాశి {
    స్థిర స్థిరరాశి std::string unknown_name = "UNKNOWN_OP_NAME";
    తిరిగి unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    స్వయంచాలక it = m_కణుపు_dynamic_dims.find(m_కణుపు_info_list[కణుపు_idx].కణుపు);
    ఒకవేళ (it == m_కణుపు_dynamic_dims.end()) {
        తిరిగి -1;
    }
    తిరిగి it->second;
}

స్థిరరాశి std::string & GgmlOvDecoder::get_op_name(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    తిరిగి m_కణుపు_info_list[కణుపు_idx].కణుపు_name;
}

int32_t * GgmlOvDecoder::get_ఆదానం_op_params(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి {
    తిరిగి m_కణుపు_info_list[కణుపు_idx].కణుపు_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    తిరిగి m_కణుపు_info_list[కణుపు_idx].కణుపు->op_params;
}

size_t GgmlOvDecoder::get_output_op_స్థానభ్రంశం(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    తిరిగి m_కణుపు_info_list[కణుపు_idx].కణుపు->view_offs;
}

శూన్య GgmlOvDecoder::సందర్శించు_subgraph(std::function<శూన్య(std::shared_ptr<GgmlDecoder>, పూర్ణాంకం కణుపు_idx)> కణుపు_visitor) స్థిరరాశి {
    కొరకు (పూర్ణాంకం కణుపు_idx = 0; కణుపు_idx < m_cgraph->n_nodes; కణుపు_idx++) {
        ఒకవేళ (m_cgraph->nodes[కణుపు_idx]->op == GGML_OP_NONE) {
            కొనసాగు;
        }
        కణుపు_visitor(std::make_shared<GgmlOvDecoder>(*ఇది), కణుపు_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(స్థిరరాశి ggml_tensor * కణుపు) {
    ఎంపిక (కణుపు->op) {
    స్థితి GGML_OP_UNARY:
        తిరిగి std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(కణుపు));
    స్థితి GGML_OP_GLU:
        తిరిగి std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(కణుపు));
    అప్రమేయ:
        తిరిగి std::string("GGML_OP_") + ggml_op_name(కణుపు->op);
    }
}

స్థిరరాశి std::string & GgmlOvDecoder::get_op_type(పూర్ణాంకం కణుపు_idx) స్థిరరాశి {
    తిరిగి m_కణుపు_info_list[కణుపు_idx].కణుపు_op_type;
}

స్థిరరాశి std::string & GgmlOvDecoder::get_op_type() స్థిరరాశి {
    స్థిర స్థిరరాశి std::string unknown_op = "UNKNOWN_GGML_OP";
    తిరిగి unknown_op;
}

శూన్య GgmlOvDecoder::compute_కణుపు_dynamic_dims() {
    స్వయంచాలక సందర్శించు_కణుపు = [&](స్వయంచాలక && self, ggml_tensor * కణుపు) -> శూన్య {
        ఒకవేళ (!కణుపు) {
            తిరిగి;
        }

        ఒకవేళ (కణుపు->op == GGML_OP_CPY) {
            m_కణుపు_dynamic_dims[కణుపు] = -1;
        }

        ఒకవేళ (m_కణుపు_dynamic_dims.count(కణుపు)) {
            తిరిగి;
        }
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = కణుపు->src[i];
            ఒకవేళ (src == శూన్యసూచిక) {
                కొనసాగు;
            }
            నిర్మాణం ggml_tensor * మూలం_src = శూన్యసూచిక;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            ఒకవేళ (మూలం_src) {
                ఒకవేళ (is_inp_tok(మూలం_src, కణుపు) || is_inp_pos(మూలం_src, కణుపు) || is_output_idx(మూలం_src, కణుపు)) {
                    m_కణుపు_dynamic_dims[మూలం_src] = 0;
                    m_కణుపు_dynamic_dims[src] = m_కణుపు_dynamic_dims[మూలం_src];
                    కొనసాగు;
                }
                self(self, మూలం_src);
                m_కణుపు_dynamic_dims[src] = m_కణుపు_dynamic_dims[మూలం_src];
            } లేకపోతే {
                ఒకవేళ (is_inp_tok(src, కణుపు) || is_inp_pos(src, కణుపు) || is_output_idx(src, కణుపు)) {
                    m_కణుపు_dynamic_dims[src] = 0;
                    కొనసాగు;
                }
                ఒకవేళ (is_inp_scale_అడ్డువరుసలు(src, కణుపు)) {
                    m_కణుపు_dynamic_dims[src] = 1;
                    కొనసాగు;
                }
                ఒకవేళ (కణుపు->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_కణుపు_dynamic_dims[src] = 1;
                    కొనసాగు;
                }
                self(self, src);
            }
        }
        ఎంపిక (కణుపు->op) {
        స్థితి GGML_OP_NONE:
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు->src[1]] != -1) {
                స్వయంచాలక dynamic_dim_idx = m_కణుపు_dynamic_dims[కణుపు->src[1]];
                ఒకవేళ (dynamic_dim_idx == 0) {
                    m_కణుపు_dynamic_dims[కణుపు] = 1;
                } లేకపోతే {
                    స్వయంచాలక dynamic_dim_stride = కణుపు->src[1]->nb[dynamic_dim_idx] / ggml_type_కొలత(కణుపు->src[1]->type) *
                                              ggml_type_కొలత(కణుపు->src[0]->type);
                    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
                        ఒకవేళ (dynamic_dim_stride == కణుపు->src[0]->nb[i]) {
                            m_కణుపు_dynamic_dims[కణుపు] = i;
                            ఆపు;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            ఆపు;
        స్థితి GGML_OP_MUL:
        స్థితి GGML_OP_MUL_MAT:
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు->src[0]] != -1) {
                m_కణుపు_dynamic_dims[కణుపు] = m_కణుపు_dynamic_dims[కణుపు->src[0]];
            }
            ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు->src[1]] != -1) {
                m_కణుపు_dynamic_dims[కణుపు] = m_కణుపు_dynamic_dims[కణుపు->src[1]];
            }
            ఆపు;
        స్థితి GGML_OP_క్రమమార్పు:
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు->src[0]] != -1) {
                స్వయంచాలక dynamic_dim_idx = m_కణుపు_dynamic_dims[కణుపు->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
                    ఒకవేళ (కణుపు->op_params[i] == dynamic_dim_idx) {
                        m_కణుపు_dynamic_dims[కణుపు] = i;
                        ఆపు;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            ఆపు;
        స్థితి GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు->src[0]] != -1) {
                ఒకవేళ (కణుపు->src[0]->op == GGML_OP_NONE) {
                    m_కణుపు_dynamic_dims[కణుపు] = m_కణుపు_dynamic_dims[కణుపు->src[0]];
                    ఆపు;
                }
                స్వయంచాలక dynamic_dim_idx = m_కణుపు_dynamic_dims[కణుపు->src[0]];
                స్వయంచాలక dynamic_dim_విలువ = కణుపు->src[0]->ne[dynamic_dim_idx];
                స్వయంచాలక dynamic_dim_stride =
                    కణుపు->src[0]->nb[dynamic_dim_idx] / ggml_type_కొలత(కణుపు->src[0]->type) * ggml_type_కొలత(కణుపు->type);
                కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
                    ఒకవేళ (కణుపు->nb[i] == dynamic_dim_stride) {
                        m_కణుపు_dynamic_dims[కణుపు] = i;
                        ఆపు;
                    }
                }
                ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు] != -1 && dynamic_dim_విలువ != కణుపు->ne[m_కణుపు_dynamic_dims[కణుపు]]) {
                    m_కణుపు_dynamic_dims[కణుపు] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    ఒకవేళ (ggml_nelements(కణుపు) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      కణుపు->name, కణుపు->src[0]->name);
                    }
                }
            }
            ఆపు;
        }
        స్థితి GGML_OP_TRANSPOSE:
        స్థితి GGML_OP_RESHAPE: {
            ఒకవేళ (is_same_shape(కణుపు->src[0], కణుపు)) {
                m_కణుపు_dynamic_dims[కణుపు] = m_కణుపు_dynamic_dims[కణుపు->src[0]];
                ఆపు;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు->src[0]] != -1) {
                స్వయంచాలక dynamic_dim_idx = m_కణుపు_dynamic_dims[కణుపు->src[0]];
                స్వయంచాలక dynamic_dim_stride = కణుపు->src[0]->nb[dynamic_dim_idx];
                కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
                    ఒకవేళ (కణుపు->nb[i] == dynamic_dim_stride && కణుపు->ne[i] == కణుపు->src[0]->ne[dynamic_dim_idx]) {
                        m_కణుపు_dynamic_dims[కణుపు] = i;
                        ఆపు;
                    }
                }
                ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", కణుపు->name);
                }
            }
            ఆపు;
        }
        స్థితి GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr పూర్ణాంకం q_గమ్యం_బహిర్గతం[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు->src[0]] != -1) {
                స్వయంచాలక q_dynamic_dim = m_కణుపు_dynamic_dims[కణుపు->src[0]];
                m_కణుపు_dynamic_dims[కణుపు] = q_గమ్యం_బహిర్గతం[q_dynamic_dim];
            }
            ఆపు;
        }
        స్థితి GGML_OP_CONT:
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు->src[0]] != -1) {
                స్వయంచాలక dynamic_dim_idx = m_కణుపు_dynamic_dims[కణుపు->src[0]];
                ఒకవేళ (ggml_are_same_shape(కణుపు, కణుపు->src[0])) {
                    m_కణుపు_dynamic_dims[కణుపు] = dynamic_dim_idx;
                } లేకపోతే {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_కొలత(కణుపు->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (కణుపు->src[0]->ne[0] / ggml_blck_కొలత(కణుపు->src[0]->type));
                    కొరకు (పూర్ణాంకం i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * కణుపు->src[0]->ne[i - 1];
                    }

                    స్వయంచాలక dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_కొలత(కణుపు->src[0]->type) *
                                              ggml_type_కొలత(కణుపు->type);
                    పూర్ణాంకం matched_dim_లెక్క = 0;
                    పూర్ణాంకం మొదటి_matched_dim = -1;
                    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
                        ఒకవేళ (కణుపు->nb[i] == dynamic_dim_stride && కణుపు->ne[i] == కణుపు->src[0]->ne[dynamic_dim_idx]) {
                            ఒకవేళ (మొదటి_matched_dim == -1) {
                                మొదటి_matched_dim = i;
                            }
                            m_కణుపు_dynamic_dims[కణుపు] = i;
                            matched_dim_లెక్క++;
                        }
                    }
                    ఒకవేళ (matched_dim_లెక్క > 1 && కణుపు->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_కణుపు_dynamic_dims[కణుపు] = మొదటి_matched_dim;
                    } లేకపోతే ఒకవేళ (matched_dim_లెక్క != 1) {
                        m_కణుపు_dynamic_dims[కణుపు] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      కణుపు->name, కణుపు->src[0]->name);
                    }
                }
            }
            ఆపు;
        స్థితి GGML_OP_CONCAT:
            కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
                ఒకవేళ (కణుపు->src[0]->ne[i] != కణుపు->ne[i]) {
                    m_కణుపు_dynamic_dims[కణుపు] = i;
                    ఆపు;
                }
            }
            ఆపు;
        స్థితి GGML_OP_SSM_CONV:
        స్థితి GGML_OP_GATED_DELTA_NET:
            m_కణుపు_dynamic_dims[కణుపు] = 1;
            ఆపు;
        స్థితి GGML_OP_RMS_NORM:
        స్థితి GGML_OP_L2_NORM:
        స్థితి GGML_OP_NORM:
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_SUB:
        స్థితి GGML_OP_GLU:
        స్థితి GGML_OP_ROPE:
        స్థితి GGML_OP_SCALE:
        స్థితి GGML_OP_SOFT_MAX:
        స్థితి GGML_OP_ARGSORT:
        స్థితి GGML_OP_కూడు_ID:
        స్థితి GGML_OP_UNARY:
        స్థితి GGML_OP_CUMSUM:
        స్థితి GGML_OP_FILL:
        స్థితి GGML_OP_సమితి:
        స్థితి GGML_OP_DIAG:
        స్థితి GGML_OP_TRI:
        స్థితి GGML_OP_REPEAT:
        స్థితి GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        స్థితి GGML_OP_DIV:
        స్థితి GGML_OP_CLAMP:
        స్థితి GGML_OP_PAD:
        స్థితి GGML_OP_UPSCALE:
        స్థితి GGML_OP_SIN:
        స్థితి GGML_OP_COS:
        స్థితి GGML_OP_LOG:
        స్థితి GGML_OP_ROLL:
            m_కణుపు_dynamic_dims[కణుపు] = m_కణుపు_dynamic_dims[కణుపు->src[0]];
            ఆపు;
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_కణుపు_dynamic_dims[కణుపు] =
                (m_కణుపు_dynamic_dims[కణుపు->src[0]] == 0) ? -1 : m_కణుపు_dynamic_dims[కణుపు->src[0]];
            ఆపు;
        స్థితి GGML_OP_MUL_MAT_ID:
        స్థితి GGML_OP_SOLVE_TRI:
            m_కణుపు_dynamic_dims[కణుపు] = m_కణుపు_dynamic_dims[కణుపు->src[1]];
            ఆపు;
        స్థితి GGML_OP_CPY:
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
        స్థితి GGML_OP_కూడిక:
        స్థితి GGML_OP_సగటు:
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఆపు;
        స్థితి GGML_OP_IM2_నిలువు: {
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు->src[1]] != -1) {
                స్థిరరాశి తార్కిక is_2D = కణుపు->op_params[6] == 1;
                స్థిరరాశి పూర్ణాంకం src_dyn = m_కణుపు_dynamic_dims[కణుపు->src[1]];
                ఒకవేళ (is_2D) {
                    ఒకవేళ (src_dyn == 0) {
                        m_కణుపు_dynamic_dims[కణుపు] = 1;  // IW -> OW
                    } లేకపోతే ఒకవేళ (src_dyn == 1) {
                        m_కణుపు_dynamic_dims[కణుపు] = 2;  // IH -> OH
                    } లేకపోతే ఒకవేళ (src_dyn == 3) {
                        m_కణుపు_dynamic_dims[కణుపు] = 3;  // N  -> N
                    }
                } లేకపోతే {
                    ఒకవేళ (src_dyn == 0) {
                        m_కణుపు_dynamic_dims[కణుపు] = 1;  // IW -> OW
                    } లేకపోతే ఒకవేళ (src_dyn == 2) {
                        m_కణుపు_dynamic_dims[కణుపు] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు] != -1) {
                    OPENVINO_ASSERT(కణుపు->src[1]->ne[src_dyn] == కణుపు->ne[m_కణుపు_dynamic_dims[కణుపు]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(కణుపు->name) +
                                        " and its src[1]: " + std::string(కణుపు->src[1]->name));
                }
            }
            ఆపు;
        }
        స్థితి GGML_OP_IM2_నిలువు_3_D: {
            m_కణుపు_dynamic_dims[కణుపు] = -1;
            ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు->src[1]] != -1) {
                స్థిరరాశి పూర్ణాంకం src_dyn = m_కణుపు_dynamic_dims[కణుపు->src[1]];
                ఒకవేళ (src_dyn == 0) {
                    m_కణుపు_dynamic_dims[కణుపు] = 1;  // IW -> OW
                } లేకపోతే ఒకవేళ (src_dyn == 1) {
                    m_కణుపు_dynamic_dims[కణుపు] = 2;  // IH -> OH
                } లేకపోతే ఒకవేళ (src_dyn == 3) {
                    m_కణుపు_dynamic_dims[కణుపు] = 3;  // N  -> N
                }
                ఒకవేళ (m_కణుపు_dynamic_dims[కణుపు] != -1) {
                    OPENVINO_ASSERT(కణుపు->src[1]->ne[src_dyn] == కణుపు->ne[m_కణుపు_dynamic_dims[కణుపు]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(కణుపు->name) +
                                        " and its src[1]: " + std::string(కణుపు->src[1]->name));
                }
            }
            ఆపు;
        }
        అప్రమేయ:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(కణుపు->op), కణుపు->name);
            ఆపు;
        }
    };

    కొరకు (పూర్ణాంకం i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * కణుపు = m_cgraph->nodes[i];
        సందర్శించు_కణుపు(సందర్శించు_కణుపు, కణుపు);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    ఒకవేళ (0) {
        కొరకు (పూర్ణాంకం i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * కణుపు = m_cgraph->nodes[i];
            పూర్ణాంకం dynamic_dim = m_కణుపు_dynamic_dims[కణుపు];
            std::బయటకు << "[" << i << "] " << "node_name: " << కణుపు->name << " op: " << ggml_op_name(కణుపు->op)
                      << " shape: [";
            కొరకు (పూర్ణాంకం j = 0; j < 4; j++) {
                ఒకవేళ (j == dynamic_dim) {
                    std::బయటకు << "*";
                } లేకపోతే {
                    std::బయటకు << కణుపు->ne[j];
                }
                ఒకవేళ (j < 3) {
                    std::బయటకు << ", ";
                }
            }
            std::బయటకు << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = కణుపు->src[j];
                ఒకవేళ (src == శూన్యసూచిక) {
                    కొనసాగు;
                }
                పూర్ణాంకం src_dynamic_dim = m_కణుపు_dynamic_dims[src];
                std::బయటకు << "    [" << j << "] src_name: " << src->name << " [";
                కొరకు (పూర్ణాంకం k = 0; k < 4; k++) {
                    ఒకవేళ (k == src_dynamic_dim) {
                        std::బయటకు << "*";
                    } లేకపోతే {
                        std::బయటకు << src->ne[k];
                    }
                    ఒకవేళ (k < 3) {
                        std::బయటకు << ", ";
                    }
                }
                std::బయటకు << "]" << '\n';
            }
            std::బయటకు << '\n';
        }
    }
}
