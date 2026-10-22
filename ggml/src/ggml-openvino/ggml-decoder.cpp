#شامل "ggml-decoder.h"

#شامل "ggml-impl.h"
#شامل "ggml-openvino-extra.h"
#شامل "ggml-openvino.h"
#شامل "ggml-quants.h"
#شامل "ggml.h"
#شامل "utils.h"

#شامل <algorithm>
#شامل <cassert>
#شامل <cstddef>
#شامل <cstdint>
#شامل <cstdlib>
#شامل <fstream>
#شامل <iomanip>
#شامل <map>
#شامل <memory>
#شامل <mutex>
#شامل <openvino/core/dimension.hpp>
#شامل <openvino/core/except.hpp>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/partial_shape.hpp>
#شامل <openvino/core/type/bfloat16.hpp>
#شامل <openvino/core/type/element_type.hpp>
#شامل <openvino/core/type/float16.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/runtime/tensor.hpp>
#شامل <ostream>
#شامل <set>
#شامل <stdexcept>
#شامل <string>
#شامل <cstring>
#شامل <unordered_map>
#شامل <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             منطقی is_static,
                             منطقی is_stateful,
                             منطقی model_is_splitted,
                             منطقی is_prefill,
                             صحیح prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(نادرست),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    ایستا منطقی printed_address_map = نادرست;
    اگر (!printed_address_map) {
        اگر (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = درست;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_درون_داد_output();
    compute_گره_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    برای (صحیح گره_n = 0; گره_n < cgraph->n_nodes; گره_n++) {
        m_گره_info_list[گره_n].گره_op_case = compute_op_case(m_گره_info_list[گره_n].گره);
        m_گره_info_list[گره_n].گره_op_type = compute_op_type(m_گره_info_list[گره_n].گره);
    }

    add_extra_inputs();
}

تهی GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_گره_info_list.clear();
    set_درون_داد_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
منطقی GgmlOvDecoder::is_bound_to(ثابت ggml_cgraph * cgraph) ثابت {
    بازگشت m_cgraph == cgraph && cgraph->n_nodes > 0 && m_گره_info_list.size() == (size_t) cgraph->n_nodes &&
           m_گره_info_list.front().گره == cgraph->nodes[0] &&
           m_گره_info_list.back().گره == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = درست;
    set_درون_داد_output();
    compute_model_inputs();
    compute_model_outputs();
    برای (صحیح گره_n = 0; گره_n < cgraph->n_nodes; گره_n++) {
        m_گره_info_list[گره_n].گره_op_case = compute_op_case(m_گره_info_list[گره_n].گره);
        m_گره_info_list[گره_n].گره_op_type = compute_op_type(m_گره_info_list[گره_n].گره);
    }
}

فضاینام {
منطقی is_inplace_op(ثابت ggml_tensor * گره) {
    بازگشت گره->op == GGML_OP_SET_ROWS || گره->op == GGML_OP_CPY || (گره->op == GGML_OP_SCALE && گره->view_src);
}

منطقی is_same_shape(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        اگر (a->ne[i] != b->ne[i]) {
            بازگشت نادرست;
        }
    }
    بازگشت درست;
}

منطقی is_conv_states_all_tensor(ثابت ggml_tensor * tensor) {
    بازگشت tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

منطقی is_full_single_slot_writeback(ثابت ggml_tensor * گره) {
    بازگشت گره->view_src != nullptr && گره->view_src->ne[1] == 1 && گره->src[1] != nullptr &&
           گره->src[1]->op == GGML_OP_VIEW && گره->src[1]->view_src == گره->view_src &&
           گره->src[1]->view_offs == 0 && ggml_nbytes(گره->src[1]) == ggml_nbytes(گره->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
منطقی is_moe_expert_sum_add(ثابت ggml_tensor * گره) {
    std::vector<ثابت ggml_tensor *> leaves;
    ثابت ggml_tensor * cur = گره;
    وقتی (cur->op == GGML_OP_ADD) {
        اگر (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            بازگشت نادرست;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ثابت ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    برای (ثابت ggml_tensor * leaf : leaves) {
        اگر (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            بازگشت نادرست;
        }
        ثابت ggml_tensor * leaf_base = leaf->src[0];
        اگر (base == nullptr) {
            base = leaf_base;
        } وگرنه اگر (leaf_base != base) {
            بازگشت نادرست;
        }
        اگر (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            بازگشت نادرست;
        }
        اگر (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            بازگشت نادرست;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        اگر (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            بازگشت نادرست;
        }
    }

    بازگشت base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor) {
    اگر (tensor == nullptr) {
        بازگشت "";
    }
    اگر ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ثابت auto * گره = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        اگر (گره != cgraph->nodes + cgraph->n_nodes) {
            بازگشت std::string(tensor->name) + "#n" + std::to_string(گره - cgraph->nodes);
        }
        ثابت auto * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        اگر (leaf != cgraph->leafs + cgraph->n_leafs) {
            بازگشت std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    بازگشت tensor->name;
}

ایستا std::string get_tensor_ov_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor) {
    بازگشت GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

ایستا std::string get_tensor_graph_درون_داد_ov_name(ثابت GgmlOvDecoder * decoder,
                                                  ثابت ggml_cgraph * cgraph,
                                                  ثابت ggml_tensor * tensor,
                                                  ثابت ggml_tensor * op) {
    اگر (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        بازگشت "inp_pos";
    }
    اگر (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        بازگشت "embd";
    }
    اگر (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ثابت منطقی is_swa = decoder->is_swa_mask(tensor);
        اگر (decoder->is_stateful()) {
            بازگشت is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        اگر (is_swa) {
            بازگشت get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    بازگشت get_tensor_ov_name(cgraph, tensor);
}

تهی GgmlOvDecoder::set_درون_داد_output() {
    برای (صحیح گره_n = 0; گره_n < m_cgraph->n_nodes; گره_n++) {
        auto * گره = m_cgraph->nodes[گره_n];

        گره_Info current_گره_info;
        auto گره_name = get_tensor_ov_name(m_cgraph, گره);

        current_گره_info.گره = گره;
        current_گره_info.گره_name = گره_name;
        current_گره_info.گره_op_case = 0;
        current_گره_info.data_addr = گره->data;

        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = گره->src[i];
            اگر (src == nullptr) {
                ادامه;
            }
            auto src_name = get_tensor_ov_name(m_cgraph, src);
            اگر (src->flags & GGML_TENSOR_FLAG_درون_داد) {
                src_name = get_tensor_graph_درون_داد_ov_name(این, m_cgraph, src, گره);
            }
            current_گره_info.گره_inputs[src_name] = src;
            current_گره_info.گره_inputs_names.push_back(src_name);

            اگر (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_گره_info.گره_inputs_views[src_name])> view_chain;
                auto * current = src;

                وقتی (current != nullptr) {
                    auto current_name = get_tensor_ov_name(m_cgraph, current);
                    اگر (current->flags & GGML_TENSOR_FLAG_درون_داد) {
                        current_name = get_tensor_graph_درون_داد_ov_name(این, m_cgraph, current, گره);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    اگر (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } وگرنه {
                        بشکن;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_گره_info.گره_inputs_views[src_name] = view_chain;
            }
        }

        m_گره_info_list.push_back(current_گره_info);
    }
}

صحیح GgmlOvDecoder::compute_op_case(ثابت ggml_tensor * گره) ثابت {
    صحیح op_case = 0;
    انتخاب (گره->op) {
    حالت GGML_OP_RESHAPE: {
        اگر (m_naive) {
            بشکن;
        }
        auto name = std::string(گره->name);
        auto * src = گره->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        منطقی recurrent_sequence = نادرست;
        برای (صحیح i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ثابت auto * consumer = m_cgraph->nodes[i];
            اگر (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == گره) {
                بازگشت 1;
            } وگرنه اگر (consumer->op == GGML_OP_SSM_CONV) {
                ثابت auto * concat = consumer->src[0];
                اگر (concat->op == GGML_OP_CONCAT) {
                    ثابت auto * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == گره;
                }
            } وگرنه اگر (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                ثابت auto * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == گره;
            }
        }
        اگر (recurrent_sequence && گره->ne[0] == src->ne[0] && گره->ne[3] == 1) {
            بازگشت 6;
        }
        اگر (گره->ne[0] == src->ne[0] && گره->ne[2] == 1 && گره->ne[3] == 1) {
            بازگشت 5;
        }
        اگر (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == گره->ne[0] && src->src[0]->ne[1] == گره->ne[1]) {
            op_case = 4;
        } وگرنه اگر (گره->ne[0] * گره->ne[1] == src->ne[0]) {
            op_case = 1;
        } وگرنه اگر (src->ne[0] * src->ne[1] == گره->ne[0]) {
            op_case = 2;
            اگر (src->ne[2] * src->ne[3] == گره->ne[1]) {
                op_case = 5;
            }
        } وگرنه اگر (گره->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == گره->ne[1]) {
            op_case = 3;
        } وگرنه اگر (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } وگرنه اگر (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } وگرنه اگر (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        اگر (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            ثابت auto * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            منطقی recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            برای (صحیح i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ثابت auto * consumer = m_cgraph->nodes[i];
                اگر (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    برای (صحیح j : {3, 4}) {
                        ثابت auto * gate = consumer->src[j];
                        اگر (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == گره;
                    }
                } وگرنه اگر (consumer->op == GGML_OP_MUL) {
                    برای (صحیح j = 0; j < 2; ++j) {
                        ثابت auto * gate = consumer->src[j];
                        ثابت auto * norm = consumer->src[1 - j];
                        اگر (gate->op != GGML_OP_UNARY || gate->src[0] != گره) {
                            ادامه;
                        }
                        اگر (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            اگر (recurrent) {
                op_case = 9;
            }
        }
        بشکن;
    }
    حالت GGML_OP_PERMUTE: {
        اگر (گره->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } وگرنه اگر (گره->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(گره->view_src->name);
            صحیح layer = extract_layer_from_name(src_name).value();
            اگر (ggml_is_contiguous(گره->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                اگر (!is_swa_layer(layer)) {
                    op_case = 3;
                } وگرنه {
                    op_case = 4;
                }
            } وگرنه {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                اگر (!is_swa_layer(layer)) {
                    op_case = 5;
                } وگرنه {
                    op_case = 6;
                }
            }
        } وگرنه {
            // rope'ed query tensor
            op_case = 2;
        }
        بشکن;
    }
    حالت GGML_OP_MUL_MAT: {
        اگر (گره->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        بشکن;
    }
    حالت GGML_OP_GET_ROWS: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        اگر (گره->src[0]->op == GGML_OP_RESHAPE && گره->src[0]->src[0] != nullptr &&
            is_recurrent_cache(گره->src[0]->src[0])) {
            ثابت منطقی single_slot = گره->src[0]->src[0]->ne[1] == 1;
            اگر (گره->src[1]->op == GGML_OP_VIEW) {
                op_case = (گره->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } وگرنه اگر (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        بشکن;
    }
    حالت GGML_OP_ROPE: {
        ثابت صحیح mode = گره->op_params[2];
        انتخاب (mode) {
        حالت GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            بشکن;
        }
        حالت GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            بشکن;
        }
        حالت GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            بشکن;
        }
        حالت GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            بشکن;
        }
        پیشفرض:
            op_case = 0;
            بشکن;
        }
        بشکن;
    }
    حالت GGML_OP_VIEW: {
        اگر (!m_model_params.has_rs_rollback && گره->src[0] != nullptr &&
            گره->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            بشکن;
        }
        اگر (m_is_static && گره->src[0] != nullptr &&
            (گره->src[0]->op == GGML_OP_GATED_DELTA_NET || گره->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            بشکن;
        }
        اگر (گره->src[0]->op == GGML_OP_VIEW) {
            auto * src = گره->src[0];
            اگر (ggml_nelements(گره) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            اگر (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            auto * src = گره->src[0];
            اگر (ggml_nelements(گره) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                اگر (گره->ne[0] == src->ne[0] && گره->ne[1] == src->ne[2] && گره->ne[2] == 1 &&
                    گره->nb[0] == src->nb[0] && گره->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    بشکن;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                صحیح diff_شمار = 0;
                برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                    اگر (گره->ne[i] != src->ne[i]) {
                        diff_شمار++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    اگر (گره->ne[i] > src->ne[i]) {
                        بازگشت 0;
                    }
                }
                اگر (diff_شمار >= 1) {
                    op_case = 0;
                }
            }
        }
        بشکن;
    }
    حالت GGML_OP_RMS_NORM: {
        اگر (گره->src[0]->op == GGML_OP_VIEW) {
            اگر (is_same_shape(گره->src[0]->src[0], گره->src[0])) {
                op_case = 1;
            } وگرنه اگر (!m_model_params.has_rs_rollback &&
                       گره->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } وگرنه اگر (گره->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        بشکن;
    }
    حالت GGML_OP_POOL_2D: {
        ثابت ggml_op_pool pool_mode = static_cast<ggml_op_pool>(گره->op_params[0]);
        انتخاب (pool_mode) {
        حالت GGML_OP_POOL_MAX: {
            op_case = 1;
            بشکن;
        }
        حالت GGML_OP_POOL_AVG: {
            op_case = 2;
            بشکن;
        }
        پیشفرض:
            op_case = 0;
            بشکن;
        }
        بشکن;
    }
    حالت GGML_OP_UPSCALE: {
        ثابت int32_t mode_flags = گره->op_params[0];
        ثابت ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        انتخاب (scale_mode) {
        حالت GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            بشکن;
        }
        حالت GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            بشکن;
        }
        حالت GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            بشکن;
        }
        پیشفرض:
            op_case = 0;
            بشکن;
        }
        بشکن;
    }
    حالت GGML_OP_CPY: {
        اگر (گره->src[0]->op == GGML_OP_VIEW) {
            اگر (گره->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                اگر (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(گره) ? 7 : 10;
                } وگرنه {
                    op_case = 1;
                }
            } وگرنه اگر (GgmlOvDecoder::is_conv_state_writeback(گره)) {
                op_case = is_full_single_slot_writeback(گره) ? 8 : 2;
                بشکن;
            } وگرنه اگر (is_conv_states_all_tensor(گره->view_src) && گره->src[1] != nullptr &&
                       گره->src[1]->op == GGML_OP_VIEW && گره->src[1]->view_src == گره->view_src) {
                op_case = 4;
                بشکن;
            } وگرنه اگر (گره->src[0]->src[0]->op == GGML_OP_GET_ROWS && گره->src[1] != nullptr &&
                       گره->src[1]->op == GGML_OP_VIEW && گره->src[1]->view_src != nullptr &&
                       is_recurrent_cache(گره->src[1]->view_src) && گره->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } وگرنه اگر (گره->src[0]->op == GGML_OP_GET_ROWS && گره->src[1] != nullptr &&
                   گره->src[1]->op == GGML_OP_VIEW && گره->src[1]->view_src != nullptr &&
                   is_recurrent_cache(گره->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = گره->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } وگرنه اگر (گره->src[1] != nullptr && گره->src[1]->op == GGML_OP_VIEW && گره->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ثابت ggml_tensor * kv_buf = گره->src[1]->view_src;
            اگر (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                برای (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
                    ثابت ggml_tensor * n = m_cgraph->nodes[i];
                    اگر (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        ادامه;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    اگر ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        اگر (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        بشکن;
                    }
                }
            }
        }
        بشکن;
    }
    حالت GGML_OP_ADD: {
        اگر (is_moe_expert_sum_add(گره)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        بشکن;
    }
    حالت GGML_OP_SCALE: {
        اگر (گره->view_src && گره->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = گره->view_src->ne[1] == 1 ? 2 : 1;
        }
        بشکن;
    }
    حالت GGML_OP_L2_NORM: {
        اگر (std::string(گره->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        بشکن;
    }
    حالت GGML_OP_FLASH_ATTN_EXT: {
        اگر (گره->src[1] != nullptr && گره->src[1]->op == GGML_OP_VIEW && گره->src[1]->view_src != nullptr) {
            ثابت ggml_tensor * kv_buf = گره->src[1]->view_src;
            اگر (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (گره->src[3] != nullptr) ? 1 : 2;
            }
        }
        بشکن;
    }
    پیشفرض:
        بشکن;
    }
    بازگشت op_case;
}

std::optional<صحیح> extract_layer_from_name(ثابت std::string & name) {
    size_t pos1 = name.find("_l");
    اگر (pos1 == std::string::npos) {
        بازگشت std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    اگر (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    صحیح layer = std::stoi(layer_str);
    بازگشت layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
ایستا صحیح get_swa_window_from_mask(ثابت ggml_tensor * mask) {
    اگر (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        بازگشت -1;
    }
    اگر (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    ثابت int64_t n_kv = mask->ne[0];
    ثابت int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    برای (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        برای (int64_t c = 0; c < n_kv; c++) {
            ثابت size_t i = (size_t) r * n_kv + c;
            ثابت اعشاری v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((ثابت ggml_fp16_t *) mask->data)[i]) :
                                                          ((ثابت اعشاری *) mask->data)[i];
            اگر (v > -INFINITY) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    بازگشت window > 0 ? (صحیح) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, منطقی is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    auto get_attention_pattern_case = [](ثابت ggml_tensor * گره) -> صحیح {
        اگر (گره == nullptr) {
            بازگشت -1;
        }

        انتخاب (گره->op) {
        حالت GGML_OP_FLASH_ATTN_EXT:
            اگر (گره->src[0] == nullptr || گره->src[1] == nullptr) {
                بازگشت -1;
            }
            انتخاب (گره->src[1]->op) {
            حالت GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                اگر (گره->src[3] != nullptr && گره->src[1]->src[0] != nullptr &&
                    گره->src[1]->src[0]->op == GGML_OP_VIEW) {
                    بازگشت 0;
                }
                بشکن;
            حالت GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                اگر (گره->src[3] != nullptr && گره->src[1]->src[0] != nullptr &&
                    گره->src[1]->src[0]->op == GGML_OP_PERMUTE && گره->src[1]->src[0]->src[0] != nullptr &&
                    گره->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    بازگشت 1;
                }
                بشکن;
            حالت GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                اگر (گره->src[1]->view_src != nullptr) {
                    اگر (گره->src[3] != nullptr) {
                        بازگشت 4;  // decoder self-attention
                    }
                    بازگشت 5;      // cross-attention or encoder self-attention
                }
                بشکن;
            پیشفرض:
                بشکن;
            }
            بشکن;
        حالت GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            اگر (گره->src[0] != nullptr && گره->src[1] != nullptr && گره->src[0]->op == GGML_OP_MUL_MAT &&
                گره->src[0]->src[0] != nullptr && گره->src[0]->src[1] != nullptr &&
                گره->src[0]->src[0]->op == GGML_OP_PERMUTE && گره->src[0]->src[0]->src[0] != nullptr &&
                گره->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                بازگشت 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            اگر (گره->src[0]->op == GGML_OP_ADD && گره->src[0]->src[0] != nullptr &&
                گره->src[0]->src[0]->op == GGML_OP_MUL_MAT && گره->src[0]->src[0]->src[0] != nullptr &&
                گره->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                بازگشت 3;
            }
            بشکن;
        پیشفرض:
            بشکن;
        }

        بازگشت -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    auto get_attention_op_mask = [&get_attention_pattern_case](ثابت ggml_tensor * گره) -> ثابت ggml_tensor * {
        انتخاب (get_attention_pattern_case(گره)) {
        حالت 0:
        حالت 1:
            بازگشت گره->src[3];
        حالت 2:
        حالت 3:
            بازگشت گره->src[1];
        پیشفرض:
            بازگشت nullptr;
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
        std::map<صحیح, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<صحیح, ثابت ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
            ثابت ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            اگر (mask == nullptr) {
                ادامه;
            }
            ثابت ggml_tensor * cache_k_permute = nullptr;
            انتخاب (get_attention_pattern_case(cgraph->nodes[i])) {
            حالت 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     بشکن;
            حالت 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             بشکن;
            حالت 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             بشکن;
            پیشفرض: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     بشکن;
            }
            ثابت ggml_tensor * cache_k_view = cache_k_permute->src[0];
            اگر (cache_k_view->op != GGML_OP_VIEW) {
                ادامه;
            }
            ثابت ggml_tensor * leaf = cache_k_view->src[0];
            auto layer = extract_layer_from_name(leaf->name);
            اگر (!layer.has_value()) {
                ادامه;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        برای (ثابت auto & [layer, extent] : layer_extent) {
            اگر (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                اگر (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        اگر (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            برای (ثابت auto & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (long) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    منطقی rope_seen = نادرست;
    برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گره = cgraph->nodes[i];
        ثابت صحیح attention_pattern_case = get_attention_pattern_case(گره);
        اگر (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            انتخاب (attention_pattern_case) {
            حالت 0:
                cache_k_permute = گره->src[1];
                mask = گره->src[3];
                بشکن;
            حالت 1:
                cache_k_permute = گره->src[1]->src[0];
                mask = گره->src[3];
                بشکن;
            حالت 2:
                cache_k_permute = گره->src[0]->src[0];
                mask = گره->src[1];
                بشکن;
            حالت 3:
                cache_k_permute = گره->src[0]->src[0]->src[0];
                mask = گره->src[1];
                بشکن;
            حالت 4:
            حالت 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                auto * cache_k_view = گره->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = گره->src[0]->ne[1];
                اگر (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } وگرنه {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                ادامه;
            }
            پیشفرض:
                بشکن;
            }

            أكد(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.درون_داد_len = گره->src[0]->ne[1];
            compute_params.token_len_per_seq = گره->src[0]->ne[1];

            auto * cache_k_view = cache_k_permute->src[0];
            اگر (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                ادامه;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            صحیح layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ثابت منطقی layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            اگر (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } وگرنه {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, اندازه(size_t));
            compute_params.seq_active_start = offset / seq_size;

            اگر (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } وگرنه {
                compute_params.attention_size = mask->ne[0];
            }
            اگر (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        اگر (گره->op == GGML_OP_MUL_MAT && گره->src[0]->op == GGML_OP_PERMUTE &&
            گره->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(گره->src[0]->view_src, گره->view_src)) {
            اگر (گره->src[1]->op == GGML_OP_PERMUTE && گره->src[1]->src[0]->op == GGML_OP_VIEW &&
                گره->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = گره->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        اگر (گره->op == GGML_OP_TRANSPOSE && گره->src[0]->op == GGML_OP_PERMUTE &&
            گره->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = گره->ne[0];
            اگر (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        اگر (گره->op == GGML_OP_ROPE) {
            اگر (compute_params.token_len_per_seq == -1 && گره->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(گره->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(اندازه(model_params.rope_params) == اندازه(int32_t) * 16, "rope_params size");
            اگر (!rope_seen) {
                memcpy(model_params.rope_params, گره->op_params, اندازه(int32_t) * 16);
                rope_seen = درست;
            } وگرنه اگر (memcmp(model_params.rope_params, گره->op_params, اندازه(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = درست;
            }
        }
        اگر (گره->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = گره->src[0]->ne[0];
        }
        اگر (گره->op == GGML_OP_SCALE && گره->view_src != nullptr && is_recurrent_cache(گره->view_src)) {
            اگر (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = گره->view_src->ne[1];
            } وگرنه {
                GGML_ASSERT(model_params.n_rs_slots == گره->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(گره) / گره->view_src->ne[0];
            compute_params.cache_rs_reset_idx = گره->src[0]->view_offs / گره->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        اگر (گره->op == GGML_OP_CPY && گره->view_src != nullptr && is_recurrent_cache(گره->view_src) &&
            گره->src[1] != nullptr && گره->src[1]->op == GGML_OP_VIEW && گره->src[1]->view_src == گره->view_src) {
            ثابت منطقی is_conv = is_conv_state_writeback(گره);
            ثابت منطقی is_gdn = گره->src[0]->op == GGML_OP_VIEW && گره->src[0]->src[0] != nullptr &&
                                گره->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ثابت منطقی is_extra = گره->src[0]->op == GGML_OP_GET_ROWS;
            ثابت منطقی is_gdn_rollback = is_gdn && is_same_shape(گره->src[0], گره->src[1]);

            ثابت ggml_tensor * dest_view = گره->src[1];
            ثابت ggml_tensor * cache = گره->view_src;
            ثابت size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            اگر (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = درست;
            }
            اگر (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(گره)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (صحیح) (dest_view->view_offs / row_bytes);
                اگر (is_conv) {
                    writeback.src_begin = (صحیح) (گره->src[0]->view_offs / گره->src[0]->view_src->nb[0]);
                } وگرنه اگر (is_gdn_rollback) {
                    writeback.src_begin = (صحیح) (گره->src[0]->view_offs / گره->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, گره)] = writeback;
            }
            اگر ((is_conv || is_gdn) && !is_full_single_slot_writeback(گره)) {
                compute_params.s_copy_active_slot_len = (صحیح) dest_view->ne[1];
            }
        }
    }
    اگر (model_params.n_heads_kv == -1) {
        برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
            ثابت auto * گره = cgraph->nodes[i];
            ثابت ggml_tensor * mask = nullptr;
            اگر (گره->op == GGML_OP_SOFT_MAX) {
                mask = گره->src[1];
            } وگرنه اگر (گره->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = گره->src[3];
            } وگرنه {
                ادامه;
            }
            اگر (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_درون_داد) ||
                گره->src[0] == nullptr) {
                ادامه;
            }
            model_params.is_cacheless_attn = درست;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.درون_داد_len = گره->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.درون_داد_len;
            بشکن;
        }
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    اگر (model_params.is_cacheless_attn) {
        برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
            ثابت auto * گره = cgraph->nodes[i];
            اگر (گره->op == GGML_OP_GET_ROWS && is_output_idx(گره->src[1], گره)) {
                compute_params.output_len = گره->src[1]->ne[0];
                بشکن;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    اگر (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    بازگشت {model_params, compute_params};
}

تهی GgmlOvDecoder::validate_cgraph() ثابت {
    اگر (m_model_params.n_seq > 1 && m_is_static == درست) {
        پرتابکن std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_درون_داد_shape(ثابت ggml_tensor * op,
                                                      ثابت ggml_tensor * درون_داد,
                                                      صحیح dynamic_dim_index) ثابت {
    اگر (m_naive) {
        بازگشت درون_داد != nullptr ? ov::PartialShape{get_shape(درون_داد)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape درون_داد_shape;

    اگر (is_inp_tok(درون_داد, op) || is_inp_pos(درون_داد, op)) {
        // tokens or positions
        صحیح len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        اگر (m_is_static && is_inp_pos(درون_داد, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        درون_داد_shape = ov::PartialShape{1, 1, 1, len};

    } وگرنه اگر (is_output_idx(درون_داد, op)) {
        // output index
        درون_داد_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } وگرنه اگر (is_inp_mean(درون_داد, op)) {
        درون_داد_shape = m_is_static ? ov::PartialShape{1, 1, درون_داد->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } وگرنه اگر (is_inp_scale_rows(درون_داد, op)) {
        درون_داد_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } وگرنه اگر (is_inp_mask(درون_داد, op)) {
        // mask
        اگر (m_is_static) {
            درون_داد_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } وگرنه اگر (m_is_stateful) {
            درون_داد_shape = ov::PartialShape{1, 1, -1, -1};
        } وگرنه {
            درون_داد_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } وگرنه اگر (is_recurrent_cache(درون_داد)) {
        درون_داد_shape = ov::PartialShape{get_shape(درون_داد)};
        اگر (!m_is_static && !m_is_stateful && درون_داد->ne[1] > 1) {
            درون_داد_shape[2] = -1;
        }

    } وگرنه اگر (is_kvcache(درون_داد, op)) {
        // kvcache
        درون_داد_shape = ov::PartialShape{get_shape(درون_داد)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ثابت منطقی is_flat_kv = (درون_داد->ne[1] == 1 && درون_داد->ne[2] == 1 && درون_داد->ne[3] == 1);
        اگر (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            درون_داد_shape[2] = -1;
        }
        اگر (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            أكد(درون_داد_shape.size() == 4 && درون_داد_shape[0] == 1 && درون_داد_shape[1] == 1 &&
                   درون_داد_shape[2].is_dynamic() && درون_داد_shape[3].is_static());
            ثابت صحیح n_heads_kv = get_n_heads_kv_for_tensor(درون_داد);
            أكد(n_heads_kv > 0 && درون_داد_shape[3].get_length() % n_heads_kv == 0);
            ثابت int64_t combined_dim = درون_داد_shape[3].get_length();  // n_heads_kv * head_size
            ثابت int64_t head_size = combined_dim / n_heads_kv;
            درون_داد_shape = {درون_داد_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } وگرنه اگر (is_kv_idx(درون_داد, op)) {
        // kv update index
        صحیح len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        درون_داد_shape = ov::PartialShape{1, 1, 1, len};

    } وگرنه اگر (is_inp_s_copy(درون_داد, op) || is_s_copy_leaf(درون_داد)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        درون_داد_shape = m_is_static ? ov::PartialShape{get_shape(درون_داد)} : ov::PartialShape{1, 1, 1, -1};

    } وگرنه {
        درون_داد_shape = ov::PartialShape{get_shape(درون_داد)};
    }
    اگر (dynamic_dim_index != -1 && m_model_is_splitted) {
        درون_داد_shape[3 - dynamic_dim_index] = -1;
    }
    اگر (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_درون_داد && op->src[1] == درون_داد) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        اگر (m_is_static) {
            ثابت int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            درون_داد_shape[2] = seq_active;
            درون_داد_shape[3] = seq_active;
        } وگرنه {
            درون_داد_shape[2] = -1;
            درون_داد_shape[3] = -1;
        }
    }
    بازگشت درون_داد_shape;
}

منطقی GgmlOvDecoder::is_s_copy_leaf(ثابت ggml_tensor * tensor) ثابت {
    اگر (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        بازگشت نادرست;
    }
    برای (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت ggml_tensor * گره = m_cgraph->nodes[i];
        اگر (گره->op != GGML_OP_GET_ROWS || گره->src[0] == nullptr || گره->src[1] == nullptr) {
            ادامه;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ثابت ggml_tensor * idx = گره->src[1];
        وقتی (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        اگر (idx != tensor) {
            ادامه;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ثابت ggml_tensor * data = گره->src[0];
        وقتی (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        اگر (data != nullptr && is_recurrent_cache(data)) {
            بازگشت درست;
        }
    }
    بازگشت نادرست;
}

تهی GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_درون_داد = [این](ثابت std::string & name, int64_t value, منطقی force_parameter = نادرست) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    اگر (m_compute_params.attention_size != -1) {
        create_1d_درون_داد("attention_size", m_compute_params.attention_size);
    }
    اگر (m_compute_params.attention_size_static != -1) {
        create_1d_درون_داد("attention_size_static", m_compute_params.attention_size_static);
    }
    اگر (m_compute_params.attention_size_swa != -1) {
        create_1d_درون_داد("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    اگر (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_درون_داد("swa_window", m_compute_params.swa_window);
    }
    create_1d_درون_داد("n_seq_active", m_compute_params.n_seq_active);
    create_1d_درون_داد("seq_active_start", m_compute_params.seq_active_start);
    create_1d_درون_داد("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    اگر (m_compute_params.token_len_per_seq != -1) {
        create_1d_درون_داد("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    اگر (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_درون_داد("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/درست);
        create_1d_درون_داد("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/درست);
    }

    اگر (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_درون_داد("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        اگر (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_درون_داد("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/درست);
        }
    }

    برای (ثابت auto & [گره_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_درون_داد("rs_slot_begin_" + گره_name, writeback.slot_begin);
        اگر (!m_is_static && writeback.src_begin >= 0) {
            create_1d_درون_داد("rs_src_begin_" + گره_name, writeback.src_begin);
        }
    }
}

منطقی GgmlOvDecoder::گره_is_used_as_src(ثابت صحیح گره_idx) {
    ggml_tensor * گره = m_cgraph->nodes[گره_idx];
    برای (صحیح i = گره_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_گره = m_cgraph->nodes[i];
        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            اگر (other_گره->src[j] == گره) {
                بازگشت درست;
            }
        }
    }
    بازگشت نادرست;
}

تهی GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    برای (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * گره = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        اگر (گره->op == GGML_OP_NONE && گره_is_used_as_src(i)) {
            std::string گره_name = get_tensor_ov_name(m_cgraph, گره);
            اگر (m_model_weights.find(گره_name) == m_model_weights.end()) {
                m_inputs[گره_name] = گره;
                m_model_inputs[گره_name] = {get_ov_type(گره),
                                             get_graph_درون_داد_shape(گره, nullptr, m_گره_dynamic_dims[گره])};
            }
            ادامه;
        }
        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = گره->src[i];
            اگر (src == nullptr) {
                ادامه;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            اگر (src->flags & GGML_TENSOR_FLAG_درون_داد) {
                src_name = get_tensor_graph_درون_داد_ov_name(این, m_cgraph, src, گره);
            }
            اگر (m_model_weights.find(src_name) != m_model_weights.end()) {
                ادامه;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            اگر (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                ادامه;
            }

            منطقی is_intermediate_گره = نادرست;
            برای (ثابت auto & گره_info : m_گره_info_list) {
                اگر (گره_info.گره == src) {
                    is_intermediate_گره = درست;
                    بشکن;
                }
            }
            اگر (is_intermediate_گره) {
                ادامه;
            }
            اگر (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                ادامه;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            اگر (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                اگر (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            وقتی (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_درون_داد_shape(گره, src, m_گره_dynamic_dims[src])};
        }
    }
}

تهی GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    برای (صحیح گره_n = 0; گره_n < m_cgraph->n_nodes; گره_n++) {
        auto * cur_گره = m_cgraph->nodes[گره_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        اگر (cur_گره->op == GGML_OP_NONE || cur_گره->op == GGML_OP_VIEW || cur_گره->op == GGML_OP_RESHAPE) {
            ادامه;
        }
        اگر (::is_inplace_op(cur_گره) && ggml_nbytes(cur_گره) == 0) {
            ادامه;
        }
        auto cur_گره_use_شمار = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_گره)];
        اگر (cur_گره_use_شمار == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            اگر (cur_گره != nullptr && ::is_inplace_op(cur_گره) && ggml_nbytes(cur_گره) > 0) {
                cur_گره = cur_گره->view_src;
            }
        } وگرنه {
            صحیح درون_داد_use_شمار = 0;
            برای (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * گره = m_cgraph->nodes[i];
                برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                    اگر (گره->src[j] != پوچ && گره->src[j] == cur_گره) {
                        درون_داد_use_شمار++;
                    }
                }
            }
            اگر (درون_داد_use_شمار == cur_گره_use_شمار) {
                cur_گره = nullptr;
            }
        }
        اگر (cur_گره != nullptr) {
            std::string cur_گره_name = get_tensor_ov_name(m_cgraph, cur_گره);
            m_model_outputs[cur_گره_name] = cur_گره;
            m_model_output_names.insert(cur_گره_name);
        }
    }
}

ثابت ggml_tensor * GgmlOvDecoder::get_tensor_used_op(ثابت ggml_tensor * tensor) ثابت {
    اگر (tensor == nullptr) {
        بازگشت nullptr;
    }
    برای (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت auto * گره = m_cgraph->nodes[i];
        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            اگر (گره->src[j] == tensor) {
                بازگشت گره;
            }
        }
    }
    بازگشت nullptr;
}

ثابت ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ثابت std::string & name) ثابت {
    برای (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت auto * گره = m_cgraph->nodes[i];
        برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ثابت auto * src = گره->src[j];
            اگر (src == nullptr) {
                بشکن;
            }
            اگر (get_tensor_ov_name(m_cgraph, src) == name) {
                بازگشت src;
            }
        }
    }
    بازگشت nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() ثابت {
    std::map<std::string, std::string> kv_param_res_names;
    برای (ثابت auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    بازگشت kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
ایستا منطقی is_mul_mat_id_expert_weight(ثابت ggml_tensor * گره, صحیح src_index) {
    بازگشت گره->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, منطقی naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    برای (صحیح گره_i = 0; گره_i < n_nodes; گره_i++) {
        auto * گره = nodes[گره_i];
        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = گره->src[i];
            اگر (src == nullptr) {
                ادامه;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            اگر (is_rope_freqs_weight(base, گره)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            اگر (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(گره, i)) {
                اگر (model_weights.find(src_name) == model_weights.end()) {
                    auto weight_گره = create_weight_گره(base, naive);
                    weight_گره->set_friendly_name(src_name);
                    model_weights[src_name] = weight_گره;
                }
            }
        }
    }
    بازگشت model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
ایستا std::mutex g_nonov_weight_cache_mutex;
ایستا std::unordered_map<ثابت تهی *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    برای (صحیح گره_i = 0; گره_i < cgraph->n_nodes; گره_i++) {
        auto * گره = cgraph->nodes[گره_i];
        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = گره->src[i];
            اگر (src == nullptr) {
                ادامه;
            }
            ثابت ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            اگر (is_rope_freqs_weight(base, گره)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            اگر (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    بازگشت names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_گره(ggml_tensor * tensor, منطقی naive) {
    ثابت منطقی is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    اگر (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        اگر (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            اگر (weight_extra->weight_گره) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                بازگشت weight_extra->weight_گره;
            }
        } وگرنه اگر (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            اگر (quant_extra->weight_گره) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                بازگشت quant_extra->weight_گره;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    اگر (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        auto weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        auto weight_گره = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_گره->set_friendly_name(tensor->name);
        بازگشت weight_گره;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ثابت منطقی cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    اگر (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        auto it = g_nonov_weight_cache.find(tensor->data);
        اگر (it != g_nonov_weight_cache.end()) {
            بازگشت it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    ایستا ثابت std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    اگر (weight_types.find(tensor->type) == weight_types.end()) {
        پرتابکن std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    اگر (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        اگر (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } وگرنه {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } وگرنه {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_گره->set_friendly_name(tensor->name);
    اگر (!is_ov_buffer) {
        اگر (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            auto [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_گره);
            بازگشت it->second;
        }
        بازگشت ov_weight.weight_گره;
    }

    ggml_openvino_extra_base * extra;
    اگر (ov_weight.is_quantized()) {
        extra = جدید ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_گره);
    } وگرنه {
        extra = جدید ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_گره);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    بازگشت ov_weight.weight_گره;
}

تهی GgmlOvDecoder::dump_cgraph(ثابت ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    اگر (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        بازگشت;
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
    برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گره = cgraph->nodes[i];

        // Get buffer type name
        ثابت نویسه * buf_name = "none";
        ggml_backend_buffer_t buf = گره->view_src ? گره->view_src->buffer : گره->buffer;
        اگر (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << گره->ne[0] << ", "
             << std::setw(5) << گره->ne[1] << ", "
             << std::setw(5) << گره->ne[2] << ", "
             << std::setw(5) << گره->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(گره->op) << std::right << " "
             << std::left << std::setw(45) << گره->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << گره->nb[0] << ", "
             << std::setw(5) << گره->nb[1] << ", "
             << std::setw(5) << گره->nb[2] << ", "
             << std::setw(5) << گره->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            اگر (auto* src = گره->src[i]) {
                // Get buffer type name for source
                ثابت نویسه * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                اگر (src_buf) {
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
    برای (صحیح i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * گره = cgraph->leafs[i];

        // Get buffer type name for leaf
        ثابت نویسه * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = گره->view_src ? گره->view_src->buffer : گره->buffer;
        اگر (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << گره->ne[0] << ", "
             << std::setw(5) << گره->ne[1] << "] "
             << std::setw(8) << ggml_op_name(گره->op) << " "
             << std::setw(16) << ggml_get_name(گره)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

تهی print_tensor_address_map(ثابت ggml_cgraph * cgraph) {
    std::map<تهی *, std::vector<std::string>> address_map;
    برای (صحیح گره_n = 0; گره_n < cgraph->n_nodes; گره_n++) {
        auto * گره = cgraph->nodes[گره_n];
        اگر (گره->data) {
            auto it = address_map.find(گره->data);
            اگر (it == address_map.end()) {
                address_map[گره->data] = std::vector<std::string>();
            }
            address_map[گره->data].push_back(گره->name);
        }
    }
    برای (ثابت auto & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        برای (ثابت auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ثابت ggml_tensor * tensor) {
    std::vector<size_t> shape;
    برای (صحیح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    بازگشت shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(ثابت ggml_tensor * tensor) {
    std::vector<size_t> stride;
    برای (صحیح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    بازگشت stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(ثابت ggml_tensor * tensor) {
    انتخاب (tensor->type) {
    حالت GGML_TYPE_F64:
        بازگشت ov::element::f64;
    حالت GGML_TYPE_F32:
        بازگشت ov::element::f32;
    حالت GGML_TYPE_F16:
        بازگشت ov::element::f16;
    حالت GGML_TYPE_BF16:
        بازگشت ov::element::bf16;
    حالت GGML_TYPE_I8:
        بازگشت ov::element::i8;
    حالت GGML_TYPE_I16:
        بازگشت ov::element::i16;
    حالت GGML_TYPE_I32:
        بازگشت ov::element::i32;
    حالت GGML_TYPE_I64:
        بازگشت ov::element::i64;
    پیشفرض:
        بازگشت ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_درون_داد_shape(صحیح گره_idx, ثابت std::string & name) ثابت {
    بازگشت ov::PartialShape(get_shape(m_گره_info_list[گره_idx].گره_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_درون_داد_stride(صحیح گره_idx, ثابت std::string & name) ثابت {
    بازگشت get_stride(m_گره_info_list[گره_idx].گره_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_درون_داد_size(صحیح گره_idx, ثابت std::string & name) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        بازگشت it->second.size();
    }
    بازگشت 0;
}

size_t GgmlOvDecoder::get_view_درون_داد_offset(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        اگر (view_index < it->second.size()) {
            بازگشت it->second[view_index].second->view_offs;
        }
    }
    بازگشت 0;
}

size_t GgmlOvDecoder::get_view_درون_داد_src_offset(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        اگر (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            اگر (view_tensor && view_tensor->src[0]) {
                بازگشت view_tensor->src[0]->view_offs;
            }
        }
    }
    بازگشت 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_درون_داد_stride(صحیح گره_idx,
                                                         ثابت std::string & name,
                                                         size_t view_index) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        اگر (view_index < it->second.size()) {
            بازگشت get_stride(it->second[view_index].second);
        }
    }
    بازگشت {};
}

std::vector<size_t> GgmlOvDecoder::get_view_درون_داد_src_stride(صحیح گره_idx,
                                                             ثابت std::string & name,
                                                             size_t view_index) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        اگر (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            اگر (view_tensor && view_tensor->src[0]) {
                بازگشت get_stride(view_tensor->src[0]);
            }
        }
    }
    بازگشت {};
}

ov::Shape GgmlOvDecoder::get_view_درون_داد_ggml_shape(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        اگر (view_index < it->second.size()) {
            بازگشت get_shape(it->second[view_index].second);
        }
    }
    بازگشت {};
}

ov::Shape GgmlOvDecoder::get_view_درون_داد_src_ggml_shape(صحیح گره_idx,
                                                       ثابت std::string & name,
                                                       size_t view_index) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        اگر (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            اگر (view_tensor && view_tensor->src[0]) {
                بازگشت get_shape(view_tensor->src[0]);
            }
        }
    }
    بازگشت {};
}

ov::PartialShape GgmlOvDecoder::get_view_درون_داد_ov_shape(صحیح گره_idx,
                                                        ثابت std::string & name,
                                                        size_t view_index) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        اگر (view_index < it->second.size()) {
            auto * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            auto dynamic_it = m_گره_dynamic_dims.find(tensor);
            اگر (dynamic_it != m_گره_dynamic_dims.end() && dynamic_it->second != -1) {
                صحیح dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            بازگشت shape;
        }
    }
    بازگشت {};
}

ov::PartialShape GgmlOvDecoder::get_view_درون_داد_src_ov_shape(صحیح گره_idx,
                                                            ثابت std::string & name,
                                                            size_t view_index) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        اگر (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            اگر (view_tensor && view_tensor->src[0]) {
                auto * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                auto dynamic_it = m_گره_dynamic_dims.find(src_tensor);
                اگر (dynamic_it != m_گره_dynamic_dims.end() && dynamic_it->second != -1) {
                    صحیح dynamic_dim_index = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
                }

                بازگشت shape;
            }
        }
    }
    بازگشت {};
}

std::string GgmlOvDecoder::get_view_درون_داد_name(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        اگر (view_index < it->second.size()) {
            بازگشت it->second[view_index].first;
        }
    }
    بازگشت "";
}

std::string GgmlOvDecoder::get_view_درون_داد_src_name(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت {
    auto it = m_گره_info_list[گره_idx].گره_inputs_views.find(name);
    اگر (it != m_گره_info_list[گره_idx].گره_inputs_views.end()) {
        اگر (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            اگر (view_tensor && view_tensor->src[0]) {
                بازگشت get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    بازگشت "";
}

ov::element::Type GgmlOvDecoder::get_درون_داد_type(صحیح گره_idx, ثابت std::string & name) ثابت {
    بازگشت get_ov_type(m_گره_info_list[گره_idx].گره_inputs.at(name));
}

size_t GgmlOvDecoder::get_درون_داد_size() ثابت {
    بازگشت m_model_inputs.size();
}

size_t GgmlOvDecoder::get_درون_داد_size(صحیح گره_idx) ثابت {
    بازگشت m_گره_info_list[گره_idx].گره_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_درون_داد_names(صحیح گره_idx) ثابت {
    بازگشت m_گره_info_list[گره_idx].گره_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(صحیح گره_idx) ثابت {
    auto * ggml_tensor = m_گره_info_list[گره_idx].گره;
    بازگشت ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ثابت صحیح گره_idx) ثابت {
    بازگشت get_ov_type(m_گره_info_list[گره_idx].گره);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(صحیح گره_idx) ثابت {
    auto * ggml_tensor = m_گره_info_list[گره_idx].گره;
    بازگشت get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(صحیح گره_idx) ثابت {
    auto * گره = m_گره_info_list[گره_idx].گره;
    اگر (گره->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        برای (صحیح i = گره_idx + 1; i < m_cgraph->n_nodes; i++) {
            auto * consumer = m_cgraph->nodes[i];
            اگر (consumer->op != GGML_OP_VIEW || consumer->src[0] != گره) {
                ادامه;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            auto & name = consumer->view_offs == 0 ? attn_name : state_name;
            اگر (!name.empty()) {
                بازگشت {m_گره_info_list[گره_idx].گره_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        اگر (!attn_name.empty() && !state_name.empty()) {
            بازگشت {attn_name, state_name};
        }
    }
    بازگشت {m_گره_info_list[گره_idx].گره_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(صحیح گره_idx) ثابت {
    auto * گره = m_گره_info_list[گره_idx].گره;
    اگر (!::is_inplace_op(گره) || گره->view_src == nullptr || ggml_nbytes(گره) == 0) {
        بازگشت "";
    }
    ثابت صحیح op_case = m_گره_info_list[گره_idx].گره_op_case;
    اگر (گره->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        بازگشت "";
    }
    بازگشت get_tensor_ov_name(m_cgraph, گره->view_src);
}

منطقی GgmlOvDecoder::is_view_like_alias_of(صحیح گره_idx, ثابت std::string & view_src_name) ثابت {
    auto * گره = m_گره_info_list[گره_idx].گره;
    اگر (گره->view_src == nullptr || get_tensor_ov_name(m_cgraph, گره->view_src) != view_src_name) {
        بازگشت نادرست;
    }
    بازگشت گره->op == GGML_OP_RESHAPE || گره->op == GGML_OP_VIEW;
}

ثابت std::string & GgmlOvDecoder::get_op_name() ثابت {
    ایستا ثابت std::string unknown_name = "UNKNOWN_OP_NAME";
    بازگشت unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(صحیح گره_idx) ثابت {
    auto it = m_گره_dynamic_dims.find(m_گره_info_list[گره_idx].گره);
    اگر (it == m_گره_dynamic_dims.end()) {
        بازگشت -1;
    }
    بازگشت it->second;
}

ثابت std::string & GgmlOvDecoder::get_op_name(صحیح گره_idx) ثابت {
    بازگشت m_گره_info_list[گره_idx].گره_name;
}

int32_t * GgmlOvDecoder::get_درون_داد_op_params(صحیح گره_idx, ثابت std::string & name) ثابت {
    بازگشت m_گره_info_list[گره_idx].گره_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(صحیح گره_idx) ثابت {
    بازگشت m_گره_info_list[گره_idx].گره->op_params;
}

size_t GgmlOvDecoder::get_output_op_offset(صحیح گره_idx) ثابت {
    بازگشت m_گره_info_list[گره_idx].گره->view_offs;
}

تهی GgmlOvDecoder::visit_subgraph(std::function<تهی(std::shared_ptr<GgmlDecoder>, صحیح گره_idx)> گره_visitor) ثابت {
    برای (صحیح گره_idx = 0; گره_idx < m_cgraph->n_nodes; گره_idx++) {
        اگر (m_cgraph->nodes[گره_idx]->op == GGML_OP_NONE) {
            ادامه;
        }
        گره_visitor(std::make_shared<GgmlOvDecoder>(*این), گره_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ثابت ggml_tensor * گره) {
    انتخاب (گره->op) {
    حالت GGML_OP_UNARY:
        بازگشت std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(گره));
    حالت GGML_OP_GLU:
        بازگشت std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(گره));
    پیشفرض:
        بازگشت std::string("GGML_OP_") + ggml_op_name(گره->op);
    }
}

ثابت std::string & GgmlOvDecoder::get_op_type(صحیح گره_idx) ثابت {
    بازگشت m_گره_info_list[گره_idx].گره_op_type;
}

ثابت std::string & GgmlOvDecoder::get_op_type() ثابت {
    ایستا ثابت std::string unknown_op = "UNKNOWN_GGML_OP";
    بازگشت unknown_op;
}

تهی GgmlOvDecoder::compute_گره_dynamic_dims() {
    auto visit_گره = [&](auto && self, ggml_tensor * گره) -> تهی {
        اگر (!گره) {
            بازگشت;
        }

        اگر (گره->op == GGML_OP_CPY) {
            m_گره_dynamic_dims[گره] = -1;
        }

        اگر (m_گره_dynamic_dims.count(گره)) {
            بازگشت;
        }
        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = گره->src[i];
            اگر (src == nullptr) {
                ادامه;
            }
            ساختار ggml_tensor * ریشه_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            اگر (ریشه_src) {
                اگر (is_inp_tok(ریشه_src, گره) || is_inp_pos(ریشه_src, گره) || is_output_idx(ریشه_src, گره)) {
                    m_گره_dynamic_dims[ریشه_src] = 0;
                    m_گره_dynamic_dims[src] = m_گره_dynamic_dims[ریشه_src];
                    ادامه;
                }
                self(self, ریشه_src);
                m_گره_dynamic_dims[src] = m_گره_dynamic_dims[ریشه_src];
            } وگرنه {
                اگر (is_inp_tok(src, گره) || is_inp_pos(src, گره) || is_output_idx(src, گره)) {
                    m_گره_dynamic_dims[src] = 0;
                    ادامه;
                }
                اگر (is_inp_scale_rows(src, گره)) {
                    m_گره_dynamic_dims[src] = 1;
                    ادامه;
                }
                اگر (گره->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_گره_dynamic_dims[src] = 1;
                    ادامه;
                }
                self(self, src);
            }
        }
        انتخاب (گره->op) {
        حالت GGML_OP_NONE:
            m_گره_dynamic_dims[گره] = -1;
            بشکن;
        حالت GGML_OP_GET_ROWS:
            m_گره_dynamic_dims[گره] = -1;
            اگر (m_گره_dynamic_dims[گره->src[1]] != -1) {
                auto dynamic_dim_idx = m_گره_dynamic_dims[گره->src[1]];
                اگر (dynamic_dim_idx == 0) {
                    m_گره_dynamic_dims[گره] = 1;
                } وگرنه {
                    auto dynamic_dim_stride = گره->src[1]->nb[dynamic_dim_idx] / ggml_type_size(گره->src[1]->type) *
                                              ggml_type_size(گره->src[0]->type);
                    برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                        اگر (dynamic_dim_stride == گره->src[0]->nb[i]) {
                            m_گره_dynamic_dims[گره] = i;
                            بشکن;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            بشکن;
        حالت GGML_OP_MUL:
        حالت GGML_OP_MUL_MAT:
            m_گره_dynamic_dims[گره] = -1;
            اگر (m_گره_dynamic_dims[گره->src[0]] != -1) {
                m_گره_dynamic_dims[گره] = m_گره_dynamic_dims[گره->src[0]];
            }
            اگر (m_گره_dynamic_dims[گره->src[1]] != -1) {
                m_گره_dynamic_dims[گره] = m_گره_dynamic_dims[گره->src[1]];
            }
            بشکن;
        حالت GGML_OP_PERMUTE:
            m_گره_dynamic_dims[گره] = -1;
            اگر (m_گره_dynamic_dims[گره->src[0]] != -1) {
                auto dynamic_dim_idx = m_گره_dynamic_dims[گره->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                    اگر (گره->op_params[i] == dynamic_dim_idx) {
                        m_گره_dynamic_dims[گره] = i;
                        بشکن;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            بشکن;
        حالت GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_گره_dynamic_dims[گره] = -1;
            اگر (m_گره_dynamic_dims[گره->src[0]] != -1) {
                اگر (گره->src[0]->op == GGML_OP_NONE) {
                    m_گره_dynamic_dims[گره] = m_گره_dynamic_dims[گره->src[0]];
                    بشکن;
                }
                auto dynamic_dim_idx = m_گره_dynamic_dims[گره->src[0]];
                auto dynamic_dim_مقدار = گره->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    گره->src[0]->nb[dynamic_dim_idx] / ggml_type_size(گره->src[0]->type) * ggml_type_size(گره->type);
                برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                    اگر (گره->nb[i] == dynamic_dim_stride) {
                        m_گره_dynamic_dims[گره] = i;
                        بشکن;
                    }
                }
                اگر (m_گره_dynamic_dims[گره] != -1 && dynamic_dim_مقدار != گره->ne[m_گره_dynamic_dims[گره]]) {
                    m_گره_dynamic_dims[گره] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    اگر (ggml_nelements(گره) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      گره->name, گره->src[0]->name);
                    }
                }
            }
            بشکن;
        }
        حالت GGML_OP_TRANSPOSE:
        حالت GGML_OP_RESHAPE: {
            اگر (is_same_shape(گره->src[0], گره)) {
                m_گره_dynamic_dims[گره] = m_گره_dynamic_dims[گره->src[0]];
                بشکن;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_گره_dynamic_dims[گره] = -1;
            اگر (m_گره_dynamic_dims[گره->src[0]] != -1) {
                auto dynamic_dim_idx = m_گره_dynamic_dims[گره->src[0]];
                auto dynamic_dim_stride = گره->src[0]->nb[dynamic_dim_idx];
                برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                    اگر (گره->nb[i] == dynamic_dim_stride && گره->ne[i] == گره->src[0]->ne[dynamic_dim_idx]) {
                        m_گره_dynamic_dims[گره] = i;
                        بشکن;
                    }
                }
                اگر (m_گره_dynamic_dims[گره] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", گره->name);
                }
            }
            بشکن;
        }
        حالت GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr صحیح q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_گره_dynamic_dims[گره] = -1;
            اگر (m_گره_dynamic_dims[گره->src[0]] != -1) {
                auto q_dynamic_dim = m_گره_dynamic_dims[گره->src[0]];
                m_گره_dynamic_dims[گره] = q_to_out[q_dynamic_dim];
            }
            بشکن;
        }
        حالت GGML_OP_CONT:
            m_گره_dynamic_dims[گره] = -1;
            اگر (m_گره_dynamic_dims[گره->src[0]] != -1) {
                auto dynamic_dim_idx = m_گره_dynamic_dims[گره->src[0]];
                اگر (ggml_are_same_shape(گره, گره->src[0])) {
                    m_گره_dynamic_dims[گره] = dynamic_dim_idx;
                } وگرنه {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(گره->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (گره->src[0]->ne[0] / ggml_blck_size(گره->src[0]->type));
                    برای (صحیح i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * گره->src[0]->ne[i - 1];
                    }

                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(گره->src[0]->type) *
                                              ggml_type_size(گره->type);
                    صحیح matched_dim_شمار = 0;
                    صحیح first_matched_dim = -1;
                    برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                        اگر (گره->nb[i] == dynamic_dim_stride && گره->ne[i] == گره->src[0]->ne[dynamic_dim_idx]) {
                            اگر (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_گره_dynamic_dims[گره] = i;
                            matched_dim_شمار++;
                        }
                    }
                    اگر (matched_dim_شمار > 1 && گره->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_گره_dynamic_dims[گره] = first_matched_dim;
                    } وگرنه اگر (matched_dim_شمار != 1) {
                        m_گره_dynamic_dims[گره] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      گره->name, گره->src[0]->name);
                    }
                }
            }
            بشکن;
        حالت GGML_OP_CONCAT:
            برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
                اگر (گره->src[0]->ne[i] != گره->ne[i]) {
                    m_گره_dynamic_dims[گره] = i;
                    بشکن;
                }
            }
            بشکن;
        حالت GGML_OP_SSM_CONV:
        حالت GGML_OP_GATED_DELTA_NET:
            m_گره_dynamic_dims[گره] = 1;
            بشکن;
        حالت GGML_OP_RMS_NORM:
        حالت GGML_OP_L2_NORM:
        حالت GGML_OP_NORM:
        حالت GGML_OP_ADD:
        حالت GGML_OP_SUB:
        حالت GGML_OP_GLU:
        حالت GGML_OP_ROPE:
        حالت GGML_OP_SCALE:
        حالت GGML_OP_SOFT_MAX:
        حالت GGML_OP_ARGSORT:
        حالت GGML_OP_ADD_ID:
        حالت GGML_OP_UNARY:
        حالت GGML_OP_CUMSUM:
        حالت GGML_OP_FILL:
        حالت GGML_OP_SET:
        حالت GGML_OP_DIAG:
        حالت GGML_OP_TRI:
        حالت GGML_OP_REPEAT:
        حالت GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        حالت GGML_OP_DIV:
        حالت GGML_OP_CLAMP:
        حالت GGML_OP_PAD:
        حالت GGML_OP_UPSCALE:
        حالت GGML_OP_SIN:
        حالت GGML_OP_COS:
        حالت GGML_OP_LOG:
        حالت GGML_OP_ROLL:
            m_گره_dynamic_dims[گره] = m_گره_dynamic_dims[گره->src[0]];
            بشکن;
        حالت GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_گره_dynamic_dims[گره] =
                (m_گره_dynamic_dims[گره->src[0]] == 0) ? -1 : m_گره_dynamic_dims[گره->src[0]];
            بشکن;
        حالت GGML_OP_MUL_MAT_ID:
        حالت GGML_OP_SOLVE_TRI:
            m_گره_dynamic_dims[گره] = m_گره_dynamic_dims[گره->src[1]];
            بشکن;
        حالت GGML_OP_CPY:
        حالت GGML_OP_SET_ROWS:
        حالت GGML_OP_SUM:
        حالت GGML_OP_MEAN:
            m_گره_dynamic_dims[گره] = -1;
            بشکن;
        حالت GGML_OP_IM2COL: {
            m_گره_dynamic_dims[گره] = -1;
            اگر (m_گره_dynamic_dims[گره->src[1]] != -1) {
                ثابت منطقی is_2D = گره->op_params[6] == 1;
                ثابت صحیح src_dyn = m_گره_dynamic_dims[گره->src[1]];
                اگر (is_2D) {
                    اگر (src_dyn == 0) {
                        m_گره_dynamic_dims[گره] = 1;  // IW -> OW
                    } وگرنه اگر (src_dyn == 1) {
                        m_گره_dynamic_dims[گره] = 2;  // IH -> OH
                    } وگرنه اگر (src_dyn == 3) {
                        m_گره_dynamic_dims[گره] = 3;  // N  -> N
                    }
                } وگرنه {
                    اگر (src_dyn == 0) {
                        m_گره_dynamic_dims[گره] = 1;  // IW -> OW
                    } وگرنه اگر (src_dyn == 2) {
                        m_گره_dynamic_dims[گره] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                اگر (m_گره_dynamic_dims[گره] != -1) {
                    OPENVINO_ASSERT(گره->src[1]->ne[src_dyn] == گره->ne[m_گره_dynamic_dims[گره]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(گره->name) +
                                        " and its src[1]: " + std::string(گره->src[1]->name));
                }
            }
            بشکن;
        }
        حالت GGML_OP_IM2COL_3D: {
            m_گره_dynamic_dims[گره] = -1;
            اگر (m_گره_dynamic_dims[گره->src[1]] != -1) {
                ثابت صحیح src_dyn = m_گره_dynamic_dims[گره->src[1]];
                اگر (src_dyn == 0) {
                    m_گره_dynamic_dims[گره] = 1;  // IW -> OW
                } وگرنه اگر (src_dyn == 1) {
                    m_گره_dynamic_dims[گره] = 2;  // IH -> OH
                } وگرنه اگر (src_dyn == 3) {
                    m_گره_dynamic_dims[گره] = 3;  // N  -> N
                }
                اگر (m_گره_dynamic_dims[گره] != -1) {
                    OPENVINO_ASSERT(گره->src[1]->ne[src_dyn] == گره->ne[m_گره_dynamic_dims[گره]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(گره->name) +
                                        " and its src[1]: " + std::string(گره->src[1]->name));
                }
            }
            بشکن;
        }
        پیشفرض:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(گره->op), گره->name);
            بشکن;
        }
    };

    برای (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * گره = m_cgraph->nodes[i];
        visit_گره(visit_گره, گره);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    اگر (0) {
        برای (صحیح i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * گره = m_cgraph->nodes[i];
            صحیح dynamic_dim = m_گره_dynamic_dims[گره];
            std::cout << "[" << i << "] " << "node_name: " << گره->name << " op: " << ggml_op_name(گره->op)
                      << " shape: [";
            برای (صحیح j = 0; j < 4; j++) {
                اگر (j == dynamic_dim) {
                    std::cout << "*";
                } وگرنه {
                    std::cout << گره->ne[j];
                }
                اگر (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            برای (صحیح j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = گره->src[j];
                اگر (src == nullptr) {
                    ادامه;
                }
                صحیح src_dynamic_dim = m_گره_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                برای (صحیح k = 0; k < 4; k++) {
                    اگر (k == src_dynamic_dim) {
                        std::cout << "*";
                    } وگرنه {
                        std::cout << src->ne[k];
                    }
                    اگر (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
