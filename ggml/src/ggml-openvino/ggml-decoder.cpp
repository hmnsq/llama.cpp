#ضمن "ggml-decoder.h"

#ضمن "ggml-impl.h"
#ضمن "ggml-openvino-extra.h"
#ضمن "ggml-openvino.h"
#ضمن "ggml-quants.h"
#ضمن "ggml.h"
#ضمن "utils.h"

#ضمن <algorithm>
#ضمن <cassert>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <cstdlib>
#ضمن <fstream>
#ضمن <iomanip>
#ضمن <map>
#ضمن <memory>
#ضمن <mutex>
#ضمن <openvino/core/dimension.hpp>
#ضمن <openvino/core/except.hpp>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/partial_shape.hpp>
#ضمن <openvino/core/type/bfloat16.hpp>
#ضمن <openvino/core/type/element_type.hpp>
#ضمن <openvino/core/type/float16.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/runtime/tensor.hpp>
#ضمن <ostream>
#ضمن <set>
#ضمن <stdexcept>
#ضمن <string>
#ضمن <cstring>
#ضمن <unordered_map>
#ضمن <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             منطقي is_static,
                             منطقي is_stateful,
                             منطقي model_is_splitted,
                             منطقي is_prefill,
                             صحيح prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(خطأ),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    ساكن منطقي printed_address_map = خطأ;
    إذا (!printed_address_map) {
        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = true;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_إدخال_output();
    compute_عقدة_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    لكل (صحيح عقدة_n = 0; عقدة_n < cgraph->n_nodes; عقدة_n++) {
        m_عقدة_info_list[عقدة_n].عقدة_op_case = compute_op_case(m_عقدة_info_list[عقدة_n].عقدة);
        m_عقدة_info_list[عقدة_n].عقدة_op_type = compute_op_type(m_عقدة_info_list[عقدة_n].عقدة);
    }

    add_extra_inputs();
}

فراغ GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_عقدة_info_list.clear();
    set_إدخال_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
منطقي GgmlOvDecoder::is_bound_to(ثابت ggml_cgraph * cgraph) ثابت {
    إرجاع m_cgraph == cgraph && cgraph->n_nodes > 0 && m_عقدة_info_list.size() == (size_t) cgraph->n_nodes &&
           m_عقدة_info_list.front().عقدة == cgraph->nodes[0] &&
           m_عقدة_info_list.back().عقدة == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = true;
    set_إدخال_output();
    compute_model_inputs();
    compute_model_outputs();
    لكل (صحيح عقدة_n = 0; عقدة_n < cgraph->n_nodes; عقدة_n++) {
        m_عقدة_info_list[عقدة_n].عقدة_op_case = compute_op_case(m_عقدة_info_list[عقدة_n].عقدة);
        m_عقدة_info_list[عقدة_n].عقدة_op_type = compute_op_type(m_عقدة_info_list[عقدة_n].عقدة);
    }
}

نطاق {
منطقي is_inplace_op(ثابت ggml_tensor * عقدة) {
    إرجاع عقدة->op == GGML_OP_SET_ROWS || عقدة->op == GGML_OP_CPY || (عقدة->op == GGML_OP_SCALE && عقدة->view_src);
}

منطقي is_same_shape(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        إذا (a->ne[i] != b->ne[i]) {
            إرجاع خطأ;
        }
    }
    إرجاع true;
}

منطقي is_conv_states_all_tensor(ثابت ggml_tensor * tensor) {
    إرجاع tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

منطقي is_full_single_slot_writeback(ثابت ggml_tensor * عقدة) {
    إرجاع عقدة->view_src != nullptr && عقدة->view_src->ne[1] == 1 && عقدة->src[1] != nullptr &&
           عقدة->src[1]->op == GGML_OP_VIEW && عقدة->src[1]->view_src == عقدة->view_src &&
           عقدة->src[1]->view_offs == 0 && ggml_nbytes(عقدة->src[1]) == ggml_nbytes(عقدة->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
منطقي is_moe_expert_sum_add(ثابت ggml_tensor * عقدة) {
    std::vector<ثابت ggml_tensor *> leaves;
    ثابت ggml_tensor * cur = عقدة;
    بينما (cur->op == GGML_OP_ADD) {
        إذا (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            إرجاع خطأ;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ثابت ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    لكل (ثابت ggml_tensor * leaf : leaves) {
        إذا (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            إرجاع خطأ;
        }
        ثابت ggml_tensor * leaf_base = leaf->src[0];
        إذا (base == nullptr) {
            base = leaf_base;
        } وإلا إذا (leaf_base != base) {
            إرجاع خطأ;
        }
        إذا (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            إرجاع خطأ;
        }
        إذا (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            إرجاع خطأ;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        إذا (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            إرجاع خطأ;
        }
    }

    إرجاع base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor) {
    إذا (tensor == nullptr) {
        إرجاع "";
    }
    إذا ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ثابت auto * عقدة = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        إذا (عقدة != cgraph->nodes + cgraph->n_nodes) {
            إرجاع std::string(tensor->name) + "#n" + std::to_string(عقدة - cgraph->nodes);
        }
        ثابت auto * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        إذا (leaf != cgraph->leafs + cgraph->n_leafs) {
            إرجاع std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    إرجاع tensor->name;
}

ساكن std::string get_tensor_ov_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor) {
    إرجاع GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

ساكن std::string get_tensor_graph_إدخال_ov_name(ثابت GgmlOvDecoder * decoder,
                                                  ثابت ggml_cgraph * cgraph,
                                                  ثابت ggml_tensor * tensor,
                                                  ثابت ggml_tensor * op) {
    إذا (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        إرجاع "inp_pos";
    }
    إذا (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        إرجاع "embd";
    }
    إذا (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ثابت منطقي is_swa = decoder->is_swa_mask(tensor);
        إذا (decoder->is_stateful()) {
            إرجاع is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        إذا (is_swa) {
            إرجاع get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    إرجاع get_tensor_ov_name(cgraph, tensor);
}

فراغ GgmlOvDecoder::set_إدخال_output() {
    لكل (صحيح عقدة_n = 0; عقدة_n < m_cgraph->n_nodes; عقدة_n++) {
        auto * عقدة = m_cgraph->nodes[عقدة_n];

        عقدة_Info current_عقدة_info;
        auto عقدة_name = get_tensor_ov_name(m_cgraph, عقدة);

        current_عقدة_info.عقدة = عقدة;
        current_عقدة_info.عقدة_name = عقدة_name;
        current_عقدة_info.عقدة_op_case = 0;
        current_عقدة_info.data_addr = عقدة->data;

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = عقدة->src[i];
            إذا (src == nullptr) {
                استمر;
            }
            auto src_name = get_tensor_ov_name(m_cgraph, src);
            إذا (src->flags & GGML_TENSOR_FLAG_إدخال) {
                src_name = get_tensor_graph_إدخال_ov_name(هذا, m_cgraph, src, عقدة);
            }
            current_عقدة_info.عقدة_inputs[src_name] = src;
            current_عقدة_info.عقدة_inputs_names.push_back(src_name);

            إذا (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_عقدة_info.عقدة_inputs_views[src_name])> view_chain;
                auto * current = src;

                بينما (current != nullptr) {
                    auto current_name = get_tensor_ov_name(m_cgraph, current);
                    إذا (current->flags & GGML_TENSOR_FLAG_إدخال) {
                        current_name = get_tensor_graph_إدخال_ov_name(هذا, m_cgraph, current, عقدة);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    إذا (current->src[0] != nullptr && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } وإلا {
                        اكسر;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_عقدة_info.عقدة_inputs_views[src_name] = view_chain;
            }
        }

        m_عقدة_info_list.push_back(current_عقدة_info);
    }
}

صحيح GgmlOvDecoder::compute_op_case(ثابت ggml_tensor * عقدة) ثابت {
    صحيح op_case = 0;
    بدل (عقدة->op) {
    حالة GGML_OP_RESHAPE: {
        إذا (m_naive) {
            اكسر;
        }
        auto name = std::string(عقدة->name);
        auto * src = عقدة->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        منطقي recurrent_sequence = خطأ;
        لكل (صحيح i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ثابت auto * consumer = m_cgraph->nodes[i];
            إذا (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == عقدة) {
                إرجاع 1;
            } وإلا إذا (consumer->op == GGML_OP_SSM_CONV) {
                ثابت auto * concat = consumer->src[0];
                إذا (concat->op == GGML_OP_CONCAT) {
                    ثابت auto * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == عقدة;
                }
            } وإلا إذا (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                ثابت auto * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == عقدة;
            }
        }
        إذا (recurrent_sequence && عقدة->ne[0] == src->ne[0] && عقدة->ne[3] == 1) {
            إرجاع 6;
        }
        إذا (عقدة->ne[0] == src->ne[0] && عقدة->ne[2] == 1 && عقدة->ne[3] == 1) {
            إرجاع 5;
        }
        إذا (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == عقدة->ne[0] && src->src[0]->ne[1] == عقدة->ne[1]) {
            op_case = 4;
        } وإلا إذا (عقدة->ne[0] * عقدة->ne[1] == src->ne[0]) {
            op_case = 1;
        } وإلا إذا (src->ne[0] * src->ne[1] == عقدة->ne[0]) {
            op_case = 2;
            إذا (src->ne[2] * src->ne[3] == عقدة->ne[1]) {
                op_case = 5;
            }
        } وإلا إذا (عقدة->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == عقدة->ne[1]) {
            op_case = 3;
        } وإلا إذا (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } وإلا إذا (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } وإلا إذا (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        إذا (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            ثابت auto * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            منطقي recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
            لكل (صحيح i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ثابت auto * consumer = m_cgraph->nodes[i];
                إذا (consumer->op == GGML_OP_GATED_DELTA_NET) {
                    لكل (صحيح j : {3, 4}) {
                        ثابت auto * gate = consumer->src[j];
                        إذا (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == عقدة;
                    }
                } وإلا إذا (consumer->op == GGML_OP_MUL) {
                    لكل (صحيح j = 0; j < 2; ++j) {
                        ثابت auto * gate = consumer->src[j];
                        ثابت auto * norm = consumer->src[1 - j];
                        إذا (gate->op != GGML_OP_UNARY || gate->src[0] != عقدة) {
                            استمر;
                        }
                        إذا (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            إذا (recurrent) {
                op_case = 9;
            }
        }
        اكسر;
    }
    حالة GGML_OP_PERMUTE: {
        إذا (عقدة->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } وإلا إذا (عقدة->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(عقدة->view_src->name);
            صحيح layer = extract_layer_from_name(src_name).value();
            إذا (ggml_is_contiguous(عقدة->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                إذا (!is_swa_layer(layer)) {
                    op_case = 3;
                } وإلا {
                    op_case = 4;
                }
            } وإلا {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                إذا (!is_swa_layer(layer)) {
                    op_case = 5;
                } وإلا {
                    op_case = 6;
                }
            }
        } وإلا {
            // rope'ed query tensor
            op_case = 2;
        }
        اكسر;
    }
    حالة GGML_OP_MUL_MAT: {
        إذا (عقدة->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        اكسر;
    }
    حالة GGML_OP_GET_ROWS: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        إذا (عقدة->src[0]->op == GGML_OP_RESHAPE && عقدة->src[0]->src[0] != nullptr &&
            is_recurrent_cache(عقدة->src[0]->src[0])) {
            ثابت منطقي single_slot = عقدة->src[0]->src[0]->ne[1] == 1;
            إذا (عقدة->src[1]->op == GGML_OP_VIEW) {
                op_case = (عقدة->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
            } وإلا إذا (single_slot) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        اكسر;
    }
    حالة GGML_OP_ROPE: {
        ثابت صحيح mode = عقدة->op_params[2];
        بدل (mode) {
        حالة GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            اكسر;
        }
        حالة GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            اكسر;
        }
        حالة GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            اكسر;
        }
        حالة GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            اكسر;
        }
        افتراضي:
            op_case = 0;
            اكسر;
        }
        اكسر;
    }
    حالة GGML_OP_VIEW: {
        إذا (!m_model_params.has_rs_rollback && عقدة->src[0] != nullptr &&
            عقدة->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            اكسر;
        }
        إذا (m_is_static && عقدة->src[0] != nullptr &&
            (عقدة->src[0]->op == GGML_OP_GATED_DELTA_NET || عقدة->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            اكسر;
        }
        إذا (عقدة->src[0]->op == GGML_OP_VIEW) {
            auto * src = عقدة->src[0];
            إذا (ggml_nelements(عقدة) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            إذا (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            auto * src = عقدة->src[0];
            إذا (ggml_nelements(عقدة) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                إذا (عقدة->ne[0] == src->ne[0] && عقدة->ne[1] == src->ne[2] && عقدة->ne[2] == 1 &&
                    عقدة->nb[0] == src->nb[0] && عقدة->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    اكسر;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                صحيح diff_تعداد = 0;
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (عقدة->ne[i] != src->ne[i]) {
                        diff_تعداد++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    إذا (عقدة->ne[i] > src->ne[i]) {
                        إرجاع 0;
                    }
                }
                إذا (diff_تعداد >= 1) {
                    op_case = 0;
                }
            }
        }
        اكسر;
    }
    حالة GGML_OP_RMS_NORM: {
        إذا (عقدة->src[0]->op == GGML_OP_VIEW) {
            إذا (is_same_shape(عقدة->src[0]->src[0], عقدة->src[0])) {
                op_case = 1;
            } وإلا إذا (!m_model_params.has_rs_rollback &&
                       عقدة->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } وإلا إذا (عقدة->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        اكسر;
    }
    حالة GGML_OP_POOL_2D: {
        ثابت ggml_op_pool pool_mode = static_cast<ggml_op_pool>(عقدة->op_params[0]);
        بدل (pool_mode) {
        حالة GGML_OP_POOL_MAX: {
            op_case = 1;
            اكسر;
        }
        حالة GGML_OP_POOL_AVG: {
            op_case = 2;
            اكسر;
        }
        افتراضي:
            op_case = 0;
            اكسر;
        }
        اكسر;
    }
    حالة GGML_OP_UPSCALE: {
        ثابت int32_t mode_flags = عقدة->op_params[0];
        ثابت ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        بدل (scale_mode) {
        حالة GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            اكسر;
        }
        حالة GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            اكسر;
        }
        حالة GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            اكسر;
        }
        افتراضي:
            op_case = 0;
            اكسر;
        }
        اكسر;
    }
    حالة GGML_OP_CPY: {
        إذا (عقدة->src[0]->op == GGML_OP_VIEW) {
            إذا (عقدة->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                إذا (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(عقدة) ? 7 : 10;
                } وإلا {
                    op_case = 1;
                }
            } وإلا إذا (GgmlOvDecoder::is_conv_state_writeback(عقدة)) {
                op_case = is_full_single_slot_writeback(عقدة) ? 8 : 2;
                اكسر;
            } وإلا إذا (is_conv_states_all_tensor(عقدة->view_src) && عقدة->src[1] != nullptr &&
                       عقدة->src[1]->op == GGML_OP_VIEW && عقدة->src[1]->view_src == عقدة->view_src) {
                op_case = 4;
                اكسر;
            } وإلا إذا (عقدة->src[0]->src[0]->op == GGML_OP_GET_ROWS && عقدة->src[1] != nullptr &&
                       عقدة->src[1]->op == GGML_OP_VIEW && عقدة->src[1]->view_src != nullptr &&
                       is_recurrent_cache(عقدة->src[1]->view_src) && عقدة->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } وإلا إذا (عقدة->src[0]->op == GGML_OP_GET_ROWS && عقدة->src[1] != nullptr &&
                   عقدة->src[1]->op == GGML_OP_VIEW && عقدة->src[1]->view_src != nullptr &&
                   is_recurrent_cache(عقدة->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = عقدة->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } وإلا إذا (عقدة->src[1] != nullptr && عقدة->src[1]->op == GGML_OP_VIEW && عقدة->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ثابت ggml_tensor * kv_buf = عقدة->src[1]->view_src;
            إذا (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
                    ثابت ggml_tensor * n = m_cgraph->nodes[i];
                    إذا (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        استمر;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    إذا ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        إذا (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        اكسر;
                    }
                }
            }
        }
        اكسر;
    }
    حالة GGML_OP_ADD: {
        إذا (is_moe_expert_sum_add(عقدة)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        اكسر;
    }
    حالة GGML_OP_SCALE: {
        إذا (عقدة->view_src && عقدة->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = عقدة->view_src->ne[1] == 1 ? 2 : 1;
        }
        اكسر;
    }
    حالة GGML_OP_L2_NORM: {
        إذا (std::string(عقدة->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        اكسر;
    }
    حالة GGML_OP_FLASH_ATTN_EXT: {
        إذا (عقدة->src[1] != nullptr && عقدة->src[1]->op == GGML_OP_VIEW && عقدة->src[1]->view_src != nullptr) {
            ثابت ggml_tensor * kv_buf = عقدة->src[1]->view_src;
            إذا (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (عقدة->src[3] != nullptr) ? 1 : 2;
            }
        }
        اكسر;
    }
    افتراضي:
        اكسر;
    }
    إرجاع op_case;
}

std::optional<صحيح> extract_layer_from_name(ثابت std::string & name) {
    size_t pos1 = name.find("_l");
    إذا (pos1 == std::string::npos) {
        إرجاع std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    إذا (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    صحيح layer = std::stoi(layer_str);
    إرجاع layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
ساكن صحيح get_swa_window_from_mask(ثابت ggml_tensor * mask) {
    إذا (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        إرجاع -1;
    }
    إذا (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        إرجاع -1;
    }

    ثابت int64_t n_kv = mask->ne[0];
    ثابت int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    لكل (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        لكل (int64_t c = 0; c < n_kv; c++) {
            ثابت size_t i = (size_t) r * n_kv + c;
            ثابت عائم v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((ثابت ggml_fp16_t *) mask->data)[i]) :
                                                          ((ثابت عائم *) mask->data)[i];
            إذا (v > -INFINITY) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    إرجاع window > 0 ? (صحيح) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, منطقي is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    auto get_attention_pattern_case = [](ثابت ggml_tensor * عقدة) -> صحيح {
        إذا (عقدة == nullptr) {
            إرجاع -1;
        }

        بدل (عقدة->op) {
        حالة GGML_OP_FLASH_ATTN_EXT:
            إذا (عقدة->src[0] == nullptr || عقدة->src[1] == nullptr) {
                إرجاع -1;
            }
            بدل (عقدة->src[1]->op) {
            حالة GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                إذا (عقدة->src[3] != nullptr && عقدة->src[1]->src[0] != nullptr &&
                    عقدة->src[1]->src[0]->op == GGML_OP_VIEW) {
                    إرجاع 0;
                }
                اكسر;
            حالة GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                إذا (عقدة->src[3] != nullptr && عقدة->src[1]->src[0] != nullptr &&
                    عقدة->src[1]->src[0]->op == GGML_OP_PERMUTE && عقدة->src[1]->src[0]->src[0] != nullptr &&
                    عقدة->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    إرجاع 1;
                }
                اكسر;
            حالة GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                إذا (عقدة->src[1]->view_src != nullptr) {
                    إذا (عقدة->src[3] != nullptr) {
                        إرجاع 4;  // decoder self-attention
                    }
                    إرجاع 5;      // cross-attention or encoder self-attention
                }
                اكسر;
            افتراضي:
                اكسر;
            }
            اكسر;
        حالة GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            إذا (عقدة->src[0] != nullptr && عقدة->src[1] != nullptr && عقدة->src[0]->op == GGML_OP_MUL_MAT &&
                عقدة->src[0]->src[0] != nullptr && عقدة->src[0]->src[1] != nullptr &&
                عقدة->src[0]->src[0]->op == GGML_OP_PERMUTE && عقدة->src[0]->src[0]->src[0] != nullptr &&
                عقدة->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                إرجاع 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            إذا (عقدة->src[0]->op == GGML_OP_ADD && عقدة->src[0]->src[0] != nullptr &&
                عقدة->src[0]->src[0]->op == GGML_OP_MUL_MAT && عقدة->src[0]->src[0]->src[0] != nullptr &&
                عقدة->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                إرجاع 3;
            }
            اكسر;
        افتراضي:
            اكسر;
        }

        إرجاع -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    auto get_attention_op_mask = [&get_attention_pattern_case](ثابت ggml_tensor * عقدة) -> ثابت ggml_tensor * {
        بدل (get_attention_pattern_case(عقدة)) {
        حالة 0:
        حالة 1:
            إرجاع عقدة->src[3];
        حالة 2:
        حالة 3:
            إرجاع عقدة->src[1];
        افتراضي:
            إرجاع nullptr;
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
        std::map<صحيح, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<صحيح, ثابت ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ثابت ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            إذا (mask == nullptr) {
                استمر;
            }
            ثابت ggml_tensor * cache_k_permute = nullptr;
            بدل (get_attention_pattern_case(cgraph->nodes[i])) {
            حالة 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     اكسر;
            حالة 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             اكسر;
            حالة 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             اكسر;
            افتراضي: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     اكسر;
            }
            ثابت ggml_tensor * cache_k_view = cache_k_permute->src[0];
            إذا (cache_k_view->op != GGML_OP_VIEW) {
                استمر;
            }
            ثابت ggml_tensor * leaf = cache_k_view->src[0];
            auto layer = extract_layer_from_name(leaf->name);
            إذا (!layer.has_value()) {
                استمر;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        لكل (ثابت auto & [layer, extent] : layer_extent) {
            إذا (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                إذا (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            لكل (ثابت auto & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (long) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    منطقي rope_seen = خطأ;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * عقدة = cgraph->nodes[i];
        ثابت صحيح attention_pattern_case = get_attention_pattern_case(عقدة);
        إذا (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            بدل (attention_pattern_case) {
            حالة 0:
                cache_k_permute = عقدة->src[1];
                mask = عقدة->src[3];
                اكسر;
            حالة 1:
                cache_k_permute = عقدة->src[1]->src[0];
                mask = عقدة->src[3];
                اكسر;
            حالة 2:
                cache_k_permute = عقدة->src[0]->src[0];
                mask = عقدة->src[1];
                اكسر;
            حالة 3:
                cache_k_permute = عقدة->src[0]->src[0]->src[0];
                mask = عقدة->src[1];
                اكسر;
            حالة 4:
            حالة 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                auto * cache_k_view = عقدة->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = عقدة->src[0]->ne[1];
                إذا (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } وإلا {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                استمر;
            }
            افتراضي:
                اكسر;
            }

            أكد(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.إدخال_len = عقدة->src[0]->ne[1];
            compute_params.token_len_per_seq = عقدة->src[0]->ne[1];

            auto * cache_k_view = cache_k_permute->src[0];
            إذا (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                استمر;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            صحيح layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ثابت منطقي layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            إذا (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } وإلا {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, حجم(size_t));
            compute_params.seq_active_start = offset / seq_size;

            إذا (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
            } وإلا {
                compute_params.attention_size = mask->ne[0];
            }
            إذا (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        إذا (عقدة->op == GGML_OP_MUL_MAT && عقدة->src[0]->op == GGML_OP_PERMUTE &&
            عقدة->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(عقدة->src[0]->view_src, عقدة->view_src)) {
            إذا (عقدة->src[1]->op == GGML_OP_PERMUTE && عقدة->src[1]->src[0]->op == GGML_OP_VIEW &&
                عقدة->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = عقدة->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        إذا (عقدة->op == GGML_OP_TRANSPOSE && عقدة->src[0]->op == GGML_OP_PERMUTE &&
            عقدة->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = عقدة->ne[0];
            إذا (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        إذا (عقدة->op == GGML_OP_ROPE) {
            إذا (compute_params.token_len_per_seq == -1 && عقدة->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(عقدة->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(حجم(model_params.rope_params) == حجم(int32_t) * 16, "rope_params size");
            إذا (!rope_seen) {
                memcpy(model_params.rope_params, عقدة->op_params, حجم(int32_t) * 16);
                rope_seen = true;
            } وإلا إذا (memcmp(model_params.rope_params, عقدة->op_params, حجم(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = true;
            }
        }
        إذا (عقدة->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = عقدة->src[0]->ne[0];
        }
        إذا (عقدة->op == GGML_OP_SCALE && عقدة->view_src != nullptr && is_recurrent_cache(عقدة->view_src)) {
            إذا (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = عقدة->view_src->ne[1];
            } وإلا {
                GGML_ASSERT(model_params.n_rs_slots == عقدة->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(عقدة) / عقدة->view_src->ne[0];
            compute_params.cache_rs_reset_idx = عقدة->src[0]->view_offs / عقدة->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        إذا (عقدة->op == GGML_OP_CPY && عقدة->view_src != nullptr && is_recurrent_cache(عقدة->view_src) &&
            عقدة->src[1] != nullptr && عقدة->src[1]->op == GGML_OP_VIEW && عقدة->src[1]->view_src == عقدة->view_src) {
            ثابت منطقي is_conv = is_conv_state_writeback(عقدة);
            ثابت منطقي is_gdn = عقدة->src[0]->op == GGML_OP_VIEW && عقدة->src[0]->src[0] != nullptr &&
                                عقدة->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ثابت منطقي is_extra = عقدة->src[0]->op == GGML_OP_GET_ROWS;
            ثابت منطقي is_gdn_rollback = is_gdn && is_same_shape(عقدة->src[0], عقدة->src[1]);

            ثابت ggml_tensor * dest_view = عقدة->src[1];
            ثابت ggml_tensor * cache = عقدة->view_src;
            ثابت size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            إذا (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = true;
            }
            إذا (row_bytes > 0 && (is_conv || is_gdn || is_extra) && !is_full_single_slot_writeback(عقدة)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (صحيح) (dest_view->view_offs / row_bytes);
                إذا (is_conv) {
                    writeback.src_begin = (صحيح) (عقدة->src[0]->view_offs / عقدة->src[0]->view_src->nb[0]);
                } وإلا إذا (is_gdn_rollback) {
                    writeback.src_begin = (صحيح) (عقدة->src[0]->view_offs / عقدة->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, عقدة)] = writeback;
            }
            إذا ((is_conv || is_gdn) && !is_full_single_slot_writeback(عقدة)) {
                compute_params.s_copy_active_slot_len = (صحيح) dest_view->ne[1];
            }
        }
    }
    إذا (model_params.n_heads_kv == -1) {
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ثابت auto * عقدة = cgraph->nodes[i];
            ثابت ggml_tensor * mask = nullptr;
            إذا (عقدة->op == GGML_OP_SOFT_MAX) {
                mask = عقدة->src[1];
            } وإلا إذا (عقدة->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = عقدة->src[3];
            } وإلا {
                استمر;
            }
            إذا (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_إدخال) ||
                عقدة->src[0] == nullptr) {
                استمر;
            }
            model_params.is_cacheless_attn = true;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.إدخال_len = عقدة->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.إدخال_len;
            اكسر;
        }
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    إذا (model_params.is_cacheless_attn) {
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ثابت auto * عقدة = cgraph->nodes[i];
            إذا (عقدة->op == GGML_OP_GET_ROWS && is_output_idx(عقدة->src[1], عقدة)) {
                compute_params.output_len = عقدة->src[1]->ne[0];
                اكسر;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    إذا (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    إرجاع {model_params, compute_params};
}

فراغ GgmlOvDecoder::validate_cgraph() ثابت {
    إذا (m_model_params.n_seq > 1 && m_is_static == true) {
        ارم std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_إدخال_shape(ثابت ggml_tensor * op,
                                                      ثابت ggml_tensor * إدخال,
                                                      صحيح dynamic_dim_index) ثابت {
    إذا (m_naive) {
        إرجاع إدخال != nullptr ? ov::PartialShape{get_shape(إدخال)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape إدخال_shape;

    إذا (is_inp_tok(إدخال, op) || is_inp_pos(إدخال, op)) {
        // tokens or positions
        صحيح len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        إذا (m_is_static && is_inp_pos(إدخال, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        إدخال_shape = ov::PartialShape{1, 1, 1, len};

    } وإلا إذا (is_output_idx(إدخال, op)) {
        // output index
        إدخال_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } وإلا إذا (is_inp_mean(إدخال, op)) {
        إدخال_shape = m_is_static ? ov::PartialShape{1, 1, إدخال->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, -1, -1};

    } وإلا إذا (is_inp_scale_rows(إدخال, op)) {
        إدخال_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } وإلا إذا (is_inp_mask(إدخال, op)) {
        // mask
        إذا (m_is_static) {
            إدخال_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } وإلا إذا (m_is_stateful) {
            إدخال_shape = ov::PartialShape{1, 1, -1, -1};
        } وإلا {
            إدخال_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } وإلا إذا (is_recurrent_cache(إدخال)) {
        إدخال_shape = ov::PartialShape{get_shape(إدخال)};
        إذا (!m_is_static && !m_is_stateful && إدخال->ne[1] > 1) {
            إدخال_shape[2] = -1;
        }

    } وإلا إذا (is_kvcache(إدخال, op)) {
        // kvcache
        إدخال_shape = ov::PartialShape{get_shape(إدخال)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ثابت منطقي is_flat_kv = (إدخال->ne[1] == 1 && إدخال->ne[2] == 1 && إدخال->ne[3] == 1);
        إذا (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            إدخال_shape[2] = -1;
        }
        إذا (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            أكد(إدخال_shape.size() == 4 && إدخال_shape[0] == 1 && إدخال_shape[1] == 1 &&
                   إدخال_shape[2].is_dynamic() && إدخال_shape[3].is_static());
            ثابت صحيح n_heads_kv = get_n_heads_kv_for_tensor(إدخال);
            أكد(n_heads_kv > 0 && إدخال_shape[3].get_length() % n_heads_kv == 0);
            ثابت int64_t combined_dim = إدخال_shape[3].get_length();  // n_heads_kv * head_size
            ثابت int64_t head_size = combined_dim / n_heads_kv;
            إدخال_shape = {إدخال_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } وإلا إذا (is_kv_idx(إدخال, op)) {
        // kv update index
        صحيح len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        إدخال_shape = ov::PartialShape{1, 1, 1, len};

    } وإلا إذا (is_inp_s_copy(إدخال, op) || is_s_copy_leaf(إدخال)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        إدخال_shape = m_is_static ? ov::PartialShape{get_shape(إدخال)} : ov::PartialShape{1, 1, 1, -1};

    } وإلا {
        إدخال_shape = ov::PartialShape{get_shape(إدخال)};
    }
    إذا (dynamic_dim_index != -1 && m_model_is_splitted) {
        إدخال_shape[3 - dynamic_dim_index] = -1;
    }
    إذا (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_إدخال && op->src[1] == إدخال) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        إذا (m_is_static) {
            ثابت int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            إدخال_shape[2] = seq_active;
            إدخال_shape[3] = seq_active;
        } وإلا {
            إدخال_shape[2] = -1;
            إدخال_shape[3] = -1;
        }
    }
    إرجاع إدخال_shape;
}

منطقي GgmlOvDecoder::is_s_copy_leaf(ثابت ggml_tensor * tensor) ثابت {
    إذا (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        إرجاع خطأ;
    }
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت ggml_tensor * عقدة = m_cgraph->nodes[i];
        إذا (عقدة->op != GGML_OP_GET_ROWS || عقدة->src[0] == nullptr || عقدة->src[1] == nullptr) {
            استمر;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ثابت ggml_tensor * idx = عقدة->src[1];
        بينما (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        إذا (idx != tensor) {
            استمر;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ثابت ggml_tensor * data = عقدة->src[0];
        بينما (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        إذا (data != nullptr && is_recurrent_cache(data)) {
            إرجاع true;
        }
    }
    إرجاع خطأ;
}

فراغ GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_إدخال = [هذا](ثابت std::string & name, int64_t value, منطقي force_parameter = خطأ) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    إذا (m_compute_params.attention_size != -1) {
        create_1d_إدخال("attention_size", m_compute_params.attention_size);
    }
    إذا (m_compute_params.attention_size_static != -1) {
        create_1d_إدخال("attention_size_static", m_compute_params.attention_size_static);
    }
    إذا (m_compute_params.attention_size_swa != -1) {
        create_1d_إدخال("attention_size_swa", m_compute_params.attention_size_swa);
    }
    // only the stateful SWA mask consumes this
    إذا (is_stateful() && m_compute_params.swa_window != -1) {
        create_1d_إدخال("swa_window", m_compute_params.swa_window);
    }
    create_1d_إدخال("n_seq_active", m_compute_params.n_seq_active);
    create_1d_إدخال("seq_active_start", m_compute_params.seq_active_start);
    create_1d_إدخال("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    إذا (m_compute_params.token_len_per_seq != -1) {
        create_1d_إدخال("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    إذا (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_إدخال("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/true);
        create_1d_إدخال("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/true);
    }

    إذا (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_إدخال("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        إذا (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_إدخال("chunk_valid_len", get_static_n_tokens(), /*force_parameter=*/true);
        }
    }

    لكل (ثابت auto & [عقدة_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_إدخال("rs_slot_begin_" + عقدة_name, writeback.slot_begin);
        إذا (!m_is_static && writeback.src_begin >= 0) {
            create_1d_إدخال("rs_src_begin_" + عقدة_name, writeback.src_begin);
        }
    }
}

منطقي GgmlOvDecoder::عقدة_is_used_as_src(ثابت صحيح عقدة_idx) {
    ggml_tensor * عقدة = m_cgraph->nodes[عقدة_idx];
    لكل (صحيح i = عقدة_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_عقدة = m_cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (other_عقدة->src[j] == عقدة) {
                إرجاع true;
            }
        }
    }
    إرجاع خطأ;
}

فراغ GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * عقدة = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        إذا (عقدة->op == GGML_OP_NONE && عقدة_is_used_as_src(i)) {
            std::string عقدة_name = get_tensor_ov_name(m_cgraph, عقدة);
            إذا (m_model_weights.find(عقدة_name) == m_model_weights.end()) {
                m_inputs[عقدة_name] = عقدة;
                m_model_inputs[عقدة_name] = {get_ov_type(عقدة),
                                             get_graph_إدخال_shape(عقدة, nullptr, m_عقدة_dynamic_dims[عقدة])};
            }
            استمر;
        }
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = عقدة->src[i];
            إذا (src == nullptr) {
                استمر;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            إذا (src->flags & GGML_TENSOR_FLAG_إدخال) {
                src_name = get_tensor_graph_إدخال_ov_name(هذا, m_cgraph, src, عقدة);
            }
            إذا (m_model_weights.find(src_name) != m_model_weights.end()) {
                استمر;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            إذا (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                استمر;
            }

            منطقي is_intermediate_عقدة = خطأ;
            لكل (ثابت auto & عقدة_info : m_عقدة_info_list) {
                إذا (عقدة_info.عقدة == src) {
                    is_intermediate_عقدة = true;
                    اكسر;
                }
            }
            إذا (is_intermediate_عقدة) {
                استمر;
            }
            إذا (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                استمر;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            إذا (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                إذا (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            بينما (src->op == GGML_OP_VIEW && src->src[0] != nullptr) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_إدخال_shape(عقدة, src, m_عقدة_dynamic_dims[src])};
        }
    }
}

فراغ GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    لكل (صحيح عقدة_n = 0; عقدة_n < m_cgraph->n_nodes; عقدة_n++) {
        auto * cur_عقدة = m_cgraph->nodes[عقدة_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        إذا (cur_عقدة->op == GGML_OP_NONE || cur_عقدة->op == GGML_OP_VIEW || cur_عقدة->op == GGML_OP_RESHAPE) {
            استمر;
        }
        إذا (::is_inplace_op(cur_عقدة) && ggml_nbytes(cur_عقدة) == 0) {
            استمر;
        }
        auto cur_عقدة_use_تعداد = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_عقدة)];
        إذا (cur_عقدة_use_تعداد == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            إذا (cur_عقدة != nullptr && ::is_inplace_op(cur_عقدة) && ggml_nbytes(cur_عقدة) > 0) {
                cur_عقدة = cur_عقدة->view_src;
            }
        } وإلا {
            صحيح إدخال_use_تعداد = 0;
            لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * عقدة = m_cgraph->nodes[i];
                لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                    إذا (عقدة->src[j] != عدم && عقدة->src[j] == cur_عقدة) {
                        إدخال_use_تعداد++;
                    }
                }
            }
            إذا (إدخال_use_تعداد == cur_عقدة_use_تعداد) {
                cur_عقدة = nullptr;
            }
        }
        إذا (cur_عقدة != nullptr) {
            std::string cur_عقدة_name = get_tensor_ov_name(m_cgraph, cur_عقدة);
            m_model_outputs[cur_عقدة_name] = cur_عقدة;
            m_model_output_names.insert(cur_عقدة_name);
        }
    }
}

ثابت ggml_tensor * GgmlOvDecoder::get_tensor_used_op(ثابت ggml_tensor * tensor) ثابت {
    إذا (tensor == nullptr) {
        إرجاع nullptr;
    }
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت auto * عقدة = m_cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (عقدة->src[j] == tensor) {
                إرجاع عقدة;
            }
        }
    }
    إرجاع nullptr;
}

ثابت ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ثابت std::string & name) ثابت {
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت auto * عقدة = m_cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            ثابت auto * src = عقدة->src[j];
            إذا (src == nullptr) {
                اكسر;
            }
            إذا (get_tensor_ov_name(m_cgraph, src) == name) {
                إرجاع src;
            }
        }
    }
    إرجاع nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() ثابت {
    std::map<std::string, std::string> kv_param_res_names;
    لكل (ثابت auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    إرجاع kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
ساكن منطقي is_mul_mat_id_expert_weight(ثابت ggml_tensor * عقدة, صحيح src_index) {
    إرجاع عقدة->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, منطقي naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    لكل (صحيح عقدة_i = 0; عقدة_i < n_nodes; عقدة_i++) {
        auto * عقدة = nodes[عقدة_i];
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = عقدة->src[i];
            إذا (src == nullptr) {
                استمر;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            إذا (is_rope_freqs_weight(base, عقدة)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            إذا (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(عقدة, i)) {
                إذا (model_weights.find(src_name) == model_weights.end()) {
                    auto weight_عقدة = create_weight_عقدة(base, naive);
                    weight_عقدة->set_friendly_name(src_name);
                    model_weights[src_name] = weight_عقدة;
                }
            }
        }
    }
    إرجاع model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
ساكن std::mutex g_nonov_weight_cache_mutex;
ساكن std::unordered_map<ثابت فراغ *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    لكل (صحيح عقدة_i = 0; عقدة_i < cgraph->n_nodes; عقدة_i++) {
        auto * عقدة = cgraph->nodes[عقدة_i];
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = عقدة->src[i];
            إذا (src == nullptr) {
                استمر;
            }
            ثابت ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            إذا (is_rope_freqs_weight(base, عقدة)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            إذا (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    إرجاع names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_عقدة(ggml_tensor * tensor, منطقي naive) {
    ثابت منطقي is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    إذا (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        إذا (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            إذا (weight_extra->weight_عقدة) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                إرجاع weight_extra->weight_عقدة;
            }
        } وإلا إذا (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            إذا (quant_extra->weight_عقدة) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                إرجاع quant_extra->weight_عقدة;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    إذا (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        auto weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        auto weight_عقدة = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_عقدة->set_friendly_name(tensor->name);
        إرجاع weight_عقدة;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ثابت منطقي cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    إذا (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        auto it = g_nonov_weight_cache.find(tensor->data);
        إذا (it != g_nonov_weight_cache.end()) {
            إرجاع it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    ساكن ثابت std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    إذا (weight_types.find(tensor->type) == weight_types.end()) {
        ارم std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    إذا (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        إذا (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } وإلا {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } وإلا {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_عقدة->set_friendly_name(tensor->name);
    إذا (!is_ov_buffer) {
        إذا (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            auto [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_عقدة);
            إرجاع it->second;
        }
        إرجاع ov_weight.weight_عقدة;
    }

    ggml_openvino_extra_base * extra;
    إذا (ov_weight.is_quantized()) {
        extra = جديد ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_عقدة);
    } وإلا {
        extra = جديد ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_عقدة);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    إرجاع ov_weight.weight_عقدة;
}

فراغ GgmlOvDecoder::dump_cgraph(ثابت ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    إذا (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        إرجاع;
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
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * عقدة = cgraph->nodes[i];

        // Get buffer type name
        ثابت حرف * buf_name = "none";
        ggml_backend_buffer_t buf = عقدة->view_src ? عقدة->view_src->buffer : عقدة->buffer;
        إذا (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << عقدة->ne[0] << ", "
             << std::setw(5) << عقدة->ne[1] << ", "
             << std::setw(5) << عقدة->ne[2] << ", "
             << std::setw(5) << عقدة->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(عقدة->op) << std::right << " "
             << std::left << std::setw(45) << عقدة->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << عقدة->nb[0] << ", "
             << std::setw(5) << عقدة->nb[1] << ", "
             << std::setw(5) << عقدة->nb[2] << ", "
             << std::setw(5) << عقدة->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            إذا (auto* src = عقدة->src[i]) {
                // Get buffer type name for source
                ثابت حرف * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                إذا (src_buf) {
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
    لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * عقدة = cgraph->leafs[i];

        // Get buffer type name for leaf
        ثابت حرف * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = عقدة->view_src ? عقدة->view_src->buffer : عقدة->buffer;
        إذا (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << عقدة->ne[0] << ", "
             << std::setw(5) << عقدة->ne[1] << "] "
             << std::setw(8) << ggml_op_name(عقدة->op) << " "
             << std::setw(16) << ggml_get_name(عقدة)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

فراغ print_tensor_address_map(ثابت ggml_cgraph * cgraph) {
    std::map<فراغ *, std::vector<std::string>> address_map;
    لكل (صحيح عقدة_n = 0; عقدة_n < cgraph->n_nodes; عقدة_n++) {
        auto * عقدة = cgraph->nodes[عقدة_n];
        إذا (عقدة->data) {
            auto it = address_map.find(عقدة->data);
            إذا (it == address_map.end()) {
                address_map[عقدة->data] = std::vector<std::string>();
            }
            address_map[عقدة->data].push_back(عقدة->name);
        }
    }
    لكل (ثابت auto & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        لكل (ثابت auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ثابت ggml_tensor * tensor) {
    std::vector<size_t> shape;
    لكل (صحيح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    إرجاع shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(ثابت ggml_tensor * tensor) {
    std::vector<size_t> stride;
    لكل (صحيح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    إرجاع stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(ثابت ggml_tensor * tensor) {
    بدل (tensor->type) {
    حالة GGML_TYPE_F64:
        إرجاع ov::element::f64;
    حالة GGML_TYPE_F32:
        إرجاع ov::element::f32;
    حالة GGML_TYPE_F16:
        إرجاع ov::element::f16;
    حالة GGML_TYPE_BF16:
        إرجاع ov::element::bf16;
    حالة GGML_TYPE_I8:
        إرجاع ov::element::i8;
    حالة GGML_TYPE_I16:
        إرجاع ov::element::i16;
    حالة GGML_TYPE_I32:
        إرجاع ov::element::i32;
    حالة GGML_TYPE_I64:
        إرجاع ov::element::i64;
    افتراضي:
        إرجاع ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_إدخال_shape(صحيح عقدة_idx, ثابت std::string & name) ثابت {
    إرجاع ov::PartialShape(get_shape(m_عقدة_info_list[عقدة_idx].عقدة_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_إدخال_stride(صحيح عقدة_idx, ثابت std::string & name) ثابت {
    إرجاع get_stride(m_عقدة_info_list[عقدة_idx].عقدة_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_إدخال_size(صحيح عقدة_idx, ثابت std::string & name) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إرجاع it->second.size();
    }
    إرجاع 0;
}

size_t GgmlOvDecoder::get_view_إدخال_offset(صحيح عقدة_idx, ثابت std::string & name, size_t view_index) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إذا (view_index < it->second.size()) {
            إرجاع it->second[view_index].second->view_offs;
        }
    }
    إرجاع 0;
}

size_t GgmlOvDecoder::get_view_إدخال_src_offset(صحيح عقدة_idx, ثابت std::string & name, size_t view_index) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إذا (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            إذا (view_tensor && view_tensor->src[0]) {
                إرجاع view_tensor->src[0]->view_offs;
            }
        }
    }
    إرجاع 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_إدخال_stride(صحيح عقدة_idx,
                                                         ثابت std::string & name,
                                                         size_t view_index) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إذا (view_index < it->second.size()) {
            إرجاع get_stride(it->second[view_index].second);
        }
    }
    إرجاع {};
}

std::vector<size_t> GgmlOvDecoder::get_view_إدخال_src_stride(صحيح عقدة_idx,
                                                             ثابت std::string & name,
                                                             size_t view_index) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إذا (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            إذا (view_tensor && view_tensor->src[0]) {
                إرجاع get_stride(view_tensor->src[0]);
            }
        }
    }
    إرجاع {};
}

ov::Shape GgmlOvDecoder::get_view_إدخال_ggml_shape(صحيح عقدة_idx, ثابت std::string & name, size_t view_index) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إذا (view_index < it->second.size()) {
            إرجاع get_shape(it->second[view_index].second);
        }
    }
    إرجاع {};
}

ov::Shape GgmlOvDecoder::get_view_إدخال_src_ggml_shape(صحيح عقدة_idx,
                                                       ثابت std::string & name,
                                                       size_t view_index) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إذا (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            إذا (view_tensor && view_tensor->src[0]) {
                إرجاع get_shape(view_tensor->src[0]);
            }
        }
    }
    إرجاع {};
}

ov::PartialShape GgmlOvDecoder::get_view_إدخال_ov_shape(صحيح عقدة_idx,
                                                        ثابت std::string & name,
                                                        size_t view_index) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إذا (view_index < it->second.size()) {
            auto * tensor = it->second[view_index].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            auto dynamic_it = m_عقدة_dynamic_dims.find(tensor);
            إذا (dynamic_it != m_عقدة_dynamic_dims.end() && dynamic_it->second != -1) {
                صحيح dynamic_dim_index = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
            }

            إرجاع shape;
        }
    }
    إرجاع {};
}

ov::PartialShape GgmlOvDecoder::get_view_إدخال_src_ov_shape(صحيح عقدة_idx,
                                                            ثابت std::string & name,
                                                            size_t view_index) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إذا (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            إذا (view_tensor && view_tensor->src[0]) {
                auto * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                auto dynamic_it = m_عقدة_dynamic_dims.find(src_tensor);
                إذا (dynamic_it != m_عقدة_dynamic_dims.end() && dynamic_it->second != -1) {
                    صحيح dynamic_dim_index = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_index] = m_is_static ? get_static_n_tokens() : -1;
                }

                إرجاع shape;
            }
        }
    }
    إرجاع {};
}

std::string GgmlOvDecoder::get_view_إدخال_name(صحيح عقدة_idx, ثابت std::string & name, size_t view_index) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إذا (view_index < it->second.size()) {
            إرجاع it->second[view_index].first;
        }
    }
    إرجاع "";
}

std::string GgmlOvDecoder::get_view_إدخال_src_name(صحيح عقدة_idx, ثابت std::string & name, size_t view_index) ثابت {
    auto it = m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.find(name);
    إذا (it != m_عقدة_info_list[عقدة_idx].عقدة_inputs_views.end()) {
        إذا (view_index < it->second.size()) {
            auto * view_tensor = it->second[view_index].second;
            إذا (view_tensor && view_tensor->src[0]) {
                إرجاع get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    إرجاع "";
}

ov::element::Type GgmlOvDecoder::get_إدخال_type(صحيح عقدة_idx, ثابت std::string & name) ثابت {
    إرجاع get_ov_type(m_عقدة_info_list[عقدة_idx].عقدة_inputs.at(name));
}

size_t GgmlOvDecoder::get_إدخال_size() ثابت {
    إرجاع m_model_inputs.size();
}

size_t GgmlOvDecoder::get_إدخال_size(صحيح عقدة_idx) ثابت {
    إرجاع m_عقدة_info_list[عقدة_idx].عقدة_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_إدخال_names(صحيح عقدة_idx) ثابت {
    إرجاع m_عقدة_info_list[عقدة_idx].عقدة_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(صحيح عقدة_idx) ثابت {
    auto * ggml_tensor = m_عقدة_info_list[عقدة_idx].عقدة;
    إرجاع ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ثابت صحيح عقدة_idx) ثابت {
    إرجاع get_ov_type(m_عقدة_info_list[عقدة_idx].عقدة);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(صحيح عقدة_idx) ثابت {
    auto * ggml_tensor = m_عقدة_info_list[عقدة_idx].عقدة;
    إرجاع get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(صحيح عقدة_idx) ثابت {
    auto * عقدة = m_عقدة_info_list[عقدة_idx].عقدة;
    إذا (عقدة->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        لكل (صحيح i = عقدة_idx + 1; i < m_cgraph->n_nodes; i++) {
            auto * consumer = m_cgraph->nodes[i];
            إذا (consumer->op != GGML_OP_VIEW || consumer->src[0] != عقدة) {
                استمر;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            auto & name = consumer->view_offs == 0 ? attn_name : state_name;
            إذا (!name.empty()) {
                إرجاع {m_عقدة_info_list[عقدة_idx].عقدة_name};
            }
            name = get_tensor_ov_name(m_cgraph, consumer);
        }
        إذا (!attn_name.empty() && !state_name.empty()) {
            إرجاع {attn_name, state_name};
        }
    }
    إرجاع {m_عقدة_info_list[عقدة_idx].عقدة_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(صحيح عقدة_idx) ثابت {
    auto * عقدة = m_عقدة_info_list[عقدة_idx].عقدة;
    إذا (!::is_inplace_op(عقدة) || عقدة->view_src == nullptr || ggml_nbytes(عقدة) == 0) {
        إرجاع "";
    }
    ثابت صحيح op_case = m_عقدة_info_list[عقدة_idx].عقدة_op_case;
    إذا (عقدة->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        إرجاع "";
    }
    إرجاع get_tensor_ov_name(m_cgraph, عقدة->view_src);
}

منطقي GgmlOvDecoder::is_view_like_alias_of(صحيح عقدة_idx, ثابت std::string & view_src_name) ثابت {
    auto * عقدة = m_عقدة_info_list[عقدة_idx].عقدة;
    إذا (عقدة->view_src == nullptr || get_tensor_ov_name(m_cgraph, عقدة->view_src) != view_src_name) {
        إرجاع خطأ;
    }
    إرجاع عقدة->op == GGML_OP_RESHAPE || عقدة->op == GGML_OP_VIEW;
}

ثابت std::string & GgmlOvDecoder::get_op_name() ثابت {
    ساكن ثابت std::string unknown_name = "UNKNOWN_OP_NAME";
    إرجاع unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(صحيح عقدة_idx) ثابت {
    auto it = m_عقدة_dynamic_dims.find(m_عقدة_info_list[عقدة_idx].عقدة);
    إذا (it == m_عقدة_dynamic_dims.end()) {
        إرجاع -1;
    }
    إرجاع it->second;
}

ثابت std::string & GgmlOvDecoder::get_op_name(صحيح عقدة_idx) ثابت {
    إرجاع m_عقدة_info_list[عقدة_idx].عقدة_name;
}

int32_t * GgmlOvDecoder::get_إدخال_op_params(صحيح عقدة_idx, ثابت std::string & name) ثابت {
    إرجاع m_عقدة_info_list[عقدة_idx].عقدة_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(صحيح عقدة_idx) ثابت {
    إرجاع m_عقدة_info_list[عقدة_idx].عقدة->op_params;
}

size_t GgmlOvDecoder::get_output_op_offset(صحيح عقدة_idx) ثابت {
    إرجاع m_عقدة_info_list[عقدة_idx].عقدة->view_offs;
}

فراغ GgmlOvDecoder::visit_subgraph(std::function<فراغ(std::shared_ptr<GgmlDecoder>, صحيح عقدة_idx)> عقدة_visitor) ثابت {
    لكل (صحيح عقدة_idx = 0; عقدة_idx < m_cgraph->n_nodes; عقدة_idx++) {
        إذا (m_cgraph->nodes[عقدة_idx]->op == GGML_OP_NONE) {
            استمر;
        }
        عقدة_visitor(std::make_shared<GgmlOvDecoder>(*هذا), عقدة_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ثابت ggml_tensor * عقدة) {
    بدل (عقدة->op) {
    حالة GGML_OP_UNARY:
        إرجاع std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(عقدة));
    حالة GGML_OP_GLU:
        إرجاع std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(عقدة));
    افتراضي:
        إرجاع std::string("GGML_OP_") + ggml_op_name(عقدة->op);
    }
}

ثابت std::string & GgmlOvDecoder::get_op_type(صحيح عقدة_idx) ثابت {
    إرجاع m_عقدة_info_list[عقدة_idx].عقدة_op_type;
}

ثابت std::string & GgmlOvDecoder::get_op_type() ثابت {
    ساكن ثابت std::string unknown_op = "UNKNOWN_GGML_OP";
    إرجاع unknown_op;
}

فراغ GgmlOvDecoder::compute_عقدة_dynamic_dims() {
    auto visit_عقدة = [&](auto && self, ggml_tensor * عقدة) -> فراغ {
        إذا (!عقدة) {
            إرجاع;
        }

        إذا (عقدة->op == GGML_OP_CPY) {
            m_عقدة_dynamic_dims[عقدة] = -1;
        }

        إذا (m_عقدة_dynamic_dims.count(عقدة)) {
            إرجاع;
        }
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = عقدة->src[i];
            إذا (src == nullptr) {
                استمر;
            }
            بنية ggml_tensor * جذر_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            إذا (جذر_src) {
                إذا (is_inp_tok(جذر_src, عقدة) || is_inp_pos(جذر_src, عقدة) || is_output_idx(جذر_src, عقدة)) {
                    m_عقدة_dynamic_dims[جذر_src] = 0;
                    m_عقدة_dynamic_dims[src] = m_عقدة_dynamic_dims[جذر_src];
                    استمر;
                }
                self(self, جذر_src);
                m_عقدة_dynamic_dims[src] = m_عقدة_dynamic_dims[جذر_src];
            } وإلا {
                إذا (is_inp_tok(src, عقدة) || is_inp_pos(src, عقدة) || is_output_idx(src, عقدة)) {
                    m_عقدة_dynamic_dims[src] = 0;
                    استمر;
                }
                إذا (is_inp_scale_rows(src, عقدة)) {
                    m_عقدة_dynamic_dims[src] = 1;
                    استمر;
                }
                إذا (عقدة->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_عقدة_dynamic_dims[src] = 1;
                    استمر;
                }
                self(self, src);
            }
        }
        بدل (عقدة->op) {
        حالة GGML_OP_NONE:
            m_عقدة_dynamic_dims[عقدة] = -1;
            اكسر;
        حالة GGML_OP_GET_ROWS:
            m_عقدة_dynamic_dims[عقدة] = -1;
            إذا (m_عقدة_dynamic_dims[عقدة->src[1]] != -1) {
                auto dynamic_dim_idx = m_عقدة_dynamic_dims[عقدة->src[1]];
                إذا (dynamic_dim_idx == 0) {
                    m_عقدة_dynamic_dims[عقدة] = 1;
                } وإلا {
                    auto dynamic_dim_stride = عقدة->src[1]->nb[dynamic_dim_idx] / ggml_type_size(عقدة->src[1]->type) *
                                              ggml_type_size(عقدة->src[0]->type);
                    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                        إذا (dynamic_dim_stride == عقدة->src[0]->nb[i]) {
                            m_عقدة_dynamic_dims[عقدة] = i;
                            اكسر;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            اكسر;
        حالة GGML_OP_MUL:
        حالة GGML_OP_MUL_MAT:
            m_عقدة_dynamic_dims[عقدة] = -1;
            إذا (m_عقدة_dynamic_dims[عقدة->src[0]] != -1) {
                m_عقدة_dynamic_dims[عقدة] = m_عقدة_dynamic_dims[عقدة->src[0]];
            }
            إذا (m_عقدة_dynamic_dims[عقدة->src[1]] != -1) {
                m_عقدة_dynamic_dims[عقدة] = m_عقدة_dynamic_dims[عقدة->src[1]];
            }
            اكسر;
        حالة GGML_OP_PERMUTE:
            m_عقدة_dynamic_dims[عقدة] = -1;
            إذا (m_عقدة_dynamic_dims[عقدة->src[0]] != -1) {
                auto dynamic_dim_idx = m_عقدة_dynamic_dims[عقدة->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (عقدة->op_params[i] == dynamic_dim_idx) {
                        m_عقدة_dynamic_dims[عقدة] = i;
                        اكسر;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            اكسر;
        حالة GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_عقدة_dynamic_dims[عقدة] = -1;
            إذا (m_عقدة_dynamic_dims[عقدة->src[0]] != -1) {
                إذا (عقدة->src[0]->op == GGML_OP_NONE) {
                    m_عقدة_dynamic_dims[عقدة] = m_عقدة_dynamic_dims[عقدة->src[0]];
                    اكسر;
                }
                auto dynamic_dim_idx = m_عقدة_dynamic_dims[عقدة->src[0]];
                auto dynamic_dim_قيمة = عقدة->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    عقدة->src[0]->nb[dynamic_dim_idx] / ggml_type_size(عقدة->src[0]->type) * ggml_type_size(عقدة->type);
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (عقدة->nb[i] == dynamic_dim_stride) {
                        m_عقدة_dynamic_dims[عقدة] = i;
                        اكسر;
                    }
                }
                إذا (m_عقدة_dynamic_dims[عقدة] != -1 && dynamic_dim_قيمة != عقدة->ne[m_عقدة_dynamic_dims[عقدة]]) {
                    m_عقدة_dynamic_dims[عقدة] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    إذا (ggml_nelements(عقدة) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      عقدة->name, عقدة->src[0]->name);
                    }
                }
            }
            اكسر;
        }
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RESHAPE: {
            إذا (is_same_shape(عقدة->src[0], عقدة)) {
                m_عقدة_dynamic_dims[عقدة] = m_عقدة_dynamic_dims[عقدة->src[0]];
                اكسر;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_عقدة_dynamic_dims[عقدة] = -1;
            إذا (m_عقدة_dynamic_dims[عقدة->src[0]] != -1) {
                auto dynamic_dim_idx = m_عقدة_dynamic_dims[عقدة->src[0]];
                auto dynamic_dim_stride = عقدة->src[0]->nb[dynamic_dim_idx];
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (عقدة->nb[i] == dynamic_dim_stride && عقدة->ne[i] == عقدة->src[0]->ne[dynamic_dim_idx]) {
                        m_عقدة_dynamic_dims[عقدة] = i;
                        اكسر;
                    }
                }
                إذا (m_عقدة_dynamic_dims[عقدة] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", عقدة->name);
                }
            }
            اكسر;
        }
        حالة GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr صحيح q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_عقدة_dynamic_dims[عقدة] = -1;
            إذا (m_عقدة_dynamic_dims[عقدة->src[0]] != -1) {
                auto q_dynamic_dim = m_عقدة_dynamic_dims[عقدة->src[0]];
                m_عقدة_dynamic_dims[عقدة] = q_to_out[q_dynamic_dim];
            }
            اكسر;
        }
        حالة GGML_OP_CONT:
            m_عقدة_dynamic_dims[عقدة] = -1;
            إذا (m_عقدة_dynamic_dims[عقدة->src[0]] != -1) {
                auto dynamic_dim_idx = m_عقدة_dynamic_dims[عقدة->src[0]];
                إذا (ggml_are_same_shape(عقدة, عقدة->src[0])) {
                    m_عقدة_dynamic_dims[عقدة] = dynamic_dim_idx;
                } وإلا {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(عقدة->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (عقدة->src[0]->ne[0] / ggml_blck_size(عقدة->src[0]->type));
                    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * عقدة->src[0]->ne[i - 1];
                    }

                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(عقدة->src[0]->type) *
                                              ggml_type_size(عقدة->type);
                    صحيح matched_dim_تعداد = 0;
                    صحيح first_matched_dim = -1;
                    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                        إذا (عقدة->nb[i] == dynamic_dim_stride && عقدة->ne[i] == عقدة->src[0]->ne[dynamic_dim_idx]) {
                            إذا (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_عقدة_dynamic_dims[عقدة] = i;
                            matched_dim_تعداد++;
                        }
                    }
                    إذا (matched_dim_تعداد > 1 && عقدة->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_عقدة_dynamic_dims[عقدة] = first_matched_dim;
                    } وإلا إذا (matched_dim_تعداد != 1) {
                        m_عقدة_dynamic_dims[عقدة] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      عقدة->name, عقدة->src[0]->name);
                    }
                }
            }
            اكسر;
        حالة GGML_OP_CONCAT:
            لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                إذا (عقدة->src[0]->ne[i] != عقدة->ne[i]) {
                    m_عقدة_dynamic_dims[عقدة] = i;
                    اكسر;
                }
            }
            اكسر;
        حالة GGML_OP_SSM_CONV:
        حالة GGML_OP_GATED_DELTA_NET:
            m_عقدة_dynamic_dims[عقدة] = 1;
            اكسر;
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_L2_NORM:
        حالة GGML_OP_NORM:
        حالة GGML_OP_ADD:
        حالة GGML_OP_SUB:
        حالة GGML_OP_GLU:
        حالة GGML_OP_ROPE:
        حالة GGML_OP_SCALE:
        حالة GGML_OP_SOFT_MAX:
        حالة GGML_OP_ARGSORT:
        حالة GGML_OP_ADD_ID:
        حالة GGML_OP_UNARY:
        حالة GGML_OP_CUMSUM:
        حالة GGML_OP_FILL:
        حالة GGML_OP_SET:
        حالة GGML_OP_DIAG:
        حالة GGML_OP_TRI:
        حالة GGML_OP_REPEAT:
        حالة GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        حالة GGML_OP_DIV:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_PAD:
        حالة GGML_OP_UPSCALE:
        حالة GGML_OP_SIN:
        حالة GGML_OP_COS:
        حالة GGML_OP_LOG:
        حالة GGML_OP_ROLL:
            m_عقدة_dynamic_dims[عقدة] = m_عقدة_dynamic_dims[عقدة->src[0]];
            اكسر;
        حالة GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_عقدة_dynamic_dims[عقدة] =
                (m_عقدة_dynamic_dims[عقدة->src[0]] == 0) ? -1 : m_عقدة_dynamic_dims[عقدة->src[0]];
            اكسر;
        حالة GGML_OP_MUL_MAT_ID:
        حالة GGML_OP_SOLVE_TRI:
            m_عقدة_dynamic_dims[عقدة] = m_عقدة_dynamic_dims[عقدة->src[1]];
            اكسر;
        حالة GGML_OP_CPY:
        حالة GGML_OP_SET_ROWS:
        حالة GGML_OP_SUM:
        حالة GGML_OP_MEAN:
            m_عقدة_dynamic_dims[عقدة] = -1;
            اكسر;
        حالة GGML_OP_IM2COL: {
            m_عقدة_dynamic_dims[عقدة] = -1;
            إذا (m_عقدة_dynamic_dims[عقدة->src[1]] != -1) {
                ثابت منطقي is_2D = عقدة->op_params[6] == 1;
                ثابت صحيح src_dyn = m_عقدة_dynamic_dims[عقدة->src[1]];
                إذا (is_2D) {
                    إذا (src_dyn == 0) {
                        m_عقدة_dynamic_dims[عقدة] = 1;  // IW -> OW
                    } وإلا إذا (src_dyn == 1) {
                        m_عقدة_dynamic_dims[عقدة] = 2;  // IH -> OH
                    } وإلا إذا (src_dyn == 3) {
                        m_عقدة_dynamic_dims[عقدة] = 3;  // N  -> N
                    }
                } وإلا {
                    إذا (src_dyn == 0) {
                        m_عقدة_dynamic_dims[عقدة] = 1;  // IW -> OW
                    } وإلا إذا (src_dyn == 2) {
                        m_عقدة_dynamic_dims[عقدة] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                إذا (m_عقدة_dynamic_dims[عقدة] != -1) {
                    OPENVINO_ASSERT(عقدة->src[1]->ne[src_dyn] == عقدة->ne[m_عقدة_dynamic_dims[عقدة]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(عقدة->name) +
                                        " and its src[1]: " + std::string(عقدة->src[1]->name));
                }
            }
            اكسر;
        }
        حالة GGML_OP_IM2COL_3D: {
            m_عقدة_dynamic_dims[عقدة] = -1;
            إذا (m_عقدة_dynamic_dims[عقدة->src[1]] != -1) {
                ثابت صحيح src_dyn = m_عقدة_dynamic_dims[عقدة->src[1]];
                إذا (src_dyn == 0) {
                    m_عقدة_dynamic_dims[عقدة] = 1;  // IW -> OW
                } وإلا إذا (src_dyn == 1) {
                    m_عقدة_dynamic_dims[عقدة] = 2;  // IH -> OH
                } وإلا إذا (src_dyn == 3) {
                    m_عقدة_dynamic_dims[عقدة] = 3;  // N  -> N
                }
                إذا (m_عقدة_dynamic_dims[عقدة] != -1) {
                    OPENVINO_ASSERT(عقدة->src[1]->ne[src_dyn] == عقدة->ne[m_عقدة_dynamic_dims[عقدة]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(عقدة->name) +
                                        " and its src[1]: " + std::string(عقدة->src[1]->name));
                }
            }
            اكسر;
        }
        افتراضي:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(عقدة->op), عقدة->name);
            اكسر;
        }
    };

    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * عقدة = m_cgraph->nodes[i];
        visit_عقدة(visit_عقدة, عقدة);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    إذا (0) {
        لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * عقدة = m_cgraph->nodes[i];
            صحيح dynamic_dim = m_عقدة_dynamic_dims[عقدة];
            std::cout << "[" << i << "] " << "node_name: " << عقدة->name << " op: " << ggml_op_name(عقدة->op)
                      << " shape: [";
            لكل (صحيح j = 0; j < 4; j++) {
                إذا (j == dynamic_dim) {
                    std::cout << "*";
                } وإلا {
                    std::cout << عقدة->ne[j];
                }
                إذا (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = عقدة->src[j];
                إذا (src == nullptr) {
                    استمر;
                }
                صحيح src_dynamic_dim = m_عقدة_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                لكل (صحيح k = 0; k < 4; k++) {
                    إذا (k == src_dynamic_dim) {
                        std::cout << "*";
                    } وإلا {
                        std::cout << src->ne[k];
                    }
                    إذا (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
