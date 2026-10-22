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
                             صحيح prefill_chunk_سائيز) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(خطأ),
    m_prefill_chunk_سائيز(prefill_chunk_سائيز),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    ساكن منطقي printed_پتو_map = خطأ;
    إذا (!printed_پتو_map) {
        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_پتو_map = سچ;
            print_tensor_پتو_map(cgraph);
        }
    }

    validate_cgraph();

    سيٽ_داخلا_output();
    compute_ڳنڍ_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    لكل (صحيح ڳنڍ_n = 0; ڳنڍ_n < cgraph->n_nodes; ڳنڍ_n++) {
        m_ڳنڍ_info_list[ڳنڍ_n].ڳنڍ_op_case = compute_op_case(m_ڳنڍ_info_list[ڳنڍ_n].ڳنڍ);
        m_ڳنڍ_info_list[ڳنڍ_n].ڳنڍ_op_type = compute_op_type(m_ڳنڍ_info_list[ڳنڍ_n].ڳنڍ);
    }

    جوڙيو_extra_inputs();
}

فراغ GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_ڳنڍ_info_list.clear();
    سيٽ_داخلا_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
منطقي GgmlOvDecoder::is_bound_منزل(ثابت ggml_cgraph * cgraph) ثابت {
    أرجع m_cgraph == cgraph && cgraph->n_nodes > 0 && m_ڳنڍ_info_list.size() == (size_t) cgraph->n_nodes &&
           m_ڳنڍ_info_list.front().ڳنڍ == cgraph->nodes[0] &&
           m_ڳنڍ_info_list.back().ڳنڍ == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = سچ;
    سيٽ_داخلا_output();
    compute_model_inputs();
    compute_model_outputs();
    لكل (صحيح ڳنڍ_n = 0; ڳنڍ_n < cgraph->n_nodes; ڳنڍ_n++) {
        m_ڳنڍ_info_list[ڳنڍ_n].ڳنڍ_op_case = compute_op_case(m_ڳنڍ_info_list[ڳنڍ_n].ڳنڍ);
        m_ڳنڍ_info_list[ڳنڍ_n].ڳنڍ_op_type = compute_op_type(m_ڳنڍ_info_list[ڳنڍ_n].ڳنڍ);
    }
}

نطاق {
منطقي is_inplace_op(ثابت ggml_tensor * ڳنڍ) {
    أرجع ڳنڍ->op == GGML_OP_سيٽ_صفون || ڳنڍ->op == GGML_OP_CPY || (ڳنڍ->op == GGML_OP_SCALE && ڳنڍ->view_src);
}

منطقي is_same_shape(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        إذا (a->ne[i] != b->ne[i]) {
            أرجع خطأ;
        }
    }
    أرجع سچ;
}

منطقي is_conv_حالتون_all_tensor(ثابت ggml_tensor * tensor) {
    أرجع tensor != خالي_اشارو && لڙي_ڀيٽ(tensor->name, "conv_states_all", لڙي_ڊيگهه("conv_states_all")) == 0;
}

منطقي is_ڀريل_single_جڳهه_writeback(ثابت ggml_tensor * ڳنڍ) {
    أرجع ڳنڍ->view_src != خالي_اشارو && ڳنڍ->view_src->ne[1] == 1 && ڳنڍ->src[1] != خالي_اشارو &&
           ڳنڍ->src[1]->op == GGML_OP_VIEW && ڳنڍ->src[1]->view_src == ڳنڍ->view_src &&
           ڳنڍ->src[1]->view_offs == 0 && ggml_nbytes(ڳنڍ->src[1]) == ggml_nbytes(ڳنڍ->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
منطقي is_moe_expert_جوڙ_جوڙيو(ثابت ggml_tensor * ڳنڍ) {
    std::vector<ثابت ggml_tensor *> leaves;
    ثابت ggml_tensor * cur = ڳنڍ;
    طالما (cur->op == GGML_OP_جوڙيو) {
        إذا (cur->src[0] == خالي_اشارو || cur->src[1] == خالي_اشارو) {
            أرجع خطأ;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ثابت ggml_tensor * بنياد = خالي_اشارو;
    std::set<int64_t> plane_indices;
    لكل (ثابت ggml_tensor * پن : leaves) {
        إذا (پن->op != GGML_OP_VIEW || پن->src[0] == خالي_اشارو) {
            أرجع خطأ;
        }
        ثابت ggml_tensor * پن_بنياد = پن->src[0];
        إذا (بنياد == خالي_اشارو) {
            بنياد = پن_بنياد;
        } وإلا إذا (پن_بنياد != بنياد) {
            أرجع خطأ;
        }
        إذا (پن->ne[0] != بنياد->ne[0] || پن->ne[1] != بنياد->ne[2] || پن->ne[2] != 1 || پن->ne[3] != 1 ||
            پن->nb[1] != بنياد->nb[2]) {
            أرجع خطأ;
        }
        إذا (بنياد->nb[1] == 0 || پن->view_offs % بنياد->nb[1] != 0) {
            أرجع خطأ;
        }
        int64_t plane = static_cast<int64_t>(پن->view_offs / بنياد->nb[1]);
        إذا (plane < 0 || plane >= بنياد->ne[1] || !plane_indices.insert(plane).second) {
            أرجع خطأ;
        }
    }

    أرجع بنياد != خالي_اشارو && بنياد->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(بنياد->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor) {
    إذا (tensor == خالي_اشارو) {
        أرجع "";
    }
    إذا ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, خالي_اشارو)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ثابت خودڪار * ڳنڍ = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        إذا (ڳنڍ != cgraph->nodes + cgraph->n_nodes) {
            أرجع std::string(tensor->name) + "#n" + std::to_string(ڳنڍ - cgraph->nodes);
        }
        ثابت خودڪار * پن = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        إذا (پن != cgraph->leafs + cgraph->n_leafs) {
            أرجع std::string(tensor->name) + "#l" + std::to_string(پن - cgraph->leafs);
        }
    }
    أرجع tensor->name;
}

ساكن std::string get_tensor_ov_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor) {
    أرجع GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

ساكن std::string get_tensor_گراف_داخلا_ov_name(ثابت GgmlOvDecoder * decoder,
                                                  ثابت ggml_cgraph * cgraph,
                                                  ثابت ggml_tensor * tensor,
                                                  ثابت ggml_tensor * op) {
    إذا (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        أرجع "inp_pos";
    }
    إذا (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        أرجع "embd";
    }
    إذا (GgmlOvDecoder::is_inp_نقاب(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ثابت منطقي is_swa = decoder->is_swa_نقاب(tensor);
        إذا (decoder->is_stateful()) {
            أرجع is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        إذا (is_swa) {
            أرجع get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    أرجع get_tensor_ov_name(cgraph, tensor);
}

فراغ GgmlOvDecoder::سيٽ_داخلا_output() {
    لكل (صحيح ڳنڍ_n = 0; ڳنڍ_n < m_cgraph->n_nodes; ڳنڍ_n++) {
        خودڪار * ڳنڍ = m_cgraph->nodes[ڳنڍ_n];

        ڳنڍ_Info current_ڳنڍ_info;
        خودڪار ڳنڍ_name = get_tensor_ov_name(m_cgraph, ڳنڍ);

        current_ڳنڍ_info.ڳنڍ = ڳنڍ;
        current_ڳنڍ_info.ڳنڍ_name = ڳنڍ_name;
        current_ڳنڍ_info.ڳنڍ_op_case = 0;
        current_ڳنڍ_info.معلومات_addr = ڳنڍ->معلومات;

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            خودڪار * src = ڳنڍ->src[i];
            إذا (src == خالي_اشارو) {
                استمر;
            }
            خودڪار src_name = get_tensor_ov_name(m_cgraph, src);
            إذا (src->flags & GGML_TENSOR_FLAG_داخلا) {
                src_name = get_tensor_گراف_داخلا_ov_name(هذا, m_cgraph, src, ڳنڍ);
            }
            current_ڳنڍ_info.ڳنڍ_inputs[src_name] = src;
            current_ڳنڍ_info.ڳنڍ_inputs_نالا.push_back(src_name);

            إذا (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_ڳنڍ_info.ڳنڍ_inputs_views[src_name])> view_chain;
                خودڪار * current = src;

                طالما (current != خالي_اشارو) {
                    خودڪار current_name = get_tensor_ov_name(m_cgraph, current);
                    إذا (current->flags & GGML_TENSOR_FLAG_داخلا) {
                        current_name = get_tensor_گراف_داخلا_ov_name(هذا, m_cgraph, current, ڳنڍ);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    إذا (current->src[0] != خالي_اشارو && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } وإلا {
                        اكسر;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_ڳنڍ_info.ڳنڍ_inputs_views[src_name] = view_chain;
            }
        }

        m_ڳنڍ_info_list.push_back(current_ڳنڍ_info);
    }
}

صحيح GgmlOvDecoder::compute_op_case(ثابت ggml_tensor * ڳنڍ) ثابت {
    صحيح op_case = 0;
    بدل (ڳنڍ->op) {
    حالة GGML_OP_RESHAPE: {
        إذا (m_naive) {
            اكسر;
        }
        خودڪار name = std::string(ڳنڍ->name);
        خودڪار * src = ڳنڍ->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        منطقي recurrent_sequence = خطأ;
        لكل (صحيح i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ثابت خودڪار * استعمال_ڪندڙ = m_cgraph->nodes[i];
            إذا (استعمال_ڪندڙ->op == GGML_OP_MUL_MAT_ID && استعمال_ڪندڙ->src[1] == ڳنڍ) {
                أرجع 1;
            } وإلا إذا (استعمال_ڪندڙ->op == GGML_OP_SSM_CONV) {
                ثابت خودڪار * concat = استعمال_ڪندڙ->src[0];
                إذا (concat->op == GGML_OP_CONCAT) {
                    ثابت خودڪار * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == ڳنڍ;
                }
            } وإلا إذا (استعمال_ڪندڙ->op == GGML_OP_UNARY && ggml_get_unary_op(استعمال_ڪندڙ) == GGML_UNARY_OP_SOFTPLUS) {
                ثابت خودڪار * biased = استعمال_ڪندڙ->src[0];
                recurrent_sequence = biased->op == GGML_OP_جوڙيو && biased->src[0] == ڳنڍ;
            }
        }
        إذا (recurrent_sequence && ڳنڍ->ne[0] == src->ne[0] && ڳنڍ->ne[3] == 1) {
            أرجع 6;
        }
        إذا (ڳنڍ->ne[0] == src->ne[0] && ڳنڍ->ne[2] == 1 && ڳنڍ->ne[3] == 1) {
            أرجع 5;
        }
        إذا (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == ڳنڍ->ne[0] && src->src[0]->ne[1] == ڳنڍ->ne[1]) {
            op_case = 4;
        } وإلا إذا (ڳنڍ->ne[0] * ڳنڍ->ne[1] == src->ne[0]) {
            op_case = 1;
        } وإلا إذا (src->ne[0] * src->ne[1] == ڳنڍ->ne[0]) {
            op_case = 2;
            إذا (src->ne[2] * src->ne[3] == ڳنڍ->ne[1]) {
                op_case = 5;
            }
        } وإلا إذا (ڳنڍ->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == ڳنڍ->ne[1]) {
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
            ثابت خودڪار * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            منطقي recurrent = gather->op == GGML_OP_GET_صفون && is_recurrent_cache(gather->src[0]);
            لكل (صحيح i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ثابت خودڪار * استعمال_ڪندڙ = m_cgraph->nodes[i];
                إذا (استعمال_ڪندڙ->op == GGML_OP_GATED_DELTA_NET) {
                    لكل (صحيح j : {3, 4}) {
                        ثابت خودڪار * gate = استعمال_ڪندڙ->src[j];
                        إذا (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == ڳنڍ;
                    }
                } وإلا إذا (استعمال_ڪندڙ->op == GGML_OP_MUL) {
                    لكل (صحيح j = 0; j < 2; ++j) {
                        ثابت خودڪار * gate = استعمال_ڪندڙ->src[j];
                        ثابت خودڪار * norm = استعمال_ڪندڙ->src[1 - j];
                        إذا (gate->op != GGML_OP_UNARY || gate->src[0] != ڳنڍ) {
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
    حالة GGML_OP_ترتيب_بدلايو: {
        إذا (ڳنڍ->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } وإلا إذا (ڳنڍ->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(ڳنڍ->view_src->name);
            صحيح layer = ڪڍو_layer_from_name(src_name).value();
            إذا (ggml_is_contiguous(ڳنڍ->src[0])) {
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
        إذا (ڳنڍ->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        اكسر;
    }
    حالة GGML_OP_GET_صفون: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        إذا (ڳنڍ->src[0]->op == GGML_OP_RESHAPE && ڳنڍ->src[0]->src[0] != خالي_اشارو &&
            is_recurrent_cache(ڳنڍ->src[0]->src[0])) {
            ثابت منطقي single_جڳهه = ڳنڍ->src[0]->src[0]->ne[1] == 1;
            إذا (ڳنڍ->src[1]->op == GGML_OP_VIEW) {
                op_case = (ڳنڍ->src[1]->view_offs == 0 ? 1 : 2) + (single_جڳهه ? 2 : 0);
            } وإلا إذا (single_جڳهه) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        اكسر;
    }
    حالة GGML_OP_ROPE: {
        ثابت صحيح mode = ڳنڍ->op_params[2];
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
        إذا (!m_model_params.has_rs_rollback && ڳنڍ->src[0] != خالي_اشارو &&
            ڳنڍ->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            اكسر;
        }
        إذا (m_is_static && ڳنڍ->src[0] != خالي_اشارو &&
            (ڳنڍ->src[0]->op == GGML_OP_GATED_DELTA_NET || ڳنڍ->src[0]->op == GGML_OP_CONCAT)) {
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
        إذا (ڳنڍ->src[0]->op == GGML_OP_VIEW) {
            خودڪار * src = ڳنڍ->src[0];
            إذا (ggml_nelements(ڳنڍ) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            إذا (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            خودڪار * src = ڳنڍ->src[0];
            إذا (ggml_nelements(ڳنڍ) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                إذا (ڳنڍ->ne[0] == src->ne[0] && ڳنڍ->ne[1] == src->ne[2] && ڳنڍ->ne[2] == 1 &&
                    ڳنڍ->nb[0] == src->nb[0] && ڳنڍ->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    اكسر;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                صحيح diff_ڳڻپ = 0;
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (ڳنڍ->ne[i] != src->ne[i]) {
                        diff_ڳڻپ++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    إذا (ڳنڍ->ne[i] > src->ne[i]) {
                        أرجع 0;
                    }
                }
                إذا (diff_ڳڻپ >= 1) {
                    op_case = 0;
                }
            }
        }
        اكسر;
    }
    حالة GGML_OP_RMS_NORM: {
        إذا (ڳنڍ->src[0]->op == GGML_OP_VIEW) {
            إذا (is_same_shape(ڳنڍ->src[0]->src[0], ڳنڍ->src[0])) {
                op_case = 1;
            } وإلا إذا (!m_model_params.has_rs_rollback &&
                       ڳنڍ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } وإلا إذا (ڳنڍ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        اكسر;
    }
    حالة GGML_OP_POOL_2D: {
        ثابت ggml_op_pool pool_گهڻائي = static_cast<ggml_op_pool>(ڳنڍ->op_params[0]);
        بدل (pool_گهڻائي) {
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
        ثابت int32_t گهڻائي_flags = ڳنڍ->op_params[0];
        ثابت ggml_scale_گهڻائي scale_گهڻائي = static_cast<ggml_scale_گهڻائي>(گهڻائي_flags & 0xFF);
        بدل (scale_گهڻائي) {
        حالة GGML_SCALE_گهڻائي_سڀ_کان_ويجهو: {
            op_case = 1;
            اكسر;
        }
        حالة GGML_SCALE_گهڻائي_BILINEAR: {
            op_case = 2;
            اكسر;
        }
        حالة GGML_SCALE_گهڻائي_BICUBIC: {
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
        إذا (ڳنڍ->src[0]->op == GGML_OP_VIEW) {
            إذا (ڳنڍ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                إذا (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_ڀريل_single_جڳهه_writeback(ڳنڍ) ? 7 : 10;
                } وإلا {
                    op_case = 1;
                }
            } وإلا إذا (GgmlOvDecoder::is_conv_state_writeback(ڳنڍ)) {
                op_case = is_ڀريل_single_جڳهه_writeback(ڳنڍ) ? 8 : 2;
                اكسر;
            } وإلا إذا (is_conv_حالتون_all_tensor(ڳنڍ->view_src) && ڳنڍ->src[1] != خالي_اشارو &&
                       ڳنڍ->src[1]->op == GGML_OP_VIEW && ڳنڍ->src[1]->view_src == ڳنڍ->view_src) {
                op_case = 4;
                اكسر;
            } وإلا إذا (ڳنڍ->src[0]->src[0]->op == GGML_OP_GET_صفون && ڳنڍ->src[1] != خالي_اشارو &&
                       ڳنڍ->src[1]->op == GGML_OP_VIEW && ڳنڍ->src[1]->view_src != خالي_اشارو &&
                       is_recurrent_cache(ڳنڍ->src[1]->view_src) && ڳنڍ->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } وإلا إذا (ڳنڍ->src[0]->op == GGML_OP_GET_صفون && ڳنڍ->src[1] != خالي_اشارو &&
                   ڳنڍ->src[1]->op == GGML_OP_VIEW && ڳنڍ->src[1]->view_src != خالي_اشارو &&
                   is_recurrent_cache(ڳنڍ->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = ڳنڍ->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } وإلا إذا (ڳنڍ->src[1] != خالي_اشارو && ڳنڍ->src[1]->op == GGML_OP_VIEW && ڳنڍ->src[1]->view_src != خالي_اشارو) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ثابت ggml_tensor * kv_buf = ڳنڍ->src[1]->view_src;
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
                    إذا ((n->src[1] != خالي_اشارو && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != خالي_اشارو && n->src[2]->view_src == kv_buf)) {
                        إذا (n->src[3] != خالي_اشارو) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        اكسر;
                    }
                }
            }
        }
        اكسر;
    }
    حالة GGML_OP_جوڙيو: {
        إذا (is_moe_expert_جوڙ_جوڙيو(ڳنڍ)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        اكسر;
    }
    حالة GGML_OP_SCALE: {
        إذا (ڳنڍ->view_src && ڳنڍ->بفر->usage == GGML_BACKEND_بفر_USAGE_ANY) {
            op_case = ڳنڍ->view_src->ne[1] == 1 ? 2 : 1;
        }
        اكسر;
    }
    حالة GGML_OP_L2_NORM: {
        إذا (std::string(ڳنڍ->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        اكسر;
    }
    حالة GGML_OP_FLASH_ATTN_EXT: {
        إذا (ڳنڍ->src[1] != خالي_اشارو && ڳنڍ->src[1]->op == GGML_OP_VIEW && ڳنڍ->src[1]->view_src != خالي_اشارو) {
            ثابت ggml_tensor * kv_buf = ڳنڍ->src[1]->view_src;
            إذا (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (ڳنڍ->src[3] != خالي_اشارو) ? 1 : 2;
            }
        }
        اكسر;
    }
    افتراضي:
        اكسر;
    }
    أرجع op_case;
}

std::optional<صحيح> ڪڍو_layer_from_name(ثابت std::string & name) {
    size_t pos1 = name.find("_l");
    إذا (pos1 == std::string::npos) {
        أرجع std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    إذا (pos2 == std::string::npos) {
        pos2 = name.ڊيگهائي();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    صحيح layer = std::stoi(layer_str);
    أرجع layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
ساكن صحيح get_swa_دري_from_نقاب(ثابت ggml_tensor * نقاب) {
    إذا (نقاب->معلومات == خالي_اشارو || !ggml_backend_بفر_is_host(نقاب->بفر)) {
        أرجع -1;
    }
    إذا (نقاب->type != GGML_TYPE_F16 && نقاب->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت int64_t n_kv = نقاب->ne[0];
    ثابت int64_t n_ٽوڪن = نقاب->ne[1];
    int64_t دري = 0;

    لكل (int64_t r = 0; r < n_ٽوڪن; r++) {
        int64_t kept = 0;
        لكل (int64_t c = 0; c < n_kv; c++) {
            ثابت size_t i = (size_t) r * n_kv + c;
            ثابت عائم v = نقاب->type == GGML_TYPE_F16 ? ggml_fp16_منزل_fp32(((ثابت ggml_fp16_t *) نقاب->معلومات)[i]) :
                                                          ((ثابت عائم *) نقاب->معلومات)[i];
            إذا (v > -INFINITY) {
                kept++;
            }
        }
        دري = std::max(دري, kept);
    }

    أرجع دري > 0 ? (صحيح) دري : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, منطقي is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    خودڪار get_attention_نمونو_case = [](ثابت ggml_tensor * ڳنڍ) -> صحيح {
        إذا (ڳنڍ == خالي_اشارو) {
            أرجع -1;
        }

        بدل (ڳنڍ->op) {
        حالة GGML_OP_FLASH_ATTN_EXT:
            إذا (ڳنڍ->src[0] == خالي_اشارو || ڳنڍ->src[1] == خالي_اشارو) {
                أرجع -1;
            }
            بدل (ڳنڍ->src[1]->op) {
            حالة GGML_OP_ترتيب_بدلايو:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                إذا (ڳنڍ->src[3] != خالي_اشارو && ڳنڍ->src[1]->src[0] != خالي_اشارو &&
                    ڳنڍ->src[1]->src[0]->op == GGML_OP_VIEW) {
                    أرجع 0;
                }
                اكسر;
            حالة GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                إذا (ڳنڍ->src[3] != خالي_اشارو && ڳنڍ->src[1]->src[0] != خالي_اشارو &&
                    ڳنڍ->src[1]->src[0]->op == GGML_OP_ترتيب_بدلايو && ڳنڍ->src[1]->src[0]->src[0] != خالي_اشارو &&
                    ڳنڍ->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    أرجع 1;
                }
                اكسر;
            حالة GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                إذا (ڳنڍ->src[1]->view_src != خالي_اشارو) {
                    إذا (ڳنڍ->src[3] != خالي_اشارو) {
                        أرجع 4;  // decoder self-attention
                    }
                    أرجع 5;      // cross-attention or encoder self-attention
                }
                اكسر;
            افتراضي:
                اكسر;
            }
            اكسر;
        حالة GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            إذا (ڳنڍ->src[0] != خالي_اشارو && ڳنڍ->src[1] != خالي_اشارو && ڳنڍ->src[0]->op == GGML_OP_MUL_MAT &&
                ڳنڍ->src[0]->src[0] != خالي_اشارو && ڳنڍ->src[0]->src[1] != خالي_اشارو &&
                ڳنڍ->src[0]->src[0]->op == GGML_OP_ترتيب_بدلايو && ڳنڍ->src[0]->src[0]->src[0] != خالي_اشارو &&
                ڳنڍ->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                أرجع 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            إذا (ڳنڍ->src[0]->op == GGML_OP_جوڙيو && ڳنڍ->src[0]->src[0] != خالي_اشارو &&
                ڳنڍ->src[0]->src[0]->op == GGML_OP_MUL_MAT && ڳنڍ->src[0]->src[0]->src[0] != خالي_اشارو &&
                ڳنڍ->src[0]->src[0]->src[0]->op == GGML_OP_ترتيب_بدلايو) {
                أرجع 3;
            }
            اكسر;
        افتراضي:
            اكسر;
        }

        أرجع -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    خودڪار get_attention_op_نقاب = [&get_attention_نمونو_case](ثابت ggml_tensor * ڳنڍ) -> ثابت ggml_tensor * {
        بدل (get_attention_نمونو_case(ڳنڍ)) {
        حالة 0:
        حالة 1:
            أرجع ڳنڍ->src[3];
        حالة 2:
        حالة 3:
            أرجع ڳنڍ->src[1];
        افتراضي:
            أرجع خالي_اشارو;
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
        std::map<صحيح, ثابت ggml_tensor *> layer_نقاب;            // layer -> mask it consumes
        int64_t max_extent = 0;

        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ثابت ggml_tensor * نقاب = get_attention_op_نقاب(cgraph->nodes[i]);
            إذا (نقاب == خالي_اشارو) {
                استمر;
            }
            ثابت ggml_tensor * cache_k_ترتيب_بدلايو = خالي_اشارو;
            بدل (get_attention_نمونو_case(cgraph->nodes[i])) {
            حالة 0:  cache_k_ترتيب_بدلايو = cgraph->nodes[i]->src[1];                     اكسر;
            حالة 1:  cache_k_ترتيب_بدلايو = cgraph->nodes[i]->src[1]->src[0];             اكسر;
            حالة 2:  cache_k_ترتيب_بدلايو = cgraph->nodes[i]->src[0]->src[0];             اكسر;
            افتراضي: cache_k_ترتيب_بدلايو = cgraph->nodes[i]->src[0]->src[0]->src[0];     اكسر;
            }
            ثابت ggml_tensor * cache_k_view = cache_k_ترتيب_بدلايو->src[0];
            إذا (cache_k_view->op != GGML_OP_VIEW) {
                استمر;
            }
            ثابت ggml_tensor * پن = cache_k_view->src[0];
            خودڪار layer = ڪڍو_layer_from_name(پن->name);
            إذا (!layer.has_value()) {
                استمر;
            }
            layer_extent[layer.value()] = پن->ne[1];
            layer_نقاب[layer.value()] = نقاب;
            max_extent = std::max(max_extent, پن->ne[1]);
        }

        لكل (ثابت خودڪار & [layer, extent] : layer_extent) {
            إذا (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                إذا (model_params.swa_نقاب == خالي_اشارو) {
                    model_params.swa_نقاب = layer_نقاب[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string في_layer;
            لكل (ثابت خودڪار & [layer, extent] : layer_extent) {
                في_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (ڊگهو) max_extent, model_params.swa_layers.size(), في_layer.c_str());
        }
    }

    منطقي rope_ڏٺل = خطأ;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ڳنڍ = cgraph->nodes[i];
        ثابت صحيح attention_نمونو_case = get_attention_نمونو_case(ڳنڍ);
        إذا (attention_نمونو_case != -1) {
            ggml_tensor * cache_k_ترتيب_بدلايو = خالي_اشارو;
            ggml_tensor * نقاب = خالي_اشارو;

            بدل (attention_نمونو_case) {
            حالة 0:
                cache_k_ترتيب_بدلايو = ڳنڍ->src[1];
                نقاب = ڳنڍ->src[3];
                اكسر;
            حالة 1:
                cache_k_ترتيب_بدلايو = ڳنڍ->src[1]->src[0];
                نقاب = ڳنڍ->src[3];
                اكسر;
            حالة 2:
                cache_k_ترتيب_بدلايو = ڳنڍ->src[0]->src[0];
                نقاب = ڳنڍ->src[1];
                اكسر;
            حالة 3:
                cache_k_ترتيب_بدلايو = ڳنڍ->src[0]->src[0]->src[0];
                نقاب = ڳنڍ->src[1];
                اكسر;
            حالة 4:
            حالة 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                خودڪار * cache_k_view = ڳنڍ->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_في_seq = ڳنڍ->src[0]->ne[1];
                إذا (attention_نمونو_case == 4) {
                    compute_params.attention_سائيز = cache_k_view->ne[1];
                } وإلا {
                    compute_params.attention_سائيز_static = cache_k_view->ne[1];
                }
                استمر;
            }
            افتراضي:
                اكسر;
            }

            أكد(cache_k_ترتيب_بدلايو != خالي_اشارو);

            model_params.سر_سائيز = cache_k_ترتيب_بدلايو->ne[0];
            model_params.n_heads_kv = cache_k_ترتيب_بدلايو->ne[2];
            compute_params.داخلا_len = ڳنڍ->src[0]->ne[1];
            compute_params.token_len_في_seq = ڳنڍ->src[0]->ne[1];

            خودڪار * cache_k_view = cache_k_ترتيب_بدلايو->src[0];
            إذا (cache_k_view->op != GGML_OP_VIEW || نقاب == خالي_اشارو) {
                استمر;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            صحيح layer = ڪڍو_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ثابت منطقي layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_بفر_ctx_id = ggml_backend_openvino_بفر_get_ctx_id(cache_k->بفر);
            model_params.n_heads_kv_في_layer[layer] = cache_k_ترتيب_بدلايو->ne[2];
            إذا (layer_is_swa) {
                model_params.ctx_في_seq_swa = cache_k->ne[1];
            } وإلا {
                model_params.ctx_في_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_سرگرم = نقاب->ne[3];
            خودڪار seq_سائيز = cache_k->ne[0] * cache_k->ne[1] * ggml_type_سائيز(cache_k->type);
            size_t وٿي;
            يادگيري_نقل(&وٿي, cache_k_view->op_params, حجم(size_t));
            compute_params.seq_سرگرم_شروعات = وٿي / seq_سائيز;

            إذا (layer_is_swa) {
                compute_params.attention_سائيز_swa = نقاب->ne[0];
                compute_params.swa_دري = get_swa_دري_from_نقاب(نقاب);
            } وإلا {
                compute_params.attention_سائيز = نقاب->ne[0];
            }
            إذا (is_static) {
                compute_params.attention_سائيز = model_params.ctx_في_seq;
                compute_params.attention_سائيز_swa = model_params.ctx_في_seq_swa;
                compute_params.token_len_في_seq = 1;
            }
        }

        إذا (ڳنڍ->op == GGML_OP_MUL_MAT && ڳنڍ->src[0]->op == GGML_OP_ترتيب_بدلايو &&
            ڳنڍ->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(ڳنڍ->src[0]->view_src, ڳنڍ->view_src)) {
            إذا (ڳنڍ->src[1]->op == GGML_OP_ترتيب_بدلايو && ڳنڍ->src[1]->src[0]->op == GGML_OP_VIEW &&
                ڳنڍ->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_سائيز = ڳنڍ->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        إذا (ڳنڍ->op == GGML_OP_TRANSPOSE && ڳنڍ->src[0]->op == GGML_OP_ترتيب_بدلايو &&
            ڳنڍ->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_سائيز = ڳنڍ->ne[0];
            إذا (is_static) {
                compute_params.attention_سائيز = model_params.ctx_في_seq;
            }
        }
        إذا (ڳنڍ->op == GGML_OP_ROPE) {
            إذا (compute_params.token_len_في_seq == -1 && ڳنڍ->src[1] != خالي_اشارو) {
                compute_params.token_len_في_seq = ggml_nelements(ڳنڍ->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(حجم(model_params.rope_params) == حجم(int32_t) * 16, "rope_params size");
            إذا (!rope_ڏٺل) {
                يادگيري_نقل(model_params.rope_params, ڳنڍ->op_params, حجم(int32_t) * 16);
                rope_ڏٺل = سچ;
            } وإلا إذا (memcmp(model_params.rope_params, ڳنڍ->op_params, حجم(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = سچ;
            }
        }
        إذا (ڳنڍ->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_سائيز = ڳنڍ->src[0]->ne[0];
        }
        إذا (ڳنڍ->op == GGML_OP_SCALE && ڳنڍ->view_src != خالي_اشارو && is_recurrent_cache(ڳنڍ->view_src)) {
            إذا (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = ڳنڍ->view_src->ne[1];
            } وإلا {
                GGML_ASSERT(model_params.n_rs_slots == ڳنڍ->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(ڳنڍ) / ڳنڍ->view_src->ne[0];
            compute_params.cache_rs_reset_idx = ڳنڍ->src[0]->view_offs / ڳنڍ->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        إذا (ڳنڍ->op == GGML_OP_CPY && ڳنڍ->view_src != خالي_اشارو && is_recurrent_cache(ڳنڍ->view_src) &&
            ڳنڍ->src[1] != خالي_اشارو && ڳنڍ->src[1]->op == GGML_OP_VIEW && ڳنڍ->src[1]->view_src == ڳنڍ->view_src) {
            ثابت منطقي is_conv = is_conv_state_writeback(ڳنڍ);
            ثابت منطقي is_gdn = ڳنڍ->src[0]->op == GGML_OP_VIEW && ڳنڍ->src[0]->src[0] != خالي_اشارو &&
                                ڳنڍ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ثابت منطقي is_extra = ڳنڍ->src[0]->op == GGML_OP_GET_صفون;
            ثابت منطقي is_gdn_rollback = is_gdn && is_same_shape(ڳنڍ->src[0], ڳنڍ->src[1]);

            ثابت ggml_tensor * dest_view = ڳنڍ->src[1];
            ثابت ggml_tensor * cache = ڳنڍ->view_src;
            ثابت size_t صف_بائيٽون = cache->ne[0] * ggml_type_سائيز(cache->type);
            إذا (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = سچ;
            }
            إذا (صف_بائيٽون > 0 && (is_conv || is_gdn || is_extra) && !is_ڀريل_single_جڳهه_writeback(ڳنڍ)) {
                ComputeParams::RsWriteback writeback;
                writeback.جڳهه_begin = (صحيح) (dest_view->view_offs / صف_بائيٽون);
                إذا (is_conv) {
                    writeback.src_begin = (صحيح) (ڳنڍ->src[0]->view_offs / ڳنڍ->src[0]->view_src->nb[0]);
                } وإلا إذا (is_gdn_rollback) {
                    writeback.src_begin = (صحيح) (ڳنڍ->src[0]->view_offs / ڳنڍ->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, ڳنڍ)] = writeback;
            }
            إذا ((is_conv || is_gdn) && !is_ڀريل_single_جڳهه_writeback(ڳنڍ)) {
                compute_params.s_copy_سرگرم_جڳهه_len = (صحيح) dest_view->ne[1];
            }
        }
    }
    إذا (model_params.n_heads_kv == -1) {
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ثابت خودڪار * ڳنڍ = cgraph->nodes[i];
            ثابت ggml_tensor * نقاب = خالي_اشارو;
            إذا (ڳنڍ->op == GGML_OP_SOFT_MAX) {
                نقاب = ڳنڍ->src[1];
            } وإلا إذا (ڳنڍ->op == GGML_OP_FLASH_ATTN_EXT) {
                نقاب = ڳنڍ->src[3];
            } وإلا {
                استمر;
            }
            إذا (نقاب == خالي_اشارو || نقاب->op != GGML_OP_NONE || !(نقاب->flags & GGML_TENSOR_FLAG_داخلا) ||
                ڳنڍ->src[0] == خالي_اشارو) {
                استمر;
            }
            model_params.is_cacheless_attn = سچ;
            model_params.n_seq = 1;
            model_params.ctx_في_seq = نقاب->ne[0];
            compute_params.داخلا_len = ڳنڍ->src[0]->ne[1];
            compute_params.token_len_في_seq = compute_params.داخلا_len;
            اكسر;
        }
    }

    خودڪار * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    إذا (model_params.is_cacheless_attn) {
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ثابت خودڪار * ڳنڍ = cgraph->nodes[i];
            إذا (ڳنڍ->op == GGML_OP_GET_صفون && is_output_idx(ڳنڍ->src[1], ڳنڍ)) {
                compute_params.output_len = ڳنڍ->src[1]->ne[0];
                اكسر;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    إذا (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_في_seq * model_params.n_seq;
    أرجع {model_params, compute_params};
}

فراغ GgmlOvDecoder::validate_cgraph() ثابت {
    إذا (m_model_params.n_seq > 1 && m_is_static == سچ) {
        ارم std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_گراف_داخلا_shape(ثابت ggml_tensor * op,
                                                      ثابت ggml_tensor * داخلا_2,
                                                      صحيح dynamic_dim_اشاريو) ثابت {
    إذا (m_naive) {
        أرجع داخلا_2 != خالي_اشارو ? ov::PartialShape{get_shape(داخلا_2)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape داخلا_shape;

    إذا (is_inp_tok(داخلا_2, op) || is_inp_pos(داخلا_2, op)) {
        // tokens or positions
        صحيح len = m_is_static ? (m_is_prefill ? m_prefill_chunk_سائيز : 1) : -1;
        إذا (m_is_static && is_inp_pos(داخلا_2, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        داخلا_shape = ov::PartialShape{1, 1, 1, len};

    } وإلا إذا (is_output_idx(داخلا_2, op)) {
        // output index
        داخلا_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } وإلا إذا (is_inp_سراسري(داخلا_2, op)) {
        داخلا_shape = m_is_static ? ov::PartialShape{1, 1, داخلا_2->ne[1], m_prefill_chunk_سائيز} :
                                    ov::PartialShape{1, 1, -1, -1};

    } وإلا إذا (is_inp_scale_صفون(داخلا_2, op)) {
        داخلا_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_سائيز : 1) : -1, 1};

    } وإلا إذا (is_inp_نقاب(داخلا_2, op)) {
        //نقاب
        إذا (m_is_static) {
            داخلا_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_سائيز : 1, m_model_params.ctx};
        } وإلا إذا (m_is_stateful) {
            داخلا_shape = ov::PartialShape{1, 1, -1, -1};
        } وإلا {
            داخلا_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } وإلا إذا (is_recurrent_cache(داخلا_2)) {
        داخلا_shape = ov::PartialShape{get_shape(داخلا_2)};
        إذا (!m_is_static && !m_is_stateful && داخلا_2->ne[1] > 1) {
            داخلا_shape[2] = -1;
        }

    } وإلا إذا (is_kvcache(داخلا_2, op)) {
        // kvcache
        داخلا_shape = ov::PartialShape{get_shape(داخلا_2)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ثابت منطقي is_flat_kv = (داخلا_2->ne[1] == 1 && داخلا_2->ne[2] == 1 && داخلا_2->ne[3] == 1);
        إذا (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            داخلا_shape[2] = -1;
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
            أكد(داخلا_shape.size() == 4 && داخلا_shape[0] == 1 && داخلا_shape[1] == 1 &&
                   داخلا_shape[2].is_dynamic() && داخلا_shape[3].is_static());
            ثابت صحيح n_heads_kv = get_n_heads_kv_for_tensor(داخلا_2);
            أكد(n_heads_kv > 0 && داخلا_shape[3].get_ڊيگهائي() % n_heads_kv == 0);
            ثابت int64_t combined_dim = داخلا_shape[3].get_ڊيگهائي();  // n_heads_kv * head_size
            ثابت int64_t سر_سائيز = combined_dim / n_heads_kv;
            داخلا_shape = {داخلا_shape[0], ov::Dimension::dynamic(), n_heads_kv, سر_سائيز};
        }

    } وإلا إذا (is_kv_idx(داخلا_2, op)) {
        // kv update index
        صحيح len = m_is_static ? (m_is_prefill ? m_prefill_chunk_سائيز : 1) : -1;
        داخلا_shape = ov::PartialShape{1, 1, 1, len};

    } وإلا إذا (is_inp_s_copy(داخلا_2, op) || is_s_copy_پن(داخلا_2)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        داخلا_shape = m_is_static ? ov::PartialShape{get_shape(داخلا_2)} : ov::PartialShape{1, 1, 1, -1};

    } وإلا {
        داخلا_shape = ov::PartialShape{get_shape(داخلا_2)};
    }
    إذا (dynamic_dim_اشاريو != -1 && m_model_is_splitted) {
        داخلا_shape[3 - dynamic_dim_اشاريو] = -1;
    }
    إذا (op->op == GGML_OP_SOFT_MAX && op->src[1] != خالي_اشارو && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_داخلا && op->src[1] == داخلا_2) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        إذا (m_is_static) {
            ثابت int64_t seq_سرگرم = m_is_prefill ? m_prefill_chunk_سائيز : 1;
            داخلا_shape[2] = seq_سرگرم;
            داخلا_shape[3] = seq_سرگرم;
        } وإلا {
            داخلا_shape[2] = -1;
            داخلا_shape[3] = -1;
        }
    }
    أرجع داخلا_shape;
}

منطقي GgmlOvDecoder::is_s_copy_پن(ثابت ggml_tensor * tensor) ثابت {
    إذا (tensor == خالي_اشارو || tensor->op != GGML_OP_NONE || m_cgraph == خالي_اشارو) {
        أرجع خطأ;
    }
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت ggml_tensor * ڳنڍ = m_cgraph->nodes[i];
        إذا (ڳنڍ->op != GGML_OP_GET_صفون || ڳنڍ->src[0] == خالي_اشارو || ڳنڍ->src[1] == خالي_اشارو) {
            استمر;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ثابت ggml_tensor * idx = ڳنڍ->src[1];
        طالما (idx != خالي_اشارو && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        إذا (idx != tensor) {
            استمر;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ثابت ggml_tensor * معلومات = ڳنڍ->src[0];
        طالما (معلومات != خالي_اشارو && (معلومات->op == GGML_OP_VIEW || معلومات->op == GGML_OP_RESHAPE)) {
            معلومات = معلومات->src[0];
        }
        إذا (معلومات != خالي_اشارو && is_recurrent_cache(معلومات)) {
            أرجع سچ;
        }
    }
    أرجع خطأ;
}

فراغ GgmlOvDecoder::جوڙيو_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    خودڪار create_1d_داخلا = [هذا](ثابت std::string & name, int64_t value, منطقي force_parameter = خطأ) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    إذا (m_compute_params.attention_سائيز != -1) {
        create_1d_داخلا("attention_size", m_compute_params.attention_سائيز);
    }
    إذا (m_compute_params.attention_سائيز_static != -1) {
        create_1d_داخلا("attention_size_static", m_compute_params.attention_سائيز_static);
    }
    إذا (m_compute_params.attention_سائيز_swa != -1) {
        create_1d_داخلا("attention_size_swa", m_compute_params.attention_سائيز_swa);
    }
    // only the stateful SWA mask consumes this
    إذا (is_stateful() && m_compute_params.swa_دري != -1) {
        create_1d_داخلا("swa_window", m_compute_params.swa_دري);
    }
    create_1d_داخلا("n_seq_active", m_compute_params.n_seq_سرگرم);
    create_1d_داخلا("seq_active_start", m_compute_params.seq_سرگرم_شروعات);
    create_1d_داخلا("seq_active_end", m_compute_params.seq_سرگرم_شروعات + m_compute_params.n_seq_سرگرم);
    إذا (m_compute_params.token_len_في_seq != -1) {
        create_1d_داخلا("token_len_per_seq", m_compute_params.token_len_في_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    إذا (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_داخلا("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/سچ);
        create_1d_داخلا("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/سچ);
    }

    إذا (m_compute_params.s_copy_سرگرم_جڳهه_len != -1) {
        create_1d_داخلا("s_copy_active_slot_len", m_compute_params.s_copy_سرگرم_جڳهه_len);
        إذا (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_داخلا("chunk_valid_len", get_static_n_ٽوڪن(), /*force_parameter=*/سچ);
        }
    }

    لكل (ثابت خودڪار & [ڳنڍ_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_داخلا("rs_slot_begin_" + ڳنڍ_name, writeback.جڳهه_begin);
        إذا (!m_is_static && writeback.src_begin >= 0) {
            create_1d_داخلا("rs_src_begin_" + ڳنڍ_name, writeback.src_begin);
        }
    }
}

منطقي GgmlOvDecoder::ڳنڍ_is_استعمال_ٿيل_as_src(ثابت صحيح ڳنڍ_idx) {
    ggml_tensor * ڳنڍ = m_cgraph->nodes[ڳنڍ_idx];
    لكل (صحيح i = ڳنڍ_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_ڳنڍ = m_cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (other_ڳنڍ->src[j] == ڳنڍ) {
                أرجع سچ;
            }
        }
    }
    أرجع خطأ;
}

فراغ GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ڳنڍ = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        إذا (ڳنڍ->op == GGML_OP_NONE && ڳنڍ_is_استعمال_ٿيل_as_src(i)) {
            std::string ڳنڍ_name = get_tensor_ov_name(m_cgraph, ڳنڍ);
            إذا (m_model_weights.find(ڳنڍ_name) == m_model_weights.end()) {
                m_inputs[ڳنڍ_name] = ڳنڍ;
                m_model_inputs[ڳنڍ_name] = {get_ov_type(ڳنڍ),
                                             get_گراف_داخلا_shape(ڳنڍ, خالي_اشارو, m_ڳنڍ_dynamic_dims[ڳنڍ])};
            }
            استمر;
        }
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            خودڪار * src = ڳنڍ->src[i];
            إذا (src == خالي_اشارو) {
                استمر;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            إذا (src->flags & GGML_TENSOR_FLAG_داخلا) {
                src_name = get_tensor_گراف_داخلا_ov_name(هذا, m_cgraph, src, ڳنڍ);
            }
            إذا (m_model_weights.find(src_name) != m_model_weights.end()) {
                استمر;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            إذا (src->view_src != خالي_اشارو &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                استمر;
            }

            منطقي is_intermediate_ڳنڍ = خطأ;
            لكل (ثابت خودڪار & ڳنڍ_info : m_ڳنڍ_info_list) {
                إذا (ڳنڍ_info.ڳنڍ == src) {
                    is_intermediate_ڳنڍ = سچ;
                    اكسر;
                }
            }
            إذا (is_intermediate_ڳنڍ) {
                استمر;
            }
            إذا (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                استمر;
            }

            m_inputs[src_name] = src;

            ggml_backend_بفر * بفر = src->بفر;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            إذا (بفر->usage == GGML_BACKEND_بفر_USAGE_ANY) {
                إذا (خودڪار it = std::find(m_model_params.kv_نالا.begin(), m_model_params.kv_نالا.end(), src_name);
                    it == m_model_params.kv_نالا.end()) {
                    m_model_params.kv_نالا.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            طالما (src->op == GGML_OP_VIEW && src->src[0] != خالي_اشارو) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_گراف_داخلا_shape(ڳنڍ, src, m_ڳنڍ_dynamic_dims[src])};
        }
    }
}

فراغ GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_نالا.clear();
    لكل (صحيح ڳنڍ_n = 0; ڳنڍ_n < m_cgraph->n_nodes; ڳنڍ_n++) {
        خودڪار * cur_ڳنڍ = m_cgraph->nodes[ڳنڍ_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        إذا (cur_ڳنڍ->op == GGML_OP_NONE || cur_ڳنڍ->op == GGML_OP_VIEW || cur_ڳنڍ->op == GGML_OP_RESHAPE) {
            استمر;
        }
        إذا (::is_inplace_op(cur_ڳنڍ) && ggml_nbytes(cur_ڳنڍ) == 0) {
            استمر;
        }
        خودڪار cur_ڳنڍ_use_ڳڻپ = m_cgraph->use_counts[ggml_hash_ڳوليو(&m_cgraph->گهميل_hash_سيٽ, cur_ڳنڍ)];
        إذا (cur_ڳنڍ_use_ڳڻپ == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            إذا (cur_ڳنڍ != خالي_اشارو && ::is_inplace_op(cur_ڳنڍ) && ggml_nbytes(cur_ڳنڍ) > 0) {
                cur_ڳنڍ = cur_ڳنڍ->view_src;
            }
        } وإلا {
            صحيح داخلا_use_ڳڻپ = 0;
            لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * ڳنڍ = m_cgraph->nodes[i];
                لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                    إذا (ڳنڍ->src[j] != عدم && ڳنڍ->src[j] == cur_ڳنڍ) {
                        داخلا_use_ڳڻپ++;
                    }
                }
            }
            إذا (داخلا_use_ڳڻپ == cur_ڳنڍ_use_ڳڻپ) {
                cur_ڳنڍ = خالي_اشارو;
            }
        }
        إذا (cur_ڳنڍ != خالي_اشارو) {
            std::string cur_ڳنڍ_name = get_tensor_ov_name(m_cgraph, cur_ڳنڍ);
            m_model_outputs[cur_ڳنڍ_name] = cur_ڳنڍ;
            m_model_output_نالا.insert(cur_ڳنڍ_name);
        }
    }
}

ثابت ggml_tensor * GgmlOvDecoder::get_tensor_استعمال_ٿيل_op(ثابت ggml_tensor * tensor) ثابت {
    إذا (tensor == خالي_اشارو) {
        أرجع خالي_اشارو;
    }
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت خودڪار * ڳنڍ = m_cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (ڳنڍ->src[j] == tensor) {
                أرجع ڳنڍ;
            }
        }
    }
    أرجع خالي_اشارو;
}

ثابت ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ثابت std::string & name) ثابت {
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت خودڪار * ڳنڍ = m_cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            ثابت خودڪار * src = ڳنڍ->src[j];
            إذا (src == خالي_اشارو) {
                اكسر;
            }
            إذا (get_tensor_ov_name(m_cgraph, src) == name) {
                أرجع src;
            }
        }
    }
    أرجع خالي_اشارو;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_نالا() ثابت {
    std::map<std::string, std::string> kv_param_res_نالا;
    لكل (ثابت خودڪار & name : m_model_params.kv_نالا) {
        kv_param_res_نالا[name] = name;
    }
    أرجع kv_param_res_نالا;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
ساكن منطقي is_mul_mat_id_expert_وزن(ثابت ggml_tensor * ڳنڍ, صحيح src_اشاريو) {
    أرجع ڳنڍ->op == GGML_OP_MUL_MAT_ID && src_اشاريو == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_وزن_nodes(ggml_cgraph * cgraph, منطقي naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    خودڪار * nodes = cgraph->nodes;
    خودڪار n_nodes = cgraph->n_nodes;
    لكل (صحيح ڳنڍ_i = 0; ڳنڍ_i < n_nodes; ڳنڍ_i++) {
        خودڪار * ڳنڍ = nodes[ڳنڍ_i];
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            خودڪار * src = ڳنڍ->src[i];
            إذا (src == خالي_اشارو) {
                استمر;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * بنياد = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, بنياد);
            إذا (is_rope_freqs_وزن(بنياد, ڳنڍ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_بفر * بفر = بنياد->بفر;
            إذا (بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS || ggml_is_quantized(بنياد->type) ||
                is_mul_mat_id_expert_وزن(ڳنڍ, i)) {
                إذا (model_weights.find(src_name) == model_weights.end()) {
                    خودڪار وزن_ڳنڍ = create_وزن_ڳنڍ(بنياد, naive);
                    وزن_ڳنڍ->سيٽ_friendly_name(src_name);
                    model_weights[src_name] = وزن_ڳنڍ;
                }
            }
        }
    }
    أرجع model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
ساكن std::mutex g_nonov_وزن_cache_mutex;
ساكن std::unordered_map<ثابت فراغ *, std::shared_ptr<ov::Node>> g_nonov_وزن_cache;

std::set<std::string> GgmlOvDecoder::collect_وزن_نالا(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> نالا;
    لكل (صحيح ڳنڍ_i = 0; ڳنڍ_i < cgraph->n_nodes; ڳنڍ_i++) {
        خودڪار * ڳنڍ = cgraph->nodes[ڳنڍ_i];
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            خودڪار * src = ڳنڍ->src[i];
            إذا (src == خالي_اشارو) {
                استمر;
            }
            ثابت ggml_tensor * بنياد = src->view_src ? src->view_src : src;
            std::string src_name(بنياد->name);
            إذا (is_rope_freqs_وزن(بنياد, ڳنڍ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_بفر * بفر = بنياد->بفر;
            إذا (بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS || ggml_is_quantized(بنياد->type)) {
                نالا.insert(src_name);
            }
        }
    }
    أرجع نالا;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_وزن_ڳنڍ(ggml_tensor * tensor, منطقي naive) {
    ثابت منطقي is_ov_بفر = ggml_backend_بفر_is_openvino(tensor->بفر);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    إذا (tensor->extra) {
        OPENVINO_ASSERT(is_ov_بفر, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        خودڪار * extra_بنياد = static_cast<ggml_openvino_extra_بنياد *>(tensor->extra);

        إذا (extra_بنياد->type == ggml_openvino_extra_بنياد::Type::وزن_2) {
            // F16/F32/BF16 weight with shared-memory constant
            خودڪار * وزن_extra = static_cast<ggml_openvino_وزن_extra *>(tensor->extra);
            إذا (وزن_extra->وزن_ڳنڍ) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                أرجع وزن_extra->وزن_ڳنڍ;
            }
        } وإلا إذا (extra_بنياد->type == ggml_openvino_extra_بنياد::Type::QUANTIZED_وزن) {
            // Quantized weight with pre-extracted data
            خودڪار * quant_extra = static_cast<ggml_openvino_quantized_وزن_extra *>(tensor->extra);
            إذا (quant_extra->وزن_ڳنڍ) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                أرجع quant_extra->وزن_ڳنڍ;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    إذا (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        خودڪار وزن_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->معلومات);
        خودڪار وزن_ڳنڍ = std::make_shared<ov::op::v0::Constant>(وزن_tensor);
        وزن_ڳنڍ->سيٽ_friendly_name(tensor->name);
        أرجع وزن_ڳنڍ;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ثابت منطقي cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_بفر &&
                                 !naive && tensor->معلومات != خالي_اشارو;
    إذا (cacheable_nonov) {
        std::lock_guard<std::mutex> تالو(g_nonov_وزن_cache_mutex);
        خودڪار it = g_nonov_وزن_cache.find(tensor->معلومات);
        إذا (it != g_nonov_وزن_cache.end()) {
            أرجع it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    ساكن ثابت std::set<ggml_type> وزن_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    إذا (وزن_types.find(tensor->type) == وزن_types.end()) {
        ارم std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    Ov_وزن ov_وزن;
    إذا (ggml_is_quantized(tensor->type)) {
        خودڪار use_bias = naive;
        إذا (is_ov_بفر) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_سائيز = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_سائيز);
            يادگيري_نقل(tmp.معلومات(), tensor->معلومات, raw_سائيز);
            ov_وزن = process_وزن_tensor(tensor, tmp.معلومات(), tensor->معلومات, use_bias);
        } وإلا {
            ov_وزن = process_وزن_tensor(tensor, tensor->معلومات, خالي_اشارو, use_bias);
        }
    } وإلا {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_وزن = process_وزن_tensor(tensor, tensor->معلومات, tensor->معلومات);
    }

    ov_وزن.وزن_ڳنڍ->سيٽ_friendly_name(tensor->name);
    إذا (!is_ov_بفر) {
        إذا (cacheable_nonov) {
            std::lock_guard<std::mutex> تالو(g_nonov_وزن_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            خودڪار [it, inserted] = g_nonov_وزن_cache.emplace(tensor->معلومات, ov_وزن.وزن_ڳنڍ);
            أرجع it->second;
        }
        أرجع ov_وزن.وزن_ڳنڍ;
    }

    ggml_openvino_extra_بنياد * extra;
    إذا (ov_وزن.is_quantized()) {
        extra = جديد ggml_openvino_quantized_وزن_extra(std::move(ov_وزن.weights), std::move(ov_وزن.scales),
                                                         std::move(ov_وزن.zp), ov_وزن.وزن_ڳنڍ);
    } وإلا {
        extra = جديد ggml_openvino_وزن_extra(std::move(ov_وزن.weights), ov_وزن.وزن_ڳنڍ);
    }
    ggml_openvino_بفر_رجسٽر_extra(tensor, extra);

    أرجع ov_وزن.وزن_ڳنڍ;
}

فراغ GgmlOvDecoder::dump_cgraph(ثابت ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    إذا (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        أرجع;
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
        ggml_tensor * ڳنڍ = cgraph->nodes[i];

        // Get buffer type name
        ثابت حرف * buf_name = "none";
        ggml_backend_بفر_t buf = ڳنڍ->view_src ? ڳنڍ->view_src->بفر : ڳنڍ->بفر;
        إذا (buf) {
            buf_name = ggml_backend_بفر_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ڳنڍ->ne[0] << ", "
             << std::setw(5) << ڳنڍ->ne[1] << ", "
             << std::setw(5) << ڳنڍ->ne[2] << ", "
             << std::setw(5) << ڳنڍ->ne[3] << "] "
             << std::کاٻي << std::setw(20) << ggml_op_name(ڳنڍ->op) << std::ساڄي << " "
             << std::کاٻي << std::setw(45) << ڳنڍ->name << std::ساڄي
             << std::setw(2) << "[ "
             << std::setw(0) << ڳنڍ->nb[0] << ", "
             << std::setw(5) << ڳنڍ->nb[1] << ", "
             << std::setw(5) << ڳنڍ->nb[2] << ", "
             << std::setw(5) << ڳنڍ->nb[3] << "] "
             << std::ساڄي << std::setw(15) << buf_name << std::ساڄي
             << "\n";

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            إذا (خودڪار* src = ڳنڍ->src[i]) {
                // Get buffer type name for source
                ثابت حرف * src_buf_name = "none";
                ggml_backend_بفر_t src_buf = src->view_src ? src->view_src->بفر : src->بفر;
                إذا (src_buf) {
                    src_buf_name = ggml_backend_بفر_name(src_buf);
                }

                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::کاٻي << std::setw(12) << ggml_op_name(src->op) << std::ساڄي;
                file << std::کاٻي << std::setw(30) << src->name << std::ساڄي
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << std::ساڄي << std::setw(15) << src_buf_name << std::ساڄي
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * ڳنڍ = cgraph->leafs[i];

        // Get buffer type name for leaf
        ثابت حرف * پن_buf_name = "none";
        ggml_backend_بفر_t پن_buf = ڳنڍ->view_src ? ڳنڍ->view_src->بفر : ڳنڍ->بفر;
        إذا (پن_buf) {
            پن_buf_name = ggml_backend_بفر_name(پن_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ڳنڍ->ne[0] << ", "
             << std::setw(5) << ڳنڍ->ne[1] << "] "
             << std::setw(8) << ggml_op_name(ڳنڍ->op) << " "
             << std::setw(16) << ggml_get_name(ڳنڍ)
             << std::setw(20) << پن_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

فراغ print_tensor_پتو_map(ثابت ggml_cgraph * cgraph) {
    std::map<فراغ *, std::vector<std::string>> پتو_map;
    لكل (صحيح ڳنڍ_n = 0; ڳنڍ_n < cgraph->n_nodes; ڳنڍ_n++) {
        خودڪار * ڳنڍ = cgraph->nodes[ڳنڍ_n];
        إذا (ڳنڍ->معلومات) {
            خودڪار it = پتو_map.find(ڳنڍ->معلومات);
            إذا (it == پتو_map.end()) {
                پتو_map[ڳنڍ->معلومات] = std::vector<std::string>();
            }
            پتو_map[ڳنڍ->معلومات].push_back(ڳنڍ->name);
        }
    }
    لكل (ثابت خودڪار & pair : پتو_map) {
        std::ٻاهر << "Address: " << pair.پهريون << '\n';
        لكل (ثابت خودڪار & name : pair.second) {
            std::ٻاهر << name << " ; ";
        }
        std::ٻاهر << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ثابت ggml_tensor * tensor) {
    std::vector<size_t> shape;
    لكل (صحيح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    أرجع shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(ثابت ggml_tensor * tensor) {
    std::vector<size_t> stride;
    لكل (صحيح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    أرجع stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(ثابت ggml_tensor * tensor) {
    بدل (tensor->type) {
    حالة GGML_TYPE_F64:
        أرجع ov::element::f64;
    حالة GGML_TYPE_F32:
        أرجع ov::element::f32;
    حالة GGML_TYPE_F16:
        أرجع ov::element::f16;
    حالة GGML_TYPE_BF16:
        أرجع ov::element::bf16;
    حالة GGML_TYPE_I8:
        أرجع ov::element::i8;
    حالة GGML_TYPE_I16:
        أرجع ov::element::i16;
    حالة GGML_TYPE_I32:
        أرجع ov::element::i32;
    حالة GGML_TYPE_I64:
        أرجع ov::element::i64;
    افتراضي:
        أرجع ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_داخلا_shape(صحيح ڳنڍ_idx, ثابت std::string & name) ثابت {
    أرجع ov::PartialShape(get_shape(m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_داخلا_stride(صحيح ڳنڍ_idx, ثابت std::string & name) ثابت {
    أرجع get_stride(m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_داخلا_سائيز(صحيح ڳنڍ_idx, ثابت std::string & name) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        أرجع it->second.size();
    }
    أرجع 0;
}

size_t GgmlOvDecoder::get_view_داخلا_وٿي(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        إذا (view_اشاريو < it->second.size()) {
            أرجع it->second[view_اشاريو].second->view_offs;
        }
    }
    أرجع 0;
}

size_t GgmlOvDecoder::get_view_داخلا_src_وٿي(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        إذا (view_اشاريو < it->second.size()) {
            خودڪار * view_tensor = it->second[view_اشاريو].second;
            إذا (view_tensor && view_tensor->src[0]) {
                أرجع view_tensor->src[0]->view_offs;
            }
        }
    }
    أرجع 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_داخلا_stride(صحيح ڳنڍ_idx,
                                                         ثابت std::string & name,
                                                         size_t view_اشاريو) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        إذا (view_اشاريو < it->second.size()) {
            أرجع get_stride(it->second[view_اشاريو].second);
        }
    }
    أرجع {};
}

std::vector<size_t> GgmlOvDecoder::get_view_داخلا_src_stride(صحيح ڳنڍ_idx,
                                                             ثابت std::string & name,
                                                             size_t view_اشاريو) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        إذا (view_اشاريو < it->second.size()) {
            خودڪار * view_tensor = it->second[view_اشاريو].second;
            إذا (view_tensor && view_tensor->src[0]) {
                أرجع get_stride(view_tensor->src[0]);
            }
        }
    }
    أرجع {};
}

ov::Shape GgmlOvDecoder::get_view_داخلا_ggml_shape(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        إذا (view_اشاريو < it->second.size()) {
            أرجع get_shape(it->second[view_اشاريو].second);
        }
    }
    أرجع {};
}

ov::Shape GgmlOvDecoder::get_view_داخلا_src_ggml_shape(صحيح ڳنڍ_idx,
                                                       ثابت std::string & name,
                                                       size_t view_اشاريو) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        إذا (view_اشاريو < it->second.size()) {
            خودڪار * view_tensor = it->second[view_اشاريو].second;
            إذا (view_tensor && view_tensor->src[0]) {
                أرجع get_shape(view_tensor->src[0]);
            }
        }
    }
    أرجع {};
}

ov::PartialShape GgmlOvDecoder::get_view_داخلا_ov_shape(صحيح ڳنڍ_idx,
                                                        ثابت std::string & name,
                                                        size_t view_اشاريو) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        إذا (view_اشاريو < it->second.size()) {
            خودڪار * tensor = it->second[view_اشاريو].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            خودڪار dynamic_it = m_ڳنڍ_dynamic_dims.find(tensor);
            إذا (dynamic_it != m_ڳنڍ_dynamic_dims.end() && dynamic_it->second != -1) {
                صحيح dynamic_dim_اشاريو = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_اشاريو] = m_is_static ? get_static_n_ٽوڪن() : -1;
            }

            أرجع shape;
        }
    }
    أرجع {};
}

ov::PartialShape GgmlOvDecoder::get_view_داخلا_src_ov_shape(صحيح ڳنڍ_idx,
                                                            ثابت std::string & name,
                                                            size_t view_اشاريو) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        إذا (view_اشاريو < it->second.size()) {
            خودڪار * view_tensor = it->second[view_اشاريو].second;
            إذا (view_tensor && view_tensor->src[0]) {
                خودڪار * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                خودڪار dynamic_it = m_ڳنڍ_dynamic_dims.find(src_tensor);
                إذا (dynamic_it != m_ڳنڍ_dynamic_dims.end() && dynamic_it->second != -1) {
                    صحيح dynamic_dim_اشاريو = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_اشاريو] = m_is_static ? get_static_n_ٽوڪن() : -1;
                }

                أرجع shape;
            }
        }
    }
    أرجع {};
}

std::string GgmlOvDecoder::get_view_داخلا_name(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        إذا (view_اشاريو < it->second.size()) {
            أرجع it->second[view_اشاريو].پهريون;
        }
    }
    أرجع "";
}

std::string GgmlOvDecoder::get_view_داخلا_src_name(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت {
    خودڪار it = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.find(name);
    إذا (it != m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_views.end()) {
        إذا (view_اشاريو < it->second.size()) {
            خودڪار * view_tensor = it->second[view_اشاريو].second;
            إذا (view_tensor && view_tensor->src[0]) {
                أرجع get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    أرجع "";
}

ov::element::Type GgmlOvDecoder::get_داخلا_type(صحيح ڳنڍ_idx, ثابت std::string & name) ثابت {
    أرجع get_ov_type(m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs.at(name));
}

size_t GgmlOvDecoder::get_داخلا_سائيز() ثابت {
    أرجع m_model_inputs.size();
}

size_t GgmlOvDecoder::get_داخلا_سائيز(صحيح ڳنڍ_idx) ثابت {
    أرجع m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_نالا.size();
}

std::vector<std::string> GgmlOvDecoder::get_داخلا_نالا(صحيح ڳنڍ_idx) ثابت {
    أرجع m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs_نالا;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(صحيح ڳنڍ_idx) ثابت {
    خودڪار * ggml_tensor = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ;
    أرجع ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ثابت صحيح ڳنڍ_idx) ثابت {
    أرجع get_ov_type(m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(صحيح ڳنڍ_idx) ثابت {
    خودڪار * ggml_tensor = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ;
    أرجع get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_نالا(صحيح ڳنڍ_idx) ثابت {
    خودڪار * ڳنڍ = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ;
    إذا (ڳنڍ->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        لكل (صحيح i = ڳنڍ_idx + 1; i < m_cgraph->n_nodes; i++) {
            خودڪار * استعمال_ڪندڙ = m_cgraph->nodes[i];
            إذا (استعمال_ڪندڙ->op != GGML_OP_VIEW || استعمال_ڪندڙ->src[0] != ڳنڍ) {
                استمر;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            خودڪار & name = استعمال_ڪندڙ->view_offs == 0 ? attn_name : state_name;
            إذا (!name.خالي_جايون()) {
                أرجع {m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_name};
            }
            name = get_tensor_ov_name(m_cgraph, استعمال_ڪندڙ);
        }
        إذا (!attn_name.خالي_جايون() && !state_name.خالي_جايون()) {
            أرجع {attn_name, state_name};
        }
    }
    أرجع {m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(صحيح ڳنڍ_idx) ثابت {
    خودڪار * ڳنڍ = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ;
    إذا (!::is_inplace_op(ڳنڍ) || ڳنڍ->view_src == خالي_اشارو || ggml_nbytes(ڳنڍ) == 0) {
        أرجع "";
    }
    ثابت صحيح op_case = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_op_case;
    إذا (ڳنڍ->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_سرگرم_جڳهه_len == -1) {
        أرجع "";
    }
    أرجع get_tensor_ov_name(m_cgraph, ڳنڍ->view_src);
}

منطقي GgmlOvDecoder::is_view_like_alias_جو(صحيح ڳنڍ_idx, ثابت std::string & view_src_name) ثابت {
    خودڪار * ڳنڍ = m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ;
    إذا (ڳنڍ->view_src == خالي_اشارو || get_tensor_ov_name(m_cgraph, ڳنڍ->view_src) != view_src_name) {
        أرجع خطأ;
    }
    أرجع ڳنڍ->op == GGML_OP_RESHAPE || ڳنڍ->op == GGML_OP_VIEW;
}

ثابت std::string & GgmlOvDecoder::get_op_name() ثابت {
    ساكن ثابت std::string unknown_name = "UNKNOWN_OP_NAME";
    أرجع unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(صحيح ڳنڍ_idx) ثابت {
    خودڪار it = m_ڳنڍ_dynamic_dims.find(m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ);
    إذا (it == m_ڳنڍ_dynamic_dims.end()) {
        أرجع -1;
    }
    أرجع it->second;
}

ثابت std::string & GgmlOvDecoder::get_op_name(صحيح ڳنڍ_idx) ثابت {
    أرجع m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_name;
}

int32_t * GgmlOvDecoder::get_داخلا_op_params(صحيح ڳنڍ_idx, ثابت std::string & name) ثابت {
    أرجع m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(صحيح ڳنڍ_idx) ثابت {
    أرجع m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ->op_params;
}

size_t GgmlOvDecoder::get_output_op_وٿي(صحيح ڳنڍ_idx) ثابت {
    أرجع m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ->view_offs;
}

فراغ GgmlOvDecoder::گهمو_subgraph(std::function<فراغ(std::shared_ptr<GgmlDecoder>, صحيح ڳنڍ_idx)> ڳنڍ_visitor) ثابت {
    لكل (صحيح ڳنڍ_idx = 0; ڳنڍ_idx < m_cgraph->n_nodes; ڳنڍ_idx++) {
        إذا (m_cgraph->nodes[ڳنڍ_idx]->op == GGML_OP_NONE) {
            استمر;
        }
        ڳنڍ_visitor(std::make_shared<GgmlOvDecoder>(*هذا), ڳنڍ_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ثابت ggml_tensor * ڳنڍ) {
    بدل (ڳنڍ->op) {
    حالة GGML_OP_UNARY:
        أرجع std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(ڳنڍ));
    حالة GGML_OP_GLU:
        أرجع std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(ڳنڍ));
    افتراضي:
        أرجع std::string("GGML_OP_") + ggml_op_name(ڳنڍ->op);
    }
}

ثابت std::string & GgmlOvDecoder::get_op_type(صحيح ڳنڍ_idx) ثابت {
    أرجع m_ڳنڍ_info_list[ڳنڍ_idx].ڳنڍ_op_type;
}

ثابت std::string & GgmlOvDecoder::get_op_type() ثابت {
    ساكن ثابت std::string unknown_op = "UNKNOWN_GGML_OP";
    أرجع unknown_op;
}

فراغ GgmlOvDecoder::compute_ڳنڍ_dynamic_dims() {
    خودڪار گهمو_ڳنڍ = [&](خودڪار && self, ggml_tensor * ڳنڍ) -> فراغ {
        إذا (!ڳنڍ) {
            أرجع;
        }

        إذا (ڳنڍ->op == GGML_OP_CPY) {
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
        }

        إذا (m_ڳنڍ_dynamic_dims.count(ڳنڍ)) {
            أرجع;
        }
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = ڳنڍ->src[i];
            إذا (src == خالي_اشارو) {
                استمر;
            }
            بنية ggml_tensor * پاڙ_src = خالي_اشارو;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            إذا (پاڙ_src) {
                إذا (is_inp_tok(پاڙ_src, ڳنڍ) || is_inp_pos(پاڙ_src, ڳنڍ) || is_output_idx(پاڙ_src, ڳنڍ)) {
                    m_ڳنڍ_dynamic_dims[پاڙ_src] = 0;
                    m_ڳنڍ_dynamic_dims[src] = m_ڳنڍ_dynamic_dims[پاڙ_src];
                    استمر;
                }
                self(self, پاڙ_src);
                m_ڳنڍ_dynamic_dims[src] = m_ڳنڍ_dynamic_dims[پاڙ_src];
            } وإلا {
                إذا (is_inp_tok(src, ڳنڍ) || is_inp_pos(src, ڳنڍ) || is_output_idx(src, ڳنڍ)) {
                    m_ڳنڍ_dynamic_dims[src] = 0;
                    استمر;
                }
                إذا (is_inp_scale_صفون(src, ڳنڍ)) {
                    m_ڳنڍ_dynamic_dims[src] = 1;
                    استمر;
                }
                إذا (ڳنڍ->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_ڳنڍ_dynamic_dims[src] = 1;
                    استمر;
                }
                self(self, src);
            }
        }
        بدل (ڳنڍ->op) {
        حالة GGML_OP_NONE:
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            اكسر;
        حالة GGML_OP_GET_صفون:
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            إذا (m_ڳنڍ_dynamic_dims[ڳنڍ->src[1]] != -1) {
                خودڪار dynamic_dim_idx = m_ڳنڍ_dynamic_dims[ڳنڍ->src[1]];
                إذا (dynamic_dim_idx == 0) {
                    m_ڳنڍ_dynamic_dims[ڳنڍ] = 1;
                } وإلا {
                    خودڪار dynamic_dim_stride = ڳنڍ->src[1]->nb[dynamic_dim_idx] / ggml_type_سائيز(ڳنڍ->src[1]->type) *
                                              ggml_type_سائيز(ڳنڍ->src[0]->type);
                    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                        إذا (dynamic_dim_stride == ڳنڍ->src[0]->nb[i]) {
                            m_ڳنڍ_dynamic_dims[ڳنڍ] = i;
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
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            إذا (m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]] != -1) {
                m_ڳنڍ_dynamic_dims[ڳنڍ] = m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]];
            }
            إذا (m_ڳنڍ_dynamic_dims[ڳنڍ->src[1]] != -1) {
                m_ڳنڍ_dynamic_dims[ڳنڍ] = m_ڳنڍ_dynamic_dims[ڳنڍ->src[1]];
            }
            اكسر;
        حالة GGML_OP_ترتيب_بدلايو:
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            إذا (m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]] != -1) {
                خودڪار dynamic_dim_idx = m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (ڳنڍ->op_params[i] == dynamic_dim_idx) {
                        m_ڳنڍ_dynamic_dims[ڳنڍ] = i;
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
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            إذا (m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]] != -1) {
                إذا (ڳنڍ->src[0]->op == GGML_OP_NONE) {
                    m_ڳنڍ_dynamic_dims[ڳنڍ] = m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]];
                    اكسر;
                }
                خودڪار dynamic_dim_idx = m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]];
                خودڪار dynamic_dim_قدر = ڳنڍ->src[0]->ne[dynamic_dim_idx];
                خودڪار dynamic_dim_stride =
                    ڳنڍ->src[0]->nb[dynamic_dim_idx] / ggml_type_سائيز(ڳنڍ->src[0]->type) * ggml_type_سائيز(ڳنڍ->type);
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (ڳنڍ->nb[i] == dynamic_dim_stride) {
                        m_ڳنڍ_dynamic_dims[ڳنڍ] = i;
                        اكسر;
                    }
                }
                إذا (m_ڳنڍ_dynamic_dims[ڳنڍ] != -1 && dynamic_dim_قدر != ڳنڍ->ne[m_ڳنڍ_dynamic_dims[ڳنڍ]]) {
                    m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    إذا (ggml_nelements(ڳنڍ) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      ڳنڍ->name, ڳنڍ->src[0]->name);
                    }
                }
            }
            اكسر;
        }
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RESHAPE: {
            إذا (is_same_shape(ڳنڍ->src[0], ڳنڍ)) {
                m_ڳنڍ_dynamic_dims[ڳنڍ] = m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]];
                اكسر;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            إذا (m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]] != -1) {
                خودڪار dynamic_dim_idx = m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]];
                خودڪار dynamic_dim_stride = ڳنڍ->src[0]->nb[dynamic_dim_idx];
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (ڳنڍ->nb[i] == dynamic_dim_stride && ڳنڍ->ne[i] == ڳنڍ->src[0]->ne[dynamic_dim_idx]) {
                        m_ڳنڍ_dynamic_dims[ڳنڍ] = i;
                        اكسر;
                    }
                }
                إذا (m_ڳنڍ_dynamic_dims[ڳنڍ] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", ڳنڍ->name);
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
            constexpr صحيح q_منزل_ٻاهريون[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            إذا (m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]] != -1) {
                خودڪار q_dynamic_dim = m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]];
                m_ڳنڍ_dynamic_dims[ڳنڍ] = q_منزل_ٻاهريون[q_dynamic_dim];
            }
            اكسر;
        }
        حالة GGML_OP_CONT:
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            إذا (m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]] != -1) {
                خودڪار dynamic_dim_idx = m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]];
                إذا (ggml_are_same_shape(ڳنڍ, ڳنڍ->src[0])) {
                    m_ڳنڍ_dynamic_dims[ڳنڍ] = dynamic_dim_idx;
                } وإلا {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_سائيز(ڳنڍ->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (ڳنڍ->src[0]->ne[0] / ggml_blck_سائيز(ڳنڍ->src[0]->type));
                    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * ڳنڍ->src[0]->ne[i - 1];
                    }

                    خودڪار dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_سائيز(ڳنڍ->src[0]->type) *
                                              ggml_type_سائيز(ڳنڍ->type);
                    صحيح matched_dim_ڳڻپ = 0;
                    صحيح پهريون_matched_dim = -1;
                    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                        إذا (ڳنڍ->nb[i] == dynamic_dim_stride && ڳنڍ->ne[i] == ڳنڍ->src[0]->ne[dynamic_dim_idx]) {
                            إذا (پهريون_matched_dim == -1) {
                                پهريون_matched_dim = i;
                            }
                            m_ڳنڍ_dynamic_dims[ڳنڍ] = i;
                            matched_dim_ڳڻپ++;
                        }
                    }
                    إذا (matched_dim_ڳڻپ > 1 && ڳنڍ->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_ڳنڍ_dynamic_dims[ڳنڍ] = پهريون_matched_dim;
                    } وإلا إذا (matched_dim_ڳڻپ != 1) {
                        m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      ڳنڍ->name, ڳنڍ->src[0]->name);
                    }
                }
            }
            اكسر;
        حالة GGML_OP_CONCAT:
            لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                إذا (ڳنڍ->src[0]->ne[i] != ڳنڍ->ne[i]) {
                    m_ڳنڍ_dynamic_dims[ڳنڍ] = i;
                    اكسر;
                }
            }
            اكسر;
        حالة GGML_OP_SSM_CONV:
        حالة GGML_OP_GATED_DELTA_NET:
            m_ڳنڍ_dynamic_dims[ڳنڍ] = 1;
            اكسر;
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_L2_NORM:
        حالة GGML_OP_NORM:
        حالة GGML_OP_جوڙيو:
        حالة GGML_OP_SUB:
        حالة GGML_OP_GLU:
        حالة GGML_OP_ROPE:
        حالة GGML_OP_SCALE:
        حالة GGML_OP_SOFT_MAX:
        حالة GGML_OP_ARGSORT:
        حالة GGML_OP_جوڙيو_ID:
        حالة GGML_OP_UNARY:
        حالة GGML_OP_CUMSUM:
        حالة GGML_OP_FILL:
        حالة GGML_OP_سيٽ:
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
            m_ڳنڍ_dynamic_dims[ڳنڍ] = m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]];
            اكسر;
        حالة GGML_OP_جوڙ_صفون:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_ڳنڍ_dynamic_dims[ڳنڍ] =
                (m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]] == 0) ? -1 : m_ڳنڍ_dynamic_dims[ڳنڍ->src[0]];
            اكسر;
        حالة GGML_OP_MUL_MAT_ID:
        حالة GGML_OP_SOLVE_TRI:
            m_ڳنڍ_dynamic_dims[ڳنڍ] = m_ڳنڍ_dynamic_dims[ڳنڍ->src[1]];
            اكسر;
        حالة GGML_OP_CPY:
        حالة GGML_OP_سيٽ_صفون:
        حالة GGML_OP_جوڙ:
        حالة GGML_OP_سراسري:
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            اكسر;
        حالة GGML_OP_IM2_کانو: {
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            إذا (m_ڳنڍ_dynamic_dims[ڳنڍ->src[1]] != -1) {
                ثابت منطقي is_2D = ڳنڍ->op_params[6] == 1;
                ثابت صحيح src_dyn = m_ڳنڍ_dynamic_dims[ڳنڍ->src[1]];
                إذا (is_2D) {
                    إذا (src_dyn == 0) {
                        m_ڳنڍ_dynamic_dims[ڳنڍ] = 1;  // IW -> OW
                    } وإلا إذا (src_dyn == 1) {
                        m_ڳنڍ_dynamic_dims[ڳنڍ] = 2;  // IH -> OH
                    } وإلا إذا (src_dyn == 3) {
                        m_ڳنڍ_dynamic_dims[ڳنڍ] = 3;  // N  -> N
                    }
                } وإلا {
                    إذا (src_dyn == 0) {
                        m_ڳنڍ_dynamic_dims[ڳنڍ] = 1;  // IW -> OW
                    } وإلا إذا (src_dyn == 2) {
                        m_ڳنڍ_dynamic_dims[ڳنڍ] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                إذا (m_ڳنڍ_dynamic_dims[ڳنڍ] != -1) {
                    OPENVINO_ASSERT(ڳنڍ->src[1]->ne[src_dyn] == ڳنڍ->ne[m_ڳنڍ_dynamic_dims[ڳنڍ]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(ڳنڍ->name) +
                                        " and its src[1]: " + std::string(ڳنڍ->src[1]->name));
                }
            }
            اكسر;
        }
        حالة GGML_OP_IM2_کانو_3_D: {
            m_ڳنڍ_dynamic_dims[ڳنڍ] = -1;
            إذا (m_ڳنڍ_dynamic_dims[ڳنڍ->src[1]] != -1) {
                ثابت صحيح src_dyn = m_ڳنڍ_dynamic_dims[ڳنڍ->src[1]];
                إذا (src_dyn == 0) {
                    m_ڳنڍ_dynamic_dims[ڳنڍ] = 1;  // IW -> OW
                } وإلا إذا (src_dyn == 1) {
                    m_ڳنڍ_dynamic_dims[ڳنڍ] = 2;  // IH -> OH
                } وإلا إذا (src_dyn == 3) {
                    m_ڳنڍ_dynamic_dims[ڳنڍ] = 3;  // N  -> N
                }
                إذا (m_ڳنڍ_dynamic_dims[ڳنڍ] != -1) {
                    OPENVINO_ASSERT(ڳنڍ->src[1]->ne[src_dyn] == ڳنڍ->ne[m_ڳنڍ_dynamic_dims[ڳنڍ]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(ڳنڍ->name) +
                                        " and its src[1]: " + std::string(ڳنڍ->src[1]->name));
                }
            }
            اكسر;
        }
        افتراضي:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(ڳنڍ->op), ڳنڍ->name);
            اكسر;
        }
    };

    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ڳنڍ = m_cgraph->nodes[i];
        گهمو_ڳنڍ(گهمو_ڳنڍ, ڳنڍ);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    إذا (0) {
        لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * ڳنڍ = m_cgraph->nodes[i];
            صحيح dynamic_dim = m_ڳنڍ_dynamic_dims[ڳنڍ];
            std::ٻاهر << "[" << i << "] " << "node_name: " << ڳنڍ->name << " op: " << ggml_op_name(ڳنڍ->op)
                      << " shape: [";
            لكل (صحيح j = 0; j < 4; j++) {
                إذا (j == dynamic_dim) {
                    std::ٻاهر << "*";
                } وإلا {
                    std::ٻاهر << ڳنڍ->ne[j];
                }
                إذا (j < 3) {
                    std::ٻاهر << ", ";
                }
            }
            std::ٻاهر << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = ڳنڍ->src[j];
                إذا (src == خالي_اشارو) {
                    استمر;
                }
                صحيح src_dynamic_dim = m_ڳنڍ_dynamic_dims[src];
                std::ٻاهر << "    [" << j << "] src_name: " << src->name << " [";
                لكل (صحيح k = 0; k < 4; k++) {
                    إذا (k == src_dynamic_dim) {
                        std::ٻاهر << "*";
                    } وإلا {
                        std::ٻاهر << src->ne[k];
                    }
                    إذا (k < 3) {
                        std::ٻاهر << ", ";
                    }
                }
                std::ٻاهر << "]" << '\n';
            }
            std::ٻاهر << '\n';
        }
    }
}
