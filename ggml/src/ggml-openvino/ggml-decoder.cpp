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
                             صحيح prefill_chunk_ناپ) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(خطأ),
    m_prefill_chunk_ناپ(prefill_chunk_ناپ),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    ساكن منطقي printed_پتہ_map = خطأ;
    إذا (!printed_پتہ_map) {
        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_پتہ_map = سچ;
            print_tensor_پتہ_map(cgraph);
        }
    }

    validate_cgraph();

    سیٹ_داخلہ_output();
    compute_گرہ_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    لكل (صحيح گرہ_n = 0; گرہ_n < cgraph->n_nodes; گرہ_n++) {
        m_گرہ_info_list[گرہ_n].گرہ_op_case = compute_op_case(m_گرہ_info_list[گرہ_n].گرہ);
        m_گرہ_info_list[گرہ_n].گرہ_op_type = compute_op_type(m_گرہ_info_list[گرہ_n].گرہ);
    }

    جوڑو_extra_inputs();
}

فراغ GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_گرہ_info_list.clear();
    سیٹ_داخلہ_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
منطقي GgmlOvDecoder::is_bound_منزل(ثابت ggml_cgraph * cgraph) ثابت {
    أرجع m_cgraph == cgraph && cgraph->n_nodes > 0 && m_گرہ_info_list.size() == (size_t) cgraph->n_nodes &&
           m_گرہ_info_list.front().گرہ == cgraph->nodes[0] &&
           m_گرہ_info_list.back().گرہ == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = سچ;
    سیٹ_داخلہ_output();
    compute_model_inputs();
    compute_model_outputs();
    لكل (صحيح گرہ_n = 0; گرہ_n < cgraph->n_nodes; گرہ_n++) {
        m_گرہ_info_list[گرہ_n].گرہ_op_case = compute_op_case(m_گرہ_info_list[گرہ_n].گرہ);
        m_گرہ_info_list[گرہ_n].گرہ_op_type = compute_op_type(m_گرہ_info_list[گرہ_n].گرہ);
    }
}

نطاق {
منطقي is_inplace_op(ثابت ggml_tensor * گرہ) {
    أرجع گرہ->op == GGML_OP_سیٹ_صفاں || گرہ->op == GGML_OP_CPY || (گرہ->op == GGML_OP_SCALE && گرہ->view_src);
}

منطقي is_same_shape(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        إذا (a->ne[i] != b->ne[i]) {
            أرجع خطأ;
        }
    }
    أرجع سچ;
}

منطقي is_conv_حالتاں_all_tensor(ثابت ggml_tensor * tensor) {
    أرجع tensor != خالی_سوچک && لڑی_مقابلہ(tensor->name, "conv_states_all", لڑی_لمبائی("conv_states_all")) == 0;
}

منطقي is_بھریا_single_تھاں_writeback(ثابت ggml_tensor * گرہ) {
    أرجع گرہ->view_src != خالی_سوچک && گرہ->view_src->ne[1] == 1 && گرہ->src[1] != خالی_سوچک &&
           گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src == گرہ->view_src &&
           گرہ->src[1]->view_offs == 0 && ggml_nbytes(گرہ->src[1]) == ggml_nbytes(گرہ->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
منطقي is_moe_expert_جوڑ_جوڑو(ثابت ggml_tensor * گرہ) {
    std::vector<ثابت ggml_tensor *> leaves;
    ثابت ggml_tensor * cur = گرہ;
    طالما (cur->op == GGML_OP_جوڑو) {
        إذا (cur->src[0] == خالی_سوچک || cur->src[1] == خالی_سوچک) {
            أرجع خطأ;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ثابت ggml_tensor * ادھار = خالی_سوچک;
    std::set<int64_t> plane_indices;
    لكل (ثابت ggml_tensor * پتی : leaves) {
        إذا (پتی->op != GGML_OP_VIEW || پتی->src[0] == خالی_سوچک) {
            أرجع خطأ;
        }
        ثابت ggml_tensor * پتی_ادھار = پتی->src[0];
        إذا (ادھار == خالی_سوچک) {
            ادھار = پتی_ادھار;
        } وإلا إذا (پتی_ادھار != ادھار) {
            أرجع خطأ;
        }
        إذا (پتی->ne[0] != ادھار->ne[0] || پتی->ne[1] != ادھار->ne[2] || پتی->ne[2] != 1 || پتی->ne[3] != 1 ||
            پتی->nb[1] != ادھار->nb[2]) {
            أرجع خطأ;
        }
        إذا (ادھار->nb[1] == 0 || پتی->view_offs % ادھار->nb[1] != 0) {
            أرجع خطأ;
        }
        int64_t plane = static_cast<int64_t>(پتی->view_offs / ادھار->nb[1]);
        إذا (plane < 0 || plane >= ادھار->ne[1] || !plane_indices.insert(plane).second) {
            أرجع خطأ;
        }
    }

    أرجع ادھار != خالی_سوچک && ادھار->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(ادھار->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor) {
    إذا (tensor == خالی_سوچک) {
        أرجع "";
    }
    إذا ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, خالی_سوچک)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ثابت خودکار * گرہ = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        إذا (گرہ != cgraph->nodes + cgraph->n_nodes) {
            أرجع std::string(tensor->name) + "#n" + std::to_string(گرہ - cgraph->nodes);
        }
        ثابت خودکار * پتی = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        إذا (پتی != cgraph->leafs + cgraph->n_leafs) {
            أرجع std::string(tensor->name) + "#l" + std::to_string(پتی - cgraph->leafs);
        }
    }
    أرجع tensor->name;
}

ساكن std::string get_tensor_ov_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor) {
    أرجع GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

ساكن std::string get_tensor_گراف_داخلہ_ov_name(ثابت GgmlOvDecoder * decoder,
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

فراغ GgmlOvDecoder::سیٹ_داخلہ_output() {
    لكل (صحيح گرہ_n = 0; گرہ_n < m_cgraph->n_nodes; گرہ_n++) {
        خودکار * گرہ = m_cgraph->nodes[گرہ_n];

        گرہ_Info current_گرہ_info;
        خودکار گرہ_name = get_tensor_ov_name(m_cgraph, گرہ);

        current_گرہ_info.گرہ = گرہ;
        current_گرہ_info.گرہ_name = گرہ_name;
        current_گرہ_info.گرہ_op_case = 0;
        current_گرہ_info.انکڑے_addr = گرہ->انکڑے;

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            خودکار * src = گرہ->src[i];
            إذا (src == خالی_سوچک) {
                استمر;
            }
            خودکار src_name = get_tensor_ov_name(m_cgraph, src);
            إذا (src->flags & GGML_TENSOR_FLAG_داخلہ) {
                src_name = get_tensor_گراف_داخلہ_ov_name(هذا, m_cgraph, src, گرہ);
            }
            current_گرہ_info.گرہ_inputs[src_name] = src;
            current_گرہ_info.گرہ_inputs_ناں.push_back(src_name);

            إذا (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_گرہ_info.گرہ_inputs_views[src_name])> view_chain;
                خودکار * current = src;

                طالما (current != خالی_سوچک) {
                    خودکار current_name = get_tensor_ov_name(m_cgraph, current);
                    إذا (current->flags & GGML_TENSOR_FLAG_داخلہ) {
                        current_name = get_tensor_گراف_داخلہ_ov_name(هذا, m_cgraph, current, گرہ);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    إذا (current->src[0] != خالی_سوچک && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } وإلا {
                        اكسر;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_گرہ_info.گرہ_inputs_views[src_name] = view_chain;
            }
        }

        m_گرہ_info_list.push_back(current_گرہ_info);
    }
}

صحيح GgmlOvDecoder::compute_op_case(ثابت ggml_tensor * گرہ) ثابت {
    صحيح op_case = 0;
    بدل (گرہ->op) {
    حالة GGML_OP_RESHAPE: {
        إذا (m_naive) {
            اكسر;
        }
        خودکار name = std::string(گرہ->name);
        خودکار * src = گرہ->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        منطقي recurrent_sequence = خطأ;
        لكل (صحيح i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ثابت خودکار * کھپتکار = m_cgraph->nodes[i];
            إذا (کھپتکار->op == GGML_OP_MUL_MAT_ID && کھپتکار->src[1] == گرہ) {
                أرجع 1;
            } وإلا إذا (کھپتکار->op == GGML_OP_SSM_CONV) {
                ثابت خودکار * concat = کھپتکار->src[0];
                إذا (concat->op == GGML_OP_CONCAT) {
                    ثابت خودکار * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == گرہ;
                }
            } وإلا إذا (کھپتکار->op == GGML_OP_UNARY && ggml_get_unary_op(کھپتکار) == GGML_UNARY_OP_SOFTPLUS) {
                ثابت خودکار * biased = کھپتکار->src[0];
                recurrent_sequence = biased->op == GGML_OP_جوڑو && biased->src[0] == گرہ;
            }
        }
        إذا (recurrent_sequence && گرہ->ne[0] == src->ne[0] && گرہ->ne[3] == 1) {
            أرجع 6;
        }
        إذا (گرہ->ne[0] == src->ne[0] && گرہ->ne[2] == 1 && گرہ->ne[3] == 1) {
            أرجع 5;
        }
        إذا (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == گرہ->ne[0] && src->src[0]->ne[1] == گرہ->ne[1]) {
            op_case = 4;
        } وإلا إذا (گرہ->ne[0] * گرہ->ne[1] == src->ne[0]) {
            op_case = 1;
        } وإلا إذا (src->ne[0] * src->ne[1] == گرہ->ne[0]) {
            op_case = 2;
            إذا (src->ne[2] * src->ne[3] == گرہ->ne[1]) {
                op_case = 5;
            }
        } وإلا إذا (گرہ->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == گرہ->ne[1]) {
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
            ثابت خودکار * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            منطقي recurrent = gather->op == GGML_OP_GET_صفاں && is_recurrent_cache(gather->src[0]);
            لكل (صحيح i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ثابت خودکار * کھپتکار = m_cgraph->nodes[i];
                إذا (کھپتکار->op == GGML_OP_GATED_DELTA_NET) {
                    لكل (صحيح j : {3, 4}) {
                        ثابت خودکار * gate = کھپتکار->src[j];
                        إذا (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == گرہ;
                    }
                } وإلا إذا (کھپتکار->op == GGML_OP_MUL) {
                    لكل (صحيح j = 0; j < 2; ++j) {
                        ثابت خودکار * gate = کھپتکار->src[j];
                        ثابت خودکار * norm = کھپتکار->src[1 - j];
                        إذا (gate->op != GGML_OP_UNARY || gate->src[0] != گرہ) {
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
    حالة GGML_OP_ترتیب_بدلو: {
        إذا (گرہ->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } وإلا إذا (گرہ->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(گرہ->view_src->name);
            صحيح layer = کڈھو_layer_from_name(src_name).value();
            إذا (ggml_is_contiguous(گرہ->src[0])) {
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
        إذا (گرہ->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        اكسر;
    }
    حالة GGML_OP_GET_صفاں: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        إذا (گرہ->src[0]->op == GGML_OP_RESHAPE && گرہ->src[0]->src[0] != خالی_سوچک &&
            is_recurrent_cache(گرہ->src[0]->src[0])) {
            ثابت منطقي single_تھاں = گرہ->src[0]->src[0]->ne[1] == 1;
            إذا (گرہ->src[1]->op == GGML_OP_VIEW) {
                op_case = (گرہ->src[1]->view_offs == 0 ? 1 : 2) + (single_تھاں ? 2 : 0);
            } وإلا إذا (single_تھاں) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        اكسر;
    }
    حالة GGML_OP_ROPE: {
        ثابت صحيح mode = گرہ->op_params[2];
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
        إذا (!m_model_params.has_rs_rollback && گرہ->src[0] != خالی_سوچک &&
            گرہ->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            اكسر;
        }
        إذا (m_is_static && گرہ->src[0] != خالی_سوچک &&
            (گرہ->src[0]->op == GGML_OP_GATED_DELTA_NET || گرہ->src[0]->op == GGML_OP_CONCAT)) {
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
        إذا (گرہ->src[0]->op == GGML_OP_VIEW) {
            خودکار * src = گرہ->src[0];
            إذا (ggml_nelements(گرہ) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            إذا (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            خودکار * src = گرہ->src[0];
            إذا (ggml_nelements(گرہ) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                إذا (گرہ->ne[0] == src->ne[0] && گرہ->ne[1] == src->ne[2] && گرہ->ne[2] == 1 &&
                    گرہ->nb[0] == src->nb[0] && گرہ->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    اكسر;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                صحيح diff_گنتی = 0;
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (گرہ->ne[i] != src->ne[i]) {
                        diff_گنتی++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    إذا (گرہ->ne[i] > src->ne[i]) {
                        أرجع 0;
                    }
                }
                إذا (diff_گنتی >= 1) {
                    op_case = 0;
                }
            }
        }
        اكسر;
    }
    حالة GGML_OP_RMS_NORM: {
        إذا (گرہ->src[0]->op == GGML_OP_VIEW) {
            إذا (is_same_shape(گرہ->src[0]->src[0], گرہ->src[0])) {
                op_case = 1;
            } وإلا إذا (!m_model_params.has_rs_rollback &&
                       گرہ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } وإلا إذا (گرہ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        اكسر;
    }
    حالة GGML_OP_POOL_2D: {
        ثابت ggml_op_pool pool_منوال = static_cast<ggml_op_pool>(گرہ->op_params[0]);
        بدل (pool_منوال) {
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
        ثابت int32_t منوال_flags = گرہ->op_params[0];
        ثابت ggml_scale_منوال scale_منوال = static_cast<ggml_scale_منوال>(منوال_flags & 0xFF);
        بدل (scale_منوال) {
        حالة GGML_SCALE_منوال_سبھ_توں_نیڑے: {
            op_case = 1;
            اكسر;
        }
        حالة GGML_SCALE_منوال_BILINEAR: {
            op_case = 2;
            اكسر;
        }
        حالة GGML_SCALE_منوال_BICUBIC: {
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
        إذا (گرہ->src[0]->op == GGML_OP_VIEW) {
            إذا (گرہ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                إذا (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_بھریا_single_تھاں_writeback(گرہ) ? 7 : 10;
                } وإلا {
                    op_case = 1;
                }
            } وإلا إذا (GgmlOvDecoder::is_conv_state_writeback(گرہ)) {
                op_case = is_بھریا_single_تھاں_writeback(گرہ) ? 8 : 2;
                اكسر;
            } وإلا إذا (is_conv_حالتاں_all_tensor(گرہ->view_src) && گرہ->src[1] != خالی_سوچک &&
                       گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src == گرہ->view_src) {
                op_case = 4;
                اكسر;
            } وإلا إذا (گرہ->src[0]->src[0]->op == GGML_OP_GET_صفاں && گرہ->src[1] != خالی_سوچک &&
                       گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src != خالی_سوچک &&
                       is_recurrent_cache(گرہ->src[1]->view_src) && گرہ->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } وإلا إذا (گرہ->src[0]->op == GGML_OP_GET_صفاں && گرہ->src[1] != خالی_سوچک &&
                   گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src != خالی_سوچک &&
                   is_recurrent_cache(گرہ->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = گرہ->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } وإلا إذا (گرہ->src[1] != خالی_سوچک && گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src != خالی_سوچک) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ثابت ggml_tensor * kv_buf = گرہ->src[1]->view_src;
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
                    إذا ((n->src[1] != خالی_سوچک && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != خالی_سوچک && n->src[2]->view_src == kv_buf)) {
                        إذا (n->src[3] != خالی_سوچک) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        اكسر;
                    }
                }
            }
        }
        اكسر;
    }
    حالة GGML_OP_جوڑو: {
        إذا (is_moe_expert_جوڑ_جوڑو(گرہ)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        اكسر;
    }
    حالة GGML_OP_SCALE: {
        إذا (گرہ->view_src && گرہ->بفر->usage == GGML_BACKEND_بفر_USAGE_ANY) {
            op_case = گرہ->view_src->ne[1] == 1 ? 2 : 1;
        }
        اكسر;
    }
    حالة GGML_OP_L2_NORM: {
        إذا (std::string(گرہ->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        اكسر;
    }
    حالة GGML_OP_FLASH_ATTN_EXT: {
        إذا (گرہ->src[1] != خالی_سوچک && گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src != خالی_سوچک) {
            ثابت ggml_tensor * kv_buf = گرہ->src[1]->view_src;
            إذا (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (گرہ->src[3] != خالی_سوچک) ? 1 : 2;
            }
        }
        اكسر;
    }
    افتراضي:
        اكسر;
    }
    أرجع op_case;
}

std::optional<صحيح> کڈھو_layer_from_name(ثابت std::string & name) {
    size_t pos1 = name.find("_l");
    إذا (pos1 == std::string::npos) {
        أرجع std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    إذا (pos2 == std::string::npos) {
        pos2 = name.لمائی();
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
ساكن صحيح get_swa_کھڑکی_from_نقاب(ثابت ggml_tensor * نقاب) {
    إذا (نقاب->انکڑے == خالی_سوچک || !ggml_backend_بفر_is_host(نقاب->بفر)) {
        أرجع -1;
    }
    إذا (نقاب->type != GGML_TYPE_F16 && نقاب->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت int64_t n_kv = نقاب->ne[0];
    ثابت int64_t n_ٹوکن = نقاب->ne[1];
    int64_t کھڑکی = 0;

    لكل (int64_t r = 0; r < n_ٹوکن; r++) {
        int64_t kept = 0;
        لكل (int64_t c = 0; c < n_kv; c++) {
            ثابت size_t i = (size_t) r * n_kv + c;
            ثابت عائم v = نقاب->type == GGML_TYPE_F16 ? ggml_fp16_منزل_fp32(((ثابت ggml_fp16_t *) نقاب->انکڑے)[i]) :
                                                          ((ثابت عائم *) نقاب->انکڑے)[i];
            إذا (v > -INFINITY) {
                kept++;
            }
        }
        کھڑکی = std::max(کھڑکی, kept);
    }

    أرجع کھڑکی > 0 ? (صحيح) کھڑکی : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, منطقي is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    خودکار get_attention_نمونہ_case = [](ثابت ggml_tensor * گرہ) -> صحيح {
        إذا (گرہ == خالی_سوچک) {
            أرجع -1;
        }

        بدل (گرہ->op) {
        حالة GGML_OP_FLASH_ATTN_EXT:
            إذا (گرہ->src[0] == خالی_سوچک || گرہ->src[1] == خالی_سوچک) {
                أرجع -1;
            }
            بدل (گرہ->src[1]->op) {
            حالة GGML_OP_ترتیب_بدلو:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                إذا (گرہ->src[3] != خالی_سوچک && گرہ->src[1]->src[0] != خالی_سوچک &&
                    گرہ->src[1]->src[0]->op == GGML_OP_VIEW) {
                    أرجع 0;
                }
                اكسر;
            حالة GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                إذا (گرہ->src[3] != خالی_سوچک && گرہ->src[1]->src[0] != خالی_سوچک &&
                    گرہ->src[1]->src[0]->op == GGML_OP_ترتیب_بدلو && گرہ->src[1]->src[0]->src[0] != خالی_سوچک &&
                    گرہ->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    أرجع 1;
                }
                اكسر;
            حالة GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                إذا (گرہ->src[1]->view_src != خالی_سوچک) {
                    إذا (گرہ->src[3] != خالی_سوچک) {
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
            إذا (گرہ->src[0] != خالی_سوچک && گرہ->src[1] != خالی_سوچک && گرہ->src[0]->op == GGML_OP_MUL_MAT &&
                گرہ->src[0]->src[0] != خالی_سوچک && گرہ->src[0]->src[1] != خالی_سوچک &&
                گرہ->src[0]->src[0]->op == GGML_OP_ترتیب_بدلو && گرہ->src[0]->src[0]->src[0] != خالی_سوچک &&
                گرہ->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                أرجع 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            إذا (گرہ->src[0]->op == GGML_OP_جوڑو && گرہ->src[0]->src[0] != خالی_سوچک &&
                گرہ->src[0]->src[0]->op == GGML_OP_MUL_MAT && گرہ->src[0]->src[0]->src[0] != خالی_سوچک &&
                گرہ->src[0]->src[0]->src[0]->op == GGML_OP_ترتیب_بدلو) {
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
    خودکار get_attention_op_نقاب = [&get_attention_نمونہ_case](ثابت ggml_tensor * گرہ) -> ثابت ggml_tensor * {
        بدل (get_attention_نمونہ_case(گرہ)) {
        حالة 0:
        حالة 1:
            أرجع گرہ->src[3];
        حالة 2:
        حالة 3:
            أرجع گرہ->src[1];
        افتراضي:
            أرجع خالی_سوچک;
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
            إذا (نقاب == خالی_سوچک) {
                استمر;
            }
            ثابت ggml_tensor * cache_k_ترتیب_بدلو = خالی_سوچک;
            بدل (get_attention_نمونہ_case(cgraph->nodes[i])) {
            حالة 0:  cache_k_ترتیب_بدلو = cgraph->nodes[i]->src[1];                     اكسر;
            حالة 1:  cache_k_ترتیب_بدلو = cgraph->nodes[i]->src[1]->src[0];             اكسر;
            حالة 2:  cache_k_ترتیب_بدلو = cgraph->nodes[i]->src[0]->src[0];             اكسر;
            افتراضي: cache_k_ترتیب_بدلو = cgraph->nodes[i]->src[0]->src[0]->src[0];     اكسر;
            }
            ثابت ggml_tensor * cache_k_view = cache_k_ترتیب_بدلو->src[0];
            إذا (cache_k_view->op != GGML_OP_VIEW) {
                استمر;
            }
            ثابت ggml_tensor * پتی = cache_k_view->src[0];
            خودکار layer = کڈھو_layer_from_name(پتی->name);
            إذا (!layer.has_value()) {
                استمر;
            }
            layer_extent[layer.value()] = پتی->ne[1];
            layer_نقاب[layer.value()] = نقاب;
            max_extent = std::max(max_extent, پتی->ne[1]);
        }

        لكل (ثابت خودکار & [layer, extent] : layer_extent) {
            إذا (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                إذا (model_params.swa_نقاب == خالی_سوچک) {
                    model_params.swa_نقاب = layer_نقاب[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string فی_layer;
            لكل (ثابت خودکار & [layer, extent] : layer_extent) {
                فی_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (لما) max_extent, model_params.swa_layers.size(), فی_layer.c_str());
        }
    }

    منطقي rope_تکیا = خطأ;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = cgraph->nodes[i];
        ثابت صحيح attention_نمونہ_case = get_attention_نمونہ_case(گرہ);
        إذا (attention_نمونہ_case != -1) {
            ggml_tensor * cache_k_ترتیب_بدلو = خالی_سوچک;
            ggml_tensor * نقاب = خالی_سوچک;

            بدل (attention_نمونہ_case) {
            حالة 0:
                cache_k_ترتیب_بدلو = گرہ->src[1];
                نقاب = گرہ->src[3];
                اكسر;
            حالة 1:
                cache_k_ترتیب_بدلو = گرہ->src[1]->src[0];
                نقاب = گرہ->src[3];
                اكسر;
            حالة 2:
                cache_k_ترتیب_بدلو = گرہ->src[0]->src[0];
                نقاب = گرہ->src[1];
                اكسر;
            حالة 3:
                cache_k_ترتیب_بدلو = گرہ->src[0]->src[0]->src[0];
                نقاب = گرہ->src[1];
                اكسر;
            حالة 4:
            حالة 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                خودکار * cache_k_view = گرہ->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_فی_seq = گرہ->src[0]->ne[1];
                إذا (attention_نمونہ_case == 4) {
                    compute_params.attention_ناپ = cache_k_view->ne[1];
                } وإلا {
                    compute_params.attention_ناپ_static = cache_k_view->ne[1];
                }
                استمر;
            }
            افتراضي:
                اكسر;
            }

            أكد(cache_k_ترتیب_بدلو != خالی_سوچک);

            model_params.سر_ناپ = cache_k_ترتیب_بدلو->ne[0];
            model_params.n_heads_kv = cache_k_ترتیب_بدلو->ne[2];
            compute_params.داخلہ_len = گرہ->src[0]->ne[1];
            compute_params.token_len_فی_seq = گرہ->src[0]->ne[1];

            خودکار * cache_k_view = cache_k_ترتیب_بدلو->src[0];
            إذا (cache_k_view->op != GGML_OP_VIEW || نقاب == خالی_سوچک) {
                استمر;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            صحيح layer = کڈھو_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ثابت منطقي layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_بفر_ctx_id = ggml_backend_openvino_بفر_get_ctx_id(cache_k->بفر);
            model_params.n_heads_kv_فی_layer[layer] = cache_k_ترتیب_بدلو->ne[2];
            إذا (layer_is_swa) {
                model_params.ctx_فی_seq_swa = cache_k->ne[1];
            } وإلا {
                model_params.ctx_فی_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_سرگرم = نقاب->ne[3];
            خودکار seq_ناپ = cache_k->ne[0] * cache_k->ne[1] * ggml_type_ناپ(cache_k->type);
            size_t وتھ;
            حافظہ_نقل(&وتھ, cache_k_view->op_params, حجم(size_t));
            compute_params.seq_سرگرم_شروع = وتھ / seq_ناپ;

            إذا (layer_is_swa) {
                compute_params.attention_ناپ_swa = نقاب->ne[0];
                compute_params.swa_کھڑکی = get_swa_کھڑکی_from_نقاب(نقاب);
            } وإلا {
                compute_params.attention_ناپ = نقاب->ne[0];
            }
            إذا (is_static) {
                compute_params.attention_ناپ = model_params.ctx_فی_seq;
                compute_params.attention_ناپ_swa = model_params.ctx_فی_seq_swa;
                compute_params.token_len_فی_seq = 1;
            }
        }

        إذا (گرہ->op == GGML_OP_MUL_MAT && گرہ->src[0]->op == GGML_OP_ترتیب_بدلو &&
            گرہ->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(گرہ->src[0]->view_src, گرہ->view_src)) {
            إذا (گرہ->src[1]->op == GGML_OP_ترتیب_بدلو && گرہ->src[1]->src[0]->op == GGML_OP_VIEW &&
                گرہ->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_ناپ = گرہ->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        إذا (گرہ->op == GGML_OP_TRANSPOSE && گرہ->src[0]->op == GGML_OP_ترتیب_بدلو &&
            گرہ->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_ناپ = گرہ->ne[0];
            إذا (is_static) {
                compute_params.attention_ناپ = model_params.ctx_فی_seq;
            }
        }
        إذا (گرہ->op == GGML_OP_ROPE) {
            إذا (compute_params.token_len_فی_seq == -1 && گرہ->src[1] != خالی_سوچک) {
                compute_params.token_len_فی_seq = ggml_nelements(گرہ->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(حجم(model_params.rope_params) == حجم(int32_t) * 16, "rope_params size");
            إذا (!rope_تکیا) {
                حافظہ_نقل(model_params.rope_params, گرہ->op_params, حجم(int32_t) * 16);
                rope_تکیا = سچ;
            } وإلا إذا (memcmp(model_params.rope_params, گرہ->op_params, حجم(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = سچ;
            }
        }
        إذا (گرہ->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_ناپ = گرہ->src[0]->ne[0];
        }
        إذا (گرہ->op == GGML_OP_SCALE && گرہ->view_src != خالی_سوچک && is_recurrent_cache(گرہ->view_src)) {
            إذا (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = گرہ->view_src->ne[1];
            } وإلا {
                GGML_ASSERT(model_params.n_rs_slots == گرہ->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(گرہ) / گرہ->view_src->ne[0];
            compute_params.cache_rs_reset_idx = گرہ->src[0]->view_offs / گرہ->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        إذا (گرہ->op == GGML_OP_CPY && گرہ->view_src != خالی_سوچک && is_recurrent_cache(گرہ->view_src) &&
            گرہ->src[1] != خالی_سوچک && گرہ->src[1]->op == GGML_OP_VIEW && گرہ->src[1]->view_src == گرہ->view_src) {
            ثابت منطقي is_conv = is_conv_state_writeback(گرہ);
            ثابت منطقي is_gdn = گرہ->src[0]->op == GGML_OP_VIEW && گرہ->src[0]->src[0] != خالی_سوچک &&
                                گرہ->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ثابت منطقي is_extra = گرہ->src[0]->op == GGML_OP_GET_صفاں;
            ثابت منطقي is_gdn_rollback = is_gdn && is_same_shape(گرہ->src[0], گرہ->src[1]);

            ثابت ggml_tensor * dest_view = گرہ->src[1];
            ثابت ggml_tensor * cache = گرہ->view_src;
            ثابت size_t صف_بائٹاں = cache->ne[0] * ggml_type_ناپ(cache->type);
            إذا (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = سچ;
            }
            إذا (صف_بائٹاں > 0 && (is_conv || is_gdn || is_extra) && !is_بھریا_single_تھاں_writeback(گرہ)) {
                ComputeParams::RsWriteback writeback;
                writeback.تھاں_begin = (صحيح) (dest_view->view_offs / صف_بائٹاں);
                إذا (is_conv) {
                    writeback.src_begin = (صحيح) (گرہ->src[0]->view_offs / گرہ->src[0]->view_src->nb[0]);
                } وإلا إذا (is_gdn_rollback) {
                    writeback.src_begin = (صحيح) (گرہ->src[0]->view_offs / گرہ->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, گرہ)] = writeback;
            }
            إذا ((is_conv || is_gdn) && !is_بھریا_single_تھاں_writeback(گرہ)) {
                compute_params.s_copy_سرگرم_تھاں_len = (صحيح) dest_view->ne[1];
            }
        }
    }
    إذا (model_params.n_heads_kv == -1) {
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ثابت خودکار * گرہ = cgraph->nodes[i];
            ثابت ggml_tensor * نقاب = خالی_سوچک;
            إذا (گرہ->op == GGML_OP_SOFT_MAX) {
                نقاب = گرہ->src[1];
            } وإلا إذا (گرہ->op == GGML_OP_FLASH_ATTN_EXT) {
                نقاب = گرہ->src[3];
            } وإلا {
                استمر;
            }
            إذا (نقاب == خالی_سوچک || نقاب->op != GGML_OP_NONE || !(نقاب->flags & GGML_TENSOR_FLAG_داخلہ) ||
                گرہ->src[0] == خالی_سوچک) {
                استمر;
            }
            model_params.is_cacheless_attn = سچ;
            model_params.n_seq = 1;
            model_params.ctx_فی_seq = نقاب->ne[0];
            compute_params.داخلہ_len = گرہ->src[0]->ne[1];
            compute_params.token_len_فی_seq = compute_params.داخلہ_len;
            اكسر;
        }
    }

    خودکار * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    إذا (model_params.is_cacheless_attn) {
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ثابت خودکار * گرہ = cgraph->nodes[i];
            إذا (گرہ->op == GGML_OP_GET_صفاں && is_output_idx(گرہ->src[1], گرہ)) {
                compute_params.output_len = گرہ->src[1]->ne[0];
                اكسر;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    إذا (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_فی_seq * model_params.n_seq;
    أرجع {model_params, compute_params};
}

فراغ GgmlOvDecoder::validate_cgraph() ثابت {
    إذا (m_model_params.n_seq > 1 && m_is_static == سچ) {
        ارم std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_گراف_داخلہ_shape(ثابت ggml_tensor * op,
                                                      ثابت ggml_tensor * داخلہ,
                                                      صحيح dynamic_dim_اشاریہ) ثابت {
    إذا (m_naive) {
        أرجع داخلہ != خالی_سوچک ? ov::PartialShape{get_shape(داخلہ)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape داخلہ_shape;

    إذا (is_inp_tok(داخلہ, op) || is_inp_pos(داخلہ, op)) {
        // tokens or positions
        صحيح len = m_is_static ? (m_is_prefill ? m_prefill_chunk_ناپ : 1) : -1;
        إذا (m_is_static && is_inp_pos(داخلہ, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        داخلہ_shape = ov::PartialShape{1, 1, 1, len};

    } وإلا إذا (is_output_idx(داخلہ, op)) {
        // output index
        داخلہ_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } وإلا إذا (is_inp_اوسط(داخلہ, op)) {
        داخلہ_shape = m_is_static ? ov::PartialShape{1, 1, داخلہ->ne[1], m_prefill_chunk_ناپ} :
                                    ov::PartialShape{1, 1, -1, -1};

    } وإلا إذا (is_inp_scale_صفاں(داخلہ, op)) {
        داخلہ_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_ناپ : 1) : -1, 1};

    } وإلا إذا (is_inp_نقاب(داخلہ, op)) {
        //نقاب
        إذا (m_is_static) {
            داخلہ_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_ناپ : 1, m_model_params.ctx};
        } وإلا إذا (m_is_stateful) {
            داخلہ_shape = ov::PartialShape{1, 1, -1, -1};
        } وإلا {
            داخلہ_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } وإلا إذا (is_recurrent_cache(داخلہ)) {
        داخلہ_shape = ov::PartialShape{get_shape(داخلہ)};
        إذا (!m_is_static && !m_is_stateful && داخلہ->ne[1] > 1) {
            داخلہ_shape[2] = -1;
        }

    } وإلا إذا (is_kvcache(داخلہ, op)) {
        // kvcache
        داخلہ_shape = ov::PartialShape{get_shape(داخلہ)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ثابت منطقي is_flat_kv = (داخلہ->ne[1] == 1 && داخلہ->ne[2] == 1 && داخلہ->ne[3] == 1);
        إذا (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            داخلہ_shape[2] = -1;
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
            أكد(داخلہ_shape.size() == 4 && داخلہ_shape[0] == 1 && داخلہ_shape[1] == 1 &&
                   داخلہ_shape[2].is_dynamic() && داخلہ_shape[3].is_static());
            ثابت صحيح n_heads_kv = get_n_heads_kv_for_tensor(داخلہ);
            أكد(n_heads_kv > 0 && داخلہ_shape[3].get_لمائی() % n_heads_kv == 0);
            ثابت int64_t combined_dim = داخلہ_shape[3].get_لمائی();  // n_heads_kv * head_size
            ثابت int64_t سر_ناپ = combined_dim / n_heads_kv;
            داخلہ_shape = {داخلہ_shape[0], ov::Dimension::dynamic(), n_heads_kv, سر_ناپ};
        }

    } وإلا إذا (is_kv_idx(داخلہ, op)) {
        // kv update index
        صحيح len = m_is_static ? (m_is_prefill ? m_prefill_chunk_ناپ : 1) : -1;
        داخلہ_shape = ov::PartialShape{1, 1, 1, len};

    } وإلا إذا (is_inp_s_copy(داخلہ, op) || is_s_copy_پتی(داخلہ)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        داخلہ_shape = m_is_static ? ov::PartialShape{get_shape(داخلہ)} : ov::PartialShape{1, 1, 1, -1};

    } وإلا {
        داخلہ_shape = ov::PartialShape{get_shape(داخلہ)};
    }
    إذا (dynamic_dim_اشاریہ != -1 && m_model_is_splitted) {
        داخلہ_shape[3 - dynamic_dim_اشاریہ] = -1;
    }
    إذا (op->op == GGML_OP_SOFT_MAX && op->src[1] != خالی_سوچک && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_داخلہ && op->src[1] == داخلہ) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        إذا (m_is_static) {
            ثابت int64_t seq_سرگرم = m_is_prefill ? m_prefill_chunk_ناپ : 1;
            داخلہ_shape[2] = seq_سرگرم;
            داخلہ_shape[3] = seq_سرگرم;
        } وإلا {
            داخلہ_shape[2] = -1;
            داخلہ_shape[3] = -1;
        }
    }
    أرجع داخلہ_shape;
}

منطقي GgmlOvDecoder::is_s_copy_پتی(ثابت ggml_tensor * tensor) ثابت {
    إذا (tensor == خالی_سوچک || tensor->op != GGML_OP_NONE || m_cgraph == خالی_سوچک) {
        أرجع خطأ;
    }
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت ggml_tensor * گرہ = m_cgraph->nodes[i];
        إذا (گرہ->op != GGML_OP_GET_صفاں || گرہ->src[0] == خالی_سوچک || گرہ->src[1] == خالی_سوچک) {
            استمر;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ثابت ggml_tensor * idx = گرہ->src[1];
        طالما (idx != خالی_سوچک && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        إذا (idx != tensor) {
            استمر;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ثابت ggml_tensor * انکڑے = گرہ->src[0];
        طالما (انکڑے != خالی_سوچک && (انکڑے->op == GGML_OP_VIEW || انکڑے->op == GGML_OP_RESHAPE)) {
            انکڑے = انکڑے->src[0];
        }
        إذا (انکڑے != خالی_سوچک && is_recurrent_cache(انکڑے)) {
            أرجع سچ;
        }
    }
    أرجع خطأ;
}

فراغ GgmlOvDecoder::جوڑو_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    خودکار create_1d_داخلہ = [هذا](ثابت std::string & name, int64_t value, منطقي force_parameter = خطأ) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    إذا (m_compute_params.attention_ناپ != -1) {
        create_1d_داخلہ("attention_size", m_compute_params.attention_ناپ);
    }
    إذا (m_compute_params.attention_ناپ_static != -1) {
        create_1d_داخلہ("attention_size_static", m_compute_params.attention_ناپ_static);
    }
    إذا (m_compute_params.attention_ناپ_swa != -1) {
        create_1d_داخلہ("attention_size_swa", m_compute_params.attention_ناپ_swa);
    }
    // only the stateful SWA mask consumes this
    إذا (is_stateful() && m_compute_params.swa_کھڑکی != -1) {
        create_1d_داخلہ("swa_window", m_compute_params.swa_کھڑکی);
    }
    create_1d_داخلہ("n_seq_active", m_compute_params.n_seq_سرگرم);
    create_1d_داخلہ("seq_active_start", m_compute_params.seq_سرگرم_شروع);
    create_1d_داخلہ("seq_active_end", m_compute_params.seq_سرگرم_شروع + m_compute_params.n_seq_سرگرم);
    إذا (m_compute_params.token_len_فی_seq != -1) {
        create_1d_داخلہ("token_len_per_seq", m_compute_params.token_len_فی_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    إذا (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_داخلہ("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/سچ);
        create_1d_داخلہ("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/سچ);
    }

    إذا (m_compute_params.s_copy_سرگرم_تھاں_len != -1) {
        create_1d_داخلہ("s_copy_active_slot_len", m_compute_params.s_copy_سرگرم_تھاں_len);
        إذا (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_داخلہ("chunk_valid_len", get_static_n_ٹوکن(), /*force_parameter=*/سچ);
        }
    }

    لكل (ثابت خودکار & [گرہ_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_داخلہ("rs_slot_begin_" + گرہ_name, writeback.تھاں_begin);
        إذا (!m_is_static && writeback.src_begin >= 0) {
            create_1d_داخلہ("rs_src_begin_" + گرہ_name, writeback.src_begin);
        }
    }
}

منطقي GgmlOvDecoder::گرہ_is_ورتیا_as_src(ثابت صحيح گرہ_idx) {
    ggml_tensor * گرہ = m_cgraph->nodes[گرہ_idx];
    لكل (صحيح i = گرہ_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_گرہ = m_cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (other_گرہ->src[j] == گرہ) {
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
        ggml_tensor * گرہ = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        إذا (گرہ->op == GGML_OP_NONE && گرہ_is_ورتیا_as_src(i)) {
            std::string گرہ_name = get_tensor_ov_name(m_cgraph, گرہ);
            إذا (m_model_weights.find(گرہ_name) == m_model_weights.end()) {
                m_inputs[گرہ_name] = گرہ;
                m_model_inputs[گرہ_name] = {get_ov_type(گرہ),
                                             get_گراف_داخلہ_shape(گرہ, خالی_سوچک, m_گرہ_dynamic_dims[گرہ])};
            }
            استمر;
        }
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            خودکار * src = گرہ->src[i];
            إذا (src == خالی_سوچک) {
                استمر;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            إذا (src->flags & GGML_TENSOR_FLAG_داخلہ) {
                src_name = get_tensor_گراف_داخلہ_ov_name(هذا, m_cgraph, src, گرہ);
            }
            إذا (m_model_weights.find(src_name) != m_model_weights.end()) {
                استمر;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            إذا (src->view_src != خالی_سوچک &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                استمر;
            }

            منطقي is_intermediate_گرہ = خطأ;
            لكل (ثابت خودکار & گرہ_info : m_گرہ_info_list) {
                إذا (گرہ_info.گرہ == src) {
                    is_intermediate_گرہ = سچ;
                    اكسر;
                }
            }
            إذا (is_intermediate_گرہ) {
                استمر;
            }
            إذا (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                استمر;
            }

            m_inputs[src_name] = src;

            ggml_backend_بفر * بفر = src->بفر;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            إذا (بفر->usage == GGML_BACKEND_بفر_USAGE_ANY) {
                إذا (خودکار it = std::find(m_model_params.kv_ناں.begin(), m_model_params.kv_ناں.end(), src_name);
                    it == m_model_params.kv_ناں.end()) {
                    m_model_params.kv_ناں.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            طالما (src->op == GGML_OP_VIEW && src->src[0] != خالی_سوچک) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_گراف_داخلہ_shape(گرہ, src, m_گرہ_dynamic_dims[src])};
        }
    }
}

فراغ GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_ناں.clear();
    لكل (صحيح گرہ_n = 0; گرہ_n < m_cgraph->n_nodes; گرہ_n++) {
        خودکار * cur_گرہ = m_cgraph->nodes[گرہ_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        إذا (cur_گرہ->op == GGML_OP_NONE || cur_گرہ->op == GGML_OP_VIEW || cur_گرہ->op == GGML_OP_RESHAPE) {
            استمر;
        }
        إذا (::is_inplace_op(cur_گرہ) && ggml_nbytes(cur_گرہ) == 0) {
            استمر;
        }
        خودکار cur_گرہ_use_گنتی = m_cgraph->use_counts[ggml_hash_لبھو(&m_cgraph->ویکھیا_hash_سیٹ, cur_گرہ)];
        إذا (cur_گرہ_use_گنتی == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            إذا (cur_گرہ != خالی_سوچک && ::is_inplace_op(cur_گرہ) && ggml_nbytes(cur_گرہ) > 0) {
                cur_گرہ = cur_گرہ->view_src;
            }
        } وإلا {
            صحيح داخلہ_use_گنتی = 0;
            لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * گرہ = m_cgraph->nodes[i];
                لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                    إذا (گرہ->src[j] != عدم && گرہ->src[j] == cur_گرہ) {
                        داخلہ_use_گنتی++;
                    }
                }
            }
            إذا (داخلہ_use_گنتی == cur_گرہ_use_گنتی) {
                cur_گرہ = خالی_سوچک;
            }
        }
        إذا (cur_گرہ != خالی_سوچک) {
            std::string cur_گرہ_name = get_tensor_ov_name(m_cgraph, cur_گرہ);
            m_model_outputs[cur_گرہ_name] = cur_گرہ;
            m_model_output_ناں.insert(cur_گرہ_name);
        }
    }
}

ثابت ggml_tensor * GgmlOvDecoder::get_tensor_ورتیا_op(ثابت ggml_tensor * tensor) ثابت {
    إذا (tensor == خالی_سوچک) {
        أرجع خالی_سوچک;
    }
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت خودکار * گرہ = m_cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (گرہ->src[j] == tensor) {
                أرجع گرہ;
            }
        }
    }
    أرجع خالی_سوچک;
}

ثابت ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ثابت std::string & name) ثابت {
    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ثابت خودکار * گرہ = m_cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            ثابت خودکار * src = گرہ->src[j];
            إذا (src == خالی_سوچک) {
                اكسر;
            }
            إذا (get_tensor_ov_name(m_cgraph, src) == name) {
                أرجع src;
            }
        }
    }
    أرجع خالی_سوچک;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_ناں() ثابت {
    std::map<std::string, std::string> kv_param_res_ناں;
    لكل (ثابت خودکار & name : m_model_params.kv_ناں) {
        kv_param_res_ناں[name] = name;
    }
    أرجع kv_param_res_ناں;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
ساكن منطقي is_mul_mat_id_expert_بھار(ثابت ggml_tensor * گرہ, صحيح src_اشاریہ) {
    أرجع گرہ->op == GGML_OP_MUL_MAT_ID && src_اشاریہ == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_بھار_nodes(ggml_cgraph * cgraph, منطقي naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    خودکار * nodes = cgraph->nodes;
    خودکار n_nodes = cgraph->n_nodes;
    لكل (صحيح گرہ_i = 0; گرہ_i < n_nodes; گرہ_i++) {
        خودکار * گرہ = nodes[گرہ_i];
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            خودکار * src = گرہ->src[i];
            إذا (src == خالی_سوچک) {
                استمر;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * ادھار = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, ادھار);
            إذا (is_rope_freqs_بھار(ادھار, گرہ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_بفر * بفر = ادھار->بفر;
            إذا (بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS || ggml_is_quantized(ادھار->type) ||
                is_mul_mat_id_expert_بھار(گرہ, i)) {
                إذا (model_weights.find(src_name) == model_weights.end()) {
                    خودکار بھار_گرہ = create_بھار_گرہ(ادھار, naive);
                    بھار_گرہ->سیٹ_friendly_name(src_name);
                    model_weights[src_name] = بھار_گرہ;
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
ساكن std::mutex g_nonov_بھار_cache_mutex;
ساكن std::unordered_map<ثابت فراغ *, std::shared_ptr<ov::Node>> g_nonov_بھار_cache;

std::set<std::string> GgmlOvDecoder::collect_بھار_ناں(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> ناں;
    لكل (صحيح گرہ_i = 0; گرہ_i < cgraph->n_nodes; گرہ_i++) {
        خودکار * گرہ = cgraph->nodes[گرہ_i];
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            خودکار * src = گرہ->src[i];
            إذا (src == خالی_سوچک) {
                استمر;
            }
            ثابت ggml_tensor * ادھار = src->view_src ? src->view_src : src;
            std::string src_name(ادھار->name);
            إذا (is_rope_freqs_بھار(ادھار, گرہ)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_بفر * بفر = ادھار->بفر;
            إذا (بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS || ggml_is_quantized(ادھار->type)) {
                ناں.insert(src_name);
            }
        }
    }
    أرجع ناں;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_بھار_گرہ(ggml_tensor * tensor, منطقي naive) {
    ثابت منطقي is_ov_بفر = ggml_backend_بفر_is_openvino(tensor->بفر);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    إذا (tensor->extra) {
        OPENVINO_ASSERT(is_ov_بفر, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        خودکار * extra_ادھار = static_cast<ggml_openvino_extra_ادھار *>(tensor->extra);

        إذا (extra_ادھار->type == ggml_openvino_extra_ادھار::Type::بھار_2) {
            // F16/F32/BF16 weight with shared-memory constant
            خودکار * بھار_extra = static_cast<ggml_openvino_بھار_extra *>(tensor->extra);
            إذا (بھار_extra->بھار_گرہ) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                أرجع بھار_extra->بھار_گرہ;
            }
        } وإلا إذا (extra_ادھار->type == ggml_openvino_extra_ادھار::Type::QUANTIZED_بھار) {
            // Quantized weight with pre-extracted data
            خودکار * quant_extra = static_cast<ggml_openvino_quantized_بھار_extra *>(tensor->extra);
            إذا (quant_extra->بھار_گرہ) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                أرجع quant_extra->بھار_گرہ;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    إذا (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        خودکار بھار_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->انکڑے);
        خودکار بھار_گرہ = std::make_shared<ov::op::v0::Constant>(بھار_tensor);
        بھار_گرہ->سیٹ_friendly_name(tensor->name);
        أرجع بھار_گرہ;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ثابت منطقي cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_بفر &&
                                 !naive && tensor->انکڑے != خالی_سوچک;
    إذا (cacheable_nonov) {
        std::lock_guard<std::mutex> جندرا(g_nonov_بھار_cache_mutex);
        خودکار it = g_nonov_بھار_cache.find(tensor->انکڑے);
        إذا (it != g_nonov_بھار_cache.end()) {
            أرجع it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    ساكن ثابت std::set<ggml_type> بھار_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    إذا (بھار_types.find(tensor->type) == بھار_types.end()) {
        ارم std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    Ov_بھار ov_بھار;
    إذا (ggml_is_quantized(tensor->type)) {
        خودکار use_bias = naive;
        إذا (is_ov_بفر) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_ناپ = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_ناپ);
            حافظہ_نقل(tmp.انکڑے(), tensor->انکڑے, raw_ناپ);
            ov_بھار = process_بھار_tensor(tensor, tmp.انکڑے(), tensor->انکڑے, use_bias);
        } وإلا {
            ov_بھار = process_بھار_tensor(tensor, tensor->انکڑے, خالی_سوچک, use_bias);
        }
    } وإلا {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_بھار = process_بھار_tensor(tensor, tensor->انکڑے, tensor->انکڑے);
    }

    ov_بھار.بھار_گرہ->سیٹ_friendly_name(tensor->name);
    إذا (!is_ov_بفر) {
        إذا (cacheable_nonov) {
            std::lock_guard<std::mutex> جندرا(g_nonov_بھار_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            خودکار [it, inserted] = g_nonov_بھار_cache.emplace(tensor->انکڑے, ov_بھار.بھار_گرہ);
            أرجع it->second;
        }
        أرجع ov_بھار.بھار_گرہ;
    }

    ggml_openvino_extra_ادھار * extra;
    إذا (ov_بھار.is_quantized()) {
        extra = جديد ggml_openvino_quantized_بھار_extra(std::move(ov_بھار.weights), std::move(ov_بھار.scales),
                                                         std::move(ov_بھار.zp), ov_بھار.بھار_گرہ);
    } وإلا {
        extra = جديد ggml_openvino_بھار_extra(std::move(ov_بھار.weights), ov_بھار.بھار_گرہ);
    }
    ggml_openvino_بفر_رجسٹر_extra(tensor, extra);

    أرجع ov_بھار.بھار_گرہ;
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
        ggml_tensor * گرہ = cgraph->nodes[i];

        // Get buffer type name
        ثابت حرف * buf_name = "none";
        ggml_backend_بفر_t buf = گرہ->view_src ? گرہ->view_src->بفر : گرہ->بفر;
        إذا (buf) {
            buf_name = ggml_backend_بفر_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << گرہ->ne[0] << ", "
             << std::setw(5) << گرہ->ne[1] << ", "
             << std::setw(5) << گرہ->ne[2] << ", "
             << std::setw(5) << گرہ->ne[3] << "] "
             << std::کھبے << std::setw(20) << ggml_op_name(گرہ->op) << std::سجے << " "
             << std::کھبے << std::setw(45) << گرہ->name << std::سجے
             << std::setw(2) << "[ "
             << std::setw(0) << گرہ->nb[0] << ", "
             << std::setw(5) << گرہ->nb[1] << ", "
             << std::setw(5) << گرہ->nb[2] << ", "
             << std::setw(5) << گرہ->nb[3] << "] "
             << std::سجے << std::setw(15) << buf_name << std::سجے
             << "\n";

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            إذا (خودکار* src = گرہ->src[i]) {
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
                << i << ": " << std::کھبے << std::setw(12) << ggml_op_name(src->op) << std::سجے;
                file << std::کھبے << std::setw(30) << src->name << std::سجے
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << std::سجے << std::setw(15) << src_buf_name << std::سجے
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    لكل (صحيح i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * گرہ = cgraph->leafs[i];

        // Get buffer type name for leaf
        ثابت حرف * پتی_buf_name = "none";
        ggml_backend_بفر_t پتی_buf = گرہ->view_src ? گرہ->view_src->بفر : گرہ->بفر;
        إذا (پتی_buf) {
            پتی_buf_name = ggml_backend_بفر_name(پتی_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << گرہ->ne[0] << ", "
             << std::setw(5) << گرہ->ne[1] << "] "
             << std::setw(8) << ggml_op_name(گرہ->op) << " "
             << std::setw(16) << ggml_get_name(گرہ)
             << std::setw(20) << پتی_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

فراغ print_tensor_پتہ_map(ثابت ggml_cgraph * cgraph) {
    std::map<فراغ *, std::vector<std::string>> پتہ_map;
    لكل (صحيح گرہ_n = 0; گرہ_n < cgraph->n_nodes; گرہ_n++) {
        خودکار * گرہ = cgraph->nodes[گرہ_n];
        إذا (گرہ->انکڑے) {
            خودکار it = پتہ_map.find(گرہ->انکڑے);
            إذا (it == پتہ_map.end()) {
                پتہ_map[گرہ->انکڑے] = std::vector<std::string>();
            }
            پتہ_map[گرہ->انکڑے].push_back(گرہ->name);
        }
    }
    لكل (ثابت خودکار & pair : پتہ_map) {
        std::باہر << "Address: " << pair.پہلا << '\n';
        لكل (ثابت خودکار & name : pair.second) {
            std::باہر << name << " ; ";
        }
        std::باہر << "\n\n";
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

ov::PartialShape GgmlOvDecoder::get_داخلہ_shape(صحيح گرہ_idx, ثابت std::string & name) ثابت {
    أرجع ov::PartialShape(get_shape(m_گرہ_info_list[گرہ_idx].گرہ_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_داخلہ_stride(صحيح گرہ_idx, ثابت std::string & name) ثابت {
    أرجع get_stride(m_گرہ_info_list[گرہ_idx].گرہ_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_داخلہ_ناپ(صحيح گرہ_idx, ثابت std::string & name) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        أرجع it->second.size();
    }
    أرجع 0;
}

size_t GgmlOvDecoder::get_view_داخلہ_وتھ(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        إذا (view_اشاریہ < it->second.size()) {
            أرجع it->second[view_اشاریہ].second->view_offs;
        }
    }
    أرجع 0;
}

size_t GgmlOvDecoder::get_view_داخلہ_src_وتھ(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        إذا (view_اشاریہ < it->second.size()) {
            خودکار * view_tensor = it->second[view_اشاریہ].second;
            إذا (view_tensor && view_tensor->src[0]) {
                أرجع view_tensor->src[0]->view_offs;
            }
        }
    }
    أرجع 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_داخلہ_stride(صحيح گرہ_idx,
                                                         ثابت std::string & name,
                                                         size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        إذا (view_اشاریہ < it->second.size()) {
            أرجع get_stride(it->second[view_اشاریہ].second);
        }
    }
    أرجع {};
}

std::vector<size_t> GgmlOvDecoder::get_view_داخلہ_src_stride(صحيح گرہ_idx,
                                                             ثابت std::string & name,
                                                             size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        إذا (view_اشاریہ < it->second.size()) {
            خودکار * view_tensor = it->second[view_اشاریہ].second;
            إذا (view_tensor && view_tensor->src[0]) {
                أرجع get_stride(view_tensor->src[0]);
            }
        }
    }
    أرجع {};
}

ov::Shape GgmlOvDecoder::get_view_داخلہ_ggml_shape(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        إذا (view_اشاریہ < it->second.size()) {
            أرجع get_shape(it->second[view_اشاریہ].second);
        }
    }
    أرجع {};
}

ov::Shape GgmlOvDecoder::get_view_داخلہ_src_ggml_shape(صحيح گرہ_idx,
                                                       ثابت std::string & name,
                                                       size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        إذا (view_اشاریہ < it->second.size()) {
            خودکار * view_tensor = it->second[view_اشاریہ].second;
            إذا (view_tensor && view_tensor->src[0]) {
                أرجع get_shape(view_tensor->src[0]);
            }
        }
    }
    أرجع {};
}

ov::PartialShape GgmlOvDecoder::get_view_داخلہ_ov_shape(صحيح گرہ_idx,
                                                        ثابت std::string & name,
                                                        size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        إذا (view_اشاریہ < it->second.size()) {
            خودکار * tensor = it->second[view_اشاریہ].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            خودکار dynamic_it = m_گرہ_dynamic_dims.find(tensor);
            إذا (dynamic_it != m_گرہ_dynamic_dims.end() && dynamic_it->second != -1) {
                صحيح dynamic_dim_اشاریہ = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_اشاریہ] = m_is_static ? get_static_n_ٹوکن() : -1;
            }

            أرجع shape;
        }
    }
    أرجع {};
}

ov::PartialShape GgmlOvDecoder::get_view_داخلہ_src_ov_shape(صحيح گرہ_idx,
                                                            ثابت std::string & name,
                                                            size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        إذا (view_اشاریہ < it->second.size()) {
            خودکار * view_tensor = it->second[view_اشاریہ].second;
            إذا (view_tensor && view_tensor->src[0]) {
                خودکار * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                خودکار dynamic_it = m_گرہ_dynamic_dims.find(src_tensor);
                إذا (dynamic_it != m_گرہ_dynamic_dims.end() && dynamic_it->second != -1) {
                    صحيح dynamic_dim_اشاریہ = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_اشاریہ] = m_is_static ? get_static_n_ٹوکن() : -1;
                }

                أرجع shape;
            }
        }
    }
    أرجع {};
}

std::string GgmlOvDecoder::get_view_داخلہ_name(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        إذا (view_اشاریہ < it->second.size()) {
            أرجع it->second[view_اشاریہ].پہلا;
        }
    }
    أرجع "";
}

std::string GgmlOvDecoder::get_view_داخلہ_src_name(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت {
    خودکار it = m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.find(name);
    إذا (it != m_گرہ_info_list[گرہ_idx].گرہ_inputs_views.end()) {
        إذا (view_اشاریہ < it->second.size()) {
            خودکار * view_tensor = it->second[view_اشاریہ].second;
            إذا (view_tensor && view_tensor->src[0]) {
                أرجع get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    أرجع "";
}

ov::element::Type GgmlOvDecoder::get_داخلہ_type(صحيح گرہ_idx, ثابت std::string & name) ثابت {
    أرجع get_ov_type(m_گرہ_info_list[گرہ_idx].گرہ_inputs.at(name));
}

size_t GgmlOvDecoder::get_داخلہ_ناپ() ثابت {
    أرجع m_model_inputs.size();
}

size_t GgmlOvDecoder::get_داخلہ_ناپ(صحيح گرہ_idx) ثابت {
    أرجع m_گرہ_info_list[گرہ_idx].گرہ_inputs_ناں.size();
}

std::vector<std::string> GgmlOvDecoder::get_داخلہ_ناں(صحيح گرہ_idx) ثابت {
    أرجع m_گرہ_info_list[گرہ_idx].گرہ_inputs_ناں;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(صحيح گرہ_idx) ثابت {
    خودکار * ggml_tensor = m_گرہ_info_list[گرہ_idx].گرہ;
    أرجع ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ثابت صحيح گرہ_idx) ثابت {
    أرجع get_ov_type(m_گرہ_info_list[گرہ_idx].گرہ);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(صحيح گرہ_idx) ثابت {
    خودکار * ggml_tensor = m_گرہ_info_list[گرہ_idx].گرہ;
    أرجع get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_ناں(صحيح گرہ_idx) ثابت {
    خودکار * گرہ = m_گرہ_info_list[گرہ_idx].گرہ;
    إذا (گرہ->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        لكل (صحيح i = گرہ_idx + 1; i < m_cgraph->n_nodes; i++) {
            خودکار * کھپتکار = m_cgraph->nodes[i];
            إذا (کھپتکار->op != GGML_OP_VIEW || کھپتکار->src[0] != گرہ) {
                استمر;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            خودکار & name = کھپتکار->view_offs == 0 ? attn_name : state_name;
            إذا (!name.خالی_تھاواں()) {
                أرجع {m_گرہ_info_list[گرہ_idx].گرہ_name};
            }
            name = get_tensor_ov_name(m_cgraph, کھپتکار);
        }
        إذا (!attn_name.خالی_تھاواں() && !state_name.خالی_تھاواں()) {
            أرجع {attn_name, state_name};
        }
    }
    أرجع {m_گرہ_info_list[گرہ_idx].گرہ_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(صحيح گرہ_idx) ثابت {
    خودکار * گرہ = m_گرہ_info_list[گرہ_idx].گرہ;
    إذا (!::is_inplace_op(گرہ) || گرہ->view_src == خالی_سوچک || ggml_nbytes(گرہ) == 0) {
        أرجع "";
    }
    ثابت صحيح op_case = m_گرہ_info_list[گرہ_idx].گرہ_op_case;
    إذا (گرہ->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_سرگرم_تھاں_len == -1) {
        أرجع "";
    }
    أرجع get_tensor_ov_name(m_cgraph, گرہ->view_src);
}

منطقي GgmlOvDecoder::is_view_like_alias_دا(صحيح گرہ_idx, ثابت std::string & view_src_name) ثابت {
    خودکار * گرہ = m_گرہ_info_list[گرہ_idx].گرہ;
    إذا (گرہ->view_src == خالی_سوچک || get_tensor_ov_name(m_cgraph, گرہ->view_src) != view_src_name) {
        أرجع خطأ;
    }
    أرجع گرہ->op == GGML_OP_RESHAPE || گرہ->op == GGML_OP_VIEW;
}

ثابت std::string & GgmlOvDecoder::get_op_name() ثابت {
    ساكن ثابت std::string unknown_name = "UNKNOWN_OP_NAME";
    أرجع unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(صحيح گرہ_idx) ثابت {
    خودکار it = m_گرہ_dynamic_dims.find(m_گرہ_info_list[گرہ_idx].گرہ);
    إذا (it == m_گرہ_dynamic_dims.end()) {
        أرجع -1;
    }
    أرجع it->second;
}

ثابت std::string & GgmlOvDecoder::get_op_name(صحيح گرہ_idx) ثابت {
    أرجع m_گرہ_info_list[گرہ_idx].گرہ_name;
}

int32_t * GgmlOvDecoder::get_داخلہ_op_params(صحيح گرہ_idx, ثابت std::string & name) ثابت {
    أرجع m_گرہ_info_list[گرہ_idx].گرہ_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(صحيح گرہ_idx) ثابت {
    أرجع m_گرہ_info_list[گرہ_idx].گرہ->op_params;
}

size_t GgmlOvDecoder::get_output_op_وتھ(صحيح گرہ_idx) ثابت {
    أرجع m_گرہ_info_list[گرہ_idx].گرہ->view_offs;
}

فراغ GgmlOvDecoder::ویکھو_subgraph(std::function<فراغ(std::shared_ptr<GgmlDecoder>, صحيح گرہ_idx)> گرہ_visitor) ثابت {
    لكل (صحيح گرہ_idx = 0; گرہ_idx < m_cgraph->n_nodes; گرہ_idx++) {
        إذا (m_cgraph->nodes[گرہ_idx]->op == GGML_OP_NONE) {
            استمر;
        }
        گرہ_visitor(std::make_shared<GgmlOvDecoder>(*هذا), گرہ_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ثابت ggml_tensor * گرہ) {
    بدل (گرہ->op) {
    حالة GGML_OP_UNARY:
        أرجع std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(گرہ));
    حالة GGML_OP_GLU:
        أرجع std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(گرہ));
    افتراضي:
        أرجع std::string("GGML_OP_") + ggml_op_name(گرہ->op);
    }
}

ثابت std::string & GgmlOvDecoder::get_op_type(صحيح گرہ_idx) ثابت {
    أرجع m_گرہ_info_list[گرہ_idx].گرہ_op_type;
}

ثابت std::string & GgmlOvDecoder::get_op_type() ثابت {
    ساكن ثابت std::string unknown_op = "UNKNOWN_GGML_OP";
    أرجع unknown_op;
}

فراغ GgmlOvDecoder::compute_گرہ_dynamic_dims() {
    خودکار ویکھو_گرہ = [&](خودکار && self, ggml_tensor * گرہ) -> فراغ {
        إذا (!گرہ) {
            أرجع;
        }

        إذا (گرہ->op == GGML_OP_CPY) {
            m_گرہ_dynamic_dims[گرہ] = -1;
        }

        إذا (m_گرہ_dynamic_dims.count(گرہ)) {
            أرجع;
        }
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = گرہ->src[i];
            إذا (src == خالی_سوچک) {
                استمر;
            }
            بنية ggml_tensor * جڑ_src = خالی_سوچک;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            إذا (جڑ_src) {
                إذا (is_inp_tok(جڑ_src, گرہ) || is_inp_pos(جڑ_src, گرہ) || is_output_idx(جڑ_src, گرہ)) {
                    m_گرہ_dynamic_dims[جڑ_src] = 0;
                    m_گرہ_dynamic_dims[src] = m_گرہ_dynamic_dims[جڑ_src];
                    استمر;
                }
                self(self, جڑ_src);
                m_گرہ_dynamic_dims[src] = m_گرہ_dynamic_dims[جڑ_src];
            } وإلا {
                إذا (is_inp_tok(src, گرہ) || is_inp_pos(src, گرہ) || is_output_idx(src, گرہ)) {
                    m_گرہ_dynamic_dims[src] = 0;
                    استمر;
                }
                إذا (is_inp_scale_صفاں(src, گرہ)) {
                    m_گرہ_dynamic_dims[src] = 1;
                    استمر;
                }
                إذا (گرہ->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_گرہ_dynamic_dims[src] = 1;
                    استمر;
                }
                self(self, src);
            }
        }
        بدل (گرہ->op) {
        حالة GGML_OP_NONE:
            m_گرہ_dynamic_dims[گرہ] = -1;
            اكسر;
        حالة GGML_OP_GET_صفاں:
            m_گرہ_dynamic_dims[گرہ] = -1;
            إذا (m_گرہ_dynamic_dims[گرہ->src[1]] != -1) {
                خودکار dynamic_dim_idx = m_گرہ_dynamic_dims[گرہ->src[1]];
                إذا (dynamic_dim_idx == 0) {
                    m_گرہ_dynamic_dims[گرہ] = 1;
                } وإلا {
                    خودکار dynamic_dim_stride = گرہ->src[1]->nb[dynamic_dim_idx] / ggml_type_ناپ(گرہ->src[1]->type) *
                                              ggml_type_ناپ(گرہ->src[0]->type);
                    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                        إذا (dynamic_dim_stride == گرہ->src[0]->nb[i]) {
                            m_گرہ_dynamic_dims[گرہ] = i;
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
            m_گرہ_dynamic_dims[گرہ] = -1;
            إذا (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[0]];
            }
            إذا (m_گرہ_dynamic_dims[گرہ->src[1]] != -1) {
                m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[1]];
            }
            اكسر;
        حالة GGML_OP_ترتیب_بدلو:
            m_گرہ_dynamic_dims[گرہ] = -1;
            إذا (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                خودکار dynamic_dim_idx = m_گرہ_dynamic_dims[گرہ->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (گرہ->op_params[i] == dynamic_dim_idx) {
                        m_گرہ_dynamic_dims[گرہ] = i;
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
            m_گرہ_dynamic_dims[گرہ] = -1;
            إذا (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                إذا (گرہ->src[0]->op == GGML_OP_NONE) {
                    m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[0]];
                    اكسر;
                }
                خودکار dynamic_dim_idx = m_گرہ_dynamic_dims[گرہ->src[0]];
                خودکار dynamic_dim_قدر = گرہ->src[0]->ne[dynamic_dim_idx];
                خودکار dynamic_dim_stride =
                    گرہ->src[0]->nb[dynamic_dim_idx] / ggml_type_ناپ(گرہ->src[0]->type) * ggml_type_ناپ(گرہ->type);
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (گرہ->nb[i] == dynamic_dim_stride) {
                        m_گرہ_dynamic_dims[گرہ] = i;
                        اكسر;
                    }
                }
                إذا (m_گرہ_dynamic_dims[گرہ] != -1 && dynamic_dim_قدر != گرہ->ne[m_گرہ_dynamic_dims[گرہ]]) {
                    m_گرہ_dynamic_dims[گرہ] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    إذا (ggml_nelements(گرہ) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      گرہ->name, گرہ->src[0]->name);
                    }
                }
            }
            اكسر;
        }
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RESHAPE: {
            إذا (is_same_shape(گرہ->src[0], گرہ)) {
                m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[0]];
                اكسر;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_گرہ_dynamic_dims[گرہ] = -1;
            إذا (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                خودکار dynamic_dim_idx = m_گرہ_dynamic_dims[گرہ->src[0]];
                خودکار dynamic_dim_stride = گرہ->src[0]->nb[dynamic_dim_idx];
                لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                    إذا (گرہ->nb[i] == dynamic_dim_stride && گرہ->ne[i] == گرہ->src[0]->ne[dynamic_dim_idx]) {
                        m_گرہ_dynamic_dims[گرہ] = i;
                        اكسر;
                    }
                }
                إذا (m_گرہ_dynamic_dims[گرہ] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", گرہ->name);
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
            constexpr صحيح q_منزل_باہرلا[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_گرہ_dynamic_dims[گرہ] = -1;
            إذا (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                خودکار q_dynamic_dim = m_گرہ_dynamic_dims[گرہ->src[0]];
                m_گرہ_dynamic_dims[گرہ] = q_منزل_باہرلا[q_dynamic_dim];
            }
            اكسر;
        }
        حالة GGML_OP_CONT:
            m_گرہ_dynamic_dims[گرہ] = -1;
            إذا (m_گرہ_dynamic_dims[گرہ->src[0]] != -1) {
                خودکار dynamic_dim_idx = m_گرہ_dynamic_dims[گرہ->src[0]];
                إذا (ggml_are_same_shape(گرہ, گرہ->src[0])) {
                    m_گرہ_dynamic_dims[گرہ] = dynamic_dim_idx;
                } وإلا {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_ناپ(گرہ->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (گرہ->src[0]->ne[0] / ggml_blck_ناپ(گرہ->src[0]->type));
                    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * گرہ->src[0]->ne[i - 1];
                    }

                    خودکار dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_ناپ(گرہ->src[0]->type) *
                                              ggml_type_ناپ(گرہ->type);
                    صحيح matched_dim_گنتی = 0;
                    صحيح پہلا_matched_dim = -1;
                    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                        إذا (گرہ->nb[i] == dynamic_dim_stride && گرہ->ne[i] == گرہ->src[0]->ne[dynamic_dim_idx]) {
                            إذا (پہلا_matched_dim == -1) {
                                پہلا_matched_dim = i;
                            }
                            m_گرہ_dynamic_dims[گرہ] = i;
                            matched_dim_گنتی++;
                        }
                    }
                    إذا (matched_dim_گنتی > 1 && گرہ->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_گرہ_dynamic_dims[گرہ] = پہلا_matched_dim;
                    } وإلا إذا (matched_dim_گنتی != 1) {
                        m_گرہ_dynamic_dims[گرہ] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      گرہ->name, گرہ->src[0]->name);
                    }
                }
            }
            اكسر;
        حالة GGML_OP_CONCAT:
            لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
                إذا (گرہ->src[0]->ne[i] != گرہ->ne[i]) {
                    m_گرہ_dynamic_dims[گرہ] = i;
                    اكسر;
                }
            }
            اكسر;
        حالة GGML_OP_SSM_CONV:
        حالة GGML_OP_GATED_DELTA_NET:
            m_گرہ_dynamic_dims[گرہ] = 1;
            اكسر;
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_L2_NORM:
        حالة GGML_OP_NORM:
        حالة GGML_OP_جوڑو:
        حالة GGML_OP_SUB:
        حالة GGML_OP_GLU:
        حالة GGML_OP_ROPE:
        حالة GGML_OP_SCALE:
        حالة GGML_OP_SOFT_MAX:
        حالة GGML_OP_ARGSORT:
        حالة GGML_OP_جوڑو_ID:
        حالة GGML_OP_UNARY:
        حالة GGML_OP_CUMSUM:
        حالة GGML_OP_FILL:
        حالة GGML_OP_سیٹ:
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
            m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[0]];
            اكسر;
        حالة GGML_OP_جوڑ_صفاں:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_گرہ_dynamic_dims[گرہ] =
                (m_گرہ_dynamic_dims[گرہ->src[0]] == 0) ? -1 : m_گرہ_dynamic_dims[گرہ->src[0]];
            اكسر;
        حالة GGML_OP_MUL_MAT_ID:
        حالة GGML_OP_SOLVE_TRI:
            m_گرہ_dynamic_dims[گرہ] = m_گرہ_dynamic_dims[گرہ->src[1]];
            اكسر;
        حالة GGML_OP_CPY:
        حالة GGML_OP_سیٹ_صفاں:
        حالة GGML_OP_جوڑ:
        حالة GGML_OP_اوسط:
            m_گرہ_dynamic_dims[گرہ] = -1;
            اكسر;
        حالة GGML_OP_IM2_خانہ: {
            m_گرہ_dynamic_dims[گرہ] = -1;
            إذا (m_گرہ_dynamic_dims[گرہ->src[1]] != -1) {
                ثابت منطقي is_2D = گرہ->op_params[6] == 1;
                ثابت صحيح src_dyn = m_گرہ_dynamic_dims[گرہ->src[1]];
                إذا (is_2D) {
                    إذا (src_dyn == 0) {
                        m_گرہ_dynamic_dims[گرہ] = 1;  // IW -> OW
                    } وإلا إذا (src_dyn == 1) {
                        m_گرہ_dynamic_dims[گرہ] = 2;  // IH -> OH
                    } وإلا إذا (src_dyn == 3) {
                        m_گرہ_dynamic_dims[گرہ] = 3;  // N  -> N
                    }
                } وإلا {
                    إذا (src_dyn == 0) {
                        m_گرہ_dynamic_dims[گرہ] = 1;  // IW -> OW
                    } وإلا إذا (src_dyn == 2) {
                        m_گرہ_dynamic_dims[گرہ] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                إذا (m_گرہ_dynamic_dims[گرہ] != -1) {
                    OPENVINO_ASSERT(گرہ->src[1]->ne[src_dyn] == گرہ->ne[m_گرہ_dynamic_dims[گرہ]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(گرہ->name) +
                                        " and its src[1]: " + std::string(گرہ->src[1]->name));
                }
            }
            اكسر;
        }
        حالة GGML_OP_IM2_خانہ_3_D: {
            m_گرہ_dynamic_dims[گرہ] = -1;
            إذا (m_گرہ_dynamic_dims[گرہ->src[1]] != -1) {
                ثابت صحيح src_dyn = m_گرہ_dynamic_dims[گرہ->src[1]];
                إذا (src_dyn == 0) {
                    m_گرہ_dynamic_dims[گرہ] = 1;  // IW -> OW
                } وإلا إذا (src_dyn == 1) {
                    m_گرہ_dynamic_dims[گرہ] = 2;  // IH -> OH
                } وإلا إذا (src_dyn == 3) {
                    m_گرہ_dynamic_dims[گرہ] = 3;  // N  -> N
                }
                إذا (m_گرہ_dynamic_dims[گرہ] != -1) {
                    OPENVINO_ASSERT(گرہ->src[1]->ne[src_dyn] == گرہ->ne[m_گرہ_dynamic_dims[گرہ]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(گرہ->name) +
                                        " and its src[1]: " + std::string(گرہ->src[1]->name));
                }
            }
            اكسر;
        }
        افتراضي:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(گرہ->op), گرہ->name);
            اكسر;
        }
    };

    لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = m_cgraph->nodes[i];
        ویکھو_گرہ(ویکھو_گرہ, گرہ);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    إذا (0) {
        لكل (صحيح i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * گرہ = m_cgraph->nodes[i];
            صحيح dynamic_dim = m_گرہ_dynamic_dims[گرہ];
            std::باہر << "[" << i << "] " << "node_name: " << گرہ->name << " op: " << ggml_op_name(گرہ->op)
                      << " shape: [";
            لكل (صحيح j = 0; j < 4; j++) {
                إذا (j == dynamic_dim) {
                    std::باہر << "*";
                } وإلا {
                    std::باہر << گرہ->ne[j];
                }
                إذا (j < 3) {
                    std::باہر << ", ";
                }
            }
            std::باہر << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = گرہ->src[j];
                إذا (src == خالی_سوچک) {
                    استمر;
                }
                صحيح src_dynamic_dim = m_گرہ_dynamic_dims[src];
                std::باہر << "    [" << j << "] src_name: " << src->name << " [";
                لكل (صحيح k = 0; k < 4; k++) {
                    إذا (k == src_dynamic_dim) {
                        std::باہر << "*";
                    } وإلا {
                        std::باہر << src->ne[k];
                    }
                    إذا (k < 3) {
                        std::باہر << ", ";
                    }
                }
                std::باہر << "]" << '\n';
            }
            std::باہر << '\n';
        }
    }
}
