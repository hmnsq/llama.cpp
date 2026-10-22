#ಅನ್ತರ್ಭಾವ "ggml-decoder.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino-extra.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino.h"
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "utils.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <iomanip>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <openvino/core/dimension.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/except.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/partial_shape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/bfloat16.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/element_type.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/float16.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/tensor.hpp>
#ಅನ್ತರ್ಭಾವ <ostream>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <stdexcept>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             ತಾರ್ಕಿಕ is_static,
                             ತಾರ್ಕಿಕ is_stateful,
                             ತಾರ್ಕಿಕ model_is_splitted,
                             ತಾರ್ಕಿಕ is_prefill,
                             ಪೂರ್ಣಾಂಕ prefill_chunk_ಗಾತ್ರ) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(ಸುಳ್ಳು),
    m_prefill_chunk_ಗಾತ್ರ(prefill_chunk_ಗಾತ್ರ),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    ಸ್ಥಿರ ತಾರ್ಕಿಕ printed_ವಿಳಾಸ_map = ಸುಳ್ಳು;
    ಆದರೆ (!printed_ವಿಳಾಸ_map) {
        ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_ವಿಳಾಸ_map = ನಿಜ;
            print_tensor_ವಿಳಾಸ_map(cgraph);
        }
    }

    validate_cgraph();

    ಗಣ_ಒಳಹರಿವು_output();
    compute_ಗಂಟು_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_n = 0; ಗಂಟು_n < cgraph->n_nodes; ಗಂಟು_n++) {
        m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು_op_case = compute_op_case(m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು);
        m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು_op_type = compute_op_type(m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು);
    }

    ಕೂಡಿಸು_extra_inputs();
}

ಶೂನ್ಯ GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_ಗಂಟು_info_list.clear();
    ಗಣ_ಒಳಹರಿವು_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
ತಾರ್ಕಿಕ GgmlOvDecoder::is_bound_ಗಮ್ಯ(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ m_cgraph == cgraph && cgraph->n_nodes > 0 && m_ಗಂಟು_info_list.size() == (size_t) cgraph->n_nodes &&
           m_ಗಂಟು_info_list.front().ಗಂಟು == cgraph->nodes[0] &&
           m_ಗಂಟು_info_list.back().ಗಂಟು == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = ನಿಜ;
    ಗಣ_ಒಳಹರಿವು_output();
    compute_model_inputs();
    compute_model_outputs();
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_n = 0; ಗಂಟು_n < cgraph->n_nodes; ಗಂಟು_n++) {
        m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು_op_case = compute_op_case(m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು);
        m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು_op_type = compute_op_type(m_ಗಂಟು_info_list[ಗಂಟು_n].ಗಂಟು);
    }
}

ನಾಮಾಕಾಶ {
ತಾರ್ಕಿಕ is_inplace_op(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ಹಿಂತಿರುಗಿ ಗಂಟು->op == GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು || ಗಂಟು->op == GGML_OP_CPY || (ಗಂಟು->op == GGML_OP_SCALE && ಗಂಟು->view_src);
}

ತಾರ್ಕಿಕ is_same_shape(ಸ್ಥಿರಾಂಕ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ggml_tensor * b) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
        ಆದರೆ (a->ne[i] != b->ne[i]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ is_conv_ಸ್ಥಿತಿಗಳು_all_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ tensor != ಶೂನ್ಯಸೂಚಕ && ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(tensor->name, "conv_states_all", ಅಕ್ಷರಸರಣಿ_ಉದ್ದ("conv_states_all")) == 0;
}

ತಾರ್ಕಿಕ is_ತುಂಬಿದ_single_ಸ್ಥಾನ_writeback(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ಹಿಂತಿರುಗಿ ಗಂಟು->view_src != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->view_src->ne[1] == 1 && ಗಂಟು->src[1] != ಶೂನ್ಯಸೂಚಕ &&
           ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src == ಗಂಟು->view_src &&
           ಗಂಟು->src[1]->view_offs == 0 && ggml_nbytes(ಗಂಟು->src[1]) == ggml_nbytes(ಗಂಟು->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
ತಾರ್ಕಿಕ is_moe_expert_ಮೊತ್ತ_ಕೂಡಿಸು(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> leaves;
    ಸ್ಥಿರಾಂಕ ggml_tensor * cur = ಗಂಟು;
    ಆಗ (cur->op == GGML_OP_ಕೂಡಿಸು) {
        ಆದರೆ (cur->src[0] == ಶೂನ್ಯಸೂಚಕ || cur->src[1] == ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ = ಶೂನ್ಯಸೂಚಕ;
    std::set<int64_t> plane_indices;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_tensor * ಎಲೆ : leaves) {
        ಆದರೆ (ಎಲೆ->op != GGML_OP_VIEW || ಎಲೆ->src[0] == ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಎಲೆ_ತಳ = ಎಲೆ->src[0];
        ಆದರೆ (ತಳ == ಶೂನ್ಯಸೂಚಕ) {
            ತಳ = ಎಲೆ_ತಳ;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಎಲೆ_ತಳ != ತಳ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ಎಲೆ->ne[0] != ತಳ->ne[0] || ಎಲೆ->ne[1] != ತಳ->ne[2] || ಎಲೆ->ne[2] != 1 || ಎಲೆ->ne[3] != 1 ||
            ಎಲೆ->nb[1] != ತಳ->nb[2]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ತಳ->nb[1] == 0 || ಎಲೆ->view_offs % ತಳ->nb[1] != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        int64_t plane = static_cast<int64_t>(ಎಲೆ->view_offs / ತಳ->nb[1]);
        ಆದರೆ (plane < 0 || plane >= ತಳ->ne[1] || !plane_indices.insert(plane).second) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ತಳ != ಶೂನ್ಯಸೂಚಕ && ತಳ->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(ತಳ->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ "";
    }
    ಆದರೆ ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, ಶೂನ್ಯಸೂಚಕ)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        ಆದರೆ (ಗಂಟು != cgraph->nodes + cgraph->n_nodes) {
            ಹಿಂತಿರುಗಿ std::string(tensor->name) + "#n" + std::to_string(ಗಂಟು - cgraph->nodes);
        }
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ಎಲೆ = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        ಆದರೆ (ಎಲೆ != cgraph->leafs + cgraph->n_leafs) {
            ಹಿಂತಿರುಗಿ std::string(tensor->name) + "#l" + std::to_string(ಎಲೆ - cgraph->leafs);
        }
    }
    ಹಿಂತಿರುಗಿ tensor->name;
}

ಸ್ಥಿರ std::string get_tensor_ov_name(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

ಸ್ಥಿರ std::string get_tensor_ನಕ್ಷೆ_ಒಳಹರಿವು_ov_name(ಸ್ಥಿರಾಂಕ GgmlOvDecoder * decoder,
                                                  ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph,
                                                  ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                                                  ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಆದರೆ (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        ಹಿಂತಿರುಗಿ "inp_pos";
    }
    ಆದರೆ (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        ಹಿಂತಿರುಗಿ "embd";
    }
    ಆದರೆ (GgmlOvDecoder::is_inp_ಮುಖವಾಡ(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_swa = decoder->is_swa_ಮುಖವಾಡ(tensor);
        ಆದರೆ (decoder->is_stateful()) {
            ಹಿಂತಿರುಗಿ is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        ಆದರೆ (is_swa) {
            ಹಿಂತಿರುಗಿ get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    ಹಿಂತಿರುಗಿ get_tensor_ov_name(cgraph, tensor);
}

ಶೂನ್ಯ GgmlOvDecoder::ಗಣ_ಒಳಹರಿವು_output() {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_n = 0; ಗಂಟು_n < m_cgraph->n_nodes; ಗಂಟು_n++) {
        ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = m_cgraph->nodes[ಗಂಟು_n];

        ಗಂಟು_Info current_ಗಂಟು_info;
        ಸ್ವಯಂಚಾಲಿತ ಗಂಟು_name = get_tensor_ov_name(m_cgraph, ಗಂಟು);

        current_ಗಂಟು_info.ಗಂಟು = ಗಂಟು;
        current_ಗಂಟು_info.ಗಂಟು_name = ಗಂಟು_name;
        current_ಗಂಟು_info.ಗಂಟು_op_case = 0;
        current_ಗಂಟು_info.ದತ್ತಾಂಶ_addr = ಗಂಟು->ದತ್ತಾಂಶ;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಸ್ವಯಂಚಾಲಿತ * src = ಗಂಟು->src[i];
            ಆದರೆ (src == ಶೂನ್ಯಸೂಚಕ) {
                ಮುಂದುವರಿ;
            }
            ಸ್ವಯಂಚಾಲಿತ src_name = get_tensor_ov_name(m_cgraph, src);
            ಆದರೆ (src->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
                src_name = get_tensor_ನಕ್ಷೆ_ಒಳಹರಿವು_ov_name(ಇದು, m_cgraph, src, ಗಂಟು);
            }
            current_ಗಂಟು_info.ಗಂಟು_inputs[src_name] = src;
            current_ಗಂಟು_info.ಗಂಟು_inputs_ಹೆಸರುಗಳು.push_back(src_name);

            ಆದರೆ (src->op == GGML_OP_VIEW) {
                // Traverse upward through nested VIEW operations
                std::remove_reference_t<decltype(current_ಗಂಟು_info.ಗಂಟು_inputs_views[src_name])> view_chain;
                ಸ್ವಯಂಚಾಲಿತ * current = src;

                ಆಗ (current != ಶೂನ್ಯಸೂಚಕ) {
                    ಸ್ವಯಂಚಾಲಿತ current_name = get_tensor_ov_name(m_cgraph, current);
                    ಆದರೆ (current->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
                        current_name = get_tensor_ನಕ್ಷೆ_ಒಳಹರಿವು_ov_name(ಇದು, m_cgraph, current, ಗಂಟು);
                    }
                    view_chain.emplace_back(current_name, current);
                    // If current src is also a VIEW, continue traversing
                    ಆದರೆ (current->src[0] != ಶೂನ್ಯಸೂಚಕ && current->src[0]->op == GGML_OP_VIEW) {
                        current = current->src[0];
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ನಿಲ್ಲಿಸು;
                    }
                }

                // Assign all collected view inputs to node_inputs_views
                current_ಗಂಟು_info.ಗಂಟು_inputs_views[src_name] = view_chain;
            }
        }

        m_ಗಂಟು_info_list.push_back(current_ಗಂಟು_info);
    }
}

ಪೂರ್ಣಾಂಕ GgmlOvDecoder::compute_op_case(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) ಸ್ಥಿರಾಂಕ {
    ಪೂರ್ಣಾಂಕ op_case = 0;
    ಆಯ್ಕೆ (ಗಂಟು->op) {
    ಸ್ಥಿತಿ GGML_OP_RESHAPE: {
        ಆದರೆ (m_naive) {
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ವಯಂಚಾಲಿತ name = std::string(ಗಂಟು->name);
        ಸ್ವಯಂಚಾಲಿತ * src = ಗಂಟು->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        ತಾರ್ಕಿಕ recurrent_sequence = ಸುಳ್ಳು;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ಗ್ರಾಹಕ = m_cgraph->nodes[i];
            ಆದರೆ (ಗ್ರಾಹಕ->op == GGML_OP_MUL_MAT_ID && ಗ್ರಾಹಕ->src[1] == ಗಂಟು) {
                ಹಿಂತಿರುಗಿ 1;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗ್ರಾಹಕ->op == GGML_OP_SSM_CONV) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * concat = ಗ್ರಾಹಕ->src[0];
                ಆದರೆ (concat->op == GGML_OP_CONCAT) {
                    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == ಗಂಟು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗ್ರಾಹಕ->op == GGML_OP_UNARY && ggml_get_unary_op(ಗ್ರಾಹಕ) == GGML_UNARY_OP_SOFTPLUS) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * biased = ಗ್ರಾಹಕ->src[0];
                recurrent_sequence = biased->op == GGML_OP_ಕೂಡಿಸು && biased->src[0] == ಗಂಟು;
            }
        }
        ಆದರೆ (recurrent_sequence && ಗಂಟು->ne[0] == src->ne[0] && ಗಂಟು->ne[3] == 1) {
            ಹಿಂತಿರುಗಿ 6;
        }
        ಆದರೆ (ಗಂಟು->ne[0] == src->ne[0] && ಗಂಟು->ne[2] == 1 && ಗಂಟು->ne[3] == 1) {
            ಹಿಂತಿರುಗಿ 5;
        }
        ಆದರೆ (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == ಗಂಟು->ne[0] && src->src[0]->ne[1] == ಗಂಟು->ne[1]) {
            op_case = 4;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->ne[0] * ಗಂಟು->ne[1] == src->ne[0]) {
            op_case = 1;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src->ne[0] * src->ne[1] == ಗಂಟು->ne[0]) {
            op_case = 2;
            ಆದರೆ (src->ne[2] * src->ne[3] == ಗಂಟು->ne[1]) {
                op_case = 5;
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == ಗಂಟು->ne[1]) {
            op_case = 3;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (name.find("linear_attn_qkv_mixed") == 0 || name.find("alpha") == 0) {
            op_case = 6;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (name.find("linear_attn_out") == 0) {
            op_case = 7;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (name.find("state_predelta") == 0) {
            op_case = 8;
        }
        ಆದರೆ (op_case == 1 && m_is_stateful) {
            // Recurrent convolution and GDN gates retain their rank-4 layout.
            // The gathered states may come through a view of the one gather of all states (see build_rs).
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
            ತಾರ್ಕಿಕ recurrent = gather->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು && is_recurrent_cache(gather->src[0]);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ಗ್ರಾಹಕ = m_cgraph->nodes[i];
                ಆದರೆ (ಗ್ರಾಹಕ->op == GGML_OP_GATED_DELTA_NET) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j : {3, 4}) {
                        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * gate = ಗ್ರಾಹಕ->src[j];
                        ಆದರೆ (gate->op == GGML_OP_UNARY) {
                            gate = gate->src[0];
                        }
                        recurrent = recurrent || gate == ಗಂಟು;
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗ್ರಾಹಕ->op == GGML_OP_MUL) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 2; ++j) {
                        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * gate = ಗ್ರಾಹಕ->src[j];
                        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * norm = ಗ್ರಾಹಕ->src[1 - j];
                        ಆದರೆ (gate->op != GGML_OP_UNARY || gate->src[0] != ಗಂಟು) {
                            ಮುಂದುವರಿ;
                        }
                        ಆದರೆ (norm->op == GGML_OP_MUL) {
                            norm = norm->src[0];
                        }
                        recurrent = recurrent || (norm->op == GGML_OP_RMS_NORM && norm->src[0]->op == GGML_OP_VIEW &&
                                                   norm->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET);
                    }
                }
            }
            ಆದರೆ (recurrent) {
                op_case = 9;
            }
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ: {
        ಆದರೆ (ಗಂಟು->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(ಗಂಟು->view_src->name);
            ಪೂರ್ಣಾಂಕ layer = ಹೊರತೆಗೆ_layer_from_name(src_name).value();
            ಆದರೆ (ggml_is_contiguous(ಗಂಟು->src[0])) {
                // -  19: [    64,     8,   256,     1] VIEW            cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_k_l0                    [ 2,  1024, 1048576, 1048576]
                // -  20: [    64,   256,     8,     1] PERMUTE         cache_k_l0 (view) (permuted)  [ 2,  1024,   128, 1048576]
                //         [    64,     8,   256,     1]      0: VIEW     cache_k_l0 (view)             [ 2,   128,  1024, 1048576]
                ಆದರೆ (!is_swa_layer(layer)) {
                    op_case = 3;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    op_case = 4;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                // special case of cache v when `-fa off`
                // -  17: [   256,     8,    64,     1] VIEW            cache_v_l0 (view)             [ 2, 131072,  2048, 1048576]
                //         [   512,  1024,     1,     1]      0: NONE     cache_v_l0                   [ 2,  1024, 1048576, 1048576]
                // -  18: [   256,    64,     8,     1] PERMUTE         cache_v_l0 (view) (permuted)  [ 2,  2048, 131072, 1048576]
                //         [   256,     8,    64,     1]      0: VIEW     cache_v_l0 (view)            [ 2, 131072,  2048, 1048576]
                ಆದರೆ (!is_swa_layer(layer)) {
                    op_case = 5;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    op_case = 6;
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            // rope'ed query tensor
            op_case = 2;
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_MUL_MAT: {
        ಆದರೆ (ಗಂಟು->src[1]->op == GGML_OP_SOFT_MAX) {
            // In the case of `-fa off`, softmax is used, v_trans=true, the dynamic dim is ne[0] for cache_v
            op_case = 2;
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು: {
        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
        // when one gather covers all states (see build_rs).
        // op_case 1/2: active/extra rows of a multi-slot cache
        // op_case 3/4: active/extra rows of a single-slot cache
        ಆದರೆ (ಗಂಟು->src[0]->op == GGML_OP_RESHAPE && ಗಂಟು->src[0]->src[0] != ಶೂನ್ಯಸೂಚಕ &&
            is_recurrent_cache(ಗಂಟು->src[0]->src[0])) {
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ single_ಸ್ಥಾನ = ಗಂಟು->src[0]->src[0]->ne[1] == 1;
            ಆದರೆ (ಗಂಟು->src[1]->op == GGML_OP_VIEW) {
                op_case = (ಗಂಟು->src[1]->view_offs == 0 ? 1 : 2) + (single_ಸ್ಥಾನ ? 2 : 0);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (single_ಸ್ಥಾನ) {
                // a single-slot cache holds exactly the one state the gather selects
                op_case = 3;
            }
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_ROPE: {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode = ಗಂಟು->op_params[2];
        ಆಯ್ಕೆ (mode) {
        ಸ್ಥಿತಿ GGML_ROPE_TYPE_NEOX: {
            op_case = 1;
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_ROPE_TYPE_IMROPE: {
            op_case = 2;
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_ROPE_TYPE_VISION: {
            op_case = 3;
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_ROPE_TYPE_MROPE: {
            op_case = 4;
            ನಿಲ್ಲಿಸು;
        }
        ಪೂರ್ವನಿಯೋಜಿತ:
            op_case = 0;
            ನಿಲ್ಲಿಸು;
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_VIEW: {
        ಆದರೆ (!m_model_params.has_rs_rollback && ಗಂಟು->src[0] != ಶೂನ್ಯಸೂಚಕ &&
            ಗಂಟು->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // The GDN translator publishes native attention/state outputs under these VIEW names.
            op_case = 2;
            ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (m_is_static && ಗಂಟು->src[0] != ಶೂನ್ಯಸೂಚಕ &&
            (ಗಂಟು->src[0]->op == GGML_OP_GATED_DELTA_NET || ಗಂಟು->src[0]->op == GGML_OP_CONCAT)) {
            // VIEW slicing a GATED_DELTA_NET combined [attn|state] output, or the conv_input
            // CONCAT. The consuming CPY/RMS_NORM op recovers the true window at runtime via
            // ssm_state_size / the fixed conv kernel width, so this VIEW must stay an identity
            // pass-through of the full source here too (it already is on the dynamic path);
            // otherwise the generic static-mode Slice below would bake in the *captured*
            // cgraph's token count, which is wrong once the compiled static model runs with a
            // different token count (prefill chunk size or 1).
            op_case = 1;
            ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (ಗಂಟು->src[0]->op == GGML_OP_VIEW) {
            ಸ್ವಯಂಚಾಲಿತ * src = ಗಂಟು->src[0];
            ಆದರೆ (ggml_nelements(ಗಂಟು) != ggml_nelements(src)) {
                // throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 0;
            ಆದರೆ (m_model_is_splitted && m_model_inputs.find(get_tensor_ov_name(m_cgraph, src)) != m_model_inputs.end()) {
                op_case = 0;
            }
        }
        {
            ಸ್ವಯಂಚಾಲಿತ * src = ಗಂಟು->src[0];
            ಆದರೆ (ggml_nelements(ಗಂಟು) != ggml_nelements(src)) {
                // Case 4: select one slice on src dim1 (via view offset), keep src dim2 as output dim1.
                // Typical pattern:
                //   src: ne=[N, M, K, 1], nb=[b0, b1, b2, b3]
                //   dst: ne=[N, K, 1, 1], nb=[b0, b2, b3, b3]
                ಆದರೆ (ಗಂಟು->ne[0] == src->ne[0] && ಗಂಟು->ne[1] == src->ne[2] && ಗಂಟು->ne[2] == 1 &&
                    ಗಂಟು->nb[0] == src->nb[0] && ಗಂಟು->nb[1] == src->nb[2] && src->ne[1] > 1) {
                    op_case = 0;
                    ನಿಲ್ಲಿಸು;
                }

                // General case 3: shape differs from source (one or more dims) and is handled as VIEW slicing.
                ಪೂರ್ಣಾಂಕ diff_ಎಣಿಕೆ = 0;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
                    ಆದರೆ (ಗಂಟು->ne[i] != src->ne[i]) {
                        diff_ಎಣಿಕೆ++;
                    }
                    // if node ne[i] > src ne[i], case = 0
                    ಆದರೆ (ಗಂಟು->ne[i] > src->ne[i]) {
                        ಹಿಂತಿರುಗಿ 0;
                    }
                }
                ಆದರೆ (diff_ಎಣಿಕೆ >= 1) {
                    op_case = 0;
                }
            }
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_RMS_NORM: {
        ಆದರೆ (ಗಂಟು->src[0]->op == GGML_OP_VIEW) {
            ಆದರೆ (is_same_shape(ಗಂಟು->src[0]->src[0], ಗಂಟು->src[0])) {
                op_case = 1;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!m_model_params.has_rs_rollback &&
                       ಗಂಟು->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_POOL_2D: {
        ಸ್ಥಿರಾಂಕ ggml_op_pool pool_ಬಹುಲಕ = static_cast<ggml_op_pool>(ಗಂಟು->op_params[0]);
        ಆಯ್ಕೆ (pool_ಬಹುಲಕ) {
        ಸ್ಥಿತಿ GGML_OP_POOL_MAX: {
            op_case = 1;
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_OP_POOL_AVG: {
            op_case = 2;
            ನಿಲ್ಲಿಸು;
        }
        ಪೂರ್ವನಿಯೋಜಿತ:
            op_case = 0;
            ನಿಲ್ಲಿಸು;
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_UPSCALE: {
        ಸ್ಥಿರಾಂಕ int32_t ಬಹುಲಕ_flags = ಗಂಟು->op_params[0];
        ಸ್ಥಿರಾಂಕ ggml_scale_ಬಹುಲಕ scale_ಬಹುಲಕ = static_cast<ggml_scale_ಬಹುಲಕ>(ಬಹುಲಕ_flags & 0xFF);
        ಆಯ್ಕೆ (scale_ಬಹುಲಕ) {
        ಸ್ಥಿತಿ GGML_SCALE_ಬಹುಲಕ_ಹತ್ತಿರದ: {
            op_case = 1;
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_SCALE_ಬಹುಲಕ_BILINEAR: {
            op_case = 2;
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_SCALE_ಬಹುಲಕ_BICUBIC: {
            op_case = 3;
            ನಿಲ್ಲಿಸು;
        }
        ಪೂರ್ವನಿಯೋಜಿತ:
            op_case = 0;
            ನಿಲ್ಲಿಸು;
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_CPY: {
        ಆದರೆ (ಗಂಟು->src[0]->op == GGML_OP_VIEW) {
            ಆದರೆ (ಗಂಟು->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                ಆದರೆ (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_ತುಂಬಿದ_single_ಸ್ಥಾನ_writeback(ಗಂಟು) ? 7 : 10;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    op_case = 1;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (GgmlOvDecoder::is_conv_state_writeback(ಗಂಟು)) {
                op_case = is_ತುಂಬಿದ_single_ಸ್ಥಾನ_writeback(ಗಂಟು) ? 8 : 2;
                ನಿಲ್ಲಿಸು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_conv_ಸ್ಥಿತಿಗಳು_all_tensor(ಗಂಟು->view_src) && ಗಂಟು->src[1] != ಶೂನ್ಯಸೂಚಕ &&
                       ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src == ಗಂಟು->view_src) {
                op_case = 4;
                ನಿಲ್ಲಿಸು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->src[0]->src[0]->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು && ಗಂಟು->src[1] != ಶೂನ್ಯಸೂಚಕ &&
                       ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src != ಶೂನ್ಯಸೂಚಕ &&
                       is_recurrent_cache(ಗಂಟು->src[1]->view_src) && ಗಂಟು->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->src[0]->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು && ಗಂಟು->src[1] != ಶೂನ್ಯಸೂಚಕ &&
                   ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src != ಶೂನ್ಯಸೂಚಕ &&
                   is_recurrent_cache(ಗಂಟು->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = ಗಂಟು->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->src[1] != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src != ಶೂನ್ಯಸೂಚಕ) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            ಸ್ಥಿರಾಂಕ ggml_tensor * kv_buf = ಗಂಟು->src[1]->view_src;
            ಆದರೆ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m_cgraph->n_nodes; i++) {
                    ಸ್ಥಿರಾಂಕ ggml_tensor * n = m_cgraph->nodes[i];
                    ಆದರೆ (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        ಮುಂದುವರಿ;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    ಆದರೆ ((n->src[1] != ಶೂನ್ಯಸೂಚಕ && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != ಶೂನ್ಯಸೂಚಕ && n->src[2]->view_src == kv_buf)) {
                        ಆದರೆ (n->src[3] != ಶೂನ್ಯಸೂಚಕ) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        ನಿಲ್ಲಿಸು;
                    }
                }
            }
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು: {
        ಆದರೆ (is_moe_expert_ಮೊತ್ತ_ಕೂಡಿಸು(ಗಂಟು)) {
            // Outermost ADD of a MoE expert-plane sum chain: translated as a single
            // ReduceSum over the base tensor instead of N-1 chained Adds over N Slices.
            op_case = 1;
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_SCALE: {
        ಆದರೆ (ಗಂಟು->view_src && ಗಂಟು->ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_ANY) {
            op_case = ಗಂಟು->view_src->ne[1] == 1 ? 2 : 1;
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_L2_NORM: {
        ಆದರೆ (std::string(ಗಂಟು->name).find("predelta") != std::string::npos) {
            op_case = 1;
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT: {
        ಆದರೆ (ಗಂಟು->src[1] != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src != ಶೂನ್ಯಸೂಚಕ) {
            ಸ್ಥಿರಾಂಕ ggml_tensor * kv_buf = ಗಂಟು->src[1]->view_src;
            ಆದರೆ (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = (ಗಂಟು->src[3] != ಶೂನ್ಯಸೂಚಕ) ? 1 : 2;
            }
        }
        ನಿಲ್ಲಿಸು;
    }
    ಪೂರ್ವನಿಯೋಜಿತ:
        ನಿಲ್ಲಿಸು;
    }
    ಹಿಂತಿರುಗಿ op_case;
}

std::optional<ಪೂರ್ಣಾಂಕ> ಹೊರತೆಗೆ_layer_from_name(ಸ್ಥಿರಾಂಕ std::string & name) {
    size_t pos1 = name.find("_l");
    ಆದರೆ (pos1 == std::string::npos) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    ಆದರೆ (pos2 == std::string::npos) {
        pos2 = name.ಉದ್ದಳತೆ();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    ಪೂರ್ಣಾಂಕ layer = std::stoi(layer_str);
    ಹಿಂತಿರುಗಿ layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ get_swa_ಕಿಟಕಿ_from_ಮುಖವಾಡ(ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ) {
    ಆದರೆ (ಮುಖವಾಡ->ದತ್ತಾಂಶ == ಶೂನ್ಯಸೂಚಕ || !ggml_backend_ಬಫರ್_is_host(ಮುಖವಾಡ->ಬಫರ್)) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆದರೆ (ಮುಖವಾಡ->type != GGML_TYPE_F16 && ಮುಖವಾಡ->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t n_kv = ಮುಖವಾಡ->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = ಮುಖವಾಡ->ne[1];
    int64_t ಕಿಟಕಿ = 0;

    ಗಾಗಿ (int64_t r = 0; r < n_ಟೋಕನುಗಳು; r++) {
        int64_t kept = 0;
        ಗಾಗಿ (int64_t c = 0; c < n_kv; c++) {
            ಸ್ಥಿರಾಂಕ size_t i = (size_t) r * n_kv + c;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = ಮುಖವಾಡ->type == GGML_TYPE_F16 ? ggml_fp16_ಗಮ್ಯ_fp32(((ಸ್ಥಿರಾಂಕ ggml_fp16_t *) ಮುಖವಾಡ->ದತ್ತಾಂಶ)[i]) :
                                                          ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ)[i];
            ಆದರೆ (v > -INFINITY) {
                kept++;
            }
        }
        ಕಿಟಕಿ = std::max(ಕಿಟಕಿ, kept);
    }

    ಹಿಂತಿರುಗಿ ಕಿಟಕಿ > 0 ? (ಪೂರ್ಣಾಂಕ) ಕಿಟಕಿ : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, ತಾರ್ಕಿಕ is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    ಸ್ವಯಂಚಾಲಿತ get_attention_ನಮೂನೆ_case = [](ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) -> ಪೂರ್ಣಾಂಕ {
        ಆದರೆ (ಗಂಟು == ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ -1;
        }

        ಆಯ್ಕೆ (ಗಂಟು->op) {
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            ಆದರೆ (ಗಂಟು->src[0] == ಶೂನ್ಯಸೂಚಕ || ಗಂಟು->src[1] == ಶೂನ್ಯಸೂಚಕ) {
                ಹಿಂತಿರುಗಿ -1;
            }
            ಆಯ್ಕೆ (ಗಂಟು->src[1]->op) {
            ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                ಆದರೆ (ಗಂಟು->src[3] != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->src[1]->src[0] != ಶೂನ್ಯಸೂಚಕ &&
                    ಗಂಟು->src[1]->src[0]->op == GGML_OP_VIEW) {
                    ಹಿಂತಿರುಗಿ 0;
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                ಆದರೆ (ಗಂಟು->src[3] != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->src[1]->src[0] != ಶೂನ್ಯಸೂಚಕ &&
                    ಗಂಟು->src[1]->src[0]->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ && ಗಂಟು->src[1]->src[0]->src[0] != ಶೂನ್ಯಸೂಚಕ &&
                    ಗಂಟು->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    ಹಿಂತಿರುಗಿ 1;
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                ಆದರೆ (ಗಂಟು->src[1]->view_src != ಶೂನ್ಯಸೂಚಕ) {
                    ಆದರೆ (ಗಂಟು->src[3] != ಶೂನ್ಯಸೂಚಕ) {
                        ಹಿಂತಿರುಗಿ 4;  // decoder self-attention
                    }
                    ಹಿಂತಿರುಗಿ 5;      // cross-attention or encoder self-attention
                }
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            ಆದರೆ (ಗಂಟು->src[0] != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->src[1] != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->src[0]->op == GGML_OP_MUL_MAT &&
                ಗಂಟು->src[0]->src[0] != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->src[0]->src[1] != ಶೂನ್ಯಸೂಚಕ &&
                ಗಂಟು->src[0]->src[0]->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ && ಗಂಟು->src[0]->src[0]->src[0] != ಶೂನ್ಯಸೂಚಕ &&
                ಗಂಟು->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                ಹಿಂತಿರುಗಿ 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            ಆದರೆ (ಗಂಟು->src[0]->op == GGML_OP_ಕೂಡಿಸು && ಗಂಟು->src[0]->src[0] != ಶೂನ್ಯಸೂಚಕ &&
                ಗಂಟು->src[0]->src[0]->op == GGML_OP_MUL_MAT && ಗಂಟು->src[0]->src[0]->src[0] != ಶೂನ್ಯಸೂಚಕ &&
                ಗಂಟು->src[0]->src[0]->src[0]->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ) {
                ಹಿಂತಿರುಗಿ 3;
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
        }

        ಹಿಂತಿರುಗಿ -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    ಸ್ವಯಂಚಾಲಿತ get_attention_op_ಮುಖವಾಡ = [&get_attention_ನಮೂನೆ_case](ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) -> ಸ್ಥಿರಾಂಕ ggml_tensor * {
        ಆಯ್ಕೆ (get_attention_ನಮೂನೆ_case(ಗಂಟು)) {
        ಸ್ಥಿತಿ 0:
        ಸ್ಥಿತಿ 1:
            ಹಿಂತಿರುಗಿ ಗಂಟು->src[3];
        ಸ್ಥಿತಿ 2:
        ಸ್ಥಿತಿ 3:
            ಹಿಂತಿರುಗಿ ಗಂಟು->src[1];
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
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
        std::map<ಪೂರ್ಣಾಂಕ, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<ಪೂರ್ಣಾಂಕ, ಸ್ಥಿರಾಂಕ ggml_tensor *> layer_ಮುಖವಾಡ;            // layer -> mask it consumes
        int64_t max_extent = 0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
            ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ = get_attention_op_ಮುಖವಾಡ(cgraph->nodes[i]);
            ಆದರೆ (ಮುಖವಾಡ == ಶೂನ್ಯಸೂಚಕ) {
                ಮುಂದುವರಿ;
            }
            ಸ್ಥಿರಾಂಕ ggml_tensor * cache_k_ಕ್ರಮಬದಲಾವಣೆ = ಶೂನ್ಯಸೂಚಕ;
            ಆಯ್ಕೆ (get_attention_ನಮೂನೆ_case(cgraph->nodes[i])) {
            ಸ್ಥಿತಿ 0:  cache_k_ಕ್ರಮಬದಲಾವಣೆ = cgraph->nodes[i]->src[1];                     ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 1:  cache_k_ಕ್ರಮಬದಲಾವಣೆ = cgraph->nodes[i]->src[1]->src[0];             ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 2:  cache_k_ಕ್ರಮಬದಲಾವಣೆ = cgraph->nodes[i]->src[0]->src[0];             ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ: cache_k_ಕ್ರಮಬದಲಾವಣೆ = cgraph->nodes[i]->src[0]->src[0]->src[0];     ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿರಾಂಕ ggml_tensor * cache_k_view = cache_k_ಕ್ರಮಬದಲಾವಣೆ->src[0];
            ಆದರೆ (cache_k_view->op != GGML_OP_VIEW) {
                ಮುಂದುವರಿ;
            }
            ಸ್ಥಿರಾಂಕ ggml_tensor * ಎಲೆ = cache_k_view->src[0];
            ಸ್ವಯಂಚಾಲಿತ layer = ಹೊರತೆಗೆ_layer_from_name(ಎಲೆ->name);
            ಆದರೆ (!layer.has_value()) {
                ಮುಂದುವರಿ;
            }
            layer_extent[layer.value()] = ಎಲೆ->ne[1];
            layer_ಮುಖವಾಡ[layer.value()] = ಮುಖವಾಡ;
            max_extent = std::max(max_extent, ಎಲೆ->ne[1]);
        }

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & [layer, extent] : layer_extent) {
            ಆದರೆ (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                ಆದರೆ (model_params.swa_ಮುಖವಾಡ == ಶೂನ್ಯಸೂಚಕ) {
                    model_params.swa_ಮುಖವಾಡ = layer_ಮುಖವಾಡ[layer];
                }
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string ಪ್ರತಿ_layer;
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & [layer, extent] : layer_extent) {
                ಪ್ರತಿ_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (ದೀರ್ಘ) max_extent, model_params.swa_layers.size(), ಪ್ರತಿ_layer.c_str());
        }
    }

    ತಾರ್ಕಿಕ rope_ನೋಡಿದ = ಸುಳ್ಳು;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = cgraph->nodes[i];
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ attention_ನಮೂನೆ_case = get_attention_ನಮೂನೆ_case(ಗಂಟು);
        ಆದರೆ (attention_ನಮೂನೆ_case != -1) {
            ggml_tensor * cache_k_ಕ್ರಮಬದಲಾವಣೆ = ಶೂನ್ಯಸೂಚಕ;
            ggml_tensor * ಮುಖವಾಡ = ಶೂನ್ಯಸೂಚಕ;

            ಆಯ್ಕೆ (attention_ನಮೂನೆ_case) {
            ಸ್ಥಿತಿ 0:
                cache_k_ಕ್ರಮಬದಲಾವಣೆ = ಗಂಟು->src[1];
                ಮುಖವಾಡ = ಗಂಟು->src[3];
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 1:
                cache_k_ಕ್ರಮಬದಲಾವಣೆ = ಗಂಟು->src[1]->src[0];
                ಮುಖವಾಡ = ಗಂಟು->src[3];
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 2:
                cache_k_ಕ್ರಮಬದಲಾವಣೆ = ಗಂಟು->src[0]->src[0];
                ಮುಖವಾಡ = ಗಂಟು->src[1];
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 3:
                cache_k_ಕ್ರಮಬದಲಾವಣೆ = ಗಂಟು->src[0]->src[0]->src[0];
                ಮುಖವಾಡ = ಗಂಟು->src[1];
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ 4:
            ಸ್ಥಿತಿ 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                ಸ್ವಯಂಚಾಲಿತ * cache_k_view = ಗಂಟು->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_ಪ್ರತಿ_seq = ಗಂಟು->src[0]->ne[1];
                ಆದರೆ (attention_ನಮೂನೆ_case == 4) {
                    compute_params.attention_ಗಾತ್ರ = cache_k_view->ne[1];
                } ಇಲ್ಲದಿದ್ದರೆ {
                    compute_params.attention_ಗಾತ್ರ_static = cache_k_view->ne[1];
                }
                ಮುಂದುವರಿ;
            }
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
            }

            ಪ್ರತಿಜ್ಞಾ(cache_k_ಕ್ರಮಬದಲಾವಣೆ != ಶೂನ್ಯಸೂಚಕ);

            model_params.ತಲೆ_ಗಾತ್ರ = cache_k_ಕ್ರಮಬದಲಾವಣೆ->ne[0];
            model_params.n_heads_kv = cache_k_ಕ್ರಮಬದಲಾವಣೆ->ne[2];
            compute_params.ಒಳಹರಿವು_len = ಗಂಟು->src[0]->ne[1];
            compute_params.token_len_ಪ್ರತಿ_seq = ಗಂಟು->src[0]->ne[1];

            ಸ್ವಯಂಚಾಲಿತ * cache_k_view = cache_k_ಕ್ರಮಬದಲಾವಣೆ->src[0];
            ಆದರೆ (cache_k_view->op != GGML_OP_VIEW || ಮುಖವಾಡ == ಶೂನ್ಯಸೂಚಕ) {
                ಮುಂದುವರಿ;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            ಪೂರ್ಣಾಂಕ layer = ಹೊರತೆಗೆ_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_ಬಫರ್_ctx_id = ggml_backend_openvino_ಬಫರ್_get_ctx_id(cache_k->ಬಫರ್);
            model_params.n_heads_kv_ಪ್ರತಿ_layer[layer] = cache_k_ಕ್ರಮಬದಲಾವಣೆ->ne[2];
            ಆದರೆ (layer_is_swa) {
                model_params.ctx_ಪ್ರತಿ_seq_swa = cache_k->ne[1];
            } ಇಲ್ಲದಿದ್ದರೆ {
                model_params.ctx_ಪ್ರತಿ_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_ಸಕ್ರಿಯ = ಮುಖವಾಡ->ne[3];
            ಸ್ವಯಂಚಾಲಿತ seq_ಗಾತ್ರ = cache_k->ne[0] * cache_k->ne[1] * ggml_type_ಗಾತ್ರ(cache_k->type);
            size_t ಸ್ಥಾನಾಂತರ;
            ಸ್ಮರಣೆ_ನಕಲು(&ಸ್ಥಾನಾಂತರ, cache_k_view->op_params, ಗಾತ್ರ(size_t));
            compute_params.seq_ಸಕ್ರಿಯ_ಆರಂಭ = ಸ್ಥಾನಾಂತರ / seq_ಗಾತ್ರ;

            ಆದರೆ (layer_is_swa) {
                compute_params.attention_ಗಾತ್ರ_swa = ಮುಖವಾಡ->ne[0];
                compute_params.swa_ಕಿಟಕಿ = get_swa_ಕಿಟಕಿ_from_ಮುಖವಾಡ(ಮುಖವಾಡ);
            } ಇಲ್ಲದಿದ್ದರೆ {
                compute_params.attention_ಗಾತ್ರ = ಮುಖವಾಡ->ne[0];
            }
            ಆದರೆ (is_static) {
                compute_params.attention_ಗಾತ್ರ = model_params.ctx_ಪ್ರತಿ_seq;
                compute_params.attention_ಗಾತ್ರ_swa = model_params.ctx_ಪ್ರತಿ_seq_swa;
                compute_params.token_len_ಪ್ರತಿ_seq = 1;
            }
        }

        ಆದರೆ (ಗಂಟು->op == GGML_OP_MUL_MAT && ಗಂಟು->src[0]->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ &&
            ಗಂಟು->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(ಗಂಟು->src[0]->view_src, ಗಂಟು->view_src)) {
            ಆದರೆ (ಗಂಟು->src[1]->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ && ಗಂಟು->src[1]->src[0]->op == GGML_OP_VIEW &&
                ಗಂಟು->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_ಗಾತ್ರ = ಗಂಟು->ne[0];
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        ಆದರೆ (ಗಂಟು->op == GGML_OP_TRANSPOSE && ಗಂಟು->src[0]->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ &&
            ಗಂಟು->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_ಗಾತ್ರ = ಗಂಟು->ne[0];
            ಆದರೆ (is_static) {
                compute_params.attention_ಗಾತ್ರ = model_params.ctx_ಪ್ರತಿ_seq;
            }
        }
        ಆದರೆ (ಗಂಟು->op == GGML_OP_ROPE) {
            ಆದರೆ (compute_params.token_len_ಪ್ರತಿ_seq == -1 && ಗಂಟು->src[1] != ಶೂನ್ಯಸೂಚಕ) {
                compute_params.token_len_ಪ್ರತಿ_seq = ggml_nelements(ಗಂಟು->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(ಗಾತ್ರ(model_params.rope_params) == ಗಾತ್ರ(int32_t) * 16, "rope_params size");
            ಆದರೆ (!rope_ನೋಡಿದ) {
                ಸ್ಮರಣೆ_ನಕಲು(model_params.rope_params, ಗಂಟು->op_params, ಗಾತ್ರ(int32_t) * 16);
                rope_ನೋಡಿದ = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (memcmp(model_params.rope_params, ಗಂಟು->op_params, ಗಾತ್ರ(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = ನಿಜ;
            }
        }
        ಆದರೆ (ಗಂಟು->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_ಗಾತ್ರ = ಗಂಟು->src[0]->ne[0];
        }
        ಆದರೆ (ಗಂಟು->op == GGML_OP_SCALE && ಗಂಟು->view_src != ಶೂನ್ಯಸೂಚಕ && is_recurrent_cache(ಗಂಟು->view_src)) {
            ಆದರೆ (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = ಗಂಟು->view_src->ne[1];
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ASSERT(model_params.n_rs_slots == ಗಂಟು->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(ಗಂಟು) / ಗಂಟು->view_src->ne[0];
            compute_params.cache_rs_reset_idx = ಗಂಟು->src[0]->view_offs / ಗಂಟು->view_src->ne[0];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        ಆದರೆ (ಗಂಟು->op == GGML_OP_CPY && ಗಂಟು->view_src != ಶೂನ್ಯಸೂಚಕ && is_recurrent_cache(ಗಂಟು->view_src) &&
            ಗಂಟು->src[1] != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src == ಗಂಟು->view_src) {
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_conv = is_conv_state_writeback(ಗಂಟು);
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_gdn = ಗಂಟು->src[0]->op == GGML_OP_VIEW && ಗಂಟು->src[0]->src[0] != ಶೂನ್ಯಸೂಚಕ &&
                                ಗಂಟು->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_extra = ಗಂಟು->src[0]->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_gdn_rollback = is_gdn && is_same_shape(ಗಂಟು->src[0], ಗಂಟು->src[1]);

            ಸ್ಥಿರಾಂಕ ggml_tensor * dest_view = ಗಂಟು->src[1];
            ಸ್ಥಿರಾಂಕ ggml_tensor * cache = ಗಂಟು->view_src;
            ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = cache->ne[0] * ggml_type_ಗಾತ್ರ(cache->type);
            ಆದರೆ (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = ನಿಜ;
            }
            ಆದರೆ (ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು > 0 && (is_conv || is_gdn || is_extra) && !is_ತುಂಬಿದ_single_ಸ್ಥಾನ_writeback(ಗಂಟು)) {
                ComputeParams::RsWriteback writeback;
                writeback.ಸ್ಥಾನ_begin = (ಪೂರ್ಣಾಂಕ) (dest_view->view_offs / ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು);
                ಆದರೆ (is_conv) {
                    writeback.src_begin = (ಪೂರ್ಣಾಂಕ) (ಗಂಟು->src[0]->view_offs / ಗಂಟು->src[0]->view_src->nb[0]);
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_gdn_rollback) {
                    writeback.src_begin = (ಪೂರ್ಣಾಂಕ) (ಗಂಟು->src[0]->view_offs / ಗಂಟು->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, ಗಂಟು)] = writeback;
            }
            ಆದರೆ ((is_conv || is_gdn) && !is_ತುಂಬಿದ_single_ಸ್ಥಾನ_writeback(ಗಂಟು)) {
                compute_params.s_copy_ಸಕ್ರಿಯ_ಸ್ಥಾನ_len = (ಪೂರ್ಣಾಂಕ) dest_view->ne[1];
            }
        }
    }
    ಆದರೆ (model_params.n_heads_kv == -1) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = cgraph->nodes[i];
            ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ = ಶೂನ್ಯಸೂಚಕ;
            ಆದರೆ (ಗಂಟು->op == GGML_OP_SOFT_MAX) {
                ಮುಖವಾಡ = ಗಂಟು->src[1];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->op == GGML_OP_FLASH_ATTN_EXT) {
                ಮುಖವಾಡ = ಗಂಟು->src[3];
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (ಮುಖವಾಡ == ಶೂನ್ಯಸೂಚಕ || ಮುಖವಾಡ->op != GGML_OP_NONE || !(ಮುಖವಾಡ->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) ||
                ಗಂಟು->src[0] == ಶೂನ್ಯಸೂಚಕ) {
                ಮುಂದುವರಿ;
            }
            model_params.is_cacheless_attn = ನಿಜ;
            model_params.n_seq = 1;
            model_params.ctx_ಪ್ರತಿ_seq = ಮುಖವಾಡ->ne[0];
            compute_params.ಒಳಹರಿವು_len = ಗಂಟು->src[0]->ne[1];
            compute_params.token_len_ಪ್ರತಿ_seq = compute_params.ಒಳಹರಿವು_len;
            ನಿಲ್ಲಿಸು;
        }
    }

    ಸ್ವಯಂಚಾಲಿತ * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    ಆದರೆ (model_params.is_cacheless_attn) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = cgraph->nodes[i];
            ಆದರೆ (ಗಂಟು->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು && is_output_idx(ಗಂಟು->src[1], ಗಂಟು)) {
                compute_params.output_len = ಗಂಟು->src[1]->ne[0];
                ನಿಲ್ಲಿಸು;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    ಆದರೆ (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_ಪ್ರತಿ_seq * model_params.n_seq;
    ಹಿಂತಿರುಗಿ {model_params, compute_params};
}

ಶೂನ್ಯ GgmlOvDecoder::validate_cgraph() ಸ್ಥಿರಾಂಕ {
    ಆದರೆ (m_model_params.n_seq > 1 && m_is_static == ನಿಜ) {
        ಎಸೆ std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_ನಕ್ಷೆ_ಒಳಹರಿವು_shape(ಸ್ಥಿರಾಂಕ ggml_tensor * op,
                                                      ಸ್ಥಿರಾಂಕ ggml_tensor * ಒಳಹರಿವು,
                                                      ಪೂರ್ಣಾಂಕ dynamic_dim_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಆದರೆ (m_naive) {
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು != ಶೂನ್ಯಸೂಚಕ ? ov::PartialShape{get_shape(ಒಳಹರಿವು)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape ಒಳಹರಿವು_shape;

    ಆದರೆ (is_inp_tok(ಒಳಹರಿವು, op) || is_inp_pos(ಒಳಹರಿವು, op)) {
        // tokens or positions
        ಪೂರ್ಣಾಂಕ len = m_is_static ? (m_is_prefill ? m_prefill_chunk_ಗಾತ್ರ : 1) : -1;
        ಆದರೆ (m_is_static && is_inp_pos(ಒಳಹರಿವು, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            len *= get_inp_pos_n_planes(op);
        }
        ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, 1, len};

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_output_idx(ಒಳಹರಿವು, op)) {
        // output index
        ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_inp_ಸರಾಸರಿ(ಒಳಹರಿವು, op)) {
        ಒಳಹರಿವು_shape = m_is_static ? ov::PartialShape{1, 1, ಒಳಹರಿವು->ne[1], m_prefill_chunk_ಗಾತ್ರ} :
                                    ov::PartialShape{1, 1, -1, -1};

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_inp_scale_ಅಡ್ಡಸಾಲುಗಳು(ಒಳಹರಿವು, op)) {
        ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_ಗಾತ್ರ : 1) : -1, 1};

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_inp_ಮುಖವಾಡ(ಒಳಹರಿವು, op)) {
        //ಮುಖವಾಡ
        ಆದರೆ (m_is_static) {
            ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_ಗಾತ್ರ : 1, m_model_params.ctx};
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_is_stateful) {
            ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, -1, -1};
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಒಳಹರಿವು_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_recurrent_cache(ಒಳಹರಿವು)) {
        ಒಳಹರಿವು_shape = ov::PartialShape{get_shape(ಒಳಹರಿವು)};
        ಆದರೆ (!m_is_static && !m_is_stateful && ಒಳಹರಿವು->ne[1] > 1) {
            ಒಳಹರಿವು_shape[2] = -1;
        }

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_kvcache(ಒಳಹರಿವು, op)) {
        // kvcache
        ಒಳಹರಿವು_shape = ov::PartialShape{get_shape(ಒಳಹರಿವು)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_flat_kv = (ಒಳಹರಿವು->ne[1] == 1 && ಒಳಹರಿವು->ne[2] == 1 && ಒಳಹರಿವು->ne[3] == 1);
        ಆದರೆ (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            ಒಳಹರಿವು_shape[2] = -1;
        }
        ಆದರೆ (is_stateful() && !is_flat_kv) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            ಪ್ರತಿಜ್ಞಾ(ಒಳಹರಿವು_shape.size() == 4 && ಒಳಹರಿವು_shape[0] == 1 && ಒಳಹರಿವು_shape[1] == 1 &&
                   ಒಳಹರಿವು_shape[2].is_dynamic() && ಒಳಹರಿವು_shape[3].is_static());
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_heads_kv = get_n_heads_kv_for_tensor(ಒಳಹರಿವು);
            ಪ್ರತಿಜ್ಞಾ(n_heads_kv > 0 && ಒಳಹರಿವು_shape[3].get_ಉದ್ದಳತೆ() % n_heads_kv == 0);
            ಸ್ಥಿರಾಂಕ int64_t combined_dim = ಒಳಹರಿವು_shape[3].get_ಉದ್ದಳತೆ();  // n_heads_kv * head_size
            ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_ಗಾತ್ರ = combined_dim / n_heads_kv;
            ಒಳಹರಿವು_shape = {ಒಳಹರಿವು_shape[0], ov::Dimension::dynamic(), n_heads_kv, ತಲೆ_ಗಾತ್ರ};
        }

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_kv_idx(ಒಳಹರಿವು, op)) {
        // kv update index
        ಪೂರ್ಣಾಂಕ len = m_is_static ? (m_is_prefill ? m_prefill_chunk_ಗಾತ್ರ : 1) : -1;
        ಒಳಹರಿವು_shape = ov::PartialShape{1, 1, 1, len};

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_inp_s_copy(ಒಳಹರಿವು, op) || is_s_copy_ಎಲೆ(ಒಳಹರಿವು)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        ಒಳಹರಿವು_shape = m_is_static ? ov::PartialShape{get_shape(ಒಳಹರಿವು)} : ov::PartialShape{1, 1, 1, -1};

    } ಇಲ್ಲದಿದ್ದರೆ {
        ಒಳಹರಿವು_shape = ov::PartialShape{get_shape(ಒಳಹರಿವು)};
    }
    ಆದರೆ (dynamic_dim_ಸೂಚ್ಯಂಕ != -1 && m_model_is_splitted) {
        ಒಳಹರಿವು_shape[3 - dynamic_dim_ಸೂಚ್ಯಂಕ] = -1;
    }
    ಆದರೆ (op->op == GGML_OP_SOFT_MAX && op->src[1] != ಶೂನ್ಯಸೂಚಕ && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು && op->src[1] == ಒಳಹರಿವು) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        ಆದರೆ (m_is_static) {
            ಸ್ಥಿರಾಂಕ int64_t seq_ಸಕ್ರಿಯ = m_is_prefill ? m_prefill_chunk_ಗಾತ್ರ : 1;
            ಒಳಹರಿವು_shape[2] = seq_ಸಕ್ರಿಯ;
            ಒಳಹರಿವು_shape[3] = seq_ಸಕ್ರಿಯ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಒಳಹರಿವು_shape[2] = -1;
            ಒಳಹರಿವು_shape[3] = -1;
        }
    }
    ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_shape;
}

ತಾರ್ಕಿಕ GgmlOvDecoder::is_s_copy_ಎಲೆ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) ಸ್ಥಿರಾಂಕ {
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ || tensor->op != GGML_OP_NONE || m_cgraph == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m_cgraph->n_nodes; i++) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು = m_cgraph->nodes[i];
        ಆದರೆ (ಗಂಟು->op != GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು || ಗಂಟು->src[0] == ಶೂನ್ಯಸೂಚಕ || ಗಂಟು->src[1] == ಶೂನ್ಯಸೂಚಕ) {
            ಮುಂದುವರಿ;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        ಸ್ಥಿರಾಂಕ ggml_tensor * idx = ಗಂಟು->src[1];
        ಆಗ (idx != ಶೂನ್ಯಸೂಚಕ && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        ಆದರೆ (idx != tensor) {
            ಮುಂದುವರಿ;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        ಸ್ಥಿರಾಂಕ ggml_tensor * ದತ್ತಾಂಶ = ಗಂಟು->src[0];
        ಆಗ (ದತ್ತಾಂಶ != ಶೂನ್ಯಸೂಚಕ && (ದತ್ತಾಂಶ->op == GGML_OP_VIEW || ದತ್ತಾಂಶ->op == GGML_OP_RESHAPE)) {
            ದತ್ತಾಂಶ = ದತ್ತಾಂಶ->src[0];
        }
        ಆದರೆ (ದತ್ತಾಂಶ != ಶೂನ್ಯಸೂಚಕ && is_recurrent_cache(ದತ್ತಾಂಶ)) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಶೂನ್ಯ GgmlOvDecoder::ಕೂಡಿಸು_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    ಸ್ವಯಂಚಾಲಿತ create_1d_ಒಳಹರಿವು = [ಇದು](ಸ್ಥಿರಾಂಕ std::string & name, int64_t value, ತಾರ್ಕಿಕ force_parameter = ಸುಳ್ಳು) {
        m_model_extra_inputs[name] = {ov::element::i64, ov::Shape{1}, value, force_parameter || !m_is_static};
    };

    ಆದರೆ (m_compute_params.attention_ಗಾತ್ರ != -1) {
        create_1d_ಒಳಹರಿವು("attention_size", m_compute_params.attention_ಗಾತ್ರ);
    }
    ಆದರೆ (m_compute_params.attention_ಗಾತ್ರ_static != -1) {
        create_1d_ಒಳಹರಿವು("attention_size_static", m_compute_params.attention_ಗಾತ್ರ_static);
    }
    ಆದರೆ (m_compute_params.attention_ಗಾತ್ರ_swa != -1) {
        create_1d_ಒಳಹರಿವು("attention_size_swa", m_compute_params.attention_ಗಾತ್ರ_swa);
    }
    // only the stateful SWA mask consumes this
    ಆದರೆ (is_stateful() && m_compute_params.swa_ಕಿಟಕಿ != -1) {
        create_1d_ಒಳಹರಿವು("swa_window", m_compute_params.swa_ಕಿಟಕಿ);
    }
    create_1d_ಒಳಹರಿವು("n_seq_active", m_compute_params.n_seq_ಸಕ್ರಿಯ);
    create_1d_ಒಳಹರಿವು("seq_active_start", m_compute_params.seq_ಸಕ್ರಿಯ_ಆರಂಭ);
    create_1d_ಒಳಹರಿವು("seq_active_end", m_compute_params.seq_ಸಕ್ರಿಯ_ಆರಂಭ + m_compute_params.n_seq_ಸಕ್ರಿಯ);
    ಆದರೆ (m_compute_params.token_len_ಪ್ರತಿ_seq != -1) {
        create_1d_ಒಳಹರಿವು("token_len_per_seq", m_compute_params.token_len_ಪ್ರತಿ_seq);
    }
    // create_1d_input("token_len", m_compute_params.token_len_per_seq * m_compute_params.n_seq_active);

    ಆದರೆ (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
        // Whether/which cache slot to reset varies per compute call (e.g. a new sequence starting
        // vs. continued decoding). can_reuse_statically() does not invalidate the cached static
        // model on ComputeParams changes, so these must stay runtime Parameters even when static
        // (scale.cpp op_case 1 only uses them in value comparisons, never as Slice bounds, so this
        // does not reintroduce dynamic shapes).
        create_1d_ಒಳಹರಿವು("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, /*force_parameter=*/ನಿಜ);
        create_1d_ಒಳಹರಿವು("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, /*force_parameter=*/ನಿಜ);
    }

    ಆದರೆ (m_compute_params.s_copy_ಸಕ್ರಿಯ_ಸ್ಥಾನ_len != -1) {
        create_1d_ಒಳಹರಿವು("s_copy_active_slot_len", m_compute_params.s_copy_ಸಕ್ರಿಯ_ಸ್ಥಾನ_len);
        ಆದರೆ (m_is_static) {
            // Number of real tokens in the current prefill chunk. The last chunk is padded with
            // fabricated token ids; attention masks them out, but the recurrent (GDN/conv) path
            // would otherwise fold them into cache_r/cache_s permanently. Varies per chunk, so it
            // must stay a runtime Parameter; it is only compared against a Range or used as Gather
            // indices, so it does not make any shape dynamic.
            create_1d_ಒಳಹರಿವು("chunk_valid_len", get_static_n_ಟೋಕನುಗಳು(), /*force_parameter=*/ನಿಜ);
        }
    }

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & [ಗಂಟು_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_ಒಳಹರಿವು("rs_slot_begin_" + ಗಂಟು_name, writeback.ಸ್ಥಾನ_begin);
        ಆದರೆ (!m_is_static && writeback.src_begin >= 0) {
            create_1d_ಒಳಹರಿವು("rs_src_begin_" + ಗಂಟು_name, writeback.src_begin);
        }
    }
}

ತಾರ್ಕಿಕ GgmlOvDecoder::ಗಂಟು_is_ಬಳಸಿದ_as_src(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ggml_tensor * ಗಂಟು = m_cgraph->nodes[ಗಂಟು_idx];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಗಂಟು_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_ಗಂಟು = m_cgraph->nodes[i];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಆದರೆ (other_ಗಂಟು->src[j] == ಗಂಟು) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಶೂನ್ಯ GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        ಆದರೆ (ಗಂಟು->op == GGML_OP_NONE && ಗಂಟು_is_ಬಳಸಿದ_as_src(i)) {
            std::string ಗಂಟು_name = get_tensor_ov_name(m_cgraph, ಗಂಟು);
            ಆದರೆ (m_model_weights.find(ಗಂಟು_name) == m_model_weights.end()) {
                m_inputs[ಗಂಟು_name] = ಗಂಟು;
                m_model_inputs[ಗಂಟು_name] = {get_ov_type(ಗಂಟು),
                                             get_ನಕ್ಷೆ_ಒಳಹರಿವು_shape(ಗಂಟು, ಶೂನ್ಯಸೂಚಕ, m_ಗಂಟು_dynamic_dims[ಗಂಟು])};
            }
            ಮುಂದುವರಿ;
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಸ್ವಯಂಚಾಲಿತ * src = ಗಂಟು->src[i];
            ಆದರೆ (src == ಶೂನ್ಯಸೂಚಕ) {
                ಮುಂದುವರಿ;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            ಆದರೆ (src->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
                src_name = get_tensor_ನಕ್ಷೆ_ಒಳಹರಿವು_ov_name(ಇದು, m_cgraph, src, ಗಂಟು);
            }
            ಆದರೆ (m_model_weights.find(src_name) != m_model_weights.end()) {
                ಮುಂದುವರಿ;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            ಆದರೆ (src->view_src != ಶೂನ್ಯಸೂಚಕ &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                ಮುಂದುವರಿ;
            }

            ತಾರ್ಕಿಕ is_intermediate_ಗಂಟು = ಸುಳ್ಳು;
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಗಂಟು_info : m_ಗಂಟು_info_list) {
                ಆದರೆ (ಗಂಟು_info.ಗಂಟು == src) {
                    is_intermediate_ಗಂಟು = ನಿಜ;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (is_intermediate_ಗಂಟು) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                ಮುಂದುವರಿ;
            }

            m_inputs[src_name] = src;

            ggml_backend_ಬಫರ್ * ಬಫರ್ = src->ಬಫರ್;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            ಆದರೆ (ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_ANY) {
                ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ it = std::find(m_model_params.kv_ಹೆಸರುಗಳು.begin(), m_model_params.kv_ಹೆಸರುಗಳು.end(), src_name);
                    it == m_model_params.kv_ಹೆಸರುಗಳು.end()) {
                    m_model_params.kv_ಹೆಸರುಗಳು.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor.
            ಆಗ (src->op == GGML_OP_VIEW && src->src[0] != ಶೂನ್ಯಸೂಚಕ) {
                src = src->src[0];
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_ನಕ್ಷೆ_ಒಳಹರಿವು_shape(ಗಂಟು, src, m_ಗಂಟು_dynamic_dims[src])};
        }
    }
}

ಶೂನ್ಯ GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_ಹೆಸರುಗಳು.clear();
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_n = 0; ಗಂಟು_n < m_cgraph->n_nodes; ಗಂಟು_n++) {
        ಸ್ವಯಂಚಾಲಿತ * cur_ಗಂಟು = m_cgraph->nodes[ಗಂಟು_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        ಆದರೆ (cur_ಗಂಟು->op == GGML_OP_NONE || cur_ಗಂಟು->op == GGML_OP_VIEW || cur_ಗಂಟು->op == GGML_OP_RESHAPE) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (::is_inplace_op(cur_ಗಂಟು) && ggml_nbytes(cur_ಗಂಟು) == 0) {
            ಮುಂದುವರಿ;
        }
        ಸ್ವಯಂಚಾಲಿತ cur_ಗಂಟು_use_ಎಣಿಕೆ = m_cgraph->use_counts[ggml_hash_ಕಂಡುಹಿಡಿ(&m_cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ, cur_ಗಂಟು)];
        ಆದರೆ (cur_ಗಂಟು_use_ಎಣಿಕೆ == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            ಆದರೆ (cur_ಗಂಟು != ಶೂನ್ಯಸೂಚಕ && ::is_inplace_op(cur_ಗಂಟು) && ggml_nbytes(cur_ಗಂಟು) > 0) {
                cur_ಗಂಟು = cur_ಗಂಟು->view_src;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_use_ಎಣಿಕೆ = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * ಗಂಟು = m_cgraph->nodes[i];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
                    ಆದರೆ (ಗಂಟು->src[j] != ಖಾಲಿ && ಗಂಟು->src[j] == cur_ಗಂಟು) {
                        ಒಳಹರಿವು_use_ಎಣಿಕೆ++;
                    }
                }
            }
            ಆದರೆ (ಒಳಹರಿವು_use_ಎಣಿಕೆ == cur_ಗಂಟು_use_ಎಣಿಕೆ) {
                cur_ಗಂಟು = ಶೂನ್ಯಸೂಚಕ;
            }
        }
        ಆದರೆ (cur_ಗಂಟು != ಶೂನ್ಯಸೂಚಕ) {
            std::string cur_ಗಂಟು_name = get_tensor_ov_name(m_cgraph, cur_ಗಂಟು);
            m_model_outputs[cur_ಗಂಟು_name] = cur_ಗಂಟು;
            m_model_output_ಹೆಸರುಗಳು.insert(cur_ಗಂಟು_name);
        }
    }
}

ಸ್ಥಿರಾಂಕ ggml_tensor * GgmlOvDecoder::get_tensor_ಬಳಸಿದ_op(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) ಸ್ಥಿರಾಂಕ {
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m_cgraph->n_nodes; i++) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = m_cgraph->nodes[i];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಆದರೆ (ಗಂಟು->src[j] == tensor) {
                ಹಿಂತಿರುಗಿ ಗಂಟು;
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
}

ಸ್ಥಿರಾಂಕ ggml_tensor * GgmlOvDecoder::get_tensor_from_name(ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m_cgraph->n_nodes; i++) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = m_cgraph->nodes[i];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src = ಗಂಟು->src[j];
            ಆದರೆ (src == ಶೂನ್ಯಸೂಚಕ) {
                ನಿಲ್ಲಿಸು;
            }
            ಆದರೆ (get_tensor_ov_name(m_cgraph, src) == name) {
                ಹಿಂತಿರುಗಿ src;
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_ಹೆಸರುಗಳು() ಸ್ಥಿರಾಂಕ {
    std::map<std::string, std::string> kv_param_res_ಹೆಸರುಗಳು;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & name : m_model_params.kv_ಹೆಸರುಗಳು) {
        kv_param_res_ಹೆಸರುಗಳು[name] = name;
    }
    ಹಿಂತಿರುಗಿ kv_param_res_ಹೆಸರುಗಳು;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mul_mat_id_expert_ತೂಕ(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು, ಪೂರ್ಣಾಂಕ src_ಸೂಚ್ಯಂಕ) {
    ಹಿಂತಿರುಗಿ ಗಂಟು->op == GGML_OP_MUL_MAT_ID && src_ಸೂಚ್ಯಂಕ == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_ತೂಕ_nodes(ggml_cgraph * cgraph, ತಾರ್ಕಿಕ naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    ಸ್ವಯಂಚಾಲಿತ * nodes = cgraph->nodes;
    ಸ್ವಯಂಚಾಲಿತ n_nodes = cgraph->n_nodes;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_i = 0; ಗಂಟು_i < n_nodes; ಗಂಟು_i++) {
        ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = nodes[ಗಂಟು_i];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಸ್ವಯಂಚಾಲಿತ * src = ಗಂಟು->src[i];
            ಆದರೆ (src == ಶೂನ್ಯಸೂಚಕ) {
                ಮುಂದುವರಿ;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * ತಳ = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, ತಳ);
            ಆದರೆ (is_rope_freqs_ತೂಕ(ತಳ, ಗಂಟು)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_ಬಫರ್ * ಬಫರ್ = ತಳ->ಬಫರ್;
            ಆದರೆ (ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS || ggml_is_quantized(ತಳ->type) ||
                is_mul_mat_id_expert_ತೂಕ(ಗಂಟು, i)) {
                ಆದರೆ (model_weights.find(src_name) == model_weights.end()) {
                    ಸ್ವಯಂಚಾಲಿತ ತೂಕ_ಗಂಟು = create_ತೂಕ_ಗಂಟು(ತಳ, naive);
                    ತೂಕ_ಗಂಟು->ಗಣ_friendly_name(src_name);
                    model_weights[src_name] = ತೂಕ_ಗಂಟು;
                }
            }
        }
    }
    ಹಿಂತಿರುಗಿ model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
ಸ್ಥಿರ std::mutex g_nonov_ತೂಕ_cache_mutex;
ಸ್ಥಿರ std::unordered_map<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *, std::shared_ptr<ov::Node>> g_nonov_ತೂಕ_cache;

std::set<std::string> GgmlOvDecoder::collect_ತೂಕ_ಹೆಸರುಗಳು(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> ಹೆಸರುಗಳು;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_i = 0; ಗಂಟು_i < cgraph->n_nodes; ಗಂಟು_i++) {
        ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = cgraph->nodes[ಗಂಟು_i];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಸ್ವಯಂಚಾಲಿತ * src = ಗಂಟು->src[i];
            ಆದರೆ (src == ಶೂನ್ಯಸೂಚಕ) {
                ಮುಂದುವರಿ;
            }
            ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ = src->view_src ? src->view_src : src;
            std::string src_name(ತಳ->name);
            ಆದರೆ (is_rope_freqs_ತೂಕ(ತಳ, ಗಂಟು)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_ಬಫರ್ * ಬಫರ್ = ತಳ->ಬಫರ್;
            ಆದರೆ (ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS || ggml_is_quantized(ತಳ->type)) {
                ಹೆಸರುಗಳು.insert(src_name);
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಹೆಸರುಗಳು;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_ತೂಕ_ಗಂಟು(ggml_tensor * tensor, ತಾರ್ಕಿಕ naive) {
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ov_ಬಫರ್ = ggml_backend_ಬಫರ್_is_openvino(tensor->ಬಫರ್);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    ಆದರೆ (tensor->extra) {
        OPENVINO_ASSERT(is_ov_ಬಫರ್, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        ಸ್ವಯಂಚಾಲಿತ * extra_ತಳ = static_cast<ggml_openvino_extra_ತಳ *>(tensor->extra);

        ಆದರೆ (extra_ತಳ->type == ggml_openvino_extra_ತಳ::Type::ತೂಕ_2) {
            // F16/F32/BF16 weight with shared-memory constant
            ಸ್ವಯಂಚಾಲಿತ * ತೂಕ_extra = static_cast<ggml_openvino_ತೂಕ_extra *>(tensor->extra);
            ಆದರೆ (ತೂಕ_extra->ತೂಕ_ಗಂಟು) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                ಹಿಂತಿರುಗಿ ತೂಕ_extra->ತೂಕ_ಗಂಟು;
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (extra_ತಳ->type == ggml_openvino_extra_ತಳ::Type::QUANTIZED_ತೂಕ) {
            // Quantized weight with pre-extracted data
            ಸ್ವಯಂಚಾಲಿತ * quant_extra = static_cast<ggml_openvino_quantized_ತೂಕ_extra *>(tensor->extra);
            ಆದರೆ (quant_extra->ತೂಕ_ಗಂಟು) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                ಹಿಂತಿರುಗಿ quant_extra->ತೂಕ_ಗಂಟು;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    ಆದರೆ (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        ಸ್ವಯಂಚಾಲಿತ ತೂಕ_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->ದತ್ತಾಂಶ);
        ಸ್ವಯಂಚಾಲಿತ ತೂಕ_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(ತೂಕ_tensor);
        ತೂಕ_ಗಂಟು->ಗಣ_friendly_name(tensor->name);
        ಹಿಂತಿರುಗಿ ತೂಕ_ಗಂಟು;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_ಬಫರ್ &&
                                 !naive && tensor->ದತ್ತಾಂಶ != ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (cacheable_nonov) {
        std::lock_guard<std::mutex> ಬೀಗ(g_nonov_ತೂಕ_cache_mutex);
        ಸ್ವಯಂಚಾಲಿತ it = g_nonov_ತೂಕ_cache.find(tensor->ದತ್ತಾಂಶ);
        ಆದರೆ (it != g_nonov_ತೂಕ_cache.end()) {
            ಹಿಂತಿರುಗಿ it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::set<ggml_type> ತೂಕ_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    ಆದರೆ (ತೂಕ_types.find(tensor->type) == ತೂಕ_types.end()) {
        ಎಸೆ std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    Ov_ತೂಕ ov_ತೂಕ;
    ಆದರೆ (ggml_is_quantized(tensor->type)) {
        ಸ್ವಯಂಚಾಲಿತ use_bias = naive;
        ಆದರೆ (is_ov_ಬಫರ್) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_ಗಾತ್ರ = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_ಗಾತ್ರ);
            ಸ್ಮರಣೆ_ನಕಲು(tmp.ದತ್ತಾಂಶ(), tensor->ದತ್ತಾಂಶ, raw_ಗಾತ್ರ);
            ov_ತೂಕ = process_ತೂಕ_tensor(tensor, tmp.ದತ್ತಾಂಶ(), tensor->ದತ್ತಾಂಶ, use_bias);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ov_ತೂಕ = process_ತೂಕ_tensor(tensor, tensor->ದತ್ತಾಂಶ, ಶೂನ್ಯಸೂಚಕ, use_bias);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_ತೂಕ = process_ತೂಕ_tensor(tensor, tensor->ದತ್ತಾಂಶ, tensor->ದತ್ತಾಂಶ);
    }

    ov_ತೂಕ.ತೂಕ_ಗಂಟು->ಗಣ_friendly_name(tensor->name);
    ಆದರೆ (!is_ov_ಬಫರ್) {
        ಆದರೆ (cacheable_nonov) {
            std::lock_guard<std::mutex> ಬೀಗ(g_nonov_ತೂಕ_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            ಸ್ವಯಂಚಾಲಿತ [it, inserted] = g_nonov_ತೂಕ_cache.emplace(tensor->ದತ್ತಾಂಶ, ov_ತೂಕ.ತೂಕ_ಗಂಟು);
            ಹಿಂತಿರುಗಿ it->second;
        }
        ಹಿಂತಿರುಗಿ ov_ತೂಕ.ತೂಕ_ಗಂಟು;
    }

    ggml_openvino_extra_ತಳ * extra;
    ಆದರೆ (ov_ತೂಕ.is_quantized()) {
        extra = ಹೊಸ ggml_openvino_quantized_ತೂಕ_extra(std::move(ov_ತೂಕ.weights), std::move(ov_ತೂಕ.scales),
                                                         std::move(ov_ತೂಕ.zp), ov_ತೂಕ.ತೂಕ_ಗಂಟು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        extra = ಹೊಸ ggml_openvino_ತೂಕ_extra(std::move(ov_ತೂಕ.weights), ov_ತೂಕ.ತೂಕ_ಗಂಟು);
    }
    ggml_openvino_ಬಫರ್_ರಿಜಿಸ್ಟರ್_extra(tensor, extra);

    ಹಿಂತಿರುಗಿ ov_ತೂಕ.ತೂಕ_ಗಂಟು;
}

ಶೂನ್ಯ GgmlOvDecoder::dump_cgraph(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    ಆದರೆ (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        ಹಿಂತಿರುಗಿ;
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
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        // Get buffer type name
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * buf_name = "none";
        ggml_backend_ಬಫರ್_t buf = ಗಂಟು->view_src ? ಗಂಟು->view_src->ಬಫರ್ : ಗಂಟು->ಬಫರ್;
        ಆದರೆ (buf) {
            buf_name = ggml_backend_ಬಫರ್_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ಗಂಟು->ne[0] << ", "
             << std::setw(5) << ಗಂಟು->ne[1] << ", "
             << std::setw(5) << ಗಂಟು->ne[2] << ", "
             << std::setw(5) << ಗಂಟು->ne[3] << "] "
             << std::ಎಡ << std::setw(20) << ggml_op_name(ಗಂಟು->op) << std::ಬಲ << " "
             << std::ಎಡ << std::setw(45) << ಗಂಟು->name << std::ಬಲ
             << std::setw(2) << "[ "
             << std::setw(0) << ಗಂಟು->nb[0] << ", "
             << std::setw(5) << ಗಂಟು->nb[1] << ", "
             << std::setw(5) << ಗಂಟು->nb[2] << ", "
             << std::setw(5) << ಗಂಟು->nb[3] << "] "
             << std::ಬಲ << std::setw(15) << buf_name << std::ಬಲ
             << "\n";

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ* src = ಗಂಟು->src[i]) {
                // Get buffer type name for source
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src_buf_name = "none";
                ggml_backend_ಬಫರ್_t src_buf = src->view_src ? src->view_src->ಬಫರ್ : src->ಬಫರ್;
                ಆದರೆ (src_buf) {
                    src_buf_name = ggml_backend_ಬಫರ್_name(src_buf);
                }

                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::ಎಡ << std::setw(12) << ggml_op_name(src->op) << std::ಬಲ;
                file << std::ಎಡ << std::setw(30) << src->name << std::ಬಲ
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << std::ಬಲ << std::setw(15) << src_buf_name << std::ಬಲ
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * ಗಂಟು = cgraph->leafs[i];

        // Get buffer type name for leaf
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಎಲೆ_buf_name = "none";
        ggml_backend_ಬಫರ್_t ಎಲೆ_buf = ಗಂಟು->view_src ? ಗಂಟು->view_src->ಬಫರ್ : ಗಂಟು->ಬಫರ್;
        ಆದರೆ (ಎಲೆ_buf) {
            ಎಲೆ_buf_name = ggml_backend_ಬಫರ್_name(ಎಲೆ_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << ಗಂಟು->ne[0] << ", "
             << std::setw(5) << ಗಂಟು->ne[1] << "] "
             << std::setw(8) << ggml_op_name(ಗಂಟು->op) << " "
             << std::setw(16) << ggml_get_name(ಗಂಟು)
             << std::setw(20) << ಎಲೆ_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

ಶೂನ್ಯ print_tensor_ವಿಳಾಸ_map(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph) {
    std::map<ಶೂನ್ಯ *, std::vector<std::string>> ವಿಳಾಸ_map;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_n = 0; ಗಂಟು_n < cgraph->n_nodes; ಗಂಟು_n++) {
        ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = cgraph->nodes[ಗಂಟು_n];
        ಆದರೆ (ಗಂಟು->ದತ್ತಾಂಶ) {
            ಸ್ವಯಂಚಾಲಿತ it = ವಿಳಾಸ_map.find(ಗಂಟು->ದತ್ತಾಂಶ);
            ಆದರೆ (it == ವಿಳಾಸ_map.end()) {
                ವಿಳಾಸ_map[ಗಂಟು->ದತ್ತಾಂಶ] = std::vector<std::string>();
            }
            ವಿಳಾಸ_map[ಗಂಟು->ದತ್ತಾಂಶ].push_back(ಗಂಟು->name);
        }
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & pair : ವಿಳಾಸ_map) {
        std::ಹೊರಗೆ << "Address: " << pair.ಮೊದಲ << '\n';
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & name : pair.second) {
            std::ಹೊರಗೆ << name << " ; ";
        }
        std::ಹೊರಗೆ << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    std::vector<size_t> shape;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    ಹಿಂತಿರುಗಿ shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    std::vector<size_t> stride;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    ಹಿಂತಿರುಗಿ stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಆಯ್ಕೆ (tensor->type) {
    ಸ್ಥಿತಿ GGML_TYPE_F64:
        ಹಿಂತಿರುಗಿ ov::element::f64;
    ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಹಿಂತಿರುಗಿ ov::element::f32;
    ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಹಿಂತಿರುಗಿ ov::element::f16;
    ಸ್ಥಿತಿ GGML_TYPE_BF16:
        ಹಿಂತಿರುಗಿ ov::element::bf16;
    ಸ್ಥಿತಿ GGML_TYPE_I8:
        ಹಿಂತಿರುಗಿ ov::element::i8;
    ಸ್ಥಿತಿ GGML_TYPE_I16:
        ಹಿಂತಿರುಗಿ ov::element::i16;
    ಸ್ಥಿತಿ GGML_TYPE_I32:
        ಹಿಂತಿರುಗಿ ov::element::i32;
    ಸ್ಥಿತಿ GGML_TYPE_I64:
        ಹಿಂತಿರುಗಿ ov::element::i64;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_ಒಳಹರಿವು_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ ov::PartialShape(get_shape(m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ get_stride(m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs.at(name));
}

size_t GgmlOvDecoder::get_view_ಒಳಹರಿವು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಹಿಂತಿರುಗಿ it->second.size();
    }
    ಹಿಂತಿರುಗಿ 0;
}

size_t GgmlOvDecoder::get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಆದರೆ (view_ಸೂಚ್ಯಂಕ < it->second.size()) {
            ಹಿಂತಿರುಗಿ it->second[view_ಸೂಚ್ಯಂಕ].second->view_offs;
        }
    }
    ಹಿಂತಿರುಗಿ 0;
}

size_t GgmlOvDecoder::get_view_ಒಳಹರಿವು_src_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಆದರೆ (view_ಸೂಚ್ಯಂಕ < it->second.size()) {
            ಸ್ವಯಂಚಾಲಿತ * view_tensor = it->second[view_ಸೂಚ್ಯಂಕ].second;
            ಆದರೆ (view_tensor && view_tensor->src[0]) {
                ಹಿಂತಿರುಗಿ view_tensor->src[0]->view_offs;
            }
        }
    }
    ಹಿಂತಿರುಗಿ 0;
}

std::vector<size_t> GgmlOvDecoder::get_view_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                         ಸ್ಥಿರಾಂಕ std::string & name,
                                                         size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಆದರೆ (view_ಸೂಚ್ಯಂಕ < it->second.size()) {
            ಹಿಂತಿರುಗಿ get_stride(it->second[view_ಸೂಚ್ಯಂಕ].second);
        }
    }
    ಹಿಂತಿರುಗಿ {};
}

std::vector<size_t> GgmlOvDecoder::get_view_ಒಳಹರಿವು_src_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                             ಸ್ಥಿರಾಂಕ std::string & name,
                                                             size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಆದರೆ (view_ಸೂಚ್ಯಂಕ < it->second.size()) {
            ಸ್ವಯಂಚಾಲಿತ * view_tensor = it->second[view_ಸೂಚ್ಯಂಕ].second;
            ಆದರೆ (view_tensor && view_tensor->src[0]) {
                ಹಿಂತಿರುಗಿ get_stride(view_tensor->src[0]);
            }
        }
    }
    ಹಿಂತಿರುಗಿ {};
}

ov::Shape GgmlOvDecoder::get_view_ಒಳಹರಿವು_ggml_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಆದರೆ (view_ಸೂಚ್ಯಂಕ < it->second.size()) {
            ಹಿಂತಿರುಗಿ get_shape(it->second[view_ಸೂಚ್ಯಂಕ].second);
        }
    }
    ಹಿಂತಿರುಗಿ {};
}

ov::Shape GgmlOvDecoder::get_view_ಒಳಹರಿವು_src_ggml_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                       ಸ್ಥಿರಾಂಕ std::string & name,
                                                       size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಆದರೆ (view_ಸೂಚ್ಯಂಕ < it->second.size()) {
            ಸ್ವಯಂಚಾಲಿತ * view_tensor = it->second[view_ಸೂಚ್ಯಂಕ].second;
            ಆದರೆ (view_tensor && view_tensor->src[0]) {
                ಹಿಂತಿರುಗಿ get_shape(view_tensor->src[0]);
            }
        }
    }
    ಹಿಂತಿರುಗಿ {};
}

ov::PartialShape GgmlOvDecoder::get_view_ಒಳಹರಿವು_ov_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                        ಸ್ಥಿರಾಂಕ std::string & name,
                                                        size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಆದರೆ (view_ಸೂಚ್ಯಂಕ < it->second.size()) {
            ಸ್ವಯಂಚಾಲಿತ * tensor = it->second[view_ಸೂಚ್ಯಂಕ].second;
            ov::PartialShape shape = ov::PartialShape{get_shape(tensor)};

            // Check if this tensor has a dynamic dimension
            ಸ್ವಯಂಚಾಲಿತ dynamic_it = m_ಗಂಟು_dynamic_dims.find(tensor);
            ಆದರೆ (dynamic_it != m_ಗಂಟು_dynamic_dims.end() && dynamic_it->second != -1) {
                ಪೂರ್ಣಾಂಕ dynamic_dim_ಸೂಚ್ಯಂಕ = dynamic_it->second;
                // GGML uses reverse indexing, so convert to OpenVINO indexing
                shape[3 - dynamic_dim_ಸೂಚ್ಯಂಕ] = m_is_static ? get_static_n_ಟೋಕನುಗಳು() : -1;
            }

            ಹಿಂತಿರುಗಿ shape;
        }
    }
    ಹಿಂತಿರುಗಿ {};
}

ov::PartialShape GgmlOvDecoder::get_view_ಒಳಹರಿವು_src_ov_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                            ಸ್ಥಿರಾಂಕ std::string & name,
                                                            size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಆದರೆ (view_ಸೂಚ್ಯಂಕ < it->second.size()) {
            ಸ್ವಯಂಚಾಲಿತ * view_tensor = it->second[view_ಸೂಚ್ಯಂಕ].second;
            ಆದರೆ (view_tensor && view_tensor->src[0]) {
                ಸ್ವಯಂಚಾಲಿತ * src_tensor = view_tensor->src[0];
                ov::PartialShape shape = ov::PartialShape{get_shape(src_tensor)};

                // Check if this tensor has a dynamic dimension
                ಸ್ವಯಂಚಾಲಿತ dynamic_it = m_ಗಂಟು_dynamic_dims.find(src_tensor);
                ಆದರೆ (dynamic_it != m_ಗಂಟು_dynamic_dims.end() && dynamic_it->second != -1) {
                    ಪೂರ್ಣಾಂಕ dynamic_dim_ಸೂಚ್ಯಂಕ = dynamic_it->second;
                    // GGML uses reverse indexing, so convert to OpenVINO indexing
                    shape[3 - dynamic_dim_ಸೂಚ್ಯಂಕ] = m_is_static ? get_static_n_ಟೋಕನುಗಳು() : -1;
                }

                ಹಿಂತಿರುಗಿ shape;
            }
        }
    }
    ಹಿಂತಿರುಗಿ {};
}

std::string GgmlOvDecoder::get_view_ಒಳಹರಿವು_name(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಆದರೆ (view_ಸೂಚ್ಯಂಕ < it->second.size()) {
            ಹಿಂತಿರುಗಿ it->second[view_ಸೂಚ್ಯಂಕ].ಮೊದಲ;
        }
    }
    ಹಿಂತಿರುಗಿ "";
}

std::string GgmlOvDecoder::get_view_ಒಳಹರಿವು_src_name(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.find(name);
    ಆದರೆ (it != m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_views.end()) {
        ಆದರೆ (view_ಸೂಚ್ಯಂಕ < it->second.size()) {
            ಸ್ವಯಂಚಾಲಿತ * view_tensor = it->second[view_ಸೂಚ್ಯಂಕ].second;
            ಆದರೆ (view_tensor && view_tensor->src[0]) {
                ಹಿಂತಿರುಗಿ get_tensor_ov_name(m_cgraph, view_tensor->src[0]);
            }
        }
    }
    ಹಿಂತಿರುಗಿ "";
}

ov::element::Type GgmlOvDecoder::get_ಒಳಹರಿವು_type(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ get_ov_type(m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs.at(name));
}

size_t GgmlOvDecoder::get_ಒಳಹರಿವು_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ m_model_inputs.size();
}

size_t GgmlOvDecoder::get_ಒಳಹರಿವು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_ಹೆಸರುಗಳು.size();
}

std::vector<std::string> GgmlOvDecoder::get_ಒಳಹರಿವು_ಹೆಸರುಗಳು(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs_ಹೆಸರುಗಳು;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ * ggml_tensor = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು;
    ಹಿಂತಿರುಗಿ ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ get_ov_type(m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು);
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ * ggml_tensor = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು;
    ಹಿಂತಿರುಗಿ get_stride(ggml_tensor);
}

std::vector<std::string> GgmlOvDecoder::get_output_ಹೆಸರುಗಳು(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು;
    ಆದರೆ (ಗಂಟು->op == GGML_OP_GATED_DELTA_NET && !m_model_params.has_rs_rollback) {
        std::string attn_name;
        std::string state_name;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಗಂಟು_idx + 1; i < m_cgraph->n_nodes; i++) {
            ಸ್ವಯಂಚಾಲಿತ * ಗ್ರಾಹಕ = m_cgraph->nodes[i];
            ಆದರೆ (ಗ್ರಾಹಕ->op != GGML_OP_VIEW || ಗ್ರಾಹಕ->src[0] != ಗಂಟು) {
                ಮುಂದುವರಿ;
            }
            // GGML packs [attention | state]. The attention VIEW starts at offset 0 and the
            // state VIEW starts after the token-dependent attention segment.
            ಸ್ವಯಂಚಾಲಿತ & name = ಗ್ರಾಹಕ->view_offs == 0 ? attn_name : state_name;
            ಆದರೆ (!name.ಖಾಲಿಗಳು()) {
                ಹಿಂತಿರುಗಿ {m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_name};
            }
            name = get_tensor_ov_name(m_cgraph, ಗ್ರಾಹಕ);
        }
        ಆದರೆ (!attn_name.ಖಾಲಿಗಳು() && !state_name.ಖಾಲಿಗಳು()) {
            ಹಿಂತಿರುಗಿ {attn_name, state_name};
        }
    }
    ಹಿಂತಿರುಗಿ {m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು;
    ಆದರೆ (!::is_inplace_op(ಗಂಟು) || ಗಂಟು->view_src == ಶೂನ್ಯಸೂಚಕ || ggml_nbytes(ಗಂಟು) == 0) {
        ಹಿಂತಿರುಗಿ "";
    }
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ op_case = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_op_case;
    ಆದರೆ (ಗಂಟು->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_ಸಕ್ರಿಯ_ಸ್ಥಾನ_len == -1) {
        ಹಿಂತಿರುಗಿ "";
    }
    ಹಿಂತಿರುಗಿ get_tensor_ov_name(m_cgraph, ಗಂಟು->view_src);
}

ತಾರ್ಕಿಕ GgmlOvDecoder::is_view_like_alias_ದ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & view_src_name) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು;
    ಆದರೆ (ಗಂಟು->view_src == ಶೂನ್ಯಸೂಚಕ || get_tensor_ov_name(m_cgraph, ಗಂಟು->view_src) != view_src_name) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ಗಂಟು->op == GGML_OP_RESHAPE || ಗಂಟು->op == GGML_OP_VIEW;
}

ಸ್ಥಿರಾಂಕ std::string & GgmlOvDecoder::get_op_name() ಸ್ಥಿರಾಂಕ {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::string unknown_name = "UNKNOWN_OP_NAME";
    ಹಿಂತಿರುಗಿ unknown_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಸ್ವಯಂಚಾಲಿತ it = m_ಗಂಟು_dynamic_dims.find(m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು);
    ಆದರೆ (it == m_ಗಂಟು_dynamic_dims.end()) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಹಿಂತಿರುಗಿ it->second;
}

ಸ್ಥಿರಾಂಕ std::string & GgmlOvDecoder::get_op_name(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_name;
}

int32_t * GgmlOvDecoder::get_ಒಳಹರಿವು_op_params(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು->op_params;
}

size_t GgmlOvDecoder::get_output_op_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು->view_offs;
}

ಶೂನ್ಯ GgmlOvDecoder::ಭೇಟಿ_subgraph(std::function<ಶೂನ್ಯ(std::shared_ptr<GgmlDecoder>, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx)> ಗಂಟು_visitor) ಸ್ಥಿರಾಂಕ {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_idx = 0; ಗಂಟು_idx < m_cgraph->n_nodes; ಗಂಟು_idx++) {
        ಆದರೆ (m_cgraph->nodes[ಗಂಟು_idx]->op == GGML_OP_NONE) {
            ಮುಂದುವರಿ;
        }
        ಗಂಟು_visitor(std::make_shared<GgmlOvDecoder>(*ಇದು), ಗಂಟು_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ಆಯ್ಕೆ (ಗಂಟು->op) {
    ಸ್ಥಿತಿ GGML_OP_UNARY:
        ಹಿಂತಿರುಗಿ std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(ಗಂಟು));
    ಸ್ಥಿತಿ GGML_OP_GLU:
        ಹಿಂತಿರುಗಿ std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(ಗಂಟು));
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ std::string("GGML_OP_") + ggml_op_name(ಗಂಟು->op);
    }
}

ಸ್ಥಿರಾಂಕ std::string & GgmlOvDecoder::get_op_type(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_op_type;
}

ಸ್ಥಿರಾಂಕ std::string & GgmlOvDecoder::get_op_type() ಸ್ಥಿರಾಂಕ {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::string unknown_op = "UNKNOWN_GGML_OP";
    ಹಿಂತಿರುಗಿ unknown_op;
}

ಶೂನ್ಯ GgmlOvDecoder::compute_ಗಂಟು_dynamic_dims() {
    ಸ್ವಯಂಚಾಲಿತ ಭೇಟಿ_ಗಂಟು = [&](ಸ್ವಯಂಚಾಲಿತ && self, ggml_tensor * ಗಂಟು) -> ಶೂನ್ಯ {
        ಆದರೆ (!ಗಂಟು) {
            ಹಿಂತಿರುಗಿ;
        }

        ಆದರೆ (ಗಂಟು->op == GGML_OP_CPY) {
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
        }

        ಆದರೆ (m_ಗಂಟು_dynamic_dims.count(ಗಂಟು)) {
            ಹಿಂತಿರುಗಿ;
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = ಗಂಟು->src[i];
            ಆದರೆ (src == ಶೂನ್ಯಸೂಚಕ) {
                ಮುಂದುವರಿ;
            }
            ರಚನೆ ggml_tensor * ಬೇರು_src = ಶೂನ್ಯಸೂಚಕ;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            ಆದರೆ (ಬೇರು_src) {
                ಆದರೆ (is_inp_tok(ಬೇರು_src, ಗಂಟು) || is_inp_pos(ಬೇರು_src, ಗಂಟು) || is_output_idx(ಬೇರು_src, ಗಂಟು)) {
                    m_ಗಂಟು_dynamic_dims[ಬೇರು_src] = 0;
                    m_ಗಂಟು_dynamic_dims[src] = m_ಗಂಟು_dynamic_dims[ಬೇರು_src];
                    ಮುಂದುವರಿ;
                }
                self(self, ಬೇರು_src);
                m_ಗಂಟು_dynamic_dims[src] = m_ಗಂಟು_dynamic_dims[ಬೇರು_src];
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (is_inp_tok(src, ಗಂಟು) || is_inp_pos(src, ಗಂಟು) || is_output_idx(src, ಗಂಟು)) {
                    m_ಗಂಟು_dynamic_dims[src] = 0;
                    ಮುಂದುವರಿ;
                }
                ಆದರೆ (is_inp_scale_ಅಡ್ಡಸಾಲುಗಳು(src, ಗಂಟು)) {
                    m_ಗಂಟು_dynamic_dims[src] = 1;
                    ಮುಂದುವರಿ;
                }
                ಆದರೆ (ಗಂಟು->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                    m_ಗಂಟು_dynamic_dims[src] = 1;
                    ಮುಂದುವರಿ;
                }
                self(self, src);
            }
        }
        ಆಯ್ಕೆ (ಗಂಟು->op) {
        ಸ್ಥಿತಿ GGML_OP_NONE:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]] != -1) {
                ಸ್ವಯಂಚಾಲಿತ dynamic_dim_idx = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]];
                ಆದರೆ (dynamic_dim_idx == 0) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 1;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಸ್ವಯಂಚಾಲಿತ dynamic_dim_stride = ಗಂಟು->src[1]->nb[dynamic_dim_idx] / ggml_type_ಗಾತ್ರ(ಗಂಟು->src[1]->type) *
                                              ggml_type_ಗಾತ್ರ(ಗಂಟು->src[0]->type);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
                        ಆದರೆ (dynamic_dim_stride == ಗಂಟು->src[0]->nb[i]) {
                            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                            ನಿಲ್ಲಿಸು;
                        }
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[1]: " + std::string(node->src[1]->name));
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL:
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
            }
            ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]] != -1) {
                m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]];
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                ಸ್ವಯಂಚಾಲಿತ dynamic_dim_idx = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                // auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
                    ಆದರೆ (ಗಂಟು->op_params[i] == dynamic_dim_idx) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                        ನಿಲ್ಲಿಸು;
                    }
                }
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                ಆದರೆ (ಗಂಟು->src[0]->op == GGML_OP_NONE) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                    ನಿಲ್ಲಿಸು;
                }
                ಸ್ವಯಂಚಾಲಿತ dynamic_dim_idx = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                ಸ್ವಯಂಚಾಲಿತ dynamic_dim_ಮೌಲ್ಯ = ಗಂಟು->src[0]->ne[dynamic_dim_idx];
                ಸ್ವಯಂಚಾಲಿತ dynamic_dim_stride =
                    ಗಂಟು->src[0]->nb[dynamic_dim_idx] / ggml_type_ಗಾತ್ರ(ಗಂಟು->src[0]->type) * ggml_type_ಗಾತ್ರ(ಗಂಟು->type);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
                    ಆದರೆ (ಗಂಟು->nb[i] == dynamic_dim_stride) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                        ನಿಲ್ಲಿಸು;
                    }
                }
                ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು] != -1 && dynamic_dim_ಮೌಲ್ಯ != ಗಂಟು->ne[m_ಗಂಟು_dynamic_dims[ಗಂಟು]]) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    ಆದರೆ (ggml_nelements(ಗಂಟು) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      ಗಂಟು->name, ಗಂಟು->src[0]->name);
                    }
                }
            }
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE: {
            ಆದರೆ (is_same_shape(ಗಂಟು->src[0], ಗಂಟು)) {
                m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                ನಿಲ್ಲಿಸು;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                ಸ್ವಯಂಚಾಲಿತ dynamic_dim_idx = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                ಸ್ವಯಂಚಾಲಿತ dynamic_dim_stride = ಗಂಟು->src[0]->nb[dynamic_dim_idx];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
                    ಆದರೆ (ಗಂಟು->nb[i] == dynamic_dim_stride && ಗಂಟು->ne[i] == ಗಂಟು->src[0]->ne[dynamic_dim_idx]) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                        ನಿಲ್ಲಿಸು;
                    }
                }
                ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", ಗಂಟು->name);
                }
            }
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT: {
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
            constexpr ಪೂರ್ಣಾಂಕ q_ಗಮ್ಯ_ಹೊರಹೋಗು[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                ಸ್ವಯಂಚಾಲಿತ q_dynamic_dim = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                m_ಗಂಟು_dynamic_dims[ಗಂಟು] = q_ಗಮ್ಯ_ಹೊರಹೋಗು[q_dynamic_dim];
            }
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_OP_CONT:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] != -1) {
                ಸ್ವಯಂಚಾಲಿತ dynamic_dim_idx = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
                ಆದರೆ (ggml_are_same_shape(ಗಂಟು, ಗಂಟು->src[0])) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = dynamic_dim_idx;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_ಗಾತ್ರ(ಗಂಟು->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (ಗಂಟು->src[0]->ne[0] / ggml_blck_ಗಾತ್ರ(ಗಂಟು->src[0]->type));
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * ಗಂಟು->src[0]->ne[i - 1];
                    }

                    ಸ್ವಯಂಚಾಲಿತ dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_ಗಾತ್ರ(ಗಂಟು->src[0]->type) *
                                              ggml_type_ಗಾತ್ರ(ಗಂಟು->type);
                    ಪೂರ್ಣಾಂಕ matched_dim_ಎಣಿಕೆ = 0;
                    ಪೂರ್ಣಾಂಕ ಮೊದಲ_matched_dim = -1;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
                        ಆದರೆ (ಗಂಟು->nb[i] == dynamic_dim_stride && ಗಂಟು->ne[i] == ಗಂಟು->src[0]->ne[dynamic_dim_idx]) {
                            ಆದರೆ (ಮೊದಲ_matched_dim == -1) {
                                ಮೊದಲ_matched_dim = i;
                            }
                            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                            matched_dim_ಎಣಿಕೆ++;
                        }
                    }
                    ಆದರೆ (matched_dim_ಎಣಿಕೆ > 1 && ಗಂಟು->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = ಮೊದಲ_matched_dim;
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (matched_dim_ಎಣಿಕೆ != 1) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      ಗಂಟು->name, ಗಂಟು->src[0]->name);
                    }
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONCAT:
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
                ಆದರೆ (ಗಂಟು->src[0]->ne[i] != ಗಂಟು->ne[i]) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = i;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 1;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
        ಸ್ಥಿತಿ GGML_OP_NORM:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_SUB:
        ಸ್ಥಿತಿ GGML_OP_GLU:
        ಸ್ಥಿತಿ GGML_OP_ROPE:
        ಸ್ಥಿತಿ GGML_OP_SCALE:
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
        ಸ್ಥಿತಿ GGML_OP_UNARY:
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
        ಸ್ಥಿತಿ GGML_OP_FILL:
        ಸ್ಥಿತಿ GGML_OP_ಗಣ:
        ಸ್ಥಿತಿ GGML_OP_DIAG:
        ಸ್ಥಿತಿ GGML_OP_TRI:
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
        ಸ್ಥಿತಿ GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        ಸ್ಥಿತಿ GGML_OP_DIV:
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
        ಸ್ಥಿತಿ GGML_OP_PAD:
        ಸ್ಥಿತಿ GGML_OP_UPSCALE:
        ಸ್ಥಿತಿ GGML_OP_SIN:
        ಸ್ಥಿತಿ GGML_OP_COS:
        ಸ್ಥಿತಿ GGML_OP_LOG:
        ಸ್ಥಿತಿ GGML_OP_ROLL:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] =
                (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]] == 0) ? -1 : m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[0]];
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]];
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CPY:
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು: {
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]] != -1) {
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_2D = ಗಂಟು->op_params[6] == 1;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src_dyn = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]];
                ಆದರೆ (is_2D) {
                    ಆದರೆ (src_dyn == 0) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 1;  // IW -> OW
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src_dyn == 1) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 2;  // IH -> OH
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src_dyn == 3) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 3;  // N  -> N
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಆದರೆ (src_dyn == 0) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 1;  // IW -> OW
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src_dyn == 2) {
                        m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು] != -1) {
                    OPENVINO_ASSERT(ಗಂಟು->src[1]->ne[src_dyn] == ಗಂಟು->ne[m_ಗಂಟು_dynamic_dims[ಗಂಟು]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(ಗಂಟು->name) +
                                        " and its src[1]: " + std::string(ಗಂಟು->src[1]->name));
                }
            }
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D: {
            m_ಗಂಟು_dynamic_dims[ಗಂಟು] = -1;
            ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]] != -1) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src_dyn = m_ಗಂಟು_dynamic_dims[ಗಂಟು->src[1]];
                ಆದರೆ (src_dyn == 0) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 1;  // IW -> OW
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src_dyn == 1) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 2;  // IH -> OH
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src_dyn == 3) {
                    m_ಗಂಟು_dynamic_dims[ಗಂಟು] = 3;  // N  -> N
                }
                ಆದರೆ (m_ಗಂಟು_dynamic_dims[ಗಂಟು] != -1) {
                    OPENVINO_ASSERT(ಗಂಟು->src[1]->ne[src_dyn] == ಗಂಟು->ne[m_ಗಂಟು_dynamic_dims[ಗಂಟು]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(ಗಂಟು->name) +
                                        " and its src[1]: " + std::string(ಗಂಟು->src[1]->name));
                }
            }
            ನಿಲ್ಲಿಸು;
        }
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(ಗಂಟು->op), ಗಂಟು->name);
            ನಿಲ್ಲಿಸು;
        }
    };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = m_cgraph->nodes[i];
        ಭೇಟಿ_ಗಂಟು(ಭೇಟಿ_ಗಂಟು, ಗಂಟು);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    ಆದರೆ (0) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * ಗಂಟು = m_cgraph->nodes[i];
            ಪೂರ್ಣಾಂಕ dynamic_dim = m_ಗಂಟು_dynamic_dims[ಗಂಟು];
            std::ಹೊರಗೆ << "[" << i << "] " << "node_name: " << ಗಂಟು->name << " op: " << ggml_op_name(ಗಂಟು->op)
                      << " shape: [";
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) {
                ಆದರೆ (j == dynamic_dim) {
                    std::ಹೊರಗೆ << "*";
                } ಇಲ್ಲದಿದ್ದರೆ {
                    std::ಹೊರಗೆ << ಗಂಟು->ne[j];
                }
                ಆದರೆ (j < 3) {
                    std::ಹೊರಗೆ << ", ";
                }
            }
            std::ಹೊರಗೆ << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = ಗಂಟು->src[j];
                ಆದರೆ (src == ಶೂನ್ಯಸೂಚಕ) {
                    ಮುಂದುವರಿ;
                }
                ಪೂರ್ಣಾಂಕ src_dynamic_dim = m_ಗಂಟು_dynamic_dims[src];
                std::ಹೊರಗೆ << "    [" << j << "] src_name: " << src->name << " [";
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; k++) {
                    ಆದರೆ (k == src_dynamic_dim) {
                        std::ಹೊರಗೆ << "*";
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        std::ಹೊರಗೆ << src->ne[k];
                    }
                    ಆದರೆ (k < 3) {
                        std::ಹೊರಗೆ << ", ";
                    }
                }
                std::ಹೊರಗೆ << "]" << '\n';
            }
            std::ಹೊರಗೆ << '\n';
        }
    }
}
