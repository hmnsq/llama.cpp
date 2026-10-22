#ಅನ್ತರ್ಭಾವ "utils.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino-extra.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino/ggml-decoder.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "model-cache.h"
#ಅನ್ತರ್ಭಾವ "openvino/frontend.h"
#ಅನ್ತರ್ಭಾವ "openvino/input_model.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <functional>
#ಅನ್ತರ್ಭಾವ <iomanip>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/any.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/graph_util.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/shape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/float16.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/manager.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/openvino.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/compiled_model.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/infer_request.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/intel_npu/properties.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/properties.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/tensor.hpp>
#ಅನ್ತರ್ಭಾವ <optional>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ {
// For a KV cache input, return an ov::Tensor sized to n_kv (== attention_size
// for that layer) instead of the fully-allocated ctx_per_seq. Pre-conditions:
//   * non-static (CPU/GPU) backend, single sequence, seq_active_start == 0
//   * ggml KV layout is a contiguous [1, 1, ctx_per_seq, n_heads_kv*head_size]
//     so the first n_kv rows are the live prefix and shrinking the ctx axis
//     gives a valid tensor over the same host storage
//   * not an SWA layer (ring cache): once the window has wrapped the first
//     n_kv rows no longer contain the live prefix
// On any unmet pre-condition returns std::nullopt; the caller falls back to
// the full-size tensor.
std::optional<ov::Tensor> try_make_kv_sliced_tensor(ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    ಸ್ಥಿರಾಂಕ std::string & name,
                                                    ಸ್ಥಿರಾಂಕ ggml_tensor * ggml_tensor) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kv_ತುಂಡು_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    ಆದರೆ (kv_ತುಂಡು_disabled) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
    ಆದರೆ (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
    ಆದರೆ (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * op = ggml_decoder->get_tensor_ಬಳಸಿದ_op(ggml_tensor);
    ಆದರೆ (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & compute_params = ggml_decoder->get_compute_params();
    ಆದರೆ (compute_params.n_seq_ಸಕ್ರಿಯ != 1 || compute_params.seq_ಸಕ್ರಿಯ_ಆರಂಭ != 0) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }

    ಪೂರ್ಣಾಂಕ layer;
    ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ layer_opt = ಹೊರತೆಗೆ_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_swa = ggml_decoder->is_swa_layer(layer);
    ಆದರೆ (is_swa) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ctx_ಪ್ರತಿ_seq = ggml_decoder->get_ctx_ಪ್ರತಿ_seq();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_kv = compute_params.attention_ಗಾತ್ರ;
    ಆದರೆ (ctx_ಪ್ರತಿ_seq <= 0 || n_kv <= 0 || n_kv >= ctx_ಪ್ರತಿ_seq) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }

    ov::Shape ತುಂಬಿದ_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    ಆದರೆ (ತುಂಬಿದ_shape.size() != 4 || ತುಂಬಿದ_shape[0] != 1 || ತುಂಬಿದ_shape[1] != 1 ||
        static_cast<ಪೂರ್ಣಾಂಕ>(ತುಂಬಿದ_shape[2]) != ctx_ಪ್ರತಿ_seq) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }

    ov::Shape sliced_shape = ತುಂಬಿದ_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    ಆದರೆ (ggml_openvino_ಬಫರ್_is_remote(ggml_tensor)) {
        ಸ್ವಯಂಚಾಲಿತ remote_context = ggml_openvino_get_remote_context();
        ಸ್ವಯಂಚಾಲಿತ gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        ಹಿಂತಿರುಗಿ gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->ದತ್ತಾಂಶ);
    }

    ಹಿಂತಿರುಗಿ ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->ದತ್ತಾಂಶ);
}

ಸ್ಥಿರ uint64_t ggml_openvino_model_cache_extra_cfg(ತಾರ್ಕಿಕ stateful, ತಾರ್ಕಿಕ recurrent) {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ manual_gqa_enabled = manual_gqa_env != ಶೂನ್ಯಸೂಚಕ ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    ಹಿಂತಿರುಗಿ extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_ತೂಕ_ಹೆಸರುಗಳು(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> ಹೆಸರುಗಳು;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & name : GgmlOvDecoder::collect_ತೂಕ_ಹೆಸರುಗಳು(cgraph)) {
        ಹೆಸರುಗಳು[name] = ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ ಹೆಸರುಗಳು;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
ಸ್ಥಿರ std::string compiled_ನಕ್ಷೆ_ಕೀಲಿ(ಸ್ಥಿರಾಂಕ ggml_cgraph * graph,
                                      ಸ್ಥಿರಾಂಕ GgmlOvDecoder & decoder,
                                      ಸ್ಥಿರಾಂಕ std::string & device,
                                      ಪೂರ್ಣಾಂಕ prefill_chunk_ಗಾತ್ರ = 0,
                                      ತಾರ್ಕಿಕ disk_cache = ಸುಳ್ಳು) {
    std::string ಕೀಲಿ;
    ಸ್ವಯಂಚಾಲಿತ append = [&ಕೀಲಿ](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & value) {
        ಕೀಲಿ.append(reinterpret_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(&value), ಗಾತ್ರ(value));
    };
    ಸ್ವಯಂಚಾಲಿತ append_string = [&](ಸ್ಥಿರಾಂಕ std::string & value) {
        append(value.size());
        ಕೀಲಿ.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_ಗಾತ್ರ);
    ತಾರ್ಕಿಕ has_ತೂಕ_ಬಫರ್_id = ಸುಳ್ಳು;
    std::unordered_map<ಸ್ಥಿರಾಂಕ ggml_tensor *, size_t> ids;
    std::function<ಶೂನ್ಯ(ಸ್ಥಿರಾಂಕ ggml_tensor *)> ಭೇಟಿ = [&](ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
        ಆದರೆ (!tensor) {
            append(size_t(0));
            ಹಿಂತಿರುಗಿ;
        }
        ಸ್ವಯಂಚಾಲಿತ inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.ಮೊದಲ->second);
        ಆದರೆ (!inserted.second) {
            ಹಿಂತಿರುಗಿ;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ತಳ = tensor->view_src ? tensor->view_src : tensor;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ತೂಕ = ತಳ->ಬಫರ್ && ತಳ->ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS;
        append(ತೂಕ);
        ಆದರೆ (ತೂಕ && !disk_cache) {
            ಸ್ಥಿರಾಂಕ size_t ಬಫರ್_id = ggml_backend_openvino_ಬಫರ್_get_ctx_id(ತಳ->ಬಫರ್);
            has_ತೂಕ_ಬಫರ್_id |= ಬಫರ್_id != 0;
            append(ಬಫರ್_id);
            append(tensor->ದತ್ತಾಂಶ);
        }
        ಭೇಟಿ(tensor->view_src);
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src : tensor->src) {
            ಭೇಟಿ(src);
        }
    };
    append(graph->n_nodes);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; ++i) {
        ಭೇಟಿ(graph->nodes[i]);
    }
    append(graph->n_leafs);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_leafs; ++i) {
        ಭೇಟಿ(graph->leafs[i]);
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಒಳಹರಿವು : decoder.get_model_extra_inputs()) {
        append_string(ಒಳಹರಿವು.ಮೊದಲ);
        append_string(ಒಳಹರಿವು.second.type.get_type_name());
        append(ಒಳಹರಿವು.second.shape.size());
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ dim : ಒಳಹರಿವು.second.shape) {
            append(dim);
        }
        append(ಒಳಹರಿವು.second.is_parameter);
        ಆದರೆ (!ಒಳಹರಿವು.second.is_parameter) {
            append(ಒಳಹರಿವು.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    ಹಿಂತಿರುಗಿ disk_cache || has_ತೂಕ_ಬಫರ್_id ? ಕೀಲಿ : std::string{};
}

ಸ್ಥಿರ std::string dynamic_ನಕ್ಷೆ_signature(ಸ್ಥಿರಾಂಕ ggml_cgraph * graph,
                                           ಸ್ಥಿರಾಂಕ GgmlOvDecoder & decoder,
                                           ಸ್ಥಿರಾಂಕ ModelParams & params) {
    std::string ಕೀಲಿ = "dynamic-1";
    ಸ್ವಯಂಚಾಲಿತ append = [&ಕೀಲಿ](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & value) {
        ಕೀಲಿ.append(reinterpret_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(&value), ಗಾತ್ರ(value));
    };
    ಸ್ವಯಂಚಾಲಿತ append_string = [&](ಸ್ಥಿರಾಂಕ std::string & value) {
        append(value.size());
        ಕೀಲಿ.append(value);
    };
    ನಕ್ಷೆ_ಕೀಲಿ ನಕ್ಷೆ_id(graph, ನಿಜ);
    append(ನಕ್ಷೆ_id.n_nodes);
    append(ನಕ್ಷೆ_id.n_leaves);
    append_string(ನಕ್ಷೆ_id.ಮೊದಲ_ಗಂಟು_name);
    append_string(ನಕ್ಷೆ_id.ಕೊನೆಯ_ಗಂಟು_name);
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & name : ನಕ್ಷೆ_id.ಒಳಹರಿವು_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ layer : params.swa_layers) {
        append(layer);
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & [layer, heads] : params.n_heads_kv_ಪ್ರತಿ_layer) {
        append(layer);
        append(heads);
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & [name, ಒಳಹರಿವು] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(ಒಳಹರಿವು.type.get_type_name());
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & [name, ಒಳಹರಿವು] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(ಒಳಹರಿವು.type.get_type_name());
        append(ಒಳಹರಿವು.is_parameter);
        ಆದರೆ (!ಒಳಹರಿವು.is_parameter) {
            append(ಒಳಹರಿವು.value);
        }
    }
    ಹಿಂತಿರುಗಿ ಕೀಲಿ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ compiled_model_matches_ನಕ್ಷೆ(ಸ್ಥಿರಾಂಕ ov::CompiledModel & model,
                                         ಸ್ಥಿರಾಂಕ std::vector<std::string> & inputs,
                                         ಸ್ಥಿರಾಂಕ std::vector<std::string> & outputs,
                                         ಸ್ಥಿರಾಂಕ GgmlOvDecoder & decoder) {
    ಆದರೆ (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ನಕ್ಷೆ_inputs = decoder.get_model_inputs();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & extra_inputs = decoder.get_model_extra_inputs();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ನಕ್ಷೆ_outputs = decoder.get_model_outputs();
    ಗಾಗಿ (size_t i = 0; i < inputs.size(); ++i) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಪೋರ್ಟ್ = model.ಒಳಹರಿವು(i);
        ಸ್ವಯಂಚಾಲಿತ ನಕ್ಷೆ_it = ನಕ್ಷೆ_inputs.find(inputs[i]);
        ಆದರೆ (ನಕ್ಷೆ_it != ನಕ್ಷೆ_inputs.end()) {
            ಆದರೆ (ಪೋರ್ಟ್.get_element_type() != ನಕ್ಷೆ_it->second.type ||
                !ಪೋರ್ಟ್.get_partial_shape().compatible(ನಕ್ಷೆ_it->second.shape)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಮುಂದುವರಿ;
        }
        ಸ್ವಯಂಚಾಲಿತ extra_it = extra_inputs.find(inputs[i]);
        ಆದರೆ (extra_it == extra_inputs.end() || ಪೋರ್ಟ್.get_element_type() != extra_it->second.type ||
            !ಪೋರ್ಟ್.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಗಾಗಿ (size_t i = 0; i < outputs.size(); ++i) {
        ಸ್ವಯಂಚಾಲಿತ ನಕ್ಷೆ_it = ನಕ್ಷೆ_outputs.find(outputs[i]);
        ಆದರೆ (ನಕ್ಷೆ_it == ನಕ್ಷೆ_outputs.end()) {
            ಆದರೆ (outputs[i].compare(0, 8, "__debug_") != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಮುಂದುವರಿ;
        }
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಪೋರ್ಟ್ = model.output(i);
        ಆದರೆ (ಪೋರ್ಟ್.get_element_type() != GgmlOvDecoder::get_ov_type(ನಕ್ಷೆ_it->second) ||
            !ಪೋರ್ಟ್.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(ನಕ್ಷೆ_it->second)))) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ std::string ov_profiling_csv_ಕ್ಷೇತ್ರ(ಸ್ಥಿರಾಂಕ std::string & value) {
    std::string escaped = "\"";
    ಗಾಗಿ (ಅಕ್ಷರ c : value) {
        escaped += c;
        ಆದರೆ (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    ಹಿಂತಿರುಗಿ escaped;
}

ಸ್ಥಿರ ಶೂನ್ಯ dump_ov_profiling_info(ಸ್ಥಿರಾಂಕ ov::InferRequest & infer_request) {
    ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರ std::atomic<uint64_t> inference_ಸೂಚ್ಯಂಕ{0};
    ಸ್ಥಿರಾಂಕ uint64_t ಸೂಚ್ಯಂಕ = inference_ಸೂಚ್ಯಂಕ.fetch_add(1);
    ಸ್ಥಿರಾಂಕ std::string path = "openvino_profile_" + std::to_string(ಸೂಚ್ಯಂಕ) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    ಆದರೆ (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        ಹಿಂತಿರುಗಿ;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t ಒಟ್ಟು_real_ಸಮಯ = 0;
    int64_t ಒಟ್ಟು_cpu_ಸಮಯ = 0;
    size_t executed_ಎಣಿಕೆ = 0;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & info : infer_request.get_profiling_info()) {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸ್ಥಿತಿ_2 = "NOT_RUN";
        ಆದರೆ (info.ಸ್ಥಿತಿ_2 == ov::ProfilingInfo::ಸ್ಥಿತಿ_3::EXECUTED) {
            ಸ್ಥಿತಿ_2 = "EXECUTED";
            ಒಟ್ಟು_real_ಸಮಯ += info.real_ಸಮಯ.count();
            ಒಟ್ಟು_cpu_ಸಮಯ += info.cpu_ಸಮಯ.count();
            executed_ಎಣಿಕೆ++;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (info.ಸ್ಥಿತಿ_2 == ov::ProfilingInfo::ಸ್ಥಿತಿ_3::OPTIMIZED_ಹೊರಹೋಗು) {
            ಸ್ಥಿತಿ_2 = "OPTIMIZED_OUT";
        }

        output << ಸ್ಥಿತಿ_2 << ',' << info.real_ಸಮಯ.count() << ',' << info.cpu_ಸಮಯ.count() << ','
               << info.ಆರಂಭ_ಸಮಯ.count() << ',' << ov_profiling_csv_ಕ್ಷೇತ್ರ(info.node_type) << ','
               << ov_profiling_csv_ಕ್ಷೇತ್ರ(info.ಗಂಟು_name) << ',' << ov_profiling_csv_ಕ್ಷೇತ್ರ(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_ಎಣಿಕೆ, ಒಟ್ಟು_real_ಸಮಯ / 1000.0, ಒಟ್ಟು_cpu_ಸಮಯ / 1000.0);
}

ಸ್ಥಿರ ಶೂನ್ಯ reset_single_ಸ್ಥಾನ_recurrent_cache(ggml_cgraph * cgraph,
                                              ಸ್ಥಿರಾಂಕ ModelParams & model_params,
                                              ಸ್ಥಿರಾಂಕ ComputeParams & compute_params) {
    ಆದರೆ (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        ಹಿಂತಿರುಗಿ;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_ಬಫರ್_t> buffers;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = cgraph->nodes[i];
        ಆದರೆ (ಗಂಟು->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(ಗಂಟು->view_src) &&
            ಗಂಟು->view_src->ne[1] == 1 && ಗಂಟು->view_src->ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ) {
            buffers.insert(ಗಂಟು->view_src->ಬಫರ್);
        }
    }
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ * ಬಫರ್ : buffers) {
        ggml_backend_ಬಫರ್_clear(ಬಫರ್, 0);
    }
}

ov::Tensor create_ov_output_tensor(ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   ಸ್ಥಿರಾಂಕ std::shared_ptr<ov::InferRequest> & infer_request,
                                   ಪೂರ್ಣಾಂಕ output_ಸೂಚ್ಯಂಕ,
                                   ಸ್ಥಿರಾಂಕ ggml_tensor * ggml_tensor) {
    ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        ಹಿಂತಿರುಗಿ *sliced;
    }

    ಸ್ವಯಂಚಾಲಿತ output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    ಶೂನ್ಯ * output_ದತ್ತಾಂಶ = ggml_tensor->ದತ್ತಾಂಶ;
    ಆದರೆ (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_ಸೂಚ್ಯಂಕ).get_shape();
    } ಇಲ್ಲದಿದ್ದರೆ {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        ಆದರೆ (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != ಶೂನ್ಯಸೂಚಕ &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_ದತ್ತಾಂಶ = ggml_tensor->view_src->ದತ್ತಾಂಶ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    ಆದರೆ (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_ಬಫರ್_is_remote(ggml_tensor) &&
        ggml_tensor->extra != ಶೂನ್ಯಸೂಚಕ) {
        ಸ್ವಯಂಚಾಲಿತ * extra_ತಳ = static_cast<ggml_openvino_extra_ತಳ *>(ggml_tensor->extra);
        ಆದರೆ (extra_ತಳ->type == ggml_openvino_extra_ತಳ::Type::TENSOR) {
            ಸ್ವಯಂಚಾಲಿತ * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_ತಳ);
            ಆದರೆ (tensor_extra->tensor != ಶೂನ್ಯಸೂಚಕ && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                ಹಿಂತಿರುಗಿ *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_ದತ್ತಾಂಶ);
    ಹಿಂತಿರುಗಿ output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_seq_axis_2(ಸ್ಥಿರಾಂಕ ov::Tensor & kv_tensor, size_t n_heads_kv) {
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲುಗಳು = kv_tensor.get_shape()[2];
    ಸ್ಥಿರಾಂಕ size_t ತಲೆ_ಗಾತ್ರ = kv_tensor.get_shape()[3] / n_heads_kv;
    ಸ್ಥಿರಾಂಕ size_t elem = kv_tensor.get_element_type().size();
    ಸ್ಥಿರಾಂಕ size_t ತಲೆ_ಬೈಟುಗಳು = ತಲೆ_ಗಾತ್ರ * elem;

    ov::Tensor ಹೊರಹೋಗು(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, ಅಡ್ಡಸಾಲುಗಳು, ತಲೆ_ಗಾತ್ರ});
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(kv_tensor.ದತ್ತಾಂಶ());
    ಸ್ವಯಂಚಾಲಿತ * dst = static_cast<uint8_t *>(ಹೊರಹೋಗು.ದತ್ತಾಂಶ());
    ಗಾಗಿ (size_t s = 0; s < ಅಡ್ಡಸಾಲುಗಳು; s++) {
        ಗಾಗಿ (size_t h = 0; h < n_heads_kv; h++) {
            ಸ್ಮರಣೆ_ನಕಲು(dst + (h * ಅಡ್ಡಸಾಲುಗಳು + s) * ತಲೆ_ಬೈಟುಗಳು, src + (s * n_heads_kv + h) * ತಲೆ_ಬೈಟುಗಳು, ತಲೆ_ಬೈಟುಗಳು);
        }
    }
    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> ಶೂನ್ಯ ಗಣ_zero_diagonal(std::vector<T> & ಮಾತೃಕೆ, size_t ಅಡ್ಡಸಾಲುಗಳು, size_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, T zero_ಮೌಲ್ಯ = T{}) {
    ಗಾಗಿ (size_t i = 0; i < ಅಡ್ಡಸಾಲುಗಳು; ++i) {
        size_t diag_ಲಂಬಸಾಲು = std::min(i, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 - 1);
        ಮಾತೃಕೆ[i * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 + diag_ಲಂಬಸಾಲು] = zero_ಮೌಲ್ಯ;
    }
}

ov::Tensor make_contiguous_ವಿಭಜಿಸು_ಒಳಹರಿವು_tensor(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ggml_tensor, ಸ್ಥಿರಾಂಕ ov::Shape & ಒಳಹರಿವು_shape) {
    ಸ್ಥಿರಾಂಕ size_t element_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(ggml_tensor->type);
    ಸ್ಥಿರಾಂಕ size_t ಖಂಡ_ಗಾತ್ರ_3 = ggml_blck_ಗಾತ್ರ(ggml_tensor->type);

    GGML_ASSERT(ಖಂಡ_ಗಾತ್ರ_3 == 1 && "non-contiguous split inputs must be plain element types");

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಮೂಲ_tensor = ggml_tensor->view_src != ಶೂನ್ಯಸೂಚಕ ? ggml_tensor->view_src : ggml_tensor;
    ಸ್ಥಿರಾಂಕ size_t ಮೂಲ_ಸ್ಥಾನಾಂತರ = ggml_tensor->view_src != ಶೂನ್ಯಸೂಚಕ ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> ಮೂಲ_ದತ್ತಾಂಶ(ggml_nbytes(ಮೂಲ_tensor));
    ggml_backend_tensor_get(ಮೂಲ_tensor, ಮೂಲ_ದತ್ತಾಂಶ.ದತ್ತಾಂಶ(), 0, ಮೂಲ_ದತ್ತಾಂಶ.size());

    ov::Tensor ಒಳಹರಿವು_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ಒಳಹರಿವು_shape);
    ಸ್ವಯಂಚಾಲಿತ * dst = static_cast<uint8_t *>(ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ());
    size_t dst_ಸ್ಥಾನಾಂತರ = 0;

    ಗಾಗಿ (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        ಗಾಗಿ (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            ಗಾಗಿ (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                ಗಾಗಿ (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    ಸ್ಥಿರಾಂಕ size_t src_ಸ್ಥಾನಾಂತರ = ಮೂಲ_ಸ್ಥಾನಾಂತರ + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::ಸ್ಮರಣೆ_ನಕಲು(dst + dst_ಸ್ಥಾನಾಂತರ, ಮೂಲ_ದತ್ತಾಂಶ.ದತ್ತಾಂಶ() + src_ಸ್ಥಾನಾಂತರ, element_ಗಾತ್ರ);
                    dst_ಸ್ಥಾನಾಂತರ += element_ಗಾತ್ರ;
                }
            }
        }
    }

    ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
}

ov::Tensor convert_ggml_ಒಳಹರಿವು_ಗಮ್ಯ_ov(ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ಸ್ಥಿರಾಂಕ std::string & name) {
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ggml_tensor = ggml_decoder->get_ಒಳಹರಿವು_ggml_tensor(name);

    ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        ಹಿಂತಿರುಗಿ *sliced;
    }

    ಆದರೆ (ggml_tensor->extra != ಶೂನ್ಯಸೂಚಕ && !ggml_decoder->is_splited_model()) {
        ಸ್ವಯಂಚಾಲಿತ * extra_ತಳ = static_cast<ggml_openvino_extra_ತಳ *>(ggml_tensor->extra);
        ಆದರೆ (extra_ತಳ->type == ggml_openvino_extra_ತಳ::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            ಸ್ವಯಂಚಾಲಿತ * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_ತಳ);
            ಹಿಂತಿರುಗಿ *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    ಸ್ವಯಂಚಾಲಿತ * ಒಳಹರಿವು_ದತ್ತಾಂಶ = ggml_tensor->ದತ್ತಾಂಶ;
    ov::Shape ಒಳಹರಿವು_shape;
    ಆದರೆ (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        ಒಳಹರಿವು_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಒಳಹರಿವು_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    ಆದರೆ (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        ಹಿಂತಿರುಗಿ make_contiguous_ವಿಭಜಿಸು_ಒಳಹರಿವು_tensor(ggml_tensor, ಒಳಹರಿವು_shape);
    }

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ಒಳಹರಿವು_shape, ಒಳಹರಿವು_ದತ್ತಾಂಶ);
    ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
}

ov::Tensor get_ov_ಒಳಹರಿವು_tensor(ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ಸ್ಥಿರಾಂಕ std::string & param_name) {
    ov::Tensor ಒಳಹರಿವು_tensor;
    ಸ್ವಯಂಚಾಲಿತ extra_ಒಳಹರಿವು = ggml_decoder->get_model_extra_inputs().find(param_name);
    ಆದರೆ (extra_ಒಳಹರಿವು != ggml_decoder->get_model_extra_inputs().end()) {
        ಒಳಹರಿವು_tensor = ov::Tensor(extra_ಒಳಹರಿವು->second.type, extra_ಒಳಹರಿವು->second.shape);
        *ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<int64_t>() = extra_ಒಳಹರಿವು->second.value;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಒಳಹರಿವು_tensor = convert_ggml_ಒಳಹರಿವು_ಗಮ್ಯ_ov(ggml_decoder, param_name);
    }
    ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
}

ov::Tensor get_ov_ಒಳಹರಿವು_tensor_static_decode(ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             ಸ್ಥಿರಾಂಕ std::string & param_name) {
    // NPU decoding stage
    ಆದರೆ (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        ಹಿಂತಿರುಗಿ get_ov_ಒಳಹರಿವು_tensor(ggml_decoder, param_name);
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ggml_tensor = ggml_decoder->get_ಒಳಹರಿವು_ggml_tensor(param_name);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * op = ggml_decoder->get_tensor_ಬಳಸಿದ_op(ggml_tensor);

    ಆದರೆ (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        ಪ್ರತಿಜ್ಞಾ(ggml_tensor->ne[0] == n_planes);
        ov::Shape ಒಳಹರಿವು_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor ಒಳಹರಿವು_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ಒಳಹರಿವು_shape);
        std::ಸ್ಮರಣೆ_ನಕಲು(ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ(), ggml_tensor->ದತ್ತಾಂಶ, n_planes * ggml_type_ಗಾತ್ರ(ggml_tensor->type));
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }

    ಆದರೆ (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape ಒಳಹರಿವು_shape = {1, 1, 1, 1};
        ov::Tensor ಒಳಹರಿವು_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ಒಳಹರಿವು_shape);
        int32_t inp_ಹೊರಹೋಗು_id = *((int32_t *) ggml_tensor->ದತ್ತಾಂಶ);
        ಪ್ರತಿಜ್ಞಾ(ggml_tensor->ne[0] == 1);
        ಪ್ರತಿಜ್ಞಾ(inp_ಹೊರಹೋಗು_id == 0);
        *ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<int32_t>() = inp_ಹೊರಹೋಗು_id;
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }

    ಆದರೆ (GgmlOvDecoder::is_inp_ಮುಖವಾಡ(ggml_tensor, op)) {
        size_t context_ಗಾತ್ರ = ggml_decoder->get_ctx_ಗಾತ್ರ();
        ಆದರೆ (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_ದತ್ತಾಂಶ =
                pad_ಒಳಹರಿವು<ggml_fp16_t>(ggml_tensor, 1, context_ಗಾತ್ರ, GGML_FP32_ಗಮ್ಯ_FP16(-INFINITY));
            ov::Tensor ಒಳಹರಿವು_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_ಗಾತ್ರ});
            std::ಸ್ಮರಣೆ_ನಕಲು(ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ(), padded_ದತ್ತಾಂಶ.ದತ್ತಾಂಶ(), padded_ದತ್ತಾಂಶ.size() * ಗಾತ್ರ(ggml_fp16_t));
            ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
        }

        std::vector<ದಶಮಾಂಶ> padded_ದತ್ತಾಂಶ = pad_ಒಳಹರಿವು<ದಶಮಾಂಶ>(ggml_tensor, 1, context_ಗಾತ್ರ, -INFINITY);
        ov::Tensor ಒಳಹರಿವು_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_ಗಾತ್ರ});
        ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ_ptr = ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<ದಶಮಾಂಶ>();
        std::copy(padded_ದತ್ತಾಂಶ.begin(), padded_ದತ್ತಾಂಶ.begin() + context_ಗಾತ್ರ, ದತ್ತಾಂಶ_ptr);
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }

    ಹಿಂತಿರುಗಿ get_ov_ಒಳಹರಿವು_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_ಒಳಹರಿವು_tensor_static_prefill(ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              ಸ್ಥಿರಾಂಕ std::string & param_name,
                                              ಪೂರ್ಣಾಂಕ chunk_ಸೂಚ್ಯಂಕ) {
    // NPU prompt processing stage
    ಸ್ಥಿರಾಂಕ size_t ಒಳಹರಿವು_len = ggml_decoder->get_ಒಳಹರಿವು_len();
    ಸ್ಥಿರಾಂಕ size_t chunk_ಗಾತ್ರ = ggml_decoder->m_prefill_chunk_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t chunk_valid_ಗಾತ್ರ = std::min(chunk_ಗಾತ್ರ, ಒಳಹರಿವು_len - chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ size_t chunk_pad_ಗಾತ್ರ = chunk_ಗಾತ್ರ - chunk_valid_ಗಾತ್ರ;

    ಆದರೆ (param_name == "chunk_valid_len") {
        ov::Tensor ಒಳಹರಿವು_tensor(ov::element::i64, ov::Shape{1});
        *ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<int64_t>() = (int64_t) chunk_valid_ಗಾತ್ರ;
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }
    ಆದರೆ (chunk_ಸೂಚ್ಯಂಕ > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor ಒಳಹರಿವು_tensor(ov::element::i64, ov::Shape{1});
        *ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<int64_t>() = 0;
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }
    ಆದರೆ (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        ಹಿಂತಿರುಗಿ get_ov_ಒಳಹರಿವು_tensor(ggml_decoder, param_name);
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ggml_tensor = ggml_decoder->get_ಒಳಹರಿವು_ggml_tensor(param_name);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * op = ggml_decoder->get_tensor_ಬಳಸಿದ_op(ggml_tensor);

    ಆದರೆ (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        ಸ್ಥಿರಾಂಕ size_t element_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(ggml_tensor->type);
        ov::Shape ಒಳಹರಿವು_shape = {1, 1, 1, (size_t) n_planes * chunk_ಗಾತ್ರ};
        ov::Tensor ಒಳಹರಿವು_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ಒಳಹರಿವು_shape);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ p = 0; p < n_planes; p++) {
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src =
                (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ggml_tensor->ದತ್ತಾಂಶ + (p * ಒಳಹರಿವು_len + chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ) * element_ಗಾತ್ರ;
            ಅಕ್ಷರ * dst = (ಅಕ್ಷರ *) ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ() + p * chunk_ಗಾತ್ರ * element_ಗಾತ್ರ;
            std::ಸ್ಮರಣೆ_ನಕಲು(dst, src, chunk_valid_ಗಾತ್ರ * element_ಗಾತ್ರ);
            ಆದರೆ (chunk_pad_ಗಾತ್ರ > 0) {
                ಆದರೆ (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t ಕೊನೆಯ_ಮೌಲ್ಯ = *((ಸ್ಥಿರಾಂಕ int32_t *) src + chunk_valid_ಗಾತ್ರ - 1);
                    int32_t * ಹೊರಹೋಗು = (int32_t *) dst;
                    std::fill(ಹೊರಹೋಗು + chunk_valid_ಗಾತ್ರ, ಹೊರಹೋಗು + chunk_ಗಾತ್ರ, ಕೊನೆಯ_ಮೌಲ್ಯ + 1);
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t ಕೊನೆಯ_ಮೌಲ್ಯ = *((ಸ್ಥಿರಾಂಕ int64_t *) src + chunk_valid_ಗಾತ್ರ - 1);
                    int64_t * ಹೊರಹೋಗು = (int64_t *) dst;
                    std::fill(ಹೊರಹೋಗು + chunk_valid_ಗಾತ್ರ, ಹೊರಹೋಗು + chunk_ಗಾತ್ರ, ಕೊನೆಯ_ಮೌಲ್ಯ + 1);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಎಸೆ std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }

    ಆದರೆ (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape ಒಳಹರಿವು_shape = {1, 1, 1, chunk_ಗಾತ್ರ};
        ov::Tensor ಒಳಹರಿವು_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ಒಳಹರಿವು_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(ggml_tensor->type);
        ಶೂನ್ಯ * ಒಳಹರಿವು_ದತ್ತಾಂಶ = (ಅಕ್ಷರ *) ggml_tensor->ದತ್ತಾಂಶ + chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ * element_ಗಾತ್ರ;
        std::ಸ್ಮರಣೆ_ನಕಲು(ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ(), ಒಳಹರಿವು_ದತ್ತಾಂಶ, chunk_valid_ಗಾತ್ರ * element_ಗಾತ್ರ);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        ಆದರೆ (chunk_pad_ಗಾತ್ರ > 0) {
            ಆದರೆ (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t ಕೊನೆಯ_ಮೌಲ್ಯ =
                    *((int32_t *) ggml_tensor->ದತ್ತಾಂಶ + (chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ + chunk_valid_ಗಾತ್ರ - 1));
                int32_t * output_ದತ್ತಾಂಶ = ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<int32_t>();
                std::fill(output_ದತ್ತಾಂಶ + chunk_valid_ಗಾತ್ರ, output_ದತ್ತಾಂಶ + chunk_ಗಾತ್ರ, ಕೊನೆಯ_ಮೌಲ್ಯ + 1);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t ಕೊನೆಯ_ಮೌಲ್ಯ =
                    *((int64_t *) ggml_tensor->ದತ್ತಾಂಶ + (chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ + chunk_valid_ಗಾತ್ರ - 1));
                int64_t * output_ದತ್ತಾಂಶ = ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<int64_t>();
                std::fill(output_ದತ್ತಾಂಶ + chunk_valid_ಗಾತ್ರ, output_ದತ್ತಾಂಶ + chunk_ಗಾತ್ರ, ಕೊನೆಯ_ಮೌಲ್ಯ + 1);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಎಸೆ std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }

    ಆದರೆ (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape ಒಳಹರಿವು_shape = {1, 1, 1, output_len};
        ov::Tensor ಒಳಹರಿವು_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ಒಳಹರಿವು_shape);
        ಆದರೆ (ggml_tensor->ne[0] == 0) {
            *ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<int32_t>() = 0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ_addr = ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<int32_t>();
            ಗಾಗಿ (size_t i = 0; i < output_len; i++) {
                ದತ್ತಾಂಶ_addr[i] = ((int32_t *) ggml_tensor->ದತ್ತಾಂಶ)[i] % chunk_ಗಾತ್ರ;
            }
        }
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }

    ಆದರೆ (GgmlOvDecoder::is_inp_scale_ಅಡ್ಡಸಾಲುಗಳು(ggml_tensor, op)) {
        ov::Tensor ಒಳಹರಿವು_tensor(ov::element::f32, ov::Shape{1, 1, chunk_ಗಾತ್ರ, 1});
        ಸ್ವಯಂಚಾಲಿತ * dst = ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<ದಶಮಾಂಶ>();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(ggml_tensor->ದತ್ತಾಂಶ) + chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ;
        std::copy(src, src + chunk_valid_ಗಾತ್ರ, dst);
        std::fill(dst + chunk_valid_ಗಾತ್ರ, dst + chunk_ಗಾತ್ರ, 1.0f);
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }

    ಆದರೆ (GgmlOvDecoder::is_inp_ಸರಾಸರಿ(ggml_tensor, op)) {
        ಸ್ಥಿರಾಂಕ size_t n_seqs = ggml_tensor->ne[1];
        ಸ್ಥಿರಾಂಕ size_t src_stride = ggml_tensor->ne[0];
        ಸ್ಥಿರಾಂಕ size_t copy_len = std::min<size_t>(chunk_valid_ಗಾತ್ರ, src_stride - chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ);
        ov::Tensor ಒಳಹರಿವು_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_ಗಾತ್ರ});
        ಸ್ವಯಂಚಾಲಿತ * dst = ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<ದಶಮಾಂಶ>();
        std::fill(dst, dst + n_seqs * chunk_ಗಾತ್ರ, 0.0f);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(ggml_tensor->ದತ್ತಾಂಶ) + chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ;
        ಗಾಗಿ (size_t s = 0; s < n_seqs; s++) {
            std::ಸ್ಮರಣೆ_ನಕಲು(dst + s * chunk_ಗಾತ್ರ, src + s * src_stride, copy_len * ಗಾತ್ರ(ದಶಮಾಂಶ));
        }
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }

    ಆದರೆ (GgmlOvDecoder::is_inp_ಮುಖವಾಡ(ggml_tensor, op)) {
        size_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 = ggml_tensor->ne[0];
        size_t ಅಡ್ಡಸಾಲುಗಳು = ggml_tensor->ne[1];
        size_t chunk_valid_ಅಡ್ಡಸಾಲುಗಳು = std::min(chunk_ಗಾತ್ರ, ಅಡ್ಡಸಾಲುಗಳು - chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ);
        size_t context_ಗಾತ್ರ = ggml_decoder->get_ctx_ಗಾತ್ರ();
        ಆದರೆ (ggml_tensor->type == GGML_TYPE_F16) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ggml_ದತ್ತಾಂಶ =
                static_cast<ಸ್ಥಿರಾಂಕ ggml_fp16_t *>(ggml_tensor->ದತ್ತಾಂಶ) + chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2;
            std::vector<ggml_fp16_t> padded_ದತ್ತಾಂಶ = pad_ಒಳಹರಿವು<ggml_fp16_t>(ggml_ದತ್ತಾಂಶ, chunk_valid_ಅಡ್ಡಸಾಲುಗಳು, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, chunk_ಗಾತ್ರ,
                                                                          context_ಗಾತ್ರ, GGML_FP32_ಗಮ್ಯ_FP16(-INFINITY));
            ಗಣ_zero_diagonal(padded_ದತ್ತಾಂಶ, chunk_ಗಾತ್ರ, context_ಗಾತ್ರ, GGML_FP32_ಗಮ್ಯ_FP16(0.0f));
            ov::Tensor ಒಳಹರಿವು_tensor(ov::element::f16, ov::Shape{1, 1, chunk_ಗಾತ್ರ, context_ಗಾತ್ರ});
            std::ಸ್ಮರಣೆ_ನಕಲು(ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ(), padded_ದತ್ತಾಂಶ.ದತ್ತಾಂಶ(), padded_ದತ್ತಾಂಶ.size() * ಗಾತ್ರ(ggml_fp16_t));
            ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ggml_ದತ್ತಾಂಶ = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(ggml_tensor->ದತ್ತಾಂಶ) + chunk_ಸೂಚ್ಯಂಕ * chunk_ಗಾತ್ರ * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2;
        std::vector<ದಶಮಾಂಶ> padded_ದತ್ತಾಂಶ =
            pad_ಒಳಹರಿವು<ದಶಮಾಂಶ>(ggml_ದತ್ತಾಂಶ, chunk_valid_ಅಡ್ಡಸಾಲುಗಳು, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, chunk_ಗಾತ್ರ, context_ಗಾತ್ರ, -INFINITY);
        ಗಣ_zero_diagonal(padded_ದತ್ತಾಂಶ, chunk_ಗಾತ್ರ, context_ಗಾತ್ರ);
        ov::Tensor ಒಳಹರಿವು_tensor(ov::element::f32, ov::Shape{1, 1, chunk_ಗಾತ್ರ, context_ಗಾತ್ರ});
        ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ_ptr = ಒಳಹರಿವು_tensor.ದತ್ತಾಂಶ<ದಶಮಾಂಶ>();
        std::copy(padded_ದತ್ತಾಂಶ.begin(), padded_ದತ್ತಾಂಶ.begin() + chunk_ಗಾತ್ರ * context_ಗಾತ್ರ, ದತ್ತಾಂಶ_ptr);
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು_tensor;
    }

    ಹಿಂತಿರುಗಿ get_ov_ಒಳಹರಿವು_tensor(ggml_decoder, param_name);
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               ಸ್ಥಿರಾಂಕ std::string & device,
                               ಸ್ಥಿರಾಂಕ ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    ಆದರೆ (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_ಬೀಗ(cache.mutex);
    ತಾರ್ಕಿಕ naive = ನಿಜ;
    ಸ್ವಯಂಚಾಲಿತ model_weights = GgmlOvDecoder::create_ತೂಕ_nodes(cgraph, naive);
    ಸ್ವಯಂಚಾಲಿತ decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_model = std::make_shared<ov::frontend::ggml::ಒಳಹರಿವು_Model>(decoder);
    ಸ್ವಯಂಚಾಲಿತ model = ov::frontend::ggml::Front_ಕೊನೆ::convert(ಒಳಹರಿವು_model, naive);
    ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    ಸ್ವಯಂಚಾಲಿತ remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    ಆದರೆ (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_ಬಹುಲಕ.name()] = ov::hint::Execution_ಬಹುಲಕ::PERFORMANCE;
    } ಇಲ್ಲದಿದ್ದರೆ {
        compile_config[ov::hint::execution_ಬಹುಲಕ.name()] = ov::hint::Execution_ಬಹುಲಕ::ACCURACY;
    }
    ಆದರೆ (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } ಇಲ್ಲದಿದ್ದರೆ {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> ಒಳಹರಿವು_ಹೆಸರುಗಳು;
    std::vector<std::string> output_ಹೆಸರುಗಳು;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & param : model->get_parameters()) {
        ಒಳಹರಿವು_ಹೆಸರುಗಳು.push_back(param->get_friendly_name());
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಫಲಿತಾಂಶ : model->get_results()) {
        output_ಹೆಸರುಗಳು.push_back(ಫಲಿತಾಂಶ->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    ಒಳಹರಿವು_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_ಬೀಗ.unlock();

    ಗಾಗಿ (size_t i = 0; i < ಒಳಹರಿವು_ಹೆಸರುಗಳು.size(); i++) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & param_name = ಒಳಹರಿವು_ಹೆಸರುಗಳು[i];
        ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_tensor = get_ov_ಒಳಹರಿವು_tensor(decoder, param_name);
        infer_request->ಗಣ_ಒಳಹರಿವು_tensor(i, ಒಳಹರಿವು_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    ಗಾಗಿ (size_t i = 0; i < output_ಹೆಸರುಗಳು.size(); i++) {
        ಸ್ವಯಂಚಾಲಿತ output_tensor = infer_request->get_output_tensor(i);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & model_outputs = decoder->get_model_outputs();
        ಸ್ವಯಂಚಾಲಿತ model_output_it = model_outputs.find(output_ಹೆಸರುಗಳು[i]);
        ಆದರೆ (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_ಹೆಸರುಗಳು[i], output_tensor, output_tensor.ದತ್ತಾಂಶ());
            }
            ಮುಂದುವರಿ;
        }
        ಸ್ವಯಂಚಾಲಿತ * ggml_tensor = model_output_it->second;
        std::ಸ್ಮರಣೆ_ನಕಲು(ggml_tensor->ದತ್ತಾಂಶ, output_tensor.ದತ್ತಾಂಶ(), output_tensor.get_byte_ಗಾತ್ರ());
    }
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ov_ನಕ್ಷೆ_compute_dynamic(ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ std::shared_ptr<ov_runtime_context> & r_ctx) {
    ಸ್ವಯಂಚಾಲಿತ & core = ov_singleton_core();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & config = ggml_openvino_get_compile_config();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & device = r_ctx->device;
    ಸ್ಥಿರ ಸ್ವಯಂಚಾಲಿತ is_static = ಸುಳ್ಳು;

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    ಸ್ವಯಂಚಾಲಿತ ಆರಂಭ_ಸಮಯ = ggml_ಸಮಯ_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    ಸ್ಥಿರಾಂಕ int64_t cache_ಕೀಲಿ_ಆರಂಭ_ಸಮಯ = ggml_ಸಮಯ_us();
    ನಕ್ಷೆ_ಕೀಲಿ ಕೀಲಿ(cgraph);
    ಸ್ಥಿರಾಂಕ int64_t cache_ಕೀಲಿ_compute_ಸಮಯ = ggml_ಸಮಯ_us() - cache_ಕೀಲಿ_ಆರಂಭ_ಸಮಯ;
    int64_t cache_lookup_ಸಮಯ = 0;
    ತಾರ್ಕಿಕ ಕೀಲಿ_ನೋಡಿದ = ಸುಳ್ಳು;
    ಆದರೆ (!cache_disabled) {
        ಸ್ಥಿರಾಂಕ int64_t cache_lookup_ಆರಂಭ_ಸಮಯ = ggml_ಸಮಯ_us();
        std::lock_guard<std::mutex> map_ಬೀಗ(r_ctx->ctx_mutex);
        ಕೀಲಿ_ನೋಡಿದ = r_ctx->decoder_cache.find(ಕೀಲಿ) != r_ctx->decoder_cache.end();
        cache_lookup_ಸಮಯ += ggml_ಸಮಯ_us() - cache_lookup_ಆರಂಭ_ಸಮಯ;
    }

    ಸ್ಥಿರಾಂಕ int64_t model_ವಿಭಜಿಸು_check_ಆರಂಭ_ಸಮಯ = ggml_ಸಮಯ_us();
    ತಾರ್ಕಿಕ model_is_splitted = ಕೀಲಿ_ನೋಡಿದ ? ಸುಳ್ಳು : is_model_splitted(cgraph);
    ಸ್ಥಿರಾಂಕ int64_t model_ವಿಭಜಿಸು_check_ಸಮಯ = ggml_ಸಮಯ_us() - model_ವಿಭಜಿಸು_check_ಆರಂಭ_ಸಮಯ;

    ಆದರೆ (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
    }
    ಆದರೆ (is_naive(cgraph)) {
        ಆದರೆ (!model_is_splitted) {
            ಹಿಂತಿರುಗಿ naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    ಸ್ಥಿರಾಂಕ int64_t ನಕ್ಷೆ_param_compute_ಆರಂಭ_ಸಮಯ = ggml_ಸಮಯ_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    ಸ್ಥಿರಾಂಕ int64_t ನಕ್ಷೆ_param_compute_ಸಮಯ = ggml_ಸಮಯ_us() - ನಕ್ಷೆ_param_compute_ಆರಂಭ_ಸಮಯ;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cache_enabled = !model_is_splitted && !cache_disabled;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ stateful_recurrent = r_ctx->stateful && m_params.state_ಗಾತ್ರ >= 0;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    ಆದರೆ (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_ಸಕ್ರಿಯ > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
    }
    ಆದರೆ (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
    }
    ತಾರ್ಕಿಕ cache_ದೊರಕಿತು = ಸುಳ್ಳು;

    int64_t decoder_ಕೊನೆ_ಸಮಯ;
    int64_t conversion_ಕೊನೆ_ಸಮಯ;
    int64_t compile_ಕೊನೆ_ಸಮಯ;
    int64_t infer_ಕೊನೆ_ಸಮಯ;
    int64_t ov_raw_infer_ಆರಂಭ;

    {
        std::shared_ptr<decoder_runtime_ctx> ನಮೂದು;
        ModelParams old_m_params;

        ಸ್ಥಿರಾಂಕ int64_t cache_lookup_ಆರಂಭ_ಸಮಯ = ggml_ಸಮಯ_us();
        ಆದರೆ (cache_enabled) {
            std::lock_guard<std::mutex> map_ಬೀಗ(r_ctx->ctx_mutex);
            ಸ್ವಯಂಚಾಲಿತ it = r_ctx->decoder_cache.find(ಕೀಲಿ);
            cache_ದೊರಕಿತು = it != r_ctx->decoder_cache.end();
            ಆದರೆ (cache_ದೊರಕಿತು) {
                ನಮೂದು = it->second;
            } ಇಲ್ಲದಿದ್ದರೆ {
                r_ctx->clear_caches_locked();
                ಸ್ವಯಂಚಾಲಿತ mutex = std::make_shared<std::mutex>();
                ನಮೂದು = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[ಕೀಲಿ] = ನಮೂದು;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ವಯಂಚಾಲಿತ mutex = std::make_shared<std::mutex>();
            ನಮೂದು = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_ದೊರಕಿತು = ಸುಳ್ಳು;
        }
        cache_lookup_ಸಮಯ += ggml_ಸಮಯ_us() - cache_lookup_ಆರಂಭ_ಸಮಯ;

        std::lock_guard<std::mutex> ಬೀಗ(*(ನಮೂದು->mutex));
        cache_ದೊರಕಿತು = cache_ದೊರಕಿತು && ನಮೂದು->ptr && r_ctx->infer_request_cache.count(ಕೀಲಿ) != 0;

        ಆದರೆ (cache_ದೊರಕಿತು) {
            ggml_decoder = ನಮೂದು->ptr;
            old_m_params = ggml_decoder->get_model_params();
            ಆದರೆ (!ggml_decoder->is_splited_model()) {
                cache_ದೊರಕಿತು = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು;
        std::vector<std::string> ov_output_ಹೆಸರುಗಳು;

        ಆದರೆ (cache_ದೊರಕಿತು) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->ಗಣ_compute_params(c_params);
            ggml_decoder->ಗಣ_model_params(m_params);
            ಆದರೆ (old_m_params.kv_ಬಫರ್_ಬದಲಾಯಿತು(m_params) || !ggml_decoder->is_bound_ಗಮ್ಯ(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->ಕೂಡಿಸು_extra_inputs();
            {
                std::lock_guard<std::mutex> map_ಬೀಗ(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(ಕೀಲಿ);
                ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು = r_ctx->ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_cache.at(ಕೀಲಿ);
                ov_output_ಹೆಸರುಗಳು = r_ctx->ov_output_ಹೆಸರುಗಳು_cache.at(ಕೀಲಿ);
            }

            ಆದರೆ (stateful_kv_only) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_ದತ್ತಾಂಶ = (int32_t *) inp_pos->ದತ್ತಾಂಶ;
                ಸ್ವಯಂಚಾಲಿತ pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                ಆದರೆ (pos_ದತ್ತಾಂಶ[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_ಗಾತ್ರ = pos_shape[3];
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (r_ctx->stateful_kv_ಗಾತ್ರ == static_cast<size_t>(pos_ದತ್ತಾಂಶ[0])) {
                    r_ctx->stateful_kv_ಗಾತ್ರ += pos_shape[3];
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಸ್ಥಿರಾಂಕ size_t pos_begin = static_cast<size_t>(pos_ದತ್ತಾಂಶ[0]);
                    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ refill = pos_begin > r_ctx->stateful_kv_ಗಾತ್ರ;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    ಆದರೆ (refill && !ggml_decoder->get_model_params().swa_layers.ಖಾಲಿಗಳು()) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_swa = ggml_decoder->get_compute_params().swa_ಕಿಟಕಿ;
                        ಆದರೆ (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_ಗಾತ್ರ, n_swa);
                            ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
                        }
                    }

                    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿತಿಗಳು = infer_request->query_state();
                    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ state : ಸ್ಥಿತಿಗಳು) {
                        ಸ್ವಯಂಚಾಲಿತ state_tensor = state.get_state();
                        ಸ್ವಯಂಚಾಲಿತ state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ it = r_ctx->kv_state_ಒಳಹರಿವು_name_map.find(state.get_name());
                            it != r_ctx->kv_state_ಒಳಹರಿವು_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        ಪೂರ್ಣಾಂಕ n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ layer = ಹೊರತೆಗೆ_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ relayout_this_state = relayout_enabled;
                        ಸ್ಥಿರಾಂಕ size_t seq_axis = relayout_this_state ? 2 : 1;
                        ಸ್ಥಿರಾಂಕ size_t ತಲೆ_axis = seq_axis == 2 ? 1 : 2;

                        ಆದರೆ (refill) {
                            ಆದರೆ (state_name.ಖಾಲಿಗಳು()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
                            }
                            ಸ್ವಯಂಚಾಲಿತ kv_tensor = get_ov_ಒಳಹರಿವು_tensor(ggml_decoder, state_name);
                            ಆದರೆ (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                ov::Shape refill_shape(4);
                                refill_shape[0] = state_tensor_shape[0];
                                refill_shape[seq_axis] = kv_tensor.get_shape()[2];
                                refill_shape[ತಲೆ_axis] = state_tensor_shape[ತಲೆ_axis];
                                refill_shape[3] = state_tensor_shape[3];
                                kv_tensor.ಗಣ_shape(refill_shape);
                                state_tensor = kv_tensor;
                            }
                            state_tensor_shape = state_tensor.get_shape();
                        }
                        // Only ever shrink to a prefix the source really has. Slicing past it used to
                        // surface as a bare ov::Exception from the ROI constructor.
                        ಆದರೆ (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end(state_tensor_shape.begin(), state_tensor_shape.end());
                        end[seq_axis] = pos_begin;
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.ಗಣ_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_ಗಾತ್ರ = pos_begin + pos_shape[3];
                }
            }

            decoder_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();
            conversion_ಕೊನೆ_ಸಮಯ = decoder_ಕೊನೆ_ಸಮಯ;
            compile_ಕೊನೆ_ಸಮಯ = decoder_ಕೊನೆ_ಸಮಯ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            ಸ್ವಯಂಚಾಲಿತ shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_ಬೀಗ(shared_cache->mutex);
            ಸ್ವಯಂಚಾಲಿತ ತೂಕ_ಹೆಸರುಗಳು = get_ತೂಕ_ಹೆಸರುಗಳು(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, ತೂಕ_ಹೆಸರುಗಳು,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ಸ್ಥಿರಾಂಕ std::string model_cache_dir = ggml_openvino_model_cache_dir();
            ಸ್ಥಿರಾಂಕ std::string exact_ಕೀಲಿ = cache_enabled ? compiled_ನಕ್ಷೆ_ಕೀಲಿ(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_ಮಾರ್ಗ, manifest_ಮಾರ್ಗ;
            ಆದರೆ (!model_cache_dir.ಖಾಲಿಗಳು() && !model_is_splitted) {
                ಸ್ಥಿರಾಂಕ uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/ನಿಜ, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_ನಕ್ಷೆ_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/ನಿಜ, m_params.rope_params, 16, extra_cfg,
                                                    compiled_ನಕ್ಷೆ_ಕೀಲಿ(cgraph, *ggml_decoder, device, 0, ನಿಜ));
                blob_ಮಾರ್ಗ = ggml_openvino_model_cache_blob_ಮಾರ್ಗ(model_cache_dir, model_fp);
                manifest_ಮಾರ್ಗ = ggml_openvino_model_cache_manifest_ಮಾರ್ಗ(model_cache_dir, model_fp);
            }
            std::string shared_ಕೀಲಿ =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_ಕೀಲಿ) : "";
            ov::CompiledModel shared_model;
            ತಾರ್ಕಿಕ imported = ಸುಳ್ಳು;
            ಸ್ವಯಂಚಾಲಿತ shared_it = shared_cache->graphs.find(shared_ಕೀಲಿ);
            ಆದರೆ (!shared_ಕೀಲಿ.ಖಾಲಿಗಳು() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_ನಕ್ಷೆ(shared_it->second.decode, shared_it->second.ಒಳಹರಿವು_ಹೆಸರುಗಳು,
                                                               shared_it->second.output_ಹೆಸರುಗಳು, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು = shared_it->second.ಒಳಹರಿವು_ಹೆಸರುಗಳು;
                ov_output_ಹೆಸರುಗಳು = shared_it->second.output_ಹೆಸರುಗಳು;
                imported = ನಿಜ;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_ಕೀಲಿ = exact_ಕೀಲಿ;
                shared_it = shared_cache->graphs.find(shared_ಕೀಲಿ);
                ಆದರೆ (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು = shared_it->second.ಒಳಹರಿವು_ಹೆಸರುಗಳು;
                    ov_output_ಹೆಸರುಗಳು = shared_it->second.output_ಹೆಸರುಗಳು;
                    imported = ನಿಜ;
                }
                model_fp = exact_fp;
                blob_ಮಾರ್ಗ = ggml_openvino_model_cache_blob_ಮಾರ್ಗ(model_cache_dir, model_fp);
                manifest_ಮಾರ್ಗ = ggml_openvino_model_cache_manifest_ಮಾರ್ಗ(model_cache_dir, model_fp);
            }
            ಆದರೆ (cache_enabled) {
                std::lock_guard<std::mutex> map_ಬೀಗ(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(ಕೀಲಿ);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            ಆದರೆ (!model_cache_dir.ಖಾಲಿಗಳು()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_ಬಹುಲಕ.name()] = ov::Cache_ಬಹುಲಕ::OPTIMIZE_ವೇಗ;
            }
            ಆದರೆ (!imported && !model_cache_dir.ಖಾಲಿಗಳು() && !model_is_splitted) {
                ಸ್ಥಿರಾಂಕ uint64_t preferred_fp = model_fp;
                ತಾರ್ಕಿಕ dynamic_incompatible = preferred_fp == exact_fp;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ candidate = 0; candidate < 2; ++candidate) {
                    ಆದರೆ (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        ನಿಲ್ಲಿಸು;
                    }
                    ಸ್ಥಿರಾಂಕ uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    ಸ್ಥಿರಾಂಕ std::string candidate_blob =
                        ggml_openvino_model_cache_blob_ಮಾರ್ಗ(model_cache_dir, candidate_fp);
                    ಸ್ಥಿರಾಂಕ std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_ಮಾರ್ಗ(model_cache_dir, candidate_fp);
                    std::ifstream blob_ಒಳ(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    ಆದರೆ (!blob_ಒಳ.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        ಮುಂದುವರಿ;
                    }
                    ಸ್ಥಿರಾಂಕ int64_t import_ಆರಂಭ = ggml_ಸಮಯ_us();
                    ಪ್ರಯತ್ನಿಸು {
                        ov::CompiledModel cm;
                        ಸ್ವಯಂಚಾಲಿತ remote_context = ggml_openvino_get_remote_context();
                        ಆದರೆ (remote_context.has_value()) {
                            cm = core.import_model(blob_ಒಳ, remote_context.value(), mc_config);
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            cm = core.import_model(blob_ಒಳ, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        ಆದರೆ (!saved_inputs.ಖಾಲಿಗಳು() || !saved_outputs.ಖಾಲಿಗಳು()) {
                            ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು = std::move(saved_inputs);
                            ov_output_ಹೆಸರುಗಳು = std::move(saved_outputs);
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & p : cm.inputs()) {
                                ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು.push_back(p.get_ಗಂಟು()->get_friendly_name());
                            }
                            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & o : cm.outputs()) {
                                ov_output_ಹೆಸರುಗಳು.push_back(o.get_ಗಂಟು()->get_friendly_name());
                            }
                        }
                        ಆದರೆ (!compiled_model_matches_ನಕ್ಷೆ(cm, ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು, ov_output_ಹೆಸರುಗಳು, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು.clear();
                            ov_output_ಹೆಸರುಗಳು.clear();
                            ಮುಂದುವರಿ;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        ನಮೂದು->ptr = ggml_decoder;
                        imported = ನಿಜ;
                        model_fp = candidate_fp;
                        blob_ಮಾರ್ಗ = candidate_blob;
                        manifest_ಮಾರ್ಗ = candidate_manifest;
                        ಆದರೆ (candidate_fp == exact_fp) {
                            shared_ಕೀಲಿ = exact_ಕೀಲಿ;
                            shared_it = shared_cache->graphs.find(shared_ಕೀಲಿ);
                        }
                        ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_ಸಮಯ_us() - import_ಆರಂಭ) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = ಸುಳ್ಳು;
                        ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು.clear();
                        ov_output_ಹೆಸರುಗಳು.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                ಆದರೆ (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_ಕೀಲಿ = exact_ಕೀಲಿ;
                    shared_it = shared_cache->graphs.find(shared_ಕೀಲಿ);
                    blob_ಮಾರ್ಗ = ggml_openvino_model_cache_blob_ಮಾರ್ಗ(model_cache_dir, model_fp);
                    manifest_ಮಾರ್ಗ = ggml_openvino_model_cache_manifest_ಮಾರ್ಗ(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            ಆದರೆ (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_ಮಾರ್ಗ.c_str());
                ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
            }
            ಆದರೆ (!imported && ggml_openvino_ತೂಕ_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
            }
            ಆದರೆ (imported) {
                decoder_ಕೊನೆ_ಸಮಯ = conversion_ಕೊನೆ_ಸಮಯ = compile_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ವಯಂಚಾಲಿತ model_weights = GgmlOvDecoder::create_ತೂಕ_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();

                ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_model = std::make_shared<ov::frontend::ggml::ಒಳಹರಿವು_Model>(ggml_decoder);
                model = ov::frontend::ggml::Front_ಕೊನೆ::convert(ಒಳಹರಿವು_model);
                ggml_decoder->clear_model_weights();
                conversion_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();

                ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    ಅಕ್ಷರ timestamped_filename[64];
                    ಸ್ವಯಂಚಾಲಿತ timestamp = (ದೀರ್ಘ ದೀರ್ಘ) ggml_ಸಮಯ_us();
                    snprintf(timestamped_filename, ಗಾತ್ರ(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                ಸ್ಥಿರಾಂಕ ov::AnyMap & compile_config = model_cache_dir.ಖಾಲಿಗಳು() ? config : mc_config;
                ov::CompiledModel compiled_model;
                ಸ್ವಯಂಚಾಲಿತ remote_context = ggml_openvino_get_remote_context();
                ಆದರೆ (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();

                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ov_param : model->get_parameters()) {
                    ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು.push_back(ov_param->get_friendly_name());
                }
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ov_output : model->get_results()) {
                    ov_output_ಹೆಸರುಗಳು.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                ಆದರೆ (!model_cache_dir.ಖಾಲಿಗಳು() && !model_is_splitted && model_fp != 0) {
                    ಪ್ರಯತ್ನಿಸು {
                        ಸ್ಥಿರಾಂಕ std::string blob_tmp = ggml_openvino_model_cache_ತಾತ್ಕಾಲಿಕ_ಮಾರ್ಗ(blob_ಮಾರ್ಗ);
                        ಸ್ಥಿರಾಂಕ std::string manifest_tmp = ggml_openvino_model_cache_ತಾತ್ಕಾಲಿಕ_ಮಾರ್ಗ(manifest_ಮಾರ್ಗ);
                        ಆದರೆ (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು,
                                                                     ov_output_ಹೆಸರುಗಳು)) {
                            std::ofstream blob_ಹೊರಹೋಗು(blob_tmp, std::ios::binary | std::ios::trunc);
                            ಆದರೆ (blob_ಹೊರಹೋಗು.is_open()) {
                                compiled_model.export_model(blob_ಹೊರಹೋಗು);
                                blob_ಹೊರಹೋಗು.close();
                                ಆದರೆ (blob_ಹೊರಹೋಗು.good()) {
                                    ಆದರೆ (std::rename(blob_tmp.c_str(), blob_ಮಾರ್ಗ.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_ಮಾರ್ಗ.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_ಮಾರ್ಗ.c_str());
                                    } ಇಲ್ಲದಿದ್ದರೆ {
                                        std::remove(blob_tmp.c_str());
                                        std::remove(manifest_tmp.c_str());
                                    }
                                } ಇಲ್ಲದಿದ್ದರೆ {
                                    std::remove(blob_tmp.c_str());
                                    std::remove(manifest_tmp.c_str());
                                }
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                ನಮೂದು->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            ನಮೂದು->ptr = ggml_decoder;
            ಆದರೆ (!shared_ಕೀಲಿ.ಖಾಲಿಗಳು() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_ಕೀಲಿ,
                                             ov_compiled_ನಕ್ಷೆ{shared_model, {}, ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು, ov_output_ಹೆಸರುಗಳು});
            }
            ಆದರೆ (cache_enabled) {
                std::lock_guard<std::mutex> map_ಬೀಗ(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[ಕೀಲಿ] = infer_request;
                r_ctx->ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_cache[ಕೀಲಿ] = ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು;
                r_ctx->ov_output_ಹೆಸರುಗಳು_cache[ಕೀಲಿ] = ov_output_ಹೆಸರುಗಳು;
            }

            ಆದರೆ (stateful_kv_only && cache_enabled) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * inp_pos = get_inp_pos_tensor(cgraph);
                ಸ್ವಯಂಚಾಲಿತ pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                ಸ್ಥಿರಾಂಕ int32_t pos_begin = ((int32_t *) inp_pos->ದತ್ತಾಂಶ)[0];
                ಆದರೆ (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
                }
                r_ctx->stateful_kv_ಗಾತ್ರ = pos_shape[3];
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ kv_param_res_ಹೆಸರುಗಳು = ggml_decoder->get_kv_param_res_ಹೆಸರುಗಳು();
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & pair : kv_param_res_ಹೆಸರುಗಳು) {
                    r_ctx->kv_state_ಒಳಹರಿವು_name_map[pair.ಮೊದಲ + pair.second] = pair.ಮೊದಲ;
                }
            }
        }

        ಆದರೆ (stateful_recurrent) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * inp_pos = get_inp_pos_tensor(cgraph);
            ಸ್ಥಿರಾಂಕ int32_t pos_begin = static_cast<ಸ್ಥಿರಾಂಕ int32_t *>(inp_pos->ದತ್ತಾಂಶ)[0];
            ಆದರೆ (pos_begin == 0) {
                infer_request->reset_state();
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!cache_ದೊರಕಿತು || old_m_params.kv_ಬಫರ್_ಬದಲಾಯಿತು(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_ಗಾತ್ರ) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            reset_single_ಸ್ಥಾನ_recurrent_cache(cgraph, m_params, c_params);
        }

        ಗಾಗಿ (size_t i = 0; i < ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು.size(); i++) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & param_name = ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು[i];
            ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_tensor = get_ov_ಒಳಹರಿವು_tensor(ggml_decoder, param_name);
            infer_request->ಗಣ_ಒಳಹರಿವು_tensor(i, ಒಳಹರಿವು_tensor);

            ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_ಒಳಹರಿವು_tensor_info(param_name, ಒಳಹರಿವು_tensor);
            }
        }

        ಗಾಗಿ (size_t i = 0; i < ov_output_ಹೆಸರುಗಳು.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & model_outputs = ggml_decoder->get_model_outputs();
            ಸ್ವಯಂಚಾಲಿತ model_output_it = model_outputs.find(ov_output_ಹೆಸರುಗಳು[i]);
            ಆದರೆ (model_output_it == model_outputs.end()) {
                ಮುಂದುವರಿ;
            }
            ಸ್ವಯಂಚಾಲಿತ * ggml_tensor = model_output_it->second;
            ಆದರೆ (ggml_nbytes(ggml_tensor) == 0) {
                ಮುಂದುವರಿ;
            }
            ಸ್ವಯಂಚಾಲಿತ output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->ಗಣ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_ಆರಂಭ = ggml_ಸಮಯ_us();
        infer_request->infer();
        infer_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();
        ಆದರೆ (stateful_recurrent) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_ಗಾತ್ರ =
                static_cast<ಸ್ಥಿರಾಂಕ int32_t *>(inp_pos->ದತ್ತಾಂಶ)[0] + get_inp_pos_n_ಟೋಕನುಗಳು(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            ಗಾಗಿ (size_t i = 0; i < ov_output_ಹೆಸರುಗಳು.size(); i++) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_ಹೆಸರುಗಳು[i], output_tensor, output_tensor.ದತ್ತಾಂಶ());
            }
        }

        ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_ಕೊನೆ_ಸಮಯ - ಆರಂಭ_ಸಮಯ) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_ಕೀಲಿ_compute_ಸಮಯ / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_ವಿಭಜಿಸು_check_ಸಮಯ / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", ನಕ್ಷೆ_param_compute_ಸಮಯ / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_ಸಮಯ / 1000.0);
            ಆದರೆ (!cache_ದೊರಕಿತು) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_ಕೊನೆ_ಸಮಯ - decoder_ಕೊನೆ_ಸಮಯ) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_ಕೊನೆ_ಸಮಯ - conversion_ಕೊನೆ_ಸಮಯ) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_ಕೊನೆ_ಸಮಯ - compile_ಕೊನೆ_ಸಮಯ) / 1000.0);
            GGML_LOG_INFO("    - OV raw infer time: %.3f ms \n", (infer_ಕೊನೆ_ಸಮಯ - ov_raw_infer_ಆರಂಭ) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. Release only while holding the compilation mutex so another context cannot
    // be reading host weights during conversion/compilation. Pin the shared compiled
    // models across backend teardown; a later context can create its own request without
    // reading the dropped pages. A new, uncached graph still fails fast above.
    ಆದರೆ (cache_ದೊರಕಿತು && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_ಬೀಗ(r_ctx->compiled_cache->mutex);
        ಆದರೆ (!ggml_openvino_ತೂಕ_buffers_released()) {
            ggml_openvino_release_ತೂಕ_buffers();
        }
    }

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ov::AnyMap without_npuw(ಸ್ಥಿರಾಂಕ ov::AnyMap & config) {
    ov::AnyMap ಹೊರಹೋಗು;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & kv : config) {
        ಆದರೆ (kv.ಮೊದಲ.rfind("NPUW", 0) == 0 || kv.ಮೊದಲ == "NPU_USE_NPUW") {
            ಮುಂದುವರಿ;
        }
        ಹೊರಹೋಗು.insert(kv);
    }
    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ov_ನಕ್ಷೆ_compute_static(ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ std::shared_ptr<ov_runtime_context> & r_ctx) {
    ಸ್ವಯಂಚಾಲಿತ & core = ov_singleton_core();

    ಸ್ವಯಂಚಾಲಿತ get_prefill_chunk_ಗಾತ್ರ = [] {
        ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ chunk_ಗಾತ್ರ = []() {
            ಪೂರ್ಣಾಂಕ env_prefill_chunk_ಗಾತ್ರ = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            ಹಿಂತಿರುಗಿ env_prefill_chunk_ಗಾತ್ರ > 0 ? env_prefill_chunk_ಗಾತ್ರ : 256;
        }();
        ಹಿಂತಿರುಗಿ chunk_ಗಾತ್ರ;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    ಸ್ಥಿರ std::string device = ggml_openvino_get_device_name();
    ಸ್ಥಿರ ಸ್ವಯಂಚಾಲಿತ is_static = ನಿಜ;
    ಸ್ಥಿರ ಸ್ವಯಂಚಾಲಿತ stateful = ಸುಳ್ಳು;

    ಸ್ವಯಂಚಾಲಿತ prefill_chunk_ಗಾತ್ರ = get_prefill_chunk_ಗಾತ್ರ();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & config = ggml_openvino_get_compile_config();

    ಆದರೆ (is_naive(cgraph)) {
        ಹಿಂತಿರುಗಿ naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    ಸ್ವಯಂಚಾಲಿತ ಆರಂಭ_ಸಮಯ = ggml_ಸಮಯ_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * inp_pos = get_inp_pos_tensor(cgraph);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ no_kv_cache = m_params.is_cacheless_attn;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ is_prefill = no_kv_cache ? ನಿಜ : get_is_prefill(cgraph, inp_pos);
    ಸ್ಥಿರಾಂಕ ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    ಆದರೆ (m_params.n_heads_kv == -1) {
        prefill_chunk_ಗಾತ್ರ = inp_pos->ne[0];
    }
    ನಕ್ಷೆ_ಕೀಲಿ ಕೀಲಿ(cgraph);
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    ತಾರ್ಕಿಕ cache_ದೊರಕಿತು = ಸುಳ್ಳು;

    int64_t decoder_ಕೊನೆ_ಸಮಯ;
    int64_t conversion_ಕೊನೆ_ಸಮಯ;
    int64_t compile_ಕೊನೆ_ಸಮಯ;
    int64_t infer_ಕೊನೆ_ಸಮಯ;
    int64_t ov_raw_infer_ಆರಂಭ;
    int64_t ov_raw_infer_ಒಟ್ಟು = 0;

    std::shared_ptr<decoder_runtime_ctx> ನಮೂದು;
    ModelParams old_m_params;

    ಆದರೆ (cache_enabled) {
        std::lock_guard<std::mutex> map_ಬೀಗ(r_ctx->ctx_mutex);
        ಸ್ವಯಂಚಾಲಿತ it = r_ctx->decoder_cache.find(ಕೀಲಿ);
        cache_ದೊರಕಿತು = it != r_ctx->decoder_cache.end();
        ಆದರೆ (cache_ದೊರಕಿತು) {
            ನಮೂದು = it->second;
        } ಇಲ್ಲದಿದ್ದರೆ {
            r_ctx->clear_caches_locked();
            ಸ್ವಯಂಚಾಲಿತ mutex = std::make_shared<std::mutex>();
            ನಮೂದು = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[ಕೀಲಿ] = ನಮೂದು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ವಯಂಚಾಲಿತ mutex = std::make_shared<std::mutex>();
        ನಮೂದು = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_ದೊರಕಿತು = ಸುಳ್ಳು;
    }

    std::lock_guard<std::mutex> ಬೀಗ(*(ನಮೂದು->mutex));
    cache_ದೊರಕಿತು = cache_ದೊರಕಿತು && ನಮೂದು->ptr && r_ctx->infer_request_cache.count(ಕೀಲಿ) != 0 &&
                r_ctx->infer_request_cache_prefill.count(ಕೀಲಿ) != 0;

    ಆದರೆ (cache_ದೊರಕಿತು) {
        ggml_decoder = ನಮೂದು->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_ದೊರಕಿತು = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local;
    std::vector<std::string> ov_output_ಹೆಸರುಗಳು_local;

    ಆದರೆ (cache_ದೊರಕಿತು) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->ಗಣ_model_params(m_params);
        ggml_decoder->ಗಣ_compute_params(c_params);
        ಆದರೆ (old_m_params.kv_ಬಫರ್_ಬದಲಾಯಿತು(m_params) || !ggml_decoder->is_bound_ಗಮ್ಯ(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->ಕೂಡಿಸು_extra_inputs();
        {
            std::lock_guard<std::mutex> map_ಬೀಗ(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(ಕೀಲಿ) : r_ctx->infer_request_cache.at(ಕೀಲಿ);
            ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local = r_ctx->ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_cache.at(ಕೀಲಿ);
            ov_output_ಹೆಸರುಗಳು_local = r_ctx->ov_output_ಹೆಸರುಗಳು_cache.at(ಕೀಲಿ);
        }

        decoder_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();
        conversion_ಕೊನೆ_ಸಮಯ = decoder_ಕೊನೆ_ಸಮಯ;
        compile_ಕೊನೆ_ಸಮಯ = decoder_ಕೊನೆ_ಸಮಯ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (cache_enabled) {
            std::lock_guard<std::mutex> map_ಬೀಗ(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(ಕೀಲಿ);
            r_ctx->infer_request_cache_prefill.erase(ಕೀಲಿ);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        ಸ್ವಯಂಚಾಲಿತ shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_ಬೀಗ(shared_cache->mutex);
        ಸ್ವಯಂಚಾಲಿತ ತೂಕ_ಹೆಸರುಗಳು = get_ತೂಕ_ಹೆಸರುಗಳು(cgraph);
        ಸ್ವಯಂಚಾಲಿತ local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, ತೂಕ_ಹೆಸರುಗಳು, is_static,
                                                             stateful, ಸುಳ್ಳು, is_prefill, prefill_chunk_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ std::string shared_ಕೀಲಿ =
            cache_enabled ? compiled_ನಕ್ಷೆ_ಕೀಲಿ(cgraph, *local_decoder, device, prefill_chunk_ಗಾತ್ರ) : "";
        ಸ್ವಯಂಚಾಲಿತ shared_it = shared_cache->graphs.find(shared_ಕೀಲಿ);
        ಆದರೆ (!shared_ಕೀಲಿ.ಖಾಲಿಗಳು() && shared_it != shared_cache->graphs.end()) {
            ಸ್ವಯಂಚಾಲಿತ & compiled = shared_it->second;
            ಸ್ವಯಂಚಾಲಿತ prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            ಸ್ವಯಂಚಾಲಿತ decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            ನಮೂದು->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local = compiled.ಒಳಹರಿವು_ಹೆಸರುಗಳು;
            ov_output_ಹೆಸರುಗಳು_local = compiled.output_ಹೆಸರುಗಳು;
            r_ctx->infer_request_cache_prefill[ಕೀಲಿ] = prefill_request;
            r_ctx->infer_request_cache[ಕೀಲಿ] = decode_request;
            r_ctx->ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_cache[ಕೀಲಿ] = ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local;
            r_ctx->ov_output_ಹೆಸರುಗಳು_cache[ಕೀಲಿ] = ov_output_ಹೆಸರುಗಳು_local;
            decoder_ಕೊನೆ_ಸಮಯ = conversion_ಕೊನೆ_ಸಮಯ = compile_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } ಇಲ್ಲದಿದ್ದರೆ {
            std::shared_ptr<ov::Model> model;
            ಸ್ವಯಂಚಾಲಿತ model_weights = GgmlOvDecoder::create_ತೂಕ_nodes(cgraph);

            ಸ್ವಯಂಚಾಲಿತ ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, ಸುಳ್ಳು, ನಿಜ, prefill_chunk_ಗಾತ್ರ);
            ಸ್ವಯಂಚಾಲಿತ ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, ಸುಳ್ಳು, ಸುಳ್ಳು, prefill_chunk_ಗಾತ್ರ);
            decoder_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();

            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ dump_ir_timestamp = static_cast<ದೀರ್ಘ ದೀರ್ಘ>(ggml_ಸಮಯ_us());

            ಸ್ವಯಂಚಾಲಿತ build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlOvDecoder> & decoder, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಟ್ಯಾಗ್,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_ಕೊನೆ_ಸಮಯ, int64_t & local_compile_ಕೊನೆ_ಸಮಯ) {
                ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_model = std::make_shared<ov::frontend::ggml::ಒಳಹರಿವು_Model>(decoder);
                model = ov::frontend::ggml::Front_ಕೊನೆ::convert(ಒಳಹರಿವು_model);
                decoder->clear_model_weights();
                local_conversion_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();

                ಆದರೆ (dump_ir) {
                    ಅಕ್ಷರ timestamped_filename[64];
                    snprintf(timestamped_filename, ಗಾತ್ರ(timestamped_filename), "model_%s_%lld.xml", ಟ್ಯಾಗ್,
                             dump_ir_timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                compiled_model = core.compile_model(model, device, compile_config);
                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                local_compile_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();
            };
            std::shared_ptr<ov::Model> model_prefill;
            std::shared_ptr<ov::Model> model_decode;
            ov::CompiledModel compiled_model_prefill;
            ov::CompiledModel compiled_model_decode;
            std::shared_ptr<ov::InferRequest> infer_request_prefill;
            std::shared_ptr<ov::InferRequest> infer_request_decode;
            int64_t prefill_conversion_ಕೊನೆ_ಸಮಯ;
            int64_t decode_conversion_ಕೊನೆ_ಸಮಯ;
            int64_t prefill_compile_ಕೊನೆ_ಸಮಯ;
            int64_t decode_compile_ಕೊನೆ_ಸಮಯ;
            build_static_model(ggml_decoder_prefill, "prefill", model_prefill, compiled_model_prefill,
                               infer_request_prefill, prefill_conversion_ಕೊನೆ_ಸಮಯ, prefill_compile_ಕೊನೆ_ಸಮಯ);
            ಆದರೆ (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_ಕೊನೆ_ಸಮಯ = prefill_conversion_ಕೊನೆ_ಸಮಯ;
                decode_compile_ಕೊನೆ_ಸಮಯ = prefill_compile_ಕೊನೆ_ಸಮಯ;
            } ಇಲ್ಲದಿದ್ದರೆ {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_ಕೊನೆ_ಸಮಯ, decode_compile_ಕೊನೆ_ಸಮಯ);
            }
            conversion_ಕೊನೆ_ಸಮಯ = std::max(prefill_conversion_ಕೊನೆ_ಸಮಯ, decode_conversion_ಕೊನೆ_ಸಮಯ);
            compile_ಕೊನೆ_ಸಮಯ = std::max(prefill_compile_ಕೊನೆ_ಸಮಯ, decode_compile_ಕೊನೆ_ಸಮಯ);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            ನಮೂದು->ptr = ggml_decoder;

            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ov_param : model->get_parameters()) {
                ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local.push_back(ov_param->get_friendly_name());
            }
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ov_output : model->get_results()) {
                ov_output_ಹೆಸರುಗಳು_local.push_back(ov_output->get_friendly_name());
            }

            ಆದರೆ (!shared_ಕೀಲಿ.ಖಾಲಿಗಳು()) {
                shared_cache->graphs.emplace(
                    shared_ಕೀಲಿ, ov_compiled_ನಕ್ಷೆ{compiled_model_decode, compiled_model_prefill, ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local,
                                                  ov_output_ಹೆಸರುಗಳು_local});
            }

            ಆದರೆ (cache_enabled) {
                std::lock_guard<std::mutex> map_ಬೀಗ(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[ಕೀಲಿ] = infer_request_prefill;
                r_ctx->infer_request_cache[ಕೀಲಿ] = infer_request_decode;
                r_ctx->ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_cache[ಕೀಲಿ] = ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local;
                r_ctx->ov_output_ಹೆಸರುಗಳು_cache[ಕೀಲಿ] = ov_output_ಹೆಸರುಗಳು_local;
            }
        }
    }

    reset_single_ಸ್ಥಾನ_recurrent_cache(cgraph, m_params, c_params);

    ಆದರೆ (is_prefill) {
        ಸ್ವಯಂಚಾಲಿತ inp_len = get_inp_pos_n_ಟೋಕನುಗಳು(cgraph, inp_pos);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ chunk_ಸೂಚ್ಯಂಕ = 0; chunk_ಸೂಚ್ಯಂಕ * prefill_chunk_ಗಾತ್ರ < inp_len; chunk_ಸೂಚ್ಯಂಕ++) {
            ಗಾಗಿ (size_t i = 0; i < ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local.size(); i++) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & param_name = ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local[i];
                ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_tensor = get_ov_ಒಳಹರಿವು_tensor_static_prefill(ggml_decoder, param_name, chunk_ಸೂಚ್ಯಂಕ);
                infer_request->ಗಣ_ಒಳಹರಿವು_tensor(i, ಒಳಹರಿವು_tensor);

                ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_tensor = infer_request->get_ಒಳಹರಿವು_tensor(i);
                    print_ಒಳಹರಿವು_tensor_info(param_name, ಒಳಹರಿವು_tensor);
                }
            }

            ಗಾಗಿ (size_t i = 0; i < ov_output_ಹೆಸರುಗಳು_local.size(); i++) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & model_outputs = ggml_decoder->get_model_outputs();
                ಸ್ವಯಂಚಾಲಿತ model_output_it = model_outputs.find(ov_output_ಹೆಸರುಗಳು_local[i]);
                ಆದರೆ (model_output_it == model_outputs.end()) {
                    ಮುಂದುವರಿ;
                }
                ಸ್ವಯಂಚಾಲಿತ * ggml_tensor = model_output_it->second;
                ಆದರೆ (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    ಮುಂದುವರಿ;
                }
                ಸ್ವಯಂಚಾಲಿತ output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->ಗಣ_output_tensor(i, output_tensor);
            }

            ov_raw_infer_ಆರಂಭ = ggml_ಸಮಯ_us();
            infer_request->infer();
            ov_raw_infer_ಒಟ್ಟು += ggml_ಸಮಯ_us() - ov_raw_infer_ಆರಂಭ;

            ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                ಗಾಗಿ (size_t i = 0; i < ov_output_ಹೆಸರುಗಳು_local.size(); i++) {
                    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_ಹೆಸರುಗಳು_local[i], output_tensor, output_tensor.ದತ್ತಾಂಶ());
                }
            }
        }
        infer_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (size_t i = 0; i < ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local.size(); i++) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & param_name = ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_local[i];
            ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_tensor = get_ov_ಒಳಹರಿವು_tensor_static_decode(ggml_decoder, param_name);
            infer_request->ಗಣ_ಒಳಹರಿವು_tensor(i, ಒಳಹರಿವು_tensor);

            ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_tensor = infer_request->get_ಒಳಹರಿವು_tensor(i);
                print_ಒಳಹರಿವು_tensor_info(param_name, ಒಳಹರಿವು_tensor);
            }
        }

        ಗಾಗಿ (size_t i = 0; i < ov_output_ಹೆಸರುಗಳು_local.size(); i++) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & model_outputs = ggml_decoder->get_model_outputs();
            ಸ್ವಯಂಚಾಲಿತ model_output_it = model_outputs.find(ov_output_ಹೆಸರುಗಳು_local[i]);
            ಆದರೆ (model_output_it == model_outputs.end()) {
                ಮುಂದುವರಿ;
            }
            ಸ್ವಯಂಚಾಲಿತ * ggml_tensor = model_output_it->second;
            ಆದರೆ (ggml_nbytes(ggml_tensor) == 0) {
                ಮುಂದುವರಿ;
            }
            ಸ್ವಯಂಚಾಲಿತ output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->ಗಣ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_ಆರಂಭ = ggml_ಸಮಯ_us();
        infer_request->infer();
        infer_ಕೊನೆ_ಸಮಯ = ggml_ಸಮಯ_us();
        ov_raw_infer_ಒಟ್ಟು = infer_ಕೊನೆ_ಸಮಯ - ov_raw_infer_ಆರಂಭ;

        ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            ಗಾಗಿ (size_t i = 0; i < ov_output_ಹೆಸರುಗಳು_local.size(); i++) {
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_ಹೆಸರುಗಳು_local[i], output_tensor, output_tensor.ದತ್ತಾಂಶ());
            }
        }
    }

    ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_ಕೊನೆ_ಸಮಯ - ಆರಂಭ_ಸಮಯ) / 1000.0);
        ಆದರೆ (!cache_ದೊರಕಿತು) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_ಕೊನೆ_ಸಮಯ - decoder_ಕೊನೆ_ಸಮಯ) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_ಕೊನೆ_ಸಮಯ - conversion_ಕೊನೆ_ಸಮಯ) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_ಕೊನೆ_ಸಮಯ - compile_ಕೊನೆ_ಸಮಯ) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_ಒಟ್ಟು / 1000.0);
    }

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
ಸ್ಥಿರ ggml_cgraph * get_compute_ನಕ್ಷೆ(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    ಸ್ವಯಂಚಾಲಿತ is_skipped = [](ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
        ಹಿಂತಿರುಗಿ ಗಂಟು->op != GGML_OP_NONE && !(ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    ಆದರೆ (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        ಹಿಂತಿರುಗಿ cgraph;
    }
    ಸ್ವಯಂಚಾಲಿತ & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) { ಹಿಂತಿರುಗಿ !is_skipped(ಗಂಟು); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.ದತ್ತಾಂಶ();
    compute.graph.n_nodes = (ಪೂರ್ಣಾಂಕ) compute.nodes.size();
    ಹಿಂತಿರುಗಿ &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ov_ನಕ್ಷೆ_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    ಪ್ರಯತ್ನಿಸು {
        ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != ಶೂನ್ಯಸೂಚಕ);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_ಬೀಗ(r_ctx->execution_mutex);
        cgraph = get_compute_ನಕ್ಷೆ(cgraph, *r_ctx);

        ಹಿಂತಿರುಗಿ is_static ? ov_ನಕ್ಷೆ_compute_static(cgraph, r_ctx) : ov_ನಕ್ಷೆ_compute_dynamic(cgraph, r_ctx);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
    } ಹಿಡಿ (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
ತಾರ್ಕಿಕ is_model_splitted(ggml_cgraph * cgraph) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    ಆದರೆ (!fallback_enabled) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    ಆದರೆ (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = cgraph->nodes[i];
        ಪೂರ್ಣಾಂಕ use_ಎಣಿಕೆ = cgraph->use_counts[ggml_hash_ಕಂಡುಹಿಡಿ(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ, ಗಂಟು)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        ಆದರೆ ((cgraph->n_nodes <= 1 && use_ಎಣಿಕೆ == 0) ||
            (cgraph->n_nodes <= 1 && ಗಂಟು->op == GGML_OP_VIEW && use_ಎಣಿಕೆ == 1 && ಗಂಟು->src[0] != ಶೂನ್ಯಸೂಚಕ &&
             ಗಂಟು->src[0]->op == GGML_OP_NONE)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_use_ಎಣಿಕೆ = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_ಗಂಟು = cgraph->nodes[j];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < GGML_MAX_SRC; k++) {
                ಆದರೆ (other_ಗಂಟು->src[k] == ಗಂಟು) {
                    ಒಳಹರಿವು_use_ಎಣಿಕೆ++;
                }
            }
        }
        ಆದರೆ (use_ಎಣಿಕೆ != ಒಳಹರಿವು_use_ಎಣಿಕೆ && ಗಂಟು->op != GGML_OP_NONE) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    ಆದರೆ (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_ತೂಕ_ಹೆಸರುಗಳು(cgraph);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & kv : GgmlOvDecoder::create_ತೂಕ_nodes(cgraph, ನಿಜ)) {
            model_weights.insert(kv.ಮೊದಲ);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = cgraph->nodes[i];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = ಗಂಟು->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            ಆದರೆ (src != ಶೂನ್ಯಸೂಚಕ && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.ಖಾಲಿಗಳು() == ಸುಳ್ಳು &&
                model_leafs.find(src) == model_leafs.end()) {
                ಆದರೆ (GgmlOvDecoder::is_inp_tok(src, ಗಂಟು)) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ತಾರ್ಕಿಕ is_naive(ggml_cgraph * cgraph) {
    constexpr ಪೂರ್ಣಾಂಕ naive_ನಕ್ಷೆ_ಗಾತ್ರ_ಮಿತಿ = 20;
    ಪೂರ್ಣಾಂಕ count = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ಆದರೆ (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    ಹಿಂತಿರುಗಿ count < naive_ನಕ್ಷೆ_ಗಾತ್ರ_ಮಿತಿ;
}

size_t checksum(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size) {
    ಸ್ಥಿರಾಂಕ uint8_t * ಬೈಟುಗಳು = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(ದತ್ತಾಂಶ);
    size_t ಮೊತ್ತ = 0;
    ಗಾಗಿ (size_t i = 0; i < size; ++i) {
        ಮೊತ್ತ += (uint8_t) i;
        ಮೊತ್ತ += ಬೈಟುಗಳು[i];
    }
    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

ತಾರ್ಕಿಕ save_ggml_tensor_ದತ್ತಾಂಶ_ಗಮ್ಯ_txt(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ std::string & file_ಮಾರ್ಗ) {
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ || tensor->ದತ್ತಾಂಶ == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    std::ofstream ಹೊರಹೋಗು(file_ಮಾರ್ಗ);
    ಆದರೆ (!ಹೊರಹೋಗು.is_open()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ size_t n = ggml_nelements(tensor);
    ಹೊರಹೋಗು << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    ಆಯ್ಕೆ (tensor->type) {
    ಸ್ಥಿತಿ GGML_TYPE_F32: {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(tensor->ದತ್ತಾಂಶ);
        ಗಾಗಿ (size_t i = 0; i < n; ++i) {
            ಹೊರಹೋಗು << ದತ್ತಾಂಶ[i] << '\n';
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_TYPE_F16: {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<ಸ್ಥಿರಾಂಕ ggml_fp16_t *>(tensor->ದತ್ತಾಂಶ);
        ಗಾಗಿ (size_t i = 0; i < n; ++i) {
            ಹೊರಹೋಗು << ggml_fp16_ಗಮ್ಯ_fp32(ದತ್ತಾಂಶ[i]) << '\n';
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_TYPE_BF16: {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<ಸ್ಥಿರಾಂಕ ggml_bf16_t *>(tensor->ದತ್ತಾಂಶ);
        ಗಾಗಿ (size_t i = 0; i < n; ++i) {
            ಹೊರಹೋಗು << ggml_bf16_ಗಮ್ಯ_fp32(ದತ್ತಾಂಶ[i]) << '\n';
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_TYPE_I32: {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<ಸ್ಥಿರಾಂಕ int32_t *>(tensor->ದತ್ತಾಂಶ);
        ಗಾಗಿ (size_t i = 0; i < n; ++i) {
            ಹೊರಹೋಗು << ದತ್ತಾಂಶ[i] << '\n';
        }
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ GGML_TYPE_I64: {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ದತ್ತಾಂಶ = static_cast<ಸ್ಥಿರಾಂಕ int64_t *>(tensor->ದತ್ತಾಂಶ);
        ಗಾಗಿ (size_t i = 0; i < n; ++i) {
            ಹೊರಹೋಗು << ದತ್ತಾಂಶ[i] << '\n';
        }
        ನಿಲ್ಲಿಸು;
    }
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹೊರಹೋಗು << "unsupported tensor type for text dump" << '\n';
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಶೂನ್ಯ print_ಒಳಹರಿವು_tensor_info(ಸ್ಥಿರಾಂಕ std::string & name, ಸ್ಥಿರಾಂಕ ov::Tensor & tensor) {
    std::ಹೊರಗೆ << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.ದತ್ತಾಂಶ()
              << '\n';
    ಆಯ್ಕೆ (tensor.get_element_type()) {
    ಸ್ಥಿತಿ ov::element::f32: {
        ಆದರೆ (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::ಹೊರಗೆ << *(tensor.ದತ್ತಾಂಶ<ದಶಮಾಂಶ>()) << '\n';
        } ಇಲ್ಲದಿದ್ದರೆ {
            size_t ಅಡ್ಡಸಾಲುಗಳು = tensor.get_shape()[2];
            size_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 = tensor.get_shape()[3];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ದತ್ತಾಂಶ = tensor.ದತ್ತಾಂಶ<ದಶಮಾಂಶ>();
            ಗಾಗಿ (size_t i = 0; i < ಅಡ್ಡಸಾಲುಗಳು; ++i) {
                ಗಾಗಿ (size_t j = 0; j < ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2; ++j) {
                    ದಶಮಾಂಶ val = ದತ್ತಾಂಶ[i * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 + j];
                    ಆದರೆ (std::isinf(val) && val < 0) {
                        std::ಹೊರಗೆ << std::setw(5) << "-inf";
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        std::ಹೊರಗೆ << std::setw(5) << val;
                    }
                }
                std::ಹೊರಗೆ << '\n';
            }
        }

        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ ov::element::f16:
        std::ಹೊರಗೆ << *(tensor.ದತ್ತಾಂಶ<ov::float16>()) << '\n';
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ ov::element::i32:
        ಗಾಗಿ (size_t i = 0; i < tensor.get_ಗಾತ್ರ(); ++i) {
            std::ಹೊರಗೆ << tensor.ದತ್ತಾಂಶ<int32_t>()[i] << ' ';
        }
        std::ಹೊರಗೆ << '\n';
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ ov::element::i64:
        ಗಾಗಿ (size_t i = 0; i < tensor.get_ಗಾತ್ರ(); ++i) {
            std::ಹೊರಗೆ << tensor.ದತ್ತಾಂಶ<int64_t>()[i] << ' ';
        }
        std::ಹೊರಗೆ << '\n';
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ನಿಲ್ಲಿಸು;
    }
}

ಶೂನ್ಯ print_output_tensor_info(ಸ್ಥಿರಾಂಕ std::string & name, ಸ್ಥಿರಾಂಕ ov::Tensor & tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * output_dst) {
    std::ಹೊರಗೆ << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    ಸ್ವಯಂಚಾಲಿತ print_float_stats = [](ಸ್ಥಿರಾಂಕ std::string & type_name, size_t size, ಸ್ವಯಂಚಾಲಿತ get_ಮೌಲ್ಯ) {
        ಆದರೆ (size == 0) {
            ಹಿಂತಿರುಗಿ;
        }

        ದಶಮಾಂಶ ಮೊದಲ = get_ಮೌಲ್ಯ(0);
        ದಶಮಾಂಶ min = ಮೊದಲ;
        ದಶಮಾಂಶ max = ಮೊದಲ;
        ದ್ವಿಗುಣ ಮೊತ್ತ = ಮೊದಲ;

        ಗಾಗಿ (size_t i = 1; i < size; ++i) {
            ದಶಮಾಂಶ v = get_ಮೌಲ್ಯ(i);
            min = std::min(v, min);
            max = std::max(v, max);
            ಮೊತ್ತ += v;
        }
        ದ್ವಿಗುಣ ಸರಾಸರಿ_2 = ಮೊತ್ತ / size;

        std::ಹೊರಗೆ << std::ಬಲ << std::setw(6) << type_name << std::ಬಲ << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::ಹೊರಗೆ << std::ಬಲ << std::setw(6) << "" << std::ಬಲ << std::setw(12) << ಮೊದಲ << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << ಸರಾಸರಿ_2 << '\n';
    };

    ಆಯ್ಕೆ (tensor.get_element_type()) {
    ಸ್ಥಿತಿ ov::element::f32: {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ದತ್ತಾಂಶ = tensor.ದತ್ತಾಂಶ<ದಶಮಾಂಶ>();
        size_t size = tensor.get_ಗಾತ್ರ();
        print_float_stats("[f32]", size, [ದತ್ತಾಂಶ](size_t i) { ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ[i]; });
        ನಿಲ್ಲಿಸು;
    }
    ಸ್ಥಿತಿ ov::element::f16: {
        ಸ್ಥಿರಾಂಕ ov::float16 * ದತ್ತಾಂಶ = tensor.ದತ್ತಾಂಶ<ov::float16>();
        size_t size = tensor.get_ಗಾತ್ರ();
        print_float_stats("[f16]", size, [ದತ್ತಾಂಶ](size_t i) { ಹಿಂತಿರುಗಿ static_cast<ದಶಮಾಂಶ>(ದತ್ತಾಂಶ[i]); });
        ನಿಲ್ಲಿಸು;
    }
    ಪೂರ್ವನಿಯೋಜಿತ:
        ನಿಲ್ಲಿಸು;
    }
}

ಸ್ಥಿರಾಂಕ ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; ++i) {
        ಸ್ವಯಂಚಾಲಿತ * op = cgraph->nodes[i];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; ++j) {
            ಸ್ವಯಂಚಾಲಿತ * src = op->src[j];
            ಆದರೆ (src == ಶೂನ್ಯಸೂಚಕ) {
                ನಿಲ್ಲಿಸು;
            }
            ಆದರೆ (GgmlOvDecoder::is_inp_pos(src, op)) {
                ಹಿಂತಿರುಗಿ src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    ಎಸೆ std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_ಟೋಕನುಗಳು(ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    ಪೂರ್ಣಾಂಕ n_planes = 1;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; ++i) {
        ಸ್ವಯಂಚಾಲಿತ * op = cgraph->nodes[i];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; ++j) {
            ಆದರೆ (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                ನಿಲ್ಲಿಸು;
            }
        }
    }
    ಹಿಂತಿರುಗಿ inp_pos->ne[0] / n_planes;
}

ತಾರ್ಕಿಕ get_is_prefill(ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ggml_tensor * inp_pos) {
    ಹಿಂತಿರುಗಿ get_inp_pos_n_ಟೋಕನುಗಳು(cgraph, inp_pos) > 1;
}
