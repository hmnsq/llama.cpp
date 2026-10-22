#ਅਨ੍ਤਰ੍ਭਾਵ "utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino-extra.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino/ggml-decoder.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "model-cache.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/frontend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/input_model.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <functional>
#ਅਨ੍ਤਰ੍ਭਾਵ <iomanip>
#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/any.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/graph_util.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/shape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type/float16.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/manager.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/openvino.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/compiled_model.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/infer_request.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/intel_npu/properties.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/properties.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/tensor.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <optional>
#ਅਨ੍ਤਰ੍ਭਾਵ <set>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ {
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
std::optional<ov::Tensor> try_make_kv_sliced_tensor(ਅਚਲ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    ਅਚਲ std::string & name,
                                                    ਅਚਲ ggml_tensor * ggml_tensor) {
    ਸਥਿਰ ਅਚਲ ਤਾਰਕਿਕ kv_ਟੁਕੜਾ_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    ਜੇ (kv_ਟੁਕੜਾ_disabled) {
        ਵਾਪਸ std::nullopt;
    }
    ਜੇ (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        ਵਾਪਸ std::nullopt;
    }
    ਜੇ (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ std::nullopt;
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ * op = ggml_decoder->get_tensor_ਵਰਤਿਆ_op(ggml_tensor);
    ਜੇ (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        ਵਾਪਸ std::nullopt;
    }

    ਅਚਲ ਸਵੈਚਾਲਿਤ & compute_params = ggml_decoder->get_compute_params();
    ਜੇ (compute_params.n_seq_ਸਰਗਰਮ != 1 || compute_params.seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ != 0) {
        ਵਾਪਸ std::nullopt;
    }

    ਪੂਰਨ layer;
    ਜੇ (ਸਵੈਚਾਲਿਤ layer_opt = ਕੱਢੋ_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ std::nullopt;
    }

    ਅਚਲ ਤਾਰਕਿਕ is_swa = ggml_decoder->is_swa_layer(layer);
    ਜੇ (is_swa) {
        ਵਾਪਸ std::nullopt;
    }
    ਅਚਲ ਪੂਰਨ ctx_ਪ੍ਰਤੀ_seq = ggml_decoder->get_ctx_ਪ੍ਰਤੀ_seq();
    ਅਚਲ ਪੂਰਨ n_kv = compute_params.attention_ਮਾਪ;
    ਜੇ (ctx_ਪ੍ਰਤੀ_seq <= 0 || n_kv <= 0 || n_kv >= ctx_ਪ੍ਰਤੀ_seq) {
        ਵਾਪਸ std::nullopt;
    }

    ov::Shape ਭਰਿਆ_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    ਜੇ (ਭਰਿਆ_shape.size() != 4 || ਭਰਿਆ_shape[0] != 1 || ਭਰਿਆ_shape[1] != 1 ||
        static_cast<ਪੂਰਨ>(ਭਰਿਆ_shape[2]) != ctx_ਪ੍ਰਤੀ_seq) {
        ਵਾਪਸ std::nullopt;
    }

    ov::Shape sliced_shape = ਭਰਿਆ_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    ਜੇ (ggml_openvino_ਬਫਰ_is_remote(ggml_tensor)) {
        ਸਵੈਚਾਲਿਤ remote_context = ggml_openvino_get_remote_context();
        ਸਵੈਚਾਲਿਤ gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        ਵਾਪਸ gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->ਅੰਕੜੇ);
    }

    ਵਾਪਸ ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->ਅੰਕੜੇ);
}

ਸਥਿਰ uint64_t ggml_openvino_model_cache_extra_cfg(ਤਾਰਕਿਕ stateful, ਤਾਰਕਿਕ recurrent) {
    ਅਚਲ ਅੱਖਰ * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    ਅਚਲ ਤਾਰਕਿਕ manual_gqa_enabled = manual_gqa_env != ਖਾਲੀ_ਸੂਚਕ ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    ਵਾਪਸ extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_ਭਾਰ_ਨਾਂ(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> ਨਾਂ;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & name : GgmlOvDecoder::collect_ਭਾਰ_ਨਾਂ(cgraph)) {
        ਨਾਂ[name] = ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ ਨਾਂ;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
ਸਥਿਰ std::string compiled_ਗ੍ਰਾਫ_ਕੁੰਜੀ(ਅਚਲ ggml_cgraph * graph,
                                      ਅਚਲ GgmlOvDecoder & decoder,
                                      ਅਚਲ std::string & device,
                                      ਪੂਰਨ prefill_chunk_ਮਾਪ = 0,
                                      ਤਾਰਕਿਕ disk_cache = ਝੂਠ) {
    std::string ਕੁੰਜੀ;
    ਸਵੈਚਾਲਿਤ append = [&ਕੁੰਜੀ](ਅਚਲ ਸਵੈਚਾਲਿਤ & value) {
        ਕੁੰਜੀ.append(reinterpret_cast<ਅਚਲ ਅੱਖਰ *>(&value), ਆਕਾਰ(value));
    };
    ਸਵੈਚਾਲਿਤ append_string = [&](ਅਚਲ std::string & value) {
        append(value.size());
        ਕੁੰਜੀ.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_ਮਾਪ);
    ਤਾਰਕਿਕ has_ਭਾਰ_ਬਫਰ_id = ਝੂਠ;
    std::unordered_map<ਅਚਲ ggml_tensor *, size_t> ids;
    std::function<ਖਾਲੀ(ਅਚਲ ggml_tensor *)> ਦੇਖੋ = [&](ਅਚਲ ggml_tensor * tensor) {
        ਜੇ (!tensor) {
            append(size_t(0));
            ਵਾਪਸ;
        }
        ਸਵੈਚਾਲਿਤ inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.ਪਹਿਲਾ->second);
        ਜੇ (!inserted.second) {
            ਵਾਪਸ;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        ਅਚਲ ਸਵੈਚਾਲਿਤ * ਅਧਾਰ = tensor->view_src ? tensor->view_src : tensor;
        ਅਚਲ ਤਾਰਕਿਕ ਭਾਰ = ਅਧਾਰ->ਬਫਰ && ਅਧਾਰ->ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS;
        append(ਭਾਰ);
        ਜੇ (ਭਾਰ && !disk_cache) {
            ਅਚਲ size_t ਬਫਰ_id = ggml_backend_openvino_ਬਫਰ_get_ctx_id(ਅਧਾਰ->ਬਫਰ);
            has_ਭਾਰ_ਬਫਰ_id |= ਬਫਰ_id != 0;
            append(ਬਫਰ_id);
            append(tensor->ਅੰਕੜੇ);
        }
        ਦੇਖੋ(tensor->view_src);
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * src : tensor->src) {
            ਦੇਖੋ(src);
        }
    };
    append(graph->n_nodes);
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; ++i) {
        ਦੇਖੋ(graph->nodes[i]);
    }
    append(graph->n_leafs);
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_leafs; ++i) {
        ਦੇਖੋ(graph->leafs[i]);
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਨਿਵੇਸ਼ : decoder.get_model_extra_inputs()) {
        append_string(ਨਿਵੇਸ਼.ਪਹਿਲਾ);
        append_string(ਨਿਵੇਸ਼.second.type.get_type_name());
        append(ਨਿਵੇਸ਼.second.shape.size());
        ਲਈ (ਸਵੈਚਾਲਿਤ dim : ਨਿਵੇਸ਼.second.shape) {
            append(dim);
        }
        append(ਨਿਵੇਸ਼.second.is_parameter);
        ਜੇ (!ਨਿਵੇਸ਼.second.is_parameter) {
            append(ਨਿਵੇਸ਼.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    ਵਾਪਸ disk_cache || has_ਭਾਰ_ਬਫਰ_id ? ਕੁੰਜੀ : std::string{};
}

ਸਥਿਰ std::string dynamic_ਗ੍ਰਾਫ_signature(ਅਚਲ ggml_cgraph * graph,
                                           ਅਚਲ GgmlOvDecoder & decoder,
                                           ਅਚਲ ModelParams & params) {
    std::string ਕੁੰਜੀ = "dynamic-1";
    ਸਵੈਚਾਲਿਤ append = [&ਕੁੰਜੀ](ਅਚਲ ਸਵੈਚਾਲਿਤ & value) {
        ਕੁੰਜੀ.append(reinterpret_cast<ਅਚਲ ਅੱਖਰ *>(&value), ਆਕਾਰ(value));
    };
    ਸਵੈਚਾਲਿਤ append_string = [&](ਅਚਲ std::string & value) {
        append(value.size());
        ਕੁੰਜੀ.append(value);
    };
    ਗ੍ਰਾਫ_ਕੁੰਜੀ ਗ੍ਰਾਫ_id(graph, ਸੱਚ);
    append(ਗ੍ਰਾਫ_id.n_nodes);
    append(ਗ੍ਰਾਫ_id.n_leaves);
    append_string(ਗ੍ਰਾਫ_id.ਪਹਿਲਾ_ਗੰਢ_name);
    append_string(ਗ੍ਰਾਫ_id.ਆਖਰੀ_ਗੰਢ_name);
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & name : ਗ੍ਰਾਫ_id.ਨਿਵੇਸ਼_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    ਲਈ (ਪੂਰਨ layer : params.swa_layers) {
        append(layer);
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & [layer, heads] : params.n_heads_kv_ਪ੍ਰਤੀ_layer) {
        append(layer);
        append(heads);
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & [name, ਨਿਵੇਸ਼] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(ਨਿਵੇਸ਼.type.get_type_name());
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & [name, ਨਿਵੇਸ਼] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(ਨਿਵੇਸ਼.type.get_type_name());
        append(ਨਿਵੇਸ਼.is_parameter);
        ਜੇ (!ਨਿਵੇਸ਼.is_parameter) {
            append(ਨਿਵੇਸ਼.value);
        }
    }
    ਵਾਪਸ ਕੁੰਜੀ;
}

ਸਥਿਰ ਤਾਰਕਿਕ compiled_model_matches_ਗ੍ਰਾਫ(ਅਚਲ ov::CompiledModel & model,
                                         ਅਚਲ std::vector<std::string> & inputs,
                                         ਅਚਲ std::vector<std::string> & outputs,
                                         ਅਚਲ GgmlOvDecoder & decoder) {
    ਜੇ (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ & ਗ੍ਰਾਫ_inputs = decoder.get_model_inputs();
    ਅਚਲ ਸਵੈਚਾਲਿਤ & extra_inputs = decoder.get_model_extra_inputs();
    ਅਚਲ ਸਵੈਚਾਲਿਤ & ਗ੍ਰਾਫ_outputs = decoder.get_model_outputs();
    ਲਈ (size_t i = 0; i < inputs.size(); ++i) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ & ਪੋਰਟ = model.ਨਿਵੇਸ਼(i);
        ਸਵੈਚਾਲਿਤ ਗ੍ਰਾਫ_it = ਗ੍ਰਾਫ_inputs.find(inputs[i]);
        ਜੇ (ਗ੍ਰਾਫ_it != ਗ੍ਰਾਫ_inputs.end()) {
            ਜੇ (ਪੋਰਟ.get_element_type() != ਗ੍ਰਾਫ_it->second.type ||
                !ਪੋਰਟ.get_partial_shape().compatible(ਗ੍ਰਾਫ_it->second.shape)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜਾਰੀ;
        }
        ਸਵੈਚਾਲਿਤ extra_it = extra_inputs.find(inputs[i]);
        ਜੇ (extra_it == extra_inputs.end() || ਪੋਰਟ.get_element_type() != extra_it->second.type ||
            !ਪੋਰਟ.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਲਈ (size_t i = 0; i < outputs.size(); ++i) {
        ਸਵੈਚਾਲਿਤ ਗ੍ਰਾਫ_it = ਗ੍ਰਾਫ_outputs.find(outputs[i]);
        ਜੇ (ਗ੍ਰਾਫ_it == ਗ੍ਰਾਫ_outputs.end()) {
            ਜੇ (outputs[i].compare(0, 8, "__debug_") != 0) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜਾਰੀ;
        }
        ਅਚਲ ਸਵੈਚਾਲਿਤ & ਪੋਰਟ = model.output(i);
        ਜੇ (ਪੋਰਟ.get_element_type() != GgmlOvDecoder::get_ov_type(ਗ੍ਰਾਫ_it->second) ||
            !ਪੋਰਟ.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(ਗ੍ਰਾਫ_it->second)))) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ std::string ov_profiling_csv_ਖੇਤਰ(ਅਚਲ std::string & value) {
    std::string escaped = "\"";
    ਲਈ (ਅੱਖਰ c : value) {
        escaped += c;
        ਜੇ (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    ਵਾਪਸ escaped;
}

ਸਥਿਰ ਖਾਲੀ dump_ov_profiling_info(ਅਚਲ ov::InferRequest & infer_request) {
    ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        ਵਾਪਸ;
    }

    ਸਥਿਰ std::atomic<uint64_t> inference_ਸੂਚਕਾਂਕ{0};
    ਅਚਲ uint64_t ਸੂਚਕਾਂਕ = inference_ਸੂਚਕਾਂਕ.fetch_add(1);
    ਅਚਲ std::string path = "openvino_profile_" + std::to_string(ਸੂਚਕਾਂਕ) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    ਜੇ (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        ਵਾਪਸ;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t ਕੁੱਲ_real_ਸਮਾਂ = 0;
    int64_t ਕੁੱਲ_cpu_ਸਮਾਂ = 0;
    size_t executed_ਗਿਣਤੀ = 0;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & info : infer_request.get_profiling_info()) {
        ਅਚਲ ਅੱਖਰ * ਸਥਿਤੀ = "NOT_RUN";
        ਜੇ (info.ਸਥਿਤੀ == ov::ProfilingInfo::ਸਥਿਤੀ_2::EXECUTED) {
            ਸਥਿਤੀ = "EXECUTED";
            ਕੁੱਲ_real_ਸਮਾਂ += info.real_ਸਮਾਂ.count();
            ਕੁੱਲ_cpu_ਸਮਾਂ += info.cpu_ਸਮਾਂ.count();
            executed_ਗਿਣਤੀ++;
        } ਨਹੀਂਤਾਂ ਜੇ (info.ਸਥਿਤੀ == ov::ProfilingInfo::ਸਥਿਤੀ_2::OPTIMIZED_ਬਾਹਰਲਾ) {
            ਸਥਿਤੀ = "OPTIMIZED_OUT";
        }

        output << ਸਥਿਤੀ << ',' << info.real_ਸਮਾਂ.count() << ',' << info.cpu_ਸਮਾਂ.count() << ','
               << info.ਸ਼ੁਰੂ_ਸਮਾਂ.count() << ',' << ov_profiling_csv_ਖੇਤਰ(info.node_type) << ','
               << ov_profiling_csv_ਖੇਤਰ(info.ਗੰਢ_name) << ',' << ov_profiling_csv_ਖੇਤਰ(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_ਗਿਣਤੀ, ਕੁੱਲ_real_ਸਮਾਂ / 1000.0, ਕੁੱਲ_cpu_ਸਮਾਂ / 1000.0);
}

ਸਥਿਰ ਖਾਲੀ reset_single_ਥਾਂ_recurrent_cache(ggml_cgraph * cgraph,
                                              ਅਚਲ ModelParams & model_params,
                                              ਅਚਲ ComputeParams & compute_params) {
    ਜੇ (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        ਵਾਪਸ;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_ਬਫਰ_t> buffers;
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਸਵੈਚਾਲਿਤ * ਗੰਢ = cgraph->nodes[i];
        ਜੇ (ਗੰਢ->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(ਗੰਢ->view_src) &&
            ਗੰਢ->view_src->ne[1] == 1 && ਗੰਢ->view_src->ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ) {
            buffers.insert(ਗੰਢ->view_src->ਬਫਰ);
        }
    }
    ਲਈ (ਸਵੈਚਾਲਿਤ * ਬਫਰ : buffers) {
        ggml_backend_ਬਫਰ_clear(ਬਫਰ, 0);
    }
}

ov::Tensor create_ov_output_tensor(ਅਚਲ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   ਅਚਲ std::shared_ptr<ov::InferRequest> & infer_request,
                                   ਪੂਰਨ output_ਸੂਚਕਾਂਕ,
                                   ਅਚਲ ggml_tensor * ggml_tensor) {
    ਜੇ (ਸਵੈਚਾਲਿਤ sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        ਵਾਪਸ *sliced;
    }

    ਸਵੈਚਾਲਿਤ output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    ਖਾਲੀ * output_ਅੰਕੜੇ = ggml_tensor->ਅੰਕੜੇ;
    ਜੇ (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_ਸੂਚਕਾਂਕ).get_shape();
    } ਨਹੀਂਤਾਂ {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        ਜੇ (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != ਖਾਲੀ_ਸੂਚਕ &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_ਅੰਕੜੇ = ggml_tensor->view_src->ਅੰਕੜੇ;
        } ਨਹੀਂਤਾਂ {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    ਜੇ (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_ਬਫਰ_is_remote(ggml_tensor) &&
        ggml_tensor->extra != ਖਾਲੀ_ਸੂਚਕ) {
        ਸਵੈਚਾਲਿਤ * extra_ਅਧਾਰ = static_cast<ggml_openvino_extra_ਅਧਾਰ *>(ggml_tensor->extra);
        ਜੇ (extra_ਅਧਾਰ->type == ggml_openvino_extra_ਅਧਾਰ::Type::TENSOR) {
            ਸਵੈਚਾਲਿਤ * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_ਅਧਾਰ);
            ਜੇ (tensor_extra->tensor != ਖਾਲੀ_ਸੂਚਕ && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                ਵਾਪਸ *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_ਅੰਕੜੇ);
    ਵਾਪਸ output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_seq_axis_2(ਅਚਲ ov::Tensor & kv_tensor, size_t n_heads_kv) {
    ਅਚਲ size_t ਪੰਕਤੀਆਂ = kv_tensor.get_shape()[2];
    ਅਚਲ size_t ਸਿਰ_ਮਾਪ = kv_tensor.get_shape()[3] / n_heads_kv;
    ਅਚਲ size_t elem = kv_tensor.get_element_type().size();
    ਅਚਲ size_t ਸਿਰ_ਬਾਈਟਾਂ = ਸਿਰ_ਮਾਪ * elem;

    ov::Tensor ਬਾਹਰਲਾ(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, ਪੰਕਤੀਆਂ, ਸਿਰ_ਮਾਪ});
    ਅਚਲ ਸਵੈਚਾਲਿਤ * src = static_cast<ਅਚਲ uint8_t *>(kv_tensor.ਅੰਕੜੇ());
    ਸਵੈਚਾਲਿਤ * dst = static_cast<uint8_t *>(ਬਾਹਰਲਾ.ਅੰਕੜੇ());
    ਲਈ (size_t s = 0; s < ਪੰਕਤੀਆਂ; s++) {
        ਲਈ (size_t h = 0; h < n_heads_kv; h++) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst + (h * ਪੰਕਤੀਆਂ + s) * ਸਿਰ_ਬਾਈਟਾਂ, src + (s * n_heads_kv + h) * ਸਿਰ_ਬਾਈਟਾਂ, ਸਿਰ_ਬਾਈਟਾਂ);
        }
    }
    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> ਖਾਲੀ ਸੈੱਟ_zero_diagonal(std::vector<T> & ਮੈਟ੍ਰਿਕਸ, size_t ਪੰਕਤੀਆਂ, size_t ਖਾਨੇ_ਗਿਣਤੀ_2, T zero_ਮੁੱਲ = T{}) {
    ਲਈ (size_t i = 0; i < ਪੰਕਤੀਆਂ; ++i) {
        size_t diag_ਖਾਨਾ = std::min(i, ਖਾਨੇ_ਗਿਣਤੀ_2 - 1);
        ਮੈਟ੍ਰਿਕਸ[i * ਖਾਨੇ_ਗਿਣਤੀ_2 + diag_ਖਾਨਾ] = zero_ਮੁੱਲ;
    }
}

ov::Tensor make_contiguous_ਵੰਡੋ_ਨਿਵੇਸ਼_tensor(ਅਚਲ ਬਣਤਰ ggml_tensor * ggml_tensor, ਅਚਲ ov::Shape & ਨਿਵੇਸ਼_shape) {
    ਅਚਲ size_t element_ਮਾਪ = ggml_type_ਮਾਪ(ggml_tensor->type);
    ਅਚਲ size_t ਖੰਡ_ਮਾਪ_3 = ggml_blck_ਮਾਪ(ggml_tensor->type);

    GGML_ASSERT(ਖੰਡ_ਮਾਪ_3 == 1 && "non-contiguous split inputs must be plain element types");

    ਅਚਲ ਬਣਤਰ ggml_tensor * ਸਰੋਤ_tensor = ggml_tensor->view_src != ਖਾਲੀ_ਸੂਚਕ ? ggml_tensor->view_src : ggml_tensor;
    ਅਚਲ size_t ਸਰੋਤ_ਵਿੱਥ = ggml_tensor->view_src != ਖਾਲੀ_ਸੂਚਕ ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> ਸਰੋਤ_ਅੰਕੜੇ(ggml_nbytes(ਸਰੋਤ_tensor));
    ggml_backend_tensor_get(ਸਰੋਤ_tensor, ਸਰੋਤ_ਅੰਕੜੇ.ਅੰਕੜੇ(), 0, ਸਰੋਤ_ਅੰਕੜੇ.size());

    ov::Tensor ਨਿਵੇਸ਼_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ਨਿਵੇਸ਼_shape);
    ਸਵੈਚਾਲਿਤ * dst = static_cast<uint8_t *>(ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ());
    size_t dst_ਵਿੱਥ = 0;

    ਲਈ (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        ਲਈ (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            ਲਈ (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                ਲਈ (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    ਅਚਲ size_t src_ਵਿੱਥ = ਸਰੋਤ_ਵਿੱਥ + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst + dst_ਵਿੱਥ, ਸਰੋਤ_ਅੰਕੜੇ.ਅੰਕੜੇ() + src_ਵਿੱਥ, element_ਮਾਪ);
                    dst_ਵਿੱਥ += element_ਮਾਪ;
                }
            }
        }
    }

    ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
}

ov::Tensor convert_ggml_ਨਿਵੇਸ਼_ਮੰਜ਼ਿਲ_ov(ਅਚਲ std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ਅਚਲ std::string & name) {
    ਅਚਲ ਸਵੈਚਾਲਿਤ * ggml_tensor = ggml_decoder->get_ਨਿਵੇਸ਼_ggml_tensor(name);

    ਜੇ (ਸਵੈਚਾਲਿਤ sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        ਵਾਪਸ *sliced;
    }

    ਜੇ (ggml_tensor->extra != ਖਾਲੀ_ਸੂਚਕ && !ggml_decoder->is_splited_model()) {
        ਸਵੈਚਾਲਿਤ * extra_ਅਧਾਰ = static_cast<ggml_openvino_extra_ਅਧਾਰ *>(ggml_tensor->extra);
        ਜੇ (extra_ਅਧਾਰ->type == ggml_openvino_extra_ਅਧਾਰ::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            ਸਵੈਚਾਲਿਤ * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_ਅਧਾਰ);
            ਵਾਪਸ *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    ਸਵੈਚਾਲਿਤ * ਨਿਵੇਸ਼_ਅੰਕੜੇ = ggml_tensor->ਅੰਕੜੇ;
    ov::Shape ਨਿਵੇਸ਼_shape;
    ਜੇ (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        ਨਿਵੇਸ਼_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } ਨਹੀਂਤਾਂ {
        ਨਿਵੇਸ਼_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    ਜੇ (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        ਵਾਪਸ make_contiguous_ਵੰਡੋ_ਨਿਵੇਸ਼_tensor(ggml_tensor, ਨਿਵੇਸ਼_shape);
    }

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ਨਿਵੇਸ਼_shape, ਨਿਵੇਸ਼_ਅੰਕੜੇ);
    ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
}

ov::Tensor get_ov_ਨਿਵੇਸ਼_tensor(ਅਚਲ std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ਅਚਲ std::string & param_name) {
    ov::Tensor ਨਿਵੇਸ਼_tensor;
    ਸਵੈਚਾਲਿਤ extra_ਨਿਵੇਸ਼ = ggml_decoder->get_model_extra_inputs().find(param_name);
    ਜੇ (extra_ਨਿਵੇਸ਼ != ggml_decoder->get_model_extra_inputs().end()) {
        ਨਿਵੇਸ਼_tensor = ov::Tensor(extra_ਨਿਵੇਸ਼->second.type, extra_ਨਿਵੇਸ਼->second.shape);
        *ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<int64_t>() = extra_ਨਿਵੇਸ਼->second.value;
    } ਨਹੀਂਤਾਂ {
        ਨਿਵੇਸ਼_tensor = convert_ggml_ਨਿਵੇਸ਼_ਮੰਜ਼ਿਲ_ov(ggml_decoder, param_name);
    }
    ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
}

ov::Tensor get_ov_ਨਿਵੇਸ਼_tensor_static_decode(ਅਚਲ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             ਅਚਲ std::string & param_name) {
    // NPU decoding stage
    ਜੇ (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        ਵਾਪਸ get_ov_ਨਿਵੇਸ਼_tensor(ggml_decoder, param_name);
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ * ggml_tensor = ggml_decoder->get_ਨਿਵੇਸ਼_ggml_tensor(param_name);
    ਅਚਲ ਸਵੈਚਾਲਿਤ * op = ggml_decoder->get_tensor_ਵਰਤਿਆ_op(ggml_tensor);

    ਜੇ (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        ਅਚਲ ਪੂਰਨ n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        ਪ੍ਰਤਿਜ੍ਞਾ(ggml_tensor->ne[0] == n_planes);
        ov::Shape ਨਿਵੇਸ਼_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor ਨਿਵੇਸ਼_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ਨਿਵੇਸ਼_shape);
        std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ(), ggml_tensor->ਅੰਕੜੇ, n_planes * ggml_type_ਮਾਪ(ggml_tensor->type));
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }

    ਜੇ (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape ਨਿਵੇਸ਼_shape = {1, 1, 1, 1};
        ov::Tensor ਨਿਵੇਸ਼_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ਨਿਵੇਸ਼_shape);
        int32_t inp_ਬਾਹਰਲਾ_id = *((int32_t *) ggml_tensor->ਅੰਕੜੇ);
        ਪ੍ਰਤਿਜ੍ਞਾ(ggml_tensor->ne[0] == 1);
        ਪ੍ਰਤਿਜ੍ਞਾ(inp_ਬਾਹਰਲਾ_id == 0);
        *ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<int32_t>() = inp_ਬਾਹਰਲਾ_id;
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }

    ਜੇ (GgmlOvDecoder::is_inp_ਨਕਾਬ(ggml_tensor, op)) {
        size_t context_ਮਾਪ = ggml_decoder->get_ctx_ਮਾਪ();
        ਜੇ (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_ਅੰਕੜੇ =
                pad_ਨਿਵੇਸ਼<ggml_fp16_t>(ggml_tensor, 1, context_ਮਾਪ, GGML_FP32_ਮੰਜ਼ਿਲ_FP16(-INFINITY));
            ov::Tensor ਨਿਵੇਸ਼_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_ਮਾਪ});
            std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ(), padded_ਅੰਕੜੇ.ਅੰਕੜੇ(), padded_ਅੰਕੜੇ.size() * ਆਕਾਰ(ggml_fp16_t));
            ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
        }

        std::vector<ਦਸ਼ਮਲਵ> padded_ਅੰਕੜੇ = pad_ਨਿਵੇਸ਼<ਦਸ਼ਮਲਵ>(ggml_tensor, 1, context_ਮਾਪ, -INFINITY);
        ov::Tensor ਨਿਵੇਸ਼_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_ਮਾਪ});
        ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ_ptr = ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<ਦਸ਼ਮਲਵ>();
        std::copy(padded_ਅੰਕੜੇ.begin(), padded_ਅੰਕੜੇ.begin() + context_ਮਾਪ, ਅੰਕੜੇ_ptr);
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }

    ਵਾਪਸ get_ov_ਨਿਵੇਸ਼_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_ਨਿਵੇਸ਼_tensor_static_prefill(ਅਚਲ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              ਅਚਲ std::string & param_name,
                                              ਪੂਰਨ chunk_ਸੂਚਕਾਂਕ) {
    // NPU prompt processing stage
    ਅਚਲ size_t ਨਿਵੇਸ਼_len = ggml_decoder->get_ਨਿਵੇਸ਼_len();
    ਅਚਲ size_t chunk_ਮਾਪ = ggml_decoder->m_prefill_chunk_ਮਾਪ;
    ਅਚਲ size_t chunk_valid_ਮਾਪ = std::min(chunk_ਮਾਪ, ਨਿਵੇਸ਼_len - chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ);
    ਅਚਲ size_t chunk_pad_ਮਾਪ = chunk_ਮਾਪ - chunk_valid_ਮਾਪ;

    ਜੇ (param_name == "chunk_valid_len") {
        ov::Tensor ਨਿਵੇਸ਼_tensor(ov::element::i64, ov::Shape{1});
        *ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<int64_t>() = (int64_t) chunk_valid_ਮਾਪ;
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }
    ਜੇ (chunk_ਸੂਚਕਾਂਕ > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor ਨਿਵੇਸ਼_tensor(ov::element::i64, ov::Shape{1});
        *ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<int64_t>() = 0;
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }
    ਜੇ (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        ਵਾਪਸ get_ov_ਨਿਵੇਸ਼_tensor(ggml_decoder, param_name);
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ * ggml_tensor = ggml_decoder->get_ਨਿਵੇਸ਼_ggml_tensor(param_name);
    ਅਚਲ ਸਵੈਚਾਲਿਤ * op = ggml_decoder->get_tensor_ਵਰਤਿਆ_op(ggml_tensor);

    ਜੇ (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        ਅਚਲ ਪੂਰਨ n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        ਅਚਲ size_t element_ਮਾਪ = ggml_type_ਮਾਪ(ggml_tensor->type);
        ov::Shape ਨਿਵੇਸ਼_shape = {1, 1, 1, (size_t) n_planes * chunk_ਮਾਪ};
        ov::Tensor ਨਿਵੇਸ਼_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ਨਿਵੇਸ਼_shape);
        ਲਈ (ਪੂਰਨ p = 0; p < n_planes; p++) {
            ਅਚਲ ਅੱਖਰ * src =
                (ਅਚਲ ਅੱਖਰ *) ggml_tensor->ਅੰਕੜੇ + (p * ਨਿਵੇਸ਼_len + chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ) * element_ਮਾਪ;
            ਅੱਖਰ * dst = (ਅੱਖਰ *) ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ() + p * chunk_ਮਾਪ * element_ਮਾਪ;
            std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst, src, chunk_valid_ਮਾਪ * element_ਮਾਪ);
            ਜੇ (chunk_pad_ਮਾਪ > 0) {
                ਜੇ (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t ਆਖਰੀ_ਮੁੱਲ = *((ਅਚਲ int32_t *) src + chunk_valid_ਮਾਪ - 1);
                    int32_t * ਬਾਹਰਲਾ = (int32_t *) dst;
                    std::fill(ਬਾਹਰਲਾ + chunk_valid_ਮਾਪ, ਬਾਹਰਲਾ + chunk_ਮਾਪ, ਆਖਰੀ_ਮੁੱਲ + 1);
                } ਨਹੀਂਤਾਂ ਜੇ (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t ਆਖਰੀ_ਮੁੱਲ = *((ਅਚਲ int64_t *) src + chunk_valid_ਮਾਪ - 1);
                    int64_t * ਬਾਹਰਲਾ = (int64_t *) dst;
                    std::fill(ਬਾਹਰਲਾ + chunk_valid_ਮਾਪ, ਬਾਹਰਲਾ + chunk_ਮਾਪ, ਆਖਰੀ_ਮੁੱਲ + 1);
                } ਨਹੀਂਤਾਂ {
                    ਸੁੱਟੋ std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }

    ਜੇ (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape ਨਿਵੇਸ਼_shape = {1, 1, 1, chunk_ਮਾਪ};
        ov::Tensor ਨਿਵੇਸ਼_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ਨਿਵੇਸ਼_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_ਮਾਪ = ggml_type_ਮਾਪ(ggml_tensor->type);
        ਖਾਲੀ * ਨਿਵੇਸ਼_ਅੰਕੜੇ = (ਅੱਖਰ *) ggml_tensor->ਅੰਕੜੇ + chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ * element_ਮਾਪ;
        std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ(), ਨਿਵੇਸ਼_ਅੰਕੜੇ, chunk_valid_ਮਾਪ * element_ਮਾਪ);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        ਜੇ (chunk_pad_ਮਾਪ > 0) {
            ਜੇ (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t ਆਖਰੀ_ਮੁੱਲ =
                    *((int32_t *) ggml_tensor->ਅੰਕੜੇ + (chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ + chunk_valid_ਮਾਪ - 1));
                int32_t * output_ਅੰਕੜੇ = ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<int32_t>();
                std::fill(output_ਅੰਕੜੇ + chunk_valid_ਮਾਪ, output_ਅੰਕੜੇ + chunk_ਮਾਪ, ਆਖਰੀ_ਮੁੱਲ + 1);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t ਆਖਰੀ_ਮੁੱਲ =
                    *((int64_t *) ggml_tensor->ਅੰਕੜੇ + (chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ + chunk_valid_ਮਾਪ - 1));
                int64_t * output_ਅੰਕੜੇ = ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<int64_t>();
                std::fill(output_ਅੰਕੜੇ + chunk_valid_ਮਾਪ, output_ਅੰਕੜੇ + chunk_ਮਾਪ, ਆਖਰੀ_ਮੁੱਲ + 1);
            } ਨਹੀਂਤਾਂ {
                ਸੁੱਟੋ std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }

    ਜੇ (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape ਨਿਵੇਸ਼_shape = {1, 1, 1, output_len};
        ov::Tensor ਨਿਵੇਸ਼_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ਨਿਵੇਸ਼_shape);
        ਜੇ (ggml_tensor->ne[0] == 0) {
            *ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<int32_t>() = 0;
        } ਨਹੀਂਤਾਂ {
            ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ_addr = ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<int32_t>();
            ਲਈ (size_t i = 0; i < output_len; i++) {
                ਅੰਕੜੇ_addr[i] = ((int32_t *) ggml_tensor->ਅੰਕੜੇ)[i] % chunk_ਮਾਪ;
            }
        }
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }

    ਜੇ (GgmlOvDecoder::is_inp_scale_ਪੰਕਤੀਆਂ(ggml_tensor, op)) {
        ov::Tensor ਨਿਵੇਸ਼_tensor(ov::element::f32, ov::Shape{1, 1, chunk_ਮਾਪ, 1});
        ਸਵੈਚਾਲਿਤ * dst = ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<ਦਸ਼ਮਲਵ>();
        ਅਚਲ ਸਵੈਚਾਲਿਤ * src = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(ggml_tensor->ਅੰਕੜੇ) + chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ;
        std::copy(src, src + chunk_valid_ਮਾਪ, dst);
        std::fill(dst + chunk_valid_ਮਾਪ, dst + chunk_ਮਾਪ, 1.0f);
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }

    ਜੇ (GgmlOvDecoder::is_inp_ਔਸਤ(ggml_tensor, op)) {
        ਅਚਲ size_t n_seqs = ggml_tensor->ne[1];
        ਅਚਲ size_t src_stride = ggml_tensor->ne[0];
        ਅਚਲ size_t copy_len = std::min<size_t>(chunk_valid_ਮਾਪ, src_stride - chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ);
        ov::Tensor ਨਿਵੇਸ਼_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_ਮਾਪ});
        ਸਵੈਚਾਲਿਤ * dst = ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<ਦਸ਼ਮਲਵ>();
        std::fill(dst, dst + n_seqs * chunk_ਮਾਪ, 0.0f);
        ਅਚਲ ਸਵੈਚਾਲਿਤ * src = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(ggml_tensor->ਅੰਕੜੇ) + chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ;
        ਲਈ (size_t s = 0; s < n_seqs; s++) {
            std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst + s * chunk_ਮਾਪ, src + s * src_stride, copy_len * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        }
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }

    ਜੇ (GgmlOvDecoder::is_inp_ਨਕਾਬ(ggml_tensor, op)) {
        size_t ਖਾਨੇ_ਗਿਣਤੀ_2 = ggml_tensor->ne[0];
        size_t ਪੰਕਤੀਆਂ = ggml_tensor->ne[1];
        size_t chunk_valid_ਪੰਕਤੀਆਂ = std::min(chunk_ਮਾਪ, ਪੰਕਤੀਆਂ - chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ);
        size_t context_ਮਾਪ = ggml_decoder->get_ctx_ਮਾਪ();
        ਜੇ (ggml_tensor->type == GGML_TYPE_F16) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * ggml_ਅੰਕੜੇ =
                static_cast<ਅਚਲ ggml_fp16_t *>(ggml_tensor->ਅੰਕੜੇ) + chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ * ਖਾਨੇ_ਗਿਣਤੀ_2;
            std::vector<ggml_fp16_t> padded_ਅੰਕੜੇ = pad_ਨਿਵੇਸ਼<ggml_fp16_t>(ggml_ਅੰਕੜੇ, chunk_valid_ਪੰਕਤੀਆਂ, ਖਾਨੇ_ਗਿਣਤੀ_2, chunk_ਮਾਪ,
                                                                          context_ਮਾਪ, GGML_FP32_ਮੰਜ਼ਿਲ_FP16(-INFINITY));
            ਸੈੱਟ_zero_diagonal(padded_ਅੰਕੜੇ, chunk_ਮਾਪ, context_ਮਾਪ, GGML_FP32_ਮੰਜ਼ਿਲ_FP16(0.0f));
            ov::Tensor ਨਿਵੇਸ਼_tensor(ov::element::f16, ov::Shape{1, 1, chunk_ਮਾਪ, context_ਮਾਪ});
            std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ(), padded_ਅੰਕੜੇ.ਅੰਕੜੇ(), padded_ਅੰਕੜੇ.size() * ਆਕਾਰ(ggml_fp16_t));
            ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ * ggml_ਅੰਕੜੇ = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(ggml_tensor->ਅੰਕੜੇ) + chunk_ਸੂਚਕਾਂਕ * chunk_ਮਾਪ * ਖਾਨੇ_ਗਿਣਤੀ_2;
        std::vector<ਦਸ਼ਮਲਵ> padded_ਅੰਕੜੇ =
            pad_ਨਿਵੇਸ਼<ਦਸ਼ਮਲਵ>(ggml_ਅੰਕੜੇ, chunk_valid_ਪੰਕਤੀਆਂ, ਖਾਨੇ_ਗਿਣਤੀ_2, chunk_ਮਾਪ, context_ਮਾਪ, -INFINITY);
        ਸੈੱਟ_zero_diagonal(padded_ਅੰਕੜੇ, chunk_ਮਾਪ, context_ਮਾਪ);
        ov::Tensor ਨਿਵੇਸ਼_tensor(ov::element::f32, ov::Shape{1, 1, chunk_ਮਾਪ, context_ਮਾਪ});
        ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ_ptr = ਨਿਵੇਸ਼_tensor.ਅੰਕੜੇ<ਦਸ਼ਮਲਵ>();
        std::copy(padded_ਅੰਕੜੇ.begin(), padded_ਅੰਕੜੇ.begin() + chunk_ਮਾਪ * context_ਮਾਪ, ਅੰਕੜੇ_ptr);
        ਵਾਪਸ ਨਿਵੇਸ਼_tensor;
    }

    ਵਾਪਸ get_ov_ਨਿਵੇਸ਼_tensor(ggml_decoder, param_name);
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               ਅਚਲ std::string & device,
                               ਅਚਲ ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    ਜੇ (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_ਤਾਲਾ(cache.mutex);
    ਤਾਰਕਿਕ naive = ਸੱਚ;
    ਸਵੈਚਾਲਿਤ model_weights = GgmlOvDecoder::create_ਭਾਰ_nodes(cgraph, naive);
    ਸਵੈਚਾਲਿਤ decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_model = std::make_shared<ov::frontend::ggml::ਨਿਵੇਸ਼_Model>(decoder);
    ਸਵੈਚਾਲਿਤ model = ov::frontend::ggml::Front_ਸਿਰਾ::convert(ਨਿਵੇਸ਼_model, naive);
    ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    ਸਵੈਚਾਲਿਤ remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    ਜੇ (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_ਬਹੁਲਕ.name()] = ov::hint::Execution_ਬਹੁਲਕ::PERFORMANCE;
    } ਨਹੀਂਤਾਂ {
        compile_config[ov::hint::execution_ਬਹੁਲਕ.name()] = ov::hint::Execution_ਬਹੁਲਕ::ACCURACY;
    }
    ਜੇ (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } ਨਹੀਂਤਾਂ {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> ਨਿਵੇਸ਼_ਨਾਂ;
    std::vector<std::string> output_ਨਾਂ;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & param : model->get_parameters()) {
        ਨਿਵੇਸ਼_ਨਾਂ.push_back(param->get_friendly_name());
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਨਤੀਜਾ : model->get_results()) {
        output_ਨਾਂ.push_back(ਨਤੀਜਾ->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    ਨਿਵੇਸ਼_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_ਤਾਲਾ.unlock();

    ਲਈ (size_t i = 0; i < ਨਿਵੇਸ਼_ਨਾਂ.size(); i++) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ & param_name = ਨਿਵੇਸ਼_ਨਾਂ[i];
        ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_tensor = get_ov_ਨਿਵੇਸ਼_tensor(decoder, param_name);
        infer_request->ਸੈੱਟ_ਨਿਵੇਸ਼_tensor(i, ਨਿਵੇਸ਼_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    ਲਈ (size_t i = 0; i < output_ਨਾਂ.size(); i++) {
        ਸਵੈਚਾਲਿਤ output_tensor = infer_request->get_output_tensor(i);
        ਅਚਲ ਸਵੈਚਾਲਿਤ & model_outputs = decoder->get_model_outputs();
        ਸਵੈਚਾਲਿਤ model_output_it = model_outputs.find(output_ਨਾਂ[i]);
        ਜੇ (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_ਨਾਂ[i], output_tensor, output_tensor.ਅੰਕੜੇ());
            }
            ਜਾਰੀ;
        }
        ਸਵੈਚਾਲਿਤ * ggml_tensor = model_output_it->second;
        std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(ggml_tensor->ਅੰਕੜੇ, output_tensor.ਅੰਕੜੇ(), output_tensor.get_byte_ਮਾਪ());
    }
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ov_ਗ੍ਰਾਫ_compute_dynamic(ggml_cgraph * cgraph, ਅਚਲ std::shared_ptr<ov_runtime_context> & r_ctx) {
    ਸਵੈਚਾਲਿਤ & core = ov_singleton_core();
    ਅਚਲ ਸਵੈਚਾਲਿਤ & config = ggml_openvino_get_compile_config();
    ਅਚਲ ਸਵੈਚਾਲਿਤ & device = r_ctx->device;
    ਸਥਿਰ ਸਵੈਚਾਲਿਤ is_static = ਝੂਠ;

    ਸਥਿਰ ਅਚਲ ਤਾਰਕਿਕ cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    ਸਵੈਚਾਲਿਤ ਸ਼ੁਰੂ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    ਅਚਲ int64_t cache_ਕੁੰਜੀ_ਸ਼ੁਰੂ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
    ਗ੍ਰਾਫ_ਕੁੰਜੀ ਕੁੰਜੀ(cgraph);
    ਅਚਲ int64_t cache_ਕੁੰਜੀ_compute_ਸਮਾਂ = ggml_ਸਮਾਂ_us() - cache_ਕੁੰਜੀ_ਸ਼ੁਰੂ_ਸਮਾਂ;
    int64_t cache_lookup_ਸਮਾਂ = 0;
    ਤਾਰਕਿਕ ਕੁੰਜੀ_ਵੇਖਿਆ = ਝੂਠ;
    ਜੇ (!cache_disabled) {
        ਅਚਲ int64_t cache_lookup_ਸ਼ੁਰੂ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
        std::lock_guard<std::mutex> map_ਤਾਲਾ(r_ctx->ctx_mutex);
        ਕੁੰਜੀ_ਵੇਖਿਆ = r_ctx->decoder_cache.find(ਕੁੰਜੀ) != r_ctx->decoder_cache.end();
        cache_lookup_ਸਮਾਂ += ggml_ਸਮਾਂ_us() - cache_lookup_ਸ਼ੁਰੂ_ਸਮਾਂ;
    }

    ਅਚਲ int64_t model_ਵੰਡੋ_check_ਸ਼ੁਰੂ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
    ਤਾਰਕਿਕ model_is_splitted = ਕੁੰਜੀ_ਵੇਖਿਆ ? ਝੂਠ : is_model_splitted(cgraph);
    ਅਚਲ int64_t model_ਵੰਡੋ_check_ਸਮਾਂ = ggml_ਸਮਾਂ_us() - model_ਵੰਡੋ_check_ਸ਼ੁਰੂ_ਸਮਾਂ;

    ਜੇ (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
    }
    ਜੇ (is_naive(cgraph)) {
        ਜੇ (!model_is_splitted) {
            ਵਾਪਸ naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    ਅਚਲ int64_t ਗ੍ਰਾਫ_param_compute_ਸ਼ੁਰੂ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    ਅਚਲ int64_t ਗ੍ਰਾਫ_param_compute_ਸਮਾਂ = ggml_ਸਮਾਂ_us() - ਗ੍ਰਾਫ_param_compute_ਸ਼ੁਰੂ_ਸਮਾਂ;

    ਅਚਲ ਤਾਰਕਿਕ cache_enabled = !model_is_splitted && !cache_disabled;
    ਅਚਲ ਤਾਰਕਿਕ stateful_recurrent = r_ctx->stateful && m_params.state_ਮਾਪ >= 0;
    ਅਚਲ ਤਾਰਕਿਕ stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    ਜੇ (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_ਸਰਗਰਮ > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
    }
    ਜੇ (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
    }
    ਤਾਰਕਿਕ cache_ਪ੍ਰਾਪਤੀ = ਝੂਠ;

    int64_t decoder_ਸਿਰਾ_ਸਮਾਂ;
    int64_t conversion_ਸਿਰਾ_ਸਮਾਂ;
    int64_t compile_ਸਿਰਾ_ਸਮਾਂ;
    int64_t infer_ਸਿਰਾ_ਸਮਾਂ;
    int64_t ov_raw_infer_ਸ਼ੁਰੂ;

    {
        std::shared_ptr<decoder_runtime_ctx> ਇੰਦਰਾਜ;
        ModelParams old_m_params;

        ਅਚਲ int64_t cache_lookup_ਸ਼ੁਰੂ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
        ਜੇ (cache_enabled) {
            std::lock_guard<std::mutex> map_ਤਾਲਾ(r_ctx->ctx_mutex);
            ਸਵੈਚਾਲਿਤ it = r_ctx->decoder_cache.find(ਕੁੰਜੀ);
            cache_ਪ੍ਰਾਪਤੀ = it != r_ctx->decoder_cache.end();
            ਜੇ (cache_ਪ੍ਰਾਪਤੀ) {
                ਇੰਦਰਾਜ = it->second;
            } ਨਹੀਂਤਾਂ {
                r_ctx->clear_caches_locked();
                ਸਵੈਚਾਲਿਤ mutex = std::make_shared<std::mutex>();
                ਇੰਦਰਾਜ = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[ਕੁੰਜੀ] = ਇੰਦਰਾਜ;
            }
        } ਨਹੀਂਤਾਂ {
            ਸਵੈਚਾਲਿਤ mutex = std::make_shared<std::mutex>();
            ਇੰਦਰਾਜ = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_ਪ੍ਰਾਪਤੀ = ਝੂਠ;
        }
        cache_lookup_ਸਮਾਂ += ggml_ਸਮਾਂ_us() - cache_lookup_ਸ਼ੁਰੂ_ਸਮਾਂ;

        std::lock_guard<std::mutex> ਤਾਲਾ(*(ਇੰਦਰਾਜ->mutex));
        cache_ਪ੍ਰਾਪਤੀ = cache_ਪ੍ਰਾਪਤੀ && ਇੰਦਰਾਜ->ptr && r_ctx->infer_request_cache.count(ਕੁੰਜੀ) != 0;

        ਜੇ (cache_ਪ੍ਰਾਪਤੀ) {
            ggml_decoder = ਇੰਦਰਾਜ->ptr;
            old_m_params = ggml_decoder->get_model_params();
            ਜੇ (!ggml_decoder->is_splited_model()) {
                cache_ਪ੍ਰਾਪਤੀ = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_ਨਿਵੇਸ਼_ਨਾਂ;
        std::vector<std::string> ov_output_ਨਾਂ;

        ਜੇ (cache_ਪ੍ਰਾਪਤੀ) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->ਸੈੱਟ_compute_params(c_params);
            ggml_decoder->ਸੈੱਟ_model_params(m_params);
            ਜੇ (old_m_params.kv_ਬਫਰ_ਬਦਲਿਆ(m_params) || !ggml_decoder->is_bound_ਮੰਜ਼ਿਲ(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->ਜੋੜੋ_extra_inputs();
            {
                std::lock_guard<std::mutex> map_ਤਾਲਾ(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(ਕੁੰਜੀ);
                ov_ਨਿਵੇਸ਼_ਨਾਂ = r_ctx->ov_ਨਿਵੇਸ਼_ਨਾਂ_cache.at(ਕੁੰਜੀ);
                ov_output_ਨਾਂ = r_ctx->ov_output_ਨਾਂ_cache.at(ਕੁੰਜੀ);
            }

            ਜੇ (stateful_kv_only) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_ਅੰਕੜੇ = (int32_t *) inp_pos->ਅੰਕੜੇ;
                ਸਵੈਚਾਲਿਤ pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                ਜੇ (pos_ਅੰਕੜੇ[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_ਮਾਪ = pos_shape[3];
                } ਨਹੀਂਤਾਂ ਜੇ (r_ctx->stateful_kv_ਮਾਪ == static_cast<size_t>(pos_ਅੰਕੜੇ[0])) {
                    r_ctx->stateful_kv_ਮਾਪ += pos_shape[3];
                } ਨਹੀਂਤਾਂ {
                    ਅਚਲ size_t pos_begin = static_cast<size_t>(pos_ਅੰਕੜੇ[0]);
                    ਅਚਲ ਤਾਰਕਿਕ refill = pos_begin > r_ctx->stateful_kv_ਮਾਪ;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    ਜੇ (refill && !ggml_decoder->get_model_params().swa_layers.ਖਾਲੀ_ਥਾਵਾਂ()) {
                        ਅਚਲ ਪੂਰਨ n_swa = ggml_decoder->get_compute_params().swa_ਖਿੜਕੀ;
                        ਜੇ (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_ਮਾਪ, n_swa);
                            ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
                        }
                    }

                    ਅਚਲ ਤਾਰਕਿਕ relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    ਸਵੈਚਾਲਿਤ ਹਾਲਤਾਂ = infer_request->query_state();
                    ਲਈ (ਸਵੈਚਾਲਿਤ state : ਹਾਲਤਾਂ) {
                        ਸਵੈਚਾਲਿਤ state_tensor = state.get_state();
                        ਸਵੈਚਾਲਿਤ state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        ਜੇ (ਸਵੈਚਾਲਿਤ it = r_ctx->kv_state_ਨਿਵੇਸ਼_name_map.find(state.get_name());
                            it != r_ctx->kv_state_ਨਿਵੇਸ਼_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        ਪੂਰਨ n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        ਜੇ (ਸਵੈਚਾਲਿਤ layer = ਕੱਢੋ_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        ਅਚਲ ਤਾਰਕਿਕ relayout_this_state = relayout_enabled;
                        ਅਚਲ size_t seq_axis = relayout_this_state ? 2 : 1;
                        ਅਚਲ size_t ਸਿਰ_axis = seq_axis == 2 ? 1 : 2;

                        ਜੇ (refill) {
                            ਜੇ (state_name.ਖਾਲੀ_ਥਾਵਾਂ()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
                            }
                            ਸਵੈਚਾਲਿਤ kv_tensor = get_ov_ਨਿਵੇਸ਼_tensor(ggml_decoder, state_name);
                            ਜੇ (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } ਨਹੀਂਤਾਂ {
                                ov::Shape refill_shape(4);
                                refill_shape[0] = state_tensor_shape[0];
                                refill_shape[seq_axis] = kv_tensor.get_shape()[2];
                                refill_shape[ਸਿਰ_axis] = state_tensor_shape[ਸਿਰ_axis];
                                refill_shape[3] = state_tensor_shape[3];
                                kv_tensor.ਸੈੱਟ_shape(refill_shape);
                                state_tensor = kv_tensor;
                            }
                            state_tensor_shape = state_tensor.get_shape();
                        }
                        // Only ever shrink to a prefix the source really has. Slicing past it used to
                        // surface as a bare ov::Exception from the ROI constructor.
                        ਜੇ (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end(state_tensor_shape.begin(), state_tensor_shape.end());
                        end[seq_axis] = pos_begin;
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.ਸੈੱਟ_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_ਮਾਪ = pos_begin + pos_shape[3];
                }
            }

            decoder_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
            conversion_ਸਿਰਾ_ਸਮਾਂ = decoder_ਸਿਰਾ_ਸਮਾਂ;
            compile_ਸਿਰਾ_ਸਮਾਂ = decoder_ਸਿਰਾ_ਸਮਾਂ;
        } ਨਹੀਂਤਾਂ {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            ਸਵੈਚਾਲਿਤ shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_ਤਾਲਾ(shared_cache->mutex);
            ਸਵੈਚਾਲਿਤ ਭਾਰ_ਨਾਂ = get_ਭਾਰ_ਨਾਂ(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, ਭਾਰ_ਨਾਂ,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ਅਚਲ std::string model_cache_dir = ggml_openvino_model_cache_dir();
            ਅਚਲ std::string exact_ਕੁੰਜੀ = cache_enabled ? compiled_ਗ੍ਰਾਫ_ਕੁੰਜੀ(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_ਰਸਤਾ, manifest_ਰਸਤਾ;
            ਜੇ (!model_cache_dir.ਖਾਲੀ_ਥਾਵਾਂ() && !model_is_splitted) {
                ਅਚਲ uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/ਸੱਚ, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_ਗ੍ਰਾਫ_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/ਸੱਚ, m_params.rope_params, 16, extra_cfg,
                                                    compiled_ਗ੍ਰਾਫ_ਕੁੰਜੀ(cgraph, *ggml_decoder, device, 0, ਸੱਚ));
                blob_ਰਸਤਾ = ggml_openvino_model_cache_blob_ਰਸਤਾ(model_cache_dir, model_fp);
                manifest_ਰਸਤਾ = ggml_openvino_model_cache_manifest_ਰਸਤਾ(model_cache_dir, model_fp);
            }
            std::string shared_ਕੁੰਜੀ =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_ਕੁੰਜੀ) : "";
            ov::CompiledModel shared_model;
            ਤਾਰਕਿਕ imported = ਝੂਠ;
            ਸਵੈਚਾਲਿਤ shared_it = shared_cache->graphs.find(shared_ਕੁੰਜੀ);
            ਜੇ (!shared_ਕੁੰਜੀ.ਖਾਲੀ_ਥਾਵਾਂ() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_ਗ੍ਰਾਫ(shared_it->second.decode, shared_it->second.ਨਿਵੇਸ਼_ਨਾਂ,
                                                               shared_it->second.output_ਨਾਂ, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_ਨਿਵੇਸ਼_ਨਾਂ = shared_it->second.ਨਿਵੇਸ਼_ਨਾਂ;
                ov_output_ਨਾਂ = shared_it->second.output_ਨਾਂ;
                imported = ਸੱਚ;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } ਨਹੀਂਤਾਂ ਜੇ (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_ਕੁੰਜੀ = exact_ਕੁੰਜੀ;
                shared_it = shared_cache->graphs.find(shared_ਕੁੰਜੀ);
                ਜੇ (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_ਨਿਵੇਸ਼_ਨਾਂ = shared_it->second.ਨਿਵੇਸ਼_ਨਾਂ;
                    ov_output_ਨਾਂ = shared_it->second.output_ਨਾਂ;
                    imported = ਸੱਚ;
                }
                model_fp = exact_fp;
                blob_ਰਸਤਾ = ggml_openvino_model_cache_blob_ਰਸਤਾ(model_cache_dir, model_fp);
                manifest_ਰਸਤਾ = ggml_openvino_model_cache_manifest_ਰਸਤਾ(model_cache_dir, model_fp);
            }
            ਜੇ (cache_enabled) {
                std::lock_guard<std::mutex> map_ਤਾਲਾ(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(ਕੁੰਜੀ);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            ਜੇ (!model_cache_dir.ਖਾਲੀ_ਥਾਵਾਂ()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_ਬਹੁਲਕ.name()] = ov::Cache_ਬਹੁਲਕ::OPTIMIZE_ਰਫਤਾਰ;
            }
            ਜੇ (!imported && !model_cache_dir.ਖਾਲੀ_ਥਾਵਾਂ() && !model_is_splitted) {
                ਅਚਲ uint64_t preferred_fp = model_fp;
                ਤਾਰਕਿਕ dynamic_incompatible = preferred_fp == exact_fp;
                ਲਈ (ਪੂਰਨ candidate = 0; candidate < 2; ++candidate) {
                    ਜੇ (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        ਰੋਕੋ;
                    }
                    ਅਚਲ uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    ਅਚਲ std::string candidate_blob =
                        ggml_openvino_model_cache_blob_ਰਸਤਾ(model_cache_dir, candidate_fp);
                    ਅਚਲ std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_ਰਸਤਾ(model_cache_dir, candidate_fp);
                    std::ifstream blob_ਵਿੱਚ(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    ਜੇ (!blob_ਵਿੱਚ.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        ਜਾਰੀ;
                    }
                    ਅਚਲ int64_t import_ਸ਼ੁਰੂ = ggml_ਸਮਾਂ_us();
                    ਕੋਸ਼ਿਸ਼ {
                        ov::CompiledModel cm;
                        ਸਵੈਚਾਲਿਤ remote_context = ggml_openvino_get_remote_context();
                        ਜੇ (remote_context.has_value()) {
                            cm = core.import_model(blob_ਵਿੱਚ, remote_context.value(), mc_config);
                        } ਨਹੀਂਤਾਂ {
                            cm = core.import_model(blob_ਵਿੱਚ, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        ਜੇ (!saved_inputs.ਖਾਲੀ_ਥਾਵਾਂ() || !saved_outputs.ਖਾਲੀ_ਥਾਵਾਂ()) {
                            ov_ਨਿਵੇਸ਼_ਨਾਂ = std::move(saved_inputs);
                            ov_output_ਨਾਂ = std::move(saved_outputs);
                        } ਨਹੀਂਤਾਂ {
                            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & p : cm.inputs()) {
                                ov_ਨਿਵੇਸ਼_ਨਾਂ.push_back(p.get_ਗੰਢ()->get_friendly_name());
                            }
                            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & o : cm.outputs()) {
                                ov_output_ਨਾਂ.push_back(o.get_ਗੰਢ()->get_friendly_name());
                            }
                        }
                        ਜੇ (!compiled_model_matches_ਗ੍ਰਾਫ(cm, ov_ਨਿਵੇਸ਼_ਨਾਂ, ov_output_ਨਾਂ, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_ਨਿਵੇਸ਼_ਨਾਂ.clear();
                            ov_output_ਨਾਂ.clear();
                            ਜਾਰੀ;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        ਇੰਦਰਾਜ->ptr = ggml_decoder;
                        imported = ਸੱਚ;
                        model_fp = candidate_fp;
                        blob_ਰਸਤਾ = candidate_blob;
                        manifest_ਰਸਤਾ = candidate_manifest;
                        ਜੇ (candidate_fp == exact_fp) {
                            shared_ਕੁੰਜੀ = exact_ਕੁੰਜੀ;
                            shared_it = shared_cache->graphs.find(shared_ਕੁੰਜੀ);
                        }
                        ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_ਸਮਾਂ_us() - import_ਸ਼ੁਰੂ) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } ਫੜੋ (ਅਚਲ std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = ਝੂਠ;
                        ov_ਨਿਵੇਸ਼_ਨਾਂ.clear();
                        ov_output_ਨਾਂ.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                ਜੇ (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_ਕੁੰਜੀ = exact_ਕੁੰਜੀ;
                    shared_it = shared_cache->graphs.find(shared_ਕੁੰਜੀ);
                    blob_ਰਸਤਾ = ggml_openvino_model_cache_blob_ਰਸਤਾ(model_cache_dir, model_fp);
                    manifest_ਰਸਤਾ = ggml_openvino_model_cache_manifest_ਰਸਤਾ(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            ਜੇ (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_ਰਸਤਾ.c_str());
                ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
            }
            ਜੇ (!imported && ggml_openvino_ਭਾਰ_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
            }
            ਜੇ (imported) {
                decoder_ਸਿਰਾ_ਸਮਾਂ = conversion_ਸਿਰਾ_ਸਮਾਂ = compile_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
            } ਨਹੀਂਤਾਂ {
                ਸਵੈਚਾਲਿਤ model_weights = GgmlOvDecoder::create_ਭਾਰ_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();

                ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_model = std::make_shared<ov::frontend::ggml::ਨਿਵੇਸ਼_Model>(ggml_decoder);
                model = ov::frontend::ggml::Front_ਸਿਰਾ::convert(ਨਿਵੇਸ਼_model);
                ggml_decoder->clear_model_weights();
                conversion_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();

                ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    ਅੱਖਰ timestamped_filename[64];
                    ਸਵੈਚਾਲਿਤ timestamp = (ਲੰਮਾ ਲੰਮਾ) ggml_ਸਮਾਂ_us();
                    snprintf(timestamped_filename, ਆਕਾਰ(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                ਅਚਲ ov::AnyMap & compile_config = model_cache_dir.ਖਾਲੀ_ਥਾਵਾਂ() ? config : mc_config;
                ov::CompiledModel compiled_model;
                ਸਵੈਚਾਲਿਤ remote_context = ggml_openvino_get_remote_context();
                ਜੇ (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } ਨਹੀਂਤਾਂ {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();

                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ov_param : model->get_parameters()) {
                    ov_ਨਿਵੇਸ਼_ਨਾਂ.push_back(ov_param->get_friendly_name());
                }
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ov_output : model->get_results()) {
                    ov_output_ਨਾਂ.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                ਜੇ (!model_cache_dir.ਖਾਲੀ_ਥਾਵਾਂ() && !model_is_splitted && model_fp != 0) {
                    ਕੋਸ਼ਿਸ਼ {
                        ਅਚਲ std::string blob_tmp = ggml_openvino_model_cache_ਆਰਜ਼ੀ_ਰਸਤਾ(blob_ਰਸਤਾ);
                        ਅਚਲ std::string manifest_tmp = ggml_openvino_model_cache_ਆਰਜ਼ੀ_ਰਸਤਾ(manifest_ਰਸਤਾ);
                        ਜੇ (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_ਨਿਵੇਸ਼_ਨਾਂ,
                                                                     ov_output_ਨਾਂ)) {
                            std::ofstream blob_ਬਾਹਰਲਾ(blob_tmp, std::ios::binary | std::ios::trunc);
                            ਜੇ (blob_ਬਾਹਰਲਾ.is_open()) {
                                compiled_model.export_model(blob_ਬਾਹਰਲਾ);
                                blob_ਬਾਹਰਲਾ.close();
                                ਜੇ (blob_ਬਾਹਰਲਾ.good()) {
                                    ਜੇ (std::rename(blob_tmp.c_str(), blob_ਰਸਤਾ.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_ਰਸਤਾ.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_ਰਸਤਾ.c_str());
                                    } ਨਹੀਂਤਾਂ {
                                        std::remove(blob_tmp.c_str());
                                        std::remove(manifest_tmp.c_str());
                                    }
                                } ਨਹੀਂਤਾਂ {
                                    std::remove(blob_tmp.c_str());
                                    std::remove(manifest_tmp.c_str());
                                }
                            } ਨਹੀਂਤਾਂ {
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    } ਫੜੋ (ਅਚਲ std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                ਇੰਦਰਾਜ->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            ਇੰਦਰਾਜ->ptr = ggml_decoder;
            ਜੇ (!shared_ਕੁੰਜੀ.ਖਾਲੀ_ਥਾਵਾਂ() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_ਕੁੰਜੀ,
                                             ov_compiled_ਗ੍ਰਾਫ{shared_model, {}, ov_ਨਿਵੇਸ਼_ਨਾਂ, ov_output_ਨਾਂ});
            }
            ਜੇ (cache_enabled) {
                std::lock_guard<std::mutex> map_ਤਾਲਾ(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[ਕੁੰਜੀ] = infer_request;
                r_ctx->ov_ਨਿਵੇਸ਼_ਨਾਂ_cache[ਕੁੰਜੀ] = ov_ਨਿਵੇਸ਼_ਨਾਂ;
                r_ctx->ov_output_ਨਾਂ_cache[ਕੁੰਜੀ] = ov_output_ਨਾਂ;
            }

            ਜੇ (stateful_kv_only && cache_enabled) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ * inp_pos = get_inp_pos_tensor(cgraph);
                ਸਵੈਚਾਲਿਤ pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                ਅਚਲ int32_t pos_begin = ((int32_t *) inp_pos->ਅੰਕੜੇ)[0];
                ਜੇ (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
                }
                r_ctx->stateful_kv_ਮਾਪ = pos_shape[3];
                ਅਚਲ ਸਵੈਚਾਲਿਤ kv_param_res_ਨਾਂ = ggml_decoder->get_kv_param_res_ਨਾਂ();
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & pair : kv_param_res_ਨਾਂ) {
                    r_ctx->kv_state_ਨਿਵੇਸ਼_name_map[pair.ਪਹਿਲਾ + pair.second] = pair.ਪਹਿਲਾ;
                }
            }
        }

        ਜੇ (stateful_recurrent) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * inp_pos = get_inp_pos_tensor(cgraph);
            ਅਚਲ int32_t pos_begin = static_cast<ਅਚਲ int32_t *>(inp_pos->ਅੰਕੜੇ)[0];
            ਜੇ (pos_begin == 0) {
                infer_request->reset_state();
            } ਨਹੀਂਤਾਂ ਜੇ (!cache_ਪ੍ਰਾਪਤੀ || old_m_params.kv_ਬਫਰ_ਬਦਲਿਆ(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_ਮਾਪ) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
            }
        } ਨਹੀਂਤਾਂ {
            reset_single_ਥਾਂ_recurrent_cache(cgraph, m_params, c_params);
        }

        ਲਈ (size_t i = 0; i < ov_ਨਿਵੇਸ਼_ਨਾਂ.size(); i++) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ & param_name = ov_ਨਿਵੇਸ਼_ਨਾਂ[i];
            ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_tensor = get_ov_ਨਿਵੇਸ਼_tensor(ggml_decoder, param_name);
            infer_request->ਸੈੱਟ_ਨਿਵੇਸ਼_tensor(i, ਨਿਵੇਸ਼_tensor);

            ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_ਨਿਵੇਸ਼_tensor_info(param_name, ਨਿਵੇਸ਼_tensor);
            }
        }

        ਲਈ (size_t i = 0; i < ov_output_ਨਾਂ.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            ਅਚਲ ਸਵੈਚਾਲਿਤ & model_outputs = ggml_decoder->get_model_outputs();
            ਸਵੈਚਾਲਿਤ model_output_it = model_outputs.find(ov_output_ਨਾਂ[i]);
            ਜੇ (model_output_it == model_outputs.end()) {
                ਜਾਰੀ;
            }
            ਸਵੈਚਾਲਿਤ * ggml_tensor = model_output_it->second;
            ਜੇ (ggml_nbytes(ggml_tensor) == 0) {
                ਜਾਰੀ;
            }
            ਸਵੈਚਾਲਿਤ output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->ਸੈੱਟ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_ਸ਼ੁਰੂ = ggml_ਸਮਾਂ_us();
        infer_request->infer();
        infer_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
        ਜੇ (stateful_recurrent) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_ਮਾਪ =
                static_cast<ਅਚਲ int32_t *>(inp_pos->ਅੰਕੜੇ)[0] + get_inp_pos_n_ਟੋਕਨ(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            ਲਈ (size_t i = 0; i < ov_output_ਨਾਂ.size(); i++) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_ਨਾਂ[i], output_tensor, output_tensor.ਅੰਕੜੇ());
            }
        }

        ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_ਸਿਰਾ_ਸਮਾਂ - ਸ਼ੁਰੂ_ਸਮਾਂ) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_ਕੁੰਜੀ_compute_ਸਮਾਂ / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_ਵੰਡੋ_check_ਸਮਾਂ / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", ਗ੍ਰਾਫ_param_compute_ਸਮਾਂ / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_ਸਮਾਂ / 1000.0);
            ਜੇ (!cache_ਪ੍ਰਾਪਤੀ) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_ਸਿਰਾ_ਸਮਾਂ - decoder_ਸਿਰਾ_ਸਮਾਂ) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_ਸਿਰਾ_ਸਮਾਂ - conversion_ਸਿਰਾ_ਸਮਾਂ) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_ਸਿਰਾ_ਸਮਾਂ - compile_ਸਿਰਾ_ਸਮਾਂ) / 1000.0);
            GGML_LOG_INFO("    - OV raw infer time: %.3f ms \n", (infer_ਸਿਰਾ_ਸਮਾਂ - ov_raw_infer_ਸ਼ੁਰੂ) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. Release only while holding the compilation mutex so another context cannot
    // be reading host weights during conversion/compilation. Pin the shared compiled
    // models across backend teardown; a later context can create its own request without
    // reading the dropped pages. A new, uncached graph still fails fast above.
    ਜੇ (cache_ਪ੍ਰਾਪਤੀ && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_ਤਾਲਾ(r_ctx->compiled_cache->mutex);
        ਜੇ (!ggml_openvino_ਭਾਰ_buffers_released()) {
            ggml_openvino_release_ਭਾਰ_buffers();
        }
    }

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ov::AnyMap without_npuw(ਅਚਲ ov::AnyMap & config) {
    ov::AnyMap ਬਾਹਰਲਾ;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & kv : config) {
        ਜੇ (kv.ਪਹਿਲਾ.rfind("NPUW", 0) == 0 || kv.ਪਹਿਲਾ == "NPU_USE_NPUW") {
            ਜਾਰੀ;
        }
        ਬਾਹਰਲਾ.insert(kv);
    }
    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ov_ਗ੍ਰਾਫ_compute_static(ggml_cgraph * cgraph, ਅਚਲ std::shared_ptr<ov_runtime_context> & r_ctx) {
    ਸਵੈਚਾਲਿਤ & core = ov_singleton_core();

    ਸਵੈਚਾਲਿਤ get_prefill_chunk_ਮਾਪ = [] {
        ਸਥਿਰ ਅਚਲ ਪੂਰਨ chunk_ਮਾਪ = []() {
            ਪੂਰਨ env_prefill_chunk_ਮਾਪ = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            ਵਾਪਸ env_prefill_chunk_ਮਾਪ > 0 ? env_prefill_chunk_ਮਾਪ : 256;
        }();
        ਵਾਪਸ chunk_ਮਾਪ;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    ਸਥਿਰ std::string device = ggml_openvino_get_device_name();
    ਸਥਿਰ ਸਵੈਚਾਲਿਤ is_static = ਸੱਚ;
    ਸਥਿਰ ਸਵੈਚਾਲਿਤ stateful = ਝੂਠ;

    ਸਵੈਚਾਲਿਤ prefill_chunk_ਮਾਪ = get_prefill_chunk_ਮਾਪ();
    ਅਚਲ ਸਵੈਚਾਲਿਤ & config = ggml_openvino_get_compile_config();

    ਜੇ (is_naive(cgraph)) {
        ਵਾਪਸ naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    ਸਵੈਚਾਲਿਤ ਸ਼ੁਰੂ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    ਅਚਲ ਸਵੈਚਾਲਿਤ * inp_pos = get_inp_pos_tensor(cgraph);
    ਅਚਲ ਤਾਰਕਿਕ no_kv_cache = m_params.is_cacheless_attn;
    ਅਚਲ ਸਵੈਚਾਲਿਤ is_prefill = no_kv_cache ? ਸੱਚ : get_is_prefill(cgraph, inp_pos);
    ਅਚਲ ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    ਜੇ (m_params.n_heads_kv == -1) {
        prefill_chunk_ਮਾਪ = inp_pos->ne[0];
    }
    ਗ੍ਰਾਫ_ਕੁੰਜੀ ਕੁੰਜੀ(cgraph);
    ਸਥਿਰ ਅਚਲ ਤਾਰਕਿਕ cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    ਤਾਰਕਿਕ cache_ਪ੍ਰਾਪਤੀ = ਝੂਠ;

    int64_t decoder_ਸਿਰਾ_ਸਮਾਂ;
    int64_t conversion_ਸਿਰਾ_ਸਮਾਂ;
    int64_t compile_ਸਿਰਾ_ਸਮਾਂ;
    int64_t infer_ਸਿਰਾ_ਸਮਾਂ;
    int64_t ov_raw_infer_ਸ਼ੁਰੂ;
    int64_t ov_raw_infer_ਕੁੱਲ = 0;

    std::shared_ptr<decoder_runtime_ctx> ਇੰਦਰਾਜ;
    ModelParams old_m_params;

    ਜੇ (cache_enabled) {
        std::lock_guard<std::mutex> map_ਤਾਲਾ(r_ctx->ctx_mutex);
        ਸਵੈਚਾਲਿਤ it = r_ctx->decoder_cache.find(ਕੁੰਜੀ);
        cache_ਪ੍ਰਾਪਤੀ = it != r_ctx->decoder_cache.end();
        ਜੇ (cache_ਪ੍ਰਾਪਤੀ) {
            ਇੰਦਰਾਜ = it->second;
        } ਨਹੀਂਤਾਂ {
            r_ctx->clear_caches_locked();
            ਸਵੈਚਾਲਿਤ mutex = std::make_shared<std::mutex>();
            ਇੰਦਰਾਜ = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[ਕੁੰਜੀ] = ਇੰਦਰਾਜ;
        }
    } ਨਹੀਂਤਾਂ {
        ਸਵੈਚਾਲਿਤ mutex = std::make_shared<std::mutex>();
        ਇੰਦਰਾਜ = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_ਪ੍ਰਾਪਤੀ = ਝੂਠ;
    }

    std::lock_guard<std::mutex> ਤਾਲਾ(*(ਇੰਦਰਾਜ->mutex));
    cache_ਪ੍ਰਾਪਤੀ = cache_ਪ੍ਰਾਪਤੀ && ਇੰਦਰਾਜ->ptr && r_ctx->infer_request_cache.count(ਕੁੰਜੀ) != 0 &&
                r_ctx->infer_request_cache_prefill.count(ਕੁੰਜੀ) != 0;

    ਜੇ (cache_ਪ੍ਰਾਪਤੀ) {
        ggml_decoder = ਇੰਦਰਾਜ->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_ਪ੍ਰਾਪਤੀ = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_ਨਿਵੇਸ਼_ਨਾਂ_local;
    std::vector<std::string> ov_output_ਨਾਂ_local;

    ਜੇ (cache_ਪ੍ਰਾਪਤੀ) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->ਸੈੱਟ_model_params(m_params);
        ggml_decoder->ਸੈੱਟ_compute_params(c_params);
        ਜੇ (old_m_params.kv_ਬਫਰ_ਬਦਲਿਆ(m_params) || !ggml_decoder->is_bound_ਮੰਜ਼ਿਲ(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->ਜੋੜੋ_extra_inputs();
        {
            std::lock_guard<std::mutex> map_ਤਾਲਾ(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(ਕੁੰਜੀ) : r_ctx->infer_request_cache.at(ਕੁੰਜੀ);
            ov_ਨਿਵੇਸ਼_ਨਾਂ_local = r_ctx->ov_ਨਿਵੇਸ਼_ਨਾਂ_cache.at(ਕੁੰਜੀ);
            ov_output_ਨਾਂ_local = r_ctx->ov_output_ਨਾਂ_cache.at(ਕੁੰਜੀ);
        }

        decoder_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
        conversion_ਸਿਰਾ_ਸਮਾਂ = decoder_ਸਿਰਾ_ਸਮਾਂ;
        compile_ਸਿਰਾ_ਸਮਾਂ = decoder_ਸਿਰਾ_ਸਮਾਂ;
    } ਨਹੀਂਤਾਂ {
        ਜੇ (cache_enabled) {
            std::lock_guard<std::mutex> map_ਤਾਲਾ(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(ਕੁੰਜੀ);
            r_ctx->infer_request_cache_prefill.erase(ਕੁੰਜੀ);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        ਸਵੈਚਾਲਿਤ shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_ਤਾਲਾ(shared_cache->mutex);
        ਸਵੈਚਾਲਿਤ ਭਾਰ_ਨਾਂ = get_ਭਾਰ_ਨਾਂ(cgraph);
        ਸਵੈਚਾਲਿਤ local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, ਭਾਰ_ਨਾਂ, is_static,
                                                             stateful, ਝੂਠ, is_prefill, prefill_chunk_ਮਾਪ);
        ਅਚਲ std::string shared_ਕੁੰਜੀ =
            cache_enabled ? compiled_ਗ੍ਰਾਫ_ਕੁੰਜੀ(cgraph, *local_decoder, device, prefill_chunk_ਮਾਪ) : "";
        ਸਵੈਚਾਲਿਤ shared_it = shared_cache->graphs.find(shared_ਕੁੰਜੀ);
        ਜੇ (!shared_ਕੁੰਜੀ.ਖਾਲੀ_ਥਾਵਾਂ() && shared_it != shared_cache->graphs.end()) {
            ਸਵੈਚਾਲਿਤ & compiled = shared_it->second;
            ਸਵੈਚਾਲਿਤ prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            ਸਵੈਚਾਲਿਤ decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            ਇੰਦਰਾਜ->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_ਨਿਵੇਸ਼_ਨਾਂ_local = compiled.ਨਿਵੇਸ਼_ਨਾਂ;
            ov_output_ਨਾਂ_local = compiled.output_ਨਾਂ;
            r_ctx->infer_request_cache_prefill[ਕੁੰਜੀ] = prefill_request;
            r_ctx->infer_request_cache[ਕੁੰਜੀ] = decode_request;
            r_ctx->ov_ਨਿਵੇਸ਼_ਨਾਂ_cache[ਕੁੰਜੀ] = ov_ਨਿਵੇਸ਼_ਨਾਂ_local;
            r_ctx->ov_output_ਨਾਂ_cache[ਕੁੰਜੀ] = ov_output_ਨਾਂ_local;
            decoder_ਸਿਰਾ_ਸਮਾਂ = conversion_ਸਿਰਾ_ਸਮਾਂ = compile_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } ਨਹੀਂਤਾਂ {
            std::shared_ptr<ov::Model> model;
            ਸਵੈਚਾਲਿਤ model_weights = GgmlOvDecoder::create_ਭਾਰ_nodes(cgraph);

            ਸਵੈਚਾਲਿਤ ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, ਝੂਠ, ਸੱਚ, prefill_chunk_ਮਾਪ);
            ਸਵੈਚਾਲਿਤ ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, ਝੂਠ, ਝੂਠ, prefill_chunk_ਮਾਪ);
            decoder_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();

            ਅਚਲ ਤਾਰਕਿਕ dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            ਅਚਲ ਸਵੈਚਾਲਿਤ dump_ir_timestamp = static_cast<ਲੰਮਾ ਲੰਮਾ>(ggml_ਸਮਾਂ_us());

            ਸਵੈਚਾਲਿਤ build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          ਅਚਲ std::shared_ptr<GgmlOvDecoder> & decoder, ਅਚਲ ਅੱਖਰ * ਟੈਗ,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_ਸਿਰਾ_ਸਮਾਂ, int64_t & local_compile_ਸਿਰਾ_ਸਮਾਂ) {
                ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_model = std::make_shared<ov::frontend::ggml::ਨਿਵੇਸ਼_Model>(decoder);
                model = ov::frontend::ggml::Front_ਸਿਰਾ::convert(ਨਿਵੇਸ਼_model);
                decoder->clear_model_weights();
                local_conversion_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();

                ਜੇ (dump_ir) {
                    ਅੱਖਰ timestamped_filename[64];
                    snprintf(timestamped_filename, ਆਕਾਰ(timestamped_filename), "model_%s_%lld.xml", ਟੈਗ,
                             dump_ir_timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                compiled_model = core.compile_model(model, device, compile_config);
                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                local_compile_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
            };
            std::shared_ptr<ov::Model> model_prefill;
            std::shared_ptr<ov::Model> model_decode;
            ov::CompiledModel compiled_model_prefill;
            ov::CompiledModel compiled_model_decode;
            std::shared_ptr<ov::InferRequest> infer_request_prefill;
            std::shared_ptr<ov::InferRequest> infer_request_decode;
            int64_t prefill_conversion_ਸਿਰਾ_ਸਮਾਂ;
            int64_t decode_conversion_ਸਿਰਾ_ਸਮਾਂ;
            int64_t prefill_compile_ਸਿਰਾ_ਸਮਾਂ;
            int64_t decode_compile_ਸਿਰਾ_ਸਮਾਂ;
            build_static_model(ggml_decoder_prefill, "prefill", model_prefill, compiled_model_prefill,
                               infer_request_prefill, prefill_conversion_ਸਿਰਾ_ਸਮਾਂ, prefill_compile_ਸਿਰਾ_ਸਮਾਂ);
            ਜੇ (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_ਸਿਰਾ_ਸਮਾਂ = prefill_conversion_ਸਿਰਾ_ਸਮਾਂ;
                decode_compile_ਸਿਰਾ_ਸਮਾਂ = prefill_compile_ਸਿਰਾ_ਸਮਾਂ;
            } ਨਹੀਂਤਾਂ {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_ਸਿਰਾ_ਸਮਾਂ, decode_compile_ਸਿਰਾ_ਸਮਾਂ);
            }
            conversion_ਸਿਰਾ_ਸਮਾਂ = std::max(prefill_conversion_ਸਿਰਾ_ਸਮਾਂ, decode_conversion_ਸਿਰਾ_ਸਮਾਂ);
            compile_ਸਿਰਾ_ਸਮਾਂ = std::max(prefill_compile_ਸਿਰਾ_ਸਮਾਂ, decode_compile_ਸਿਰਾ_ਸਮਾਂ);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            ਇੰਦਰਾਜ->ptr = ggml_decoder;

            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ov_param : model->get_parameters()) {
                ov_ਨਿਵੇਸ਼_ਨਾਂ_local.push_back(ov_param->get_friendly_name());
            }
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ov_output : model->get_results()) {
                ov_output_ਨਾਂ_local.push_back(ov_output->get_friendly_name());
            }

            ਜੇ (!shared_ਕੁੰਜੀ.ਖਾਲੀ_ਥਾਵਾਂ()) {
                shared_cache->graphs.emplace(
                    shared_ਕੁੰਜੀ, ov_compiled_ਗ੍ਰਾਫ{compiled_model_decode, compiled_model_prefill, ov_ਨਿਵੇਸ਼_ਨਾਂ_local,
                                                  ov_output_ਨਾਂ_local});
            }

            ਜੇ (cache_enabled) {
                std::lock_guard<std::mutex> map_ਤਾਲਾ(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[ਕੁੰਜੀ] = infer_request_prefill;
                r_ctx->infer_request_cache[ਕੁੰਜੀ] = infer_request_decode;
                r_ctx->ov_ਨਿਵੇਸ਼_ਨਾਂ_cache[ਕੁੰਜੀ] = ov_ਨਿਵੇਸ਼_ਨਾਂ_local;
                r_ctx->ov_output_ਨਾਂ_cache[ਕੁੰਜੀ] = ov_output_ਨਾਂ_local;
            }
        }
    }

    reset_single_ਥਾਂ_recurrent_cache(cgraph, m_params, c_params);

    ਜੇ (is_prefill) {
        ਸਵੈਚਾਲਿਤ inp_len = get_inp_pos_n_ਟੋਕਨ(cgraph, inp_pos);
        ਲਈ (ਪੂਰਨ chunk_ਸੂਚਕਾਂਕ = 0; chunk_ਸੂਚਕਾਂਕ * prefill_chunk_ਮਾਪ < inp_len; chunk_ਸੂਚਕਾਂਕ++) {
            ਲਈ (size_t i = 0; i < ov_ਨਿਵੇਸ਼_ਨਾਂ_local.size(); i++) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ & param_name = ov_ਨਿਵੇਸ਼_ਨਾਂ_local[i];
                ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_tensor = get_ov_ਨਿਵੇਸ਼_tensor_static_prefill(ggml_decoder, param_name, chunk_ਸੂਚਕਾਂਕ);
                infer_request->ਸੈੱਟ_ਨਿਵੇਸ਼_tensor(i, ਨਿਵੇਸ਼_tensor);

                ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    ਅਚਲ ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_tensor = infer_request->get_ਨਿਵੇਸ਼_tensor(i);
                    print_ਨਿਵੇਸ਼_tensor_info(param_name, ਨਿਵੇਸ਼_tensor);
                }
            }

            ਲਈ (size_t i = 0; i < ov_output_ਨਾਂ_local.size(); i++) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ & model_outputs = ggml_decoder->get_model_outputs();
                ਸਵੈਚਾਲਿਤ model_output_it = model_outputs.find(ov_output_ਨਾਂ_local[i]);
                ਜੇ (model_output_it == model_outputs.end()) {
                    ਜਾਰੀ;
                }
                ਸਵੈਚਾਲਿਤ * ggml_tensor = model_output_it->second;
                ਜੇ (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    ਜਾਰੀ;
                }
                ਸਵੈਚਾਲਿਤ output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->ਸੈੱਟ_output_tensor(i, output_tensor);
            }

            ov_raw_infer_ਸ਼ੁਰੂ = ggml_ਸਮਾਂ_us();
            infer_request->infer();
            ov_raw_infer_ਕੁੱਲ += ggml_ਸਮਾਂ_us() - ov_raw_infer_ਸ਼ੁਰੂ;

            ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                ਲਈ (size_t i = 0; i < ov_output_ਨਾਂ_local.size(); i++) {
                    ਅਚਲ ਸਵੈਚਾਲਿਤ output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_ਨਾਂ_local[i], output_tensor, output_tensor.ਅੰਕੜੇ());
                }
            }
        }
        infer_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
    } ਨਹੀਂਤਾਂ {
        ਲਈ (size_t i = 0; i < ov_ਨਿਵੇਸ਼_ਨਾਂ_local.size(); i++) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ & param_name = ov_ਨਿਵੇਸ਼_ਨਾਂ_local[i];
            ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_tensor = get_ov_ਨਿਵੇਸ਼_tensor_static_decode(ggml_decoder, param_name);
            infer_request->ਸੈੱਟ_ਨਿਵੇਸ਼_tensor(i, ਨਿਵੇਸ਼_tensor);

            ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_tensor = infer_request->get_ਨਿਵੇਸ਼_tensor(i);
                print_ਨਿਵੇਸ਼_tensor_info(param_name, ਨਿਵੇਸ਼_tensor);
            }
        }

        ਲਈ (size_t i = 0; i < ov_output_ਨਾਂ_local.size(); i++) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ & model_outputs = ggml_decoder->get_model_outputs();
            ਸਵੈਚਾਲਿਤ model_output_it = model_outputs.find(ov_output_ਨਾਂ_local[i]);
            ਜੇ (model_output_it == model_outputs.end()) {
                ਜਾਰੀ;
            }
            ਸਵੈਚਾਲਿਤ * ggml_tensor = model_output_it->second;
            ਜੇ (ggml_nbytes(ggml_tensor) == 0) {
                ਜਾਰੀ;
            }
            ਸਵੈਚਾਲਿਤ output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->ਸੈੱਟ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_ਸ਼ੁਰੂ = ggml_ਸਮਾਂ_us();
        infer_request->infer();
        infer_ਸਿਰਾ_ਸਮਾਂ = ggml_ਸਮਾਂ_us();
        ov_raw_infer_ਕੁੱਲ = infer_ਸਿਰਾ_ਸਮਾਂ - ov_raw_infer_ਸ਼ੁਰੂ;

        ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            ਲਈ (size_t i = 0; i < ov_output_ਨਾਂ_local.size(); i++) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_ਨਾਂ_local[i], output_tensor, output_tensor.ਅੰਕੜੇ());
            }
        }
    }

    ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_ਸਿਰਾ_ਸਮਾਂ - ਸ਼ੁਰੂ_ਸਮਾਂ) / 1000.0);
        ਜੇ (!cache_ਪ੍ਰਾਪਤੀ) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_ਸਿਰਾ_ਸਮਾਂ - decoder_ਸਿਰਾ_ਸਮਾਂ) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_ਸਿਰਾ_ਸਮਾਂ - conversion_ਸਿਰਾ_ਸਮਾਂ) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_ਸਿਰਾ_ਸਮਾਂ - compile_ਸਿਰਾ_ਸਮਾਂ) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_ਕੁੱਲ / 1000.0);
    }

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
ਸਥਿਰ ggml_cgraph * get_compute_ਗ੍ਰਾਫ(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    ਸਵੈਚਾਲਿਤ is_skipped = [](ਅਚਲ ggml_tensor * ਗੰਢ) {
        ਵਾਪਸ ਗੰਢ->op != GGML_OP_NONE && !(ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    ਜੇ (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        ਵਾਪਸ cgraph;
    }
    ਸਵੈਚਾਲਿਤ & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](ਅਚਲ ggml_tensor * ਗੰਢ) { ਵਾਪਸ !is_skipped(ਗੰਢ); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.ਅੰਕੜੇ();
    compute.graph.n_nodes = (ਪੂਰਨ) compute.nodes.size();
    ਵਾਪਸ &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
ਗਿਣਤੀ ggml_ਸਥਿਤੀ ov_ਗ੍ਰਾਫ_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    ਕੋਸ਼ਿਸ਼ {
        ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != ਖਾਲੀ_ਸੂਚਕ);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_ਤਾਲਾ(r_ctx->execution_mutex);
        cgraph = get_compute_ਗ੍ਰਾਫ(cgraph, *r_ctx);

        ਵਾਪਸ is_static ? ov_ਗ੍ਰਾਫ_compute_static(cgraph, r_ctx) : ov_ਗ੍ਰਾਫ_compute_dynamic(cgraph, r_ctx);
    } ਫੜੋ (ਅਚਲ ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
    } ਫੜੋ (ਅਚਲ std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
    } ਫੜੋ (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
ਤਾਰਕਿਕ is_model_splitted(ggml_cgraph * cgraph) {
    ਸਥਿਰ ਅਚਲ ਤਾਰਕਿਕ fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    ਜੇ (!fallback_enabled) {
        ਵਾਪਸ ਝੂਠ;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    ਜੇ (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ਗੰਢ = cgraph->nodes[i];
        ਪੂਰਨ use_ਗਿਣਤੀ = cgraph->use_counts[ggml_hash_ਲੱਭੋ(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ, ਗੰਢ)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        ਜੇ ((cgraph->n_nodes <= 1 && use_ਗਿਣਤੀ == 0) ||
            (cgraph->n_nodes <= 1 && ਗੰਢ->op == GGML_OP_VIEW && use_ਗਿਣਤੀ == 1 && ਗੰਢ->src[0] != ਖਾਲੀ_ਸੂਚਕ &&
             ਗੰਢ->src[0]->op == GGML_OP_NONE)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਪੂਰਨ ਨਿਵੇਸ਼_use_ਗਿਣਤੀ = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_ਗੰਢ = cgraph->nodes[j];
            ਲਈ (ਪੂਰਨ k = 0; k < GGML_MAX_SRC; k++) {
                ਜੇ (other_ਗੰਢ->src[k] == ਗੰਢ) {
                    ਨਿਵੇਸ਼_use_ਗਿਣਤੀ++;
                }
            }
        }
        ਜੇ (use_ਗਿਣਤੀ != ਨਿਵੇਸ਼_use_ਗਿਣਤੀ && ਗੰਢ->op != GGML_OP_NONE) {
            ਵਾਪਸ ਸੱਚ;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    ਜੇ (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_ਭਾਰ_ਨਾਂ(cgraph);
    } ਨਹੀਂਤਾਂ {
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & kv : GgmlOvDecoder::create_ਭਾਰ_nodes(cgraph, ਸੱਚ)) {
            model_weights.insert(kv.ਪਹਿਲਾ);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ਗੰਢ = cgraph->nodes[i];
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = ਗੰਢ->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            ਜੇ (src != ਖਾਲੀ_ਸੂਚਕ && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.ਖਾਲੀ_ਥਾਵਾਂ() == ਝੂਠ &&
                model_leafs.find(src) == model_leafs.end()) {
                ਜੇ (GgmlOvDecoder::is_inp_tok(src, ਗੰਢ)) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਵਾਪਸ ਸੱਚ;
            }
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

ਤਾਰਕਿਕ is_naive(ggml_cgraph * cgraph) {
    constexpr ਪੂਰਨ naive_ਗ੍ਰਾਫ_ਮਾਪ_ਹੱਦ = 20;
    ਪੂਰਨ count = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਜੇ (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    ਵਾਪਸ count < naive_ਗ੍ਰਾਫ_ਮਾਪ_ਹੱਦ;
}

size_t checksum(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size) {
    ਅਚਲ uint8_t * ਬਾਈਟਾਂ = static_cast<ਅਚਲ uint8_t *>(ਅੰਕੜੇ);
    size_t ਜੋੜ = 0;
    ਲਈ (size_t i = 0; i < size; ++i) {
        ਜੋੜ += (uint8_t) i;
        ਜੋੜ += ਬਾਈਟਾਂ[i];
    }
    ਵਾਪਸ ਜੋੜ;
}

ਤਾਰਕਿਕ save_ggml_tensor_ਅੰਕੜੇ_ਮੰਜ਼ਿਲ_txt(ਅਚਲ ggml_tensor * tensor, ਅਚਲ std::string & file_ਰਸਤਾ) {
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ || tensor->ਅੰਕੜੇ == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਝੂਠ;
    }

    std::ofstream ਬਾਹਰਲਾ(file_ਰਸਤਾ);
    ਜੇ (!ਬਾਹਰਲਾ.is_open()) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ size_t n = ggml_nelements(tensor);
    ਬਾਹਰਲਾ << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    ਚੋਣ (tensor->type) {
    ਹਾਲਤ GGML_TYPE_F32: {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(tensor->ਅੰਕੜੇ);
        ਲਈ (size_t i = 0; i < n; ++i) {
            ਬਾਹਰਲਾ << ਅੰਕੜੇ[i] << '\n';
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_TYPE_F16: {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<ਅਚਲ ggml_fp16_t *>(tensor->ਅੰਕੜੇ);
        ਲਈ (size_t i = 0; i < n; ++i) {
            ਬਾਹਰਲਾ << ggml_fp16_ਮੰਜ਼ਿਲ_fp32(ਅੰਕੜੇ[i]) << '\n';
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_TYPE_BF16: {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<ਅਚਲ ggml_bf16_t *>(tensor->ਅੰਕੜੇ);
        ਲਈ (size_t i = 0; i < n; ++i) {
            ਬਾਹਰਲਾ << ggml_bf16_ਮੰਜ਼ਿਲ_fp32(ਅੰਕੜੇ[i]) << '\n';
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_TYPE_I32: {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<ਅਚਲ int32_t *>(tensor->ਅੰਕੜੇ);
        ਲਈ (size_t i = 0; i < n; ++i) {
            ਬਾਹਰਲਾ << ਅੰਕੜੇ[i] << '\n';
        }
        ਰੋਕੋ;
    }
    ਹਾਲਤ GGML_TYPE_I64: {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * ਅੰਕੜੇ = static_cast<ਅਚਲ int64_t *>(tensor->ਅੰਕੜੇ);
        ਲਈ (size_t i = 0; i < n; ++i) {
            ਬਾਹਰਲਾ << ਅੰਕੜੇ[i] << '\n';
        }
        ਰੋਕੋ;
    }
    ਮੂਲ:
        ਬਾਹਰਲਾ << "unsupported tensor type for text dump" << '\n';
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਖਾਲੀ print_ਨਿਵੇਸ਼_tensor_info(ਅਚਲ std::string & name, ਅਚਲ ov::Tensor & tensor) {
    std::ਬਾਹਰ << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.ਅੰਕੜੇ()
              << '\n';
    ਚੋਣ (tensor.get_element_type()) {
    ਹਾਲਤ ov::element::f32: {
        ਜੇ (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::ਬਾਹਰ << *(tensor.ਅੰਕੜੇ<ਦਸ਼ਮਲਵ>()) << '\n';
        } ਨਹੀਂਤਾਂ {
            size_t ਪੰਕਤੀਆਂ = tensor.get_shape()[2];
            size_t ਖਾਨੇ_ਗਿਣਤੀ_2 = tensor.get_shape()[3];
            ਅਚਲ ਦਸ਼ਮਲਵ * ਅੰਕੜੇ = tensor.ਅੰਕੜੇ<ਦਸ਼ਮਲਵ>();
            ਲਈ (size_t i = 0; i < ਪੰਕਤੀਆਂ; ++i) {
                ਲਈ (size_t j = 0; j < ਖਾਨੇ_ਗਿਣਤੀ_2; ++j) {
                    ਦਸ਼ਮਲਵ val = ਅੰਕੜੇ[i * ਖਾਨੇ_ਗਿਣਤੀ_2 + j];
                    ਜੇ (std::isinf(val) && val < 0) {
                        std::ਬਾਹਰ << std::setw(5) << "-inf";
                    } ਨਹੀਂਤਾਂ {
                        std::ਬਾਹਰ << std::setw(5) << val;
                    }
                }
                std::ਬਾਹਰ << '\n';
            }
        }

        ਰੋਕੋ;
    }
    ਹਾਲਤ ov::element::f16:
        std::ਬਾਹਰ << *(tensor.ਅੰਕੜੇ<ov::float16>()) << '\n';
        ਰੋਕੋ;
    ਹਾਲਤ ov::element::i32:
        ਲਈ (size_t i = 0; i < tensor.get_ਮਾਪ(); ++i) {
            std::ਬਾਹਰ << tensor.ਅੰਕੜੇ<int32_t>()[i] << ' ';
        }
        std::ਬਾਹਰ << '\n';
        ਰੋਕੋ;
    ਹਾਲਤ ov::element::i64:
        ਲਈ (size_t i = 0; i < tensor.get_ਮਾਪ(); ++i) {
            std::ਬਾਹਰ << tensor.ਅੰਕੜੇ<int64_t>()[i] << ' ';
        }
        std::ਬਾਹਰ << '\n';
        ਰੋਕੋ;
    ਮੂਲ:
        ਰੋਕੋ;
    }
}

ਖਾਲੀ print_output_tensor_info(ਅਚਲ std::string & name, ਅਚਲ ov::Tensor & tensor, ਅਚਲ ਖਾਲੀ * output_dst) {
    std::ਬਾਹਰ << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    ਸਵੈਚਾਲਿਤ print_float_stats = [](ਅਚਲ std::string & type_name, size_t size, ਸਵੈਚਾਲਿਤ get_ਮੁੱਲ) {
        ਜੇ (size == 0) {
            ਵਾਪਸ;
        }

        ਦਸ਼ਮਲਵ ਪਹਿਲਾ = get_ਮੁੱਲ(0);
        ਦਸ਼ਮਲਵ min = ਪਹਿਲਾ;
        ਦਸ਼ਮਲਵ max = ਪਹਿਲਾ;
        ਦੁੱਗਣਾ ਜੋੜ = ਪਹਿਲਾ;

        ਲਈ (size_t i = 1; i < size; ++i) {
            ਦਸ਼ਮਲਵ v = get_ਮੁੱਲ(i);
            min = std::min(v, min);
            max = std::max(v, max);
            ਜੋੜ += v;
        }
        ਦੁੱਗਣਾ ਔਸਤ_2 = ਜੋੜ / size;

        std::ਬਾਹਰ << std::ਸੱਜੇ << std::setw(6) << type_name << std::ਸੱਜੇ << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::ਬਾਹਰ << std::ਸੱਜੇ << std::setw(6) << "" << std::ਸੱਜੇ << std::setw(12) << ਪਹਿਲਾ << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << ਔਸਤ_2 << '\n';
    };

    ਚੋਣ (tensor.get_element_type()) {
    ਹਾਲਤ ov::element::f32: {
        ਅਚਲ ਦਸ਼ਮਲਵ * ਅੰਕੜੇ = tensor.ਅੰਕੜੇ<ਦਸ਼ਮਲਵ>();
        size_t size = tensor.get_ਮਾਪ();
        print_float_stats("[f32]", size, [ਅੰਕੜੇ](size_t i) { ਵਾਪਸ ਅੰਕੜੇ[i]; });
        ਰੋਕੋ;
    }
    ਹਾਲਤ ov::element::f16: {
        ਅਚਲ ov::float16 * ਅੰਕੜੇ = tensor.ਅੰਕੜੇ<ov::float16>();
        size_t size = tensor.get_ਮਾਪ();
        print_float_stats("[f16]", size, [ਅੰਕੜੇ](size_t i) { ਵਾਪਸ static_cast<ਦਸ਼ਮਲਵ>(ਅੰਕੜੇ[i]); });
        ਰੋਕੋ;
    }
    ਮੂਲ:
        ਰੋਕੋ;
    }
}

ਅਚਲ ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; ++i) {
        ਸਵੈਚਾਲਿਤ * op = cgraph->nodes[i];
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; ++j) {
            ਸਵੈਚਾਲਿਤ * src = op->src[j];
            ਜੇ (src == ਖਾਲੀ_ਸੂਚਕ) {
                ਰੋਕੋ;
            }
            ਜੇ (GgmlOvDecoder::is_inp_pos(src, op)) {
                ਵਾਪਸ src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    ਸੁੱਟੋ std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_ਟੋਕਨ(ggml_cgraph * cgraph, ਅਚਲ ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    ਪੂਰਨ n_planes = 1;
    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; ++i) {
        ਸਵੈਚਾਲਿਤ * op = cgraph->nodes[i];
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; ++j) {
            ਜੇ (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                ਰੋਕੋ;
            }
        }
    }
    ਵਾਪਸ inp_pos->ne[0] / n_planes;
}

ਤਾਰਕਿਕ get_is_prefill(ggml_cgraph * cgraph, ਅਚਲ ggml_tensor * inp_pos) {
    ਵਾਪਸ get_inp_pos_n_ਟੋਕਨ(cgraph, inp_pos) > 1;
}
