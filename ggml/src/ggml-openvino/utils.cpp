#അന്തര്ഭാവ "utils.h"

#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-openvino-extra.h"
#അന്തര്ഭാവ "ggml-openvino.h"
#അന്തര്ഭാവ "ggml-openvino/ggml-decoder.h"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "model-cache.h"
#അന്തര്ഭാവ "openvino/frontend.h"
#അന്തര്ഭാവ "openvino/input_model.h"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cassert>
#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <fstream>
#അന്തര്ഭാവ <functional>
#അന്തര്ഭാവ <iomanip>
#അന്തര്ഭാവ <iostream>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/core/any.hpp>
#അന്തര്ഭാവ <openvino/core/graph_util.hpp>
#അന്തര്ഭാവ <openvino/core/shape.hpp>
#അന്തര്ഭാവ <openvino/core/type/float16.hpp>
#അന്തര്ഭാവ <openvino/frontend/manager.hpp>
#അന്തര്ഭാവ <openvino/openvino.hpp>
#അന്തര്ഭാവ <openvino/runtime/compiled_model.hpp>
#അന്തര്ഭാവ <openvino/runtime/infer_request.hpp>
#അന്തര്ഭാവ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#അന്തര്ഭാവ <openvino/runtime/intel_npu/properties.hpp>
#അന്തര്ഭാവ <openvino/runtime/properties.hpp>
#അന്തര്ഭാവ <openvino/runtime/tensor.hpp>
#അന്തര്ഭാവ <optional>
#അന്തര്ഭാവ <set>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <vector>

നാമാകാശ {
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
std::optional<ov::Tensor> try_make_kv_sliced_tensor(സ്ഥിരം std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    സ്ഥിരം std::string & name,
                                                    സ്ഥിരം ggml_tensor * ggml_tensor) {
    സ്ഥിതം സ്ഥിരം തർക്കം kv_slice_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    എങ്കിൽ (kv_slice_disabled) {
        തിരികെ std::nullopt;
    }
    എങ്കിൽ (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        തിരികെ std::nullopt;
    }
    എങ്കിൽ (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != nullptr) {
        തിരികെ std::nullopt;
    }
    സ്ഥിരം auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);
    എങ്കിൽ (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        തിരികെ std::nullopt;
    }

    സ്ഥിരം auto & compute_params = ggml_decoder->get_compute_params();
    എങ്കിൽ (compute_params.n_seq_active != 1 || compute_params.seq_active_start != 0) {
        തിരികെ std::nullopt;
    }

    പൂർണ്ണസംഖ്യ layer;
    എങ്കിൽ (auto layer_opt = extract_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } അല്ലെങ്കിൽ {
        തിരികെ std::nullopt;
    }

    സ്ഥിരം തർക്കം is_swa = ggml_decoder->is_swa_layer(layer);
    എങ്കിൽ (is_swa) {
        തിരികെ std::nullopt;
    }
    സ്ഥിരം പൂർണ്ണസംഖ്യ ctx_per_seq = ggml_decoder->get_ctx_per_seq();
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_kv = compute_params.attention_size;
    എങ്കിൽ (ctx_per_seq <= 0 || n_kv <= 0 || n_kv >= ctx_per_seq) {
        തിരികെ std::nullopt;
    }

    ov::Shape full_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    എങ്കിൽ (full_shape.size() != 4 || full_shape[0] != 1 || full_shape[1] != 1 ||
        static_cast<പൂർണ്ണസംഖ്യ>(full_shape[2]) != ctx_per_seq) {
        തിരികെ std::nullopt;
    }

    ov::Shape sliced_shape = full_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    എങ്കിൽ (ggml_openvino_buffer_is_remote(ggml_tensor)) {
        auto remote_context = ggml_openvino_get_remote_context();
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        തിരികെ gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
    }

    തിരികെ ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
}

സ്ഥിതം uint64_t ggml_openvino_model_cache_extra_cfg(തർക്കം stateful, തർക്കം recurrent) {
    സ്ഥിരം അക്ഷരം * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    സ്ഥിരം തർക്കം manual_gqa_enabled = manual_gqa_env != nullptr ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    തിരികെ extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_weight_names(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> names;
    വേണ്ടി (സ്ഥിരം auto & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
        names[name] = nullptr;
    }
    തിരികെ names;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
സ്ഥിതം std::string compiled_graph_താക്കോൽ(സ്ഥിരം ggml_cgraph * graph,
                                      സ്ഥിരം GgmlOvDecoder & decoder,
                                      സ്ഥിരം std::string & device,
                                      പൂർണ്ണസംഖ്യ prefill_chunk_size = 0,
                                      തർക്കം disk_cache = അസത്യം) {
    std::string താക്കോൽ;
    auto append = [&താക്കോൽ](സ്ഥിരം auto & value) {
        താക്കോൽ.append(reinterpret_cast<സ്ഥിരം അക്ഷരം *>(&value), വലുപ്പം(value));
    };
    auto append_string = [&](സ്ഥിരം std::string & value) {
        append(value.size());
        താക്കോൽ.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_size);
    തർക്കം has_weight_buffer_id = അസത്യം;
    std::unordered_map<സ്ഥിരം ggml_tensor *, size_t> ids;
    std::function<ശൂന്യം(സ്ഥിരം ggml_tensor *)> visit = [&](സ്ഥിരം ggml_tensor * tensor) {
        എങ്കിൽ (!tensor) {
            append(size_t(0));
            തിരികെ;
        }
        auto inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.first->second);
        എങ്കിൽ (!inserted.second) {
            തിരികെ;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        സ്ഥിരം auto * base = tensor->view_src ? tensor->view_src : tensor;
        സ്ഥിരം തർക്കം weight = base->buffer && base->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        append(weight);
        എങ്കിൽ (weight && !disk_cache) {
            സ്ഥിരം size_t buffer_id = ggml_backend_openvino_buffer_get_ctx_id(base->buffer);
            has_weight_buffer_id |= buffer_id != 0;
            append(buffer_id);
            append(tensor->data);
        }
        visit(tensor->view_src);
        വേണ്ടി (സ്ഥിരം auto * src : tensor->src) {
            visit(src);
        }
    };
    append(graph->n_nodes);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; ++i) {
        visit(graph->nodes[i]);
    }
    append(graph->n_leafs);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_leafs; ++i) {
        visit(graph->leafs[i]);
    }
    വേണ്ടി (സ്ഥിരം auto & നിവേശം : decoder.get_model_extra_inputs()) {
        append_string(നിവേശം.first);
        append_string(നിവേശം.second.type.get_type_name());
        append(നിവേശം.second.shape.size());
        വേണ്ടി (auto dim : നിവേശം.second.shape) {
            append(dim);
        }
        append(നിവേശം.second.is_parameter);
        എങ്കിൽ (!നിവേശം.second.is_parameter) {
            append(നിവേശം.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    തിരികെ disk_cache || has_weight_buffer_id ? താക്കോൽ : std::string{};
}

സ്ഥിതം std::string dynamic_graph_signature(സ്ഥിരം ggml_cgraph * graph,
                                           സ്ഥിരം GgmlOvDecoder & decoder,
                                           സ്ഥിരം ModelParams & params) {
    std::string താക്കോൽ = "dynamic-1";
    auto append = [&താക്കോൽ](സ്ഥിരം auto & value) {
        താക്കോൽ.append(reinterpret_cast<സ്ഥിരം അക്ഷരം *>(&value), വലുപ്പം(value));
    };
    auto append_string = [&](സ്ഥിരം std::string & value) {
        append(value.size());
        താക്കോൽ.append(value);
    };
    graph_താക്കോൽ graph_id(graph, സത്യം);
    append(graph_id.n_nodes);
    append(graph_id.n_leaves);
    append_string(graph_id.first_കണ്ണി_name);
    append_string(graph_id.last_കണ്ണി_name);
    വേണ്ടി (സ്ഥിരം auto & name : graph_id.നിവേശം_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    വേണ്ടി (പൂർണ്ണസംഖ്യ layer : params.swa_layers) {
        append(layer);
    }
    വേണ്ടി (സ്ഥിരം auto & [layer, heads] : params.n_heads_kv_per_layer) {
        append(layer);
        append(heads);
    }
    വേണ്ടി (സ്ഥിരം auto & [name, നിവേശം] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(നിവേശം.type.get_type_name());
    }
    വേണ്ടി (സ്ഥിരം auto & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    വേണ്ടി (സ്ഥിരം auto & [name, നിവേശം] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(നിവേശം.type.get_type_name());
        append(നിവേശം.is_parameter);
        എങ്കിൽ (!നിവേശം.is_parameter) {
            append(നിവേശം.value);
        }
    }
    തിരികെ താക്കോൽ;
}

സ്ഥിതം തർക്കം compiled_model_matches_graph(സ്ഥിരം ov::CompiledModel & model,
                                         സ്ഥിരം std::vector<std::string> & inputs,
                                         സ്ഥിരം std::vector<std::string> & outputs,
                                         സ്ഥിരം GgmlOvDecoder & decoder) {
    എങ്കിൽ (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം auto & graph_inputs = decoder.get_model_inputs();
    സ്ഥിരം auto & extra_inputs = decoder.get_model_extra_inputs();
    സ്ഥിരം auto & graph_outputs = decoder.get_model_outputs();
    വേണ്ടി (size_t i = 0; i < inputs.size(); ++i) {
        സ്ഥിരം auto & port = model.നിവേശം(i);
        auto graph_it = graph_inputs.find(inputs[i]);
        എങ്കിൽ (graph_it != graph_inputs.end()) {
            എങ്കിൽ (port.get_element_type() != graph_it->second.type ||
                !port.get_partial_shape().compatible(graph_it->second.shape)) {
                തിരികെ അസത്യം;
            }
            തുടരുക;
        }
        auto extra_it = extra_inputs.find(inputs[i]);
        എങ്കിൽ (extra_it == extra_inputs.end() || port.get_element_type() != extra_it->second.type ||
            !port.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            തിരികെ അസത്യം;
        }
    }
    വേണ്ടി (size_t i = 0; i < outputs.size(); ++i) {
        auto graph_it = graph_outputs.find(outputs[i]);
        എങ്കിൽ (graph_it == graph_outputs.end()) {
            എങ്കിൽ (outputs[i].compare(0, 8, "__debug_") != 0) {
                തിരികെ അസത്യം;
            }
            തുടരുക;
        }
        സ്ഥിരം auto & port = model.output(i);
        എങ്കിൽ (port.get_element_type() != GgmlOvDecoder::get_ov_type(graph_it->second) ||
            !port.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(graph_it->second)))) {
            തിരികെ അസത്യം;
        }
    }
    തിരികെ സത്യം;
}

സ്ഥിതം std::string ov_profiling_csv_field(സ്ഥിരം std::string & value) {
    std::string escaped = "\"";
    വേണ്ടി (അക്ഷരം c : value) {
        escaped += c;
        എങ്കിൽ (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    തിരികെ escaped;
}

സ്ഥിതം ശൂന്യം dump_ov_profiling_info(സ്ഥിരം ov::InferRequest & infer_request) {
    എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        തിരികെ;
    }

    സ്ഥിതം std::atomic<uint64_t> inference_index{0};
    സ്ഥിരം uint64_t index = inference_index.fetch_add(1);
    സ്ഥിരം std::string path = "openvino_profile_" + std::to_string(index) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    എങ്കിൽ (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        തിരികെ;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t total_real_time = 0;
    int64_t total_cpu_time = 0;
    size_t executed_എണ്ണം = 0;
    വേണ്ടി (സ്ഥിരം auto & info : infer_request.get_profiling_info()) {
        സ്ഥിരം അക്ഷരം * status = "NOT_RUN";
        എങ്കിൽ (info.status == ov::ProfilingInfo::Status::EXECUTED) {
            status = "EXECUTED";
            total_real_time += info.real_time.count();
            total_cpu_time += info.cpu_time.count();
            executed_എണ്ണം++;
        } അല്ലെങ്കിൽ എങ്കിൽ (info.status == ov::ProfilingInfo::Status::OPTIMIZED_OUT) {
            status = "OPTIMIZED_OUT";
        }

        output << status << ',' << info.real_time.count() << ',' << info.cpu_time.count() << ','
               << info.start_time.count() << ',' << ov_profiling_csv_field(info.node_type) << ','
               << ov_profiling_csv_field(info.കണ്ണി_name) << ',' << ov_profiling_csv_field(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_എണ്ണം, total_real_time / 1000.0, total_cpu_time / 1000.0);
}

സ്ഥിതം ശൂന്യം reset_single_slot_recurrent_cache(ggml_cgraph * cgraph,
                                              സ്ഥിരം ModelParams & model_params,
                                              സ്ഥിരം ComputeParams & compute_params) {
    എങ്കിൽ (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        തിരികെ;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_buffer_t> buffers;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        auto * കണ്ണി = cgraph->nodes[i];
        എങ്കിൽ (കണ്ണി->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(കണ്ണി->view_src) &&
            കണ്ണി->view_src->ne[1] == 1 && കണ്ണി->view_src->buffer != nullptr) {
            buffers.insert(കണ്ണി->view_src->buffer);
        }
    }
    വേണ്ടി (auto * buffer : buffers) {
        ggml_backend_buffer_clear(buffer, 0);
    }
}

ov::Tensor create_ov_output_tensor(സ്ഥിരം std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   സ്ഥിരം std::shared_ptr<ov::InferRequest> & infer_request,
                                   പൂർണ്ണസംഖ്യ output_index,
                                   സ്ഥിരം ggml_tensor * ggml_tensor) {
    എങ്കിൽ (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        തിരികെ *sliced;
    }

    auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    ശൂന്യം * output_data = ggml_tensor->data;
    എങ്കിൽ (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_index).get_shape();
    } അല്ലെങ്കിൽ {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        എങ്കിൽ (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != nullptr &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_data = ggml_tensor->view_src->data;
        } അല്ലെങ്കിൽ {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    എങ്കിൽ (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_buffer_is_remote(ggml_tensor) &&
        ggml_tensor->extra != nullptr) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        എങ്കിൽ (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            എങ്കിൽ (tensor_extra->tensor != nullptr && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                തിരികെ *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_data);
    തിരികെ output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_rows_to_seq_axis_2(സ്ഥിരം ov::Tensor & kv_tensor, size_t n_heads_kv) {
    സ്ഥിരം size_t rows = kv_tensor.get_shape()[2];
    സ്ഥിരം size_t head_size = kv_tensor.get_shape()[3] / n_heads_kv;
    സ്ഥിരം size_t elem = kv_tensor.get_element_type().size();
    സ്ഥിരം size_t head_bytes = head_size * elem;

    ov::Tensor out(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, rows, head_size});
    സ്ഥിരം auto * src = static_cast<സ്ഥിരം uint8_t *>(kv_tensor.data());
    auto * dst = static_cast<uint8_t *>(out.data());
    വേണ്ടി (size_t s = 0; s < rows; s++) {
        വേണ്ടി (size_t h = 0; h < n_heads_kv; h++) {
            memcpy(dst + (h * rows + s) * head_bytes, src + (s * n_heads_kv + h) * head_bytes, head_bytes);
        }
    }
    തിരികെ out;
}

template <typename T> ശൂന്യം set_zero_diagonal(std::vector<T> & matrix, size_t rows, size_t cols, T zero_മൂല്യം = T{}) {
    വേണ്ടി (size_t i = 0; i < rows; ++i) {
        size_t diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = zero_മൂല്യം;
    }
}

ov::Tensor make_contiguous_split_നിവേശം_tensor(സ്ഥിരം ഘടന ggml_tensor * ggml_tensor, സ്ഥിരം ov::Shape & നിവേശം_shape) {
    സ്ഥിരം size_t element_size = ggml_type_size(ggml_tensor->type);
    സ്ഥിരം size_t block_size = ggml_blck_size(ggml_tensor->type);

    GGML_ASSERT(block_size == 1 && "non-contiguous split inputs must be plain element types");

    സ്ഥിരം ഘടന ggml_tensor * source_tensor = ggml_tensor->view_src != nullptr ? ggml_tensor->view_src : ggml_tensor;
    സ്ഥിരം size_t source_offset = ggml_tensor->view_src != nullptr ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> source_data(ggml_nbytes(source_tensor));
    ggml_backend_tensor_get(source_tensor, source_data.data(), 0, source_data.size());

    ov::Tensor നിവേശം_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), നിവേശം_shape);
    auto * dst = static_cast<uint8_t *>(നിവേശം_tensor.data());
    size_t dst_offset = 0;

    വേണ്ടി (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        വേണ്ടി (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            വേണ്ടി (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                വേണ്ടി (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    സ്ഥിരം size_t src_offset = source_offset + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::memcpy(dst + dst_offset, source_data.data() + src_offset, element_size);
                    dst_offset += element_size;
                }
            }
        }
    }

    തിരികെ നിവേശം_tensor;
}

ov::Tensor convert_ggml_നിവേശം_to_ov(സ്ഥിരം std::shared_ptr<GgmlOvDecoder> & ggml_decoder, സ്ഥിരം std::string & name) {
    സ്ഥിരം auto * ggml_tensor = ggml_decoder->get_നിവേശം_ggml_tensor(name);

    എങ്കിൽ (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        തിരികെ *sliced;
    }

    എങ്കിൽ (ggml_tensor->extra != nullptr && !ggml_decoder->is_splited_model()) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        എങ്കിൽ (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            തിരികെ *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    auto * നിവേശം_data = ggml_tensor->data;
    ov::Shape നിവേശം_shape;
    എങ്കിൽ (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        നിവേശം_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } അല്ലെങ്കിൽ {
        നിവേശം_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    എങ്കിൽ (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        തിരികെ make_contiguous_split_നിവേശം_tensor(ggml_tensor, നിവേശം_shape);
    }

    auto നിവേശം_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), നിവേശം_shape, നിവേശം_data);
    തിരികെ നിവേശം_tensor;
}

ov::Tensor get_ov_നിവേശം_tensor(സ്ഥിരം std::shared_ptr<GgmlOvDecoder> & ggml_decoder, സ്ഥിരം std::string & param_name) {
    ov::Tensor നിവേശം_tensor;
    auto extra_നിവേശം = ggml_decoder->get_model_extra_inputs().find(param_name);
    എങ്കിൽ (extra_നിവേശം != ggml_decoder->get_model_extra_inputs().end()) {
        നിവേശം_tensor = ov::Tensor(extra_നിവേശം->second.type, extra_നിവേശം->second.shape);
        *നിവേശം_tensor.data<int64_t>() = extra_നിവേശം->second.value;
    } അല്ലെങ്കിൽ {
        നിവേശം_tensor = convert_ggml_നിവേശം_to_ov(ggml_decoder, param_name);
    }
    തിരികെ നിവേശം_tensor;
}

ov::Tensor get_ov_നിവേശം_tensor_static_decode(സ്ഥിരം std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             സ്ഥിരം std::string & param_name) {
    // NPU decoding stage
    എങ്കിൽ (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        തിരികെ get_ov_നിവേശം_tensor(ggml_decoder, param_name);
    }
    സ്ഥിരം auto * ggml_tensor = ggml_decoder->get_നിവേശം_ggml_tensor(param_name);
    സ്ഥിരം auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    എങ്കിൽ (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        സ്ഥിരം പൂർണ്ണസംഖ്യ n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        പ്രതിജ്ഞാ(ggml_tensor->ne[0] == n_planes);
        ov::Shape നിവേശം_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor നിവേശം_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), നിവേശം_shape);
        std::memcpy(നിവേശം_tensor.data(), ggml_tensor->data, n_planes * ggml_type_size(ggml_tensor->type));
        തിരികെ നിവേശം_tensor;
    }

    എങ്കിൽ (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape നിവേശം_shape = {1, 1, 1, 1};
        ov::Tensor നിവേശം_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), നിവേശം_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        പ്രതിജ്ഞാ(ggml_tensor->ne[0] == 1);
        പ്രതിജ്ഞാ(inp_out_id == 0);
        *നിവേശം_tensor.data<int32_t>() = inp_out_id;
        തിരികെ നിവേശം_tensor;
    }

    എങ്കിൽ (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t context_size = ggml_decoder->get_ctx_size();
        എങ്കിൽ (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_data =
                pad_നിവേശം<ggml_fp16_t>(ggml_tensor, 1, context_size, GGML_FP32_TO_FP16(-INFINITY));
            ov::Tensor നിവേശം_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_size});
            std::memcpy(നിവേശം_tensor.data(), padded_data.data(), padded_data.size() * വലുപ്പം(ggml_fp16_t));
            തിരികെ നിവേശം_tensor;
        }

        std::vector<ദശാംശം> padded_data = pad_നിവേശം<ദശാംശം>(ggml_tensor, 1, context_size, -INFINITY);
        ov::Tensor നിവേശം_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        auto * data_ptr = നിവേശം_tensor.data<ദശാംശം>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
        തിരികെ നിവേശം_tensor;
    }

    തിരികെ get_ov_നിവേശം_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_നിവേശം_tensor_static_prefill(സ്ഥിരം std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              സ്ഥിരം std::string & param_name,
                                              പൂർണ്ണസംഖ്യ chunk_index) {
    // NPU prompt processing stage
    സ്ഥിരം size_t നിവേശം_len = ggml_decoder->get_നിവേശം_len();
    സ്ഥിരം size_t chunk_size = ggml_decoder->m_prefill_chunk_size;
    സ്ഥിരം size_t chunk_valid_size = std::min(chunk_size, നിവേശം_len - chunk_index * chunk_size);
    സ്ഥിരം size_t chunk_pad_size = chunk_size - chunk_valid_size;

    എങ്കിൽ (param_name == "chunk_valid_len") {
        ov::Tensor നിവേശം_tensor(ov::element::i64, ov::Shape{1});
        *നിവേശം_tensor.data<int64_t>() = (int64_t) chunk_valid_size;
        തിരികെ നിവേശം_tensor;
    }
    എങ്കിൽ (chunk_index > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor നിവേശം_tensor(ov::element::i64, ov::Shape{1});
        *നിവേശം_tensor.data<int64_t>() = 0;
        തിരികെ നിവേശം_tensor;
    }
    എങ്കിൽ (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        തിരികെ get_ov_നിവേശം_tensor(ggml_decoder, param_name);
    }
    സ്ഥിരം auto * ggml_tensor = ggml_decoder->get_നിവേശം_ggml_tensor(param_name);
    സ്ഥിരം auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    എങ്കിൽ (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        സ്ഥിരം പൂർണ്ണസംഖ്യ n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        സ്ഥിരം size_t element_size = ggml_type_size(ggml_tensor->type);
        ov::Shape നിവേശം_shape = {1, 1, 1, (size_t) n_planes * chunk_size};
        ov::Tensor നിവേശം_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), നിവേശം_shape);
        വേണ്ടി (പൂർണ്ണസംഖ്യ p = 0; p < n_planes; p++) {
            സ്ഥിരം അക്ഷരം * src =
                (സ്ഥിരം അക്ഷരം *) ggml_tensor->data + (p * നിവേശം_len + chunk_index * chunk_size) * element_size;
            അക്ഷരം * dst = (അക്ഷരം *) നിവേശം_tensor.data() + p * chunk_size * element_size;
            std::memcpy(dst, src, chunk_valid_size * element_size);
            എങ്കിൽ (chunk_pad_size > 0) {
                എങ്കിൽ (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t last_മൂല്യം = *((സ്ഥിരം int32_t *) src + chunk_valid_size - 1);
                    int32_t * out = (int32_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_മൂല്യം + 1);
                } അല്ലെങ്കിൽ എങ്കിൽ (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t last_മൂല്യം = *((സ്ഥിരം int64_t *) src + chunk_valid_size - 1);
                    int64_t * out = (int64_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_മൂല്യം + 1);
                } അല്ലെങ്കിൽ {
                    എറിയുക std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        തിരികെ നിവേശം_tensor;
    }

    എങ്കിൽ (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape നിവേശം_shape = {1, 1, 1, chunk_size};
        ov::Tensor നിവേശം_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), നിവേശം_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        ശൂന്യം * നിവേശം_data = (അക്ഷരം *) ggml_tensor->data + chunk_index * chunk_size * element_size;
        std::memcpy(നിവേശം_tensor.data(), നിവേശം_data, chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        എങ്കിൽ (chunk_pad_size > 0) {
            എങ്കിൽ (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_മൂല്യം =
                    *((int32_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int32_t * output_data = നിവേശം_tensor.data<int32_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_മൂല്യം + 1);
            } അല്ലെങ്കിൽ എങ്കിൽ (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_മൂല്യം =
                    *((int64_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int64_t * output_data = നിവേശം_tensor.data<int64_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_മൂല്യം + 1);
            } അല്ലെങ്കിൽ {
                എറിയുക std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        തിരികെ നിവേശം_tensor;
    }

    എങ്കിൽ (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape നിവേശം_shape = {1, 1, 1, output_len};
        ov::Tensor നിവേശം_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), നിവേശം_shape);
        എങ്കിൽ (ggml_tensor->ne[0] == 0) {
            *നിവേശം_tensor.data<int32_t>() = 0;
        } അല്ലെങ്കിൽ {
            auto * data_addr = നിവേശം_tensor.data<int32_t>();
            വേണ്ടി (size_t i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        തിരികെ നിവേശം_tensor;
    }

    എങ്കിൽ (GgmlOvDecoder::is_inp_scale_rows(ggml_tensor, op)) {
        ov::Tensor നിവേശം_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, 1});
        auto * dst = നിവേശം_tensor.data<ദശാംശം>();
        സ്ഥിരം auto * src = static_cast<സ്ഥിരം ദശാംശം *>(ggml_tensor->data) + chunk_index * chunk_size;
        std::copy(src, src + chunk_valid_size, dst);
        std::fill(dst + chunk_valid_size, dst + chunk_size, 1.0f);
        തിരികെ നിവേശം_tensor;
    }

    എങ്കിൽ (GgmlOvDecoder::is_inp_mean(ggml_tensor, op)) {
        സ്ഥിരം size_t n_seqs = ggml_tensor->ne[1];
        സ്ഥിരം size_t src_stride = ggml_tensor->ne[0];
        സ്ഥിരം size_t copy_len = std::min<size_t>(chunk_valid_size, src_stride - chunk_index * chunk_size);
        ov::Tensor നിവേശം_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_size});
        auto * dst = നിവേശം_tensor.data<ദശാംശം>();
        std::fill(dst, dst + n_seqs * chunk_size, 0.0f);
        സ്ഥിരം auto * src = static_cast<സ്ഥിരം ദശാംശം *>(ggml_tensor->data) + chunk_index * chunk_size;
        വേണ്ടി (size_t s = 0; s < n_seqs; s++) {
            std::memcpy(dst + s * chunk_size, src + s * src_stride, copy_len * വലുപ്പം(ദശാംശം));
        }
        തിരികെ നിവേശം_tensor;
    }

    എങ്കിൽ (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = ggml_decoder->get_ctx_size();
        എങ്കിൽ (ggml_tensor->type == GGML_TYPE_F16) {
            സ്ഥിരം auto * ggml_data =
                static_cast<സ്ഥിരം ggml_fp16_t *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
            std::vector<ggml_fp16_t> padded_data = pad_നിവേശം<ggml_fp16_t>(ggml_data, chunk_valid_rows, cols, chunk_size,
                                                                          context_size, GGML_FP32_TO_FP16(-INFINITY));
            set_zero_diagonal(padded_data, chunk_size, context_size, GGML_FP32_TO_FP16(0.0f));
            ov::Tensor നിവേശം_tensor(ov::element::f16, ov::Shape{1, 1, chunk_size, context_size});
            std::memcpy(നിവേശം_tensor.data(), padded_data.data(), padded_data.size() * വലുപ്പം(ggml_fp16_t));
            തിരികെ നിവേശം_tensor;
        }

        സ്ഥിരം auto * ggml_data = static_cast<സ്ഥിരം ദശാംശം *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
        std::vector<ദശാംശം> padded_data =
            pad_നിവേശം<ദശാംശം>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor നിവേശം_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        auto * data_ptr = നിവേശം_tensor.data<ദശാംശം>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        തിരികെ നിവേശം_tensor;
    }

    തിരികെ get_ov_നിവേശം_tensor(ggml_decoder, param_name);
}

എണ്ണൽ ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               സ്ഥിരം std::string & device,
                               സ്ഥിരം ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    എങ്കിൽ (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        തിരികെ GGML_STATUS_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_lock(cache.mutex);
    തർക്കം naive = സത്യം;
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    auto നിവേശം_model = std::make_shared<ov::frontend::ggml::നിവേശം_Model>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(നിവേശം_model, naive);
    എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    എങ്കിൽ (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    } അല്ലെങ്കിൽ {
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    }
    എങ്കിൽ (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } അല്ലെങ്കിൽ {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> നിവേശം_names;
    std::vector<std::string> output_names;
    വേണ്ടി (സ്ഥിരം auto & param : model->get_parameters()) {
        നിവേശം_names.push_back(param->get_friendly_name());
    }
    വേണ്ടി (സ്ഥിരം auto & result : model->get_results()) {
        output_names.push_back(result->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    നിവേശം_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_lock.unlock();

    വേണ്ടി (size_t i = 0; i < നിവേശം_names.size(); i++) {
        സ്ഥിരം auto & param_name = നിവേശം_names[i];
        auto നിവേശം_tensor = get_ov_നിവേശം_tensor(decoder, param_name);
        infer_request->set_നിവേശം_tensor(i, നിവേശം_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    വേണ്ടി (size_t i = 0; i < output_names.size(); i++) {
        auto output_tensor = infer_request->get_output_tensor(i);
        സ്ഥിരം auto & model_outputs = decoder->get_model_outputs();
        auto model_output_it = model_outputs.find(output_names[i]);
        എങ്കിൽ (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_names[i], output_tensor, output_tensor.data());
            }
            തുടരുക;
        }
        auto * ggml_tensor = model_output_it->second;
        std::memcpy(ggml_tensor->data, output_tensor.data(), output_tensor.get_byte_size());
    }
    തിരികെ GGML_STATUS_SUCCESS;
}

എണ്ണൽ ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, സ്ഥിരം std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();
    സ്ഥിരം auto & config = ggml_openvino_get_compile_config();
    സ്ഥിരം auto & device = r_ctx->device;
    സ്ഥിതം auto is_static = അസത്യം;

    സ്ഥിതം സ്ഥിരം തർക്കം cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    auto start_time = ggml_time_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    സ്ഥിരം int64_t cache_താക്കോൽ_start_time = ggml_time_us();
    graph_താക്കോൽ താക്കോൽ(cgraph);
    സ്ഥിരം int64_t cache_താക്കോൽ_compute_time = ggml_time_us() - cache_താക്കോൽ_start_time;
    int64_t cache_lookup_time = 0;
    തർക്കം താക്കോൽ_seen = അസത്യം;
    എങ്കിൽ (!cache_disabled) {
        സ്ഥിരം int64_t cache_lookup_start_time = ggml_time_us();
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        താക്കോൽ_seen = r_ctx->decoder_cache.find(താക്കോൽ) != r_ctx->decoder_cache.end();
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;
    }

    സ്ഥിരം int64_t model_split_check_start_time = ggml_time_us();
    തർക്കം model_is_splitted = താക്കോൽ_seen ? അസത്യം : is_model_splitted(cgraph);
    സ്ഥിരം int64_t model_split_check_time = ggml_time_us() - model_split_check_start_time;

    എങ്കിൽ (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        തിരികെ GGML_STATUS_FAILED;
    }
    എങ്കിൽ (is_naive(cgraph)) {
        എങ്കിൽ (!model_is_splitted) {
            തിരികെ naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    സ്ഥിരം int64_t graph_param_compute_start_time = ggml_time_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    സ്ഥിരം int64_t graph_param_compute_time = ggml_time_us() - graph_param_compute_start_time;

    സ്ഥിരം തർക്കം cache_enabled = !model_is_splitted && !cache_disabled;
    സ്ഥിരം തർക്കം stateful_recurrent = r_ctx->stateful && m_params.state_size >= 0;
    സ്ഥിരം തർക്കം stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    എങ്കിൽ (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_active > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        തിരികെ GGML_STATUS_FAILED;
    }
    എങ്കിൽ (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        തിരികെ GGML_STATUS_FAILED;
    }
    തർക്കം cache_hit = അസത്യം;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        സ്ഥിരം int64_t cache_lookup_start_time = ggml_time_us();
        എങ്കിൽ (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            auto it = r_ctx->decoder_cache.find(താക്കോൽ);
            cache_hit = it != r_ctx->decoder_cache.end();
            എങ്കിൽ (cache_hit) {
                entry = it->second;
            } അല്ലെങ്കിൽ {
                r_ctx->clear_caches_locked();
                auto mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[താക്കോൽ] = entry;
            }
        } അല്ലെങ്കിൽ {
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_hit = അസത്യം;
        }
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;

        std::lock_guard<std::mutex> lock(*(entry->mutex));
        cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(താക്കോൽ) != 0;

        എങ്കിൽ (cache_hit) {
            ggml_decoder = entry->ptr;
            old_m_params = ggml_decoder->get_model_params();
            എങ്കിൽ (!ggml_decoder->is_splited_model()) {
                cache_hit = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_നിവേശം_names;
        std::vector<std::string> ov_output_names;

        എങ്കിൽ (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            എങ്കിൽ (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(താക്കോൽ);
                ov_നിവേശം_names = r_ctx->ov_നിവേശം_names_cache.at(താക്കോൽ);
                ov_output_names = r_ctx->ov_output_names_cache.at(താക്കോൽ);
            }

            എങ്കിൽ (stateful_kv_only) {
                സ്ഥിരം auto * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_data = (int32_t *) inp_pos->data;
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                എങ്കിൽ (pos_data[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_size = pos_shape[3];
                } അല്ലെങ്കിൽ എങ്കിൽ (r_ctx->stateful_kv_size == static_cast<size_t>(pos_data[0])) {
                    r_ctx->stateful_kv_size += pos_shape[3];
                } അല്ലെങ്കിൽ {
                    സ്ഥിരം size_t pos_begin = static_cast<size_t>(pos_data[0]);
                    സ്ഥിരം തർക്കം refill = pos_begin > r_ctx->stateful_kv_size;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    എങ്കിൽ (refill && !ggml_decoder->get_model_params().swa_layers.empty()) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ n_swa = ggml_decoder->get_compute_params().swa_window;
                        എങ്കിൽ (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_size, n_swa);
                            തിരികെ GGML_STATUS_FAILED;
                        }
                    }

                    സ്ഥിരം തർക്കം relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    auto states = infer_request->query_state();
                    വേണ്ടി (auto state : states) {
                        auto state_tensor = state.get_state();
                        auto state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        എങ്കിൽ (auto it = r_ctx->kv_state_നിവേശം_name_map.find(state.get_name());
                            it != r_ctx->kv_state_നിവേശം_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        പൂർണ്ണസംഖ്യ n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        എങ്കിൽ (auto layer = extract_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        സ്ഥിരം തർക്കം relayout_this_state = relayout_enabled;
                        സ്ഥിരം size_t seq_axis = relayout_this_state ? 2 : 1;
                        സ്ഥിരം size_t head_axis = seq_axis == 2 ? 1 : 2;

                        എങ്കിൽ (refill) {
                            എങ്കിൽ (state_name.empty()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                തിരികെ GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_നിവേശം_tensor(ggml_decoder, state_name);
                            എങ്കിൽ (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_rows_to_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } അല്ലെങ്കിൽ {
                                ov::Shape refill_shape(4);
                                refill_shape[0] = state_tensor_shape[0];
                                refill_shape[seq_axis] = kv_tensor.get_shape()[2];
                                refill_shape[head_axis] = state_tensor_shape[head_axis];
                                refill_shape[3] = state_tensor_shape[3];
                                kv_tensor.set_shape(refill_shape);
                                state_tensor = kv_tensor;
                            }
                            state_tensor_shape = state_tensor.get_shape();
                        }
                        // Only ever shrink to a prefix the source really has. Slicing past it used to
                        // surface as a bare ov::Exception from the ROI constructor.
                        എങ്കിൽ (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            തിരികെ GGML_STATUS_FAILED;
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end(state_tensor_shape.begin(), state_tensor_shape.end());
                        end[seq_axis] = pos_begin;
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.set_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_size = pos_begin + pos_shape[3];
                }
            }

            decoder_end_time = ggml_time_us();
            conversion_end_time = decoder_end_time;
            compile_end_time = decoder_end_time;
        } അല്ലെങ്കിൽ {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            auto shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
            auto weight_names = get_weight_names(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            സ്ഥിരം std::string model_cache_dir = ggml_openvino_model_cache_dir();
            സ്ഥിരം std::string exact_താക്കോൽ = cache_enabled ? compiled_graph_താക്കോൽ(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_path, manifest_path;
            എങ്കിൽ (!model_cache_dir.empty() && !model_is_splitted) {
                സ്ഥിരം uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/സത്യം, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_graph_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/സത്യം, m_params.rope_params, 16, extra_cfg,
                                                    compiled_graph_താക്കോൽ(cgraph, *ggml_decoder, device, 0, സത്യം));
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            std::string shared_താക്കോൽ =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_താക്കോൽ) : "";
            ov::CompiledModel shared_model;
            തർക്കം imported = അസത്യം;
            auto shared_it = shared_cache->graphs.find(shared_താക്കോൽ);
            എങ്കിൽ (!shared_താക്കോൽ.empty() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_graph(shared_it->second.decode, shared_it->second.നിവേശം_names,
                                                               shared_it->second.output_names, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_നിവേശം_names = shared_it->second.നിവേശം_names;
                ov_output_names = shared_it->second.output_names;
                imported = സത്യം;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } അല്ലെങ്കിൽ എങ്കിൽ (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_താക്കോൽ = exact_താക്കോൽ;
                shared_it = shared_cache->graphs.find(shared_താക്കോൽ);
                എങ്കിൽ (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_നിവേശം_names = shared_it->second.നിവേശം_names;
                    ov_output_names = shared_it->second.output_names;
                    imported = സത്യം;
                }
                model_fp = exact_fp;
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            എങ്കിൽ (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(താക്കോൽ);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            എങ്കിൽ (!model_cache_dir.empty()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
            }
            എങ്കിൽ (!imported && !model_cache_dir.empty() && !model_is_splitted) {
                സ്ഥിരം uint64_t preferred_fp = model_fp;
                തർക്കം dynamic_incompatible = preferred_fp == exact_fp;
                വേണ്ടി (പൂർണ്ണസംഖ്യ candidate = 0; candidate < 2; ++candidate) {
                    എങ്കിൽ (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        നിർത്തുക;
                    }
                    സ്ഥിരം uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    സ്ഥിരം std::string candidate_blob =
                        ggml_openvino_model_cache_blob_path(model_cache_dir, candidate_fp);
                    സ്ഥിരം std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_path(model_cache_dir, candidate_fp);
                    std::ifstream blob_in(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    എങ്കിൽ (!blob_in.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        തുടരുക;
                    }
                    സ്ഥിരം int64_t import_start = ggml_time_us();
                    ശ്രമിക്കുക {
                        ov::CompiledModel cm;
                        auto remote_context = ggml_openvino_get_remote_context();
                        എങ്കിൽ (remote_context.has_value()) {
                            cm = core.import_model(blob_in, remote_context.value(), mc_config);
                        } അല്ലെങ്കിൽ {
                            cm = core.import_model(blob_in, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        എങ്കിൽ (!saved_inputs.empty() || !saved_outputs.empty()) {
                            ov_നിവേശം_names = std::move(saved_inputs);
                            ov_output_names = std::move(saved_outputs);
                        } അല്ലെങ്കിൽ {
                            വേണ്ടി (സ്ഥിരം auto & p : cm.inputs()) {
                                ov_നിവേശം_names.push_back(p.get_കണ്ണി()->get_friendly_name());
                            }
                            വേണ്ടി (സ്ഥിരം auto & o : cm.outputs()) {
                                ov_output_names.push_back(o.get_കണ്ണി()->get_friendly_name());
                            }
                        }
                        എങ്കിൽ (!compiled_model_matches_graph(cm, ov_നിവേശം_names, ov_output_names, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_നിവേശം_names.clear();
                            ov_output_names.clear();
                            തുടരുക;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        entry->ptr = ggml_decoder;
                        imported = സത്യം;
                        model_fp = candidate_fp;
                        blob_path = candidate_blob;
                        manifest_path = candidate_manifest;
                        എങ്കിൽ (candidate_fp == exact_fp) {
                            shared_താക്കോൽ = exact_താക്കോൽ;
                            shared_it = shared_cache->graphs.find(shared_താക്കോൽ);
                        }
                        എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_time_us() - import_start) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } പിടിക്കുക (സ്ഥിരം std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = അസത്യം;
                        ov_നിവേശം_names.clear();
                        ov_output_names.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                എങ്കിൽ (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_താക്കോൽ = exact_താക്കോൽ;
                    shared_it = shared_cache->graphs.find(shared_താക്കോൽ);
                    blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                    manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            എങ്കിൽ (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_path.c_str());
                തിരികെ GGML_STATUS_FAILED;
            }
            എങ്കിൽ (!imported && ggml_openvino_weight_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                തിരികെ GGML_STATUS_FAILED;
            }
            എങ്കിൽ (imported) {
                decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            } അല്ലെങ്കിൽ {
                auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_end_time = ggml_time_us();

                auto നിവേശം_model = std::make_shared<ov::frontend::ggml::നിവേശം_Model>(ggml_decoder);
                model = ov::frontend::ggml::FrontEnd::convert(നിവേശം_model);
                ggml_decoder->clear_model_weights();
                conversion_end_time = ggml_time_us();

                എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    അക്ഷരം timestamped_filename[64];
                    auto timestamp = (long long) ggml_time_us();
                    snprintf(timestamped_filename, വലുപ്പം(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                സ്ഥിരം ov::AnyMap & compile_config = model_cache_dir.empty() ? config : mc_config;
                ov::CompiledModel compiled_model;
                auto remote_context = ggml_openvino_get_remote_context();
                എങ്കിൽ (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } അല്ലെങ്കിൽ {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_end_time = ggml_time_us();

                വേണ്ടി (സ്ഥിരം auto & ov_param : model->get_parameters()) {
                    ov_നിവേശം_names.push_back(ov_param->get_friendly_name());
                }
                വേണ്ടി (സ്ഥിരം auto & ov_output : model->get_results()) {
                    ov_output_names.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                എങ്കിൽ (!model_cache_dir.empty() && !model_is_splitted && model_fp != 0) {
                    ശ്രമിക്കുക {
                        സ്ഥിരം std::string blob_tmp = ggml_openvino_model_cache_temp_path(blob_path);
                        സ്ഥിരം std::string manifest_tmp = ggml_openvino_model_cache_temp_path(manifest_path);
                        എങ്കിൽ (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_നിവേശം_names,
                                                                     ov_output_names)) {
                            std::ofstream blob_out(blob_tmp, std::ios::binary | std::ios::trunc);
                            എങ്കിൽ (blob_out.is_open()) {
                                compiled_model.export_model(blob_out);
                                blob_out.close();
                                എങ്കിൽ (blob_out.good()) {
                                    എങ്കിൽ (std::rename(blob_tmp.c_str(), blob_path.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_path.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_path.c_str());
                                    } അല്ലെങ്കിൽ {
                                        std::remove(blob_tmp.c_str());
                                        std::remove(manifest_tmp.c_str());
                                    }
                                } അല്ലെങ്കിൽ {
                                    std::remove(blob_tmp.c_str());
                                    std::remove(manifest_tmp.c_str());
                                }
                            } അല്ലെങ്കിൽ {
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    } പിടിക്കുക (സ്ഥിരം std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                entry->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            entry->ptr = ggml_decoder;
            എങ്കിൽ (!shared_താക്കോൽ.empty() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_താക്കോൽ,
                                             ov_compiled_graph{shared_model, {}, ov_നിവേശം_names, ov_output_names});
            }
            എങ്കിൽ (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[താക്കോൽ] = infer_request;
                r_ctx->ov_നിവേശം_names_cache[താക്കോൽ] = ov_നിവേശം_names;
                r_ctx->ov_output_names_cache[താക്കോൽ] = ov_output_names;
            }

            എങ്കിൽ (stateful_kv_only && cache_enabled) {
                സ്ഥിരം auto * inp_pos = get_inp_pos_tensor(cgraph);
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                സ്ഥിരം int32_t pos_begin = ((int32_t *) inp_pos->data)[0];
                എങ്കിൽ (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    തിരികെ GGML_STATUS_FAILED;
                }
                r_ctx->stateful_kv_size = pos_shape[3];
                സ്ഥിരം auto kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                വേണ്ടി (സ്ഥിരം auto & pair : kv_param_res_names) {
                    r_ctx->kv_state_നിവേശം_name_map[pair.first + pair.second] = pair.first;
                }
            }
        }

        എങ്കിൽ (stateful_recurrent) {
            സ്ഥിരം auto * inp_pos = get_inp_pos_tensor(cgraph);
            സ്ഥിരം int32_t pos_begin = static_cast<സ്ഥിരം int32_t *>(inp_pos->data)[0];
            എങ്കിൽ (pos_begin == 0) {
                infer_request->reset_state();
            } അല്ലെങ്കിൽ എങ്കിൽ (!cache_hit || old_m_params.kv_buffer_changed(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_size) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                തിരികെ GGML_STATUS_FAILED;
            }
        } അല്ലെങ്കിൽ {
            reset_single_slot_recurrent_cache(cgraph, m_params, c_params);
        }

        വേണ്ടി (size_t i = 0; i < ov_നിവേശം_names.size(); i++) {
            സ്ഥിരം auto & param_name = ov_നിവേശം_names[i];
            auto നിവേശം_tensor = get_ov_നിവേശം_tensor(ggml_decoder, param_name);
            infer_request->set_നിവേശം_tensor(i, നിവേശം_tensor);

            എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_നിവേശം_tensor_info(param_name, നിവേശം_tensor);
            }
        }

        വേണ്ടി (size_t i = 0; i < ov_output_names.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            സ്ഥിരം auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names[i]);
            എങ്കിൽ (model_output_it == model_outputs.end()) {
                തുടരുക;
            }
            auto * ggml_tensor = model_output_it->second;
            എങ്കിൽ (ggml_nbytes(ggml_tensor) == 0) {
                തുടരുക;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        എങ്കിൽ (stateful_recurrent) {
            സ്ഥിരം auto * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_size =
                static_cast<സ്ഥിരം int32_t *>(inp_pos->data)[0] + get_inp_pos_n_tokens(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            വേണ്ടി (size_t i = 0; i < ov_output_names.size(); i++) {
                സ്ഥിരം auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }

        എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_താക്കോൽ_compute_time / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_split_check_time / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", graph_param_compute_time / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_time / 1000.0);
            എങ്കിൽ (!cache_hit) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_end_time - decoder_end_time) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
            GGML_LOG_INFO("    - OV raw infer time: %.3f ms \n", (infer_end_time - ov_raw_infer_start) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. Release only while holding the compilation mutex so another context cannot
    // be reading host weights during conversion/compilation. Pin the shared compiled
    // models across backend teardown; a later context can create its own request without
    // reading the dropped pages. A new, uncached graph still fails fast above.
    എങ്കിൽ (cache_hit && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_lock(r_ctx->compiled_cache->mutex);
        എങ്കിൽ (!ggml_openvino_weight_buffers_released()) {
            ggml_openvino_release_weight_buffers();
        }
    }

    തിരികെ GGML_STATUS_SUCCESS;
}

ov::AnyMap without_npuw(സ്ഥിരം ov::AnyMap & config) {
    ov::AnyMap out;
    വേണ്ടി (സ്ഥിരം auto & kv : config) {
        എങ്കിൽ (kv.first.rfind("NPUW", 0) == 0 || kv.first == "NPU_USE_NPUW") {
            തുടരുക;
        }
        out.insert(kv);
    }
    തിരികെ out;
}

എണ്ണൽ ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, സ്ഥിരം std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();

    auto get_prefill_chunk_size = [] {
        സ്ഥിതം സ്ഥിരം പൂർണ്ണസംഖ്യ chunk_size = []() {
            പൂർണ്ണസംഖ്യ env_prefill_chunk_size = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            തിരികെ env_prefill_chunk_size > 0 ? env_prefill_chunk_size : 256;
        }();
        തിരികെ chunk_size;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    സ്ഥിതം std::string device = ggml_openvino_get_device_name();
    സ്ഥിതം auto is_static = സത്യം;
    സ്ഥിതം auto stateful = അസത്യം;

    auto prefill_chunk_size = get_prefill_chunk_size();
    സ്ഥിരം auto & config = ggml_openvino_get_compile_config();

    എങ്കിൽ (is_naive(cgraph)) {
        തിരികെ naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    സ്ഥിരം auto * inp_pos = get_inp_pos_tensor(cgraph);
    സ്ഥിരം തർക്കം no_kv_cache = m_params.is_cacheless_attn;
    സ്ഥിരം auto is_prefill = no_kv_cache ? സത്യം : get_is_prefill(cgraph, inp_pos);
    സ്ഥിരം ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    എങ്കിൽ (m_params.n_heads_kv == -1) {
        prefill_chunk_size = inp_pos->ne[0];
    }
    graph_താക്കോൽ താക്കോൽ(cgraph);
    സ്ഥിതം സ്ഥിരം തർക്കം cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    തർക്കം cache_hit = അസത്യം;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;
    int64_t ov_raw_infer_total = 0;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    എങ്കിൽ (cache_enabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        auto it = r_ctx->decoder_cache.find(താക്കോൽ);
        cache_hit = it != r_ctx->decoder_cache.end();
        എങ്കിൽ (cache_hit) {
            entry = it->second;
        } അല്ലെങ്കിൽ {
            r_ctx->clear_caches_locked();
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[താക്കോൽ] = entry;
        }
    } അല്ലെങ്കിൽ {
        auto mutex = std::make_shared<std::mutex>();
        entry = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_hit = അസത്യം;
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));
    cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(താക്കോൽ) != 0 &&
                r_ctx->infer_request_cache_prefill.count(താക്കോൽ) != 0;

    എങ്കിൽ (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_നിവേശം_names_local;
    std::vector<std::string> ov_output_names_local;

    എങ്കിൽ (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        എങ്കിൽ (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(താക്കോൽ) : r_ctx->infer_request_cache.at(താക്കോൽ);
            ov_നിവേശം_names_local = r_ctx->ov_നിവേശം_names_cache.at(താക്കോൽ);
            ov_output_names_local = r_ctx->ov_output_names_cache.at(താക്കോൽ);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(താക്കോൽ);
            r_ctx->infer_request_cache_prefill.erase(താക്കോൽ);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        auto shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
        auto weight_names = get_weight_names(cgraph);
        auto local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names, is_static,
                                                             stateful, അസത്യം, is_prefill, prefill_chunk_size);
        സ്ഥിരം std::string shared_താക്കോൽ =
            cache_enabled ? compiled_graph_താക്കോൽ(cgraph, *local_decoder, device, prefill_chunk_size) : "";
        auto shared_it = shared_cache->graphs.find(shared_താക്കോൽ);
        എങ്കിൽ (!shared_താക്കോൽ.empty() && shared_it != shared_cache->graphs.end()) {
            auto & compiled = shared_it->second;
            auto prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            auto decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            entry->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_നിവേശം_names_local = compiled.നിവേശം_names;
            ov_output_names_local = compiled.output_names;
            r_ctx->infer_request_cache_prefill[താക്കോൽ] = prefill_request;
            r_ctx->infer_request_cache[താക്കോൽ] = decode_request;
            r_ctx->ov_നിവേശം_names_cache[താക്കോൽ] = ov_നിവേശം_names_local;
            r_ctx->ov_output_names_cache[താക്കോൽ] = ov_output_names_local;
            decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } അല്ലെങ്കിൽ {
            std::shared_ptr<ov::Model> model;
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

            auto ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, അസത്യം, സത്യം, prefill_chunk_size);
            auto ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, അസത്യം, അസത്യം, prefill_chunk_size);
            decoder_end_time = ggml_time_us();

            സ്ഥിരം തർക്കം dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            സ്ഥിരം auto dump_ir_timestamp = static_cast<long long>(ggml_time_us());

            auto build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          സ്ഥിരം std::shared_ptr<GgmlOvDecoder> & decoder, സ്ഥിരം അക്ഷരം * tag,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_end_time, int64_t & local_compile_end_time) {
                auto നിവേശം_model = std::make_shared<ov::frontend::ggml::നിവേശം_Model>(decoder);
                model = ov::frontend::ggml::FrontEnd::convert(നിവേശം_model);
                decoder->clear_model_weights();
                local_conversion_end_time = ggml_time_us();

                എങ്കിൽ (dump_ir) {
                    അക്ഷരം timestamped_filename[64];
                    snprintf(timestamped_filename, വലുപ്പം(timestamped_filename), "model_%s_%lld.xml", tag,
                             dump_ir_timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                compiled_model = core.compile_model(model, device, compile_config);
                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                local_compile_end_time = ggml_time_us();
            };
            std::shared_ptr<ov::Model> model_prefill;
            std::shared_ptr<ov::Model> model_decode;
            ov::CompiledModel compiled_model_prefill;
            ov::CompiledModel compiled_model_decode;
            std::shared_ptr<ov::InferRequest> infer_request_prefill;
            std::shared_ptr<ov::InferRequest> infer_request_decode;
            int64_t prefill_conversion_end_time;
            int64_t decode_conversion_end_time;
            int64_t prefill_compile_end_time;
            int64_t decode_compile_end_time;
            build_static_model(ggml_decoder_prefill, "prefill", model_prefill, compiled_model_prefill,
                               infer_request_prefill, prefill_conversion_end_time, prefill_compile_end_time);
            എങ്കിൽ (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_end_time = prefill_conversion_end_time;
                decode_compile_end_time = prefill_compile_end_time;
            } അല്ലെങ്കിൽ {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_end_time, decode_compile_end_time);
            }
            conversion_end_time = std::max(prefill_conversion_end_time, decode_conversion_end_time);
            compile_end_time = std::max(prefill_compile_end_time, decode_compile_end_time);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            entry->ptr = ggml_decoder;

            വേണ്ടി (സ്ഥിരം auto & ov_param : model->get_parameters()) {
                ov_നിവേശം_names_local.push_back(ov_param->get_friendly_name());
            }
            വേണ്ടി (സ്ഥിരം auto & ov_output : model->get_results()) {
                ov_output_names_local.push_back(ov_output->get_friendly_name());
            }

            എങ്കിൽ (!shared_താക്കോൽ.empty()) {
                shared_cache->graphs.emplace(
                    shared_താക്കോൽ, ov_compiled_graph{compiled_model_decode, compiled_model_prefill, ov_നിവേശം_names_local,
                                                  ov_output_names_local});
            }

            എങ്കിൽ (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[താക്കോൽ] = infer_request_prefill;
                r_ctx->infer_request_cache[താക്കോൽ] = infer_request_decode;
                r_ctx->ov_നിവേശം_names_cache[താക്കോൽ] = ov_നിവേശം_names_local;
                r_ctx->ov_output_names_cache[താക്കോൽ] = ov_output_names_local;
            }
        }
    }

    reset_single_slot_recurrent_cache(cgraph, m_params, c_params);

    എങ്കിൽ (is_prefill) {
        auto inp_len = get_inp_pos_n_tokens(cgraph, inp_pos);
        വേണ്ടി (പൂർണ്ണസംഖ്യ chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            വേണ്ടി (size_t i = 0; i < ov_നിവേശം_names_local.size(); i++) {
                സ്ഥിരം auto & param_name = ov_നിവേശം_names_local[i];
                auto നിവേശം_tensor = get_ov_നിവേശം_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_നിവേശം_tensor(i, നിവേശം_tensor);

                എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    സ്ഥിരം auto നിവേശം_tensor = infer_request->get_നിവേശം_tensor(i);
                    print_നിവേശം_tensor_info(param_name, നിവേശം_tensor);
                }
            }

            വേണ്ടി (size_t i = 0; i < ov_output_names_local.size(); i++) {
                സ്ഥിരം auto & model_outputs = ggml_decoder->get_model_outputs();
                auto model_output_it = model_outputs.find(ov_output_names_local[i]);
                എങ്കിൽ (model_output_it == model_outputs.end()) {
                    തുടരുക;
                }
                auto * ggml_tensor = model_output_it->second;
                എങ്കിൽ (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    തുടരുക;
                }
                auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            ov_raw_infer_start = ggml_time_us();
            infer_request->infer();
            ov_raw_infer_total += ggml_time_us() - ov_raw_infer_start;

            എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                വേണ്ടി (size_t i = 0; i < ov_output_names_local.size(); i++) {
                    സ്ഥിരം auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } അല്ലെങ്കിൽ {
        വേണ്ടി (size_t i = 0; i < ov_നിവേശം_names_local.size(); i++) {
            സ്ഥിരം auto & param_name = ov_നിവേശം_names_local[i];
            auto നിവേശം_tensor = get_ov_നിവേശം_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_നിവേശം_tensor(i, നിവേശം_tensor);

            എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                സ്ഥിരം auto നിവേശം_tensor = infer_request->get_നിവേശം_tensor(i);
                print_നിവേശം_tensor_info(param_name, നിവേശം_tensor);
            }
        }

        വേണ്ടി (size_t i = 0; i < ov_output_names_local.size(); i++) {
            സ്ഥിരം auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names_local[i]);
            എങ്കിൽ (model_output_it == model_outputs.end()) {
                തുടരുക;
            }
            auto * ggml_tensor = model_output_it->second;
            എങ്കിൽ (ggml_nbytes(ggml_tensor) == 0) {
                തുടരുക;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        ov_raw_infer_total = infer_end_time - ov_raw_infer_start;

        എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            വേണ്ടി (size_t i = 0; i < ov_output_names_local.size(); i++) {
                സ്ഥിരം auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
            }
        }
    }

    എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
        എങ്കിൽ (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_end_time - decoder_end_time) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_total / 1000.0);
    }

    തിരികെ GGML_STATUS_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
സ്ഥിതം ggml_cgraph * get_compute_graph(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    auto is_skipped = [](സ്ഥിരം ggml_tensor * കണ്ണി) {
        തിരികെ കണ്ണി->op != GGML_OP_NONE && !(കണ്ണി->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    എങ്കിൽ (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        തിരികെ cgraph;
    }
    auto & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](സ്ഥിരം ggml_tensor * കണ്ണി) { തിരികെ !is_skipped(കണ്ണി); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.data();
    compute.graph.n_nodes = (പൂർണ്ണസംഖ്യ) compute.nodes.size();
    തിരികെ &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
എണ്ണൽ ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    ശ്രമിക്കുക {
        എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        സ്ഥിരം auto is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_lock(r_ctx->execution_mutex);
        cgraph = get_compute_graph(cgraph, *r_ctx);

        തിരികെ is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } പിടിക്കുക (സ്ഥിരം ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        തിരികെ GGML_STATUS_FAILED;
    } പിടിക്കുക (സ്ഥിരം std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        തിരികെ GGML_STATUS_FAILED;
    } പിടിക്കുക (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        തിരികെ GGML_STATUS_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
തർക്കം is_model_splitted(ggml_cgraph * cgraph) {
    സ്ഥിതം സ്ഥിരം തർക്കം fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    എങ്കിൽ (!fallback_enabled) {
        തിരികെ അസത്യം;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    എങ്കിൽ (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        തിരികെ അസത്യം;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * കണ്ണി = cgraph->nodes[i];
        പൂർണ്ണസംഖ്യ use_എണ്ണം = cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, കണ്ണി)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        എങ്കിൽ ((cgraph->n_nodes <= 1 && use_എണ്ണം == 0) ||
            (cgraph->n_nodes <= 1 && കണ്ണി->op == GGML_OP_VIEW && use_എണ്ണം == 1 && കണ്ണി->src[0] != nullptr &&
             കണ്ണി->src[0]->op == GGML_OP_NONE)) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_PERMUTE)) {
            തിരികെ അസത്യം;
        }
        പൂർണ്ണസംഖ്യ നിവേശം_use_എണ്ണം = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_കണ്ണി = cgraph->nodes[j];
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < GGML_MAX_SRC; k++) {
                എങ്കിൽ (other_കണ്ണി->src[k] == കണ്ണി) {
                    നിവേശം_use_എണ്ണം++;
                }
            }
        }
        എങ്കിൽ (use_എണ്ണം != നിവേശം_use_എണ്ണം && കണ്ണി->op != GGML_OP_NONE) {
            തിരികെ സത്യം;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    എങ്കിൽ (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_weight_names(cgraph);
    } അല്ലെങ്കിൽ {
        വേണ്ടി (സ്ഥിരം auto & kv : GgmlOvDecoder::create_weight_nodes(cgraph, സത്യം)) {
            model_weights.insert(kv.first);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * കണ്ണി = cgraph->nodes[i];
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = കണ്ണി->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            എങ്കിൽ (src != nullptr && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.empty() == അസത്യം &&
                model_leafs.find(src) == model_leafs.end()) {
                എങ്കിൽ (GgmlOvDecoder::is_inp_tok(src, കണ്ണി)) {
                    തിരികെ അസത്യം;
                }
                തിരികെ സത്യം;
            }
        }
    }
    തിരികെ അസത്യം;
}

തർക്കം is_naive(ggml_cgraph * cgraph) {
    constexpr പൂർണ്ണസംഖ്യ naive_graph_size_threshold = 20;
    പൂർണ്ണസംഖ്യ count = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        എങ്കിൽ (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    തിരികെ count < naive_graph_size_threshold;
}

size_t checksum(സ്ഥിരം ശൂന്യം * data, size_t size) {
    സ്ഥിരം uint8_t * bytes = static_cast<സ്ഥിരം uint8_t *>(data);
    size_t sum = 0;
    വേണ്ടി (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    തിരികെ sum;
}

തർക്കം save_ggml_tensor_data_to_txt(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം std::string & file_path) {
    എങ്കിൽ (tensor == nullptr || tensor->data == nullptr) {
        തിരികെ അസത്യം;
    }

    std::ofstream out(file_path);
    എങ്കിൽ (!out.is_open()) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം size_t n = ggml_nelements(tensor);
    out << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    തിരഞ്ഞെടുപ്പ് (tensor->type) {
    അവസ്ഥ GGML_TYPE_F32: {
        സ്ഥിരം auto * data = static_cast<സ്ഥിരം ദശാംശം *>(tensor->data);
        വേണ്ടി (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_TYPE_F16: {
        സ്ഥിരം auto * data = static_cast<സ്ഥിരം ggml_fp16_t *>(tensor->data);
        വേണ്ടി (size_t i = 0; i < n; ++i) {
            out << ggml_fp16_to_fp32(data[i]) << '\n';
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_TYPE_BF16: {
        സ്ഥിരം auto * data = static_cast<സ്ഥിരം ggml_bf16_t *>(tensor->data);
        വേണ്ടി (size_t i = 0; i < n; ++i) {
            out << ggml_bf16_to_fp32(data[i]) << '\n';
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_TYPE_I32: {
        സ്ഥിരം auto * data = static_cast<സ്ഥിരം int32_t *>(tensor->data);
        വേണ്ടി (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        നിർത്തുക;
    }
    അവസ്ഥ GGML_TYPE_I64: {
        സ്ഥിരം auto * data = static_cast<സ്ഥിരം int64_t *>(tensor->data);
        വേണ്ടി (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        നിർത്തുക;
    }
    സ്ഥിരസ്ഥിതി:
        out << "unsupported tensor type for text dump" << '\n';
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

ശൂന്യം print_നിവേശം_tensor_info(സ്ഥിരം std::string & name, സ്ഥിരം ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << '\n';
    തിരഞ്ഞെടുപ്പ് (tensor.get_element_type()) {
    അവസ്ഥ ov::element::f32: {
        എങ്കിൽ (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::cout << *(tensor.data<ദശാംശം>()) << '\n';
        } അല്ലെങ്കിൽ {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            സ്ഥിരം ദശാംശം * data = tensor.data<ദശാംശം>();
            വേണ്ടി (size_t i = 0; i < rows; ++i) {
                വേണ്ടി (size_t j = 0; j < cols; ++j) {
                    ദശാംശം val = data[i * cols + j];
                    എങ്കിൽ (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } അല്ലെങ്കിൽ {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << '\n';
            }
        }

        നിർത്തുക;
    }
    അവസ്ഥ ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << '\n';
        നിർത്തുക;
    അവസ്ഥ ov::element::i32:
        വേണ്ടി (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << ' ';
        }
        std::cout << '\n';
        നിർത്തുക;
    അവസ്ഥ ov::element::i64:
        വേണ്ടി (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << ' ';
        }
        std::cout << '\n';
        നിർത്തുക;
    സ്ഥിരസ്ഥിതി:
        നിർത്തുക;
    }
}

ശൂന്യം print_output_tensor_info(സ്ഥിരം std::string & name, സ്ഥിരം ov::Tensor & tensor, സ്ഥിരം ശൂന്യം * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    auto print_float_stats = [](സ്ഥിരം std::string & type_name, size_t size, auto get_മൂല്യം) {
        എങ്കിൽ (size == 0) {
            തിരികെ;
        }

        ദശാംശം first = get_മൂല്യം(0);
        ദശാംശം min = first;
        ദശാംശം max = first;
        ഇരട്ട sum = first;

        വേണ്ടി (size_t i = 1; i < size; ++i) {
            ദശാംശം v = get_മൂല്യം(i);
            min = std::min(v, min);
            max = std::max(v, max);
            sum += v;
        }
        ഇരട്ട mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << '\n';
    };

    തിരഞ്ഞെടുപ്പ് (tensor.get_element_type()) {
    അവസ്ഥ ov::element::f32: {
        സ്ഥിരം ദശാംശം * data = tensor.data<ദശാംശം>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { തിരികെ data[i]; });
        നിർത്തുക;
    }
    അവസ്ഥ ov::element::f16: {
        സ്ഥിരം ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { തിരികെ static_cast<ദശാംശം>(data[i]); });
        നിർത്തുക;
    }
    സ്ഥിരസ്ഥിതി:
        നിർത്തുക;
    }
}

സ്ഥിരം ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            എങ്കിൽ (src == nullptr) {
                നിർത്തുക;
            }
            എങ്കിൽ (GgmlOvDecoder::is_inp_pos(src, op)) {
                തിരികെ src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    എറിയുക std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_tokens(ggml_cgraph * cgraph, സ്ഥിരം ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    പൂർണ്ണസംഖ്യ n_planes = 1;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; ++j) {
            എങ്കിൽ (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                നിർത്തുക;
            }
        }
    }
    തിരികെ inp_pos->ne[0] / n_planes;
}

തർക്കം get_is_prefill(ggml_cgraph * cgraph, സ്ഥിരം ggml_tensor * inp_pos) {
    തിരികെ get_inp_pos_n_tokens(cgraph, inp_pos) > 1;
}
