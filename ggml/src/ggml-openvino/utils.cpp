#ଅନ୍ତର୍ଭାଵ "utils.h"

#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-openvino-extra.h"
#ଅନ୍ତର୍ଭାଵ "ggml-openvino.h"
#ଅନ୍ତର୍ଭାଵ "ggml-openvino/ggml-decoder.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "model-cache.h"
#ଅନ୍ତର୍ଭାଵ "openvino/frontend.h"
#ଅନ୍ତର୍ଭାଵ "openvino/input_model.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cassert>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <fstream>
#ଅନ୍ତର୍ଭାଵ <functional>
#ଅନ୍ତର୍ଭାଵ <iomanip>
#ଅନ୍ତର୍ଭାଵ <iostream>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/core/any.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/graph_util.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/shape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/type/float16.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/frontend/manager.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/openvino.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/runtime/compiled_model.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/runtime/infer_request.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/runtime/intel_npu/properties.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/runtime/properties.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/runtime/tensor.hpp>
#ଅନ୍ତର୍ଭାଵ <optional>
#ଅନ୍ତର୍ଭାଵ <set>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ {
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
std::optional<ov::Tensor> try_make_kv_sliced_tensor(ଅଚଳ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    ଅଚଳ std::string & name,
                                                    ଅଚଳ ggml_tensor * ggml_tensor) {
    ସ୍ଥିର ଅଚଳ ତାର୍କିକ kv_slice_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    ଯଦି (kv_slice_disabled) {
        ଫେରନ୍ତୁ std::nullopt;
    }
    ଯଦି (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        ଫେରନ୍ତୁ std::nullopt;
    }
    ଯଦି (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != nullptr) {
        ଫେରନ୍ତୁ std::nullopt;
    }
    ଅଚଳ auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);
    ଯଦି (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        ଫେରନ୍ତୁ std::nullopt;
    }

    ଅଚଳ auto & compute_params = ggml_decoder->get_compute_params();
    ଯଦି (compute_params.n_seq_active != 1 || compute_params.seq_active_start != 0) {
        ଫେରନ୍ତୁ std::nullopt;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା layer;
    ଯଦି (auto layer_opt = extract_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ std::nullopt;
    }

    ଅଚଳ ତାର୍କିକ is_swa = ggml_decoder->is_swa_layer(layer);
    ଯଦି (is_swa) {
        ଫେରନ୍ତୁ std::nullopt;
    }
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ctx_per_seq = ggml_decoder->get_ctx_per_seq();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_kv = compute_params.attention_size;
    ଯଦି (ctx_per_seq <= 0 || n_kv <= 0 || n_kv >= ctx_per_seq) {
        ଫେରନ୍ତୁ std::nullopt;
    }

    ov::Shape full_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    ଯଦି (full_shape.size() != 4 || full_shape[0] != 1 || full_shape[1] != 1 ||
        static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(full_shape[2]) != ctx_per_seq) {
        ଫେରନ୍ତୁ std::nullopt;
    }

    ov::Shape sliced_shape = full_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    ଯଦି (ggml_openvino_buffer_is_remote(ggml_tensor)) {
        auto remote_context = ggml_openvino_get_remote_context();
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        ଫେରନ୍ତୁ gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
    }

    ଫେରନ୍ତୁ ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
}

ସ୍ଥିର uint64_t ggml_openvino_model_cache_extra_cfg(ତାର୍କିକ stateful, ତାର୍କିକ recurrent) {
    ଅଚଳ ଅକ୍ଷର * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    ଅଚଳ ତାର୍କିକ manual_gqa_enabled = manual_gqa_env != nullptr ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    ଫେରନ୍ତୁ extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_weight_names(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> names;
    ପାଇଁ (ଅଚଳ auto & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
        names[name] = nullptr;
    }
    ଫେରନ୍ତୁ names;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
ସ୍ଥିର std::string compiled_graph_ଚାବି(ଅଚଳ ggml_cgraph * graph,
                                      ଅଚଳ GgmlOvDecoder & decoder,
                                      ଅଚଳ std::string & device,
                                      ପୂର୍ଣ୍ଣସଂଖ୍ୟା prefill_chunk_size = 0,
                                      ତାର୍କିକ disk_cache = ମିଥ୍ୟା) {
    std::string ଚାବି;
    auto append = [&ଚାବି](ଅଚଳ auto & value) {
        ଚାବି.append(reinterpret_cast<ଅଚଳ ଅକ୍ଷର *>(&value), ଆକାର(value));
    };
    auto append_string = [&](ଅଚଳ std::string & value) {
        append(value.size());
        ଚାବି.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_size);
    ତାର୍କିକ has_weight_buffer_id = ମିଥ୍ୟା;
    std::unordered_map<ଅଚଳ ggml_tensor *, size_t> ids;
    std::function<ଶୂନ୍ୟ(ଅଚଳ ggml_tensor *)> visit = [&](ଅଚଳ ggml_tensor * tensor) {
        ଯଦି (!tensor) {
            append(size_t(0));
            ଫେରନ୍ତୁ;
        }
        auto inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.first->second);
        ଯଦି (!inserted.second) {
            ଫେରନ୍ତୁ;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        ଅଚଳ auto * base = tensor->view_src ? tensor->view_src : tensor;
        ଅଚଳ ତାର୍କିକ weight = base->buffer && base->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        append(weight);
        ଯଦି (weight && !disk_cache) {
            ଅଚଳ size_t buffer_id = ggml_backend_openvino_buffer_get_ctx_id(base->buffer);
            has_weight_buffer_id |= buffer_id != 0;
            append(buffer_id);
            append(tensor->data);
        }
        visit(tensor->view_src);
        ପାଇଁ (ଅଚଳ auto * src : tensor->src) {
            visit(src);
        }
    };
    append(graph->n_nodes);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_nodes; ++i) {
        visit(graph->nodes[i]);
    }
    append(graph->n_leafs);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_leafs; ++i) {
        visit(graph->leafs[i]);
    }
    ପାଇଁ (ଅଚଳ auto & ନିବେଶ : decoder.get_model_extra_inputs()) {
        append_string(ନିବେଶ.first);
        append_string(ନିବେଶ.second.type.get_type_name());
        append(ନିବେଶ.second.shape.size());
        ପାଇଁ (auto dim : ନିବେଶ.second.shape) {
            append(dim);
        }
        append(ନିବେଶ.second.is_parameter);
        ଯଦି (!ନିବେଶ.second.is_parameter) {
            append(ନିବେଶ.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    ଫେରନ୍ତୁ disk_cache || has_weight_buffer_id ? ଚାବି : std::string{};
}

ସ୍ଥିର std::string dynamic_graph_signature(ଅଚଳ ggml_cgraph * graph,
                                           ଅଚଳ GgmlOvDecoder & decoder,
                                           ଅଚଳ ModelParams & params) {
    std::string ଚାବି = "dynamic-1";
    auto append = [&ଚାବି](ଅଚଳ auto & value) {
        ଚାବି.append(reinterpret_cast<ଅଚଳ ଅକ୍ଷର *>(&value), ଆକାର(value));
    };
    auto append_string = [&](ଅଚଳ std::string & value) {
        append(value.size());
        ଚାବି.append(value);
    };
    graph_ଚାବି graph_id(graph, ସତ୍ୟ);
    append(graph_id.n_nodes);
    append(graph_id.n_leaves);
    append_string(graph_id.first_ଗଣ୍ଠି_name);
    append_string(graph_id.last_ଗଣ୍ଠି_name);
    ପାଇଁ (ଅଚଳ auto & name : graph_id.ନିବେଶ_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା layer : params.swa_layers) {
        append(layer);
    }
    ପାଇଁ (ଅଚଳ auto & [layer, heads] : params.n_heads_kv_per_layer) {
        append(layer);
        append(heads);
    }
    ପାଇଁ (ଅଚଳ auto & [name, ନିବେଶ] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(ନିବେଶ.type.get_type_name());
    }
    ପାଇଁ (ଅଚଳ auto & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    ପାଇଁ (ଅଚଳ auto & [name, ନିବେଶ] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(ନିବେଶ.type.get_type_name());
        append(ନିବେଶ.is_parameter);
        ଯଦି (!ନିବେଶ.is_parameter) {
            append(ନିବେଶ.value);
        }
    }
    ଫେରନ୍ତୁ ଚାବି;
}

ସ୍ଥିର ତାର୍କିକ compiled_model_matches_graph(ଅଚଳ ov::CompiledModel & model,
                                         ଅଚଳ std::vector<std::string> & inputs,
                                         ଅଚଳ std::vector<std::string> & outputs,
                                         ଅଚଳ GgmlOvDecoder & decoder) {
    ଯଦି (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ auto & graph_inputs = decoder.get_model_inputs();
    ଅଚଳ auto & extra_inputs = decoder.get_model_extra_inputs();
    ଅଚଳ auto & graph_outputs = decoder.get_model_outputs();
    ପାଇଁ (size_t i = 0; i < inputs.size(); ++i) {
        ଅଚଳ auto & port = model.ନିବେଶ(i);
        auto graph_it = graph_inputs.find(inputs[i]);
        ଯଦି (graph_it != graph_inputs.end()) {
            ଯଦି (port.get_element_type() != graph_it->second.type ||
                !port.get_partial_shape().compatible(graph_it->second.shape)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଜାରି;
        }
        auto extra_it = extra_inputs.find(inputs[i]);
        ଯଦି (extra_it == extra_inputs.end() || port.get_element_type() != extra_it->second.type ||
            !port.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }
    ପାଇଁ (size_t i = 0; i < outputs.size(); ++i) {
        auto graph_it = graph_outputs.find(outputs[i]);
        ଯଦି (graph_it == graph_outputs.end()) {
            ଯଦି (outputs[i].compare(0, 8, "__debug_") != 0) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଜାରି;
        }
        ଅଚଳ auto & port = model.output(i);
        ଯଦି (port.get_element_type() != GgmlOvDecoder::get_ov_type(graph_it->second) ||
            !port.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(graph_it->second)))) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର std::string ov_profiling_csv_field(ଅଚଳ std::string & value) {
    std::string escaped = "\"";
    ପାଇଁ (ଅକ୍ଷର c : value) {
        escaped += c;
        ଯଦି (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    ଫେରନ୍ତୁ escaped;
}

ସ୍ଥିର ଶୂନ୍ୟ dump_ov_profiling_info(ଅଚଳ ov::InferRequest & infer_request) {
    ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        ଫେରନ୍ତୁ;
    }

    ସ୍ଥିର std::atomic<uint64_t> inference_index{0};
    ଅଚଳ uint64_t index = inference_index.fetch_add(1);
    ଅଚଳ std::string path = "openvino_profile_" + std::to_string(index) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    ଯଦି (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        ଫେରନ୍ତୁ;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t total_real_time = 0;
    int64_t total_cpu_time = 0;
    size_t executed_ଗଣନା = 0;
    ପାଇଁ (ଅଚଳ auto & info : infer_request.get_profiling_info()) {
        ଅଚଳ ଅକ୍ଷର * status = "NOT_RUN";
        ଯଦି (info.status == ov::ProfilingInfo::Status::EXECUTED) {
            status = "EXECUTED";
            total_real_time += info.real_time.count();
            total_cpu_time += info.cpu_time.count();
            executed_ଗଣନା++;
        } ଅନ୍ୟଥା ଯଦି (info.status == ov::ProfilingInfo::Status::OPTIMIZED_OUT) {
            status = "OPTIMIZED_OUT";
        }

        output << status << ',' << info.real_time.count() << ',' << info.cpu_time.count() << ','
               << info.start_time.count() << ',' << ov_profiling_csv_field(info.node_type) << ','
               << ov_profiling_csv_field(info.ଗଣ୍ଠି_name) << ',' << ov_profiling_csv_field(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_ଗଣନା, total_real_time / 1000.0, total_cpu_time / 1000.0);
}

ସ୍ଥିର ଶୂନ୍ୟ reset_single_slot_recurrent_cache(ggml_cgraph * cgraph,
                                              ଅଚଳ ModelParams & model_params,
                                              ଅଚଳ ComputeParams & compute_params) {
    ଯଦି (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        ଫେରନ୍ତୁ;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_buffer_t> buffers;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        auto * ଗଣ୍ଠି = cgraph->nodes[i];
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(ଗଣ୍ଠି->view_src) &&
            ଗଣ୍ଠି->view_src->ne[1] == 1 && ଗଣ୍ଠି->view_src->buffer != nullptr) {
            buffers.insert(ଗଣ୍ଠି->view_src->buffer);
        }
    }
    ପାଇଁ (auto * buffer : buffers) {
        ggml_backend_buffer_clear(buffer, 0);
    }
}

ov::Tensor create_ov_output_tensor(ଅଚଳ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   ଅଚଳ std::shared_ptr<ov::InferRequest> & infer_request,
                                   ପୂର୍ଣ୍ଣସଂଖ୍ୟା output_index,
                                   ଅଚଳ ggml_tensor * ggml_tensor) {
    ଯଦି (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        ଫେରନ୍ତୁ *sliced;
    }

    auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    ଶୂନ୍ୟ * output_data = ggml_tensor->data;
    ଯଦି (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_index).get_shape();
    } ଅନ୍ୟଥା {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        ଯଦି (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != nullptr &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_data = ggml_tensor->view_src->data;
        } ଅନ୍ୟଥା {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    ଯଦି (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_buffer_is_remote(ggml_tensor) &&
        ggml_tensor->extra != nullptr) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        ଯଦି (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            ଯଦି (tensor_extra->tensor != nullptr && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                ଫେରନ୍ତୁ *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_data);
    ଫେରନ୍ତୁ output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_rows_to_seq_axis_2(ଅଚଳ ov::Tensor & kv_tensor, size_t n_heads_kv) {
    ଅଚଳ size_t rows = kv_tensor.get_shape()[2];
    ଅଚଳ size_t head_size = kv_tensor.get_shape()[3] / n_heads_kv;
    ଅଚଳ size_t elem = kv_tensor.get_element_type().size();
    ଅଚଳ size_t head_bytes = head_size * elem;

    ov::Tensor out(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, rows, head_size});
    ଅଚଳ auto * src = static_cast<ଅଚଳ uint8_t *>(kv_tensor.data());
    auto * dst = static_cast<uint8_t *>(out.data());
    ପାଇଁ (size_t s = 0; s < rows; s++) {
        ପାଇଁ (size_t h = 0; h < n_heads_kv; h++) {
            memcpy(dst + (h * rows + s) * head_bytes, src + (s * n_heads_kv + h) * head_bytes, head_bytes);
        }
    }
    ଫେରନ୍ତୁ out;
}

template <typename T> ଶୂନ୍ୟ set_zero_diagonal(std::vector<T> & matrix, size_t rows, size_t cols, T zero_ମୂଲ୍ୟ = T{}) {
    ପାଇଁ (size_t i = 0; i < rows; ++i) {
        size_t diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = zero_ମୂଲ୍ୟ;
    }
}

ov::Tensor make_contiguous_split_ନିବେଶ_tensor(ଅଚଳ ସଂରଚନା ggml_tensor * ggml_tensor, ଅଚଳ ov::Shape & ନିବେଶ_shape) {
    ଅଚଳ size_t element_size = ggml_type_size(ggml_tensor->type);
    ଅଚଳ size_t block_size = ggml_blck_size(ggml_tensor->type);

    GGML_ASSERT(block_size == 1 && "non-contiguous split inputs must be plain element types");

    ଅଚଳ ସଂରଚନା ggml_tensor * source_tensor = ggml_tensor->view_src != nullptr ? ggml_tensor->view_src : ggml_tensor;
    ଅଚଳ size_t source_offset = ggml_tensor->view_src != nullptr ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> source_data(ggml_nbytes(source_tensor));
    ggml_backend_tensor_get(source_tensor, source_data.data(), 0, source_data.size());

    ov::Tensor ନିବେଶ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ନିବେଶ_shape);
    auto * dst = static_cast<uint8_t *>(ନିବେଶ_tensor.data());
    size_t dst_offset = 0;

    ପାଇଁ (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        ପାଇଁ (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            ପାଇଁ (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                ପାଇଁ (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    ଅଚଳ size_t src_offset = source_offset + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::memcpy(dst + dst_offset, source_data.data() + src_offset, element_size);
                    dst_offset += element_size;
                }
            }
        }
    }

    ଫେରନ୍ତୁ ନିବେଶ_tensor;
}

ov::Tensor convert_ggml_ନିବେଶ_to_ov(ଅଚଳ std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ଅଚଳ std::string & name) {
    ଅଚଳ auto * ggml_tensor = ggml_decoder->get_ନିବେଶ_ggml_tensor(name);

    ଯଦି (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        ଫେରନ୍ତୁ *sliced;
    }

    ଯଦି (ggml_tensor->extra != nullptr && !ggml_decoder->is_splited_model()) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        ଯଦି (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            ଫେରନ୍ତୁ *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    auto * ନିବେଶ_data = ggml_tensor->data;
    ov::Shape ନିବେଶ_shape;
    ଯଦି (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        ନିବେଶ_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } ଅନ୍ୟଥା {
        ନିବେଶ_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    ଯଦି (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        ଫେରନ୍ତୁ make_contiguous_split_ନିବେଶ_tensor(ggml_tensor, ନିବେଶ_shape);
    }

    auto ନିବେଶ_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ନିବେଶ_shape, ନିବେଶ_data);
    ଫେରନ୍ତୁ ନିବେଶ_tensor;
}

ov::Tensor get_ov_ନିବେଶ_tensor(ଅଚଳ std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ଅଚଳ std::string & param_name) {
    ov::Tensor ନିବେଶ_tensor;
    auto extra_ନିବେଶ = ggml_decoder->get_model_extra_inputs().find(param_name);
    ଯଦି (extra_ନିବେଶ != ggml_decoder->get_model_extra_inputs().end()) {
        ନିବେଶ_tensor = ov::Tensor(extra_ନିବେଶ->second.type, extra_ନିବେଶ->second.shape);
        *ନିବେଶ_tensor.data<int64_t>() = extra_ନିବେଶ->second.value;
    } ଅନ୍ୟଥା {
        ନିବେଶ_tensor = convert_ggml_ନିବେଶ_to_ov(ggml_decoder, param_name);
    }
    ଫେରନ୍ତୁ ନିବେଶ_tensor;
}

ov::Tensor get_ov_ନିବେଶ_tensor_static_decode(ଅଚଳ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             ଅଚଳ std::string & param_name) {
    // NPU decoding stage
    ଯଦି (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        ଫେରନ୍ତୁ get_ov_ନିବେଶ_tensor(ggml_decoder, param_name);
    }
    ଅଚଳ auto * ggml_tensor = ggml_decoder->get_ନିବେଶ_ggml_tensor(param_name);
    ଅଚଳ auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    ଯଦି (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        ପ୍ରତିଜ୍ଞା(ggml_tensor->ne[0] == n_planes);
        ov::Shape ନିବେଶ_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor ନିବେଶ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ନିବେଶ_shape);
        std::memcpy(ନିବେଶ_tensor.data(), ggml_tensor->data, n_planes * ggml_type_size(ggml_tensor->type));
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }

    ଯଦି (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape ନିବେଶ_shape = {1, 1, 1, 1};
        ov::Tensor ନିବେଶ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ନିବେଶ_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        ପ୍ରତିଜ୍ଞା(ggml_tensor->ne[0] == 1);
        ପ୍ରତିଜ୍ଞା(inp_out_id == 0);
        *ନିବେଶ_tensor.data<int32_t>() = inp_out_id;
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }

    ଯଦି (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t context_size = ggml_decoder->get_ctx_size();
        ଯଦି (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_data =
                pad_ନିବେଶ<ggml_fp16_t>(ggml_tensor, 1, context_size, GGML_FP32_TO_FP16(-INFINITY));
            ov::Tensor ନିବେଶ_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_size});
            std::memcpy(ନିବେଶ_tensor.data(), padded_data.data(), padded_data.size() * ଆକାର(ggml_fp16_t));
            ଫେରନ୍ତୁ ନିବେଶ_tensor;
        }

        std::vector<ଦଶମିକ> padded_data = pad_ନିବେଶ<ଦଶମିକ>(ggml_tensor, 1, context_size, -INFINITY);
        ov::Tensor ନିବେଶ_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        auto * data_ptr = ନିବେଶ_tensor.data<ଦଶମିକ>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }

    ଫେରନ୍ତୁ get_ov_ନିବେଶ_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_ନିବେଶ_tensor_static_prefill(ଅଚଳ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              ଅଚଳ std::string & param_name,
                                              ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_index) {
    // NPU prompt processing stage
    ଅଚଳ size_t ନିବେଶ_len = ggml_decoder->get_ନିବେଶ_len();
    ଅଚଳ size_t chunk_size = ggml_decoder->m_prefill_chunk_size;
    ଅଚଳ size_t chunk_valid_size = std::min(chunk_size, ନିବେଶ_len - chunk_index * chunk_size);
    ଅଚଳ size_t chunk_pad_size = chunk_size - chunk_valid_size;

    ଯଦି (param_name == "chunk_valid_len") {
        ov::Tensor ନିବେଶ_tensor(ov::element::i64, ov::Shape{1});
        *ନିବେଶ_tensor.data<int64_t>() = (int64_t) chunk_valid_size;
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }
    ଯଦି (chunk_index > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor ନିବେଶ_tensor(ov::element::i64, ov::Shape{1});
        *ନିବେଶ_tensor.data<int64_t>() = 0;
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }
    ଯଦି (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        ଫେରନ୍ତୁ get_ov_ନିବେଶ_tensor(ggml_decoder, param_name);
    }
    ଅଚଳ auto * ggml_tensor = ggml_decoder->get_ନିବେଶ_ggml_tensor(param_name);
    ଅଚଳ auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    ଯଦି (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        ଅଚଳ size_t element_size = ggml_type_size(ggml_tensor->type);
        ov::Shape ନିବେଶ_shape = {1, 1, 1, (size_t) n_planes * chunk_size};
        ov::Tensor ନିବେଶ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ନିବେଶ_shape);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା p = 0; p < n_planes; p++) {
            ଅଚଳ ଅକ୍ଷର * src =
                (ଅଚଳ ଅକ୍ଷର *) ggml_tensor->data + (p * ନିବେଶ_len + chunk_index * chunk_size) * element_size;
            ଅକ୍ଷର * dst = (ଅକ୍ଷର *) ନିବେଶ_tensor.data() + p * chunk_size * element_size;
            std::memcpy(dst, src, chunk_valid_size * element_size);
            ଯଦି (chunk_pad_size > 0) {
                ଯଦି (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t last_ମୂଲ୍ୟ = *((ଅଚଳ int32_t *) src + chunk_valid_size - 1);
                    int32_t * out = (int32_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_ମୂଲ୍ୟ + 1);
                } ଅନ୍ୟଥା ଯଦି (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t last_ମୂଲ୍ୟ = *((ଅଚଳ int64_t *) src + chunk_valid_size - 1);
                    int64_t * out = (int64_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_ମୂଲ୍ୟ + 1);
                } ଅନ୍ୟଥା {
                    ଫିଙ୍ଗ std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }

    ଯଦି (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape ନିବେଶ_shape = {1, 1, 1, chunk_size};
        ov::Tensor ନିବେଶ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ନିବେଶ_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        ଶୂନ୍ୟ * ନିବେଶ_data = (ଅକ୍ଷର *) ggml_tensor->data + chunk_index * chunk_size * element_size;
        std::memcpy(ନିବେଶ_tensor.data(), ନିବେଶ_data, chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        ଯଦି (chunk_pad_size > 0) {
            ଯଦି (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_ମୂଲ୍ୟ =
                    *((int32_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int32_t * output_data = ନିବେଶ_tensor.data<int32_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_ମୂଲ୍ୟ + 1);
            } ଅନ୍ୟଥା ଯଦି (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_ମୂଲ୍ୟ =
                    *((int64_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int64_t * output_data = ନିବେଶ_tensor.data<int64_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_ମୂଲ୍ୟ + 1);
            } ଅନ୍ୟଥା {
                ଫିଙ୍ଗ std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }

    ଯଦି (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape ନିବେଶ_shape = {1, 1, 1, output_len};
        ov::Tensor ନିବେଶ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ନିବେଶ_shape);
        ଯଦି (ggml_tensor->ne[0] == 0) {
            *ନିବେଶ_tensor.data<int32_t>() = 0;
        } ଅନ୍ୟଥା {
            auto * data_addr = ନିବେଶ_tensor.data<int32_t>();
            ପାଇଁ (size_t i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }

    ଯଦି (GgmlOvDecoder::is_inp_scale_rows(ggml_tensor, op)) {
        ov::Tensor ନିବେଶ_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, 1});
        auto * dst = ନିବେଶ_tensor.data<ଦଶମିକ>();
        ଅଚଳ auto * src = static_cast<ଅଚଳ ଦଶମିକ *>(ggml_tensor->data) + chunk_index * chunk_size;
        std::copy(src, src + chunk_valid_size, dst);
        std::fill(dst + chunk_valid_size, dst + chunk_size, 1.0f);
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }

    ଯଦି (GgmlOvDecoder::is_inp_mean(ggml_tensor, op)) {
        ଅଚଳ size_t n_seqs = ggml_tensor->ne[1];
        ଅଚଳ size_t src_stride = ggml_tensor->ne[0];
        ଅଚଳ size_t copy_len = std::min<size_t>(chunk_valid_size, src_stride - chunk_index * chunk_size);
        ov::Tensor ନିବେଶ_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_size});
        auto * dst = ନିବେଶ_tensor.data<ଦଶମିକ>();
        std::fill(dst, dst + n_seqs * chunk_size, 0.0f);
        ଅଚଳ auto * src = static_cast<ଅଚଳ ଦଶମିକ *>(ggml_tensor->data) + chunk_index * chunk_size;
        ପାଇଁ (size_t s = 0; s < n_seqs; s++) {
            std::memcpy(dst + s * chunk_size, src + s * src_stride, copy_len * ଆକାର(ଦଶମିକ));
        }
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }

    ଯଦି (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = ggml_decoder->get_ctx_size();
        ଯଦି (ggml_tensor->type == GGML_TYPE_F16) {
            ଅଚଳ auto * ggml_data =
                static_cast<ଅଚଳ ggml_fp16_t *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
            std::vector<ggml_fp16_t> padded_data = pad_ନିବେଶ<ggml_fp16_t>(ggml_data, chunk_valid_rows, cols, chunk_size,
                                                                          context_size, GGML_FP32_TO_FP16(-INFINITY));
            set_zero_diagonal(padded_data, chunk_size, context_size, GGML_FP32_TO_FP16(0.0f));
            ov::Tensor ନିବେଶ_tensor(ov::element::f16, ov::Shape{1, 1, chunk_size, context_size});
            std::memcpy(ନିବେଶ_tensor.data(), padded_data.data(), padded_data.size() * ଆକାର(ggml_fp16_t));
            ଫେରନ୍ତୁ ନିବେଶ_tensor;
        }

        ଅଚଳ auto * ggml_data = static_cast<ଅଚଳ ଦଶମିକ *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
        std::vector<ଦଶମିକ> padded_data =
            pad_ନିବେଶ<ଦଶମିକ>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor ନିବେଶ_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        auto * data_ptr = ନିବେଶ_tensor.data<ଦଶମିକ>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        ଫେରନ୍ତୁ ନିବେଶ_tensor;
    }

    ଫେରନ୍ତୁ get_ov_ନିବେଶ_tensor(ggml_decoder, param_name);
}

ଗଣନା ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               ଅଚଳ std::string & device,
                               ଅଚଳ ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    ଯଦି (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_lock(cache.mutex);
    ତାର୍କିକ naive = ସତ୍ୟ;
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    auto ନିବେଶ_model = std::make_shared<ov::frontend::ggml::ନିବେଶ_Model>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(ନିବେଶ_model, naive);
    ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    ଯଦି (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    } ଅନ୍ୟଥା {
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    }
    ଯଦି (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } ଅନ୍ୟଥା {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> ନିବେଶ_names;
    std::vector<std::string> output_names;
    ପାଇଁ (ଅଚଳ auto & param : model->get_parameters()) {
        ନିବେଶ_names.push_back(param->get_friendly_name());
    }
    ପାଇଁ (ଅଚଳ auto & result : model->get_results()) {
        output_names.push_back(result->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    ନିବେଶ_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_lock.unlock();

    ପାଇଁ (size_t i = 0; i < ନିବେଶ_names.size(); i++) {
        ଅଚଳ auto & param_name = ନିବେଶ_names[i];
        auto ନିବେଶ_tensor = get_ov_ନିବେଶ_tensor(decoder, param_name);
        infer_request->set_ନିବେଶ_tensor(i, ନିବେଶ_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    ପାଇଁ (size_t i = 0; i < output_names.size(); i++) {
        auto output_tensor = infer_request->get_output_tensor(i);
        ଅଚଳ auto & model_outputs = decoder->get_model_outputs();
        auto model_output_it = model_outputs.find(output_names[i]);
        ଯଦି (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_names[i], output_tensor, output_tensor.data());
            }
            ଜାରି;
        }
        auto * ggml_tensor = model_output_it->second;
        std::memcpy(ggml_tensor->data, output_tensor.data(), output_tensor.get_byte_size());
    }
    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ଗଣନା ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, ଅଚଳ std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();
    ଅଚଳ auto & config = ggml_openvino_get_compile_config();
    ଅଚଳ auto & device = r_ctx->device;
    ସ୍ଥିର auto is_static = ମିଥ୍ୟା;

    ସ୍ଥିର ଅଚଳ ତାର୍କିକ cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    auto start_time = ggml_time_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    ଅଚଳ int64_t cache_ଚାବି_start_time = ggml_time_us();
    graph_ଚାବି ଚାବି(cgraph);
    ଅଚଳ int64_t cache_ଚାବି_compute_time = ggml_time_us() - cache_ଚାବି_start_time;
    int64_t cache_lookup_time = 0;
    ତାର୍କିକ ଚାବି_seen = ମିଥ୍ୟା;
    ଯଦି (!cache_disabled) {
        ଅଚଳ int64_t cache_lookup_start_time = ggml_time_us();
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        ଚାବି_seen = r_ctx->decoder_cache.find(ଚାବି) != r_ctx->decoder_cache.end();
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;
    }

    ଅଚଳ int64_t model_split_check_start_time = ggml_time_us();
    ତାର୍କିକ model_is_splitted = ଚାବି_seen ? ମିଥ୍ୟା : is_model_splitted(cgraph);
    ଅଚଳ int64_t model_split_check_time = ggml_time_us() - model_split_check_start_time;

    ଯଦି (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        ଫେରନ୍ତୁ GGML_STATUS_FAILED;
    }
    ଯଦି (is_naive(cgraph)) {
        ଯଦି (!model_is_splitted) {
            ଫେରନ୍ତୁ naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    ଅଚଳ int64_t graph_param_compute_start_time = ggml_time_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    ଅଚଳ int64_t graph_param_compute_time = ggml_time_us() - graph_param_compute_start_time;

    ଅଚଳ ତାର୍କିକ cache_enabled = !model_is_splitted && !cache_disabled;
    ଅଚଳ ତାର୍କିକ stateful_recurrent = r_ctx->stateful && m_params.state_size >= 0;
    ଅଚଳ ତାର୍କିକ stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    ଯଦି (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_active > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        ଫେରନ୍ତୁ GGML_STATUS_FAILED;
    }
    ଯଦି (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        ଫେରନ୍ତୁ GGML_STATUS_FAILED;
    }
    ତାର୍କିକ cache_hit = ମିଥ୍ୟା;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        ଅଚଳ int64_t cache_lookup_start_time = ggml_time_us();
        ଯଦି (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            auto it = r_ctx->decoder_cache.find(ଚାବି);
            cache_hit = it != r_ctx->decoder_cache.end();
            ଯଦି (cache_hit) {
                entry = it->second;
            } ଅନ୍ୟଥା {
                r_ctx->clear_caches_locked();
                auto mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[ଚାବି] = entry;
            }
        } ଅନ୍ୟଥା {
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_hit = ମିଥ୍ୟା;
        }
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;

        std::lock_guard<std::mutex> lock(*(entry->mutex));
        cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(ଚାବି) != 0;

        ଯଦି (cache_hit) {
            ggml_decoder = entry->ptr;
            old_m_params = ggml_decoder->get_model_params();
            ଯଦି (!ggml_decoder->is_splited_model()) {
                cache_hit = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_ନିବେଶ_names;
        std::vector<std::string> ov_output_names;

        ଯଦି (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            ଯଦି (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(ଚାବି);
                ov_ନିବେଶ_names = r_ctx->ov_ନିବେଶ_names_cache.at(ଚାବି);
                ov_output_names = r_ctx->ov_output_names_cache.at(ଚାବି);
            }

            ଯଦି (stateful_kv_only) {
                ଅଚଳ auto * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_data = (int32_t *) inp_pos->data;
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                ଯଦି (pos_data[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_size = pos_shape[3];
                } ଅନ୍ୟଥା ଯଦି (r_ctx->stateful_kv_size == static_cast<size_t>(pos_data[0])) {
                    r_ctx->stateful_kv_size += pos_shape[3];
                } ଅନ୍ୟଥା {
                    ଅଚଳ size_t pos_begin = static_cast<size_t>(pos_data[0]);
                    ଅଚଳ ତାର୍କିକ refill = pos_begin > r_ctx->stateful_kv_size;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    ଯଦି (refill && !ggml_decoder->get_model_params().swa_layers.empty()) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_swa = ggml_decoder->get_compute_params().swa_window;
                        ଯଦି (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_size, n_swa);
                            ଫେରନ୍ତୁ GGML_STATUS_FAILED;
                        }
                    }

                    ଅଚଳ ତାର୍କିକ relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    auto states = infer_request->query_state();
                    ପାଇଁ (auto state : states) {
                        auto state_tensor = state.get_state();
                        auto state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        ଯଦି (auto it = r_ctx->kv_state_ନିବେଶ_name_map.find(state.get_name());
                            it != r_ctx->kv_state_ନିବେଶ_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        ଯଦି (auto layer = extract_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        ଅଚଳ ତାର୍କିକ relayout_this_state = relayout_enabled;
                        ଅଚଳ size_t seq_axis = relayout_this_state ? 2 : 1;
                        ଅଚଳ size_t head_axis = seq_axis == 2 ? 1 : 2;

                        ଯଦି (refill) {
                            ଯଦି (state_name.empty()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                ଫେରନ୍ତୁ GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_ନିବେଶ_tensor(ggml_decoder, state_name);
                            ଯଦି (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_rows_to_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } ଅନ୍ୟଥା {
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
                        ଯଦି (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            ଫେରନ୍ତୁ GGML_STATUS_FAILED;
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
        } ଅନ୍ୟଥା {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            auto shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
            auto weight_names = get_weight_names(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ଅଚଳ std::string model_cache_dir = ggml_openvino_model_cache_dir();
            ଅଚଳ std::string exact_ଚାବି = cache_enabled ? compiled_graph_ଚାବି(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_path, manifest_path;
            ଯଦି (!model_cache_dir.empty() && !model_is_splitted) {
                ଅଚଳ uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/ସତ୍ୟ, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_graph_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/ସତ୍ୟ, m_params.rope_params, 16, extra_cfg,
                                                    compiled_graph_ଚାବି(cgraph, *ggml_decoder, device, 0, ସତ୍ୟ));
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            std::string shared_ଚାବି =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_ଚାବି) : "";
            ov::CompiledModel shared_model;
            ତାର୍କିକ imported = ମିଥ୍ୟା;
            auto shared_it = shared_cache->graphs.find(shared_ଚାବି);
            ଯଦି (!shared_ଚାବି.empty() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_graph(shared_it->second.decode, shared_it->second.ନିବେଶ_names,
                                                               shared_it->second.output_names, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_ନିବେଶ_names = shared_it->second.ନିବେଶ_names;
                ov_output_names = shared_it->second.output_names;
                imported = ସତ୍ୟ;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } ଅନ୍ୟଥା ଯଦି (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_ଚାବି = exact_ଚାବି;
                shared_it = shared_cache->graphs.find(shared_ଚାବି);
                ଯଦି (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_ନିବେଶ_names = shared_it->second.ନିବେଶ_names;
                    ov_output_names = shared_it->second.output_names;
                    imported = ସତ୍ୟ;
                }
                model_fp = exact_fp;
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            ଯଦି (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(ଚାବି);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            ଯଦି (!model_cache_dir.empty()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
            }
            ଯଦି (!imported && !model_cache_dir.empty() && !model_is_splitted) {
                ଅଚଳ uint64_t preferred_fp = model_fp;
                ତାର୍କିକ dynamic_incompatible = preferred_fp == exact_fp;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା candidate = 0; candidate < 2; ++candidate) {
                    ଯଦି (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        ଭାଙ୍ଗ;
                    }
                    ଅଚଳ uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    ଅଚଳ std::string candidate_blob =
                        ggml_openvino_model_cache_blob_path(model_cache_dir, candidate_fp);
                    ଅଚଳ std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_path(model_cache_dir, candidate_fp);
                    std::ifstream blob_in(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    ଯଦି (!blob_in.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        ଜାରି;
                    }
                    ଅଚଳ int64_t import_start = ggml_time_us();
                    ଚେଷ୍ଟା {
                        ov::CompiledModel cm;
                        auto remote_context = ggml_openvino_get_remote_context();
                        ଯଦି (remote_context.has_value()) {
                            cm = core.import_model(blob_in, remote_context.value(), mc_config);
                        } ଅନ୍ୟଥା {
                            cm = core.import_model(blob_in, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        ଯଦି (!saved_inputs.empty() || !saved_outputs.empty()) {
                            ov_ନିବେଶ_names = std::move(saved_inputs);
                            ov_output_names = std::move(saved_outputs);
                        } ଅନ୍ୟଥା {
                            ପାଇଁ (ଅଚଳ auto & p : cm.inputs()) {
                                ov_ନିବେଶ_names.push_back(p.get_ଗଣ୍ଠି()->get_friendly_name());
                            }
                            ପାଇଁ (ଅଚଳ auto & o : cm.outputs()) {
                                ov_output_names.push_back(o.get_ଗଣ୍ଠି()->get_friendly_name());
                            }
                        }
                        ଯଦି (!compiled_model_matches_graph(cm, ov_ନିବେଶ_names, ov_output_names, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_ନିବେଶ_names.clear();
                            ov_output_names.clear();
                            ଜାରି;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        entry->ptr = ggml_decoder;
                        imported = ସତ୍ୟ;
                        model_fp = candidate_fp;
                        blob_path = candidate_blob;
                        manifest_path = candidate_manifest;
                        ଯଦି (candidate_fp == exact_fp) {
                            shared_ଚାବି = exact_ଚାବି;
                            shared_it = shared_cache->graphs.find(shared_ଚାବି);
                        }
                        ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_time_us() - import_start) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } ଧର (ଅଚଳ std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = ମିଥ୍ୟା;
                        ov_ନିବେଶ_names.clear();
                        ov_output_names.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                ଯଦି (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_ଚାବି = exact_ଚାବି;
                    shared_it = shared_cache->graphs.find(shared_ଚାବି);
                    blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                    manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            ଯଦି (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_path.c_str());
                ଫେରନ୍ତୁ GGML_STATUS_FAILED;
            }
            ଯଦି (!imported && ggml_openvino_weight_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                ଫେରନ୍ତୁ GGML_STATUS_FAILED;
            }
            ଯଦି (imported) {
                decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            } ଅନ୍ୟଥା {
                auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_end_time = ggml_time_us();

                auto ନିବେଶ_model = std::make_shared<ov::frontend::ggml::ନିବେଶ_Model>(ggml_decoder);
                model = ov::frontend::ggml::FrontEnd::convert(ନିବେଶ_model);
                ggml_decoder->clear_model_weights();
                conversion_end_time = ggml_time_us();

                ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    ଅକ୍ଷର timestamped_filename[64];
                    auto timestamp = (long long) ggml_time_us();
                    snprintf(timestamped_filename, ଆକାର(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                ଅଚଳ ov::AnyMap & compile_config = model_cache_dir.empty() ? config : mc_config;
                ov::CompiledModel compiled_model;
                auto remote_context = ggml_openvino_get_remote_context();
                ଯଦି (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } ଅନ୍ୟଥା {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_end_time = ggml_time_us();

                ପାଇଁ (ଅଚଳ auto & ov_param : model->get_parameters()) {
                    ov_ନିବେଶ_names.push_back(ov_param->get_friendly_name());
                }
                ପାଇଁ (ଅଚଳ auto & ov_output : model->get_results()) {
                    ov_output_names.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                ଯଦି (!model_cache_dir.empty() && !model_is_splitted && model_fp != 0) {
                    ଚେଷ୍ଟା {
                        ଅଚଳ std::string blob_tmp = ggml_openvino_model_cache_temp_path(blob_path);
                        ଅଚଳ std::string manifest_tmp = ggml_openvino_model_cache_temp_path(manifest_path);
                        ଯଦି (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_ନିବେଶ_names,
                                                                     ov_output_names)) {
                            std::ofstream blob_out(blob_tmp, std::ios::binary | std::ios::trunc);
                            ଯଦି (blob_out.is_open()) {
                                compiled_model.export_model(blob_out);
                                blob_out.close();
                                ଯଦି (blob_out.good()) {
                                    ଯଦି (std::rename(blob_tmp.c_str(), blob_path.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_path.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_path.c_str());
                                    } ଅନ୍ୟଥା {
                                        std::remove(blob_tmp.c_str());
                                        std::remove(manifest_tmp.c_str());
                                    }
                                } ଅନ୍ୟଥା {
                                    std::remove(blob_tmp.c_str());
                                    std::remove(manifest_tmp.c_str());
                                }
                            } ଅନ୍ୟଥା {
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    } ଧର (ଅଚଳ std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                entry->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            entry->ptr = ggml_decoder;
            ଯଦି (!shared_ଚାବି.empty() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_ଚାବି,
                                             ov_compiled_graph{shared_model, {}, ov_ନିବେଶ_names, ov_output_names});
            }
            ଯଦି (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[ଚାବି] = infer_request;
                r_ctx->ov_ନିବେଶ_names_cache[ଚାବି] = ov_ନିବେଶ_names;
                r_ctx->ov_output_names_cache[ଚାବି] = ov_output_names;
            }

            ଯଦି (stateful_kv_only && cache_enabled) {
                ଅଚଳ auto * inp_pos = get_inp_pos_tensor(cgraph);
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                ଅଚଳ int32_t pos_begin = ((int32_t *) inp_pos->data)[0];
                ଯଦି (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    ଫେରନ୍ତୁ GGML_STATUS_FAILED;
                }
                r_ctx->stateful_kv_size = pos_shape[3];
                ଅଚଳ auto kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                ପାଇଁ (ଅଚଳ auto & pair : kv_param_res_names) {
                    r_ctx->kv_state_ନିବେଶ_name_map[pair.first + pair.second] = pair.first;
                }
            }
        }

        ଯଦି (stateful_recurrent) {
            ଅଚଳ auto * inp_pos = get_inp_pos_tensor(cgraph);
            ଅଚଳ int32_t pos_begin = static_cast<ଅଚଳ int32_t *>(inp_pos->data)[0];
            ଯଦି (pos_begin == 0) {
                infer_request->reset_state();
            } ଅନ୍ୟଥା ଯଦି (!cache_hit || old_m_params.kv_buffer_changed(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_size) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                ଫେରନ୍ତୁ GGML_STATUS_FAILED;
            }
        } ଅନ୍ୟଥା {
            reset_single_slot_recurrent_cache(cgraph, m_params, c_params);
        }

        ପାଇଁ (size_t i = 0; i < ov_ନିବେଶ_names.size(); i++) {
            ଅଚଳ auto & param_name = ov_ନିବେଶ_names[i];
            auto ନିବେଶ_tensor = get_ov_ନିବେଶ_tensor(ggml_decoder, param_name);
            infer_request->set_ନିବେଶ_tensor(i, ନିବେଶ_tensor);

            ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_ନିବେଶ_tensor_info(param_name, ନିବେଶ_tensor);
            }
        }

        ପାଇଁ (size_t i = 0; i < ov_output_names.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            ଅଚଳ auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names[i]);
            ଯଦି (model_output_it == model_outputs.end()) {
                ଜାରି;
            }
            auto * ggml_tensor = model_output_it->second;
            ଯଦି (ggml_nbytes(ggml_tensor) == 0) {
                ଜାରି;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        ଯଦି (stateful_recurrent) {
            ଅଚଳ auto * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_size =
                static_cast<ଅଚଳ int32_t *>(inp_pos->data)[0] + get_inp_pos_n_tokens(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            ପାଇଁ (size_t i = 0; i < ov_output_names.size(); i++) {
                ଅଚଳ auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }

        ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_ଚାବି_compute_time / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_split_check_time / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", graph_param_compute_time / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_time / 1000.0);
            ଯଦି (!cache_hit) {
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
    ଯଦି (cache_hit && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_lock(r_ctx->compiled_cache->mutex);
        ଯଦି (!ggml_openvino_weight_buffers_released()) {
            ggml_openvino_release_weight_buffers();
        }
    }

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ov::AnyMap without_npuw(ଅଚଳ ov::AnyMap & config) {
    ov::AnyMap out;
    ପାଇଁ (ଅଚଳ auto & kv : config) {
        ଯଦି (kv.first.rfind("NPUW", 0) == 0 || kv.first == "NPU_USE_NPUW") {
            ଜାରି;
        }
        out.insert(kv);
    }
    ଫେରନ୍ତୁ out;
}

ଗଣନା ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, ଅଚଳ std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();

    auto get_prefill_chunk_size = [] {
        ସ୍ଥିର ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_size = []() {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା env_prefill_chunk_size = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            ଫେରନ୍ତୁ env_prefill_chunk_size > 0 ? env_prefill_chunk_size : 256;
        }();
        ଫେରନ୍ତୁ chunk_size;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    ସ୍ଥିର std::string device = ggml_openvino_get_device_name();
    ସ୍ଥିର auto is_static = ସତ୍ୟ;
    ସ୍ଥିର auto stateful = ମିଥ୍ୟା;

    auto prefill_chunk_size = get_prefill_chunk_size();
    ଅଚଳ auto & config = ggml_openvino_get_compile_config();

    ଯଦି (is_naive(cgraph)) {
        ଫେରନ୍ତୁ naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    ଅଚଳ auto * inp_pos = get_inp_pos_tensor(cgraph);
    ଅଚଳ ତାର୍କିକ no_kv_cache = m_params.is_cacheless_attn;
    ଅଚଳ auto is_prefill = no_kv_cache ? ସତ୍ୟ : get_is_prefill(cgraph, inp_pos);
    ଅଚଳ ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    ଯଦି (m_params.n_heads_kv == -1) {
        prefill_chunk_size = inp_pos->ne[0];
    }
    graph_ଚାବି ଚାବି(cgraph);
    ସ୍ଥିର ଅଚଳ ତାର୍କିକ cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    ତାର୍କିକ cache_hit = ମିଥ୍ୟା;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;
    int64_t ov_raw_infer_total = 0;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    ଯଦି (cache_enabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        auto it = r_ctx->decoder_cache.find(ଚାବି);
        cache_hit = it != r_ctx->decoder_cache.end();
        ଯଦି (cache_hit) {
            entry = it->second;
        } ଅନ୍ୟଥା {
            r_ctx->clear_caches_locked();
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[ଚାବି] = entry;
        }
    } ଅନ୍ୟଥା {
        auto mutex = std::make_shared<std::mutex>();
        entry = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_hit = ମିଥ୍ୟା;
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));
    cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(ଚାବି) != 0 &&
                r_ctx->infer_request_cache_prefill.count(ଚାବି) != 0;

    ଯଦି (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_ନିବେଶ_names_local;
    std::vector<std::string> ov_output_names_local;

    ଯଦି (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        ଯଦି (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(ଚାବି) : r_ctx->infer_request_cache.at(ଚାବି);
            ov_ନିବେଶ_names_local = r_ctx->ov_ନିବେଶ_names_cache.at(ଚାବି);
            ov_output_names_local = r_ctx->ov_output_names_cache.at(ଚାବି);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } ଅନ୍ୟଥା {
        ଯଦି (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(ଚାବି);
            r_ctx->infer_request_cache_prefill.erase(ଚାବି);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        auto shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
        auto weight_names = get_weight_names(cgraph);
        auto local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names, is_static,
                                                             stateful, ମିଥ୍ୟା, is_prefill, prefill_chunk_size);
        ଅଚଳ std::string shared_ଚାବି =
            cache_enabled ? compiled_graph_ଚାବି(cgraph, *local_decoder, device, prefill_chunk_size) : "";
        auto shared_it = shared_cache->graphs.find(shared_ଚାବି);
        ଯଦି (!shared_ଚାବି.empty() && shared_it != shared_cache->graphs.end()) {
            auto & compiled = shared_it->second;
            auto prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            auto decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            entry->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_ନିବେଶ_names_local = compiled.ନିବେଶ_names;
            ov_output_names_local = compiled.output_names;
            r_ctx->infer_request_cache_prefill[ଚାବି] = prefill_request;
            r_ctx->infer_request_cache[ଚାବି] = decode_request;
            r_ctx->ov_ନିବେଶ_names_cache[ଚାବି] = ov_ନିବେଶ_names_local;
            r_ctx->ov_output_names_cache[ଚାବି] = ov_output_names_local;
            decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } ଅନ୍ୟଥା {
            std::shared_ptr<ov::Model> model;
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

            auto ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, ମିଥ୍ୟା, ସତ୍ୟ, prefill_chunk_size);
            auto ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, ମିଥ୍ୟା, ମିଥ୍ୟା, prefill_chunk_size);
            decoder_end_time = ggml_time_us();

            ଅଚଳ ତାର୍କିକ dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            ଅଚଳ auto dump_ir_timestamp = static_cast<long long>(ggml_time_us());

            auto build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          ଅଚଳ std::shared_ptr<GgmlOvDecoder> & decoder, ଅଚଳ ଅକ୍ଷର * tag,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_end_time, int64_t & local_compile_end_time) {
                auto ନିବେଶ_model = std::make_shared<ov::frontend::ggml::ନିବେଶ_Model>(decoder);
                model = ov::frontend::ggml::FrontEnd::convert(ନିବେଶ_model);
                decoder->clear_model_weights();
                local_conversion_end_time = ggml_time_us();

                ଯଦି (dump_ir) {
                    ଅକ୍ଷର timestamped_filename[64];
                    snprintf(timestamped_filename, ଆକାର(timestamped_filename), "model_%s_%lld.xml", tag,
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
            ଯଦି (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_end_time = prefill_conversion_end_time;
                decode_compile_end_time = prefill_compile_end_time;
            } ଅନ୍ୟଥା {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_end_time, decode_compile_end_time);
            }
            conversion_end_time = std::max(prefill_conversion_end_time, decode_conversion_end_time);
            compile_end_time = std::max(prefill_compile_end_time, decode_compile_end_time);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            entry->ptr = ggml_decoder;

            ପାଇଁ (ଅଚଳ auto & ov_param : model->get_parameters()) {
                ov_ନିବେଶ_names_local.push_back(ov_param->get_friendly_name());
            }
            ପାଇଁ (ଅଚଳ auto & ov_output : model->get_results()) {
                ov_output_names_local.push_back(ov_output->get_friendly_name());
            }

            ଯଦି (!shared_ଚାବି.empty()) {
                shared_cache->graphs.emplace(
                    shared_ଚାବି, ov_compiled_graph{compiled_model_decode, compiled_model_prefill, ov_ନିବେଶ_names_local,
                                                  ov_output_names_local});
            }

            ଯଦି (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[ଚାବି] = infer_request_prefill;
                r_ctx->infer_request_cache[ଚାବି] = infer_request_decode;
                r_ctx->ov_ନିବେଶ_names_cache[ଚାବି] = ov_ନିବେଶ_names_local;
                r_ctx->ov_output_names_cache[ଚାବି] = ov_output_names_local;
            }
        }
    }

    reset_single_slot_recurrent_cache(cgraph, m_params, c_params);

    ଯଦି (is_prefill) {
        auto inp_len = get_inp_pos_n_tokens(cgraph, inp_pos);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            ପାଇଁ (size_t i = 0; i < ov_ନିବେଶ_names_local.size(); i++) {
                ଅଚଳ auto & param_name = ov_ନିବେଶ_names_local[i];
                auto ନିବେଶ_tensor = get_ov_ନିବେଶ_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_ନିବେଶ_tensor(i, ନିବେଶ_tensor);

                ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    ଅଚଳ auto ନିବେଶ_tensor = infer_request->get_ନିବେଶ_tensor(i);
                    print_ନିବେଶ_tensor_info(param_name, ନିବେଶ_tensor);
                }
            }

            ପାଇଁ (size_t i = 0; i < ov_output_names_local.size(); i++) {
                ଅଚଳ auto & model_outputs = ggml_decoder->get_model_outputs();
                auto model_output_it = model_outputs.find(ov_output_names_local[i]);
                ଯଦି (model_output_it == model_outputs.end()) {
                    ଜାରି;
                }
                auto * ggml_tensor = model_output_it->second;
                ଯଦି (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    ଜାରି;
                }
                auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            ov_raw_infer_start = ggml_time_us();
            infer_request->infer();
            ov_raw_infer_total += ggml_time_us() - ov_raw_infer_start;

            ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                ପାଇଁ (size_t i = 0; i < ov_output_names_local.size(); i++) {
                    ଅଚଳ auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } ଅନ୍ୟଥା {
        ପାଇଁ (size_t i = 0; i < ov_ନିବେଶ_names_local.size(); i++) {
            ଅଚଳ auto & param_name = ov_ନିବେଶ_names_local[i];
            auto ନିବେଶ_tensor = get_ov_ନିବେଶ_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_ନିବେଶ_tensor(i, ନିବେଶ_tensor);

            ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                ଅଚଳ auto ନିବେଶ_tensor = infer_request->get_ନିବେଶ_tensor(i);
                print_ନିବେଶ_tensor_info(param_name, ନିବେଶ_tensor);
            }
        }

        ପାଇଁ (size_t i = 0; i < ov_output_names_local.size(); i++) {
            ଅଚଳ auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names_local[i]);
            ଯଦି (model_output_it == model_outputs.end()) {
                ଜାରି;
            }
            auto * ggml_tensor = model_output_it->second;
            ଯଦି (ggml_nbytes(ggml_tensor) == 0) {
                ଜାରି;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        ov_raw_infer_total = infer_end_time - ov_raw_infer_start;

        ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            ପାଇଁ (size_t i = 0; i < ov_output_names_local.size(); i++) {
                ଅଚଳ auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
            }
        }
    }

    ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
        ଯଦି (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_end_time - decoder_end_time) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_total / 1000.0);
    }

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
ସ୍ଥିର ggml_cgraph * get_compute_graph(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    auto is_skipped = [](ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
        ଫେରନ୍ତୁ ଗଣ୍ଠି->op != GGML_OP_NONE && !(ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    ଯଦି (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        ଫେରନ୍ତୁ cgraph;
    }
    auto & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](ଅଚଳ ggml_tensor * ଗଣ୍ଠି) { ଫେରନ୍ତୁ !is_skipped(ଗଣ୍ଠି); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.data();
    compute.graph.n_nodes = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) compute.nodes.size();
    ଫେରନ୍ତୁ &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
ଗଣନା ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    ଚେଷ୍ଟା {
        ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        ଅଚଳ auto is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_lock(r_ctx->execution_mutex);
        cgraph = get_compute_graph(cgraph, *r_ctx);

        ଫେରନ୍ତୁ is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } ଧର (ଅଚଳ ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        ଫେରନ୍ତୁ GGML_STATUS_FAILED;
    } ଧର (ଅଚଳ std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        ଫେରନ୍ତୁ GGML_STATUS_FAILED;
    } ଧର (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        ଫେରନ୍ତୁ GGML_STATUS_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
ତାର୍କିକ is_model_splitted(ggml_cgraph * cgraph) {
    ସ୍ଥିର ଅଚଳ ତାର୍କିକ fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    ଯଦି (!fallback_enabled) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    ଯଦି (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା use_ଗଣନା = cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, ଗଣ୍ଠି)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        ଯଦି ((cgraph->n_nodes <= 1 && use_ଗଣନା == 0) ||
            (cgraph->n_nodes <= 1 && ଗଣ୍ଠି->op == GGML_OP_VIEW && use_ଗଣନା == 1 && ଗଣ୍ଠି->src[0] != nullptr &&
             ଗଣ୍ଠି->src[0]->op == GGML_OP_NONE)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_PERMUTE)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_use_ଗଣନା = 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_ଗଣ୍ଠି = cgraph->nodes[j];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < GGML_MAX_SRC; k++) {
                ଯଦି (other_ଗଣ୍ଠି->src[k] == ଗଣ୍ଠି) {
                    ନିବେଶ_use_ଗଣନା++;
                }
            }
        }
        ଯଦି (use_ଗଣନା != ନିବେଶ_use_ଗଣନା && ଗଣ୍ଠି->op != GGML_OP_NONE) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    ଯଦି (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_weight_names(cgraph);
    } ଅନ୍ୟଥା {
        ପାଇଁ (ଅଚଳ auto & kv : GgmlOvDecoder::create_weight_nodes(cgraph, ସତ୍ୟ)) {
            model_weights.insert(kv.first);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = ଗଣ୍ଠି->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            ଯଦି (src != nullptr && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.empty() == ମିଥ୍ୟା &&
                model_leafs.find(src) == model_leafs.end()) {
                ଯଦି (GgmlOvDecoder::is_inp_tok(src, ଗଣ୍ଠି)) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ତାର୍କିକ is_naive(ggml_cgraph * cgraph) {
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା naive_graph_size_threshold = 20;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା count = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ଯଦି (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    ଫେରନ୍ତୁ count < naive_graph_size_threshold;
}

size_t checksum(ଅଚଳ ଶୂନ୍ୟ * data, size_t size) {
    ଅଚଳ uint8_t * bytes = static_cast<ଅଚଳ uint8_t *>(data);
    size_t sum = 0;
    ପାଇଁ (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    ଫେରନ୍ତୁ sum;
}

ତାର୍କିକ save_ggml_tensor_data_to_txt(ଅଚଳ ggml_tensor * tensor, ଅଚଳ std::string & file_path) {
    ଯଦି (tensor == nullptr || tensor->data == nullptr) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    std::ofstream out(file_path);
    ଯଦି (!out.is_open()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ size_t n = ggml_nelements(tensor);
    out << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    ବାଛ (tensor->type) {
    ସ୍ଥିତି GGML_TYPE_F32: {
        ଅଚଳ auto * data = static_cast<ଅଚଳ ଦଶମିକ *>(tensor->data);
        ପାଇଁ (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_TYPE_F16: {
        ଅଚଳ auto * data = static_cast<ଅଚଳ ggml_fp16_t *>(tensor->data);
        ପାଇଁ (size_t i = 0; i < n; ++i) {
            out << ggml_fp16_to_fp32(data[i]) << '\n';
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_TYPE_BF16: {
        ଅଚଳ auto * data = static_cast<ଅଚଳ ggml_bf16_t *>(tensor->data);
        ପାଇଁ (size_t i = 0; i < n; ++i) {
            out << ggml_bf16_to_fp32(data[i]) << '\n';
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_TYPE_I32: {
        ଅଚଳ auto * data = static_cast<ଅଚଳ int32_t *>(tensor->data);
        ପାଇଁ (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି GGML_TYPE_I64: {
        ଅଚଳ auto * data = static_cast<ଅଚଳ int64_t *>(tensor->data);
        ପାଇଁ (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        ଭାଙ୍ଗ;
    }
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        out << "unsupported tensor type for text dump" << '\n';
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ଶୂନ୍ୟ print_ନିବେଶ_tensor_info(ଅଚଳ std::string & name, ଅଚଳ ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << '\n';
    ବାଛ (tensor.get_element_type()) {
    ସ୍ଥିତି ov::element::f32: {
        ଯଦି (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::cout << *(tensor.data<ଦଶମିକ>()) << '\n';
        } ଅନ୍ୟଥା {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            ଅଚଳ ଦଶମିକ * data = tensor.data<ଦଶମିକ>();
            ପାଇଁ (size_t i = 0; i < rows; ++i) {
                ପାଇଁ (size_t j = 0; j < cols; ++j) {
                    ଦଶମିକ val = data[i * cols + j];
                    ଯଦି (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } ଅନ୍ୟଥା {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << '\n';
            }
        }

        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << '\n';
        ଭାଙ୍ଗ;
    ସ୍ଥିତି ov::element::i32:
        ପାଇଁ (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << ' ';
        }
        std::cout << '\n';
        ଭାଙ୍ଗ;
    ସ୍ଥିତି ov::element::i64:
        ପାଇଁ (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << ' ';
        }
        std::cout << '\n';
        ଭାଙ୍ଗ;
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        ଭାଙ୍ଗ;
    }
}

ଶୂନ୍ୟ print_output_tensor_info(ଅଚଳ std::string & name, ଅଚଳ ov::Tensor & tensor, ଅଚଳ ଶୂନ୍ୟ * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    auto print_float_stats = [](ଅଚଳ std::string & type_name, size_t size, auto get_ମୂଲ୍ୟ) {
        ଯଦି (size == 0) {
            ଫେରନ୍ତୁ;
        }

        ଦଶମିକ first = get_ମୂଲ୍ୟ(0);
        ଦଶମିକ min = first;
        ଦଶମିକ max = first;
        ଦ୍ୱିଗୁଣ sum = first;

        ପାଇଁ (size_t i = 1; i < size; ++i) {
            ଦଶମିକ v = get_ମୂଲ୍ୟ(i);
            min = std::min(v, min);
            max = std::max(v, max);
            sum += v;
        }
        ଦ୍ୱିଗୁଣ mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << '\n';
    };

    ବାଛ (tensor.get_element_type()) {
    ସ୍ଥିତି ov::element::f32: {
        ଅଚଳ ଦଶମିକ * data = tensor.data<ଦଶମିକ>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { ଫେରନ୍ତୁ data[i]; });
        ଭାଙ୍ଗ;
    }
    ସ୍ଥିତି ov::element::f16: {
        ଅଚଳ ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { ଫେରନ୍ତୁ static_cast<ଦଶମିକ>(data[i]); });
        ଭାଙ୍ଗ;
    }
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        ଭାଙ୍ଗ;
    }
}

ଅଚଳ ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            ଯଦି (src == nullptr) {
                ଭାଙ୍ଗ;
            }
            ଯଦି (GgmlOvDecoder::is_inp_pos(src, op)) {
                ଫେରନ୍ତୁ src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    ଫିଙ୍ଗ std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_tokens(ggml_cgraph * cgraph, ଅଚଳ ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_planes = 1;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; ++j) {
            ଯଦି (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                ଭାଙ୍ଗ;
            }
        }
    }
    ଫେରନ୍ତୁ inp_pos->ne[0] / n_planes;
}

ତାର୍କିକ get_is_prefill(ggml_cgraph * cgraph, ଅଚଳ ggml_tensor * inp_pos) {
    ଫେରନ୍ତୁ get_inp_pos_n_tokens(cgraph, inp_pos) > 1;
}
