#அந்தர்பாவ "utils.h"

#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-openvino-extra.h"
#அந்தர்பாவ "ggml-openvino.h"
#அந்தர்பாவ "ggml-openvino/ggml-decoder.h"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "model-cache.h"
#அந்தர்பாவ "openvino/frontend.h"
#அந்தர்பாவ "openvino/input_model.h"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cassert>
#அந்தர்பாவ <cmath>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <fstream>
#அந்தர்பாவ <functional>
#அந்தர்பாவ <iomanip>
#அந்தர்பாவ <iostream>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/any.hpp>
#அந்தர்பாவ <openvino/core/graph_util.hpp>
#அந்தர்பாவ <openvino/core/shape.hpp>
#அந்தர்பாவ <openvino/core/type/float16.hpp>
#அந்தர்பாவ <openvino/frontend/manager.hpp>
#அந்தர்பாவ <openvino/openvino.hpp>
#அந்தர்பாவ <openvino/runtime/compiled_model.hpp>
#அந்தர்பாவ <openvino/runtime/infer_request.hpp>
#அந்தர்பாவ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#அந்தர்பாவ <openvino/runtime/intel_npu/properties.hpp>
#அந்தர்பாவ <openvino/runtime/properties.hpp>
#அந்தர்பாவ <openvino/runtime/tensor.hpp>
#அந்தர்பாவ <optional>
#அந்தர்பாவ <set>
#அந்தர்பாவ <string>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <vector>

நாமாகாஶ {
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
std::optional<ov::Tensor> try_make_kv_sliced_tensor(மாறிலி std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    மாறிலி std::string & name,
                                                    மாறிலி ggml_tensor * ggml_tensor) {
    நிலையான மாறிலி தருக்க kv_slice_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    எனில் (kv_slice_disabled) {
        திரும்பு std::nullopt;
    }
    எனில் (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        திரும்பு std::nullopt;
    }
    எனில் (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != nullptr) {
        திரும்பு std::nullopt;
    }
    மாறிலி auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);
    எனில் (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        திரும்பு std::nullopt;
    }

    மாறிலி auto & compute_params = ggml_decoder->get_compute_params();
    எனில் (compute_params.n_seq_active != 1 || compute_params.seq_active_start != 0) {
        திரும்பு std::nullopt;
    }

    முழு layer;
    எனில் (auto layer_opt = extract_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } இல்லையெனில் {
        திரும்பு std::nullopt;
    }

    மாறிலி தருக்க is_swa = ggml_decoder->is_swa_layer(layer);
    எனில் (is_swa) {
        திரும்பு std::nullopt;
    }
    மாறிலி முழு ctx_per_seq = ggml_decoder->get_ctx_per_seq();
    மாறிலி முழு n_kv = compute_params.attention_size;
    எனில் (ctx_per_seq <= 0 || n_kv <= 0 || n_kv >= ctx_per_seq) {
        திரும்பு std::nullopt;
    }

    ov::Shape full_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    எனில் (full_shape.size() != 4 || full_shape[0] != 1 || full_shape[1] != 1 ||
        static_cast<முழு>(full_shape[2]) != ctx_per_seq) {
        திரும்பு std::nullopt;
    }

    ov::Shape sliced_shape = full_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    எனில் (ggml_openvino_buffer_is_remote(ggml_tensor)) {
        auto remote_context = ggml_openvino_get_remote_context();
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        திரும்பு gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
    }

    திரும்பு ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
}

நிலையான uint64_t ggml_openvino_model_cache_extra_cfg(தருக்க stateful, தருக்க recurrent) {
    மாறிலி எழுத்து * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    மாறிலி தருக்க manual_gqa_enabled = manual_gqa_env != nullptr ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    திரும்பு extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_weight_names(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> names;
    க்கு (மாறிலி auto & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
        names[name] = nullptr;
    }
    திரும்பு names;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
நிலையான std::string compiled_graph_சாவி(மாறிலி ggml_cgraph * graph,
                                      மாறிலி GgmlOvDecoder & decoder,
                                      மாறிலி std::string & device,
                                      முழு prefill_chunk_size = 0,
                                      தருக்க disk_cache = பொய்) {
    std::string சாவி;
    auto append = [&சாவி](மாறிலி auto & value) {
        சாவி.append(reinterpret_cast<மாறிலி எழுத்து *>(&value), அளவு(value));
    };
    auto append_string = [&](மாறிலி std::string & value) {
        append(value.size());
        சாவி.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_size);
    தருக்க has_weight_buffer_id = பொய்;
    std::unordered_map<மாறிலி ggml_tensor *, size_t> ids;
    std::function<வெறுமை(மாறிலி ggml_tensor *)> visit = [&](மாறிலி ggml_tensor * tensor) {
        எனில் (!tensor) {
            append(size_t(0));
            திரும்பு;
        }
        auto inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.first->second);
        எனில் (!inserted.second) {
            திரும்பு;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        மாறிலி auto * base = tensor->view_src ? tensor->view_src : tensor;
        மாறிலி தருக்க weight = base->buffer && base->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        append(weight);
        எனில் (weight && !disk_cache) {
            மாறிலி size_t buffer_id = ggml_backend_openvino_buffer_get_ctx_id(base->buffer);
            has_weight_buffer_id |= buffer_id != 0;
            append(buffer_id);
            append(tensor->data);
        }
        visit(tensor->view_src);
        க்கு (மாறிலி auto * src : tensor->src) {
            visit(src);
        }
    };
    append(graph->n_nodes);
    க்கு (முழு i = 0; i < graph->n_nodes; ++i) {
        visit(graph->nodes[i]);
    }
    append(graph->n_leafs);
    க்கு (முழு i = 0; i < graph->n_leafs; ++i) {
        visit(graph->leafs[i]);
    }
    க்கு (மாறிலி auto & உள்ளீடு : decoder.get_model_extra_inputs()) {
        append_string(உள்ளீடு.first);
        append_string(உள்ளீடு.second.type.get_type_name());
        append(உள்ளீடு.second.shape.size());
        க்கு (auto dim : உள்ளீடு.second.shape) {
            append(dim);
        }
        append(உள்ளீடு.second.is_parameter);
        எனில் (!உள்ளீடு.second.is_parameter) {
            append(உள்ளீடு.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    திரும்பு disk_cache || has_weight_buffer_id ? சாவி : std::string{};
}

நிலையான std::string dynamic_graph_signature(மாறிலி ggml_cgraph * graph,
                                           மாறிலி GgmlOvDecoder & decoder,
                                           மாறிலி ModelParams & params) {
    std::string சாவி = "dynamic-1";
    auto append = [&சாவி](மாறிலி auto & value) {
        சாவி.append(reinterpret_cast<மாறிலி எழுத்து *>(&value), அளவு(value));
    };
    auto append_string = [&](மாறிலி std::string & value) {
        append(value.size());
        சாவி.append(value);
    };
    graph_சாவி graph_id(graph, உண்மை);
    append(graph_id.n_nodes);
    append(graph_id.n_leaves);
    append_string(graph_id.first_முனை_name);
    append_string(graph_id.last_முனை_name);
    க்கு (மாறிலி auto & name : graph_id.உள்ளீடு_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    க்கு (முழு layer : params.swa_layers) {
        append(layer);
    }
    க்கு (மாறிலி auto & [layer, heads] : params.n_heads_kv_per_layer) {
        append(layer);
        append(heads);
    }
    க்கு (மாறிலி auto & [name, உள்ளீடு] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(உள்ளீடு.type.get_type_name());
    }
    க்கு (மாறிலி auto & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    க்கு (மாறிலி auto & [name, உள்ளீடு] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(உள்ளீடு.type.get_type_name());
        append(உள்ளீடு.is_parameter);
        எனில் (!உள்ளீடு.is_parameter) {
            append(உள்ளீடு.value);
        }
    }
    திரும்பு சாவி;
}

நிலையான தருக்க compiled_model_matches_graph(மாறிலி ov::CompiledModel & model,
                                         மாறிலி std::vector<std::string> & inputs,
                                         மாறிலி std::vector<std::string> & outputs,
                                         மாறிலி GgmlOvDecoder & decoder) {
    எனில் (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        திரும்பு பொய்;
    }
    மாறிலி auto & graph_inputs = decoder.get_model_inputs();
    மாறிலி auto & extra_inputs = decoder.get_model_extra_inputs();
    மாறிலி auto & graph_outputs = decoder.get_model_outputs();
    க்கு (size_t i = 0; i < inputs.size(); ++i) {
        மாறிலி auto & port = model.உள்ளீடு(i);
        auto graph_it = graph_inputs.find(inputs[i]);
        எனில் (graph_it != graph_inputs.end()) {
            எனில் (port.get_element_type() != graph_it->second.type ||
                !port.get_partial_shape().compatible(graph_it->second.shape)) {
                திரும்பு பொய்;
            }
            தொடர்;
        }
        auto extra_it = extra_inputs.find(inputs[i]);
        எனில் (extra_it == extra_inputs.end() || port.get_element_type() != extra_it->second.type ||
            !port.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            திரும்பு பொய்;
        }
    }
    க்கு (size_t i = 0; i < outputs.size(); ++i) {
        auto graph_it = graph_outputs.find(outputs[i]);
        எனில் (graph_it == graph_outputs.end()) {
            எனில் (outputs[i].compare(0, 8, "__debug_") != 0) {
                திரும்பு பொய்;
            }
            தொடர்;
        }
        மாறிலி auto & port = model.output(i);
        எனில் (port.get_element_type() != GgmlOvDecoder::get_ov_type(graph_it->second) ||
            !port.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(graph_it->second)))) {
            திரும்பு பொய்;
        }
    }
    திரும்பு உண்மை;
}

நிலையான std::string ov_profiling_csv_field(மாறிலி std::string & value) {
    std::string escaped = "\"";
    க்கு (எழுத்து c : value) {
        escaped += c;
        எனில் (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    திரும்பு escaped;
}

நிலையான வெறுமை dump_ov_profiling_info(மாறிலி ov::InferRequest & infer_request) {
    எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        திரும்பு;
    }

    நிலையான std::atomic<uint64_t> inference_index{0};
    மாறிலி uint64_t index = inference_index.fetch_add(1);
    மாறிலி std::string path = "openvino_profile_" + std::to_string(index) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    எனில் (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        திரும்பு;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t total_real_time = 0;
    int64_t total_cpu_time = 0;
    size_t executed_மொத்தம் = 0;
    க்கு (மாறிலி auto & info : infer_request.get_profiling_info()) {
        மாறிலி எழுத்து * status = "NOT_RUN";
        எனில் (info.status == ov::ProfilingInfo::Status::EXECUTED) {
            status = "EXECUTED";
            total_real_time += info.real_time.count();
            total_cpu_time += info.cpu_time.count();
            executed_மொத்தம்++;
        } இல்லையெனில் எனில் (info.status == ov::ProfilingInfo::Status::OPTIMIZED_OUT) {
            status = "OPTIMIZED_OUT";
        }

        output << status << ',' << info.real_time.count() << ',' << info.cpu_time.count() << ','
               << info.start_time.count() << ',' << ov_profiling_csv_field(info.node_type) << ','
               << ov_profiling_csv_field(info.முனை_name) << ',' << ov_profiling_csv_field(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_மொத்தம், total_real_time / 1000.0, total_cpu_time / 1000.0);
}

நிலையான வெறுமை reset_single_slot_recurrent_cache(ggml_cgraph * cgraph,
                                              மாறிலி ModelParams & model_params,
                                              மாறிலி ComputeParams & compute_params) {
    எனில் (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        திரும்பு;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_buffer_t> buffers;
    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        auto * முனை = cgraph->nodes[i];
        எனில் (முனை->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(முனை->view_src) &&
            முனை->view_src->ne[1] == 1 && முனை->view_src->buffer != nullptr) {
            buffers.insert(முனை->view_src->buffer);
        }
    }
    க்கு (auto * buffer : buffers) {
        ggml_backend_buffer_clear(buffer, 0);
    }
}

ov::Tensor create_ov_output_tensor(மாறிலி std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   மாறிலி std::shared_ptr<ov::InferRequest> & infer_request,
                                   முழு output_index,
                                   மாறிலி ggml_tensor * ggml_tensor) {
    எனில் (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        திரும்பு *sliced;
    }

    auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    வெறுமை * output_data = ggml_tensor->data;
    எனில் (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_index).get_shape();
    } இல்லையெனில் {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        எனில் (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != nullptr &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_data = ggml_tensor->view_src->data;
        } இல்லையெனில் {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    எனில் (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_buffer_is_remote(ggml_tensor) &&
        ggml_tensor->extra != nullptr) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        எனில் (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            எனில் (tensor_extra->tensor != nullptr && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                திரும்பு *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_data);
    திரும்பு output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_rows_to_seq_axis_2(மாறிலி ov::Tensor & kv_tensor, size_t n_heads_kv) {
    மாறிலி size_t rows = kv_tensor.get_shape()[2];
    மாறிலி size_t head_size = kv_tensor.get_shape()[3] / n_heads_kv;
    மாறிலி size_t elem = kv_tensor.get_element_type().size();
    மாறிலி size_t head_bytes = head_size * elem;

    ov::Tensor out(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, rows, head_size});
    மாறிலி auto * src = static_cast<மாறிலி uint8_t *>(kv_tensor.data());
    auto * dst = static_cast<uint8_t *>(out.data());
    க்கு (size_t s = 0; s < rows; s++) {
        க்கு (size_t h = 0; h < n_heads_kv; h++) {
            memcpy(dst + (h * rows + s) * head_bytes, src + (s * n_heads_kv + h) * head_bytes, head_bytes);
        }
    }
    திரும்பு out;
}

template <typename T> வெறுமை set_zero_diagonal(std::vector<T> & matrix, size_t rows, size_t cols, T zero_மதிப்பு = T{}) {
    க்கு (size_t i = 0; i < rows; ++i) {
        size_t diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = zero_மதிப்பு;
    }
}

ov::Tensor make_contiguous_split_உள்ளீடு_tensor(மாறிலி கட்டமைப்பு ggml_tensor * ggml_tensor, மாறிலி ov::Shape & உள்ளீடு_shape) {
    மாறிலி size_t element_size = ggml_type_size(ggml_tensor->type);
    மாறிலி size_t block_size = ggml_blck_size(ggml_tensor->type);

    GGML_ASSERT(block_size == 1 && "non-contiguous split inputs must be plain element types");

    மாறிலி கட்டமைப்பு ggml_tensor * source_tensor = ggml_tensor->view_src != nullptr ? ggml_tensor->view_src : ggml_tensor;
    மாறிலி size_t source_offset = ggml_tensor->view_src != nullptr ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> source_data(ggml_nbytes(source_tensor));
    ggml_backend_tensor_get(source_tensor, source_data.data(), 0, source_data.size());

    ov::Tensor உள்ளீடு_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), உள்ளீடு_shape);
    auto * dst = static_cast<uint8_t *>(உள்ளீடு_tensor.data());
    size_t dst_offset = 0;

    க்கு (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        க்கு (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            க்கு (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                க்கு (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    மாறிலி size_t src_offset = source_offset + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::memcpy(dst + dst_offset, source_data.data() + src_offset, element_size);
                    dst_offset += element_size;
                }
            }
        }
    }

    திரும்பு உள்ளீடு_tensor;
}

ov::Tensor convert_ggml_உள்ளீடு_to_ov(மாறிலி std::shared_ptr<GgmlOvDecoder> & ggml_decoder, மாறிலி std::string & name) {
    மாறிலி auto * ggml_tensor = ggml_decoder->get_உள்ளீடு_ggml_tensor(name);

    எனில் (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        திரும்பு *sliced;
    }

    எனில் (ggml_tensor->extra != nullptr && !ggml_decoder->is_splited_model()) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        எனில் (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            திரும்பு *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    auto * உள்ளீடு_data = ggml_tensor->data;
    ov::Shape உள்ளீடு_shape;
    எனில் (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        உள்ளீடு_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } இல்லையெனில் {
        உள்ளீடு_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    எனில் (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        திரும்பு make_contiguous_split_உள்ளீடு_tensor(ggml_tensor, உள்ளீடு_shape);
    }

    auto உள்ளீடு_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), உள்ளீடு_shape, உள்ளீடு_data);
    திரும்பு உள்ளீடு_tensor;
}

ov::Tensor get_ov_உள்ளீடு_tensor(மாறிலி std::shared_ptr<GgmlOvDecoder> & ggml_decoder, மாறிலி std::string & param_name) {
    ov::Tensor உள்ளீடு_tensor;
    auto extra_உள்ளீடு = ggml_decoder->get_model_extra_inputs().find(param_name);
    எனில் (extra_உள்ளீடு != ggml_decoder->get_model_extra_inputs().end()) {
        உள்ளீடு_tensor = ov::Tensor(extra_உள்ளீடு->second.type, extra_உள்ளீடு->second.shape);
        *உள்ளீடு_tensor.data<int64_t>() = extra_உள்ளீடு->second.value;
    } இல்லையெனில் {
        உள்ளீடு_tensor = convert_ggml_உள்ளீடு_to_ov(ggml_decoder, param_name);
    }
    திரும்பு உள்ளீடு_tensor;
}

ov::Tensor get_ov_உள்ளீடு_tensor_static_decode(மாறிலி std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             மாறிலி std::string & param_name) {
    // NPU decoding stage
    எனில் (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        திரும்பு get_ov_உள்ளீடு_tensor(ggml_decoder, param_name);
    }
    மாறிலி auto * ggml_tensor = ggml_decoder->get_உள்ளீடு_ggml_tensor(param_name);
    மாறிலி auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    எனில் (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        மாறிலி முழு n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        ப்ரதிஜ்ஞா(ggml_tensor->ne[0] == n_planes);
        ov::Shape உள்ளீடு_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor உள்ளீடு_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), உள்ளீடு_shape);
        std::memcpy(உள்ளீடு_tensor.data(), ggml_tensor->data, n_planes * ggml_type_size(ggml_tensor->type));
        திரும்பு உள்ளீடு_tensor;
    }

    எனில் (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape உள்ளீடு_shape = {1, 1, 1, 1};
        ov::Tensor உள்ளீடு_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), உள்ளீடு_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        ப்ரதிஜ்ஞா(ggml_tensor->ne[0] == 1);
        ப்ரதிஜ்ஞா(inp_out_id == 0);
        *உள்ளீடு_tensor.data<int32_t>() = inp_out_id;
        திரும்பு உள்ளீடு_tensor;
    }

    எனில் (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t context_size = ggml_decoder->get_ctx_size();
        எனில் (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_data =
                pad_உள்ளீடு<ggml_fp16_t>(ggml_tensor, 1, context_size, GGML_FP32_TO_FP16(-INFINITY));
            ov::Tensor உள்ளீடு_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_size});
            std::memcpy(உள்ளீடு_tensor.data(), padded_data.data(), padded_data.size() * அளவு(ggml_fp16_t));
            திரும்பு உள்ளீடு_tensor;
        }

        std::vector<தசம> padded_data = pad_உள்ளீடு<தசம>(ggml_tensor, 1, context_size, -INFINITY);
        ov::Tensor உள்ளீடு_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        auto * data_ptr = உள்ளீடு_tensor.data<தசம>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
        திரும்பு உள்ளீடு_tensor;
    }

    திரும்பு get_ov_உள்ளீடு_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_உள்ளீடு_tensor_static_prefill(மாறிலி std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              மாறிலி std::string & param_name,
                                              முழு chunk_index) {
    // NPU prompt processing stage
    மாறிலி size_t உள்ளீடு_len = ggml_decoder->get_உள்ளீடு_len();
    மாறிலி size_t chunk_size = ggml_decoder->m_prefill_chunk_size;
    மாறிலி size_t chunk_valid_size = std::min(chunk_size, உள்ளீடு_len - chunk_index * chunk_size);
    மாறிலி size_t chunk_pad_size = chunk_size - chunk_valid_size;

    எனில் (param_name == "chunk_valid_len") {
        ov::Tensor உள்ளீடு_tensor(ov::element::i64, ov::Shape{1});
        *உள்ளீடு_tensor.data<int64_t>() = (int64_t) chunk_valid_size;
        திரும்பு உள்ளீடு_tensor;
    }
    எனில் (chunk_index > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor உள்ளீடு_tensor(ov::element::i64, ov::Shape{1});
        *உள்ளீடு_tensor.data<int64_t>() = 0;
        திரும்பு உள்ளீடு_tensor;
    }
    எனில் (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        திரும்பு get_ov_உள்ளீடு_tensor(ggml_decoder, param_name);
    }
    மாறிலி auto * ggml_tensor = ggml_decoder->get_உள்ளீடு_ggml_tensor(param_name);
    மாறிலி auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    எனில் (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        மாறிலி முழு n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        மாறிலி size_t element_size = ggml_type_size(ggml_tensor->type);
        ov::Shape உள்ளீடு_shape = {1, 1, 1, (size_t) n_planes * chunk_size};
        ov::Tensor உள்ளீடு_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), உள்ளீடு_shape);
        க்கு (முழு p = 0; p < n_planes; p++) {
            மாறிலி எழுத்து * src =
                (மாறிலி எழுத்து *) ggml_tensor->data + (p * உள்ளீடு_len + chunk_index * chunk_size) * element_size;
            எழுத்து * dst = (எழுத்து *) உள்ளீடு_tensor.data() + p * chunk_size * element_size;
            std::memcpy(dst, src, chunk_valid_size * element_size);
            எனில் (chunk_pad_size > 0) {
                எனில் (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t last_மதிப்பு = *((மாறிலி int32_t *) src + chunk_valid_size - 1);
                    int32_t * out = (int32_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_மதிப்பு + 1);
                } இல்லையெனில் எனில் (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t last_மதிப்பு = *((மாறிலி int64_t *) src + chunk_valid_size - 1);
                    int64_t * out = (int64_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_மதிப்பு + 1);
                } இல்லையெனில் {
                    எறி std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        திரும்பு உள்ளீடு_tensor;
    }

    எனில் (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape உள்ளீடு_shape = {1, 1, 1, chunk_size};
        ov::Tensor உள்ளீடு_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), உள்ளீடு_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        வெறுமை * உள்ளீடு_data = (எழுத்து *) ggml_tensor->data + chunk_index * chunk_size * element_size;
        std::memcpy(உள்ளீடு_tensor.data(), உள்ளீடு_data, chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        எனில் (chunk_pad_size > 0) {
            எனில் (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_மதிப்பு =
                    *((int32_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int32_t * output_data = உள்ளீடு_tensor.data<int32_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_மதிப்பு + 1);
            } இல்லையெனில் எனில் (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_மதிப்பு =
                    *((int64_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int64_t * output_data = உள்ளீடு_tensor.data<int64_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_மதிப்பு + 1);
            } இல்லையெனில் {
                எறி std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        திரும்பு உள்ளீடு_tensor;
    }

    எனில் (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape உள்ளீடு_shape = {1, 1, 1, output_len};
        ov::Tensor உள்ளீடு_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), உள்ளீடு_shape);
        எனில் (ggml_tensor->ne[0] == 0) {
            *உள்ளீடு_tensor.data<int32_t>() = 0;
        } இல்லையெனில் {
            auto * data_addr = உள்ளீடு_tensor.data<int32_t>();
            க்கு (size_t i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        திரும்பு உள்ளீடு_tensor;
    }

    எனில் (GgmlOvDecoder::is_inp_scale_rows(ggml_tensor, op)) {
        ov::Tensor உள்ளீடு_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, 1});
        auto * dst = உள்ளீடு_tensor.data<தசம>();
        மாறிலி auto * src = static_cast<மாறிலி தசம *>(ggml_tensor->data) + chunk_index * chunk_size;
        std::copy(src, src + chunk_valid_size, dst);
        std::fill(dst + chunk_valid_size, dst + chunk_size, 1.0f);
        திரும்பு உள்ளீடு_tensor;
    }

    எனில் (GgmlOvDecoder::is_inp_mean(ggml_tensor, op)) {
        மாறிலி size_t n_seqs = ggml_tensor->ne[1];
        மாறிலி size_t src_stride = ggml_tensor->ne[0];
        மாறிலி size_t copy_len = std::min<size_t>(chunk_valid_size, src_stride - chunk_index * chunk_size);
        ov::Tensor உள்ளீடு_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_size});
        auto * dst = உள்ளீடு_tensor.data<தசம>();
        std::fill(dst, dst + n_seqs * chunk_size, 0.0f);
        மாறிலி auto * src = static_cast<மாறிலி தசம *>(ggml_tensor->data) + chunk_index * chunk_size;
        க்கு (size_t s = 0; s < n_seqs; s++) {
            std::memcpy(dst + s * chunk_size, src + s * src_stride, copy_len * அளவு(தசம));
        }
        திரும்பு உள்ளீடு_tensor;
    }

    எனில் (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = ggml_decoder->get_ctx_size();
        எனில் (ggml_tensor->type == GGML_TYPE_F16) {
            மாறிலி auto * ggml_data =
                static_cast<மாறிலி ggml_fp16_t *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
            std::vector<ggml_fp16_t> padded_data = pad_உள்ளீடு<ggml_fp16_t>(ggml_data, chunk_valid_rows, cols, chunk_size,
                                                                          context_size, GGML_FP32_TO_FP16(-INFINITY));
            set_zero_diagonal(padded_data, chunk_size, context_size, GGML_FP32_TO_FP16(0.0f));
            ov::Tensor உள்ளீடு_tensor(ov::element::f16, ov::Shape{1, 1, chunk_size, context_size});
            std::memcpy(உள்ளீடு_tensor.data(), padded_data.data(), padded_data.size() * அளவு(ggml_fp16_t));
            திரும்பு உள்ளீடு_tensor;
        }

        மாறிலி auto * ggml_data = static_cast<மாறிலி தசம *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
        std::vector<தசம> padded_data =
            pad_உள்ளீடு<தசம>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor உள்ளீடு_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        auto * data_ptr = உள்ளீடு_tensor.data<தசம>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        திரும்பு உள்ளீடு_tensor;
    }

    திரும்பு get_ov_உள்ளீடு_tensor(ggml_decoder, param_name);
}

எண்ணிக்கை ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               மாறிலி std::string & device,
                               மாறிலி ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    எனில் (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        திரும்பு GGML_STATUS_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_lock(cache.mutex);
    தருக்க naive = உண்மை;
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    auto உள்ளீடு_model = std::make_shared<ov::frontend::ggml::உள்ளீடு_Model>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(உள்ளீடு_model, naive);
    எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    எனில் (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    } இல்லையெனில் {
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    }
    எனில் (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } இல்லையெனில் {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> உள்ளீடு_names;
    std::vector<std::string> output_names;
    க்கு (மாறிலி auto & param : model->get_parameters()) {
        உள்ளீடு_names.push_back(param->get_friendly_name());
    }
    க்கு (மாறிலி auto & result : model->get_results()) {
        output_names.push_back(result->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    உள்ளீடு_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_lock.unlock();

    க்கு (size_t i = 0; i < உள்ளீடு_names.size(); i++) {
        மாறிலி auto & param_name = உள்ளீடு_names[i];
        auto உள்ளீடு_tensor = get_ov_உள்ளீடு_tensor(decoder, param_name);
        infer_request->set_உள்ளீடு_tensor(i, உள்ளீடு_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    க்கு (size_t i = 0; i < output_names.size(); i++) {
        auto output_tensor = infer_request->get_output_tensor(i);
        மாறிலி auto & model_outputs = decoder->get_model_outputs();
        auto model_output_it = model_outputs.find(output_names[i]);
        எனில் (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_names[i], output_tensor, output_tensor.data());
            }
            தொடர்;
        }
        auto * ggml_tensor = model_output_it->second;
        std::memcpy(ggml_tensor->data, output_tensor.data(), output_tensor.get_byte_size());
    }
    திரும்பு GGML_STATUS_SUCCESS;
}

எண்ணிக்கை ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, மாறிலி std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();
    மாறிலி auto & config = ggml_openvino_get_compile_config();
    மாறிலி auto & device = r_ctx->device;
    நிலையான auto is_static = பொய்;

    நிலையான மாறிலி தருக்க cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    auto start_time = ggml_time_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    மாறிலி int64_t cache_சாவி_start_time = ggml_time_us();
    graph_சாவி சாவி(cgraph);
    மாறிலி int64_t cache_சாவி_compute_time = ggml_time_us() - cache_சாவி_start_time;
    int64_t cache_lookup_time = 0;
    தருக்க சாவி_seen = பொய்;
    எனில் (!cache_disabled) {
        மாறிலி int64_t cache_lookup_start_time = ggml_time_us();
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        சாவி_seen = r_ctx->decoder_cache.find(சாவி) != r_ctx->decoder_cache.end();
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;
    }

    மாறிலி int64_t model_split_check_start_time = ggml_time_us();
    தருக்க model_is_splitted = சாவி_seen ? பொய் : is_model_splitted(cgraph);
    மாறிலி int64_t model_split_check_time = ggml_time_us() - model_split_check_start_time;

    எனில் (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        திரும்பு GGML_STATUS_FAILED;
    }
    எனில் (is_naive(cgraph)) {
        எனில் (!model_is_splitted) {
            திரும்பு naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    மாறிலி int64_t graph_param_compute_start_time = ggml_time_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    மாறிலி int64_t graph_param_compute_time = ggml_time_us() - graph_param_compute_start_time;

    மாறிலி தருக்க cache_enabled = !model_is_splitted && !cache_disabled;
    மாறிலி தருக்க stateful_recurrent = r_ctx->stateful && m_params.state_size >= 0;
    மாறிலி தருக்க stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    எனில் (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_active > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        திரும்பு GGML_STATUS_FAILED;
    }
    எனில் (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        திரும்பு GGML_STATUS_FAILED;
    }
    தருக்க cache_hit = பொய்;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        மாறிலி int64_t cache_lookup_start_time = ggml_time_us();
        எனில் (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            auto it = r_ctx->decoder_cache.find(சாவி);
            cache_hit = it != r_ctx->decoder_cache.end();
            எனில் (cache_hit) {
                entry = it->second;
            } இல்லையெனில் {
                r_ctx->clear_caches_locked();
                auto mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[சாவி] = entry;
            }
        } இல்லையெனில் {
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_hit = பொய்;
        }
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;

        std::lock_guard<std::mutex> lock(*(entry->mutex));
        cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(சாவி) != 0;

        எனில் (cache_hit) {
            ggml_decoder = entry->ptr;
            old_m_params = ggml_decoder->get_model_params();
            எனில் (!ggml_decoder->is_splited_model()) {
                cache_hit = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_உள்ளீடு_names;
        std::vector<std::string> ov_output_names;

        எனில் (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            எனில் (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(சாவி);
                ov_உள்ளீடு_names = r_ctx->ov_உள்ளீடு_names_cache.at(சாவி);
                ov_output_names = r_ctx->ov_output_names_cache.at(சாவி);
            }

            எனில் (stateful_kv_only) {
                மாறிலி auto * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_data = (int32_t *) inp_pos->data;
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                எனில் (pos_data[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_size = pos_shape[3];
                } இல்லையெனில் எனில் (r_ctx->stateful_kv_size == static_cast<size_t>(pos_data[0])) {
                    r_ctx->stateful_kv_size += pos_shape[3];
                } இல்லையெனில் {
                    மாறிலி size_t pos_begin = static_cast<size_t>(pos_data[0]);
                    மாறிலி தருக்க refill = pos_begin > r_ctx->stateful_kv_size;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    எனில் (refill && !ggml_decoder->get_model_params().swa_layers.empty()) {
                        மாறிலி முழு n_swa = ggml_decoder->get_compute_params().swa_window;
                        எனில் (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_size, n_swa);
                            திரும்பு GGML_STATUS_FAILED;
                        }
                    }

                    மாறிலி தருக்க relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    auto states = infer_request->query_state();
                    க்கு (auto state : states) {
                        auto state_tensor = state.get_state();
                        auto state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        எனில் (auto it = r_ctx->kv_state_உள்ளீடு_name_map.find(state.get_name());
                            it != r_ctx->kv_state_உள்ளீடு_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        முழு n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        எனில் (auto layer = extract_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        மாறிலி தருக்க relayout_this_state = relayout_enabled;
                        மாறிலி size_t seq_axis = relayout_this_state ? 2 : 1;
                        மாறிலி size_t head_axis = seq_axis == 2 ? 1 : 2;

                        எனில் (refill) {
                            எனில் (state_name.empty()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                திரும்பு GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_உள்ளீடு_tensor(ggml_decoder, state_name);
                            எனில் (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_rows_to_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } இல்லையெனில் {
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
                        எனில் (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            திரும்பு GGML_STATUS_FAILED;
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
        } இல்லையெனில் {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            auto shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
            auto weight_names = get_weight_names(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            மாறிலி std::string model_cache_dir = ggml_openvino_model_cache_dir();
            மாறிலி std::string exact_சாவி = cache_enabled ? compiled_graph_சாவி(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_path, manifest_path;
            எனில் (!model_cache_dir.empty() && !model_is_splitted) {
                மாறிலி uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/உண்மை, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_graph_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/உண்மை, m_params.rope_params, 16, extra_cfg,
                                                    compiled_graph_சாவி(cgraph, *ggml_decoder, device, 0, உண்மை));
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            std::string shared_சாவி =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_சாவி) : "";
            ov::CompiledModel shared_model;
            தருக்க imported = பொய்;
            auto shared_it = shared_cache->graphs.find(shared_சாவி);
            எனில் (!shared_சாவி.empty() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_graph(shared_it->second.decode, shared_it->second.உள்ளீடு_names,
                                                               shared_it->second.output_names, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_உள்ளீடு_names = shared_it->second.உள்ளீடு_names;
                ov_output_names = shared_it->second.output_names;
                imported = உண்மை;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } இல்லையெனில் எனில் (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_சாவி = exact_சாவி;
                shared_it = shared_cache->graphs.find(shared_சாவி);
                எனில் (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_உள்ளீடு_names = shared_it->second.உள்ளீடு_names;
                    ov_output_names = shared_it->second.output_names;
                    imported = உண்மை;
                }
                model_fp = exact_fp;
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            எனில் (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(சாவி);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            எனில் (!model_cache_dir.empty()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
            }
            எனில் (!imported && !model_cache_dir.empty() && !model_is_splitted) {
                மாறிலி uint64_t preferred_fp = model_fp;
                தருக்க dynamic_incompatible = preferred_fp == exact_fp;
                க்கு (முழு candidate = 0; candidate < 2; ++candidate) {
                    எனில் (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        நிறுத்து;
                    }
                    மாறிலி uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    மாறிலி std::string candidate_blob =
                        ggml_openvino_model_cache_blob_path(model_cache_dir, candidate_fp);
                    மாறிலி std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_path(model_cache_dir, candidate_fp);
                    std::ifstream blob_in(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    எனில் (!blob_in.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        தொடர்;
                    }
                    மாறிலி int64_t import_start = ggml_time_us();
                    முயல் {
                        ov::CompiledModel cm;
                        auto remote_context = ggml_openvino_get_remote_context();
                        எனில் (remote_context.has_value()) {
                            cm = core.import_model(blob_in, remote_context.value(), mc_config);
                        } இல்லையெனில் {
                            cm = core.import_model(blob_in, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        எனில் (!saved_inputs.empty() || !saved_outputs.empty()) {
                            ov_உள்ளீடு_names = std::move(saved_inputs);
                            ov_output_names = std::move(saved_outputs);
                        } இல்லையெனில் {
                            க்கு (மாறிலி auto & p : cm.inputs()) {
                                ov_உள்ளீடு_names.push_back(p.get_முனை()->get_friendly_name());
                            }
                            க்கு (மாறிலி auto & o : cm.outputs()) {
                                ov_output_names.push_back(o.get_முனை()->get_friendly_name());
                            }
                        }
                        எனில் (!compiled_model_matches_graph(cm, ov_உள்ளீடு_names, ov_output_names, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_உள்ளீடு_names.clear();
                            ov_output_names.clear();
                            தொடர்;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        entry->ptr = ggml_decoder;
                        imported = உண்மை;
                        model_fp = candidate_fp;
                        blob_path = candidate_blob;
                        manifest_path = candidate_manifest;
                        எனில் (candidate_fp == exact_fp) {
                            shared_சாவி = exact_சாவி;
                            shared_it = shared_cache->graphs.find(shared_சாவி);
                        }
                        எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_time_us() - import_start) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } பிடி (மாறிலி std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = பொய்;
                        ov_உள்ளீடு_names.clear();
                        ov_output_names.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                எனில் (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_சாவி = exact_சாவி;
                    shared_it = shared_cache->graphs.find(shared_சாவி);
                    blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                    manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            எனில் (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_path.c_str());
                திரும்பு GGML_STATUS_FAILED;
            }
            எனில் (!imported && ggml_openvino_weight_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                திரும்பு GGML_STATUS_FAILED;
            }
            எனில் (imported) {
                decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            } இல்லையெனில் {
                auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_end_time = ggml_time_us();

                auto உள்ளீடு_model = std::make_shared<ov::frontend::ggml::உள்ளீடு_Model>(ggml_decoder);
                model = ov::frontend::ggml::FrontEnd::convert(உள்ளீடு_model);
                ggml_decoder->clear_model_weights();
                conversion_end_time = ggml_time_us();

                எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    எழுத்து timestamped_filename[64];
                    auto timestamp = (long long) ggml_time_us();
                    snprintf(timestamped_filename, அளவு(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                மாறிலி ov::AnyMap & compile_config = model_cache_dir.empty() ? config : mc_config;
                ov::CompiledModel compiled_model;
                auto remote_context = ggml_openvino_get_remote_context();
                எனில் (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } இல்லையெனில் {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_end_time = ggml_time_us();

                க்கு (மாறிலி auto & ov_param : model->get_parameters()) {
                    ov_உள்ளீடு_names.push_back(ov_param->get_friendly_name());
                }
                க்கு (மாறிலி auto & ov_output : model->get_results()) {
                    ov_output_names.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                எனில் (!model_cache_dir.empty() && !model_is_splitted && model_fp != 0) {
                    முயல் {
                        மாறிலி std::string blob_tmp = ggml_openvino_model_cache_temp_path(blob_path);
                        மாறிலி std::string manifest_tmp = ggml_openvino_model_cache_temp_path(manifest_path);
                        எனில் (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_உள்ளீடு_names,
                                                                     ov_output_names)) {
                            std::ofstream blob_out(blob_tmp, std::ios::binary | std::ios::trunc);
                            எனில் (blob_out.is_open()) {
                                compiled_model.export_model(blob_out);
                                blob_out.close();
                                எனில் (blob_out.good()) {
                                    எனில் (std::rename(blob_tmp.c_str(), blob_path.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_path.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_path.c_str());
                                    } இல்லையெனில் {
                                        std::remove(blob_tmp.c_str());
                                        std::remove(manifest_tmp.c_str());
                                    }
                                } இல்லையெனில் {
                                    std::remove(blob_tmp.c_str());
                                    std::remove(manifest_tmp.c_str());
                                }
                            } இல்லையெனில் {
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    } பிடி (மாறிலி std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                entry->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            entry->ptr = ggml_decoder;
            எனில் (!shared_சாவி.empty() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_சாவி,
                                             ov_compiled_graph{shared_model, {}, ov_உள்ளீடு_names, ov_output_names});
            }
            எனில் (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[சாவி] = infer_request;
                r_ctx->ov_உள்ளீடு_names_cache[சாவி] = ov_உள்ளீடு_names;
                r_ctx->ov_output_names_cache[சாவி] = ov_output_names;
            }

            எனில் (stateful_kv_only && cache_enabled) {
                மாறிலி auto * inp_pos = get_inp_pos_tensor(cgraph);
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                மாறிலி int32_t pos_begin = ((int32_t *) inp_pos->data)[0];
                எனில் (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    திரும்பு GGML_STATUS_FAILED;
                }
                r_ctx->stateful_kv_size = pos_shape[3];
                மாறிலி auto kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                க்கு (மாறிலி auto & pair : kv_param_res_names) {
                    r_ctx->kv_state_உள்ளீடு_name_map[pair.first + pair.second] = pair.first;
                }
            }
        }

        எனில் (stateful_recurrent) {
            மாறிலி auto * inp_pos = get_inp_pos_tensor(cgraph);
            மாறிலி int32_t pos_begin = static_cast<மாறிலி int32_t *>(inp_pos->data)[0];
            எனில் (pos_begin == 0) {
                infer_request->reset_state();
            } இல்லையெனில் எனில் (!cache_hit || old_m_params.kv_buffer_changed(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_size) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                திரும்பு GGML_STATUS_FAILED;
            }
        } இல்லையெனில் {
            reset_single_slot_recurrent_cache(cgraph, m_params, c_params);
        }

        க்கு (size_t i = 0; i < ov_உள்ளீடு_names.size(); i++) {
            மாறிலி auto & param_name = ov_உள்ளீடு_names[i];
            auto உள்ளீடு_tensor = get_ov_உள்ளீடு_tensor(ggml_decoder, param_name);
            infer_request->set_உள்ளீடு_tensor(i, உள்ளீடு_tensor);

            எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_உள்ளீடு_tensor_info(param_name, உள்ளீடு_tensor);
            }
        }

        க்கு (size_t i = 0; i < ov_output_names.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            மாறிலி auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names[i]);
            எனில் (model_output_it == model_outputs.end()) {
                தொடர்;
            }
            auto * ggml_tensor = model_output_it->second;
            எனில் (ggml_nbytes(ggml_tensor) == 0) {
                தொடர்;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        எனில் (stateful_recurrent) {
            மாறிலி auto * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_size =
                static_cast<மாறிலி int32_t *>(inp_pos->data)[0] + get_inp_pos_n_tokens(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            க்கு (size_t i = 0; i < ov_output_names.size(); i++) {
                மாறிலி auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }

        எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_சாவி_compute_time / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_split_check_time / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", graph_param_compute_time / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_time / 1000.0);
            எனில் (!cache_hit) {
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
    எனில் (cache_hit && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_lock(r_ctx->compiled_cache->mutex);
        எனில் (!ggml_openvino_weight_buffers_released()) {
            ggml_openvino_release_weight_buffers();
        }
    }

    திரும்பு GGML_STATUS_SUCCESS;
}

ov::AnyMap without_npuw(மாறிலி ov::AnyMap & config) {
    ov::AnyMap out;
    க்கு (மாறிலி auto & kv : config) {
        எனில் (kv.first.rfind("NPUW", 0) == 0 || kv.first == "NPU_USE_NPUW") {
            தொடர்;
        }
        out.insert(kv);
    }
    திரும்பு out;
}

எண்ணிக்கை ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, மாறிலி std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();

    auto get_prefill_chunk_size = [] {
        நிலையான மாறிலி முழு chunk_size = []() {
            முழு env_prefill_chunk_size = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            திரும்பு env_prefill_chunk_size > 0 ? env_prefill_chunk_size : 256;
        }();
        திரும்பு chunk_size;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    நிலையான std::string device = ggml_openvino_get_device_name();
    நிலையான auto is_static = உண்மை;
    நிலையான auto stateful = பொய்;

    auto prefill_chunk_size = get_prefill_chunk_size();
    மாறிலி auto & config = ggml_openvino_get_compile_config();

    எனில் (is_naive(cgraph)) {
        திரும்பு naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    மாறிலி auto * inp_pos = get_inp_pos_tensor(cgraph);
    மாறிலி தருக்க no_kv_cache = m_params.is_cacheless_attn;
    மாறிலி auto is_prefill = no_kv_cache ? உண்மை : get_is_prefill(cgraph, inp_pos);
    மாறிலி ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    எனில் (m_params.n_heads_kv == -1) {
        prefill_chunk_size = inp_pos->ne[0];
    }
    graph_சாவி சாவி(cgraph);
    நிலையான மாறிலி தருக்க cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    தருக்க cache_hit = பொய்;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;
    int64_t ov_raw_infer_total = 0;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    எனில் (cache_enabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        auto it = r_ctx->decoder_cache.find(சாவி);
        cache_hit = it != r_ctx->decoder_cache.end();
        எனில் (cache_hit) {
            entry = it->second;
        } இல்லையெனில் {
            r_ctx->clear_caches_locked();
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[சாவி] = entry;
        }
    } இல்லையெனில் {
        auto mutex = std::make_shared<std::mutex>();
        entry = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_hit = பொய்;
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));
    cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(சாவி) != 0 &&
                r_ctx->infer_request_cache_prefill.count(சாவி) != 0;

    எனில் (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_உள்ளீடு_names_local;
    std::vector<std::string> ov_output_names_local;

    எனில் (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        எனில் (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(சாவி) : r_ctx->infer_request_cache.at(சாவி);
            ov_உள்ளீடு_names_local = r_ctx->ov_உள்ளீடு_names_cache.at(சாவி);
            ov_output_names_local = r_ctx->ov_output_names_cache.at(சாவி);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } இல்லையெனில் {
        எனில் (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(சாவி);
            r_ctx->infer_request_cache_prefill.erase(சாவி);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        auto shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
        auto weight_names = get_weight_names(cgraph);
        auto local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names, is_static,
                                                             stateful, பொய், is_prefill, prefill_chunk_size);
        மாறிலி std::string shared_சாவி =
            cache_enabled ? compiled_graph_சாவி(cgraph, *local_decoder, device, prefill_chunk_size) : "";
        auto shared_it = shared_cache->graphs.find(shared_சாவி);
        எனில் (!shared_சாவி.empty() && shared_it != shared_cache->graphs.end()) {
            auto & compiled = shared_it->second;
            auto prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            auto decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            entry->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_உள்ளீடு_names_local = compiled.உள்ளீடு_names;
            ov_output_names_local = compiled.output_names;
            r_ctx->infer_request_cache_prefill[சாவி] = prefill_request;
            r_ctx->infer_request_cache[சாவி] = decode_request;
            r_ctx->ov_உள்ளீடு_names_cache[சாவி] = ov_உள்ளீடு_names_local;
            r_ctx->ov_output_names_cache[சாவி] = ov_output_names_local;
            decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } இல்லையெனில் {
            std::shared_ptr<ov::Model> model;
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

            auto ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, பொய், உண்மை, prefill_chunk_size);
            auto ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, பொய், பொய், prefill_chunk_size);
            decoder_end_time = ggml_time_us();

            மாறிலி தருக்க dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            மாறிலி auto dump_ir_timestamp = static_cast<long long>(ggml_time_us());

            auto build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          மாறிலி std::shared_ptr<GgmlOvDecoder> & decoder, மாறிலி எழுத்து * tag,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_end_time, int64_t & local_compile_end_time) {
                auto உள்ளீடு_model = std::make_shared<ov::frontend::ggml::உள்ளீடு_Model>(decoder);
                model = ov::frontend::ggml::FrontEnd::convert(உள்ளீடு_model);
                decoder->clear_model_weights();
                local_conversion_end_time = ggml_time_us();

                எனில் (dump_ir) {
                    எழுத்து timestamped_filename[64];
                    snprintf(timestamped_filename, அளவு(timestamped_filename), "model_%s_%lld.xml", tag,
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
            எனில் (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_end_time = prefill_conversion_end_time;
                decode_compile_end_time = prefill_compile_end_time;
            } இல்லையெனில் {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_end_time, decode_compile_end_time);
            }
            conversion_end_time = std::max(prefill_conversion_end_time, decode_conversion_end_time);
            compile_end_time = std::max(prefill_compile_end_time, decode_compile_end_time);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            entry->ptr = ggml_decoder;

            க்கு (மாறிலி auto & ov_param : model->get_parameters()) {
                ov_உள்ளீடு_names_local.push_back(ov_param->get_friendly_name());
            }
            க்கு (மாறிலி auto & ov_output : model->get_results()) {
                ov_output_names_local.push_back(ov_output->get_friendly_name());
            }

            எனில் (!shared_சாவி.empty()) {
                shared_cache->graphs.emplace(
                    shared_சாவி, ov_compiled_graph{compiled_model_decode, compiled_model_prefill, ov_உள்ளீடு_names_local,
                                                  ov_output_names_local});
            }

            எனில் (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[சாவி] = infer_request_prefill;
                r_ctx->infer_request_cache[சாவி] = infer_request_decode;
                r_ctx->ov_உள்ளீடு_names_cache[சாவி] = ov_உள்ளீடு_names_local;
                r_ctx->ov_output_names_cache[சாவி] = ov_output_names_local;
            }
        }
    }

    reset_single_slot_recurrent_cache(cgraph, m_params, c_params);

    எனில் (is_prefill) {
        auto inp_len = get_inp_pos_n_tokens(cgraph, inp_pos);
        க்கு (முழு chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            க்கு (size_t i = 0; i < ov_உள்ளீடு_names_local.size(); i++) {
                மாறிலி auto & param_name = ov_உள்ளீடு_names_local[i];
                auto உள்ளீடு_tensor = get_ov_உள்ளீடு_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_உள்ளீடு_tensor(i, உள்ளீடு_tensor);

                எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    மாறிலி auto உள்ளீடு_tensor = infer_request->get_உள்ளீடு_tensor(i);
                    print_உள்ளீடு_tensor_info(param_name, உள்ளீடு_tensor);
                }
            }

            க்கு (size_t i = 0; i < ov_output_names_local.size(); i++) {
                மாறிலி auto & model_outputs = ggml_decoder->get_model_outputs();
                auto model_output_it = model_outputs.find(ov_output_names_local[i]);
                எனில் (model_output_it == model_outputs.end()) {
                    தொடர்;
                }
                auto * ggml_tensor = model_output_it->second;
                எனில் (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    தொடர்;
                }
                auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            ov_raw_infer_start = ggml_time_us();
            infer_request->infer();
            ov_raw_infer_total += ggml_time_us() - ov_raw_infer_start;

            எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                க்கு (size_t i = 0; i < ov_output_names_local.size(); i++) {
                    மாறிலி auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } இல்லையெனில் {
        க்கு (size_t i = 0; i < ov_உள்ளீடு_names_local.size(); i++) {
            மாறிலி auto & param_name = ov_உள்ளீடு_names_local[i];
            auto உள்ளீடு_tensor = get_ov_உள்ளீடு_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_உள்ளீடு_tensor(i, உள்ளீடு_tensor);

            எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                மாறிலி auto உள்ளீடு_tensor = infer_request->get_உள்ளீடு_tensor(i);
                print_உள்ளீடு_tensor_info(param_name, உள்ளீடு_tensor);
            }
        }

        க்கு (size_t i = 0; i < ov_output_names_local.size(); i++) {
            மாறிலி auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names_local[i]);
            எனில் (model_output_it == model_outputs.end()) {
                தொடர்;
            }
            auto * ggml_tensor = model_output_it->second;
            எனில் (ggml_nbytes(ggml_tensor) == 0) {
                தொடர்;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        ov_raw_infer_total = infer_end_time - ov_raw_infer_start;

        எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            க்கு (size_t i = 0; i < ov_output_names_local.size(); i++) {
                மாறிலி auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
            }
        }
    }

    எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
        எனில் (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_end_time - decoder_end_time) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_total / 1000.0);
    }

    திரும்பு GGML_STATUS_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
நிலையான ggml_cgraph * get_compute_graph(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    auto is_skipped = [](மாறிலி ggml_tensor * முனை) {
        திரும்பு முனை->op != GGML_OP_NONE && !(முனை->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    எனில் (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        திரும்பு cgraph;
    }
    auto & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](மாறிலி ggml_tensor * முனை) { திரும்பு !is_skipped(முனை); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.data();
    compute.graph.n_nodes = (முழு) compute.nodes.size();
    திரும்பு &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
எண்ணிக்கை ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    முயல் {
        எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        மாறிலி auto is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_lock(r_ctx->execution_mutex);
        cgraph = get_compute_graph(cgraph, *r_ctx);

        திரும்பு is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } பிடி (மாறிலி ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        திரும்பு GGML_STATUS_FAILED;
    } பிடி (மாறிலி std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        திரும்பு GGML_STATUS_FAILED;
    } பிடி (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        திரும்பு GGML_STATUS_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
தருக்க is_model_splitted(ggml_cgraph * cgraph) {
    நிலையான மாறிலி தருக்க fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    எனில் (!fallback_enabled) {
        திரும்பு பொய்;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    எனில் (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        திரும்பு பொய்;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * முனை = cgraph->nodes[i];
        முழு use_மொத்தம் = cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, முனை)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        எனில் ((cgraph->n_nodes <= 1 && use_மொத்தம் == 0) ||
            (cgraph->n_nodes <= 1 && முனை->op == GGML_OP_VIEW && use_மொத்தம் == 1 && முனை->src[0] != nullptr &&
             முனை->src[0]->op == GGML_OP_NONE)) {
            திரும்பு பொய்;
        }
        எனில் (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_PERMUTE)) {
            திரும்பு பொய்;
        }
        முழு உள்ளீடு_use_மொத்தம் = 0;
        க்கு (முழு j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_முனை = cgraph->nodes[j];
            க்கு (முழு k = 0; k < GGML_MAX_SRC; k++) {
                எனில் (other_முனை->src[k] == முனை) {
                    உள்ளீடு_use_மொத்தம்++;
                }
            }
        }
        எனில் (use_மொத்தம் != உள்ளீடு_use_மொத்தம் && முனை->op != GGML_OP_NONE) {
            திரும்பு உண்மை;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    எனில் (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_weight_names(cgraph);
    } இல்லையெனில் {
        க்கு (மாறிலி auto & kv : GgmlOvDecoder::create_weight_nodes(cgraph, உண்மை)) {
            model_weights.insert(kv.first);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * முனை = cgraph->nodes[i];
        க்கு (முழு j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = முனை->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            எனில் (src != nullptr && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.empty() == பொய் &&
                model_leafs.find(src) == model_leafs.end()) {
                எனில் (GgmlOvDecoder::is_inp_tok(src, முனை)) {
                    திரும்பு பொய்;
                }
                திரும்பு உண்மை;
            }
        }
    }
    திரும்பு பொய்;
}

தருக்க is_naive(ggml_cgraph * cgraph) {
    constexpr முழு naive_graph_size_threshold = 20;
    முழு count = 0;
    க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
        எனில் (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    திரும்பு count < naive_graph_size_threshold;
}

size_t checksum(மாறிலி வெறுமை * data, size_t size) {
    மாறிலி uint8_t * bytes = static_cast<மாறிலி uint8_t *>(data);
    size_t sum = 0;
    க்கு (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    திரும்பு sum;
}

தருக்க save_ggml_tensor_data_to_txt(மாறிலி ggml_tensor * tensor, மாறிலி std::string & file_path) {
    எனில் (tensor == nullptr || tensor->data == nullptr) {
        திரும்பு பொய்;
    }

    std::ofstream out(file_path);
    எனில் (!out.is_open()) {
        திரும்பு பொய்;
    }

    மாறிலி size_t n = ggml_nelements(tensor);
    out << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    தேர்வு (tensor->type) {
    நிலை GGML_TYPE_F32: {
        மாறிலி auto * data = static_cast<மாறிலி தசம *>(tensor->data);
        க்கு (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        நிறுத்து;
    }
    நிலை GGML_TYPE_F16: {
        மாறிலி auto * data = static_cast<மாறிலி ggml_fp16_t *>(tensor->data);
        க்கு (size_t i = 0; i < n; ++i) {
            out << ggml_fp16_to_fp32(data[i]) << '\n';
        }
        நிறுத்து;
    }
    நிலை GGML_TYPE_BF16: {
        மாறிலி auto * data = static_cast<மாறிலி ggml_bf16_t *>(tensor->data);
        க்கு (size_t i = 0; i < n; ++i) {
            out << ggml_bf16_to_fp32(data[i]) << '\n';
        }
        நிறுத்து;
    }
    நிலை GGML_TYPE_I32: {
        மாறிலி auto * data = static_cast<மாறிலி int32_t *>(tensor->data);
        க்கு (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        நிறுத்து;
    }
    நிலை GGML_TYPE_I64: {
        மாறிலி auto * data = static_cast<மாறிலி int64_t *>(tensor->data);
        க்கு (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        நிறுத்து;
    }
    இயல்பு:
        out << "unsupported tensor type for text dump" << '\n';
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

வெறுமை print_உள்ளீடு_tensor_info(மாறிலி std::string & name, மாறிலி ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << '\n';
    தேர்வு (tensor.get_element_type()) {
    நிலை ov::element::f32: {
        எனில் (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::cout << *(tensor.data<தசம>()) << '\n';
        } இல்லையெனில் {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            மாறிலி தசம * data = tensor.data<தசம>();
            க்கு (size_t i = 0; i < rows; ++i) {
                க்கு (size_t j = 0; j < cols; ++j) {
                    தசம val = data[i * cols + j];
                    எனில் (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } இல்லையெனில் {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << '\n';
            }
        }

        நிறுத்து;
    }
    நிலை ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << '\n';
        நிறுத்து;
    நிலை ov::element::i32:
        க்கு (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << ' ';
        }
        std::cout << '\n';
        நிறுத்து;
    நிலை ov::element::i64:
        க்கு (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << ' ';
        }
        std::cout << '\n';
        நிறுத்து;
    இயல்பு:
        நிறுத்து;
    }
}

வெறுமை print_output_tensor_info(மாறிலி std::string & name, மாறிலி ov::Tensor & tensor, மாறிலி வெறுமை * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    auto print_float_stats = [](மாறிலி std::string & type_name, size_t size, auto get_மதிப்பு) {
        எனில் (size == 0) {
            திரும்பு;
        }

        தசம first = get_மதிப்பு(0);
        தசம min = first;
        தசம max = first;
        இரட்டை sum = first;

        க்கு (size_t i = 1; i < size; ++i) {
            தசம v = get_மதிப்பு(i);
            min = std::min(v, min);
            max = std::max(v, max);
            sum += v;
        }
        இரட்டை mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << '\n';
    };

    தேர்வு (tensor.get_element_type()) {
    நிலை ov::element::f32: {
        மாறிலி தசம * data = tensor.data<தசம>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { திரும்பு data[i]; });
        நிறுத்து;
    }
    நிலை ov::element::f16: {
        மாறிலி ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { திரும்பு static_cast<தசம>(data[i]); });
        நிறுத்து;
    }
    இயல்பு:
        நிறுத்து;
    }
}

மாறிலி ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    க்கு (முழு i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        க்கு (முழு j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            எனில் (src == nullptr) {
                நிறுத்து;
            }
            எனில் (GgmlOvDecoder::is_inp_pos(src, op)) {
                திரும்பு src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    எறி std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_tokens(ggml_cgraph * cgraph, மாறிலி ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    முழு n_planes = 1;
    க்கு (முழு i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        க்கு (முழு j = 0; j < GGML_MAX_SRC; ++j) {
            எனில் (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                நிறுத்து;
            }
        }
    }
    திரும்பு inp_pos->ne[0] / n_planes;
}

தருக்க get_is_prefill(ggml_cgraph * cgraph, மாறிலி ggml_tensor * inp_pos) {
    திரும்பு get_inp_pos_n_tokens(cgraph, inp_pos) > 1;
}
