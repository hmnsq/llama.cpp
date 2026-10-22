#অন্তর্ভাৱ "utils.h"

#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml-openvino-extra.h"
#অন্তর্ভাৱ "ggml-openvino.h"
#অন্তর্ভাৱ "ggml-openvino/ggml-decoder.h"
#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "model-cache.h"
#অন্তর্ভাৱ "openvino/frontend.h"
#অন্তর্ভাৱ "openvino/input_model.h"

#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <cassert>
#অন্তর্ভাৱ <cmath>
#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <cstdlib>
#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <fstream>
#অন্তর্ভাৱ <functional>
#অন্তর্ভাৱ <iomanip>
#অন্তর্ভাৱ <iostream>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/core/any.hpp>
#অন্তর্ভাৱ <openvino/core/graph_util.hpp>
#অন্তর্ভাৱ <openvino/core/shape.hpp>
#অন্তর্ভাৱ <openvino/core/type/float16.hpp>
#অন্তর্ভাৱ <openvino/frontend/manager.hpp>
#অন্তর্ভাৱ <openvino/openvino.hpp>
#অন্তর্ভাৱ <openvino/runtime/compiled_model.hpp>
#অন্তর্ভাৱ <openvino/runtime/infer_request.hpp>
#অন্তর্ভাৱ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#অন্তর্ভাৱ <openvino/runtime/intel_npu/properties.hpp>
#অন্তর্ভাৱ <openvino/runtime/properties.hpp>
#অন্তর্ভাৱ <openvino/runtime/tensor.hpp>
#অন্তর্ভাৱ <optional>
#অন্তর্ভাৱ <set>
#অন্তর্ভাৱ <string>
#অন্তর্ভাৱ <unordered_map>
#অন্তর্ভাৱ <vector>

নামাকাশ {
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
std::optional<ov::Tensor> try_make_kv_sliced_tensor(ধ্রুৱ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    ধ্রুৱ std::string & name,
                                                    ধ্রুৱ ggml_tensor * ggml_tensor) {
    স্থির ধ্রুৱ তার্কিক kv_slice_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    যদি (kv_slice_disabled) {
        ঘূৰাই std::nullopt;
    }
    যদি (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        ঘূৰাই std::nullopt;
    }
    যদি (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != nullptr) {
        ঘূৰাই std::nullopt;
    }
    ধ্রুৱ auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);
    যদি (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        ঘূৰাই std::nullopt;
    }

    ধ্রুৱ auto & compute_params = ggml_decoder->get_compute_params();
    যদি (compute_params.n_seq_active != 1 || compute_params.seq_active_start != 0) {
        ঘূৰাই std::nullopt;
    }

    পূর্ণাঙ্ক layer;
    যদি (auto layer_opt = extract_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } অন্যথা {
        ঘূৰাই std::nullopt;
    }

    ধ্রুৱ তার্কিক is_swa = ggml_decoder->is_swa_layer(layer);
    যদি (is_swa) {
        ঘূৰাই std::nullopt;
    }
    ধ্রুৱ পূর্ণাঙ্ক ctx_per_seq = ggml_decoder->get_ctx_per_seq();
    ধ্রুৱ পূর্ণাঙ্ক n_kv = compute_params.attention_size;
    যদি (ctx_per_seq <= 0 || n_kv <= 0 || n_kv >= ctx_per_seq) {
        ঘূৰাই std::nullopt;
    }

    ov::Shape full_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    যদি (full_shape.size() != 4 || full_shape[0] != 1 || full_shape[1] != 1 ||
        static_cast<পূর্ণাঙ্ক>(full_shape[2]) != ctx_per_seq) {
        ঘূৰাই std::nullopt;
    }

    ov::Shape sliced_shape = full_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    যদি (ggml_openvino_buffer_is_remote(ggml_tensor)) {
        auto remote_context = ggml_openvino_get_remote_context();
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        ঘূৰাই gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
    }

    ঘূৰাই ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
}

স্থির uint64_t ggml_openvino_model_cache_extra_cfg(তার্কিক stateful, তার্কিক recurrent) {
    ধ্রুৱ ৱর্ণ * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    ধ্রুৱ তার্কিক manual_gqa_enabled = manual_gqa_env != nullptr ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    ঘূৰাই extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_weight_names(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> names;
    বাবে (ধ্রুৱ auto & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
        names[name] = nullptr;
    }
    ঘূৰাই names;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
স্থির std::string compiled_graph_চাবি(ধ্রুৱ ggml_cgraph * graph,
                                      ধ্রুৱ GgmlOvDecoder & decoder,
                                      ধ্রুৱ std::string & device,
                                      পূর্ণাঙ্ক prefill_chunk_size = 0,
                                      তার্কিক disk_cache = মিছা) {
    std::string চাবি;
    auto append = [&চাবি](ধ্রুৱ auto & value) {
        চাবি.append(reinterpret_cast<ধ্রুৱ ৱর্ণ *>(&value), পরিমাণ(value));
    };
    auto append_string = [&](ধ্রুৱ std::string & value) {
        append(value.size());
        চাবি.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_size);
    তার্কিক has_weight_buffer_id = মিছা;
    std::unordered_map<ধ্রুৱ ggml_tensor *, size_t> ids;
    std::function<শূন্য(ধ্রুৱ ggml_tensor *)> visit = [&](ধ্রুৱ ggml_tensor * tensor) {
        যদি (!tensor) {
            append(size_t(0));
            ঘূৰাই;
        }
        auto inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.first->second);
        যদি (!inserted.second) {
            ঘূৰাই;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        ধ্রুৱ auto * base = tensor->view_src ? tensor->view_src : tensor;
        ধ্রুৱ তার্কিক weight = base->buffer && base->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        append(weight);
        যদি (weight && !disk_cache) {
            ধ্রুৱ size_t buffer_id = ggml_backend_openvino_buffer_get_ctx_id(base->buffer);
            has_weight_buffer_id |= buffer_id != 0;
            append(buffer_id);
            append(tensor->data);
        }
        visit(tensor->view_src);
        বাবে (ধ্রুৱ auto * src : tensor->src) {
            visit(src);
        }
    };
    append(graph->n_nodes);
    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; ++i) {
        visit(graph->nodes[i]);
    }
    append(graph->n_leafs);
    বাবে (পূর্ণাঙ্ক i = 0; i < graph->n_leafs; ++i) {
        visit(graph->leafs[i]);
    }
    বাবে (ধ্রুৱ auto & প্ৰবেশ : decoder.get_model_extra_inputs()) {
        append_string(প্ৰবেশ.first);
        append_string(প্ৰবেশ.second.type.get_type_name());
        append(প্ৰবেশ.second.shape.size());
        বাবে (auto dim : প্ৰবেশ.second.shape) {
            append(dim);
        }
        append(প্ৰবেশ.second.is_parameter);
        যদি (!প্ৰবেশ.second.is_parameter) {
            append(প্ৰবেশ.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    ঘূৰাই disk_cache || has_weight_buffer_id ? চাবি : std::string{};
}

স্থির std::string dynamic_graph_signature(ধ্রুৱ ggml_cgraph * graph,
                                           ধ্রুৱ GgmlOvDecoder & decoder,
                                           ধ্রুৱ ModelParams & params) {
    std::string চাবি = "dynamic-1";
    auto append = [&চাবি](ধ্রুৱ auto & value) {
        চাবি.append(reinterpret_cast<ধ্রুৱ ৱর্ণ *>(&value), পরিমাণ(value));
    };
    auto append_string = [&](ধ্রুৱ std::string & value) {
        append(value.size());
        চাবি.append(value);
    };
    graph_চাবি graph_id(graph, সঁচা);
    append(graph_id.n_nodes);
    append(graph_id.n_leaves);
    append_string(graph_id.first_গ্ৰন্থি_name);
    append_string(graph_id.last_গ্ৰন্থি_name);
    বাবে (ধ্রুৱ auto & name : graph_id.প্ৰবেশ_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    বাবে (পূর্ণাঙ্ক layer : params.swa_layers) {
        append(layer);
    }
    বাবে (ধ্রুৱ auto & [layer, heads] : params.n_heads_kv_per_layer) {
        append(layer);
        append(heads);
    }
    বাবে (ধ্রুৱ auto & [name, প্ৰবেশ] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(প্ৰবেশ.type.get_type_name());
    }
    বাবে (ধ্রুৱ auto & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    বাবে (ধ্রুৱ auto & [name, প্ৰবেশ] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(প্ৰবেশ.type.get_type_name());
        append(প্ৰবেশ.is_parameter);
        যদি (!প্ৰবেশ.is_parameter) {
            append(প্ৰবেশ.value);
        }
    }
    ঘূৰাই চাবি;
}

স্থির তার্কিক compiled_model_matches_graph(ধ্রুৱ ov::CompiledModel & model,
                                         ধ্রুৱ std::vector<std::string> & inputs,
                                         ধ্রুৱ std::vector<std::string> & outputs,
                                         ধ্রুৱ GgmlOvDecoder & decoder) {
    যদি (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        ঘূৰাই মিছা;
    }
    ধ্রুৱ auto & graph_inputs = decoder.get_model_inputs();
    ধ্রুৱ auto & extra_inputs = decoder.get_model_extra_inputs();
    ধ্রুৱ auto & graph_outputs = decoder.get_model_outputs();
    বাবে (size_t i = 0; i < inputs.size(); ++i) {
        ধ্রুৱ auto & port = model.প্ৰবেশ(i);
        auto graph_it = graph_inputs.find(inputs[i]);
        যদি (graph_it != graph_inputs.end()) {
            যদি (port.get_element_type() != graph_it->second.type ||
                !port.get_partial_shape().compatible(graph_it->second.shape)) {
                ঘূৰাই মিছা;
            }
            অনুৱর্ত;
        }
        auto extra_it = extra_inputs.find(inputs[i]);
        যদি (extra_it == extra_inputs.end() || port.get_element_type() != extra_it->second.type ||
            !port.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            ঘূৰাই মিছা;
        }
    }
    বাবে (size_t i = 0; i < outputs.size(); ++i) {
        auto graph_it = graph_outputs.find(outputs[i]);
        যদি (graph_it == graph_outputs.end()) {
            যদি (outputs[i].compare(0, 8, "__debug_") != 0) {
                ঘূৰাই মিছা;
            }
            অনুৱর্ত;
        }
        ধ্রুৱ auto & port = model.output(i);
        যদি (port.get_element_type() != GgmlOvDecoder::get_ov_type(graph_it->second) ||
            !port.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(graph_it->second)))) {
            ঘূৰাই মিছা;
        }
    }
    ঘূৰাই সঁচা;
}

স্থির std::string ov_profiling_csv_field(ধ্রুৱ std::string & value) {
    std::string escaped = "\"";
    বাবে (ৱর্ণ c : value) {
        escaped += c;
        যদি (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    ঘূৰাই escaped;
}

স্থির শূন্য dump_ov_profiling_info(ধ্রুৱ ov::InferRequest & infer_request) {
    যদি (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        ঘূৰাই;
    }

    স্থির std::atomic<uint64_t> inference_index{0};
    ধ্রুৱ uint64_t index = inference_index.fetch_add(1);
    ধ্রুৱ std::string path = "openvino_profile_" + std::to_string(index) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    যদি (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        ঘূৰাই;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t total_real_time = 0;
    int64_t total_cpu_time = 0;
    size_t executed_গুনতি = 0;
    বাবে (ধ্রুৱ auto & info : infer_request.get_profiling_info()) {
        ধ্রুৱ ৱর্ণ * status = "NOT_RUN";
        যদি (info.status == ov::ProfilingInfo::Status::EXECUTED) {
            status = "EXECUTED";
            total_real_time += info.real_time.count();
            total_cpu_time += info.cpu_time.count();
            executed_গুনতি++;
        } অন্যথা যদি (info.status == ov::ProfilingInfo::Status::OPTIMIZED_OUT) {
            status = "OPTIMIZED_OUT";
        }

        output << status << ',' << info.real_time.count() << ',' << info.cpu_time.count() << ','
               << info.start_time.count() << ',' << ov_profiling_csv_field(info.node_type) << ','
               << ov_profiling_csv_field(info.গ্ৰন্থি_name) << ',' << ov_profiling_csv_field(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_গুনতি, total_real_time / 1000.0, total_cpu_time / 1000.0);
}

স্থির শূন্য reset_single_slot_recurrent_cache(ggml_cgraph * cgraph,
                                              ধ্রুৱ ModelParams & model_params,
                                              ধ্রুৱ ComputeParams & compute_params) {
    যদি (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        ঘূৰাই;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_buffer_t> buffers;
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        auto * গ্ৰন্থি = cgraph->nodes[i];
        যদি (গ্ৰন্থি->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(গ্ৰন্থি->view_src) &&
            গ্ৰন্থি->view_src->ne[1] == 1 && গ্ৰন্থি->view_src->buffer != nullptr) {
            buffers.insert(গ্ৰন্থি->view_src->buffer);
        }
    }
    বাবে (auto * buffer : buffers) {
        ggml_backend_buffer_clear(buffer, 0);
    }
}

ov::Tensor create_ov_output_tensor(ধ্রুৱ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   ধ্রুৱ std::shared_ptr<ov::InferRequest> & infer_request,
                                   পূর্ণাঙ্ক output_index,
                                   ধ্রুৱ ggml_tensor * ggml_tensor) {
    যদি (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        ঘূৰাই *sliced;
    }

    auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    শূন্য * output_data = ggml_tensor->data;
    যদি (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_index).get_shape();
    } অন্যথা {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        যদি (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != nullptr &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_data = ggml_tensor->view_src->data;
        } অন্যথা {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    যদি (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_buffer_is_remote(ggml_tensor) &&
        ggml_tensor->extra != nullptr) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        যদি (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            যদি (tensor_extra->tensor != nullptr && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                ঘূৰাই *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_data);
    ঘূৰাই output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_rows_to_seq_axis_2(ধ্রুৱ ov::Tensor & kv_tensor, size_t n_heads_kv) {
    ধ্রুৱ size_t rows = kv_tensor.get_shape()[2];
    ধ্রুৱ size_t head_size = kv_tensor.get_shape()[3] / n_heads_kv;
    ধ্রুৱ size_t elem = kv_tensor.get_element_type().size();
    ধ্রুৱ size_t head_bytes = head_size * elem;

    ov::Tensor out(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, rows, head_size});
    ধ্রুৱ auto * src = static_cast<ধ্রুৱ uint8_t *>(kv_tensor.data());
    auto * dst = static_cast<uint8_t *>(out.data());
    বাবে (size_t s = 0; s < rows; s++) {
        বাবে (size_t h = 0; h < n_heads_kv; h++) {
            memcpy(dst + (h * rows + s) * head_bytes, src + (s * n_heads_kv + h) * head_bytes, head_bytes);
        }
    }
    ঘূৰাই out;
}

template <typename T> শূন্য set_zero_diagonal(std::vector<T> & matrix, size_t rows, size_t cols, T zero_মান = T{}) {
    বাবে (size_t i = 0; i < rows; ++i) {
        size_t diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = zero_মান;
    }
}

ov::Tensor make_contiguous_split_প্ৰবেশ_tensor(ধ্রুৱ সংরচনা ggml_tensor * ggml_tensor, ধ্রুৱ ov::Shape & প্ৰবেশ_shape) {
    ধ্রুৱ size_t element_size = ggml_type_size(ggml_tensor->type);
    ধ্রুৱ size_t block_size = ggml_blck_size(ggml_tensor->type);

    GGML_ASSERT(block_size == 1 && "non-contiguous split inputs must be plain element types");

    ধ্রুৱ সংরচনা ggml_tensor * source_tensor = ggml_tensor->view_src != nullptr ? ggml_tensor->view_src : ggml_tensor;
    ধ্রুৱ size_t source_offset = ggml_tensor->view_src != nullptr ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> source_data(ggml_nbytes(source_tensor));
    ggml_backend_tensor_get(source_tensor, source_data.data(), 0, source_data.size());

    ov::Tensor প্ৰবেশ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), প্ৰবেশ_shape);
    auto * dst = static_cast<uint8_t *>(প্ৰবেশ_tensor.data());
    size_t dst_offset = 0;

    বাবে (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        বাবে (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            বাবে (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                বাবে (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    ধ্রুৱ size_t src_offset = source_offset + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::memcpy(dst + dst_offset, source_data.data() + src_offset, element_size);
                    dst_offset += element_size;
                }
            }
        }
    }

    ঘূৰাই প্ৰবেশ_tensor;
}

ov::Tensor convert_ggml_প্ৰবেশ_to_ov(ধ্রুৱ std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ধ্রুৱ std::string & name) {
    ধ্রুৱ auto * ggml_tensor = ggml_decoder->get_প্ৰবেশ_ggml_tensor(name);

    যদি (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        ঘূৰাই *sliced;
    }

    যদি (ggml_tensor->extra != nullptr && !ggml_decoder->is_splited_model()) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        যদি (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            ঘূৰাই *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    auto * প্ৰবেশ_data = ggml_tensor->data;
    ov::Shape প্ৰবেশ_shape;
    যদি (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        প্ৰবেশ_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } অন্যথা {
        প্ৰবেশ_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    যদি (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        ঘূৰাই make_contiguous_split_প্ৰবেশ_tensor(ggml_tensor, প্ৰবেশ_shape);
    }

    auto প্ৰবেশ_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), প্ৰবেশ_shape, প্ৰবেশ_data);
    ঘূৰাই প্ৰবেশ_tensor;
}

ov::Tensor get_ov_প্ৰবেশ_tensor(ধ্রুৱ std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ধ্রুৱ std::string & param_name) {
    ov::Tensor প্ৰবেশ_tensor;
    auto extra_প্ৰবেশ = ggml_decoder->get_model_extra_inputs().find(param_name);
    যদি (extra_প্ৰবেশ != ggml_decoder->get_model_extra_inputs().end()) {
        প্ৰবেশ_tensor = ov::Tensor(extra_প্ৰবেশ->second.type, extra_প্ৰবেশ->second.shape);
        *প্ৰবেশ_tensor.data<int64_t>() = extra_প্ৰবেশ->second.value;
    } অন্যথা {
        প্ৰবেশ_tensor = convert_ggml_প্ৰবেশ_to_ov(ggml_decoder, param_name);
    }
    ঘূৰাই প্ৰবেশ_tensor;
}

ov::Tensor get_ov_প্ৰবেশ_tensor_static_decode(ধ্রুৱ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             ধ্রুৱ std::string & param_name) {
    // NPU decoding stage
    যদি (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        ঘূৰাই get_ov_প্ৰবেশ_tensor(ggml_decoder, param_name);
    }
    ধ্রুৱ auto * ggml_tensor = ggml_decoder->get_প্ৰবেশ_ggml_tensor(param_name);
    ধ্রুৱ auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    যদি (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        ধ্রুৱ পূর্ণাঙ্ক n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        প্রতিজ্ঞা(ggml_tensor->ne[0] == n_planes);
        ov::Shape প্ৰবেশ_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor প্ৰবেশ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), প্ৰবেশ_shape);
        std::memcpy(প্ৰবেশ_tensor.data(), ggml_tensor->data, n_planes * ggml_type_size(ggml_tensor->type));
        ঘূৰাই প্ৰবেশ_tensor;
    }

    যদি (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape প্ৰবেশ_shape = {1, 1, 1, 1};
        ov::Tensor প্ৰবেশ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), প্ৰবেশ_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        প্রতিজ্ঞা(ggml_tensor->ne[0] == 1);
        প্রতিজ্ঞা(inp_out_id == 0);
        *প্ৰবেশ_tensor.data<int32_t>() = inp_out_id;
        ঘূৰাই প্ৰবেশ_tensor;
    }

    যদি (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t context_size = ggml_decoder->get_ctx_size();
        যদি (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_data =
                pad_প্ৰবেশ<ggml_fp16_t>(ggml_tensor, 1, context_size, GGML_FP32_TO_FP16(-INFINITY));
            ov::Tensor প্ৰবেশ_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_size});
            std::memcpy(প্ৰবেশ_tensor.data(), padded_data.data(), padded_data.size() * পরিমাণ(ggml_fp16_t));
            ঘূৰাই প্ৰবেশ_tensor;
        }

        std::vector<ভিন্ন> padded_data = pad_প্ৰবেশ<ভিন্ন>(ggml_tensor, 1, context_size, -INFINITY);
        ov::Tensor প্ৰবেশ_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        auto * data_ptr = প্ৰবেশ_tensor.data<ভিন্ন>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
        ঘূৰাই প্ৰবেশ_tensor;
    }

    ঘূৰাই get_ov_প্ৰবেশ_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_প্ৰবেশ_tensor_static_prefill(ধ্রুৱ std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              ধ্রুৱ std::string & param_name,
                                              পূর্ণাঙ্ক chunk_index) {
    // NPU prompt processing stage
    ধ্রুৱ size_t প্ৰবেশ_len = ggml_decoder->get_প্ৰবেশ_len();
    ধ্রুৱ size_t chunk_size = ggml_decoder->m_prefill_chunk_size;
    ধ্রুৱ size_t chunk_valid_size = std::min(chunk_size, প্ৰবেশ_len - chunk_index * chunk_size);
    ধ্রুৱ size_t chunk_pad_size = chunk_size - chunk_valid_size;

    যদি (param_name == "chunk_valid_len") {
        ov::Tensor প্ৰবেশ_tensor(ov::element::i64, ov::Shape{1});
        *প্ৰবেশ_tensor.data<int64_t>() = (int64_t) chunk_valid_size;
        ঘূৰাই প্ৰবেশ_tensor;
    }
    যদি (chunk_index > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor প্ৰবেশ_tensor(ov::element::i64, ov::Shape{1});
        *প্ৰবেশ_tensor.data<int64_t>() = 0;
        ঘূৰাই প্ৰবেশ_tensor;
    }
    যদি (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        ঘূৰাই get_ov_প্ৰবেশ_tensor(ggml_decoder, param_name);
    }
    ধ্রুৱ auto * ggml_tensor = ggml_decoder->get_প্ৰবেশ_ggml_tensor(param_name);
    ধ্রুৱ auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    যদি (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        ধ্রুৱ পূর্ণাঙ্ক n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        ধ্রুৱ size_t element_size = ggml_type_size(ggml_tensor->type);
        ov::Shape প্ৰবেশ_shape = {1, 1, 1, (size_t) n_planes * chunk_size};
        ov::Tensor প্ৰবেশ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), প্ৰবেশ_shape);
        বাবে (পূর্ণাঙ্ক p = 0; p < n_planes; p++) {
            ধ্রুৱ ৱর্ণ * src =
                (ধ্রুৱ ৱর্ণ *) ggml_tensor->data + (p * প্ৰবেশ_len + chunk_index * chunk_size) * element_size;
            ৱর্ণ * dst = (ৱর্ণ *) প্ৰবেশ_tensor.data() + p * chunk_size * element_size;
            std::memcpy(dst, src, chunk_valid_size * element_size);
            যদি (chunk_pad_size > 0) {
                যদি (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t last_মান = *((ধ্রুৱ int32_t *) src + chunk_valid_size - 1);
                    int32_t * out = (int32_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_মান + 1);
                } অন্যথা যদি (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t last_মান = *((ধ্রুৱ int64_t *) src + chunk_valid_size - 1);
                    int64_t * out = (int64_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_মান + 1);
                } অন্যথা {
                    ক্ষিপ std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        ঘূৰাই প্ৰবেশ_tensor;
    }

    যদি (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape প্ৰবেশ_shape = {1, 1, 1, chunk_size};
        ov::Tensor প্ৰবেশ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), প্ৰবেশ_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        শূন্য * প্ৰবেশ_data = (ৱর্ণ *) ggml_tensor->data + chunk_index * chunk_size * element_size;
        std::memcpy(প্ৰবেশ_tensor.data(), প্ৰবেশ_data, chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        যদি (chunk_pad_size > 0) {
            যদি (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_মান =
                    *((int32_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int32_t * output_data = প্ৰবেশ_tensor.data<int32_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_মান + 1);
            } অন্যথা যদি (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_মান =
                    *((int64_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int64_t * output_data = প্ৰবেশ_tensor.data<int64_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_মান + 1);
            } অন্যথা {
                ক্ষিপ std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        ঘূৰাই প্ৰবেশ_tensor;
    }

    যদি (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape প্ৰবেশ_shape = {1, 1, 1, output_len};
        ov::Tensor প্ৰবেশ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), প্ৰবেশ_shape);
        যদি (ggml_tensor->ne[0] == 0) {
            *প্ৰবেশ_tensor.data<int32_t>() = 0;
        } অন্যথা {
            auto * data_addr = প্ৰবেশ_tensor.data<int32_t>();
            বাবে (size_t i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        ঘূৰাই প্ৰবেশ_tensor;
    }

    যদি (GgmlOvDecoder::is_inp_scale_rows(ggml_tensor, op)) {
        ov::Tensor প্ৰবেশ_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, 1});
        auto * dst = প্ৰবেশ_tensor.data<ভিন্ন>();
        ধ্রুৱ auto * src = static_cast<ধ্রুৱ ভিন্ন *>(ggml_tensor->data) + chunk_index * chunk_size;
        std::copy(src, src + chunk_valid_size, dst);
        std::fill(dst + chunk_valid_size, dst + chunk_size, 1.0f);
        ঘূৰাই প্ৰবেশ_tensor;
    }

    যদি (GgmlOvDecoder::is_inp_mean(ggml_tensor, op)) {
        ধ্রুৱ size_t n_seqs = ggml_tensor->ne[1];
        ধ্রুৱ size_t src_stride = ggml_tensor->ne[0];
        ধ্রুৱ size_t copy_len = std::min<size_t>(chunk_valid_size, src_stride - chunk_index * chunk_size);
        ov::Tensor প্ৰবেশ_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_size});
        auto * dst = প্ৰবেশ_tensor.data<ভিন্ন>();
        std::fill(dst, dst + n_seqs * chunk_size, 0.0f);
        ধ্রুৱ auto * src = static_cast<ধ্রুৱ ভিন্ন *>(ggml_tensor->data) + chunk_index * chunk_size;
        বাবে (size_t s = 0; s < n_seqs; s++) {
            std::memcpy(dst + s * chunk_size, src + s * src_stride, copy_len * পরিমাণ(ভিন্ন));
        }
        ঘূৰাই প্ৰবেশ_tensor;
    }

    যদি (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = ggml_decoder->get_ctx_size();
        যদি (ggml_tensor->type == GGML_TYPE_F16) {
            ধ্রুৱ auto * ggml_data =
                static_cast<ধ্রুৱ ggml_fp16_t *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
            std::vector<ggml_fp16_t> padded_data = pad_প্ৰবেশ<ggml_fp16_t>(ggml_data, chunk_valid_rows, cols, chunk_size,
                                                                          context_size, GGML_FP32_TO_FP16(-INFINITY));
            set_zero_diagonal(padded_data, chunk_size, context_size, GGML_FP32_TO_FP16(0.0f));
            ov::Tensor প্ৰবেশ_tensor(ov::element::f16, ov::Shape{1, 1, chunk_size, context_size});
            std::memcpy(প্ৰবেশ_tensor.data(), padded_data.data(), padded_data.size() * পরিমাণ(ggml_fp16_t));
            ঘূৰাই প্ৰবেশ_tensor;
        }

        ধ্রুৱ auto * ggml_data = static_cast<ধ্রুৱ ভিন্ন *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
        std::vector<ভিন্ন> padded_data =
            pad_প্ৰবেশ<ভিন্ন>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor প্ৰবেশ_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        auto * data_ptr = প্ৰবেশ_tensor.data<ভিন্ন>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        ঘূৰাই প্ৰবেশ_tensor;
    }

    ঘূৰাই get_ov_প্ৰবেশ_tensor(ggml_decoder, param_name);
}

গণনা ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               ধ্রুৱ std::string & device,
                               ধ্রুৱ ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    যদি (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        ঘূৰাই GGML_STATUS_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_lock(cache.mutex);
    তার্কিক naive = সঁচা;
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    auto প্ৰবেশ_model = std::make_shared<ov::frontend::ggml::প্ৰবেশ_Model>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(প্ৰবেশ_model, naive);
    যদি (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    যদি (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    } অন্যথা {
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    }
    যদি (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } অন্যথা {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> প্ৰবেশ_names;
    std::vector<std::string> output_names;
    বাবে (ধ্রুৱ auto & param : model->get_parameters()) {
        প্ৰবেশ_names.push_back(param->get_friendly_name());
    }
    বাবে (ধ্রুৱ auto & result : model->get_results()) {
        output_names.push_back(result->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    প্ৰবেশ_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_lock.unlock();

    বাবে (size_t i = 0; i < প্ৰবেশ_names.size(); i++) {
        ধ্রুৱ auto & param_name = প্ৰবেশ_names[i];
        auto প্ৰবেশ_tensor = get_ov_প্ৰবেশ_tensor(decoder, param_name);
        infer_request->set_প্ৰবেশ_tensor(i, প্ৰবেশ_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    বাবে (size_t i = 0; i < output_names.size(); i++) {
        auto output_tensor = infer_request->get_output_tensor(i);
        ধ্রুৱ auto & model_outputs = decoder->get_model_outputs();
        auto model_output_it = model_outputs.find(output_names[i]);
        যদি (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            যদি (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_names[i], output_tensor, output_tensor.data());
            }
            অনুৱর্ত;
        }
        auto * ggml_tensor = model_output_it->second;
        std::memcpy(ggml_tensor->data, output_tensor.data(), output_tensor.get_byte_size());
    }
    ঘূৰাই GGML_STATUS_SUCCESS;
}

গণনা ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, ধ্রুৱ std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();
    ধ্রুৱ auto & config = ggml_openvino_get_compile_config();
    ধ্রুৱ auto & device = r_ctx->device;
    স্থির auto is_static = মিছা;

    স্থির ধ্রুৱ তার্কিক cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    auto start_time = ggml_time_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    ধ্রুৱ int64_t cache_চাবি_start_time = ggml_time_us();
    graph_চাবি চাবি(cgraph);
    ধ্রুৱ int64_t cache_চাবি_compute_time = ggml_time_us() - cache_চাবি_start_time;
    int64_t cache_lookup_time = 0;
    তার্কিক চাবি_seen = মিছা;
    যদি (!cache_disabled) {
        ধ্রুৱ int64_t cache_lookup_start_time = ggml_time_us();
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        চাবি_seen = r_ctx->decoder_cache.find(চাবি) != r_ctx->decoder_cache.end();
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;
    }

    ধ্রুৱ int64_t model_split_check_start_time = ggml_time_us();
    তার্কিক model_is_splitted = চাবি_seen ? মিছা : is_model_splitted(cgraph);
    ধ্রুৱ int64_t model_split_check_time = ggml_time_us() - model_split_check_start_time;

    যদি (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        ঘূৰাই GGML_STATUS_FAILED;
    }
    যদি (is_naive(cgraph)) {
        যদি (!model_is_splitted) {
            ঘূৰাই naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    ধ্রুৱ int64_t graph_param_compute_start_time = ggml_time_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    ধ্রুৱ int64_t graph_param_compute_time = ggml_time_us() - graph_param_compute_start_time;

    ধ্রুৱ তার্কিক cache_enabled = !model_is_splitted && !cache_disabled;
    ধ্রুৱ তার্কিক stateful_recurrent = r_ctx->stateful && m_params.state_size >= 0;
    ধ্রুৱ তার্কিক stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    যদি (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_active > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        ঘূৰাই GGML_STATUS_FAILED;
    }
    যদি (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        ঘূৰাই GGML_STATUS_FAILED;
    }
    তার্কিক cache_hit = মিছা;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        ধ্রুৱ int64_t cache_lookup_start_time = ggml_time_us();
        যদি (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            auto it = r_ctx->decoder_cache.find(চাবি);
            cache_hit = it != r_ctx->decoder_cache.end();
            যদি (cache_hit) {
                entry = it->second;
            } অন্যথা {
                r_ctx->clear_caches_locked();
                auto mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[চাবি] = entry;
            }
        } অন্যথা {
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_hit = মিছা;
        }
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;

        std::lock_guard<std::mutex> lock(*(entry->mutex));
        cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(চাবি) != 0;

        যদি (cache_hit) {
            ggml_decoder = entry->ptr;
            old_m_params = ggml_decoder->get_model_params();
            যদি (!ggml_decoder->is_splited_model()) {
                cache_hit = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_প্ৰবেশ_names;
        std::vector<std::string> ov_output_names;

        যদি (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            যদি (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(চাবি);
                ov_প্ৰবেশ_names = r_ctx->ov_প্ৰবেশ_names_cache.at(চাবি);
                ov_output_names = r_ctx->ov_output_names_cache.at(চাবি);
            }

            যদি (stateful_kv_only) {
                ধ্রুৱ auto * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_data = (int32_t *) inp_pos->data;
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                যদি (pos_data[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_size = pos_shape[3];
                } অন্যথা যদি (r_ctx->stateful_kv_size == static_cast<size_t>(pos_data[0])) {
                    r_ctx->stateful_kv_size += pos_shape[3];
                } অন্যথা {
                    ধ্রুৱ size_t pos_begin = static_cast<size_t>(pos_data[0]);
                    ধ্রুৱ তার্কিক refill = pos_begin > r_ctx->stateful_kv_size;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    যদি (refill && !ggml_decoder->get_model_params().swa_layers.empty()) {
                        ধ্রুৱ পূর্ণাঙ্ক n_swa = ggml_decoder->get_compute_params().swa_window;
                        যদি (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_size, n_swa);
                            ঘূৰাই GGML_STATUS_FAILED;
                        }
                    }

                    ধ্রুৱ তার্কিক relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    auto states = infer_request->query_state();
                    বাবে (auto state : states) {
                        auto state_tensor = state.get_state();
                        auto state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        যদি (auto it = r_ctx->kv_state_প্ৰবেশ_name_map.find(state.get_name());
                            it != r_ctx->kv_state_প্ৰবেশ_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        পূর্ণাঙ্ক n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        যদি (auto layer = extract_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        ধ্রুৱ তার্কিক relayout_this_state = relayout_enabled;
                        ধ্রুৱ size_t seq_axis = relayout_this_state ? 2 : 1;
                        ধ্রুৱ size_t head_axis = seq_axis == 2 ? 1 : 2;

                        যদি (refill) {
                            যদি (state_name.empty()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                ঘূৰাই GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_প্ৰবেশ_tensor(ggml_decoder, state_name);
                            যদি (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_rows_to_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } অন্যথা {
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
                        যদি (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            ঘূৰাই GGML_STATUS_FAILED;
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
        } অন্যথা {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            auto shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
            auto weight_names = get_weight_names(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ধ্রুৱ std::string model_cache_dir = ggml_openvino_model_cache_dir();
            ধ্রুৱ std::string exact_চাবি = cache_enabled ? compiled_graph_চাবি(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_path, manifest_path;
            যদি (!model_cache_dir.empty() && !model_is_splitted) {
                ধ্রুৱ uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/সঁচা, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_graph_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/সঁচা, m_params.rope_params, 16, extra_cfg,
                                                    compiled_graph_চাবি(cgraph, *ggml_decoder, device, 0, সঁচা));
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            std::string shared_চাবি =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_চাবি) : "";
            ov::CompiledModel shared_model;
            তার্কিক imported = মিছা;
            auto shared_it = shared_cache->graphs.find(shared_চাবি);
            যদি (!shared_চাবি.empty() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_graph(shared_it->second.decode, shared_it->second.প্ৰবেশ_names,
                                                               shared_it->second.output_names, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_প্ৰবেশ_names = shared_it->second.প্ৰবেশ_names;
                ov_output_names = shared_it->second.output_names;
                imported = সঁচা;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } অন্যথা যদি (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_চাবি = exact_চাবি;
                shared_it = shared_cache->graphs.find(shared_চাবি);
                যদি (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_প্ৰবেশ_names = shared_it->second.প্ৰবেশ_names;
                    ov_output_names = shared_it->second.output_names;
                    imported = সঁচা;
                }
                model_fp = exact_fp;
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            যদি (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(চাবি);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            যদি (!model_cache_dir.empty()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
            }
            যদি (!imported && !model_cache_dir.empty() && !model_is_splitted) {
                ধ্রুৱ uint64_t preferred_fp = model_fp;
                তার্কিক dynamic_incompatible = preferred_fp == exact_fp;
                বাবে (পূর্ণাঙ্ক candidate = 0; candidate < 2; ++candidate) {
                    যদি (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        ভঙ্গ;
                    }
                    ধ্রুৱ uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    ধ্রুৱ std::string candidate_blob =
                        ggml_openvino_model_cache_blob_path(model_cache_dir, candidate_fp);
                    ধ্রুৱ std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_path(model_cache_dir, candidate_fp);
                    std::ifstream blob_in(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    যদি (!blob_in.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        অনুৱর্ত;
                    }
                    ধ্রুৱ int64_t import_start = ggml_time_us();
                    যত্ন {
                        ov::CompiledModel cm;
                        auto remote_context = ggml_openvino_get_remote_context();
                        যদি (remote_context.has_value()) {
                            cm = core.import_model(blob_in, remote_context.value(), mc_config);
                        } অন্যথা {
                            cm = core.import_model(blob_in, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        যদি (!saved_inputs.empty() || !saved_outputs.empty()) {
                            ov_প্ৰবেশ_names = std::move(saved_inputs);
                            ov_output_names = std::move(saved_outputs);
                        } অন্যথা {
                            বাবে (ধ্রুৱ auto & p : cm.inputs()) {
                                ov_প্ৰবেশ_names.push_back(p.get_গ্ৰন্থি()->get_friendly_name());
                            }
                            বাবে (ধ্রুৱ auto & o : cm.outputs()) {
                                ov_output_names.push_back(o.get_গ্ৰন্থি()->get_friendly_name());
                            }
                        }
                        যদি (!compiled_model_matches_graph(cm, ov_প্ৰবেশ_names, ov_output_names, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_প্ৰবেশ_names.clear();
                            ov_output_names.clear();
                            অনুৱর্ত;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        entry->ptr = ggml_decoder;
                        imported = সঁচা;
                        model_fp = candidate_fp;
                        blob_path = candidate_blob;
                        manifest_path = candidate_manifest;
                        যদি (candidate_fp == exact_fp) {
                            shared_চাবি = exact_চাবি;
                            shared_it = shared_cache->graphs.find(shared_চাবি);
                        }
                        যদি (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_time_us() - import_start) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } গৃহাণ (ধ্রুৱ std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = মিছা;
                        ov_প্ৰবেশ_names.clear();
                        ov_output_names.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                যদি (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_চাবি = exact_চাবি;
                    shared_it = shared_cache->graphs.find(shared_চাবি);
                    blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                    manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            যদি (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_path.c_str());
                ঘূৰাই GGML_STATUS_FAILED;
            }
            যদি (!imported && ggml_openvino_weight_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                ঘূৰাই GGML_STATUS_FAILED;
            }
            যদি (imported) {
                decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            } অন্যথা {
                auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_end_time = ggml_time_us();

                auto প্ৰবেশ_model = std::make_shared<ov::frontend::ggml::প্ৰবেশ_Model>(ggml_decoder);
                model = ov::frontend::ggml::FrontEnd::convert(প্ৰবেশ_model);
                ggml_decoder->clear_model_weights();
                conversion_end_time = ggml_time_us();

                যদি (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    ৱর্ণ timestamped_filename[64];
                    auto timestamp = (long long) ggml_time_us();
                    snprintf(timestamped_filename, পরিমাণ(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                ধ্রুৱ ov::AnyMap & compile_config = model_cache_dir.empty() ? config : mc_config;
                ov::CompiledModel compiled_model;
                auto remote_context = ggml_openvino_get_remote_context();
                যদি (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } অন্যথা {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_end_time = ggml_time_us();

                বাবে (ধ্রুৱ auto & ov_param : model->get_parameters()) {
                    ov_প্ৰবেশ_names.push_back(ov_param->get_friendly_name());
                }
                বাবে (ধ্রুৱ auto & ov_output : model->get_results()) {
                    ov_output_names.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                যদি (!model_cache_dir.empty() && !model_is_splitted && model_fp != 0) {
                    যত্ন {
                        ধ্রুৱ std::string blob_tmp = ggml_openvino_model_cache_temp_path(blob_path);
                        ধ্রুৱ std::string manifest_tmp = ggml_openvino_model_cache_temp_path(manifest_path);
                        যদি (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_প্ৰবেশ_names,
                                                                     ov_output_names)) {
                            std::ofstream blob_out(blob_tmp, std::ios::binary | std::ios::trunc);
                            যদি (blob_out.is_open()) {
                                compiled_model.export_model(blob_out);
                                blob_out.close();
                                যদি (blob_out.good()) {
                                    যদি (std::rename(blob_tmp.c_str(), blob_path.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_path.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_path.c_str());
                                    } অন্যথা {
                                        std::remove(blob_tmp.c_str());
                                        std::remove(manifest_tmp.c_str());
                                    }
                                } অন্যথা {
                                    std::remove(blob_tmp.c_str());
                                    std::remove(manifest_tmp.c_str());
                                }
                            } অন্যথা {
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    } গৃহাণ (ধ্রুৱ std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                entry->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            entry->ptr = ggml_decoder;
            যদি (!shared_চাবি.empty() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_চাবি,
                                             ov_compiled_graph{shared_model, {}, ov_প্ৰবেশ_names, ov_output_names});
            }
            যদি (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[চাবি] = infer_request;
                r_ctx->ov_প্ৰবেশ_names_cache[চাবি] = ov_প্ৰবেশ_names;
                r_ctx->ov_output_names_cache[চাবি] = ov_output_names;
            }

            যদি (stateful_kv_only && cache_enabled) {
                ধ্রুৱ auto * inp_pos = get_inp_pos_tensor(cgraph);
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                ধ্রুৱ int32_t pos_begin = ((int32_t *) inp_pos->data)[0];
                যদি (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    ঘূৰাই GGML_STATUS_FAILED;
                }
                r_ctx->stateful_kv_size = pos_shape[3];
                ধ্রুৱ auto kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                বাবে (ধ্রুৱ auto & pair : kv_param_res_names) {
                    r_ctx->kv_state_প্ৰবেশ_name_map[pair.first + pair.second] = pair.first;
                }
            }
        }

        যদি (stateful_recurrent) {
            ধ্রুৱ auto * inp_pos = get_inp_pos_tensor(cgraph);
            ধ্রুৱ int32_t pos_begin = static_cast<ধ্রুৱ int32_t *>(inp_pos->data)[0];
            যদি (pos_begin == 0) {
                infer_request->reset_state();
            } অন্যথা যদি (!cache_hit || old_m_params.kv_buffer_changed(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_size) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                ঘূৰাই GGML_STATUS_FAILED;
            }
        } অন্যথা {
            reset_single_slot_recurrent_cache(cgraph, m_params, c_params);
        }

        বাবে (size_t i = 0; i < ov_প্ৰবেশ_names.size(); i++) {
            ধ্রুৱ auto & param_name = ov_প্ৰবেশ_names[i];
            auto প্ৰবেশ_tensor = get_ov_প্ৰবেশ_tensor(ggml_decoder, param_name);
            infer_request->set_প্ৰবেশ_tensor(i, প্ৰবেশ_tensor);

            যদি (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_প্ৰবেশ_tensor_info(param_name, প্ৰবেশ_tensor);
            }
        }

        বাবে (size_t i = 0; i < ov_output_names.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            ধ্রুৱ auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names[i]);
            যদি (model_output_it == model_outputs.end()) {
                অনুৱর্ত;
            }
            auto * ggml_tensor = model_output_it->second;
            যদি (ggml_nbytes(ggml_tensor) == 0) {
                অনুৱর্ত;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        যদি (stateful_recurrent) {
            ধ্রুৱ auto * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_size =
                static_cast<ধ্রুৱ int32_t *>(inp_pos->data)[0] + get_inp_pos_n_tokens(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        যদি (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            বাবে (size_t i = 0; i < ov_output_names.size(); i++) {
                ধ্রুৱ auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }

        যদি (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_চাবি_compute_time / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_split_check_time / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", graph_param_compute_time / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_time / 1000.0);
            যদি (!cache_hit) {
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
    যদি (cache_hit && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_lock(r_ctx->compiled_cache->mutex);
        যদি (!ggml_openvino_weight_buffers_released()) {
            ggml_openvino_release_weight_buffers();
        }
    }

    ঘূৰাই GGML_STATUS_SUCCESS;
}

ov::AnyMap without_npuw(ধ্রুৱ ov::AnyMap & config) {
    ov::AnyMap out;
    বাবে (ধ্রুৱ auto & kv : config) {
        যদি (kv.first.rfind("NPUW", 0) == 0 || kv.first == "NPU_USE_NPUW") {
            অনুৱর্ত;
        }
        out.insert(kv);
    }
    ঘূৰাই out;
}

গণনা ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, ধ্রুৱ std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();

    auto get_prefill_chunk_size = [] {
        স্থির ধ্রুৱ পূর্ণাঙ্ক chunk_size = []() {
            পূর্ণাঙ্ক env_prefill_chunk_size = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            ঘূৰাই env_prefill_chunk_size > 0 ? env_prefill_chunk_size : 256;
        }();
        ঘূৰাই chunk_size;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    স্থির std::string device = ggml_openvino_get_device_name();
    স্থির auto is_static = সঁচা;
    স্থির auto stateful = মিছা;

    auto prefill_chunk_size = get_prefill_chunk_size();
    ধ্রুৱ auto & config = ggml_openvino_get_compile_config();

    যদি (is_naive(cgraph)) {
        ঘূৰাই naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    ধ্রুৱ auto * inp_pos = get_inp_pos_tensor(cgraph);
    ধ্রুৱ তার্কিক no_kv_cache = m_params.is_cacheless_attn;
    ধ্রুৱ auto is_prefill = no_kv_cache ? সঁচা : get_is_prefill(cgraph, inp_pos);
    ধ্রুৱ ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    যদি (m_params.n_heads_kv == -1) {
        prefill_chunk_size = inp_pos->ne[0];
    }
    graph_চাবি চাবি(cgraph);
    স্থির ধ্রুৱ তার্কিক cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    তার্কিক cache_hit = মিছা;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;
    int64_t ov_raw_infer_total = 0;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    যদি (cache_enabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        auto it = r_ctx->decoder_cache.find(চাবি);
        cache_hit = it != r_ctx->decoder_cache.end();
        যদি (cache_hit) {
            entry = it->second;
        } অন্যথা {
            r_ctx->clear_caches_locked();
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[চাবি] = entry;
        }
    } অন্যথা {
        auto mutex = std::make_shared<std::mutex>();
        entry = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_hit = মিছা;
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));
    cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(চাবি) != 0 &&
                r_ctx->infer_request_cache_prefill.count(চাবি) != 0;

    যদি (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_প্ৰবেশ_names_local;
    std::vector<std::string> ov_output_names_local;

    যদি (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        যদি (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(চাবি) : r_ctx->infer_request_cache.at(চাবি);
            ov_প্ৰবেশ_names_local = r_ctx->ov_প্ৰবেশ_names_cache.at(চাবি);
            ov_output_names_local = r_ctx->ov_output_names_cache.at(চাবি);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } অন্যথা {
        যদি (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(চাবি);
            r_ctx->infer_request_cache_prefill.erase(চাবি);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        auto shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
        auto weight_names = get_weight_names(cgraph);
        auto local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names, is_static,
                                                             stateful, মিছা, is_prefill, prefill_chunk_size);
        ধ্রুৱ std::string shared_চাবি =
            cache_enabled ? compiled_graph_চাবি(cgraph, *local_decoder, device, prefill_chunk_size) : "";
        auto shared_it = shared_cache->graphs.find(shared_চাবি);
        যদি (!shared_চাবি.empty() && shared_it != shared_cache->graphs.end()) {
            auto & compiled = shared_it->second;
            auto prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            auto decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            entry->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_প্ৰবেশ_names_local = compiled.প্ৰবেশ_names;
            ov_output_names_local = compiled.output_names;
            r_ctx->infer_request_cache_prefill[চাবি] = prefill_request;
            r_ctx->infer_request_cache[চাবি] = decode_request;
            r_ctx->ov_প্ৰবেশ_names_cache[চাবি] = ov_প্ৰবেশ_names_local;
            r_ctx->ov_output_names_cache[চাবি] = ov_output_names_local;
            decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } অন্যথা {
            std::shared_ptr<ov::Model> model;
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

            auto ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, মিছা, সঁচা, prefill_chunk_size);
            auto ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, মিছা, মিছা, prefill_chunk_size);
            decoder_end_time = ggml_time_us();

            ধ্রুৱ তার্কিক dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            ধ্রুৱ auto dump_ir_timestamp = static_cast<long long>(ggml_time_us());

            auto build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          ধ্রুৱ std::shared_ptr<GgmlOvDecoder> & decoder, ধ্রুৱ ৱর্ণ * tag,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_end_time, int64_t & local_compile_end_time) {
                auto প্ৰবেশ_model = std::make_shared<ov::frontend::ggml::প্ৰবেশ_Model>(decoder);
                model = ov::frontend::ggml::FrontEnd::convert(প্ৰবেশ_model);
                decoder->clear_model_weights();
                local_conversion_end_time = ggml_time_us();

                যদি (dump_ir) {
                    ৱর্ণ timestamped_filename[64];
                    snprintf(timestamped_filename, পরিমাণ(timestamped_filename), "model_%s_%lld.xml", tag,
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
            যদি (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_end_time = prefill_conversion_end_time;
                decode_compile_end_time = prefill_compile_end_time;
            } অন্যথা {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_end_time, decode_compile_end_time);
            }
            conversion_end_time = std::max(prefill_conversion_end_time, decode_conversion_end_time);
            compile_end_time = std::max(prefill_compile_end_time, decode_compile_end_time);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            entry->ptr = ggml_decoder;

            বাবে (ধ্রুৱ auto & ov_param : model->get_parameters()) {
                ov_প্ৰবেশ_names_local.push_back(ov_param->get_friendly_name());
            }
            বাবে (ধ্রুৱ auto & ov_output : model->get_results()) {
                ov_output_names_local.push_back(ov_output->get_friendly_name());
            }

            যদি (!shared_চাবি.empty()) {
                shared_cache->graphs.emplace(
                    shared_চাবি, ov_compiled_graph{compiled_model_decode, compiled_model_prefill, ov_প্ৰবেশ_names_local,
                                                  ov_output_names_local});
            }

            যদি (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[চাবি] = infer_request_prefill;
                r_ctx->infer_request_cache[চাবি] = infer_request_decode;
                r_ctx->ov_প্ৰবেশ_names_cache[চাবি] = ov_প্ৰবেশ_names_local;
                r_ctx->ov_output_names_cache[চাবি] = ov_output_names_local;
            }
        }
    }

    reset_single_slot_recurrent_cache(cgraph, m_params, c_params);

    যদি (is_prefill) {
        auto inp_len = get_inp_pos_n_tokens(cgraph, inp_pos);
        বাবে (পূর্ণাঙ্ক chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            বাবে (size_t i = 0; i < ov_প্ৰবেশ_names_local.size(); i++) {
                ধ্রুৱ auto & param_name = ov_প্ৰবেশ_names_local[i];
                auto প্ৰবেশ_tensor = get_ov_প্ৰবেশ_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_প্ৰবেশ_tensor(i, প্ৰবেশ_tensor);

                যদি (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    ধ্রুৱ auto প্ৰবেশ_tensor = infer_request->get_প্ৰবেশ_tensor(i);
                    print_প্ৰবেশ_tensor_info(param_name, প্ৰবেশ_tensor);
                }
            }

            বাবে (size_t i = 0; i < ov_output_names_local.size(); i++) {
                ধ্রুৱ auto & model_outputs = ggml_decoder->get_model_outputs();
                auto model_output_it = model_outputs.find(ov_output_names_local[i]);
                যদি (model_output_it == model_outputs.end()) {
                    অনুৱর্ত;
                }
                auto * ggml_tensor = model_output_it->second;
                যদি (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    অনুৱর্ত;
                }
                auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            ov_raw_infer_start = ggml_time_us();
            infer_request->infer();
            ov_raw_infer_total += ggml_time_us() - ov_raw_infer_start;

            যদি (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                বাবে (size_t i = 0; i < ov_output_names_local.size(); i++) {
                    ধ্রুৱ auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } অন্যথা {
        বাবে (size_t i = 0; i < ov_প্ৰবেশ_names_local.size(); i++) {
            ধ্রুৱ auto & param_name = ov_প্ৰবেশ_names_local[i];
            auto প্ৰবেশ_tensor = get_ov_প্ৰবেশ_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_প্ৰবেশ_tensor(i, প্ৰবেশ_tensor);

            যদি (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                ধ্রুৱ auto প্ৰবেশ_tensor = infer_request->get_প্ৰবেশ_tensor(i);
                print_প্ৰবেশ_tensor_info(param_name, প্ৰবেশ_tensor);
            }
        }

        বাবে (size_t i = 0; i < ov_output_names_local.size(); i++) {
            ধ্রুৱ auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names_local[i]);
            যদি (model_output_it == model_outputs.end()) {
                অনুৱর্ত;
            }
            auto * ggml_tensor = model_output_it->second;
            যদি (ggml_nbytes(ggml_tensor) == 0) {
                অনুৱর্ত;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        ov_raw_infer_total = infer_end_time - ov_raw_infer_start;

        যদি (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            বাবে (size_t i = 0; i < ov_output_names_local.size(); i++) {
                ধ্রুৱ auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
            }
        }
    }

    যদি (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
        যদি (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_end_time - decoder_end_time) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_total / 1000.0);
    }

    ঘূৰাই GGML_STATUS_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
স্থির ggml_cgraph * get_compute_graph(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    auto is_skipped = [](ধ্রুৱ ggml_tensor * গ্ৰন্থি) {
        ঘূৰাই গ্ৰন্থি->op != GGML_OP_NONE && !(গ্ৰন্থি->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    যদি (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        ঘূৰাই cgraph;
    }
    auto & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](ধ্রুৱ ggml_tensor * গ্ৰন্থি) { ঘূৰাই !is_skipped(গ্ৰন্থি); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.data();
    compute.graph.n_nodes = (পূর্ণাঙ্ক) compute.nodes.size();
    ঘূৰাই &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
গণনা ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    যত্ন {
        যদি (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        ধ্রুৱ auto is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_lock(r_ctx->execution_mutex);
        cgraph = get_compute_graph(cgraph, *r_ctx);

        ঘূৰাই is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } গৃহাণ (ধ্রুৱ ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        ঘূৰাই GGML_STATUS_FAILED;
    } গৃহাণ (ধ্রুৱ std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        ঘূৰাই GGML_STATUS_FAILED;
    } গৃহাণ (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        ঘূৰাই GGML_STATUS_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
তার্কিক is_model_splitted(ggml_cgraph * cgraph) {
    স্থির ধ্রুৱ তার্কিক fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    যদি (!fallback_enabled) {
        ঘূৰাই মিছা;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    যদি (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        ঘূৰাই মিছা;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];
        পূর্ণাঙ্ক use_গুনতি = cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, গ্ৰন্থি)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        যদি ((cgraph->n_nodes <= 1 && use_গুনতি == 0) ||
            (cgraph->n_nodes <= 1 && গ্ৰন্থি->op == GGML_OP_VIEW && use_গুনতি == 1 && গ্ৰন্থি->src[0] != nullptr &&
             গ্ৰন্থি->src[0]->op == GGML_OP_NONE)) {
            ঘূৰাই মিছা;
        }
        যদি (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_PERMUTE)) {
            ঘূৰাই মিছা;
        }
        পূর্ণাঙ্ক প্ৰবেশ_use_গুনতি = 0;
        বাবে (পূর্ণাঙ্ক j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_গ্ৰন্থি = cgraph->nodes[j];
            বাবে (পূর্ণাঙ্ক k = 0; k < GGML_MAX_SRC; k++) {
                যদি (other_গ্ৰন্থি->src[k] == গ্ৰন্থি) {
                    প্ৰবেশ_use_গুনতি++;
                }
            }
        }
        যদি (use_গুনতি != প্ৰবেশ_use_গুনতি && গ্ৰন্থি->op != GGML_OP_NONE) {
            ঘূৰাই সঁচা;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    যদি (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_weight_names(cgraph);
    } অন্যথা {
        বাবে (ধ্রুৱ auto & kv : GgmlOvDecoder::create_weight_nodes(cgraph, সঁচা)) {
            model_weights.insert(kv.first);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * গ্ৰন্থি = cgraph->nodes[i];
        বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = গ্ৰন্থি->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            যদি (src != nullptr && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.empty() == মিছা &&
                model_leafs.find(src) == model_leafs.end()) {
                যদি (GgmlOvDecoder::is_inp_tok(src, গ্ৰন্থি)) {
                    ঘূৰাই মিছা;
                }
                ঘূৰাই সঁচা;
            }
        }
    }
    ঘূৰাই মিছা;
}

তার্কিক is_naive(ggml_cgraph * cgraph) {
    constexpr পূর্ণাঙ্ক naive_graph_size_threshold = 20;
    পূর্ণাঙ্ক count = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
        যদি (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    ঘূৰাই count < naive_graph_size_threshold;
}

size_t checksum(ধ্রুৱ শূন্য * data, size_t size) {
    ধ্রুৱ uint8_t * bytes = static_cast<ধ্রুৱ uint8_t *>(data);
    size_t sum = 0;
    বাবে (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    ঘূৰাই sum;
}

তার্কিক save_ggml_tensor_data_to_txt(ধ্রুৱ ggml_tensor * tensor, ধ্রুৱ std::string & file_path) {
    যদি (tensor == nullptr || tensor->data == nullptr) {
        ঘূৰাই মিছা;
    }

    std::ofstream out(file_path);
    যদি (!out.is_open()) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ size_t n = ggml_nelements(tensor);
    out << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    ৱিকল্প (tensor->type) {
    পক্ষ GGML_TYPE_F32: {
        ধ্রুৱ auto * data = static_cast<ধ্রুৱ ভিন্ন *>(tensor->data);
        বাবে (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        ভঙ্গ;
    }
    পক্ষ GGML_TYPE_F16: {
        ধ্রুৱ auto * data = static_cast<ধ্রুৱ ggml_fp16_t *>(tensor->data);
        বাবে (size_t i = 0; i < n; ++i) {
            out << ggml_fp16_to_fp32(data[i]) << '\n';
        }
        ভঙ্গ;
    }
    পক্ষ GGML_TYPE_BF16: {
        ধ্রুৱ auto * data = static_cast<ধ্রুৱ ggml_bf16_t *>(tensor->data);
        বাবে (size_t i = 0; i < n; ++i) {
            out << ggml_bf16_to_fp32(data[i]) << '\n';
        }
        ভঙ্গ;
    }
    পক্ষ GGML_TYPE_I32: {
        ধ্রুৱ auto * data = static_cast<ধ্রুৱ int32_t *>(tensor->data);
        বাবে (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        ভঙ্গ;
    }
    পক্ষ GGML_TYPE_I64: {
        ধ্রুৱ auto * data = static_cast<ধ্রুৱ int64_t *>(tensor->data);
        বাবে (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        ভঙ্গ;
    }
    স্ৱভাৱ:
        out << "unsupported tensor type for text dump" << '\n';
        ঘূৰাই মিছা;
    }

    ঘূৰাই সঁচা;
}

শূন্য print_প্ৰবেশ_tensor_info(ধ্রুৱ std::string & name, ধ্রুৱ ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << '\n';
    ৱিকল্প (tensor.get_element_type()) {
    পক্ষ ov::element::f32: {
        যদি (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::cout << *(tensor.data<ভিন্ন>()) << '\n';
        } অন্যথা {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            ধ্রুৱ ভিন্ন * data = tensor.data<ভিন্ন>();
            বাবে (size_t i = 0; i < rows; ++i) {
                বাবে (size_t j = 0; j < cols; ++j) {
                    ভিন্ন val = data[i * cols + j];
                    যদি (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } অন্যথা {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << '\n';
            }
        }

        ভঙ্গ;
    }
    পক্ষ ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << '\n';
        ভঙ্গ;
    পক্ষ ov::element::i32:
        বাবে (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << ' ';
        }
        std::cout << '\n';
        ভঙ্গ;
    পক্ষ ov::element::i64:
        বাবে (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << ' ';
        }
        std::cout << '\n';
        ভঙ্গ;
    স্ৱভাৱ:
        ভঙ্গ;
    }
}

শূন্য print_output_tensor_info(ধ্রুৱ std::string & name, ধ্রুৱ ov::Tensor & tensor, ধ্রুৱ শূন্য * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    auto print_float_stats = [](ধ্রুৱ std::string & type_name, size_t size, auto get_মান) {
        যদি (size == 0) {
            ঘূৰাই;
        }

        ভিন্ন first = get_মান(0);
        ভিন্ন min = first;
        ভিন্ন max = first;
        দ্ৱিগুণ sum = first;

        বাবে (size_t i = 1; i < size; ++i) {
            ভিন্ন v = get_মান(i);
            min = std::min(v, min);
            max = std::max(v, max);
            sum += v;
        }
        দ্ৱিগুণ mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << '\n';
    };

    ৱিকল্প (tensor.get_element_type()) {
    পক্ষ ov::element::f32: {
        ধ্রুৱ ভিন্ন * data = tensor.data<ভিন্ন>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { ঘূৰাই data[i]; });
        ভঙ্গ;
    }
    পক্ষ ov::element::f16: {
        ধ্রুৱ ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { ঘূৰাই static_cast<ভিন্ন>(data[i]); });
        ভঙ্গ;
    }
    স্ৱভাৱ:
        ভঙ্গ;
    }
}

ধ্রুৱ ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            যদি (src == nullptr) {
                ভঙ্গ;
            }
            যদি (GgmlOvDecoder::is_inp_pos(src, op)) {
                ঘূৰাই src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    ক্ষিপ std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_tokens(ggml_cgraph * cgraph, ধ্রুৱ ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    পূর্ণাঙ্ক n_planes = 1;
    বাবে (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        বাবে (পূর্ণাঙ্ক j = 0; j < GGML_MAX_SRC; ++j) {
            যদি (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                ভঙ্গ;
            }
        }
    }
    ঘূৰাই inp_pos->ne[0] / n_planes;
}

তার্কিক get_is_prefill(ggml_cgraph * cgraph, ধ্রুৱ ggml_tensor * inp_pos) {
    ঘূৰাই get_inp_pos_n_tokens(cgraph, inp_pos) > 1;
}
