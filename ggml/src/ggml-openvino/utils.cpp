#ضمن "utils.h"

#ضمن "ggml-impl.h"
#ضمن "ggml-openvino-extra.h"
#ضمن "ggml-openvino.h"
#ضمن "ggml-openvino/ggml-decoder.h"
#ضمن "ggml.h"
#ضمن "model-cache.h"
#ضمن "openvino/frontend.h"
#ضمن "openvino/input_model.h"

#ضمن <algorithm>
#ضمن <cassert>
#ضمن <cmath>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <fstream>
#ضمن <functional>
#ضمن <iomanip>
#ضمن <iostream>
#ضمن <memory>
#ضمن <openvino/core/any.hpp>
#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/shape.hpp>
#ضمن <openvino/core/type/float16.hpp>
#ضمن <openvino/frontend/manager.hpp>
#ضمن <openvino/openvino.hpp>
#ضمن <openvino/runtime/compiled_model.hpp>
#ضمن <openvino/runtime/infer_request.hpp>
#ضمن <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#ضمن <openvino/runtime/intel_npu/properties.hpp>
#ضمن <openvino/runtime/properties.hpp>
#ضمن <openvino/runtime/tensor.hpp>
#ضمن <optional>
#ضمن <set>
#ضمن <string>
#ضمن <unordered_map>
#ضمن <vector>

نطاق {
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
std::optional<ov::Tensor> try_make_kv_sliced_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    ثابت std::string & name,
                                                    ثابت ggml_tensor * ggml_tensor) {
    ساكن ثابت منطقي kv_slice_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    إذا (kv_slice_disabled) {
        أرجع std::nullopt;
    }
    إذا (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        أرجع std::nullopt;
    }
    إذا (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != nullptr) {
        أرجع std::nullopt;
    }
    ثابت auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);
    إذا (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        أرجع std::nullopt;
    }

    ثابت auto & compute_params = ggml_decoder->get_compute_params();
    إذا (compute_params.n_seq_active != 1 || compute_params.seq_active_start != 0) {
        أرجع std::nullopt;
    }

    صحيح layer;
    إذا (auto layer_opt = extract_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } وإلا {
        أرجع std::nullopt;
    }

    ثابت منطقي is_swa = ggml_decoder->is_swa_layer(layer);
    إذا (is_swa) {
        أرجع std::nullopt;
    }
    ثابت صحيح ctx_per_seq = ggml_decoder->get_ctx_per_seq();
    ثابت صحيح n_kv = compute_params.attention_size;
    إذا (ctx_per_seq <= 0 || n_kv <= 0 || n_kv >= ctx_per_seq) {
        أرجع std::nullopt;
    }

    ov::Shape full_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    إذا (full_shape.size() != 4 || full_shape[0] != 1 || full_shape[1] != 1 ||
        static_cast<صحيح>(full_shape[2]) != ctx_per_seq) {
        أرجع std::nullopt;
    }

    ov::Shape sliced_shape = full_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    إذا (ggml_openvino_buffer_is_remote(ggml_tensor)) {
        auto remote_context = ggml_openvino_get_remote_context();
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        أرجع gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
    }

    أرجع ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
}

ساكن uint64_t ggml_openvino_model_cache_extra_cfg(منطقي stateful, منطقي recurrent) {
    ثابت حرف * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    ثابت منطقي manual_gqa_enabled = manual_gqa_env != nullptr ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    أرجع extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_weight_names(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> names;
    لكل (ثابت auto & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
        names[name] = nullptr;
    }
    أرجع names;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
ساكن std::string compiled_graph_کلید(ثابت ggml_cgraph * graph,
                                      ثابت GgmlOvDecoder & decoder,
                                      ثابت std::string & device,
                                      صحيح prefill_chunk_size = 0,
                                      منطقي disk_cache = خطأ) {
    std::string کلید;
    auto append = [&کلید](ثابت auto & value) {
        کلید.append(reinterpret_cast<ثابت حرف *>(&value), حجم(value));
    };
    auto append_string = [&](ثابت std::string & value) {
        append(value.size());
        کلید.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_size);
    منطقي has_weight_buffer_id = خطأ;
    std::unordered_map<ثابت ggml_tensor *, size_t> ids;
    std::function<فراغ(ثابت ggml_tensor *)> visit = [&](ثابت ggml_tensor * tensor) {
        إذا (!tensor) {
            append(size_t(0));
            أرجع;
        }
        auto inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.first->second);
        إذا (!inserted.second) {
            أرجع;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        ثابت auto * base = tensor->view_src ? tensor->view_src : tensor;
        ثابت منطقي weight = base->buffer && base->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        append(weight);
        إذا (weight && !disk_cache) {
            ثابت size_t buffer_id = ggml_backend_openvino_buffer_get_ctx_id(base->buffer);
            has_weight_buffer_id |= buffer_id != 0;
            append(buffer_id);
            append(tensor->data);
        }
        visit(tensor->view_src);
        لكل (ثابت auto * src : tensor->src) {
            visit(src);
        }
    };
    append(graph->n_nodes);
    لكل (صحيح i = 0; i < graph->n_nodes; ++i) {
        visit(graph->nodes[i]);
    }
    append(graph->n_leafs);
    لكل (صحيح i = 0; i < graph->n_leafs; ++i) {
        visit(graph->leafs[i]);
    }
    لكل (ثابت auto & داخلہ : decoder.get_model_extra_inputs()) {
        append_string(داخلہ.first);
        append_string(داخلہ.second.type.get_type_name());
        append(داخلہ.second.shape.size());
        لكل (auto dim : داخلہ.second.shape) {
            append(dim);
        }
        append(داخلہ.second.is_parameter);
        إذا (!داخلہ.second.is_parameter) {
            append(داخلہ.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    أرجع disk_cache || has_weight_buffer_id ? کلید : std::string{};
}

ساكن std::string dynamic_graph_signature(ثابت ggml_cgraph * graph,
                                           ثابت GgmlOvDecoder & decoder,
                                           ثابت ModelParams & params) {
    std::string کلید = "dynamic-1";
    auto append = [&کلید](ثابت auto & value) {
        کلید.append(reinterpret_cast<ثابت حرف *>(&value), حجم(value));
    };
    auto append_string = [&](ثابت std::string & value) {
        append(value.size());
        کلید.append(value);
    };
    graph_کلید graph_id(graph, true);
    append(graph_id.n_nodes);
    append(graph_id.n_leaves);
    append_string(graph_id.first_گرہ_name);
    append_string(graph_id.last_گرہ_name);
    لكل (ثابت auto & name : graph_id.داخلہ_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    لكل (صحيح layer : params.swa_layers) {
        append(layer);
    }
    لكل (ثابت auto & [layer, heads] : params.n_heads_kv_per_layer) {
        append(layer);
        append(heads);
    }
    لكل (ثابت auto & [name, داخلہ] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(داخلہ.type.get_type_name());
    }
    لكل (ثابت auto & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    لكل (ثابت auto & [name, داخلہ] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(داخلہ.type.get_type_name());
        append(داخلہ.is_parameter);
        إذا (!داخلہ.is_parameter) {
            append(داخلہ.value);
        }
    }
    أرجع کلید;
}

ساكن منطقي compiled_model_matches_graph(ثابت ov::CompiledModel & model,
                                         ثابت std::vector<std::string> & inputs,
                                         ثابت std::vector<std::string> & outputs,
                                         ثابت GgmlOvDecoder & decoder) {
    إذا (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        أرجع خطأ;
    }
    ثابت auto & graph_inputs = decoder.get_model_inputs();
    ثابت auto & extra_inputs = decoder.get_model_extra_inputs();
    ثابت auto & graph_outputs = decoder.get_model_outputs();
    لكل (size_t i = 0; i < inputs.size(); ++i) {
        ثابت auto & port = model.داخلہ(i);
        auto graph_it = graph_inputs.find(inputs[i]);
        إذا (graph_it != graph_inputs.end()) {
            إذا (port.get_element_type() != graph_it->second.type ||
                !port.get_partial_shape().compatible(graph_it->second.shape)) {
                أرجع خطأ;
            }
            استمر;
        }
        auto extra_it = extra_inputs.find(inputs[i]);
        إذا (extra_it == extra_inputs.end() || port.get_element_type() != extra_it->second.type ||
            !port.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            أرجع خطأ;
        }
    }
    لكل (size_t i = 0; i < outputs.size(); ++i) {
        auto graph_it = graph_outputs.find(outputs[i]);
        إذا (graph_it == graph_outputs.end()) {
            إذا (outputs[i].compare(0, 8, "__debug_") != 0) {
                أرجع خطأ;
            }
            استمر;
        }
        ثابت auto & port = model.output(i);
        إذا (port.get_element_type() != GgmlOvDecoder::get_ov_type(graph_it->second) ||
            !port.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(graph_it->second)))) {
            أرجع خطأ;
        }
    }
    أرجع true;
}

ساكن std::string ov_profiling_csv_field(ثابت std::string & value) {
    std::string escaped = "\"";
    لكل (حرف c : value) {
        escaped += c;
        إذا (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    أرجع escaped;
}

ساكن فراغ dump_ov_profiling_info(ثابت ov::InferRequest & infer_request) {
    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        أرجع;
    }

    ساكن std::atomic<uint64_t> inference_index{0};
    ثابت uint64_t index = inference_index.fetch_add(1);
    ثابت std::string path = "openvino_profile_" + std::to_string(index) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    إذا (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        أرجع;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t total_real_time = 0;
    int64_t total_cpu_time = 0;
    size_t executed_گنتی = 0;
    لكل (ثابت auto & info : infer_request.get_profiling_info()) {
        ثابت حرف * status = "NOT_RUN";
        إذا (info.status == ov::ProfilingInfo::Status::EXECUTED) {
            status = "EXECUTED";
            total_real_time += info.real_time.count();
            total_cpu_time += info.cpu_time.count();
            executed_گنتی++;
        } وإلا إذا (info.status == ov::ProfilingInfo::Status::OPTIMIZED_OUT) {
            status = "OPTIMIZED_OUT";
        }

        output << status << ',' << info.real_time.count() << ',' << info.cpu_time.count() << ','
               << info.start_time.count() << ',' << ov_profiling_csv_field(info.node_type) << ','
               << ov_profiling_csv_field(info.گرہ_name) << ',' << ov_profiling_csv_field(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_گنتی, total_real_time / 1000.0, total_cpu_time / 1000.0);
}

ساكن فراغ reset_single_slot_recurrent_cache(ggml_cgraph * cgraph,
                                              ثابت ModelParams & model_params,
                                              ثابت ComputeParams & compute_params) {
    إذا (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        أرجع;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_buffer_t> buffers;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        auto * گرہ = cgraph->nodes[i];
        إذا (گرہ->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(گرہ->view_src) &&
            گرہ->view_src->ne[1] == 1 && گرہ->view_src->buffer != nullptr) {
            buffers.insert(گرہ->view_src->buffer);
        }
    }
    لكل (auto * buffer : buffers) {
        ggml_backend_buffer_clear(buffer, 0);
    }
}

ov::Tensor create_ov_output_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   ثابت std::shared_ptr<ov::InferRequest> & infer_request,
                                   صحيح output_index,
                                   ثابت ggml_tensor * ggml_tensor) {
    إذا (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        أرجع *sliced;
    }

    auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    فراغ * output_data = ggml_tensor->data;
    إذا (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_index).get_shape();
    } وإلا {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        إذا (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != nullptr &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_data = ggml_tensor->view_src->data;
        } وإلا {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    إذا (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_buffer_is_remote(ggml_tensor) &&
        ggml_tensor->extra != nullptr) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        إذا (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            إذا (tensor_extra->tensor != nullptr && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                أرجع *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_data);
    أرجع output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_rows_to_seq_axis_2(ثابت ov::Tensor & kv_tensor, size_t n_heads_kv) {
    ثابت size_t rows = kv_tensor.get_shape()[2];
    ثابت size_t head_size = kv_tensor.get_shape()[3] / n_heads_kv;
    ثابت size_t elem = kv_tensor.get_element_type().size();
    ثابت size_t head_bytes = head_size * elem;

    ov::Tensor out(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, rows, head_size});
    ثابت auto * src = static_cast<ثابت uint8_t *>(kv_tensor.data());
    auto * dst = static_cast<uint8_t *>(out.data());
    لكل (size_t s = 0; s < rows; s++) {
        لكل (size_t h = 0; h < n_heads_kv; h++) {
            memcpy(dst + (h * rows + s) * head_bytes, src + (s * n_heads_kv + h) * head_bytes, head_bytes);
        }
    }
    أرجع out;
}

template <typename T> فراغ set_zero_diagonal(std::vector<T> & matrix, size_t rows, size_t cols, T zero_قدر = T{}) {
    لكل (size_t i = 0; i < rows; ++i) {
        size_t diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = zero_قدر;
    }
}

ov::Tensor make_contiguous_split_داخلہ_tensor(ثابت بنية ggml_tensor * ggml_tensor, ثابت ov::Shape & داخلہ_shape) {
    ثابت size_t element_size = ggml_type_size(ggml_tensor->type);
    ثابت size_t block_size = ggml_blck_size(ggml_tensor->type);

    GGML_ASSERT(block_size == 1 && "non-contiguous split inputs must be plain element types");

    ثابت بنية ggml_tensor * source_tensor = ggml_tensor->view_src != nullptr ? ggml_tensor->view_src : ggml_tensor;
    ثابت size_t source_offset = ggml_tensor->view_src != nullptr ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> source_data(ggml_nbytes(source_tensor));
    ggml_backend_tensor_get(source_tensor, source_data.data(), 0, source_data.size());

    ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
    auto * dst = static_cast<uint8_t *>(داخلہ_tensor.data());
    size_t dst_offset = 0;

    لكل (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        لكل (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            لكل (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                لكل (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    ثابت size_t src_offset = source_offset + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::memcpy(dst + dst_offset, source_data.data() + src_offset, element_size);
                    dst_offset += element_size;
                }
            }
        }
    }

    أرجع داخلہ_tensor;
}

ov::Tensor convert_ggml_داخلہ_to_ov(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & name) {
    ثابت auto * ggml_tensor = ggml_decoder->get_داخلہ_ggml_tensor(name);

    إذا (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        أرجع *sliced;
    }

    إذا (ggml_tensor->extra != nullptr && !ggml_decoder->is_splited_model()) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        إذا (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            أرجع *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    auto * داخلہ_data = ggml_tensor->data;
    ov::Shape داخلہ_shape;
    إذا (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        داخلہ_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } وإلا {
        داخلہ_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    إذا (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        أرجع make_contiguous_split_داخلہ_tensor(ggml_tensor, داخلہ_shape);
    }

    auto داخلہ_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape, داخلہ_data);
    أرجع داخلہ_tensor;
}

ov::Tensor get_ov_داخلہ_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & param_name) {
    ov::Tensor داخلہ_tensor;
    auto extra_داخلہ = ggml_decoder->get_model_extra_inputs().find(param_name);
    إذا (extra_داخلہ != ggml_decoder->get_model_extra_inputs().end()) {
        داخلہ_tensor = ov::Tensor(extra_داخلہ->second.type, extra_داخلہ->second.shape);
        *داخلہ_tensor.data<int64_t>() = extra_داخلہ->second.value;
    } وإلا {
        داخلہ_tensor = convert_ggml_داخلہ_to_ov(ggml_decoder, param_name);
    }
    أرجع داخلہ_tensor;
}

ov::Tensor get_ov_داخلہ_tensor_static_decode(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             ثابت std::string & param_name) {
    // NPU decoding stage
    إذا (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        أرجع get_ov_داخلہ_tensor(ggml_decoder, param_name);
    }
    ثابت auto * ggml_tensor = ggml_decoder->get_داخلہ_ggml_tensor(param_name);
    ثابت auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    إذا (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        ثابت صحيح n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        أكد(ggml_tensor->ne[0] == n_planes);
        ov::Shape داخلہ_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        std::memcpy(داخلہ_tensor.data(), ggml_tensor->data, n_planes * ggml_type_size(ggml_tensor->type));
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape داخلہ_shape = {1, 1, 1, 1};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        أكد(ggml_tensor->ne[0] == 1);
        أكد(inp_out_id == 0);
        *داخلہ_tensor.data<int32_t>() = inp_out_id;
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t context_size = ggml_decoder->get_ctx_size();
        إذا (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_data =
                pad_داخلہ<ggml_fp16_t>(ggml_tensor, 1, context_size, GGML_FP32_TO_FP16(-INFINITY));
            ov::Tensor داخلہ_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_size});
            std::memcpy(داخلہ_tensor.data(), padded_data.data(), padded_data.size() * حجم(ggml_fp16_t));
            أرجع داخلہ_tensor;
        }

        std::vector<عائم> padded_data = pad_داخلہ<عائم>(ggml_tensor, 1, context_size, -INFINITY);
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        auto * data_ptr = داخلہ_tensor.data<عائم>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
        أرجع داخلہ_tensor;
    }

    أرجع get_ov_داخلہ_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_داخلہ_tensor_static_prefill(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              ثابت std::string & param_name,
                                              صحيح chunk_index) {
    // NPU prompt processing stage
    ثابت size_t داخلہ_len = ggml_decoder->get_داخلہ_len();
    ثابت size_t chunk_size = ggml_decoder->m_prefill_chunk_size;
    ثابت size_t chunk_valid_size = std::min(chunk_size, داخلہ_len - chunk_index * chunk_size);
    ثابت size_t chunk_pad_size = chunk_size - chunk_valid_size;

    إذا (param_name == "chunk_valid_len") {
        ov::Tensor داخلہ_tensor(ov::element::i64, ov::Shape{1});
        *داخلہ_tensor.data<int64_t>() = (int64_t) chunk_valid_size;
        أرجع داخلہ_tensor;
    }
    إذا (chunk_index > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor داخلہ_tensor(ov::element::i64, ov::Shape{1});
        *داخلہ_tensor.data<int64_t>() = 0;
        أرجع داخلہ_tensor;
    }
    إذا (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        أرجع get_ov_داخلہ_tensor(ggml_decoder, param_name);
    }
    ثابت auto * ggml_tensor = ggml_decoder->get_داخلہ_ggml_tensor(param_name);
    ثابت auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    إذا (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        ثابت صحيح n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        ثابت size_t element_size = ggml_type_size(ggml_tensor->type);
        ov::Shape داخلہ_shape = {1, 1, 1, (size_t) n_planes * chunk_size};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        لكل (صحيح p = 0; p < n_planes; p++) {
            ثابت حرف * src =
                (ثابت حرف *) ggml_tensor->data + (p * داخلہ_len + chunk_index * chunk_size) * element_size;
            حرف * dst = (حرف *) داخلہ_tensor.data() + p * chunk_size * element_size;
            std::memcpy(dst, src, chunk_valid_size * element_size);
            إذا (chunk_pad_size > 0) {
                إذا (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t last_قدر = *((ثابت int32_t *) src + chunk_valid_size - 1);
                    int32_t * out = (int32_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_قدر + 1);
                } وإلا إذا (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t last_قدر = *((ثابت int64_t *) src + chunk_valid_size - 1);
                    int64_t * out = (int64_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_قدر + 1);
                } وإلا {
                    ارم std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape داخلہ_shape = {1, 1, 1, chunk_size};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        فراغ * داخلہ_data = (حرف *) ggml_tensor->data + chunk_index * chunk_size * element_size;
        std::memcpy(داخلہ_tensor.data(), داخلہ_data, chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        إذا (chunk_pad_size > 0) {
            إذا (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_قدر =
                    *((int32_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int32_t * output_data = داخلہ_tensor.data<int32_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_قدر + 1);
            } وإلا إذا (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_قدر =
                    *((int64_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int64_t * output_data = داخلہ_tensor.data<int64_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_قدر + 1);
            } وإلا {
                ارم std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape داخلہ_shape = {1, 1, 1, output_len};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        إذا (ggml_tensor->ne[0] == 0) {
            *داخلہ_tensor.data<int32_t>() = 0;
        } وإلا {
            auto * data_addr = داخلہ_tensor.data<int32_t>();
            لكل (size_t i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_scale_rows(ggml_tensor, op)) {
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, 1});
        auto * dst = داخلہ_tensor.data<عائم>();
        ثابت auto * src = static_cast<ثابت عائم *>(ggml_tensor->data) + chunk_index * chunk_size;
        std::copy(src, src + chunk_valid_size, dst);
        std::fill(dst + chunk_valid_size, dst + chunk_size, 1.0f);
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_mean(ggml_tensor, op)) {
        ثابت size_t n_seqs = ggml_tensor->ne[1];
        ثابت size_t src_stride = ggml_tensor->ne[0];
        ثابت size_t copy_len = std::min<size_t>(chunk_valid_size, src_stride - chunk_index * chunk_size);
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_size});
        auto * dst = داخلہ_tensor.data<عائم>();
        std::fill(dst, dst + n_seqs * chunk_size, 0.0f);
        ثابت auto * src = static_cast<ثابت عائم *>(ggml_tensor->data) + chunk_index * chunk_size;
        لكل (size_t s = 0; s < n_seqs; s++) {
            std::memcpy(dst + s * chunk_size, src + s * src_stride, copy_len * حجم(عائم));
        }
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = ggml_decoder->get_ctx_size();
        إذا (ggml_tensor->type == GGML_TYPE_F16) {
            ثابت auto * ggml_data =
                static_cast<ثابت ggml_fp16_t *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
            std::vector<ggml_fp16_t> padded_data = pad_داخلہ<ggml_fp16_t>(ggml_data, chunk_valid_rows, cols, chunk_size,
                                                                          context_size, GGML_FP32_TO_FP16(-INFINITY));
            set_zero_diagonal(padded_data, chunk_size, context_size, GGML_FP32_TO_FP16(0.0f));
            ov::Tensor داخلہ_tensor(ov::element::f16, ov::Shape{1, 1, chunk_size, context_size});
            std::memcpy(داخلہ_tensor.data(), padded_data.data(), padded_data.size() * حجم(ggml_fp16_t));
            أرجع داخلہ_tensor;
        }

        ثابت auto * ggml_data = static_cast<ثابت عائم *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
        std::vector<عائم> padded_data =
            pad_داخلہ<عائم>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        auto * data_ptr = داخلہ_tensor.data<عائم>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        أرجع داخلہ_tensor;
    }

    أرجع get_ov_داخلہ_tensor(ggml_decoder, param_name);
}

تعداد ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               ثابت std::string & device,
                               ثابت ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    إذا (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        أرجع GGML_STATUS_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_lock(cache.mutex);
    منطقي naive = true;
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    auto داخلہ_model = std::make_shared<ov::frontend::ggml::داخلہ_Model>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(داخلہ_model, naive);
    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    إذا (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    } وإلا {
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    }
    إذا (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } وإلا {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> داخلہ_names;
    std::vector<std::string> output_names;
    لكل (ثابت auto & param : model->get_parameters()) {
        داخلہ_names.push_back(param->get_friendly_name());
    }
    لكل (ثابت auto & result : model->get_results()) {
        output_names.push_back(result->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    داخلہ_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_lock.unlock();

    لكل (size_t i = 0; i < داخلہ_names.size(); i++) {
        ثابت auto & param_name = داخلہ_names[i];
        auto داخلہ_tensor = get_ov_داخلہ_tensor(decoder, param_name);
        infer_request->set_داخلہ_tensor(i, داخلہ_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    لكل (size_t i = 0; i < output_names.size(); i++) {
        auto output_tensor = infer_request->get_output_tensor(i);
        ثابت auto & model_outputs = decoder->get_model_outputs();
        auto model_output_it = model_outputs.find(output_names[i]);
        إذا (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_names[i], output_tensor, output_tensor.data());
            }
            استمر;
        }
        auto * ggml_tensor = model_output_it->second;
        std::memcpy(ggml_tensor->data, output_tensor.data(), output_tensor.get_byte_size());
    }
    أرجع GGML_STATUS_SUCCESS;
}

تعداد ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();
    ثابت auto & config = ggml_openvino_get_compile_config();
    ثابت auto & device = r_ctx->device;
    ساكن auto is_static = خطأ;

    ساكن ثابت منطقي cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    auto start_time = ggml_time_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    ثابت int64_t cache_کلید_start_time = ggml_time_us();
    graph_کلید کلید(cgraph);
    ثابت int64_t cache_کلید_compute_time = ggml_time_us() - cache_کلید_start_time;
    int64_t cache_lookup_time = 0;
    منطقي کلید_seen = خطأ;
    إذا (!cache_disabled) {
        ثابت int64_t cache_lookup_start_time = ggml_time_us();
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        کلید_seen = r_ctx->decoder_cache.find(کلید) != r_ctx->decoder_cache.end();
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;
    }

    ثابت int64_t model_split_check_start_time = ggml_time_us();
    منطقي model_is_splitted = کلید_seen ? خطأ : is_model_splitted(cgraph);
    ثابت int64_t model_split_check_time = ggml_time_us() - model_split_check_start_time;

    إذا (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        أرجع GGML_STATUS_FAILED;
    }
    إذا (is_naive(cgraph)) {
        إذا (!model_is_splitted) {
            أرجع naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    ثابت int64_t graph_param_compute_start_time = ggml_time_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    ثابت int64_t graph_param_compute_time = ggml_time_us() - graph_param_compute_start_time;

    ثابت منطقي cache_enabled = !model_is_splitted && !cache_disabled;
    ثابت منطقي stateful_recurrent = r_ctx->stateful && m_params.state_size >= 0;
    ثابت منطقي stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    إذا (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_active > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        أرجع GGML_STATUS_FAILED;
    }
    إذا (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        أرجع GGML_STATUS_FAILED;
    }
    منطقي cache_hit = خطأ;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        ثابت int64_t cache_lookup_start_time = ggml_time_us();
        إذا (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            auto it = r_ctx->decoder_cache.find(کلید);
            cache_hit = it != r_ctx->decoder_cache.end();
            إذا (cache_hit) {
                entry = it->second;
            } وإلا {
                r_ctx->clear_caches_locked();
                auto mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[کلید] = entry;
            }
        } وإلا {
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_hit = خطأ;
        }
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;

        std::lock_guard<std::mutex> lock(*(entry->mutex));
        cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(کلید) != 0;

        إذا (cache_hit) {
            ggml_decoder = entry->ptr;
            old_m_params = ggml_decoder->get_model_params();
            إذا (!ggml_decoder->is_splited_model()) {
                cache_hit = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_داخلہ_names;
        std::vector<std::string> ov_output_names;

        إذا (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            إذا (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(کلید);
                ov_داخلہ_names = r_ctx->ov_داخلہ_names_cache.at(کلید);
                ov_output_names = r_ctx->ov_output_names_cache.at(کلید);
            }

            إذا (stateful_kv_only) {
                ثابت auto * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_data = (int32_t *) inp_pos->data;
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                إذا (pos_data[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_size = pos_shape[3];
                } وإلا إذا (r_ctx->stateful_kv_size == static_cast<size_t>(pos_data[0])) {
                    r_ctx->stateful_kv_size += pos_shape[3];
                } وإلا {
                    ثابت size_t pos_begin = static_cast<size_t>(pos_data[0]);
                    ثابت منطقي refill = pos_begin > r_ctx->stateful_kv_size;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    إذا (refill && !ggml_decoder->get_model_params().swa_layers.empty()) {
                        ثابت صحيح n_swa = ggml_decoder->get_compute_params().swa_window;
                        إذا (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_size, n_swa);
                            أرجع GGML_STATUS_FAILED;
                        }
                    }

                    ثابت منطقي relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    auto states = infer_request->query_state();
                    لكل (auto state : states) {
                        auto state_tensor = state.get_state();
                        auto state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        إذا (auto it = r_ctx->kv_state_داخلہ_name_map.find(state.get_name());
                            it != r_ctx->kv_state_داخلہ_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        صحيح n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        إذا (auto layer = extract_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        ثابت منطقي relayout_this_state = relayout_enabled;
                        ثابت size_t seq_axis = relayout_this_state ? 2 : 1;
                        ثابت size_t head_axis = seq_axis == 2 ? 1 : 2;

                        إذا (refill) {
                            إذا (state_name.empty()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                أرجع GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_داخلہ_tensor(ggml_decoder, state_name);
                            إذا (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_rows_to_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } وإلا {
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
                        إذا (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            أرجع GGML_STATUS_FAILED;
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
        } وإلا {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            auto shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
            auto weight_names = get_weight_names(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ثابت std::string model_cache_dir = ggml_openvino_model_cache_dir();
            ثابت std::string exact_کلید = cache_enabled ? compiled_graph_کلید(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_path, manifest_path;
            إذا (!model_cache_dir.empty() && !model_is_splitted) {
                ثابت uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/true, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_graph_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/true, m_params.rope_params, 16, extra_cfg,
                                                    compiled_graph_کلید(cgraph, *ggml_decoder, device, 0, true));
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            std::string shared_کلید =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_کلید) : "";
            ov::CompiledModel shared_model;
            منطقي imported = خطأ;
            auto shared_it = shared_cache->graphs.find(shared_کلید);
            إذا (!shared_کلید.empty() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_graph(shared_it->second.decode, shared_it->second.داخلہ_names,
                                                               shared_it->second.output_names, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_داخلہ_names = shared_it->second.داخلہ_names;
                ov_output_names = shared_it->second.output_names;
                imported = true;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } وإلا إذا (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_کلید = exact_کلید;
                shared_it = shared_cache->graphs.find(shared_کلید);
                إذا (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_داخلہ_names = shared_it->second.داخلہ_names;
                    ov_output_names = shared_it->second.output_names;
                    imported = true;
                }
                model_fp = exact_fp;
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(کلید);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            إذا (!model_cache_dir.empty()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
            }
            إذا (!imported && !model_cache_dir.empty() && !model_is_splitted) {
                ثابت uint64_t preferred_fp = model_fp;
                منطقي dynamic_incompatible = preferred_fp == exact_fp;
                لكل (صحيح candidate = 0; candidate < 2; ++candidate) {
                    إذا (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        اكسر;
                    }
                    ثابت uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    ثابت std::string candidate_blob =
                        ggml_openvino_model_cache_blob_path(model_cache_dir, candidate_fp);
                    ثابت std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_path(model_cache_dir, candidate_fp);
                    std::ifstream blob_in(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    إذا (!blob_in.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        استمر;
                    }
                    ثابت int64_t import_start = ggml_time_us();
                    حاول {
                        ov::CompiledModel cm;
                        auto remote_context = ggml_openvino_get_remote_context();
                        إذا (remote_context.has_value()) {
                            cm = core.import_model(blob_in, remote_context.value(), mc_config);
                        } وإلا {
                            cm = core.import_model(blob_in, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        إذا (!saved_inputs.empty() || !saved_outputs.empty()) {
                            ov_داخلہ_names = std::move(saved_inputs);
                            ov_output_names = std::move(saved_outputs);
                        } وإلا {
                            لكل (ثابت auto & p : cm.inputs()) {
                                ov_داخلہ_names.push_back(p.get_گرہ()->get_friendly_name());
                            }
                            لكل (ثابت auto & o : cm.outputs()) {
                                ov_output_names.push_back(o.get_گرہ()->get_friendly_name());
                            }
                        }
                        إذا (!compiled_model_matches_graph(cm, ov_داخلہ_names, ov_output_names, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_داخلہ_names.clear();
                            ov_output_names.clear();
                            استمر;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        entry->ptr = ggml_decoder;
                        imported = true;
                        model_fp = candidate_fp;
                        blob_path = candidate_blob;
                        manifest_path = candidate_manifest;
                        إذا (candidate_fp == exact_fp) {
                            shared_کلید = exact_کلید;
                            shared_it = shared_cache->graphs.find(shared_کلید);
                        }
                        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_time_us() - import_start) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } امسك (ثابت std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = خطأ;
                        ov_داخلہ_names.clear();
                        ov_output_names.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                إذا (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_کلید = exact_کلید;
                    shared_it = shared_cache->graphs.find(shared_کلید);
                    blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                    manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            إذا (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_path.c_str());
                أرجع GGML_STATUS_FAILED;
            }
            إذا (!imported && ggml_openvino_weight_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                أرجع GGML_STATUS_FAILED;
            }
            إذا (imported) {
                decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            } وإلا {
                auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_end_time = ggml_time_us();

                auto داخلہ_model = std::make_shared<ov::frontend::ggml::داخلہ_Model>(ggml_decoder);
                model = ov::frontend::ggml::FrontEnd::convert(داخلہ_model);
                ggml_decoder->clear_model_weights();
                conversion_end_time = ggml_time_us();

                إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    حرف timestamped_filename[64];
                    auto timestamp = (long long) ggml_time_us();
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                ثابت ov::AnyMap & compile_config = model_cache_dir.empty() ? config : mc_config;
                ov::CompiledModel compiled_model;
                auto remote_context = ggml_openvino_get_remote_context();
                إذا (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } وإلا {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_end_time = ggml_time_us();

                لكل (ثابت auto & ov_param : model->get_parameters()) {
                    ov_داخلہ_names.push_back(ov_param->get_friendly_name());
                }
                لكل (ثابت auto & ov_output : model->get_results()) {
                    ov_output_names.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                إذا (!model_cache_dir.empty() && !model_is_splitted && model_fp != 0) {
                    حاول {
                        ثابت std::string blob_tmp = ggml_openvino_model_cache_temp_path(blob_path);
                        ثابت std::string manifest_tmp = ggml_openvino_model_cache_temp_path(manifest_path);
                        إذا (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_داخلہ_names,
                                                                     ov_output_names)) {
                            std::ofstream blob_out(blob_tmp, std::ios::binary | std::ios::trunc);
                            إذا (blob_out.is_open()) {
                                compiled_model.export_model(blob_out);
                                blob_out.close();
                                إذا (blob_out.good()) {
                                    إذا (std::rename(blob_tmp.c_str(), blob_path.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_path.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_path.c_str());
                                    } وإلا {
                                        std::remove(blob_tmp.c_str());
                                        std::remove(manifest_tmp.c_str());
                                    }
                                } وإلا {
                                    std::remove(blob_tmp.c_str());
                                    std::remove(manifest_tmp.c_str());
                                }
                            } وإلا {
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    } امسك (ثابت std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                entry->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            entry->ptr = ggml_decoder;
            إذا (!shared_کلید.empty() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_کلید,
                                             ov_compiled_graph{shared_model, {}, ov_داخلہ_names, ov_output_names});
            }
            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[کلید] = infer_request;
                r_ctx->ov_داخلہ_names_cache[کلید] = ov_داخلہ_names;
                r_ctx->ov_output_names_cache[کلید] = ov_output_names;
            }

            إذا (stateful_kv_only && cache_enabled) {
                ثابت auto * inp_pos = get_inp_pos_tensor(cgraph);
                auto pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                ثابت int32_t pos_begin = ((int32_t *) inp_pos->data)[0];
                إذا (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    أرجع GGML_STATUS_FAILED;
                }
                r_ctx->stateful_kv_size = pos_shape[3];
                ثابت auto kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                لكل (ثابت auto & pair : kv_param_res_names) {
                    r_ctx->kv_state_داخلہ_name_map[pair.first + pair.second] = pair.first;
                }
            }
        }

        إذا (stateful_recurrent) {
            ثابت auto * inp_pos = get_inp_pos_tensor(cgraph);
            ثابت int32_t pos_begin = static_cast<ثابت int32_t *>(inp_pos->data)[0];
            إذا (pos_begin == 0) {
                infer_request->reset_state();
            } وإلا إذا (!cache_hit || old_m_params.kv_buffer_changed(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_size) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                أرجع GGML_STATUS_FAILED;
            }
        } وإلا {
            reset_single_slot_recurrent_cache(cgraph, m_params, c_params);
        }

        لكل (size_t i = 0; i < ov_داخلہ_names.size(); i++) {
            ثابت auto & param_name = ov_داخلہ_names[i];
            auto داخلہ_tensor = get_ov_داخلہ_tensor(ggml_decoder, param_name);
            infer_request->set_داخلہ_tensor(i, داخلہ_tensor);

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_داخلہ_tensor_info(param_name, داخلہ_tensor);
            }
        }

        لكل (size_t i = 0; i < ov_output_names.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            ثابت auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names[i]);
            إذا (model_output_it == model_outputs.end()) {
                استمر;
            }
            auto * ggml_tensor = model_output_it->second;
            إذا (ggml_nbytes(ggml_tensor) == 0) {
                استمر;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        إذا (stateful_recurrent) {
            ثابت auto * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_size =
                static_cast<ثابت int32_t *>(inp_pos->data)[0] + get_inp_pos_n_tokens(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            لكل (size_t i = 0; i < ov_output_names.size(); i++) {
                ثابت auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_کلید_compute_time / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_split_check_time / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", graph_param_compute_time / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_time / 1000.0);
            إذا (!cache_hit) {
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
    إذا (cache_hit && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_lock(r_ctx->compiled_cache->mutex);
        إذا (!ggml_openvino_weight_buffers_released()) {
            ggml_openvino_release_weight_buffers();
        }
    }

    أرجع GGML_STATUS_SUCCESS;
}

ov::AnyMap without_npuw(ثابت ov::AnyMap & config) {
    ov::AnyMap out;
    لكل (ثابت auto & kv : config) {
        إذا (kv.first.rfind("NPUW", 0) == 0 || kv.first == "NPU_USE_NPUW") {
            استمر;
        }
        out.insert(kv);
    }
    أرجع out;
}

تعداد ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    auto & core = ov_singleton_core();

    auto get_prefill_chunk_size = [] {
        ساكن ثابت صحيح chunk_size = []() {
            صحيح env_prefill_chunk_size = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            أرجع env_prefill_chunk_size > 0 ? env_prefill_chunk_size : 256;
        }();
        أرجع chunk_size;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    ساكن std::string device = ggml_openvino_get_device_name();
    ساكن auto is_static = true;
    ساكن auto stateful = خطأ;

    auto prefill_chunk_size = get_prefill_chunk_size();
    ثابت auto & config = ggml_openvino_get_compile_config();

    إذا (is_naive(cgraph)) {
        أرجع naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    ثابت auto * inp_pos = get_inp_pos_tensor(cgraph);
    ثابت منطقي no_kv_cache = m_params.is_cacheless_attn;
    ثابت auto is_prefill = no_kv_cache ? true : get_is_prefill(cgraph, inp_pos);
    ثابت ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    إذا (m_params.n_heads_kv == -1) {
        prefill_chunk_size = inp_pos->ne[0];
    }
    graph_کلید کلید(cgraph);
    ساكن ثابت منطقي cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    منطقي cache_hit = خطأ;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;
    int64_t ov_raw_infer_total = 0;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    إذا (cache_enabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        auto it = r_ctx->decoder_cache.find(کلید);
        cache_hit = it != r_ctx->decoder_cache.end();
        إذا (cache_hit) {
            entry = it->second;
        } وإلا {
            r_ctx->clear_caches_locked();
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[کلید] = entry;
        }
    } وإلا {
        auto mutex = std::make_shared<std::mutex>();
        entry = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_hit = خطأ;
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));
    cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(کلید) != 0 &&
                r_ctx->infer_request_cache_prefill.count(کلید) != 0;

    إذا (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_داخلہ_names_local;
    std::vector<std::string> ov_output_names_local;

    إذا (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        إذا (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(کلید) : r_ctx->infer_request_cache.at(کلید);
            ov_داخلہ_names_local = r_ctx->ov_داخلہ_names_cache.at(کلید);
            ov_output_names_local = r_ctx->ov_output_names_cache.at(کلید);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } وإلا {
        إذا (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(کلید);
            r_ctx->infer_request_cache_prefill.erase(کلید);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        auto shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
        auto weight_names = get_weight_names(cgraph);
        auto local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names, is_static,
                                                             stateful, خطأ, is_prefill, prefill_chunk_size);
        ثابت std::string shared_کلید =
            cache_enabled ? compiled_graph_کلید(cgraph, *local_decoder, device, prefill_chunk_size) : "";
        auto shared_it = shared_cache->graphs.find(shared_کلید);
        إذا (!shared_کلید.empty() && shared_it != shared_cache->graphs.end()) {
            auto & compiled = shared_it->second;
            auto prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            auto decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            entry->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_داخلہ_names_local = compiled.داخلہ_names;
            ov_output_names_local = compiled.output_names;
            r_ctx->infer_request_cache_prefill[کلید] = prefill_request;
            r_ctx->infer_request_cache[کلید] = decode_request;
            r_ctx->ov_داخلہ_names_cache[کلید] = ov_داخلہ_names_local;
            r_ctx->ov_output_names_cache[کلید] = ov_output_names_local;
            decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } وإلا {
            std::shared_ptr<ov::Model> model;
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

            auto ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, خطأ, true, prefill_chunk_size);
            auto ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, خطأ, خطأ, prefill_chunk_size);
            decoder_end_time = ggml_time_us();

            ثابت منطقي dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            ثابت auto dump_ir_timestamp = static_cast<long long>(ggml_time_us());

            auto build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          ثابت std::shared_ptr<GgmlOvDecoder> & decoder, ثابت حرف * tag,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_end_time, int64_t & local_compile_end_time) {
                auto داخلہ_model = std::make_shared<ov::frontend::ggml::داخلہ_Model>(decoder);
                model = ov::frontend::ggml::FrontEnd::convert(داخلہ_model);
                decoder->clear_model_weights();
                local_conversion_end_time = ggml_time_us();

                إذا (dump_ir) {
                    حرف timestamped_filename[64];
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%s_%lld.xml", tag,
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
            إذا (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_end_time = prefill_conversion_end_time;
                decode_compile_end_time = prefill_compile_end_time;
            } وإلا {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_end_time, decode_compile_end_time);
            }
            conversion_end_time = std::max(prefill_conversion_end_time, decode_conversion_end_time);
            compile_end_time = std::max(prefill_compile_end_time, decode_compile_end_time);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            entry->ptr = ggml_decoder;

            لكل (ثابت auto & ov_param : model->get_parameters()) {
                ov_داخلہ_names_local.push_back(ov_param->get_friendly_name());
            }
            لكل (ثابت auto & ov_output : model->get_results()) {
                ov_output_names_local.push_back(ov_output->get_friendly_name());
            }

            إذا (!shared_کلید.empty()) {
                shared_cache->graphs.emplace(
                    shared_کلید, ov_compiled_graph{compiled_model_decode, compiled_model_prefill, ov_داخلہ_names_local,
                                                  ov_output_names_local});
            }

            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[کلید] = infer_request_prefill;
                r_ctx->infer_request_cache[کلید] = infer_request_decode;
                r_ctx->ov_داخلہ_names_cache[کلید] = ov_داخلہ_names_local;
                r_ctx->ov_output_names_cache[کلید] = ov_output_names_local;
            }
        }
    }

    reset_single_slot_recurrent_cache(cgraph, m_params, c_params);

    إذا (is_prefill) {
        auto inp_len = get_inp_pos_n_tokens(cgraph, inp_pos);
        لكل (صحيح chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            لكل (size_t i = 0; i < ov_داخلہ_names_local.size(); i++) {
                ثابت auto & param_name = ov_داخلہ_names_local[i];
                auto داخلہ_tensor = get_ov_داخلہ_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_داخلہ_tensor(i, داخلہ_tensor);

                إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    ثابت auto داخلہ_tensor = infer_request->get_داخلہ_tensor(i);
                    print_داخلہ_tensor_info(param_name, داخلہ_tensor);
                }
            }

            لكل (size_t i = 0; i < ov_output_names_local.size(); i++) {
                ثابت auto & model_outputs = ggml_decoder->get_model_outputs();
                auto model_output_it = model_outputs.find(ov_output_names_local[i]);
                إذا (model_output_it == model_outputs.end()) {
                    استمر;
                }
                auto * ggml_tensor = model_output_it->second;
                إذا (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    استمر;
                }
                auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            ov_raw_infer_start = ggml_time_us();
            infer_request->infer();
            ov_raw_infer_total += ggml_time_us() - ov_raw_infer_start;

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                لكل (size_t i = 0; i < ov_output_names_local.size(); i++) {
                    ثابت auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } وإلا {
        لكل (size_t i = 0; i < ov_داخلہ_names_local.size(); i++) {
            ثابت auto & param_name = ov_داخلہ_names_local[i];
            auto داخلہ_tensor = get_ov_داخلہ_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_داخلہ_tensor(i, داخلہ_tensor);

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                ثابت auto داخلہ_tensor = infer_request->get_داخلہ_tensor(i);
                print_داخلہ_tensor_info(param_name, داخلہ_tensor);
            }
        }

        لكل (size_t i = 0; i < ov_output_names_local.size(); i++) {
            ثابت auto & model_outputs = ggml_decoder->get_model_outputs();
            auto model_output_it = model_outputs.find(ov_output_names_local[i]);
            إذا (model_output_it == model_outputs.end()) {
                استمر;
            }
            auto * ggml_tensor = model_output_it->second;
            إذا (ggml_nbytes(ggml_tensor) == 0) {
                استمر;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        ov_raw_infer_total = infer_end_time - ov_raw_infer_start;

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            لكل (size_t i = 0; i < ov_output_names_local.size(); i++) {
                ثابت auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
            }
        }
    }

    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
        إذا (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_end_time - decoder_end_time) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_total / 1000.0);
    }

    أرجع GGML_STATUS_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
ساكن ggml_cgraph * get_compute_graph(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    auto is_skipped = [](ثابت ggml_tensor * گرہ) {
        أرجع گرہ->op != GGML_OP_NONE && !(گرہ->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    إذا (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        أرجع cgraph;
    }
    auto & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](ثابت ggml_tensor * گرہ) { أرجع !is_skipped(گرہ); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.data();
    compute.graph.n_nodes = (صحيح) compute.nodes.size();
    أرجع &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
تعداد ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    حاول {
        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        ثابت auto is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_lock(r_ctx->execution_mutex);
        cgraph = get_compute_graph(cgraph, *r_ctx);

        أرجع is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } امسك (ثابت ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        أرجع GGML_STATUS_FAILED;
    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        أرجع GGML_STATUS_FAILED;
    } امسك (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        أرجع GGML_STATUS_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
منطقي is_model_splitted(ggml_cgraph * cgraph) {
    ساكن ثابت منطقي fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    إذا (!fallback_enabled) {
        أرجع خطأ;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    إذا (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        أرجع خطأ;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = cgraph->nodes[i];
        صحيح use_گنتی = cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, گرہ)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        إذا ((cgraph->n_nodes <= 1 && use_گنتی == 0) ||
            (cgraph->n_nodes <= 1 && گرہ->op == GGML_OP_VIEW && use_گنتی == 1 && گرہ->src[0] != nullptr &&
             گرہ->src[0]->op == GGML_OP_NONE)) {
            أرجع خطأ;
        }
        إذا (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_PERMUTE)) {
            أرجع خطأ;
        }
        صحيح داخلہ_use_گنتی = 0;
        لكل (صحيح j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_گرہ = cgraph->nodes[j];
            لكل (صحيح k = 0; k < GGML_MAX_SRC; k++) {
                إذا (other_گرہ->src[k] == گرہ) {
                    داخلہ_use_گنتی++;
                }
            }
        }
        إذا (use_گنتی != داخلہ_use_گنتی && گرہ->op != GGML_OP_NONE) {
            أرجع true;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    إذا (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_weight_names(cgraph);
    } وإلا {
        لكل (ثابت auto & kv : GgmlOvDecoder::create_weight_nodes(cgraph, true)) {
            model_weights.insert(kv.first);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = گرہ->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            إذا (src != nullptr && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.empty() == خطأ &&
                model_leafs.find(src) == model_leafs.end()) {
                إذا (GgmlOvDecoder::is_inp_tok(src, گرہ)) {
                    أرجع خطأ;
                }
                أرجع true;
            }
        }
    }
    أرجع خطأ;
}

منطقي is_naive(ggml_cgraph * cgraph) {
    constexpr صحيح naive_graph_size_threshold = 20;
    صحيح count = 0;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        إذا (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    أرجع count < naive_graph_size_threshold;
}

size_t checksum(ثابت فراغ * data, size_t size) {
    ثابت uint8_t * bytes = static_cast<ثابت uint8_t *>(data);
    size_t sum = 0;
    لكل (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    أرجع sum;
}

منطقي save_ggml_tensor_data_to_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_path) {
    إذا (tensor == nullptr || tensor->data == nullptr) {
        أرجع خطأ;
    }

    std::ofstream out(file_path);
    إذا (!out.is_open()) {
        أرجع خطأ;
    }

    ثابت size_t n = ggml_nelements(tensor);
    out << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    بدل (tensor->type) {
    حالة GGML_TYPE_F32: {
        ثابت auto * data = static_cast<ثابت عائم *>(tensor->data);
        لكل (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_F16: {
        ثابت auto * data = static_cast<ثابت ggml_fp16_t *>(tensor->data);
        لكل (size_t i = 0; i < n; ++i) {
            out << ggml_fp16_to_fp32(data[i]) << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_BF16: {
        ثابت auto * data = static_cast<ثابت ggml_bf16_t *>(tensor->data);
        لكل (size_t i = 0; i < n; ++i) {
            out << ggml_bf16_to_fp32(data[i]) << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_I32: {
        ثابت auto * data = static_cast<ثابت int32_t *>(tensor->data);
        لكل (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_I64: {
        ثابت auto * data = static_cast<ثابت int64_t *>(tensor->data);
        لكل (size_t i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        اكسر;
    }
    افتراضي:
        out << "unsupported tensor type for text dump" << '\n';
        أرجع خطأ;
    }

    أرجع true;
}

فراغ print_داخلہ_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << '\n';
    بدل (tensor.get_element_type()) {
    حالة ov::element::f32: {
        إذا (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::cout << *(tensor.data<عائم>()) << '\n';
        } وإلا {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            ثابت عائم * data = tensor.data<عائم>();
            لكل (size_t i = 0; i < rows; ++i) {
                لكل (size_t j = 0; j < cols; ++j) {
                    عائم val = data[i * cols + j];
                    إذا (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } وإلا {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << '\n';
            }
        }

        اكسر;
    }
    حالة ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << '\n';
        اكسر;
    حالة ov::element::i32:
        لكل (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << ' ';
        }
        std::cout << '\n';
        اكسر;
    حالة ov::element::i64:
        لكل (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << ' ';
        }
        std::cout << '\n';
        اكسر;
    افتراضي:
        اكسر;
    }
}

فراغ print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت فراغ * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    auto print_float_stats = [](ثابت std::string & type_name, size_t size, auto get_قدر) {
        إذا (size == 0) {
            أرجع;
        }

        عائم first = get_قدر(0);
        عائم min = first;
        عائم max = first;
        مزدوج sum = first;

        لكل (size_t i = 1; i < size; ++i) {
            عائم v = get_قدر(i);
            min = std::min(v, min);
            max = std::max(v, max);
            sum += v;
        }
        مزدوج mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << '\n';
    };

    بدل (tensor.get_element_type()) {
    حالة ov::element::f32: {
        ثابت عائم * data = tensor.data<عائم>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { أرجع data[i]; });
        اكسر;
    }
    حالة ov::element::f16: {
        ثابت ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { أرجع static_cast<عائم>(data[i]); });
        اكسر;
    }
    افتراضي:
        اكسر;
    }
}

ثابت ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            إذا (src == nullptr) {
                اكسر;
            }
            إذا (GgmlOvDecoder::is_inp_pos(src, op)) {
                أرجع src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    ارم std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_tokens(ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    صحيح n_planes = 1;
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; ++j) {
            إذا (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                اكسر;
            }
        }
    }
    أرجع inp_pos->ne[0] / n_planes;
}

منطقي get_is_prefill(ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos) {
    أرجع get_inp_pos_n_tokens(cgraph, inp_pos) > 1;
}
