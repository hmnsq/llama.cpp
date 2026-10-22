#अन्तर्भाव "utils.h"

#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-openvino-extra.h"
#अन्तर्भाव "ggml-openvino.h"
#अन्तर्भाव "ggml-openvino/ggml-decoder.h"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "model-cache.h"
#अन्तर्भाव "openvino/frontend.h"
#अन्तर्भाव "openvino/input_model.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cassert>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>
#अन्तर्भाव <fstream>
#अन्तर्भाव <functional>
#अन्तर्भाव <iomanip>
#अन्तर्भाव <iostream>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/any.hpp>
#अन्तर्भाव <openvino/core/graph_util.hpp>
#अन्तर्भाव <openvino/core/shape.hpp>
#अन्तर्भाव <openvino/core/type/float16.hpp>
#अन्तर्भाव <openvino/frontend/manager.hpp>
#अन्तर्भाव <openvino/openvino.hpp>
#अन्तर्भाव <openvino/runtime/compiled_model.hpp>
#अन्तर्भाव <openvino/runtime/infer_request.hpp>
#अन्तर्भाव <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#अन्तर्भाव <openvino/runtime/intel_npu/properties.hpp>
#अन्तर्भाव <openvino/runtime/properties.hpp>
#अन्तर्भाव <openvino/runtime/tensor.hpp>
#अन्तर्भाव <optional>
#अन्तर्भाव <set>
#अन्तर्भाव <string>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <vector>

नामाकाश {
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
std::optional<ov::Tensor> try_make_kv_sliced_tensor(स्थिरांक std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    स्थिरांक std::string & name,
                                                    स्थिरांक ggml_tensor * ggml_tensor) {
    स्थिर स्थिरांक तार्किक kv_slice_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    जर (kv_slice_disabled) {
        परत std::nullopt;
    }
    जर (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        परत std::nullopt;
    }
    जर (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != nullptr) {
        परत std::nullopt;
    }
    स्थिरांक स्वतः * op = ggml_decoder->get_tensor_used_op(ggml_tensor);
    जर (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        परत std::nullopt;
    }

    स्थिरांक स्वतः & compute_params = ggml_decoder->get_compute_params();
    जर (compute_params.n_seq_active != 1 || compute_params.seq_active_start != 0) {
        परत std::nullopt;
    }

    पूर्णांक layer;
    जर (स्वतः layer_opt = extract_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } अन्यथा {
        परत std::nullopt;
    }

    स्थिरांक तार्किक is_swa = ggml_decoder->is_swa_layer(layer);
    जर (is_swa) {
        परत std::nullopt;
    }
    स्थिरांक पूर्णांक ctx_per_seq = ggml_decoder->get_ctx_per_seq();
    स्थिरांक पूर्णांक n_kv = compute_params.attention_size;
    जर (ctx_per_seq <= 0 || n_kv <= 0 || n_kv >= ctx_per_seq) {
        परत std::nullopt;
    }

    ov::Shape full_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    जर (full_shape.size() != 4 || full_shape[0] != 1 || full_shape[1] != 1 ||
        static_cast<पूर्णांक>(full_shape[2]) != ctx_per_seq) {
        परत std::nullopt;
    }

    ov::Shape sliced_shape = full_shape;
    sliced_shape[2] = static_cast<माप_प्रकार>(n_kv);

    जर (ggml_openvino_buffer_is_remote(ggml_tensor)) {
        स्वतः remote_context = ggml_openvino_get_remote_context();
        स्वतः gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        परत gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
    }

    परत ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->data);
}

स्थिर uint64_t ggml_openvino_model_cache_extra_cfg(तार्किक stateful, तार्किक recurrent) {
    स्थिरांक अक्षर * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    स्थिरांक तार्किक manual_gqa_enabled = manual_gqa_env != nullptr ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    परत extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_weight_names(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> names;
    साठी (स्थिरांक स्वतः & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
        names[name] = nullptr;
    }
    परत names;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
स्थिर std::string compiled_graph_किल्ली(स्थिरांक ggml_cgraph * graph,
                                      स्थिरांक GgmlOvDecoder & decoder,
                                      स्थिरांक std::string & device,
                                      पूर्णांक prefill_chunk_size = 0,
                                      तार्किक disk_cache = असत्य) {
    std::string किल्ली;
    स्वतः append = [&किल्ली](स्थिरांक स्वतः & value) {
        किल्ली.append(reinterpret_cast<स्थिरांक अक्षर *>(&value), आकार(value));
    };
    स्वतः append_string = [&](स्थिरांक std::string & value) {
        append(value.size());
        किल्ली.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_size);
    तार्किक has_weight_buffer_id = असत्य;
    std::unordered_map<स्थिरांक ggml_tensor *, माप_प्रकार> ids;
    std::function<शून्य(स्थिरांक ggml_tensor *)> visit = [&](स्थिरांक ggml_tensor * tensor) {
        जर (!tensor) {
            append(माप_प्रकार(0));
            परत;
        }
        स्वतः inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.first->second);
        जर (!inserted.second) {
            परत;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        स्थिरांक स्वतः * base = tensor->view_src ? tensor->view_src : tensor;
        स्थिरांक तार्किक weight = base->buffer && base->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        append(weight);
        जर (weight && !disk_cache) {
            स्थिरांक माप_प्रकार buffer_id = ggml_backend_openvino_buffer_get_ctx_id(base->buffer);
            has_weight_buffer_id |= buffer_id != 0;
            append(buffer_id);
            append(tensor->data);
        }
        visit(tensor->view_src);
        साठी (स्थिरांक स्वतः * src : tensor->src) {
            visit(src);
        }
    };
    append(graph->n_nodes);
    साठी (पूर्णांक i = 0; i < graph->n_nodes; ++i) {
        visit(graph->nodes[i]);
    }
    append(graph->n_leafs);
    साठी (पूर्णांक i = 0; i < graph->n_leafs; ++i) {
        visit(graph->leafs[i]);
    }
    साठी (स्थिरांक स्वतः & आदान : decoder.get_model_extra_inputs()) {
        append_string(आदान.first);
        append_string(आदान.second.type.get_type_name());
        append(आदान.second.shape.size());
        साठी (स्वतः dim : आदान.second.shape) {
            append(dim);
        }
        append(आदान.second.is_parameter);
        जर (!आदान.second.is_parameter) {
            append(आदान.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    परत disk_cache || has_weight_buffer_id ? किल्ली : std::string{};
}

स्थिर std::string dynamic_graph_signature(स्थिरांक ggml_cgraph * graph,
                                           स्थिरांक GgmlOvDecoder & decoder,
                                           स्थिरांक ModelParams & params) {
    std::string किल्ली = "dynamic-1";
    स्वतः append = [&किल्ली](स्थिरांक स्वतः & value) {
        किल्ली.append(reinterpret_cast<स्थिरांक अक्षर *>(&value), आकार(value));
    };
    स्वतः append_string = [&](स्थिरांक std::string & value) {
        append(value.size());
        किल्ली.append(value);
    };
    graph_किल्ली graph_id(graph, सत्य);
    append(graph_id.n_nodes);
    append(graph_id.n_leaves);
    append_string(graph_id.first_गाठ_name);
    append_string(graph_id.last_गाठ_name);
    साठी (स्थिरांक स्वतः & name : graph_id.आदान_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    साठी (पूर्णांक layer : params.swa_layers) {
        append(layer);
    }
    साठी (स्थिरांक स्वतः & [layer, heads] : params.n_heads_kv_per_layer) {
        append(layer);
        append(heads);
    }
    साठी (स्थिरांक स्वतः & [name, आदान] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(आदान.type.get_type_name());
    }
    साठी (स्थिरांक स्वतः & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    साठी (स्थिरांक स्वतः & [name, आदान] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(आदान.type.get_type_name());
        append(आदान.is_parameter);
        जर (!आदान.is_parameter) {
            append(आदान.value);
        }
    }
    परत किल्ली;
}

स्थिर तार्किक compiled_model_matches_graph(स्थिरांक ov::CompiledModel & model,
                                         स्थिरांक std::vector<std::string> & inputs,
                                         स्थिरांक std::vector<std::string> & outputs,
                                         स्थिरांक GgmlOvDecoder & decoder) {
    जर (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        परत असत्य;
    }
    स्थिरांक स्वतः & graph_inputs = decoder.get_model_inputs();
    स्थिरांक स्वतः & extra_inputs = decoder.get_model_extra_inputs();
    स्थिरांक स्वतः & graph_outputs = decoder.get_model_outputs();
    साठी (माप_प्रकार i = 0; i < inputs.size(); ++i) {
        स्थिरांक स्वतः & port = model.आदान(i);
        स्वतः graph_it = graph_inputs.find(inputs[i]);
        जर (graph_it != graph_inputs.end()) {
            जर (port.get_element_type() != graph_it->second.type ||
                !port.get_partial_shape().compatible(graph_it->second.shape)) {
                परत असत्य;
            }
            सुरूठेवा;
        }
        स्वतः extra_it = extra_inputs.find(inputs[i]);
        जर (extra_it == extra_inputs.end() || port.get_element_type() != extra_it->second.type ||
            !port.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            परत असत्य;
        }
    }
    साठी (माप_प्रकार i = 0; i < outputs.size(); ++i) {
        स्वतः graph_it = graph_outputs.find(outputs[i]);
        जर (graph_it == graph_outputs.end()) {
            जर (outputs[i].compare(0, 8, "__debug_") != 0) {
                परत असत्य;
            }
            सुरूठेवा;
        }
        स्थिरांक स्वतः & port = model.output(i);
        जर (port.get_element_type() != GgmlOvDecoder::get_ov_type(graph_it->second) ||
            !port.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(graph_it->second)))) {
            परत असत्य;
        }
    }
    परत सत्य;
}

स्थिर std::string ov_profiling_csv_field(स्थिरांक std::string & value) {
    std::string escaped = "\"";
    साठी (अक्षर c : value) {
        escaped += c;
        जर (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    परत escaped;
}

स्थिर शून्य dump_ov_profiling_info(स्थिरांक ov::InferRequest & infer_request) {
    जर (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        परत;
    }

    स्थिर std::atomic<uint64_t> inference_index{0};
    स्थिरांक uint64_t index = inference_index.fetch_add(1);
    स्थिरांक std::string path = "openvino_profile_" + std::to_string(index) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    जर (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        परत;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t total_real_time = 0;
    int64_t total_cpu_time = 0;
    माप_प्रकार executed_मोजणी = 0;
    साठी (स्थिरांक स्वतः & info : infer_request.get_profiling_info()) {
        स्थिरांक अक्षर * status = "NOT_RUN";
        जर (info.status == ov::ProfilingInfo::Status::EXECUTED) {
            status = "EXECUTED";
            total_real_time += info.real_time.count();
            total_cpu_time += info.cpu_time.count();
            executed_मोजणी++;
        } अन्यथा जर (info.status == ov::ProfilingInfo::Status::OPTIMIZED_OUT) {
            status = "OPTIMIZED_OUT";
        }

        output << status << ',' << info.real_time.count() << ',' << info.cpu_time.count() << ','
               << info.start_time.count() << ',' << ov_profiling_csv_field(info.node_type) << ','
               << ov_profiling_csv_field(info.गाठ_name) << ',' << ov_profiling_csv_field(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_मोजणी, total_real_time / 1000.0, total_cpu_time / 1000.0);
}

स्थिर शून्य reset_single_slot_recurrent_cache(ggml_cgraph * cgraph,
                                              स्थिरांक ModelParams & model_params,
                                              स्थिरांक ComputeParams & compute_params) {
    जर (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        परत;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_buffer_t> buffers;
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        स्वतः * गाठ = cgraph->nodes[i];
        जर (गाठ->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(गाठ->view_src) &&
            गाठ->view_src->ne[1] == 1 && गाठ->view_src->buffer != nullptr) {
            buffers.insert(गाठ->view_src->buffer);
        }
    }
    साठी (स्वतः * buffer : buffers) {
        ggml_backend_buffer_clear(buffer, 0);
    }
}

ov::Tensor create_ov_output_tensor(स्थिरांक std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   स्थिरांक std::shared_ptr<ov::InferRequest> & infer_request,
                                   पूर्णांक output_index,
                                   स्थिरांक ggml_tensor * ggml_tensor) {
    जर (स्वतः sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        परत *sliced;
    }

    स्वतः output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    शून्य * output_data = ggml_tensor->data;
    जर (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_index).get_shape();
    } अन्यथा {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        जर (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != nullptr &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_data = ggml_tensor->view_src->data;
        } अन्यथा {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    जर (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_buffer_is_remote(ggml_tensor) &&
        ggml_tensor->extra != nullptr) {
        स्वतः * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        जर (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            स्वतः * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            जर (tensor_extra->tensor != nullptr && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                परत *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_data);
    परत output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_rows_to_seq_axis_2(स्थिरांक ov::Tensor & kv_tensor, माप_प्रकार n_heads_kv) {
    स्थिरांक माप_प्रकार rows = kv_tensor.get_shape()[2];
    स्थिरांक माप_प्रकार head_size = kv_tensor.get_shape()[3] / n_heads_kv;
    स्थिरांक माप_प्रकार elem = kv_tensor.get_element_type().size();
    स्थिरांक माप_प्रकार head_bytes = head_size * elem;

    ov::Tensor out(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, rows, head_size});
    स्थिरांक स्वतः * src = static_cast<स्थिरांक uint8_t *>(kv_tensor.data());
    स्वतः * dst = static_cast<uint8_t *>(out.data());
    साठी (माप_प्रकार s = 0; s < rows; s++) {
        साठी (माप_प्रकार h = 0; h < n_heads_kv; h++) {
            स_नकल(dst + (h * rows + s) * head_bytes, src + (s * n_heads_kv + h) * head_bytes, head_bytes);
        }
    }
    परत out;
}

ढाँचा <typename T> शून्य set_zero_diagonal(std::vector<T> & matrix, माप_प्रकार rows, माप_प्रकार cols, T zero_मूल्य = T{}) {
    साठी (माप_प्रकार i = 0; i < rows; ++i) {
        माप_प्रकार diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = zero_मूल्य;
    }
}

ov::Tensor make_contiguous_split_आदान_tensor(स्थिरांक संरचना ggml_tensor * ggml_tensor, स्थिरांक ov::Shape & आदान_shape) {
    स्थिरांक माप_प्रकार element_size = ggml_type_size(ggml_tensor->type);
    स्थिरांक माप_प्रकार block_size = ggml_blck_size(ggml_tensor->type);

    GGML_ASSERT(block_size == 1 && "non-contiguous split inputs must be plain element types");

    स्थिरांक संरचना ggml_tensor * source_tensor = ggml_tensor->view_src != nullptr ? ggml_tensor->view_src : ggml_tensor;
    स्थिरांक माप_प्रकार source_offset = ggml_tensor->view_src != nullptr ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> source_data(ggml_nbytes(source_tensor));
    ggml_backend_tensor_get(source_tensor, source_data.data(), 0, source_data.size());

    ov::Tensor आदान_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), आदान_shape);
    स्वतः * dst = static_cast<uint8_t *>(आदान_tensor.data());
    माप_प्रकार dst_offset = 0;

    साठी (माप_प्रकार i3 = 0; i3 < static_cast<माप_प्रकार>(ggml_tensor->ne[3]); ++i3) {
        साठी (माप_प्रकार i2 = 0; i2 < static_cast<माप_प्रकार>(ggml_tensor->ne[2]); ++i2) {
            साठी (माप_प्रकार i1 = 0; i1 < static_cast<माप_प्रकार>(ggml_tensor->ne[1]); ++i1) {
                साठी (माप_प्रकार i0 = 0; i0 < static_cast<माप_प्रकार>(ggml_tensor->ne[0]); ++i0) {
                    स्थिरांक माप_प्रकार src_offset = source_offset + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::स_नकल(dst + dst_offset, source_data.data() + src_offset, element_size);
                    dst_offset += element_size;
                }
            }
        }
    }

    परत आदान_tensor;
}

ov::Tensor convert_ggml_आदान_to_ov(स्थिरांक std::shared_ptr<GgmlOvDecoder> & ggml_decoder, स्थिरांक std::string & name) {
    स्थिरांक स्वतः * ggml_tensor = ggml_decoder->get_आदान_ggml_tensor(name);

    जर (स्वतः sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        परत *sliced;
    }

    जर (ggml_tensor->extra != nullptr && !ggml_decoder->is_splited_model()) {
        स्वतः * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        जर (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            स्वतः * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            परत *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    स्वतः * आदान_data = ggml_tensor->data;
    ov::Shape आदान_shape;
    जर (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        आदान_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } अन्यथा {
        आदान_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    जर (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        परत make_contiguous_split_आदान_tensor(ggml_tensor, आदान_shape);
    }

    स्वतः आदान_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), आदान_shape, आदान_data);
    परत आदान_tensor;
}

ov::Tensor get_ov_आदान_tensor(स्थिरांक std::shared_ptr<GgmlOvDecoder> & ggml_decoder, स्थिरांक std::string & param_name) {
    ov::Tensor आदान_tensor;
    स्वतः extra_आदान = ggml_decoder->get_model_extra_inputs().find(param_name);
    जर (extra_आदान != ggml_decoder->get_model_extra_inputs().end()) {
        आदान_tensor = ov::Tensor(extra_आदान->second.type, extra_आदान->second.shape);
        *आदान_tensor.data<int64_t>() = extra_आदान->second.value;
    } अन्यथा {
        आदान_tensor = convert_ggml_आदान_to_ov(ggml_decoder, param_name);
    }
    परत आदान_tensor;
}

ov::Tensor get_ov_आदान_tensor_static_decode(स्थिरांक std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             स्थिरांक std::string & param_name) {
    // NPU decoding stage
    जर (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        परत get_ov_आदान_tensor(ggml_decoder, param_name);
    }
    स्थिरांक स्वतः * ggml_tensor = ggml_decoder->get_आदान_ggml_tensor(param_name);
    स्थिरांक स्वतः * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    जर (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        स्थिरांक पूर्णांक n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        प्रतिज्ञा(ggml_tensor->ne[0] == n_planes);
        ov::Shape आदान_shape = {1, 1, 1, (माप_प्रकार) n_planes};
        ov::Tensor आदान_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), आदान_shape);
        std::स_नकल(आदान_tensor.data(), ggml_tensor->data, n_planes * ggml_type_size(ggml_tensor->type));
        परत आदान_tensor;
    }

    जर (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape आदान_shape = {1, 1, 1, 1};
        ov::Tensor आदान_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), आदान_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        प्रतिज्ञा(ggml_tensor->ne[0] == 1);
        प्रतिज्ञा(inp_out_id == 0);
        *आदान_tensor.data<int32_t>() = inp_out_id;
        परत आदान_tensor;
    }

    जर (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        माप_प्रकार context_size = ggml_decoder->get_ctx_size();
        जर (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_data =
                pad_आदान<ggml_fp16_t>(ggml_tensor, 1, context_size, GGML_FP32_TO_FP16(-अनन्त));
            ov::Tensor आदान_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_size});
            std::स_नकल(आदान_tensor.data(), padded_data.data(), padded_data.size() * आकार(ggml_fp16_t));
            परत आदान_tensor;
        }

        std::vector<दशांश> padded_data = pad_आदान<दशांश>(ggml_tensor, 1, context_size, -अनन्त);
        ov::Tensor आदान_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        स्वतः * data_ptr = आदान_tensor.data<दशांश>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
        परत आदान_tensor;
    }

    परत get_ov_आदान_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_आदान_tensor_static_prefill(स्थिरांक std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              स्थिरांक std::string & param_name,
                                              पूर्णांक chunk_index) {
    // NPU prompt processing stage
    स्थिरांक माप_प्रकार आदान_len = ggml_decoder->get_आदान_len();
    स्थिरांक माप_प्रकार chunk_size = ggml_decoder->m_prefill_chunk_size;
    स्थिरांक माप_प्रकार chunk_valid_size = std::min(chunk_size, आदान_len - chunk_index * chunk_size);
    स्थिरांक माप_प्रकार chunk_pad_size = chunk_size - chunk_valid_size;

    जर (param_name == "chunk_valid_len") {
        ov::Tensor आदान_tensor(ov::element::i64, ov::Shape{1});
        *आदान_tensor.data<int64_t>() = (int64_t) chunk_valid_size;
        परत आदान_tensor;
    }
    जर (chunk_index > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor आदान_tensor(ov::element::i64, ov::Shape{1});
        *आदान_tensor.data<int64_t>() = 0;
        परत आदान_tensor;
    }
    जर (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        परत get_ov_आदान_tensor(ggml_decoder, param_name);
    }
    स्थिरांक स्वतः * ggml_tensor = ggml_decoder->get_आदान_ggml_tensor(param_name);
    स्थिरांक स्वतः * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    जर (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        स्थिरांक पूर्णांक n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        स्थिरांक माप_प्रकार element_size = ggml_type_size(ggml_tensor->type);
        ov::Shape आदान_shape = {1, 1, 1, (माप_प्रकार) n_planes * chunk_size};
        ov::Tensor आदान_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), आदान_shape);
        साठी (पूर्णांक p = 0; p < n_planes; p++) {
            स्थिरांक अक्षर * src =
                (स्थिरांक अक्षर *) ggml_tensor->data + (p * आदान_len + chunk_index * chunk_size) * element_size;
            अक्षर * dst = (अक्षर *) आदान_tensor.data() + p * chunk_size * element_size;
            std::स_नकल(dst, src, chunk_valid_size * element_size);
            जर (chunk_pad_size > 0) {
                जर (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t last_मूल्य = *((स्थिरांक int32_t *) src + chunk_valid_size - 1);
                    int32_t * out = (int32_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_मूल्य + 1);
                } अन्यथा जर (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t last_मूल्य = *((स्थिरांक int64_t *) src + chunk_valid_size - 1);
                    int64_t * out = (int64_t *) dst;
                    std::fill(out + chunk_valid_size, out + chunk_size, last_मूल्य + 1);
                } अन्यथा {
                    फेका std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        परत आदान_tensor;
    }

    जर (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape आदान_shape = {1, 1, 1, chunk_size};
        ov::Tensor आदान_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), आदान_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        माप_प्रकार element_size = ggml_type_size(ggml_tensor->type);
        शून्य * आदान_data = (अक्षर *) ggml_tensor->data + chunk_index * chunk_size * element_size;
        std::स_नकल(आदान_tensor.data(), आदान_data, chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        जर (chunk_pad_size > 0) {
            जर (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_मूल्य =
                    *((int32_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int32_t * output_data = आदान_tensor.data<int32_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_मूल्य + 1);
            } अन्यथा जर (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_मूल्य =
                    *((int64_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int64_t * output_data = आदान_tensor.data<int64_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_मूल्य + 1);
            } अन्यथा {
                फेका std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        परत आदान_tensor;
    }

    जर (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        माप_प्रकार output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape आदान_shape = {1, 1, 1, output_len};
        ov::Tensor आदान_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), आदान_shape);
        जर (ggml_tensor->ne[0] == 0) {
            *आदान_tensor.data<int32_t>() = 0;
        } अन्यथा {
            स्वतः * data_addr = आदान_tensor.data<int32_t>();
            साठी (माप_प्रकार i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        परत आदान_tensor;
    }

    जर (GgmlOvDecoder::is_inp_scale_rows(ggml_tensor, op)) {
        ov::Tensor आदान_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, 1});
        स्वतः * dst = आदान_tensor.data<दशांश>();
        स्थिरांक स्वतः * src = static_cast<स्थिरांक दशांश *>(ggml_tensor->data) + chunk_index * chunk_size;
        std::copy(src, src + chunk_valid_size, dst);
        std::fill(dst + chunk_valid_size, dst + chunk_size, 1.0f);
        परत आदान_tensor;
    }

    जर (GgmlOvDecoder::is_inp_mean(ggml_tensor, op)) {
        स्थिरांक माप_प्रकार n_seqs = ggml_tensor->ne[1];
        स्थिरांक माप_प्रकार src_stride = ggml_tensor->ne[0];
        स्थिरांक माप_प्रकार copy_len = std::min<माप_प्रकार>(chunk_valid_size, src_stride - chunk_index * chunk_size);
        ov::Tensor आदान_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_size});
        स्वतः * dst = आदान_tensor.data<दशांश>();
        std::fill(dst, dst + n_seqs * chunk_size, 0.0f);
        स्थिरांक स्वतः * src = static_cast<स्थिरांक दशांश *>(ggml_tensor->data) + chunk_index * chunk_size;
        साठी (माप_प्रकार s = 0; s < n_seqs; s++) {
            std::स_नकल(dst + s * chunk_size, src + s * src_stride, copy_len * आकार(दशांश));
        }
        परत आदान_tensor;
    }

    जर (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        माप_प्रकार cols = ggml_tensor->ne[0];
        माप_प्रकार rows = ggml_tensor->ne[1];
        माप_प्रकार chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        माप_प्रकार context_size = ggml_decoder->get_ctx_size();
        जर (ggml_tensor->type == GGML_TYPE_F16) {
            स्थिरांक स्वतः * ggml_data =
                static_cast<स्थिरांक ggml_fp16_t *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
            std::vector<ggml_fp16_t> padded_data = pad_आदान<ggml_fp16_t>(ggml_data, chunk_valid_rows, cols, chunk_size,
                                                                          context_size, GGML_FP32_TO_FP16(-अनन्त));
            set_zero_diagonal(padded_data, chunk_size, context_size, GGML_FP32_TO_FP16(0.0f));
            ov::Tensor आदान_tensor(ov::element::f16, ov::Shape{1, 1, chunk_size, context_size});
            std::स_नकल(आदान_tensor.data(), padded_data.data(), padded_data.size() * आकार(ggml_fp16_t));
            परत आदान_tensor;
        }

        स्थिरांक स्वतः * ggml_data = static_cast<स्थिरांक दशांश *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
        std::vector<दशांश> padded_data =
            pad_आदान<दशांश>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -अनन्त);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor आदान_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        स्वतः * data_ptr = आदान_tensor.data<दशांश>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        परत आदान_tensor;
    }

    परत get_ov_आदान_tensor(ggml_decoder, param_name);
}

गणना ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               स्थिरांक std::string & device,
                               स्थिरांक ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    जर (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        परत GGML_STATUS_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_lock(cache.mutex);
    तार्किक naive = सत्य;
    स्वतः model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    स्वतः decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    स्वतः आदान_model = std::make_shared<ov::frontend::ggml::आदान_Model>(decoder);
    स्वतः model = ov::frontend::ggml::FrontEnd::convert(आदान_model, naive);
    जर (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    स्वतः remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    जर (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    } अन्यथा {
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    }
    जर (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } अन्यथा {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> आदान_names;
    std::vector<std::string> output_names;
    साठी (स्थिरांक स्वतः & param : model->get_parameters()) {
        आदान_names.push_back(param->get_friendly_name());
    }
    साठी (स्थिरांक स्वतः & result : model->get_results()) {
        output_names.push_back(result->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    आदान_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_lock.unlock();

    साठी (माप_प्रकार i = 0; i < आदान_names.size(); i++) {
        स्थिरांक स्वतः & param_name = आदान_names[i];
        स्वतः आदान_tensor = get_ov_आदान_tensor(decoder, param_name);
        infer_request->set_आदान_tensor(i, आदान_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    साठी (माप_प्रकार i = 0; i < output_names.size(); i++) {
        स्वतः output_tensor = infer_request->get_output_tensor(i);
        स्थिरांक स्वतः & model_outputs = decoder->get_model_outputs();
        स्वतः model_output_it = model_outputs.find(output_names[i]);
        जर (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            जर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_names[i], output_tensor, output_tensor.data());
            }
            सुरूठेवा;
        }
        स्वतः * ggml_tensor = model_output_it->second;
        std::स_नकल(ggml_tensor->data, output_tensor.data(), output_tensor.get_byte_size());
    }
    परत GGML_STATUS_SUCCESS;
}

गणना ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, स्थिरांक std::shared_ptr<ov_runtime_context> & r_ctx) {
    स्वतः & core = ov_singleton_core();
    स्थिरांक स्वतः & config = ggml_openvino_get_compile_config();
    स्थिरांक स्वतः & device = r_ctx->device;
    स्थिर स्वतः is_static = असत्य;

    स्थिर स्थिरांक तार्किक cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    स्वतः start_time = ggml_time_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    स्थिरांक int64_t cache_किल्ली_start_time = ggml_time_us();
    graph_किल्ली किल्ली(cgraph);
    स्थिरांक int64_t cache_किल्ली_compute_time = ggml_time_us() - cache_किल्ली_start_time;
    int64_t cache_lookup_time = 0;
    तार्किक किल्ली_seen = असत्य;
    जर (!cache_disabled) {
        स्थिरांक int64_t cache_lookup_start_time = ggml_time_us();
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        किल्ली_seen = r_ctx->decoder_cache.find(किल्ली) != r_ctx->decoder_cache.end();
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;
    }

    स्थिरांक int64_t model_split_check_start_time = ggml_time_us();
    तार्किक model_is_splitted = किल्ली_seen ? असत्य : is_model_splitted(cgraph);
    स्थिरांक int64_t model_split_check_time = ggml_time_us() - model_split_check_start_time;

    जर (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        परत GGML_STATUS_FAILED;
    }
    जर (is_naive(cgraph)) {
        जर (!model_is_splitted) {
            परत naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    स्थिरांक int64_t graph_param_compute_start_time = ggml_time_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    स्थिरांक int64_t graph_param_compute_time = ggml_time_us() - graph_param_compute_start_time;

    स्थिरांक तार्किक cache_enabled = !model_is_splitted && !cache_disabled;
    स्थिरांक तार्किक stateful_recurrent = r_ctx->stateful && m_params.state_size >= 0;
    स्थिरांक तार्किक stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    जर (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_active > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        परत GGML_STATUS_FAILED;
    }
    जर (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        परत GGML_STATUS_FAILED;
    }
    तार्किक cache_hit = असत्य;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        स्थिरांक int64_t cache_lookup_start_time = ggml_time_us();
        जर (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            स्वतः it = r_ctx->decoder_cache.find(किल्ली);
            cache_hit = it != r_ctx->decoder_cache.end();
            जर (cache_hit) {
                entry = it->second;
            } अन्यथा {
                r_ctx->clear_caches_locked();
                स्वतः mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[किल्ली] = entry;
            }
        } अन्यथा {
            स्वतः mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_hit = असत्य;
        }
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;

        std::lock_guard<std::mutex> lock(*(entry->mutex));
        cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(किल्ली) != 0;

        जर (cache_hit) {
            ggml_decoder = entry->ptr;
            old_m_params = ggml_decoder->get_model_params();
            जर (!ggml_decoder->is_splited_model()) {
                cache_hit = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_आदान_names;
        std::vector<std::string> ov_output_names;

        जर (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            जर (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(किल्ली);
                ov_आदान_names = r_ctx->ov_आदान_names_cache.at(किल्ली);
                ov_output_names = r_ctx->ov_output_names_cache.at(किल्ली);
            }

            जर (stateful_kv_only) {
                स्थिरांक स्वतः * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_data = (int32_t *) inp_pos->data;
                स्वतः pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                जर (pos_data[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_size = pos_shape[3];
                } अन्यथा जर (r_ctx->stateful_kv_size == static_cast<माप_प्रकार>(pos_data[0])) {
                    r_ctx->stateful_kv_size += pos_shape[3];
                } अन्यथा {
                    स्थिरांक माप_प्रकार pos_begin = static_cast<माप_प्रकार>(pos_data[0]);
                    स्थिरांक तार्किक refill = pos_begin > r_ctx->stateful_kv_size;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    जर (refill && !ggml_decoder->get_model_params().swa_layers.empty()) {
                        स्थिरांक पूर्णांक n_swa = ggml_decoder->get_compute_params().swa_window;
                        जर (n_swa < 0 || static_cast<माप_प्रकार>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_size, n_swa);
                            परत GGML_STATUS_FAILED;
                        }
                    }

                    स्थिरांक तार्किक relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    स्वतः states = infer_request->query_state();
                    साठी (स्वतः state : states) {
                        स्वतः state_tensor = state.get_state();
                        स्वतः state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        जर (स्वतः it = r_ctx->kv_state_आदान_name_map.find(state.get_name());
                            it != r_ctx->kv_state_आदान_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        पूर्णांक n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        जर (स्वतः layer = extract_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        स्थिरांक तार्किक relayout_this_state = relayout_enabled;
                        स्थिरांक माप_प्रकार seq_axis = relayout_this_state ? 2 : 1;
                        स्थिरांक माप_प्रकार head_axis = seq_axis == 2 ? 1 : 2;

                        जर (refill) {
                            जर (state_name.empty()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                परत GGML_STATUS_FAILED;
                            }
                            स्वतः kv_tensor = get_ov_आदान_tensor(ggml_decoder, state_name);
                            जर (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_rows_to_seq_axis_2(kv_tensor, (माप_प्रकार) n_heads_kv);
                            } अन्यथा {
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
                        जर (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            परत GGML_STATUS_FAILED;
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
        } अन्यथा {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            स्वतः shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
            स्वतः weight_names = get_weight_names(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            स्थिरांक std::string model_cache_dir = ggml_openvino_model_cache_dir();
            स्थिरांक std::string exact_किल्ली = cache_enabled ? compiled_graph_किल्ली(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_path, manifest_path;
            जर (!model_cache_dir.empty() && !model_is_splitted) {
                स्थिरांक uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/सत्य, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_graph_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/सत्य, m_params.rope_params, 16, extra_cfg,
                                                    compiled_graph_किल्ली(cgraph, *ggml_decoder, device, 0, सत्य));
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            std::string shared_किल्ली =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_किल्ली) : "";
            ov::CompiledModel shared_model;
            तार्किक imported = असत्य;
            स्वतः shared_it = shared_cache->graphs.find(shared_किल्ली);
            जर (!shared_किल्ली.empty() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_graph(shared_it->second.decode, shared_it->second.आदान_names,
                                                               shared_it->second.output_names, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_आदान_names = shared_it->second.आदान_names;
                ov_output_names = shared_it->second.output_names;
                imported = सत्य;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } अन्यथा जर (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_किल्ली = exact_किल्ली;
                shared_it = shared_cache->graphs.find(shared_किल्ली);
                जर (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_आदान_names = shared_it->second.आदान_names;
                    ov_output_names = shared_it->second.output_names;
                    imported = सत्य;
                }
                model_fp = exact_fp;
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            जर (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(किल्ली);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            जर (!model_cache_dir.empty()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
            }
            जर (!imported && !model_cache_dir.empty() && !model_is_splitted) {
                स्थिरांक uint64_t preferred_fp = model_fp;
                तार्किक dynamic_incompatible = preferred_fp == exact_fp;
                साठी (पूर्णांक candidate = 0; candidate < 2; ++candidate) {
                    जर (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        थांबा;
                    }
                    स्थिरांक uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    स्थिरांक std::string candidate_blob =
                        ggml_openvino_model_cache_blob_path(model_cache_dir, candidate_fp);
                    स्थिरांक std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_path(model_cache_dir, candidate_fp);
                    std::ifstream blob_in(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    जर (!blob_in.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        सुरूठेवा;
                    }
                    स्थिरांक int64_t import_start = ggml_time_us();
                    प्रयत्न {
                        ov::CompiledModel cm;
                        स्वतः remote_context = ggml_openvino_get_remote_context();
                        जर (remote_context.has_value()) {
                            cm = core.import_model(blob_in, remote_context.value(), mc_config);
                        } अन्यथा {
                            cm = core.import_model(blob_in, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        जर (!saved_inputs.empty() || !saved_outputs.empty()) {
                            ov_आदान_names = std::move(saved_inputs);
                            ov_output_names = std::move(saved_outputs);
                        } अन्यथा {
                            साठी (स्थिरांक स्वतः & p : cm.inputs()) {
                                ov_आदान_names.push_back(p.get_गाठ()->get_friendly_name());
                            }
                            साठी (स्थिरांक स्वतः & o : cm.outputs()) {
                                ov_output_names.push_back(o.get_गाठ()->get_friendly_name());
                            }
                        }
                        जर (!compiled_model_matches_graph(cm, ov_आदान_names, ov_output_names, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_आदान_names.clear();
                            ov_output_names.clear();
                            सुरूठेवा;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        entry->ptr = ggml_decoder;
                        imported = सत्य;
                        model_fp = candidate_fp;
                        blob_path = candidate_blob;
                        manifest_path = candidate_manifest;
                        जर (candidate_fp == exact_fp) {
                            shared_किल्ली = exact_किल्ली;
                            shared_it = shared_cache->graphs.find(shared_किल्ली);
                        }
                        जर (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_time_us() - import_start) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } पकडा (स्थिरांक std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = असत्य;
                        ov_आदान_names.clear();
                        ov_output_names.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                जर (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_किल्ली = exact_किल्ली;
                    shared_it = shared_cache->graphs.find(shared_किल्ली);
                    blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                    manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            जर (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_path.c_str());
                परत GGML_STATUS_FAILED;
            }
            जर (!imported && ggml_openvino_weight_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                परत GGML_STATUS_FAILED;
            }
            जर (imported) {
                decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            } अन्यथा {
                स्वतः model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_end_time = ggml_time_us();

                स्वतः आदान_model = std::make_shared<ov::frontend::ggml::आदान_Model>(ggml_decoder);
                model = ov::frontend::ggml::FrontEnd::convert(आदान_model);
                ggml_decoder->clear_model_weights();
                conversion_end_time = ggml_time_us();

                जर (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    अक्षर timestamped_filename[64];
                    स्वतः timestamp = (दीर्घ दीर्घ) ggml_time_us();
                    snprintf(timestamped_filename, आकार(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                स्थिरांक ov::AnyMap & compile_config = model_cache_dir.empty() ? config : mc_config;
                ov::CompiledModel compiled_model;
                स्वतः remote_context = ggml_openvino_get_remote_context();
                जर (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } अन्यथा {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_end_time = ggml_time_us();

                साठी (स्थिरांक स्वतः & ov_param : model->get_parameters()) {
                    ov_आदान_names.push_back(ov_param->get_friendly_name());
                }
                साठी (स्थिरांक स्वतः & ov_output : model->get_results()) {
                    ov_output_names.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                जर (!model_cache_dir.empty() && !model_is_splitted && model_fp != 0) {
                    प्रयत्न {
                        स्थिरांक std::string blob_tmp = ggml_openvino_model_cache_temp_path(blob_path);
                        स्थिरांक std::string manifest_tmp = ggml_openvino_model_cache_temp_path(manifest_path);
                        जर (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_आदान_names,
                                                                     ov_output_names)) {
                            std::ofstream blob_out(blob_tmp, std::ios::binary | std::ios::trunc);
                            जर (blob_out.is_open()) {
                                compiled_model.export_model(blob_out);
                                blob_out.बंद();
                                जर (blob_out.good()) {
                                    जर (std::नाम(blob_tmp.c_str(), blob_path.c_str()) == 0 &&
                                        std::नाम(manifest_tmp.c_str(), manifest_path.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_path.c_str());
                                    } अन्यथा {
                                        std::हटाओ(blob_tmp.c_str());
                                        std::हटाओ(manifest_tmp.c_str());
                                    }
                                } अन्यथा {
                                    std::हटाओ(blob_tmp.c_str());
                                    std::हटाओ(manifest_tmp.c_str());
                                }
                            } अन्यथा {
                                std::हटाओ(manifest_tmp.c_str());
                            }
                        }
                    } पकडा (स्थिरांक std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                entry->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            entry->ptr = ggml_decoder;
            जर (!shared_किल्ली.empty() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_किल्ली,
                                             ov_compiled_graph{shared_model, {}, ov_आदान_names, ov_output_names});
            }
            जर (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[किल्ली] = infer_request;
                r_ctx->ov_आदान_names_cache[किल्ली] = ov_आदान_names;
                r_ctx->ov_output_names_cache[किल्ली] = ov_output_names;
            }

            जर (stateful_kv_only && cache_enabled) {
                स्थिरांक स्वतः * inp_pos = get_inp_pos_tensor(cgraph);
                स्वतः pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                स्थिरांक int32_t pos_begin = ((int32_t *) inp_pos->data)[0];
                जर (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    परत GGML_STATUS_FAILED;
                }
                r_ctx->stateful_kv_size = pos_shape[3];
                स्थिरांक स्वतः kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                साठी (स्थिरांक स्वतः & pair : kv_param_res_names) {
                    r_ctx->kv_state_आदान_name_map[pair.first + pair.second] = pair.first;
                }
            }
        }

        जर (stateful_recurrent) {
            स्थिरांक स्वतः * inp_pos = get_inp_pos_tensor(cgraph);
            स्थिरांक int32_t pos_begin = static_cast<स्थिरांक int32_t *>(inp_pos->data)[0];
            जर (pos_begin == 0) {
                infer_request->reset_state();
            } अन्यथा जर (!cache_hit || old_m_params.kv_buffer_changed(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<माप_प्रकार>(pos_begin) != r_ctx->stateful_kv_size) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                परत GGML_STATUS_FAILED;
            }
        } अन्यथा {
            reset_single_slot_recurrent_cache(cgraph, m_params, c_params);
        }

        साठी (माप_प्रकार i = 0; i < ov_आदान_names.size(); i++) {
            स्थिरांक स्वतः & param_name = ov_आदान_names[i];
            स्वतः आदान_tensor = get_ov_आदान_tensor(ggml_decoder, param_name);
            infer_request->set_आदान_tensor(i, आदान_tensor);

            जर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_आदान_tensor_info(param_name, आदान_tensor);
            }
        }

        साठी (माप_प्रकार i = 0; i < ov_output_names.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            स्थिरांक स्वतः & model_outputs = ggml_decoder->get_model_outputs();
            स्वतः model_output_it = model_outputs.find(ov_output_names[i]);
            जर (model_output_it == model_outputs.end()) {
                सुरूठेवा;
            }
            स्वतः * ggml_tensor = model_output_it->second;
            जर (ggml_nbytes(ggml_tensor) == 0) {
                सुरूठेवा;
            }
            स्वतः output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        जर (stateful_recurrent) {
            स्थिरांक स्वतः * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_size =
                static_cast<स्थिरांक int32_t *>(inp_pos->data)[0] + get_inp_pos_n_tokens(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        जर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            साठी (माप_प्रकार i = 0; i < ov_output_names.size(); i++) {
                स्थिरांक स्वतः output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }

        जर (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_किल्ली_compute_time / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_split_check_time / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", graph_param_compute_time / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_time / 1000.0);
            जर (!cache_hit) {
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
    जर (cache_hit && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_lock(r_ctx->compiled_cache->mutex);
        जर (!ggml_openvino_weight_buffers_released()) {
            ggml_openvino_release_weight_buffers();
        }
    }

    परत GGML_STATUS_SUCCESS;
}

ov::AnyMap without_npuw(स्थिरांक ov::AnyMap & config) {
    ov::AnyMap out;
    साठी (स्थिरांक स्वतः & kv : config) {
        जर (kv.first.rfind("NPUW", 0) == 0 || kv.first == "NPU_USE_NPUW") {
            सुरूठेवा;
        }
        out.insert(kv);
    }
    परत out;
}

गणना ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, स्थिरांक std::shared_ptr<ov_runtime_context> & r_ctx) {
    स्वतः & core = ov_singleton_core();

    स्वतः get_prefill_chunk_size = [] {
        स्थिर स्थिरांक पूर्णांक chunk_size = []() {
            पूर्णांक env_prefill_chunk_size = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            परत env_prefill_chunk_size > 0 ? env_prefill_chunk_size : 256;
        }();
        परत chunk_size;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    स्थिर std::string device = ggml_openvino_get_device_name();
    स्थिर स्वतः is_static = सत्य;
    स्थिर स्वतः stateful = असत्य;

    स्वतः prefill_chunk_size = get_prefill_chunk_size();
    स्थिरांक स्वतः & config = ggml_openvino_get_compile_config();

    जर (is_naive(cgraph)) {
        परत naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    स्वतः start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    स्थिरांक स्वतः * inp_pos = get_inp_pos_tensor(cgraph);
    स्थिरांक तार्किक no_kv_cache = m_params.is_cacheless_attn;
    स्थिरांक स्वतः is_prefill = no_kv_cache ? सत्य : get_is_prefill(cgraph, inp_pos);
    स्थिरांक ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    जर (m_params.n_heads_kv == -1) {
        prefill_chunk_size = inp_pos->ne[0];
    }
    graph_किल्ली किल्ली(cgraph);
    स्थिर स्थिरांक तार्किक cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    तार्किक cache_hit = असत्य;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;
    int64_t ov_raw_infer_total = 0;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    जर (cache_enabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        स्वतः it = r_ctx->decoder_cache.find(किल्ली);
        cache_hit = it != r_ctx->decoder_cache.end();
        जर (cache_hit) {
            entry = it->second;
        } अन्यथा {
            r_ctx->clear_caches_locked();
            स्वतः mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[किल्ली] = entry;
        }
    } अन्यथा {
        स्वतः mutex = std::make_shared<std::mutex>();
        entry = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_hit = असत्य;
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));
    cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(किल्ली) != 0 &&
                r_ctx->infer_request_cache_prefill.count(किल्ली) != 0;

    जर (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_आदान_names_local;
    std::vector<std::string> ov_output_names_local;

    जर (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        जर (old_m_params.kv_buffer_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(किल्ली) : r_ctx->infer_request_cache.at(किल्ली);
            ov_आदान_names_local = r_ctx->ov_आदान_names_cache.at(किल्ली);
            ov_output_names_local = r_ctx->ov_output_names_cache.at(किल्ली);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } अन्यथा {
        जर (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(किल्ली);
            r_ctx->infer_request_cache_prefill.erase(किल्ली);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        स्वतः shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
        स्वतः weight_names = get_weight_names(cgraph);
        स्वतः local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names, is_static,
                                                             stateful, असत्य, is_prefill, prefill_chunk_size);
        स्थिरांक std::string shared_किल्ली =
            cache_enabled ? compiled_graph_किल्ली(cgraph, *local_decoder, device, prefill_chunk_size) : "";
        स्वतः shared_it = shared_cache->graphs.find(shared_किल्ली);
        जर (!shared_किल्ली.empty() && shared_it != shared_cache->graphs.end()) {
            स्वतः & compiled = shared_it->second;
            स्वतः prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            स्वतः decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            entry->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_आदान_names_local = compiled.आदान_names;
            ov_output_names_local = compiled.output_names;
            r_ctx->infer_request_cache_prefill[किल्ली] = prefill_request;
            r_ctx->infer_request_cache[किल्ली] = decode_request;
            r_ctx->ov_आदान_names_cache[किल्ली] = ov_आदान_names_local;
            r_ctx->ov_output_names_cache[किल्ली] = ov_output_names_local;
            decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } अन्यथा {
            std::shared_ptr<ov::Model> model;
            स्वतः model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

            स्वतः ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, असत्य, सत्य, prefill_chunk_size);
            स्वतः ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, असत्य, असत्य, prefill_chunk_size);
            decoder_end_time = ggml_time_us();

            स्थिरांक तार्किक dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            स्थिरांक स्वतः dump_ir_timestamp = static_cast<दीर्घ दीर्घ>(ggml_time_us());

            स्वतः build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          स्थिरांक std::shared_ptr<GgmlOvDecoder> & decoder, स्थिरांक अक्षर * tag,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_end_time, int64_t & local_compile_end_time) {
                स्वतः आदान_model = std::make_shared<ov::frontend::ggml::आदान_Model>(decoder);
                model = ov::frontend::ggml::FrontEnd::convert(आदान_model);
                decoder->clear_model_weights();
                local_conversion_end_time = ggml_time_us();

                जर (dump_ir) {
                    अक्षर timestamped_filename[64];
                    snprintf(timestamped_filename, आकार(timestamped_filename), "model_%s_%lld.xml", tag,
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
            जर (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_end_time = prefill_conversion_end_time;
                decode_compile_end_time = prefill_compile_end_time;
            } अन्यथा {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_end_time, decode_compile_end_time);
            }
            conversion_end_time = std::max(prefill_conversion_end_time, decode_conversion_end_time);
            compile_end_time = std::max(prefill_compile_end_time, decode_compile_end_time);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            entry->ptr = ggml_decoder;

            साठी (स्थिरांक स्वतः & ov_param : model->get_parameters()) {
                ov_आदान_names_local.push_back(ov_param->get_friendly_name());
            }
            साठी (स्थिरांक स्वतः & ov_output : model->get_results()) {
                ov_output_names_local.push_back(ov_output->get_friendly_name());
            }

            जर (!shared_किल्ली.empty()) {
                shared_cache->graphs.emplace(
                    shared_किल्ली, ov_compiled_graph{compiled_model_decode, compiled_model_prefill, ov_आदान_names_local,
                                                  ov_output_names_local});
            }

            जर (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[किल्ली] = infer_request_prefill;
                r_ctx->infer_request_cache[किल्ली] = infer_request_decode;
                r_ctx->ov_आदान_names_cache[किल्ली] = ov_आदान_names_local;
                r_ctx->ov_output_names_cache[किल्ली] = ov_output_names_local;
            }
        }
    }

    reset_single_slot_recurrent_cache(cgraph, m_params, c_params);

    जर (is_prefill) {
        स्वतः inp_len = get_inp_pos_n_tokens(cgraph, inp_pos);
        साठी (पूर्णांक chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            साठी (माप_प्रकार i = 0; i < ov_आदान_names_local.size(); i++) {
                स्थिरांक स्वतः & param_name = ov_आदान_names_local[i];
                स्वतः आदान_tensor = get_ov_आदान_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_आदान_tensor(i, आदान_tensor);

                जर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    स्थिरांक स्वतः आदान_tensor = infer_request->get_आदान_tensor(i);
                    print_आदान_tensor_info(param_name, आदान_tensor);
                }
            }

            साठी (माप_प्रकार i = 0; i < ov_output_names_local.size(); i++) {
                स्थिरांक स्वतः & model_outputs = ggml_decoder->get_model_outputs();
                स्वतः model_output_it = model_outputs.find(ov_output_names_local[i]);
                जर (model_output_it == model_outputs.end()) {
                    सुरूठेवा;
                }
                स्वतः * ggml_tensor = model_output_it->second;
                जर (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    सुरूठेवा;
                }
                स्वतः output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            ov_raw_infer_start = ggml_time_us();
            infer_request->infer();
            ov_raw_infer_total += ggml_time_us() - ov_raw_infer_start;

            जर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                साठी (माप_प्रकार i = 0; i < ov_output_names_local.size(); i++) {
                    स्थिरांक स्वतः output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } अन्यथा {
        साठी (माप_प्रकार i = 0; i < ov_आदान_names_local.size(); i++) {
            स्थिरांक स्वतः & param_name = ov_आदान_names_local[i];
            स्वतः आदान_tensor = get_ov_आदान_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_आदान_tensor(i, आदान_tensor);

            जर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                स्थिरांक स्वतः आदान_tensor = infer_request->get_आदान_tensor(i);
                print_आदान_tensor_info(param_name, आदान_tensor);
            }
        }

        साठी (माप_प्रकार i = 0; i < ov_output_names_local.size(); i++) {
            स्थिरांक स्वतः & model_outputs = ggml_decoder->get_model_outputs();
            स्वतः model_output_it = model_outputs.find(ov_output_names_local[i]);
            जर (model_output_it == model_outputs.end()) {
                सुरूठेवा;
            }
            स्वतः * ggml_tensor = model_output_it->second;
            जर (ggml_nbytes(ggml_tensor) == 0) {
                सुरूठेवा;
            }
            स्वतः output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        ov_raw_infer_total = infer_end_time - ov_raw_infer_start;

        जर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            साठी (माप_प्रकार i = 0; i < ov_output_names_local.size(); i++) {
                स्थिरांक स्वतः output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
            }
        }
    }

    जर (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
        जर (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_end_time - decoder_end_time) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_total / 1000.0);
    }

    परत GGML_STATUS_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
स्थिर ggml_cgraph * get_compute_graph(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    स्वतः is_skipped = [](स्थिरांक ggml_tensor * गाठ) {
        परत गाठ->op != GGML_OP_NONE && !(गाठ->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    जर (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        परत cgraph;
    }
    स्वतः & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](स्थिरांक ggml_tensor * गाठ) { परत !is_skipped(गाठ); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.data();
    compute.graph.n_nodes = (पूर्णांक) compute.nodes.size();
    परत &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
गणना ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    प्रयत्न {
        जर (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        स्थिरांक स्वतः is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_lock(r_ctx->execution_mutex);
        cgraph = get_compute_graph(cgraph, *r_ctx);

        परत is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } पकडा (स्थिरांक ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        परत GGML_STATUS_FAILED;
    } पकडा (स्थिरांक std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        परत GGML_STATUS_FAILED;
    } पकडा (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        परत GGML_STATUS_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
तार्किक is_model_splitted(ggml_cgraph * cgraph) {
    स्थिर स्थिरांक तार्किक fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    जर (!fallback_enabled) {
        परत असत्य;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    जर (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        परत असत्य;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * गाठ = cgraph->nodes[i];
        पूर्णांक use_मोजणी = cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, गाठ)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        जर ((cgraph->n_nodes <= 1 && use_मोजणी == 0) ||
            (cgraph->n_nodes <= 1 && गाठ->op == GGML_OP_VIEW && use_मोजणी == 1 && गाठ->src[0] != nullptr &&
             गाठ->src[0]->op == GGML_OP_NONE)) {
            परत असत्य;
        }
        जर (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_PERMUTE)) {
            परत असत्य;
        }
        पूर्णांक आदान_use_मोजणी = 0;
        साठी (पूर्णांक j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_गाठ = cgraph->nodes[j];
            साठी (पूर्णांक k = 0; k < GGML_MAX_SRC; k++) {
                जर (other_गाठ->src[k] == गाठ) {
                    आदान_use_मोजणी++;
                }
            }
        }
        जर (use_मोजणी != आदान_use_मोजणी && गाठ->op != GGML_OP_NONE) {
            परत सत्य;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    जर (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_weight_names(cgraph);
    } अन्यथा {
        साठी (स्थिरांक स्वतः & kv : GgmlOvDecoder::create_weight_nodes(cgraph, सत्य)) {
            model_weights.insert(kv.first);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * गाठ = cgraph->nodes[i];
        साठी (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = गाठ->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            जर (src != nullptr && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.empty() == असत्य &&
                model_leafs.find(src) == model_leafs.end()) {
                जर (GgmlOvDecoder::is_inp_tok(src, गाठ)) {
                    परत असत्य;
                }
                परत सत्य;
            }
        }
    }
    परत असत्य;
}

तार्किक is_naive(ggml_cgraph * cgraph) {
    constexpr पूर्णांक naive_graph_size_threshold = 20;
    पूर्णांक count = 0;
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        जर (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    परत count < naive_graph_size_threshold;
}

माप_प्रकार checksum(स्थिरांक शून्य * data, माप_प्रकार size) {
    स्थिरांक uint8_t * bytes = static_cast<स्थिरांक uint8_t *>(data);
    माप_प्रकार sum = 0;
    साठी (माप_प्रकार i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    परत sum;
}

तार्किक save_ggml_tensor_data_to_txt(स्थिरांक ggml_tensor * tensor, स्थिरांक std::string & file_path) {
    जर (tensor == nullptr || tensor->data == nullptr) {
        परत असत्य;
    }

    std::ofstream out(file_path);
    जर (!out.is_open()) {
        परत असत्य;
    }

    स्थिरांक माप_प्रकार n = ggml_nelements(tensor);
    out << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    निवड (tensor->type) {
    स्थिती GGML_TYPE_F32: {
        स्थिरांक स्वतः * data = static_cast<स्थिरांक दशांश *>(tensor->data);
        साठी (माप_प्रकार i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        थांबा;
    }
    स्थिती GGML_TYPE_F16: {
        स्थिरांक स्वतः * data = static_cast<स्थिरांक ggml_fp16_t *>(tensor->data);
        साठी (माप_प्रकार i = 0; i < n; ++i) {
            out << ggml_fp16_to_fp32(data[i]) << '\n';
        }
        थांबा;
    }
    स्थिती GGML_TYPE_BF16: {
        स्थिरांक स्वतः * data = static_cast<स्थिरांक ggml_bf16_t *>(tensor->data);
        साठी (माप_प्रकार i = 0; i < n; ++i) {
            out << ggml_bf16_to_fp32(data[i]) << '\n';
        }
        थांबा;
    }
    स्थिती GGML_TYPE_I32: {
        स्थिरांक स्वतः * data = static_cast<स्थिरांक int32_t *>(tensor->data);
        साठी (माप_प्रकार i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        थांबा;
    }
    स्थिती GGML_TYPE_I64: {
        स्थिरांक स्वतः * data = static_cast<स्थिरांक int64_t *>(tensor->data);
        साठी (माप_प्रकार i = 0; i < n; ++i) {
            out << data[i] << '\n';
        }
        थांबा;
    }
    पूर्वनिर्धारित:
        out << "unsupported tensor type for text dump" << '\n';
        परत असत्य;
    }

    परत सत्य;
}

शून्य print_आदान_tensor_info(स्थिरांक std::string & name, स्थिरांक ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << '\n';
    निवड (tensor.get_element_type()) {
    स्थिती ov::element::f32: {
        जर (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::cout << *(tensor.data<दशांश>()) << '\n';
        } अन्यथा {
            माप_प्रकार rows = tensor.get_shape()[2];
            माप_प्रकार cols = tensor.get_shape()[3];
            स्थिरांक दशांश * data = tensor.data<दशांश>();
            साठी (माप_प्रकार i = 0; i < rows; ++i) {
                साठी (माप_प्रकार j = 0; j < cols; ++j) {
                    दशांश val = data[i * cols + j];
                    जर (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } अन्यथा {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << '\n';
            }
        }

        थांबा;
    }
    स्थिती ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << '\n';
        थांबा;
    स्थिती ov::element::i32:
        साठी (माप_प्रकार i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << ' ';
        }
        std::cout << '\n';
        थांबा;
    स्थिती ov::element::i64:
        साठी (माप_प्रकार i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << ' ';
        }
        std::cout << '\n';
        थांबा;
    पूर्वनिर्धारित:
        थांबा;
    }
}

शून्य print_output_tensor_info(स्थिरांक std::string & name, स्थिरांक ov::Tensor & tensor, स्थिरांक शून्य * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    स्वतः print_float_stats = [](स्थिरांक std::string & type_name, माप_प्रकार size, स्वतः get_मूल्य) {
        जर (size == 0) {
            परत;
        }

        दशांश first = get_मूल्य(0);
        दशांश min = first;
        दशांश max = first;
        द्विगुण sum = first;

        साठी (माप_प्रकार i = 1; i < size; ++i) {
            दशांश v = get_मूल्य(i);
            min = std::min(v, min);
            max = std::max(v, max);
            sum += v;
        }
        द्विगुण mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << '\n';
    };

    निवड (tensor.get_element_type()) {
    स्थिती ov::element::f32: {
        स्थिरांक दशांश * data = tensor.data<दशांश>();
        माप_प्रकार size = tensor.get_size();
        print_float_stats("[f32]", size, [data](माप_प्रकार i) { परत data[i]; });
        थांबा;
    }
    स्थिती ov::element::f16: {
        स्थिरांक ov::float16 * data = tensor.data<ov::float16>();
        माप_प्रकार size = tensor.get_size();
        print_float_stats("[f16]", size, [data](माप_प्रकार i) { परत static_cast<दशांश>(data[i]); });
        थांबा;
    }
    पूर्वनिर्धारित:
        थांबा;
    }
}

स्थिरांक ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; ++i) {
        स्वतः * op = cgraph->nodes[i];
        साठी (पूर्णांक j = 0; j < GGML_MAX_SRC; ++j) {
            स्वतः * src = op->src[j];
            जर (src == nullptr) {
                थांबा;
            }
            जर (GgmlOvDecoder::is_inp_pos(src, op)) {
                परत src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    फेका std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_tokens(ggml_cgraph * cgraph, स्थिरांक ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    पूर्णांक n_planes = 1;
    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; ++i) {
        स्वतः * op = cgraph->nodes[i];
        साठी (पूर्णांक j = 0; j < GGML_MAX_SRC; ++j) {
            जर (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                थांबा;
            }
        }
    }
    परत inp_pos->ne[0] / n_planes;
}

तार्किक get_is_prefill(ggml_cgraph * cgraph, स्थिरांक ggml_tensor * inp_pos) {
    परत get_inp_pos_n_tokens(cgraph, inp_pos) > 1;
}
