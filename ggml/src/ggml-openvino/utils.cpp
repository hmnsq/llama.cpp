#समावेश "utils.h"

#समावेश "ggml-impl.h"
#समावेश "ggml-openvino-extra.h"
#समावेश "ggml-openvino.h"
#समावेश "ggml-openvino/ggml-decoder.h"
#समावेश "ggml.h"
#समावेश "model-cache.h"
#समावेश "openvino/frontend.h"
#समावेश "openvino/input_model.h"

#समावेश <algorithm>
#समावेश <cassert>
#समावेश <cmath>
#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <cstdlib>
#समावेश <cstring>
#समावेश <fstream>
#समावेश <functional>
#समावेश <iomanip>
#समावेश <iostream>
#समावेश <memory>
#समावेश <openvino/core/any.hpp>
#समावेश <openvino/core/graph_util.hpp>
#समावेश <openvino/core/shape.hpp>
#समावेश <openvino/core/type/float16.hpp>
#समावेश <openvino/frontend/manager.hpp>
#समावेश <openvino/openvino.hpp>
#समावेश <openvino/runtime/compiled_model.hpp>
#समावेश <openvino/runtime/infer_request.hpp>
#समावेश <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#समावेश <openvino/runtime/intel_npu/properties.hpp>
#समावेश <openvino/runtime/properties.hpp>
#समावेश <openvino/runtime/tensor.hpp>
#समावेश <optional>
#समावेश <set>
#समावेश <string>
#समावेश <unordered_map>
#समावेश <vector>

namespace {
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
std::optional<ov::Tensor> try_make_kv_sliced_tensor(स्थिर std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    स्थिर std::string & name,
                                                    स्थिर ggml_tensor * ggml_tensor) {
    अटल स्थिर bool kv_slice_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    अगर (kv_slice_disabled) {
        वापस std::nullopt;
    }
    अगर (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        वापस std::nullopt;
    }
    अगर (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != nullptr) {
        वापस std::nullopt;
    }
    स्थिर स्वतः * op = ggml_decoder->get_tensor_used_op(ggml_tensor);
    अगर (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        वापस std::nullopt;
    }

    स्थिर स्वतः & compute_params = ggml_decoder->get_compute_params();
    अगर (compute_params.n_seq_active != 1 || compute_params.seq_active_start != 0) {
        वापस std::nullopt;
    }

    पूर्णांक layer;
    अगर (स्वतः layer_opt = extract_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } अन्यथा {
        वापस std::nullopt;
    }

    स्थिर bool is_swa = ggml_decoder->is_swa_layer(layer);
    अगर (is_swa) {
        वापस std::nullopt;
    }
    स्थिर पूर्णांक ctx_per_seq = ggml_decoder->get_ctx_per_seq();
    स्थिर पूर्णांक n_kv = compute_params.attention_आकार;
    अगर (ctx_per_seq <= 0 || n_kv <= 0 || n_kv >= ctx_per_seq) {
        वापस std::nullopt;
    }

    ov::Shape full_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    अगर (full_shape.size() != 4 || full_shape[0] != 1 || full_shape[1] != 1 ||
        static_cast<पूर्णांक>(full_shape[2]) != ctx_per_seq) {
        वापस std::nullopt;
    }

    ov::Shape sliced_shape = full_shape;
    sliced_shape[2] = static_cast<माप_प्रकार>(n_kv);

    अगर (ggml_openvino_बफर_is_remote(ggml_tensor)) {
        स्वतः remote_context = ggml_openvino_get_remote_context();
        स्वतः gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        वापस gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->तथ्य);
    }

    वापस ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->तथ्य);
}

अटल uint64_t ggml_openvino_model_cache_extra_cfg(bool stateful, bool recurrent) {
    स्थिर अक्षर * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    स्थिर bool manual_gqa_enabled = manual_gqa_env != nullptr ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    वापस extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_weight_names(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> names;
    क्रम (स्थिर स्वतः & name : GgmlOvDecoder::collect_weight_names(cgraph)) {
        names[name] = nullptr;
    }
    वापस names;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
अटल std::string compiled_graph_कुंजी(स्थिर ggml_cgraph * graph,
                                      स्थिर GgmlOvDecoder & decoder,
                                      स्थिर std::string & device,
                                      पूर्णांक prefill_chunk_आकार = 0,
                                      bool disk_cache = false) {
    std::string कुंजी;
    स्वतः append = [&कुंजी](स्थिर स्वतः & value) {
        कुंजी.append(reinterpret_cast<स्थिर अक्षर *>(&value), माप(value));
    };
    स्वतः append_string = [&](स्थिर std::string & value) {
        append(value.size());
        कुंजी.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_आकार);
    bool has_weight_बफर_id = false;
    std::unordered_map<स्थिर ggml_tensor *, माप_प्रकार> ids;
    std::function<व्योम(स्थिर ggml_tensor *)> visit = [&](स्थिर ggml_tensor * tensor) {
        अगर (!tensor) {
            append(माप_प्रकार(0));
            वापस;
        }
        स्वतः inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.first->second);
        अगर (!inserted.second) {
            वापस;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        स्थिर स्वतः * base = tensor->view_src ? tensor->view_src : tensor;
        स्थिर bool weight = base->बफर && base->बफर->usage == GGML_BACKEND_बफर_USAGE_WEIGHTS;
        append(weight);
        अगर (weight && !disk_cache) {
            स्थिर माप_प्रकार बफर_id = ggml_backend_openvino_बफर_get_ctx_id(base->बफर);
            has_weight_बफर_id |= बफर_id != 0;
            append(बफर_id);
            append(tensor->तथ्य);
        }
        visit(tensor->view_src);
        क्रम (स्थिर स्वतः * src : tensor->src) {
            visit(src);
        }
    };
    append(graph->n_nodes);
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; ++i) {
        visit(graph->nodes[i]);
    }
    append(graph->n_leafs);
    क्रम (पूर्णांक i = 0; i < graph->n_leafs; ++i) {
        visit(graph->leafs[i]);
    }
    क्रम (स्थिर स्वतः & निवेश : decoder.get_model_extra_inputs()) {
        append_string(निवेश.first);
        append_string(निवेश.second.type.get_type_name());
        append(निवेश.second.shape.size());
        क्रम (स्वतः dim : निवेश.second.shape) {
            append(dim);
        }
        append(निवेश.second.is_parameter);
        अगर (!निवेश.second.is_parameter) {
            append(निवेश.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    वापस disk_cache || has_weight_बफर_id ? कुंजी : std::string{};
}

अटल std::string dynamic_graph_signature(स्थिर ggml_cgraph * graph,
                                           स्थिर GgmlOvDecoder & decoder,
                                           स्थिर ModelParams & params) {
    std::string कुंजी = "dynamic-1";
    स्वतः append = [&कुंजी](स्थिर स्वतः & value) {
        कुंजी.append(reinterpret_cast<स्थिर अक्षर *>(&value), माप(value));
    };
    स्वतः append_string = [&](स्थिर std::string & value) {
        append(value.size());
        कुंजी.append(value);
    };
    graph_कुंजी graph_id(graph, true);
    append(graph_id.n_nodes);
    append(graph_id.n_leaves);
    append_string(graph_id.first_गाँठ_name);
    append_string(graph_id.last_गाँठ_name);
    क्रम (स्थिर स्वतः & name : graph_id.निवेश_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    क्रम (पूर्णांक layer : params.swa_layers) {
        append(layer);
    }
    क्रम (स्थिर स्वतः & [layer, heads] : params.n_heads_kv_per_layer) {
        append(layer);
        append(heads);
    }
    क्रम (स्थिर स्वतः & [name, निवेश] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(निवेश.type.get_type_name());
    }
    क्रम (स्थिर स्वतः & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    क्रम (स्थिर स्वतः & [name, निवेश] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(निवेश.type.get_type_name());
        append(निवेश.is_parameter);
        अगर (!निवेश.is_parameter) {
            append(निवेश.value);
        }
    }
    वापस कुंजी;
}

अटल bool compiled_model_matches_graph(स्थिर ov::CompiledModel & model,
                                         स्थिर std::vector<std::string> & inputs,
                                         स्थिर std::vector<std::string> & outputs,
                                         स्थिर GgmlOvDecoder & decoder) {
    अगर (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        वापस false;
    }
    स्थिर स्वतः & graph_inputs = decoder.get_model_inputs();
    स्थिर स्वतः & extra_inputs = decoder.get_model_extra_inputs();
    स्थिर स्वतः & graph_outputs = decoder.get_model_outputs();
    क्रम (माप_प्रकार i = 0; i < inputs.size(); ++i) {
        स्थिर स्वतः & port = model.निवेश(i);
        स्वतः graph_it = graph_inputs.find(inputs[i]);
        अगर (graph_it != graph_inputs.end()) {
            अगर (port.get_element_type() != graph_it->second.type ||
                !port.get_partial_shape().compatible(graph_it->second.shape)) {
                वापस false;
            }
            जारी;
        }
        स्वतः extra_it = extra_inputs.find(inputs[i]);
        अगर (extra_it == extra_inputs.end() || port.get_element_type() != extra_it->second.type ||
            !port.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            वापस false;
        }
    }
    क्रम (माप_प्रकार i = 0; i < outputs.size(); ++i) {
        स्वतः graph_it = graph_outputs.find(outputs[i]);
        अगर (graph_it == graph_outputs.end()) {
            अगर (outputs[i].compare(0, 8, "__debug_") != 0) {
                वापस false;
            }
            जारी;
        }
        स्थिर स्वतः & port = model.output(i);
        अगर (port.get_element_type() != GgmlOvDecoder::get_ov_type(graph_it->second) ||
            !port.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(graph_it->second)))) {
            वापस false;
        }
    }
    वापस true;
}

अटल std::string ov_profiling_csv_field(स्थिर std::string & value) {
    std::string escaped = "\"";
    क्रम (अक्षर c : value) {
        escaped += c;
        अगर (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    वापस escaped;
}

अटल व्योम dump_ov_profiling_info(स्थिर ov::InferRequest & infer_request) {
    अगर (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        वापस;
    }

    अटल std::atomic<uint64_t> inference_सूचकांक{0};
    स्थिर uint64_t सूचकांक = inference_सूचकांक.fetch_add(1);
    स्थिर std::string path = "openvino_profile_" + std::to_string(सूचकांक) + ".csv";
    std::बख_प्रवाह output(path, std::पनप::छाँटो);
    अगर (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        वापस;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t योग_real_time = 0;
    int64_t योग_cpu_time = 0;
    माप_प्रकार executed_गिनती = 0;
    क्रम (स्थिर स्वतः & info : infer_request.get_profiling_info()) {
        स्थिर अक्षर * status = "NOT_RUN";
        अगर (info.status == ov::ProfilingInfo::Status::EXECUTED) {
            status = "EXECUTED";
            योग_real_time += info.real_time.count();
            योग_cpu_time += info.cpu_time.count();
            executed_गिनती++;
        } अन्यथा अगर (info.status == ov::ProfilingInfo::Status::OPTIMIZED_OUT) {
            status = "OPTIMIZED_OUT";
        }

        output << status << ',' << info.real_time.count() << ',' << info.cpu_time.count() << ','
               << info.start_time.count() << ',' << ov_profiling_csv_field(info.node_type) << ','
               << ov_profiling_csv_field(info.गाँठ_name) << ',' << ov_profiling_csv_field(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_गिनती, योग_real_time / 1000.0, योग_cpu_time / 1000.0);
}

अटल व्योम reset_single_slot_recurrent_cache(ggml_cgraph * cgraph,
                                              स्थिर ModelParams & model_params,
                                              स्थिर ComputeParams & compute_params) {
    अगर (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        वापस;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_बफर_t> buffers;
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        स्वतः * गाँठ = cgraph->nodes[i];
        अगर (गाँठ->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(गाँठ->view_src) &&
            गाँठ->view_src->ne[1] == 1 && गाँठ->view_src->बफर != nullptr) {
            buffers.insert(गाँठ->view_src->बफर);
        }
    }
    क्रम (स्वतः * बफर : buffers) {
        ggml_backend_बफर_clear(बफर, 0);
    }
}

ov::Tensor create_ov_output_tensor(स्थिर std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   स्थिर std::shared_ptr<ov::InferRequest> & infer_request,
                                   पूर्णांक output_सूचकांक,
                                   स्थिर ggml_tensor * ggml_tensor) {
    अगर (स्वतः sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        वापस *sliced;
    }

    स्वतः output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    व्योम * output_तथ्य = ggml_tensor->तथ्य;
    अगर (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_सूचकांक).get_shape();
    } अन्यथा {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        अगर (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != nullptr &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_तथ्य = ggml_tensor->view_src->तथ्य;
        } अन्यथा {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    अगर (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_बफर_is_remote(ggml_tensor) &&
        ggml_tensor->extra != nullptr) {
        स्वतः * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        अगर (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            स्वतः * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            अगर (tensor_extra->tensor != nullptr && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                वापस *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_तथ्य);
    वापस output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_rows_to_seq_axis_2(स्थिर ov::Tensor & kv_tensor, माप_प्रकार n_heads_kv) {
    स्थिर माप_प्रकार rows = kv_tensor.get_shape()[2];
    स्थिर माप_प्रकार शीर्ष_आकार = kv_tensor.get_shape()[3] / n_heads_kv;
    स्थिर माप_प्रकार elem = kv_tensor.get_element_type().size();
    स्थिर माप_प्रकार शीर्ष_bytes = शीर्ष_आकार * elem;

    ov::Tensor out(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, rows, शीर्ष_आकार});
    स्थिर स्वतः * src = static_cast<स्थिर uint8_t *>(kv_tensor.तथ्य());
    स्वतः * dst = static_cast<uint8_t *>(out.तथ्य());
    क्रम (माप_प्रकार s = 0; s < rows; s++) {
        क्रम (माप_प्रकार h = 0; h < n_heads_kv; h++) {
            स_नकल(dst + (h * rows + s) * शीर्ष_bytes, src + (s * n_heads_kv + h) * शीर्ष_bytes, शीर्ष_bytes);
        }
    }
    वापस out;
}

ढाँचा <typename T> व्योम set_zero_diagonal(std::vector<T> & matrix, माप_प्रकार rows, माप_प्रकार cols, T zero_मूल्य = T{}) {
    क्रम (माप_प्रकार i = 0; i < rows; ++i) {
        माप_प्रकार diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = zero_मूल्य;
    }
}

ov::Tensor make_contiguous_split_निवेश_tensor(स्थिर काष्ठा ggml_tensor * ggml_tensor, स्थिर ov::Shape & निवेश_shape) {
    स्थिर माप_प्रकार element_आकार = ggml_type_आकार(ggml_tensor->type);
    स्थिर माप_प्रकार block_आकार_2 = ggml_blck_आकार(ggml_tensor->type);

    GGML_ASSERT(block_आकार_2 == 1 && "non-contiguous split inputs must be plain element types");

    स्थिर काष्ठा ggml_tensor * source_tensor = ggml_tensor->view_src != nullptr ? ggml_tensor->view_src : ggml_tensor;
    स्थिर माप_प्रकार source_offset = ggml_tensor->view_src != nullptr ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> source_तथ्य(ggml_nbytes(source_tensor));
    ggml_backend_tensor_get(source_tensor, source_तथ्य.तथ्य(), 0, source_तथ्य.size());

    ov::Tensor निवेश_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), निवेश_shape);
    स्वतः * dst = static_cast<uint8_t *>(निवेश_tensor.तथ्य());
    माप_प्रकार dst_offset = 0;

    क्रम (माप_प्रकार i3 = 0; i3 < static_cast<माप_प्रकार>(ggml_tensor->ne[3]); ++i3) {
        क्रम (माप_प्रकार i2 = 0; i2 < static_cast<माप_प्रकार>(ggml_tensor->ne[2]); ++i2) {
            क्रम (माप_प्रकार i1 = 0; i1 < static_cast<माप_प्रकार>(ggml_tensor->ne[1]); ++i1) {
                क्रम (माप_प्रकार i0 = 0; i0 < static_cast<माप_प्रकार>(ggml_tensor->ne[0]); ++i0) {
                    स्थिर माप_प्रकार src_offset = source_offset + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::स_नकल(dst + dst_offset, source_तथ्य.तथ्य() + src_offset, element_आकार);
                    dst_offset += element_आकार;
                }
            }
        }
    }

    वापस निवेश_tensor;
}

ov::Tensor convert_ggml_निवेश_to_ov(स्थिर std::shared_ptr<GgmlOvDecoder> & ggml_decoder, स्थिर std::string & name) {
    स्थिर स्वतः * ggml_tensor = ggml_decoder->get_निवेश_ggml_tensor(name);

    अगर (स्वतः sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        वापस *sliced;
    }

    अगर (ggml_tensor->extra != nullptr && !ggml_decoder->is_splited_model()) {
        स्वतः * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        अगर (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            स्वतः * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            वापस *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    स्वतः * निवेश_तथ्य = ggml_tensor->तथ्य;
    ov::Shape निवेश_shape;
    अगर (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        निवेश_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } अन्यथा {
        निवेश_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    अगर (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        वापस make_contiguous_split_निवेश_tensor(ggml_tensor, निवेश_shape);
    }

    स्वतः निवेश_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), निवेश_shape, निवेश_तथ्य);
    वापस निवेश_tensor;
}

ov::Tensor get_ov_निवेश_tensor(स्थिर std::shared_ptr<GgmlOvDecoder> & ggml_decoder, स्थिर std::string & param_name) {
    ov::Tensor निवेश_tensor;
    स्वतः extra_निवेश = ggml_decoder->get_model_extra_inputs().find(param_name);
    अगर (extra_निवेश != ggml_decoder->get_model_extra_inputs().end()) {
        निवेश_tensor = ov::Tensor(extra_निवेश->second.type, extra_निवेश->second.shape);
        *निवेश_tensor.तथ्य<int64_t>() = extra_निवेश->second.value;
    } अन्यथा {
        निवेश_tensor = convert_ggml_निवेश_to_ov(ggml_decoder, param_name);
    }
    वापस निवेश_tensor;
}

ov::Tensor get_ov_निवेश_tensor_static_decode(स्थिर std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             स्थिर std::string & param_name) {
    // NPU decoding stage
    अगर (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        वापस get_ov_निवेश_tensor(ggml_decoder, param_name);
    }
    स्थिर स्वतः * ggml_tensor = ggml_decoder->get_निवेश_ggml_tensor(param_name);
    स्थिर स्वतः * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    अगर (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        स्थिर पूर्णांक n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        निश्चित(ggml_tensor->ne[0] == n_planes);
        ov::Shape निवेश_shape = {1, 1, 1, (माप_प्रकार) n_planes};
        ov::Tensor निवेश_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), निवेश_shape);
        std::स_नकल(निवेश_tensor.तथ्य(), ggml_tensor->तथ्य, n_planes * ggml_type_आकार(ggml_tensor->type));
        वापस निवेश_tensor;
    }

    अगर (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape निवेश_shape = {1, 1, 1, 1};
        ov::Tensor निवेश_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), निवेश_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->तथ्य);
        निश्चित(ggml_tensor->ne[0] == 1);
        निश्चित(inp_out_id == 0);
        *निवेश_tensor.तथ्य<int32_t>() = inp_out_id;
        वापस निवेश_tensor;
    }

    अगर (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        माप_प्रकार context_आकार = ggml_decoder->get_ctx_आकार();
        अगर (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_तथ्य =
                pad_निवेश<ggml_fp16_t>(ggml_tensor, 1, context_आकार, GGML_FP32_TO_FP16(-अनन्त));
            ov::Tensor निवेश_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_आकार});
            std::स_नकल(निवेश_tensor.तथ्य(), padded_तथ्य.तथ्य(), padded_तथ्य.size() * माप(ggml_fp16_t));
            वापस निवेश_tensor;
        }

        std::vector<भग्न> padded_तथ्य = pad_निवेश<भग्न>(ggml_tensor, 1, context_आकार, -अनन्त);
        ov::Tensor निवेश_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_आकार});
        स्वतः * तथ्य_ptr = निवेश_tensor.तथ्य<भग्न>();
        std::copy(padded_तथ्य.begin(), padded_तथ्य.begin() + context_आकार, तथ्य_ptr);
        वापस निवेश_tensor;
    }

    वापस get_ov_निवेश_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_निवेश_tensor_static_prefill(स्थिर std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              स्थिर std::string & param_name,
                                              पूर्णांक chunk_सूचकांक) {
    // NPU prompt processing stage
    स्थिर माप_प्रकार निवेश_len = ggml_decoder->get_निवेश_len();
    स्थिर माप_प्रकार chunk_आकार = ggml_decoder->m_prefill_chunk_आकार;
    स्थिर माप_प्रकार chunk_valid_आकार = std::min(chunk_आकार, निवेश_len - chunk_सूचकांक * chunk_आकार);
    स्थिर माप_प्रकार chunk_pad_आकार = chunk_आकार - chunk_valid_आकार;

    अगर (param_name == "chunk_valid_len") {
        ov::Tensor निवेश_tensor(ov::element::i64, ov::Shape{1});
        *निवेश_tensor.तथ्य<int64_t>() = (int64_t) chunk_valid_आकार;
        वापस निवेश_tensor;
    }
    अगर (chunk_सूचकांक > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor निवेश_tensor(ov::element::i64, ov::Shape{1});
        *निवेश_tensor.तथ्य<int64_t>() = 0;
        वापस निवेश_tensor;
    }
    अगर (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        वापस get_ov_निवेश_tensor(ggml_decoder, param_name);
    }
    स्थिर स्वतः * ggml_tensor = ggml_decoder->get_निवेश_ggml_tensor(param_name);
    स्थिर स्वतः * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    अगर (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        स्थिर पूर्णांक n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        स्थिर माप_प्रकार element_आकार = ggml_type_आकार(ggml_tensor->type);
        ov::Shape निवेश_shape = {1, 1, 1, (माप_प्रकार) n_planes * chunk_आकार};
        ov::Tensor निवेश_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), निवेश_shape);
        क्रम (पूर्णांक p = 0; p < n_planes; p++) {
            स्थिर अक्षर * src =
                (स्थिर अक्षर *) ggml_tensor->तथ्य + (p * निवेश_len + chunk_सूचकांक * chunk_आकार) * element_आकार;
            अक्षर * dst = (अक्षर *) निवेश_tensor.तथ्य() + p * chunk_आकार * element_आकार;
            std::स_नकल(dst, src, chunk_valid_आकार * element_आकार);
            अगर (chunk_pad_आकार > 0) {
                अगर (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t last_मूल्य = *((स्थिर int32_t *) src + chunk_valid_आकार - 1);
                    int32_t * out = (int32_t *) dst;
                    std::fill(out + chunk_valid_आकार, out + chunk_आकार, last_मूल्य + 1);
                } अन्यथा अगर (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t last_मूल्य = *((स्थिर int64_t *) src + chunk_valid_आकार - 1);
                    int64_t * out = (int64_t *) dst;
                    std::fill(out + chunk_valid_आकार, out + chunk_आकार, last_मूल्य + 1);
                } अन्यथा {
                    फेंको std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        वापस निवेश_tensor;
    }

    अगर (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape निवेश_shape = {1, 1, 1, chunk_आकार};
        ov::Tensor निवेश_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), निवेश_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        माप_प्रकार element_आकार = ggml_type_आकार(ggml_tensor->type);
        व्योम * निवेश_तथ्य = (अक्षर *) ggml_tensor->तथ्य + chunk_सूचकांक * chunk_आकार * element_आकार;
        std::स_नकल(निवेश_tensor.तथ्य(), निवेश_तथ्य, chunk_valid_आकार * element_आकार);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        अगर (chunk_pad_आकार > 0) {
            अगर (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_मूल्य =
                    *((int32_t *) ggml_tensor->तथ्य + (chunk_सूचकांक * chunk_आकार + chunk_valid_आकार - 1));
                int32_t * output_तथ्य = निवेश_tensor.तथ्य<int32_t>();
                std::fill(output_तथ्य + chunk_valid_आकार, output_तथ्य + chunk_आकार, last_मूल्य + 1);
            } अन्यथा अगर (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_मूल्य =
                    *((int64_t *) ggml_tensor->तथ्य + (chunk_सूचकांक * chunk_आकार + chunk_valid_आकार - 1));
                int64_t * output_तथ्य = निवेश_tensor.तथ्य<int64_t>();
                std::fill(output_तथ्य + chunk_valid_आकार, output_तथ्य + chunk_आकार, last_मूल्य + 1);
            } अन्यथा {
                फेंको std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        वापस निवेश_tensor;
    }

    अगर (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        माप_प्रकार output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape निवेश_shape = {1, 1, 1, output_len};
        ov::Tensor निवेश_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), निवेश_shape);
        अगर (ggml_tensor->ne[0] == 0) {
            *निवेश_tensor.तथ्य<int32_t>() = 0;
        } अन्यथा {
            स्वतः * तथ्य_addr = निवेश_tensor.तथ्य<int32_t>();
            क्रम (माप_प्रकार i = 0; i < output_len; i++) {
                तथ्य_addr[i] = ((int32_t *) ggml_tensor->तथ्य)[i] % chunk_आकार;
            }
        }
        वापस निवेश_tensor;
    }

    अगर (GgmlOvDecoder::is_inp_scale_rows(ggml_tensor, op)) {
        ov::Tensor निवेश_tensor(ov::element::f32, ov::Shape{1, 1, chunk_आकार, 1});
        स्वतः * dst = निवेश_tensor.तथ्य<भग्न>();
        स्थिर स्वतः * src = static_cast<स्थिर भग्न *>(ggml_tensor->तथ्य) + chunk_सूचकांक * chunk_आकार;
        std::copy(src, src + chunk_valid_आकार, dst);
        std::fill(dst + chunk_valid_आकार, dst + chunk_आकार, 1.0f);
        वापस निवेश_tensor;
    }

    अगर (GgmlOvDecoder::is_inp_mean(ggml_tensor, op)) {
        स्थिर माप_प्रकार n_seqs = ggml_tensor->ne[1];
        स्थिर माप_प्रकार src_stride = ggml_tensor->ne[0];
        स्थिर माप_प्रकार copy_len = std::min<माप_प्रकार>(chunk_valid_आकार, src_stride - chunk_सूचकांक * chunk_आकार);
        ov::Tensor निवेश_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_आकार});
        स्वतः * dst = निवेश_tensor.तथ्य<भग्न>();
        std::fill(dst, dst + n_seqs * chunk_आकार, 0.0f);
        स्थिर स्वतः * src = static_cast<स्थिर भग्न *>(ggml_tensor->तथ्य) + chunk_सूचकांक * chunk_आकार;
        क्रम (माप_प्रकार s = 0; s < n_seqs; s++) {
            std::स_नकल(dst + s * chunk_आकार, src + s * src_stride, copy_len * माप(भग्न));
        }
        वापस निवेश_tensor;
    }

    अगर (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        माप_प्रकार cols = ggml_tensor->ne[0];
        माप_प्रकार rows = ggml_tensor->ne[1];
        माप_प्रकार chunk_valid_rows = std::min(chunk_आकार, rows - chunk_सूचकांक * chunk_आकार);
        माप_प्रकार context_आकार = ggml_decoder->get_ctx_आकार();
        अगर (ggml_tensor->type == GGML_TYPE_F16) {
            स्थिर स्वतः * ggml_तथ्य =
                static_cast<स्थिर ggml_fp16_t *>(ggml_tensor->तथ्य) + chunk_सूचकांक * chunk_आकार * cols;
            std::vector<ggml_fp16_t> padded_तथ्य = pad_निवेश<ggml_fp16_t>(ggml_तथ्य, chunk_valid_rows, cols, chunk_आकार,
                                                                          context_आकार, GGML_FP32_TO_FP16(-अनन्त));
            set_zero_diagonal(padded_तथ्य, chunk_आकार, context_आकार, GGML_FP32_TO_FP16(0.0f));
            ov::Tensor निवेश_tensor(ov::element::f16, ov::Shape{1, 1, chunk_आकार, context_आकार});
            std::स_नकल(निवेश_tensor.तथ्य(), padded_तथ्य.तथ्य(), padded_तथ्य.size() * माप(ggml_fp16_t));
            वापस निवेश_tensor;
        }

        स्थिर स्वतः * ggml_तथ्य = static_cast<स्थिर भग्न *>(ggml_tensor->तथ्य) + chunk_सूचकांक * chunk_आकार * cols;
        std::vector<भग्न> padded_तथ्य =
            pad_निवेश<भग्न>(ggml_तथ्य, chunk_valid_rows, cols, chunk_आकार, context_आकार, -अनन्त);
        set_zero_diagonal(padded_तथ्य, chunk_आकार, context_आकार);
        ov::Tensor निवेश_tensor(ov::element::f32, ov::Shape{1, 1, chunk_आकार, context_आकार});
        स्वतः * तथ्य_ptr = निवेश_tensor.तथ्य<भग्न>();
        std::copy(padded_तथ्य.begin(), padded_तथ्य.begin() + chunk_आकार * context_आकार, तथ्य_ptr);
        वापस निवेश_tensor;
    }

    वापस get_ov_निवेश_tensor(ggml_decoder, param_name);
}

क्रमागत ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               स्थिर std::string & device,
                               स्थिर ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    अगर (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        वापस GGML_STATUS_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_lock(cache.mutex);
    bool naive = true;
    स्वतः model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    स्वतः decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    स्वतः निवेश_model = std::make_shared<ov::frontend::ggml::निवेश_Model>(decoder);
    स्वतः model = ov::frontend::ggml::FrontEnd::convert(निवेश_model, naive);
    अगर (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    स्वतः remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    अगर (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    } अन्यथा {
        compile_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    }
    अगर (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } अन्यथा {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> निवेश_names;
    std::vector<std::string> output_names;
    क्रम (स्थिर स्वतः & param : model->get_parameters()) {
        निवेश_names.push_back(param->get_friendly_name());
    }
    क्रम (स्थिर स्वतः & परिणाम : model->get_results()) {
        output_names.push_back(परिणाम->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    निवेश_model.reset();
    decoder->clear_model_weights();
    model_weights.सफा();
    compile_lock.unlock();

    क्रम (माप_प्रकार i = 0; i < निवेश_names.size(); i++) {
        स्थिर स्वतः & param_name = निवेश_names[i];
        स्वतः निवेश_tensor = get_ov_निवेश_tensor(decoder, param_name);
        infer_request->set_निवेश_tensor(i, निवेश_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    क्रम (माप_प्रकार i = 0; i < output_names.size(); i++) {
        स्वतः output_tensor = infer_request->get_output_tensor(i);
        स्थिर स्वतः & model_outputs = decoder->get_model_outputs();
        स्वतः model_output_it = model_outputs.find(output_names[i]);
        अगर (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            अगर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_names[i], output_tensor, output_tensor.तथ्य());
            }
            जारी;
        }
        स्वतः * ggml_tensor = model_output_it->second;
        std::स_नकल(ggml_tensor->तथ्य, output_tensor.तथ्य(), output_tensor.get_byte_आकार());
    }
    वापस GGML_STATUS_SUCCESS;
}

क्रमागत ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, स्थिर std::shared_ptr<ov_runtime_context> & r_ctx) {
    स्वतः & core = ov_singleton_core();
    स्थिर स्वतः & config = ggml_openvino_get_compile_config();
    स्थिर स्वतः & device = r_ctx->device;
    अटल स्वतः is_static = false;

    अटल स्थिर bool cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    स्वतः start_time = ggml_time_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    स्थिर int64_t cache_कुंजी_start_time = ggml_time_us();
    graph_कुंजी कुंजी(cgraph);
    स्थिर int64_t cache_कुंजी_compute_time = ggml_time_us() - cache_कुंजी_start_time;
    int64_t cache_lookup_time = 0;
    bool कुंजी_seen = false;
    अगर (!cache_disabled) {
        स्थिर int64_t cache_lookup_start_time = ggml_time_us();
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        कुंजी_seen = r_ctx->decoder_cache.find(कुंजी) != r_ctx->decoder_cache.end();
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;
    }

    स्थिर int64_t model_split_check_start_time = ggml_time_us();
    bool model_is_splitted = कुंजी_seen ? false : is_model_splitted(cgraph);
    स्थिर int64_t model_split_check_time = ggml_time_us() - model_split_check_start_time;

    अगर (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        वापस GGML_STATUS_FAILED;
    }
    अगर (is_naive(cgraph)) {
        अगर (!model_is_splitted) {
            वापस naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    स्थिर int64_t graph_param_compute_start_time = ggml_time_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    स्थिर int64_t graph_param_compute_time = ggml_time_us() - graph_param_compute_start_time;

    स्थिर bool cache_enabled = !model_is_splitted && !cache_disabled;
    स्थिर bool stateful_recurrent = r_ctx->stateful && m_params.state_आकार >= 0;
    स्थिर bool stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    अगर (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_active > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        वापस GGML_STATUS_FAILED;
    }
    अगर (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        वापस GGML_STATUS_FAILED;
    }
    bool cache_hit = false;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        स्थिर int64_t cache_lookup_start_time = ggml_time_us();
        अगर (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            स्वतः it = r_ctx->decoder_cache.find(कुंजी);
            cache_hit = it != r_ctx->decoder_cache.end();
            अगर (cache_hit) {
                entry = it->second;
            } अन्यथा {
                r_ctx->clear_caches_locked();
                स्वतः mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[कुंजी] = entry;
            }
        } अन्यथा {
            स्वतः mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_hit = false;
        }
        cache_lookup_time += ggml_time_us() - cache_lookup_start_time;

        std::lock_guard<std::mutex> lock(*(entry->mutex));
        cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(कुंजी) != 0;

        अगर (cache_hit) {
            ggml_decoder = entry->ptr;
            old_m_params = ggml_decoder->get_model_params();
            अगर (!ggml_decoder->is_splited_model()) {
                cache_hit = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_निवेश_names;
        std::vector<std::string> ov_output_names;

        अगर (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            अगर (old_m_params.kv_बफर_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(कुंजी);
                ov_निवेश_names = r_ctx->ov_निवेश_names_cache.at(कुंजी);
                ov_output_names = r_ctx->ov_output_names_cache.at(कुंजी);
            }

            अगर (stateful_kv_only) {
                स्थिर स्वतः * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_तथ्य = (int32_t *) inp_pos->तथ्य;
                स्वतः pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                अगर (pos_तथ्य[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_आकार = pos_shape[3];
                } अन्यथा अगर (r_ctx->stateful_kv_आकार == static_cast<माप_प्रकार>(pos_तथ्य[0])) {
                    r_ctx->stateful_kv_आकार += pos_shape[3];
                } अन्यथा {
                    स्थिर माप_प्रकार pos_begin = static_cast<माप_प्रकार>(pos_तथ्य[0]);
                    स्थिर bool refill = pos_begin > r_ctx->stateful_kv_आकार;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    अगर (refill && !ggml_decoder->get_model_params().swa_layers.empty()) {
                        स्थिर पूर्णांक n_swa = ggml_decoder->get_compute_params().swa_window;
                        अगर (n_swa < 0 || static_cast<माप_प्रकार>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_आकार, n_swa);
                            वापस GGML_STATUS_FAILED;
                        }
                    }

                    स्थिर bool relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    स्वतः states = infer_request->query_state();
                    क्रम (स्वतः state : states) {
                        स्वतः state_tensor = state.get_state();
                        स्वतः state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        अगर (स्वतः it = r_ctx->kv_state_निवेश_name_map.find(state.get_name());
                            it != r_ctx->kv_state_निवेश_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        पूर्णांक n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        अगर (स्वतः layer = extract_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        स्थिर bool relayout_this_state = relayout_enabled;
                        स्थिर माप_प्रकार seq_axis = relayout_this_state ? 2 : 1;
                        स्थिर माप_प्रकार शीर्ष_axis = seq_axis == 2 ? 1 : 2;

                        अगर (refill) {
                            अगर (state_name.empty()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                वापस GGML_STATUS_FAILED;
                            }
                            स्वतः kv_tensor = get_ov_निवेश_tensor(ggml_decoder, state_name);
                            अगर (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_rows_to_seq_axis_2(kv_tensor, (माप_प्रकार) n_heads_kv);
                            } अन्यथा {
                                ov::Shape refill_shape(4);
                                refill_shape[0] = state_tensor_shape[0];
                                refill_shape[seq_axis] = kv_tensor.get_shape()[2];
                                refill_shape[शीर्ष_axis] = state_tensor_shape[शीर्ष_axis];
                                refill_shape[3] = state_tensor_shape[3];
                                kv_tensor.set_shape(refill_shape);
                                state_tensor = kv_tensor;
                            }
                            state_tensor_shape = state_tensor.get_shape();
                        }
                        // Only ever shrink to a prefix the source really has. Slicing past it used to
                        // surface as a bare ov::Exception from the ROI constructor.
                        अगर (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            वापस GGML_STATUS_FAILED;
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end(state_tensor_shape.begin(), state_tensor_shape.end());
                        end[seq_axis] = pos_begin;
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.set_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_आकार = pos_begin + pos_shape[3];
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
            स्थिर std::string model_cache_dir = ggml_openvino_model_cache_dir();
            स्थिर std::string exact_कुंजी = cache_enabled ? compiled_graph_कुंजी(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_path, manifest_path;
            अगर (!model_cache_dir.empty() && !model_is_splitted) {
                स्थिर uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/true, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_graph_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/true, m_params.rope_params, 16, extra_cfg,
                                                    compiled_graph_कुंजी(cgraph, *ggml_decoder, device, 0, true));
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            std::string shared_कुंजी =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_कुंजी) : "";
            ov::CompiledModel shared_model;
            bool imported = false;
            स्वतः shared_it = shared_cache->graphs.find(shared_कुंजी);
            अगर (!shared_कुंजी.empty() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_graph(shared_it->second.decode, shared_it->second.निवेश_names,
                                                               shared_it->second.output_names, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_निवेश_names = shared_it->second.निवेश_names;
                ov_output_names = shared_it->second.output_names;
                imported = true;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } अन्यथा अगर (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_कुंजी = exact_कुंजी;
                shared_it = shared_cache->graphs.find(shared_कुंजी);
                अगर (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_निवेश_names = shared_it->second.निवेश_names;
                    ov_output_names = shared_it->second.output_names;
                    imported = true;
                }
                model_fp = exact_fp;
                blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            }
            अगर (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(कुंजी);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            अगर (!model_cache_dir.empty()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
            }
            अगर (!imported && !model_cache_dir.empty() && !model_is_splitted) {
                स्थिर uint64_t preferred_fp = model_fp;
                bool dynamic_incompatible = preferred_fp == exact_fp;
                क्रम (पूर्णांक candidate = 0; candidate < 2; ++candidate) {
                    अगर (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        अवरोध;
                    }
                    स्थिर uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    स्थिर std::string candidate_blob =
                        ggml_openvino_model_cache_blob_path(model_cache_dir, candidate_fp);
                    स्थिर std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_path(model_cache_dir, candidate_fp);
                    std::अख_प्रवाह blob_in(candidate_blob, std::पनप::द्विग);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    अगर (!blob_in.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        जारी;
                    }
                    स्थिर int64_t import_start = ggml_time_us();
                    प्रयास {
                        ov::CompiledModel cm;
                        स्वतः remote_context = ggml_openvino_get_remote_context();
                        अगर (remote_context.has_value()) {
                            cm = core.import_model(blob_in, remote_context.value(), mc_config);
                        } अन्यथा {
                            cm = core.import_model(blob_in, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        अगर (!saved_inputs.empty() || !saved_outputs.empty()) {
                            ov_निवेश_names = std::move(saved_inputs);
                            ov_output_names = std::move(saved_outputs);
                        } अन्यथा {
                            क्रम (स्थिर स्वतः & p : cm.inputs()) {
                                ov_निवेश_names.push_back(p.get_गाँठ()->get_friendly_name());
                            }
                            क्रम (स्थिर स्वतः & o : cm.outputs()) {
                                ov_output_names.push_back(o.get_गाँठ()->get_friendly_name());
                            }
                        }
                        अगर (!compiled_model_matches_graph(cm, ov_निवेश_names, ov_output_names, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_निवेश_names.सफा();
                            ov_output_names.सफा();
                            जारी;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        entry->ptr = ggml_decoder;
                        imported = true;
                        model_fp = candidate_fp;
                        blob_path = candidate_blob;
                        manifest_path = candidate_manifest;
                        अगर (candidate_fp == exact_fp) {
                            shared_कुंजी = exact_कुंजी;
                            shared_it = shared_cache->graphs.find(shared_कुंजी);
                        }
                        अगर (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_time_us() - import_start) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } पकड़ो (स्थिर std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = false;
                        ov_निवेश_names.सफा();
                        ov_output_names.सफा();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                अगर (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_कुंजी = exact_कुंजी;
                    shared_it = shared_cache->graphs.find(shared_कुंजी);
                    blob_path = ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
                    manifest_path = ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            अगर (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_path.c_str());
                वापस GGML_STATUS_FAILED;
            }
            अगर (!imported && ggml_openvino_weight_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                वापस GGML_STATUS_FAILED;
            }
            अगर (imported) {
                decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            } अन्यथा {
                स्वतः model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_end_time = ggml_time_us();

                स्वतः निवेश_model = std::make_shared<ov::frontend::ggml::निवेश_Model>(ggml_decoder);
                model = ov::frontend::ggml::FrontEnd::convert(निवेश_model);
                ggml_decoder->clear_model_weights();
                conversion_end_time = ggml_time_us();

                अगर (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    अक्षर timestamped_filename[64];
                    स्वतः timestamp = (दीर्घ दीर्घ) ggml_time_us();
                    snprintf(timestamped_filename, माप(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                स्थिर ov::AnyMap & compile_config = model_cache_dir.empty() ? config : mc_config;
                ov::CompiledModel compiled_model;
                स्वतः remote_context = ggml_openvino_get_remote_context();
                अगर (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } अन्यथा {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_end_time = ggml_time_us();

                क्रम (स्थिर स्वतः & ov_param : model->get_parameters()) {
                    ov_निवेश_names.push_back(ov_param->get_friendly_name());
                }
                क्रम (स्थिर स्वतः & ov_output : model->get_results()) {
                    ov_output_names.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                अगर (!model_cache_dir.empty() && !model_is_splitted && model_fp != 0) {
                    प्रयास {
                        स्थिर std::string blob_tmp = ggml_openvino_model_cache_अस्थायी_path(blob_path);
                        स्थिर std::string manifest_tmp = ggml_openvino_model_cache_अस्थायी_path(manifest_path);
                        अगर (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_निवेश_names,
                                                                     ov_output_names)) {
                            std::बख_प्रवाह blob_out(blob_tmp, std::पनप::द्विग | std::पनप::छाँटो);
                            अगर (blob_out.is_open()) {
                                compiled_model.export_model(blob_out);
                                blob_out.बंद();
                                अगर (blob_out.अच्छा()) {
                                    अगर (std::नाम(blob_tmp.c_str(), blob_path.c_str()) == 0 &&
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
                    } पकड़ो (स्थिर std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                entry->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            entry->ptr = ggml_decoder;
            अगर (!shared_कुंजी.empty() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_कुंजी,
                                             ov_compiled_graph{shared_model, {}, ov_निवेश_names, ov_output_names});
            }
            अगर (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[कुंजी] = infer_request;
                r_ctx->ov_निवेश_names_cache[कुंजी] = ov_निवेश_names;
                r_ctx->ov_output_names_cache[कुंजी] = ov_output_names;
            }

            अगर (stateful_kv_only && cache_enabled) {
                स्थिर स्वतः * inp_pos = get_inp_pos_tensor(cgraph);
                स्वतः pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                स्थिर int32_t pos_begin = ((int32_t *) inp_pos->तथ्य)[0];
                अगर (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    वापस GGML_STATUS_FAILED;
                }
                r_ctx->stateful_kv_आकार = pos_shape[3];
                स्थिर स्वतः kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                क्रम (स्थिर स्वतः & pair : kv_param_res_names) {
                    r_ctx->kv_state_निवेश_name_map[pair.first + pair.second] = pair.first;
                }
            }
        }

        अगर (stateful_recurrent) {
            स्थिर स्वतः * inp_pos = get_inp_pos_tensor(cgraph);
            स्थिर int32_t pos_begin = static_cast<स्थिर int32_t *>(inp_pos->तथ्य)[0];
            अगर (pos_begin == 0) {
                infer_request->reset_state();
            } अन्यथा अगर (!cache_hit || old_m_params.kv_बफर_changed(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<माप_प्रकार>(pos_begin) != r_ctx->stateful_kv_आकार) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                वापस GGML_STATUS_FAILED;
            }
        } अन्यथा {
            reset_single_slot_recurrent_cache(cgraph, m_params, c_params);
        }

        क्रम (माप_प्रकार i = 0; i < ov_निवेश_names.size(); i++) {
            स्थिर स्वतः & param_name = ov_निवेश_names[i];
            स्वतः निवेश_tensor = get_ov_निवेश_tensor(ggml_decoder, param_name);
            infer_request->set_निवेश_tensor(i, निवेश_tensor);

            अगर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_निवेश_tensor_info(param_name, निवेश_tensor);
            }
        }

        क्रम (माप_प्रकार i = 0; i < ov_output_names.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            स्थिर स्वतः & model_outputs = ggml_decoder->get_model_outputs();
            स्वतः model_output_it = model_outputs.find(ov_output_names[i]);
            अगर (model_output_it == model_outputs.end()) {
                जारी;
            }
            स्वतः * ggml_tensor = model_output_it->second;
            अगर (ggml_nbytes(ggml_tensor) == 0) {
                जारी;
            }
            स्वतः output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        अगर (stateful_recurrent) {
            स्थिर स्वतः * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_आकार =
                static_cast<स्थिर int32_t *>(inp_pos->तथ्य)[0] + get_inp_pos_n_tokens(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        अगर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            क्रम (माप_प्रकार i = 0; i < ov_output_names.size(); i++) {
                स्थिर स्वतः output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.तथ्य());
            }
        }

        अगर (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_कुंजी_compute_time / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_split_check_time / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", graph_param_compute_time / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_time / 1000.0);
            अगर (!cache_hit) {
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
    अगर (cache_hit && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_lock(r_ctx->compiled_cache->mutex);
        अगर (!ggml_openvino_weight_buffers_released()) {
            ggml_openvino_release_weight_buffers();
        }
    }

    वापस GGML_STATUS_SUCCESS;
}

ov::AnyMap without_npuw(स्थिर ov::AnyMap & config) {
    ov::AnyMap out;
    क्रम (स्थिर स्वतः & kv : config) {
        अगर (kv.first.rfind("NPUW", 0) == 0 || kv.first == "NPU_USE_NPUW") {
            जारी;
        }
        out.insert(kv);
    }
    वापस out;
}

क्रमागत ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, स्थिर std::shared_ptr<ov_runtime_context> & r_ctx) {
    स्वतः & core = ov_singleton_core();

    स्वतः get_prefill_chunk_आकार = [] {
        अटल स्थिर पूर्णांक chunk_आकार = []() {
            पूर्णांक env_prefill_chunk_आकार = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            वापस env_prefill_chunk_आकार > 0 ? env_prefill_chunk_आकार : 256;
        }();
        वापस chunk_आकार;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    अटल std::string device = ggml_openvino_get_device_name();
    अटल स्वतः is_static = true;
    अटल स्वतः stateful = false;

    स्वतः prefill_chunk_आकार = get_prefill_chunk_आकार();
    स्थिर स्वतः & config = ggml_openvino_get_compile_config();

    अगर (is_naive(cgraph)) {
        वापस naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    स्वतः start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    स्थिर स्वतः * inp_pos = get_inp_pos_tensor(cgraph);
    स्थिर bool no_kv_cache = m_params.is_cacheless_attn;
    स्थिर स्वतः is_prefill = no_kv_cache ? true : get_is_prefill(cgraph, inp_pos);
    स्थिर ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    अगर (m_params.n_heads_kv == -1) {
        prefill_chunk_आकार = inp_pos->ne[0];
    }
    graph_कुंजी कुंजी(cgraph);
    अटल स्थिर bool cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    bool cache_hit = false;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;
    int64_t ov_raw_infer_start;
    int64_t ov_raw_infer_योग = 0;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    अगर (cache_enabled) {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        स्वतः it = r_ctx->decoder_cache.find(कुंजी);
        cache_hit = it != r_ctx->decoder_cache.end();
        अगर (cache_hit) {
            entry = it->second;
        } अन्यथा {
            r_ctx->clear_caches_locked();
            स्वतः mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[कुंजी] = entry;
        }
    } अन्यथा {
        स्वतः mutex = std::make_shared<std::mutex>();
        entry = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_hit = false;
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));
    cache_hit = cache_hit && entry->ptr && r_ctx->infer_request_cache.count(कुंजी) != 0 &&
                r_ctx->infer_request_cache_prefill.count(कुंजी) != 0;

    अगर (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_निवेश_names_local;
    std::vector<std::string> ov_output_names_local;

    अगर (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        अगर (old_m_params.kv_बफर_changed(m_params) || !ggml_decoder->is_bound_to(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(कुंजी) : r_ctx->infer_request_cache.at(कुंजी);
            ov_निवेश_names_local = r_ctx->ov_निवेश_names_cache.at(कुंजी);
            ov_output_names_local = r_ctx->ov_output_names_cache.at(कुंजी);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } अन्यथा {
        अगर (cache_enabled) {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(कुंजी);
            r_ctx->infer_request_cache_prefill.erase(कुंजी);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        स्वतः shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_lock(shared_cache->mutex);
        स्वतः weight_names = get_weight_names(cgraph);
        स्वतः local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, weight_names, is_static,
                                                             stateful, false, is_prefill, prefill_chunk_आकार);
        स्थिर std::string shared_कुंजी =
            cache_enabled ? compiled_graph_कुंजी(cgraph, *local_decoder, device, prefill_chunk_आकार) : "";
        स्वतः shared_it = shared_cache->graphs.find(shared_कुंजी);
        अगर (!shared_कुंजी.empty() && shared_it != shared_cache->graphs.end()) {
            स्वतः & compiled = shared_it->second;
            स्वतः prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            स्वतः decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            entry->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_निवेश_names_local = compiled.निवेश_names;
            ov_output_names_local = compiled.output_names;
            r_ctx->infer_request_cache_prefill[कुंजी] = prefill_request;
            r_ctx->infer_request_cache[कुंजी] = decode_request;
            r_ctx->ov_निवेश_names_cache[कुंजी] = ov_निवेश_names_local;
            r_ctx->ov_output_names_cache[कुंजी] = ov_output_names_local;
            decoder_end_time = conversion_end_time = compile_end_time = ggml_time_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } अन्यथा {
            std::shared_ptr<ov::Model> model;
            स्वतः model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

            स्वतः ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, false, true, prefill_chunk_आकार);
            स्वतः ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, false, false, prefill_chunk_आकार);
            decoder_end_time = ggml_time_us();

            स्थिर bool dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            स्थिर स्वतः dump_ir_timestamp = static_cast<दीर्घ दीर्घ>(ggml_time_us());

            स्वतः build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          स्थिर std::shared_ptr<GgmlOvDecoder> & decoder, स्थिर अक्षर * tag,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_end_time, int64_t & local_compile_end_time) {
                स्वतः निवेश_model = std::make_shared<ov::frontend::ggml::निवेश_Model>(decoder);
                model = ov::frontend::ggml::FrontEnd::convert(निवेश_model);
                decoder->clear_model_weights();
                local_conversion_end_time = ggml_time_us();

                अगर (dump_ir) {
                    अक्षर timestamped_filename[64];
                    snprintf(timestamped_filename, माप(timestamped_filename), "model_%s_%lld.xml", tag,
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
            अगर (no_kv_cache) {
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

            क्रम (स्थिर स्वतः & ov_param : model->get_parameters()) {
                ov_निवेश_names_local.push_back(ov_param->get_friendly_name());
            }
            क्रम (स्थिर स्वतः & ov_output : model->get_results()) {
                ov_output_names_local.push_back(ov_output->get_friendly_name());
            }

            अगर (!shared_कुंजी.empty()) {
                shared_cache->graphs.emplace(
                    shared_कुंजी, ov_compiled_graph{compiled_model_decode, compiled_model_prefill, ov_निवेश_names_local,
                                                  ov_output_names_local});
            }

            अगर (cache_enabled) {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[कुंजी] = infer_request_prefill;
                r_ctx->infer_request_cache[कुंजी] = infer_request_decode;
                r_ctx->ov_निवेश_names_cache[कुंजी] = ov_निवेश_names_local;
                r_ctx->ov_output_names_cache[कुंजी] = ov_output_names_local;
            }
        }
    }

    reset_single_slot_recurrent_cache(cgraph, m_params, c_params);

    अगर (is_prefill) {
        स्वतः inp_len = get_inp_pos_n_tokens(cgraph, inp_pos);
        क्रम (पूर्णांक chunk_सूचकांक = 0; chunk_सूचकांक * prefill_chunk_आकार < inp_len; chunk_सूचकांक++) {
            क्रम (माप_प्रकार i = 0; i < ov_निवेश_names_local.size(); i++) {
                स्थिर स्वतः & param_name = ov_निवेश_names_local[i];
                स्वतः निवेश_tensor = get_ov_निवेश_tensor_static_prefill(ggml_decoder, param_name, chunk_सूचकांक);
                infer_request->set_निवेश_tensor(i, निवेश_tensor);

                अगर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    स्थिर स्वतः निवेश_tensor = infer_request->get_निवेश_tensor(i);
                    print_निवेश_tensor_info(param_name, निवेश_tensor);
                }
            }

            क्रम (माप_प्रकार i = 0; i < ov_output_names_local.size(); i++) {
                स्थिर स्वतः & model_outputs = ggml_decoder->get_model_outputs();
                स्वतः model_output_it = model_outputs.find(ov_output_names_local[i]);
                अगर (model_output_it == model_outputs.end()) {
                    जारी;
                }
                स्वतः * ggml_tensor = model_output_it->second;
                अगर (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    जारी;
                }
                स्वतः output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            ov_raw_infer_start = ggml_time_us();
            infer_request->infer();
            ov_raw_infer_योग += ggml_time_us() - ov_raw_infer_start;

            अगर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                क्रम (माप_प्रकार i = 0; i < ov_output_names_local.size(); i++) {
                    स्थिर स्वतः output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.तथ्य());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } अन्यथा {
        क्रम (माप_प्रकार i = 0; i < ov_निवेश_names_local.size(); i++) {
            स्थिर स्वतः & param_name = ov_निवेश_names_local[i];
            स्वतः निवेश_tensor = get_ov_निवेश_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_निवेश_tensor(i, निवेश_tensor);

            अगर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                स्थिर स्वतः निवेश_tensor = infer_request->get_निवेश_tensor(i);
                print_निवेश_tensor_info(param_name, निवेश_tensor);
            }
        }

        क्रम (माप_प्रकार i = 0; i < ov_output_names_local.size(); i++) {
            स्थिर स्वतः & model_outputs = ggml_decoder->get_model_outputs();
            स्वतः model_output_it = model_outputs.find(ov_output_names_local[i]);
            अगर (model_output_it == model_outputs.end()) {
                जारी;
            }
            स्वतः * ggml_tensor = model_output_it->second;
            अगर (ggml_nbytes(ggml_tensor) == 0) {
                जारी;
            }
            स्वतः output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        ov_raw_infer_start = ggml_time_us();
        infer_request->infer();
        infer_end_time = ggml_time_us();
        ov_raw_infer_योग = infer_end_time - ov_raw_infer_start;

        अगर (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            क्रम (माप_प्रकार i = 0; i < ov_output_names_local.size(); i++) {
                स्थिर स्वतः output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.तथ्य());
            }
        }
    }

    अगर (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_end_time - start_time) / 1000.0);
        अगर (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_end_time - decoder_end_time) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_end_time - conversion_end_time) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_end_time - compile_end_time) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_योग / 1000.0);
    }

    वापस GGML_STATUS_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
अटल ggml_cgraph * get_compute_graph(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    स्वतः is_skipped = [](स्थिर ggml_tensor * गाँठ) {
        वापस गाँठ->op != GGML_OP_NONE && !(गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    अगर (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        वापस cgraph;
    }
    स्वतः & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.सफा();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](स्थिर ggml_tensor * गाँठ) { वापस !is_skipped(गाँठ); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.तथ्य();
    compute.graph.n_nodes = (पूर्णांक) compute.nodes.size();
    वापस &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
क्रमागत ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    प्रयास {
        अगर (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        स्थिर स्वतः is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_lock(r_ctx->execution_mutex);
        cgraph = get_compute_graph(cgraph, *r_ctx);

        वापस is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } पकड़ो (स्थिर ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        वापस GGML_STATUS_FAILED;
    } पकड़ो (स्थिर std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        वापस GGML_STATUS_FAILED;
    } पकड़ो (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        वापस GGML_STATUS_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
bool is_model_splitted(ggml_cgraph * cgraph) {
    अटल स्थिर bool fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    अगर (!fallback_enabled) {
        वापस false;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    अगर (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        वापस false;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * गाँठ = cgraph->nodes[i];
        पूर्णांक use_गिनती = cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, गाँठ)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        अगर ((cgraph->n_nodes <= 1 && use_गिनती == 0) ||
            (cgraph->n_nodes <= 1 && गाँठ->op == GGML_OP_VIEW && use_गिनती == 1 && गाँठ->src[0] != nullptr &&
             गाँठ->src[0]->op == GGML_OP_NONE)) {
            वापस false;
        }
        अगर (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_PERMUTE)) {
            वापस false;
        }
        पूर्णांक निवेश_use_गिनती = 0;
        क्रम (पूर्णांक j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_गाँठ = cgraph->nodes[j];
            क्रम (पूर्णांक k = 0; k < GGML_MAX_SRC; k++) {
                अगर (other_गाँठ->src[k] == गाँठ) {
                    निवेश_use_गिनती++;
                }
            }
        }
        अगर (use_गिनती != निवेश_use_गिनती && गाँठ->op != GGML_OP_NONE) {
            वापस true;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    अगर (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_weight_names(cgraph);
    } अन्यथा {
        क्रम (स्थिर स्वतः & kv : GgmlOvDecoder::create_weight_nodes(cgraph, true)) {
            model_weights.insert(kv.first);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * गाँठ = cgraph->nodes[i];
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = गाँठ->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            अगर (src != nullptr && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.empty() == false &&
                model_leafs.find(src) == model_leafs.end()) {
                अगर (GgmlOvDecoder::is_inp_tok(src, गाँठ)) {
                    वापस false;
                }
                वापस true;
            }
        }
    }
    वापस false;
}

bool is_naive(ggml_cgraph * cgraph) {
    constexpr पूर्णांक naive_graph_आकार_threshold = 20;
    पूर्णांक count = 0;
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        अगर (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    वापस count < naive_graph_आकार_threshold;
}

माप_प्रकार checksum(स्थिर व्योम * तथ्य, माप_प्रकार size) {
    स्थिर uint8_t * bytes = static_cast<स्थिर uint8_t *>(तथ्य);
    माप_प्रकार योग_2 = 0;
    क्रम (माप_प्रकार i = 0; i < size; ++i) {
        योग_2 += (uint8_t) i;
        योग_2 += bytes[i];
    }
    वापस योग_2;
}

bool save_ggml_tensor_तथ्य_to_txt(स्थिर ggml_tensor * tensor, स्थिर std::string & file_path) {
    अगर (tensor == nullptr || tensor->तथ्य == nullptr) {
        वापस false;
    }

    std::बख_प्रवाह out(file_path);
    अगर (!out.is_open()) {
        वापस false;
    }

    स्थिर माप_प्रकार n = ggml_nelements(tensor);
    out << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    चयन (tensor->type) {
    हाल GGML_TYPE_F32: {
        स्थिर स्वतः * तथ्य = static_cast<स्थिर भग्न *>(tensor->तथ्य);
        क्रम (माप_प्रकार i = 0; i < n; ++i) {
            out << तथ्य[i] << '\n';
        }
        अवरोध;
    }
    हाल GGML_TYPE_F16: {
        स्थिर स्वतः * तथ्य = static_cast<स्थिर ggml_fp16_t *>(tensor->तथ्य);
        क्रम (माप_प्रकार i = 0; i < n; ++i) {
            out << ggml_fp16_to_fp32(तथ्य[i]) << '\n';
        }
        अवरोध;
    }
    हाल GGML_TYPE_BF16: {
        स्थिर स्वतः * तथ्य = static_cast<स्थिर ggml_bf16_t *>(tensor->तथ्य);
        क्रम (माप_प्रकार i = 0; i < n; ++i) {
            out << ggml_bf16_to_fp32(तथ्य[i]) << '\n';
        }
        अवरोध;
    }
    हाल GGML_TYPE_I32: {
        स्थिर स्वतः * तथ्य = static_cast<स्थिर int32_t *>(tensor->तथ्य);
        क्रम (माप_प्रकार i = 0; i < n; ++i) {
            out << तथ्य[i] << '\n';
        }
        अवरोध;
    }
    हाल GGML_TYPE_I64: {
        स्थिर स्वतः * तथ्य = static_cast<स्थिर int64_t *>(tensor->तथ्य);
        क्रम (माप_प्रकार i = 0; i < n; ++i) {
            out << तथ्य[i] << '\n';
        }
        अवरोध;
    }
    शेष:
        out << "unsupported tensor type for text dump" << '\n';
        वापस false;
    }

    वापस true;
}

व्योम print_निवेश_tensor_info(स्थिर std::string & name, स्थिर ov::Tensor & tensor) {
    std::अ_बाहर << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.तथ्य()
              << '\n';
    चयन (tensor.get_element_type()) {
    हाल ov::element::f32: {
        अगर (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::अ_बाहर << *(tensor.तथ्य<भग्न>()) << '\n';
        } अन्यथा {
            माप_प्रकार rows = tensor.get_shape()[2];
            माप_प्रकार cols = tensor.get_shape()[3];
            स्थिर भग्न * तथ्य = tensor.तथ्य<भग्न>();
            क्रम (माप_प्रकार i = 0; i < rows; ++i) {
                क्रम (माप_प्रकार j = 0; j < cols; ++j) {
                    भग्न val = तथ्य[i * cols + j];
                    अगर (std::isinf(val) && val < 0) {
                        std::अ_बाहर << std::रखो_चौड़ाई(5) << "-inf";
                    } अन्यथा {
                        std::अ_बाहर << std::रखो_चौड़ाई(5) << val;
                    }
                }
                std::अ_बाहर << '\n';
            }
        }

        अवरोध;
    }
    हाल ov::element::f16:
        std::अ_बाहर << *(tensor.तथ्य<ov::float16>()) << '\n';
        अवरोध;
    हाल ov::element::i32:
        क्रम (माप_प्रकार i = 0; i < tensor.get_आकार(); ++i) {
            std::अ_बाहर << tensor.तथ्य<int32_t>()[i] << ' ';
        }
        std::अ_बाहर << '\n';
        अवरोध;
    हाल ov::element::i64:
        क्रम (माप_प्रकार i = 0; i < tensor.get_आकार(); ++i) {
            std::अ_बाहर << tensor.तथ्य<int64_t>()[i] << ' ';
        }
        std::अ_बाहर << '\n';
        अवरोध;
    शेष:
        अवरोध;
    }
}

व्योम print_output_tensor_info(स्थिर std::string & name, स्थिर ov::Tensor & tensor, स्थिर व्योम * output_dst) {
    std::अ_बाहर << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    स्वतः print_float_stats = [](स्थिर std::string & type_name, माप_प्रकार size, स्वतः get_मूल्य) {
        अगर (size == 0) {
            वापस;
        }

        भग्न first = get_मूल्य(0);
        भग्न min = first;
        भग्न max = first;
        द्विगुन योग_2 = first;

        क्रम (माप_प्रकार i = 1; i < size; ++i) {
            भग्न v = get_मूल्य(i);
            min = std::min(v, min);
            max = std::max(v, max);
            योग_2 += v;
        }
        द्विगुन mean = योग_2 / size;

        std::अ_बाहर << std::दाँए << std::रखो_चौड़ाई(6) << type_name << std::दाँए << std::रखो_चौड़ाई(12) << "First" << std::रखो_चौड़ाई(12)
                  << "Min" << std::रखो_चौड़ाई(12) << "Max" << std::रखो_चौड़ाई(12) << "Mean" << '\n';
        std::अ_बाहर << std::दाँए << std::रखो_चौड़ाई(6) << "" << std::दाँए << std::रखो_चौड़ाई(12) << first << std::रखो_चौड़ाई(12) << min
                  << std::रखो_चौड़ाई(12) << max << std::रखो_चौड़ाई(12) << mean << '\n';
    };

    चयन (tensor.get_element_type()) {
    हाल ov::element::f32: {
        स्थिर भग्न * तथ्य = tensor.तथ्य<भग्न>();
        माप_प्रकार size = tensor.get_आकार();
        print_float_stats("[f32]", size, [तथ्य](माप_प्रकार i) { वापस तथ्य[i]; });
        अवरोध;
    }
    हाल ov::element::f16: {
        स्थिर ov::float16 * तथ्य = tensor.तथ्य<ov::float16>();
        माप_प्रकार size = tensor.get_आकार();
        print_float_stats("[f16]", size, [तथ्य](माप_प्रकार i) { वापस static_cast<भग्न>(तथ्य[i]); });
        अवरोध;
    }
    शेष:
        अवरोध;
    }
}

स्थिर ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; ++i) {
        स्वतः * op = cgraph->nodes[i];
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; ++j) {
            स्वतः * src = op->src[j];
            अगर (src == nullptr) {
                अवरोध;
            }
            अगर (GgmlOvDecoder::is_inp_pos(src, op)) {
                वापस src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    फेंको std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_tokens(ggml_cgraph * cgraph, स्थिर ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    पूर्णांक n_planes = 1;
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; ++i) {
        स्वतः * op = cgraph->nodes[i];
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; ++j) {
            अगर (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                अवरोध;
            }
        }
    }
    वापस inp_pos->ne[0] / n_planes;
}

bool get_is_prefill(ggml_cgraph * cgraph, स्थिर ggml_tensor * inp_pos) {
    वापस get_inp_pos_n_tokens(cgraph, inp_pos) > 1;
}
