#అన్తర్భావ "utils.h"

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-openvino-extra.h"
#అన్తర్భావ "ggml-openvino.h"
#అన్తర్భావ "ggml-openvino/ggml-decoder.h"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "model-cache.h"
#అన్తర్భావ "openvino/frontend.h"
#అన్తర్భావ "openvino/input_model.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cassert>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>
#అన్తర్భావ <fstream>
#అన్తర్భావ <functional>
#అన్తర్భావ <iomanip>
#అన్తర్భావ <iostream>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/any.hpp>
#అన్తర్భావ <openvino/core/graph_util.hpp>
#అన్తర్భావ <openvino/core/shape.hpp>
#అన్తర్భావ <openvino/core/type/float16.hpp>
#అన్తర్భావ <openvino/frontend/manager.hpp>
#అన్తర్భావ <openvino/openvino.hpp>
#అన్తర్భావ <openvino/runtime/compiled_model.hpp>
#అన్తర్భావ <openvino/runtime/infer_request.hpp>
#అన్తర్భావ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#అన్తర్భావ <openvino/runtime/intel_npu/properties.hpp>
#అన్తర్భావ <openvino/runtime/properties.hpp>
#అన్తర్భావ <openvino/runtime/tensor.hpp>
#అన్తర్భావ <optional>
#అన్తర్భావ <set>
#అన్తర్భావ <string>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <vector>

నామాకాశ {
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
std::optional<ov::Tensor> try_make_kv_sliced_tensor(స్థిరరాశి std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                                    స్థిరరాశి std::string & name,
                                                    స్థిరరాశి ggml_tensor * ggml_tensor) {
    స్థిర స్థిరరాశి తార్కిక kv_ముక్క_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    ఒకవేళ (kv_ముక్క_disabled) {
        తిరిగి std::nullopt;
    }
    ఒకవేళ (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        తిరిగి std::nullopt;
    }
    ఒకవేళ (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != శూన్యసూచిక) {
        తిరిగి std::nullopt;
    }
    స్థిరరాశి స్వయంచాలక * op = ggml_decoder->get_tensor_వాడిన_op(ggml_tensor);
    ఒకవేళ (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        తిరిగి std::nullopt;
    }

    స్థిరరాశి స్వయంచాలక & compute_params = ggml_decoder->get_compute_params();
    ఒకవేళ (compute_params.n_seq_చురుకైన != 1 || compute_params.seq_చురుకైన_ప్రారంభం != 0) {
        తిరిగి std::nullopt;
    }

    పూర్ణాంకం layer;
    ఒకవేళ (స్వయంచాలక layer_opt = వెలికితీయి_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } లేకపోతే {
        తిరిగి std::nullopt;
    }

    స్థిరరాశి తార్కిక is_swa = ggml_decoder->is_swa_layer(layer);
    ఒకవేళ (is_swa) {
        తిరిగి std::nullopt;
    }
    స్థిరరాశి పూర్ణాంకం ctx_చొప్పున_seq = ggml_decoder->get_ctx_చొప్పున_seq();
    స్థిరరాశి పూర్ణాంకం n_kv = compute_params.attention_కొలత;
    ఒకవేళ (ctx_చొప్పున_seq <= 0 || n_kv <= 0 || n_kv >= ctx_చొప్పున_seq) {
        తిరిగి std::nullopt;
    }

    ov::Shape నిండిన_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    ఒకవేళ (నిండిన_shape.size() != 4 || నిండిన_shape[0] != 1 || నిండిన_shape[1] != 1 ||
        static_cast<పూర్ణాంకం>(నిండిన_shape[2]) != ctx_చొప్పున_seq) {
        తిరిగి std::nullopt;
    }

    ov::Shape sliced_shape = నిండిన_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    ఒకవేళ (ggml_openvino_బఫరు_is_remote(ggml_tensor)) {
        స్వయంచాలక remote_context = ggml_openvino_get_remote_context();
        స్వయంచాలక gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        తిరిగి gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->దత్తాంశం);
    }

    తిరిగి ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->దత్తాంశం);
}

స్థిర uint64_t ggml_openvino_model_cache_extra_cfg(తార్కిక stateful, తార్కిక recurrent) {
    స్థిరరాశి అక్షరం * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    స్థిరరాశి తార్కిక manual_gqa_enabled = manual_gqa_env != శూన్యసూచిక ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    తిరిగి extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_బరువు_పేర్లు(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> పేర్లు;
    కొరకు (స్థిరరాశి స్వయంచాలక & name : GgmlOvDecoder::collect_బరువు_పేర్లు(cgraph)) {
        పేర్లు[name] = శూన్యసూచిక;
    }
    తిరిగి పేర్లు;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
స్థిర std::string compiled_లేఖాచిత్రం_కీలకం(స్థిరరాశి ggml_cgraph * graph,
                                      స్థిరరాశి GgmlOvDecoder & decoder,
                                      స్థిరరాశి std::string & device,
                                      పూర్ణాంకం prefill_chunk_కొలత = 0,
                                      తార్కిక disk_cache = అబద్ధం) {
    std::string కీలకం;
    స్వయంచాలక append = [&కీలకం](స్థిరరాశి స్వయంచాలక & value) {
        కీలకం.append(reinterpret_cast<స్థిరరాశి అక్షరం *>(&value), పరిమాణం(value));
    };
    స్వయంచాలక append_string = [&](స్థిరరాశి std::string & value) {
        append(value.size());
        కీలకం.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_కొలత);
    తార్కిక has_బరువు_బఫరు_id = అబద్ధం;
    std::unordered_map<స్థిరరాశి ggml_tensor *, size_t> ids;
    std::function<శూన్య(స్థిరరాశి ggml_tensor *)> సందర్శించు = [&](స్థిరరాశి ggml_tensor * tensor) {
        ఒకవేళ (!tensor) {
            append(size_t(0));
            తిరిగి;
        }
        స్వయంచాలక inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.మొదటి->second);
        ఒకవేళ (!inserted.second) {
            తిరిగి;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        స్థిరరాశి స్వయంచాలక * పునాది = tensor->view_src ? tensor->view_src : tensor;
        స్థిరరాశి తార్కిక బరువు = పునాది->బఫరు && పునాది->బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_WEIGHTS;
        append(బరువు);
        ఒకవేళ (బరువు && !disk_cache) {
            స్థిరరాశి size_t బఫరు_id = ggml_backend_openvino_బఫరు_get_ctx_id(పునాది->బఫరు);
            has_బరువు_బఫరు_id |= బఫరు_id != 0;
            append(బఫరు_id);
            append(tensor->దత్తాంశం);
        }
        సందర్శించు(tensor->view_src);
        కొరకు (స్థిరరాశి స్వయంచాలక * src : tensor->src) {
            సందర్శించు(src);
        }
    };
    append(graph->n_nodes);
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; ++i) {
        సందర్శించు(graph->nodes[i]);
    }
    append(graph->n_leafs);
    కొరకు (పూర్ణాంకం i = 0; i < graph->n_leafs; ++i) {
        సందర్శించు(graph->leafs[i]);
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & ఆదానం : decoder.get_model_extra_inputs()) {
        append_string(ఆదానం.మొదటి);
        append_string(ఆదానం.second.type.get_type_name());
        append(ఆదానం.second.shape.size());
        కొరకు (స్వయంచాలక dim : ఆదానం.second.shape) {
            append(dim);
        }
        append(ఆదానం.second.is_parameter);
        ఒకవేళ (!ఆదానం.second.is_parameter) {
            append(ఆదానం.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    తిరిగి disk_cache || has_బరువు_బఫరు_id ? కీలకం : std::string{};
}

స్థిర std::string dynamic_లేఖాచిత్రం_signature(స్థిరరాశి ggml_cgraph * graph,
                                           స్థిరరాశి GgmlOvDecoder & decoder,
                                           స్థిరరాశి ModelParams & params) {
    std::string కీలకం = "dynamic-1";
    స్వయంచాలక append = [&కీలకం](స్థిరరాశి స్వయంచాలక & value) {
        కీలకం.append(reinterpret_cast<స్థిరరాశి అక్షరం *>(&value), పరిమాణం(value));
    };
    స్వయంచాలక append_string = [&](స్థిరరాశి std::string & value) {
        append(value.size());
        కీలకం.append(value);
    };
    లేఖాచిత్రం_కీలకం లేఖాచిత్రం_id(graph, నిజం);
    append(లేఖాచిత్రం_id.n_nodes);
    append(లేఖాచిత్రం_id.n_leaves);
    append_string(లేఖాచిత్రం_id.మొదటి_కణుపు_name);
    append_string(లేఖాచిత్రం_id.చివరి_కణుపు_name);
    కొరకు (స్థిరరాశి స్వయంచాలక & name : లేఖాచిత్రం_id.ఆదానం_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    కొరకు (పూర్ణాంకం layer : params.swa_layers) {
        append(layer);
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & [layer, heads] : params.n_heads_kv_చొప్పున_layer) {
        append(layer);
        append(heads);
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & [name, ఆదానం] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(ఆదానం.type.get_type_name());
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & [name, ఆదానం] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(ఆదానం.type.get_type_name());
        append(ఆదానం.is_parameter);
        ఒకవేళ (!ఆదానం.is_parameter) {
            append(ఆదానం.value);
        }
    }
    తిరిగి కీలకం;
}

స్థిర తార్కిక compiled_model_matches_లేఖాచిత్రం(స్థిరరాశి ov::CompiledModel & model,
                                         స్థిరరాశి std::vector<std::string> & inputs,
                                         స్థిరరాశి std::vector<std::string> & outputs,
                                         స్థిరరాశి GgmlOvDecoder & decoder) {
    ఒకవేళ (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి స్వయంచాలక & లేఖాచిత్రం_inputs = decoder.get_model_inputs();
    స్థిరరాశి స్వయంచాలక & extra_inputs = decoder.get_model_extra_inputs();
    స్థిరరాశి స్వయంచాలక & లేఖాచిత్రం_outputs = decoder.get_model_outputs();
    కొరకు (size_t i = 0; i < inputs.size(); ++i) {
        స్థిరరాశి స్వయంచాలక & పోర్టు = model.ఆదానం(i);
        స్వయంచాలక లేఖాచిత్రం_it = లేఖాచిత్రం_inputs.find(inputs[i]);
        ఒకవేళ (లేఖాచిత్రం_it != లేఖాచిత్రం_inputs.end()) {
            ఒకవేళ (పోర్టు.get_element_type() != లేఖాచిత్రం_it->second.type ||
                !పోర్టు.get_partial_shape().compatible(లేఖాచిత్రం_it->second.shape)) {
                తిరిగి అబద్ధం;
            }
            కొనసాగు;
        }
        స్వయంచాలక extra_it = extra_inputs.find(inputs[i]);
        ఒకవేళ (extra_it == extra_inputs.end() || పోర్టు.get_element_type() != extra_it->second.type ||
            !పోర్టు.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            తిరిగి అబద్ధం;
        }
    }
    కొరకు (size_t i = 0; i < outputs.size(); ++i) {
        స్వయంచాలక లేఖాచిత్రం_it = లేఖాచిత్రం_outputs.find(outputs[i]);
        ఒకవేళ (లేఖాచిత్రం_it == లేఖాచిత్రం_outputs.end()) {
            ఒకవేళ (outputs[i].compare(0, 8, "__debug_") != 0) {
                తిరిగి అబద్ధం;
            }
            కొనసాగు;
        }
        స్థిరరాశి స్వయంచాలక & పోర్టు = model.output(i);
        ఒకవేళ (పోర్టు.get_element_type() != GgmlOvDecoder::get_ov_type(లేఖాచిత్రం_it->second) ||
            !పోర్టు.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(లేఖాచిత్రం_it->second)))) {
            తిరిగి అబద్ధం;
        }
    }
    తిరిగి నిజం;
}

స్థిర std::string ov_profiling_csv_క్షేత్రం(స్థిరరాశి std::string & value) {
    std::string escaped = "\"";
    కొరకు (అక్షరం c : value) {
        escaped += c;
        ఒకవేళ (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    తిరిగి escaped;
}

స్థిర శూన్య dump_ov_profiling_info(స్థిరరాశి ov::InferRequest & infer_request) {
    ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        తిరిగి;
    }

    స్థిర std::atomic<uint64_t> inference_సూచీ{0};
    స్థిరరాశి uint64_t సూచీ = inference_సూచీ.fetch_add(1);
    స్థిరరాశి std::string path = "openvino_profile_" + std::to_string(సూచీ) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    ఒకవేళ (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        తిరిగి;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t మొత్తం_real_సమయం = 0;
    int64_t మొత్తం_cpu_సమయం = 0;
    size_t executed_లెక్క = 0;
    కొరకు (స్థిరరాశి స్వయంచాలక & info : infer_request.get_profiling_info()) {
        స్థిరరాశి అక్షరం * స్థితి_2 = "NOT_RUN";
        ఒకవేళ (info.స్థితి_2 == ov::ProfilingInfo::స్థితి_3::EXECUTED) {
            స్థితి_2 = "EXECUTED";
            మొత్తం_real_సమయం += info.real_సమయం.count();
            మొత్తం_cpu_సమయం += info.cpu_సమయం.count();
            executed_లెక్క++;
        } లేకపోతే ఒకవేళ (info.స్థితి_2 == ov::ProfilingInfo::స్థితి_3::OPTIMIZED_బహిర్గతం) {
            స్థితి_2 = "OPTIMIZED_OUT";
        }

        output << స్థితి_2 << ',' << info.real_సమయం.count() << ',' << info.cpu_సమయం.count() << ','
               << info.ప్రారంభం_సమయం.count() << ',' << ov_profiling_csv_క్షేత్రం(info.node_type) << ','
               << ov_profiling_csv_క్షేత్రం(info.కణుపు_name) << ',' << ov_profiling_csv_క్షేత్రం(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_లెక్క, మొత్తం_real_సమయం / 1000.0, మొత్తం_cpu_సమయం / 1000.0);
}

స్థిర శూన్య reset_single_స్థానం_recurrent_cache(ggml_cgraph * cgraph,
                                              స్థిరరాశి ModelParams & model_params,
                                              స్థిరరాశి ComputeParams & compute_params) {
    ఒకవేళ (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        తిరిగి;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_బఫరు_t> buffers;
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        స్వయంచాలక * కణుపు = cgraph->nodes[i];
        ఒకవేళ (కణుపు->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(కణుపు->view_src) &&
            కణుపు->view_src->ne[1] == 1 && కణుపు->view_src->బఫరు != శూన్యసూచిక) {
            buffers.insert(కణుపు->view_src->బఫరు);
        }
    }
    కొరకు (స్వయంచాలక * బఫరు : buffers) {
        ggml_backend_బఫరు_clear(బఫరు, 0);
    }
}

ov::Tensor create_ov_output_tensor(స్థిరరాశి std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   స్థిరరాశి std::shared_ptr<ov::InferRequest> & infer_request,
                                   పూర్ణాంకం output_సూచీ,
                                   స్థిరరాశి ggml_tensor * ggml_tensor) {
    ఒకవేళ (స్వయంచాలక sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        తిరిగి *sliced;
    }

    స్వయంచాలక output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    శూన్య * output_దత్తాంశం = ggml_tensor->దత్తాంశం;
    ఒకవేళ (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_సూచీ).get_shape();
    } లేకపోతే {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        ఒకవేళ (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != శూన్యసూచిక &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_దత్తాంశం = ggml_tensor->view_src->దత్తాంశం;
        } లేకపోతే {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    ఒకవేళ (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_బఫరు_is_remote(ggml_tensor) &&
        ggml_tensor->extra != శూన్యసూచిక) {
        స్వయంచాలక * extra_పునాది = static_cast<ggml_openvino_extra_పునాది *>(ggml_tensor->extra);
        ఒకవేళ (extra_పునాది->type == ggml_openvino_extra_పునాది::Type::TENSOR) {
            స్వయంచాలక * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_పునాది);
            ఒకవేళ (tensor_extra->tensor != శూన్యసూచిక && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                తిరిగి *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_దత్తాంశం);
    తిరిగి output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_అడ్డువరుసలు_గమ్యం_seq_axis_2(స్థిరరాశి ov::Tensor & kv_tensor, size_t n_heads_kv) {
    స్థిరరాశి size_t అడ్డువరుసలు = kv_tensor.get_shape()[2];
    స్థిరరాశి size_t తల_కొలత = kv_tensor.get_shape()[3] / n_heads_kv;
    స్థిరరాశి size_t elem = kv_tensor.get_element_type().size();
    స్థిరరాశి size_t తల_బైట్లు = తల_కొలత * elem;

    ov::Tensor బహిర్గతం(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, అడ్డువరుసలు, తల_కొలత});
    స్థిరరాశి స్వయంచాలక * src = static_cast<స్థిరరాశి uint8_t *>(kv_tensor.దత్తాంశం());
    స్వయంచాలక * dst = static_cast<uint8_t *>(బహిర్గతం.దత్తాంశం());
    కొరకు (size_t s = 0; s < అడ్డువరుసలు; s++) {
        కొరకు (size_t h = 0; h < n_heads_kv; h++) {
            జ్ఞాపకం_నకలు(dst + (h * అడ్డువరుసలు + s) * తల_బైట్లు, src + (s * n_heads_kv + h) * తల_బైట్లు, తల_బైట్లు);
        }
    }
    తిరిగి బహిర్గతం;
}

మూస <రకం_పేరు T> శూన్య సమితి_zero_diagonal(std::vector<T> & మాతృక, size_t అడ్డువరుసలు, size_t నిలువుల_సంఖ్య_2, T zero_విలువ = T{}) {
    కొరకు (size_t i = 0; i < అడ్డువరుసలు; ++i) {
        size_t diag_నిలువు = std::min(i, నిలువుల_సంఖ్య_2 - 1);
        మాతృక[i * నిలువుల_సంఖ్య_2 + diag_నిలువు] = zero_విలువ;
    }
}

ov::Tensor make_contiguous_విభజించు_ఆదానం_tensor(స్థిరరాశి నిర్మాణం ggml_tensor * ggml_tensor, స్థిరరాశి ov::Shape & ఆదానం_shape) {
    స్థిరరాశి size_t element_కొలత = ggml_type_కొలత(ggml_tensor->type);
    స్థిరరాశి size_t ఖండం_కొలత_3 = ggml_blck_కొలత(ggml_tensor->type);

    GGML_ASSERT(ఖండం_కొలత_3 == 1 && "non-contiguous split inputs must be plain element types");

    స్థిరరాశి నిర్మాణం ggml_tensor * ఆధారం_tensor = ggml_tensor->view_src != శూన్యసూచిక ? ggml_tensor->view_src : ggml_tensor;
    స్థిరరాశి size_t ఆధారం_స్థానభ్రంశం = ggml_tensor->view_src != శూన్యసూచిక ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> ఆధారం_దత్తాంశం(ggml_nbytes(ఆధారం_tensor));
    ggml_backend_tensor_get(ఆధారం_tensor, ఆధారం_దత్తాంశం.దత్తాంశం(), 0, ఆధారం_దత్తాంశం.size());

    ov::Tensor ఆదానం_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ఆదానం_shape);
    స్వయంచాలక * dst = static_cast<uint8_t *>(ఆదానం_tensor.దత్తాంశం());
    size_t dst_స్థానభ్రంశం = 0;

    కొరకు (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        కొరకు (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            కొరకు (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                కొరకు (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    స్థిరరాశి size_t src_స్థానభ్రంశం = ఆధారం_స్థానభ్రంశం + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::జ్ఞాపకం_నకలు(dst + dst_స్థానభ్రంశం, ఆధారం_దత్తాంశం.దత్తాంశం() + src_స్థానభ్రంశం, element_కొలత);
                    dst_స్థానభ్రంశం += element_కొలత;
                }
            }
        }
    }

    తిరిగి ఆదానం_tensor;
}

ov::Tensor convert_ggml_ఆదానం_గమ్యం_ov(స్థిరరాశి std::shared_ptr<GgmlOvDecoder> & ggml_decoder, స్థిరరాశి std::string & name) {
    స్థిరరాశి స్వయంచాలక * ggml_tensor = ggml_decoder->get_ఆదానం_ggml_tensor(name);

    ఒకవేళ (స్వయంచాలక sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        తిరిగి *sliced;
    }

    ఒకవేళ (ggml_tensor->extra != శూన్యసూచిక && !ggml_decoder->is_splited_model()) {
        స్వయంచాలక * extra_పునాది = static_cast<ggml_openvino_extra_పునాది *>(ggml_tensor->extra);
        ఒకవేళ (extra_పునాది->type == ggml_openvino_extra_పునాది::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            స్వయంచాలక * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_పునాది);
            తిరిగి *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    స్వయంచాలక * ఆదానం_దత్తాంశం = ggml_tensor->దత్తాంశం;
    ov::Shape ఆదానం_shape;
    ఒకవేళ (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        ఆదానం_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } లేకపోతే {
        ఆదానం_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    ఒకవేళ (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        తిరిగి make_contiguous_విభజించు_ఆదానం_tensor(ggml_tensor, ఆదానం_shape);
    }

    స్వయంచాలక ఆదానం_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ఆదానం_shape, ఆదానం_దత్తాంశం);
    తిరిగి ఆదానం_tensor;
}

ov::Tensor get_ov_ఆదానం_tensor(స్థిరరాశి std::shared_ptr<GgmlOvDecoder> & ggml_decoder, స్థిరరాశి std::string & param_name) {
    ov::Tensor ఆదానం_tensor;
    స్వయంచాలక extra_ఆదానం = ggml_decoder->get_model_extra_inputs().find(param_name);
    ఒకవేళ (extra_ఆదానం != ggml_decoder->get_model_extra_inputs().end()) {
        ఆదానం_tensor = ov::Tensor(extra_ఆదానం->second.type, extra_ఆదానం->second.shape);
        *ఆదానం_tensor.దత్తాంశం<int64_t>() = extra_ఆదానం->second.value;
    } లేకపోతే {
        ఆదానం_tensor = convert_ggml_ఆదానం_గమ్యం_ov(ggml_decoder, param_name);
    }
    తిరిగి ఆదానం_tensor;
}

ov::Tensor get_ov_ఆదానం_tensor_static_decode(స్థిరరాశి std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             స్థిరరాశి std::string & param_name) {
    // NPU decoding stage
    ఒకవేళ (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        తిరిగి get_ov_ఆదానం_tensor(ggml_decoder, param_name);
    }
    స్థిరరాశి స్వయంచాలక * ggml_tensor = ggml_decoder->get_ఆదానం_ggml_tensor(param_name);
    స్థిరరాశి స్వయంచాలక * op = ggml_decoder->get_tensor_వాడిన_op(ggml_tensor);

    ఒకవేళ (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        స్థిరరాశి పూర్ణాంకం n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        ప్రతిజ్ఞా(ggml_tensor->ne[0] == n_planes);
        ov::Shape ఆదానం_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor ఆదానం_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ఆదానం_shape);
        std::జ్ఞాపకం_నకలు(ఆదానం_tensor.దత్తాంశం(), ggml_tensor->దత్తాంశం, n_planes * ggml_type_కొలత(ggml_tensor->type));
        తిరిగి ఆదానం_tensor;
    }

    ఒకవేళ (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape ఆదానం_shape = {1, 1, 1, 1};
        ov::Tensor ఆదానం_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ఆదానం_shape);
        int32_t inp_బహిర్గతం_id = *((int32_t *) ggml_tensor->దత్తాంశం);
        ప్రతిజ్ఞా(ggml_tensor->ne[0] == 1);
        ప్రతిజ్ఞా(inp_బహిర్గతం_id == 0);
        *ఆదానం_tensor.దత్తాంశం<int32_t>() = inp_బహిర్గతం_id;
        తిరిగి ఆదానం_tensor;
    }

    ఒకవేళ (GgmlOvDecoder::is_inp_ముసుగు(ggml_tensor, op)) {
        size_t context_కొలత = ggml_decoder->get_ctx_కొలత();
        ఒకవేళ (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_దత్తాంశం =
                pad_ఆదానం<ggml_fp16_t>(ggml_tensor, 1, context_కొలత, GGML_FP32_గమ్యం_FP16(-INFINITY));
            ov::Tensor ఆదానం_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_కొలత});
            std::జ్ఞాపకం_నకలు(ఆదానం_tensor.దత్తాంశం(), padded_దత్తాంశం.దత్తాంశం(), padded_దత్తాంశం.size() * పరిమాణం(ggml_fp16_t));
            తిరిగి ఆదానం_tensor;
        }

        std::vector<దశాంశం> padded_దత్తాంశం = pad_ఆదానం<దశాంశం>(ggml_tensor, 1, context_కొలత, -INFINITY);
        ov::Tensor ఆదానం_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_కొలత});
        స్వయంచాలక * దత్తాంశం_ptr = ఆదానం_tensor.దత్తాంశం<దశాంశం>();
        std::copy(padded_దత్తాంశం.begin(), padded_దత్తాంశం.begin() + context_కొలత, దత్తాంశం_ptr);
        తిరిగి ఆదానం_tensor;
    }

    తిరిగి get_ov_ఆదానం_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_ఆదానం_tensor_static_prefill(స్థిరరాశి std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              స్థిరరాశి std::string & param_name,
                                              పూర్ణాంకం chunk_సూచీ) {
    // NPU prompt processing stage
    స్థిరరాశి size_t ఆదానం_len = ggml_decoder->get_ఆదానం_len();
    స్థిరరాశి size_t chunk_కొలత = ggml_decoder->m_prefill_chunk_కొలత;
    స్థిరరాశి size_t chunk_valid_కొలత = std::min(chunk_కొలత, ఆదానం_len - chunk_సూచీ * chunk_కొలత);
    స్థిరరాశి size_t chunk_pad_కొలత = chunk_కొలత - chunk_valid_కొలత;

    ఒకవేళ (param_name == "chunk_valid_len") {
        ov::Tensor ఆదానం_tensor(ov::element::i64, ov::Shape{1});
        *ఆదానం_tensor.దత్తాంశం<int64_t>() = (int64_t) chunk_valid_కొలత;
        తిరిగి ఆదానం_tensor;
    }
    ఒకవేళ (chunk_సూచీ > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor ఆదానం_tensor(ov::element::i64, ov::Shape{1});
        *ఆదానం_tensor.దత్తాంశం<int64_t>() = 0;
        తిరిగి ఆదానం_tensor;
    }
    ఒకవేళ (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        తిరిగి get_ov_ఆదానం_tensor(ggml_decoder, param_name);
    }
    స్థిరరాశి స్వయంచాలక * ggml_tensor = ggml_decoder->get_ఆదానం_ggml_tensor(param_name);
    స్థిరరాశి స్వయంచాలక * op = ggml_decoder->get_tensor_వాడిన_op(ggml_tensor);

    ఒకవేళ (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        స్థిరరాశి పూర్ణాంకం n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        స్థిరరాశి size_t element_కొలత = ggml_type_కొలత(ggml_tensor->type);
        ov::Shape ఆదానం_shape = {1, 1, 1, (size_t) n_planes * chunk_కొలత};
        ov::Tensor ఆదానం_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ఆదానం_shape);
        కొరకు (పూర్ణాంకం p = 0; p < n_planes; p++) {
            స్థిరరాశి అక్షరం * src =
                (స్థిరరాశి అక్షరం *) ggml_tensor->దత్తాంశం + (p * ఆదానం_len + chunk_సూచీ * chunk_కొలత) * element_కొలత;
            అక్షరం * dst = (అక్షరం *) ఆదానం_tensor.దత్తాంశం() + p * chunk_కొలత * element_కొలత;
            std::జ్ఞాపకం_నకలు(dst, src, chunk_valid_కొలత * element_కొలత);
            ఒకవేళ (chunk_pad_కొలత > 0) {
                ఒకవేళ (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t చివరి_విలువ = *((స్థిరరాశి int32_t *) src + chunk_valid_కొలత - 1);
                    int32_t * బహిర్గతం = (int32_t *) dst;
                    std::fill(బహిర్గతం + chunk_valid_కొలత, బహిర్గతం + chunk_కొలత, చివరి_విలువ + 1);
                } లేకపోతే ఒకవేళ (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t చివరి_విలువ = *((స్థిరరాశి int64_t *) src + chunk_valid_కొలత - 1);
                    int64_t * బహిర్గతం = (int64_t *) dst;
                    std::fill(బహిర్గతం + chunk_valid_కొలత, బహిర్గతం + chunk_కొలత, చివరి_విలువ + 1);
                } లేకపోతే {
                    విసురు std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        తిరిగి ఆదానం_tensor;
    }

    ఒకవేళ (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape ఆదానం_shape = {1, 1, 1, chunk_కొలత};
        ov::Tensor ఆదానం_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ఆదానం_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_కొలత = ggml_type_కొలత(ggml_tensor->type);
        శూన్య * ఆదానం_దత్తాంశం = (అక్షరం *) ggml_tensor->దత్తాంశం + chunk_సూచీ * chunk_కొలత * element_కొలత;
        std::జ్ఞాపకం_నకలు(ఆదానం_tensor.దత్తాంశం(), ఆదానం_దత్తాంశం, chunk_valid_కొలత * element_కొలత);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        ఒకవేళ (chunk_pad_కొలత > 0) {
            ఒకవేళ (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t చివరి_విలువ =
                    *((int32_t *) ggml_tensor->దత్తాంశం + (chunk_సూచీ * chunk_కొలత + chunk_valid_కొలత - 1));
                int32_t * output_దత్తాంశం = ఆదానం_tensor.దత్తాంశం<int32_t>();
                std::fill(output_దత్తాంశం + chunk_valid_కొలత, output_దత్తాంశం + chunk_కొలత, చివరి_విలువ + 1);
            } లేకపోతే ఒకవేళ (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t చివరి_విలువ =
                    *((int64_t *) ggml_tensor->దత్తాంశం + (chunk_సూచీ * chunk_కొలత + chunk_valid_కొలత - 1));
                int64_t * output_దత్తాంశం = ఆదానం_tensor.దత్తాంశం<int64_t>();
                std::fill(output_దత్తాంశం + chunk_valid_కొలత, output_దత్తాంశం + chunk_కొలత, చివరి_విలువ + 1);
            } లేకపోతే {
                విసురు std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        తిరిగి ఆదానం_tensor;
    }

    ఒకవేళ (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape ఆదానం_shape = {1, 1, 1, output_len};
        ov::Tensor ఆదానం_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ఆదానం_shape);
        ఒకవేళ (ggml_tensor->ne[0] == 0) {
            *ఆదానం_tensor.దత్తాంశం<int32_t>() = 0;
        } లేకపోతే {
            స్వయంచాలక * దత్తాంశం_addr = ఆదానం_tensor.దత్తాంశం<int32_t>();
            కొరకు (size_t i = 0; i < output_len; i++) {
                దత్తాంశం_addr[i] = ((int32_t *) ggml_tensor->దత్తాంశం)[i] % chunk_కొలత;
            }
        }
        తిరిగి ఆదానం_tensor;
    }

    ఒకవేళ (GgmlOvDecoder::is_inp_scale_అడ్డువరుసలు(ggml_tensor, op)) {
        ov::Tensor ఆదానం_tensor(ov::element::f32, ov::Shape{1, 1, chunk_కొలత, 1});
        స్వయంచాలక * dst = ఆదానం_tensor.దత్తాంశం<దశాంశం>();
        స్థిరరాశి స్వయంచాలక * src = static_cast<స్థిరరాశి దశాంశం *>(ggml_tensor->దత్తాంశం) + chunk_సూచీ * chunk_కొలత;
        std::copy(src, src + chunk_valid_కొలత, dst);
        std::fill(dst + chunk_valid_కొలత, dst + chunk_కొలత, 1.0f);
        తిరిగి ఆదానం_tensor;
    }

    ఒకవేళ (GgmlOvDecoder::is_inp_సగటు(ggml_tensor, op)) {
        స్థిరరాశి size_t n_seqs = ggml_tensor->ne[1];
        స్థిరరాశి size_t src_stride = ggml_tensor->ne[0];
        స్థిరరాశి size_t copy_len = std::min<size_t>(chunk_valid_కొలత, src_stride - chunk_సూచీ * chunk_కొలత);
        ov::Tensor ఆదానం_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_కొలత});
        స్వయంచాలక * dst = ఆదానం_tensor.దత్తాంశం<దశాంశం>();
        std::fill(dst, dst + n_seqs * chunk_కొలత, 0.0f);
        స్థిరరాశి స్వయంచాలక * src = static_cast<స్థిరరాశి దశాంశం *>(ggml_tensor->దత్తాంశం) + chunk_సూచీ * chunk_కొలత;
        కొరకు (size_t s = 0; s < n_seqs; s++) {
            std::జ్ఞాపకం_నకలు(dst + s * chunk_కొలత, src + s * src_stride, copy_len * పరిమాణం(దశాంశం));
        }
        తిరిగి ఆదానం_tensor;
    }

    ఒకవేళ (GgmlOvDecoder::is_inp_ముసుగు(ggml_tensor, op)) {
        size_t నిలువుల_సంఖ్య_2 = ggml_tensor->ne[0];
        size_t అడ్డువరుసలు = ggml_tensor->ne[1];
        size_t chunk_valid_అడ్డువరుసలు = std::min(chunk_కొలత, అడ్డువరుసలు - chunk_సూచీ * chunk_కొలత);
        size_t context_కొలత = ggml_decoder->get_ctx_కొలత();
        ఒకవేళ (ggml_tensor->type == GGML_TYPE_F16) {
            స్థిరరాశి స్వయంచాలక * ggml_దత్తాంశం =
                static_cast<స్థిరరాశి ggml_fp16_t *>(ggml_tensor->దత్తాంశం) + chunk_సూచీ * chunk_కొలత * నిలువుల_సంఖ్య_2;
            std::vector<ggml_fp16_t> padded_దత్తాంశం = pad_ఆదానం<ggml_fp16_t>(ggml_దత్తాంశం, chunk_valid_అడ్డువరుసలు, నిలువుల_సంఖ్య_2, chunk_కొలత,
                                                                          context_కొలత, GGML_FP32_గమ్యం_FP16(-INFINITY));
            సమితి_zero_diagonal(padded_దత్తాంశం, chunk_కొలత, context_కొలత, GGML_FP32_గమ్యం_FP16(0.0f));
            ov::Tensor ఆదానం_tensor(ov::element::f16, ov::Shape{1, 1, chunk_కొలత, context_కొలత});
            std::జ్ఞాపకం_నకలు(ఆదానం_tensor.దత్తాంశం(), padded_దత్తాంశం.దత్తాంశం(), padded_దత్తాంశం.size() * పరిమాణం(ggml_fp16_t));
            తిరిగి ఆదానం_tensor;
        }

        స్థిరరాశి స్వయంచాలక * ggml_దత్తాంశం = static_cast<స్థిరరాశి దశాంశం *>(ggml_tensor->దత్తాంశం) + chunk_సూచీ * chunk_కొలత * నిలువుల_సంఖ్య_2;
        std::vector<దశాంశం> padded_దత్తాంశం =
            pad_ఆదానం<దశాంశం>(ggml_దత్తాంశం, chunk_valid_అడ్డువరుసలు, నిలువుల_సంఖ్య_2, chunk_కొలత, context_కొలత, -INFINITY);
        సమితి_zero_diagonal(padded_దత్తాంశం, chunk_కొలత, context_కొలత);
        ov::Tensor ఆదానం_tensor(ov::element::f32, ov::Shape{1, 1, chunk_కొలత, context_కొలత});
        స్వయంచాలక * దత్తాంశం_ptr = ఆదానం_tensor.దత్తాంశం<దశాంశం>();
        std::copy(padded_దత్తాంశం.begin(), padded_దత్తాంశం.begin() + chunk_కొలత * context_కొలత, దత్తాంశం_ptr);
        తిరిగి ఆదానం_tensor;
    }

    తిరిగి get_ov_ఆదానం_tensor(ggml_decoder, param_name);
}

గణన ggml_స్థితి naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               స్థిరరాశి std::string & device,
                               స్థిరరాశి ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    ఒకవేళ (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        తిరిగి GGML_స్థితి_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_తాళం(cache.mutex);
    తార్కిక naive = నిజం;
    స్వయంచాలక model_weights = GgmlOvDecoder::create_బరువు_nodes(cgraph, naive);
    స్వయంచాలక decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    స్వయంచాలక ఆదానం_model = std::make_shared<ov::frontend::ggml::ఆదానం_Model>(decoder);
    స్వయంచాలక model = ov::frontend::ggml::Front_చివర::convert(ఆదానం_model, naive);
    ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    స్వయంచాలక remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    ఒకవేళ (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_బహుళకం.name()] = ov::hint::Execution_బహుళకం::PERFORMANCE;
    } లేకపోతే {
        compile_config[ov::hint::execution_బహుళకం.name()] = ov::hint::Execution_బహుళకం::ACCURACY;
    }
    ఒకవేళ (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } లేకపోతే {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> ఆదానం_పేర్లు;
    std::vector<std::string> output_పేర్లు;
    కొరకు (స్థిరరాశి స్వయంచాలక & param : model->get_parameters()) {
        ఆదానం_పేర్లు.push_back(param->get_friendly_name());
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & ఫలితం : model->get_results()) {
        output_పేర్లు.push_back(ఫలితం->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    ఆదానం_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_తాళం.unlock();

    కొరకు (size_t i = 0; i < ఆదానం_పేర్లు.size(); i++) {
        స్థిరరాశి స్వయంచాలక & param_name = ఆదానం_పేర్లు[i];
        స్వయంచాలక ఆదానం_tensor = get_ov_ఆదానం_tensor(decoder, param_name);
        infer_request->సమితి_ఆదానం_tensor(i, ఆదానం_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    కొరకు (size_t i = 0; i < output_పేర్లు.size(); i++) {
        స్వయంచాలక output_tensor = infer_request->get_output_tensor(i);
        స్థిరరాశి స్వయంచాలక & model_outputs = decoder->get_model_outputs();
        స్వయంచాలక model_output_it = model_outputs.find(output_పేర్లు[i]);
        ఒకవేళ (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_పేర్లు[i], output_tensor, output_tensor.దత్తాంశం());
            }
            కొనసాగు;
        }
        స్వయంచాలక * ggml_tensor = model_output_it->second;
        std::జ్ఞాపకం_నకలు(ggml_tensor->దత్తాంశం, output_tensor.దత్తాంశం(), output_tensor.get_byte_కొలత());
    }
    తిరిగి GGML_స్థితి_SUCCESS;
}

గణన ggml_స్థితి ov_లేఖాచిత్రం_compute_dynamic(ggml_cgraph * cgraph, స్థిరరాశి std::shared_ptr<ov_runtime_context> & r_ctx) {
    స్వయంచాలక & core = ov_singleton_core();
    స్థిరరాశి స్వయంచాలక & config = ggml_openvino_get_compile_config();
    స్థిరరాశి స్వయంచాలక & device = r_ctx->device;
    స్థిర స్వయంచాలక is_static = అబద్ధం;

    స్థిర స్థిరరాశి తార్కిక cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    స్వయంచాలక ప్రారంభం_సమయం = ggml_సమయం_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    స్థిరరాశి int64_t cache_కీలకం_ప్రారంభం_సమయం = ggml_సమయం_us();
    లేఖాచిత్రం_కీలకం కీలకం(cgraph);
    స్థిరరాశి int64_t cache_కీలకం_compute_సమయం = ggml_సమయం_us() - cache_కీలకం_ప్రారంభం_సమయం;
    int64_t cache_lookup_సమయం = 0;
    తార్కిక కీలకం_చూసిన = అబద్ధం;
    ఒకవేళ (!cache_disabled) {
        స్థిరరాశి int64_t cache_lookup_ప్రారంభం_సమయం = ggml_సమయం_us();
        std::lock_guard<std::mutex> map_తాళం(r_ctx->ctx_mutex);
        కీలకం_చూసిన = r_ctx->decoder_cache.find(కీలకం) != r_ctx->decoder_cache.end();
        cache_lookup_సమయం += ggml_సమయం_us() - cache_lookup_ప్రారంభం_సమయం;
    }

    స్థిరరాశి int64_t model_విభజించు_check_ప్రారంభం_సమయం = ggml_సమయం_us();
    తార్కిక model_is_splitted = కీలకం_చూసిన ? అబద్ధం : is_model_splitted(cgraph);
    స్థిరరాశి int64_t model_విభజించు_check_సమయం = ggml_సమయం_us() - model_విభజించు_check_ప్రారంభం_సమయం;

    ఒకవేళ (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        తిరిగి GGML_స్థితి_FAILED;
    }
    ఒకవేళ (is_naive(cgraph)) {
        ఒకవేళ (!model_is_splitted) {
            తిరిగి naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    స్థిరరాశి int64_t లేఖాచిత్రం_param_compute_ప్రారంభం_సమయం = ggml_సమయం_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    స్థిరరాశి int64_t లేఖాచిత్రం_param_compute_సమయం = ggml_సమయం_us() - లేఖాచిత్రం_param_compute_ప్రారంభం_సమయం;

    స్థిరరాశి తార్కిక cache_enabled = !model_is_splitted && !cache_disabled;
    స్థిరరాశి తార్కిక stateful_recurrent = r_ctx->stateful && m_params.state_కొలత >= 0;
    స్థిరరాశి తార్కిక stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    ఒకవేళ (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_చురుకైన > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        తిరిగి GGML_స్థితి_FAILED;
    }
    ఒకవేళ (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        తిరిగి GGML_స్థితి_FAILED;
    }
    తార్కిక cache_తగిలింది = అబద్ధం;

    int64_t decoder_చివర_సమయం;
    int64_t conversion_చివర_సమయం;
    int64_t compile_చివర_సమయం;
    int64_t infer_చివర_సమయం;
    int64_t ov_raw_infer_ప్రారంభం;

    {
        std::shared_ptr<decoder_runtime_ctx> నమోదు_2;
        ModelParams old_m_params;

        స్థిరరాశి int64_t cache_lookup_ప్రారంభం_సమయం = ggml_సమయం_us();
        ఒకవేళ (cache_enabled) {
            std::lock_guard<std::mutex> map_తాళం(r_ctx->ctx_mutex);
            స్వయంచాలక it = r_ctx->decoder_cache.find(కీలకం);
            cache_తగిలింది = it != r_ctx->decoder_cache.end();
            ఒకవేళ (cache_తగిలింది) {
                నమోదు_2 = it->second;
            } లేకపోతే {
                r_ctx->clear_caches_locked();
                స్వయంచాలక mutex = std::make_shared<std::mutex>();
                నమోదు_2 = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[కీలకం] = నమోదు_2;
            }
        } లేకపోతే {
            స్వయంచాలక mutex = std::make_shared<std::mutex>();
            నమోదు_2 = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_తగిలింది = అబద్ధం;
        }
        cache_lookup_సమయం += ggml_సమయం_us() - cache_lookup_ప్రారంభం_సమయం;

        std::lock_guard<std::mutex> తాళం(*(నమోదు_2->mutex));
        cache_తగిలింది = cache_తగిలింది && నమోదు_2->ptr && r_ctx->infer_request_cache.count(కీలకం) != 0;

        ఒకవేళ (cache_తగిలింది) {
            ggml_decoder = నమోదు_2->ptr;
            old_m_params = ggml_decoder->get_model_params();
            ఒకవేళ (!ggml_decoder->is_splited_model()) {
                cache_తగిలింది = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_ఆదానం_పేర్లు;
        std::vector<std::string> ov_output_పేర్లు;

        ఒకవేళ (cache_తగిలింది) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->సమితి_compute_params(c_params);
            ggml_decoder->సమితి_model_params(m_params);
            ఒకవేళ (old_m_params.kv_బఫరు_మారింది(m_params) || !ggml_decoder->is_bound_గమ్యం(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->కూడు_extra_inputs();
            {
                std::lock_guard<std::mutex> map_తాళం(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(కీలకం);
                ov_ఆదానం_పేర్లు = r_ctx->ov_ఆదానం_పేర్లు_cache.at(కీలకం);
                ov_output_పేర్లు = r_ctx->ov_output_పేర్లు_cache.at(కీలకం);
            }

            ఒకవేళ (stateful_kv_only) {
                స్థిరరాశి స్వయంచాలక * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_దత్తాంశం = (int32_t *) inp_pos->దత్తాంశం;
                స్వయంచాలక pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                ఒకవేళ (pos_దత్తాంశం[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_కొలత = pos_shape[3];
                } లేకపోతే ఒకవేళ (r_ctx->stateful_kv_కొలత == static_cast<size_t>(pos_దత్తాంశం[0])) {
                    r_ctx->stateful_kv_కొలత += pos_shape[3];
                } లేకపోతే {
                    స్థిరరాశి size_t pos_begin = static_cast<size_t>(pos_దత్తాంశం[0]);
                    స్థిరరాశి తార్కిక refill = pos_begin > r_ctx->stateful_kv_కొలత;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    ఒకవేళ (refill && !ggml_decoder->get_model_params().swa_layers.ఖాళీలు()) {
                        స్థిరరాశి పూర్ణాంకం n_swa = ggml_decoder->get_compute_params().swa_కిటికీ;
                        ఒకవేళ (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_కొలత, n_swa);
                            తిరిగి GGML_స్థితి_FAILED;
                        }
                    }

                    స్థిరరాశి తార్కిక relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    స్వయంచాలక స్థితులు = infer_request->query_state();
                    కొరకు (స్వయంచాలక state : స్థితులు) {
                        స్వయంచాలక state_tensor = state.get_state();
                        స్వయంచాలక state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        ఒకవేళ (స్వయంచాలక it = r_ctx->kv_state_ఆదానం_name_map.find(state.get_name());
                            it != r_ctx->kv_state_ఆదానం_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        పూర్ణాంకం n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        ఒకవేళ (స్వయంచాలక layer = వెలికితీయి_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        స్థిరరాశి తార్కిక relayout_this_state = relayout_enabled;
                        స్థిరరాశి size_t seq_axis = relayout_this_state ? 2 : 1;
                        స్థిరరాశి size_t తల_axis = seq_axis == 2 ? 1 : 2;

                        ఒకవేళ (refill) {
                            ఒకవేళ (state_name.ఖాళీలు()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                తిరిగి GGML_స్థితి_FAILED;
                            }
                            స్వయంచాలక kv_tensor = get_ov_ఆదానం_tensor(ggml_decoder, state_name);
                            ఒకవేళ (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_అడ్డువరుసలు_గమ్యం_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } లేకపోతే {
                                ov::Shape refill_shape(4);
                                refill_shape[0] = state_tensor_shape[0];
                                refill_shape[seq_axis] = kv_tensor.get_shape()[2];
                                refill_shape[తల_axis] = state_tensor_shape[తల_axis];
                                refill_shape[3] = state_tensor_shape[3];
                                kv_tensor.సమితి_shape(refill_shape);
                                state_tensor = kv_tensor;
                            }
                            state_tensor_shape = state_tensor.get_shape();
                        }
                        // Only ever shrink to a prefix the source really has. Slicing past it used to
                        // surface as a bare ov::Exception from the ROI constructor.
                        ఒకవేళ (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            తిరిగి GGML_స్థితి_FAILED;
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end(state_tensor_shape.begin(), state_tensor_shape.end());
                        end[seq_axis] = pos_begin;
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.సమితి_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_కొలత = pos_begin + pos_shape[3];
                }
            }

            decoder_చివర_సమయం = ggml_సమయం_us();
            conversion_చివర_సమయం = decoder_చివర_సమయం;
            compile_చివర_సమయం = decoder_చివర_సమయం;
        } లేకపోతే {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            స్వయంచాలక shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_తాళం(shared_cache->mutex);
            స్వయంచాలక బరువు_పేర్లు = get_బరువు_పేర్లు(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, బరువు_పేర్లు,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            స్థిరరాశి std::string model_cache_dir = ggml_openvino_model_cache_dir();
            స్థిరరాశి std::string exact_కీలకం = cache_enabled ? compiled_లేఖాచిత్రం_కీలకం(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_మార్గం, manifest_మార్గం;
            ఒకవేళ (!model_cache_dir.ఖాళీలు() && !model_is_splitted) {
                స్థిరరాశి uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/నిజం, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_లేఖాచిత్రం_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/నిజం, m_params.rope_params, 16, extra_cfg,
                                                    compiled_లేఖాచిత్రం_కీలకం(cgraph, *ggml_decoder, device, 0, నిజం));
                blob_మార్గం = ggml_openvino_model_cache_blob_మార్గం(model_cache_dir, model_fp);
                manifest_మార్గం = ggml_openvino_model_cache_manifest_మార్గం(model_cache_dir, model_fp);
            }
            std::string shared_కీలకం =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_కీలకం) : "";
            ov::CompiledModel shared_model;
            తార్కిక imported = అబద్ధం;
            స్వయంచాలక shared_it = shared_cache->graphs.find(shared_కీలకం);
            ఒకవేళ (!shared_కీలకం.ఖాళీలు() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_లేఖాచిత్రం(shared_it->second.decode, shared_it->second.ఆదానం_పేర్లు,
                                                               shared_it->second.output_పేర్లు, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_ఆదానం_పేర్లు = shared_it->second.ఆదానం_పేర్లు;
                ov_output_పేర్లు = shared_it->second.output_పేర్లు;
                imported = నిజం;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } లేకపోతే ఒకవేళ (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_కీలకం = exact_కీలకం;
                shared_it = shared_cache->graphs.find(shared_కీలకం);
                ఒకవేళ (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_ఆదానం_పేర్లు = shared_it->second.ఆదానం_పేర్లు;
                    ov_output_పేర్లు = shared_it->second.output_పేర్లు;
                    imported = నిజం;
                }
                model_fp = exact_fp;
                blob_మార్గం = ggml_openvino_model_cache_blob_మార్గం(model_cache_dir, model_fp);
                manifest_మార్గం = ggml_openvino_model_cache_manifest_మార్గం(model_cache_dir, model_fp);
            }
            ఒకవేళ (cache_enabled) {
                std::lock_guard<std::mutex> map_తాళం(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(కీలకం);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            ఒకవేళ (!model_cache_dir.ఖాళీలు()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_బహుళకం.name()] = ov::Cache_బహుళకం::OPTIMIZE_వేగం;
            }
            ఒకవేళ (!imported && !model_cache_dir.ఖాళీలు() && !model_is_splitted) {
                స్థిరరాశి uint64_t preferred_fp = model_fp;
                తార్కిక dynamic_incompatible = preferred_fp == exact_fp;
                కొరకు (పూర్ణాంకం candidate = 0; candidate < 2; ++candidate) {
                    ఒకవేళ (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        ఆపు;
                    }
                    స్థిరరాశి uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    స్థిరరాశి std::string candidate_blob =
                        ggml_openvino_model_cache_blob_మార్గం(model_cache_dir, candidate_fp);
                    స్థిరరాశి std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_మార్గం(model_cache_dir, candidate_fp);
                    std::ifstream blob_లోపల(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    ఒకవేళ (!blob_లోపల.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        కొనసాగు;
                    }
                    స్థిరరాశి int64_t import_ప్రారంభం = ggml_సమయం_us();
                    ప్రయత్నించు {
                        ov::CompiledModel cm;
                        స్వయంచాలక remote_context = ggml_openvino_get_remote_context();
                        ఒకవేళ (remote_context.has_value()) {
                            cm = core.import_model(blob_లోపల, remote_context.value(), mc_config);
                        } లేకపోతే {
                            cm = core.import_model(blob_లోపల, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        ఒకవేళ (!saved_inputs.ఖాళీలు() || !saved_outputs.ఖాళీలు()) {
                            ov_ఆదానం_పేర్లు = std::move(saved_inputs);
                            ov_output_పేర్లు = std::move(saved_outputs);
                        } లేకపోతే {
                            కొరకు (స్థిరరాశి స్వయంచాలక & p : cm.inputs()) {
                                ov_ఆదానం_పేర్లు.push_back(p.get_కణుపు()->get_friendly_name());
                            }
                            కొరకు (స్థిరరాశి స్వయంచాలక & o : cm.outputs()) {
                                ov_output_పేర్లు.push_back(o.get_కణుపు()->get_friendly_name());
                            }
                        }
                        ఒకవేళ (!compiled_model_matches_లేఖాచిత్రం(cm, ov_ఆదానం_పేర్లు, ov_output_పేర్లు, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_ఆదానం_పేర్లు.clear();
                            ov_output_పేర్లు.clear();
                            కొనసాగు;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        నమోదు_2->ptr = ggml_decoder;
                        imported = నిజం;
                        model_fp = candidate_fp;
                        blob_మార్గం = candidate_blob;
                        manifest_మార్గం = candidate_manifest;
                        ఒకవేళ (candidate_fp == exact_fp) {
                            shared_కీలకం = exact_కీలకం;
                            shared_it = shared_cache->graphs.find(shared_కీలకం);
                        }
                        ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_సమయం_us() - import_ప్రారంభం) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } పట్టుకో (స్థిరరాశి std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = అబద్ధం;
                        ov_ఆదానం_పేర్లు.clear();
                        ov_output_పేర్లు.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                ఒకవేళ (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_కీలకం = exact_కీలకం;
                    shared_it = shared_cache->graphs.find(shared_కీలకం);
                    blob_మార్గం = ggml_openvino_model_cache_blob_మార్గం(model_cache_dir, model_fp);
                    manifest_మార్గం = ggml_openvino_model_cache_manifest_మార్గం(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            ఒకవేళ (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_మార్గం.c_str());
                తిరిగి GGML_స్థితి_FAILED;
            }
            ఒకవేళ (!imported && ggml_openvino_బరువు_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                తిరిగి GGML_స్థితి_FAILED;
            }
            ఒకవేళ (imported) {
                decoder_చివర_సమయం = conversion_చివర_సమయం = compile_చివర_సమయం = ggml_సమయం_us();
            } లేకపోతే {
                స్వయంచాలక model_weights = GgmlOvDecoder::create_బరువు_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_చివర_సమయం = ggml_సమయం_us();

                స్వయంచాలక ఆదానం_model = std::make_shared<ov::frontend::ggml::ఆదానం_Model>(ggml_decoder);
                model = ov::frontend::ggml::Front_చివర::convert(ఆదానం_model);
                ggml_decoder->clear_model_weights();
                conversion_చివర_సమయం = ggml_సమయం_us();

                ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    అక్షరం timestamped_filename[64];
                    స్వయంచాలక timestamp = (దీర్ఘ దీర్ఘ) ggml_సమయం_us();
                    snprintf(timestamped_filename, పరిమాణం(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                స్థిరరాశి ov::AnyMap & compile_config = model_cache_dir.ఖాళీలు() ? config : mc_config;
                ov::CompiledModel compiled_model;
                స్వయంచాలక remote_context = ggml_openvino_get_remote_context();
                ఒకవేళ (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } లేకపోతే {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_చివర_సమయం = ggml_సమయం_us();

                కొరకు (స్థిరరాశి స్వయంచాలక & ov_param : model->get_parameters()) {
                    ov_ఆదానం_పేర్లు.push_back(ov_param->get_friendly_name());
                }
                కొరకు (స్థిరరాశి స్వయంచాలక & ov_output : model->get_results()) {
                    ov_output_పేర్లు.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                ఒకవేళ (!model_cache_dir.ఖాళీలు() && !model_is_splitted && model_fp != 0) {
                    ప్రయత్నించు {
                        స్థిరరాశి std::string blob_tmp = ggml_openvino_model_cache_తాత్కాలిక_మార్గం(blob_మార్గం);
                        స్థిరరాశి std::string manifest_tmp = ggml_openvino_model_cache_తాత్కాలిక_మార్గం(manifest_మార్గం);
                        ఒకవేళ (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_ఆదానం_పేర్లు,
                                                                     ov_output_పేర్లు)) {
                            std::ofstream blob_బహిర్గతం(blob_tmp, std::ios::binary | std::ios::trunc);
                            ఒకవేళ (blob_బహిర్గతం.is_open()) {
                                compiled_model.export_model(blob_బహిర్గతం);
                                blob_బహిర్గతం.close();
                                ఒకవేళ (blob_బహిర్గతం.good()) {
                                    ఒకవేళ (std::rename(blob_tmp.c_str(), blob_మార్గం.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_మార్గం.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_మార్గం.c_str());
                                    } లేకపోతే {
                                        std::remove(blob_tmp.c_str());
                                        std::remove(manifest_tmp.c_str());
                                    }
                                } లేకపోతే {
                                    std::remove(blob_tmp.c_str());
                                    std::remove(manifest_tmp.c_str());
                                }
                            } లేకపోతే {
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    } పట్టుకో (స్థిరరాశి std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                నమోదు_2->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            నమోదు_2->ptr = ggml_decoder;
            ఒకవేళ (!shared_కీలకం.ఖాళీలు() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_కీలకం,
                                             ov_compiled_లేఖాచిత్రం{shared_model, {}, ov_ఆదానం_పేర్లు, ov_output_పేర్లు});
            }
            ఒకవేళ (cache_enabled) {
                std::lock_guard<std::mutex> map_తాళం(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[కీలకం] = infer_request;
                r_ctx->ov_ఆదానం_పేర్లు_cache[కీలకం] = ov_ఆదానం_పేర్లు;
                r_ctx->ov_output_పేర్లు_cache[కీలకం] = ov_output_పేర్లు;
            }

            ఒకవేళ (stateful_kv_only && cache_enabled) {
                స్థిరరాశి స్వయంచాలక * inp_pos = get_inp_pos_tensor(cgraph);
                స్వయంచాలక pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                స్థిరరాశి int32_t pos_begin = ((int32_t *) inp_pos->దత్తాంశం)[0];
                ఒకవేళ (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    తిరిగి GGML_స్థితి_FAILED;
                }
                r_ctx->stateful_kv_కొలత = pos_shape[3];
                స్థిరరాశి స్వయంచాలక kv_param_res_పేర్లు = ggml_decoder->get_kv_param_res_పేర్లు();
                కొరకు (స్థిరరాశి స్వయంచాలక & pair : kv_param_res_పేర్లు) {
                    r_ctx->kv_state_ఆదానం_name_map[pair.మొదటి + pair.second] = pair.మొదటి;
                }
            }
        }

        ఒకవేళ (stateful_recurrent) {
            స్థిరరాశి స్వయంచాలక * inp_pos = get_inp_pos_tensor(cgraph);
            స్థిరరాశి int32_t pos_begin = static_cast<స్థిరరాశి int32_t *>(inp_pos->దత్తాంశం)[0];
            ఒకవేళ (pos_begin == 0) {
                infer_request->reset_state();
            } లేకపోతే ఒకవేళ (!cache_తగిలింది || old_m_params.kv_బఫరు_మారింది(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_కొలత) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                తిరిగి GGML_స్థితి_FAILED;
            }
        } లేకపోతే {
            reset_single_స్థానం_recurrent_cache(cgraph, m_params, c_params);
        }

        కొరకు (size_t i = 0; i < ov_ఆదానం_పేర్లు.size(); i++) {
            స్థిరరాశి స్వయంచాలక & param_name = ov_ఆదానం_పేర్లు[i];
            స్వయంచాలక ఆదానం_tensor = get_ov_ఆదానం_tensor(ggml_decoder, param_name);
            infer_request->సమితి_ఆదానం_tensor(i, ఆదానం_tensor);

            ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_ఆదానం_tensor_info(param_name, ఆదానం_tensor);
            }
        }

        కొరకు (size_t i = 0; i < ov_output_పేర్లు.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            స్థిరరాశి స్వయంచాలక & model_outputs = ggml_decoder->get_model_outputs();
            స్వయంచాలక model_output_it = model_outputs.find(ov_output_పేర్లు[i]);
            ఒకవేళ (model_output_it == model_outputs.end()) {
                కొనసాగు;
            }
            స్వయంచాలక * ggml_tensor = model_output_it->second;
            ఒకవేళ (ggml_nbytes(ggml_tensor) == 0) {
                కొనసాగు;
            }
            స్వయంచాలక output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->సమితి_output_tensor(i, output_tensor);
        }

        ov_raw_infer_ప్రారంభం = ggml_సమయం_us();
        infer_request->infer();
        infer_చివర_సమయం = ggml_సమయం_us();
        ఒకవేళ (stateful_recurrent) {
            స్థిరరాశి స్వయంచాలక * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_కొలత =
                static_cast<స్థిరరాశి int32_t *>(inp_pos->దత్తాంశం)[0] + get_inp_pos_n_టోకెన్లు(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            కొరకు (size_t i = 0; i < ov_output_పేర్లు.size(); i++) {
                స్థిరరాశి స్వయంచాలక output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_పేర్లు[i], output_tensor, output_tensor.దత్తాంశం());
            }
        }

        ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_చివర_సమయం - ప్రారంభం_సమయం) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_కీలకం_compute_సమయం / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_విభజించు_check_సమయం / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", లేఖాచిత్రం_param_compute_సమయం / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_సమయం / 1000.0);
            ఒకవేళ (!cache_తగిలింది) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_చివర_సమయం - decoder_చివర_సమయం) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_చివర_సమయం - conversion_చివర_సమయం) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_చివర_సమయం - compile_చివర_సమయం) / 1000.0);
            GGML_LOG_INFO("    - OV raw infer time: %.3f ms \n", (infer_చివర_సమయం - ov_raw_infer_ప్రారంభం) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. Release only while holding the compilation mutex so another context cannot
    // be reading host weights during conversion/compilation. Pin the shared compiled
    // models across backend teardown; a later context can create its own request without
    // reading the dropped pages. A new, uncached graph still fails fast above.
    ఒకవేళ (cache_తగిలింది && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_తాళం(r_ctx->compiled_cache->mutex);
        ఒకవేళ (!ggml_openvino_బరువు_buffers_released()) {
            ggml_openvino_release_బరువు_buffers();
        }
    }

    తిరిగి GGML_స్థితి_SUCCESS;
}

ov::AnyMap without_npuw(స్థిరరాశి ov::AnyMap & config) {
    ov::AnyMap బహిర్గతం;
    కొరకు (స్థిరరాశి స్వయంచాలక & kv : config) {
        ఒకవేళ (kv.మొదటి.rfind("NPUW", 0) == 0 || kv.మొదటి == "NPU_USE_NPUW") {
            కొనసాగు;
        }
        బహిర్గతం.insert(kv);
    }
    తిరిగి బహిర్గతం;
}

గణన ggml_స్థితి ov_లేఖాచిత్రం_compute_static(ggml_cgraph * cgraph, స్థిరరాశి std::shared_ptr<ov_runtime_context> & r_ctx) {
    స్వయంచాలక & core = ov_singleton_core();

    స్వయంచాలక get_prefill_chunk_కొలత = [] {
        స్థిర స్థిరరాశి పూర్ణాంకం chunk_కొలత = []() {
            పూర్ణాంకం env_prefill_chunk_కొలత = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            తిరిగి env_prefill_chunk_కొలత > 0 ? env_prefill_chunk_కొలత : 256;
        }();
        తిరిగి chunk_కొలత;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    స్థిర std::string device = ggml_openvino_get_device_name();
    స్థిర స్వయంచాలక is_static = నిజం;
    స్థిర స్వయంచాలక stateful = అబద్ధం;

    స్వయంచాలక prefill_chunk_కొలత = get_prefill_chunk_కొలత();
    స్థిరరాశి స్వయంచాలక & config = ggml_openvino_get_compile_config();

    ఒకవేళ (is_naive(cgraph)) {
        తిరిగి naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    స్వయంచాలక ప్రారంభం_సమయం = ggml_సమయం_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    స్థిరరాశి స్వయంచాలక * inp_pos = get_inp_pos_tensor(cgraph);
    స్థిరరాశి తార్కిక no_kv_cache = m_params.is_cacheless_attn;
    స్థిరరాశి స్వయంచాలక is_prefill = no_kv_cache ? నిజం : get_is_prefill(cgraph, inp_pos);
    స్థిరరాశి ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    ఒకవేళ (m_params.n_heads_kv == -1) {
        prefill_chunk_కొలత = inp_pos->ne[0];
    }
    లేఖాచిత్రం_కీలకం కీలకం(cgraph);
    స్థిర స్థిరరాశి తార్కిక cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    తార్కిక cache_తగిలింది = అబద్ధం;

    int64_t decoder_చివర_సమయం;
    int64_t conversion_చివర_సమయం;
    int64_t compile_చివర_సమయం;
    int64_t infer_చివర_సమయం;
    int64_t ov_raw_infer_ప్రారంభం;
    int64_t ov_raw_infer_మొత్తం = 0;

    std::shared_ptr<decoder_runtime_ctx> నమోదు_2;
    ModelParams old_m_params;

    ఒకవేళ (cache_enabled) {
        std::lock_guard<std::mutex> map_తాళం(r_ctx->ctx_mutex);
        స్వయంచాలక it = r_ctx->decoder_cache.find(కీలకం);
        cache_తగిలింది = it != r_ctx->decoder_cache.end();
        ఒకవేళ (cache_తగిలింది) {
            నమోదు_2 = it->second;
        } లేకపోతే {
            r_ctx->clear_caches_locked();
            స్వయంచాలక mutex = std::make_shared<std::mutex>();
            నమోదు_2 = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[కీలకం] = నమోదు_2;
        }
    } లేకపోతే {
        స్వయంచాలక mutex = std::make_shared<std::mutex>();
        నమోదు_2 = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_తగిలింది = అబద్ధం;
    }

    std::lock_guard<std::mutex> తాళం(*(నమోదు_2->mutex));
    cache_తగిలింది = cache_తగిలింది && నమోదు_2->ptr && r_ctx->infer_request_cache.count(కీలకం) != 0 &&
                r_ctx->infer_request_cache_prefill.count(కీలకం) != 0;

    ఒకవేళ (cache_తగిలింది) {
        ggml_decoder = నమోదు_2->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_తగిలింది = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_ఆదానం_పేర్లు_local;
    std::vector<std::string> ov_output_పేర్లు_local;

    ఒకవేళ (cache_తగిలింది) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->సమితి_model_params(m_params);
        ggml_decoder->సమితి_compute_params(c_params);
        ఒకవేళ (old_m_params.kv_బఫరు_మారింది(m_params) || !ggml_decoder->is_bound_గమ్యం(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->కూడు_extra_inputs();
        {
            std::lock_guard<std::mutex> map_తాళం(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(కీలకం) : r_ctx->infer_request_cache.at(కీలకం);
            ov_ఆదానం_పేర్లు_local = r_ctx->ov_ఆదానం_పేర్లు_cache.at(కీలకం);
            ov_output_పేర్లు_local = r_ctx->ov_output_పేర్లు_cache.at(కీలకం);
        }

        decoder_చివర_సమయం = ggml_సమయం_us();
        conversion_చివర_సమయం = decoder_చివర_సమయం;
        compile_చివర_సమయం = decoder_చివర_సమయం;
    } లేకపోతే {
        ఒకవేళ (cache_enabled) {
            std::lock_guard<std::mutex> map_తాళం(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(కీలకం);
            r_ctx->infer_request_cache_prefill.erase(కీలకం);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        స్వయంచాలక shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_తాళం(shared_cache->mutex);
        స్వయంచాలక బరువు_పేర్లు = get_బరువు_పేర్లు(cgraph);
        స్వయంచాలక local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, బరువు_పేర్లు, is_static,
                                                             stateful, అబద్ధం, is_prefill, prefill_chunk_కొలత);
        స్థిరరాశి std::string shared_కీలకం =
            cache_enabled ? compiled_లేఖాచిత్రం_కీలకం(cgraph, *local_decoder, device, prefill_chunk_కొలత) : "";
        స్వయంచాలక shared_it = shared_cache->graphs.find(shared_కీలకం);
        ఒకవేళ (!shared_కీలకం.ఖాళీలు() && shared_it != shared_cache->graphs.end()) {
            స్వయంచాలక & compiled = shared_it->second;
            స్వయంచాలక prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            స్వయంచాలక decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            నమోదు_2->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_ఆదానం_పేర్లు_local = compiled.ఆదానం_పేర్లు;
            ov_output_పేర్లు_local = compiled.output_పేర్లు;
            r_ctx->infer_request_cache_prefill[కీలకం] = prefill_request;
            r_ctx->infer_request_cache[కీలకం] = decode_request;
            r_ctx->ov_ఆదానం_పేర్లు_cache[కీలకం] = ov_ఆదానం_పేర్లు_local;
            r_ctx->ov_output_పేర్లు_cache[కీలకం] = ov_output_పేర్లు_local;
            decoder_చివర_సమయం = conversion_చివర_సమయం = compile_చివర_సమయం = ggml_సమయం_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } లేకపోతే {
            std::shared_ptr<ov::Model> model;
            స్వయంచాలక model_weights = GgmlOvDecoder::create_బరువు_nodes(cgraph);

            స్వయంచాలక ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, అబద్ధం, నిజం, prefill_chunk_కొలత);
            స్వయంచాలక ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, అబద్ధం, అబద్ధం, prefill_chunk_కొలత);
            decoder_చివర_సమయం = ggml_సమయం_us();

            స్థిరరాశి తార్కిక dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            స్థిరరాశి స్వయంచాలక dump_ir_timestamp = static_cast<దీర్ఘ దీర్ఘ>(ggml_సమయం_us());

            స్వయంచాలక build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          స్థిరరాశి std::shared_ptr<GgmlOvDecoder> & decoder, స్థిరరాశి అక్షరం * ట్యాగ్,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_చివర_సమయం, int64_t & local_compile_చివర_సమయం) {
                స్వయంచాలక ఆదానం_model = std::make_shared<ov::frontend::ggml::ఆదానం_Model>(decoder);
                model = ov::frontend::ggml::Front_చివర::convert(ఆదానం_model);
                decoder->clear_model_weights();
                local_conversion_చివర_సమయం = ggml_సమయం_us();

                ఒకవేళ (dump_ir) {
                    అక్షరం timestamped_filename[64];
                    snprintf(timestamped_filename, పరిమాణం(timestamped_filename), "model_%s_%lld.xml", ట్యాగ్,
                             dump_ir_timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                compiled_model = core.compile_model(model, device, compile_config);
                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                local_compile_చివర_సమయం = ggml_సమయం_us();
            };
            std::shared_ptr<ov::Model> model_prefill;
            std::shared_ptr<ov::Model> model_decode;
            ov::CompiledModel compiled_model_prefill;
            ov::CompiledModel compiled_model_decode;
            std::shared_ptr<ov::InferRequest> infer_request_prefill;
            std::shared_ptr<ov::InferRequest> infer_request_decode;
            int64_t prefill_conversion_చివర_సమయం;
            int64_t decode_conversion_చివర_సమయం;
            int64_t prefill_compile_చివర_సమయం;
            int64_t decode_compile_చివర_సమయం;
            build_static_model(ggml_decoder_prefill, "prefill", model_prefill, compiled_model_prefill,
                               infer_request_prefill, prefill_conversion_చివర_సమయం, prefill_compile_చివర_సమయం);
            ఒకవేళ (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_చివర_సమయం = prefill_conversion_చివర_సమయం;
                decode_compile_చివర_సమయం = prefill_compile_చివర_సమయం;
            } లేకపోతే {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_చివర_సమయం, decode_compile_చివర_సమయం);
            }
            conversion_చివర_సమయం = std::max(prefill_conversion_చివర_సమయం, decode_conversion_చివర_సమయం);
            compile_చివర_సమయం = std::max(prefill_compile_చివర_సమయం, decode_compile_చివర_సమయం);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            నమోదు_2->ptr = ggml_decoder;

            కొరకు (స్థిరరాశి స్వయంచాలక & ov_param : model->get_parameters()) {
                ov_ఆదానం_పేర్లు_local.push_back(ov_param->get_friendly_name());
            }
            కొరకు (స్థిరరాశి స్వయంచాలక & ov_output : model->get_results()) {
                ov_output_పేర్లు_local.push_back(ov_output->get_friendly_name());
            }

            ఒకవేళ (!shared_కీలకం.ఖాళీలు()) {
                shared_cache->graphs.emplace(
                    shared_కీలకం, ov_compiled_లేఖాచిత్రం{compiled_model_decode, compiled_model_prefill, ov_ఆదానం_పేర్లు_local,
                                                  ov_output_పేర్లు_local});
            }

            ఒకవేళ (cache_enabled) {
                std::lock_guard<std::mutex> map_తాళం(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[కీలకం] = infer_request_prefill;
                r_ctx->infer_request_cache[కీలకం] = infer_request_decode;
                r_ctx->ov_ఆదానం_పేర్లు_cache[కీలకం] = ov_ఆదానం_పేర్లు_local;
                r_ctx->ov_output_పేర్లు_cache[కీలకం] = ov_output_పేర్లు_local;
            }
        }
    }

    reset_single_స్థానం_recurrent_cache(cgraph, m_params, c_params);

    ఒకవేళ (is_prefill) {
        స్వయంచాలక inp_len = get_inp_pos_n_టోకెన్లు(cgraph, inp_pos);
        కొరకు (పూర్ణాంకం chunk_సూచీ = 0; chunk_సూచీ * prefill_chunk_కొలత < inp_len; chunk_సూచీ++) {
            కొరకు (size_t i = 0; i < ov_ఆదానం_పేర్లు_local.size(); i++) {
                స్థిరరాశి స్వయంచాలక & param_name = ov_ఆదానం_పేర్లు_local[i];
                స్వయంచాలక ఆదానం_tensor = get_ov_ఆదానం_tensor_static_prefill(ggml_decoder, param_name, chunk_సూచీ);
                infer_request->సమితి_ఆదానం_tensor(i, ఆదానం_tensor);

                ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    స్థిరరాశి స్వయంచాలక ఆదానం_tensor = infer_request->get_ఆదానం_tensor(i);
                    print_ఆదానం_tensor_info(param_name, ఆదానం_tensor);
                }
            }

            కొరకు (size_t i = 0; i < ov_output_పేర్లు_local.size(); i++) {
                స్థిరరాశి స్వయంచాలక & model_outputs = ggml_decoder->get_model_outputs();
                స్వయంచాలక model_output_it = model_outputs.find(ov_output_పేర్లు_local[i]);
                ఒకవేళ (model_output_it == model_outputs.end()) {
                    కొనసాగు;
                }
                స్వయంచాలక * ggml_tensor = model_output_it->second;
                ఒకవేళ (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    కొనసాగు;
                }
                స్వయంచాలక output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->సమితి_output_tensor(i, output_tensor);
            }

            ov_raw_infer_ప్రారంభం = ggml_సమయం_us();
            infer_request->infer();
            ov_raw_infer_మొత్తం += ggml_సమయం_us() - ov_raw_infer_ప్రారంభం;

            ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                కొరకు (size_t i = 0; i < ov_output_పేర్లు_local.size(); i++) {
                    స్థిరరాశి స్వయంచాలక output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_పేర్లు_local[i], output_tensor, output_tensor.దత్తాంశం());
                }
            }
        }
        infer_చివర_సమయం = ggml_సమయం_us();
    } లేకపోతే {
        కొరకు (size_t i = 0; i < ov_ఆదానం_పేర్లు_local.size(); i++) {
            స్థిరరాశి స్వయంచాలక & param_name = ov_ఆదానం_పేర్లు_local[i];
            స్వయంచాలక ఆదానం_tensor = get_ov_ఆదానం_tensor_static_decode(ggml_decoder, param_name);
            infer_request->సమితి_ఆదానం_tensor(i, ఆదానం_tensor);

            ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                స్థిరరాశి స్వయంచాలక ఆదానం_tensor = infer_request->get_ఆదానం_tensor(i);
                print_ఆదానం_tensor_info(param_name, ఆదానం_tensor);
            }
        }

        కొరకు (size_t i = 0; i < ov_output_పేర్లు_local.size(); i++) {
            స్థిరరాశి స్వయంచాలక & model_outputs = ggml_decoder->get_model_outputs();
            స్వయంచాలక model_output_it = model_outputs.find(ov_output_పేర్లు_local[i]);
            ఒకవేళ (model_output_it == model_outputs.end()) {
                కొనసాగు;
            }
            స్వయంచాలక * ggml_tensor = model_output_it->second;
            ఒకవేళ (ggml_nbytes(ggml_tensor) == 0) {
                కొనసాగు;
            }
            స్వయంచాలక output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->సమితి_output_tensor(i, output_tensor);
        }

        ov_raw_infer_ప్రారంభం = ggml_సమయం_us();
        infer_request->infer();
        infer_చివర_సమయం = ggml_సమయం_us();
        ov_raw_infer_మొత్తం = infer_చివర_సమయం - ov_raw_infer_ప్రారంభం;

        ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            కొరకు (size_t i = 0; i < ov_output_పేర్లు_local.size(); i++) {
                స్థిరరాశి స్వయంచాలక output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_పేర్లు_local[i], output_tensor, output_tensor.దత్తాంశం());
            }
        }
    }

    ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_చివర_సమయం - ప్రారంభం_సమయం) / 1000.0);
        ఒకవేళ (!cache_తగిలింది) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_చివర_సమయం - decoder_చివర_సమయం) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_చివర_సమయం - conversion_చివర_సమయం) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_చివర_సమయం - compile_చివర_సమయం) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_మొత్తం / 1000.0);
    }

    తిరిగి GGML_స్థితి_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
స్థిర ggml_cgraph * get_compute_లేఖాచిత్రం(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    స్వయంచాలక is_skipped = [](స్థిరరాశి ggml_tensor * కణుపు) {
        తిరిగి కణుపు->op != GGML_OP_NONE && !(కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    ఒకవేళ (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        తిరిగి cgraph;
    }
    స్వయంచాలక & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](స్థిరరాశి ggml_tensor * కణుపు) { తిరిగి !is_skipped(కణుపు); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.దత్తాంశం();
    compute.graph.n_nodes = (పూర్ణాంకం) compute.nodes.size();
    తిరిగి &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
గణన ggml_స్థితి ov_లేఖాచిత్రం_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    ప్రయత్నించు {
        ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        స్థిరరాశి స్వయంచాలక is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != శూన్యసూచిక);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_తాళం(r_ctx->execution_mutex);
        cgraph = get_compute_లేఖాచిత్రం(cgraph, *r_ctx);

        తిరిగి is_static ? ov_లేఖాచిత్రం_compute_static(cgraph, r_ctx) : ov_లేఖాచిత్రం_compute_dynamic(cgraph, r_ctx);
    } పట్టుకో (స్థిరరాశి ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        తిరిగి GGML_స్థితి_FAILED;
    } పట్టుకో (స్థిరరాశి std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        తిరిగి GGML_స్థితి_FAILED;
    } పట్టుకో (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        తిరిగి GGML_స్థితి_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
తార్కిక is_model_splitted(ggml_cgraph * cgraph) {
    స్థిర స్థిరరాశి తార్కిక fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    ఒకవేళ (!fallback_enabled) {
        తిరిగి అబద్ధం;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    ఒకవేళ (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        తిరిగి అబద్ధం;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * కణుపు = cgraph->nodes[i];
        పూర్ణాంకం use_లెక్క = cgraph->use_counts[ggml_hash_కనుగొను(&cgraph->సందర్శించిన_hash_సమితి, కణుపు)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        ఒకవేళ ((cgraph->n_nodes <= 1 && use_లెక్క == 0) ||
            (cgraph->n_nodes <= 1 && కణుపు->op == GGML_OP_VIEW && use_లెక్క == 1 && కణుపు->src[0] != శూన్యసూచిక &&
             కణుపు->src[0]->op == GGML_OP_NONE)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_క్రమమార్పు)) {
            తిరిగి అబద్ధం;
        }
        పూర్ణాంకం ఆదానం_use_లెక్క = 0;
        కొరకు (పూర్ణాంకం j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_కణుపు = cgraph->nodes[j];
            కొరకు (పూర్ణాంకం k = 0; k < GGML_MAX_SRC; k++) {
                ఒకవేళ (other_కణుపు->src[k] == కణుపు) {
                    ఆదానం_use_లెక్క++;
                }
            }
        }
        ఒకవేళ (use_లెక్క != ఆదానం_use_లెక్క && కణుపు->op != GGML_OP_NONE) {
            తిరిగి నిజం;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    ఒకవేళ (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_బరువు_పేర్లు(cgraph);
    } లేకపోతే {
        కొరకు (స్థిరరాశి స్వయంచాలక & kv : GgmlOvDecoder::create_బరువు_nodes(cgraph, నిజం)) {
            model_weights.insert(kv.మొదటి);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * కణుపు = cgraph->nodes[i];
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = కణుపు->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            ఒకవేళ (src != శూన్యసూచిక && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.ఖాళీలు() == అబద్ధం &&
                model_leafs.find(src) == model_leafs.end()) {
                ఒకవేళ (GgmlOvDecoder::is_inp_tok(src, కణుపు)) {
                    తిరిగి అబద్ధం;
                }
                తిరిగి నిజం;
            }
        }
    }
    తిరిగి అబద్ధం;
}

తార్కిక is_naive(ggml_cgraph * cgraph) {
    constexpr పూర్ణాంకం naive_లేఖాచిత్రం_కొలత_పరిమితి = 20;
    పూర్ణాంకం count = 0;
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        ఒకవేళ (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    తిరిగి count < naive_లేఖాచిత్రం_కొలత_పరిమితి;
}

size_t checksum(స్థిరరాశి శూన్య * దత్తాంశం, size_t size) {
    స్థిరరాశి uint8_t * బైట్లు = static_cast<స్థిరరాశి uint8_t *>(దత్తాంశం);
    size_t కూడిక = 0;
    కొరకు (size_t i = 0; i < size; ++i) {
        కూడిక += (uint8_t) i;
        కూడిక += బైట్లు[i];
    }
    తిరిగి కూడిక;
}

తార్కిక save_ggml_tensor_దత్తాంశం_గమ్యం_txt(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి std::string & file_మార్గం) {
    ఒకవేళ (tensor == శూన్యసూచిక || tensor->దత్తాంశం == శూన్యసూచిక) {
        తిరిగి అబద్ధం;
    }

    std::ofstream బహిర్గతం(file_మార్గం);
    ఒకవేళ (!బహిర్గతం.is_open()) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి size_t n = ggml_nelements(tensor);
    బహిర్గతం << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    ఎంపిక (tensor->type) {
    స్థితి GGML_TYPE_F32: {
        స్థిరరాశి స్వయంచాలక * దత్తాంశం = static_cast<స్థిరరాశి దశాంశం *>(tensor->దత్తాంశం);
        కొరకు (size_t i = 0; i < n; ++i) {
            బహిర్గతం << దత్తాంశం[i] << '\n';
        }
        ఆపు;
    }
    స్థితి GGML_TYPE_F16: {
        స్థిరరాశి స్వయంచాలక * దత్తాంశం = static_cast<స్థిరరాశి ggml_fp16_t *>(tensor->దత్తాంశం);
        కొరకు (size_t i = 0; i < n; ++i) {
            బహిర్గతం << ggml_fp16_గమ్యం_fp32(దత్తాంశం[i]) << '\n';
        }
        ఆపు;
    }
    స్థితి GGML_TYPE_BF16: {
        స్థిరరాశి స్వయంచాలక * దత్తాంశం = static_cast<స్థిరరాశి ggml_bf16_t *>(tensor->దత్తాంశం);
        కొరకు (size_t i = 0; i < n; ++i) {
            బహిర్గతం << ggml_bf16_గమ్యం_fp32(దత్తాంశం[i]) << '\n';
        }
        ఆపు;
    }
    స్థితి GGML_TYPE_I32: {
        స్థిరరాశి స్వయంచాలక * దత్తాంశం = static_cast<స్థిరరాశి int32_t *>(tensor->దత్తాంశం);
        కొరకు (size_t i = 0; i < n; ++i) {
            బహిర్గతం << దత్తాంశం[i] << '\n';
        }
        ఆపు;
    }
    స్థితి GGML_TYPE_I64: {
        స్థిరరాశి స్వయంచాలక * దత్తాంశం = static_cast<స్థిరరాశి int64_t *>(tensor->దత్తాంశం);
        కొరకు (size_t i = 0; i < n; ++i) {
            బహిర్గతం << దత్తాంశం[i] << '\n';
        }
        ఆపు;
    }
    అప్రమేయ:
        బహిర్గతం << "unsupported tensor type for text dump" << '\n';
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

శూన్య print_ఆదానం_tensor_info(స్థిరరాశి std::string & name, స్థిరరాశి ov::Tensor & tensor) {
    std::బయటకు << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.దత్తాంశం()
              << '\n';
    ఎంపిక (tensor.get_element_type()) {
    స్థితి ov::element::f32: {
        ఒకవేళ (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::బయటకు << *(tensor.దత్తాంశం<దశాంశం>()) << '\n';
        } లేకపోతే {
            size_t అడ్డువరుసలు = tensor.get_shape()[2];
            size_t నిలువుల_సంఖ్య_2 = tensor.get_shape()[3];
            స్థిరరాశి దశాంశం * దత్తాంశం = tensor.దత్తాంశం<దశాంశం>();
            కొరకు (size_t i = 0; i < అడ్డువరుసలు; ++i) {
                కొరకు (size_t j = 0; j < నిలువుల_సంఖ్య_2; ++j) {
                    దశాంశం val = దత్తాంశం[i * నిలువుల_సంఖ్య_2 + j];
                    ఒకవేళ (std::isinf(val) && val < 0) {
                        std::బయటకు << std::setw(5) << "-inf";
                    } లేకపోతే {
                        std::బయటకు << std::setw(5) << val;
                    }
                }
                std::బయటకు << '\n';
            }
        }

        ఆపు;
    }
    స్థితి ov::element::f16:
        std::బయటకు << *(tensor.దత్తాంశం<ov::float16>()) << '\n';
        ఆపు;
    స్థితి ov::element::i32:
        కొరకు (size_t i = 0; i < tensor.get_కొలత(); ++i) {
            std::బయటకు << tensor.దత్తాంశం<int32_t>()[i] << ' ';
        }
        std::బయటకు << '\n';
        ఆపు;
    స్థితి ov::element::i64:
        కొరకు (size_t i = 0; i < tensor.get_కొలత(); ++i) {
            std::బయటకు << tensor.దత్తాంశం<int64_t>()[i] << ' ';
        }
        std::బయటకు << '\n';
        ఆపు;
    అప్రమేయ:
        ఆపు;
    }
}

శూన్య print_output_tensor_info(స్థిరరాశి std::string & name, స్థిరరాశి ov::Tensor & tensor, స్థిరరాశి శూన్య * output_dst) {
    std::బయటకు << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    స్వయంచాలక print_float_stats = [](స్థిరరాశి std::string & type_name, size_t size, స్వయంచాలక get_విలువ) {
        ఒకవేళ (size == 0) {
            తిరిగి;
        }

        దశాంశం మొదటి = get_విలువ(0);
        దశాంశం min = మొదటి;
        దశాంశం max = మొదటి;
        ద్విగుణ కూడిక = మొదటి;

        కొరకు (size_t i = 1; i < size; ++i) {
            దశాంశం v = get_విలువ(i);
            min = std::min(v, min);
            max = std::max(v, max);
            కూడిక += v;
        }
        ద్విగుణ సగటు_2 = కూడిక / size;

        std::బయటకు << std::కుడి << std::setw(6) << type_name << std::కుడి << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::బయటకు << std::కుడి << std::setw(6) << "" << std::కుడి << std::setw(12) << మొదటి << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << సగటు_2 << '\n';
    };

    ఎంపిక (tensor.get_element_type()) {
    స్థితి ov::element::f32: {
        స్థిరరాశి దశాంశం * దత్తాంశం = tensor.దత్తాంశం<దశాంశం>();
        size_t size = tensor.get_కొలత();
        print_float_stats("[f32]", size, [దత్తాంశం](size_t i) { తిరిగి దత్తాంశం[i]; });
        ఆపు;
    }
    స్థితి ov::element::f16: {
        స్థిరరాశి ov::float16 * దత్తాంశం = tensor.దత్తాంశం<ov::float16>();
        size_t size = tensor.get_కొలత();
        print_float_stats("[f16]", size, [దత్తాంశం](size_t i) { తిరిగి static_cast<దశాంశం>(దత్తాంశం[i]); });
        ఆపు;
    }
    అప్రమేయ:
        ఆపు;
    }
}

స్థిరరాశి ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; ++i) {
        స్వయంచాలక * op = cgraph->nodes[i];
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; ++j) {
            స్వయంచాలక * src = op->src[j];
            ఒకవేళ (src == శూన్యసూచిక) {
                ఆపు;
            }
            ఒకవేళ (GgmlOvDecoder::is_inp_pos(src, op)) {
                తిరిగి src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    విసురు std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_టోకెన్లు(ggml_cgraph * cgraph, స్థిరరాశి ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    పూర్ణాంకం n_planes = 1;
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; ++i) {
        స్వయంచాలక * op = cgraph->nodes[i];
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; ++j) {
            ఒకవేళ (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                ఆపు;
            }
        }
    }
    తిరిగి inp_pos->ne[0] / n_planes;
}

తార్కిక get_is_prefill(ggml_cgraph * cgraph, స్థిరరాశి ggml_tensor * inp_pos) {
    తిరిగి get_inp_pos_n_టోకెన్లు(cgraph, inp_pos) > 1;
}
