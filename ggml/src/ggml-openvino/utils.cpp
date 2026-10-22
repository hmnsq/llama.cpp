#شامل "utils.h"

#شامل "ggml-impl.h"
#شامل "ggml-openvino-extra.h"
#شامل "ggml-openvino.h"
#شامل "ggml-openvino/ggml-decoder.h"
#شامل "ggml.h"
#شامل "model-cache.h"
#شامل "openvino/frontend.h"
#شامل "openvino/input_model.h"

#شامل <algorithm>
#شامل <cassert>
#شامل <cmath>
#شامل <cstddef>
#شامل <cstdint>
#شامل <cstdlib>
#شامل <cstring>
#شامل <fstream>
#شامل <functional>
#شامل <iomanip>
#شامل <iostream>
#شامل <memory>
#شامل <openvino/core/any.hpp>
#شامل <openvino/core/graph_util.hpp>
#شامل <openvino/core/shape.hpp>
#شامل <openvino/core/type/float16.hpp>
#شامل <openvino/frontend/manager.hpp>
#شامل <openvino/openvino.hpp>
#شامل <openvino/runtime/compiled_model.hpp>
#شامل <openvino/runtime/infer_request.hpp>
#شامل <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#شامل <openvino/runtime/intel_npu/properties.hpp>
#شامل <openvino/runtime/properties.hpp>
#شامل <openvino/runtime/tensor.hpp>
#شامل <optional>
#شامل <set>
#شامل <string>
#شامل <unordered_map>
#شامل <vector>

نامخلا {
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
    ساکن ثابت منطقی kv_قاش_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    اگر (kv_قاش_disabled) {
        واپس std::nullopt;
    }
    اگر (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        واپس std::nullopt;
    }
    اگر (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != خالی_اشاریہ) {
        واپس std::nullopt;
    }
    ثابت خودکار * op = ggml_decoder->get_tensor_مستعمل_op(ggml_tensor);
    اگر (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        واپس std::nullopt;
    }

    ثابت خودکار & compute_params = ggml_decoder->get_compute_params();
    اگر (compute_params.n_seq_فعال != 1 || compute_params.seq_فعال_آغاز != 0) {
        واپس std::nullopt;
    }

    صحیح layer;
    اگر (خودکار layer_opt = نکالو_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } ورنہ {
        واپس std::nullopt;
    }

    ثابت منطقی is_swa = ggml_decoder->is_swa_layer(layer);
    اگر (is_swa) {
        واپس std::nullopt;
    }
    ثابت صحیح ctx_فی_seq = ggml_decoder->get_ctx_فی_seq();
    ثابت صحیح n_kv = compute_params.attention_سائز;
    اگر (ctx_فی_seq <= 0 || n_kv <= 0 || n_kv >= ctx_فی_seq) {
        واپس std::nullopt;
    }

    ov::Shape بھرا_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    اگر (بھرا_shape.size() != 4 || بھرا_shape[0] != 1 || بھرا_shape[1] != 1 ||
        static_cast<صحیح>(بھرا_shape[2]) != ctx_فی_seq) {
        واپس std::nullopt;
    }

    ov::Shape sliced_shape = بھرا_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    اگر (ggml_openvino_بفر_is_remote(ggml_tensor)) {
        خودکار remote_context = ggml_openvino_get_remote_context();
        خودکار gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        واپس gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->ڈیٹا);
    }

    واپس ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->ڈیٹا);
}

ساکن uint64_t ggml_openvino_model_cache_extra_cfg(منطقی stateful, منطقی recurrent) {
    ثابت حرف * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    ثابت منطقی manual_gqa_enabled = manual_gqa_env != خالی_اشاریہ ?
                                        ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0 :
                                        ggml_openvino_is_gpu();

    uint64_t extra_cfg = 1;  // Graph-ordinal port names (invalidate older disk-cache blobs).
    extra_cfg = extra_cfg * 131 + (stateful ? (recurrent ? 2u : 1u) : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (manual_gqa_enabled ? 1u : 0u);
    extra_cfg = extra_cfg * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? 1u : 0u);
    واپس extra_cfg;
}

std::map<std::string, std::shared_ptr<ov::Node>> get_وزن_نام(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> نام;
    برائے (ثابت خودکار & name : GgmlOvDecoder::collect_وزن_نام(cgraph)) {
        نام[name] = خالی_اشاریہ;
    }
    واپس نام;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
ساکن std::string compiled_گراف_کلید(ثابت ggml_cgraph * graph,
                                      ثابت GgmlOvDecoder & decoder,
                                      ثابت std::string & device,
                                      صحیح prefill_chunk_سائز = 0,
                                      منطقی disk_cache = جھوٹ) {
    std::string کلید;
    خودکار append = [&کلید](ثابت خودکار & value) {
        کلید.append(reinterpret_cast<ثابت حرف *>(&value), حجم(value));
    };
    خودکار append_string = [&](ثابت std::string & value) {
        append(value.size());
        کلید.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_سائز);
    منطقی has_وزن_بفر_id = جھوٹ;
    std::unordered_map<ثابت ggml_tensor *, size_t> ids;
    std::function<خالی(ثابت ggml_tensor *)> دورہ = [&](ثابت ggml_tensor * tensor) {
        اگر (!tensor) {
            append(size_t(0));
            واپس;
        }
        خودکار inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.پہلا->second);
        اگر (!inserted.second) {
            واپس;
        }
        append_string(tensor->name);
        append(tensor->type);
        append(tensor->op);
        append(tensor->flags);
        append(tensor->ne);
        append(tensor->nb);
        append(tensor->op_params);
        append(tensor->view_offs);
        ثابت خودکار * بنیاد = tensor->view_src ? tensor->view_src : tensor;
        ثابت منطقی وزن = بنیاد->بفر && بنیاد->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS;
        append(وزن);
        اگر (وزن && !disk_cache) {
            ثابت size_t بفر_id = ggml_backend_openvino_بفر_get_ctx_id(بنیاد->بفر);
            has_وزن_بفر_id |= بفر_id != 0;
            append(بفر_id);
            append(tensor->ڈیٹا);
        }
        دورہ(tensor->view_src);
        برائے (ثابت خودکار * src : tensor->src) {
            دورہ(src);
        }
    };
    append(graph->n_nodes);
    برائے (صحیح i = 0; i < graph->n_nodes; ++i) {
        دورہ(graph->nodes[i]);
    }
    append(graph->n_leafs);
    برائے (صحیح i = 0; i < graph->n_leafs; ++i) {
        دورہ(graph->leafs[i]);
    }
    برائے (ثابت خودکار & داخلہ : decoder.get_model_extra_inputs()) {
        append_string(داخلہ.پہلا);
        append_string(داخلہ.second.type.get_type_name());
        append(داخلہ.second.shape.size());
        برائے (خودکار dim : داخلہ.second.shape) {
            append(dim);
        }
        append(داخلہ.second.is_parameter);
        اگر (!داخلہ.second.is_parameter) {
            append(داخلہ.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    واپس disk_cache || has_وزن_بفر_id ? کلید : std::string{};
}

ساکن std::string dynamic_گراف_signature(ثابت ggml_cgraph * graph,
                                           ثابت GgmlOvDecoder & decoder,
                                           ثابت ModelParams & params) {
    std::string کلید = "dynamic-1";
    خودکار append = [&کلید](ثابت خودکار & value) {
        کلید.append(reinterpret_cast<ثابت حرف *>(&value), حجم(value));
    };
    خودکار append_string = [&](ثابت std::string & value) {
        append(value.size());
        کلید.append(value);
    };
    گراف_کلید گراف_id(graph, سچ);
    append(گراف_id.n_nodes);
    append(گراف_id.n_leaves);
    append_string(گراف_id.پہلا_گرہ_name);
    append_string(گراف_id.آخری_گرہ_name);
    برائے (ثابت خودکار & name : گراف_id.داخلہ_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    برائے (صحیح layer : params.swa_layers) {
        append(layer);
    }
    برائے (ثابت خودکار & [layer, heads] : params.n_heads_kv_فی_layer) {
        append(layer);
        append(heads);
    }
    برائے (ثابت خودکار & [name, داخلہ] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(داخلہ.type.get_type_name());
    }
    برائے (ثابت خودکار & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    برائے (ثابت خودکار & [name, داخلہ] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(داخلہ.type.get_type_name());
        append(داخلہ.is_parameter);
        اگر (!داخلہ.is_parameter) {
            append(داخلہ.value);
        }
    }
    واپس کلید;
}

ساکن منطقی compiled_model_matches_گراف(ثابت ov::CompiledModel & model,
                                         ثابت std::vector<std::string> & inputs,
                                         ثابت std::vector<std::string> & outputs,
                                         ثابت GgmlOvDecoder & decoder) {
    اگر (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        واپس جھوٹ;
    }
    ثابت خودکار & گراف_inputs = decoder.get_model_inputs();
    ثابت خودکار & extra_inputs = decoder.get_model_extra_inputs();
    ثابت خودکار & گراف_outputs = decoder.get_model_outputs();
    برائے (size_t i = 0; i < inputs.size(); ++i) {
        ثابت خودکار & پورٹ = model.داخلہ(i);
        خودکار گراف_it = گراف_inputs.find(inputs[i]);
        اگر (گراف_it != گراف_inputs.end()) {
            اگر (پورٹ.get_element_type() != گراف_it->second.type ||
                !پورٹ.get_partial_shape().compatible(گراف_it->second.shape)) {
                واپس جھوٹ;
            }
            جاری;
        }
        خودکار extra_it = extra_inputs.find(inputs[i]);
        اگر (extra_it == extra_inputs.end() || پورٹ.get_element_type() != extra_it->second.type ||
            !پورٹ.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            واپس جھوٹ;
        }
    }
    برائے (size_t i = 0; i < outputs.size(); ++i) {
        خودکار گراف_it = گراف_outputs.find(outputs[i]);
        اگر (گراف_it == گراف_outputs.end()) {
            اگر (outputs[i].compare(0, 8, "__debug_") != 0) {
                واپس جھوٹ;
            }
            جاری;
        }
        ثابت خودکار & پورٹ = model.output(i);
        اگر (پورٹ.get_element_type() != GgmlOvDecoder::get_ov_type(گراف_it->second) ||
            !پورٹ.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(گراف_it->second)))) {
            واپس جھوٹ;
        }
    }
    واپس سچ;
}

ساکن std::string ov_profiling_csv_میدان(ثابت std::string & value) {
    std::string escaped = "\"";
    برائے (حرف c : value) {
        escaped += c;
        اگر (c == '"') {
            escaped += c;
        }
    }
    escaped += '"';
    واپس escaped;
}

ساکن خالی dump_ov_profiling_info(ثابت ov::InferRequest & infer_request) {
    اگر (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") < 2) {
        واپس;
    }

    ساکن std::atomic<uint64_t> inference_اشاریہ{0};
    ثابت uint64_t اشاریہ = inference_اشاریہ.fetch_add(1);
    ثابت std::string path = "openvino_profile_" + std::to_string(اشاریہ) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    اگر (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        واپس;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t کل_real_وقت = 0;
    int64_t کل_cpu_وقت = 0;
    size_t executed_گنتی = 0;
    برائے (ثابت خودکار & info : infer_request.get_profiling_info()) {
        ثابت حرف * حالت_2 = "NOT_RUN";
        اگر (info.حالت_2 == ov::ProfilingInfo::حالت_3::EXECUTED) {
            حالت_2 = "EXECUTED";
            کل_real_وقت += info.real_وقت.count();
            کل_cpu_وقت += info.cpu_وقت.count();
            executed_گنتی++;
        } ورنہ اگر (info.حالت_2 == ov::ProfilingInfo::حالت_3::OPTIMIZED_برآمد) {
            حالت_2 = "OPTIMIZED_OUT";
        }

        output << حالت_2 << ',' << info.real_وقت.count() << ',' << info.cpu_وقت.count() << ','
               << info.آغاز_وقت.count() << ',' << ov_profiling_csv_میدان(info.node_type) << ','
               << ov_profiling_csv_میدان(info.گرہ_name) << ',' << ov_profiling_csv_میدان(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_گنتی, کل_real_وقت / 1000.0, کل_cpu_وقت / 1000.0);
}

ساکن خالی reset_single_خانہ_recurrent_cache(ggml_cgraph * cgraph,
                                              ثابت ModelParams & model_params,
                                              ثابت ComputeParams & compute_params) {
    اگر (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        واپس;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_بفر_t> buffers;
    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        خودکار * گرہ = cgraph->nodes[i];
        اگر (گرہ->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(گرہ->view_src) &&
            گرہ->view_src->ne[1] == 1 && گرہ->view_src->بفر != خالی_اشاریہ) {
            buffers.insert(گرہ->view_src->بفر);
        }
    }
    برائے (خودکار * بفر : buffers) {
        ggml_backend_بفر_clear(بفر, 0);
    }
}

ov::Tensor create_ov_output_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   ثابت std::shared_ptr<ov::InferRequest> & infer_request,
                                   صحیح output_اشاریہ,
                                   ثابت ggml_tensor * ggml_tensor) {
    اگر (خودکار sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        واپس *sliced;
    }

    خودکار output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    خالی * output_ڈیٹا = ggml_tensor->ڈیٹا;
    اگر (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_اشاریہ).get_shape();
    } ورنہ {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        اگر (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != خالی_اشاریہ &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_ڈیٹا = ggml_tensor->view_src->ڈیٹا;
        } ورنہ {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    اگر (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_بفر_is_remote(ggml_tensor) &&
        ggml_tensor->extra != خالی_اشاریہ) {
        خودکار * extra_بنیاد = static_cast<ggml_openvino_extra_بنیاد *>(ggml_tensor->extra);
        اگر (extra_بنیاد->type == ggml_openvino_extra_بنیاد::Type::TENSOR) {
            خودکار * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_بنیاد);
            اگر (tensor_extra->tensor != خالی_اشاریہ && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                واپس *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_ڈیٹا);
    واپس output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_صفیں_مقصد_seq_axis_2(ثابت ov::Tensor & kv_tensor, size_t n_heads_kv) {
    ثابت size_t صفیں = kv_tensor.get_shape()[2];
    ثابت size_t سر_سائز = kv_tensor.get_shape()[3] / n_heads_kv;
    ثابت size_t elem = kv_tensor.get_element_type().size();
    ثابت size_t سر_بائٹیں = سر_سائز * elem;

    ov::Tensor برآمد(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, صفیں, سر_سائز});
    ثابت خودکار * src = static_cast<ثابت uint8_t *>(kv_tensor.ڈیٹا());
    خودکار * dst = static_cast<uint8_t *>(برآمد.ڈیٹا());
    برائے (size_t s = 0; s < صفیں; s++) {
        برائے (size_t h = 0; h < n_heads_kv; h++) {
            حافظہ_نقل(dst + (h * صفیں + s) * سر_بائٹیں, src + (s * n_heads_kv + h) * سر_بائٹیں, سر_بائٹیں);
        }
    }
    واپس برآمد;
}

سانچہ <قسم_نام T> خالی سیٹ_zero_diagonal(std::vector<T> & میٹرکس, size_t صفیں, size_t ستون_تعداد_2, T zero_قدر = T{}) {
    برائے (size_t i = 0; i < صفیں; ++i) {
        size_t diag_ستون = std::min(i, ستون_تعداد_2 - 1);
        میٹرکس[i * ستون_تعداد_2 + diag_ستون] = zero_قدر;
    }
}

ov::Tensor make_contiguous_بانٹو_داخلہ_tensor(ثابت ساخت ggml_tensor * ggml_tensor, ثابت ov::Shape & داخلہ_shape) {
    ثابت size_t element_سائز = ggml_type_سائز(ggml_tensor->type);
    ثابت size_t بلاک_سائز_3 = ggml_blck_سائز(ggml_tensor->type);

    GGML_ASSERT(بلاک_سائز_3 == 1 && "non-contiguous split inputs must be plain element types");

    ثابت ساخت ggml_tensor * ماخذ_tensor = ggml_tensor->view_src != خالی_اشاریہ ? ggml_tensor->view_src : ggml_tensor;
    ثابت size_t ماخذ_انحراف = ggml_tensor->view_src != خالی_اشاریہ ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> ماخذ_ڈیٹا(ggml_nbytes(ماخذ_tensor));
    ggml_backend_tensor_get(ماخذ_tensor, ماخذ_ڈیٹا.ڈیٹا(), 0, ماخذ_ڈیٹا.size());

    ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
    خودکار * dst = static_cast<uint8_t *>(داخلہ_tensor.ڈیٹا());
    size_t dst_انحراف = 0;

    برائے (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        برائے (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            برائے (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                برائے (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    ثابت size_t src_انحراف = ماخذ_انحراف + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::حافظہ_نقل(dst + dst_انحراف, ماخذ_ڈیٹا.ڈیٹا() + src_انحراف, element_سائز);
                    dst_انحراف += element_سائز;
                }
            }
        }
    }

    واپس داخلہ_tensor;
}

ov::Tensor convert_ggml_داخلہ_مقصد_ov(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & name) {
    ثابت خودکار * ggml_tensor = ggml_decoder->get_داخلہ_ggml_tensor(name);

    اگر (خودکار sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        واپس *sliced;
    }

    اگر (ggml_tensor->extra != خالی_اشاریہ && !ggml_decoder->is_splited_model()) {
        خودکار * extra_بنیاد = static_cast<ggml_openvino_extra_بنیاد *>(ggml_tensor->extra);
        اگر (extra_بنیاد->type == ggml_openvino_extra_بنیاد::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            خودکار * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_بنیاد);
            واپس *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    خودکار * داخلہ_ڈیٹا = ggml_tensor->ڈیٹا;
    ov::Shape داخلہ_shape;
    اگر (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        داخلہ_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } ورنہ {
        داخلہ_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    اگر (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        واپس make_contiguous_بانٹو_داخلہ_tensor(ggml_tensor, داخلہ_shape);
    }

    خودکار داخلہ_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape, داخلہ_ڈیٹا);
    واپس داخلہ_tensor;
}

ov::Tensor get_ov_داخلہ_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & param_name) {
    ov::Tensor داخلہ_tensor;
    خودکار extra_داخلہ = ggml_decoder->get_model_extra_inputs().find(param_name);
    اگر (extra_داخلہ != ggml_decoder->get_model_extra_inputs().end()) {
        داخلہ_tensor = ov::Tensor(extra_داخلہ->second.type, extra_داخلہ->second.shape);
        *داخلہ_tensor.ڈیٹا<int64_t>() = extra_داخلہ->second.value;
    } ورنہ {
        داخلہ_tensor = convert_ggml_داخلہ_مقصد_ov(ggml_decoder, param_name);
    }
    واپس داخلہ_tensor;
}

ov::Tensor get_ov_داخلہ_tensor_static_decode(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             ثابت std::string & param_name) {
    // NPU decoding stage
    اگر (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        واپس get_ov_داخلہ_tensor(ggml_decoder, param_name);
    }
    ثابت خودکار * ggml_tensor = ggml_decoder->get_داخلہ_ggml_tensor(param_name);
    ثابت خودکار * op = ggml_decoder->get_tensor_مستعمل_op(ggml_tensor);

    اگر (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        ثابت صحیح n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        أكد(ggml_tensor->ne[0] == n_planes);
        ov::Shape داخلہ_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        std::حافظہ_نقل(داخلہ_tensor.ڈیٹا(), ggml_tensor->ڈیٹا, n_planes * ggml_type_سائز(ggml_tensor->type));
        واپس داخلہ_tensor;
    }

    اگر (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape داخلہ_shape = {1, 1, 1, 1};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        int32_t inp_برآمد_id = *((int32_t *) ggml_tensor->ڈیٹا);
        أكد(ggml_tensor->ne[0] == 1);
        أكد(inp_برآمد_id == 0);
        *داخلہ_tensor.ڈیٹا<int32_t>() = inp_برآمد_id;
        واپس داخلہ_tensor;
    }

    اگر (GgmlOvDecoder::is_inp_نقاب(ggml_tensor, op)) {
        size_t context_سائز = ggml_decoder->get_ctx_سائز();
        اگر (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_ڈیٹا =
                pad_داخلہ<ggml_fp16_t>(ggml_tensor, 1, context_سائز, GGML_FP32_مقصد_FP16(-INFINITY));
            ov::Tensor داخلہ_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_سائز});
            std::حافظہ_نقل(داخلہ_tensor.ڈیٹا(), padded_ڈیٹا.ڈیٹا(), padded_ڈیٹا.size() * حجم(ggml_fp16_t));
            واپس داخلہ_tensor;
        }

        std::vector<اعشاری> padded_ڈیٹا = pad_داخلہ<اعشاری>(ggml_tensor, 1, context_سائز, -INFINITY);
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_سائز});
        خودکار * ڈیٹا_ptr = داخلہ_tensor.ڈیٹا<اعشاری>();
        std::copy(padded_ڈیٹا.begin(), padded_ڈیٹا.begin() + context_سائز, ڈیٹا_ptr);
        واپس داخلہ_tensor;
    }

    واپس get_ov_داخلہ_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_داخلہ_tensor_static_prefill(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              ثابت std::string & param_name,
                                              صحیح chunk_اشاریہ) {
    // NPU prompt processing stage
    ثابت size_t داخلہ_len = ggml_decoder->get_داخلہ_len();
    ثابت size_t chunk_سائز = ggml_decoder->m_prefill_chunk_سائز;
    ثابت size_t chunk_valid_سائز = std::min(chunk_سائز, داخلہ_len - chunk_اشاریہ * chunk_سائز);
    ثابت size_t chunk_pad_سائز = chunk_سائز - chunk_valid_سائز;

    اگر (param_name == "chunk_valid_len") {
        ov::Tensor داخلہ_tensor(ov::element::i64, ov::Shape{1});
        *داخلہ_tensor.ڈیٹا<int64_t>() = (int64_t) chunk_valid_سائز;
        واپس داخلہ_tensor;
    }
    اگر (chunk_اشاریہ > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor داخلہ_tensor(ov::element::i64, ov::Shape{1});
        *داخلہ_tensor.ڈیٹا<int64_t>() = 0;
        واپس داخلہ_tensor;
    }
    اگر (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        واپس get_ov_داخلہ_tensor(ggml_decoder, param_name);
    }
    ثابت خودکار * ggml_tensor = ggml_decoder->get_داخلہ_ggml_tensor(param_name);
    ثابت خودکار * op = ggml_decoder->get_tensor_مستعمل_op(ggml_tensor);

    اگر (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        ثابت صحیح n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        ثابت size_t element_سائز = ggml_type_سائز(ggml_tensor->type);
        ov::Shape داخلہ_shape = {1, 1, 1, (size_t) n_planes * chunk_سائز};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        برائے (صحیح p = 0; p < n_planes; p++) {
            ثابت حرف * src =
                (ثابت حرف *) ggml_tensor->ڈیٹا + (p * داخلہ_len + chunk_اشاریہ * chunk_سائز) * element_سائز;
            حرف * dst = (حرف *) داخلہ_tensor.ڈیٹا() + p * chunk_سائز * element_سائز;
            std::حافظہ_نقل(dst, src, chunk_valid_سائز * element_سائز);
            اگر (chunk_pad_سائز > 0) {
                اگر (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t آخری_قدر = *((ثابت int32_t *) src + chunk_valid_سائز - 1);
                    int32_t * برآمد = (int32_t *) dst;
                    std::fill(برآمد + chunk_valid_سائز, برآمد + chunk_سائز, آخری_قدر + 1);
                } ورنہ اگر (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t آخری_قدر = *((ثابت int64_t *) src + chunk_valid_سائز - 1);
                    int64_t * برآمد = (int64_t *) dst;
                    std::fill(برآمد + chunk_valid_سائز, برآمد + chunk_سائز, آخری_قدر + 1);
                } ورنہ {
                    پھینکو std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        واپس داخلہ_tensor;
    }

    اگر (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape داخلہ_shape = {1, 1, 1, chunk_سائز};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_سائز = ggml_type_سائز(ggml_tensor->type);
        خالی * داخلہ_ڈیٹا = (حرف *) ggml_tensor->ڈیٹا + chunk_اشاریہ * chunk_سائز * element_سائز;
        std::حافظہ_نقل(داخلہ_tensor.ڈیٹا(), داخلہ_ڈیٹا, chunk_valid_سائز * element_سائز);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        اگر (chunk_pad_سائز > 0) {
            اگر (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t آخری_قدر =
                    *((int32_t *) ggml_tensor->ڈیٹا + (chunk_اشاریہ * chunk_سائز + chunk_valid_سائز - 1));
                int32_t * output_ڈیٹا = داخلہ_tensor.ڈیٹا<int32_t>();
                std::fill(output_ڈیٹا + chunk_valid_سائز, output_ڈیٹا + chunk_سائز, آخری_قدر + 1);
            } ورنہ اگر (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t آخری_قدر =
                    *((int64_t *) ggml_tensor->ڈیٹا + (chunk_اشاریہ * chunk_سائز + chunk_valid_سائز - 1));
                int64_t * output_ڈیٹا = داخلہ_tensor.ڈیٹا<int64_t>();
                std::fill(output_ڈیٹا + chunk_valid_سائز, output_ڈیٹا + chunk_سائز, آخری_قدر + 1);
            } ورنہ {
                پھینکو std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        واپس داخلہ_tensor;
    }

    اگر (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape داخلہ_shape = {1, 1, 1, output_len};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        اگر (ggml_tensor->ne[0] == 0) {
            *داخلہ_tensor.ڈیٹا<int32_t>() = 0;
        } ورنہ {
            خودکار * ڈیٹا_addr = داخلہ_tensor.ڈیٹا<int32_t>();
            برائے (size_t i = 0; i < output_len; i++) {
                ڈیٹا_addr[i] = ((int32_t *) ggml_tensor->ڈیٹا)[i] % chunk_سائز;
            }
        }
        واپس داخلہ_tensor;
    }

    اگر (GgmlOvDecoder::is_inp_scale_صفیں(ggml_tensor, op)) {
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, chunk_سائز, 1});
        خودکار * dst = داخلہ_tensor.ڈیٹا<اعشاری>();
        ثابت خودکار * src = static_cast<ثابت اعشاری *>(ggml_tensor->ڈیٹا) + chunk_اشاریہ * chunk_سائز;
        std::copy(src, src + chunk_valid_سائز, dst);
        std::fill(dst + chunk_valid_سائز, dst + chunk_سائز, 1.0f);
        واپس داخلہ_tensor;
    }

    اگر (GgmlOvDecoder::is_inp_اوسط(ggml_tensor, op)) {
        ثابت size_t n_seqs = ggml_tensor->ne[1];
        ثابت size_t src_stride = ggml_tensor->ne[0];
        ثابت size_t copy_len = std::min<size_t>(chunk_valid_سائز, src_stride - chunk_اشاریہ * chunk_سائز);
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_سائز});
        خودکار * dst = داخلہ_tensor.ڈیٹا<اعشاری>();
        std::fill(dst, dst + n_seqs * chunk_سائز, 0.0f);
        ثابت خودکار * src = static_cast<ثابت اعشاری *>(ggml_tensor->ڈیٹا) + chunk_اشاریہ * chunk_سائز;
        برائے (size_t s = 0; s < n_seqs; s++) {
            std::حافظہ_نقل(dst + s * chunk_سائز, src + s * src_stride, copy_len * حجم(اعشاری));
        }
        واپس داخلہ_tensor;
    }

    اگر (GgmlOvDecoder::is_inp_نقاب(ggml_tensor, op)) {
        size_t ستون_تعداد_2 = ggml_tensor->ne[0];
        size_t صفیں = ggml_tensor->ne[1];
        size_t chunk_valid_صفیں = std::min(chunk_سائز, صفیں - chunk_اشاریہ * chunk_سائز);
        size_t context_سائز = ggml_decoder->get_ctx_سائز();
        اگر (ggml_tensor->type == GGML_TYPE_F16) {
            ثابت خودکار * ggml_ڈیٹا =
                static_cast<ثابت ggml_fp16_t *>(ggml_tensor->ڈیٹا) + chunk_اشاریہ * chunk_سائز * ستون_تعداد_2;
            std::vector<ggml_fp16_t> padded_ڈیٹا = pad_داخلہ<ggml_fp16_t>(ggml_ڈیٹا, chunk_valid_صفیں, ستون_تعداد_2, chunk_سائز,
                                                                          context_سائز, GGML_FP32_مقصد_FP16(-INFINITY));
            سیٹ_zero_diagonal(padded_ڈیٹا, chunk_سائز, context_سائز, GGML_FP32_مقصد_FP16(0.0f));
            ov::Tensor داخلہ_tensor(ov::element::f16, ov::Shape{1, 1, chunk_سائز, context_سائز});
            std::حافظہ_نقل(داخلہ_tensor.ڈیٹا(), padded_ڈیٹا.ڈیٹا(), padded_ڈیٹا.size() * حجم(ggml_fp16_t));
            واپس داخلہ_tensor;
        }

        ثابت خودکار * ggml_ڈیٹا = static_cast<ثابت اعشاری *>(ggml_tensor->ڈیٹا) + chunk_اشاریہ * chunk_سائز * ستون_تعداد_2;
        std::vector<اعشاری> padded_ڈیٹا =
            pad_داخلہ<اعشاری>(ggml_ڈیٹا, chunk_valid_صفیں, ستون_تعداد_2, chunk_سائز, context_سائز, -INFINITY);
        سیٹ_zero_diagonal(padded_ڈیٹا, chunk_سائز, context_سائز);
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, chunk_سائز, context_سائز});
        خودکار * ڈیٹا_ptr = داخلہ_tensor.ڈیٹا<اعشاری>();
        std::copy(padded_ڈیٹا.begin(), padded_ڈیٹا.begin() + chunk_سائز * context_سائز, ڈیٹا_ptr);
        واپس داخلہ_tensor;
    }

    واپس get_ov_داخلہ_tensor(ggml_decoder, param_name);
}

شمار ggml_حالت naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               ثابت std::string & device,
                               ثابت ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    اگر (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        واپس GGML_حالت_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_تالا(cache.mutex);
    منطقی naive = سچ;
    خودکار model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph, naive);
    خودکار decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    خودکار داخلہ_model = std::make_shared<ov::frontend::ggml::داخلہ_Model>(decoder);
    خودکار model = ov::frontend::ggml::Front_انتہا::convert(داخلہ_model, naive);
    اگر (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    خودکار remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    اگر (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_منوال.name()] = ov::hint::Execution_منوال::PERFORMANCE;
    } ورنہ {
        compile_config[ov::hint::execution_منوال.name()] = ov::hint::Execution_منوال::ACCURACY;
    }
    اگر (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } ورنہ {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> داخلہ_نام;
    std::vector<std::string> output_نام;
    برائے (ثابت خودکار & param : model->get_parameters()) {
        داخلہ_نام.push_back(param->get_friendly_name());
    }
    برائے (ثابت خودکار & نتیجہ : model->get_results()) {
        output_نام.push_back(نتیجہ->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    داخلہ_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_تالا.unlock();

    برائے (size_t i = 0; i < داخلہ_نام.size(); i++) {
        ثابت خودکار & param_name = داخلہ_نام[i];
        خودکار داخلہ_tensor = get_ov_داخلہ_tensor(decoder, param_name);
        infer_request->سیٹ_داخلہ_tensor(i, داخلہ_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    برائے (size_t i = 0; i < output_نام.size(); i++) {
        خودکار output_tensor = infer_request->get_output_tensor(i);
        ثابت خودکار & model_outputs = decoder->get_model_outputs();
        خودکار model_output_it = model_outputs.find(output_نام[i]);
        اگر (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            اگر (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_نام[i], output_tensor, output_tensor.ڈیٹا());
            }
            جاری;
        }
        خودکار * ggml_tensor = model_output_it->second;
        std::حافظہ_نقل(ggml_tensor->ڈیٹا, output_tensor.ڈیٹا(), output_tensor.get_byte_سائز());
    }
    واپس GGML_حالت_SUCCESS;
}

شمار ggml_حالت ov_گراف_compute_dynamic(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    خودکار & core = ov_singleton_core();
    ثابت خودکار & config = ggml_openvino_get_compile_config();
    ثابت خودکار & device = r_ctx->device;
    ساکن خودکار is_static = جھوٹ;

    ساکن ثابت منطقی cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    خودکار آغاز_وقت = ggml_وقت_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    ثابت int64_t cache_کلید_آغاز_وقت = ggml_وقت_us();
    گراف_کلید کلید(cgraph);
    ثابت int64_t cache_کلید_compute_وقت = ggml_وقت_us() - cache_کلید_آغاز_وقت;
    int64_t cache_lookup_وقت = 0;
    منطقی کلید_نظر_آیا = جھوٹ;
    اگر (!cache_disabled) {
        ثابت int64_t cache_lookup_آغاز_وقت = ggml_وقت_us();
        std::lock_guard<std::mutex> map_تالا(r_ctx->ctx_mutex);
        کلید_نظر_آیا = r_ctx->decoder_cache.find(کلید) != r_ctx->decoder_cache.end();
        cache_lookup_وقت += ggml_وقت_us() - cache_lookup_آغاز_وقت;
    }

    ثابت int64_t model_بانٹو_check_آغاز_وقت = ggml_وقت_us();
    منطقی model_is_splitted = کلید_نظر_آیا ? جھوٹ : is_model_splitted(cgraph);
    ثابت int64_t model_بانٹو_check_وقت = ggml_وقت_us() - model_بانٹو_check_آغاز_وقت;

    اگر (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        واپس GGML_حالت_FAILED;
    }
    اگر (is_naive(cgraph)) {
        اگر (!model_is_splitted) {
            واپس naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
        }
    }

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    ثابت int64_t گراف_param_compute_آغاز_وقت = ggml_وقت_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    ثابت int64_t گراف_param_compute_وقت = ggml_وقت_us() - گراف_param_compute_آغاز_وقت;

    ثابت منطقی cache_enabled = !model_is_splitted && !cache_disabled;
    ثابت منطقی stateful_recurrent = r_ctx->stateful && m_params.state_سائز >= 0;
    ثابت منطقی stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    اگر (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_فعال > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        واپس GGML_حالت_FAILED;
    }
    اگر (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        واپس GGML_حالت_FAILED;
    }
    منطقی cache_کامیابی = جھوٹ;

    int64_t decoder_انتہا_وقت;
    int64_t conversion_انتہا_وقت;
    int64_t compile_انتہا_وقت;
    int64_t infer_انتہا_وقت;
    int64_t ov_raw_infer_آغاز;

    {
        std::shared_ptr<decoder_runtime_ctx> اندراج_2;
        ModelParams old_m_params;

        ثابت int64_t cache_lookup_آغاز_وقت = ggml_وقت_us();
        اگر (cache_enabled) {
            std::lock_guard<std::mutex> map_تالا(r_ctx->ctx_mutex);
            خودکار it = r_ctx->decoder_cache.find(کلید);
            cache_کامیابی = it != r_ctx->decoder_cache.end();
            اگر (cache_کامیابی) {
                اندراج_2 = it->second;
            } ورنہ {
                r_ctx->clear_caches_locked();
                خودکار mutex = std::make_shared<std::mutex>();
                اندراج_2 = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[کلید] = اندراج_2;
            }
        } ورنہ {
            خودکار mutex = std::make_shared<std::mutex>();
            اندراج_2 = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_کامیابی = جھوٹ;
        }
        cache_lookup_وقت += ggml_وقت_us() - cache_lookup_آغاز_وقت;

        std::lock_guard<std::mutex> تالا(*(اندراج_2->mutex));
        cache_کامیابی = cache_کامیابی && اندراج_2->ptr && r_ctx->infer_request_cache.count(کلید) != 0;

        اگر (cache_کامیابی) {
            ggml_decoder = اندراج_2->ptr;
            old_m_params = ggml_decoder->get_model_params();
            اگر (!ggml_decoder->is_splited_model()) {
                cache_کامیابی = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_داخلہ_نام;
        std::vector<std::string> ov_output_نام;

        اگر (cache_کامیابی) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->سیٹ_compute_params(c_params);
            ggml_decoder->سیٹ_model_params(m_params);
            اگر (old_m_params.kv_بفر_بدلا(m_params) || !ggml_decoder->is_bound_مقصد(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->جوڑو_extra_inputs();
            {
                std::lock_guard<std::mutex> map_تالا(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(کلید);
                ov_داخلہ_نام = r_ctx->ov_داخلہ_نام_cache.at(کلید);
                ov_output_نام = r_ctx->ov_output_نام_cache.at(کلید);
            }

            اگر (stateful_kv_only) {
                ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_ڈیٹا = (int32_t *) inp_pos->ڈیٹا;
                خودکار pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                اگر (pos_ڈیٹا[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_سائز = pos_shape[3];
                } ورنہ اگر (r_ctx->stateful_kv_سائز == static_cast<size_t>(pos_ڈیٹا[0])) {
                    r_ctx->stateful_kv_سائز += pos_shape[3];
                } ورنہ {
                    ثابت size_t pos_begin = static_cast<size_t>(pos_ڈیٹا[0]);
                    ثابت منطقی refill = pos_begin > r_ctx->stateful_kv_سائز;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    اگر (refill && !ggml_decoder->get_model_params().swa_layers.خالی_خانے()) {
                        ثابت صحیح n_swa = ggml_decoder->get_compute_params().swa_کھڑکی;
                        اگر (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_سائز, n_swa);
                            واپس GGML_حالت_FAILED;
                        }
                    }

                    ثابت منطقی relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    خودکار حالتیں = infer_request->query_state();
                    برائے (خودکار state : حالتیں) {
                        خودکار state_tensor = state.get_state();
                        خودکار state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        اگر (خودکار it = r_ctx->kv_state_داخلہ_name_map.find(state.get_name());
                            it != r_ctx->kv_state_داخلہ_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        صحیح n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        اگر (خودکار layer = نکالو_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        ثابت منطقی relayout_this_state = relayout_enabled;
                        ثابت size_t seq_axis = relayout_this_state ? 2 : 1;
                        ثابت size_t سر_axis = seq_axis == 2 ? 1 : 2;

                        اگر (refill) {
                            اگر (state_name.خالی_خانے()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                واپس GGML_حالت_FAILED;
                            }
                            خودکار kv_tensor = get_ov_داخلہ_tensor(ggml_decoder, state_name);
                            اگر (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_صفیں_مقصد_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } ورنہ {
                                ov::Shape refill_shape(4);
                                refill_shape[0] = state_tensor_shape[0];
                                refill_shape[seq_axis] = kv_tensor.get_shape()[2];
                                refill_shape[سر_axis] = state_tensor_shape[سر_axis];
                                refill_shape[3] = state_tensor_shape[3];
                                kv_tensor.سیٹ_shape(refill_shape);
                                state_tensor = kv_tensor;
                            }
                            state_tensor_shape = state_tensor.get_shape();
                        }
                        // Only ever shrink to a prefix the source really has. Slicing past it used to
                        // surface as a bare ov::Exception from the ROI constructor.
                        اگر (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            واپس GGML_حالت_FAILED;
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end(state_tensor_shape.begin(), state_tensor_shape.end());
                        end[seq_axis] = pos_begin;
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.سیٹ_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_سائز = pos_begin + pos_shape[3];
                }
            }

            decoder_انتہا_وقت = ggml_وقت_us();
            conversion_انتہا_وقت = decoder_انتہا_وقت;
            compile_انتہا_وقت = decoder_انتہا_وقت;
        } ورنہ {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            خودکار shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_تالا(shared_cache->mutex);
            خودکار وزن_نام = get_وزن_نام(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, وزن_نام,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ثابت std::string model_cache_dir = ggml_openvino_model_cache_dir();
            ثابت std::string exact_کلید = cache_enabled ? compiled_گراف_کلید(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_راستہ, manifest_راستہ;
            اگر (!model_cache_dir.خالی_خانے() && !model_is_splitted) {
                ثابت uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/سچ, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_گراف_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/سچ, m_params.rope_params, 16, extra_cfg,
                                                    compiled_گراف_کلید(cgraph, *ggml_decoder, device, 0, سچ));
                blob_راستہ = ggml_openvino_model_cache_blob_راستہ(model_cache_dir, model_fp);
                manifest_راستہ = ggml_openvino_model_cache_manifest_راستہ(model_cache_dir, model_fp);
            }
            std::string shared_کلید =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_کلید) : "";
            ov::CompiledModel shared_model;
            منطقی imported = جھوٹ;
            خودکار shared_it = shared_cache->graphs.find(shared_کلید);
            اگر (!shared_کلید.خالی_خانے() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_گراف(shared_it->second.decode, shared_it->second.داخلہ_نام,
                                                               shared_it->second.output_نام, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_داخلہ_نام = shared_it->second.داخلہ_نام;
                ov_output_نام = shared_it->second.output_نام;
                imported = سچ;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } ورنہ اگر (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_کلید = exact_کلید;
                shared_it = shared_cache->graphs.find(shared_کلید);
                اگر (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_داخلہ_نام = shared_it->second.داخلہ_نام;
                    ov_output_نام = shared_it->second.output_نام;
                    imported = سچ;
                }
                model_fp = exact_fp;
                blob_راستہ = ggml_openvino_model_cache_blob_راستہ(model_cache_dir, model_fp);
                manifest_راستہ = ggml_openvino_model_cache_manifest_راستہ(model_cache_dir, model_fp);
            }
            اگر (cache_enabled) {
                std::lock_guard<std::mutex> map_تالا(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(کلید);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            اگر (!model_cache_dir.خالی_خانے()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_منوال.name()] = ov::Cache_منوال::OPTIMIZE_رفتار;
            }
            اگر (!imported && !model_cache_dir.خالی_خانے() && !model_is_splitted) {
                ثابت uint64_t preferred_fp = model_fp;
                منطقی dynamic_incompatible = preferred_fp == exact_fp;
                برائے (صحیح candidate = 0; candidate < 2; ++candidate) {
                    اگر (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        توڑو;
                    }
                    ثابت uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    ثابت std::string candidate_blob =
                        ggml_openvino_model_cache_blob_راستہ(model_cache_dir, candidate_fp);
                    ثابت std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_راستہ(model_cache_dir, candidate_fp);
                    std::ifstream blob_اندر(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    اگر (!blob_اندر.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        جاری;
                    }
                    ثابت int64_t import_آغاز = ggml_وقت_us();
                    کوشش {
                        ov::CompiledModel cm;
                        خودکار remote_context = ggml_openvino_get_remote_context();
                        اگر (remote_context.has_value()) {
                            cm = core.import_model(blob_اندر, remote_context.value(), mc_config);
                        } ورنہ {
                            cm = core.import_model(blob_اندر, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        اگر (!saved_inputs.خالی_خانے() || !saved_outputs.خالی_خانے()) {
                            ov_داخلہ_نام = std::move(saved_inputs);
                            ov_output_نام = std::move(saved_outputs);
                        } ورنہ {
                            برائے (ثابت خودکار & p : cm.inputs()) {
                                ov_داخلہ_نام.push_back(p.get_گرہ()->get_friendly_name());
                            }
                            برائے (ثابت خودکار & o : cm.outputs()) {
                                ov_output_نام.push_back(o.get_گرہ()->get_friendly_name());
                            }
                        }
                        اگر (!compiled_model_matches_گراف(cm, ov_داخلہ_نام, ov_output_نام, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_داخلہ_نام.clear();
                            ov_output_نام.clear();
                            جاری;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        اندراج_2->ptr = ggml_decoder;
                        imported = سچ;
                        model_fp = candidate_fp;
                        blob_راستہ = candidate_blob;
                        manifest_راستہ = candidate_manifest;
                        اگر (candidate_fp == exact_fp) {
                            shared_کلید = exact_کلید;
                            shared_it = shared_cache->graphs.find(shared_کلید);
                        }
                        اگر (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_وقت_us() - import_آغاز) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } پکڑو (ثابت std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = جھوٹ;
                        ov_داخلہ_نام.clear();
                        ov_output_نام.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                اگر (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_کلید = exact_کلید;
                    shared_it = shared_cache->graphs.find(shared_کلید);
                    blob_راستہ = ggml_openvino_model_cache_blob_راستہ(model_cache_dir, model_fp);
                    manifest_راستہ = ggml_openvino_model_cache_manifest_راستہ(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            اگر (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_راستہ.c_str());
                واپس GGML_حالت_FAILED;
            }
            اگر (!imported && ggml_openvino_وزن_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                واپس GGML_حالت_FAILED;
            }
            اگر (imported) {
                decoder_انتہا_وقت = conversion_انتہا_وقت = compile_انتہا_وقت = ggml_وقت_us();
            } ورنہ {
                خودکار model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_انتہا_وقت = ggml_وقت_us();

                خودکار داخلہ_model = std::make_shared<ov::frontend::ggml::داخلہ_Model>(ggml_decoder);
                model = ov::frontend::ggml::Front_انتہا::convert(داخلہ_model);
                ggml_decoder->clear_model_weights();
                conversion_انتہا_وقت = ggml_وقت_us();

                اگر (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    حرف timestamped_filename[64];
                    خودکار timestamp = (طویل طویل) ggml_وقت_us();
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                ثابت ov::AnyMap & compile_config = model_cache_dir.خالی_خانے() ? config : mc_config;
                ov::CompiledModel compiled_model;
                خودکار remote_context = ggml_openvino_get_remote_context();
                اگر (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } ورنہ {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_انتہا_وقت = ggml_وقت_us();

                برائے (ثابت خودکار & ov_param : model->get_parameters()) {
                    ov_داخلہ_نام.push_back(ov_param->get_friendly_name());
                }
                برائے (ثابت خودکار & ov_output : model->get_results()) {
                    ov_output_نام.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                اگر (!model_cache_dir.خالی_خانے() && !model_is_splitted && model_fp != 0) {
                    کوشش {
                        ثابت std::string blob_tmp = ggml_openvino_model_cache_عارضی_راستہ(blob_راستہ);
                        ثابت std::string manifest_tmp = ggml_openvino_model_cache_عارضی_راستہ(manifest_راستہ);
                        اگر (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_داخلہ_نام,
                                                                     ov_output_نام)) {
                            std::ofstream blob_برآمد(blob_tmp, std::ios::binary | std::ios::trunc);
                            اگر (blob_برآمد.is_open()) {
                                compiled_model.export_model(blob_برآمد);
                                blob_برآمد.close();
                                اگر (blob_برآمد.good()) {
                                    اگر (std::rename(blob_tmp.c_str(), blob_راستہ.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_راستہ.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_راستہ.c_str());
                                    } ورنہ {
                                        std::remove(blob_tmp.c_str());
                                        std::remove(manifest_tmp.c_str());
                                    }
                                } ورنہ {
                                    std::remove(blob_tmp.c_str());
                                    std::remove(manifest_tmp.c_str());
                                }
                            } ورنہ {
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    } پکڑو (ثابت std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache export failed: %s\n", e.what());
                    }
                }

                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                shared_model = compiled_model;
                اندراج_2->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            اندراج_2->ptr = ggml_decoder;
            اگر (!shared_کلید.خالی_خانے() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_کلید,
                                             ov_compiled_گراف{shared_model, {}, ov_داخلہ_نام, ov_output_نام});
            }
            اگر (cache_enabled) {
                std::lock_guard<std::mutex> map_تالا(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[کلید] = infer_request;
                r_ctx->ov_داخلہ_نام_cache[کلید] = ov_داخلہ_نام;
                r_ctx->ov_output_نام_cache[کلید] = ov_output_نام;
            }

            اگر (stateful_kv_only && cache_enabled) {
                ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
                خودکار pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                ثابت int32_t pos_begin = ((int32_t *) inp_pos->ڈیٹا)[0];
                اگر (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    واپس GGML_حالت_FAILED;
                }
                r_ctx->stateful_kv_سائز = pos_shape[3];
                ثابت خودکار kv_param_res_نام = ggml_decoder->get_kv_param_res_نام();
                برائے (ثابت خودکار & pair : kv_param_res_نام) {
                    r_ctx->kv_state_داخلہ_name_map[pair.پہلا + pair.second] = pair.پہلا;
                }
            }
        }

        اگر (stateful_recurrent) {
            ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
            ثابت int32_t pos_begin = static_cast<ثابت int32_t *>(inp_pos->ڈیٹا)[0];
            اگر (pos_begin == 0) {
                infer_request->reset_state();
            } ورنہ اگر (!cache_کامیابی || old_m_params.kv_بفر_بدلا(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_سائز) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                واپس GGML_حالت_FAILED;
            }
        } ورنہ {
            reset_single_خانہ_recurrent_cache(cgraph, m_params, c_params);
        }

        برائے (size_t i = 0; i < ov_داخلہ_نام.size(); i++) {
            ثابت خودکار & param_name = ov_داخلہ_نام[i];
            خودکار داخلہ_tensor = get_ov_داخلہ_tensor(ggml_decoder, param_name);
            infer_request->سیٹ_داخلہ_tensor(i, داخلہ_tensor);

            اگر (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_داخلہ_tensor_info(param_name, داخلہ_tensor);
            }
        }

        برائے (size_t i = 0; i < ov_output_نام.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            ثابت خودکار & model_outputs = ggml_decoder->get_model_outputs();
            خودکار model_output_it = model_outputs.find(ov_output_نام[i]);
            اگر (model_output_it == model_outputs.end()) {
                جاری;
            }
            خودکار * ggml_tensor = model_output_it->second;
            اگر (ggml_nbytes(ggml_tensor) == 0) {
                جاری;
            }
            خودکار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->سیٹ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_آغاز = ggml_وقت_us();
        infer_request->infer();
        infer_انتہا_وقت = ggml_وقت_us();
        اگر (stateful_recurrent) {
            ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_سائز =
                static_cast<ثابت int32_t *>(inp_pos->ڈیٹا)[0] + get_inp_pos_n_علامات(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        اگر (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            برائے (size_t i = 0; i < ov_output_نام.size(); i++) {
                ثابت خودکار output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_نام[i], output_tensor, output_tensor.ڈیٹا());
            }
        }

        اگر (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_انتہا_وقت - آغاز_وقت) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_کلید_compute_وقت / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_بانٹو_check_وقت / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", گراف_param_compute_وقت / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_وقت / 1000.0);
            اگر (!cache_کامیابی) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_انتہا_وقت - decoder_انتہا_وقت) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_انتہا_وقت - conversion_انتہا_وقت) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_انتہا_وقت - compile_انتہا_وقت) / 1000.0);
            GGML_LOG_INFO("    - OV raw infer time: %.3f ms \n", (infer_انتہا_وقت - ov_raw_infer_آغاز) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. Release only while holding the compilation mutex so another context cannot
    // be reading host weights during conversion/compilation. Pin the shared compiled
    // models across backend teardown; a later context can create its own request without
    // reading the dropped pages. A new, uncached graph still fails fast above.
    اگر (cache_کامیابی && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_تالا(r_ctx->compiled_cache->mutex);
        اگر (!ggml_openvino_وزن_buffers_released()) {
            ggml_openvino_release_وزن_buffers();
        }
    }

    واپس GGML_حالت_SUCCESS;
}

ov::AnyMap without_npuw(ثابت ov::AnyMap & config) {
    ov::AnyMap برآمد;
    برائے (ثابت خودکار & kv : config) {
        اگر (kv.پہلا.rfind("NPUW", 0) == 0 || kv.پہلا == "NPU_USE_NPUW") {
            جاری;
        }
        برآمد.insert(kv);
    }
    واپس برآمد;
}

شمار ggml_حالت ov_گراف_compute_static(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    خودکار & core = ov_singleton_core();

    خودکار get_prefill_chunk_سائز = [] {
        ساکن ثابت صحیح chunk_سائز = []() {
            صحیح env_prefill_chunk_سائز = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            واپس env_prefill_chunk_سائز > 0 ? env_prefill_chunk_سائز : 256;
        }();
        واپس chunk_سائز;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    ساکن std::string device = ggml_openvino_get_device_name();
    ساکن خودکار is_static = سچ;
    ساکن خودکار stateful = جھوٹ;

    خودکار prefill_chunk_سائز = get_prefill_chunk_سائز();
    ثابت خودکار & config = ggml_openvino_get_compile_config();

    اگر (is_naive(cgraph)) {
        واپس naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    خودکار آغاز_وقت = ggml_وقت_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
    ثابت منطقی no_kv_cache = m_params.is_cacheless_attn;
    ثابت خودکار is_prefill = no_kv_cache ? سچ : get_is_prefill(cgraph, inp_pos);
    ثابت ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    اگر (m_params.n_heads_kv == -1) {
        prefill_chunk_سائز = inp_pos->ne[0];
    }
    گراف_کلید کلید(cgraph);
    ساکن ثابت منطقی cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    منطقی cache_کامیابی = جھوٹ;

    int64_t decoder_انتہا_وقت;
    int64_t conversion_انتہا_وقت;
    int64_t compile_انتہا_وقت;
    int64_t infer_انتہا_وقت;
    int64_t ov_raw_infer_آغاز;
    int64_t ov_raw_infer_کل = 0;

    std::shared_ptr<decoder_runtime_ctx> اندراج_2;
    ModelParams old_m_params;

    اگر (cache_enabled) {
        std::lock_guard<std::mutex> map_تالا(r_ctx->ctx_mutex);
        خودکار it = r_ctx->decoder_cache.find(کلید);
        cache_کامیابی = it != r_ctx->decoder_cache.end();
        اگر (cache_کامیابی) {
            اندراج_2 = it->second;
        } ورنہ {
            r_ctx->clear_caches_locked();
            خودکار mutex = std::make_shared<std::mutex>();
            اندراج_2 = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[کلید] = اندراج_2;
        }
    } ورنہ {
        خودکار mutex = std::make_shared<std::mutex>();
        اندراج_2 = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_کامیابی = جھوٹ;
    }

    std::lock_guard<std::mutex> تالا(*(اندراج_2->mutex));
    cache_کامیابی = cache_کامیابی && اندراج_2->ptr && r_ctx->infer_request_cache.count(کلید) != 0 &&
                r_ctx->infer_request_cache_prefill.count(کلید) != 0;

    اگر (cache_کامیابی) {
        ggml_decoder = اندراج_2->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_کامیابی = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_داخلہ_نام_local;
    std::vector<std::string> ov_output_نام_local;

    اگر (cache_کامیابی) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->سیٹ_model_params(m_params);
        ggml_decoder->سیٹ_compute_params(c_params);
        اگر (old_m_params.kv_بفر_بدلا(m_params) || !ggml_decoder->is_bound_مقصد(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->جوڑو_extra_inputs();
        {
            std::lock_guard<std::mutex> map_تالا(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(کلید) : r_ctx->infer_request_cache.at(کلید);
            ov_داخلہ_نام_local = r_ctx->ov_داخلہ_نام_cache.at(کلید);
            ov_output_نام_local = r_ctx->ov_output_نام_cache.at(کلید);
        }

        decoder_انتہا_وقت = ggml_وقت_us();
        conversion_انتہا_وقت = decoder_انتہا_وقت;
        compile_انتہا_وقت = decoder_انتہا_وقت;
    } ورنہ {
        اگر (cache_enabled) {
            std::lock_guard<std::mutex> map_تالا(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(کلید);
            r_ctx->infer_request_cache_prefill.erase(کلید);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        خودکار shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_تالا(shared_cache->mutex);
        خودکار وزن_نام = get_وزن_نام(cgraph);
        خودکار local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, وزن_نام, is_static,
                                                             stateful, جھوٹ, is_prefill, prefill_chunk_سائز);
        ثابت std::string shared_کلید =
            cache_enabled ? compiled_گراف_کلید(cgraph, *local_decoder, device, prefill_chunk_سائز) : "";
        خودکار shared_it = shared_cache->graphs.find(shared_کلید);
        اگر (!shared_کلید.خالی_خانے() && shared_it != shared_cache->graphs.end()) {
            خودکار & compiled = shared_it->second;
            خودکار prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            خودکار decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            اندراج_2->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_داخلہ_نام_local = compiled.داخلہ_نام;
            ov_output_نام_local = compiled.output_نام;
            r_ctx->infer_request_cache_prefill[کلید] = prefill_request;
            r_ctx->infer_request_cache[کلید] = decode_request;
            r_ctx->ov_داخلہ_نام_cache[کلید] = ov_داخلہ_نام_local;
            r_ctx->ov_output_نام_cache[کلید] = ov_output_نام_local;
            decoder_انتہا_وقت = conversion_انتہا_وقت = compile_انتہا_وقت = ggml_وقت_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } ورنہ {
            std::shared_ptr<ov::Model> model;
            خودکار model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph);

            خودکار ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, جھوٹ, سچ, prefill_chunk_سائز);
            خودکار ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, جھوٹ, جھوٹ, prefill_chunk_سائز);
            decoder_انتہا_وقت = ggml_وقت_us();

            ثابت منطقی dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            ثابت خودکار dump_ir_timestamp = static_cast<طویل طویل>(ggml_وقت_us());

            خودکار build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          ثابت std::shared_ptr<GgmlOvDecoder> & decoder, ثابت حرف * لیبل,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_انتہا_وقت, int64_t & local_compile_انتہا_وقت) {
                خودکار داخلہ_model = std::make_shared<ov::frontend::ggml::داخلہ_Model>(decoder);
                model = ov::frontend::ggml::Front_انتہا::convert(داخلہ_model);
                decoder->clear_model_weights();
                local_conversion_انتہا_وقت = ggml_وقت_us();

                اگر (dump_ir) {
                    حرف timestamped_filename[64];
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%s_%lld.xml", لیبل,
                             dump_ir_timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                compiled_model = core.compile_model(model, device, compile_config);
                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                local_compile_انتہا_وقت = ggml_وقت_us();
            };
            std::shared_ptr<ov::Model> model_prefill;
            std::shared_ptr<ov::Model> model_decode;
            ov::CompiledModel compiled_model_prefill;
            ov::CompiledModel compiled_model_decode;
            std::shared_ptr<ov::InferRequest> infer_request_prefill;
            std::shared_ptr<ov::InferRequest> infer_request_decode;
            int64_t prefill_conversion_انتہا_وقت;
            int64_t decode_conversion_انتہا_وقت;
            int64_t prefill_compile_انتہا_وقت;
            int64_t decode_compile_انتہا_وقت;
            build_static_model(ggml_decoder_prefill, "prefill", model_prefill, compiled_model_prefill,
                               infer_request_prefill, prefill_conversion_انتہا_وقت, prefill_compile_انتہا_وقت);
            اگر (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_انتہا_وقت = prefill_conversion_انتہا_وقت;
                decode_compile_انتہا_وقت = prefill_compile_انتہا_وقت;
            } ورنہ {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_انتہا_وقت, decode_compile_انتہا_وقت);
            }
            conversion_انتہا_وقت = std::max(prefill_conversion_انتہا_وقت, decode_conversion_انتہا_وقت);
            compile_انتہا_وقت = std::max(prefill_compile_انتہا_وقت, decode_compile_انتہا_وقت);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            اندراج_2->ptr = ggml_decoder;

            برائے (ثابت خودکار & ov_param : model->get_parameters()) {
                ov_داخلہ_نام_local.push_back(ov_param->get_friendly_name());
            }
            برائے (ثابت خودکار & ov_output : model->get_results()) {
                ov_output_نام_local.push_back(ov_output->get_friendly_name());
            }

            اگر (!shared_کلید.خالی_خانے()) {
                shared_cache->graphs.emplace(
                    shared_کلید, ov_compiled_گراف{compiled_model_decode, compiled_model_prefill, ov_داخلہ_نام_local,
                                                  ov_output_نام_local});
            }

            اگر (cache_enabled) {
                std::lock_guard<std::mutex> map_تالا(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[کلید] = infer_request_prefill;
                r_ctx->infer_request_cache[کلید] = infer_request_decode;
                r_ctx->ov_داخلہ_نام_cache[کلید] = ov_داخلہ_نام_local;
                r_ctx->ov_output_نام_cache[کلید] = ov_output_نام_local;
            }
        }
    }

    reset_single_خانہ_recurrent_cache(cgraph, m_params, c_params);

    اگر (is_prefill) {
        خودکار inp_len = get_inp_pos_n_علامات(cgraph, inp_pos);
        برائے (صحیح chunk_اشاریہ = 0; chunk_اشاریہ * prefill_chunk_سائز < inp_len; chunk_اشاریہ++) {
            برائے (size_t i = 0; i < ov_داخلہ_نام_local.size(); i++) {
                ثابت خودکار & param_name = ov_داخلہ_نام_local[i];
                خودکار داخلہ_tensor = get_ov_داخلہ_tensor_static_prefill(ggml_decoder, param_name, chunk_اشاریہ);
                infer_request->سیٹ_داخلہ_tensor(i, داخلہ_tensor);

                اگر (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    ثابت خودکار داخلہ_tensor = infer_request->get_داخلہ_tensor(i);
                    print_داخلہ_tensor_info(param_name, داخلہ_tensor);
                }
            }

            برائے (size_t i = 0; i < ov_output_نام_local.size(); i++) {
                ثابت خودکار & model_outputs = ggml_decoder->get_model_outputs();
                خودکار model_output_it = model_outputs.find(ov_output_نام_local[i]);
                اگر (model_output_it == model_outputs.end()) {
                    جاری;
                }
                خودکار * ggml_tensor = model_output_it->second;
                اگر (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    جاری;
                }
                خودکار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->سیٹ_output_tensor(i, output_tensor);
            }

            ov_raw_infer_آغاز = ggml_وقت_us();
            infer_request->infer();
            ov_raw_infer_کل += ggml_وقت_us() - ov_raw_infer_آغاز;

            اگر (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                برائے (size_t i = 0; i < ov_output_نام_local.size(); i++) {
                    ثابت خودکار output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_نام_local[i], output_tensor, output_tensor.ڈیٹا());
                }
            }
        }
        infer_انتہا_وقت = ggml_وقت_us();
    } ورنہ {
        برائے (size_t i = 0; i < ov_داخلہ_نام_local.size(); i++) {
            ثابت خودکار & param_name = ov_داخلہ_نام_local[i];
            خودکار داخلہ_tensor = get_ov_داخلہ_tensor_static_decode(ggml_decoder, param_name);
            infer_request->سیٹ_داخلہ_tensor(i, داخلہ_tensor);

            اگر (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                ثابت خودکار داخلہ_tensor = infer_request->get_داخلہ_tensor(i);
                print_داخلہ_tensor_info(param_name, داخلہ_tensor);
            }
        }

        برائے (size_t i = 0; i < ov_output_نام_local.size(); i++) {
            ثابت خودکار & model_outputs = ggml_decoder->get_model_outputs();
            خودکار model_output_it = model_outputs.find(ov_output_نام_local[i]);
            اگر (model_output_it == model_outputs.end()) {
                جاری;
            }
            خودکار * ggml_tensor = model_output_it->second;
            اگر (ggml_nbytes(ggml_tensor) == 0) {
                جاری;
            }
            خودکار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->سیٹ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_آغاز = ggml_وقت_us();
        infer_request->infer();
        infer_انتہا_وقت = ggml_وقت_us();
        ov_raw_infer_کل = infer_انتہا_وقت - ov_raw_infer_آغاز;

        اگر (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            برائے (size_t i = 0; i < ov_output_نام_local.size(); i++) {
                ثابت خودکار output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_نام_local[i], output_tensor, output_tensor.ڈیٹا());
            }
        }
    }

    اگر (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_انتہا_وقت - آغاز_وقت) / 1000.0);
        اگر (!cache_کامیابی) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_انتہا_وقت - decoder_انتہا_وقت) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_انتہا_وقت - conversion_انتہا_وقت) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_انتہا_وقت - compile_انتہا_وقت) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_کل / 1000.0);
    }

    واپس GGML_حالت_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
ساکن ggml_cgraph * get_compute_گراف(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    خودکار is_skipped = [](ثابت ggml_tensor * گرہ) {
        واپس گرہ->op != GGML_OP_NONE && !(گرہ->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    اگر (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        واپس cgraph;
    }
    خودکار & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](ثابت ggml_tensor * گرہ) { واپس !is_skipped(گرہ); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.ڈیٹا();
    compute.graph.n_nodes = (صحیح) compute.nodes.size();
    واپس &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
شمار ggml_حالت ov_گراف_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    کوشش {
        اگر (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        ثابت خودکار is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != خالی_اشاریہ);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_تالا(r_ctx->execution_mutex);
        cgraph = get_compute_گراف(cgraph, *r_ctx);

        واپس is_static ? ov_گراف_compute_static(cgraph, r_ctx) : ov_گراف_compute_dynamic(cgraph, r_ctx);
    } پکڑو (ثابت ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        واپس GGML_حالت_FAILED;
    } پکڑو (ثابت std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        واپس GGML_حالت_FAILED;
    } پکڑو (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        واپس GGML_حالت_FAILED;
    }
}

// Detect whether a cgraph is a split subgraph or not.
// Step 1 compares each node's recorded use_count with actual fan-out references in node->src.
// Step 2 verifies that node inputs come from model nodes/weights/leafs; external sources imply split.
منطقی is_model_splitted(ggml_cgraph * cgraph) {
    ساکن ثابت منطقی fallback_enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_FALLBACK") != 0;
    اگر (!fallback_enabled) {
        واپس جھوٹ;
    }

    // Backend op tests execute each node through ggml_graph_view(), which preserves the original
    // graph use_counts while exposing only one node. Treat those single-node views as regular
    // naive graphs so intermediate ops do not look like split-model fragments.
    اگر (cgraph->n_nodes <= 1 && cgraph->n_leafs == 0) {
        واپس جھوٹ;
    }

    // check the nodes of the model are used by the following nodes, through compare the node's use count and the count of nodes that use it as input. If does not match, return true, else return false.
    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = cgraph->nodes[i];
        صحیح use_گنتی = cgraph->use_counts[ggml_hash_ڈھونڈو(&cgraph->دیکھا_hash_سیٹ, گرہ)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        اگر ((cgraph->n_nodes <= 1 && use_گنتی == 0) ||
            (cgraph->n_nodes <= 1 && گرہ->op == GGML_OP_VIEW && use_گنتی == 1 && گرہ->src[0] != خالی_اشاریہ &&
             گرہ->src[0]->op == GGML_OP_NONE)) {
            واپس جھوٹ;
        }
        اگر (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_ترتیب_بدلو)) {
            واپس جھوٹ;
        }
        صحیح داخلہ_use_گنتی = 0;
        برائے (صحیح j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_گرہ = cgraph->nodes[j];
            برائے (صحیح k = 0; k < GGML_MAX_SRC; k++) {
                اگر (other_گرہ->src[k] == گرہ) {
                    داخلہ_use_گنتی++;
                }
            }
        }
        اگر (use_گنتی != داخلہ_use_گنتی && گرہ->op != GGML_OP_NONE) {
            واپس سچ;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    اگر (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_وزن_نام(cgraph);
    } ورنہ {
        برائے (ثابت خودکار & kv : GgmlOvDecoder::create_وزن_nodes(cgraph, سچ)) {
            model_weights.insert(kv.پہلا);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = cgraph->nodes[i];
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = گرہ->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            اگر (src != خالی_اشاریہ && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.خالی_خانے() == جھوٹ &&
                model_leafs.find(src) == model_leafs.end()) {
                اگر (GgmlOvDecoder::is_inp_tok(src, گرہ)) {
                    واپس جھوٹ;
                }
                واپس سچ;
            }
        }
    }
    واپس جھوٹ;
}

منطقی is_naive(ggml_cgraph * cgraph) {
    constexpr صحیح naive_گراف_سائز_حد = 20;
    صحیح count = 0;
    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        اگر (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    واپس count < naive_گراف_سائز_حد;
}

size_t checksum(ثابت خالی * ڈیٹا, size_t size) {
    ثابت uint8_t * بائٹیں = static_cast<ثابت uint8_t *>(ڈیٹا);
    size_t مجموع = 0;
    برائے (size_t i = 0; i < size; ++i) {
        مجموع += (uint8_t) i;
        مجموع += بائٹیں[i];
    }
    واپس مجموع;
}

منطقی save_ggml_tensor_ڈیٹا_مقصد_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_راستہ) {
    اگر (tensor == خالی_اشاریہ || tensor->ڈیٹا == خالی_اشاریہ) {
        واپس جھوٹ;
    }

    std::ofstream برآمد(file_راستہ);
    اگر (!برآمد.is_open()) {
        واپس جھوٹ;
    }

    ثابت size_t n = ggml_nelements(tensor);
    برآمد << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    انتخاب (tensor->type) {
    حالت GGML_TYPE_F32: {
        ثابت خودکار * ڈیٹا = static_cast<ثابت اعشاری *>(tensor->ڈیٹا);
        برائے (size_t i = 0; i < n; ++i) {
            برآمد << ڈیٹا[i] << '\n';
        }
        توڑو;
    }
    حالت GGML_TYPE_F16: {
        ثابت خودکار * ڈیٹا = static_cast<ثابت ggml_fp16_t *>(tensor->ڈیٹا);
        برائے (size_t i = 0; i < n; ++i) {
            برآمد << ggml_fp16_مقصد_fp32(ڈیٹا[i]) << '\n';
        }
        توڑو;
    }
    حالت GGML_TYPE_BF16: {
        ثابت خودکار * ڈیٹا = static_cast<ثابت ggml_bf16_t *>(tensor->ڈیٹا);
        برائے (size_t i = 0; i < n; ++i) {
            برآمد << ggml_bf16_مقصد_fp32(ڈیٹا[i]) << '\n';
        }
        توڑو;
    }
    حالت GGML_TYPE_I32: {
        ثابت خودکار * ڈیٹا = static_cast<ثابت int32_t *>(tensor->ڈیٹا);
        برائے (size_t i = 0; i < n; ++i) {
            برآمد << ڈیٹا[i] << '\n';
        }
        توڑو;
    }
    حالت GGML_TYPE_I64: {
        ثابت خودکار * ڈیٹا = static_cast<ثابت int64_t *>(tensor->ڈیٹا);
        برائے (size_t i = 0; i < n; ++i) {
            برآمد << ڈیٹا[i] << '\n';
        }
        توڑو;
    }
    طےشدہ:
        برآمد << "unsupported tensor type for text dump" << '\n';
        واپس جھوٹ;
    }

    واپس سچ;
}

خالی print_داخلہ_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor) {
    std::خارج << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.ڈیٹا()
              << '\n';
    انتخاب (tensor.get_element_type()) {
    حالت ov::element::f32: {
        اگر (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::خارج << *(tensor.ڈیٹا<اعشاری>()) << '\n';
        } ورنہ {
            size_t صفیں = tensor.get_shape()[2];
            size_t ستون_تعداد_2 = tensor.get_shape()[3];
            ثابت اعشاری * ڈیٹا = tensor.ڈیٹا<اعشاری>();
            برائے (size_t i = 0; i < صفیں; ++i) {
                برائے (size_t j = 0; j < ستون_تعداد_2; ++j) {
                    اعشاری val = ڈیٹا[i * ستون_تعداد_2 + j];
                    اگر (std::isinf(val) && val < 0) {
                        std::خارج << std::setw(5) << "-inf";
                    } ورنہ {
                        std::خارج << std::setw(5) << val;
                    }
                }
                std::خارج << '\n';
            }
        }

        توڑو;
    }
    حالت ov::element::f16:
        std::خارج << *(tensor.ڈیٹا<ov::float16>()) << '\n';
        توڑو;
    حالت ov::element::i32:
        برائے (size_t i = 0; i < tensor.get_سائز(); ++i) {
            std::خارج << tensor.ڈیٹا<int32_t>()[i] << ' ';
        }
        std::خارج << '\n';
        توڑو;
    حالت ov::element::i64:
        برائے (size_t i = 0; i < tensor.get_سائز(); ++i) {
            std::خارج << tensor.ڈیٹا<int64_t>()[i] << ' ';
        }
        std::خارج << '\n';
        توڑو;
    طےشدہ:
        توڑو;
    }
}

خالی print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت خالی * output_dst) {
    std::خارج << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    خودکار print_float_stats = [](ثابت std::string & type_name, size_t size, خودکار get_قدر) {
        اگر (size == 0) {
            واپس;
        }

        اعشاری پہلا = get_قدر(0);
        اعشاری min = پہلا;
        اعشاری max = پہلا;
        دوہرا مجموع = پہلا;

        برائے (size_t i = 1; i < size; ++i) {
            اعشاری v = get_قدر(i);
            min = std::min(v, min);
            max = std::max(v, max);
            مجموع += v;
        }
        دوہرا اوسط_2 = مجموع / size;

        std::خارج << std::دائیں << std::setw(6) << type_name << std::دائیں << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::خارج << std::دائیں << std::setw(6) << "" << std::دائیں << std::setw(12) << پہلا << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << اوسط_2 << '\n';
    };

    انتخاب (tensor.get_element_type()) {
    حالت ov::element::f32: {
        ثابت اعشاری * ڈیٹا = tensor.ڈیٹا<اعشاری>();
        size_t size = tensor.get_سائز();
        print_float_stats("[f32]", size, [ڈیٹا](size_t i) { واپس ڈیٹا[i]; });
        توڑو;
    }
    حالت ov::element::f16: {
        ثابت ov::float16 * ڈیٹا = tensor.ڈیٹا<ov::float16>();
        size_t size = tensor.get_سائز();
        print_float_stats("[f16]", size, [ڈیٹا](size_t i) { واپس static_cast<اعشاری>(ڈیٹا[i]); });
        توڑو;
    }
    طےشدہ:
        توڑو;
    }
}

ثابت ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    برائے (صحیح i = 0; i < cgraph->n_nodes; ++i) {
        خودکار * op = cgraph->nodes[i];
        برائے (صحیح j = 0; j < GGML_MAX_SRC; ++j) {
            خودکار * src = op->src[j];
            اگر (src == خالی_اشاریہ) {
                توڑو;
            }
            اگر (GgmlOvDecoder::is_inp_pos(src, op)) {
                واپس src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    پھینکو std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

int64_t get_inp_pos_n_علامات(ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    صحیح n_planes = 1;
    برائے (صحیح i = 0; i < cgraph->n_nodes; ++i) {
        خودکار * op = cgraph->nodes[i];
        برائے (صحیح j = 0; j < GGML_MAX_SRC; ++j) {
            اگر (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                توڑو;
            }
        }
    }
    واپس inp_pos->ne[0] / n_planes;
}

منطقی get_is_prefill(ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos) {
    واپس get_inp_pos_n_علامات(cgraph, inp_pos) > 1;
}
