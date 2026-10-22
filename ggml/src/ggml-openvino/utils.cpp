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
    ساكن ثابت منطقي kv_ٹکڑا_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    إذا (kv_ٹکڑا_disabled) {
        أرجع std::nullopt;
    }
    إذا (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        أرجع std::nullopt;
    }
    إذا (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != خالی_سوچک) {
        أرجع std::nullopt;
    }
    ثابت خودکار * op = ggml_decoder->get_tensor_ورتیا_op(ggml_tensor);
    إذا (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        أرجع std::nullopt;
    }

    ثابت خودکار & compute_params = ggml_decoder->get_compute_params();
    إذا (compute_params.n_seq_سرگرم != 1 || compute_params.seq_سرگرم_شروع != 0) {
        أرجع std::nullopt;
    }

    صحيح layer;
    إذا (خودکار layer_opt = کڈھو_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } وإلا {
        أرجع std::nullopt;
    }

    ثابت منطقي is_swa = ggml_decoder->is_swa_layer(layer);
    إذا (is_swa) {
        أرجع std::nullopt;
    }
    ثابت صحيح ctx_فی_seq = ggml_decoder->get_ctx_فی_seq();
    ثابت صحيح n_kv = compute_params.attention_ناپ;
    إذا (ctx_فی_seq <= 0 || n_kv <= 0 || n_kv >= ctx_فی_seq) {
        أرجع std::nullopt;
    }

    ov::Shape بھریا_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    إذا (بھریا_shape.size() != 4 || بھریا_shape[0] != 1 || بھریا_shape[1] != 1 ||
        static_cast<صحيح>(بھریا_shape[2]) != ctx_فی_seq) {
        أرجع std::nullopt;
    }

    ov::Shape sliced_shape = بھریا_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    إذا (ggml_openvino_بفر_is_remote(ggml_tensor)) {
        خودکار remote_context = ggml_openvino_get_remote_context();
        خودکار gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        أرجع gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->انکڑے);
    }

    أرجع ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->انکڑے);
}

ساكن uint64_t ggml_openvino_model_cache_extra_cfg(منطقي stateful, منطقي recurrent) {
    ثابت حرف * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    ثابت منطقي manual_gqa_enabled = manual_gqa_env != خالی_سوچک ?
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

std::map<std::string, std::shared_ptr<ov::Node>> get_بھار_ناں(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> ناں;
    لكل (ثابت خودکار & name : GgmlOvDecoder::collect_بھار_ناں(cgraph)) {
        ناں[name] = خالی_سوچک;
    }
    أرجع ناں;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
ساكن std::string compiled_گراف_کلید(ثابت ggml_cgraph * graph,
                                      ثابت GgmlOvDecoder & decoder,
                                      ثابت std::string & device,
                                      صحيح prefill_chunk_ناپ = 0,
                                      منطقي disk_cache = خطأ) {
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
    append(prefill_chunk_ناپ);
    منطقي has_بھار_بفر_id = خطأ;
    std::unordered_map<ثابت ggml_tensor *, size_t> ids;
    std::function<فراغ(ثابت ggml_tensor *)> ویکھو = [&](ثابت ggml_tensor * tensor) {
        إذا (!tensor) {
            append(size_t(0));
            أرجع;
        }
        خودکار inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.پہلا->second);
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
        ثابت خودکار * ادھار = tensor->view_src ? tensor->view_src : tensor;
        ثابت منطقي بھار = ادھار->بفر && ادھار->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS;
        append(بھار);
        إذا (بھار && !disk_cache) {
            ثابت size_t بفر_id = ggml_backend_openvino_بفر_get_ctx_id(ادھار->بفر);
            has_بھار_بفر_id |= بفر_id != 0;
            append(بفر_id);
            append(tensor->انکڑے);
        }
        ویکھو(tensor->view_src);
        لكل (ثابت خودکار * src : tensor->src) {
            ویکھو(src);
        }
    };
    append(graph->n_nodes);
    لكل (صحيح i = 0; i < graph->n_nodes; ++i) {
        ویکھو(graph->nodes[i]);
    }
    append(graph->n_leafs);
    لكل (صحيح i = 0; i < graph->n_leafs; ++i) {
        ویکھو(graph->leafs[i]);
    }
    لكل (ثابت خودکار & داخلہ : decoder.get_model_extra_inputs()) {
        append_string(داخلہ.پہلا);
        append_string(داخلہ.second.type.get_type_name());
        append(داخلہ.second.shape.size());
        لكل (خودکار dim : داخلہ.second.shape) {
            append(dim);
        }
        append(داخلہ.second.is_parameter);
        إذا (!داخلہ.second.is_parameter) {
            append(داخلہ.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    أرجع disk_cache || has_بھار_بفر_id ? کلید : std::string{};
}

ساكن std::string dynamic_گراف_signature(ثابت ggml_cgraph * graph,
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
    لكل (ثابت خودکار & name : گراف_id.داخلہ_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    لكل (صحيح layer : params.swa_layers) {
        append(layer);
    }
    لكل (ثابت خودکار & [layer, heads] : params.n_heads_kv_فی_layer) {
        append(layer);
        append(heads);
    }
    لكل (ثابت خودکار & [name, داخلہ] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(داخلہ.type.get_type_name());
    }
    لكل (ثابت خودکار & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    لكل (ثابت خودکار & [name, داخلہ] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(داخلہ.type.get_type_name());
        append(داخلہ.is_parameter);
        إذا (!داخلہ.is_parameter) {
            append(داخلہ.value);
        }
    }
    أرجع کلید;
}

ساكن منطقي compiled_model_matches_گراف(ثابت ov::CompiledModel & model,
                                         ثابت std::vector<std::string> & inputs,
                                         ثابت std::vector<std::string> & outputs,
                                         ثابت GgmlOvDecoder & decoder) {
    إذا (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        أرجع خطأ;
    }
    ثابت خودکار & گراف_inputs = decoder.get_model_inputs();
    ثابت خودکار & extra_inputs = decoder.get_model_extra_inputs();
    ثابت خودکار & گراف_outputs = decoder.get_model_outputs();
    لكل (size_t i = 0; i < inputs.size(); ++i) {
        ثابت خودکار & پورٹ = model.داخلہ(i);
        خودکار گراف_it = گراف_inputs.find(inputs[i]);
        إذا (گراف_it != گراف_inputs.end()) {
            إذا (پورٹ.get_element_type() != گراف_it->second.type ||
                !پورٹ.get_partial_shape().compatible(گراف_it->second.shape)) {
                أرجع خطأ;
            }
            استمر;
        }
        خودکار extra_it = extra_inputs.find(inputs[i]);
        إذا (extra_it == extra_inputs.end() || پورٹ.get_element_type() != extra_it->second.type ||
            !پورٹ.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            أرجع خطأ;
        }
    }
    لكل (size_t i = 0; i < outputs.size(); ++i) {
        خودکار گراف_it = گراف_outputs.find(outputs[i]);
        إذا (گراف_it == گراف_outputs.end()) {
            إذا (outputs[i].compare(0, 8, "__debug_") != 0) {
                أرجع خطأ;
            }
            استمر;
        }
        ثابت خودکار & پورٹ = model.output(i);
        إذا (پورٹ.get_element_type() != GgmlOvDecoder::get_ov_type(گراف_it->second) ||
            !پورٹ.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(گراف_it->second)))) {
            أرجع خطأ;
        }
    }
    أرجع سچ;
}

ساكن std::string ov_profiling_csv_میدان(ثابت std::string & value) {
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

    ساكن std::atomic<uint64_t> inference_اشاریہ{0};
    ثابت uint64_t اشاریہ = inference_اشاریہ.fetch_add(1);
    ثابت std::string path = "openvino_profile_" + std::to_string(اشاریہ) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    إذا (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        أرجع;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t کل_real_ویلا = 0;
    int64_t کل_cpu_ویلا = 0;
    size_t executed_گنتی = 0;
    لكل (ثابت خودکار & info : infer_request.get_profiling_info()) {
        ثابت حرف * حالت = "NOT_RUN";
        إذا (info.حالت == ov::ProfilingInfo::حالت_2::EXECUTED) {
            حالت = "EXECUTED";
            کل_real_ویلا += info.real_ویلا.count();
            کل_cpu_ویلا += info.cpu_ویلا.count();
            executed_گنتی++;
        } وإلا إذا (info.حالت == ov::ProfilingInfo::حالت_2::OPTIMIZED_باہرلا) {
            حالت = "OPTIMIZED_OUT";
        }

        output << حالت << ',' << info.real_ویلا.count() << ',' << info.cpu_ویلا.count() << ','
               << info.شروع_ویلا.count() << ',' << ov_profiling_csv_میدان(info.node_type) << ','
               << ov_profiling_csv_میدان(info.گرہ_name) << ',' << ov_profiling_csv_میدان(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_گنتی, کل_real_ویلا / 1000.0, کل_cpu_ویلا / 1000.0);
}

ساكن فراغ reset_single_تھاں_recurrent_cache(ggml_cgraph * cgraph,
                                              ثابت ModelParams & model_params,
                                              ثابت ComputeParams & compute_params) {
    إذا (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        أرجع;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_بفر_t> buffers;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        خودکار * گرہ = cgraph->nodes[i];
        إذا (گرہ->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(گرہ->view_src) &&
            گرہ->view_src->ne[1] == 1 && گرہ->view_src->بفر != خالی_سوچک) {
            buffers.insert(گرہ->view_src->بفر);
        }
    }
    لكل (خودکار * بفر : buffers) {
        ggml_backend_بفر_clear(بفر, 0);
    }
}

ov::Tensor create_ov_output_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   ثابت std::shared_ptr<ov::InferRequest> & infer_request,
                                   صحيح output_اشاریہ,
                                   ثابت ggml_tensor * ggml_tensor) {
    إذا (خودکار sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        أرجع *sliced;
    }

    خودکار output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    فراغ * output_انکڑے = ggml_tensor->انکڑے;
    إذا (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_اشاریہ).get_shape();
    } وإلا {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        إذا (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != خالی_سوچک &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_انکڑے = ggml_tensor->view_src->انکڑے;
        } وإلا {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    إذا (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_بفر_is_remote(ggml_tensor) &&
        ggml_tensor->extra != خالی_سوچک) {
        خودکار * extra_ادھار = static_cast<ggml_openvino_extra_ادھار *>(ggml_tensor->extra);
        إذا (extra_ادھار->type == ggml_openvino_extra_ادھار::Type::TENSOR) {
            خودکار * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_ادھار);
            إذا (tensor_extra->tensor != خالی_سوچک && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                أرجع *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_انکڑے);
    أرجع output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_صفاں_منزل_seq_axis_2(ثابت ov::Tensor & kv_tensor, size_t n_heads_kv) {
    ثابت size_t صفاں = kv_tensor.get_shape()[2];
    ثابت size_t سر_ناپ = kv_tensor.get_shape()[3] / n_heads_kv;
    ثابت size_t elem = kv_tensor.get_element_type().size();
    ثابت size_t سر_بائٹاں = سر_ناپ * elem;

    ov::Tensor باہرلا(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, صفاں, سر_ناپ});
    ثابت خودکار * src = static_cast<ثابت uint8_t *>(kv_tensor.انکڑے());
    خودکار * dst = static_cast<uint8_t *>(باہرلا.انکڑے());
    لكل (size_t s = 0; s < صفاں; s++) {
        لكل (size_t h = 0; h < n_heads_kv; h++) {
            حافظہ_نقل(dst + (h * صفاں + s) * سر_بائٹاں, src + (s * n_heads_kv + h) * سر_بائٹاں, سر_بائٹاں);
        }
    }
    أرجع باہرلا;
}

سانچا <قسم_ناں T> فراغ سیٹ_zero_diagonal(std::vector<T> & میٹرکس, size_t صفاں, size_t خانے_گنتی_2, T zero_قدر = T{}) {
    لكل (size_t i = 0; i < صفاں; ++i) {
        size_t diag_خانہ = std::min(i, خانے_گنتی_2 - 1);
        میٹرکس[i * خانے_گنتی_2 + diag_خانہ] = zero_قدر;
    }
}

ov::Tensor make_contiguous_ونڈو_داخلہ_tensor(ثابت بنية ggml_tensor * ggml_tensor, ثابت ov::Shape & داخلہ_shape) {
    ثابت size_t element_ناپ = ggml_type_ناپ(ggml_tensor->type);
    ثابت size_t بلاک_ناپ_3 = ggml_blck_ناپ(ggml_tensor->type);

    GGML_ASSERT(بلاک_ناپ_3 == 1 && "non-contiguous split inputs must be plain element types");

    ثابت بنية ggml_tensor * سروت_tensor = ggml_tensor->view_src != خالی_سوچک ? ggml_tensor->view_src : ggml_tensor;
    ثابت size_t سروت_وتھ = ggml_tensor->view_src != خالی_سوچک ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> سروت_انکڑے(ggml_nbytes(سروت_tensor));
    ggml_backend_tensor_get(سروت_tensor, سروت_انکڑے.انکڑے(), 0, سروت_انکڑے.size());

    ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
    خودکار * dst = static_cast<uint8_t *>(داخلہ_tensor.انکڑے());
    size_t dst_وتھ = 0;

    لكل (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        لكل (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            لكل (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                لكل (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    ثابت size_t src_وتھ = سروت_وتھ + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::حافظہ_نقل(dst + dst_وتھ, سروت_انکڑے.انکڑے() + src_وتھ, element_ناپ);
                    dst_وتھ += element_ناپ;
                }
            }
        }
    }

    أرجع داخلہ_tensor;
}

ov::Tensor convert_ggml_داخلہ_منزل_ov(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & name) {
    ثابت خودکار * ggml_tensor = ggml_decoder->get_داخلہ_ggml_tensor(name);

    إذا (خودکار sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        أرجع *sliced;
    }

    إذا (ggml_tensor->extra != خالی_سوچک && !ggml_decoder->is_splited_model()) {
        خودکار * extra_ادھار = static_cast<ggml_openvino_extra_ادھار *>(ggml_tensor->extra);
        إذا (extra_ادھار->type == ggml_openvino_extra_ادھار::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            خودکار * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_ادھار);
            أرجع *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    خودکار * داخلہ_انکڑے = ggml_tensor->انکڑے;
    ov::Shape داخلہ_shape;
    إذا (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        داخلہ_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } وإلا {
        داخلہ_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    إذا (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        أرجع make_contiguous_ونڈو_داخلہ_tensor(ggml_tensor, داخلہ_shape);
    }

    خودکار داخلہ_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape, داخلہ_انکڑے);
    أرجع داخلہ_tensor;
}

ov::Tensor get_ov_داخلہ_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & param_name) {
    ov::Tensor داخلہ_tensor;
    خودکار extra_داخلہ = ggml_decoder->get_model_extra_inputs().find(param_name);
    إذا (extra_داخلہ != ggml_decoder->get_model_extra_inputs().end()) {
        داخلہ_tensor = ov::Tensor(extra_داخلہ->second.type, extra_داخلہ->second.shape);
        *داخلہ_tensor.انکڑے<int64_t>() = extra_داخلہ->second.value;
    } وإلا {
        داخلہ_tensor = convert_ggml_داخلہ_منزل_ov(ggml_decoder, param_name);
    }
    أرجع داخلہ_tensor;
}

ov::Tensor get_ov_داخلہ_tensor_static_decode(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             ثابت std::string & param_name) {
    // NPU decoding stage
    إذا (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        أرجع get_ov_داخلہ_tensor(ggml_decoder, param_name);
    }
    ثابت خودکار * ggml_tensor = ggml_decoder->get_داخلہ_ggml_tensor(param_name);
    ثابت خودکار * op = ggml_decoder->get_tensor_ورتیا_op(ggml_tensor);

    إذا (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        ثابت صحيح n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        أكد(ggml_tensor->ne[0] == n_planes);
        ov::Shape داخلہ_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        std::حافظہ_نقل(داخلہ_tensor.انکڑے(), ggml_tensor->انکڑے, n_planes * ggml_type_ناپ(ggml_tensor->type));
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape داخلہ_shape = {1, 1, 1, 1};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        int32_t inp_باہرلا_id = *((int32_t *) ggml_tensor->انکڑے);
        أكد(ggml_tensor->ne[0] == 1);
        أكد(inp_باہرلا_id == 0);
        *داخلہ_tensor.انکڑے<int32_t>() = inp_باہرلا_id;
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_نقاب(ggml_tensor, op)) {
        size_t context_ناپ = ggml_decoder->get_ctx_ناپ();
        إذا (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_انکڑے =
                pad_داخلہ<ggml_fp16_t>(ggml_tensor, 1, context_ناپ, GGML_FP32_منزل_FP16(-INFINITY));
            ov::Tensor داخلہ_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_ناپ});
            std::حافظہ_نقل(داخلہ_tensor.انکڑے(), padded_انکڑے.انکڑے(), padded_انکڑے.size() * حجم(ggml_fp16_t));
            أرجع داخلہ_tensor;
        }

        std::vector<عائم> padded_انکڑے = pad_داخلہ<عائم>(ggml_tensor, 1, context_ناپ, -INFINITY);
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_ناپ});
        خودکار * انکڑے_ptr = داخلہ_tensor.انکڑے<عائم>();
        std::copy(padded_انکڑے.begin(), padded_انکڑے.begin() + context_ناپ, انکڑے_ptr);
        أرجع داخلہ_tensor;
    }

    أرجع get_ov_داخلہ_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_داخلہ_tensor_static_prefill(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              ثابت std::string & param_name,
                                              صحيح chunk_اشاریہ) {
    // NPU prompt processing stage
    ثابت size_t داخلہ_len = ggml_decoder->get_داخلہ_len();
    ثابت size_t chunk_ناپ = ggml_decoder->m_prefill_chunk_ناپ;
    ثابت size_t chunk_valid_ناپ = std::min(chunk_ناپ, داخلہ_len - chunk_اشاریہ * chunk_ناپ);
    ثابت size_t chunk_pad_ناپ = chunk_ناپ - chunk_valid_ناپ;

    إذا (param_name == "chunk_valid_len") {
        ov::Tensor داخلہ_tensor(ov::element::i64, ov::Shape{1});
        *داخلہ_tensor.انکڑے<int64_t>() = (int64_t) chunk_valid_ناپ;
        أرجع داخلہ_tensor;
    }
    إذا (chunk_اشاریہ > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor داخلہ_tensor(ov::element::i64, ov::Shape{1});
        *داخلہ_tensor.انکڑے<int64_t>() = 0;
        أرجع داخلہ_tensor;
    }
    إذا (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        أرجع get_ov_داخلہ_tensor(ggml_decoder, param_name);
    }
    ثابت خودکار * ggml_tensor = ggml_decoder->get_داخلہ_ggml_tensor(param_name);
    ثابت خودکار * op = ggml_decoder->get_tensor_ورتیا_op(ggml_tensor);

    إذا (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        ثابت صحيح n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        ثابت size_t element_ناپ = ggml_type_ناپ(ggml_tensor->type);
        ov::Shape داخلہ_shape = {1, 1, 1, (size_t) n_planes * chunk_ناپ};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        لكل (صحيح p = 0; p < n_planes; p++) {
            ثابت حرف * src =
                (ثابت حرف *) ggml_tensor->انکڑے + (p * داخلہ_len + chunk_اشاریہ * chunk_ناپ) * element_ناپ;
            حرف * dst = (حرف *) داخلہ_tensor.انکڑے() + p * chunk_ناپ * element_ناپ;
            std::حافظہ_نقل(dst, src, chunk_valid_ناپ * element_ناپ);
            إذا (chunk_pad_ناپ > 0) {
                إذا (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t آخری_قدر = *((ثابت int32_t *) src + chunk_valid_ناپ - 1);
                    int32_t * باہرلا = (int32_t *) dst;
                    std::fill(باہرلا + chunk_valid_ناپ, باہرلا + chunk_ناپ, آخری_قدر + 1);
                } وإلا إذا (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t آخری_قدر = *((ثابت int64_t *) src + chunk_valid_ناپ - 1);
                    int64_t * باہرلا = (int64_t *) dst;
                    std::fill(باہرلا + chunk_valid_ناپ, باہرلا + chunk_ناپ, آخری_قدر + 1);
                } وإلا {
                    ارم std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape داخلہ_shape = {1, 1, 1, chunk_ناپ};
        ov::Tensor داخلہ_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلہ_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_ناپ = ggml_type_ناپ(ggml_tensor->type);
        فراغ * داخلہ_انکڑے = (حرف *) ggml_tensor->انکڑے + chunk_اشاریہ * chunk_ناپ * element_ناپ;
        std::حافظہ_نقل(داخلہ_tensor.انکڑے(), داخلہ_انکڑے, chunk_valid_ناپ * element_ناپ);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        إذا (chunk_pad_ناپ > 0) {
            إذا (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t آخری_قدر =
                    *((int32_t *) ggml_tensor->انکڑے + (chunk_اشاریہ * chunk_ناپ + chunk_valid_ناپ - 1));
                int32_t * output_انکڑے = داخلہ_tensor.انکڑے<int32_t>();
                std::fill(output_انکڑے + chunk_valid_ناپ, output_انکڑے + chunk_ناپ, آخری_قدر + 1);
            } وإلا إذا (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t آخری_قدر =
                    *((int64_t *) ggml_tensor->انکڑے + (chunk_اشاریہ * chunk_ناپ + chunk_valid_ناپ - 1));
                int64_t * output_انکڑے = داخلہ_tensor.انکڑے<int64_t>();
                std::fill(output_انکڑے + chunk_valid_ناپ, output_انکڑے + chunk_ناپ, آخری_قدر + 1);
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
            *داخلہ_tensor.انکڑے<int32_t>() = 0;
        } وإلا {
            خودکار * انکڑے_addr = داخلہ_tensor.انکڑے<int32_t>();
            لكل (size_t i = 0; i < output_len; i++) {
                انکڑے_addr[i] = ((int32_t *) ggml_tensor->انکڑے)[i] % chunk_ناپ;
            }
        }
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_scale_صفاں(ggml_tensor, op)) {
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, chunk_ناپ, 1});
        خودکار * dst = داخلہ_tensor.انکڑے<عائم>();
        ثابت خودکار * src = static_cast<ثابت عائم *>(ggml_tensor->انکڑے) + chunk_اشاریہ * chunk_ناپ;
        std::copy(src, src + chunk_valid_ناپ, dst);
        std::fill(dst + chunk_valid_ناپ, dst + chunk_ناپ, 1.0f);
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_اوسط(ggml_tensor, op)) {
        ثابت size_t n_seqs = ggml_tensor->ne[1];
        ثابت size_t src_stride = ggml_tensor->ne[0];
        ثابت size_t copy_len = std::min<size_t>(chunk_valid_ناپ, src_stride - chunk_اشاریہ * chunk_ناپ);
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_ناپ});
        خودکار * dst = داخلہ_tensor.انکڑے<عائم>();
        std::fill(dst, dst + n_seqs * chunk_ناپ, 0.0f);
        ثابت خودکار * src = static_cast<ثابت عائم *>(ggml_tensor->انکڑے) + chunk_اشاریہ * chunk_ناپ;
        لكل (size_t s = 0; s < n_seqs; s++) {
            std::حافظہ_نقل(dst + s * chunk_ناپ, src + s * src_stride, copy_len * حجم(عائم));
        }
        أرجع داخلہ_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_نقاب(ggml_tensor, op)) {
        size_t خانے_گنتی_2 = ggml_tensor->ne[0];
        size_t صفاں = ggml_tensor->ne[1];
        size_t chunk_valid_صفاں = std::min(chunk_ناپ, صفاں - chunk_اشاریہ * chunk_ناپ);
        size_t context_ناپ = ggml_decoder->get_ctx_ناپ();
        إذا (ggml_tensor->type == GGML_TYPE_F16) {
            ثابت خودکار * ggml_انکڑے =
                static_cast<ثابت ggml_fp16_t *>(ggml_tensor->انکڑے) + chunk_اشاریہ * chunk_ناپ * خانے_گنتی_2;
            std::vector<ggml_fp16_t> padded_انکڑے = pad_داخلہ<ggml_fp16_t>(ggml_انکڑے, chunk_valid_صفاں, خانے_گنتی_2, chunk_ناپ,
                                                                          context_ناپ, GGML_FP32_منزل_FP16(-INFINITY));
            سیٹ_zero_diagonal(padded_انکڑے, chunk_ناپ, context_ناپ, GGML_FP32_منزل_FP16(0.0f));
            ov::Tensor داخلہ_tensor(ov::element::f16, ov::Shape{1, 1, chunk_ناپ, context_ناپ});
            std::حافظہ_نقل(داخلہ_tensor.انکڑے(), padded_انکڑے.انکڑے(), padded_انکڑے.size() * حجم(ggml_fp16_t));
            أرجع داخلہ_tensor;
        }

        ثابت خودکار * ggml_انکڑے = static_cast<ثابت عائم *>(ggml_tensor->انکڑے) + chunk_اشاریہ * chunk_ناپ * خانے_گنتی_2;
        std::vector<عائم> padded_انکڑے =
            pad_داخلہ<عائم>(ggml_انکڑے, chunk_valid_صفاں, خانے_گنتی_2, chunk_ناپ, context_ناپ, -INFINITY);
        سیٹ_zero_diagonal(padded_انکڑے, chunk_ناپ, context_ناپ);
        ov::Tensor داخلہ_tensor(ov::element::f32, ov::Shape{1, 1, chunk_ناپ, context_ناپ});
        خودکار * انکڑے_ptr = داخلہ_tensor.انکڑے<عائم>();
        std::copy(padded_انکڑے.begin(), padded_انکڑے.begin() + chunk_ناپ * context_ناپ, انکڑے_ptr);
        أرجع داخلہ_tensor;
    }

    أرجع get_ov_داخلہ_tensor(ggml_decoder, param_name);
}

تعداد ggml_حالت naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               ثابت std::string & device,
                               ثابت ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    إذا (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        أرجع GGML_حالت_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_جندرا(cache.mutex);
    منطقي naive = سچ;
    خودکار model_weights = GgmlOvDecoder::create_بھار_nodes(cgraph, naive);
    خودکار decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    خودکار داخلہ_model = std::make_shared<ov::frontend::ggml::داخلہ_Model>(decoder);
    خودکار model = ov::frontend::ggml::Front_سرا::convert(داخلہ_model, naive);
    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    خودکار remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    إذا (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_منوال.name()] = ov::hint::Execution_منوال::PERFORMANCE;
    } وإلا {
        compile_config[ov::hint::execution_منوال.name()] = ov::hint::Execution_منوال::ACCURACY;
    }
    إذا (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } وإلا {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> داخلہ_ناں;
    std::vector<std::string> output_ناں;
    لكل (ثابت خودکار & param : model->get_parameters()) {
        داخلہ_ناں.push_back(param->get_friendly_name());
    }
    لكل (ثابت خودکار & نتیجہ : model->get_results()) {
        output_ناں.push_back(نتیجہ->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    داخلہ_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_جندرا.unlock();

    لكل (size_t i = 0; i < داخلہ_ناں.size(); i++) {
        ثابت خودکار & param_name = داخلہ_ناں[i];
        خودکار داخلہ_tensor = get_ov_داخلہ_tensor(decoder, param_name);
        infer_request->سیٹ_داخلہ_tensor(i, داخلہ_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    لكل (size_t i = 0; i < output_ناں.size(); i++) {
        خودکار output_tensor = infer_request->get_output_tensor(i);
        ثابت خودکار & model_outputs = decoder->get_model_outputs();
        خودکار model_output_it = model_outputs.find(output_ناں[i]);
        إذا (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_ناں[i], output_tensor, output_tensor.انکڑے());
            }
            استمر;
        }
        خودکار * ggml_tensor = model_output_it->second;
        std::حافظہ_نقل(ggml_tensor->انکڑے, output_tensor.انکڑے(), output_tensor.get_byte_ناپ());
    }
    أرجع GGML_حالت_SUCCESS;
}

تعداد ggml_حالت ov_گراف_compute_dynamic(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    خودکار & core = ov_singleton_core();
    ثابت خودکار & config = ggml_openvino_get_compile_config();
    ثابت خودکار & device = r_ctx->device;
    ساكن خودکار is_static = خطأ;

    ساكن ثابت منطقي cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    خودکار شروع_ویلا = ggml_ویلا_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    ثابت int64_t cache_کلید_شروع_ویلا = ggml_ویلا_us();
    گراف_کلید کلید(cgraph);
    ثابت int64_t cache_کلید_compute_ویلا = ggml_ویلا_us() - cache_کلید_شروع_ویلا;
    int64_t cache_lookup_ویلا = 0;
    منطقي کلید_تکیا = خطأ;
    إذا (!cache_disabled) {
        ثابت int64_t cache_lookup_شروع_ویلا = ggml_ویلا_us();
        std::lock_guard<std::mutex> map_جندرا(r_ctx->ctx_mutex);
        کلید_تکیا = r_ctx->decoder_cache.find(کلید) != r_ctx->decoder_cache.end();
        cache_lookup_ویلا += ggml_ویلا_us() - cache_lookup_شروع_ویلا;
    }

    ثابت int64_t model_ونڈو_check_شروع_ویلا = ggml_ویلا_us();
    منطقي model_is_splitted = کلید_تکیا ? خطأ : is_model_splitted(cgraph);
    ثابت int64_t model_ونڈو_check_ویلا = ggml_ویلا_us() - model_ونڈو_check_شروع_ویلا;

    إذا (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        أرجع GGML_حالت_FAILED;
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
    ثابت int64_t گراف_param_compute_شروع_ویلا = ggml_ویلا_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    ثابت int64_t گراف_param_compute_ویلا = ggml_ویلا_us() - گراف_param_compute_شروع_ویلا;

    ثابت منطقي cache_enabled = !model_is_splitted && !cache_disabled;
    ثابت منطقي stateful_recurrent = r_ctx->stateful && m_params.state_ناپ >= 0;
    ثابت منطقي stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    إذا (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_سرگرم > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        أرجع GGML_حالت_FAILED;
    }
    إذا (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        أرجع GGML_حالت_FAILED;
    }
    منطقي cache_کامیابی = خطأ;

    int64_t decoder_سرا_ویلا;
    int64_t conversion_سرا_ویلا;
    int64_t compile_سرا_ویلا;
    int64_t infer_سرا_ویلا;
    int64_t ov_raw_infer_شروع;

    {
        std::shared_ptr<decoder_runtime_ctx> اندراج;
        ModelParams old_m_params;

        ثابت int64_t cache_lookup_شروع_ویلا = ggml_ویلا_us();
        إذا (cache_enabled) {
            std::lock_guard<std::mutex> map_جندرا(r_ctx->ctx_mutex);
            خودکار it = r_ctx->decoder_cache.find(کلید);
            cache_کامیابی = it != r_ctx->decoder_cache.end();
            إذا (cache_کامیابی) {
                اندراج = it->second;
            } وإلا {
                r_ctx->clear_caches_locked();
                خودکار mutex = std::make_shared<std::mutex>();
                اندراج = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[کلید] = اندراج;
            }
        } وإلا {
            خودکار mutex = std::make_shared<std::mutex>();
            اندراج = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_کامیابی = خطأ;
        }
        cache_lookup_ویلا += ggml_ویلا_us() - cache_lookup_شروع_ویلا;

        std::lock_guard<std::mutex> جندرا(*(اندراج->mutex));
        cache_کامیابی = cache_کامیابی && اندراج->ptr && r_ctx->infer_request_cache.count(کلید) != 0;

        إذا (cache_کامیابی) {
            ggml_decoder = اندراج->ptr;
            old_m_params = ggml_decoder->get_model_params();
            إذا (!ggml_decoder->is_splited_model()) {
                cache_کامیابی = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_داخلہ_ناں;
        std::vector<std::string> ov_output_ناں;

        إذا (cache_کامیابی) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->سیٹ_compute_params(c_params);
            ggml_decoder->سیٹ_model_params(m_params);
            إذا (old_m_params.kv_بفر_بدلیا(m_params) || !ggml_decoder->is_bound_منزل(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->جوڑو_extra_inputs();
            {
                std::lock_guard<std::mutex> map_جندرا(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(کلید);
                ov_داخلہ_ناں = r_ctx->ov_داخلہ_ناں_cache.at(کلید);
                ov_output_ناں = r_ctx->ov_output_ناں_cache.at(کلید);
            }

            إذا (stateful_kv_only) {
                ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_انکڑے = (int32_t *) inp_pos->انکڑے;
                خودکار pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                إذا (pos_انکڑے[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_ناپ = pos_shape[3];
                } وإلا إذا (r_ctx->stateful_kv_ناپ == static_cast<size_t>(pos_انکڑے[0])) {
                    r_ctx->stateful_kv_ناپ += pos_shape[3];
                } وإلا {
                    ثابت size_t pos_begin = static_cast<size_t>(pos_انکڑے[0]);
                    ثابت منطقي refill = pos_begin > r_ctx->stateful_kv_ناپ;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    إذا (refill && !ggml_decoder->get_model_params().swa_layers.خالی_تھاواں()) {
                        ثابت صحيح n_swa = ggml_decoder->get_compute_params().swa_کھڑکی;
                        إذا (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_ناپ, n_swa);
                            أرجع GGML_حالت_FAILED;
                        }
                    }

                    ثابت منطقي relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    خودکار حالتاں = infer_request->query_state();
                    لكل (خودکار state : حالتاں) {
                        خودکار state_tensor = state.get_state();
                        خودکار state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        إذا (خودکار it = r_ctx->kv_state_داخلہ_name_map.find(state.get_name());
                            it != r_ctx->kv_state_داخلہ_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        صحيح n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        إذا (خودکار layer = کڈھو_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        ثابت منطقي relayout_this_state = relayout_enabled;
                        ثابت size_t seq_axis = relayout_this_state ? 2 : 1;
                        ثابت size_t سر_axis = seq_axis == 2 ? 1 : 2;

                        إذا (refill) {
                            إذا (state_name.خالی_تھاواں()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                أرجع GGML_حالت_FAILED;
                            }
                            خودکار kv_tensor = get_ov_داخلہ_tensor(ggml_decoder, state_name);
                            إذا (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_صفاں_منزل_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } وإلا {
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
                        إذا (state_tensor_shape[seq_axis] < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: state '%s' holds %zu tokens on axis "
                                "%zu, cannot resume at position %zu\n",
                                state.get_name().c_str(), state_tensor_shape[seq_axis], seq_axis, pos_begin);
                            أرجع GGML_حالت_FAILED;
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end(state_tensor_shape.begin(), state_tensor_shape.end());
                        end[seq_axis] = pos_begin;
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.سیٹ_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_ناپ = pos_begin + pos_shape[3];
                }
            }

            decoder_سرا_ویلا = ggml_ویلا_us();
            conversion_سرا_ویلا = decoder_سرا_ویلا;
            compile_سرا_ویلا = decoder_سرا_ویلا;
        } وإلا {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            خودکار shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_جندرا(shared_cache->mutex);
            خودکار بھار_ناں = get_بھار_ناں(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, بھار_ناں,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ثابت std::string model_cache_dir = ggml_openvino_model_cache_dir();
            ثابت std::string exact_کلید = cache_enabled ? compiled_گراف_کلید(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_رستہ, manifest_رستہ;
            إذا (!model_cache_dir.خالی_تھاواں() && !model_is_splitted) {
                ثابت uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/سچ, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_گراف_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/سچ, m_params.rope_params, 16, extra_cfg,
                                                    compiled_گراف_کلید(cgraph, *ggml_decoder, device, 0, سچ));
                blob_رستہ = ggml_openvino_model_cache_blob_رستہ(model_cache_dir, model_fp);
                manifest_رستہ = ggml_openvino_model_cache_manifest_رستہ(model_cache_dir, model_fp);
            }
            std::string shared_کلید =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_کلید) : "";
            ov::CompiledModel shared_model;
            منطقي imported = خطأ;
            خودکار shared_it = shared_cache->graphs.find(shared_کلید);
            إذا (!shared_کلید.خالی_تھاواں() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_گراف(shared_it->second.decode, shared_it->second.داخلہ_ناں,
                                                               shared_it->second.output_ناں, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_داخلہ_ناں = shared_it->second.داخلہ_ناں;
                ov_output_ناں = shared_it->second.output_ناں;
                imported = سچ;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } وإلا إذا (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_کلید = exact_کلید;
                shared_it = shared_cache->graphs.find(shared_کلید);
                إذا (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_داخلہ_ناں = shared_it->second.داخلہ_ناں;
                    ov_output_ناں = shared_it->second.output_ناں;
                    imported = سچ;
                }
                model_fp = exact_fp;
                blob_رستہ = ggml_openvino_model_cache_blob_رستہ(model_cache_dir, model_fp);
                manifest_رستہ = ggml_openvino_model_cache_manifest_رستہ(model_cache_dir, model_fp);
            }
            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_جندرا(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(کلید);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            إذا (!model_cache_dir.خالی_تھاواں()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_منوال.name()] = ov::Cache_منوال::OPTIMIZE_رفتار;
            }
            إذا (!imported && !model_cache_dir.خالی_تھاواں() && !model_is_splitted) {
                ثابت uint64_t preferred_fp = model_fp;
                منطقي dynamic_incompatible = preferred_fp == exact_fp;
                لكل (صحيح candidate = 0; candidate < 2; ++candidate) {
                    إذا (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        اكسر;
                    }
                    ثابت uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    ثابت std::string candidate_blob =
                        ggml_openvino_model_cache_blob_رستہ(model_cache_dir, candidate_fp);
                    ثابت std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_رستہ(model_cache_dir, candidate_fp);
                    std::ifstream blob_وچ(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    إذا (!blob_وچ.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        استمر;
                    }
                    ثابت int64_t import_شروع = ggml_ویلا_us();
                    حاول {
                        ov::CompiledModel cm;
                        خودکار remote_context = ggml_openvino_get_remote_context();
                        إذا (remote_context.has_value()) {
                            cm = core.import_model(blob_وچ, remote_context.value(), mc_config);
                        } وإلا {
                            cm = core.import_model(blob_وچ, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        إذا (!saved_inputs.خالی_تھاواں() || !saved_outputs.خالی_تھاواں()) {
                            ov_داخلہ_ناں = std::move(saved_inputs);
                            ov_output_ناں = std::move(saved_outputs);
                        } وإلا {
                            لكل (ثابت خودکار & p : cm.inputs()) {
                                ov_داخلہ_ناں.push_back(p.get_گرہ()->get_friendly_name());
                            }
                            لكل (ثابت خودکار & o : cm.outputs()) {
                                ov_output_ناں.push_back(o.get_گرہ()->get_friendly_name());
                            }
                        }
                        إذا (!compiled_model_matches_گراف(cm, ov_داخلہ_ناں, ov_output_ناں, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_داخلہ_ناں.clear();
                            ov_output_ناں.clear();
                            استمر;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        اندراج->ptr = ggml_decoder;
                        imported = سچ;
                        model_fp = candidate_fp;
                        blob_رستہ = candidate_blob;
                        manifest_رستہ = candidate_manifest;
                        إذا (candidate_fp == exact_fp) {
                            shared_کلید = exact_کلید;
                            shared_it = shared_cache->graphs.find(shared_کلید);
                        }
                        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_ویلا_us() - import_شروع) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } امسك (ثابت std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = خطأ;
                        ov_داخلہ_ناں.clear();
                        ov_output_ناں.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                إذا (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_کلید = exact_کلید;
                    shared_it = shared_cache->graphs.find(shared_کلید);
                    blob_رستہ = ggml_openvino_model_cache_blob_رستہ(model_cache_dir, model_fp);
                    manifest_رستہ = ggml_openvino_model_cache_manifest_رستہ(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            إذا (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_رستہ.c_str());
                أرجع GGML_حالت_FAILED;
            }
            إذا (!imported && ggml_openvino_بھار_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                أرجع GGML_حالت_FAILED;
            }
            إذا (imported) {
                decoder_سرا_ویلا = conversion_سرا_ویلا = compile_سرا_ویلا = ggml_ویلا_us();
            } وإلا {
                خودکار model_weights = GgmlOvDecoder::create_بھار_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_سرا_ویلا = ggml_ویلا_us();

                خودکار داخلہ_model = std::make_shared<ov::frontend::ggml::داخلہ_Model>(ggml_decoder);
                model = ov::frontend::ggml::Front_سرا::convert(داخلہ_model);
                ggml_decoder->clear_model_weights();
                conversion_سرا_ویلا = ggml_ویلا_us();

                إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    حرف timestamped_filename[64];
                    خودکار timestamp = (لما لما) ggml_ویلا_us();
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                ثابت ov::AnyMap & compile_config = model_cache_dir.خالی_تھاواں() ? config : mc_config;
                ov::CompiledModel compiled_model;
                خودکار remote_context = ggml_openvino_get_remote_context();
                إذا (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } وإلا {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_سرا_ویلا = ggml_ویلا_us();

                لكل (ثابت خودکار & ov_param : model->get_parameters()) {
                    ov_داخلہ_ناں.push_back(ov_param->get_friendly_name());
                }
                لكل (ثابت خودکار & ov_output : model->get_results()) {
                    ov_output_ناں.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                إذا (!model_cache_dir.خالی_تھاواں() && !model_is_splitted && model_fp != 0) {
                    حاول {
                        ثابت std::string blob_tmp = ggml_openvino_model_cache_عارضی_رستہ(blob_رستہ);
                        ثابت std::string manifest_tmp = ggml_openvino_model_cache_عارضی_رستہ(manifest_رستہ);
                        إذا (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_داخلہ_ناں,
                                                                     ov_output_ناں)) {
                            std::ofstream blob_باہرلا(blob_tmp, std::ios::binary | std::ios::trunc);
                            إذا (blob_باہرلا.is_open()) {
                                compiled_model.export_model(blob_باہرلا);
                                blob_باہرلا.close();
                                إذا (blob_باہرلا.good()) {
                                    إذا (std::rename(blob_tmp.c_str(), blob_رستہ.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_رستہ.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_رستہ.c_str());
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
                اندراج->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            اندراج->ptr = ggml_decoder;
            إذا (!shared_کلید.خالی_تھاواں() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_کلید,
                                             ov_compiled_گراف{shared_model, {}, ov_داخلہ_ناں, ov_output_ناں});
            }
            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_جندرا(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[کلید] = infer_request;
                r_ctx->ov_داخلہ_ناں_cache[کلید] = ov_داخلہ_ناں;
                r_ctx->ov_output_ناں_cache[کلید] = ov_output_ناں;
            }

            إذا (stateful_kv_only && cache_enabled) {
                ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
                خودکار pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                ثابت int32_t pos_begin = ((int32_t *) inp_pos->انکڑے)[0];
                إذا (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    أرجع GGML_حالت_FAILED;
                }
                r_ctx->stateful_kv_ناپ = pos_shape[3];
                ثابت خودکار kv_param_res_ناں = ggml_decoder->get_kv_param_res_ناں();
                لكل (ثابت خودکار & pair : kv_param_res_ناں) {
                    r_ctx->kv_state_داخلہ_name_map[pair.پہلا + pair.second] = pair.پہلا;
                }
            }
        }

        إذا (stateful_recurrent) {
            ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
            ثابت int32_t pos_begin = static_cast<ثابت int32_t *>(inp_pos->انکڑے)[0];
            إذا (pos_begin == 0) {
                infer_request->reset_state();
            } وإلا إذا (!cache_کامیابی || old_m_params.kv_بفر_بدلیا(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_ناپ) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                أرجع GGML_حالت_FAILED;
            }
        } وإلا {
            reset_single_تھاں_recurrent_cache(cgraph, m_params, c_params);
        }

        لكل (size_t i = 0; i < ov_داخلہ_ناں.size(); i++) {
            ثابت خودکار & param_name = ov_داخلہ_ناں[i];
            خودکار داخلہ_tensor = get_ov_داخلہ_tensor(ggml_decoder, param_name);
            infer_request->سیٹ_داخلہ_tensor(i, داخلہ_tensor);

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_داخلہ_tensor_info(param_name, داخلہ_tensor);
            }
        }

        لكل (size_t i = 0; i < ov_output_ناں.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            ثابت خودکار & model_outputs = ggml_decoder->get_model_outputs();
            خودکار model_output_it = model_outputs.find(ov_output_ناں[i]);
            إذا (model_output_it == model_outputs.end()) {
                استمر;
            }
            خودکار * ggml_tensor = model_output_it->second;
            إذا (ggml_nbytes(ggml_tensor) == 0) {
                استمر;
            }
            خودکار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->سیٹ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_شروع = ggml_ویلا_us();
        infer_request->infer();
        infer_سرا_ویلا = ggml_ویلا_us();
        إذا (stateful_recurrent) {
            ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_ناپ =
                static_cast<ثابت int32_t *>(inp_pos->انکڑے)[0] + get_inp_pos_n_ٹوکن(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            لكل (size_t i = 0; i < ov_output_ناں.size(); i++) {
                ثابت خودکار output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_ناں[i], output_tensor, output_tensor.انکڑے());
            }
        }

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_سرا_ویلا - شروع_ویلا) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_کلید_compute_ویلا / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_ونڈو_check_ویلا / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", گراف_param_compute_ویلا / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_ویلا / 1000.0);
            إذا (!cache_کامیابی) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_سرا_ویلا - decoder_سرا_ویلا) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_سرا_ویلا - conversion_سرا_ویلا) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_سرا_ویلا - compile_سرا_ویلا) / 1000.0);
            GGML_LOG_INFO("    - OV raw infer time: %.3f ms \n", (infer_سرا_ویلا - ov_raw_infer_شروع) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. Release only while holding the compilation mutex so another context cannot
    // be reading host weights during conversion/compilation. Pin the shared compiled
    // models across backend teardown; a later context can create its own request without
    // reading the dropped pages. A new, uncached graph still fails fast above.
    إذا (cache_کامیابی && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_جندرا(r_ctx->compiled_cache->mutex);
        إذا (!ggml_openvino_بھار_buffers_released()) {
            ggml_openvino_release_بھار_buffers();
        }
    }

    أرجع GGML_حالت_SUCCESS;
}

ov::AnyMap without_npuw(ثابت ov::AnyMap & config) {
    ov::AnyMap باہرلا;
    لكل (ثابت خودکار & kv : config) {
        إذا (kv.پہلا.rfind("NPUW", 0) == 0 || kv.پہلا == "NPU_USE_NPUW") {
            استمر;
        }
        باہرلا.insert(kv);
    }
    أرجع باہرلا;
}

تعداد ggml_حالت ov_گراف_compute_static(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    خودکار & core = ov_singleton_core();

    خودکار get_prefill_chunk_ناپ = [] {
        ساكن ثابت صحيح chunk_ناپ = []() {
            صحيح env_prefill_chunk_ناپ = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            أرجع env_prefill_chunk_ناپ > 0 ? env_prefill_chunk_ناپ : 256;
        }();
        أرجع chunk_ناپ;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    ساكن std::string device = ggml_openvino_get_device_name();
    ساكن خودکار is_static = سچ;
    ساكن خودکار stateful = خطأ;

    خودکار prefill_chunk_ناپ = get_prefill_chunk_ناپ();
    ثابت خودکار & config = ggml_openvino_get_compile_config();

    إذا (is_naive(cgraph)) {
        أرجع naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    خودکار شروع_ویلا = ggml_ویلا_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
    ثابت منطقي no_kv_cache = m_params.is_cacheless_attn;
    ثابت خودکار is_prefill = no_kv_cache ? سچ : get_is_prefill(cgraph, inp_pos);
    ثابت ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    إذا (m_params.n_heads_kv == -1) {
        prefill_chunk_ناپ = inp_pos->ne[0];
    }
    گراف_کلید کلید(cgraph);
    ساكن ثابت منطقي cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    منطقي cache_کامیابی = خطأ;

    int64_t decoder_سرا_ویلا;
    int64_t conversion_سرا_ویلا;
    int64_t compile_سرا_ویلا;
    int64_t infer_سرا_ویلا;
    int64_t ov_raw_infer_شروع;
    int64_t ov_raw_infer_کل = 0;

    std::shared_ptr<decoder_runtime_ctx> اندراج;
    ModelParams old_m_params;

    إذا (cache_enabled) {
        std::lock_guard<std::mutex> map_جندرا(r_ctx->ctx_mutex);
        خودکار it = r_ctx->decoder_cache.find(کلید);
        cache_کامیابی = it != r_ctx->decoder_cache.end();
        إذا (cache_کامیابی) {
            اندراج = it->second;
        } وإلا {
            r_ctx->clear_caches_locked();
            خودکار mutex = std::make_shared<std::mutex>();
            اندراج = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[کلید] = اندراج;
        }
    } وإلا {
        خودکار mutex = std::make_shared<std::mutex>();
        اندراج = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_کامیابی = خطأ;
    }

    std::lock_guard<std::mutex> جندرا(*(اندراج->mutex));
    cache_کامیابی = cache_کامیابی && اندراج->ptr && r_ctx->infer_request_cache.count(کلید) != 0 &&
                r_ctx->infer_request_cache_prefill.count(کلید) != 0;

    إذا (cache_کامیابی) {
        ggml_decoder = اندراج->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_کامیابی = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_داخلہ_ناں_local;
    std::vector<std::string> ov_output_ناں_local;

    إذا (cache_کامیابی) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->سیٹ_model_params(m_params);
        ggml_decoder->سیٹ_compute_params(c_params);
        إذا (old_m_params.kv_بفر_بدلیا(m_params) || !ggml_decoder->is_bound_منزل(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->جوڑو_extra_inputs();
        {
            std::lock_guard<std::mutex> map_جندرا(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(کلید) : r_ctx->infer_request_cache.at(کلید);
            ov_داخلہ_ناں_local = r_ctx->ov_داخلہ_ناں_cache.at(کلید);
            ov_output_ناں_local = r_ctx->ov_output_ناں_cache.at(کلید);
        }

        decoder_سرا_ویلا = ggml_ویلا_us();
        conversion_سرا_ویلا = decoder_سرا_ویلا;
        compile_سرا_ویلا = decoder_سرا_ویلا;
    } وإلا {
        إذا (cache_enabled) {
            std::lock_guard<std::mutex> map_جندرا(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(کلید);
            r_ctx->infer_request_cache_prefill.erase(کلید);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        خودکار shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_جندرا(shared_cache->mutex);
        خودکار بھار_ناں = get_بھار_ناں(cgraph);
        خودکار local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, بھار_ناں, is_static,
                                                             stateful, خطأ, is_prefill, prefill_chunk_ناپ);
        ثابت std::string shared_کلید =
            cache_enabled ? compiled_گراف_کلید(cgraph, *local_decoder, device, prefill_chunk_ناپ) : "";
        خودکار shared_it = shared_cache->graphs.find(shared_کلید);
        إذا (!shared_کلید.خالی_تھاواں() && shared_it != shared_cache->graphs.end()) {
            خودکار & compiled = shared_it->second;
            خودکار prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            خودکار decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            اندراج->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_داخلہ_ناں_local = compiled.داخلہ_ناں;
            ov_output_ناں_local = compiled.output_ناں;
            r_ctx->infer_request_cache_prefill[کلید] = prefill_request;
            r_ctx->infer_request_cache[کلید] = decode_request;
            r_ctx->ov_داخلہ_ناں_cache[کلید] = ov_داخلہ_ناں_local;
            r_ctx->ov_output_ناں_cache[کلید] = ov_output_ناں_local;
            decoder_سرا_ویلا = conversion_سرا_ویلا = compile_سرا_ویلا = ggml_ویلا_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } وإلا {
            std::shared_ptr<ov::Model> model;
            خودکار model_weights = GgmlOvDecoder::create_بھار_nodes(cgraph);

            خودکار ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, خطأ, سچ, prefill_chunk_ناپ);
            خودکار ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, خطأ, خطأ, prefill_chunk_ناپ);
            decoder_سرا_ویلا = ggml_ویلا_us();

            ثابت منطقي dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            ثابت خودکار dump_ir_timestamp = static_cast<لما لما>(ggml_ویلا_us());

            خودکار build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          ثابت std::shared_ptr<GgmlOvDecoder> & decoder, ثابت حرف * ٹیگ,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_سرا_ویلا, int64_t & local_compile_سرا_ویلا) {
                خودکار داخلہ_model = std::make_shared<ov::frontend::ggml::داخلہ_Model>(decoder);
                model = ov::frontend::ggml::Front_سرا::convert(داخلہ_model);
                decoder->clear_model_weights();
                local_conversion_سرا_ویلا = ggml_ویلا_us();

                إذا (dump_ir) {
                    حرف timestamped_filename[64];
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%s_%lld.xml", ٹیگ,
                             dump_ir_timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                compiled_model = core.compile_model(model, device, compile_config);
                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                local_compile_سرا_ویلا = ggml_ویلا_us();
            };
            std::shared_ptr<ov::Model> model_prefill;
            std::shared_ptr<ov::Model> model_decode;
            ov::CompiledModel compiled_model_prefill;
            ov::CompiledModel compiled_model_decode;
            std::shared_ptr<ov::InferRequest> infer_request_prefill;
            std::shared_ptr<ov::InferRequest> infer_request_decode;
            int64_t prefill_conversion_سرا_ویلا;
            int64_t decode_conversion_سرا_ویلا;
            int64_t prefill_compile_سرا_ویلا;
            int64_t decode_compile_سرا_ویلا;
            build_static_model(ggml_decoder_prefill, "prefill", model_prefill, compiled_model_prefill,
                               infer_request_prefill, prefill_conversion_سرا_ویلا, prefill_compile_سرا_ویلا);
            إذا (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_سرا_ویلا = prefill_conversion_سرا_ویلا;
                decode_compile_سرا_ویلا = prefill_compile_سرا_ویلا;
            } وإلا {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_سرا_ویلا, decode_compile_سرا_ویلا);
            }
            conversion_سرا_ویلا = std::max(prefill_conversion_سرا_ویلا, decode_conversion_سرا_ویلا);
            compile_سرا_ویلا = std::max(prefill_compile_سرا_ویلا, decode_compile_سرا_ویلا);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            اندراج->ptr = ggml_decoder;

            لكل (ثابت خودکار & ov_param : model->get_parameters()) {
                ov_داخلہ_ناں_local.push_back(ov_param->get_friendly_name());
            }
            لكل (ثابت خودکار & ov_output : model->get_results()) {
                ov_output_ناں_local.push_back(ov_output->get_friendly_name());
            }

            إذا (!shared_کلید.خالی_تھاواں()) {
                shared_cache->graphs.emplace(
                    shared_کلید, ov_compiled_گراف{compiled_model_decode, compiled_model_prefill, ov_داخلہ_ناں_local,
                                                  ov_output_ناں_local});
            }

            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_جندرا(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[کلید] = infer_request_prefill;
                r_ctx->infer_request_cache[کلید] = infer_request_decode;
                r_ctx->ov_داخلہ_ناں_cache[کلید] = ov_داخلہ_ناں_local;
                r_ctx->ov_output_ناں_cache[کلید] = ov_output_ناں_local;
            }
        }
    }

    reset_single_تھاں_recurrent_cache(cgraph, m_params, c_params);

    إذا (is_prefill) {
        خودکار inp_len = get_inp_pos_n_ٹوکن(cgraph, inp_pos);
        لكل (صحيح chunk_اشاریہ = 0; chunk_اشاریہ * prefill_chunk_ناپ < inp_len; chunk_اشاریہ++) {
            لكل (size_t i = 0; i < ov_داخلہ_ناں_local.size(); i++) {
                ثابت خودکار & param_name = ov_داخلہ_ناں_local[i];
                خودکار داخلہ_tensor = get_ov_داخلہ_tensor_static_prefill(ggml_decoder, param_name, chunk_اشاریہ);
                infer_request->سیٹ_داخلہ_tensor(i, داخلہ_tensor);

                إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    ثابت خودکار داخلہ_tensor = infer_request->get_داخلہ_tensor(i);
                    print_داخلہ_tensor_info(param_name, داخلہ_tensor);
                }
            }

            لكل (size_t i = 0; i < ov_output_ناں_local.size(); i++) {
                ثابت خودکار & model_outputs = ggml_decoder->get_model_outputs();
                خودکار model_output_it = model_outputs.find(ov_output_ناں_local[i]);
                إذا (model_output_it == model_outputs.end()) {
                    استمر;
                }
                خودکار * ggml_tensor = model_output_it->second;
                إذا (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    استمر;
                }
                خودکار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->سیٹ_output_tensor(i, output_tensor);
            }

            ov_raw_infer_شروع = ggml_ویلا_us();
            infer_request->infer();
            ov_raw_infer_کل += ggml_ویلا_us() - ov_raw_infer_شروع;

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                لكل (size_t i = 0; i < ov_output_ناں_local.size(); i++) {
                    ثابت خودکار output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_ناں_local[i], output_tensor, output_tensor.انکڑے());
                }
            }
        }
        infer_سرا_ویلا = ggml_ویلا_us();
    } وإلا {
        لكل (size_t i = 0; i < ov_داخلہ_ناں_local.size(); i++) {
            ثابت خودکار & param_name = ov_داخلہ_ناں_local[i];
            خودکار داخلہ_tensor = get_ov_داخلہ_tensor_static_decode(ggml_decoder, param_name);
            infer_request->سیٹ_داخلہ_tensor(i, داخلہ_tensor);

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                ثابت خودکار داخلہ_tensor = infer_request->get_داخلہ_tensor(i);
                print_داخلہ_tensor_info(param_name, داخلہ_tensor);
            }
        }

        لكل (size_t i = 0; i < ov_output_ناں_local.size(); i++) {
            ثابت خودکار & model_outputs = ggml_decoder->get_model_outputs();
            خودکار model_output_it = model_outputs.find(ov_output_ناں_local[i]);
            إذا (model_output_it == model_outputs.end()) {
                استمر;
            }
            خودکار * ggml_tensor = model_output_it->second;
            إذا (ggml_nbytes(ggml_tensor) == 0) {
                استمر;
            }
            خودکار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->سیٹ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_شروع = ggml_ویلا_us();
        infer_request->infer();
        infer_سرا_ویلا = ggml_ویلا_us();
        ov_raw_infer_کل = infer_سرا_ویلا - ov_raw_infer_شروع;

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            لكل (size_t i = 0; i < ov_output_ناں_local.size(); i++) {
                ثابت خودکار output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_ناں_local[i], output_tensor, output_tensor.انکڑے());
            }
        }
    }

    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_سرا_ویلا - شروع_ویلا) / 1000.0);
        إذا (!cache_کامیابی) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_سرا_ویلا - decoder_سرا_ویلا) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_سرا_ویلا - conversion_سرا_ویلا) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_سرا_ویلا - compile_سرا_ویلا) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_کل / 1000.0);
    }

    أرجع GGML_حالت_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
ساكن ggml_cgraph * get_compute_گراف(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    خودکار is_skipped = [](ثابت ggml_tensor * گرہ) {
        أرجع گرہ->op != GGML_OP_NONE && !(گرہ->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    إذا (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        أرجع cgraph;
    }
    خودکار & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](ثابت ggml_tensor * گرہ) { أرجع !is_skipped(گرہ); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.انکڑے();
    compute.graph.n_nodes = (صحيح) compute.nodes.size();
    أرجع &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
تعداد ggml_حالت ov_گراف_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    حاول {
        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        ثابت خودکار is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != خالی_سوچک);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_جندرا(r_ctx->execution_mutex);
        cgraph = get_compute_گراف(cgraph, *r_ctx);

        أرجع is_static ? ov_گراف_compute_static(cgraph, r_ctx) : ov_گراف_compute_dynamic(cgraph, r_ctx);
    } امسك (ثابت ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        أرجع GGML_حالت_FAILED;
    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        أرجع GGML_حالت_FAILED;
    } امسك (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        أرجع GGML_حالت_FAILED;
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
        صحيح use_گنتی = cgraph->use_counts[ggml_hash_لبھو(&cgraph->ویکھیا_hash_سیٹ, گرہ)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        إذا ((cgraph->n_nodes <= 1 && use_گنتی == 0) ||
            (cgraph->n_nodes <= 1 && گرہ->op == GGML_OP_VIEW && use_گنتی == 1 && گرہ->src[0] != خالی_سوچک &&
             گرہ->src[0]->op == GGML_OP_NONE)) {
            أرجع خطأ;
        }
        إذا (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_ترتیب_بدلو)) {
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
            أرجع سچ;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    إذا (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_بھار_ناں(cgraph);
    } وإلا {
        لكل (ثابت خودکار & kv : GgmlOvDecoder::create_بھار_nodes(cgraph, سچ)) {
            model_weights.insert(kv.پہلا);
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
            إذا (src != خالی_سوچک && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.خالی_تھاواں() == خطأ &&
                model_leafs.find(src) == model_leafs.end()) {
                إذا (GgmlOvDecoder::is_inp_tok(src, گرہ)) {
                    أرجع خطأ;
                }
                أرجع سچ;
            }
        }
    }
    أرجع خطأ;
}

منطقي is_naive(ggml_cgraph * cgraph) {
    constexpr صحيح naive_گراف_ناپ_حد = 20;
    صحيح count = 0;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        إذا (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    أرجع count < naive_گراف_ناپ_حد;
}

size_t checksum(ثابت فراغ * انکڑے, size_t size) {
    ثابت uint8_t * بائٹاں = static_cast<ثابت uint8_t *>(انکڑے);
    size_t جوڑ = 0;
    لكل (size_t i = 0; i < size; ++i) {
        جوڑ += (uint8_t) i;
        جوڑ += بائٹاں[i];
    }
    أرجع جوڑ;
}

منطقي save_ggml_tensor_انکڑے_منزل_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_رستہ) {
    إذا (tensor == خالی_سوچک || tensor->انکڑے == خالی_سوچک) {
        أرجع خطأ;
    }

    std::ofstream باہرلا(file_رستہ);
    إذا (!باہرلا.is_open()) {
        أرجع خطأ;
    }

    ثابت size_t n = ggml_nelements(tensor);
    باہرلا << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    بدل (tensor->type) {
    حالة GGML_TYPE_F32: {
        ثابت خودکار * انکڑے = static_cast<ثابت عائم *>(tensor->انکڑے);
        لكل (size_t i = 0; i < n; ++i) {
            باہرلا << انکڑے[i] << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_F16: {
        ثابت خودکار * انکڑے = static_cast<ثابت ggml_fp16_t *>(tensor->انکڑے);
        لكل (size_t i = 0; i < n; ++i) {
            باہرلا << ggml_fp16_منزل_fp32(انکڑے[i]) << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_BF16: {
        ثابت خودکار * انکڑے = static_cast<ثابت ggml_bf16_t *>(tensor->انکڑے);
        لكل (size_t i = 0; i < n; ++i) {
            باہرلا << ggml_bf16_منزل_fp32(انکڑے[i]) << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_I32: {
        ثابت خودکار * انکڑے = static_cast<ثابت int32_t *>(tensor->انکڑے);
        لكل (size_t i = 0; i < n; ++i) {
            باہرلا << انکڑے[i] << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_I64: {
        ثابت خودکار * انکڑے = static_cast<ثابت int64_t *>(tensor->انکڑے);
        لكل (size_t i = 0; i < n; ++i) {
            باہرلا << انکڑے[i] << '\n';
        }
        اكسر;
    }
    افتراضي:
        باہرلا << "unsupported tensor type for text dump" << '\n';
        أرجع خطأ;
    }

    أرجع سچ;
}

فراغ print_داخلہ_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor) {
    std::باہر << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.انکڑے()
              << '\n';
    بدل (tensor.get_element_type()) {
    حالة ov::element::f32: {
        إذا (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::باہر << *(tensor.انکڑے<عائم>()) << '\n';
        } وإلا {
            size_t صفاں = tensor.get_shape()[2];
            size_t خانے_گنتی_2 = tensor.get_shape()[3];
            ثابت عائم * انکڑے = tensor.انکڑے<عائم>();
            لكل (size_t i = 0; i < صفاں; ++i) {
                لكل (size_t j = 0; j < خانے_گنتی_2; ++j) {
                    عائم val = انکڑے[i * خانے_گنتی_2 + j];
                    إذا (std::isinf(val) && val < 0) {
                        std::باہر << std::setw(5) << "-inf";
                    } وإلا {
                        std::باہر << std::setw(5) << val;
                    }
                }
                std::باہر << '\n';
            }
        }

        اكسر;
    }
    حالة ov::element::f16:
        std::باہر << *(tensor.انکڑے<ov::float16>()) << '\n';
        اكسر;
    حالة ov::element::i32:
        لكل (size_t i = 0; i < tensor.get_ناپ(); ++i) {
            std::باہر << tensor.انکڑے<int32_t>()[i] << ' ';
        }
        std::باہر << '\n';
        اكسر;
    حالة ov::element::i64:
        لكل (size_t i = 0; i < tensor.get_ناپ(); ++i) {
            std::باہر << tensor.انکڑے<int64_t>()[i] << ' ';
        }
        std::باہر << '\n';
        اكسر;
    افتراضي:
        اكسر;
    }
}

فراغ print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت فراغ * output_dst) {
    std::باہر << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    خودکار print_float_stats = [](ثابت std::string & type_name, size_t size, خودکار get_قدر) {
        إذا (size == 0) {
            أرجع;
        }

        عائم پہلا = get_قدر(0);
        عائم min = پہلا;
        عائم max = پہلا;
        مزدوج جوڑ = پہلا;

        لكل (size_t i = 1; i < size; ++i) {
            عائم v = get_قدر(i);
            min = std::min(v, min);
            max = std::max(v, max);
            جوڑ += v;
        }
        مزدوج اوسط_2 = جوڑ / size;

        std::باہر << std::سجے << std::setw(6) << type_name << std::سجے << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::باہر << std::سجے << std::setw(6) << "" << std::سجے << std::setw(12) << پہلا << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << اوسط_2 << '\n';
    };

    بدل (tensor.get_element_type()) {
    حالة ov::element::f32: {
        ثابت عائم * انکڑے = tensor.انکڑے<عائم>();
        size_t size = tensor.get_ناپ();
        print_float_stats("[f32]", size, [انکڑے](size_t i) { أرجع انکڑے[i]; });
        اكسر;
    }
    حالة ov::element::f16: {
        ثابت ov::float16 * انکڑے = tensor.انکڑے<ov::float16>();
        size_t size = tensor.get_ناپ();
        print_float_stats("[f16]", size, [انکڑے](size_t i) { أرجع static_cast<عائم>(انکڑے[i]); });
        اكسر;
    }
    افتراضي:
        اكسر;
    }
}

ثابت ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        خودکار * op = cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; ++j) {
            خودکار * src = op->src[j];
            إذا (src == خالی_سوچک) {
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

int64_t get_inp_pos_n_ٹوکن(ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    صحيح n_planes = 1;
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        خودکار * op = cgraph->nodes[i];
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
    أرجع get_inp_pos_n_ٹوکن(cgraph, inp_pos) > 1;
}
