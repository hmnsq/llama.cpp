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
    ساكن ثابت منطقي kv_ٽڪرو_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    إذا (kv_ٽڪرو_disabled) {
        أرجع std::nullopt;
    }
    إذا (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        أرجع std::nullopt;
    }
    إذا (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != خالي_اشارو) {
        أرجع std::nullopt;
    }
    ثابت خودڪار * op = ggml_decoder->get_tensor_استعمال_ٿيل_op(ggml_tensor);
    إذا (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        أرجع std::nullopt;
    }

    ثابت خودڪار & compute_params = ggml_decoder->get_compute_params();
    إذا (compute_params.n_seq_سرگرم != 1 || compute_params.seq_سرگرم_شروعات != 0) {
        أرجع std::nullopt;
    }

    صحيح layer;
    إذا (خودڪار layer_opt = ڪڍو_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } وإلا {
        أرجع std::nullopt;
    }

    ثابت منطقي is_swa = ggml_decoder->is_swa_layer(layer);
    إذا (is_swa) {
        أرجع std::nullopt;
    }
    ثابت صحيح ctx_في_seq = ggml_decoder->get_ctx_في_seq();
    ثابت صحيح n_kv = compute_params.attention_سائيز;
    إذا (ctx_في_seq <= 0 || n_kv <= 0 || n_kv >= ctx_في_seq) {
        أرجع std::nullopt;
    }

    ov::Shape ڀريل_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    إذا (ڀريل_shape.size() != 4 || ڀريل_shape[0] != 1 || ڀريل_shape[1] != 1 ||
        static_cast<صحيح>(ڀريل_shape[2]) != ctx_في_seq) {
        أرجع std::nullopt;
    }

    ov::Shape sliced_shape = ڀريل_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    إذا (ggml_openvino_بفر_is_remote(ggml_tensor)) {
        خودڪار remote_context = ggml_openvino_get_remote_context();
        خودڪار gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        أرجع gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->معلومات);
    }

    أرجع ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->معلومات);
}

ساكن uint64_t ggml_openvino_model_cache_extra_cfg(منطقي stateful, منطقي recurrent) {
    ثابت حرف * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    ثابت منطقي manual_gqa_enabled = manual_gqa_env != خالي_اشارو ?
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

std::map<std::string, std::shared_ptr<ov::Node>> get_وزن_نالا(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> نالا;
    لكل (ثابت خودڪار & name : GgmlOvDecoder::collect_وزن_نالا(cgraph)) {
        نالا[name] = خالي_اشارو;
    }
    أرجع نالا;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
ساكن std::string compiled_گراف_ڪنجي(ثابت ggml_cgraph * graph,
                                      ثابت GgmlOvDecoder & decoder,
                                      ثابت std::string & device,
                                      صحيح prefill_chunk_سائيز = 0,
                                      منطقي disk_cache = خطأ) {
    std::string ڪنجي;
    خودڪار append = [&ڪنجي](ثابت خودڪار & value) {
        ڪنجي.append(reinterpret_cast<ثابت حرف *>(&value), حجم(value));
    };
    خودڪار append_string = [&](ثابت std::string & value) {
        append(value.size());
        ڪنجي.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_سائيز);
    منطقي has_وزن_بفر_id = خطأ;
    std::unordered_map<ثابت ggml_tensor *, size_t> ids;
    std::function<فراغ(ثابت ggml_tensor *)> گهمو = [&](ثابت ggml_tensor * tensor) {
        إذا (!tensor) {
            append(size_t(0));
            أرجع;
        }
        خودڪار inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.پهريون->second);
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
        ثابت خودڪار * بنياد = tensor->view_src ? tensor->view_src : tensor;
        ثابت منطقي وزن = بنياد->بفر && بنياد->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS;
        append(وزن);
        إذا (وزن && !disk_cache) {
            ثابت size_t بفر_id = ggml_backend_openvino_بفر_get_ctx_id(بنياد->بفر);
            has_وزن_بفر_id |= بفر_id != 0;
            append(بفر_id);
            append(tensor->معلومات);
        }
        گهمو(tensor->view_src);
        لكل (ثابت خودڪار * src : tensor->src) {
            گهمو(src);
        }
    };
    append(graph->n_nodes);
    لكل (صحيح i = 0; i < graph->n_nodes; ++i) {
        گهمو(graph->nodes[i]);
    }
    append(graph->n_leafs);
    لكل (صحيح i = 0; i < graph->n_leafs; ++i) {
        گهمو(graph->leafs[i]);
    }
    لكل (ثابت خودڪار & داخلا_2 : decoder.get_model_extra_inputs()) {
        append_string(داخلا_2.پهريون);
        append_string(داخلا_2.second.type.get_type_name());
        append(داخلا_2.second.shape.size());
        لكل (خودڪار dim : داخلا_2.second.shape) {
            append(dim);
        }
        append(داخلا_2.second.is_parameter);
        إذا (!داخلا_2.second.is_parameter) {
            append(داخلا_2.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    أرجع disk_cache || has_وزن_بفر_id ? ڪنجي : std::string{};
}

ساكن std::string dynamic_گراف_signature(ثابت ggml_cgraph * graph,
                                           ثابت GgmlOvDecoder & decoder,
                                           ثابت ModelParams & params) {
    std::string ڪنجي = "dynamic-1";
    خودڪار append = [&ڪنجي](ثابت خودڪار & value) {
        ڪنجي.append(reinterpret_cast<ثابت حرف *>(&value), حجم(value));
    };
    خودڪار append_string = [&](ثابت std::string & value) {
        append(value.size());
        ڪنجي.append(value);
    };
    گراف_ڪنجي گراف_id(graph, سچ);
    append(گراف_id.n_nodes);
    append(گراف_id.n_leaves);
    append_string(گراف_id.پهريون_ڳنڍ_name);
    append_string(گراف_id.آخري_ڳنڍ_name);
    لكل (ثابت خودڪار & name : گراف_id.داخلا_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    لكل (صحيح layer : params.swa_layers) {
        append(layer);
    }
    لكل (ثابت خودڪار & [layer, heads] : params.n_heads_kv_في_layer) {
        append(layer);
        append(heads);
    }
    لكل (ثابت خودڪار & [name, داخلا_2] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(داخلا_2.type.get_type_name());
    }
    لكل (ثابت خودڪار & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    لكل (ثابت خودڪار & [name, داخلا_2] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(داخلا_2.type.get_type_name());
        append(داخلا_2.is_parameter);
        إذا (!داخلا_2.is_parameter) {
            append(داخلا_2.value);
        }
    }
    أرجع ڪنجي;
}

ساكن منطقي compiled_model_matches_گراف(ثابت ov::CompiledModel & model,
                                         ثابت std::vector<std::string> & inputs,
                                         ثابت std::vector<std::string> & outputs,
                                         ثابت GgmlOvDecoder & decoder) {
    إذا (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        أرجع خطأ;
    }
    ثابت خودڪار & گراف_inputs = decoder.get_model_inputs();
    ثابت خودڪار & extra_inputs = decoder.get_model_extra_inputs();
    ثابت خودڪار & گراف_outputs = decoder.get_model_outputs();
    لكل (size_t i = 0; i < inputs.size(); ++i) {
        ثابت خودڪار & پورٽ = model.داخلا_2(i);
        خودڪار گراف_it = گراف_inputs.find(inputs[i]);
        إذا (گراف_it != گراف_inputs.end()) {
            إذا (پورٽ.get_element_type() != گراف_it->second.type ||
                !پورٽ.get_partial_shape().compatible(گراف_it->second.shape)) {
                أرجع خطأ;
            }
            استمر;
        }
        خودڪار extra_it = extra_inputs.find(inputs[i]);
        إذا (extra_it == extra_inputs.end() || پورٽ.get_element_type() != extra_it->second.type ||
            !پورٽ.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            أرجع خطأ;
        }
    }
    لكل (size_t i = 0; i < outputs.size(); ++i) {
        خودڪار گراف_it = گراف_outputs.find(outputs[i]);
        إذا (گراف_it == گراف_outputs.end()) {
            إذا (outputs[i].compare(0, 8, "__debug_") != 0) {
                أرجع خطأ;
            }
            استمر;
        }
        ثابت خودڪار & پورٽ = model.output(i);
        إذا (پورٽ.get_element_type() != GgmlOvDecoder::get_ov_type(گراف_it->second) ||
            !پورٽ.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(گراف_it->second)))) {
            أرجع خطأ;
        }
    }
    أرجع سچ;
}

ساكن std::string ov_profiling_csv_ميدان(ثابت std::string & value) {
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

    ساكن std::atomic<uint64_t> inference_اشاريو{0};
    ثابت uint64_t اشاريو = inference_اشاريو.fetch_add(1);
    ثابت std::string path = "openvino_profile_" + std::to_string(اشاريو) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    إذا (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        أرجع;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t ڪل_real_وقت = 0;
    int64_t ڪل_cpu_وقت = 0;
    size_t executed_ڳڻپ = 0;
    لكل (ثابت خودڪار & info : infer_request.get_profiling_info()) {
        ثابت حرف * حالت = "NOT_RUN";
        إذا (info.حالت == ov::ProfilingInfo::حالت_2::EXECUTED) {
            حالت = "EXECUTED";
            ڪل_real_وقت += info.real_وقت.count();
            ڪل_cpu_وقت += info.cpu_وقت.count();
            executed_ڳڻپ++;
        } وإلا إذا (info.حالت == ov::ProfilingInfo::حالت_2::OPTIMIZED_ٻاهريون) {
            حالت = "OPTIMIZED_OUT";
        }

        output << حالت << ',' << info.real_وقت.count() << ',' << info.cpu_وقت.count() << ','
               << info.شروعات_وقت.count() << ',' << ov_profiling_csv_ميدان(info.node_type) << ','
               << ov_profiling_csv_ميدان(info.ڳنڍ_name) << ',' << ov_profiling_csv_ميدان(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_ڳڻپ, ڪل_real_وقت / 1000.0, ڪل_cpu_وقت / 1000.0);
}

ساكن فراغ reset_single_جڳهه_recurrent_cache(ggml_cgraph * cgraph,
                                              ثابت ModelParams & model_params,
                                              ثابت ComputeParams & compute_params) {
    إذا (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        أرجع;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_بفر_t> buffers;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        خودڪار * ڳنڍ = cgraph->nodes[i];
        إذا (ڳنڍ->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(ڳنڍ->view_src) &&
            ڳنڍ->view_src->ne[1] == 1 && ڳنڍ->view_src->بفر != خالي_اشارو) {
            buffers.insert(ڳنڍ->view_src->بفر);
        }
    }
    لكل (خودڪار * بفر : buffers) {
        ggml_backend_بفر_clear(بفر, 0);
    }
}

ov::Tensor create_ov_output_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   ثابت std::shared_ptr<ov::InferRequest> & infer_request,
                                   صحيح output_اشاريو,
                                   ثابت ggml_tensor * ggml_tensor) {
    إذا (خودڪار sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        أرجع *sliced;
    }

    خودڪار output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    فراغ * output_معلومات = ggml_tensor->معلومات;
    إذا (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_اشاريو).get_shape();
    } وإلا {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        إذا (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != خالي_اشارو &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_معلومات = ggml_tensor->view_src->معلومات;
        } وإلا {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    إذا (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_بفر_is_remote(ggml_tensor) &&
        ggml_tensor->extra != خالي_اشارو) {
        خودڪار * extra_بنياد = static_cast<ggml_openvino_extra_بنياد *>(ggml_tensor->extra);
        إذا (extra_بنياد->type == ggml_openvino_extra_بنياد::Type::TENSOR) {
            خودڪار * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_بنياد);
            إذا (tensor_extra->tensor != خالي_اشارو && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                أرجع *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_معلومات);
    أرجع output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_صفون_منزل_seq_axis_2(ثابت ov::Tensor & kv_tensor, size_t n_heads_kv) {
    ثابت size_t صفون = kv_tensor.get_shape()[2];
    ثابت size_t سر_سائيز = kv_tensor.get_shape()[3] / n_heads_kv;
    ثابت size_t elem = kv_tensor.get_element_type().size();
    ثابت size_t سر_بائيٽون = سر_سائيز * elem;

    ov::Tensor ٻاهريون(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, صفون, سر_سائيز});
    ثابت خودڪار * src = static_cast<ثابت uint8_t *>(kv_tensor.معلومات());
    خودڪار * dst = static_cast<uint8_t *>(ٻاهريون.معلومات());
    لكل (size_t s = 0; s < صفون; s++) {
        لكل (size_t h = 0; h < n_heads_kv; h++) {
            يادگيري_نقل(dst + (h * صفون + s) * سر_بائيٽون, src + (s * n_heads_kv + h) * سر_بائيٽون, سر_بائيٽون);
        }
    }
    أرجع ٻاهريون;
}

سانچو <قسم_نالو T> فراغ سيٽ_zero_diagonal(std::vector<T> & ميٽرڪس, size_t صفون, size_t کانن_ڳڻپ_2, T zero_قدر = T{}) {
    لكل (size_t i = 0; i < صفون; ++i) {
        size_t diag_کانو = std::min(i, کانن_ڳڻپ_2 - 1);
        ميٽرڪس[i * کانن_ڳڻپ_2 + diag_کانو] = zero_قدر;
    }
}

ov::Tensor make_contiguous_ورهايو_داخلا_tensor(ثابت بنية ggml_tensor * ggml_tensor, ثابت ov::Shape & داخلا_shape) {
    ثابت size_t element_سائيز = ggml_type_سائيز(ggml_tensor->type);
    ثابت size_t بلاڪ_سائيز_3 = ggml_blck_سائيز(ggml_tensor->type);

    GGML_ASSERT(بلاڪ_سائيز_3 == 1 && "non-contiguous split inputs must be plain element types");

    ثابت بنية ggml_tensor * ذريعو_tensor = ggml_tensor->view_src != خالي_اشارو ? ggml_tensor->view_src : ggml_tensor;
    ثابت size_t ذريعو_وٿي = ggml_tensor->view_src != خالي_اشارو ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> ذريعو_معلومات(ggml_nbytes(ذريعو_tensor));
    ggml_backend_tensor_get(ذريعو_tensor, ذريعو_معلومات.معلومات(), 0, ذريعو_معلومات.size());

    ov::Tensor داخلا_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلا_shape);
    خودڪار * dst = static_cast<uint8_t *>(داخلا_tensor.معلومات());
    size_t dst_وٿي = 0;

    لكل (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        لكل (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            لكل (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                لكل (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    ثابت size_t src_وٿي = ذريعو_وٿي + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::يادگيري_نقل(dst + dst_وٿي, ذريعو_معلومات.معلومات() + src_وٿي, element_سائيز);
                    dst_وٿي += element_سائيز;
                }
            }
        }
    }

    أرجع داخلا_tensor;
}

ov::Tensor convert_ggml_داخلا_منزل_ov(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & name) {
    ثابت خودڪار * ggml_tensor = ggml_decoder->get_داخلا_ggml_tensor(name);

    إذا (خودڪار sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        أرجع *sliced;
    }

    إذا (ggml_tensor->extra != خالي_اشارو && !ggml_decoder->is_splited_model()) {
        خودڪار * extra_بنياد = static_cast<ggml_openvino_extra_بنياد *>(ggml_tensor->extra);
        إذا (extra_بنياد->type == ggml_openvino_extra_بنياد::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            خودڪار * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_بنياد);
            أرجع *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    خودڪار * داخلا_معلومات = ggml_tensor->معلومات;
    ov::Shape داخلا_shape;
    إذا (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        داخلا_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } وإلا {
        داخلا_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    إذا (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        أرجع make_contiguous_ورهايو_داخلا_tensor(ggml_tensor, داخلا_shape);
    }

    خودڪار داخلا_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلا_shape, داخلا_معلومات);
    أرجع داخلا_tensor;
}

ov::Tensor get_ov_داخلا_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & param_name) {
    ov::Tensor داخلا_tensor;
    خودڪار extra_داخلا = ggml_decoder->get_model_extra_inputs().find(param_name);
    إذا (extra_داخلا != ggml_decoder->get_model_extra_inputs().end()) {
        داخلا_tensor = ov::Tensor(extra_داخلا->second.type, extra_داخلا->second.shape);
        *داخلا_tensor.معلومات<int64_t>() = extra_داخلا->second.value;
    } وإلا {
        داخلا_tensor = convert_ggml_داخلا_منزل_ov(ggml_decoder, param_name);
    }
    أرجع داخلا_tensor;
}

ov::Tensor get_ov_داخلا_tensor_static_decode(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             ثابت std::string & param_name) {
    // NPU decoding stage
    إذا (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        أرجع get_ov_داخلا_tensor(ggml_decoder, param_name);
    }
    ثابت خودڪار * ggml_tensor = ggml_decoder->get_داخلا_ggml_tensor(param_name);
    ثابت خودڪار * op = ggml_decoder->get_tensor_استعمال_ٿيل_op(ggml_tensor);

    إذا (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        ثابت صحيح n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        أكد(ggml_tensor->ne[0] == n_planes);
        ov::Shape داخلا_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor داخلا_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلا_shape);
        std::يادگيري_نقل(داخلا_tensor.معلومات(), ggml_tensor->معلومات, n_planes * ggml_type_سائيز(ggml_tensor->type));
        أرجع داخلا_tensor;
    }

    إذا (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape داخلا_shape = {1, 1, 1, 1};
        ov::Tensor داخلا_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلا_shape);
        int32_t inp_ٻاهريون_id = *((int32_t *) ggml_tensor->معلومات);
        أكد(ggml_tensor->ne[0] == 1);
        أكد(inp_ٻاهريون_id == 0);
        *داخلا_tensor.معلومات<int32_t>() = inp_ٻاهريون_id;
        أرجع داخلا_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_نقاب(ggml_tensor, op)) {
        size_t context_سائيز = ggml_decoder->get_ctx_سائيز();
        إذا (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_معلومات =
                pad_داخلا<ggml_fp16_t>(ggml_tensor, 1, context_سائيز, GGML_FP32_منزل_FP16(-INFINITY));
            ov::Tensor داخلا_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_سائيز});
            std::يادگيري_نقل(داخلا_tensor.معلومات(), padded_معلومات.معلومات(), padded_معلومات.size() * حجم(ggml_fp16_t));
            أرجع داخلا_tensor;
        }

        std::vector<عائم> padded_معلومات = pad_داخلا<عائم>(ggml_tensor, 1, context_سائيز, -INFINITY);
        ov::Tensor داخلا_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_سائيز});
        خودڪار * معلومات_ptr = داخلا_tensor.معلومات<عائم>();
        std::copy(padded_معلومات.begin(), padded_معلومات.begin() + context_سائيز, معلومات_ptr);
        أرجع داخلا_tensor;
    }

    أرجع get_ov_داخلا_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_داخلا_tensor_static_prefill(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              ثابت std::string & param_name,
                                              صحيح chunk_اشاريو) {
    // NPU prompt processing stage
    ثابت size_t داخلا_len = ggml_decoder->get_داخلا_len();
    ثابت size_t chunk_سائيز = ggml_decoder->m_prefill_chunk_سائيز;
    ثابت size_t chunk_valid_سائيز = std::min(chunk_سائيز, داخلا_len - chunk_اشاريو * chunk_سائيز);
    ثابت size_t chunk_pad_سائيز = chunk_سائيز - chunk_valid_سائيز;

    إذا (param_name == "chunk_valid_len") {
        ov::Tensor داخلا_tensor(ov::element::i64, ov::Shape{1});
        *داخلا_tensor.معلومات<int64_t>() = (int64_t) chunk_valid_سائيز;
        أرجع داخلا_tensor;
    }
    إذا (chunk_اشاريو > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor داخلا_tensor(ov::element::i64, ov::Shape{1});
        *داخلا_tensor.معلومات<int64_t>() = 0;
        أرجع داخلا_tensor;
    }
    إذا (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        أرجع get_ov_داخلا_tensor(ggml_decoder, param_name);
    }
    ثابت خودڪار * ggml_tensor = ggml_decoder->get_داخلا_ggml_tensor(param_name);
    ثابت خودڪار * op = ggml_decoder->get_tensor_استعمال_ٿيل_op(ggml_tensor);

    إذا (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        ثابت صحيح n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        ثابت size_t element_سائيز = ggml_type_سائيز(ggml_tensor->type);
        ov::Shape داخلا_shape = {1, 1, 1, (size_t) n_planes * chunk_سائيز};
        ov::Tensor داخلا_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلا_shape);
        لكل (صحيح p = 0; p < n_planes; p++) {
            ثابت حرف * src =
                (ثابت حرف *) ggml_tensor->معلومات + (p * داخلا_len + chunk_اشاريو * chunk_سائيز) * element_سائيز;
            حرف * dst = (حرف *) داخلا_tensor.معلومات() + p * chunk_سائيز * element_سائيز;
            std::يادگيري_نقل(dst, src, chunk_valid_سائيز * element_سائيز);
            إذا (chunk_pad_سائيز > 0) {
                إذا (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t آخري_قدر = *((ثابت int32_t *) src + chunk_valid_سائيز - 1);
                    int32_t * ٻاهريون = (int32_t *) dst;
                    std::fill(ٻاهريون + chunk_valid_سائيز, ٻاهريون + chunk_سائيز, آخري_قدر + 1);
                } وإلا إذا (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t آخري_قدر = *((ثابت int64_t *) src + chunk_valid_سائيز - 1);
                    int64_t * ٻاهريون = (int64_t *) dst;
                    std::fill(ٻاهريون + chunk_valid_سائيز, ٻاهريون + chunk_سائيز, آخري_قدر + 1);
                } وإلا {
                    ارم std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        أرجع داخلا_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape داخلا_shape = {1, 1, 1, chunk_سائيز};
        ov::Tensor داخلا_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلا_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_سائيز = ggml_type_سائيز(ggml_tensor->type);
        فراغ * داخلا_معلومات = (حرف *) ggml_tensor->معلومات + chunk_اشاريو * chunk_سائيز * element_سائيز;
        std::يادگيري_نقل(داخلا_tensor.معلومات(), داخلا_معلومات, chunk_valid_سائيز * element_سائيز);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        إذا (chunk_pad_سائيز > 0) {
            إذا (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t آخري_قدر =
                    *((int32_t *) ggml_tensor->معلومات + (chunk_اشاريو * chunk_سائيز + chunk_valid_سائيز - 1));
                int32_t * output_معلومات = داخلا_tensor.معلومات<int32_t>();
                std::fill(output_معلومات + chunk_valid_سائيز, output_معلومات + chunk_سائيز, آخري_قدر + 1);
            } وإلا إذا (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t آخري_قدر =
                    *((int64_t *) ggml_tensor->معلومات + (chunk_اشاريو * chunk_سائيز + chunk_valid_سائيز - 1));
                int64_t * output_معلومات = داخلا_tensor.معلومات<int64_t>();
                std::fill(output_معلومات + chunk_valid_سائيز, output_معلومات + chunk_سائيز, آخري_قدر + 1);
            } وإلا {
                ارم std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        أرجع داخلا_tensor;
    }

    إذا (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape داخلا_shape = {1, 1, 1, output_len};
        ov::Tensor داخلا_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), داخلا_shape);
        إذا (ggml_tensor->ne[0] == 0) {
            *داخلا_tensor.معلومات<int32_t>() = 0;
        } وإلا {
            خودڪار * معلومات_addr = داخلا_tensor.معلومات<int32_t>();
            لكل (size_t i = 0; i < output_len; i++) {
                معلومات_addr[i] = ((int32_t *) ggml_tensor->معلومات)[i] % chunk_سائيز;
            }
        }
        أرجع داخلا_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_scale_صفون(ggml_tensor, op)) {
        ov::Tensor داخلا_tensor(ov::element::f32, ov::Shape{1, 1, chunk_سائيز, 1});
        خودڪار * dst = داخلا_tensor.معلومات<عائم>();
        ثابت خودڪار * src = static_cast<ثابت عائم *>(ggml_tensor->معلومات) + chunk_اشاريو * chunk_سائيز;
        std::copy(src, src + chunk_valid_سائيز, dst);
        std::fill(dst + chunk_valid_سائيز, dst + chunk_سائيز, 1.0f);
        أرجع داخلا_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_سراسري(ggml_tensor, op)) {
        ثابت size_t n_seqs = ggml_tensor->ne[1];
        ثابت size_t src_stride = ggml_tensor->ne[0];
        ثابت size_t copy_len = std::min<size_t>(chunk_valid_سائيز, src_stride - chunk_اشاريو * chunk_سائيز);
        ov::Tensor داخلا_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_سائيز});
        خودڪار * dst = داخلا_tensor.معلومات<عائم>();
        std::fill(dst, dst + n_seqs * chunk_سائيز, 0.0f);
        ثابت خودڪار * src = static_cast<ثابت عائم *>(ggml_tensor->معلومات) + chunk_اشاريو * chunk_سائيز;
        لكل (size_t s = 0; s < n_seqs; s++) {
            std::يادگيري_نقل(dst + s * chunk_سائيز, src + s * src_stride, copy_len * حجم(عائم));
        }
        أرجع داخلا_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_نقاب(ggml_tensor, op)) {
        size_t کانن_ڳڻپ_2 = ggml_tensor->ne[0];
        size_t صفون = ggml_tensor->ne[1];
        size_t chunk_valid_صفون = std::min(chunk_سائيز, صفون - chunk_اشاريو * chunk_سائيز);
        size_t context_سائيز = ggml_decoder->get_ctx_سائيز();
        إذا (ggml_tensor->type == GGML_TYPE_F16) {
            ثابت خودڪار * ggml_معلومات =
                static_cast<ثابت ggml_fp16_t *>(ggml_tensor->معلومات) + chunk_اشاريو * chunk_سائيز * کانن_ڳڻپ_2;
            std::vector<ggml_fp16_t> padded_معلومات = pad_داخلا<ggml_fp16_t>(ggml_معلومات, chunk_valid_صفون, کانن_ڳڻپ_2, chunk_سائيز,
                                                                          context_سائيز, GGML_FP32_منزل_FP16(-INFINITY));
            سيٽ_zero_diagonal(padded_معلومات, chunk_سائيز, context_سائيز, GGML_FP32_منزل_FP16(0.0f));
            ov::Tensor داخلا_tensor(ov::element::f16, ov::Shape{1, 1, chunk_سائيز, context_سائيز});
            std::يادگيري_نقل(داخلا_tensor.معلومات(), padded_معلومات.معلومات(), padded_معلومات.size() * حجم(ggml_fp16_t));
            أرجع داخلا_tensor;
        }

        ثابت خودڪار * ggml_معلومات = static_cast<ثابت عائم *>(ggml_tensor->معلومات) + chunk_اشاريو * chunk_سائيز * کانن_ڳڻپ_2;
        std::vector<عائم> padded_معلومات =
            pad_داخلا<عائم>(ggml_معلومات, chunk_valid_صفون, کانن_ڳڻپ_2, chunk_سائيز, context_سائيز, -INFINITY);
        سيٽ_zero_diagonal(padded_معلومات, chunk_سائيز, context_سائيز);
        ov::Tensor داخلا_tensor(ov::element::f32, ov::Shape{1, 1, chunk_سائيز, context_سائيز});
        خودڪار * معلومات_ptr = داخلا_tensor.معلومات<عائم>();
        std::copy(padded_معلومات.begin(), padded_معلومات.begin() + chunk_سائيز * context_سائيز, معلومات_ptr);
        أرجع داخلا_tensor;
    }

    أرجع get_ov_داخلا_tensor(ggml_decoder, param_name);
}

تعداد ggml_حالت naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               ثابت std::string & device,
                               ثابت ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    إذا (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        أرجع GGML_حالت_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_تالو(cache.mutex);
    منطقي naive = سچ;
    خودڪار model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph, naive);
    خودڪار decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    خودڪار داخلا_model = std::make_shared<ov::frontend::ggml::داخلا_Model>(decoder);
    خودڪار model = ov::frontend::ggml::Front_پڇاڙي::convert(داخلا_model, naive);
    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    خودڪار remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    إذا (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_گهڻائي.name()] = ov::hint::Execution_گهڻائي::PERFORMANCE;
    } وإلا {
        compile_config[ov::hint::execution_گهڻائي.name()] = ov::hint::Execution_گهڻائي::ACCURACY;
    }
    إذا (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } وإلا {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> داخلا_نالا;
    std::vector<std::string> output_نالا;
    لكل (ثابت خودڪار & param : model->get_parameters()) {
        داخلا_نالا.push_back(param->get_friendly_name());
    }
    لكل (ثابت خودڪار & نتيجو : model->get_results()) {
        output_نالا.push_back(نتيجو->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    داخلا_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_تالو.unlock();

    لكل (size_t i = 0; i < داخلا_نالا.size(); i++) {
        ثابت خودڪار & param_name = داخلا_نالا[i];
        خودڪار داخلا_tensor = get_ov_داخلا_tensor(decoder, param_name);
        infer_request->سيٽ_داخلا_tensor(i, داخلا_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    لكل (size_t i = 0; i < output_نالا.size(); i++) {
        خودڪار output_tensor = infer_request->get_output_tensor(i);
        ثابت خودڪار & model_outputs = decoder->get_model_outputs();
        خودڪار model_output_it = model_outputs.find(output_نالا[i]);
        إذا (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_نالا[i], output_tensor, output_tensor.معلومات());
            }
            استمر;
        }
        خودڪار * ggml_tensor = model_output_it->second;
        std::يادگيري_نقل(ggml_tensor->معلومات, output_tensor.معلومات(), output_tensor.get_byte_سائيز());
    }
    أرجع GGML_حالت_SUCCESS;
}

تعداد ggml_حالت ov_گراف_compute_dynamic(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    خودڪار & core = ov_singleton_core();
    ثابت خودڪار & config = ggml_openvino_get_compile_config();
    ثابت خودڪار & device = r_ctx->device;
    ساكن خودڪار is_static = خطأ;

    ساكن ثابت منطقي cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    خودڪار شروعات_وقت = ggml_وقت_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    ثابت int64_t cache_ڪنجي_شروعات_وقت = ggml_وقت_us();
    گراف_ڪنجي ڪنجي(cgraph);
    ثابت int64_t cache_ڪنجي_compute_وقت = ggml_وقت_us() - cache_ڪنجي_شروعات_وقت;
    int64_t cache_lookup_وقت = 0;
    منطقي ڪنجي_ڏٺل = خطأ;
    إذا (!cache_disabled) {
        ثابت int64_t cache_lookup_شروعات_وقت = ggml_وقت_us();
        std::lock_guard<std::mutex> map_تالو(r_ctx->ctx_mutex);
        ڪنجي_ڏٺل = r_ctx->decoder_cache.find(ڪنجي) != r_ctx->decoder_cache.end();
        cache_lookup_وقت += ggml_وقت_us() - cache_lookup_شروعات_وقت;
    }

    ثابت int64_t model_ورهايو_check_شروعات_وقت = ggml_وقت_us();
    منطقي model_is_splitted = ڪنجي_ڏٺل ? خطأ : is_model_splitted(cgraph);
    ثابت int64_t model_ورهايو_check_وقت = ggml_وقت_us() - model_ورهايو_check_شروعات_وقت;

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
    ثابت int64_t گراف_param_compute_شروعات_وقت = ggml_وقت_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    ثابت int64_t گراف_param_compute_وقت = ggml_وقت_us() - گراف_param_compute_شروعات_وقت;

    ثابت منطقي cache_enabled = !model_is_splitted && !cache_disabled;
    ثابت منطقي stateful_recurrent = r_ctx->stateful && m_params.state_سائيز >= 0;
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
    منطقي cache_ڪاميابي = خطأ;

    int64_t decoder_پڇاڙي_وقت;
    int64_t conversion_پڇاڙي_وقت;
    int64_t compile_پڇاڙي_وقت;
    int64_t infer_پڇاڙي_وقت;
    int64_t ov_raw_infer_شروعات;

    {
        std::shared_ptr<decoder_runtime_ctx> داخلا;
        ModelParams old_m_params;

        ثابت int64_t cache_lookup_شروعات_وقت = ggml_وقت_us();
        إذا (cache_enabled) {
            std::lock_guard<std::mutex> map_تالو(r_ctx->ctx_mutex);
            خودڪار it = r_ctx->decoder_cache.find(ڪنجي);
            cache_ڪاميابي = it != r_ctx->decoder_cache.end();
            إذا (cache_ڪاميابي) {
                داخلا = it->second;
            } وإلا {
                r_ctx->clear_caches_locked();
                خودڪار mutex = std::make_shared<std::mutex>();
                داخلا = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[ڪنجي] = داخلا;
            }
        } وإلا {
            خودڪار mutex = std::make_shared<std::mutex>();
            داخلا = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_ڪاميابي = خطأ;
        }
        cache_lookup_وقت += ggml_وقت_us() - cache_lookup_شروعات_وقت;

        std::lock_guard<std::mutex> تالو(*(داخلا->mutex));
        cache_ڪاميابي = cache_ڪاميابي && داخلا->ptr && r_ctx->infer_request_cache.count(ڪنجي) != 0;

        إذا (cache_ڪاميابي) {
            ggml_decoder = داخلا->ptr;
            old_m_params = ggml_decoder->get_model_params();
            إذا (!ggml_decoder->is_splited_model()) {
                cache_ڪاميابي = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_داخلا_نالا;
        std::vector<std::string> ov_output_نالا;

        إذا (cache_ڪاميابي) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->سيٽ_compute_params(c_params);
            ggml_decoder->سيٽ_model_params(m_params);
            إذا (old_m_params.kv_بفر_بدليو(m_params) || !ggml_decoder->is_bound_منزل(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->جوڙيو_extra_inputs();
            {
                std::lock_guard<std::mutex> map_تالو(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(ڪنجي);
                ov_داخلا_نالا = r_ctx->ov_داخلا_نالا_cache.at(ڪنجي);
                ov_output_نالا = r_ctx->ov_output_نالا_cache.at(ڪنجي);
            }

            إذا (stateful_kv_only) {
                ثابت خودڪار * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_معلومات = (int32_t *) inp_pos->معلومات;
                خودڪار pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                إذا (pos_معلومات[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_سائيز = pos_shape[3];
                } وإلا إذا (r_ctx->stateful_kv_سائيز == static_cast<size_t>(pos_معلومات[0])) {
                    r_ctx->stateful_kv_سائيز += pos_shape[3];
                } وإلا {
                    ثابت size_t pos_begin = static_cast<size_t>(pos_معلومات[0]);
                    ثابت منطقي refill = pos_begin > r_ctx->stateful_kv_سائيز;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    إذا (refill && !ggml_decoder->get_model_params().swa_layers.خالي_جايون()) {
                        ثابت صحيح n_swa = ggml_decoder->get_compute_params().swa_دري;
                        إذا (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_سائيز, n_swa);
                            أرجع GGML_حالت_FAILED;
                        }
                    }

                    ثابت منطقي relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    خودڪار حالتون = infer_request->query_state();
                    لكل (خودڪار state : حالتون) {
                        خودڪار state_tensor = state.get_state();
                        خودڪار state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        إذا (خودڪار it = r_ctx->kv_state_داخلا_name_map.find(state.get_name());
                            it != r_ctx->kv_state_داخلا_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        صحيح n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        إذا (خودڪار layer = ڪڍو_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        ثابت منطقي relayout_this_state = relayout_enabled;
                        ثابت size_t seq_axis = relayout_this_state ? 2 : 1;
                        ثابت size_t سر_axis = seq_axis == 2 ? 1 : 2;

                        إذا (refill) {
                            إذا (state_name.خالي_جايون()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                أرجع GGML_حالت_FAILED;
                            }
                            خودڪار kv_tensor = get_ov_داخلا_tensor(ggml_decoder, state_name);
                            إذا (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_صفون_منزل_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } وإلا {
                                ov::Shape refill_shape(4);
                                refill_shape[0] = state_tensor_shape[0];
                                refill_shape[seq_axis] = kv_tensor.get_shape()[2];
                                refill_shape[سر_axis] = state_tensor_shape[سر_axis];
                                refill_shape[3] = state_tensor_shape[3];
                                kv_tensor.سيٽ_shape(refill_shape);
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
                        state.سيٽ_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_سائيز = pos_begin + pos_shape[3];
                }
            }

            decoder_پڇاڙي_وقت = ggml_وقت_us();
            conversion_پڇاڙي_وقت = decoder_پڇاڙي_وقت;
            compile_پڇاڙي_وقت = decoder_پڇاڙي_وقت;
        } وإلا {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            خودڪار shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_تالو(shared_cache->mutex);
            خودڪار وزن_نالا = get_وزن_نالا(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, وزن_نالا,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ثابت std::string model_cache_dir = ggml_openvino_model_cache_dir();
            ثابت std::string exact_ڪنجي = cache_enabled ? compiled_گراف_ڪنجي(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_رستو, manifest_رستو;
            إذا (!model_cache_dir.خالي_جايون() && !model_is_splitted) {
                ثابت uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/سچ, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_گراف_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/سچ, m_params.rope_params, 16, extra_cfg,
                                                    compiled_گراف_ڪنجي(cgraph, *ggml_decoder, device, 0, سچ));
                blob_رستو = ggml_openvino_model_cache_blob_رستو(model_cache_dir, model_fp);
                manifest_رستو = ggml_openvino_model_cache_manifest_رستو(model_cache_dir, model_fp);
            }
            std::string shared_ڪنجي =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_ڪنجي) : "";
            ov::CompiledModel shared_model;
            منطقي imported = خطأ;
            خودڪار shared_it = shared_cache->graphs.find(shared_ڪنجي);
            إذا (!shared_ڪنجي.خالي_جايون() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_گراف(shared_it->second.decode, shared_it->second.داخلا_نالا,
                                                               shared_it->second.output_نالا, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_داخلا_نالا = shared_it->second.داخلا_نالا;
                ov_output_نالا = shared_it->second.output_نالا;
                imported = سچ;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } وإلا إذا (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_ڪنجي = exact_ڪنجي;
                shared_it = shared_cache->graphs.find(shared_ڪنجي);
                إذا (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_داخلا_نالا = shared_it->second.داخلا_نالا;
                    ov_output_نالا = shared_it->second.output_نالا;
                    imported = سچ;
                }
                model_fp = exact_fp;
                blob_رستو = ggml_openvino_model_cache_blob_رستو(model_cache_dir, model_fp);
                manifest_رستو = ggml_openvino_model_cache_manifest_رستو(model_cache_dir, model_fp);
            }
            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_تالو(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(ڪنجي);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            إذا (!model_cache_dir.خالي_جايون()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_گهڻائي.name()] = ov::Cache_گهڻائي::OPTIMIZE_رفتار;
            }
            إذا (!imported && !model_cache_dir.خالي_جايون() && !model_is_splitted) {
                ثابت uint64_t preferred_fp = model_fp;
                منطقي dynamic_incompatible = preferred_fp == exact_fp;
                لكل (صحيح candidate = 0; candidate < 2; ++candidate) {
                    إذا (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        اكسر;
                    }
                    ثابت uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    ثابت std::string candidate_blob =
                        ggml_openvino_model_cache_blob_رستو(model_cache_dir, candidate_fp);
                    ثابت std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_رستو(model_cache_dir, candidate_fp);
                    std::ifstream blob_اندريون(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    إذا (!blob_اندريون.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        استمر;
                    }
                    ثابت int64_t import_شروعات = ggml_وقت_us();
                    حاول {
                        ov::CompiledModel cm;
                        خودڪار remote_context = ggml_openvino_get_remote_context();
                        إذا (remote_context.has_value()) {
                            cm = core.import_model(blob_اندريون, remote_context.value(), mc_config);
                        } وإلا {
                            cm = core.import_model(blob_اندريون, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        إذا (!saved_inputs.خالي_جايون() || !saved_outputs.خالي_جايون()) {
                            ov_داخلا_نالا = std::move(saved_inputs);
                            ov_output_نالا = std::move(saved_outputs);
                        } وإلا {
                            لكل (ثابت خودڪار & p : cm.inputs()) {
                                ov_داخلا_نالا.push_back(p.get_ڳنڍ()->get_friendly_name());
                            }
                            لكل (ثابت خودڪار & o : cm.outputs()) {
                                ov_output_نالا.push_back(o.get_ڳنڍ()->get_friendly_name());
                            }
                        }
                        إذا (!compiled_model_matches_گراف(cm, ov_داخلا_نالا, ov_output_نالا, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_داخلا_نالا.clear();
                            ov_output_نالا.clear();
                            استمر;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        داخلا->ptr = ggml_decoder;
                        imported = سچ;
                        model_fp = candidate_fp;
                        blob_رستو = candidate_blob;
                        manifest_رستو = candidate_manifest;
                        إذا (candidate_fp == exact_fp) {
                            shared_ڪنجي = exact_ڪنجي;
                            shared_it = shared_cache->graphs.find(shared_ڪنجي);
                        }
                        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_وقت_us() - import_شروعات) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } امسك (ثابت std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = خطأ;
                        ov_داخلا_نالا.clear();
                        ov_output_نالا.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                إذا (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_ڪنجي = exact_ڪنجي;
                    shared_it = shared_cache->graphs.find(shared_ڪنجي);
                    blob_رستو = ggml_openvino_model_cache_blob_رستو(model_cache_dir, model_fp);
                    manifest_رستو = ggml_openvino_model_cache_manifest_رستو(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            إذا (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_رستو.c_str());
                أرجع GGML_حالت_FAILED;
            }
            إذا (!imported && ggml_openvino_وزن_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                أرجع GGML_حالت_FAILED;
            }
            إذا (imported) {
                decoder_پڇاڙي_وقت = conversion_پڇاڙي_وقت = compile_پڇاڙي_وقت = ggml_وقت_us();
            } وإلا {
                خودڪار model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_پڇاڙي_وقت = ggml_وقت_us();

                خودڪار داخلا_model = std::make_shared<ov::frontend::ggml::داخلا_Model>(ggml_decoder);
                model = ov::frontend::ggml::Front_پڇاڙي::convert(داخلا_model);
                ggml_decoder->clear_model_weights();
                conversion_پڇاڙي_وقت = ggml_وقت_us();

                إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    حرف timestamped_filename[64];
                    خودڪار timestamp = (ڊگهو ڊگهو) ggml_وقت_us();
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                ثابت ov::AnyMap & compile_config = model_cache_dir.خالي_جايون() ? config : mc_config;
                ov::CompiledModel compiled_model;
                خودڪار remote_context = ggml_openvino_get_remote_context();
                إذا (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } وإلا {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_پڇاڙي_وقت = ggml_وقت_us();

                لكل (ثابت خودڪار & ov_param : model->get_parameters()) {
                    ov_داخلا_نالا.push_back(ov_param->get_friendly_name());
                }
                لكل (ثابت خودڪار & ov_output : model->get_results()) {
                    ov_output_نالا.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                إذا (!model_cache_dir.خالي_جايون() && !model_is_splitted && model_fp != 0) {
                    حاول {
                        ثابت std::string blob_tmp = ggml_openvino_model_cache_عارضي_رستو(blob_رستو);
                        ثابت std::string manifest_tmp = ggml_openvino_model_cache_عارضي_رستو(manifest_رستو);
                        إذا (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_داخلا_نالا,
                                                                     ov_output_نالا)) {
                            std::ofstream blob_ٻاهريون(blob_tmp, std::ios::binary | std::ios::trunc);
                            إذا (blob_ٻاهريون.is_open()) {
                                compiled_model.export_model(blob_ٻاهريون);
                                blob_ٻاهريون.close();
                                إذا (blob_ٻاهريون.good()) {
                                    إذا (std::rename(blob_tmp.c_str(), blob_رستو.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_رستو.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_رستو.c_str());
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
                داخلا->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            داخلا->ptr = ggml_decoder;
            إذا (!shared_ڪنجي.خالي_جايون() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_ڪنجي,
                                             ov_compiled_گراف{shared_model, {}, ov_داخلا_نالا, ov_output_نالا});
            }
            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_تالو(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[ڪنجي] = infer_request;
                r_ctx->ov_داخلا_نالا_cache[ڪنجي] = ov_داخلا_نالا;
                r_ctx->ov_output_نالا_cache[ڪنجي] = ov_output_نالا;
            }

            إذا (stateful_kv_only && cache_enabled) {
                ثابت خودڪار * inp_pos = get_inp_pos_tensor(cgraph);
                خودڪار pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                ثابت int32_t pos_begin = ((int32_t *) inp_pos->معلومات)[0];
                إذا (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    أرجع GGML_حالت_FAILED;
                }
                r_ctx->stateful_kv_سائيز = pos_shape[3];
                ثابت خودڪار kv_param_res_نالا = ggml_decoder->get_kv_param_res_نالا();
                لكل (ثابت خودڪار & pair : kv_param_res_نالا) {
                    r_ctx->kv_state_داخلا_name_map[pair.پهريون + pair.second] = pair.پهريون;
                }
            }
        }

        إذا (stateful_recurrent) {
            ثابت خودڪار * inp_pos = get_inp_pos_tensor(cgraph);
            ثابت int32_t pos_begin = static_cast<ثابت int32_t *>(inp_pos->معلومات)[0];
            إذا (pos_begin == 0) {
                infer_request->reset_state();
            } وإلا إذا (!cache_ڪاميابي || old_m_params.kv_بفر_بدليو(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_سائيز) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                أرجع GGML_حالت_FAILED;
            }
        } وإلا {
            reset_single_جڳهه_recurrent_cache(cgraph, m_params, c_params);
        }

        لكل (size_t i = 0; i < ov_داخلا_نالا.size(); i++) {
            ثابت خودڪار & param_name = ov_داخلا_نالا[i];
            خودڪار داخلا_tensor = get_ov_داخلا_tensor(ggml_decoder, param_name);
            infer_request->سيٽ_داخلا_tensor(i, داخلا_tensor);

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_داخلا_tensor_info(param_name, داخلا_tensor);
            }
        }

        لكل (size_t i = 0; i < ov_output_نالا.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            ثابت خودڪار & model_outputs = ggml_decoder->get_model_outputs();
            خودڪار model_output_it = model_outputs.find(ov_output_نالا[i]);
            إذا (model_output_it == model_outputs.end()) {
                استمر;
            }
            خودڪار * ggml_tensor = model_output_it->second;
            إذا (ggml_nbytes(ggml_tensor) == 0) {
                استمر;
            }
            خودڪار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->سيٽ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_شروعات = ggml_وقت_us();
        infer_request->infer();
        infer_پڇاڙي_وقت = ggml_وقت_us();
        إذا (stateful_recurrent) {
            ثابت خودڪار * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_سائيز =
                static_cast<ثابت int32_t *>(inp_pos->معلومات)[0] + get_inp_pos_n_ٽوڪن(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            لكل (size_t i = 0; i < ov_output_نالا.size(); i++) {
                ثابت خودڪار output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_نالا[i], output_tensor, output_tensor.معلومات());
            }
        }

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_پڇاڙي_وقت - شروعات_وقت) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_ڪنجي_compute_وقت / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_ورهايو_check_وقت / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", گراف_param_compute_وقت / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_وقت / 1000.0);
            إذا (!cache_ڪاميابي) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_پڇاڙي_وقت - decoder_پڇاڙي_وقت) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_پڇاڙي_وقت - conversion_پڇاڙي_وقت) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_پڇاڙي_وقت - compile_پڇاڙي_وقت) / 1000.0);
            GGML_LOG_INFO("    - OV raw infer time: %.3f ms \n", (infer_پڇاڙي_وقت - ov_raw_infer_شروعات) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. Release only while holding the compilation mutex so another context cannot
    // be reading host weights during conversion/compilation. Pin the shared compiled
    // models across backend teardown; a later context can create its own request without
    // reading the dropped pages. A new, uncached graph still fails fast above.
    إذا (cache_ڪاميابي && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_تالو(r_ctx->compiled_cache->mutex);
        إذا (!ggml_openvino_وزن_buffers_released()) {
            ggml_openvino_release_وزن_buffers();
        }
    }

    أرجع GGML_حالت_SUCCESS;
}

ov::AnyMap without_npuw(ثابت ov::AnyMap & config) {
    ov::AnyMap ٻاهريون;
    لكل (ثابت خودڪار & kv : config) {
        إذا (kv.پهريون.rfind("NPUW", 0) == 0 || kv.پهريون == "NPU_USE_NPUW") {
            استمر;
        }
        ٻاهريون.insert(kv);
    }
    أرجع ٻاهريون;
}

تعداد ggml_حالت ov_گراف_compute_static(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    خودڪار & core = ov_singleton_core();

    خودڪار get_prefill_chunk_سائيز = [] {
        ساكن ثابت صحيح chunk_سائيز = []() {
            صحيح env_prefill_chunk_سائيز = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            أرجع env_prefill_chunk_سائيز > 0 ? env_prefill_chunk_سائيز : 256;
        }();
        أرجع chunk_سائيز;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    ساكن std::string device = ggml_openvino_get_device_name();
    ساكن خودڪار is_static = سچ;
    ساكن خودڪار stateful = خطأ;

    خودڪار prefill_chunk_سائيز = get_prefill_chunk_سائيز();
    ثابت خودڪار & config = ggml_openvino_get_compile_config();

    إذا (is_naive(cgraph)) {
        أرجع naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    خودڪار شروعات_وقت = ggml_وقت_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    ثابت خودڪار * inp_pos = get_inp_pos_tensor(cgraph);
    ثابت منطقي no_kv_cache = m_params.is_cacheless_attn;
    ثابت خودڪار is_prefill = no_kv_cache ? سچ : get_is_prefill(cgraph, inp_pos);
    ثابت ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    إذا (m_params.n_heads_kv == -1) {
        prefill_chunk_سائيز = inp_pos->ne[0];
    }
    گراف_ڪنجي ڪنجي(cgraph);
    ساكن ثابت منطقي cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    منطقي cache_ڪاميابي = خطأ;

    int64_t decoder_پڇاڙي_وقت;
    int64_t conversion_پڇاڙي_وقت;
    int64_t compile_پڇاڙي_وقت;
    int64_t infer_پڇاڙي_وقت;
    int64_t ov_raw_infer_شروعات;
    int64_t ov_raw_infer_ڪل = 0;

    std::shared_ptr<decoder_runtime_ctx> داخلا;
    ModelParams old_m_params;

    إذا (cache_enabled) {
        std::lock_guard<std::mutex> map_تالو(r_ctx->ctx_mutex);
        خودڪار it = r_ctx->decoder_cache.find(ڪنجي);
        cache_ڪاميابي = it != r_ctx->decoder_cache.end();
        إذا (cache_ڪاميابي) {
            داخلا = it->second;
        } وإلا {
            r_ctx->clear_caches_locked();
            خودڪار mutex = std::make_shared<std::mutex>();
            داخلا = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[ڪنجي] = داخلا;
        }
    } وإلا {
        خودڪار mutex = std::make_shared<std::mutex>();
        داخلا = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_ڪاميابي = خطأ;
    }

    std::lock_guard<std::mutex> تالو(*(داخلا->mutex));
    cache_ڪاميابي = cache_ڪاميابي && داخلا->ptr && r_ctx->infer_request_cache.count(ڪنجي) != 0 &&
                r_ctx->infer_request_cache_prefill.count(ڪنجي) != 0;

    إذا (cache_ڪاميابي) {
        ggml_decoder = داخلا->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_ڪاميابي = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_داخلا_نالا_local;
    std::vector<std::string> ov_output_نالا_local;

    إذا (cache_ڪاميابي) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->سيٽ_model_params(m_params);
        ggml_decoder->سيٽ_compute_params(c_params);
        إذا (old_m_params.kv_بفر_بدليو(m_params) || !ggml_decoder->is_bound_منزل(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->جوڙيو_extra_inputs();
        {
            std::lock_guard<std::mutex> map_تالو(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(ڪنجي) : r_ctx->infer_request_cache.at(ڪنجي);
            ov_داخلا_نالا_local = r_ctx->ov_داخلا_نالا_cache.at(ڪنجي);
            ov_output_نالا_local = r_ctx->ov_output_نالا_cache.at(ڪنجي);
        }

        decoder_پڇاڙي_وقت = ggml_وقت_us();
        conversion_پڇاڙي_وقت = decoder_پڇاڙي_وقت;
        compile_پڇاڙي_وقت = decoder_پڇاڙي_وقت;
    } وإلا {
        إذا (cache_enabled) {
            std::lock_guard<std::mutex> map_تالو(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(ڪنجي);
            r_ctx->infer_request_cache_prefill.erase(ڪنجي);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        خودڪار shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_تالو(shared_cache->mutex);
        خودڪار وزن_نالا = get_وزن_نالا(cgraph);
        خودڪار local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, وزن_نالا, is_static,
                                                             stateful, خطأ, is_prefill, prefill_chunk_سائيز);
        ثابت std::string shared_ڪنجي =
            cache_enabled ? compiled_گراف_ڪنجي(cgraph, *local_decoder, device, prefill_chunk_سائيز) : "";
        خودڪار shared_it = shared_cache->graphs.find(shared_ڪنجي);
        إذا (!shared_ڪنجي.خالي_جايون() && shared_it != shared_cache->graphs.end()) {
            خودڪار & compiled = shared_it->second;
            خودڪار prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            خودڪار decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            داخلا->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_داخلا_نالا_local = compiled.داخلا_نالا;
            ov_output_نالا_local = compiled.output_نالا;
            r_ctx->infer_request_cache_prefill[ڪنجي] = prefill_request;
            r_ctx->infer_request_cache[ڪنجي] = decode_request;
            r_ctx->ov_داخلا_نالا_cache[ڪنجي] = ov_داخلا_نالا_local;
            r_ctx->ov_output_نالا_cache[ڪنجي] = ov_output_نالا_local;
            decoder_پڇاڙي_وقت = conversion_پڇاڙي_وقت = compile_پڇاڙي_وقت = ggml_وقت_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } وإلا {
            std::shared_ptr<ov::Model> model;
            خودڪار model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph);

            خودڪار ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, خطأ, سچ, prefill_chunk_سائيز);
            خودڪار ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, خطأ, خطأ, prefill_chunk_سائيز);
            decoder_پڇاڙي_وقت = ggml_وقت_us();

            ثابت منطقي dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            ثابت خودڪار dump_ir_timestamp = static_cast<ڊگهو ڊگهو>(ggml_وقت_us());

            خودڪار build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          ثابت std::shared_ptr<GgmlOvDecoder> & decoder, ثابت حرف * ٽيگ,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_پڇاڙي_وقت, int64_t & local_compile_پڇاڙي_وقت) {
                خودڪار داخلا_model = std::make_shared<ov::frontend::ggml::داخلا_Model>(decoder);
                model = ov::frontend::ggml::Front_پڇاڙي::convert(داخلا_model);
                decoder->clear_model_weights();
                local_conversion_پڇاڙي_وقت = ggml_وقت_us();

                إذا (dump_ir) {
                    حرف timestamped_filename[64];
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%s_%lld.xml", ٽيگ,
                             dump_ir_timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                compiled_model = core.compile_model(model, device, compile_config);
                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                local_compile_پڇاڙي_وقت = ggml_وقت_us();
            };
            std::shared_ptr<ov::Model> model_prefill;
            std::shared_ptr<ov::Model> model_decode;
            ov::CompiledModel compiled_model_prefill;
            ov::CompiledModel compiled_model_decode;
            std::shared_ptr<ov::InferRequest> infer_request_prefill;
            std::shared_ptr<ov::InferRequest> infer_request_decode;
            int64_t prefill_conversion_پڇاڙي_وقت;
            int64_t decode_conversion_پڇاڙي_وقت;
            int64_t prefill_compile_پڇاڙي_وقت;
            int64_t decode_compile_پڇاڙي_وقت;
            build_static_model(ggml_decoder_prefill, "prefill", model_prefill, compiled_model_prefill,
                               infer_request_prefill, prefill_conversion_پڇاڙي_وقت, prefill_compile_پڇاڙي_وقت);
            إذا (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_پڇاڙي_وقت = prefill_conversion_پڇاڙي_وقت;
                decode_compile_پڇاڙي_وقت = prefill_compile_پڇاڙي_وقت;
            } وإلا {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_پڇاڙي_وقت, decode_compile_پڇاڙي_وقت);
            }
            conversion_پڇاڙي_وقت = std::max(prefill_conversion_پڇاڙي_وقت, decode_conversion_پڇاڙي_وقت);
            compile_پڇاڙي_وقت = std::max(prefill_compile_پڇاڙي_وقت, decode_compile_پڇاڙي_وقت);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            داخلا->ptr = ggml_decoder;

            لكل (ثابت خودڪار & ov_param : model->get_parameters()) {
                ov_داخلا_نالا_local.push_back(ov_param->get_friendly_name());
            }
            لكل (ثابت خودڪار & ov_output : model->get_results()) {
                ov_output_نالا_local.push_back(ov_output->get_friendly_name());
            }

            إذا (!shared_ڪنجي.خالي_جايون()) {
                shared_cache->graphs.emplace(
                    shared_ڪنجي, ov_compiled_گراف{compiled_model_decode, compiled_model_prefill, ov_داخلا_نالا_local,
                                                  ov_output_نالا_local});
            }

            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_تالو(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[ڪنجي] = infer_request_prefill;
                r_ctx->infer_request_cache[ڪنجي] = infer_request_decode;
                r_ctx->ov_داخلا_نالا_cache[ڪنجي] = ov_داخلا_نالا_local;
                r_ctx->ov_output_نالا_cache[ڪنجي] = ov_output_نالا_local;
            }
        }
    }

    reset_single_جڳهه_recurrent_cache(cgraph, m_params, c_params);

    إذا (is_prefill) {
        خودڪار inp_len = get_inp_pos_n_ٽوڪن(cgraph, inp_pos);
        لكل (صحيح chunk_اشاريو = 0; chunk_اشاريو * prefill_chunk_سائيز < inp_len; chunk_اشاريو++) {
            لكل (size_t i = 0; i < ov_داخلا_نالا_local.size(); i++) {
                ثابت خودڪار & param_name = ov_داخلا_نالا_local[i];
                خودڪار داخلا_tensor = get_ov_داخلا_tensor_static_prefill(ggml_decoder, param_name, chunk_اشاريو);
                infer_request->سيٽ_داخلا_tensor(i, داخلا_tensor);

                إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    ثابت خودڪار داخلا_tensor = infer_request->get_داخلا_tensor(i);
                    print_داخلا_tensor_info(param_name, داخلا_tensor);
                }
            }

            لكل (size_t i = 0; i < ov_output_نالا_local.size(); i++) {
                ثابت خودڪار & model_outputs = ggml_decoder->get_model_outputs();
                خودڪار model_output_it = model_outputs.find(ov_output_نالا_local[i]);
                إذا (model_output_it == model_outputs.end()) {
                    استمر;
                }
                خودڪار * ggml_tensor = model_output_it->second;
                إذا (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    استمر;
                }
                خودڪار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->سيٽ_output_tensor(i, output_tensor);
            }

            ov_raw_infer_شروعات = ggml_وقت_us();
            infer_request->infer();
            ov_raw_infer_ڪل += ggml_وقت_us() - ov_raw_infer_شروعات;

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                لكل (size_t i = 0; i < ov_output_نالا_local.size(); i++) {
                    ثابت خودڪار output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_نالا_local[i], output_tensor, output_tensor.معلومات());
                }
            }
        }
        infer_پڇاڙي_وقت = ggml_وقت_us();
    } وإلا {
        لكل (size_t i = 0; i < ov_داخلا_نالا_local.size(); i++) {
            ثابت خودڪار & param_name = ov_داخلا_نالا_local[i];
            خودڪار داخلا_tensor = get_ov_داخلا_tensor_static_decode(ggml_decoder, param_name);
            infer_request->سيٽ_داخلا_tensor(i, داخلا_tensor);

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                ثابت خودڪار داخلا_tensor = infer_request->get_داخلا_tensor(i);
                print_داخلا_tensor_info(param_name, داخلا_tensor);
            }
        }

        لكل (size_t i = 0; i < ov_output_نالا_local.size(); i++) {
            ثابت خودڪار & model_outputs = ggml_decoder->get_model_outputs();
            خودڪار model_output_it = model_outputs.find(ov_output_نالا_local[i]);
            إذا (model_output_it == model_outputs.end()) {
                استمر;
            }
            خودڪار * ggml_tensor = model_output_it->second;
            إذا (ggml_nbytes(ggml_tensor) == 0) {
                استمر;
            }
            خودڪار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->سيٽ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_شروعات = ggml_وقت_us();
        infer_request->infer();
        infer_پڇاڙي_وقت = ggml_وقت_us();
        ov_raw_infer_ڪل = infer_پڇاڙي_وقت - ov_raw_infer_شروعات;

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            لكل (size_t i = 0; i < ov_output_نالا_local.size(); i++) {
                ثابت خودڪار output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_نالا_local[i], output_tensor, output_tensor.معلومات());
            }
        }
    }

    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_پڇاڙي_وقت - شروعات_وقت) / 1000.0);
        إذا (!cache_ڪاميابي) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_پڇاڙي_وقت - decoder_پڇاڙي_وقت) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_پڇاڙي_وقت - conversion_پڇاڙي_وقت) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_پڇاڙي_وقت - compile_پڇاڙي_وقت) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_ڪل / 1000.0);
    }

    أرجع GGML_حالت_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
ساكن ggml_cgraph * get_compute_گراف(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    خودڪار is_skipped = [](ثابت ggml_tensor * ڳنڍ) {
        أرجع ڳنڍ->op != GGML_OP_NONE && !(ڳنڍ->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    إذا (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        أرجع cgraph;
    }
    خودڪار & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](ثابت ggml_tensor * ڳنڍ) { أرجع !is_skipped(ڳنڍ); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.معلومات();
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

        ثابت خودڪار is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != خالي_اشارو);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_تالو(r_ctx->execution_mutex);
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
        ggml_tensor * ڳنڍ = cgraph->nodes[i];
        صحيح use_ڳڻپ = cgraph->use_counts[ggml_hash_ڳوليو(&cgraph->گهميل_hash_سيٽ, ڳنڍ)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        إذا ((cgraph->n_nodes <= 1 && use_ڳڻپ == 0) ||
            (cgraph->n_nodes <= 1 && ڳنڍ->op == GGML_OP_VIEW && use_ڳڻپ == 1 && ڳنڍ->src[0] != خالي_اشارو &&
             ڳنڍ->src[0]->op == GGML_OP_NONE)) {
            أرجع خطأ;
        }
        إذا (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_ترتيب_بدلايو)) {
            أرجع خطأ;
        }
        صحيح داخلا_use_ڳڻپ = 0;
        لكل (صحيح j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_ڳنڍ = cgraph->nodes[j];
            لكل (صحيح k = 0; k < GGML_MAX_SRC; k++) {
                إذا (other_ڳنڍ->src[k] == ڳنڍ) {
                    داخلا_use_ڳڻپ++;
                }
            }
        }
        إذا (use_ڳڻپ != داخلا_use_ڳڻپ && ڳنڍ->op != GGML_OP_NONE) {
            أرجع سچ;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    إذا (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_وزن_نالا(cgraph);
    } وإلا {
        لكل (ثابت خودڪار & kv : GgmlOvDecoder::create_وزن_nodes(cgraph, سچ)) {
            model_weights.insert(kv.پهريون);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ڳنڍ = cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = ڳنڍ->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            إذا (src != خالي_اشارو && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.خالي_جايون() == خطأ &&
                model_leafs.find(src) == model_leafs.end()) {
                إذا (GgmlOvDecoder::is_inp_tok(src, ڳنڍ)) {
                    أرجع خطأ;
                }
                أرجع سچ;
            }
        }
    }
    أرجع خطأ;
}

منطقي is_naive(ggml_cgraph * cgraph) {
    constexpr صحيح naive_گراف_سائيز_حد = 20;
    صحيح count = 0;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        إذا (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    أرجع count < naive_گراف_سائيز_حد;
}

size_t checksum(ثابت فراغ * معلومات, size_t size) {
    ثابت uint8_t * بائيٽون = static_cast<ثابت uint8_t *>(معلومات);
    size_t جوڙ = 0;
    لكل (size_t i = 0; i < size; ++i) {
        جوڙ += (uint8_t) i;
        جوڙ += بائيٽون[i];
    }
    أرجع جوڙ;
}

منطقي save_ggml_tensor_معلومات_منزل_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_رستو) {
    إذا (tensor == خالي_اشارو || tensor->معلومات == خالي_اشارو) {
        أرجع خطأ;
    }

    std::ofstream ٻاهريون(file_رستو);
    إذا (!ٻاهريون.is_open()) {
        أرجع خطأ;
    }

    ثابت size_t n = ggml_nelements(tensor);
    ٻاهريون << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    بدل (tensor->type) {
    حالة GGML_TYPE_F32: {
        ثابت خودڪار * معلومات = static_cast<ثابت عائم *>(tensor->معلومات);
        لكل (size_t i = 0; i < n; ++i) {
            ٻاهريون << معلومات[i] << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_F16: {
        ثابت خودڪار * معلومات = static_cast<ثابت ggml_fp16_t *>(tensor->معلومات);
        لكل (size_t i = 0; i < n; ++i) {
            ٻاهريون << ggml_fp16_منزل_fp32(معلومات[i]) << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_BF16: {
        ثابت خودڪار * معلومات = static_cast<ثابت ggml_bf16_t *>(tensor->معلومات);
        لكل (size_t i = 0; i < n; ++i) {
            ٻاهريون << ggml_bf16_منزل_fp32(معلومات[i]) << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_I32: {
        ثابت خودڪار * معلومات = static_cast<ثابت int32_t *>(tensor->معلومات);
        لكل (size_t i = 0; i < n; ++i) {
            ٻاهريون << معلومات[i] << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_I64: {
        ثابت خودڪار * معلومات = static_cast<ثابت int64_t *>(tensor->معلومات);
        لكل (size_t i = 0; i < n; ++i) {
            ٻاهريون << معلومات[i] << '\n';
        }
        اكسر;
    }
    افتراضي:
        ٻاهريون << "unsupported tensor type for text dump" << '\n';
        أرجع خطأ;
    }

    أرجع سچ;
}

فراغ print_داخلا_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor) {
    std::ٻاهر << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.معلومات()
              << '\n';
    بدل (tensor.get_element_type()) {
    حالة ov::element::f32: {
        إذا (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::ٻاهر << *(tensor.معلومات<عائم>()) << '\n';
        } وإلا {
            size_t صفون = tensor.get_shape()[2];
            size_t کانن_ڳڻپ_2 = tensor.get_shape()[3];
            ثابت عائم * معلومات = tensor.معلومات<عائم>();
            لكل (size_t i = 0; i < صفون; ++i) {
                لكل (size_t j = 0; j < کانن_ڳڻپ_2; ++j) {
                    عائم val = معلومات[i * کانن_ڳڻپ_2 + j];
                    إذا (std::isinf(val) && val < 0) {
                        std::ٻاهر << std::setw(5) << "-inf";
                    } وإلا {
                        std::ٻاهر << std::setw(5) << val;
                    }
                }
                std::ٻاهر << '\n';
            }
        }

        اكسر;
    }
    حالة ov::element::f16:
        std::ٻاهر << *(tensor.معلومات<ov::float16>()) << '\n';
        اكسر;
    حالة ov::element::i32:
        لكل (size_t i = 0; i < tensor.get_سائيز(); ++i) {
            std::ٻاهر << tensor.معلومات<int32_t>()[i] << ' ';
        }
        std::ٻاهر << '\n';
        اكسر;
    حالة ov::element::i64:
        لكل (size_t i = 0; i < tensor.get_سائيز(); ++i) {
            std::ٻاهر << tensor.معلومات<int64_t>()[i] << ' ';
        }
        std::ٻاهر << '\n';
        اكسر;
    افتراضي:
        اكسر;
    }
}

فراغ print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت فراغ * output_dst) {
    std::ٻاهر << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    خودڪار print_float_stats = [](ثابت std::string & type_name, size_t size, خودڪار get_قدر) {
        إذا (size == 0) {
            أرجع;
        }

        عائم پهريون = get_قدر(0);
        عائم min = پهريون;
        عائم max = پهريون;
        مزدوج جوڙ = پهريون;

        لكل (size_t i = 1; i < size; ++i) {
            عائم v = get_قدر(i);
            min = std::min(v, min);
            max = std::max(v, max);
            جوڙ += v;
        }
        مزدوج سراسري_2 = جوڙ / size;

        std::ٻاهر << std::ساڄي << std::setw(6) << type_name << std::ساڄي << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::ٻاهر << std::ساڄي << std::setw(6) << "" << std::ساڄي << std::setw(12) << پهريون << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << سراسري_2 << '\n';
    };

    بدل (tensor.get_element_type()) {
    حالة ov::element::f32: {
        ثابت عائم * معلومات = tensor.معلومات<عائم>();
        size_t size = tensor.get_سائيز();
        print_float_stats("[f32]", size, [معلومات](size_t i) { أرجع معلومات[i]; });
        اكسر;
    }
    حالة ov::element::f16: {
        ثابت ov::float16 * معلومات = tensor.معلومات<ov::float16>();
        size_t size = tensor.get_سائيز();
        print_float_stats("[f16]", size, [معلومات](size_t i) { أرجع static_cast<عائم>(معلومات[i]); });
        اكسر;
    }
    افتراضي:
        اكسر;
    }
}

ثابت ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        خودڪار * op = cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; ++j) {
            خودڪار * src = op->src[j];
            إذا (src == خالي_اشارو) {
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

int64_t get_inp_pos_n_ٽوڪن(ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    صحيح n_planes = 1;
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        خودڪار * op = cgraph->nodes[i];
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
    أرجع get_inp_pos_n_ٽوڪن(cgraph, inp_pos) > 1;
}
