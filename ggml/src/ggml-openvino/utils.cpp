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
    ساكن ثابت منطقي kv_ټوټه_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    إذا (kv_ټوټه_disabled) {
        أرجع std::nullopt;
    }
    إذا (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        أرجع std::nullopt;
    }
    إذا (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != تش_اشاره) {
        أرجع std::nullopt;
    }
    ثابت خپلکاری * op = ggml_decoder->get_tensor_کارول_شوی_op(ggml_tensor);
    إذا (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        أرجع std::nullopt;
    }

    ثابت خپلکاری & compute_params = ggml_decoder->get_compute_params();
    إذا (compute_params.n_seq_فعال != 1 || compute_params.seq_فعال_پیل != 0) {
        أرجع std::nullopt;
    }

    صحيح layer;
    إذا (خپلکاری layer_opt = ایستل_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } وإلا {
        أرجع std::nullopt;
    }

    ثابت منطقي is_swa = ggml_decoder->is_swa_layer(layer);
    إذا (is_swa) {
        أرجع std::nullopt;
    }
    ثابت صحيح ctx_فی_seq = ggml_decoder->get_ctx_فی_seq();
    ثابت صحيح n_kv = compute_params.attention_اندازه;
    إذا (ctx_فی_seq <= 0 || n_kv <= 0 || n_kv >= ctx_فی_seq) {
        أرجع std::nullopt;
    }

    ov::Shape ډک_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    إذا (ډک_shape.size() != 4 || ډک_shape[0] != 1 || ډک_shape[1] != 1 ||
        static_cast<صحيح>(ډک_shape[2]) != ctx_فی_seq) {
        أرجع std::nullopt;
    }

    ov::Shape sliced_shape = ډک_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    إذا (ggml_openvino_بفر_is_remote(ggml_tensor)) {
        خپلکاری remote_context = ggml_openvino_get_remote_context();
        خپلکاری gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        أرجع gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->معلومات);
    }

    أرجع ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->معلومات);
}

ساكن uint64_t ggml_openvino_model_cache_extra_cfg(منطقي stateful, منطقي recurrent) {
    ثابت حرف * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    ثابت منطقي manual_gqa_enabled = manual_gqa_env != تش_اشاره ?
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

std::map<std::string, std::shared_ptr<ov::Node>> get_وزن_نومونه(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> نومونه;
    لكل (ثابت خپلکاری & name : GgmlOvDecoder::collect_وزن_نومونه(cgraph)) {
        نومونه[name] = تش_اشاره;
    }
    أرجع نومونه;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
ساكن std::string compiled_ګراف_کلی(ثابت ggml_cgraph * graph,
                                      ثابت GgmlOvDecoder & decoder,
                                      ثابت std::string & device,
                                      صحيح prefill_chunk_اندازه = 0,
                                      منطقي disk_cache = خطأ) {
    std::string کلی;
    خپلکاری append = [&کلی](ثابت خپلکاری & value) {
        کلی.append(reinterpret_cast<ثابت حرف *>(&value), حجم(value));
    };
    خپلکاری append_string = [&](ثابت std::string & value) {
        append(value.size());
        کلی.append(value);
    };
    append_string(device);
    append(decoder.is_static());
    append(decoder.is_stateful());
    append(prefill_chunk_اندازه);
    منطقي has_وزن_بفر_id = خطأ;
    std::unordered_map<ثابت ggml_tensor *, size_t> ids;
    std::function<فراغ(ثابت ggml_tensor *)> لیدنه = [&](ثابت ggml_tensor * tensor) {
        إذا (!tensor) {
            append(size_t(0));
            أرجع;
        }
        خپلکاری inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.لومړی->second);
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
        ثابت خپلکاری * بنسټ = tensor->view_src ? tensor->view_src : tensor;
        ثابت منطقي وزن = بنسټ->بفر && بنسټ->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS;
        append(وزن);
        إذا (وزن && !disk_cache) {
            ثابت size_t بفر_id = ggml_backend_openvino_بفر_get_ctx_id(بنسټ->بفر);
            has_وزن_بفر_id |= بفر_id != 0;
            append(بفر_id);
            append(tensor->معلومات);
        }
        لیدنه(tensor->view_src);
        لكل (ثابت خپلکاری * src : tensor->src) {
            لیدنه(src);
        }
    };
    append(graph->n_nodes);
    لكل (صحيح i = 0; i < graph->n_nodes; ++i) {
        لیدنه(graph->nodes[i]);
    }
    append(graph->n_leafs);
    لكل (صحيح i = 0; i < graph->n_leafs; ++i) {
        لیدنه(graph->leafs[i]);
    }
    لكل (ثابت خپلکاری & ننوت : decoder.get_model_extra_inputs()) {
        append_string(ننوت.لومړی);
        append_string(ننوت.second.type.get_type_name());
        append(ننوت.second.shape.size());
        لكل (خپلکاری dim : ننوت.second.shape) {
            append(dim);
        }
        append(ننوت.second.is_parameter);
        إذا (!ننوت.second.is_parameter) {
            append(ننوت.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    أرجع disk_cache || has_وزن_بفر_id ? کلی : std::string{};
}

ساكن std::string dynamic_ګراف_signature(ثابت ggml_cgraph * graph,
                                           ثابت GgmlOvDecoder & decoder,
                                           ثابت ModelParams & params) {
    std::string کلی = "dynamic-1";
    خپلکاری append = [&کلی](ثابت خپلکاری & value) {
        کلی.append(reinterpret_cast<ثابت حرف *>(&value), حجم(value));
    };
    خپلکاری append_string = [&](ثابت std::string & value) {
        append(value.size());
        کلی.append(value);
    };
    ګراف_کلی ګراف_id(graph, ریښتیا);
    append(ګراف_id.n_nodes);
    append(ګراف_id.n_leaves);
    append_string(ګراف_id.لومړی_غوټه_name);
    append_string(ګراف_id.وروستی_غوټه_name);
    لكل (ثابت خپلکاری & name : ګراف_id.ننوت_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    لكل (صحيح layer : params.swa_layers) {
        append(layer);
    }
    لكل (ثابت خپلکاری & [layer, heads] : params.n_heads_kv_فی_layer) {
        append(layer);
        append(heads);
    }
    لكل (ثابت خپلکاری & [name, ننوت] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(ننوت.type.get_type_name());
    }
    لكل (ثابت خپلکاری & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    لكل (ثابت خپلکاری & [name, ننوت] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(ننوت.type.get_type_name());
        append(ننوت.is_parameter);
        إذا (!ننوت.is_parameter) {
            append(ننوت.value);
        }
    }
    أرجع کلی;
}

ساكن منطقي compiled_model_matches_ګراف(ثابت ov::CompiledModel & model,
                                         ثابت std::vector<std::string> & inputs,
                                         ثابت std::vector<std::string> & outputs,
                                         ثابت GgmlOvDecoder & decoder) {
    إذا (model.inputs().size() != inputs.size() || model.outputs().size() != outputs.size()) {
        أرجع خطأ;
    }
    ثابت خپلکاری & ګراف_inputs = decoder.get_model_inputs();
    ثابت خپلکاری & extra_inputs = decoder.get_model_extra_inputs();
    ثابت خپلکاری & ګراف_outputs = decoder.get_model_outputs();
    لكل (size_t i = 0; i < inputs.size(); ++i) {
        ثابت خپلکاری & پورټ = model.ننوت(i);
        خپلکاری ګراف_it = ګراف_inputs.find(inputs[i]);
        إذا (ګراف_it != ګراف_inputs.end()) {
            إذا (پورټ.get_element_type() != ګراف_it->second.type ||
                !پورټ.get_partial_shape().compatible(ګراف_it->second.shape)) {
                أرجع خطأ;
            }
            استمر;
        }
        خپلکاری extra_it = extra_inputs.find(inputs[i]);
        إذا (extra_it == extra_inputs.end() || پورټ.get_element_type() != extra_it->second.type ||
            !پورټ.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
            أرجع خطأ;
        }
    }
    لكل (size_t i = 0; i < outputs.size(); ++i) {
        خپلکاری ګراف_it = ګراف_outputs.find(outputs[i]);
        إذا (ګراف_it == ګراف_outputs.end()) {
            إذا (outputs[i].compare(0, 8, "__debug_") != 0) {
                أرجع خطأ;
            }
            استمر;
        }
        ثابت خپلکاری & پورټ = model.output(i);
        إذا (پورټ.get_element_type() != GgmlOvDecoder::get_ov_type(ګراف_it->second) ||
            !پورټ.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(ګراف_it->second)))) {
            أرجع خطأ;
        }
    }
    أرجع ریښتیا;
}

ساكن std::string ov_profiling_csv_ډګر(ثابت std::string & value) {
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

    ساكن std::atomic<uint64_t> inference_شاخص{0};
    ثابت uint64_t شاخص = inference_شاخص.fetch_add(1);
    ثابت std::string path = "openvino_profile_" + std::to_string(شاخص) + ".csv";
    std::ofstream output(path, std::ios::trunc);
    إذا (!output.is_open()) {
        GGML_LOG_WARN("ggml-openvino: failed to write profiling data to %s\n", path.c_str());
        أرجع;
    }

    output << "status,real_time_us,cpu_time_us,start_time_us,node_type,node_name,exec_type\n";
    int64_t ټول_real_وخت = 0;
    int64_t ټول_cpu_وخت = 0;
    size_t executed_شمېر = 0;
    لكل (ثابت خپلکاری & info : infer_request.get_profiling_info()) {
        ثابت حرف * حالت = "NOT_RUN";
        إذا (info.حالت == ov::ProfilingInfo::حالت_2::EXECUTED) {
            حالت = "EXECUTED";
            ټول_real_وخت += info.real_وخت.count();
            ټول_cpu_وخت += info.cpu_وخت.count();
            executed_شمېر++;
        } وإلا إذا (info.حالت == ov::ProfilingInfo::حالت_2::OPTIMIZED_بهر) {
            حالت = "OPTIMIZED_OUT";
        }

        output << حالت << ',' << info.real_وخت.count() << ',' << info.cpu_وخت.count() << ','
               << info.پیل_وخت.count() << ',' << ov_profiling_csv_ډګر(info.node_type) << ','
               << ov_profiling_csv_ډګر(info.غوټه_name) << ',' << ov_profiling_csv_ډګر(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_شمېر, ټول_real_وخت / 1000.0, ټول_cpu_وخت / 1000.0);
}

ساكن فراغ reset_single_خونه_recurrent_cache(ggml_cgraph * cgraph,
                                              ثابت ModelParams & model_params,
                                              ثابت ComputeParams & compute_params) {
    إذا (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        أرجع;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_بفر_t> buffers;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        خپلکاری * غوټه = cgraph->nodes[i];
        إذا (غوټه->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(غوټه->view_src) &&
            غوټه->view_src->ne[1] == 1 && غوټه->view_src->بفر != تش_اشاره) {
            buffers.insert(غوټه->view_src->بفر);
        }
    }
    لكل (خپلکاری * بفر : buffers) {
        ggml_backend_بفر_clear(بفر, 0);
    }
}

ov::Tensor create_ov_output_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   ثابت std::shared_ptr<ov::InferRequest> & infer_request,
                                   صحيح output_شاخص,
                                   ثابت ggml_tensor * ggml_tensor) {
    إذا (خپلکاری sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        أرجع *sliced;
    }

    خپلکاری output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    فراغ * output_معلومات = ggml_tensor->معلومات;
    إذا (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_شاخص).get_shape();
    } وإلا {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        إذا (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != تش_اشاره &&
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
        ggml_tensor->extra != تش_اشاره) {
        خپلکاری * extra_بنسټ = static_cast<ggml_openvino_extra_بنسټ *>(ggml_tensor->extra);
        إذا (extra_بنسټ->type == ggml_openvino_extra_بنسټ::Type::TENSOR) {
            خپلکاری * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_بنسټ);
            إذا (tensor_extra->tensor != تش_اشاره && tensor_extra->tensor->get_element_type() == output_type &&
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
ov::Tensor kv_لیکې_منزل_seq_axis_2(ثابت ov::Tensor & kv_tensor, size_t n_heads_kv) {
    ثابت size_t لیکې = kv_tensor.get_shape()[2];
    ثابت size_t سر_اندازه = kv_tensor.get_shape()[3] / n_heads_kv;
    ثابت size_t elem = kv_tensor.get_element_type().size();
    ثابت size_t سر_بایټونه = سر_اندازه * elem;

    ov::Tensor بهر(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, لیکې, سر_اندازه});
    ثابت خپلکاری * src = static_cast<ثابت uint8_t *>(kv_tensor.معلومات());
    خپلکاری * dst = static_cast<uint8_t *>(بهر.معلومات());
    لكل (size_t s = 0; s < لیکې; s++) {
        لكل (size_t h = 0; h < n_heads_kv; h++) {
            حافظه_نقل(dst + (h * لیکې + s) * سر_بایټونه, src + (s * n_heads_kv + h) * سر_بایټونه, سر_بایټونه);
        }
    }
    أرجع بهر;
}

کينډۍ <ډول_نوم T> فراغ سېټ_zero_diagonal(std::vector<T> & ماتریکس, size_t لیکې, size_t ستون_شمېر_2, T zero_ارزښت = T{}) {
    لكل (size_t i = 0; i < لیکې; ++i) {
        size_t diag_ستون = std::min(i, ستون_شمېر_2 - 1);
        ماتریکس[i * ستون_شمېر_2 + diag_ستون] = zero_ارزښت;
    }
}

ov::Tensor make_contiguous_وېشل_ننوت_tensor(ثابت بنية ggml_tensor * ggml_tensor, ثابت ov::Shape & ننوت_shape) {
    ثابت size_t element_اندازه = ggml_type_اندازه(ggml_tensor->type);
    ثابت size_t بلاک_اندازه_3 = ggml_blck_اندازه(ggml_tensor->type);

    GGML_ASSERT(بلاک_اندازه_3 == 1 && "non-contiguous split inputs must be plain element types");

    ثابت بنية ggml_tensor * سرچینه_tensor = ggml_tensor->view_src != تش_اشاره ? ggml_tensor->view_src : ggml_tensor;
    ثابت size_t سرچینه_دننه_واټن = ggml_tensor->view_src != تش_اشاره ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> سرچینه_معلومات(ggml_nbytes(سرچینه_tensor));
    ggml_backend_tensor_get(سرچینه_tensor, سرچینه_معلومات.معلومات(), 0, سرچینه_معلومات.size());

    ov::Tensor ننوت_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ننوت_shape);
    خپلکاری * dst = static_cast<uint8_t *>(ننوت_tensor.معلومات());
    size_t dst_دننه_واټن = 0;

    لكل (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        لكل (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            لكل (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                لكل (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    ثابت size_t src_دننه_واټن = سرچینه_دننه_واټن + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::حافظه_نقل(dst + dst_دننه_واټن, سرچینه_معلومات.معلومات() + src_دننه_واټن, element_اندازه);
                    dst_دننه_واټن += element_اندازه;
                }
            }
        }
    }

    أرجع ننوت_tensor;
}

ov::Tensor convert_ggml_ننوت_منزل_ov(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & name) {
    ثابت خپلکاری * ggml_tensor = ggml_decoder->get_ننوت_ggml_tensor(name);

    إذا (خپلکاری sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        أرجع *sliced;
    }

    إذا (ggml_tensor->extra != تش_اشاره && !ggml_decoder->is_splited_model()) {
        خپلکاری * extra_بنسټ = static_cast<ggml_openvino_extra_بنسټ *>(ggml_tensor->extra);
        إذا (extra_بنسټ->type == ggml_openvino_extra_بنسټ::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            خپلکاری * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_بنسټ);
            أرجع *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    خپلکاری * ننوت_معلومات = ggml_tensor->معلومات;
    ov::Shape ننوت_shape;
    إذا (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        ننوت_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } وإلا {
        ننوت_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    إذا (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        أرجع make_contiguous_وېشل_ننوت_tensor(ggml_tensor, ننوت_shape);
    }

    خپلکاری ننوت_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ننوت_shape, ننوت_معلومات);
    أرجع ننوت_tensor;
}

ov::Tensor get_ov_ننوت_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & param_name) {
    ov::Tensor ننوت_tensor;
    خپلکاری extra_ننوت = ggml_decoder->get_model_extra_inputs().find(param_name);
    إذا (extra_ننوت != ggml_decoder->get_model_extra_inputs().end()) {
        ننوت_tensor = ov::Tensor(extra_ننوت->second.type, extra_ننوت->second.shape);
        *ننوت_tensor.معلومات<int64_t>() = extra_ننوت->second.value;
    } وإلا {
        ننوت_tensor = convert_ggml_ننوت_منزل_ov(ggml_decoder, param_name);
    }
    أرجع ننوت_tensor;
}

ov::Tensor get_ov_ننوت_tensor_static_decode(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             ثابت std::string & param_name) {
    // NPU decoding stage
    إذا (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        أرجع get_ov_ننوت_tensor(ggml_decoder, param_name);
    }
    ثابت خپلکاری * ggml_tensor = ggml_decoder->get_ننوت_ggml_tensor(param_name);
    ثابت خپلکاری * op = ggml_decoder->get_tensor_کارول_شوی_op(ggml_tensor);

    إذا (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        ثابت صحيح n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        أكد(ggml_tensor->ne[0] == n_planes);
        ov::Shape ننوت_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor ننوت_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ننوت_shape);
        std::حافظه_نقل(ننوت_tensor.معلومات(), ggml_tensor->معلومات, n_planes * ggml_type_اندازه(ggml_tensor->type));
        أرجع ننوت_tensor;
    }

    إذا (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape ننوت_shape = {1, 1, 1, 1};
        ov::Tensor ننوت_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ننوت_shape);
        int32_t inp_بهر_id = *((int32_t *) ggml_tensor->معلومات);
        أكد(ggml_tensor->ne[0] == 1);
        أكد(inp_بهر_id == 0);
        *ننوت_tensor.معلومات<int32_t>() = inp_بهر_id;
        أرجع ننوت_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_ماسک(ggml_tensor, op)) {
        size_t context_اندازه = ggml_decoder->get_ctx_اندازه();
        إذا (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_معلومات =
                pad_ننوت<ggml_fp16_t>(ggml_tensor, 1, context_اندازه, GGML_FP32_منزل_FP16(-INFINITY));
            ov::Tensor ننوت_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_اندازه});
            std::حافظه_نقل(ننوت_tensor.معلومات(), padded_معلومات.معلومات(), padded_معلومات.size() * حجم(ggml_fp16_t));
            أرجع ننوت_tensor;
        }

        std::vector<عائم> padded_معلومات = pad_ننوت<عائم>(ggml_tensor, 1, context_اندازه, -INFINITY);
        ov::Tensor ننوت_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_اندازه});
        خپلکاری * معلومات_ptr = ننوت_tensor.معلومات<عائم>();
        std::copy(padded_معلومات.begin(), padded_معلومات.begin() + context_اندازه, معلومات_ptr);
        أرجع ننوت_tensor;
    }

    أرجع get_ov_ننوت_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_ننوت_tensor_static_prefill(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              ثابت std::string & param_name,
                                              صحيح chunk_شاخص) {
    // NPU prompt processing stage
    ثابت size_t ننوت_len = ggml_decoder->get_ننوت_len();
    ثابت size_t chunk_اندازه = ggml_decoder->m_prefill_chunk_اندازه;
    ثابت size_t chunk_valid_اندازه = std::min(chunk_اندازه, ننوت_len - chunk_شاخص * chunk_اندازه);
    ثابت size_t chunk_pad_اندازه = chunk_اندازه - chunk_valid_اندازه;

    إذا (param_name == "chunk_valid_len") {
        ov::Tensor ننوت_tensor(ov::element::i64, ov::Shape{1});
        *ننوت_tensor.معلومات<int64_t>() = (int64_t) chunk_valid_اندازه;
        أرجع ننوت_tensor;
    }
    إذا (chunk_شاخص > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor ننوت_tensor(ov::element::i64, ov::Shape{1});
        *ننوت_tensor.معلومات<int64_t>() = 0;
        أرجع ننوت_tensor;
    }
    إذا (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        أرجع get_ov_ننوت_tensor(ggml_decoder, param_name);
    }
    ثابت خپلکاری * ggml_tensor = ggml_decoder->get_ننوت_ggml_tensor(param_name);
    ثابت خپلکاری * op = ggml_decoder->get_tensor_کارول_شوی_op(ggml_tensor);

    إذا (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        ثابت صحيح n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        ثابت size_t element_اندازه = ggml_type_اندازه(ggml_tensor->type);
        ov::Shape ننوت_shape = {1, 1, 1, (size_t) n_planes * chunk_اندازه};
        ov::Tensor ننوت_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ننوت_shape);
        لكل (صحيح p = 0; p < n_planes; p++) {
            ثابت حرف * src =
                (ثابت حرف *) ggml_tensor->معلومات + (p * ننوت_len + chunk_شاخص * chunk_اندازه) * element_اندازه;
            حرف * dst = (حرف *) ننوت_tensor.معلومات() + p * chunk_اندازه * element_اندازه;
            std::حافظه_نقل(dst, src, chunk_valid_اندازه * element_اندازه);
            إذا (chunk_pad_اندازه > 0) {
                إذا (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t وروستی_ارزښت = *((ثابت int32_t *) src + chunk_valid_اندازه - 1);
                    int32_t * بهر = (int32_t *) dst;
                    std::fill(بهر + chunk_valid_اندازه, بهر + chunk_اندازه, وروستی_ارزښت + 1);
                } وإلا إذا (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t وروستی_ارزښت = *((ثابت int64_t *) src + chunk_valid_اندازه - 1);
                    int64_t * بهر = (int64_t *) dst;
                    std::fill(بهر + chunk_valid_اندازه, بهر + chunk_اندازه, وروستی_ارزښت + 1);
                } وإلا {
                    ارم std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        أرجع ننوت_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape ننوت_shape = {1, 1, 1, chunk_اندازه};
        ov::Tensor ننوت_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ننوت_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_اندازه = ggml_type_اندازه(ggml_tensor->type);
        فراغ * ننوت_معلومات = (حرف *) ggml_tensor->معلومات + chunk_شاخص * chunk_اندازه * element_اندازه;
        std::حافظه_نقل(ننوت_tensor.معلومات(), ننوت_معلومات, chunk_valid_اندازه * element_اندازه);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        إذا (chunk_pad_اندازه > 0) {
            إذا (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t وروستی_ارزښت =
                    *((int32_t *) ggml_tensor->معلومات + (chunk_شاخص * chunk_اندازه + chunk_valid_اندازه - 1));
                int32_t * output_معلومات = ننوت_tensor.معلومات<int32_t>();
                std::fill(output_معلومات + chunk_valid_اندازه, output_معلومات + chunk_اندازه, وروستی_ارزښت + 1);
            } وإلا إذا (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t وروستی_ارزښت =
                    *((int64_t *) ggml_tensor->معلومات + (chunk_شاخص * chunk_اندازه + chunk_valid_اندازه - 1));
                int64_t * output_معلومات = ننوت_tensor.معلومات<int64_t>();
                std::fill(output_معلومات + chunk_valid_اندازه, output_معلومات + chunk_اندازه, وروستی_ارزښت + 1);
            } وإلا {
                ارم std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        أرجع ننوت_tensor;
    }

    إذا (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape ننوت_shape = {1, 1, 1, output_len};
        ov::Tensor ننوت_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), ننوت_shape);
        إذا (ggml_tensor->ne[0] == 0) {
            *ننوت_tensor.معلومات<int32_t>() = 0;
        } وإلا {
            خپلکاری * معلومات_addr = ننوت_tensor.معلومات<int32_t>();
            لكل (size_t i = 0; i < output_len; i++) {
                معلومات_addr[i] = ((int32_t *) ggml_tensor->معلومات)[i] % chunk_اندازه;
            }
        }
        أرجع ننوت_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_scale_لیکې(ggml_tensor, op)) {
        ov::Tensor ننوت_tensor(ov::element::f32, ov::Shape{1, 1, chunk_اندازه, 1});
        خپلکاری * dst = ننوت_tensor.معلومات<عائم>();
        ثابت خپلکاری * src = static_cast<ثابت عائم *>(ggml_tensor->معلومات) + chunk_شاخص * chunk_اندازه;
        std::copy(src, src + chunk_valid_اندازه, dst);
        std::fill(dst + chunk_valid_اندازه, dst + chunk_اندازه, 1.0f);
        أرجع ننوت_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_منځنۍ(ggml_tensor, op)) {
        ثابت size_t n_seqs = ggml_tensor->ne[1];
        ثابت size_t src_stride = ggml_tensor->ne[0];
        ثابت size_t copy_len = std::min<size_t>(chunk_valid_اندازه, src_stride - chunk_شاخص * chunk_اندازه);
        ov::Tensor ننوت_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_اندازه});
        خپلکاری * dst = ننوت_tensor.معلومات<عائم>();
        std::fill(dst, dst + n_seqs * chunk_اندازه, 0.0f);
        ثابت خپلکاری * src = static_cast<ثابت عائم *>(ggml_tensor->معلومات) + chunk_شاخص * chunk_اندازه;
        لكل (size_t s = 0; s < n_seqs; s++) {
            std::حافظه_نقل(dst + s * chunk_اندازه, src + s * src_stride, copy_len * حجم(عائم));
        }
        أرجع ننوت_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_ماسک(ggml_tensor, op)) {
        size_t ستون_شمېر_2 = ggml_tensor->ne[0];
        size_t لیکې = ggml_tensor->ne[1];
        size_t chunk_valid_لیکې = std::min(chunk_اندازه, لیکې - chunk_شاخص * chunk_اندازه);
        size_t context_اندازه = ggml_decoder->get_ctx_اندازه();
        إذا (ggml_tensor->type == GGML_TYPE_F16) {
            ثابت خپلکاری * ggml_معلومات =
                static_cast<ثابت ggml_fp16_t *>(ggml_tensor->معلومات) + chunk_شاخص * chunk_اندازه * ستون_شمېر_2;
            std::vector<ggml_fp16_t> padded_معلومات = pad_ننوت<ggml_fp16_t>(ggml_معلومات, chunk_valid_لیکې, ستون_شمېر_2, chunk_اندازه,
                                                                          context_اندازه, GGML_FP32_منزل_FP16(-INFINITY));
            سېټ_zero_diagonal(padded_معلومات, chunk_اندازه, context_اندازه, GGML_FP32_منزل_FP16(0.0f));
            ov::Tensor ننوت_tensor(ov::element::f16, ov::Shape{1, 1, chunk_اندازه, context_اندازه});
            std::حافظه_نقل(ننوت_tensor.معلومات(), padded_معلومات.معلومات(), padded_معلومات.size() * حجم(ggml_fp16_t));
            أرجع ننوت_tensor;
        }

        ثابت خپلکاری * ggml_معلومات = static_cast<ثابت عائم *>(ggml_tensor->معلومات) + chunk_شاخص * chunk_اندازه * ستون_شمېر_2;
        std::vector<عائم> padded_معلومات =
            pad_ننوت<عائم>(ggml_معلومات, chunk_valid_لیکې, ستون_شمېر_2, chunk_اندازه, context_اندازه, -INFINITY);
        سېټ_zero_diagonal(padded_معلومات, chunk_اندازه, context_اندازه);
        ov::Tensor ننوت_tensor(ov::element::f32, ov::Shape{1, 1, chunk_اندازه, context_اندازه});
        خپلکاری * معلومات_ptr = ننوت_tensor.معلومات<عائم>();
        std::copy(padded_معلومات.begin(), padded_معلومات.begin() + chunk_اندازه * context_اندازه, معلومات_ptr);
        أرجع ننوت_tensor;
    }

    أرجع get_ov_ننوت_tensor(ggml_decoder, param_name);
}

تعداد ggml_حالت naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               ثابت std::string & device,
                               ثابت ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    إذا (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        أرجع GGML_حالت_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_قلف(cache.mutex);
    منطقي naive = ریښتیا;
    خپلکاری model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph, naive);
    خپلکاری decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    خپلکاری ننوت_model = std::make_shared<ov::frontend::ggml::ننوت_Model>(decoder);
    خپلکاری model = ov::frontend::ggml::Front_پای_ټکی::convert(ننوت_model, naive);
    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    خپلکاری remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    إذا (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_ډېرپېښ.name()] = ov::hint::Execution_ډېرپېښ::PERFORMANCE;
    } وإلا {
        compile_config[ov::hint::execution_ډېرپېښ.name()] = ov::hint::Execution_ډېرپېښ::ACCURACY;
    }
    إذا (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } وإلا {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> ننوت_نومونه;
    std::vector<std::string> output_نومونه;
    لكل (ثابت خپلکاری & param : model->get_parameters()) {
        ننوت_نومونه.push_back(param->get_friendly_name());
    }
    لكل (ثابت خپلکاری & پایله : model->get_results()) {
        output_نومونه.push_back(پایله->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    ننوت_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_قلف.unlock();

    لكل (size_t i = 0; i < ننوت_نومونه.size(); i++) {
        ثابت خپلکاری & param_name = ننوت_نومونه[i];
        خپلکاری ننوت_tensor = get_ov_ننوت_tensor(decoder, param_name);
        infer_request->سېټ_ننوت_tensor(i, ننوت_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    لكل (size_t i = 0; i < output_نومونه.size(); i++) {
        خپلکاری output_tensor = infer_request->get_output_tensor(i);
        ثابت خپلکاری & model_outputs = decoder->get_model_outputs();
        خپلکاری model_output_it = model_outputs.find(output_نومونه[i]);
        إذا (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_نومونه[i], output_tensor, output_tensor.معلومات());
            }
            استمر;
        }
        خپلکاری * ggml_tensor = model_output_it->second;
        std::حافظه_نقل(ggml_tensor->معلومات, output_tensor.معلومات(), output_tensor.get_byte_اندازه());
    }
    أرجع GGML_حالت_SUCCESS;
}

تعداد ggml_حالت ov_ګراف_compute_dynamic(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    خپلکاری & core = ov_singleton_core();
    ثابت خپلکاری & config = ggml_openvino_get_compile_config();
    ثابت خپلکاری & device = r_ctx->device;
    ساكن خپلکاری is_static = خطأ;

    ساكن ثابت منطقي cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    خپلکاری پیل_وخت = ggml_وخت_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    ثابت int64_t cache_کلی_پیل_وخت = ggml_وخت_us();
    ګراف_کلی کلی(cgraph);
    ثابت int64_t cache_کلی_compute_وخت = ggml_وخت_us() - cache_کلی_پیل_وخت;
    int64_t cache_lookup_وخت = 0;
    منطقي کلی_کتل_شوی = خطأ;
    إذا (!cache_disabled) {
        ثابت int64_t cache_lookup_پیل_وخت = ggml_وخت_us();
        std::lock_guard<std::mutex> map_قلف(r_ctx->ctx_mutex);
        کلی_کتل_شوی = r_ctx->decoder_cache.find(کلی) != r_ctx->decoder_cache.end();
        cache_lookup_وخت += ggml_وخت_us() - cache_lookup_پیل_وخت;
    }

    ثابت int64_t model_وېشل_check_پیل_وخت = ggml_وخت_us();
    منطقي model_is_splitted = کلی_کتل_شوی ? خطأ : is_model_splitted(cgraph);
    ثابت int64_t model_وېشل_check_وخت = ggml_وخت_us() - model_وېشل_check_پیل_وخت;

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
    ثابت int64_t ګراف_param_compute_پیل_وخت = ggml_وخت_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    ثابت int64_t ګراف_param_compute_وخت = ggml_وخت_us() - ګراف_param_compute_پیل_وخت;

    ثابت منطقي cache_enabled = !model_is_splitted && !cache_disabled;
    ثابت منطقي stateful_recurrent = r_ctx->stateful && m_params.state_اندازه >= 0;
    ثابت منطقي stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    إذا (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_فعال > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        أرجع GGML_حالت_FAILED;
    }
    إذا (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        أرجع GGML_حالت_FAILED;
    }
    منطقي cache_موندنه = خطأ;

    int64_t decoder_پای_ټکی_وخت;
    int64_t conversion_پای_ټکی_وخت;
    int64_t compile_پای_ټکی_وخت;
    int64_t infer_پای_ټکی_وخت;
    int64_t ov_raw_infer_پیل;

    {
        std::shared_ptr<decoder_runtime_ctx> ثبت;
        ModelParams old_m_params;

        ثابت int64_t cache_lookup_پیل_وخت = ggml_وخت_us();
        إذا (cache_enabled) {
            std::lock_guard<std::mutex> map_قلف(r_ctx->ctx_mutex);
            خپلکاری it = r_ctx->decoder_cache.find(کلی);
            cache_موندنه = it != r_ctx->decoder_cache.end();
            إذا (cache_موندنه) {
                ثبت = it->second;
            } وإلا {
                r_ctx->clear_caches_locked();
                خپلکاری mutex = std::make_shared<std::mutex>();
                ثبت = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[کلی] = ثبت;
            }
        } وإلا {
            خپلکاری mutex = std::make_shared<std::mutex>();
            ثبت = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_موندنه = خطأ;
        }
        cache_lookup_وخت += ggml_وخت_us() - cache_lookup_پیل_وخت;

        std::lock_guard<std::mutex> قلف(*(ثبت->mutex));
        cache_موندنه = cache_موندنه && ثبت->ptr && r_ctx->infer_request_cache.count(کلی) != 0;

        إذا (cache_موندنه) {
            ggml_decoder = ثبت->ptr;
            old_m_params = ggml_decoder->get_model_params();
            إذا (!ggml_decoder->is_splited_model()) {
                cache_موندنه = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_ننوت_نومونه;
        std::vector<std::string> ov_output_نومونه;

        إذا (cache_موندنه) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->سېټ_compute_params(c_params);
            ggml_decoder->سېټ_model_params(m_params);
            إذا (old_m_params.kv_بفر_بدل_شوی(m_params) || !ggml_decoder->is_bound_منزل(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->جمع_کړه_extra_inputs();
            {
                std::lock_guard<std::mutex> map_قلف(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(کلی);
                ov_ننوت_نومونه = r_ctx->ov_ننوت_نومونه_cache.at(کلی);
                ov_output_نومونه = r_ctx->ov_output_نومونه_cache.at(کلی);
            }

            إذا (stateful_kv_only) {
                ثابت خپلکاری * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_معلومات = (int32_t *) inp_pos->معلومات;
                خپلکاری pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                إذا (pos_معلومات[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_اندازه = pos_shape[3];
                } وإلا إذا (r_ctx->stateful_kv_اندازه == static_cast<size_t>(pos_معلومات[0])) {
                    r_ctx->stateful_kv_اندازه += pos_shape[3];
                } وإلا {
                    ثابت size_t pos_begin = static_cast<size_t>(pos_معلومات[0]);
                    ثابت منطقي refill = pos_begin > r_ctx->stateful_kv_اندازه;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    إذا (refill && !ggml_decoder->get_model_params().swa_layers.تشه()) {
                        ثابت صحيح n_swa = ggml_decoder->get_compute_params().swa_کړکۍ;
                        إذا (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_اندازه, n_swa);
                            أرجع GGML_حالت_FAILED;
                        }
                    }

                    ثابت منطقي relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    خپلکاری حالتونه = infer_request->query_state();
                    لكل (خپلکاری state : حالتونه) {
                        خپلکاری state_tensor = state.get_state();
                        خپلکاری state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        إذا (خپلکاری it = r_ctx->kv_state_ننوت_name_map.find(state.get_name());
                            it != r_ctx->kv_state_ننوت_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        صحيح n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        إذا (خپلکاری layer = ایستل_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        ثابت منطقي relayout_this_state = relayout_enabled;
                        ثابت size_t seq_axis = relayout_this_state ? 2 : 1;
                        ثابت size_t سر_axis = seq_axis == 2 ? 1 : 2;

                        إذا (refill) {
                            إذا (state_name.تشه()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                أرجع GGML_حالت_FAILED;
                            }
                            خپلکاری kv_tensor = get_ov_ننوت_tensor(ggml_decoder, state_name);
                            إذا (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_لیکې_منزل_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } وإلا {
                                ov::Shape refill_shape(4);
                                refill_shape[0] = state_tensor_shape[0];
                                refill_shape[seq_axis] = kv_tensor.get_shape()[2];
                                refill_shape[سر_axis] = state_tensor_shape[سر_axis];
                                refill_shape[3] = state_tensor_shape[3];
                                kv_tensor.سېټ_shape(refill_shape);
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
                        state.سېټ_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_اندازه = pos_begin + pos_shape[3];
                }
            }

            decoder_پای_ټکی_وخت = ggml_وخت_us();
            conversion_پای_ټکی_وخت = decoder_پای_ټکی_وخت;
            compile_پای_ټکی_وخت = decoder_پای_ټکی_وخت;
        } وإلا {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            خپلکاری shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_قلف(shared_cache->mutex);
            خپلکاری وزن_نومونه = get_وزن_نومونه(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, وزن_نومونه,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ثابت std::string model_cache_dir = ggml_openvino_model_cache_dir();
            ثابت std::string exact_کلی = cache_enabled ? compiled_ګراف_کلی(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_لار, manifest_لار;
            إذا (!model_cache_dir.تشه() && !model_is_splitted) {
                ثابت uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/ریښتیا, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_ګراف_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/ریښتیا, m_params.rope_params, 16, extra_cfg,
                                                    compiled_ګراف_کلی(cgraph, *ggml_decoder, device, 0, ریښتیا));
                blob_لار = ggml_openvino_model_cache_blob_لار(model_cache_dir, model_fp);
                manifest_لار = ggml_openvino_model_cache_manifest_لار(model_cache_dir, model_fp);
            }
            std::string shared_کلی =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_کلی) : "";
            ov::CompiledModel shared_model;
            منطقي imported = خطأ;
            خپلکاری shared_it = shared_cache->graphs.find(shared_کلی);
            إذا (!shared_کلی.تشه() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_ګراف(shared_it->second.decode, shared_it->second.ننوت_نومونه,
                                                               shared_it->second.output_نومونه, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_ننوت_نومونه = shared_it->second.ننوت_نومونه;
                ov_output_نومونه = shared_it->second.output_نومونه;
                imported = ریښتیا;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } وإلا إذا (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_کلی = exact_کلی;
                shared_it = shared_cache->graphs.find(shared_کلی);
                إذا (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_ننوت_نومونه = shared_it->second.ننوت_نومونه;
                    ov_output_نومونه = shared_it->second.output_نومونه;
                    imported = ریښتیا;
                }
                model_fp = exact_fp;
                blob_لار = ggml_openvino_model_cache_blob_لار(model_cache_dir, model_fp);
                manifest_لار = ggml_openvino_model_cache_manifest_لار(model_cache_dir, model_fp);
            }
            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_قلف(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(کلی);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            إذا (!model_cache_dir.تشه()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_ډېرپېښ.name()] = ov::Cache_ډېرپېښ::OPTIMIZE_چټکتیا;
            }
            إذا (!imported && !model_cache_dir.تشه() && !model_is_splitted) {
                ثابت uint64_t preferred_fp = model_fp;
                منطقي dynamic_incompatible = preferred_fp == exact_fp;
                لكل (صحيح candidate = 0; candidate < 2; ++candidate) {
                    إذا (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        اكسر;
                    }
                    ثابت uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    ثابت std::string candidate_blob =
                        ggml_openvino_model_cache_blob_لار(model_cache_dir, candidate_fp);
                    ثابت std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_لار(model_cache_dir, candidate_fp);
                    std::ifstream blob_دننه(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    إذا (!blob_دننه.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        استمر;
                    }
                    ثابت int64_t import_پیل = ggml_وخت_us();
                    حاول {
                        ov::CompiledModel cm;
                        خپلکاری remote_context = ggml_openvino_get_remote_context();
                        إذا (remote_context.has_value()) {
                            cm = core.import_model(blob_دننه, remote_context.value(), mc_config);
                        } وإلا {
                            cm = core.import_model(blob_دننه, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        إذا (!saved_inputs.تشه() || !saved_outputs.تشه()) {
                            ov_ننوت_نومونه = std::move(saved_inputs);
                            ov_output_نومونه = std::move(saved_outputs);
                        } وإلا {
                            لكل (ثابت خپلکاری & p : cm.inputs()) {
                                ov_ننوت_نومونه.push_back(p.get_غوټه()->get_friendly_name());
                            }
                            لكل (ثابت خپلکاری & o : cm.outputs()) {
                                ov_output_نومونه.push_back(o.get_غوټه()->get_friendly_name());
                            }
                        }
                        إذا (!compiled_model_matches_ګراف(cm, ov_ننوت_نومونه, ov_output_نومونه, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_ننوت_نومونه.clear();
                            ov_output_نومونه.clear();
                            استمر;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        ثبت->ptr = ggml_decoder;
                        imported = ریښتیا;
                        model_fp = candidate_fp;
                        blob_لار = candidate_blob;
                        manifest_لار = candidate_manifest;
                        إذا (candidate_fp == exact_fp) {
                            shared_کلی = exact_کلی;
                            shared_it = shared_cache->graphs.find(shared_کلی);
                        }
                        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_وخت_us() - import_پیل) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } امسك (ثابت std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = خطأ;
                        ov_ننوت_نومونه.clear();
                        ov_output_نومونه.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                إذا (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_کلی = exact_کلی;
                    shared_it = shared_cache->graphs.find(shared_کلی);
                    blob_لار = ggml_openvino_model_cache_blob_لار(model_cache_dir, model_fp);
                    manifest_لار = ggml_openvino_model_cache_manifest_لار(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            إذا (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_لار.c_str());
                أرجع GGML_حالت_FAILED;
            }
            إذا (!imported && ggml_openvino_وزن_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                أرجع GGML_حالت_FAILED;
            }
            إذا (imported) {
                decoder_پای_ټکی_وخت = conversion_پای_ټکی_وخت = compile_پای_ټکی_وخت = ggml_وخت_us();
            } وإلا {
                خپلکاری model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_پای_ټکی_وخت = ggml_وخت_us();

                خپلکاری ننوت_model = std::make_shared<ov::frontend::ggml::ننوت_Model>(ggml_decoder);
                model = ov::frontend::ggml::Front_پای_ټکی::convert(ننوت_model);
                ggml_decoder->clear_model_weights();
                conversion_پای_ټکی_وخت = ggml_وخت_us();

                إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    حرف timestamped_filename[64];
                    خپلکاری timestamp = (اوږد اوږد) ggml_وخت_us();
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                ثابت ov::AnyMap & compile_config = model_cache_dir.تشه() ? config : mc_config;
                ov::CompiledModel compiled_model;
                خپلکاری remote_context = ggml_openvino_get_remote_context();
                إذا (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } وإلا {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_پای_ټکی_وخت = ggml_وخت_us();

                لكل (ثابت خپلکاری & ov_param : model->get_parameters()) {
                    ov_ننوت_نومونه.push_back(ov_param->get_friendly_name());
                }
                لكل (ثابت خپلکاری & ov_output : model->get_results()) {
                    ov_output_نومونه.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                إذا (!model_cache_dir.تشه() && !model_is_splitted && model_fp != 0) {
                    حاول {
                        ثابت std::string blob_tmp = ggml_openvino_model_cache_لنډمهاله_لار(blob_لار);
                        ثابت std::string manifest_tmp = ggml_openvino_model_cache_لنډمهاله_لار(manifest_لار);
                        إذا (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_ننوت_نومونه,
                                                                     ov_output_نومونه)) {
                            std::ofstream blob_بهر(blob_tmp, std::ios::binary | std::ios::trunc);
                            إذا (blob_بهر.is_open()) {
                                compiled_model.export_model(blob_بهر);
                                blob_بهر.close();
                                إذا (blob_بهر.good()) {
                                    إذا (std::rename(blob_tmp.c_str(), blob_لار.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_لار.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_لار.c_str());
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
                ثبت->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            ثبت->ptr = ggml_decoder;
            إذا (!shared_کلی.تشه() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_کلی,
                                             ov_compiled_ګراف{shared_model, {}, ov_ننوت_نومونه, ov_output_نومونه});
            }
            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_قلف(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[کلی] = infer_request;
                r_ctx->ov_ننوت_نومونه_cache[کلی] = ov_ننوت_نومونه;
                r_ctx->ov_output_نومونه_cache[کلی] = ov_output_نومونه;
            }

            إذا (stateful_kv_only && cache_enabled) {
                ثابت خپلکاری * inp_pos = get_inp_pos_tensor(cgraph);
                خپلکاری pos_shape = GgmlOvDecoder::get_shape(inp_pos);
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
                r_ctx->stateful_kv_اندازه = pos_shape[3];
                ثابت خپلکاری kv_param_res_نومونه = ggml_decoder->get_kv_param_res_نومونه();
                لكل (ثابت خپلکاری & pair : kv_param_res_نومونه) {
                    r_ctx->kv_state_ننوت_name_map[pair.لومړی + pair.second] = pair.لومړی;
                }
            }
        }

        إذا (stateful_recurrent) {
            ثابت خپلکاری * inp_pos = get_inp_pos_tensor(cgraph);
            ثابت int32_t pos_begin = static_cast<ثابت int32_t *>(inp_pos->معلومات)[0];
            إذا (pos_begin == 0) {
                infer_request->reset_state();
            } وإلا إذا (!cache_موندنه || old_m_params.kv_بفر_بدل_شوی(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_اندازه) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                أرجع GGML_حالت_FAILED;
            }
        } وإلا {
            reset_single_خونه_recurrent_cache(cgraph, m_params, c_params);
        }

        لكل (size_t i = 0; i < ov_ننوت_نومونه.size(); i++) {
            ثابت خپلکاری & param_name = ov_ننوت_نومونه[i];
            خپلکاری ننوت_tensor = get_ov_ننوت_tensor(ggml_decoder, param_name);
            infer_request->سېټ_ننوت_tensor(i, ننوت_tensor);

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_ننوت_tensor_info(param_name, ننوت_tensor);
            }
        }

        لكل (size_t i = 0; i < ov_output_نومونه.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            ثابت خپلکاری & model_outputs = ggml_decoder->get_model_outputs();
            خپلکاری model_output_it = model_outputs.find(ov_output_نومونه[i]);
            إذا (model_output_it == model_outputs.end()) {
                استمر;
            }
            خپلکاری * ggml_tensor = model_output_it->second;
            إذا (ggml_nbytes(ggml_tensor) == 0) {
                استمر;
            }
            خپلکاری output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->سېټ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_پیل = ggml_وخت_us();
        infer_request->infer();
        infer_پای_ټکی_وخت = ggml_وخت_us();
        إذا (stateful_recurrent) {
            ثابت خپلکاری * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_اندازه =
                static_cast<ثابت int32_t *>(inp_pos->معلومات)[0] + get_inp_pos_n_نښې(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            لكل (size_t i = 0; i < ov_output_نومونه.size(); i++) {
                ثابت خپلکاری output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_نومونه[i], output_tensor, output_tensor.معلومات());
            }
        }

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_پای_ټکی_وخت - پیل_وخت) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_کلی_compute_وخت / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_وېشل_check_وخت / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", ګراف_param_compute_وخت / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_وخت / 1000.0);
            إذا (!cache_موندنه) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_پای_ټکی_وخت - decoder_پای_ټکی_وخت) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_پای_ټکی_وخت - conversion_پای_ټکی_وخت) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_پای_ټکی_وخت - compile_پای_ټکی_وخت) / 1000.0);
            GGML_LOG_INFO("    - OV raw infer time: %.3f ms \n", (infer_پای_ټکی_وخت - ov_raw_infer_پیل) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. Release only while holding the compilation mutex so another context cannot
    // be reading host weights during conversion/compilation. Pin the shared compiled
    // models across backend teardown; a later context can create its own request without
    // reading the dropped pages. A new, uncached graph still fails fast above.
    إذا (cache_موندنه && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_قلف(r_ctx->compiled_cache->mutex);
        إذا (!ggml_openvino_وزن_buffers_released()) {
            ggml_openvino_release_وزن_buffers();
        }
    }

    أرجع GGML_حالت_SUCCESS;
}

ov::AnyMap without_npuw(ثابت ov::AnyMap & config) {
    ov::AnyMap بهر;
    لكل (ثابت خپلکاری & kv : config) {
        إذا (kv.لومړی.rfind("NPUW", 0) == 0 || kv.لومړی == "NPU_USE_NPUW") {
            استمر;
        }
        بهر.insert(kv);
    }
    أرجع بهر;
}

تعداد ggml_حالت ov_ګراف_compute_static(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    خپلکاری & core = ov_singleton_core();

    خپلکاری get_prefill_chunk_اندازه = [] {
        ساكن ثابت صحيح chunk_اندازه = []() {
            صحيح env_prefill_chunk_اندازه = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            أرجع env_prefill_chunk_اندازه > 0 ? env_prefill_chunk_اندازه : 256;
        }();
        أرجع chunk_اندازه;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    ساكن std::string device = ggml_openvino_get_device_name();
    ساكن خپلکاری is_static = ریښتیا;
    ساكن خپلکاری stateful = خطأ;

    خپلکاری prefill_chunk_اندازه = get_prefill_chunk_اندازه();
    ثابت خپلکاری & config = ggml_openvino_get_compile_config();

    إذا (is_naive(cgraph)) {
        أرجع naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    خپلکاری پیل_وخت = ggml_وخت_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    ثابت خپلکاری * inp_pos = get_inp_pos_tensor(cgraph);
    ثابت منطقي no_kv_cache = m_params.is_cacheless_attn;
    ثابت خپلکاری is_prefill = no_kv_cache ? ریښتیا : get_is_prefill(cgraph, inp_pos);
    ثابت ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    إذا (m_params.n_heads_kv == -1) {
        prefill_chunk_اندازه = inp_pos->ne[0];
    }
    ګراف_کلی کلی(cgraph);
    ساكن ثابت منطقي cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    منطقي cache_موندنه = خطأ;

    int64_t decoder_پای_ټکی_وخت;
    int64_t conversion_پای_ټکی_وخت;
    int64_t compile_پای_ټکی_وخت;
    int64_t infer_پای_ټکی_وخت;
    int64_t ov_raw_infer_پیل;
    int64_t ov_raw_infer_ټول = 0;

    std::shared_ptr<decoder_runtime_ctx> ثبت;
    ModelParams old_m_params;

    إذا (cache_enabled) {
        std::lock_guard<std::mutex> map_قلف(r_ctx->ctx_mutex);
        خپلکاری it = r_ctx->decoder_cache.find(کلی);
        cache_موندنه = it != r_ctx->decoder_cache.end();
        إذا (cache_موندنه) {
            ثبت = it->second;
        } وإلا {
            r_ctx->clear_caches_locked();
            خپلکاری mutex = std::make_shared<std::mutex>();
            ثبت = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[کلی] = ثبت;
        }
    } وإلا {
        خپلکاری mutex = std::make_shared<std::mutex>();
        ثبت = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_موندنه = خطأ;
    }

    std::lock_guard<std::mutex> قلف(*(ثبت->mutex));
    cache_موندنه = cache_موندنه && ثبت->ptr && r_ctx->infer_request_cache.count(کلی) != 0 &&
                r_ctx->infer_request_cache_prefill.count(کلی) != 0;

    إذا (cache_موندنه) {
        ggml_decoder = ثبت->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_موندنه = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_ننوت_نومونه_local;
    std::vector<std::string> ov_output_نومونه_local;

    إذا (cache_موندنه) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->سېټ_model_params(m_params);
        ggml_decoder->سېټ_compute_params(c_params);
        إذا (old_m_params.kv_بفر_بدل_شوی(m_params) || !ggml_decoder->is_bound_منزل(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->جمع_کړه_extra_inputs();
        {
            std::lock_guard<std::mutex> map_قلف(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(کلی) : r_ctx->infer_request_cache.at(کلی);
            ov_ننوت_نومونه_local = r_ctx->ov_ننوت_نومونه_cache.at(کلی);
            ov_output_نومونه_local = r_ctx->ov_output_نومونه_cache.at(کلی);
        }

        decoder_پای_ټکی_وخت = ggml_وخت_us();
        conversion_پای_ټکی_وخت = decoder_پای_ټکی_وخت;
        compile_پای_ټکی_وخت = decoder_پای_ټکی_وخت;
    } وإلا {
        إذا (cache_enabled) {
            std::lock_guard<std::mutex> map_قلف(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(کلی);
            r_ctx->infer_request_cache_prefill.erase(کلی);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        خپلکاری shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_قلف(shared_cache->mutex);
        خپلکاری وزن_نومونه = get_وزن_نومونه(cgraph);
        خپلکاری local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, وزن_نومونه, is_static,
                                                             stateful, خطأ, is_prefill, prefill_chunk_اندازه);
        ثابت std::string shared_کلی =
            cache_enabled ? compiled_ګراف_کلی(cgraph, *local_decoder, device, prefill_chunk_اندازه) : "";
        خپلکاری shared_it = shared_cache->graphs.find(shared_کلی);
        إذا (!shared_کلی.تشه() && shared_it != shared_cache->graphs.end()) {
            خپلکاری & compiled = shared_it->second;
            خپلکاری prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            خپلکاری decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            ثبت->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_ننوت_نومونه_local = compiled.ننوت_نومونه;
            ov_output_نومونه_local = compiled.output_نومونه;
            r_ctx->infer_request_cache_prefill[کلی] = prefill_request;
            r_ctx->infer_request_cache[کلی] = decode_request;
            r_ctx->ov_ننوت_نومونه_cache[کلی] = ov_ننوت_نومونه_local;
            r_ctx->ov_output_نومونه_cache[کلی] = ov_output_نومونه_local;
            decoder_پای_ټکی_وخت = conversion_پای_ټکی_وخت = compile_پای_ټکی_وخت = ggml_وخت_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } وإلا {
            std::shared_ptr<ov::Model> model;
            خپلکاری model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph);

            خپلکاری ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, خطأ, ریښتیا, prefill_chunk_اندازه);
            خپلکاری ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, خطأ, خطأ, prefill_chunk_اندازه);
            decoder_پای_ټکی_وخت = ggml_وخت_us();

            ثابت منطقي dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            ثابت خپلکاری dump_ir_timestamp = static_cast<اوږد اوږد>(ggml_وخت_us());

            خپلکاری build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          ثابت std::shared_ptr<GgmlOvDecoder> & decoder, ثابت حرف * نښان,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_پای_ټکی_وخت, int64_t & local_compile_پای_ټکی_وخت) {
                خپلکاری ننوت_model = std::make_shared<ov::frontend::ggml::ننوت_Model>(decoder);
                model = ov::frontend::ggml::Front_پای_ټکی::convert(ننوت_model);
                decoder->clear_model_weights();
                local_conversion_پای_ټکی_وخت = ggml_وخت_us();

                إذا (dump_ir) {
                    حرف timestamped_filename[64];
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%s_%lld.xml", نښان,
                             dump_ir_timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                compiled_model = core.compile_model(model, device, compile_config);
                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                local_compile_پای_ټکی_وخت = ggml_وخت_us();
            };
            std::shared_ptr<ov::Model> model_prefill;
            std::shared_ptr<ov::Model> model_decode;
            ov::CompiledModel compiled_model_prefill;
            ov::CompiledModel compiled_model_decode;
            std::shared_ptr<ov::InferRequest> infer_request_prefill;
            std::shared_ptr<ov::InferRequest> infer_request_decode;
            int64_t prefill_conversion_پای_ټکی_وخت;
            int64_t decode_conversion_پای_ټکی_وخت;
            int64_t prefill_compile_پای_ټکی_وخت;
            int64_t decode_compile_پای_ټکی_وخت;
            build_static_model(ggml_decoder_prefill, "prefill", model_prefill, compiled_model_prefill,
                               infer_request_prefill, prefill_conversion_پای_ټکی_وخت, prefill_compile_پای_ټکی_وخت);
            إذا (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_پای_ټکی_وخت = prefill_conversion_پای_ټکی_وخت;
                decode_compile_پای_ټکی_وخت = prefill_compile_پای_ټکی_وخت;
            } وإلا {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_پای_ټکی_وخت, decode_compile_پای_ټکی_وخت);
            }
            conversion_پای_ټکی_وخت = std::max(prefill_conversion_پای_ټکی_وخت, decode_conversion_پای_ټکی_وخت);
            compile_پای_ټکی_وخت = std::max(prefill_compile_پای_ټکی_وخت, decode_compile_پای_ټکی_وخت);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            ثبت->ptr = ggml_decoder;

            لكل (ثابت خپلکاری & ov_param : model->get_parameters()) {
                ov_ننوت_نومونه_local.push_back(ov_param->get_friendly_name());
            }
            لكل (ثابت خپلکاری & ov_output : model->get_results()) {
                ov_output_نومونه_local.push_back(ov_output->get_friendly_name());
            }

            إذا (!shared_کلی.تشه()) {
                shared_cache->graphs.emplace(
                    shared_کلی, ov_compiled_ګراف{compiled_model_decode, compiled_model_prefill, ov_ننوت_نومونه_local,
                                                  ov_output_نومونه_local});
            }

            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_قلف(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[کلی] = infer_request_prefill;
                r_ctx->infer_request_cache[کلی] = infer_request_decode;
                r_ctx->ov_ننوت_نومونه_cache[کلی] = ov_ننوت_نومونه_local;
                r_ctx->ov_output_نومونه_cache[کلی] = ov_output_نومونه_local;
            }
        }
    }

    reset_single_خونه_recurrent_cache(cgraph, m_params, c_params);

    إذا (is_prefill) {
        خپلکاری inp_len = get_inp_pos_n_نښې(cgraph, inp_pos);
        لكل (صحيح chunk_شاخص = 0; chunk_شاخص * prefill_chunk_اندازه < inp_len; chunk_شاخص++) {
            لكل (size_t i = 0; i < ov_ننوت_نومونه_local.size(); i++) {
                ثابت خپلکاری & param_name = ov_ننوت_نومونه_local[i];
                خپلکاری ننوت_tensor = get_ov_ننوت_tensor_static_prefill(ggml_decoder, param_name, chunk_شاخص);
                infer_request->سېټ_ننوت_tensor(i, ننوت_tensor);

                إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    ثابت خپلکاری ننوت_tensor = infer_request->get_ننوت_tensor(i);
                    print_ننوت_tensor_info(param_name, ننوت_tensor);
                }
            }

            لكل (size_t i = 0; i < ov_output_نومونه_local.size(); i++) {
                ثابت خپلکاری & model_outputs = ggml_decoder->get_model_outputs();
                خپلکاری model_output_it = model_outputs.find(ov_output_نومونه_local[i]);
                إذا (model_output_it == model_outputs.end()) {
                    استمر;
                }
                خپلکاری * ggml_tensor = model_output_it->second;
                إذا (ggml_nbytes(ggml_tensor) == 0) {
                    // Zero-row in-place writeback (e.g. the empty s_copy defrag remainder). The OV
                    // Result is the full cache, so binding it over this 0-byte buffer overflows it.
                    استمر;
                }
                خپلکاری output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->سېټ_output_tensor(i, output_tensor);
            }

            ov_raw_infer_پیل = ggml_وخت_us();
            infer_request->infer();
            ov_raw_infer_ټول += ggml_وخت_us() - ov_raw_infer_پیل;

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                لكل (size_t i = 0; i < ov_output_نومونه_local.size(); i++) {
                    ثابت خپلکاری output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_نومونه_local[i], output_tensor, output_tensor.معلومات());
                }
            }
        }
        infer_پای_ټکی_وخت = ggml_وخت_us();
    } وإلا {
        لكل (size_t i = 0; i < ov_ننوت_نومونه_local.size(); i++) {
            ثابت خپلکاری & param_name = ov_ننوت_نومونه_local[i];
            خپلکاری ننوت_tensor = get_ov_ننوت_tensor_static_decode(ggml_decoder, param_name);
            infer_request->سېټ_ننوت_tensor(i, ننوت_tensor);

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                ثابت خپلکاری ننوت_tensor = infer_request->get_ننوت_tensor(i);
                print_ننوت_tensor_info(param_name, ننوت_tensor);
            }
        }

        لكل (size_t i = 0; i < ov_output_نومونه_local.size(); i++) {
            ثابت خپلکاری & model_outputs = ggml_decoder->get_model_outputs();
            خپلکاری model_output_it = model_outputs.find(ov_output_نومونه_local[i]);
            إذا (model_output_it == model_outputs.end()) {
                استمر;
            }
            خپلکاری * ggml_tensor = model_output_it->second;
            إذا (ggml_nbytes(ggml_tensor) == 0) {
                استمر;
            }
            خپلکاری output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->سېټ_output_tensor(i, output_tensor);
        }

        ov_raw_infer_پیل = ggml_وخت_us();
        infer_request->infer();
        infer_پای_ټکی_وخت = ggml_وخت_us();
        ov_raw_infer_ټول = infer_پای_ټکی_وخت - ov_raw_infer_پیل;

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            لكل (size_t i = 0; i < ov_output_نومونه_local.size(); i++) {
                ثابت خپلکاری output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_نومونه_local[i], output_tensor, output_tensor.معلومات());
            }
        }
    }

    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_پای_ټکی_وخت - پیل_وخت) / 1000.0);
        إذا (!cache_موندنه) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_پای_ټکی_وخت - decoder_پای_ټکی_وخت) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_پای_ټکی_وخت - conversion_پای_ټکی_وخت) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_پای_ټکی_وخت - compile_پای_ټکی_وخت) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_ټول / 1000.0);
    }

    أرجع GGML_حالت_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
ساكن ggml_cgraph * get_compute_ګراف(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    خپلکاری is_skipped = [](ثابت ggml_tensor * غوټه) {
        أرجع غوټه->op != GGML_OP_NONE && !(غوټه->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    إذا (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        أرجع cgraph;
    }
    خپلکاری & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](ثابت ggml_tensor * غوټه) { أرجع !is_skipped(غوټه); });
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
تعداد ggml_حالت ov_ګراف_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    حاول {
        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        ثابت خپلکاری is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != تش_اشاره);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_قلف(r_ctx->execution_mutex);
        cgraph = get_compute_ګراف(cgraph, *r_ctx);

        أرجع is_static ? ov_ګراف_compute_static(cgraph, r_ctx) : ov_ګراف_compute_dynamic(cgraph, r_ctx);
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
        ggml_tensor * غوټه = cgraph->nodes[i];
        صحيح use_شمېر = cgraph->use_counts[ggml_hash_پیدا_کړه(&cgraph->لیدل_شوی_hash_سېټ, غوټه)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        إذا ((cgraph->n_nodes <= 1 && use_شمېر == 0) ||
            (cgraph->n_nodes <= 1 && غوټه->op == GGML_OP_VIEW && use_شمېر == 1 && غوټه->src[0] != تش_اشاره &&
             غوټه->src[0]->op == GGML_OP_NONE)) {
            أرجع خطأ;
        }
        إذا (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_ترتیب_بدلول)) {
            أرجع خطأ;
        }
        صحيح ننوت_use_شمېر = 0;
        لكل (صحيح j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_غوټه = cgraph->nodes[j];
            لكل (صحيح k = 0; k < GGML_MAX_SRC; k++) {
                إذا (other_غوټه->src[k] == غوټه) {
                    ننوت_use_شمېر++;
                }
            }
        }
        إذا (use_شمېر != ننوت_use_شمېر && غوټه->op != GGML_OP_NONE) {
            أرجع ریښتیا;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    إذا (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_وزن_نومونه(cgraph);
    } وإلا {
        لكل (ثابت خپلکاری & kv : GgmlOvDecoder::create_وزن_nodes(cgraph, ریښتیا)) {
            model_weights.insert(kv.لومړی);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * غوټه = cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = غوټه->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            إذا (src != تش_اشاره && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.تشه() == خطأ &&
                model_leafs.find(src) == model_leafs.end()) {
                إذا (GgmlOvDecoder::is_inp_tok(src, غوټه)) {
                    أرجع خطأ;
                }
                أرجع ریښتیا;
            }
        }
    }
    أرجع خطأ;
}

منطقي is_naive(ggml_cgraph * cgraph) {
    constexpr صحيح naive_ګراف_اندازه_پوله = 20;
    صحيح count = 0;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        إذا (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    أرجع count < naive_ګراف_اندازه_پوله;
}

size_t checksum(ثابت فراغ * معلومات, size_t size) {
    ثابت uint8_t * بایټونه = static_cast<ثابت uint8_t *>(معلومات);
    size_t مجموعه = 0;
    لكل (size_t i = 0; i < size; ++i) {
        مجموعه += (uint8_t) i;
        مجموعه += بایټونه[i];
    }
    أرجع مجموعه;
}

منطقي save_ggml_tensor_معلومات_منزل_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_لار) {
    إذا (tensor == تش_اشاره || tensor->معلومات == تش_اشاره) {
        أرجع خطأ;
    }

    std::ofstream بهر(file_لار);
    إذا (!بهر.is_open()) {
        أرجع خطأ;
    }

    ثابت size_t n = ggml_nelements(tensor);
    بهر << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    بدل (tensor->type) {
    حالة GGML_TYPE_F32: {
        ثابت خپلکاری * معلومات = static_cast<ثابت عائم *>(tensor->معلومات);
        لكل (size_t i = 0; i < n; ++i) {
            بهر << معلومات[i] << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_F16: {
        ثابت خپلکاری * معلومات = static_cast<ثابت ggml_fp16_t *>(tensor->معلومات);
        لكل (size_t i = 0; i < n; ++i) {
            بهر << ggml_fp16_منزل_fp32(معلومات[i]) << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_BF16: {
        ثابت خپلکاری * معلومات = static_cast<ثابت ggml_bf16_t *>(tensor->معلومات);
        لكل (size_t i = 0; i < n; ++i) {
            بهر << ggml_bf16_منزل_fp32(معلومات[i]) << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_I32: {
        ثابت خپلکاری * معلومات = static_cast<ثابت int32_t *>(tensor->معلومات);
        لكل (size_t i = 0; i < n; ++i) {
            بهر << معلومات[i] << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_I64: {
        ثابت خپلکاری * معلومات = static_cast<ثابت int64_t *>(tensor->معلومات);
        لكل (size_t i = 0; i < n; ++i) {
            بهر << معلومات[i] << '\n';
        }
        اكسر;
    }
    افتراضي:
        بهر << "unsupported tensor type for text dump" << '\n';
        أرجع خطأ;
    }

    أرجع ریښتیا;
}

فراغ print_ننوت_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor) {
    std::وتنه << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.معلومات()
              << '\n';
    بدل (tensor.get_element_type()) {
    حالة ov::element::f32: {
        إذا (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::وتنه << *(tensor.معلومات<عائم>()) << '\n';
        } وإلا {
            size_t لیکې = tensor.get_shape()[2];
            size_t ستون_شمېر_2 = tensor.get_shape()[3];
            ثابت عائم * معلومات = tensor.معلومات<عائم>();
            لكل (size_t i = 0; i < لیکې; ++i) {
                لكل (size_t j = 0; j < ستون_شمېر_2; ++j) {
                    عائم val = معلومات[i * ستون_شمېر_2 + j];
                    إذا (std::isinf(val) && val < 0) {
                        std::وتنه << std::setw(5) << "-inf";
                    } وإلا {
                        std::وتنه << std::setw(5) << val;
                    }
                }
                std::وتنه << '\n';
            }
        }

        اكسر;
    }
    حالة ov::element::f16:
        std::وتنه << *(tensor.معلومات<ov::float16>()) << '\n';
        اكسر;
    حالة ov::element::i32:
        لكل (size_t i = 0; i < tensor.get_اندازه(); ++i) {
            std::وتنه << tensor.معلومات<int32_t>()[i] << ' ';
        }
        std::وتنه << '\n';
        اكسر;
    حالة ov::element::i64:
        لكل (size_t i = 0; i < tensor.get_اندازه(); ++i) {
            std::وتنه << tensor.معلومات<int64_t>()[i] << ' ';
        }
        std::وتنه << '\n';
        اكسر;
    افتراضي:
        اكسر;
    }
}

فراغ print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت فراغ * output_dst) {
    std::وتنه << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    خپلکاری print_float_stats = [](ثابت std::string & type_name, size_t size, خپلکاری get_ارزښت) {
        إذا (size == 0) {
            أرجع;
        }

        عائم لومړی = get_ارزښت(0);
        عائم min = لومړی;
        عائم max = لومړی;
        مزدوج مجموعه = لومړی;

        لكل (size_t i = 1; i < size; ++i) {
            عائم v = get_ارزښت(i);
            min = std::min(v, min);
            max = std::max(v, max);
            مجموعه += v;
        }
        مزدوج منځنۍ_2 = مجموعه / size;

        std::وتنه << std::ښی << std::setw(6) << type_name << std::ښی << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::وتنه << std::ښی << std::setw(6) << "" << std::ښی << std::setw(12) << لومړی << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << منځنۍ_2 << '\n';
    };

    بدل (tensor.get_element_type()) {
    حالة ov::element::f32: {
        ثابت عائم * معلومات = tensor.معلومات<عائم>();
        size_t size = tensor.get_اندازه();
        print_float_stats("[f32]", size, [معلومات](size_t i) { أرجع معلومات[i]; });
        اكسر;
    }
    حالة ov::element::f16: {
        ثابت ov::float16 * معلومات = tensor.معلومات<ov::float16>();
        size_t size = tensor.get_اندازه();
        print_float_stats("[f16]", size, [معلومات](size_t i) { أرجع static_cast<عائم>(معلومات[i]); });
        اكسر;
    }
    افتراضي:
        اكسر;
    }
}

ثابت ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        خپلکاری * op = cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; ++j) {
            خپلکاری * src = op->src[j];
            إذا (src == تش_اشاره) {
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

int64_t get_inp_pos_n_نښې(ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos) {
    // IMROPE stacks n_planes (t/h/w/e) position planes into inp_pos, so ne[0] is
    // n_planes * n_tokens. Callers that need a token count must divide the planes out.
    صحيح n_planes = 1;
    لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
        خپلکاری * op = cgraph->nodes[i];
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
    أرجع get_inp_pos_n_نښې(cgraph, inp_pos) > 1;
}
