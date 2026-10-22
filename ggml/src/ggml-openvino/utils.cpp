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
    ساكن ثابت منطقي kv_برش_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE");
    إذا (kv_برش_disabled) {
        أرجع std::nullopt;
    }
    إذا (ggml_decoder->is_static() || ggml_decoder->is_stateful()) {
        أرجع std::nullopt;
    }
    إذا (ggml_tensor->op != GGML_OP_NONE || ggml_tensor->view_src != اشاره_تهی) {
        أرجع std::nullopt;
    }
    ثابت خودکار * op = ggml_decoder->get_tensor_استفاده_شده_op(ggml_tensor);
    إذا (!GgmlOvDecoder::is_kvcache(ggml_tensor, op)) {
        أرجع std::nullopt;
    }

    ثابت خودکار & compute_params = ggml_decoder->get_compute_params();
    إذا (compute_params.n_seq_فعال != 1 || compute_params.seq_فعال_شروع != 0) {
        أرجع std::nullopt;
    }

    صحيح layer;
    إذا (خودکار layer_opt = استخراج_layer_from_name(name); layer_opt.has_value()) {
        layer = layer_opt.value();
    } وإلا {
        أرجع std::nullopt;
    }

    ثابت منطقي is_swa = ggml_decoder->is_swa_layer(layer);
    إذا (is_swa) {
        أرجع std::nullopt;
    }
    ثابت صحيح ctx_در_هر_seq = ggml_decoder->get_ctx_در_هر_seq();
    ثابت صحيح n_kv = compute_params.attention_اندازه;
    إذا (ctx_در_هر_seq <= 0 || n_kv <= 0 || n_kv >= ctx_در_هر_seq) {
        أرجع std::nullopt;
    }

    ov::Shape پر_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    إذا (پر_shape.size() != 4 || پر_shape[0] != 1 || پر_shape[1] != 1 ||
        static_cast<صحيح>(پر_shape[2]) != ctx_در_هر_seq) {
        أرجع std::nullopt;
    }

    ov::Shape sliced_shape = پر_shape;
    sliced_shape[2] = static_cast<size_t>(n_kv);

    إذا (ggml_openvino_میانگیر_is_remote(ggml_tensor)) {
        خودکار remote_context = ggml_openvino_get_remote_context();
        خودکار gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        أرجع gpu_context.create_tensor(ggml_decoder->get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->داده);
    }

    أرجع ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), sliced_shape, ggml_tensor->داده);
}

ساكن uint64_t ggml_openvino_model_cache_extra_cfg(منطقي stateful, منطقي recurrent) {
    ثابت حرف * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    ثابت منطقي manual_gqa_enabled = manual_gqa_env != اشاره_تهی ?
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

std::map<std::string, std::shared_ptr<ov::Node>> get_وزن_نامها(ggml_cgraph * cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> نامها;
    لكل (ثابت خودکار & name : GgmlOvDecoder::collect_وزن_نامها(cgraph)) {
        نامها[name] = اشاره_تهی;
    }
    أرجع نامها;
}

// A conservative, exact in-process key, evaluated only on a context-local cache
// miss. Include topology, layouts, op parameters, constant extra inputs and weight
// allocation identities. Never use a sampled weight hash or a graph name alone:
// different models can have identical topology. OV buffer IDs survive address reuse.
ساكن std::string compiled_گراف_کلید(ثابت ggml_cgraph * graph,
                                      ثابت GgmlOvDecoder & decoder,
                                      ثابت std::string & device,
                                      صحيح prefill_chunk_اندازه = 0,
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
    append(prefill_chunk_اندازه);
    منطقي has_وزن_میانگیر_id = خطأ;
    std::unordered_map<ثابت ggml_tensor *, size_t> ids;
    std::function<فراغ(ثابت ggml_tensor *)> بازدید = [&](ثابت ggml_tensor * tensor) {
        إذا (!tensor) {
            append(size_t(0));
            أرجع;
        }
        خودکار inserted = ids.emplace(tensor, ids.size() + 1);
        append(inserted.اولی->second);
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
        ثابت خودکار * پایه = tensor->view_src ? tensor->view_src : tensor;
        ثابت منطقي وزن = پایه->میانگیر && پایه->میانگیر->usage == GGML_BACKEND_میانگیر_USAGE_WEIGHTS;
        append(وزن);
        إذا (وزن && !disk_cache) {
            ثابت size_t میانگیر_id = ggml_backend_openvino_میانگیر_get_ctx_id(پایه->میانگیر);
            has_وزن_میانگیر_id |= میانگیر_id != 0;
            append(میانگیر_id);
            append(tensor->داده);
        }
        بازدید(tensor->view_src);
        لكل (ثابت خودکار * src : tensor->src) {
            بازدید(src);
        }
    };
    append(graph->n_nodes);
    لكل (صحيح i = 0; i < graph->n_nodes; ++i) {
        بازدید(graph->nodes[i]);
    }
    append(graph->n_leafs);
    لكل (صحيح i = 0; i < graph->n_leafs; ++i) {
        بازدید(graph->leafs[i]);
    }
    لكل (ثابت خودکار & درون_داد : decoder.get_model_extra_inputs()) {
        append_string(درون_داد.اولی);
        append_string(درون_داد.second.type.get_type_name());
        append(درون_داد.second.shape.size());
        لكل (خودکار dim : درون_داد.second.shape) {
            append(dim);
        }
        append(درون_داد.second.is_parameter);
        إذا (!درون_داد.second.is_parameter) {
            append(درون_داد.second.value);
        }
    }
    // Without an allocation generation, pointer reuse could select stale weights.
    // Such graphs still get private requests; they simply do not share compilation.
    أرجع disk_cache || has_وزن_میانگیر_id ? کلید : std::string{};
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
    گراف_کلید گراف_id(graph, درست);
    append(گراف_id.n_nodes);
    append(گراف_id.n_leaves);
    append_string(گراف_id.اولی_گره_name);
    append_string(گراف_id.آخرین_گره_name);
    لكل (ثابت خودکار & name : گراف_id.درون_داد_srcs) {
        append_string(name);
    }
    append(params.n_rs_slots);
    append(params.has_rs_rollback);
    append(params.mixed_rope_params);
    append(params.is_cacheless_attn);
    لكل (صحيح layer : params.swa_layers) {
        append(layer);
    }
    لكل (ثابت خودکار & [layer, heads] : params.n_heads_kv_در_هر_layer) {
        append(layer);
        append(heads);
    }
    لكل (ثابت خودکار & [name, درون_داد] : decoder.get_model_inputs()) {
        append_string(name);
        append_string(درون_داد.type.get_type_name());
    }
    لكل (ثابت خودکار & [name, tensor] : decoder.get_model_outputs()) {
        append_string(name);
        append(tensor->type);
    }
    لكل (ثابت خودکار & [name, درون_داد] : decoder.get_model_extra_inputs()) {
        append_string(name);
        append_string(درون_داد.type.get_type_name());
        append(درون_داد.is_parameter);
        إذا (!درون_داد.is_parameter) {
            append(درون_داد.value);
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
        ثابت خودکار & پورت = model.درون_داد(i);
        خودکار گراف_it = گراف_inputs.find(inputs[i]);
        إذا (گراف_it != گراف_inputs.end()) {
            إذا (پورت.get_element_type() != گراف_it->second.type ||
                !پورت.get_partial_shape().compatible(گراف_it->second.shape)) {
                أرجع خطأ;
            }
            استمر;
        }
        خودکار extra_it = extra_inputs.find(inputs[i]);
        إذا (extra_it == extra_inputs.end() || پورت.get_element_type() != extra_it->second.type ||
            !پورت.get_partial_shape().compatible(ov::PartialShape(extra_it->second.shape))) {
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
        ثابت خودکار & پورت = model.output(i);
        إذا (پورت.get_element_type() != GgmlOvDecoder::get_ov_type(گراف_it->second) ||
            !پورت.get_partial_shape().compatible(ov::PartialShape(GgmlOvDecoder::get_shape(گراف_it->second)))) {
            أرجع خطأ;
        }
    }
    أرجع درست;
}

ساكن std::string ov_profiling_csv_فیلد(ثابت std::string & value) {
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
    int64_t مجموع_real_زمان = 0;
    int64_t مجموع_cpu_زمان = 0;
    size_t executed_شمار = 0;
    لكل (ثابت خودکار & info : infer_request.get_profiling_info()) {
        ثابت حرف * وضعیت = "NOT_RUN";
        إذا (info.وضعیت == ov::ProfilingInfo::وضعیت_2::EXECUTED) {
            وضعیت = "EXECUTED";
            مجموع_real_زمان += info.real_زمان.count();
            مجموع_cpu_زمان += info.cpu_زمان.count();
            executed_شمار++;
        } وإلا إذا (info.وضعیت == ov::ProfilingInfo::وضعیت_2::OPTIMIZED_بیرون) {
            وضعیت = "OPTIMIZED_OUT";
        }

        output << وضعیت << ',' << info.real_زمان.count() << ',' << info.cpu_زمان.count() << ','
               << info.شروع_زمان.count() << ',' << ov_profiling_csv_فیلد(info.node_type) << ','
               << ov_profiling_csv_فیلد(info.گره_name) << ',' << ov_profiling_csv_فیلد(info.exec_type) << '\n';
    }

    GGML_LOG_INFO("ggml-openvino: profile %s: %zu executed nodes, %.3f ms device, %.3f ms CPU\n", path.c_str(),
                  executed_شمار, مجموع_real_زمان / 1000.0, مجموع_cpu_زمان / 1000.0);
}

ساكن فراغ reset_single_جایگاه_recurrent_cache(ggml_cgraph * cgraph,
                                              ثابت ModelParams & model_params,
                                              ثابت ComputeParams & compute_params) {
    إذا (model_params.n_rs_slots != 1 || compute_params.cache_rs_reset_len == 0) {
        أرجع;
    }
    GGML_ASSERT(compute_params.cache_rs_reset_idx == 0 && compute_params.cache_rs_reset_len == 1);

    std::set<ggml_backend_میانگیر_t> buffers;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        خودکار * گره = cgraph->nodes[i];
        إذا (گره->op == GGML_OP_SCALE && GgmlOvDecoder::is_recurrent_cache(گره->view_src) &&
            گره->view_src->ne[1] == 1 && گره->view_src->میانگیر != اشاره_تهی) {
            buffers.insert(گره->view_src->میانگیر);
        }
    }
    لكل (خودکار * میانگیر : buffers) {
        ggml_backend_میانگیر_clear(میانگیر, 0);
    }
}

ov::Tensor create_ov_output_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                   ثابت std::shared_ptr<ov::InferRequest> & infer_request,
                                   صحيح output_شاخص,
                                   ثابت ggml_tensor * ggml_tensor) {
    إذا (خودکار sliced = try_make_kv_sliced_tensor(ggml_decoder, std::string(ggml_tensor->name), ggml_tensor)) {
        أرجع *sliced;
    }

    خودکار output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    فراغ * output_داده = ggml_tensor->داده;
    إذا (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_شاخص).get_shape();
    } وإلا {
        // For a CPY into a padded view_src (e.g. a padded KV cache buffer), the
        // OV ScatterUpdate node outputs the full view_src shape, not the CPY node's
        // own (smaller) shape.  Using the CPY shape here causes set_output_tensor to
        // fail with a shape-incompatibility error.  Use view_src's shape and data
        // pointer instead so the OV tensor matches the model output exactly.
        إذا (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != اشاره_تهی &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
            output_داده = ggml_tensor->view_src->داده;
        } وإلا {
            output_shape = GgmlOvDecoder::get_shape(ggml_tensor);
        }
    }

    // The sliced path above covers eligible KV outputs. This also covers recurrent state and full-size KV fallbacks.
    إذا (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS") && !ggml_decoder->is_static() &&
        !ggml_decoder->is_splited_model() && ggml_openvino_میانگیر_is_remote(ggml_tensor) &&
        ggml_tensor->extra != اشاره_تهی) {
        خودکار * extra_پایه = static_cast<ggml_openvino_extra_پایه *>(ggml_tensor->extra);
        إذا (extra_پایه->type == ggml_openvino_extra_پایه::Type::TENSOR) {
            خودکار * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_پایه);
            إذا (tensor_extra->tensor != اشاره_تهی && tensor_extra->tensor->get_element_type() == output_type &&
                tensor_extra->tensor->get_shape() == output_shape) {
                أرجع *tensor_extra->tensor;
            }
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_داده);
    أرجع output_tensor;
}

// Rewrite ggml's KV rows into a relayout state that keeps the sequence on dim 2.
// ggml stores [seq][n_heads_kv * head_size]; the state wants [1, n_heads_kv, seq, head_size],
// a different element order, so the rows are copied instead of reinterpreted.
ov::Tensor kv_ردیفها_مقصد_seq_axis_2(ثابت ov::Tensor & kv_tensor, size_t n_heads_kv) {
    ثابت size_t ردیفها = kv_tensor.get_shape()[2];
    ثابت size_t سر_اندازه = kv_tensor.get_shape()[3] / n_heads_kv;
    ثابت size_t elem = kv_tensor.get_element_type().size();
    ثابت size_t سر_بایتها = سر_اندازه * elem;

    ov::Tensor بیرون(kv_tensor.get_element_type(), ov::Shape{1, n_heads_kv, ردیفها, سر_اندازه});
    ثابت خودکار * src = static_cast<ثابت uint8_t *>(kv_tensor.داده());
    خودکار * dst = static_cast<uint8_t *>(بیرون.داده());
    لكل (size_t s = 0; s < ردیفها; s++) {
        لكل (size_t h = 0; h < n_heads_kv; h++) {
            کپی_حافظه(dst + (h * ردیفها + s) * سر_بایتها, src + (s * n_heads_kv + h) * سر_بایتها, سر_بایتها);
        }
    }
    أرجع بیرون;
}

قالب <نام_نوع T> فراغ مجموعه_zero_diagonal(std::vector<T> & ماتریس, size_t ردیفها, size_t تعداد_ستون_2, T zero_مقدار = T{}) {
    لكل (size_t i = 0; i < ردیفها; ++i) {
        size_t diag_ستون = std::min(i, تعداد_ستون_2 - 1);
        ماتریس[i * تعداد_ستون_2 + diag_ستون] = zero_مقدار;
    }
}

ov::Tensor make_contiguous_تقسیم_درون_داد_tensor(ثابت بنية ggml_tensor * ggml_tensor, ثابت ov::Shape & درون_داد_shape) {
    ثابت size_t element_اندازه = ggml_type_اندازه(ggml_tensor->type);
    ثابت size_t بلوک_اندازه_3 = ggml_blck_اندازه(ggml_tensor->type);

    GGML_ASSERT(بلوک_اندازه_3 == 1 && "non-contiguous split inputs must be plain element types");

    ثابت بنية ggml_tensor * منبع_tensor = ggml_tensor->view_src != اشاره_تهی ? ggml_tensor->view_src : ggml_tensor;
    ثابت size_t منبع_جابه_جایی = ggml_tensor->view_src != اشاره_تهی ? ggml_tensor->view_offs : 0;

    std::vector<uint8_t> منبع_داده(ggml_nbytes(منبع_tensor));
    ggml_backend_tensor_get(منبع_tensor, منبع_داده.داده(), 0, منبع_داده.size());

    ov::Tensor درون_داد_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), درون_داد_shape);
    خودکار * dst = static_cast<uint8_t *>(درون_داد_tensor.داده());
    size_t dst_جابه_جایی = 0;

    لكل (size_t i3 = 0; i3 < static_cast<size_t>(ggml_tensor->ne[3]); ++i3) {
        لكل (size_t i2 = 0; i2 < static_cast<size_t>(ggml_tensor->ne[2]); ++i2) {
            لكل (size_t i1 = 0; i1 < static_cast<size_t>(ggml_tensor->ne[1]); ++i1) {
                لكل (size_t i0 = 0; i0 < static_cast<size_t>(ggml_tensor->ne[0]); ++i0) {
                    ثابت size_t src_جابه_جایی = منبع_جابه_جایی + i3 * ggml_tensor->nb[3] + i2 * ggml_tensor->nb[2] +
                                              i1 * ggml_tensor->nb[1] + i0 * ggml_tensor->nb[0];
                    std::کپی_حافظه(dst + dst_جابه_جایی, منبع_داده.داده() + src_جابه_جایی, element_اندازه);
                    dst_جابه_جایی += element_اندازه;
                }
            }
        }
    }

    أرجع درون_داد_tensor;
}

ov::Tensor convert_ggml_درون_داد_مقصد_ov(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & name) {
    ثابت خودکار * ggml_tensor = ggml_decoder->get_درون_داد_ggml_tensor(name);

    إذا (خودکار sliced = try_make_kv_sliced_tensor(ggml_decoder, name, ggml_tensor)) {
        أرجع *sliced;
    }

    إذا (ggml_tensor->extra != اشاره_تهی && !ggml_decoder->is_splited_model()) {
        خودکار * extra_پایه = static_cast<ggml_openvino_extra_پایه *>(ggml_tensor->extra);
        إذا (extra_پایه->type == ggml_openvino_extra_پایه::Type::TENSOR) {
            // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
            خودکار * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_پایه);
            أرجع *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    خودکار * درون_داد_داده = ggml_tensor->داده;
    ov::Shape درون_داد_shape;
    إذا (ggml_tensor->op == GGML_OP_VIEW && !ggml_decoder->is_splited_model()) {
        // This case is added to make test-backend-ops work
        درون_داد_shape = GgmlOvDecoder::get_shape(ggml_tensor->view_src);
    } وإلا {
        درون_داد_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    }

    إذا (ggml_decoder->is_splited_model() && !ggml_is_contiguous(ggml_tensor)) {
        أرجع make_contiguous_تقسیم_درون_داد_tensor(ggml_tensor, درون_داد_shape);
    }

    خودکار درون_داد_tensor = ov::Tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), درون_داد_shape, درون_داد_داده);
    أرجع درون_داد_tensor;
}

ov::Tensor get_ov_درون_داد_tensor(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder, ثابت std::string & param_name) {
    ov::Tensor درون_داد_tensor;
    خودکار extra_درون_داد = ggml_decoder->get_model_extra_inputs().find(param_name);
    إذا (extra_درون_داد != ggml_decoder->get_model_extra_inputs().end()) {
        درون_داد_tensor = ov::Tensor(extra_درون_داد->second.type, extra_درون_داد->second.shape);
        *درون_داد_tensor.داده<int64_t>() = extra_درون_داد->second.value;
    } وإلا {
        درون_داد_tensor = convert_ggml_درون_داد_مقصد_ov(ggml_decoder, param_name);
    }
    أرجع درون_داد_tensor;
}

ov::Tensor get_ov_درون_داد_tensor_static_decode(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                             ثابت std::string & param_name) {
    // NPU decoding stage
    إذا (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        أرجع get_ov_درون_داد_tensor(ggml_decoder, param_name);
    }
    ثابت خودکار * ggml_tensor = ggml_decoder->get_درون_داد_ggml_tensor(param_name);
    ثابت خودکار * op = ggml_decoder->get_tensor_استفاده_شده_op(ggml_tensor);

    إذا (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        // IMROPE's inp_pos holds one value per t/h/w/e plane instead of a single position;
        // with a single decode token the planes are still contiguous, so a flat copy works.
        ثابت صحيح n_planes = GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ? GgmlOvDecoder::get_inp_pos_n_planes(op) : 1;
        أكد(ggml_tensor->ne[0] == n_planes);
        ov::Shape درون_داد_shape = {1, 1, 1, (size_t) n_planes};
        ov::Tensor درون_داد_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), درون_داد_shape);
        std::کپی_حافظه(درون_داد_tensor.داده(), ggml_tensor->داده, n_planes * ggml_type_اندازه(ggml_tensor->type));
        أرجع درون_داد_tensor;
    }

    إذا (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape درون_داد_shape = {1, 1, 1, 1};
        ov::Tensor درون_داد_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), درون_داد_shape);
        int32_t inp_بیرون_id = *((int32_t *) ggml_tensor->داده);
        أكد(ggml_tensor->ne[0] == 1);
        أكد(inp_بیرون_id == 0);
        *درون_داد_tensor.داده<int32_t>() = inp_بیرون_id;
        أرجع درون_داد_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_ماسک(ggml_tensor, op)) {
        size_t context_اندازه = ggml_decoder->get_ctx_اندازه();
        إذا (ggml_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> padded_داده =
                pad_درون_داد<ggml_fp16_t>(ggml_tensor, 1, context_اندازه, GGML_FP32_مقصد_FP16(-INFINITY));
            ov::Tensor درون_داد_tensor(ov::element::f16, ov::Shape{1, 1, 1, context_اندازه});
            std::کپی_حافظه(درون_داد_tensor.داده(), padded_داده.داده(), padded_داده.size() * حجم(ggml_fp16_t));
            أرجع درون_داد_tensor;
        }

        std::vector<عائم> padded_داده = pad_درون_داد<عائم>(ggml_tensor, 1, context_اندازه, -INFINITY);
        ov::Tensor درون_داد_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_اندازه});
        خودکار * داده_ptr = درون_داد_tensor.داده<عائم>();
        std::copy(padded_داده.begin(), padded_داده.begin() + context_اندازه, داده_ptr);
        أرجع درون_داد_tensor;
    }

    أرجع get_ov_درون_داد_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_درون_داد_tensor_static_prefill(ثابت std::shared_ptr<GgmlOvDecoder> & ggml_decoder,
                                              ثابت std::string & param_name,
                                              صحيح chunk_شاخص) {
    // NPU prompt processing stage
    ثابت size_t درون_داد_len = ggml_decoder->get_درون_داد_len();
    ثابت size_t chunk_اندازه = ggml_decoder->m_prefill_chunk_اندازه;
    ثابت size_t chunk_valid_اندازه = std::min(chunk_اندازه, درون_داد_len - chunk_شاخص * chunk_اندازه);
    ثابت size_t chunk_pad_اندازه = chunk_اندازه - chunk_valid_اندازه;

    إذا (param_name == "chunk_valid_len") {
        ov::Tensor درون_داد_tensor(ov::element::i64, ov::Shape{1});
        *درون_داد_tensor.داده<int64_t>() = (int64_t) chunk_valid_اندازه;
        أرجع درون_داد_tensor;
    }
    إذا (chunk_شاخص > 0 && param_name == "cache_rs_reset_len") {
        // The recurrent-state clear belongs to the start of the sequence. Re-applying it on every
        // chunk would wipe the state accumulated by the preceding chunks, so disable it (a zero
        // length makes scale.cpp's keep-mask select every slot) after the first chunk.
        ov::Tensor درون_داد_tensor(ov::element::i64, ov::Shape{1});
        *درون_داد_tensor.داده<int64_t>() = 0;
        أرجع درون_داد_tensor;
    }
    إذا (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        أرجع get_ov_درون_داد_tensor(ggml_decoder, param_name);
    }
    ثابت خودکار * ggml_tensor = ggml_decoder->get_درون_داد_ggml_tensor(param_name);
    ثابت خودکار * op = ggml_decoder->get_tensor_استفاده_شده_op(ggml_tensor);

    إذا (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        // IMROPE: inp_pos stacks n_planes (t/h/w/e) position planes, each of length
        // input_len; pad every plane independently so they stay aligned to chunk_size.
        ثابت صحيح n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        ثابت size_t element_اندازه = ggml_type_اندازه(ggml_tensor->type);
        ov::Shape درون_داد_shape = {1, 1, 1, (size_t) n_planes * chunk_اندازه};
        ov::Tensor درون_داد_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), درون_داد_shape);
        لكل (صحيح p = 0; p < n_planes; p++) {
            ثابت حرف * src =
                (ثابت حرف *) ggml_tensor->داده + (p * درون_داد_len + chunk_شاخص * chunk_اندازه) * element_اندازه;
            حرف * dst = (حرف *) درون_داد_tensor.داده() + p * chunk_اندازه * element_اندازه;
            std::کپی_حافظه(dst, src, chunk_valid_اندازه * element_اندازه);
            إذا (chunk_pad_اندازه > 0) {
                إذا (ggml_tensor->type == GGML_TYPE_I32) {
                    int32_t آخرین_مقدار = *((ثابت int32_t *) src + chunk_valid_اندازه - 1);
                    int32_t * بیرون = (int32_t *) dst;
                    std::fill(بیرون + chunk_valid_اندازه, بیرون + chunk_اندازه, آخرین_مقدار + 1);
                } وإلا إذا (ggml_tensor->type == GGML_TYPE_I64) {
                    int64_t آخرین_مقدار = *((ثابت int64_t *) src + chunk_valid_اندازه - 1);
                    int64_t * بیرون = (int64_t *) dst;
                    std::fill(بیرون + chunk_valid_اندازه, بیرون + chunk_اندازه, آخرین_مقدار + 1);
                } وإلا {
                    ارم std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        أرجع درون_داد_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape درون_داد_shape = {1, 1, 1, chunk_اندازه};
        ov::Tensor درون_داد_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), درون_داد_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_اندازه = ggml_type_اندازه(ggml_tensor->type);
        فراغ * درون_داد_داده = (حرف *) ggml_tensor->داده + chunk_شاخص * chunk_اندازه * element_اندازه;
        std::کپی_حافظه(درون_داد_tensor.داده(), درون_داد_داده, chunk_valid_اندازه * element_اندازه);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        إذا (chunk_pad_اندازه > 0) {
            إذا (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t آخرین_مقدار =
                    *((int32_t *) ggml_tensor->داده + (chunk_شاخص * chunk_اندازه + chunk_valid_اندازه - 1));
                int32_t * output_داده = درون_داد_tensor.داده<int32_t>();
                std::fill(output_داده + chunk_valid_اندازه, output_داده + chunk_اندازه, آخرین_مقدار + 1);
            } وإلا إذا (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t آخرین_مقدار =
                    *((int64_t *) ggml_tensor->داده + (chunk_شاخص * chunk_اندازه + chunk_valid_اندازه - 1));
                int64_t * output_داده = درون_داد_tensor.داده<int64_t>();
                std::fill(output_داده + chunk_valid_اندازه, output_داده + chunk_اندازه, آخرین_مقدار + 1);
            } وإلا {
                ارم std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        أرجع درون_داد_tensor;
    }

    إذا (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape درون_داد_shape = {1, 1, 1, output_len};
        ov::Tensor درون_داد_tensor(GgmlOvDecoder::get_ov_type(ggml_tensor), درون_داد_shape);
        إذا (ggml_tensor->ne[0] == 0) {
            *درون_داد_tensor.داده<int32_t>() = 0;
        } وإلا {
            خودکار * داده_addr = درون_داد_tensor.داده<int32_t>();
            لكل (size_t i = 0; i < output_len; i++) {
                داده_addr[i] = ((int32_t *) ggml_tensor->داده)[i] % chunk_اندازه;
            }
        }
        أرجع درون_داد_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_scale_ردیفها(ggml_tensor, op)) {
        ov::Tensor درون_داد_tensor(ov::element::f32, ov::Shape{1, 1, chunk_اندازه, 1});
        خودکار * dst = درون_داد_tensor.داده<عائم>();
        ثابت خودکار * src = static_cast<ثابت عائم *>(ggml_tensor->داده) + chunk_شاخص * chunk_اندازه;
        std::copy(src, src + chunk_valid_اندازه, dst);
        std::fill(dst + chunk_valid_اندازه, dst + chunk_اندازه, 1.0f);
        أرجع درون_داد_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_میانگین(ggml_tensor, op)) {
        ثابت size_t n_seqs = ggml_tensor->ne[1];
        ثابت size_t src_stride = ggml_tensor->ne[0];
        ثابت size_t copy_len = std::min<size_t>(chunk_valid_اندازه, src_stride - chunk_شاخص * chunk_اندازه);
        ov::Tensor درون_داد_tensor(ov::element::f32, ov::Shape{1, 1, n_seqs, chunk_اندازه});
        خودکار * dst = درون_داد_tensor.داده<عائم>();
        std::fill(dst, dst + n_seqs * chunk_اندازه, 0.0f);
        ثابت خودکار * src = static_cast<ثابت عائم *>(ggml_tensor->داده) + chunk_شاخص * chunk_اندازه;
        لكل (size_t s = 0; s < n_seqs; s++) {
            std::کپی_حافظه(dst + s * chunk_اندازه, src + s * src_stride, copy_len * حجم(عائم));
        }
        أرجع درون_داد_tensor;
    }

    إذا (GgmlOvDecoder::is_inp_ماسک(ggml_tensor, op)) {
        size_t تعداد_ستون_2 = ggml_tensor->ne[0];
        size_t ردیفها = ggml_tensor->ne[1];
        size_t chunk_valid_ردیفها = std::min(chunk_اندازه, ردیفها - chunk_شاخص * chunk_اندازه);
        size_t context_اندازه = ggml_decoder->get_ctx_اندازه();
        إذا (ggml_tensor->type == GGML_TYPE_F16) {
            ثابت خودکار * ggml_داده =
                static_cast<ثابت ggml_fp16_t *>(ggml_tensor->داده) + chunk_شاخص * chunk_اندازه * تعداد_ستون_2;
            std::vector<ggml_fp16_t> padded_داده = pad_درون_داد<ggml_fp16_t>(ggml_داده, chunk_valid_ردیفها, تعداد_ستون_2, chunk_اندازه,
                                                                          context_اندازه, GGML_FP32_مقصد_FP16(-INFINITY));
            مجموعه_zero_diagonal(padded_داده, chunk_اندازه, context_اندازه, GGML_FP32_مقصد_FP16(0.0f));
            ov::Tensor درون_داد_tensor(ov::element::f16, ov::Shape{1, 1, chunk_اندازه, context_اندازه});
            std::کپی_حافظه(درون_داد_tensor.داده(), padded_داده.داده(), padded_داده.size() * حجم(ggml_fp16_t));
            أرجع درون_داد_tensor;
        }

        ثابت خودکار * ggml_داده = static_cast<ثابت عائم *>(ggml_tensor->داده) + chunk_شاخص * chunk_اندازه * تعداد_ستون_2;
        std::vector<عائم> padded_داده =
            pad_درون_داد<عائم>(ggml_داده, chunk_valid_ردیفها, تعداد_ستون_2, chunk_اندازه, context_اندازه, -INFINITY);
        مجموعه_zero_diagonal(padded_داده, chunk_اندازه, context_اندازه);
        ov::Tensor درون_داد_tensor(ov::element::f32, ov::Shape{1, 1, chunk_اندازه, context_اندازه});
        خودکار * داده_ptr = درون_داد_tensor.داده<عائم>();
        std::copy(padded_داده.begin(), padded_داده.begin() + chunk_اندازه * context_اندازه, داده_ptr);
        أرجع درون_داد_tensor;
    }

    أرجع get_ov_درون_داد_tensor(ggml_decoder, param_name);
}

تعداد ggml_وضعیت naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               ثابت std::string & device,
                               ثابت ov::AnyMap & config,
                               ov_compiled_model_cache & cache) {
    إذا (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        أرجع GGML_وضعیت_SUCCESS;
    }

    std::unique_lock<std::mutex> compile_قفل(cache.mutex);
    منطقي naive = درست;
    خودکار model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph, naive);
    خودکار decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    خودکار درون_داد_model = std::make_shared<ov::frontend::ggml::درون_داد_Model>(decoder);
    خودکار model = ov::frontend::ggml::Front_انتها::convert(درون_داد_model, naive);
    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    خودکار remote_context = ggml_openvino_get_remote_context();
    ov::AnyMap compile_config = config;
    إذا (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        compile_config[ov::hint::execution_نما.name()] = ov::hint::Execution_نما::PERFORMANCE;
    } وإلا {
        compile_config[ov::hint::execution_نما.name()] = ov::hint::Execution_نما::ACCURACY;
    }
    إذا (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), compile_config).create_infer_request());
    } وإلا {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, device, compile_config).create_infer_request());
    }
    std::vector<std::string> درون_داد_نامها;
    std::vector<std::string> output_نامها;
    لكل (ثابت خودکار & param : model->get_parameters()) {
        درون_داد_نامها.push_back(param->get_friendly_name());
    }
    لكل (ثابت خودکار & نتیجه : model->get_results()) {
        output_نامها.push_back(نتیجه->get_friendly_name());
    }
    // Destroy the frontend graph under the compilation lock as well: it can
    // still own edges into the shared weight nodes.
    model.reset();
    درون_داد_model.reset();
    decoder->clear_model_weights();
    model_weights.clear();
    compile_قفل.unlock();

    لكل (size_t i = 0; i < درون_داد_نامها.size(); i++) {
        ثابت خودکار & param_name = درون_داد_نامها[i];
        خودکار درون_داد_tensor = get_ov_درون_داد_tensor(decoder, param_name);
        infer_request->مجموعه_درون_داد_tensor(i, درون_داد_tensor);
    }

    // Use get_output_tensor + memcpy instead of set_output_tensor to avoid memory overwritten
    // when i/o buffer overlaps, e.g. the cgraph is a single PERMUTE

    infer_request->infer();

    لكل (size_t i = 0; i < output_نامها.size(); i++) {
        خودکار output_tensor = infer_request->get_output_tensor(i);
        ثابت خودکار & model_outputs = decoder->get_model_outputs();
        خودکار model_output_it = model_outputs.find(output_نامها[i]);
        إذا (model_output_it == model_outputs.end()) {
            // Debug-only output added via GGML_OPENVINO_DEBUG_NODE; nothing to copy into.
            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                print_output_tensor_info(output_نامها[i], output_tensor, output_tensor.داده());
            }
            استمر;
        }
        خودکار * ggml_tensor = model_output_it->second;
        std::کپی_حافظه(ggml_tensor->داده, output_tensor.داده(), output_tensor.get_byte_اندازه());
    }
    أرجع GGML_وضعیت_SUCCESS;
}

تعداد ggml_وضعیت ov_گراف_compute_dynamic(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    خودکار & core = ov_singleton_core();
    ثابت خودکار & config = ggml_openvino_get_compile_config();
    ثابت خودکار & device = r_ctx->device;
    ساكن خودکار is_static = خطأ;

    ساكن ثابت منطقي cache_disabled = ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");

    خودکار شروع_زمان = ggml_زمان_us();

    // is_model_splitted is O(n_nodes^2) plus a create_weight_nodes scan and takes ~20 ms
    // on a Llama-1B decode graph. It is called once per graph_compute invocation but the
    // graph shape is identical across all decode steps, so memoize by graph_key: compute
    // graph_key first (a few hundred us), and if the same key is already in decoder_cache
    // we know the graph is not splitted (only not-splitted graphs get inserted there).
    ثابت int64_t cache_کلید_شروع_زمان = ggml_زمان_us();
    گراف_کلید کلید(cgraph);
    ثابت int64_t cache_کلید_compute_زمان = ggml_زمان_us() - cache_کلید_شروع_زمان;
    int64_t cache_lookup_زمان = 0;
    منطقي کلید_دیده = خطأ;
    إذا (!cache_disabled) {
        ثابت int64_t cache_lookup_شروع_زمان = ggml_زمان_us();
        std::lock_guard<std::mutex> map_قفل(r_ctx->ctx_mutex);
        کلید_دیده = r_ctx->decoder_cache.find(کلید) != r_ctx->decoder_cache.end();
        cache_lookup_زمان += ggml_زمان_us() - cache_lookup_شروع_زمان;
    }

    ثابت int64_t model_تقسیم_check_شروع_زمان = ggml_زمان_us();
    منطقي model_is_splitted = کلید_دیده ? خطأ : is_model_splitted(cgraph);
    ثابت int64_t model_تقسیم_check_زمان = ggml_زمان_us() - model_تقسیم_check_شروع_زمان;

    إذا (ggml_openvino_model_cache_only() && (model_is_splitted || is_naive(cgraph))) {
        GGML_LOG_ERROR("ggml-openvino: cache-only mode requires a complete model graph\n");
        أرجع GGML_وضعیت_FAILED;
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
    ثابت int64_t گراف_param_compute_شروع_زمان = ggml_زمان_us();
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);
    ثابت int64_t گراف_param_compute_زمان = ggml_زمان_us() - گراف_param_compute_شروع_زمان;

    ثابت منطقي cache_enabled = !model_is_splitted && !cache_disabled;
    ثابت منطقي stateful_recurrent = r_ctx->stateful && m_params.state_اندازه >= 0;
    ثابت منطقي stateful_kv_only = r_ctx->stateful && !stateful_recurrent;
    إذا (r_ctx->stateful &&
        (m_params.n_seq > 1 || c_params.n_seq_فعال > 1 || m_params.n_rs_slots > 1 || m_params.has_rs_rollback)) {
        GGML_LOG_ERROR("OpenVINO stateful execution requires a single slot without recurrent rollback. Use -np 1.\n");
        أرجع GGML_وضعیت_FAILED;
    }
    إذا (stateful_recurrent && !cache_enabled) {
        GGML_LOG_ERROR("OpenVINO recurrent stateful execution requires an unsplit graph with model caching enabled.\n");
        أرجع GGML_وضعیت_FAILED;
    }
    منطقي cache_اصابت = خطأ;

    int64_t decoder_انتها_زمان;
    int64_t conversion_انتها_زمان;
    int64_t compile_انتها_زمان;
    int64_t infer_انتها_زمان;
    int64_t ov_raw_infer_شروع;

    {
        std::shared_ptr<decoder_runtime_ctx> مدخل;
        ModelParams old_m_params;

        ثابت int64_t cache_lookup_شروع_زمان = ggml_زمان_us();
        إذا (cache_enabled) {
            std::lock_guard<std::mutex> map_قفل(r_ctx->ctx_mutex);
            خودکار it = r_ctx->decoder_cache.find(کلید);
            cache_اصابت = it != r_ctx->decoder_cache.end();
            إذا (cache_اصابت) {
                مدخل = it->second;
            } وإلا {
                r_ctx->clear_caches_locked();
                خودکار mutex = std::make_shared<std::mutex>();
                مدخل = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[کلید] = مدخل;
            }
        } وإلا {
            خودکار mutex = std::make_shared<std::mutex>();
            مدخل = std::make_shared<decoder_runtime_ctx>(mutex);
            cache_اصابت = خطأ;
        }
        cache_lookup_زمان += ggml_زمان_us() - cache_lookup_شروع_زمان;

        std::lock_guard<std::mutex> قفل(*(مدخل->mutex));
        cache_اصابت = cache_اصابت && مدخل->ptr && r_ctx->infer_request_cache.count(کلید) != 0;

        إذا (cache_اصابت) {
            ggml_decoder = مدخل->ptr;
            old_m_params = ggml_decoder->get_model_params();
            إذا (!ggml_decoder->is_splited_model()) {
                cache_اصابت = old_m_params.can_reuse_dynamically(m_params);
            }
        }

        std::vector<std::string> ov_درون_داد_نامها;
        std::vector<std::string> ov_output_نامها;

        إذا (cache_اصابت) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->مجموعه_compute_params(c_params);
            ggml_decoder->مجموعه_model_params(m_params);
            إذا (old_m_params.kv_میانگیر_تغییریافته(m_params) || !ggml_decoder->is_bound_مقصد(cgraph)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->جمع_کن_extra_inputs();
            {
                std::lock_guard<std::mutex> map_قفل(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(کلید);
                ov_درون_داد_نامها = r_ctx->ov_درون_داد_نامها_cache.at(کلید);
                ov_output_نامها = r_ctx->ov_output_نامها_cache.at(کلید);
            }

            إذا (stateful_kv_only) {
                ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_داده = (int32_t *) inp_pos->داده;
                خودکار pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                إذا (pos_داده[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_اندازه = pos_shape[3];
                } وإلا إذا (r_ctx->stateful_kv_اندازه == static_cast<size_t>(pos_داده[0])) {
                    r_ctx->stateful_kv_اندازه += pos_shape[3];
                } وإلا {
                    ثابت size_t pos_begin = static_cast<size_t>(pos_داده[0]);
                    ثابت منطقي refill = pos_begin > r_ctx->stateful_kv_اندازه;

                    // A refill seeds the state from ggml's KV cache, so it needs that cache to be a
                    // plain prefix: cell i must hold position i. An SWA layer keeps only the last
                    // n_swa positions, so once a position leaves the window ggml drops it and the
                    // remaining cells shift - cell i stops holding position i. While every position
                    // is still inside the window nothing has been dropped and the refill is sound.
                    إذا (refill && !ggml_decoder->get_model_params().swa_layers.خالی()) {
                        ثابت صحيح n_swa = ggml_decoder->get_compute_params().swa_پنجره;
                        إذا (n_swa < 0 || static_cast<size_t>(n_swa) < pos_begin) {
                            GGML_LOG_ERROR(
                                "GGML OpenVINO backend stateful inference failed: cannot resume at position %zu from a "
                                "state that holds %zu tokens, because the sliding-window layers keep only the last %d "
                                "positions. Run without GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                                pos_begin, r_ctx->stateful_kv_اندازه, n_swa);
                            أرجع GGML_وضعیت_FAILED;
                        }
                    }

                    ثابت منطقي relayout_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT");

                    خودکار حالتها = infer_request->query_state();
                    لكل (خودکار state : حالتها) {
                        خودکار state_tensor = state.get_state();
                        خودکار state_tensor_shape = state_tensor.get_shape();

                        std::string state_name;
                        إذا (خودکار it = r_ctx->kv_state_درون_داد_name_map.find(state.get_name());
                            it != r_ctx->kv_state_درون_داد_name_map.end()) {
                            state_name = it->second;
                        }

                        // Which axis holds the sequence: pass::KVStateSeqAxis moves it from dim 1
                        // to dim 2. The head count is still needed below, because only a 1-head
                        // state stays byte-compatible with ggml's cache buffer. gemma-4 12B mixes
                        // 1-head full layers with 8-head sliding layers, so it is per state.
                        صحيح n_heads_kv = ggml_decoder->get_model_params().n_heads_kv;
                        إذا (خودکار layer = استخراج_layer_from_name(state_name); layer.has_value()) {
                            n_heads_kv = ggml_decoder->get_n_heads_kv_for_layer(layer.value());
                        }
                        ثابت منطقي relayout_this_state = relayout_enabled;
                        ثابت size_t seq_axis = relayout_this_state ? 2 : 1;
                        ثابت size_t سر_axis = seq_axis == 2 ? 1 : 2;

                        إذا (refill) {
                            إذا (state_name.خالی()) {
                                GGML_LOG_ERROR(
                                    "GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                أرجع GGML_وضعیت_FAILED;
                            }
                            خودکار kv_tensor = get_ov_درون_داد_tensor(ggml_decoder, state_name);
                            إذا (relayout_this_state && n_heads_kv != 1) {
                                // several heads with seq on dim 2: not the same bytes as ggml's
                                // buffer, so the rows have to be copied into the new order
                                state_tensor = kv_ردیفها_مقصد_seq_axis_2(kv_tensor, (size_t) n_heads_kv);
                            } وإلا {
                                ov::Shape refill_shape(4);
                                refill_shape[0] = state_tensor_shape[0];
                                refill_shape[seq_axis] = kv_tensor.get_shape()[2];
                                refill_shape[سر_axis] = state_tensor_shape[سر_axis];
                                refill_shape[3] = state_tensor_shape[3];
                                kv_tensor.مجموعه_shape(refill_shape);
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
                            أرجع GGML_وضعیت_FAILED;
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end(state_tensor_shape.begin(), state_tensor_shape.end());
                        end[seq_axis] = pos_begin;
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.مجموعه_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_اندازه = pos_begin + pos_shape[3];
                }
            }

            decoder_انتها_زمان = ggml_زمان_us();
            conversion_انتها_زمان = decoder_انتها_زمان;
            compile_انتها_زمان = decoder_انتها_زمان;
        } وإلا {
            // Compilation can mutate shared weight nodes, so serialize cold paths.
            // The lock is released before binding tensors or running inference.
            خودکار shared_cache = r_ctx->compiled_cache;
            std::unique_lock<std::mutex> compile_قفل(shared_cache->mutex);
            خودکار وزن_نامها = get_وزن_نامها(cgraph);
            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, وزن_نامها,
                                                          is_static, r_ctx->stateful, model_is_splitted);
            ثابت std::string model_cache_dir = ggml_openvino_model_cache_dir();
            ثابت std::string exact_کلید = cache_enabled ? compiled_گراف_کلید(cgraph, *ggml_decoder, device) : "";
            uint64_t model_fp = 0;
            uint64_t exact_fp = 0;
            std::string blob_مسیر, manifest_مسیر;
            إذا (!model_cache_dir.خالی() && !model_is_splitted) {
                ثابت uint64_t extra_cfg =
                    ggml_openvino_model_cache_extra_cfg(r_ctx->stateful, stateful_recurrent);
                model_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/درست, m_params.rope_params, 16, extra_cfg,
                                                    dynamic_گراف_signature(cgraph, *ggml_decoder, m_params));
                exact_fp =
                    ggml_openvino_model_fingerprint(cgraph, device, /*fa=*/درست, m_params.rope_params, 16, extra_cfg,
                                                    compiled_گراف_کلید(cgraph, *ggml_decoder, device, 0, درست));
                blob_مسیر = ggml_openvino_model_cache_blob_مسیر(model_cache_dir, model_fp);
                manifest_مسیر = ggml_openvino_model_cache_manifest_مسیر(model_cache_dir, model_fp);
            }
            std::string shared_کلید =
                cache_enabled ? (model_fp ? "dynamic:" + std::to_string(model_fp) : exact_کلید) : "";
            ov::CompiledModel shared_model;
            منطقي imported = خطأ;
            خودکار shared_it = shared_cache->graphs.find(shared_کلید);
            إذا (!shared_کلید.خالی() && shared_it != shared_cache->graphs.end() &&
                (model_fp == 0 || compiled_model_matches_گراف(shared_it->second.decode, shared_it->second.درون_داد_نامها,
                                                               shared_it->second.output_نامها, *ggml_decoder))) {
                shared_model = shared_it->second.decode;
                infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                ov_درون_داد_نامها = shared_it->second.درون_داد_نامها;
                ov_output_نامها = shared_it->second.output_نامها;
                imported = درست;
                GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (dynamic)\n");
            } وإلا إذا (model_fp && shared_it != shared_cache->graphs.end()) {
                shared_کلید = exact_کلید;
                shared_it = shared_cache->graphs.find(shared_کلید);
                إذا (shared_it != shared_cache->graphs.end()) {
                    shared_model = shared_it->second.decode;
                    infer_request = std::make_shared<ov::InferRequest>(shared_model.create_infer_request());
                    ov_درون_داد_نامها = shared_it->second.درون_داد_نامها;
                    ov_output_نامها = shared_it->second.output_نامها;
                    imported = درست;
                }
                model_fp = exact_fp;
                blob_مسیر = ggml_openvino_model_cache_blob_مسیر(model_cache_dir, model_fp);
                manifest_مسیر = ggml_openvino_model_cache_manifest_مسیر(model_cache_dir, model_fp);
            }
            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_قفل(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(کلید);
            }

            // Import standalone blobs with weights before graph conversion and compilation.
            // Split models use the plugin cache instead. The decoder still maps graph I/O.
            ov::AnyMap mc_config = config;
            إذا (!model_cache_dir.خالی()) {
                mc_config.erase("CACHE_DIR");
                mc_config[ov::cache_نما.name()] = ov::Cache_نما::OPTIMIZE_سرعت;
            }
            إذا (!imported && !model_cache_dir.خالی() && !model_is_splitted) {
                ثابت uint64_t preferred_fp = model_fp;
                منطقي dynamic_incompatible = preferred_fp == exact_fp;
                لكل (صحيح candidate = 0; candidate < 2; ++candidate) {
                    إذا (imported || (candidate == 1 && preferred_fp == exact_fp)) {
                        اكسر;
                    }
                    ثابت uint64_t candidate_fp = candidate == 0 ? preferred_fp : exact_fp;
                    ثابت std::string candidate_blob =
                        ggml_openvino_model_cache_blob_مسیر(model_cache_dir, candidate_fp);
                    ثابت std::string candidate_manifest =
                        ggml_openvino_model_cache_manifest_مسیر(model_cache_dir, candidate_fp);
                    std::ifstream blob_درون(candidate_blob, std::ios::binary);
                    std::vector<std::string> saved_inputs, saved_outputs;
                    إذا (!blob_درون.is_open() ||
                        !ggml_openvino_model_cache_verify_manifest(candidate_manifest, cgraph, candidate_fp,
                                                                   saved_inputs, saved_outputs)) {
                        استمر;
                    }
                    ثابت int64_t import_شروع = ggml_زمان_us();
                    حاول {
                        ov::CompiledModel cm;
                        خودکار remote_context = ggml_openvino_get_remote_context();
                        إذا (remote_context.has_value()) {
                            cm = core.import_model(blob_درون, remote_context.value(), mc_config);
                        } وإلا {
                            cm = core.import_model(blob_درون, device, mc_config);
                        }
                        // CPU serialization can rename Results that share a name with a Parameter.
                        إذا (!saved_inputs.خالی() || !saved_outputs.خالی()) {
                            ov_درون_داد_نامها = std::move(saved_inputs);
                            ov_output_نامها = std::move(saved_outputs);
                        } وإلا {
                            لكل (ثابت خودکار & p : cm.inputs()) {
                                ov_درون_داد_نامها.push_back(p.get_گره()->get_friendly_name());
                            }
                            لكل (ثابت خودکار & o : cm.outputs()) {
                                ov_output_نامها.push_back(o.get_گره()->get_friendly_name());
                            }
                        }
                        إذا (!compiled_model_matches_گراف(cm, ov_درون_داد_نامها, ov_output_نامها, *ggml_decoder)) {
                            dynamic_incompatible |= candidate == 0;
                            ov_درون_داد_نامها.clear();
                            ov_output_نامها.clear();
                            استمر;
                        }
                        infer_request = std::make_shared<ov::InferRequest>(cm.create_infer_request());
                        shared_model = cm;
                        مدخل->ptr = ggml_decoder;
                        imported = درست;
                        model_fp = candidate_fp;
                        blob_مسیر = candidate_blob;
                        manifest_مسیر = candidate_manifest;
                        إذا (candidate_fp == exact_fp) {
                            shared_کلید = exact_کلید;
                            shared_it = shared_cache->graphs.find(shared_کلید);
                        }
                        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
                            GGML_LOG_INFO("  - Model cache import time: %.3f ms \n",
                                          (ggml_زمان_us() - import_شروع) / 1000.0);
                        }
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", candidate_blob.c_str());
                    } امسك (ثابت std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                        dynamic_incompatible |= candidate == 0;
                        imported = خطأ;
                        ov_درون_داد_نامها.clear();
                        ov_output_نامها.clear();
                        infer_request.reset();
                        shared_model = {};
                    }
                }
                إذا (!imported && dynamic_incompatible) {
                    model_fp = exact_fp;
                    shared_کلید = exact_کلید;
                    shared_it = shared_cache->graphs.find(shared_کلید);
                    blob_مسیر = ggml_openvino_model_cache_blob_مسیر(model_cache_dir, model_fp);
                    manifest_مسیر = ggml_openvino_model_cache_manifest_مسیر(model_cache_dir, model_fp);
                }
            }

            std::shared_ptr<ov::Model> model;
            إذا (!imported && ggml_openvino_model_cache_only()) {
                GGML_LOG_ERROR(
                    "ggml-openvino: missing or incompatible compiled model: %s; run once without "
                    "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY\n",
                    blob_مسیر.c_str());
                أرجع GGML_وضعیت_FAILED;
            }
            إذا (!imported && ggml_openvino_وزن_buffers_released()) {
                GGML_LOG_ERROR("ggml-openvino: cannot compile a new graph after releasing host weights\n");
                أرجع GGML_وضعیت_FAILED;
            }
            إذا (imported) {
                decoder_انتها_زمان = conversion_انتها_زمان = compile_انتها_زمان = ggml_زمان_us();
            } وإلا {
                خودکار model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph);

                ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                               r_ctx->stateful, model_is_splitted);
                decoder_انتها_زمان = ggml_زمان_us();

                خودکار درون_داد_model = std::make_shared<ov::frontend::ggml::درون_داد_Model>(ggml_decoder);
                model = ov::frontend::ggml::Front_انتها::convert(درون_داد_model);
                ggml_decoder->clear_model_weights();
                conversion_انتها_زمان = ggml_زمان_us();

                إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR")) {
                    حرف timestamped_filename[64];
                    خودکار timestamp = (دراز دراز) ggml_زمان_us();
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                // Use the cache-stripped config when the frontend model cache is active, so
                // the resulting CompiledModel can be exported and later re-imported.
                ثابت ov::AnyMap & compile_config = model_cache_dir.خالی() ? config : mc_config;
                ov::CompiledModel compiled_model;
                خودکار remote_context = ggml_openvino_get_remote_context();
                إذا (remote_context.has_value()) {
                    compiled_model = core.compile_model(model, remote_context.value(), compile_config);
                } وإلا {
                    compiled_model = core.compile_model(model, device, compile_config);
                }
                compile_انتها_زمان = ggml_زمان_us();

                لكل (ثابت خودکار & ov_param : model->get_parameters()) {
                    ov_درون_داد_نامها.push_back(ov_param->get_friendly_name());
                }
                لكل (ثابت خودکار & ov_output : model->get_results()) {
                    ov_output_نامها.push_back(ov_output->get_friendly_name());
                }

                // Export to the frontend model cache for next time. Publish the blob first,
                // then the manifest, so a cache hit only sees fully written artifacts.
                إذا (!model_cache_dir.خالی() && !model_is_splitted && model_fp != 0) {
                    حاول {
                        ثابت std::string blob_tmp = ggml_openvino_model_cache_موقت_مسیر(blob_مسیر);
                        ثابت std::string manifest_tmp = ggml_openvino_model_cache_موقت_مسیر(manifest_مسیر);
                        إذا (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp, ov_درون_داد_نامها,
                                                                     ov_output_نامها)) {
                            std::ofstream blob_بیرون(blob_tmp, std::ios::binary | std::ios::trunc);
                            إذا (blob_بیرون.is_open()) {
                                compiled_model.export_model(blob_بیرون);
                                blob_بیرون.close();
                                إذا (blob_بیرون.good()) {
                                    إذا (std::rename(blob_tmp.c_str(), blob_مسیر.c_str()) == 0 &&
                                        std::rename(manifest_tmp.c_str(), manifest_مسیر.c_str()) == 0) {
                                        GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_مسیر.c_str());
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
                مدخل->ptr = ggml_decoder;

            }  // end non-imported (compile) path

            مدخل->ptr = ggml_decoder;
            إذا (!shared_کلید.خالی() && shared_it == shared_cache->graphs.end()) {
                shared_cache->graphs.emplace(shared_کلید,
                                             ov_compiled_گراف{shared_model, {}, ov_درون_داد_نامها, ov_output_نامها});
            }
            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_قفل(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[کلید] = infer_request;
                r_ctx->ov_درون_داد_نامها_cache[کلید] = ov_درون_داد_نامها;
                r_ctx->ov_output_نامها_cache[کلید] = ov_output_نامها;
            }

            إذا (stateful_kv_only && cache_enabled) {
                ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
                خودکار pos_shape = GgmlOvDecoder::get_shape(inp_pos);
                // A freshly compiled model starts with an empty state, so it can only serve a
                // sequence from its beginning. A non-zero start position means the KV history was
                // built elsewhere (a restored ggml cache), which the state cannot adopt.
                ثابت int32_t pos_begin = ((int32_t *) inp_pos->داده)[0];
                إذا (pos_begin != 0) {
                    GGML_LOG_ERROR(
                        "GGML OpenVINO backend stateful inference failed: a new model was compiled for a sequence that "
                        "starts at position %d, but its state is empty. Run without "
                        "GGML_OPENVINO_STATEFUL_EXECUTION.\n",
                        pos_begin);
                    أرجع GGML_وضعیت_FAILED;
                }
                r_ctx->stateful_kv_اندازه = pos_shape[3];
                ثابت خودکار kv_param_res_نامها = ggml_decoder->get_kv_param_res_نامها();
                لكل (ثابت خودکار & pair : kv_param_res_نامها) {
                    r_ctx->kv_state_درون_داد_name_map[pair.اولی + pair.second] = pair.اولی;
                }
            }
        }

        إذا (stateful_recurrent) {
            ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
            ثابت int32_t pos_begin = static_cast<ثابت int32_t *>(inp_pos->داده)[0];
            إذا (pos_begin == 0) {
                infer_request->reset_state();
            } وإلا إذا (!cache_اصابت || old_m_params.kv_میانگیر_تغییریافته(m_params) || c_params.cache_rs_reset_len > 0 ||
                       pos_begin < 0 || static_cast<size_t>(pos_begin) != r_ctx->stateful_kv_اندازه) {
                GGML_LOG_ERROR(
                    "OpenVINO recurrent stateful execution cannot restore or rewind a sequence. Restart at position 0 "
                    "or disable GGML_OPENVINO_STATEFUL_EXECUTION.\n");
                أرجع GGML_وضعیت_FAILED;
            }
        } وإلا {
            reset_single_جایگاه_recurrent_cache(cgraph, m_params, c_params);
        }

        لكل (size_t i = 0; i < ov_درون_داد_نامها.size(); i++) {
            ثابت خودکار & param_name = ov_درون_داد_نامها[i];
            خودکار درون_داد_tensor = get_ov_درون_داد_tensor(ggml_decoder, param_name);
            infer_request->مجموعه_درون_داد_tensor(i, درون_داد_tensor);

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                print_درون_داد_tensor_info(param_name, درون_داد_tensor);
            }
        }

        لكل (size_t i = 0; i < ov_output_نامها.size(); i++) {
            // Debug-only outputs added via GGML_OPENVINO_DEBUG_NODE (see
            // translate_session.cpp) have no corresponding ggml tensor; leave
            // them unbound so OpenVINO allocates its own tensor for them,
            // rather than aliasing a ggml buffer that may be overwritten by a
            // later in-place op before we get to read it.
            ثابت خودکار & model_outputs = ggml_decoder->get_model_outputs();
            خودکار model_output_it = model_outputs.find(ov_output_نامها[i]);
            إذا (model_output_it == model_outputs.end()) {
                استمر;
            }
            خودکار * ggml_tensor = model_output_it->second;
            إذا (ggml_nbytes(ggml_tensor) == 0) {
                استمر;
            }
            خودکار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->مجموعه_output_tensor(i, output_tensor);
        }

        ov_raw_infer_شروع = ggml_زمان_us();
        infer_request->infer();
        infer_انتها_زمان = ggml_زمان_us();
        إذا (stateful_recurrent) {
            ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
            r_ctx->stateful_kv_اندازه =
                static_cast<ثابت int32_t *>(inp_pos->داده)[0] + get_inp_pos_n_نشانهها(cgraph, inp_pos);
        }
        dump_ov_profiling_info(*infer_request);

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            لكل (size_t i = 0; i < ov_output_نامها.size(); i++) {
                ثابت خودکار output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_نامها[i], output_tensor, output_tensor.داده());
            }
        }

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_انتها_زمان - شروع_زمان) / 1000.0);
            GGML_LOG_INFO("    - cache key compute time: %.3f ms \n", cache_کلید_compute_زمان / 1000.0);
            GGML_LOG_INFO("    - model split check time: %.3f ms \n", model_تقسیم_check_زمان / 1000.0);
            GGML_LOG_INFO("    - graph param compute time: %.3f ms \n", گراف_param_compute_زمان / 1000.0);
            GGML_LOG_INFO("    - cache lookup time: %.3f ms \n", cache_lookup_زمان / 1000.0);
            إذا (!cache_اصابت) {
                GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n",
                              (conversion_انتها_زمان - decoder_انتها_زمان) / 1000.0);
                GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_انتها_زمان - conversion_انتها_زمان) / 1000.0);
            }
            GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_انتها_زمان - compile_انتها_زمان) / 1000.0);
            GGML_LOG_INFO("    - OV raw infer time: %.3f ms \n", (infer_انتها_زمان - ov_raw_infer_شروع) / 1000.0);
        }
    }

    // GGML_OPENVINO_RELEASE_WEIGHTS (or GGML_OPENVINO_MEMORY_OPTIMIZE on GPU): the plugin holds its own device copy of
    // every weight after compile, so the host weight buffers can be dropped to reclaim
    // RSS. Release only while holding the compilation mutex so another context cannot
    // be reading host weights during conversion/compilation. Pin the shared compiled
    // models across backend teardown; a later context can create its own request without
    // reading the dropped pages. A new, uncached graph still fails fast above.
    إذا (cache_اصابت && ggml_openvino_release_weights_enabled()) {
        std::lock_guard<std::mutex> compile_قفل(r_ctx->compiled_cache->mutex);
        إذا (!ggml_openvino_وزن_buffers_released()) {
            ggml_openvino_release_وزن_buffers();
        }
    }

    أرجع GGML_وضعیت_SUCCESS;
}

ov::AnyMap without_npuw(ثابت ov::AnyMap & config) {
    ov::AnyMap بیرون;
    لكل (ثابت خودکار & kv : config) {
        إذا (kv.اولی.rfind("NPUW", 0) == 0 || kv.اولی == "NPU_USE_NPUW") {
            استمر;
        }
        بیرون.insert(kv);
    }
    أرجع بیرون;
}

تعداد ggml_وضعیت ov_گراف_compute_static(ggml_cgraph * cgraph, ثابت std::shared_ptr<ov_runtime_context> & r_ctx) {
    خودکار & core = ov_singleton_core();

    خودکار get_prefill_chunk_اندازه = [] {
        ساكن ثابت صحيح chunk_اندازه = []() {
            صحيح env_prefill_chunk_اندازه = ggml_openvino_getenv_int("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
            أرجع env_prefill_chunk_اندازه > 0 ? env_prefill_chunk_اندازه : 256;
        }();
        أرجع chunk_اندازه;
    };

    // Normally NPU, but honors GGML_OPENVINO_DEVICE so GGML_OPENVINO_FORCE_STATIC can run the
    // static-shape path on CPU/GPU to isolate translation bugs from NPUW/NPU-driver issues.
    ساكن std::string device = ggml_openvino_get_device_name();
    ساكن خودکار is_static = درست;
    ساكن خودکار stateful = خطأ;

    خودکار prefill_chunk_اندازه = get_prefill_chunk_اندازه();
    ثابت خودکار & config = ggml_openvino_get_compile_config();

    إذا (is_naive(cgraph)) {
        أرجع naive_compute(cgraph, core, device, config, *r_ctx->compiled_cache);
    }

    خودکار شروع_زمان = ggml_زمان_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    ثابت خودکار * inp_pos = get_inp_pos_tensor(cgraph);
    ثابت منطقي no_kv_cache = m_params.is_cacheless_attn;
    ثابت خودکار is_prefill = no_kv_cache ? درست : get_is_prefill(cgraph, inp_pos);
    ثابت ov::AnyMap compile_config = no_kv_cache ? without_npuw(config) : config;
    إذا (m_params.n_heads_kv == -1) {
        prefill_chunk_اندازه = inp_pos->ne[0];
    }
    گراف_کلید کلید(cgraph);
    ساكن ثابت منطقي cache_enabled = !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_CACHE");
    منطقي cache_اصابت = خطأ;

    int64_t decoder_انتها_زمان;
    int64_t conversion_انتها_زمان;
    int64_t compile_انتها_زمان;
    int64_t infer_انتها_زمان;
    int64_t ov_raw_infer_شروع;
    int64_t ov_raw_infer_مجموع = 0;

    std::shared_ptr<decoder_runtime_ctx> مدخل;
    ModelParams old_m_params;

    إذا (cache_enabled) {
        std::lock_guard<std::mutex> map_قفل(r_ctx->ctx_mutex);
        خودکار it = r_ctx->decoder_cache.find(کلید);
        cache_اصابت = it != r_ctx->decoder_cache.end();
        إذا (cache_اصابت) {
            مدخل = it->second;
        } وإلا {
            r_ctx->clear_caches_locked();
            خودکار mutex = std::make_shared<std::mutex>();
            مدخل = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[کلید] = مدخل;
        }
    } وإلا {
        خودکار mutex = std::make_shared<std::mutex>();
        مدخل = std::make_shared<decoder_runtime_ctx>(mutex);
        cache_اصابت = خطأ;
    }

    std::lock_guard<std::mutex> قفل(*(مدخل->mutex));
    cache_اصابت = cache_اصابت && مدخل->ptr && r_ctx->infer_request_cache.count(کلید) != 0 &&
                r_ctx->infer_request_cache_prefill.count(کلید) != 0;

    إذا (cache_اصابت) {
        ggml_decoder = مدخل->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_اصابت = old_m_params.can_reuse_statically(m_params);
    }

    std::vector<std::string> ov_درون_داد_نامها_local;
    std::vector<std::string> ov_output_نامها_local;

    إذا (cache_اصابت) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->مجموعه_model_params(m_params);
        ggml_decoder->مجموعه_compute_params(c_params);
        إذا (old_m_params.kv_میانگیر_تغییریافته(m_params) || !ggml_decoder->is_bound_مقصد(cgraph)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->جمع_کن_extra_inputs();
        {
            std::lock_guard<std::mutex> map_قفل(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(کلید) : r_ctx->infer_request_cache.at(کلید);
            ov_درون_داد_نامها_local = r_ctx->ov_درون_داد_نامها_cache.at(کلید);
            ov_output_نامها_local = r_ctx->ov_output_نامها_cache.at(کلید);
        }

        decoder_انتها_زمان = ggml_زمان_us();
        conversion_انتها_زمان = decoder_انتها_زمان;
        compile_انتها_زمان = decoder_انتها_زمان;
    } وإلا {
        إذا (cache_enabled) {
            std::lock_guard<std::mutex> map_قفل(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(کلید);
            r_ctx->infer_request_cache_prefill.erase(کلید);
        }

        // Static execution shares a compiled prefill/decode pair. Each backend
        // creates and retains its own requests for both phases.
        خودکار shared_cache = r_ctx->compiled_cache;
        std::unique_lock<std::mutex> compile_قفل(shared_cache->mutex);
        خودکار وزن_نامها = get_وزن_نامها(cgraph);
        خودکار local_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, وزن_نامها, is_static,
                                                             stateful, خطأ, is_prefill, prefill_chunk_اندازه);
        ثابت std::string shared_کلید =
            cache_enabled ? compiled_گراف_کلید(cgraph, *local_decoder, device, prefill_chunk_اندازه) : "";
        خودکار shared_it = shared_cache->graphs.find(shared_کلید);
        إذا (!shared_کلید.خالی() && shared_it != shared_cache->graphs.end()) {
            خودکار & compiled = shared_it->second;
            خودکار prefill_request = std::make_shared<ov::InferRequest>(compiled.prefill.create_infer_request());
            خودکار decode_request = no_kv_cache ?
                                      prefill_request :
                                      std::make_shared<ov::InferRequest>(compiled.decode.create_infer_request());
            ggml_decoder = local_decoder;
            مدخل->ptr = ggml_decoder;
            infer_request = is_prefill ? prefill_request : decode_request;
            ov_درون_داد_نامها_local = compiled.درون_داد_نامها;
            ov_output_نامها_local = compiled.output_نامها;
            r_ctx->infer_request_cache_prefill[کلید] = prefill_request;
            r_ctx->infer_request_cache[کلید] = decode_request;
            r_ctx->ov_درون_داد_نامها_cache[کلید] = ov_درون_داد_نامها_local;
            r_ctx->ov_output_نامها_cache[کلید] = ov_output_نامها_local;
            decoder_انتها_زمان = conversion_انتها_زمان = compile_انتها_زمان = ggml_زمان_us();
            GGML_LOG_DEBUG("ggml-openvino: shared compiled model HIT (static)\n");
        } وإلا {
            std::shared_ptr<ov::Model> model;
            خودکار model_weights = GgmlOvDecoder::create_وزن_nodes(cgraph);

            خودکار ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(
                cgraph, m_params, c_params, model_weights, is_static, stateful, خطأ, درست, prefill_chunk_اندازه);
            خودکار ggml_decoder_decode =
                no_kv_cache ? ggml_decoder_prefill :
                              std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                              stateful, خطأ, خطأ, prefill_chunk_اندازه);
            decoder_انتها_زمان = ggml_زمان_us();

            ثابت منطقي dump_ir = ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_IR");
            ثابت خودکار dump_ir_timestamp = static_cast<دراز دراز>(ggml_زمان_us());

            خودکار build_static_model = [&core, &compile_config, dump_ir, dump_ir_timestamp](
                                          ثابت std::shared_ptr<GgmlOvDecoder> & decoder, ثابت حرف * برچسب,
                                          std::shared_ptr<ov::Model> & model, ov::CompiledModel & compiled_model,
                                          std::shared_ptr<ov::InferRequest> & infer_request,
                                          int64_t & local_conversion_انتها_زمان, int64_t & local_compile_انتها_زمان) {
                خودکار درون_داد_model = std::make_shared<ov::frontend::ggml::درون_داد_Model>(decoder);
                model = ov::frontend::ggml::Front_انتها::convert(درون_داد_model);
                decoder->clear_model_weights();
                local_conversion_انتها_زمان = ggml_زمان_us();

                إذا (dump_ir) {
                    حرف timestamped_filename[64];
                    snprintf(timestamped_filename, حجم(timestamped_filename), "model_%s_%lld.xml", برچسب,
                             dump_ir_timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                compiled_model = core.compile_model(model, device, compile_config);
                infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
                local_compile_انتها_زمان = ggml_زمان_us();
            };
            std::shared_ptr<ov::Model> model_prefill;
            std::shared_ptr<ov::Model> model_decode;
            ov::CompiledModel compiled_model_prefill;
            ov::CompiledModel compiled_model_decode;
            std::shared_ptr<ov::InferRequest> infer_request_prefill;
            std::shared_ptr<ov::InferRequest> infer_request_decode;
            int64_t prefill_conversion_انتها_زمان;
            int64_t decode_conversion_انتها_زمان;
            int64_t prefill_compile_انتها_زمان;
            int64_t decode_compile_انتها_زمان;
            build_static_model(ggml_decoder_prefill, "prefill", model_prefill, compiled_model_prefill,
                               infer_request_prefill, prefill_conversion_انتها_زمان, prefill_compile_انتها_زمان);
            إذا (no_kv_cache) {
                model_decode = model_prefill;
                compiled_model_decode = compiled_model_prefill;
                infer_request_decode = infer_request_prefill;
                decode_conversion_انتها_زمان = prefill_conversion_انتها_زمان;
                decode_compile_انتها_زمان = prefill_compile_انتها_زمان;
            } وإلا {
                build_static_model(ggml_decoder_decode, "decode", model_decode, compiled_model_decode,
                                   infer_request_decode, decode_conversion_انتها_زمان, decode_compile_انتها_زمان);
            }
            conversion_انتها_زمان = std::max(prefill_conversion_انتها_زمان, decode_conversion_انتها_زمان);
            compile_انتها_زمان = std::max(prefill_compile_انتها_زمان, decode_compile_انتها_زمان);

            model = is_prefill ? model_prefill : model_decode;
            ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
            infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
            مدخل->ptr = ggml_decoder;

            لكل (ثابت خودکار & ov_param : model->get_parameters()) {
                ov_درون_داد_نامها_local.push_back(ov_param->get_friendly_name());
            }
            لكل (ثابت خودکار & ov_output : model->get_results()) {
                ov_output_نامها_local.push_back(ov_output->get_friendly_name());
            }

            إذا (!shared_کلید.خالی()) {
                shared_cache->graphs.emplace(
                    shared_کلید, ov_compiled_گراف{compiled_model_decode, compiled_model_prefill, ov_درون_داد_نامها_local,
                                                  ov_output_نامها_local});
            }

            إذا (cache_enabled) {
                std::lock_guard<std::mutex> map_قفل(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache_prefill[کلید] = infer_request_prefill;
                r_ctx->infer_request_cache[کلید] = infer_request_decode;
                r_ctx->ov_درون_داد_نامها_cache[کلید] = ov_درون_داد_نامها_local;
                r_ctx->ov_output_نامها_cache[کلید] = ov_output_نامها_local;
            }
        }
    }

    reset_single_جایگاه_recurrent_cache(cgraph, m_params, c_params);

    إذا (is_prefill) {
        خودکار inp_len = get_inp_pos_n_نشانهها(cgraph, inp_pos);
        لكل (صحيح chunk_شاخص = 0; chunk_شاخص * prefill_chunk_اندازه < inp_len; chunk_شاخص++) {
            لكل (size_t i = 0; i < ov_درون_داد_نامها_local.size(); i++) {
                ثابت خودکار & param_name = ov_درون_داد_نامها_local[i];
                خودکار درون_داد_tensor = get_ov_درون_داد_tensor_static_prefill(ggml_decoder, param_name, chunk_شاخص);
                infer_request->مجموعه_درون_داد_tensor(i, درون_داد_tensor);

                إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                    ثابت خودکار درون_داد_tensor = infer_request->get_درون_داد_tensor(i);
                    print_درون_داد_tensor_info(param_name, درون_داد_tensor);
                }
            }

            لكل (size_t i = 0; i < ov_output_نامها_local.size(); i++) {
                ثابت خودکار & model_outputs = ggml_decoder->get_model_outputs();
                خودکار model_output_it = model_outputs.find(ov_output_نامها_local[i]);
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
                infer_request->مجموعه_output_tensor(i, output_tensor);
            }

            ov_raw_infer_شروع = ggml_زمان_us();
            infer_request->infer();
            ov_raw_infer_مجموع += ggml_زمان_us() - ov_raw_infer_شروع;

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
                ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
                لكل (size_t i = 0; i < ov_output_نامها_local.size(); i++) {
                    ثابت خودکار output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_نامها_local[i], output_tensor, output_tensor.داده());
                }
            }
        }
        infer_انتها_زمان = ggml_زمان_us();
    } وإلا {
        لكل (size_t i = 0; i < ov_درون_داد_نامها_local.size(); i++) {
            ثابت خودکار & param_name = ov_درون_داد_نامها_local[i];
            خودکار درون_داد_tensor = get_ov_درون_داد_tensor_static_decode(ggml_decoder, param_name);
            infer_request->مجموعه_درون_داد_tensor(i, درون_داد_tensor);

            إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_INPUT")) {
                ثابت خودکار درون_داد_tensor = infer_request->get_درون_داد_tensor(i);
                print_درون_داد_tensor_info(param_name, درون_داد_tensor);
            }
        }

        لكل (size_t i = 0; i < ov_output_نامها_local.size(); i++) {
            ثابت خودکار & model_outputs = ggml_decoder->get_model_outputs();
            خودکار model_output_it = model_outputs.find(ov_output_نامها_local[i]);
            إذا (model_output_it == model_outputs.end()) {
                استمر;
            }
            خودکار * ggml_tensor = model_output_it->second;
            إذا (ggml_nbytes(ggml_tensor) == 0) {
                استمر;
            }
            خودکار output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->مجموعه_output_tensor(i, output_tensor);
        }

        ov_raw_infer_شروع = ggml_زمان_us();
        infer_request->infer();
        infer_انتها_زمان = ggml_زمان_us();
        ov_raw_infer_مجموع = infer_انتها_زمان - ov_raw_infer_شروع;

        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DEBUG_OUTPUT") ||
            ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
            لكل (size_t i = 0; i < ov_output_نامها_local.size(); i++) {
                ثابت خودکار output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_نامها_local[i], output_tensor, output_tensor.داده());
            }
        }
    }

    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %.3f ms \n", (decoder_انتها_زمان - شروع_زمان) / 1000.0);
        إذا (!cache_اصابت) {
            GGML_LOG_INFO("  - Graph conversion time: %.3f ms \n", (conversion_انتها_زمان - decoder_انتها_زمان) / 1000.0);
            GGML_LOG_INFO("  - Graph compile time: %.3f ms \n", (compile_انتها_زمان - conversion_انتها_زمان) / 1000.0);
        }
        GGML_LOG_INFO("  - Graph inference time: %.3f ms \n", (infer_انتها_زمان - compile_انتها_زمان) / 1000.0);
        GGML_LOG_INFO("  - OV raw infer time: %.3f ms \n", ov_raw_infer_مجموع / 1000.0);
    }

    أرجع GGML_وضعیت_SUCCESS;
}
}  // namespace

// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
ساكن ggml_cgraph * get_compute_گراف(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
    خودکار is_skipped = [](ثابت ggml_tensor * گره) {
        أرجع گره->op != GGML_OP_NONE && !(گره->flags & GGML_TENSOR_FLAG_COMPUTE);
    };
    إذا (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
        أرجع cgraph;
    }
    خودکار & compute = r_ctx.compute_graphs[cgraph];
    compute.nodes.clear();
    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
                 [&](ثابت ggml_tensor * گره) { أرجع !is_skipped(گره); });
    compute.graph = *cgraph;
    compute.graph.nodes = compute.nodes.داده();
    compute.graph.n_nodes = (صحيح) compute.nodes.size();
    أرجع &compute.graph;
}

// Both execution paths use two cache levels:
// 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
// 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
//    compile if needed, then create a private request from the compiled model.
// The shared lock covers compilation and frontend cleanup, never inference.
تعداد ggml_وضعیت ov_گراف_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    حاول {
        إذا (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        ثابت خودکار is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != اشاره_تهی);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        std::lock_guard<std::mutex> execution_قفل(r_ctx->execution_mutex);
        cgraph = get_compute_گراف(cgraph, *r_ctx);

        أرجع is_static ? ov_گراف_compute_static(cgraph, r_ctx) : ov_گراف_compute_dynamic(cgraph, r_ctx);
    } امسك (ثابت ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        أرجع GGML_وضعیت_FAILED;
    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        أرجع GGML_وضعیت_FAILED;
    } امسك (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        أرجع GGML_وضعیت_FAILED;
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
        ggml_tensor * گره = cgraph->nodes[i];
        صحيح use_شمار = cgraph->use_counts[ggml_hash_پیدا_کن(&cgraph->دیدهشده_hash_مجموعه, گره)];
        // TODO: this is a workround for the tests case from llama.cpp, fix should from the root cause in the future.
        إذا ((cgraph->n_nodes <= 1 && use_شمار == 0) ||
            (cgraph->n_nodes <= 1 && گره->op == GGML_OP_VIEW && use_شمار == 1 && گره->src[0] != اشاره_تهی &&
             گره->src[0]->op == GGML_OP_NONE)) {
            أرجع خطأ;
        }
        إذا (cgraph->n_nodes == 1 &&
            (cgraph->nodes[0]->op == GGML_OP_TRANSPOSE || cgraph->nodes[0]->op == GGML_OP_جایگشت)) {
            أرجع خطأ;
        }
        صحيح درون_داد_use_شمار = 0;
        لكل (صحيح j = 0; j < cgraph->n_nodes; j++) {
            ggml_tensor * other_گره = cgraph->nodes[j];
            لكل (صحيح k = 0; k < GGML_MAX_SRC; k++) {
                إذا (other_گره->src[k] == گره) {
                    درون_داد_use_شمار++;
                }
            }
        }
        إذا (use_شمار != درون_داد_use_شمار && گره->op != GGML_OP_NONE) {
            أرجع درست;
        }
    }
    // if all nodes's src node's src is not come from the nodes in the model, we think the model is splitted. This is a complementary check for the above check, because for some special case like the output node is not used by any node, the use count and input use count are both 0, we can not determine whether the model is splitted or not just based on the first check.
    // Only weight-name membership is needed below. With GGML_OPENVINO_REDUCE_COMPILE_MEM
    // use the name-only collector (no weight extraction); otherwise keep the original
    // behavior of building (naive) weight nodes and take their names.
    std::set<std::string> model_weights;
    إذا (ggml_openvino_reduce_compile_mem_enabled()) {
        model_weights = GgmlOvDecoder::collect_وزن_نامها(cgraph);
    } وإلا {
        لكل (ثابت خودکار & kv : GgmlOvDecoder::create_وزن_nodes(cgraph, درست)) {
            model_weights.insert(kv.اولی);
        }
    }
    std::set<ggml_tensor *> model_nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    // leaf nodes
    std::set<ggml_tensor *> model_leafs(cgraph->leafs, cgraph->leafs + cgraph->n_leafs);
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گره = cgraph->nodes[i];
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = گره->src[j];
            // the src is also not the model weights, we think the model is splitted.
            // the src is also not in model leafs, we think the model is splitted.
            إذا (src != اشاره_تهی && model_nodes.find(src) == model_nodes.end() &&
                model_weights.find(std::string(src->name)) == model_weights.end() && !model_leafs.خالی() == خطأ &&
                model_leafs.find(src) == model_leafs.end()) {
                إذا (GgmlOvDecoder::is_inp_tok(src, گره)) {
                    أرجع خطأ;
                }
                أرجع درست;
            }
        }
    }
    أرجع خطأ;
}

منطقي is_naive(ggml_cgraph * cgraph) {
    constexpr صحيح naive_گراف_اندازه_آستانه = 20;
    صحيح count = 0;
    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        إذا (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    أرجع count < naive_گراف_اندازه_آستانه;
}

size_t checksum(ثابت فراغ * داده, size_t size) {
    ثابت uint8_t * بایتها = static_cast<ثابت uint8_t *>(داده);
    size_t جمع = 0;
    لكل (size_t i = 0; i < size; ++i) {
        جمع += (uint8_t) i;
        جمع += بایتها[i];
    }
    أرجع جمع;
}

منطقي save_ggml_tensor_داده_مقصد_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_مسیر) {
    إذا (tensor == اشاره_تهی || tensor->داده == اشاره_تهی) {
        أرجع خطأ;
    }

    std::ofstream بیرون(file_مسیر);
    إذا (!بیرون.is_open()) {
        أرجع خطأ;
    }

    ثابت size_t n = ggml_nelements(tensor);
    بیرون << "name: " << tensor->name << ", type: " << ggml_type_name(tensor->type) << ", shape: [" << tensor->ne[0]
        << ", " << tensor->ne[1] << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]" << ", elements: " << n
        << ", data:" << '\n';

    بدل (tensor->type) {
    حالة GGML_TYPE_F32: {
        ثابت خودکار * داده = static_cast<ثابت عائم *>(tensor->داده);
        لكل (size_t i = 0; i < n; ++i) {
            بیرون << داده[i] << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_F16: {
        ثابت خودکار * داده = static_cast<ثابت ggml_fp16_t *>(tensor->داده);
        لكل (size_t i = 0; i < n; ++i) {
            بیرون << ggml_fp16_مقصد_fp32(داده[i]) << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_BF16: {
        ثابت خودکار * داده = static_cast<ثابت ggml_bf16_t *>(tensor->داده);
        لكل (size_t i = 0; i < n; ++i) {
            بیرون << ggml_bf16_مقصد_fp32(داده[i]) << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_I32: {
        ثابت خودکار * داده = static_cast<ثابت int32_t *>(tensor->داده);
        لكل (size_t i = 0; i < n; ++i) {
            بیرون << داده[i] << '\n';
        }
        اكسر;
    }
    حالة GGML_TYPE_I64: {
        ثابت خودکار * داده = static_cast<ثابت int64_t *>(tensor->داده);
        لكل (size_t i = 0; i < n; ++i) {
            بیرون << داده[i] << '\n';
        }
        اكسر;
    }
    افتراضي:
        بیرون << "unsupported tensor type for text dump" << '\n';
        أرجع خطأ;
    }

    أرجع درست;
}

فراغ print_درون_داد_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor) {
    std::خروجی << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.داده()
              << '\n';
    بدل (tensor.get_element_type()) {
    حالة ov::element::f32: {
        إذا (name.find("self_kq_mask") == std::string::npos && name.find("KQ_mask") == std::string::npos) {
            std::خروجی << *(tensor.داده<عائم>()) << '\n';
        } وإلا {
            size_t ردیفها = tensor.get_shape()[2];
            size_t تعداد_ستون_2 = tensor.get_shape()[3];
            ثابت عائم * داده = tensor.داده<عائم>();
            لكل (size_t i = 0; i < ردیفها; ++i) {
                لكل (size_t j = 0; j < تعداد_ستون_2; ++j) {
                    عائم val = داده[i * تعداد_ستون_2 + j];
                    إذا (std::isinf(val) && val < 0) {
                        std::خروجی << std::setw(5) << "-inf";
                    } وإلا {
                        std::خروجی << std::setw(5) << val;
                    }
                }
                std::خروجی << '\n';
            }
        }

        اكسر;
    }
    حالة ov::element::f16:
        std::خروجی << *(tensor.داده<ov::float16>()) << '\n';
        اكسر;
    حالة ov::element::i32:
        لكل (size_t i = 0; i < tensor.get_اندازه(); ++i) {
            std::خروجی << tensor.داده<int32_t>()[i] << ' ';
        }
        std::خروجی << '\n';
        اكسر;
    حالة ov::element::i64:
        لكل (size_t i = 0; i < tensor.get_اندازه(); ++i) {
            std::خروجی << tensor.داده<int64_t>()[i] << ' ';
        }
        std::خروجی << '\n';
        اكسر;
    افتراضي:
        اكسر;
    }
}

فراغ print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت فراغ * output_dst) {
    std::خروجی << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << '\n';

    خودکار print_float_stats = [](ثابت std::string & type_name, size_t size, خودکار get_مقدار) {
        إذا (size == 0) {
            أرجع;
        }

        عائم اولی = get_مقدار(0);
        عائم min = اولی;
        عائم max = اولی;
        مزدوج جمع = اولی;

        لكل (size_t i = 1; i < size; ++i) {
            عائم v = get_مقدار(i);
            min = std::min(v, min);
            max = std::max(v, max);
            جمع += v;
        }
        مزدوج میانگین_2 = جمع / size;

        std::خروجی << std::راست << std::setw(6) << type_name << std::راست << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << '\n';
        std::خروجی << std::راست << std::setw(6) << "" << std::راست << std::setw(12) << اولی << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << میانگین_2 << '\n';
    };

    بدل (tensor.get_element_type()) {
    حالة ov::element::f32: {
        ثابت عائم * داده = tensor.داده<عائم>();
        size_t size = tensor.get_اندازه();
        print_float_stats("[f32]", size, [داده](size_t i) { أرجع داده[i]; });
        اكسر;
    }
    حالة ov::element::f16: {
        ثابت ov::float16 * داده = tensor.داده<ov::float16>();
        size_t size = tensor.get_اندازه();
        print_float_stats("[f16]", size, [داده](size_t i) { أرجع static_cast<عائم>(داده[i]); });
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
            إذا (src == اشاره_تهی) {
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

int64_t get_inp_pos_n_نشانهها(ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos) {
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
    أرجع get_inp_pos_n_نشانهها(cgraph, inp_pos) > 1;
}
