#ضمن "translate_session.h"

#ضمن "ggml-impl.h"
#ضمن "ggml-openvino/ggml-openvino-extra.h"
#ضمن "ggml-openvino/openvino/node_context.h"
#ضمن "ggml-openvino/openvino/utils.h"
#ضمن "input_model.h"
#ضمن "pass/fuse_argsort_topk.h"
#ضمن "pass/fuse_moe_router.h"
#ضمن "pass/align_eltwise_ranks.h"
#ضمن "pass/fuse_moe_compressed.h"
#ضمن "pass/fuse_to_conv.h"
#ضمن "pass/kv_state_seq_axis.h"
#ضمن "pass/mark_decompression_convert_constant_folding.h"
#ضمن "pass/mark_dequantization_subgraph.h"
#ضمن "pass/squeeze_matmul.h"
#ضمن "rt_info/weightless_caching_attributes.hpp"

#ضمن <algorithm>
#ضمن <cstdint>
#ضمن <cstdlib>
#ضمن <map>
#ضمن <memory>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/preprocess/pre_post_process.hpp>
#ضمن <openvino/core/shape.hpp>
#ضمن <openvino/core/type.hpp>
#ضمن <openvino/core/type/element_type.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/convert_like.hpp>
#ضمن <openvino/op/cos.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/greater_eq.hpp>
#ضمن <openvino/op/less.hpp>
#ضمن <openvino/op/logical_and.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/parameter.hpp>
#ضمن <openvino/op/range.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/result.hpp>
#ضمن <openvino/op/select.hpp>
#ضمن <openvino/op/sin.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/strided_slice.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/constant_folding.hpp>
#ضمن <openvino/pass/make_stateful.hpp>
#ضمن <limits>
#ضمن <sstream>

نطاق ov {
نطاق frontend {
نطاق ggml {

using نطاق ov::op;

نطاق {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(ثابت std::string & name,
                                                        ثابت ModelInputInfo & input_info) {
    auto param_node = std::make_shared<ov::op::v0::Parameter>(input_info.type, input_info.shape);
    param_node->set_friendly_name(name);
    param_node->output(0).get_tensor().set_names({name});
    أرجع param_node;
}

std::shared_ptr<ov::Node> create_extra_input(ثابت std::string & name, ثابت ModelExtraInputInfo & input_info) {
    𐡄𐡍 (input_info.is_parameter) {
        auto param_node = std::make_shared<ov::op::v0::Parameter>(input_info.type, input_info.shape);
        param_node->set_friendly_name(name);
        param_node->output(0).get_tensor().set_names({name});
        أرجع param_node;
    }

    auto constant = std::make_shared<ov::op::v0::Constant>(input_info.type, input_info.shape,
                                                          std::vector<int64_t>{input_info.value});
    constant->set_friendly_name(name);
    أرجع constant;
}

ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    ثابت std::shared_ptr<ov::Model> & model,
    ثابت std::map<std::string, std::string> & kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    ثابت auto & params = model->get_parameters();
    ثابت auto & results = model->get_results();

    لكل (ثابت auto & param_res : kv_param_res_names) {
        ثابت auto & param_name = param_res.first;
        ثابت auto & res_name = param_res.second;

        auto param_it = std::find_if(params.begin(), params.end(), [&](ثابت std::shared_ptr<v0::Parameter> & node) {
            أرجع node->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        auto res_it = std::find_if(results.begin(), results.end(), [&](ثابت std::shared_ptr<v0::Result> & node) {
            أرجع node->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::Result> res = *res_it;
        pairs.emplace_back(param, res);
    }
    أرجع pairs;
}

فراغ add_sliced_mask_stateful(TensorMap & tensor_map, منطقي imrope) {
    auto create_sliced_mask = [&](ثابت std::string & mask_name, ثابت std::string & sliced_name) {
        𐡄𐡍 ((tensor_map.find(mask_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end()) &&
            (tensor_map.find("inp_pos") != tensor_map.end())) {
            auto token_len_per_seq = tensor_map.at("token_len_per_seq").get_node_shared_ptr();
            auto mask = tensor_map.at(mask_name).get_node_shared_ptr();
            std::shared_ptr<ov::Node> mask_sliced = mask;
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
            // IMROPE's fourth position plane is zero for text; use the token's first plane.
            ov::Output<ov::Node> last_index = neg_one;
            𐡄𐡍 (imrope) {
                last_index = std::make_shared<ov::op::v1::Add>(token_len_per_seq, neg_one);
            }
            auto last_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, last_index, three);
            auto last_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                last_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), خطأ);
            auto last_inp_pos_cvt = std::make_shared<ov::op::v0::Convert>(last_inp_pos_1d, ov::element::i64);
            auto last_inp_pos_inc = std::make_shared<ov::op::v1::Add>(last_inp_pos_cvt, one);

            mask_sliced = std::make_shared<ov::op::v8::Slice>(mask, zero, last_inp_pos_inc, step, axes);
            mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
            mask_sliced->set_friendly_name(sliced_name);

            tensor_map.insert({sliced_name, mask_sliced->output(0)});
        }
    };

    create_sliced_mask("self_kq_mask", "KQ_mask_sliced");
    create_sliced_mask("KQ_mask", "KQ_mask_sliced");
    create_sliced_mask("self_kq_mask_swa", "KQ_mask_swa_sliced");
}

// Rebuild the sliding-window mask from absolute positions.
// ggml caps self_kq_mask_swa at the size of its own SWA cache, but the stateful KV state is
// Concat-appended and grows without bound, so past that cap the two disagree on length and the
// mask add fails. A pure-Concat state is ordered by position, so positions can rebuild the mask.
// swa_window holds the real n_swa, read back from the ggml mask in ggml-decoder.cpp.
// No-op when the graph has no SWA mask, or when the window could not be read back.
فراغ add_position_mask_stateful_swa(TensorMap & tensor_map) {
    𐡄𐡍 (tensor_map.find("self_kq_mask_swa") == tensor_map.end() || tensor_map.find("inp_pos") == tensor_map.end() ||
        tensor_map.find("swa_window") == tensor_map.end()) {
        أرجع;
    }

    auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();

    auto zero_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto one_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    auto query_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::i64);
    auto query_pos_1d = std::make_shared<ov::op::v1::Reshape>(
        query_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), خطأ);

    auto last_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
    auto last_pos_1d = std::make_shared<ov::op::v1::Reshape>(last_pos, one_i64, خطأ);
    auto last_pos_cvt = std::make_shared<ov::op::v0::Convert>(last_pos_1d, ov::element::i64);
    auto total_len = std::make_shared<ov::op::v1::Add>(last_pos_cvt, one_i64);
    auto total_len_scalar = std::make_shared<ov::op::v0::Squeeze>(total_len);

    auto cached_pos = std::make_shared<ov::op::v4::Range>(
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}), total_len_scalar,
        ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);

    auto query_col = std::make_shared<ov::op::v1::Reshape>(
        query_pos_1d, ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1}), خطأ);
    auto cached_row = std::make_shared<ov::op::v1::Reshape>(
        cached_pos, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, -1}), خطأ);
    auto diff = std::make_shared<ov::op::v1::Subtract>(query_col, cached_row);

    auto swa_window = tensor_map.at("swa_window").get_node_shared_ptr();
    auto window = std::make_shared<ov::op::v0::Convert>(swa_window, ov::element::i64);
    auto causal_ok = std::make_shared<ov::op::v1::GreaterEqual>(diff, zero_i64);
    auto window_ok = std::make_shared<ov::op::v1::Less>(diff, window);
    auto keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_ok, window_ok);

    auto zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    auto neg_inf_f = ov::op::v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<عائم>::infinity()});
    std::shared_ptr<ov::Node> mask = std::make_shared<ov::op::v1::Select>(keep, zero_f, neg_inf_f);

    auto batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    mask = std::make_shared<ov::op::v0::Unsqueeze>(mask, batch_axis);
    mask = std::make_shared<ov::op::v0::Unsqueeze>(mask, batch_axis);
    mask = std::make_shared<ov::op::v0::Convert>(mask, ov::element::f16);
    mask->set_friendly_name("KQ_mask_swa_sliced");

    tensor_map["KQ_mask_swa_sliced"] = mask->output(0);
}

فراغ add_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    𐡄𐡍 (ggml_model_decoder.has_mixed_rope_params()) {
        أرجع;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    𐡄𐡍 (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == nullptr) {
        أرجع;
    }
    auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    𐡄𐡍 (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_node_shared_ptr();
    }

    auto sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_weight);
    auto sin_theta = sin_cos.first;
    auto cos_theta = sin_cos.second;

    cos_theta.get_node_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_node_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
فراغ preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    𐡄𐡍 (ggml_model_decoder.is_stateful()) {
        add_sliced_mask_stateful(tensor_map, ggml_model_decoder.get_rope_params()[2] == GGML_ROPE_TYPE_IMROPE);
        add_position_mask_stateful_swa(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(ثابت frontend::InputModel::Ptr & input_model,
                                   ثابت std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   منطقي naive) :
    m_input_model(input_model),
    m_translator_map(translator_map),
    m_ov_model(nullptr),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    𐡄𐡍 (m_ov_model) {
        أرجع m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model);
    أرجع m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(ثابت frontend::InputModel::Ptr & input_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    ثابت auto & ggml_model = ov::as_type_ptr<InputModel>(input_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    لكل (ثابت auto & it : ggml_model_decoder->get_model_inputs()) {
        auto param_node = create_parameter(it.first, it.second);
        params.push_back(param_node);
        (*tensor_map)[it.first] = param_node;
    }

    لكل (ثابت auto & it : ggml_model_decoder->get_model_extra_inputs()) {
        auto input_node = create_extra_input(it.first, it.second);
        𐡄𐡍 (it.second.is_parameter) {
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(input_node));
        }
        (*tensor_map)[it.first] = input_node;
    }

    لكل (ثابت auto & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    auto translate_node = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح node_idx) {
        auto operation_type = decoder->get_op_type(node_idx);
        𐡄𐡍 (operation_type == "GGML_OP_NONE") {
            أرجع ov::OutputVector{};
        }

        ثابت auto & node_output_names = decoder->get_output_names(node_idx);
        𐡄𐡍 (operation_type == "GGML_OP_VIEW" && decoder->get_op_case(node_idx) == 2 && node_output_names.size() == 1) {
            auto direct_output = tensor_map->find(node_output_names[0]);
            𐡄𐡍 (direct_output != tensor_map->end()) {
                // GDN publishes its native attention/state outputs under the two GGML VIEW names.
                // Keep those mappings instead of rebuilding slices of a packed temporary.
                أرجع ov::OutputVector{direct_output->second};
            }
        }

        auto it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        NodeContext node_context(decoder, tensor_map, node_idx, هذا);
        ov::OutputVector converted_outputs = it->second(node_context);

        FRONT_END_OP_CONVERSION_CHECK(node_output_names.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      node_output_names.size(), " and ", converted_outputs.size(), " respectively.");

        لكل (size_t i = 0; i < node_output_names.size(); ++i) {
            auto output_name = node_output_names[i];
            𐡄𐡍 (i < converted_outputs.size() && converted_outputs[i].get_node_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
        أرجع converted_outputs;
    };

    // To handle cases like this
    // 3: [ 18432,     1,     1,     1] RESHAPE              cache_r_l0 (reshaped)#3
    //     [ 18432,     1,     1,     1]            0: NONE        cache_r_l0
    // 4: [     0,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)#4
    //     [ 18432,     1,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)#3
    // 5: [     0,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)#5
    //     [     0,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)#4
    // 6: [     1,     1,     1,     1] VIEW                  (view)#6
    //     [     1,     1,     1,     1]            0: NONE        leaf_5
    // 7: [ 18432,     1,     1,     1] GET_ROWS             conv_states-0#7
    //     [ 18432,     1,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)#3
    //     [     1,     1,     1,     1]            1: VIEW         (view)#6
    // The scale is in-place which modifies cache_r_l0 (reshaped)#3
    // The translation of scale overwrites cache_r in the tensor_map,
    // but we also need to overwrite the old cache_r_l0 (reshaped)#3
    auto refresh_inplace_aliases = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح inplace_node_idx,
                                       ثابت std::string & view_src_name) {
        لكل (صحيح node_idx = 0; node_idx < inplace_node_idx; node_idx++) {
            𐡄𐡍 (decoder->is_view_like_alias_of(node_idx, view_src_name)) {
                translate_node(decoder, node_idx);
            }
        }
    };

    auto node_visitor = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح node_idx) {
        auto converted_outputs = translate_node(decoder, node_idx);
        𐡄𐡍 (converted_outputs.empty()) {
            أرجع;
        }
        ثابت auto inplace_src = decoder->get_inplace_op_src(node_idx);
        𐡄𐡍 (inplace_src.empty()) {
            أرجع;
        }
        𐡄𐡍 (converted_outputs[0].get_node_shared_ptr() != nullptr) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, node_idx, inplace_src);
    };

    𐡄𐡍 (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->visit_subgraph(node_visitor);

    لكل (ثابت auto & name : ggml_model_decoder->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        auto result = std::make_shared<v0::Result>(tensor_map->at(name));
        result->set_friendly_name(name);
        results.push_back(result);
    }

    // Debug-only hook: GGML_OPENVINO_DEBUG_NODE=<name1>,<name2>,... adds extra
    // Result nodes for arbitrary intermediate tensors (looked up by name in
    // tensor_map), on top of the real model outputs above. These debug
    // Results are deliberately NOT added to ggml_decoder's model outputs, so
    // the caller (ov_graph_compute_dynamic in utils.cpp) will not bind them
    // to any ggml tensor buffer -- OpenVINO allocates its own tensor for
    // them. This avoids the risk of reading a ggml buffer that has since
    // been overwritten by a later in-place op (ggml aggressively reuses
    // buffers), which can happen if trying to inspect an intermediate value
    // via GGML_OPENVINO_DEBUG_OUTPUT by hacking it into a real output.
    //
    // tensor_map keys are usually the plain ggml tensor name (e.g. "embd"),
    // but tensors that are recomputed multiple times in the same cgraph
    // (GGML_TENSOR_FLAG_COMPUTE) are disambiguated with a "#<hash>" suffix
    // (e.g. "cache_k_l0#4853", see get_tensor_ov_name()) which is not
    // predictable ahead of time. To keep the env var usable, a requested
    // name is matched either exactly, or as the "name" part before "#" of a
    // suffixed key (first match wins; ambiguous requests should include the
    // full "name#hash" form seen in a previous run's log/dump).
    𐡄𐡍 (ثابت حرف * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        std::stringstream ss(debug_nodes);
        std::string name;
        طالما (std::getline(ss, name, ',')) {
            auto it = tensor_map->find(name);
            𐡄𐡍 (it == tensor_map->end()) {
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](ثابت auto & entry) {
                    أرجع entry.first.compare(0, name.size(), name) == 0 && entry.first.size() > name.size() &&
                           entry.first[name.size()] == '#';
                });
            }
            𐡄𐡍 (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                استمر;
            }
            auto result = std::make_shared<v0::Result>(it->second);
            result->set_friendly_name("__debug_" + it->first);
            results.push_back(result);
        }
    }

    ov::ParameterVector used_params;
    لكل (ثابت auto & param : params) {
        𐡄𐡍 (!param->output(0).get_target_inputs().empty()) {
            used_params.push_back(param);
        }
    }
    // if (auto diff = params.size() - used_params.size()) {
    //     GGML_LOG_INFO("%zu parameters are not used in the model.", diff);
    // }
    resulting_model = std::make_shared<Model>(results, used_params);

    apply_transformations(resulting_model);

    // Set WeightlessCacheAttribute on large constants to avoid unnecessary memory copies
    // in the NPUW plugin. Without this attribute, NPUW's LazyTensor constructor
    // (lazy_tensor.cpp, op::Const::Const) will memcpy every constant "in case export
    // occurs", doubling memory usage per compile_model call.
    //
    // The bin_offset field serves as a unique key (not a real file offset) — this is
    // the same convention the GPU plugin uses for non-IR models (see
    // Plugin::set_weightless_cache_attributes in intel_gpu/src/plugin/plugin.cpp).
    // Each constant must have a distinct bin_offset, otherwise GPU's weightless cache
    // import will map multiple constants to the same data.
    //
    // Small constants (< 16 elements) are excluded since they may be introduced by
    // optimization patterns and the overhead is negligible.
    //
    // Note: use shape_size() rather than byte_size()/element_type().size() - GatherMatmul's default
    // bias is a Constant(element::dynamic, Shape{0}), whose element_type().size() is 0 and would
    // divide by zero.
    size_t offset = 0;
    لكل (auto & node : resulting_model->get_ordered_ops()) {
        𐡄𐡍 (auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node); cnst && ov::shape_size(cnst->get_shape()) >= 16) {
            auto & rt_info = cnst->get_rt_info();
            𐡄𐡍 (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_size(), offset++, cnst->get_element_type());
            }
        }
    }
    أرجع resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    auto ggml_model_decoder = ov::as_type_ptr<InputModel>(m_input_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(𐡒𐡔𐡈);
        manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
        // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
        // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
        // so it survives ConstantFolding regardless of whether the target plugin's own
        // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
        manager.register_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
        manager.register_pass<pass::FuseToConv>();
        manager.register_pass<pass::FuseArgsortTopK>();

        // MOECompressed has no CPU plugin implementation, so enable it by default only on GPU.
        // GGML_OPENVINO_MOE_OP=0 keeps the unfused GatherMatmul path for fallback/debugging.
        𐡄𐡍 (ggml_openvino_is_gpu() &&
            ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) != 0) {
            manager.register_pass<pass::FuseMoeRouter>();
            manager.register_pass<pass::FuseMoeCompressed>();
            // Same fusion for models whose gate and up projections share one fused expert
            // weight (gemma-4). It only matches that shape and only when the experts carry an
            // integer zero point, so it is a no-op on the separate-gate/up models above.
            manager.register_pass<pass::FuseMoeCompressedFusedGateUp>();
        }

        // Workaround for an OpenVINO GPU-plugin defect: an eltwise op whose operands differ in
        // rank is computed wrongly once the plugin fuses it as a post-op into `rms`. gemma-4
        // dense under stateful execution adds a rank-3 residual to the rank-4 norm output and
        // decodes garbage on GPU while CPU is correct. Equalising the ranks is a no-op for
        // NUMPY broadcasting and makes the fused path correct.
        // Remove once the plugin guards that fusion. Opt out with
        // GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN=1.
        𐡄𐡍 (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
            manager.register_pass<pass::AlignEltwiseOperandRanks>();
        }

        𐡄𐡍 (ggml_model_decoder->is_stateful()) {
            ثابت auto kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
            ثابت auto kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
            manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
            // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
            𐡄𐡍 (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
                manager.register_pass<pass::KVStateSeqAxis>();
            }
        }

        𐡄𐡍 (ggml_model_decoder->is_static()) {
            manager.register_pass<pass::SqueezeMatmul>();
        }
        manager.run_passes(model);
        𐡄𐡍 (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            لكل (size_t i = 0; i < model->get_output_size(); i++) {
                auto model_output_shape = model->output(i).get_partial_shape();
                𐡄𐡍 (model_output_shape.rank().is_static() && model_output_shape.rank().get_length() == 3) {
                    ppp.output(i).postprocess().custom([](ثابت ov::Output<ov::Node>& node) {
                        auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        أرجع std::make_shared<ov::op::v0::Unsqueeze>(node, axes);
                    });
                }
            }
            model = ppp.build();
        }
    }
    أرجع model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
