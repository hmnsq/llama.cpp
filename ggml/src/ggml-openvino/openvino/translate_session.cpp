#অন্তর্ভাব "translate_session.h"

#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-openvino/ggml-openvino-extra.h"
#অন্তর্ভাব "ggml-openvino/openvino/node_context.h"
#অন্তর্ভাব "ggml-openvino/openvino/utils.h"
#অন্তর্ভাব "input_model.h"
#অন্তর্ভাব "pass/fuse_argsort_topk.h"
#অন্তর্ভাব "pass/fuse_moe_router.h"
#অন্তর্ভাব "pass/align_eltwise_ranks.h"
#অন্তর্ভাব "pass/fuse_moe_compressed.h"
#অন্তর্ভাব "pass/fuse_to_conv.h"
#অন্তর্ভাব "pass/kv_state_seq_axis.h"
#অন্তর্ভাব "pass/mark_decompression_convert_constant_folding.h"
#অন্তর্ভাব "pass/mark_dequantization_subgraph.h"
#অন্তর্ভাব "pass/squeeze_matmul.h"
#অন্তর্ভাব "rt_info/weightless_caching_attributes.hpp"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <map>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/core/node.hpp>
#অন্তর্ভাব <openvino/core/preprocess/pre_post_process.hpp>
#অন্তর্ভাব <openvino/core/shape.hpp>
#অন্তর্ভাব <openvino/core/type.hpp>
#অন্তর্ভাব <openvino/core/type/element_type.hpp>
#অন্তর্ভাব <openvino/op/add.hpp>
#অন্তর্ভাব <openvino/op/broadcast.hpp>
#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/convert_like.hpp>
#অন্তর্ভাব <openvino/op/cos.hpp>
#অন্তর্ভাব <openvino/op/divide.hpp>
#অন্তর্ভাব <openvino/op/gather.hpp>
#অন্তর্ভাব <openvino/op/greater_eq.hpp>
#অন্তর্ভাব <openvino/op/less.hpp>
#অন্তর্ভাব <openvino/op/logical_and.hpp>
#অন্তর্ভাব <openvino/op/multiply.hpp>
#অন্তর্ভাব <openvino/op/parameter.hpp>
#অন্তর্ভাব <openvino/op/range.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/result.hpp>
#অন্তর্ভাব <openvino/op/select.hpp>
#অন্তর্ভাব <openvino/op/sin.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>
#অন্তর্ভাব <openvino/op/squeeze.hpp>
#অন্তর্ভাব <openvino/op/strided_slice.hpp>
#অন্তর্ভাব <openvino/op/subtract.hpp>
#অন্তর্ভাব <openvino/op/transpose.hpp>
#অন্তর্ভাব <openvino/op/unsqueeze.hpp>
#অন্তর্ভাব <openvino/pass/constant_folding.hpp>
#অন্তর্ভাব <openvino/pass/make_stateful.hpp>
#অন্তর্ভাব <limits>
#অন্তর্ভাব <sstream>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {

using নামাকাশ ov::op;

নামাকাশ {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(ধ্রুব std::string & name,
                                                        ধ্রুব Model_প্রবেশ_Info & প্রবেশ_info) {
    auto param_গ্রন্থি = std::make_shared<ov::op::v0::Parameter>(প্রবেশ_info.type, প্রবেশ_info.shape);
    param_গ্রন্থি->set_friendly_name(name);
    param_গ্রন্থি->output(0).get_tensor().set_names({name});
    প্রত্যাগম param_গ্রন্থি;
}

std::shared_ptr<ov::Node> create_extra_প্রবেশ(ধ্রুব std::string & name, ধ্রুব Model_Extra_প্রবেশ_Info & প্রবেশ_info) {
    যদি (প্রবেশ_info.is_parameter) {
        auto param_গ্রন্থি = std::make_shared<ov::op::v0::Parameter>(প্রবেশ_info.type, প্রবেশ_info.shape);
        param_গ্রন্থি->set_friendly_name(name);
        param_গ্রন্থি->output(0).get_tensor().set_names({name});
        প্রত্যাগম param_গ্রন্থি;
    }

    auto constant = std::make_shared<ov::op::v0::Constant>(প্রবেশ_info.type, প্রবেশ_info.shape,
                                                          std::vector<int64_t>{প্রবেশ_info.value});
    constant->set_friendly_name(name);
    প্রত্যাগম constant;
}

ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    ধ্রুব std::shared_ptr<ov::Model> & model,
    ধ্রুব std::map<std::string, std::string> & kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    ধ্রুব auto & params = model->get_parameters();
    ধ্রুব auto & results = model->get_results();

    প্রতি (ধ্রুব auto & param_res : kv_param_res_names) {
        ধ্রুব auto & param_name = param_res.first;
        ধ্রুব auto & res_name = param_res.second;

        auto param_it = std::find_if(params.begin(), params.end(), [&](ধ্রুব std::shared_ptr<v0::Parameter> & গ্রন্থি) {
            প্রত্যাগম গ্রন্থি->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        auto res_it = std::find_if(results.begin(), results.end(), [&](ধ্রুব std::shared_ptr<v0::Result> & গ্রন্থি) {
            প্রত্যাগম গ্রন্থি->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::Result> res = *res_it;
        pairs.emplace_back(param, res);
    }
    প্রত্যাগম pairs;
}

শূন্য add_sliced_mask_stateful(TensorMap & tensor_map, তার্কিক imrope) {
    auto create_sliced_mask = [&](ধ্রুব std::string & mask_name, ধ্রুব std::string & sliced_name) {
        যদি ((tensor_map.find(mask_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end()) &&
            (tensor_map.find("inp_pos") != tensor_map.end())) {
            auto token_len_per_seq = tensor_map.at("token_len_per_seq").get_গ্রন্থি_shared_ptr();
            auto mask = tensor_map.at(mask_name).get_গ্রন্থি_shared_ptr();
            std::shared_ptr<ov::Node> mask_sliced = mask;
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            auto inp_pos = tensor_map.at("inp_pos").get_গ্রন্থি_shared_ptr();
            // IMROPE's fourth position plane is zero for text; use the token's first plane.
            ov::Output<ov::Node> last_index = neg_one;
            যদি (imrope) {
                last_index = std::make_shared<ov::op::v1::Add>(token_len_per_seq, neg_one);
            }
            auto last_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, last_index, three);
            auto last_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                last_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), অসত্য);
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
শূন্য add_position_mask_stateful_swa(TensorMap & tensor_map) {
    যদি (tensor_map.find("self_kq_mask_swa") == tensor_map.end() || tensor_map.find("inp_pos") == tensor_map.end() ||
        tensor_map.find("swa_window") == tensor_map.end()) {
        প্রত্যাগম;
    }

    auto inp_pos = tensor_map.at("inp_pos").get_গ্রন্থি_shared_ptr();

    auto zero_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto one_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    auto query_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::i64);
    auto query_pos_1d = std::make_shared<ov::op::v1::Reshape>(
        query_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), অসত্য);

    auto last_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
    auto last_pos_1d = std::make_shared<ov::op::v1::Reshape>(last_pos, one_i64, অসত্য);
    auto last_pos_cvt = std::make_shared<ov::op::v0::Convert>(last_pos_1d, ov::element::i64);
    auto total_len = std::make_shared<ov::op::v1::Add>(last_pos_cvt, one_i64);
    auto total_len_scalar = std::make_shared<ov::op::v0::Squeeze>(total_len);

    auto cached_pos = std::make_shared<ov::op::v4::Range>(
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}), total_len_scalar,
        ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);

    auto query_col = std::make_shared<ov::op::v1::Reshape>(
        query_pos_1d, ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1}), অসত্য);
    auto cached_row = std::make_shared<ov::op::v1::Reshape>(
        cached_pos, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, -1}), অসত্য);
    auto diff = std::make_shared<ov::op::v1::Subtract>(query_col, cached_row);

    auto swa_window = tensor_map.at("swa_window").get_গ্রন্থি_shared_ptr();
    auto window = std::make_shared<ov::op::v0::Convert>(swa_window, ov::element::i64);
    auto causal_ok = std::make_shared<ov::op::v1::GreaterEqual>(diff, zero_i64);
    auto window_ok = std::make_shared<ov::op::v1::Less>(diff, window);
    auto keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_ok, window_ok);

    auto zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    auto neg_inf_f = ov::op::v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<ভিন্ন>::infinity()});
    std::shared_ptr<ov::Node> mask = std::make_shared<ov::op::v1::Select>(keep, zero_f, neg_inf_f);

    auto batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    mask = std::make_shared<ov::op::v0::Unsqueeze>(mask, batch_axis);
    mask = std::make_shared<ov::op::v0::Unsqueeze>(mask, batch_axis);
    mask = std::make_shared<ov::op::v0::Convert>(mask, ov::element::f16);
    mask->set_friendly_name("KQ_mask_swa_sliced");

    tensor_map["KQ_mask_swa_sliced"] = mask->output(0);
}

শূন্য add_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    যদি (ggml_model_decoder.has_mixed_rope_params()) {
        প্রত্যাগম;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    যদি (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == nullptr) {
        প্রত্যাগম;
    }
    auto inp_pos = tensor_map.at("inp_pos").get_গ্রন্থি_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    যদি (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_গ্রন্থি_shared_ptr();
    }

    auto sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_weight);
    auto sin_theta = sin_cos.first;
    auto cos_theta = sin_cos.second;

    cos_theta.get_গ্রন্থি_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_গ্রন্থি_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
শূন্য preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    যদি (ggml_model_decoder.is_stateful()) {
        add_sliced_mask_stateful(tensor_map, ggml_model_decoder.get_rope_params()[2] == GGML_ROPE_TYPE_IMROPE);
        add_position_mask_stateful_swa(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(ধ্রুব frontend::প্রবেশ_Model::Ptr & প্রবেশ_model,
                                   ধ্রুব std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   তার্কিক naive) :
    m_প্রবেশ_model(প্রবেশ_model),
    m_translator_map(translator_map),
    m_ov_model(nullptr),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    যদি (m_ov_model) {
        প্রত্যাগম m_ov_model;
    }
    m_ov_model = translate_graph(m_প্রবেশ_model);
    প্রত্যাগম m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(ধ্রুব frontend::প্রবেশ_Model::Ptr & প্রবেশ_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    ধ্রুব auto & ggml_model = ov::as_type_ptr<প্রবেশ_Model>(প্রবেশ_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    প্রতি (ধ্রুব auto & it : ggml_model_decoder->get_model_inputs()) {
        auto param_গ্রন্থি = create_parameter(it.first, it.second);
        params.push_back(param_গ্রন্থি);
        (*tensor_map)[it.first] = param_গ্রন্থি;
    }

    প্রতি (ধ্রুব auto & it : ggml_model_decoder->get_model_extra_inputs()) {
        auto প্রবেশ_গ্রন্থি = create_extra_প্রবেশ(it.first, it.second);
        যদি (it.second.is_parameter) {
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(প্রবেশ_গ্রন্থি));
        }
        (*tensor_map)[it.first] = প্রবেশ_গ্রন্থি;
    }

    প্রতি (ধ্রুব auto & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    auto translate_গ্রন্থি = [&](ধ্রুব std::shared_ptr<GgmlDecoder> & decoder, পূর্ণাঙ্ক গ্রন্থি_idx) {
        auto operation_type = decoder->get_op_type(গ্রন্থি_idx);
        যদি (operation_type == "GGML_OP_NONE") {
            প্রত্যাগম ov::OutputVector{};
        }

        ধ্রুব auto & গ্রন্থি_output_names = decoder->get_output_names(গ্রন্থি_idx);
        যদি (operation_type == "GGML_OP_VIEW" && decoder->get_op_case(গ্রন্থি_idx) == 2 && গ্রন্থি_output_names.size() == 1) {
            auto direct_output = tensor_map->find(গ্রন্থি_output_names[0]);
            যদি (direct_output != tensor_map->end()) {
                // GDN publishes its native attention/state outputs under the two GGML VIEW names.
                // Keep those mappings instead of rebuilding slices of a packed temporary.
                প্রত্যাগম ov::OutputVector{direct_output->second};
            }
        }

        auto it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        গ্রন্থি_Context গ্রন্থি_context(decoder, tensor_map, গ্রন্থি_idx, এতত্);
        ov::OutputVector converted_outputs = it->second(গ্রন্থি_context);

        FRONT_END_OP_CONVERSION_CHECK(গ্রন্থি_output_names.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      গ্রন্থি_output_names.size(), " and ", converted_outputs.size(), " respectively.");

        প্রতি (size_t i = 0; i < গ্রন্থি_output_names.size(); ++i) {
            auto output_name = গ্রন্থি_output_names[i];
            যদি (i < converted_outputs.size() && converted_outputs[i].get_গ্রন্থি_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
        প্রত্যাগম converted_outputs;
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
    auto refresh_inplace_aliases = [&](ধ্রুব std::shared_ptr<GgmlDecoder> & decoder, পূর্ণাঙ্ক inplace_গ্রন্থি_idx,
                                       ধ্রুব std::string & view_src_name) {
        প্রতি (পূর্ণাঙ্ক গ্রন্থি_idx = 0; গ্রন্থি_idx < inplace_গ্রন্থি_idx; গ্রন্থি_idx++) {
            যদি (decoder->is_view_like_alias_of(গ্রন্থি_idx, view_src_name)) {
                translate_গ্রন্থি(decoder, গ্রন্থি_idx);
            }
        }
    };

    auto গ্রন্থি_visitor = [&](ধ্রুব std::shared_ptr<GgmlDecoder> & decoder, পূর্ণাঙ্ক গ্রন্থি_idx) {
        auto converted_outputs = translate_গ্রন্থি(decoder, গ্রন্থি_idx);
        যদি (converted_outputs.empty()) {
            প্রত্যাগম;
        }
        ধ্রুব auto inplace_src = decoder->get_inplace_op_src(গ্রন্থি_idx);
        যদি (inplace_src.empty()) {
            প্রত্যাগম;
        }
        যদি (converted_outputs[0].get_গ্রন্থি_shared_ptr() != nullptr) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, গ্রন্থি_idx, inplace_src);
    };

    যদি (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->visit_subgraph(গ্রন্থি_visitor);

    প্রতি (ধ্রুব auto & name : ggml_model_decoder->get_model_output_names()) {
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
    যদি (ধ্রুব বর্ণ * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        std::stringstream ss(debug_nodes);
        std::string name;
        যাবত্ (std::getline(ss, name, ',')) {
            auto it = tensor_map->find(name);
            যদি (it == tensor_map->end()) {
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](ধ্রুব auto & entry) {
                    প্রত্যাগম entry.first.compare(0, name.size(), name) == 0 && entry.first.size() > name.size() &&
                           entry.first[name.size()] == '#';
                });
            }
            যদি (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                অনুবর্ত;
            }
            auto result = std::make_shared<v0::Result>(it->second);
            result->set_friendly_name("__debug_" + it->first);
            results.push_back(result);
        }
    }

    ov::ParameterVector used_params;
    প্রতি (ধ্রুব auto & param : params) {
        যদি (!param->output(0).get_target_inputs().empty()) {
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
    প্রতি (auto & গ্রন্থি : resulting_model->get_ordered_ops()) {
        যদি (auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(গ্রন্থি); cnst && ov::shape_size(cnst->get_shape()) >= 16) {
            auto & rt_info = cnst->get_rt_info();
            যদি (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_size(), offset++, cnst->get_element_type());
            }
        }
    }
    প্রত্যাগম resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    auto ggml_model_decoder = ov::as_type_ptr<প্রবেশ_Model>(m_প্রবেশ_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(সত্য);
        manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
        // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
        // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
        // so it survives ConstantFolding regardless of whether the target plugin's own
        // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
        manager.register_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
        manager.register_pass<pass::FuseToConv>();
        manager.register_pass<pass::Fuse_Argsort_শীর্ষ_K>();

        // MOECompressed has no CPU plugin implementation, so enable it by default only on GPU.
        // GGML_OPENVINO_MOE_OP=0 keeps the unfused GatherMatmul path for fallback/debugging.
        যদি (ggml_openvino_is_gpu() &&
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
        যদি (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
            manager.register_pass<pass::AlignEltwiseOperandRanks>();
        }

        যদি (ggml_model_decoder->is_stateful()) {
            ধ্রুব auto kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
            ধ্রুব auto kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
            manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
            // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
            যদি (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
                manager.register_pass<pass::KVStateSeqAxis>();
            }
        }

        যদি (ggml_model_decoder->is_static()) {
            manager.register_pass<pass::SqueezeMatmul>();
        }
        manager.run_passes(model);
        যদি (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            প্রতি (size_t i = 0; i < model->get_output_size(); i++) {
                auto model_output_shape = model->output(i).get_partial_shape();
                যদি (model_output_shape.rank().is_static() && model_output_shape.rank().get_length() == 3) {
                    ppp.output(i).postprocess().custom([](ধ্রুব ov::Output<ov::Node>& গ্রন্থি) {
                        auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        প্রত্যাগম std::make_shared<ov::op::v0::Unsqueeze>(গ্রন্থি, axes);
                    });
                }
            }
            model = ppp.build();
        }
    }
    প্রত্যাগম model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
