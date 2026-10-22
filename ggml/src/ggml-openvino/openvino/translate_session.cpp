#அந்தர்பாவ "translate_session.h"

#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-openvino/ggml-openvino-extra.h"
#அந்தர்பாவ "ggml-openvino/openvino/node_context.h"
#அந்தர்பாவ "ggml-openvino/openvino/utils.h"
#அந்தர்பாவ "input_model.h"
#அந்தர்பாவ "pass/fuse_argsort_topk.h"
#அந்தர்பாவ "pass/fuse_moe_router.h"
#அந்தர்பாவ "pass/align_eltwise_ranks.h"
#அந்தர்பாவ "pass/fuse_moe_compressed.h"
#அந்தர்பாவ "pass/fuse_to_conv.h"
#அந்தர்பாவ "pass/kv_state_seq_axis.h"
#அந்தர்பாவ "pass/mark_decompression_convert_constant_folding.h"
#அந்தர்பாவ "pass/mark_dequantization_subgraph.h"
#அந்தர்பாவ "pass/squeeze_matmul.h"
#அந்தர்பாவ "rt_info/weightless_caching_attributes.hpp"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <map>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/node.hpp>
#அந்தர்பாவ <openvino/core/preprocess/pre_post_process.hpp>
#அந்தர்பாவ <openvino/core/shape.hpp>
#அந்தர்பாவ <openvino/core/type.hpp>
#அந்தர்பாவ <openvino/core/type/element_type.hpp>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/broadcast.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/convert_like.hpp>
#அந்தர்பாவ <openvino/op/cos.hpp>
#அந்தர்பாவ <openvino/op/divide.hpp>
#அந்தர்பாவ <openvino/op/gather.hpp>
#அந்தர்பாவ <openvino/op/greater_eq.hpp>
#அந்தர்பாவ <openvino/op/less.hpp>
#அந்தர்பாவ <openvino/op/logical_and.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/parameter.hpp>
#அந்தர்பாவ <openvino/op/range.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/result.hpp>
#அந்தர்பாவ <openvino/op/select.hpp>
#அந்தர்பாவ <openvino/op/sin.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/squeeze.hpp>
#அந்தர்பாவ <openvino/op/strided_slice.hpp>
#அந்தர்பாவ <openvino/op/subtract.hpp>
#அந்தர்பாவ <openvino/op/transpose.hpp>
#அந்தர்பாவ <openvino/op/unsqueeze.hpp>
#அந்தர்பாவ <openvino/pass/constant_folding.hpp>
#அந்தர்பாவ <openvino/pass/make_stateful.hpp>
#அந்தர்பாவ <limits>
#அந்தர்பாவ <sstream>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {

using நாமாகாஶ ov::op;

நாமாகாஶ {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(மாறிலி std::string & name,
                                                        மாறிலி Model_உள்ளீடு_Info & உள்ளீடு_info) {
    auto param_முனை = std::make_shared<ov::op::v0::Parameter>(உள்ளீடு_info.type, உள்ளீடு_info.shape);
    param_முனை->set_friendly_name(name);
    param_முனை->output(0).get_tensor().set_names({name});
    திரும்பு param_முனை;
}

std::shared_ptr<ov::Node> create_extra_உள்ளீடு(மாறிலி std::string & name, மாறிலி Model_Extra_உள்ளீடு_Info & உள்ளீடு_info) {
    எனில் (உள்ளீடு_info.is_parameter) {
        auto param_முனை = std::make_shared<ov::op::v0::Parameter>(உள்ளீடு_info.type, உள்ளீடு_info.shape);
        param_முனை->set_friendly_name(name);
        param_முனை->output(0).get_tensor().set_names({name});
        திரும்பு param_முனை;
    }

    auto constant = std::make_shared<ov::op::v0::Constant>(உள்ளீடு_info.type, உள்ளீடு_info.shape,
                                                          std::vector<int64_t>{உள்ளீடு_info.value});
    constant->set_friendly_name(name);
    திரும்பு constant;
}

ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    மாறிலி std::shared_ptr<ov::Model> & model,
    மாறிலி std::map<std::string, std::string> & kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    மாறிலி auto & params = model->get_parameters();
    மாறிலி auto & results = model->get_results();

    க்கு (மாறிலி auto & param_res : kv_param_res_names) {
        மாறிலி auto & param_name = param_res.first;
        மாறிலி auto & res_name = param_res.second;

        auto param_it = std::find_if(params.begin(), params.end(), [&](மாறிலி std::shared_ptr<v0::Parameter> & முனை) {
            திரும்பு முனை->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        auto res_it = std::find_if(results.begin(), results.end(), [&](மாறிலி std::shared_ptr<v0::Result> & முனை) {
            திரும்பு முனை->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::Result> res = *res_it;
        pairs.emplace_back(param, res);
    }
    திரும்பு pairs;
}

வெறுமை add_sliced_mask_stateful(TensorMap & tensor_map, தருக்க imrope) {
    auto create_sliced_mask = [&](மாறிலி std::string & mask_name, மாறிலி std::string & sliced_name) {
        எனில் ((tensor_map.find(mask_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end()) &&
            (tensor_map.find("inp_pos") != tensor_map.end())) {
            auto token_len_per_seq = tensor_map.at("token_len_per_seq").get_முனை_shared_ptr();
            auto mask = tensor_map.at(mask_name).get_முனை_shared_ptr();
            std::shared_ptr<ov::Node> mask_sliced = mask;
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            auto inp_pos = tensor_map.at("inp_pos").get_முனை_shared_ptr();
            // IMROPE's fourth position plane is zero for text; use the token's first plane.
            ov::Output<ov::Node> last_index = neg_one;
            எனில் (imrope) {
                last_index = std::make_shared<ov::op::v1::Add>(token_len_per_seq, neg_one);
            }
            auto last_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, last_index, three);
            auto last_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                last_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), பொய்);
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
வெறுமை add_position_mask_stateful_swa(TensorMap & tensor_map) {
    எனில் (tensor_map.find("self_kq_mask_swa") == tensor_map.end() || tensor_map.find("inp_pos") == tensor_map.end() ||
        tensor_map.find("swa_window") == tensor_map.end()) {
        திரும்பு;
    }

    auto inp_pos = tensor_map.at("inp_pos").get_முனை_shared_ptr();

    auto zero_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto one_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    auto query_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::i64);
    auto query_pos_1d = std::make_shared<ov::op::v1::Reshape>(
        query_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), பொய்);

    auto last_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
    auto last_pos_1d = std::make_shared<ov::op::v1::Reshape>(last_pos, one_i64, பொய்);
    auto last_pos_cvt = std::make_shared<ov::op::v0::Convert>(last_pos_1d, ov::element::i64);
    auto total_len = std::make_shared<ov::op::v1::Add>(last_pos_cvt, one_i64);
    auto total_len_scalar = std::make_shared<ov::op::v0::Squeeze>(total_len);

    auto cached_pos = std::make_shared<ov::op::v4::Range>(
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}), total_len_scalar,
        ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);

    auto query_col = std::make_shared<ov::op::v1::Reshape>(
        query_pos_1d, ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1}), பொய்);
    auto cached_row = std::make_shared<ov::op::v1::Reshape>(
        cached_pos, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, -1}), பொய்);
    auto diff = std::make_shared<ov::op::v1::Subtract>(query_col, cached_row);

    auto swa_window = tensor_map.at("swa_window").get_முனை_shared_ptr();
    auto window = std::make_shared<ov::op::v0::Convert>(swa_window, ov::element::i64);
    auto causal_ok = std::make_shared<ov::op::v1::GreaterEqual>(diff, zero_i64);
    auto window_ok = std::make_shared<ov::op::v1::Less>(diff, window);
    auto keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_ok, window_ok);

    auto zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    auto neg_inf_f = ov::op::v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<தசம>::infinity()});
    std::shared_ptr<ov::Node> mask = std::make_shared<ov::op::v1::Select>(keep, zero_f, neg_inf_f);

    auto batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    mask = std::make_shared<ov::op::v0::Unsqueeze>(mask, batch_axis);
    mask = std::make_shared<ov::op::v0::Unsqueeze>(mask, batch_axis);
    mask = std::make_shared<ov::op::v0::Convert>(mask, ov::element::f16);
    mask->set_friendly_name("KQ_mask_swa_sliced");

    tensor_map["KQ_mask_swa_sliced"] = mask->output(0);
}

வெறுமை add_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    எனில் (ggml_model_decoder.has_mixed_rope_params()) {
        திரும்பு;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    எனில் (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == nullptr) {
        திரும்பு;
    }
    auto inp_pos = tensor_map.at("inp_pos").get_முனை_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    எனில் (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_முனை_shared_ptr();
    }

    auto sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_weight);
    auto sin_theta = sin_cos.first;
    auto cos_theta = sin_cos.second;

    cos_theta.get_முனை_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_முனை_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
வெறுமை preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    எனில் (ggml_model_decoder.is_stateful()) {
        add_sliced_mask_stateful(tensor_map, ggml_model_decoder.get_rope_params()[2] == GGML_ROPE_TYPE_IMROPE);
        add_position_mask_stateful_swa(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(மாறிலி frontend::உள்ளீடு_Model::Ptr & உள்ளீடு_model,
                                   மாறிலி std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   தருக்க naive) :
    m_உள்ளீடு_model(உள்ளீடு_model),
    m_translator_map(translator_map),
    m_ov_model(nullptr),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    எனில் (m_ov_model) {
        திரும்பு m_ov_model;
    }
    m_ov_model = translate_graph(m_உள்ளீடு_model);
    திரும்பு m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(மாறிலி frontend::உள்ளீடு_Model::Ptr & உள்ளீடு_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    மாறிலி auto & ggml_model = ov::as_type_ptr<உள்ளீடு_Model>(உள்ளீடு_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    க்கு (மாறிலி auto & it : ggml_model_decoder->get_model_inputs()) {
        auto param_முனை = create_parameter(it.first, it.second);
        params.push_back(param_முனை);
        (*tensor_map)[it.first] = param_முனை;
    }

    க்கு (மாறிலி auto & it : ggml_model_decoder->get_model_extra_inputs()) {
        auto உள்ளீடு_முனை = create_extra_உள்ளீடு(it.first, it.second);
        எனில் (it.second.is_parameter) {
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(உள்ளீடு_முனை));
        }
        (*tensor_map)[it.first] = உள்ளீடு_முனை;
    }

    க்கு (மாறிலி auto & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    auto translate_முனை = [&](மாறிலி std::shared_ptr<GgmlDecoder> & decoder, முழு முனை_idx) {
        auto operation_type = decoder->get_op_type(முனை_idx);
        எனில் (operation_type == "GGML_OP_NONE") {
            திரும்பு ov::OutputVector{};
        }

        மாறிலி auto & முனை_output_names = decoder->get_output_names(முனை_idx);
        எனில் (operation_type == "GGML_OP_VIEW" && decoder->get_op_case(முனை_idx) == 2 && முனை_output_names.size() == 1) {
            auto direct_output = tensor_map->find(முனை_output_names[0]);
            எனில் (direct_output != tensor_map->end()) {
                // GDN publishes its native attention/state outputs under the two GGML VIEW names.
                // Keep those mappings instead of rebuilding slices of a packed temporary.
                திரும்பு ov::OutputVector{direct_output->second};
            }
        }

        auto it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        முனை_Context முனை_context(decoder, tensor_map, முனை_idx, இது);
        ov::OutputVector converted_outputs = it->second(முனை_context);

        FRONT_END_OP_CONVERSION_CHECK(முனை_output_names.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      முனை_output_names.size(), " and ", converted_outputs.size(), " respectively.");

        க்கு (size_t i = 0; i < முனை_output_names.size(); ++i) {
            auto output_name = முனை_output_names[i];
            எனில் (i < converted_outputs.size() && converted_outputs[i].get_முனை_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
        திரும்பு converted_outputs;
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
    auto refresh_inplace_aliases = [&](மாறிலி std::shared_ptr<GgmlDecoder> & decoder, முழு inplace_முனை_idx,
                                       மாறிலி std::string & view_src_name) {
        க்கு (முழு முனை_idx = 0; முனை_idx < inplace_முனை_idx; முனை_idx++) {
            எனில் (decoder->is_view_like_alias_of(முனை_idx, view_src_name)) {
                translate_முனை(decoder, முனை_idx);
            }
        }
    };

    auto முனை_visitor = [&](மாறிலி std::shared_ptr<GgmlDecoder> & decoder, முழு முனை_idx) {
        auto converted_outputs = translate_முனை(decoder, முனை_idx);
        எனில் (converted_outputs.empty()) {
            திரும்பு;
        }
        மாறிலி auto inplace_src = decoder->get_inplace_op_src(முனை_idx);
        எனில் (inplace_src.empty()) {
            திரும்பு;
        }
        எனில் (converted_outputs[0].get_முனை_shared_ptr() != nullptr) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, முனை_idx, inplace_src);
    };

    எனில் (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->visit_subgraph(முனை_visitor);

    க்கு (மாறிலி auto & name : ggml_model_decoder->get_model_output_names()) {
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
    எனில் (மாறிலி எழுத்து * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        std::stringstream ss(debug_nodes);
        std::string name;
        போது (std::getline(ss, name, ',')) {
            auto it = tensor_map->find(name);
            எனில் (it == tensor_map->end()) {
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](மாறிலி auto & entry) {
                    திரும்பு entry.first.compare(0, name.size(), name) == 0 && entry.first.size() > name.size() &&
                           entry.first[name.size()] == '#';
                });
            }
            எனில் (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                தொடர்;
            }
            auto result = std::make_shared<v0::Result>(it->second);
            result->set_friendly_name("__debug_" + it->first);
            results.push_back(result);
        }
    }

    ov::ParameterVector used_params;
    க்கு (மாறிலி auto & param : params) {
        எனில் (!param->output(0).get_target_inputs().empty()) {
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
    க்கு (auto & முனை : resulting_model->get_ordered_ops()) {
        எனில் (auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(முனை); cnst && ov::shape_size(cnst->get_shape()) >= 16) {
            auto & rt_info = cnst->get_rt_info();
            எனில் (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_size(), offset++, cnst->get_element_type());
            }
        }
    }
    திரும்பு resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    auto ggml_model_decoder = ov::as_type_ptr<உள்ளீடு_Model>(m_உள்ளீடு_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(உண்மை);
        manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
        // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
        // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
        // so it survives ConstantFolding regardless of whether the target plugin's own
        // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
        manager.register_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
        manager.register_pass<pass::FuseToConv>();
        manager.register_pass<pass::Fuse_Argsort_உச்சி_K>();

        // MOECompressed has no CPU plugin implementation, so enable it by default only on GPU.
        // GGML_OPENVINO_MOE_OP=0 keeps the unfused GatherMatmul path for fallback/debugging.
        எனில் (ggml_openvino_is_gpu() &&
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
        எனில் (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
            manager.register_pass<pass::AlignEltwiseOperandRanks>();
        }

        எனில் (ggml_model_decoder->is_stateful()) {
            மாறிலி auto kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
            மாறிலி auto kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
            manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
            // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
            எனில் (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
                manager.register_pass<pass::KVStateSeqAxis>();
            }
        }

        எனில் (ggml_model_decoder->is_static()) {
            manager.register_pass<pass::SqueezeMatmul>();
        }
        manager.run_passes(model);
        எனில் (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            க்கு (size_t i = 0; i < model->get_output_size(); i++) {
                auto model_output_shape = model->output(i).get_partial_shape();
                எனில் (model_output_shape.rank().is_static() && model_output_shape.rank().get_length() == 3) {
                    ppp.output(i).postprocess().custom([](மாறிலி ov::Output<ov::Node>& முனை) {
                        auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        திரும்பு std::make_shared<ov::op::v0::Unsqueeze>(முனை, axes);
                    });
                }
            }
            model = ppp.build();
        }
    }
    திரும்பு model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
