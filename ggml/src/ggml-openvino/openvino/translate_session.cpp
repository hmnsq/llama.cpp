#అన్తర్భావ "translate_session.h"

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-openvino/ggml-openvino-extra.h"
#అన్తర్భావ "ggml-openvino/openvino/node_context.h"
#అన్తర్భావ "ggml-openvino/openvino/utils.h"
#అన్తర్భావ "input_model.h"
#అన్తర్భావ "pass/fuse_argsort_topk.h"
#అన్తర్భావ "pass/fuse_moe_router.h"
#అన్తర్భావ "pass/align_eltwise_ranks.h"
#అన్తర్భావ "pass/fuse_moe_compressed.h"
#అన్తర్భావ "pass/fuse_to_conv.h"
#అన్తర్భావ "pass/kv_state_seq_axis.h"
#అన్తర్భావ "pass/mark_decompression_convert_constant_folding.h"
#అన్తర్భావ "pass/mark_dequantization_subgraph.h"
#అన్తర్భావ "pass/squeeze_matmul.h"
#అన్తర్భావ "rt_info/weightless_caching_attributes.hpp"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <map>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/preprocess/pre_post_process.hpp>
#అన్తర్భావ <openvino/core/shape.hpp>
#అన్తర్భావ <openvino/core/type.hpp>
#అన్తర్భావ <openvino/core/type/element_type.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/convert_like.hpp>
#అన్తర్భావ <openvino/op/cos.hpp>
#అన్తర్భావ <openvino/op/divide.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/greater_eq.hpp>
#అన్తర్భావ <openvino/op/less.hpp>
#అన్తర్భావ <openvino/op/logical_and.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/parameter.hpp>
#అన్తర్భావ <openvino/op/range.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/result.hpp>
#అన్తర్భావ <openvino/op/select.hpp>
#అన్తర్భావ <openvino/op/sin.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/strided_slice.hpp>
#అన్తర్భావ <openvino/op/subtract.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <openvino/pass/constant_folding.hpp>
#అన్తర్భావ <openvino/pass/make_stateful.hpp>
#అన్తర్భావ <limits>
#అన్తర్భావ <sstream>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {

ఉపయోగించి నామాకాశ ov::op;

నామాకాశ {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(స్థిరరాశి std::string & name,
                                                        స్థిరరాశి Model_ఆదానం_Info & ఆదానం_info) {
    స్వయంచాలక param_కణుపు = std::make_shared<ov::op::v0::Parameter>(ఆదానం_info.type, ఆదానం_info.shape);
    param_కణుపు->సమితి_friendly_name(name);
    param_కణుపు->output(0).get_tensor().సమితి_పేర్లు({name});
    తిరిగి param_కణుపు;
}

std::shared_ptr<ov::Node> create_extra_ఆదానం(స్థిరరాశి std::string & name, స్థిరరాశి Model_Extra_ఆదానం_Info & ఆదానం_info) {
    ఒకవేళ (ఆదానం_info.is_parameter) {
        స్వయంచాలక param_కణుపు = std::make_shared<ov::op::v0::Parameter>(ఆదానం_info.type, ఆదానం_info.shape);
        param_కణుపు->సమితి_friendly_name(name);
        param_కణుపు->output(0).get_tensor().సమితి_పేర్లు({name});
        తిరిగి param_కణుపు;
    }

    స్వయంచాలక constant = std::make_shared<ov::op::v0::Constant>(ఆదానం_info.type, ఆదానం_info.shape,
                                                          std::vector<int64_t>{ఆదానం_info.value});
    constant->సమితి_friendly_name(name);
    తిరిగి constant;
}

ov::pass::MakeStateful::Param_Res_జతలు get_kv_param_res_జతలు(
    స్థిరరాశి std::shared_ptr<ov::Model> & model,
    స్థిరరాశి std::map<std::string, std::string> & kv_param_res_పేర్లు) {
    ov::pass::MakeStateful::Param_Res_జతలు జతలు;
    స్థిరరాశి స్వయంచాలక & params = model->get_parameters();
    స్థిరరాశి స్వయంచాలక & results = model->get_results();

    కొరకు (స్థిరరాశి స్వయంచాలక & param_res : kv_param_res_పేర్లు) {
        స్థిరరాశి స్వయంచాలక & param_name = param_res.మొదటి;
        స్థిరరాశి స్వయంచాలక & res_name = param_res.second;

        స్వయంచాలక param_it = std::find_if(params.begin(), params.end(), [&](స్థిరరాశి std::shared_ptr<v0::Parameter> & కణుపు) {
            తిరిగి కణుపు->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        స్వయంచాలక res_it = std::find_if(results.begin(), results.end(), [&](స్థిరరాశి std::shared_ptr<v0::ఫలితం_2> & కణుపు) {
            తిరిగి కణుపు->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::ఫలితం_2> res = *res_it;
        జతలు.emplace_back(param, res);
    }
    తిరిగి జతలు;
}

శూన్య కూడు_sliced_ముసుగు_stateful(TensorMap & tensor_map, తార్కిక imrope) {
    స్వయంచాలక create_sliced_ముసుగు = [&](స్థిరరాశి std::string & ముసుగు_name, స్థిరరాశి std::string & sliced_name) {
        ఒకవేళ ((tensor_map.find(ముసుగు_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end()) &&
            (tensor_map.find("inp_pos") != tensor_map.end())) {
            స్వయంచాలక token_len_చొప్పున_seq = tensor_map.at("token_len_per_seq").get_కణుపు_shared_ptr();
            స్వయంచాలక ముసుగు = tensor_map.at(ముసుగు_name).get_కణుపు_shared_ptr();
            std::shared_ptr<ov::Node> ముసుగు_sliced = ముసుగు;
            స్వయంచాలక one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            స్వయంచాలక zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            స్వయంచాలక three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            స్వయంచాలక neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            స్వయంచాలక అడుగు = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            స్వయంచాలక axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            స్వయంచాలక inp_pos = tensor_map.at("inp_pos").get_కణుపు_shared_ptr();
            // IMROPE's fourth position plane is zero for text; use the token's first plane.
            ov::Output<ov::Node> చివరి_సూచీ = neg_one;
            ఒకవేళ (imrope) {
                చివరి_సూచీ = std::make_shared<ov::op::v1::కూడు>(token_len_చొప్పున_seq, neg_one);
            }
            స్వయంచాలక చివరి_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, చివరి_సూచీ, three);
            స్వయంచాలక చివరి_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                చివరి_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), అబద్ధం);
            స్వయంచాలక చివరి_inp_pos_cvt = std::make_shared<ov::op::v0::Convert>(చివరి_inp_pos_1d, ov::element::i64);
            స్వయంచాలక చివరి_inp_pos_inc = std::make_shared<ov::op::v1::కూడు>(చివరి_inp_pos_cvt, one);

            ముసుగు_sliced = std::make_shared<ov::op::v8::ముక్క>(ముసుగు, zero, చివరి_inp_pos_inc, అడుగు, axes);
            ముసుగు_sliced = std::make_shared<ov::op::v0::Convert>(ముసుగు_sliced, ov::element::f16);
            ముసుగు_sliced->సమితి_friendly_name(sliced_name);

            tensor_map.insert({sliced_name, ముసుగు_sliced->output(0)});
        }
    };

    create_sliced_ముసుగు("self_kq_mask", "KQ_mask_sliced");
    create_sliced_ముసుగు("KQ_mask", "KQ_mask_sliced");
    create_sliced_ముసుగు("self_kq_mask_swa", "KQ_mask_swa_sliced");
}

// Rebuild the sliding-window mask from absolute positions.
// ggml caps self_kq_mask_swa at the size of its own SWA cache, but the stateful KV state is
// Concat-appended and grows without bound, so past that cap the two disagree on length and the
// mask add fails. A pure-Concat state is ordered by position, so positions can rebuild the mask.
// swa_window holds the real n_swa, read back from the ggml mask in ggml-decoder.cpp.
// No-op when the graph has no SWA mask, or when the window could not be read back.
శూన్య కూడు_position_ముసుగు_stateful_swa(TensorMap & tensor_map) {
    ఒకవేళ (tensor_map.find("self_kq_mask_swa") == tensor_map.end() || tensor_map.find("inp_pos") == tensor_map.end() ||
        tensor_map.find("swa_window") == tensor_map.end()) {
        తిరిగి;
    }

    స్వయంచాలక inp_pos = tensor_map.at("inp_pos").get_కణుపు_shared_ptr();

    స్వయంచాలక zero_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    స్వయంచాలక one_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    స్వయంచాలక three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    స్వయంచాలక neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    స్వయంచాలక query_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::i64);
    స్వయంచాలక query_pos_1d = std::make_shared<ov::op::v1::Reshape>(
        query_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), అబద్ధం);

    స్వయంచాలక చివరి_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
    స్వయంచాలక చివరి_pos_1d = std::make_shared<ov::op::v1::Reshape>(చివరి_pos, one_i64, అబద్ధం);
    స్వయంచాలక చివరి_pos_cvt = std::make_shared<ov::op::v0::Convert>(చివరి_pos_1d, ov::element::i64);
    స్వయంచాలక మొత్తం_len = std::make_shared<ov::op::v1::కూడు>(చివరి_pos_cvt, one_i64);
    స్వయంచాలక మొత్తం_len_scalar = std::make_shared<ov::op::v0::Squeeze>(మొత్తం_len);

    స్వయంచాలక cached_pos = std::make_shared<ov::op::v4::Range>(
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}), మొత్తం_len_scalar,
        ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);

    స్వయంచాలక query_నిలువు = std::make_shared<ov::op::v1::Reshape>(
        query_pos_1d, ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1}), అబద్ధం);
    స్వయంచాలక cached_అడ్డువరుస = std::make_shared<ov::op::v1::Reshape>(
        cached_pos, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, -1}), అబద్ధం);
    స్వయంచాలక diff = std::make_shared<ov::op::v1::వ్యవకలనం>(query_నిలువు, cached_అడ్డువరుస);

    స్వయంచాలక swa_కిటికీ = tensor_map.at("swa_window").get_కణుపు_shared_ptr();
    స్వయంచాలక కిటికీ = std::make_shared<ov::op::v0::Convert>(swa_కిటికీ, ov::element::i64);
    స్వయంచాలక causal_ok = std::make_shared<ov::op::v1::GreaterEqual>(diff, zero_i64);
    స్వయంచాలక కిటికీ_ok = std::make_shared<ov::op::v1::Less>(diff, కిటికీ);
    స్వయంచాలక keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_ok, కిటికీ_ok);

    స్వయంచాలక zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    స్వయంచాలక neg_inf_f = ov::op::v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<దశాంశం>::infinity()});
    std::shared_ptr<ov::Node> ముసుగు = std::make_shared<ov::op::v1::Select>(keep, zero_f, neg_inf_f);

    స్వయంచాలక batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    ముసుగు = std::make_shared<ov::op::v0::Unsqueeze>(ముసుగు, batch_axis);
    ముసుగు = std::make_shared<ov::op::v0::Unsqueeze>(ముసుగు, batch_axis);
    ముసుగు = std::make_shared<ov::op::v0::Convert>(ముసుగు, ov::element::f16);
    ముసుగు->సమితి_friendly_name("KQ_mask_swa_sliced");

    tensor_map["KQ_mask_swa_sliced"] = ముసుగు->output(0);
}

శూన్య కూడు_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    ఒకవేళ (ggml_model_decoder.has_mixed_rope_params()) {
        తిరిగి;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    ఒకవేళ (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == శూన్యసూచిక) {
        తిరిగి;
    }
    స్వయంచాలక inp_pos = tensor_map.at("inp_pos").get_కణుపు_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_బరువు;
    ఒకవేళ (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_బరువు = tensor_map.at("rope_freqs.weight").get_కణుపు_shared_ptr();
    }

    స్వయంచాలక sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_బరువు);
    స్వయంచాలక sin_theta = sin_cos.మొదటి;
    స్వయంచాలక cos_theta = sin_cos.second;

    cos_theta.get_కణుపు_shared_ptr()->సమితి_friendly_name("rope_cos");
    sin_theta.get_కణుపు_shared_ptr()->సమితి_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
శూన్య preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    ఒకవేళ (ggml_model_decoder.is_stateful()) {
        కూడు_sliced_ముసుగు_stateful(tensor_map, ggml_model_decoder.get_rope_params()[2] == GGML_ROPE_TYPE_IMROPE);
        కూడు_position_ముసుగు_stateful_swa(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(స్థిరరాశి frontend::ఆదానం_Model::Ptr & ఆదానం_model,
                                   స్థిరరాశి std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   తార్కిక naive) :
    m_ఆదానం_model(ఆదానం_model),
    m_translator_map(translator_map),
    m_ov_model(శూన్యసూచిక),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    ఒకవేళ (m_ov_model) {
        తిరిగి m_ov_model;
    }
    m_ov_model = translate_లేఖాచిత్రం(m_ఆదానం_model);
    తిరిగి m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_లేఖాచిత్రం(స్థిరరాశి frontend::ఆదానం_Model::Ptr & ఆదానం_model) {
    ov::Parameter_సదిశ params;
    ov::ఫలితం_సదిశ results;
    స్వయంచాలక tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    స్థిరరాశి స్వయంచాలక & ggml_model = ov::as_type_ptr<ఆదానం_Model>(ఆదానం_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    కొరకు (స్థిరరాశి స్వయంచాలక & it : ggml_model_decoder->get_model_inputs()) {
        స్వయంచాలక param_కణుపు = create_parameter(it.మొదటి, it.second);
        params.push_back(param_కణుపు);
        (*tensor_map)[it.మొదటి] = param_కణుపు;
    }

    కొరకు (స్థిరరాశి స్వయంచాలక & it : ggml_model_decoder->get_model_extra_inputs()) {
        స్వయంచాలక ఆదానం_కణుపు = create_extra_ఆదానం(it.మొదటి, it.second);
        ఒకవేళ (it.second.is_parameter) {
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(ఆదానం_కణుపు));
        }
        (*tensor_map)[it.మొదటి] = ఆదానం_కణుపు;
    }

    కొరకు (స్థిరరాశి స్వయంచాలక & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.మొదటి] = it.second;
    }

    స్వయంచాలక translate_కణుపు = [&](స్థిరరాశి std::shared_ptr<GgmlDecoder> & decoder, పూర్ణాంకం కణుపు_idx) {
        స్వయంచాలక operation_type = decoder->get_op_type(కణుపు_idx);
        ఒకవేళ (operation_type == "GGML_OP_NONE") {
            తిరిగి ov::Output_సదిశ{};
        }

        స్థిరరాశి స్వయంచాలక & కణుపు_output_పేర్లు = decoder->get_output_పేర్లు(కణుపు_idx);
        ఒకవేళ (operation_type == "GGML_OP_VIEW" && decoder->get_op_case(కణుపు_idx) == 2 && కణుపు_output_పేర్లు.size() == 1) {
            స్వయంచాలక ప్రత్యక్ష_output = tensor_map->find(కణుపు_output_పేర్లు[0]);
            ఒకవేళ (ప్రత్యక్ష_output != tensor_map->end()) {
                // GDN publishes its native attention/state outputs under the two GGML VIEW names.
                // Keep those mappings instead of rebuilding slices of a packed temporary.
                తిరిగి ov::Output_సదిశ{ప్రత్యక్ష_output->second};
            }
        }

        స్వయంచాలక it = m_translator_map.find(operation_type);
        FRONT_చివర_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        కణుపు_Context కణుపు_context(decoder, tensor_map, కణుపు_idx, ఇది);
        ov::Output_సదిశ converted_outputs = it->second(కణుపు_context);

        FRONT_చివర_OP_CONVERSION_CHECK(కణుపు_output_పేర్లు.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      కణుపు_output_పేర్లు.size(), " and ", converted_outputs.size(), " respectively.");

        కొరకు (size_t i = 0; i < కణుపు_output_పేర్లు.size(); ++i) {
            స్వయంచాలక output_name = కణుపు_output_పేర్లు[i];
            ఒకవేళ (i < converted_outputs.size() && converted_outputs[i].get_కణుపు_shared_ptr() != శూన్యసూచిక) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
        తిరిగి converted_outputs;
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
    స్వయంచాలక refresh_inplace_aliases = [&](స్థిరరాశి std::shared_ptr<GgmlDecoder> & decoder, పూర్ణాంకం inplace_కణుపు_idx,
                                       స్థిరరాశి std::string & view_src_name) {
        కొరకు (పూర్ణాంకం కణుపు_idx = 0; కణుపు_idx < inplace_కణుపు_idx; కణుపు_idx++) {
            ఒకవేళ (decoder->is_view_like_alias_యొక్క(కణుపు_idx, view_src_name)) {
                translate_కణుపు(decoder, కణుపు_idx);
            }
        }
    };

    స్వయంచాలక కణుపు_visitor = [&](స్థిరరాశి std::shared_ptr<GgmlDecoder> & decoder, పూర్ణాంకం కణుపు_idx) {
        స్వయంచాలక converted_outputs = translate_కణుపు(decoder, కణుపు_idx);
        ఒకవేళ (converted_outputs.ఖాళీలు()) {
            తిరిగి;
        }
        స్థిరరాశి స్వయంచాలక inplace_src = decoder->get_inplace_op_src(కణుపు_idx);
        ఒకవేళ (inplace_src.ఖాళీలు()) {
            తిరిగి;
        }
        ఒకవేళ (converted_outputs[0].get_కణుపు_shared_ptr() != శూన్యసూచిక) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, కణుపు_idx, inplace_src);
    };

    ఒకవేళ (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->సందర్శించు_subgraph(కణుపు_visitor);

    కొరకు (స్థిరరాశి స్వయంచాలక & name : ggml_model_decoder->get_model_output_పేర్లు()) {
        FRONT_చివర_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        స్వయంచాలక ఫలితం = std::make_shared<v0::ఫలితం_2>(tensor_map->at(name));
        ఫలితం->సమితి_friendly_name(name);
        results.push_back(ఫలితం);
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
    ఒకవేళ (స్థిరరాశి అక్షరం * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        std::stringstream ss(debug_nodes);
        std::string name;
        అయితే (std::getline(ss, name, ',')) {
            స్వయంచాలక it = tensor_map->find(name);
            ఒకవేళ (it == tensor_map->end()) {
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](స్థిరరాశి స్వయంచాలక & నమోదు_2) {
                    తిరిగి నమోదు_2.మొదటి.compare(0, name.size(), name) == 0 && నమోదు_2.మొదటి.size() > name.size() &&
                           నమోదు_2.మొదటి[name.size()] == '#';
                });
            }
            ఒకవేళ (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                కొనసాగు;
            }
            స్వయంచాలక ఫలితం = std::make_shared<v0::ఫలితం_2>(it->second);
            ఫలితం->సమితి_friendly_name("__debug_" + it->మొదటి);
            results.push_back(ఫలితం);
        }
    }

    ov::Parameter_సదిశ వాడిన_params;
    కొరకు (స్థిరరాశి స్వయంచాలక & param : params) {
        ఒకవేళ (!param->output(0).get_లక్ష్యం_inputs().ఖాళీలు()) {
            వాడిన_params.push_back(param);
        }
    }
    // if (auto diff = params.size() - used_params.size()) {
    //     GGML_LOG_INFO("%zu parameters are not used in the model.", diff);
    // }
    resulting_model = std::make_shared<Model>(results, వాడిన_params);

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
    size_t స్థానభ్రంశం = 0;
    కొరకు (స్వయంచాలక & కణుపు : resulting_model->get_ordered_ops()) {
        ఒకవేళ (స్వయంచాలక cnst = ov::as_type_ptr<ov::op::v0::Constant>(కణుపు); cnst && ov::shape_కొలత(cnst->get_shape()) >= 16) {
            స్వయంచాలక & rt_info = cnst->get_rt_info();
            ఒకవేళ (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_కొలత(), స్థానభ్రంశం++, cnst->get_element_type());
            }
        }
    }
    తిరిగి resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    స్వయంచాలక ggml_model_decoder = ov::as_type_ptr<ఆదానం_Model>(m_ఆదానం_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.సమితి_చొప్పున_pass_validation(నిజం);
        manager.రిజిస్టరు_pass<ov::pass::MarkCompressedFloatConstants>();
        // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
        // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
        // so it survives ConstantFolding regardless of whether the target plugin's own
        // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
        manager.రిజిస్టరు_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
        manager.రిజిస్టరు_pass<pass::Fuse_గమ్యం_Conv>();
        manager.రిజిస్టరు_pass<pass::Fuse_Argsort_శిఖరం_K>();

        // MOECompressed has no CPU plugin implementation, so enable it by default only on GPU.
        // GGML_OPENVINO_MOE_OP=0 keeps the unfused GatherMatmul path for fallback/debugging.
        ఒకవేళ (ggml_openvino_is_gpu() &&
            ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) != 0) {
            manager.రిజిస్టరు_pass<pass::Fuse_Moe_మార్గనిర్దేశకం>();
            manager.రిజిస్టరు_pass<pass::FuseMoeCompressed>();
            // Same fusion for models whose gate and up projections share one fused expert
            // weight (gemma-4). It only matches that shape and only when the experts carry an
            // integer zero point, so it is a no-op on the separate-gate/up models above.
            manager.రిజిస్టరు_pass<pass::Fuse_Moe_Compressed_Fused_Gate_పైకి>();
        }

        // Workaround for an OpenVINO GPU-plugin defect: an eltwise op whose operands differ in
        // rank is computed wrongly once the plugin fuses it as a post-op into `rms`. gemma-4
        // dense under stateful execution adds a rank-3 residual to the rank-4 norm output and
        // decodes garbage on GPU while CPU is correct. Equalising the ranks is a no-op for
        // NUMPY broadcasting and makes the fused path correct.
        // Remove once the plugin guards that fusion. Opt out with
        // GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN=1.
        ఒకవేళ (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
            manager.రిజిస్టరు_pass<pass::Align_Eltwise_నిర్వాహ్యం_Ranks>();
        }

        ఒకవేళ (ggml_model_decoder->is_stateful()) {
            స్థిరరాశి స్వయంచాలక kv_param_res_పేర్లు = ggml_model_decoder->get_kv_param_res_పేర్లు();
            స్థిరరాశి స్వయంచాలక kv_param_res_జతలు = get_kv_param_res_జతలు(model, kv_param_res_పేర్లు);
            manager.రిజిస్టరు_pass<ov::pass::MakeStateful>(kv_param_res_జతలు);
            // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
            ఒకవేళ (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
                manager.రిజిస్టరు_pass<pass::KVStateSeqAxis>();
            }
        }

        ఒకవేళ (ggml_model_decoder->is_static()) {
            manager.రిజిస్టరు_pass<pass::SqueezeMatmul>();
        }
        manager.నిరంతరం_passes(model);
        ఒకవేళ (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            కొరకు (size_t i = 0; i < model->get_output_కొలత(); i++) {
                స్వయంచాలక model_output_shape = model->output(i).get_partial_shape();
                ఒకవేళ (model_output_shape.rank().is_static() && model_output_shape.rank().get_నిడివి() == 3) {
                    ppp.output(i).postprocess().custom([](స్థిరరాశి ov::Output<ov::Node>& కణుపు) {
                        స్వయంచాలక axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        తిరిగి std::make_shared<ov::op::v0::Unsqueeze>(కణుపు, axes);
                    });
                }
            }
            model = ppp.build();
        }
    }
    తిరిగి model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
