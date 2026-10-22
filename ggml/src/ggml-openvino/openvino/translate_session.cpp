#ಅನ್ತರ್ಭಾವ "translate_session.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino/ggml-openvino-extra.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino/openvino/node_context.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino/openvino/utils.h"
#ಅನ್ತರ್ಭಾವ "input_model.h"
#ಅನ್ತರ್ಭಾವ "pass/fuse_argsort_topk.h"
#ಅನ್ತರ್ಭಾವ "pass/fuse_moe_router.h"
#ಅನ್ತರ್ಭಾವ "pass/align_eltwise_ranks.h"
#ಅನ್ತರ್ಭಾವ "pass/fuse_moe_compressed.h"
#ಅನ್ತರ್ಭಾವ "pass/fuse_to_conv.h"
#ಅನ್ತರ್ಭಾವ "pass/kv_state_seq_axis.h"
#ಅನ್ತರ್ಭಾವ "pass/mark_decompression_convert_constant_folding.h"
#ಅನ್ತರ್ಭಾವ "pass/mark_dequantization_subgraph.h"
#ಅನ್ತರ್ಭಾವ "pass/squeeze_matmul.h"
#ಅನ್ತರ್ಭಾವ "rt_info/weightless_caching_attributes.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/preprocess/pre_post_process.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/shape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type/element_type.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert_like.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/cos.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/divide.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/greater_eq.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/less.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/logical_and.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/parameter.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/range.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/result.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/select.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/sin.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/strided_slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/subtract.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/constant_folding.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/make_stateful.hpp>
#ಅನ್ತರ್ಭಾವ <limits>
#ಅನ್ತರ್ಭಾವ <sstream>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {

ಬಳಸಿ ನಾಮಾಕಾಶ ov::op;

ನಾಮಾಕಾಶ {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(ಸ್ಥಿರಾಂಕ std::string & name,
                                                        ಸ್ಥಿರಾಂಕ Model_ಒಳಹರಿವು_Info & ಒಳಹರಿವು_info) {
    ಸ್ವಯಂಚಾಲಿತ param_ಗಂಟು = std::make_shared<ov::op::v0::Parameter>(ಒಳಹರಿವು_info.type, ಒಳಹರಿವು_info.shape);
    param_ಗಂಟು->ಗಣ_friendly_name(name);
    param_ಗಂಟು->output(0).get_tensor().ಗಣ_ಹೆಸರುಗಳು({name});
    ಹಿಂತಿರುಗಿ param_ಗಂಟು;
}

std::shared_ptr<ov::Node> create_extra_ಒಳಹರಿವು(ಸ್ಥಿರಾಂಕ std::string & name, ಸ್ಥಿರಾಂಕ Model_Extra_ಒಳಹರಿವು_Info & ಒಳಹರಿವು_info) {
    ಆದರೆ (ಒಳಹರಿವು_info.is_parameter) {
        ಸ್ವಯಂಚಾಲಿತ param_ಗಂಟು = std::make_shared<ov::op::v0::Parameter>(ಒಳಹರಿವು_info.type, ಒಳಹರಿವು_info.shape);
        param_ಗಂಟು->ಗಣ_friendly_name(name);
        param_ಗಂಟು->output(0).get_tensor().ಗಣ_ಹೆಸರುಗಳು({name});
        ಹಿಂತಿರುಗಿ param_ಗಂಟು;
    }

    ಸ್ವಯಂಚಾಲಿತ constant = std::make_shared<ov::op::v0::Constant>(ಒಳಹರಿವು_info.type, ಒಳಹರಿವು_info.shape,
                                                          std::vector<int64_t>{ಒಳಹರಿವು_info.value});
    constant->ಗಣ_friendly_name(name);
    ಹಿಂತಿರುಗಿ constant;
}

ov::pass::MakeStateful::Param_Res_ಜೋಡಿಗಳು get_kv_param_res_ಜೋಡಿಗಳು(
    ಸ್ಥಿರಾಂಕ std::shared_ptr<ov::Model> & model,
    ಸ್ಥಿರಾಂಕ std::map<std::string, std::string> & kv_param_res_ಹೆಸರುಗಳು) {
    ov::pass::MakeStateful::Param_Res_ಜೋಡಿಗಳು ಜೋಡಿಗಳು;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & params = model->get_parameters();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & results = model->get_results();

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & param_res : kv_param_res_ಹೆಸರುಗಳು) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & param_name = param_res.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & res_name = param_res.second;

        ಸ್ವಯಂಚಾಲಿತ param_it = std::find_if(params.begin(), params.end(), [&](ಸ್ಥಿರಾಂಕ std::shared_ptr<v0::Parameter> & ಗಂಟು) {
            ಹಿಂತಿರುಗಿ ಗಂಟು->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        ಸ್ವಯಂಚಾಲಿತ res_it = std::find_if(results.begin(), results.end(), [&](ಸ್ಥಿರಾಂಕ std::shared_ptr<v0::ಫಲಿತಾಂಶ_2> & ಗಂಟು) {
            ಹಿಂತಿರುಗಿ ಗಂಟು->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::ಫಲಿತಾಂಶ_2> res = *res_it;
        ಜೋಡಿಗಳು.emplace_back(param, res);
    }
    ಹಿಂತಿರುಗಿ ಜೋಡಿಗಳು;
}

ಶೂನ್ಯ ಕೂಡಿಸು_sliced_ಮುಖವಾಡ_stateful(TensorMap & tensor_map, ತಾರ್ಕಿಕ imrope) {
    ಸ್ವಯಂಚಾಲಿತ create_sliced_ಮುಖವಾಡ = [&](ಸ್ಥಿರಾಂಕ std::string & ಮುಖವಾಡ_name, ಸ್ಥಿರಾಂಕ std::string & sliced_name) {
        ಆದರೆ ((tensor_map.find(ಮುಖವಾಡ_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end()) &&
            (tensor_map.find("inp_pos") != tensor_map.end())) {
            ಸ್ವಯಂಚಾಲಿತ token_len_ಪ್ರತಿ_seq = tensor_map.at("token_len_per_seq").get_ಗಂಟು_shared_ptr();
            ಸ್ವಯಂಚಾಲಿತ ಮುಖವಾಡ = tensor_map.at(ಮುಖವಾಡ_name).get_ಗಂಟು_shared_ptr();
            std::shared_ptr<ov::Node> ಮುಖವಾಡ_sliced = ಮುಖವಾಡ;
            ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            ಸ್ವಯಂಚಾಲಿತ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            ಸ್ವಯಂಚಾಲಿತ three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ಸ್ವಯಂಚಾಲಿತ neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            ಸ್ವಯಂಚಾಲಿತ axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            ಸ್ವಯಂಚಾಲಿತ inp_pos = tensor_map.at("inp_pos").get_ಗಂಟು_shared_ptr();
            // IMROPE's fourth position plane is zero for text; use the token's first plane.
            ov::Output<ov::Node> ಕೊನೆಯ_ಸೂಚ್ಯಂಕ = neg_one;
            ಆದರೆ (imrope) {
                ಕೊನೆಯ_ಸೂಚ್ಯಂಕ = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(token_len_ಪ್ರತಿ_seq, neg_one);
            }
            ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, ಕೊನೆಯ_ಸೂಚ್ಯಂಕ, three);
            ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                ಕೊನೆಯ_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), ಸುಳ್ಳು);
            ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_inp_pos_cvt = std::make_shared<ov::op::v0::Convert>(ಕೊನೆಯ_inp_pos_1d, ov::element::i64);
            ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_inp_pos_inc = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(ಕೊನೆಯ_inp_pos_cvt, one);

            ಮುಖವಾಡ_sliced = std::make_shared<ov::op::v8::ತುಂಡು>(ಮುಖವಾಡ, zero, ಕೊನೆಯ_inp_pos_inc, ಹೆಜ್ಜೆ, axes);
            ಮುಖವಾಡ_sliced = std::make_shared<ov::op::v0::Convert>(ಮುಖವಾಡ_sliced, ov::element::f16);
            ಮುಖವಾಡ_sliced->ಗಣ_friendly_name(sliced_name);

            tensor_map.insert({sliced_name, ಮುಖವಾಡ_sliced->output(0)});
        }
    };

    create_sliced_ಮುಖವಾಡ("self_kq_mask", "KQ_mask_sliced");
    create_sliced_ಮುಖವಾಡ("KQ_mask", "KQ_mask_sliced");
    create_sliced_ಮುಖವಾಡ("self_kq_mask_swa", "KQ_mask_swa_sliced");
}

// Rebuild the sliding-window mask from absolute positions.
// ggml caps self_kq_mask_swa at the size of its own SWA cache, but the stateful KV state is
// Concat-appended and grows without bound, so past that cap the two disagree on length and the
// mask add fails. A pure-Concat state is ordered by position, so positions can rebuild the mask.
// swa_window holds the real n_swa, read back from the ggml mask in ggml-decoder.cpp.
// No-op when the graph has no SWA mask, or when the window could not be read back.
ಶೂನ್ಯ ಕೂಡಿಸು_position_ಮುಖವಾಡ_stateful_swa(TensorMap & tensor_map) {
    ಆದರೆ (tensor_map.find("self_kq_mask_swa") == tensor_map.end() || tensor_map.find("inp_pos") == tensor_map.end() ||
        tensor_map.find("swa_window") == tensor_map.end()) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ವಯಂಚಾಲಿತ inp_pos = tensor_map.at("inp_pos").get_ಗಂಟು_shared_ptr();

    ಸ್ವಯಂಚಾಲಿತ zero_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    ಸ್ವಯಂಚಾಲಿತ one_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ಸ್ವಯಂಚಾಲಿತ three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    ಸ್ವಯಂಚಾಲಿತ neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    ಸ್ವಯಂಚಾಲಿತ query_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ query_pos_1d = std::make_shared<ov::op::v1::Reshape>(
        query_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), ಸುಳ್ಳು);

    ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
    ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_pos_1d = std::make_shared<ov::op::v1::Reshape>(ಕೊನೆಯ_pos, one_i64, ಸುಳ್ಳು);
    ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_pos_cvt = std::make_shared<ov::op::v0::Convert>(ಕೊನೆಯ_pos_1d, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ ಒಟ್ಟು_len = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(ಕೊನೆಯ_pos_cvt, one_i64);
    ಸ್ವಯಂಚಾಲಿತ ಒಟ್ಟು_len_scalar = std::make_shared<ov::op::v0::Squeeze>(ಒಟ್ಟು_len);

    ಸ್ವಯಂಚಾಲಿತ cached_pos = std::make_shared<ov::op::v4::Range>(
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}), ಒಟ್ಟು_len_scalar,
        ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);

    ಸ್ವಯಂಚಾಲಿತ query_ಲಂಬಸಾಲು = std::make_shared<ov::op::v1::Reshape>(
        query_pos_1d, ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1}), ಸುಳ್ಳು);
    ಸ್ವಯಂಚಾಲಿತ cached_ಅಡ್ಡಸಾಲು = std::make_shared<ov::op::v1::Reshape>(
        cached_pos, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, -1}), ಸುಳ್ಳು);
    ಸ್ವಯಂಚಾಲಿತ diff = std::make_shared<ov::op::v1::ಕಳೆ>(query_ಲಂಬಸಾಲು, cached_ಅಡ್ಡಸಾಲು);

    ಸ್ವಯಂಚಾಲಿತ swa_ಕಿಟಕಿ = tensor_map.at("swa_window").get_ಗಂಟು_shared_ptr();
    ಸ್ವಯಂಚಾಲಿತ ಕಿಟಕಿ = std::make_shared<ov::op::v0::Convert>(swa_ಕಿಟಕಿ, ov::element::i64);
    ಸ್ವಯಂಚಾಲಿತ causal_ok = std::make_shared<ov::op::v1::GreaterEqual>(diff, zero_i64);
    ಸ್ವಯಂಚಾಲಿತ ಕಿಟಕಿ_ok = std::make_shared<ov::op::v1::Less>(diff, ಕಿಟಕಿ);
    ಸ್ವಯಂಚಾಲಿತ keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_ok, ಕಿಟಕಿ_ok);

    ಸ್ವಯಂಚಾಲಿತ zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    ಸ್ವಯಂಚಾಲಿತ neg_inf_f = ov::op::v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<ದಶಮಾಂಶ>::infinity()});
    std::shared_ptr<ov::Node> ಮುಖವಾಡ = std::make_shared<ov::op::v1::Select>(keep, zero_f, neg_inf_f);

    ಸ್ವಯಂಚಾಲಿತ batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    ಮುಖವಾಡ = std::make_shared<ov::op::v0::Unsqueeze>(ಮುಖವಾಡ, batch_axis);
    ಮುಖವಾಡ = std::make_shared<ov::op::v0::Unsqueeze>(ಮುಖವಾಡ, batch_axis);
    ಮುಖವಾಡ = std::make_shared<ov::op::v0::Convert>(ಮುಖವಾಡ, ov::element::f16);
    ಮುಖವಾಡ->ಗಣ_friendly_name("KQ_mask_swa_sliced");

    tensor_map["KQ_mask_swa_sliced"] = ಮುಖವಾಡ->output(0);
}

ಶೂನ್ಯ ಕೂಡಿಸು_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    ಆದರೆ (ggml_model_decoder.has_mixed_rope_params()) {
        ಹಿಂತಿರುಗಿ;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    ಆದರೆ (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ವಯಂಚಾಲಿತ inp_pos = tensor_map.at("inp_pos").get_ಗಂಟು_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_ತೂಕ;
    ಆದರೆ (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_ತೂಕ = tensor_map.at("rope_freqs.weight").get_ಗಂಟು_shared_ptr();
    }

    ಸ್ವಯಂಚಾಲಿತ sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_ತೂಕ);
    ಸ್ವಯಂಚಾಲಿತ sin_theta = sin_cos.ಮೊದಲ;
    ಸ್ವಯಂಚಾಲಿತ cos_theta = sin_cos.second;

    cos_theta.get_ಗಂಟು_shared_ptr()->ಗಣ_friendly_name("rope_cos");
    sin_theta.get_ಗಂಟು_shared_ptr()->ಗಣ_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
ಶೂನ್ಯ preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    ಆದರೆ (ggml_model_decoder.is_stateful()) {
        ಕೂಡಿಸು_sliced_ಮುಖವಾಡ_stateful(tensor_map, ggml_model_decoder.get_rope_params()[2] == GGML_ROPE_TYPE_IMROPE);
        ಕೂಡಿಸು_position_ಮುಖವಾಡ_stateful_swa(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(ಸ್ಥಿರಾಂಕ frontend::ಒಳಹರಿವು_Model::Ptr & ಒಳಹರಿವು_model,
                                   ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   ತಾರ್ಕಿಕ naive) :
    m_ಒಳಹರಿವು_model(ಒಳಹರಿವು_model),
    m_translator_map(translator_map),
    m_ov_model(ಶೂನ್ಯಸೂಚಕ),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    ಆದರೆ (m_ov_model) {
        ಹಿಂತಿರುಗಿ m_ov_model;
    }
    m_ov_model = translate_ನಕ್ಷೆ(m_ಒಳಹರಿವು_model);
    ಹಿಂತಿರುಗಿ m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_ನಕ್ಷೆ(ಸ್ಥಿರಾಂಕ frontend::ಒಳಹರಿವು_Model::Ptr & ಒಳಹರಿವು_model) {
    ov::Parameter_ಸದಿಶ params;
    ov::ಫಲಿತಾಂಶ_ಸದಿಶ results;
    ಸ್ವಯಂಚಾಲಿತ tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ggml_model = ov::as_type_ptr<ಒಳಹರಿವು_Model>(ಒಳಹರಿವು_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & it : ggml_model_decoder->get_model_inputs()) {
        ಸ್ವಯಂಚಾಲಿತ param_ಗಂಟು = create_parameter(it.ಮೊದಲ, it.second);
        params.push_back(param_ಗಂಟು);
        (*tensor_map)[it.ಮೊದಲ] = param_ಗಂಟು;
    }

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & it : ggml_model_decoder->get_model_extra_inputs()) {
        ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_ಗಂಟು = create_extra_ಒಳಹರಿವು(it.ಮೊದಲ, it.second);
        ಆದರೆ (it.second.is_parameter) {
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(ಒಳಹರಿವು_ಗಂಟು));
        }
        (*tensor_map)[it.ಮೊದಲ] = ಒಳಹರಿವು_ಗಂಟು;
    }

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.ಮೊದಲ] = it.second;
    }

    ಸ್ವಯಂಚಾಲಿತ translate_ಗಂಟು = [&](ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlDecoder> & decoder, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
        ಸ್ವಯಂಚಾಲಿತ operation_type = decoder->get_op_type(ಗಂಟು_idx);
        ಆದರೆ (operation_type == "GGML_OP_NONE") {
            ಹಿಂತಿರುಗಿ ov::Output_ಸದಿಶ{};
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಗಂಟು_output_ಹೆಸರುಗಳು = decoder->get_output_ಹೆಸರುಗಳು(ಗಂಟು_idx);
        ಆದರೆ (operation_type == "GGML_OP_VIEW" && decoder->get_op_case(ಗಂಟು_idx) == 2 && ಗಂಟು_output_ಹೆಸರುಗಳು.size() == 1) {
            ಸ್ವಯಂಚಾಲಿತ ನೇರ_output = tensor_map->find(ಗಂಟು_output_ಹೆಸರುಗಳು[0]);
            ಆದರೆ (ನೇರ_output != tensor_map->end()) {
                // GDN publishes its native attention/state outputs under the two GGML VIEW names.
                // Keep those mappings instead of rebuilding slices of a packed temporary.
                ಹಿಂತಿರುಗಿ ov::Output_ಸದಿಶ{ನೇರ_output->second};
            }
        }

        ಸ್ವಯಂಚಾಲಿತ it = m_translator_map.find(operation_type);
        FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        ಗಂಟು_Context ಗಂಟು_context(decoder, tensor_map, ಗಂಟು_idx, ಇದು);
        ov::Output_ಸದಿಶ converted_outputs = it->second(ಗಂಟು_context);

        FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಗಂಟು_output_ಹೆಸರುಗಳು.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      ಗಂಟು_output_ಹೆಸರುಗಳು.size(), " and ", converted_outputs.size(), " respectively.");

        ಗಾಗಿ (size_t i = 0; i < ಗಂಟು_output_ಹೆಸರುಗಳು.size(); ++i) {
            ಸ್ವಯಂಚಾಲಿತ output_name = ಗಂಟು_output_ಹೆಸರುಗಳು[i];
            ಆದರೆ (i < converted_outputs.size() && converted_outputs[i].get_ಗಂಟು_shared_ptr() != ಶೂನ್ಯಸೂಚಕ) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
        ಹಿಂತಿರುಗಿ converted_outputs;
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
    ಸ್ವಯಂಚಾಲಿತ refresh_inplace_aliases = [&](ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlDecoder> & decoder, ಪೂರ್ಣಾಂಕ inplace_ಗಂಟು_idx,
                                       ಸ್ಥಿರಾಂಕ std::string & view_src_name) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_idx = 0; ಗಂಟು_idx < inplace_ಗಂಟು_idx; ಗಂಟು_idx++) {
            ಆದರೆ (decoder->is_view_like_alias_ದ(ಗಂಟು_idx, view_src_name)) {
                translate_ಗಂಟು(decoder, ಗಂಟು_idx);
            }
        }
    };

    ಸ್ವಯಂಚಾಲಿತ ಗಂಟು_visitor = [&](ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlDecoder> & decoder, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
        ಸ್ವಯಂಚಾಲಿತ converted_outputs = translate_ಗಂಟು(decoder, ಗಂಟು_idx);
        ಆದರೆ (converted_outputs.ಖಾಲಿಗಳು()) {
            ಹಿಂತಿರುಗಿ;
        }
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ inplace_src = decoder->get_inplace_op_src(ಗಂಟು_idx);
        ಆದರೆ (inplace_src.ಖಾಲಿಗಳು()) {
            ಹಿಂತಿರುಗಿ;
        }
        ಆದರೆ (converted_outputs[0].get_ಗಂಟು_shared_ptr() != ಶೂನ್ಯಸೂಚಕ) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, ಗಂಟು_idx, inplace_src);
    };

    ಆದರೆ (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->ಭೇಟಿ_subgraph(ಗಂಟು_visitor);

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & name : ggml_model_decoder->get_model_output_ಹೆಸರುಗಳು()) {
        FRONT_ಕೊನೆ_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ = std::make_shared<v0::ಫಲಿತಾಂಶ_2>(tensor_map->at(name));
        ಫಲಿತಾಂಶ->ಗಣ_friendly_name(name);
        results.push_back(ಫಲಿತಾಂಶ);
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
    ಆದರೆ (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        std::stringstream ss(debug_nodes);
        std::string name;
        ಆಗ (std::getline(ss, name, ',')) {
            ಸ್ವಯಂಚಾಲಿತ it = tensor_map->find(name);
            ಆದರೆ (it == tensor_map->end()) {
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ನಮೂದು) {
                    ಹಿಂತಿರುಗಿ ನಮೂದು.ಮೊದಲ.compare(0, name.size(), name) == 0 && ನಮೂದು.ಮೊದಲ.size() > name.size() &&
                           ನಮೂದು.ಮೊದಲ[name.size()] == '#';
                });
            }
            ಆದರೆ (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                ಮುಂದುವರಿ;
            }
            ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ = std::make_shared<v0::ಫಲಿತಾಂಶ_2>(it->second);
            ಫಲಿತಾಂಶ->ಗಣ_friendly_name("__debug_" + it->ಮೊದಲ);
            results.push_back(ಫಲಿತಾಂಶ);
        }
    }

    ov::Parameter_ಸದಿಶ ಬಳಸಿದ_params;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & param : params) {
        ಆದರೆ (!param->output(0).get_ಗುರಿ_inputs().ಖಾಲಿಗಳು()) {
            ಬಳಸಿದ_params.push_back(param);
        }
    }
    // if (auto diff = params.size() - used_params.size()) {
    //     GGML_LOG_INFO("%zu parameters are not used in the model.", diff);
    // }
    resulting_model = std::make_shared<Model>(results, ಬಳಸಿದ_params);

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
    size_t ಸ್ಥಾನಾಂತರ = 0;
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & ಗಂಟು : resulting_model->get_ordered_ops()) {
        ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ cnst = ov::as_type_ptr<ov::op::v0::Constant>(ಗಂಟು); cnst && ov::shape_ಗಾತ್ರ(cnst->get_shape()) >= 16) {
            ಸ್ವಯಂಚಾಲಿತ & rt_info = cnst->get_rt_info();
            ಆದರೆ (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_ಗಾತ್ರ(), ಸ್ಥಾನಾಂತರ++, cnst->get_element_type());
            }
        }
    }
    ಹಿಂತಿರುಗಿ resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    ಸ್ವಯಂಚಾಲಿತ ggml_model_decoder = ov::as_type_ptr<ಒಳಹರಿವು_Model>(m_ಒಳಹರಿವು_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.ಗಣ_ಪ್ರತಿ_pass_validation(ನಿಜ);
        manager.ರಿಜಿಸ್ಟರ್_pass<ov::pass::MarkCompressedFloatConstants>();
        // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
        // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
        // so it survives ConstantFolding regardless of whether the target plugin's own
        // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
        manager.ರಿಜಿಸ್ಟರ್_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
        manager.ರಿಜಿಸ್ಟರ್_pass<pass::Fuse_ಗಮ್ಯ_Conv>();
        manager.ರಿಜಿಸ್ಟರ್_pass<pass::Fuse_Argsort_ತುದಿ_K>();

        // MOECompressed has no CPU plugin implementation, so enable it by default only on GPU.
        // GGML_OPENVINO_MOE_OP=0 keeps the unfused GatherMatmul path for fallback/debugging.
        ಆದರೆ (ggml_openvino_is_gpu() &&
            ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) != 0) {
            manager.ರಿಜಿಸ್ಟರ್_pass<pass::Fuse_Moe_ಮಾರ್ಗನಿರ್ದೇಶಕ>();
            manager.ರಿಜಿಸ್ಟರ್_pass<pass::FuseMoeCompressed>();
            // Same fusion for models whose gate and up projections share one fused expert
            // weight (gemma-4). It only matches that shape and only when the experts carry an
            // integer zero point, so it is a no-op on the separate-gate/up models above.
            manager.ರಿಜಿಸ್ಟರ್_pass<pass::Fuse_Moe_Compressed_Fused_Gate_ಮೇಲೆ>();
        }

        // Workaround for an OpenVINO GPU-plugin defect: an eltwise op whose operands differ in
        // rank is computed wrongly once the plugin fuses it as a post-op into `rms`. gemma-4
        // dense under stateful execution adds a rank-3 residual to the rank-4 norm output and
        // decodes garbage on GPU while CPU is correct. Equalising the ranks is a no-op for
        // NUMPY broadcasting and makes the fused path correct.
        // Remove once the plugin guards that fusion. Opt out with
        // GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN=1.
        ಆದರೆ (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
            manager.ರಿಜಿಸ್ಟರ್_pass<pass::Align_Eltwise_ಕಾರ್ಯಾಂಶ_Ranks>();
        }

        ಆದರೆ (ggml_model_decoder->is_stateful()) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ kv_param_res_ಹೆಸರುಗಳು = ggml_model_decoder->get_kv_param_res_ಹೆಸರುಗಳು();
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ kv_param_res_ಜೋಡಿಗಳು = get_kv_param_res_ಜೋಡಿಗಳು(model, kv_param_res_ಹೆಸರುಗಳು);
            manager.ರಿಜಿಸ್ಟರ್_pass<ov::pass::MakeStateful>(kv_param_res_ಜೋಡಿಗಳು);
            // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
            ಆದರೆ (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
                manager.ರಿಜಿಸ್ಟರ್_pass<pass::KVStateSeqAxis>();
            }
        }

        ಆದರೆ (ggml_model_decoder->is_static()) {
            manager.ರಿಜಿಸ್ಟರ್_pass<pass::SqueezeMatmul>();
        }
        manager.ಸತತ_passes(model);
        ಆದರೆ (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            ಗಾಗಿ (size_t i = 0; i < model->get_output_ಗಾತ್ರ(); i++) {
                ಸ್ವಯಂಚಾಲಿತ model_output_shape = model->output(i).get_partial_shape();
                ಆದರೆ (model_output_shape.rank().is_static() && model_output_shape.rank().get_ಉದ್ದಳತೆ() == 3) {
                    ppp.output(i).postprocess().custom([](ಸ್ಥಿರಾಂಕ ov::Output<ov::Node>& ಗಂಟು) {
                        ಸ್ವಯಂಚಾಲಿತ axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        ಹಿಂತಿರುಗಿ std::make_shared<ov::op::v0::Unsqueeze>(ಗಂಟು, axes);
                    });
                }
            }
            model = ppp.build();
        }
    }
    ಹಿಂತಿರುಗಿ model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
