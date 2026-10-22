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

ورتوں نطاق ov::op;

نطاق {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(ثابت std::string & name,
                                                        ثابت Model_داخلہ_Info & داخلہ_info) {
    خودکار param_گرہ = std::make_shared<ov::op::v0::Parameter>(داخلہ_info.type, داخلہ_info.shape);
    param_گرہ->سیٹ_friendly_name(name);
    param_گرہ->output(0).get_tensor().سیٹ_ناں({name});
    أرجع param_گرہ;
}

std::shared_ptr<ov::Node> create_extra_داخلہ(ثابت std::string & name, ثابت Model_Extra_داخلہ_Info & داخلہ_info) {
    إذا (داخلہ_info.is_parameter) {
        خودکار param_گرہ = std::make_shared<ov::op::v0::Parameter>(داخلہ_info.type, داخلہ_info.shape);
        param_گرہ->سیٹ_friendly_name(name);
        param_گرہ->output(0).get_tensor().سیٹ_ناں({name});
        أرجع param_گرہ;
    }

    خودکار constant = std::make_shared<ov::op::v0::Constant>(داخلہ_info.type, داخلہ_info.shape,
                                                          std::vector<int64_t>{داخلہ_info.value});
    constant->سیٹ_friendly_name(name);
    أرجع constant;
}

ov::pass::MakeStateful::Param_Res_جوڑے get_kv_param_res_جوڑے(
    ثابت std::shared_ptr<ov::Model> & model,
    ثابت std::map<std::string, std::string> & kv_param_res_ناں) {
    ov::pass::MakeStateful::Param_Res_جوڑے جوڑے;
    ثابت خودکار & params = model->get_parameters();
    ثابت خودکار & results = model->get_results();

    لكل (ثابت خودکار & param_res : kv_param_res_ناں) {
        ثابت خودکار & param_name = param_res.پہلا;
        ثابت خودکار & res_name = param_res.second;

        خودکار param_it = std::find_if(params.begin(), params.end(), [&](ثابت std::shared_ptr<v0::Parameter> & گرہ) {
            أرجع گرہ->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        خودکار res_it = std::find_if(results.begin(), results.end(), [&](ثابت std::shared_ptr<v0::نتیجہ_2> & گرہ) {
            أرجع گرہ->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::نتیجہ_2> res = *res_it;
        جوڑے.emplace_back(param, res);
    }
    أرجع جوڑے;
}

فراغ جوڑو_sliced_نقاب_stateful(TensorMap & tensor_map, منطقي imrope) {
    خودکار create_sliced_نقاب = [&](ثابت std::string & نقاب_name, ثابت std::string & sliced_name) {
        إذا ((tensor_map.find(نقاب_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end()) &&
            (tensor_map.find("inp_pos") != tensor_map.end())) {
            خودکار token_len_فی_seq = tensor_map.at("token_len_per_seq").get_گرہ_shared_ptr();
            خودکار نقاب = tensor_map.at(نقاب_name).get_گرہ_shared_ptr();
            std::shared_ptr<ov::Node> نقاب_sliced = نقاب;
            خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خودکار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خودکار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            خودکار قدم = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            خودکار axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            خودکار inp_pos = tensor_map.at("inp_pos").get_گرہ_shared_ptr();
            // IMROPE's fourth position plane is zero for text; use the token's first plane.
            ov::Output<ov::Node> آخری_اشاریہ = neg_one;
            إذا (imrope) {
                آخری_اشاریہ = std::make_shared<ov::op::v1::جوڑو>(token_len_فی_seq, neg_one);
            }
            خودکار آخری_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, آخری_اشاریہ, three);
            خودکار آخری_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                آخری_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), خطأ);
            خودکار آخری_inp_pos_cvt = std::make_shared<ov::op::v0::Convert>(آخری_inp_pos_1d, ov::element::i64);
            خودکار آخری_inp_pos_inc = std::make_shared<ov::op::v1::جوڑو>(آخری_inp_pos_cvt, one);

            نقاب_sliced = std::make_shared<ov::op::v8::ٹکڑا>(نقاب, zero, آخری_inp_pos_inc, قدم, axes);
            نقاب_sliced = std::make_shared<ov::op::v0::Convert>(نقاب_sliced, ov::element::f16);
            نقاب_sliced->سیٹ_friendly_name(sliced_name);

            tensor_map.insert({sliced_name, نقاب_sliced->output(0)});
        }
    };

    create_sliced_نقاب("self_kq_mask", "KQ_mask_sliced");
    create_sliced_نقاب("KQ_mask", "KQ_mask_sliced");
    create_sliced_نقاب("self_kq_mask_swa", "KQ_mask_swa_sliced");
}

// Rebuild the sliding-window mask from absolute positions.
// ggml caps self_kq_mask_swa at the size of its own SWA cache, but the stateful KV state is
// Concat-appended and grows without bound, so past that cap the two disagree on length and the
// mask add fails. A pure-Concat state is ordered by position, so positions can rebuild the mask.
// swa_window holds the real n_swa, read back from the ggml mask in ggml-decoder.cpp.
// No-op when the graph has no SWA mask, or when the window could not be read back.
فراغ جوڑو_position_نقاب_stateful_swa(TensorMap & tensor_map) {
    إذا (tensor_map.find("self_kq_mask_swa") == tensor_map.end() || tensor_map.find("inp_pos") == tensor_map.end() ||
        tensor_map.find("swa_window") == tensor_map.end()) {
        أرجع;
    }

    خودکار inp_pos = tensor_map.at("inp_pos").get_گرہ_shared_ptr();

    خودکار zero_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    خودکار one_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خودکار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    خودکار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    خودکار query_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::i64);
    خودکار query_pos_1d = std::make_shared<ov::op::v1::Reshape>(
        query_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), خطأ);

    خودکار آخری_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
    خودکار آخری_pos_1d = std::make_shared<ov::op::v1::Reshape>(آخری_pos, one_i64, خطأ);
    خودکار آخری_pos_cvt = std::make_shared<ov::op::v0::Convert>(آخری_pos_1d, ov::element::i64);
    خودکار کل_len = std::make_shared<ov::op::v1::جوڑو>(آخری_pos_cvt, one_i64);
    خودکار کل_len_scalar = std::make_shared<ov::op::v0::Squeeze>(کل_len);

    خودکار cached_pos = std::make_shared<ov::op::v4::Range>(
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}), کل_len_scalar,
        ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);

    خودکار query_خانہ = std::make_shared<ov::op::v1::Reshape>(
        query_pos_1d, ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1}), خطأ);
    خودکار cached_صف = std::make_shared<ov::op::v1::Reshape>(
        cached_pos, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, -1}), خطأ);
    خودکار diff = std::make_shared<ov::op::v1::گھٹاؤ>(query_خانہ, cached_صف);

    خودکار swa_کھڑکی = tensor_map.at("swa_window").get_گرہ_shared_ptr();
    خودکار کھڑکی = std::make_shared<ov::op::v0::Convert>(swa_کھڑکی, ov::element::i64);
    خودکار causal_ok = std::make_shared<ov::op::v1::GreaterEqual>(diff, zero_i64);
    خودکار کھڑکی_ok = std::make_shared<ov::op::v1::Less>(diff, کھڑکی);
    خودکار keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_ok, کھڑکی_ok);

    خودکار zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    خودکار neg_inf_f = ov::op::v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<عائم>::infinity()});
    std::shared_ptr<ov::Node> نقاب = std::make_shared<ov::op::v1::Select>(keep, zero_f, neg_inf_f);

    خودکار batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    نقاب = std::make_shared<ov::op::v0::Unsqueeze>(نقاب, batch_axis);
    نقاب = std::make_shared<ov::op::v0::Unsqueeze>(نقاب, batch_axis);
    نقاب = std::make_shared<ov::op::v0::Convert>(نقاب, ov::element::f16);
    نقاب->سیٹ_friendly_name("KQ_mask_swa_sliced");

    tensor_map["KQ_mask_swa_sliced"] = نقاب->output(0);
}

فراغ جوڑو_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    إذا (ggml_model_decoder.has_mixed_rope_params()) {
        أرجع;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    إذا (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == خالی_سوچک) {
        أرجع;
    }
    خودکار inp_pos = tensor_map.at("inp_pos").get_گرہ_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_بھار;
    إذا (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_بھار = tensor_map.at("rope_freqs.weight").get_گرہ_shared_ptr();
    }

    خودکار sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_بھار);
    خودکار sin_theta = sin_cos.پہلا;
    خودکار cos_theta = sin_cos.second;

    cos_theta.get_گرہ_shared_ptr()->سیٹ_friendly_name("rope_cos");
    sin_theta.get_گرہ_shared_ptr()->سیٹ_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
فراغ preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    إذا (ggml_model_decoder.is_stateful()) {
        جوڑو_sliced_نقاب_stateful(tensor_map, ggml_model_decoder.get_rope_params()[2] == GGML_ROPE_TYPE_IMROPE);
        جوڑو_position_نقاب_stateful_swa(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(ثابت frontend::داخلہ_Model::Ptr & داخلہ_model,
                                   ثابت std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   منطقي naive) :
    m_داخلہ_model(داخلہ_model),
    m_translator_map(translator_map),
    m_ov_model(خالی_سوچک),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    إذا (m_ov_model) {
        أرجع m_ov_model;
    }
    m_ov_model = translate_گراف(m_داخلہ_model);
    أرجع m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_گراف(ثابت frontend::داخلہ_Model::Ptr & داخلہ_model) {
    ov::Parameter_سمتیہ params;
    ov::نتیجہ_سمتیہ results;
    خودکار tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    ثابت خودکار & ggml_model = ov::as_type_ptr<داخلہ_Model>(داخلہ_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    لكل (ثابت خودکار & it : ggml_model_decoder->get_model_inputs()) {
        خودکار param_گرہ = create_parameter(it.پہلا, it.second);
        params.push_back(param_گرہ);
        (*tensor_map)[it.پہلا] = param_گرہ;
    }

    لكل (ثابت خودکار & it : ggml_model_decoder->get_model_extra_inputs()) {
        خودکار داخلہ_گرہ = create_extra_داخلہ(it.پہلا, it.second);
        إذا (it.second.is_parameter) {
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(داخلہ_گرہ));
        }
        (*tensor_map)[it.پہلا] = داخلہ_گرہ;
    }

    لكل (ثابت خودکار & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.پہلا] = it.second;
    }

    خودکار translate_گرہ = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح گرہ_idx) {
        خودکار operation_type = decoder->get_op_type(گرہ_idx);
        إذا (operation_type == "GGML_OP_NONE") {
            أرجع ov::Output_سمتیہ{};
        }

        ثابت خودکار & گرہ_output_ناں = decoder->get_output_ناں(گرہ_idx);
        إذا (operation_type == "GGML_OP_VIEW" && decoder->get_op_case(گرہ_idx) == 2 && گرہ_output_ناں.size() == 1) {
            خودکار سدھا_output = tensor_map->find(گرہ_output_ناں[0]);
            إذا (سدھا_output != tensor_map->end()) {
                // GDN publishes its native attention/state outputs under the two GGML VIEW names.
                // Keep those mappings instead of rebuilding slices of a packed temporary.
                أرجع ov::Output_سمتیہ{سدھا_output->second};
            }
        }

        خودکار it = m_translator_map.find(operation_type);
        FRONT_سرا_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        گرہ_Context گرہ_context(decoder, tensor_map, گرہ_idx, هذا);
        ov::Output_سمتیہ converted_outputs = it->second(گرہ_context);

        FRONT_سرا_OP_CONVERSION_CHECK(گرہ_output_ناں.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      گرہ_output_ناں.size(), " and ", converted_outputs.size(), " respectively.");

        لكل (size_t i = 0; i < گرہ_output_ناں.size(); ++i) {
            خودکار output_name = گرہ_output_ناں[i];
            إذا (i < converted_outputs.size() && converted_outputs[i].get_گرہ_shared_ptr() != خالی_سوچک) {
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
    خودکار refresh_inplace_aliases = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح inplace_گرہ_idx,
                                       ثابت std::string & view_src_name) {
        لكل (صحيح گرہ_idx = 0; گرہ_idx < inplace_گرہ_idx; گرہ_idx++) {
            إذا (decoder->is_view_like_alias_دا(گرہ_idx, view_src_name)) {
                translate_گرہ(decoder, گرہ_idx);
            }
        }
    };

    خودکار گرہ_visitor = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح گرہ_idx) {
        خودکار converted_outputs = translate_گرہ(decoder, گرہ_idx);
        إذا (converted_outputs.خالی_تھاواں()) {
            أرجع;
        }
        ثابت خودکار inplace_src = decoder->get_inplace_op_src(گرہ_idx);
        إذا (inplace_src.خالی_تھاواں()) {
            أرجع;
        }
        إذا (converted_outputs[0].get_گرہ_shared_ptr() != خالی_سوچک) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, گرہ_idx, inplace_src);
    };

    إذا (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->ویکھو_subgraph(گرہ_visitor);

    لكل (ثابت خودکار & name : ggml_model_decoder->get_model_output_ناں()) {
        FRONT_سرا_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        خودکار نتیجہ = std::make_shared<v0::نتیجہ_2>(tensor_map->at(name));
        نتیجہ->سیٹ_friendly_name(name);
        results.push_back(نتیجہ);
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
    إذا (ثابت حرف * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        std::stringstream ss(debug_nodes);
        std::string name;
        طالما (std::getline(ss, name, ',')) {
            خودکار it = tensor_map->find(name);
            إذا (it == tensor_map->end()) {
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](ثابت خودکار & اندراج) {
                    أرجع اندراج.پہلا.compare(0, name.size(), name) == 0 && اندراج.پہلا.size() > name.size() &&
                           اندراج.پہلا[name.size()] == '#';
                });
            }
            إذا (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                استمر;
            }
            خودکار نتیجہ = std::make_shared<v0::نتیجہ_2>(it->second);
            نتیجہ->سیٹ_friendly_name("__debug_" + it->پہلا);
            results.push_back(نتیجہ);
        }
    }

    ov::Parameter_سمتیہ ورتیا_params;
    لكل (ثابت خودکار & param : params) {
        إذا (!param->output(0).get_نشانہ_inputs().خالی_تھاواں()) {
            ورتیا_params.push_back(param);
        }
    }
    // if (auto diff = params.size() - used_params.size()) {
    //     GGML_LOG_INFO("%zu parameters are not used in the model.", diff);
    // }
    resulting_model = std::make_shared<Model>(results, ورتیا_params);

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
    size_t وتھ = 0;
    لكل (خودکار & گرہ : resulting_model->get_ordered_ops()) {
        إذا (خودکار cnst = ov::as_type_ptr<ov::op::v0::Constant>(گرہ); cnst && ov::shape_ناپ(cnst->get_shape()) >= 16) {
            خودکار & rt_info = cnst->get_rt_info();
            إذا (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_ناپ(), وتھ++, cnst->get_element_type());
            }
        }
    }
    أرجع resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    خودکار ggml_model_decoder = ov::as_type_ptr<داخلہ_Model>(m_داخلہ_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.سیٹ_فی_pass_validation(سچ);
        manager.رجسٹر_pass<ov::pass::MarkCompressedFloatConstants>();
        // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
        // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
        // so it survives ConstantFolding regardless of whether the target plugin's own
        // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
        manager.رجسٹر_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
        manager.رجسٹر_pass<pass::Fuse_منزل_Conv>();
        manager.رجسٹر_pass<pass::Fuse_Argsort_چوٹی_K>();

        // MOECompressed has no CPU plugin implementation, so enable it by default only on GPU.
        // GGML_OPENVINO_MOE_OP=0 keeps the unfused GatherMatmul path for fallback/debugging.
        إذا (ggml_openvino_is_gpu() &&
            ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) != 0) {
            manager.رجسٹر_pass<pass::Fuse_Moe_راؤٹر>();
            manager.رجسٹر_pass<pass::FuseMoeCompressed>();
            // Same fusion for models whose gate and up projections share one fused expert
            // weight (gemma-4). It only matches that shape and only when the experts carry an
            // integer zero point, so it is a no-op on the separate-gate/up models above.
            manager.رجسٹر_pass<pass::Fuse_Moe_Compressed_Fused_Gate_اتے>();
        }

        // Workaround for an OpenVINO GPU-plugin defect: an eltwise op whose operands differ in
        // rank is computed wrongly once the plugin fuses it as a post-op into `rms`. gemma-4
        // dense under stateful execution adds a rank-3 residual to the rank-4 norm output and
        // decodes garbage on GPU while CPU is correct. Equalising the ranks is a no-op for
        // NUMPY broadcasting and makes the fused path correct.
        // Remove once the plugin guards that fusion. Opt out with
        // GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN=1.
        إذا (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
            manager.رجسٹر_pass<pass::Align_Eltwise_عملیہ_Ranks>();
        }

        إذا (ggml_model_decoder->is_stateful()) {
            ثابت خودکار kv_param_res_ناں = ggml_model_decoder->get_kv_param_res_ناں();
            ثابت خودکار kv_param_res_جوڑے = get_kv_param_res_جوڑے(model, kv_param_res_ناں);
            manager.رجسٹر_pass<ov::pass::MakeStateful>(kv_param_res_جوڑے);
            // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
            إذا (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
                manager.رجسٹر_pass<pass::KVStateSeqAxis>();
            }
        }

        إذا (ggml_model_decoder->is_static()) {
            manager.رجسٹر_pass<pass::SqueezeMatmul>();
        }
        manager.لگاتار_passes(model);
        إذا (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            لكل (size_t i = 0; i < model->get_output_ناپ(); i++) {
                خودکار model_output_shape = model->output(i).get_partial_shape();
                إذا (model_output_shape.rank().is_static() && model_output_shape.rank().get_لمائی() == 3) {
                    ppp.output(i).postprocess().custom([](ثابت ov::Output<ov::Node>& گرہ) {
                        خودکار axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        أرجع std::make_shared<ov::op::v0::Unsqueeze>(گرہ, axes);
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
