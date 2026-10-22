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

استفاده نطاق ov::op;

نطاق {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(ثابت std::string & name,
                                                        ثابت Model_درون_داد_Info & درون_داد_info) {
    خودکار param_گره = std::make_shared<ov::op::v0::Parameter>(درون_داد_info.type, درون_داد_info.shape);
    param_گره->مجموعه_friendly_name(name);
    param_گره->output(0).get_tensor().مجموعه_نامها({name});
    أرجع param_گره;
}

std::shared_ptr<ov::Node> create_extra_درون_داد(ثابت std::string & name, ثابت Model_Extra_درون_داد_Info & درون_داد_info) {
    إذا (درون_داد_info.is_parameter) {
        خودکار param_گره = std::make_shared<ov::op::v0::Parameter>(درون_داد_info.type, درون_داد_info.shape);
        param_گره->مجموعه_friendly_name(name);
        param_گره->output(0).get_tensor().مجموعه_نامها({name});
        أرجع param_گره;
    }

    خودکار constant = std::make_shared<ov::op::v0::Constant>(درون_داد_info.type, درون_داد_info.shape,
                                                          std::vector<int64_t>{درون_داد_info.value});
    constant->مجموعه_friendly_name(name);
    أرجع constant;
}

ov::pass::MakeStateful::Param_Res_جفتها get_kv_param_res_جفتها(
    ثابت std::shared_ptr<ov::Model> & model,
    ثابت std::map<std::string, std::string> & kv_param_res_نامها) {
    ov::pass::MakeStateful::Param_Res_جفتها جفتها;
    ثابت خودکار & params = model->get_parameters();
    ثابت خودکار & results = model->get_results();

    لكل (ثابت خودکار & param_res : kv_param_res_نامها) {
        ثابت خودکار & param_name = param_res.اولی;
        ثابت خودکار & res_name = param_res.second;

        خودکار param_it = std::find_if(params.begin(), params.end(), [&](ثابت std::shared_ptr<v0::Parameter> & گره) {
            أرجع گره->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        خودکار res_it = std::find_if(results.begin(), results.end(), [&](ثابت std::shared_ptr<v0::نتیجه_2> & گره) {
            أرجع گره->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::نتیجه_2> res = *res_it;
        جفتها.emplace_back(param, res);
    }
    أرجع جفتها;
}

فراغ جمع_کن_sliced_ماسک_stateful(TensorMap & tensor_map, منطقي imrope) {
    خودکار create_sliced_ماسک = [&](ثابت std::string & ماسک_name, ثابت std::string & sliced_name) {
        إذا ((tensor_map.find(ماسک_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end()) &&
            (tensor_map.find("inp_pos") != tensor_map.end())) {
            خودکار token_len_در_هر_seq = tensor_map.at("token_len_per_seq").get_گره_shared_ptr();
            خودکار ماسک = tensor_map.at(ماسک_name).get_گره_shared_ptr();
            std::shared_ptr<ov::Node> ماسک_sliced = ماسک;
            خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خودکار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خودکار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            خودکار گام = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            خودکار axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            خودکار inp_pos = tensor_map.at("inp_pos").get_گره_shared_ptr();
            // IMROPE's fourth position plane is zero for text; use the token's first plane.
            ov::Output<ov::Node> آخرین_شاخص = neg_one;
            إذا (imrope) {
                آخرین_شاخص = std::make_shared<ov::op::v1::جمع_کن>(token_len_در_هر_seq, neg_one);
            }
            خودکار آخرین_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, آخرین_شاخص, three);
            خودکار آخرین_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                آخرین_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), خطأ);
            خودکار آخرین_inp_pos_cvt = std::make_shared<ov::op::v0::Convert>(آخرین_inp_pos_1d, ov::element::i64);
            خودکار آخرین_inp_pos_inc = std::make_shared<ov::op::v1::جمع_کن>(آخرین_inp_pos_cvt, one);

            ماسک_sliced = std::make_shared<ov::op::v8::برش>(ماسک, zero, آخرین_inp_pos_inc, گام, axes);
            ماسک_sliced = std::make_shared<ov::op::v0::Convert>(ماسک_sliced, ov::element::f16);
            ماسک_sliced->مجموعه_friendly_name(sliced_name);

            tensor_map.insert({sliced_name, ماسک_sliced->output(0)});
        }
    };

    create_sliced_ماسک("self_kq_mask", "KQ_mask_sliced");
    create_sliced_ماسک("KQ_mask", "KQ_mask_sliced");
    create_sliced_ماسک("self_kq_mask_swa", "KQ_mask_swa_sliced");
}

// Rebuild the sliding-window mask from absolute positions.
// ggml caps self_kq_mask_swa at the size of its own SWA cache, but the stateful KV state is
// Concat-appended and grows without bound, so past that cap the two disagree on length and the
// mask add fails. A pure-Concat state is ordered by position, so positions can rebuild the mask.
// swa_window holds the real n_swa, read back from the ggml mask in ggml-decoder.cpp.
// No-op when the graph has no SWA mask, or when the window could not be read back.
فراغ جمع_کن_position_ماسک_stateful_swa(TensorMap & tensor_map) {
    إذا (tensor_map.find("self_kq_mask_swa") == tensor_map.end() || tensor_map.find("inp_pos") == tensor_map.end() ||
        tensor_map.find("swa_window") == tensor_map.end()) {
        أرجع;
    }

    خودکار inp_pos = tensor_map.at("inp_pos").get_گره_shared_ptr();

    خودکار zero_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    خودکار one_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خودکار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    خودکار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    خودکار query_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::i64);
    خودکار query_pos_1d = std::make_shared<ov::op::v1::Reshape>(
        query_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), خطأ);

    خودکار آخرین_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
    خودکار آخرین_pos_1d = std::make_shared<ov::op::v1::Reshape>(آخرین_pos, one_i64, خطأ);
    خودکار آخرین_pos_cvt = std::make_shared<ov::op::v0::Convert>(آخرین_pos_1d, ov::element::i64);
    خودکار مجموع_len = std::make_shared<ov::op::v1::جمع_کن>(آخرین_pos_cvt, one_i64);
    خودکار مجموع_len_scalar = std::make_shared<ov::op::v0::Squeeze>(مجموع_len);

    خودکار cached_pos = std::make_shared<ov::op::v4::Range>(
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}), مجموع_len_scalar,
        ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);

    خودکار query_ستون = std::make_shared<ov::op::v1::Reshape>(
        query_pos_1d, ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1}), خطأ);
    خودکار cached_ردیف = std::make_shared<ov::op::v1::Reshape>(
        cached_pos, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, -1}), خطأ);
    خودکار diff = std::make_shared<ov::op::v1::تفریق>(query_ستون, cached_ردیف);

    خودکار swa_پنجره = tensor_map.at("swa_window").get_گره_shared_ptr();
    خودکار پنجره = std::make_shared<ov::op::v0::Convert>(swa_پنجره, ov::element::i64);
    خودکار causal_ok = std::make_shared<ov::op::v1::GreaterEqual>(diff, zero_i64);
    خودکار پنجره_ok = std::make_shared<ov::op::v1::Less>(diff, پنجره);
    خودکار keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_ok, پنجره_ok);

    خودکار zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    خودکار neg_inf_f = ov::op::v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<عائم>::infinity()});
    std::shared_ptr<ov::Node> ماسک = std::make_shared<ov::op::v1::Select>(keep, zero_f, neg_inf_f);

    خودکار batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    ماسک = std::make_shared<ov::op::v0::Unsqueeze>(ماسک, batch_axis);
    ماسک = std::make_shared<ov::op::v0::Unsqueeze>(ماسک, batch_axis);
    ماسک = std::make_shared<ov::op::v0::Convert>(ماسک, ov::element::f16);
    ماسک->مجموعه_friendly_name("KQ_mask_swa_sliced");

    tensor_map["KQ_mask_swa_sliced"] = ماسک->output(0);
}

فراغ جمع_کن_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    إذا (ggml_model_decoder.has_mixed_rope_params()) {
        أرجع;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    إذا (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == اشاره_تهی) {
        أرجع;
    }
    خودکار inp_pos = tensor_map.at("inp_pos").get_گره_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_وزن;
    إذا (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_وزن = tensor_map.at("rope_freqs.weight").get_گره_shared_ptr();
    }

    خودکار sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_وزن);
    خودکار sin_theta = sin_cos.اولی;
    خودکار cos_theta = sin_cos.second;

    cos_theta.get_گره_shared_ptr()->مجموعه_friendly_name("rope_cos");
    sin_theta.get_گره_shared_ptr()->مجموعه_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
فراغ preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    إذا (ggml_model_decoder.is_stateful()) {
        جمع_کن_sliced_ماسک_stateful(tensor_map, ggml_model_decoder.get_rope_params()[2] == GGML_ROPE_TYPE_IMROPE);
        جمع_کن_position_ماسک_stateful_swa(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(ثابت frontend::درون_داد_Model::Ptr & درون_داد_model,
                                   ثابت std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   منطقي naive) :
    m_درون_داد_model(درون_داد_model),
    m_translator_map(translator_map),
    m_ov_model(اشاره_تهی),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    إذا (m_ov_model) {
        أرجع m_ov_model;
    }
    m_ov_model = translate_گراف(m_درون_داد_model);
    أرجع m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_گراف(ثابت frontend::درون_داد_Model::Ptr & درون_داد_model) {
    ov::Parameter_وکتور params;
    ov::نتیجه_وکتور results;
    خودکار tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    ثابت خودکار & ggml_model = ov::as_type_ptr<درون_داد_Model>(درون_داد_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    لكل (ثابت خودکار & it : ggml_model_decoder->get_model_inputs()) {
        خودکار param_گره = create_parameter(it.اولی, it.second);
        params.push_back(param_گره);
        (*tensor_map)[it.اولی] = param_گره;
    }

    لكل (ثابت خودکار & it : ggml_model_decoder->get_model_extra_inputs()) {
        خودکار درون_داد_گره = create_extra_درون_داد(it.اولی, it.second);
        إذا (it.second.is_parameter) {
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(درون_داد_گره));
        }
        (*tensor_map)[it.اولی] = درون_داد_گره;
    }

    لكل (ثابت خودکار & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.اولی] = it.second;
    }

    خودکار translate_گره = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح گره_idx) {
        خودکار operation_type = decoder->get_op_type(گره_idx);
        إذا (operation_type == "GGML_OP_NONE") {
            أرجع ov::Output_وکتور{};
        }

        ثابت خودکار & گره_output_نامها = decoder->get_output_نامها(گره_idx);
        إذا (operation_type == "GGML_OP_VIEW" && decoder->get_op_case(گره_idx) == 2 && گره_output_نامها.size() == 1) {
            خودکار مستقیم_output = tensor_map->find(گره_output_نامها[0]);
            إذا (مستقیم_output != tensor_map->end()) {
                // GDN publishes its native attention/state outputs under the two GGML VIEW names.
                // Keep those mappings instead of rebuilding slices of a packed temporary.
                أرجع ov::Output_وکتور{مستقیم_output->second};
            }
        }

        خودکار it = m_translator_map.find(operation_type);
        FRONT_انتها_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        گره_Context گره_context(decoder, tensor_map, گره_idx, هذا);
        ov::Output_وکتور converted_outputs = it->second(گره_context);

        FRONT_انتها_OP_CONVERSION_CHECK(گره_output_نامها.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      گره_output_نامها.size(), " and ", converted_outputs.size(), " respectively.");

        لكل (size_t i = 0; i < گره_output_نامها.size(); ++i) {
            خودکار output_name = گره_output_نامها[i];
            إذا (i < converted_outputs.size() && converted_outputs[i].get_گره_shared_ptr() != اشاره_تهی) {
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
    خودکار refresh_inplace_aliases = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح inplace_گره_idx,
                                       ثابت std::string & view_src_name) {
        لكل (صحيح گره_idx = 0; گره_idx < inplace_گره_idx; گره_idx++) {
            إذا (decoder->is_view_like_alias_از(گره_idx, view_src_name)) {
                translate_گره(decoder, گره_idx);
            }
        }
    };

    خودکار گره_visitor = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح گره_idx) {
        خودکار converted_outputs = translate_گره(decoder, گره_idx);
        إذا (converted_outputs.خالی()) {
            أرجع;
        }
        ثابت خودکار inplace_src = decoder->get_inplace_op_src(گره_idx);
        إذا (inplace_src.خالی()) {
            أرجع;
        }
        إذا (converted_outputs[0].get_گره_shared_ptr() != اشاره_تهی) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, گره_idx, inplace_src);
    };

    إذا (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->بازدید_subgraph(گره_visitor);

    لكل (ثابت خودکار & name : ggml_model_decoder->get_model_output_نامها()) {
        FRONT_انتها_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        خودکار نتیجه = std::make_shared<v0::نتیجه_2>(tensor_map->at(name));
        نتیجه->مجموعه_friendly_name(name);
        results.push_back(نتیجه);
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
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](ثابت خودکار & مدخل) {
                    أرجع مدخل.اولی.compare(0, name.size(), name) == 0 && مدخل.اولی.size() > name.size() &&
                           مدخل.اولی[name.size()] == '#';
                });
            }
            إذا (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                استمر;
            }
            خودکار نتیجه = std::make_shared<v0::نتیجه_2>(it->second);
            نتیجه->مجموعه_friendly_name("__debug_" + it->اولی);
            results.push_back(نتیجه);
        }
    }

    ov::Parameter_وکتور استفاده_شده_params;
    لكل (ثابت خودکار & param : params) {
        إذا (!param->output(0).get_هدف_inputs().خالی()) {
            استفاده_شده_params.push_back(param);
        }
    }
    // if (auto diff = params.size() - used_params.size()) {
    //     GGML_LOG_INFO("%zu parameters are not used in the model.", diff);
    // }
    resulting_model = std::make_shared<Model>(results, استفاده_شده_params);

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
    size_t جابه_جایی = 0;
    لكل (خودکار & گره : resulting_model->get_ordered_ops()) {
        إذا (خودکار cnst = ov::as_type_ptr<ov::op::v0::Constant>(گره); cnst && ov::shape_اندازه(cnst->get_shape()) >= 16) {
            خودکار & rt_info = cnst->get_rt_info();
            إذا (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_اندازه(), جابه_جایی++, cnst->get_element_type());
            }
        }
    }
    أرجع resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    خودکار ggml_model_decoder = ov::as_type_ptr<درون_داد_Model>(m_درون_داد_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.مجموعه_در_هر_pass_validation(درست);
        manager.ثبات_pass<ov::pass::MarkCompressedFloatConstants>();
        // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
        // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
        // so it survives ConstantFolding regardless of whether the target plugin's own
        // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
        manager.ثبات_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
        manager.ثبات_pass<pass::Fuse_مقصد_Conv>();
        manager.ثبات_pass<pass::Fuse_Argsort_بالاترین_K>();

        // MOECompressed has no CPU plugin implementation, so enable it by default only on GPU.
        // GGML_OPENVINO_MOE_OP=0 keeps the unfused GatherMatmul path for fallback/debugging.
        إذا (ggml_openvino_is_gpu() &&
            ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) != 0) {
            manager.ثبات_pass<pass::Fuse_Moe_مسیریاب>();
            manager.ثبات_pass<pass::FuseMoeCompressed>();
            // Same fusion for models whose gate and up projections share one fused expert
            // weight (gemma-4). It only matches that shape and only when the experts carry an
            // integer zero point, so it is a no-op on the separate-gate/up models above.
            manager.ثبات_pass<pass::Fuse_Moe_Compressed_Fused_Gate_به_بالا>();
        }

        // Workaround for an OpenVINO GPU-plugin defect: an eltwise op whose operands differ in
        // rank is computed wrongly once the plugin fuses it as a post-op into `rms`. gemma-4
        // dense under stateful execution adds a rank-3 residual to the rank-4 norm output and
        // decodes garbage on GPU while CPU is correct. Equalising the ranks is a no-op for
        // NUMPY broadcasting and makes the fused path correct.
        // Remove once the plugin guards that fusion. Opt out with
        // GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN=1.
        إذا (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
            manager.ثبات_pass<pass::Align_Eltwise_عملوند_Ranks>();
        }

        إذا (ggml_model_decoder->is_stateful()) {
            ثابت خودکار kv_param_res_نامها = ggml_model_decoder->get_kv_param_res_نامها();
            ثابت خودکار kv_param_res_جفتها = get_kv_param_res_جفتها(model, kv_param_res_نامها);
            manager.ثبات_pass<ov::pass::MakeStateful>(kv_param_res_جفتها);
            // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
            إذا (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
                manager.ثبات_pass<pass::KVStateSeqAxis>();
            }
        }

        إذا (ggml_model_decoder->is_static()) {
            manager.ثبات_pass<pass::SqueezeMatmul>();
        }
        manager.پیاپی_passes(model);
        إذا (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            لكل (size_t i = 0; i < model->get_output_اندازه(); i++) {
                خودکار model_output_shape = model->output(i).get_partial_shape();
                إذا (model_output_shape.rank().is_static() && model_output_shape.rank().get_درازی() == 3) {
                    ppp.output(i).postprocess().custom([](ثابت ov::Output<ov::Node>& گره) {
                        خودکار axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        أرجع std::make_shared<ov::op::v0::Unsqueeze>(گره, axes);
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
