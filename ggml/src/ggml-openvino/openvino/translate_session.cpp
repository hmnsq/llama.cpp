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

استعمال نطاق ov::op;

نطاق {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(ثابت std::string & name,
                                                        ثابت Model_داخلا_Info & داخلا_info) {
    خودڪار param_ڳنڍ = std::make_shared<ov::op::v0::Parameter>(داخلا_info.type, داخلا_info.shape);
    param_ڳنڍ->سيٽ_friendly_name(name);
    param_ڳنڍ->output(0).get_tensor().سيٽ_نالا({name});
    أرجع param_ڳنڍ;
}

std::shared_ptr<ov::Node> create_extra_داخلا(ثابت std::string & name, ثابت Model_Extra_داخلا_Info & داخلا_info) {
    إذا (داخلا_info.is_parameter) {
        خودڪار param_ڳنڍ = std::make_shared<ov::op::v0::Parameter>(داخلا_info.type, داخلا_info.shape);
        param_ڳنڍ->سيٽ_friendly_name(name);
        param_ڳنڍ->output(0).get_tensor().سيٽ_نالا({name});
        أرجع param_ڳنڍ;
    }

    خودڪار constant = std::make_shared<ov::op::v0::Constant>(داخلا_info.type, داخلا_info.shape,
                                                          std::vector<int64_t>{داخلا_info.value});
    constant->سيٽ_friendly_name(name);
    أرجع constant;
}

ov::pass::MakeStateful::Param_Res_جوڙا get_kv_param_res_جوڙا(
    ثابت std::shared_ptr<ov::Model> & model,
    ثابت std::map<std::string, std::string> & kv_param_res_نالا) {
    ov::pass::MakeStateful::Param_Res_جوڙا جوڙا;
    ثابت خودڪار & params = model->get_parameters();
    ثابت خودڪار & results = model->get_results();

    لكل (ثابت خودڪار & param_res : kv_param_res_نالا) {
        ثابت خودڪار & param_name = param_res.پهريون;
        ثابت خودڪار & res_name = param_res.second;

        خودڪار param_it = std::find_if(params.begin(), params.end(), [&](ثابت std::shared_ptr<v0::Parameter> & ڳنڍ) {
            أرجع ڳنڍ->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        خودڪار res_it = std::find_if(results.begin(), results.end(), [&](ثابت std::shared_ptr<v0::نتيجو_2> & ڳنڍ) {
            أرجع ڳنڍ->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::نتيجو_2> res = *res_it;
        جوڙا.emplace_back(param, res);
    }
    أرجع جوڙا;
}

فراغ جوڙيو_sliced_نقاب_stateful(TensorMap & tensor_map, منطقي imrope) {
    خودڪار create_sliced_نقاب = [&](ثابت std::string & نقاب_name, ثابت std::string & sliced_name) {
        إذا ((tensor_map.find(نقاب_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end()) &&
            (tensor_map.find("inp_pos") != tensor_map.end())) {
            خودڪار token_len_في_seq = tensor_map.at("token_len_per_seq").get_ڳنڍ_shared_ptr();
            خودڪار نقاب = tensor_map.at(نقاب_name).get_ڳنڍ_shared_ptr();
            std::shared_ptr<ov::Node> نقاب_sliced = نقاب;
            خودڪار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            خودڪار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            خودڪار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خودڪار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            خودڪار قدم = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            خودڪار axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            خودڪار inp_pos = tensor_map.at("inp_pos").get_ڳنڍ_shared_ptr();
            // IMROPE's fourth position plane is zero for text; use the token's first plane.
            ov::Output<ov::Node> آخري_اشاريو = neg_one;
            إذا (imrope) {
                آخري_اشاريو = std::make_shared<ov::op::v1::جوڙيو>(token_len_في_seq, neg_one);
            }
            خودڪار آخري_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, آخري_اشاريو, three);
            خودڪار آخري_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                آخري_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), خطأ);
            خودڪار آخري_inp_pos_cvt = std::make_shared<ov::op::v0::Convert>(آخري_inp_pos_1d, ov::element::i64);
            خودڪار آخري_inp_pos_inc = std::make_shared<ov::op::v1::جوڙيو>(آخري_inp_pos_cvt, one);

            نقاب_sliced = std::make_shared<ov::op::v8::ٽڪرو>(نقاب, zero, آخري_inp_pos_inc, قدم, axes);
            نقاب_sliced = std::make_shared<ov::op::v0::Convert>(نقاب_sliced, ov::element::f16);
            نقاب_sliced->سيٽ_friendly_name(sliced_name);

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
فراغ جوڙيو_position_نقاب_stateful_swa(TensorMap & tensor_map) {
    إذا (tensor_map.find("self_kq_mask_swa") == tensor_map.end() || tensor_map.find("inp_pos") == tensor_map.end() ||
        tensor_map.find("swa_window") == tensor_map.end()) {
        أرجع;
    }

    خودڪار inp_pos = tensor_map.at("inp_pos").get_ڳنڍ_shared_ptr();

    خودڪار zero_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    خودڪار one_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خودڪار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    خودڪار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    خودڪار query_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::i64);
    خودڪار query_pos_1d = std::make_shared<ov::op::v1::Reshape>(
        query_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), خطأ);

    خودڪار آخري_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
    خودڪار آخري_pos_1d = std::make_shared<ov::op::v1::Reshape>(آخري_pos, one_i64, خطأ);
    خودڪار آخري_pos_cvt = std::make_shared<ov::op::v0::Convert>(آخري_pos_1d, ov::element::i64);
    خودڪار ڪل_len = std::make_shared<ov::op::v1::جوڙيو>(آخري_pos_cvt, one_i64);
    خودڪار ڪل_len_scalar = std::make_shared<ov::op::v0::Squeeze>(ڪل_len);

    خودڪار cached_pos = std::make_shared<ov::op::v4::Range>(
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}), ڪل_len_scalar,
        ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);

    خودڪار query_کانو = std::make_shared<ov::op::v1::Reshape>(
        query_pos_1d, ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1}), خطأ);
    خودڪار cached_صف = std::make_shared<ov::op::v1::Reshape>(
        cached_pos, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, -1}), خطأ);
    خودڪار diff = std::make_shared<ov::op::v1::ڪٽيو>(query_کانو, cached_صف);

    خودڪار swa_دري = tensor_map.at("swa_window").get_ڳنڍ_shared_ptr();
    خودڪار دري = std::make_shared<ov::op::v0::Convert>(swa_دري, ov::element::i64);
    خودڪار causal_ok = std::make_shared<ov::op::v1::GreaterEqual>(diff, zero_i64);
    خودڪار دري_ok = std::make_shared<ov::op::v1::Less>(diff, دري);
    خودڪار keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_ok, دري_ok);

    خودڪار zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    خودڪار neg_inf_f = ov::op::v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<عائم>::infinity()});
    std::shared_ptr<ov::Node> نقاب = std::make_shared<ov::op::v1::Select>(keep, zero_f, neg_inf_f);

    خودڪار batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    نقاب = std::make_shared<ov::op::v0::Unsqueeze>(نقاب, batch_axis);
    نقاب = std::make_shared<ov::op::v0::Unsqueeze>(نقاب, batch_axis);
    نقاب = std::make_shared<ov::op::v0::Convert>(نقاب, ov::element::f16);
    نقاب->سيٽ_friendly_name("KQ_mask_swa_sliced");

    tensor_map["KQ_mask_swa_sliced"] = نقاب->output(0);
}

فراغ جوڙيو_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    إذا (ggml_model_decoder.has_mixed_rope_params()) {
        أرجع;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    إذا (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == خالي_اشارو) {
        أرجع;
    }
    خودڪار inp_pos = tensor_map.at("inp_pos").get_ڳنڍ_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_وزن;
    إذا (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_وزن = tensor_map.at("rope_freqs.weight").get_ڳنڍ_shared_ptr();
    }

    خودڪار sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_وزن);
    خودڪار sin_theta = sin_cos.پهريون;
    خودڪار cos_theta = sin_cos.second;

    cos_theta.get_ڳنڍ_shared_ptr()->سيٽ_friendly_name("rope_cos");
    sin_theta.get_ڳنڍ_shared_ptr()->سيٽ_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
فراغ preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    إذا (ggml_model_decoder.is_stateful()) {
        جوڙيو_sliced_نقاب_stateful(tensor_map, ggml_model_decoder.get_rope_params()[2] == GGML_ROPE_TYPE_IMROPE);
        جوڙيو_position_نقاب_stateful_swa(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(ثابت frontend::داخلا_Model::Ptr & داخلا_model,
                                   ثابت std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   منطقي naive) :
    m_داخلا_model(داخلا_model),
    m_translator_map(translator_map),
    m_ov_model(خالي_اشارو),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    إذا (m_ov_model) {
        أرجع m_ov_model;
    }
    m_ov_model = translate_گراف(m_داخلا_model);
    أرجع m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_گراف(ثابت frontend::داخلا_Model::Ptr & داخلا_model) {
    ov::Parameter_ويڪٽر params;
    ov::نتيجو_ويڪٽر results;
    خودڪار tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    ثابت خودڪار & ggml_model = ov::as_type_ptr<داخلا_Model>(داخلا_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    لكل (ثابت خودڪار & it : ggml_model_decoder->get_model_inputs()) {
        خودڪار param_ڳنڍ = create_parameter(it.پهريون, it.second);
        params.push_back(param_ڳنڍ);
        (*tensor_map)[it.پهريون] = param_ڳنڍ;
    }

    لكل (ثابت خودڪار & it : ggml_model_decoder->get_model_extra_inputs()) {
        خودڪار داخلا_ڳنڍ = create_extra_داخلا(it.پهريون, it.second);
        إذا (it.second.is_parameter) {
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(داخلا_ڳنڍ));
        }
        (*tensor_map)[it.پهريون] = داخلا_ڳنڍ;
    }

    لكل (ثابت خودڪار & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.پهريون] = it.second;
    }

    خودڪار translate_ڳنڍ = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح ڳنڍ_idx) {
        خودڪار operation_type = decoder->get_op_type(ڳنڍ_idx);
        إذا (operation_type == "GGML_OP_NONE") {
            أرجع ov::Output_ويڪٽر{};
        }

        ثابت خودڪار & ڳنڍ_output_نالا = decoder->get_output_نالا(ڳنڍ_idx);
        إذا (operation_type == "GGML_OP_VIEW" && decoder->get_op_case(ڳنڍ_idx) == 2 && ڳنڍ_output_نالا.size() == 1) {
            خودڪار سڌو_output = tensor_map->find(ڳنڍ_output_نالا[0]);
            إذا (سڌو_output != tensor_map->end()) {
                // GDN publishes its native attention/state outputs under the two GGML VIEW names.
                // Keep those mappings instead of rebuilding slices of a packed temporary.
                أرجع ov::Output_ويڪٽر{سڌو_output->second};
            }
        }

        خودڪار it = m_translator_map.find(operation_type);
        FRONT_پڇاڙي_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        ڳنڍ_Context ڳنڍ_context(decoder, tensor_map, ڳنڍ_idx, هذا);
        ov::Output_ويڪٽر converted_outputs = it->second(ڳنڍ_context);

        FRONT_پڇاڙي_OP_CONVERSION_CHECK(ڳنڍ_output_نالا.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      ڳنڍ_output_نالا.size(), " and ", converted_outputs.size(), " respectively.");

        لكل (size_t i = 0; i < ڳنڍ_output_نالا.size(); ++i) {
            خودڪار output_name = ڳنڍ_output_نالا[i];
            إذا (i < converted_outputs.size() && converted_outputs[i].get_ڳنڍ_shared_ptr() != خالي_اشارو) {
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
    خودڪار refresh_inplace_aliases = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح inplace_ڳنڍ_idx,
                                       ثابت std::string & view_src_name) {
        لكل (صحيح ڳنڍ_idx = 0; ڳنڍ_idx < inplace_ڳنڍ_idx; ڳنڍ_idx++) {
            إذا (decoder->is_view_like_alias_جو(ڳنڍ_idx, view_src_name)) {
                translate_ڳنڍ(decoder, ڳنڍ_idx);
            }
        }
    };

    خودڪار ڳنڍ_visitor = [&](ثابت std::shared_ptr<GgmlDecoder> & decoder, صحيح ڳنڍ_idx) {
        خودڪار converted_outputs = translate_ڳنڍ(decoder, ڳنڍ_idx);
        إذا (converted_outputs.خالي_جايون()) {
            أرجع;
        }
        ثابت خودڪار inplace_src = decoder->get_inplace_op_src(ڳنڍ_idx);
        إذا (inplace_src.خالي_جايون()) {
            أرجع;
        }
        إذا (converted_outputs[0].get_ڳنڍ_shared_ptr() != خالي_اشارو) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, ڳنڍ_idx, inplace_src);
    };

    إذا (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->گهمو_subgraph(ڳنڍ_visitor);

    لكل (ثابت خودڪار & name : ggml_model_decoder->get_model_output_نالا()) {
        FRONT_پڇاڙي_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        خودڪار نتيجو = std::make_shared<v0::نتيجو_2>(tensor_map->at(name));
        نتيجو->سيٽ_friendly_name(name);
        results.push_back(نتيجو);
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
            خودڪار it = tensor_map->find(name);
            إذا (it == tensor_map->end()) {
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](ثابت خودڪار & داخلا) {
                    أرجع داخلا.پهريون.compare(0, name.size(), name) == 0 && داخلا.پهريون.size() > name.size() &&
                           داخلا.پهريون[name.size()] == '#';
                });
            }
            إذا (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                استمر;
            }
            خودڪار نتيجو = std::make_shared<v0::نتيجو_2>(it->second);
            نتيجو->سيٽ_friendly_name("__debug_" + it->پهريون);
            results.push_back(نتيجو);
        }
    }

    ov::Parameter_ويڪٽر استعمال_ٿيل_params;
    لكل (ثابت خودڪار & param : params) {
        إذا (!param->output(0).get_هدف_inputs().خالي_جايون()) {
            استعمال_ٿيل_params.push_back(param);
        }
    }
    // if (auto diff = params.size() - used_params.size()) {
    //     GGML_LOG_INFO("%zu parameters are not used in the model.", diff);
    // }
    resulting_model = std::make_shared<Model>(results, استعمال_ٿيل_params);

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
    size_t وٿي = 0;
    لكل (خودڪار & ڳنڍ : resulting_model->get_ordered_ops()) {
        إذا (خودڪار cnst = ov::as_type_ptr<ov::op::v0::Constant>(ڳنڍ); cnst && ov::shape_سائيز(cnst->get_shape()) >= 16) {
            خودڪار & rt_info = cnst->get_rt_info();
            إذا (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_سائيز(), وٿي++, cnst->get_element_type());
            }
        }
    }
    أرجع resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    خودڪار ggml_model_decoder = ov::as_type_ptr<داخلا_Model>(m_داخلا_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.سيٽ_في_pass_validation(سچ);
        manager.رجسٽر_pass<ov::pass::MarkCompressedFloatConstants>();
        // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
        // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
        // so it survives ConstantFolding regardless of whether the target plugin's own
        // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
        manager.رجسٽر_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
        manager.رجسٽر_pass<pass::Fuse_منزل_Conv>();
        manager.رجسٽر_pass<pass::Fuse_Argsort_چوٽي_K>();

        // MOECompressed has no CPU plugin implementation, so enable it by default only on GPU.
        // GGML_OPENVINO_MOE_OP=0 keeps the unfused GatherMatmul path for fallback/debugging.
        إذا (ggml_openvino_is_gpu() &&
            ggml_openvino_getenv_int("GGML_OPENVINO_MOE_OP", 1) != 0) {
            manager.رجسٽر_pass<pass::Fuse_Moe_رائوٽر>();
            manager.رجسٽر_pass<pass::FuseMoeCompressed>();
            // Same fusion for models whose gate and up projections share one fused expert
            // weight (gemma-4). It only matches that shape and only when the experts carry an
            // integer zero point, so it is a no-op on the separate-gate/up models above.
            manager.رجسٽر_pass<pass::Fuse_Moe_Compressed_Fused_Gate_مٿي>();
        }

        // Workaround for an OpenVINO GPU-plugin defect: an eltwise op whose operands differ in
        // rank is computed wrongly once the plugin fuses it as a post-op into `rms`. gemma-4
        // dense under stateful execution adds a rank-3 residual to the rank-4 norm output and
        // decodes garbage on GPU while CPU is correct. Equalising the ranks is a no-op for
        // NUMPY broadcasting and makes the fused path correct.
        // Remove once the plugin guards that fusion. Opt out with
        // GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN=1.
        إذا (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
            manager.رجسٽر_pass<pass::Align_Eltwise_عمليو_Ranks>();
        }

        إذا (ggml_model_decoder->is_stateful()) {
            ثابت خودڪار kv_param_res_نالا = ggml_model_decoder->get_kv_param_res_نالا();
            ثابت خودڪار kv_param_res_جوڙا = get_kv_param_res_جوڙا(model, kv_param_res_نالا);
            manager.رجسٽر_pass<ov::pass::MakeStateful>(kv_param_res_جوڙا);
            // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
            إذا (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
                manager.رجسٽر_pass<pass::KVStateSeqAxis>();
            }
        }

        إذا (ggml_model_decoder->is_static()) {
            manager.رجسٽر_pass<pass::SqueezeMatmul>();
        }
        manager.لاڳيتو_passes(model);
        إذا (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            لكل (size_t i = 0; i < model->get_output_سائيز(); i++) {
                خودڪار model_output_shape = model->output(i).get_partial_shape();
                إذا (model_output_shape.rank().is_static() && model_output_shape.rank().get_ڊيگهائي() == 3) {
                    ppp.output(i).postprocess().custom([](ثابت ov::Output<ov::Node>& ڳنڍ) {
                        خودڪار axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        أرجع std::make_shared<ov::op::v0::Unsqueeze>(ڳنڍ, axes);
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
