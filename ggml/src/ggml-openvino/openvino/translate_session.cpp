#अन्तर्भाव "translate_session.h"

#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-openvino/ggml-openvino-extra.h"
#अन्तर्भाव "ggml-openvino/openvino/node_context.h"
#अन्तर्भाव "ggml-openvino/openvino/utils.h"
#अन्तर्भाव "input_model.h"
#अन्तर्भाव "pass/fuse_argsort_topk.h"
#अन्तर्भाव "pass/fuse_moe_router.h"
#अन्तर्भाव "pass/align_eltwise_ranks.h"
#अन्तर्भाव "pass/fuse_moe_compressed.h"
#अन्तर्भाव "pass/fuse_to_conv.h"
#अन्तर्भाव "pass/kv_state_seq_axis.h"
#अन्तर्भाव "pass/mark_decompression_convert_constant_folding.h"
#अन्तर्भाव "pass/mark_dequantization_subgraph.h"
#अन्तर्भाव "pass/squeeze_matmul.h"
#अन्तर्भाव "rt_info/weightless_caching_attributes.hpp"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <map>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/preprocess/pre_post_process.hpp>
#अन्तर्भाव <openvino/core/shape.hpp>
#अन्तर्भाव <openvino/core/type.hpp>
#अन्तर्भाव <openvino/core/type/element_type.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/convert_like.hpp>
#अन्तर्भाव <openvino/op/cos.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/greater_eq.hpp>
#अन्तर्भाव <openvino/op/less.hpp>
#अन्तर्भाव <openvino/op/logical_and.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/parameter.hpp>
#अन्तर्भाव <openvino/op/range.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/result.hpp>
#अन्तर्भाव <openvino/op/select.hpp>
#अन्तर्भाव <openvino/op/sin.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/strided_slice.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <openvino/pass/constant_folding.hpp>
#अन्तर्भाव <openvino/pass/make_stateful.hpp>
#अन्तर्भाव <limits>
#अन्तर्भाव <sstream>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {

using नामाकाश ov::op;

नामाकाश {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(अचर std::string & name,
                                                        अचर Model_आगत_Info & आगत_info) {
    स्वतः param_गाँठो = std::make_shared<ov::op::v0::Parameter>(आगत_info.type, आगत_info.shape);
    param_गाँठो->set_friendly_name(name);
    param_गाँठो->output(0).get_tensor().set_names({name});
    फर्काऊ param_गाँठो;
}

std::shared_ptr<ov::Node> create_extra_आगत(अचर std::string & name, अचर Model_Extra_आगत_Info & आगत_info) {
    यदि (आगत_info.is_parameter) {
        स्वतः param_गाँठो = std::make_shared<ov::op::v0::Parameter>(आगत_info.type, आगत_info.shape);
        param_गाँठो->set_friendly_name(name);
        param_गाँठो->output(0).get_tensor().set_names({name});
        फर्काऊ param_गाँठो;
    }

    स्वतः constant = std::make_shared<ov::op::v0::Constant>(आगत_info.type, आगत_info.shape,
                                                          std::vector<int64_t>{आगत_info.value});
    constant->set_friendly_name(name);
    फर्काऊ constant;
}

ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    अचर std::shared_ptr<ov::Model> & model,
    अचर std::map<std::string, std::string> & kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    अचर स्वतः & params = model->get_parameters();
    अचर स्वतः & results = model->get_results();

    लागि (अचर स्वतः & param_res : kv_param_res_names) {
        अचर स्वतः & param_name = param_res.first;
        अचर स्वतः & res_name = param_res.second;

        स्वतः param_it = std::find_if(params.begin(), params.end(), [&](अचर std::shared_ptr<v0::Parameter> & गाँठो) {
            फर्काऊ गाँठो->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        स्वतः res_it = std::find_if(results.begin(), results.end(), [&](अचर std::shared_ptr<v0::Result> & गाँठो) {
            फर्काऊ गाँठो->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::Result> res = *res_it;
        pairs.emplace_back(param, res);
    }
    फर्काऊ pairs;
}

शून्य add_sliced_mask_stateful(TensorMap & tensor_map, तार्किक imrope) {
    स्वतः create_sliced_mask = [&](अचर std::string & mask_name, अचर std::string & sliced_name) {
        यदि ((tensor_map.find(mask_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end()) &&
            (tensor_map.find("inp_pos") != tensor_map.end())) {
            स्वतः token_len_per_seq = tensor_map.at("token_len_per_seq").get_गाँठो_shared_ptr();
            स्वतः mask = tensor_map.at(mask_name).get_गाँठो_shared_ptr();
            std::shared_ptr<ov::Node> mask_sliced = mask;
            स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            स्वतः zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            स्वतः three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            स्वतः neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            स्वतः step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            स्वतः axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            स्वतः inp_pos = tensor_map.at("inp_pos").get_गाँठो_shared_ptr();
            // IMROPE's fourth position plane is zero for text; use the token's first plane.
            ov::Output<ov::Node> last_index = neg_one;
            यदि (imrope) {
                last_index = std::make_shared<ov::op::v1::Add>(token_len_per_seq, neg_one);
            }
            स्वतः last_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, last_index, three);
            स्वतः last_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                last_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), असत्य);
            स्वतः last_inp_pos_cvt = std::make_shared<ov::op::v0::Convert>(last_inp_pos_1d, ov::element::i64);
            स्वतः last_inp_pos_inc = std::make_shared<ov::op::v1::Add>(last_inp_pos_cvt, one);

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
शून्य add_position_mask_stateful_swa(TensorMap & tensor_map) {
    यदि (tensor_map.find("self_kq_mask_swa") == tensor_map.end() || tensor_map.find("inp_pos") == tensor_map.end() ||
        tensor_map.find("swa_window") == tensor_map.end()) {
        फर्काऊ;
    }

    स्वतः inp_pos = tensor_map.at("inp_pos").get_गाँठो_shared_ptr();

    स्वतः zero_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    स्वतः one_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वतः three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    स्वतः neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    स्वतः query_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::i64);
    स्वतः query_pos_1d = std::make_shared<ov::op::v1::Reshape>(
        query_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), असत्य);

    स्वतः last_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
    स्वतः last_pos_1d = std::make_shared<ov::op::v1::Reshape>(last_pos, one_i64, असत्य);
    स्वतः last_pos_cvt = std::make_shared<ov::op::v0::Convert>(last_pos_1d, ov::element::i64);
    स्वतः total_len = std::make_shared<ov::op::v1::Add>(last_pos_cvt, one_i64);
    स्वतः total_len_scalar = std::make_shared<ov::op::v0::Squeeze>(total_len);

    स्वतः cached_pos = std::make_shared<ov::op::v4::Range>(
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}), total_len_scalar,
        ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);

    स्वतः query_col = std::make_shared<ov::op::v1::Reshape>(
        query_pos_1d, ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1}), असत्य);
    स्वतः cached_row = std::make_shared<ov::op::v1::Reshape>(
        cached_pos, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, -1}), असत्य);
    स्वतः diff = std::make_shared<ov::op::v1::Subtract>(query_col, cached_row);

    स्वतः swa_window = tensor_map.at("swa_window").get_गाँठो_shared_ptr();
    स्वतः window = std::make_shared<ov::op::v0::Convert>(swa_window, ov::element::i64);
    स्वतः causal_ok = std::make_shared<ov::op::v1::GreaterEqual>(diff, zero_i64);
    स्वतः window_ok = std::make_shared<ov::op::v1::Less>(diff, window);
    स्वतः keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_ok, window_ok);

    स्वतः zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    स्वतः neg_inf_f = ov::op::v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<भिन्न>::infinity()});
    std::shared_ptr<ov::Node> mask = std::make_shared<ov::op::v1::Select>(keep, zero_f, neg_inf_f);

    स्वतः batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    mask = std::make_shared<ov::op::v0::Unsqueeze>(mask, batch_axis);
    mask = std::make_shared<ov::op::v0::Unsqueeze>(mask, batch_axis);
    mask = std::make_shared<ov::op::v0::Convert>(mask, ov::element::f16);
    mask->set_friendly_name("KQ_mask_swa_sliced");

    tensor_map["KQ_mask_swa_sliced"] = mask->output(0);
}

शून्य add_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    यदि (ggml_model_decoder.has_mixed_rope_params()) {
        फर्काऊ;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    यदि (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == nullptr) {
        फर्काऊ;
    }
    स्वतः inp_pos = tensor_map.at("inp_pos").get_गाँठो_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    यदि (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_गाँठो_shared_ptr();
    }

    स्वतः sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_weight);
    स्वतः sin_theta = sin_cos.first;
    स्वतः cos_theta = sin_cos.second;

    cos_theta.get_गाँठो_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_गाँठो_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
शून्य preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    यदि (ggml_model_decoder.is_stateful()) {
        add_sliced_mask_stateful(tensor_map, ggml_model_decoder.get_rope_params()[2] == GGML_ROPE_TYPE_IMROPE);
        add_position_mask_stateful_swa(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(अचर frontend::आगत_Model::Ptr & आगत_model,
                                   अचर std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   तार्किक naive) :
    m_आगत_model(आगत_model),
    m_translator_map(translator_map),
    m_ov_model(nullptr),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    यदि (m_ov_model) {
        फर्काऊ m_ov_model;
    }
    m_ov_model = translate_graph(m_आगत_model);
    फर्काऊ m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(अचर frontend::आगत_Model::Ptr & आगत_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    स्वतः tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    अचर स्वतः & ggml_model = ov::as_type_ptr<आगत_Model>(आगत_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    लागि (अचर स्वतः & it : ggml_model_decoder->get_model_inputs()) {
        स्वतः param_गाँठो = create_parameter(it.first, it.second);
        params.push_back(param_गाँठो);
        (*tensor_map)[it.first] = param_गाँठो;
    }

    लागि (अचर स्वतः & it : ggml_model_decoder->get_model_extra_inputs()) {
        स्वतः आगत_गाँठो = create_extra_आगत(it.first, it.second);
        यदि (it.second.is_parameter) {
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(आगत_गाँठो));
        }
        (*tensor_map)[it.first] = आगत_गाँठो;
    }

    लागि (अचर स्वतः & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    स्वतः translate_गाँठो = [&](अचर std::shared_ptr<GgmlDecoder> & decoder, पूर्णाङ्क गाँठो_idx) {
        स्वतः operation_type = decoder->get_op_type(गाँठो_idx);
        यदि (operation_type == "GGML_OP_NONE") {
            फर्काऊ ov::OutputVector{};
        }

        अचर स्वतः & गाँठो_output_names = decoder->get_output_names(गाँठो_idx);
        यदि (operation_type == "GGML_OP_VIEW" && decoder->get_op_case(गाँठो_idx) == 2 && गाँठो_output_names.size() == 1) {
            स्वतः direct_output = tensor_map->find(गाँठो_output_names[0]);
            यदि (direct_output != tensor_map->end()) {
                // GDN publishes its native attention/state outputs under the two GGML VIEW names.
                // Keep those mappings instead of rebuilding slices of a packed temporary.
                फर्काऊ ov::OutputVector{direct_output->second};
            }
        }

        स्वतः it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        गाँठो_Context गाँठो_context(decoder, tensor_map, गाँठो_idx, एतत्);
        ov::OutputVector converted_outputs = it->second(गाँठो_context);

        FRONT_END_OP_CONVERSION_CHECK(गाँठो_output_names.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      गाँठो_output_names.size(), " and ", converted_outputs.size(), " respectively.");

        लागि (माप_प्रकार i = 0; i < गाँठो_output_names.size(); ++i) {
            स्वतः output_name = गाँठो_output_names[i];
            यदि (i < converted_outputs.size() && converted_outputs[i].get_गाँठो_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
        फर्काऊ converted_outputs;
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
    स्वतः refresh_inplace_aliases = [&](अचर std::shared_ptr<GgmlDecoder> & decoder, पूर्णाङ्क inplace_गाँठो_idx,
                                       अचर std::string & view_src_name) {
        लागि (पूर्णाङ्क गाँठो_idx = 0; गाँठो_idx < inplace_गाँठो_idx; गाँठो_idx++) {
            यदि (decoder->is_view_like_alias_of(गाँठो_idx, view_src_name)) {
                translate_गाँठो(decoder, गाँठो_idx);
            }
        }
    };

    स्वतः गाँठो_visitor = [&](अचर std::shared_ptr<GgmlDecoder> & decoder, पूर्णाङ्क गाँठो_idx) {
        स्वतः converted_outputs = translate_गाँठो(decoder, गाँठो_idx);
        यदि (converted_outputs.empty()) {
            फर्काऊ;
        }
        अचर स्वतः inplace_src = decoder->get_inplace_op_src(गाँठो_idx);
        यदि (inplace_src.empty()) {
            फर्काऊ;
        }
        यदि (converted_outputs[0].get_गाँठो_shared_ptr() != nullptr) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, गाँठो_idx, inplace_src);
    };

    यदि (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->visit_subgraph(गाँठो_visitor);

    लागि (अचर स्वतः & name : ggml_model_decoder->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        स्वतः result = std::make_shared<v0::Result>(tensor_map->at(name));
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
    यदि (अचर वर्ण * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        std::stringstream ss(debug_nodes);
        std::string name;
        जबसम्म (std::getline(ss, name, ',')) {
            स्वतः it = tensor_map->find(name);
            यदि (it == tensor_map->end()) {
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](अचर स्वतः & entry) {
                    फर्काऊ entry.first.compare(0, name.size(), name) == 0 && entry.first.size() > name.size() &&
                           entry.first[name.size()] == '#';
                });
            }
            यदि (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                जारी;
            }
            स्वतः result = std::make_shared<v0::Result>(it->second);
            result->set_friendly_name("__debug_" + it->first);
            results.push_back(result);
        }
    }

    ov::ParameterVector used_params;
    लागि (अचर स्वतः & param : params) {
        यदि (!param->output(0).get_target_inputs().empty()) {
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
    माप_प्रकार offset = 0;
    लागि (स्वतः & गाँठो : resulting_model->get_ordered_ops()) {
        यदि (स्वतः cnst = ov::as_type_ptr<ov::op::v0::Constant>(गाँठो); cnst && ov::shape_size(cnst->get_shape()) >= 16) {
            स्वतः & rt_info = cnst->get_rt_info();
            यदि (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_size(), offset++, cnst->get_element_type());
            }
        }
    }
    फर्काऊ resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    स्वतः ggml_model_decoder = ov::as_type_ptr<आगत_Model>(m_आगत_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(सत्य);
        manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
        // Marks the Convert/Subtract/Multiply nodes of our GatherMatmul dequantization chain
        // (make_int4_weights/make_int8_weights, for_gather_matmul=true) with disable_constant_folding,
        // so it survives ConstantFolding regardless of whether the target plugin's own
        // is_decompression_multiply() recognizes GatherMatmul as a valid consumer.
        manager.register_pass<ov::pass::MarkDequantization>(
            std::vector<ov::element::Type>{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
        manager.register_pass<pass::FuseToConv>();
        manager.register_pass<pass::Fuse_Argsort_शीर्ष_K>();

        // MOECompressed has no CPU plugin implementation, so enable it by default only on GPU.
        // GGML_OPENVINO_MOE_OP=0 keeps the unfused GatherMatmul path for fallback/debugging.
        यदि (ggml_openvino_is_gpu() &&
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
        यदि (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
            manager.register_pass<pass::AlignEltwiseOperandRanks>();
        }

        यदि (ggml_model_decoder->is_stateful()) {
            अचर स्वतः kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
            अचर स्वतः kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
            manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
            // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
            यदि (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
                manager.register_pass<pass::KVStateSeqAxis>();
            }
        }

        यदि (ggml_model_decoder->is_static()) {
            manager.register_pass<pass::SqueezeMatmul>();
        }
        manager.run_passes(model);
        यदि (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            लागि (माप_प्रकार i = 0; i < model->get_output_size(); i++) {
                स्वतः model_output_shape = model->output(i).get_partial_shape();
                यदि (model_output_shape.rank().is_static() && model_output_shape.rank().get_length() == 3) {
                    ppp.output(i).postprocess().custom([](अचर ov::Output<ov::Node>& गाँठो) {
                        स्वतः axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        फर्काऊ std::make_shared<ov::op::v0::Unsqueeze>(गाँठो, axes);
                    });
                }
            }
            model = ppp.build();
        }
    }
    फर्काऊ model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
