#समावेश "translate_session.h"

#समावेश "ggml-impl.h"
#समावेश "ggml-openvino/ggml-openvino-extra.h"
#समावेश "ggml-openvino/openvino/node_context.h"
#समावेश "ggml-openvino/openvino/utils.h"
#समावेश "input_model.h"
#समावेश "pass/fuse_argsort_topk.h"
#समावेश "pass/fuse_moe_router.h"
#समावेश "pass/align_eltwise_ranks.h"
#समावेश "pass/fuse_moe_compressed.h"
#समावेश "pass/fuse_to_conv.h"
#समावेश "pass/kv_state_seq_axis.h"
#समावेश "pass/mark_decompression_convert_constant_folding.h"
#समावेश "pass/mark_dequantization_subgraph.h"
#समावेश "pass/squeeze_matmul.h"
#समावेश "rt_info/weightless_caching_attributes.hpp"

#समावेश <algorithm>
#समावेश <cstdint>
#समावेश <cstdlib>
#समावेश <map>
#समावेश <memory>
#समावेश <openvino/core/node.hpp>
#समावेश <openvino/core/preprocess/pre_post_process.hpp>
#समावेश <openvino/core/shape.hpp>
#समावेश <openvino/core/type.hpp>
#समावेश <openvino/core/type/element_type.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/broadcast.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/convert_like.hpp>
#समावेश <openvino/op/cos.hpp>
#समावेश <openvino/op/divide.hpp>
#समावेश <openvino/op/gather.hpp>
#समावेश <openvino/op/greater_eq.hpp>
#समावेश <openvino/op/less.hpp>
#समावेश <openvino/op/logical_and.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/parameter.hpp>
#समावेश <openvino/op/range.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/result.hpp>
#समावेश <openvino/op/select.hpp>
#समावेश <openvino/op/sin.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/squeeze.hpp>
#समावेश <openvino/op/strided_slice.hpp>
#समावेश <openvino/op/subtract.hpp>
#समावेश <openvino/op/transpose.hpp>
#समावेश <openvino/op/unsqueeze.hpp>
#समावेश <openvino/pass/constant_folding.hpp>
#समावेश <openvino/pass/make_stateful.hpp>
#समावेश <limits>
#समावेश <sstream>

namespace ov {
namespace frontend {
namespace ggml {

using namespace ov::op;

namespace {

std::shared_ptr<ov::op::v0::Parameter> create_parameter(स्थिर std::string & name,
                                                        स्थिर Model_निवेश_Info & निवेश_info) {
    स्वतः param_गाँठ = std::make_shared<ov::op::v0::Parameter>(निवेश_info.type, निवेश_info.shape);
    param_गाँठ->set_friendly_name(name);
    param_गाँठ->output(0).get_tensor().set_names({name});
    वापस param_गाँठ;
}

std::shared_ptr<ov::Node> create_extra_निवेश(स्थिर std::string & name, स्थिर Model_Extra_निवेश_Info & निवेश_info) {
    अगर (निवेश_info.is_parameter) {
        स्वतः param_गाँठ = std::make_shared<ov::op::v0::Parameter>(निवेश_info.type, निवेश_info.shape);
        param_गाँठ->set_friendly_name(name);
        param_गाँठ->output(0).get_tensor().set_names({name});
        वापस param_गाँठ;
    }

    स्वतः constant = std::make_shared<ov::op::v0::Constant>(निवेश_info.type, निवेश_info.shape,
                                                          std::vector<int64_t>{निवेश_info.value});
    constant->set_friendly_name(name);
    वापस constant;
}

ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    स्थिर std::shared_ptr<ov::Model> & model,
    स्थिर std::map<std::string, std::string> & kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    स्थिर स्वतः & params = model->get_parameters();
    स्थिर स्वतः & results = model->get_results();

    क्रम (स्थिर स्वतः & param_res : kv_param_res_names) {
        स्थिर स्वतः & param_name = param_res.first;
        स्थिर स्वतः & res_name = param_res.second;

        स्वतः param_it = std::find_if(params.begin(), params.end(), [&](स्थिर std::shared_ptr<v0::Parameter> & गाँठ) {
            वापस गाँठ->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        स्वतः res_it = std::find_if(results.begin(), results.end(), [&](स्थिर std::shared_ptr<v0::परिणाम_2> & गाँठ) {
            वापस गाँठ->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::परिणाम_2> res = *res_it;
        pairs.emplace_back(param, res);
    }
    वापस pairs;
}

व्योम add_sliced_mask_stateful(TensorMap & tensor_map, bool imrope) {
    स्वतः create_sliced_mask = [&](स्थिर std::string & mask_name, स्थिर std::string & sliced_name) {
        अगर ((tensor_map.find(mask_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end()) &&
            (tensor_map.find("inp_pos") != tensor_map.end())) {
            स्वतः token_len_per_seq = tensor_map.at("token_len_per_seq").get_गाँठ_shared_ptr();
            स्वतः mask = tensor_map.at(mask_name).get_गाँठ_shared_ptr();
            std::shared_ptr<ov::Node> mask_sliced = mask;
            स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            स्वतः zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            स्वतः three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            स्वतः neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            स्वतः step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            स्वतः axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

            स्वतः inp_pos = tensor_map.at("inp_pos").get_गाँठ_shared_ptr();
            // IMROPE's fourth position plane is zero for text; use the token's first plane.
            ov::Output<ov::Node> last_सूचकांक = neg_one;
            अगर (imrope) {
                last_सूचकांक = std::make_shared<ov::op::v1::Add>(token_len_per_seq, neg_one);
            }
            स्वतः last_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, last_सूचकांक, three);
            स्वतः last_inp_pos_1d = std::make_shared<ov::op::v1::Reshape>(
                last_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), false);
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
व्योम add_position_mask_stateful_swa(TensorMap & tensor_map) {
    अगर (tensor_map.find("self_kq_mask_swa") == tensor_map.end() || tensor_map.find("inp_pos") == tensor_map.end() ||
        tensor_map.find("swa_window") == tensor_map.end()) {
        वापस;
    }

    स्वतः inp_pos = tensor_map.at("inp_pos").get_गाँठ_shared_ptr();

    स्वतः zero_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    स्वतः one_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वतः three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    स्वतः neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    स्वतः query_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::i64);
    स्वतः query_pos_1d = std::make_shared<ov::op::v1::Reshape>(
        query_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);

    स्वतः last_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one, three);
    स्वतः last_pos_1d = std::make_shared<ov::op::v1::Reshape>(last_pos, one_i64, false);
    स्वतः last_pos_cvt = std::make_shared<ov::op::v0::Convert>(last_pos_1d, ov::element::i64);
    स्वतः योग_len = std::make_shared<ov::op::v1::Add>(last_pos_cvt, one_i64);
    स्वतः योग_len_scalar = std::make_shared<ov::op::v0::Squeeze>(योग_len);

    स्वतः cached_pos = std::make_shared<ov::op::v4::Range>(
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}), योग_len_scalar,
        ov::op::v0::Constant::create(ov::element::i64, {}, {1}), ov::element::i64);

    स्वतः query_col = std::make_shared<ov::op::v1::Reshape>(
        query_pos_1d, ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1}), false);
    स्वतः cached_row = std::make_shared<ov::op::v1::Reshape>(
        cached_pos, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, -1}), false);
    स्वतः diff = std::make_shared<ov::op::v1::Subtract>(query_col, cached_row);

    स्वतः swa_window = tensor_map.at("swa_window").get_गाँठ_shared_ptr();
    स्वतः window = std::make_shared<ov::op::v0::Convert>(swa_window, ov::element::i64);
    स्वतः causal_ok = std::make_shared<ov::op::v1::GreaterEqual>(diff, zero_i64);
    स्वतः window_ok = std::make_shared<ov::op::v1::Less>(diff, window);
    स्वतः keep = std::make_shared<ov::op::v1::LogicalAnd>(causal_ok, window_ok);

    स्वतः zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    स्वतः neg_inf_f = ov::op::v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<भग्न>::infinity()});
    std::shared_ptr<ov::Node> mask = std::make_shared<ov::op::v1::Select>(keep, zero_f, neg_inf_f);

    स्वतः batch_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    mask = std::make_shared<ov::op::v0::Unsqueeze>(mask, batch_axis);
    mask = std::make_shared<ov::op::v0::Unsqueeze>(mask, batch_axis);
    mask = std::make_shared<ov::op::v0::Convert>(mask, ov::element::f16);
    mask->set_friendly_name("KQ_mask_swa_sliced");

    tensor_map["KQ_mask_swa_sliced"] = mask->output(0);
}

व्योम add_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    // When ROPE ops in the graph have divergent op_params (e.g. gemma4's mixed
    // SWA/non-SWA layers with different n_dims or freq_base), a shared sin/cos
    // precompute cannot broadcast across every ROPE use. Skip it here and let
    // translate_rope() build sin/cos per-op from its own op_params.
    अगर (ggml_model_decoder.has_mixed_rope_params()) {
        वापस;
    }
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    अगर (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == nullptr) {
        वापस;
    }
    स्वतः inp_pos = tensor_map.at("inp_pos").get_गाँठ_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    अगर (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_गाँठ_shared_ptr();
    }

    स्वतः sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_weight);
    स्वतः sin_theta = sin_cos.first;
    स्वतः cos_theta = sin_cos.second;

    cos_theta.get_गाँठ_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_गाँठ_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
व्योम preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    अगर (ggml_model_decoder.is_stateful()) {
        add_sliced_mask_stateful(tensor_map, ggml_model_decoder.get_rope_params()[2] == GGML_ROPE_TYPE_IMROPE);
        add_position_mask_stateful_swa(tensor_map);
    }
    // This optimization is error-prone
    // add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(स्थिर frontend::निवेश_Model::Ptr & निवेश_model,
                                   स्थिर std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   bool naive) :
    m_निवेश_model(निवेश_model),
    m_translator_map(translator_map),
    m_ov_model(nullptr),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    अगर (m_ov_model) {
        वापस m_ov_model;
    }
    m_ov_model = translate_graph(m_निवेश_model);
    वापस m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(स्थिर frontend::निवेश_Model::Ptr & निवेश_model) {
    ov::ParameterVector params;
    ov::परिणाम_Vector results;
    स्वतः tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    स्थिर स्वतः & ggml_model = ov::as_type_ptr<निवेश_Model>(निवेश_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    क्रम (स्थिर स्वतः & it : ggml_model_decoder->get_model_inputs()) {
        स्वतः param_गाँठ = create_parameter(it.first, it.second);
        params.push_back(param_गाँठ);
        (*tensor_map)[it.first] = param_गाँठ;
    }

    क्रम (स्थिर स्वतः & it : ggml_model_decoder->get_model_extra_inputs()) {
        स्वतः निवेश_गाँठ = create_extra_निवेश(it.first, it.second);
        अगर (it.second.is_parameter) {
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(निवेश_गाँठ));
        }
        (*tensor_map)[it.first] = निवेश_गाँठ;
    }

    क्रम (स्थिर स्वतः & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    स्वतः translate_गाँठ = [&](स्थिर std::shared_ptr<GgmlDecoder> & decoder, पूर्णांक गाँठ_idx) {
        स्वतः operation_type = decoder->get_op_type(गाँठ_idx);
        अगर (operation_type == "GGML_OP_NONE") {
            वापस ov::OutputVector{};
        }

        स्थिर स्वतः & गाँठ_output_names = decoder->get_output_names(गाँठ_idx);
        अगर (operation_type == "GGML_OP_VIEW" && decoder->get_op_case(गाँठ_idx) == 2 && गाँठ_output_names.size() == 1) {
            स्वतः direct_output = tensor_map->find(गाँठ_output_names[0]);
            अगर (direct_output != tensor_map->end()) {
                // GDN publishes its native attention/state outputs under the two GGML VIEW names.
                // Keep those mappings instead of rebuilding slices of a packed temporary.
                वापस ov::OutputVector{direct_output->second};
            }
        }

        स्वतः it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        गाँठ_Context गाँठ_context(decoder, tensor_map, गाँठ_idx, यह);
        ov::OutputVector converted_outputs = it->second(गाँठ_context);

        FRONT_END_OP_CONVERSION_CHECK(गाँठ_output_names.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      गाँठ_output_names.size(), " and ", converted_outputs.size(), " respectively.");

        क्रम (माप_प्रकार i = 0; i < गाँठ_output_names.size(); ++i) {
            स्वतः output_name = गाँठ_output_names[i];
            अगर (i < converted_outputs.size() && converted_outputs[i].get_गाँठ_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
        वापस converted_outputs;
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
    स्वतः refresh_inplace_aliases = [&](स्थिर std::shared_ptr<GgmlDecoder> & decoder, पूर्णांक inplace_गाँठ_idx,
                                       स्थिर std::string & view_src_name) {
        क्रम (पूर्णांक गाँठ_idx = 0; गाँठ_idx < inplace_गाँठ_idx; गाँठ_idx++) {
            अगर (decoder->is_view_like_alias_of(गाँठ_idx, view_src_name)) {
                translate_गाँठ(decoder, गाँठ_idx);
            }
        }
    };

    स्वतः गाँठ_visitor = [&](स्थिर std::shared_ptr<GgmlDecoder> & decoder, पूर्णांक गाँठ_idx) {
        स्वतः converted_outputs = translate_गाँठ(decoder, गाँठ_idx);
        अगर (converted_outputs.empty()) {
            वापस;
        }
        स्थिर स्वतः inplace_src = decoder->get_inplace_op_src(गाँठ_idx);
        अगर (inplace_src.empty()) {
            वापस;
        }
        अगर (converted_outputs[0].get_गाँठ_shared_ptr() != nullptr) {
            (*tensor_map)[inplace_src] = converted_outputs[0];
        }
        refresh_inplace_aliases(decoder, गाँठ_idx, inplace_src);
    };

    अगर (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->visit_subgraph(गाँठ_visitor);

    क्रम (स्थिर स्वतः & name : ggml_model_decoder->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        स्वतः परिणाम = std::make_shared<v0::परिणाम_2>(tensor_map->at(name));
        परिणाम->set_friendly_name(name);
        results.push_back(परिणाम);
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
    अगर (स्थिर अक्षर * debug_nodes = ggml_openvino_getenv_str("GGML_OPENVINO_DEBUG_NODE")) {
        std::stringstream ss(debug_nodes);
        std::string name;
        जबतक (std::लो_पंक्ति(ss, name, ',')) {
            स्वतः it = tensor_map->find(name);
            अगर (it == tensor_map->end()) {
                it = std::find_if(tensor_map->begin(), tensor_map->end(), [&](स्थिर स्वतः & entry) {
                    वापस entry.first.compare(0, name.size(), name) == 0 && entry.first.size() > name.size() &&
                           entry.first[name.size()] == '#';
                });
            }
            अगर (it == tensor_map->end()) {
                GGML_LOG_WARN("GGML_OPENVINO_DEBUG_NODE: node '%s' not found in tensor map, skipping\n", name.c_str());
                जारी;
            }
            स्वतः परिणाम = std::make_shared<v0::परिणाम_2>(it->second);
            परिणाम->set_friendly_name("__debug_" + it->first);
            results.push_back(परिणाम);
        }
    }

    ov::ParameterVector used_params;
    क्रम (स्थिर स्वतः & param : params) {
        अगर (!param->output(0).get_target_inputs().empty()) {
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
    क्रम (स्वतः & गाँठ : resulting_model->get_ordered_ops()) {
        अगर (स्वतः cnst = ov::as_type_ptr<ov::op::v0::Constant>(गाँठ); cnst && ov::shape_आकार(cnst->get_shape()) >= 16) {
            स्वतः & rt_info = cnst->get_rt_info();
            अगर (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_आकार(), offset++, cnst->get_element_type());
            }
        }
    }
    वापस resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    स्वतः ggml_model_decoder = ov::as_type_ptr<निवेश_Model>(m_निवेश_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(true);
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
        अगर (ggml_openvino_is_gpu() &&
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
        अगर (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
            manager.register_pass<pass::AlignEltwiseOperandRanks>();
        }

        अगर (ggml_model_decoder->is_stateful()) {
            स्थिर स्वतः kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
            स्थिर स्वतः kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
            manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
            // Must run after MakeStateful, which is what creates the ReadValue/Assign pairs.
            अगर (!ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT")) {
                manager.register_pass<pass::KVStateSeqAxis>();
            }
        }

        अगर (ggml_model_decoder->is_static()) {
            manager.register_pass<pass::SqueezeMatmul>();
        }
        manager.run_passes(model);
        अगर (ggml_model_decoder->is_stateful()) {
            ov::preprocess::PrePostProcessor ppp(model);
            क्रम (माप_प्रकार i = 0; i < model->get_output_आकार(); i++) {
                स्वतः model_output_shape = model->output(i).get_partial_shape();
                अगर (model_output_shape.rank().is_static() && model_output_shape.rank().get_लंबाई() == 3) {
                    ppp.output(i).postprocess().custom([](स्थिर ov::Output<ov::Node>& गाँठ) {
                        स्वतः axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        वापस std::make_shared<ov::op::v0::Unsqueeze>(गाँठ, axes);
                    });
                }
            }
            model = ppp.build();
        }
    }
    वापस model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
