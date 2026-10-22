#pragma once

#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <map>
#অন্তর্ভাৱ <openvino/core/node.hpp>
#অন্তর্ভাৱ <openvino/core/partial_shape.hpp>
#অন্তর্ভাৱ <openvino/core/shape.hpp>
#অন্তর্ভাৱ <openvino/frontend/decoder.hpp>
#অন্তর্ভাৱ <set>
#অন্তর্ভাৱ <string>

namespace ov {
namespace frontend {
namespace ggml {

সংরচনা Model_প্ৰবেশ_Info {
    element::Type type;
    PartialShape shape;
};

সংরচনা Model_Extra_প্ৰবেশ_Info {
    element::Type type;
    Shape shape;
    int64_t মান;
    bool is_parameter;
};

class GgmlDecoder : সার্ৱজনিক DecoderBase {
সার্ৱজনিক:
    আভাস ov::Any get_attribute(ধ্রুৱ std::string & name) ধ্রুৱ = 0;

    আভাস PartialShape get_প্ৰবেশ_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name) ধ্রুৱ = 0;

    আভাস std::vector<size_t> get_প্ৰবেশ_stride(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name) ধ্রুৱ = 0;

    আভাস size_t get_view_প্ৰবেশ_size(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name) ধ্রুৱ = 0;

    আভাস size_t get_view_প্ৰবেশ_offset(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ = 0;

    আভাস size_t get_view_প্ৰবেশ_src_offset(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ = 0;

    আভাস std::vector<size_t> get_view_প্ৰবেশ_stride(পূর্ণাঙ্ক গ্ৰন্থি_idx,
                                                      ধ্রুৱ std::string & name,
                                                      size_t view_index) ধ্রুৱ = 0;

    আভাস std::vector<size_t> get_view_প্ৰবেশ_src_stride(পূর্ণাঙ্ক গ্ৰন্থি_idx,
                                                          ধ্রুৱ std::string & name,
                                                          size_t view_index) ধ্রুৱ = 0;

    আভাস Shape get_view_প্ৰবেশ_ggml_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ = 0;

    আভাস Shape get_view_প্ৰবেশ_src_ggml_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ = 0;

    আভাস PartialShape get_view_প্ৰবেশ_ov_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ = 0;

    আভাস PartialShape get_view_প্ৰবেশ_src_ov_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx,
                                                     ধ্রুৱ std::string & name,
                                                     size_t view_index) ধ্রুৱ = 0;

    আভাস std::string get_view_প্ৰবেশ_name(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ = 0;

    আভাস std::string get_view_প্ৰবেশ_src_name(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name, size_t view_index) ধ্রুৱ = 0;

    আভাস element::Type get_প্ৰবেশ_type(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name) ধ্রুৱ = 0;

    আভাস size_t get_প্ৰবেশ_size() ধ্রুৱ = 0;

    আভাস size_t get_প্ৰবেশ_size(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস শূন্য get_প্ৰবেশ_গ্ৰন্থি(size_t প্ৰবেশ_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ধ্রুৱ = 0;

    আভাস std::vector<std::string> get_প্ৰবেশ_names(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস PartialShape get_output_shape(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস element::Type get_output_type(ধ্রুৱ পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস std::vector<size_t> get_output_stride(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস int32_t * get_প্ৰবেশ_op_params(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & name) ধ্রুৱ = 0;

    আভাস int32_t * get_output_op_params(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস size_t get_output_op_offset(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস std::vector<std::string> get_output_names(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস std::string get_inplace_op_src(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস bool is_view_like_alias_of(পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::string & view_src_name) ধ্রুৱ = 0;

    আভাস ধ্রুৱ std::string & get_op_type() ধ্রুৱ = 0;

    আভাস ধ্রুৱ std::string & get_op_type(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস ধ্রুৱ std::string & get_op_name() ধ্রুৱ = 0;

    আভাস ধ্রুৱ std::string & get_op_name(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস শূন্য visit_subgraph(std::function<শূন্য(std::shared_ptr<GgmlDecoder>, পূর্ণাঙ্ক গ্ৰন্থি_idx)> গ্ৰন্থি_visitor) ধ্রুৱ = 0;

    আভাস পূর্ণাঙ্ক get_op_case(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;

    আভাস ধ্রুৱ std::map<std::string, Model_প্ৰবেশ_Info> & get_model_inputs() ধ্রুৱ = 0;
    আভাস ধ্রুৱ std::map<std::string, Model_Extra_প্ৰবেশ_Info> & get_model_extra_inputs() ধ্রুৱ = 0;
    আভাস ধ্রুৱ std::map<std::string, std::shared_ptr<ov::গ্ৰন্থি_2>> & get_model_weights() ধ্রুৱ = 0;
    আভাস std::set<std::string> get_model_output_names() ধ্রুৱ = 0;

    আভাস int32_t * get_rope_params() ধ্রুৱ = 0;

    আভাস bool has_mixed_rope_params() ধ্রুৱ = 0;

    আভাস পূর্ণাঙ্ক get_ssm_state_size() ধ্রুৱ = 0;

    আভাস std::map<std::string, std::string> get_kv_param_res_names() ধ্রুৱ = 0;

    আভাস bool is_static() ধ্রুৱ = 0;

    আভাস bool is_stateful() ধ্রুৱ = 0;

    আভাস bool is_splited_model() ধ্রুৱ = 0;

    আভাস পূর্ণাঙ্ক is_swa_layer(পূর্ণাঙ্ক layer) ধ্রুৱ = 0;

    আভাস int32_t get_op_dynamic_dim(পূর্ণাঙ্ক গ্ৰন্থি_idx) ধ্রুৱ = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
