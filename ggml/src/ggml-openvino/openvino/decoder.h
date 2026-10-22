#pragma once

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <map>
#অন্তর্ভাব <openvino/core/node.hpp>
#অন্তর্ভাব <openvino/core/partial_shape.hpp>
#অন্তর্ভাব <openvino/core/shape.hpp>
#অন্তর্ভাব <openvino/frontend/decoder.hpp>
#অন্তর্ভাব <set>
#অন্তর্ভাব <string>

namespace ov {
namespace frontend {
namespace ggml {

সংরচনা Model_প্রবেশ_Info {
    element::Type type;
    PartialShape shape;
};

সংরচনা Model_Extra_প্রবেশ_Info {
    element::Type type;
    Shape shape;
    int64_t মান;
    bool is_parameter;
};

class GgmlDecoder : সার্বজনিক DecoderBase {
সার্বজনিক:
    আভাস ov::Any get_attribute(ধ্রুব std::string & name) ধ্রুব = 0;

    আভাস PartialShape get_প্রবেশ_shape(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name) ধ্রুব = 0;

    আভাস std::vector<size_t> get_প্রবেশ_stride(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name) ধ্রুব = 0;

    আভাস size_t get_view_প্রবেশ_size(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name) ধ্রুব = 0;

    আভাস size_t get_view_প্রবেশ_offset(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name, size_t view_index) ধ্রুব = 0;

    আভাস size_t get_view_প্রবেশ_src_offset(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name, size_t view_index) ধ্রুব = 0;

    আভাস std::vector<size_t> get_view_প্রবেশ_stride(পূর্ণাঙ্ক গ্রন্থি_idx,
                                                      ধ্রুব std::string & name,
                                                      size_t view_index) ধ্রুব = 0;

    আভাস std::vector<size_t> get_view_প্রবেশ_src_stride(পূর্ণাঙ্ক গ্রন্থি_idx,
                                                          ধ্রুব std::string & name,
                                                          size_t view_index) ধ্রুব = 0;

    আভাস Shape get_view_প্রবেশ_ggml_shape(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name, size_t view_index) ধ্রুব = 0;

    আভাস Shape get_view_প্রবেশ_src_ggml_shape(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name, size_t view_index) ধ্রুব = 0;

    আভাস PartialShape get_view_প্রবেশ_ov_shape(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name, size_t view_index) ধ্রুব = 0;

    আভাস PartialShape get_view_প্রবেশ_src_ov_shape(পূর্ণাঙ্ক গ্রন্থি_idx,
                                                     ধ্রুব std::string & name,
                                                     size_t view_index) ধ্রুব = 0;

    আভাস std::string get_view_প্রবেশ_name(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name, size_t view_index) ধ্রুব = 0;

    আভাস std::string get_view_প্রবেশ_src_name(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name, size_t view_index) ধ্রুব = 0;

    আভাস element::Type get_প্রবেশ_type(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name) ধ্রুব = 0;

    আভাস size_t get_প্রবেশ_size() ধ্রুব = 0;

    আভাস size_t get_প্রবেশ_size(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস শূন্য get_প্রবেশ_গ্রন্থি(size_t প্রবেশ_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ধ্রুব = 0;

    আভাস std::vector<std::string> get_প্রবেশ_names(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস PartialShape get_output_shape(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস element::Type get_output_type(ধ্রুব পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস std::vector<size_t> get_output_stride(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস int32_t * get_প্রবেশ_op_params(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & name) ধ্রুব = 0;

    আভাস int32_t * get_output_op_params(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস size_t get_output_op_offset(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস std::vector<std::string> get_output_names(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস std::string get_inplace_op_src(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস bool is_view_like_alias_of(পূর্ণাঙ্ক গ্রন্থি_idx, ধ্রুব std::string & view_src_name) ধ্রুব = 0;

    আভাস ধ্রুব std::string & get_op_type() ধ্রুব = 0;

    আভাস ধ্রুব std::string & get_op_type(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস ধ্রুব std::string & get_op_name() ধ্রুব = 0;

    আভাস ধ্রুব std::string & get_op_name(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস শূন্য visit_subgraph(std::function<শূন্য(std::shared_ptr<GgmlDecoder>, পূর্ণাঙ্ক গ্রন্থি_idx)> গ্রন্থি_visitor) ধ্রুব = 0;

    আভাস পূর্ণাঙ্ক get_op_case(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;

    আভাস ধ্রুব std::map<std::string, Model_প্রবেশ_Info> & get_model_inputs() ধ্রুব = 0;
    আভাস ধ্রুব std::map<std::string, Model_Extra_প্রবেশ_Info> & get_model_extra_inputs() ধ্রুব = 0;
    আভাস ধ্রুব std::map<std::string, std::shared_ptr<ov::গ্রন্থি_2>> & get_model_weights() ধ্রুব = 0;
    আভাস std::set<std::string> get_model_output_names() ধ্রুব = 0;

    আভাস int32_t * get_rope_params() ধ্রুব = 0;

    আভাস bool has_mixed_rope_params() ধ্রুব = 0;

    আভাস পূর্ণাঙ্ক get_ssm_state_size() ধ্রুব = 0;

    আভাস std::map<std::string, std::string> get_kv_param_res_names() ধ্রুব = 0;

    আভাস bool is_static() ধ্রুব = 0;

    আভাস bool is_stateful() ধ্রুব = 0;

    আভাস bool is_splited_model() ধ্রুব = 0;

    আভাস পূর্ণাঙ্ক is_swa_layer(পূর্ণাঙ্ক layer) ধ্রুব = 0;

    আভাস int32_t get_op_dynamic_dim(পূর্ণাঙ্ক গ্রন্থি_idx) ধ্রুব = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
