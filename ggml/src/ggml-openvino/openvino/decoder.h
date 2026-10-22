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

কাঠামো Model_প্রবেশ_Info {
    element::Type type;
    PartialShape shape;
};

কাঠামো Model_Extra_প্রবেশ_Info {
    element::Type type;
    Shape shape;
    int64_t মান;
    bool is_parameter;
};

class GgmlDecoder : সর্বজনীন DecoderBase {
সর্বজনীন:
    আভাসী ov::Any get_attribute(ধ্রুবক std::string & name) ধ্রুবক = 0;

    আভাসী PartialShape get_প্রবেশ_shape(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name) ধ্রুবক = 0;

    আভাসী std::vector<size_t> get_প্রবেশ_stride(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name) ধ্রুবক = 0;

    আভাসী size_t get_view_প্রবেশ_আকার(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name) ধ্রুবক = 0;

    আভাসী size_t get_view_প্রবেশ_offset(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name, size_t view_সূচক) ধ্রুবক = 0;

    আভাসী size_t get_view_প্রবেশ_src_offset(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name, size_t view_সূচক) ধ্রুবক = 0;

    আভাসী std::vector<size_t> get_view_প্রবেশ_stride(পূর্ণসংখ্যা গ্রন্থি_idx,
                                                      ধ্রুবক std::string & name,
                                                      size_t view_সূচক) ধ্রুবক = 0;

    আভাসী std::vector<size_t> get_view_প্রবেশ_src_stride(পূর্ণসংখ্যা গ্রন্থি_idx,
                                                          ধ্রুবক std::string & name,
                                                          size_t view_সূচক) ধ্রুবক = 0;

    আভাসী Shape get_view_প্রবেশ_ggml_shape(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name, size_t view_সূচক) ধ্রুবক = 0;

    আভাসী Shape get_view_প্রবেশ_src_ggml_shape(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name, size_t view_সূচক) ধ্রুবক = 0;

    আভাসী PartialShape get_view_প্রবেশ_ov_shape(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name, size_t view_সূচক) ধ্রুবক = 0;

    আভাসী PartialShape get_view_প্রবেশ_src_ov_shape(পূর্ণসংখ্যা গ্রন্থি_idx,
                                                     ধ্রুবক std::string & name,
                                                     size_t view_সূচক) ধ্রুবক = 0;

    আভাসী std::string get_view_প্রবেশ_name(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name, size_t view_সূচক) ধ্রুবক = 0;

    আভাসী std::string get_view_প্রবেশ_src_name(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name, size_t view_সূচক) ধ্রুবক = 0;

    আভাসী element::Type get_প্রবেশ_type(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name) ধ্রুবক = 0;

    আভাসী size_t get_প্রবেশ_আকার() ধ্রুবক = 0;

    আভাসী size_t get_প্রবেশ_আকার(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী শূন্য get_প্রবেশ_গ্রন্থি(size_t প্রবেশ_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_সূচক) ধ্রুবক = 0;

    আভাসী std::vector<std::string> get_প্রবেশ_names(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী PartialShape get_output_shape(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী element::Type get_output_type(ধ্রুবক পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী std::vector<size_t> get_output_stride(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী int32_t * get_প্রবেশ_op_params(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name) ধ্রুবক = 0;

    আভাসী int32_t * get_output_op_params(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী size_t get_output_op_offset(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী std::vector<std::string> get_output_names(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী std::string get_inplace_op_src(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী bool is_view_like_alias_of(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & view_src_name) ধ্রুবক = 0;

    আভাসী ধ্রুবক std::string & get_op_type() ধ্রুবক = 0;

    আভাসী ধ্রুবক std::string & get_op_type(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী ধ্রুবক std::string & get_op_name() ধ্রুবক = 0;

    আভাসী ধ্রুবক std::string & get_op_name(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী শূন্য visit_subgraph(std::function<শূন্য(std::shared_ptr<GgmlDecoder>, পূর্ণসংখ্যা গ্রন্থি_idx)> গ্রন্থি_visitor) ধ্রুবক = 0;

    আভাসী পূর্ণসংখ্যা get_op_case(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;

    আভাসী ধ্রুবক std::map<std::string, Model_প্রবেশ_Info> & get_model_inputs() ধ্রুবক = 0;
    আভাসী ধ্রুবক std::map<std::string, Model_Extra_প্রবেশ_Info> & get_model_extra_inputs() ধ্রুবক = 0;
    আভাসী ধ্রুবক std::map<std::string, std::shared_ptr<ov::গ্রন্থি_2>> & get_model_weights() ধ্রুবক = 0;
    আভাসী std::set<std::string> get_model_output_names() ধ্রুবক = 0;

    আভাসী int32_t * get_rope_params() ধ্রুবক = 0;

    আভাসী bool has_mixed_rope_params() ধ্রুবক = 0;

    আভাসী পূর্ণসংখ্যা get_ssm_state_আকার() ধ্রুবক = 0;

    আভাসী std::map<std::string, std::string> get_kv_param_res_names() ধ্রুবক = 0;

    আভাসী bool is_static() ধ্রুবক = 0;

    আভাসী bool is_stateful() ধ্রুবক = 0;

    আভাসী bool is_splited_model() ধ্রুবক = 0;

    আভাসী পূর্ণসংখ্যা is_swa_layer(পূর্ণসংখ্যা layer) ধ্রুবক = 0;

    আভাসী int32_t get_op_dynamic_dim(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
