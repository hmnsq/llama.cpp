#pragma once

#شامل <cstdint>
#شامل <map>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/partial_shape.hpp>
#شامل <openvino/core/shape.hpp>
#شامل <openvino/frontend/decoder.hpp>
#شامل <set>
#شامل <string>

namespace ov {
namespace frontend {
namespace ggml {

ساختار Model_درون_داد_Info {
    element::Type type;
    PartialShape shape;
};

ساختار Model_Extra_درون_داد_Info {
    element::Type type;
    Shape shape;
    int64_t مقدار;
    bool is_parameter;
};

class GgmlDecoder : عمومی DecoderBase {
عمومی:
    مجازی ov::Any get_attribute(ثابت std::string & name) ثابت = 0;

    مجازی PartialShape get_درون_داد_shape(صحیح گره_idx, ثابت std::string & name) ثابت = 0;

    مجازی std::vector<size_t> get_درون_داد_stride(صحیح گره_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_view_درون_داد_size(صحیح گره_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_view_درون_داد_offset(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    مجازی size_t get_view_درون_داد_src_offset(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    مجازی std::vector<size_t> get_view_درون_داد_stride(صحیح گره_idx,
                                                      ثابت std::string & name,
                                                      size_t view_index) ثابت = 0;

    مجازی std::vector<size_t> get_view_درون_داد_src_stride(صحیح گره_idx,
                                                          ثابت std::string & name,
                                                          size_t view_index) ثابت = 0;

    مجازی Shape get_view_درون_داد_ggml_shape(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    مجازی Shape get_view_درون_داد_src_ggml_shape(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    مجازی PartialShape get_view_درون_داد_ov_shape(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    مجازی PartialShape get_view_درون_داد_src_ov_shape(صحیح گره_idx,
                                                     ثابت std::string & name,
                                                     size_t view_index) ثابت = 0;

    مجازی std::string get_view_درون_داد_name(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    مجازی std::string get_view_درون_داد_src_name(صحیح گره_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    مجازی element::Type get_درون_داد_type(صحیح گره_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_درون_داد_size() ثابت = 0;

    مجازی size_t get_درون_داد_size(صحیح گره_idx) ثابت = 0;

    مجازی تهی get_درون_داد_گره(size_t درون_داد_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ثابت = 0;

    مجازی std::vector<std::string> get_درون_داد_names(صحیح گره_idx) ثابت = 0;

    مجازی PartialShape get_output_shape(صحیح گره_idx) ثابت = 0;

    مجازی element::Type get_output_type(ثابت صحیح گره_idx) ثابت = 0;

    مجازی std::vector<size_t> get_output_stride(صحیح گره_idx) ثابت = 0;

    مجازی int32_t * get_درون_داد_op_params(صحیح گره_idx, ثابت std::string & name) ثابت = 0;

    مجازی int32_t * get_output_op_params(صحیح گره_idx) ثابت = 0;

    مجازی size_t get_output_op_offset(صحیح گره_idx) ثابت = 0;

    مجازی std::vector<std::string> get_output_names(صحیح گره_idx) ثابت = 0;

    مجازی std::string get_inplace_op_src(صحیح گره_idx) ثابت = 0;

    مجازی bool is_view_like_alias_of(صحیح گره_idx, ثابت std::string & view_src_name) ثابت = 0;

    مجازی ثابت std::string & get_op_type() ثابت = 0;

    مجازی ثابت std::string & get_op_type(صحیح گره_idx) ثابت = 0;

    مجازی ثابت std::string & get_op_name() ثابت = 0;

    مجازی ثابت std::string & get_op_name(صحیح گره_idx) ثابت = 0;

    مجازی تهی visit_subgraph(std::function<تهی(std::shared_ptr<GgmlDecoder>, صحیح گره_idx)> گره_visitor) ثابت = 0;

    مجازی صحیح get_op_case(صحیح گره_idx) ثابت = 0;

    مجازی ثابت std::map<std::string, Model_درون_داد_Info> & get_model_inputs() ثابت = 0;
    مجازی ثابت std::map<std::string, Model_Extra_درون_داد_Info> & get_model_extra_inputs() ثابت = 0;
    مجازی ثابت std::map<std::string, std::shared_ptr<ov::گره_2>> & get_model_weights() ثابت = 0;
    مجازی std::set<std::string> get_model_output_names() ثابت = 0;

    مجازی int32_t * get_rope_params() ثابت = 0;

    مجازی bool has_mixed_rope_params() ثابت = 0;

    مجازی صحیح get_ssm_state_size() ثابت = 0;

    مجازی std::map<std::string, std::string> get_kv_param_res_names() ثابت = 0;

    مجازی bool is_static() ثابت = 0;

    مجازی bool is_stateful() ثابت = 0;

    مجازی bool is_splited_model() ثابت = 0;

    مجازی صحیح is_swa_layer(صحیح layer) ثابت = 0;

    مجازی int32_t get_op_dynamic_dim(صحیح گره_idx) ثابت = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
