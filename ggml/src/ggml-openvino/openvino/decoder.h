#pragma once

#अन्तर्भाव <cstdint>
#अन्तर्भाव <map>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/partial_shape.hpp>
#अन्तर्भाव <openvino/core/shape.hpp>
#अन्तर्भाव <openvino/frontend/decoder.hpp>
#अन्तर्भाव <set>
#अन्तर्भाव <string>

namespace ov {
namespace frontend {
namespace ggml {

संरचना Model_आदान_Info {
    element::Type type;
    PartialShape shape;
};

संरचना Model_Extra_आदान_Info {
    element::Type type;
    Shape shape;
    int64_t मूल्य;
    bool is_parameter;
};

class GgmlDecoder : सार्वजनिक DecoderBase {
सार्वजनिक:
    आभास ov::Any get_attribute(ध्रुव std::string & name) ध्रुव = 0;

    आभास PartialShape get_आदान_shape(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name) ध्रुव = 0;

    आभास std::vector<size_t> get_आदान_stride(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name) ध्रुव = 0;

    आभास size_t get_view_आदान_size(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name) ध्रुव = 0;

    आभास size_t get_view_आदान_offset(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name, size_t view_index) ध्रुव = 0;

    आभास size_t get_view_आदान_src_offset(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name, size_t view_index) ध्रुव = 0;

    आभास std::vector<size_t> get_view_आदान_stride(पूर्णाङ्क गाठ_idx,
                                                      ध्रुव std::string & name,
                                                      size_t view_index) ध्रुव = 0;

    आभास std::vector<size_t> get_view_आदान_src_stride(पूर्णाङ्क गाठ_idx,
                                                          ध्रुव std::string & name,
                                                          size_t view_index) ध्रुव = 0;

    आभास Shape get_view_आदान_ggml_shape(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name, size_t view_index) ध्रुव = 0;

    आभास Shape get_view_आदान_src_ggml_shape(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name, size_t view_index) ध्रुव = 0;

    आभास PartialShape get_view_आदान_ov_shape(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name, size_t view_index) ध्रुव = 0;

    आभास PartialShape get_view_आदान_src_ov_shape(पूर्णाङ्क गाठ_idx,
                                                     ध्रुव std::string & name,
                                                     size_t view_index) ध्रुव = 0;

    आभास std::string get_view_आदान_name(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name, size_t view_index) ध्रुव = 0;

    आभास std::string get_view_आदान_src_name(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name, size_t view_index) ध्रुव = 0;

    आभास element::Type get_आदान_type(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name) ध्रुव = 0;

    आभास size_t get_आदान_size() ध्रुव = 0;

    आभास size_t get_आदान_size(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास शून्य get_आदान_गाठ(size_t आदान_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ध्रुव = 0;

    आभास std::vector<std::string> get_आदान_names(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास PartialShape get_output_shape(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास element::Type get_output_type(ध्रुव पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास std::vector<size_t> get_output_stride(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास int32_t * get_आदान_op_params(पूर्णाङ्क गाठ_idx, ध्रुव std::string & name) ध्रुव = 0;

    आभास int32_t * get_output_op_params(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास size_t get_output_op_offset(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास std::vector<std::string> get_output_names(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास std::string get_inplace_op_src(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास bool is_view_like_alias_of(पूर्णाङ्क गाठ_idx, ध्रुव std::string & view_src_name) ध्रुव = 0;

    आभास ध्रुव std::string & get_op_type() ध्रुव = 0;

    आभास ध्रुव std::string & get_op_type(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास ध्रुव std::string & get_op_name() ध्रुव = 0;

    आभास ध्रुव std::string & get_op_name(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास शून्य visit_subgraph(std::function<शून्य(std::shared_ptr<GgmlDecoder>, पूर्णाङ्क गाठ_idx)> गाठ_visitor) ध्रुव = 0;

    आभास पूर्णाङ्क get_op_case(पूर्णाङ्क गाठ_idx) ध्रुव = 0;

    आभास ध्रुव std::map<std::string, Model_आदान_Info> & get_model_inputs() ध्रुव = 0;
    आभास ध्रुव std::map<std::string, Model_Extra_आदान_Info> & get_model_extra_inputs() ध्रुव = 0;
    आभास ध्रुव std::map<std::string, std::shared_ptr<ov::गाठ_2>> & get_model_weights() ध्रुव = 0;
    आभास std::set<std::string> get_model_output_names() ध्रुव = 0;

    आभास int32_t * get_rope_params() ध्रुव = 0;

    आभास bool has_mixed_rope_params() ध्रुव = 0;

    आभास पूर्णाङ्क get_ssm_state_size() ध्रुव = 0;

    आभास std::map<std::string, std::string> get_kv_param_res_names() ध्रुव = 0;

    आभास bool is_static() ध्रुव = 0;

    आभास bool is_stateful() ध्रुव = 0;

    आभास bool is_splited_model() ध्रुव = 0;

    आभास पूर्णाङ्क is_swa_layer(पूर्णाङ्क layer) ध्रुव = 0;

    आभास int32_t get_op_dynamic_dim(पूर्णाङ्क गाठ_idx) ध्रुव = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
