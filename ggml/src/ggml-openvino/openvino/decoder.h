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

संरचना Model_आगत_Info {
    element::Type type;
    PartialShape shape;
};

संरचना Model_Extra_आगत_Info {
    element::Type type;
    Shape shape;
    int64_t मान;
    bool is_parameter;
};

class GgmlDecoder : सार्वजनिक DecoderBase {
सार्वजनिक:
    आभास ov::Any get_attribute(अचर std::string & name) अचर = 0;

    आभास PartialShape get_आगत_shape(पूर्णाङ्क गाँठो_idx, अचर std::string & name) अचर = 0;

    आभास std::vector<माप_प्रकार> get_आगत_stride(पूर्णाङ्क गाँठो_idx, अचर std::string & name) अचर = 0;

    आभास माप_प्रकार get_view_आगत_size(पूर्णाङ्क गाँठो_idx, अचर std::string & name) अचर = 0;

    आभास माप_प्रकार get_view_आगत_offset(पूर्णाङ्क गाँठो_idx, अचर std::string & name, माप_प्रकार view_index) अचर = 0;

    आभास माप_प्रकार get_view_आगत_src_offset(पूर्णाङ्क गाँठो_idx, अचर std::string & name, माप_प्रकार view_index) अचर = 0;

    आभास std::vector<माप_प्रकार> get_view_आगत_stride(पूर्णाङ्क गाँठो_idx,
                                                      अचर std::string & name,
                                                      माप_प्रकार view_index) अचर = 0;

    आभास std::vector<माप_प्रकार> get_view_आगत_src_stride(पूर्णाङ्क गाँठो_idx,
                                                          अचर std::string & name,
                                                          माप_प्रकार view_index) अचर = 0;

    आभास Shape get_view_आगत_ggml_shape(पूर्णाङ्क गाँठो_idx, अचर std::string & name, माप_प्रकार view_index) अचर = 0;

    आभास Shape get_view_आगत_src_ggml_shape(पूर्णाङ्क गाँठो_idx, अचर std::string & name, माप_प्रकार view_index) अचर = 0;

    आभास PartialShape get_view_आगत_ov_shape(पूर्णाङ्क गाँठो_idx, अचर std::string & name, माप_प्रकार view_index) अचर = 0;

    आभास PartialShape get_view_आगत_src_ov_shape(पूर्णाङ्क गाँठो_idx,
                                                     अचर std::string & name,
                                                     माप_प्रकार view_index) अचर = 0;

    आभास std::string get_view_आगत_name(पूर्णाङ्क गाँठो_idx, अचर std::string & name, माप_प्रकार view_index) अचर = 0;

    आभास std::string get_view_आगत_src_name(पूर्णाङ्क गाँठो_idx, अचर std::string & name, माप_प्रकार view_index) अचर = 0;

    आभास element::Type get_आगत_type(पूर्णाङ्क गाँठो_idx, अचर std::string & name) अचर = 0;

    आभास माप_प्रकार get_आगत_size() अचर = 0;

    आभास माप_प्रकार get_आगत_size(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास शून्य get_आगत_गाँठो(माप_प्रकार आगत_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                माप_प्रकार & producer_output_port_index) अचर = 0;

    आभास std::vector<std::string> get_आगत_names(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास PartialShape get_output_shape(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास element::Type get_output_type(अचर पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास std::vector<माप_प्रकार> get_output_stride(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास int32_t * get_आगत_op_params(पूर्णाङ्क गाँठो_idx, अचर std::string & name) अचर = 0;

    आभास int32_t * get_output_op_params(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास माप_प्रकार get_output_op_offset(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास std::vector<std::string> get_output_names(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास std::string get_inplace_op_src(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास bool is_view_like_alias_of(पूर्णाङ्क गाँठो_idx, अचर std::string & view_src_name) अचर = 0;

    आभास अचर std::string & get_op_type() अचर = 0;

    आभास अचर std::string & get_op_type(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास अचर std::string & get_op_name() अचर = 0;

    आभास अचर std::string & get_op_name(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास शून्य visit_subgraph(std::function<शून्य(std::shared_ptr<GgmlDecoder>, पूर्णाङ्क गाँठो_idx)> गाँठो_visitor) अचर = 0;

    आभास पूर्णाङ्क get_op_case(पूर्णाङ्क गाँठो_idx) अचर = 0;

    आभास अचर std::map<std::string, Model_आगत_Info> & get_model_inputs() अचर = 0;
    आभास अचर std::map<std::string, Model_Extra_आगत_Info> & get_model_extra_inputs() अचर = 0;
    आभास अचर std::map<std::string, std::shared_ptr<ov::गाँठो_2>> & get_model_weights() अचर = 0;
    आभास std::set<std::string> get_model_output_names() अचर = 0;

    आभास int32_t * get_rope_params() अचर = 0;

    आभास bool has_mixed_rope_params() अचर = 0;

    आभास पूर्णाङ्क get_ssm_state_size() अचर = 0;

    आभास std::map<std::string, std::string> get_kv_param_res_names() अचर = 0;

    आभास bool is_static() अचर = 0;

    आभास bool is_stateful() अचर = 0;

    आभास bool is_splited_model() अचर = 0;

    आभास पूर्णाङ्क is_swa_layer(पूर्णाङ्क layer) अचर = 0;

    आभास int32_t get_op_dynamic_dim(पूर्णाङ्क गाँठो_idx) अचर = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
