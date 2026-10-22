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
    आभासी ov::Any get_attribute(स्थिरांक std::string & name) स्थिरांक = 0;

    आभासी PartialShape get_आदान_shape(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक = 0;

    आभासी std::vector<माप_प्रकार> get_आदान_stride(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक = 0;

    आभासी माप_प्रकार get_view_आदान_size(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक = 0;

    आभासी माप_प्रकार get_view_आदान_offset(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक = 0;

    आभासी माप_प्रकार get_view_आदान_src_offset(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक = 0;

    आभासी std::vector<माप_प्रकार> get_view_आदान_stride(पूर्णांक गाठ_idx,
                                                      स्थिरांक std::string & name,
                                                      माप_प्रकार view_index) स्थिरांक = 0;

    आभासी std::vector<माप_प्रकार> get_view_आदान_src_stride(पूर्णांक गाठ_idx,
                                                          स्थिरांक std::string & name,
                                                          माप_प्रकार view_index) स्थिरांक = 0;

    आभासी Shape get_view_आदान_ggml_shape(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक = 0;

    आभासी Shape get_view_आदान_src_ggml_shape(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक = 0;

    आभासी PartialShape get_view_आदान_ov_shape(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक = 0;

    आभासी PartialShape get_view_आदान_src_ov_shape(पूर्णांक गाठ_idx,
                                                     स्थिरांक std::string & name,
                                                     माप_प्रकार view_index) स्थिरांक = 0;

    आभासी std::string get_view_आदान_name(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक = 0;

    आभासी std::string get_view_आदान_src_name(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक = 0;

    आभासी element::Type get_आदान_type(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक = 0;

    आभासी माप_प्रकार get_आदान_size() स्थिरांक = 0;

    आभासी माप_प्रकार get_आदान_size(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी शून्य get_आदान_गाठ(माप_प्रकार आदान_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                माप_प्रकार & producer_output_port_index) स्थिरांक = 0;

    आभासी std::vector<std::string> get_आदान_names(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी PartialShape get_output_shape(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी element::Type get_output_type(स्थिरांक पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी std::vector<माप_प्रकार> get_output_stride(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी int32_t * get_आदान_op_params(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक = 0;

    आभासी int32_t * get_output_op_params(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी माप_प्रकार get_output_op_offset(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी std::vector<std::string> get_output_names(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी std::string get_inplace_op_src(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी bool is_view_like_alias_of(पूर्णांक गाठ_idx, स्थिरांक std::string & view_src_name) स्थिरांक = 0;

    आभासी स्थिरांक std::string & get_op_type() स्थिरांक = 0;

    आभासी स्थिरांक std::string & get_op_type(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी स्थिरांक std::string & get_op_name() स्थिरांक = 0;

    आभासी स्थिरांक std::string & get_op_name(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी शून्य visit_subgraph(std::function<शून्य(std::shared_ptr<GgmlDecoder>, पूर्णांक गाठ_idx)> गाठ_visitor) स्थिरांक = 0;

    आभासी पूर्णांक get_op_case(पूर्णांक गाठ_idx) स्थिरांक = 0;

    आभासी स्थिरांक std::map<std::string, Model_आदान_Info> & get_model_inputs() स्थिरांक = 0;
    आभासी स्थिरांक std::map<std::string, Model_Extra_आदान_Info> & get_model_extra_inputs() स्थिरांक = 0;
    आभासी स्थिरांक std::map<std::string, std::shared_ptr<ov::गाठ_2>> & get_model_weights() स्थिरांक = 0;
    आभासी std::set<std::string> get_model_output_names() स्थिरांक = 0;

    आभासी int32_t * get_rope_params() स्थिरांक = 0;

    आभासी bool has_mixed_rope_params() स्थिरांक = 0;

    आभासी पूर्णांक get_ssm_state_size() स्थिरांक = 0;

    आभासी std::map<std::string, std::string> get_kv_param_res_names() स्थिरांक = 0;

    आभासी bool is_static() स्थिरांक = 0;

    आभासी bool is_stateful() स्थिरांक = 0;

    आभासी bool is_splited_model() स्थिरांक = 0;

    आभासी पूर्णांक is_swa_layer(पूर्णांक layer) स्थिरांक = 0;

    आभासी int32_t get_op_dynamic_dim(पूर्णांक गाठ_idx) स्थिरांक = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
