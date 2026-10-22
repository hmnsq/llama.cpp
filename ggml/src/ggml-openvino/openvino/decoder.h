#निर्देशः once

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

संरचना Model_निवेशः_Info {
    element::Type type;
    PartialShape shape;
};

संरचना Model_Extra_निवेशः_Info {
    element::Type type;
    Shape shape;
    int64_t मानम्;
    bool is_parameter;
};

class GgmlDecoder : सार्वजनिक Decoder_आधारः {
सार्वजनिक:
    आभास ov::Any get_attribute(ध्रुव std::string & name) ध्रुव = 0;

    आभास PartialShape get_निवेशः_shape(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name) ध्रुव = 0;

    आभास std::सदिशम्<size_t> get_निवेशः_stride(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name) ध्रुव = 0;

    आभास size_t get_view_निवेशः_आकार(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name) ध्रुव = 0;

    आभास size_t get_view_निवेशः_विस्थापनम्(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_सूचकांक) ध्रुव = 0;

    आभास size_t get_view_निवेशः_src_विस्थापनम्(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_सूचकांक) ध्रुव = 0;

    आभास std::सदिशम्<size_t> get_view_निवेशः_stride(पूर्णाङ्क ग्रन्थिः_idx,
                                                      ध्रुव std::string & name,
                                                      size_t view_सूचकांक) ध्रुव = 0;

    आभास std::सदिशम्<size_t> get_view_निवेशः_src_stride(पूर्णाङ्क ग्रन्थिः_idx,
                                                          ध्रुव std::string & name,
                                                          size_t view_सूचकांक) ध्रुव = 0;

    आभास Shape get_view_निवेशः_ggml_shape(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_सूचकांक) ध्रुव = 0;

    आभास Shape get_view_निवेशः_src_ggml_shape(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_सूचकांक) ध्रुव = 0;

    आभास PartialShape get_view_निवेशः_ov_shape(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_सूचकांक) ध्रुव = 0;

    आभास PartialShape get_view_निवेशः_src_ov_shape(पूर्णाङ्क ग्रन्थिः_idx,
                                                     ध्रुव std::string & name,
                                                     size_t view_सूचकांक) ध्रुव = 0;

    आभास std::string get_view_निवेशः_name(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_सूचकांक) ध्रुव = 0;

    आभास std::string get_view_निवेशः_src_name(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name, size_t view_सूचकांक) ध्रुव = 0;

    आभास element::Type get_निवेशः_type(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name) ध्रुव = 0;

    आभास size_t get_निवेशः_आकार() ध्रुव = 0;

    आभास size_t get_निवेशः_आकार(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास शून्य get_निवेशः_ग्रन्थिः(size_t निवेशः_पत्तनम्_idx,
                                std::string & उत्पादकः_name,
                                std::string & उत्पादकः_output_पत्तनम्_name,
                                size_t & उत्पादकः_output_पत्तनम्_सूचकांक) ध्रुव = 0;

    आभास std::सदिशम्<std::string> get_निवेशः_नामानि(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास PartialShape get_output_shape(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास element::Type get_output_type(ध्रुव पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास std::सदिशम्<size_t> get_output_stride(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास int32_t * get_निवेशः_op_params(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & name) ध्रुव = 0;

    आभास int32_t * get_output_op_params(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास size_t get_output_op_विस्थापनम्(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास std::सदिशम्<std::string> get_output_नामानि(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास std::string get_inplace_op_src(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास bool is_view_like_alias_सम्बन्धि(पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::string & view_src_name) ध्रुव = 0;

    आभास ध्रुव std::string & get_op_type() ध्रुव = 0;

    आभास ध्रुव std::string & get_op_type(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास ध्रुव std::string & get_op_name() ध्रुव = 0;

    आभास ध्रुव std::string & get_op_name(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास शून्य दर्शय_subgraph(std::function<शून्य(std::shared_ptr<GgmlDecoder>, पूर्णाङ्क ग्रन्थिः_idx)> ग्रन्थिः_visitor) ध्रुव = 0;

    आभास पूर्णाङ्क get_op_case(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;

    आभास ध्रुव std::map<std::string, Model_निवेशः_Info> & get_model_inputs() ध्रुव = 0;
    आभास ध्रुव std::map<std::string, Model_Extra_निवेशः_Info> & get_model_extra_inputs() ध्रुव = 0;
    आभास ध्रुव std::map<std::string, std::shared_ptr<ov::ग्रन्थिः_2>> & get_model_weights() ध्रुव = 0;
    आभास std::समुच्चयः<std::string> get_model_output_नामानि() ध्रुव = 0;

    आभास int32_t * get_rope_params() ध्रुव = 0;

    आभास bool has_mixed_rope_params() ध्रुव = 0;

    आभास पूर्णाङ्क get_ssm_state_आकार() ध्रुव = 0;

    आभास std::map<std::string, std::string> get_kv_param_res_नामानि() ध्रुव = 0;

    आभास bool is_static() ध्रुव = 0;

    आभास bool is_stateful() ध्रुव = 0;

    आभास bool is_splited_model() ध्रुव = 0;

    आभास पूर्णाङ्क is_swa_layer(पूर्णाङ्क layer) ध्रुव = 0;

    आभास int32_t get_op_dynamic_dim(पूर्णाङ्क ग्रन्थिः_idx) ध्रुव = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
