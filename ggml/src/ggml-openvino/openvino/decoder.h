#ہدایت once

#ضمن <cstdint>
#ضمن <map>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/partial_shape.hpp>
#ضمن <openvino/core/shape.hpp>
#ضمن <openvino/frontend/decoder.hpp>
#ضمن <set>
#ضمن <string>

namespace ov {
namespace frontend {
namespace ggml {

بنية Model_داخلہ_Info {
    element::Type type;
    PartialShape shape;
};

بنية Model_Extra_داخلہ_Info {
    element::Type type;
    Shape shape;
    int64_t قدر;
    bool is_parameter;
};

class GgmlDecoder : عام Decoder_ادھار {
عام:
    مجازی ov::Any get_attribute(ثابت std::string & name) ثابت = 0;

    مجازی PartialShape get_داخلہ_shape(صحيح گرہ_idx, ثابت std::string & name) ثابت = 0;

    مجازی std::سمتیہ<size_t> get_داخلہ_stride(صحيح گرہ_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_view_داخلہ_ناپ(صحيح گرہ_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_view_داخلہ_وتھ(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی size_t get_view_داخلہ_src_وتھ(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی std::سمتیہ<size_t> get_view_داخلہ_stride(صحيح گرہ_idx,
                                                      ثابت std::string & name,
                                                      size_t view_اشاریہ) ثابت = 0;

    مجازی std::سمتیہ<size_t> get_view_داخلہ_src_stride(صحيح گرہ_idx,
                                                          ثابت std::string & name,
                                                          size_t view_اشاریہ) ثابت = 0;

    مجازی Shape get_view_داخلہ_ggml_shape(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی Shape get_view_داخلہ_src_ggml_shape(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی PartialShape get_view_داخلہ_ov_shape(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی PartialShape get_view_داخلہ_src_ov_shape(صحيح گرہ_idx,
                                                     ثابت std::string & name,
                                                     size_t view_اشاریہ) ثابت = 0;

    مجازی std::string get_view_داخلہ_name(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی std::string get_view_داخلہ_src_name(صحيح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی element::Type get_داخلہ_type(صحيح گرہ_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_داخلہ_ناپ() ثابت = 0;

    مجازی size_t get_داخلہ_ناپ(صحيح گرہ_idx) ثابت = 0;

    مجازی فراغ get_داخلہ_گرہ(size_t داخلہ_پورٹ_idx,
                                std::string & پیداکار_name,
                                std::string & پیداکار_output_پورٹ_name,
                                size_t & پیداکار_output_پورٹ_اشاریہ) ثابت = 0;

    مجازی std::سمتیہ<std::string> get_داخلہ_ناں(صحيح گرہ_idx) ثابت = 0;

    مجازی PartialShape get_output_shape(صحيح گرہ_idx) ثابت = 0;

    مجازی element::Type get_output_type(ثابت صحيح گرہ_idx) ثابت = 0;

    مجازی std::سمتیہ<size_t> get_output_stride(صحيح گرہ_idx) ثابت = 0;

    مجازی int32_t * get_داخلہ_op_params(صحيح گرہ_idx, ثابت std::string & name) ثابت = 0;

    مجازی int32_t * get_output_op_params(صحيح گرہ_idx) ثابت = 0;

    مجازی size_t get_output_op_وتھ(صحيح گرہ_idx) ثابت = 0;

    مجازی std::سمتیہ<std::string> get_output_ناں(صحيح گرہ_idx) ثابت = 0;

    مجازی std::string get_inplace_op_src(صحيح گرہ_idx) ثابت = 0;

    مجازی bool is_view_like_alias_دا(صحيح گرہ_idx, ثابت std::string & view_src_name) ثابت = 0;

    مجازی ثابت std::string & get_op_type() ثابت = 0;

    مجازی ثابت std::string & get_op_type(صحيح گرہ_idx) ثابت = 0;

    مجازی ثابت std::string & get_op_name() ثابت = 0;

    مجازی ثابت std::string & get_op_name(صحيح گرہ_idx) ثابت = 0;

    مجازی فراغ ویکھو_subgraph(std::function<فراغ(std::shared_ptr<GgmlDecoder>, صحيح گرہ_idx)> گرہ_visitor) ثابت = 0;

    مجازی صحيح get_op_case(صحيح گرہ_idx) ثابت = 0;

    مجازی ثابت std::map<std::string, Model_داخلہ_Info> & get_model_inputs() ثابت = 0;
    مجازی ثابت std::map<std::string, Model_Extra_داخلہ_Info> & get_model_extra_inputs() ثابت = 0;
    مجازی ثابت std::map<std::string, std::shared_ptr<ov::گرہ_2>> & get_model_weights() ثابت = 0;
    مجازی std::سیٹ<std::string> get_model_output_ناں() ثابت = 0;

    مجازی int32_t * get_rope_params() ثابت = 0;

    مجازی bool has_mixed_rope_params() ثابت = 0;

    مجازی صحيح get_ssm_state_ناپ() ثابت = 0;

    مجازی std::map<std::string, std::string> get_kv_param_res_ناں() ثابت = 0;

    مجازی bool is_static() ثابت = 0;

    مجازی bool is_stateful() ثابت = 0;

    مجازی bool is_splited_model() ثابت = 0;

    مجازی صحيح is_swa_layer(صحيح layer) ثابت = 0;

    مجازی int32_t get_op_dynamic_dim(صحيح گرہ_idx) ثابت = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
