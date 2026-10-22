#ہدایت once

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

ساخت Model_داخلہ_Info {
    element::Type type;
    PartialShape shape;
};

ساخت Model_Extra_داخلہ_Info {
    element::Type type;
    Shape shape;
    int64_t قدر;
    bool is_parameter;
};

class GgmlDecoder : عوامی Decoder_بنیاد {
عوامی:
    مجازی ov::Any get_attribute(ثابت std::string & name) ثابت = 0;

    مجازی PartialShape get_داخلہ_shape(صحیح گرہ_idx, ثابت std::string & name) ثابت = 0;

    مجازی std::سمتیہ<size_t> get_داخلہ_stride(صحیح گرہ_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_view_داخلہ_سائز(صحیح گرہ_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_view_داخلہ_انحراف(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی size_t get_view_داخلہ_src_انحراف(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی std::سمتیہ<size_t> get_view_داخلہ_stride(صحیح گرہ_idx,
                                                      ثابت std::string & name,
                                                      size_t view_اشاریہ) ثابت = 0;

    مجازی std::سمتیہ<size_t> get_view_داخلہ_src_stride(صحیح گرہ_idx,
                                                          ثابت std::string & name,
                                                          size_t view_اشاریہ) ثابت = 0;

    مجازی Shape get_view_داخلہ_ggml_shape(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی Shape get_view_داخلہ_src_ggml_shape(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی PartialShape get_view_داخلہ_ov_shape(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی PartialShape get_view_داخلہ_src_ov_shape(صحیح گرہ_idx,
                                                     ثابت std::string & name,
                                                     size_t view_اشاریہ) ثابت = 0;

    مجازی std::string get_view_داخلہ_name(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی std::string get_view_داخلہ_src_name(صحیح گرہ_idx, ثابت std::string & name, size_t view_اشاریہ) ثابت = 0;

    مجازی element::Type get_داخلہ_type(صحیح گرہ_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_داخلہ_سائز() ثابت = 0;

    مجازی size_t get_داخلہ_سائز(صحیح گرہ_idx) ثابت = 0;

    مجازی خالی get_داخلہ_گرہ(size_t داخلہ_پورٹ_idx,
                                std::string & پیداکار_name,
                                std::string & پیداکار_output_پورٹ_name,
                                size_t & پیداکار_output_پورٹ_اشاریہ) ثابت = 0;

    مجازی std::سمتیہ<std::string> get_داخلہ_نام(صحیح گرہ_idx) ثابت = 0;

    مجازی PartialShape get_output_shape(صحیح گرہ_idx) ثابت = 0;

    مجازی element::Type get_output_type(ثابت صحیح گرہ_idx) ثابت = 0;

    مجازی std::سمتیہ<size_t> get_output_stride(صحیح گرہ_idx) ثابت = 0;

    مجازی int32_t * get_داخلہ_op_params(صحیح گرہ_idx, ثابت std::string & name) ثابت = 0;

    مجازی int32_t * get_output_op_params(صحیح گرہ_idx) ثابت = 0;

    مجازی size_t get_output_op_انحراف(صحیح گرہ_idx) ثابت = 0;

    مجازی std::سمتیہ<std::string> get_output_نام(صحیح گرہ_idx) ثابت = 0;

    مجازی std::string get_inplace_op_src(صحیح گرہ_idx) ثابت = 0;

    مجازی bool is_view_like_alias_کا(صحیح گرہ_idx, ثابت std::string & view_src_name) ثابت = 0;

    مجازی ثابت std::string & get_op_type() ثابت = 0;

    مجازی ثابت std::string & get_op_type(صحیح گرہ_idx) ثابت = 0;

    مجازی ثابت std::string & get_op_name() ثابت = 0;

    مجازی ثابت std::string & get_op_name(صحیح گرہ_idx) ثابت = 0;

    مجازی خالی دورہ_subgraph(std::function<خالی(std::shared_ptr<GgmlDecoder>, صحیح گرہ_idx)> گرہ_visitor) ثابت = 0;

    مجازی صحیح get_op_case(صحیح گرہ_idx) ثابت = 0;

    مجازی ثابت std::map<std::string, Model_داخلہ_Info> & get_model_inputs() ثابت = 0;
    مجازی ثابت std::map<std::string, Model_Extra_داخلہ_Info> & get_model_extra_inputs() ثابت = 0;
    مجازی ثابت std::map<std::string, std::shared_ptr<ov::گرہ_2>> & get_model_weights() ثابت = 0;
    مجازی std::سیٹ<std::string> get_model_output_نام() ثابت = 0;

    مجازی int32_t * get_rope_params() ثابت = 0;

    مجازی bool has_mixed_rope_params() ثابت = 0;

    مجازی صحیح get_ssm_state_سائز() ثابت = 0;

    مجازی std::map<std::string, std::string> get_kv_param_res_نام() ثابت = 0;

    مجازی bool is_static() ثابت = 0;

    مجازی bool is_stateful() ثابت = 0;

    مجازی bool is_splited_model() ثابت = 0;

    مجازی صحیح is_swa_layer(صحیح layer) ثابت = 0;

    مجازی int32_t get_op_dynamic_dim(صحیح گرہ_idx) ثابت = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
