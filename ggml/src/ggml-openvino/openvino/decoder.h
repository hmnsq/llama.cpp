#رهنمود once

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

بنية Model_درون_داد_Info {
    element::Type type;
    PartialShape shape;
};

بنية Model_Extra_درون_داد_Info {
    element::Type type;
    Shape shape;
    int64_t مقدار;
    bool is_parameter;
};

class GgmlDecoder : عام Decoder_پایه {
عام:
    مجازی ov::Any get_attribute(ثابت std::string & name) ثابت = 0;

    مجازی PartialShape get_درون_داد_shape(صحيح گره_idx, ثابت std::string & name) ثابت = 0;

    مجازی std::وکتور<size_t> get_درون_داد_stride(صحيح گره_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_view_درون_داد_اندازه(صحيح گره_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_view_درون_داد_جابه_جایی(صحيح گره_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازی size_t get_view_درون_داد_src_جابه_جایی(صحيح گره_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازی std::وکتور<size_t> get_view_درون_داد_stride(صحيح گره_idx,
                                                      ثابت std::string & name,
                                                      size_t view_شاخص) ثابت = 0;

    مجازی std::وکتور<size_t> get_view_درون_داد_src_stride(صحيح گره_idx,
                                                          ثابت std::string & name,
                                                          size_t view_شاخص) ثابت = 0;

    مجازی Shape get_view_درون_داد_ggml_shape(صحيح گره_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازی Shape get_view_درون_داد_src_ggml_shape(صحيح گره_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازی PartialShape get_view_درون_داد_ov_shape(صحيح گره_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازی PartialShape get_view_درون_داد_src_ov_shape(صحيح گره_idx,
                                                     ثابت std::string & name,
                                                     size_t view_شاخص) ثابت = 0;

    مجازی std::string get_view_درون_داد_name(صحيح گره_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازی std::string get_view_درون_داد_src_name(صحيح گره_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازی element::Type get_درون_داد_type(صحيح گره_idx, ثابت std::string & name) ثابت = 0;

    مجازی size_t get_درون_داد_اندازه() ثابت = 0;

    مجازی size_t get_درون_داد_اندازه(صحيح گره_idx) ثابت = 0;

    مجازی فراغ get_درون_داد_گره(size_t درون_داد_پورت_idx,
                                std::string & تولیدکننده_name,
                                std::string & تولیدکننده_output_پورت_name,
                                size_t & تولیدکننده_output_پورت_شاخص) ثابت = 0;

    مجازی std::وکتور<std::string> get_درون_داد_نامها(صحيح گره_idx) ثابت = 0;

    مجازی PartialShape get_output_shape(صحيح گره_idx) ثابت = 0;

    مجازی element::Type get_output_type(ثابت صحيح گره_idx) ثابت = 0;

    مجازی std::وکتور<size_t> get_output_stride(صحيح گره_idx) ثابت = 0;

    مجازی int32_t * get_درون_داد_op_params(صحيح گره_idx, ثابت std::string & name) ثابت = 0;

    مجازی int32_t * get_output_op_params(صحيح گره_idx) ثابت = 0;

    مجازی size_t get_output_op_جابه_جایی(صحيح گره_idx) ثابت = 0;

    مجازی std::وکتور<std::string> get_output_نامها(صحيح گره_idx) ثابت = 0;

    مجازی std::string get_inplace_op_src(صحيح گره_idx) ثابت = 0;

    مجازی bool is_view_like_alias_از(صحيح گره_idx, ثابت std::string & view_src_name) ثابت = 0;

    مجازی ثابت std::string & get_op_type() ثابت = 0;

    مجازی ثابت std::string & get_op_type(صحيح گره_idx) ثابت = 0;

    مجازی ثابت std::string & get_op_name() ثابت = 0;

    مجازی ثابت std::string & get_op_name(صحيح گره_idx) ثابت = 0;

    مجازی فراغ بازدید_subgraph(std::function<فراغ(std::shared_ptr<GgmlDecoder>, صحيح گره_idx)> گره_visitor) ثابت = 0;

    مجازی صحيح get_op_case(صحيح گره_idx) ثابت = 0;

    مجازی ثابت std::map<std::string, Model_درون_داد_Info> & get_model_inputs() ثابت = 0;
    مجازی ثابت std::map<std::string, Model_Extra_درون_داد_Info> & get_model_extra_inputs() ثابت = 0;
    مجازی ثابت std::map<std::string, std::shared_ptr<ov::گره_2>> & get_model_weights() ثابت = 0;
    مجازی std::مجموعه<std::string> get_model_output_نامها() ثابت = 0;

    مجازی int32_t * get_rope_params() ثابت = 0;

    مجازی bool has_mixed_rope_params() ثابت = 0;

    مجازی صحيح get_ssm_state_اندازه() ثابت = 0;

    مجازی std::map<std::string, std::string> get_kv_param_res_نامها() ثابت = 0;

    مجازی bool is_static() ثابت = 0;

    مجازی bool is_stateful() ثابت = 0;

    مجازی bool is_splited_model() ثابت = 0;

    مجازی صحيح is_swa_layer(صحيح layer) ثابت = 0;

    مجازی int32_t get_op_dynamic_dim(صحيح گره_idx) ثابت = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
