#لارښوونه once

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

بنية Model_ننوت_Info {
    element::Type type;
    PartialShape shape;
};

بنية Model_Extra_ننوت_Info {
    element::Type type;
    Shape shape;
    int64_t ارزښت;
    bool is_parameter;
};

class GgmlDecoder : عام Decoder_بنسټ {
عام:
    مجازي ov::Any get_attribute(ثابت std::string & name) ثابت = 0;

    مجازي PartialShape get_ننوت_shape(صحيح غوټه_idx, ثابت std::string & name) ثابت = 0;

    مجازي std::ویکټور<size_t> get_ننوت_stride(صحيح غوټه_idx, ثابت std::string & name) ثابت = 0;

    مجازي size_t get_view_ننوت_اندازه(صحيح غوټه_idx, ثابت std::string & name) ثابت = 0;

    مجازي size_t get_view_ننوت_دننه_واټن(صحيح غوټه_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازي size_t get_view_ننوت_src_دننه_واټن(صحيح غوټه_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازي std::ویکټور<size_t> get_view_ننوت_stride(صحيح غوټه_idx,
                                                      ثابت std::string & name,
                                                      size_t view_شاخص) ثابت = 0;

    مجازي std::ویکټور<size_t> get_view_ننوت_src_stride(صحيح غوټه_idx,
                                                          ثابت std::string & name,
                                                          size_t view_شاخص) ثابت = 0;

    مجازي Shape get_view_ننوت_ggml_shape(صحيح غوټه_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازي Shape get_view_ننوت_src_ggml_shape(صحيح غوټه_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازي PartialShape get_view_ننوت_ov_shape(صحيح غوټه_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازي PartialShape get_view_ننوت_src_ov_shape(صحيح غوټه_idx,
                                                     ثابت std::string & name,
                                                     size_t view_شاخص) ثابت = 0;

    مجازي std::string get_view_ننوت_name(صحيح غوټه_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازي std::string get_view_ننوت_src_name(صحيح غوټه_idx, ثابت std::string & name, size_t view_شاخص) ثابت = 0;

    مجازي element::Type get_ننوت_type(صحيح غوټه_idx, ثابت std::string & name) ثابت = 0;

    مجازي size_t get_ننوت_اندازه() ثابت = 0;

    مجازي size_t get_ننوت_اندازه(صحيح غوټه_idx) ثابت = 0;

    مجازي فراغ get_ننوت_غوټه(size_t ننوت_پورټ_idx,
                                std::string & تولیدوونکی_name,
                                std::string & تولیدوونکی_output_پورټ_name,
                                size_t & تولیدوونکی_output_پورټ_شاخص) ثابت = 0;

    مجازي std::ویکټور<std::string> get_ننوت_نومونه(صحيح غوټه_idx) ثابت = 0;

    مجازي PartialShape get_output_shape(صحيح غوټه_idx) ثابت = 0;

    مجازي element::Type get_output_type(ثابت صحيح غوټه_idx) ثابت = 0;

    مجازي std::ویکټور<size_t> get_output_stride(صحيح غوټه_idx) ثابت = 0;

    مجازي int32_t * get_ننوت_op_params(صحيح غوټه_idx, ثابت std::string & name) ثابت = 0;

    مجازي int32_t * get_output_op_params(صحيح غوټه_idx) ثابت = 0;

    مجازي size_t get_output_op_دننه_واټن(صحيح غوټه_idx) ثابت = 0;

    مجازي std::ویکټور<std::string> get_output_نومونه(صحيح غوټه_idx) ثابت = 0;

    مجازي std::string get_inplace_op_src(صحيح غوټه_idx) ثابت = 0;

    مجازي bool is_view_like_alias_د(صحيح غوټه_idx, ثابت std::string & view_src_name) ثابت = 0;

    مجازي ثابت std::string & get_op_type() ثابت = 0;

    مجازي ثابت std::string & get_op_type(صحيح غوټه_idx) ثابت = 0;

    مجازي ثابت std::string & get_op_name() ثابت = 0;

    مجازي ثابت std::string & get_op_name(صحيح غوټه_idx) ثابت = 0;

    مجازي فراغ لیدنه_subgraph(std::function<فراغ(std::shared_ptr<GgmlDecoder>, صحيح غوټه_idx)> غوټه_visitor) ثابت = 0;

    مجازي صحيح get_op_case(صحيح غوټه_idx) ثابت = 0;

    مجازي ثابت std::map<std::string, Model_ننوت_Info> & get_model_inputs() ثابت = 0;
    مجازي ثابت std::map<std::string, Model_Extra_ننوت_Info> & get_model_extra_inputs() ثابت = 0;
    مجازي ثابت std::map<std::string, std::shared_ptr<ov::غوټه_2>> & get_model_weights() ثابت = 0;
    مجازي std::سېټ<std::string> get_model_output_نومونه() ثابت = 0;

    مجازي int32_t * get_rope_params() ثابت = 0;

    مجازي bool has_mixed_rope_params() ثابت = 0;

    مجازي صحيح get_ssm_state_اندازه() ثابت = 0;

    مجازي std::map<std::string, std::string> get_kv_param_res_نومونه() ثابت = 0;

    مجازي bool is_static() ثابت = 0;

    مجازي bool is_stateful() ثابت = 0;

    مجازي bool is_splited_model() ثابت = 0;

    مجازي صحيح is_swa_layer(صحيح layer) ثابت = 0;

    مجازي int32_t get_op_dynamic_dim(صحيح غوټه_idx) ثابت = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
