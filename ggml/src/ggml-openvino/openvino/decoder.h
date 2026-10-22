#هدايت once

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

بنية Model_داخلا_Info {
    element::Type type;
    PartialShape shape;
};

بنية Model_Extra_داخلا_Info {
    element::Type type;
    Shape shape;
    int64_t قدر;
    bool is_parameter;
};

class GgmlDecoder : عام Decoder_بنياد {
عام:
    مجازي ov::Any get_attribute(ثابت std::string & name) ثابت = 0;

    مجازي PartialShape get_داخلا_shape(صحيح ڳنڍ_idx, ثابت std::string & name) ثابت = 0;

    مجازي std::ويڪٽر<size_t> get_داخلا_stride(صحيح ڳنڍ_idx, ثابت std::string & name) ثابت = 0;

    مجازي size_t get_view_داخلا_سائيز(صحيح ڳنڍ_idx, ثابت std::string & name) ثابت = 0;

    مجازي size_t get_view_داخلا_وٿي(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت = 0;

    مجازي size_t get_view_داخلا_src_وٿي(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت = 0;

    مجازي std::ويڪٽر<size_t> get_view_داخلا_stride(صحيح ڳنڍ_idx,
                                                      ثابت std::string & name,
                                                      size_t view_اشاريو) ثابت = 0;

    مجازي std::ويڪٽر<size_t> get_view_داخلا_src_stride(صحيح ڳنڍ_idx,
                                                          ثابت std::string & name,
                                                          size_t view_اشاريو) ثابت = 0;

    مجازي Shape get_view_داخلا_ggml_shape(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت = 0;

    مجازي Shape get_view_داخلا_src_ggml_shape(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت = 0;

    مجازي PartialShape get_view_داخلا_ov_shape(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت = 0;

    مجازي PartialShape get_view_داخلا_src_ov_shape(صحيح ڳنڍ_idx,
                                                     ثابت std::string & name,
                                                     size_t view_اشاريو) ثابت = 0;

    مجازي std::string get_view_داخلا_name(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت = 0;

    مجازي std::string get_view_داخلا_src_name(صحيح ڳنڍ_idx, ثابت std::string & name, size_t view_اشاريو) ثابت = 0;

    مجازي element::Type get_داخلا_type(صحيح ڳنڍ_idx, ثابت std::string & name) ثابت = 0;

    مجازي size_t get_داخلا_سائيز() ثابت = 0;

    مجازي size_t get_داخلا_سائيز(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي فراغ get_داخلا_ڳنڍ(size_t داخلا_پورٽ_idx,
                                std::string & پيداوار_ڪندڙ_name,
                                std::string & پيداوار_ڪندڙ_output_پورٽ_name,
                                size_t & پيداوار_ڪندڙ_output_پورٽ_اشاريو) ثابت = 0;

    مجازي std::ويڪٽر<std::string> get_داخلا_نالا(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي PartialShape get_output_shape(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي element::Type get_output_type(ثابت صحيح ڳنڍ_idx) ثابت = 0;

    مجازي std::ويڪٽر<size_t> get_output_stride(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي int32_t * get_داخلا_op_params(صحيح ڳنڍ_idx, ثابت std::string & name) ثابت = 0;

    مجازي int32_t * get_output_op_params(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي size_t get_output_op_وٿي(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي std::ويڪٽر<std::string> get_output_نالا(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي std::string get_inplace_op_src(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي bool is_view_like_alias_جو(صحيح ڳنڍ_idx, ثابت std::string & view_src_name) ثابت = 0;

    مجازي ثابت std::string & get_op_type() ثابت = 0;

    مجازي ثابت std::string & get_op_type(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي ثابت std::string & get_op_name() ثابت = 0;

    مجازي ثابت std::string & get_op_name(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي فراغ گهمو_subgraph(std::function<فراغ(std::shared_ptr<GgmlDecoder>, صحيح ڳنڍ_idx)> ڳنڍ_visitor) ثابت = 0;

    مجازي صحيح get_op_case(صحيح ڳنڍ_idx) ثابت = 0;

    مجازي ثابت std::map<std::string, Model_داخلا_Info> & get_model_inputs() ثابت = 0;
    مجازي ثابت std::map<std::string, Model_Extra_داخلا_Info> & get_model_extra_inputs() ثابت = 0;
    مجازي ثابت std::map<std::string, std::shared_ptr<ov::ڳنڍ_2>> & get_model_weights() ثابت = 0;
    مجازي std::سيٽ<std::string> get_model_output_نالا() ثابت = 0;

    مجازي int32_t * get_rope_params() ثابت = 0;

    مجازي bool has_mixed_rope_params() ثابت = 0;

    مجازي صحيح get_ssm_state_سائيز() ثابت = 0;

    مجازي std::map<std::string, std::string> get_kv_param_res_نالا() ثابت = 0;

    مجازي bool is_static() ثابت = 0;

    مجازي bool is_stateful() ثابت = 0;

    مجازي bool is_splited_model() ثابت = 0;

    مجازي صحيح is_swa_layer(صحيح layer) ثابت = 0;

    مجازي int32_t get_op_dynamic_dim(صحيح ڳنڍ_idx) ثابت = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
