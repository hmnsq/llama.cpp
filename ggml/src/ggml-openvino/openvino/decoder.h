#pragma once

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

بنية Model_ماسوقن_Info {
    element::Type type;
    PartialShape shape;
};

بنية Model_Extra_ماسوقن_Info {
    element::Type type;
    Shape shape;
    int64_t نيلاي;
    bool is_parameter;
};

class GgmlDecoder : عام DecoderBase {
عام:
    virtual ov::Any get_attribute(ثابت std::string & name) ثابت = 0;

    virtual PartialShape get_ماسوقن_shape(صحيح نود_idx, ثابت std::string & name) ثابت = 0;

    virtual std::vector<size_t> get_ماسوقن_stride(صحيح نود_idx, ثابت std::string & name) ثابت = 0;

    virtual size_t get_view_ماسوقن_size(صحيح نود_idx, ثابت std::string & name) ثابت = 0;

    virtual size_t get_view_ماسوقن_offset(صحيح نود_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    virtual size_t get_view_ماسوقن_src_offset(صحيح نود_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    virtual std::vector<size_t> get_view_ماسوقن_stride(صحيح نود_idx,
                                                      ثابت std::string & name,
                                                      size_t view_index) ثابت = 0;

    virtual std::vector<size_t> get_view_ماسوقن_src_stride(صحيح نود_idx,
                                                          ثابت std::string & name,
                                                          size_t view_index) ثابت = 0;

    virtual Shape get_view_ماسوقن_ggml_shape(صحيح نود_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    virtual Shape get_view_ماسوقن_src_ggml_shape(صحيح نود_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    virtual PartialShape get_view_ماسوقن_ov_shape(صحيح نود_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    virtual PartialShape get_view_ماسوقن_src_ov_shape(صحيح نود_idx,
                                                     ثابت std::string & name,
                                                     size_t view_index) ثابت = 0;

    virtual std::string get_view_ماسوقن_name(صحيح نود_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    virtual std::string get_view_ماسوقن_src_name(صحيح نود_idx, ثابت std::string & name, size_t view_index) ثابت = 0;

    virtual element::Type get_ماسوقن_type(صحيح نود_idx, ثابت std::string & name) ثابت = 0;

    virtual size_t get_ماسوقن_size() ثابت = 0;

    virtual size_t get_ماسوقن_size(صحيح نود_idx) ثابت = 0;

    virtual فراغ get_ماسوقن_نود(size_t ماسوقن_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ثابت = 0;

    virtual std::vector<std::string> get_ماسوقن_names(صحيح نود_idx) ثابت = 0;

    virtual PartialShape get_output_shape(صحيح نود_idx) ثابت = 0;

    virtual element::Type get_output_type(ثابت صحيح نود_idx) ثابت = 0;

    virtual std::vector<size_t> get_output_stride(صحيح نود_idx) ثابت = 0;

    virtual int32_t * get_ماسوقن_op_params(صحيح نود_idx, ثابت std::string & name) ثابت = 0;

    virtual int32_t * get_output_op_params(صحيح نود_idx) ثابت = 0;

    virtual size_t get_output_op_offset(صحيح نود_idx) ثابت = 0;

    virtual std::vector<std::string> get_output_names(صحيح نود_idx) ثابت = 0;

    virtual std::string get_inplace_op_src(صحيح نود_idx) ثابت = 0;

    virtual bool is_view_like_alias_of(صحيح نود_idx, ثابت std::string & view_src_name) ثابت = 0;

    virtual ثابت std::string & get_op_type() ثابت = 0;

    virtual ثابت std::string & get_op_type(صحيح نود_idx) ثابت = 0;

    virtual ثابت std::string & get_op_name() ثابت = 0;

    virtual ثابت std::string & get_op_name(صحيح نود_idx) ثابت = 0;

    virtual فراغ visit_subgraph(std::function<فراغ(std::shared_ptr<GgmlDecoder>, صحيح نود_idx)> نود_visitor) ثابت = 0;

    virtual صحيح get_op_case(صحيح نود_idx) ثابت = 0;

    virtual ثابت std::map<std::string, Model_ماسوقن_Info> & get_model_inputs() ثابت = 0;
    virtual ثابت std::map<std::string, Model_Extra_ماسوقن_Info> & get_model_extra_inputs() ثابت = 0;
    virtual ثابت std::map<std::string, std::shared_ptr<ov::نود_2>> & get_model_weights() ثابت = 0;
    virtual std::set<std::string> get_model_output_names() ثابت = 0;

    virtual int32_t * get_rope_params() ثابت = 0;

    virtual bool has_mixed_rope_params() ثابت = 0;

    virtual صحيح get_ssm_state_size() ثابت = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() ثابت = 0;

    virtual bool is_static() ثابت = 0;

    virtual bool is_stateful() ثابت = 0;

    virtual bool is_splited_model() ثابت = 0;

    virtual صحيح is_swa_layer(صحيح layer) ثابت = 0;

    virtual int32_t get_op_dynamic_dim(صحيح نود_idx) ثابت = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
