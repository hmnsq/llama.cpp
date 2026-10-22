#pragma once

#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <map>
#ଅନ୍ତର୍ଭାଵ <openvino/core/node.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/partial_shape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/shape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/frontend/decoder.hpp>
#ଅନ୍ତର୍ଭାଵ <set>
#ଅନ୍ତର୍ଭାଵ <string>

namespace ov {
namespace frontend {
namespace ggml {

ସଂରଚନା Model_ନିବେଶ_Info {
    element::Type type;
    PartialShape shape;
};

ସଂରଚନା Model_Extra_ନିବେଶ_Info {
    element::Type type;
    Shape shape;
    int64_t ମୂଲ୍ୟ;
    bool is_parameter;
};

class GgmlDecoder : ସର୍ବସାଧାରଣ DecoderBase {
ସର୍ବସାଧାରଣ:
    ଆଭାସୀ ov::Any get_attribute(ଅଚଳ std::string & name) ଅଚଳ = 0;

    ଆଭାସୀ PartialShape get_ନିବେଶ_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ = 0;

    ଆଭାସୀ std::vector<size_t> get_ନିବେଶ_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ = 0;

    ଆଭାସୀ size_t get_view_ନିବେଶ_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ = 0;

    ଆଭାସୀ size_t get_view_ନିବେଶ_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ = 0;

    ଆଭାସୀ size_t get_view_ନିବେଶ_src_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ = 0;

    ଆଭାସୀ std::vector<size_t> get_view_ନିବେଶ_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                      ଅଚଳ std::string & name,
                                                      size_t view_index) ଅଚଳ = 0;

    ଆଭାସୀ std::vector<size_t> get_view_ନିବେଶ_src_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                          ଅଚଳ std::string & name,
                                                          size_t view_index) ଅଚଳ = 0;

    ଆଭାସୀ Shape get_view_ନିବେଶ_ggml_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ = 0;

    ଆଭାସୀ Shape get_view_ନିବେଶ_src_ggml_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ = 0;

    ଆଭାସୀ PartialShape get_view_ନିବେଶ_ov_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ = 0;

    ଆଭାସୀ PartialShape get_view_ନିବେଶ_src_ov_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                     ଅଚଳ std::string & name,
                                                     size_t view_index) ଅଚଳ = 0;

    ଆଭାସୀ std::string get_view_ନିବେଶ_name(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ = 0;

    ଆଭାସୀ std::string get_view_ନିବେଶ_src_name(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ = 0;

    ଆଭାସୀ element::Type get_ନିବେଶ_type(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ = 0;

    ଆଭାସୀ size_t get_ନିବେଶ_size() ଅଚଳ = 0;

    ଆଭାସୀ size_t get_ନିବେଶ_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ ଶୂନ୍ୟ get_ନିବେଶ_ଗଣ୍ଠି(size_t ନିବେଶ_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ଅଚଳ = 0;

    ଆଭାସୀ std::vector<std::string> get_ନିବେଶ_names(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ PartialShape get_output_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ element::Type get_output_type(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ std::vector<size_t> get_output_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ int32_t * get_ନିବେଶ_op_params(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ = 0;

    ଆଭାସୀ int32_t * get_output_op_params(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ size_t get_output_op_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ std::vector<std::string> get_output_names(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ std::string get_inplace_op_src(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ bool is_view_like_alias_of(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & view_src_name) ଅଚଳ = 0;

    ଆଭାସୀ ଅଚଳ std::string & get_op_type() ଅଚଳ = 0;

    ଆଭାସୀ ଅଚଳ std::string & get_op_type(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ ଅଚଳ std::string & get_op_name() ଅଚଳ = 0;

    ଆଭାସୀ ଅଚଳ std::string & get_op_name(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ ଶୂନ୍ୟ visit_subgraph(std::function<ଶୂନ୍ୟ(std::shared_ptr<GgmlDecoder>, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx)> ଗଣ୍ଠି_visitor) ଅଚଳ = 0;

    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_op_case(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;

    ଆଭାସୀ ଅଚଳ std::map<std::string, Model_ନିବେଶ_Info> & get_model_inputs() ଅଚଳ = 0;
    ଆଭାସୀ ଅଚଳ std::map<std::string, Model_Extra_ନିବେଶ_Info> & get_model_extra_inputs() ଅଚଳ = 0;
    ଆଭାସୀ ଅଚଳ std::map<std::string, std::shared_ptr<ov::ଗଣ୍ଠି_2>> & get_model_weights() ଅଚଳ = 0;
    ଆଭାସୀ std::set<std::string> get_model_output_names() ଅଚଳ = 0;

    ଆଭାସୀ int32_t * get_rope_params() ଅଚଳ = 0;

    ଆଭାସୀ bool has_mixed_rope_params() ଅଚଳ = 0;

    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_ssm_state_size() ଅଚଳ = 0;

    ଆଭାସୀ std::map<std::string, std::string> get_kv_param_res_names() ଅଚଳ = 0;

    ଆଭାସୀ bool is_static() ଅଚଳ = 0;

    ଆଭାସୀ bool is_stateful() ଅଚଳ = 0;

    ଆଭାସୀ bool is_splited_model() ଅଚଳ = 0;

    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା is_swa_layer(ପୂର୍ଣ୍ଣସଂଖ୍ୟା layer) ଅଚଳ = 0;

    ଆଭାସୀ int32_t get_op_dynamic_dim(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
