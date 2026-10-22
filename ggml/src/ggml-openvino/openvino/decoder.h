#pragma once

#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <map>
#અન્તર્ભાવ <openvino/core/node.hpp>
#અન્તર્ભાવ <openvino/core/partial_shape.hpp>
#અન્તર્ભાવ <openvino/core/shape.hpp>
#અન્તર્ભાવ <openvino/frontend/decoder.hpp>
#અન્તર્ભાવ <set>
#અન્તર્ભાવ <string>

namespace ov {
namespace frontend {
namespace ggml {

સંરચના Model_આગત_Info {
    element::Type type;
    PartialShape shape;
};

સંરચના Model_Extra_આગત_Info {
    element::Type type;
    Shape shape;
    int64_t કિંમત;
    bool is_parameter;
};

class GgmlDecoder : જાહેર DecoderBase {
જાહેર:
    આભાસી ov::Any get_attribute(અચલ std::string & name) અચલ = 0;

    આભાસી PartialShape get_આગત_shape(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ = 0;

    આભાસી std::vector<size_t> get_આગત_stride(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ = 0;

    આભાસી size_t get_view_આગત_size(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ = 0;

    આભાસી size_t get_view_આગત_offset(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ = 0;

    આભાસી size_t get_view_આગત_src_offset(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ = 0;

    આભાસી std::vector<size_t> get_view_આગત_stride(પૂર્ણાંક ગાંઠ_idx,
                                                      અચલ std::string & name,
                                                      size_t view_index) અચલ = 0;

    આભાસી std::vector<size_t> get_view_આગત_src_stride(પૂર્ણાંક ગાંઠ_idx,
                                                          અચલ std::string & name,
                                                          size_t view_index) અચલ = 0;

    આભાસી Shape get_view_આગત_ggml_shape(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ = 0;

    આભાસી Shape get_view_આગત_src_ggml_shape(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ = 0;

    આભાસી PartialShape get_view_આગત_ov_shape(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ = 0;

    આભાસી PartialShape get_view_આગત_src_ov_shape(પૂર્ણાંક ગાંઠ_idx,
                                                     અચલ std::string & name,
                                                     size_t view_index) અચલ = 0;

    આભાસી std::string get_view_આગત_name(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ = 0;

    આભાસી std::string get_view_આગત_src_name(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ = 0;

    આભાસી element::Type get_આગત_type(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ = 0;

    આભાસી size_t get_આગત_size() અચલ = 0;

    આભાસી size_t get_આગત_size(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી શૂન્ય get_આગત_ગાંઠ(size_t આગત_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) અચલ = 0;

    આભાસી std::vector<std::string> get_આગત_names(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી PartialShape get_output_shape(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી element::Type get_output_type(અચલ પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી std::vector<size_t> get_output_stride(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી int32_t * get_આગત_op_params(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ = 0;

    આભાસી int32_t * get_output_op_params(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી size_t get_output_op_offset(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી std::vector<std::string> get_output_names(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી std::string get_inplace_op_src(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી bool is_view_like_alias_of(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & view_src_name) અચલ = 0;

    આભાસી અચલ std::string & get_op_type() અચલ = 0;

    આભાસી અચલ std::string & get_op_type(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી અચલ std::string & get_op_name() અચલ = 0;

    આભાસી અચલ std::string & get_op_name(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી શૂન્ય visit_subgraph(std::function<શૂન્ય(std::shared_ptr<GgmlDecoder>, પૂર્ણાંક ગાંઠ_idx)> ગાંઠ_visitor) અચલ = 0;

    આભાસી પૂર્ણાંક get_op_case(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;

    આભાસી અચલ std::map<std::string, Model_આગત_Info> & get_model_inputs() અચલ = 0;
    આભાસી અચલ std::map<std::string, Model_Extra_આગત_Info> & get_model_extra_inputs() અચલ = 0;
    આભાસી અચલ std::map<std::string, std::shared_ptr<ov::ગાંઠ_2>> & get_model_weights() અચલ = 0;
    આભાસી std::set<std::string> get_model_output_names() અચલ = 0;

    આભાસી int32_t * get_rope_params() અચલ = 0;

    આભાસી bool has_mixed_rope_params() અચલ = 0;

    આભાસી પૂર્ણાંક get_ssm_state_size() અચલ = 0;

    આભાસી std::map<std::string, std::string> get_kv_param_res_names() અચલ = 0;

    આભાસી bool is_static() અચલ = 0;

    આભાસી bool is_stateful() અચલ = 0;

    આભાસી bool is_splited_model() અચલ = 0;

    આભાસી પૂર્ણાંક is_swa_layer(પૂર્ણાંક layer) અચલ = 0;

    આભાસી int32_t get_op_dynamic_dim(પૂર્ણાંક ગાંઠ_idx) અચલ = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
