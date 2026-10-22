#pragma once

#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <map>
#അന്തര്ഭാവ <openvino/core/node.hpp>
#അന്തര്ഭാവ <openvino/core/partial_shape.hpp>
#അന്തര്ഭാവ <openvino/core/shape.hpp>
#അന്തര്ഭാവ <openvino/frontend/decoder.hpp>
#അന്തര്ഭാവ <set>
#അന്തര്ഭാവ <string>

namespace ov {
namespace frontend {
namespace ggml {

ഘടന Model_നിവേശം_Info {
    element::Type type;
    PartialShape shape;
};

ഘടന Model_Extra_നിവേശം_Info {
    element::Type type;
    Shape shape;
    int64_t മൂല്യം;
    bool is_parameter;
};

class GgmlDecoder : പൊതു DecoderBase {
പൊതു:
    മായാ ov::Any get_attribute(സ്ഥിരം std::string & name) സ്ഥിരം = 0;

    മായാ PartialShape get_നിവേശം_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം = 0;

    മായാ std::vector<size_t> get_നിവേശം_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം = 0;

    മായാ size_t get_view_നിവേശം_size(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം = 0;

    മായാ size_t get_view_നിവേശം_offset(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം = 0;

    മായാ size_t get_view_നിവേശം_src_offset(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം = 0;

    മായാ std::vector<size_t> get_view_നിവേശം_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                      സ്ഥിരം std::string & name,
                                                      size_t view_index) സ്ഥിരം = 0;

    മായാ std::vector<size_t> get_view_നിവേശം_src_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                          സ്ഥിരം std::string & name,
                                                          size_t view_index) സ്ഥിരം = 0;

    മായാ Shape get_view_നിവേശം_ggml_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം = 0;

    മായാ Shape get_view_നിവേശം_src_ggml_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം = 0;

    മായാ PartialShape get_view_നിവേശം_ov_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം = 0;

    മായാ PartialShape get_view_നിവേശം_src_ov_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                     സ്ഥിരം std::string & name,
                                                     size_t view_index) സ്ഥിരം = 0;

    മായാ std::string get_view_നിവേശം_name(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം = 0;

    മായാ std::string get_view_നിവേശം_src_name(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം = 0;

    മായാ element::Type get_നിവേശം_type(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം = 0;

    മായാ size_t get_നിവേശം_size() സ്ഥിരം = 0;

    മായാ size_t get_നിവേശം_size(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ ശൂന്യം get_നിവേശം_കണ്ണി(size_t നിവേശം_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) സ്ഥിരം = 0;

    മായാ std::vector<std::string> get_നിവേശം_names(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ PartialShape get_output_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ element::Type get_output_type(സ്ഥിരം പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ std::vector<size_t> get_output_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ int32_t * get_നിവേശം_op_params(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം = 0;

    മായാ int32_t * get_output_op_params(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ size_t get_output_op_offset(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ std::vector<std::string> get_output_names(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ std::string get_inplace_op_src(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ bool is_view_like_alias_of(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & view_src_name) സ്ഥിരം = 0;

    മായാ സ്ഥിരം std::string & get_op_type() സ്ഥിരം = 0;

    മായാ സ്ഥിരം std::string & get_op_type(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ സ്ഥിരം std::string & get_op_name() സ്ഥിരം = 0;

    മായാ സ്ഥിരം std::string & get_op_name(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ ശൂന്യം visit_subgraph(std::function<ശൂന്യം(std::shared_ptr<GgmlDecoder>, പൂർണ്ണസംഖ്യ കണ്ണി_idx)> കണ്ണി_visitor) സ്ഥിരം = 0;

    മായാ പൂർണ്ണസംഖ്യ get_op_case(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;

    മായാ സ്ഥിരം std::map<std::string, Model_നിവേശം_Info> & get_model_inputs() സ്ഥിരം = 0;
    മായാ സ്ഥിരം std::map<std::string, Model_Extra_നിവേശം_Info> & get_model_extra_inputs() സ്ഥിരം = 0;
    മായാ സ്ഥിരം std::map<std::string, std::shared_ptr<ov::കണ്ണി_2>> & get_model_weights() സ്ഥിരം = 0;
    മായാ std::set<std::string> get_model_output_names() സ്ഥിരം = 0;

    മായാ int32_t * get_rope_params() സ്ഥിരം = 0;

    മായാ bool has_mixed_rope_params() സ്ഥിരം = 0;

    മായാ പൂർണ്ണസംഖ്യ get_ssm_state_size() സ്ഥിരം = 0;

    മായാ std::map<std::string, std::string> get_kv_param_res_names() സ്ഥിരം = 0;

    മായാ bool is_static() സ്ഥിരം = 0;

    മായാ bool is_stateful() സ്ഥിരം = 0;

    മായാ bool is_splited_model() സ്ഥിരം = 0;

    മായാ പൂർണ്ണസംഖ്യ is_swa_layer(പൂർണ്ണസംഖ്യ layer) സ്ഥിരം = 0;

    മായാ int32_t get_op_dynamic_dim(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
