#आशय once

#समावेश <cstdint>
#समावेश <map>
#समावेश <openvino/core/node.hpp>
#समावेश <openvino/core/partial_shape.hpp>
#समावेश <openvino/core/shape.hpp>
#समावेश <openvino/frontend/decoder.hpp>
#समावेश <set>
#समावेश <string>

namespace ov {
namespace frontend {
namespace ggml {

काष्ठा Model_निवेश_Info {
    element::Type type;
    PartialShape shape;
};

काष्ठा Model_Extra_निवेश_Info {
    element::Type type;
    Shape shape;
    int64_t मूल्य;
    bool is_parameter;
};

class GgmlDecoder : खुला DecoderBase {
खुला:
    भव ov::Any get_attribute(स्थिर std::string & name) स्थिर = 0;

    भव PartialShape get_निवेश_shape(पूर्णांक गाँठ_idx, स्थिर std::string & name) स्थिर = 0;

    भव std::vector<माप_प्रकार> get_निवेश_stride(पूर्णांक गाँठ_idx, स्थिर std::string & name) स्थिर = 0;

    भव माप_प्रकार get_view_निवेश_आकार(पूर्णांक गाँठ_idx, स्थिर std::string & name) स्थिर = 0;

    भव माप_प्रकार get_view_निवेश_offset(पूर्णांक गाँठ_idx, स्थिर std::string & name, माप_प्रकार view_सूचकांक) स्थिर = 0;

    भव माप_प्रकार get_view_निवेश_src_offset(पूर्णांक गाँठ_idx, स्थिर std::string & name, माप_प्रकार view_सूचकांक) स्थिर = 0;

    भव std::vector<माप_प्रकार> get_view_निवेश_stride(पूर्णांक गाँठ_idx,
                                                      स्थिर std::string & name,
                                                      माप_प्रकार view_सूचकांक) स्थिर = 0;

    भव std::vector<माप_प्रकार> get_view_निवेश_src_stride(पूर्णांक गाँठ_idx,
                                                          स्थिर std::string & name,
                                                          माप_प्रकार view_सूचकांक) स्थिर = 0;

    भव Shape get_view_निवेश_ggml_shape(पूर्णांक गाँठ_idx, स्थिर std::string & name, माप_प्रकार view_सूचकांक) स्थिर = 0;

    भव Shape get_view_निवेश_src_ggml_shape(पूर्णांक गाँठ_idx, स्थिर std::string & name, माप_प्रकार view_सूचकांक) स्थिर = 0;

    भव PartialShape get_view_निवेश_ov_shape(पूर्णांक गाँठ_idx, स्थिर std::string & name, माप_प्रकार view_सूचकांक) स्थिर = 0;

    भव PartialShape get_view_निवेश_src_ov_shape(पूर्णांक गाँठ_idx,
                                                     स्थिर std::string & name,
                                                     माप_प्रकार view_सूचकांक) स्थिर = 0;

    भव std::string get_view_निवेश_name(पूर्णांक गाँठ_idx, स्थिर std::string & name, माप_प्रकार view_सूचकांक) स्थिर = 0;

    भव std::string get_view_निवेश_src_name(पूर्णांक गाँठ_idx, स्थिर std::string & name, माप_प्रकार view_सूचकांक) स्थिर = 0;

    भव element::Type get_निवेश_type(पूर्णांक गाँठ_idx, स्थिर std::string & name) स्थिर = 0;

    भव माप_प्रकार get_निवेश_आकार() स्थिर = 0;

    भव माप_प्रकार get_निवेश_आकार(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव व्योम get_निवेश_गाँठ(माप_प्रकार निवेश_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                माप_प्रकार & producer_output_port_सूचकांक) स्थिर = 0;

    भव std::vector<std::string> get_निवेश_names(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव PartialShape get_output_shape(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव element::Type get_output_type(स्थिर पूर्णांक गाँठ_idx) स्थिर = 0;

    भव std::vector<माप_प्रकार> get_output_stride(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव int32_t * get_निवेश_op_params(पूर्णांक गाँठ_idx, स्थिर std::string & name) स्थिर = 0;

    भव int32_t * get_output_op_params(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव माप_प्रकार get_output_op_offset(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव std::vector<std::string> get_output_names(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव std::string get_inplace_op_src(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव bool is_view_like_alias_of(पूर्णांक गाँठ_idx, स्थिर std::string & view_src_name) स्थिर = 0;

    भव स्थिर std::string & get_op_type() स्थिर = 0;

    भव स्थिर std::string & get_op_type(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव स्थिर std::string & get_op_name() स्थिर = 0;

    भव स्थिर std::string & get_op_name(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव व्योम visit_subgraph(std::function<व्योम(std::shared_ptr<GgmlDecoder>, पूर्णांक गाँठ_idx)> गाँठ_visitor) स्थिर = 0;

    भव पूर्णांक get_op_case(पूर्णांक गाँठ_idx) स्थिर = 0;

    भव स्थिर std::map<std::string, Model_निवेश_Info> & get_model_inputs() स्थिर = 0;
    भव स्थिर std::map<std::string, Model_Extra_निवेश_Info> & get_model_extra_inputs() स्थिर = 0;
    भव स्थिर std::map<std::string, std::shared_ptr<ov::गाँठ_2>> & get_model_weights() स्थिर = 0;
    भव std::set<std::string> get_model_output_names() स्थिर = 0;

    भव int32_t * get_rope_params() स्थिर = 0;

    भव bool has_mixed_rope_params() स्थिर = 0;

    भव पूर्णांक get_ssm_state_आकार() स्थिर = 0;

    भव std::map<std::string, std::string> get_kv_param_res_names() स्थिर = 0;

    भव bool is_static() स्थिर = 0;

    भव bool is_stateful() स्थिर = 0;

    भव bool is_splited_model() स्थिर = 0;

    भव पूर्णांक is_swa_layer(पूर्णांक layer) स्थिर = 0;

    भव int32_t get_op_dynamic_dim(पूर्णांक गाँठ_idx) स्थिर = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
