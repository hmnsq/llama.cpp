#pragma once

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/partial_shape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/shape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/decoder.hpp>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <string>

namespace ov {
namespace frontend {
namespace ggml {

ಸಂರಚನಾ Model_ಒಳಹರಿವು_Info {
    element::Type type;
    PartialShape shape;
};

ಸಂರಚನಾ Model_Extra_ಒಳಹರಿವು_Info {
    element::Type type;
    Shape shape;
    int64_t ಮೌಲ್ಯ;
    bool is_parameter;
};

class GgmlDecoder : ಸಾರ್ವಜನಿಕ DecoderBase {
ಸಾರ್ವಜನಿಕ:
    ಆಭಾಸ ov::Any get_attribute(ಧ್ರುವ std::string & name) ಧ್ರುವ = 0;

    ಆಭಾಸ PartialShape get_ಒಳಹರಿವು_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ = 0;

    ಆಭಾಸ std::vector<size_t> get_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ = 0;

    ಆಭಾಸ size_t get_view_ಒಳಹರಿವು_size(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ = 0;

    ಆಭಾಸ size_t get_view_ಒಳಹರಿವು_offset(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ = 0;

    ಆಭಾಸ size_t get_view_ಒಳಹರಿವು_src_offset(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ = 0;

    ಆಭಾಸ std::vector<size_t> get_view_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                      ಧ್ರುವ std::string & name,
                                                      size_t view_index) ಧ್ರುವ = 0;

    ಆಭಾಸ std::vector<size_t> get_view_ಒಳಹರಿವು_src_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                          ಧ್ರುವ std::string & name,
                                                          size_t view_index) ಧ್ರುವ = 0;

    ಆಭಾಸ Shape get_view_ಒಳಹರಿವು_ggml_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ = 0;

    ಆಭಾಸ Shape get_view_ಒಳಹರಿವು_src_ggml_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ = 0;

    ಆಭಾಸ PartialShape get_view_ಒಳಹರಿವು_ov_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ = 0;

    ಆಭಾಸ PartialShape get_view_ಒಳಹರಿವು_src_ov_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                     ಧ್ರುವ std::string & name,
                                                     size_t view_index) ಧ್ರುವ = 0;

    ಆಭಾಸ std::string get_view_ಒಳಹರಿವು_name(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ = 0;

    ಆಭಾಸ std::string get_view_ಒಳಹರಿವು_src_name(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ = 0;

    ಆಭಾಸ element::Type get_ಒಳಹರಿವು_type(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ = 0;

    ಆಭಾಸ size_t get_ಒಳಹರಿವು_size() ಧ್ರುವ = 0;

    ಆಭಾಸ size_t get_ಒಳಹರಿವು_size(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ ಶೂನ್ಯ get_ಒಳಹರಿವು_ಗಂಟು(size_t ಒಳಹರಿವು_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ಧ್ರುವ = 0;

    ಆಭಾಸ std::vector<std::string> get_ಒಳಹರಿವು_names(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ PartialShape get_output_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ element::Type get_output_type(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ std::vector<size_t> get_output_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ int32_t * get_ಒಳಹರಿವು_op_params(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ = 0;

    ಆಭಾಸ int32_t * get_output_op_params(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ size_t get_output_op_offset(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ std::vector<std::string> get_output_names(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ std::string get_inplace_op_src(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ bool is_view_like_alias_of(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & view_src_name) ಧ್ರುವ = 0;

    ಆಭಾಸ ಧ್ರುವ std::string & get_op_type() ಧ್ರುವ = 0;

    ಆಭಾಸ ಧ್ರುವ std::string & get_op_type(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ ಧ್ರುವ std::string & get_op_name() ಧ್ರುವ = 0;

    ಆಭಾಸ ಧ್ರುವ std::string & get_op_name(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ ಶೂನ್ಯ visit_subgraph(std::function<ಶೂನ್ಯ(std::shared_ptr<GgmlDecoder>, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx)> ಗಂಟು_visitor) ಧ್ರುವ = 0;

    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ get_op_case(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;

    ಆಭಾಸ ಧ್ರುವ std::map<std::string, Model_ಒಳಹರಿವು_Info> & get_model_inputs() ಧ್ರುವ = 0;
    ಆಭಾಸ ಧ್ರುವ std::map<std::string, Model_Extra_ಒಳಹರಿವು_Info> & get_model_extra_inputs() ಧ್ರುವ = 0;
    ಆಭಾಸ ಧ್ರುವ std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> & get_model_weights() ಧ್ರುವ = 0;
    ಆಭಾಸ std::set<std::string> get_model_output_names() ಧ್ರುವ = 0;

    ಆಭಾಸ int32_t * get_rope_params() ಧ್ರುವ = 0;

    ಆಭಾಸ bool has_mixed_rope_params() ಧ್ರುವ = 0;

    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ get_ssm_state_size() ಧ್ರುವ = 0;

    ಆಭಾಸ std::map<std::string, std::string> get_kv_param_res_names() ಧ್ರುವ = 0;

    ಆಭಾಸ bool is_static() ಧ್ರುವ = 0;

    ಆಭಾಸ bool is_stateful() ಧ್ರುವ = 0;

    ಆಭಾಸ bool is_splited_model() ಧ್ರುವ = 0;

    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ is_swa_layer(ಪೂರ್ಣಾಙ್ಕ layer) ಧ್ರುವ = 0;

    ಆಭಾಸ int32_t get_op_dynamic_dim(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
