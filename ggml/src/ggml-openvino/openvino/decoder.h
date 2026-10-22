#ಸೂಚನೆ once

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

ರಚನೆ Model_ಒಳಹರಿವು_Info {
    element::Type type;
    PartialShape shape;
};

ರಚನೆ Model_Extra_ಒಳಹರಿವು_Info {
    element::Type type;
    Shape shape;
    int64_t ಮೌಲ್ಯ;
    bool is_parameter;
};

class GgmlDecoder : ಸಾರ್ವಜನಿಕ Decoder_ತಳ {
ಸಾರ್ವಜನಿಕ:
    ವಾಸ್ತವಿಕ ov::Any get_attribute(ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ PartialShape get_ಒಳಹರಿವು_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<size_t> get_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ size_t get_view_ಒಳಹರಿವು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ size_t get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ size_t get_view_ಒಳಹರಿವು_src_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<size_t> get_view_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                      ಸ್ಥಿರಾಂಕ std::string & name,
                                                      size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<size_t> get_view_ಒಳಹರಿವು_src_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                          ಸ್ಥಿರಾಂಕ std::string & name,
                                                          size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ Shape get_view_ಒಳಹರಿವು_ggml_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ Shape get_view_ಒಳಹರಿವು_src_ggml_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ PartialShape get_view_ಒಳಹರಿವು_ov_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ PartialShape get_view_ಒಳಹರಿವು_src_ov_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                     ಸ್ಥಿರಾಂಕ std::string & name,
                                                     size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ std::string get_view_ಒಳಹರಿವು_name(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ std::string get_view_ಒಳಹರಿವು_src_name(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ element::Type get_ಒಳಹರಿವು_type(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ size_t get_ಒಳಹರಿವು_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ size_t get_ಒಳಹರಿವು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ ಶೂನ್ಯ get_ಒಳಹರಿವು_ಗಂಟು(size_t ಒಳಹರಿವು_ಪೋರ್ಟ್_idx,
                                std::string & ಉತ್ಪಾದಕ_name,
                                std::string & ಉತ್ಪಾದಕ_output_ಪೋರ್ಟ್_name,
                                size_t & ಉತ್ಪಾದಕ_output_ಪೋರ್ಟ್_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<std::string> get_ಒಳಹರಿವು_ಹೆಸರುಗಳು(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ PartialShape get_output_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ element::Type get_output_type(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<size_t> get_output_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ int32_t * get_ಒಳಹರಿವು_op_params(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ int32_t * get_output_op_params(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ size_t get_output_op_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<std::string> get_output_ಹೆಸರುಗಳು(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ std::string get_inplace_op_src(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ bool is_view_like_alias_ದ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & view_src_name) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::string & get_op_type() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::string & get_op_type(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::string & get_op_name() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::string & get_op_name(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ ಶೂನ್ಯ ಭೇಟಿ_subgraph(std::function<ಶೂನ್ಯ(std::shared_ptr<GgmlDecoder>, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx)> ಗಂಟು_visitor) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ get_op_case(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::map<std::string, Model_ಒಳಹರಿವು_Info> & get_model_inputs() ಸ್ಥಿರಾಂಕ = 0;
    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::map<std::string, Model_Extra_ಒಳಹರಿವು_Info> & get_model_extra_inputs() ಸ್ಥಿರಾಂಕ = 0;
    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> & get_model_weights() ಸ್ಥಿರಾಂಕ = 0;
    ವಾಸ್ತವಿಕ std::ಗಣ<std::string> get_model_output_ಹೆಸರುಗಳು() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ int32_t * get_rope_params() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ bool has_mixed_rope_params() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ get_ssm_state_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ std::map<std::string, std::string> get_kv_param_res_ಹೆಸರುಗಳು() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ bool is_static() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ bool is_stateful() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ bool is_splited_model() ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ is_swa_layer(ಪೂರ್ಣಾಂಕ layer) ಸ್ಥಿರಾಂಕ = 0;

    ವಾಸ್ತವಿಕ int32_t get_op_dynamic_dim(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
