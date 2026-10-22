#pragma once

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <map>
#அந்தர்பாவ <openvino/core/node.hpp>
#அந்தர்பாவ <openvino/core/partial_shape.hpp>
#அந்தர்பாவ <openvino/core/shape.hpp>
#அந்தர்பாவ <openvino/frontend/decoder.hpp>
#அந்தர்பாவ <set>
#அந்தர்பாவ <string>

namespace ov {
namespace frontend {
namespace ggml {

கட்டமைப்பு Model_உள்ளீடு_Info {
    element::Type type;
    PartialShape shape;
};

கட்டமைப்பு Model_Extra_உள்ளீடு_Info {
    element::Type type;
    Shape shape;
    int64_t மதிப்பு;
    bool is_parameter;
};

class GgmlDecoder : பொது DecoderBase {
பொது:
    மெய்நிகர் ov::Any get_attribute(மாறிலி std::string & name) மாறிலி = 0;

    மெய்நிகர் PartialShape get_உள்ளீடு_shape(முழு முனை_idx, மாறிலி std::string & name) மாறிலி = 0;

    மெய்நிகர் std::vector<size_t> get_உள்ளீடு_stride(முழு முனை_idx, மாறிலி std::string & name) மாறிலி = 0;

    மெய்நிகர் size_t get_view_உள்ளீடு_size(முழு முனை_idx, மாறிலி std::string & name) மாறிலி = 0;

    மெய்நிகர் size_t get_view_உள்ளீடு_offset(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி = 0;

    மெய்நிகர் size_t get_view_உள்ளீடு_src_offset(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி = 0;

    மெய்நிகர் std::vector<size_t> get_view_உள்ளீடு_stride(முழு முனை_idx,
                                                      மாறிலி std::string & name,
                                                      size_t view_index) மாறிலி = 0;

    மெய்நிகர் std::vector<size_t> get_view_உள்ளீடு_src_stride(முழு முனை_idx,
                                                          மாறிலி std::string & name,
                                                          size_t view_index) மாறிலி = 0;

    மெய்நிகர் Shape get_view_உள்ளீடு_ggml_shape(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி = 0;

    மெய்நிகர் Shape get_view_உள்ளீடு_src_ggml_shape(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி = 0;

    மெய்நிகர் PartialShape get_view_உள்ளீடு_ov_shape(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி = 0;

    மெய்நிகர் PartialShape get_view_உள்ளீடு_src_ov_shape(முழு முனை_idx,
                                                     மாறிலி std::string & name,
                                                     size_t view_index) மாறிலி = 0;

    மெய்நிகர் std::string get_view_உள்ளீடு_name(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி = 0;

    மெய்நிகர் std::string get_view_உள்ளீடு_src_name(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி = 0;

    மெய்நிகர் element::Type get_உள்ளீடு_type(முழு முனை_idx, மாறிலி std::string & name) மாறிலி = 0;

    மெய்நிகர் size_t get_உள்ளீடு_size() மாறிலி = 0;

    மெய்நிகர் size_t get_உள்ளீடு_size(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் வெறுமை get_உள்ளீடு_முனை(size_t உள்ளீடு_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) மாறிலி = 0;

    மெய்நிகர் std::vector<std::string> get_உள்ளீடு_names(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் PartialShape get_output_shape(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் element::Type get_output_type(மாறிலி முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் std::vector<size_t> get_output_stride(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் int32_t * get_உள்ளீடு_op_params(முழு முனை_idx, மாறிலி std::string & name) மாறிலி = 0;

    மெய்நிகர் int32_t * get_output_op_params(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் size_t get_output_op_offset(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் std::vector<std::string> get_output_names(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் std::string get_inplace_op_src(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் bool is_view_like_alias_of(முழு முனை_idx, மாறிலி std::string & view_src_name) மாறிலி = 0;

    மெய்நிகர் மாறிலி std::string & get_op_type() மாறிலி = 0;

    மெய்நிகர் மாறிலி std::string & get_op_type(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் மாறிலி std::string & get_op_name() மாறிலி = 0;

    மெய்நிகர் மாறிலி std::string & get_op_name(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் வெறுமை visit_subgraph(std::function<வெறுமை(std::shared_ptr<GgmlDecoder>, முழு முனை_idx)> முனை_visitor) மாறிலி = 0;

    மெய்நிகர் முழு get_op_case(முழு முனை_idx) மாறிலி = 0;

    மெய்நிகர் மாறிலி std::map<std::string, Model_உள்ளீடு_Info> & get_model_inputs() மாறிலி = 0;
    மெய்நிகர் மாறிலி std::map<std::string, Model_Extra_உள்ளீடு_Info> & get_model_extra_inputs() மாறிலி = 0;
    மெய்நிகர் மாறிலி std::map<std::string, std::shared_ptr<ov::முனை_2>> & get_model_weights() மாறிலி = 0;
    மெய்நிகர் std::set<std::string> get_model_output_names() மாறிலி = 0;

    மெய்நிகர் int32_t * get_rope_params() மாறிலி = 0;

    மெய்நிகர் bool has_mixed_rope_params() மாறிலி = 0;

    மெய்நிகர் முழு get_ssm_state_size() மாறிலி = 0;

    மெய்நிகர் std::map<std::string, std::string> get_kv_param_res_names() மாறிலி = 0;

    மெய்நிகர் bool is_static() மாறிலி = 0;

    மெய்நிகர் bool is_stateful() மாறிலி = 0;

    மெய்நிகர் bool is_splited_model() மாறிலி = 0;

    மெய்நிகர் முழு is_swa_layer(முழு layer) மாறிலி = 0;

    மெய்நிகர் int32_t get_op_dynamic_dim(முழு முனை_idx) மாறிலி = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
