#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/partial_shape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/shape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/decoder.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <set>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

namespace ov {
namespace frontend {
namespace ggml {

ਬਣਤਰ Model_ਨਿਵੇਸ਼_Info {
    element::Type type;
    PartialShape shape;
};

ਬਣਤਰ Model_Extra_ਨਿਵੇਸ਼_Info {
    element::Type type;
    Shape shape;
    int64_t ਮੁੱਲ;
    bool is_parameter;
};

class GgmlDecoder : ਜਨਤਕ Decoder_ਅਧਾਰ {
ਜਨਤਕ:
    ਆਭਾਸੀ ov::Any get_attribute(ਅਚਲ std::string & name) ਅਚਲ = 0;

    ਆਭਾਸੀ PartialShape get_ਨਿਵੇਸ਼_shape(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ = 0;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<size_t> get_ਨਿਵੇਸ਼_stride(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ = 0;

    ਆਭਾਸੀ size_t get_view_ਨਿਵੇਸ਼_ਮਾਪ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ = 0;

    ਆਭਾਸੀ size_t get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ size_t get_view_ਨਿਵੇਸ਼_src_ਵਿੱਥ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<size_t> get_view_ਨਿਵੇਸ਼_stride(ਪੂਰਨ ਗੰਢ_idx,
                                                      ਅਚਲ std::string & name,
                                                      size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<size_t> get_view_ਨਿਵੇਸ਼_src_stride(ਪੂਰਨ ਗੰਢ_idx,
                                                          ਅਚਲ std::string & name,
                                                          size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ Shape get_view_ਨਿਵੇਸ਼_ggml_shape(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ Shape get_view_ਨਿਵੇਸ਼_src_ggml_shape(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ PartialShape get_view_ਨਿਵੇਸ਼_ov_shape(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ PartialShape get_view_ਨਿਵੇਸ਼_src_ov_shape(ਪੂਰਨ ਗੰਢ_idx,
                                                     ਅਚਲ std::string & name,
                                                     size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ std::string get_view_ਨਿਵੇਸ਼_name(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ std::string get_view_ਨਿਵੇਸ਼_src_name(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ element::Type get_ਨਿਵੇਸ਼_type(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ = 0;

    ਆਭਾਸੀ size_t get_ਨਿਵੇਸ਼_ਮਾਪ() ਅਚਲ = 0;

    ਆਭਾਸੀ size_t get_ਨਿਵੇਸ਼_ਮਾਪ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ ਖਾਲੀ get_ਨਿਵੇਸ਼_ਗੰਢ(size_t ਨਿਵੇਸ਼_ਪੋਰਟ_idx,
                                std::string & ਉਤਪਾਦਕ_name,
                                std::string & ਉਤਪਾਦਕ_output_ਪੋਰਟ_name,
                                size_t & ਉਤਪਾਦਕ_output_ਪੋਰਟ_ਸੂਚਕਾਂਕ) ਅਚਲ = 0;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<std::string> get_ਨਿਵੇਸ਼_ਨਾਂ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ PartialShape get_output_shape(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ element::Type get_output_type(ਅਚਲ ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<size_t> get_output_stride(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ int32_t * get_ਨਿਵੇਸ਼_op_params(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & name) ਅਚਲ = 0;

    ਆਭਾਸੀ int32_t * get_output_op_params(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ size_t get_output_op_ਵਿੱਥ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ std::ਸਦਿਸ਼<std::string> get_output_ਨਾਂ(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ std::string get_inplace_op_src(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ bool is_view_like_alias_ਦਾ(ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::string & view_src_name) ਅਚਲ = 0;

    ਆਭਾਸੀ ਅਚਲ std::string & get_op_type() ਅਚਲ = 0;

    ਆਭਾਸੀ ਅਚਲ std::string & get_op_type(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ ਅਚਲ std::string & get_op_name() ਅਚਲ = 0;

    ਆਭਾਸੀ ਅਚਲ std::string & get_op_name(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ ਖਾਲੀ ਦੇਖੋ_subgraph(std::function<ਖਾਲੀ(std::shared_ptr<GgmlDecoder>, ਪੂਰਨ ਗੰਢ_idx)> ਗੰਢ_visitor) ਅਚਲ = 0;

    ਆਭਾਸੀ ਪੂਰਨ get_op_case(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;

    ਆਭਾਸੀ ਅਚਲ std::map<std::string, Model_ਨਿਵੇਸ਼_Info> & get_model_inputs() ਅਚਲ = 0;
    ਆਭਾਸੀ ਅਚਲ std::map<std::string, Model_Extra_ਨਿਵੇਸ਼_Info> & get_model_extra_inputs() ਅਚਲ = 0;
    ਆਭਾਸੀ ਅਚਲ std::map<std::string, std::shared_ptr<ov::ਗੰਢ_2>> & get_model_weights() ਅਚਲ = 0;
    ਆਭਾਸੀ std::ਸੈੱਟ<std::string> get_model_output_ਨਾਂ() ਅਚਲ = 0;

    ਆਭਾਸੀ int32_t * get_rope_params() ਅਚਲ = 0;

    ਆਭਾਸੀ bool has_mixed_rope_params() ਅਚਲ = 0;

    ਆਭਾਸੀ ਪੂਰਨ get_ssm_state_ਮਾਪ() ਅਚਲ = 0;

    ਆਭਾਸੀ std::map<std::string, std::string> get_kv_param_res_ਨਾਂ() ਅਚਲ = 0;

    ਆਭਾਸੀ bool is_static() ਅਚਲ = 0;

    ਆਭਾਸੀ bool is_stateful() ਅਚਲ = 0;

    ਆਭਾਸੀ bool is_splited_model() ਅਚਲ = 0;

    ਆਭਾਸੀ ਪੂਰਨ is_swa_layer(ਪੂਰਨ layer) ਅਚਲ = 0;

    ਆਭਾਸੀ int32_t get_op_dynamic_dim(ਪੂਰਨ ਗੰਢ_idx) ਅਚਲ = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
