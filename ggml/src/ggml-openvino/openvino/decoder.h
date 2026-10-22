#సూచన once

#అన్తర్భావ <cstdint>
#అన్తర్భావ <map>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/partial_shape.hpp>
#అన్తర్భావ <openvino/core/shape.hpp>
#అన్తర్భావ <openvino/frontend/decoder.hpp>
#అన్తర్భావ <set>
#అన్తర్భావ <string>

namespace ov {
namespace frontend {
namespace ggml {

నిర్మాణం Model_ఆదానం_Info {
    element::Type type;
    PartialShape shape;
};

నిర్మాణం Model_Extra_ఆదానం_Info {
    element::Type type;
    Shape shape;
    int64_t విలువ;
    bool is_parameter;
};

class GgmlDecoder : బహిరంగ Decoder_పునాది {
బహిరంగ:
    మాయా ov::Any get_attribute(స్థిరరాశి std::string & name) స్థిరరాశి = 0;

    మాయా PartialShape get_ఆదానం_shape(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి = 0;

    మాయా std::సదిశ<size_t> get_ఆదానం_stride(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి = 0;

    మాయా size_t get_view_ఆదానం_కొలత(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి = 0;

    మాయా size_t get_view_ఆదానం_స్థానభ్రంశం(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి = 0;

    మాయా size_t get_view_ఆదానం_src_స్థానభ్రంశం(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి = 0;

    మాయా std::సదిశ<size_t> get_view_ఆదానం_stride(పూర్ణాంకం కణుపు_idx,
                                                      స్థిరరాశి std::string & name,
                                                      size_t view_సూచీ) స్థిరరాశి = 0;

    మాయా std::సదిశ<size_t> get_view_ఆదానం_src_stride(పూర్ణాంకం కణుపు_idx,
                                                          స్థిరరాశి std::string & name,
                                                          size_t view_సూచీ) స్థిరరాశి = 0;

    మాయా Shape get_view_ఆదానం_ggml_shape(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి = 0;

    మాయా Shape get_view_ఆదానం_src_ggml_shape(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి = 0;

    మాయా PartialShape get_view_ఆదానం_ov_shape(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి = 0;

    మాయా PartialShape get_view_ఆదానం_src_ov_shape(పూర్ణాంకం కణుపు_idx,
                                                     స్థిరరాశి std::string & name,
                                                     size_t view_సూచీ) స్థిరరాశి = 0;

    మాయా std::string get_view_ఆదానం_name(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి = 0;

    మాయా std::string get_view_ఆదానం_src_name(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి = 0;

    మాయా element::Type get_ఆదానం_type(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి = 0;

    మాయా size_t get_ఆదానం_కొలత() స్థిరరాశి = 0;

    మాయా size_t get_ఆదానం_కొలత(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా శూన్య get_ఆదానం_కణుపు(size_t ఆదానం_పోర్టు_idx,
                                std::string & ఉత్పాదకుడు_name,
                                std::string & ఉత్పాదకుడు_output_పోర్టు_name,
                                size_t & ఉత్పాదకుడు_output_పోర్టు_సూచీ) స్థిరరాశి = 0;

    మాయా std::సదిశ<std::string> get_ఆదానం_పేర్లు(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా PartialShape get_output_shape(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా element::Type get_output_type(స్థిరరాశి పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా std::సదిశ<size_t> get_output_stride(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా int32_t * get_ఆదానం_op_params(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి = 0;

    మాయా int32_t * get_output_op_params(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా size_t get_output_op_స్థానభ్రంశం(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా std::సదిశ<std::string> get_output_పేర్లు(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా std::string get_inplace_op_src(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా bool is_view_like_alias_యొక్క(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & view_src_name) స్థిరరాశి = 0;

    మాయా స్థిరరాశి std::string & get_op_type() స్థిరరాశి = 0;

    మాయా స్థిరరాశి std::string & get_op_type(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా స్థిరరాశి std::string & get_op_name() స్థిరరాశి = 0;

    మాయా స్థిరరాశి std::string & get_op_name(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా శూన్య సందర్శించు_subgraph(std::function<శూన్య(std::shared_ptr<GgmlDecoder>, పూర్ణాంకం కణుపు_idx)> కణుపు_visitor) స్థిరరాశి = 0;

    మాయా పూర్ణాంకం get_op_case(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;

    మాయా స్థిరరాశి std::map<std::string, Model_ఆదానం_Info> & get_model_inputs() స్థిరరాశి = 0;
    మాయా స్థిరరాశి std::map<std::string, Model_Extra_ఆదానం_Info> & get_model_extra_inputs() స్థిరరాశి = 0;
    మాయా స్థిరరాశి std::map<std::string, std::shared_ptr<ov::కణుపు_2>> & get_model_weights() స్థిరరాశి = 0;
    మాయా std::సమితి<std::string> get_model_output_పేర్లు() స్థిరరాశి = 0;

    మాయా int32_t * get_rope_params() స్థిరరాశి = 0;

    మాయా bool has_mixed_rope_params() స్థిరరాశి = 0;

    మాయా పూర్ణాంకం get_ssm_state_కొలత() స్థిరరాశి = 0;

    మాయా std::map<std::string, std::string> get_kv_param_res_పేర్లు() స్థిరరాశి = 0;

    మాయా bool is_static() స్థిరరాశి = 0;

    మాయా bool is_stateful() స్థిరరాశి = 0;

    మాయా bool is_splited_model() స్థిరరాశి = 0;

    మాయా పూర్ణాంకం is_swa_layer(పూర్ణాంకం layer) స్థిరరాశి = 0;

    మాయా int32_t get_op_dynamic_dim(పూర్ణాంకం కణుపు_idx) స్థిరరాశి = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
