#pragma once

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/frontend/decoder.hpp>
#include <set>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

struct ModelInputInfo {
    element::Type type;
    PartialShape shape;
};

struct ModelExtraInputInfo {
    element::Type type;
    Shape shape;
    int64_t value;
    bool is_parameter;
};

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string & name) const = 0;

    virtual PartialShape get_input_shape(цео node_idx, const std::string & name) const = 0;

    virtual std::vector<size_t> get_input_stride(цео node_idx, const std::string & name) const = 0;

    virtual size_t get_view_input_size(цео node_idx, const std::string & name) const = 0;

    virtual size_t get_view_input_offset(цео node_idx, const std::string & name, size_t view_index) const = 0;

    virtual size_t get_view_input_src_offset(цео node_idx, const std::string & name, size_t view_index) const = 0;

    virtual std::vector<size_t> get_view_input_stride(цео node_idx,
                                                      const std::string & name,
                                                      size_t view_index) const = 0;

    virtual std::vector<size_t> get_view_input_src_stride(цео node_idx,
                                                          const std::string & name,
                                                          size_t view_index) const = 0;

    virtual Shape get_view_input_ggml_shape(цео node_idx, const std::string & name, size_t view_index) const = 0;

    virtual Shape get_view_input_src_ggml_shape(цео node_idx, const std::string & name, size_t view_index) const = 0;

    virtual PartialShape get_view_input_ov_shape(цео node_idx, const std::string & name, size_t view_index) const = 0;

    virtual PartialShape get_view_input_src_ov_shape(цео node_idx,
                                                     const std::string & name,
                                                     size_t view_index) const = 0;

    virtual std::string get_view_input_name(цео node_idx, const std::string & name, size_t view_index) const = 0;

    virtual std::string get_view_input_src_name(цео node_idx, const std::string & name, size_t view_index) const = 0;

    virtual element::Type get_input_type(цео node_idx, const std::string & name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(цео node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(цео node_idx) const = 0;

    virtual PartialShape get_output_shape(цео node_idx) const = 0;

    virtual element::Type get_output_type(const цео node_idx) const = 0;

    virtual std::vector<size_t> get_output_stride(цео node_idx) const = 0;

    virtual int32_t * get_input_op_params(цео node_idx, const std::string & name) const = 0;

    virtual int32_t * get_output_op_params(цео node_idx) const = 0;

    virtual size_t get_output_op_offset(цео node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(цео node_idx) const = 0;

    virtual std::string get_inplace_op_src(цео node_idx) const = 0;

    virtual bool is_view_like_alias_of(цео node_idx, const std::string & view_src_name) const = 0;

    virtual const std::string & get_op_type() const = 0;

    virtual const std::string & get_op_type(цео node_idx) const = 0;

    virtual const std::string & get_op_name() const = 0;

    virtual const std::string & get_op_name(цео node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, цео node_idx)> node_visitor) const = 0;

    virtual цео get_op_case(цео node_idx) const = 0;

    virtual const std::map<std::string, ModelInputInfo> & get_model_inputs() const = 0;
    virtual const std::map<std::string, ModelExtraInputInfo> & get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_weights() const = 0;
    virtual std::set<std::string> get_model_output_names() const = 0;

    virtual int32_t * get_rope_params() const = 0;

    virtual bool has_mixed_rope_params() const = 0;

    virtual цео get_ssm_state_size() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual bool is_splited_model() const = 0;

    virtual цео is_swa_layer(цео layer) const = 0;

    virtual int32_t get_op_dynamic_dim(цео node_idx) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
