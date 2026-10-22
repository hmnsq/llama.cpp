#pragma once

#ضمن "decoder.h"

#ضمن <cstdint>
#ضمن <openvino/frontend/node_context.hpp>
#ضمن <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

تعريفنوع std::map<std::string, Output<Node>> TensorMap;

class NodeContext : عام frontend::NodeContext {
عام:
    NodeContext(ثابت std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                صحيح node_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_node_idx(node_idx),
        m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession * get_translate_session() ثابت { ܦܢܐ m_translate_session; }

    ثابت std::vector<std::string> & get_input_names() ثابت { ܦܢܐ m_input_names; }

    ثابت std::vector<std::string> & get_output_names() ثابت { ܦܢܐ m_output_names; }

    size_t get_input_size() ثابت override { ܦܢܐ m_decoder->get_input_size(m_node_idx); }

    ov::element::Type get_input_type(size_t index) ثابت {
        ܦܢܐ m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) ثابت {
        ܦܢܐ m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) ثابت {
        ܦܢܐ m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() ثابت { ܦܢܐ m_output_names[0]; }

    PartialShape get_output_shape() ثابت { ܦܢܐ m_decoder->get_output_shape(m_node_idx); }

    int32_t * get_input_op_params(size_t index) ثابت {
        ܦܢܐ m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    size_t get_view_input_size(size_t index) ثابت {
        ܦܢܐ m_decoder->get_view_input_size(m_node_idx, m_input_names[index]);
    }

    size_t get_view_input_offset(size_t index, size_t view_index) ثابت {
        ܦܢܐ m_decoder->get_view_input_offset(m_node_idx, m_input_names[index], view_index);
    }

    size_t get_view_input_src_offset(size_t index, size_t view_index) ثابت {
        ܦܢܐ m_decoder->get_view_input_src_offset(m_node_idx, m_input_names[index], view_index);
    }

    std::vector<size_t> get_view_input_stride(size_t index, size_t view_index) ثابت {
        ܦܢܐ m_decoder->get_view_input_stride(m_node_idx, m_input_names[index], view_index);
    }

    std::vector<size_t> get_view_input_src_stride(size_t index, size_t view_index) ثابت {
        ܦܢܐ m_decoder->get_view_input_src_stride(m_node_idx, m_input_names[index], view_index);
    }

    ov::Shape get_view_input_ggml_shape(size_t index, size_t view_index) ثابت {
        ܦܢܐ m_decoder->get_view_input_ggml_shape(m_node_idx, m_input_names[index], view_index);
    }

    ov::Shape get_view_input_src_ggml_shape(size_t index, size_t view_index) ثابت {
        ܦܢܐ m_decoder->get_view_input_src_ggml_shape(m_node_idx, m_input_names[index], view_index);
    }

    ov::PartialShape get_view_input_ov_shape(size_t index, size_t view_index) ثابت {
        ܦܢܐ m_decoder->get_view_input_ov_shape(m_node_idx, m_input_names[index], view_index);
    }

    ov::PartialShape get_view_input_src_ov_shape(size_t index, size_t view_index) ثابت {
        ܦܢܐ m_decoder->get_view_input_src_ov_shape(m_node_idx, m_input_names[index], view_index);
    }

    std::string get_view_input_name(size_t index, size_t view_index) ثابت {
        ܦܢܐ m_decoder->get_view_input_name(m_node_idx, m_input_names[index], view_index);
    }

    std::string get_view_input_src_name(size_t index, size_t view_index) ثابت {
        ܦܢܐ m_decoder->get_view_input_src_name(m_node_idx, m_input_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ثابت { ܦܢܐ m_decoder->get_op_dynamic_dim(m_node_idx); }

    int32_t * get_output_op_params() ثابت { ܦܢܐ m_decoder->get_output_op_params(m_node_idx); }

    size_t get_output_op_offset() ثابت { ܦܢܐ m_decoder->get_output_op_offset(m_node_idx); }

    ov::element::Type get_output_type() ثابت { ܦܢܐ m_decoder->get_output_type(m_node_idx); }

    std::vector<size_t> get_output_stride() ثابت { ܦܢܐ m_decoder->get_output_stride(m_node_idx); }

    Output<Node> get_input(صحيح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_input_size = m_decoder->get_view_input_size(m_node_idx, m_input_names[idx]);
        ܐܢ (view_input_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_input_src_name(m_node_idx, m_input_names[idx], view_input_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_input_names[idx]);
            ܐܢ (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                ܐܢ (base_it == m_tensor_map->end() ||
                    view_it->second.get_node_shared_ptr() != base_it->second.get_node_shared_ptr()) {
                    ܦܢܐ view_it->second;
                }
                ܦܢܐ base_it->second;
            }
            ܐܢ (!base_name.empty()) {
                ܦܢܐ m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        ܦܢܐ m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(ثابت std::string & name) ثابت override {
        ܐܢ (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        ܦܢܐ m_tensor_map->at(name);
    }

    bool has_input(ثابت std::string & name) ثابت { ܦܢܐ m_tensor_map->find(name) != m_tensor_map->end(); }

    فراغ put_shared(ثابت std::string & name, ثابت Output<Node> & value) ثابت {
        m_tensor_map->insert({name, value});
    }

    ثابت std::string & get_name() ثابت override { ܦܢܐ m_decoder->get_op_name(m_node_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { ܦܢܐ m_decoder->get_attribute(name); }

    صحيح get_op_case() ثابت { ܦܢܐ m_decoder->get_op_case(m_node_idx); }

    bool is_static() ثابت { ܦܢܐ m_decoder->is_static(); }

    bool is_stateful() ثابت { ܦܢܐ m_decoder->is_stateful(); }

    صحيح get_ssm_state_size() ثابت { ܦܢܐ m_decoder->get_ssm_state_size(); }

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحيح m_node_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ثابت ov::frontend::ggml::NodeContext &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
