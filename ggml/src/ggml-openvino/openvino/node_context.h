#pragma once

#અન્તર્ભાવ "decoder.h"

#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <openvino/frontend/node_context.hpp>
#અન્તર્ભાવ <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

પ્રકારવ્યાખ્યા std::map<std::string, Output<ગાંઠ_2>> TensorMap;

class ગાંઠ_Context : જાહેર frontend::ગાંઠ_Context {
જાહેર:
    ગાંઠ_Context(અચલ std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                પૂર્ણાંક ગાંઠ_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::ગાંઠ_Context(decoder->get_op_type(ગાંઠ_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_ગાંઠ_idx(ગાંઠ_idx),
        m_translate_session(translate_session) {
        m_આગત_names = decoder->get_આગત_names(m_ગાંઠ_idx);
        m_output_names = decoder->get_output_names(m_ગાંઠ_idx);
    }

    TranslateSession * get_translate_session() અચલ { પાછા m_translate_session; }

    અચલ std::vector<std::string> & get_આગત_names() અચલ { પાછા m_આગત_names; }

    અચલ std::vector<std::string> & get_output_names() અચલ { પાછા m_output_names; }

    size_t get_આગત_size() અચલ override { પાછા m_decoder->get_આગત_size(m_ગાંઠ_idx); }

    ov::element::Type get_આગત_type(size_t index) અચલ {
        પાછા m_decoder->get_આગત_type(m_ગાંઠ_idx, m_આગત_names[index]);
    }

    PartialShape get_આગત_shape(size_t આગત_index) અચલ {
        પાછા m_decoder->get_આગત_shape(m_ગાંઠ_idx, m_આગત_names[આગત_index]);
    }

    std::vector<size_t> get_આગત_stride(size_t index) અચલ {
        પાછા m_decoder->get_આગત_stride(m_ગાંઠ_idx, m_આગત_names[index]);
    }

    std::string get_output_name() અચલ { પાછા m_output_names[0]; }

    PartialShape get_output_shape() અચલ { પાછા m_decoder->get_output_shape(m_ગાંઠ_idx); }

    int32_t * get_આગત_op_params(size_t index) અચલ {
        પાછા m_decoder->get_આગત_op_params(m_ગાંઠ_idx, m_આગત_names[index]);
    }

    size_t get_view_આગત_size(size_t index) અચલ {
        પાછા m_decoder->get_view_આગત_size(m_ગાંઠ_idx, m_આગત_names[index]);
    }

    size_t get_view_આગત_offset(size_t index, size_t view_index) અચલ {
        પાછા m_decoder->get_view_આગત_offset(m_ગાંઠ_idx, m_આગત_names[index], view_index);
    }

    size_t get_view_આગત_src_offset(size_t index, size_t view_index) અચલ {
        પાછા m_decoder->get_view_આગત_src_offset(m_ગાંઠ_idx, m_આગત_names[index], view_index);
    }

    std::vector<size_t> get_view_આગત_stride(size_t index, size_t view_index) અચલ {
        પાછા m_decoder->get_view_આગત_stride(m_ગાંઠ_idx, m_આગત_names[index], view_index);
    }

    std::vector<size_t> get_view_આગત_src_stride(size_t index, size_t view_index) અચલ {
        પાછા m_decoder->get_view_આગત_src_stride(m_ગાંઠ_idx, m_આગત_names[index], view_index);
    }

    ov::Shape get_view_આગત_ggml_shape(size_t index, size_t view_index) અચલ {
        પાછા m_decoder->get_view_આગત_ggml_shape(m_ગાંઠ_idx, m_આગત_names[index], view_index);
    }

    ov::Shape get_view_આગત_src_ggml_shape(size_t index, size_t view_index) અચલ {
        પાછા m_decoder->get_view_આગત_src_ggml_shape(m_ગાંઠ_idx, m_આગત_names[index], view_index);
    }

    ov::PartialShape get_view_આગત_ov_shape(size_t index, size_t view_index) અચલ {
        પાછા m_decoder->get_view_આગત_ov_shape(m_ગાંઠ_idx, m_આગત_names[index], view_index);
    }

    ov::PartialShape get_view_આગત_src_ov_shape(size_t index, size_t view_index) અચલ {
        પાછા m_decoder->get_view_આગત_src_ov_shape(m_ગાંઠ_idx, m_આગત_names[index], view_index);
    }

    std::string get_view_આગત_name(size_t index, size_t view_index) અચલ {
        પાછા m_decoder->get_view_આગત_name(m_ગાંઠ_idx, m_આગત_names[index], view_index);
    }

    std::string get_view_આગત_src_name(size_t index, size_t view_index) અચલ {
        પાછા m_decoder->get_view_આગત_src_name(m_ગાંઠ_idx, m_આગત_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() અચલ { પાછા m_decoder->get_op_dynamic_dim(m_ગાંઠ_idx); }

    int32_t * get_output_op_params() અચલ { પાછા m_decoder->get_output_op_params(m_ગાંઠ_idx); }

    size_t get_output_op_offset() અચલ { પાછા m_decoder->get_output_op_offset(m_ગાંઠ_idx); }

    ov::element::Type get_output_type() અચલ { પાછા m_decoder->get_output_type(m_ગાંઠ_idx); }

    std::vector<size_t> get_output_stride() અચલ { પાછા m_decoder->get_output_stride(m_ગાંઠ_idx); }

    Output<ગાંઠ_2> get_આગત(પૂર્ણાંક idx) અચલ override {
        // Check if this input is a VIEW
        size_t view_આગત_size = m_decoder->get_view_આગત_size(m_ગાંઠ_idx, m_આગત_names[idx]);
        જો (view_આગત_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_આગત_src_name(m_ગાંઠ_idx, m_આગત_names[idx], view_આગત_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_આગત_names[idx]);
            જો (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                જો (base_it == m_tensor_map->end() ||
                    view_it->second.get_ગાંઠ_shared_ptr() != base_it->second.get_ગાંઠ_shared_ptr()) {
                    પાછા view_it->second;
                }
                પાછા base_it->second;
            }
            જો (!base_name.empty()) {
                પાછા m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        પાછા m_tensor_map->at(m_આગત_names[idx]);
    }

    Output<ગાંઠ_2> get_આગત(અચલ std::string & name) અચલ override {
        જો (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        પાછા m_tensor_map->at(name);
    }

    bool has_આગત(અચલ std::string & name) અચલ { પાછા m_tensor_map->find(name) != m_tensor_map->end(); }

    શૂન્ય put_shared(અચલ std::string & name, અચલ Output<ગાંઠ_2> & કિંમત) અચલ {
        m_tensor_map->ઉમેરો({name, કિંમત});
    }

    અચલ std::string & get_name() અચલ override { પાછા m_decoder->get_op_name(m_ગાંઠ_idx); }

    ov::Any get_attribute_as_any(અચલ std::string & name) અચલ override { પાછા m_decoder->get_attribute(name); }

    પૂર્ણાંક get_op_case() અચલ { પાછા m_decoder->get_op_case(m_ગાંઠ_idx); }

    bool is_static() અચલ { પાછા m_decoder->is_static(); }

    bool is_stateful() અચલ { પાછા m_decoder->is_stateful(); }

    પૂર્ણાંક get_ssm_state_size() અચલ { પાછા m_decoder->get_ssm_state_size(); }

ખાનગી:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    પૂર્ણાંક m_ગાંઠ_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_આગત_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(અચલ ov::frontend::ggml::ગાંઠ_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
