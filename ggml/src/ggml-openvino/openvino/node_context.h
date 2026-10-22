#pragma once

#அந்தர்பாவ "decoder.h"

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <openvino/frontend/node_context.hpp>
#அந்தர்பாவ <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

வகைவரை std::map<std::string, Output<முனை_2>> TensorMap;

class முனை_Context : பொது frontend::முனை_Context {
பொது:
    முனை_Context(மாறிலி std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                முழு முனை_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::முனை_Context(decoder->get_op_type(முனை_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_முனை_idx(முனை_idx),
        m_translate_session(translate_session) {
        m_உள்ளீடு_names = decoder->get_உள்ளீடு_names(m_முனை_idx);
        m_output_names = decoder->get_output_names(m_முனை_idx);
    }

    TranslateSession * get_translate_session() மாறிலி { திரும்பு m_translate_session; }

    மாறிலி std::vector<std::string> & get_உள்ளீடு_names() மாறிலி { திரும்பு m_உள்ளீடு_names; }

    மாறிலி std::vector<std::string> & get_output_names() மாறிலி { திரும்பு m_output_names; }

    size_t get_உள்ளீடு_size() மாறிலி override { திரும்பு m_decoder->get_உள்ளீடு_size(m_முனை_idx); }

    ov::element::Type get_உள்ளீடு_type(size_t index) மாறிலி {
        திரும்பு m_decoder->get_உள்ளீடு_type(m_முனை_idx, m_உள்ளீடு_names[index]);
    }

    PartialShape get_உள்ளீடு_shape(size_t உள்ளீடு_index) மாறிலி {
        திரும்பு m_decoder->get_உள்ளீடு_shape(m_முனை_idx, m_உள்ளீடு_names[உள்ளீடு_index]);
    }

    std::vector<size_t> get_உள்ளீடு_stride(size_t index) மாறிலி {
        திரும்பு m_decoder->get_உள்ளீடு_stride(m_முனை_idx, m_உள்ளீடு_names[index]);
    }

    std::string get_output_name() மாறிலி { திரும்பு m_output_names[0]; }

    PartialShape get_output_shape() மாறிலி { திரும்பு m_decoder->get_output_shape(m_முனை_idx); }

    int32_t * get_உள்ளீடு_op_params(size_t index) மாறிலி {
        திரும்பு m_decoder->get_உள்ளீடு_op_params(m_முனை_idx, m_உள்ளீடு_names[index]);
    }

    size_t get_view_உள்ளீடு_size(size_t index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_size(m_முனை_idx, m_உள்ளீடு_names[index]);
    }

    size_t get_view_உள்ளீடு_offset(size_t index, size_t view_index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_offset(m_முனை_idx, m_உள்ளீடு_names[index], view_index);
    }

    size_t get_view_உள்ளீடு_src_offset(size_t index, size_t view_index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_src_offset(m_முனை_idx, m_உள்ளீடு_names[index], view_index);
    }

    std::vector<size_t> get_view_உள்ளீடு_stride(size_t index, size_t view_index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_stride(m_முனை_idx, m_உள்ளீடு_names[index], view_index);
    }

    std::vector<size_t> get_view_உள்ளீடு_src_stride(size_t index, size_t view_index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_src_stride(m_முனை_idx, m_உள்ளீடு_names[index], view_index);
    }

    ov::Shape get_view_உள்ளீடு_ggml_shape(size_t index, size_t view_index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_ggml_shape(m_முனை_idx, m_உள்ளீடு_names[index], view_index);
    }

    ov::Shape get_view_உள்ளீடு_src_ggml_shape(size_t index, size_t view_index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_src_ggml_shape(m_முனை_idx, m_உள்ளீடு_names[index], view_index);
    }

    ov::PartialShape get_view_உள்ளீடு_ov_shape(size_t index, size_t view_index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_ov_shape(m_முனை_idx, m_உள்ளீடு_names[index], view_index);
    }

    ov::PartialShape get_view_உள்ளீடு_src_ov_shape(size_t index, size_t view_index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_src_ov_shape(m_முனை_idx, m_உள்ளீடு_names[index], view_index);
    }

    std::string get_view_உள்ளீடு_name(size_t index, size_t view_index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_name(m_முனை_idx, m_உள்ளீடு_names[index], view_index);
    }

    std::string get_view_உள்ளீடு_src_name(size_t index, size_t view_index) மாறிலி {
        திரும்பு m_decoder->get_view_உள்ளீடு_src_name(m_முனை_idx, m_உள்ளீடு_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() மாறிலி { திரும்பு m_decoder->get_op_dynamic_dim(m_முனை_idx); }

    int32_t * get_output_op_params() மாறிலி { திரும்பு m_decoder->get_output_op_params(m_முனை_idx); }

    size_t get_output_op_offset() மாறிலி { திரும்பு m_decoder->get_output_op_offset(m_முனை_idx); }

    ov::element::Type get_output_type() மாறிலி { திரும்பு m_decoder->get_output_type(m_முனை_idx); }

    std::vector<size_t> get_output_stride() மாறிலி { திரும்பு m_decoder->get_output_stride(m_முனை_idx); }

    Output<முனை_2> get_உள்ளீடு(முழு idx) மாறிலி override {
        // Check if this input is a VIEW
        size_t view_உள்ளீடு_size = m_decoder->get_view_உள்ளீடு_size(m_முனை_idx, m_உள்ளீடு_names[idx]);
        எனில் (view_உள்ளீடு_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_உள்ளீடு_src_name(m_முனை_idx, m_உள்ளீடு_names[idx], view_உள்ளீடு_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_உள்ளீடு_names[idx]);
            எனில் (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                எனில் (base_it == m_tensor_map->end() ||
                    view_it->second.get_முனை_shared_ptr() != base_it->second.get_முனை_shared_ptr()) {
                    திரும்பு view_it->second;
                }
                திரும்பு base_it->second;
            }
            எனில் (!base_name.empty()) {
                திரும்பு m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        திரும்பு m_tensor_map->at(m_உள்ளீடு_names[idx]);
    }

    Output<முனை_2> get_உள்ளீடு(மாறிலி std::string & name) மாறிலி override {
        எனில் (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        திரும்பு m_tensor_map->at(name);
    }

    bool has_உள்ளீடு(மாறிலி std::string & name) மாறிலி { திரும்பு m_tensor_map->find(name) != m_tensor_map->end(); }

    வெறுமை put_shared(மாறிலி std::string & name, மாறிலி Output<முனை_2> & மதிப்பு) மாறிலி {
        m_tensor_map->செருகு({name, மதிப்பு});
    }

    மாறிலி std::string & get_name() மாறிலி override { திரும்பு m_decoder->get_op_name(m_முனை_idx); }

    ov::Any get_attribute_as_any(மாறிலி std::string & name) மாறிலி override { திரும்பு m_decoder->get_attribute(name); }

    முழு get_op_case() மாறிலி { திரும்பு m_decoder->get_op_case(m_முனை_idx); }

    bool is_static() மாறிலி { திரும்பு m_decoder->is_static(); }

    bool is_stateful() மாறிலி { திரும்பு m_decoder->is_stateful(); }

    முழு get_ssm_state_size() மாறிலி { திரும்பு m_decoder->get_ssm_state_size(); }

தனிப்பட்ட:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    முழு m_முனை_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_உள்ளீடு_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(மாறிலி ov::frontend::ggml::முனை_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
