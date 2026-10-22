#pragma once

#ಅನ್ತರ್ಭಾವ "decoder.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/node_context.hpp>
#ಅನ್ತರ್ಭಾವ <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

ಪ್ರಕಾರನಿರ್ದೇಶ std::map<std::string, Output<ಗಂಟು_2>> TensorMap;

class ಗಂಟು_Context : ಸಾರ್ವಜನಿಕ frontend::ಗಂಟು_Context {
ಸಾರ್ವಜನಿಕ:
    ಗಂಟು_Context(ಧ್ರುವ std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::ಗಂಟು_Context(decoder->get_op_type(ಗಂಟು_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_ಗಂಟು_idx(ಗಂಟು_idx),
        m_translate_session(translate_session) {
        m_ಒಳಹರಿವು_names = decoder->get_ಒಳಹರಿವು_names(m_ಗಂಟು_idx);
        m_output_names = decoder->get_output_names(m_ಗಂಟು_idx);
    }

    TranslateSession * get_translate_session() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_translate_session; }

    ಧ್ರುವ std::vector<std::string> & get_ಒಳಹರಿವು_names() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_ಒಳಹರಿವು_names; }

    ಧ್ರುವ std::vector<std::string> & get_output_names() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_output_names; }

    size_t get_ಒಳಹರಿವು_size() ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ m_decoder->get_ಒಳಹರಿವು_size(m_ಗಂಟು_idx); }

    ov::element::Type get_ಒಳಹರಿವು_type(size_t index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_ಒಳಹರಿವು_type(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index]);
    }

    PartialShape get_ಒಳಹರಿವು_shape(size_t ಒಳಹರಿವು_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_ಒಳಹರಿವು_shape(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[ಒಳಹರಿವು_index]);
    }

    std::vector<size_t> get_ಒಳಹರಿವು_stride(size_t index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_ಒಳಹರಿವು_stride(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index]);
    }

    std::string get_output_name() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_output_names[0]; }

    PartialShape get_output_shape() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_decoder->get_output_shape(m_ಗಂಟು_idx); }

    int32_t * get_ಒಳಹರಿವು_op_params(size_t index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_ಒಳಹರಿವು_op_params(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index]);
    }

    size_t get_view_ಒಳಹರಿವು_size(size_t index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_size(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index]);
    }

    size_t get_view_ಒಳಹರಿವು_offset(size_t index, size_t view_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_offset(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index], view_index);
    }

    size_t get_view_ಒಳಹರಿವು_src_offset(size_t index, size_t view_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_src_offset(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index], view_index);
    }

    std::vector<size_t> get_view_ಒಳಹರಿವು_stride(size_t index, size_t view_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_stride(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index], view_index);
    }

    std::vector<size_t> get_view_ಒಳಹರಿವು_src_stride(size_t index, size_t view_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_src_stride(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index], view_index);
    }

    ov::Shape get_view_ಒಳಹರಿವು_ggml_shape(size_t index, size_t view_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_ggml_shape(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index], view_index);
    }

    ov::Shape get_view_ಒಳಹರಿವು_src_ggml_shape(size_t index, size_t view_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_src_ggml_shape(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index], view_index);
    }

    ov::PartialShape get_view_ಒಳಹರಿವು_ov_shape(size_t index, size_t view_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_ov_shape(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index], view_index);
    }

    ov::PartialShape get_view_ಒಳಹರಿವು_src_ov_shape(size_t index, size_t view_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_src_ov_shape(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index], view_index);
    }

    std::string get_view_ಒಳಹರಿವು_name(size_t index, size_t view_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_name(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index], view_index);
    }

    std::string get_view_ಒಳಹರಿವು_src_name(size_t index, size_t view_index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_decoder->get_view_ಒಳಹರಿವು_src_name(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_decoder->get_op_dynamic_dim(m_ಗಂಟು_idx); }

    int32_t * get_output_op_params() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_decoder->get_output_op_params(m_ಗಂಟು_idx); }

    size_t get_output_op_offset() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_decoder->get_output_op_offset(m_ಗಂಟು_idx); }

    ov::element::Type get_output_type() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_decoder->get_output_type(m_ಗಂಟು_idx); }

    std::vector<size_t> get_output_stride() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_decoder->get_output_stride(m_ಗಂಟು_idx); }

    Output<ಗಂಟು_2> get_ಒಳಹರಿವು(ಪೂರ್ಣಾಙ್ಕ idx) ಧ್ರುವ override {
        // Check if this input is a VIEW
        size_t view_ಒಳಹರಿವು_size = m_decoder->get_view_ಒಳಹರಿವು_size(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[idx]);
        ಯದಿ (view_ಒಳಹರಿವು_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_ಒಳಹರಿವು_src_name(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_names[idx], view_ಒಳಹರಿವು_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_ಒಳಹರಿವು_names[idx]);
            ಯದಿ (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                ಯದಿ (base_it == m_tensor_map->end() ||
                    view_it->second.get_ಗಂಟು_shared_ptr() != base_it->second.get_ಗಂಟು_shared_ptr()) {
                    ಪ್ರತ್ಯಾಗಮ view_it->second;
                }
                ಪ್ರತ್ಯಾಗಮ base_it->second;
            }
            ಯದಿ (!base_name.empty()) {
                ಪ್ರತ್ಯಾಗಮ m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        ಪ್ರತ್ಯಾಗಮ m_tensor_map->at(m_ಒಳಹರಿವು_names[idx]);
    }

    Output<ಗಂಟು_2> get_ಒಳಹರಿವು(ಧ್ರುವ std::string & name) ಧ್ರುವ override {
        ಯದಿ (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        ಪ್ರತ್ಯಾಗಮ m_tensor_map->at(name);
    }

    bool has_ಒಳಹರಿವು(ಧ್ರುವ std::string & name) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_tensor_map->find(name) != m_tensor_map->end(); }

    ಶೂನ್ಯ put_shared(ಧ್ರುವ std::string & name, ಧ್ರುವ Output<ಗಂಟು_2> & ಮೌಲ್ಯ) ಧ್ರುವ {
        m_tensor_map->ಸೇರಿಸು({name, ಮೌಲ್ಯ});
    }

    ಧ್ರುವ std::string & get_name() ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ m_decoder->get_op_name(m_ಗಂಟು_idx); }

    ov::Any get_attribute_as_any(ಧ್ರುವ std::string & name) ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ m_decoder->get_attribute(name); }

    ಪೂರ್ಣಾಙ್ಕ get_op_case() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_decoder->get_op_case(m_ಗಂಟು_idx); }

    bool is_static() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_decoder->is_static(); }

    bool is_stateful() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_decoder->is_stateful(); }

    ಪೂರ್ಣಾಙ್ಕ get_ssm_state_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_decoder->get_ssm_state_size(); }

ಗುಪ್ತ:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    ಪೂರ್ಣಾಙ್ಕ m_ಗಂಟು_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_ಒಳಹರಿವು_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ಧ್ರುವ ov::frontend::ggml::ಗಂಟು_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
