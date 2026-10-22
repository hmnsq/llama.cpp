#pragma once

#अन्तर्भाव "decoder.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <openvino/frontend/node_context.hpp>
#अन्तर्भाव <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

प्रकारनिर्देश std::map<std::string, Output<गाँठ_2>> TensorMap;

class गाँठ_Context : सार्वजनिक frontend::गाँठ_Context {
सार्वजनिक:
    गाँठ_Context(ध्रुव std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                पूर्णाङ्क गाँठ_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::गाँठ_Context(decoder->get_op_type(गाँठ_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_गाँठ_idx(गाँठ_idx),
        m_translate_session(translate_session) {
        m_निवेश_names = decoder->get_निवेश_names(m_गाँठ_idx);
        m_output_names = decoder->get_output_names(m_गाँठ_idx);
    }

    TranslateSession * get_translate_session() ध्रुव { प्रत्यागम m_translate_session; }

    ध्रुव std::vector<std::string> & get_निवेश_names() ध्रुव { प्रत्यागम m_निवेश_names; }

    ध्रुव std::vector<std::string> & get_output_names() ध्रुव { प्रत्यागम m_output_names; }

    size_t get_निवेश_size() ध्रुव override { प्रत्यागम m_decoder->get_निवेश_size(m_गाँठ_idx); }

    ov::element::Type get_निवेश_type(size_t index) ध्रुव {
        प्रत्यागम m_decoder->get_निवेश_type(m_गाँठ_idx, m_निवेश_names[index]);
    }

    PartialShape get_निवेश_shape(size_t निवेश_index) ध्रुव {
        प्रत्यागम m_decoder->get_निवेश_shape(m_गाँठ_idx, m_निवेश_names[निवेश_index]);
    }

    std::vector<size_t> get_निवेश_stride(size_t index) ध्रुव {
        प्रत्यागम m_decoder->get_निवेश_stride(m_गाँठ_idx, m_निवेश_names[index]);
    }

    std::string get_output_name() ध्रुव { प्रत्यागम m_output_names[0]; }

    PartialShape get_output_shape() ध्रुव { प्रत्यागम m_decoder->get_output_shape(m_गाँठ_idx); }

    int32_t * get_निवेश_op_params(size_t index) ध्रुव {
        प्रत्यागम m_decoder->get_निवेश_op_params(m_गाँठ_idx, m_निवेश_names[index]);
    }

    size_t get_view_निवेश_size(size_t index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_size(m_गाँठ_idx, m_निवेश_names[index]);
    }

    size_t get_view_निवेश_offset(size_t index, size_t view_index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_offset(m_गाँठ_idx, m_निवेश_names[index], view_index);
    }

    size_t get_view_निवेश_src_offset(size_t index, size_t view_index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_src_offset(m_गाँठ_idx, m_निवेश_names[index], view_index);
    }

    std::vector<size_t> get_view_निवेश_stride(size_t index, size_t view_index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_stride(m_गाँठ_idx, m_निवेश_names[index], view_index);
    }

    std::vector<size_t> get_view_निवेश_src_stride(size_t index, size_t view_index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_src_stride(m_गाँठ_idx, m_निवेश_names[index], view_index);
    }

    ov::Shape get_view_निवेश_ggml_shape(size_t index, size_t view_index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_ggml_shape(m_गाँठ_idx, m_निवेश_names[index], view_index);
    }

    ov::Shape get_view_निवेश_src_ggml_shape(size_t index, size_t view_index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_src_ggml_shape(m_गाँठ_idx, m_निवेश_names[index], view_index);
    }

    ov::PartialShape get_view_निवेश_ov_shape(size_t index, size_t view_index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_ov_shape(m_गाँठ_idx, m_निवेश_names[index], view_index);
    }

    ov::PartialShape get_view_निवेश_src_ov_shape(size_t index, size_t view_index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_src_ov_shape(m_गाँठ_idx, m_निवेश_names[index], view_index);
    }

    std::string get_view_निवेश_name(size_t index, size_t view_index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_name(m_गाँठ_idx, m_निवेश_names[index], view_index);
    }

    std::string get_view_निवेश_src_name(size_t index, size_t view_index) ध्रुव {
        प्रत्यागम m_decoder->get_view_निवेश_src_name(m_गाँठ_idx, m_निवेश_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ध्रुव { प्रत्यागम m_decoder->get_op_dynamic_dim(m_गाँठ_idx); }

    int32_t * get_output_op_params() ध्रुव { प्रत्यागम m_decoder->get_output_op_params(m_गाँठ_idx); }

    size_t get_output_op_offset() ध्रुव { प्रत्यागम m_decoder->get_output_op_offset(m_गाँठ_idx); }

    ov::element::Type get_output_type() ध्रुव { प्रत्यागम m_decoder->get_output_type(m_गाँठ_idx); }

    std::vector<size_t> get_output_stride() ध्रुव { प्रत्यागम m_decoder->get_output_stride(m_गाँठ_idx); }

    Output<गाँठ_2> get_निवेश(पूर्णाङ्क idx) ध्रुव override {
        // Check if this input is a VIEW
        size_t view_निवेश_size = m_decoder->get_view_निवेश_size(m_गाँठ_idx, m_निवेश_names[idx]);
        यदि (view_निवेश_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_निवेश_src_name(m_गाँठ_idx, m_निवेश_names[idx], view_निवेश_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_निवेश_names[idx]);
            यदि (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                यदि (base_it == m_tensor_map->end() ||
                    view_it->second.get_गाँठ_shared_ptr() != base_it->second.get_गाँठ_shared_ptr()) {
                    प्रत्यागम view_it->second;
                }
                प्रत्यागम base_it->second;
            }
            यदि (!base_name.empty()) {
                प्रत्यागम m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        प्रत्यागम m_tensor_map->at(m_निवेश_names[idx]);
    }

    Output<गाँठ_2> get_निवेश(ध्रुव std::string & name) ध्रुव override {
        यदि (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        प्रत्यागम m_tensor_map->at(name);
    }

    bool has_निवेश(ध्रुव std::string & name) ध्रुव { प्रत्यागम m_tensor_map->find(name) != m_tensor_map->end(); }

    शून्य put_shared(ध्रुव std::string & name, ध्रुव Output<गाँठ_2> & मूल्य) ध्रुव {
        m_tensor_map->जोड़ो({name, मूल्य});
    }

    ध्रुव std::string & get_name() ध्रुव override { प्रत्यागम m_decoder->get_op_name(m_गाँठ_idx); }

    ov::Any get_attribute_as_any(ध्रुव std::string & name) ध्रुव override { प्रत्यागम m_decoder->get_attribute(name); }

    पूर्णाङ्क get_op_case() ध्रुव { प्रत्यागम m_decoder->get_op_case(m_गाँठ_idx); }

    bool is_static() ध्रुव { प्रत्यागम m_decoder->is_static(); }

    bool is_stateful() ध्रुव { प्रत्यागम m_decoder->is_stateful(); }

    पूर्णाङ्क get_ssm_state_size() ध्रुव { प्रत्यागम m_decoder->get_ssm_state_size(); }

गुप्त:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    पूर्णाङ्क m_गाँठ_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_निवेश_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ध्रुव ov::frontend::ggml::गाँठ_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
