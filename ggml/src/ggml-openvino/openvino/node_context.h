#pragma once

#अन्तर्भाव "decoder.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <openvino/frontend/node_context.hpp>
#अन्तर्भाव <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

प्रकारनिर्देश std::map<std::string, Output<ग्रन्थिः_2>> TensorMap;

class ग्रन्थिः_Context : सार्वजनिक frontend::ग्रन्थिः_Context {
सार्वजनिक:
    ग्रन्थिः_Context(ध्रुव std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                पूर्णाङ्क ग्रन्थिः_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::ग्रन्थिः_Context(decoder->get_op_type(ग्रन्थिः_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_ग्रन्थिः_idx(ग्रन्थिः_idx),
        m_translate_session(translate_session) {
        m_निवेशः_names = decoder->get_निवेशः_names(m_ग्रन्थिः_idx);
        m_output_names = decoder->get_output_names(m_ग्रन्थिः_idx);
    }

    TranslateSession * get_translate_session() ध्रुव { पटिनिवत्त m_translate_session; }

    ध्रुव std::vector<std::string> & get_निवेशः_names() ध्रुव { पटिनिवत्त m_निवेशः_names; }

    ध्रुव std::vector<std::string> & get_output_names() ध्रुव { पटिनिवत्त m_output_names; }

    size_t get_निवेशः_size() ध्रुव override { पटिनिवत्त m_decoder->get_निवेशः_size(m_ग्रन्थिः_idx); }

    ov::element::Type get_निवेशः_type(size_t index) ध्रुव {
        पटिनिवत्त m_decoder->get_निवेशः_type(m_ग्रन्थिः_idx, m_निवेशः_names[index]);
    }

    PartialShape get_निवेशः_shape(size_t निवेशः_index) ध्रुव {
        पटिनिवत्त m_decoder->get_निवेशः_shape(m_ग्रन्थिः_idx, m_निवेशः_names[निवेशः_index]);
    }

    std::vector<size_t> get_निवेशः_stride(size_t index) ध्रुव {
        पटिनिवत्त m_decoder->get_निवेशः_stride(m_ग्रन्थिः_idx, m_निवेशः_names[index]);
    }

    std::string get_output_name() ध्रुव { पटिनिवत्त m_output_names[0]; }

    PartialShape get_output_shape() ध्रुव { पटिनिवत्त m_decoder->get_output_shape(m_ग्रन्थिः_idx); }

    int32_t * get_निवेशः_op_params(size_t index) ध्रुव {
        पटिनिवत्त m_decoder->get_निवेशः_op_params(m_ग्रन्थिः_idx, m_निवेशः_names[index]);
    }

    size_t get_view_निवेशः_size(size_t index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_size(m_ग्रन्थिः_idx, m_निवेशः_names[index]);
    }

    size_t get_view_निवेशः_offset(size_t index, size_t view_index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_offset(m_ग्रन्थिः_idx, m_निवेशः_names[index], view_index);
    }

    size_t get_view_निवेशः_src_offset(size_t index, size_t view_index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_src_offset(m_ग्रन्थिः_idx, m_निवेशः_names[index], view_index);
    }

    std::vector<size_t> get_view_निवेशः_stride(size_t index, size_t view_index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_stride(m_ग्रन्थिः_idx, m_निवेशः_names[index], view_index);
    }

    std::vector<size_t> get_view_निवेशः_src_stride(size_t index, size_t view_index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_src_stride(m_ग्रन्थिः_idx, m_निवेशः_names[index], view_index);
    }

    ov::Shape get_view_निवेशः_ggml_shape(size_t index, size_t view_index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_ggml_shape(m_ग्रन्थिः_idx, m_निवेशः_names[index], view_index);
    }

    ov::Shape get_view_निवेशः_src_ggml_shape(size_t index, size_t view_index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_src_ggml_shape(m_ग्रन्थिः_idx, m_निवेशः_names[index], view_index);
    }

    ov::PartialShape get_view_निवेशः_ov_shape(size_t index, size_t view_index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_ov_shape(m_ग्रन्थिः_idx, m_निवेशः_names[index], view_index);
    }

    ov::PartialShape get_view_निवेशः_src_ov_shape(size_t index, size_t view_index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_src_ov_shape(m_ग्रन्थिः_idx, m_निवेशः_names[index], view_index);
    }

    std::string get_view_निवेशः_name(size_t index, size_t view_index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_name(m_ग्रन्थिः_idx, m_निवेशः_names[index], view_index);
    }

    std::string get_view_निवेशः_src_name(size_t index, size_t view_index) ध्रुव {
        पटिनिवत्त m_decoder->get_view_निवेशः_src_name(m_ग्रन्थिः_idx, m_निवेशः_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ध्रुव { पटिनिवत्त m_decoder->get_op_dynamic_dim(m_ग्रन्थिः_idx); }

    int32_t * get_output_op_params() ध्रुव { पटिनिवत्त m_decoder->get_output_op_params(m_ग्रन्थिः_idx); }

    size_t get_output_op_offset() ध्रुव { पटिनिवत्त m_decoder->get_output_op_offset(m_ग्रन्थिः_idx); }

    ov::element::Type get_output_type() ध्रुव { पटिनिवत्त m_decoder->get_output_type(m_ग्रन्थिः_idx); }

    std::vector<size_t> get_output_stride() ध्रुव { पटिनिवत्त m_decoder->get_output_stride(m_ग्रन्थिः_idx); }

    Output<ग्रन्थिः_2> get_निवेशः(पूर्णाङ्क idx) ध्रुव override {
        // Check if this input is a VIEW
        size_t view_निवेशः_size = m_decoder->get_view_निवेशः_size(m_ग्रन्थिः_idx, m_निवेशः_names[idx]);
        सचे (view_निवेशः_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_निवेशः_src_name(m_ग्रन्थिः_idx, m_निवेशः_names[idx], view_निवेशः_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_निवेशः_names[idx]);
            सचे (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                सचे (base_it == m_tensor_map->end() ||
                    view_it->second.get_ग्रन्थिः_shared_ptr() != base_it->second.get_ग्रन्थिः_shared_ptr()) {
                    पटिनिवत्त view_it->second;
                }
                पटिनिवत्त base_it->second;
            }
            सचे (!base_name.empty()) {
                पटिनिवत्त m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        पटिनिवत्त m_tensor_map->at(m_निवेशः_names[idx]);
    }

    Output<ग्रन्थिः_2> get_निवेशः(ध्रुव std::string & name) ध्रुव override {
        सचे (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        पटिनिवत्त m_tensor_map->at(name);
    }

    bool has_निवेशः(ध्रुव std::string & name) ध्रुव { पटिनिवत्त m_tensor_map->find(name) != m_tensor_map->end(); }

    शून्य put_shared(ध्रुव std::string & name, ध्रुव Output<ग्रन्थिः_2> & मानम्) ध्रुव {
        m_tensor_map->निवेशय({name, मानम्});
    }

    ध्रुव std::string & get_name() ध्रुव override { पटिनिवत्त m_decoder->get_op_name(m_ग्रन्थिः_idx); }

    ov::Any get_attribute_as_any(ध्रुव std::string & name) ध्रुव override { पटिनिवत्त m_decoder->get_attribute(name); }

    पूर्णाङ्क get_op_case() ध्रुव { पटिनिवत्त m_decoder->get_op_case(m_ग्रन्थिः_idx); }

    bool is_static() ध्रुव { पटिनिवत्त m_decoder->is_static(); }

    bool is_stateful() ध्रुव { पटिनिवत्त m_decoder->is_stateful(); }

    पूर्णाङ्क get_ssm_state_size() ध्रुव { पटिनिवत्त m_decoder->get_ssm_state_size(); }

गुप्त:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    पूर्णाङ्क m_ग्रन्थिः_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_निवेशः_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ध्रुव ov::frontend::ggml::ग्रन्थिः_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
