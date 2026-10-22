#pragma once

#अन्तर्भाव "decoder.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <openvino/frontend/node_context.hpp>
#अन्तर्भाव <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

प्रकारनिर्देश std::map<std::string, Output<गाँठो_2>> TensorMap;

class गाँठो_Context : सार्वजनिक frontend::गाँठो_Context {
सार्वजनिक:
    गाँठो_Context(अचर std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                पूर्णाङ्क गाँठो_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::गाँठो_Context(decoder->get_op_type(गाँठो_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_गाँठो_idx(गाँठो_idx),
        m_translate_session(translate_session) {
        m_आगत_names = decoder->get_आगत_names(m_गाँठो_idx);
        m_output_names = decoder->get_output_names(m_गाँठो_idx);
    }

    TranslateSession * get_translate_session() अचर { फर्काऊ m_translate_session; }

    अचर std::vector<std::string> & get_आगत_names() अचर { फर्काऊ m_आगत_names; }

    अचर std::vector<std::string> & get_output_names() अचर { फर्काऊ m_output_names; }

    माप_प्रकार get_आगत_size() अचर override { फर्काऊ m_decoder->get_आगत_size(m_गाँठो_idx); }

    ov::element::Type get_आगत_type(माप_प्रकार index) अचर {
        फर्काऊ m_decoder->get_आगत_type(m_गाँठो_idx, m_आगत_names[index]);
    }

    PartialShape get_आगत_shape(माप_प्रकार आगत_index) अचर {
        फर्काऊ m_decoder->get_आगत_shape(m_गाँठो_idx, m_आगत_names[आगत_index]);
    }

    std::vector<माप_प्रकार> get_आगत_stride(माप_प्रकार index) अचर {
        फर्काऊ m_decoder->get_आगत_stride(m_गाँठो_idx, m_आगत_names[index]);
    }

    std::string get_output_name() अचर { फर्काऊ m_output_names[0]; }

    PartialShape get_output_shape() अचर { फर्काऊ m_decoder->get_output_shape(m_गाँठो_idx); }

    int32_t * get_आगत_op_params(माप_प्रकार index) अचर {
        फर्काऊ m_decoder->get_आगत_op_params(m_गाँठो_idx, m_आगत_names[index]);
    }

    माप_प्रकार get_view_आगत_size(माप_प्रकार index) अचर {
        फर्काऊ m_decoder->get_view_आगत_size(m_गाँठो_idx, m_आगत_names[index]);
    }

    माप_प्रकार get_view_आगत_offset(माप_प्रकार index, माप_प्रकार view_index) अचर {
        फर्काऊ m_decoder->get_view_आगत_offset(m_गाँठो_idx, m_आगत_names[index], view_index);
    }

    माप_प्रकार get_view_आगत_src_offset(माप_प्रकार index, माप_प्रकार view_index) अचर {
        फर्काऊ m_decoder->get_view_आगत_src_offset(m_गाँठो_idx, m_आगत_names[index], view_index);
    }

    std::vector<माप_प्रकार> get_view_आगत_stride(माप_प्रकार index, माप_प्रकार view_index) अचर {
        फर्काऊ m_decoder->get_view_आगत_stride(m_गाँठो_idx, m_आगत_names[index], view_index);
    }

    std::vector<माप_प्रकार> get_view_आगत_src_stride(माप_प्रकार index, माप_प्रकार view_index) अचर {
        फर्काऊ m_decoder->get_view_आगत_src_stride(m_गाँठो_idx, m_आगत_names[index], view_index);
    }

    ov::Shape get_view_आगत_ggml_shape(माप_प्रकार index, माप_प्रकार view_index) अचर {
        फर्काऊ m_decoder->get_view_आगत_ggml_shape(m_गाँठो_idx, m_आगत_names[index], view_index);
    }

    ov::Shape get_view_आगत_src_ggml_shape(माप_प्रकार index, माप_प्रकार view_index) अचर {
        फर्काऊ m_decoder->get_view_आगत_src_ggml_shape(m_गाँठो_idx, m_आगत_names[index], view_index);
    }

    ov::PartialShape get_view_आगत_ov_shape(माप_प्रकार index, माप_प्रकार view_index) अचर {
        फर्काऊ m_decoder->get_view_आगत_ov_shape(m_गाँठो_idx, m_आगत_names[index], view_index);
    }

    ov::PartialShape get_view_आगत_src_ov_shape(माप_प्रकार index, माप_प्रकार view_index) अचर {
        फर्काऊ m_decoder->get_view_आगत_src_ov_shape(m_गाँठो_idx, m_आगत_names[index], view_index);
    }

    std::string get_view_आगत_name(माप_प्रकार index, माप_प्रकार view_index) अचर {
        फर्काऊ m_decoder->get_view_आगत_name(m_गाँठो_idx, m_आगत_names[index], view_index);
    }

    std::string get_view_आगत_src_name(माप_प्रकार index, माप_प्रकार view_index) अचर {
        फर्काऊ m_decoder->get_view_आगत_src_name(m_गाँठो_idx, m_आगत_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() अचर { फर्काऊ m_decoder->get_op_dynamic_dim(m_गाँठो_idx); }

    int32_t * get_output_op_params() अचर { फर्काऊ m_decoder->get_output_op_params(m_गाँठो_idx); }

    माप_प्रकार get_output_op_offset() अचर { फर्काऊ m_decoder->get_output_op_offset(m_गाँठो_idx); }

    ov::element::Type get_output_type() अचर { फर्काऊ m_decoder->get_output_type(m_गाँठो_idx); }

    std::vector<माप_प्रकार> get_output_stride() अचर { फर्काऊ m_decoder->get_output_stride(m_गाँठो_idx); }

    Output<गाँठो_2> get_आगत(पूर्णाङ्क idx) अचर override {
        // Check if this input is a VIEW
        माप_प्रकार view_आगत_size = m_decoder->get_view_आगत_size(m_गाँठो_idx, m_आगत_names[idx]);
        यदि (view_आगत_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_आगत_src_name(m_गाँठो_idx, m_आगत_names[idx], view_आगत_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            स्वतः view_it = m_tensor_map->find(m_आगत_names[idx]);
            यदि (!base_name.empty() && view_it != m_tensor_map->end()) {
                स्वतः base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                यदि (base_it == m_tensor_map->end() ||
                    view_it->second.get_गाँठो_shared_ptr() != base_it->second.get_गाँठो_shared_ptr()) {
                    फर्काऊ view_it->second;
                }
                फर्काऊ base_it->second;
            }
            यदि (!base_name.empty()) {
                फर्काऊ m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        फर्काऊ m_tensor_map->at(m_आगत_names[idx]);
    }

    Output<गाँठो_2> get_आगत(अचर std::string & name) अचर override {
        यदि (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        फर्काऊ m_tensor_map->at(name);
    }

    bool has_आगत(अचर std::string & name) अचर { फर्काऊ m_tensor_map->find(name) != m_tensor_map->end(); }

    शून्य put_shared(अचर std::string & name, अचर Output<गाँठो_2> & मान) अचर {
        m_tensor_map->घुसाउ({name, मान});
    }

    अचर std::string & get_name() अचर override { फर्काऊ m_decoder->get_op_name(m_गाँठो_idx); }

    ov::Any get_attribute_as_any(अचर std::string & name) अचर override { फर्काऊ m_decoder->get_attribute(name); }

    पूर्णाङ्क get_op_case() अचर { फर्काऊ m_decoder->get_op_case(m_गाँठो_idx); }

    bool is_static() अचर { फर्काऊ m_decoder->is_static(); }

    bool is_stateful() अचर { फर्काऊ m_decoder->is_stateful(); }

    पूर्णाङ्क get_ssm_state_size() अचर { फर्काऊ m_decoder->get_ssm_state_size(); }

गुप्त:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    पूर्णाङ्क m_गाँठो_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_आगत_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(अचर ov::frontend::ggml::गाँठो_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
