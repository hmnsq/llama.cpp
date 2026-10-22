#आशय once

#समावेश "decoder.h"

#समावेश <cstdint>
#समावेश <openvino/frontend/node_context.hpp>
#समावेश <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

प्रकार std::map<std::string, Output<गाँठ_2>> TensorMap;

class गाँठ_Context : खुला frontend::गाँठ_Context {
खुला:
    गाँठ_Context(स्थिर std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                पूर्णांक गाँठ_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::गाँठ_Context(decoder->get_op_type(गाँठ_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_गाँठ_idx(गाँठ_idx),
        m_translate_session(translate_session) {
        m_निवेश_names = decoder->get_निवेश_names(m_गाँठ_idx);
        m_output_names = decoder->get_output_names(m_गाँठ_idx);
    }

    TranslateSession * get_translate_session() स्थिर { वापस m_translate_session; }

    स्थिर std::vector<std::string> & get_निवेश_names() स्थिर { वापस m_निवेश_names; }

    स्थिर std::vector<std::string> & get_output_names() स्थिर { वापस m_output_names; }

    माप_प्रकार get_निवेश_आकार() स्थिर override { वापस m_decoder->get_निवेश_आकार(m_गाँठ_idx); }

    ov::element::Type get_निवेश_type(माप_प्रकार सूचकांक) स्थिर {
        वापस m_decoder->get_निवेश_type(m_गाँठ_idx, m_निवेश_names[सूचकांक]);
    }

    PartialShape get_निवेश_shape(माप_प्रकार निवेश_सूचकांक) स्थिर {
        वापस m_decoder->get_निवेश_shape(m_गाँठ_idx, m_निवेश_names[निवेश_सूचकांक]);
    }

    std::vector<माप_प्रकार> get_निवेश_stride(माप_प्रकार सूचकांक) स्थिर {
        वापस m_decoder->get_निवेश_stride(m_गाँठ_idx, m_निवेश_names[सूचकांक]);
    }

    std::string get_output_name() स्थिर { वापस m_output_names[0]; }

    PartialShape get_output_shape() स्थिर { वापस m_decoder->get_output_shape(m_गाँठ_idx); }

    int32_t * get_निवेश_op_params(माप_प्रकार सूचकांक) स्थिर {
        वापस m_decoder->get_निवेश_op_params(m_गाँठ_idx, m_निवेश_names[सूचकांक]);
    }

    माप_प्रकार get_view_निवेश_आकार(माप_प्रकार सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_आकार(m_गाँठ_idx, m_निवेश_names[सूचकांक]);
    }

    माप_प्रकार get_view_निवेश_offset(माप_प्रकार सूचकांक, माप_प्रकार view_सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_offset(m_गाँठ_idx, m_निवेश_names[सूचकांक], view_सूचकांक);
    }

    माप_प्रकार get_view_निवेश_src_offset(माप_प्रकार सूचकांक, माप_प्रकार view_सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_src_offset(m_गाँठ_idx, m_निवेश_names[सूचकांक], view_सूचकांक);
    }

    std::vector<माप_प्रकार> get_view_निवेश_stride(माप_प्रकार सूचकांक, माप_प्रकार view_सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_stride(m_गाँठ_idx, m_निवेश_names[सूचकांक], view_सूचकांक);
    }

    std::vector<माप_प्रकार> get_view_निवेश_src_stride(माप_प्रकार सूचकांक, माप_प्रकार view_सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_src_stride(m_गाँठ_idx, m_निवेश_names[सूचकांक], view_सूचकांक);
    }

    ov::Shape get_view_निवेश_ggml_shape(माप_प्रकार सूचकांक, माप_प्रकार view_सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_ggml_shape(m_गाँठ_idx, m_निवेश_names[सूचकांक], view_सूचकांक);
    }

    ov::Shape get_view_निवेश_src_ggml_shape(माप_प्रकार सूचकांक, माप_प्रकार view_सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_src_ggml_shape(m_गाँठ_idx, m_निवेश_names[सूचकांक], view_सूचकांक);
    }

    ov::PartialShape get_view_निवेश_ov_shape(माप_प्रकार सूचकांक, माप_प्रकार view_सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_ov_shape(m_गाँठ_idx, m_निवेश_names[सूचकांक], view_सूचकांक);
    }

    ov::PartialShape get_view_निवेश_src_ov_shape(माप_प्रकार सूचकांक, माप_प्रकार view_सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_src_ov_shape(m_गाँठ_idx, m_निवेश_names[सूचकांक], view_सूचकांक);
    }

    std::string get_view_निवेश_name(माप_प्रकार सूचकांक, माप_प्रकार view_सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_name(m_गाँठ_idx, m_निवेश_names[सूचकांक], view_सूचकांक);
    }

    std::string get_view_निवेश_src_name(माप_प्रकार सूचकांक, माप_प्रकार view_सूचकांक) स्थिर {
        वापस m_decoder->get_view_निवेश_src_name(m_गाँठ_idx, m_निवेश_names[सूचकांक], view_सूचकांक);
    }

    int32_t get_op_dynamic_dim() स्थिर { वापस m_decoder->get_op_dynamic_dim(m_गाँठ_idx); }

    int32_t * get_output_op_params() स्थिर { वापस m_decoder->get_output_op_params(m_गाँठ_idx); }

    माप_प्रकार get_output_op_offset() स्थिर { वापस m_decoder->get_output_op_offset(m_गाँठ_idx); }

    ov::element::Type get_output_type() स्थिर { वापस m_decoder->get_output_type(m_गाँठ_idx); }

    std::vector<माप_प्रकार> get_output_stride() स्थिर { वापस m_decoder->get_output_stride(m_गाँठ_idx); }

    Output<गाँठ_2> get_निवेश(पूर्णांक idx) स्थिर override {
        // Check if this input is a VIEW
        माप_प्रकार view_निवेश_आकार = m_decoder->get_view_निवेश_आकार(m_गाँठ_idx, m_निवेश_names[idx]);
        अगर (view_निवेश_आकार > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_निवेश_src_name(m_गाँठ_idx, m_निवेश_names[idx], view_निवेश_आकार - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            स्वतः view_it = m_tensor_map->find(m_निवेश_names[idx]);
            अगर (!base_name.empty() && view_it != m_tensor_map->end()) {
                स्वतः base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                अगर (base_it == m_tensor_map->end() ||
                    view_it->second.get_गाँठ_shared_ptr() != base_it->second.get_गाँठ_shared_ptr()) {
                    वापस view_it->second;
                }
                वापस base_it->second;
            }
            अगर (!base_name.empty()) {
                वापस m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        वापस m_tensor_map->at(m_निवेश_names[idx]);
    }

    Output<गाँठ_2> get_निवेश(स्थिर std::string & name) स्थिर override {
        अगर (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        वापस m_tensor_map->at(name);
    }

    bool has_निवेश(स्थिर std::string & name) स्थिर { वापस m_tensor_map->find(name) != m_tensor_map->end(); }

    व्योम put_shared(स्थिर std::string & name, स्थिर Output<गाँठ_2> & मूल्य) स्थिर {
        m_tensor_map->जोड़ो({name, मूल्य});
    }

    स्थिर std::string & get_name() स्थिर override { वापस m_decoder->get_op_name(m_गाँठ_idx); }

    ov::Any get_attribute_as_any(स्थिर std::string & name) स्थिर override { वापस m_decoder->get_attribute(name); }

    पूर्णांक get_op_case() स्थिर { वापस m_decoder->get_op_case(m_गाँठ_idx); }

    bool is_static() स्थिर { वापस m_decoder->is_static(); }

    bool is_stateful() स्थिर { वापस m_decoder->is_stateful(); }

    पूर्णांक get_ssm_state_आकार() स्थिर { वापस m_decoder->get_ssm_state_आकार(); }

निजी:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    पूर्णांक m_गाँठ_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_निवेश_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(स्थिर ov::frontend::ggml::गाँठ_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
