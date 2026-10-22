#pragma once

#अन्तर्भाव "decoder.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <openvino/frontend/node_context.hpp>
#अन्तर्भाव <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

प्रकारव्याख्या std::map<std::string, Output<गाठ_2>> TensorMap;

class गाठ_Context : सार्वजनिक frontend::गाठ_Context {
सार्वजनिक:
    गाठ_Context(स्थिरांक std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                पूर्णांक गाठ_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::गाठ_Context(decoder->get_op_type(गाठ_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_गाठ_idx(गाठ_idx),
        m_translate_session(translate_session) {
        m_आदान_names = decoder->get_आदान_names(m_गाठ_idx);
        m_output_names = decoder->get_output_names(m_गाठ_idx);
    }

    TranslateSession * get_translate_session() स्थिरांक { परत m_translate_session; }

    स्थिरांक std::vector<std::string> & get_आदान_names() स्थिरांक { परत m_आदान_names; }

    स्थिरांक std::vector<std::string> & get_output_names() स्थिरांक { परत m_output_names; }

    माप_प्रकार get_आदान_size() स्थिरांक override { परत m_decoder->get_आदान_size(m_गाठ_idx); }

    ov::element::Type get_आदान_type(माप_प्रकार index) स्थिरांक {
        परत m_decoder->get_आदान_type(m_गाठ_idx, m_आदान_names[index]);
    }

    PartialShape get_आदान_shape(माप_प्रकार आदान_index) स्थिरांक {
        परत m_decoder->get_आदान_shape(m_गाठ_idx, m_आदान_names[आदान_index]);
    }

    std::vector<माप_प्रकार> get_आदान_stride(माप_प्रकार index) स्थिरांक {
        परत m_decoder->get_आदान_stride(m_गाठ_idx, m_आदान_names[index]);
    }

    std::string get_output_name() स्थिरांक { परत m_output_names[0]; }

    PartialShape get_output_shape() स्थिरांक { परत m_decoder->get_output_shape(m_गाठ_idx); }

    int32_t * get_आदान_op_params(माप_प्रकार index) स्थिरांक {
        परत m_decoder->get_आदान_op_params(m_गाठ_idx, m_आदान_names[index]);
    }

    माप_प्रकार get_view_आदान_size(माप_प्रकार index) स्थिरांक {
        परत m_decoder->get_view_आदान_size(m_गाठ_idx, m_आदान_names[index]);
    }

    माप_प्रकार get_view_आदान_offset(माप_प्रकार index, माप_प्रकार view_index) स्थिरांक {
        परत m_decoder->get_view_आदान_offset(m_गाठ_idx, m_आदान_names[index], view_index);
    }

    माप_प्रकार get_view_आदान_src_offset(माप_प्रकार index, माप_प्रकार view_index) स्थिरांक {
        परत m_decoder->get_view_आदान_src_offset(m_गाठ_idx, m_आदान_names[index], view_index);
    }

    std::vector<माप_प्रकार> get_view_आदान_stride(माप_प्रकार index, माप_प्रकार view_index) स्थिरांक {
        परत m_decoder->get_view_आदान_stride(m_गाठ_idx, m_आदान_names[index], view_index);
    }

    std::vector<माप_प्रकार> get_view_आदान_src_stride(माप_प्रकार index, माप_प्रकार view_index) स्थिरांक {
        परत m_decoder->get_view_आदान_src_stride(m_गाठ_idx, m_आदान_names[index], view_index);
    }

    ov::Shape get_view_आदान_ggml_shape(माप_प्रकार index, माप_प्रकार view_index) स्थिरांक {
        परत m_decoder->get_view_आदान_ggml_shape(m_गाठ_idx, m_आदान_names[index], view_index);
    }

    ov::Shape get_view_आदान_src_ggml_shape(माप_प्रकार index, माप_प्रकार view_index) स्थिरांक {
        परत m_decoder->get_view_आदान_src_ggml_shape(m_गाठ_idx, m_आदान_names[index], view_index);
    }

    ov::PartialShape get_view_आदान_ov_shape(माप_प्रकार index, माप_प्रकार view_index) स्थिरांक {
        परत m_decoder->get_view_आदान_ov_shape(m_गाठ_idx, m_आदान_names[index], view_index);
    }

    ov::PartialShape get_view_आदान_src_ov_shape(माप_प्रकार index, माप_प्रकार view_index) स्थिरांक {
        परत m_decoder->get_view_आदान_src_ov_shape(m_गाठ_idx, m_आदान_names[index], view_index);
    }

    std::string get_view_आदान_name(माप_प्रकार index, माप_प्रकार view_index) स्थिरांक {
        परत m_decoder->get_view_आदान_name(m_गाठ_idx, m_आदान_names[index], view_index);
    }

    std::string get_view_आदान_src_name(माप_प्रकार index, माप_प्रकार view_index) स्थिरांक {
        परत m_decoder->get_view_आदान_src_name(m_गाठ_idx, m_आदान_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() स्थिरांक { परत m_decoder->get_op_dynamic_dim(m_गाठ_idx); }

    int32_t * get_output_op_params() स्थिरांक { परत m_decoder->get_output_op_params(m_गाठ_idx); }

    माप_प्रकार get_output_op_offset() स्थिरांक { परत m_decoder->get_output_op_offset(m_गाठ_idx); }

    ov::element::Type get_output_type() स्थिरांक { परत m_decoder->get_output_type(m_गाठ_idx); }

    std::vector<माप_प्रकार> get_output_stride() स्थिरांक { परत m_decoder->get_output_stride(m_गाठ_idx); }

    Output<गाठ_2> get_आदान(पूर्णांक idx) स्थिरांक override {
        // Check if this input is a VIEW
        माप_प्रकार view_आदान_size = m_decoder->get_view_आदान_size(m_गाठ_idx, m_आदान_names[idx]);
        जर (view_आदान_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_आदान_src_name(m_गाठ_idx, m_आदान_names[idx], view_आदान_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            स्वतः view_it = m_tensor_map->find(m_आदान_names[idx]);
            जर (!base_name.empty() && view_it != m_tensor_map->end()) {
                स्वतः base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                जर (base_it == m_tensor_map->end() ||
                    view_it->second.get_गाठ_shared_ptr() != base_it->second.get_गाठ_shared_ptr()) {
                    परत view_it->second;
                }
                परत base_it->second;
            }
            जर (!base_name.empty()) {
                परत m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        परत m_tensor_map->at(m_आदान_names[idx]);
    }

    Output<गाठ_2> get_आदान(स्थिरांक std::string & name) स्थिरांक override {
        जर (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        परत m_tensor_map->at(name);
    }

    bool has_आदान(स्थिरांक std::string & name) स्थिरांक { परत m_tensor_map->find(name) != m_tensor_map->end(); }

    शून्य put_shared(स्थिरांक std::string & name, स्थिरांक Output<गाठ_2> & मूल्य) स्थिरांक {
        m_tensor_map->घाला({name, मूल्य});
    }

    स्थिरांक std::string & get_name() स्थिरांक override { परत m_decoder->get_op_name(m_गाठ_idx); }

    ov::Any get_attribute_as_any(स्थिरांक std::string & name) स्थिरांक override { परत m_decoder->get_attribute(name); }

    पूर्णांक get_op_case() स्थिरांक { परत m_decoder->get_op_case(m_गाठ_idx); }

    bool is_static() स्थिरांक { परत m_decoder->is_static(); }

    bool is_stateful() स्थिरांक { परत m_decoder->is_stateful(); }

    पूर्णांक get_ssm_state_size() स्थिरांक { परत m_decoder->get_ssm_state_size(); }

खाजगी:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    पूर्णांक m_गाठ_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_आदान_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(स्थिरांक ov::frontend::ggml::गाठ_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
