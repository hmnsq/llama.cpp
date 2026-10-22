#pragma once

#ضمن "decoder.h"

#ضمن <cstdint>
#ضمن <openvino/frontend/node_context.hpp>
#ضمن <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

تعريفنوع std::map<std::string, Output<گرہ_2>> TensorMap;

class گرہ_Context : عام frontend::گرہ_Context {
عام:
    گرہ_Context(ثابت std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                صحيح گرہ_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::گرہ_Context(decoder->get_op_type(گرہ_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_گرہ_idx(گرہ_idx),
        m_translate_session(translate_session) {
        m_داخلہ_names = decoder->get_داخلہ_names(m_گرہ_idx);
        m_output_names = decoder->get_output_names(m_گرہ_idx);
    }

    TranslateSession * get_translate_session() ثابت { أرجع m_translate_session; }

    ثابت std::vector<std::string> & get_داخلہ_names() ثابت { أرجع m_داخلہ_names; }

    ثابت std::vector<std::string> & get_output_names() ثابت { أرجع m_output_names; }

    size_t get_داخلہ_size() ثابت override { أرجع m_decoder->get_داخلہ_size(m_گرہ_idx); }

    ov::element::Type get_داخلہ_type(size_t index) ثابت {
        أرجع m_decoder->get_داخلہ_type(m_گرہ_idx, m_داخلہ_names[index]);
    }

    PartialShape get_داخلہ_shape(size_t داخلہ_index) ثابت {
        أرجع m_decoder->get_داخلہ_shape(m_گرہ_idx, m_داخلہ_names[داخلہ_index]);
    }

    std::vector<size_t> get_داخلہ_stride(size_t index) ثابت {
        أرجع m_decoder->get_داخلہ_stride(m_گرہ_idx, m_داخلہ_names[index]);
    }

    std::string get_output_name() ثابت { أرجع m_output_names[0]; }

    PartialShape get_output_shape() ثابت { أرجع m_decoder->get_output_shape(m_گرہ_idx); }

    int32_t * get_داخلہ_op_params(size_t index) ثابت {
        أرجع m_decoder->get_داخلہ_op_params(m_گرہ_idx, m_داخلہ_names[index]);
    }

    size_t get_view_داخلہ_size(size_t index) ثابت {
        أرجع m_decoder->get_view_داخلہ_size(m_گرہ_idx, m_داخلہ_names[index]);
    }

    size_t get_view_داخلہ_offset(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_داخلہ_offset(m_گرہ_idx, m_داخلہ_names[index], view_index);
    }

    size_t get_view_داخلہ_src_offset(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_داخلہ_src_offset(m_گرہ_idx, m_داخلہ_names[index], view_index);
    }

    std::vector<size_t> get_view_داخلہ_stride(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_داخلہ_stride(m_گرہ_idx, m_داخلہ_names[index], view_index);
    }

    std::vector<size_t> get_view_داخلہ_src_stride(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_داخلہ_src_stride(m_گرہ_idx, m_داخلہ_names[index], view_index);
    }

    ov::Shape get_view_داخلہ_ggml_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_داخلہ_ggml_shape(m_گرہ_idx, m_داخلہ_names[index], view_index);
    }

    ov::Shape get_view_داخلہ_src_ggml_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_داخلہ_src_ggml_shape(m_گرہ_idx, m_داخلہ_names[index], view_index);
    }

    ov::PartialShape get_view_داخلہ_ov_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_داخلہ_ov_shape(m_گرہ_idx, m_داخلہ_names[index], view_index);
    }

    ov::PartialShape get_view_داخلہ_src_ov_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_داخلہ_src_ov_shape(m_گرہ_idx, m_داخلہ_names[index], view_index);
    }

    std::string get_view_داخلہ_name(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_داخلہ_name(m_گرہ_idx, m_داخلہ_names[index], view_index);
    }

    std::string get_view_داخلہ_src_name(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_داخلہ_src_name(m_گرہ_idx, m_داخلہ_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ثابت { أرجع m_decoder->get_op_dynamic_dim(m_گرہ_idx); }

    int32_t * get_output_op_params() ثابت { أرجع m_decoder->get_output_op_params(m_گرہ_idx); }

    size_t get_output_op_offset() ثابت { أرجع m_decoder->get_output_op_offset(m_گرہ_idx); }

    ov::element::Type get_output_type() ثابت { أرجع m_decoder->get_output_type(m_گرہ_idx); }

    std::vector<size_t> get_output_stride() ثابت { أرجع m_decoder->get_output_stride(m_گرہ_idx); }

    Output<گرہ_2> get_داخلہ(صحيح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_داخلہ_size = m_decoder->get_view_داخلہ_size(m_گرہ_idx, m_داخلہ_names[idx]);
        إذا (view_داخلہ_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_داخلہ_src_name(m_گرہ_idx, m_داخلہ_names[idx], view_داخلہ_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_داخلہ_names[idx]);
            إذا (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                إذا (base_it == m_tensor_map->end() ||
                    view_it->second.get_گرہ_shared_ptr() != base_it->second.get_گرہ_shared_ptr()) {
                    أرجع view_it->second;
                }
                أرجع base_it->second;
            }
            إذا (!base_name.empty()) {
                أرجع m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        أرجع m_tensor_map->at(m_داخلہ_names[idx]);
    }

    Output<گرہ_2> get_داخلہ(ثابت std::string & name) ثابت override {
        إذا (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        أرجع m_tensor_map->at(name);
    }

    bool has_داخلہ(ثابت std::string & name) ثابت { أرجع m_tensor_map->find(name) != m_tensor_map->end(); }

    فراغ put_shared(ثابت std::string & name, ثابت Output<گرہ_2> & قدر) ثابت {
        m_tensor_map->داخل_کرو({name, قدر});
    }

    ثابت std::string & get_name() ثابت override { أرجع m_decoder->get_op_name(m_گرہ_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { أرجع m_decoder->get_attribute(name); }

    صحيح get_op_case() ثابت { أرجع m_decoder->get_op_case(m_گرہ_idx); }

    bool is_static() ثابت { أرجع m_decoder->is_static(); }

    bool is_stateful() ثابت { أرجع m_decoder->is_stateful(); }

    صحيح get_ssm_state_size() ثابت { أرجع m_decoder->get_ssm_state_size(); }

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحيح m_گرہ_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_داخلہ_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ثابت ov::frontend::ggml::گرہ_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
