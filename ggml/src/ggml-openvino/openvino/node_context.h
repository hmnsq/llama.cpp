#pragma once

#ضمن "decoder.h"

#ضمن <cstdint>
#ضمن <openvino/frontend/node_context.hpp>
#ضمن <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

تعريفنوع std::map<std::string, Output<گرێ_2>> TensorMap;

class گرێ_Context : عام frontend::گرێ_Context {
عام:
    گرێ_Context(ثابت std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                صحيح گرێ_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::گرێ_Context(decoder->get_op_type(گرێ_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_گرێ_idx(گرێ_idx),
        m_translate_session(translate_session) {
        m_تێخستن_names = decoder->get_تێخستن_names(m_گرێ_idx);
        m_output_names = decoder->get_output_names(m_گرێ_idx);
    }

    TranslateSession * get_translate_session() ثابت { أرجع m_translate_session; }

    ثابت std::vector<std::string> & get_تێخستن_names() ثابت { أرجع m_تێخستن_names; }

    ثابت std::vector<std::string> & get_output_names() ثابت { أرجع m_output_names; }

    size_t get_تێخستن_size() ثابت override { أرجع m_decoder->get_تێخستن_size(m_گرێ_idx); }

    ov::element::Type get_تێخستن_type(size_t index) ثابت {
        أرجع m_decoder->get_تێخستن_type(m_گرێ_idx, m_تێخستن_names[index]);
    }

    PartialShape get_تێخستن_shape(size_t تێخستن_index) ثابت {
        أرجع m_decoder->get_تێخستن_shape(m_گرێ_idx, m_تێخستن_names[تێخستن_index]);
    }

    std::vector<size_t> get_تێخستن_stride(size_t index) ثابت {
        أرجع m_decoder->get_تێخستن_stride(m_گرێ_idx, m_تێخستن_names[index]);
    }

    std::string get_output_name() ثابت { أرجع m_output_names[0]; }

    PartialShape get_output_shape() ثابت { أرجع m_decoder->get_output_shape(m_گرێ_idx); }

    int32_t * get_تێخستن_op_params(size_t index) ثابت {
        أرجع m_decoder->get_تێخستن_op_params(m_گرێ_idx, m_تێخستن_names[index]);
    }

    size_t get_view_تێخستن_size(size_t index) ثابت {
        أرجع m_decoder->get_view_تێخستن_size(m_گرێ_idx, m_تێخستن_names[index]);
    }

    size_t get_view_تێخستن_offset(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_تێخستن_offset(m_گرێ_idx, m_تێخستن_names[index], view_index);
    }

    size_t get_view_تێخستن_src_offset(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_تێخستن_src_offset(m_گرێ_idx, m_تێخستن_names[index], view_index);
    }

    std::vector<size_t> get_view_تێخستن_stride(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_تێخستن_stride(m_گرێ_idx, m_تێخستن_names[index], view_index);
    }

    std::vector<size_t> get_view_تێخستن_src_stride(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_تێخستن_src_stride(m_گرێ_idx, m_تێخستن_names[index], view_index);
    }

    ov::Shape get_view_تێخستن_ggml_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_تێخستن_ggml_shape(m_گرێ_idx, m_تێخستن_names[index], view_index);
    }

    ov::Shape get_view_تێخستن_src_ggml_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_تێخستن_src_ggml_shape(m_گرێ_idx, m_تێخستن_names[index], view_index);
    }

    ov::PartialShape get_view_تێخستن_ov_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_تێخستن_ov_shape(m_گرێ_idx, m_تێخستن_names[index], view_index);
    }

    ov::PartialShape get_view_تێخستن_src_ov_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_تێخستن_src_ov_shape(m_گرێ_idx, m_تێخستن_names[index], view_index);
    }

    std::string get_view_تێخستن_name(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_تێخستن_name(m_گرێ_idx, m_تێخستن_names[index], view_index);
    }

    std::string get_view_تێخستن_src_name(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_تێخستن_src_name(m_گرێ_idx, m_تێخستن_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ثابت { أرجع m_decoder->get_op_dynamic_dim(m_گرێ_idx); }

    int32_t * get_output_op_params() ثابت { أرجع m_decoder->get_output_op_params(m_گرێ_idx); }

    size_t get_output_op_offset() ثابت { أرجع m_decoder->get_output_op_offset(m_گرێ_idx); }

    ov::element::Type get_output_type() ثابت { أرجع m_decoder->get_output_type(m_گرێ_idx); }

    std::vector<size_t> get_output_stride() ثابت { أرجع m_decoder->get_output_stride(m_گرێ_idx); }

    Output<گرێ_2> get_تێخستن(صحيح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_تێخستن_size = m_decoder->get_view_تێخستن_size(m_گرێ_idx, m_تێخستن_names[idx]);
        إذا (view_تێخستن_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_تێخستن_src_name(m_گرێ_idx, m_تێخستن_names[idx], view_تێخستن_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_تێخستن_names[idx]);
            إذا (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                إذا (base_it == m_tensor_map->end() ||
                    view_it->second.get_گرێ_shared_ptr() != base_it->second.get_گرێ_shared_ptr()) {
                    أرجع view_it->second;
                }
                أرجع base_it->second;
            }
            إذا (!base_name.empty()) {
                أرجع m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        أرجع m_tensor_map->at(m_تێخستن_names[idx]);
    }

    Output<گرێ_2> get_تێخستن(ثابت std::string & name) ثابت override {
        إذا (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        أرجع m_tensor_map->at(name);
    }

    bool has_تێخستن(ثابت std::string & name) ثابت { أرجع m_tensor_map->find(name) != m_tensor_map->end(); }

    فراغ put_shared(ثابت std::string & name, ثابت Output<گرێ_2> & بەها) ثابت {
        m_tensor_map->زیادکردن({name, بەها});
    }

    ثابت std::string & get_name() ثابت override { أرجع m_decoder->get_op_name(m_گرێ_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { أرجع m_decoder->get_attribute(name); }

    صحيح get_op_case() ثابت { أرجع m_decoder->get_op_case(m_گرێ_idx); }

    bool is_static() ثابت { أرجع m_decoder->is_static(); }

    bool is_stateful() ثابت { أرجع m_decoder->is_stateful(); }

    صحيح get_ssm_state_size() ثابت { أرجع m_decoder->get_ssm_state_size(); }

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحيح m_گرێ_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_تێخستن_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ثابت ov::frontend::ggml::گرێ_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
