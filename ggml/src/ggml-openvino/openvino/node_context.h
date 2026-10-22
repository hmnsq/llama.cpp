#pragma once

#ضمن "decoder.h"

#ضمن <cstdint>
#ضمن <openvino/frontend/node_context.hpp>
#ضمن <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

تعريفنوع std::map<std::string, Output<تۈگۈن_2>> TensorMap;

class تۈگۈن_Context : عام frontend::تۈگۈن_Context {
عام:
    تۈگۈن_Context(ثابت std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                صحيح تۈگۈن_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::تۈگۈن_Context(decoder->get_op_type(تۈگۈن_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_تۈگۈن_idx(تۈگۈن_idx),
        m_translate_session(translate_session) {
        m_كىرگۈزۈش_names = decoder->get_كىرگۈزۈش_names(m_تۈگۈن_idx);
        m_output_names = decoder->get_output_names(m_تۈگۈن_idx);
    }

    TranslateSession * get_translate_session() ثابت { أرجع m_translate_session; }

    ثابت std::vector<std::string> & get_كىرگۈزۈش_names() ثابت { أرجع m_كىرگۈزۈش_names; }

    ثابت std::vector<std::string> & get_output_names() ثابت { أرجع m_output_names; }

    size_t get_كىرگۈزۈش_size() ثابت override { أرجع m_decoder->get_كىرگۈزۈش_size(m_تۈگۈن_idx); }

    ov::element::Type get_كىرگۈزۈش_type(size_t index) ثابت {
        أرجع m_decoder->get_كىرگۈزۈش_type(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index]);
    }

    PartialShape get_كىرگۈزۈش_shape(size_t كىرگۈزۈش_index) ثابت {
        أرجع m_decoder->get_كىرگۈزۈش_shape(m_تۈگۈن_idx, m_كىرگۈزۈش_names[كىرگۈزۈش_index]);
    }

    std::vector<size_t> get_كىرگۈزۈش_stride(size_t index) ثابت {
        أرجع m_decoder->get_كىرگۈزۈش_stride(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index]);
    }

    std::string get_output_name() ثابت { أرجع m_output_names[0]; }

    PartialShape get_output_shape() ثابت { أرجع m_decoder->get_output_shape(m_تۈگۈن_idx); }

    int32_t * get_كىرگۈزۈش_op_params(size_t index) ثابت {
        أرجع m_decoder->get_كىرگۈزۈش_op_params(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index]);
    }

    size_t get_view_كىرگۈزۈش_size(size_t index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_size(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index]);
    }

    size_t get_view_كىرگۈزۈش_offset(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_offset(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index], view_index);
    }

    size_t get_view_كىرگۈزۈش_src_offset(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_src_offset(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index], view_index);
    }

    std::vector<size_t> get_view_كىرگۈزۈش_stride(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_stride(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index], view_index);
    }

    std::vector<size_t> get_view_كىرگۈزۈش_src_stride(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_src_stride(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index], view_index);
    }

    ov::Shape get_view_كىرگۈزۈش_ggml_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_ggml_shape(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index], view_index);
    }

    ov::Shape get_view_كىرگۈزۈش_src_ggml_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_src_ggml_shape(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index], view_index);
    }

    ov::PartialShape get_view_كىرگۈزۈش_ov_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_ov_shape(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index], view_index);
    }

    ov::PartialShape get_view_كىرگۈزۈش_src_ov_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_src_ov_shape(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index], view_index);
    }

    std::string get_view_كىرگۈزۈش_name(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_name(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index], view_index);
    }

    std::string get_view_كىرگۈزۈش_src_name(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_كىرگۈزۈش_src_name(m_تۈگۈن_idx, m_كىرگۈزۈش_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ثابت { أرجع m_decoder->get_op_dynamic_dim(m_تۈگۈن_idx); }

    int32_t * get_output_op_params() ثابت { أرجع m_decoder->get_output_op_params(m_تۈگۈن_idx); }

    size_t get_output_op_offset() ثابت { أرجع m_decoder->get_output_op_offset(m_تۈگۈن_idx); }

    ov::element::Type get_output_type() ثابت { أرجع m_decoder->get_output_type(m_تۈگۈن_idx); }

    std::vector<size_t> get_output_stride() ثابت { أرجع m_decoder->get_output_stride(m_تۈگۈن_idx); }

    Output<تۈگۈن_2> get_كىرگۈزۈش(صحيح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_كىرگۈزۈش_size = m_decoder->get_view_كىرگۈزۈش_size(m_تۈگۈن_idx, m_كىرگۈزۈش_names[idx]);
        إذا (view_كىرگۈزۈش_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_كىرگۈزۈش_src_name(m_تۈگۈن_idx, m_كىرگۈزۈش_names[idx], view_كىرگۈزۈش_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_كىرگۈزۈش_names[idx]);
            إذا (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                إذا (base_it == m_tensor_map->end() ||
                    view_it->second.get_تۈگۈن_shared_ptr() != base_it->second.get_تۈگۈن_shared_ptr()) {
                    أرجع view_it->second;
                }
                أرجع base_it->second;
            }
            إذا (!base_name.empty()) {
                أرجع m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        أرجع m_tensor_map->at(m_كىرگۈزۈش_names[idx]);
    }

    Output<تۈگۈن_2> get_كىرگۈزۈش(ثابت std::string & name) ثابت override {
        إذا (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        أرجع m_tensor_map->at(name);
    }

    bool has_كىرگۈزۈش(ثابت std::string & name) ثابت { أرجع m_tensor_map->find(name) != m_tensor_map->end(); }

    فراغ put_shared(ثابت std::string & name, ثابت Output<تۈگۈن_2> & قىممەت) ثابت {
        m_tensor_map->قىستۇرۇش({name, قىممەت});
    }

    ثابت std::string & get_name() ثابت override { أرجع m_decoder->get_op_name(m_تۈگۈن_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { أرجع m_decoder->get_attribute(name); }

    صحيح get_op_case() ثابت { أرجع m_decoder->get_op_case(m_تۈگۈن_idx); }

    bool is_static() ثابت { أرجع m_decoder->is_static(); }

    bool is_stateful() ثابت { أرجع m_decoder->is_stateful(); }

    صحيح get_ssm_state_size() ثابت { أرجع m_decoder->get_ssm_state_size(); }

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحيح m_تۈگۈن_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_كىرگۈزۈش_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ثابت ov::frontend::ggml::تۈگۈن_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
