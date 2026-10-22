#pragma once

#ضمن "decoder.h"

#ضمن <cstdint>
#ضمن <openvino/frontend/node_context.hpp>
#ضمن <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

تعريفنوع std::map<std::string, Output<عقدة_2>> TensorMap;

class عقدة_Context : عام frontend::عقدة_Context {
عام:
    عقدة_Context(ثابت std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                صحيح عقدة_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::عقدة_Context(decoder->get_op_type(عقدة_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_عقدة_idx(عقدة_idx),
        m_translate_session(translate_session) {
        m_مدخل_names = decoder->get_مدخل_names(m_عقدة_idx);
        m_output_names = decoder->get_output_names(m_عقدة_idx);
    }

    TranslateSession * get_translate_session() ثابت { أرجع m_translate_session; }

    ثابت std::vector<std::string> & get_مدخل_names() ثابت { أرجع m_مدخل_names; }

    ثابت std::vector<std::string> & get_output_names() ثابت { أرجع m_output_names; }

    size_t get_مدخل_size() ثابت override { أرجع m_decoder->get_مدخل_size(m_عقدة_idx); }

    ov::element::Type get_مدخل_type(size_t index) ثابت {
        أرجع m_decoder->get_مدخل_type(m_عقدة_idx, m_مدخل_names[index]);
    }

    PartialShape get_مدخل_shape(size_t مدخل_index) ثابت {
        أرجع m_decoder->get_مدخل_shape(m_عقدة_idx, m_مدخل_names[مدخل_index]);
    }

    std::vector<size_t> get_مدخل_stride(size_t index) ثابت {
        أرجع m_decoder->get_مدخل_stride(m_عقدة_idx, m_مدخل_names[index]);
    }

    std::string get_output_name() ثابت { أرجع m_output_names[0]; }

    PartialShape get_output_shape() ثابت { أرجع m_decoder->get_output_shape(m_عقدة_idx); }

    int32_t * get_مدخل_op_params(size_t index) ثابت {
        أرجع m_decoder->get_مدخل_op_params(m_عقدة_idx, m_مدخل_names[index]);
    }

    size_t get_view_مدخل_size(size_t index) ثابت {
        أرجع m_decoder->get_view_مدخل_size(m_عقدة_idx, m_مدخل_names[index]);
    }

    size_t get_view_مدخل_offset(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_مدخل_offset(m_عقدة_idx, m_مدخل_names[index], view_index);
    }

    size_t get_view_مدخل_src_offset(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_مدخل_src_offset(m_عقدة_idx, m_مدخل_names[index], view_index);
    }

    std::vector<size_t> get_view_مدخل_stride(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_مدخل_stride(m_عقدة_idx, m_مدخل_names[index], view_index);
    }

    std::vector<size_t> get_view_مدخل_src_stride(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_مدخل_src_stride(m_عقدة_idx, m_مدخل_names[index], view_index);
    }

    ov::Shape get_view_مدخل_ggml_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_مدخل_ggml_shape(m_عقدة_idx, m_مدخل_names[index], view_index);
    }

    ov::Shape get_view_مدخل_src_ggml_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_مدخل_src_ggml_shape(m_عقدة_idx, m_مدخل_names[index], view_index);
    }

    ov::PartialShape get_view_مدخل_ov_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_مدخل_ov_shape(m_عقدة_idx, m_مدخل_names[index], view_index);
    }

    ov::PartialShape get_view_مدخل_src_ov_shape(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_مدخل_src_ov_shape(m_عقدة_idx, m_مدخل_names[index], view_index);
    }

    std::string get_view_مدخل_name(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_مدخل_name(m_عقدة_idx, m_مدخل_names[index], view_index);
    }

    std::string get_view_مدخل_src_name(size_t index, size_t view_index) ثابت {
        أرجع m_decoder->get_view_مدخل_src_name(m_عقدة_idx, m_مدخل_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ثابت { أرجع m_decoder->get_op_dynamic_dim(m_عقدة_idx); }

    int32_t * get_output_op_params() ثابت { أرجع m_decoder->get_output_op_params(m_عقدة_idx); }

    size_t get_output_op_offset() ثابت { أرجع m_decoder->get_output_op_offset(m_عقدة_idx); }

    ov::element::Type get_output_type() ثابت { أرجع m_decoder->get_output_type(m_عقدة_idx); }

    std::vector<size_t> get_output_stride() ثابت { أرجع m_decoder->get_output_stride(m_عقدة_idx); }

    Output<عقدة_2> get_مدخل(صحيح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_مدخل_size = m_decoder->get_view_مدخل_size(m_عقدة_idx, m_مدخل_names[idx]);
        إذا (view_مدخل_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_مدخل_src_name(m_عقدة_idx, m_مدخل_names[idx], view_مدخل_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_مدخل_names[idx]);
            إذا (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                إذا (base_it == m_tensor_map->end() ||
                    view_it->second.get_عقدة_shared_ptr() != base_it->second.get_عقدة_shared_ptr()) {
                    أرجع view_it->second;
                }
                أرجع base_it->second;
            }
            إذا (!base_name.empty()) {
                أرجع m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        أرجع m_tensor_map->at(m_مدخل_names[idx]);
    }

    Output<عقدة_2> get_مدخل(ثابت std::string & name) ثابت override {
        إذا (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        أرجع m_tensor_map->at(name);
    }

    bool has_مدخل(ثابت std::string & name) ثابت { أرجع m_tensor_map->find(name) != m_tensor_map->end(); }

    فراغ put_shared(ثابت std::string & name, ثابت Output<عقدة_2> & قيمة) ثابت {
        m_tensor_map->أدرج({name, قيمة});
    }

    ثابت std::string & get_name() ثابت override { أرجع m_decoder->get_op_name(m_عقدة_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { أرجع m_decoder->get_attribute(name); }

    صحيح get_op_case() ثابت { أرجع m_decoder->get_op_case(m_عقدة_idx); }

    bool is_static() ثابت { أرجع m_decoder->is_static(); }

    bool is_stateful() ثابت { أرجع m_decoder->is_stateful(); }

    صحيح get_ssm_state_size() ثابت { أرجع m_decoder->get_ssm_state_size(); }

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحيح m_عقدة_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_مدخل_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ثابت ov::frontend::ggml::عقدة_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
