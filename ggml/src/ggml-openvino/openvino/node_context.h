#pragma once

#অন্তর্ভাব "decoder.h"

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <openvino/frontend/node_context.hpp>
#অন্তর্ভাব <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

প্রকারনির্দেশ std::map<std::string, Output<গ্রন্থি_2>> TensorMap;

class গ্রন্থি_Context : সার্বজনিক frontend::গ্রন্থি_Context {
সার্বজনিক:
    গ্রন্থি_Context(ধ্রুব std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                পূর্ণাঙ্ক গ্রন্থি_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::গ্রন্থি_Context(decoder->get_op_type(গ্রন্থি_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_গ্রন্থি_idx(গ্রন্থি_idx),
        m_translate_session(translate_session) {
        m_প্রবেশ_names = decoder->get_প্রবেশ_names(m_গ্রন্থি_idx);
        m_output_names = decoder->get_output_names(m_গ্রন্থি_idx);
    }

    TranslateSession * get_translate_session() ধ্রুব { প্রত্যাগম m_translate_session; }

    ধ্রুব std::vector<std::string> & get_প্রবেশ_names() ধ্রুব { প্রত্যাগম m_প্রবেশ_names; }

    ধ্রুব std::vector<std::string> & get_output_names() ধ্রুব { প্রত্যাগম m_output_names; }

    size_t get_প্রবেশ_size() ধ্রুব override { প্রত্যাগম m_decoder->get_প্রবেশ_size(m_গ্রন্থি_idx); }

    ov::element::Type get_প্রবেশ_type(size_t index) ধ্রুব {
        প্রত্যাগম m_decoder->get_প্রবেশ_type(m_গ্রন্থি_idx, m_প্রবেশ_names[index]);
    }

    PartialShape get_প্রবেশ_shape(size_t প্রবেশ_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_প্রবেশ_shape(m_গ্রন্থি_idx, m_প্রবেশ_names[প্রবেশ_index]);
    }

    std::vector<size_t> get_প্রবেশ_stride(size_t index) ধ্রুব {
        প্রত্যাগম m_decoder->get_প্রবেশ_stride(m_গ্রন্থি_idx, m_প্রবেশ_names[index]);
    }

    std::string get_output_name() ধ্রুব { প্রত্যাগম m_output_names[0]; }

    PartialShape get_output_shape() ধ্রুব { প্রত্যাগম m_decoder->get_output_shape(m_গ্রন্থি_idx); }

    int32_t * get_প্রবেশ_op_params(size_t index) ধ্রুব {
        প্রত্যাগম m_decoder->get_প্রবেশ_op_params(m_গ্রন্থি_idx, m_প্রবেশ_names[index]);
    }

    size_t get_view_প্রবেশ_size(size_t index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_size(m_গ্রন্থি_idx, m_প্রবেশ_names[index]);
    }

    size_t get_view_প্রবেশ_offset(size_t index, size_t view_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_offset(m_গ্রন্থি_idx, m_প্রবেশ_names[index], view_index);
    }

    size_t get_view_প্রবেশ_src_offset(size_t index, size_t view_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_src_offset(m_গ্রন্থি_idx, m_প্রবেশ_names[index], view_index);
    }

    std::vector<size_t> get_view_প্রবেশ_stride(size_t index, size_t view_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_stride(m_গ্রন্থি_idx, m_প্রবেশ_names[index], view_index);
    }

    std::vector<size_t> get_view_প্রবেশ_src_stride(size_t index, size_t view_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_src_stride(m_গ্রন্থি_idx, m_প্রবেশ_names[index], view_index);
    }

    ov::Shape get_view_প্রবেশ_ggml_shape(size_t index, size_t view_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_ggml_shape(m_গ্রন্থি_idx, m_প্রবেশ_names[index], view_index);
    }

    ov::Shape get_view_প্রবেশ_src_ggml_shape(size_t index, size_t view_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_src_ggml_shape(m_গ্রন্থি_idx, m_প্রবেশ_names[index], view_index);
    }

    ov::PartialShape get_view_প্রবেশ_ov_shape(size_t index, size_t view_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_ov_shape(m_গ্রন্থি_idx, m_প্রবেশ_names[index], view_index);
    }

    ov::PartialShape get_view_প্রবেশ_src_ov_shape(size_t index, size_t view_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_src_ov_shape(m_গ্রন্থি_idx, m_প্রবেশ_names[index], view_index);
    }

    std::string get_view_প্রবেশ_name(size_t index, size_t view_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_name(m_গ্রন্থি_idx, m_প্রবেশ_names[index], view_index);
    }

    std::string get_view_প্রবেশ_src_name(size_t index, size_t view_index) ধ্রুব {
        প্রত্যাগম m_decoder->get_view_প্রবেশ_src_name(m_গ্রন্থি_idx, m_প্রবেশ_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ধ্রুব { প্রত্যাগম m_decoder->get_op_dynamic_dim(m_গ্রন্থি_idx); }

    int32_t * get_output_op_params() ধ্রুব { প্রত্যাগম m_decoder->get_output_op_params(m_গ্রন্থি_idx); }

    size_t get_output_op_offset() ধ্রুব { প্রত্যাগম m_decoder->get_output_op_offset(m_গ্রন্থি_idx); }

    ov::element::Type get_output_type() ধ্রুব { প্রত্যাগম m_decoder->get_output_type(m_গ্রন্থি_idx); }

    std::vector<size_t> get_output_stride() ধ্রুব { প্রত্যাগম m_decoder->get_output_stride(m_গ্রন্থি_idx); }

    Output<গ্রন্থি_2> get_প্রবেশ(পূর্ণাঙ্ক idx) ধ্রুব override {
        // Check if this input is a VIEW
        size_t view_প্রবেশ_size = m_decoder->get_view_প্রবেশ_size(m_গ্রন্থি_idx, m_প্রবেশ_names[idx]);
        যদি (view_প্রবেশ_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_প্রবেশ_src_name(m_গ্রন্থি_idx, m_প্রবেশ_names[idx], view_প্রবেশ_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_প্রবেশ_names[idx]);
            যদি (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                যদি (base_it == m_tensor_map->end() ||
                    view_it->second.get_গ্রন্থি_shared_ptr() != base_it->second.get_গ্রন্থি_shared_ptr()) {
                    প্রত্যাগম view_it->second;
                }
                প্রত্যাগম base_it->second;
            }
            যদি (!base_name.empty()) {
                প্রত্যাগম m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        প্রত্যাগম m_tensor_map->at(m_প্রবেশ_names[idx]);
    }

    Output<গ্রন্থি_2> get_প্রবেশ(ধ্রুব std::string & name) ধ্রুব override {
        যদি (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        প্রত্যাগম m_tensor_map->at(name);
    }

    bool has_প্রবেশ(ধ্রুব std::string & name) ধ্রুব { প্রত্যাগম m_tensor_map->find(name) != m_tensor_map->end(); }

    শূন্য put_shared(ধ্রুব std::string & name, ধ্রুব Output<গ্রন্থি_2> & মান) ধ্রুব {
        m_tensor_map->ঢোকাও({name, মান});
    }

    ধ্রুব std::string & get_name() ধ্রুব override { প্রত্যাগম m_decoder->get_op_name(m_গ্রন্থি_idx); }

    ov::Any get_attribute_as_any(ধ্রুব std::string & name) ধ্রুব override { প্রত্যাগম m_decoder->get_attribute(name); }

    পূর্ণাঙ্ক get_op_case() ধ্রুব { প্রত্যাগম m_decoder->get_op_case(m_গ্রন্থি_idx); }

    bool is_static() ধ্রুব { প্রত্যাগম m_decoder->is_static(); }

    bool is_stateful() ধ্রুব { প্রত্যাগম m_decoder->is_stateful(); }

    পূর্ণাঙ্ক get_ssm_state_size() ধ্রুব { প্রত্যাগম m_decoder->get_ssm_state_size(); }

গুপ্ত:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    পূর্ণাঙ্ক m_গ্রন্থি_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_প্রবেশ_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ধ্রুব ov::frontend::ggml::গ্রন্থি_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
