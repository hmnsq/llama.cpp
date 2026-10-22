#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "decoder.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/node_context.hpp>
#ಅನ್ತರ್ಭಾವ <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ std::map<std::string, Output<ಗಂಟು_2>> TensorMap;

class ಗಂಟು_Context : ಸಾರ್ವಜನಿಕ frontend::ಗಂಟು_Context {
ಸಾರ್ವಜನಿಕ:
    ಗಂಟು_Context(ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::ಗಂಟು_Context(decoder->get_op_type(ಗಂಟು_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_ಗಂಟು_idx(ಗಂಟು_idx),
        m_translate_session(translate_session) {
        m_ಒಳಹರಿವು_ಹೆಸರುಗಳು = decoder->get_ಒಳಹರಿವು_ಹೆಸರುಗಳು(m_ಗಂಟು_idx);
        m_output_ಹೆಸರುಗಳು = decoder->get_output_ಹೆಸರುಗಳು(m_ಗಂಟು_idx);
    }

    TranslateSession * get_translate_session() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_translate_session; }

    ಸ್ಥಿರಾಂಕ std::ಸದಿಶ<std::string> & get_ಒಳಹರಿವು_ಹೆಸರುಗಳು() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_ಒಳಹರಿವು_ಹೆಸರುಗಳು; }

    ಸ್ಥಿರಾಂಕ std::ಸದಿಶ<std::string> & get_output_ಹೆಸರುಗಳು() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_output_ಹೆಸರುಗಳು; }

    size_t get_ಒಳಹರಿವು_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ m_decoder->get_ಒಳಹರಿವು_ಗಾತ್ರ(m_ಗಂಟು_idx); }

    ov::element::Type get_ಒಳಹರಿವು_type(size_t ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_ಒಳಹರಿವು_type(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ]);
    }

    PartialShape get_ಒಳಹರಿವು_shape(size_t ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_ಒಳಹರಿವು_shape(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ]);
    }

    std::ಸದಿಶ<size_t> get_ಒಳಹರಿವು_stride(size_t ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_ಒಳಹರಿವು_stride(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ]);
    }

    std::string get_output_name() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_output_ಹೆಸರುಗಳು[0]; }

    PartialShape get_output_shape() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_decoder->get_output_shape(m_ಗಂಟು_idx); }

    int32_t * get_ಒಳಹರಿವು_op_params(size_t ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_ಒಳಹರಿವು_op_params(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ]);
    }

    size_t get_view_ಒಳಹರಿವು_ಗಾತ್ರ(size_t ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_ಗಾತ್ರ(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ]);
    }

    size_t get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(size_t ಸೂಚ್ಯಂಕ, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ], view_ಸೂಚ್ಯಂಕ);
    }

    size_t get_view_ಒಳಹರಿವು_src_ಸ್ಥಾನಾಂತರ(size_t ಸೂಚ್ಯಂಕ, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_src_ಸ್ಥಾನಾಂತರ(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ], view_ಸೂಚ್ಯಂಕ);
    }

    std::ಸದಿಶ<size_t> get_view_ಒಳಹರಿವು_stride(size_t ಸೂಚ್ಯಂಕ, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_stride(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ], view_ಸೂಚ್ಯಂಕ);
    }

    std::ಸದಿಶ<size_t> get_view_ಒಳಹರಿವು_src_stride(size_t ಸೂಚ್ಯಂಕ, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_src_stride(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ], view_ಸೂಚ್ಯಂಕ);
    }

    ov::Shape get_view_ಒಳಹರಿವು_ggml_shape(size_t ಸೂಚ್ಯಂಕ, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_ggml_shape(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ], view_ಸೂಚ್ಯಂಕ);
    }

    ov::Shape get_view_ಒಳಹರಿವು_src_ggml_shape(size_t ಸೂಚ್ಯಂಕ, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_src_ggml_shape(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ], view_ಸೂಚ್ಯಂಕ);
    }

    ov::PartialShape get_view_ಒಳಹರಿವು_ov_shape(size_t ಸೂಚ್ಯಂಕ, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_ov_shape(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ], view_ಸೂಚ್ಯಂಕ);
    }

    ov::PartialShape get_view_ಒಳಹರಿವು_src_ov_shape(size_t ಸೂಚ್ಯಂಕ, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_src_ov_shape(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ], view_ಸೂಚ್ಯಂಕ);
    }

    std::string get_view_ಒಳಹರಿವು_name(size_t ಸೂಚ್ಯಂಕ, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_name(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ], view_ಸೂಚ್ಯಂಕ);
    }

    std::string get_view_ಒಳಹರಿವು_src_name(size_t ಸೂಚ್ಯಂಕ, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_decoder->get_view_ಒಳಹರಿವು_src_name(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[ಸೂಚ್ಯಂಕ], view_ಸೂಚ್ಯಂಕ);
    }

    int32_t get_op_dynamic_dim() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_decoder->get_op_dynamic_dim(m_ಗಂಟು_idx); }

    int32_t * get_output_op_params() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_decoder->get_output_op_params(m_ಗಂಟು_idx); }

    size_t get_output_op_ಸ್ಥಾನಾಂತರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_decoder->get_output_op_ಸ್ಥಾನಾಂತರ(m_ಗಂಟು_idx); }

    ov::element::Type get_output_type() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_decoder->get_output_type(m_ಗಂಟು_idx); }

    std::ಸದಿಶ<size_t> get_output_stride() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_decoder->get_output_stride(m_ಗಂಟು_idx); }

    Output<ಗಂಟು_2> get_ಒಳಹರಿವು(ಪೂರ್ಣಾಂಕ idx) ಸ್ಥಿರಾಂಕ override {
        // Check if this input is a VIEW
        size_t view_ಒಳಹರಿವು_ಗಾತ್ರ = m_decoder->get_view_ಒಳಹರಿವು_ಗಾತ್ರ(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[idx]);
        ಆದರೆ (view_ಒಳಹರಿವು_ಗಾತ್ರ > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string ತಳ_name =
                m_decoder->get_view_ಒಳಹರಿವು_src_name(m_ಗಂಟು_idx, m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[idx], view_ಒಳಹರಿವು_ಗಾತ್ರ - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            ಸ್ವಯಂಚಾಲಿತ view_it = m_tensor_map->ಕಂಡುಹಿಡಿ(m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[idx]);
            ಆದರೆ (!ತಳ_name.ಖಾಲಿಗಳು() && view_it != m_tensor_map->ಕೊನೆ()) {
                ಸ್ವಯಂಚಾಲಿತ ತಳ_it = m_tensor_map->ಕಂಡುಹಿಡಿ(ತಳ_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                ಆದರೆ (ತಳ_it == m_tensor_map->ಕೊನೆ() ||
                    view_it->ಎರಡನೇ.get_ಗಂಟು_shared_ptr() != ತಳ_it->ಎರಡನೇ.get_ಗಂಟು_shared_ptr()) {
                    ಹಿಂತಿರುಗಿ view_it->ಎರಡನೇ;
                }
                ಹಿಂತಿರುಗಿ ತಳ_it->ಎರಡನೇ;
            }
            ಆದರೆ (!ತಳ_name.ಖಾಲಿಗಳು()) {
                ಹಿಂತಿರುಗಿ m_tensor_map->at(ತಳ_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        ಹಿಂತಿರುಗಿ m_tensor_map->at(m_ಒಳಹರಿವು_ಹೆಸರುಗಳು[idx]);
    }

    Output<ಗಂಟು_2> get_ಒಳಹರಿವು(ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ override {
        ಆದರೆ (m_tensor_map->ಕಂಡುಹಿಡಿ(name) == m_tensor_map->ಕೊನೆ()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        ಹಿಂತಿರುಗಿ m_tensor_map->at(name);
    }

    bool has_ಒಳಹರಿವು(ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_tensor_map->ಕಂಡುಹಿಡಿ(name) != m_tensor_map->ಕೊನೆ(); }

    ಶೂನ್ಯ put_shared(ಸ್ಥಿರಾಂಕ std::string & name, ಸ್ಥಿರಾಂಕ Output<ಗಂಟು_2> & ಮೌಲ್ಯ) ಸ್ಥಿರಾಂಕ {
        m_tensor_map->ಸೇರಿಸು({name, ಮೌಲ್ಯ});
    }

    ಸ್ಥಿರಾಂಕ std::string & get_name() ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ m_decoder->get_op_name(m_ಗಂಟು_idx); }

    ov::Any get_attribute_as_any(ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ m_decoder->get_attribute(name); }

    ಪೂರ್ಣಾಂಕ get_op_case() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_decoder->get_op_case(m_ಗಂಟು_idx); }

    bool is_static() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_decoder->is_static(); }

    bool is_stateful() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_decoder->is_stateful(); }

    ಪೂರ್ಣಾಂಕ get_ssm_state_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_decoder->get_ssm_state_ಗಾತ್ರ(); }

ಖಾಸಗಿ:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    ಪೂರ್ಣಾಂಕ m_ಗಂಟು_idx;
    TranslateSession * m_translate_session;
    std::ಸದಿಶ<std::string> m_ಒಳಹರಿವು_ಹೆಸರುಗಳು;
    std::ಸದಿಶ<std::string> m_output_ಹೆಸರುಗಳು;
};

using CreatorFunction = std::function<ov::Output_ಸದಿಶ(ಸ್ಥಿರಾಂಕ ov::frontend::ggml::ಗಂಟು_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
