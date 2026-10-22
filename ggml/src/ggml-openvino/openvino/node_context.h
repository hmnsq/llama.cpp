#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "decoder.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/node_context.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::map<std::string, Output<ਗੰਢ_2>> TensorMap;

class ਗੰਢ_Context : ਜਨਤਕ frontend::ਗੰਢ_Context {
ਜਨਤਕ:
    ਗੰਢ_Context(ਅਚਲ std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                ਪੂਰਨ ਗੰਢ_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::ਗੰਢ_Context(decoder->get_op_type(ਗੰਢ_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_ਗੰਢ_idx(ਗੰਢ_idx),
        m_translate_session(translate_session) {
        m_ਨਿਵੇਸ਼_ਨਾਂ = decoder->get_ਨਿਵੇਸ਼_ਨਾਂ(m_ਗੰਢ_idx);
        m_output_ਨਾਂ = decoder->get_output_ਨਾਂ(m_ਗੰਢ_idx);
    }

    TranslateSession * get_translate_session() ਅਚਲ { ਵਾਪਸ m_translate_session; }

    ਅਚਲ std::ਸਦਿਸ਼<std::string> & get_ਨਿਵੇਸ਼_ਨਾਂ() ਅਚਲ { ਵਾਪਸ m_ਨਿਵੇਸ਼_ਨਾਂ; }

    ਅਚਲ std::ਸਦਿਸ਼<std::string> & get_output_ਨਾਂ() ਅਚਲ { ਵਾਪਸ m_output_ਨਾਂ; }

    size_t get_ਨਿਵੇਸ਼_ਮਾਪ() ਅਚਲ override { ਵਾਪਸ m_decoder->get_ਨਿਵੇਸ਼_ਮਾਪ(m_ਗੰਢ_idx); }

    ov::element::Type get_ਨਿਵੇਸ਼_type(size_t ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_ਨਿਵੇਸ਼_type(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ]);
    }

    PartialShape get_ਨਿਵੇਸ਼_shape(size_t ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_ਨਿਵੇਸ਼_shape(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ]);
    }

    std::ਸਦਿਸ਼<size_t> get_ਨਿਵੇਸ਼_stride(size_t ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_ਨਿਵੇਸ਼_stride(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ]);
    }

    std::string get_output_name() ਅਚਲ { ਵਾਪਸ m_output_ਨਾਂ[0]; }

    PartialShape get_output_shape() ਅਚਲ { ਵਾਪਸ m_decoder->get_output_shape(m_ਗੰਢ_idx); }

    int32_t * get_ਨਿਵੇਸ਼_op_params(size_t ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_ਨਿਵੇਸ਼_op_params(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ]);
    }

    size_t get_view_ਨਿਵੇਸ਼_ਮਾਪ(size_t ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_ਮਾਪ(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ]);
    }

    size_t get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(size_t ਸੂਚਕਾਂਕ, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ], view_ਸੂਚਕਾਂਕ);
    }

    size_t get_view_ਨਿਵੇਸ਼_src_ਵਿੱਥ(size_t ਸੂਚਕਾਂਕ, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_src_ਵਿੱਥ(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ], view_ਸੂਚਕਾਂਕ);
    }

    std::ਸਦਿਸ਼<size_t> get_view_ਨਿਵੇਸ਼_stride(size_t ਸੂਚਕਾਂਕ, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_stride(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ], view_ਸੂਚਕਾਂਕ);
    }

    std::ਸਦਿਸ਼<size_t> get_view_ਨਿਵੇਸ਼_src_stride(size_t ਸੂਚਕਾਂਕ, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_src_stride(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ], view_ਸੂਚਕਾਂਕ);
    }

    ov::Shape get_view_ਨਿਵੇਸ਼_ggml_shape(size_t ਸੂਚਕਾਂਕ, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_ggml_shape(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ], view_ਸੂਚਕਾਂਕ);
    }

    ov::Shape get_view_ਨਿਵੇਸ਼_src_ggml_shape(size_t ਸੂਚਕਾਂਕ, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_src_ggml_shape(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ], view_ਸੂਚਕਾਂਕ);
    }

    ov::PartialShape get_view_ਨਿਵੇਸ਼_ov_shape(size_t ਸੂਚਕਾਂਕ, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_ov_shape(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ], view_ਸੂਚਕਾਂਕ);
    }

    ov::PartialShape get_view_ਨਿਵੇਸ਼_src_ov_shape(size_t ਸੂਚਕਾਂਕ, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_src_ov_shape(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ], view_ਸੂਚਕਾਂਕ);
    }

    std::string get_view_ਨਿਵੇਸ਼_name(size_t ਸੂਚਕਾਂਕ, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_name(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ], view_ਸੂਚਕਾਂਕ);
    }

    std::string get_view_ਨਿਵੇਸ਼_src_name(size_t ਸੂਚਕਾਂਕ, size_t view_ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ m_decoder->get_view_ਨਿਵੇਸ਼_src_name(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[ਸੂਚਕਾਂਕ], view_ਸੂਚਕਾਂਕ);
    }

    int32_t get_op_dynamic_dim() ਅਚਲ { ਵਾਪਸ m_decoder->get_op_dynamic_dim(m_ਗੰਢ_idx); }

    int32_t * get_output_op_params() ਅਚਲ { ਵਾਪਸ m_decoder->get_output_op_params(m_ਗੰਢ_idx); }

    size_t get_output_op_ਵਿੱਥ() ਅਚਲ { ਵਾਪਸ m_decoder->get_output_op_ਵਿੱਥ(m_ਗੰਢ_idx); }

    ov::element::Type get_output_type() ਅਚਲ { ਵਾਪਸ m_decoder->get_output_type(m_ਗੰਢ_idx); }

    std::ਸਦਿਸ਼<size_t> get_output_stride() ਅਚਲ { ਵਾਪਸ m_decoder->get_output_stride(m_ਗੰਢ_idx); }

    Output<ਗੰਢ_2> get_ਨਿਵੇਸ਼(ਪੂਰਨ idx) ਅਚਲ override {
        // Check if this input is a VIEW
        size_t view_ਨਿਵੇਸ਼_ਮਾਪ = m_decoder->get_view_ਨਿਵੇਸ਼_ਮਾਪ(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[idx]);
        ਜੇ (view_ਨਿਵੇਸ਼_ਮਾਪ > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string ਅਧਾਰ_name =
                m_decoder->get_view_ਨਿਵੇਸ਼_src_name(m_ਗੰਢ_idx, m_ਨਿਵੇਸ਼_ਨਾਂ[idx], view_ਨਿਵੇਸ਼_ਮਾਪ - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            ਸਵੈਚਾਲਿਤ view_it = m_tensor_map->ਲੱਭੋ(m_ਨਿਵੇਸ਼_ਨਾਂ[idx]);
            ਜੇ (!ਅਧਾਰ_name.ਖਾਲੀ_ਥਾਵਾਂ() && view_it != m_tensor_map->ਸਿਰਾ()) {
                ਸਵੈਚਾਲਿਤ ਅਧਾਰ_it = m_tensor_map->ਲੱਭੋ(ਅਧਾਰ_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                ਜੇ (ਅਧਾਰ_it == m_tensor_map->ਸਿਰਾ() ||
                    view_it->ਦੂਜਾ.get_ਗੰਢ_shared_ptr() != ਅਧਾਰ_it->ਦੂਜਾ.get_ਗੰਢ_shared_ptr()) {
                    ਵਾਪਸ view_it->ਦੂਜਾ;
                }
                ਵਾਪਸ ਅਧਾਰ_it->ਦੂਜਾ;
            }
            ਜੇ (!ਅਧਾਰ_name.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਵਾਪਸ m_tensor_map->at(ਅਧਾਰ_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        ਵਾਪਸ m_tensor_map->at(m_ਨਿਵੇਸ਼_ਨਾਂ[idx]);
    }

    Output<ਗੰਢ_2> get_ਨਿਵੇਸ਼(ਅਚਲ std::string & name) ਅਚਲ override {
        ਜੇ (m_tensor_map->ਲੱਭੋ(name) == m_tensor_map->ਸਿਰਾ()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        ਵਾਪਸ m_tensor_map->at(name);
    }

    bool has_ਨਿਵੇਸ਼(ਅਚਲ std::string & name) ਅਚਲ { ਵਾਪਸ m_tensor_map->ਲੱਭੋ(name) != m_tensor_map->ਸਿਰਾ(); }

    ਖਾਲੀ put_shared(ਅਚਲ std::string & name, ਅਚਲ Output<ਗੰਢ_2> & ਮੁੱਲ) ਅਚਲ {
        m_tensor_map->ਪਾਓ({name, ਮੁੱਲ});
    }

    ਅਚਲ std::string & get_name() ਅਚਲ override { ਵਾਪਸ m_decoder->get_op_name(m_ਗੰਢ_idx); }

    ov::Any get_attribute_as_any(ਅਚਲ std::string & name) ਅਚਲ override { ਵਾਪਸ m_decoder->get_attribute(name); }

    ਪੂਰਨ get_op_case() ਅਚਲ { ਵਾਪਸ m_decoder->get_op_case(m_ਗੰਢ_idx); }

    bool is_static() ਅਚਲ { ਵਾਪਸ m_decoder->is_static(); }

    bool is_stateful() ਅਚਲ { ਵਾਪਸ m_decoder->is_stateful(); }

    ਪੂਰਨ get_ssm_state_ਮਾਪ() ਅਚਲ { ਵਾਪਸ m_decoder->get_ssm_state_ਮਾਪ(); }

ਨਿਜੀ:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    ਪੂਰਨ m_ਗੰਢ_idx;
    TranslateSession * m_translate_session;
    std::ਸਦਿਸ਼<std::string> m_ਨਿਵੇਸ਼_ਨਾਂ;
    std::ਸਦਿਸ਼<std::string> m_output_ਨਾਂ;
};

using CreatorFunction = std::function<ov::Output_ਸਦਿਸ਼(ਅਚਲ ov::frontend::ggml::ਗੰਢ_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
