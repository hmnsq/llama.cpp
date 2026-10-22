#సూచన once

#అన్తర్భావ "decoder.h"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <openvino/frontend/node_context.hpp>
#అన్తర్భావ <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

రకనిర్వచనం std::map<std::string, Output<కణుపు_2>> TensorMap;

class కణుపు_Context : బహిరంగ frontend::కణుపు_Context {
బహిరంగ:
    కణుపు_Context(స్థిరరాశి std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                పూర్ణాంకం కణుపు_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::కణుపు_Context(decoder->get_op_type(కణుపు_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_కణుపు_idx(కణుపు_idx),
        m_translate_session(translate_session) {
        m_ఆదానం_పేర్లు = decoder->get_ఆదానం_పేర్లు(m_కణుపు_idx);
        m_output_పేర్లు = decoder->get_output_పేర్లు(m_కణుపు_idx);
    }

    TranslateSession * get_translate_session() స్థిరరాశి { తిరిగి m_translate_session; }

    స్థిరరాశి std::సదిశ<std::string> & get_ఆదానం_పేర్లు() స్థిరరాశి { తిరిగి m_ఆదానం_పేర్లు; }

    స్థిరరాశి std::సదిశ<std::string> & get_output_పేర్లు() స్థిరరాశి { తిరిగి m_output_పేర్లు; }

    size_t get_ఆదానం_కొలత() స్థిరరాశి override { తిరిగి m_decoder->get_ఆదానం_కొలత(m_కణుపు_idx); }

    ov::element::Type get_ఆదానం_type(size_t సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_ఆదానం_type(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ]);
    }

    PartialShape get_ఆదానం_shape(size_t ఆదానం_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_ఆదానం_shape(m_కణుపు_idx, m_ఆదానం_పేర్లు[ఆదానం_సూచీ]);
    }

    std::సదిశ<size_t> get_ఆదానం_stride(size_t సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_ఆదానం_stride(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ]);
    }

    std::string get_output_name() స్థిరరాశి { తిరిగి m_output_పేర్లు[0]; }

    PartialShape get_output_shape() స్థిరరాశి { తిరిగి m_decoder->get_output_shape(m_కణుపు_idx); }

    int32_t * get_ఆదానం_op_params(size_t సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_ఆదానం_op_params(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ]);
    }

    size_t get_view_ఆదానం_కొలత(size_t సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_కొలత(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ]);
    }

    size_t get_view_ఆదానం_స్థానభ్రంశం(size_t సూచీ, size_t view_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_స్థానభ్రంశం(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ], view_సూచీ);
    }

    size_t get_view_ఆదానం_src_స్థానభ్రంశం(size_t సూచీ, size_t view_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_src_స్థానభ్రంశం(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ], view_సూచీ);
    }

    std::సదిశ<size_t> get_view_ఆదానం_stride(size_t సూచీ, size_t view_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_stride(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ], view_సూచీ);
    }

    std::సదిశ<size_t> get_view_ఆదానం_src_stride(size_t సూచీ, size_t view_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_src_stride(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ], view_సూచీ);
    }

    ov::Shape get_view_ఆదానం_ggml_shape(size_t సూచీ, size_t view_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_ggml_shape(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ], view_సూచీ);
    }

    ov::Shape get_view_ఆదానం_src_ggml_shape(size_t సూచీ, size_t view_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_src_ggml_shape(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ], view_సూచీ);
    }

    ov::PartialShape get_view_ఆదానం_ov_shape(size_t సూచీ, size_t view_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_ov_shape(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ], view_సూచీ);
    }

    ov::PartialShape get_view_ఆదానం_src_ov_shape(size_t సూచీ, size_t view_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_src_ov_shape(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ], view_సూచీ);
    }

    std::string get_view_ఆదానం_name(size_t సూచీ, size_t view_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_name(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ], view_సూచీ);
    }

    std::string get_view_ఆదానం_src_name(size_t సూచీ, size_t view_సూచీ) స్థిరరాశి {
        తిరిగి m_decoder->get_view_ఆదానం_src_name(m_కణుపు_idx, m_ఆదానం_పేర్లు[సూచీ], view_సూచీ);
    }

    int32_t get_op_dynamic_dim() స్థిరరాశి { తిరిగి m_decoder->get_op_dynamic_dim(m_కణుపు_idx); }

    int32_t * get_output_op_params() స్థిరరాశి { తిరిగి m_decoder->get_output_op_params(m_కణుపు_idx); }

    size_t get_output_op_స్థానభ్రంశం() స్థిరరాశి { తిరిగి m_decoder->get_output_op_స్థానభ్రంశం(m_కణుపు_idx); }

    ov::element::Type get_output_type() స్థిరరాశి { తిరిగి m_decoder->get_output_type(m_కణుపు_idx); }

    std::సదిశ<size_t> get_output_stride() స్థిరరాశి { తిరిగి m_decoder->get_output_stride(m_కణుపు_idx); }

    Output<కణుపు_2> get_ఆదానం(పూర్ణాంకం idx) స్థిరరాశి override {
        // Check if this input is a VIEW
        size_t view_ఆదానం_కొలత = m_decoder->get_view_ఆదానం_కొలత(m_కణుపు_idx, m_ఆదానం_పేర్లు[idx]);
        ఒకవేళ (view_ఆదానం_కొలత > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string పునాది_name =
                m_decoder->get_view_ఆదానం_src_name(m_కణుపు_idx, m_ఆదానం_పేర్లు[idx], view_ఆదానం_కొలత - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            స్వయంచాలక view_it = m_tensor_map->కనుగొను(m_ఆదానం_పేర్లు[idx]);
            ఒకవేళ (!పునాది_name.ఖాళీలు() && view_it != m_tensor_map->చివర()) {
                స్వయంచాలక పునాది_it = m_tensor_map->కనుగొను(పునాది_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                ఒకవేళ (పునాది_it == m_tensor_map->చివర() ||
                    view_it->రెండవ.get_కణుపు_shared_ptr() != పునాది_it->రెండవ.get_కణుపు_shared_ptr()) {
                    తిరిగి view_it->రెండవ;
                }
                తిరిగి పునాది_it->రెండవ;
            }
            ఒకవేళ (!పునాది_name.ఖాళీలు()) {
                తిరిగి m_tensor_map->at(పునాది_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        తిరిగి m_tensor_map->at(m_ఆదానం_పేర్లు[idx]);
    }

    Output<కణుపు_2> get_ఆదానం(స్థిరరాశి std::string & name) స్థిరరాశి override {
        ఒకవేళ (m_tensor_map->కనుగొను(name) == m_tensor_map->చివర()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        తిరిగి m_tensor_map->at(name);
    }

    bool has_ఆదానం(స్థిరరాశి std::string & name) స్థిరరాశి { తిరిగి m_tensor_map->కనుగొను(name) != m_tensor_map->చివర(); }

    శూన్య put_shared(స్థిరరాశి std::string & name, స్థిరరాశి Output<కణుపు_2> & విలువ) స్థిరరాశి {
        m_tensor_map->చొప్పించు({name, విలువ});
    }

    స్థిరరాశి std::string & get_name() స్థిరరాశి override { తిరిగి m_decoder->get_op_name(m_కణుపు_idx); }

    ov::Any get_attribute_as_any(స్థిరరాశి std::string & name) స్థిరరాశి override { తిరిగి m_decoder->get_attribute(name); }

    పూర్ణాంకం get_op_case() స్థిరరాశి { తిరిగి m_decoder->get_op_case(m_కణుపు_idx); }

    bool is_static() స్థిరరాశి { తిరిగి m_decoder->is_static(); }

    bool is_stateful() స్థిరరాశి { తిరిగి m_decoder->is_stateful(); }

    పూర్ణాంకం get_ssm_state_కొలత() స్థిరరాశి { తిరిగి m_decoder->get_ssm_state_కొలత(); }

అంతరంగ:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    పూర్ణాంకం m_కణుపు_idx;
    TranslateSession * m_translate_session;
    std::సదిశ<std::string> m_ఆదానం_పేర్లు;
    std::సదిశ<std::string> m_output_పేర్లు;
};

using CreatorFunction = std::function<ov::Output_సదిశ(స్థిరరాశి ov::frontend::ggml::కణుపు_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
