#ہدایت once

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
        m_داخلہ_ناں = decoder->get_داخلہ_ناں(m_گرہ_idx);
        m_output_ناں = decoder->get_output_ناں(m_گرہ_idx);
    }

    TranslateSession * get_translate_session() ثابت { أرجع m_translate_session; }

    ثابت std::سمتیہ<std::string> & get_داخلہ_ناں() ثابت { أرجع m_داخلہ_ناں; }

    ثابت std::سمتیہ<std::string> & get_output_ناں() ثابت { أرجع m_output_ناں; }

    size_t get_داخلہ_ناپ() ثابت override { أرجع m_decoder->get_داخلہ_ناپ(m_گرہ_idx); }

    ov::element::Type get_داخلہ_type(size_t اشاریہ) ثابت {
        أرجع m_decoder->get_داخلہ_type(m_گرہ_idx, m_داخلہ_ناں[اشاریہ]);
    }

    PartialShape get_داخلہ_shape(size_t داخلہ_اشاریہ) ثابت {
        أرجع m_decoder->get_داخلہ_shape(m_گرہ_idx, m_داخلہ_ناں[داخلہ_اشاریہ]);
    }

    std::سمتیہ<size_t> get_داخلہ_stride(size_t اشاریہ) ثابت {
        أرجع m_decoder->get_داخلہ_stride(m_گرہ_idx, m_داخلہ_ناں[اشاریہ]);
    }

    std::string get_output_name() ثابت { أرجع m_output_ناں[0]; }

    PartialShape get_output_shape() ثابت { أرجع m_decoder->get_output_shape(m_گرہ_idx); }

    int32_t * get_داخلہ_op_params(size_t اشاریہ) ثابت {
        أرجع m_decoder->get_داخلہ_op_params(m_گرہ_idx, m_داخلہ_ناں[اشاریہ]);
    }

    size_t get_view_داخلہ_ناپ(size_t اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_ناپ(m_گرہ_idx, m_داخلہ_ناں[اشاریہ]);
    }

    size_t get_view_داخلہ_وتھ(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_وتھ(m_گرہ_idx, m_داخلہ_ناں[اشاریہ], view_اشاریہ);
    }

    size_t get_view_داخلہ_src_وتھ(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_src_وتھ(m_گرہ_idx, m_داخلہ_ناں[اشاریہ], view_اشاریہ);
    }

    std::سمتیہ<size_t> get_view_داخلہ_stride(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_stride(m_گرہ_idx, m_داخلہ_ناں[اشاریہ], view_اشاریہ);
    }

    std::سمتیہ<size_t> get_view_داخلہ_src_stride(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_src_stride(m_گرہ_idx, m_داخلہ_ناں[اشاریہ], view_اشاریہ);
    }

    ov::Shape get_view_داخلہ_ggml_shape(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_ggml_shape(m_گرہ_idx, m_داخلہ_ناں[اشاریہ], view_اشاریہ);
    }

    ov::Shape get_view_داخلہ_src_ggml_shape(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_src_ggml_shape(m_گرہ_idx, m_داخلہ_ناں[اشاریہ], view_اشاریہ);
    }

    ov::PartialShape get_view_داخلہ_ov_shape(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_ov_shape(m_گرہ_idx, m_داخلہ_ناں[اشاریہ], view_اشاریہ);
    }

    ov::PartialShape get_view_داخلہ_src_ov_shape(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_src_ov_shape(m_گرہ_idx, m_داخلہ_ناں[اشاریہ], view_اشاریہ);
    }

    std::string get_view_داخلہ_name(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_name(m_گرہ_idx, m_داخلہ_ناں[اشاریہ], view_اشاریہ);
    }

    std::string get_view_داخلہ_src_name(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        أرجع m_decoder->get_view_داخلہ_src_name(m_گرہ_idx, m_داخلہ_ناں[اشاریہ], view_اشاریہ);
    }

    int32_t get_op_dynamic_dim() ثابت { أرجع m_decoder->get_op_dynamic_dim(m_گرہ_idx); }

    int32_t * get_output_op_params() ثابت { أرجع m_decoder->get_output_op_params(m_گرہ_idx); }

    size_t get_output_op_وتھ() ثابت { أرجع m_decoder->get_output_op_وتھ(m_گرہ_idx); }

    ov::element::Type get_output_type() ثابت { أرجع m_decoder->get_output_type(m_گرہ_idx); }

    std::سمتیہ<size_t> get_output_stride() ثابت { أرجع m_decoder->get_output_stride(m_گرہ_idx); }

    Output<گرہ_2> get_داخلہ(صحيح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_داخلہ_ناپ = m_decoder->get_view_داخلہ_ناپ(m_گرہ_idx, m_داخلہ_ناں[idx]);
        إذا (view_داخلہ_ناپ > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string ادھار_name =
                m_decoder->get_view_داخلہ_src_name(m_گرہ_idx, m_داخلہ_ناں[idx], view_داخلہ_ناپ - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            خودکار view_it = m_tensor_map->لبھو(m_داخلہ_ناں[idx]);
            إذا (!ادھار_name.خالی_تھاواں() && view_it != m_tensor_map->سرا()) {
                خودکار ادھار_it = m_tensor_map->لبھو(ادھار_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                إذا (ادھار_it == m_tensor_map->سرا() ||
                    view_it->دوجا.get_گرہ_shared_ptr() != ادھار_it->دوجا.get_گرہ_shared_ptr()) {
                    أرجع view_it->دوجا;
                }
                أرجع ادھار_it->دوجا;
            }
            إذا (!ادھار_name.خالی_تھاواں()) {
                أرجع m_tensor_map->at(ادھار_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        أرجع m_tensor_map->at(m_داخلہ_ناں[idx]);
    }

    Output<گرہ_2> get_داخلہ(ثابت std::string & name) ثابت override {
        إذا (m_tensor_map->لبھو(name) == m_tensor_map->سرا()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        أرجع m_tensor_map->at(name);
    }

    bool has_داخلہ(ثابت std::string & name) ثابت { أرجع m_tensor_map->لبھو(name) != m_tensor_map->سرا(); }

    فراغ put_shared(ثابت std::string & name, ثابت Output<گرہ_2> & قدر) ثابت {
        m_tensor_map->داخل_کرو({name, قدر});
    }

    ثابت std::string & get_name() ثابت override { أرجع m_decoder->get_op_name(m_گرہ_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { أرجع m_decoder->get_attribute(name); }

    صحيح get_op_case() ثابت { أرجع m_decoder->get_op_case(m_گرہ_idx); }

    bool is_static() ثابت { أرجع m_decoder->is_static(); }

    bool is_stateful() ثابت { أرجع m_decoder->is_stateful(); }

    صحيح get_ssm_state_ناپ() ثابت { أرجع m_decoder->get_ssm_state_ناپ(); }

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحيح m_گرہ_idx;
    TranslateSession * m_translate_session;
    std::سمتیہ<std::string> m_داخلہ_ناں;
    std::سمتیہ<std::string> m_output_ناں;
};

using CreatorFunction = std::function<ov::Output_سمتیہ(ثابت ov::frontend::ggml::گرہ_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
