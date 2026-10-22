#رهنمود once

#ضمن "decoder.h"

#ضمن <cstdint>
#ضمن <openvino/frontend/node_context.hpp>
#ضمن <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

تعريفنوع std::map<std::string, Output<گره_2>> TensorMap;

class گره_Context : عام frontend::گره_Context {
عام:
    گره_Context(ثابت std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                صحيح گره_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::گره_Context(decoder->get_op_type(گره_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_گره_idx(گره_idx),
        m_translate_session(translate_session) {
        m_درون_داد_نامها = decoder->get_درون_داد_نامها(m_گره_idx);
        m_output_نامها = decoder->get_output_نامها(m_گره_idx);
    }

    TranslateSession * get_translate_session() ثابت { أرجع m_translate_session; }

    ثابت std::وکتور<std::string> & get_درون_داد_نامها() ثابت { أرجع m_درون_داد_نامها; }

    ثابت std::وکتور<std::string> & get_output_نامها() ثابت { أرجع m_output_نامها; }

    size_t get_درون_داد_اندازه() ثابت override { أرجع m_decoder->get_درون_داد_اندازه(m_گره_idx); }

    ov::element::Type get_درون_داد_type(size_t شاخص) ثابت {
        أرجع m_decoder->get_درون_داد_type(m_گره_idx, m_درون_داد_نامها[شاخص]);
    }

    PartialShape get_درون_داد_shape(size_t درون_داد_شاخص) ثابت {
        أرجع m_decoder->get_درون_داد_shape(m_گره_idx, m_درون_داد_نامها[درون_داد_شاخص]);
    }

    std::وکتور<size_t> get_درون_داد_stride(size_t شاخص) ثابت {
        أرجع m_decoder->get_درون_داد_stride(m_گره_idx, m_درون_داد_نامها[شاخص]);
    }

    std::string get_output_name() ثابت { أرجع m_output_نامها[0]; }

    PartialShape get_output_shape() ثابت { أرجع m_decoder->get_output_shape(m_گره_idx); }

    int32_t * get_درون_داد_op_params(size_t شاخص) ثابت {
        أرجع m_decoder->get_درون_داد_op_params(m_گره_idx, m_درون_داد_نامها[شاخص]);
    }

    size_t get_view_درون_داد_اندازه(size_t شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_اندازه(m_گره_idx, m_درون_داد_نامها[شاخص]);
    }

    size_t get_view_درون_داد_جابه_جایی(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_جابه_جایی(m_گره_idx, m_درون_داد_نامها[شاخص], view_شاخص);
    }

    size_t get_view_درون_داد_src_جابه_جایی(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_src_جابه_جایی(m_گره_idx, m_درون_داد_نامها[شاخص], view_شاخص);
    }

    std::وکتور<size_t> get_view_درون_داد_stride(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_stride(m_گره_idx, m_درون_داد_نامها[شاخص], view_شاخص);
    }

    std::وکتور<size_t> get_view_درون_داد_src_stride(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_src_stride(m_گره_idx, m_درون_داد_نامها[شاخص], view_شاخص);
    }

    ov::Shape get_view_درون_داد_ggml_shape(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_ggml_shape(m_گره_idx, m_درون_داد_نامها[شاخص], view_شاخص);
    }

    ov::Shape get_view_درون_داد_src_ggml_shape(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_src_ggml_shape(m_گره_idx, m_درون_داد_نامها[شاخص], view_شاخص);
    }

    ov::PartialShape get_view_درون_داد_ov_shape(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_ov_shape(m_گره_idx, m_درون_داد_نامها[شاخص], view_شاخص);
    }

    ov::PartialShape get_view_درون_داد_src_ov_shape(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_src_ov_shape(m_گره_idx, m_درون_داد_نامها[شاخص], view_شاخص);
    }

    std::string get_view_درون_داد_name(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_name(m_گره_idx, m_درون_داد_نامها[شاخص], view_شاخص);
    }

    std::string get_view_درون_داد_src_name(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_درون_داد_src_name(m_گره_idx, m_درون_داد_نامها[شاخص], view_شاخص);
    }

    int32_t get_op_dynamic_dim() ثابت { أرجع m_decoder->get_op_dynamic_dim(m_گره_idx); }

    int32_t * get_output_op_params() ثابت { أرجع m_decoder->get_output_op_params(m_گره_idx); }

    size_t get_output_op_جابه_جایی() ثابت { أرجع m_decoder->get_output_op_جابه_جایی(m_گره_idx); }

    ov::element::Type get_output_type() ثابت { أرجع m_decoder->get_output_type(m_گره_idx); }

    std::وکتور<size_t> get_output_stride() ثابت { أرجع m_decoder->get_output_stride(m_گره_idx); }

    Output<گره_2> get_درون_داد(صحيح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_درون_داد_اندازه = m_decoder->get_view_درون_داد_اندازه(m_گره_idx, m_درون_داد_نامها[idx]);
        إذا (view_درون_داد_اندازه > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string پایه_name =
                m_decoder->get_view_درون_داد_src_name(m_گره_idx, m_درون_داد_نامها[idx], view_درون_داد_اندازه - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            خودکار view_it = m_tensor_map->پیدا_کن(m_درون_داد_نامها[idx]);
            إذا (!پایه_name.خالی() && view_it != m_tensor_map->انتها()) {
                خودکار پایه_it = m_tensor_map->پیدا_کن(پایه_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                إذا (پایه_it == m_tensor_map->انتها() ||
                    view_it->دومی.get_گره_shared_ptr() != پایه_it->دومی.get_گره_shared_ptr()) {
                    أرجع view_it->دومی;
                }
                أرجع پایه_it->دومی;
            }
            إذا (!پایه_name.خالی()) {
                أرجع m_tensor_map->at(پایه_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        أرجع m_tensor_map->at(m_درون_داد_نامها[idx]);
    }

    Output<گره_2> get_درون_داد(ثابت std::string & name) ثابت override {
        إذا (m_tensor_map->پیدا_کن(name) == m_tensor_map->انتها()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        أرجع m_tensor_map->at(name);
    }

    bool has_درون_داد(ثابت std::string & name) ثابت { أرجع m_tensor_map->پیدا_کن(name) != m_tensor_map->انتها(); }

    فراغ put_shared(ثابت std::string & name, ثابت Output<گره_2> & مقدار) ثابت {
        m_tensor_map->درج({name, مقدار});
    }

    ثابت std::string & get_name() ثابت override { أرجع m_decoder->get_op_name(m_گره_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { أرجع m_decoder->get_attribute(name); }

    صحيح get_op_case() ثابت { أرجع m_decoder->get_op_case(m_گره_idx); }

    bool is_static() ثابت { أرجع m_decoder->is_static(); }

    bool is_stateful() ثابت { أرجع m_decoder->is_stateful(); }

    صحيح get_ssm_state_اندازه() ثابت { أرجع m_decoder->get_ssm_state_اندازه(); }

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحيح m_گره_idx;
    TranslateSession * m_translate_session;
    std::وکتور<std::string> m_درون_داد_نامها;
    std::وکتور<std::string> m_output_نامها;
};

using CreatorFunction = std::function<ov::Output_وکتور(ثابت ov::frontend::ggml::گره_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
