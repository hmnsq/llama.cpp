#ہدایت once

#شامل "decoder.h"

#شامل <cstdint>
#شامل <openvino/frontend/node_context.hpp>
#شامل <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

قسمتعریف std::map<std::string, Output<گرہ_2>> TensorMap;

class گرہ_Context : عوامی frontend::گرہ_Context {
عوامی:
    گرہ_Context(ثابت std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                صحیح گرہ_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::گرہ_Context(decoder->get_op_type(گرہ_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_گرہ_idx(گرہ_idx),
        m_translate_session(translate_session) {
        m_داخلہ_نام = decoder->get_داخلہ_نام(m_گرہ_idx);
        m_output_نام = decoder->get_output_نام(m_گرہ_idx);
    }

    TranslateSession * get_translate_session() ثابت { واپس m_translate_session; }

    ثابت std::سمتیہ<std::string> & get_داخلہ_نام() ثابت { واپس m_داخلہ_نام; }

    ثابت std::سمتیہ<std::string> & get_output_نام() ثابت { واپس m_output_نام; }

    size_t get_داخلہ_سائز() ثابت override { واپس m_decoder->get_داخلہ_سائز(m_گرہ_idx); }

    ov::element::Type get_داخلہ_type(size_t اشاریہ) ثابت {
        واپس m_decoder->get_داخلہ_type(m_گرہ_idx, m_داخلہ_نام[اشاریہ]);
    }

    PartialShape get_داخلہ_shape(size_t داخلہ_اشاریہ) ثابت {
        واپس m_decoder->get_داخلہ_shape(m_گرہ_idx, m_داخلہ_نام[داخلہ_اشاریہ]);
    }

    std::سمتیہ<size_t> get_داخلہ_stride(size_t اشاریہ) ثابت {
        واپس m_decoder->get_داخلہ_stride(m_گرہ_idx, m_داخلہ_نام[اشاریہ]);
    }

    std::string get_output_name() ثابت { واپس m_output_نام[0]; }

    PartialShape get_output_shape() ثابت { واپس m_decoder->get_output_shape(m_گرہ_idx); }

    int32_t * get_داخلہ_op_params(size_t اشاریہ) ثابت {
        واپس m_decoder->get_داخلہ_op_params(m_گرہ_idx, m_داخلہ_نام[اشاریہ]);
    }

    size_t get_view_داخلہ_سائز(size_t اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_سائز(m_گرہ_idx, m_داخلہ_نام[اشاریہ]);
    }

    size_t get_view_داخلہ_انحراف(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_انحراف(m_گرہ_idx, m_داخلہ_نام[اشاریہ], view_اشاریہ);
    }

    size_t get_view_داخلہ_src_انحراف(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_src_انحراف(m_گرہ_idx, m_داخلہ_نام[اشاریہ], view_اشاریہ);
    }

    std::سمتیہ<size_t> get_view_داخلہ_stride(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_stride(m_گرہ_idx, m_داخلہ_نام[اشاریہ], view_اشاریہ);
    }

    std::سمتیہ<size_t> get_view_داخلہ_src_stride(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_src_stride(m_گرہ_idx, m_داخلہ_نام[اشاریہ], view_اشاریہ);
    }

    ov::Shape get_view_داخلہ_ggml_shape(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_ggml_shape(m_گرہ_idx, m_داخلہ_نام[اشاریہ], view_اشاریہ);
    }

    ov::Shape get_view_داخلہ_src_ggml_shape(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_src_ggml_shape(m_گرہ_idx, m_داخلہ_نام[اشاریہ], view_اشاریہ);
    }

    ov::PartialShape get_view_داخلہ_ov_shape(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_ov_shape(m_گرہ_idx, m_داخلہ_نام[اشاریہ], view_اشاریہ);
    }

    ov::PartialShape get_view_داخلہ_src_ov_shape(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_src_ov_shape(m_گرہ_idx, m_داخلہ_نام[اشاریہ], view_اشاریہ);
    }

    std::string get_view_داخلہ_name(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_name(m_گرہ_idx, m_داخلہ_نام[اشاریہ], view_اشاریہ);
    }

    std::string get_view_داخلہ_src_name(size_t اشاریہ, size_t view_اشاریہ) ثابت {
        واپس m_decoder->get_view_داخلہ_src_name(m_گرہ_idx, m_داخلہ_نام[اشاریہ], view_اشاریہ);
    }

    int32_t get_op_dynamic_dim() ثابت { واپس m_decoder->get_op_dynamic_dim(m_گرہ_idx); }

    int32_t * get_output_op_params() ثابت { واپس m_decoder->get_output_op_params(m_گرہ_idx); }

    size_t get_output_op_انحراف() ثابت { واپس m_decoder->get_output_op_انحراف(m_گرہ_idx); }

    ov::element::Type get_output_type() ثابت { واپس m_decoder->get_output_type(m_گرہ_idx); }

    std::سمتیہ<size_t> get_output_stride() ثابت { واپس m_decoder->get_output_stride(m_گرہ_idx); }

    Output<گرہ_2> get_داخلہ(صحیح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_داخلہ_سائز = m_decoder->get_view_داخلہ_سائز(m_گرہ_idx, m_داخلہ_نام[idx]);
        اگر (view_داخلہ_سائز > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string بنیاد_name =
                m_decoder->get_view_داخلہ_src_name(m_گرہ_idx, m_داخلہ_نام[idx], view_داخلہ_سائز - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            خودکار view_it = m_tensor_map->ڈھونڈو(m_داخلہ_نام[idx]);
            اگر (!بنیاد_name.خالی_خانے() && view_it != m_tensor_map->انتہا()) {
                خودکار بنیاد_it = m_tensor_map->ڈھونڈو(بنیاد_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                اگر (بنیاد_it == m_tensor_map->انتہا() ||
                    view_it->دوسرا.get_گرہ_shared_ptr() != بنیاد_it->دوسرا.get_گرہ_shared_ptr()) {
                    واپس view_it->دوسرا;
                }
                واپس بنیاد_it->دوسرا;
            }
            اگر (!بنیاد_name.خالی_خانے()) {
                واپس m_tensor_map->at(بنیاد_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        واپس m_tensor_map->at(m_داخلہ_نام[idx]);
    }

    Output<گرہ_2> get_داخلہ(ثابت std::string & name) ثابت override {
        اگر (m_tensor_map->ڈھونڈو(name) == m_tensor_map->انتہا()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        واپس m_tensor_map->at(name);
    }

    bool has_داخلہ(ثابت std::string & name) ثابت { واپس m_tensor_map->ڈھونڈو(name) != m_tensor_map->انتہا(); }

    خالی put_shared(ثابت std::string & name, ثابت Output<گرہ_2> & قدر) ثابت {
        m_tensor_map->داخل_کرو({name, قدر});
    }

    ثابت std::string & get_name() ثابت override { واپس m_decoder->get_op_name(m_گرہ_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { واپس m_decoder->get_attribute(name); }

    صحیح get_op_case() ثابت { واپس m_decoder->get_op_case(m_گرہ_idx); }

    bool is_static() ثابت { واپس m_decoder->is_static(); }

    bool is_stateful() ثابت { واپس m_decoder->is_stateful(); }

    صحیح get_ssm_state_سائز() ثابت { واپس m_decoder->get_ssm_state_سائز(); }

نجی:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحیح m_گرہ_idx;
    TranslateSession * m_translate_session;
    std::سمتیہ<std::string> m_داخلہ_نام;
    std::سمتیہ<std::string> m_output_نام;
};

using CreatorFunction = std::function<ov::Output_سمتیہ(ثابت ov::frontend::ggml::گرہ_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
