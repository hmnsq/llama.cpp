#رهنمود once

#ضمن "decoder.h"

#ضمن <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_انتها;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class درون_داد_Model : عام ov::frontend::درون_داد_Model {
    دوست class ::ov::frontend::ggml::Front_انتها;

عام:
    explicit درون_داد_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder);

    ثابت std::shared_ptr<GgmlDecoder> & get_model_decoder() ثابت;

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
