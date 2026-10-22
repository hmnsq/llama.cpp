#هدايت once

#ضمن "decoder.h"

#ضمن <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_پڇاڙي;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class داخلا_Model : عام ov::frontend::داخلا_Model {
    دوست class ::ov::frontend::ggml::Front_پڇاڙي;

عام:
    explicit داخلا_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder);

    ثابت std::shared_ptr<GgmlDecoder> & get_model_decoder() ثابت;

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
