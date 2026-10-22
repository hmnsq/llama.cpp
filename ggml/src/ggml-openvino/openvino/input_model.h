#ہدایت once

#ضمن "decoder.h"

#ضمن <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_سرا;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class داخلہ_Model : عام ov::frontend::داخلہ_Model {
    یار class ::ov::frontend::ggml::Front_سرا;

عام:
    explicit داخلہ_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder);

    ثابت std::shared_ptr<GgmlDecoder> & get_model_decoder() ثابت;

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
