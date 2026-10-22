#لارښوونه once

#ضمن "decoder.h"

#ضمن <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_پای_ټکی;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class ننوت_Model : عام ov::frontend::ننوت_Model {
    ملګری class ::ov::frontend::ggml::Front_پای_ټکی;

عام:
    explicit ننوت_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder);

    ثابت std::shared_ptr<GgmlDecoder> & get_model_decoder() ثابت;

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
