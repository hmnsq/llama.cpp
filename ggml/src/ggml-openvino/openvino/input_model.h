#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "decoder.h"

#ಅನ್ತರ್ಭಾವ <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_ಕೊನೆ;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class ಒಳಹರಿವು_Model : ಸಾರ್ವಜನಿಕ ov::frontend::ಒಳಹರಿವು_Model {
    ಮಿತ್ರ class ::ov::frontend::ggml::Front_ಕೊನೆ;

ಸಾರ್ವಜನಿಕ:
    explicit ಒಳಹರಿವು_Model(ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlDecoder> & gdecoder);

    ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlDecoder> & get_model_decoder() ಸ್ಥಿರಾಂಕ;

ಖಾಸಗಿ:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
