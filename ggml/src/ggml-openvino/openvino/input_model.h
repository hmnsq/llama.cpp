#ہدایت once

#شامل "decoder.h"

#شامل <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_انتہا;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class داخلہ_Model : عوامی ov::frontend::داخلہ_Model {
    دوست class ::ov::frontend::ggml::Front_انتہا;

عوامی:
    explicit داخلہ_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder);

    ثابت std::shared_ptr<GgmlDecoder> & get_model_decoder() ثابت;

نجی:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
