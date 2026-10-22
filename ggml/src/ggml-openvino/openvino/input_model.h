#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "decoder.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_ਸਿਰਾ;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class ਨਿਵੇਸ਼_Model : ਜਨਤਕ ov::frontend::ਨਿਵੇਸ਼_Model {
    ਮਿੱਤਰ class ::ov::frontend::ggml::Front_ਸਿਰਾ;

ਜਨਤਕ:
    explicit ਨਿਵੇਸ਼_Model(ਅਚਲ std::shared_ptr<GgmlDecoder> & gdecoder);

    ਅਚਲ std::shared_ptr<GgmlDecoder> & get_model_decoder() ਅਚਲ;

ਨਿਜੀ:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
