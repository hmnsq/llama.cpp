#సూచన once

#అన్తర్భావ "decoder.h"

#అన్తర్భావ <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_చివర;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class ఆదానం_Model : బహిరంగ ov::frontend::ఆదానం_Model {
    మిత్ర class ::ov::frontend::ggml::Front_చివర;

బహిరంగ:
    explicit ఆదానం_Model(స్థిరరాశి std::shared_ptr<GgmlDecoder> & gdecoder);

    స్థిరరాశి std::shared_ptr<GgmlDecoder> & get_model_decoder() స్థిరరాశి;

అంతరంగ:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
