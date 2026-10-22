#निर्देशः once

#अन्तर्भाव "decoder.h"

#अन्तर्भाव <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_अन्तः;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class निवेशः_Model : सार्वजनिक ov::frontend::निवेशः_Model {
    मित्रम् class ::ov::frontend::ggml::Front_अन्तः;

सार्वजनिक:
    explicit निवेशः_Model(ध्रुव std::shared_ptr<GgmlDecoder> & gdecoder);

    ध्रुव std::shared_ptr<GgmlDecoder> & get_model_decoder() ध्रुव;

गुप्त:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
