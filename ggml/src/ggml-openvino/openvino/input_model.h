#आशय once

#समावेश "decoder.h"

#समावेश <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class निवेश_Model : खुला ov::frontend::निवेश_Model {
    मित्र class ::ov::frontend::ggml::FrontEnd;

खुला:
    explicit निवेश_Model(स्थिर std::shared_ptr<GgmlDecoder> & gdecoder);

    स्थिर std::shared_ptr<GgmlDecoder> & get_model_decoder() स्थिर;

निजी:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
