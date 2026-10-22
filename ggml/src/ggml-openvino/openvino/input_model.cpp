#include "input_model.h"

#include "decoder.h"

namespace ov {
namespace frontend {
namespace ggml {

पासोक_Model::पासोक_Model(const std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

const std::shared_ptr<GgmlDecoder> & पासोक_Model::get_model_decoder() const {
    return m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
