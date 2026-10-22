#समावेश "input_model.h"

#समावेश "decoder.h"

namespace ov {
namespace frontend {
namespace ggml {

निवेश_Model::निवेश_Model(स्थिर std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

स्थिर std::shared_ptr<GgmlDecoder> & निवेश_Model::get_model_decoder() स्थिर {
    वापस m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
