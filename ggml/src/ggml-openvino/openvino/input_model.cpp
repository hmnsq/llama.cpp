#شامل "input_model.h"

#شامل "decoder.h"

نامخلا ov {
نامخلا frontend {
نامخلا ggml {

داخلہ_Model::داخلہ_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ثابت std::shared_ptr<GgmlDecoder> & داخلہ_Model::get_model_decoder() ثابت {
    واپس m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
