#અન્તર્ભાવ "input_model.h"

#અન્તર્ભાવ "decoder.h"

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {

આગત_Model::આગત_Model(અચલ std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

અચલ std::shared_ptr<GgmlDecoder> & આગત_Model::get_model_decoder() અચલ {
    પાછા m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
