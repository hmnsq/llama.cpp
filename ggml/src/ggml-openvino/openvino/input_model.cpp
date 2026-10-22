#ଅନ୍ତର୍ଭାଵ "input_model.h"

#ଅନ୍ତର୍ଭାଵ "decoder.h"

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {

ନିବେଶ_Model::ନିବେଶ_Model(ଅଚଳ std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ଅଚଳ std::shared_ptr<GgmlDecoder> & ନିବେଶ_Model::get_model_decoder() ଅଚଳ {
    ଫେରନ୍ତୁ m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
