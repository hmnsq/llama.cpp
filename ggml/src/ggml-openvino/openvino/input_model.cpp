#अन्तर्भाव "input_model.h"

#अन्तर्भाव "decoder.h"

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {

आगत_Model::आगत_Model(अचर std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

अचर std::shared_ptr<GgmlDecoder> & आगत_Model::get_model_decoder() अचर {
    फर्काऊ m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
