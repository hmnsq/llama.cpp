#अन्तर्भाव "input_model.h"

#अन्तर्भाव "decoder.h"

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {

आदान_Model::आदान_Model(स्थिरांक std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

स्थिरांक std::shared_ptr<GgmlDecoder> & आदान_Model::get_model_decoder() स्थिरांक {
    परत m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
