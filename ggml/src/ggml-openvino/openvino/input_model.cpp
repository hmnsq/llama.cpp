#अन्तर्भाव "input_model.h"

#अन्तर्भाव "decoder.h"

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {

आदान_Model::आदान_Model(ध्रुव std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ध्रुव std::shared_ptr<GgmlDecoder> & आदान_Model::get_model_decoder() ध्रुव {
    प्रत्यागम m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
