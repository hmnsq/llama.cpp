#अन्तर्भाव "input_model.h"

#अन्तर्भाव "decoder.h"

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {

निवेश_Model::निवेश_Model(ध्रुव std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ध्रुव std::shared_ptr<GgmlDecoder> & निवेश_Model::get_model_decoder() ध्रुव {
    प्रत्यागम m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
