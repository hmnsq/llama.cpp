#अन्तर्भाव "input_model.h"

#अन्तर्भाव "decoder.h"

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {

निवेशः_Model::निवेशः_Model(ध्रुव std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ध्रुव std::shared_ptr<GgmlDecoder> & निवेशः_Model::get_model_decoder() ध्रुव {
    पटिनिवत्त m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
