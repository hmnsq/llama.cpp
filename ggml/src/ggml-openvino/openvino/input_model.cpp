#অন্তর্ভাব "input_model.h"

#অন্তর্ভাব "decoder.h"

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {

প্রবেশ_Model::প্রবেশ_Model(ধ্রুবক std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ধ্রুবক std::shared_ptr<GgmlDecoder> & প্রবেশ_Model::get_model_decoder() ধ্রুবক {
    ফেরত m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
