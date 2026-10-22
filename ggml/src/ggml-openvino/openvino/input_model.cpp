#অন্তর্ভাৱ "input_model.h"

#অন্তর্ভাৱ "decoder.h"

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {

প্ৰবেশ_Model::প্ৰবেশ_Model(ধ্রুৱ std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ধ্রুৱ std::shared_ptr<GgmlDecoder> & প্ৰবেশ_Model::get_model_decoder() ধ্রুৱ {
    ঘূৰাই m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
