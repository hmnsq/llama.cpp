#ಅನ್ತರ್ಭಾವ "input_model.h"

#ಅನ್ತರ್ಭಾವ "decoder.h"

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {

ಒಳಹರಿವು_Model::ಒಳಹರಿವು_Model(ಧ್ರುವ std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ಧ್ರುವ std::shared_ptr<GgmlDecoder> & ಒಳಹರಿವು_Model::get_model_decoder() ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
