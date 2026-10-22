#ಅನ್ತರ್ಭಾವ "input_model.h"

#ಅನ್ತರ್ಭಾವ "decoder.h"

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {

ಒಳಹರಿವು_Model::ಒಳಹರಿವು_Model(ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ಸ್ಥಿರಾಂಕ std::shared_ptr<GgmlDecoder> & ಒಳಹರಿವು_Model::get_model_decoder() ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
