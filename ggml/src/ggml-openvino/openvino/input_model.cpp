#അന്തര്ഭാവ "input_model.h"

#അന്തര്ഭാവ "decoder.h"

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {

നിവേശം_Model::നിവേശം_Model(സ്ഥിരം std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

സ്ഥിരം std::shared_ptr<GgmlDecoder> & നിവേശം_Model::get_model_decoder() സ്ഥിരം {
    തിരികെ m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
