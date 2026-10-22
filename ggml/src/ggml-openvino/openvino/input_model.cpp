#ਅਨ੍ਤਰ੍ਭਾਵ "input_model.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "decoder.h"

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {

ਨਿਵੇਸ਼_Model::ਨਿਵੇਸ਼_Model(ਅਚਲ std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

ਅਚਲ std::shared_ptr<GgmlDecoder> & ਨਿਵੇਸ਼_Model::get_model_decoder() ਅਚਲ {
    ਵਾਪਸ m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
