#அந்தர்பாவ "input_model.h"

#அந்தர்பாவ "decoder.h"

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {

உள்ளீடு_Model::உள்ளீடு_Model(மாறிலி std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

மாறிலி std::shared_ptr<GgmlDecoder> & உள்ளீடு_Model::get_model_decoder() மாறிலி {
    திரும்பு m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
