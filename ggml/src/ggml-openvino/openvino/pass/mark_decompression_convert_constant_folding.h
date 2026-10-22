#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "mark_decompression_convert_constant_folding.h"
#ಅನ್ತರ್ಭಾವ "openvino/core/visibility.hpp"
#ಅನ್ತರ್ಭಾವ "openvino/pass/matcher_pass.hpp"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#ಇಲ್ಲದಿದ್ದರೆ
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    ಇಲ್ಲದಿದ್ದರೆ
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#ಆದರೆ_ಅಂತ್ಯ      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : ಸಾರ್ವಜನಿಕ MatcherPass {
ಸಾರ್ವಜನಿಕ:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
