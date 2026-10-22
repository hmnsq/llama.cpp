#pragma once

#ಅನ್ತರ್ಭಾವ "mark_decompression_convert_constant_folding.h"
#ಅನ್ತರ್ಭಾವ "openvino/core/visibility.hpp"
#ಅನ್ತರ್ಭಾವ "openvino/pass/matcher_pass.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#ಅನ್ಯಥಾ
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    ಅನ್ಯಥಾ
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY

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
