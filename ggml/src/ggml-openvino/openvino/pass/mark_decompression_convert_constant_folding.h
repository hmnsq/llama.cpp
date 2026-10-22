#pragma once

#અન્તર્ભાવ "mark_decompression_convert_constant_folding.h"
#અન્તર્ભાવ "openvino/core/visibility.hpp"
#અન્તર્ભાવ "openvino/pass/matcher_pass.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#નહીંતર
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    નહીંતર
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : જાહેર MatcherPass {
જાહેર:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
