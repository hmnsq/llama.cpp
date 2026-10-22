#आशय once

#समावेश "mark_decompression_convert_constant_folding.h"
#समावेश "openvino/core/visibility.hpp"
#समावेश "openvino/pass/matcher_pass.hpp"

#अगर_घोषित OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#अन्यथा
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    अन्यथा
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#पूर्ण_अगर      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : खुला MatcherPass {
खुला:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
