// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/core_visibility.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/runtime_attribute.hpp>

ਨਾਮਾਕਾਸ਼ ov {

/**
 * @brief Holds weightless caching attributes of a single constant.
 *
 * WeightlessCacheAttribute class represents runtime info attribute that holds
 * the values of original size of the constant in bytes and the binary offset of the
 * constant's data in the weights file used by the weightless caching mechanism. It's
 * not copyable in case the data was changed (the original node was replaced by a new
 * one produced during the tranformation pipeline) - in that case weightless caching
 * can't be used for that constant.
 */
ਵਰਗ OPENVINO_API WeightlessCacheAttribute : ਜਨਤਕ RuntimeAttribute {
ਜਨਤਕ:
    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute)

    WeightlessCacheAttribute() = ਮਿਟਾਓ;

    WeightlessCacheAttribute(size_t original_ਮਾਪ, size_t bin_ਵਿੱਥ, ov::element::Type original_dtype)
        : original_ਮਾਪ(original_ਮਾਪ),
          bin_ਵਿੱਥ(bin_ਵਿੱਥ),
          original_dtype(original_dtype) {}

    ਤਾਰਕਿਕ is_copyable() ਅਚਲ override;

    size_t original_ਮਾਪ;
    size_t bin_ਵਿੱਥ;
    ov::element::Type original_dtype;
};

}  // namespace ov
