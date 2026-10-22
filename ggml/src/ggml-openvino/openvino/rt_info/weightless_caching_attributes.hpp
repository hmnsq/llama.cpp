// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <openvino/core/core_visibility.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/runtime_attribute.hpp>

ನಾಮಾಕಾಶ ov {

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
ವರ್ಗ OPENVINO_API WeightlessCacheAttribute : ಸಾರ್ವಜನಿಕ RuntimeAttribute {
ಸಾರ್ವಜನಿಕ:
    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute)

    WeightlessCacheAttribute() = ಅಳಿಸು;

    WeightlessCacheAttribute(size_t original_ಗಾತ್ರ, size_t bin_ಸ್ಥಾನಾಂತರ, ov::element::Type original_dtype)
        : original_ಗಾತ್ರ(original_ಗಾತ್ರ),
          bin_ಸ್ಥಾನಾಂತರ(bin_ಸ್ಥಾನಾಂತರ),
          original_dtype(original_dtype) {}

    ತಾರ್ಕಿಕ is_copyable() ಸ್ಥಿರಾಂಕ override;

    size_t original_ಗಾತ್ರ;
    size_t bin_ಸ್ಥಾನಾಂತರ;
    ov::element::Type original_dtype;
};

}  // namespace ov
