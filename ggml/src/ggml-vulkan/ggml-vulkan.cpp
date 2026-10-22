#ಅನ್ತರ್ಭಾವ "ggml-vulkan-common.h"

ಸ್ಥಿರ vk_device_architecture get_device_architecture(ಧ್ರುವ vk::PhysicalDevice& device) {
    vk::PhysicalDeviceProperties props = device.getProperties();

    ಯದಿ (props.vendorID == VK_VENDOR_ID_AMD) {
        ಧ್ರುವ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ತಾರ್ಕಿಕ amd_shader_core_properties = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ integer_dot_product = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ subgroup_size_control = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ shader_float8 = ಅಸತ್ಯ;

        ಪ್ರತಿ (ಧ್ರುವ auto& properties : ext_props) {
            ಯದಿ (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_product = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_size_control = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_EXT_shader_float8", properties.extensionName) == 0) {
                shader_float8 = ಸತ್ಯ;
            }
        }

        ಯದಿ (!amd_shader_core_properties || !integer_dot_product || !subgroup_size_control) {
            ಪ್ರತ್ಯಾಗಮ vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_props;
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;

        props2.pNext = &shader_core_props_amd;
        shader_core_props_amd.pNext = &integer_dot_props;
        integer_dot_props.pNext = &subgroup_size_control_props;

        device.getProperties2(&props2);

        ಯದಿ (subgroup_size_control_props.maxSubgroupSize == 64 && subgroup_size_control_props.minSubgroupSize == 64) {
            ಪ್ರತ್ಯಾಗಮ vk_device_architecture::AMD_GCN;
        }
        ಯದಿ (subgroup_size_control_props.maxSubgroupSize == 64 && subgroup_size_control_props.minSubgroupSize == 32) {
            // RDNA
            ಯದಿ (shader_core_props_amd.wavefrontsPerSimd == 20) {
                ಪ್ರತ್ಯಾಗಮ vk_device_architecture::AMD_RDNA1;
            }
            ಯದಿ (shader_float8) {
                ಪ್ರತ್ಯಾಗಮ vk_device_architecture::AMD_RDNA4;
            }
            ಯದಿ (integer_dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated) {
                ಪ್ರತ್ಯಾಗಮ vk_device_architecture::AMD_RDNA3;
            }
            ಪ್ರತ್ಯಾಗಮ vk_device_architecture::AMD_RDNA2;
        }
    } ಅನ್ಯಥಾ ಯದಿ (props.vendorID == VK_VENDOR_ID_INTEL) {
        ಧ್ರುವ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ತಾರ್ಕಿಕ subgroup_size_control = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ integer_dot_product = ಅಸತ್ಯ;

        ಪ್ರತಿ (ಧ್ರುವ auto& properties : ext_props) {
            ಯದಿ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_size_control = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_product = ಸತ್ಯ;
            }
        }

        ಯದಿ (!subgroup_size_control || !integer_dot_product) {
            ಪ್ರತ್ಯಾಗಮ vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_props;

        props2.pNext = &subgroup_size_control_props;
        subgroup_size_control_props.pNext = &integer_dot_props;
        device.getProperties2(&props2);

        ಯದಿ (subgroup_size_control_props.minSubgroupSize == 16) {
            // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
            // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
            // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
            // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
            ಪ್ರತ್ಯಾಗಮ vk_device_architecture::INTEL_XE2;
        } ಅನ್ಯಥಾ ಯದಿ (subgroup_size_control_props.minSubgroupSize == 8 &&
                 integer_dot_product && integer_dot_props.integerDotProduct4x8BitPackedSignedAccelerated) {
            ಪ್ರತ್ಯಾಗಮ vk_device_architecture::INTEL_XE1;
        }
    } ಅನ್ಯಥಾ ಯದಿ (props.vendorID == VK_VENDOR_ID_NVIDIA) {
        ಧ್ರುವ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ತಾರ್ಕಿಕ cooperative_matrix = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ sm_builtins = ಅಸತ್ಯ;

        // Detect "pre-turing" based on lack of coopmat support.
        ಪ್ರತಿ (ಧ್ರುವ auto& properties : ext_props) {
            ಯದಿ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_matrix = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = ಸತ್ಯ;
            }
        }

        ಯದಿ (!cooperative_matrix) {
            ಪ್ರತ್ಯಾಗಮ vk_device_architecture::NVIDIA_PRE_TURING;
        }

        ಯದಿ (sm_builtins) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;

            props2.pNext = &sm_props;

            device.getProperties2(&props2);

            // Turing has 32, following architectures have 48
            ಯದಿ (sm_props.shaderWarpsPerSM == 32) {
                ಪ್ರತ್ಯಾಗಮ vk_device_architecture::NVIDIA_TURING;
            }
        }
    } ಅನ್ಯಥಾ ಯದಿ(props.vendorID == VK_VENDOR_ID_QUALCOMM){
        ಧ್ರುವ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ತಾರ್ಕಿಕ cooperative_matrix = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ cooperative_matrix_conversion = ಅಸತ್ಯ;

        ಪ್ರತಿ (ಧ್ರುವ auto& properties : ext_props) {
            ಯದಿ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_matrix = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_QCOM_cooperative_matrix_conversion", properties.extensionName) == 0) {
                cooperative_matrix_conversion = ಸತ್ಯ;
            }
        }

        ಯದಿ (cooperative_matrix && cooperative_matrix_conversion) {
            ಪ್ರತ್ಯಾಗಮ vk_device_architecture::QUALCOMM_ADRENO;
        }
    }
    ಪ್ರತ್ಯಾಗಮ vk_device_architecture::OTHER;
}

ತಾರ್ಕಿಕ ggml_vk_lightning_indexer_k_type_supported(ggml_type type) {
    ಪ್ರತ್ಯಾಗಮ std::find(lightning_indexer_k_types.begin(), lightning_indexer_k_types.end(), type) != lightning_indexer_k_types.end();
}
ಶೂನ್ಯ ggml_vk_print_device_fault_info(ಧ್ರುವ vk_device& device) {
    ಯದಿ (!device->device_fault || !device->pfn_vkGetDeviceFaultInfoEXT) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    VkDeviceFaultCountsEXT fault_counts {};
    fault_counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    VkResult res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, nullptr);
    ಯದಿ (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", res);
        ಪ್ರತ್ಯಾಗಮ;
    }

    std::vector<VkDeviceFaultAddressInfoEXT> address_infos(fault_counts.address_Info_ಎಣಿಕೆ);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(fault_counts.vendor_Info_ಎಣಿಕೆ);

    VkDeviceFaultInfoEXT fault_info {};
    fault_info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    fault_info.pAddressInfos = address_infos.data();
    fault_info.pVendorInfos = vendor_infos.data();

    res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, &fault_info);
    ಯದಿ (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", res);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (fault_counts.address_Info_ಎಣಿಕೆ == 0 && fault_counts.vendor_Info_ಎಣಿಕೆ == 0 && fault_info.description[0] == '\0') {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (fault_info.description[0] != '\0') {
        GGML_LOG_ERROR("ggml_vulkan: device fault on %s: %s\n", device->name.c_str(), fault_info.description);
    }

    ಪ್ರತಿ (uint32_t i = 0; i < fault_counts.address_Info_ಎಣಿಕೆ; i++) {
        ಧ್ರುವ auto& info = address_infos[i];
        GGML_LOG_CONT("  address fault %u: type=%d address=0x%llx precision=0x%llx\n",
                i, (ಪೂರ್ಣಾಙ್ಕ)info.addressType,
                (unsigned long long)info.reportedAddress,
                (unsigned long long)info.addressPrecision);
    }
    ಪ್ರತಿ (uint32_t i = 0; i < fault_counts.vendor_Info_ಎಣಿಕೆ; i++) {
        ಧ್ರುವ auto& info = vendor_infos[i];
        GGML_LOG_CONT("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n",
                i, info.description,
                (unsigned long long)info.vendorFaultCode,
                (unsigned long long)info.vendorFaultData);
    }
}
uint64_t ggml_vk_get_ಗಂಟು_flops(ಧ್ರುವ ggml_tensor * ಗಂಟು) {
    ಯದಿ (ಗಂಟು->op == GGML_OP_MUL_MAT || ಗಂಟು->op == GGML_OP_MUL_MAT_ID) {
        ಧ್ರುವ uint64_t m     = ಗಂಟು->ne[0];
        ಧ್ರುವ uint64_t n     = ಗಂಟು->ne[1];
        ಧ್ರುವ uint64_t k     = ಗಂಟು->src[1]->ne[0];
        ಧ್ರುವ uint64_t batch = ಗಂಟು->ne[2] * ಗಂಟು->ne[3];
        ಪ್ರತ್ಯಾಗಮ m * n * (k + (k - 1)) * batch;
    }
    ಯದಿ (ಗಂಟು->op == GGML_OP_CONV_2D || ಗಂಟು->op == GGML_OP_CONV_TRANSPOSE_2D) {
        ಧ್ರುವ ggml_tensor * knl = ಗಂಟು->src[0];
        ಧ್ರುವ uint64_t Cout  = ಗಂಟು->ne[2];
        ಧ್ರುವ uint64_t size_K = ಗಂಟು->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        ಧ್ರುವ uint64_t size_N = ಗಂಟು->ne[3] * ಗಂಟು->ne[0] * ಗಂಟು->ne[1];
        ಪ್ರತ್ಯಾಗಮ Cout * size_N * (size_K + (size_K - 1));
    }
    ಯದಿ (ಗಂಟು->op == GGML_OP_CONV_3D) {
        ಧ್ರುವ ggml_tensor * knl = ಗಂಟು->src[0];
        ಧ್ರುವ uint64_t OC     = ggml_get_op_params_i32(ಗಂಟು, 11);
        ಧ್ರುವ uint64_t IC     = ggml_get_op_params_i32(ಗಂಟು, 9);
        ಧ್ರುವ uint64_t size_K = IC * knl->ne[0] * knl->ne[1] * knl->ne[2];
        ಧ್ರುವ uint64_t size_N = ಗಂಟು->ne[3] / OC * ಗಂಟು->ne[0] * ಗಂಟು->ne[1] * ಗಂಟು->ne[2];
        ಪ್ರತ್ಯಾಗಮ OC * size_N * (size_K + (size_K - 1));
    }
    ಯದಿ (ಗಂಟು->op == GGML_OP_FLASH_ATTN_EXT) {
        ಧ್ರುವ ggml_tensor * q = ಗಂಟು->src[0];
        ಧ್ರುವ ggml_tensor * k = ಗಂಟು->src[1];
        ಧ್ರುವ ggml_tensor * v = ಗಂಟು->src[2];
        ಪ್ರತ್ಯಾಗಮ 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
    }
    ಪ್ರತ್ಯಾಗಮ 0;
}
ಶೂನ್ಯ ggml_vk_print_ಗಂಟು_list(ಧ್ರುವ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ start, ಪೂರ್ಣಾಙ್ಕ end) {
    uint64_t total_flops = 0;
    ಪೂರ್ಣಾಙ್ಕ n_ops = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = start; j <= end && j < cgraph->n_nodes; j++) {
        uint64_t flops = ggml_vk_get_ಗಂಟು_flops(cgraph->nodes[j]);
        total_flops += flops;
        n_ops++;
        ಯದಿ (flops > 0) {
            GGML_LOG_CONT("  node %d: %s (%s) [%.2f GFLOP]\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op),
                    flops / 1e9);
        } ಅನ್ಯಥಾ {
            GGML_LOG_CONT("  node %d: %s (%s)\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op));
        }
    }
    GGML_LOG_CONT("  total: %d ops, %.2f GFLOP\n", n_ops, total_flops / 1e9);
}
ಶೂನ್ಯ ggml_vk_print_device_lost_info(ಧ್ರುವ vk_device& device) {
    ggml_vk_print_device_fault_info(device);
    ಯದಿ (device->serialize_submissions && device->diag_cgraph != nullptr && device->diag_prev_start >= 0) {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n",
                device->name.c_str(), device->diag_prev_start, device->diag_prev_end);
        ggml_vk_print_ಗಂಟು_list(device->diag_cgraph, device->diag_prev_start, device->diag_prev_end);
    } ಅನ್ಯಥಾ {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s\n", device->name.c_str());
    }
}
ಶೂನ್ಯ * ಧ್ರುವ vk_ptr_base = (ಶೂನ್ಯ *)(uintptr_t) 0x1000;  // NOLINT

uint64_t vk_tensor_offset(ಧ್ರುವ ggml_tensor * tensor) {
    ಯದಿ (tensor->view_src) {
        ಪ್ರತ್ಯಾಗಮ (uint8_t *) tensor->view_src->data - (uint8_t *) vk_ptr_base;
    }
    ಪ್ರತ್ಯಾಗಮ (uint8_t *) tensor->data - (uint8_t *) vk_ptr_base;
}

size_t ggml_vk_tensor_buffer_offset(ಧ್ರುವ ggml_backend_vk_context * ctx, ಧ್ರುವ ggml_tensor * t) {
    // vk_tensor_offset() is relative to vk_ptr_base, but mapped host tensors need an offset relative to their Vulkan buffer.
    ಯದಿ (ctx->device->uma) {
        vk_buffer buf = nullptr;
        size_t off = 0;
        ggml_vk_host_get(ctx->device, t->data, buf, off);
        ಯದಿ (buf) {
            ಪ್ರತ್ಯಾಗಮ off;
        }
    }
    ಪ್ರತ್ಯಾಗಮ (size_t)(vk_tensor_offset(t) + t->view_offs);
}
size_t ggml_vk_descriptor_offset(size_t tensor_offset, size_t alignment, size_t type_size) {
    // Move the descriptor back until its distance to the tensor is divisible by the tensor type size.
    size_t descriptor_offset = tensor_offset & ~(alignment - 1);
    ಯಾವತ್ ((tensor_offset - descriptor_offset) % type_size != 0) {
        GGML_ASSERT(descriptor_offset >= alignment);
        descriptor_offset -= alignment;
    }

    ಪ್ರತ್ಯಾಗಮ descriptor_offset;
}
uint32_t get_misalign_bytes(ಧ್ರುವ ggml_backend_vk_context * ctx, ಧ್ರುವ ggml_tensor * t) {
    ಧ್ರುವ size_t tensor_offset = ggml_vk_tensor_buffer_offset(ctx, t);
    ಧ್ರುವ size_t descriptor_offset = ggml_vk_descriptor_offset(
        tensor_offset, ctx->device->properties.limits.minStorageBufferOffsetAlignment, ggml_type_size(t->type));
    GGML_ASSERT(tensor_offset - descriptor_offset <= UINT32_MAX);
    ಪ್ರತ್ಯಾಗಮ tensor_offset - descriptor_offset;
}

uint32_t ggml_vk_concat_unit_size(ggml_type type) {
    ಧ್ರುವ uint32_t type_size = ggml_type_size(type);

    ಯದಿ (!ggml_is_quantized(type)) {
        ಪ್ರತ್ಯಾಗಮ type_size;
    }

    // Use the widest existing concat shader that evenly divides a quant block.
    ಯದಿ (type_size % 8 == 0) {
        ಪ್ರತ್ಯಾಗಮ 8;
    }
    ಯದಿ (type_size % 4 == 0) {
        ಪ್ರತ್ಯಾಗಮ 4;
    }
    ಯದಿ (type_size % 2 == 0) {
        ಪ್ರತ್ಯಾಗಮ 2;
    }
    ಪ್ರತ್ಯಾಗಮ 1;
}
ತಾರ್ಕಿಕ ggml_vk_concat_supported(ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * dst) {
    ಯದಿ (src0->type != src1->type || src0->type != dst->type) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_quantized(src0->type)) {
        ಧ್ರುವ size_t type_size = ggml_type_size(src0->type);
        ಪ್ರತ್ಯಾಗಮ type_size == 1 || type_size == 2 || type_size == 4 || type_size == 8;
    }

    // Quantized tensor rows are block-aligned when created.
    ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(src1) && ggml_is_contiguous_rows(dst);
}
ಸ್ಥಿರ ತಾರ್ಕಿಕ vk_instance_initialized = ಅಸತ್ಯ;

vk_instance_t vk_instance;

ಸ್ಥಿರ VkDeviceSize ggml_vk_get_max_buffer_range(ಧ್ರುವ ggml_backend_vk_context * ctx, ಧ್ರುವ vk_buffer &buf, ಧ್ರುವ VkDeviceSize offset) {
    ಧ್ರುವ VkDeviceSize range = std::min(VkDeviceSize{buf->size - offset},
                                        VkDeviceSize{ctx->device->properties.limits.maxStorageBufferRange});
    ಪ್ರತ್ಯಾಗಮ range;
}

ಶೂನ್ಯ ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    ಯದಿ (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, ಸತ್ಯ, UINT64_MAX), "almost_ready_fence", ctx->device);
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = ಅಸತ್ಯ;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    vk::Result result;
    ಪ್ರತಿ (;;) {
        ಯತ್ನ {
            result = ctx->device->device.getFenceStatus(ctx->fence);
        } ಗೃಹಾಣ (vk::DeviceLostError &) {
            ggml_vk_print_device_lost_info(ctx->device);
            GGML_LOG_ERROR("ggml_vulkan: getFenceStatus at %s:%d\n", __FILE__, __LINE__);
            ಕ್ಷಿಪ;
        }
        ಯದಿ (result == vk::Result::eSuccess) {
            ಭಙ್ಗ;
        }
        ಯದಿ (result != vk::Result::eNotReady) {
            GGML_LOG_ERROR("ggml_vulkan: error %s at %s:%d\n", to_string(result).c_str(), __FILE__, __LINE__);
            ಕ್ಷಿಪ vk::SystemError(vk::make_error_code(result), "ggml_vulkan: getFenceStatus");
        }
        ಪ್ರತಿ (uint32_t i = 0; i < 100; ++i) {
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
        }
    }
    ctx->device->device.resetFences({ ctx->fence });
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_strip_decode_vector(ಧ್ರುವ uint32_t * code, size_t word_ಎಣಿಕೆ, std::vector<uint32_t> & out) {
    ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ kDecodeVectorExt[] = "SPV_NV_cooperative_matrix_decode_vector";

    ಯದಿ (word_ಎಣಿಕೆ < 5) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ uses_decode_vector = ಅಸತ್ಯ;
    ಪ್ರತಿ (size_t pos = 5; pos < word_ಎಣಿಕೆ; ) {
        uint32_t word = code[pos];
        uint32_t wc   = word >> spv::Word_ಎಣಿಕೆ_Shift;
        uint32_t op   = word & spv::OpCodeMask;
        GGML_ASSERT(wc > 0 && pos + wc <= word_ಎಣಿಕೆ);
        ಯದಿ (op == spv::OpExtension && wc >= 2) {
            ಧ್ರುವ ವರ್ಣ * s = reinterpret_cast<ಧ್ರುವ ವರ್ಣ *>(&code[pos + 1]);
            ಯದಿ (strcmp(s, kDecodeVectorExt) == 0) {
                uses_decode_vector = ಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
        pos += wc;
    }

    ಯದಿ (!uses_decode_vector) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    VK_LOG_DEBUG("ggml_vk_strip_decode_vector: stripping SPV_NV_cooperative_matrix_decode_vector");

    // Bulk-copy unchanged runs and only break the run when an instruction needs to
    // be dropped or patched. Use reserve + insert/push_back so the destination buffer
    // is touched exactly once (no zero-initialization pass from resize()).
    out.clear();
    out.reserve(word_ಎಣಿಕೆ);

    size_t run_start = 0;
    auto flush_run = [&](size_t up_to) {
        ಯದಿ (up_to > run_start) {
            out.insert(out.end(), code + run_start, code + up_to);
        }
    };

    ಪ್ರತಿ (size_t pos = 5; pos < word_ಎಣಿಕೆ; ) {
        uint32_t word = code[pos];
        uint32_t wc   = word >> spv::Word_ಎಣಿಕೆ_Shift;
        uint32_t op   = word & spv::OpCodeMask;
        GGML_ASSERT(wc > 0 && pos + wc <= word_ಎಣಿಕೆ);

        ಯದಿ (op == spv::OpExtension && wc >= 2) {
            ಧ್ರುವ ವರ್ಣ * s = reinterpret_cast<ಧ್ರುವ ವರ್ಣ *>(&code[pos + 1]);
            ಯದಿ (strcmp(s, kDecodeVectorExt) == 0) {
                flush_run(pos);
                pos += wc;
                run_start = pos;
                ಅನುವರ್ತ;
            }
        }

        ಯದಿ (op == spv::OpCapability && wc == 2 && code[pos + 1] == kSpvCapabilityCooperativeMatrixDecodeVectorNV) {
            flush_run(pos);
            pos += wc;
            run_start = pos;
            ಅನುವರ್ತ;
        }

        ಯದಿ (op == kSpvOpCooperativeMatrixLoadTensorNV) {
            // [opcode/wc][ResultType][Result][Pointer][Object][TensorLayout][MemOperand mask][mem extras...][TA mask][ta extras...]
            GGML_ASSERT(wc >= 8);

            uint32_t mem_mask = code[pos + 6];
            size_t   cur      = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += (mem_mask & 0x2)     ? 1 : 0; // Aligned
            cur += (mem_mask & 0x8)     ? 1 : 0; // MakePointerAvailable
            cur += (mem_mask & 0x10)    ? 1 : 0; // MakePointerVisible
            cur += (mem_mask & 0x10000) ? 1 : 0; // AliasScopeINTELMask
            cur += (mem_mask & 0x20000) ? 1 : 0; // NoAliasINTELMask
            GGML_ASSERT(cur < pos + wc);

            uint32_t ta_mask = code[cur];
            ಯದಿ ((ta_mask & kSpvTensorAddressingDecodeVectorFuncBit) == 0) {
                pos += wc;
                ಅನುವರ್ತ; // leave instruction inside the current unchanged run
            }

            flush_run(pos);

            // Append unchanged prefix of the instruction (header through the mem-extras).
            size_t inst_start = out.size();
            size_t pre_n      = cur - pos;
            out.insert(out.end(), code + pos, code + pos + pre_n);

            // Emit TA mask with the DecodeVectorFunc bit cleared.
            out.push_back(ta_mask & ~kSpvTensorAddressingDecodeVectorFuncBit);

            // TA extras: TensorView (0x1) and DecodeFunc (0x2) are kept verbatim;
            // DecodeVectorFunc (0x4) is dropped along with its trailing id operand.
            size_t keep_ta_extras = ((ta_mask & 0x1) ? 1 : 0) + ((ta_mask & 0x2) ? 1 : 0);
            ಯದಿ (keep_ta_extras) {
                out.insert(out.end(), code + cur + 1, code + cur + 1 + keep_ta_extras);
            }

            GGML_ASSERT(wc == pre_n + 1 + keep_ta_extras + 1);

            // Patch the instruction header with the new (one-shorter) word count.
            uint32_t new_wc = wc - 1;
            out[inst_start] = (new_wc << spv::Word_ಎಣಿಕೆ_Shift) | op;

            pos += wc;
            run_start = pos;
            ಅನುವರ್ತ;
        }

        pos += wc;
    }

    flush_run(word_ಎಣಿಕೆ);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_roll_bk_loop(ಧ್ರುವ uint32_t * code, size_t word_ಎಣಿಕೆ, std::vector<uint32_t> & out) {
    ಯದಿ (word_ಎಣಿಕೆ < 5) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಸಂರಚನಾ vk_spv_loop {
        size_t   header;
        size_t   end;
        uint32_t control;
    };

    std::vector<vk_spv_loop> loops;

    // Collect a list of all loops in the module.
    ಪ್ರತಿ (size_t pos = 5; pos < word_ಎಣಿಕೆ; ) {
        ಧ್ರುವ uint32_t wc = code[pos] >> spv::Word_ಎಣಿಕೆ_Shift;
        ಧ್ರುವ uint32_t op = code[pos] & spv::OpCodeMask;
        ಯದಿ (wc == 0 || pos + wc > word_ಎಣಿಕೆ) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (op == spv::OpLoopMerge && wc >= 4) { loops.push_back({ pos, 0, code[pos + 3] }); }

        ಯದಿ (op == spv::OpLabel && wc >= 2) {
            ಪ್ರತಿ (auto & l : loops) {
                ಯದಿ (l.end == 0 && code[l.header + 1] == code[pos + 1]) { l.end = pos; }
            }
        }

        pos += wc;
    }

    auto encloses = [](ಧ್ರುವ vk_spv_loop & a, ಧ್ರುವ vk_spv_loop & b) {
        ಪ್ರತ್ಯಾಗಮ a.header < b.header && b.header < a.end;
    };

    // Find the BK loop.
    ಧ್ರುವ vk_spv_loop * bk = nullptr;
    ಪ್ರತಿ (ಧ್ರುವ auto & h : loops) {
        ಯದಿ (h.control != spv::LoopControlUnrollMask) {
            ಅನುವರ್ತ;
        }
        ಧ್ರುವ vk_spv_loop * parent = nullptr;
        ತಾರ್ಕಿಕ has_child = ಅಸತ್ಯ;
        ಪ್ರತಿ (ಧ್ರುವ auto & g : loops) {
            ಯದಿ (encloses(g, h) && (!parent || g.header > parent->header)) {
                parent = &g;
            }
            ಯದಿ (encloses(h, g)) {
                has_child = ಸತ್ಯ;
            }
        }
        // BK loop should be the last loop nested inside the loop with no hint
        // and have at least one child loop.
        ಯದಿ (parent &&
            parent->control == spv::LoopControlMaskNone &&
            has_child &&
            (!bk || h.header > bk->header)) {
            bk = &h;
        }
    }
    ಯದಿ (!bk) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // set DontUnroll instead of Unroll
    out.assign(code, code + word_ಎಣಿಕೆ);
    out[bk->header + 3] = spv::LoopControlDontUnrollMask;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_size, ಧ್ರುವ ಶೂನ್ಯ* spv_data, ಧ್ರುವ std::string entrypoint,
                                         uint32_t parameter_ಎಣಿಕೆ, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         ತಾರ್ಕಿಕ disable_robustness, ತಾರ್ಕಿಕ require_full_subgroups, uint32_t required_subgroup_size) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_ಎಣಿಕೆ <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_full_subgroups << ", " << required_subgroup_size << ")");
    GGML_ASSERT(parameter_ಎಣಿಕೆ > 0);
    GGML_ASSERT(parameter_ಎಣಿಕೆ <= MAX_PARAMETER_ಎಣಿಕೆ);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, reinterpret_cast<ಧ್ರುವ uint32_t *>(spv_data));

    // Patch SPIR-V to enable supported FP16 float controls, avoiding the need
    // for separate shader variants.
    std::vector<uint32_t> spirv;
    ಯದಿ (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {
        ಧ್ರುವ uint32_t* spv_ಪದಗಳು = reinterpret_cast<ಧ್ರುವ uint32_t *>(spv_data);
        size_t word_ಎಣಿಕೆ = spv_size / ಪರಿಮಾಣ(uint32_t);
        spirv.assign(spv_ಪದಗಳು, spv_ಪದಗಳು + word_ಎಣಿಕೆ);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        size_t pos = 5; // skip header
        size_t cap_ಸೇರಿಸು_pos = pos;
        size_t ext_ಸೇರಿಸು_pos = pos;
        size_t exec_ಸೇರಿಸು_pos = pos;
        uint32_t entry_point_id = 0;

        ಯಾವತ್ (pos < spirv.size()) {
            uint32_t opcode = spirv[pos] & spv::OpCodeMask;
            uint32_t len    = spirv[pos] >> spv::Word_ಎಣಿಕೆ_Shift;
            ಯದಿ (len == 0) ಭಙ್ಗ;

            ಯದಿ (opcode == spv::OpCapability) {
                cap_ಸೇರಿಸು_pos = pos + len;
                ext_ಸೇರಿಸು_pos = pos + len;
            } ಅನ್ಯಥಾ ಯದಿ (opcode == spv::OpExtension) {
                ext_ಸೇರಿಸು_pos = pos + len;
            } ಅನ್ಯಥಾ ಯದಿ (opcode == spv::OpEntryPoint) {
                entry_point_id = spirv[pos + 2];
                exec_ಸೇರಿಸು_pos = pos + len;
            } ಅನ್ಯಥಾ ಯದಿ (opcode == spv::OpExecutionMode || opcode == spv::OpExecutionModeId) {
                exec_ಸೇರಿಸು_pos = pos + len;
            } ಅನ್ಯಥಾ ಯದಿ (entry_point_id != 0) {
                ಭಙ್ಗ;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        ಯದಿ (device->float_controls_rte_fp16) {
            // OpExecutionMode %entrypoint RoundingModeRTE 16
            uint32_t exec_mode[] = { (4u << spv::Word_ಎಣಿಕೆ_Shift) | spv::OpExecutionMode, entry_point_id, spv::ExecutionModeRoundingModeRTE, 16 };
            spirv.insert(spirv.begin() + exec_ಸೇರಿಸು_pos, std::begin(exec_mode), std::end(exec_mode));
        }

        ಯದಿ (device->float_controls_denorm_preserve_fp16) {
            // OpExecutionMode %entrypoint DenormPreserve 16
            uint32_t exec_mode[] = { (4u << spv::Word_ಎಣಿಕೆ_Shift) | spv::OpExecutionMode, entry_point_id, spv::ExecutionModeDenormPreserve, 16 };
            spirv.insert(spirv.begin() + exec_ಸೇರಿಸು_pos, std::begin(exec_mode), std::end(exec_mode));
        }

        // OpExtension "SPV_KHR_float_controls"
        ಧ್ರುವ ವರ್ಣ ext_str[] = "SPV_KHR_float_controls";
        size_t ext_str_ಪದಗಳು = CEIL_DIV(ಪರಿಮಾಣ(ext_str), ಪರಿಮಾಣ(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_ಪದಗಳು, 0);
        extension[0] = (uint32_t)((1 + ext_str_ಪದಗಳು) << spv::Word_ಎಣಿಕೆ_Shift) | spv::OpExtension;
        memcpy(&extension[1], ext_str, ಪರಿಮಾಣ(ext_str));
        spirv.insert(spirv.begin() + ext_ಸೇರಿಸು_pos, extension.begin(), extension.end());

        ಯದಿ (device->float_controls_rte_fp16) {
            // OpCapability RoundingModeRTE
            uint32_t capability[] = { (2u << spv::Word_ಎಣಿಕೆ_Shift) | spv::OpCapability, spv::CapabilityRoundingModeRTE };
            spirv.insert(spirv.begin() + cap_ಸೇರಿಸು_pos, std::begin(capability), std::end(capability));
        }

        ಯದಿ (device->float_controls_denorm_preserve_fp16) {
            // OpCapability DenormPreserve
            uint32_t capability[] = { (2u << spv::Word_ಎಣಿಕೆ_Shift) | spv::OpCapability, spv::CapabilityDenormPreserve };
            spirv.insert(spirv.begin() + cap_ಸೇರಿಸು_pos, std::begin(capability), std::end(capability));
        }

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * ಪರಿಮಾಣ(uint32_t), spirv.data());
    }

#ಯದಿ defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
    ಯದಿ (device->coopmat2 && !device->coopmat2_decode_vector) {
        ಧ್ರುವ uint32_t * src   = spirv.empty() ? reinterpret_cast<ಧ್ರುವ uint32_t *>(spv_data) : spirv.data();
        size_t           src_n = spirv.empty() ? spv_size / ಪರಿಮಾಣ(uint32_t) : spirv.size();
        std::vector<uint32_t> stripped;
        ಯದಿ (ggml_vk_strip_decode_vector(src, src_n, stripped)) {
            spirv = std::move(stripped);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * ಪರಿಮಾಣ(uint32_t), spirv.data());
        }
    }
#endif

#ಯದಿ VK_HEADER_VERSION >= 287
    // Roll the mul_mm BK loop on Asahi Linux. Skip bf16 and the mul_mmq pipelines.
    ಯದಿ (device->driver_id == vk::DriverId::eMesaHoneykrisp &&
        pipeline->name.rfind("matmul", 0) == 0 &&
        pipeline->name.find("bf16") == std::string::npos &&
        pipeline->name.find("q8_1") == std::string::npos) {
        ಧ್ರುವ uint32_t * src   = spirv.empty() ? reinterpret_cast<ಧ್ರುವ uint32_t *>(spv_data) : spirv.data();
        size_t           src_n = spirv.empty() ? spv_size / ಪರಿಮಾಣ(uint32_t) : spirv.size();
        std::vector<uint32_t> rolled;
        ಯದಿ (ggml_vk_roll_bk_loop(src, src_n, rolled)) {
            spirv = std::move(rolled);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * ಪರಿಮಾಣ(uint32_t), spirv.data());
        }
    }
#endif

    ಯತ್ನ {
        pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);
    } ಗೃಹಾಣ (ಧ್ರುವ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: shader module creation failed for " << pipeline->name << ": " << e.what() << std::endl;
        ಕ್ಷಿಪ;
    }

    vk::ತಳ್ಳು_Constant_Range pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        pipeline->ತಳ್ಳು_constant_size
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    ಯತ್ನ {
        pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);
    } ಗೃಹಾಣ (ಧ್ರುವ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: pipeline layout creation failed for " << pipeline->name << ": " << e.what() << std::endl;
        ಕ್ಷಿಪ;
    }

    std::vector<vk::SpecializationMapEntry> specialization_entries(specialization_constants.size());

    ಪ್ರತಿ (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].offset = i * ಪರಿಮಾಣ(uint32_t);
        specialization_entries[i].size = ಪರಿಮಾಣ(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.data(),
        specialization_constants.size() * ಪರಿಮಾಣ(uint32_t),
        specialization_constants.data()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    ಯದಿ (device->subgroup_require_full_support && require_full_subgroups) {
        pipeline_shader_stage_create_flags |= vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::ShaderStageFlagBits::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pipeline_shader_stage_required_subgroup_size_create_info;
    pipeline_shader_stage_required_subgroup_size_create_info.requiredSubgroupSize = required_subgroup_size;
    ಯದಿ (device->subgroup_size_control && required_subgroup_size > 0) {
        GGML_ASSERT(device->subgroup_min_size <= required_subgroup_size && required_subgroup_size <= device->subgroup_max_size);
        pipeline_shader_create_info.setPNext(&pipeline_shader_stage_required_subgroup_size_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::PipelineCreateFlagBits::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    ಯದಿ (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
        rci.uniformBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
        compute_pipeline_create_info.setPNext(&rci);
    }

#ಯದಿ defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    ಯದಿ (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64BitIndexingEXT;
        ಯದಿ (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.setPNext(compute_pipeline_create_info.pNext);
        compute_pipeline_create_info.setPNext(&pipelineFlags2CreateInfo);
    }
#endif

    ಯತ್ನ {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } ಗೃಹಾಣ (ಧ್ರುವ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: compute pipeline creation failed for " << pipeline->name << ": " << e.what() << std::endl;
        ಕ್ಷಿಪ;
    }

    ಯದಿ (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vkSetDebugUtilsObjectNameEXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    ಯದಿ (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        auto statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);

        ತಾರ್ಕಿಕ print_stats = !vk_pipeline_stats_filter.empty() &&
                           pipeline->name.find(vk_pipeline_stats_filter) != std::string::npos;
        ಯದಿ (print_stats) {
            std::cerr << "ggml_vulkan: pipeline stats for " << pipeline->name << ":" << std::endl;
        }

        ಪ್ರತಿ (auto & s : statistics) {
            ಯದಿ (print_stats) {
                std::cerr << "ggml_vulkan:   " << s.name.data() << ": ";
                ವಿಕಲ್ಪ (s.format) {
                    ಪಕ್ಷ vk::PipelineExecutableStatisticFormatKHR::eBool32:
                        std::cerr << (s.value.b32 ? "true" : "false");
                        ಭಙ್ಗ;
                    ಪಕ್ಷ vk::PipelineExecutableStatisticFormatKHR::eInt64:
                        std::cerr << s.value.i64;
                        ಭಙ್ಗ;
                    ಪಕ್ಷ vk::PipelineExecutableStatisticFormatKHR::eUint64:
                        std::cerr << s.value.u64;
                        ಭಙ್ಗ;
                    ಪಕ್ಷ vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                        std::cerr << s.value.f64;
                        ಭಙ್ಗ;
                }
                std::cerr << std::endl;
            }
            // "Register Count" is reported by NVIDIA drivers.
            ಯದಿ (strcmp(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->register_ಎಣಿಕೆ = (uint32_t)s.value.u64;
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = ಸತ್ಯ;
        pipeline->compile_pending = ಅಸತ್ಯ;
    }
    device->compile_cv.notify_all();
}

ಶೂನ್ಯ ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

ಶೂನ್ಯ ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_set_requirements += n;
    ಯದಿ (!pipeline->compiled) {
        ggml_vk_load_shaders(ctx->device, pipeline);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}

ಶೂನ್ಯ ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    ಯದಿ (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_set_requirements) {
        // Enough descriptors are available
        ಪ್ರತ್ಯಾಗಮ;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_set_requirements});
    uint32_t to_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_SIZE;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_SIZE;

    ಯಾವತ್ (to_alloc > 0) {
        ಧ್ರುವ uint32_t alloc_ಎಣಿಕೆ = std::min(pool_remaining, to_alloc);
        to_alloc -= alloc_ಎಣಿಕೆ;
        pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE;

        ಯದಿ (pool_idx >= ctx->descriptor_pools.size()) {
            vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, MAX_PARAMETER_ಎಣಿಕೆ * VK_DEVICE_DESCRIPTOR_POOL_SIZE);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_SIZE, descriptor_pool_size);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::DescriptorSetLayout> layouts(alloc_ಎಣಿಕೆ);
        ಪ್ರತಿ (uint32_t i = 0; i < alloc_ಎಣಿಕೆ; i++) {
            layouts[i] = device->dsl;
        }
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(ctx->descriptor_pools[pool_idx], alloc_ಎಣಿಕೆ, layouts.data());
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
    ctx->descriptor_set_bindings.resize(ctx->descriptor_sets.size());
}

ಸ್ಥಿರ vk_command_buffer* ggml_vk_create_cmd_buffer(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    vk::CommandBufferAllocateInfo command_buffer_alloc_info(
        p.pool,
        vk::CommandBufferLevel::ePrimary,
        1);
    ಧ್ರುವ std::vector<vk::CommandBuffer> cmd_buffers = device->device.allocateCommandBuffers(command_buffer_alloc_info);
    p.cmd_buffers.push_back({ cmd_buffers.front(), 0, ಸತ್ಯ });
    ಪ್ರತ್ಯಾಗಮ &p.cmd_buffers[p.cmd_buffers.size()-1];
}

ಶೂನ್ಯ ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    ಯದಿ (ctx->seqs.empty()) {
        ಯದಿ (fence) {
            ctx->p->q->handle->submit({}, fence);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    ಪೂರ್ಣಾಙ್ಕ idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    ಪ್ರತಿ (ಧ್ರುವ auto& sequence : ctx->seqs) {
        reserve += sequence.size();
    }

    // Pre-reserve vectors to prevent reallocation, which invalidates pointers
    tl_wait_semaphores.reserve(reserve);
    tl_wait_vals.reserve(reserve);
    tl_signal_semaphores.reserve(reserve);
    tl_signal_vals.reserve(reserve);
    tl_submit_infos.reserve(reserve);
    submit_infos.reserve(reserve);
    stage_flags.reserve(reserve);

    ಪ್ರತಿ (ಧ್ರುವ auto& sequence : ctx->seqs) {
        ಪ್ರತಿ (ಧ್ರುವ auto& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            ಪ್ರತಿ (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            ಪ್ರತಿ (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].data(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].data(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].pNext = nullptr;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].data(),
                stage_flags[idx].data(),
                1,
                &submission.buffer->buf,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].data(),
            };
            si.setPNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    ctx->p->q->handle->submit(submit_infos, fence);

    ctx->seqs.clear();
}

uint32_t ggml_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props, ಧ್ರುವ vk::QueueFlags& required, ಧ್ರುವ vk::QueueFlags& avoid, int32_t compute_index, uint32_t min_ಅಂಕೆ_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    ಧ್ರುವ uint32_t qfsize = queue_family_props.size();

    // Try with avoid preferences first
    ಪ್ರತಿ (uint32_t i = 0; i < qfsize; i++) {
        ಯದಿ (queue_family_props[i].queue_ಎಣಿಕೆ >= min_ಅಂಕೆ_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid)) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
    }

    // Fall back to only required
    ಪ್ರತಿ (size_t i = 0; i < qfsize; i++) {
        ಯದಿ (queue_family_props[i].queue_ಎಣಿಕೆ >= min_ಅಂಕೆ_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
    }

    // Fall back to reusing compute queue
    ಪ್ರತಿ (size_t i = 0; i < qfsize; i++) {
        ಯದಿ (queue_family_props[i].queue_ಎಣಿಕೆ >= min_ಅಂಕೆ_queues && queue_family_props[i].queueFlags & required) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
    }

    // Fall back to ignoring min_num_queries
    ಪ್ರತಿ (size_t i = 0; i < qfsize; i++) {
        ಯದಿ (queue_family_props[i].queueFlags & required) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    ಯದಿ (compute_index >= 0) {
        ಪ್ರತ್ಯಾಗಮ compute_index;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::endl;

    ಪ್ರತಿ(auto &q_family : queue_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.queue_ಎಣಿಕೆ) << " flags: " + to_string(q_family.queueFlags) << std::endl;
    }
    abort();
}

std::unique_ptr<vk_queue> ggml_vk_create_queue(vk_device& device, uint32_t queue_family_index, uint32_t queue_index, vk::PipelineStageFlags&& stage_flags, ತಾರ್ಕಿಕ transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    auto q = std::make_unique<vk_queue>();
    q->queue_family_index = queue_family_index;
    q->transfer_only = transfer_only;

    std::shared_ptr<vk_queue_handle> h;
    vk::DeviceQueueInfo2 queue_info2{};
    queue_info2.queueFamilyIndex = queue_family_index;
    queue_info2.queueIndex = queue_index;

    ಯದಿ (device->has_internally_synchronized_queues) {
        h = std::make_shared<vk_queue_handle_unsynchronized>();
        queue_info2.flags = eInternallySynchronizedKHR;
    } ಅನ್ಯಥಾ {
        h = std::make_shared<vk_queue_handle_synchronized>();
    }

    h->queue = device->device.getQueue2(queue_info2);
    h->device = device;
    // Avoid concurrent submissions on NVIDIA due to driver bug.
    ಯದಿ (device->vendor_id == VK_VENDOR_ID_NVIDIA) {
        h->device_submit_mutex = &device->queue_submit_mutex;
    }
    q->handle = h;

    q->cmd_pool.init(device, q.get());

    q->stage_flags = stage_flags;
    ಪ್ರತ್ಯಾಗಮ q;
}

std::unique_ptr<vk_queue> ggml_vk_create_aliased_queue(vk_device& device, ಧ್ರುವ std::unique_ptr<vk_queue>& source) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    auto q = std::make_unique<vk_queue>();
    q->handle = source->handle;
    q->queue_family_index = source->queue_family_index;
    q->stage_flags = source->stage_flags;
    q->transfer_only = source->transfer_only;
    q->cmd_pool.init(device, q.get());
    ಪ್ರತ್ಯಾಗಮ q;
}

vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context result = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << result << ")");
    ctx->gc.contexts.emplace_back(result);
    result->p = &p;
    ಪ್ರತ್ಯಾಗಮ result;
}

vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context result = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << result << ")");
    result->p = &p;
    ಪ್ರತ್ಯಾಗಮ result;
}

ಸ್ಥಿರ vk_semaphore * ggml_vk_create_ದ್ವಿಮಾನ_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::e_ದ್ವಿಮಾನ, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    ಪ್ರತ್ಯಾಗಮ &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

ಸ್ಥಿರ vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    ಯದಿ (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.setPNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    ಪ್ರತ್ಯಾಗಮ &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

ಸ್ಥಿರ vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    ಯದಿ (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    ಪ್ರತ್ಯಾಗಮ ctx->gc.events[ctx->event_idx++];
}

ಶೂನ್ಯ ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    // Don't clear the command buffers and mark them as not in use.
    // This allows us to reuse them
    ಪ್ರತಿ (auto& cmd_buffer : p.cmd_buffers) {
        cmd_buffer.in_use = ಅಸತ್ಯ;
    }
}

ಶೂನ್ಯ ggml_vk_queue_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    ಸ್ಥಿರ constexpr uint32_t cleanup_frequency = 10;

    ಯದಿ (device->compute_queue && device->compute_queue->cmd_pool.buffers_in_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_queue->cmd_pool);
    }
    ಯದಿ (device->transfer_queue && device->transfer_queue->cmd_pool.buffers_in_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_queue->cmd_pool);
    }
}

vk_subbuffer ggml_vk_subbuffer(ಧ್ರುವ ggml_backend_vk_context* ctx, ಧ್ರುವ vk_buffer& buf, size_t offset) {
    ಪ್ರತ್ಯಾಗಮ { buf, offset, ggml_vk_get_max_buffer_range(ctx, buf, offset) };
}

ಶೂನ್ಯ ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    ಧ್ರುವ ತಾರ್ಕಿಕ transfer_queue = subctx->p->q->transfer_only;

    ಯದಿ (ctx) {
        ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = ಅಸತ್ಯ;
    }

    subctx->s->buffer->buf.pipelineBarrier(
        subctx->p->q->stage_flags,
        subctx->p->q->stage_flags,
        {},
        { {
          { !transfer_queue ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) },
          { !transfer_queue ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) }
        } },
        {},
        {}
    );
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_reset_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->buffer->buf.resetEvent(
        event,
        ctx->p->q->stage_flags
    );
}

ಶೂನ್ಯ ggml_vk_set_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->buffer->buf.setEvent(
        event,
        ctx->p->q->stage_flags
    );
}

ಶೂನ್ಯ ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    ಯದಿ (events.empty()) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ctx->s->buffer->buf.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

ಸ್ಥಿರ vk_fa_tuning_params get_fa_tuning_params_scalar(ಧ್ರುವ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ತಾರ್ಕಿಕ f32acc) {

    vk_fa_tuning_params result{};
    result.path = FA_SCALAR;

    ಯದಿ (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        result.subgroup_size = 32;
        result.disable_subgroups = ಸತ್ಯ;
    } ಅನ್ಯಥಾ ಯದಿ (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        result.subgroup_size = n_rows < 4 ? 32 : device->subgroup_size;
    } ಅನ್ಯಥಾ {
        result.subgroup_size = device->subgroup_size;
    }

    // Row split splits the workgroup so that synchronization only has to happen within subgroups, which avoids barriers
    uint32_t row_split_max_hsk = 64;
    ಯದಿ (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        row_split_max_hsk = n_rows <= 8 ? 64 : 128;
    }
    result.row_split = (n_rows < 4 || hsk <= row_split_max_hsk) ? 1 : 4;

    ಯದಿ (result.subgroup_size > 32 && (n_rows < 4 || hsk < (result.row_split == 1 ? 128 : 64))) {
        result.workgroup_size = result.subgroup_size * 2;
    } ಅನ್ಯಥಾ {
        result.workgroup_size = result.subgroup_size * 4;
    }

    ಧ್ರುವ uint32_t D = hsk | hsv;

    ಧ್ರುವ ತಾರ್ಕಿಕ reduce_block_rows = D & 8 || n_kv < 1024 || device->vendor_id == VK_VENDOR_ID_INTEL;

    ಯದಿ (n_rows == 1) {
        result.block_rows = 1;
        result.block_cols = 64;
    } ಅನ್ಯಥಾ {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        ಯದಿ (result.row_split == 1) {
            result.block_rows = n_rows == 2 ? 2 : ((n_rows <= 4 || reduce_block_rows) ? 4 : 8);
        } ಅನ್ಯಥಾ {
            result.block_rows = n_rows <= 4 ? 4 : ((n_rows <= 8 || reduce_block_rows) ? 8 : 16);
        }

        result.block_cols = (D & 8) ? 64 : 32;
    }

    ಧ್ರುವ uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit

    result.d_split = std::min(std::min(result.subgroup_size, 8u), D_lsb / 4);

    result.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    ಯದಿ (!reduce_block_rows && !ggml_vk_flash_attn_scalar_shmem_support(device, result, hsk, hsv, f32acc, k_type, v_type)) {
        result.block_rows /= 2;
    }

    // On AMD RDNA, for small head sizes and big batch size the shader uses few registers, so too many subgroups get scheduled
    // at once and end up thrashing the cache. Fix this by setting a large (unused) shmem buffer that reduces occupancy.
    // This targets an occupancy of 4 subgroups per SIMD.
    ಯದಿ (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize == 65536) {
        ಯದಿ (device->architecture != AMD_GCN && n_rows >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            result.limit_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } ಅನ್ಯಥಾ ಯದಿ (device->architecture == AMD_GCN && n_rows <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            result.limit_occupancy_shmem = (n_rows < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸ್ಥಿರ vk_fa_tuning_params get_fa_tuning_params_coopmat1(ಧ್ರುವ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ತಾರ್ಕಿಕ f32acc) {
    GGML_UNUSED(n_rows);
    GGML_UNUSED(n_kv);
    GGML_UNUSED(k_type);
    GGML_UNUSED(v_type);
    GGML_UNUSED(f32acc);

    vk_fa_tuning_params result{};
    result.path = FA_COOPMAT1;

    ಧ್ರುವ uint32_t D = hsk | hsv;

    ಧ್ರುವ uint32_t coopmat_block_rows = 16;
    ಧ್ರುವ uint32_t coopmat_block_cols = 16;

    ಧ್ರುವ uint32_t ಅಂಕೆ_subgroups = 4;

    result.block_rows = coopmat_block_rows;
    result.block_cols = coopmat_block_cols * ಅಂಕೆ_subgroups;
    result.row_split = ಅಂಕೆ_subgroups;
    result.subgroup_size = device->subgroup_size;
    result.workgroup_size = ಅಂಕೆ_subgroups * result.subgroup_size;

    ಧ್ರುವ uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit
    result.d_split = std::min(std::min(result.subgroup_size, 8u), D_lsb / 4);

    result.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸ್ಥಿರ vk_fa_tuning_params get_fa_tuning_params_coopmat2(ಧ್ರುವ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ತಾರ್ಕಿಕ f32acc) {
    GGML_UNUSED(n_kv);
    GGML_UNUSED(f32acc);

    vk_fa_tuning_params result{};
    result.path = FA_COOPMAT2;

    ಧ್ರುವ uint32_t D = hsk | hsv;

    ಧ್ರುವ ತಾರ್ಕಿಕ small_rows = n_rows < 32;

    ಯದಿ (small_rows) {
        result.block_rows = 32;
        result.block_cols = 32;
    } ಅನ್ಯಥಾ ಯದಿ (ggml_is_quantized(k_type) || ggml_is_quantized(v_type) || hsk >= 256 || hsv >= 256) {
        result.block_rows = (hsk >= 512 || hsv >= 512) ? 32 : 64;
        result.block_cols = 32;
    } ಅನ್ಯಥಾ {
        result.block_rows = 64;
        result.block_cols = 64;
    }

    result.subgroup_size = device->subgroup_size;
    result.workgroup_size = (small_rows && (D % 32) == 0) ? 256 : 128;

    ಪ್ರತ್ಯಾಗಮ result;
}

vk_fa_tuning_params get_fa_tuning_params(ಧ್ರುವ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ತಾರ್ಕಿಕ f32acc) {
    FaCodePath path = device->coopmat2 ? FA_COOPMAT2 :
                      device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    ಯದಿ (path == FA_COOPMAT2 && k_type == GGML_TYPE_BF16 && !device->coopmat2_bf16_support) {
        path = FA_COOPMAT1;
    }
    ಯದಿ (path == FA_COOPMAT1 && k_type == GGML_TYPE_BF16 && !device->coopmat_bf16_support) {
        path = FA_SCALAR;
    }

    ಯದಿ (path == FA_COOPMAT1 && device->architecture == vk_device_architecture::NVIDIA_TURING) {
        // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
        path = FA_SCALAR;
    }

    ಯದಿ (path == FA_COOPMAT1) {
        ತಾರ್ಕಿಕ shape_ok = (f32acc && device->coopmat_support_16x16x16_f32acc) ||
                        (!f32acc && device->coopmat_support_16x16x16_f16acc);
        ಧ್ರುವ vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
        ತಾರ್ಕಿಕ shmem_ok = ggml_vk_flash_attn_coopmat_shmem_support(device, params, hsk, hsv, f32acc, k_type, v_type);

        ಯದಿ (!shape_ok || !shmem_ok) {
            path = FA_SCALAR;
        }
    }

    // scalar is faster than coopmat when N==1
    ಯದಿ (n_rows == 1 && (path == FA_COOPMAT1 || path == FA_COOPMAT2)) {
        path = FA_SCALAR;
    }

    ವಿಕಲ್ಪ (path) {
    ಪಕ್ಷ FA_SCALAR:
        ಪ್ರತ್ಯಾಗಮ get_fa_tuning_params_scalar(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    ಪಕ್ಷ FA_COOPMAT1:
        ಪ್ರತ್ಯಾಗಮ get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    ಪಕ್ಷ FA_COOPMAT2:
        ಪ್ರತ್ಯಾಗಮ get_fa_tuning_params_coopmat2(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    ಸ್ವಭಾವ:
        ಕ್ಷಿಪ std::runtime_error("unsupported FaCodePath");
    }
}

vk_fa_pipeline_state get_fa_pipeline_state(ಧ್ರುವ vk_device& device, ಧ್ರುವ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, ತಾರ್ಕಿಕ aligned, ತಾರ್ಕಿಕ f32acc,
                                                  ತಾರ್ಕಿಕ use_mask, ತಾರ್ಕಿಕ use_mask_opt, ತಾರ್ಕಿಕ use_logit_softcap, ತಾರ್ಕಿಕ use_sparse, ggml_type k_type, ggml_type v_type) {
    ಧ್ರುವ ತಾರ್ಕಿಕ old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_mask_opt      ? 1 : 0) |
                     (use_mask          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0) |
                     (use_sparse        ? 16 : 0);

    ಧ್ರುವ uint32_t subgroup_size = params.disable_subgroups ? 0 : params.subgroup_size;

    ಪ್ರತ್ಯಾಗಮ vk_fa_pipeline_state{hsk, hsv, params.block_rows, params.block_cols, params.d_split, params.row_split, params.shmem_staging, params.path, params.workgroup_size, subgroup_size, aligned, f32acc, flags, params.limit_occupancy_shmem, k_type, v_type};
}

ಸ್ಥಿರ uint32_t fa_block_bytes(ggml_type t) {
    ಯದಿ (t == GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ 16u;
    }
    ಪ್ರತ್ಯಾಗಮ (uint32_t) ggml_type_size(t);
}

ಸ್ಥಿರ std::vector<uint32_t> get_fa_spec_constants(ಧ್ರುವ vk_fa_pipeline_state& state) {
    ಪ್ರತ್ಯಾಗಮ {
        /* 0 WorkGroupSize   */ state.workgroup_size,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_split,
        /* 7 row_split       */ state.row_split,
        /* 8 SubGroupSize    */ state.subgroup_size,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.limit_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_block_bytes(state.k_type),
        /*15 FaBlockBytesV   */ fa_block_bytes(state.v_type),
    };
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_matmul_shmem_support(ಧ್ರುವ vk_device& device, ಧ್ರುವ std::vector<uint32_t>& warptile, ತಾರ್ಕಿಕ mul_mat_id, ggml_type src0_type) {

    uint32_t lut_size = 0;
    ವಿಕಲ್ಪ (src0_type) {
    ಪಕ್ಷ GGML_TYPE_IQ1_S:
    ಪಕ್ಷ GGML_TYPE_IQ1_M:
        // Regular matmul uses the compact uint16_t IQ1 grid; the expanded
        // uint32_t grid is only enabled for the q8_1/int-dot vector path.
        lut_size = 2*2048;
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
        lut_size = 8*256;
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_TYPE_IQ2_XS:
        lut_size = 8*512;
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_TYPE_IQ2_S:
        lut_size = 8*1024;
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
        lut_size = 4*256;
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_TYPE_IQ3_S:
        lut_size = 4*512;
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_TYPE_IQ4_NL:
    ಪಕ್ಷ GGML_TYPE_IQ4_XS:
    ಪಕ್ಷ GGML_TYPE_MXFP4:
        lut_size = 4*16;
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_size = 4*16 + 128u * (uint32_t)ಪರಿಮಾಣ(ಭಿನ್ನ);
        ಭಙ್ಗ;
    ಸ್ವಭಾವ:
        ಭಙ್ಗ;
    }

    // Needs to be kept up to date on shader changes
    // Needs to stay aligned with ggml_vk_mul_mm_spec.
    ಧ್ರುವ ತಾರ್ಕಿಕ intel_shmem_stride_pad_zero = device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
                                              device->driver_id == vk::DriverId::eIntelProprietaryWindows;
    ಧ್ರುವ uint32_t bank_conflict_offset = intel_shmem_stride_pad_zero ? 0 : (device->coopmat_support ? 8 : 1);
    ಧ್ರುವ uint32_t type_size = device->fp16 ? ಪರಿಮಾಣ(ggml_fp16_t) : ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t warps = warptile[0] / warptile[10];

    ಧ್ರುವ uint32_t load_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_offset) * type_size;
    ಧ್ರುವ uint32_t mmid_row_ids = mul_mat_id ? (warptile[2] * 2 * ಪರಿಮಾಣ(uint16_t)) : 0;
    ಧ್ರುವ uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * ಪರಿಮಾಣ(ಭಿನ್ನ) : 0;

    ಧ್ರುವ uint32_t total_size = load_bufs + mmid_row_ids + coopmat_stage + lut_size;
    ಧ್ರುವ ತಾರ್ಕಿಕ supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    ಪ್ರತ್ಯಾಗಮ supported;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_matmul_int_shmem_support(ಧ್ರುವ vk_device& device, ಧ್ರುವ std::vector<uint32_t>& warptile, ತಾರ್ಕಿಕ mul_mat_id, ggml_type src0_type) {

    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    ಧ್ರುವ uint32_t fp_size   = device->fp16 ? 2u : 4u;
    ಧ್ರುವ uint32_t fp_align  = fp_size;
    ಧ್ರುವ uint32_t fp2_size  = 2u * fp_size;
    ಧ್ರುವ uint32_t fp2_align = device->fp16 ? 4u : 8u;

    ಸಂರಚನಾ member { uint32_t size, align; };
    auto std430_size = [](std::initializer_list<member> members) {
        uint32_t off = 0, struct_align = 1;
        ಪ್ರತಿ (ಧ್ರುವ auto &m : members) {
            off = (off + m.align - 1) & ~(m.align - 1);
            off += m.size;
            struct_align = std::max(struct_align, m.align);
        }
        ಪ್ರತ್ಯಾಗಮ (off + struct_align - 1) & ~(struct_align - 1);
    };

    uint32_t block_a_size = 0;
    ವಿಕಲ್ಪ (src0_type) {
        ಪಕ್ಷ GGML_TYPE_Q2_0:    block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  ಭಙ್ಗ; // qs[8] + dm
        ಪಕ್ಷ GGML_TYPE_Q4_0:    block_a_size = std430_size({{16, 4}, {fp_size,  fp_align}});                  ಭಙ್ಗ; // qs[16/4] + dm
        ಪಕ್ಷ GGML_TYPE_Q4_1:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 ಭಙ್ಗ; // qs[16/4] + dm(vec2)
        ಪಕ್ಷ GGML_TYPE_Q5_0:    block_a_size = std430_size({{16, 4}, {4, 4}, {fp_size,  fp_align}});          ಭಙ್ಗ; // qs[16/4] + qh + dm
        ಪಕ್ಷ GGML_TYPE_Q5_1:    block_a_size = std430_size({{16, 4}, {4, 4}, {fp2_size, fp2_align}});         ಭಙ್ಗ; // qs[16/4] + qh + dm(vec2)
        ಪಕ್ಷ GGML_TYPE_Q8_0:    block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  ಭಙ್ಗ; // qs[8] + dm
        ಪಕ್ಷ GGML_TYPE_IQ4_XS:  block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  ಭಙ್ಗ; // qs[8] + d
        ಪಕ್ಷ GGML_TYPE_MXFP4:   block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  ಭಙ್ಗ; // qs[8] + d
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:  block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  ಭಙ್ಗ; // qs[8] + d
        ಪಕ್ಷ GGML_TYPE_NVFP4:   block_a_size = std430_size({{32, 4}, {fp2_size, fp2_align}});                 ಭಙ್ಗ; // qs[8] + d_scales(vec2)
        ಪಕ್ಷ GGML_TYPE_Q2_K:    block_a_size = std430_size({{ 8, 4}, {2, 2}, {fp2_size, fp2_align}});         ಭಙ್ಗ; // qs[2] + scales(u8vec2) + dm(vec2)
        ಪಕ್ಷ GGML_TYPE_Q3_K:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 ಭಙ್ಗ; // qs[4] + d_scales(vec2)
        ಪಕ್ಷ GGML_TYPE_Q4_K:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 ಭಙ್ಗ; // qs[4] + dm(vec2)
        ಪಕ್ಷ GGML_TYPE_Q5_K:    block_a_size = std430_size({{32, 4}, {fp2_size, fp2_align}});                 ಭಙ್ಗ; // qs[8] + dm(vec2)
        ಪಕ್ಷ GGML_TYPE_Q6_K:    block_a_size = std430_size({{32, 4}, {fp2_size, fp2_align}});                 ಭಙ್ಗ; // qs[8] + d_scales(vec2)
        ಪಕ್ಷ GGML_TYPE_IQ3_S:   block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  ಭಙ್ಗ; // qs[8] + d
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // IQ3_S also copies its 512-entry grid into shared memory (types.glsl, init_iq_shmem)
    ಧ್ರುವ uint32_t lut_size = (src0_type == GGML_TYPE_IQ3_S) ? 4*512 : 0;

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    ಧ್ರುವ uint32_t block_b_size = std430_size({{32, 4}, {fp2_size, fp2_align}});

    ಧ್ರುವ uint32_t BM = warptile[1];
    ಧ್ರುವ uint32_t BN = warptile[2];
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    ಧ್ರುವ uint32_t BK_STEP = mul_mat_id ? 1u : 4u;

    ಧ್ರುವ uint32_t buf_a_size = BM * BK_STEP * block_a_size;
    ಧ್ರುವ uint32_t buf_b_size = BN * BK_STEP * block_b_size;
    ಧ್ರುವ uint32_t mmid_row_ids = mul_mat_id ? (BN * 2u * (uint32_t)ಪರಿಮಾಣ(uint16_t)) : 0u;

    ಧ್ರುವ uint32_t total_size = buf_a_size + buf_b_size + mmid_row_ids + lut_size;
    ಧ್ರುವ ತಾರ್ಕಿಕ supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_matmul_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << total_size << ", supported=" << supported);

    ಪ್ರತ್ಯಾಗಮ supported;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_matmul_cm1_int_shmem_support(ಧ್ರುವ vk_device& device, ಧ್ರುವ std::vector<uint32_t>& warptile, ತಾರ್ಕಿಕ mul_mat_id, ggml_type src0_type) {

    ತಾರ್ಕಿಕ kscales2 = ಅಸತ್ಯ;    // two scale sets per block
    ತಾರ್ಕಿಕ has_dm   = ಅಸತ್ಯ;    // d+m as vec2 + b-side sum
    ತಾರ್ಕಿಕ has_kvalues = ಅಸತ್ಯ;
    ವಿಕಲ್ಪ (src0_type) {
        ಪಕ್ಷ GGML_TYPE_Q4_0: ಪಕ್ಷ GGML_TYPE_Q5_0: ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_1: ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q4_K: ಪಕ್ಷ GGML_TYPE_Q5_K:
            has_dm = ಸತ್ಯ;                          ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ4_NL: ಪಕ್ಷ GGML_TYPE_IQ4_XS: ಪಕ್ಷ GGML_TYPE_MXFP4:
            has_kvalues = ಸತ್ಯ;                     ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q3_K: ಪಕ್ಷ GGML_TYPE_Q6_K:
            kscales2 = ಸತ್ಯ;                        ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_NVFP4:
            kscales2 = ಸತ್ಯ; has_kvalues = ಸತ್ಯ;    ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t BM         = warptile[1];
    ಧ್ರುವ uint32_t BN         = warptile[2];

    ಧ್ರುವ uint32_t BK      = 32;
    ಧ್ರುವ uint32_t BK_STEP = mul_mat_id ? 2u : 4u;
    ಧ್ರುವ uint32_t QPITCH  = BK_STEP * (BK / 4u) + 4u;
    ಧ್ರುವ uint32_t KSCALES = kscales2 ? 2u : 1u;

    uint32_t total = 0;
    total += BM * QPITCH * (uint32_t)ಪರಿಮಾಣ(uint32_t);   // buf_a_qs
    total += BN * QPITCH * (uint32_t)ಪರಿಮಾಣ(uint32_t);   // buf_b_qs
    total += has_dm ? (BM * BK_STEP * 2u * (uint32_t)ಪರಿಮಾಣ(ಭಿನ್ನ))   // buf_a_dm (vec2)
                    : (BM * BK_STEP * KSCALES * (uint32_t)ಪರಿಮಾಣ(ಭಿನ್ನ)); // buf_a_d
    total += BN * BK_STEP * (uint32_t)ಪರಿಮಾಣ(ಭಿನ್ನ);     // buf_b_d
    ಯದಿ (has_dm) {
        total += BN * BK_STEP * (uint32_t)ಪರಿಮಾಣ(ಭಿನ್ನ); // buf_b_s
    }
    ಯದಿ (has_kvalues) {
        total += 16u * (uint32_t)ಪರಿಮಾಣ(int8_t);         // cm1_kvalues[16]
    }
    ಯದಿ (src0_type == GGML_TYPE_NVFP4 && !device->ocp_fp4) {
        total += 128u * (uint32_t)ಪರಿಮಾಣ(ಭಿನ್ನ);         // ue4m3_fp32_lut[128]
    }
    ಯದಿ (mul_mat_id) {
        total += BN * 2u * (uint32_t)ಪರಿಮಾಣ(uint16_t);   // row_ids[BN] (u16vec2)
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ supported = total <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_matmul_cm1_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << total << ", supported=" << supported);

    ಪ್ರತ್ಯಾಗಮ supported;
}

ಸ್ಥಿರ ಧ್ರುವ std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

ಸ್ಥಿರ ಧ್ರುವ std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

ಸ್ಥಿರ std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_SIZE
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_SIZE
    },
};

uint32_t get_subgroup_size(ಧ್ರುವ std::string &pipeline_name, ಧ್ರುವ vk_device_architecture &arch) {
    ಪ್ರತಿ (ಧ್ರುವ auto &config : gpu_pipeline_configs) {
        ಯದಿ (config.arch == arch) {
            auto pipIt = config.pipelines.find(pipeline_name);
            ಯದಿ (pipIt != config.pipelines.end()) {
                ಪ್ರತ್ಯಾಗಮ pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](ಧ್ರುವ auto &a, ಧ್ರುವ auto &b) { ಪ್ರತ್ಯಾಗಮ a.first.size() > b.first.size(); });
            ಪ್ರತಿ (ಧ್ರುವ auto &entry : sorted_pipelines) {
                ಯದಿ (pipeline_name.find(entry.first) != std::string::npos) {
                    ಪ್ರತ್ಯಾಗಮ entry.second;
                }
            }
            ಪ್ರತ್ಯಾಗಮ config.default_subgroup_size;
        }
    }
    ಪ್ರತ್ಯಾಗಮ 0; // If no matching configuration is found
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_fa_type_needs_shmem(ggml_type type) {
    ವಿಕಲ್ಪ (type) {
    ಪಕ್ಷ GGML_TYPE_IQ4_NL:
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_fa_scalar_uses_mmq(ಧ್ರುವ vk_device& device, ggml_type k_type, ggml_type v_type) {
#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    ಪ್ರತ್ಯಾಗಮ device->integer_dot_product && device->subgroup_clustered &&
           !ggml_vk_fa_type_needs_shmem(v_type) &&
           (k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q4_1 ||
            k_type == GGML_TYPE_Q5_0 || k_type == GGML_TYPE_Q5_1 ||
            k_type == GGML_TYPE_Q8_0);
#ಅನ್ಯಥಾ
    GGML_UNUSED(device);
    GGML_UNUSED(k_type);
    GGML_UNUSED(v_type);
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
#endif
}

ಶೂನ್ಯ ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // some shaders have a minimum subgroup size
    ಧ್ರುವ uint32_t subgroup_size_8 = std::max(device->subgroup_size, 8u);
    ಧ್ರುವ uint32_t subgroup_size_16 = std::max(device->subgroup_size, 16u);
    ಧ್ರುವ uint32_t subgroup_size_32 = std::max(device->subgroup_size, 32u);

    // clamp WARP for l_/m_ warptiles so WM <= BM (breaks on subgroupSize > 64)
    ಧ್ರುವ uint32_t mm_warp_8  = std::min(subgroup_size_8,  64u);
    ಧ್ರುವ uint32_t mm_warp_16 = std::min(subgroup_size_16, 64u);

    ಧ್ರುವ uint32_t mul_mat_subgroup_size = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
    ಧ್ರುವ uint32_t mul_mat_subgroup_size_8 = std::max(mul_mat_subgroup_size, 8u);
    ಧ್ರುವ uint32_t mul_mat_subgroup_size_16 = std::max(mul_mat_subgroup_size, 16u);
    ಧ್ರುವ uint32_t mul_mat_subgroup_size_32 = std::max(mul_mat_subgroup_size, 32u);
    ಧ್ರುವ uint32_t mul_mat_mm_warp_8  = std::min(mul_mat_subgroup_size_8,  64u);
    ಧ್ರುವ uint32_t mul_mat_mm_warp_16 = std::min(mul_mat_subgroup_size_16, 64u);

    ಧ್ರುವ ತಾರ್ಕಿಕ subgroup_min_size_16 = (!device->subgroup_size_control && device->subgroup_size >= 16) ||
                                      (device->subgroup_size_control && device->subgroup_max_size >= 16);

    // mulmat
    // Warptile layout (indices match mul_mm.comp constantIDs):
    //   [0..9]  : BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK
    //   [10]    : WARP / required_subgroup_size (read via WARP_SIZE_IDX)
    ಸ್ಥಿರ constexpr size_t WARP_SIZE_IDX = 10;
    std::vector<uint32_t> l_warptile, m_warptile, s_warptile,
                          l_warptile_id, m_warptile_id, s_warptile_id,
                          l_warptile_mmq, m_warptile_mmq, s_warptile_mmq,
                          l_warptile_mmq_int, m_warptile_mmq_int, s_warptile_mmq_int,
                          l_warptile_mmq_cm1_int, m_warptile_mmq_cm1_int, s_warptile_mmq_cm1_int,
                          l_warptile_mmq_cm1_int_k, m_warptile_mmq_cm1_int_k, s_warptile_mmq_cm1_int_k,
                          l_warptile_mmq_int_k, m_warptile_mmq_int_k, s_warptile_mmq_int_k,
                          l_warptile_mmq_k, m_warptile_mmq_k, s_warptile_mmq_k,
                          l_warptile_mmqid, m_warptile_mmqid, s_warptile_mmqid,
                          l_warptile_mmqid_int, m_warptile_mmqid_int, s_warptile_mmqid_int,
                          l_warptile_mmqid_int_k, m_warptile_mmqid_int_k, s_warptile_mmqid_int_k;
    std::array<uint32_t, 3> l_wg_denoms, m_wg_denoms, s_wg_denoms,
                            l_mmq_wg_denoms, m_mmq_wg_denoms, s_mmq_wg_denoms,
                            l_mmq_wg_denoms_k, m_mmq_wg_denoms_k, s_mmq_wg_denoms_k,
                            l_mmq_cm1_wg_denoms_k, m_mmq_cm1_wg_denoms_k, s_mmq_cm1_wg_denoms_k,
                            l_mmqid_wg_denoms, m_mmqid_wg_denoms, s_mmqid_wg_denoms;

    uint32_t l_align, m_align, s_align;

    // RDNA3.5 preferred wave32 here
    ಧ್ರುವ ತಾರ್ಕಿಕ cm1_use_wave32 = device->vendor_id == VK_VENDOR_ID_AMD &&
                                device->subgroup_size_control &&
                                device->subgroup_min_size <= 32 && device->subgroup_max_size >= 32;
    ಧ್ರುವ uint32_t cm1_sg = cm1_use_wave32 ? 32 : device->subgroup_size;

    vk_pipeline wait_pipeline;
    CompileTask claimed_task {};
    ತಾರ್ಕಿಕ has_claimed_task = ಅಸತ್ಯ;

    // The rest of the walk reads and writes shared device state, so hold the
    // lock until we're done deciding what to compile.
    std::unique_lock<std::mutex> compile_lock(device->compile_mutex);

    ಯದಿ (device->coopmat2) {
        // spec constants and tile sizes for non-quant matmul/matmul_id
        l_warptile = { 256, 128, 256, 64, 1 };
        m_warptile = { 256, 128, 128, 64, 0 };
        s_warptile = { 128,  64,  64, 64, 0 };
        l_wg_denoms = {128, 256, 1 };
        m_wg_denoms = {128, 128, 1 };
        s_wg_denoms = { 64,  64, 1 };

        // spec constants and tile sizes for quant matmul (non-Qi_K)
        l_warptile_mmq = { 256, 128, 256, 64, 1 };
        m_warptile_mmq = { 256, 128, 128, 64, 1 };
        s_warptile_mmq = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms = { 128, 256, 1 };
        m_mmq_wg_denoms = { 128, 128, 1 };
        s_mmq_wg_denoms = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul (Qi_K)
        l_warptile_mmq_k = { 256, 128, 256, 64, 1 };
        m_warptile_mmq_k = { 256, 128, 128, 64, 1 };
        s_warptile_mmq_k = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms_k = { 128, 256, 1 };
        m_mmq_wg_denoms_k = { 128, 128, 1 };
        s_mmq_wg_denoms_k = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul_id
        ಧ್ರುವ uint32_t mmqid_bk = device->coopmat2_decode_vector ? 64u : 32u;
        l_warptile_mmqid = { 256, 128, 128, mmqid_bk, 1 };
        m_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        s_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } ಅನ್ಯಥಾ {
        // Matrix cores require different warp group sizes
        ಧ್ರುವ uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        ಧ್ರುವ uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        ಧ್ರುವ uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        ಧ್ರುವ uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        ಧ್ರುವ uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        ಧ್ರುವ uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        ಧ್ರುವ uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        ಧ್ರುವ uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        ಧ್ರುವ uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        ಧ್ರುವ uint32_t itm = device->coopmat_int_m;
        ಧ್ರುವ uint32_t itn = device->coopmat_int_n;
        ಧ್ರುವ uint32_t itk = device->coopmat_int_k;

        ಧ್ರುವ uint32_t s_warptile_wm = device->subgroup_size == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile = { 128,              64,  64, 16, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile = { subgroup_size_32, 32,  32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

        l_warptile_mmq = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile_mmq = { 128,              64,  64, 32, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile_mmq = { subgroup_size_32, 32,  32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, mm_warp_8,     32, 2, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int = { subgroup_size_32, 32,  32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_size_8 };

        ಧ್ರುವ auto cm1_bs = [cm1_sg](uint32_t bm, uint32_t bn) {
            ಪ್ರತ್ಯಾಗಮ cm1_sg * (bm / std::min(cm1_sg, bm)) * (bn / 32);
        };

        l_warptile_mmq_cm1_int = { cm1_bs(128, 128), 128, 128, 32, std::min(cm1_sg, 128u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        m_warptile_mmq_cm1_int = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        s_warptile_mmq_cm1_int = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };

        l_warptile_mmq_cm1_int_k = { cm1_bs( 64, 128),  64, 128, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        m_warptile_mmq_cm1_int_k = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        s_warptile_mmq_cm1_int_k = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };

        l_mmq_cm1_wg_denoms_k = { l_warptile_mmq_cm1_int_k[1], l_warptile_mmq_cm1_int_k[2], 1 };
        m_mmq_cm1_wg_denoms_k = { m_warptile_mmq_cm1_int_k[1], m_warptile_mmq_cm1_int_k[2], 1 };
        s_mmq_cm1_wg_denoms_k = { s_warptile_mmq_cm1_int_k[1], s_warptile_mmq_cm1_int_k[2], 1 };

        // K-quants use even more registers, mitigate by setting WMITER to 1
        l_warptile_mmq_int_k = { 128,               128, 128, 32, mm_warp_8 * 2, 64, 1, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int_k = { 128,                64,  64, 32, mm_warp_8,     32, 1, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int_k = { subgroup_size_32,   32,  32, 32, s_warptile_wm, 32, 1, 2, 1, 1, subgroup_size_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_mm_warp_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_mm_warp_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_16 };
        s_warptile_id = { mul_mat_subgroup_size_16,  32,  32, 16, s_warptile_wm,          32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_8 };
        s_warptile_mmqid = { mul_mat_subgroup_size_32,   32,  32, 32, s_warptile_wm,         32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, 4, 4, 1, mul_mat_mm_warp_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, 2, 2, 1, mul_mat_mm_warp_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_size_32,   32,  32, 32, s_warptile_wm,         32, 2, 2, 1, 1, mul_mat_subgroup_size_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_mm_warp_16 * 2, 64, 1, 4, 4, 1, mul_mat_mm_warp_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_mm_warp_16,     32, 1, 2, 2, 1, mul_mat_mm_warp_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_size_32, 32,  32, 32, s_warptile_wm,          32, 1, 2, 1, 1, mul_mat_subgroup_size_16 };

        // chip specific tuning
        ಯದಿ ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } ಅನ್ಯಥಾ ಯದಿ (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
        } ಅನ್ಯಥಾ ಯದಿ (device->vendor_id == VK_VENDOR_ID_QUALCOMM && device->coopmat_support) {
            m_warptile     = { 64, 64, 64, 16, 64, 64, 1, tm_l, tn_l, tk_l, 64 };
            m_warptile_mmq = { 64, 64, 64, 32, 64, 64, 1, tm_m, tn_m, tk_m, 64 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        ಯದಿ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support) {
            // Xe1/Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, mm_warp_8, 32, 2, tm_l, tn_l, tk_l, mm_warp_8 };
            ಯದಿ (device->architecture == INTEL_XE1) {
                l_warptile_mmq  = { 512, 256, 128, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 256, 128, 1 };
                l_align         = 32;  //set as BK
            } ಅನ್ಯಥಾ {
                l_warptile_mmq  = { 512, 128, 256, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 128, 256, 1 };
                l_align         = 32;  //set as BK
            }
        }

        ಧ್ರುವ ತಾರ್ಕಿಕ use_cm1_int = device->coopmat_int_support &&
                                 (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);

        ಪ್ರತಿ (uint32_t i = 0; i < GGML_TYPE_ಎಣಿಕೆ; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            ಯದಿ (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, ಅಸತ್ಯ, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::endl;
                ಕ್ಷಿಪ std::runtime_error("Shared memory size too small for matrix multiplication.");
            } ಅನ್ಯಥಾ ಯದಿ (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, ಅಸತ್ಯ, t)) {
                device->mul_mat_m[i] = ಅಸತ್ಯ;
                device->mul_mat_l[i] = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, ಅಸತ್ಯ, t)) {
                device->mul_mat_l[i] = ಅಸತ್ಯ;
            }

            // Disable mul_mat_id if not enough shared memory is available
            ಯದಿ (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, ಸತ್ಯ, t)) {
                device->mul_mat_id_s[i] = ಅಸತ್ಯ;
                device->mul_mat_id_m[i] = ಅಸತ್ಯ;
                device->mul_mat_id_l[i] = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, ಸತ್ಯ, t)) {
                device->mul_mat_id_m[i] = ಅಸತ್ಯ;
                device->mul_mat_id_l[i] = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, ಸತ್ಯ, t)) {
                device->mul_mat_id_l[i] = ಅಸತ್ಯ;
            }

            // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
            // K-quants and IQ3_S use the _int_k warptiles, others use _int.
            // cm1 splits k-tiles on the KSCALES==2 types and shares tiles between dense/id.
            ಧ್ರುವ ತಾರ್ಕಿಕ is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                     t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                     t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ3_S);
            ಧ್ರುವ ತಾರ್ಕಿಕ cm1_k_tile = (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q6_K ||
                                     t == GGML_TYPE_NVFP4);

            ಧ್ರುವ auto & s_int   = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmq_int_k     : s_warptile_mmq_int);
            ಧ್ರುವ auto & m_int   = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmq_int_k     : m_warptile_mmq_int);
            ಧ್ರುವ auto & l_int   = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmq_int_k     : l_warptile_mmq_int);
            ಧ್ರುವ auto & s_intid = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmqid_int_k   : s_warptile_mmqid_int);
            ಧ್ರುವ auto & m_intid = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmqid_int_k   : m_warptile_mmqid_int);
            ಧ್ರುವ auto & l_intid = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmqid_int_k   : l_warptile_mmqid_int);

            ಧ್ರುವ auto int_shmem_support = [&](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ id) {
                ಪ್ರತ್ಯಾಗಮ use_cm1_int ? ggml_vk_matmul_cm1_int_shmem_support(device, wt, id, t)
                                   : ggml_vk_matmul_int_shmem_support(device, wt, id, t);
            };

            ಯದಿ (!int_shmem_support(s_int, ಅಸತ್ಯ)) {
                device->mul_mat_s_int[i] = ಅಸತ್ಯ;
                device->mul_mat_m_int[i] = ಅಸತ್ಯ;
                device->mul_mat_l_int[i] = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (!int_shmem_support(m_int, ಅಸತ್ಯ)) {
                device->mul_mat_m_int[i] = ಅಸತ್ಯ;
                device->mul_mat_l_int[i] = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (!int_shmem_support(l_int, ಅಸತ್ಯ)) {
                device->mul_mat_l_int[i] = ಅಸತ್ಯ;
            }

            ಯದಿ (!int_shmem_support(s_intid, ಸತ್ಯ)) {
                device->mul_mat_id_s_int[i] = ಅಸತ್ಯ;
                device->mul_mat_id_m_int[i] = ಅಸತ್ಯ;
                device->mul_mat_id_l_int[i] = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (!int_shmem_support(m_intid, ಸತ್ಯ)) {
                device->mul_mat_id_m_int[i] = ಅಸತ್ಯ;
                device->mul_mat_id_l_int[i] = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (!int_shmem_support(l_intid, ಸತ್ಯ)) {
                device->mul_mat_id_l_int[i] = ಅಸತ್ಯ;
            }
        }
    }

    auto ಧ್ರುವ &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& base_pipeline, ಧ್ರುವ ವರ್ಣ *name, size_t spv_size, ಧ್ರುವ ಶೂನ್ಯ* spv_data, ಧ್ರುವ ವರ್ಣ *entrypoint,
                                              uint32_t parameter_ಎಣಿಕೆ, uint32_t ತಳ್ಳು_constant_size, std::array<uint32_t, 3> wg_denoms, ಧ್ರುವ std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, ತಾರ್ಕಿಕ disable_robustness = ಅಸತ್ಯ, ತಾರ್ಕಿಕ require_full_subgroups = ಅಸತ್ಯ, uint32_t required_subgroup_size = 0) {

        ಯದಿ (!require_full_subgroups && required_subgroup_size == 0) {
            required_subgroup_size = get_subgroup_size(name, device->architecture);
        }

        vk_pipeline *ptr = &base_pipeline;

        ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_pipelines = 1;
#ಯದಿ defined(VK_EXT_shader_64bit_indexing)
        ಯದಿ (device->shader_64b_indexing) {
            ಅಂಕೆ_pipelines = 2;
        }
#endif
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_pipelines; ++i, ptr = &(*ptr)->next) {
            vk_pipeline &pipeline = *ptr;
            ಯದಿ (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            ಯದಿ (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_ಎಣಿಕೆ = parameter_ಎಣಿಕೆ;
                pipeline->ತಳ್ಳು_constant_size = ತಳ್ಳು_constant_size;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = ಸತ್ಯ;
#ಯದಿ defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#endif
            }

            // We only care about the pipeline this call asked for; the rest
            // (including the 64-bit indexing variant) are handled by their
            // own request_descriptor_sets / load_shaders calls.
            ಯದಿ (pipeline.get() != requested.get()) {
                ಅನುವರ್ತ;
            }

            ಯದಿ (pipeline->compiled) {
                ಅನುವರ್ತ;
            }

            wait_pipeline = pipeline;

            ಯದಿ (!pipeline->compile_pending) {
                pipeline->compile_pending = ಸತ್ಯ;
                claimed_task.pipeline = pipeline;
                claimed_task.spv_size = spv_size;
                claimed_task.spv_data = spv_data;
                claimed_task.entrypoint = entrypoint;
                claimed_task.parameter_ಎಣಿಕೆ = parameter_ಎಣಿಕೆ;
                claimed_task.wg_denoms = wg_denoms;
                claimed_task.specialization_constants = specialization_constants;
                claimed_task.disable_robustness = disable_robustness;
                claimed_task.require_full_subgroups = require_full_subgroups;
                claimed_task.required_subgroup_size = required_subgroup_size;
                has_claimed_task = ಸತ್ಯ;
            }
        }
    };

    auto ಧ್ರುವ &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, ಧ್ರುವ std::string &name, size_t spv_size, ಧ್ರುವ ಶೂನ್ಯ* spv_data, ಧ್ರುವ ವರ್ಣ *entrypoint,
                                              uint32_t parameter_ಎಣಿಕೆ, uint32_t ತಳ್ಳು_constant_size, std::array<uint32_t, 3> wg_denoms, ಧ್ರುವ std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, ತಾರ್ಕಿಕ disable_robustness = ಅಸತ್ಯ, ತಾರ್ಕಿಕ require_full_subgroups = ಅಸತ್ಯ, uint32_t required_subgroup_size = 0) {
        ಪ್ರತ್ಯಾಗಮ ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_size, spv_data, entrypoint,
                                       parameter_ಎಣಿಕೆ, ತಳ್ಳು_constant_size, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_full_subgroups, required_subgroup_size);
    };

    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V
    // quant type is selected at runtime via the FaTypeK / FaTypeV spec constants.

    ಪ್ರತಿ (auto &fa : device->pipeline_flash_attn_f32_f16) {
        ಯದಿ (fa.first.path != FA_SCALAR) ಅನುವರ್ತ;
        ಧ್ರುವ uint32_t Br = fa.first.Br;
        ಧ್ರುವ uint32_t Bc = fa.first.Bc;
        ಧ್ರುವ ತಾರ್ಕಿಕ aligned = fa.first.aligned;
        ಧ್ರುವ ತಾರ್ಕಿಕ f32acc = fa.first.f32acc;
        ಧ್ರುವ uint32_t fa_sgs = fa.first.subgroup_size;
        ಧ್ರುವ ತಾರ್ಕಿಕ fa_ds = fa.first.subgroup_size == 0;

        ಧ್ರುವ ತಾರ್ಕಿಕ bf16_kv = fa.first.k_type == GGML_TYPE_BF16;
        ಧ್ರುವ ತಾರ್ಕಿಕ use_mmq = ggml_vk_fa_scalar_uses_mmq(device, fa.first.k_type, fa.first.v_type);
        ಧ್ರುವ ಶೂನ್ಯ * spv_data = nullptr;
        size_t spv_size = 0;
        ಧ್ರುವ ವರ್ಣ *name = nullptr;
        ಯದಿ (bf16_kv) {
            spv_data = flash_attn_f32_f16_fp32_data;
            spv_size = flash_attn_f32_f16_fp32_len;
            name = aligned ? "flash_attn_f32_bf16_aligned" : "flash_attn_f32_bf16";
        } ಅನ್ಯಥಾ ಯದಿ (use_mmq) {
#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ಯದಿ (device->fp16) {
                ಯದಿ (f32acc) { spv_data = flash_attn_f32_f16_int8_data;        spv_size = flash_attn_f32_f16_int8_len; }
                ಅನ್ಯಥಾ        { spv_data = flash_attn_f32_f16_f16acc_int8_data; spv_size = flash_attn_f32_f16_f16acc_int8_len; }
            } ಅನ್ಯಥಾ {
                spv_data = flash_attn_f32_f16_fp32_int8_data;
                spv_size = flash_attn_f32_f16_fp32_int8_len;
            }
#endif
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        } ಅನ್ಯಥಾ {
            ಯದಿ (device->fp16) {
                ಯದಿ (device->dot2_f16) {
                    ಯದಿ (f32acc) { spv_data = flash_attn_f32_f16_dot2_data;        spv_size = flash_attn_f32_f16_dot2_len; }
                    ಅನ್ಯಥಾ        { spv_data = flash_attn_f32_f16_dot2_f16acc_data; spv_size = flash_attn_f32_f16_dot2_f16acc_len; }
                } ಅನ್ಯಥಾ {
                    ಯದಿ (f32acc) { spv_data = flash_attn_f32_f16_data;        spv_size = flash_attn_f32_f16_len; }
                    ಅನ್ಯಥಾ        { spv_data = flash_attn_f32_f16_f16acc_data; spv_size = flash_attn_f32_f16_f16acc_len; }
                }
            } ಅನ್ಯಥಾ {
                spv_data = flash_attn_f32_f16_fp32_data;
                spv_size = flash_attn_f32_f16_fp32_len;
            }
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        }
        ggml_vk_create_pipeline(device, fa.second, name, spv_size, spv_data, "main", 8,
                                ಪರಿಮಾಣ(vk_flash_attn_ತಳ್ಳು_constants), {Br, 1, 1},
                                get_fa_spec_constants(fa.first), aligned ? Bc : 1, ಸತ್ಯ,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }

#ಯದಿ defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ಯದಿ (device->coopmat1_fa_support) {
        ಪ್ರತಿ (auto &fa : device->pipeline_flash_attn_f32_f16) {
            ಯದಿ (fa.first.path != FA_COOPMAT1) ಅನುವರ್ತ;
            ಧ್ರುವ uint32_t Br = fa.first.Br;
            ಧ್ರುವ uint32_t Bc = fa.first.Bc;
            ಧ್ರುವ ತಾರ್ಕಿಕ aligned = fa.first.aligned;
            ಧ್ರುವ ತಾರ್ಕಿಕ f32acc = fa.first.f32acc;
            ಧ್ರುವ uint32_t fa_sgs = fa.first.subgroup_size;
            ಧ್ರುವ ತಾರ್ಕಿಕ fa_ds = fa.first.subgroup_size == 0;

            ಧ್ರುವ ತಾರ್ಕಿಕ bf16_kv = fa.first.k_type == GGML_TYPE_BF16;

            ಧ್ರುವ ಶೂನ್ಯ * spv_data;
            size_t spv_size;
            ಧ್ರುವ ವರ್ಣ *name;
            ಯದಿ (bf16_kv) {
#ಯದಿ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ಯದಿ (!device->coopmat_bf16_support) ಅನುವರ್ತ;
                spv_data = flash_attn_f32_f16_bf16_cm1_data;
                spv_size = flash_attn_f32_f16_bf16_cm1_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm1" : "flash_attn_f32_bf16_cm1";
#ಅನ್ಯಥಾ
                ಅನುವರ್ತ;
#endif
            } ಅನ್ಯಥಾ {
                ಯದಿ (f32acc) { spv_data = flash_attn_f32_f16_cm1_data;        spv_size = flash_attn_f32_f16_cm1_len; }
                ಅನ್ಯಥಾ        { spv_data = flash_attn_f32_f16_f16acc_cm1_data; spv_size = flash_attn_f32_f16_f16acc_cm1_len; }
                name = aligned ? "flash_attn_f32_f16_aligned_cm1" : "flash_attn_f32_f16_cm1";
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_size, spv_data, "main", 8,
                                    ಪರಿಮಾಣ(vk_flash_attn_ತಳ್ಳು_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.first), aligned ? Bc : 1, ಸತ್ಯ,
                                    !fa_ds, !fa_ds ? fa_sgs : 0);
        }
    }
#endif

#ಯದಿ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    ಯದಿ (device->coopmat2) {
        ಪ್ರತಿ (auto &fa : device->pipeline_flash_attn_f32_f16) {
            ಯದಿ (fa.first.path != FA_COOPMAT2) ಅನುವರ್ತ;
            ಧ್ರುವ uint32_t Br = fa.first.Br;
            ಧ್ರುವ uint32_t Bc = fa.first.Bc;
            ಧ್ರುವ ತಾರ್ಕಿಕ aligned = fa.first.aligned;
            ಧ್ರುವ ತಾರ್ಕಿಕ f32acc = fa.first.f32acc;

            ಧ್ರುವ ತಾರ್ಕಿಕ bf16_kv = fa.first.k_type == GGML_TYPE_BF16;
            ಧ್ರುವ ಶೂನ್ಯ * spv_data;
            size_t spv_size;
            ಧ್ರುವ ವರ್ಣ * name;
            ಯದಿ (bf16_kv) {
#ಯದಿ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ಯದಿ (!device->coopmat2_bf16_support) ಅನುವರ್ತ;
                spv_data = flash_attn_f32_f16_bf16_cm2_data;
                spv_size = flash_attn_f32_f16_bf16_cm2_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm2" : "flash_attn_f32_bf16_cm2";
#ಅನ್ಯಥಾ
                ಅನುವರ್ತ;
#endif
            } ಅನ್ಯಥಾ ಯದಿ (aligned) {
                ಯದಿ (f32acc) { spv_data = flash_attn_f32_f16_cm2_data;        spv_size = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_aligned_f32acc_cm2"; }
                ಅನ್ಯಥಾ        { spv_data = flash_attn_f32_f16_f16acc_cm2_data; spv_size = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_aligned_f16acc_cm2"; }
            } ಅನ್ಯಥಾ {
                ಯದಿ (f32acc) { spv_data = flash_attn_f32_f16_cm2_data;        spv_size = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_f32acc_cm2"; }
                ಅನ್ಯಥಾ        { spv_data = flash_attn_f32_f16_f16acc_cm2_data; spv_size = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_f16acc_cm2"; }
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_size, spv_data, "main", 8,
                                    ಪರಿಮಾಣ(vk_flash_attn_ತಳ್ಳು_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.first), aligned ? Bc : 1, ಸತ್ಯ, ಅಸತ್ಯ, 0);
        }
    }
#endif

    auto ಧ್ರುವ &ggml_vk_mul_mm_spec = [&device](std::vector<uint32_t> spec, ತಾರ್ಕಿಕ aligned) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        ಯದಿ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=12: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=13: APPLY_SLM_A_RESHAPE = true
        }
        ಪ್ರತ್ಯಾಗಮ spec;
    };

    auto ಧ್ರುವ &ggml_vk_mul_mm_spec_quant = [&device](std::vector<uint32_t> spec, ತಾರ್ಕಿಕ aligned, uint32_t type) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        spec.push_back(type);               // constantID=12: MmTypeA
        ಯದಿ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=13: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=14: APPLY_SLM_A_RESHAPE = true
        }
        ಪ್ರತ್ಯಾಗಮ spec;
    };

    ಸ್ಥಿರ ಧ್ರುವ ggml_type non_lut_quant_types[] = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    };

#define FOR_EACH_LUT_TYPE_NONFP4(X) \
    X(GGML_TYPE_IQ1_S,   iq1_s)   \
    X(GGML_TYPE_IQ1_M,   iq1_m)   \
    X(GGML_TYPE_IQ2_XXS, iq2_xxs) \
    X(GGML_TYPE_IQ2_XS,  iq2_xs)  \
    X(GGML_TYPE_IQ2_S,   iq2_s)   \
    X(GGML_TYPE_IQ3_XXS, iq3_xxs) \
    X(GGML_TYPE_IQ3_S,   iq3_s)   \
    X(GGML_TYPE_IQ4_XS,  iq4_xs)  \
    X(GGML_TYPE_IQ4_NL,  iq4_nl)
#define FOR_EACH_LUT_FP4_TYPE(X) \
    X(GGML_TYPE_MXFP4,   mxfp4)   \
    X(GGML_TYPE_NVFP4,   nvfp4)
#define FOR_EACH_LUT_TYPE(X) \
    FOR_EACH_LUT_TYPE_NONFP4(X)  \
    FOR_EACH_LUT_FP4_TYPE(X)

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mul_mat_id_param_ಎಣಿಕೆ = 5;

    using spec_fn_t = std::function<std::vector<uint32_t>(ಧ್ರುವ std::vector<uint32_t>&, ತಾರ್ಕಿಕ)>;
    auto ಧ್ರುವ &create_mm_pipelines = [&](
        ಧ್ರುವ vk_matmul_pipeline_ಕೀಲಿ& ಕೀಲಿ,
        ಧ್ರುವ std::vector<vk_tile_config>& tile_configs,
        ಧ್ರುವ std::string& shader_name, size_t spv_len, ಧ್ರುವ ಶೂನ್ಯ* spv_data,
        uint32_t ತಳ್ಳು_constant_size, uint32_t param_ಎಣಿಕೆ,
        ಧ್ರುವ spec_fn_t& spec_fn,
        ತಾರ್ಕಿಕ disable_robustness = ಅಸತ್ಯ, ತಾರ್ಕಿಕ require_full_subgroups = ಅಸತ್ಯ, uint32_t required_subgroup_size = 0,
        ತಾರ್ಕಿಕ create_aligned = ಸತ್ಯ, ತಾರ್ಕಿಕ pin_subgroup_to_warp = ಅಸತ್ಯ
    ) {
        auto& vec = device->pipeline_matmul[ಕೀಲಿ];
        ಧ್ರುವ ತಾರ್ಕಿಕ first_call = vec.empty();
        ಪ್ರತಿ (size_t i = 0; i < tile_configs.size(); i++) {
            ಧ್ರುವ auto& tc = tile_configs[i];

            // Intel coopmat1 pins the required subgroup size to each warptile's WARP element.
            ಧ್ರುವ uint32_t rsgs = pin_subgroup_to_warp ? tc.warptile[WARP_SIZE_IDX] : required_subgroup_size;
            ಧ್ರುವ ತಾರ್ಕಿಕ     rfs  = require_full_subgroups || pin_subgroup_to_warp;

            ಯದಿ (first_call) {
                vk_matmul_pipeline_pair pair{};
                pair.align = tc.align;
                std::string suffix = "_" + std::to_string(i);
                pair.unaligned = std::make_shared<vk_pipeline_struct>();
                ಯದಿ (create_aligned) {
                    pair.aligned = std::make_shared<vk_pipeline_struct>();
                }
                vec.push_back(pair);
            }

            ggml_vk_create_pipeline(device, vec[i].unaligned,
                vec[i].unaligned->name.empty() ? (shader_name + "_" + std::to_string(i)).c_str() : vec[i].unaligned->name.c_str(),
                spv_len, spv_data, "main", param_ಎಣಿಕೆ, ತಳ್ಳು_constant_size,
                tc.wg_denoms, spec_fn(tc.warptile, ಅಸತ್ಯ), 1,
                disable_robustness, rfs, rsgs);

            ಯದಿ (vec[i].aligned) {
                ggml_vk_create_pipeline(device, vec[i].aligned,
                    vec[i].aligned->name.empty() ? (shader_name + "_aligned_" + std::to_string(i)).c_str() : vec[i].aligned->name.c_str(),
                    spv_len, spv_data, "main", param_ಎಣಿಕೆ, ತಳ್ಳು_constant_size,
                    tc.wg_denoms, spec_fn(tc.warptile, ಸತ್ಯ), tc.align,
                    disable_robustness, rfs, rsgs);
            }
        }
    };

    auto filter_tc = [&](ಧ್ರುವ std::vector<vk_tile_config>& configs, ggml_type type, ತಾರ್ಕಿಕ is_id, ತಾರ್ಕಿಕ is_int = ಅಸತ್ಯ) -> std::vector<vk_tile_config> {
        std::vector<vk_tile_config> result;
        ತಾರ್ಕಿಕ enabled[3];
        ಯದಿ (is_int) {
            enabled[0] = is_id ? device->mul_mat_id_s_int[type] : device->mul_mat_s_int[type];
            enabled[1] = is_id ? device->mul_mat_id_m_int[type] : device->mul_mat_m_int[type];
            enabled[2] = is_id ? device->mul_mat_id_l_int[type] : device->mul_mat_l_int[type];
        } ಅನ್ಯಥಾ {
            enabled[0] = is_id ? device->mul_mat_id_s[type] : device->mul_mat_s[type];
            enabled[1] = is_id ? device->mul_mat_id_m[type] : device->mul_mat_m[type];
            enabled[2] = is_id ? device->mul_mat_id_l[type] : device->mul_mat_l[type];
        }
        ಪ್ರತಿ (size_t i = 0; i < configs.size() && i < 3; i++) {
            ಯದಿ (enabled[i]) result.push_back(configs[i]);
        }
        ಪ್ರತ್ಯಾಗಮ result;
    };

    std::vector<vk_tile_config> tc_mm = {{s_warptile, s_wg_denoms, s_align}, {m_warptile, m_wg_denoms, m_align}, {l_warptile, l_wg_denoms, l_align}};
    std::vector<vk_tile_config> tc_mmq = {{s_warptile_mmq, s_mmq_wg_denoms, s_align}, {m_warptile_mmq, m_mmq_wg_denoms, m_align}, {l_warptile_mmq, l_mmq_wg_denoms, l_align}};

#ಯದಿ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    ಯದಿ (device->coopmat2) {
        auto ಧ್ರುವ &ggml_vk_mul_mm_cm2_spec = [&](std::vector<uint32_t> spec, ತಾರ್ಕಿಕ aligned, uint32_t type = UINT32_MAX) {
            spec.push_back(aligned ? 1u : 0u);        // ALIGNED
            spec.push_back(device->subgroup_size);     // subgroup_size
            ಯದಿ (type != UINT32_MAX) {
                spec.push_back(type);                  // MmTypeA
                spec.push_back((uint32_t)ggml_type_size((ggml_type)type)); // MmABlockBytes
            }
            ಪ್ರತ್ಯಾಗಮ spec;
        };

        std::vector<vk_tile_config> tc_mmq_k = {{s_warptile_mmq_k, s_mmq_wg_denoms_k, s_align}, {m_warptile_mmq_k, m_mmq_wg_denoms_k, m_align}, {l_warptile_mmq_k, l_mmq_wg_denoms_k, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmqid_wg_denoms, s_align}, {m_warptile_mmqid, m_mmqid_wg_denoms, m_align}, {l_warptile_mmqid, l_mmqid_wg_denoms, l_align}};

        spec_fn_t cm2_spec = [&](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಪ್ರತ್ಯಾಗಮ ggml_vk_mul_mm_cm2_spec(wt, a); };

        // F16 x F16
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm2_len, matmul_f16_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f16",        matmul_f16_cm2_len,        matmul_f16_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ);
#ಯದಿ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ಯದಿ (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_bf16", matmul_bf16_cm2_len, matmul_bf16_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ);
        }
#endif
        ಪ್ರತಿ (ಧ್ರುವ auto type : non_lut_quant_types) {
            // regression in unified shader on Ampere
            ಯದಿ (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                ಅನುವರ್ತ;
            }
            auto& tc = ((type >= GGML_TYPE_Q2_K && type <= GGML_TYPE_Q6_K) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) ? tc_mmq_k : tc_mmq;
            spec_fn_t qs = [&, type](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಪ್ರತ್ಯಾಗಮ ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm2_len, matmul_quant_f16_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, qs, ಸತ್ಯ);
            create_mm_pipelines({type, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc, "matmul_quant_f16",        matmul_quant_f16_cm2_len,        matmul_quant_f16_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, qs, ಸತ್ಯ);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mmq_k, "matmul_q4_k_f16_f16acc", matmul_q4_k_f16_f16acc_cm2_len, matmul_q4_k_f16_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_k, "matmul_q4_k_f16",        matmul_q4_k_f16_cm2_len,        matmul_q4_k_f16_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mmq_k, "matmul_q5_k_f16_f16acc", matmul_q5_k_f16_f16acc_cm2_len, matmul_q5_k_f16_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_k, "matmul_q5_k_f16",        matmul_q5_k_f16_cm2_len,        matmul_q5_k_f16_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ);
#define X_CM2(TYPE, tstr) \
        { auto tc = filter_tc(tc_mmq, TYPE, ಅಸತ್ಯ); \
          ಯದಿ (!tc.empty()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm2_len, matmul_##tstr##_f16_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm2_len,        matmul_##tstr##_f16_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2)
#ಯದಿ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ಯದಿ (device->ocp_fp4) {
#define X_CM2_OCP(TYPE, tstr) \
            { auto tc = filter_tc(tc_mmq, TYPE, ಅಸತ್ಯ); \
              ಯದಿ (!tc.empty()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm2_len, matmul_##tstr##_f16_ocp_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm2_len,        matmul_##tstr##_f16_ocp_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ಸತ್ಯ); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_OCP)
#undef X_CM2_OCP
        } ಅನ್ಯಥಾ
#endif
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2)
        }
#undef X_CM2

        GGML_ASSERT(device->subgroup_ballot);

        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc_mm, "matmul_id_subgroup_f16_f16acc", matmul_id_subgroup_f16_f16acc_cm2_len, matmul_id_subgroup_f16_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_subgroup_f16",        matmul_id_subgroup_f16_cm2_len,        matmul_id_subgroup_f16_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ);
#ಯದಿ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ಯದಿ (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm2_len, matmul_id_subgroup_bf16_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ);
        }
#endif
        ಪ್ರತಿ (ಧ್ರುವ auto type : non_lut_quant_types) {
            ಯದಿ (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                ಅನುವರ್ತ;
            }
            spec_fn_t qs_id = [&, type](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಪ್ರತ್ಯಾಗಮ ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc_mmqid, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm2_len, matmul_id_subgroup_quant_f16_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, qs_id, ಸತ್ಯ);
            create_mm_pipelines({type, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm2_len,        matmul_id_subgroup_quant_f16_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, qs_id, ಸತ್ಯ);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc_mmqid, "matmul_id_subgroup_q4_k_f16_f16acc", matmul_id_subgroup_q4_k_f16_f16acc_cm2_len, matmul_id_subgroup_q4_k_f16_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_subgroup_q4_k_f16",        matmul_id_subgroup_q4_k_f16_cm2_len,        matmul_id_subgroup_q4_k_f16_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc_mmqid, "matmul_id_subgroup_q5_k_f16_f16acc", matmul_id_subgroup_q5_k_f16_f16acc_cm2_len, matmul_id_subgroup_q5_k_f16_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_subgroup_q5_k_f16",        matmul_id_subgroup_q5_k_f16_cm2_len,        matmul_id_subgroup_q5_k_f16_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ);
#define X_CM2_ID(TYPE, tstr) \
        { auto tc = filter_tc(tc_mmqid, TYPE, ಸತ್ಯ); \
          ಯದಿ (!tc.empty()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm2_len,        matmul_id_subgroup_##tstr##_f16_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2_ID)
#ಯದಿ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ಯದಿ (device->ocp_fp4) {
#define X_CM2_ID_OCP(TYPE, tstr) \
            { auto tc = filter_tc(tc_mmqid, TYPE, ಸತ್ಯ); \
              ಯದಿ (!tc.empty()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm2_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm2_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ಸತ್ಯ); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID_OCP)
#undef X_CM2_ID_OCP
        } ಅನ್ಯಥಾ
#endif
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID)
        }
#undef X_CM2_ID
    } ಅನ್ಯಥಾ
#endif  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#ಯದಿ defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ಯದಿ (device->coopmat_support) {
        spec_fn_t cm1_spec = [&](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಪ್ರತ್ಯಾಗಮ ggml_vk_mul_mm_spec(wt, a); };

        // Intel coopmat1 pins each pipeline's required subgroup size to its warptile WARP element.
        ಧ್ರುವ ತಾರ್ಕಿಕ cm1_pin = device->vendor_id == VK_VENDOR_ID_INTEL;

        // Intel coopmat1 uses a dedicated large-tile config for quant matmul_id.
        std::vector<vk_tile_config> tc_mmq_id = tc_mmq;
        ಯದಿ (cm1_pin) {
            tc_mmq_id[2] = { { 512, 128, 128, 32, 32, 32, 2, device->coopmat_m, device->coopmat_n, device->coopmat_k, 32 }, { 128, 128, 1 }, 32 };
        }

        auto cm1_create = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಧ್ರುವ std::vector<vk_tile_config>& tc_base,
                              ಧ್ರುವ std::string& name, size_t len, ಧ್ರುವ ಶೂನ್ಯ* data, uint32_t pc_size, uint32_t pc) {
            auto tc = filter_tc(tc_base, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id);
            ಯದಿ (!tc.empty()) create_mm_pipelines(ಕೀಲಿ, tc, name, len, data, pc_size, pc, cm1_spec, ಅಸತ್ಯ, ಸತ್ಯ, 0, ಸತ್ಯ, cm1_pin);
        };
        auto cm1_create_quant = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಧ್ರುವ std::vector<vk_tile_config>& tc_base,
                                    ಧ್ರುವ std::string& name, size_t len, ಧ್ರುವ ಶೂನ್ಯ* data, uint32_t pc_size, uint32_t pc) {
            spec_fn_t qs = [&, type_a=ಕೀಲಿ.type_a](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಪ್ರತ್ಯಾಗಮ ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
            auto tc = filter_tc(tc_base, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id);
            ಯದಿ (!tc.empty()) create_mm_pipelines(ಕೀಲಿ, tc, name, len, data, pc_size, pc, qs, ಅಸತ್ಯ, ಸತ್ಯ, 0, ಸತ್ಯ, cm1_pin);
        };
        // int8 MMQ helper: per-type cm1 shader, warptile passed as-is (carries DEVICE_ARCH in
        // spec constant WARP_SIZE_IDX+1), subgroup size pinned to the warptile WARP element, no aligned variant.
        auto cm1_create_mmq = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಧ್ರುವ std::vector<vk_tile_config>& tc_base,
                                  ಧ್ರುವ std::string& name, size_t len, ಧ್ರುವ ಶೂನ್ಯ* data, uint32_t pc_size, uint32_t pc) {
            spec_fn_t identity = [](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ) { ಪ್ರತ್ಯಾಗಮ wt; };
            auto tc = filter_tc(tc_base, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id, ಸತ್ಯ);
            ಯದಿ (!tc.empty()) create_mm_pipelines(ಕೀಲಿ, tc, name, len, data, pc_size, pc, identity, ಅಸತ್ಯ, ಅಸತ್ಯ, 0, ಅಸತ್ಯ, ಸತ್ಯ);
        };

        std::vector<vk_tile_config> tc_mmq_cm1_int = {
            {s_warptile_mmq_cm1_int, s_mmq_wg_denoms, s_align},
            {m_warptile_mmq_cm1_int, m_mmq_wg_denoms, m_align},
            {l_warptile_mmq_cm1_int, l_mmq_wg_denoms, l_align},
        };
        std::vector<vk_tile_config> tc_mmq_cm1_int_k = {
            {s_warptile_mmq_cm1_int_k, s_mmq_cm1_wg_denoms_k, s_align},
            {m_warptile_mmq_cm1_int_k, m_mmq_cm1_wg_denoms_k, m_align},
            {l_warptile_mmq_cm1_int_k, l_mmq_cm1_wg_denoms_k, l_align},
        };

        // Some quants are not performant on RDNA4, those fall back to FP16 matmul
        ಧ್ರುವ ತಾರ್ಕಿಕ rdna3 = device->architecture == AMD_RDNA3;
        ಧ್ರುವ ತಾರ್ಕಿಕ rdna4 = device->architecture == AMD_RDNA4;

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f32_f32",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_data,     ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
        cm1_create({GGML_TYPE_F32, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f32_f16",     matmul_f32_f16_cm1_len,     matmul_f32_f16_cm1_data,     ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
        ಯದಿ (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm1_len, matmul_f16_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mm, "matmul_f16_f32_f16acc", matmul_f16_f32_f16acc_cm1_len, matmul_f16_f32_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
        }
        ಯದಿ (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f16",      matmul_f16_cm1_len,      matmul_f16_cm1_data,      ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f16_f32",  matmul_f16_f32_cm1_len,  matmul_f16_f32_cm1_data,  ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
        }
#ಯದಿ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ಯದಿ (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_bf16", matmul_bf16_cm1_len, matmul_bf16_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
        }
#endif
        ಪ್ರತಿ (ಧ್ರುವ auto type : non_lut_quant_types) {
            ಯದಿ (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mmq, "matmul_quant_f32_f16acc", matmul_quant_f32_f16acc_cm1_len, matmul_quant_f32_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mmq, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm1_len, matmul_quant_f16_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
            ಯದಿ (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq, "matmul_quant_f32",        matmul_quant_f32_cm1_len,        matmul_quant_f32_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq, "matmul_quant_f16",        matmul_quant_f16_cm1_len,        matmul_quant_f16_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#define X_CM1(TYPE, tstr) \
        ಯದಿ (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mmq, "matmul_" #tstr "_f32_f16acc", matmul_##tstr##_f32_f16acc_cm1_len, matmul_##tstr##_f32_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mmq, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm1_len, matmul_##tstr##_f16_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
        } \
        ಯದಿ (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq, "matmul_" #tstr "_f32",        matmul_##tstr##_f32_cm1_len,        matmul_##tstr##_f32_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm1_len,        matmul_##tstr##_f16_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1)
#ಯದಿ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ಯದಿ (device->ocp_fp4) {
#define X_CM1_OCP(TYPE, tstr) \
            ಯದಿ (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mmq, "matmul_" #tstr "_f32_ocp_f16acc", matmul_##tstr##_f32_ocp_f16acc_cm1_len, matmul_##tstr##_f32_ocp_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mmq, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm1_len, matmul_##tstr##_f16_ocp_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
            } \
            ಯದಿ (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq, "matmul_" #tstr "_f32_ocp",        matmul_##tstr##_f32_ocp_cm1_len,        matmul_##tstr##_f32_ocp_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm1_len,        matmul_##tstr##_f16_ocp_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_OCP)
#undef X_CM1_OCP
        } ಅನ್ಯಥಾ
#endif
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1)
        }
#undef X_CM1

        ಯದಿ (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_q4_0_q8_1",   matmul_q4_0_q8_1_cm1_len,   matmul_q4_0_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            ಯದಿ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_q4_1_q8_1",   matmul_q4_1_q8_1_cm1_len,   matmul_q4_1_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_q5_0_q8_1",   matmul_q5_0_q8_1_cm1_len,   matmul_q5_0_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            ಯದಿ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_q5_1_q8_1",   matmul_q5_1_q8_1_cm1_len,   matmul_q5_1_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_q8_0_q8_1",   matmul_q8_0_q8_1_cm1_len,   matmul_q8_0_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_iq4_nl_q8_1", matmul_iq4_nl_q8_1_cm1_len, matmul_iq4_nl_q8_1_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_cm1_len, matmul_iq4_xs_q8_1_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_mxfp4_q8_1",  matmul_mxfp4_q8_1_cm1_len,  matmul_mxfp4_q8_1_cm1_data,  ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int_k, "matmul_q3_k_q8_1",   matmul_q3_k_q8_1_cm1_len,   matmul_q3_k_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            ಯದಿ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_q4_k_q8_1",   matmul_q4_k_q8_1_cm1_len,   matmul_q4_k_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); }
            ಯದಿ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_q5_k_q8_1",   matmul_q5_k_q8_1_cm1_len,   matmul_q5_k_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int_k, "matmul_q6_k_q8_1",   matmul_q6_k_q8_1_cm1_len,   matmul_q6_k_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            ಯದಿ (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int_k, "matmul_nvfp4_q8_1",  matmul_nvfp4_q8_1_cm1_len,  matmul_nvfp4_q8_1_cm1_data,  ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); }
        }

        GGML_ASSERT(device->subgroup_ballot);

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_cm1_len, matmul_id_subgroup_f32_f32_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
        ಯದಿ (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc_mm, "matmul_id_subgroup_f16_f16acc",     matmul_id_subgroup_f16_f16acc_cm1_len,     matmul_id_subgroup_f16_f16acc_cm1_data,     ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸತ್ಯ, ಸತ್ಯ},  tc_mm, "matmul_id_subgroup_f16_f32_f16acc", matmul_id_subgroup_f16_f32_f16acc_cm1_len, matmul_id_subgroup_f16_f32_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
        }
        ಯದಿ (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_cm1_len,     matmul_id_subgroup_f16_cm1_data,     ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_cm1_len, matmul_id_subgroup_f16_f32_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
        }
#ಯದಿ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ಯದಿ (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm1_len, matmul_id_subgroup_bf16_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
        }
#endif
        ಪ್ರತಿ (ಧ್ರುವ auto type : non_lut_quant_types) {
            ಯದಿ (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ಸತ್ಯ, ಸತ್ಯ},  tc_mmq_id, "matmul_id_subgroup_quant_f32_f16acc", matmul_id_subgroup_quant_f32_f16acc_cm1_len, matmul_id_subgroup_quant_f32_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                cm1_create_quant({type, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc_mmq_id, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm1_len, matmul_id_subgroup_quant_f16_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            }
            ಯದಿ (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_id, "matmul_id_subgroup_quant_f32",        matmul_id_subgroup_quant_f32_cm1_len,        matmul_id_subgroup_quant_f32_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                cm1_create_quant({type, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_id, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm1_len,        matmul_id_subgroup_quant_f16_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#define X_CM1_ID(TYPE, tstr) \
        ಯದಿ (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ಸತ್ಯ, ಸತ್ಯ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_f16acc", matmul_id_subgroup_##tstr##_f32_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
            cm1_create({TYPE, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
        } \
        ಯದಿ (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32",        matmul_id_subgroup_##tstr##_f32_cm1_len,        matmul_id_subgroup_##tstr##_f32_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
            cm1_create({TYPE, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm1_len,        matmul_id_subgroup_##tstr##_f16_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1_ID)
#ಯದಿ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ಯದಿ (device->ocp_fp4) {
#define X_CM1_ID_OCP(TYPE, tstr) \
            ಯದಿ (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ಸತ್ಯ, ಸತ್ಯ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp_f16acc", matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
                cm1_create({TYPE, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
            } \
            ಯದಿ (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp",        matmul_id_subgroup_##tstr##_f32_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f32_ocp_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
                cm1_create({TYPE, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm1_data,        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID_OCP)
#undef X_CM1_ID_OCP
        } ಅನ್ಯಥಾ
#endif
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID)
        }
#undef X_CM1_ID

        ಯದಿ (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_0_q8_1",   matmul_id_subgroup_q4_0_q8_1_cm1_len,   matmul_id_subgroup_q4_0_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q4_1,   GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_1_q8_1",   matmul_id_subgroup_q4_1_q8_1_cm1_len,   matmul_id_subgroup_q4_1_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_0_q8_1",   matmul_id_subgroup_q5_0_q8_1_cm1_len,   matmul_id_subgroup_q5_0_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q5_1,   GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_1_q8_1",   matmul_id_subgroup_q5_1_q8_1_cm1_len,   matmul_id_subgroup_q5_1_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q8_0_q8_1",   matmul_id_subgroup_q8_0_q8_1_cm1_len,   matmul_id_subgroup_q8_0_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_nl_q8_1", matmul_id_subgroup_iq4_nl_q8_1_cm1_len, matmul_id_subgroup_iq4_nl_q8_1_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_cm1_len, matmul_id_subgroup_iq4_xs_q8_1_cm1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_id_subgroup_mxfp4_q8_1",  matmul_id_subgroup_mxfp4_q8_1_cm1_len,  matmul_id_subgroup_mxfp4_q8_1_cm1_data,  ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q3_k_q8_1",   matmul_id_subgroup_q3_k_q8_1_cm1_len,   matmul_id_subgroup_q3_k_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q4_K,   GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_k_q8_1",   matmul_id_subgroup_q4_k_q8_1_cm1_len,   matmul_id_subgroup_q4_k_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q5_K,   GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_k_q8_1",   matmul_id_subgroup_q5_k_q8_1_cm1_len,   matmul_id_subgroup_q5_k_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q6_k_q8_1",   matmul_id_subgroup_q6_k_q8_1_cm1_len,   matmul_id_subgroup_q6_k_q8_1_cm1_data,   ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            ಯದಿ (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_cm1_int_k, "matmul_id_subgroup_nvfp4_q8_1",  matmul_id_subgroup_nvfp4_q8_1_cm1_len,  matmul_id_subgroup_nvfp4_q8_1_cm1_data,  ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); }
        }
    } ಅನ್ಯಥಾ
#endif  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    {
        // Helper for subgroup path with dot2 selection and filtering
        auto sg_create = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಧ್ರುವ std::vector<vk_tile_config>& tc_base,
                             ಧ್ರುವ std::string& name, size_t len, ಧ್ರುವ ಶೂನ್ಯ* data, uint32_t pc_size, uint32_t pc,
                             uint32_t rsgs = 0) {
            auto tc = filter_tc(tc_base, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id);
            ಯದಿ (!tc.empty()) create_mm_pipelines(ಕೀಲಿ, tc, name, len, data, pc_size, pc,
                [&](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಪ್ರತ್ಯಾಗಮ ggml_vk_mul_mm_spec(wt, a); },
                ಅಸತ್ಯ, rsgs > 0, rsgs);
        };
        auto sg_create_quant = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಧ್ರುವ std::vector<vk_tile_config>& tc_base,
                                   ಧ್ರುವ std::string& name, size_t len, ಧ್ರುವ ಶೂನ್ಯ* data, uint32_t pc_size, uint32_t pc,
                                   uint32_t rsgs = 0) {
            auto tc = filter_tc(tc_base, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id);
            ಯದಿ (!tc.empty()) {
                spec_fn_t qs = [&, type_a=ಕೀಲಿ.type_a](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಪ್ರತ್ಯಾಗಮ ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
                create_mm_pipelines(ಕೀಲಿ, tc, name, len, data, pc_size, pc, qs, ಅಸತ್ಯ, rsgs > 0, rsgs);
            }
        };
#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        auto sg_create_mmq = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಧ್ರುವ std::vector<vk_tile_config>& tc_base,
                                 ಧ್ರುವ std::string& name, size_t len, ಧ್ರುವ ಶೂನ್ಯ* data, uint32_t pc_size, uint32_t pc,
                                 uint32_t rsgs = 0) {
            auto tc = filter_tc(tc_base, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id, ಸತ್ಯ);
            ಯದಿ (!tc.empty()) {
                spec_fn_t identity = [](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ) { ಪ್ರತ್ಯಾಗಮ wt; };
                create_mm_pipelines(ಕೀಲಿ, tc, name, len, data, pc_size, pc, identity, ಅಸತ್ಯ, rsgs > 0, rsgs, ಅಸತ್ಯ);
            }
        };
#endif

        std::vector<vk_tile_config> tc_id = {{s_warptile_id, s_wg_denoms, s_align}, {m_warptile_id, m_wg_denoms, m_align}, {l_warptile_id, l_wg_denoms, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid, l_mmq_wg_denoms, l_align}};

        ಯದಿ (device->fp16) {
            // FP16 subgroup path - with dot2 runtime selection
            #define SPV_DOT2(NAME) (device->dot2_f16 ? NAME ## _dot2_len : NAME ## _len), (device->dot2_f16 ? NAME ## _dot2_data : NAME ## _data)
            #define SPV_DOT2_F16ACC(NAME) (device->dot2_f16 ? NAME ## _dot2_f16acc_len : NAME ## _f16acc_len), (device->dot2_f16 ? NAME ## _dot2_f16acc_data : NAME ## _f16acc_data)

            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f32_f32", SPV_DOT2(matmul_f32_f32), ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f32_f16", SPV_DOT2(matmul_f32_f16), ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mm, "matmul_f16_f16acc",     SPV_DOT2_F16ACC(matmul_f16),     ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f16",            SPV_DOT2(matmul_f16),            ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mm, "matmul_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_f16_f32), ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f16_f32",        SPV_DOT2(matmul_f16_f32),        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            // BF16 - no dot2
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_bf16", matmul_bf16_len, matmul_bf16_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);

            ಪ್ರತಿ (ಧ್ರುವ auto type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mmq, "matmul_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_quant_f32), ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_quant({type, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq, "matmul_quant_f32",        SPV_DOT2(matmul_quant_f32),        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
    #define X_SG(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, ಅಸತ್ಯ, ಸತ್ಯ},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
            sg_create({TYPE, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG)
#undef X_SG

#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ಯದಿ (device->integer_dot_product) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_len, matmul_q2_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_len, matmul_q4_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_len, matmul_q4_1_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_len, matmul_q5_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_len, matmul_q5_1_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_len, matmul_q8_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_mxfp4_q8_1", matmul_mxfp4_q8_1_len, matmul_mxfp4_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_len, matmul_iq4_xs_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_len, matmul_q2_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_len, matmul_q3_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_len, matmul_q4_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_len, matmul_q5_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_len, matmul_q6_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_len, matmul_iq3_s_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
#endif

            ಯದಿ (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_size_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_id, "matmul_id_subgroup_f32_f32", SPV_DOT2(matmul_id_subgroup_f32_f32), ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc_id, "matmul_id_subgroup_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_subgroup_f16),     ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_id, "matmul_id_subgroup_f16",            SPV_DOT2(matmul_id_subgroup_f16),            ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸತ್ಯ, ಸತ್ಯ},  tc_id, "matmul_id_subgroup_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_f16_f32), ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_id, "matmul_id_subgroup_f16_f32",        SPV_DOT2(matmul_id_subgroup_f16_f32),        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_id, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_len, matmul_id_subgroup_bf16_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                ಪ್ರತಿ (ಧ್ರುವ auto type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ಸತ್ಯ, ಸತ್ಯ},  tc_mmqid, "matmul_id_subgroup_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_quant_f32), ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                    sg_create_quant({type, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_subgroup_quant_f32",        SPV_DOT2(matmul_id_subgroup_quant_f32),        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                }
        #define X_SG_ID_SUB(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ಸತ್ಯ, ಸತ್ಯ},  tc_mmqid, "matmul_id_subgroup_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_##tstr##_f32), ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size); \
                sg_create({TYPE, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32",        SPV_DOT2(matmul_id_subgroup_##tstr##_f32),        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB)
#undef X_SG_ID_SUB
#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                ಯದಿ (device->integer_dot_product) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_subgroup_q2_0_q8_1", matmul_id_subgroup_q2_0_q8_1_len, matmul_id_subgroup_q2_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_subgroup_q4_0_q8_1", matmul_id_subgroup_q4_0_q8_1_len, matmul_id_subgroup_q4_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_subgroup_q4_1_q8_1", matmul_id_subgroup_q4_1_q8_1_len, matmul_id_subgroup_q4_1_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_subgroup_q5_0_q8_1", matmul_id_subgroup_q5_0_q8_1_len, matmul_id_subgroup_q5_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_subgroup_q5_1_q8_1", matmul_id_subgroup_q5_1_q8_1_len, matmul_id_subgroup_q5_1_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_subgroup_q8_0_q8_1", matmul_id_subgroup_q8_0_q8_1_len, matmul_id_subgroup_q8_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_subgroup_mxfp4_q8_1", matmul_id_subgroup_mxfp4_q8_1_len, matmul_id_subgroup_mxfp4_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_len, matmul_id_subgroup_iq4_xs_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_subgroup_q2_k_q8_1", matmul_id_subgroup_q2_k_q8_1_len, matmul_id_subgroup_q2_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_subgroup_q3_k_q8_1", matmul_id_subgroup_q3_k_q8_1_len, matmul_id_subgroup_q3_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_subgroup_q4_k_q8_1", matmul_id_subgroup_q4_k_q8_1_len, matmul_id_subgroup_q4_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_subgroup_q5_k_q8_1", matmul_id_subgroup_q5_k_q8_1_len, matmul_id_subgroup_q5_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_subgroup_q6_k_q8_1", matmul_id_subgroup_q6_k_q8_1_len, matmul_id_subgroup_q6_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_subgroup_iq3_s_q8_1", matmul_id_subgroup_iq3_s_q8_1_len, matmul_id_subgroup_iq3_s_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                }
#endif
            } ಅನ್ಯಥಾ {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_f32_f32", SPV_DOT2(matmul_id_f32_f32), ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸತ್ಯ, ಸತ್ಯ},  tc_mm, "matmul_id_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_f16),     ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_f16",            SPV_DOT2(matmul_id_f16),            ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸತ್ಯ, ಸತ್ಯ},  tc_mm, "matmul_id_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_f16_f32), ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_f16_f32",        SPV_DOT2(matmul_id_f16_f32),        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_bf16", matmul_id_bf16_len, matmul_id_bf16_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                ಪ್ರತಿ (ಧ್ರುವ auto type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ಸತ್ಯ, ಸತ್ಯ},  tc_mmqid, "matmul_id_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_quant_f32), ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_quant({type, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_quant_f32",        SPV_DOT2(matmul_id_quant_f32),        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                }
        #define X_SG_ID(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ಸತ್ಯ, ಸತ್ಯ},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
                sg_create({TYPE, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                FOR_EACH_LUT_TYPE(X_SG_ID)
#undef X_SG_ID
#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                ಯದಿ (device->integer_dot_product) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_q2_0_q8_1", matmul_id_q2_0_q8_1_len, matmul_id_q2_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_q4_0_q8_1", matmul_id_q4_0_q8_1_len, matmul_id_q4_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_q4_1_q8_1", matmul_id_q4_1_q8_1_len, matmul_id_q4_1_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_q5_0_q8_1", matmul_id_q5_0_q8_1_len, matmul_id_q5_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_q5_1_q8_1", matmul_id_q5_1_q8_1_len, matmul_id_q5_1_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_q8_0_q8_1", matmul_id_q8_0_q8_1_len, matmul_id_q8_0_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_mxfp4_q8_1", matmul_id_mxfp4_q8_1_len, matmul_id_mxfp4_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int, "matmul_id_iq4_xs_q8_1", matmul_id_iq4_xs_q8_1_len, matmul_id_iq4_xs_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_q2_k_q8_1", matmul_id_q2_k_q8_1_len, matmul_id_q2_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_q3_k_q8_1", matmul_id_q3_k_q8_1_len, matmul_id_q3_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_q4_k_q8_1", matmul_id_q4_k_q8_1_len, matmul_id_q4_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_q5_k_q8_1", matmul_id_q5_k_q8_1_len, matmul_id_q5_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_q6_k_q8_1", matmul_id_q6_k_q8_1_len, matmul_id_q6_k_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid_int_k, "matmul_id_iq3_s_q8_1", matmul_id_iq3_s_q8_1_len, matmul_id_iq3_s_q8_1_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                }
#endif
            }
            #undef SPV_DOT2
            #undef SPV_DOT2_F16ACC
        } ಅನ್ಯಥಾ {
            // FP32-only fallback path
            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f32_f32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f32_f16", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f16",     matmul_f16_fp32_len,     matmul_f16_fp32_data,     ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_f16_f32", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_bf16",  matmul_bf16_fp32_len,    matmul_bf16_fp32_data,    ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);

            ಪ್ರತಿ (ಧ್ರುವ auto type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq, "matmul_quant_f32", matmul_quant_f32_fp32_len, matmul_quant_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
    #define X_SG_FP32(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq, "matmul_" #tstr "_f32", matmul_##tstr##_f32_fp32_len, matmul_##tstr##_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG_FP32)
#undef X_SG_FP32

#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ಯದಿ (device->integer_dot_product) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_fp32_len, matmul_q2_0_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_fp32_len, matmul_q4_0_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_fp32_len, matmul_q4_1_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_fp32_len, matmul_q5_0_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_fp32_len, matmul_q5_1_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_fp32_len, matmul_q8_0_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_fp32_len, matmul_iq4_xs_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_fp32_len, matmul_q2_k_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_fp32_len, matmul_q3_k_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_fp32_len, matmul_q4_k_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_fp32_len, matmul_q5_k_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_fp32_len, matmul_q6_k_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_fp32_len, matmul_iq3_s_q8_1_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
#endif

            ಯದಿ (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_size_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_id, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_fp32_len, matmul_id_subgroup_f32_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_id, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_fp32_len,     matmul_id_subgroup_f16_fp32_data,     ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_id, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_fp32_len, matmul_id_subgroup_f16_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_id, "matmul_id_subgroup_bf16",  matmul_id_subgroup_bf16_fp32_len,    matmul_id_subgroup_bf16_fp32_data,    ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size_16);
                ಪ್ರತಿ (ಧ್ರುವ auto type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_subgroup_quant_f32", matmul_id_subgroup_quant_f32_fp32_len, matmul_id_subgroup_quant_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                }
        #define X_SG_ID_SUB_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32", matmul_id_subgroup_##tstr##_f32_fp32_len, matmul_id_subgroup_##tstr##_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_size);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB_FP32)
#undef X_SG_ID_SUB_FP32
            } ಅನ್ಯಥಾ {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_f32_f32", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_f16",     matmul_id_f16_fp32_len,     matmul_id_f16_fp32_data,     ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_f16_f32", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mm, "matmul_id_bf16",  matmul_id_bf16_fp32_len,    matmul_id_bf16_fp32_data,    ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                ಪ್ರತಿ (ಧ್ರುವ auto type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_quant_f32", matmul_id_quant_f32_fp32_len, matmul_id_quant_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                }
        #define X_SG_ID_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ಸತ್ಯ, ಅಸತ್ಯ}, tc_mmqid, "matmul_id_" #tstr "_f32", matmul_id_##tstr##_f32_fp32_len, matmul_id_##tstr##_f32_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                FOR_EACH_LUT_TYPE(X_SG_ID_FP32)
#undef X_SG_ID_FP32
            }
        }
    }
#undef FOR_EACH_LUT_TYPE
#undef FOR_EACH_LUT_TYPE_NONFP4
#undef FOR_EACH_LUT_FP4_TYPE
    // BF16 fallback for coopmat devices without bf16 coopmat support
    ಯದಿ ((device->coopmat2 || device->coopmat_support)
#ಯದಿ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#endif
        ) {
        ಧ್ರುವ uint32_t s_warptile_wm_bf16 = device->subgroup_size == 8 ? 8 : 32;
        std::vector<vk_tile_config> tc_bf16_fb = {
            {{ subgroup_size_32, 32, 32, 16, s_warptile_wm_bf16, 32, 2, 2, 2, 1, subgroup_size_8 }, {32, 32, 1}, s_align},
            {{ 128, 64, 64, 16, mm_warp_8, 32, 2, 4, 2, 1, mm_warp_8 }, {64, 64, 1}, m_align},
            {{ 128, 128, 128, 16, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 }, {128, 128, 1}, l_align},
        };
        auto tc_bf16_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, ಅಸತ್ಯ);
        auto tc_bf16_id_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, ಸತ್ಯ);
        spec_fn_t bf16_spec = [&](ಧ್ರುವ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಪ್ರತ್ಯಾಗಮ ggml_vk_mul_mm_spec(wt, a); };
        ಯದಿ (!tc_bf16_filtered.empty()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ಅಸತ್ಯ, ಅಸತ್ಯ}, tc_bf16_filtered, "matmul_bf16", matmul_bf16_fp32_len, matmul_bf16_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_ತಳ್ಳು_constants), 3, bf16_spec);
        }
        ಯದಿ (!tc_bf16_id_filtered.empty()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸತ್ಯ, ಅಸತ್ಯ}, tc_bf16_id_filtered, "matmul_id_bf16", matmul_id_bf16_fp32_len, matmul_id_bf16_fp32_data, ಪರಿಮಾಣ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, bf16_spec);
        }
    }

    // Set up tile selector functions
    ಯದಿ (device->coopmat2) {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t shader_core_ಎಣಿಕೆ,
                                          ಧ್ರುವ std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ಯದಿ (configs.size() <= 1) ಪ್ರತ್ಯಾಗಮ 0;
            uint32_t last = (uint32_t)configs.size() - 1;
            ಯದಿ (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                ಪ್ರತ್ಯಾಗಮ (n > crossover) ? 1 : 0;
            }
            // 3+ configs: s=0, m=1, l=2
            ಧ್ರುವ uint32_t tiles_l = CEIL_DIV(m, configs[last].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[last].unaligned->wg_denoms[1]);
            ಧ್ರುವ uint32_t tiles_m = CEIL_DIV(m, configs[1].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[1].unaligned->wg_denoms[1]);
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            ತಾರ್ಕಿಕ prefer_large = tiles_m > shader_core_ಎಣಿಕೆ || tiles_l > shader_core_ಎಣಿಕೆ ||
                                (tiles_l <= shader_core_ಎಣಿಕೆ / 3 && tiles_m > shader_core_ಎಣಿಕೆ / 2);
            ಯದಿ (n > crossover_large && prefer_large) ಪ್ರತ್ಯಾಗಮ last;
            uint32_t crossover_medium_m = configs[0].unaligned->wg_denoms[0];
            uint32_t crossover_medium_n = configs[0].unaligned->wg_denoms[1];
            ಯದಿ (m > crossover_medium_m && n > crossover_medium_n) ಪ್ರತ್ಯಾಗಮ 1;
            ಪ್ರತ್ಯಾಗಮ 0;
        };
        device->matmul_id_tile_selector = [](uint32_t /*m*/, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                             ಧ್ರುವ std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ಯದಿ (configs.size() <= 1) ಪ್ರತ್ಯಾಗಮ 0;
            uint32_t last = (uint32_t)configs.size() - 1;
            ಯದಿ (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                ಪ್ರತ್ಯಾಗಮ (n > crossover) ? 1 : 0;
            }
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            ಯದಿ (n > crossover_large) ಪ್ರತ್ಯಾಗಮ last;
            uint32_t crossover_medium = configs[0].unaligned->wg_denoms[1];
            ಯದಿ (n > crossover_medium) ಪ್ರತ್ಯಾಗಮ 1;
            ಪ್ರತ್ಯಾಗಮ 0;
        };
    } ಅನ್ಯಥಾ {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                          ಧ್ರುವ std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ಯದಿ (configs.size() <= 1) ಪ್ರತ್ಯಾಗಮ 0;
            ಯದಿ (m <= 32 || n <= 32) ಪ್ರತ್ಯಾಗಮ 0;
            ಯದಿ (configs.size() == 2) ಪ್ರತ್ಯಾಗಮ 1;
            ಯದಿ (m <= 64 || n <= 64) ಪ್ರತ್ಯಾಗಮ 1;
            ಪ್ರತ್ಯಾಗಮ (uint32_t)configs.size() - 1;
        };
        device->matmul_id_tile_selector = device->matmul_tile_selector;
    }

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    auto ಧ್ರುವ &rm_iq_int = [](uint32_t i) { ಪ್ರತ್ಯಾಗಮ i == 0 ? 8u : 4u; };
    ಯದಿ (device->vendor_id == VK_VENDOR_ID_AMD) {
        ಯದಿ (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } ಅನ್ಯಥಾ ಯದಿ (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    // RDNA3/4: above four columns, static 4 rows for all types bench faster than the default
    ಧ್ರುವ ತಾರ್ಕಿಕ is_rdna3_or_4 = device->vendor_id == VK_VENDOR_ID_AMD && (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);
    auto ಧ್ರುವ &rm_int_n = [&](uint32_t rows, uint32_t i) { ಪ್ರತ್ಯಾಗಮ (is_rdna3_or_4 && i >= 4) ? 4u : rows; };
    // RDNA3/4 and NVIDIA except pre-Turing: use 4 rows for MUL_MAT_ID MMVQ.
    auto ಧ್ರುವ &rm_id = [&](uint32_t rows) {
        ಯದಿ (device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) {
            ಪ್ರತ್ಯಾಗಮ 4u;
        }
        ಪ್ರತ್ಯಾಗಮ is_rdna3_or_4 ? 4u : rows;
    };
    uint32_t rm_iq = 2 * rm_kq;

    ಧ್ರುವ ತಾರ್ಕಿಕ use_subgroups = device->subgroup_arithmetic;
    // The Imagination proprietary compiler rejects the subgroup-only dequant mul_mat_vec
    // shaders that require a subgroup size >= 16; fall back to shared-memory reduction.
    ಧ್ರುವ ತಾರ್ಕಿಕ is_imagination_proprietary =
        device->driver_id == vk::DriverId::eImaginationProprietary;
    // Ensure a subgroup size >= 16 is available
    ಧ್ರುವ ತಾರ್ಕಿಕ use_subgroups16 = use_subgroups && subgroup_min_size_16 && !is_imagination_proprietary;

    ಧ್ರುವ uint32_t subgroup_size = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control && device->subgroup_min_size <= 16 && device->subgroup_max_size >= 16) ? 16 : device->subgroup_size;
    ಧ್ರುವ uint32_t subgroup_size16 = std::max(subgroup_size, 16u);

    ಧ್ರುವ uint32_t force_subgroup_size = use_subgroups ? subgroup_size : 0;
    ಧ್ರುವ uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    ಸ್ಥಿರ constexpr uint32_t mul_mat_vec_ಅಂಕೆ_bindings = 5;
    ಸ್ಥಿರ constexpr uint32_t mul_mat_vec_id_ಅಂಕೆ_bindings = 6;

#ಯದಿ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
#define OCP_DMMV_LEN(NAME, REDUC)  (device->ocp_fp4 ? NAME ## _ocp_len[REDUC]  : NAME ## _len[REDUC])
#define OCP_DMMV_DATA(NAME, REDUC) (device->ocp_fp4 ? NAME ## _ocp_data[REDUC] : NAME ## _data[REDUC])
#ಅನ್ಯಥಾ
#define OCP_DMMV_LEN(NAME, REDUC)  NAME ## _len[REDUC]
#define OCP_DMMV_DATA(NAME, REDUC) NAME ## _data[REDUC]
#endif

    ಪ್ರತಿ (uint32_t w = 0; w < DMMV_WG_SIZE_ಎಣಿಕೆ; ++w) {
        ಧ್ರುವ uint32_t wg_size_subgroup   = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size : (subgroup_size * 4);
        ಧ್ರುವ uint32_t wg_size_subgroup16 = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        ಧ್ರುವ shader_reduction_mode reduc = (use_subgroups && w == DMMV_WG_SIZE_SUBGROUP) ? SHADER_REDUCTION_MODE_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_SIZE_LARGE) ? SHADER_REDUCTION_MODE_HYBRID :
                                            SHADER_REDUCTION_MODE_SHMEM;

        ಧ್ರುವ shader_reduction_mode reduc16 = (use_subgroups16 && w == DMMV_WG_SIZE_SUBGROUP) ? SHADER_REDUCTION_MODE_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_SIZE_LARGE) ? SHADER_REDUCTION_MODE_HYBRID :
                                              SHADER_REDUCTION_MODE_SHMEM;

        ಪ್ರತಿ (uint32_t i = 0; i < mul_mat_vec_max_cols; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_data[reduc],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {1, 1, 1}, {wg_size_subgroup, 1, i+1}, 1, ಅಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_data[reduc],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, ಅಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, ಅಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f32_f32", arr_dmmv_q2_0_f32_f32_len[reduc], arr_dmmv_q2_0_f32_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f32_f32", arr_dmmv_tq1_0_f32_f32_len[reduc16], arr_dmmv_tq1_0_f32_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f32_f32", arr_dmmv_tq2_0_f32_f32_len[reduc16], arr_dmmv_tq2_0_f32_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_data[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_data[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_data[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_data[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_data[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f32_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f32_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_data[reduc],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {1, 1, 1}, {wg_size_subgroup, 1, i+1}, 1, ಅಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_data[reduc],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, ಅಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, ಅಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f16_f32", arr_dmmv_q2_0_f16_f32_len[reduc], arr_dmmv_q2_0_f16_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq, i+1}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f16_f32", arr_dmmv_tq1_0_f16_f32_len[reduc16], arr_dmmv_tq1_0_f16_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f16_f32", arr_dmmv_tq2_0_f16_f32_len[reduc16], arr_dmmv_tq2_0_f16_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_data[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_data[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_data[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_data[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_data[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_data[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_data[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_data[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f16_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_mxfp4_f16_f32, reduc16), "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f16_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_nvfp4_f16_f32, reduc16), "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);

#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ಯದಿ (device->integer_dot_product) {
                ಧ್ರುವ uint32_t subgroup_size_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
                ಧ್ರುವ uint32_t wg_size_subgroup_int = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size_int : (subgroup_size_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_q8_1_f32", arr_dmmv_q2_0_q8_1_f32_len[reduc], arr_dmmv_q2_0_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(2*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(2*rm_stdq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {1*rm_iq_int(i), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {1*rm_iq_int(i), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ4_XS][i], "mul_mat_vec_iq4_xs_q8_1_f32", arr_dmmv_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_iq4_xs_q8_1_f32_data[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);

            }
#endif // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_data[reduc],     "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {1, 1, 1}, {wg_size_subgroup, 1}, 1, ಅಸತ್ಯ, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_data[reduc],     "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {2, 1, 1}, {wg_size_subgroup, 2}, 1, ಅಸತ್ಯ, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_data[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {2, 1, 1}, {wg_size_subgroup, 2}, 1, ಅಸತ್ಯ, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_f32",       arr_dmmv_id_q2_0_f32_f32_len[reduc],    arr_dmmv_id_q2_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_data[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_data[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq}, 1, ಸತ್ಯ, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ1_0], "mul_mat_vec_id_tq1_0_f32",     arr_dmmv_id_tq1_0_f32_f32_len[reduc16],   arr_dmmv_id_tq1_0_f32_f32_data[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ2_0], "mul_mat_vec_id_tq2_0_f32",     arr_dmmv_id_tq2_0_f32_f32_len[reduc16],   arr_dmmv_id_tq2_0_f32_f32_data[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_data[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_data[reduc16],  "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_mxfp4_f32_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_id_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_nvfp4_f32_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_id_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, ಸತ್ಯ, use_subgroups16, force_subgroup_size16);

#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ಯದಿ (device->integer_dot_product) {
            ಧ್ರುವ uint32_t subgroup_size_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
            ಧ್ರುವ uint32_t wg_size_subgroup_int = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size_int : (subgroup_size_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_q8_1_f32", arr_dmmv_id_q2_0_q8_1_f32_len[reduc], arr_dmmv_id_q2_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(2*rm_kq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(2*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(2*rm_stdq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(2*rm_kq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_kq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_kq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_kq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_kq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {1*rm_iq_int(0), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(0)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {1*rm_iq_int(0), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(0)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ4_XS], "mul_mat_vec_id_iq4_xs_q8_1_f32", arr_dmmv_id_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_id_iq4_xs_q8_1_f32_data[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ಸತ್ಯ, use_subgroups, subgroup_size_int);
        }
#endif // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#undef OCP_DMMV_DATA
#undef OCP_DMMV_LEN

#ಯದಿ !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_UNUSED(rm_stdq_int);
    GGML_UNUSED(rm_kq_int);
    GGML_UNUSED(is_rdna3_or_4);
    GGML_UNUSED(rm_int_n);
    GGML_UNUSED(rm_id);
    GGML_UNUSED(rm_iq_int);
#endif

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_data,  "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_0], "dequant_q2_0", dequant_q2_0_len, dequant_q2_0_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_transpose[GGML_TYPE_Q8_0], "dequant_q8_0_transpose", dequant_q8_0_transpose_len, dequant_q8_0_transpose_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ1_0], "dequant_tq1_0", dequant_tq1_0_len, dequant_tq1_0_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ2_0], "dequant_tq2_0", dequant_tq2_0_len, dequant_tq2_0_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_data,   "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_data,   "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_data,  "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_data,   "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_data, "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_data,   "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_data,  "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_data,  "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_data,   "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_data,   "main", 2, 5 * ಪರಿಮಾಣ(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F32 ], "get_rows_f32",  get_rows_f32_len,  get_rows_f32_data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F16 ], "get_rows_f16",  get_rows_f16_len,  get_rows_f16_data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_BF16], "get_rows_bf16", get_rows_bf16_len, get_rows_bf16_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q1_0], "get_rows_q1_0", get_rows_q1_0_len, get_rows_q1_0_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q2_0], "get_rows_q2_0", get_rows_q2_0_len, get_rows_q2_0_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_0], "get_rows_q4_0", get_rows_q4_0_len, get_rows_q4_0_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_1], "get_rows_q4_1", get_rows_q4_1_len, get_rows_q4_1_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_0], "get_rows_q5_0", get_rows_q5_0_len, get_rows_q5_0_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_1], "get_rows_q5_1", get_rows_q5_1_len, get_rows_q5_1_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q8_0], "get_rows_q8_0", get_rows_q8_0_len, get_rows_q8_0_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q2_K], "get_rows_q2_k", get_rows_q2_k_len, get_rows_q2_k_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q3_K], "get_rows_q3_k", get_rows_q3_k_len, get_rows_q3_k_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_K], "get_rows_q4_k", get_rows_q4_k_len, get_rows_q4_k_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_K], "get_rows_q5_k", get_rows_q5_k_len, get_rows_q5_k_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q6_K], "get_rows_q6_k", get_rows_q6_k_len, get_rows_q6_k_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_TQ1_0], "get_rows_tq1_0", get_rows_tq1_0_len, get_rows_tq1_0_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_TQ2_0], "get_rows_tq2_0", get_rows_tq2_0_len, get_rows_tq2_0_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_rows_iq1_s_len,   get_rows_iq1_s_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_rows_iq1_m_len,   get_rows_iq1_m_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_rows_iq2_xxs_len, get_rows_iq2_xxs_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_rows_iq2_xs_len,  get_rows_iq2_xs_data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_rows_iq2_s_len,   get_rows_iq2_s_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_rows_iq3_xxs_len, get_rows_iq3_xxs_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_rows_iq3_s_len,   get_rows_iq3_s_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_rows_iq4_xs_len,  get_rows_iq4_xs_data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_rows_iq4_nl_len,  get_rows_iq4_nl_data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_rows_mxfp4_len,   get_rows_mxfp4_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_rows_nvfp4_len,   get_rows_nvfp4_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_I32],     "get_rows_i32",     get_rows_i32_len,     get_rows_i32_data,     "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_rows_f32_f32_len,  get_rows_f32_f32_data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_rows_f16_f32_len,  get_rows_f16_f32_data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_rows_bf16_f32_len, get_rows_bf16_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_rows_q1_0_f32_len, get_rows_q1_0_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q2_0], "get_rows_q2_0_f32", get_rows_q2_0_f32_len, get_rows_q2_0_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_rows_q4_0_f32_len, get_rows_q4_0_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_rows_q4_1_f32_len, get_rows_q4_1_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_rows_q5_0_f32_len, get_rows_q5_0_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_rows_q5_1_f32_len, get_rows_q5_1_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_rows_q8_0_f32_len, get_rows_q8_0_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_rows_q2_k_f32_len, get_rows_q2_k_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_rows_q3_k_f32_len, get_rows_q3_k_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_rows_q4_k_f32_len, get_rows_q4_k_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_rows_q5_k_f32_len, get_rows_q5_k_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_rows_q6_k_f32_len, get_rows_q6_k_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_TQ1_0], "get_rows_tq1_0_f32", get_rows_tq1_0_f32_len, get_rows_tq1_0_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_TQ2_0], "get_rows_tq2_0_f32", get_rows_tq2_0_f32_len, get_rows_tq2_0_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_rows_iq1_s_f32_len,   get_rows_iq1_s_f32_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_rows_iq1_m_f32_len,   get_rows_iq1_m_f32_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_rows_iq2_xxs_f32_len, get_rows_iq2_xxs_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_rows_iq2_xs_f32_len,  get_rows_iq2_xs_f32_data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_rows_iq2_s_f32_len,   get_rows_iq2_s_f32_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_rows_iq3_xxs_f32_len, get_rows_iq3_xxs_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_rows_iq3_s_f32_len,   get_rows_iq3_s_f32_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_rows_iq4_xs_f32_len,  get_rows_iq4_xs_f32_data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_rows_iq4_nl_f32_len,  get_rows_iq4_nl_f32_data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_rows_mxfp4_f32_len,   get_rows_mxfp4_f32_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_rows_nvfp4_f32_len,   get_rows_nvfp4_f32_data,   "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_back_f32, "get_rows_back_f32", get_rows_back_f32_len, get_rows_back_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1, ಸತ್ಯ);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_split_k_reduce, "split_k_reduce", split_k_reduce_len, split_k_reduce_data, "main", 2, 2 * ಪರಿಮಾಣ(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_split_k_reduce, "fa_split_k_reduce", fa_split_k_reduce_len, fa_split_k_reduce_data, "main", 3, ಪರಿಮಾಣ(vk_op_flash_attn_split_k_reduce_ತಳ್ಳು_constants), {1, device->subgroup_size, 1}, {device->subgroup_size}, 1, ಸತ್ಯ);

#ಯದಿ defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ಯದಿ (device->vendor_id == VK_VENDOR_ID_INTEL && (device->architecture == INTEL_XE2 || (device->architecture == INTEL_XE1 && device->coopmat_support && device->uma))) {
        auto upper_power_of_2 = [&](uint32_t in) {
            GGML_ASSERT(in != 0);
            ಯದಿ (in <= 1) ಪ್ರತ್ಯಾಗಮ 1u;
            uint32_t ret = in - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            ಪ್ರತ್ಯಾಗಮ ret + 1;
            };

        uint32_t xe_native_sub_group_size = 16;
        ಯದಿ (device->architecture == INTEL_XE1) {
            xe_native_sub_group_size = 8;
        }

        ಪ್ರತಿ (auto& it : device->pipeline_xe_fa_decode_dual_phases) {
            ಧ್ರುವ uint32_t split_p_chunk = 32;
            auto HdQk = it.first;
            auto& pipelines = it.second;
            uint32_t head_dim_qk = std::get<0>(HdQk);
            uint32_t head_dim_pv = std::get<1>(HdQk);
            uint32_t gqa_ratio = std::get<2>(HdQk);
            uint32_t q_len = std::get<3>(HdQk);
            ಧ್ರುವ uint32_t out_dim_per_wg = gqa_ratio > 16 ? 8 : 16;
            uint32_t aligned_q_len = upper_power_of_2(q_len);
            uint32_t group_sz_ph1 = std::min(std::max(aligned_q_len * xe_native_sub_group_size, 64u), 256u);
            uint32_t out_per_wg_ph1 = std::min(q_len, 256u / xe_native_sub_group_size);
            uint32_t aligned_gqa_ratio = upper_power_of_2(gqa_ratio);
            uint32_t split_p_per_iter_ph2 = 256;
            uint32_t split_p_per_warp = 16;
            uint32_t group_sz_ph2 = (split_p_per_iter_ph2 / split_p_per_warp) * xe_native_sub_group_size;
            uint32_t out_per_wg_ph2 = std::min(std::max(16u / aligned_gqa_ratio, 1u), q_len);
            ggml_vk_create_pipeline(device, pipelines.first, "xe_fa_decode_ph1", fa_decode_ph1_cm1_len, fa_decode_ph1_cm1_data, "main", 5, ಪರಿಮಾಣ(vk_fa_xe_opt_ತಳ್ಳು_constants), { 1, 32, 1 }, { group_sz_ph1, gqa_ratio, head_dim_qk, xe_native_sub_group_size, split_p_chunk, out_per_wg_ph1 }, 1, ಅಸತ್ಯ, ಸತ್ಯ, xe_native_sub_group_size);
            ggml_vk_create_pipeline(device, pipelines.second, "xe_fa_decode_ph2", fa_decode_ph2_cm1_len, fa_decode_ph2_cm1_data, "main", 5, ಪರಿಮಾಣ(vk_fa_xe_opt_ತಳ್ಳು_constants), { 1, 1, 1 }, { group_sz_ph2, gqa_ratio, head_dim_pv, out_per_wg_ph2, xe_native_sub_group_size, split_p_per_iter_ph2, split_p_chunk, out_dim_per_wg }, 1, ಅಸತ್ಯ, ಸತ್ಯ, xe_native_sub_group_size);
        }
    }
#endif

    ಪ್ರತಿ (auto &it : device->pipeline_fa_mask_opt) {
        auto BrBc = it.first;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_mask_opt_len, fa_mask_opt_data, "main", 2, ಪರಿಮಾಣ(vk_op_flash_attn_mask_opt_ತಳ್ಳು_constants), {1, 1, 1}, {128, 128 / device->subgroup_size, BrBc.first, BrBc.second}, 1, ಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);
    }

    {
        // Large workgroup so the per-row KV scan parallelizes; capped to device limits.
        ಧ್ರುವ uint32_t compact_max = std::min({1024u, device->properties.limits.maxComputeWorkGroupInvocations, device->properties.limits.maxComputeWorkGroupSize[0]});

        // Fast ballot prefix-sum path when the device supports full subgroups; otherwise
        // a shared-memory prefix-sum fallback. Both emit a deterministic ascending list.
        device->fa_sparse_compact_use_subgroups = device->subgroup_ballot && device->subgroup_require_full_support;
        ಯದಿ (device->fa_sparse_compact_use_subgroups) {
            ಧ್ರುವ uint32_t compact_wg = std::max(device->subgroup_size, (compact_max / device->subgroup_size) * device->subgroup_size);
            ಧ್ರುವ uint32_t compact_ಅಂಕೆ_sg = compact_wg / device->subgroup_size;
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact_subgroup, "fa_sparse_compact_subgroup", fa_sparse_compact_subgroup_len, fa_sparse_compact_subgroup_data, "main", 2, ಪರಿಮಾಣ(vk_op_flash_attn_sparse_compact_ತಳ್ಳು_constants), {1, 1, 1}, {compact_wg, compact_ಅಂಕೆ_sg}, 1, ಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);
        } ಅನ್ಯಥಾ {
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact, "fa_sparse_compact", fa_sparse_compact_len, fa_sparse_compact_data, "main", 2, ಪರಿಮಾಣ(vk_op_flash_attn_sparse_compact_ತಳ್ಳು_constants), {1, 1, 1}, {compact_max}, 1, ಸತ್ಯ);
        }
    }

    ಯದಿ (device->subgroup_clustered && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_data, "main", 2, ಪರಿಮಾಣ(vk_quantize_q8_1_ತಳ್ಳು_constants), {32 * device->subgroup_size / 8, 1, 1}, { device->subgroup_size }, 1, ಸತ್ಯ, ಸತ್ಯ);
    } ಅನ್ಯಥಾ {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_data, "main", 2, ಪರಿಮಾಣ(vk_quantize_q8_1_ತಳ್ಳು_constants), {32 * device->subgroup_size / 8, 1, 1}, { device->subgroup_size }, 1);
    }

    ಪ್ರತಿ (uint32_t i = 0; i < p021_max_gqa_ratio; ++i) {
        ಯದಿ (device->subgroup_arithmetic && device->subgroup_require_full_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_add_len, mul_mat_vec_p021_f16_f32_subgroup_add_data, "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_p021_ತಳ್ಳು_constants), {1, 1, 1}, {device->subgroup_size, i + 1}, 1, ಸತ್ಯ, ಸತ್ಯ);
        } ಅನ್ಯಥಾ {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_data,              "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_p021_ತಳ್ಳು_constants), {1, 1, 1}, {device->subgroup_size, i + 1}, 1, ಸತ್ಯ);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_data, "main", mul_mat_vec_ಅಂಕೆ_bindings, ಪರಿಮಾಣ(vk_mat_vec_nc_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_group_norm_f32, "group_norm_f32", group_norm_f32_len, group_norm_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 0}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_f32, "rms_norm_mul_add_f32", rms_norm_mul_add_f32_len, rms_norm_mul_add_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1, 0}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_mul_f32, "rms_norm_mul_add_mul_f32", rms_norm_mul_add_f32_len, rms_norm_mul_add_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1, 1}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_partials_f32, "rms_norm_mul_add_partials_f32", rms_norm_mul_add_partials_f32_len, rms_norm_mul_add_partials_f32_data, "main", 6, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1, 0}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_mul_partials_f32, "rms_norm_mul_add_mul_partials_f32", rms_norm_mul_add_partials_f32_len, rms_norm_mul_add_partials_f32_data, "main", 6, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1, 1}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_set_rows_f32_f32, "rms_norm_set_rows_f32_f32", rms_norm_set_rows_f32_f32_len, rms_norm_set_rows_f32_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 0}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_set_rows_f32_f16, "rms_norm_set_rows_f32_f16", rms_norm_set_rows_f32_f16_len, rms_norm_set_rows_f32_f16_data, "main", 4, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 0}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 0}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1}, 1, ಸತ್ಯ);

    ಯದಿ (ಪರಿಮಾಣ(vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants) <= device->properties.limits.max_ತಳ್ಳು_Constants_Size) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_data, "main", 7, ಪರಿಮಾಣ(vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1}, 1, ಸತ್ಯ);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_data, "main", 7, ಪರಿಮಾಣ(vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1}, 1, ಸತ್ಯ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_back_f32, "rms_norm_back_f32", rms_norm_back_f32_len, rms_norm_back_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_32, "cpy_32", cpy_32_len, cpy_32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_16, "cpy_16", cpy_16_len, cpy_16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_data,"main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_data,"main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f16,"cpy_bf16_f16",cpy_bf16_f16_len,cpy_bf16_f16_data,"main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_32, "contig_cpy_32", contig_cpy_32_len, contig_cpy_32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_16, "contig_cpy_16", contig_cpy_16_len, contig_cpy_16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_data,"main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_data,"main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f16,"contig_cpy_bf16_f16",contig_cpy_bf16_f16_len,contig_cpy_bf16_f16_data,"main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_32, "cpy_transpose_02_32", cpy_transpose_02_32_len, cpy_transpose_02_32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_16, "cpy_transpose_02_16", cpy_transpose_02_16_len, cpy_transpose_02_16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q2_0], "cpy_f32_q2_0", cpy_f32_q2_0_len, cpy_f32_q2_0_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);

#define SET_ROWS(src_idx, src, itype) \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_F32],  "set_rows_" #src "_f32" #itype,  set_rows_ ## src ## _f32 ## itype ## _len,  set_rows_ ## src ## _f32 ## itype ## _data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_F16],  "set_rows_" #src "_f16" #itype,  set_rows_ ## src ## _f16 ## itype ## _len,  set_rows_ ## src ## _f16 ## itype ## _data,  "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_BF16], "set_rows_" #src "_bf16" #itype, set_rows_ ## src ## _bf16 ## itype ## _len, set_rows_ ## src ## _bf16 ## itype ## _data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q1_0], "set_rows_" #src "_q1_0" #itype, set_rows_ ## src ## _q1_0 ## itype ## _len, set_rows_ ## src ## _q1_0 ## itype ## _data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q2_0], "set_rows_" #src "_q2_0" #itype, set_rows_ ## src ## _q2_0 ## itype ## _len, set_rows_ ## src ## _q2_0 ## itype ## _data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q4_0], "set_rows_" #src "_q4_0" #itype, set_rows_ ## src ## _q4_0 ## itype ## _len, set_rows_ ## src ## _q4_0 ## itype ## _data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q4_1], "set_rows_" #src "_q4_1" #itype, set_rows_ ## src ## _q4_1 ## itype ## _len, set_rows_ ## src ## _q4_1 ## itype ## _data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q5_0], "set_rows_" #src "_q5_0" #itype, set_rows_ ## src ## _q5_0 ## itype ## _len, set_rows_ ## src ## _q5_0 ## itype ## _data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q5_1], "set_rows_" #src "_q5_1" #itype, set_rows_ ## src ## _q5_1 ## itype ## _len, set_rows_ ## src ## _q5_1 ## itype ## _data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q8_0], "set_rows_" #src "_q8_0" #itype, set_rows_ ## src ## _q8_0 ## itype ## _len, set_rows_ ## src ## _q8_0 ## itype ## _data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_IQ4_NL], "set_rows_" #src "_iq4_nl" #itype, set_rows_ ## src ## _iq4_nl ## itype ## _len, set_rows_ ## src ## _iq4_nl ## itype ## _data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ಸತ್ಯ);

    SET_ROWS(0, f32, _i32)
    SET_ROWS(0, f32, _i64)
    SET_ROWS(1, f16, _i32)
    SET_ROWS(1, f16, _i64)
#undef SET_ROWS


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q2_0], "cpy_q2_0_f32", cpy_q2_0_f32_len, cpy_q2_0_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q2_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    auto get_suffix = [](ತಾರ್ಕಿಕ src0_f16, ತಾರ್ಕಿಕ src1_f16, ತಾರ್ಕಿಕ dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        ಪ್ರತ್ಯಾಗಮ s;
    };

#define CREATE_ದ್ವಿಮಾನ(name, namemod, spec, bindings) \
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s0 : {0,1}) ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s1 : {0,1}) ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## _data[s0][s1][d], \
                                "main", (bindings), ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, spec, 1);

    CREATE_ದ್ವಿಮಾನ(add, , {0}, 4)
    CREATE_ದ್ವಿಮಾನ(add, _norepeat, {1}, 4)
    CREATE_ದ್ವಿಮಾನ(sub, , {0}, 3)
    CREATE_ದ್ವಿಮಾನ(sub, _norepeat, {1}, 3)
    CREATE_ದ್ವಿಮಾನ(mul, , {0}, 3)
    CREATE_ದ್ವಿಮಾನ(mul, _norepeat, {1}, 3)
    CREATE_ದ್ವಿಮಾನ(div, , {0}, 3)
    CREATE_ದ್ವಿಮಾನ(div, _norepeat, {1}, 3)
    CREATE_ದ್ವಿಮಾನ(add_rms, , {0}, 4)
    CREATE_ದ್ವಿಮಾನ(add_rms, _norepeat, {1}, 4)
#undef CREATE_ದ್ವಿಮಾನ

    ಯದಿ (device->multi_add) {
        ಪ್ರತಿ (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_add[i],     "multi_add_f32_"     + std::to_string(i+1), multi_add_f32_len,     multi_add_f32_data,     "main", MAX_PARAMETER_ಎಣಿಕೆ, ಪರಿಮಾಣ(vk_op_multi_add_ತಳ್ಳು_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_add_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_add_rms_f32_len, multi_add_rms_f32_data, "main", MAX_PARAMETER_ಎಣಿಕೆ, ಪರಿಮಾಣ(vk_op_multi_add_ತಳ್ಳು_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_add_id_f32, "add_id_f32", add_id_f32_len, add_id_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_add_id_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_set_f32, "set_f32", acc_f32_len, acc_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {0, 0}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_i8, "concat_i8", concat_i8_len, concat_i8_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i16, "concat_i16", concat_i16_len, concat_i16_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i64, "concat_i64", concat_i64_len, concat_i64_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_nearest_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_upscale_ತಳ್ಳು_constants), {512, 1, 1}, {GGML_SCALE_MODE_NEAREST}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_upscale_ತಳ್ಳು_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_upscale_ತಳ್ಳು_constants), {512, 1, 1}, {GGML_SCALE_MODE_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_upscale_ತಳ್ಳು_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_pad_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pad_reflect_1d_f32, "pad_reflect_1d_f32", pad_reflect_1d_f32_len, pad_reflect_1d_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i32, "repeat_i32", repeat_i32_len, repeat_i32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_back_f32, "repeat_back_f32", repeat_back_f32_len, repeat_back_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i16, "repeat_i16", repeat_i16_len, repeat_i16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

#define CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 2, ಪರಿಮಾಣ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(elu)
    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_quick)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(sqr)
    CREATE_UNARY(sqrt)
    CREATE_UNARY(sin)
    CREATE_UNARY(cos)
    CREATE_UNARY(clamp)
    CREATE_UNARY(leaky_relu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(abs)
    CREATE_UNARY(softplus)
    CREATE_UNARY(step)
    CREATE_UNARY(round)
    CREATE_UNARY(ceil)
    CREATE_UNARY(floor)
    CREATE_UNARY(trunc)
    CREATE_UNARY(sgn)
    CREATE_UNARY(exp)
    CREATE_UNARY(expm1)
#undef CREATE_UNARY

// spec constants: {norepeat, op_on_b}
#define CREATE_UNARY_MUL(name, idx) \
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ dt = 0; dt < 2; ++dt) { \
        ಧ್ರುವ size_t len_ = dt ? name ## _mul_f16_len : name ## _mul_f32_len; \
        ಧ್ರುವ unsigned ವರ್ಣ * data_ = dt ? name ## _mul_f16_data : name ## _mul_f32_data; \
        ಧ್ರುವ std::string dts_ = dt ? "f16" : "f32"; \
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ob = 0; ob < 2; ++ob) \
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ nr = 0; nr < 2; ++nr) \
                ggml_vk_create_pipeline(device, device->pipeline_unary_mul[(idx)][dt][nr][ob], \
                    (#name "_mul" + std::string(ob ? "_b" : "") + "_" + dts_ + (nr ? "_norepeat" : "")).c_str(), \
                    len_, data_, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, \
                    { (uint32_t) nr, (uint32_t) ob }, 1); \
    }

    CREATE_UNARY_MUL(gelu, 0)
    CREATE_UNARY_MUL(sigmoid, 1)
    CREATE_UNARY_MUL(silu, 2)
    CREATE_UNARY_MUL(softplus, 3)
#undef CREATE_UNARY_MUL

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_data, "main", 1, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_data, "main", 1, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_data, "main", 1, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

#define CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_glu_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1, ಸತ್ಯ);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 3, ಪರಿಮಾಣ(vk_op_glu_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1, ಸತ್ಯ);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(swiglu_clamp)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_quick)
#undef CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_silu_back_f32, "silu_back_f32", silu_back_f32_len, silu_back_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_mask_inf_f32, "diag_mask_inf_f32", diag_mask_inf_f32_len, diag_mask_inf_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_diag_mask_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1, ಸತ್ಯ);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, ಪರಿಮಾಣ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, ಪರಿಮಾಣ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_back_f32, "soft_max_back_f32", soft_max_back_f32_len, soft_max_back_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_size }, 1, ಸತ್ಯ);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_data,     "main", 6, ಪರಿಮಾಣ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_data,     "main", 6, ಪರಿಮಾಣ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_data,     "main", 6, ಪರಿಮಾಣ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_data, "main", 6, ಪರಿಮಾಣ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_data, "main", 6, ಪರಿಮಾಣ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_data, "main", 6, ಪರಿಮಾಣ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ಸತ್ಯ);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_data, "main", 5, ಪರಿಮಾಣ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);

    ಪ್ರತಿ (uint32_t i = 0; i < ಅಂಕೆ_argsort_pipelines; ++i) {
        uint32_t BLOCK_SIZE = 1u << std::min(i, device->max_workgroup_size_log2);
        ಯದಿ (i <= device->max_workgroup_size_log2 &&
            2 * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * BLOCK_SIZE <= device->properties.limits.maxComputeSharedMemorySize) {
            ಧ್ರುವ uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_argsort_ತಳ್ಳು_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, NCOLS_PADDED_LOG2}, 1, ಸತ್ಯ);
        }
        ಧ್ರುವ uint32_t WG_UNROLL_FACTOR = BLOCK_SIZE > 1 ? 2 : 1;
        BLOCK_SIZE /= WG_UNROLL_FACTOR;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_argsort_ತಳ್ಳು_constants), {BLOCK_SIZE * WG_UNROLL_FACTOR, 1, 1}, {BLOCK_SIZE, WG_UNROLL_FACTOR}, 1, ಸತ್ಯ);
    }

    ಪ್ರತಿ (uint32_t i = 0; i < ಅಂಕೆ_topk_pipelines; ++i) {
        ಧ್ರುವ uint32_t BLOCK_SIZE = 1u << i;
        ಧ್ರುವ uint32_t NCOLS_PADDED_LOG2 = i;
        ಯದಿ (i <= device->max_workgroup_size_log2) {
            uint32_t nary_shmem = 2 * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * BLOCK_SIZE +
                                  ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * device->subgroup_size +
                                  2 * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) +
                                  2 * (BLOCK_SIZE / device->subgroup_size) * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ);
            ಯದಿ (device->subgroup_arithmetic && device->subgroup_require_full_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.maxComputeSharedMemorySize) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_ಹುಡುಕು_f32_len, topk_nary_ಹುಡುಕು_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_topk_ತಳ್ಳು_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, device->subgroup_size, device->subgroup_size_log2}, 1, ಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);
            } ಅನ್ಯಥಾ ಯದಿ (2 * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * BLOCK_SIZE <= device->properties.limits.maxComputeSharedMemorySize) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_topk_ತಳ್ಳು_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, NCOLS_PADDED_LOG2}, 1, ಸತ್ಯ);
            }
        }
    }

    // large-k fallback: one workgroup per row, radix-select instead of a full sort. The QSA
    // variant (spec constant 1) additionally gathers the qwen4 indexer input on the fly.
    {
        ಧ್ರುವ uint32_t BLOCK_SIZE = 1u << std::min(10u, device->max_workgroup_size_log2);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_f32, "topk_radix_f32", topk_radix_select_f32_len, topk_radix_select_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_topk_radix_ತಳ್ಳು_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, 0}, 1, ಸತ್ಯ);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_qsa, "topk_radix_qsa", topk_radix_select_f32_len, topk_radix_select_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_topk_radix_ತಳ್ಳು_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, 1}, 1, ಸತ್ಯ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sum_rows_f32, "sum_rows_f32", sum_rows_f32_len, sum_rows_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_sum_rows_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32, "cross_entropy_loss_f32", cross_entropy_loss_f32_len, cross_entropy_loss_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32_wg512, "cross_entropy_loss_f32_wg512", cross_entropy_loss_f32_len, cross_entropy_loss_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_back_f32, "cross_entropy_loss_back_f32", cross_entropy_loss_back_f32_len, cross_entropy_loss_back_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_back_f32_wg512, "cross_entropy_loss_back_f32_wg512", cross_entropy_loss_back_f32_len, cross_entropy_loss_back_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { 512 }, 1);
    // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will crash when using fwht kernels so we gate that here
    ಧ್ರುವ ತಾರ್ಕಿಕ can_use_fwht = device->driver_id != vk::DriverId::eIntelProprietaryWindows ||
        !ggml_vk_intel_windows_driver_in_range(device->properties.driverVersion, 101, 8509, 101, 8860);
    ಯದಿ (can_use_fwht && device->subgroup_basic && device->subgroup_shuffle) {
        ಪೂರ್ಣಾಙ್ಕ idx = 0;
        ಪ್ರತಿ (uint32_t n : {64, 128, 256, 512}) {
            ಯದಿ (device->subgroup_size <= n) {
                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_fwht_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_size, n }, 1, ಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);
            }
            ++idx;
        }
    } ಅನ್ಯಥಾ ಯದಿ (can_use_fwht) {
        ಪೂರ್ಣಾಙ್ಕ idx = 0;
        ಪ್ರತಿ (uint32_t n : {64, 128, 256, 512}) {
            ಧ್ರುವ uint32_t block_size = std::min(device->subgroup_size, n);
            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_fwht_ತಳ್ಳು_constants), {1, 1, 1}, { block_size, n }, 1);
            ++idx;
        }
    }

    ಧ್ರುವ uint32_t cumsum_elem_per_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32,       "cumsum_f32", cumsum_f32_len, cumsum_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_sum_rows_ತಳ್ಳು_constants), {1, 1, 1}, { 256, device->subgroup_size, cumsum_elem_per_thread }, 1, ಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_sum_rows_ತಳ್ಳು_constants), {1, 1, 1}, { 128, device->subgroup_size, 1 }, 1, ಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_sum_rows_ತಳ್ಳು_constants), {256, 1, 1}, { 256, device->subgroup_size }, 1, ಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_sum_rows_ತಳ್ಳು_constants), {256, 1, 1}, { 256, device->subgroup_size }, 1, ಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);

    ggml_vk_create_pipeline(device, device->pipeline_ಎಣಿಕೆ_equal_i32, "count_equal_i32", ಎಣಿಕೆ_equal_i32_len, ಎಣಿಕೆ_equal_i32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, { device->subgroup_size }, 1);

    ಯದಿ (device->subgroup_arithmetic && device->subgroup_vote && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ಎಣಿಕೆ_experts, "count_experts", ಎಣಿಕೆ_experts_subgroup_len, ಎಣಿಕೆ_experts_subgroup_data, "main", 2, ಪರಿಮಾಣ(vk_op_ಎಣಿಕೆ_experts_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1, ಸತ್ಯ, ಸತ್ಯ);
    } ಅನ್ಯಥಾ {
        ggml_vk_create_pipeline(device, device->pipeline_ಎಣಿಕೆ_experts, "count_experts", ಎಣಿಕೆ_experts_len, ಎಣಿಕೆ_experts_data, "main", 2, ಪರಿಮಾಣ(vk_op_ಎಣಿಕೆ_experts_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1, ಸತ್ಯ);
    }

    // comb holds a token's 4x4 matrix in one 16-lane slice of a subgroup, so it
    // needs at least 16 lanes, pinned to a known size.
    ಯದಿ (device->subgroup_basic && device->subgroup_shuffle && device->subgroup_require_full_support && device->subgroup_size >= 16) {
        ಧ್ರುವ uint32_t tokens_per_workgroup = 4 * (device->subgroup_size / 16);
        ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_comb_f32, "dsv4_hc_comb_f32", dsv4_hc_comb_f32_len, dsv4_hc_comb_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_dsv4_hc_comb_ತಳ್ಳು_constants), {tokens_per_workgroup, 1, 1}, { device->subgroup_size }, 1, ಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);
    }

    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_f32,        "dsv4_hc_pre_f32",        dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_data,  "main", 3, ಪರಿಮಾಣ(vk_op_dsv4_hc_pre_ತಳ್ಳು_constants),  {256, 1, 1}, { 256, 0 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_gated_f32,  "dsv4_hc_pre_gated_f32",  dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_data,  "main", 3, ಪರಿಮಾಣ(vk_op_dsv4_hc_pre_ತಳ್ಳು_constants),  {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_f32,       "dsv4_hc_post_f32",       dsv4_hc_post_f32_len, dsv4_hc_post_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_dsv4_hc_post_ತಳ್ಳು_constants), {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_nocomb_f32,"dsv4_hc_post_nocomb_f32",dsv4_hc_post_f32_len, dsv4_hc_post_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_dsv4_hc_post_ತಳ್ಳು_constants), {256, 1, 1}, { 256, 0 }, 1);

    ಪ್ರತಿ (auto &s : device->pipeline_solve_tri_f32) {
        ಧ್ರುವ vk_solve_tri_pipeline_state &state = s.first;

        // Max number of rows to load at a time, limited by shared memory
        ಧ್ರುವ uint32_t batch_N = device->properties.limits.maxComputeSharedMemorySize / ((state.N + state.K) * ಪರಿಮಾಣ(ಭಿನ್ನ));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        ಧ್ರುವ uint32_t block_size = std::max(128u, 1u << (uint32_t)ceilf(log2f(ಭಿನ್ನ(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_data, "main", 3,
            ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, block_size }, 1, ಸತ್ಯ);
    }

#define IM2COL(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## _data, "main", 2, ಪರಿಮಾಣ(vk_op_im2col_ತಳ್ಳು_constants), {512, 1, 1}, { device->subgroup_size }, 1, ಸತ್ಯ);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## _data, "main", 2, ಪರಿಮಾಣ(vk_op_im2col_3d_ತಳ್ಳು_constants), {512, 1, 1}, { 512 }, 1, ಸತ್ಯ);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## _data, "main", 2, ಪರಿಮಾಣ(vk_op_im2col_ತಳ್ಳು_constants), {512, 1, 1}, { device->subgroup_size }, 1, ಸತ್ಯ);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## _data, "main", 2, ಪರಿಮಾಣ(vk_op_im2col_3d_ತಳ್ಳು_constants), {512, 1, 1}, { 512 }, 1, ಸತ್ಯ);
    ಯದಿ (device->shader_int64 && device->buffer_device_address) {
        IM2COL(_bda)
    } ಅನ್ಯಥಾ {
        IM2COL()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_timestep_embedding_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_conv_transpose_1d_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f32,  "col2im_1d_f32",  col2im_1d_f32_len,  col2im_1d_f32_data,  "main", 2, ಪರಿಮಾಣ(vk_op_col2im_1d_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f16,  "col2im_1d_f16",  col2im_1d_f16_len,  col2im_1d_f16_data,  "main", 2, ಪರಿಮಾಣ(vk_op_col2im_1d_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1, ಸತ್ಯ);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_bf16, "col2im_1d_bf16", col2im_1d_bf16_len, col2im_1d_bf16_data, "main", 2, ಪರಿಮಾಣ(vk_op_col2im_1d_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1, ಸತ್ಯ);

    ggml_vk_create_pipeline(device, device->pipeline_out_prod_f32, "out_prod_f32", out_prod_f32_len, out_prod_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_snake_f32,  "snake_f32",  snake_f32_len,  snake_f32_data,  "main", 4, ಪರಿಮಾಣ(vk_op_snake_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_f16,  "snake_f16",  snake_f16_len,  snake_f16_data,  "main", 4, ಪರಿಮಾಣ(vk_op_snake_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_bf16, "snake_bf16", snake_bf16_len, snake_bf16_data, "main", 4, ಪರಿಮಾಣ(vk_op_snake_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool1d_f32, "pool1d_f32", pool1d_f32_len, pool1d_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_pool1d_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_pool2d_ತಳ್ಳು_constants), {1, 1, 1}, {512}, 1);
    ಯದಿ (device->subgroup_arithmetic && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_pool2d_subgroup_f32, "pool2d_subgroup_f32", pool2d_subgroup_f32_len, pool2d_subgroup_f32_data, "main", 2, ಪರಿಮಾಣ(vk_op_pool2d_ತಳ್ಳು_constants), {1, 1, 1}, {128}, 1, ಅಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_data, "main", 7, ಪರಿಮಾಣ(vk_op_rwkv_wkv6_ತಳ್ಳು_constants), {1, 1, 1}, {device->subgroup_size}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_data, "main", 8, ಪರಿಮಾಣ(vk_op_rwkv_wkv7_ತಳ್ಳು_constants), {1, 1, 1}, {device->subgroup_size}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_gated_linear_attn_f32, "gated_linear_attn_f32", gated_linear_attn_f32_len, gated_linear_attn_f32_data, "main", 6, ಪರಿಮಾಣ(vk_op_gated_linear_attn_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ಪ್ರತಿ (ggml_type k_type : lightning_indexer_k_types) {
        ಧ್ರುವ std::string name = "lightning_indexer_" + std::string(ggml_type_name(k_type)) + "_k_f32";
        ggml_vk_create_pipeline(device, device->pipeline_lightning_indexer_f32[k_type], name.c_str(), lightning_indexer_f32_len, lightning_indexer_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_lightning_indexer_ತಳ್ಳು_constants), {1, 1, 1}, {(uint32_t)k_type, fa_block_bytes(k_type)}, 1, ಸತ್ಯ);
    }

    {
        ಧ್ರುವ uint32_t gdn_sizes[] = {16, 32, 64, 128};
        ಧ್ರುವ ವರ್ಣ * gdn_names[][2] = {
            {"gated_delta_net_f32_d16",     "gated_delta_net_f32_d16_kda"},
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        ಪ್ರತಿ (uint32_t si = 0; si < 4; si++) {
            ಧ್ರುವ uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            // Intel Xe regresses at SIMD32 for this scan; prefer a narrower subgroup.
            uint32_t gdn_subgroup_size = device->subgroup_size;
            ಯದಿ (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control &&
                device->subgroup_min_size <= 16u && device->subgroup_max_size >= 16u) {
                gdn_subgroup_size = 16u;
            }
            uint32_t lanes_per_column;
            ಯದಿ (device->vendor_id == VK_VENDOR_ID_INTEL) {
                // Intel Xe: full-width reduction (min rows/lane) ~10x over the COLS_PER_WG=8 rule.
                lanes_per_column = std::min(gdn_subgroup_size, S_V);
            } ಅನ್ಯಥಾ ಯದಿ (S_V >= 128u && device->subgroup_clustered) {
                // COLS_PER_WG=8: measured optimum (Ampere sg32->4, Vega20 sg64->8).
                lanes_per_column = std::max(1u, gdn_subgroup_size / 8u);
            } ಅನ್ಯಥಾ {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_per_column = std::min(S_V, gdn_subgroup_size);
            }

            // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
            // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
            ಯಾವತ್ (lanes_per_column > 1u) {
                ಧ್ರುವ ತಾರ್ಕಿಕ valid_lanes = (gdn_subgroup_size % lanes_per_column) == 0 &&
                                         (S_V % lanes_per_column) == 0;
                ಧ್ರುವ uint32_t cols_per_wg = valid_lanes ? gdn_subgroup_size / lanes_per_column : 0;
                ಯದಿ (valid_lanes && cols_per_wg > 0 && (S_V % cols_per_wg) == 0) {
                    ಭಙ್ಗ;
                }
                lanes_per_column >>= 1u;
            }

            GGML_ASSERT((gdn_subgroup_size % lanes_per_column) == 0);
            GGML_ASSERT((S_V % lanes_per_column) == 0);
            GGML_ASSERT((S_V % (gdn_subgroup_size / lanes_per_column)) == 0);

            ಧ್ರುವ ತಾರ್ಕಿಕ need_partial_subgroup_reduce = lanes_per_column != 1u && lanes_per_column < gdn_subgroup_size;
            ಧ್ರುವ ತಾರ್ಕಿಕ use_clustered_reduce = device->subgroup_arithmetic && device->subgroup_clustered && need_partial_subgroup_reduce;
            ಧ್ರುವ ತಾರ್ಕಿಕ use_subgroup_reduce = device->subgroup_arithmetic && !need_partial_subgroup_reduce;
            ಧ್ರುವ ತಾರ್ಕಿಕ use_subgroup_ops = use_clustered_reduce || use_subgroup_reduce;
            size_t gdn_len;
            ಧ್ರುವ ಶೂನ್ಯ * gdn_data;
            ಯದಿ (use_clustered_reduce) {
                gdn_len = gated_delta_net_f32_len;
                gdn_data = (ಧ್ರುವ ಶೂನ್ಯ *)gated_delta_net_f32_data;
            } ಅನ್ಯಥಾ ಯದಿ (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_data = (ಧ್ರುವ ಶೂನ್ಯ *)gated_delta_net_f32_nocluster_data;
            } ಅನ್ಯಥಾ {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_data = (ಧ್ರುವ ಶೂನ್ಯ *)gated_delta_net_f32_shmem_data;
            }

            ಧ್ರುವ uint32_t cols_per_wg = gdn_subgroup_size / lanes_per_column;
            ಧ್ರುವ std::array<uint32_t, 3> wg_denoms = {1u, 1u, cols_per_wg};

            ಪ್ರತಿ (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_names[si][kda], gdn_len, gdn_data, "main", 7, ಪರಿಮಾಣ(vk_op_gated_delta_net_ತಳ್ಳು_constants),
                    wg_denoms, {S_V, kda, gdn_subgroup_size, lanes_per_column}, 1, ಸತ್ಯ, use_subgroup_ops, gdn_subgroup_size);
            }
        }
    }

    ಯದಿ (device->subgroup_arithmetic && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", ssm_scan_subgroup_f32_len, ssm_scan_subgroup_f32_data, "main", 8, ಪರಿಮಾಣ(vk_op_ssm_scan_ತಳ್ಳು_constants), {1, 1, 1}, {128, device->subgroup_size}, 1, ಸತ್ಯ, ಸತ್ಯ);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", ssm_scan_subgroup_f32_len, ssm_scan_subgroup_f32_data, "main", 8, ಪರಿಮಾಣ(vk_op_ssm_scan_ತಳ್ಳು_constants), {1, 1, 1}, {256, device->subgroup_size}, 1, ಸತ್ಯ, ಸತ್ಯ);
    } ಅನ್ಯಥಾ {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", ssm_scan_f32_len, ssm_scan_f32_data, "main", 8, ಪರಿಮಾಣ(vk_op_ssm_scan_ತಳ್ಳು_constants), {1, 1, 1}, {128, device->subgroup_size, 16}, 1, ಸತ್ಯ, ಸತ್ಯ);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", ssm_scan_f32_len, ssm_scan_f32_data, "main", 8, ಪರಿಮಾಣ(vk_op_ssm_scan_ತಳ್ಳು_constants), {1, 1, 1}, {256, device->subgroup_size, 16}, 1, ಸತ್ಯ, ಸತ್ಯ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32,           "ssm_conv_f32",           ssm_conv_f32_len, ssm_conv_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_ssm_conv_ತಳ್ಳು_constants), {32, 16, 1}, {32, 16, 0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_silu_f32,      "ssm_conv_silu_f32",      ssm_conv_f32_len, ssm_conv_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_ssm_conv_ತಳ್ಳು_constants), {32, 16, 1}, {32, 16, 0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", ssm_conv_f32_len, ssm_conv_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_ssm_conv_ತಳ್ಳು_constants), {32, 16, 1}, {32, 16, 1, 1}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_step_adamw_f32, "opt_step_adamw_f32", opt_step_adamw_f32_len, opt_step_adamw_f32_data, "main", 5, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_step_sgd_f32, "opt_step_sgd_f32", opt_step_sgd_f32_len, opt_step_sgd_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d, conv3d
    ಪ್ರತಿ (uint32_t s = 0; s < CONV_SHAPE_ಎಣಿಕೆ; ++s) {
        // smaller WG for the small-tile fallback gives more concurrent WGs per SM
        uint32_t conv2d_WG_SIZE  = (s == CONV_SHAPE_64x32) ? 128 : 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_block_size conv2d_BS = vk_conv_block_sizes[s];
        ತಾರ್ಕಿಕ conv2d_UNROLL = ಸತ್ಯ;

#ಯದಿ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ಯದಿ (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#endif

        ಯದಿ (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = ಅಸತ್ಯ;
        } ಅನ್ಯಥಾ ಯದಿ (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            ಯದಿ (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = ಅಸತ್ಯ;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        ತಾರ್ಕಿಕ allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        ತಾರ್ಕಿಕ allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        ಯದಿ (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_size,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        // cm1 is used only when cm2 is unavailable; capped at 64x128 (due to shared memory size).
        // Requires 16x16x16 f16-acc since that's the fragment shape hard-coded in the shader.
        // Subgroup size must be 32 or 64 (to keep WG_SIZE sane) and we need
        // subgroup_size_control to force the driver to actually use it.
        ತಾರ್ಕಿಕ conv2d_use_cm1 = ಅಸತ್ಯ;
#ಯದಿ defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        conv2d_use_cm1 = !device->coopmat2 &&
                         device->coopmat_support && device->coopmat_support_16x16x16_f16acc &&
                         device->subgroup_size_control &&
                         (device->subgroup_size == 32 || device->subgroup_size == 64) &&
                         s != CONV_SHAPE_128x128;
#endif

        ಧ್ರುವ uint32_t conv2d_cm1_shmem_pad = 8;

        auto shmem_req = [&](uint32_t pad, ತಾರ್ಕಿಕ csh_store, ತಾರ್ಕಿಕ fp16_shmem) {
            ಧ್ರುವ uint32_t elem_size = fp16_shmem ? (uint32_t)ಪರಿಮಾಣ(uint16_t) : (uint32_t)ಪರಿಮಾಣ(ಭಿನ್ನ);
            ಧ್ರುವ uint32_t csh_elems = csh_store ? conv2d_BS.K * conv2d_BS.NPQ : 0u;
            ಪ್ರತ್ಯಾಗಮ (conv2d_BS.K * (conv2d_BS.CRS + pad) + conv2d_BS.CRS * (conv2d_BS.NPQ + pad) + csh_elems) * elem_size;
        };

        // 2D, transpose-2D, and 3D conv use the same KxCRS @ CRSxNPQ shmem
        // layout. cm1 needs Csh for output, so check before applying cm1 params.
        ಯದಿ (conv2d_use_cm1 && device->properties.limits.maxComputeSharedMemorySize < shmem_req(conv2d_cm1_shmem_pad, ಸತ್ಯ, ಸತ್ಯ)) {
            conv2d_use_cm1 = ಅಸತ್ಯ;
        }

        uint32_t conv2d_WM = 16, conv2d_WN = 16;  // cm1 subgroup tile, ignored otherwise
        ಯದಿ (conv2d_use_cm1) {
            conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
            // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
            // (i.e. 8 subgroups for sg=32, 4 subgroups for sg=64).
            ಧ್ರುವ ತಾರ್ಕಿಕ sg64 = (device->subgroup_size == 64);
            ವಿಕಲ್ಪ (s) {
                ಪಕ್ಷ CONV_SHAPE_64x32:   conv2d_WM = sg64 ? 32 : 16; conv2d_WN = 16; ಭಙ್ಗ;
                ಪಕ್ಷ CONV_SHAPE_64x128:  conv2d_WM = 32; conv2d_WN = sg64 ? 64 : 32; ಭಙ್ಗ;
                ಪಕ್ಷ CONV_SHAPE_32x256:  conv2d_WM = sg64 ? 16 : 32; conv2d_WN = sg64 ? 128 : 32; ಭಙ್ಗ;
                ಸ್ವಭಾವ: ಭಙ್ಗ;
            }
            ಧ್ರುವ uint32_t warps_M = conv2d_BS.K / conv2d_WM;
            ಧ್ರುವ uint32_t warps_N = conv2d_BS.NPQ / conv2d_WN;
            conv2d_WG_SIZE         = warps_M * warps_N * device->subgroup_size;
        }

        // stage cm2 accumulator through shmem for coalesced global stores;
        // skipped on 128x128 where the extra Csh footprint hurts occupancy.
        // cm1 always uses the staged path.
        uint32_t conv2d_csh_store = (device->coopmat2 && s != CONV_SHAPE_128x128) ? 1u : 0u;
        ಯದಿ (conv2d_use_cm1) {
            conv2d_csh_store = 1;
        }

        // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
        ಧ್ರುವ ತಾರ್ಕಿಕ conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1;

        // shrink CRS if the non-cm1 config still doesn't fit
        ಯದಿ (device->properties.limits.maxComputeSharedMemorySize < shmem_req(conv2d_SHMEM_PAD, conv2d_csh_store, conv2d_use_fp16_shmem)) {
            GGML_ASSERT(!conv2d_use_cm1);
            conv2d_BS.CRS = 8;
            ಯದಿ (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_size, conv2d_BS.CRS);
            }
            conv2d_csh_store = 0;
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_SIZE, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

        // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
        ಧ್ರುವ uint32_t conv2d_required_subgroup_size = conv2d_use_cm1 ? device->subgroup_size : 0;

#define CREATE_CONV(name, type_suffix, spv_suffix) \
        ಪ್ರತಿ (auto &c : device->pipeline_##name##type_suffix[s]) { \
            ಧ್ರುವ vk_conv2d_pipeline_state &state = c.first;  \
            std::vector<uint32_t> spec_constants_cpy = spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            spec_constants_cpy.push_back(state.aligned); \
            spec_constants_cpy.push_back(conv2d_csh_store); \
            spec_constants_cpy.push_back(conv2d_WM); \
            spec_constants_cpy.push_back(conv2d_WN); \
            ggml_vk_create_pipeline( \
                device, c.second, #name #type_suffix, \
                name##type_suffix##spv_suffix##_len, name##type_suffix##spv_suffix##_data, "main", 3, \
                ಪರಿಮಾಣ(vk_op_conv2d_ತಳ್ಳು_constants), wg_denoms, spec_constants_cpy, 1, ಸತ್ಯ, use_collectives || conv2d_required_subgroup_size, conv2d_required_subgroup_size);    \
        }
#define CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#ಯದಿ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ಯದಿ (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } ಅನ್ಯಥಾ
#endif
#ಯದಿ defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        ಯದಿ (conv2d_use_cm1) {
            CREATE_CONVS(_cm1)
        } ಅನ್ಯಥಾ
#endif
        ಯದಿ (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } ಅನ್ಯಥಾ {
            CREATE_CONVS( )
        }
#undef CREATE_CONV
#undef CREATE_CONVS

        std::vector<uint32_t> conv3d_spec_constants = { conv2d_WG_SIZE, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, conv2d_SHMEM_PAD };
#define CREATE_CONV3D(type_suffix, spv_suffix) \
        ಪ್ರತಿ (auto &c : device->pipeline_conv3d##type_suffix[s]) { \
            ಧ್ರುವ vk_conv3d_pipeline_state &state = c.first; \
            std::vector<uint32_t> spec_constants_cpy = conv3d_spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.s2); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.p2); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.d2); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            spec_constants_cpy.push_back(state.KD); \
            spec_constants_cpy.push_back(state.aligned); \
            spec_constants_cpy.push_back(conv2d_csh_store); \
            spec_constants_cpy.push_back(conv2d_WM); \
            spec_constants_cpy.push_back(conv2d_WN); \
            ggml_vk_create_pipeline( \
                device, c.second, "conv3d" #type_suffix, \
                conv3d##type_suffix##spv_suffix##_len, conv3d##type_suffix##spv_suffix##_data, "main", 3, \
                ಪರಿಮಾಣ(vk_op_conv3d_ತಳ್ಳು_constants), wg_denoms, spec_constants_cpy, 1, ಸತ್ಯ, conv2d_required_subgroup_size != 0, conv2d_required_subgroup_size); \
        }
#ಯದಿ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ಯದಿ (device->coopmat2) {
            CREATE_CONV3D(_f32, _cm2)
            CREATE_CONV3D(_f16_f32, _cm2)
        } ಅನ್ಯಥಾ
#endif
#ಯದಿ defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        ಯದಿ (conv2d_use_cm1) {
            CREATE_CONV3D(_f32, _cm1)
            CREATE_CONV3D(_f16_f32, _cm1)
        } ಅನ್ಯಥಾ
#endif
        ಯದಿ (conv2d_UNROLL) {
            CREATE_CONV3D(_f32, _unroll)
            CREATE_CONV3D(_f16_f32, _unroll)
        } ಅನ್ಯಥಾ {
            CREATE_CONV3D(_f32, )
            CREATE_CONV3D(_f16_f32, )
        }
#undef CREATE_CONV3D
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_conv2d_dw_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_conv2d_dw_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_conv2d_dw_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_data, "main", 3, ಪರಿಮಾಣ(vk_op_conv2d_dw_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ಪ್ರತಿ (uint32_t use_ತಳ್ಳು = 0; use_ತಳ್ಳು < 2; ++use_ತಳ್ಳು) {
        ಪ್ರತಿ (uint32_t i = 0; i < ಅಂಕೆ_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_ತಳ್ಳು], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_data, "main", 4, ಪರಿಮಾಣ(vk_op_topk_moe_ತಳ್ಳು_constants), {1, 1, 1}, {device->subgroup_size, 1u<<i, use_ತಳ್ಳು}, 1, ಸತ್ಯ, ಸತ್ಯ, device->subgroup_size);
        }
    }

    // Drop compile_mutex so other threads can walk while we compile.
    compile_lock.unlock();

    // Compile what we claimed; create_pipeline_func reacquires compile_mutex
    // at the end to flip compile_pending/compiled and notify waiters.
    ಯದಿ (has_claimed_task) {
        auto & task = claimed_task;
        ggml_vk_create_pipeline_func(device, task.pipeline, task.spv_size, task.spv_data,
                                     task.entrypoint, task.parameter_ಎಣಿಕೆ, task.wg_denoms,
                                     task.specialization_constants, task.disable_robustness,
                                     task.require_full_subgroups, task.required_subgroup_size);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    ಯದಿ (wait_pipeline) {
        std::unique_lock<std::mutex> wait_lock(device->compile_mutex);
        device->compile_cv.wait(wait_lock, [&] {
            ಪ್ರತ್ಯಾಗಮ wait_pipeline->compiled.load();
        });
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_external_memory_host_usable(vk::DriverId driver_id) {
    // Disable external_memory_host until https://github.com/KhronosGroup/MoltenVK/pull/2622
    // is available in the Vulkan SDK.
    ಪ್ರತ್ಯಾಗಮ driver_id != vk::DriverId::eMoltenvk;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_host_buffer_import_usable(uint32_t vendor_id) {
    // amdgpu re-faults all pages of every imported host allocation on each submission
    ಪ್ರತ್ಯಾಗಮ vendor_id != VK_VENDOR_ID_AMD;
}

vk_device ggml_vk_get_device(size_t idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ಯದಿ (vk_instance.devices[idx] == nullptr) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();

        device->memory_logger = std::unique_ptr<vk_memory_logger>(ನೂತನ vk_memory_logger());

        size_t dev_ಅಂಕೆ = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        ಯದಿ (dev_ಅಂಕೆ >= physical_devices.size()) {
            std::cerr << "ggml_vulkan: Device with index " << dev_ಅಂಕೆ << " does not exist." << std::endl;
            ಕ್ಷಿಪ std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_ಅಂಕೆ];
        ಧ್ರುವ std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        device->architecture = get_device_architecture(device->physical_device);

        ಧ್ರುವ ವರ್ಣ* GGML_VK_PREFER_HOST_MEMORY = getenv("GGML_VK_PREFER_HOST_MEMORY");
        device->prefer_host_memory = GGML_VK_PREFER_HOST_MEMORY != nullptr;

        ಧ್ರುವ ವರ್ಣ* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
        device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != nullptr;

        ಧ್ರುವ ವರ್ಣ* GGML_VK_ALLOW_SYSMEM_FALLBACK = getenv("GGML_VK_ALLOW_SYSMEM_FALLBACK");
        device->allow_sysmem_fallback = GGML_VK_ALLOW_SYSMEM_FALLBACK != nullptr;

        ಧ್ರುವ ವರ್ಣ* GGML_VK_DISABLE_GRAPH_OPTIMIZE = getenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE");
        device->disable_graph_optimize = GGML_VK_DISABLE_GRAPH_OPTIMIZE != nullptr;

        ತಾರ್ಕಿಕ fp16_storage = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ fp16_compute = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ maintenance4_support = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ sm_builtins = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ amd_shader_core_properties2 = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ pipeline_robustness = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ coopmat2_support = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ coopmat2_decode_vector_support = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ pipeline_executable_properties_support = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ internally_sync_support = ಅಸತ್ಯ;
        device->coopmat_support = ಅಸತ್ಯ;
        device->integer_dot_product = ಅಸತ್ಯ;
        device->shader_64b_indexing = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ bfloat16_support = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ dot2_f16_support = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ ocp_microscaling_extension = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ shader_float8_extension = ಅಸತ್ಯ;

        ಪ್ರತಿ (ಧ್ರುವ auto& properties : ext_props) {
            ಯದಿ (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
                amd_shader_core_properties2 = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_EXT_pipeline_robustness", properties.extensionName) == 0) {
                pipeline_robustness = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                device->subgroup_size_control = ಸತ್ಯ;
#ಯದಿ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = ಸತ್ಯ;
                device->coopmat_m = 0;
                device->coopmat_n = 0;
                device->coopmat_k = 0;
#endif
#ಯದಿ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2")) {
                coopmat2_support = ಸತ್ಯ;
#endif
            } ಅನ್ಯಥಾ ಯದಿ (strcmp(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME, properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
                coopmat2_decode_vector_support = ಸತ್ಯ;
#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
                device->integer_dot_product = ಸತ್ಯ;
#endif
#ಯದಿ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_BFLOAT16")) {
                bfloat16_support = ಸತ್ಯ;
#endif
#ಯದಿ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
            } ಅನ್ಯಥಾ ಯದಿ (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
                ocp_microscaling_extension = ಸತ್ಯ;
#endif
#ಯದಿ defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            } ಅನ್ಯಥಾ ಯದಿ (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
                shader_float8_extension = ಸತ್ಯ;
#endif
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_DOT2")) {
                dot2_f16_support = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_pipeline_executable_properties", properties.extensionName) == 0) {
                pipeline_executable_properties_support = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_EXT_memory_priority", properties.extensionName) == 0 &&
                       getenv("GGML_VK_ENABLE_MEMORY_PRIORITY")) {
                device->memory_priority = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_EXT_external_memory_host", properties.extensionName) == 0) {
                device->external_memory_host = ಸತ್ಯ;
#ಯದಿ defined(VK_EXT_shader_64bit_indexing)
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_EXT_shader_64bit_indexing", properties.extensionName) == 0) {
                device->shader_64b_indexing = ಸತ್ಯ;
#endif
            } ಅನ್ಯಥಾ ಯದಿ (strcmp(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, properties.extensionName) == 0) {
                internally_sync_support = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_EXT_device_fault", properties.extensionName) == 0) {
                device->device_fault = ಸತ್ಯ;
            }
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceMaintenance3Properties props3;
        vk::PhysicalDeviceMaintenance4Properties props4;
        vk::PhysicalDeviceSubgroupProperties subgroup_props;
        vk::PhysicalDeviceDriverProperties driver_props;
        vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;
        vk::PhysicalDeviceShaderCoreProperties2AMD amd_shader_core_properties2_props;
        vk::PhysicalDeviceVulkan11Properties vk11_props;
        vk::PhysicalDeviceVulkan12Properties vk12_props;
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR shader_integer_dot_product_props;
        vk::PhysicalDeviceExternalMemoryHostPropertiesEXT external_memory_host_props;

        props2.pNext = &props3;
        props3.pNext = &subgroup_props;
        subgroup_props.pNext = &driver_props;
        driver_props.pNext = &vk11_props;
        vk11_props.pNext = &vk12_props;

        VkBaseOutStructure * last_struct = (VkBaseOutStructure *)&vk12_props;

        ಯದಿ (maintenance4_support) {
            last_struct->pNext = (VkBaseOutStructure *)&props4;
            last_struct = (VkBaseOutStructure *)&props4;
        }
        ಯದಿ (sm_builtins) {
            last_struct->pNext = (VkBaseOutStructure *)&sm_props;
            last_struct = (VkBaseOutStructure *)&sm_props;
        }
        ಯದಿ (amd_shader_core_properties2) {
            last_struct->pNext = (VkBaseOutStructure *)&amd_shader_core_properties2_props;
            last_struct = (VkBaseOutStructure *)&amd_shader_core_properties2_props;
        }
        ಯದಿ (device->subgroup_size_control) {
            last_struct->pNext = (VkBaseOutStructure *)&subgroup_size_control_props;
            last_struct = (VkBaseOutStructure *)&subgroup_size_control_props;
        }

#ಯದಿ defined(VK_NV_cooperative_matrix2)
        vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
        ಯದಿ (coopmat2_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat2_props;
            last_struct = (VkBaseOutStructure *)&coopmat2_props;
        }
#endif

        ಯದಿ (device->integer_dot_product) {
            last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_props;
            last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_props;
        }

        ಯದಿ (device->external_memory_host) {
            last_struct->pNext = (VkBaseOutStructure *)&external_memory_host_props;
            last_struct = (VkBaseOutStructure *)&external_memory_host_props;
        }

        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;
        device->vendor_id = device->properties.vendorID;
        device->driver_id = driver_props.driverID;

        device->external_memory_host = device->external_memory_host && ggml_vk_external_memory_host_usable(device->driver_id);

        // Implementing the async backend interfaces seems broken on older Intel HW,
        // see https://github.com/ggml-org/llama.cpp/issues/17302.
        device->support_async = (device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 std::string(device->properties.deviceName.data()).find("(DG1)") == std::string::npos) &&
                                getenv("GGML_VK_DISABLE_ASYNC") == nullptr;

        ಯದಿ (!device->support_async) {
            GGML_LOG_DEBUG("ggml_vulkan: WARNING: Async execution disabled on certain Intel devices.\n");
        }

        ಧ್ರುವ ವರ್ಣ* GGML_VK_FORCE_MAX_ALLOCATION_SIZE = getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        ಯದಿ (GGML_VK_FORCE_MAX_ALLOCATION_SIZE != nullptr) {
            device->max_memory_allocation_size = std::stoull(GGML_VK_FORCE_MAX_ALLOCATION_SIZE);
        } ಅನ್ಯಥಾ ಯದಿ (maintenance4_support) {
            device->max_memory_allocation_size = std::min(props3.maxMemoryAllocationSize, props4.maxBufferSize);
        } ಅನ್ಯಥಾ {
            device->max_memory_allocation_size = props3.maxMemoryAllocationSize;
        }

        ಧ್ರುವ ವರ್ಣ* GGML_VK_FORCE_MAX_BUFFER_SIZE = getenv("GGML_VK_FORCE_MAX_BUFFER_SIZE");

        ಯದಿ (GGML_VK_FORCE_MAX_BUFFER_SIZE != nullptr) {
            device->max_buffer_size = std::stoull(GGML_VK_FORCE_MAX_BUFFER_SIZE);
        } ಅನ್ಯಥಾ ಯದಿ (maintenance4_support) {
            device->max_buffer_size = props4.maxBufferSize;
        } ಅನ್ಯಥಾ {
            device->max_buffer_size = device->max_memory_allocation_size;
        }

        ಧ್ರುವ ವರ್ಣ* GGML_VK_SUBALLOCATION_BLOCK_SIZE = getenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE");

        ಯದಿ (GGML_VK_SUBALLOCATION_BLOCK_SIZE != nullptr) {
            device->suballocation_block_size = std::stoull(GGML_VK_SUBALLOCATION_BLOCK_SIZE);
        } ಅನ್ಯಥಾ {
            // Limit batching of allocations to 1GB by default to avoid fragmentation issues
            device->suballocation_block_size = 1024*1024*1024;
        }
        device->suballocation_block_size = std::min(device->suballocation_block_size, device->max_memory_allocation_size);

        device->subgroup_size = subgroup_props.subgroupSize;
        device->subgroup_size_log2 = uint32_t(log2f(ಭಿನ್ನ(device->subgroup_size)));
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        ಯದಿ (sm_builtins) {
            device->shader_core_ಎಣಿಕೆ = sm_props.shaderSMCount;
        } ಅನ್ಯಥಾ ಯದಿ (amd_shader_core_properties2) {
            device->shader_core_ಎಣಿಕೆ = amd_shader_core_properties2_props.active_Compute_Unit_ಎಣಿಕೆ;
        } ಅನ್ಯಥಾ ಯದಿ (device->vendor_id == VK_VENDOR_ID_INTEL) {
            device->shader_core_ಎಣಿಕೆ = ggml_vk_intel_shader_core_ಎಣಿಕೆ(device->physical_device);
        } ಅನ್ಯಥಾ {
            device->shader_core_ಎಣಿಕೆ = 0;
        }
        device->float_controls_rte_fp16 = vk12_props.shaderRoundingModeRTEFloat16;
        device->float_controls_denorm_preserve_fp16 = vk12_props.shaderDenormPreserveFloat16;

        device->subgroup_basic = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                 (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eBasic);
        device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                      (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eArithmetic);
#ifdef __APPLE__
        // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
        ಯದಿ (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_arithmetic = ಅಸತ್ಯ;
        }
#endif
        device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                   (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eShuffle);
#ifdef __APPLE__
        ಯದಿ (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_shuffle = ಅಸತ್ಯ;
        }
#endif
        device->subgroup_clustered = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                     (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eClustered);

        device->subgroup_ballot = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                  (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eBallot);

        device->subgroup_vote = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eVote);

        // Submit at least every 100 nodes, in case there are workloads without as much matmul.
        device->max_nodes_per_submit = 100;
        ಧ್ರುವ ವರ್ಣ* GGML_VK_MAX_NODES_PER_SUBMIT = getenv("GGML_VK_MAX_NODES_PER_SUBMIT");
        ಯದಿ (GGML_VK_MAX_NODES_PER_SUBMIT != nullptr) {
            uint32_t max_nodes_per_submit = std::stoul(GGML_VK_MAX_NODES_PER_SUBMIT);
            device->max_nodes_per_submit = std::max(max_nodes_per_submit, 1u);
        }

        ಧ್ರುವ ತಾರ್ಕಿಕ force_disable_f16 = getenv("GGML_VK_DISABLE_F16") != nullptr;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        ಯದಿ (!ggml_vk_khr_cooperative_matrix_support(device->properties, driver_props, device->architecture)) {
            device->coopmat_support = ಅಸತ್ಯ;
        }

        device->integer_dot_product = device->integer_dot_product && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated;

        device->min_imported_host_pointer_alignment = external_memory_host_props.minImportedHostPointerAlignment;

        device->max_workgroup_size_log2 = uint32_t(log2f(ಭಿನ್ನ(device->properties.limits.maxComputeWorkGroupInvocations)));

        std::vector<vk::QueueFamilyProperties> queue_family_props = device->physical_device.getQueueFamilyProperties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        // Allow overriding avoiding the graphics queue because it can increase performance on RADV
        ಧ್ರುವ ತಾರ್ಕಿಕ allow_graphics_queue = (getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE") != nullptr);
        ಧ್ರುವ vk::QueueFlagBits graphics_flag = allow_graphics_queue ? (vk::QueueFlagBits)0 : vk::QueueFlagBits::eGraphics;
        ಧ್ರುವ uint32_t compute_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eCompute, graphics_flag, -1, 1);
        ಧ್ರುವ uint32_t transfer_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eCompute | graphics_flag, compute_queue_family_index, 1);

        ಧ್ರುವ ಭಿನ್ನ priorities[] = { 1.0f, 1.0f };
        device->single_queue = compute_queue_family_index == transfer_queue_family_index && queue_family_props[compute_queue_family_index].queue_ಎಣಿಕೆ == 1;

        std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
        vk::DeviceCreateInfo device_create_info{};
        std::vector<ಧ್ರುವ ವರ್ಣ *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.pNext = nullptr;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.pNext = nullptr;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.pNext = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.pNext = nullptr;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.pNext = &vk12_features;

        last_struct = (VkBaseOutStructure *)&vk12_features;

        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR internally_synchronized_queues_features{};
        internally_synchronized_queues_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
        internally_synchronized_queues_features.pNext = nullptr;
        internally_synchronized_queues_features.internallySynchronizedQueues = VK_FALSE;

        ಯದಿ (internally_sync_support) {
            last_struct->pNext = (VkBaseOutStructure *)&internally_synchronized_queues_features;
            last_struct = (VkBaseOutStructure *)&internally_synchronized_queues_features;
            device_extensions.push_back(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
        }

        VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
        pl_robustness_features.pNext = nullptr;
        pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
        pl_robustness_features.pipelineRobustness = VK_FALSE;

        ಯದಿ (pipeline_robustness) {
            last_struct->pNext = (VkBaseOutStructure *)&pl_robustness_features;
            last_struct = (VkBaseOutStructure *)&pl_robustness_features;
            device_extensions.push_back("VK_EXT_pipeline_robustness");
        }

        VkPhysicalDeviceMemoryPriorityFeaturesEXT memory_priority_features;
        memory_priority_features.pNext = nullptr;
        memory_priority_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
        memory_priority_features.memoryPriority = VK_FALSE;
        ಯದಿ (device->memory_priority) {
            last_struct->pNext = (VkBaseOutStructure *)&memory_priority_features;
            last_struct = (VkBaseOutStructure *)&memory_priority_features;
            device_extensions.push_back("VK_EXT_memory_priority");
        }

        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_size_control_features;
        subgroup_size_control_features.pNext = nullptr;
        subgroup_size_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        subgroup_size_control_features.computeFullSubgroups = ಅಸತ್ಯ;
        subgroup_size_control_features.subgroupSizeControl = ಅಸತ್ಯ;

        ಯದಿ (device->subgroup_size_control) {
            last_struct->pNext = (VkBaseOutStructure *)&subgroup_size_control_features;
            last_struct = (VkBaseOutStructure *)&subgroup_size_control_features;
        }

#ಯದಿ defined(VK_KHR_cooperative_matrix)
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
        coopmat_features.pNext = nullptr;
        coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coopmat_features.cooperativeMatrix = VK_FALSE;

        ಯದಿ (device->coopmat_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat_features;
            last_struct = (VkBaseOutStructure *)&coopmat_features;
        }
#endif

#ಯದಿ defined(VK_NV_cooperative_matrix2)
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
        coopmat2_features.pNext = nullptr;
        coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
        ಯದಿ (coopmat2_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat2_features;
            last_struct = (VkBaseOutStructure *)&coopmat2_features;
            device_extensions.push_back("VK_NV_cooperative_matrix2");
        }
#endif

        VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV coopmat2_decode_vector_features {};
        coopmat2_decode_vector_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV;
        ಯದಿ (coopmat2_decode_vector_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat2_decode_vector_features;
            last_struct = (VkBaseOutStructure *)&coopmat2_decode_vector_features;
            device_extensions.push_back(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME);
        }

#ಯದಿ defined(VK_KHR_shader_bfloat16)
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
        bfloat16_features.pNext = nullptr;
        bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        ಯದಿ (bfloat16_support) {
            last_struct->pNext = (VkBaseOutStructure *)&bfloat16_features;
            last_struct = (VkBaseOutStructure *)&bfloat16_features;
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#endif

        VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
        ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
        ಯದಿ (ocp_microscaling_extension) {
            last_struct->pNext = (VkBaseOutStructure *)&ocp_microscaling_features;
            last_struct = (VkBaseOutStructure *)&ocp_microscaling_features;
            device_extensions.push_back(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
        shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
        ಯದಿ (shader_float8_extension) {
            last_struct->pNext = (VkBaseOutStructure *)&shader_float8_features;
            last_struct = (VkBaseOutStructure *)&shader_float8_features;
            device_extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        }

        VkPhysicalDeviceMaintenance4Features maint4_features {};
        maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        ಯದಿ (maintenance4_support) {
            last_struct->pNext = (VkBaseOutStructure *)&maint4_features;
            last_struct = (VkBaseOutStructure *)&maint4_features;
            device_extensions.push_back("VK_KHR_maintenance4");
        }

        VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features {};
        shader_integer_dot_product_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
        ಯದಿ (device->integer_dot_product) {
            last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_features;
            last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_features;
            device_extensions.push_back("VK_KHR_shader_integer_dot_product");
        }

        VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE dot2_features {};
        dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE;
        ಯದಿ (dot2_f16_support) {
            last_struct->pNext = (VkBaseOutStructure *)&dot2_features;
            last_struct = (VkBaseOutStructure *)&dot2_features;
            device_extensions.push_back("VK_VALVE_shader_mixed_float_dot_product");
        }

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_features {};
        pep_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        ಯದಿ (pipeline_executable_properties_support) {
            last_struct->pNext = (VkBaseOutStructure *)&pep_features;
            last_struct = (VkBaseOutStructure *)&pep_features;
            device_extensions.push_back("VK_KHR_pipeline_executable_properties");
        }

        ಯದಿ (device->external_memory_host) {
            device_extensions.push_back("VK_EXT_external_memory_host");
        }

#ಯದಿ defined(VK_EXT_shader_64bit_indexing)
        VkPhysicalDeviceShader64BitIndexingFeaturesEXT shader_64bit_indexing_features {};
        shader_64bit_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_BIT_INDEXING_FEATURES_EXT;
        ಯದಿ (device->shader_64b_indexing) {
            last_struct->pNext = (VkBaseOutStructure *)&shader_64bit_indexing_features;
            last_struct = (VkBaseOutStructure *)&shader_64bit_indexing_features;
            device_extensions.push_back("VK_EXT_shader_64bit_indexing");
        }
#endif

        VkPhysicalDeviceFaultFeaturesEXT fault_features {};
        fault_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        ಯದಿ (device->device_fault) {
            last_struct->pNext = (VkBaseOutStructure *)&fault_features;
            last_struct = (VkBaseOutStructure *)&fault_features;
            device_extensions.push_back("VK_EXT_device_fault");
        }

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->device_fault = device->device_fault && fault_features.deviceFault;

        device->has_internally_synchronized_queues = internally_synchronized_queues_features.internallySynchronizedQueues;

        // Build queue create infos only after querying whether internally synchronized queues are enabled.
        // getQueue2() later uses the same flag, so creation/retrieval must stay consistent.
        vk::DeviceQueueCreateFlags queue_flags = device->has_internally_synchronized_queues ?
                                                eInternallySynchronizedKHR :
                                                vk::DeviceQueueCreateFlags();

        ಯದಿ (compute_queue_family_index != transfer_queue_family_index) {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_index, 1, priorities});
            device_queue_create_infos.push_back({queue_flags, transfer_queue_family_index, 1, priorities + 1});
        } ಅನ್ಯಥಾ ಯದಿ(!device->single_queue) {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_index, 2, priorities});
        } ಅನ್ಯಥಾ {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_index, 1, priorities});
        }

        device->pipeline_executable_properties_support = pipeline_executable_properties_support;

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

#ಯದಿ defined(VK_KHR_shader_bfloat16)
        device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#ಅನ್ಯಥಾ
        device->bf16 = ಅಸತ್ಯ;
#endif

        device->dot2_f16 = dot2_f16_support && dot2_features.shaderMixedFloatDotProductFloat16AccFloat32;
        device->ocp_fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                          shader_float8_extension && shader_float8_features.shaderFloat8 &&
                          !getenv("GGML_VK_DISABLE_OCP_FP4");

        device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

        device->multi_add = vk12_props.shaderRoundingModeRTEFloat16 &&
                            device->properties.limits.max_ತಳ್ಳು_Constants_Size >= ಪರಿಮಾಣ(vk_op_multi_add_ತಳ್ಳು_constants) &&
                            getenv("GGML_VK_DISABLE_MULTI_ADD") == nullptr;

        device->shader_int64 = device_features2.features.shaderInt64;
        device->buffer_device_address = vk12_features.bufferDeviceAddress;
        device->vulkan_memory_model = vk12_features.vulkanMemoryModel;

        ಯದಿ (device->subgroup_size_control) {
            device->subgroup_min_size = subgroup_size_control_props.minSubgroupSize;
            device->subgroup_max_size = subgroup_size_control_props.maxSubgroupSize;
            device_extensions.push_back("VK_EXT_subgroup_size_control");
        }

        device->subgroup_size_control = device->subgroup_size_control &&
                (subgroup_size_control_props.requiredSubgroupSizeStages & vk::ShaderStageFlagBits::eCompute) &&
                subgroup_size_control_features.subgroupSizeControl;

        device->subgroup_require_full_support = subgroup_size_control_features.computeFullSubgroups;

#ಯದಿ defined(VK_KHR_cooperative_matrix)
        device->coopmat_support = device->coopmat_support && coopmat_features.cooperativeMatrix;
        device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_full_support;
#endif

        ಯದಿ (coopmat2_support) {
#ಯದಿ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            ಯದಿ (coopmat2_features.cooperativeMatrixWorkgroupScope &&
                coopmat2_features.cooperativeMatrixFlexibleDimensions &&
                coopmat2_features.cooperativeMatrixReductions &&
                coopmat2_features.cooperativeMatrixConversions &&
                coopmat2_features.cooperativeMatrixPerElementOperations &&
                coopmat2_features.cooperativeMatrixTensorAddressing &&
                coopmat2_features.cooperativeMatrixBlockLoads &&
                vk12_features.bufferDeviceAddress) {

                std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV> flexible_dimensions;
                uint32_t count = 0;

                PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV
                    _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV =
                        (PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV)
                        vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

                _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, nullptr);

                VkCooperativeMatrixFlexibleDimensionsPropertiesNV empty_prop {};
                empty_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
                flexible_dimensions.resize(count, empty_prop);

                _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, flexible_dimensions.data());

                ತಾರ್ಕಿಕ found_fp16_128 = ಅಸತ್ಯ,
                     found_fp16_256 = ಅಸತ್ಯ,
                     found_fp32_128 = ಅಸತ್ಯ,
                     found_fp32_256 = ಅಸತ್ಯ;
                ತಾರ್ಕಿಕ found_bf16_128 = ಅಸತ್ಯ,
                     found_bf16_256 = ಅಸತ್ಯ;
                // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
                // with 32x16x16 and 256 with 32x32x16.
                ಪ್ರತಿ (auto &prop : flexible_dimensions) {
                    ಯದಿ (prop.saturatingAccumulation == VK_FALSE &&
                        prop.scope == VK_SCOPE_WORKGROUP_KHR) {

                        ಯದಿ (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {

                            ಯದಿ (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                ಯದಿ (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    found_fp16_128 = ಸತ್ಯ;
                                }
                                ಯದಿ (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    found_fp32_128 = ಸತ್ಯ;
                                }
                            }
                            ಯದಿ (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                ಯದಿ (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    found_fp16_256 = ಸತ್ಯ;
                                }
                                ಯದಿ (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    found_fp32_256 = ಸತ್ಯ;
                                }
                            }
                        }

#ಯದಿ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                        ಯದಿ (bfloat16_support &&
                            prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {

                            ಯದಿ (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                found_bf16_128 = ಸತ್ಯ;
                            }
                            ಯದಿ (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                found_bf16_256 = ಸತ್ಯ;
                            }
                        }
#endif
                    }
                }
                ಯದಿ (found_fp16_128 && found_fp16_256 &&
                    found_fp32_128 && found_fp32_256 &&
                    coopmat2_props.cooperativeMatrixFlexibleDimensionsMaxDimension >= 512) {
                    device->coopmat2 = ಸತ್ಯ;
                    device->coopmat2_bf16_support = found_bf16_128 && found_bf16_256;
                    device->coopmat2_decode_vector = coopmat2_decode_vector_support && coopmat2_decode_vector_features.cooperativeMatrixDecodeVector;
                }
            }
#endif
        }

        ಯದಿ (!vk11_features.storageBuffer16BitAccess) {
            std::cerr << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::endl;
            ಕ್ಷಿಪ std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#ifdef GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#endif

        ಯದಿ (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }

#ಯದಿ defined(VK_KHR_cooperative_matrix)
        ಯದಿ (device->coopmat_support) {
            // Query supported shapes
            std::vector<VkCooperativeMatrixPropertiesKHR> cm_props;

            PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
                (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(vk_instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

            uint32_t cm_props_ಅಂಕೆ;

            pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_ಅಂಕೆ, nullptr);

            cm_props.resize(cm_props_ಅಂಕೆ);

            ಪ್ರತಿ (auto& prop : cm_props) {
                prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }

            pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_ಅಂಕೆ, cm_props.data());

            VK_LOG_DEBUG("ggml_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

            ಪ್ರತಿ (auto& prop : cm_props) {
                VK_LOG_DEBUG("ggml_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " << vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.ResultType) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk::ScopeKHR)prop.scope));

                ಯದಿ ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    ಯದಿ ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
                        (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eFloat32) {
                        // coopmat sizes not set yet
                        ಯದಿ (device->coopmat_m == 0) {
                            device->coopmat_acc_f32_support = ಸತ್ಯ;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } ಅನ್ಯಥಾ ಯದಿ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f32_support = ಸತ್ಯ;
                        }
                        ಯದಿ (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f32acc = ಸತ್ಯ;
                        }
                    } ಅನ್ಯಥಾ ಯದಿ ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
                               (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eFloat16) {
                        // coopmat sizes not set yet
                        ಯದಿ (device->coopmat_m == 0) {
                            device->coopmat_acc_f16_support = ಸತ್ಯ;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } ಅನ್ಯಥಾ ಯದಿ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f16_support = ಸತ್ಯ;
                        }
                        ಯದಿ (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f16acc = ಸತ್ಯ;
                        }
                    }
                } ಅನ್ಯಥಾ ಯದಿ ((vk::ComponentTypeKHR)prop.AType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.BType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.CType      == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
                           device->coopmat_int_m == 0
                ) {
                    device->coopmat_int_support = ಸತ್ಯ;
                    device->coopmat_int_m = prop.MSize;
                    device->coopmat_int_n = prop.NSize;
                    device->coopmat_int_k = prop.KSize;
                }
#ಯದಿ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ಯದಿ (bfloat16_support &&
                    prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    // coopmat sizes not set yet
                    ಯದಿ (device->coopmat_m == 0) {
                        device->coopmat_bf16_support = ಸತ್ಯ;
                        device->coopmat_m = prop.MSize;
                        device->coopmat_n = prop.NSize;
                        device->coopmat_k = prop.KSize;
                    } ಅನ್ಯಥಾ ಯದಿ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                        // Only enable if shape is identical
                        device->coopmat_bf16_support = ಸತ್ಯ;
                    }
                }
#endif
            }

            ಯದಿ (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
                // No suitable matmul mode found
                GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
                device->coopmat_support = ಅಸತ್ಯ;
            }
        }

        ಯದಿ (device->coopmat_support) {
            device_extensions.push_back("VK_KHR_cooperative_matrix");
        }
#endif
        device->name = GGML_VK_NAME + std::to_string(idx);

        device_create_info
            .setFlags(vk::DeviceCreateFlags())
            .setQueueCreateInfos(device_queue_create_infos)
            .setPEnabledExtensionNames(device_extensions);
        device_create_info.setPNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        ಯದಿ (device->device_fault) {
            device->pfn_vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)
                vkGetDeviceProcAddr(device->device, "vkGetDeviceFaultInfoEXT");
        }

        // Queues
        device->compute_queue = ggml_vk_create_queue(device, compute_queue_family_index, 0, { vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer }, ಅಸತ್ಯ);

        // Shaders
        // Disable matmul tile sizes early if performance low or not supported
        ಪ್ರತಿ (uint32_t i = 0; i < GGML_TYPE_ಎಣಿಕೆ; ++i) {
            ವಿಕಲ್ಪ (device->vendor_id) {
#ifndef GGML_VULKAN_RUN_TESTS
            ಪಕ್ಷ VK_VENDOR_ID_AMD:
                device->mul_mat_l[i]    = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary;
                device->mul_mat_m[i]    = ಸತ್ಯ;
                device->mul_mat_s[i]    = ಸತ್ಯ;
                device->mul_mat_id_l[i] = ಅಸತ್ಯ;
                device->mul_mat_id_m[i] = ಸತ್ಯ;
                device->mul_mat_id_s[i] = ಸತ್ಯ;
                ಭಙ್ಗ;
            ಪಕ್ಷ VK_VENDOR_ID_INTEL: {
                // Current Windows driver does not expose BF16 support.
                // We only want to use l_warptile if coopmat is available
                ಧ್ರುವ ತಾರ್ಕಿಕ use_l_warptile = (i == GGML_TYPE_BF16) ? (device->coopmat_bf16_support && device->coopmat_support) : device->coopmat_support;
                device->mul_mat_l[i] = use_l_warptile;
                device->mul_mat_id_l[i] = use_l_warptile;
                device->mul_mat_m[i] = ಸತ್ಯ;
                device->mul_mat_s[i] = ಸತ್ಯ;
                device->mul_mat_id_m[i] = ಸತ್ಯ;
                device->mul_mat_id_s[i] = ಸತ್ಯ;
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ VK_VENDOR_ID_APPLE:
                device->mul_mat_l[i] = ಅಸತ್ಯ;
                device->mul_mat_m[i] = ಸತ್ಯ;
                device->mul_mat_s[i] = ಅಸತ್ಯ;
                device->mul_mat_id_l[i] = ಅಸತ್ಯ;
                device->mul_mat_id_m[i] = ಸತ್ಯ;
                device->mul_mat_id_s[i] = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            ಪಕ್ಷ VK_VENDOR_ID_QUALCOMM:
                device->mul_mat_l[i] = ಅಸತ್ಯ;
                device->mul_mat_m[i] = ಸತ್ಯ;
                device->mul_mat_s[i] = !device->coopmat_support;
                device->mul_mat_id_l[i] = ಅಸತ್ಯ;
                device->mul_mat_id_m[i] = ಸತ್ಯ;
                device->mul_mat_id_s[i] = !device->coopmat_support;
                ಭಙ್ಗ;
            ಪಕ್ಷ VK_VENDOR_ID_SAMSUNG:
                device->mul_mat_l[i] = ಅಸತ್ಯ;
                device->mul_mat_m[i] = ಸತ್ಯ;
                device->mul_mat_s[i] = ಸತ್ಯ;
                device->mul_mat_id_l[i] = ಅಸತ್ಯ;
                device->mul_mat_id_m[i] = ಸತ್ಯ;
                device->mul_mat_id_s[i] = ಸತ್ಯ;
                ಭಙ್ಗ;
#endif
            ಸ್ವಭಾವ:
                device->mul_mat_l[i] = ಸತ್ಯ;
                device->mul_mat_m[i] = ಸತ್ಯ;
                device->mul_mat_s[i] = ಸತ್ಯ;
                device->mul_mat_id_l[i] = ಸತ್ಯ;
                device->mul_mat_id_m[i] = ಸತ್ಯ;
                device->mul_mat_id_s[i] = ಸತ್ಯ;
                ಭಙ್ಗ;
            }

#ಯದಿ VK_HEADER_VERSION >= 287
            // Honeykrisp driver for Asahi Linux doesn't report VK_VENDOR_ID_APPLE.
            // Check for Honeykrisp driver and force same configuration as the VK_VENDOR_ID_APPLE case.
            ಯದಿ (device->driver_id == vk::DriverId::eMesaHoneykrisp) {
                device->mul_mat_l[i] = ಅಸತ್ಯ;
                device->mul_mat_m[i] = ಸತ್ಯ;
                device->mul_mat_s[i] = ಅಸತ್ಯ;
                device->mul_mat_id_l[i] = ಅಸತ್ಯ;
                device->mul_mat_id_m[i] = ಸತ್ಯ;
                device->mul_mat_id_s[i] = ಅಸತ್ಯ;
            }
#endif

            device->mul_mat_l_int[i]    = device->mul_mat_l[i];
            device->mul_mat_m_int[i]    = device->mul_mat_m[i];
            device->mul_mat_s_int[i]    = device->mul_mat_s[i];
            device->mul_mat_id_l_int[i] = device->mul_mat_id_l[i];
            device->mul_mat_id_m_int[i] = device->mul_mat_id_m[i];
            device->mul_mat_id_s_int[i] = device->mul_mat_id_s[i];
        }


        std::vector<vk::DescriptorSetLayoutBinding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        ಪ್ರತಿ (uint32_t i = 0; i < MAX_PARAMETER_ಎಣಿಕೆ; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::DescriptorSetLayoutBindingFlagsCreateInfo dslbfci = { dsl_binding_flags };

        vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
            {},
            dsl_binding);
        descriptor_set_layout_create_info.setPNext(&dslbfci);
        device->dsl = device->device.createDescriptorSetLayout(descriptor_set_layout_create_info);

        ggml_vk_load_shaders(device);

        // Prefer a dedicated transfer queue on AMD dGPUs (non-GCN) when graphics queue use is disabled.
        ಧ್ರುವ ತಾರ್ಕಿಕ prefers_transfer_queue =
            device->vendor_id == VK_VENDOR_ID_AMD &&
            device->architecture != AMD_GCN &&
            !device->uma &&
            !allow_graphics_queue;

        ಯದಿ (!device->single_queue) {
            ಧ್ರುವ uint32_t transfer_queue_index = compute_queue_family_index == transfer_queue_family_index ? 1 : 0;
            device->transfer_queue = ggml_vk_create_queue(device, transfer_queue_family_index, transfer_queue_index, { vk::PipelineStageFlagBits::eTransfer }, ಸತ್ಯ);

            device->async_use_transfer_queue = prefers_transfer_queue || (getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != nullptr);
        } ಅನ್ಯಥಾ {
            device->transfer_queue = ggml_vk_create_aliased_queue(device, device->compute_queue);

            device->async_use_transfer_queue = ಅಸತ್ಯ;
        }

        device->fence = device->device.createFence({});

        device->buffer_type = {
            /* .iface    = */ ggml_backend_vk_buffer_type_interface,
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
            /* .context  = */ ನೂತನ ggml_backend_vk_buffer_type_context{ device->name, device },
        };

        device->idx = idx;

        device->serialize_submissions = getenv("GGML_VK_SERIALIZE_SUBMISSIONS") != nullptr;

        device->disable_fusion = getenv("GGML_VK_DISABLE_FUSION") != nullptr;

        device->disable_descriptor_reuse = getenv("GGML_VK_DISABLE_DESCRIPTOR_REUSE") != nullptr;

        device->add_rms_fusion = !device->disable_fusion &&
                                 device->subgroup_arithmetic &&
                                 device->vendor_id != VK_VENDOR_ID_INTEL;
        device->partials_binding_alignment =
            std::max(4u, (uint32_t)device->properties.limits.minStorageBufferOffsetAlignment);

        device->mmvq_mode = 0;
        ಯದಿ (getenv("GGML_VK_DISABLE_MMVQ")) {
            device->mmvq_mode = -1;
        } ಅನ್ಯಥಾ ಯದಿ (getenv("GGML_VK_FORCE_MMVQ")) {
            device->mmvq_mode = 1;
        }

        // publish the device only when it is fully built
        vk_instance.devices[idx] = device;

        ಪ್ರತ್ಯಾಗಮ device;
    }

    ಪ್ರತ್ಯಾಗಮ vk_instance.devices[idx];
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_gpu_info(size_t idx) {
    GGML_ASSERT(idx < vk_instance.device_indices.size());
    size_t dev_ಅಂಕೆ = vk_instance.device_indices[idx];
    VK_LOG_DEBUG("ggml_vk_print_gpu_info(" << dev_ಅಂಕೆ << ")");
    GGML_ASSERT(vk_instance_initialized);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    ಯದಿ (dev_ಅಂಕೆ >= devices.size()) {
        std::cerr << "ggml_vulkan: Device with index " << dev_ಅಂಕೆ << " does not exist." << std::endl;
        ಕ್ಷಿಪ std::runtime_error("Device not found");
    }

    vk::PhysicalDevice physical_device = devices[dev_ಅಂಕೆ];
    std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

    ತಾರ್ಕಿಕ fp16_storage = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ fp16_compute = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ coopmat_support = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ coopmat2_support = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ coopmat2_decode_vector_support = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ integer_dot_product = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ bfloat16_support = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ dot2_f16_support = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ ocp_microscaling_extension = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ shader_float8_extension = ಅಸತ್ಯ;

    ಪ್ರತಿ (auto properties : ext_props) {
        ಯದಿ (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = ಸತ್ಯ;
        } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = ಸತ್ಯ;
#ಯದಿ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
       } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = ಸತ್ಯ;
#endif
#ಯದಿ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = ಸತ್ಯ;
#endif
        } ಅನ್ಯಥಾ ಯದಿ (strcmp(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME, properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_vector_support = ಸತ್ಯ;
#ಯದಿ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_product = ಸತ್ಯ;
#endif
#ಯದಿ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = ಸತ್ಯ;
#endif
#ಯದಿ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
        } ಅನ್ಯಥಾ ಯದಿ (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
            ocp_microscaling_extension = ಸತ್ಯ;
#endif
#ಯದಿ defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        } ಅನ್ಯಥಾ ಯದಿ (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
            shader_float8_extension = ಸತ್ಯ;
#endif
        } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = ಸತ್ಯ;
        }
    }

    ಧ್ರುವ vk_device_architecture device_architecture = get_device_architecture(physical_device);

    ಧ್ರುವ ವರ್ಣ* GGML_VK_DISABLE_F16 = getenv("GGML_VK_DISABLE_F16");
    ತಾರ್ಕಿಕ force_disable_f16 = GGML_VK_DISABLE_F16 != nullptr;

    ತಾರ್ಕಿಕ fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR shader_integer_dot_product_props;
    props2.pNext = &props3;
    props3.pNext = &subgroup_props;
    subgroup_props.pNext = &driver_props;

    // Pointer to the last chain element
    VkBaseOutStructure * last_struct = (VkBaseOutStructure *)&driver_props;

    ಯದಿ (integer_dot_product) {
        last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_props;
        last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_props;
    }

    physical_device.getProperties2(&props2);

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.pNext = nullptr;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.pNext = nullptr;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.pNext = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.pNext = nullptr;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.pNext = &vk12_features;

    // Pointer to the last chain element
    last_struct = (VkBaseOutStructure *)&vk12_features;

#ಯದಿ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
    coopmat_features.pNext = nullptr;
    coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopmat_features.cooperativeMatrix = VK_FALSE;

    ಯದಿ (coopmat_support) {
        last_struct->pNext = (VkBaseOutStructure *)&coopmat_features;
        last_struct = (VkBaseOutStructure *)&coopmat_features;
    }
#endif

    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features {};
    shader_integer_dot_product_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    ಯದಿ (integer_dot_product) {
        last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_features;
        last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_features;
    }

#ಯದಿ defined(VK_KHR_shader_bfloat16)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
    bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    ಯದಿ (bfloat16_support) {
        last_struct->pNext = (VkBaseOutStructure *)&bfloat16_features;
        last_struct = (VkBaseOutStructure *)&bfloat16_features;
    }
#endif

#ಯದಿ defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
    coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
    ಯದಿ (coopmat2_support) {
        last_struct->pNext = (VkBaseOutStructure *)&coopmat2_features;
        last_struct = (VkBaseOutStructure *)&coopmat2_features;
    }
#endif

    VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV coopmat2_decode_vector_features {};
    coopmat2_decode_vector_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV;
    ಯದಿ (coopmat2_decode_vector_support) {
        last_struct->pNext = (VkBaseOutStructure *)&coopmat2_decode_vector_features;
        last_struct = (VkBaseOutStructure *)&coopmat2_decode_vector_features;
    }

    VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE dot2_features {};
    dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE;
    ಯದಿ (dot2_f16_support) {
        last_struct->pNext = (VkBaseOutStructure *)&dot2_features;
        last_struct = (VkBaseOutStructure *)&dot2_features;
    }

#ಯದಿ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
    ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
    shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    ಯದಿ (ocp_microscaling_extension) {
        last_struct->pNext = (VkBaseOutStructure *)&ocp_microscaling_features;
        last_struct = (VkBaseOutStructure *)&ocp_microscaling_features;
    }
    ಯದಿ (shader_float8_extension) {
        last_struct->pNext = (VkBaseOutStructure *)&shader_float8_features;
        last_struct = (VkBaseOutStructure *)&shader_float8_features;
    }
#endif

    vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 && vk12_features.shaderFloat16;

#ಯದಿ defined(VK_KHR_shader_bfloat16)
    ತಾರ್ಕಿಕ bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#ಅನ್ಯಥಾ
    ತಾರ್ಕಿಕ bf16 = ಅಸತ್ಯ;
#endif

    uint32_t default_subgroup_size = get_subgroup_size("", device_architecture);
    ಧ್ರುವ size_t subgroup_size = (default_subgroup_size != 0) ? default_subgroup_size : subgroup_props.subgroupSize;
    ಧ್ರುವ ತಾರ್ಕಿಕ uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    integer_dot_product = integer_dot_product
                       && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated
                       && shader_integer_dot_product_features.shaderIntegerDotProduct;

    coopmat_support = coopmat_support
#ಯದಿ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                   && coopmat_features.cooperativeMatrix
#endif
                   && ggml_vk_khr_cooperative_matrix_support(props2.properties, driver_props, device_architecture);

#ಯದಿ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    coopmat2_support = coopmat2_support &&
                       coopmat2_features.cooperativeMatrixWorkgroupScope &&
                       coopmat2_features.cooperativeMatrixFlexibleDimensions &&
                       coopmat2_features.cooperativeMatrixReductions &&
                       coopmat2_features.cooperativeMatrixConversions &&
                       coopmat2_features.cooperativeMatrixPerElementOperations &&
                       coopmat2_features.cooperativeMatrixTensorAddressing &&
                       coopmat2_features.cooperativeMatrixBlockLoads;
#ಅನ್ಯಥಾ
    coopmat2_support = ಅಸತ್ಯ;
#endif

    coopmat2_decode_vector_support = coopmat2_decode_vector_support && coopmat2_decode_vector_features.cooperativeMatrixDecodeVector;
#ಯದಿ !defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
    coopmat2_decode_vector_support = ಅಸತ್ಯ;
#endif

    std::string matrix_cores = coopmat2_support ? (coopmat2_decode_vector_support ? "NV_coopmat2v" : "NV_coopmat2")
                             : coopmat_support  ? "KHR_coopmat"
                             : "none";

    ತಾರ್ಕಿಕ dot2_f16 = dot2_f16_support && dot2_features.shaderMixedFloatDotProductFloat16AccFloat32;
    ಧ್ರುವ ವರ್ಣ *fp16_str = fp16 ? (dot2_f16 ? "dot2" : "1") : "0";
#ಯದಿ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    ಧ್ರುವ ತಾರ್ಕಿಕ fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                     shader_float8_extension && shader_float8_features.shaderFloat8 &&
                     !getenv("GGML_VK_DISABLE_OCP_FP4");
#ಅನ್ಯಥಾ
    GGML_UNUSED(ocp_microscaling_extension);
    GGML_UNUSED(shader_float8_extension);
    ಧ್ರುವ ತಾರ್ಕಿಕ fp4 = ಅಸತ್ಯ;
#endif

    std::string device_name = props2.properties.deviceName.data();
    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
              idx, device_name.c_str(), driver_props.driverName.data(), uma, fp16_str, bf16, fp4, subgroup_size,
              props2.properties.limits.maxComputeSharedMemorySize, integer_dot_product, matrix_cores.c_str());

    ಯದಿ (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
        GGML_LOG_DEBUG("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n");
    }
}

ಸ್ಥಿರ DispatchLoaderDynamic ggml_vk_default_dispatcher_instance;

DispatchLoaderDynamic & ggml_vk_default_dispatcher() {
    ಪ್ರತ್ಯಾಗಮ ggml_vk_default_dispatcher_instance;
}

ಶೂನ್ಯ ggml_vk_instance_init() {
    ಯದಿ (vk_instance_initialized) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    ggml_vk_default_dispatcher_instance.init(vkGetInstanceProcAddr);

    // vkEnumerateInstanceVersion is Vulkan 1.1. A null value indicated Vulkan 1.0.
    ಯದಿ (ggml_vk_default_dispatcher_instance.vkEnumerateInstanceVersion == nullptr) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        ಕ್ಷಿಪ vk::SystemError(vk::Result::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    uint32_t api_version = vk::enumerateInstanceVersion();

    ಯದಿ (api_version < VK_API_VERSION_1_2) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        ಕ್ಷಿಪ vk::SystemError(vk::Result::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, nullptr, 0, api_version };

    ಧ್ರುವ std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    ಧ್ರುವ ತಾರ್ಕಿಕ layer_settings = ggml_vk_instance_layer_settings_available();
#ifdef __APPLE__
    ಧ್ರುವ ತಾರ್ಕಿಕ portability_enumeration_ext = ggml_vk_instance_portability_enumeration_ext_available(instance_extensions);
#endif
    ಧ್ರುವ ತಾರ್ಕಿಕ debug_utils_ext = ggml_vk_instance_debug_utils_ext_available(instance_extensions) && getenv("GGML_VK_DEBUG_MARKERS") != nullptr;
    std::vector<ಧ್ರುವ ವರ್ಣ*> layers;

    ಯದಿ (layer_settings) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<ಧ್ರುವ ವರ್ಣ*> extensions;
    ಯದಿ (layer_settings) {
        extensions.push_back("VK_EXT_layer_settings");
    }
#ifdef __APPLE__
    ಯದಿ (portability_enumeration_ext) {
        extensions.push_back("VK_KHR_portability_enumeration");
    }
#endif
    ಯದಿ (debug_utils_ext) {
        extensions.push_back("VK_EXT_debug_utils");
    }
    VkBool32 enable_best_practice = layer_settings;
    std::vector<vk::LayerSettingEXT> settings = {
        {
            "VK_LAYER_KHRONOS_validation",
            "validate_best_practices",
            vk::LayerSettingTypeEXT::eBool32,
            1,
            &enable_best_practice
        },
    };
    vk::LayerSettingsCreateInfoEXT layer_setting_info(settings);
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions, &layer_setting_info);
#ifdef __APPLE__
    ಯದಿ (portability_enumeration_ext) {
        instance_create_info.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
#endif

    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = ಸತ್ಯ;

    ಯದಿ (debug_utils_ext) {
        vk_instance.debug_utils_support              = ಸತ್ಯ;
        vk_instance.pfn_vkSetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkSetDebugUtilsObjectNameEXT");
        vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT = (PFN_vkQueueBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vkQueueEndDebugUtilsLabelEXT = (PFN_vkQueueEndDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueEndDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdEndDebugUtilsLabelEXT =   (PFN_vkCmdEndDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdEndDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_ಸೇರಿಸು_Debug_Utils_Label_EXT = (PFN_vk_Cmd_ಸೇರಿಸು_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdInsertDebugUtilsLabelEXT");
    }

    vk_perf_logger_enabled = getenv("GGML_VK_PERF_LOGGER") != nullptr;
    vk_perf_logger_concurrent = getenv("GGML_VK_PERF_LOGGER_CONCURRENT") != nullptr;
    vk_enable_sync_logger = getenv("GGML_VK_SYNC_LOGGER") != nullptr;
    vk_memory_logger_enabled = getenv("GGML_VK_MEMORY_LOGGER") != nullptr;
    ಧ್ರುವ ವರ್ಣ* GGML_VK_PIPELINE_STATS = getenv("GGML_VK_PIPELINE_STATS");
    ಯದಿ (GGML_VK_PIPELINE_STATS != nullptr) {
        vk_pipeline_stats_filter = GGML_VK_PIPELINE_STATS;
    }
    ಧ್ರುವ ವರ್ಣ* GGML_VK_PERF_LOGGER_FREQUENCY = getenv("GGML_VK_PERF_LOGGER_FREQUENCY");

    ಯದಿ (GGML_VK_PERF_LOGGER_FREQUENCY != nullptr) {
        vk_perf_logger_frequency = std::stoul(GGML_VK_PERF_LOGGER_FREQUENCY);
    }

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    ವರ್ಣ * devices_env = getenv("GGML_VK_VISIBLE_DEVICES");
    ಯದಿ (devices_env != nullptr) {
        size_t ಅಂಕೆ_available_devices = devices.size();

        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        size_t tmp;
        ಯಾವತ್ (ss >> tmp) {
            ಯದಿ(tmp >= ಅಂಕೆ_available_devices) {
                std::cerr << "ggml_vulkan: Invalid device index " << tmp << " in GGML_VK_VISIBLE_DEVICES." << std::endl;
                ಕ್ಷಿಪ std::runtime_error("Invalid Vulkan device index");
            }
            vk_instance.device_indices.push_back(tmp);
        }
    } ಅನ್ಯಥಾ {
        // If no vulkan devices are found, return early
        ಯದಿ (devices.empty()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            ಪ್ರತ್ಯಾಗಮ;
        }

        // Default to using all dedicated GPUs
        ಪ್ರತಿ (size_t i = 0; i < devices.size(); i++) {
            vk::PhysicalDeviceProperties2 new_props;
            vk::PhysicalDeviceDriverProperties new_driver;
            vk::PhysicalDeviceIDProperties new_id;
            new_props.pNext = &new_driver;
            new_driver.pNext = &new_id;
            devices[i].getProperties2(&new_props);

            ಯದಿ ((new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || new_props.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) && ggml_vk_device_is_supported(devices[i])) {
                // Check if there are two physical devices corresponding to the same GPU
                // This handles the case where the same GPU appears with different drivers (e.g., RADV + AMDVLK on Linux),
                // see https://github.com/ggml-org/llama.cpp/pull/7582 for original deduplication.
                // MoltenVK on macOS may report the same UUID for distinct GPUs on multi-GPU cards,
                // see https://github.com/KhronosGroup/MoltenVK/issues/2683. Skip when both old/new
                // driver is MoltenVK
                auto old_device = std::find_if(
                    vk_instance.device_indices.begin(),
                    vk_instance.device_indices.end(),
                    [&devices, &new_id, &new_driver](ಧ್ರುವ size_t k){
                        vk::PhysicalDeviceProperties2 old_props;
                        vk::PhysicalDeviceDriverProperties old_driver;
                        vk::PhysicalDeviceIDProperties old_id;
                        old_props.pNext = &old_driver;
                        old_driver.pNext = &old_id;
                        devices[k].getProperties2(&old_props);

                        ತಾರ್ಕಿಕ same_uuid = std::equal(std::begin(old_id.deviceUUID), std::end(old_id.deviceUUID), std::begin(new_id.deviceUUID));
                        same_uuid = same_uuid || (
                            old_id.deviceLUIDValid && new_id.deviceLUIDValid &&
                            std::equal(std::begin(old_id.deviceLUID), std::end(old_id.deviceLUID), std::begin(new_id.deviceLUID))
                        );
                        ತಾರ್ಕಿಕ both_molten_vk = (new_driver.driverID == vk::DriverId::eMoltenvk && old_driver.driverID == vk::DriverId::eMoltenvk);

                        ಪ್ರತ್ಯಾಗಮ same_uuid && !both_molten_vk;
                    }
                );
                ಯದಿ (old_device == vk_instance.device_indices.end()) {
                    vk_instance.device_indices.push_back(i);
                } ಅನ್ಯಥಾ {
                    // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
                    // This can cause error when splitting layers aross the devices, need to keep only 1
                    VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

                    vk::PhysicalDeviceProperties2 old_props;
                    vk::PhysicalDeviceDriverProperties old_driver;
                    old_props.pNext = &old_driver;
                    devices[*old_device].getProperties2(&old_props);

                    std::map<vk::DriverId, ಪೂರ್ಣಾಙ್ಕ> driver_priorities {};
                    ಪೂರ್ಣಾಙ್ಕ old_priority = std::numeric_limits<ಪೂರ್ಣಾಙ್ಕ>::max();
                    ಪೂರ್ಣಾಙ್ಕ new_priority = std::numeric_limits<ಪೂರ್ಣಾಙ್ಕ>::max();

                    // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
                    // Smaller number -> higher priority
                    ವಿಕಲ್ಪ (old_props.properties.vendorID) {
                        ಪಕ್ಷ VK_VENDOR_ID_AMD:
                            driver_priorities[vk::DriverId::eMesaRadv] = 1;
                            driver_priorities[vk::DriverId::eAmdOpenSource] = 2;
                            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
                            ಭಙ್ಗ;
                        ಪಕ್ಷ VK_VENDOR_ID_INTEL:
                            driver_priorities[vk::DriverId::eIntelOpenSourceMESA] = 1;
                            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
                            ಭಙ್ಗ;
                        ಪಕ್ಷ VK_VENDOR_ID_NVIDIA:
                            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
#ಯದಿ defined(VK_API_VERSION_1_3) && VK_HEADER_VERSION >= 235
                            driver_priorities[vk::DriverId::eMesaNvk] = 2;
#endif
                            ಭಙ್ಗ;
                        ಪಕ್ಷ VK_VENDOR_ID_QUALCOMM:
                            driver_priorities[vk::DriverId::eQualcommProprietary] = 1;
                            driver_priorities[vk::DriverId::eMesaTurnip] = 2;
                            ಭಙ್ಗ;
                    }
                    driver_priorities[vk::DriverId::eMesaDozen] = 100;

                    ಯದಿ (driver_priorities.count(old_driver.driverID)) {
                        old_priority = driver_priorities[old_driver.driverID];
                    }
                    ಯದಿ (driver_priorities.count(new_driver.driverID)) {
                        new_priority = driver_priorities[new_driver.driverID];
                    }

                    ಯದಿ (new_priority < old_priority) {
                        auto r = std::remove(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
                        vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
                        vk_instance.device_indices.push_back(i);

                        VK_LOG_DEBUG("Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device << " driver " << old_driver.driverName);
                    }
                    ಅನ್ಯಥಾ {
                        VK_LOG_DEBUG("Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i << " driver " << new_driver.driverName << std::endl);
                    }
                }
            }
        }

        // If no GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        ಯದಿ (vk_instance.device_indices.empty()) {
            ಪ್ರತಿ (size_t i = 0; i < devices.size(); i++) {
                ಯದಿ (devices[i].getProperties().deviceType != vk::PhysicalDeviceType::eCpu) {
                    vk_instance.device_indices.push_back(i);
                    ಭಙ್ಗ;
                }
            }
        }

        ಯದಿ (vk_instance.device_indices.empty()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            ಪ್ರತ್ಯಾಗಮ;
        }
    }
    GGML_LOG_DEBUG("ggml_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

    size_t host_buffer_import_ಎಣಿಕೆ = 0;
    ಪ್ರತಿ (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
        vk::PhysicalDevice vkdev = devices[vk_instance.device_indices[i]];
        std::vector<vk::ExtensionProperties> extensionprops = vkdev.enumerateDeviceExtensionProperties();

        ತಾರ್ಕಿಕ membudget_supported = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ external_memory_host_supported = ಅಸತ್ಯ;
        ಪ್ರತಿ (ಧ್ರುವ auto & ext : extensionprops) {
            ಯದಿ (strcmp(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
                membudget_supported = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (strcmp("VK_EXT_external_memory_host", ext.extensionName) == 0) {
                external_memory_host_supported = ಸತ್ಯ;
            }
        }

        vk_instance.device_supports_membudget.push_back(membudget_supported);

        ತಾರ್ಕಿಕ host_buffer_import = ಅಸತ್ಯ;
        ಯದಿ (external_memory_host_supported) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceDriverProperties driver_props;
            vk::PhysicalDeviceExternalMemoryHostPropertiesEXT ext_props;
            props2.pNext = &driver_props;
            driver_props.pNext = &ext_props;
            vkdev.getProperties2(&props2);
            host_buffer_import = ggml_vk_external_memory_host_usable(driver_props.driverID) &&
                                 ggml_vk_host_buffer_import_usable(props2.properties.vendorID);
            ಯದಿ (host_buffer_import) {
                vk_instance.host_import_alignment = std::max<size_t>(vk_instance.host_import_alignment, ext_props.minImportedHostPointerAlignment);
                host_buffer_import_ಎಣಿಕೆ++;
            }
        }
        vk_instance.device_host_buffer_import.push_back(host_buffer_import);

        ggml_vk_print_gpu_info(i);
    }

    vk_instance.multi_device = host_buffer_import_ಎಣಿಕೆ > 1 && vk_instance.device_host_buffer_import[0];
}

ಶೂನ್ಯ ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx) {
    VK_LOG_DEBUG("ggml_vk_init(" << ctx->name << ", " << idx << ")");
    ggml_vk_instance_init();
    GGML_ASSERT(idx < vk_instance.device_indices.size());

    ctx->name = GGML_VK_NAME + std::to_string(idx);

    ctx->device = ggml_vk_get_device(idx);

    ctx->semaphore_idx = 0;
    ctx->event_idx = 0;

    ctx->prealloc_size_x = 0;
    ctx->prealloc_size_y = 0;
    ctx->prealloc_size_split_k = 0;
    // Fixed size of 1KB, for deterministic behavior
    ctx->prealloc_size_add_rms_partials = 1024;

    ctx->fence = ctx->device->device.createFence({});
    ctx->almost_ready_fence = ctx->device->device.createFence({});

    ctx->compute_cmd_pool.init(ctx->device, ctx->device->compute_queue.get());
    ಯದಿ (ctx->device->async_use_transfer_queue) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.setPNext(&tci);
        ctx->transfer_semaphore.s = ctx->device->device.createSemaphore(ci);
        ctx->transfer_semaphore.value = 0;

        ctx->transfer_cmd_pool.init(ctx->device, ctx->device->transfer_queue.get());
    }

    ಯದಿ (vk_perf_logger_enabled) {
        ctx->perf_logger = std::unique_ptr<vk_perf_logger>(ನೂತನ vk_perf_logger());
    }

#ifdef GGML_VULKAN_CHECK_RESULTS
    ಧ್ರುವ ವರ್ಣ* skip_checks = getenv("GGML_VULKAN_SKIP_CHECKS");
    vk_skip_checks = (skip_checks == ರಿಕ್ತ ? 0 : atoi(skip_checks));
    ಧ್ರುವ ವರ್ಣ* output_tensor = getenv("GGML_VULKAN_OUTPUT_TENSOR");
    vk_output_tensor = (output_tensor == ರಿಕ್ತ ? 0 : atoi(output_tensor));
#endif
}

vk_pipeline ggml_vk_get_to_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_F32:
        ಪಕ್ಷ GGML_TYPE_Q1_0:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
        ಪಕ್ಷ GGML_TYPE_Q2_K:
        ಪಕ್ಷ GGML_TYPE_Q3_K:
        ಪಕ್ಷ GGML_TYPE_Q4_K:
        ಪಕ್ಷ GGML_TYPE_Q5_K:
        ಪಕ್ಷ GGML_TYPE_Q6_K:
        ಪಕ್ಷ GGML_TYPE_IQ1_S:
        ಪಕ್ಷ GGML_TYPE_IQ1_M:
        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
        ಪಕ್ಷ GGML_TYPE_IQ2_S:
        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ3_S:
        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
        ಪಕ್ಷ GGML_TYPE_MXFP4:
        ಪಕ್ಷ GGML_TYPE_NVFP4:
        ಪಕ್ಷ GGML_TYPE_TQ1_0:
        ಪಕ್ಷ GGML_TYPE_TQ2_0:
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_dequant[type];
}

ಸ್ಥಿರ vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t ಅಂಕೆ_cols, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16 || b_type == GGML_TYPE_Q8_1);
    GGML_ASSERT(ಅಂಕೆ_cols >= 1 && ಅಂಕೆ_cols <= mul_mat_vec_max_cols);

    ಯದಿ (b_type == GGML_TYPE_Q8_1) {
        ವಿಕಲ್ಪ (a_type) {
            ಪಕ್ಷ GGML_TYPE_Q2_0:
            ಪಕ್ಷ GGML_TYPE_Q4_0:
            ಪಕ್ಷ GGML_TYPE_Q4_1:
            ಪಕ್ಷ GGML_TYPE_Q5_0:
            ಪಕ್ಷ GGML_TYPE_Q5_1:
            ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಪಕ್ಷ GGML_TYPE_MXFP4:
            ಪಕ್ಷ GGML_TYPE_Q2_K:
            ಪಕ್ಷ GGML_TYPE_Q3_K:
            ಪಕ್ಷ GGML_TYPE_Q4_K:
            ಪಕ್ಷ GGML_TYPE_Q5_K:
            ಪಕ್ಷ GGML_TYPE_Q6_K:
            ಪಕ್ಷ GGML_TYPE_IQ1_S:
            ಪಕ್ಷ GGML_TYPE_IQ1_M:
            ಪಕ್ಷ GGML_TYPE_IQ4_XS:
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    }

    ವಿಕಲ್ಪ (a_type) {
        ಪಕ್ಷ GGML_TYPE_F32:
        ಪಕ್ಷ GGML_TYPE_F16:
        ಪಕ್ಷ GGML_TYPE_BF16:
        ಪಕ್ಷ GGML_TYPE_Q1_0:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
        ಪಕ್ಷ GGML_TYPE_Q2_K:
        ಪಕ್ಷ GGML_TYPE_Q3_K:
        ಪಕ್ಷ GGML_TYPE_Q4_K:
        ಪಕ್ಷ GGML_TYPE_Q5_K:
        ಪಕ್ಷ GGML_TYPE_Q6_K:
        ಪಕ್ಷ GGML_TYPE_IQ1_S:
        ಪಕ್ಷ GGML_TYPE_IQ1_M:
        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
        ಪಕ್ಷ GGML_TYPE_IQ2_S:
        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ3_S:
        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
        ಪಕ್ಷ GGML_TYPE_MXFP4:
        ಪಕ್ಷ GGML_TYPE_NVFP4:
        ಪಕ್ಷ GGML_TYPE_TQ1_0:
        ಪಕ್ಷ GGML_TYPE_TQ2_0:
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
    ಯದಿ ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        ಯದಿ (a_type == GGML_TYPE_Q6_K) {
            ಯದಿ (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_SIZE_LARGE;
            }
        } ಅನ್ಯಥಾ {
            ಯದಿ (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_SIZE_LARGE;
            }
        }
    }

    ಯದಿ (b_type == GGML_TYPE_Q8_1) {
        ಯದಿ (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
        }
        ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][ಅಂಕೆ_cols-1];
    }

    ಪ್ರತ್ಯಾಗಮ b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][ಅಂಕೆ_cols-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][ಅಂಕೆ_cols-1];
}

ಸ್ಥಿರ vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec_id()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_Q8_1);

    ಯದಿ (b_type == GGML_TYPE_Q8_1) {
        ವಿಕಲ್ಪ (a_type) {
            ಪಕ್ಷ GGML_TYPE_Q2_0:
            ಪಕ್ಷ GGML_TYPE_Q4_0:
            ಪಕ್ಷ GGML_TYPE_Q4_1:
            ಪಕ್ಷ GGML_TYPE_Q5_0:
            ಪಕ್ಷ GGML_TYPE_Q5_1:
            ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಪಕ್ಷ GGML_TYPE_MXFP4:
            ಪಕ್ಷ GGML_TYPE_Q2_K:
            ಪಕ್ಷ GGML_TYPE_Q3_K:
            ಪಕ್ಷ GGML_TYPE_Q4_K:
            ಪಕ್ಷ GGML_TYPE_Q5_K:
            ಪಕ್ಷ GGML_TYPE_Q6_K:
            ಪಕ್ಷ GGML_TYPE_IQ1_S:
            ಪಕ್ಷ GGML_TYPE_IQ1_M:
            ಪಕ್ಷ GGML_TYPE_IQ4_XS:
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    }

    ವಿಕಲ್ಪ (a_type) {
        ಪಕ್ಷ GGML_TYPE_F32:
        ಪಕ್ಷ GGML_TYPE_F16:
        ಪಕ್ಷ GGML_TYPE_BF16:
        ಪಕ್ಷ GGML_TYPE_Q1_0:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
        ಪಕ್ಷ GGML_TYPE_Q2_K:
        ಪಕ್ಷ GGML_TYPE_Q3_K:
        ಪಕ್ಷ GGML_TYPE_Q4_K:
        ಪಕ್ಷ GGML_TYPE_Q5_K:
        ಪಕ್ಷ GGML_TYPE_Q6_K:
        ಪಕ್ಷ GGML_TYPE_IQ1_S:
        ಪಕ್ಷ GGML_TYPE_IQ1_M:
        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
        ಪಕ್ಷ GGML_TYPE_IQ2_S:
        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ3_S:
        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
        ಪಕ್ಷ GGML_TYPE_MXFP4:
        ಪಕ್ಷ GGML_TYPE_NVFP4:
        ಪಕ್ಷ GGML_TYPE_TQ1_0:
        ಪಕ್ಷ GGML_TYPE_TQ2_0:
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
    ಯದಿ ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        ಯದಿ (a_type == GGML_TYPE_Q6_K) {
            ಯದಿ (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_SIZE_LARGE;
            }
        } ಅನ್ಯಥಾ {
            ಯದಿ (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_SIZE_LARGE;
            }
        }
    }

    ಯದಿ (b_type == GGML_TYPE_Q8_1) {
        ಯದಿ (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
        }
        ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[dmmv_wg][a_type];
    }

    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_dequant_mul_mat_vec_id_f32[dmmv_wg][a_type];
}

vk_subbuffer ggml_vk_tensor_subbuffer(
    ಧ್ರುವ ggml_backend_vk_context * ctx, ಧ್ರುವ ggml_tensor * tensor, ತಾರ್ಕಿಕ allow_misalign) {

    vk_buffer buffer = nullptr;
    size_t offset = 0;
    ಯದಿ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, tensor->data, buffer, offset);
    }
    ಯದಿ (!buffer) {
        auto buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;
        buffer = buf_ctx->dev_buffer;
        offset = vk_tensor_offset(tensor) + tensor->view_offs;
    }
    GGML_ASSERT(buffer != nullptr);

    size_t size = ggml_nbytes(tensor);

    ಧ್ರುವ size_t descriptor_offset = ggml_vk_descriptor_offset(
        offset, ctx->device->properties.limits.minStorageBufferOffsetAlignment, ggml_type_size(tensor->type));
    ಧ್ರುವ size_t misalign_bytes = offset - descriptor_offset;
    // The shader must support misaligned offsets when indexing into the buffer
    GGML_ASSERT(allow_misalign || misalign_bytes == 0);
    offset = descriptor_offset;
    size += misalign_bytes;

    ಪ್ರತ್ಯಾಗಮ vk_subbuffer{buffer, offset, size};
}

ಸ್ಥಿರ vk_command_buffer* ggml_vk_get_or_create_cmd_buffer(vk_device& device, vk_command_pool& pool) {
    ಪ್ರತಿ (auto& cmd_buffer : pool.cmd_buffers) {
        ಯದಿ (!cmd_buffer.in_use) {
            cmd_buffer.use_counter++;
            cmd_buffer.in_use = ಸತ್ಯ;
            ಪ್ರತ್ಯಾಗಮ &cmd_buffer;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ggml_vk_create_cmd_buffer(device, pool);
}

ಸ್ಥಿರ vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, ತಾರ್ಕಿಕ one_time = ಸತ್ಯ) {
    vk_submission s;
    s.buffer = ggml_vk_get_or_create_cmd_buffer(device, p);
    ಯದಿ (one_time) {
        s.buffer->buf.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    } ಅನ್ಯಥಾ {
        s.buffer->buf.begin({ vk::CommandBufferUsageFlags{} });
    }

    ಪ್ರತ್ಯಾಗಮ s;
}

ಶೂನ್ಯ ggml_vk_cmd_label_begin(vk::CommandBuffer buf, ಧ್ರುವ ವರ್ಣ * name) {
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<ಭಿನ್ನ, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT(buf, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

ಶೂನ್ಯ ggml_vk_ctx_end(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    ಯದಿ (ctx->s == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    // close open labels so this buffer is balanced; reopened in ggml_vk_ctx_begin
    ಯದಿ (vk_instance.debug_utils_support) {
        ಪ್ರತಿ (size_t i = 0; i < ctx->debug_labels.size(); i++) {
            vk_instance.pfn_vkCmdEndDebugUtilsLabelEXT(ctx->s->buffer->buf);
        }
        // the enclosing per-command-buffer region
        vk_instance.pfn_vkCmdEndDebugUtilsLabelEXT(ctx->s->buffer->buf);
    }

    ctx->s->buffer->buf.end();
    ctx->s = nullptr;
}

ಶೂನ್ಯ ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    ಯದಿ (subctx->s != nullptr) {
        ggml_vk_ctx_end(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->p) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].data();

    ಯದಿ (vk_instance.debug_utils_support) {
        // outermost region, one per command buffer, so the gaps between submits stand out
        ಧ್ರುವ std::string name = "submit " + std::to_string(device->debug_cmdbuf_idx++);
        ggml_vk_cmd_label_begin(subctx->s->buffer->buf, name.c_str());

        // reopen labels left open when the previous command buffer was submitted
        ಪ್ರತಿ (ಧ್ರುವ std::string & label : subctx->debug_labels) {
            ggml_vk_cmd_label_begin(subctx->s->buffer->buf, label.c_str());
        }
    }
}

vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx) {
    vk_context result;
    ಯದಿ (!ctx->compute_ctx.expired()) {
        result = ctx->compute_ctx.lock();
    } ಅನ್ಯಥಾ {
        result = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);

        ctx->compute_ctx = result;
        ggml_vk_ctx_begin(ctx->device, result);
    }

    ಯದಿ (ctx->device->async_use_transfer_queue && ctx->transfer_semaphore_last_submitted < ctx->transfer_semaphore.value) {
        result->s->wait_semaphores.push_back(ctx->transfer_semaphore);
        ctx->transfer_semaphore_last_submitted = ctx->transfer_semaphore.value;
    }

    ಪ್ರತ್ಯಾಗಮ result;
}

vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx) {
    vk_context result;
    ಯದಿ (!ctx->transfer_ctx.expired()) {
        result = ctx->transfer_ctx.lock();
    } ಅನ್ಯಥಾ {
        result = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);

        ctx->transfer_ctx = result;
        ggml_vk_ctx_begin(ctx->device, result);
    }

    ಪ್ರತ್ಯಾಗಮ result;
}

ತಾರ್ಕಿಕ ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx) {
    ಯದಿ (!ctx->device->async_use_transfer_queue || ctx->transfer_ctx.expired()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    vk_context cpy_ctx = ctx->transfer_ctx.lock();
    ggml_vk_ctx_end(cpy_ctx);

    ಪ್ರತಿ (auto& cpy : cpy_ctx->in_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_semaphore.value++;
    cpy_ctx->seqs.back().back().signal_semaphores.push_back(ctx->transfer_semaphore);

    ggml_vk_submit(cpy_ctx, {});
    ctx->transfer_ctx.reset();
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

size_t ggml_vk_align_size(size_t width, size_t align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << width << ", " << align << ")");
    ಪ್ರತ್ಯಾಗಮ CEIL_DIV(width, align) * align;
}

ಶೂನ್ಯ deferred_memcpy(ಶೂನ್ಯ * dst, ಧ್ರುವ ಶೂನ್ಯ * src, size_t size, std::vector<vk_staging_memcpy>* memcpys) {
    ಯದಿ (memcpys == nullptr) {
        memcpy(dst, src, size);
    } ಅನ್ಯಥಾ {
        memcpys->emplace_back(dst, src, size);
    }
}

ಶೂನ್ಯ deferred_memset(ಶೂನ್ಯ * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets) {
    ಯದಿ (memsets == nullptr) {
        memset(dst, val, size);
    } ಅನ್ಯಥಾ {
        memsets->emplace_back(dst, val, size);
    }
}

ಸ್ಥಿರ uint32_t ggml_vk_guess_split_k(ggml_backend_vk_context * ctx, uint32_t m, uint32_t n, uint32_t k, ತಾರ್ಕಿಕ disable_split_k, ಧ್ರುವ vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_split_k << ")");

    ಯದಿ (disable_split_k) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    uint32_t split_k = 1;
    ಯದಿ (ctx->device->shader_core_ಎಣಿಕೆ != 0 && n >= pipeline->wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
        uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

        ಯದಿ (k >= 2048) {
            ಯದಿ (m_tiles * n_tiles <= ctx->device->shader_core_ಎಣಿಕೆ / 2) {
                split_k = ctx->device->shader_core_ಎಣಿಕೆ / (m_tiles * n_tiles);
            } ಅನ್ಯಥಾ ಯದಿ (m_tiles * n_tiles <= ctx->device->shader_core_ಎಣಿಕೆ * 2 / 3) {
                split_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            split_k = std::min(split_k, 8u);

            // ggml_vk_matmul will align the splits to be a multiple of 256.
            // If this rounded up size would cause the last split to be empty,
            // then reduce the split count.
            ಯಾವತ್ (ಸತ್ಯ) {
                ಯದಿ (split_k == 1) {
                    ಭಙ್ಗ;
                }
                uint32_t k_split = CEIL_DIV(k, split_k);
                k_split = ROUNDUP_POW2(k_split, 256);
                ಯದಿ (k_split * (split_k - 1) < k) {
                    ಭಙ್ಗ;
                }
                split_k--;
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ split_k;
}

ಶೂನ್ಯ ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& split_k_buffer,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t split_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
        uint32_t padded_n, uint64_t b_col_bytes, uint64_t d_col_bytes) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer << ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size << "), split_k: (" << (split_k_buffer.buffer != nullptr ? split_k_buffer.buffer->buffer : VK_NULL_HANDLE) << ", " << split_k_buffer.offset << ", " << split_k_buffer.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << split_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
    ಯದಿ (split_k == 1) {
        // a small tile can still produce more workgroups than the device allows on the n axis:
        // split the dispatch into n-chunks, shifting the B and D column windows
        ಧ್ರುವ uint64_t max_n = (uint64_t)ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1] * pipeline->wg_denoms[1];
        ಧ್ರುವ uint64_t n_chunks = CEIL_DIV(n, max_n);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, n_chunks * CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]));

        ಪ್ರತಿ (uint64_t n_base = 0; n_base < n; n_base += max_n) {
            ಧ್ರುವ uint32_t n_chunk = (uint32_t) std::min(n - n_base, max_n);
            GGML_ASSERT(n_base * b_col_bytes < b.size && n_base * d_col_bytes < d.size);
            GGML_ASSERT(b.size != VK_WHOLE_SIZE && d.size != VK_WHOLE_SIZE);
            vk_subbuffer b_chunk = b;
            vk_subbuffer d_chunk = d;
            b_chunk.offset += n_base * b_col_bytes;
            b_chunk.size   -= n_base * b_col_bytes;
            d_chunk.offset += n_base * d_col_bytes;
            d_chunk.size   -= n_base * d_col_bytes;

            uint32_t base_work_group_z = 0;
            ಯಾವತ್ (base_work_group_z < batch) {
                uint32_t groups_z = std::min(batch - base_work_group_z, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]);

                ಧ್ರುವ vk_mat_mat_ತಳ್ಳು_constants pc = { m, n_chunk, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, base_work_group_z, batch, k, ne02, ne12, broadcast2, broadcast3, uint32_t(padded_n - n_base) };
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b_chunk, d_chunk }, pc, { m, n_chunk, groups_z });
                base_work_group_z += groups_z;
            }
        }
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    GGML_ASSERT(batch_stride_d == m * n);

    // Round the split size up to a multiple of 256 (k-quant alignment)
    uint32_t k_split = CEIL_DIV(k, split_k);
    k_split = ROUNDUP_POW2(k_split, 256);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]));

    uint32_t base_work_group_z = 0;
    ಯಾವತ್ (base_work_group_z < batch) {
        uint32_t groups_z = std::min(batch - base_work_group_z, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]);

        ಧ್ರುವ vk_mat_mat_ತಳ್ಳು_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, base_work_group_z, batch, k_split, ne02, ne12, broadcast2, broadcast3, padded_n };
        // Make sure enough workgroups get assigned for split k to work
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, split_k_buffer }, pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * split_k, n, groups_z });
        base_work_group_z += groups_z;
    }
    ggml_vk_sync_buffers(ctx, subctx);
    ಧ್ರುವ std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), split_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_split_k_reduce, { split_k_buffer, d }, pc2, { m * n * batch, 1, 1 });
    ctx->prealloc_split_k_need_sync = ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_get_mul_mat_mat_f16acc(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    ಯದಿ (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (src1_type == GGML_TYPE_Q8_1) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (src0_type == GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support);
    }
    // quant types
    ಯದಿ (ctx->device->coopmat2) {
        ಪ್ರತ್ಯಾಗಮ prec == GGML_PREC_DEFAULT;
    }
    ಯದಿ (ctx->device->coopmat_support) {
        ಪ್ರತ್ಯಾಗಮ ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT;
    }
    ಪ್ರತ್ಯಾಗಮ ctx->device->fp16 && prec == GGML_PREC_DEFAULT;
}

ಸ್ಥಿರ ಧ್ರುವ std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec, ತಾರ್ಕಿಕ mul_mat_id = ಅಸತ್ಯ) {
    ತಾರ್ಕಿಕ f16acc = ggml_vk_get_mul_mat_mat_f16acc(ctx, src0_type, src1_type, prec);
    vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ{src0_type, src1_type, mul_mat_id, f16acc};
    auto it = ctx->device->pipeline_matmul.find(ಕೀಲಿ);
    ಯದಿ (it == ctx->device->pipeline_matmul.end() || it->second.empty()) {
        // Try without f16acc
        ಯದಿ (f16acc) {
            ಕೀಲಿ.f16acc = ಅಸತ್ಯ;
            it = ctx->device->pipeline_matmul.find(ಕೀಲಿ);
            ಯದಿ (it != ctx->device->pipeline_matmul.end() && !it->second.empty()) ಪ್ರತ್ಯಾಗಮ &it->second;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪ್ರತ್ಯಾಗಮ &it->second;
}

ಸ್ಥಿರ vk_pipeline ggml_vk_guess_matmul_pipeline_map(ggml_backend_vk_context * ctx,
        ಧ್ರುವ std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, ತಾರ್ಕಿಕ aligned, ತಾರ್ಕಿಕ mul_mat_id) {
    auto& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_ಎಣಿಕೆ, configs);
    ಯದಿ (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    ಪ್ರತ್ಯಾಗಮ (aligned && configs[idx].aligned) ? configs[idx].aligned : configs[idx].unaligned;
}

ಸ್ಥಿರ uint32_t ggml_vk_guess_matmul_pipeline_align_map(ggml_backend_vk_context * ctx,
        ಧ್ರುವ std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, ತಾರ್ಕಿಕ mul_mat_id) {
    auto& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_ಎಣಿಕೆ, configs);
    ಯದಿ (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    ಪ್ರತ್ಯಾಗಮ configs[idx].align;
}

ಸ್ಥಿರ uint64_t ggml_vk_mul_mat_id_max_tiles(uint64_t n_experts, uint64_t n_rows, uint32_t bn) {
    // Each nonempty expert can add at most one partial tile to the combined grid.
    ಪ್ರತ್ಯಾಗಮ std::min(n_rows, CEIL_DIV(n_rows, bn) + std::min(n_experts, n_rows) - 1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids, ಧ್ರುವ vk_subbuffer & expert_ಎಣಿಕೆ_buf,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
        uint32_t row_ids_offset, uint32_t max_tiles) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer << ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size << "), ids: (" << ids.buffer->buffer << ", " << ids.offset << ", " << ids.size << "), expert_count: (" << expert_ಎಣಿಕೆ_buf.buffer->buffer << ", " << expert_ಎಣಿಕೆ_buf.offset << ", " << expert_ಎಣಿಕೆ_buf.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    ಧ್ರುವ vk_mat_mat_id_ತಳ್ಳು_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11, n_as, row_ids_offset };
    ಧ್ರುವ uint32_t tiles_y = std::min(max_tiles, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
    ಧ್ರುವ std::array<uint32_t, 3> elements = { m, tiles_y * pipeline->wg_denoms[1], uint32_t(CEIL_DIV(max_tiles, tiles_y)) };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids, expert_ಎಣಿಕೆ_buf }, pc, elements);
}

ತಾರ್ಕಿಕ ggml_vk_dim01_contiguous(ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_size(tensor->type) &&
        (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2]*tensor->ne[2]);
}

// Batch stride in elements of a tensor read in place.
ಸ್ಥಿರ uint32_t ggml_vk_batch_stride(ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ (uint32_t)(tensor->nb[2] / ggml_type_size(tensor->type) * ggml_blck_size(tensor->type));
}

vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ಧ್ರುವ ggml_tensor * src, ಧ್ರುವ ggml_tensor * dst, ggml_type to) {

    // Choose "contiguous copy" shader if src/dst are contiguous
    ತಾರ್ಕಿಕ contig = ggml_is_contiguous(src) && (!dst || ggml_is_contiguous(dst));

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    ತಾರ್ಕಿಕ transpose = dst && src->nb[1] == ggml_type_size(to) && ggml_are_same_shape(dst, src);

    ಯದಿ (transpose && src->type == to) {
        ಯದಿ (ggml_type_size(to) == 4) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_transpose_32;
        } ಅನ್ಯಥಾ ಯದಿ (ggml_type_size(to) == 2) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_transpose_16;
        }
    }

    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
    ತಾರ್ಕಿಕ transpose02 = dst && !contig && src->nb[2] == ggml_type_size(to) &&
                       ggml_is_contiguous(dst) && ggml_are_same_shape(dst, src);

    ಯದಿ (transpose02 && src->type == to) {
        ಯದಿ (ggml_type_size(to) == 4) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_transpose_02_32;
        } ಅನ್ಯಥಾ ಯದಿ (ggml_type_size(to) == 2) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_transpose_02_16;
        }
    }

    ಯದಿ (src->type == GGML_TYPE_F32 && to == GGML_TYPE_F16) {
        ಯದಿ (contig) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_contig_cpy_f32_f16;
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_f32_f16;
        }
    }
    ಯದಿ (src->type == GGML_TYPE_F16 && to == GGML_TYPE_F32) {
        ಯದಿ (contig) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_contig_cpy_f16_f32;
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_f16_f32;
        }
    }
    ಯದಿ (src->type == GGML_TYPE_F32 && to == GGML_TYPE_BF16) {
        ಯದಿ (contig) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_contig_cpy_f32_bf16;
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_f32_bf16;
        }
    }
    ಯದಿ (src->type == GGML_TYPE_BF16 && to == GGML_TYPE_F32) {
        ಯದಿ (contig) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_contig_cpy_bf16_f32;
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_bf16_f32;
        }
    }
    ಯದಿ (src->type == GGML_TYPE_BF16 && to == GGML_TYPE_F16) {
        ಯದಿ (contig) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_contig_cpy_bf16_f16;
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_bf16_f16;
        }
    }
    ಯದಿ (src->type == GGML_TYPE_F32 && to == GGML_TYPE_I32) {
        ಯದಿ (contig) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_contig_cpy_f32_i32;
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_f32_i32;
        }
    }
    ಯದಿ (src->type == GGML_TYPE_I32 && to == GGML_TYPE_F32) {
        ಯದಿ (contig) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_contig_cpy_i32_f32;
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_i32_f32;
        }
    }
    ಯದಿ (src->type == GGML_TYPE_F32) {
        ವಿಕಲ್ಪ (to) {
        ಪಕ್ಷ GGML_TYPE_Q1_0:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_f32_quant[to];
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
        }
    }

    ಯದಿ (to == GGML_TYPE_F32) {
        ವಿಕಲ್ಪ (src->type) {
        ಪಕ್ಷ GGML_TYPE_Q1_0:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_quant_f32[src->type];
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
        }
    }

    ಯದಿ (src->type == to) {
        // Copy two or four bytes at a time, depending on block size.
        // For quantized types, we scale by block size/type size. But
        // this path is also used for bf16->bf16 for example, where the
        // type size must be exactly 2 or 4.
        GGML_ASSERT(ggml_is_quantized(to) || ggml_type_size(src->type) == 2 || ggml_type_size(src->type) == 4);
        ಯದಿ ((ggml_type_size(src->type) % 4) == 0) {
            ಯದಿ (contig) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_contig_cpy_32;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_32;
            }
        } ಅನ್ಯಥಾ {
            ಯದಿ (contig) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_contig_cpy_16;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cpy_16;
            }
        }
    }

    std::cerr << "Missing CPY op for types: " << ggml_type_name(src->type) << " " << ggml_type_name(to) << std::endl;
    GGML_ABORT("fatal error");
}

ಶೂನ್ಯ ggml_vk_cpy_to_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ vk_subbuffer & in, ಧ್ರುವ vk_subbuffer & out) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ")");

    ಧ್ರುವ uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    ಯದಿ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } ಅನ್ಯಥಾ ಯದಿ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } ಅನ್ಯಥಾ {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_ತಳ್ಳು_constants pc = vk_op_unary_ತಳ್ಳು_constants_init(tensor, tensor, ne);
    pc.nb10 = 1;
    pc.nb11 = (uint32_t)tensor->ne[0];
    pc.nb12 = (uint32_t)(tensor->ne[0] * tensor->ne[1]);
    pc.nb13 = (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]);
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_cpy_to_strided(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ಧ್ರುವ ggml_tensor * tensor,
        ಧ್ರುವ vk_subbuffer & in, ಧ್ರುವ vk_subbuffer & out,
        uint32_t nb10, uint32_t nb11, uint32_t nb12, uint32_t nb13) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_strided((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "dst_nb=(" << nb10 << ", " << nb11 << ", " << nb12 << ", " << nb13 << "), buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ")");

    ಧ್ರುವ uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    ಯದಿ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } ಅನ್ಯಥಾ ಯದಿ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } ಅನ್ಯಥಾ {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_ತಳ್ಳು_constants pc = vk_op_unary_ತಳ್ಳು_constants_init(tensor, tensor, ne);
    pc.nb10 = nb10;
    pc.nb11 = nb11;
    pc.nb12 = nb12;
    pc.nb13 = nb13;
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type) {
    ವಿಕಲ್ಪ(type) {
        ಪಕ್ಷ GGML_TYPE_Q8_1:
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_quantize_q8_1_x4;
        ಸ್ವಭಾವ:
            std::cerr << "Missing quantize pipeline for type: " << ggml_type_name(type) << std::endl;
            GGML_ABORT("fatal error");
    }
}

ಶೂನ್ಯ ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ vk_subbuffer & in, ಧ್ರುವ vk_subbuffer & out, uint32_t ne) {
    VK_LOG_DEBUG("ggml_vk_quantize_q8_1(" << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ", " << ne << ")");

    vk_pipeline pipeline = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);

    ಧ್ರುವ uint32_t ಅಂಕೆ_blocks = CEIL_DIV(ne, pipeline->wg_denoms[0]);
    // clamp the number of elements to the max workgroup count. The shader will iterate over the total number of blocks.
    ಧ್ರುವ uint64_t max_elements = std::min<uint64_t>(uint64_t{ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[0]} * pipeline->wg_denoms[0], std::numeric_limits<uint32_t>::max());
    ಧ್ರುವ uint32_t elements = std::min(ne, static_cast<uint32_t>(max_elements));

    ಧ್ರುವ vk_quantize_q8_1_ತಳ್ಳು_constants pc = {
        ne,
        ಅಂಕೆ_blocks,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, pc, { elements, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}

ಸ್ಥಿರ vk_pipeline ggml_vk_get_64b_indexing_pipeline(ggml_backend_vk_context * ctx, vk_pipeline &pipeline) {
    GGML_UNUSED(ctx);
#ಯದಿ defined(VK_EXT_shader_64bit_indexing)
    vk_pipeline *ptr = &pipeline;
    ಯಾವತ್ (*ptr) {
        ಯದಿ ((*ptr)->is_64b_indexing) {
            ಪ್ರತ್ಯಾಗಮ *ptr;
        }
        ptr = &(*ptr)->next;
    }
#endif
    ಪ್ರತ್ಯಾಗಮ pipeline;
}

// perf-heuristic for choosing mul_mat b type, based on shader availability
ಸ್ಥಿರ ggml_type ggml_vk_mul_mat_b_type(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ಧ್ರುವ ggml_tensor * src1,
        ggml_type f16_type, ggml_prec prec, ತಾರ್ಕಿಕ mul_mat_id, ತಾರ್ಕಿಕ can_quantize_y, ತಾರ್ಕಿಕ prefer_f16_b) {
    auto available = [&](ggml_type b) {
        ಪ್ರತ್ಯಾಗಮ ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0_type, b, prec, mul_mat_id) != nullptr;
    };
    ಯದಿ (can_quantize_y && available(GGML_TYPE_Q8_1))             ಪ್ರತ್ಯಾಗಮ GGML_TYPE_Q8_1;
    ಯದಿ (prefer_f16_b   && available(f16_type))                   ಪ್ರತ್ಯಾಗಮ f16_type;
    ಯದಿ (ggml_vk_dim01_contiguous(src1) && available(src1->type)) ಪ್ರತ್ಯಾಗಮ src1->type;
    ಯದಿ (src1->type == GGML_TYPE_BF16 && available(GGML_TYPE_F32)) ಪ್ರತ್ಯಾಗಮ GGML_TYPE_F32;
    ಪ್ರತ್ಯಾಗಮ f16_type;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst, ತಾರ್ಕಿಕ disable_split_k) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << ggml_type_name(src0->type) << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << ggml_type_name(src1->type) << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << ggml_type_name(dst->type) << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ಧ್ರುವ uint64_t ne00 = src0->ne[0];
    ಧ್ರುವ uint64_t ne01 = src0->ne[1];
    ಧ್ರುವ uint64_t ne02 = src0->ne[2];
    ಧ್ರುವ uint64_t ne03 = src0->ne[3];

    ಧ್ರುವ uint64_t ne10 = src1->ne[0];
    ಧ್ರುವ uint64_t ne11 = src1->ne[1];
    ಧ್ರುವ uint64_t ne12 = src1->ne[2];
    ಧ್ರುವ uint64_t ne13 = src1->ne[3];

    ಧ್ರುವ uint64_t ne21 = dst->ne[1];
    ಧ್ರುವ uint32_t stride_d = dst->nb[1] / ggml_type_size(dst->type);
    ಧ್ರುವ uint32_t stride_batch_d = stride_d*ne21;

    ಧ್ರುವ uint64_t r2 = ne12 / ne02;
    ಧ್ರುವ uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *)src1->buffer->context;

    vk_buffer d_Qx = nullptr;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;

    ತಾರ್ಕಿಕ src0_uma = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ src1_uma = ಅಸತ್ಯ;

    ಯದಿ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    ಧ್ರುವ ತಾರ್ಕಿಕ is_coopmat = ctx->device->coopmat_support || ctx->device->coopmat2;
    ಧ್ರುವ ತಾರ್ಕಿಕ can_quantize_y = (ctx->device->integer_dot_product || ctx->device->coopmat_int_support) &&
                                src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // coopmat shaders are fast enough that memory bandwidth reduction from f16 activations helps
    ಧ್ರುವ ತಾರ್ಕಿಕ prefer_f16_b = is_coopmat && ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ಧ್ರುವ ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], ಅಸತ್ಯ, can_quantize_y, prefer_f16_b);

    ಧ್ರುವ ತಾರ್ಕಿಕ quantize_y       = y_kernel_type == GGML_TYPE_Q8_1;
    ಧ್ರುವ ತಾರ್ಕಿಕ y_f32_kernel     = y_kernel_type == GGML_TYPE_F32;
    ಧ್ರುವ ತಾರ್ಕಿಕ y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type);

    ಧ್ರುವ std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0]);

    ಧ್ರುವ ತಾರ್ಕಿಕ qx_needs_dequant = mmp_map == nullptr || x_non_contig;
    ಧ್ರುವ ತಾರ್ಕಿಕ qy_needs_dequant = y_needs_reformat;

    ಯದಿ (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0]);
    }

    GGML_ASSERT(mmp_map != nullptr);

    ಧ್ರುವ uint32_t kpad = quantize_y ? 0 : ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, ne11, ಅಸತ್ಯ));
    ಧ್ರುವ ತಾರ್ಕಿಕ aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, ne11, aligned, ಅಸತ್ಯ);

    ಯದಿ (ggml_nbytes(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) : ne11;
    ಧ್ರುವ uint64_t x_ne = ggml_nelements(src0);
    // 128 elements per Q8_1 x4 block
    ಧ್ರುವ uint64_t y_ne = padded_n * ne10 * ne12 * ne13;
    ಧ್ರುವ uint64_t d_ne = ggml_nelements(dst);

    ಧ್ರುವ uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);

    ಧ್ರುವ uint64_t qx_sz = ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type);
    ಧ್ರುವ uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    ಧ್ರುವ uint64_t x_sz = !qx_needs_dequant ? qx_sz : ಪರಿಮಾಣ(ggml_fp16_t) * x_ne;
    ಧ್ರುವ uint64_t y_sz = quantize_y ? (ggml_vk_align_size(y_ne, 128) * ggml_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1)) : (y_f32_kernel ? ಪರಿಮಾಣ(ಭಿನ್ನ) * y_ne : ಪರಿಮಾಣ(ggml_fp16_t) * y_ne);
    ಧ್ರುವ uint64_t d_sz = ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne;

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    vk_pipeline to_q8_1 = nullptr;

    ಯದಿ (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type);
    } ಅನ್ಯಥಾ {
        to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type);
    }
    ಯದಿ (y_needs_reformat) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, y_kernel_type);
    } ಅನ್ಯಥಾ {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    ಯದಿ (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    {
        ಧ್ರುವ uint64_t split_k_size = split_k > 1 ? d_sz * split_k : 0;
        ಯದಿ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.maxStorageBufferRange) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.maxStorageBufferRange) ||
                (split_k > 1 && split_k_size > ctx->device->properties.limits.maxStorageBufferRange)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ಯದಿ (qx_needs_dequant && ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz) {
            ctx->prealloc_size_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ (split_k > 1 && ctx->prealloc_size_split_k < split_k_size) {
            ctx->prealloc_size_split_k = split_k_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ಯದಿ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        ಯದಿ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        ಯದಿ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
        ಯದಿ (split_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, 1);
        }
    }

    vk_buffer d_D = dst_buf_ctx->dev_buffer;
    ಧ್ರುವ uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    GGML_ASSERT(d_D->size >= d_buf_offset + d_sz);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    ಯದಿ (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_buffer;
        qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    ಯದಿ (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_buffer;
        qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    ಯದಿ (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } ಅನ್ಯಥಾ {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ಯದಿ (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } ಅನ್ಯಥಾ ಯದಿ (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } ಅನ್ಯಥಾ {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    ಯದಿ (x_non_contig || qx_needs_dequant) {
        ಯದಿ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ಯದಿ (x_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_offset), ggml_vk_subbuffer(ctx, d_X, 0));
    } ಅನ್ಯಥಾ ಯದಿ (qx_needs_dequant) {
        ಧ್ರುವ std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0, { vk_subbuffer{ d_Qx, qx_buf_offset, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)(x_ne), 1, 1});
        ggml_vk_sync_buffers(ctx, subctx);
    }
    ಯದಿ (y_needs_reformat) {
        ಯದಿ (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            ಯದಿ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0));
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;
        }
    }
    ಯದಿ (quantize_y) {
        ಯದಿ (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            ಯದಿ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;
        }
    }

    // The partial tiles of a quantized A rely on the bound range to read zeros past the last row,
    // so the range stays exact: the strided extent of a tensor read in place, the staged size otherwise.
    ಧ್ರುವ uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ಧ್ರುವ uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    ಯದಿ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    ಯದಿ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    // column stride in bytes of the B buffer: raw src1 keeps its own stride, staging buffers are tightly packed
    ಧ್ರುವ uint64_t b_col_bytes = (y_needs_reformat || qy_needs_dequant || quantize_y)
        ? (quantize_y ? (uint64_t)(ne10 / ggml_blck_size(GGML_TYPE_Q8_1)) * ggml_type_size(GGML_TYPE_Q8_1) : (uint64_t)ne10 * ಪರಿಮಾಣ(ggml_fp16_t))
        : (uint64_t)src1->nb[1];

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_offset, x_range }, { d_Y, y_buf_offset, y_range },
        ggml_vk_subbuffer(ctx, d_D, d_buf_offset), { ctx->prealloc_split_k, 0, d_sz * split_k },
        ne01, ne11, ne10,
        ne10, ne10, stride_d, stride_batch_x, stride_batch_y, stride_batch_d,
        split_k, ne12*ne13, ne02, ne12, r2, r3, padded_n,
        b_col_bytes, (uint64_t)dst->nb[1]
    );  // NOLINT

    ಯದಿ (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = ಸತ್ಯ;
    }
    ಯದಿ (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = ಸತ್ಯ;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_should_use_mmvq(ಧ್ರುವ vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
    ಯದಿ (device->mmvq_mode == 1) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    } ಅನ್ಯಥಾ ಯದಿ (device->mmvq_mode == -1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // q6_k only has 2-byte alignment which makes it somewhat problematic,
    // using MMVQ is only a win on Intel.
    ತಾರ್ಕಿಕ mmvq_q6 = device->vendor_id == VK_VENDOR_ID_INTEL;
    ಯದಿ (src0_type == GGML_TYPE_Q6_K && !mmvq_q6) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // MMVQ is generally good for batches
    ಯದಿ (n > 1) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    // Quantization overhead is not worth it for small k
    ವಿಕಲ್ಪ (device->vendor_id) {
    ಪಕ್ಷ VK_VENDOR_ID_NVIDIA:
        ಯದಿ (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_IQ1_S || src0_type == GGML_TYPE_IQ1_M) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (k <= 4096) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ವಿಕಲ್ಪ (src0_type) {
        ಪಕ್ಷ GGML_TYPE_MXFP4:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಪ್ರತ್ಯಾಗಮ device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    ಪಕ್ಷ VK_VENDOR_ID_AMD:
        ಯದಿ (k < 2048) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ವಿಕಲ್ಪ (src0_type) {
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            ಪ್ರತ್ಯಾಗಮ device->architecture == vk_device_architecture::AMD_GCN;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    ಪಕ್ಷ VK_VENDOR_ID_INTEL:
        ಯದಿ (device->architecture == vk_device_architecture::INTEL_XE2) {
            ಯದಿ (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_Q6_K) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }

        ಯದಿ (device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            // Intel Windows proprietary driver MMVQ performance for !Q2/Q3/Q6 is worse than fp16,
            // see https://github.com/ggml-org/llama.cpp/issues/17628 and
            // https://github.com/ggml-org/llama.cpp/pull/23056
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (k < 2048) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ವಿಕಲ್ಪ (src0_type) {
        // From tests on A770 Linux, may need more tuning
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    ಪಕ್ಷ VK_VENDOR_ID_QUALCOMM:
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    GGML_UNUSED(m);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_vec_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ತಾರ್ಕಿಕ swap_inputs = ಅಸತ್ಯ) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ಧ್ರುವ ggml_tensor * src0 = dst->src[swap_inputs ? 1 : 0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[swap_inputs ? 0 : 1];

    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << ")),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ಧ್ರುವ uint64_t ne00 = src0->ne[0];
    ಧ್ರುವ uint64_t ne01 = src0->ne[1];
    ಧ್ರುವ uint64_t ne02 = src0->ne[2];
    ಧ್ರುವ uint64_t ne03 = src0->ne[3];

    ಧ್ರುವ uint64_t ne10 = src1->ne[0];
    ಧ್ರುವ uint64_t ne11 = src1->ne[1];
    ಧ್ರುವ uint64_t ne12 = src1->ne[2];
    ಧ್ರುವ uint64_t ne13 = src1->ne[3];

    ಧ್ರುವ uint64_t ne20 = dst->ne[swap_inputs ? 1 : 0];
    ಧ್ರುವ uint64_t ne21 = dst->ne[swap_inputs ? 0 : 1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ಧ್ರುವ uint64_t r2 = ne12 / ne02;
    ಧ್ರುವ uint64_t r3 = ne13 / ne03;

    // batch_n indicates that we need to compute a few vector results, and this assumes
    // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
    GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
    ತಾರ್ಕಿಕ batch_n = ne11 > 1;

    ಧ್ರುವ ತಾರ್ಕಿಕ x_non_contig = !ggml_vk_dim01_contiguous(src0);

    // BF16 src1 has no mul_mat_vec B-type; widen to F32
    ಧ್ರುವ ತಾರ್ಕಿಕ widen_bf16 = src1->type == GGML_TYPE_BF16;
    ಧ್ರುವ ತಾರ್ಕಿಕ y_non_contig = !ggml_vk_dim01_contiguous(src1) || widen_bf16;

    ಧ್ರುವ ggml_type y_kernel_type = widen_bf16 ? GGML_TYPE_F32 : src1->type;
    ಧ್ರುವ ತಾರ್ಕಿಕ f16_f32_kernel = y_kernel_type == GGML_TYPE_F32;
    ತಾರ್ಕಿಕ quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    ಯದಿ (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type);
    }
    ಯದಿ (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, y_kernel_type);
    } ಅನ್ಯಥಾ {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, GGML_TYPE_Q8_1, ne11, ne20, ne00) : nullptr;
    vk_pipeline to_q8_1 = nullptr;

    ಯದಿ (dmmv == nullptr) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, y_kernel_type, ne11, ne20, ne00);
        quantize_y = ಅಸತ್ಯ;
    }

    ಯದಿ (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    ಯದಿ (ggml_nbytes(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ qx_needs_dequant = x_non_contig;
    ಧ್ರುವ ತಾರ್ಕಿಕ qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    ಧ್ರುವ uint64_t x_ne = ggml_nelements(src0);
    ಧ್ರುವ uint64_t y_ne = ggml_nelements(src1);

    ಧ್ರುವ uint64_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    ಧ್ರುವ uint64_t x_sz = x_non_contig ? ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) : qx_sz;
    ಧ್ರುವ uint64_t y_sz = quantize_y ? (ggml_vk_align_size(y_ne, 128) * ggml_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1)) :
                         (f16_f32_kernel ? ಪರಿಮಾಣ(ಭಿನ್ನ) * y_ne : ಪರಿಮಾಣ(ggml_fp16_t) * y_ne);

    {
        ಯದಿ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.maxStorageBufferRange) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.maxStorageBufferRange)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ಯದಿ (qx_needs_dequant && ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz) {
            ctx->prealloc_size_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ಯದಿ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        ಯದಿ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        ಯದಿ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_X, d_Y;

    ಯದಿ (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } ಅನ್ಯಥಾ {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ಯದಿ (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } ಅನ್ಯಥಾ {
        d_Y = d_Qy;
    }

    ಯದಿ (x_non_contig) {
        ಯದಿ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        GGML_ASSERT(x_sz == ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment));
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, d_Qx, d_X);
    }
    ಯದಿ (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_size(y_kernel_type) * y_ne);
        ಯದಿ (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            ಯದಿ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;
        }
    }
    ಯದಿ (quantize_y) {
        ಯದಿ (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            ಯದಿ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;
        }
    }

    // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
    uint32_t stride_batch_x = batch_n ? 0 : (qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0));
    uint32_t stride_batch_y = batch_n ? ne10 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));
    uint32_t stride_batch_d = batch_n ? ne20 : (ne20*ne21);

    ಯದಿ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    ಯದಿ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    ಧ್ರುವ uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    ಯದಿ (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    vk_subbuffer d_F0 = d_D;
    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
        ಧ್ರುವ ggml_tensor * add = cgraph->nodes[ಗಂಟು_idx + 1];
        ಧ್ರುವ ggml_tensor * bias = add->src[0] == dst ? add->src[1] : add->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops == 2) {
        ಧ್ರುವ ggml_tensor * add = cgraph->nodes[ಗಂಟು_idx + 2];
        ಧ್ರುವ ggml_tensor * bias = add->src[0] == cgraph->nodes[ಗಂಟು_idx + 1] ? add->src[1] : add->src[0];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    ggml_pipeline_request_descriptor_sets(ctx, dmmv, CEIL_DIV(ne12 * ne13, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]));

    uint32_t base_work_group_y = 0;
    ಯಾವತ್ (base_work_group_y < ne12 * ne13) {

        uint32_t groups_y = std::min((uint32_t)(ne12 * ne13) - base_work_group_y, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
        ಧ್ರುವ vk_mat_vec_ತಳ್ಳು_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, stride_batch_d,
            fusion_flags, base_work_group_y,
            (uint32_t)ne02, (uint32_t)ne12, (uint32_t)r2, (uint32_t)r3,
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
                                  {
                                    d_X,
                                    d_Y,
                                    d_D,
                                    d_F0,
                                    d_F1,
                                  },
                                  pc, { groups_x, groups_y, groups_z });
        base_work_group_y += groups_y;
    }

    ಯದಿ (x_non_contig) {
        ctx->prealloc_x_need_sync = ಸತ್ಯ;
    }
    ಯದಿ (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = ಸತ್ಯ;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ಧ್ರುವ uint64_t ne00 = src0->ne[0];
    ಧ್ರುವ uint64_t ne01 = src0->ne[1];
    ಧ್ರುವ uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    //const uint64_t ne10 = src1->ne[0];
    ಧ್ರುವ uint64_t ne11 = src1->ne[1];
    ಧ್ರುವ uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    uint32_t gqa_ratio = (uint32_t)ne12 / (uint32_t)ne02;
    ಯದಿ (gqa_ratio > 8 || gqa_ratio == 0 || ne12 != ne02 * gqa_ratio) {
        gqa_ratio = 1;
    }

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ratio - 1];

    ಯದಿ (ggml_nbytes(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops], ಸತ್ಯ);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, ಸತ್ಯ);

    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
        ಧ್ರುವ ggml_tensor * add = cgraph->nodes[ಗಂಟು_idx + 1];
        ಧ್ರುವ ggml_tensor * bias = add->src[0] == dst ? add->src[1] : add->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops > 1) {
        ಧ್ರುವ ggml_tensor * bias = cgraph->nodes[ಗಂಟು_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute

    vk_mat_vec_p021_ತಳ್ಳು_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
        0, 0, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, nullptr, nullptr, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops]);

    uint32_t workgroups_z = (uint32_t)ne12;
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    ಯದಿ (gqa_ratio > 1) {
        workgroups_z /= gqa_ratio;
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { 1, (uint32_t)ne01, workgroups_z });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ಧ್ರುವ uint64_t ne00 = src0->ne[0];
    ಧ್ರುವ uint64_t ne01 = src0->ne[1];
    ಧ್ರುವ uint64_t ne02 = src0->ne[2];
    ಧ್ರುವ uint64_t ne03 = src0->ne[3];

    ಧ್ರುವ uint64_t nb01 = src0->nb[1];
    ಧ್ರುವ uint64_t nb02 = src0->nb[2];

    ಧ್ರುವ uint64_t nb12 = src1->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    ಧ್ರುವ uint64_t ne11 = src1->ne[1];
    ಧ್ರುವ uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ಧ್ರುವ uint32_t nb03 = (uint32_t)(src0->nb[3] / ಪರಿಮಾಣ(ggml_fp16_t));
    ಧ್ರುವ uint32_t nb13 = (uint32_t)(src1->nb[3] / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ uint32_t nb23 = (uint32_t)(dst->nb[3] / ಪರಿಮಾಣ(ಭಿನ್ನ));

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

    ಧ್ರುವ uint32_t row_stride_x = nb01 / ಪರಿಮಾಣ(ggml_fp16_t);
    ಧ್ರುವ uint32_t channel_stride_x = nb02 / ಪರಿಮಾಣ(ggml_fp16_t);
    ಧ್ರುವ uint32_t channel_stride_y = nb12 / ಪರಿಮಾಣ(ಭಿನ್ನ);

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_nc_f16_f32;
    ಯದಿ (ggml_nbytes(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops], ಸತ್ಯ);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, ಸತ್ಯ);
    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
        ಧ್ರುವ ggml_tensor * add = cgraph->nodes[ಗಂಟು_idx + 1];
        ಧ್ರುವ ggml_tensor * bias = add->src[0] == dst ? add->src[1] : add->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops > 1) {
        ಧ್ರುವ ggml_tensor * bias = cgraph->nodes[ಗಂಟು_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute
    vk_mat_vec_nc_ತಳ್ಳು_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01,
        row_stride_x, channel_stride_x, channel_stride_y,
        (uint32_t)(ne12 / ne02), (uint32_t)ne12,
        0, 0,
        nb03, nb13, nb23, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, nullptr, nullptr, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops]);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { (uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12 });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_vk_fwht_pipeline_idx(int64_t n) {
    ವಿಕಲ್ಪ (n) {
        ಪಕ್ಷ 64:  ಪ್ರತ್ಯಾಗಮ 0;
        ಪಕ್ಷ 128: ಪ್ರತ್ಯಾಗಮ 1;
        ಪಕ್ಷ 256: ಪ್ರತ್ಯಾಗಮ 2;
        ಪಕ್ಷ 512: ಪ್ರತ್ಯಾಗಮ 3;
        ಸ್ವಭಾವ:  ಪ್ರತ್ಯಾಗಮ -1;
    }
}

ತಾರ್ಕಿಕ ggml_vk_can_use_fwht(ಧ್ರುವ ggml_backend_vk_context * ctx, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * dst) {
    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx = ggml_vk_fwht_pipeline_idx(src1->ne[0]);
    ಯದಿ (idx < 0 || ctx->device->pipeline_fwht_f32[idx] == nullptr) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous(src1)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    GGML_ASSERT(ggml_is_contiguous(dst));

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಶೂನ್ಯ ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx = ggml_vk_fwht_pipeline_idx(src->ne[0]);
    vk_pipeline pipeline = ctx->device->pipeline_fwht_f32[idx];

    ಧ್ರುವ uint32_t rows_per_workgroup = 4;
    ಧ್ರುವ uint32_t n_rows = (uint32_t)ggml_nrows(src);
    ಧ್ರುವ uint32_t max_workgroups_x = ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[0];

    ಧ್ರುವ uint32_t total_workgroups = CEIL_DIV(n_rows, rows_per_workgroup);
    ಧ್ರುವ uint32_t workgroups_x = std::min(total_workgroups, max_workgroups_x);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಧ್ರುವ vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src, ಸತ್ಯ);
    ಧ್ರುವ vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ಸತ್ಯ);

    vk_op_fwht_ತಳ್ಳು_constants pc = {
        n_rows,
        0,
        0,
        1.0f / std::sqrt((ಭಿನ್ನ)src->ne[0]),
    };
    init_pushconst_tensor_offsets(ctx, pc, src, nullptr, nullptr, nullptr, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });
}

ಸ್ಥಿರ uint32_t ggml_vk_nb_elem(ಧ್ರುವ ggml_tensor * t, ಪೂರ್ಣಾಙ್ಕ i) {
    ಪ್ರತ್ಯಾಗಮ (uint32_t)(t->nb[i] / ggml_type_size(t->type));
}

ಶೂನ್ಯ ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * mixes, ಧ್ರುವ ggml_tensor * scale, ಧ್ರುವ ggml_tensor * base, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_comb(" << mixes << ", " << scale << ", " << base << ", " << dst << ")");

    vk_pipeline pipeline = ctx->device->pipeline_dsv4_hc_comb_f32;
    GGML_ASSERT(pipeline != nullptr);

    ಧ್ರುವ uint32_t n_tokens = (uint32_t)mixes->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಧ್ರುವ vk_subbuffer mixes_buf = ggml_vk_tensor_subbuffer(ctx, mixes, ಸತ್ಯ);
    ಧ್ರುವ vk_subbuffer scale_buf = ggml_vk_tensor_subbuffer(ctx, scale, ಸತ್ಯ);
    ಧ್ರುವ vk_subbuffer base_buf  = ggml_vk_tensor_subbuffer(ctx, base,  ಸತ್ಯ);
    ಧ್ರುವ vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, dst,   ಸತ್ಯ);

    vk_op_dsv4_hc_comb_ತಳ್ಳು_constants pc = {
        n_tokens,
        ggml_vk_nb_elem(mixes, 0), ggml_vk_nb_elem(mixes, 1),
        ggml_vk_nb_elem(scale, 0),
        ggml_vk_nb_elem(base,  0),
        ggml_vk_nb_elem(dst,   0), ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2),
        0, 0, 0, 0,
        ggml_get_op_params_f32(dst, 0),
        (uint32_t)ggml_get_op_params_i32(dst, 1),
    };
    init_pushconst_tensor_offsets(ctx, pc, mixes, scale, base, nullptr, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { mixes_buf, scale_buf, base_buf, dst_buf }, pc, { n_tokens, 1, 1 });
}

ಶೂನ್ಯ ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * x, ಧ್ರುವ ggml_tensor * weights, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_pre(" << x << ", " << weights << ", " << dst << ")");

    ಧ್ರುವ ಭಿನ್ನ scale = ggml_get_op_params_f32(dst, 0);
    ಧ್ರುವ ತಾರ್ಕಿಕ  gated = ggml_get_op_params_i32(dst, 1) != 0;

    vk_pipeline pipeline = gated ? ctx->device->pipeline_dsv4_hc_pre_gated_f32 : ctx->device->pipeline_dsv4_hc_pre_f32;
    GGML_ASSERT(pipeline != nullptr);

    ಧ್ರುವ uint32_t n_embd   = (uint32_t)x->ne[0];
    ಧ್ರುವ uint32_t n_tokens = (uint32_t)x->ne[2];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಧ್ರುವ vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,       ಸತ್ಯ);
    ಧ್ರುವ vk_subbuffer w_buf = ggml_vk_tensor_subbuffer(ctx, weights, ಸತ್ಯ);
    ಧ್ರುವ vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,     ಸತ್ಯ);

    vk_op_dsv4_hc_pre_ತಳ್ಳು_constants pc = {
        n_embd, n_tokens,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1), ggml_vk_nb_elem(x, 2),
        ggml_vk_nb_elem(weights, 0), ggml_vk_nb_elem(weights, 1), ggml_vk_nb_elem(weights, 2),
        ggml_vk_nb_elem(dst, 0), ggml_vk_nb_elem(dst, 1),
        0, 0, 0,
        scale,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, weights, nullptr, nullptr, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, w_buf, d_buf }, pc, { n_embd, n_tokens, 1 });
}

ಶೂನ್ಯ ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * x, ಧ್ರುವ ggml_tensor * residual, ಧ್ರುವ ggml_tensor * post, ಧ್ರುವ ggml_tensor * comb, ggml_tensor * dst, ಧ್ರುವ ggml_tensor * gate_scale_in) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_post(" << x << ", " << residual << ", " << post << ", " << comb << ", " << dst << ")");

    vk_pipeline pipeline = comb ? ctx->device->pipeline_dsv4_hc_post_f32 : ctx->device->pipeline_dsv4_hc_post_nocomb_f32;
    GGML_ASSERT(pipeline != nullptr);

    ಧ್ರುವ uint32_t n_embd   = (uint32_t)x->ne[0];
    ಧ್ರುವ uint32_t n_tokens = (uint32_t)x->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಧ್ರುವ vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,        ಸತ್ಯ);
    ಧ್ರುವ vk_subbuffer r_buf = ggml_vk_tensor_subbuffer(ctx, residual, ಸತ್ಯ);
    // with a fused gate, post is scale(sigmoid(scale(p_src))) and the shader applies it to p_src
    ಧ್ರುವ ggml_tensor * p_src = gate_scale_in ? gate_scale_in->src[0] : post;
    ಧ್ರುವ vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p_src,    ಸತ್ಯ);
    ಧ್ರುವ vk_subbuffer c_buf = comb ? ggml_vk_tensor_subbuffer(ctx, comb, ಸತ್ಯ) : x_buf;
    ಧ್ರುವ vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,      ಸತ್ಯ);

    vk_op_dsv4_hc_post_ತಳ್ಳು_constants pc = {
        n_embd, n_tokens,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1),
        ggml_vk_nb_elem(residual, 0), ggml_vk_nb_elem(residual, 1), ggml_vk_nb_elem(residual, 2),
        ggml_vk_nb_elem(p_src, 0), ggml_vk_nb_elem(p_src, 1),
        comb ? ggml_vk_nb_elem(comb, 0) : 0, comb ? ggml_vk_nb_elem(comb, 1) : 0, comb ? ggml_vk_nb_elem(comb, 2) : 0,
        ggml_vk_nb_elem(dst,  0), ggml_vk_nb_elem(dst,  1), ggml_vk_nb_elem(dst,  2),
        0, 0, 0, 0, 0,
        gate_scale_in ? 1u : 0u,
        gate_scale_in ? ggml_get_op_params_f32(gate_scale_in, 0) : 1.0f,
        gate_scale_in ? ggml_get_op_params_f32(post, 0) : 1.0f,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, residual, p_src, comb, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, c_buf, d_buf }, pc, { n_embd, n_tokens, 1 });
}

ಶೂನ್ಯ ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

    // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
    // where the M dimension is very large.
    // Split_k doesn't work with M splitting.
    // This only supports batchsize == 1.
    ಧ್ರುವ size_t nbytes = ggml_nbytes(src0);
    ಧ್ರುವ ತಾರ್ಕಿಕ needs_split = dst->ne[2] == 1 && dst->ne[3] == 1 && nbytes > ctx->device->properties.limits.maxStorageBufferRange;
    ಯದಿ (needs_split) {
        // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
        ಧ್ರುವ uint32_t M_split = ctx->device->properties.limits.maxStorageBufferRange / (2 * src0->nb[1]);
        uint32_t m_offset = 0;
        ಯಾವತ್ (m_offset < dst->ne[0]) {
            ಧ್ರುವ uint32_t cur_M_size = std::min(M_split, (uint32_t)(dst->ne[0] - m_offset));
            ggml_tensor dst2 = *dst;
            ggml_tensor src02 = *src0;

            dst2.view_src = dst->view_src ? dst->view_src : dst;
            src02.view_src = src0->view_src ? src0->view_src : src0;

            dst2.view_offs += m_offset * dst->nb[0];
            src02.view_offs += m_offset * src0->nb[1];
            dst2.ne[0] = cur_M_size;
            src02.ne[1] = cur_M_size;

            ggml_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, ಸತ್ಯ);

            m_offset += cur_M_size;
        }
    } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_use_fwht(ctx, src1, dst)) {
        ggml_vk_fwht(ctx, subctx, src1, dst);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1 &&
        // detect 0213 permutation, and batch size of 1
        src0->nb[0] <= src0->nb[2] &&
        src0->nb[2] <= src0->nb[1] &&
        src0->nb[1] <= src0->nb[3] &&
        src1->nb[0] <= src1->nb[2] &&
        src1->nb[2] <= src1->nb[1] &&
        src1->nb[1] <= src1->nb[3] &&
        src0->ne[3] == 1 &&
        src1->ne[3] == 1 &&
        src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1] &&
        src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, cgraph, ಗಂಟು_idx);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1 &&
               !ggml_is_permuted(src0) && !ggml_is_permuted(src1) &&
               src0->ne[3] <= ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[0] &&
               src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1] &&
               src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, cgraph, ಗಂಟು_idx);
    // With one output row, B^T*A has the same flat output as A^T*B.
    } ಅನ್ಯಥಾ ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops == 0 &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
               (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16 || ggml_is_quantized(src1->type)) &&
               dst->ne[0] == 1 && dst->ne[1] > mul_mat_vec_max_cols &&
               src0->ne[2] == 1 && src0->ne[3] == 1 &&
               src1->ne[2] == 1 && src1->ne[3] == 1 &&
               ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
               get_misalign_bytes(ctx, src0) == 0 && get_misalign_bytes(ctx, src1) == 0 && get_misalign_bytes(ctx, dst) == 0) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, ಗಂಟು_idx, ಸತ್ಯ);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
    } ಅನ್ಯಥಾ ಯದಿ ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, ಗಂಟು_idx);
    } ಅನ್ಯಥಾ {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, ಅಸತ್ಯ);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ಧ್ರುವ uint64_t ne00 = src0->ne[0];
    ಧ್ರುವ uint64_t ne01 = src0->ne[1];
    ಧ್ರುವ uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ಧ್ರುವ uint64_t ne10 = src1->ne[0];
    ಧ್ರುವ uint64_t ne11 = src1->ne[1];
    ಧ್ರುವ uint64_t ne12 = src1->ne[2];
    ಧ್ರುವ uint64_t ne13 = src1->ne[3];

    ಧ್ರುವ uint64_t nei0 = ids->ne[0];
    ಧ್ರುವ uint64_t nei1 = ids->ne[1];

    ಧ್ರುವ uint32_t nbi0 = ids->nb[0];
    ಧ್ರುವ uint32_t nbi1 = ids->nb[1];

    ಧ್ರುವ uint64_t ne20 = dst->ne[0];
    ಧ್ರುವ uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ಧ್ರುವ uint64_t n_as = ne02;

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *)src1->buffer->context;

    vk_buffer d_Qx = nullptr;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;

    ತಾರ್ಕಿಕ src0_uma = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ src1_uma = ಅಸತ್ಯ;

    ಯದಿ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    ಧ್ರುವ ತಾರ್ಕಿಕ x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
#ಯದಿ defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
    // B must already be, or be convertible to, the matmul B type used by this path.
    ಧ್ರುವ ತಾರ್ಕಿಕ y_decode_vector_supported = ctx->device->coopmat2_decode_vector &&
                                           (f16_type != GGML_TYPE_BF16 || ctx->device->coopmat2_bf16_support) &&
                                           (src1->type == GGML_TYPE_F32 || src1->type == f16_type);
    // If B is copied to prealloc_y, we can choose a 4-element-aligned row stride.
    ಧ್ರುವ ತಾರ್ಕಿಕ y_decode_vector_uses_prealloc = !ggml_vk_dim01_contiguous(src1) || src1->type != f16_type;
    // Direct B reads are safe only if row starts and the original buffer offset are 4-element aligned.
    ಧ್ರುವ ತಾರ್ಕಿಕ y_decode_vector_aligned =
        (ne10 % 4 == 0) &&
        (y_decode_vector_uses_prealloc || get_misalign_bytes(ctx, src1) % (4 * ggml_type_size(src1->type)) == 0);
    // Stage B only when decode-vector is available and direct B reads would be misaligned.
    ಧ್ರುವ ತಾರ್ಕಿಕ y_decode_vector_staging = y_decode_vector_supported && !y_decode_vector_aligned;
#ಅನ್ಯಥಾ
    ಧ್ರುವ ತಾರ್ಕಿಕ y_decode_vector_staging = ಅಸತ್ಯ;
#endif
    ಧ್ರುವ ತಾರ್ಕಿಕ can_quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // Intel coopmat1 matrix cores prefer f16 activations for quantized weights
    ಧ್ರುವ ತಾರ್ಕಿಕ prefer_f16_b = ctx->device->coopmat_support && !ctx->device->coopmat2 &&
                              ctx->device->vendor_id == VK_VENDOR_ID_INTEL &&
                              ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ಧ್ರುವ ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], ಸತ್ಯ, can_quantize_y, prefer_f16_b);

    ಧ್ರುವ ತಾರ್ಕಿಕ quantize_y   = y_kernel_type == GGML_TYPE_Q8_1;
    ಧ್ರುವ ತಾರ್ಕಿಕ y_f32_kernel = y_kernel_type == GGML_TYPE_F32;

    ಧ್ರುವ std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0], ಸತ್ಯ);

    ಧ್ರುವ ತಾರ್ಕಿಕ qx_needs_dequant = mmp_map == nullptr || x_non_contig;

    ಯದಿ (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0], ಸತ್ಯ);
    }

    GGML_ASSERT(mmp_map != nullptr);

    // Coopmat2 MUL_MAT_ID BK specialization constants in ggml_vk_load_shaders are at most 64.
    ಧ್ರುವ uint32_t y_staged_row_stride = ctx->device->coopmat2 && !quantize_y ? ggml_vk_align_size(ne10, 64) : ne10;
    ಧ್ರುವ ತಾರ್ಕಿಕ y_needs_k_padding = ne10 != y_staged_row_stride;

    ಧ್ರುವ ತಾರ್ಕಿಕ y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type ||
                                                  y_decode_vector_staging || y_needs_k_padding);
    ಧ್ರುವ ತಾರ್ಕಿಕ qy_needs_dequant = y_needs_reformat;

    ಧ್ರುವ uint32_t kpad = quantize_y ? 0 : ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, nei1, ಸತ್ಯ));
    ಧ್ರುವ ತಾರ್ಕಿಕ aligned = !quantize_y && ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, nei1, aligned, ಸತ್ಯ);

    ಧ್ರುವ uint32_t bn = pipeline->wg_denoms[1];
    ಧ್ರುವ uint64_t max_tiles = ggml_vk_mul_mat_id_max_tiles(n_as, nei0 * nei1, bn);
    ಧ್ರುವ uint64_t row_ids_offset = 1 + 2 * max_tiles;
    ಧ್ರುವ uint64_t row_map_ಪದಗಳು = row_ids_offset + max_tiles * bn;

    ಯದಿ (ggml_nbytes(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }
    ಧ್ರುವ uint64_t x_ne = ggml_nelements(src0);
    ಧ್ರುವ uint64_t y_ne = (uint64_t)y_staged_row_stride * ne11 * ne12 * ne13;
    ಧ್ರುವ uint64_t d_ne = ggml_nelements(dst);

    ಧ್ರುವ uint64_t qx_sz = ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type);
    ಧ್ರುವ uint64_t qy_sz = ggml_type_size(src1->type) * ggml_nelements(src1) / ggml_blck_size(src1->type);
    ಧ್ರುವ uint64_t x_sz = !qx_needs_dequant ? qx_sz : ಪರಿಮಾಣ(ggml_fp16_t) * x_ne;
    ಧ್ರುವ uint64_t y_sz = quantize_y ? (ggml_vk_align_size(y_ne, 128) * ggml_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1)) : (y_f32_kernel ? ಪರಿಮಾಣ(ಭಿನ್ನ) * y_ne : ಪರಿಮಾಣ(ggml_fp16_t) * y_ne);
    ಧ್ರುವ uint64_t d_sz = ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne;

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    vk_pipeline to_q8_1 = nullptr;

    auto make_y_staged_dst = [&]() {
        ggml_tensor y_staged_dst = *src1;
        y_staged_dst.type = f16_type;
        y_staged_dst.nb[0] = ggml_type_size(f16_type);
        y_staged_dst.nb[1] = y_staged_dst.nb[0] * y_staged_row_stride;
        y_staged_dst.nb[2] = y_staged_dst.nb[1] * ne11;
        y_staged_dst.nb[3] = y_staged_dst.nb[2] * y_staged_dst.ne[2];
        ಪ್ರತ್ಯಾಗಮ y_staged_dst;
    };

    ಯದಿ (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type);
    } ಅನ್ಯಥಾ {
        to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type);
    }
    ಯದಿ (y_needs_reformat) {
        ggml_tensor y_staged_dst;
        ಧ್ರುವ ggml_tensor * y_staged_dst_ptr = nullptr;
        ಯದಿ (y_needs_k_padding) {
            y_staged_dst = make_y_staged_dst();
            y_staged_dst_ptr = &y_staged_dst;
        }

        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, y_staged_dst_ptr, f16_type);
    } ಅನ್ಯಥಾ {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    ಯದಿ (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }
    vk_pipeline ಎಣಿಕೆ_experts = ctx->device->pipeline_ಎಣಿಕೆ_experts;

    ಧ್ರುವ size_t expert_data_size = ಪರಿಮಾಣ(uint32_t) * row_map_ಪದಗಳು;

    {
        ಯದಿ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.maxStorageBufferRange) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.maxStorageBufferRange)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ಯದಿ (qx_needs_dequant && ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz) {
            ctx->prealloc_size_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ (ctx->prealloc_size_split_k < expert_data_size) {
            ctx->prealloc_size_split_k = expert_data_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ಯದಿ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        ಯದಿ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        ಯದಿ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, ಎಣಿಕೆ_experts, 1);
    }

    vk_buffer d_D = dst_buf_ctx->dev_buffer;
    ಧ್ರುವ uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    ಯದಿ (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_buffer;
        qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    ಯದಿ (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_buffer;
        qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    ಯದಿ (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } ಅನ್ಯಥಾ {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ಯದಿ (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } ಅನ್ಯಥಾ ಯದಿ (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } ಅನ್ಯಥಾ {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    ಯದಿ (x_non_contig || qx_needs_dequant) {
        ಯದಿ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids, ಸತ್ಯ);

    // Count how many times each expert is used
    vk_subbuffer expert_ಎಣಿಕೆ_buf = { ctx->prealloc_split_k, 0, expert_data_size };
    ಯದಿ (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }
    {
        vk_op_ಎಣಿಕೆ_experts_ತಳ್ಳು_constants pc = { (uint32_t)nei0,
                                           (uint32_t)nei1,
                                           (uint32_t)(nbi0 / ggml_type_size(ids->type)),
                                           (uint32_t)(nbi1 / ggml_type_size(ids->type)),
                                           (uint32_t)(get_misalign_bytes(ctx, ids) / ggml_type_size(ids->type)),
                                           (uint32_t)n_as,
                                           0, 0, bn, (uint32_t)row_ids_offset };
        init_pushconst_fastdiv(pc);
        ggml_vk_dispatch_pipeline(ctx, subctx, ಎಣಿಕೆ_experts,
            { d_ids, expert_ಎಣಿಕೆ_buf }, pc,
            { 1, 1, 1 });
    }

    ಯದಿ (x_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_offset), ggml_vk_subbuffer(ctx, d_X, 0));
    } ಅನ್ಯಥಾ ಯದಿ (qx_needs_dequant) {
        ಧ್ರುವ std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0,
            { vk_subbuffer{ d_Qx, qx_buf_offset, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)x_ne, 1, 1});
    }
    ಯದಿ (y_needs_reformat) {
        ಯದಿ (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded != y_needs_k_padding) {
            ಯದಿ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ಯದಿ (y_needs_k_padding) {
                GGML_ASSERT(y_sz % 4 == 0);
                // Zero B padding because clamping only A can produce 0 * Inf or NaN.
                subctx->s->buffer->buf.fillBuffer(d_Y->buffer, 0, y_sz, 0);
                ggml_vk_sync_buffers(ctx, subctx);
                ಧ್ರುವ ggml_tensor y_staged_dst = make_y_staged_dst();
                ಧ್ರುವ uint32_t y_staged_dst_type_size = ggml_type_size(y_staged_dst.type);
                ggml_vk_cpy_to_strided(
                    ctx, subctx, to_fp16_vk_1, src1,
                    ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0),
                    (uint32_t)(y_staged_dst.nb[0] / y_staged_dst_type_size),
                    (uint32_t)(y_staged_dst.nb[1] / y_staged_dst_type_size),
                    (uint32_t)(y_staged_dst.nb[2] / y_staged_dst_type_size),
                    (uint32_t)(y_staged_dst.nb[3] / y_staged_dst_type_size));
            } ಅನ್ಯಥಾ {
                ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0));
            }
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = y_needs_k_padding;
        }
    }
    ಯದಿ (quantize_y) {
        ಯದಿ (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            ಯದಿ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;
        }
    }
    ggml_vk_sync_buffers(ctx, subctx);

    ಧ್ರುವ uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ಧ್ರುವ uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_b_y = y_needs_k_padding ? y_staged_row_stride : ne10;
    uint32_t stride_batch_y = y_needs_k_padding ? y_staged_row_stride * ne11 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));

    ಯದಿ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    ಯದಿ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_offset, x_range }, { d_Y, y_buf_offset, y_range },
        { d_D, d_buf_offset, d_sz }, std::move(d_ids), expert_ಎಣಿಕೆ_buf,
        ne01, ne21, ne10, ne10, stride_b_y, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_size(ids->type), ne11, row_ids_offset, max_tiles
    );  // NOLINT

    ಯದಿ (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = ಸತ್ಯ;
    }
    ಯದಿ (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = ಸತ್ಯ;
    }
    ctx->prealloc_split_k_need_sync = ಸತ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * ids = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ಧ್ರುವ uint64_t ne00 = src0->ne[0];
    ಧ್ರುವ uint64_t ne01 = src0->ne[1];
    // const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ಧ್ರುವ uint64_t ne10 = src1->ne[0];
    ಧ್ರುವ uint64_t ne11 = src1->ne[1];
    ಧ್ರುವ uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ಧ್ರುವ uint64_t nei0 = ids->ne[0];
    ಧ್ರುವ uint64_t nei1 = ids->ne[1];
    ಧ್ರುವ uint32_t nbi1 = (uint32_t)(ids->nb[1] / ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));

    ಧ್ರುವ uint64_t ne20 = dst->ne[0];
    ಧ್ರುವ uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ಧ್ರುವ ತಾರ್ಕಿಕ x_non_contig = !ggml_vk_dim01_contiguous(src0);
    ಧ್ರುವ ತಾರ್ಕಿಕ y_non_contig = !ggml_vk_dim01_contiguous(src1);

    ಧ್ರುವ ತಾರ್ಕಿಕ f16_f32_kernel = src1->type == GGML_TYPE_F32;
    ತಾರ್ಕಿಕ quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne12, ne10, src0->type);

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    ಯದಿ (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type);
    }
    ಯದಿ (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, src1->type);
    } ಅನ್ಯಥಾ {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, GGML_TYPE_Q8_1, ne20, ne00) : nullptr;
    vk_pipeline to_q8_1 = nullptr;

    ಯದಿ (dmmv == nullptr) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type, ne20, ne00);
        quantize_y = ಅಸತ್ಯ;
    }

    ಯದಿ (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ qx_needs_dequant = x_non_contig;
    ಧ್ರುವ ತಾರ್ಕಿಕ qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    ಯದಿ (ggml_nbytes(src0) > ctx->device->properties.limits.maxStorageBufferRange) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    ಧ್ರುವ uint64_t x_ne = ggml_nelements(src0);
    ಧ್ರುವ uint64_t y_ne = ggml_nelements(src1);

    ಧ್ರುವ uint64_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    ಧ್ರುವ uint64_t x_sz = x_non_contig ? ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) : qx_sz;
    ಧ್ರುವ uint64_t y_sz = quantize_y ? (ggml_vk_align_size(y_ne, 128) * ggml_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1)) :
                                       (f16_f32_kernel ? ಪರಿಮಾಣ(ಭಿನ್ನ) * y_ne : ಪರಿಮಾಣ(ggml_fp16_t) * y_ne);

    {
        ಯದಿ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.maxStorageBufferRange) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.maxStorageBufferRange)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ಯದಿ (qx_needs_dequant && ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz) {
            ctx->prealloc_size_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ಯದಿ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        ಯದಿ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        ಯದಿ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, nei1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids);
    vk_subbuffer d_F0 = d_D;
    vk_subbuffer d_X, d_Y;

    ಯದಿ (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } ಅನ್ಯಥಾ {
        d_X = d_Qx;
    }
    ಯದಿ (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } ಅನ್ಯಥಾ {
        d_Y = d_Qy;
    }

    ಯದಿ (x_non_contig) {
        ಯದಿ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ಯದಿ (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment));
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, d_Qx, d_X);
    }
    ಯದಿ (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_size(src1->type) * y_ne);
        ಯದಿ (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            ಯದಿ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;
        }
    }
    ಯದಿ (quantize_y) {
        ಯದಿ (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
            ctx->prealloc_y_last_tensor_used != src1 ||
            ctx->prealloc_y_last_k_padded) {
            ಯದಿ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
            ctx->prealloc_y_last_tensor_used = src1;
            ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;
        }
    }

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    ಯದಿ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[2] / ggml_type_size(src1->type);
    }

    ಧ್ರುವ uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    ಯದಿ (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
        ಧ್ರುವ ggml_tensor * bias = cgraph->nodes[ಗಂಟು_idx + 1]->src[1];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);

        ಯದಿ (cgraph->nodes[ಗಂಟು_idx + 1]->op == GGML_OP_MUL) {
            fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE0;
        } ಅನ್ಯಥಾ {
            GGML_ASSERT(cgraph->nodes[ಗಂಟು_idx + 1]->op == GGML_OP_ADD_ID);
            fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
        }
    }

    vk_subbuffer d_F1 = d_D;
    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops > 1) {
        ಧ್ರುವ ggml_tensor * scale = cgraph->nodes[ಗಂಟು_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, scale);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE1;
    }

    // Loop over the batch dimension
    ಪ್ರತಿ (uint32_t expert_i1 = 0; expert_i1 < nei1; ++expert_i1) {
        ಧ್ರುವ vk_mat_vec_id_ತಳ್ಳು_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, (uint32_t)(ne20 * ne21),
            fusion_flags,
            (uint32_t)nei0, (uint32_t)ne11, expert_i1, nbi1
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
            {
                d_X,
                d_Y,
                d_D,
                d_F0,
                d_F1,
                d_ids,
            },
            pc, { groups_x, (uint32_t)nei0, groups_z });
    }

    ಯದಿ (x_non_contig) {
        ctx->prealloc_x_need_sync = ಸತ್ಯ;
    }
    ಯದಿ (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = ಸತ್ಯ;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_use_mul_mat_vec_id(ಧ್ರುವ ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * ids = dst->src[2];
    ಪ್ರತ್ಯಾಗಮ ids->ne[1] <= 8 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
}

ತಾರ್ಕಿಕ ggml_vk_use_mul_mat_vec_id(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಪ್ರತ್ಯಾಗಮ ggml_vk_use_mul_mat_vec_id(cgraph->nodes[ಗಂಟು_idx]);
}

ಶೂನ್ಯ ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    ಯದಿ (ggml_vk_use_mul_mat_vec_id(cgraph, ಗಂಟು_idx)) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, cgraph, ಗಂಟು_idx);
    } ಅನ್ಯಥಾ {
        ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    }
}

ತಾರ್ಕಿಕ ggml_vk_flash_attn_scalar_shmem_support(ಧ್ರುವ vk_device& device, ಧ್ರುವ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, ತಾರ್ಕಿಕ f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_UNUSED(f32acc);
    // Needs to be kept up to date on shader changes
    ಧ್ರುವ uint32_t wg_size = params.workgroup_size;
    ಧ್ರುವ uint32_t Br = params.block_rows;
    ಧ್ರುವ uint32_t Bc = params.block_cols;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    ಧ್ರುವ uint32_t float_type_size = (device->fp16 && k_type != GGML_TYPE_BF16) ? ಪರಿಮಾಣ(ggml_fp16_t) : ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಧ್ರುವ ತಾರ್ಕಿಕ mmq = ggml_vk_fa_scalar_uses_mmq(device, k_type, v_type);

    // tmpsh is overestimated slightly
    ಧ್ರುವ uint32_t tmpsh = wg_size * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t tmpshv4 = wg_size * 4 * float_type_size;

    ಧ್ರುವ uint32_t masksh = Bc * (Br + 1) * float_type_size;
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ಧ್ರುವ uint32_t iq_shmem = 16 * float_type_size;

    uint32_t Qf, kvsh, kblocksh_size;
    ಯದಿ (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        ಧ್ರುವ uint32_t block_b_size = 8 * ಪರಿಮಾಣ(int32_t) + 2 * float_type_size;
        Qf = Br * (hsk / 32) * block_b_size;

        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = params.shmem_staging ? Bc * (hsv / 4 + 1) * 4 * float_type_size : 4 * float_type_size;

        // The mixed MMQ shader uses a superset block_a_cache that fits every
        // FA-supported quant: int32_t qs[8] + uint32_t qh + FLOAT_TYPEV2 dm.
        // Single-scale types leave dm.y unused; non-Q5_* leave qh unused.
        ಧ್ರುವ uint32_t block_a_size = 8 * ಪರಿಮಾಣ(int32_t) + ಪರಿಮಾಣ(uint32_t) + 2 * float_type_size;
        kblocksh_size = params.shmem_staging ? Bc * (hsk / 32) * block_a_size : block_a_size;
    } ಅನ್ಯಥಾ {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_size;

        ಧ್ರುವ uint32_t D = std::max(hsk, hsv);
        kvsh = params.shmem_staging ? Bc * (D / 4 + 1) * 4 * float_type_size : 4 * float_type_size;

        kblocksh_size = 0;
    }

    ಧ್ರುವ uint32_t total_size = tmpsh + tmpshv4 + masksh + iq_shmem + Qf + kvsh + kblocksh_size;
    ಧ್ರುವ ತಾರ್ಕಿಕ supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_flash_attn_scalar_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", mmq=" << mmq << ", total_size=" << total_size << ", supported=" << supported);

    ಪ್ರತ್ಯಾಗಮ supported;
}

ತಾರ್ಕಿಕ ggml_vk_flash_attn_coopmat_shmem_support(ಧ್ರುವ vk_device& device, ಧ್ರುವ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, ತಾರ್ಕಿಕ f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_UNUSED(v_type);
    // Needs to be kept up to date on shader changes
    ಧ್ರುವ uint32_t Br = params.block_rows;
    ಧ್ರುವ uint32_t Bc = params.block_cols;

    ಧ್ರುವ uint32_t MatBr = 16, MatBc = 16;

    ಧ್ರುವ uint32_t row_split = Bc / MatBc;

    ಧ್ರುವ uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);
    ಧ್ರುವ uint32_t hsv_pad = ROUNDUP_POW2(hsv, 16);

    ಧ್ರುವ uint32_t acctype = f32acc ? 4 : 2;
    ಧ್ರುವ uint32_t f16vec4 = 8;

    ಧ್ರುವ uint32_t tmpsh = (Bc / MatBc) * ಪರಿಮಾಣ(ಭಿನ್ನ);
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ಧ್ರುವ uint32_t iq_shmem = 16 * ಪರಿಮಾಣ(ggml_fp16_t);

    ಧ್ರುವ uint32_t qstride = hsk_pad / 4 + 2;
    ಧ್ರುವ uint32_t Qf = Br * qstride * f16vec4;

    ಧ್ರುವ uint32_t psh_stride = Br / 4 + 2;
    ಧ್ರುವ uint32_t Psh = Bc * psh_stride * f16vec4;

    ಧ್ರುವ uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    ಧ್ರುವ uint32_t sfsh = Bc * sfshstride * acctype;

    ಧ್ರುವ uint32_t kvshstride = (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : MatBr) / 4 + 2;
    ಧ್ರುವ uint32_t vsh_stride = MatBc / 4 * row_split;
    ಧ್ರುವ uint32_t ksh = ((kvshstride >= vsh_stride) ? (Bc * kvshstride) : (Bc * vsh_stride)) * f16vec4;

    // BF16 PVMat accumulator is f32 (no bf16 accumulator support), so pvsh is vec4 (16 bytes)
    ಧ್ರುವ uint32_t pvsh_elem_size = (k_type == GGML_TYPE_BF16) ? 16u : f16vec4;
    ಧ್ರುವ uint32_t osh_stride = params.row_split * MatBr / 4;
    ಧ್ರುವ uint32_t pvsh = MatBc * osh_stride * pvsh_elem_size;

    ಧ್ರುವ uint32_t slope = Br * acctype;

    ಧ್ರುವ uint32_t total_size = tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + slope;
    ಧ್ರುವ ತಾರ್ಕಿಕ supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << total_size << ", supported=" << supported);

    ಪ್ರತ್ಯಾಗಮ supported;
}

ಶೂನ್ಯ ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * q, ಧ್ರುವ ggml_tensor * k, ಧ್ರುವ ggml_tensor * v, ಧ್ರುವ ggml_tensor * mask, ಧ್ರುವ ggml_tensor * sinks, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    ಯದಿ (sinks) {
        std::cerr << "), (" << sinks << ", name=" << sinks->name << ", type=" << sinks->type << ", ne0=" << sinks->ne[0] << ", ne1=" << sinks->ne[1] << ", ne2=" << sinks->ne[2] << ", ne3=" << sinks->ne[3] << ", nb0=" << sinks->nb[0] << ", nb1=" << sinks->nb[1] << ", nb2=" << sinks->nb[2] << ", nb3=" << sinks->nb[3];
    }
    std::cerr << "))");

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ಧ್ರುವ uint32_t nem0 = mask ? mask->ne[0] : 0;
    ಧ್ರುವ uint32_t nem1 = mask ? mask->ne[1] : 0;
    ಧ್ರುವ uint32_t nem2 = mask ? mask->ne[2] : 0;
    ಧ್ರುವ uint32_t nem3 = mask ? mask->ne[3] : 0;

    ಧ್ರುವ uint32_t HSK = nek0;
    ಧ್ರುವ uint32_t HSV = nev0;
    uint32_t N = neq1;
    ಧ್ರುವ uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಪರಿಮಾಣ(ಭಿನ್ನ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ಪ್ರತಿಜ್ಞಾ(dst->type == GGML_TYPE_F32);
    ಪ್ರತಿಜ್ಞಾ(q->type == GGML_TYPE_F32);
    uint32_t gqa_ratio = 1;
    uint32_t qk_ratio = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    ಧ್ರುವ ತಾರ್ಕಿಕ f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;

    // dequant K/V once into an f16 scratch, reordered KV layout so FA can read without a stride
    auto is_dense_kv_cache = [](ಧ್ರುವ ggml_tensor * t) {
        ಪ್ರತ್ಯಾಗಮ t->nb[0] == ggml_type_size(t->type) &&
               t->nb[2] == ggml_row_size(t->type, t->ne[0]) &&
               t->nb[1] == t->nb[2] * t->ne[2] &&
               (t->ne[3] == 1 || t->nb[3] == t->nb[1] * t->ne[1]);
    };
    ಧ್ರುವ ತಾರ್ಕಿಕ k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
    ಧ್ರುವ ತಾರ್ಕಿಕ v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
    ಧ್ರುವ ತಾರ್ಕಿಕ kv_raw = (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && k->type == v->type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    ಧ್ರುವ ತಾರ್ಕಿಕ kv_raw_strided = ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
                                ctx->device->architecture != vk_device_architecture::AMD_RDNA4 &&
                                neq1 >= 256 &&
                                kv_raw &&
                                (k->nb[1] != (uint64_t)HSK * ggml_type_size(k->type) ||
                                 v->nb[1] != (uint64_t)HSV * ggml_type_size(v->type)) &&
                                (HSK % 8) == 0 && (HSV % 8) == 0;
    ಧ್ರುವ size_t kv_scratch_ts = kv_raw_strided ? ggml_type_size(k->type) : ಪರಿಮಾಣ(ggml_fp16_t);
    ಧ್ರುವ ತಾರ್ಕಿಕ use_dequant_kv = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                                is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                                (uint64_t)ggml_nelements(k) * kv_scratch_ts <= ctx->device->properties.limits.maxStorageBufferRange &&
                                (uint64_t)ggml_nelements(v) * kv_scratch_ts <= ctx->device->properties.limits.maxStorageBufferRange &&
                                (kv_raw_strided ||
                                 (ctx->device->pipeline_dequant_transpose[k->type] != nullptr &&
                                  ctx->device->pipeline_dequant_transpose[v->type] != nullptr)) &&
                                // coopmat2 path does not benefit from the f16 scratch
                                !ctx->device->coopmat2 &&
                                // Intel Xe1 regresses, see PR 25494
                                (ctx->device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 (ctx->device->coopmat_support && ctx->device->architecture != vk_device_architecture::INTEL_XE1));
    ಧ್ರುವ ggml_type k_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : k->type;
    ಧ್ರುವ ggml_type v_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : v->type;

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
    ಧ್ರುವ uint32_t max_gqa = std::min(tuning_params.block_rows, 32u);

    ಯದಿ (N <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa &&
        qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_ratio = qk_ratio;
        N = gqa_ratio;
        workgroups_y /= gqa_ratio;
    }

    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    ಭಿನ್ನ scale         = 1.0f;
    ಭಿನ್ನ max_bias      = 0.0f;
    ಭಿನ್ನ logit_softcap = 0.0f;

    memcpy(&scale,         (ಧ್ರುವ ಭಿನ್ನ *) dst->op_params + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&max_bias,      (ಧ್ರುವ ಭಿನ್ನ *) dst->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&logit_softcap, (ಧ್ರುವ ಭಿನ್ನ *) dst->op_params + 2, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಯದಿ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // Sparse mask hint (op_params[4]): compact the <= n_kv_max finite positions and gather only those.
    ಧ್ರುವ int32_t n_kv_max = mask ? ggml_get_op_params_i32(dst, 4) : 0;
    ಸ್ಥಿರ ಧ್ರುವ ತಾರ್ಕಿಕ disable_sparse = getenv("GGML_VK_FA_SPARSE_DISABLE") != nullptr;
    ಧ್ರುವ ತಾರ್ಕಿಕ kv_f16 = k_type_eff == GGML_TYPE_F16 && v_type_eff == GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    ಧ್ರುವ int64_t min_ratio = tuning_params.path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16);
    // coopmat2 vector decode requires 8B strides.
    auto sparse_gather_aligned = [](ಧ್ರುವ ggml_tensor * t) {
        ಪ್ರತ್ಯಾಗಮ (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
               (t->nb[1] | t->nb[2] | t->nb[3]) % (4 * ಪರಿಮಾಣ(ggml_fp16_t)) == 0;
    };
    ಧ್ರುವ ತಾರ್ಕಿಕ use_sparse = !disable_sparse && n_kv_max > 0 && mask &&
                            max_bias == 0.0f && logit_softcap == 0.0f &&
                            (tuning_params.path != FA_COOPMAT2 || (sparse_gather_aligned(k) && sparse_gather_aligned(v))) &&
                            nem0 == KV &&
                            (int64_t)KV >= std::max<int64_t>(4096, min_ratio * (int64_t)n_kv_max) &&
                            (gqa_ratio > 1 || (tuning_params.path == FA_SCALAR && N == 1));

    ಧ್ರುವ uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_size(q->type));
    uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_size(k->type));
    uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_size(v->type));

    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    ಯದಿ (k->type == GGML_TYPE_F32) {
        k_stride /= 4;
    }
    ಯದಿ (v->type == GGML_TYPE_F32) {
        v_stride /= 4;
    }

    uint32_t nbk2_eff = (uint32_t)nbk2, nbk3_eff = (uint32_t)nbk3;
    uint32_t nbv2_eff = (uint32_t)nbv2, nbv3_eff = (uint32_t)nbv3;
    ಯದಿ (use_dequant_kv) {
        k_stride = HSK;
        v_stride = HSV;
        nbk2_eff = (uint32_t)((uint64_t)HSK * KV * ಪರಿಮಾಣ(ggml_fp16_t));
        nbk3_eff = (uint32_t)((uint64_t)HSK * KV * nek2 * ಪರಿಮಾಣ(ggml_fp16_t));
        nbv2_eff = (uint32_t)((uint64_t)HSV * KV * ಪರಿಮಾಣ(ggml_fp16_t));
        nbv3_eff = (uint32_t)((uint64_t)HSV * KV * nev2 * ಪರಿಮಾಣ(ggml_fp16_t));
    }
    ಧ್ರುವ uint32_t alignment = tuning_params.block_cols;
    ತಾರ್ಕಿಕ aligned = (KV % alignment) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    ಯದಿ (((HSK | HSV) % 16) != 0 && tuning_params.path == FA_COOPMAT2) {
        aligned = ಅಸತ್ಯ;
    }

    // Only use mask opt when the mask is fairly large. This hasn't been tuned extensively.
    ತಾರ್ಕಿಕ use_mask_opt = mask && !use_sparse && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.block_cols * 16
                        && (ctx->device->architecture != vk_device_architecture::AMD_GCN || HSK > 256 || HSV > 256);
    vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
                                                                   mask != nullptr, use_mask_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    vk_pipeline pipeline = nullptr;

    ತಾರ್ಕಿಕ xe_fa_opt = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ fa_copy_qstate = ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ xe_fa_supported_platform =
        (ctx->device.get()->architecture == INTEL_XE2 && ctx->device.get()->properties.deviceID != 0xFD80 && ctx->device.get()->properties.deviceID != 0xFD81) ||
        (ctx->device.get()->architecture == INTEL_XE1 && ctx->device.get()->coopmat_support && ctx->device.get()->uma);
    ತಾರ್ಕಿಕ xe_fa_supported_usage = neq0 % 32 == 0 && nev0 % 16 == 0 && q->nb[1] > q->nb[2] && k->nb[1] > k->nb[2] && v->nb[1] > v->nb[2] && mask != nullptr;
    ತಾರ್ಕಿಕ xe_fa_supported_dtype = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && (mask != nullptr && mask->type == GGML_TYPE_F16);
    std::pair<vk_pipeline, vk_pipeline> xe_fa_pipeline_dual_phases = { nullptr , nullptr };
    vk_pipeline xe_fa_pipeline = nullptr;
    size_t size_p = 0;
    size_t size_group_max = 0;

    {
        std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
        auto &pipelines = ctx->device->pipeline_flash_attn_f32_f16;
        auto it = pipelines.find(fa_pipeline_state);
        ಯದಿ (it != pipelines.end()) {
            pipeline = it->second;
        } ಅನ್ಯಥಾ {
            pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
        }
    }

    ಪ್ರತಿಜ್ಞಾ(pipeline);
    // Compile early to initialize wg_denoms.
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    uint32_t split_kv = KV;
    uint32_t split_k = 1;

    // Intel Alchemist prefers more workgroups
    ಧ್ರುವ uint32_t shader_core_ಎಣಿಕೆ_multiplier = (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ctx->device->architecture != INTEL_XE2) ? 2 : 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    ಧ್ರುವ uint32_t shader_core_ಎಣಿಕೆ = ctx->device->shader_core_ಎಣಿಕೆ ? ctx->device->shader_core_ಎಣಿಕೆ * shader_core_ಎಣಿಕೆ_multiplier : 16;

    ಧ್ರುವ uint32_t Br = fa_pipeline_state.Br;
    ಧ್ರುವ uint32_t Bc = fa_pipeline_state.Bc;

    GGML_ASSERT(Br == pipeline->wg_denoms[0]);
    ಧ್ರುವ uint32_t Tr = CEIL_DIV(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for occupancy.
    ಯದಿ (use_sparse) {
        split_kv = (uint32_t)n_kv_max;
        ಧ್ರುವ uint32_t total_blocks = CEIL_DIV((uint32_t)n_kv_max, Bc);
        ಧ್ರುವ uint32_t base_wgs = (gqa_ratio > 1 ? workgroups_x : Tr) * workgroups_y * workgroups_z;
        ಯದಿ (base_wgs < shader_core_ಎಣಿಕೆ * 2) {
            split_k = shader_core_ಎಣಿಕೆ * 2 / base_wgs;
        }
        split_k = std::max(1u, std::min(split_k, total_blocks));
        // Match the shader's per-split block count so no split is empty.
        ಧ್ರುವ uint32_t per_blocks = CEIL_DIV(total_blocks, split_k);
        split_k = CEIL_DIV(total_blocks, per_blocks);
    } ಅನ್ಯಥಾ ಯದಿ (gqa_ratio > 1 && workgroups_x <= Br) {
        split_k = shader_core_ಎಣಿಕೆ * 2 / (workgroups_x * workgroups_y * workgroups_z);
    } ಅನ್ಯಥಾ ಯದಿ (gqa_ratio <= 1) {
        uint32_t total_wgs_no_split = Tr * workgroups_y * workgroups_z;
        ಯದಿ (total_wgs_no_split < shader_core_ಎಣಿಕೆ * 2) {
            split_k = shader_core_ಎಣಿಕೆ * 2 / total_wgs_no_split;
        }
    }

    ಯದಿ (!use_sparse && split_k > 1) {
        // Try to evenly split KV into split_k chunks, but it needs to be a multiple
        // of "align", so recompute split_k based on that.
        split_kv = ROUNDUP_POW2(std::max(1u, KV / split_k), alignment);
        split_k = CEIL_DIV(KV, split_kv);
        xe_fa_opt = xe_fa_supported_platform && xe_fa_supported_usage && xe_fa_supported_dtype;
        ಯದಿ (xe_fa_opt) {
            ಧ್ರುವ uint32_t split_p_size = 32;
            ಧ್ರುವ size_t max_dim = (nek1 + split_p_size - 1) / split_p_size;
            ಧ್ರುವ size_t p_dim = max_dim * split_p_size;
#ಯದಿ defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            auto& pipelines = ctx->device->pipeline_xe_fa_decode_dual_phases;
            auto it = pipelines.find({ (uint32_t)neq0, (uint32_t)nev0, qk_ratio, (uint32_t)neq1 });
            ಯದಿ (it != pipelines.end()) {
                xe_fa_pipeline_dual_phases = it->second;
            } ಅನ್ಯಥಾ {
                pipelines[{(uint32_t)neq0, (uint32_t)nev0, qk_ratio, (uint32_t)neq1}] = xe_fa_pipeline_dual_phases = std::make_pair(std::make_shared<vk_pipeline_struct>(), std::make_shared<vk_pipeline_struct>());
            }
#endif
            ಯದಿ (xe_fa_pipeline_dual_phases.first == nullptr || xe_fa_pipeline_dual_phases.second == nullptr) {
                xe_fa_opt = ಅಸತ್ಯ;
                fa_copy_qstate = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ {
                size_p = neq1 * neq2 * p_dim * neq3 * ಪರಿಮಾಣ(ggml_fp16_t);
                size_group_max = neq1 * neq2 * max_dim * neq3 * ಪರಿಮಾಣ(ಭಿನ್ನ);
                size_t temp_size = ggml_nelements(q) * ಪರಿಮಾಣ(ggml_fp16_t) + size_p + size_group_max;
                fa_copy_qstate = ಸತ್ಯ;
                ಯದಿ (ctx->prealloc_size_x < temp_size) {
                    ctx->prealloc_size_x = temp_size;
                    ggml_vk_preallocate_buffers(ctx, subctx);
                }

                ಯದಿ (ctx->prealloc_x_need_sync) {
                    ggml_vk_sync_buffers(ctx, subctx);
                }
            }
        }
    }

    ಯದಿ (xe_fa_opt == ಸತ್ಯ) {
        use_mask_opt = ಅಸತ್ಯ;
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    // For matrices, the order is (inner to outer) [HSV, ne1, k, ne2, ne3].
    // For L/M, the order is (inner to outer) [ne1, k, ne2, ne3].
    ಧ್ರುವ uint64_t split_k_size = split_k > 1 ? (HSV * ne1 * ಪರಿಮಾಣ(ಭಿನ್ನ) + ne1 * ಪರಿಮಾಣ(ಭಿನ್ನ) * 2) * split_k * ne2 * ne3 : 0;
    ಯದಿ (split_k_size > ctx->device->properties.limits.maxStorageBufferRange) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    ಯದಿ (ctx->prealloc_size_split_k < split_k_size) {
        ctx->prealloc_size_split_k = split_k_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    ಧ್ರುವ uint32_t mask_opt_ಅಂಕೆ_dwords = CEIL_DIV(nem0, 16 * Bc);
    ಧ್ರುವ uint64_t mask_opt_size = ಪರಿಮಾಣ(uint32_t) * mask_opt_ಅಂಕೆ_dwords * CEIL_DIV(nem1, Br) * nem2 * nem3;

    vk_pipeline pipeline_fa_mask_opt = nullptr;
    ಯದಿ (use_mask_opt) {
        {
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            auto &pipelines = ctx->device->pipeline_fa_mask_opt;
            auto it = pipelines.find({Br, Bc});
            ಯದಿ (it != pipelines.end()) {
                pipeline_fa_mask_opt = it->second;
            } ಅನ್ಯಥಾ {
                pipelines[{Br, Bc}] = pipeline_fa_mask_opt = std::make_shared<vk_pipeline_struct>();
            }
        }
        ಪ್ರತಿಜ್ಞಾ(pipeline_fa_mask_opt);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_fa_mask_opt, 1);

        ಯದಿ (ctx->prealloc_size_y < mask_opt_size) {
            ctx->prealloc_size_y = mask_opt_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    ಧ್ರುವ uint64_t sparse_idx_size = use_sparse
        ? ಪರಿಮಾಣ(int32_t) * (uint64_t)n_kv_max * nem1 * nem2 * nem3
        : 0;
    vk_pipeline sparse_compact_pipeline = ctx->device->fa_sparse_compact_use_subgroups
        ? ctx->device->pipeline_fa_sparse_compact_subgroup
        : ctx->device->pipeline_fa_sparse_compact;
    ಯದಿ (use_sparse) {
        ggml_pipeline_request_descriptor_sets(ctx, sparse_compact_pipeline, 1);
        ಯದಿ (ctx->prealloc_size_y < sparse_idx_size) {
            ctx->prealloc_size_y = sparse_idx_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ಧ್ರುವ uint32_t n_head_kv   = neq2;
    ಧ್ರುವ uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((ಭಿನ್ನ) n_head_kv));
    ಧ್ರುವ ಭಿನ್ನ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ಧ್ರುವ ಭಿನ್ನ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer mask_buf = mask ? ggml_vk_tensor_subbuffer(ctx, mask) : q_buf;
    vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
    vk_subbuffer mask_opt_buf = use_mask_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    vk_subbuffer sparse_buf = use_sparse ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    ಯದಿ (use_mask_opt || use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_last_pipeline_used = nullptr;
        ctx->prealloc_y_last_tensor_used = nullptr;
    }

    ಯದಿ (use_dequant_kv) {
        ಧ್ರುವ uint64_t k_sz = (uint64_t)ggml_nelements(k) * kv_scratch_ts;
        ಧ್ರುವ uint64_t v_sz = (uint64_t)ggml_nelements(v) * kv_scratch_ts;
        ಯದಿ (ctx->prealloc_size_x < k_sz + v_sz) {
            ctx->prealloc_size_x = k_sz + v_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        vk_subbuffer k_dst = vk_subbuffer{ ctx->prealloc_x, 0,    k_sz };
        vk_subbuffer v_dst = vk_subbuffer{ ctx->prealloc_x, k_sz, v_sz };
        ಧ್ರುವ uint32_t k_nel = (uint32_t)ggml_nelements(k);
        ಧ್ರುವ uint32_t v_nel = (uint32_t)ggml_nelements(v);
        ಯದಿ (kv_raw_strided) {
            vk_pipeline cp_k = ggml_vk_get_cpy_pipeline(ctx, k, nullptr, k->type);
            vk_pipeline cp_v = ggml_vk_get_cpy_pipeline(ctx, v, nullptr, v->type);
            ggml_pipeline_request_descriptor_sets(ctx, cp_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, cp_v, 1);
            ಯದಿ (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            auto make_pc = [](uint32_t hs, uint32_t nh, uint32_t kv, uint32_t nel) {
                vk_op_unary_ತಳ್ಳು_constants pc{};
                pc.ne = nel;
                pc.ne00 = hs; pc.ne01 = nh;      pc.ne02 = kv; pc.ne03 = nel / (hs * nh * kv);
                pc.nb00 = 1;  pc.nb01 = hs;      pc.nb02 = hs * nh; pc.nb03 = hs * nh * kv;
                pc.ne10 = hs; pc.ne11 = nh;      pc.ne12 = kv; pc.ne13 = pc.ne03;
                pc.nb10 = 1;  pc.nb11 = hs * kv; pc.nb12 = hs; pc.nb13 = hs * kv * nh;
                init_pushconst_fastdiv(pc);
                ಪ್ರತ್ಯಾಗಮ pc;
            };
            auto cpy_elems = [](uint32_t ne) -> std::array<uint32_t, 3> {
                ಯದಿ (ne > 262144) { ಪ್ರತ್ಯಾಗಮ { 512, 512, CEIL_DIV(ne, 262144) }; }
                ಯದಿ (ne > 512)    { ಪ್ರತ್ಯಾಗಮ { 512, CEIL_DIV(ne, 512), 1 }; }
                ಪ್ರತ್ಯಾಗಮ { ne, 1, 1 };
            };
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_k, { k_buf, k_dst },
                make_pc((uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, k_nel), cpy_elems(k_nel));
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_v, { v_buf, v_dst },
                make_pc((uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, v_nel), cpy_elems(v_nel));
        } ಅನ್ಯಥಾ {
            vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
            vk_pipeline tr_v = ctx->device->pipeline_dequant_transpose[v->type];
            ggml_pipeline_request_descriptor_sets(ctx, tr_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, tr_v, 1);
            ಯದಿ (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            { ಧ್ರುವ std::vector<uint32_t> pc = { (uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, 0, k_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_k, { k_buf, k_dst }, pc, { k_nel, 1, 1 }); }
            { ಧ್ರುವ std::vector<uint32_t> pc = { (uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, 0, v_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_v, { v_buf, v_dst }, pc, { v_nel, 1, 1 }); }
        }
        ggml_vk_sync_buffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    uint32_t mask_n_head_log2 = ((sinks != nullptr) << 24) | n_head_log2;

    ಯದಿ (use_mask_opt)
    {
        ಧ್ರುವ vk_op_flash_attn_mask_opt_ತಳ್ಳು_constants opt_pc = {
            nem0,
            nem1,
            nem2,
            (uint32_t)(mask->nb[1] / ಪರಿಮಾಣ(ggml_fp16_t)),
            (uint32_t)(mask->nb[2] / ಪರಿಮಾಣ(ggml_fp16_t)),
            (uint32_t)(mask->nb[3] / ಪರಿಮಾಣ(ggml_fp16_t)),
            mask_opt_ಅಂಕೆ_dwords,
            mask_opt_ಅಂಕೆ_dwords * CEIL_DIV(nem1, Br),
            mask_opt_ಅಂಕೆ_dwords * CEIL_DIV(nem1, Br) * nem2,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline_fa_mask_opt,
                                  { mask_buf, mask_opt_buf }, opt_pc,
                                  { mask_opt_ಅಂಕೆ_dwords, CEIL_DIV(nem1, Br), nem2 * nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ಯದಿ (use_sparse)
    {
        ಧ್ರುವ vk_op_flash_attn_sparse_compact_ತಳ್ಳು_constants sc_pc = {
            KV,
            nem1,
            nem2,
            (uint32_t)(mask->nb[1] / ಪರಿಮಾಣ(ggml_fp16_t)),
            (uint32_t)(mask->nb[2] / ಪರಿಮಾಣ(ggml_fp16_t)),
            (uint32_t)(mask->nb[3] / ಪರಿಮಾಣ(ggml_fp16_t)),
            (uint32_t)n_kv_max,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, sparse_compact_pipeline,
                                  { mask_buf, sparse_buf }, sc_pc,
                                  { nem1, nem2, nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ಧ್ರುವ vk_flash_attn_ತಳ್ಳು_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, nbk2_eff, nbk3_eff,
                                              v_stride, nbv2_eff, nbv3_eff,
                                              scale, max_bias, logit_softcap,
                                              mask_n_head_log2, m0, m1,
                                              gqa_ratio, split_kv, split_k };

    ಯದಿ (xe_fa_opt && split_k > 1) {
        auto upper_power_of_2 = [&](uint32_t in) {
            GGML_ASSERT(in != 0);
            ಯದಿ (in <= 1) ಪ್ರತ್ಯಾಗಮ 1u;
            uint32_t ret = in - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            ಪ್ರತ್ಯಾಗಮ ret + 1;
        };
        auto to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, q->type);
        ಧ್ರುವ uint32_t out_dim_per_wg = qk_ratio > 16 ? 8 : 16;
        size_t x_ne = ggml_nelements(q);
        size_t temp_buf_offset = 0;
        uint32_t head_stride_k = uint32_t(nbk2 / ggml_type_size(k->type));
        uint32_t head_stride_v = uint32_t(nbv2 / ggml_type_size(v->type));
        uint32_t batch_stride_q = uint32_t(nbq3 / ggml_type_size(q->type));
        uint32_t batch_stride_k = uint32_t(nbk3 / ggml_type_size(k->type));
        uint32_t batch_stride_v = uint32_t(nbv3 / ggml_type_size(v->type));
        uint32_t batch_stride_m = mask ? uint32_t(mask->nb[3] / ggml_type_size(mask->type)) : 0u;
        uint32_t batch_stride_o = uint32_t(nb3 / ggml_type_size(dst->type));
        vk_fa_xe_opt_ತಳ್ಳು_constants pc_ph1 = { (uint32_t)nek1, (uint32_t)neq1, (uint32_t)neq2, (uint32_t)nek2, qk_ratio, 1, (sinks != nullptr) ? 1u : 0u, (uint32_t)k_stride, head_stride_k,
            batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_m, batch_stride_o, scale };
        vk_fa_xe_opt_ತಳ್ಳು_constants pc_ph2 = pc_ph1;
        pc_ph2.nbkv_tok = v_stride;
        pc_ph2.nbkv_head = head_stride_v;
        vk_subbuffer q_temp_buf = fa_copy_qstate ? ggml_vk_subbuffer(ctx, ctx->prealloc_x, temp_buf_offset) : q_buf;
        temp_buf_offset += fa_copy_qstate ? x_ne * ಪರಿಮಾಣ(ggml_fp16_t) : 0;
        vk_subbuffer p_temp_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, temp_buf_offset);
        temp_buf_offset += size_p;
        vk_subbuffer max_temp_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, temp_buf_offset);
        temp_buf_offset += size_group_max;
        uint32_t xe_native_sub_group_size = ctx->device.get()->architecture == INTEL_XE1 ? 8 : 16;
        uint32_t aligned_gqa_ratio = upper_power_of_2(qk_ratio);
        uint32_t out_per_wg_ph1 = std::min(256u / xe_native_sub_group_size, (uint32_t)neq1);
        uint32_t out_per_wg_ph2 = std::min(std::max(16u / aligned_gqa_ratio, 1u), (uint32_t)neq1);
        uint32_t ph1_wg = ((neq1 + out_per_wg_ph1 - 1) / out_per_wg_ph1) * nek2;
        uint32_t ph2_wg = ((neq1 + out_per_wg_ph2 - 1) / out_per_wg_ph2) * ne0 / out_dim_per_wg;
        ಯದಿ (fa_copy_qstate) {
            ಧ್ರುವ std::vector<uint32_t> pc_cpy_fp16 =
            { (uint32_t)q->ne[0], (uint32_t)q->ne[1], (uint32_t)q->ne[2], (uint32_t)q->ne[3], (uint32_t)(x_ne) };
            ggml_vk_sync_buffers(ctx, subctx);
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0, { q_buf, q_temp_buf }, pc_cpy_fp16, { (uint32_t)(x_ne), 1, 1 });
        }

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.first, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.first,
            { q_temp_buf, k_buf, mask_buf, p_temp_buf, max_temp_buf },
            pc_ph1, { (uint32_t)ph1_wg, (uint32_t)nek1, (uint32_t)neq3 });

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.second, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.second,
            { p_temp_buf, v_buf, max_temp_buf, sinks_buf, dst_buf },
            pc_ph2, { (uint32_t)ph2_wg, (uint32_t)nev2, (uint32_t)neq3 });

        ctx->prealloc_x_need_sync = ಸತ್ಯ;
    } ಅನ್ಯಥಾ ಯದಿ (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_split_k_reduce, 1);

        ಯದಿ (ctx->prealloc_split_k_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        // We reuse workgroups_x to mean the number of splits, so we need to
        // cancel out the divide by wg_denoms[0].
        uint32_t dispatch_x;
        ಯದಿ (gqa_ratio > 1) {
            workgroups_x *= pipeline->wg_denoms[0];
            dispatch_x = split_k * workgroups_x;
        } ಅನ್ಯಥಾ {
            dispatch_x = Tr * split_k * pipeline->wg_denoms[0];
        }

        vk_subbuffer split_k_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_split_k, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, mask_buf, sinks_buf, split_k_buf, mask_opt_buf, sparse_buf},
                                    pc, { dispatch_x, workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(ctx, subctx);
        ಧ್ರುವ vk_op_flash_attn_split_k_reduce_ತಳ್ಳು_constants pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3, split_k, (sinks != nullptr) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_split_k_reduce,
                                    {split_k_buf, sinks_buf, dst_buf},
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)(ne2 * ne3) });
        ctx->prealloc_split_k_need_sync = ಸತ್ಯ;
    } ಅನ್ಯಥಾ {
        ಯದಿ (gqa_ratio > 1) {
            // When using gqa, we want one actual workgroup per batch, so cancel out wg_denoms
            workgroups_x *= pipeline->wg_denoms[0];
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf, sparse_buf},
                                    pc, { workgroups_x, workgroups_y, workgroups_z });
    }

    ಯದಿ (use_dequant_kv) {
        ctx->prealloc_x_need_sync = ಸತ್ಯ;
    }
    ಯದಿ (use_mask_opt || use_sparse) {
        ctx->prealloc_y_need_sync = ಸತ್ಯ;
    }
}

ಸ್ಥಿರ vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
    auto n_tiles = [&](vk_conv_shapes s) {
        ಪ್ರತ್ಯಾಗಮ CEIL_DIV(K, vk_conv_block_sizes[s].K)
            * CEIL_DIV(NPQ, vk_conv_block_sizes[s].NPQ);
    };

    // We can't query number of shader cores on Intel, use 32 as a placeholder
    // so small convolutions will still choose a smaller tile.
    ಧ್ರುವ uint32_t shader_core_ಎಣಿಕೆ = ctx->device->shader_core_ಎಣಿಕೆ > 0 ? ctx->device->shader_core_ಎಣಿಕೆ : 32;

    // 128x128 isn't used with cm1 due to shared memory size; fall through to a smaller tile.
    ತಾರ್ಕಿಕ allow_128x128 = ಸತ್ಯ;
#ಯದಿ defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ಯದಿ (!ctx->device->coopmat2 && ctx->device->coopmat_support && ctx->device->coopmat_support_16x16x16_f16acc) {
        allow_128x128 = ಅಸತ್ಯ;
    }
#endif

    ಯದಿ (allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_128x128) >= shader_core_ಎಣಿಕೆ * 2) {
        ಪ್ರತ್ಯಾಗಮ CONV_SHAPE_128x128;
    } ಅನ್ಯಥಾ ಯದಿ (K <= 32 && n_tiles(CONV_SHAPE_32x256) >= shader_core_ಎಣಿಕೆ * 2) {
        ಪ್ರತ್ಯಾಗಮ CONV_SHAPE_32x256;
    } ಅನ್ಯಥಾ ಯದಿ (K <= 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_ಎಣಿಕೆ * 2) {
        ಪ್ರತ್ಯಾಗಮ CONV_SHAPE_64x128;
    } ಅನ್ಯಥಾ ಯದಿ (!allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_ಎಣಿಕೆ * 2) {
        // cm1 fallback for large K when 128x128 isn't available
        ಪ್ರತ್ಯಾಗಮ CONV_SHAPE_64x128;
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ CONV_SHAPE_64x32;
    }
}

ಸ್ಥಿರ vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * dst, ggml_op op) {
    ವಿಕಲ್ಪ (op) {
    ಪಕ್ಷ GGML_OP_GET_ROWS:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        ಯದಿ (src0->type == GGML_TYPE_I32) {
            // i32 src only supports i32 result
            GGML_ASSERT(dst->type == GGML_TYPE_I32);
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_get_rows[src0->type];
        }
        ಯದಿ (dst->type == GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_get_rows[src0->type];
        }
        ಯದಿ (dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_get_rows_f32[src0->type];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_GET_ROWS_BACK:
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_get_rows_back_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_ACC:
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_acc_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_SET:
        ಯದಿ (src0->type == src1->type && src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_set_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_ADD:
    ಪಕ್ಷ GGML_OP_SUB:
    ಪಕ್ಷ GGML_OP_MUL:
    ಪಕ್ಷ GGML_OP_DIV:
        ಯದಿ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
        ವಿಕಲ್ಪ (op) {
        ಪಕ್ಷ GGML_OP_ADD:
        {
            ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
                ಯದಿ (ctx->do_add_rms_partials) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_multi_add_rms[ctx->ಅಂಕೆ_additional_fused_ops];
                } ಅನ್ಯಥಾ {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_multi_add[ctx->ಅಂಕೆ_additional_fused_ops];
                }
            }
            ಯದಿ (ctx->do_add_rms_partials) {
                auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_add_rms_norepeat : ctx->device->pipeline_add_rms;
                ಪ್ರತ್ಯಾಗಮ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            } ಅನ್ಯಥಾ {
                auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_add_norepeat : ctx->device->pipeline_add;
                ಪ್ರತ್ಯಾಗಮ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            }
        }
        ಪಕ್ಷ GGML_OP_SUB:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            ಪ್ರತ್ಯಾಗಮ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        ಪಕ್ಷ GGML_OP_MUL:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            ಪ್ರತ್ಯಾಗಮ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        ಪಕ್ಷ GGML_OP_DIV:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            ಪ್ರತ್ಯಾಗಮ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_ADD_ID:
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_add_id_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_OUT_PROD:
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_out_prod_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_CONCAT: {
        ಯದಿ (!ggml_vk_concat_supported(src0, src1, dst)) {
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
        ವಿಕಲ್ಪ (ggml_vk_concat_unit_size(src0->type)) {
        ಪಕ್ಷ 1:
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_concat_i8;
        ಪಕ್ಷ 2:
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_concat_i16;
        ಪಕ್ಷ 4:
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_concat_i32;
        ಪಕ್ಷ 8:
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_concat_i64;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    }
    ಪಕ್ಷ GGML_OP_UPSCALE:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t mode = (ggml_get_op_params_i32(dst, 0) & (0xFF | GGML_SCALE_FLAG_ANTIALIAS));
            ವಿಕಲ್ಪ (mode) {
                ಪಕ್ಷ GGML_SCALE_MODE_NEAREST:
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_upscale_nearest_f32;
                ಪಕ್ಷ GGML_SCALE_MODE_BILINEAR:
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_upscale_bilinear_f32;
                ಪಕ್ಷ GGML_SCALE_MODE_BICUBIC:
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_upscale_bicubic_f32;
                ಪಕ್ಷ GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS:
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_upscale_bilinear_antialias_f32;
                ಸ್ವಭಾವ:
                    ಪ್ರತ್ಯಾಗಮ nullptr;
            }
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_SCALE:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_scale_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_SQR:
        ಯದಿ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_sqr[dst->type == GGML_TYPE_F16];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_SQRT:
        ಯದಿ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_sqrt[dst->type == GGML_TYPE_F16];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_SIN:
        ಯದಿ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_sin[dst->type == GGML_TYPE_F16];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_COS:
        ಯದಿ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cos[dst->type == GGML_TYPE_F16];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_LOG:
        ಯದಿ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_log[dst->type == GGML_TYPE_F16];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_TRI:
        ಯದಿ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_tri[dst->type == GGML_TYPE_F16];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_DIAG:
        ಯದಿ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_diag[dst->type == GGML_TYPE_F16];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_CLAMP:
        ಯದಿ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_clamp[dst->type == GGML_TYPE_F16];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_PAD:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_pad_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_PAD_REFLECT_1D:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_pad_reflect_1d_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_ROLL:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_roll_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_REPEAT:
        ಯದಿ (ggml_type_size(src0->type) == ಪರಿಮಾಣ(ಭಿನ್ನ) && ggml_type_size(dst->type) == ಪರಿಮಾಣ(ಭಿನ್ನ)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_repeat_i32;
        }
        ಯದಿ (ggml_type_size(src0->type) == 2 && ggml_type_size(dst->type) == 2) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_repeat_i16;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_REPEAT_BACK:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_repeat_back_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_CPY:
    ಪಕ್ಷ GGML_OP_CONT:
    ಪಕ್ಷ GGML_OP_DUP:
        ಪ್ರತ್ಯಾಗಮ ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    ಪಕ್ಷ GGML_OP_SET_ROWS:
        {
            ಯದಿ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src_idx = src0->type == GGML_TYPE_F16;
            ಯದಿ (src1->type == GGML_TYPE_I64) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_set_rows_i64[src_idx][dst->type];
            } ಅನ್ಯಥಾ ಯದಿ (src1->type == GGML_TYPE_I32) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_set_rows_i32[src_idx][dst->type];
            }
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    ಪಕ್ಷ GGML_OP_SILU_BACK:
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_silu_back_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_NORM:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_norm_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_GROUP_NORM:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_group_norm_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_RMS_NORM:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಯದಿ (ctx->do_add_rms_partials) {
                ಪ್ರತ್ಯಾಗಮ ctx->fused_rms_norm_mode == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_partials_f32 : ctx->device->pipeline_rms_norm_partials_f32;
            }
            ಪ್ರತ್ಯಾಗಮ ctx->fused_rms_norm_mode == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_RMS_NORM_BACK:
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rms_norm_back_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_L2_NORM:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_l2_norm_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_UNARY:
        ಯದಿ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        ವಿಕಲ್ಪ (ggml_get_unary_op(dst)) {
            ಪಕ್ಷ GGML_UNARY_OP_EXP:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_exp[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_EXPM1:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_expm1[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_ELU:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_elu[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_SILU:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_GELU:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_GELU_ERF:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_GELU_QUICK:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_gelu_quick[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_RELU:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_XIELU:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_xielu[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_NEG:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_neg[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_TANH:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_SIGMOID:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_HARDSIGMOID:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_hardsigmoid[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_HARDSWISH:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_hardswish[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_ABS:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_abs[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_SOFTPLUS:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_softplus[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_STEP:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_step[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_ROUND:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_round[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_CEIL:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_ceil[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_FLOOR:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_floor[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_TRUNC:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_trunc[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_UNARY_OP_SGN:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_sgn[dst->type == GGML_TYPE_F16];
            ಸ್ವಭಾವ:
                ಭಙ್ಗ;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_GLU:
        ಯದಿ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        ವಿಕಲ್ಪ (ggml_get_glu_op(dst)) {
            ಪಕ್ಷ GGML_GLU_OP_GEGLU:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_GLU_OP_REGLU:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_GLU_OP_SWIGLU:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_GLU_OP_SWIGLU_OAI:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_GLU_OP_SWIGLU_CLAMP:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_swiglu_clamp[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_GLU_OP_GEGLU_ERF:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            ಪಕ್ಷ GGML_GLU_OP_GEGLU_QUICK:
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_geglu_quick[dst->type == GGML_TYPE_F16];
            ಸ್ವಭಾವ:
                ಭಙ್ಗ;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_DIAG_MASK_INF:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_diag_mask_inf_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops) {
            uint32_t idx = (uint32_t)ceilf(log2f(ಭಿನ್ನ(dst->ne[0])));
            GGML_ASSERT(idx < ಅಂಕೆ_topk_moe_pipelines);
            // use n_experts from push constant if it's not equal to the power of two spec constant
            ತಾರ್ಕಿಕ use_ತಳ್ಳು = dst->ne[0] != (1u << idx);
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_topk_moe[idx][use_ತಳ್ಳು];
        }

        ಯದಿ (src0->type == GGML_TYPE_F32 && (src1 == nullptr || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_SOFT_MAX_BACK:
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_soft_max_back_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_ROPE:
    ಪಕ್ಷ GGML_OP_ROPE_BACK:
        {
            ಧ್ರುವ ggml_tensor *rope = ctx->ಅಂಕೆ_additional_fused_ops == 2 ? dst->src[0]->src[0] : dst;
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode = ((ಧ್ರುವ int32_t *) rope->op_params)[2];
            ಧ್ರುವ ತಾರ್ಕಿಕ is_neox = mode & GGML_ROPE_TYPE_NEOX;
            ಧ್ರುವ ತಾರ್ಕಿಕ is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            ಧ್ರುವ ತಾರ್ಕಿಕ is_vision = mode == GGML_ROPE_TYPE_VISION;

            ಯದಿ (is_neox) {
                ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_neox_f32;
                }
                ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_neox_f32_f16;
                }
                ಯದಿ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_neox_f16;
                }
            } ಅನ್ಯಥಾ ಯದಿ (is_mrope && !is_vision) {
                ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_multi_f32;
                }
                ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_multi_f32_f16;
                }
                ಯದಿ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_multi_f16;
                }
            } ಅನ್ಯಥಾ ಯದಿ (is_vision) {
                ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_vision_f32;
                }
                ಯದಿ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_vision_f16;
                }
            } ಅನ್ಯಥಾ {
                ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_norm_f32;
                }
                ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_norm_f32_f16;
                }
                ಯದಿ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rope_norm_f16;
                }
            }
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    ಪಕ್ಷ GGML_OP_SUM:
    ಪಕ್ಷ GGML_OP_SUM_ROWS:
    ಪಕ್ಷ GGML_OP_MEAN:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_sum_rows_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_CROSS_ENTROPY_LOSS:
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ src0->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        // src0 is the scalar grad; src1 is logits
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2 && src2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ src1->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_back_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_back_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_CUMSUM:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಯದಿ (src0->ne[0] <= 512) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cumsum_small_f32;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_cumsum_f32;
            }
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_SOLVE_TRI:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

            vk_solve_tri_pipeline_state solve_tri_pipeline_state(src0->ne[0], src1->ne[0]);

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                auto it = ctx->device->pipeline_solve_tri_f32.find(solve_tri_pipeline_state);
                ಯದಿ (it != ctx->device->pipeline_solve_tri_f32.end()) {
                    pipeline = it->second;
                } ಅನ್ಯಥಾ {
                    ctx->device->pipeline_solve_tri_f32[solve_tri_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            ಪ್ರತ್ಯಾಗಮ pipeline;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_ARGMAX:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_argmax_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_ಎಣಿಕೆ_EQUAL:
        ಯದಿ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_ಎಣಿಕೆ_equal_i32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_IM2COL:
        ಯದಿ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_im2col_f32;
        }
        ಯದಿ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_im2col_f32_f16;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_IM2COL_3D:
        ಯದಿ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_im2col_3d_f32;
        }
        ಯದಿ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_im2col_3d_f32_f16;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_TIMESTEP_EMBEDDING:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_timestep_embedding_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_1D:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_conv_transpose_1d_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_COL2IM_1D:
        ವಿಕಲ್ಪ (src0->type) {
            ಪಕ್ಷ GGML_TYPE_F32:  ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_col2im_1d_f32;
            ಪಕ್ಷ GGML_TYPE_F16:  ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_col2im_1d_f16;
            ಪಕ್ಷ GGML_TYPE_BF16: ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_col2im_1d_bf16;
            ಸ್ವಭಾವ:             ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    ಪಕ್ಷ GGML_OP_POOL_1D:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_pool1d_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_POOL_2D:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            // Wide windows use neighboring subgroup lanes to read neighboring inputs.
            ಯದಿ (ctx->device->pipeline_pool2d_subgroup_f32 && dst->op_params[1] >= int32_t(ctx->device->subgroup_size)) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_pool2d_subgroup_f32;
            }
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_pool2d_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_RWKV_WKV6:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rwkv_wkv6_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_RWKV_WKV7:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_rwkv_wkv7_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_GATED_LINEAR_ATTN:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_gated_linear_attn_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_LIGHTNING_INDEXER:
        // only the k type selects a pipeline, the other types are fixed by ggml_lightning_indexer()
        ಯದಿ (ggml_vk_lightning_indexer_k_type_supported(src1->type)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_lightning_indexer_f32[src1->type];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_GATED_DELTA_NET:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಧ್ರುವ uint32_t S_v = dst->src[2]->ne[0];
            ಧ್ರುವ uint32_t kda = (dst->src[3]->ne[0] == (int64_t)S_v) ? 1 : 0;
            uint32_t si;
            ವಿಕಲ್ಪ (S_v) {
                ಪಕ್ಷ 16:  si = 0; ಭಙ್ಗ;
                ಪಕ್ಷ 32:  si = 1; ಭಙ್ಗ;
                ಪಕ್ಷ 64:  si = 2; ಭಙ್ಗ;
                ಪಕ್ಷ 128: si = 3; ಭಙ್ಗ;
                ಸ್ವಭಾವ: ಪ್ರತ್ಯಾಗಮ nullptr;
            }
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_gated_delta_net[si][kda];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_SSM_SCAN:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಧ್ರುವ uint32_t d_state = src0->ne[0];
            ಯದಿ (d_state == 128) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_ssm_scan_f32_d128;
            } ಅನ್ಯಥಾ ಯದಿ (d_state == 256) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_ssm_scan_f32_d256;
            }
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_SSM_CONV:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ವಿಕಲ್ಪ (ctx->ಅಂಕೆ_additional_fused_ops) {
                ಪಕ್ಷ 0:  ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_ssm_conv_f32;
                ಪಕ್ಷ 1:  ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_ssm_conv_silu_f32;
                ಪಕ್ಷ 2:  ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_ssm_conv_bias_silu_f32;
                ಸ್ವಭಾವ: ಪ್ರತ್ಯಾಗಮ nullptr;
            }
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_OPT_STEP_ADAMW:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_opt_step_adamw_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_OPT_STEP_SGD:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_opt_step_sgd_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_LEAKY_RELU:
        ಯದಿ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_leaky_relu[dst->type == GGML_TYPE_F16];
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_CONV_2D:
    ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_2D:
        ಯದಿ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t K = dst->ne[2]; // Cout
            uint32_t NPQ = dst->ne[3] * dst->ne[1] * dst->ne[0]; // N * OH * OW
            vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, K, NPQ);

            ತಾರ್ಕಿಕ transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;
            uint32_t KW = (uint32_t)src0->ne[0];
            uint32_t KH = (uint32_t)src0->ne[1];
            uint32_t s0 = (uint32_t)(ggml_get_op_params_i32(dst, 0));
            uint32_t s1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 1) : s0;
            uint32_t p0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 2) : 0;
            uint32_t p1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 3) : 0;
            uint32_t d0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 4) : 1;
            uint32_t d1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 5) : 1;

            // tile-aligned shapes let the shader skip bounds checks
            ಧ್ರುವ uint32_t Cin = (uint32_t)src1->ne[2];
            ಧ್ರುವ uint32_t CRS = Cin * KW * KH;
            ಧ್ರುವ uint32_t BS_K   = vk_conv_block_sizes[shape].K;
            ಧ್ರುವ uint32_t BS_CRS = vk_conv_block_sizes[shape].CRS;
            ಧ್ರುವ uint32_t BS_NPQ = vk_conv_block_sizes[shape].NPQ;
            ಧ್ರುವ uint32_t aligned = ((K   % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv2d_pipeline_state conv2d_pipeline_state(s0, s1, p0, p1, d0, d1, KW, KH, aligned);

            std::map<vk_conv2d_pipeline_state, vk_pipeline> *pipelines = nullptr;
            ಯದಿ (op == GGML_OP_CONV_2D) {
                ಯದಿ (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv2d_f32[shape];
                } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv2d_f16_f32[shape];
                }
            } ಅನ್ಯಥಾ ಯದಿ (op == GGML_OP_CONV_TRANSPOSE_2D) {
                ಯದಿ (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f32[shape];
                } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f16_f32[shape];
                }
            }

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                auto it = pipelines->find(conv2d_pipeline_state);
                ಯದಿ (it != pipelines->end()) {
                    pipeline = it->second;
                } ಅನ್ಯಥಾ {
                    (*pipelines)[conv2d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            ಪ್ರತ್ಯಾಗಮ pipeline;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_CONV_2D_DW:
        ಯದಿ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಯದಿ (ggml_is_contiguous(src1)) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_conv2d_dw_whcn_f32;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_is_contiguous_channels(src1)) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            ಯದಿ (ggml_is_contiguous(src1)) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_conv2d_dw_whcn_f16_f32;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_is_contiguous_channels(src1)) {
                ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_conv2d_dw_cwhn_f16_f32;
            }
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_CONV_3D:
        ಯದಿ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಧ್ರುವ uint32_t OC = (uint32_t)ggml_get_op_params_i32(dst, 11);
            ಧ್ರುವ uint32_t IC = (uint32_t)ggml_get_op_params_i32(dst, 9);
            ಧ್ರುವ uint32_t N  = (uint32_t)ggml_get_op_params_i32(dst, 10);
            ಧ್ರುವ uint32_t NPQ = N * dst->ne[2] * dst->ne[1] * dst->ne[0];
            ಧ್ರುವ vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, OC, NPQ);

            ಧ್ರುವ uint32_t KW = (uint32_t)src0->ne[0];
            ಧ್ರುವ uint32_t KH = (uint32_t)src0->ne[1];
            ಧ್ರುವ uint32_t KD = (uint32_t)src0->ne[2];
            ಧ್ರುವ uint32_t s0 = (uint32_t)ggml_get_op_params_i32(dst, 0);
            ಧ್ರುವ uint32_t s1 = (uint32_t)ggml_get_op_params_i32(dst, 1);
            ಧ್ರುವ uint32_t s2 = (uint32_t)ggml_get_op_params_i32(dst, 2);
            ಧ್ರುವ uint32_t p0 = (uint32_t)ggml_get_op_params_i32(dst, 3);
            ಧ್ರುವ uint32_t p1 = (uint32_t)ggml_get_op_params_i32(dst, 4);
            ಧ್ರುವ uint32_t p2 = (uint32_t)ggml_get_op_params_i32(dst, 5);
            ಧ್ರುವ uint32_t d0 = (uint32_t)ggml_get_op_params_i32(dst, 6);
            ಧ್ರುವ uint32_t d1 = (uint32_t)ggml_get_op_params_i32(dst, 7);
            ಧ್ರುವ uint32_t d2 = (uint32_t)ggml_get_op_params_i32(dst, 8);

            ಧ್ರುವ uint32_t CRS = IC * KW * KH * KD;
            ಧ್ರುವ uint32_t BS_K   = vk_conv_block_sizes[shape].K;
            ಧ್ರುವ uint32_t BS_CRS = vk_conv_block_sizes[shape].CRS;
            ಧ್ರುವ uint32_t BS_NPQ = vk_conv_block_sizes[shape].NPQ;
            ಧ್ರುವ uint32_t aligned = ((OC  % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv3d_pipeline_state conv3d_pipeline_state(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned);

            std::map<vk_conv3d_pipeline_state, vk_pipeline> *pipelines = nullptr;
            ಯದಿ (src0->type == GGML_TYPE_F32) {
                pipelines = &ctx->device->pipeline_conv3d_f32[shape];
            } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16) {
                pipelines = &ctx->device->pipeline_conv3d_f16_f32[shape];
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                auto it = pipelines->find(conv3d_pipeline_state);
                ಯದಿ (it != pipelines->end()) {
                    pipeline = it->second;
                } ಅನ್ಯಥಾ {
                    (*pipelines)[conv3d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            ಪ್ರತ್ಯಾಗಮ pipeline;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_ADD1:
        ಯದಿ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_add1_f16_f16;
        }
        ಯದಿ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_add1_f16_f32;
        }
        ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_add1_f32_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_ARANGE:
        ಯದಿ (dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_arange_f32;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಪಕ್ಷ GGML_OP_FILL:
        ಯದಿ (dst->type == GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_fill_f32;
        }
        ಯದಿ (dst->type == GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ctx->device->pipeline_fill_f16;
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    GGML_UNUSED(src2);
}

template<typename PC>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ಧ್ರುವ ggml_tensor * src3, ggml_tensor * dst, ggml_op op, PC&& pc, vk_pipeline pipeline_override = nullptr) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    ಯದಿ (src1 != nullptr) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    ಯದಿ (src2 != nullptr) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    ಯದಿ (src3 != nullptr) {
        std::cerr << "), (" << src3 << ", name=" << src3->name << ", type=" << src3->type << ", ne0=" << src3->ne[0] << ", ne1=" << src3->ne[1] << ", ne2=" << src3->ne[2] << ", ne3=" << src3->ne[3] << ", nb0=" << src3->nb[0] << ", nb1=" << src3->nb[1] << ", nb2=" << src3->nb[2] << ", nb3=" << src3->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_ROWS || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == nullptr || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(dst->buffer != nullptr);
    ಧ್ರುವ uint64_t ne00 = src0->ne[0];
    ಧ್ರುವ uint64_t ne01 = src0->ne[1];
    ಧ್ರುವ uint64_t ne02 = src0->ne[2];
    ಧ್ರುವ uint64_t ne03 = src0->ne[3];

    ಧ್ರುವ ತಾರ್ಕಿಕ use_src1 = src1 != nullptr;
    ಧ್ರುವ uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    ಧ್ರುವ uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    ಧ್ರುವ uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    ಧ್ರುವ uint64_t ne13 = use_src1 ? src1->ne[3] : 0;

    ಧ್ರುವ ತಾರ್ಕಿಕ use_src2 = src2 != nullptr;
    ಧ್ರುವ ತಾರ್ಕಿಕ use_src3 = src3 != nullptr;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline;
    ಯದಿ (pipeline_override) {
        pipeline = pipeline_override;
    } ಅನ್ಯಥಾ {
        pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);
    }

    ಯದಿ (pipeline == nullptr) {
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
        ಯದಿ (src1 != nullptr) {
            std::cerr << " and " << ggml_type_name(src1->type);
        }
        std::cerr << " to " << ggml_type_name(dst->type) << std::endl;
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0, ಸತ್ಯ);
    vk_subbuffer src1_buf = use_src1 ? ggml_vk_tensor_subbuffer(ctx, src1, ಸತ್ಯ) : vk_subbuffer{};
    vk_subbuffer src2_buf = use_src2 ? ggml_vk_tensor_subbuffer(ctx, src2, ಸತ್ಯ) : vk_subbuffer{};
    vk_subbuffer src3_buf = use_src3 ? ggml_vk_tensor_subbuffer(ctx, src3, ಸತ್ಯ) : vk_subbuffer{};
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ಸತ್ಯ);

    // Compute misalignment offset for descriptors and store it in in push constants.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, src3, dst);

    std::array<uint32_t, 3> elements;

    ವಿಕಲ್ಪ (op) {
    ಪಕ್ಷ GGML_OP_NORM:
    ಪಕ್ಷ GGML_OP_RMS_NORM_BACK:
    ಪಕ್ಷ GGML_OP_L2_NORM:
    ಪಕ್ಷ GGML_OP_SOFT_MAX:
    ಪಕ್ಷ GGML_OP_SOFT_MAX_BACK:
    ಪಕ್ಷ GGML_OP_SUM_ROWS:
    ಪಕ್ಷ GGML_OP_CUMSUM:
    ಪಕ್ಷ GGML_OP_MEAN:
    ಪಕ್ಷ GGML_OP_ARGMAX:
        {
            ಧ್ರುವ uint32_t nr = ggml_nrows(src0);
            ಯದಿ (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } ಅನ್ಯಥಾ ಯದಿ (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } ಅನ್ಯಥಾ {
                elements = { nr, 1, 1 };
            }
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SOLVE_TRI:
        {
            uint32_t nr = (uint32_t)(ne02 * ne03);
            ಯದಿ (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } ಅನ್ಯಥಾ ಯದಿ (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } ಅನ್ಯಥಾ {
                elements = { nr, 1, 1 };
            }
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_RMS_NORM:
        ಯದಿ (ctx->do_add_rms_partials) {
            // Run one element per thread, 128 threads per workgroup
            elements = { (uint32_t)CEIL_DIV(ne00, 128), 1, 1 };
        } ಅನ್ಯಥಾ {
            elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03 };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]);
        }
        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_SUM:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_GROUP_NORM:
        {
            ಧ್ರುವ uint32_t ಅಂಕೆ_groups = dst->op_params[0];
            elements = { ಅಂಕೆ_groups * (uint32_t)src0->ne[3], 1, 1 };
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_DIAG_MASK_INF:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ROPE:
    ಪಕ್ಷ GGML_OP_ROPE_BACK:
        {
            uint32_t nrows = (uint32_t)ggml_nrows(src0);
            uint32_t z = 1;
            ಯದಿ (nrows > ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[0]) {
                z = CEIL_DIV(nrows, 32768);
                nrows = 32768;
            }
            elements = { nrows, (uint32_t)ne00, z };

        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_GET_ROWS:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]);
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_GET_ROWS_BACK:
        elements = { (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], 1 };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ARGSORT:
        GGML_ASSERT(0);
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_IM2COL:
        {
            ಧ್ರುವ ತಾರ್ಕಿಕ is_2D = dst->op_params[6] == 1;

            ಧ್ರುವ uint32_t IC = src1->ne[is_2D ? 2 : 1];

            ಧ್ರುವ uint32_t KH = is_2D ? src0->ne[1] : 1;
            ಧ್ರುವ uint32_t KW =         src0->ne[0];

            ಧ್ರುವ uint32_t OH = is_2D ? dst->ne[2] : 1;
            ಧ್ರುವ uint32_t OW =         dst->ne[1];

            ಧ್ರುವ uint32_t batch = src1->ne[is_2D ? 3 : 2];

            ಧ್ರುವ uint32_t CHW = IC * KH * KW;
            // Cap X workgroups to limit concurrent IC channel reads.
            // The shader loops over X to cover the full CHW dimension.
            // AMD prefers a lower limit
            ಧ್ರುವ uint32_t min_cap = ctx->device->vendor_id == VK_VENDOR_ID_AMD ? 512u : 4096u;
            ಧ್ರುವ uint32_t x_elements = std::min(CHW, std::max(min_cap, OW * KH * KW));
            elements = { x_elements, OW, OH * batch };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]);
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_IM2COL_3D:
        {
            ಧ್ರುವ uint32_t IC = ((ಧ್ರುವ uint32_t *)(dst->op_params))[9];

            ಧ್ರುವ uint32_t N  = ne13 / IC;

            ಧ್ರುವ uint32_t KD = ne02;
            ಧ್ರುವ uint32_t KH = ne01;
            ಧ್ರುವ uint32_t KW = ne00;

            ಧ್ರುವ uint32_t OD = dst->ne[3] / N;
            ಧ್ರುವ uint32_t OH = dst->ne[2];
            ಧ್ರುವ uint32_t OW = dst->ne[1];

            ಧ್ರುವ uint32_t IC_KD_KH_KW = IC*KD*KH*KW;
            ಧ್ರುವ uint32_t N_OD_OH = N*OD*OH;

            elements = { IC_KD_KH_KW, OW, N_OD_OH };
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]);
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_TIMESTEP_EMBEDDING:
        {
            ಧ್ರುವ uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_COL2IM_1D:
        {
            elements = { uint32_t(dst->ne[0]), uint32_t(dst->ne[1]), 1 };
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_POOL_1D:
        {
            ಧ್ರುವ uint32_t N = dst->ne[3] * dst->ne[2];
            ಧ್ರುವ uint32_t OC = dst->ne[1];
            ಧ್ರುವ uint32_t OL = dst->ne[0];
            elements = { N * OC * OL, 1, 1};
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_POOL_2D:
        {
            ಧ್ರುವ uint32_t outputs = ggml_nelements(dst);
            ಧ್ರುವ uint32_t outputs_per_workgroup = pipeline == ctx->device->pipeline_pool2d_subgroup_f32 ? 128 / ctx->device->subgroup_size : 512;
            ಧ್ರುವ uint32_t workgroups = CEIL_DIV(outputs, outputs_per_workgroup);
            ಧ್ರುವ uint32_t groups_y = CEIL_DIV(workgroups, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[0]);
            ಧ್ರುವ uint32_t groups_x = CEIL_DIV(workgroups, groups_y);
            elements = { groups_x, groups_y, 1 };
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CONV_2D:
    ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_2D:
        ಯದಿ constexpr (std::is_same_v<PC, vk_op_conv2d_ತಳ್ಳು_constants>) {
            ಧ್ರುವ uint32_t NPQ = pc.N * pc.OH * pc.OW;
            ಧ್ರುವ vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.Cout, NPQ);
            ಧ್ರುವ uint32_t NPQ_blocks = CEIL_DIV(NPQ, vk_conv_block_sizes[shape].NPQ);

            elements = { pc.Cout, NPQ_blocks, 1 };
            ಯದಿ (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } ಅನ್ಯಥಾ {
            GGML_ABORT("invalid push constant type for CONV_2D");
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CONV_3D:
        ಯದಿ constexpr (std::is_same_v<PC, vk_op_conv3d_ತಳ್ಳು_constants>) {
            ಧ್ರುವ uint32_t NPQ = pc.N * pc.OD * pc.OH * pc.OW;
            ಧ್ರುವ vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.OC, NPQ);
            ಧ್ರುವ uint32_t NPQ_blocks = CEIL_DIV(NPQ, vk_conv_block_sizes[shape].NPQ);

            elements = { pc.OC, NPQ_blocks, 1 };
            ಯದಿ (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } ಅನ್ಯಥಾ {
            GGML_ABORT("invalid push constant type for CONV_3D");
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ADD:
    ಪಕ್ಷ GGML_OP_SUB:
    ಪಕ್ಷ GGML_OP_DIV:
    ಪಕ್ಷ GGML_OP_MUL:
    ಪಕ್ಷ GGML_OP_ADD1:
    ಪಕ್ಷ GGML_OP_OUT_PROD:
    ಪಕ್ಷ GGML_OP_ARANGE:
    ಪಕ್ಷ GGML_OP_FILL:
    ಪಕ್ಷ GGML_OP_SCALE:
    ಪಕ್ಷ GGML_OP_SQR:
    ಪಕ್ಷ GGML_OP_SQRT:
    ಪಕ್ಷ GGML_OP_SIN:
    ಪಕ್ಷ GGML_OP_COS:
    ಪಕ್ಷ GGML_OP_LOG:
    ಪಕ್ಷ GGML_OP_TRI:
    ಪಕ್ಷ GGML_OP_DIAG:
    ಪಕ್ಷ GGML_OP_CLAMP:
    ಪಕ್ಷ GGML_OP_LEAKY_RELU:
    ಪಕ್ಷ GGML_OP_PAD:
    ಪಕ್ಷ GGML_OP_PAD_REFLECT_1D:
    ಪಕ್ಷ GGML_OP_ROLL:
    ಪಕ್ಷ GGML_OP_REPEAT:
    ಪಕ್ಷ GGML_OP_REPEAT_BACK:
    ಪಕ್ಷ GGML_OP_CPY:
    ಪಕ್ಷ GGML_OP_CONCAT:
    ಪಕ್ಷ GGML_OP_UPSCALE:
    ಪಕ್ಷ GGML_OP_UNARY:
    ಪಕ್ಷ GGML_OP_GLU:
    ಪಕ್ಷ GGML_OP_CONV_2D_DW:
        {
            uint32_t ne = ggml_nelements(dst);
            ಯದಿ (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_size(src0->type);
                ಯದಿ ((ggml_type_size(src0->type) % 4) == 0) {
                    ne *= ggml_type_size(src0->type) / 4;
                } ಅನ್ಯಥಾ {
                    ne *= ggml_type_size(src0->type) / 2;
                }
            }
            ಯದಿ (op == GGML_OP_CONCAT && ggml_is_quantized(dst->type)) {
                ne = ne / ggml_blck_size(dst->type) * ggml_type_size(dst->type) / ggml_vk_concat_unit_size(dst->type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            ಯದಿ (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_size(dst->type));
            }
            ಯದಿ (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } ಅನ್ಯಥಾ ಯದಿ (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } ಅನ್ಯಥಾ {
                elements = { ne, 1, 1 };
            }

            ಯದಿ (pipeline == ctx->device->pipeline_cpy_transpose_02_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_02_16) {
                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[2], 32);
                elements[2] = (uint32_t)(dst->ne[1]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]);
            } ಅನ್ಯಥಾ ಯದಿ (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[1], 32);
                elements[2] = (uint32_t)(dst->ne[2]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]);
            }
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ADD_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SET_ROWS:
        {
            uint32_t ne = ggml_nelements(src0);
            ಯದಿ (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_size(dst->type));
            } ಅನ್ಯಥಾ {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            ಯದಿ (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } ಅನ್ಯಥಾ ಯದಿ (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } ಅನ್ಯಥಾ {
                elements = { ne, 1, 1 };
            }
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SSM_CONV:
        {
            ಧ್ರುವ uint32_t nr  = src0->ne[1];
            ಧ್ರುವ uint32_t n_t = dst->ne[1];
            ಧ್ರುವ uint32_t n_s = dst->ne[2];
            elements = { nr, n_t, n_s };
        }
        ಭಙ್ಗ;
    ಸ್ವಭಾವ:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        ಭಙ್ಗ;
    }

    ಯದಿ (op == GGML_OP_ADD || op == GGML_OP_RMS_NORM) {
        vk_subbuffer a_buf = src0_buf;
        ಯದಿ (ctx->do_add_rms_partials) {
            a_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_add_rms_partials, ctx->prealloc_size_add_rms_partials_offset);
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, src1_buf, dst_buf, a_buf }, pc, elements);
    } ಅನ್ಯಥಾ ಯದಿ (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc, elements);
    } ಅನ್ಯಥಾ ಯದಿ (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, subbuf2, dst_buf }, pc, elements);
    } ಅನ್ಯಥಾ ಯದಿ (op == GGML_OP_ROPE || op == GGML_OP_ROPE_BACK) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        vk_subbuffer subbuf3 = use_src3 ? src3_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, subbuf2, dst_buf, subbuf3 }, pc, elements);
    } ಅನ್ಯಥಾ ಯದಿ (op == GGML_OP_IM2COL || op == GGML_OP_IM2COL_3D) {
        ಯದಿ (ctx->device->shader_int64 && ctx->device->buffer_device_address) {
            // buffer device address path doesn't use dst buffer
            dst_buf.size = 1;
        }
        // im2col uses only src1 and dst buffers
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src1_buf, dst_buf }, pc, elements);
    } ಅನ್ಯಥಾ ಯದಿ (op == GGML_OP_ಎಣಿಕೆ_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_buffer_memset_async(subctx, dst_buf.buffer, dst_buf.offset, 0, dst_buf.size);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } ಅನ್ಯಥಾ ಯದಿ (op == GGML_OP_OPT_STEP_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf }, pc, elements);
    } ಅನ್ಯಥಾ ಯದಿ (use_src3) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pc, elements);
    } ಅನ್ಯಥಾ ಯದಿ (use_src2) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, dst_buf }, pc, elements);
    } ಅನ್ಯಥಾ ಯದಿ (use_src1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } ಅನ್ಯಥಾ {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, dst_buf }, pc, elements);
    }
}

ಶೂನ್ಯ ggml_vk_get_rows(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_GET_ROWS, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_get_rows_back(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_GET_ROWS_BACK, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2], (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ಪೂರ್ಣಾಙ್ಕ nb1 = dst->op_params[0] / src0_type_size; // 4 bytes of float32
    ಪೂರ್ಣಾಙ್ಕ nb2 = dst->op_params[1] / src0_type_size; // 4 bytes of float32
    ಪೂರ್ಣಾಙ್ಕ nb3 = dst->op_params[2] / src0_type_size; // 4 bytes of float32
    ಪೂರ್ಣಾಙ್ಕ offset = dst->op_params[3] / src0_type_size; // offset in bytes

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, dst->op, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        0,
        0.0f, 0.0f, offset,
    });
}

ಶೂನ್ಯ ggml_vk_multi_add(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ggml_tensor *first_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];
    ಧ್ರುವ ggml_tensor *dst = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops];

    // Make a list of all the tensors used by the op.
    // Last element of the list is the dest tensor.
    ಧ್ರುವ ggml_tensor *tensors[MAX_PARAMETER_ಎಣಿಕೆ];
    uint32_t ಅಂಕೆ_srcs = ctx->ಅಂಕೆ_additional_fused_ops + 2;
    uint32_t ಅಂಕೆ_tensors = ಅಂಕೆ_srcs + 1;
    GGML_ASSERT(ಅಂಕೆ_tensors + ctx->do_add_rms_partials <= MAX_PARAMETER_ಎಣಿಕೆ);

    tensors[0] = first_ಗಂಟು->src[0];
    tensors[1] = first_ಗಂಟು->src[1];
    ಪ್ರತಿ (int32_t i = 0; i < ctx->ಅಂಕೆ_additional_fused_ops; ++i) {
        // check whether the previous result is src[0] or src[1]
        ಯದಿ (cgraph->nodes[ಗಂಟು_idx + i] == cgraph->nodes[ಗಂಟು_idx + i + 1]->src[0]) {
            tensors[i+2] = cgraph->nodes[ಗಂಟು_idx + i + 1]->src[1];
        } ಅನ್ಯಥಾ {
            tensors[i+2] = cgraph->nodes[ಗಂಟು_idx + i + 1]->src[0];
        }
    }
    tensors[ಅಂಕೆ_srcs] = dst;

    vk_op_multi_add_ತಳ್ಳು_constants pc;
    pc.ne20 = (uint32_t)dst->ne[0];
    pc.ne21 = (uint32_t)dst->ne[1];
    pc.ne22 = (uint32_t)dst->ne[2];
    pc.ne23 = (uint32_t)dst->ne[3];

    ಪ್ರತಿ (uint32_t i = 0; i < ಅಂಕೆ_tensors; ++i) {
        ಧ್ರುವ ggml_tensor *t = tensors[i];
        pc.nb[i][0] = (uint32_t)t->nb[0] / ಪರಿಮಾಣ(ಭಿನ್ನ);
        pc.nb[i][1] = (uint32_t)t->nb[1] / ಪರಿಮಾಣ(ಭಿನ್ನ);
        pc.nb[i][2] = (uint32_t)t->nb[2] / ಪರಿಮಾಣ(ಭಿನ್ನ);
        pc.nb[i][3] = (uint32_t)t->nb[3] / ಪರಿಮಾಣ(ಭಿನ್ನ);
    }
    pc.rms_partials = ctx->do_add_rms_partials;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, tensors[0], tensors[1], nullptr, dst, dst->op);

    ಯದಿ (pipeline == nullptr) {
        std::cerr << "ggml_vulkan: Error: Missing multi_add";
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ggml_backend_vk_buffer_context * buf_ctx[MAX_PARAMETER_ಎಣಿಕೆ];
    vk_buffer buf[MAX_PARAMETER_ಎಣಿಕೆ];
    size_t offset[MAX_PARAMETER_ಎಣಿಕೆ];
    ತಾರ್ಕಿಕ uma[MAX_PARAMETER_ಎಣಿಕೆ];

    ಪ್ರತಿ (uint32_t i = 0; i < ಅಂಕೆ_tensors; ++i) {
        buf_ctx[i] = (ggml_backend_vk_buffer_context *)tensors[i]->buffer->context;
        buf[i] = nullptr;
        offset[i] = 0;
        uma[i] = ಅಸತ್ಯ;

        ಯದಿ (ctx->device->uma) {
            ggml_vk_host_get(ctx->device, tensors[i]->data, buf[i], offset[i]);
            uma[i] = buf[i] != nullptr;
        }
        ಯದಿ (!uma[i]) {
            buf[i] = buf_ctx[i]->dev_buffer;
            offset[i] = vk_tensor_offset(tensors[i]) + tensors[i]->view_offs;
        }
        GGML_ASSERT(buf[i] != nullptr);
    }
    // If any remaining descriptors are unused, just point them at src[0]
    ಪ್ರತಿ (uint32_t i = ಅಂಕೆ_tensors; i < MAX_PARAMETER_ಎಣಿಕೆ; ++i) {
        buf[i] = buf[0];
        offset[i] = 0;
    }
    ಯದಿ (ctx->do_add_rms_partials) {
        buf[ಅಂಕೆ_tensors] = ctx->prealloc_add_rms_partials;
        offset[ಅಂಕೆ_tensors] = ctx->prealloc_size_add_rms_partials_offset;
    }

    std::array<uint32_t, 3> elements;

    uint32_t ne = ggml_nelements(dst);
    ಯದಿ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } ಅನ್ಯಥಾ ಯದಿ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } ಅನ್ಯಥಾ {
        elements = { ne, 1, 1 };
    }

    static_assert(MAX_PARAMETER_ಎಣಿಕೆ == 12);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            ggml_vk_subbuffer(ctx, buf[0], offset[0]),
            ggml_vk_subbuffer(ctx, buf[1], offset[1]),
            ggml_vk_subbuffer(ctx, buf[2], offset[2]),
            ggml_vk_subbuffer(ctx, buf[3], offset[3]),
            ggml_vk_subbuffer(ctx, buf[4], offset[4]),
            ggml_vk_subbuffer(ctx, buf[5], offset[5]),
            ggml_vk_subbuffer(ctx, buf[6], offset[6]),
            ggml_vk_subbuffer(ctx, buf[7], offset[7]),
            ggml_vk_subbuffer(ctx, buf[8], offset[8]),
            ggml_vk_subbuffer(ctx, buf[9], offset[9]),
            ggml_vk_subbuffer(ctx, buf[10], offset[10]),
            ggml_vk_subbuffer(ctx, buf[11], offset[11]),
        }, pc, elements);
}

ಶೂನ್ಯ ggml_vk_add(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_ADD, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, ctx->do_add_rms_partials,
    });
}

ಶೂನ್ಯ ggml_vk_out_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_OUT_PROD, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3],
        (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3],
        (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3],
        (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SUB, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_MUL, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

ಪೂರ್ಣಾಙ್ಕ ggml_vk_unary_mul_op_index(ggml_unary_op op) {
    ವಿಕಲ್ಪ (op) {
        ಪಕ್ಷ GGML_UNARY_OP_GELU:     ಪ್ರತ್ಯಾಗಮ 0;
        ಪಕ್ಷ GGML_UNARY_OP_SIGMOID:  ಪ್ರತ್ಯಾಗಮ 1;
        ಪಕ್ಷ GGML_UNARY_OP_SILU:     ಪ್ರತ್ಯಾಗಮ 2;
        ಪಕ್ಷ GGML_UNARY_OP_SOFTPLUS: ಪ್ರತ್ಯಾಗಮ 3;
        ಸ್ವಭಾವ:                     ಪ್ರತ್ಯಾಗಮ -1;
    }
}

ಶೂನ್ಯ ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ggml_tensor * unary = cgraph->nodes[ಗಂಟು_idx];
    ggml_tensor * mul = cgraph->nodes[ಗಂಟು_idx + 1];

    // unary on src1 that tiles into src0
    ಧ್ರುವ ತಾರ್ಕಿಕ op_on_b = mul->src[1] == unary &&
                         !ggml_are_same_shape(unary->src[0], mul->src[0]) &&
                         ggml_can_repeat(unary, mul->src[0]);

    ಧ್ರುವ ggml_tensor * src0 = op_on_b ? mul->src[0] : unary->src[0];
    ಧ್ರುವ ggml_tensor * src1 = op_on_b ? unary->src[0] :
        ((mul->src[0] == unary) ? mul->src[1] : mul->src[0]);

    ಧ್ರುವ ತಾರ್ಕಿಕ f16 = src0->type == GGML_TYPE_F16;
    ಧ್ರುವ ತಾರ್ಕಿಕ norepeat = ggml_are_same_shape(src0, src1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ oi = ggml_vk_unary_mul_op_index(ggml_get_unary_op(unary));
    ಯದಿ (oi < 0) {
        GGML_ABORT("fatal error");
    }
    vk_pipeline pipeline = ctx->device->pipeline_unary_mul[oi][f16][norepeat][op_on_b];

    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(mul->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, mul, GGML_OP_UNARY, {
        (uint32_t)ggml_nelements(op_on_b ? mul : src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) mul->ne[0], (uint32_t) mul->ne[1], (uint32_t) mul->ne[2],(uint32_t) mul->ne[3], (uint32_t) mul->nb[0] /  dst_type_size, (uint32_t) mul->nb[1] /  dst_type_size, (uint32_t) mul->nb[2] /  dst_type_size, (uint32_t) mul->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    }, pipeline);
}

ಶೂನ್ಯ ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_DIV, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_add_id(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t src2_type_size = ggml_type_size(src2->type);

    ggml_vk_op_f32<vk_op_add_id_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_ADD_ID, {
        (uint32_t)dst->ne[0],
        (uint32_t)dst->ne[1],
        (uint32_t)src0->nb[1] / src0_type_size,
        (uint32_t)src0->nb[2] / src0_type_size,
        (uint32_t)src1->nb[1] / src1_type_size,
        (uint32_t)src2->nb[1] / src2_type_size,
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_op_f32_wkv(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ಧ್ರುವ vk_op_rwkv_wkv6_ತಳ್ಳು_constants&& pc, ಪೂರ್ಣಾಙ್ಕ version) {
    GGML_ASSERT(version == 6 || version == 7);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_srcs = version == 6 ? 6 : 7;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_srcs; i++) {
        GGML_ASSERT(!ggml_is_quantized(dst->src[i]->type));
    }

    GGML_ASSERT(dst->buffer != nullptr);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_srcs; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements = {
        (uint32_t)(pc.B * pc.H),
        1,
        1
    };

    ಯದಿ (version == 6) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
            pc, elements);
    } ಅನ್ಯಥಾ ಯದಿ (version == 7) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
            pc, elements);
    } ಅನ್ಯಥಾ {
        // shouldn't happen
        GGML_ASSERT(ಅಸತ್ಯ);
    }
}

ಶೂನ್ಯ ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಧ್ರುವ size_t seq_length = dst->src[0]->ne[2];
    ಧ್ರುವ size_t n_embed = dst->ne[0];
    ಧ್ರುವ size_t n_heads = dst->src[0]->ne[1];
    ಧ್ರುವ size_t n_seqs = dst->src[5]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_length,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        6
    );
}

ಶೂನ್ಯ ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಧ್ರುವ size_t seq_length = dst->src[0]->ne[2];
    ಧ್ರುವ size_t n_embed = dst->ne[0];
    ಧ್ರುವ size_t n_heads = dst->src[0]->ne[1];
    ಧ್ರುವ size_t n_seqs = dst->src[6]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_length,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        7
    );
}

ಶೂನ್ಯ ggml_vk_gated_linear_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಧ್ರುವ size_t seq_length = dst->src[0]->ne[2];
    ಧ್ರುವ size_t n_embed    = dst->ne[0];
    ಧ್ರುವ size_t n_heads    = dst->src[0]->ne[1];
    ಧ್ರುವ size_t n_seqs     = dst->src[4]->ne[1];

    ಭಿನ್ನ scale;
    memcpy(&scale, dst->op_params, ಪರಿಮಾಣ(ಭಿನ್ನ));

    GGML_ASSERT(dst->buffer != nullptr);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[5] = {};
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 5; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ಧ್ರುವ vk_op_gated_linear_attn_ತಳ್ಳು_constants pc = {
        (uint32_t)n_seqs,
        (uint32_t)seq_length,
        (uint32_t)n_embed,
        (uint32_t)n_heads,
        scale,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], dst_buf},
        pc, { (uint32_t)(n_seqs * n_heads), 1, 1 });
}

ಶೂನ್ಯ ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * q = dst->src[0];
    ಧ್ರುವ ggml_tensor * k = dst->src[1];
    ಧ್ರುವ ggml_tensor * w = dst->src[2];
    ಧ್ರುವ ggml_tensor * m = dst->src[3];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, w, dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಧ್ರುವ uint32_t n_kv      = k->ne[2];
    ಧ್ರುವ uint32_t n_heads   = q->ne[1];
    ಧ್ರುವ uint32_t n_tokens  = q->ne[2];
    ಧ್ರುವ uint32_t n_streams = q->ne[3];
    ಧ್ರುವ uint32_t n_masks   = m->ne[3];

    // one workgroup per tile of 64 keys and 8 tokens, see lightning_indexer.comp
    ಧ್ರುವ uint32_t n_tiles_kv = CEIL_DIV(n_kv, 64);
    ಧ್ರುವ uint32_t n_tiles_t  = CEIL_DIV(n_tokens, 8);

    // q, w and dst are f32 and m is f16, so their strides are passed in elements;
    // k may be quantized, so its strides stay in bytes
    ಧ್ರುವ uint32_t q_nb1 = q->nb[1] / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t q_nb2 = q->nb[2] / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t q_nb3 = q->nb[3] / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t k_nb2 = k->nb[2];
    ಧ್ರುವ uint32_t k_nb3 = k->nb[3];
    ಧ್ರುವ uint32_t w_nb1 = w->nb[1] / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t w_nb3 = w->nb[3] / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t m_nb1 = m->nb[1] / ಪರಿಮಾಣ(ggml_fp16_t);
    ಧ್ರುವ uint32_t m_nb3 = m->nb[3] / ಪರಿಮಾಣ(ggml_fp16_t);
    ಧ್ರುವ uint32_t d_nb1 = dst->nb[1] / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t d_nb3 = dst->nb[3] / ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಧ್ರುವ vk_op_lightning_indexer_ತಳ್ಳು_constants pc = {
        n_kv, n_heads, n_tokens, n_masks,
        q_nb1, q_nb2, q_nb3,
        k_nb2, k_nb3,
        w_nb1, w_nb3,
        m_nb1, m_nb3,
        d_nb1, d_nb3,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {ggml_vk_tensor_subbuffer(ctx, q), ggml_vk_tensor_subbuffer(ctx, k), ggml_vk_tensor_subbuffer(ctx, w), ggml_vk_tensor_subbuffer(ctx, m), ggml_vk_tensor_subbuffer(ctx, dst)},
        pc, {n_tiles_kv, n_tiles_t, n_streams});
}

ಶೂನ್ಯ ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src_q     = dst->src[0];
    ಧ್ರುವ ggml_tensor * src_v     = dst->src[2];
    ಧ್ರುವ ggml_tensor * src_beta  = dst->src[4];

    GGML_ASSERT(dst->buffer != nullptr);

    ಧ್ರುವ uint32_t S_v      = (uint32_t)src_v->ne[0];
    ಧ್ರುವ uint32_t H        = (uint32_t)src_v->ne[1];
    ಧ್ರುವ uint32_t n_tokens = (uint32_t)src_v->ne[2];
    ಧ್ರುವ uint32_t n_seqs   = (uint32_t)src_v->ne[3];

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ಧ್ರುವ uint32_t K = (uint32_t)ggml_get_op_params_i32(dst, 0);

    ಧ್ರುವ uint32_t s_off = S_v * H * n_tokens * n_seqs;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[6] = {};
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 6; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ಧ್ರುವ uint32_t sq1 = (uint32_t)(src_q->nb[1] / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ uint32_t sq2 = (uint32_t)(src_q->nb[2] / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ uint32_t sq3 = (uint32_t)(src_q->nb[3] / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ uint32_t sv1 = (uint32_t)(src_v->nb[1] / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ uint32_t sv2 = (uint32_t)(src_v->nb[2] / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ uint32_t sv3 = (uint32_t)(src_v->nb[3] / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ uint32_t sb1 = (uint32_t)(src_beta->nb[1] / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ uint32_t sb2 = (uint32_t)(src_beta->nb[2] / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ uint32_t sb3 = (uint32_t)(src_beta->nb[3] / ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಧ್ರುವ uint32_t neq1 = (uint32_t)src_q->ne[1];
    ಧ್ರುವ uint32_t rq3  = (uint32_t)(src_v->ne[3] / src_q->ne[3]);

    ಧ್ರುವ ಭಿನ್ನ scale = 1.0f / sqrtf((ಭಿನ್ನ)S_v);
    ಧ್ರುವ vk_op_gated_delta_net_ತಳ್ಳು_constants pc = {
        H, n_tokens, n_seqs, s_off,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        neq1, rq3,
        scale,
        K
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
        pc, { H, n_seqs, S_v });
}

ಶೂನ್ಯ ggml_vk_ssm_scan(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];
    ಧ್ರುವ ggml_tensor * src2 = dst->src[2];
    ಧ್ರುವ ggml_tensor * src3 = dst->src[3];
    ಧ್ರುವ ggml_tensor * src4 = dst->src[4];
    ಧ್ರುವ ggml_tensor * src5 = dst->src[5];

    GGML_ASSERT(dst->buffer != nullptr);

    ಧ್ರುವ uint32_t head_dim = src0->ne[1];
    ಧ್ರುವ uint32_t n_head = src1->ne[1];
    ಧ್ರುವ uint32_t n_group = src4->ne[1];
    ಧ್ರುವ uint32_t n_tok = src1->ne[2];
    ಧ್ರುವ uint32_t n_seq = src1->ne[3];

    ತಾರ್ಕಿಕ is_mamba2 = (src3->nb[1] == ಪರಿಮಾಣ(ಭಿನ್ನ));
    GGML_ASSERT(is_mamba2);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಧ್ರುವ int64_t s_off = ggml_nelements(src1) * ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಧ್ರುವ vk_op_ssm_scan_ತಳ್ಳು_constants pc = {
        (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        (uint32_t)src3->nb[1],
        (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
        (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
        (uint32_t)s_off,
        n_head, head_dim, n_group, n_tok,
        n_seq, (uint32_t) ggml_get_op_params_i32(dst, 0)
    };

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 7 && dst->src[i] != nullptr; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements;

    ಧ್ರುವ uint32_t d_state = src0->ne[0];
    uint32_t ಅಂಕೆ_subgroups = d_state / ctx->device->subgroup_size;
    ಧ್ರುವ uint32_t ಅಂಕೆ_workgroups_x = CEIL_DIV(n_head * head_dim, ಅಂಕೆ_subgroups);
    ಧ್ರುವ uint32_t ಅಂಕೆ_workgroups_y = n_seq;
    elements = { ಅಂಕೆ_workgroups_x, ಅಂಕೆ_workgroups_y, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
        pc, elements);
}

ಶೂನ್ಯ ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ggml_tensor * conv = cgraph->nodes[ಗಂಟು_idx];
    ಧ್ರುವ ggml_tensor * src0 = conv->src[0];
    ಧ್ರುವ ggml_tensor * src1 = conv->src[1];

    // Pick the destination tensor (last node in the fused chain) and the optional bias.
    // Fusion modes: 0 = ssm_conv, 1 = ssm_conv+silu, 2 = ssm_conv+add(bias)+silu.
    ggml_tensor * dst = conv;
    ಧ್ರುವ ggml_tensor * bias = nullptr;

    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops == 1) {
        dst = cgraph->nodes[ಗಂಟು_idx + 1]; // silu
    } ಅನ್ಯಥಾ ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops == 2) {
        ggml_tensor * add = cgraph->nodes[ಗಂಟು_idx + 1];
        bias = (add->src[0] == conv) ? add->src[1] : add->src[0];
        dst = cgraph->nodes[ಗಂಟು_idx + 2]; // silu
    }

    // The shader always declares 4 bindings; bind src0 as a dummy when bias isn't fused.
    ಧ್ರುವ ggml_tensor * src2 = bias ? bias : src0;

    ggml_vk_op_f32<vk_op_ssm_conv_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_SSM_CONV, {
        (uint32_t)src0->nb[1], (uint32_t)src0->nb[2],
        (uint32_t)src1->nb[1],
        (uint32_t)dst->nb[0], (uint32_t)dst->nb[1], (uint32_t)dst->nb[2],
        (uint32_t)src1->ne[0],
        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)dst->ne[1],
        (uint32_t)dst->ne[2],
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_op_f32_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ಧ್ರುವ vk_op_ತಳ್ಳು_constants&& pc) {
    ಧ್ರುವ ggml_tensor * x = dst->src[0];
    ಧ್ರುವ ggml_tensor * g = dst->src[1];
    ಧ್ರುವ ggml_tensor * gm = dst->src[2];
    ಧ್ರುವ ggml_tensor * gv = dst->src[3];
    ಧ್ರುವ ggml_tensor * p = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(gm->type == GGML_TYPE_F32);
    GGML_ASSERT(gv->type == GGML_TYPE_F32);
    GGML_ASSERT(p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->buffer != nullptr);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(gm));
    GGML_ASSERT(ggml_is_contiguous(gv));
    GGML_ASSERT(ggml_is_contiguous(p));
    GGML_ASSERT(ggml_are_same_shape(x, g));
    GGML_ASSERT(ggml_are_same_shape(x, gm));
    GGML_ASSERT(ggml_are_same_shape(x, gv));
    GGML_ASSERT(ggml_nelements(p) == 7);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, g, gm, gv, dst, GGML_OP_OPT_STEP_ADAMW);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer g_buf = ggml_vk_tensor_subbuffer(ctx, g);
    vk_subbuffer gm_buf = ggml_vk_tensor_subbuffer(ctx, gm);
    vk_subbuffer gv_buf = ggml_vk_tensor_subbuffer(ctx, gv);
    vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(x), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {x_buf, g_buf, gm_buf, gv_buf, p_buf},
        pc, elements);
}

ಶೂನ್ಯ ggml_vk_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಧ್ರುವ size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32_opt_step_adamw(
        ctx, subctx, dst,
        { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f }
    );
}

ಶೂನ್ಯ ggml_vk_opt_step_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ggml_tensor * dst) {
    ಧ್ರುವ size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_OPT_STEP_SGD, { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಪೂರ್ಣಾಙ್ಕ * op_params = (ಪೂರ್ಣಾಙ್ಕ *)dst->op_params;

    ಧ್ರುವ uint32_t unit_size = ggml_vk_concat_unit_size(dst->type);
    ಧ್ರುವ uint32_t units_per_block = ggml_type_size(dst->type) / unit_size;
    ಧ್ರುವ uint32_t block_size = ggml_blck_size(dst->type);
    ಧ್ರುವ ತಾರ್ಕಿಕ quantized = ggml_is_quantized(dst->type);

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    ಧ್ರುವ uint32_t ne00 = src0->ne[0] / block_size * units_per_block;
    ಧ್ರುವ uint32_t ne10 = src1->ne[0] / block_size * units_per_block;
    ಧ್ರುವ uint32_t ne20 =  dst->ne[0] / block_size * units_per_block;
    ಧ್ರುವ uint32_t nb00 = quantized ? 1 : src0->nb[0] / unit_size;
    ಧ್ರುವ uint32_t nb10 = quantized ? 1 : src1->nb[0] / unit_size;
    ಧ್ರುವ uint32_t nb20 = quantized ? 1 :  dst->nb[0] / unit_size;

    vk_op_concat_ತಳ್ಳು_constants pc {{
        ne20 * (uint32_t)dst->ne[1] * (uint32_t)dst->ne[2] * (uint32_t)dst->ne[3],
        ne00, (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], nb00, (uint32_t)src0->nb[1] / unit_size, (uint32_t)src0->nb[2] / unit_size, (uint32_t)src0->nb[3] / unit_size,
        ne10, (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], nb10, (uint32_t)src1->nb[1] / unit_size, (uint32_t)src1->nb[2] / unit_size, (uint32_t)src1->nb[3] / unit_size,
        ne20, (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], nb20, (uint32_t) dst->nb[1] / unit_size, (uint32_t) dst->nb[2] / unit_size, (uint32_t) dst->nb[3] / unit_size,
        0,
        0.0f, 0.0f, op_params[0],
    }};
    ggml_vk_op_f32<vk_op_concat_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONCAT, std::move(pc));
}

ಶೂನ್ಯ ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t mode = (uint32_t)ggml_get_op_params_i32(dst, 0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ಭಿನ್ನ sf0 = (ಭಿನ್ನ)ne0 / ne00;
    ಭಿನ್ನ sf1 = (ಭಿನ್ನ)ne1 / ne01;
    ಭಿನ್ನ sf2 = (ಭಿನ್ನ)ne2 / ne02;
    ಭಿನ್ನ sf3 = (ಭಿನ್ನ)ne3 / ne03;
    ಭಿನ್ನ pixel_offset = 0.5f;

    ಯದಿ (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = ne0 > 1 && ne00 > 1 ? (ಭಿನ್ನ)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (ಭಿನ್ನ)(ne1 - 1) / (ne01 - 1) : sf1;
        pixel_offset = 0.0f;
    }

    ggml_vk_op_f32<vk_op_upscale_ತಳ್ಳು_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0, 0,
        (uint32_t)ne00, (uint32_t)ne01,
        (uint32_t)nb00 / src0_type_size, (uint32_t)nb01 / src0_type_size, (uint32_t)nb02 / src0_type_size, (uint32_t)nb03 / src0_type_size,
        (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
        sf0, sf1, sf2, sf3, pixel_offset
    });
}

ಶೂನ್ಯ ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SCALE, std::move(p));
}

ಶೂನ್ಯ ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SQR, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SQRT, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_ADD1, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_arange(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");

    vk_op_ತಳ್ಳು_constants pc = {
        (uint32_t)ggml_nelements(dst),
        1,
        ggml_get_op_params_f32(dst, 0),
        ggml_get_op_params_f32(dst, 2),
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, nullptr, nullptr, nullptr, dst, GGML_OP_ARANGE);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ಅಸತ್ಯ);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(dst), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

ಶೂನ್ಯ ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_fill(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");
    ಧ್ರುವ uint64_t n = ggml_nelements(dst);
    GGML_ASSERT(n > 0);

    vk_op_ತಳ್ಳು_constants pc = {
        (uint32_t)n,
        1,
        ggml_get_op_params_f32(dst, 0),
        0.0f,
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, nullptr, nullptr, nullptr, dst, GGML_OP_FILL);
    GGML_ASSERT(pipeline != nullptr);

    // Split the task distribution to 2D to avoid exceeding maxComputeWorkGroupCount
    ಧ್ರುವ uint32_t total_wg = CEIL_DIV(n, pipeline->wg_denoms[0]);
    ಧ್ರುವ uint32_t wg_x = std::min(total_wg, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[0]);
    ಧ್ರುವ uint32_t wg_y = CEIL_DIV(total_wg, wg_x);
    GGML_ASSERT(wg_y <= ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ಅಸತ್ಯ);

    std::array<uint32_t, 3> elements = { wg_x * pipeline->wg_denoms[0], wg_y * pipeline->wg_denoms[1], 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

ಶೂನ್ಯ ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SIN, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_COS, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_LOG, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_TRI, std::move(p));
}

ಶೂನ್ಯ ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst, ggml_nelements(dst));

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_DIAG, std::move(p));
}

ಶೂನ್ಯ ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_CLAMP, std::move(p));
}

ಶೂನ್ಯ ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_pad_ತಳ್ಳು_constants p = vk_op_pad_ತಳ್ಳು_constants_init(src0, dst);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_PAD, std::move(p));
}

ಶೂನ್ಯ ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t p0 = (uint32_t)dst->op_params[0];
    ಧ್ರುವ uint32_t p1 = (uint32_t)dst->op_params[1];

    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst, ggml_nelements(dst));
    memcpy(&p.param1, &p0, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&p.param2, &p1, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_PAD_REFLECT_1D, std::move(p));
}

ಶೂನ್ಯ ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ಧ್ರುವ int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ಧ್ರುವ int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ಧ್ರುವ int32_t s2 = ggml_get_op_params_i32(dst, 2);
    ಧ್ರುವ int32_t s3 = ggml_get_op_params_i32(dst, 3);
    ಧ್ರುವ uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
    ಧ್ರುವ uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    memcpy(&p.param1, &s01_packed, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&p.param2, &s23_packed, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_ROLL, std::move(p));
}

ಶೂನ್ಯ ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_REPEAT, std::move(p));
}

ಶೂನ್ಯ ggml_vk_repeat_back(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_REPEAT_BACK, std::move(p));
}

ಶೂನ್ಯ ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t)ggml_nelements(src0);
    ಯದಿ (ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= ggml_blck_size(src0->type);
        ಯದಿ ((ggml_type_size(src0->type) % 4) == 0) {
            ne *= ggml_type_size(src0->type) / 4;
        } ಅನ್ಯಥಾ {
            ne *= ggml_type_size(src0->type) / 2;
        }
    }

    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst, ne);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_CPY, std::move(p));
}

ಶೂನ್ಯ ggml_vk_set_rows(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    // Skip empty skip_rows operations. For most ops the empty check at the start
    // of ggml_vk_build_graph is sufficient, but set_rows can have a nonempty dst
    // with empty srcs.
    ಯದಿ (ggml_is_empty(src0) || ggml_is_empty(src1)) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SET_ROWS, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_silu_back(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SILU_BACK, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ಭಿನ್ನ * op_params = (ಭಿನ್ನ *)dst->op_params;
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_NORM, std::move(p));
}

ಶೂನ್ಯ ggml_vk_group_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * int_op_params = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)dst->op_params;
    ಧ್ರುವ ಭಿನ್ನ * float_op_params = (ಧ್ರುವ ಭಿನ್ನ *)dst->op_params;

    ಧ್ರುವ uint32_t ಅಂಕೆ_groups = int_op_params[0];
    ಧ್ರುವ ಭಿನ್ನ eps = float_op_params[1];
    ಧ್ರುವ uint32_t group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + ಅಂಕೆ_groups - 1) / ಅಂಕೆ_groups);

    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_GROUP_NORM, { group_size, 0, eps, 0.0f, 0.0f, 0.0f });
}

ಸ್ಥಿರ uint32_t ggml_vk_rms_ಅಂಕೆ_partials(ggml_backend_vk_context * ctx, ಧ್ರುವ ggml_tensor *ಗಂಟು) {
    ಧ್ರುವ uint32_t ne = (uint32_t)ಗಂಟು->ne[0];
    ಧ್ರುವ uint32_t denom = ctx->device->pipeline_add_rms[0][0][0]->wg_denoms[0];
    ಧ್ರುವ uint32_t ಅಂಕೆ_partials = CEIL_DIV(ne, denom);
    ಪ್ರತ್ಯಾಗಮ ಅಂಕೆ_partials;
}

uint32_t ggml_vk_rms_partials_size(ggml_backend_vk_context * ctx, ಧ್ರುವ ggml_tensor *ಗಂಟು) {
    ಧ್ರುವ uint32_t ಅಂಕೆ_partials = ggml_vk_rms_ಅಂಕೆ_partials(ctx, ಗಂಟು);
    ಧ್ರುವ uint32_t ಅಂಕೆ_bytes = ROUNDUP_POW2(ಅಂಕೆ_partials * ಪರಿಮಾಣ(uint32_t), ctx->device->partials_binding_alignment);
    ಪ್ರತ್ಯಾಗಮ ಅಂಕೆ_bytes;
}

ಸ್ಥಿರ vk_op_rope_ತಳ್ಳು_constants ggml_vk_make_rope_constants(ಧ್ರುವ ggml_tensor *dst, ಧ್ರುವ ggml_tensor *src0, ಧ್ರುವ ತಾರ್ಕಿಕ has_ff, ತಾರ್ಕಿಕ backprop, ಧ್ರುವ uint32_t set_rows_stride) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims        = ((ಧ್ರುವ int32_t *) dst->op_params)[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode          = ((ಧ್ರುವ int32_t *) dst->op_params)[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs        = ((ಧ್ರುವ int32_t *) dst->op_params)[15];
    // const int n_ctx         = ((const int32_t *) dst->op_params)[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ctx_orig    = ((ಧ್ರುವ int32_t *) dst->op_params)[4];
    ಧ್ರುವ ಭಿನ್ನ freq_base   = ((ಧ್ರುವ ಭಿನ್ನ *)   dst->op_params)[5];
    ಧ್ರುವ ಭಿನ್ನ freq_scale  = ((ಧ್ರುವ ಭಿನ್ನ *)   dst->op_params)[6];
    ಧ್ರುವ ಭಿನ್ನ ext_factor  = ((ಧ್ರುವ ಭಿನ್ನ *)   dst->op_params)[7];
    ಧ್ರುವ ಭಿನ್ನ attn_factor = ((ಧ್ರುವ ಭಿನ್ನ *)   dst->op_params)[8];
    ಧ್ರುವ ಭಿನ್ನ beta_fast   = ((ಧ್ರುವ ಭಿನ್ನ *)   dst->op_params)[9];
    ಧ್ರುವ ಭಿನ್ನ beta_slow   = ((ಧ್ರುವ ಭಿನ್ನ *)   dst->op_params)[10];
    ಪೂರ್ಣಾಙ್ಕ sections[4] {};
    ಯದಿ (mode & GGML_ROPE_TYPE_MROPE) {
        memcpy(sections, (ಧ್ರುವ int32_t *) dst->op_params + 11, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)*4);
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    ಭಿನ್ನ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    ಧ್ರುವ ಭಿನ್ನ theta_scale = powf(freq_base, -2.0f/n_dims);

    uint32_t nb01 = src0->nb[1] / ggml_type_size(src0->type);
    uint32_t nb02 = src0->nb[2] / ggml_type_size(src0->type);
    uint32_t nb03 = src0->nb[3] / ggml_type_size(src0->type);

    uint32_t nb11 = dst->nb[1] / ggml_type_size(dst->type);
    uint32_t nb12 = dst->nb[2] / ggml_type_size(dst->type);
    uint32_t nb13 = dst->nb[3] / ggml_type_size(dst->type);

    vk_op_rope_ತಳ್ಳು_constants rope {
        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, (uint32_t)n_offs, freq_scale,
        freq_base, ext_factor, attn_factor, {corr_dims[0], corr_dims[1]}, theta_scale, has_ff,
        { sections[0], sections[1], sections[2], sections[3] }, is_imrope, backprop, set_rows_stride,

        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)src0->ne[2],
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        0, 0, // a_offset, d_offset filled in by init_pushconst_tensor_offsets
    };

    ಪ್ರತ್ಯಾಗಮ rope;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_rms_norm_finish(ggml_backend_vk_context * ctx, ಧ್ರುವ ggml_tensor * src0) {
    ಯದಿ (ctx->do_add_rms_partials_offset_calculation) {
        ctx->prealloc_size_add_rms_partials_offset += ggml_vk_rms_partials_size(ctx, src0);
        ctx->do_add_rms_partials = ಅಸತ್ಯ;
        ctx->do_add_rms_partials_offset_calculation = ಅಸತ್ಯ;
    }
}

ಶೂನ್ಯ ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಭಿನ್ನ * op_params) {
    ggml_tensor * rms = cgraph->nodes[ಗಂಟು_idx];
    ಧ್ರುವ ggml_tensor * src0 = rms->src[0];

    ಯದಿ (ctx->fused_rms_norm_mode == RMS_NORM_VIEW_SET_ROWS) {
        GGML_ASSERT(ctx->ಅಂಕೆ_additional_fused_ops == 2);
        ggml_tensor * set_rows = cgraph->nodes[ಗಂಟು_idx + 2];
        ಧ್ರುವ ggml_tensor * indices = set_rows->src[1];
        vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants pc = ggml_vk_rms_norm_ತಳ್ಳು_constants(src0, src0, set_rows, op_params[0], 0);
        init_pushconst_tensor_offsets(ctx, pc, src0, src0, nullptr, nullptr, set_rows);

        vk_pipeline pipeline = set_rows->type == GGML_TYPE_F16 ?
            ctx->device->pipeline_rms_norm_set_rows_f32_f16 : ctx->device->pipeline_rms_norm_set_rows_f32_f32;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_tensor_subbuffer(ctx, src0, ಸತ್ಯ),
                ggml_vk_tensor_subbuffer(ctx, src0, ಸತ್ಯ),
                ggml_vk_tensor_subbuffer(ctx, set_rows, ಸತ್ಯ),
                ggml_vk_tensor_subbuffer(ctx, indices),
            }, pc, {
                (uint32_t)src0->ne[1],
                std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]),
                std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]),
            });
        ggml_vk_rms_norm_finish(ctx, src0);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (ctx->fused_rms_norm_mode == RMS_NORM_MUL_ADD || ctx->fused_rms_norm_mode == RMS_NORM_MUL_ADD_MUL) {
        ggml_tensor * mul = cgraph->nodes[ಗಂಟು_idx + 1];
        ggml_tensor * add = cgraph->nodes[ಗಂಟು_idx + 2];
        ಧ್ರುವ ggml_tensor * weight = mul->src[0] == rms ? mul->src[1] : mul->src[0];
        ಧ್ರುವ ggml_tensor * residual = add->src[0] == mul ? add->src[1] : add->src[0];
        ಧ್ರುವ ತಾರ್ಕಿಕ do_post_multiply = ctx->fused_rms_norm_mode == RMS_NORM_MUL_ADD_MUL;
        GGML_ASSERT(ctx->ಅಂಕೆ_additional_fused_ops == (do_post_multiply ? 3 : 2));
        ggml_tensor * dst = do_post_multiply ? cgraph->nodes[ಗಂಟು_idx + 3] : add;
        ಧ್ರುವ ggml_tensor * post_scale = do_post_multiply ?
            (dst->src[0] == add ? dst->src[1] : dst->src[0]) : src0;

        ಧ್ರುವ uint32_t ಅಂಕೆ_partials = ctx->do_add_rms_partials ? ggml_vk_rms_ಅಂಕೆ_partials(ctx, dst) : 0;
        vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants pc = ggml_vk_rms_norm_ತಳ್ಳು_constants(src0, weight, dst, op_params[0], ಅಂಕೆ_partials);
        init_pushconst_tensor_offsets(ctx, pc, src0, weight, residual, post_scale, dst);

        vk_pipeline pipeline;
        ಯದಿ (ctx->do_add_rms_partials) {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_add_mul_partials_f32 : ctx->device->pipeline_rms_norm_mul_add_partials_f32;
        } ಅನ್ಯಥಾ {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_add_mul_f32 : ctx->device->pipeline_rms_norm_mul_add_f32;
        }
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ಯದಿ (ctx->do_add_rms_partials) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, ಸತ್ಯ),
                    ggml_vk_tensor_subbuffer(ctx, weight, ಸತ್ಯ),
                    ggml_vk_tensor_subbuffer(ctx, dst, ಸತ್ಯ),
                    ggml_vk_subbuffer(ctx, ctx->prealloc_add_rms_partials, ctx->prealloc_size_add_rms_partials_offset),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, { (uint32_t)CEIL_DIV(src0->ne[0], 128), 1, 1 });
        } ಅನ್ಯಥಾ {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, ಸತ್ಯ),
                    ggml_vk_tensor_subbuffer(ctx, weight, ಸತ್ಯ),
                    ggml_vk_tensor_subbuffer(ctx, dst, ಸತ್ಯ),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, {
                    (uint32_t)src0->ne[1],
                    std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]),
                    std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]),
                });
        }
        ggml_vk_rms_norm_finish(ctx, src0);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ggml_tensor * dst;
    ಧ್ರುವ ggml_tensor * src1;

    ಯದಿ (ctx->fused_rms_norm_mode != RMS_NORM_ಎಣಿಕೆ) {
        ggml_tensor * mul = cgraph->nodes[ಗಂಟು_idx + 1];
        dst = mul;
        src1 = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    } ಅನ್ಯಥಾ {
        dst = rms;
        src1 = src0;
    }

    ಧ್ರುವ uint32_t ಅಂಕೆ_partials = ctx->do_add_rms_partials ? ggml_vk_rms_ಅಂಕೆ_partials(ctx, dst) : 0;
    vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants bin = ggml_vk_rms_norm_ತಳ್ಳು_constants(src0, src1, dst, op_params[0], ಅಂಕೆ_partials);

    ಯದಿ (ctx->fused_rms_norm_mode == RMS_NORM_MUL_ROPE ||
        ctx->fused_rms_norm_mode == RMS_NORM_MUL_ROPE_VIEW_SET_ROWS) {
        ಸ್ಥಿರ constexpr uint32_t max_tensors = 7;
        ಧ್ರುವ ggml_tensor *tensors[max_tensors] {};

        ggml_tensor *rms = cgraph->nodes[ಗಂಟು_idx + 0];
        ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx + 1];
        ggml_tensor *rope = cgraph->nodes[ಗಂಟು_idx + 2];

        ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];

        ತಾರ್ಕಿಕ do_set_rows = ctx->fused_rms_norm_mode == RMS_NORM_MUL_ROPE_VIEW_SET_ROWS;
        GGML_ASSERT(ctx->ಅಂಕೆ_additional_fused_ops == (do_set_rows ? 4 : 2));

        tensors[0] = rms->src[0];
        tensors[1] = other_src;
        tensors[2] = mul;
        tensors[3] = rope->src[1]; // pos
        tensors[4] = rope->src[2]; // ff
        tensors[5] = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops]; // dst
        tensors[6] = do_set_rows ? tensors[5]->src[1] : nullptr;
        ಧ್ರುವ uint32_t set_rows_stride = do_set_rows ? tensors[5]->nb[1] / ggml_type_size(tensors[5]->type) : 0;

        vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants pc;
        pc.bin = bin;
        pc.rope = ggml_vk_make_rope_constants(rope, rope->src[0], tensors[4] != nullptr, ಅಸತ್ಯ, set_rows_stride);

        vk_pipeline pipeline = tensors[5]->type == GGML_TYPE_F16 ? ctx->device->pipeline_rms_norm_mul_rope_f32_f16 : ctx->device->pipeline_rms_norm_mul_rope_f32_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

        ggml_backend_vk_buffer_context * buf_ctx[max_tensors];
        vk_buffer buf[max_tensors];
        size_t offset[max_tensors];
        ತಾರ್ಕಿಕ uma[max_tensors];

        ಪ್ರತಿ (uint32_t i = 0; i < max_tensors; ++i) {
            ಯದಿ (!tensors[i]) {
                // If any remaining descriptors are unused, just point them at src[0]
                buf[i] = buf[0];
                offset[i] = 0;
                ಅನುವರ್ತ;
            }
            buf_ctx[i] = (ggml_backend_vk_buffer_context *)tensors[i]->buffer->context;
            buf[i] = nullptr;
            offset[i] = 0;
            uma[i] = ಅಸತ್ಯ;

            ಯದಿ (ctx->device->uma) {
                ggml_vk_host_get(ctx->device, tensors[i]->data, buf[i], offset[i]);
                uma[i] = buf[i] != nullptr;
            }
            ಯದಿ (!uma[i]) {
                buf[i] = buf_ctx[i]->dev_buffer;
                offset[i] = vk_tensor_offset(tensors[i]) + tensors[i]->view_offs;
            }
            GGML_ASSERT(buf[i] != nullptr);
        }

        // a_offset is unused (the fused path reads from shared memory), but the rope/set_rows dst can be misaligned.
        // Round the binding offset down to the storage buffer alignment; the in-element shift goes in pc.rope.d_offset.
        pc.rope.d_offset = get_misalign_bytes(ctx, tensors[5]) / ggml_type_size(tensors[5]->type);
        offset[5] &= ~(size_t(ctx->device->properties.limits.minStorageBufferOffsetAlignment) - 1);

        std::array<uint32_t, 3> elements;
        elements = { (uint32_t)rms->src[0]->ne[1], (uint32_t)rms->src[0]->ne[2], (uint32_t)rms->src[0]->ne[3] };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[2]);

        static_assert(max_tensors == 7);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_subbuffer(ctx, buf[0], offset[0]),
                ggml_vk_subbuffer(ctx, buf[1], offset[1]),
                ggml_vk_subbuffer(ctx, buf[2], offset[2]),
                ggml_vk_subbuffer(ctx, buf[3], offset[3]),
                ggml_vk_subbuffer(ctx, buf[4], offset[4]),
                ggml_vk_subbuffer(ctx, buf[5], offset[5]),
                ggml_vk_subbuffer(ctx, buf[6], offset[6]),
            }, pc, elements);
    } ಅನ್ಯಥಾ {
        GGML_ASSERT(ctx->fused_rms_norm_mode == RMS_NORM_MUL || ctx->fused_rms_norm_mode == RMS_NORM_ಎಣಿಕೆ);
        ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_RMS_NORM, std::move(bin));
    }

    ggml_vk_rms_norm_finish(ctx, src0);
}

ಶೂನ್ಯ ggml_vk_rms_norm_back(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಭಿನ್ನ * op_params = (ಭಿನ್ನ *)dst->op_params;
    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_RMS_NORM_BACK, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f, 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ಧ್ರುವ ಭಿನ್ನ * op_params = (ಧ್ರುವ ಭಿನ್ನ *)dst->op_params;
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = op_params[0];
    ggml_vk_op_f32<vk_op_unary_ತಳ್ಳು_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_L2_NORM, std::move(p));
}

ಶೂನ್ಯ ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_UNARY, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ಭಿನ್ನ * op_params = (ಭಿನ್ನ *)dst->op_params;
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = op_params[1];
    p.param2 = op_params[2];
    p.param3 = op_params[3];
    p.param4 = op_params[4];
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_UNARY, std::move(p));
}

ಶೂನ್ಯ ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ ಭಿನ್ನ * op_params_f = (ಧ್ರುವ ಭಿನ್ನ *)dst->op_params;

    ಧ್ರುವ ತಾರ್ಕಿಕ swapped = (ತಾರ್ಕಿಕ)dst->op_params[1];
    ಧ್ರುವ ತಾರ್ಕಿಕ split = src1 != nullptr;
    ಧ್ರುವ ಭಿನ್ನ alpha = op_params_f[2];
    ಧ್ರುವ ಭಿನ್ನ limit = op_params_f[3];

    ಯದಿ (!split) {
        GGML_ASSERT(src0->ne[0] / 2 == dst->ne[0]);
    } ಅನ್ಯಥಾ {
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] == dst->ne[0]);
        GGML_ASSERT(src0->type == src1->type);
    }

    ಧ್ರುವ uint32_t mode = split ? 2 : (swapped ? 1 : 0);
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = split ? ggml_type_size(src1->type) : src0_type_size;
    ಧ್ರುವ uint32_t dst_type_size  = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_glu_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_GLU,
        {
            (uint32_t)ggml_nelements(dst),
            (uint32_t)src0->ne[0],
            (uint32_t)dst->ne[0],
            mode,
            alpha,
            limit,
            (uint32_t)(src0->nb[0] / src0_type_size),
            (uint32_t)(src0->nb[1] / src0_type_size),
            (uint32_t)(src0->nb[2] / src0_type_size),
            (uint32_t)(src0->nb[3] / src0_type_size),
            (uint32_t)((split ? src1->nb[0] : src0->nb[0]) / src1_type_size),
            (uint32_t)((split ? src1->nb[1] : src0->nb[1]) / src1_type_size),
            (uint32_t)((split ? src1->nb[2] : src0->nb[2]) / src1_type_size),
            (uint32_t)((split ? src1->nb[3] : src0->nb[3]) / src1_type_size),
            (uint32_t)(dst->nb[0] / dst_type_size),
            (uint32_t)(dst->nb[1] / dst_type_size),
            (uint32_t)(dst->nb[2] / dst_type_size),
            (uint32_t)(dst->nb[3] / dst_type_size),
            (uint32_t)dst->ne[1],
            (uint32_t)dst->ne[2],
            0,
            0, 0, 0, 0, 0, 0,
        });
}

ಶೂನ್ಯ ggml_vk_diag_mask_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_mask_ತಳ್ಳು_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_DIAG_MASK_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] });
}

ಶೂನ್ಯ ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ಧ್ರುವ ggml_tensor * src2, ggml_tensor * dst) {
    ಭಿನ್ನ * op_params = (ಭಿನ್ನ *)dst->op_params;

    ಭಿನ್ನ scale = op_params[0];
    ಭಿನ್ನ max_bias = op_params[1];

    ಧ್ರುವ uint32_t ncols =   (uint32_t)src0->ne[0];
    ಧ್ರುವ uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    ಧ್ರುವ uint32_t nrows_y = (uint32_t)src0->ne[1];

    ಧ್ರುವ uint32_t ne12 = src1 ? (uint32_t)(src1->ne[2]) : 0u;
    ಧ್ರುವ uint32_t ne13 = src1 ? (uint32_t)(src1->ne[3]) : 0u;
    ಧ್ರುವ uint32_t nb11 = src1 ? (uint32_t)(src1->nb[1] / src1->nb[0]) : 0u;
    ಧ್ರುವ uint32_t nb12 = src1 ? (uint32_t)(src1->nb[2] / src1->nb[0]) : 0u;
    ಧ್ರುವ uint32_t nb13 = src1 ? (uint32_t)(src1->nb[3] / src1->nb[0]) : 0u;

    ಧ್ರುವ uint32_t n_head_kv   = src0->ne[2];
    ಧ್ರುವ uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((ಭಿನ್ನ) n_head_kv));

    ಧ್ರುವ ಭಿನ್ನ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ಧ್ರುವ ಭಿನ್ನ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    vk_op_soft_max_ತಳ್ಳು_constants pc {
        ncols,
        src1 != nullptr ? nrows_y : (uint32_t)0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
        ne12, ne13,
        nb11, nb12, nb13,
        scale, max_bias,
        m0, m1,
        n_head_log2,
        nrows_x,
        src2 != nullptr
    };

    ಯದಿ (ncols <= 16384) {
        ggml_vk_op_f32<vk_op_soft_max_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_SOFT_MAX, std::move(pc));
    } ಅನ್ಯಥಾ {

        vk_subbuffer buf_a = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer buf_b = src1 ? ggml_vk_tensor_subbuffer(ctx, src1) : buf_a;
        vk_subbuffer buf_c = src2 ? ggml_vk_tensor_subbuffer(ctx, src2) : buf_a;
        vk_subbuffer buf_d = ggml_vk_tensor_subbuffer(ctx, dst);

        uint32_t elems_per_wg = 128 * 4;
        uint32_t ಅಂಕೆ_wgs = CEIL_DIV(ncols, elems_per_wg);
        size_t tmp_size = ಅಂಕೆ_wgs * nrows_x * ಪರಿಮಾಣ(ಭಿನ್ನ);

        ಯದಿ (ctx->prealloc_size_x < tmp_size) {
            ctx->prealloc_size_x = tmp_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ (ctx->prealloc_size_y < tmp_size) {
            ctx->prealloc_size_y = tmp_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ (ctx->prealloc_x_need_sync || ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_subbuffer buf_x = { ctx->prealloc_x, 0, tmp_size };
        vk_subbuffer buf_y = { ctx->prealloc_y, 0, tmp_size };

        // the partial results overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_last_pipeline_used = nullptr;
        ctx->prealloc_y_last_tensor_used = nullptr;

        std::array<uint32_t, 3> elements = { ಅಂಕೆ_wgs, nrows_x, 1 };

        vk_pipeline pipeline1 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large1_f32_f16 : ctx->device->pipeline_soft_max_large1_f32;
        vk_pipeline pipeline2 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large2_f32_f16 : ctx->device->pipeline_soft_max_large2_f32;
        vk_pipeline pipeline3 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large3_f32_f16 : ctx->device->pipeline_soft_max_large3_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline3, 1);

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline3, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);

        ctx->prealloc_x_need_sync = ಸತ್ಯ;
        ctx->prealloc_y_need_sync = ಸತ್ಯ;
    }
}

ಶೂನ್ಯ ggml_vk_soft_max_back(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಭಿನ್ನ * op_params = (ಭಿನ್ನ *)dst->op_params;
    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SOFT_MAX_BACK, { (uint32_t)src0->ne[0], (uint32_t)ggml_nrows(src0), op_params[0], op_params[1], 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    topk_moe_mode mode = ctx->fused_topk_moe_mode;
    ಧ್ರುವ ತಾರ್ಕಿಕ has_bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS || mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
    ggml_tensor * logits = cgraph->nodes[ಗಂಟು_idx + 0]->src[0];
    ggml_tensor * bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[ಗಂಟು_idx + 2]->src[1] :
                         mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[ಗಂಟು_idx + 3]->src[1] :
                                                                   logits;
    ggml_tensor * weights = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops];
    ggml_tensor * ids = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[ಗಂಟು_idx + 4] :
                        mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[ಗಂಟು_idx + 5] :
                        mode == TOPK_MOE_LATE_SOFTMAX             ? cgraph->nodes[ಗಂಟು_idx + 1] :
                                                                   cgraph->nodes[ಗಂಟು_idx + 3];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_experts = logits->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_rows    = logits->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, nullptr, nullptr, nullptr, cgraph->nodes[ಗಂಟು_idx], GGML_OP_SOFT_MAX);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer bias_buf = ggml_vk_tensor_subbuffer(ctx, bias);
    vk_subbuffer weights_buf = ggml_vk_tensor_subbuffer(ctx, weights);
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, ids);

    vk_op_topk_moe_ತಳ್ಳು_constants pc {};
    pc.n_rows = n_rows;
    pc.n_experts_ತಳ್ಳು = n_experts;
    pc.n_expert_used = n_expert_used;
    pc.clamp_min = -std::numeric_limits<ಭಿನ್ನ>::infinity();
    pc.clamp_max = std::numeric_limits<ಭಿನ್ನ>::infinity();
    ಯದಿ (mode == TOPK_MOE_EARLY_SOFTMAX_NORM) {
        ggml_tensor * clamp = cgraph->nodes[ಗಂಟು_idx + 7];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    ಯದಿ (mode == TOPK_MOE_SIGMOID_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[ಗಂಟು_idx + 8];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    ಯದಿ (mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[ಗಂಟು_idx + 9];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }

#define GATING_FUNC_SOFTMAX 0
#define GATING_FUNC_SIGMOID 1
#define GATING_FUNC_SOFTMAX_WEIGHT 2
#define GATING_FUNC_SQRT_SOFTPLUS 3

    pc.gating_func = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? GATING_FUNC_SIGMOID :
                     mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? GATING_FUNC_SQRT_SOFTPLUS :
                     mode == TOPK_MOE_LATE_SOFTMAX             ? GATING_FUNC_SOFTMAX_WEIGHT :
                                                                 GATING_FUNC_SOFTMAX;
    pc.has_bias = has_bias;
    pc.with_norm = mode == TOPK_MOE_EARLY_SOFTMAX_NORM || has_bias;
    ಯದಿ (ctx->fused_topk_moe_scale) {
        GGML_ASSERT(weights->op == GGML_OP_SCALE);
        pc.output_scale = ggml_get_op_params_f32(weights, 0);
        pc.output_bias = ggml_get_op_params_f32(weights, 1);
    } ಅನ್ಯಥಾ {
        pc.output_scale = 1.0f;
        pc.output_bias = 0.0f;
    }

    GGML_ASSERT(n_expert_used <= n_experts);

    ಧ್ರುವ uint32_t rows_per_block = 4;
    std::array<uint32_t, 3> elements = { CEIL_DIV(n_rows, rows_per_block), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {logits_buf, bias_buf, weights_buf, ids_buf}, pc, elements);
}

ಶೂನ್ಯ ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ತಾರ್ಕಿಕ backprop) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];
    ಧ್ರುವ ggml_tensor * src2 = dst->src[2];
    ಧ್ರುವ ggml_tensor * src3 = nullptr;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims        = ((int32_t *) dst->op_params)[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    ಧ್ರುವ ಭಿನ್ನ freq_base   = ((ಭಿನ್ನ *)   dst->op_params)[5];
    ಧ್ರುವ ಭಿನ್ನ beta_fast   = ((ಭಿನ್ನ *)   dst->op_params)[9];
    ಧ್ರುವ ಭಿನ್ನ beta_slow   = ((ಭಿನ್ನ *)   dst->op_params)[10];
    ಪೂರ್ಣಾಙ್ಕ sections[4] {};
    ಯದಿ (mode & GGML_ROPE_TYPE_MROPE) {
        memcpy(sections, (int32_t *) dst->op_params + 11, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ)*4);
    }

    ಭಿನ್ನ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    uint32_t set_rows_stride = 0;
    // Fused rope + view + set_rows passes the set_rows destination stride in set_rows_stride
    // and overrides the dst and sets src3=row_indices
    ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
        set_rows_stride = cgraph->nodes[ಗಂಟು_idx + 2]->nb[1] / ggml_type_size(cgraph->nodes[ಗಂಟು_idx + 2]->type);
        src3 = cgraph->nodes[ಗಂಟು_idx + 2]->src[1];
        dst = cgraph->nodes[ಗಂಟು_idx + 2];
    }

    ggml_vk_op_f32<vk_op_rope_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, src2, src3, dst, GGML_OP_ROPE,
        ggml_vk_make_rope_constants(cgraph->nodes[ಗಂಟು_idx], src0, src2 != nullptr, backprop, set_rows_stride));
}

ಶೂನ್ಯ ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t * op_params = (ಧ್ರುವ uint32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);

    uint32_t ncols_pad_log2 = (uint32_t)ceilf(log2f(ಭಿನ್ನ(ncols)));
    uint32_t ncolsp2 = 1 << ncols_pad_log2;

    vk_op_argsort_ತಳ್ಳು_constants pc { ncols, ncolsp2, ncols_pad_log2, nrows, op_params[0], 0, 0, 0, 0, };

    // Pick the largest workgroup size <= ncolsp2
    uint32_t pipeline_idx = std::min(ncols_pad_log2, ಅಂಕೆ_argsort_pipelines - 1);

    uint32_t max_wg_log2 = std::min(ctx->device->max_workgroup_size_log2, ಅಂಕೆ_argsort_pipelines - 1);

    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    ತಾರ್ಕಿಕ use_small = ncols_pad_log2 <= max_wg_log2 &&
                     ctx->device->pipeline_argsort_f32[pipeline_idx] != nullptr;

    vk_pipeline pipeline = use_small ? ctx->device->pipeline_argsort_f32[pipeline_idx]
                                     : ctx->device->pipeline_argsort_large_f32[pipeline_idx];

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    ಯದಿ (!use_small) {
        ಧ್ರುವ size_t x_sz = size_t{ncolsp2} * nrows * 2 * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ);

        ಯದಿ (ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಯದಿ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        subbuf1 = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    }

    std::array<uint32_t, 3> elements;

    elements[0] = ncolsp2;
    elements[1] = std::min((uint32_t)ggml_nrows(src0), ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
    elements[2] = 1;

    // First dispatch initializes tmp_idx and does the first N passes where
    // there is only communication between threads in the same workgroup.
    {
        vk_op_argsort_ತಳ್ಳು_constants pc2 = pc;
        pc2.outer_start = 0;
        pc2.outer_end = std::min(ncols_pad_log2, max_wg_log2);
        pc2.inner_start = 0;
        pc2.inner_end = 100;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
    }
    ಯದಿ (!use_small) {
        ggml_vk_sync_buffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        ಪ್ರತಿ (uint32_t outer = max_wg_log2; outer < ncols_pad_log2; ++outer) {
            ಪ್ರತಿ (uint32_t inner = 0; inner < outer + 1; ++inner) {
                vk_op_argsort_ತಳ್ಳು_constants pc2 = pc;
                pc2.outer_start = outer;
                pc2.outer_end = outer + 1;
                pc2.inner_start = inner;
                pc2.inner_end = inner + 1;
                // When the inner idx is large enough, there's only communication
                // within a workgroup. So the remaining inner iterations can all
                // run in the same dispatch.
                ಯದಿ (outer - inner < pipeline_idx) {
                    pc2.inner_end = 100;
                    inner = outer;
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx];
                } ಅನ್ಯಥಾ {
                    // Smaller workgroup empirically seems to perform better
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx - 2];
                }
                ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
                ggml_vk_sync_buffers(ctx, subctx);
            }
        }
        ctx->prealloc_x_need_sync = ಸತ್ಯ;
    }
}

ಶೂನ್ಯ ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);
    uint32_t k = dst->ne[0];

    // tournament path is faster where it fits; use radix-select only past its k limit
    ಧ್ರುವ uint32_t k_min_pipeline = std::max((uint32_t) log2f(ಭಿನ್ನ(k)) + 1, ctx->device->subgroup_size_log2);
    ಯದಿ (k_min_pipeline >= ಅಂಕೆ_topk_pipelines || ctx->device->pipeline_topk_f32[k_min_pipeline] == nullptr) {
        vk_pipeline pipeline = ctx->device->pipeline_topk_radix_f32;
        GGML_ASSERT(pipeline != nullptr);

        ಯದಿ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_op_topk_radix_ತಳ್ಳು_constants pc { ncols, k, nrows, 0, 0, 0 };
        std::array<uint32_t, 3> elements {
            pipeline->wg_denoms[0],
            std::min(nrows, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]),
            1,
        };
        // the non-QSA path only uses bindings 0/1; bind valid buffers for the unused QSA slots
        vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }, pc, elements);
        ಪ್ರತ್ಯಾಗಮ;
    }

    vk_op_topk_ತಳ್ಳು_constants pc { ncols, ncols, ncols, k, nrows, 0, 0 };

    ಯದಿ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    std::array<uint32_t, 3> elements;
    elements[1] = std::min(nrows, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]);
    elements[2] = 1;

    uint32_t ಅಂಕೆ_elements = ncols;

    // Each iteration reduces a workgroup's worth of elements down to the K
    // largest elements. Repeat until we have the top K elements.
    // Need to do at least one iteration to write out the results.
    ತಾರ್ಕಿಕ done_one_iter = ಅಸತ್ಯ;
    uint32_t dbl_buf_index = 0;
    size_t dbl_buf_size;
    ಯಾವತ್ (ಅಂಕೆ_elements > k || !done_one_iter) {

        // Prefer going as small as num_topk_pipelines - 3 for perf reasons.
        // But if K is larger, then we need a larger workgroup
        uint32_t max_pipeline = ಅಂಕೆ_topk_pipelines - 1;
        uint32_t preferred_pipeline = std::max(ಅಂಕೆ_topk_pipelines - 3, (uint32_t)log2f(ಭಿನ್ನ(k)) + 2);
        max_pipeline = std::min(preferred_pipeline, max_pipeline);
        uint32_t min_pipeline = (uint32_t)log2f(ಭಿನ್ನ(k)) + 1;
        // require full subgroup
        min_pipeline = std::max(min_pipeline, ctx->device->subgroup_size_log2);

        uint32_t pipeline_idx = (uint32_t)ceilf(log2f(ಭಿನ್ನ(ಅಂಕೆ_elements)));
        pipeline_idx = std::min(pipeline_idx, max_pipeline);
        pipeline_idx = std::max(pipeline_idx, min_pipeline);

        ಯದಿ (ಅಂಕೆ_elements > (1u << pipeline_idx)) {
            // If we could finish on this loop iteration (i.e. a single workgroup)
            // then do so. It's better than the overhead of another pass.
            ಪ್ರತಿ (uint32_t i = pipeline_idx; i < ಅಂಕೆ_topk_pipelines; ++i) {
                ಯದಿ (ಅಂಕೆ_elements <= (1u << i)) {
                    pipeline_idx = i;
                    ಭಙ್ಗ;
                }
            }
        }

        vk_pipeline pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        // If the device doesn't support a pipeline this large, use smaller
        ಯಾವತ್ (!pipeline) {
            pipeline_idx--;
            GGML_ASSERT(pipeline_idx >= min_pipeline);
            pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        }

        vk_op_topk_ತಳ್ಳು_constants pc2 = pc;
        pc2.ncols_ಒಳಹರಿವು = ಅಂಕೆ_elements;

        // Number of elements remaining after this pass
        uint32_t ಅಂಕೆ_dst_elements = (ಅಂಕೆ_elements / pipeline->wg_denoms[0]) * k + std::min(k, ಅಂಕೆ_elements % pipeline->wg_denoms[0]);

        pc2.ncols_output = ಅಂಕೆ_dst_elements;

        ಯದಿ (!done_one_iter) {
            // Reserve space for ivec2 per element, double buffered
            // K per workgroup per row
            dbl_buf_size = ಅಂಕೆ_dst_elements * nrows * 2 * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ);
            dbl_buf_size = ROUNDUP_POW2(dbl_buf_size, ctx->device->properties.limits.minStorageBufferOffsetAlignment);
            ಧ್ರುವ size_t x_sz = dbl_buf_size * 2;

            ಯದಿ (ctx->prealloc_size_x < x_sz) {
                ctx->prealloc_size_x = x_sz;
                ggml_vk_preallocate_buffers(ctx, subctx);
            }
        }

        vk_subbuffer src_buf;
        vk_subbuffer dst_buf;

        ಯದಿ (ಅಂಕೆ_elements == ncols) {
            pc2.first_pass = 1;
            src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        } ಅನ್ಯಥಾ {
            src_buf = { ctx->prealloc_x, dbl_buf_index * dbl_buf_size, dbl_buf_size };
        }
        ಯದಿ (ಅಂಕೆ_dst_elements == k) {
            pc2.last_pass = 1;
            dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
        } ಅನ್ಯಥಾ {
            dst_buf = { ctx->prealloc_x, (dbl_buf_index ^ 1) * dbl_buf_size, dbl_buf_size };
        }

        elements[0] = ಅಂಕೆ_elements;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc2, elements);
        ಅಂಕೆ_elements = ಅಂಕೆ_dst_elements;
        dbl_buf_index ^= 1;
        ಯದಿ (ಅಂಕೆ_elements > k) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        done_one_iter = ಸತ್ಯ;
    }
    ctx->prealloc_x_need_sync = ಸತ್ಯ;
}

ಶೂನ್ಯ ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ggml_tensor * get_rows = cgraph->nodes[ಗಂಟು_idx + 0];
    ಧ್ರುವ ggml_tensor * add      = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops - 1];
    ggml_tensor *       ತುದಿ_k    = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops];

    ಧ್ರುವ ggml_tensor * scores   = get_rows->src[0]; // [n_tps, n_blocks, n_stream]
    ಧ್ರುವ ggml_tensor * cell_blk = get_rows->src[1]; // [n_kv, n_stream]

    // raw f16 mask: follow the reshape/cpy chain back to the materialized input
    ಧ್ರುವ ggml_tensor * mask = add->src[1];
    ಯಾವತ್ (mask->op == GGML_OP_RESHAPE || mask->op == GGML_OP_CPY) {
        mask = mask->src[0];
    }

    ಧ್ರುವ uint32_t n_tps    = scores->ne[0];
    ಧ್ರುವ uint32_t n_blocks = scores->ne[1];
    ಧ್ರುವ uint32_t n_stream = scores->ne[2];
    ಧ್ರುವ uint32_t n_kv     = cell_blk->ne[0];
    ಧ್ರುವ uint32_t width    = ತುದಿ_k->ne[0];
    ಧ್ರುವ uint32_t nrows    = n_tps * n_stream;

    vk_pipeline pipeline = ctx->device->pipeline_topk_radix_qsa;
    GGML_ASSERT(pipeline != nullptr);

    // scratch holds the gathered+masked input, materialized once and reused across passes
    ಧ್ರುವ size_t scratch_size = size_t{ n_kv } * nrows * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಯದಿ (ctx->prealloc_size_x < scratch_size) {
        ctx->prealloc_size_x = scratch_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    ಯದಿ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_op_topk_radix_ತಳ್ಳು_constants pc { n_kv, width, nrows, n_tps, n_blocks, n_stream };
    std::array<uint32_t, 3> elements {
        pipeline->wg_denoms[0],
        std::min(nrows, ctx->device->properties.limits.max_Compute_Work_Group_ಎಣಿಕೆ[1]),
        1,
    };
    vk_subbuffer scratch_buf { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, scores), ggml_vk_tensor_subbuffer(ctx, ತುದಿ_k),
          ggml_vk_tensor_subbuffer(ctx, cell_blk), ggml_vk_tensor_subbuffer(ctx, mask),
          scratch_buf }, pc, elements);
    ctx->prealloc_x_need_sync = ಸತ್ಯ;
}

ಶೂನ್ಯ ggml_vk_sum(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_sum_rows_ತಳ್ಳು_constants p = vk_op_sum_rows_ತಳ್ಳು_constants_init(src0, dst, ggml_nelements(src0));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SUM, p);
}

ಶೂನ್ಯ ggml_vk_sum_rows(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_sum_rows_ತಳ್ಳು_constants p = vk_op_sum_rows_ತಳ್ಳು_constants_init(src0, dst, src0->ne[0]);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SUM_ROWS, p);
}

ಶೂನ್ಯ ggml_vk_mean(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_sum_rows_ತಳ್ಳು_constants p = vk_op_sum_rows_ತಳ್ಳು_constants_init(src0, dst, src0->ne[0]);
    p.weight = 1.0f / (ಭಿನ್ನ)src0->ne[0];
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_MEAN, p);
}

ಶೂನ್ಯ ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_sum_rows_ತಳ್ಳು_constants pc = vk_op_sum_rows_ತಳ್ಳು_constants_init(src0, dst, src0->ne[0]);
    // Use the single pass shader when the rows are small or there are enough rows to fill the GPU.
    // For fewer, larger rows, use the multipass shader to spread each row across SMs.
    ಯದಿ (dst->ne[0] <= 4096 || ggml_nrows(dst) >= ctx->device->shader_core_ಎಣಿಕೆ) {
        ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_CUMSUM, pc);
        ಪ್ರತ್ಯಾಗಮ;
    }

    // First pass computes partial sums within a block, and stores the last partial
    // to the temp buffer. Second pass sums the block partials from the temp buffer
    // and adds that to the result of the first pass.
    vk_pipeline pipeline1 = ctx->device->pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline2 = ctx->device->pipeline_cumsum_multipass2_f32;
    GGML_ASSERT(pipeline1 != nullptr && pipeline2 != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);

    std::array<uint32_t, 3> elements;

    elements[0] = dst->ne[0];
    elements[1] = (uint32_t)ggml_nrows(dst);
    elements[2] = 1;

    size_t temp_size = ಪರಿಮಾಣ(ಭಿನ್ನ) * elements[0] * ggml_nrows(dst);

    ಯದಿ (ctx->prealloc_size_split_k < temp_size) {
        ctx->prealloc_size_split_k = temp_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer temp_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_split_k, 0);

    ಯದಿ (ctx->prealloc_split_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, {src_buf, dst_buf, temp_buf}, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, {src_buf, dst_buf, temp_buf}, pc, elements);

    ctx->prealloc_split_k_need_sync = ಸತ್ಯ;
}

ಸ್ಥಿರ std::array<uint32_t, 3> ggml_vk_nrows_elements(uint32_t nr) {
    ಯದಿ (nr > 262144) {
        ಪ್ರತ್ಯಾಗಮ { 512, 512, CEIL_DIV(nr, 262144) };
    }
    ಯದಿ (nr > 512) {
        ಪ್ರತ್ಯಾಗಮ { 512, CEIL_DIV(nr, 512), 1 };
    }
    ಪ್ರತ್ಯಾಗಮ { nr, 1, 1 };
}

ಶೂನ್ಯ ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    ಧ್ರುವ uint32_t nclasses = (uint32_t)src0->ne[0];
    ಧ್ರುವ uint32_t nrows    = (uint32_t)ggml_nrows(src0);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, nullptr, dst, GGML_OP_CROSS_ENTROPY_LOSS);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_sum_rows_f32, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src1_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst, ಸತ್ಯ);

    ಧ್ರುವ vk_op_ತಳ್ಳು_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };

    ಧ್ರುವ size_t tmp_size = (size_t)nrows * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಯದಿ (ctx->prealloc_size_x < tmp_size) {
        ctx->prealloc_size_x = tmp_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    ಯದಿ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_subbuffer tmp_buf = { ctx->prealloc_x, 0, tmp_size };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, tmp_buf }, pc, ggml_vk_nrows_elements(nrows));
    ggml_vk_sync_buffers(ctx, subctx);

    vk_op_sum_rows_ತಳ್ಳು_constants sp = {};
    sp.n_cols = nrows;
    sp.ne01 = 1;
    sp.ne02 = 1;
    sp.weight = 1.0f;
    init_pushconst_fastdiv(sp);
    sp.misalign_offsets = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_sum_rows_f32, { tmp_buf, dst_buf }, sp, { 1, 1, 1 });
    ctx->prealloc_x_need_sync = ಸತ್ಯ;
}

ಶೂನ್ಯ ggml_vk_cross_entropy_loss_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * grad   = dst->src[0];
    ಧ್ರುವ ggml_tensor * logits = dst->src[1];
    ಧ್ರುವ ggml_tensor * labels = dst->src[2];

    GGML_ASSERT(grad->type   == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(labels->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(labels));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(logits, labels));
    GGML_ASSERT(ggml_are_same_shape(logits, dst));

    ಧ್ರುವ uint32_t nclasses = (uint32_t)logits->ne[0];
    ಧ್ರುವ uint32_t nrows    = (uint32_t)ggml_nrows(logits);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, grad, logits, labels, dst, GGML_OP_CROSS_ENTROPY_LOSS_BACK);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer grad_buf   = ggml_vk_tensor_subbuffer(ctx, grad);
    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer labels_buf = ggml_vk_tensor_subbuffer(ctx, labels);
    vk_subbuffer dst_buf    = ggml_vk_tensor_subbuffer(ctx, dst);

    ಧ್ರುವ vk_op_ತಳ್ಳು_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { grad_buf, logits_buf, labels_buf, dst_buf }, pc, ggml_vk_nrows_elements(nrows));
}

ಶೂನ್ಯ ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_ARGMAX, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f, 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_ಎಣಿಕೆ_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_ಎಣಿಕೆ_EQUAL, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t src0_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t src1_type_size = ggml_type_size(src1->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SOLVE_TRI, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    ಧ್ರುವ int32_t s0 = dst->op_params[0];
    ಧ್ರುವ int32_t s1 = dst->op_params[1];
    ಧ್ರುವ int32_t p0 = dst->op_params[2];
    ಧ್ರುವ int32_t p1 = dst->op_params[3];
    ಧ್ರುವ int32_t d0 = dst->op_params[4];
    ಧ್ರುವ int32_t d1 = dst->op_params[5];

    ಧ್ರುವ ತಾರ್ಕಿಕ is_2D = dst->op_params[6] == 1;

    ಧ್ರುವ uint32_t IC = src1->ne[is_2D ? 2 : 1];
    ಧ್ರುವ uint32_t IH = is_2D ? src1->ne[1] : 1;
    ಧ್ರುವ uint32_t IW =         src1->ne[0];

    ಧ್ರುವ uint32_t KH = is_2D ? src0->ne[1] : 1;
    ಧ್ರುವ uint32_t KW =         src0->ne[0];

    ಧ್ರುವ uint32_t OH = is_2D ? dst->ne[2] : 1;
    ಧ್ರುವ uint32_t OW =         dst->ne[1];

    ಧ್ರುವ uint32_t offset_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    ಧ್ರುವ uint32_t batch_offset = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    ಧ್ರುವ uint32_t batch = src1->ne[is_2D ? 3 : 2];

    ಧ್ರುವ ggml_backend_vk_buffer_context * d_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ಧ್ರುವ vk_buffer d_buf = d_buf_ctx->dev_buffer;

    ಧ್ರುವ vk::DeviceAddress dst_addr = d_buf->bda_addr + vk_tensor_offset(dst) + dst->view_offs;

    ggml_vk_op_f32<vk_op_im2col_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_IM2COL, {
        dst_addr,
        batch_offset, offset_delta,
        IC, IW, IH, OW, OH, KW, KH,
        OH * batch,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1, batch * IC
    });
}

ಶೂನ್ಯ ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t *)(dst->op_params))[0];
    ಧ್ರುವ int32_t s1 = ((ಧ್ರುವ int32_t *)(dst->op_params))[1];
    ಧ್ರುವ int32_t s2 = ((ಧ್ರುವ int32_t *)(dst->op_params))[2];
    ಧ್ರುವ int32_t p0 = ((ಧ್ರುವ int32_t *)(dst->op_params))[3];
    ಧ್ರುವ int32_t p1 = ((ಧ್ರುವ int32_t *)(dst->op_params))[4];
    ಧ್ರುವ int32_t p2 = ((ಧ್ರುವ int32_t *)(dst->op_params))[5];
    ಧ್ರುವ int32_t d0 = ((ಧ್ರುವ int32_t *)(dst->op_params))[6];
    ಧ್ರುವ int32_t d1 = ((ಧ್ರುವ int32_t *)(dst->op_params))[7];
    ಧ್ರುವ int32_t d2 = ((ಧ್ರುವ int32_t *)(dst->op_params))[8];
    ಧ್ರುವ int32_t IC = ((ಧ್ರುವ int32_t *)(dst->op_params))[9];

    ಧ್ರುವ int64_t N  = ne13 / IC;
    ಧ್ರುವ int64_t ID = ne12;
    ಧ್ರುವ int64_t IH = ne11;
    ಧ್ರುವ int64_t IW = ne10;

    ಧ್ರುವ int64_t KD = ne02;
    ಧ್ರುವ int64_t KH = ne01;
    ಧ್ರುವ int64_t KW = ne00;

    ಧ್ರುವ int64_t OD = ne3 / N;
    ಧ್ರುವ int64_t OH = ne2;
    ಧ್ರುವ int64_t OW = ne1;

    ಧ್ರುವ ggml_backend_vk_buffer_context * d_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ಧ್ರುವ vk_buffer d_buf = d_buf_ctx->dev_buffer;

    ಧ್ರುವ vk::DeviceAddress dst_addr = d_buf->bda_addr + vk_tensor_offset(dst) + dst->view_offs;

    vk_op_im2col_3d_ತಳ್ಳು_constants pc {};

    pc.dst_addr = dst_addr;
    pc.nb10 = nb10 / ggml_type_size(src1->type);
    pc.nb11 = nb11 / ggml_type_size(src1->type);
    pc.nb12 = nb12 / ggml_type_size(src1->type);
    pc.nb13 = nb13 / ggml_type_size(src1->type);
    pc.s0 = s0;
    pc.s1 = s1;
    pc.s2 = s2;
    pc.p0 = p0;
    pc.p1 = p1;
    pc.p2 = p2;
    pc.d0 = d0;
    pc.d1 = d1;
    pc.d2 = d2;
    pc.IW = IW;
    pc.IH = IH;
    pc.ID = ID;
    pc.IC = IC;
    pc.KW = KW;
    pc.OH = OH;
    pc.KD_KH_KW = KD*KH*KW;
    pc.KH_KW = KH*KW;
    pc.IC_KD_KH_KW = IC*KD*KH*KW;
    pc.N_OD_OH = N*OD*OH;
    pc.OD_OH = OD*OH;
    pc.OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    pc.OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    pc.OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;

    ggml_vk_op_f32<vk_op_im2col_3d_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_IM2COL_3D, std::move(pc));
}

ಶೂನ್ಯ ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ಧ್ರುವ uint32_t dim = dst->op_params[0];
    ಧ್ರುವ uint32_t max_period = dst->op_params[1];
    ಧ್ರುವ uint32_t nb1 = dst->nb[1] / ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_ತಳ್ಳು_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    });
}

ಶೂನ್ಯ ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    // src0: (K, Cout, Cin, 1) -- kernel
    // src1: (L, Cin, 1, 1) -- input
    // dst: (*, Cout, 1, 1)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    GGML_ASSERT(nb00 == ಪರಿಮಾಣ(ಭಿನ್ನ));
    GGML_ASSERT(nb10 == ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಧ್ರುವ int32_t s0 = dst->op_params[0];

    vk_op_conv_transpose_1d_ತಳ್ಳು_constants p{};
    p.Cout = static_cast<uint32_t>(ne01);
    p.Cin = static_cast<uint32_t>(ne02);
    p.K = static_cast<uint32_t>(ne00);
    p.L = static_cast<uint32_t>(ne10);
    p.KL = static_cast<uint32_t>(ne0);
    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.s0 = static_cast<uint32_t>(s0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONV_TRANSPOSE_1D, std::move(p));
}

ಶೂನ್ಯ ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    // src0: [K_OC, T_in] columns from matmul
    // dst:  [T_out, OC]

    ಧ್ರುವ int32_t stride = dst->op_params[0];
    ಧ್ರುವ int32_t oc     = dst->op_params[1];
    ಧ್ರುವ int32_t p0     = dst->op_params[2];

    ಧ್ರುವ uint32_t K_OC  = static_cast<uint32_t>(src0->ne[0]);
    ಧ್ರುವ uint32_t T_in  = static_cast<uint32_t>(src0->ne[1]);
    ಧ್ರುವ uint32_t T_out = static_cast<uint32_t>(dst->ne[0]);
    ಧ್ರುವ uint32_t OC    = static_cast<uint32_t>(oc);
    ಧ್ರುವ uint32_t K     = K_OC / OC;

    vk_op_col2im_1d_ತಳ್ಳು_constants p{};
    p.T_out  = T_out;
    p.OC     = OC;
    p.K_OC   = K_OC;
    p.T_in   = T_in;
    p.K      = K;
    p.stride = stride;
    p.p0     = p0;

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_COL2IM_1D, std::move(p));
}

ಶೂನ್ಯ ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ggml_tensor * mul0 = cgraph->nodes[ಗಂಟು_idx + 0];
    ಧ್ರುವ ggml_tensor * sqr  = cgraph->nodes[ಗಂಟು_idx + 2];
    ಧ್ರುವ ggml_tensor * mul1 = cgraph->nodes[ಗಂಟು_idx + 3];
    ggml_tensor *       add  = cgraph->nodes[ಗಂಟು_idx + 4];

    // x carries the full activation shape, a is the broadcast operand
    ಧ್ರುವ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ಧ್ರುವ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ಧ್ರುವ ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    vk_pipeline pipeline = nullptr;
    ವಿಕಲ್ಪ (x->type) {
        ಪಕ್ಷ GGML_TYPE_F32:  pipeline = ctx->device->pipeline_snake_f32;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_F16:  pipeline = ctx->device->pipeline_snake_f16;  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_BF16: pipeline = ctx->device->pipeline_snake_bf16; ಭಙ್ಗ;
        ಸ್ವಭಾವ:             GGML_ABORT("unsupported type");
    }
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf     = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer a_buf     = ggml_vk_tensor_subbuffer(ctx, a);
    vk_subbuffer inv_b_buf = ggml_vk_tensor_subbuffer(ctx, inv_b);
    vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, add);

    vk_op_snake_ತಳ್ಳು_constants pc{};
    pc.ne0 = static_cast<uint32_t>(x->ne[0]);
    pc.ne1 = static_cast<uint32_t>(x->ne[1]);

    std::array<uint32_t, 3> elements = { pc.ne0, pc.ne1, 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, a_buf, inv_b_buf, dst_buf }, pc, elements);
}

ಶೂನ್ಯ ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ಧ್ರುವ int32_t k0 = dst->op_params[1];
    ಧ್ರುವ int32_t s0 = dst->op_params[2];
    ಧ್ರುವ int32_t p0 = dst->op_params[3];

    ಧ್ರುವ uint32_t IL = src0->ne[0];

    ಧ್ರುವ uint32_t N = dst->ne[3] * dst->ne[2];

    ಧ್ರುವ uint32_t OC = dst->ne[1];
    ಧ್ರುವ uint32_t OL = dst->ne[0];

    ಧ್ರುವ uint32_t parallel_elements = N * OC * OL;

    ggml_vk_op_f32<vk_op_pool1d_ತಳ್ಳು_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_POOL_1D, {
        IL, OL, OC,
        parallel_elements,
        op,
        k0, s0, p0,
    });
}

ಶೂನ್ಯ ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ಧ್ರುವ int32_t k1 = dst->op_params[1];
    ಧ್ರುವ int32_t k0 = dst->op_params[2];
    ಧ್ರುವ int32_t s1 = dst->op_params[3];
    ಧ್ರುವ int32_t s0 = dst->op_params[4];
    ಧ್ರುವ int32_t p1 = dst->op_params[5];
    ಧ್ರುವ int32_t p0 = dst->op_params[6];

    ಧ್ರುವ uint32_t IH = src0->ne[1];
    ಧ್ರುವ uint32_t IW = src0->ne[0];

    ಧ್ರುವ uint32_t N = dst->ne[3];

    ಧ್ರುವ uint32_t OC = dst->ne[2];
    ಧ್ರುವ uint32_t OH = dst->ne[1];
    ಧ್ರುವ uint32_t OW = dst->ne[0];

    ಧ್ರುವ uint32_t parallel_elements = N * OC * OH * OW;

    ggml_vk_op_f32<vk_op_pool2d_ತಳ್ಳು_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_POOL_2D, {
        IW, IH, OW, OH, OC,
        parallel_elements,
        op,
        k0, k1, s0, s1, p0, p1,
    });
}

ಶೂನ್ಯ ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ಧ್ರುವ ggml_tensor * src0,
                            ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS
    GGML_ASSERT(nb00 == ಪರಿಮಾಣ(ಭಿನ್ನ) || nb00 == ಪರಿಮಾಣ(ggml_fp16_t));
    GGML_ASSERT(nb10 == ಪರಿಮಾಣ(ಭಿನ್ನ));
    GGML_ASSERT(nb0 == ಪರಿಮಾಣ(ಭಿನ್ನ));

    ತಾರ್ಕಿಕ transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;

    vk_op_conv2d_ತಳ್ಳು_constants p{};
    p.Cout = static_cast<uint32_t>(!transpose ? ne03 : ne02);
    p.Cin  = static_cast<uint32_t>(!transpose ? ne02 : ne03);
    p.N    = static_cast<uint32_t>(ne13);
    GGML_ASSERT(p.Cout == ne2);
    GGML_ASSERT(p.Cin == ne12);

    p.W  = static_cast<uint32_t>(ne10);
    p.H  = static_cast<uint32_t>(ne11);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, dst->op, std::move(p));
}

ಶೂನ್ಯ ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ಧ್ರುವ ggml_tensor * src0,
                            ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS
    GGML_ASSERT(nb00 == ಪರಿಮಾಣ(ಭಿನ್ನ) || nb00 == ಪರಿಮಾಣ(ggml_fp16_t));
    GGML_ASSERT(nb10 == ಪರಿಮಾಣ(ಭಿನ್ನ));
    GGML_ASSERT(nb0 == ಪರಿಮಾಣ(ಭಿನ್ನ));

    vk_op_conv3d_ತಳ್ಳು_constants p{};
    p.IC = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 9));
    p.N  = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 10));
    p.OC = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 11));
    GGML_ASSERT(src0->ne[3] == (int64_t)p.IC * p.OC);
    GGML_ASSERT(src1->ne[3] == (int64_t)p.IC * p.N);
    GGML_ASSERT(dst->ne[3] == (int64_t)p.OC * p.N);

    p.IW = static_cast<uint32_t>(ne10);
    p.IH = static_cast<uint32_t>(ne11);
    p.ID = static_cast<uint32_t>(ne12);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);
    p.OD = static_cast<uint32_t>(ne2);

    // the shader clamps src addresses to p.IC * p.N * p.IW * p.IH * p.ID - 1 in uint32, so the
    // total input element count must fit in a uint32.
    GGML_ASSERT((uint64_t)p.IC * p.N * p.IW * p.IH * p.ID <= 0xFFFFFFFFull);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONV_3D, std::move(p));
}

ಶೂನ್ಯ ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1, ggml_tensor * dst) {
    vk_op_conv2d_dw_ತಳ್ಳು_constants p{};
    p.ne = ggml_nelements(dst);
    p.channels = dst->ne[2];
    p.batches = dst->ne[3];
    p.dst_w = dst->ne[0];
    p.dst_h = dst->ne[1];
    p.src_w = src1->ne[0];
    p.src_h = src1->ne[1];
    p.knl_w = src0->ne[0];
    p.knl_h = src0->ne[1];
    p.stride_x = dst->op_params[0];
    p.stride_y = dst->op_params[1];
    p.pad_x = dst->op_params[2];
    p.pad_y = dst->op_params[3];
    p.dilation_x = dst->op_params[4];
    p.dilation_y = dst->op_params[5];

    GGML_ASSERT(src0->ne[3] == p.channels);
    GGML_ASSERT(src1->ne[3] == p.batches);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONV_2D_DW, std::move(p));
}

ಶೂನ್ಯ ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ಧ್ರುವ ggml_tensor * src0, ggml_tensor * dst) {
    ಧ್ರುವ ಭಿನ್ನ * op_params = (ಧ್ರುವ ಭಿನ್ನ *)dst->op_params;
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_LEAKY_RELU, std::move(p));
}

ಶೂನ್ಯ ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {
#ಯದಿ defined(GGML_VULKAN_RUN_TESTS)
    ಧ್ರುವ std::vector<size_t> vals {
        512, 512, 128,
        128, 512, 512,
        4096, 512, 4096,
        11008, 512, 4096,
        4096, 512, 11008,
        32000, 512, 4096,
        8, 8, 8,
        100, 46, 576,
        623, 111, 128,
        100, 46, 558,
        512, 1, 256,
        128, 110, 622,
        511, 511, 127,
        511, 511, 7,
        511, 511, 17,
        49, 49, 128,
        128, 49, 49,
        4096, 49, 4096,
    };
    ಧ್ರುವ size_t ಅಂಕೆ_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 0, GGML_TYPE_Q4_0, ಸತ್ಯ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 1, GGML_TYPE_Q4_0, ಸತ್ಯ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 2, GGML_TYPE_Q4_0, ಸತ್ಯ);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 0, GGML_TYPE_Q8_0, ಸತ್ಯ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 1, GGML_TYPE_Q8_0, ಸತ್ಯ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 2, GGML_TYPE_Q8_0, ಸತ್ಯ);

    abort();

    ಪ್ರತಿ (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, ಭಿನ್ನ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, ಭಿನ್ನ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, ಭಿನ್ನ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, ಭಿನ್ನ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, ಭಿನ್ನ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, ಭಿನ್ನ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, ಭಿನ್ನ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, ಭಿನ್ನ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, ಭಿನ್ನ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 2);
        std::cerr << '\n' << std::endl;

        ಯದಿ (vals[i + 2] % 32 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n' << std::endl;
        }

        ಯದಿ (vals[i + 2] % 256 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n' << std::endl;
        }
    }

    GGML_ABORT("fatal error");
#endif

    ಯದಿ (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = ಸತ್ಯ;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
    }

    ಯದಿ (ctx->prealloc_x == nullptr || (ctx->prealloc_size_x > 0 && ctx->prealloc_x->size < ctx->prealloc_size_x)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_size_x << ")");
        // Resize buffer
        ಯದಿ (ctx->prealloc_x != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_x);
    }
    ಯದಿ (ctx->prealloc_y == nullptr || (ctx->prealloc_size_y > 0 && ctx->prealloc_y->size < ctx->prealloc_size_y)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_size_y << ")");
        // Resize buffer
        ಯದಿ (ctx->prealloc_y != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_y);
        ctx->prealloc_y_last_pipeline_used = nullptr;
        ctx->prealloc_y_last_tensor_used = nullptr;
        ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;
    }
    ಯದಿ (ctx->prealloc_split_k == nullptr || (ctx->prealloc_size_split_k > 0 && ctx->prealloc_split_k->size < ctx->prealloc_size_split_k)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_size_split_k << ")");
        // Resize buffer
        ಯದಿ (ctx->prealloc_split_k != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_split_k);
        }
        ctx->prealloc_split_k = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_split_k);
    }
    ಯದಿ (ctx->prealloc_add_rms_partials == nullptr || (ctx->prealloc_size_add_rms_partials > 0 && ctx->prealloc_add_rms_partials->size < ctx->prealloc_size_add_rms_partials)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_add_rms_partials << ")");
        // Resize buffer
        ಯದಿ (ctx->prealloc_add_rms_partials != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_add_rms_partials);
        }
        ctx->prealloc_add_rms_partials = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_add_rms_partials);
    }
}

ತಾರ್ಕಿಕ ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ggml_tensor *ಗಂಟು_begin, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx_begin, ತಾರ್ಕಿಕ last_ಗಂಟು, ತಾರ್ಕಿಕ almost_ready, ತಾರ್ಕಿಕ submit){
    ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];
    ಯದಿ (ggml_is_empty(ಗಂಟು) || ggml_op_is_empty(ಗಂಟು->op) || !ಗಂಟು->buffer) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << ಗಂಟು << ", " << ggml_op_name(ಗಂಟು->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = ಗಂಟು->src[0];
    ggml_tensor * src1 = ಗಂಟು->src[1];
    ggml_tensor * src2 = ಗಂಟು->src[2];
    ggml_tensor * src3 = ಗಂಟು->src[3];

    ಯದಿ (ಗಂಟು->op == GGML_OP_ADD) {
        ಪೂರ್ಣಾಙ್ಕ next_ಗಂಟು_idx = ಗಂಟು_idx + 1 + ctx->ಅಂಕೆ_additional_fused_ops;
        ಯದಿ (next_ಗಂಟು_idx < cgraph->n_nodes &&
            cgraph->nodes[next_ಗಂಟು_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[next_ಗಂಟು_idx]->src[0] == cgraph->nodes[next_ಗಂಟು_idx - 1] &&
            ggml_nrows(cgraph->nodes[next_ಗಂಟು_idx]) == 1 &&
            ctx->device->add_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_size(ctx, cgraph->nodes[ಗಂಟು_idx]);
            ctx->do_add_rms_partials_offset_calculation = ಸತ್ಯ;
            ಯದಿ (ctx->prealloc_size_add_rms_partials_offset + size <= ctx->prealloc_size_add_rms_partials) {
                ctx->do_add_rms_partials = ಸತ್ಯ;
            }
        }
    }

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    {
        // This logic detects dependencies between modes in the graph and calls ggml_vk_sync_buffers
        // to synchronize them. This handles most "normal" synchronization when computing the graph, and when
        // there is no auxiliary memory use, it shouldn't be necessary to call ggml_vk_sync_buffers
        // outside of this logic. When a node uses one of the prealloc buffers for something like
        // dequantization or split_k, additional synchronization is needed between those passes.
        ತಾರ್ಕಿಕ need_sync = ಅಸತ್ಯ;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        auto ಧ್ರುವ &overlaps_unsynced = [&](ಧ್ರುವ ggml_tensor *ಗಂಟು, ಧ್ರುವ std::vector<ಧ್ರುವ ggml_tensor *> &unsynced_nodes) -> ತಾರ್ಕಿಕ {
            ಯದಿ (unsynced_nodes.size() == 0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            auto n_base = vk_tensor_offset(ಗಂಟು) + ಗಂಟು->view_offs;
            auto n_size = ggml_nbytes(ಗಂಟು);
            ggml_backend_vk_buffer_context * a_buf_ctx = (ggml_backend_vk_buffer_context *)ಗಂಟು->buffer->context;
            vk_buffer a_buf = a_buf_ctx->dev_buffer;
            ಪ್ರತಿ (auto &other : unsynced_nodes) {
                ggml_backend_vk_buffer_context * o_buf_ctx = (ggml_backend_vk_buffer_context *)other->buffer->context;
                vk_buffer o_buf = o_buf_ctx->dev_buffer;
                ಯದಿ (a_buf == o_buf) {
                    auto o_base = vk_tensor_offset(other) + other->view_offs;
                    auto o_size = ggml_nbytes(other);

                    ಯದಿ ((o_base <= n_base && n_base < o_base + o_size) ||
                        (n_base <= o_base && o_base < n_base + n_size)) {
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    }
                }
            }
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        ಪ್ರತಿ (int32_t i = 0; i < ctx->ಅಂಕೆ_additional_fused_ops + 1 && !need_sync; ++i) {
            ಧ್ರುವ ggml_tensor *cur_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            ಯದಿ (ctx->fused_ops_write_mask & (1 << i)) {
                ಯದಿ (overlaps_unsynced(cur_ಗಂಟು, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_ಗಂಟು, ctx->unsynced_nodes_written)) {
                    need_sync = ಸತ್ಯ;
                    ಭಙ್ಗ;
                }
            }
            ಪ್ರತಿ (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                ಯದಿ (!cur_ಗಂಟು->src[j]) {
                    ಅನುವರ್ತ;
                }
                ಯದಿ (overlaps_unsynced(cur_ಗಂಟು->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = ಸತ್ಯ;
                    ಭಙ್ಗ;
                }
            }
        }

        ಯದಿ (need_sync) {
            ಯದಿ (vk_enable_sync_logger) {
                std::cerr <<  "sync" << std::endl;
            }
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            ಯದಿ (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_ಗಂಟು_idx[ctx->query_idx] = ಗಂಟು_idx;
                compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        ಪ್ರತಿ (int32_t i = 0; i < ctx->ಅಂಕೆ_additional_fused_ops + 1; ++i) {
            ಧ್ರುವ ggml_tensor *cur_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            ಯದಿ (ctx->fused_ops_write_mask & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_ಗಂಟು);
            }
            ಪ್ರತಿ (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                ಯದಿ (!cur_ಗಂಟು->src[j]) {
                    ಅನುವರ್ತ;
                }
                ctx->unsynced_nodes_read.push_back(cur_ಗಂಟು->src[j]);
            }
        }
    }
    ಯದಿ (vk_enable_sync_logger) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ctx->ಅಂಕೆ_additional_fused_ops + 1; ++i) {
            auto *n = cgraph->nodes[ಗಂಟು_idx + i];
            std::cerr << ಗಂಟು_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            ಯದಿ (n->op == GGML_OP_GLU) {
                std::cerr << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            ಯದಿ (n->op == GGML_OP_ROPE) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode = ((ಧ್ರುವ int32_t *) n->op_params)[2];
                std::cerr << " rope mode: " << mode;
            }
            std::cerr << std::endl;
        }
    }

    // closed explicitly below, and by the destructor on the early returns
    ggml_vk_debug_label dbg(compute_ctx, cgraph, ಗಂಟು_idx, ctx->ಅಂಕೆ_additional_fused_ops);

    ವಿಕಲ್ಪ (ಗಂಟು->op) {
    ಪಕ್ಷ GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_REPEAT_BACK:
        ggml_vk_repeat_back(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ACC:
    ಪಕ್ಷ GGML_OP_SET:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_GET_ROWS:
        ಯದಿ (ctx->fused_topk_qsa) {
            ggml_vk_topk_qsa(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
        } ಅನ್ಯಥಾ {
            ggml_vk_get_rows(ctx, compute_ctx, src0, src1, ಗಂಟು);
        }

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_GET_ROWS_BACK:
        ggml_vk_get_rows_back(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ADD:
        ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops) {
            ggml_vk_multi_add(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
        } ಅನ್ಯಥಾ {
            ggml_vk_add(ctx, compute_ctx, src0, src1, ಗಂಟು);
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_OUT_PROD:
        ggml_vk_out_prod(ctx, compute_ctx, src0, src1, ಗಂಟು);
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_MUL:
        ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops) {
            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
        } ಅನ್ಯಥಾ {
            ggml_vk_mul(ctx, compute_ctx, src0, src1, ಗಂಟು);
        }

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ADD_ID:
        ggml_vk_add_id(ctx, compute_ctx, src0, src1, src2, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SCALE:
        ಯದಿ (ctx->fused_hc_post_gate) {
            ggml_tensor * hc_post = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops];
            ggml_vk_dsv4_hc_post(ctx, compute_ctx, hc_post->src[0], hc_post->src[1], hc_post->src[2], hc_post->src[3], hc_post, ಗಂಟು);
        } ಅನ್ಯಥಾ {
            ggml_vk_scale(ctx, compute_ctx, src0, ಗಂಟು);
        }

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_PAD_REFLECT_1D:
        ggml_vk_pad_reflect_1d(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CPY:
    ಪಕ್ಷ GGML_OP_CONT:
    ಪಕ್ಷ GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SET_ROWS:
        ggml_vk_set_rows(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SILU_BACK:
        ggml_vk_silu_back(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_GROUP_NORM:
        ggml_vk_group_norm(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, ಗಂಟು_idx, (ಭಿನ್ನ *)ಗಂಟು->op_params);
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_RMS_NORM_BACK:
        ggml_vk_rms_norm_back(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_UNARY:
        ಯದಿ (ctx->fused_topk_moe_mode != TOPK_MOE_ಎಣಿಕೆ) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
            ಭಙ್ಗ;
        }
        ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops) {
            ggml_vk_unary_mul(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
            ಭಙ್ಗ;
        }

        ವಿಕಲ್ಪ (ggml_get_unary_op(ಗಂಟು)) {
        ಪಕ್ಷ GGML_UNARY_OP_ELU:
        ಪಕ್ಷ GGML_UNARY_OP_EXP:
        ಪಕ್ಷ GGML_UNARY_OP_EXPM1:
        ಪಕ್ಷ GGML_UNARY_OP_SILU:
        ಪಕ್ಷ GGML_UNARY_OP_GELU:
        ಪಕ್ಷ GGML_UNARY_OP_GELU_ERF:
        ಪಕ್ಷ GGML_UNARY_OP_GELU_QUICK:
        ಪಕ್ಷ GGML_UNARY_OP_RELU:
        ಪಕ್ಷ GGML_UNARY_OP_NEG:
        ಪಕ್ಷ GGML_UNARY_OP_TANH:
        ಪಕ್ಷ GGML_UNARY_OP_SIGMOID:
        ಪಕ್ಷ GGML_UNARY_OP_HARDSIGMOID:
        ಪಕ್ಷ GGML_UNARY_OP_HARDSWISH:
        ಪಕ್ಷ GGML_UNARY_OP_ABS:
        ಪಕ್ಷ GGML_UNARY_OP_SOFTPLUS:
        ಪಕ್ಷ GGML_UNARY_OP_STEP:
        ಪಕ್ಷ GGML_UNARY_OP_ROUND:
        ಪಕ್ಷ GGML_UNARY_OP_CEIL:
        ಪಕ್ಷ GGML_UNARY_OP_FLOOR:
        ಪಕ್ಷ GGML_UNARY_OP_TRUNC:
        ಪಕ್ಷ GGML_UNARY_OP_SGN:
            ggml_vk_unary(ctx, compute_ctx, src0, ಗಂಟು);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, ಗಂಟು);
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_GLU:
        ವಿಕಲ್ಪ (ggml_get_glu_op(ಗಂಟು)) {
        ಪಕ್ಷ GGML_GLU_OP_GEGLU:
        ಪಕ್ಷ GGML_GLU_OP_REGLU:
        ಪಕ್ಷ GGML_GLU_OP_SWIGLU:
        ಪಕ್ಷ GGML_GLU_OP_SWIGLU_OAI:
        ಪಕ್ಷ GGML_GLU_OP_GEGLU_ERF:
        ಪಕ್ಷ GGML_GLU_OP_GEGLU_QUICK:
        ಪಕ್ಷ GGML_GLU_OP_SWIGLU_CLAMP:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, ಗಂಟು);
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_DIAG_MASK_INF:
        ggml_vk_diag_mask_inf(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SOFT_MAX:
        ಯದಿ (ctx->fused_topk_moe_mode != TOPK_MOE_ಎಣಿಕೆ) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
        } ಅನ್ಯಥಾ {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, ಗಂಟು);
        }

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SOFT_MAX_BACK:
        ggml_vk_soft_max_back(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, ಗಂಟು_idx, ಅಸತ್ಯ);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ROPE_BACK:
        ggml_vk_rope(ctx, compute_ctx, cgraph, ಗಂಟು_idx, ಸತ್ಯ);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ARGSORT:
        ಯದಿ (ctx->fused_topk_moe_mode != TOPK_MOE_ಎಣಿಕೆ) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
        } ಅನ್ಯಥಾ {
            ggml_vk_argsort(ctx, compute_ctx, src0, ಗಂಟು);
        }

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ತುದಿ_K:
        ggml_vk_topk(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SUM:
        ggml_vk_sum(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SUM_ROWS:
        ggml_vk_sum_rows(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_DSV4_HC_COMB:
        ggml_vk_dsv4_hc_comb(ctx, compute_ctx, src0, src1, src2, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_DSV4_HC_PRE:
        ggml_vk_dsv4_hc_pre(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_DSV4_HC_POST:
        ggml_vk_dsv4_hc_post(ctx, compute_ctx, src0, src1, src2, src3, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_MEAN:
        ggml_vk_mean(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CROSS_ENTROPY_LOSS:
        ggml_vk_cross_entropy_loss(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        ggml_vk_cross_entropy_loss_back(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_ಎಣಿಕೆ_EQUAL:
        ggml_vk_ಎಣಿಕೆ_equal(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_IM2COL:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_IM2COL_3D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_COL2IM_1D:
        ggml_vk_col2im_1d(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_POOL_1D:
        ggml_vk_pool_1d(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CONV_2D:
    ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CONV_3D:
        ggml_vk_conv_3d(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, ಗಂಟು);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, ಗಂಟು_idx);

        ಭಙ್ಗ;
    ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, ಗಂಟು_idx);

        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, ಗಂಟು->src[4], ಗಂಟು);

        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_GATED_LINEAR_ATTN:
        ggml_vk_gated_linear_attn(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_LIGHTNING_INDEXER:
        ggml_vk_lightning_indexer(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_SSM_SCAN:
        ggml_vk_ssm_scan(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, cgraph, ಗಂಟು_idx);

        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_OPT_STEP_ADAMW:
        ggml_vk_opt_step_adamw(ctx, compute_ctx, ಗಂಟು);

        ಭಙ್ಗ;

    ಪಕ್ಷ GGML_OP_OPT_STEP_SGD:
        ggml_vk_opt_step_sgd(ctx, compute_ctx, src0, src1, src2, ಗಂಟು);

        ಭಙ್ಗ;
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // the submit path below can end the command buffer, so close the region first
    dbg.close();

    ctx->tensor_ctxs[ಗಂಟು_idx] = compute_ctx;

#ಯದಿ defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    last_ಗಂಟು = ಸತ್ಯ;
#endif

    ಯದಿ (submit || last_ಗಂಟು) {
        ggml_vk_ctx_end(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        ಯದಿ (last_ಗಂಟು) {
            compute_ctx->exit_tensor_idx = ಗಂಟು_idx_begin;
        }
        ಅನ್ಯಥಾ {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, ಗಂಟು_begin, ಗಂಟು_idx_begin, almost_ready);
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಶೂನ್ಯ ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, ಪೂರ್ಣಾಙ್ಕ tensor_idx, ತಾರ್ಕಿಕ almost_ready) {
    GGML_UNUSED(cgraph);
    GGML_UNUSED(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].lock();

    // Only run if ctx hasn't been submitted yet
    ಯದಿ (!subctx->seqs.empty()) {
#ifdef GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
#endif

        // Do staging buffer copies
        ಪ್ರತಿ (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        ಪ್ರತಿ (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ಯದಿ (ctx->device->serialize_submissions) {
            ggml_vk_submit(subctx, ctx->fence);
        } ಅನ್ಯಥಾ ಯದಿ (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = ಸತ್ಯ;
        } ಅನ್ಯಥಾ {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = ಸತ್ಯ;

#ifdef GGML_VULKAN_CHECK_RESULTS
        ggml_vk_synchronize(ctx);
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#endif
    }

    ಯದಿ (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        ಪ್ರತಿ (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        subctx->in_memcpys.clear();
        subctx->out_memcpys.clear();
        subctx->memsets.clear();
    }
}

ಶೂನ್ಯ ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_last_pipeline_used = {};
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = ಅಸತ್ಯ;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    ಯದಿ (ctx->device->async_use_transfer_queue) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

    ಪ್ರತಿ (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    ಪ್ರತಿ (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    ಪ್ರತಿ (auto& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
}

ಶೂನ್ಯ ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);

    ggml_vk_destroy_buffer(ctx->prealloc_x);
    ggml_vk_destroy_buffer(ctx->prealloc_y);
    ggml_vk_destroy_buffer(ctx->prealloc_split_k);
    ggml_vk_destroy_buffer(ctx->prealloc_add_rms_partials);
    ggml_vk_destroy_buffer(ctx->sync_staging);

    ctx->prealloc_y_last_pipeline_used = nullptr;
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;

    ctx->prealloc_size_x = 0;
    ctx->prealloc_size_y = 0;
    ctx->prealloc_size_split_k = 0;

    ಪ್ರತಿ (auto& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    ಪ್ರತಿ (auto& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();
    ctx->descriptor_set_bindings.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    ಯದಿ (ctx->device->async_use_transfer_queue) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
    ಯದಿ (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(ಸತ್ಯ);
    }
}

ಪೂರ್ಣಾಙ್ಕ ggml_vk_get_device_ಎಣಿಕೆ() {
    ggml_vk_instance_init();

    ಪ್ರತ್ಯಾಗಮ vk_instance.device_indices.size();
}

ಶೂನ್ಯ ggml_vk_get_device_description(ಪೂರ್ಣಾಙ್ಕ device, ವರ್ಣ * description, size_t description_size) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_size, "%s", props.deviceName.data());
}

ತಾರ್ಕಿಕ ggml_backend_buffer_is_vk(ggml_backend_buffer_t buffer) {
    ಪ್ರತ್ಯಾಗಮ buffer->buft->iface.get_name == ggml_backend_vk_buffer_type_name;
}

ಶೂನ್ಯ ggml_backend_vk_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    ggml_vk_destroy_buffer(ctx->dev_buffer);
    delete ctx;
}

ಶೂನ್ಯ * ggml_backend_vk_buffer_get_base(ggml_backend_buffer_t buffer) {
    ಪ್ರತ್ಯಾಗಮ vk_ptr_base;

    UNUSED(buffer);
}

ಗಣನಾ ggml_status ggml_backend_vk_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << buffer << " (" << buffer->context << "), " << tensor << ")");
    ಯದಿ (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
    }
    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಶೂನ್ಯ ggml_backend_vk_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << buffer << ", " << tensor << ", " << value << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_buffer_memset(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, val32, size);
}

ಶೂನ್ಯ ggml_backend_vk_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ggml_vk_buffer_write(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

ಶೂನ್ಯ ggml_backend_vk_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor_2d(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_data << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ggml_vk_buffer_write_2d(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, stride_data, stride_tensor, size, n_copies);
}

ಶೂನ್ಯ ggml_backend_vk_buffer_get_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_read(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

ಶೂನ್ಯ ggml_backend_vk_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor_2d(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_data << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_read_2d(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, stride_tensor, stride_data, size, n_copies);
}

ತಾರ್ಕಿಕ ggml_backend_vk_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    ಯದಿ (ggml_nbytes(src) == 0) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;
        ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;

        vk_buffer src_buf = src_buf_ctx->dev_buffer;
        vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

        ggml_vk_buffer_copy(dst_buf, vk_tensor_offset(dst) + dst->view_offs, src_buf, vk_tensor_offset(src) + src->view_offs, ggml_nbytes(src));

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    UNUSED(buffer);
}

ಶೂನ್ಯ ggml_backend_vk_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    ggml_vk_buffer_memset(ctx->dev_buffer, 0, value, buffer->size);
}

ಧ್ರುವ ವರ್ಣ * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    ಪ್ರತ್ಯಾಗಮ ctx->name.c_str();
}

ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;

    vk_buffer dev_buffer = nullptr;
    ಯತ್ನ {
        dev_buffer = ggml_vk_create_buffer_device(ctx->device, size);
    } ಗೃಹಾಣ (ಧ್ರುವ vk::SystemError& e) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ggml_backend_vk_buffer_context * bufctx = ನೂತನ ggml_backend_vk_buffer_context(ctx->device, std::move(dev_buffer), ctx->name);

    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buft, ggml_backend_vk_buffer_interface, bufctx, size);
}

size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    ಪ್ರತ್ಯಾಗಮ ctx->device->properties.limits.minStorageBufferOffsetAlignment;
}

size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    ಪ್ರತ್ಯಾಗಮ ctx->device->suballocation_block_size;
}

size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_nbytes(tensor);

    UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_ಅಂಕೆ) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_ಅಂಕೆ << ")");

    vk_device dev = ggml_vk_get_device(dev_ಅಂಕೆ);

    ಪ್ರತ್ಯಾಗಮ &dev->buffer_type;
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_vk_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ GGML_VK_NAME "_Host";

    UNUSED(buft);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(buffer->context);
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_vk_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    ಶೂನ್ಯ * ptr = ggml_vk_host_malloc(size);
    ಯದಿ (ptr == nullptr) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory\n");
        // fallback to cpu buffer
        ಪ್ರತ್ಯಾಗಮ ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_vk_host_buffer_free_buffer;

    ಪ್ರತ್ಯಾಗಮ buffer;

    UNUSED(buft);
}

ಸ್ಥಿರ size_t ggml_backend_vk_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ vk_instance.devices[0]->properties.limits.minMemoryMapAlignment;

    UNUSED(buft);
}

ಸ್ಥಿರ size_t ggml_backend_vk_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ vk_instance.devices[0]->suballocation_block_size;

    UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type() {
    ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_buffer_type ggml_backend_vk_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_vk_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_vk_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ nullptr,
            /* .get_alignment       = */ ggml_backend_vk_host_buffer_type_get_alignment,
            /* .get_max_size        = */ ggml_backend_vk_host_buffer_type_get_max_size,
            /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .get_alloc_size_n    = */ ರಿಕ್ತ,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ nullptr,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_vk_buffer_type_host;
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ಪ್ರತ್ಯಾಗಮ ctx->name.c_str();
}

ಶೂನ್ಯ ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    delete ctx;
    delete backend;
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_vk_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ಪ್ರತ್ಯಾಗಮ &ctx->device->buffer_type;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context cpy_ctx;

    ಯದಿ (ctx->device->async_use_transfer_queue) {
        cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
    } ಅನ್ಯಥಾ {
        cpy_ctx = ggml_vk_get_compute_ctx(ctx);
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    auto dst_offset = vk_tensor_offset(tensor) + tensor->view_offs + offset;

    ತಾರ್ಕಿಕ ret = ggml_vk_buffer_write_2d_async(cpy_ctx, buf, dst_offset, data, stride_data, stride_tensor, size, n_copies);

    ಯದಿ (!ret) {
        ಧ್ರುವ size_t staging_size = size * n_copies;
        ggml_vk_ensure_sync_staging_buffer(ctx, staging_size);
        ggml_vk_sync_buffers(nullptr, cpy_ctx);

        std::vector<vk::BufferCopy> slices(1);
        ಯದಿ (size == stride_tensor) {
            slices[0].srcOffset = 0;
            slices[0].dstOffset = dst_offset;
            slices[0].size = staging_size;
        } ಅನ್ಯಥಾ {
            slices.resize(n_copies);
            ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
                slices[i].srcOffset = i * size;
                slices[i].dstOffset = dst_offset + i * stride_tensor;
                slices[i].size = size;
            }
        }

        cpy_ctx->s->buffer->buf.copyBuffer(ctx->sync_staging->buffer, buf->buffer, slices);

        ಯದಿ (size == stride_data) {
            deferred_memcpy(ctx->sync_staging->ptr, data, staging_size, &cpy_ctx->in_memcpys);
        } ಅನ್ಯಥಾ {
            ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ctx->sync_staging->ptr + i * size, (ಧ್ರುವ uint8_t *)data + i * stride_data, size, &cpy_ctx->in_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_set_tensor_2d_async(backend, tensor, data, offset, size, 1, size, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_get_tensor_2d_async(ggml_backend_t backend, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    ಯದಿ (size == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    vk_buffer buf = buf_ctx->dev_buffer;

    auto src_offset = vk_tensor_offset(tensor) + tensor->view_offs + offset;
    ತಾರ್ಕಿಕ ret = ggml_vk_buffer_read_2d_async(compute_ctx, buf, src_offset, data, stride_tensor, stride_data, size, n_copies);

    ಯದಿ (!ret) {
        ಧ್ರುವ size_t staging_size = size * n_copies;
        ggml_vk_ensure_sync_staging_buffer(ctx, staging_size);
        ggml_vk_sync_buffers(nullptr, compute_ctx);

        std::vector<vk::BufferCopy> slices(1);
        ಯದಿ (size == stride_tensor) {
            slices[0].srcOffset = src_offset;
            slices[0].dstOffset = 0;
            slices[0].size = staging_size;
        } ಅನ್ಯಥಾ {
            slices.resize(n_copies);
            ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
                slices[i].srcOffset = src_offset + i * stride_tensor;
                slices[i].dstOffset = i * size;
                slices[i].size = size;
            }
        }

        compute_ctx->s->buffer->buf.copyBuffer(buf->buffer, ctx->sync_staging->buffer, slices);

        ಯದಿ (size == stride_data) {
            deferred_memcpy(data, ctx->sync_staging->ptr, staging_size, &compute_ctx->out_memcpys);
        } ಅನ್ಯಥಾ {
            ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)data + i * stride_data, (ಧ್ರುವ uint8_t *)ctx->sync_staging->ptr + i * size, size, &compute_ctx->out_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_get_tensor_async(ggml_backend_t backend, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_get_tensor_2d_async(backend, tensor, data, offset, size, 1, size, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async(" << src << " -> " << dst << ", size=" << ggml_nbytes(src) << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend_dst->context;

    // Skip zero-size tensors
    ಯದಿ (ggml_nbytes(src) == 0) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (dst->buffer->buft != ggml_backend_vk_get_default_buffer_type(backend_dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

    ಯದಿ (ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;

        // Async copy only works within the same device
        ಯದಿ (src_buf_ctx->dev_buffer->device != dst_buf->device) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_buffer_copy_async(compute_ctx, dst_buf, vk_tensor_offset(dst) + dst->view_offs,
                                   src_buf_ctx->dev_buffer, vk_tensor_offset(src) + src->view_offs,
                                   ggml_nbytes(src));
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (ggml_backend_buffer_is_host(src->buffer)) {
        vk_buffer pinned_buf = nullptr;
        size_t pinned_offset = 0;
        ggml_vk_host_get(ctx->device, src->data, pinned_buf, pinned_offset);
        ಯದಿ (pinned_buf == nullptr) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // If the backend is idle, use a CPU copy to avoid GPU synchronization overhead.
        ಸ್ಥಿರ constexpr size_t max_cpu_copy_size = 128 * 1024;
        ಧ್ರುವ ತಾರ್ಕಿಕ src_backend_synchronous = backend_src->iface.synchronize == nullptr;
        ಧ್ರುವ ತಾರ್ಕಿಕ transfer_idle = !ctx->device->async_use_transfer_queue ||
                                   ctx->transfer_semaphore_last_submitted == ctx->transfer_semaphore.value;
        ಧ್ರುವ ತಾರ್ಕಿಕ backend_idle = ctx->compute_ctx.expired() && ctx->transfer_ctx.expired() &&
                                  !ctx->submit_pending && !ctx->almost_ready_fence_pending && transfer_idle;
        ಧ್ರುವ ತಾರ್ಕಿಕ dst_host_coherent =
            (dst_buf->memory_property_flags & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)) ==
            (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        ಯದಿ ((backend_src == backend_dst || src_backend_synchronous) && backend_idle && dst_host_coherent && ggml_nbytes(src) <= max_cpu_copy_size) {
            ggml_vk_buffer_write(dst_buf, vk_tensor_offset(dst) + dst->view_offs, src->data, ggml_nbytes(src));
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        vk_context cpy_ctx;
        ಯದಿ (ctx->device->async_use_transfer_queue) {
            cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
        } ಅನ್ಯಥಾ {
            cpy_ctx = ggml_vk_get_compute_ctx(ctx);
        }

        ಪ್ರತ್ಯಾಗಮ ggml_vk_buffer_write_async(cpy_ctx, dst_buf,
                                          vk_tensor_offset(dst) + dst->view_offs,
                                          src->data, ggml_nbytes(src));
    }

    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ಶೂನ್ಯ ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    ತಾರ್ಕಿಕ do_transfer = !ctx->compute_ctx.expired();

    ಯದಿ (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = ಸತ್ಯ;
    }

    vk_context compute_ctx;
    vk_command_buffer* cmd_buf = nullptr;
    ಯದಿ (do_transfer) {
        compute_ctx = ctx->compute_ctx.lock();
        ಯದಿ (compute_ctx->s) {
            cmd_buf = compute_ctx->s->buffer;
        }

        ggml_vk_ctx_end(compute_ctx);

        ಪ್ರತಿ (auto& cpy : compute_ctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        ಯದಿ (ctx->device->serialize_submissions) {
            ggml_vk_submit(compute_ctx, ctx->fence);
            VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ಸತ್ಯ, UINT64_MAX), "synchronize waitForFences", ctx->device);
            ctx->device->device.resetFences({ ctx->fence });
        } ಅನ್ಯಥಾ {
            ggml_vk_submit(compute_ctx, {});
        }
        ctx->submit_pending = ಸತ್ಯ;
    }

    ಯದಿ (ctx->submit_pending) {
        ಯದಿ (ctx->device->serialize_submissions) {
            ctx->submit_pending = ಅಸತ್ಯ;
        } ಅನ್ಯಥಾ ಯದಿ (ctx->device->async_use_transfer_queue && ctx->transfer_semaphore_last_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, nullptr,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_queue->stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, nullptr,
                0, nullptr,
            };
            si.setPNext(&tl_info);
            ctx->device->compute_queue->handle->submit({ si }, ctx->fence);
            ctx->transfer_semaphore_last_submitted = ctx->transfer_semaphore.value;
        } ಅನ್ಯಥಾ {
            ctx->device->compute_queue->handle->submit({}, ctx->fence);
        }
        ಯದಿ (!ctx->device->serialize_submissions) {
            ggml_vk_wait_for_fence(ctx);
        }
        ctx->submit_pending = ಅಸತ್ಯ;
        ಯದಿ (cmd_buf) {
            cmd_buf->in_use = ಅಸತ್ಯ;
            cmd_buf->buf.reset();
        }
    }

    ಯದಿ (do_transfer) {
        ಪ್ರತಿ (auto& cpy : compute_ctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);
}

ತಾರ್ಕಿಕ ggml_vk_is_empty(ggml_tensor * ಗಂಟು) {
    ಪ್ರತ್ಯಾಗಮ ggml_is_empty(ಗಂಟು) || ಗಂಟು->op == GGML_OP_NONE || ಗಂಟು->op == GGML_OP_RESHAPE || ಗಂಟು->op == GGML_OP_TRANSPOSE || ಗಂಟು->op == GGML_OP_VIEW || ಗಂಟು->op == GGML_OP_PERMUTE;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_can_fuse_unary_mul(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ unary_idx, ಪೂರ್ಣಾಙ್ಕ mul_idx) {
    ಧ್ರುವ ggml_tensor * unary = cgraph->nodes[unary_idx];
    ಧ್ರುವ ggml_tensor * mul = cgraph->nodes[mul_idx];

    ಯದಿ (ggml_vk_unary_mul_op_index(ggml_get_unary_op(unary)) < 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (unary->type != mul->type) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (mul->src[0] != unary && mul->src[1] != unary) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
    ಯದಿ (other == nullptr || other->type != unary->type) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0])) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    // fastmod needs src to tile into dst
    ಯದಿ (mul->src[0] == unary) {
        ಪ್ರತ್ಯಾಗಮ ggml_can_repeat(other, unary);
    }
    ಪ್ರತ್ಯಾಗಮ ggml_can_repeat(unary, mul->src[0]);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_can_fuse_unary_mul_pair(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ಗಣನಾ ggml_op ops[]    = { GGML_OP_UNARY, GGML_OP_MUL };
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ           outputs[] = { ಗಂಟು_idx + 1 };
    ಪ್ರತ್ಯಾಗಮ ggml_can_fuse_subgraph(cgraph, ಗಂಟು_idx, 2, ops, outputs, 1) &&
           ggml_vk_can_fuse_unary_mul(cgraph, ಗಂಟು_idx, ಗಂಟು_idx + 1);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_can_fuse_hc_post_gate(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ggml_tensor * scale_in  = cgraph->nodes[ಗಂಟು_idx];
    ಧ್ರುವ ggml_tensor * sigmoid   = cgraph->nodes[ಗಂಟು_idx + 1];
    ಧ್ರುವ ggml_tensor * scale_out = cgraph->nodes[ಗಂಟು_idx + 2];

    // the shader folds scale -> sigmoid -> scale; a bias on either scale is not handled
    ಪ್ರತ್ಯಾಗಮ ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
           ggml_get_op_params_f32(scale_in, 1) == 0.0f &&
           ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
           scale_in->src[0]->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(scale_in->src[0], scale_out);
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse(ಧ್ರುವ ggml_backend_vk_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, std::initializer_list<ಗಣನಾ ggml_op> ops) {
    ಯದಿ (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL) {
        ಪ್ರತ್ಯಾಗಮ ggml_vk_can_fuse_unary_mul_pair(cgraph, ಗಂಟು_idx);
    }

    ಯದಿ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, ops)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ ((ops.size() == 2 || ops.size() == 3 || ops.size() == 4) &&
        ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ಧ್ರುವ ggml_tensor *rms_norm = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        ಯದಿ (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        ಯದಿ (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // rms_norm shader assumes contiguous rows
        ಯದಿ (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (ops.size() >= 3 && ops.begin()[2] == GGML_OP_ADD) {
            ಧ್ರುವ ggml_tensor *add = cgraph->nodes[ಗಂಟು_idx + 2];
            ಧ್ರುವ ggml_tensor *residual = add->src[0] == mul ? add->src[1] : add->src[0];
            ಯದಿ (add->src[0] != mul && add->src[1] != mul) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (residual->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(add, residual) || !ggml_is_contiguous(residual) ||
                !ggml_is_contiguous(add) || get_misalign_bytes(ctx, residual) != 0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಧ್ರುವ ggml_tensor *dst = add;
            ಯದಿ (ops.size() == 4) {
                ಯದಿ (ops.begin()[3] != GGML_OP_MUL) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                ಧ್ರುವ ggml_tensor *post_mul = cgraph->nodes[ಗಂಟು_idx + 3];
                ಧ್ರುವ ggml_tensor *scale = post_mul->src[0] == add ? post_mul->src[1] : post_mul->src[0];
                ಯದಿ (post_mul->src[0] != add && post_mul->src[1] != add) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                // The shader reads data_e[0], so the final multiply must use a scalar.
                ಯದಿ (scale->type != GGML_TYPE_F32 || post_mul->type != GGML_TYPE_F32 ||
                    ggml_nelements(scale) != 1 || !ggml_is_contiguous(post_mul) ||
                    get_misalign_bytes(ctx, scale) != 0) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                dst = post_mul;
            }

            ಯದಿ (get_misalign_bytes(ctx, dst) != 0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
    }

    auto ಧ್ರುವ &mm_add_ok = [&](ಧ್ರುವ ggml_tensor *mul, ಧ್ರುವ ggml_tensor *add) {
        ಧ್ರುವ ggml_tensor *bias = add->src[0] == mul ? add->src[1] : add->src[0];

        // mat-vec only
        ಯದಿ (ggml_nrows(mul) != 1) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // shaders assume the types match
        ಯದಿ (mul->type != bias->type) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // shaders reuse the D shape for bias
        ಯದಿ (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // unaligned bias isn't handled
        ಯದಿ (get_misalign_bytes(ctx, bias) != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };

    ಯದಿ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ADD) {
        // additional constraints specific to this fusion
        ಧ್ರುವ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor *add = cgraph->nodes[ಗಂಟು_idx + 1];

        ಯದಿ (!mm_add_ok(mul, add)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (ops.size() == 3) {
            ಯದಿ (ops.begin()[2] != GGML_OP_ADD) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (!mm_add_ok(add, cgraph->nodes[ಗಂಟು_idx + 2])) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
    }

    auto ಧ್ರುವ &mmid_mul_ok = [&](ಧ್ರುವ ggml_tensor *mmid, ಧ್ರುವ ggml_tensor *mul) {
        ಧ್ರುವ ggml_tensor *scale = mul->src[1];

        ಯದಿ (mmid != mul->src[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // mat-vec only
        ಯದಿ (!ggml_vk_use_mul_mat_vec_id(cgraph, ಗಂಟು_idx)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // shaders assume the types match
        ಯದಿ (mmid->type != scale->type) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // shaders assume the bias is contiguous
        ಯದಿ (!ggml_is_contiguous(scale)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // unaligned bias isn't handled
        ಯದಿ (get_misalign_bytes(ctx, scale) != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // shader only indexes by expert index
        ಯದಿ (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };

    ಯದಿ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_ADD_ID) {
        // additional constraints specific to this fusion
        ಧ್ರುವ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor *add = cgraph->nodes[ಗಂಟು_idx + 1];
        ಧ್ರುವ ggml_tensor *bias = add->src[1];

        ಯದಿ (mul != add->src[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // mat-vec only
        ಯದಿ (!ggml_vk_use_mul_mat_vec_id(cgraph, ಗಂಟು_idx)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // shaders assume the types match
        ಯದಿ (mul->type != bias->type) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // shaders assume the bias is contiguous
        ಯದಿ (!ggml_is_contiguous(bias)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        ಯದಿ (mul->src[2] != add->src[2]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // unaligned bias isn't handled
        ಯದಿ (get_misalign_bytes(ctx, bias) != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (ops.size() == 3) {
            ಯದಿ (ops.begin()[2] != GGML_OP_MUL) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಧ್ರುವ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx + 2];
            ಪ್ರತ್ಯಾಗಮ mmid_mul_ok(add, mul);
        }
    }

    ಯದಿ (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ಧ್ರುವ ggml_tensor *mmid = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx + 1];

        ಯದಿ (!mmid_mul_ok(mmid, mul)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_ssm_conv(ಧ್ರುವ ggml_backend_vk_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph,
                                      ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_extra) {
    ಧ್ರುವ ggml_tensor * conv = cgraph->nodes[ಗಂಟು_idx];
    ಯದಿ (conv->op != GGML_OP_SSM_CONV) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ggml_tensor * silu = nullptr;
    ಧ್ರುವ ggml_tensor * bias = nullptr;

    ಯದಿ (ಅಂಕೆ_extra == 1) {
        ಯದಿ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        silu = cgraph->nodes[ಗಂಟು_idx + 1];
    } ಅನ್ಯಥಾ ಯದಿ (ಅಂಕೆ_extra == 2) {
        ಯದಿ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY })) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಧ್ರುವ ggml_tensor * add = cgraph->nodes[ಗಂಟು_idx + 1];
        silu = cgraph->nodes[ಗಂಟು_idx + 2];
        bias = (add->src[0] == conv) ? add->src[1] : add->src[0];

        ಯದಿ (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // bias must be channel-wise (one element per channel of the conv output)
        ಯದಿ (ggml_nelements(bias) != conv->ne[0] || bias->ne[0] != conv->ne[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (add->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        ಯದಿ (get_misalign_bytes(ctx, bias) != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    // The shader writes to the fused dst using its own strides, but the push constants don't
    // carry a per-tensor offset, so the binding must be naturally aligned.
    ಯದಿ (get_misalign_bytes(ctx, silu) != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph,
                                      ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, topk_moe_mode mode) {

    ಧ್ರುವ ggml_tensor * softmax;
    ಧ್ರುವ ggml_tensor * weights;
    ಧ್ರುವ ggml_tensor * get_rows;
    ಧ್ರುವ ggml_tensor * argsort;

    ವಿಕಲ್ಪ (mode) {
    ಪಕ್ಷ TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[ಗಂಟು_idx + 0];
        weights = cgraph->nodes[ಗಂಟು_idx + 9];
        get_rows = cgraph->nodes[ಗಂಟು_idx + 4];
        argsort = cgraph->nodes[ಗಂಟು_idx + 2];
        ಭಙ್ಗ;
    ಪಕ್ಷ TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[ಗಂಟು_idx + 0]; // really sigmoid
        weights = cgraph->nodes[ಗಂಟು_idx + 10];
        get_rows = cgraph->nodes[ಗಂಟು_idx + 5];
        argsort = cgraph->nodes[ಗಂಟು_idx + 3];
        ಯದಿ (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // bias is expected to be 1D
        ಯದಿ (ggml_nrows(cgraph->nodes[ಗಂಟು_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[ಗಂಟು_idx + 2]->src[1])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        ಯದಿ (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS:
        softmax = cgraph->nodes[ಗಂಟು_idx + 0]; // really softplus
        weights = cgraph->nodes[ಗಂಟು_idx + 11];
        get_rows = cgraph->nodes[ಗಂಟು_idx + 6];
        argsort = cgraph->nodes[ಗಂಟು_idx + 4];
        ಯದಿ (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SOFTPLUS) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // bias is expected to be 1D
        ಯದಿ (ggml_nrows(cgraph->nodes[ಗಂಟು_idx + 3]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[ಗಂಟು_idx + 3]->src[1])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಭಙ್ಗ;
    ಪಕ್ಷ TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[ಗಂಟು_idx + 0];
        weights = cgraph->nodes[ಗಂಟು_idx + 4];
        get_rows = cgraph->nodes[ಗಂಟು_idx + 4];
        argsort = cgraph->nodes[ಗಂಟು_idx + 2];
        ಭಙ್ಗ;
    ಪಕ್ಷ TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[ಗಂಟು_idx + 4];
        weights = cgraph->nodes[ಗಂಟು_idx + 5];
        get_rows = cgraph->nodes[ಗಂಟು_idx + 2];
        argsort = cgraph->nodes[ಗಂಟು_idx + 0];
        ಭಙ್ಗ;
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ggml_tensor * probs = get_rows->src[0];
    ಯದಿ (probs->op != GGML_OP_RESHAPE) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    ಯದಿ (probs != selection_probs &&
        mode != TOPK_MOE_SIGMOID_NORM_BIAS &&
        mode != TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (softmax->op == GGML_OP_SOFT_MAX) {
        ಧ್ರುವ ಭಿನ್ನ * op_params = (ಧ್ರುವ ಭಿನ್ನ *)softmax->op_params;

        ಭಿನ್ನ scale = op_params[0];
        ಭಿನ್ನ max_bias = op_params[1];

        ಯದಿ (scale != 1.0f || max_bias != 0.0f) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // don't fuse when masks or sinks are present
        ಯದಿ (softmax->src[1] || softmax->src[2]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_expert = softmax->ne[0];
    ಯದಿ (n_expert > (1 << (ಅಂಕೆ_topk_moe_pipelines-1))) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_full_support ||
        ctx->device->disable_fusion) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_match_ops(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                              ಧ್ರುವ std::initializer_list<ggml_op> & ops) {
    ಯದಿ (ಗಂಟು_idx + (ಪೂರ್ಣಾಙ್ಕ) ops.size() > cgraph->n_nodes) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತಿ (size_t j = 0; j < ops.size(); ++j) {
        ಧ್ರುವ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + j];
        ಯದಿ (ಗಂಟು->op != ops.begin()[j] ||
            (ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (ಗಂಟು->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಯದಿ (ctx->device->disable_fusion || !ctx->device->pipeline_topk_radix_qsa) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ops = topk_qsa_pattern.size();
    ಯದಿ (!ggml_vk_match_ops(cgraph, ಗಂಟು_idx, topk_qsa_pattern) ||
        !ggml_check_edges(cgraph, ಗಂಟು_idx, topk_qsa_edges)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < n_ops - 1; ++j) {
        ಧ್ರುವ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + j];
        ಧ್ರುವ int32_t want = ಗಂಟು->op == GGML_OP_CPY ? 2 : 1;
        ಯದಿ (ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, ಗಂಟು_idx + j) != want) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಧ್ರುವ ggml_tensor * get_rows = cgraph->nodes[ಗಂಟು_idx + 0];
    ಧ್ರುವ ggml_tensor * add      = cgraph->nodes[ಗಂಟು_idx + n_ops - 2];
    ಧ್ರುವ ggml_tensor * ತುದಿ_k    = cgraph->nodes[ಗಂಟು_idx + n_ops - 1];

    ಧ್ರುವ ggml_tensor * scores   = get_rows->src[0]; // [n_tps, n_blocks, n_stream]
    ಧ್ರುವ ggml_tensor * cell_blk = get_rows->src[1]; // [n_kv, n_stream]
    ಧ್ರುವ ggml_tensor * expanded = add->src[0];      // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    ಧ್ರುವ ggml_tensor * mask = add->src[1];
    ಯಾವತ್ (mask && (mask->op == GGML_OP_RESHAPE || mask->op == GGML_OP_CPY)) {
        mask = mask->src[0];
    }
    ಯದಿ (!mask || mask->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (scores->type != GGML_TYPE_F32 || cell_blk->type != GGML_TYPE_I32 || ತುದಿ_k->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_is_contiguous(scores) || !ggml_is_contiguous(cell_blk) || !ggml_is_contiguous(mask) ||
        !ggml_is_contiguous(expanded) || !ggml_is_contiguous(ತುದಿ_k)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t n_tps    = scores->ne[0];
    ಧ್ರುವ int64_t n_blocks = scores->ne[1];
    ಧ್ರುವ int64_t n_stream = scores->ne[2];
    ಧ್ರುವ int64_t n_kv     = cell_blk->ne[0];
    ಧ್ರುವ int64_t width    = ತುದಿ_k->ne[0];

    // pin the indexer layout the shader's addressing assumes
    ಯದಿ (scores->ne[3] != 1 || cell_blk->ne[1] != n_stream || ggml_nrows(cell_blk) != n_stream ||
        ggml_nelements(mask) != n_kv * n_tps * n_stream ||
        expanded->ne[0] != n_kv || expanded->ne[1] != n_tps || expanded->ne[2] != n_stream ||
        ತುದಿ_k->ne[1] != n_tps || ತುದಿ_k->ne[2] != n_stream || ತುದಿ_k->ne[3] != 1 ||
        n_blocks <= 0 || n_kv <= 0 || width <= 0 || width > n_kv) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // only worth it in the radix regime; small k uses the faster tournament unfused
    ಧ್ರುವ uint32_t k_min_pipeline = std::max((uint32_t) log2f(ಭಿನ್ನ(width)) + 1, ctx->device->subgroup_size_log2);
    ಯದಿ (k_min_pipeline < ಅಂಕೆ_topk_pipelines && ctx->device->pipeline_topk_f32[k_min_pipeline]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph,
                                           ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ggml_tensor *rope = cgraph->nodes[ಗಂಟು_idx + 0];
    ಧ್ರುವ ggml_tensor *view = cgraph->nodes[ಗಂಟು_idx + 1];
    ಧ್ರುವ ggml_tensor *set_rows = cgraph->nodes[ಗಂಟು_idx + 2];

    // The set_rows epilogue uses one index per ne2 slice and does not encode ne3.
    ಯದಿ (rope->src[0]->ne[3] != 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // The shader reads each aligned I64 index as a uvec2 and uses its low 32 bits.
    ಯದಿ (set_rows->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(set_rows->src[1]) ||
        set_rows->nb[0] != ggml_type_size(set_rows->type) || get_misalign_bytes(ctx, set_rows->src[1]) != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // SET_ROWS consumes one flattened [ne0*ne1] row for each ne2 slice.
    ಯದಿ (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1] || view->ne[1] != rope->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(set_rows->src[1]) != rope->ne[2]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Only norm/neox/mrope/imrope shaders have the fusion code
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode = ((ಧ್ರುವ int32_t *) rope->op_params)[2];
    ಯದಿ (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_rms_norm_set_rows(ggml_backend_vk_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph,
                                               ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ggml_tensor * rms = cgraph->nodes[ಗಂಟು_idx];
    ಧ್ರುವ ggml_tensor * view = cgraph->nodes[ಗಂಟು_idx + 1];
    ಧ್ರುವ ggml_tensor * set_rows = cgraph->nodes[ಗಂಟು_idx + 2];

    // The RMS kernel reads F32 and writes directly to the F32 or F16 SET_ROWS destination.
    ಯದಿ (rms->src[0]->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) ||
        set_rows->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(set_rows->src[1]) ||
        set_rows->nb[0] != ggml_type_size(set_rows->type) || get_misalign_bytes(ctx, set_rows->src[1]) != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    // As with the ROPE epilogue, each ne2 slice supplies one flattened row and ne3 is not encoded.
    ಯದಿ (rms->ne[3] != 1 || !ggml_is_contiguous(rms->src[0]) || !ggml_is_contiguous(view)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (view->ne[0] != rms->ne[0] * rms->ne[1] || view->ne[1] != rms->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(set_rows->src[1]) != rms->ne[2]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    GGML_UNUSED(ctx);
    ಯದಿ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, snake_pattern)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ggml_tensor * mul0     = cgraph->nodes[ಗಂಟು_idx + 0];
    ಧ್ರುವ ggml_tensor * sin_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + 1];
    ಧ್ರುವ ggml_tensor * sqr      = cgraph->nodes[ಗಂಟು_idx + 2];
    ಧ್ರುವ ggml_tensor * mul1     = cgraph->nodes[ಗಂಟು_idx + 3];
    ಧ್ರುವ ggml_tensor * add      = cgraph->nodes[ಗಂಟು_idx + 4];

    ಧ್ರುವ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ಧ್ರುವ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    ಧ್ರುವ ggml_tensor * inv_b    = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];
    ಧ್ರುವ ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    ಯದಿ (x_in_add != x) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    // Shader bindings: data_a is A_TYPE so it follows x's precision, while
    // data_b and data_c are hardcoded float, so the broadcast operands must
    // be F32 regardless of x's type.
    ಯದಿ (a->type     != GGML_TYPE_F32) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (inv_b->type != GGML_TYPE_F32) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    // Chain intermediates and output share x's precision (single A_TYPE / D_TYPE pipeline).
    ಯದಿ (mul0->type     != x->type) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (sin_ಗಂಟು->type != x->type) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (sqr->type      != x->type) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (mul1->type     != x->type) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (add->type      != x->type) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (!ggml_are_same_shape(a, inv_b)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    // Dispatch is 2D over (ne0, ne1), so x and add must be 2D and a / inv_b
    // must collapse to [1, C, 1, 1]. Higher dims are not handled by the shader.
    ಯದಿ (x->ne[2]     != 1 || x->ne[3]     != 1) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (add->ne[2]   != 1 || add->ne[3]   != 1) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (a->ne[2]     != 1 || a->ne[3]     != 1) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (inv_b->ne[2] != 1 || inv_b->ne[3] != 1) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    // Shader uses idx = i0 + i1 * ne0 and reads data_b[i1] / data_c[i1],
    // so every operand must be contiguous.
    ಯದಿ (!ggml_is_contiguous(x) || !ggml_is_contiguous(add) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_vk_tensors_overlap(ಧ್ರುವ ggml_tensor * a, ಧ್ರುವ ggml_tensor * b, ತಾರ್ಕಿಕ elementwise) {
    ggml_backend_vk_buffer_context * a_buf_ctx = (ggml_backend_vk_buffer_context *)a->buffer->context;
    vk_buffer a_buf = a_buf_ctx->dev_buffer;
    ggml_backend_vk_buffer_context * b_buf_ctx = (ggml_backend_vk_buffer_context *)b->buffer->context;
    vk_buffer b_buf = b_buf_ctx->dev_buffer;
    ಯದಿ (a_buf == b_buf) {
        auto a_base = vk_tensor_offset(a) + a->view_offs;
        auto a_size = ggml_nbytes(a);
        auto b_base = vk_tensor_offset(b) + b->view_offs;
        auto b_size = ggml_nbytes(b);

        ಯದಿ (elementwise && a_base == b_base && a_size == b_size) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ ((b_base <= a_base && a_base < b_base + b_size) ||
            (a_base <= b_base && b_base < a_base + a_size)) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph,
                                               ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ggml_tensor *rms = cgraph->nodes[ಗಂಟು_idx + 0];
    ಧ್ರುವ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx + 1];
    ಧ್ರುವ ggml_tensor *rope = cgraph->nodes[ಗಂಟು_idx + 2];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode = ((ಧ್ರುವ int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    ಯದಿ (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // only norm/neox are handled in the shader
    ಯದಿ (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_NORMAL) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // shared memory size for passing data from mul->rope
    ಯದಿ (mul->ne[0] > 1024) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // conditions for pipeline creation
    ಯದಿ (ಪರಿಮಾಣ(vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants) > ctx->device->properties.limits.max_ತಳ್ಳು_Constants_Size) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

uint32_t ggml_vk_fuse_multi_add(ggml_backend_vk_context * ctx, ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {

    ಧ್ರುವ ggml_tensor *first_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];
    ಯದಿ (first_ಗಂಟು->op != GGML_OP_ADD) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (!ctx->device->multi_add) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    int32_t ಅಂಕೆ_adds = 1;
    ಯಾವತ್ (ಗಂಟು_idx + ಅಂಕೆ_adds < cgraph->n_nodes &&
           cgraph->nodes[ಗಂಟು_idx + ಅಂಕೆ_adds]->op == GGML_OP_ADD &&
           ಅಂಕೆ_adds < MAX_FUSED_ADDS) {
        ಅಂಕೆ_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    ಪ್ರತಿ (int32_t i = 0; i < ಅಂಕೆ_adds; ++i) {
        ಧ್ರುವ ggml_tensor *next_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + i];
        ಯದಿ (!ggml_are_same_shape(first_ಗಂಟು, next_ಗಂಟು->src[0]) ||
            !ggml_are_same_shape(first_ಗಂಟು, next_ಗಂಟು->src[1]) ||
            next_ಗಂಟು->type != GGML_TYPE_F32 ||
            next_ಗಂಟು->src[0]->type != GGML_TYPE_F32 ||
            next_ಗಂಟು->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_bytes(ctx, next_ಗಂಟು) ||
            get_misalign_bytes(ctx, next_ಗಂಟು->src[0]) ||
            get_misalign_bytes(ctx, next_ಗಂಟು->src[1])) {
            ಅಂಕೆ_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    ಪ್ರತಿ (int32_t i = 0; i < ಅಂಕೆ_adds; ++i) {
        adds[i] = GGML_OP_ADD;
    }

    // decrease num_adds if they can't all be fused
    ಯಾವತ್ (ಅಂಕೆ_adds > 1 && !ggml_can_fuse(cgraph, ಗಂಟು_idx, adds, ಅಂಕೆ_adds)) {
        ಅಂಕೆ_adds--;
    }

    // a single add is not "fused", so just return zero
    ಯದಿ (ಅಂಕೆ_adds == 1) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ಪ್ರತ್ಯಾಗಮ ಅಂಕೆ_adds;
}

ಸ್ಥಿರ int32_t find_first_set(uint32_t x) {
    int32_t ret = 0;
    ಯದಿ (!x) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }
    ಯಾವತ್ (!(x & 1)) {
        x >>= 1;
        ret++;
    }
    ಪ್ರತ್ಯಾಗಮ ret;
}

ಸ್ಥಿರ ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ctx->device->diag_cgraph = nullptr;
    ctx->device->diag_prev_start = -1;
    ctx->device->diag_prev_end = -1;

    ಯದಿ (vk_instance.debug_utils_support) {
        ctx->device->debug_cmdbuf_idx = 0;
    }

    // queue scope, so it encloses every submit this evaluation makes.
    // closed when the function returns
    ggml_vk_debug_label queue_dbg(ctx->device->compute_queue->handle.get(), "ggml_backend_vk_graph_compute");

    ctx->prealloc_size_add_rms_partials_offset = 0;
    ctx->do_add_rms_partials = ಅಸತ್ಯ;
    ctx->do_add_rms_partials_offset_calculation = ಅಸತ್ಯ;

    ಪೂರ್ಣಾಙ್ಕ last_ಗಂಟು = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    ಯಾವತ್ (last_ಗಂಟು > 0 && (ggml_vk_is_empty(cgraph->nodes[last_ಗಂಟು]) || ((cgraph->nodes[last_ಗಂಟು]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0))) {
        last_ಗಂಟು -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    ತಾರ್ಕಿಕ first_ಗಂಟು_in_batch = ಸತ್ಯ; // true if next node will be first node in a batch
    ಪೂರ್ಣಾಙ್ಕ submit_ಗಂಟು_idx = 0; // index to first node in a batch

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx;
    ಯದಿ (vk_perf_logger_enabled) {
        // allocate/resize the query pool
        ಯದಿ (ctx->ಅಂಕೆ_queries < cgraph->n_nodes + 1) {
            ಯದಿ (ctx->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.query_ಎಣಿಕೆ = cgraph->n_nodes + 100;
            ctx->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->ಅಂಕೆ_queries = query_create_info.query_ಎಣಿಕೆ;
            ctx->query_fusion_names.resize(ctx->ಅಂಕೆ_queries);
            ctx->query_fusion_ಗಂಟು_ಎಣಿಕೆ.resize(ctx->ಅಂಕೆ_queries);
            ctx->query_nodes.resize(ctx->ಅಂಕೆ_queries);
            ctx->query_ಗಂಟು_idx.resize(ctx->ಅಂಕೆ_queries);
        }

        ctx->device->device.resetQueryPool(ctx->query_pool, 0, cgraph->n_nodes+1);
        std::fill(ctx->query_fusion_names.begin(), ctx->query_fusion_names.end(), nullptr);
        std::fill(ctx->query_fusion_ಗಂಟು_ಎಣಿಕೆ.begin(), ctx->query_fusion_ಗಂಟು_ಎಣಿಕೆ.end(), 0);
        std::fill(ctx->query_nodes.begin(), ctx->query_nodes.end(), nullptr);
        std::fill(ctx->query_ಗಂಟು_idx.begin(), ctx->query_ಗಂಟು_idx.end(), 0);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
        compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    ctx->prealloc_y_last_pipeline_used = nullptr;
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_k_padded = ಅಸತ್ಯ;

    ಯದಿ (ctx->prealloc_size_add_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, nullptr);
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        // initialize partial sums to zero.
        ggml_vk_buffer_memset_async(compute_ctx, ctx->prealloc_add_rms_partials, 0, 0, ctx->prealloc_size_add_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of compute work using flops, and submit every 200 GFLOP
    // (and scaled down based on total graph flops, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without heavy compute.
    uint32_t submitted_nodes = 0;
    uint32_t submit_ಎಣಿಕೆ = 0;
    uint64_t batch_flops = 0;
    uint64_t total_flops = 0;
    uint64_t flops_cap = 200'000'000'000ULL;

    // On weaker AMD GPUs larger submissions can hit a driver timeout, submit more often to avoid this
    ಯದಿ (ctx->device->vendor_id == VK_VENDOR_ID_AMD && ctx->device->shader_core_ಎಣಿಕೆ > 0) {
        ಯದಿ (ctx->device->architecture == AMD_GCN && ctx->device->shader_core_ಎಣಿಕೆ < 32) {
            flops_cap = 500'000'000ULL * ctx->device->shader_core_ಎಣಿಕೆ;
        } ಅನ್ಯಥಾ ಯದಿ (ctx->device->architecture != AMD_GCN && ctx->device->shader_core_ಎಣಿಕೆ < 24) {
            flops_cap = 2'000'000'000ULL * ctx->device->shader_core_ಎಣಿಕೆ;
        }
    }
    uint64_t flops_per_submit = std::min(flops_cap, ctx->last_total_flops / 40u);

    auto ಧ್ರುವ submit_after = [&](ಪೂರ್ಣಾಙ್ಕ start, ಪೂರ್ಣಾಙ್ಕ end) {
        ಯದಿ (ctx->device->serialize_submissions) {
            ಯತ್ನ {
                auto res = ctx->device->device.waitForFences({ ctx->fence }, ಸತ್ಯ, UINT64_MAX);
                ಯದಿ (res != vk::Result::eSuccess) {
                    GGML_LOG_ERROR("ggml_vulkan: waitForFences error during serialized submission\n");
                    ಕ್ಷಿಪ vk::SystemError(vk::make_error_code(res), "ggml_vulkan: waitForFences during serialized submission");
                }
            } ಗೃಹಾಣ (vk::DeviceLostError &) {
                ggml_vk_print_device_fault_info(ctx->device);
                GGML_LOG_ERROR("ggml_vulkan: device lost on %s waiting for submission (nodes %d to %d):\n",
                        ctx->device->name.c_str(), start, end);
                ggml_vk_print_ಗಂಟು_list(cgraph, start, end);
                ಕ್ಷಿಪ;
            }
            ctx->device->device.resetFences({ ctx->fence });
            ctx->submit_pending = ಅಸತ್ಯ;
            ctx->device->diag_cgraph = cgraph;
            ctx->device->diag_prev_start = start;
            ctx->device->diag_prev_end = end;
        }
        first_ಗಂಟು_in_batch = ಸತ್ಯ;
        submitted_nodes = 0;
        batch_flops = 0;
        ಯದಿ (submit_ಎಣಿಕೆ < 3) {
            flops_per_submit *= 2;
        }
        submit_ಎಣಿಕೆ++;
    };

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < cgraph->n_nodes; i++) {
        ಯದಿ (first_ಗಂಟು_in_batch) {
            submit_ಗಂಟು_idx = i;
        }

        {
            auto ಗಂಟು_flops = ggml_vk_get_ಗಂಟು_flops(cgraph->nodes[i]);
            total_flops += ಗಂಟು_flops;

            // Flush the current batch before recording a node that would push it over the flop threshold
            ಯದಿ (flops_per_submit != 0 && submitted_nodes > 0 && batch_flops + ಗಂಟು_flops >= flops_per_submit) {
                vk_context flush_ctx = ggml_vk_get_compute_ctx(ctx);
                ggml_vk_ctx_end(flush_ctx);
                flush_ctx->exit_tensor_idx = -1;
                ctx->compute_ctx.reset();
                ggml_vk_compute_forward(ctx, cgraph, cgraph->nodes[submit_ಗಂಟು_idx], submit_ಗಂಟು_idx, ಅಸತ್ಯ);
                submit_after(submit_ಗಂಟು_idx, i - 1);
                submit_ಗಂಟು_idx = i;
            }

            batch_flops += ಗಂಟು_flops;
        }

        // op_srcs_fused_elementwise indicates whether an op's srcs all contribute to
        // the fused result in an elementwise-way. This affects whether the memory for
        // the src is allowed to overlap the memory for the destination.
        // The array is sized to handle the largest fusion (asserted later).
        ತಾರ್ಕಿಕ op_srcs_fused_elementwise[13];

        ctx->fused_topk_moe_mode = TOPK_MOE_ಎಣಿಕೆ;
        ctx->fused_topk_moe_scale = ಅಸತ್ಯ;
        ctx->fused_topk_qsa = ಅಸತ್ಯ;
        ctx->fused_hc_post_gate = ಅಸತ್ಯ;
        ctx->fused_rms_norm_mode = RMS_NORM_ಎಣಿಕೆ;
        ಧ್ರುವ ವರ್ಣ *fusion_string {};
        ಯದಿ (!ctx->device->disable_fusion) {
            uint32_t ಅಂಕೆ_adds = ggml_vk_fuse_multi_add(ctx, cgraph, i);
            ಯದಿ (ಅಂಕೆ_adds) {
                ctx->ಅಂಕೆ_additional_fused_ops = ಅಂಕೆ_adds - 1;
                fusion_string = "MULTI_ADD";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
                op_srcs_fused_elementwise[0] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಸತ್ಯ;
                op_srcs_fused_elementwise[2] = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
                op_srcs_fused_elementwise[0] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
                op_srcs_fused_elementwise[0] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಸತ್ಯ;
                op_srcs_fused_elementwise[2] = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
                op_srcs_fused_elementwise[0] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
                op_srcs_fused_elementwise[0] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, rms_norm_mul_rope_view_set_rows_pattern, { i + 4 }) &&
                       ggml_check_edges(cgraph, i, rms_norm_mul_rope_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_set_rows(ctx, cgraph, i + 2)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 4;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ROPE_VIEW_SET_ROWS;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[2] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[3] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[4] = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ROPE;
                fusion_string = "RMS_NORM_MUL_ROPE";
                // rope is approximately elementwise - whole rows are done by a single workgroup and it's row-wise
                op_srcs_fused_elementwise[0] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಸತ್ಯ;
                op_srcs_fused_elementwise[2] = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, hc_post_gate_pattern, { i + 3 }) &&
                       ggml_check_edges(cgraph, i, hc_post_gate_edges) &&
                       ggml_vk_can_fuse_hc_post_gate(cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = hc_post_gate_pattern.size() - 1;
                ctx->fused_hc_post_gate = ಸತ್ಯ;
                fusion_string = "HC_POST_GATE";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಅಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_add_mul_pattern)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 3;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ADD_MUL;
                fusion_string = "RMS_NORM_MUL_ADD_MUL";
                std::fill_n(op_srcs_fused_elementwise, 4, ಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_add_pattern)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ADD;
                fusion_string = "RMS_NORM_MUL_ADD";
                std::fill_n(op_srcs_fused_elementwise, 3, ಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, rms_norm_view_set_rows_pattern, { i + 2 }) &&
                       ggml_check_edges(cgraph, i, rms_norm_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rms_norm_set_rows(ctx, cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                ctx->fused_rms_norm_mode = RMS_NORM_VIEW_SET_ROWS;
                fusion_string = "RMS_NORM_VIEW_SET_ROWS";
                std::fill_n(op_srcs_fused_elementwise, 3, ಅಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL;
                fusion_string = "RMS_NORM_MUL";
                // rms_norm is not elementwise, but whole rows must be consumed and the scale factor computed before
                // they are overwritten, and one workgroup per row. So close enough.
                op_srcs_fused_elementwise[0] = ಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                ವಿಕಲ್ಪ (ggml_get_unary_op(cgraph->nodes[i])) {
                    ಪಕ್ಷ GGML_UNARY_OP_GELU:     fusion_string = "GELU_MUL";     ಭಙ್ಗ;
                    ಪಕ್ಷ GGML_UNARY_OP_SIGMOID:  fusion_string = "SIGMOID_MUL";  ಭಙ್ಗ;
                    ಪಕ್ಷ GGML_UNARY_OP_SILU:     fusion_string = "SILU_MUL";     ಭಙ್ಗ;
                    ಸ್ವಭಾವ:                     fusion_string = "SOFTPLUS_MUL"; ಭಙ್ಗ;
                }
                op_srcs_fused_elementwise[0] = ಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                fusion_string = "SSM_CONV_BIAS_SILU";
                // ssm_conv reads multiple input tokens per output, so it's not elementwise w.r.t. its srcs.
                // The downstream add and silu are elementwise on the conv output.
                op_srcs_fused_elementwise[0] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಸತ್ಯ;
                op_srcs_fused_elementwise[2] = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 1)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                fusion_string = "SSM_CONV_SILU";
                op_srcs_fused_elementwise[0] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, rope_view_set_rows_pattern, { i + 2 }) &&
                       ggml_check_edges(cgraph, i, rope_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rope_set_rows(ctx, cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[1] = ಅಸತ್ಯ;
                op_srcs_fused_elementwise[2] = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 4;
                fusion_string = "SNAKE";
                // elementwise=true: snake.comp is safe under exact aliasing because each
                // thread reads data_x[idx] into a register before writing data_d[idx]
                // with a data dependency on that register. The overlap check still
                // rejects partial overlaps (different base or size).
                std::fill_n(op_srcs_fused_elementwise, 5, ಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_vk_can_fuse_topk_qsa(ctx, cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_qsa_pattern.size() - 1;
                ctx->fused_topk_qsa = ಸತ್ಯ;
                fusion_string = "TOPK_QSA";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಅಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_early_softmax_norm_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 3;
                ctx->fused_topk_moe_mode = TOPK_MOE_EARLY_SOFTMAX_NORM;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಅಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sigmoid_norm_bias, { i + 4, i + 10 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_sigmoid_norm_bias_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SIGMOID_NORM_BIAS)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_moe_sigmoid_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 4;
                ctx->fused_topk_moe_mode = TOPK_MOE_SIGMOID_NORM_BIAS;
                fusion_string = "TOPK_MOE_SIGMOID_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಅಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sqrt_softplus_norm_bias, { i + 5, i + 11 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_sqrt_softplus_norm_bias_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_moe_sqrt_softplus_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 5;
                ctx->fused_topk_moe_mode = TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
                fusion_string = "TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಅಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_early_softmax_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 3;
                ctx->fused_topk_moe_mode = TOPK_MOE_EARLY_SOFTMAX;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಅಸತ್ಯ);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_late_softmax_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 1;
                ctx->fused_topk_moe_mode = TOPK_MOE_LATE_SOFTMAX;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಅಸತ್ಯ);
            }
            ಯದಿ (ctx->fused_topk_moe_mode != TOPK_MOE_ಎಣಿಕೆ) {
                // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
                ಯದಿ (ggml_can_fuse_subgraph(cgraph, i + ctx->ಅಂಕೆ_additional_fused_ops - 1, { GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE }, { i + ctx->ಅಂಕೆ_additional_fused_ops + 1 }) ||
                    ggml_can_fuse_subgraph(cgraph, i + ctx->ಅಂಕೆ_additional_fused_ops, { GGML_OP_GET_ROWS, GGML_OP_SCALE }, { i + ctx->ಅಂಕೆ_additional_fused_ops + 1 })) {
                    ctx->fused_topk_moe_scale = ಸತ್ಯ;
                    ctx->ಅಂಕೆ_additional_fused_ops++;
                    op_srcs_fused_elementwise[ctx->ಅಂಕೆ_additional_fused_ops] = ಅಸತ್ಯ;
                }
            }
        }
        GGML_ASSERT(ctx->ಅಂಕೆ_additional_fused_ops < (ಪೂರ್ಣಾಙ್ಕ)(ಪರಿಮಾಣ(op_srcs_fused_elementwise) / ಪರಿಮಾಣ(op_srcs_fused_elementwise[0])));
        ctx->fused_ops_write_mask |= 1 << ctx->ಅಂಕೆ_additional_fused_ops;

        // Check whether fusion would overwrite src operands while they're still in use.
        // If so, disable fusion.
        ಯದಿ (ctx->ಅಂಕೆ_additional_fused_ops) {
            // There are up to two output nodes - topk_moe has two.
            uint32_t bits = ctx->fused_ops_write_mask & ~(1 << ctx->ಅಂಕೆ_additional_fused_ops);
            ggml_tensor *output_nodes[2] {};
            output_nodes[0] = cgraph->nodes[i + ctx->ಅಂಕೆ_additional_fused_ops];
            ಯದಿ (bits) {
                ಪೂರ್ಣಾಙ್ಕ output_idx = find_first_set(bits);
                GGML_ASSERT(bits == (1u << output_idx));
                output_nodes[1] = cgraph->nodes[i + output_idx];
            }

            ತಾರ್ಕಿಕ need_disable = ಅಸತ್ಯ;

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 2; ++j) {
                ggml_tensor *dst = output_nodes[j];
                ಯದಿ (!dst) {
                    ಅನುವರ್ತ;
                }
                // Loop over all srcs of all nodes in the fusion. If the src overlaps
                // the destination and the src is not an intermediate node that's being
                // elided, then disable fusion.
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k <= ctx->ಅಂಕೆ_additional_fused_ops; ++k) {
                    ಪ್ರತಿ (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                        ggml_tensor *src = cgraph->nodes[i + k]->src[s];
                        ಯದಿ (!src || src->op == GGML_OP_NONE) {
                            ಅನುವರ್ತ;
                        }
                        ಯದಿ (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {
                            ತಾರ್ಕಿಕ found = ಅಸತ್ಯ;
                            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ n = 0; n < k; ++n) {
                                ಯದಿ (cgraph->nodes[i + n] == src) {
                                    found = ಸತ್ಯ;
                                    ಭಙ್ಗ;
                                }
                            }
                            ಯದಿ (!found) {
                                need_disable = ಸತ್ಯ;
                            }
                        }
                    }
                }
            }
            ಯದಿ (need_disable) {
                ctx->ಅಂಕೆ_additional_fused_ops = 0;
                ctx->fused_ops_write_mask = 1;
                ctx->fused_topk_moe_mode = TOPK_MOE_ಎಣಿಕೆ;
                ctx->fused_topk_moe_scale = ಅಸತ್ಯ;
                ctx->fused_topk_qsa = ಅಸತ್ಯ;
                ctx->fused_hc_post_gate = ಅಸತ್ಯ;
                ctx->fused_rms_norm_mode = RMS_NORM_ಎಣಿಕೆ;
                fusion_string = nullptr;
            }
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        ತಾರ್ಕಿಕ almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        ತಾರ್ಕಿಕ submit = (submitted_nodes >= ctx->device->max_nodes_per_submit) ||
                      (flops_per_submit != 0 && batch_flops >= flops_per_submit) ||
                      (i + ctx->ಅಂಕೆ_additional_fused_ops >= last_ಗಂಟು) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        ತಾರ್ಕಿಕ enqueued = ggml_vk_build_graph(ctx, cgraph, i, cgraph->nodes[submit_ಗಂಟು_idx], submit_ಗಂಟು_idx, i + ctx->ಅಂಕೆ_additional_fused_ops >= last_ಗಂಟು, almost_ready, submit);

        ಯದಿ (vk_perf_logger_enabled && enqueued) {
            compute_ctx = ggml_vk_get_compute_ctx(ctx);
            ಯದಿ (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_names[ctx->query_idx] = fusion_string;
                compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            } ಅನ್ಯಥಾ {
                // track a fusion string and number of fused ops for the current node_idx
                ctx->query_fusion_names[i] = fusion_string;
                ctx->query_fusion_ಗಂಟು_ಎಣಿಕೆ[i] = ctx->ಅಂಕೆ_additional_fused_ops;
            }
        }

        ಯದಿ (enqueued) {
            ++submitted_nodes;

#ifndef GGML_VULKAN_CHECK_RESULTS
            ಯದಿ (first_ಗಂಟು_in_batch) {
                first_ಗಂಟು_in_batch = ಅಸತ್ಯ;
            }
#endif
        }

        ಯದಿ (submit && enqueued) {
            submit_after(submit_ಗಂಟು_idx, i + (ಪೂರ್ಣಾಙ್ಕ)ctx->ಅಂಕೆ_additional_fused_ops);
        }
        i += ctx->ಅಂಕೆ_additional_fused_ops;
        ctx->ಅಂಕೆ_additional_fused_ops = 0;
        ctx->fused_ops_write_mask = 0;
    }

    ctx->last_total_flops = total_flops;

    ಯದಿ (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.lock();
        ggml_vk_ctx_end(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, ಸತ್ಯ, UINT64_MAX), "GGML_VULKAN_PERF waitForFences", ctx->device);
        ctx->device->device.resetFences({ ctx->device->fence });
        ctx->compute_ctx.reset();

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*ಪರಿಮಾಣ(uint64_t), timestamps.data(), ಪರಿಮಾಣ(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait), "get timestamp results", ctx->device);
        ಯದಿ (!vk_perf_logger_concurrent) {
            // Log each op separately
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < ctx->query_idx; i++) {
                auto ಗಂಟು = ctx->query_nodes[i];
                auto name = ctx->query_fusion_names[i];
                ctx->perf_logger->log_timing(ಗಂಟು, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        } ಅನ್ಯಥಾ {
            // Log each group of nodes
            ಪೂರ್ಣಾಙ್ಕ prev_ಗಂಟು_idx = 0;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < ctx->query_idx; i++) {
                auto cur_ಗಂಟು_idx = ctx->query_ಗಂಟು_idx[i];
                std::vector<ggml_tensor *> nodes;
                std::vector<ಧ್ರುವ ವರ್ಣ *> names;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx = prev_ಗಂಟು_idx; ಗಂಟು_idx < cur_ಗಂಟು_idx; ++ಗಂಟು_idx) {
                    ಯದಿ (ggml_op_is_empty(cgraph->nodes[ಗಂಟು_idx]->op)) {
                        ಅನುವರ್ತ;
                    }
                    nodes.push_back(cgraph->nodes[ಗಂಟು_idx]);
                    names.push_back(ctx->query_fusion_names[ಗಂಟು_idx]);
                    ಗಂಟು_idx += ctx->query_fusion_ಗಂಟು_ಎಣಿಕೆ[ಗಂಟು_idx];
                }
                prev_ಗಂಟು_idx = cur_ಗಂಟು_idx;
                ctx->perf_logger->log_timing(nodes, names, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        }
        ctx->perf_logger->print_timings();
    }

    ಯದಿ (!ctx->device->support_async) {
        ggml_vk_synchronize(ctx);
    }

    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;

    UNUSED(backend);
}

ಶೂನ್ಯ ggml_vk_graph_optimize(ggml_backend_t backend, ಸಂರಚನಾ ggml_cgraph * graph, ಸಂರಚನಾ ggml_backend_graph_optimize_params * params)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ಯದಿ (ctx->device->disable_graph_optimize) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    auto ಧ್ರುವ &is_empty = [](ಧ್ರುವ ggml_tensor * ಗಂಟು) -> ತಾರ್ಕಿಕ {
        ಪ್ರತ್ಯಾಗಮ ಗಂಟು->op == GGML_OP_NONE || ಗಂಟು->op == GGML_OP_RESHAPE || ಗಂಟು->op == GGML_OP_TRANSPOSE || ಗಂಟು->op == GGML_OP_VIEW || ಗಂಟು->op == GGML_OP_PERMUTE;
    };

    auto ಧ್ರುವ &is_src_of = [&is_empty](ಧ್ರುವ ggml_tensor *dst, ಧ್ರುವ ggml_tensor *src) -> ತಾರ್ಕಿಕ {
        auto ಧ್ರುವ &base = [](ಧ್ರುವ ggml_tensor * tensor) {
            ಪ್ರತ್ಯಾಗಮ tensor->view_src ? tensor->view_src : tensor;
        };
        ಪ್ರತಿ (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            ಯದಿ (dst->src[s] == src) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
            ಯದಿ (is_empty(dst) || is_empty(src)) {
                ಅನುವರ್ತ;
            }
            // A source view of dst may read storage written through a different view by src.
            ಯದಿ (dst->src[s] && base(dst->src[s]) == base(src)) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
            // Moving dst forward may overwrite storage still read through a view by src.
            ಯದಿ (src->src[s] && base(dst) == base(src->src[s])) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }
        // implicit dependency if they view the same tensor
        ಯದಿ (base(dst) == base(src)) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    };

    std::vector<ggml_tensor *> new_order;
    std::vector<ತಾರ್ಕಿಕ> used(graph->n_nodes, ಅಸತ್ಯ);
    std::set<ggml_tensor *> used_ಗಂಟು_set;

    ಪೂರ್ಣಾಙ್ಕ first_unused = 0;

    // scheduled or zero-compute nodes in [lo, hi)
    auto ಧ್ರುವ &empty_or_scheduled_between = [&](ಪೂರ್ಣಾಙ್ಕ lo, ಪೂರ್ಣಾಙ್ಕ hi) -> ತಾರ್ಕಿಕ {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ v = lo; v < hi; ++v) {
            ಯದಿ (!used[v] && !is_empty(graph->nodes[v])) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };
    ಯಾವತ್ (first_unused < graph->n_nodes) {
        std::vector<ಪೂರ್ಣಾಙ್ಕ> current_set;

        // Check for fusion patterns and avoid reordering them
        auto ಧ್ರುವ &match_pattern = [&](ಧ್ರುವ std::initializer_list<ggml_op> &pattern, ಪೂರ್ಣಾಙ್ಕ start) -> ತಾರ್ಕಿಕ {
            ಯದಿ (start + (ಪೂರ್ಣಾಙ್ಕ)pattern.size() <= graph->n_nodes) {
                ತಾರ್ಕಿಕ is_pattern = ಸತ್ಯ;
                ಪ್ರತಿ (size_t j = 0; j < pattern.size(); ++j) {
                    ಯದಿ (graph->nodes[start + j]->op != pattern.begin()[j] || used[start + j]) {
                        is_pattern = ಅಸತ್ಯ;
                    }
                }
                ಪ್ರತ್ಯಾಗಮ is_pattern;
            }
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        };

        auto ಧ್ರುವ &keep_pattern = [&](ಧ್ರುವ std::initializer_list<ggml_op> &pattern) -> ತಾರ್ಕಿಕ {
            ಯದಿ (match_pattern(pattern, first_unused)) {
                ಪ್ರತಿ (size_t j = 0; j < pattern.size(); ++j) {
                    new_order.push_back(graph->nodes[first_unused + j]);
                    used_ಗಂಟು_set.insert(graph->nodes[first_unused + j]);
                    used[first_unused + j] = ಸತ್ಯ;
                }
                ಯಾವತ್ (first_unused < graph->n_nodes && used[first_unused]) {
                    first_unused++;
                }
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        };

        auto ಧ್ರುವ &add_pattern_alloc_deps = [&](ಧ್ರುವ std::initializer_list<ggml_op> &pattern, ಪೂರ್ಣಾಙ್ಕ last_ಗಂಟು) {
            // Keep external inputs alive through the fused output.
            std::set<ggml_tensor *> seen;
            ಪ್ರತಿ (size_t j = 0; j < pattern.size(); ++j) {
                ggml_tensor * ಗಂಟು = graph->nodes[first_unused + j];
                ಪ್ರತಿ (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                    ggml_tensor * src = ಗಂಟು->src[s];
                    ಯದಿ (src && seen.insert(src).second) {
                        params->add_alloc_dep(params->user_data, src, graph->nodes[last_ಗಂಟು]);
                    }
                }
                seen.insert(ಗಂಟು);
            }
        };

        auto ಧ್ರುವ &keep_topk_moe_pattern = [&](ಧ್ರುವ std::initializer_list<ggml_op> &pattern) -> ತಾರ್ಕಿಕ {
            ಯದಿ (!match_pattern(pattern, first_unused)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಪೂರ್ಣಾಙ್ಕ last_ಗಂಟು = first_unused + (ಪೂರ್ಣಾಙ್ಕ) pattern.size() - 1;
            // Some TOPK_MOE variants fuse a trailing scale.
            ಯದಿ (last_ಗಂಟು + 1 < graph->n_nodes && graph->nodes[last_ಗಂಟು + 1]->op == GGML_OP_SCALE) {
                last_ಗಂಟು++;
            }

            add_pattern_alloc_deps(pattern, last_ಗಂಟು);

            ಪ್ರತ್ಯಾಗಮ keep_pattern(pattern);
        };

        ಯದಿ (keep_topk_moe_pattern(topk_moe_early_softmax_norm)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (keep_topk_moe_pattern(topk_moe_sigmoid_norm_bias)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (keep_topk_moe_pattern(topk_moe_sqrt_softplus_norm_bias)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (keep_topk_moe_pattern(topk_moe_early_softmax)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (keep_topk_moe_pattern(topk_moe_late_softmax)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (keep_pattern(snake_pattern)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (keep_pattern(topk_qsa_pattern)) {
            ಅನುವರ್ತ;
        }

        ಯದಿ (keep_pattern(rms_norm_mul_add_mul_pattern)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (keep_pattern(rms_norm_mul_add_pattern)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (keep_pattern(rms_norm_mul_rope_view_set_rows_pattern)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (keep_pattern(rms_norm_view_set_rows_pattern)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (keep_pattern(rope_view_set_rows_pattern)) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (match_pattern(hc_post_gate_pattern, first_unused)) {
            add_pattern_alloc_deps(hc_post_gate_pattern, first_unused + (ಪೂರ್ಣಾಙ್ಕ) hc_post_gate_pattern.size() - 1);
            keep_pattern(hc_post_gate_pattern);
            ಅನುವರ್ತ;
        }

        // First, grab the next unused node.
        current_set.push_back(first_unused);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_TO_CHECK = 20;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = first_unused+1; j < std::min(first_unused + ಅಂಕೆ_TO_CHECK, graph->n_nodes); ++j) {
            ಯದಿ (used[j]) {
                ಅನುವರ್ತ;
            }
            ಯದಿ (is_empty(graph->nodes[j])) {
                ಅನುವರ್ತ;
            }
            // Protect every interior QSA node (not just the start): the mask branch is
            // independent, so it gets pulled out and breaks keep_pattern otherwise.
            auto ಧ್ರುವ &in_qsa_pattern = [&](ಪೂರ್ಣಾಙ್ಕ n) -> ತಾರ್ಕಿಕ {
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ o = 0; o < (ಪೂರ್ಣಾಙ್ಕ) topk_qsa_pattern.size(); ++o) {
                    ಯದಿ (n - o >= 0 && match_pattern(topk_qsa_pattern, n - o)) {
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    }
                }
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            };
            ಯದಿ (match_pattern(topk_moe_early_softmax_norm, j) ||
                match_pattern(topk_moe_sigmoid_norm_bias, j) ||
                match_pattern(topk_moe_sqrt_softplus_norm_bias, j) ||
                match_pattern(topk_moe_early_softmax, j) ||
                match_pattern(topk_moe_late_softmax, j) ||
                match_pattern(snake_pattern, j) ||
                in_qsa_pattern(j) ||
                match_pattern(rms_norm_mul_add_mul_pattern, j) ||
                match_pattern(rms_norm_mul_add_pattern, j) ||
                match_pattern(rms_norm_mul_rope_view_set_rows_pattern, j) ||
                match_pattern(rms_norm_view_set_rows_pattern, j) ||
                match_pattern(rope_view_set_rows_pattern, j) ||
                match_pattern(hc_post_gate_pattern, j)) {
                ಅನುವರ್ತ;
            }
            ತಾರ್ಕಿಕ ok = ಸತ್ಯ;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = first_unused; c < j; ++c) {
                ಯದಿ (!used[c] &&
                    is_src_of(graph->nodes[j], graph->nodes[c]) &&
                    !(c == current_set.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL && empty_or_scheduled_between(c+1, j)) &&
                    !(c == current_set.back() && graph->nodes[c]->op == GGML_OP_UNARY && graph->nodes[j]->op == GGML_OP_MUL && empty_or_scheduled_between(c+1, j)) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_ADD_ID) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_ADD && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_UNARY)) {
                    ok = ಅಸತ್ಯ;
                    ಭಙ್ಗ;
                }
            }
            ಯದಿ (ok) {
                current_set.push_back(j);

                ಪೂರ್ಣಾಙ್ಕ rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                ಯದಿ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ಯದಿ (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == nullptr || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_set.push_back(rope_idx);
                            used[rope_idx] = ಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                    }
                }
                // Look for ROPE/RMS_NORM + VIEW + SET_ROWS and make them consecutive
                ಯದಿ (graph->nodes[rope_idx]->op == GGML_OP_ROPE || graph->nodes[rope_idx]->op == GGML_OP_RMS_NORM) {
                    ಪೂರ್ಣಾಙ್ಕ view_idx = -1;
                    ಪೂರ್ಣಾಙ್ಕ set_rows_idx = -1;
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = rope_idx + 1; k < std::min(rope_idx + 15, graph->n_nodes); ++k) {
                        ಯದಿ (used[k]) {
                            ಅನುವರ್ತ;
                        }
                        ಯದಿ (view_idx == -1 && graph->nodes[k]->op == GGML_OP_VIEW && graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            ಅನುವರ್ತ;
                        }
                        ಯದಿ (view_idx != -1 && graph->nodes[k]->op == GGML_OP_SET_ROWS && graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            set_rows_idx = k;
                            ಭಙ್ಗ;
                        }
                    }
                    ಯದಿ (set_rows_idx != -1) {
                        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idxs[] = { rope_idx, view_idx, set_rows_idx };
                        ಧ್ರುವ ggml_op ops[] = { graph->nodes[rope_idx]->op, GGML_OP_VIEW, GGML_OP_SET_ROWS };
                        ತಾರ್ಕಿಕ can_pull = ggml_can_fuse_subgraph_ext(graph, ಗಂಟು_idxs, 3, ops, &set_rows_idx, 1);

                        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = rope_idx + 1; can_pull && c < set_rows_idx; ++c) {
                            ಯದಿ (!used[c] && c != view_idx && !is_empty(graph->nodes[c]) &&
                                is_src_of(graph->nodes[set_rows_idx], graph->nodes[c])) {
                                can_pull = ಅಸತ್ಯ;
                            }
                        }

                        ಯದಿ (can_pull) {
                            current_set.push_back(view_idx);
                            current_set.push_back(set_rows_idx);
                            used[view_idx] = ಸತ್ಯ;
                            used[set_rows_idx] = ಸತ್ಯ;
                        }
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                ಯದಿ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ಯದಿ (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || used_ಗಂಟು_set.find(graph->nodes[k]->src[1]) != used_ಗಂಟು_set.end())) {
                            current_set.push_back(k);
                            used[k] = ಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                ಯದಿ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ಯದಿ (graph->nodes[k]->op == GGML_OP_ADD &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || used_ಗಂಟು_set.find(graph->nodes[k]->src[1]) != used_ಗಂಟು_set.end())) {
                            current_set.push_back(k);
                            used[k] = ಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                    }
                }
                // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
                ಯದಿ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD &&
                    graph->nodes[j-1]->op == GGML_OP_SSM_CONV) {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ಯದಿ (graph->nodes[k]->op == GGML_OP_UNARY &&
                            graph->nodes[k]->src[0] == graph->nodes[j]) {
                            current_set.push_back(k);
                            used[k] = ಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                    }
                }
                // UNARY + MUL: pull the consuming MUL forward
                ಯದಿ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_UNARY) {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ggml_tensor * mul = graph->nodes[k];
                        ಯದಿ (mul->op != GGML_OP_MUL || (mul->src[0] != graph->nodes[j] && mul->src[1] != graph->nodes[j])) {
                            ಅನುವರ್ತ;
                        }
                        ggml_tensor * other = (mul->src[0] == graph->nodes[j]) ? mul->src[1] : mul->src[0];
                        // the other src must either be weights or already processed
                        ಯದಿ (!(other->op == GGML_OP_NONE || used_ಗಂಟು_set.find(other) != used_ಗಂಟು_set.end())) {
                            ಅನುವರ್ತ;
                        }
                        ಯದಿ (!ggml_vk_can_fuse_unary_mul(graph, j, k)) {
                            ಅನುವರ್ತ;
                        }
                        current_set.push_back(k);
                        used[k] = ಸತ್ಯ;
                        ಭಙ್ಗ;
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        ಯದಿ (graph->nodes[current_set.back()]->op != GGML_OP_ADD) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = first_unused+1; j < std::min(first_unused + ಅಂಕೆ_TO_CHECK, graph->n_nodes); ++j) {
                ಯದಿ (used[j]) {
                    ಅನುವರ್ತ;
                }
                ಯದಿ (!is_empty(graph->nodes[j])) {
                    ಅನುವರ್ತ;
                }
                ತಾರ್ಕಿಕ ok = ಸತ್ಯ;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = first_unused; c < j; ++c) {
                    ತಾರ್ಕಿಕ c_in_current_set = std::find(current_set.begin(), current_set.end(), c) != current_set.end();
                    // skip views whose srcs haven't been processed.
                    ಯದಿ (!used[c] &&
                        is_src_of(graph->nodes[j], graph->nodes[c]) &&
                        !c_in_current_set) {
                        ok = ಅಸತ್ಯ;
                        ಭಙ್ಗ;
                    }
                }
                ಯದಿ (ok) {
                    current_set.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        ಪ್ರತಿ (auto c : current_set) {
            new_order.push_back(graph->nodes[c]);
            used_ಗಂಟು_set.insert(graph->nodes[c]);
            used[c] = ಸತ್ಯ;
        }
        ಯಾವತ್ (first_unused < graph->n_nodes && used[first_unused]) {
            first_unused++;
        }
    }
    // Replace the graph with the new order.
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_order[i];
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    auto* cmd_buf = compute_ctx->s->buffer; // retrieve pointer before it gets reset

    ಯದಿ (vkev->has_event) {
        // Move existing event into submitted
        vkev->events_submitted.push_back(vkev->event);
    }

    // Grab the next event and record it, create one if necessary
    ಯದಿ (vkev->events_free.empty()) {
        vkev->event = ctx->device->device.createEvent({});
    } ಅನ್ಯಥಾ {
        vkev->event = vkev->events_free.back();
        vkev->events_free.pop_back();
    }

    vkev->has_event = ಸತ್ಯ;

    ggml_vk_set_event(compute_ctx, vkev->event);

    vkev->tl_semaphore.value++;
    compute_ctx->s->signal_semaphores.push_back(vkev->tl_semaphore);
    ggml_vk_ctx_end(compute_ctx);

    ggml_vk_submit(compute_ctx, {});
    ctx->submit_pending = ಸತ್ಯ;
    vkev->cmd_buffer = cmd_buf;
    vkev->cmd_buffer_use_counter = cmd_buf->use_counter;
    ctx->compute_ctx.reset();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    ಯದಿ (vkev->has_event) {
        // Wait for latest event
        ggml_vk_wait_events(compute_ctx, { vkev->event });

        ಯದಿ (ctx->device->async_use_transfer_queue) {
            vk_context transfer_ctx = ggml_vk_get_transfer_ctx(ctx);
            transfer_ctx->s->wait_semaphores.push_back(vkev->tl_semaphore);
        }
    }
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ ರಿಕ್ತ,
    /* .graph_plan_free         = */ ರಿಕ್ತ,
    /* .graph_plan_update       = */ ರಿಕ್ತ,
    /* .graph_plan_compute      = */ ರಿಕ್ತ,
    /* .graph_compute           = */ ggml_backend_vk_graph_compute,
    /* .event_record            = */ ggml_backend_vk_event_record,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_graph_optimize,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_vk_guid() {
    ಸ್ಥಿರ ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    ಪ್ರತ್ಯಾಗಮ &guid;
}

ggml_backend_t ggml_backend_vk_init(size_t dev_ಅಂಕೆ) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_ಅಂಕೆ << ")");

    ggml_backend_vk_context * ctx = ನೂತನ ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_ಅಂಕೆ);

    ggml_backend_t vk_backend = ನೂತನ ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_ಅಂಕೆ),
        /* .context = */ ctx,
    };

    ಯದಿ (!ctx->device->support_async) {
        vk_backend->iface.get_tensor_async = nullptr;
    }

    ಪ್ರತ್ಯಾಗಮ vk_backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_vk(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ backend != ರಿಕ್ತ && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

ಪೂರ್ಣಾಙ್ಕ ggml_backend_vk_get_device_ಎಣಿಕೆ() {
    ಪ್ರತ್ಯಾಗಮ ggml_vk_get_device_ಎಣಿಕೆ();
}

ಶೂನ್ಯ ggml_backend_vk_get_device_description(ಪೂರ್ಣಾಙ್ಕ device, ವರ್ಣ * description, size_t description_size) {
    GGML_ASSERT(device < (ಪೂರ್ಣಾಙ್ಕ) vk_instance.device_indices.size());
    ಪೂರ್ಣಾಙ್ಕ dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_size);
}

ಶೂನ್ಯ ggml_backend_vk_get_device_memory(ಪೂರ್ಣಾಙ್ಕ device, size_t * ಮುಕ್ತ, size_t * total) {
    GGML_ASSERT(device < (ಪೂರ್ಣಾಙ್ಕ) vk_instance.device_indices.size());
    GGML_ASSERT(device < (ಪೂರ್ಣಾಙ್ಕ) vk_instance.device_supports_membudget.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgetprops;
    vk::PhysicalDeviceMemoryProperties2 memprops = {};
    ಧ್ರುವ ತಾರ್ಕಿಕ membudget_supported = vk_instance.device_supports_membudget[device];
    ಧ್ರುವ ತಾರ್ಕಿಕ is_integrated_gpu = vkdev.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    ಯದಿ (membudget_supported) {
        memprops.pNext = &budgetprops;
    }
    vkdev.getMemoryProperties2(&memprops);

    *total = 0;
    *ಮುಕ್ತ = 0;

    ಪ್ರತಿ (uint32_t i = 0; i < memprops.memoryProperties.memory_Heap_ಎಣಿಕೆ; ++i) {
        ಧ್ರುವ vk::MemoryHeap & heap = memprops.memoryProperties.memoryHeaps[i];

        ಯದಿ (is_integrated_gpu || (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
            *total += heap.size;

            ಯದಿ (membudget_supported && i < budgetprops.heapUsage.size()) {
                // Driver may report heapBudget > heap.size so we add a safeguard here
                // See https://github.com/ggml-org/llama.cpp/issues/29277
                ಧ್ರುವ vk::DeviceSize budget = std::min(budgetprops.heapBudget[i], heap.size);
                *ಮುಕ್ತ += budget > budgetprops.heapUsage[i] ? budget - budgetprops.heapUsage[i] : 0;
            } ಅನ್ಯಥಾ {
                *ಮುಕ್ತ += heap.size;
            }
        }
    }
}

ಸ್ಥಿರ vk::PhysicalDeviceType ggml_backend_vk_get_device_type(ಪೂರ್ಣಾಙ್ಕ device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (ಪೂರ್ಣಾಙ್ಕ) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    vk::PhysicalDeviceProperties2 props = {};
    device.getProperties2(&props);

    ಪ್ರತ್ಯಾಗಮ props.properties.deviceType;
}

ಸ್ಥಿರ std::string ggml_backend_vk_get_device_pci_id(ಪೂರ್ಣಾಙ್ಕ device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (ಪೂರ್ಣಾಙ್ಕ) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    ಧ್ರುವ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    ತಾರ್ಕಿಕ ext_support = ಅಸತ್ಯ;

    ಪ್ರತಿ (ಧ್ರುವ auto& properties : ext_props) {
        ಯದಿ (strcmp("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
            ext_support = ಸತ್ಯ;
            ಭಙ್ಗ;
        }
    }

    ಯದಿ (!ext_support) {
        ಪ್ರತ್ಯಾಗಮ "";
    }

    vk::PhysicalDeviceProperties2 props = {};
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

    props.pNext = &pci_bus_info;

    device.getProperties2(&props);

    ಧ್ರುವ uint32_t pci_domain = pci_bus_info.pciDomain;
    ಧ್ರುವ uint32_t pci_bus = pci_bus_info.pciBus;
    ಧ್ರುವ uint32_t pci_device = pci_bus_info.pciDevice;
    ಧ್ರುವ uint8_t pci_function = (uint8_t) pci_bus_info.pciFunction; // pci function is between 0 and 7, prevent printf overflow warning

    ವರ್ಣ pci_bus_id[16] = {};
    snprintf(pci_bus_id, ಪರಿಮಾಣ(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    ಪ್ರತ್ಯಾಗಮ std::string(pci_bus_id);
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಪ್ರತ್ಯಾಗಮ ctx->name.c_str();
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಪ್ರತ್ಯಾಗಮ ctx->description.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_device_get_memory(ggml_backend_dev_t device, size_t * ಮುಕ್ತ, size_t * total) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_memory(ctx->device, ಮುಕ್ತ, total);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_vk_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಪ್ರತ್ಯಾಗಮ ggml_backend_vk_buffer_type(ctx->device);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_vk_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    ಪ್ರತ್ಯಾಗಮ ggml_backend_vk_host_buffer_type();
}

ಸ್ಥಿರ ಗಣನಾ ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    ಪ್ರತ್ಯಾಗಮ ctx->is_integrated_gpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, ಸಂರಚನಾ ggml_backend_dev_props * props) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_vk_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ಸತ್ಯ,
        /* .host_buffer           = */ ಸತ್ಯ,
        /* .buffer_from_host_ptr  = */ ಅಸತ್ಯ,
        /* .events                = */ ಸತ್ಯ,
        /* .mmap_support          = */ !ctx->is_integrated_gpu,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, ಧ್ರುವ ವರ್ಣ * params) {
    UNUSED(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಪ್ರತ್ಯಾಗಮ ggml_backend_vk_init(ctx->device);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, ಧ್ರುವ ggml_tensor * op) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಧ್ರುವ vk_device& device = ggml_vk_get_device(ctx->device);

    ಧ್ರುವ ತಾರ್ಕಿಕ uses_bda = (op->op == GGML_OP_IM2COL || op->op == GGML_OP_IM2COL_3D) &&
                          device->shader_int64 && device->buffer_device_address;

    auto ಧ್ರುವ & tensor_size_supported = [&](size_t tensor_size) {
        ಯದಿ (tensor_size > device->max_buffer_size) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // For im2col shaders using BDA, maxStorageBufferRange limit doesn't apply.
        // If shader64BitIndexing is enabled, maxStorageBufferRange limit doesn't apply.
        ಯದಿ (!uses_bda && !device->shader_64b_indexing) {
            ಯದಿ (tensor_size > device->properties.limits.maxStorageBufferRange) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };
    // reject any tensors larger than the max buffer size
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಯದಿ (op->src[i] && !tensor_size_supported(ggml_nbytes(op->src[i]))) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }
    ಯದಿ (!tensor_size_supported(ggml_nbytes(op))) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ವಿಕಲ್ಪ (op->op) {
        ಪಕ್ಷ GGML_OP_UNARY:
            ವಿಕಲ್ಪ (ggml_get_unary_op(op)) {
                ಪಕ್ಷ GGML_UNARY_OP_EXP:
                ಪಕ್ಷ GGML_UNARY_OP_EXPM1:
                ಪಕ್ಷ GGML_UNARY_OP_ELU:
                ಪಕ್ಷ GGML_UNARY_OP_GELU:
                ಪಕ್ಷ GGML_UNARY_OP_GELU_ERF:
                ಪಕ್ಷ GGML_UNARY_OP_GELU_QUICK:
                ಪಕ್ಷ GGML_UNARY_OP_SILU:
                ಪಕ್ಷ GGML_UNARY_OP_RELU:
                ಪಕ್ಷ GGML_UNARY_OP_XIELU:
                ಪಕ್ಷ GGML_UNARY_OP_NEG:
                ಪಕ್ಷ GGML_UNARY_OP_TANH:
                ಪಕ್ಷ GGML_UNARY_OP_SIGMOID:
                ಪಕ್ಷ GGML_UNARY_OP_HARDSIGMOID:
                ಪಕ್ಷ GGML_UNARY_OP_HARDSWISH:
                ಪಕ್ಷ GGML_UNARY_OP_ABS:
                ಪಕ್ಷ GGML_UNARY_OP_SOFTPLUS:
                ಪಕ್ಷ GGML_UNARY_OP_STEP:
                ಪಕ್ಷ GGML_UNARY_OP_ROUND:
                ಪಕ್ಷ GGML_UNARY_OP_CEIL:
                ಪಕ್ಷ GGML_UNARY_OP_FLOOR:
                ಪಕ್ಷ GGML_UNARY_OP_TRUNC:
                ಪಕ್ಷ GGML_UNARY_OP_SGN:
                    ಪ್ರತ್ಯಾಗಮ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                ಸ್ವಭಾವ:
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        ಪಕ್ಷ GGML_OP_GLU:
            ವಿಕಲ್ಪ (ggml_get_glu_op(op)) {
                ಪಕ್ಷ GGML_GLU_OP_GEGLU:
                ಪಕ್ಷ GGML_GLU_OP_REGLU:
                ಪಕ್ಷ GGML_GLU_OP_SWIGLU:
                ಪಕ್ಷ GGML_GLU_OP_SWIGLU_OAI:
                ಪಕ್ಷ GGML_GLU_OP_GEGLU_ERF:
                ಪಕ್ಷ GGML_GLU_OP_GEGLU_QUICK:
                ಪಕ್ಷ GGML_GLU_OP_SWIGLU_CLAMP:
                    ಪ್ರತ್ಯಾಗಮ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type) &&
                           (!op->src[1] || op->src[1]->type == op->src[0]->type);
                ಸ್ವಭಾವ:
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        ಪಕ್ಷ GGML_OP_MUL_MAT:
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                ಯದಿ (op->op == GGML_OP_MUL_MAT_ID) {
                    ಯದಿ (!ggml_vk_use_mul_mat_vec_id(op)) {
                        ಧ್ರುವ ggml_tensor * ids = op->src[2];
                        // Shaders store row IDs as uint16_t slot and token indices.
                        ಯದಿ (ids->ne[0] > int64_t(UINT16_MAX) + 1 || ids->ne[1] > int64_t(UINT16_MAX) + 1) {
                            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                        }
                    }
                    ಯದಿ (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                    ಯದಿ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                }
                ವಿಕಲ್ಪ (src0_type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                    ಪಕ್ಷ GGML_TYPE_F16:
                    ಪಕ್ಷ GGML_TYPE_BF16:
                    ಪಕ್ಷ GGML_TYPE_Q1_0:
                    ಪಕ್ಷ GGML_TYPE_Q2_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_1:
                    ಪಕ್ಷ GGML_TYPE_Q5_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_1:
                    ಪಕ್ಷ GGML_TYPE_Q8_0:
                    ಪಕ್ಷ GGML_TYPE_Q2_K:
                    ಪಕ್ಷ GGML_TYPE_Q3_K:
                    ಪಕ್ಷ GGML_TYPE_Q4_K:
                    ಪಕ್ಷ GGML_TYPE_Q5_K:
                    ಪಕ್ಷ GGML_TYPE_Q6_K:
                    ಪಕ್ಷ GGML_TYPE_IQ1_S:
                    ಪಕ್ಷ GGML_TYPE_IQ1_M:
                    ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
                    ಪಕ್ಷ GGML_TYPE_IQ2_XS:
                    ಪಕ್ಷ GGML_TYPE_IQ2_S:
                    ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
                    ಪಕ್ಷ GGML_TYPE_IQ3_S:
                    ಪಕ್ಷ GGML_TYPE_IQ4_XS:
                    ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                    ಪಕ್ಷ GGML_TYPE_MXFP4:
                    ಪಕ್ಷ GGML_TYPE_NVFP4:
                    ಪಕ್ಷ GGML_TYPE_TQ1_0:
                    ಪಕ್ಷ GGML_TYPE_TQ2_0:
                        ಭಙ್ಗ;
                    ಸ್ವಭಾವ:
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಸಂರಚನಾ ggml_tensor * a;
                ಸಂರಚನಾ ggml_tensor * b;
                ಯದಿ (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } ಅನ್ಯಥಾ {
                    a = op->src[2];
                    b = op->src[1];
                }
                ಯದಿ (a->ne[3] != b->ne[3]) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (op->src[1]->type == GGML_TYPE_BF16 && op->src[0]->type != GGML_TYPE_BF16 && op->op != GGML_OP_MUL_MAT) {
                    // BF16 in src1 (widened to F32/F16) is only wired up for MUL_MAT
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
            {
                ತಾರ್ಕಿಕ coopmat2 = device->coopmat2;
                uint32_t HSK = op->src[1]->ne[0];
                uint32_t HSV = op->src[2]->ne[0];
                ಯದಿ ((HSK % 8) != 0 || (HSV % 8) != 0) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (op->src[0]->type != GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (op->type != GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                auto fa_kv_ok = [](ggml_type t) {
                    ವಿಕಲ್ಪ (t) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                    ಪಕ್ಷ GGML_TYPE_F16:
                    ಪಕ್ಷ GGML_TYPE_BF16:
                    ಪಕ್ಷ GGML_TYPE_Q8_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_1:
                    ಪಕ್ಷ GGML_TYPE_Q5_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_1:
                    ಪಕ್ಷ GGML_TYPE_Q4_0:
                    ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                };
                ಯದಿ (!fa_kv_ok(op->src[1]->type) || !fa_kv_ok(op->src[2]->type)) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ ((op->src[1]->type == GGML_TYPE_BF16) != (op->src[2]->type == GGML_TYPE_BF16)) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
                    // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        ಪಕ್ಷ GGML_OP_GET_ROWS:
            {
                ವಿಕಲ್ಪ (op->src[0]->type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                    ಪಕ್ಷ GGML_TYPE_F16:
                    ಪಕ್ಷ GGML_TYPE_BF16:
                    ಪಕ್ಷ GGML_TYPE_Q1_0:
                    ಪಕ್ಷ GGML_TYPE_Q2_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_1:
                    ಪಕ್ಷ GGML_TYPE_Q5_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_1:
                    ಪಕ್ಷ GGML_TYPE_Q8_0:
                    ಪಕ್ಷ GGML_TYPE_Q2_K:
                    ಪಕ್ಷ GGML_TYPE_Q3_K:
                    ಪಕ್ಷ GGML_TYPE_Q4_K:
                    ಪಕ್ಷ GGML_TYPE_Q5_K:
                    ಪಕ್ಷ GGML_TYPE_Q6_K:
                    ಪಕ್ಷ GGML_TYPE_IQ1_S:
                    ಪಕ್ಷ GGML_TYPE_IQ1_M:
                    ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
                    ಪಕ್ಷ GGML_TYPE_IQ2_XS:
                    ಪಕ್ಷ GGML_TYPE_IQ2_S:
                    ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
                    ಪಕ್ಷ GGML_TYPE_IQ3_S:
                    ಪಕ್ಷ GGML_TYPE_IQ4_XS:
                    ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                    ಪಕ್ಷ GGML_TYPE_MXFP4:
                    ಪಕ್ಷ GGML_TYPE_NVFP4:
                    ಪಕ್ಷ GGML_TYPE_TQ1_0:
                    ಪಕ್ಷ GGML_TYPE_TQ2_0:
                    ಪಕ್ಷ GGML_TYPE_I32:
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
            }
        ಪಕ್ಷ GGML_OP_GET_ROWS_BACK:
            ಪ್ರತ್ಯಾಗಮ op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_SET_ROWS:
            {
                ಯದಿ ((op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) ||
                    (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64)) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ವಿಕಲ್ಪ (op->type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                    ಪಕ್ಷ GGML_TYPE_F16:
                    ಪಕ್ಷ GGML_TYPE_BF16:
                    ಪಕ್ಷ GGML_TYPE_Q1_0:
                    ಪಕ್ಷ GGML_TYPE_Q2_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_1:
                    ಪಕ್ಷ GGML_TYPE_Q5_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_1:
                    ಪಕ್ಷ GGML_TYPE_Q8_0:
                    ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
            }
        ಪಕ್ಷ GGML_OP_CONT:
        ಪಕ್ಷ GGML_OP_CPY:
        ಪಕ್ಷ GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != nullptr ? op->src[1]->type : src0_type;

                ಯದಿ (src0_type == GGML_TYPE_F32) {
                    ವಿಕಲ್ಪ (src1_type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                    ಪಕ್ಷ GGML_TYPE_F16:
                    ಪಕ್ಷ GGML_TYPE_BF16:
                    ಪಕ್ಷ GGML_TYPE_Q1_0:
                    ಪಕ್ಷ GGML_TYPE_Q2_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_1:
                    ಪಕ್ಷ GGML_TYPE_Q5_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_1:
                    ಪಕ್ಷ GGML_TYPE_Q8_0:
                    ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        ಭಙ್ಗ;
                    }
                }
                ಯದಿ (src1_type == GGML_TYPE_F32) {
                    ವಿಕಲ್ಪ (src0_type) {
                    ಪಕ್ಷ GGML_TYPE_F16:
                    ಪಕ್ಷ GGML_TYPE_BF16:
                    ಪಕ್ಷ GGML_TYPE_Q1_0:
                    ಪಕ್ಷ GGML_TYPE_Q2_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_0:
                    ಪಕ್ಷ GGML_TYPE_Q4_1:
                    ಪಕ್ಷ GGML_TYPE_Q5_0:
                    ಪಕ್ಷ GGML_TYPE_Q5_1:
                    ಪಕ್ಷ GGML_TYPE_Q8_0:
                    ಪಕ್ಷ GGML_TYPE_IQ4_NL:
                        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                    ಸ್ವಭಾವ:
                        ಭಙ್ಗ;
                    }
                }

                ಯದಿ (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }

                ಯದಿ (
                    (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) ||
                    (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32)
                ) {
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }

                // We can handle copying from a type to the same type if it's
                // either not quantized or is quantized and contiguous.
                // We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                ಯದಿ (src0_type == src1_type &&
                    (!ggml_is_quantized(src0_type) || (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op))) &&
                    (ggml_type_size(src0_type) % 2) == 0) {
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        ಪಕ್ಷ GGML_OP_REPEAT:
            ಪ್ರತ್ಯಾಗಮ ggml_type_size(op->type) == ggml_type_size(op->src[0]->type) &&
                  (ggml_type_size(op->type) == ಪರಿಮಾಣ(ಭಿನ್ನ) || ggml_type_size(op->type) == 2);
        ಪಕ್ಷ GGML_OP_REPEAT_BACK:
            ಪ್ರತ್ಯಾಗಮ op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_ROPE:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous_rows(op) && ggml_is_contiguous_rows(op->src[0]);
        ಪಕ್ಷ GGML_OP_ROPE_BACK:
        ಪಕ್ಷ GGML_OP_NONE:
        ಪಕ್ಷ GGML_OP_RESHAPE:
        ಪಕ್ಷ GGML_OP_VIEW:
        ಪಕ್ಷ GGML_OP_PERMUTE:
        ಪಕ್ಷ GGML_OP_TRANSPOSE:
        ಪಕ್ಷ GGML_OP_RMS_NORM:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪಕ್ಷ GGML_OP_GROUP_NORM:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]);
        ಪಕ್ಷ GGML_OP_NORM:
        ಪಕ್ಷ GGML_OP_L2_NORM:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_ADD:
        ಪಕ್ಷ GGML_OP_SUB:
        ಪಕ್ಷ GGML_OP_MUL:
        ಪಕ್ಷ GGML_OP_DIV:
            ಪ್ರತ್ಯಾಗಮ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        ಪಕ್ಷ GGML_OP_ADD_ID:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_SILU_BACK:
        ಪಕ್ಷ GGML_OP_RMS_NORM_BACK:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_SQR:
        ಪಕ್ಷ GGML_OP_SQRT:
        ಪಕ್ಷ GGML_OP_SIN:
        ಪಕ್ಷ GGML_OP_COS:
        ಪಕ್ಷ GGML_OP_CLAMP:
        ಪಕ್ಷ GGML_OP_LEAKY_RELU:
            ಪ್ರತ್ಯಾಗಮ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        ಪಕ್ಷ GGML_OP_OPT_STEP_ADAMW:
        ಪಕ್ಷ GGML_OP_OPT_STEP_SGD:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_OUT_PROD:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && op->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_LOG:
        ಪಕ್ಷ GGML_OP_TRI:
        ಪಕ್ಷ GGML_OP_DIAG:
            ಪ್ರತ್ಯಾಗಮ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        ಪಕ್ಷ GGML_OP_ARGSORT:
            {
                ಯದಿ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                // pipeline_argsort_large_f32 requires vulkan memory model.
                ಯದಿ (device->vulkan_memory_model) {
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                } ಅನ್ಯಥಾ {
                    ಪ್ರತ್ಯಾಗಮ op->ne[0] <= (1 << std::min(device->max_workgroup_size_log2, ಅಂಕೆ_argsort_pipelines - 1));
                }
            }
        ಪಕ್ಷ GGML_OP_ತುದಿ_K:
            {
                ಯದಿ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                // large k falls back to radix-select
                ಧ್ರುವ uint32_t min_pipeline =
                    std::max((uint32_t) log2f(ಭಿನ್ನ(op->ne[0])) + 1, device->subgroup_size_log2);
                ಯದಿ (min_pipeline < ಅಂಕೆ_topk_pipelines && device->pipeline_topk_f32[min_pipeline]) {
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
                ಪ್ರತ್ಯಾಗಮ device->pipeline_topk_radix_f32 != nullptr;
            }
        ಪಕ್ಷ GGML_OP_UPSCALE:
            ಯದಿ (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                ಯದಿ ((op->op_params[0] & 0xFF) != GGML_SCALE_MODE_BILINEAR) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
            }
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_ACC:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_SET:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == op->src[1]->type && op->src[0]->type == op->type &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_I32);
        ಪಕ್ಷ GGML_OP_CONCAT: {
            ಪ್ರತ್ಯಾಗಮ ggml_vk_concat_supported(op->src[0], op->src[1], op);
        }
        ಪಕ್ಷ GGML_OP_ADD1:
            ಪ್ರತ್ಯಾಗಮ (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        ಪಕ್ಷ GGML_OP_ARANGE:
            ಪ್ರತ್ಯಾಗಮ op->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_FILL:
            ಪ್ರತ್ಯಾಗಮ op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        ಪಕ್ಷ GGML_OP_SCALE:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_PAD:
        ಪಕ್ಷ GGML_OP_PAD_REFLECT_1D:
        ಪಕ್ಷ GGML_OP_ROLL:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_DIAG_MASK_INF:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_SOFT_MAX:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && (!op->src[1] || (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16));
        ಪಕ್ಷ GGML_OP_SOFT_MAX_BACK:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_SUM:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguously_allocated(op->src[0]);
        ಪಕ್ಷ GGML_OP_SUM_ROWS:
        ಪಕ್ಷ GGML_OP_MEAN:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        ಪಕ್ಷ GGML_OP_CUMSUM:
            {
                ಯದಿ (device->subgroup_arithmetic && device->subgroup_require_full_support) {
                    ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
                }
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        ಪಕ್ಷ GGML_OP_DSV4_HC_COMB:
        ಪಕ್ಷ GGML_OP_DSV4_HC_PRE:
        ಪಕ್ಷ GGML_OP_DSV4_HC_POST:
            {
                ಯದಿ (op->type != GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಪ್ರತಿ (uint32_t i = 0; i < GGML_MAX_SRC; ++i) {
                    ಯದಿ (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                }
                // hc is hardcoded to 4 in the shaders. ggml only constrains it
                // to 4 for COMB, so PRE/POST have to be checked here.
                ಯದಿ (op->op == GGML_OP_DSV4_HC_PRE && op->src[0]->ne[1] != 4) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (op->op == GGML_OP_DSV4_HC_POST && op->src[1]->ne[1] != 4) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (op->op == GGML_OP_DSV4_HC_COMB) {
                    ಪ್ರತ್ಯಾಗಮ device->pipeline_dsv4_hc_comb_f32 != nullptr;
                }
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        ಪಕ್ಷ GGML_OP_SOLVE_TRI:
            {
                ಯದಿ (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಧ್ರುವ uint32_t N = op->src[0]->ne[0];
                ಧ್ರುವ uint32_t K = op->src[1]->ne[0];
                // K dimension limited to workgroup size
                ಯದಿ (K > 1u << device->max_workgroup_size_log2) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಧ್ರುವ uint32_t batch_N = device->properties.limits.maxComputeSharedMemorySize / ((N + K) * ಪರಿಮಾಣ(ಭಿನ್ನ));

                ಯದಿ (batch_N == 0) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        ಪಕ್ಷ GGML_OP_ARGMAX:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_CROSS_ENTROPY_LOSS:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[0], op->src[1])
                && ggml_is_contiguous(op) && ggml_is_scalar(op) && op->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32 && ggml_is_scalar(op->src[0])
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[2]) && op->src[2]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[1], op->src[2])
                && ggml_are_same_shape(op->src[1], op)
                && ggml_is_contiguous(op) && op->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_ಎಣಿಕೆ_EQUAL:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_I32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_I32;
        ಪಕ್ಷ GGML_OP_IM2COL:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[1])
                && op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        ಪಕ್ಷ GGML_OP_IM2COL_3D:
            ಪ್ರತ್ಯಾಗಮ op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        ಪಕ್ಷ GGML_OP_TIMESTEP_EMBEDDING:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_CONV_2D_DW:
            ಪ್ರತ್ಯಾಗಮ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)
                && op->src[1]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_POOL_1D:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_POOL_2D:
            ಪ್ರತ್ಯಾಗಮ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_RWKV_WKV6:
        ಪಕ್ಷ GGML_OP_RWKV_WKV7:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ; // all inputs are contiguous, see ggml.c
        ಪಕ್ಷ GGML_OP_GATED_LINEAR_ATTN:
            // the shader block size is hardcoded to head_size 64
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[0]->ne[0] == 64;
        ಪಕ್ಷ GGML_OP_LIGHTNING_INDEXER:
            {
                ಧ್ರುವ ggml_tensor * q = op->src[0];
                ಧ್ರುವ ggml_tensor * k = op->src[1];
                ಧ್ರುವ ggml_tensor * w = op->src[2];
                ಧ್ರುವ ggml_tensor * m = op->src[3];

                // the q/w/m types and the shape relationships between q, k, w, m and dst
                // are already asserted in ggml_lightning_indexer()
                ಯದಿ (!ggml_vk_lightning_indexer_k_type_supported(k->type) || !device->fp16) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                // the shader block size is hardcoded to head size 128
                ಯದಿ (q->ne[0] != 128) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                // the shader indexes the buffers by element stride, and is dispatched
                // without allow_misalign
                ಪ್ರತಿ (ಧ್ರುವ ggml_tensor * t : {q, k, w, m, op}) {
                    ಯದಿ (t->nb[0] != ggml_type_size(t->type) ||
                        (vk_tensor_offset(t) + t->view_offs) % device->properties.limits.minStorageBufferOffsetAlignment != 0) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                    // the strides get scaled down from bytes, so the division must be exact
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < GGML_MAX_DIMS; ++i) {
                        ಯದಿ (t->nb[i] % ggml_type_size(t->type) != 0) {
                            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                        }
                    }
                }
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        ಪಕ್ಷ GGML_OP_GATED_DELTA_NET:
            {
                ಧ್ರುವ uint32_t S_v = op->src[2]->ne[0];
                ಯದಿ (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 6; i++) {
                    ಯದಿ (op->src[i] == nullptr || op->src[i]->type != GGML_TYPE_F32) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                }
                ಪ್ರತ್ಯಾಗಮ op->type == GGML_TYPE_F32;
            }
        ಪಕ್ಷ GGML_OP_SSM_SCAN:
            {
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 6; i++) {
                    ಯದಿ (op->src[i] && ggml_is_quantized(op->src[i]->type)) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                }
                ಯದಿ (op->src[6] && op->src[6]->type != GGML_TYPE_I32) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                ಧ್ರುವ uint32_t d_state = op->src[0]->ne[0];
                ಧ್ರುವ uint32_t head_dim = op->src[0]->ne[1];

                ತಾರ್ಕಿಕ is_mamba2 = (op->src[3] && op->src[3]->nb[1] == ಪರಿಮಾಣ(ಭಿನ್ನ));
                ಯದಿ (!is_mamba2) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                ಯದಿ ((d_state != 128 && d_state != 256) || head_dim % 16 != 0) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                size_t shmem_size = d_state * ಪರಿಮಾಣ(ಭಿನ್ನ);

                ಯದಿ (shmem_size > device->properties.limits.maxComputeSharedMemorySize) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                ಯದಿ (!device->subgroup_basic) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        ಪಕ್ಷ GGML_OP_SSM_CONV:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_1D:
            ಪ್ರತ್ಯಾಗಮ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        ಪಕ್ಷ GGML_OP_COL2IM_1D:
            ಪ್ರತ್ಯಾಗಮ (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    op->src[0]->type == GGML_TYPE_BF16) &&
                   op->type == op->src[0]->type &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op);
        ಪಕ್ಷ GGML_OP_CONV_2D:
        ಪಕ್ಷ GGML_OP_CONV_TRANSPOSE_2D:
            {
                ಧ್ರುವ ತಾರ್ಕಿಕ transpose = op->op == GGML_OP_CONV_TRANSPOSE_2D;
                ಧ್ರುವ int64_t cout = !transpose ? op->src[0]->ne[3] : op->src[0]->ne[2];
                ಧ್ರುವ int64_t cin  = !transpose ? op->src[0]->ne[2] : op->src[0]->ne[3];

                // Channel-contiguous format is not supported yet.
                ಪ್ರತ್ಯಾಗಮ ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    (op->src[0]->nb[0] == ಪರಿಮಾಣ(ಭಿನ್ನ) || op->src[0]->nb[0] == ಪರಿಮಾಣ(ggml_fp16_t) ) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    cout == op->ne[2] &&
                    cin == op->src[1]->ne[2] &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        ಪಕ್ಷ GGML_OP_CONV_3D:
            ಪ್ರತ್ಯಾಗಮ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op);
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    UNUSED(dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ಯದಿ (buft->iface.get_name != ggml_backend_vk_buffer_type_name) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_buffer_type_context * buft_ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    ಪ್ರತ್ಯಾಗಮ buft_ctx->device->idx == ctx->device;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, ಧ್ರುವ ggml_tensor * op) {
    ggml_backend_vk_device_context * dev_ctx = (ggml_backend_vk_device_context *)dev->context;

    ಪ್ರತ್ಯಾಗಮ ggml_vk_get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

ಸ್ಥಿರ ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = ನೂತನ vk_event;
    ಯದಿ (!vkev) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    // No events initially, they get created on demand
    vkev->has_event = ಅಸತ್ಯ;

    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vkev->tl_semaphore = { device->device.createSemaphore(ci), 0 };

    ಪ್ರತ್ಯಾಗಮ ನೂತನ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroySemaphore(vkev->tl_semaphore.s);
    ಪ್ರತಿ (auto& event : vkev->events_free) {
        device->device.destroyEvent(event);
    }
    ಪ್ರತಿ (auto& event : vkev->events_submitted) {
        device->device.destroyEvent(event);
    }
    ಯದಿ (vkev->has_event) {
        device->device.destroyEvent(vkev->event);
    }
    delete vkev;
    delete event;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    // Only do something if the event has actually been used
    ಯದಿ (vkev->has_event) {
        vk::Semaphore sem = vkev->tl_semaphore.s;
        uint64_t val = vkev->tl_semaphore.value;
        vk::SemaphoreWaitInfo swi{vk::SemaphoreWaitFlags{}, sem, val};
        VK_CHECK(device->device.waitSemaphores(swi, UINT64_MAX), "event_synchronize", device);

        // Reset and move submitted events
        ಪ್ರತಿ (auto& event : vkev->events_submitted) {
            device->device.resetEvent(event);
        }
        vkev->events_free.insert(vkev->events_free.end(), vkev->events_submitted.begin(), vkev->events_submitted.end());
        vkev->events_submitted.clear();

        // Finished using current command buffer so we flag for reuse
        ಯದಿ (vkev->cmd_buffer) {
            // Only flag for reuse if it hasn't been reused already
            ಯದಿ (vkev->cmd_buffer_use_counter == vkev->cmd_buffer->use_counter) {
                vkev->cmd_buffer->in_use = ಅಸತ್ಯ;
                vkev->cmd_buffer->buf.reset();
            }
            vkev->cmd_buffer = nullptr;
        }
    }
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_vk_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_size) {
    VK_LOG_DEBUG("ggml_backend_vk_device_buffer_from_host_ptr(backend=" << dev << ", ptr=" << ptr << ", size=" << size << ")");
    GGML_UNUSED(max_tensor_size);

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);

    vk_buffer buf = ggml_vk_buffer_from_host_ptr(device, ptr, size);

    ಯದಿ (!buf) {
        ಪ್ರತ್ಯಾಗಮ {};
    }

    ggml_backend_vk_buffer_context * bufctx = ನೂತನ ggml_backend_vk_buffer_context(device, std::move(buf), device->name);

    ggml_backend_buffer_t ret = ggml_backend_buffer_init(ggml_backend_vk_device_get_buffer_type(dev), ggml_backend_vk_buffer_interface, bufctx, size);

    ಪ್ರತ್ಯಾಗಮ ret;
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_memory,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ggml_backend_vk_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ ggml_backend_vk_device_event_new,
    /* .event_free           = */ ggml_backend_vk_device_event_free,
    /* .event_synchronize    = */ ggml_backend_vk_device_event_synchronize,
};

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    ಪ್ರತ್ಯಾಗಮ GGML_VK_NAME;
}

ಸ್ಥಿರ size_t ggml_backend_vk_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    UNUSED(reg);
    ಪ್ರತ್ಯಾಗಮ ggml_backend_vk_get_device_ಎಣಿಕೆ();
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    ಸ್ಥಿರ std::vector<ggml_backend_dev_t> devices;

    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಅಸತ್ಯ;

    {
        ಸ್ಥಿರ std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        ಯದಿ (!initialized) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ggml_backend_vk_get_device_ಎಣಿಕೆ(); i++) {
                ggml_backend_vk_device_context * ctx = ನೂತನ ggml_backend_vk_device_context;
                ವರ್ಣ desc[256];
                ggml_backend_vk_get_device_description(i, desc, ಪರಿಮಾಣ(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                ctx->op_offload_min_batch_size = min_batch_size;
                devices.push_back(ನೂತನ ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = ಸತ್ಯ;
        }
    }

    GGML_ASSERT(device < devices.size());
    ಪ್ರತ್ಯಾಗಮ devices[device];
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ ರಿಕ್ತ,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    ಸ್ಥಿರ ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ nullptr,
    };
    ಯತ್ನ {
        ggml_vk_instance_init();
        ಪ್ರತ್ಯಾಗಮ &reg;
    } ಗೃಹಾಣ (ಧ್ರುವ vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        ಪ್ರತ್ಯಾಗಮ nullptr;
    } ಗೃಹಾಣ (ಧ್ರುವ std::exception &e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: " << e.what());
        ಪ್ರತ್ಯಾಗಮ nullptr;
    } ಗೃಹಾಣ (...) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: unknown exception during Vulkan init");
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
}

ತಾರ್ಕಿಕ ggml_vk_instance_layer_settings_available() {
#ifdef GGML_VULKAN_VALIDATE
    // Check if validation layer provides the extension
    ಧ್ರುವ std::string layer_name = "VK_LAYER_KHRONOS_validation";
    ಪ್ರತಿ (ಧ್ರುವ auto& layer : vk::enumerateInstanceLayerProperties()) {
        ಯದಿ (layer_name == layer.layerName.data()) {
            ಪ್ರತಿ (ಧ್ರುವ auto& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
                ಯದಿ (strcmp("VK_EXT_layer_settings", ext.extensionName.data()) == 0) {
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            }
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Validation layer or layer extension VK_EXT_layer_settings not found." << std::endl;
#endif
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_vk_instance_portability_enumeration_ext_available(ಧ್ರುವ std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef __APPLE__
    // Check for portability enumeration extension for MoltenVK support
    ಪ್ರತಿ (ಧ್ರುವ auto& properties : instance_extensions) {
        ಯದಿ (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
#endif
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    UNUSED(instance_extensions);
}

ತಾರ್ಕಿಕ ggml_vk_instance_debug_utils_ext_available(
    ಧ್ರುವ std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    ಪ್ರತಿ (ಧ್ರುವ auto & properties : instance_extensions) {
        ಯದಿ (strcmp("VK_EXT_debug_utils", properties.extensionName) == 0) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::endl;
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    UNUSED(instance_extensions);
}

ತಾರ್ಕಿಕ ggml_vk_device_is_supported(ಧ್ರುವ vk::PhysicalDevice & vkdev) {
    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.pNext = nullptr;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.pNext = &vk11_features;

    vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

    ಪ್ರತ್ಯಾಗಮ vk11_features.storageBuffer16BitAccess;
}

ತಾರ್ಕಿಕ ggml_vk_khr_cooperative_matrix_support(ಧ್ರುವ vk::PhysicalDeviceProperties& props, ಧ್ರುವ vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    ವಿಕಲ್ಪ (props.vendorID) {
    ಪಕ್ಷ VK_VENDOR_ID_INTEL:
        // Only allowing Xe2/Xe3 GPU and integrated Xe GPUs at the moment since older hardware (ex. Arc A770) has performance regressions.
        ಪ್ರತ್ಯಾಗಮ (arch == vk_device_architecture::INTEL_XE2) ||
            (arch == vk_device_architecture::INTEL_XE1 && props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu && driver_props.driverID == vk::DriverId::eIntelProprietaryWindows);
    ಪಕ್ಷ VK_VENDOR_ID_AMD:
        ಯದಿ (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::eAmdOpenSource) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            ಪ್ರತ್ಯಾಗಮ arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    ಪಕ್ಷ VK_VENDOR_ID_QUALCOMM:
        // Only allow Adreno GPUs with hardware matrix cores (Gen 6+).
        ಪ್ರತ್ಯಾಗಮ arch == vk_device_architecture::QUALCOMM_ADRENO;
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
}

uint32_t ggml_vk_intel_shader_core_ಎಣಿಕೆ(ಧ್ರುವ vk::PhysicalDevice& vkdev) {
    VkPhysicalDeviceProperties2 props = vkdev.getProperties2();

    ಯದಿ (props.properties.vendorID != VK_VENDOR_ID_INTEL) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ uint32_t device_id = props.properties.deviceID;

    ವಿಕಲ್ಪ (device_id) {
    ಪಕ್ಷ 0x56A6:  // A310
        ಪ್ರತ್ಯಾಗಮ 6;
    ಪಕ್ಷ 0x5693:  // A370M
    ಪಕ್ಷ 0x56A5:  // A380
    ಪಕ್ಷ 0x56B1:  // Pro A40/A50
        ಪ್ರತ್ಯಾಗಮ 8;
    ಪಕ್ಷ 0x5697:  // A530M
        ಪ್ರತ್ಯಾಗಮ 12;
    ಪಕ್ಷ 0x5692:  // A550M
    ಪಕ್ಷ 0x56B3:  // Pro A60
        ಪ್ರತ್ಯಾಗಮ 16;
    ಪಕ್ಷ 0x56A2:  // A580
        ಪ್ರತ್ಯಾಗಮ 24;
    ಪಕ್ಷ 0x5691:  // A730M
    ಪಕ್ಷ 0x56A1:  // A750
        ಪ್ರತ್ಯಾಗಮ 28;
    ಪಕ್ಷ 0x56A0:  // A770
    ಪಕ್ಷ 0x5690:  // A770M
        ಪ್ರತ್ಯಾಗಮ 32;
    ಪಕ್ಷ 0xE212:  // Pro B50
        ಪ್ರತ್ಯಾಗಮ 16;
    ಪಕ್ಷ 0xE20C:  // B570
        ಪ್ರತ್ಯಾಗಮ 18;
    ಪಕ್ಷ 0xE20B:  // B580
    ಪಕ್ಷ 0xE211:  // Pro B60
    ಪಕ್ಷ 0xE222:  // Pro B65
        ಪ್ರತ್ಯಾಗಮ 20;
    ಪಕ್ಷ 0xE223:  // Pro B70
        ಪ್ರತ್ಯಾಗಮ 32;
    ಪಕ್ಷ 0xB080:  // PTL Xe3 LPG 2x6 (12 subslices)
        ಪ್ರತ್ಯಾಗಮ 12;
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ 0;
    }
}

ತಾರ್ಕಿಕ ggml_vk_intel_windows_driver_in_range(uint32_t driver_version, uint32_t lower_major, uint32_t lower_minor, uint32_t upper_major, uint32_t upper_minor) {
#ಯದಿ defined(_WIN32)
    // Intel Windows encodes xxx.yyyy as [31:14].[13:0].
    ಧ್ರುವ uint32_t major = driver_version >> 14;
    ಧ್ರುವ uint32_t minor = driver_version & 0x3fff;

    ಧ್ರುವ ತಾರ್ಕಿಕ ge_lower = major > lower_major || (major == lower_major && minor >= lower_minor);
    ಧ್ರುವ ತಾರ್ಕಿಕ lt_upper = major < upper_major || (major == upper_major && minor < upper_minor);

    ಪ್ರತ್ಯಾಗಮ ge_lower && lt_upper;
#ಅನ್ಯಥಾ
    GGML_UNUSED(driver_version);
    GGML_UNUSED(lower_major);
    GGML_UNUSED(lower_minor);
    GGML_UNUSED(upper_major);
    GGML_UNUSED(upper_minor);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
#endif
}

GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)


// out-of-lined header method definitions

ಶೂನ್ಯ vk_queue_handle_synchronized::submit(vk::ArrayProxy<ಧ್ರುವ vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    ಯದಿ (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    std::lock_guard<std::mutex> guard(mutex);
    ಯತ್ನ {
        queue.submit(submits, fence);
    } ಗೃಹಾಣ (vk::DeviceLostError &) {
        ಯದಿ (auto dev = device.lock()) {
            ggml_vk_print_device_lost_info(dev);
        }
        ಕ್ಷಿಪ;
    }
}

ಶೂನ್ಯ vk_queue_handle_unsynchronized::submit(vk::ArrayProxy<ಧ್ರುವ vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    ಯದಿ (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    ಯತ್ನ {
        queue.submit(submits, fence);
    } ಗೃಹಾಣ (vk::DeviceLostError &) {
        ಯದಿ (auto dev = device.lock()) {
            ggml_vk_print_device_lost_info(dev);
        }
        ಕ್ಷಿಪ;
    }
}

vk_device_struct::~vk_device_struct() {
    // the build failed before the device was created, nothing to destroy
    ಯದಿ (device == VK_NULL_HANDLE) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    VK_LOG_DEBUG("destroy device " << name);

    device.destroyFence(fence);

    ggml_vk_destroy_buffer(sync_staging);

    ಯದಿ (compute_queue) compute_queue->cmd_pool.destroy(device);
    ಯದಿ (transfer_queue) transfer_queue->cmd_pool.destroy(device);

    // Explicitly clear to ensure queues drop their shared_ptrs to handles
    // before the Vulkan logical device instance is destroyed
    compute_queue.reset();
    transfer_queue.reset();

    ಪ್ರತಿ (auto& pipeline : all_pipelines) {
        ಯದಿ (pipeline.expired()) {
            ಅನುವರ್ತ;
        }

        vk_pipeline pl = pipeline.lock();
        ggml_vk_destroy_pipeline(device, pl);
    }
    all_pipelines.clear();

    device.destroyDescriptorSetLayout(dsl);

    device.destroy();
}

ಶೂನ್ಯ vk_perf_logger::print_timings(ತಾರ್ಕಿಕ force) {
    ಯದಿ (timings.empty()) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    print_ಎಣಿಕೆ++;
    ಯದಿ ((print_ಎಣಿಕೆ % vk_perf_logger_frequency) != 0 && !force) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    print_ಎಣಿಕೆ = 0;
    uint64_t total_all_op_times = 0;
    std::cerr << "----------------\nVulkan Timings:" << std::endl;
    ಪ್ರತಿ (ಧ್ರುವ auto & t : timings) {
        uint64_t total_op_times = 0;
        ಪ್ರತಿ (ಧ್ರುವ auto & time : t.second) {
            total_op_times += time;
        }
        std::cerr << t.first << ": " << t.second.size() << " x " << (total_op_times / t.second.size() / 1000.0)
                  << " us = " << (total_op_times / 1000.0) << " us";

        // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
        auto it = flops.find(t.first);
        ಯದಿ (it != flops.end() && (it->second).size() == t.second.size()) {
            uint64_t total_op_flops = 0;
            ಪ್ರತಿ (ಧ್ರುವ auto & elem : it->second) {
                total_op_flops += elem;
            }
            std::cerr << " ("
                      << (ದ್ವಿಗುಣ(total_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                             (ದ್ವಿಗುಣ(total_op_times) / (1000.0 * 1000.0 * 1000.0))
                      << " GFLOPS/s)";
        }

        total_all_op_times += total_op_times;

        std::cerr << std::endl;
    }

    ಯದಿ (timings.size() > 0) {
        std::cerr << "Total time: " << total_all_op_times / 1000.0 << " us." << std::endl;
    }

    timings.clear();
    flops.clear();
}

std::string vk_perf_logger::get_ಗಂಟು_fusion_name(ಧ್ರುವ ggml_tensor * ಗಂಟು, ಧ್ರುವ ವರ್ಣ *fusion_name, uint64_t *n_flops) {
    *n_flops = ggml_vk_get_ಗಂಟು_flops(ಗಂಟು);
    std::string fusion_str;
    ಯದಿ (fusion_name) {
        fusion_str = fusion_name + std::string(" ");
    }
    ಯದಿ (ಗಂಟು->op == GGML_OP_UNARY) {
        ಪ್ರತ್ಯಾಗಮ fusion_str + ggml_unary_op_name(ggml_get_unary_op(ಗಂಟು));
    }
    ಯದಿ (ಗಂಟು->op == GGML_OP_MUL_MAT || ಗಂಟು->op == GGML_OP_MUL_MAT_ID) {
        ಧ್ರುವ uint64_t m     = ಗಂಟು->ne[0];
        ಧ್ರುವ uint64_t n     = ಗಂಟು->ne[1];
        ಧ್ರುವ uint64_t k     = ಗಂಟು->src[1]->ne[0];
        ಧ್ರುವ uint64_t batch = ಗಂಟು->ne[2] * ಗಂಟು->ne[3];
        std::string    name  = ggml_op_name(ಗಂಟು->op);
        ಯದಿ ((ಗಂಟು->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_cols) ||
            (ಗಂಟು->op == GGML_OP_MUL_MAT_ID && ಗಂಟು->src[2]->ne[1] == 1)) {
            name += "_VEC";
        }
        name += " ";
        name += ggml_type_name(ಗಂಟು->src[0]->type);
        name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
        ಯದಿ (ಗಂಟು->op == GGML_OP_MUL_MAT_ID) {
            name += " n_expert=" + std::to_string(ಗಂಟು->src[0]->ne[2]);
        }
        ಯದಿ (batch > 1) {
            name += " batch=" + std::to_string(batch);
        }
        ಪ್ರತ್ಯಾಗಮ fusion_str + name;
    }
    ಯದಿ (ಗಂಟು->op == GGML_OP_CONV_2D || ಗಂಟು->op == GGML_OP_CONV_TRANSPOSE_2D) {
        std::string   name    = ggml_op_name(ಗಂಟು->op);
        ಧ್ರುವ ggml_tensor * knl = ಗಂಟು->src[0];
        uint64_t      Cout    = ಗಂಟು->ne[2];
        uint64_t      size_K  = ಗಂಟು->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        uint64_t      size_N  = ಗಂಟು->ne[3] * ಗಂಟು->ne[0] * ಗಂಟು->ne[1];
        name += " M=Cout=" + std::to_string(Cout) + ", K=Cin*KW*KH=" + std::to_string(size_K) +
                ", N=N*OW*OH=" + std::to_string(size_N);
        ಪ್ರತ್ಯಾಗಮ fusion_str + name;
    }
    ಯದಿ (ಗಂಟು->op == GGML_OP_RMS_NORM) {
        std::string   name    = ggml_op_name(ಗಂಟು->op);
        name += "(" + std::to_string(ಗಂಟು->ne[0]) + "," + std::to_string(ಗಂಟು->ne[1]) + "," + std::to_string(ಗಂಟು->ne[2]) + "," + std::to_string(ಗಂಟು->ne[3]) + ")";
        ಪ್ರತ್ಯಾಗಮ fusion_str + name;
    }
    ಯದಿ (ಗಂಟು->op == GGML_OP_FLASH_ATTN_EXT) {
        ಧ್ರುವ ggml_tensor * dst = ಗಂಟು;
        ಧ್ರುವ ggml_tensor * q = ಗಂಟು->src[0];
        ಧ್ರುವ ggml_tensor * k = ಗಂಟು->src[1];
        ಧ್ರುವ ggml_tensor * v = ಗಂಟು->src[2];
        ಧ್ರುವ ggml_tensor * m = ಗಂಟು->src[3];
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(ಗಂಟು->op) <<
            " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
            " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
            " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
            " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
            " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
        ಪ್ರತ್ಯಾಗಮ name.str();
    }
    ಯದಿ (ಗಂಟು->op == GGML_OP_ತುದಿ_K) {
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(ಗಂಟು->op) <<
            " K=" << ಗಂಟು->ne[0] <<
            " (" << ಗಂಟು->src[0]->ne[0] << "," << ಗಂಟು->src[0]->ne[1] << "," << ಗಂಟು->src[0]->ne[2] << "," << ಗಂಟು->src[0]->ne[3] << ")";
        ಪ್ರತ್ಯಾಗಮ name.str();
    }
    ಪ್ರತ್ಯಾಗಮ fusion_str + ggml_op_name(ಗಂಟು->op);
}

ggml_backend_vk_buffer_context::~ggml_backend_vk_buffer_context() {
    ggml_vk_destroy_buffer(dev_buffer);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ಧ್ರುವ std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2) {
    ಯದಿ (!vk_instance.debug_utils_support || ctx->s == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    begin(ctx, pipeline_name + " (" + std::to_string(wg0) + "," + std::to_string(wg1) + "," + std::to_string(wg2) + ")");
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ಧ್ರುವ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಪೂರ್ಣಾಙ್ಕ n_fused) {
    ಯದಿ (!vk_instance.debug_utils_support || ctx->s == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    std::string name = ggml_op_name(cgraph->nodes[ಗಂಟು_idx]->op);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i <= n_fused; i++) {
        name += "+";
        name += ggml_op_name(cgraph->nodes[ಗಂಟು_idx + i]->op);
    }
    name += " ";
    name += cgraph->nodes[ಗಂಟು_idx]->name;
    begin(ctx, name);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_queue_handle * handle, ಧ್ರುವ ವರ್ಣ * name) {
    ಯದಿ (!vk_instance.debug_utils_support || handle == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<ಭಿನ್ನ, 4>{1.0f, 1.0f, 1.0f, 1.0f};

    qhandle = handle;
    std::lock_guard<vk_queue_handle> guard(*qhandle);
    vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT(qhandle->queue, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

ಶೂನ್ಯ ggml_vk_debug_label::close() {
    ಯದಿ (subctx != nullptr) {
        // close on the current command buffer, which may differ from the one begin used
        ಯದಿ (subctx->s != nullptr) {
            vk_instance.pfn_vkCmdEndDebugUtilsLabelEXT(subctx->s->buffer->buf);
        }
        subctx->debug_labels.pop_back();
        subctx = nullptr;
    }
    ಯದಿ (qhandle != nullptr) {
        std::lock_guard<vk_queue_handle> guard(*qhandle);
        vk_instance.pfn_vkQueueEndDebugUtilsLabelEXT(qhandle->queue);
        qhandle = nullptr;
    }
}

ಶೂನ್ಯ ggml_vk_debug_label::begin(vk_context & ctx, ಧ್ರುವ std::string & name) {
    ಯದಿ (!vk_instance.debug_utils_support || ctx->s == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    subctx = ctx.get();
    subctx->debug_labels.push_back(name);
    ggml_vk_cmd_label_begin(subctx->s->buffer->buf, subctx->debug_labels.back().c_str());
}

